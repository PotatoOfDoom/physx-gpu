// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// npArtiSpatialInertia.slang
//
// P8.84: `computeSpatialInertiaW` (forwardDynamic2.cu:1114-1310) — the per-link
// body of `computeUnconstrainedSpatialInertiaLaunchPartial1T`, and the largest
// single block of real numerics in the articulation chain.
//
// Per link it builds the isolated spatial inertia from the link's inverse mass
// and inverse inertia, the Coriolis term from the parent's motion velocity, the
// gravity / external-acceleration / damping bias force, and writes nine block
// fields; then per articulation the centre of mass and total inverse mass.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// `Cm::transformInertiaTensor` is PX_CUDA_CALLABLE (CmUtils.h:57), and so are
// `storeSpatialMatrix`, `storePxMat33` and `storeSpatialVector`
// (PxgArticulation.h:130/282/…).  The fixture therefore builds the ENTIRE
// expected block image — maths and SoA packing both — by calling the engine on
// the host and comparing bit patterns.  That is the P8.69 / P8.81 arrangement
// and it is why this module is in PXG_NOCONTRACT_SHADERS: one fused
// multiply-add anywhere below puts the result off the engine's bit pattern.
//
// Every expression is spelled the way PhysX spells it, which is not always the
// obvious way:
//   * `PxVec3::normalize()` computes `1/m` ONCE and multiplies (PxVec3.h:260),
//     it does not divide three times;
//   * `PxMat33 * PxVec3` is `c0*v.x + c1*v.y + c2*v.z` (PxMat33.h:344);
//   * `PxMat33(PxQuat)` doubles the components first (PxMat33.h:136);
//   * the velocity caps compare `sqrt(cap)/sqrt(mag)`, not `sqrt(cap/mag)`.
//
// ===========================================================================
// THE CONVENTION THAT FLIPS, AND THE CUDA'S OWN NAMES BEING WRONG
// ===========================================================================
//
// `Cm::UnAlignedSpatialVector` carries (top, bottom) and what those mean
// depends on whether the vector is a MOTION or a FORCE:
//
//   mMotionVelocity   top = ANGULAR velocity,   bottom = LINEAR velocity
//   mCoriolis         top = angular accel,      bottom = linear accel
//   zExt / mZAVector  top = LINEAR force,       bottom = TORQUE
//
// That is the Featherstone dual-basis convention and it is correct.  What is
// *not* correct is the CUDA's local naming in the Coriolis block: the variables
// are literally called `force` and `torque` and are an angular and a linear
// acceleration respectively, stored top/bottom in that order.  A port that
// "fixes" the naming by swapping the slots produces a plausible-looking result
// that is wrong; the mutation sweep kills both directions.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;

// ---------------------------------------------------------------------------
// The engine's arithmetic, spelled its way
// ---------------------------------------------------------------------------

float3 asiCross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

// PxMat33::transform -- column0 * v.x + column1 * v.y + column2 * v.z
float3 asiMul(AbMat33 m, float3 v)
{
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z;
}

// PxMat33(PxQuat), PxMat33.h:136.  The doubling comes first and the products
// are formed from the doubled components -- `x2 * y`, not `2.0f * (x * y)`.
AbMat33 asiMat33FromQuat(float4 q)
{
    const float x2 = q.x + q.x;
    const float y2 = q.y + q.y;
    const float z2 = q.z + q.z;

    const float xx = x2 * q.x;
    const float yy = y2 * q.y;
    const float zz = z2 * q.z;

    const float xy = x2 * q.y;
    const float xz = x2 * q.z;
    const float xw = x2 * q.w;

    const float yz = y2 * q.z;
    const float yw = y2 * q.w;
    const float zw = z2 * q.w;

    AbMat33 m;
    m.c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    m.c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    m.c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
    return m;
}

// Cm::transformInertiaTensor (CmUtils.h:57): M * diag(invD) * M^T, written out
// with the six distinct entries computed once and mirrored.  `M(row, col)` is
// `column[col][row]` (PxMat33.h:330), so `M(1,0)` is column0's y -- the whole
// reason AbMat33 carries columns rather than a float3x3.
//
// Note the first parameter is named `invD` in the engine but this call site
// passes the NON-inverse inertia tensor.  The function does not care; the name
// is a leftover from its rigid-body use.
AbMat33 asiTransformInertiaTensor(float3 invD, AbMat33 M)
{
    const float axx = invD.x * M.c0.x, axy = invD.x * M.c0.y, axz = invD.x * M.c0.z;
    const float byx = invD.y * M.c1.x, byy = invD.y * M.c1.y, byz = invD.y * M.c1.z;
    const float czx = invD.z * M.c2.x, czy = invD.z * M.c2.y, czz = invD.z * M.c2.z;

    const float e00 = axx * M.c0.x + byx * M.c1.x + czx * M.c2.x;
    const float e11 = axy * M.c0.y + byy * M.c1.y + czy * M.c2.y;
    const float e22 = axz * M.c0.z + byz * M.c1.z + czz * M.c2.z;
    const float e01 = axx * M.c0.y + byx * M.c1.y + czx * M.c2.y;   // == e10
    const float e02 = axx * M.c0.z + byx * M.c1.z + czx * M.c2.z;   // == e20
    const float e12 = axy * M.c0.z + byy * M.c1.z + czy * M.c2.z;   // == e21

    // mIInv(row, col) = column[col][row]
    AbMat33 r;
    r.c0 = float3(e00, e01, e02);
    r.c1 = float3(e01, e11, e12);
    r.c2 = float3(e02, e12, e22);
    return r;
}

// ===========================================================================
// A FOURTH PLATFORM RULE: slangc FOLDS `x + 0.0` AWAY, AND THAT IS NOT IEEE
// ===========================================================================
//
// Measured, not assumed.  Compile
//
//     float3 acc = float3(0,0,0);  acc += v * m;   // m runtime
//
// and the SPIR-V contains the OpVectorTimesScalar and NO OpFAdd at all, at -O0.
// The fold is wrong for exactly one input: `(-0) + (+0)` is `+0` in IEEE-754 and
// `-0` after the fold.  The engine's C++ performs the add, so a port whose
// accumulator starts at a LITERAL zero diverges from it in the sign bit
// wherever the first term is a negative zero — which is not exotic here: a
// static link has `m == 0`, and every `something * m` is then a signed zero
// whose sign follows `something`.
//
// This cost the first run of testArtiSpatialInertia seven bytes, all of them
// `0x80` against `0x00`, all on the two zero-`invMass` links.
//
// The fix is to start those accumulators from a RUNTIME zero the compiler
// cannot fold.  `abs(dt) * 0.0f` survives as a real OpFMul (measured the same
// way) and is `+0` for any finite dt whatever its sign.  Every subsequent `+=`
// is then a genuine OpFAdd and matches the engine bit for bit.
//
// The earlier articulation modules are green today because their fixtures never
// produce a negative zero as an accumulator's first term — not because they are
// immune.  Any future accumulator in this port starts from asiRuntimeZero().
float asiRuntimeZero(float dt)
{
    return abs(dt) * 0.0f;
}

// PxVec3::normalize() returns the magnitude and scales by its RECIPROCAL, and
// only when the magnitude is strictly positive (PxVec3.h:295).  A bare divide
// gives NaN for a link at rest, which is the common case in a settled scene.
float asiNormalizeInPlace(inout float3 v)
{
    const float m = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f)
    {
        const float f = 1.0f / m;
        v = float3(v.x * f, v.y * f, v.z * f);
    }
    return m;
}

// ---------------------------------------------------------------------------
// The body
// ---------------------------------------------------------------------------
//
// `numLinks` comes from the RECORD here (the wrapper reads
// `articulation.data.numLinks`), not from the block header -- unlike
// computeUnconstrainedVelocities1TLaunch, which resolves it from whichever of
// the two the refill guard selected.  Transcribed, not unified.
void asiComputeSpatialInertiaW(uint64_t a, uint64_t artiBlock, uint64_t linkBlocks,
                               uint64_t dofBlocks, uint numLinks, float3 gravity, float dt,
                               bool externalForcesEveryTgsIteration, uint lane)
{
    // PxgArticulationLinkProp is 16 bytes and the device spelling is a single
    // float4 `invInertiaXYZ_invMass` over the host's PxVec3 + PxReal.
    const uint64_t props = artiPtrField(a, PXG_ARTI_LINK_PROPS);
    const uint64_t externalAccels = artiPtrField(a, PXG_ARTI_EXTERNAL_ACCELERATIONS);
    const uint64_t externalZIsolated = artiPtrField(a, PXG_ARTI_ZA_FORCES);

    const float invDt = 1.0f / dt;

    // the runtime zero every accumulator below starts from -- see the note above
    const float zf = asiRuntimeZero(dt);
    const float3 z3 = float3(zf, zf, zf);

    float3 com = z3;
    float totalMass = zf;

    // DECLARED ONCE, OUTSIDE THE LOOP, exactly as the engine does.  topLeft is
    // zeroed here and never written again; topRight's OFF-DIAGONAL entries are
    // never written at all, only its diagonal changes per link.  Hoisting the
    // zeroing into the loop would be equivalent; leaving either uninitialised
    // would not, and the fixture's non-zero poison is what makes that testable.
    const AbMat33 topLeft = abMat33Zero();
    AbMat33 topRight = abMat33Zero();

    for (uint linkID = 0u; linkID < numLinks; ++linkID)
    {
        // the engine preloads link i+1's props while working on link i, which is
        // a latency trick and nothing else -- a direct read is equivalent
        const float4 invInertiaXYZ_invMass =
            (Ptr<float4>(props))[linkID];

        const uint64_t blk = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;

        const float4 tq = abF4(blk, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
        const float4 tp = abF4(blk, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane);

        // THE INVERSES, each with its own zero guard.  A static link has
        // invMass == 0 and every component of invInertia == 0; without the
        // guards this is +inf and every downstream term is NaN.
        const float m = (invInertiaXYZ_invMass.w == 0.0f)
                            ? 0.0f : (1.0f / invInertiaXYZ_invMass.w);
        const float3 inertiaTensor = float3(
            (invInertiaXYZ_invMass.x == 0.0f) ? 0.0f : (1.0f / invInertiaXYZ_invMass.x),
            (invInertiaXYZ_invMass.y == 0.0f) ? 0.0f : (1.0f / invInertiaXYZ_invMass.y),
            (invInertiaXYZ_invMass.z == 0.0f) ? 0.0f : (1.0f / invInertiaXYZ_invMass.z));

        // topRight is the mass matrix: a diagonal of m, off-diagonals still zero
        topRight.c0.x = m;
        topRight.c1.y = m;
        topRight.c2.z = m;

        const AbMat33 bottomLeft = asiTransformInertiaTensor(inertiaTensor,
                                                             asiMat33FromQuat(tq));

        abStoreSpatialMatrix(blk, PXG_AB_LINK_mSpatialArticulatedInertia, lane,
                             topLeft, topRight, bottomLeft);
        abStorePxMat33(blk, PXG_AB_LINK_mIsolatedInertia, lane, bottomLeft);

        com += tp.xyz * m;
        totalMass += m;

        float3 velTop, velBottom;
        abLoadSpatialVector(blk, PXG_AB_LINK_mMotionVelocity, lane, velTop, velBottom);

        const float4 damping =
            abF4(blk, PXG_AB_LINK_mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW, lane);

        // ---- the Coriolis term.  `force` is an ANGULAR acceleration and
        // `torque` a LINEAR one; the names are the engine's and they are wrong.
        // `force` and `torque` are STORED as-is when the link is the root or has
        // no dofs, and only ever ASSIGNED otherwise -- no accumulation from a
        // literal zero, so they stay literal
        float3 force = float3(0.0f, 0.0f, 0.0f);
        float3 torque = float3(0.0f, 0.0f, 0.0f);

        if (linkID != 0u)
        {
            const uint parent = abU32(blk, PXG_AB_LINK_mParents, lane);
            const uint64_t pblk = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;

            float3 pVelTop, pVelBottom;
            abLoadSpatialVector(pblk, PXG_AB_LINK_mMotionVelocity, lane, pVelTop, pVelBottom);

            const float3 rw = float3(abF32(blk, PXG_AB_LINK_mRw_x, lane),
                                     abF32(blk, PXG_AB_LINK_mRw_y, lane),
                                     abF32(blk, PXG_AB_LINK_mRw_z, lane));

            torque = asiCross(pVelTop, asiCross(pVelTop, rw));

            // the dof cursor is an explicit mJointOffset here, NOT the running
            // cursor computeLinkVelocities uses -- both live in the same block
            // and after jcalc they agree, but they are different expressions
            const uint jointOffset = abU32(blk, PXG_AB_LINK_mJointOffset, lane);
            const uint nbDofs = abU8(blk, PXG_AB_LINK_mDofs, lane);

            if (nbDofs != 0u)
            {
                float3 dvTop = z3;
                float3 dvBottom = z3;
                for (uint ind = 0u; ind < nbDofs; ++ind)
                {
                    const uint64_t d =
                        dofBlocks + uint64_t(jointOffset + ind) * PXG_AB_DOF_SIZE;
                    const float jVel = abF32(d, PXG_AB_DOF_mJointVelocities, lane);
                    float3 wTop, wBottom;
                    abLoadSpatialVector(d, PXG_AB_DOF_mWorldMotionMatrix, lane, wTop, wBottom);
                    dvTop += wTop * jVel;
                    dvBottom += wBottom * jVel;
                }

                const float3 aVec = dvTop;
                force = asiCross(pVelTop, aVec);

                const float3 lVel = dvBottom;
                const float3 temp1 = 2.0f * asiCross(pVelTop, lVel);
                const float3 temp2 = asiCross(aVec, lVel);
                torque += temp1 + temp2;
            }
        }

        abStoreSpatialVector(blk, PXG_AB_LINK_mCoriolis, lane, force, torque);
        abStoreF4(blk, PXG_AB_LINK_mInvInertiaXYZ_invMassW, lane, invInertiaXYZ_invMass);

        // ---- the internal (gyroscopic) bias force ----
        float3 vA = velTop;
        float mag = asiNormalizeInPlace(vA);
        mag = min(mag, invDt);
        vA = vA * mag;

        float3 gravLinAccel = z3;
        if (abU8(blk, PXG_AB_LINK_mDisableGravity, lane) == 0u)
            gravLinAccel = -gravity;

        // FORCE convention: top is the LINEAR force, bottom the torque
        float3 zExtTop = gravLinAccel * m;
        float3 zExtBottom = z3;
        float3 zDampTop = z3;
        float3 zDampBottom = z3;
        const float3 zIntTop = z3;
        const float3 zIntBottom = asiCross(vA, asiMul(bottomLeft, vA));

        // externalAccelerations may be null; and the read-back CLEAR below is a
        // write into the RECORD, not the block -- the only one this kernel makes
        if (externalAccels != 0ull)
        {
            const uint64_t ea =
                externalAccels + uint64_t(linkID) * PXG_ARTI_SPATIAL_VECTOR_SIZE;
            Ptr<float> f = Ptr<float>(ea);
            const float3 eaTop = float3(f[0], f[1], f[2]);
            const float3 eaBottom = float3(f[3], f[4], f[5]);

            zExtTop += (-eaTop * m);
            zExtBottom += asiMul(bottomLeft, -eaBottom);

            if (abU8(blk, PXG_AB_LINK_mRetainsAcceleration, lane) == 0u)
            {
                f[0] = 0.0f; f[1] = 0.0f; f[2] = 0.0f;
                f[3] = 0.0f; f[4] = 0.0f; f[5] = 0.0f;
            }
        }

        // linear/angular damping.  ASSIGNMENT here, `+=` in the cap block below,
        // so the order of the two is load-bearing: if this guard is false zDamp
        // is still zero when the cap block adds to it.
        if (damping.x > 0.0f || damping.y > 0.0f)
        {
            const float linDamp = min(damping.x, invDt);
            const float angDamp = min(damping.y, invDt);

            zDampTop = (velBottom * linDamp * m) - zExtTop * linDamp * dt;
            zDampBottom = asiMul(bottomLeft, velTop * angDamp) - zExtBottom * angDamp * dt;
        }

        // velocity caps.  `sqrt(cap) / sqrt(mag)`, not `sqrt(cap / mag)`.
        const float angMag = velTop.x * velTop.x + velTop.y * velTop.y + velTop.z * velTop.z;
        const float linMag =
            velBottom.x * velBottom.x + velBottom.y * velBottom.y + velBottom.z * velBottom.z;
        if (angMag > damping.w || linMag > damping.z)
        {
            if (angMag > damping.w)
            {
                const float scale = 1.0f - sqrt(damping.w) / sqrt(angMag);
                const float3 tmpAng = asiMul(bottomLeft, velTop) * scale;
                zDampBottom += tmpAng * invDt;
            }

            if (linMag > damping.z)
            {
                const float scale = 1.0f - (sqrt(damping.z) / sqrt(linMag));
                const float3 tmpLin = velBottom * m * scale;
                zDampTop += tmpLin * invDt;
            }
        }

        float3 zStoreTop, zStoreBottom;
        if (externalForcesEveryTgsIteration)
        {
            // the ISOLATED external Z goes into the record's zAForces, because
            // externalAccelerations is only refreshed on demand and must not be
            // overwritten; only the damping half lands in mZAVector
            Ptr<float> z =
                Ptr<float>(externalZIsolated + uint64_t(linkID) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
            z[0] = zExtTop.x; z[1] = zExtTop.y; z[2] = zExtTop.z;
            z[3] = zExtBottom.x; z[4] = zExtBottom.y; z[5] = zExtBottom.z;
            zStoreTop = zDampTop;
            zStoreBottom = zDampBottom;
        }
        else
        {
            zStoreTop = zDampTop + zExtTop;
            zStoreBottom = zDampBottom + zExtBottom;
        }

        // stored in two places on purpose: the bias force is needed later to
        // compute the articulated transmission force
        abStoreSpatialVector(blk, PXG_AB_LINK_mZAVector, lane, zStoreTop, zStoreBottom);
        abStoreSpatialVector(blk, PXG_AB_LINK_mZAIntVector, lane, zIntTop, zIntBottom);
        abStoreSpatialVector(blk, PXG_AB_LINK_mBiasForce, lane,
                             zStoreTop + zIntTop, zStoreBottom + zIntBottom);
        abStoreF32(blk, PXG_AB_LINK_mMass, lane, m);
    }

    // NOT guarded in the engine: an articulation whose every link is static has
    // totalMass == 0 and divides by zero here.  Transcribed as-is.
    const float invMass = 1.0f / totalMass;
    com = com * invMass;

    abStoreF4(artiBlock, PXG_AB_mCOM_TotalInvMassW, lane,
              float4(com.x, com.y, com.z, invMass));
}

// ---------------------------------------------------------------------------
// The test harness entry point.  One articulation per lane, block bases handed
// in directly -- the engine index chain is the wrapper's job (P8.85).
// ---------------------------------------------------------------------------
// gravity is carried as three scalars rather than a float3: a float3 in a push
// constant is 16-byte aligned under the default layout rules and the fixture's
// C++ struct is not, which is a mismatch that compiles and then reads garbage.
struct AsiParams
{
    uint64_t articulations;   // @0   PxgArticulation[lane]
    uint64_t artiBlock;       // @8   one PxgArticulationBlockData
    uint64_t linkBlocks;      // @16  maxNumLinks link blocks
    uint64_t dofBlocks;       // @24  maxNumDofs dof blocks
    float gravityX;           // @32
    float gravityY;           // @36
    float gravityZ;           // @40
    float dt;                 // @44
    uint nbArticulations;     // @48
    uint externalForcesEveryTgsIteration;  // @52
    uint pad0;                // @56
    uint pad1;                // @60 -> 64 bytes
};

[[vk::push_constant]]
ConstantBuffer<AsiParams> gAsi;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiSpatialInertiaW(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gAsi.nbArticulations)
        return;

    const uint64_t a = gAsi.articulations + uint64_t(lane) * PXG_ARTI_STRIDE;

    // numLinks comes from the RECORD, exactly as the wrapper will read it
    const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);

    asiComputeSpatialInertiaW(a, gAsi.artiBlock, gAsi.linkBlocks, gAsi.dofBlocks, numLinks,
                              float3(gAsi.gravityX, gAsi.gravityY, gAsi.gravityZ), gAsi.dt,
                              gAsi.externalForcesEveryTgsIteration != 0u, lane);
}
