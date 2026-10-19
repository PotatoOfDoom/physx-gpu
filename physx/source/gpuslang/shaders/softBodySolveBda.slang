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

// softBodySolveBda.slang
//
// P8.0 tranche 4 — the first two kernels of the softbody FEM-volume SOLVE loop on the
// Vulkan engine ABI (BDA).  Ported line-for-line from
// gpusimulationcontroller/src/CUDA/softBodyGM.cu:
//
//   sb_gm_zeroTetMultipliers       (:1801, launch PxgSoftBodyCore.cpp:196)
//       clears every tetrahedron's Lagrange-multiplier spatial vector at the top of the
//       solve.  Without it the constraint solver accumulates into last frame's
//       multipliers and the material response drifts frame over frame.
//   sb_gm_applyExternalDeltasLaunch (:2222, launch PxgSoftBodyCore.cpp applyExternalTetraDelta)
//       folds the accumulated per-vertex correction (mSimDelta, written by the
//       contact/attachment solves; .w the mass-splitting refCount) back into velocity and,
//       in a position iteration, position and mSimDeltaPos, then zeroes the delta buffer
//       for the next iteration.  This is the softbody twin of
//       cloth_applyExternalDeltasLaunch, which the cloth arc has relied on since P7.1.
//
// They bracket the solve: zeroTetMultipliers opens it, applyExternalDeltas closes each
// iteration.  Both are pure load/scale/store — no transcendental, no warp op, no
// cross-lane communication at all.
//
// Doctrine (identical to the prior softbody tranches):
//   - the PxgSoftBody descriptor arrives as a device-pointer ARRAY read through Ptr<>
//     (the byte-exact pxgSlangSoftBodyDesc mirror); mSimVelocity_InvMass @88,
//     mSimPosition_InvMass @96, mSimDeltaPos @128, mSimTetraMultipliers @264,
//     mSimDelta @272, mNumVertsGM @396 and mNumTetsGM @400 (5.12) are the fields touched;
//   - PxgSpatialVectorBlock (PxgArticulation.h:77) is the 32-lane SoA block the
//     multipliers live in: float4 mTopxyz_bx[32] @0 (512 B) then float2 mbyz[32] @512
//     (256 B) -> 768-byte stride.  storeSpatialVector(block, v, lane) writes
//     mTopxyz_bx[lane] = (v.top.xyz, v.bottom.x) and mbyz[lane] = v.bottom.yz, so
//     zeroing a tet means two stores at block (tetIdx/32), lane (tetIdx&31).
//
// DEVIATIONS from CUDA:
//   - sb_gm_applyExternalDeltasLaunch opens with a `blockCopy` of the WHOLE 464-byte
//     PxgSoftBody descriptor into __shared__ followed by __syncthreads, then reads the
//     shared copy.  That is a pure read-caching optimisation for a block whose 64
//     threads would otherwise each pull the same descriptor words; it is dropped here
//     and every lane reads the descriptor directly through Ptr<> (the same fan-out
//     treatment the earlier softbody/cloth ports document).  Dropping it also removes
//     the kernel's only barrier, so there is no groupshared and no uniformity concern;
//   - CUDA's `delta.x *= scale; delta.y *= scale, delta.z *= scale;` (note the comma
//     operator on the second statement — a typo that happens to behave identically) is
//     written as a float3 multiply.
//
// Both kernels are NoContraction-stamped (PXG_NOCONTRACT_SHADERS): applyExternalDeltas
// computes `vel + delta * invDt`, a multiply feeding an add that would otherwise fuse
// into an FMA and break bitwise parity with the -ffp-contract=off CPU reference.  There
// is NO transcendental in either kernel — this tranche is FULLY bit-exact, no tolerance
// seam.

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimPosition_InvMass@96, ...)
import pxgSlangAtomicFloat;    // P9.95f: pxgAtomicAddF4XYZ / pxgAtomicAddF4W (the hex Jacobi tail)

// PxgSpatialVectorBlock (PxgArticulation.h:77): float4 mTopxyz_bx[32] @0, float2 mbyz[32]
// @512 -> 768-byte stride.
static const uint64_t SPATIAL_BLOCK_STRIDE = 768UL;
static const uint64_t SPATIAL_BLOCK_MBYZ_OFFSET = 512UL;

// ---------------------------------------------------------------------------
// sb_gm_zeroTetMultipliers (softBodyGM.cu:1801)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:198): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies } -> 16-byte blob.  (ABI row {8,8}.)
struct ZeroTetMultipliersParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8  -> 16
}
[[vk::push_constant]]
ConstantBuffer<ZeroTetMultipliersParams> gZero;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA; grid (nbBlocks, nbActiveSoftBodies, 1)
void sb_gm_zeroTetMultipliers(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gZero.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gZero.activeSoftbodies))[groupId.y];

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 64u;

    const uint numTets = gSoftbodies[softbodyId].mNumTetsGM;

    if (groupThreadIdx < numTets)
    {
        // storeSpatialVector(mSimTetraMultipliers[idx/32], UnAlignedSpatialVector::Zero(),
        //                    idx & 31)
        const uint64_t blockBase = gSoftbodies[softbodyId].mSimTetraMultipliers +
                                   uint64_t(groupThreadIdx >> 5) * SPATIAL_BLOCK_STRIDE;
        const uint lane = groupThreadIdx & 31u;

        Ptr<float4>(blockBase + uint64_t(lane) * 16UL)[0] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        Ptr<float2>(blockBase + SPATIAL_BLOCK_MBYZ_OFFSET + uint64_t(lane) * 8UL)[0] =
            float2(0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// sb_gm_applyExternalDeltasLaunch (softBodyGM.cu:2222)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp, applyExternalTetraDelta): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxReal invDt, bool isVelocityIteration } -> 21,
// rounded up to the pointer alignment -> 24-byte blob.  (ABI row {8,8,4,1}.)
// PhysX 5.12 (velocity-iteration semantics, 5.11): a velocity iteration writes only the
// velocity correction and skips the position and mSimDeltaPos updates.
struct ApplyExternalDeltasParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8
    float invDt;                // @16
    uint isVelocityIteration;   // @20  bool byte0 -> 24
}
[[vk::push_constant]]
ConstantBuffer<ApplyExternalDeltasParams> gApply;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_applyExternalDeltasLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gApply.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gApply.activeSoftbodies))[groupId.y];

    // The CUDA blockCopy of the whole descriptor into __shared__ + __syncthreads is a
    // read-caching optimisation only (see the header) — read it directly instead.
    Ptr<float4> curPositions = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
    Ptr<float4> vels = Ptr<float4>(gSoftbodies[softbodyId].mSimVelocity_InvMass);
    Ptr<float4> simDelta = Ptr<float4>(gSoftbodies[softbodyId].mSimDelta);
    Ptr<float4> simDeltaPos = Ptr<float4>(gSoftbodies[softbodyId].mSimDeltaPos);

    const uint vertIdx = groupThreadId.x + groupId.x * 64u;
    const uint nbVerts = gSoftbodies[softbodyId].mNumVertsGM;

    if (vertIdx < nbVerts)
    {
        const float4 pos = curPositions[vertIdx];
        const float4 vel = vels[vertIdx];

        float4 delta = simDelta[vertIdx];

        // delta.w is the mass-splitting reference count the pre-counts accumulated; the
        // finalize divides by it, never by less than 1 (PxMax(delta.w, 1.f), selectMax).
        const float scale = 1.0f / ((delta.w > 1.0f) ? delta.w : 1.0f);

        delta.x *= scale; delta.y *= scale; delta.z *= scale;
        delta.w = 0.0f;

        // Velocity iteration writes only the velocity correction, skipping the position and
        // accumulated-delta updates (Muller SCA 2020 S3.6).
        vels[vertIdx] = vel + delta * gApply.invDt;

        if ((gApply.isVelocityIteration & 0xffu) == 0u)
        {
            // P9.86: the CUDA's w lanes are IEEE `pos.w + 0.f` / `deltaPos.w + 0.f`, which
            // turn a -0 inverse mass into +0; llvmpipe folds the literal-zero w lane of these
            // vector adds, so the w lane's `0 + x` is a select.
            float4 newPos = pos + delta;
            newPos.w = (newPos.w == 0.0f) ? 0.0f : newPos.w;
            curPositions[vertIdx] = newPos;

            float4 newDeltaPos = simDeltaPos[vertIdx] + delta;
            newDeltaPos.w = (newDeltaPos.w == 0.0f) ? 0.0f : newDeltaPos.w;
            simDeltaPos[vertIdx] = newDeltaPos;
        }

        // clear delta
        simDelta[vertIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}


// ===========================================================================
// P8.0 tranche 5 — the tetrahedron ROTATION extractors (softBodyGM.cu).
//
//   sb_gm_updateTetrahedraRotationsLaunch (:419, launch PxgSoftBodyCore.cpp:743)
//       the SIMULATION mesh: one thread per sim tetrahedron, recomputes the
//       co-rotational frame from the current mSimPosition_InvMass.
//   sb_updateTetrahedraRotationsLaunch    (:461, launch PxgSoftBodyCore.cpp:786)
//       the COLLISION mesh twin, gated on eCOMPUTE_STRESS_TENSOR or
//       !eDISABLE_SELF_COLLISION (the collision-mesh rotations are only needed by the
//       stress readback and by self-collision).
//
// Both compute the deformation gradient F = P * Qinv from the tetrahedron's displacement
// field, seed the quaternion from the stored rotation (or mInitialRotation on the first
// frame, detected by the zero-initialised memory), and run Mueller's iterative polar
// decomposition extractRotation(F, q, 100) (deformableUtils.cuh:46).
//
// They differ in exactly three places, all from the .cu:
//   - buffers: mSimTetIndices @80 / mSimPosition_InvMass @96 / mSimTetraRestPoses @136 /
//     mSimTetraRotations @168 / mNumTetsGM @400   -- versus --   mTetIndices @16 /
//     mPosition_InvMass @64 / mTetraRestPoses @40 / mTetraRotations @56 / mNumTets @392;
//   - rest-pose LAYOUT: the sim variant loads through loadPxMat33 from a PxgMat33Block
//     (PxgArticulation.h:70 — float4 mCol0[32] @0, float4 mCol1[32] @512, float mCol2[32]
//     @1024 -> 1152-byte stride, the 9 elements split 4/4/1 across the three arrays),
//     while the collision variant reads a flat PxMat33 array (36-byte stride);
//   - the collision variant is wrapped in the stress/self-collision flag test; the sim
//     variant runs unconditionally.
//
// *** TOLERANCE SEAM (the first one in the softbody arc) ***
// extractRotation builds its per-iteration update as PxQuat(w, omega), and that
// constructor (PxQuat.h:86) evaluates PxSinCos -> sinf/cosf.  sin/cos have NO
// cross-backend rounding guarantee, so this kernel CANNOT be bit-exact — exactly the
// reason npSbExtractRotAPD.slang documents the Mueller variant as non-portable and
// implements the APD variant (which has no transcendental) instead.  The .cu chooses
// extractRotation here (the sb_extractRotationAPD call sits commented out one line
// above), so the seam is on the default path and cannot be avoided by configuration.
// Everything else in the chain IS bit-exact: PxVec3::normalize and
// PxQuat::getNormalized both go through PxSqrt, which is IEEE-754 correctly rounded,
// and the quaternion product / rotation-matrix build are pure arithmetic.  The unit test
// therefore compares the quaternion under a documented ULP budget rather than by raw
// bits, and pins everything up to the first sin/cos bit-exactly.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): F = P*Qinv,
// the cross/dot accumulations in the omega step and the quaternion product are all
// mul-feeding-add chains.
// ===========================================================================

// PxgMat33Block (PxgArticulation.h:70): float4 mCol0[32] @0, float4 mCol1[32] @512,
// float mCol2[32] @1024 -> 1152-byte stride.
static const uint64_t MAT33_BLOCK_STRIDE = 1152UL;
static const uint64_t MAT33_BLOCK_COL1_OFFSET = 512UL;
static const uint64_t MAT33_BLOCK_COL2_OFFSET = 1024UL;

// PxQuat operator* (PxQuat.h:388), left-to-right folds — identical to the proven
// npSbExtractRotAPD transcription.
float4 sbQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
                  a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
                  a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

// PxMat33(PxQuat) rotation ctor (PxMat33.h:136); columns out.
void sbMat33FromQuat(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
}

float sbDot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 sbCross3(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// PxMat33 P*Q: col_i = P.transform(Q.col_i) = (P.c0*v.x + P.c1*v.y) + P.c2*v.z.
float3 sbMat33Transform(float3 c0, float3 c1, float3 c2, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}

// extractRotation (deformableUtils.cuh:46) — Mueller's iterative polar decomposition.
// THE sin/cos SEAM LIVES HERE (the PxQuat(w, omega) constructor).
float4 sbExtractRotation(float3 a0, float3 a1, float3 a2, float4 q, int maxIter)
{
    const float eps = 1.0e-6f;
    for (int iter = 0; iter < maxIter; iter++)
    {
        float3 r0, r1, r2;
        sbMat33FromQuat(q, r0, r1, r2);

        float3 omega = sbCross3(r0, a0) + sbCross3(r1, a1) + sbCross3(r2, a2);

        omega *= 1.0f / (abs(sbDot3(r0, a0) + sbDot3(r1, a1) + sbDot3(r2, a2)) + eps);

        // PxVec3::normalize(): m = PxSqrt(magnitudeSquared()); if (m > 0) *this /= m;
        // returns m.  PxSqrt is IEEE-754 correctly rounded -> bit-exact.
        //
        // P9.36: `*this /= m` is operator/= (PxVec3.h:260), which forms the RECIPROCAL
        // ONCE and multiplies three times.  The comment above already said so and the
        // line below divided three times anyway.  extractRotation
        // (deformableUtils.cuh:58) reaches this through `omega.normalize()`.
        const float w = sqrt(sbDot3(omega, omega));
        if (w > 0.0f)
            omega = omega * (1.0f / w);

        // PxQuat(angleRadians, unitAxis) (PxQuat.h:86): a = angle*0.5; PxSinCos(a, s, w).
        // *** THE TOLERANCE SEAM *** — sin/cos are not bitwise portable.
        const float a = w * 0.5f;
        const float sa = sin(a);
        const float ca = cos(a);
        const float4 tempQ = float4(omega.x * sa, omega.y * sa, omega.z * sa, ca);

        q = sbQuatMul(tempQ, q);

        // PxQuat::getNormalized(): s = 1.0 / PxSqrt(magnitudeSquared()) -> bit-exact.
        const float qs = 1.0f / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        q = float4(q.x * qs, q.y * qs, q.z * qs, q.w * qs);

        // early-exit AFTER one update (the .cu comment: the expensive work is done)
        if (w < eps)
            break;
    }
    return q;
}

// Shared body of both kernels: F = P*Qinv from the displacement field, seed q, iterate.
float4 sbTetRotation(float3 u1, float3 u2, float3 u3,
                     float3 qinv0, float3 qinv1, float3 qinv2,
                     float4 storedRot, float4 initialRotation, int maxIter)
{
    // PxMat33 P(u1, u2, u3) — columns.  F = P * Qinv.
    const float3 f0 = sbMat33Transform(u1, u2, u3, qinv0);
    const float3 f1 = sbMat33Transform(u1, u2, u3, qinv1);
    const float3 f2 = sbMat33Transform(u1, u2, u3, qinv2);

    float4 q = storedRot;
    // initialize to identity quat if this is the first iteration (zero initialized memory)
    if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) == 0.0f)
        q = initialRotation;

    return sbExtractRotation(f0, f1, f2, q, maxIter);
}

// ---------------------------------------------------------------------------
// sb_gm_updateTetrahedraRotationsLaunch (softBodyGM.cu:419) — SIMULATION mesh.
// CUDA arg pack (PxgSoftBodyCore.cpp:747): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies } -> 16-byte blob.  (ABI row {8,8}.)
// ---------------------------------------------------------------------------
struct UpdateTetraRotationsParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8  -> 16
}
[[vk::push_constant]]
ConstantBuffer<UpdateTetraRotationsParams> gRot;

[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_updateTetrahedraRotationsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gRot.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gRot.activeSoftbodies))[groupId.y];

    const uint tetrahedronIdx = groupThreadId.x + groupId.x * 256u;
    const uint nbTets = gSoftbodies[softbodyId].mNumTetsGM;

    if (tetrahedronIdx < nbTets)
    {
        const float4 initialRotation = float4(gSoftbodies[softbodyId].mInitialRotationX,
                                              gSoftbodies[softbodyId].mInitialRotationY,
                                              gSoftbodies[softbodyId].mInitialRotationZ,
                                              gSoftbodies[softbodyId].mInitialRotationW);

        Ptr<uint4> tetIndices = Ptr<uint4>(gSoftbodies[softbodyId].mSimTetIndices);
        Ptr<float4> verts = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
        Ptr<float4> tetRotations = Ptr<float4>(gSoftbodies[softbodyId].mSimTetraRotations);

        const uint4 tetInd = tetIndices[tetrahedronIdx];
        const float4 v0 = verts[tetInd.x];
        const float4 v1 = verts[tetInd.y];
        const float4 v2 = verts[tetInd.z];
        const float4 v3 = verts[tetInd.w];

        // compute displacement field
        const float3 u1 = float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        const float3 u2 = float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        const float3 u3 = float3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);

        // load rest pose: loadPxMat33(restPoses[idx/32], idx&31, Qinv) over a
        // PxgMat33Block — the 9 elements are split 4/4/1 across mCol0/mCol1/mCol2.
        const uint64_t blockBase = gSoftbodies[softbodyId].mSimTetraRestPoses +
                                   uint64_t(tetrahedronIdx >> 5) * MAT33_BLOCK_STRIDE;
        const uint lane = tetrahedronIdx & 31u;
        const float4 c0v = Ptr<float4>(blockBase + uint64_t(lane) * 16UL)[0];
        const float4 c1v = Ptr<float4>(blockBase + MAT33_BLOCK_COL1_OFFSET + uint64_t(lane) * 16UL)[0];
        const float c2v = Ptr<float>(blockBase + MAT33_BLOCK_COL2_OFFSET + uint64_t(lane) * 4UL)[0];

        const float3 qinv0 = float3(c0v.x, c0v.y, c0v.z);
        const float3 qinv1 = float3(c0v.w, c1v.x, c1v.y);
        const float3 qinv2 = float3(c1v.z, c1v.w, c2v);

        const float4 q = sbTetRotation(u1, u2, u3, qinv0, qinv1, qinv2,
                                       tetRotations[tetrahedronIdx], initialRotation, 100);

        tetRotations[tetrahedronIdx] = q;
    }
}

// ---------------------------------------------------------------------------
// sb_updateTetrahedraRotationsLaunch (softBodyGM.cu:461) — COLLISION mesh.
// Same 16-byte arg pack (PxgSoftBodyCore.cpp:790).  Gated on
// eCOMPUTE_STRESS_TENSOR (mVolumeFlags) OR !eDISABLE_SELF_COLLISION (mBodyFlags).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_updateTetrahedraRotationsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gRot.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gRot.activeSoftbodies))[groupId.y];

    const uint tetrahedronIdx = groupThreadId.x + groupId.x * 256u;
    const uint nbTets = gSoftbodies[softbodyId].mNumTets;

    // The @416 fold: byte1 = mBodyFlags, bytes2..3 = mVolumeFlags.
    const uint flagsWord = gSoftbodies[softbodyId].mFlagsWord;
    const uint bodyFlags = (flagsWord >> 8) & 0xffu;
    const uint volumeFlags = (flagsWord >> 16) & 0xffffu;

    // KS - if we don't need stress tensors, we don't need to compute the tet rotations.
    if ((volumeFlags & PXG_SOFTBODY_VOLUMEFLAG_COMPUTE_STRESS_TENSOR) != 0u ||
        (bodyFlags & PXG_SOFTBODY_BODYFLAG_DISABLE_SELF_COLLISION) == 0u)
    {
        if (tetrahedronIdx < nbTets)
        {
            const float4 initialRotation = float4(gSoftbodies[softbodyId].mInitialRotationX,
                                                  gSoftbodies[softbodyId].mInitialRotationY,
                                                  gSoftbodies[softbodyId].mInitialRotationZ,
                                                  gSoftbodies[softbodyId].mInitialRotationW);

            Ptr<uint4> tetIndices = Ptr<uint4>(gSoftbodies[softbodyId].mTetIndices);
            Ptr<float4> verts = Ptr<float4>(gSoftbodies[softbodyId].mPosition_InvMass);
            Ptr<float4> tetRotations = Ptr<float4>(gSoftbodies[softbodyId].mTetraRotations);

            const uint4 tetInd = tetIndices[tetrahedronIdx];
            const float4 v0 = verts[tetInd.x];
            const float4 v1 = verts[tetInd.y];
            const float4 v2 = verts[tetInd.z];
            const float4 v3 = verts[tetInd.w];

            const float3 u1 = float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            const float3 u2 = float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            const float3 u3 = float3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);

            // FLAT PxMat33 array here (36-byte stride), not a PxgMat33Block.
            const uint64_t restBase = gSoftbodies[softbodyId].mTetraRestPoses +
                                      uint64_t(tetrahedronIdx) * 36UL;
            const float3 qinv0 = float3(Ptr<float>(restBase)[0], Ptr<float>(restBase)[1], Ptr<float>(restBase)[2]);
            const float3 qinv1 = float3(Ptr<float>(restBase)[3], Ptr<float>(restBase)[4], Ptr<float>(restBase)[5]);
            const float3 qinv2 = float3(Ptr<float>(restBase)[6], Ptr<float>(restBase)[7], Ptr<float>(restBase)[8]);

            const float4 q = sbTetRotation(u1, u2, u3, qinv0, qinv1, qinv2,
                                           tetRotations[tetrahedronIdx], initialRotation, 100);

            tetRotations[tetrahedronIdx] = q;
        }
    }
}


// ===========================================================================
// P8.0 tranche 6 — the FEM-VOLUME CONSTRAINT SOLVE, the core of the softbody solver.
//
//   sb_gm_cp_solveTetrahedronsPartitionLaunch (softBodyGM.cu:1408,
//   launch PxgSoftBodyCore.cpp:1202)
//
// One thread per tetrahedron of ONE graph-colouring partition.  It gathers the tet's four
// vertices, refreshes the co-rotational frame, runs the constitutive solve, and scatters
// the resulting position/velocity corrections back through the partition's remap table.
// Partitions are dispatched one at a time by the host, which is what makes the
// Gauss-Seidel scatter race-free without any atomics.
//
// SCOPE: the mNumTetsPerElement == 1 (pure tetrahedral mesh) path.  The hexahedral
// sibling (sb_gm_cp_solveHexahedronsPartition_parallelGS, taken when
// mNumTetsPerElement != 1) was NOT ported by this tranche, and until P9.95f the kernel
// performed no work at all for a voxel mesh -- a silent no-op for every softbody the
// stock extension path builds.  P9.95f ported it: see the P9.95f block below, which
// also carries the 9th-partition Jacobi entry that shares the same device function.
//
// Doctrine: descriptor through Ptr<> from the byte-exact pxgSlangSoftBodyDesc mirror.
// Fields touched: mSimVelocity_invMassCP @176, mSimPosition_InvMassCP @184,
// mSimRemapOutputCP @192, mSimAccumulatedPartitionsCP @208, mSimTetraMultipliers @264,
// mOrderedMaterialIndices @288, mSimTetraRestPoses @136, mSimTetraRotations @168,
// mSimPosition_InvMass @96, mSimVelocity_InvMass @88, mInitialRotation @372,
// mNumTetsGM @400, mNumPartitionsGM @404, mSimPullIndices @424, mNumTetsPerElement @436,
// plus the external PxsDeformableVolumeMaterialData array (32-byte stride:
// youngs @0, poissons @4, dynamicFriction @8, elasticityDamping @12, materialModel @16
// (PxU16), _pad @18 (PxU16), deformThreshold @20, ... -- the PhysX 5.12 layout).
//
// TOLERANCE SEAM: inherited, and only from the rotation refresh.  The solve calls
// computeGMTetrahedronRotations -> extractRotation, whose PxQuat(w, omega) evaluates
// sin/cos (see the tranche-5 header).  The CONSTITUTIVE MATHS ITSELF IS BIT-EXACT: every
// operation in volumeSolve / tetrahedronsSolveInnerNeoHookean / ARAP_constraint /
// volume_constraint / compute_deltaLambdaXgradC is mul/add/sub/div/cross/dot plus a
// single PxSqrt (IEEE-754 correctly rounded).  The unit test exploits that split — it
// drives the constitutive path with a PRE-SEEDED rotation and 0 rotation iterations so
// the material solve is compared bit-exactly, and covers the seam separately.
//
// DEVIATIONS from CUDA:
//   - the opening blockCopy of the descriptor into __shared__ + __syncthreads is dropped
//     (read-caching only), as in every prior softbody tranche;
//   - the two __syncwarp() calls are dropped.  They are NOT load-bearing here: each
//     thread reads and writes only its own tetrahedron's slots, and the gather/scatter go
//     through per-thread pull/remap indices with no cross-lane exchange whatsoever.  In
//     CUDA they exist to order the divergent pointer-selection loop against the loads
//     under the independent-thread-scheduling model; SPIR-V's memory model gives the same
//     guarantee for a single thread's own accesses without a barrier;
//   - the `#pragma unroll` 4-iteration gather/scatter loops become explicit 4-element
//     sequences (Slang has no equivalent pragma and the trip count is a literal 4).
// ===========================================================================

// Most significant bit is 0, the remaining 31 bits are 1.  The MSB of a pull index stores
// the tet's flipped state (softBodyGM.cu:1044).
static const uint PULL_IND_MASK = 0x7fffffffu;
static const float SB_STIFFNESS_THRESHOLD = 1.0e-18f;
static const float SB_ARAP_EPSILON = 1.0e-18f;

// PxDeformableVolumeMaterialModel (PxDeformableVolumeMaterial.h:43): eCO_ROTATIONAL = 0
// (the default), eNEO_HOOKEAN = 1.
static const uint PXG_SB_MATERIAL_CO_ROTATIONAL = 0u;
static const uint PXG_SB_MATERIAL_NEO_HOOKEAN = 1u;

// PxsDeformableVolumeMaterialData (PxsDeformableVolumeMaterialCore.h:50), 32-byte stride.
static const uint64_t SB_MATERIAL_STRIDE = 32UL;

struct SbMaterial
{
    float youngs;
    float poissons;
    float elasticityDamping;
    uint materialModel;
}

SbMaterial sbLoadMaterial(uint64_t materials, uint index)
{
    const uint64_t base = materials + uint64_t(index) * SB_MATERIAL_STRIDE;
    SbMaterial m;
    m.youngs = Ptr<float>(base)[0];
    m.poissons = Ptr<float>(base)[1];
    m.elasticityDamping = Ptr<float>(base)[3];
    // PhysX 5.12: materialModel is a PxU16 @16 and _pad a PxU16 @18 -- one 32-bit word.
    // dampingScale is gone from PxsDeformableVolumeMaterialData, and the GM solve damps with
    // elasticityDamping alone (compute_deltaLambdaXgradC's `damping * (...)`, where 5.6.1
    // had a separate scaledDamping = damping * toUniformReal(dampingScale)).
    const uint packed = Ptr<uint>(base + 16UL)[0];
    m.materialModel = packed & 0xffffu;
    return m;
}

// Element i of a PxU16 array at `base` (mOrderedMaterialIndices): read the containing 32-bit
// word.  P9.95f factored this out of the GS tet body so the hex path reads the same way.
uint sbReadU16(uint64_t base, uint i)
{
    const uint64_t matIdxAddr = base + uint64_t(i) * 2UL;
    // NOTE THE `~uint64_t(3)`.  Writing the obvious `matIdxAddr & ~3UL` MISCOMPILES:
    // Slang evaluates `~3UL` in 32 bits, and the resulting 0xFFFFFFFC is then
    // zero-extended for the 64-bit AND — which silently clears the TOP HALF of the
    // device address.  The read then lands at a garbage address and the driver
    // faults.  Verified in isolation (tools/lvprepro): `& ~3UL` faults, while
    // `& ~uint64_t(3)`, `& ~3ull`, `& 0xFFFFFFFFFFFFFFFCUL` and
    // `- (matIdxAddr % 4UL)` all run clean on the same shader and data.  Note that
    // `ull` DOES widen the complement correctly while `UL` does not — the sweep in
    // tools/lvprepro/ultest.slang prints the surviving address for each spelling.
    const uint matWord = Ptr<uint>(matIdxAddr & ~uint64_t(3))[0];
    // Which half of the word: derive it from the ADDRESS, not from the index's
    // parity — that only coincides when the array base is 4-aligned, and nothing
    // in the engine promises that.
    const uint matHalf = uint((matIdxAddr >> 1) & 1UL);
    return (matHalf != 0u) ? (matWord >> 16) : (matWord & 0xffffu);
}

// --- PxMat33 as a struct of three columns ---------------------------------------------
// Passing the three columns as separate parameters produced a function with 23 arguments
// and deeply nested `out` chains, which llvmpipe's JIT miscompiled into a SIGSEGV (the
// bisection: an EMPTY solve body ran clean, while EITHER material branch alone crashed —
// i.e. it tracked total call-frame width, not any one expression).  Bundling the columns
// keeps the transcription line-for-line while cutting the widest signature to 12.
struct SbMat33
{
    float3 c0, c1, c2;
}

SbMat33 sbMat33(float3 a, float3 b, float3 c) { SbMat33 m; m.c0 = a; m.c1 = b; m.c2 = c; return m; }
SbMat33 sbMat33Zero() { return sbMat33(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f)); }

// --- PxMat33 helpers, columns as three float3 (see the file header on convention) -----

// A*B: col_i(result) = A.transform(B.col_i)
SbMat33 sbMat33MulM(SbMat33 a, SbMat33 b)
{
    return sbMat33(sbMat33Transform(a.c0, a.c1, a.c2, b.c0),
                   sbMat33Transform(a.c0, a.c1, a.c2, b.c1),
                   sbMat33Transform(a.c0, a.c1, a.c2, b.c2));
}

SbMat33 sbMat33Transpose(SbMat33 m)
{
    return sbMat33(float3(m.c0[0], m.c1[0], m.c2[0]),
                   float3(m.c0[1], m.c1[1], m.c2[1]),
                   float3(m.c0[2], m.c1[2], m.c2[2]));
}

SbMat33 sbMat33FromQuatM(float4 q)
{
    float3 c0, c1, c2;
    sbMat33FromQuat(q, c0, c1, c2);
    return sbMat33(c0, c1, c2);
}

// PxMat33::getDeterminant() (PxMat33.h): column0.dot(column1.cross(column2))
float sbMat33Det(SbMat33 m) { return sbDot3(m.c0, sbCross3(m.c1, m.c2)); }

bool sbMat33IsZero(SbMat33 m)
{
    return m.c0.x == 0.0f && m.c0.y == 0.0f && m.c0.z == 0.0f &&
           m.c1.x == 0.0f && m.c1.y == 0.0f && m.c1.z == 0.0f &&
           m.c2.x == 0.0f && m.c2.y == 0.0f && m.c2.z == 0.0f;
}

// The four tet vertex weights, bundled so the constraint helpers take one argument.
struct SbTetV
{
    float4 v0, v1, v2, v3;
}

// --- compute_dCdx (softBodyGM.cu:532) -------------------------------------------------
// dC/dx = dF/dx : dC/dF, using the ROW vectors of DmInv and dCdF.
SbMat33 sbComputeDCdx(SbMat33 dCdF, SbMat33 DmInv)
{
    // row vectors of DmInv and dCdF
    const SbMat33 r = sbMat33Transpose(DmInv);
    const SbMat33 p = sbMat33Transpose(dCdF);

    return sbMat33(float3(sbDot3(r.c0, p.c0), sbDot3(r.c0, p.c1), sbDot3(r.c0, p.c2)),
                   float3(sbDot3(r.c1, p.c0), sbDot3(r.c1, p.c1), sbDot3(r.c1, p.c2)),
                   float3(sbDot3(r.c2, p.c0), sbDot3(r.c2, p.c1), sbDot3(r.c2, p.c2)));
}

// IEEE `0 + x` per component as a select (P9.86): +0 for either zero, x otherwise.
float3 sbPlusZero3(float3 v)
{
    return float3((v.x == 0.0f) ? 0.0f : v.x, (v.y == 0.0f) ? 0.0f : v.y, (v.z == 0.0f) ? 0.0f : v.z);
}

// --- compute_deltaLambdaXgradC (softBodyGM.cu:554) ------------------------------------
// Accumulates deltaLambda * dCdx into deltaLambdaXgradC and advances lambda.
void sbComputeDeltaLambdaXGradC(inout SbMat33 dl, inout float lambda, float C,
                                SbMat33 dCdF, SbMat33 DmInv, SbTetV tv,
                                float alphaTilde, float dtInv, float damping)
{
    const SbMat33 dCdx = sbComputeDCdx(dCdF, DmInv);
    const float3 dCdx1 = dCdx.c0, dCdx2 = dCdx.c1, dCdx3 = dCdx.c2;
    const float4 v0 = tv.v0, v1 = tv.v1, v2 = tv.v2, v3 = tv.v3;

    const float3 dCdx0 = -dCdx1 - dCdx2 - dCdx3;
    float deltaLambda;

    if (damping == 0.0f) // no damping
    {
        const float denom = sbDot3(dCdx0, dCdx0) * v0.w + sbDot3(dCdx1, dCdx1) * v1.w +
                            sbDot3(dCdx2, dCdx2) * v2.w + sbDot3(dCdx3, dCdx3) * v3.w + alphaTilde;
        if (abs(denom) < 1.0e-15f)
            return;

        deltaLambda = (-C - alphaTilde * lambda) / denom;
    }
    else
    {
        const float denom =
            (1.0f + damping * dtInv) * (sbDot3(dCdx0, dCdx0) * v0.w + sbDot3(dCdx1, dCdx1) * v1.w +
                                        sbDot3(dCdx2, dCdx2) * v2.w + sbDot3(dCdx3, dCdx3) * v3.w) +
            alphaTilde;

        if (abs(denom) < 1.0e-15f)
        {
            dl = sbMat33Zero();
            return;
        }

        deltaLambda = (-C - alphaTilde * lambda -
                       damping * (sbDot3(dCdx0, float3(v0.x, v0.y, v0.z)) +
                                        sbDot3(dCdx1, float3(v1.x, v1.y, v1.z)) +
                                        sbDot3(dCdx2, float3(v2.x, v2.y, v2.z)) +
                                        sbDot3(dCdx3, float3(v3.x, v3.y, v3.z)))) /
                      denom;
    }

    lambda += deltaLambda;

    // return: deltaLambda * dCdx  (ACCUMULATES — the caller may already hold a term)
    // P9.86: the CUDA accumulator starts at PxMat33(PxZero) (softBodyGM.cu:1122/1181, reset
    // at :588) and is only ever += a term (:601-603), so in IEEE it never holds -0.  After
    // inlining, the +0 seed reaches the ARAP add and llvmpipe folds `0 + x` to x, keeping a
    // -0 product; canonicalise -0 -> +0 (no-op for nonzero/inf/NaN): bit for bit the CUDA.
    dl.c0 = sbPlusZero3(dl.c0 + deltaLambda * dCdx1);
    dl.c1 = sbPlusZero3(dl.c1 + deltaLambda * dCdx2);
    dl.c2 = sbPlusZero3(dl.c2 + deltaLambda * dCdx3);
}

// --- ARAP_constraint (softBodyGM.cu:722) ----------------------------------------------
void sbARAPConstraint(inout SbMat33 dl, inout float lambda, SbTetV tv, float4 tetR,
                      SbMat33 F, float dtInv, SbMat33 DmInv,
                      float alphaTilde, float damping)
{
    const SbMat33 R = sbMat33FromQuatM(tetR);
    SbMat33 t = sbMat33(F.c0 - R.c0, F.c1 - R.c1, F.c2 - R.c2);

    // PxSqrt is IEEE-754 correctly rounded -> bit-exact.
    const float ARAP_C = sqrt(sbDot3(t.c0, t.c0) + sbDot3(t.c1, t.c1) + sbDot3(t.c2, t.c2) + SB_ARAP_EPSILON);

    const float inv = 1.0f / ARAP_C;
    t.c0 *= inv; t.c1 *= inv; t.c2 *= inv;   // dCdF

    sbComputeDeltaLambdaXGradC(dl, lambda, ARAP_C, t, DmInv, tv, alphaTilde, dtInv, damping);
}

// --- volume_constraint (softBodyGM.cu:756) --------------------------------------------
void sbVolumeConstraint(inout SbMat33 dl, inout float lambda, SbTetV tv,
                        SbMat33 F, float invDt, SbMat33 DmInv,
                        float alphaTilde, float damping)
{
    const float detF = sbMat33Det(F);
    const float C = detF - 1.0f;
    const SbMat33 dCdF = sbMat33(sbCross3(F.c1, F.c2), sbCross3(F.c2, F.c0), sbCross3(F.c0, F.c1));

    sbComputeDeltaLambdaXGradC(dl, lambda, C, dCdF, DmInv, tv, alphaTilde, invDt, damping);
}

// --- volumeSolve (softBodyGM.cu:621) --------------------------------------------------
SbMat33 sbVolumeSolve(SbMat33 P, SbTetV tv, float invDtSq, SbMat33 Qinv, float volumeLambda)
{
    const float det = sbMat33Det(Qinv);
    const float volume = 1.0f / (6.0f * det);

    const float alphaTilde = (volumeLambda == 0.0f) ? 0.0f : 1.0f / (volumeLambda * volume * invDtSq);

    const SbMat33 F = sbMat33MulM(P, Qinv);   // F = P * Qinv

    const float3 g0 = sbCross3(F.c1, F.c2);
    const float3 g1 = sbCross3(F.c2, F.c0);
    const float3 g2 = sbCross3(F.c0, F.c1);

    const float3 grad1 = g0 * Qinv.c0.x + g1 * Qinv.c1.x + g2 * Qinv.c2.x;
    const float3 grad2 = g0 * Qinv.c0.y + g1 * Qinv.c1.y + g2 * Qinv.c2.y;
    const float3 grad3 = g0 * Qinv.c0.z + g1 * Qinv.c1.z + g2 * Qinv.c2.z;
    const float3 grad0 = -grad1 - grad2 - grad3;

    const float detF = sbDot3(g2, F.c2);
    const float C = (detF - 1.0f);

    const float w = sbDot3(grad0, grad0) * tv.v0.w
                  + sbDot3(grad1, grad1) * tv.v1.w
                  + sbDot3(grad2, grad2) * tv.v2.w
                  + sbDot3(grad3, grad3) * tv.v3.w;

    const float dlambda = -C / (w + alphaTilde);

    return sbMat33(grad1 * dlambda, grad2 * dlambda, grad3 * dlambda);
}

// --- tetrahedronsSolveInnerNeoHookean (softBodyGM.cu:657) -----------------------------
SbMat33 sbSolveInnerNeoHookean(SbMat33 P, SbTetV tv, uint64_t multipliers, float invDtSq,
                               uint workIndex, SbMat33 Qinv, bool isTGS, float invE)
{
    const SbMat33 F = sbMat33MulM(P, Qinv);   // F = P * Qinv

    const float3 g1 = F.c0 * 2.0f * Qinv.c0.x + F.c1 * 2.0f * Qinv.c1.x + F.c2 * 2.0f * Qinv.c2.x;
    const float3 g2 = F.c0 * 2.0f * Qinv.c0.y + F.c1 * 2.0f * Qinv.c1.y + F.c2 * 2.0f * Qinv.c2.y;
    const float3 g3 = F.c0 * 2.0f * Qinv.c0.z + F.c1 * 2.0f * Qinv.c1.z + F.c2 * 2.0f * Qinv.c2.z;

    const float C = sbDot3(F.c0, F.c0) + sbDot3(F.c1, F.c1) + sbDot3(F.c2, F.c2) - 3.0f;

    const float3 g0 = -g1 - g2 - g3;

    const float w = sbDot3(g0, g0) * tv.v0.w + sbDot3(g1, g1) * tv.v1.w +
                    sbDot3(g2, g2) * tv.v2.w + sbDot3(g3, g3) * tv.v3.w;

    const float alpha = invDtSq * invE;
    float dlambda;

    if (!isTGS)
    {
        // mSimTetraMultipliers[workIndex/32].mbyz[workIndex&31] — a float2 in the
        // PxgSpatialVectorBlock's second array.
        const uint64_t mbyz = multipliers + uint64_t(workIndex >> 5) * SPATIAL_BLOCK_STRIDE +
                              SPATIAL_BLOCK_MBYZ_OFFSET + uint64_t(workIndex & 31u) * 8UL;
        float2 totalLambda = Ptr<float2>(mbyz)[0];

        dlambda = -(C + alpha * totalLambda.x) / (w + alpha);

        totalLambda.x += dlambda;
        Ptr<float2>(mbyz)[0] = totalLambda;
    }
    else
    {
        dlambda = -C / (w + alpha);
    }

    return sbMat33(g1 * dlambda, g2 * dlambda, g3 * dlambda);
}

// --- tetrahedronsSolveGM (softBodyGM.cu:926) — the material dispatcher ----------------
SbMat33 sbTetrahedronsSolveGM(SbMat33 U, SbTetV tv, uint64_t multipliers,
                              float invDtSq, float invDt, uint workIndex,
                              SbMat33 Qinv, float4 tetR, bool isTGS, SbMaterial material)
{
    SbMat33 dl = sbMat33Zero();

    if (material.youngs < SB_STIFFNESS_THRESHOLD)
        return dl;

    if (material.materialModel == PXG_SB_MATERIAL_NEO_HOOKEAN)
    {
        const float invE = 1.0f / material.youngs;
        const SbMat33 R = sbMat33FromQuatM(tetR);
        // P = RTranspose * PxMat33(u1, u2, u3)
        SbMat33 P = sbMat33MulM(sbMat33Transpose(R), U);

        SbMat33 d = sbSolveInnerNeoHookean(P, tv, multipliers, invDtSq, workIndex, Qinv, isTGS, invE);

        const float3 delta0 = -(d.c0 + d.c1 + d.c2) * tv.v0.w;
        P.c0 += (d.c0 * tv.v1.w - delta0);
        P.c1 += (d.c1 * tv.v2.w - delta0);
        P.c2 += (d.c2 * tv.v3.w - delta0);

        if (material.poissons > SB_STIFFNESS_THRESHOLD)
        {
            const float nu = material.poissons;
            const float volumeLambda = (nu < 0.5f - SB_STIFFNESS_THRESHOLD)
                                           ? nu / ((1.0f - 2.0f * nu) * material.youngs)
                                           : 0.0f;
            const SbMat33 sv = sbVolumeSolve(P, tv, invDtSq, Qinv, volumeLambda);
            d.c0 += sv.c0; d.c1 += sv.c1; d.c2 += sv.c2;
        }

        // deltaLambdaXgradC = R * delta
        dl = sbMat33MulM(R, d);
    }
    else // PxDeformableVolumeMaterialModel::eCO_ROTATIONAL
    {
        // lameParameters (deformableUtils.cuh:124)
        const float lambdaLame = material.youngs * material.poissons /
                                 ((1.0f + material.poissons) * (1.0f - 2.0f * material.poissons));
        const float mu = material.youngs / (2.0f * (1.0f + material.poissons));

        const float det = sbMat33Det(Qinv);
        const float volume = 1.0f / (6.0f * det);
        SbMat33 F = sbMat33MulM(U, Qinv);   // F = PxMat33(u1,u2,u3) * Qinv

        const float alphaTilde0 = invDtSq / (2.0f * mu * volume);

        float damping = material.elasticityDamping;

        const uint64_t mbyz = multipliers + uint64_t(workIndex >> 5) * SPATIAL_BLOCK_STRIDE +
                              SPATIAL_BLOCK_MBYZ_OFFSET + uint64_t(workIndex & 31u) * 8UL;
        float2 lambdas = isTGS ? float2(0.0f, 0.0f) : Ptr<float2>(mbyz)[0];

        sbARAPConstraint(dl, lambdas.x, tv, tetR, F, invDt, Qinv, alphaTilde0, damping);

        const float3 dx0 = -(dl.c0 + dl.c1 + dl.c2) * tv.v0.w;
        const float3 dx1 = dl.c0 * tv.v1.w;
        const float3 dx2 = dl.c1 * tv.v2.w;
        const float3 dx3 = dl.c2 * tv.v3.w;

        const SbMat33 X = sbMat33(U.c0 + (dx1 - dx0), U.c1 + (dx2 - dx0), U.c2 + (dx3 - dx0));

        F = sbMat33MulM(X, Qinv);   // F rebuilt

        if (material.poissons > SB_STIFFNESS_THRESHOLD)
        {
            float alphaTilde1 = 0.0f;
            float ratio = 0.0f;

            if (material.poissons < 0.5f - SB_STIFFNESS_THRESHOLD)
            {
                alphaTilde1 = invDtSq / (lambdaLame * volume);
                ratio = 2.0f * mu / lambdaLame;   // alphaTilde1/alphaTilde0 for volume damping
            }

            damping *= ratio;   // damping for the volume term

            sbVolumeConstraint(dl, lambdas.y, tv, F, invDt, Qinv, alphaTilde1, damping);
        }

        if (!isTGS)
            Ptr<float2>(mbyz)[0] = lambdas;
    }

    return dl;
}

// One vertex of the gather (softBodyGM.cu:1073-1094): the pull index selects between the
// unordered (shared) buffers and this tet's own slot in the ordered CP buffers.
void sbGather(Ptr<float4> unorderedPositions, Ptr<float4> unorderedVelocities,
              Ptr<float4> curPositions, Ptr<float4> velocities,
              uint pullIndex, uint cpIndex, out float4 p, out float4 v)
{
    if (pullIndex < PULL_IND_MASK)
    {
        p = unorderedPositions[pullIndex];
        v = unorderedVelocities[pullIndex];
    }
    else
    {
        p = curPositions[cpIndex];
        v = velocities[cpIndex];
    }
}

// One vertex of the scatter (softBodyGM.cu:1136-1145), factored out so the loop can be
// unrolled without a dynamically-indexed local array (see the call site).
void sbScatter(Ptr<float4> curPositions, Ptr<float4> velocities, uint writeIndex,
               float4 p, float4 v, float3 deltaPos, float invDt)
{
    curPositions[writeIndex] = float4(p.x + deltaPos.x, p.y + deltaPos.y, p.z + deltaPos.z, p.w);
    const float3 deltaV = deltaPos * invDt;
    velocities[writeIndex] = float4(v.x + deltaV.x, v.y + deltaV.y, v.z + deltaV.z, v.w);
}

// ===========================================================================
// P9.95f — the HEXAHEDRAL (voxel) solve.
//
//   sb_gm_cp_solveHexahedronsPartition_parallelGS<updatePosVel> (softBodyGM.cu:1313-1404)
//     with its per-sub-tet body sb_gm_cp_solveSubTetrahedron (softBodyGM.cu:1149-1205),
//   called from TWO kernels:
//     <true>  the hex `else` branch of sb_gm_cp_solveTetrahedronsPartitionLaunch
//             (softBodyGM.cu:1449-1455; launch PxgSoftBodyCore.cpp:1196-1213), and
//     <false> sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch (softBodyGM.cu:1463-1507;
//             LAUNCH PxgSoftBodyCore.cpp:1230-1247, guarded by hasExtraJacobi =
//             maxPartitions > 8 at :1185; grid (ceil(maxJacobiTets/32), nbActive, 1),
//             block SB_SOLVETETRA_LOW = 32).
//
// A voxel mesh (PxTetrahedronMeshDesc::tetsPerElement 5 or 6) is solved per ELEMENT: one
// thread gathers the voxel's 8 corners through mSimPullIndices[w] and [w + nbElements]
// (the MSB of pull[w].x is the voxel's FLIPPED bit), solves its 5 or 6 sub-tets in order
// on the gathered copies, and writes back.  The GS form writes positions/velocities in
// place and accumulates mSimDeltaPos; the Jacobi form (the cooker's 9th partition: voxels
// that share a vertex with a duplicate of their own cell, GuCookingTetrahedronMesh.cpp:
// 696-768) only atomically accumulates (p - pos, 1) into mSimDelta, which
// sb_gm_applyDeformationDeltasLaunch folds back after the iteration.
//
// CUDA line map:
//   :1044-1046  PULL_IND_MASK / FLIPPED_MASK
//   :1149-1205  sub-tet solve -> sbSolveSubTetrahedron
//   :1209       tets6PerVoxel -> sbTets6PerVoxel;  :1211-1213 tets5PerVoxel -> sbTets5PerVoxel
//   :1317-1374  gather + sub-tet loop -> sbSolveHexahedronsPartition
//   :1378-1390  GS tail -> sbHexWriteGS;  :1392-1403 Jacobi tail -> sbHexWriteJacobi
//   :1442-1455  GS kernel dispatch;  :1463-1507 Jacobi kernel
//
// TRAPS:
//   H1  early return.  On a zero rest pose (Qinv == 0) the sub-tet returns AFTER storing
//       its rotation and BEFORE any `p += (dp, 0)` (:1176-1177): p and v stay bit-
//       untouched.  The tet path (sbScatter) still scatters a zero delta -- not reused.
//   H2  w lanes.  `p += make_float4(d, 0.f)` is IEEE `p.w + 0.f`, which turns -0 into +0
//       (likewise v.w).  slangc folds a literal-zero add, so the w lane is spelled as the
//       select `(a.w == 0.0f) ? 0.0f : a.w` (P9.86).  The GS tail's `mSimDeltaPos += p - pos`
//       needs NO select: its w addend is the runtime difference p.w - pos.w (+0 for a finite
//       w), not a literal, so the IEEE add itself maps a -0 deltaPos.w to +0 and is spelled
//       as the plain add (a select there was measured redundant, P9.95f).
//   H3  no dynamically indexed locals.  CUDA's float4 p[8], v[8] indexed through the
//       table are 16 NAMED members (SbHexPV) with a select-chain get and an if-chain put,
//       and the tables are switches returning uint4 literals (the llvmpipe JIT landmine
//       documented at the tet scatter below, and the 23-argument SIGSEGV note above);
//       every signature here stays at 12 parameters or fewer.
//   H4  clamped index.  The element index feeds indirect loads (pull indices, then
//       positions), so the guarded index is clamped before use (P9.22/P9.23): Jacobi
//       min(w, end-1) (end >= 1 past the :1482 guard), GS end > 0 ? min(w, end-1) : 0.
//       The value is unchanged for every live lane.
//   H5  isFirstIteration.  The Jacobi kernel hard-wires false (:1503: 4 rotation
//       iterations even on the first solver iteration); the GS hex branch forwards the
//       push flag (100 or 4).
//   H6  the w atomic is real.  AtomicAdd(float4&, PxVec3, 1.f) (atomic.cuh:46-52) does
//       four atomicAdds, so mSimDelta.w counts the contributions; the "4th value is
//       unused" comment at :1401 means unused by the consumer, not "not added".
//   H7  blockCopy dropped, as in every softbody tranche.  This also removes the
//       Jacobi kernel's missing __syncthreads after its blockCopy (:1478; the GS kernel
//       has one at :1425) -- an upstream race masked by blockDim == warpSize.
//   H8  isTGS is byte 0 of the @24 word; bytes 25..27 and the @28 word are pad (P9.87).
//   H9  nbElements = nbTets / tpe, nbTets = mNumTetsGM (the SUB-tet count).  The sub-tet
//       slot is w + nbElements*i -- for rest poses, rotations and multipliers alike --
//       not w*tpe + i (PxgSoftBody.cpp:55-92 lays the rest poses out that way).
//
// TOLERANCE SEAM: exactly the tet path's -- only the rotation refresh (sin/cos in
// extractRotation); the sub-tet maths is the bit-exact tetrahedronsSolveGM above.
// ===========================================================================

// Opposite of PULL_IND_MASK: the MSB of pull[w].x is the voxel's flipped bit (softBodyGM.cu:1046).
static const uint FLIPPED_MASK = 0x80000000u;
static const uint SB_PARTITION_LIMIT = 8u;   // PxgSoftBody.h: the 9th partition is the Jacobi one

// tets5PerVoxel (softBodyGM.cu:1211-1213), addressed by ROW = (4*i + offset)/4: rows 0..4
// are the non-flipped voxel, rows 5..9 (flat offset 20) the flipped one.
uint4 sbTets5PerVoxel(uint row)
{
    switch (row)
    {
    case 0u: return uint4(0u, 6u, 3u, 5u);
    case 1u: return uint4(0u, 1u, 5u, 3u);
    case 2u: return uint4(6u, 7u, 3u, 5u);
    case 3u: return uint4(4u, 5u, 6u, 0u);
    case 4u: return uint4(2u, 3u, 0u, 6u);
    case 5u: return uint4(1u, 7u, 4u, 2u);
    case 6u: return uint4(1u, 0u, 2u, 4u);
    case 7u: return uint4(7u, 6u, 4u, 2u);
    case 8u: return uint4(5u, 4u, 1u, 7u);
    default: return uint4(3u, 2u, 7u, 1u);   // row 9
    }
}

// tets6PerVoxel (softBodyGM.cu:1209)
uint4 sbTets6PerVoxel(uint row)
{
    switch (row)
    {
    case 0u: return uint4(0u, 1u, 6u, 2u);
    case 1u: return uint4(0u, 1u, 4u, 6u);
    case 2u: return uint4(1u, 4u, 6u, 5u);
    case 3u: return uint4(1u, 2u, 3u, 6u);
    case 4u: return uint4(1u, 3u, 7u, 6u);
    default: return uint4(1u, 5u, 6u, 7u);   // row 5
    }
}

// The .cu's float4 p[8], v[8] as 16 NAMED members (trap H3).
struct SbHexPV
{
    float4 p0, p1, p2, p3, p4, p5, p6, p7;
    float4 v0, v1, v2, v3, v4, v5, v6, v7;
}

float4 sbHexP(SbHexPV h, uint c)
{
    return c == 0u ? h.p0 : c == 1u ? h.p1 : c == 2u ? h.p2 : c == 3u ? h.p3 :
           c == 4u ? h.p4 : c == 5u ? h.p5 : c == 6u ? h.p6 : h.p7;
}

float4 sbHexV(SbHexPV h, uint c)
{
    return c == 0u ? h.v0 : c == 1u ? h.v1 : c == 2u ? h.v2 : c == 3u ? h.v3 :
           c == 4u ? h.v4 : c == 5u ? h.v5 : c == 6u ? h.v6 : h.v7;
}

void sbHexPut(inout SbHexPV h, uint c, float4 p, float4 v)
{
    if (c == 0u) { h.p0 = p; h.v0 = v; }
    if (c == 1u) { h.p1 = p; h.v1 = v; }
    if (c == 2u) { h.p2 = p; h.v2 = v; }
    if (c == 3u) { h.p3 = p; h.v3 = v; }
    if (c == 4u) { h.p4 = p; h.v4 = v; }
    if (c == 5u) { h.p5 = p; h.v5 = v; }
    if (c == 6u) { h.p6 = p; h.v6 = v; }
    if (c == 7u) { h.p7 = p; h.v7 = v; }
}

// `a += make_float4(d.x, d.y, d.z, 0.f)` (softBodyGM.cu:1187-1204): the w lane is the IEEE
// `a.w + 0.f`, -0 -> +0 (trap H2).
float4 sbAddXyzW0(float4 a, float3 d)
{
    return float4(a.x + d.x, a.y + d.y, a.z + d.z, (a.w == 0.0f) ? 0.0f : a.w);
}

// The descriptor fields the hex solve touches, read once per thread.
struct SbHexDesc
{
    uint64_t positions;      // mSimPosition_InvMass @96
    uint64_t velocities;     // mSimVelocity_InvMass @88
    uint64_t pullInds;       // mSimPullIndices @400
    uint64_t restPoses;      // mSimTetraRestPoses @136 (PxgMat33Block, 1152-byte stride)
    uint64_t rotations;      // mSimTetraRotations @168
    uint64_t multipliers;    // mSimTetraMultipliers @264
    uint64_t deltaPos;       // mSimDeltaPos @128
    uint64_t delta;          // mSimDelta @272
    uint64_t materialIdx;    // mOrderedMaterialIndices @288 (PxU16*)
    float4 initialRotation;  // mInitialRotation @372..@384 (four scalars, not 16-aligned)
}

SbHexDesc sbHexDescOf(Ptr<PxgSoftBodyGpu> sb, uint id)
{
    SbHexDesc d;
    d.positions = sb[id].mSimPosition_InvMass;
    d.velocities = sb[id].mSimVelocity_InvMass;
    d.pullInds = sb[id].mSimPullIndices;
    d.restPoses = sb[id].mSimTetraRestPoses;
    d.rotations = sb[id].mSimTetraRotations;
    d.multipliers = sb[id].mSimTetraMultipliers;
    d.deltaPos = sb[id].mSimDeltaPos;
    d.delta = sb[id].mSimDelta;
    d.materialIdx = sb[id].mOrderedMaterialIndices;
    d.initialRotation = float4(sb[id].mInitialRotationX, sb[id].mInitialRotationY,
                               sb[id].mInitialRotationZ, sb[id].mInitialRotationW);
    return d;
}

// sb_gm_cp_solveSubTetrahedron (softBodyGM.cu:1149-1205).  c = the four voxel corners of
// sub-tet i (one table row), subIdx = its slot w + nbElements*i (trap H9).
void sbSolveSubTetrahedron(inout SbHexPV h, uint4 c, SbHexDesc d, uint subIdx, float invDtSq,
                           float invDt, bool isTGS, bool isFirstIteration, SbMaterial material)
{
    float4 p0 = sbHexP(h, c.x);
    float4 p1 = sbHexP(h, c.y);
    float4 p2 = sbHexP(h, c.z);
    float4 p3 = sbHexP(h, c.w);
    float4 v0 = sbHexV(h, c.x);
    float4 v1 = sbHexV(h, c.y);
    float4 v2 = sbHexV(h, c.z);
    float4 v3 = sbHexV(h, c.w);

    // loadPxMat33(mSimTetraRestPoses[subIdx / 32], subIdx & 31) -- the GS tet path's load
    const uint64_t restBase = d.restPoses + uint64_t(subIdx >> 5) * MAT33_BLOCK_STRIDE;
    const uint lane = subIdx & 31u;
    const float4 c0v = Ptr<float4>(restBase + uint64_t(lane) * 16UL)[0];
    const float4 c1v = Ptr<float4>(restBase + MAT33_BLOCK_COL1_OFFSET + uint64_t(lane) * 16UL)[0];
    const float c2v = Ptr<float>(restBase + MAT33_BLOCK_COL2_OFFSET + uint64_t(lane) * 4UL)[0];
    const SbMat33 Qinv = sbMat33(float3(c0v.x, c0v.y, c0v.z),
                                 float3(c0v.w, c1v.x, c1v.y),
                                 float3(c1v.z, c1v.w, c2v));

    float4 q = Ptr<float4>(d.rotations)[subIdx];

    const float3 u1 = float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    const float3 u2 = float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
    const float3 u3 = float3(p3.x - p0.x, p3.y - p0.y, p3.z - p0.z);

    {
        if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) == 0.0f)
            q = d.initialRotation;

        // computeGMTetrahedronRotations: 100 iterations on the first solver iteration, 4 after.
        const SbMat33 Fm = sbMat33MulM(sbMat33(u1, u2, u3), Qinv);
        q = sbExtractRotation(Fm.c0, Fm.c1, Fm.c2, q, isFirstIteration ? 100 : 4);

        Ptr<float4>(d.rotations)[subIdx] = q;
    }

    // softBodyGM.cu:1176 -- the EARLY RETURN: p and v untouched (trap H1)
    if (sbMat33IsZero(Qinv))
        return;

    SbTetV tv; tv.v0 = v0; tv.v1 = v1; tv.v2 = v2; tv.v3 = v3;
    const SbMat33 dl = sbTetrahedronsSolveGM(sbMat33(u1, u2, u3), tv, d.multipliers, invDtSq, invDt,
                                             subIdx, Qinv, q, isTGS, material);

    // c0 + c1 + c2, left-associated as the PxVec3 operators evaluate it (:1184)
    const float3 sum = dl.c0 + dl.c1 + dl.c2;
    const float3 deltaPos0 = -sum * p0.w;
    p0 = sbAddXyzW0(p0, deltaPos0);
    v0 = sbAddXyzW0(v0, deltaPos0 * invDt);
    const float3 deltaPos1 = dl.c0 * p1.w;
    p1 = sbAddXyzW0(p1, deltaPos1);
    v1 = sbAddXyzW0(v1, deltaPos1 * invDt);
    const float3 deltaPos2 = dl.c1 * p2.w;
    p2 = sbAddXyzW0(p2, deltaPos2);
    v2 = sbAddXyzW0(v2, deltaPos2 * invDt);
    const float3 deltaPos3 = dl.c2 * p3.w;
    p3 = sbAddXyzW0(p3, deltaPos3);
    v3 = sbAddXyzW0(v3, deltaPos3 * invDt);

    sbHexPut(h, c.x, p0, v0);
    sbHexPut(h, c.y, p1, v1);
    sbHexPut(h, c.z, p2, v2);
    sbHexPut(h, c.w, p3, v3);
}

// One corner of the GS tail (softBodyGM.cu:1382-1388), in the .cu's order: deltaPos, then
// position, then velocity.  `mSimDeltaPos[index] += p - pos` is the plain IEEE add (trap H2).
void sbHexWriteGS(SbHexDesc d, uint index, float4 p, float4 v)
{
    const float4 pos = Ptr<float4>(d.positions)[index];
    const float4 dp = Ptr<float4>(d.deltaPos)[index] + (p - pos);
    Ptr<float4>(d.deltaPos)[index] = dp;
    Ptr<float4>(d.positions)[index] = p;
    Ptr<float4>(d.velocities)[index] = v;
}

// One corner of the Jacobi tail (softBodyGM.cu:1396-1402): a FRESH position load, then
// AtomicAdd(mSimDelta[index], (p - pos).xyz, 1.f).
void sbHexWriteJacobi(SbHexDesc d, uint index, float4 p)
{
    const float4 delta4 = p - Ptr<float4>(d.positions)[index];
    pxgAtomicAddF4XYZ(d.delta, index, float3(delta4.x, delta4.y, delta4.z));
    pxgAtomicAddF4W(d.delta, index, 1.0f);   // trap H6
}

// sb_gm_cp_solveHexahedronsPartition_parallelGS<updatePosVel> (softBodyGM.cu:1313-1404)
void sbSolveHexahedronsPartition(SbHexDesc d, uint workIndex, uint nbTets, float invDt, bool isTGS,
                                 bool isFirstIteration, uint64_t materials, uint numTetsPerElement,
                                 bool updatePosVel)
{
    const float invDtSq = invDt * invDt;
    const uint nbElements = nbTets / numTetsPerElement;

    const SbMaterial material = sbLoadMaterial(materials, sbReadU16(d.materialIdx, workIndex));

    uint4 pull0 = Ptr<uint4>(d.pullInds)[workIndex];
    const uint4 pull1 = Ptr<uint4>(d.pullInds)[workIndex + nbElements];
    const bool flipped = (pull0.x & FLIPPED_MASK) != 0u;
    pull0.x &= PULL_IND_MASK;

    // All 8 corners come from the UNORDERED buffers: the hex path has no CP copies.
    Ptr<float4> pos = Ptr<float4>(d.positions);
    Ptr<float4> vel = Ptr<float4>(d.velocities);
    SbHexPV h;
    h.p0 = pos[pull0.x]; h.p1 = pos[pull0.y]; h.p2 = pos[pull0.z]; h.p3 = pos[pull0.w];
    h.p4 = pos[pull1.x]; h.p5 = pos[pull1.y]; h.p6 = pos[pull1.z]; h.p7 = pos[pull1.w];
    h.v0 = vel[pull0.x]; h.v1 = vel[pull0.y]; h.v2 = vel[pull0.z]; h.v3 = vel[pull0.w];
    h.v4 = vel[pull1.x]; h.v5 = vel[pull1.y]; h.v6 = vel[pull1.z]; h.v7 = vel[pull1.w];

    uint offset = 0u;
    if (numTetsPerElement == 5u && flipped)
        offset = 20u;

    for (uint i = 0u; i < numTetsPerElement; ++i)
    {
        // ptr[4*i + k + offset] of the .cu, one ROW at a time.  offset is 0 whenever the
        // 6-tet table is selected, but it is applied to both, as in the .cu.
        const uint4 c = (numTetsPerElement == 5u) ? sbTets5PerVoxel((4u * i + offset) / 4u)
                                                  : sbTets6PerVoxel((4u * i + offset) / 4u);
        sbSolveSubTetrahedron(h, c, d, workIndex + nbElements * i, invDtSq, invDt, isTGS,
                              isFirstIteration, material);
    }

    if (updatePosVel) // GS-style update: positions and velocities in place
    {
        sbHexWriteGS(d, pull0.x, h.p0, h.v0);
        sbHexWriteGS(d, pull0.y, h.p1, h.v1);
        sbHexWriteGS(d, pull0.z, h.p2, h.v2);
        sbHexWriteGS(d, pull0.w, h.p3, h.v3);
        sbHexWriteGS(d, pull1.x, h.p4, h.v4);
        sbHexWriteGS(d, pull1.y, h.p5, h.v5);
        sbHexWriteGS(d, pull1.z, h.p6, h.v6);
        sbHexWriteGS(d, pull1.w, h.p7, h.v7);
    }
    else // Jacobi-style update: only mSimDelta; applyDeformationDeltas moves the vertices
    {
        sbHexWriteJacobi(d, pull0.x, h.p0);
        sbHexWriteJacobi(d, pull0.y, h.p1);
        sbHexWriteJacobi(d, pull0.z, h.p2);
        sbHexWriteJacobi(d, pull0.w, h.p3);
        sbHexWriteJacobi(d, pull1.x, h.p4);
        sbHexWriteJacobi(d, pull1.y, h.p5);
        sbHexWriteJacobi(d, pull1.z, h.p6);
        sbHexWriteJacobi(d, pull1.w, h.p7);
    }
}

// ---------------------------------------------------------------------------
// sb_gm_cp_solveTetrahedronsPartitionLaunch (softBodyGM.cu:1408)
// CUDA arg pack (PxgSoftBodyCore.cpp:1202): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxU32 numActiveSoftbodies, const PxReal invDt,
// const PxU32 partitionId, bool isTGS, bool isFirstIteration,
// PxsDeformableVolumeMaterialData* materials } -> the two bools land at @28/@29 and the
// trailing pointer realigns to @32 -> 40-byte blob.  (ABI row {8,8,4,4,4,1,1,8}.)
// ---------------------------------------------------------------------------
struct SolveTetraParams
{
    uint64_t gSoftbodies;          // @0
    uint64_t activeSoftbodies;     // @8
    uint numActiveSoftbodies;      // @16
    float invDt;                   // @20
    uint partitionId;              // @24
    uint flagsWord;                // @28  byte0 = isTGS, byte1 = isFirstIteration
    uint64_t materials;            // @32  -> 40
}
[[vk::push_constant]]
ConstantBuffer<SolveTetraParams> gSolve;

[shader("compute")]
// The host block is SB_SOLVETETRA_LOW = 32 (PxgSoftBodyCore.cpp:1192), not SB_SOLVETETRA:
// 64 is guarded over-coverage (the `workIndex < endInd` test), kept so the existing grids
// do not change.  grid (nbBlocks, nbActiveSoftbodies, 1)
[numthreads(64, 1, 1)]
void sb_gm_cp_solveTetrahedronsPartitionLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gSolve.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gSolve.activeSoftbodies))[groupId.y];

    // The @416 fold: byte1 = mBodyFlags.
    const uint bodyFlags = (gSoftbodies[softbodyId].mFlagsWord >> 8) & 0xffu;
    if ((bodyFlags & PXG_SOFTBODY_BODYFLAG_KINEMATIC) != 0u)
        return;

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 64u;

    const uint nbTets = gSoftbodies[softbodyId].mNumTetsGM;
    const uint numPartitions = gSoftbodies[softbodyId].mNumPartitionsGM;

    if (nbTets > 0u && gSolve.partitionId < numPartitions)
    {
        Ptr<uint> accumulatedPartitions = Ptr<uint>(gSoftbodies[softbodyId].mSimAccumulatedPartitionsCP);
        // CLAMP THE INDEX, NOT JUST THE VALUE (P8.96c): partitionId == 0 is
        // dispatched on every solve, and the speculated 0u-1u index leaves the
        // partition-count-sized buffer.  Element 0 is in bounds by the same
        // guard that admits this dispatch.
        const uint prevIdx = gSolve.partitionId > 0u ? gSolve.partitionId - 1u : 0u;
        const uint prevAccum = accumulatedPartitions[prevIdx];
        const uint startInd = gSolve.partitionId > 0u ? prevAccum : 0u;
        const uint endInd = accumulatedPartitions[gSolve.partitionId];

        const uint workIndex = startInd + groupThreadIdx;

        // The .cu's two-level dispatch (softBodyGM.cu:1442-1455): the tetrahedral body, or
        // the hexahedral (voxel) one -- the latter a silent no-op here until P9.95f.
        if (workIndex < endInd)
        {
            const bool isTGS = (gSolve.flagsWord & 0xffu) != 0u;
            const bool isFirstIteration = ((gSolve.flagsWord >> 8) & 0xffu) != 0u;
            if (gSoftbodies[softbodyId].mNumTetsPerElement == 1u)
            {
                const float invDt = gSolve.invDt;
                const float invDtSq = invDt * invDt;

                Ptr<float4> curPositions = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMassCP);
                Ptr<float4> velocities = Ptr<float4>(gSoftbodies[softbodyId].mSimVelocity_invMassCP);
                Ptr<uint4> writeIndices = Ptr<uint4>(gSoftbodies[softbodyId].mSimRemapOutputCP);
                Ptr<float4> unorderedPositions = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
                Ptr<float4> unorderedVelocities = Ptr<float4>(gSoftbodies[softbodyId].mSimVelocity_InvMass);
                Ptr<uint4> pullInds = Ptr<uint4>(gSoftbodies[softbodyId].mSimPullIndices);

                // mOrderedMaterialIndices is a PxU16 array (sbReadU16 reads the containing word).
                const uint globalMaterialIndex = sbReadU16(gSoftbodies[softbodyId].mOrderedMaterialIndices, workIndex);
                const SbMaterial material = sbLoadMaterial(gSolve.materials, globalMaterialIndex);

                // --- gather the four vertices ------------------------------------------
                uint4 pullInd = pullInds[workIndex];
                pullInd.x &= PULL_IND_MASK;

                // Unrolled for the same reason as the scatter: no dynamically-indexed local
                // array survives into the SPIR-V.
                float4 p0, p1, p2, p3, v0, v1, v2, v3;
                sbGather(unorderedPositions, unorderedVelocities, curPositions, velocities,
                         pullInd.x, workIndex + 0u * nbTets, p0, v0);
                sbGather(unorderedPositions, unorderedVelocities, curPositions, velocities,
                         pullInd.y, workIndex + 1u * nbTets, p1, v1);
                sbGather(unorderedPositions, unorderedVelocities, curPositions, velocities,
                         pullInd.z, workIndex + 2u * nbTets, p2, v2);
                sbGather(unorderedPositions, unorderedVelocities, curPositions, velocities,
                         pullInd.w, workIndex + 3u * nbTets, p3, v3);

                // --- rest pose + rotation refresh ---------------------------------------
                const uint64_t restBase = gSoftbodies[softbodyId].mSimTetraRestPoses +
                                          uint64_t(workIndex >> 5) * MAT33_BLOCK_STRIDE;
                const uint lane = workIndex & 31u;
                const float4 c0v = Ptr<float4>(restBase + uint64_t(lane) * 16UL)[0];
                const float4 c1v = Ptr<float4>(restBase + MAT33_BLOCK_COL1_OFFSET + uint64_t(lane) * 16UL)[0];
                const float c2v = Ptr<float>(restBase + MAT33_BLOCK_COL2_OFFSET + uint64_t(lane) * 4UL)[0];
                const SbMat33 Qinv = sbMat33(float3(c0v.x, c0v.y, c0v.z),
                                             float3(c0v.w, c1v.x, c1v.y),
                                             float3(c1v.z, c1v.w, c2v));

                Ptr<float4> tetRotations = Ptr<float4>(gSoftbodies[softbodyId].mSimTetraRotations);
                float4 q = tetRotations[workIndex];

                const float3 u1 = float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
                const float3 u2 = float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
                const float3 u3 = float3(p3.x - p0.x, p3.y - p0.y, p3.z - p0.z);

                {
                    if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) == 0.0f)
                        q = float4(gSoftbodies[softbodyId].mInitialRotationX,
                                   gSoftbodies[softbodyId].mInitialRotationY,
                                   gSoftbodies[softbodyId].mInitialRotationZ,
                                   gSoftbodies[softbodyId].mInitialRotationW);

                    // computeGMTetrahedronRotations (softBodyGM.cu:288): F = P*Qinv then
                    // extractRotation with 100 iterations on the first solver iteration, 4
                    // afterwards.  THIS is where the tranche-5 sin/cos seam enters the solve.
                    const SbMat33 Fm = sbMat33MulM(sbMat33(u1, u2, u3), Qinv);
                    q = sbExtractRotation(Fm.c0, Fm.c1, Fm.c2, q, isFirstIteration ? 100 : 4);

                    tetRotations[workIndex] = q;
                }

                // --- constitutive solve, guarded on a degenerate rest pose ---------------
                // softBodyGM.cu:1123 — the scatter below STILL RUNS with a zero delta when
                // the rest pose is degenerate; only the solve is skipped.
                SbMat33 dl = sbMat33Zero();
                if (!sbMat33IsZero(Qinv))
                {
                    SbTetV tv; tv.v0 = v0; tv.v1 = v1; tv.v2 = v2; tv.v3 = v3;
                    dl = sbTetrahedronsSolveGM(sbMat33(u1, u2, u3), tv,
                                               gSoftbodies[softbodyId].mSimTetraMultipliers, invDtSq, invDt,
                                               workIndex, Qinv, q, isTGS, material);
                }

                // --- scatter -------------------------------------------------------------
                // The scatter is written out per-vertex rather than as a 4-iteration loop over
                // a local float3[3].  That indexed temporary is a llvmpipe LANDMINE: with a
                // ZERO delta the array folds to a constant and the shader runs, but the moment
                // its three entries differ the driver materialises a dynamically-indexed local
                // array and SIGSEGVs in the JIT worker.  Bisected precisely — a CONSTANT
                // non-zero delta with no matrix work at all reproduces it, which is what ruled
                // out every arithmetic suspect.  Unrolling removes the dynamic index entirely.
                const uint4 writeIndex4 = writeIndices[workIndex];

                sbScatter(curPositions, velocities, writeIndex4.x, p0, v0,
                          -(dl.c0 + dl.c1 + dl.c2) * p0.w, invDt);
                sbScatter(curPositions, velocities, writeIndex4.y, p1, v1, dl.c0 * p1.w, invDt);
                sbScatter(curPositions, velocities, writeIndex4.z, p2, v2, dl.c1 * p2.w, invDt);
                sbScatter(curPositions, velocities, writeIndex4.w, p3, v3, dl.c2 * p3.w, invDt);
            }
            else
            {
                const uint safeW = endInd > 0u ? min(workIndex, endInd - 1u) : 0u;   // trap H4
                sbSolveHexahedronsPartition(sbHexDescOf(gSoftbodies, softbodyId), safeW, nbTets, gSolve.invDt,
                                            isTGS, isFirstIteration, gSolve.materials,
                                            gSoftbodies[softbodyId].mNumTetsPerElement, true);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch (softBodyGM.cu:1463) -- P9.95f
// CUDA arg pack (PxgSoftBodyCore.cpp:1238-1243): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxU32 numActiveSoftbodies (never read, upstream
// too), const PxReal invDt, bool isTGS, PxsDeformableVolumeMaterialData* materials }
// -> isTGS @24 (pad 25..27 and @28..31), the pointer realigns to @32 -> 40-byte blob.
// (ABI row {8,8,4,4,1,8}.)
// ---------------------------------------------------------------------------
struct SolveTetraJacobiParams
{
    uint64_t gSoftbodies;          // @0
    uint64_t activeSoftbodies;     // @8
    uint numActiveSoftbodies;      // @16  never read (dead upstream, push_block_known.txt)
    float invDt;                   // @20
    uint flagsWord;                // @24  byte0 = isTGS; bytes 1..3 are pad (P9.87)
    uint _pad28;                   // @28
    uint64_t materials;            // @32  -> 40
}
[[vk::push_constant]]
ConstantBuffer<SolveTetraJacobiParams> gJacobi;

[shader("compute")]
[numthreads(32, 1, 1)] // SB_SOLVETETRA_LOW; grid (ceil(maxJacobiTets/32), nbActiveSoftbodies, 1)
void sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gJacobi.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gJacobi.activeSoftbodies))[groupId.y];

    // softBodyGM.cu:1482 -- no extra Jacobi required (read from the GLOBAL descriptor there,
    // which is the only copy here)
    if (gSoftbodies[softbodyId].mNumPartitionsGM <= SB_PARTITION_LIMIT || gSoftbodies[softbodyId].mNumTetsPerElement == 1u)
        return;

    // softBodyGM.cu:1485 -- the @416 fold: byte1 = mBodyFlags
    const uint bodyFlags = (gSoftbodies[softbodyId].mFlagsWord >> 8) & 0xffu;
    if ((bodyFlags & PXG_SOFTBODY_BODYFLAG_KINEMATIC) != 0u)
        return;

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 32u;
    const uint nbTets = gSoftbodies[softbodyId].mNumTetsGM;

    Ptr<uint> accumulatedPartitions = Ptr<uint>(gSoftbodies[softbodyId].mSimAccumulatedPartitionsCP);
    const uint startInd = accumulatedPartitions[SB_PARTITION_LIMIT - 1u];
    const uint endInd = accumulatedPartitions[SB_PARTITION_LIMIT];

    const uint workIndex = startInd + groupThreadIdx;

    if (workIndex < endInd)
    {
        const uint safeW = min(workIndex, endInd - 1u);   // trap H4; endInd >= 1 here
        const bool isTGS = (gJacobi.flagsWord & 0xffu) != 0u;
        // isFirstIteration = false: this kernel always comes after the GS steps (softBodyGM.cu:1497-1503)
        sbSolveHexahedronsPartition(sbHexDescOf(gSoftbodies, softbodyId), safeW, nbTets, gJacobi.invDt, isTGS,
                                    false, gJacobi.materials, gSoftbodies[softbodyId].mNumTetsPerElement, false);
    }
}


// ===========================================================================
// P8.0 tranche 7 — the three kernels that CLOSE the softbody step: the partition
// average, the Jacobi deformation-delta apply, and the collision-mesh skinning.
//
//   sb_gm_cp_averageVertsLaunch        (:1509, launch PxgSoftBodyCore.cpp:1298)
//       the partition solve writes each vertex once PER COPY (one copy per partition
//       the vertex participates in) into the copy buffer mSimPosition_InvMassCP; this
//       kernel folds those copies back into the single canonical vertex by averaging
//       their displacements.  Without it the partitioned Gauss-Seidel solve produces
//       no visible motion at all — the copies are written and then dropped.
//   sb_gm_applyDeformationDeltasLaunch (:2665, launch PxgSoftBodyCore.cpp:1269)
//       the Jacobi twin of sb_gm_applyExternalDeltasLaunch (tranche 4).  Same
//       load/scale/store shape, but it walks the JACOBI vertex subset via
//       mSimJacobiVertIndices and uses the per-softbody precomputed mJacobiScale @428
//       instead of the per-vertex 1/max(delta.w,1).
//   sb_gm_updateTetModelVertsLaunch    (:1589, launch PxgSoftBodyCore.cpp:828)
//       the skinning step: the SIMULATION mesh is what the solver moves, but every
//       consumer downstream (narrowphase, the render readback, the user's
//       PxDeformableVolume vertex buffer) reads the COLLISION mesh.  This kernel
//       reconstructs each collision vertex as the barycentric combination of its
//       host sim tetrahedron's four corners.  It is the LAST kernel in the step and
//       the one that makes the motion observable outside the solver.
//
// Together with tranches 4-6 this completes the softbody solve loop end to end:
// zeroTetMultipliers opens it, the partitioned tetrahedron solve runs, averageVerts
// merges the copies, applyDeformationDeltas folds the Jacobi remainder, and
// updateTetModelVerts skins the result onto the collision mesh.
//
// DEVIATIONS from CUDA (identical in kind to the earlier softbody tranches):
//   - all three open with a `blockCopy` of the whole 464-byte PxgSoftBody descriptor
//     into __shared__ followed by __syncthreads, then read the shared copy.  That is
//     a pure read-caching optimisation; it is dropped and every lane reads the
//     descriptor directly through Ptr<>, which also removes the kernels' only
//     barrier (relevant here because averageVerts' early-out sits AFTER the
//     __syncthreads in the .cu — with no barrier the return is unconditionally safe);
//   - CUDA's `delta.x *= scale; delta.y *= scale, delta.z *= scale;` in
//     applyDeformationDeltas (note the comma operator — a typo that behaves
//     identically) is written as a float3 multiply, as in tranche 4.
//
// BLOCK DIMS — note the engine's own mismatch, faithfully mirrored:
// averageVerts and updateTetModelVerts compute `numBlocks` from SB_SOLVETETRA (64)
// but launch with SB_PREINTEGRATION (1024) threads per block, so the grid covers 16x
// the vertex count.  The `groupThreadIdx < numVerts` guard makes the surplus lanes
// no-ops; the port keeps 1024 so `threadIdx.x + blockIdx.x * blockDim.x` reproduces
// the same index for the same (block, thread) pair.  applyDeformationDeltas is
// consistent at 64.
//
// All three are NoContraction-stamped (softBodySolveBda is in PXG_NOCONTRACT_SHADERS):
// averageVerts' `vel + averageDelta * invDt`, applyDeformationDeltas' identical
// `vel + delta * invDt`, and above all updateTetModelVerts' four-term barycentric
// blend `p0*b.x + p1*b.y + p2*b.z + p3*b.w` are textbook mul-feeding-add chains that
// would otherwise fuse into FMAs and break parity with the -ffp-contract=off CPU
// reference.  NO transcendental in any of the three — this tranche is FULLY
// bit-exact, no tolerance seam.
// ===========================================================================

// ---------------------------------------------------------------------------
// sb_gm_cp_averageVertsLaunch (softBodyGM.cu:1509)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:1301): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxReal invDt } -> 20, rounded up to the
// pointer alignment -> 24-byte blob.  (ABI row {8,8,4}.)
struct AverageVertsParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8
    float invDt;                // @16
    uint _pad20;                // @20  -> 24
}
[[vk::push_constant]]
ConstantBuffer<AverageVertsParams> gAvg;

[shader("compute")]
[numthreads(1024, 1, 1)] // SB_PREINTEGRATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_cp_averageVertsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gAvg.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gAvg.activeSoftbodies))[groupId.y];

    const uint bodyFlags = (gSoftbodies[softbodyId].mFlagsWord >> 8) & 0xffu;

    // A kinematic body is driven by its targets, not by the solve; a body with more
    // than one tet per element is a HEXAHEDRAL element whose copies are merged by the
    // hexahedron path instead.  Both leave the copy buffer untouched.
    if ((bodyFlags & PXG_SOFTBODY_BODYFLAG_KINEMATIC) != 0u ||
        gSoftbodies[softbodyId].mNumTetsPerElement != 1u)
        return;

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 1024u;

    const uint numVerts = gSoftbodies[softbodyId].mNumVertsGM;
    const uint numTets = gSoftbodies[softbodyId].mNumTetsGM;

    if (groupThreadIdx < numVerts && numTets > 0u)
    {
        const uint numTetsPerElement = gSoftbodies[softbodyId].mNumTetsPerElement;
        const uint numElements = numTets / numTetsPerElement;
        const uint numVertsPerElement = (numTetsPerElement == 1u) ? 4u : 8u;

        Ptr<float4> gridModelPoses = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
        Ptr<float4> gridModelVels = Ptr<float4>(gSoftbodies[softbodyId].mSimVelocity_InvMass);
        Ptr<float4> deltaPos = Ptr<float4>(gSoftbodies[softbodyId].mSimDeltaPos);

        Ptr<float4> curPositions = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMassCP);
        Ptr<uint> accumulatedCopies = Ptr<uint>(gSoftbodies[softbodyId].mSimAccumulatedCopiesCP);

        // The copy buffer holds the PARTITIONED tetrahedron corners first (one entry per
        // element corner), and the per-vertex copies only start after them.
        const uint offset = numElements * numVertsPerElement;

        const float4 pos_invMass = gridModelPoses[groupThreadIdx];

        if (pos_invMass.w > 0.0f)
        {
            // mSimAccumulatedCopiesCP is an inclusive prefix sum of the per-vertex copy
            // counts, so vertex i owns copies [accum[i-1], accum[i]).
            const uint startInd = (groupThreadIdx == 0u) ? 0u : accumulatedCopies[groupThreadIdx - 1u];
            const uint endInd = accumulatedCopies[groupThreadIdx];

            float4 diff = curPositions[offset + startInd] - pos_invMass;

            for (uint j = startInd + 1u; j < endInd; ++j)
            {
                const float4 newDiff = curPositions[offset + j] - pos_invMass;
                diff += newDiff;
            }

            const uint numCopies = endInd - startInd;
            const float scale = 1.0f / float(numCopies);

            diff.w = 0.0f;

            const float4 averageDelta = diff * scale;
            const float4 averagePos = pos_invMass + averageDelta;
            const float4 averageVel = gridModelVels[groupThreadIdx] + averageDelta * gAvg.invDt;

            gridModelPoses[groupThreadIdx] = averagePos;
            // NOTE the w channel: the velocity slot carries the vertex's INVERSE MASS,
            // not a velocity component, so it is restored from pos_invMass rather than
            // taken from averageVel.
            gridModelVels[groupThreadIdx] =
                float4(averageVel.x, averageVel.y, averageVel.z, pos_invMass.w);
            deltaPos[groupThreadIdx] = deltaPos[groupThreadIdx] + averageDelta;
        }
    }
}

// ---------------------------------------------------------------------------
// sb_gm_applyDeformationDeltasLaunch (softBodyGM.cu:2665)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:1270): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxReal invDt } -> 24-byte blob.
// (ABI row {8,8,4}.)
struct ApplyDeformationDeltasParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8
    float invDt;                // @16
    uint _pad20;                // @20  -> 24
}
[[vk::push_constant]]
ConstantBuffer<ApplyDeformationDeltasParams> gDeform;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_applyDeformationDeltasLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gDeform.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gDeform.activeSoftbodies))[groupId.y];

    Ptr<float4> curPositions = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
    Ptr<float4> vels = Ptr<float4>(gSoftbodies[softbodyId].mSimVelocity_InvMass);
    Ptr<float4> simDelta = Ptr<float4>(gSoftbodies[softbodyId].mSimDelta);
    Ptr<float4> simDeltaPos = Ptr<float4>(gSoftbodies[softbodyId].mSimDeltaPos);
    Ptr<uint> jacobiVertIndices = Ptr<uint>(gSoftbodies[softbodyId].mSimJacobiVertIndices);

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 64u;
    const uint nbJacobiVerts = gSoftbodies[softbodyId].mNumJacobiVertices;

    if (groupThreadIdx < nbJacobiVerts)
    {
        // The Jacobi subset is an INDIRECTION: lane i works on sim vertex
        // mSimJacobiVertIndices[i], not on vertex i.
        const uint vertIdx = jacobiVertIndices[groupThreadIdx];

        const float4 pos = curPositions[vertIdx];
        const float4 vel = vels[vertIdx];

        float4 delta = simDelta[vertIdx];

        // The ONLY difference from sb_gm_applyExternalDeltasLaunch: a per-softbody
        // precomputed scale (mJacobiScale @428) instead of the per-vertex
        // 1/max(delta.w, 1).  delta.w is ignored entirely here.
        const float scale = gSoftbodies[softbodyId].mJacobiScale;

        delta.x *= scale; delta.y *= scale; delta.z *= scale;
        delta.w = 0.0f;

        const float4 newVel = vel + delta * gDeform.invDt;

        // P9.86: as in sb_gm_applyExternalDeltasLaunch -- the CUDA's w lanes are IEEE
        // `pos.w + 0.f` / `deltaPos.w + 0.f` (softBodyGM.cu:2707/2710), -0 -> +0; llvmpipe
        // folds the literal-zero w lane, so the w lane's `0 + x` is a select.
        float4 newPos = pos + delta;
        newPos.w = (newPos.w == 0.0f) ? 0.0f : newPos.w;
        curPositions[vertIdx] = newPos;
        vels[vertIdx] = newVel;

        float4 newDeltaPos = simDeltaPos[vertIdx] + delta;
        newDeltaPos.w = (newDeltaPos.w == 0.0f) ? 0.0f : newDeltaPos.w;
        simDeltaPos[vertIdx] = newDeltaPos;

        // clear delta
        simDelta[vertIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// sb_gm_updateTetModelVertsLaunch (softBodyGM.cu:1589)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:830): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies } -> 16-byte blob.  (ABI row {8,8}.)
struct UpdateTetModelVertsParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8  -> 16
}
[[vk::push_constant]]
ConstantBuffer<UpdateTetModelVertsParams> gSkin;

[shader("compute")]
[numthreads(1024, 1, 1)] // SB_PREINTEGRATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_updateTetModelVertsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gSkin.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gSkin.activeSoftbodies))[groupId.y];

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 1024u;

    // mNumVerts @388 is the COLLISION mesh vertex count (mNumVertsGM @396 is the
    // simulation mesh's) — this kernel is one thread per collision vertex.
    const uint numVerts = gSoftbodies[softbodyId].mNumVerts;

    if (groupThreadIdx < numVerts)
    {
        Ptr<float4> barycentricCoordinates =
            Ptr<float4>(gSoftbodies[softbodyId].mVertsBarycentricInGridModel);
        Ptr<uint> remapTable = Ptr<uint>(gSoftbodies[softbodyId].mVertsRemapInGridModel);
        Ptr<uint4> tetsGM = Ptr<uint4>(gSoftbodies[softbodyId].mSimTetIndices);
        Ptr<float4> positionGM = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);

        Ptr<float4> curPositions = Ptr<float4>(gSoftbodies[softbodyId].mPosition_InvMass);

        const uint tetrahedronIdx = remapTable[groupThreadIdx];
        const uint4 tetInd = tetsGM[tetrahedronIdx];
        const float4 p0 = positionGM[tetInd.x];
        const float4 p1 = positionGM[tetInd.y];
        const float4 p2 = positionGM[tetInd.z];
        const float4 p3 = positionGM[tetInd.w];

        const float4 barycentric = barycentricCoordinates[groupThreadIdx];

        // The w channel rides along: the sim positions carry inverse mass there, so the
        // collision vertex ends up with the barycentric blend of the four inverse
        // masses.  The .cu notes the alternative (`newPos.w = pos.w`) and leaves it
        // commented out — curPositions.w is re-established downstream by the collision
        // pass, so the blended value is never consumed.
        const float4 newPos = p0 * barycentric.x + p1 * barycentric.y +
                              p2 * barycentric.z + p3 * barycentric.w;

        curPositions[groupThreadIdx] = newPos;
    }
}

// ===========================================================================
// P8.5 — sb_calculateStressLaunch (softBodyGM.cu:2716, launch
// PxgSoftBodyCore::calculateStress, PxgSoftBodyCore.cpp:2996, SB_CALC_STRESS).
//
// One thread per COLLISION-mesh tetrahedron computes the co-rotated Cauchy stress tensor and
// the scalar stress coefficient that goes with it, writing both back into the descriptor
// (mTetraStresses and mTetraStressCoefficient).
//
// WHY IT MATTERS HERE.  The scalar coefficient is the input to the SELF-COLLISION stress
// gate: SbMeshTreeTraverser drops a self-contact whose tetrahedron carries more stress than
// mSelfCollisionStressTolerance (softbodySoftbodyMidPhase.cu:441, ported in P8.4 tranche a).
// Until this kernel was mapped, mTetraStressCoefficient held whatever the allocation
// happened to contain, which is why PXG_SCENE=softbodyselfcollide has to set the tolerance
// to a value no stress can exceed.  That workaround can now be removed -- and the kernel's
// own launch gate says as much: it runs when eCOMPUTE_STRESS_TENSOR is set OR when
// self-collision is enabled, i.e. the stress field exists FOR the self-collision gate.
//
// *** THE GRID. ***
// grid (ceil(maxTetrahedrons / 256), nbActiveSoftbodies, 1), block (256, 1, 1).  The y-grid
// selects the active body, exactly as in the self-collision midphase, and the bridge
// forwards it.  blockDim is dropped, so [numthreads(256,1,1)] reproduces
// PxgSoftBodyKernelBlockDim::SB_UPDATEROTATION by construction and the x index is
// gtid.x + gid.x * 256 with no stride loop -- the engine sizes the grid to the mesh.
//
// DEVIATIONS from CUDA:
//   - the whole PxgSoftBody is blockCopy'd into shared memory and read from there.  That is
//     a caching optimisation, not semantics; every field is read directly through Ptr<>
//     here, as everywhere else in this port, and the __syncthreads it needs goes with it;
//   - `const PxQuat initialRotation = shSoftbody.mInitialRotation;` is loaded by the CUDA
//     and NEVER USED.  It is not reproduced -- a dead load is not ABI;
//   - mMaterialIndices is a PxU16 array; llvmpipe has no 16-bit storage in these shaders, so
//     the containing 32-bit word is read and the half extracted.
//
// NoContraction-stamped: the deformation gradient is two 3x3 products and the stress is a
// symmetrised difference, all mul-feeding-add chains.  The one non-portable operation is the
// final sqrt, which produces the scalar coefficient only.
// ===========================================================================

// PxsDeformableVolumeMaterialData (PxsDeformableVolumeMaterialCore.h:50), 32 B:
// youngs @0, poissons @4, dynamicFriction @8, ...
static const uint64_t STRESS_MAT_STRIDE = 32UL;

// PxDeformableVolumeFlag::eCOMPUTE_STRESS_TENSOR is bit 0 of mVolumeFlags (bytes 2..3 of
// mFlagsWord); eDISABLE_SELF_COLLISION is bit 0 of mBodyFlags (byte 1).
struct StressParams
{
    uint64_t softbodies;         // @0
    uint64_t activeSoftbodies;   // @8
    uint64_t materials;          // @16 -> 24
}
[[vk::push_constant]] ConstantBuffer<StressParams> gSt;

// A column-major PxMat33 as three columns, matching the 36-byte flat layout on the device.
struct Mat33 { float3 c0, c1, c2; }

Mat33 mat33Load(uint64_t base)
{
    Mat33 m;
    m.c0 = float3(Ptr<float>(base)[0], Ptr<float>(base)[1], Ptr<float>(base)[2]);
    m.c1 = float3(Ptr<float>(base)[3], Ptr<float>(base)[4], Ptr<float>(base)[5]);
    m.c2 = float3(Ptr<float>(base)[6], Ptr<float>(base)[7], Ptr<float>(base)[8]);
    return m;
}

void mat33Store(uint64_t base, Mat33 m)
{
    Ptr<float>(base)[0] = m.c0.x; Ptr<float>(base)[1] = m.c0.y; Ptr<float>(base)[2] = m.c0.z;
    Ptr<float>(base)[3] = m.c1.x; Ptr<float>(base)[4] = m.c1.y; Ptr<float>(base)[5] = m.c1.z;
    Ptr<float>(base)[6] = m.c2.x; Ptr<float>(base)[7] = m.c2.y; Ptr<float>(base)[8] = m.c2.z;
}

// PxMat33::transform (PxMat33.h:278): column0 * v.x + column1 * v.y + column2 * v.z.
float3 mat33Transform(Mat33 m, float3 v)
{
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z;
}

// PxMat33::operator*(PxMat33) (PxMat33.h:286): each result column is this->transform of the
// corresponding other column.
Mat33 mat33Mul(Mat33 a, Mat33 b)
{
    Mat33 r;
    r.c0 = mat33Transform(a, b.c0);
    r.c1 = mat33Transform(a, b.c1);
    r.c2 = mat33Transform(a, b.c2);
    return r;
}

Mat33 mat33Transpose(Mat33 m)
{
    Mat33 r;
    r.c0 = float3(m.c0.x, m.c1.x, m.c2.x);
    r.c1 = float3(m.c0.y, m.c1.y, m.c2.y);
    r.c2 = float3(m.c0.z, m.c1.z, m.c2.z);
    return r;
}

// PxMat33(PxQuat) (PxMat33.h:138), transcribed operation for operation.
Mat33 mat33FromQuat(float4 q)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    Mat33 m;
    m.c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    m.c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    m.c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
    return m;
}

// mMaterialIndices is a PxU16 array.  NOTE `& ~uint64_t(3)`: `& ~3UL` MISCOMPILES -- Slang
// evaluates the complement in 32 bits and clears the top half of the device address.
uint stressMaterialIndexAt(uint64_t base, uint tetIdx)
{
    const uint64_t addr = base + uint64_t(tetIdx) * 2UL;
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    return (addr & 2UL) != 0UL ? (word >> 16u) : (word & 0xffffu);
}

[shader("compute")]
[numthreads(256, 1, 1)]   // PxgSoftBodyKernelBlockDim::SB_UPDATEROTATION
void sb_calculateStressLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint softbodyId = Ptr<uint>(gSt.activeSoftbodies)[gid.y];
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSt.softbodies);

    const uint flagsWord = softbodies[softbodyId].mFlagsWord;
    const uint bodyFlags = (flagsWord >> 8u) & 0xFFu;
    const uint volumeFlags = (flagsWord >> 16u) & 0xFFFFu;

    // The stress field is computed when it is ASKED FOR, or when self-collision needs it.
    if (!((volumeFlags & PXG_SOFTBODY_VOLUMEFLAG_COMPUTE_STRESS_TENSOR) != 0u ||
          (bodyFlags & PXG_SOFTBODY_BODYFLAG_DISABLE_SELF_COLLISION) == 0u))
        return;

    const uint tetrahedronIdx = gtid.x + gid.x * 256u;
    if (tetrahedronIdx >= softbodies[softbodyId].mNumTets)
        return;

    Ptr<float4> positions = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    const uint4 tetIdx = Ptr<uint4>(softbodies[softbodyId].mTetIndices)[tetrahedronIdx];

    const float4 p0 = positions[tetIdx.x];
    const float4 p1 = positions[tetIdx.y];
    const float4 p2 = positions[tetIdx.z];
    const float4 p3 = positions[tetIdx.w];

    // the displacement field
    const float3 u1 = p1.xyz - p0.xyz;
    const float3 u2 = p2.xyz - p0.xyz;
    const float3 u3 = p3.xyz - p0.xyz;

    // mTetraRestPoses is a FLAT PxMat33 array here (36-byte stride), not a PxgMat33Block --
    // the same distinction sb_updateTetrahedraRotationsLaunch above turns on.
    const Mat33 qinv = mat33Load(softbodies[softbodyId].mTetraRestPoses + uint64_t(tetrahedronIdx) * 36UL);

    Mat33 pMat;
    pMat.c0 = u1; pMat.c1 = u2; pMat.c2 = u3;
    Mat33 f = mat33Mul(pMat, qinv);

    // the rotation factor, refreshed once per step by sb_updateTetrahedraRotationsLaunch
    const float4 tetraRot = Ptr<float4>(softbodies[softbodyId].mTetraRotations)[tetrahedronIdx];
    const Mat33 r = mat33FromQuat(tetraRot);

    // remove the rotation factor from the strain (corotation)
    f = mat33Mul(mat33Transpose(r), f);

    const uint matIdx = stressMaterialIndexAt(softbodies[softbodyId].mMaterialIndices, tetrahedronIdx);
    const uint64_t matBase = gSt.materials + uint64_t(matIdx) * STRESS_MAT_STRIDE;
    const float E = Ptr<float>(matBase)[0];        // youngs
    const float nu = Ptr<float>(matBase + 4UL)[0]; // poissons

    // Young's modulus -> Lame parameters, then Pixar's Stable Neo-Hookean reparameterisation
    // for rest stability.  Transcribed in the .cu's order: the two `+=`/`*=` steps come
    // AFTER both are formed from E and nu, so mu's update must not feed lambda's.
    float lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    float mu = 0.5f * E / (1.0f + nu);
    lambda += mu * (5.0f / 6.0f);
    mu = mu * (4.0f / 3.0f);

    // the Cauchy stress: eps = sym(F) - I, sigma = 2*mu*eps + lambda*tr(eps)*I
    const Mat33 fT = mat33Transpose(f);
    Mat33 eps;
    eps.c0 = (fT.c0 + f.c0) * 0.5f - float3(1.0f, 0.0f, 0.0f);
    eps.c1 = (fT.c1 + f.c1) * 0.5f - float3(0.0f, 1.0f, 0.0f);
    eps.c2 = (fT.c2 + f.c2) * 0.5f - float3(0.0f, 0.0f, 1.0f);

    const float trace = eps.c0.x + eps.c1.y + eps.c2.z;
    Mat33 sigma;
    sigma.c0 = eps.c0 * 2.0f * mu + float3(1.0f, 0.0f, 0.0f) * lambda * trace;
    sigma.c1 = eps.c1 * 2.0f * mu + float3(0.0f, 1.0f, 0.0f) * lambda * trace;
    sigma.c2 = eps.c2 * 2.0f * mu + float3(0.0f, 0.0f, 1.0f) * lambda * trace;

    // The scalar coefficient is the von-Mises-like norm of the DIAGONAL strain differences.
    // eps[i][j] is column i, element j, so eps[0][0]/eps[1][1]/eps[2][2] are the diagonal.
    const float q1 = eps.c0.x, q2 = eps.c1.y, q3 = eps.c2.z;
    const float q12 = q1 - q2, q23 = q2 - q3, q31 = q3 - q1;
    const float constant = 0.70710678118654752440084436210485f;   // 1 / sqrt(2)
    const float stress = sqrt(q12 * q12 + q23 * q23 + q31 * q31) * constant;

    mat33Store(softbodies[softbodyId].mTetraStresses + uint64_t(tetrahedronIdx) * 36UL, sigma);
    Ptr<float>(softbodies[softbodyId].mTetraStressCoefficient)[tetrahedronIdx] = stress;
}
