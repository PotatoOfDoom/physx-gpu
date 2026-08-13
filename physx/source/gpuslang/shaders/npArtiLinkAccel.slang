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

// npArtiLinkAccel.slang
//
// P8.92a: the last two pieces of `computeUnconstrainedVelocities`' fifth kernel
// that can stand on their own —
//
//   * `computeJointTransmittedFrictionForce` (forwardDynamic2.cu:2317), the
//     leaf-to-root bias-force sweep the acceleration pass ends with;
//   * `asmTranslateInertiaMass` (forwardDynamic2.cu:1569), the parallel-axis
//     theorem `computeLinkAcceleration`'s compound inertia is built out of,
//     which lives in `pxgSlangSpatialMatrix.slang` beside the rest of the algebra.
//
// `computeLinkAcceleration` itself is P8.92b and the wrapper is P8.92c.  **Zero
// new ABI for the fifth tranche running**: `mBiasForce` (20224),
// `mMotionAcceleration` (18688), `mMotionAccelerationInternal` (19456),
// `mInvInertiaXYZ_invMassW` (27648), `mAccumulatedPose` (23552), `mParents`
// (28672) and `mRw_{x,y,z}` (21120/21248/21376) were all mapped in P8.68.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// Everything the body is built from is PX_CUDA_CALLABLE — `PxQuat::rotate` and
// `rotateInv`, `PxVec3::multiply`, `FeatherstoneArticulation::
// translateSpatialVector`, `loadSpatialVector` / `storeSpatialVector`,
// `PxMat33::getTranspose` and `operator*` — so the fixture computes the whole
// expectation with the ENGINE's own arithmetic on the host and compares bit
// patterns.  Only the loop's STRUCTURE is transcribed, which is exactly where its
// traps live.
//
// ===========================================================================
// THE SEVEN TRAPS, EACH OF WHICH GETS A MUTANT
// ===========================================================================
//
//  1. **The loop is leaf-to-root and strictly sequential** —
//     `for (linkID = nbLinks - 1; linkID > 0; --linkID)` — because every
//     iteration accumulates into its PARENT, which a later iteration then reads
//     as its own `Ia`.  Reversing it, or bounding it at 0, is wrong.
//  2. **The parent's bias force is LOADED AT THE TOP**, before this link's own
//     `Ia` is stored — the P8.88 read-modify-write shape.  With a strictly
//     sequential loop that is the same thing, but a port that re-loads at the
//     store is only accidentally right.
//  3. **The inertia is the per-component RECIPROCAL of
//     `mInvInertiaXYZ_invMassW`, each with its own zero guard**, and so is the
//     mass.  A static link has `invMass == 0`, so an unguarded divide gives `inf`
//     on exactly the case the scene contains.
//  4. **The dual basis, spelled out.**  `Ia.bottom` (the TORQUE half of a FORCE)
//     picks up the inertia applied to `motionAcceleration.top` (the ANGULAR half
//     of a MOTION), and `Ia.top` (the LINEAR FORCE half) picks up
//     `motionAcceleration.bottom * m`.  Both halves cross over; a port that keeps
//     them parallel produces plausible numbers.
//  5. **The inertia is applied IN THE BODY FRAME**:
//     `rot.rotate(rot.rotateInv(a.top).multiply(inertia))`.  Dropping either
//     rotation, or swapping them, is a rotation of the result rather than an
//     obvious error.
//  6. **`motionAcceleration` is the SUM** of `mMotionAcceleration` and
//     `mMotionAccelerationInternal`.  The two are separate fields precisely
//     because the acceleration pass computes them separately.
//  7. **The root's bias force is ZEROED after the loop**, which discards
//     everything the loop accumulated into link 0.  A port that skips it leaves
//     the accumulated value, which is a plausible-looking wrong answer.
//
// And the sign: `translateSpatialVector(+rw, Ia)` here, NOT negated — the
// opposite of the acceleration direction's `-parentToChild`.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

// The per-component reciprocal with its own zero guard -- trap 3.
float alaRecip(float v) { return v == 0.0f ? 0.0f : 1.0f / v; }

// computeJointTransmittedFrictionForce (forwardDynamic2.cu:2317).
public void alaJointTransmittedFrictionForce(uint64_t linkBlocks, uint numLinks, uint lane)
{
    // leaf to root, strictly sequential -- trap 1
    for (uint linkID = numLinks - 1u; linkID > 0u; --linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const uint parent = abU32(link, PXG_AB_LINK_mParents, lane);
        const uint64_t parentLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;

        // loaded BEFORE this link's own Ia is stored -- trap 2
        float3 bfTop, bfBottom;
        abLoadSpatialVector(parentLink, PXG_AB_LINK_mBiasForce, lane, bfTop, bfBottom);
        const AbSpatialVector biasForce = asmSVec(bfTop, bfBottom);

        const float3 rw = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                 abF32(link, PXG_AB_LINK_mRw_y, lane),
                                 abF32(link, PXG_AB_LINK_mRw_z, lane));

        const float4 invInertiaXYZ_invMassW = abF4(link, PXG_AB_LINK_mInvInertiaXYZ_invMassW, lane);
        const float4 rot = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);

        const float m = alaRecip(invInertiaXYZ_invMassW.w);
        const float3 inertia = float3(alaRecip(invInertiaXYZ_invMassW.x),
                                      alaRecip(invInertiaXYZ_invMassW.y),
                                      alaRecip(invInertiaXYZ_invMassW.z));

        // the SUM of the two acceleration fields -- trap 6
        float3 aTop, aBottom, iTop, iBottom;
        abLoadSpatialVector(link, PXG_AB_LINK_mMotionAcceleration, lane, aTop, aBottom);
        abLoadSpatialVector(link, PXG_AB_LINK_mMotionAccelerationInternal, lane, iTop, iBottom);
        const AbSpatialVector motionAcceleration =
            asmSVecAdd(asmSVec(aTop, aBottom), asmSVec(iTop, iBottom));

        float3 iaTop, iaBottom;
        abLoadSpatialVector(link, PXG_AB_LINK_mBiasForce, lane, iaTop, iaBottom);

        // the dual basis and the body-frame inertia -- traps 4 and 5.
        // `PxVec3::multiply` is the COMPONENTWISE product, not a dot.
        iaBottom = iaBottom
                 + quatRotate(rot, quatRotateInv(rot, motionAcceleration.top) * inertia);
        iaTop = iaTop + motionAcceleration.bottom * m;

        abStoreSpatialVector(link, PXG_AB_LINK_mBiasForce, lane, iaTop, iaBottom);

        // NOT negated -- into the parent's frame
        const AbSpatialVector translated =
            adTranslateSpatialVector(rw, asmSVec(iaTop, iaBottom));
        const AbSpatialVector parentBias = asmSVecAdd(biasForce, translated);
        abStoreSpatialVector(parentLink, PXG_AB_LINK_mBiasForce, lane, parentBias.top,
                             parentBias.bottom);
    }

    // the root's accumulation is DISCARDED -- trap 7
    abStoreSpatialVector(linkBlocks, PXG_AB_LINK_mBiasForce, lane, float3(0.0f, 0.0f, 0.0f),
                         float3(0.0f, 0.0f, 0.0f));
}

// ===========================================================================
// The fixture entry points
// ===========================================================================

struct AlaFrictionParams
{
    uint64_t linkBlocks;   // @0
    uint64_t numLinksPer;  // @8   uint32[nbLanes] -- a DIFFERENT count per lane
    uint nbLanes;          // @16
    uint pad0;             // @20 -> 24, padded to 32 by the shim's alignment
    uint pad1;             // @24
    uint pad2;             // @28
};

[[vk::push_constant]]
ConstantBuffer<AlaFrictionParams> gAlaFriction;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJointFriction(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gAlaFriction.nbLanes)
        return;

    const uint numLinks = (Ptr<uint>(gAlaFriction.numLinksPer))[lane];
    alaJointTransmittedFrictionForce(gAlaFriction.linkBlocks, numLinks, lane);
}

// The parallel-axis theorem, dumped so `asmTranslateInertiaMass` has a fixture of
// its own rather than being checked only through the compound inertia it feeds.
struct AlaInertiaParams
{
    uint64_t inFloats;    // @0   float[nbCases * 13]  -- 9 inertia + 1 mass + 3 offset
    uint64_t outFloats;   // @8   float[nbCases * 9]
    uint nbCases;         // @16
    uint pad0;            // @20
    uint pad1;            // @24
    uint pad2;            // @28
};

[[vk::push_constant]]
ConstantBuffer<AlaInertiaParams> gAlaInertia;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiTranslateInertiaMass(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint tid = dispatchThreadId.x;
    if (tid >= gAlaInertia.nbCases)
        return;

    Ptr<float> i = Ptr<float>(gAlaInertia.inFloats + uint64_t(tid) * 13u * 4u);
    Ptr<float> o = Ptr<float>(gAlaInertia.outFloats + uint64_t(tid) * 9u * 4u);

    const AbMat33 inertia = asmM33(float3(i[0], i[1], i[2]), float3(i[3], i[4], i[5]),
                                   float3(i[6], i[7], i[8]));
    const AbMat33 r = asmTranslateInertiaMass(inertia, i[9], float3(i[10], i[11], i[12]));

    o[0] = r.c0.x; o[1] = r.c0.y; o[2] = r.c0.z;
    o[3] = r.c1.x; o[4] = r.c1.y; o[5] = r.c1.z;
    o[6] = r.c2.x; o[7] = r.c2.y; o[8] = r.c2.z;
}

// ===========================================================================
// P8.92b: computeLinkAcceleration (forwardDynamic2.cu:1931-2316)
// ===========================================================================
//
// The root-to-leaf acceleration sweep, and the thing the articulation scene has
// been waiting for since P8.65: this is what turns the spatial inertia and the
// response matrices into an acceleration that actually moves the link.
//
// It is FIVE sections, and the whole difficulty is that four of them look alike:
//
//   1. the root's EXTERNAL acceleration, from `mZAVector`;
//   2. pass one, root to leaf, propagating that acceleration down;
//   3. the root's INTERNAL acceleration, from `mZAIntVector`, plus the start of
//      the momentum bookkeeping;
//   4. pass two, root to leaf again, propagating the INTERNAL acceleration and
//      the Coriolis term;
//   5. the momentum-conservation tail, free base only.
//
// Zero new ABI, still.
//
// ===========================================================================
// PASS ONE AND PASS TWO ARE NOT THE SAME LOOP, IN SIX PLACES
// ===========================================================================
//
// Each of these gets its own mutant, because a port that shares one body between
// the two passes is wrong in exactly these six ways and in no other:
//
//   a. pass one reads the parent's `mMotionAcceleration`, pass two the parent's
//      `mMotionAccelerationInternal`;
//   b. pass one's per-dof term is `mQstZ`, pass two's is `mQstZIcInternal`;
//   c. pass one seeds `motionAcceleration` from the propagated parent value
//      alone; pass two ADDS `mCoriolis`;
//   d. pass one ASSIGNS `mJointAccel`, pass two ADDS to it;
//   e. pass one stores `mMotionAcceleration`, pass two stores
//      `mMotionAccelerationInternal`;
//   f. pass one takes its starting velocity from a load at the TOP of the
//      iteration; pass two re-loads `mMotionVelocity`, which pass one has since
//      written, and uses the PRE-update value for `preMomentum.bottom` before
//      adding `motionAcceleration * dt` to it.
//
// And one they DO share, which is a trap of its own: **`jAccel` is taken against
// the PROPAGATED parent acceleration, not against the running `motionAcceleration`
// the Coriolis term has already been added to.**  In pass two those two differ,
// so a port that uses the accumulator produces plausible numbers.
//
// ===========================================================================
// THE MOMENTUM TAIL
// ===========================================================================
//
// Free base only, three more loops over every link:
//
//   * `postRootVel` comes from `postMomentum.top` as it stands after pass two,
//     and the loop that follows MUTATES `postMomentum.top` while iterating — so
//     the two cannot be fused;
//   * `angRatio = denom == 0 ? 1 : num / denom`, a guarded ratio of two
//     MAGNITUDES (square roots), not of their squares;
//   * `deltaAng` goes through the general inverse of the compound inertia;
//   * every link's velocity picks up `(deltaAng, offset x deltaAng)` and then, in
//     a third loop, a flat `deltaLin` on the bottom half only.
//
// Two locals the CUDA computes and never uses -- `angVel` and `linVel` -- are
// dropped, and this note is the record of that rather than a transcription of a
// value nobody reads.  There is a THIRD dead thing, and unlike those two it is a
// STORE rather than a local: **pass one's `motionVelocities[linkID]` write is
// always overwritten by pass two**, which loops over exactly the same range.  It
// is transcribed anyway, and the sweep records its removal as a proven
// equivalence rather than a gap -- because the day pass two stops covering the
// same links, the engine's version is the one that stays right.
//
// One more measured equivalence, for the same reason: pass one reads its starting
// velocity BEFORE calling the propagation, and moving that read after it changes
// nothing, because the propagation writes dof fields and the link's acceleration,
// never its velocity.  Reading the WRONG link's velocity is a different matter
// and dies.  So is the "preload next link data" pair at the top of pass
// one: it is a latency optimisation for a real warp, it changes no result, and on
// llvmpipe it would only add a redundant load.
//
// ===========================================================================
// P8.40, PRE-EMPTED
// ===========================================================================
//
// `motionVelocities` is a pointer loaded out of the articulation block and then
// indexed inside three separate loops.  That is exactly the shape that faulted in
// P8.91c.  The load happens ONCE, at the top of the function, before any loop --
// which is also what the CUDA does -- and the loops do address arithmetic on the
// value rather than re-reading it.  The note is here so nobody "tidies" the load
// into the loop that uses it.

// Cm::UnAlignedSpatialVector is six tightly packed floats -- no padding, unlike
// its aligned twin.  This is the flat array the constraint prep reads, which does
// not understand block data.
void alaStoreFlatVel(uint64_t base, uint index, AbSpatialVector v)
{
    Ptr<float> p = Ptr<float>(base + uint64_t(index) * 24u);
    p[0] = v.top.x;
    p[1] = v.top.y;
    p[2] = v.top.z;
    p[3] = v.bottom.x;
    p[4] = v.bottom.y;
    p[5] = v.bottom.z;
}

AbSpatialVector alaLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 t, b;
    abLoadSpatialVector(base, fieldOffset, lane, t, b);
    return asmSVec(t, b);
}

void alaStoreSV(uint64_t base, uint fieldOffset, uint lane, AbSpatialVector v)
{
    abStoreSpatialVector(base, fieldOffset, lane, v.top, v.bottom);
}

AbSpatialVector alaNeg(AbSpatialVector v) { return asmSVec(-v.top, -v.bottom); }

// PxVec3::magnitude() == PxSqrt(magnitudeSquared()), spelled out rather than
// reached for via `length()`, for the reason P8.69 gives.
float alaMagnitude(float3 v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

// One iteration of either down-sweep.  `internal` selects between the six
// differences the header lists; everything else is shared on purpose, because the
// two CUDA loops ARE the same arithmetic over different fields.
void alaPropagateLink(uint64_t linkBlocks, uint64_t dofBlocks, uint linkID, float dt, bool internal,
                      uint lane, out AbSpatialVector motionAccelerationOut)
{
    const uint64_t tLink = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
    const uint jointOffset = abU32(tLink, PXG_AB_LINK_mJointOffset, lane);
    const uint64_t tDofs = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;
    const uint parent = abU32(tLink, PXG_AB_LINK_mParents, lane);
    const uint64_t parentLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;

    const float3 rw = float3(abF32(tLink, PXG_AB_LINK_mRw_x, lane),
                             abF32(tLink, PXG_AB_LINK_mRw_y, lane),
                             abF32(tLink, PXG_AB_LINK_mRw_z, lane));

    // difference (a): which of the parent's two acceleration fields
    const AbSpatialVector parentAccel =
        alaLoadSV(parentLink,
                  internal ? PXG_AB_LINK_mMotionAccelerationInternal
                           : PXG_AB_LINK_mMotionAcceleration,
                  lane);
    const AbSpatialVector pMotionAcceleration = adTranslateSpatialVector(-rw, parentAccel);

    // difference (c): pass two adds the Coriolis term, pass one does not
    AbSpatialVector motionAcceleration = pMotionAcceleration;
    if (internal)
        motionAcceleration =
            asmSVecAdd(motionAcceleration, alaLoadSV(tLink, PXG_AB_LINK_mCoriolis, lane));

    const uint dofs = abU8(tLink, PXG_AB_LINK_mDofs, lane);

    AbSpatialVector isWs[3];
    float qstZ[3];
    float jointVel[3];
    AbSpatialVector axes[3];
    float3 invStIsT[3];

    for (uint ind = 0u; ind < 3u; ++ind)
    {
        isWs[ind] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        qstZ[ind] = 0.0f;
        jointVel[ind] = 0.0f;
        axes[ind] = isWs[ind];
        invStIsT[ind] = float3(0.0f, 0.0f, 0.0f);

        if (ind < dofs)
        {
            const uint64_t d = tDofs + uint64_t(ind) * PXG_AB_DOF_SIZE;
            isWs[ind] = alaLoadSV(d, PXG_AB_DOF_mIsW, lane);
            // difference (b): which per-dof term
            qstZ[ind] = abF32(d, internal ? PXG_AB_DOF_mQstZIcInternal : PXG_AB_DOF_mQstZ, lane);
            jointVel[ind] = abF32(d, PXG_AB_DOF_mJointVelocities, lane);
            axes[ind] = alaLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
            invStIsT[ind] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
        }
    }

    // against the PROPAGATED parent acceleration, NOT the running accumulator --
    // in pass two the Coriolis term has already moved the latter
    float tJAccel[3];
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        float jAccel = 0.0f;
        if (ind < dofs)
            jAccel = qstZ[ind] - asmInnerProduct(isWs[ind], pMotionAcceleration);
        tJAccel[ind] = jAccel;
    }

    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofs)
        {
            const uint64_t d = tDofs + uint64_t(ind) * PXG_AB_DOF_SIZE;
            const float3 i = invStIsT[ind];
            const float jAccel = i.x * tJAccel[0] + i.y * tJAccel[1] + i.z * tJAccel[2];

            // difference (d): assign in pass one, accumulate in pass two
            abStoreF32(d, PXG_AB_DOF_mJointAccel, lane,
                       internal ? abF32(d, PXG_AB_DOF_mJointAccel, lane) + jAccel : jAccel);

            const float jVel = jointVel[ind] + jAccel * dt;
            abStoreF32(d, PXG_AB_DOF_mJointVelocities, lane, jVel);
            abStoreF32(d, PXG_AB_DOF_mJointUnconstrainedVelocities, lane, jVel);

            motionAcceleration = asmSVecAdd(motionAcceleration, asmSVecScale(axes[ind], jAccel));
        }
    }

    motionAccelerationOut = motionAcceleration;
}

public void alaComputeLinkAcceleration(uint64_t artiBlock, uint64_t linkBlocks, uint64_t dofBlocks,
                                       uint numLinks, float dt, uint lane)
{
    const uint flags = abU32(artiBlock, PXG_AB_mFlags, lane);
    const bool fixBase = (flags & PXG_ARTI_FLAG_FIX_BASE) != 0u;

    // P8.40: loaded ONCE, before every loop -- see the header
    const uint64_t motionVelocities = abU64(artiBlock, PXG_AB_mMotionVelocitiesPtr, lane);

    const float4 COM_invMassW = abF4(artiBlock, PXG_AB_mCOM_TotalInvMassW, lane);
    const float3 COM = COM_invMassW.xyz;

    const AbSpatialVector zeroSV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    AbSpatialVector preMomentum = zeroSV;
    AbSpatialVector postMomentum = zeroSV;
    AbMat33 compoundInertia = abMat33Zero();

    // ---- 1. the root's EXTERNAL acceleration ----
    if (!fixBase)
    {
        const AbSpatialMatrix invInertia =
            asmLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane);
        const AbSpatialVector za = alaLoadSV(linkBlocks, PXG_AB_LINK_mZAVector, lane);

        const AbSpatialVector accel = alaNeg(asmMulVec(invInertia, za));
        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionAcceleration, lane, accel);

        const AbSpatialVector motionVel =
            asmSVecAdd(alaLoadSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane),
                       asmSVecScale(accel, dt));

        const float mass = abF32(linkBlocks, PXG_AB_LINK_mMass, lane);
        preMomentum.top = motionVel.bottom * mass;

        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane, motionVel);
    }
    else
    {
        // A fixed base has no free acceleration and no velocity at all -- and note
        // it also zeroes mMotionAccelerationInternal, which the free-base branch
        // does NOT (section 3 writes it there).
        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionAcceleration, lane, zeroSV);
        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionAccelerationInternal, lane, zeroSV);
        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane, zeroSV);
        alaStoreFlatVel(motionVelocities, 0u, zeroSV);
    }

    // ---- 2. pass one, root to leaf ----
    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t tLink = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;

        // difference (f): the starting velocity is read BEFORE the propagation
        const AbSpatialVector linkMotionVel = alaLoadSV(tLink, PXG_AB_LINK_mMotionVelocity, lane);

        AbSpatialVector motionAcceleration;
        alaPropagateLink(linkBlocks, dofBlocks, linkID, dt, false, lane, motionAcceleration);

        // difference (e)
        alaStoreSV(tLink, PXG_AB_LINK_mMotionAcceleration, lane, motionAcceleration);

        const AbSpatialVector motionVel =
            asmSVecAdd(linkMotionVel, asmSVecScale(motionAcceleration, dt));

        const float mass = abF32(tLink, PXG_AB_LINK_mMass, lane);
        preMomentum.top = preMomentum.top + motionVel.bottom * mass;

        alaStoreFlatVel(motionVelocities, linkID, motionVel);
        alaStoreSV(tLink, PXG_AB_LINK_mMotionVelocity, lane, motionVel);
    }

    // Computed OUTSIDE the guard below and used inside it AND in pass two -- for a
    // fixed base every momentum quantity is dead, but it is still computed, and
    // transcribing that is cheaper than proving it unobservable.
    const float3 preRootVel = preMomentum.top * COM_invMassW.w;

    // ---- 3. the root's INTERNAL acceleration, and the momentum bookkeeping ----
    if (!fixBase)
    {
        const AbSpatialMatrix invInertia =
            asmLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane);
        const AbSpatialVector zaInt = alaLoadSV(linkBlocks, PXG_AB_LINK_mZAIntVector, lane);

        const AbSpatialVector accel = alaNeg(asmMulVec(invInertia, zaInt));
        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionAccelerationInternal, lane, accel);

        // addSpatialVector: the EXTERNAL field accumulates the internal one
        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionAcceleration, lane,
                   asmSVecAdd(alaLoadSV(linkBlocks, PXG_AB_LINK_mMotionAcceleration, lane), accel));

        AbSpatialVector motionVel = alaLoadSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane);

        const AbMat33 inertia = abLoadPxMat33(linkBlocks, PXG_AB_LINK_mIsolatedInertia, lane);
        const float mass = abF32(linkBlocks, PXG_AB_LINK_mMass, lane);

        const float3 pos = abF4(linkBlocks, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz;
        const float3 offset = pos - COM;

        preMomentum.bottom = asmM33Transform(inertia, motionVel.top)
                           + asmCross(offset, motionVel.bottom - preRootVel) * mass;

        compoundInertia = asmM33Add(compoundInertia,
                                    asmTranslateInertiaMass(inertia, mass, offset));

        motionVel = asmSVecAdd(motionVel, asmSVecScale(accel, dt));

        postMomentum.top = motionVel.bottom * mass;
        alaStoreSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane, motionVel);
        alaStoreFlatVel(motionVelocities, 0u, motionVel);
    }

    // ---- 4. pass two, root to leaf ----
    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t tLink = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;

        AbSpatialVector motionAcceleration;
        alaPropagateLink(linkBlocks, dofBlocks, linkID, dt, true, lane, motionAcceleration);

        // difference (e)
        alaStoreSV(tLink, PXG_AB_LINK_mMotionAccelerationInternal, lane, motionAcceleration);

        const float3 pos = abF4(tLink, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz;
        // difference (f): re-loaded, and used BEFORE the update below
        AbSpatialVector motionVel = alaLoadSV(tLink, PXG_AB_LINK_mMotionVelocity, lane);

        const AbMat33 inertia = abLoadPxMat33(tLink, PXG_AB_LINK_mIsolatedInertia, lane);
        const float mass = abF32(tLink, PXG_AB_LINK_mMass, lane);
        const float3 offset = pos - COM;

        compoundInertia = asmM33Add(compoundInertia,
                                    asmTranslateInertiaMass(inertia, mass, offset));

        // `+=` in the CUDA, so the two new terms associate with EACH OTHER first
        // and only then with the accumulator -- `x + (A + B)`, not `(x + A) + B`.
        // Measured: the other association is eight ULP out by the end of the loop.
        preMomentum.bottom = preMomentum.bottom
                           + (asmM33Transform(inertia, motionVel.top)
                              + asmCross(offset, motionVel.bottom - preRootVel) * mass);

        motionVel = asmSVecAdd(motionVel, asmSVecScale(motionAcceleration, dt));

        postMomentum.top = postMomentum.top + motionVel.bottom * mass;

        alaStoreFlatVel(motionVelocities, linkID, motionVel);
        alaStoreSV(tLink, PXG_AB_LINK_mMotionVelocity, lane, motionVel);
    }

    // ---- 5. the momentum-conservation tail, free base only ----
    if (!fixBase)
    {
        // taken from postMomentum.top as it stands NOW: the loop after the next
        // one mutates it while iterating
        const float3 postRootVel = postMomentum.top * COM_invMassW.w;

        for (uint linkID = 0u; linkID < numLinks; ++linkID)
        {
            const uint64_t tLink = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
            const AbSpatialVector motionVel = alaLoadSV(tLink, PXG_AB_LINK_mMotionVelocity, lane);

            const AbMat33 inertia = abLoadPxMat33(tLink, PXG_AB_LINK_mIsolatedInertia, lane);
            const float mass = abF32(tLink, PXG_AB_LINK_mMass, lane);

            const float3 pos = abF4(tLink, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz;
            const float3 offset = pos - COM;

            postMomentum.bottom = postMomentum.bottom
                                + (asmM33Transform(inertia, motionVel.top)
                                   + asmCross(offset, motionVel.bottom - postRootVel) * mass);
        }

        const AbMat33 invCompoundInertia = asmM33Inverse(compoundInertia);

        // a ratio of MAGNITUDES, guarded at zero -- not of their squares
        const float denom0 = alaMagnitude(postMomentum.bottom);
        const float num = alaMagnitude(preMomentum.bottom);
        const float angRatio = denom0 == 0.0f ? 1.0f : num / denom0;

        const float3 deltaAngMom = postMomentum.bottom * (angRatio - 1.0f);
        const float3 deltaAng = asmM33Transform(invCompoundInertia, deltaAngMom);

        for (uint linkID = 0u; linkID < numLinks; ++linkID)
        {
            const uint64_t tLink = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
            const float3 offset =
                abF4(tLink, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz - COM;
            const AbSpatialVector velChange = asmSVec(deltaAng, asmCross(offset, deltaAng));

            alaStoreSV(tLink, PXG_AB_LINK_mMotionVelocity, lane,
                       asmSVecAdd(alaLoadSV(tLink, PXG_AB_LINK_mMotionVelocity, lane), velChange));

            // mutated INSIDE the loop, which is why postRootVel was taken before it
            postMomentum.top =
                postMomentum.top + velChange.bottom * abF32(tLink, PXG_AB_LINK_mMass, lane);
        }

        const float3 deltaLinMom = preMomentum.top - postMomentum.top;
        const float3 deltaLin = deltaLinMom * COM_invMassW.w;

        for (uint linkID = 0u; linkID < numLinks; ++linkID)
        {
            const uint64_t tLink = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
            AbSpatialVector vel = alaLoadSV(tLink, PXG_AB_LINK_mMotionVelocity, lane);
            vel.bottom = vel.bottom + deltaLin;
            alaStoreSV(tLink, PXG_AB_LINK_mMotionVelocity, lane, vel);
            alaStoreFlatVel(motionVelocities, linkID, vel);
        }
    }
}

struct AlaAccelParams
{
    uint64_t artiBlock;    // @0
    uint64_t linkBlocks;   // @8
    uint64_t dofBlocks;    // @16
    uint64_t numLinksPer;  // @24  uint32[nbLanes]
    float dt;              // @32
    uint nbLanes;          // @36
    uint pad0;             // @40
    uint pad1;             // @44 -> 48 bytes
};

[[vk::push_constant]]
ConstantBuffer<AlaAccelParams> gAlaAccel;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiLinkAcceleration(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gAlaAccel.nbLanes)
        return;

    const uint numLinks = (Ptr<uint>(gAlaAccel.numLinksPer))[lane];
    alaComputeLinkAcceleration(gAlaAccel.artiBlock, gAlaAccel.linkBlocks, gAlaAccel.dofBlocks,
                               numLinks, gAlaAccel.dt, lane);
}

// ===========================================================================
// P8.92c: computeUnconstrainedAccelerationsLaunch1T (forwardDynamic2.cu:2358)
// ===========================================================================
//
// The last kernel of `computeUnconstrainedVelocities`' five-kernel sequence, and
// the one everything since P8.83 has been building state for: this is where the
// acceleration that moves the link is finally written back into the block and the
// flat velocity array the constraint prep reads.
//
// Eight lines of index arithmetic and two calls.  Its differences from the four
// wrappers before it, each a mutant:
//
//  1. **Block (32, `2`, 1) and grid `num1TBlocks`** — P8.85's shape, not P8.88's
//     and P8.91c's (32, 1, 1) with twice the grid.  `numWarpsPerBlock` is
//     `COMPUTE_UNCONSTRAINED_VELOCITES / 32 == 2`
//     (PxgArticulationCore.cpp:376/479).  A port that carries P8.91c's 1 forward
//     processes half the articulations.
//  2. **The guard is per THREAD again** — `globalThreadIndex < nbArticulations`,
//     an `if` rather than an early return, and there is no ballot anywhere in
//     here, so nothing needs the inactive lanes.  That is the opposite of
//     P8.91c's block-level guard, and the two kernels sit next to each other in
//     the launch sequence.
//  3. **No island-node / bodySim chain and no record at all.**  Every input comes
//     from the block; `numLinks` from `mNumLinks`, not from a record.  P8.83 and
//     P8.85 walk the chain, P8.91c reads the block's articulation index — this
//     one needs neither.
//  4. **It calls BOTH bodies**, in order: the acceleration sweep and then the
//     bias-force sweep, which reads the two acceleration fields the first one
//     wrote.  Swapping them feeds the friction pass the previous step's values.

struct AlaLaunchParams
{
    uint64_t scDesc;   // @0  PxgArticulationCoreDesc*
};

[[vk::push_constant]]
ConstantBuffer<AlaLaunchParams> gAlaLaunch;

// TWO warps per block -- deliberately not shared with npArtiMassMatrix's 1
static const uint ALA_WARPS_PER_BLOCK = 2u;

[shader("compute")]
[numthreads(32, 2, 1)]
void computeUnconstrainedAccelerationsLaunch1T(uint3 groupThreadId : SV_GroupThreadID,
                                               uint3 groupId : SV_GroupID)
{
    const uint64_t d = gAlaLaunch.scDesc;

    const uint lane = groupThreadId.x & 31u;
    const uint globalWarpIndex = groupId.x * ALA_WARPS_PER_BLOCK + groupThreadId.y;
    const uint globalThreadIndex = globalWarpIndex * 32u + groupThreadId.x;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint64_t w = uint64_t(globalWarpIndex);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBlocks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                              + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBlocks = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                             + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    // from the BLOCK, with no record in sight
    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);
    const float dt = asfloat(artiU32Field(d, PXG_ARTI_CORE_DT));

    alaComputeLinkAcceleration(artiBlock, linkBlocks, dofBlocks, numLinks, dt, lane);

    // AFTER the sweep: it reads both acceleration fields the sweep just wrote
    alaJointTransmittedFrictionForce(linkBlocks, numLinks, lane);
}
