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


// npArtiUpdateBodies.slang
//
// P8.95b: `updateBodiesLaunch_Part2` (forwardDynamic2.cu:3077) — the per-LINK
// record writeback: poses into `linkBody2Worlds` (the buffer P8.93's DMA-back
// hands to the host), velocities into `motionVelocities` and the link records,
// the incoming joint forces, and the sleep check.  Manifest 277 -> 278.
//
// THIS IS THE KERNEL THE SCENE HAS BEEN WAITING FOR SINCE P8.65: the pose the
// host reads is `bodyCore->body2World`, filled from the pinned staging buffer,
// filled by the DMA-back, filled from `linkBody2Worlds` — and only this kernel
// writes `linkBody2Worlds` from the block's `mAccumulatedPose` that P8.94d
// integrates.  With both mapped, the link's REPORTED y can finally move.
//
// ===========================================================================
// THE LAUNCH SHAPE — the chain's first Z dimension
// ===========================================================================
//
// Grid (num1TBlocks, 1, ceil(maxLinks / 4)), block (32, 2, 4)
// (PxgArticulationCore.cpp:942-965: "we reach the 1024 threads/block limit
// with 16").  One thread per (articulation, link) pair:
// `linkIndex = threadIdx.z + blockIdx.z * 4`.  TWO guards, and they are not
// one: `linkIndex < maxLinks` sits beside the articulation guard, and
// `linkIndex < linkCount` runs AFTER the block is resolved — a lane whose
// articulation has fewer links than the fleet maximum passes the first and
// fails the second.
//
// ===========================================================================
// THE TRAPS
// ===========================================================================
//
//  1. **This kernel writes NO block field at all** — every store is
//     record-side, per-(articulation, link) disjoint, which is what makes the
//     Z-parallel launch race-free.  The fixture requires every block byte back
//     at its seed.
//  2. **The root's incoming joint force is ZEROED, not computed** — and the
//     root's acceleration uses `mMotionAcceleration + deltaV * invDt` with NO
//     internal term, where every other link adds `mMotionAccelerationInternal`.
//  3. **`incomingJointForceW` is force-at-link first, then translated to the
//     JOINT by `translateSpatialVectorInPlace(-dW, ·)` with `dW =
//     Gc.rotate(Lc.p)`** — the negated, rotated child-pose offset.  The store
//     is then rotated into the child JOINT frame by `GcLc.rotateInv`, where
//     `GcLc = Gc * Lc` — a full transform product, though only its q is used.
//  4. **The initial-velocity float4s keep their .w** — invMass and
//     penBiasClamp ride in the w lanes and are read out FIRST, then written
//     back around the new xyz.  And in the RECORD the ANGULAR float4 sits at
//     offset 0, the LINEAR at 16 — the reverse of every block-side (top =
//     angular) intuition the field names suggest.
//  5. **`body2World[linkIndex]` is a PACKED 28-byte PxTransform array** — q at
//     0, p at 16, element stride 28.  A 32-byte-stride write corrupts the
//     NEXT element's quaternion.
//  6. **`posMotionV *= invDt` only when NOT integrating** — the sleep check
//     then sees a VELOCITY either way (posMotionV is a displacement when
//     integrate is false).  Linear is `.bottom`, angular is `.top`, in that
//     argument order.
//  7. **The sleep threshold and articulation index come from the BLOCK; the
//     wake counter from the RECORD** — three sources for one call.
//
// SWEEP SURVIVOR U01, proven equivalent: dropping `linkIndex < maxLinks` from
// the fleet guard changes nothing OBSERVABLE, because the second guard
// (`linkIndex < linkCount`, with linkCount <= maxLinks always) subsumes it
// before any write, and the loads between the two guards address the BLOCK,
// whose base does not depend on linkIndex — in-bounds either way.  The guard
// protects only reads that cannot go out of bounds: the P8.94b/P8.94c class,
// third tranche running.  It stays because the CUDA has it.
//
// The `integrate` bool rides at byte 12 of the arg pack ({8, 4, 1} -> 16-byte
// blob), read as byte 0 of the padded word, the P8.83 pattern.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import npArtiSleep;

static const uint AUB_WARPS_PER_BLOCK = 2u;
static const uint AUB_LINKS_PER_BLOCK_Z = 4u;

AbSpatialVector aubLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

// a 24-byte Cm::UnAlignedSpatialVector array element, written as six floats
void aubStoreSV24(uint64_t base, uint index, AbSpatialVector v)
{
    Ptr<float> f = Ptr<float>(base + uint64_t(index) * 24u);
    f[0] = v.top.x; f[1] = v.top.y; f[2] = v.top.z;
    f[3] = v.bottom.x; f[4] = v.bottom.y; f[5] = v.bottom.z;
}

struct AubParams
{
    uint64_t desc;        // @0
    float dt;             // @8
    uint integrateWord;   // @12: the CUDA bool in byte 0
};

[[vk::push_constant]]
ConstantBuffer<AubParams> gAub;

[shader("compute")]
[numthreads(32, AUB_WARPS_PER_BLOCK, AUB_LINKS_PER_BLOCK_Z)]
void updateBodiesLaunch_Part2(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint64_t d = gAub.desc;
    const float dt = gAub.dt;
    const bool integrate = (gAub.integrateWord & 0xFFu) != 0u;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);

    const uint warpIndex = groupThreadId.y;
    const uint globalWarpIndex = groupId.x * AUB_WARPS_PER_BLOCK + warpIndex;
    const uint globalThreadIndex = globalWarpIndex * 32u + groupThreadId.x;

    const uint linkIndex = groupThreadId.z + groupId.z * AUB_LINKS_PER_BLOCK_Z;

    // the FIRST guard: fleet bounds
    if (globalThreadIndex >= nbArticulations || linkIndex >= maxLinks)
        return;

    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + uint64_t(globalWarpIndex) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBlocks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                              + uint64_t(globalWarpIndex) * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;

    const uint lane = groupThreadId.x & 31u;
    const uint linkCount = abU32(artiBlock, PXG_AB_mNumLinks, lane);

    // the SECOND guard: this articulation's own link count
    if (linkIndex >= linkCount)
        return;

    const float invDt = 1.0f / dt;

    const uint articulationIndex = abU32(artiBlock, PXG_AB_mArticulationIndex, lane);
    const uint64_t a = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS)
                     + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;

    const uint64_t body2World = artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS);
    const uint64_t motionVelocities = artiPtrField(a, PXG_ARTI_MOTION_VELOCITIES);
    const float sleepThreshold = abF32(artiBlock, PXG_AB_mSleepThreshold, lane);
    const uint64_t links = artiPtrField(a, PXG_ARTI_LINKS);
    const uint64_t wakeCounters = artiPtrField(a, PXG_ARTI_LINK_WAKE_COUNTERS);
    const uint64_t sleepData = artiPtrField(a, PXG_ARTI_LINK_SLEEP_DATA);
    const uint64_t linkAccelerations = artiPtrField(a, PXG_ARTI_MOTION_ACCELERATIONS);
    const uint64_t linkIncomingJointForces = artiPtrField(a, PXG_ARTI_LINK_INCOMING_JOINT_FORCES);

    const uint64_t link = linkBlocks + uint64_t(linkIndex) * PXG_AB_LINK_SIZE;

    if (linkIndex == 0u)
    {
        // trap 2: no internal acceleration term, and the joint force is ZERO
        const AbSpatialVector accW = asmSVecAdd(
            aubLoadSV(linkBlocks, PXG_AB_LINK_mMotionAcceleration, lane),
            asmSVecScale(aubLoadSV(linkBlocks, PXG_AB_LINK_mSolverSpatialDeltaVel, lane), invDt));
        aubStoreSV24(linkAccelerations, 0u, accW);
        aubStoreSV24(linkIncomingJointForces, 0u,
                     asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f)));
    }
    else
    {
        AbSpatialVector zaExt = aubLoadSV(link, PXG_AB_LINK_mZAVector, lane);
        if (artiU8Field(d, PXG_ARTI_CORE_EXTERNAL_FORCES_EVERY_TGS_ITER) != 0u)
        {
            // 5.12: under eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS mZAVector lacks
            // the external Z; updateBodiesLaunch1T's leaves-to-root sweep left the
            // cumulative chain in the record's zAForces[link], added back here.
            Ptr<float> zc = Ptr<float>(artiPtrField(a, PXG_ARTI_ZA_FORCES)
                                       + uint64_t(linkIndex) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
            zaExt = asmSVec(zaExt.top + float3(zc[0], zc[1], zc[2]),
                            zaExt.bottom + float3(zc[3], zc[4], zc[5]));
        }
        const AbSpatialVector zaInt = aubLoadSV(link, PXG_AB_LINK_mZAIntVector, lane);
        const AbSpatialVector accExt = aubLoadSV(link, PXG_AB_LINK_mMotionAcceleration, lane);
        const AbSpatialVector accInt =
            aubLoadSV(link, PXG_AB_LINK_mMotionAccelerationInternal, lane);
        AbMat33 iTL, iTR, iBL;
        abLoadSpatialMatrix(link, PXG_AB_LINK_mSpatialArticulatedInertia, lane, iTL, iTR, iBL);
        AbSpatialMatrix inertia;
        inertia.topLeft = iTL;
        inertia.topRight = iTR;
        inertia.bottomLeft = iBL;
        const AbSpatialVector deltaV = aubLoadSV(link, PXG_AB_LINK_mSolverSpatialDeltaVel, lane);
        const AbSpatialVector impulse = aubLoadSV(link, PXG_AB_LINK_mSolverSpatialImpulse, lane);

        const float4 gcQ = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
        const float3 gcP = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz;
        const float4 lcQ = abF4(link, PXG_AB_LINK_mChildPose + PXG_AB_STB_Q, lane);
        const float3 lcP = abF4(link, PXG_AB_LINK_mChildPose + PXG_AB_STB_P, lane).xyz;
        // trap 3: the full transform product, and the rotated child offset
        const float4 gcLcQ = quatMul(gcQ, lcQ);
        const float3 dW = quatRotate(gcQ, lcP);

        const AbSpatialVector accW =
            asmSVecAdd(asmSVecAdd(accExt, accInt), asmSVecScale(deltaV, invDt));

        AbSpatialVector incoming = asmSVecAdd(
            asmMulVec(inertia, accW),
            asmSVecAdd(asmSVecAdd(zaExt, zaInt), asmSVecScale(impulse, invDt)));
        asmTranslateSpatialVectorInPlace(-dW, incoming);

        aubStoreSV24(linkAccelerations, linkIndex, accW);
        aubStoreSV24(linkIncomingJointForces, linkIndex,
                     asmSVec(quatRotateInv(gcLcQ, incoming.top),
                             quatRotateInv(gcLcQ, incoming.bottom)));
    }

    const AbSpatialVector motionV = aubLoadSV(link, PXG_AB_LINK_mMotionVelocity, lane);
    // trap 4: the record's ANGULAR float4 sits at 0, the LINEAR at 16, and
    // both keep their .w
    const uint64_t linkRec = links + uint64_t(linkIndex) * PXG_ARTI_LINK_SIZE;
    Ptr<float4> recAng = Ptr<float4>(linkRec + PXG_ARTI_LINKREC_INITIAL_ANGVEL_PENBIAS);
    Ptr<float4> recLin = Ptr<float4>(linkRec + PXG_ARTI_LINKREC_INITIAL_LINVEL_INVMASS);
    const float initialLinVelXYZ_invMassW = recLin[0].w;
    const float initialAngVelXYZ_penBiasClamp = recAng[0].w;
    Ptr<float> wcPtr = Ptr<float>(wakeCounters + uint64_t(linkIndex) * 4u);
    float lwc = wcPtr[0];
    const float4 poseQ = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
    const float3 poseP = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz;

    AbSpatialVector posMotionV = aubLoadSV(link, PXG_AB_LINK_mPosMotionVelocity, lane);
    const float4 inverseInertiaXYZ_invMass = abF4(link, PXG_AB_LINK_mInvInertiaXYZ_invMassW, lane);

    aubStoreSV24(motionVelocities, linkIndex, motionV);
    recLin[0] = float4(motionV.bottom.x, motionV.bottom.y, motionV.bottom.z,
                       initialLinVelXYZ_invMassW);
    recAng[0] = float4(motionV.top.x, motionV.top.y, motionV.top.z,
                       initialAngVelXYZ_penBiasClamp);

    // trap 5: the PACKED 28-byte PxTransform array
    {
        Ptr<float> b2w = Ptr<float>(body2World + uint64_t(linkIndex) * 28u);
        b2w[0] = poseQ.x; b2w[1] = poseQ.y; b2w[2] = poseQ.z; b2w[3] = poseQ.w;
        b2w[4] = poseP.x; b2w[5] = poseP.y; b2w[6] = poseP.z;
    }

    // trap 6
    if (!integrate)
        posMotionV = asmSVecScale(posMotionV, invDt);

    // 5.12: scDesc->isSleepingDisabled (a bool at 69) skips the sleep check AND
    // the wake-counter write-back
    if (artiU8Field(d, PXG_ARTI_CORE_IS_SLEEPING_DISABLED) == 0u)
    {
        aslSleepCheck1T(sleepData + uint64_t(linkIndex) * PXG_ARTI_LINK_SLEEP_DATA_SIZE, lwc, poseQ,
                        posMotionV.bottom, posMotionV.top, dt, sleepThreshold,
                        inverseInertiaXYZ_invMass);

        wcPtr[0] = lwc;
    }
}

// ===========================================================================
// P9.43a: `updateBodiesLaunch_Part3` (forwardDynamic2.cu:3205-3249) -- the
// per-ARTICULATION sleep roll-up, the third and last stage of
// PxgArticulationCore::updateBodies.  One thread per articulation, block
// (32, 2, 1), grid ceil(nbArticulations / 64): it takes the maximum of the
// per-link wake counters Part2 just wrote, stores it as the articulation's wake
// counter in scDesc->articulationSleepData[articulationIndex], and rewrites the
// two transition bits of internalFlags from the (old, new) zero / non-zero
// transition.  Everything else in the flags word passes through untouched.
//
// This is the kernel the articulation's SLEEP hangs on: dmaBackArticulationData
// copies these eight bytes to the host, PxgPostSolveArticulationTask reads
// wakeCounter into the core and de/activates the island node from the bits
// (PxgContext.cpp:788-799).  Unmapped, the counter never counts down and
// isSleeping() never turns true -- P9.40's guard found it skipped x240 in the
// articulation scene; PXG_SCENE=articulationjoint measures it.
//
// Same arg pack as the other two stages ({8, 4, 1}); dt and integrate are
// declared and ignored by the .cu, so they are not read here.
//
//  1. PxMax(lwc, maxWc) on float is selectMax = (lwc > maxWc) ? lwc : maxWc
//     (PxMath.h:81 -> PxUnixMathIntrinsics.h:122), not max(): on a tie -- signed
//     zeros -- the SECOND operand (the +0-seeded running max) survives, so an
//     articulation whose counters are all -0.0 ends with +0 bits (P9.86).
//  2. The transition test is on FLOATS: -0.0 counts as zero either way.
//  3. The load of linkWakeCounters[0] sits inside the loop, never hoisted: a
//     zero-link articulation may carry a null array.
//  4. Three index families: globalWarpIndex for the block, lane for the slot in
//     it, mArticulationIndex[lane] for the record AND the sleep-data entry.
// ===========================================================================
[shader("compute")]
[numthreads(32, AUB_WARPS_PER_BLOCK, 1)]
void updateBodiesLaunch_Part3(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint64_t d = gAub.desc;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint warpIndex = groupThreadId.y;
    const uint globalWarpIndex = groupId.x * AUB_WARPS_PER_BLOCK + warpIndex;
    const uint globalThreadIndex = globalWarpIndex * 32u + groupThreadId.x;
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + uint64_t(globalWarpIndex) * PXG_AB_BLOCK_SIZE;
    const uint lane = groupThreadId.x & 31u;
    const uint linkCount = abU32(artiBlock, PXG_AB_mNumLinks, lane);
    const uint articulationIndex = abU32(artiBlock, PXG_AB_mArticulationIndex, lane);
    const uint64_t a = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS)
                     + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;
    Ptr<float> linkWakeCounters = Ptr<float>(artiPtrField(a, PXG_ARTI_LINK_WAKE_COUNTERS));

    Ptr<uint2> sleep = Ptr<uint2>(artiPtrField(d, PXG_ARTI_CORE_SLEEP_DATA)
                                  + uint64_t(articulationIndex) * PXG_ARTI_SLEEP_DATA_SIZE);
    const uint2 old = sleep[0];             // { wakeCounter bits, internalFlags }
    const float oldWc = asfloat(old.x);
    float maxWc = 0.0f;
    for (uint linkID = 0; linkID < linkCount; ++linkID)
    {
        const float lwc = linkWakeCounters[linkID];
        maxWc = (lwc > maxWc) ? lwc : maxWc; // P9.86: PxMax<float> = selectMax(a,b) = a > b ? a : b (PxUnixMathIntrinsics.h:122, forwardDynamic2.cu:3236): a tie keeps maxWc, so the +0 seed survives -0 counters (trap 1)
    }

    uint newFlags = old.y & ~(PXG_BODY_ACTIVATE_THIS_FRAME | PXG_BODY_DEACTIVATE_THIS_FRAME);
    if (oldWc == 0.0f && maxWc != 0.0f)
        newFlags |= PXG_BODY_ACTIVATE_THIS_FRAME;
    else if (maxWc == 0.0f && oldWc != 0.0f)
        newFlags |= PXG_BODY_DEACTIVATE_THIS_FRAME;

    sleep[0] = uint2(asuint(maxWc), newFlags);
}
