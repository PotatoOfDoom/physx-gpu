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

// npArtiUpdateKin.slang
//
// P8.117: `artiUpdateKinematic` (articulationDirectGpuApi.cu:295, the body
// updateKinematicInternal<zeroSimOutput> at :63-292).  Recomputes every
// link's body2World from the joint positions (the user changed root pose /
// joint positions through the direct GPU API), optionally refreshes the link
// velocities and zeroes the sim outputs, then refreshes transform cache +
// world bounds for every shape of every link (the P8.117a
// pxgSlangCacheBound machinery).
//
// Launch: block (32, 2, 1) -- one warp per articulation -- GRID-STRIDED
// (grid = min(engine cap, ceil(nbArticulations/2))).  ABI row
// {8,8,8,8,8,8,4,8,8,4,1} (80-byte blob).
//
// TRAPS, each a sweep mutant unless marked:
//  1. gpuIndices may be NULL: then the kernel walks ALL scDesc->nbArticulations
//     articulations by plain index (the automatic pre-sim call).  Non-null:
//     nbElements remapped jobs (the direct-API call).
//  2. The eNEEDS_KINEMATIC_UPDATE bit (1<<20) gates the whole body AND is
//     CLEARED unconditionally once seen -- even for articulations the
//     !zeroSimOutput && !positionsDirty early-out then skips.  positionsDirty
//     is (eDIRTY_ROOT_TRANSFORM | eDIRTY_POSITIONS) read AFTER the clear.
//  3. tempParentToChilds is a PxQuat* (16-byte stride) but tempRs is a
//     PxVec3* (12-byte TIGHT stride).
//  4. The revolute jointRotation is PxQuat(-jPosition[0], axis) -- the
//     NEGATED angle -- with the shortest-angle flip (w < 0 -> negate), then
//     (jointRotation * relativeQuat).getNormalized().  The spherical builds
//     ang = sum(motionMatrix.top * -jPosition[d]), normalizes IN PLACE
//     (PxVec3::normalize: reciprocal multiply, only when m > 0), and takes
//     the identity when angle < 1e-10.
//  5. The prismatic r has the extra u * jPosition[0] term with u =
//     motionMatrix[link][0].BOTTOM (the revolute axis is .top).
//  6. Phase B (lane 0, sequential): body2World.q =
//     (parent.q * newParentToChild.getConjugate()).getNormalized(), p =
//     parent.p + q.rotate(r) -- the conjugate, and the rotate by the NEW q.
//  7. The zeroSimOutput velocity update reads motionVelocities[parent]
//     (link 0 a live input), translates by MINUS c2p, and adds
//     motionMatrix[link][ind].rotate(body2World) * jointVelocities -- the
//     UPDATED body2World.
//  8. The zeroSimOutput zero pass covers motionAccelerations AND
//     linkIncomingJointForces over numLinks*6 floats and jointAccelerations
//     over numDofs -- one strided loop, two inner guards.  (The CUDA's
//     maxLinksReal and linkCountReal are the SAME value spelled twice;
//     transcribed as one bound.)
//  9. Phase D searches the sorted PxNodeIndex array for
//     uint2(bodySimIndex, (linkID << 1) | 1) and walks BACKWARD from the
//     last match; shapeIndex 0xFFFFFFFF entries are skipped but still
//     decrement.  numShapes == 0 skips phase D entirely (launch-uniform).
//
// DOCUMENTED DEVIATION (the P8.114 class): the CUDA grid-stride loop has
// per-articulation trip counts and __syncwarp()s inside the dirty gate; on
// subgroup-8 a warp spans four subgroups and the two warps of a block can
// disagree on both.  The port runs a BLOCK-UNIFORM iteration count
// (ceil(nbArticulations / warpsPerGrid)) with dead iterations predicated by
// `live`, folds the dirty gate into an `active` predicate, and hoists the
// three __syncwarp slots (:184 after phase A, :226 after phase B, :260
// before phase D) to unconditional AllMemoryBarrierWithGroupSync inside the
// uniform loop.
//
// TRANSCENDENTALS: the revolute/spherical joint rotation goes through
// sin/cos (asQuatAngleAxis), whose GPU results are within a few ULP of the
// host libm's but not bit-equal (the P8.94 measurement).  The fixture
// compares trig-free jobs (fix/prismatic joints, zero spherical positions)
// bitwise and everything downstream of a transcendental with the
// testArtiStepKernel ULP-or-absolute tolerance.

import pxgSlangCommon;
import pxgSlangSimCoreDesc;
import pxgSlangCacheBound;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;   // PXG_AR_JOINT_* core offsets
import pxgSlangSpatialMatrix;        // AbSpatialVector / asmSVec
import pxgSlangArticulationDynamic;  // adTranslateSpatialVector
import npArtiStep;                   // asQuatAngleAxis / asQuatConjugate / asQuatNormalize

static const uint UK_WARP = 32u;

// PxArticulationJointType, shared with npArtiJointFrame's AJ_* by value.  A
// mismatch cannot be silent: every joint of that type takes the wrong branch.
static const uint UK_J_FIX = 0u;
static const uint UK_J_PRISMATIC = 1u;
static const uint UK_J_REVOLUTE = 2u;
static const uint UK_J_REVOLUTE_UNWRAPPED = 3u;
static const uint UK_J_SPHERICAL = 4u;

struct UkParams
{
    uint64_t scDesc;           // @0  const PxgArticulationCoreDesc*
    uint64_t shapeSims;        // @8  const PxgShapeSim*
    uint64_t convexShapes;     // @16 const PxgShape*
    uint64_t transformCache;   // @24 PxsCachedTransform*
    uint64_t rigidNodeIndices; // @32 const PxNodeIndex* (sorted)
    uint64_t shapeIndices;     // @40 const PxU32*
    uint numShapes;            // @48
    uint pad0;                 // @52
    uint64_t bounds;           // @56 PxBounds3*
    uint64_t gpuIndices;       // @64 const PxArticulationGPUIndex* (may be NULL)
    uint nbElements;           // @72
    uint zeroSimWord;          // @76 bool byte 0
}

[[vk::push_constant]]
ConstantBuffer<UkParams> gUk;

// the per-warp pre-clear dirty word (see the second deviation note): lane 0
// stages the flag BEFORE the clear so the other three subgroups of the warp
// gate on the value the CUDA's lockstep read would have seen
groupshared uint gUkDirty[2];

[shader("compute")]
[numthreads(32, 2, 1)]
void artiUpdateKinematic(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint threadIndexInWarp = groupThreadId.x & (UK_WARP - 1u);
    const uint globalWarpIndex = groupId.x * 2u + groupThreadId.y;
    const uint warpsPerGrid = WorkgroupCount().x * 2u;
    const bool zeroSim = (gUk.zeroSimWord & 0xFFu) != 0u;

    // trap 1: NULL index buffer -> all articulations by plain index
    const bool haveIndices = gUk.gpuIndices != 0uL;
    const uint nbArticulations = haveIndices
        ? gUk.nbElements
        : (Ptr<uint>(gUk.scDesc + PXG_ARTI_CORE_NB_ARTICULATIONS))[0];
    const uint64_t articulations = (Ptr<uint64_t>(gUk.scDesc + PXG_ARTI_CORE_ARTICULATIONS))[0];

    // the deviation: a block-uniform trip count with dead iterations predicated
    const uint iters = (nbArticulations + warpsPerGrid - 1u) / warpsPerGrid;

    for (uint it = 0u; it < iters; ++it)
    {
        const uint i = globalWarpIndex + it * warpsPerGrid;
        const bool live = i < nbArticulations;

        bool active = false;
        uint linkCount = 0u, dofCount = 0u, bodySimIndex = 0u;
        uint64_t record = 0uL;
        uint64_t joints = 0uL, jointData = 0uL, jointPositions = 0uL, jointVelocities = 0uL;
        uint64_t body2Worlds = 0uL, body2Actors = 0uL, motionMatrix = 0uL, relativeQuats = 0uL;
        uint64_t tempNtc = 0uL, tempRs = 0uL, motionVel = 0uL;
        Ptr<uint> parents;

        if (live)
        {
            const uint articulationIndex =
                haveIndices ? (Ptr<uint>(gUk.gpuIndices))[i] : i;
            record = articulations + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;
        }

        // trap 2, and the SECOND deviation: the CUDA has every lane of the
        // warp read-modify-write the dirty word in lockstep (all 32 reads
        // precede all writes).  On subgroup-8 the four subgroups of a warp
        // run the RMW sequentially, so subgroup 0's clear would make
        // subgroups 1..3 read the ALREADY-CLEARED flag and gate themselves
        // off (measured: only lanes 0..7 ran the zeroing pass).  The port
        // stages the pre-clear value through groupshared: lane 0 reads,
        // stages, and clears; every lane gates on the staged value.
        if (live && threadIndexInWarp == 0u)
        {
            Ptr<uint> dirty = Ptr<uint>(record + PXG_ARTI_DATA_gpuDirtyFlag);
            const uint f = dirty[0];
            gUkDirty[groupThreadId.y] = f;
            if ((f & PXG_ARTI_DIRTY_NEEDS_KINEMATIC_UPDATE) != 0u)
                dirty[0] = f & ~PXG_ARTI_DIRTY_NEEDS_KINEMATIC_UPDATE;
        }
        AllMemoryBarrierWithGroupSync();   // slot 0 (the staging)
        if (live)
        {
            const uint f = gUkDirty[groupThreadId.y];
            if ((f & PXG_ARTI_DIRTY_NEEDS_KINEMATIC_UPDATE) != 0u)
            {
                // positionsDirty reads the POST-clear flag in the CUDA; the
                // masked bits are unaffected by the clear
                const bool positionsDirty =
                    (f & (PXG_ARTI_DIRTY_ROOT_TRANSFORM | PXG_ARTI_DIRTY_POSITIONS)) != 0u;
                active = zeroSim || positionsDirty;
            }
        }

        if (active)
        {
            linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
            dofCount = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
            bodySimIndex = artiU32Field(record, PXG_ARTI_DATA_BODY_SIM_INDEX);
            joints = artiPtrField(record, PXG_ARTI_JOINTS);
            jointData = artiPtrField(record, PXG_ARTI_JOINT_DATA);
            jointPositions = artiPtrField(record, PXG_ARTI_JOINT_POSITIONS);
            jointVelocities = artiPtrField(record, PXG_ARTI_JOINT_VELOCITIES);
            body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);
            body2Actors = artiPtrField(record, PXG_ARTI_LINK_BODY2ACTORS);
            motionMatrix = artiPtrField(record, PXG_ARTI_MOTION_MATRIX);
            relativeQuats = artiPtrField(record, PXG_ARTI_RELATIVE_QUAT);
            tempNtc = artiPtrField(record, PXG_ARTI_TEMP_PARENT_TO_CHILDS);
            tempRs = artiPtrField(record, PXG_ARTI_TEMP_RS);
            motionVel = artiPtrField(record, PXG_ARTI_MOTION_VELOCITIES);
            parents = Ptr<uint>(artiPtrField(record, PXG_ARTI_PARENTS));
        }

        // ---- phase A: per-joint newParentToChild + r, warp-strided ----
        if (active)
        {
            for (uint linkID = 1u + threadIndexInWarp; linkID < linkCount; linkID += UK_WARP)
            {
                const uint64_t jc = joints + uint64_t(linkID) * PXG_AR_JOINT_CORE_SIZE;
                const uint jo =
                    (Ptr<uint>(jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
                const uint nbDof =
                    (Ptr<uint>(jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1]
                    & 0xFFu;
                Ptr<float> jPosition = Ptr<float>(jointPositions + uint64_t(jo) * 4u);

                // joint.childPose.p / joint.parentPose.p (PxTransform: q @0, p @16)
                Ptr<float> pp = Ptr<float>(jc + PXG_AR_JOINT_parentPose + 16u);
                Ptr<float> cp = Ptr<float>(jc + PXG_AR_JOINT_childPose + 16u);
                const float3 parentOffset = float3(pp[0], pp[1], pp[2]);
                const float3 childOffset = float3(-cp[0], -cp[1], -cp[2]);

                Ptr<float> rq = Ptr<float>(relativeQuats + uint64_t(linkID) * 16u);
                const float4 relativeQuat = float4(rq[0], rq[1], rq[2], rq[3]);

                // jointType: PxU8 at core offset 539
                const uint jointType =
                    ((Ptr<uint>(jc + 536u))[0] >> 24) & 0xFFu;

                Ptr<float> mm0 = Ptr<float>(motionMatrix + uint64_t(linkID) * PXG_ARTI_SUBSPACE_STRIDE);

                float4 newParentToChild = float4(0.0f, 0.0f, 0.0f, 1.0f);
                float3 r = float3(0.0f, 0.0f, 0.0f);
                bool haveResult = true;

                if (jointType == UK_J_PRISMATIC)
                {
                    newParentToChild = relativeQuat;
                    const float3 u = float3(mm0[3], mm0[4], mm0[5]);   // trap 5: .bottom
                    const float3 e = quatRotate(newParentToChild, parentOffset);
                    const float3 d = childOffset;
                    r = e + d + u * jPosition[0];
                }
                else if (jointType == UK_J_REVOLUTE || jointType == UK_J_REVOLUTE_UNWRAPPED)
                {
                    const float3 u = float3(mm0[0], mm0[1], mm0[2]);   // .top
                    float4 jointRotation = asQuatAngleAxis(-jPosition[0], u);   // trap 4
                    if (jointRotation.w < 0.0f)
                        jointRotation = -jointRotation;
                    newParentToChild = asQuatNormalize(quatMul(jointRotation, relativeQuat));
                    const float3 e = quatRotate(newParentToChild, parentOffset);
                    const float3 d = childOffset;
                    r = e + d;
                }
                else if (jointType == UK_J_SPHERICAL)
                {
                    float3 ang = float3(0.0f, 0.0f, 0.0f);
                    for (uint d0 = 0u; d0 < nbDof; ++d0)
                    {
                        Ptr<float> mmd = Ptr<float>(motionMatrix
                                                    + uint64_t(linkID) * PXG_ARTI_SUBSPACE_STRIDE
                                                    + uint64_t(d0) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
                        ang += float3(mmd[0], mmd[1], mmd[2]) * -jPosition[d0];
                    }
                    // PxVec3::normalize(): reciprocal multiply, only when m > 0
                    const float angle = sqrt(ang.x * ang.x + ang.y * ang.y + ang.z * ang.z);
                    if (angle > 0.0f)
                        ang = ang * (1.0f / angle);
                    float4 jointRotation = angle < 1e-10f
                        ? float4(0.0f, 0.0f, 0.0f, 1.0f)
                        : asQuatAngleAxis(angle, ang);
                    if (jointRotation.w < 0.0f)
                        jointRotation = -jointRotation;
                    newParentToChild = asQuatNormalize(quatMul(jointRotation, relativeQuat));
                    const float3 e = quatRotate(newParentToChild, parentOffset);
                    const float3 d = childOffset;
                    r = e + d;
                }
                else if (jointType == UK_J_FIX)
                {
                    newParentToChild = relativeQuat;
                    const float3 e = quatRotate(newParentToChild, parentOffset);
                    const float3 d = childOffset;
                    r = e + d;
                }
                else
                {
                    haveResult = false;   // the CUDA default: leaves temp slots untouched
                }

                if (haveResult)
                {
                    Ptr<float> tq = Ptr<float>(tempNtc + uint64_t(linkID) * 16u);   // trap 3
                    tq[0] = newParentToChild.x; tq[1] = newParentToChild.y;
                    tq[2] = newParentToChild.z; tq[3] = newParentToChild.w;
                    Ptr<float> tr = Ptr<float>(tempRs + uint64_t(linkID) * 12u);    // trap 3
                    tr[0] = r.x; tr[1] = r.y; tr[2] = r.z;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();   // slot 1 (:184)

        // ---- phase B: the sequential body2World chain (lane 0) ----
        if (active && threadIndexInWarp == 0u)
        {
            for (uint linkID = 1u; linkID < linkCount; ++linkID)
            {
                const uint parent = parents[linkID];
                Ptr<float> pbw = Ptr<float>(body2Worlds + uint64_t(parent) * PXG_ARTI_TRANSFORM_SIZE);
                const float4 pQ = float4(pbw[0], pbw[1], pbw[2], pbw[3]);
                const float3 pP = float3(pbw[4], pbw[5], pbw[6]);

                Ptr<float> tq = Ptr<float>(tempNtc + uint64_t(linkID) * 16u);
                const float4 newParentToChild = float4(tq[0], tq[1], tq[2], tq[3]);
                Ptr<float> tr = Ptr<float>(tempRs + uint64_t(linkID) * 12u);
                const float3 r = float3(tr[0], tr[1], tr[2]);

                // trap 6
                const float4 q = asQuatNormalize(quatMul(pQ, asQuatConjugate(newParentToChild)));
                const float3 p = pP + quatRotate(q, r);
                Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
                bw[0] = q.x; bw[1] = q.y; bw[2] = q.z; bw[3] = q.w;
                bw[4] = p.x; bw[5] = p.y; bw[6] = p.z;

                if (zeroSim)
                {
                    // trap 7: the link-velocity update
                    Ptr<float> pmv = Ptr<float>(motionVel + uint64_t(parent) * 24u);
                    const AbSpatialVector parentVel =
                        asmSVec(float3(pmv[0], pmv[1], pmv[2]), float3(pmv[3], pmv[4], pmv[5]));
                    const float3 c2p = p - pP;
                    AbSpatialVector linkVelocity = adTranslateSpatialVector(-c2p, parentVel);

                    const uint jo =
                        (Ptr<uint>(jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
                    const uint nbDof =
                        (Ptr<uint>(jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1]
                        & 0xFFu;
                    Ptr<float> jVelocity = Ptr<float>(jointVelocities + uint64_t(jo) * 4u);
                    for (uint ind = 0u; ind < nbDof; ++ind)
                    {
                        Ptr<float> mmd = Ptr<float>(motionMatrix
                                                    + uint64_t(linkID) * PXG_ARTI_SUBSPACE_STRIDE
                                                    + uint64_t(ind) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
                        const float jVel = jVelocity[ind];
                        const float3 wTop = quatRotate(q, float3(mmd[0], mmd[1], mmd[2]));
                        const float3 wBot = quatRotate(q, float3(mmd[3], mmd[4], mmd[5]));
                        linkVelocity = asmSVec(linkVelocity.top + wTop * jVel,
                                              linkVelocity.bottom + wBot * jVel);
                    }
                    Ptr<float> mv = Ptr<float>(motionVel + uint64_t(linkID) * 24u);
                    mv[0] = linkVelocity.top.x; mv[1] = linkVelocity.top.y;
                    mv[2] = linkVelocity.top.z;
                    mv[3] = linkVelocity.bottom.x; mv[4] = linkVelocity.bottom.y;
                    mv[5] = linkVelocity.bottom.z;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();   // slot 2 (:226)

        // ---- phase C: the zeroSimOutput sim-output zeroing (trap 8) ----
        if (active && zeroSim)
        {
            Ptr<float> linkAccelData =
                Ptr<float>(artiPtrField(record, PXG_ARTI_MOTION_ACCELERATIONS));
            Ptr<float> linkIncomingJointForceData =
                Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_INCOMING_JOINT_FORCES));
            Ptr<float> jointAccel =
                Ptr<float>(artiPtrField(record, PXG_ARTI_JOINT_ACCELERATIONS));

            const uint linkCountReal = linkCount * 6u;
            const uint bound = max(dofCount, linkCountReal);
            for (uint index = threadIndexInWarp; index < bound; index += UK_WARP)
            {
                if (index < linkCountReal)
                {
                    linkAccelData[index] = 0.0f;
                    linkIncomingJointForceData[index] = 0.0f;
                }
                if (index < dofCount)
                    jointAccel[index] = 0.0f;
            }
        }
        AllMemoryBarrierWithGroupSync();   // slot 3 (:260; the numShapes==0 return sits before it)

        // ---- phase D: transform cache + bounds per link shape (trap 9) ----
        if (active && gUk.numShapes != 0u)
        {
            Ptr<uint2> nodeIndices = Ptr<uint2>(gUk.rigidNodeIndices);
            Ptr<uint> shapeIdx = Ptr<uint>(gUk.shapeIndices);
            Ptr<PxgShapeSim> shapeSimPool = Ptr<PxgShapeSim>(gUk.shapeSims);
            Ptr<PxsCachedTransform> cacheArray = Ptr<PxsCachedTransform>(gUk.transformCache);
            Ptr<PxBounds3Gpu> boundsArray = Ptr<PxBounds3Gpu>(gUk.bounds);
            Ptr<PxgShapeGpu> convexShapes = Ptr<PxgShapeGpu>(gUk.convexShapes);

            for (uint linkID = threadIndexInWarp; linkID < linkCount; linkID += UK_WARP)
            {
                const uint2 linkNodeIndex = uint2(bodySimIndex, (linkID << 1) | 1u);
                Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
                const PxgTransform28 body2World =
                    makeTransform28(float4(bw[0], bw[1], bw[2], bw[3]), float3(bw[4], bw[5], bw[6]));
                Ptr<float> ba = Ptr<float>(body2Actors + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
                const PxgTransform28 body2Actor =
                    makeTransform28(float4(ba[0], ba[1], ba[2], ba[3]), float3(ba[4], ba[5], ba[6]));

                // P9.28: THREE sentinel-guarded BDA reads, all clamped at the ADDRESS while
                // the sentinels still decide the VALUE.
                //
                // `pos` is 0xFFFFFFFF both when the search misses and when `pos--` wraps
                // from zero, and this platform turns a per-lane guard into a mask -- so
                // nodeIndices[0xFFFFFFFF] and shapeIdx[0xFFFFFFFF] are issued anyway, at
                // thirty-two and sixty-four gigabytes out.  The read inside the WHILE
                // CONDITION is the worse of the two: `&&` over a per-lane operand is a
                // mask and not a short circuit, so the right-hand side is evaluated for
                // lanes the left-hand side excluded.
                //
                // shapeSimPool has no count in this pack, so its clamp is to index 0
                // rather than to a bound -- valid whenever the pool is non-empty, which
                // phase D's `numShapes != 0` guarantees.
                const uint lastShape = gUk.numShapes - 1u;   // numShapes != 0 here
                uint pos = binarySearchNodeIndexBda(nodeIndices, gUk.numShapes, linkNodeIndex);

                while (pos != 0xFFFFFFFFu &&
                       all(nodeIndices[min(pos, lastShape)] == linkNodeIndex))
                {
                    const uint shapeIndex = shapeIdx[min(pos, lastShape)];
                    if (shapeIndex != 0xFFFFFFFFu)
                    {
                        const PxgShapeSim shapeSim =
                            shapeSimPool[shapeIndex != 0xFFFFFFFFu ? shapeIndex : 0u];
                        const PxgTransform28 absPos =
                            getAbsPose(body2World, shapeSim.mTransform, body2Actor);
                        updateCacheAndBoundBda(absPos, shapeSim, shapeIndex, cacheArray,
                                               boundsArray, convexShapes, true);
                    }
                    pos--;
                }
            }
        }
    }
}
