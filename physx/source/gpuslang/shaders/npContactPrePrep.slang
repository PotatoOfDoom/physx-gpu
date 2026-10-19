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

// npContactPrePrep.slang
//
// Slang port of the per-lane contact-patch packing core of
// createSolverContactConstraintDescsFromPatchWithSharedMem
// (gpusolver/src/CUDA/constraintBlockPrePrep.cu L167-253) plus the
// per-lane contact-point emission (L304-319; flat-equivalent to the
// 4-float4-per-pass shared-mem chunking because chunks are 4 float4s and
// the modifiable stride is 2, so pairs never straddle a chunk; also
// matches the commented-out reference form at L410-422) and the
// mWriteback = forceIndex + contactStartIndex store (L1044):
//   - hasTargetVelocity = internalFlags & eHAS_TARGET_VELOCITY(16);
//     disableStrongFriction = hasTargetVelocity ||
//     wuFlags & eDISABLE_STRONG_FRICTION(1<<2) (5.12 writes them as two
//     solver flags; 5.6.1 wrote the latter as ePER_POINT_FRICTION);
//   - the eFORCE_NO_RESPONSE(4) gate; inside it: the modifiable stride
//     (eCOMPRESSED_MODIFIED_CONTACT(128) -> contact pointer advanced by
//     startContactIndex a SECOND time, stride 2), numContacts/
//     contactStartIndex, the eDOMINANCE_0/1 (1<<14 / 1<<15) zeroing of the
//     mass-modification lanes, normal_restitution/damping passthrough,
//     dynamic/staticFriction, disableFriction = materialFlags &
//     eDISABLE_FRICTION(1) and isAccelerationSpring = materialFlags &
//     eCOMPLIANT_ACCELERATION_SPRING(16);
//   - solver flags (5.12 PxgSolverFlags.h): eHAS_TARGET_VELOCITY(2),
//     eDISABLE_STRONG_FRICTION(4), eHAS_FORCE_THRESHOLDS(1) from
//     wuFlags & eFORCE_THRESHOLD(1<<9), eDISABLE_FRICTION(8),
//     eCOMPLIANT_ACCELERATION_SPRING(16);
//   - the PxgMaterialContactData image (PxgConstraintPrep.h L85-94):
//     (restDistance, staticFriction, dynamicFriction,
//      mNumContacts | mSolverFlags<<8 | prevFrictionPatchCount<<16);
//   - contact emission: numContacts pairs of (point_separationW,
//     targetVel_maxImpulseW), the latter (0,0,0,PX_MAX_REAL) unless
//     modifiable.
// ALL outputs bitwise.  CUDA reads UNINITIALIZED dynamic/staticFriction +
// mSolverFlags when gContactPatch is NULL (the data image is still stored),
// and NEVER writes data.pad, so byte 15 of the stored image is stack
// garbage in every case -- the port zero-initializes those lanes/bytes;
// deterministic-and-documented replacement of UB.  CUDA also leaves the prepData lanes (invMass, normal,
// damping) UNWRITTEN outside the response gate; the port zeroes them
// (consumers read them only when mNumContacts > 0).

// ---------------------------------------------------------------------------
// P5 Step 4 (BDA, ENGINE launch ABI): the FULL constraintContactBlockPrePrepLaunch
// kernel (constraintBlockPrePrep.cu L739-1051), driven exactly as
// PxgSolverCore::constraintPrePrepParallel launches it (PxgSolverCore.cpp
// L570): params { PxgPrePrepDesc*, PxgSolverSharedDesc<IterativeSolveData>* },
// grid = ceil(totalBatches*32 / 128), block = CONSTRAINT_PREPREP_BLOCK = 128
// (4 warps/block; warp a = batch a).  All data reached through the byte-exact
// BDA desc mirrors (pxgSlangPrepDesc.slang, proven by testPrepDescLayout).
// The per-lane packing core below (cppPack) is the proven arithmetic; the
// full kernel reuses it verbatim with desc-driven access.
//
// Warp-collective mapping:
//   - the shared-memory patch/contact staging (createSolverContactConstraint-
//     DescsFromPatchWithSharedMem's uint4 chunk loops) is replaced by direct
//     per-lane loads: each lane's final effect is reading ITS OWN 64-byte
//     PxContactPatch and its own numContacts*stride float4 contact stream
//     (chunks are 4 float4s and the modifiable stride is 2, so pairs never
//     straddle a chunk — same flat-equivalence the cppPack port established);
//   - warpReduction<MaxOp>(contactCount) becomes a groupshared exchange +
//     lane-0 loop over the warp's 32 counts, with the barriers hoisted to
//     UNIFORM control flow (predicated work, group-level barriers);
//   - the lane-0 atomicAdds on gDesc->sharedContactConstraintIndex /
//     sharedFrictionConstraintIndex / sharedArticulationResponseIndex are BDA
//     InterlockedAdds on the desc fields; the allocated base is broadcast
//     through groupshared exactly like CUDA's blockContactPoints[warpInBlock].
// Deviations from the .cu (all deterministic replacements of UB / benign
// races, matching the cppPack doctrine):
//   - warp-uniform batch fields (constraintType/mDescStride word,
//     mConstraintBatchIndex, mask) are stored by lane 0 only (CUDA stores the
//     same value from all 32 lanes);
//   - dynamicFriction/staticFriction/mSolverFlags read as 0 when
//     gContactPatch is NULL (CUDA stores uninitialized registers);
//   - byte 3 of the PxgMaterialContactData word (data.pad) is 0 (CUDA stores
//     a garbage register byte).
// ---------------------------------------------------------------------------

import pxgSlangPrepDesc;
import pxgSlangCommon;
import pxgSlangPrepCommon;

static const float FLT_MAXQ = 3.402823466e+38f;
static const uint MAX_PTS = 20u;

struct CpIn
{
    uint4 meta;    // hasPatch, wuFlags, internalFlags, materialFlags
    float4 mass;   // mMassModification: invMassScale0, invInertiaScale0, invMassScale1, invInertiaScale1
    float4 nr;     // normal.xyz, restitution
    float4 mat;    // dynamicFriction, staticFriction, damping, restDistance
    uint4 cnt;     // startContactIndex, nbContacts, prevFrictionPatchCount, forceIndex
    float4 pts[MAX_PTS];  // compressed contact stream (base, pre-offset)
}

struct CpOut
{
    float4 invMassOut;   // dominance-zeroed mass modification
    float4 nrOut;        // normal_restitutionW
    float4 dampDataOut;  // x = damping, y = asfloat(writeback), zw = 0
    float4 dataOut;      // PxgMaterialContactData image
    uint4 counts;        // numContacts, contactStartIndex, stride, disableStrongFriction
    float4 pointsOut[12];  // per contact: (point_separationW, targetVel_maxImpulseW)
}

struct CpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CpParams> gParams;

RWStructuredBuffer<CpIn> gIn;    // binding 0
RWStructuredBuffer<CpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cppPack(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CpIn s = gIn[tid];

    const uint hasPatch = s.meta.x;
    const uint wuFlags = s.meta.y;
    const uint internalFlags = s.meta.z;
    const uint materialFlags = s.meta.w;

    uint numContacts = 0u;
    uint stride = 1u;
    uint contactStartIndex = 0u;
    uint ptOffset = 0u;

    CpOut o;
    o.invMassOut = float4(0, 0, 0, 0);
    o.nrOut = float4(0, 0, 0, 0);
    // CUDA: dynamicFriction/staticFriction/mSolverFlags are uninitialized when
    // !hasPatch or the response gate skips; the port pins them to zero
    float dynamicFriction = 0.0f, staticFriction = 0.0f;
    uint solverFlags = 0u;
    float damping = 0.0f;
    bool disableStrongFriction = false;

    if (hasPatch != 0u)
    {
        const bool hasTargetVelocity = (internalFlags & 16u) != 0u;  // PxContactPatch::eHAS_TARGET_VELOCITY
        disableStrongFriction = hasTargetVelocity ||
                                (wuFlags & (1u << 2u)) != 0u;        // PxgNpWorkUnitFlag::eDISABLE_STRONG_FRICTION
        bool disableFriction = false;
        bool isAccelerationSpring = false;

        if ((internalFlags & 4u) == 0u)                      // eFORCE_NO_RESPONSE
        {
            const bool isModifiable = (internalFlags & 128u) != 0u;  // eCOMPRESSED_MODIFIED_CONTACT

            ptOffset = s.cnt.x;                              // contactPoints += startContactIndex
            if (isModifiable)
            {
                ptOffset += s.cnt.x;                         // += startContactIndex again
                stride = 2u;
            }

            numContacts = s.cnt.y;
            contactStartIndex = s.cnt.x;

            float4 lin0X_ang0Y_lin1Z_ang1W = s.mass;
            if ((wuFlags & (1u << 14u)) != 0u)               // eDOMINANCE_0
            {
                lin0X_ang0Y_lin1Z_ang1W.x = 0.0f;
                lin0X_ang0Y_lin1Z_ang1W.y = 0.0f;
            }
            if ((wuFlags & (1u << 15u)) != 0u)               // eDOMINANCE_1
            {
                lin0X_ang0Y_lin1Z_ang1W.z = 0.0f;
                lin0X_ang0Y_lin1Z_ang1W.w = 0.0f;
            }
            o.invMassOut = lin0X_ang0Y_lin1Z_ang1W;

            o.nrOut = s.nr;
            damping = s.mat.z;

            dynamicFriction = s.mat.x;
            staticFriction = s.mat.y;
            disableFriction = (materialFlags & 1u) != 0u;    // PxMaterialFlag::eDISABLE_FRICTION
            isAccelerationSpring = (materialFlags & 16u) != 0u;  // eCOMPLIANT_ACCELERATION_SPRING
        }

        if (hasTargetVelocity)
            solverFlags |= 2u;                               // eHAS_TARGET_VELOCITY
        if (disableStrongFriction)
            solverFlags |= 4u;                               // eDISABLE_STRONG_FRICTION
        if ((wuFlags & (1u << 9u)) != 0u)                    // eFORCE_THRESHOLD
            solverFlags |= 1u;                               // eHAS_FORCE_THRESHOLDS
        if (disableFriction)
            solverFlags |= 8u;                               // eDISABLE_FRICTION
        if (isAccelerationSpring)
            solverFlags |= 16u;                              // eCOMPLIANT_ACCELERATION_SPRING
    }

    // PxgMaterialContactData image: restDistance, staticFriction,
    // dynamicFriction, mNumContacts | mSolverFlags<<8 | prevFriction<<16
    const uint packed = (numContacts & 0xFFu) | (solverFlags << 8u) | ((s.cnt.z & 0xFFu) << 16u);
    o.dataOut = float4(s.mat.w, staticFriction, dynamicFriction, asfloat(packed));

    // n.mWriteback = forceIndex + contactStartIndex (uint wrap allowed:
    // forceIndex defaults to 0xFFFFFFFF when no force buffer exists)
    o.dampDataOut = float4(damping, asfloat(s.cnt.w + contactStartIndex), 0.0f, 0.0f);
    o.counts = uint4(numContacts, contactStartIndex, stride, disableStrongFriction ? 1u : 0u);

    // contact emission (flat-equivalent form of the shared-mem chunk loop)
    for (uint j = 0u; j < 12u; ++j)
        o.pointsOut[j] = float4(0, 0, 0, 0);
    const uint count = numContacts * stride;
    uint emit = 0u;
    for (uint a = 0u; a < count; a += stride)
    {
        o.pointsOut[emit * 2u + 0u] = s.pts[ptOffset + a];
        if (stride != 1u)
            o.pointsOut[emit * 2u + 1u] = s.pts[ptOffset + a + 1u];
        else
            o.pointsOut[emit * 2u + 1u] = float4(0.0f, 0.0f, 0.0f, FLT_MAXQ);
        emit += 1u;
    }

    gOut[tid] = o;
}

// ---------------------------------------------------------------------------
// the FULL engine kernel (see file header)
// ---------------------------------------------------------------------------

static const uint PX_INVALID_NODE = 0xFFFFFFFFu;
static const uint PXG_BATCH_SIZE = 32u;
static const uint MAX_BUCKET_BITS = 7u; // PxsContactManagerBase::MaxBucketBits

struct CppLaunchParams
{
    uint64_t prePrepDesc; // @0 PxgPrePrepDesc*
    uint64_t sharedDesc;  // @8 PxgSolverSharedDesc<IterativeSolveData>*
}

[[vk::push_constant]]
ConstantBuffer<CppLaunchParams> gCppLaunch;

// mirrors computeRemapIndexRigidBody() (constraintPrepShared.cuh L51-161)
uint computeRemapIndexRigidBody(bool isSecondBody, Ptr<uint> partitionStartIndices,
                                Ptr<uint> partitionArticStartIndices, Ptr<uint> partitionJointCounts,
                                Ptr<uint> partitionArticulationJointCounts,
                                PartitionIndexDataGpu indexData, Ptr<uint> solverBodyReferences,
                                uint currPartition, uint maxNbPartitions, uint bodyId,
                                uint activeBodyOffset, uint totalRigidBatches, uint totalArticBatches,
                                uint nbElemsPerBody, uint nbSlabs)
{
    const uint partitionIndex = indexData.partition_patch_ctype & 0xFFFFu;
    const uint cType = indexData.partition_patch_ctype >> 24;
    const uint partitionEntryIndex = indexData.mPartitionEntryIndex;
    uint index = 0u;

    // PxgEdgeType: eCONTACT_MANAGER=0, eCONSTRAINT=1, eARTICULATION_CONTACT=2,
    // eARTICULATION_CONSTRAINT=3
    if (cType == 0u)
        index = partitionStartIndices[partitionIndex] + partitionJointCounts[partitionIndex] +
                partitionEntryIndex / PXG_BATCH_SIZE;
    else if (cType == 1u)
        index = partitionStartIndices[partitionIndex] + partitionEntryIndex / PXG_BATCH_SIZE;
    else if (cType == 2u)
        index = totalRigidBatches + partitionArticStartIndices[partitionIndex] +
                partitionArticulationJointCounts[partitionIndex] + partitionEntryIndex / PXG_BATCH_SIZE;
    else if (cType == 3u)
        index = totalRigidBatches + partitionArticStartIndices[partitionIndex] +
                partitionEntryIndex / PXG_BATCH_SIZE;

    const uint batchMask = PXG_BATCH_SIZE - 1u;
    index = (index * PXG_BATCH_SIZE * 2u) * nbElemsPerBody + (partitionEntryIndex & batchMask);
    if (isSecondBody)
        index += PXG_BATCH_SIZE * nbElemsPerBody;

    if (bodyId >= activeBodyOffset)
    {
        if ((partitionIndex & (~(maxNbPartitions - 1u))) != (currPartition & (~(maxNbPartitions - 1u))) ||
            partitionIndex <= currPartition)
        {
            // We changed slabs, so we need to introduce a new solver reference
            const uint slabId = partitionIndex / maxNbPartitions;
            uint referenceId = nbSlabs * (bodyId - activeBodyOffset) + slabId;

            solverBodyReferences[referenceId] = index; // PxgSolverReferences::mRemappedBodyIndex

            referenceId = (referenceId & (~31u)) * nbElemsPerBody + (referenceId & 31u);
            index = (totalArticBatches + totalRigidBatches) * PXG_BATCH_SIZE * 2u * nbElemsPerBody +
                    referenceId;
        }
    }
    return index;
}

// CUDA warpReduction<MaxOp> exchange + the lane-0 allocated-base broadcast
// (blockContactPoints[warpIndexInBlock]); 128 threads = 4 warps per block.
groupshared uint gsContactCount[128];
groupshared uint gsWarpContactBase[4];

[shader("compute")]
[numthreads(128, 1, 1)]
void constraintContactBlockPrePrepLaunch(uint3 groupId : SV_GroupID,
                                         uint3 groupThreadId : SV_GroupThreadID)
{
    const uint tid = groupThreadId.x;
    const uint blockStride = 128u / 32u;
    const uint warpIndexInBlock = tid / 32u;
    const uint threadIndexInWarp = tid & 31u;
    const uint warpIndex = groupId.x * blockStride + warpIndexInBlock; // the batch index a

    Ptr<PxgPrePrepDescGpu> desc = Ptr<PxgPrePrepDescGpu>(gCppLaunch.prePrepDesc);
    Ptr<PxgSolverSharedDescIterGpu> sharedDesc = Ptr<PxgSolverSharedDescIterGpu>(gCppLaunch.sharedDesc);

    const uint nbArticulations = desc[0].mTotalActiveArticulations;
    Ptr<PxgBodySim> bodySims = Ptr<PxgBodySim>(sharedDesc[0].mBodySimBufferDeviceData);
    Ptr<uint> solverBodyIndices = Ptr<uint>(desc[0].solverBodyIndices);
    Ptr<float2> torsionalData = Ptr<float2>(desc[0].mTorsionalFrictionData);

    const uint numBatches = desc[0].numBatches;
    const uint numStaticBatches = desc[0].numStaticBatches;
    const uint numArtiBatches = desc[0].numArtiBatches;
    const uint numArtiStaticBatches = desc[0].numArtiStaticBatches;
    const uint numArtiSelfBatches = desc[0].numArtiSelfBatches;
    const uint dynamicBatchEnd = numBatches + numArtiBatches;
    const uint totalStaticPlusDynamic = dynamicBatchEnd + numArtiStaticBatches;
    const uint totalBatches = totalStaticPlusDynamic + numArtiSelfBatches + numStaticBatches;

    const uint nbElemsPerBody = desc[0].nbElementsPerBody;

    Ptr<PartitionIndexDataGpu> partitionIndexData = Ptr<PartitionIndexDataGpu>(desc[0].mPartitionIndices);

    const uint a = warpIndex;
    const bool warpActive = a < totalBatches;

    // ---- per-lane batch/desc walk (predicated; barriers stay uniform) ----
    uint conType = 0xFFFFFFFFu;
    uint descStride = 0u;
    uint batchIndex = 0u;      // batchHeader.mConstraintBatchIndex
    uint headerWord0 = 0u;
    uint headerMask = 0u;
    uint startPartitionIndex = 0u;
    bool isContact = false;

    Ptr<PxgBlockConstraintBatchGpu> blockBatches = Ptr<PxgBlockConstraintBatchGpu>(desc[0].blockBatches);
    Ptr<PxgBlockWorkUnit> blockWorkUnit = Ptr<PxgBlockWorkUnit>(desc[0].blockWorkUnit);
    Ptr<PxgBlockContactData> blockContactData = Ptr<PxgBlockContactData>(desc[0].blockContactData);

    if (warpActive)
    {
        Ptr<PxgConstraintBatchHeaderGpu> headers = Ptr<PxgConstraintBatchHeaderGpu>(desc[0].mBatchHeaders);
        headerWord0 = headers[a].mDescStride_constraintType;
        batchIndex = headers[a].mConstraintBatchIndex;
        startPartitionIndex = headers[a].mStartPartitionIndex;
        headerMask = headers[a].mask;
        conType = headerWord0 >> 16;       // PxgConstraintBatchHeader::constraintType
        descStride = headerWord0 & 0xFFFFu;
        // eCONTACT = 1, eARTICULATION_CONTACT = 3
        isContact = (conType == 1u) || (conType == 3u);
    }

    uint64_t contactPatchAddr = 0;  // PxContactPatch* (0 = NULL)
    uint64_t contactsAddr = 0;      // PxContact* (0 = NULL)
    uint prevFrictionPatchCount = 0u;
    uint contactCount = 0u;
    uint forceIndex = 0xFFFFFFFFu;
    uint laneFlags = 0u;
    float laneRestDistance = 0.0f;

    if (warpActive && isContact)
    {
        // Copy across data from batchHeader (CUDA: all lanes store the same
        // values; the port stores from lane 0)
        if (threadIndexInWarp == 0u)
        {
            blockBatches[a].descStride_constraintType = headerWord0;
            blockBatches[a].mConstraintBatchIndex = batchIndex;
            blockBatches[a].mask = headerMask;
        }

        blockWorkUnit[batchIndex].mFrictionPatchIndex[threadIndexInWarp] = 0xFFFFFFFFu;

        if (threadIndexInWarp < descStride)
        {
            const uint uniqueIndex = conType == 1u
                ? (Ptr<uint>(desc[0].mContactUniqueIndices))[startPartitionIndex + threadIndexInWarp]
                : (Ptr<uint>(desc[0].mArtiContactUniqueIndices))[startPartitionIndex + threadIndexInWarp];

            Ptr<PxgSolverConstraintManagerConstantsGpu> constantData =
                Ptr<PxgSolverConstraintManagerConstantsGpu>(desc[0].mContactConstantData);
            Ptr<PartitionNodeDataGpu> partitionNodeData =
                Ptr<PartitionNodeDataGpu>(desc[0].mPartitionNodeData);

            const PartitionNodeDataGpu nodeData = partitionNodeData[uniqueIndex];
            const uint2 igNodeIndexA = nodeData.mNodeIndex0;
            const uint2 igNodeIndexB = nodeData.mNodeIndex1;

            const uint nodeIndexA = igNodeIndexA.x; // PxNodeIndex::index()
            const uint nodeIndexB = igNodeIndexB.x;

            const uint nextIndexA = nodeData.mNextIndex0;
            const uint nextIndexB = nodeData.mNextIndex1;

            const bool isStaticA = nodeIndexA == PX_INVALID_NODE;
            const bool isStaticB = nodeIndexB == PX_INVALID_NODE;

            const bool isArticulationA = (igNodeIndexA.y & 1u) != 0u; // isArticulation()
            const bool isArticulationB = (igNodeIndexB.y & 1u) != 0u;

            const uint solverBodyIndexA = isStaticA ? 0u : solverBodyIndices[nodeIndexA];
            const uint solverBodyIndexB = isStaticB ? 0u : solverBodyIndices[nodeIndexB];

            blockBatches[a].bodyAIndex[threadIndexInWarp] = solverBodyIndexA;
            blockBatches[a].bodyBIndex[threadIndexInWarp] = solverBodyIndexB;
            blockBatches[a].bodyANodeIndex[threadIndexInWarp] = igNodeIndexA;
            blockBatches[a].bodyBNodeIndex[threadIndexInWarp] = igNodeIndexB;

            const uint patchIndex = (partitionIndexData[uniqueIndex].partition_patch_ctype >> 16) & 0xFFu;

            const uint maxConstraintPartitions = desc[0].mMaxConstraintPartitions;
            const uint nbSlabs = desc[0].mTotalSlabs;

            const uint currentPartition = partitionIndexData[uniqueIndex].partition_patch_ctype & 0xFFFFu;

            const uint slabId = currentPartition / maxConstraintPartitions;
            blockBatches[a].slabId[threadIndexInWarp] = slabId;

            Ptr<uint> partitionStartBatch = Ptr<uint>(desc[0].mPartitionstartBatchIndices);
            Ptr<uint> partitionArtiStartBatch = Ptr<uint>(desc[0].mPartitionArtiStartBatchIndices);
            Ptr<uint> partitionJointCounts = Ptr<uint>(desc[0].mPartitionJointCounts);
            Ptr<uint> partitionArtiJointCounts = Ptr<uint>(desc[0].mPartitionArtiJointCounts);
            Ptr<uint> solverBodyReferences = Ptr<uint>(desc[0].mSolverBodyReferences);
            Ptr<uint> slabMaskWords = Ptr<uint>(sharedDesc[0].articulationSlabMask); // uint4 elements

            // Static articulation contacts don't have all the data in the
            // partitioning assigned to be able to compute the data below
            if (a < dynamicBatchEnd)
            {
                if (!isArticulationA)
                {
                    blockBatches[a].remappedBodyAIndex[threadIndexInWarp] = computeRemapIndexRigidBody(
                        (nextIndexA & 1u) != 0u, partitionStartBatch, partitionArtiStartBatch,
                        partitionJointCounts, partitionArtiJointCounts, partitionIndexData[nextIndexA >> 1],
                        solverBodyReferences, currentPartition, desc[0].mMaxConstraintPartitions,
                        solverBodyIndexA, desc[0].mActiveBodyStartOffset, numBatches, numArtiBatches,
                        nbElemsPerBody, nbSlabs);
                }
                else
                {
                    const uint artiId = bodySims[nodeIndexA].articulationRemapId;
                    blockBatches[a].remappedBodyAIndex[threadIndexInWarp] = artiId;

                    const uint partitionId = currentPartition & (maxConstraintPartitions - 1u);
                    const uint ind = solverBodyIndexA + (slabId + partitionId * nbSlabs) * nbArticulations;

                    slabMaskWords[ind * 4u + 0u] = igNodeIndexA.y >> 1; // articulationLinkId()
                    slabMaskWords[ind * 4u + 1u] = a * 32u * 2u * nbElemsPerBody + threadIndexInWarp;
                }

                if (!isArticulationB)
                {
                    blockBatches[a].remappedBodyBIndex[threadIndexInWarp] = computeRemapIndexRigidBody(
                        (nextIndexB & 1u) != 0u, partitionStartBatch, partitionArtiStartBatch,
                        partitionJointCounts, partitionArtiJointCounts, partitionIndexData[nextIndexB >> 1],
                        solverBodyReferences, currentPartition, desc[0].mMaxConstraintPartitions,
                        solverBodyIndexB, desc[0].mActiveBodyStartOffset, numBatches, numArtiBatches,
                        nbElemsPerBody, nbSlabs);
                }
                else
                {
                    blockBatches[a].remappedBodyBIndex[threadIndexInWarp] =
                        bodySims[nodeIndexB].articulationRemapId;

                    const uint partitionId = currentPartition & (maxConstraintPartitions - 1u);
                    const uint ind = solverBodyIndexB + (slabId + partitionId * nbSlabs) * nbArticulations;

                    slabMaskWords[ind * 4u + 2u] = igNodeIndexB.y >> 1;
                    slabMaskWords[ind * 4u + 3u] = (a * 32u * 2u + 32u) * nbElemsPerBody + threadIndexInWarp;
                }
            }
            else
            {
                if (isArticulationA)
                    blockBatches[a].remappedBodyAIndex[threadIndexInWarp] =
                        bodySims[nodeIndexA].articulationRemapId;
                if (isArticulationB)
                    blockBatches[a].remappedBodyBIndex[threadIndexInWarp] =
                        bodySims[nodeIndexB].articulationRemapId;
            }

            const uint edgeIndex = constantData[uniqueIndex].mEdgeIndex;
            blockWorkUnit[batchIndex].mEdgeIndex[threadIndexInWarp] = edgeIndex;
            blockWorkUnit[batchIndex].mPatchIndex[threadIndexInWarp] = patchIndex;

            const uint npIndex = (Ptr<uint>(desc[0].mNpOutputIndices))[uniqueIndex];

            const uint tIndex = npIndex & ((1u << MAX_BUCKET_BITS) - 1u); // computeBucketIndexFromId
            const uint cmOutputIndex = desc[0].mCmOutputOffsets[tIndex] + (npIndex >> MAX_BUCKET_BITS);

            blockBatches[a].shapeInteraction[threadIndexInWarp] =
                (Ptr<uint64_t>(desc[0].mShapeInteractions))[cmOutputIndex];
            laneRestDistance = (Ptr<float>(desc[0].mRestDistances))[cmOutputIndex];
            blockWorkUnit[batchIndex].mRestDistance[threadIndexInWarp] = laneRestDistance;
            blockWorkUnit[batchIndex].mTorsionalFrictionData[threadIndexInWarp] =
                torsionalData[cmOutputIndex];

            Ptr<PxsContactManagerOutputGpu> cmOutputs =
                Ptr<PxsContactManagerOutputGpu>(desc[0].contactManagerOutputBase);
            const PxsContactManagerOutputGpu cmOutput = cmOutputs[cmOutputIndex];

            laneFlags = cmOutput.contactWord >> 16; // PxsContactManagerOutput::flags (PxU16)
            blockWorkUnit[batchIndex].mFlags[threadIndexInWarp] = laneFlags;

            const uint totalContacts = cmOutput.contactWord & 0xFFFFu; // nbContacts
            const bool hasContacts = totalContacts != 0u;

            // pointer differences against the CPU stream bases (element indices)
            const uint patchStartIndex = uint((cmOutput.contactPatches - desc[0].cpuCompressedPatchesBase) / 64);
            const uint contactStartElem = uint((cmOutput.contactPoints - desc[0].cpuCompressedContactsBase) / 16);
            if (cmOutput.contactForces != 0)
                forceIndex = uint((cmOutput.contactForces - desc[0].cpuForceBufferBase) / 4);

            prevFrictionPatchCount = (Ptr<uint>(desc[0].prevFrictionPatchCount))[edgeIndex];

            const uint nbPatches = (cmOutput.patchWord >> 8) & 0xFFu;
            (Ptr<uint>(desc[0].currFrictionPatchCount))[edgeIndex] = nbPatches;

            if (hasContacts && patchIndex < nbPatches)
            {
                if (cmOutput.contactPatches != 0)
                    contactPatchAddr = desc[0].compressedPatches + uint64_t(patchStartIndex + patchIndex) * 64;

                blockWorkUnit[batchIndex].mFrictionPatchIndex[threadIndexInWarp] =
                    patchStartIndex + patchIndex;

                if (cmOutput.contactPoints != 0)
                    contactsAddr = desc[0].compressedContacts + uint64_t(contactStartElem) * 16;

                contactCount = (Ptr<PxContactPatchGpu>(contactPatchAddr))[0].cnt0 >> 16 & 0xFFu; // nbContacts
            }
        }
    }

    // ---- warp max + allocation (uniform barriers; predicated work) ----
    gsContactCount[tid] = contactCount;
    GroupMemoryBarrierWithGroupSync();

    if (warpActive && isContact && threadIndexInWarp == 0u)
    {
        uint maxCount = 0u;
        for (uint l = 0u; l < 32u; ++l)
        {
            const uint v = gsContactCount[warpIndexInBlock * 32u + l];
            maxCount = v > maxCount ? v : maxCount;
        }

        uint contactIndex;
        InterlockedAdd(desc[0].sharedContactConstraintIndex, maxCount, contactIndex);
        gsWarpContactBase[warpIndexInBlock] = contactIndex;

        blockBatches[a].startConstraintIndex = contactIndex;
        blockBatches[a].blockContactIndex = contactIndex;

        const uint maxFrictions = 4u;
        uint startFriction;
        InterlockedAdd(desc[0].sharedFrictionConstraintIndex, maxFrictions, startFriction);
        blockBatches[a].startFrictionIndex = startFriction;

        if (conType == 3u) // eARTICULATION_CONTACT
        {
            uint startResponse;
            InterlockedAdd(desc[0].sharedArticulationResponseIndex, maxFrictions + maxCount, startResponse);
            blockBatches[a].mArticulationResponseIndex = startResponse;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (!(warpActive && isContact))
        return;

    const uint warpContactBase = gsWarpContactBase[warpIndexInBlock];

    // ---- createSolverContactConstraintDescsFromPatchWithSharedMem, per-lane
    // flat-equivalent (the proven cppPack core, desc-driven) ----
    const uint endIndex = descStride;

    uint numContacts = 0u;
    uint stride = 1u;
    uint contactStartIndex = 0u;

    if (threadIndexInWarp < endIndex)
    {
        float dynamicFriction = 0.0f, staticFriction = 0.0f;
        uint solverFlags = 0u;

        if (contactPatchAddr != 0)
        {
            const PxContactPatchGpu patch = (Ptr<PxContactPatchGpu>(contactPatchAddr))[0];
            const uint internalFlags = patch.cnt1 & 0xFFFFu;
            const uint materialFlags = patch.cnt0 >> 24;
            const uint wuFlags = laneFlags;

            const bool hasTargetVelocity = (internalFlags & 16u) != 0u;     // PxContactPatch::eHAS_TARGET_VELOCITY
            const bool disableStrongFriction = hasTargetVelocity ||
                                               (wuFlags & (1u << 2u)) != 0u; // PxgNpWorkUnitFlag::eDISABLE_STRONG_FRICTION
            bool disableFriction = false;
            bool isAccelerationSpring = false;

            if ((internalFlags & 4u) == 0u)                                // eFORCE_NO_RESPONSE
            {
                const bool isModifiable = (internalFlags & 128u) != 0u;    // eCOMPRESSED_MODIFIED_CONTACT
                const uint startCt = patch.cnt0 & 0xFFFFu;                 // startContactIndex

                contactsAddr += uint64_t(startCt) * 16;                    // contactPoints += startContactIndex
                if (isModifiable)
                {
                    contactsAddr += uint64_t(startCt) * 16;                // += startContactIndex again
                    stride = 2u;
                }

                numContacts = (patch.cnt0 >> 16) & 0xFFu;                  // nbContacts
                contactStartIndex = startCt;

                float4 lin0X_ang0Y_lin1Z_ang1W = patch.mMassModification;
                if ((wuFlags & (1u << 14u)) != 0u)                         // eDOMINANCE_0
                {
                    lin0X_ang0Y_lin1Z_ang1W.x = 0.0f;
                    lin0X_ang0Y_lin1Z_ang1W.y = 0.0f;
                }
                if ((wuFlags & (1u << 15u)) != 0u)                         // eDOMINANCE_1
                {
                    lin0X_ang0Y_lin1Z_ang1W.z = 0.0f;
                    lin0X_ang0Y_lin1Z_ang1W.w = 0.0f;
                }
                blockContactData[batchIndex].mInvMassScale[threadIndexInWarp] = lin0X_ang0Y_lin1Z_ang1W;
                blockContactData[batchIndex].normal_restitutionW[threadIndexInWarp] =
                    patch.normal_restitution;
                blockContactData[batchIndex].damping[threadIndexInWarp] = patch.damping;

                dynamicFriction = patch.dynamicFriction;
                staticFriction = patch.staticFriction;
                disableFriction = (materialFlags & 1u) != 0u;              // eDISABLE_FRICTION
                isAccelerationSpring = (materialFlags & 16u) != 0u;        // eCOMPLIANT_ACCELERATION_SPRING
            }

            if (hasTargetVelocity)
                solverFlags |= 2u;                                         // eHAS_TARGET_VELOCITY
            if (disableStrongFriction)
                solverFlags |= 4u;                                         // eDISABLE_STRONG_FRICTION
            if ((wuFlags & (1u << 9u)) != 0u)                              // eFORCE_THRESHOLD
                solverFlags |= 1u;                                         // eHAS_FORCE_THRESHOLDS
            if (disableFriction)
                solverFlags |= 8u;                                         // eDISABLE_FRICTION
            if (isAccelerationSpring)
                solverFlags |= 16u;                                        // eCOMPLIANT_ACCELERATION_SPRING
        }

        // PxgMaterialContactData image: restDistance, staticFriction,
        // dynamicFriction, mNumContacts | mSolverFlags<<8 | prevFriction<<16
        const uint packed = (numContacts & 0xFFu) | (solverFlags << 8u) |
                            ((prevFrictionPatchCount & 0xFFu) << 16u);
        blockContactData[batchIndex].contactData[threadIndexInWarp] =
            float4(laneRestDistance, staticFriction, dynamicFriction, asfloat(packed));
    }

    // contact emission (flat-equivalent form of the shared-mem chunk loop);
    // the warp's allocated element base plays CUDA's
    // blockContactPoints[warpIndexInBlock] result pointer
    Ptr<PxgBlockContactPoint> blockContactPoints = Ptr<PxgBlockContactPoint>(desc[0].blockContactPoints);
    for (uint j = 0u; j < numContacts; ++j)
    {
        Ptr<float4> src = Ptr<float4>(contactsAddr);
        blockContactPoints[warpContactBase + j].point_separationW[threadIndexInWarp] = src[j * stride];
        if (stride != 1u)
            blockContactPoints[warpContactBase + j].targetVel_maxImpulseW[threadIndexInWarp] =
                src[j * stride + 1u];
        else
            blockContactPoints[warpContactBase + j].targetVel_maxImpulseW[threadIndexInWarp] =
                float4(0.0f, 0.0f, 0.0f, FLT_MAXQ);
    }

    // n.mWriteback = forceIndex + contactStartIndex (all 32 lanes; uint wrap
    // allowed: forceIndex defaults to 0xFFFFFFFF when no force buffer exists)
    blockWorkUnit[batchIndex].mWriteback[threadIndexInWarp] = forceIndex + contactStartIndex;
}

// ---------------------------------------------------------------------------
// clearFrictionPatchCounts (P5 Step 5 BDA engine ABI)
// ---------------------------------------------------------------------------
// Port of clearFrictionPatchCounts() in constraintBlockPrePrep.cu: one thread
// per destroyed edge, zeroing that edge's previous-frame friction patch count
// so stale friction correlation data cannot carry over to a recycled edge.
// Launched from PxgCudaSolverCore/PxgTGSCudaSolverCore::gpuMemDMAUpContactData
// whenever nbDestroyedEdges > 0.  block = CLEAR_FRICTION_PATCH_COUNTS (256),
// grid = ceil(nbToClear / 256).
//
// Engine launch pack: (PxU32* frictionPatchCount, PxU32* edgeIndicesToClear,
// PxU32 nbToClear) -> argSizes {8,8,4} = 24-byte blob (count @16, tail padded
// to 8).
struct ClearFrictionPatchCountsAbiParams
{
    uint64_t frictionPatchCount;  // @0  PxU32*
    uint64_t edgeIndicesToClear;  // @8  PxU32*
    uint nbToClear;               // @16 (bytes 20..23 blob tail padding)
    uint abiPad0;                 // @20 (never read)
}                                 // = 24
[[vk::push_constant]]
ConstantBuffer<ClearFrictionPatchCountsAbiParams> gClearFpc;

[shader("compute")]
[numthreads(256, 1, 1)]
void clearFrictionPatchCounts(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= gClearFpc.nbToClear)
        return;

    Ptr<uint> frictionPatchCount = Ptr<uint>(gClearFpc.frictionPatchCount);
    Ptr<uint> edgeIndicesToClear = Ptr<uint>(gClearFpc.edgeIndicesToClear);

    const uint indexToClear = edgeIndicesToClear[dtid.x];
    frictionPatchCount[indexToClear] = 0u;
}

// ---------------------------------------------------------------------------
// constraint1DBlockPrePrepLaunch (P5 Step 5 BDA engine ABI)
// ---------------------------------------------------------------------------
// Port of constraint1DBlockPrePrepLaunch() in constraintBlockPrePrep.cu — the
// JOINT (1D-constraint) half of the block pre-prep.  The engine launches it
// with the TOTAL batch count (contacts INCLUDED:
// PxgGpuContext::doConstraintPrePrepCommon passes mNumConstraintBatches which
// sums joint AND contact batches), so it runs for EVERY scene with any
// constraint batches; warps whose batch is a contact type (eCONTACT /
// eARTICULATION_CONTACT) do nothing, exactly like the .cu's
// `if (conType == eCONSTRAINT_1D || eARTICULATION_CONSTRAINT_1D)` guard.
// For 1D batches it stages the per-batch header + per-lane body/remap data
// into the PxgBlockConstraintBatch and copies the joint prep rows
// (PxgConstraintData + Px1DConstraint) into the block-SoA prep pools, then
// allocates the batch's row range from desc->sharedJointRowIndex.
//
// Same 16-byte pack as constraintContactBlockPrePrepLaunch:
//   (PxgPrePrepDesc*, PxgSolverSharedDesc<IterativeSolveData>*) -> {8,8}.
// block = CONSTRAINT_PREPREP_BLOCK (128) = 4 warps; warpIndex selects the
// batch.
//
// llvmpipe deviations (behaviour-preserving):
//   - the warpCopy of the desc through shared memory becomes direct BDA reads;
//   - warpScan<MaxOpPxU32>(numRows) becomes a groupshared 32-lane butterfly
//     max per warp under BLOCK-uniform barriers (hoisted out of the per-warp
//     type guard; inactive/contact warps reduce zeros);
//   - the lane-31 atomicAdd on sharedJointRowIndex/sharedArticulationResponseIndex
//     is an InterlockedAdd on the desc field (as in the contact twin).

// mirrors PxgConstraintData (PxgConstraintPrep.h), 64 bytes
struct PxgConstraintDataGpu
{
    float4 mInvMassScale;              // @0
    float4 mRaWorld_linBreakForceW;    // @16
    float4 mRbWorld_angBreakForceW;    // @32
    uint4 mNumRows_Flags_StartIndex;   // @48
}                                      // = 64 (array stride)

// mirrors Px1DConstraint (PxConstraintDesc.h), 80 bytes PX_ALIGN(16)
struct Px1DConstraintGpu
{
    float4 linear0XYZ_geometricErrorW; // @0
    float4 angular0XYZ_velocityTargetW;// @16
    float4 linear1XYZ_minImpulseW;     // @32
    float4 angular1XYZ_maxImpulseW;    // @48
    float mods_restitution;            // @64 (mods.bounce.restitution / spring.stiffness)
    float mods_velocityThreshold;      // @68 (mods.bounce.velocityThreshold / spring.damping)
    uint flags_solveHint;              // @72 PxU16 flags | PxU16 solveHint << 16
    uint pad;                          // @76
}                                      // = 80 (array stride)

// per-warp numRows max scratch: 128 threads = 4 warps
groupshared uint gsJointNumRows[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void constraint1DBlockPrePrepLaunch(uint3 groupId : SV_GroupID,
                                    uint3 groupThreadId : SV_GroupThreadID)
{
    const uint tid = groupThreadId.x;
    const uint blockStride = 128u / 32u;
    const uint warpIndexInBlock = tid / 32u;
    const uint threadIndexInWarp = tid & 31u;
    const uint warpIndex = groupId.x * blockStride + warpIndexInBlock; // the batch index a

    Ptr<PxgPrePrepDescGpu> desc = Ptr<PxgPrePrepDescGpu>(gCppLaunch.prePrepDesc);
    Ptr<PxgSolverSharedDescIterGpu> sharedDesc = Ptr<PxgSolverSharedDescIterGpu>(gCppLaunch.sharedDesc);

    const uint numBatches = desc[0].numBatches;
    const uint numArtiBatches = desc[0].numArtiBatches;
    const uint numArtiStaticBatches = desc[0].numArtiStaticBatches;
    const uint numArtiSelfBatches = desc[0].numArtiSelfBatches;
    const uint totalDynamicBatches = numBatches + numArtiBatches;
    const uint totalBatches =
        totalDynamicBatches + numArtiStaticBatches + numArtiSelfBatches + desc[0].numStaticBatches;

    const uint nbElemsPerBody = desc[0].nbElementsPerBody;
    const uint nbArticulations = desc[0].mTotalActiveArticulations;
    const uint nbSlabs = desc[0].mTotalSlabs;

    Ptr<PxgBodySim> bodySims = Ptr<PxgBodySim>(sharedDesc[0].mBodySimBufferDeviceData);

    const uint a = warpIndex;
    const bool warpActive = a < totalBatches;

    uint conType = 0xFFFFFFFFu;
    uint descStride = 0u;
    uint batchIndex = 0u;
    uint headerWord0 = 0u;
    uint headerMask = 0u;
    uint startPartitionIndex = 0u;
    bool is1D = false;

    Ptr<PxgBlockConstraintBatchGpu> blockBatches = Ptr<PxgBlockConstraintBatchGpu>(desc[0].blockBatches);

    if (warpActive)
    {
        Ptr<PxgConstraintBatchHeaderGpu> headers = Ptr<PxgConstraintBatchHeaderGpu>(desc[0].mBatchHeaders);
        headerWord0 = headers[a].mDescStride_constraintType;
        batchIndex = headers[a].mConstraintBatchIndex;
        startPartitionIndex = headers[a].mStartPartitionIndex;
        headerMask = headers[a].mask;
        conType = headerWord0 >> 16;       // PxgConstraintBatchHeader::constraintType
        descStride = headerWord0 & 0xFFFFu;
        // eCONSTRAINT_1D = 0, eARTICULATION_CONSTRAINT_1D = 2
        is1D = (conType == 0u) || (conType == 2u);
    }

    uint numRows = 0u;

    if (warpActive && is1D)
    {
        // Copy across data from batchHeader (CUDA: all lanes store the same
        // values; the port stores from lane 0)
        if (threadIndexInWarp == 0u)
        {
            blockBatches[a].descStride_constraintType = headerWord0;
            blockBatches[a].mConstraintBatchIndex = batchIndex;
            blockBatches[a].mStartPartitionIndex = startPartitionIndex;
            blockBatches[a].mask = headerMask;
        }

        const uint outputRowStartIndex = batchIndex * MAX_CONSTRAINT_ROWS;

        if (threadIndexInWarp < descStride)
        {
            const uint uniqueId = conType == 0u
                ? (Ptr<uint>(desc[0].mConstraintUniqueIndices))[startPartitionIndex + threadIndexInWarp]
                : (Ptr<uint>(desc[0].mArtiConstraintUniqueIndices))[startPartitionIndex + threadIndexInWarp];

            Ptr<PartitionNodeDataGpu> partitionNodeData =
                Ptr<PartitionNodeDataGpu>(desc[0].mPartitionNodeData);
            Ptr<PartitionIndexDataGpu> partitionIndexData =
                Ptr<PartitionIndexDataGpu>(desc[0].mPartitionIndices);
            Ptr<uint> solverBodyIndices = Ptr<uint>(desc[0].solverBodyIndices);

            const PartitionNodeDataGpu nodeData = partitionNodeData[uniqueId];
            const uint2 igNodeIndexA = nodeData.mNodeIndex0;
            const uint2 igNodeIndexB = nodeData.mNodeIndex1;

            const bool isArticulationA = (igNodeIndexA.y & 1u) != 0u; // isArticulation()
            const bool isArticulationB = (igNodeIndexB.y & 1u) != 0u;

            const uint nodeIndexA = igNodeIndexA.x; // PxNodeIndex::index()
            const uint nodeIndexB = igNodeIndexB.x;

            const uint nextIndexA = nodeData.mNextIndex0;
            const uint nextIndexB = nodeData.mNextIndex1;

            const uint solverBodyIndexA =
                nodeIndexA == PX_INVALID_NODE ? 0u : solverBodyIndices[nodeIndexA];
            const uint solverBodyIndexB =
                nodeIndexB == PX_INVALID_NODE ? 0u : solverBodyIndices[nodeIndexB];

            blockBatches[a].bodyAIndex[threadIndexInWarp] = solverBodyIndexA;
            blockBatches[a].bodyBIndex[threadIndexInWarp] = solverBodyIndexB;
            blockBatches[a].bodyANodeIndex[threadIndexInWarp] = igNodeIndexA;
            blockBatches[a].bodyBNodeIndex[threadIndexInWarp] = igNodeIndexB;

            const uint currentPartition = partitionIndexData[uniqueId].partition_patch_ctype & 0xFFFFu;
            const uint maxConstraintPartitions = desc[0].mMaxConstraintPartitions;
            const uint slabId = currentPartition / maxConstraintPartitions;
            blockBatches[a].slabId[threadIndexInWarp] = slabId;

            Ptr<uint> partitionStartBatch = Ptr<uint>(desc[0].mPartitionstartBatchIndices);
            Ptr<uint> partitionArtiStartBatch = Ptr<uint>(desc[0].mPartitionArtiStartBatchIndices);
            Ptr<uint> partitionJointCounts = Ptr<uint>(desc[0].mPartitionJointCounts);
            Ptr<uint> partitionArtiJointCounts = Ptr<uint>(desc[0].mPartitionArtiJointCounts);
            Ptr<uint> solverBodyReferences = Ptr<uint>(desc[0].mSolverBodyReferences);
            Ptr<uint> slabMaskWords = Ptr<uint>(sharedDesc[0].articulationSlabMask); // uint4 elements

            // Static articulation constraints don't have all the data in the
            // partitioning assigned to be able to compute the data below
            if (a < totalDynamicBatches)
            {
                if (!isArticulationA)
                {
                    blockBatches[a].remappedBodyAIndex[threadIndexInWarp] = computeRemapIndexRigidBody(
                        (nextIndexA & 1u) != 0u, partitionStartBatch, partitionArtiStartBatch,
                        partitionJointCounts, partitionArtiJointCounts, partitionIndexData[nextIndexA >> 1],
                        solverBodyReferences, currentPartition, maxConstraintPartitions,
                        solverBodyIndexA, desc[0].mActiveBodyStartOffset, numBatches, numArtiBatches,
                        nbElemsPerBody, nbSlabs);
                }
                else
                {
                    blockBatches[a].remappedBodyAIndex[threadIndexInWarp] =
                        bodySims[nodeIndexA].articulationRemapId;

                    const uint partitionId = currentPartition & (maxConstraintPartitions - 1u);
                    const uint ind = solverBodyIndexA + (slabId + partitionId * nbSlabs) * nbArticulations;

                    slabMaskWords[ind * 4u + 0u] = igNodeIndexA.y >> 1; // articulationLinkId()
                    slabMaskWords[ind * 4u + 1u] = a * 32u * 2u * nbElemsPerBody + threadIndexInWarp;
                }

                if (!isArticulationB)
                {
                    blockBatches[a].remappedBodyBIndex[threadIndexInWarp] = computeRemapIndexRigidBody(
                        (nextIndexB & 1u) != 0u, partitionStartBatch, partitionArtiStartBatch,
                        partitionJointCounts, partitionArtiJointCounts, partitionIndexData[nextIndexB >> 1],
                        solverBodyReferences, currentPartition, maxConstraintPartitions,
                        solverBodyIndexB, desc[0].mActiveBodyStartOffset, numBatches, numArtiBatches,
                        nbElemsPerBody, nbSlabs);
                }
                else
                {
                    blockBatches[a].remappedBodyBIndex[threadIndexInWarp] =
                        bodySims[nodeIndexB].articulationRemapId;

                    const uint partitionId = currentPartition & (maxConstraintPartitions - 1u);
                    const uint ind = solverBodyIndexB + (slabId + partitionId * nbSlabs) * nbArticulations;

                    slabMaskWords[ind * 4u + 2u] = igNodeIndexB.y >> 1;
                    slabMaskWords[ind * 4u + 3u] = (a * 32u * 2u + 32u) * nbElemsPerBody + threadIndexInWarp;
                }
            }
            else
            {
                if (isArticulationA)
                    blockBatches[a].remappedBodyAIndex[threadIndexInWarp] =
                        bodySims[nodeIndexA].articulationRemapId;
                if (isArticulationB)
                    blockBatches[a].remappedBodyBIndex[threadIndexInWarp] =
                        bodySims[nodeIndexB].articulationRemapId;
            }

            // KS - doubles up as joint indices for joints!
            const uint index = (Ptr<uint>(desc[0].mNpOutputIndices))[uniqueId];

            const bool hasArticulations = isArticulationA || isArticulationB;
            Ptr<PxgConstraintDataGpu> cdata = hasArticulations
                ? Ptr<PxgConstraintDataGpu>(desc[0].artiConstraintData)
                : Ptr<PxgConstraintDataGpu>(desc[0].constraintData);
            Ptr<Px1DConstraintGpu> tRows = hasArticulations
                ? Ptr<Px1DConstraintGpu>(desc[0].artiConstraintRows)
                : Ptr<Px1DConstraintGpu>(desc[0].constraintRows);

            Ptr<PxgBlockConstraint1DData> gBlockPrepData =
                Ptr<PxgBlockConstraint1DData>(desc[0].blockPrepData);
            Ptr<PxgBlockConstraint1DVelocities> gBlockPrepVelocityData =
                Ptr<PxgBlockConstraint1DVelocities>(desc[0].blockPrepVelocityData);
            Ptr<PxgBlockConstraint1DParameters> gBlockPrepParameterData =
                Ptr<PxgBlockConstraint1DParameters>(desc[0].blockPrepParameterData);

            const float4 raWorld_linBreakForceW = cdata[index].mRaWorld_linBreakForceW;
            const float4 rbWorld_angBreakForceW = cdata[index].mRbWorld_angBreakForceW;
            const uint4 mNumRows_Flags_StartIndex = cdata[index].mNumRows_Flags_StartIndex;
            numRows = mNumRows_Flags_StartIndex.x;
            const uint rowStartIndex = mNumRows_Flags_StartIndex.z;

            gBlockPrepData[batchIndex].mInvMassScale[threadIndexInWarp] = cdata[index].mInvMassScale;
            gBlockPrepData[batchIndex].mRAWorld_linBreakForce[threadIndexInWarp] = raWorld_linBreakForceW;
            gBlockPrepData[batchIndex].mRBWorld_AngBreakForce[threadIndexInWarp] = rbWorld_angBreakForceW;
            gBlockPrepData[batchIndex].mFlags[threadIndexInWarp] = mNumRows_Flags_StartIndex.y;
            gBlockPrepData[batchIndex].mNumRows[threadIndexInWarp] = numRows;

            for (uint i = 0; i < numRows; ++i)
            {
                const Px1DConstraintGpu constraint1D = tRows[rowStartIndex + i];

                gBlockPrepVelocityData[outputRowStartIndex + i].linear0XYZ_geometricErrorW[threadIndexInWarp] =
                    constraint1D.linear0XYZ_geometricErrorW;
                gBlockPrepVelocityData[outputRowStartIndex + i].angular0XYZ_velocityTargetW[threadIndexInWarp] =
                    constraint1D.angular0XYZ_velocityTargetW;
                gBlockPrepVelocityData[outputRowStartIndex + i].linear1XYZ_minImpulseW[threadIndexInWarp] =
                    constraint1D.linear1XYZ_minImpulseW;
                gBlockPrepVelocityData[outputRowStartIndex + i].angular1XYZ_maxImpulseW[threadIndexInWarp] =
                    constraint1D.angular1XYZ_maxImpulseW;

                gBlockPrepParameterData[outputRowStartIndex + i].flags[threadIndexInWarp] =
                    constraint1D.flags_solveHint & 0xFFFFu;
                gBlockPrepParameterData[outputRowStartIndex + i].springStiffness_restitution[threadIndexInWarp] =
                    constraint1D.mods_restitution;
                gBlockPrepParameterData[outputRowStartIndex + i].springDamping_velocityThreshold[threadIndexInWarp] =
                    constraint1D.mods_velocityThreshold;
                gBlockPrepParameterData[outputRowStartIndex + i].solveHint[threadIndexInWarp] =
                    constraint1D.flags_solveHint >> 16;
            }
        }
    }

    // warpScan<MaxOpPxU32>(numRows): per-warp max through groupshared under
    // BLOCK-uniform barriers (hoisted out of the per-warp type guard; contact
    // and inactive warps reduce zeros and skip the atomic below).
    GroupMemoryBarrierWithGroupSync();
    gsJointNumRows[tid] = numRows;
    GroupMemoryBarrierWithGroupSync();
    for (uint radius = 16u; radius > 0u; radius >>= 1)
    {
        const uint partner = tid ^ radius; // stays inside the warp's 32 lanes
        const uint other = gsJointNumRows[partner];
        GroupMemoryBarrierWithGroupSync();
        gsJointNumRows[tid] = max(gsJointNumRows[tid], other);
        GroupMemoryBarrierWithGroupSync();
    }
    const uint warpMaxRows = gsJointNumRows[tid];

    // we need to update the sharedJointRowIndex
    if (warpActive && is1D && threadIndexInWarp == (32u - 1u))
    {
        uint startRow;
        InterlockedAdd(desc[0].sharedJointRowIndex, warpMaxRows, startRow);
        blockBatches[a].startConstraintIndex = startRow;

        if (conType == 2u) // eARTICULATION_CONSTRAINT_1D
        {
            uint startResponse;
            InterlockedAdd(desc[0].sharedArticulationResponseIndex, warpMaxRows, startResponse);
            blockBatches[a].mArticulationResponseIndex = startResponse;
        }
    }
}
