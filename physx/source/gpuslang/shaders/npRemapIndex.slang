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

// npRemapIndex.slang
//
// Slang port of computeRemapIndexRigidBody
// (gpusolver/src/CUDA/constraintPrepShared.cuh L51-161), the rigid-body
// output-slot remap used by both pre-prep launch kernels: the
// PxgEdgeType-switched batch index (partition start + optional joint-count
// offset + partitionEntryIndex / PXG_BATCH_SIZE, articulation types offset
// by totalRigidBatches), the batch expansion
// (index * PXG_BATCH_SIZE * 2) * nbElemsPerBody + (entry & 31) with the
// second-body PXG_BATCH_SIZE * nbElemsPerBody offset, and the slab-change
// branch (bodyId >= activeBodyOffset and a partition-group change or
// partitionIndex <= currPartition): referenceId = nbSlabs *
// (bodyId - activeBodyOffset) + slabId, the solverBodyReferences write of
// the pre-slab index (surfaced as an explicit output), the referenceId
// spread (refId & ~31) * nbElemsPerBody + (refId & 31), and the final
// accumulation-buffer index (totalArtic + totalRigid) * 64 * nbElems +
// spreadRefId.  Pure PxU32 arithmetic -- everything verifies BITWISE.
// CUDA's `index` is uninitialized for cType > 3 (UB, unreachable -- the
// enum has 4 values); the port initializes it to 0 and documents this.
// PXG_BATCH_SIZE = 32 (PxgDynamicsConfiguration.h L32).

static const uint MAX_PART = 8u;

struct RmIn
{
    uint4 m0;              // isSecondBody, cType, partitionIndex, partitionEntryIndex
    uint4 m1;              // currPartition, maxNbPartitions, bodyId, activeBodyOffset
    uint4 m2;              // totalRigidBatches, totalArticBatches, nbElemsPerBody, nbSlabs
    uint4 partStart[2];    // partitionStartIndices[8]
    uint4 artiStart[2];    // partitionArticStartIndices[8]
    uint4 jointCounts[2];  // partitionJointCounts[8]
    uint4 artiJointCounts[2];  // partitionArticulationJointCounts[8]
}

struct RmOut
{
    uint4 o0;  // finalIndex, wrote (0/1), referenceId (pre-spread), writtenValue
    uint4 o1;  // spreadReferenceId, 0, 0, 0
}

struct RmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<RmParams> gParams;

RWStructuredBuffer<RmIn> gIn;    // binding 0
RWStructuredBuffer<RmOut> gOut;  // binding 1

uint tbl(uint4 t[2], uint i) { return i < 4u ? t[0][i] : t[1][i - 4u]; }

[shader("compute")]
[numthreads(64, 1, 1)]
void remapIndex(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const RmIn s = gIn[tid];

    const uint isSecondBody = s.m0.x;
    const uint cType = s.m0.y;
    const uint partitionIndex = s.m0.z;
    const uint partitionEntryIndex = s.m0.w;
    const uint currPartition = s.m1.x;
    const uint maxNbPartitions = s.m1.y;
    const uint bodyId = s.m1.z;
    const uint activeBodyOffset = s.m1.w;
    const uint totalRigidBatches = s.m2.x;
    const uint totalArticBatches = s.m2.y;
    const uint nbElemsPerBody = s.m2.z;
    const uint nbSlabs = s.m2.w;

    uint index = 0u;  // CUDA leaves this uninitialized for cType > 3 (unreachable)
    switch (cType)
    {
    case 0u:  // eCONTACT_MANAGER
        index = tbl(s.partStart, partitionIndex) + tbl(s.jointCounts, partitionIndex) +
                partitionEntryIndex / 32u;
        break;
    case 1u:  // eCONSTRAINT
        index = tbl(s.partStart, partitionIndex) + partitionEntryIndex / 32u;
        break;
    case 2u:  // eARTICULATION_CONTACT
        index = totalRigidBatches + tbl(s.artiStart, partitionIndex) +
                tbl(s.artiJointCounts, partitionIndex) + partitionEntryIndex / 32u;
        break;
    case 3u:  // eARTICULATION_CONSTRAINT
        index = totalRigidBatches + tbl(s.artiStart, partitionIndex) +
                partitionEntryIndex / 32u;
        break;
    default:
        break;
    }

    // batch expansion (batchMask = PXG_BATCH_SIZE - 1)
    index = (index * 32u * 2u) * nbElemsPerBody + (partitionEntryIndex & 31u);
    if (isSecondBody != 0u)
        index += 32u * nbElemsPerBody;

    uint wrote = 0u, refId = 0u, written = 0u, spreadRefId = 0u;

    if (bodyId >= activeBodyOffset)
    {
        if ((partitionIndex & (~(maxNbPartitions - 1u))) !=
                (currPartition & (~(maxNbPartitions - 1u))) ||
            partitionIndex <= currPartition)
        {
            // slab changed: introduce a new solver reference
            const uint slabId = partitionIndex / maxNbPartitions;
            uint referenceId = nbSlabs * (bodyId - activeBodyOffset) + slabId;

            refId = referenceId;
            written = index;  // solverBodyReferences[referenceId].mRemappedBodyIndex = index
            wrote = 1u;

            referenceId = (referenceId & (~31u)) * nbElemsPerBody + (referenceId & 31u);
            spreadRefId = referenceId;

            index = (totalArticBatches + totalRigidBatches) * 32u * 2u * nbElemsPerBody +
                    referenceId;
        }
    }

    RmOut o;
    o.o0 = uint4(index, wrote, refId, written);
    o.o1 = uint4(spreadRefId, 0u, 0u, 0u);
    gOut[tid] = o;
}
