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

// broadphaseAggregateFinalize.slang
//
// Slang port of three small aggregate finalization kernels
// (gpubroadphase/src/CUDA/aggregate.cu):
//   - aggCopyReports: copy the found/lost aggregate pair reports into their
//     mapped output buffers (clamped to the report cap) and raise the found/lost
//     overflow flags.
//   - removeAggPairsStage3: return the just-removed pair slots to the free list
//     (numFreeIndices += nbRemoved).
//   - markAddedAndDeletedAggregatedBounds: setEmpty the integer bounds of
//     added/deleted aggregated shapes so the next broad-phase ignores them — old
//     bounds for both, new bounds only for deleted.  Uses a 2-row group
//     (row 0 = added, row 1 = deleted).

// PxgBroadPhasePair is two PxU32s, so element count = pairs * 2.
static const uint PXG_BP_PAIR_UINTS = 2u;
// PxgIntegerAABB::setEmpty(): mins = 0xff7fffff, maxes = 0x00800000.
static const uint IAABB_EMPTY_MIN = 0xff7fffffu;
static const uint IAABB_EMPTY_MAX = 0x00800000u;

struct FinalizeParams
{
    uint nbCreatedPairs;     // aggDesc->sharedFoundPairIndex
    uint nbLostPairs;        // aggDesc->sharedLostPairIndex
    uint maxFoundLostPairs;  // aggDesc->max_found_lost_pairs
    uint nbRemoved;          // aggDesc->nbRemoved
    uint numAdded;           // numAddedAggregatedBounds
    uint numDeleted;         // numDeletedAggregatedBounds
    uint gridDimX;
}

[[vk::push_constant]]
ConstantBuffer<FinalizeParams> gParams;

RWStructuredBuffer<uint> gFoundReports;   // binding 0  aggDesc->foundPairReport [n][2]
RWStructuredBuffer<uint> gLostReports;    // binding 1  aggDesc->lostPairReport  [n][2]
RWStructuredBuffer<uint> gFoundReportMap; // binding 2  aggDesc->foundPairReportMap
RWStructuredBuffer<uint> gLostReportMap;  // binding 3  aggDesc->lostPairReportMap
RWStructuredBuffer<uint> gOverflowFlags;  // binding 4  {found_pairs_overflow, lost_pairs_overflow}
RWStructuredBuffer<uint> gFreeList;       // binding 5  {numFreeIndices, maxIndex}
RWStructuredBuffer<uint> gNewBounds;      // binding 6  bpDesc->newIntegerBounds [n][6]
RWStructuredBuffer<uint> gOldBounds;      // binding 7  bpDesc->oldIntegerBounds [n][6]
RWStructuredBuffer<uint> gAddedHandles;   // binding 8  addedAggregatedBounds
RWStructuredBuffer<uint> gDeletedHandles; // binding 9  deletedAggregatedBounds

void setEmptyIAABB(uint handle)
{
    const uint b = handle * 6u;
    gOldBounds[b + 0u] = IAABB_EMPTY_MIN;
    gOldBounds[b + 1u] = IAABB_EMPTY_MIN;
    gOldBounds[b + 2u] = IAABB_EMPTY_MIN;
    gOldBounds[b + 3u] = IAABB_EMPTY_MAX;
    gOldBounds[b + 4u] = IAABB_EMPTY_MAX;
    gOldBounds[b + 5u] = IAABB_EMPTY_MAX;
}

// ---------------------------------------------------------------------------
// aggCopyReports
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void aggCopyReports(uint3 dtid : SV_DispatchThreadID)
{
    const uint nbCreatedPairs = gParams.nbCreatedPairs;
    const uint nbLostPairs = gParams.nbLostPairs;
    const uint maxPairs = gParams.maxFoundLostPairs;

    const bool foundOverflow = nbCreatedPairs > maxPairs;
    const bool lostOverflow = nbLostPairs > maxPairs;

    if (dtid.x == 0u) // threadIdx.x == 0 && blockIdx.x == 0
    {
        gOverflowFlags[0] = gOverflowFlags[0] | (foundOverflow ? 1u : 0u); // |= foundOverflow
        gOverflowFlags[1] = lostOverflow ? 1u : 0u;                        // =  lostOverflow
    }

    const uint tNbCreatedPairs = foundOverflow ? maxPairs : nbCreatedPairs;
    const uint tNbLostPairs = lostOverflow ? maxPairs : nbLostPairs;
    const uint nbCreateElements = tNbCreatedPairs * PXG_BP_PAIR_UINTS;
    const uint nbLostElements = tNbLostPairs * PXG_BP_PAIR_UINTS;
    const uint stride = gParams.gridDimX * 256u;

    for (uint i = dtid.x; i < nbCreateElements; i += stride)
        gFoundReportMap[i] = gFoundReports[i];

    for (uint i = dtid.x; i < nbLostElements; i += stride)
        gLostReportMap[i] = gLostReports[i];
}

// ---------------------------------------------------------------------------
// removeAggPairsStage3
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void removeAggPairsStage3(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x == 0u)
        gFreeList[0] = gFreeList[0] + gParams.nbRemoved;
}

// ---------------------------------------------------------------------------
// markAddedAndDeletedAggregatedBounds (row 0 = added, row 1 = deleted)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 2, 1)]
void markAddedAndDeletedAggregatedBounds(uint3 dtid : SV_DispatchThreadID)
{
    const uint row = dtid.y; // 0 -> added, 1 -> deleted
    const uint maxCount = (row == 1u) ? gParams.numDeleted : gParams.numAdded;
    const uint gi = dtid.x;

    if (gi < maxCount)
    {
        const uint handle = (row == 1u) ? gDeletedHandles[gi] : gAddedHandles[gi];

        if (row == 1u) // invalidate the new bounds too for deleted bounds
        {
            const uint b = handle * 6u;
            gNewBounds[b + 0u] = IAABB_EMPTY_MIN;
            gNewBounds[b + 1u] = IAABB_EMPTY_MIN;
            gNewBounds[b + 2u] = IAABB_EMPTY_MIN;
            gNewBounds[b + 3u] = IAABB_EMPTY_MAX;
            gNewBounds[b + 4u] = IAABB_EMPTY_MAX;
            gNewBounds[b + 5u] = IAABB_EMPTY_MAX;
        }

        setEmptyIAABB(handle);
    }
}
