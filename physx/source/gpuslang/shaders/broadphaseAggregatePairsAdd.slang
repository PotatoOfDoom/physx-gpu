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

// broadphaseAggregatePairsAdd.slang
//
// Slang port of the aggregate-pair add stages (gpubroadphase/src/CUDA/aggregate.cu):
//   - addAggPairsStage1: for each newly found aggregate pair, allocate a slot in
//     the aggPairs array — reusing a freed index (a hole below the high-water
//     mark) if one is available, otherwise appending past maxIndex — and write
//     the pair there (isNew=true, isDead=false), skipping slots past the cap.
//   - addAggPairsStage2: a single thread finalizes the free-buffer-list counts
//     (consume freed indices / grow the high-water mark) and raises the
//     aggregate-pair and found-pair overflow flags.
//
// Flat models: PxgAggregatePair -> 4 uints {actorHandle0, actorHandle1, isNew,
// isDead}; PxgFreeBufferList -> {numFreeIndices, maxIndex}; PxgBroadPhasePair ->
// 2 uints {mVolA, mVolB}; the stage-2 scalar outputs are packed into gScalarOut.

struct AddPairsParams
{
    uint nbFoundAggPairs;
    uint maxFoundLostPairs;  // aggDesc->max_found_lost_pairs
    uint maxAggPairs;        // aggDesc->max_agg_pairs
    uint gridDimX;
}

[[vk::push_constant]]
ConstantBuffer<AddPairsParams> gParams;

RWStructuredBuffer<uint> gFoundAggPairs; // binding 0  bpDesc->foundAggPairReport [nbFound][2] {mVolA,mVolB}
RWStructuredBuffer<uint> gFreeList;      // binding 1  {numFreeIndices, maxIndex}
RWStructuredBuffer<uint> gFreeIndices;   // binding 2  aggDesc->freeIndices
RWStructuredBuffer<uint> gAggPairs;      // binding 3  aggDesc->aggPairs [maxAggPairs][4]
RWStructuredBuffer<uint> gScalarOut;     // binding 4  stage-2 scalar outputs (see offsets)

// gScalarOut layout
static const uint S_AGG_OVERFLOW_COUNT = 0u; // aggPairOverflowCount (pre-clamp maxIndex)
static const uint S_AGG_OVERFLOW_FLAG  = 1u; // agg_pairs_overflow_flags
static const uint S_AGG_PAIR_COUNT     = 2u; // *aggPairCount (clamped maxIndex)
static const uint S_FOUND_OVERFLOW_FLAG = 3u; // found_pairs_overflow_flags
static const uint S_FOUND_CAND_COUNT   = 4u; // foundCandidatePairOverflowCount

// ---------------------------------------------------------------------------
// addAggPairsStage1
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void addAggPairsStage1(uint3 dtid : SV_DispatchThreadID)
{
    const uint nbFoundSafe = min(gParams.nbFoundAggPairs, gParams.maxFoundLostPairs);
    const uint numFreeIndices = gFreeList[0];
    const uint maxIndex = gFreeList[1];
    const uint maxAggPairs = gParams.maxAggPairs;
    const uint stride = gParams.gridDimX * 256u;

    for (uint i = dtid.x; i < nbFoundSafe; i += stride)
    {
        uint index = 0u;
        if (i < numFreeIndices)
        {
            const uint freeIndex = numFreeIndices - i - 1u;
            index = gFreeIndices[freeIndex];
        }
        else
        {
            const uint offset = i - numFreeIndices;
            index = maxIndex + offset;
        }

        if (index < maxAggPairs)
        {
            const uint o = index * 4u;
            gAggPairs[o + 0u] = gFoundAggPairs[i * 2u + 0u]; // mVolA
            gAggPairs[o + 1u] = gFoundAggPairs[i * 2u + 1u]; // mVolB
            gAggPairs[o + 2u] = 1u; // isNew  = true
            gAggPairs[o + 3u] = 0u; // isDead = false
        }
    }
}

// ---------------------------------------------------------------------------
// addAggPairsStage2
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void addAggPairsStage2(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x == 0u)
    {
        const uint nbFoundAggPairs = gParams.nbFoundAggPairs;
        const uint maxFL = gParams.maxFoundLostPairs;
        const uint maxAgg = gParams.maxAggPairs;

        const uint nbFoundAggPairsSafe = min(nbFoundAggPairs, maxFL);
        const uint nbFreeInds = gFreeList[0];

        uint newFreeInds = 0u;
        uint numToIncrement = 0u;
        if (nbFreeInds >= nbFoundAggPairsSafe)
            newFreeInds = nbFreeInds - nbFoundAggPairsSafe;
        else
            numToIncrement = nbFoundAggPairsSafe - nbFreeInds;

        gFreeList[0] = newFreeInds;

        uint maxIndex = gFreeList[1] + numToIncrement;

        gScalarOut[S_AGG_OVERFLOW_COUNT] = maxIndex; // pinned aggPairOverflowCount (pre-clamp)

        if (maxIndex > maxAgg)
        {
            gScalarOut[S_AGG_OVERFLOW_FLAG] = 1u;
            maxIndex = maxAgg;
        }

        gFreeList[1] = maxIndex;
        gScalarOut[S_AGG_PAIR_COUNT] = maxIndex;

        if (nbFoundAggPairs > maxFL)
        {
            gScalarOut[S_FOUND_OVERFLOW_FLAG] = 1u;
            gScalarOut[S_FOUND_CAND_COUNT] = nbFoundAggPairs;
        }
    }
}
