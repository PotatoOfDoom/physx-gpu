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

// npRigidSumBatches.slang
//
// Slang port of rigidSumInternalContactAndJointBatches1 (constraintBlock-
// PrePrep.cu L1472-1604) and rigidSumInternalContactAndJointBatches2
// (L1607-1956), the static contact/joint batch accumulation pair.
//
// Every quantity in both kernels is unsigned-integer add/max, so all the
// warp/block collectives (warpReduction Add/Max, the inclusive warpScan with
// exclusive stores, __ballot/__popc lane offsets, the shared-memory
// accumulators guarded by __syncthreads pairs) are associative reductions
// over fixed index domains; the port computes each one as a deterministic
// pure-function loop over the same domain -- bitwise identical, with no
// scheduling assumptions.  CUDA thread (block b, tid t) maps to one Slang
// thread; the per-iteration accumulators (sContactUniqueIndicesAccum etc.,
// which CUDA reads BEFORE the thread-15 update between the two
// __syncthreads) become prefix sums over the previous iterations' whole-
// block totals.
//
// Kernel 1: per-block totals of the static contact/joint counts over the
// block's strided workIndex domain, written to the four 32-entry temp
// arrays.  The unique-index arrays are plain sums; the header arrays are
// SUMS OF PER-WARP MAXES (each warp's Max-reduction at L1544 feeds warp 0's
// Add-reduction at L1574) accumulated over the iterations -- NOT the block
// max.
//
// Kernel 2: warp 0's exclusive block histograms (inclusive warpScan minus
// input); the blockIdx-0/thread-31 totals writes; per iteration the 16
// warp sums, each warp's exclusive offsets, the running accumulators, the
// per-body start indices (blockOffset, blockOffset + warp maxContact), and
// the interleaved unique-id scatter with one PxgConstraintBatchHeader image
// (mDescStride | constraintType<<16, mConstraintBatchIndex,
// mStartPartitionIndex, ballot mask) per (warp, round), contact rounds
// first, then joints; eCONSTRAINT_1D = 0, eCONTACT = 1
// (PxgSolverConstraintDesc.h L111-117).
//
// PxgKernelBlockDim::COMPUTE_STATIC_CONTACT_CONSTRAINT_COUNT = 512
// (16 warps), 32 blocks.

static const uint NB_BLOCKS = 32u;
static const uint BLOCK_DIM = 512u;
static const uint WARPS = 16u;

struct RsParams { uint nbBodies; }

[[vk::push_constant]]
ConstantBuffer<RsParams> gParams;

RWStructuredBuffer<uint> gContactCount;   // binding 0: mRigidStaticContactCounts
RWStructuredBuffer<uint> gJointCount;     // binding 1: mRigidStaticConstraintCounts
RWStructuredBuffer<uint> gTemp;           // binding 2: 4 x 32 temp block sums
                                          //  [0..31] contactUnique, [32..63] jointUnique,
                                          //  [64..95] contactHeader, [96..127] jointHeader
RWStructuredBuffer<uint> gCfg;            // binding 3: [0] contactUniqueIndexOffset,
                                          //  [1] jointUniqueIndexOffset, [2] batchOffset,
                                          //  [3] staticContactBatchOffset,
                                          //  [4] staticJointBatchOffset,
                                          //  [5] numContactBatches, [6] num1dConstraintBatches
RWStructuredBuffer<uint> gContactIds;     // binding 4: mRigidStaticContactIndices (nbBodies-strided)
RWStructuredBuffer<uint> gJointIds;       // binding 5: mRigidStaticConstraintIndices
RWStructuredBuffer<uint> gStartIndices;   // binding 6: contactStaticStartIndices[nb] ++ jointStaticStartIndices[nb]
RWStructuredBuffer<uint> gOutContactIds;  // binding 7: mContactUniqueIndices
RWStructuredBuffer<uint> gOutJointIds;    // binding 8: mConstraintUniqueIndices
RWStructuredBuffer<uint4> gHeaders;       // binding 9: batchHeaders image
RWStructuredBuffer<uint> gContactBatchIndices;  // binding 10
RWStructuredBuffer<uint> gJointBatchIndices;    // binding 11
RWStructuredBuffer<uint> gTotals;         // binding 12: [0] numStaticBatches (prePrepDesc),
                                          //  [1] numStaticContactBatches,
                                          //  [2] numStatic1dConstraintBatches,
                                          //  [3] numStaticBatches (prepDesc),
                                          //  [4] mStaticRigidBatchCount

uint iterationsPerBlock(uint nbBodies)
{
    const uint totalBlockRequired = (nbBodies + (BLOCK_DIM - 1u)) / BLOCK_DIM;
    return (totalBlockRequired + (NB_BLOCKS - 1u)) / NB_BLOCKS;
}

uint countAt(uint isJoint, uint workIndex, uint nbBodies)
{
    if (workIndex >= nbBodies)
        return 0u;
    return isJoint != 0u ? gJointCount[workIndex] : gContactCount[workIndex];
}

// ---- kernel 1: per-block sums/maxes (one lane per CUDA block) ----

[shader("compute")]
[numthreads(32, 1, 1)]
void rsBatches1(uint3 dtid : SV_DispatchThreadID)
{
    const uint blockId = dtid.x;
    if (blockId >= NB_BLOCKS) return;
    const uint nbBodies = gParams.nbBodies;
    const uint numIter = iterationsPerBlock(nbBodies);

    // unique-index arrays: plain sums; header arrays: sums of PER-WARP maxes
    uint sumC = 0u, sumJ = 0u, sumMaxC = 0u, sumMaxJ = 0u;
    for (uint i = 0u; i < numIter; ++i)
    {
        for (uint w = 0u; w < WARPS; ++w)
        {
            uint wMaxC = 0u, wMaxJ = 0u;
            for (uint l = 0u; l < 32u; ++l)
            {
                const uint workIndex = i * BLOCK_DIM + (w * 32u + l) + numIter * blockId * BLOCK_DIM;
                const uint c = countAt(0u, workIndex, nbBodies);
                const uint j = countAt(1u, workIndex, nbBodies);
                sumC += c;
                sumJ += j;
                wMaxC = c > wMaxC ? c : wMaxC;
                wMaxJ = j > wMaxJ ? j : wMaxJ;
            }
            sumMaxC += wMaxC;
            sumMaxJ += wMaxJ;
        }
    }
    gTemp[blockId] = sumC;
    gTemp[32u + blockId] = sumJ;
    gTemp[64u + blockId] = sumMaxC;
    gTemp[96u + blockId] = sumMaxJ;
}

// ---- kernel 2 helpers (pure functions of the inputs) ----

// warp sum/max of one warp's 32 counts at iteration i of block b
uint warpAgg(uint isJoint, uint isMax, uint b, uint i, uint w, uint nbBodies, uint numIter)
{
    uint acc = 0u;
    for (uint l = 0u; l < 32u; ++l)
    {
        const uint workIndex = i * BLOCK_DIM + (w * 32u + l) + numIter * b * BLOCK_DIM;
        const uint v = countAt(isJoint, workIndex, nbBodies);
        if (isMax != 0u)
            acc = v > acc ? v : acc;
        else
            acc += v;
    }
    return acc;
}

// exclusive prefix over the first `upTo` warps of block b at iteration i
uint warpPrefix(uint isJoint, uint isMax, uint b, uint i, uint upTo, uint nbBodies, uint numIter)
{
    uint acc = 0u;
    for (uint w = 0u; w < upTo; ++w)
        acc += warpAgg(isJoint, isMax, b, i, w, nbBodies, numIter);
    return acc;
}

// the shared accumulator value READ at iteration i (sum of the previous
// iterations' whole-block totals; CUDA guards the read/update with the
// __syncthreads pair at L1856-1868)
uint accumBefore(uint isJoint, uint isMax, uint b, uint i, uint nbBodies, uint numIter)
{
    uint acc = 0u;
    for (uint a = 0u; a < i; ++a)
        acc += warpPrefix(isJoint, isMax, b, a, WARPS, nbBodies, numIter);
    return acc;
}

// exclusive block histogram over gTemp array `arr` (warp 0's warpScan store)
uint blockHistogram(uint arr, uint b)
{
    uint acc = 0u;
    for (uint p = 0u; p < b; ++p)
        acc += gTemp[arr * 32u + p];
    return acc;
}

// ballot over one warp: bit l set iff count(lane l) > round
uint roundMask(uint isJoint, uint b, uint i, uint w, uint round, uint nbBodies, uint numIter)
{
    uint mask = 0u;
    for (uint l = 0u; l < 32u; ++l)
    {
        const uint workIndex = i * BLOCK_DIM + (w * 32u + l) + numIter * b * BLOCK_DIM;
        if (countAt(isJoint, workIndex, nbBodies) > round)
            mask |= (1u << l);
    }
    return mask;
}

// sum of popc(roundMask) over rounds [0, upTo)
uint strideBefore(uint isJoint, uint b, uint i, uint w, uint upTo, uint nbBodies, uint numIter)
{
    uint acc = 0u;
    for (uint r = 0u; r < upTo; ++r)
        acc += countbits(roundMask(isJoint, b, i, w, r, nbBodies, numIter));
    return acc;
}

// ---- kernel 2: one Slang thread per CUDA thread (32 x 512) ----

[shader("compute")]
[numthreads(64, 1, 1)]
void rsBatches2(uint3 dtid : SV_DispatchThreadID)
{
    const uint gt = dtid.x;
    if (gt >= NB_BLOCKS * BLOCK_DIM) return;
    const uint b = gt / BLOCK_DIM;       // blockIdx.x
    const uint tid = gt % BLOCK_DIM;     // threadIdx.x
    const uint w = tid / 32u;            // warpIndex
    const uint lane = tid & 31u;

    const uint nbBodies = gParams.nbBodies;
    const uint numIter = iterationsPerBlock(nbBodies);

    const uint contactUniqueIndexOffset = gCfg[0];
    const uint jointUniqueIndexOffset = gCfg[1];
    const uint batchOffset = gCfg[2];
    const uint staticContactBatchOffset = gCfg[3];
    const uint staticJointBatchOffset = gCfg[4];
    const uint numContactBatches = gCfg[5];
    const uint numConstraintBatches = gCfg[6];

    // blockIdx 0, threadIdx 31: the run-total writes (inclusive lane-31 scan
    // values = full 32-block sums of the header temp arrays)
    if (b == 0u && tid == 31u)
    {
        uint contactHeaderTotal = 0u, jointHeaderTotal = 0u;
        for (uint p = 0u; p < 32u; ++p)
        {
            contactHeaderTotal += gTemp[64u + p];
            jointHeaderTotal += gTemp[96u + p];
        }
        gTotals[0] = contactHeaderTotal + jointHeaderTotal;  // outNumStaticBatches
        gTotals[1] = contactHeaderTotal;                     // numStaticContactBatches
        gTotals[2] = jointHeaderTotal;                       // numStatic1dConstraintBatches
        gTotals[3] = contactHeaderTotal + jointHeaderTotal;  // numStaticBatches
        gTotals[4] = contactHeaderTotal + jointHeaderTotal;  // mStaticRigidBatchCount
    }

    const uint cHist = blockHistogram(0u, b);
    const uint jHist = blockHistogram(1u, b);
    const uint cbHist = blockHistogram(2u, b);
    const uint jbHist = blockHistogram(3u, b);

    for (uint i = 0u; i < numIter; ++i)
    {
        const uint workIndex = i * BLOCK_DIM + tid + numIter * b * BLOCK_DIM;
        const uint contactCount = countAt(0u, workIndex, nbBodies);
        const uint jointCount = countAt(1u, workIndex, nbBodies);

        const uint maxContact = warpAgg(0u, 1u, b, i, w, nbBodies, numIter);
        const uint maxJoint = warpAgg(1u, 1u, b, i, w, nbBodies, numIter);

        // exclusive warp offsets (the lanes<16 warpScan, broadcast by
        // __shfl(_, warpIndex))
        const uint contactWarpOffset = warpPrefix(0u, 0u, b, i, w, nbBodies, numIter);
        const uint jointWarpOffset = warpPrefix(1u, 0u, b, i, w, nbBodies, numIter);
        // the header offsets scan the warp MAXES
        const uint contactBlockWarpOffset = warpPrefix(0u, 1u, b, i, w, nbBodies, numIter);
        const uint jointBlockWarpOffset = warpPrefix(1u, 1u, b, i, w, nbBodies, numIter);

        uint contactOffset = contactUniqueIndexOffset + contactWarpOffset + cHist +
                             accumBefore(0u, 0u, b, i, nbBodies, numIter);
        uint jointOffset = jointUniqueIndexOffset + jointWarpOffset + jHist +
                           accumBefore(1u, 0u, b, i, nbBodies, numIter);
        const uint contactBlockOffset = contactBlockWarpOffset + cbHist +
                                        accumBefore(0u, 1u, b, i, nbBodies, numIter);
        const uint jointBlockOffset = jointBlockWarpOffset + jbHist +
                                      accumBefore(1u, 1u, b, i, nbBodies, numIter);

        uint blockOffset = contactBlockOffset + jointBlockOffset + batchOffset;

        if (workIndex < nbBodies)
        {
            gStartIndices[workIndex] = blockOffset;
            gStartIndices[nbBodies + workIndex] = blockOffset + maxContact;
        }

        // contact rounds
        for (uint r = 0u; r < maxContact; ++r)
        {
            const uint mask = roundMask(0u, b, i, w, r, nbBodies, numIter);
            const uint stride = countbits(mask);
            const uint offset = countbits(mask & ((1u << lane) - 1u));

            if (contactCount > r)
                gOutContactIds[contactOffset + offset] = gContactIds[workIndex + nbBodies * r];

            if (lane == 0u)
            {
                const uint batchIndex = contactBlockOffset + r + staticContactBatchOffset;
                // header: mDescStride u16 | constraintType u16 (eCONTACT = 1)
                gHeaders[blockOffset] = uint4(stride | (1u << 16u), batchIndex, contactOffset, mask);
                gContactBatchIndices[contactBlockOffset + numContactBatches + r] = blockOffset;
            }

            contactOffset += stride;
            blockOffset += 1u;
        }

        // joint rounds
        for (uint r = 0u; r < maxJoint; ++r)
        {
            const uint mask = roundMask(1u, b, i, w, r, nbBodies, numIter);
            const uint stride = countbits(mask);
            const uint offset = countbits(mask & ((1u << lane) - 1u));

            if (jointCount > r)
                gOutJointIds[jointOffset + offset] = gJointIds[workIndex + nbBodies * r];

            if (lane == 0u)
            {
                const uint batchIndex = jointBlockOffset + r + staticJointBatchOffset;
                // eCONSTRAINT_1D = 0
                gHeaders[blockOffset] = uint4(stride | (0u << 16u), batchIndex, jointOffset, mask);
                gJointBatchIndices[jointBlockOffset + r + numConstraintBatches] = blockOffset;
            }

            jointOffset += stride;
            blockOffset += 1u;
        }
    }
}
