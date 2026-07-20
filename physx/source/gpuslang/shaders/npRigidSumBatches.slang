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
// P5 Step 4 (BDA, ENGINE launch ABI): Slang port of
// rigidSumInternalContactAndJointBatches1 (constraintBlockPrePrep.cu
// L1472-1604) and rigidSumInternalContactAndJointBatches2 (L1607-1956), the
// static contact/joint batch accumulation pair, driven exactly as
// PxgSolverCore::constraintPrePrepParallel launches them
// (PxgSolverCore.cpp L521/L544): 32 blocks x 512 threads, params
// { PxgPrePrepDesc*, PxU32 nbBodies } for kernel 1 and { PxgPrePrepDesc*,
// PxgSolverCoreDesc*, PxgConstraintPrepareDesc*, PxU32 nbBodies } for
// kernel 2, all data reached through the byte-exact BDA desc mirrors
// (pxgSlangPrepDesc.slang / pxgSlangSolverDesc.slang, proven by
// testPrepDescLayout / testSolverDescLayout).
//
// Every quantity in both kernels is unsigned-integer add/max, so all the
// warp/block collectives (warpReduction Add/Max, the inclusive warpScan with
// exclusive stores, __ballot/__popc lane offsets, the shared-memory
// accumulators guarded by __syncthreads pairs) are associative reductions
// over fixed index domains; the port computes each one as a deterministic
// pure-function loop over the same domain -- bitwise identical, with no
// scheduling assumptions.  CUDA thread (block b, tid t) maps to Slang thread
// (group b, local t) under the ENGINE grid/block geometry; the per-iteration
// accumulators (sContactUniqueIndicesAccum etc., which CUDA reads BEFORE the
// thread-15 update between the two __syncthreads) become prefix sums over
// the previous iterations' whole-block totals.
//
// Kernel 1: per-block totals of the static contact/joint counts over the
// block's strided workIndex domain, written to the four 32-entry temp arrays
// (desc->mTempContactUniqueIndices/...).  Only CUDA thread
// threadIdx.x == warpPerBlock-1 == 15 stores (cu L1594); all other threads
// only feed the reductions, which the pure-function recomputation replaces.
// The unique-index arrays are plain sums; the header arrays are SUMS OF
// PER-WARP MAXES (each warp's Max-reduction at L1544 feeds warp 0's
// Add-reduction at L1574) accumulated over the iterations -- NOT the block
// max.
//
// Kernel 2: warp 0's exclusive block histograms (inclusive warpScan minus
// input); the blockIdx-0/thread-31 totals writes (prePrepDesc->
// numStaticBatches, prepDesc->numStatic*Batches,
// solverCoreDesc->islandContextPool[0].mStaticRigidBatchCount); per
// iteration the 16 warp sums, each warp's exclusive offsets, the running
// accumulators, the per-body start indices (blockOffset,
// blockOffset + warp maxContact), and the interleaved unique-id scatter with
// one PxgConstraintBatchHeader (mDescStride | constraintType<<16,
// mConstraintBatchIndex, mStartPartitionIndex, ballot mask) per
// (warp, round), contact rounds first, then joints; eCONSTRAINT_1D = 0,
// eCONTACT = 1 (PxgSolverConstraintDesc.h L111-117).
//
// PxgKernelBlockDim::COMPUTE_STATIC_CONTACT_CONSTRAINT_COUNT = 512
// (16 warps), 32 blocks (nbBlocksRequired in PxgSolverCore.cpp).

import pxgSlangPrepDesc;
import pxgSlangSolverDesc;

static const uint NB_BLOCKS = 32u;   // nbBlocksRequired (PxgSolverCore.cpp) == the .cu's block_size
static const uint BLOCK_DIM = 512u;  // COMPUTE_STATIC_CONTACT_CONSTRAINT_COUNT
static const uint WARPS = 16u;       // warpPerBlock

// CUDA parameter blobs (natural alignment; the bridge pads the tail).
struct Rs1Params
{
    uint64_t prePrepDesc; // @0  PxgPrePrepDesc*
    uint nbBodies;        // @8
    uint pad;             // @12 (blob rounds to 16)
}

struct Rs2Params
{
    uint64_t prePrepDesc;        // @0  PxgPrePrepDesc*
    uint64_t solverCoreDesc;     // @8  PxgSolverCoreDesc*
    uint64_t constraintPrepDesc; // @16 PxgConstraintPrepareDesc*
    uint nbBodies;               // @24
    uint pad;                    // @28 (blob rounds to 32)
}

[[vk::push_constant]]
ConstantBuffer<Rs1Params> gRs1;

uint iterationsPerBlock(uint nbBodies)
{
    const uint totalBlockRequired = (nbBodies + (BLOCK_DIM - 1u)) / BLOCK_DIM;
    return (totalBlockRequired + (NB_BLOCKS - 1u)) / NB_BLOCKS;
}

uint countAt(Ptr<uint> counts, uint workIndex, uint nbBodies)
{
    if (workIndex >= nbBodies)
        return 0u;
    return counts[workIndex];
}

// ---- kernel 1: per-block sums / sums-of-warp-maxes ----
// One CUDA thread = one Slang thread under the engine 32x512 geometry; only
// tid == warpPerBlock-1 stores, and it recomputes the whole block's
// reductions as pure loops (deterministic uint adds/maxes).

[shader("compute")]
[numthreads(512, 1, 1)]
void rigidSumInternalContactAndJointBatches1(uint3 groupId : SV_GroupID,
                                             uint3 groupThreadId : SV_GroupThreadID)
{
    const uint tid = groupThreadId.x;
    if (tid != (WARPS - 1u))   // cu L1594: if (threadIdx.x == warpPerBlock - 1)
        return;
    const uint b = groupId.x;

    Ptr<PxgPrePrepDescGpu> desc = Ptr<PxgPrePrepDescGpu>(gRs1.prePrepDesc);
    Ptr<uint> contactCounts = Ptr<uint>(desc[0].mRigidStaticContactCounts);
    Ptr<uint> jointCounts = Ptr<uint>(desc[0].mRigidStaticConstraintCounts);

    const uint nbBodies = gRs1.nbBodies;
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
                const uint workIndex = i * BLOCK_DIM + (w * 32u + l) + numIter * b * BLOCK_DIM;
                const uint cv = countAt(contactCounts, workIndex, nbBodies);
                const uint jv = countAt(jointCounts, workIndex, nbBodies);
                sumC += cv;
                sumJ += jv;
                wMaxC = cv > wMaxC ? cv : wMaxC;
                wMaxJ = jv > wMaxJ ? jv : wMaxJ;
            }
            sumMaxC += wMaxC;
            sumMaxJ += wMaxJ;
        }
    }
    (Ptr<uint>(desc[0].mTempContactUniqueIndices))[b] = sumC;
    (Ptr<uint>(desc[0].mTempConstraintUniqueIndices))[b] = sumJ;
    (Ptr<uint>(desc[0].mTempContactBlockHeader))[b] = sumMaxC;
    (Ptr<uint>(desc[0].mTempConstraintBlockHeader))[b] = sumMaxJ;
}

// ---- kernel 2 helpers (pure functions of the inputs) ----

// warp sum/max of one warp's 32 counts at iteration i of block b
uint warpAgg(Ptr<uint> counts, uint isMax, uint b, uint i, uint w, uint nbBodies, uint numIter)
{
    uint acc = 0u;
    for (uint l = 0u; l < 32u; ++l)
    {
        const uint workIndex = i * BLOCK_DIM + (w * 32u + l) + numIter * b * BLOCK_DIM;
        const uint v = countAt(counts, workIndex, nbBodies);
        if (isMax != 0u)
            acc = v > acc ? v : acc;
        else
            acc += v;
    }
    return acc;
}

// exclusive prefix over the first `upTo` warps of block b at iteration i
uint warpPrefix(Ptr<uint> counts, uint isMax, uint b, uint i, uint upTo, uint nbBodies, uint numIter)
{
    uint acc = 0u;
    for (uint w = 0u; w < upTo; ++w)
        acc += warpAgg(counts, isMax, b, i, w, nbBodies, numIter);
    return acc;
}

// the shared accumulator value READ at iteration i (sum of the previous
// iterations' whole-block totals; CUDA guards the read/update with the
// __syncthreads pair at L1856-1868)
uint accumBefore(Ptr<uint> counts, uint isMax, uint b, uint i, uint nbBodies, uint numIter)
{
    uint acc = 0u;
    for (uint a = 0u; a < i; ++a)
        acc += warpPrefix(counts, isMax, b, a, WARPS, nbBodies, numIter);
    return acc;
}

// exclusive block histogram over one 32-entry temp array (warp 0's warpScan store)
uint blockHistogram(Ptr<uint> temp, uint b)
{
    uint acc = 0u;
    for (uint p = 0u; p < b; ++p)
        acc += temp[p];
    return acc;
}

// ballot over one warp: bit l set iff count(lane l) > round
uint roundMask(Ptr<uint> counts, uint b, uint i, uint w, uint round, uint nbBodies, uint numIter)
{
    uint mask = 0u;
    for (uint l = 0u; l < 32u; ++l)
    {
        const uint workIndex = i * BLOCK_DIM + (w * 32u + l) + numIter * b * BLOCK_DIM;
        if (countAt(counts, workIndex, nbBodies) > round)
            mask |= (1u << l);
    }
    return mask;
}

// ---- kernel 2: one CUDA thread = one Slang thread (32 x 512) ----

[[vk::push_constant]]
ConstantBuffer<Rs2Params> gRs2;

[shader("compute")]
[numthreads(512, 1, 1)]
void rigidSumInternalContactAndJointBatches2(uint3 groupId : SV_GroupID,
                                             uint3 groupThreadId : SV_GroupThreadID)
{
    const uint b = groupId.x;        // blockIdx.x
    const uint tid = groupThreadId.x; // threadIdx.x
    const uint w = tid / 32u;        // warpIndex
    const uint lane = tid & 31u;

    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gRs2.prePrepDesc);
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gRs2.solverCoreDesc);
    Ptr<PxgConstraintPrepareDescGpu> constraintPrepDesc =
        Ptr<PxgConstraintPrepareDescGpu>(gRs2.constraintPrepDesc);

    Ptr<uint> staticContactCount = Ptr<uint>(prePrepDesc[0].mRigidStaticContactCounts);
    Ptr<uint> staticJointCount = Ptr<uint>(prePrepDesc[0].mRigidStaticConstraintCounts);
    Ptr<uint> contactStaticUniqueIds = Ptr<uint>(prePrepDesc[0].mRigidStaticContactIndices);
    Ptr<uint> jointStaticUniqueIndices = Ptr<uint>(prePrepDesc[0].mRigidStaticConstraintIndices);
    Ptr<uint> contactStaticStartIndices = Ptr<uint>(prePrepDesc[0].mRigidStaticContactStartIndices);
    Ptr<uint> jointStaticStartIndices = Ptr<uint>(prePrepDesc[0].mRigidStaticConstraintStartIndices);
    Ptr<PxgConstraintBatchHeaderGpu> batchHeaders =
        Ptr<PxgConstraintBatchHeaderGpu>(prePrepDesc[0].mBatchHeaders);
    Ptr<uint> outContactUniqueIds = Ptr<uint>(prePrepDesc[0].mContactUniqueIndices);
    Ptr<uint> outJointUniqueIndices = Ptr<uint>(prePrepDesc[0].mConstraintUniqueIndices);
    Ptr<uint> tempContactUnique = Ptr<uint>(prePrepDesc[0].mTempContactUniqueIndices);
    Ptr<uint> tempJointUnique = Ptr<uint>(prePrepDesc[0].mTempConstraintUniqueIndices);
    Ptr<uint> tempContactHeader = Ptr<uint>(prePrepDesc[0].mTempContactBlockHeader);
    Ptr<uint> tempJointHeader = Ptr<uint>(prePrepDesc[0].mTempConstraintBlockHeader);
    Ptr<uint> jointConstraintBatchIndices = Ptr<uint>(constraintPrepDesc[0].jointConstraintBatchIndices);
    Ptr<uint> contactConstraintBatchIndices = Ptr<uint>(constraintPrepDesc[0].contactConstraintBatchIndices);

    const uint nbBodies = gRs2.nbBodies;
    const uint numIter = iterationsPerBlock(nbBodies);

    const uint numRigidContacts = prePrepDesc[0].numTotalContacts;
    const uint numRigidJoints = prePrepDesc[0].numTotalConstraints;
    const uint numContactBatches = constraintPrepDesc[0].numContactBatches;
    const uint numConstraintBatches = constraintPrepDesc[0].num1dConstraintBatches;
    const uint numArtiBatches = prePrepDesc[0].numArtiBatches + prePrepDesc[0].numArtiSelfBatches +
                                prePrepDesc[0].numArtiStaticBatches;
    const uint batchOffset = prePrepDesc[0].numBatches + numArtiBatches;
    const uint contactUniqueIndexOffset = numRigidContacts;
    const uint jointUniqueIndexOffset = numRigidJoints;
    const uint staticContactBatchOffset = prePrepDesc[0].artiStaticContactBatchOffset +
                                          constraintPrepDesc[0].numArtiStaticContactBatches +
                                          constraintPrepDesc[0].numArtiSelfContactBatches;
    const uint staticJointBatchOffset = prePrepDesc[0].artiStaticConstraintBatchOffset +
                                        constraintPrepDesc[0].numArtiStatic1dConstraintBatches +
                                        constraintPrepDesc[0].numArtiSelf1dConstraintBatches;

    // blockIdx 0, threadIdx 31: the run-total writes (the inclusive lane-31
    // warpScan values = full 32-block sums of the header temp arrays)
    if (b == 0u && tid == 31u)
    {
        uint contactHeaderTotal = 0u, jointHeaderTotal = 0u;
        for (uint p = 0u; p < 32u; ++p)
        {
            contactHeaderTotal += tempContactHeader[p];
            jointHeaderTotal += tempJointHeader[p];
        }
        const uint totalNumStaticBatches = contactHeaderTotal + jointHeaderTotal;
        prePrepDesc[0].numStaticBatches = totalNumStaticBatches;             // outNumStaticBatches
        constraintPrepDesc[0].numStaticContactBatches = contactHeaderTotal;
        constraintPrepDesc[0].numStatic1dConstraintBatches = jointHeaderTotal;
        constraintPrepDesc[0].numStaticBatches = totalNumStaticBatches;
        // PxgIslandContext& island = solverCoreDesc->islandContextPool[0]
        Ptr<PxgIslandContextGpu> island = Ptr<PxgIslandContextGpu>(solverCoreDesc[0].islandContextPool);
        island[0].mStaticRigidBatchCount = totalNumStaticBatches;
    }

    const uint cHist = blockHistogram(tempContactUnique, b);
    const uint jHist = blockHistogram(tempJointUnique, b);
    const uint cbHist = blockHistogram(tempContactHeader, b);
    const uint jbHist = blockHistogram(tempJointHeader, b);

    for (uint i = 0u; i < numIter; ++i)
    {
        const uint workIndex = i * BLOCK_DIM + tid + numIter * b * BLOCK_DIM;
        const uint contactCount = countAt(staticContactCount, workIndex, nbBodies);
        const uint jointCount = countAt(staticJointCount, workIndex, nbBodies);

        const uint maxContact = warpAgg(staticContactCount, 1u, b, i, w, nbBodies, numIter);
        const uint maxJoint = warpAgg(staticJointCount, 1u, b, i, w, nbBodies, numIter);

        // exclusive warp offsets (the lanes<16 warpScan, broadcast by
        // __shfl(_, warpIndex))
        const uint contactWarpOffset = warpPrefix(staticContactCount, 0u, b, i, w, nbBodies, numIter);
        const uint jointWarpOffset = warpPrefix(staticJointCount, 0u, b, i, w, nbBodies, numIter);
        // the header offsets scan the warp MAXES
        const uint contactBlockWarpOffset = warpPrefix(staticContactCount, 1u, b, i, w, nbBodies, numIter);
        const uint jointBlockWarpOffset = warpPrefix(staticJointCount, 1u, b, i, w, nbBodies, numIter);

        uint contactOffset = contactUniqueIndexOffset + contactWarpOffset + cHist +
                             accumBefore(staticContactCount, 0u, b, i, nbBodies, numIter);
        uint jointOffset = jointUniqueIndexOffset + jointWarpOffset + jHist +
                           accumBefore(staticJointCount, 0u, b, i, nbBodies, numIter);
        const uint contactBlockOffset = contactBlockWarpOffset + cbHist +
                                        accumBefore(staticContactCount, 1u, b, i, nbBodies, numIter);
        const uint jointBlockOffset = jointBlockWarpOffset + jbHist +
                                      accumBefore(staticJointCount, 1u, b, i, nbBodies, numIter);

        uint blockOffset = contactBlockOffset + jointBlockOffset + batchOffset;

        if (workIndex < nbBodies)
        {
            contactStaticStartIndices[workIndex] = blockOffset;
            jointStaticStartIndices[workIndex] = blockOffset + maxContact;
        }

        // contact rounds
        for (uint r = 0u; r < maxContact; ++r)
        {
            const uint mask = roundMask(staticContactCount, b, i, w, r, nbBodies, numIter);
            const uint stride = countbits(mask);
            const uint offset = countbits(mask & ((1u << lane) - 1u));

            if (contactCount > r)
                outContactUniqueIds[contactOffset + offset] = contactStaticUniqueIds[workIndex + nbBodies * r];

            if (lane == 0u)
            {
                const uint batchIndex = contactBlockOffset + r + staticContactBatchOffset;
                // header: mDescStride u16 | constraintType u16 (eCONTACT = 1)
                batchHeaders[blockOffset].mDescStride_constraintType = stride | (1u << 16u);
                batchHeaders[blockOffset].mConstraintBatchIndex = batchIndex;
                batchHeaders[blockOffset].mStartPartitionIndex = contactOffset;
                batchHeaders[blockOffset].mask = mask;
                contactConstraintBatchIndices[contactBlockOffset + numContactBatches + r] = blockOffset;
            }

            contactOffset += stride;
            blockOffset += 1u;
        }

        // joint rounds
        for (uint r = 0u; r < maxJoint; ++r)
        {
            const uint mask = roundMask(staticJointCount, b, i, w, r, nbBodies, numIter);
            const uint stride = countbits(mask);
            const uint offset = countbits(mask & ((1u << lane) - 1u));

            if (jointCount > r)
                outJointUniqueIndices[jointOffset + offset] = jointStaticUniqueIndices[workIndex + nbBodies * r];

            if (lane == 0u)
            {
                const uint batchIndex = jointBlockOffset + r + staticJointBatchOffset;
                // eCONSTRAINT_1D = 0
                batchHeaders[blockOffset].mDescStride_constraintType = stride | (0u << 16u);
                batchHeaders[blockOffset].mConstraintBatchIndex = batchIndex;
                batchHeaders[blockOffset].mStartPartitionIndex = jointOffset;
                batchHeaders[blockOffset].mask = mask;
                jointConstraintBatchIndices[jointBlockOffset + r + numConstraintBatches] = blockOffset;
            }

            jointOffset += stride;
            blockOffset += 1u;
        }
    }
}
