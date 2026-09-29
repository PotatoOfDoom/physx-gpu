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

// thresholdStreamBda.slang -- P9.91: the FORCE-THRESHOLD REPORTING pipeline on the
// ENGINE launch ABI (gpusolver/src/CUDA/accumulateThresholdStream.cu, all fourteen
// kernels), in the order PxgCudaSolverCore / PxgTGSCudaSolverCore::
// accumulatedForceThresholdStream() launches them:
//
//   initialRanksAndBodyIndexB                     keys = nodeIndexB.index(), ranks = i
//   bodyInputAndRanksSingleBlockLaunch  } x nbPasses   the 4-bit radix passes
//   bodyInputAndRanksBlocksLaunch       }              (RADIXSORT, 256 threads)
//   initialRanksAndBodyIndexA                     keys = nodeIndexA.index() via the B ranks
//   bodyInputAndRanks* x nbPasses
//   reorganizeThresholdElements                   gather into (A, B) order
//   computeAccumulateThresholdStream / outputAccumulateThresholdStream /
//   writeoutAccumulatedForcePerObject             per-pair force sums
//   computeExceededForceThresholdElementIndice /
//   outputExceededForceThresholdElementIndice     the exceeded set
//   setThresholdElementsMask (only if nbPrev > 0) / computeThresholdElementMaskIndices /
//   outputThresholdPairsMaskIndices / createForceChangeThresholdElements
//                                                 found / lost / persistent events
//
// Every kernel reads the engine's own PxgSolverCoreDesc (and, for the exceeded-force
// pass, PxgSolverSharedDesc::dt) through buffer device addresses, exactly as the CUDA
// does.  The bodies are the ones thresholdStream.slang and thresholdForceChange.slang
// proved against a CPU mirror of the CUDA (testThresholdStream,
// testThresholdForceChange, including the P9.86f / P9.89 association spellings);
// those two files stay as they are -- their fixtures bind buffers directly -- and this
// file is what the manifest maps.  What differs from them, and why:
//
//  * `bool* thresholdStreamWriteable` is a BYTE array on the engine ABI (the binding
//    ports used a uint per element).  Four lanes share a word and four different
//    invocations write them, so every store is the InterlockedAnd/InterlockedOr pair
//    (check_subword_store.py); loads mask their byte out of the containing word.
//  * The exceeded-force and force-change element copies write the six named fields
//    and leave `pad` alone.  The CUDA copies a local `tempElement` whose pad is never
//    assigned, so the CUDA's pad byte is indeterminate; the binding ports wrote 0.
//    Not writing it is the one choice that invents no bytes (P9.87).
//  * setPersistentForceElementMask is NOT a launch of its own here: the engine has
//    no such launch.  It runs, as in the CUDA, at the head of
//    computeThresholdElementMaskIndices -- see the note there on the cross-block
//    read-after-write that comes with it.
//  * The two radix kernels are radixSortDesc.slang's proven histogram/rank bodies at
//    the solver's block size (PxgKernelBlockDim::RADIXSORT = 256, not the common
//    sort's 1024: radixSortSingleBlock<RADIXSORT / WARP_SIZE> partitions the keys by
//    blockDim.x), with the key count read from solverDesc->sharedThresholdStreamIndex.
//    The histogram counts the padding lanes the CUDA's sanitizeKeys() turns into
//    0xffffffff (digit 0xF), so radixBlockCounts holds the CUDA's numbers; the rank
//    pass never writes a padding lane, exactly as the CUDA's keysToProcess bound.
//
// Barriers: every hand-off between invocations of one workgroup that goes through
// DEVICE memory uses AllMemoryBarrierWithGroupSync (a GroupMemoryBarrier orders
// groupshared memory only).

import pxgSlangScan;
import pxgSlangSolverDesc;

static const uint BLOCK = 256;           // PxgKernelBlockDim, every entry here
static const uint NUM_BLOCKS = 32;       // PxgKernelGridDim: RADIXSORT, the block-partitioned entries
static const uint NUM_BLOCKS_WIDE = 64;  // PxgKernelGridDim: the plain grid-stride entries
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint RADIX_SIZE = 16;

// mirrors Dy::ThresholdStreamElement (DyThresholdTable.h:51), 40 bytes:
//   @0 shapeInteraction | @8 normalForce | @12 threshold | @16 nodeIndexA |
//   @24 nodeIndexB | @32 accumulatedForce | @36 pad
struct ThresholdStreamElement
{
    uint2 shapeInteraction;
    float normalForce;
    float threshold;
    uint2 nodeIndexA;   // PxNodeIndex: x = mID (index()), y = mLinkID; order = the u64 (y, x)
    uint2 nodeIndexB;
    float accumulatedForce;
    uint pad;
}

// mirrors PxgRadixSortDesc (PxgRadixSortDesc.h): 5 device pointers + PxU32 count, 48 bytes
struct SolverRadixSortDesc
{
    uint64_t inputKeys;
    uint64_t inputRanks;
    uint64_t outputKeys;
    uint64_t outputRanks;
    uint64_t radixBlockCounts;
    uint count;
    uint pad0;
}

// ---- push-constant ABIs (bridge/PxgVulkanKernelAbi.cpp) ----
// (const PxgSolverCoreDesc*, PxgRadixSortDesc*, PxU32 startBit) -> {8,8,4}
struct TsRadixParams
{
    uint64_t solverDesc;
    uint64_t rsDesc;
    uint startBit;
}
// (const PxgSolverCoreDesc*, const PxgRadixSortDesc*) -> {8,8}
struct TsDescRsParams
{
    uint64_t solverDesc;
    uint64_t rsDesc;
}
// (PxgSolverCoreDesc*) -> {8}
struct TsDescParams
{
    uint64_t solverDesc;
}
// (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveData>*) -> {8,8}
struct TsDescSharedParams
{
    uint64_t solverDesc;
    uint64_t sharedDesc;
}

[[vk::push_constant]] ConstantBuffer<TsRadixParams> gTsRadix;
[[vk::push_constant]] ConstantBuffer<TsDescRsParams> gTsRs;
[[vk::push_constant]] ConstantBuffer<TsDescParams> gTs;
[[vk::push_constant]] ConstantBuffer<TsDescSharedParams> gTsShared;

bool nodeEq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }
bool nodeLt(uint2 a, uint2 b) { return a.y < b.y || (a.y == b.y && a.x < b.x); }
bool nodeLe(uint2 a, uint2 b) { return a.y < b.y || (a.y == b.y && a.x <= b.x); }

// ---- the bool* byte array ----
// A C++ bool is one byte holding 0 or 1; the CUDA reads it as `bool`, so any
// non-zero byte is true.
bool loadBool(uint64_t base, uint i)
{
    const uint64_t addr = base + uint64_t(i);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    return ((word >> (uint(addr & 3UL) * 8u)) & 0xffu) != 0u;
}

// One byte through the containing word, as the atomic pair: the AND clears only this
// lane's bits and the OR sets only them, so the three other lanes' writers can
// interleave freely.  `& ~uint64_t(3)`, never `& ~3UL` (Slang evaluates the latter's
// complement in 32 bits and clears the top half of the address).
void storeBool(uint64_t base, uint i, bool value)
{
    const uint64_t addr = base + uint64_t(i);
    Ptr<uint> w = Ptr<uint>(addr & ~uint64_t(3));
    const uint shift = uint(addr & 3UL) * 8u;
    uint prev;
    InterlockedAnd(w[0], ~(0xffu << shift), prev);
    if (value)
        InterlockedOr(w[0], 1u << shift, prev);
}

// ===========================================================================
// the body-index radix sort (RadixSort.cuh radixSortSingleBlock /
// radixSortCalculateRanks at WARP_PERBLOCK_SIZE = 256 / 32)
// ===========================================================================

groupshared uint gsHistogram[RADIX_SIZE];
groupshared uint gsDigitStart[RADIX_SIZE];
groupshared uint gsWarpCounts[RADIX_SIZE * WARPS_PER_BLOCK];
groupshared uint gsIterTotals[RADIX_SIZE];

uint getRadix(uint key, uint startBit)
{
    return (key >> startBit) & 0xFu;
}

struct BlockChunk
{
    uint startU4;
    uint endU4;
    uint iterations;
}

// RadixSort.cuh:181-205: the keys are processed as uint4 rows, and block b owns rows
// [numIterationPerBlock * b * blockDim, +numIterationPerBlock * blockDim) clamped to
// the row count.
BlockChunk computeBlockChunk(uint blockId, uint numKeys)
{
    const uint numKeysU4 = (numKeys + 3) / 4;
    const uint totalBlocksRequired = (numKeysU4 + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlocksRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    BlockChunk c;
    c.startU4 = min(numIterationPerBlock * blockId * BLOCK, numKeysU4);
    c.endU4 = min(c.startU4 + numIterationPerBlock * BLOCK, numKeysU4);
    c.iterations = numIterationPerBlock;
    return c;
}

// mirrors bodyInputAndRanksSingleBlockLaunch -> radixSortSingleBlock: this block's
// digit histogram, written to radixBlockCounts[digit * gridDim.x + blockIdx.x].
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void bodyInputAndRanksSingleBlockLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                        uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint startBit = gTsRadix.startBit;

    const SolverRadixSortDesc rs = (Ptr<SolverRadixSortDesc>(gTsRadix.rsDesc))[0];
    const uint numKeys = uint((Ptr<PxgSolverCoreDescGpu>(gTsRadix.solverDesc))[0].sharedThresholdStreamIndex);
    Ptr<uint> inputKeys = Ptr<uint>(rs.inputKeys);
    Ptr<uint> radixCounts = Ptr<uint>(rs.radixBlockCounts);

    if (tid < RADIX_SIZE)
        gsHistogram[tid] = 0u;

    GroupMemoryBarrierWithGroupSync();

    const BlockChunk chunk = computeBlockChunk(blockId, numKeys);

    for (uint u4 = chunk.startU4 + tid; u4 < chunk.endU4; u4 += BLOCK)
    {
        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            // sanitizeKeys (RadixSort.cuh:110): a lane at or past gNumKeys of the last
            // row is counted as the key 0xffffffff, i.e. digit 0xF at every startBit.
            const uint keyIndex = u4 * 4 + c;
            const uint digit = keyIndex < numKeys ? getRadix(inputKeys[keyIndex], startBit) : 0xFu;
            InterlockedAdd(gsHistogram[digit], 1u);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid < RADIX_SIZE)
        radixCounts[tid * NUM_BLOCKS + blockId] = gsHistogram[tid];
}

// mirrors bodyInputAndRanksBlocksLaunch -> radixSortCalculateRanks: stable scatter of
// this block's keys and ranks by the current digit.  A padding lane is counted in the
// histogram (above) but never written: the CUDA bounds its scatter by keysToProcess =
// gNumOfKeys - base (RadixSort.cuh:427), and padding sorts after every real key of the
// block, so it shifts no real key's position.
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void bodyInputAndRanksBlocksLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                   uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint warpIndex = tid >> 5;
    const uint startBit = gTsRadix.startBit;

    const SolverRadixSortDesc rs = (Ptr<SolverRadixSortDesc>(gTsRadix.rsDesc))[0];
    const uint numKeys = uint((Ptr<PxgSolverCoreDescGpu>(gTsRadix.solverDesc))[0].sharedThresholdStreamIndex);
    Ptr<uint> inputKeys = Ptr<uint>(rs.inputKeys);
    Ptr<uint> inputRanks = Ptr<uint>(rs.inputRanks);
    Ptr<uint> outputKeys = Ptr<uint>(rs.outputKeys);
    Ptr<uint> outputRanks = Ptr<uint>(rs.outputRanks);
    Ptr<uint> radixCounts = Ptr<uint>(rs.radixBlockCounts);

    // ---- global write cursor per digit for this block ----
    if (tid < RADIX_SIZE)
    {
        const uint d = tid;
        uint blockPrefix = 0;
        uint digitTotal = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint cnt = radixCounts[d * NUM_BLOCKS + b];
            if (b < blockId)
                blockPrefix += cnt;
            digitTotal += cnt;
        }
        gsIterTotals[d] = digitTotal;
        gsDigitStart[d] = blockPrefix;
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        uint running = 0;
        for (uint d = 0; d < RADIX_SIZE; ++d)
        {
            gsDigitStart[d] += running;
            running += gsIterTotals[d];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // ---- iterate the chunk, ranking BLOCK rows of 4 keys per iteration ----
    const BlockChunk chunk = computeBlockChunk(blockId, numKeys);

    for (uint iter = 0; iter < chunk.iterations; ++iter)
    {
        const uint u4 = chunk.startU4 + iter * BLOCK + tid;
        const bool rowValid = u4 < chunk.endU4;

        uint keys[4];
        uint ranks[4];
        uint digits[4];
        bool valid[4];
        uint withinWarpRank[4];

        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            const uint keyIndex = u4 * 4 + c;
            valid[c] = rowValid && keyIndex < numKeys;
            keys[c] = valid[c] ? inputKeys[keyIndex] : 0u;
            ranks[c] = valid[c] ? inputRanks[keyIndex] : 0u;
            digits[c] = getRadix(keys[c], startBit);
            withinWarpRank[c] = 0u;
        }

        for (uint group = 0; group < RADIX_SIZE; group += 4)
        {
            uint accum = 0;
            [ForceUnroll]
            for (uint c = 0; c < 4; ++c)
            {
                const uint rel = digits[c] - group;
                if (valid[c] && rel < 4)
                    accum += 1u << (rel << 3);
            }

            const uint inclusive = warp32InclusiveScan(tid, accum);

            if ((tid & 31u) == 31u)
            {
                [ForceUnroll]
                for (uint dd = 0; dd < 4; ++dd)
                    gsWarpCounts[(group + dd) * WARPS_PER_BLOCK + warpIndex] = (inclusive >> (dd << 3)) & 0xFFu;
            }

            uint exclusive = inclusive - accum;

            [ForceUnroll]
            for (uint c = 0; c < 4; ++c)
            {
                const uint rel = digits[c] - group;
                if (valid[c] && rel < 4)
                {
                    withinWarpRank[c] = (exclusive >> (rel << 3)) & 0xFFu;
                    exclusive += 1u << (rel << 3);
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        if (tid < RADIX_SIZE)
        {
            uint running = 0;
            for (uint w = 0; w < WARPS_PER_BLOCK; ++w)
            {
                const uint t = gsWarpCounts[tid * WARPS_PER_BLOCK + w];
                gsWarpCounts[tid * WARPS_PER_BLOCK + w] = running;
                running += t;
            }
            gsIterTotals[tid] = running;
        }

        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            if (valid[c])
            {
                const uint d = digits[c];
                const uint outIndex = gsDigitStart[d]
                    + gsWarpCounts[d * WARPS_PER_BLOCK + warpIndex]
                    + withinWarpRank[c];
                outputKeys[outIndex] = keys[c];
                outputRanks[outIndex] = ranks[c];
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (tid < RADIX_SIZE)
            gsDigitStart[tid] += gsIterTotals[tid];

        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// sort-key extraction + reorganize (grid NUM_BLOCKS_WIDE)
// ===========================================================================

// mirrors initialRanksAndBodyIndexB (:70): keys = nodeIndexB.index(), ranks = i, and
// the tail padded up to a multiple of 4 with 0xffffffff keys / identity ranks
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void initialRanksAndBodyIndexB(uint3 dtid : SV_DispatchThreadID)
{
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTsRs.solverDesc))[0];
    const SolverRadixSortDesc rs = (Ptr<SolverRadixSortDesc>(gTsRs.rsDesc))[0];
    Ptr<ThresholdStreamElement> thresholdStream = Ptr<ThresholdStreamElement>(desc.thresholdStream);
    Ptr<uint> inputKeys = Ptr<uint>(rs.inputKeys);
    Ptr<uint> inputRanks = Ptr<uint>(rs.inputRanks);
    const uint n = uint(desc.sharedThresholdStreamIndex);
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;

    for (uint i = dtid.x; i < n; i += stride)
    {
        inputKeys[i] = thresholdStream[i].nodeIndexB.x;
        inputRanks[i] = i;
    }

    const uint remaining = (4 - (n & 3)) & 3;
    for (uint i = dtid.x; i < remaining; i += stride)
    {
        const uint index = i + n;
        inputKeys[index] = 0xffffffffu;
        inputRanks[index] = index;
    }
}

// mirrors initialRanksAndBodyIndexA (:99): after the B sort the ranks hold the B
// permutation, so keys[i] = thresholdStream[ranks[i]].nodeIndexA.index()
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void initialRanksAndBodyIndexA(uint3 dtid : SV_DispatchThreadID)
{
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTsRs.solverDesc))[0];
    const SolverRadixSortDesc rs = (Ptr<SolverRadixSortDesc>(gTsRs.rsDesc))[0];
    Ptr<ThresholdStreamElement> thresholdStream = Ptr<ThresholdStreamElement>(desc.thresholdStream);
    Ptr<uint> inputKeys = Ptr<uint>(rs.inputKeys);
    Ptr<uint> inputRanks = Ptr<uint>(rs.inputRanks);
    const uint n = uint(desc.sharedThresholdStreamIndex);
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;

    for (uint i = dtid.x; i < n; i += stride)
        inputKeys[i] = thresholdStream[inputRanks[i]].nodeIndexA.x;

    const uint remaining = (4 - (n & 3)) & 3;
    for (uint i = dtid.x; i < remaining; i += stride)
        inputKeys[i + n] = 0xffffffffu;
}

// mirrors reorganizeThresholdElements (:127): eight lanes per element copy words 0..7
// (shapeInteraction .. nodeIndexB) of element ranks[i] of the unsorted copy into slot
// i; accumulatedForce and pad of the destination are left as they are
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void reorganizeThresholdElements(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTsRs.solverDesc))[0];
    const SolverRadixSortDesc rs = (Ptr<SolverRadixSortDesc>(gTsRs.rsDesc))[0];
    Ptr<uint> dstWords = Ptr<uint>(desc.thresholdStream);
    Ptr<uint> srcWords = Ptr<uint>(desc.tmpThresholdStream);
    Ptr<uint> inputRanks = Ptr<uint>(rs.inputRanks);
    const uint n = uint(desc.sharedThresholdStreamIndex);
    const uint stride = (BLOCK * NUM_BLOCKS_WIDE) / 8;
    const uint lane8 = gtid.x & 7;

    for (uint i = dtid.x / 8; i < n; i += stride)
        dstWords[i * 10 + lane8] = srcWords[inputRanks[i] * 10 + lane8];
}

// ===========================================================================
// device-wide segmented force accumulation (grid NUM_BLOCKS)
// ===========================================================================

groupshared float gsForce[BLOCK];       // per-lane force scan scratch
groupshared uint gsPairs[BLOCK];        // per-lane pair scan scratch
groupshared float gsWarpForce[WARPS_PER_BLOCK];
groupshared uint gsWarpPairs[WARPS_PER_BLOCK];
groupshared float gsBlockForce;
groupshared uint gsBlockPairs;
groupshared float gsBlockForceAccum[NUM_BLOCKS];
groupshared uint gsBlockWriteAccum[NUM_BLOCKS];

// warpScan<AddOpPxReal> (reduction.cuh:261-275): inclusive Hillis-Steele over each
// 32-lane warp, value = value + shfl_up(value, r) for r = 1, 2, 4, 8, 16
void warpScanForce(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        float add = (lane >= off) ? gsForce[tid - off] : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        gsForce[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

void warpScanPairs(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        uint add = (lane >= off) ? gsPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsPairs[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

// the LOG2_WARP_PERBLOCK_SIZE = 3 warpScan over the 8 warp totals, as an exclusive
// prefix, folding the block total into gsBlockForce/gsBlockPairs.  All barriers are
// in uniform control flow; only lanes < 8 touch the warp-total arrays.
void scanWarpTotals(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    float f0 = active ? gsWarpForce[tid] : 0.0f;
    uint p0 = active ? gsWarpPairs[tid] : 0u;
    float fInc = f0;
    uint pInc = p0;
    for (uint off = 1; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const float fa = take ? gsWarpForce[tid - off] : 0.0f;
        const uint pa = take ? gsWarpPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        fInc += fa;
        pInc += pa;
        if (active) { gsWarpForce[tid] = fInc; gsWarpPairs[tid] = pInc; }
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) { gsWarpForce[tid] = fInc - f0; gsWarpPairs[tid] = pInc - p0; }
    // P9.86f: accumulateThresholdStream.cu:256-258 adds (accumulatedForce + forceVal), i.e.
    // ((inclusive - f) + f), which need not round back to the inclusive sum.
    if (tid == WARPS_PER_BLOCK - 1) { gsBlockForce += ((fInc - f0) + f0); gsBlockPairs += pInc; }
}

// pairs-only variant; folds into gsBlockPairs
void scanWarpTotalsPairs(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    uint p0 = active ? gsWarpPairs[tid] : 0u;
    uint pInc = p0;
    for (uint off = 1; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const uint pa = take ? gsWarpPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        pInc += pa;
        if (active) gsWarpPairs[tid] = pInc;
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) gsWarpPairs[tid] = pInc - p0;
    if (tid == WARPS_PER_BLOCK - 1) gsBlockPairs += pInc;
}

// Block-uniform exclusive scan of per-lane 0/1 flags (the CUDA's
// __popc(__ballot_sync(FULL_MASK, flag) & lanesBelow)) plus the warp-total scan.
// Returns the lane's exclusive count within its warp; gsWarpPairs[warp] then holds the
// warp's exclusive prefix within the block and gsBlockPairs the running block total.
// `warpTotalAdd` is what the CUDA adds to the lane-31 count for the warp total -- the
// flag itself (computeAccumulate / computeExceeded) or the raw mask word
// (computeThresholdElementMaskIndices, :450 vs :674).
uint scanFlags(uint tid, uint flag, uint warpTotalAdd, out uint prevBlockPairs)
{
    const uint warpIndex = tid / WARP;
    gsPairs[tid] = flag;
    GroupMemoryBarrierWithGroupSync();
    warpScanPairs(tid);
    const uint accumPairs = gsPairs[tid] - flag;
    if ((tid & (WARP - 1)) == WARP - 1)
        gsWarpPairs[warpIndex] = accumPairs + warpTotalAdd;
    prevBlockPairs = gsBlockPairs;
    GroupMemoryBarrierWithGroupSync();
    scanWarpTotalsPairs(tid);
    GroupMemoryBarrierWithGroupSync();
    return accumPairs;
}

// mirrors computeAccumulateThresholdStream (:147)
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void computeAccumulateThresholdStream(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc))[0];
    Ptr<ThresholdStreamElement> thresholdStream = Ptr<ThresholdStreamElement>(desc.thresholdStream);
    Ptr<float> accumulatedForce = Ptr<float>(desc.thresholdStreamAccumulatedForce);
    Ptr<uint> writeIndex = Ptr<uint>(desc.thresholdStreamWriteIndex);
    const uint n = uint(desc.sharedThresholdStreamIndex);

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
    {
        gsBlockForce = 0.0f;
        gsBlockPairs = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        float val = 0.0f;
        bool isNewPair = false;
        if (workIndex < n)
        {
            const ThresholdStreamElement e = thresholdStream[workIndex];
            val = e.normalForce;
            if (workIndex + 1 < n)
            {
                const ThresholdStreamElement next = thresholdStream[workIndex + 1];
                if (!(nodeEq(e.nodeIndexA, next.nodeIndexA) && nodeEq(e.nodeIndexB, next.nodeIndexB)))
                    isNewPair = true;
            }
            else
            {
                isNewPair = true;
            }
        }

        // 32-lane inclusive scans -> exclusive prefixes within the warp
        gsForce[tid] = val;
        gsPairs[tid] = isNewPair ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        warpScanForce(tid);
        warpScanPairs(tid);
        const float accumVal = gsForce[tid] - val;
        const uint accumPairs = gsPairs[tid] - (isNewPair ? 1u : 0u);

        if ((tid & (WARP - 1)) == WARP - 1)
        {
            gsWarpForce[warpIndex] = accumVal + val;
            gsWarpPairs[warpIndex] = accumPairs + (isNewPair ? 1u : 0u);
        }

        const float prevBlockForce = gsBlockForce;
        const uint prevBlockPairs = gsBlockPairs;

        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < n)
        {
            accumulatedForce[workIndex] = val + accumVal + prevBlockForce + gsWarpForce[warpIndex];
            writeIndex[workIndex] = accumPairs + prevBlockPairs + gsWarpPairs[warpIndex];
            storeBool(desc.thresholdStreamWriteable, workIndex, isNewPair);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        Ptr<float>(desc.thresholdStreamAccumulatedForceBetweenBlocks)[blockIdx] = gsBlockForce;
        Ptr<uint>(desc.thresholdStreamWriteIndexBetweenBlocks)[blockIdx] = gsBlockPairs;
    }
}

// mirrors outputAccumulateThresholdStream (:284): add the exclusive prefix of the
// per-block totals to every element of each block's region
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void outputAccumulateThresholdStream(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc))[0];
    Ptr<float> accumulatedForce = Ptr<float>(desc.thresholdStreamAccumulatedForce);
    Ptr<uint> writeIndex = Ptr<uint>(desc.thresholdStreamWriteIndex);
    const uint n = uint(desc.sharedThresholdStreamIndex);

    if (tid < NUM_BLOCKS)
    {
        gsBlockForceAccum[tid] = Ptr<float>(desc.thresholdStreamAccumulatedForceBetweenBlocks)[tid];
        gsBlockWriteAccum[tid] = Ptr<uint>(desc.thresholdStreamWriteIndexBetweenBlocks)[tid];
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        // P9.86f: the force prefix is :315's warpScan<AddOpPxReal>(val) - val -- a
        // 32-lane Hillis-Steele tree (lane i adds lane i - r at radius r = 1, 2, 4, 8,
        // 16) and then minus its own value -- not a running sum: the two associate
        // differently and round differently.  Replayed lane by lane (descending i, so
        // each radius reads the previous radius' values).  The pair counts are
        // integers, exact in either order.
        float inc[NUM_BLOCKS];
        for (uint i = 0; i < NUM_BLOCKS; ++i)
            inc[i] = gsBlockForceAccum[i];
        for (uint r = 1u; r < NUM_BLOCKS; r <<= 1)
            for (int i = int(NUM_BLOCKS) - 1; i >= int(r); --i)
                inc[i] = inc[i] + inc[i - int(r)];
        uint pAcc = 0u;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const uint p = gsBlockWriteAccum[i];
            gsBlockForceAccum[i] = inc[i] - gsBlockForceAccum[i];
            gsBlockWriteAccum[i] = pAcc;
            pAcc += p;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const float blockForceAccum = gsBlockForceAccum[blockIdx];
    const uint blockWriteAccum = gsBlockWriteAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < n)
        {
            writeIndex[workIndex] = writeIndex[workIndex] + blockWriteAccum;
            accumulatedForce[workIndex] = accumulatedForce[workIndex] + blockForceAccum;
        }
    }
}

// mirrors writeoutAccumulatedForcePerObject (:348, grid NUM_BLOCKS_WIDE): each
// pair's inclusive force prefix, taken at its last element, into the per-pair array
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void writeoutAccumulatedForcePerObject(uint3 dtid : SV_DispatchThreadID)
{
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc))[0];
    Ptr<float> accumulatedForces = Ptr<float>(desc.thresholdStreamAccumulatedForce);
    Ptr<uint> writeIndex = Ptr<uint>(desc.thresholdStreamWriteIndex);
    Ptr<float> objectPairs = Ptr<float>(desc.accumulatedForceObjectPairs);
    const uint n = uint(desc.sharedThresholdStreamIndex);
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;

    for (uint workIndex = dtid.x; workIndex < n; workIndex += stride)
    {
        const uint w = writeIndex[workIndex];
        if (loadBool(desc.thresholdStreamWriteable, workIndex))
            objectPairs[w] = accumulatedForces[workIndex];
    }
}

// ===========================================================================
// exceeded-force detection + compaction (grid NUM_BLOCKS)
// ===========================================================================

// mirrors computeExceededForceThresholdElementIndice (:376): each pair's segment force
// (prefix difference) is stored on the element, and the elements whose force exceeds
// threshold * dt get their compaction indices
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void computeExceededForceThresholdElementIndice(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTsShared.solverDesc))[0];
    // PxgSolverSharedDesc<IterativeSolveData>::dt @76 (PGS and TGS share the layout)
    const float dt = (Ptr<PxgSolverSharedDescPGSGpu>(gTsShared.sharedDesc))[0].dt;
    Ptr<ThresholdStreamElement> thresholdStream = Ptr<ThresholdStreamElement>(desc.thresholdStream);
    Ptr<float> objectPairs = Ptr<float>(desc.accumulatedForceObjectPairs);
    Ptr<uint> writeIndex = Ptr<uint>(desc.thresholdStreamWriteIndex);
    const uint n = uint(desc.sharedThresholdStreamIndex);

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockPairs = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        bool isExceededForce = false;
        if (workIndex < n)
        {
            // the write-index buffer is reused below; each element reads its own slot
            // before it writes it (:428)
            const uint w = writeIndex[workIndex];
            float accumulatedForce = objectPairs[w];
            if (w > 0)
                accumulatedForce -= objectPairs[w - 1];
            thresholdStream[workIndex].accumulatedForce = accumulatedForce;
            isExceededForce = accumulatedForce > (thresholdStream[workIndex].threshold * dt);
        }

        const uint flag = isExceededForce ? 1u : 0u;
        uint prevBlockPairs;
        const uint accumPairs = scanFlags(tid, flag, flag, prevBlockPairs);

        if (workIndex < n)
        {
            writeIndex[workIndex] = accumPairs + prevBlockPairs + gsWarpPairs[warpIndex];
            storeBool(desc.thresholdStreamWriteable, workIndex, isExceededForce);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        Ptr<uint>(desc.thresholdStreamWriteIndexBetweenBlocks)[blockIdx] = gsBlockPairs;
}

// The exclusive prefix of the 32 per-block totals (a 32-lane warpScan<AddOpPxU32> minus
// the value in the CUDA; integers, so a running sum is exact) into gsBlockWriteAccum.
void blockTotalsExclusive(uint tid, Ptr<uint> betweenBlocks)
{
    if (tid < NUM_BLOCKS)
        gsBlockWriteAccum[tid] = betweenBlocks[tid];
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint acc = 0u;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const uint p = gsBlockWriteAccum[i];
            gsBlockWriteAccum[i] = acc;
            acc += p;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// the six named fields of a Dy::ThresholdStreamElement; `pad` is not written (see the
// file header)
void storeElementFields(Ptr<ThresholdStreamElement> dst, uint i, ThresholdStreamElement e, float accumulatedForce)
{
    dst[i].shapeInteraction = e.shapeInteraction;
    dst[i].nodeIndexA = e.nodeIndexA;
    dst[i].nodeIndexB = e.nodeIndexB;
    dst[i].normalForce = e.normalForce;
    dst[i].accumulatedForce = accumulatedForce;
    dst[i].threshold = e.threshold;
}

// mirrors outputExceededForceThresholdElementIndice (:488): compact the exceeded
// elements into exceededForceElements and publish their count
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void outputExceededForceThresholdElementIndice(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc);
    const PxgSolverCoreDescGpu desc = descPtr[0];
    Ptr<ThresholdStreamElement> thresholdStream = Ptr<ThresholdStreamElement>(desc.thresholdStream);
    Ptr<ThresholdStreamElement> exceeded = Ptr<ThresholdStreamElement>(desc.exceededForceElements);
    Ptr<uint> writeIndex = Ptr<uint>(desc.thresholdStreamWriteIndex);
    const uint n = uint(desc.sharedThresholdStreamIndex);

    blockTotalsExclusive(tid, Ptr<uint>(desc.thresholdStreamWriteIndexBetweenBlocks));

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockWriteAccum = gsBlockWriteAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < n)
        {
            const uint w = writeIndex[workIndex] + blockWriteAccum;
            const bool isExceededForce = loadBool(desc.thresholdStreamWriteable, workIndex);
            if (isExceededForce)
            {
                const ThresholdStreamElement e = thresholdStream[workIndex];
                storeElementFields(exceeded, w, e, e.accumulatedForce);
            }
            if (workIndex == n - 1)
                descPtr[0].nbExceededThresholdElements = isExceededForce ? (w + 1) : w;
        }
    }
}

// ===========================================================================
// the cross-frame force-change diff
// ===========================================================================

// Dy::ThresholdStreamElement::operator<= (nodeIndexA, then nodeIndexB, as u64 orders)
bool elementLe(ThresholdStreamElement a, ThresholdStreamElement b)
{
    return nodeLt(a.nodeIndexA, b.nodeIndexA) ||
           (nodeEq(a.nodeIndexA, b.nodeIndexA) && nodeLe(a.nodeIndexB, b.nodeIndexB));
}

// reduction.cuh:729 binarySearch: the largest index with data[index] <= value, or 0
// when data[0] > value
uint binarySearchPrev(Ptr<ThresholdStreamElement> data, uint numElements, ThresholdStreamElement value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (elementLe(data[pos], value))
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// mirrors setThresholdElementsMask (:570, grid NUM_BLOCKS_WIDE; the host launches it
// only when nbPrevExceededThresholdElements > 0): a current exceeded element whose
// (nodes, shapeInteraction) is in the previous exceeded set clears both its mask
// entries -- it is persistent, not a change
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void setThresholdElementsMask(uint3 dtid : SV_DispatchThreadID)
{
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc))[0];
    Ptr<ThresholdStreamElement> exceeded = Ptr<ThresholdStreamElement>(desc.exceededForceElements);
    Ptr<ThresholdStreamElement> prev = Ptr<ThresholdStreamElement>(desc.prevExceededForceElements);
    Ptr<uint> mask = Ptr<uint>(desc.thresholdStreamWriteIndex);
    const uint nbExceeded = desc.nbExceededThresholdElements;
    const uint nbPrev = desc.nbPrevExceededThresholdElements;
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;

    for (uint workIndex = dtid.x; workIndex < nbExceeded; workIndex += stride)
    {
        const ThresholdStreamElement element = exceeded[workIndex];
        uint pos = binarySearchPrev(prev, nbPrev, element);

        bool done = false;
        while (!done)
        {
            done = true;
            const ThresholdStreamElement prePair = prev[pos];
            if (nodeEq(prePair.nodeIndexA, element.nodeIndexA) && nodeEq(prePair.nodeIndexB, element.nodeIndexB))
            {
                if (prePair.shapeInteraction.x == element.shapeInteraction.x &&
                    prePair.shapeInteraction.y == element.shapeInteraction.y)
                {
                    mask[pos] = 0u;
                    mask[nbPrev + workIndex] = 0u;
                }
                else if (pos > 1u)   // sic (:600): `pos > 1`, so slot 0 is never revisited
                {
                    pos = pos - 1u;
                    done = false;
                }
            }
        }
    }
}

// mirrors computeThresholdElementMaskIndices (:633), with setPersistentForceElementMask
// (:612) at its head exactly where the CUDA calls it.
//
// THE CUDA'S CROSS-BLOCK READ-AFTER-WRITE, KEPT.  The persistent mask of previous
// element k is written by global thread k (block k / 256) at slot
// nbPrev + nbExceeded + k, and read by whichever block's chunk holds that slot, while
// the same scan overwrites slot k of the previous-mask region in place.  Nothing
// orders one block against another inside a launch, so once the three regions span
// more than one block the CUDA's result depends on block scheduling: a real upstream
// race (documented in RUNTIME_STATUS P9.91 as a candidate, not reported).  While all
// of nbPrev * 2 + nbExceeded fit in block 0's chunk (256 entries while the total stays
// at or below 8192, when every block owns exactly 256) the writer and every reader are
// one workgroup, and the AllMemoryBarrierWithGroupSync below orders them -- that is the
// domain the fixture pins.  Beyond it this port races exactly as the CUDA does, rather
// than computing an answer the CUDA does not.
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void computeThresholdElementMaskIndices(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc))[0];
    Ptr<uint> mask = Ptr<uint>(desc.thresholdStreamWriteIndex);
    const uint nbExceeded = desc.nbExceededThresholdElements;
    const uint nbPrev = desc.nbPrevExceededThresholdElements;
    const uint total = nbExceeded + nbPrev * 2u;

    // setPersistentForceElementMask: persistent[k] = !previous[k], grid-stride over the
    // launch's own 32 * 256 threads
    {
        const uint globalThreadIdx = tid + blockIdx * BLOCK;
        const uint persistentStart = nbPrev + nbExceeded;
        for (uint workIndex = globalThreadIdx; workIndex < nbPrev; workIndex += BLOCK * NUM_BLOCKS)
            mask[persistentStart + workIndex] = (mask[workIndex] != 0u) ? 0u : 1u;
    }

    const uint totalBlockRequired = (total + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockPairs = 0u;
    AllMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        uint forceChangeMask = 0u;
        if (workIndex < total)
            forceChangeMask = mask[workIndex];

        // the lane count is __popc(__ballot_sync(forceChangeMask) & below): a flag; the
        // warp total adds the raw mask word (:674)
        uint prevBlockPairs;
        const uint accumPairs = scanFlags(tid, forceChangeMask != 0u ? 1u : 0u, forceChangeMask, prevBlockPairs);

        if (workIndex < total)
        {
            mask[workIndex] = accumPairs + prevBlockPairs + gsWarpPairs[warpIndex];
            storeBool(desc.thresholdStreamWriteable, workIndex, forceChangeMask != 0u);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        Ptr<uint>(desc.thresholdStreamWriteIndexBetweenBlocks)[blockIdx] = gsBlockPairs;
}

// mirrors outputThresholdPairsMaskIndices (:734): add the cross-block exclusive prefix
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void outputThresholdPairsMaskIndices(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc))[0];
    Ptr<uint> mask = Ptr<uint>(desc.thresholdStreamWriteIndex);
    const uint total = desc.nbExceededThresholdElements + desc.nbPrevExceededThresholdElements * 2u;

    blockTotalsExclusive(tid, Ptr<uint>(desc.thresholdStreamWriteIndexBetweenBlocks));

    const uint totalBlockRequired = (total + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockWriteAccum = gsBlockWriteAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < total)
            mask[workIndex] = mask[workIndex] + blockWriteAccum;
    }
}

// mirrors createForceChangeThresholdElements (:787, grid NUM_BLOCKS_WIDE): compact the
// lost / found / persistent elements into forceChangeThresholdElements and publish
// their count.  A LOST pair carries accumulatedForce 0.
[shader("compute")]
[numthreads(BLOCK, 1, 1)]
void createForceChangeThresholdElements(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gTs.solverDesc);
    const PxgSolverCoreDescGpu desc = descPtr[0];
    Ptr<uint> mask = Ptr<uint>(desc.thresholdStreamWriteIndex);
    Ptr<ThresholdStreamElement> exceeded = Ptr<ThresholdStreamElement>(desc.exceededForceElements);
    Ptr<ThresholdStreamElement> prev = Ptr<ThresholdStreamElement>(desc.prevExceededForceElements);
    Ptr<ThresholdStreamElement> forceChange = Ptr<ThresholdStreamElement>(desc.forceChangeThresholdElements);
    const uint nbExceeded = desc.nbExceededThresholdElements;
    const uint nbPrev = desc.nbPrevExceededThresholdElements;
    const uint total = nbExceeded + nbPrev * 2u;
    const uint persistentStart = nbPrev + nbExceeded;
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;

    for (uint workIndex = dtid.x; workIndex < total; workIndex += stride)
    {
        const bool hasForceChangeOrPersistent = loadBool(desc.thresholdStreamWriteable, workIndex);
        const uint w = mask[workIndex];

        if (hasForceChangeOrPersistent)
        {
            const bool lostPair = workIndex < nbPrev;
            const bool foundPair = (workIndex < persistentStart) && !lostPair;

            ThresholdStreamElement pair;
            if (lostPair)
                pair = prev[workIndex];
            else if (foundPair)
                pair = exceeded[workIndex - nbPrev];
            else
                pair = prev[workIndex - persistentStart];

            storeElementFields(forceChange, w, pair, lostPair ? 0.0f : pair.accumulatedForce);
        }

        if (workIndex == total - 1u)
            descPtr[0].nbForceChangeElements = hasForceChangeOrPersistent ? (w + 1u) : w;
    }
}
