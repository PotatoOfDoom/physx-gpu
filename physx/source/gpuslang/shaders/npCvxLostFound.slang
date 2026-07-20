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

// npCvxLostFound.slang
//
// Slang port of prepareLostFoundPairs_Stage1/_Stage2 (cudaGJKEPA.cu L1468-1558):
// the lost/found contact-manager compaction that reports touch changes to the
// host.  The input is a 2*numPairs flag array (first half = lost-touch flags,
// second half = found-touch flags, values 0/1):
//   Stage1 (scanKernel1of2): per-block exclusive prefix sum of the flag VALUES
//     into tempRunsum, per-block totals into blockAccumulation.
//   Stage2 (scanKernel2of2 + CompactContactManagersFunctor): the device-wide
//     exclusive prefix outOffset = tempRunsum[idx] + blockPrefix; for each idx
//     with flags[idx] != 0, compacted[outOffset] copies nbPatches / prevPatches
//     / statusFlag of allOutputManagers[idx % numPairs] (nbContacts is NOT
//     copied — commented out in CUDA) and contactManagerPtrs[outOffset] =
//     contactManagerArray[idx % numPairs]; at idx == numPairs the running
//     offset (= number of lost-touch flags) is written to
//     lostAndTotalReportedPairsCount->x, and the grand total (lost+found) to ->y.
//
// Data model (engine launch path): allOutputManagers is the real 48-byte
// PxsContactManagerOutput array — its packed flags word (allflagsStart |
// nbPatches<<8 | statusFlag<<16 | prevPatches<<24) sits at byte 32 = word 8.
// The compacted output uses the real PxsContactManagerOutputCounts little-endian
// byte layout (nbPatches | prevPatches<<8 | statusFlag<<16, unused byte 0), one
// 4-byte word.  Contact-manager pointer arrays are real 64-bit device pointers,
// copied verbatim (Ptr<uint64_t>).

// launch config: PxgNarrowPhaseBlockDims::COMPACT_LOST_FOUND_PAIRS = 512,
// PxgNarrowPhaseGridDims::COMPACT_LOST_FOUND_PAIRS = 32 (PxgNpKernelIndices.h)
static const uint BLOCK = 512;
static const uint NUM_BLOCKS = 32;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 16

// P5 Step 4 (BDA, engine launch path).  Both stages launch via the engine's
// sized form (PxgNarrowphaseCore.cpp ~L344), so each argument becomes a field
// of a single push-constant blob (all pointers -> uint64_t; every array is
// walked through Ptr<T>).  PxsContactManagerOutput is a 48-byte struct: the
// packed flags word (allflagsStart | nbPatches<<8 | statusFlag<<16 |
// prevPatches<<24) lives at byte 32 = uint word 8.  PxsContactManagerOutputCounts
// is one 4-byte word (nbPatches | prevPatches<<8 | statusFlag<<16).  The
// contact-manager pointer arrays are real 64-bit device pointers (copied
// verbatim).  uint2 lostAndTotalReportedPairsCount is {x@0, y@4} = Ptr<uint>.
static const uint OUTPUT_WORDS = 12; // sizeof(PxsContactManagerOutput)/4

// Stage 1 blob {inputFlags(8), tempRunsum(8), blockAccum(8), numPairs(4)} = 32 B
struct LostFoundStage1Params
{
    uint64_t inputFlags;  // @0  const PxU32*
    uint64_t tempRunsum;  // @8  PxU32*
    uint64_t blockAccum;  // @16 PxU32*
    uint numPairs;        // @24
}
// Stage 2 blob {inputFlags,tempRunsum,allOutputManagers,compacted,ptrs,lostTotal,
//               blockAccum (7x8=56), numPairs(4)@56, cmArray(8)@64} = 72 B
struct LostFoundStage2Params
{
    uint64_t inputFlags;        // @0  const PxU32*
    uint64_t tempRunsum;        // @8  const PxU32*
    uint64_t allOutputManagers; // @16 const PxsContactManagerOutput*
    uint64_t compacted;         // @24 PxsContactManagerOutputCounts*
    uint64_t ptrs;              // @32 PxsContactManager** (outputManagersPtrs)
    uint64_t lostTotal;         // @40 uint2*
    uint64_t blockAccum;        // @48 PxU32*
    uint numPairs;              // @56
    uint64_t cmArray;           // @64 PxsContactManager** (contactManagerArray)
}

[[vk::push_constant]]
ConstantBuffer<LostFoundStage1Params> gP1;
[[vk::push_constant]]
ConstantBuffer<LostFoundStage2Params> gP2;

groupshared uint gsPairs[BLOCK];
groupshared uint gsWarpPairs[WARPS_PER_BLOCK];
groupshared uint gsBlockPairs;
groupshared uint gsBlockAccum[NUM_BLOCKS];

void warpScanPairs(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        const uint add = (lane >= off) ? gsPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsPairs[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

void scanWarpTotalsPairs(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    const uint p0 = active ? gsWarpPairs[tid] : 0u;
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

// ---------------------------------------------------------------------------
// Stage 1: per-block exclusive scan of the flags, block totals out
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void prepareLostFoundPairs_Stage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = 2u * gP1.numPairs;
    const uint warpIndex = tid / WARP;

    Ptr<uint> gFlags = Ptr<uint>(gP1.inputFlags);
    Ptr<uint> gRunSum = Ptr<uint>(gP1.tempRunsum);
    Ptr<uint> gBlockTotals = Ptr<uint>(gP1.blockAccum);

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockPairs = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        uint val = 0u;
        if (workIndex < n)
            val = gFlags[workIndex];

        gsPairs[tid] = val;
        GroupMemoryBarrierWithGroupSync();
        warpScanPairs(tid);
        const uint accum = gsPairs[tid] - val; // exclusive within warp

        if ((tid & (WARP - 1)) == WARP - 1)
            gsWarpPairs[warpIndex] = accum + val;

        const uint prevBlock = gsBlockPairs;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotalsPairs(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < n)
            gRunSum[workIndex] = accum + prevBlock + gsWarpPairs[warpIndex];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        gBlockTotals[blockIdx] = gsBlockPairs;
}

// ---------------------------------------------------------------------------
// Stage 2: cross-block prefix + compact + counters
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void prepareLostFoundPairs_Stage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint numPairs = gP2.numPairs;
    const uint n = 2u * numPairs;

    Ptr<uint> gFlags = Ptr<uint>(gP2.inputFlags);
    Ptr<uint> gRunSum = Ptr<uint>(gP2.tempRunsum);
    Ptr<uint> gMgr = Ptr<uint>(gP2.allOutputManagers);        // 48-byte struct word view
    Ptr<uint> gCompacted = Ptr<uint>(gP2.compacted);          // 4-byte Counts view
    Ptr<uint64_t> gPtrs = Ptr<uint64_t>(gP2.ptrs);            // PxsContactManager*
    Ptr<uint64_t> gMgrHandles = Ptr<uint64_t>(gP2.cmArray);  // PxsContactManager*
    Ptr<uint> gLostTotal = Ptr<uint>(gP2.lostTotal);         // uint2 {x,y}
    Ptr<uint> gBlockTotals = Ptr<uint>(gP2.blockAccum);

    if (tid < NUM_BLOCKS)
        gsBlockAccum[tid] = gBlockTotals[tid];
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint acc = 0u;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const uint t = gsBlockAccum[i];
            gsBlockAccum[i] = acc;
            acc += t;
        }
        // WriteValueFunctor: total reported pairs (lost + found) -> y
        if (blockIdx == 0)
            gLostTotal[1] = acc;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockAccum = gsBlockAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < n)
        {
            const uint outOffset = gRunSum[workIndex] + blockAccum;

            // CompactContactManagersFunctor::operator()(idx, outOffset)
            if (gFlags[workIndex] != 0u)
            {
                const uint index = workIndex % numPairs;
                // real PxsContactManagerOutput packed flags word (byte 32 = word 8):
                //   allflagsStart | nbPatches<<8 | statusFlag<<16 | prevPatches<<24
                const uint w = gMgr[index * OUTPUT_WORDS + 8u];
                const uint nbPatches = (w >> 8u) & 0xFFu;
                const uint statusFlag = (w >> 16u) & 0xFFu;
                const uint prevPatches = (w >> 24u) & 0xFFu;
                // PxsContactManagerOutputCounts byte layout (nbContacts NOT copied)
                gCompacted[outOffset] = nbPatches | (prevPatches << 8u) | (statusFlag << 16u);
                gPtrs[outOffset] = gMgrHandles[index];
            }

            // number of lost-touch flags (exclusive prefix at the half boundary) -> x
            if (workIndex == numPairs)
                gLostTotal[0] = outOffset;
        }
    }
}
