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
// Data model: the manager's three copied PxU8 fields arrive packed in one uint
// (nbPatches | statusFlag<<8 | prevPatches<<16 | nbContacts<<24 — the top byte
// stands in for the NOT-copied fields); the compacted output uses the real
// PxsContactManagerOutputCounts little-endian byte layout
// (nbPatches | prevPatches<<8 | statusFlag<<16, unused byte 0).  Contact-manager
// pointers are modelled as opaque 32-bit handles (the kernel only copies them).

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8

struct LostFoundParams
{
    uint numPairs; // scan length is 2*numPairs
    uint pad0, pad1, pad2;
}

[[vk::push_constant]]
ConstantBuffer<LostFoundParams> gParams;

RWStructuredBuffer<uint> gFlags;       // binding 0: inputFlagsArray [2*numPairs]
RWStructuredBuffer<uint> gRunSum;      // binding 1: tempRunsumArray [2*numPairs]
RWStructuredBuffer<uint> gBlockTotals; // binding 2: blockAccumulationArray [NUM_BLOCKS]
RWStructuredBuffer<uint> gMgrWords;    // binding 3: allOutputManagers (packed) [numPairs]
RWStructuredBuffer<uint> gMgrHandles;  // binding 4: contactManagerArray (opaque) [numPairs]
RWStructuredBuffer<uint> gCompacted;   // binding 5: compactedOutputManagers (packed Counts)
RWStructuredBuffer<uint> gPtrs;        // binding 6: outputManagersPtrs (opaque)
RWStructuredBuffer<uint> gLostTotal;   // binding 7: lostAndTotalReportedPairsCount {x, y}

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
[numthreads(256, 1, 1)]
void lostFoundStage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = 2u * gParams.numPairs;
    const uint warpIndex = tid / WARP;

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
[numthreads(256, 1, 1)]
void lostFoundStage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint numPairs = gParams.numPairs;
    const uint n = 2u * numPairs;

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
                const uint w = gMgrWords[index];
                const uint nbPatches = w & 0xFFu;
                const uint statusFlag = (w >> 8u) & 0xFFu;
                const uint prevPatches = (w >> 16u) & 0xFFu;
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
