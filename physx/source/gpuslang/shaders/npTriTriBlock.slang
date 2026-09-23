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

// npTriTriBlock.slang -- P9.83b-5 (13/n).
//
//   Fixture entry for pxgSlangTriTriBlock: the two reduction.cuh block helpers,
//   the warp minIndex, and the cross-warp seed pick.
//
// ONE WORKGROUP OF 1024 THREADS PER CASE, which is new to this tranche and is the
// point.  Every fixture before it in P9.83b ran 64-thread groups over independent
// cases, because every body before it was serial.  These four are COLLECTIVES:
// their result depends on what the other 1023 threads did, and a fixture that ran
// them on 64 threads would certify a block shape the kernel never launches.
//
// FULLY BDA, no RWStructuredBuffer, for the reason npTriTriAccept.slang records.
//
// WHAT THE CASE SET HAS TO CONTAIN, beyond random bits:
//
//   * A warp with NO candidates, so minIndex runs on an EMPTY mask and returns
//     lowestSetIndex(0) = 0xFFFFFFFF.  That is the one input where the sentinel
//     appears, and :908 pairs it with a lane index it can never equal.
//   * A block where EVERY warp is empty, so the seed gate (`countersS[minIBlock]
//     != 0`) is the thing that stops the loop rather than an accident.
//   * TWO WARPS SHARING THE BLOCK MINIMUM, so the second minIndex has a tie and
//     the lowest-index rule is exercised rather than assumed.
//   * An UNMASKED lane whose separation equals its warp's masked minimum, which
//     sets a ballot bit that only the `& mask` removes.
//   * An all-true and an all-false predicate for the count and the scan, so the
//     0 and 1024 ends of both are real inputs.
//
// No NoContraction stamp, for the reason in testNpCrsWrapper2.cpp.

// ttbWarpReduceSerial calls crsContactReduce2, which lives in this MODULE; the
// block file is a textual include and cannot import it on its includer's behalf.
import pxgSlangContactReduceSerial;

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"
#include "pxgSlangTriTriBlock.slang"

struct TtbPush
{
    uint64_t cases;    // @0   TtbCase*
    uint64_t out;      // @8   uint*
    uint     numCases; // @16
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<TtbPush> gTb;

// 4480 bytes: three 1024-bit predicates then 1024 separations.
static const uint TTB_CASE_STRIDE = 4480u;
static const uint TTB_OUT_STRIDE = 1152u;   // uints

[shader("compute")]
[numthreads(1024, 1, 1)]
void ttbProbe(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint c = gid.x;
    if (c >= gTb.numCases) return;

    const uint tid = gtid.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;

    const uint64_t base = gTb.cases + uint64_t(c) * uint64_t(TTB_CASE_STRIDE);
    Ptr<uint> bits = Ptr<uint>(base);
    Ptr<float> seps = Ptr<float>(base + 384UL);

    const bool bitA = ((bits[warp] >> lane) & 1u) != 0u;              // accept2
    const bool bitB = ((bits[32u + warp] >> lane) & 1u) != 0u;        // emits
    const bool bitC = ((bits[64u + warp] >> lane) & 1u) != 0u;        // candidate
    const float sep = seps[tid];

    // :812 -- the count, then the caller's own sync for countersS reuse (:813).
    const uint nbUsed = ttbBlockCountAndBroadcast(tid, bitA);
    GroupMemoryBarrierWithGroupSync();

    // :820 -- the scan.  The CUDA does NOT sync here (:821 says so); this fixture
    // does, because it reuses gTtbCounters for the seed pick immediately after,
    // where the kernel has the tempPoints writes and their barrier in between.
    uint totalSum;
    const uint offset = ttbBlockScanExclusive(tid, bitB, totalSum);
    GroupMemoryBarrierWithGroupSync();

    // :881 -- candidateMask, then the seed pick.
    const uint candidateMask = pxgBallotSync(PXG_FULL_MASK, tid, bitC);
    uint minI;
    const TtbSeed seed = ttbBlockSeedSelect(tid, sep, candidateMask, minI);

    Ptr<uint> o = Ptr<uint>(gTb.out + uint64_t(c) * uint64_t(TTB_OUT_STRIDE) * 4UL);

    if (tid == 0u)
    {
        o[0] = nbUsed;
        o[1] = totalSum;
        o[2] = seed.minIBlock;
        o[3] = asuint(seed.minSep);
        o[4] = seed.winnerCount;
        o[5] = seed.hasCandidates ? 1u : 0u;
    }
    if (lane == 0u)
    {
        o[6u + warp] = minI;                     // the seeding lane of this warp
        o[38u + warp] = asuint(gTtbMinSep[warp]);// this warp's own minimum
        o[70u + warp] = candidateMask;
    }
    o[102u + tid] = offset;
}

// ---------------------------------------------------------------------------
// P9.83b-5 (14/n): ttbWarpReduceSerial, the per-warp contactReduce2 of :810/:931.
// ---------------------------------------------------------------------------
// Same push struct, its own case layout (16656 bytes):
//   @0      float4 pts[1024]    xyz = position, w = separation
//   @16384  uint   masks[32]    first call's per-warp input mask
//   @16512  uint   masks2[32]   second call's per-warp input mask
//   @16640  float4 normalBias   xyz = patch normal, w = clusterBias
//
// TWO CALLS BACK TO BACK, with different masks.  The first exercises the staging;
// the second exists for the trailing barrier and the slot reuse -- the result of
// call one and the input of call two travel through the same gTtbReduceMask slot.
//
// EVERY THREAD writes its own copy of the result, not lane 0 alone: the property
// under test is that warp w's answer reaches all 32 of warp w's lanes, and a
// fixture that read one lane per warp would pass a broadcast that reached one.
static const uint TTB_RED_CASE_STRIDE = 16656u;
static const uint TTB_RED_OUT_STRIDE = 2048u;   // uints

[shader("compute")]
[numthreads(1024, 1, 1)]
void ttbReduceProbe(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint c = gid.x;
    if (c >= gTb.numCases) return;

    const uint tid = gtid.x;
    const uint warp = tid >> 5;

    const uint64_t base = gTb.cases + uint64_t(c) * uint64_t(TTB_RED_CASE_STRIDE);
    const float4 p = Ptr<float4>(base)[tid];
    const uint mask1 = Ptr<uint>(base + 16384UL)[warp];
    const uint mask2 = Ptr<uint>(base + 16512UL)[warp];
    const float4 nb = Ptr<float4>(base + 16640UL)[0];

    const uint r1 = ttbWarpReduceSerial(tid, p.xyz, p.w, nb.xyz, mask1, nb.w);
    const uint r2 = ttbWarpReduceSerial(tid, p.xyz, p.w, nb.xyz, mask2, nb.w);

    Ptr<uint> o = Ptr<uint>(gTb.out + uint64_t(c) * uint64_t(TTB_RED_OUT_STRIDE) * 4UL);
    o[tid] = r1;
    o[1024u + tid] = r2;
}
