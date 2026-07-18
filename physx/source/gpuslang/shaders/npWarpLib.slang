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

// npWarpLib.slang — RUNTIME_PLAN.md P2.1 bit-level battery for the warp-32
// primitive library (pxgSlangWarpInc.slang, emulation path).  Two warps per
// workgroup prove tile isolation; one scripted test per workgroup; every
// output is compared bitwise against a CPU model of the CUDA semantics.
// Both entries use the same four buffers (script, lane values, predicates,
// generic uint4 output) so the binding layout is identical.

#define PXG_WARP_BLOCK 64
#include "pxgSlangWarpInc.slang"

struct WlParams { uint numTests; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<WlParams> gParams;

// Script per test: x = mask, y = log2width, z = srcLane (as uint bit
// pattern of the int), w = delta | (xorMask << 8).
RWStructuredBuffer<uint4> gScript;   // binding 0
RWStructuredBuffer<float> gVals;     // binding 1: 64 lane values per test
RWStructuredBuffer<uint> gPred;      // binding 2: 64 predicate flags per test
RWStructuredBuffer<uint4> gOut;      // binding 3: 3 x uint4 per thread

// gOut record 0 of thread: shfl, up, down, xor (float bits).
[shader("compute")]
[numthreads(64, 1, 1)]
void wlShuffle(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint test = gid.x;
    const uint tid = gtid.x;
    const uint base = test * 64u;

    const uint4 s = gScript[test];
    const uint mask = s.x;
    const uint width = 1u << s.y;
    const int srcLane = int(s.z);
    const uint delta = s.w & 0xffu;
    const uint xorMask = (s.w >> 8) & 0xffu;

    const float v = gVals[base + tid];

    uint4 o;
    o.x = asuint(pxgShflSyncF(mask, tid, v, srcLane, width));
    o.y = asuint(pxgShflUpSyncF(mask, tid, v, delta, width));
    o.z = asuint(pxgShflDownSyncF(mask, tid, v, delta, width));
    o.w = asuint(pxgShflXorSyncF(mask, tid, v, xorMask, width));
    gOut[(base + tid) * 3u] = o;
}

// gOut records 0..2 of thread:
//   r0 = ballot, rankInMask, countAndBroadcast, scanAddExclusive(full warp)
//   r1 = scanAddInclusive(width), reduceAddU(width), reduceOrU(width), maxWinnerLane
//   r2 = float bits of reduceAddF, reduceMinF, reduceMaxF, maxWinnerValue
[shader("compute")]
[numthreads(64, 1, 1)]
void wlBallot(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint test = gid.x;
    const uint tid = gtid.x;
    const uint base = test * 64u;

    const uint4 s = gScript[test];
    const uint mask = s.x;
    const uint log2width = s.y;

    const float v = gVals[base + tid];
    const uint uv = asuint(v);
    const bool pred = gPred[base + tid] != 0u;
    const uint lane = pxgLaneIdx(tid);

    uint4 r0;
    r0.x = pxgBallotSync(mask, tid, pred);
    r0.y = pxgWarpRankInMask(mask, lane);
    r0.z = pxgWarpCountAndBroadcast(mask, tid, pred);
    r0.w = pxgWarpScanAddExclusiveU(tid, uv & 0xffffu);

    uint4 r1;
    r1.x = pxgWarpScanAddInclusiveU(mask, tid, uv & 0xffffu, log2width);
    r1.y = pxgWarpReduceAddU(mask, tid, uv & 0xffffu, log2width);
    r1.z = pxgWarpReduceOrU(mask, tid, uv, log2width);

    uint4 r2;
    uint winner;
    r2.w = asuint(pxgWarpReduceMaxWinnerF(mask, tid, v, log2width, winner));
    r1.w = winner;
    r2.x = asuint(pxgWarpReduceAddF(mask, tid, v, log2width));
    r2.y = asuint(pxgWarpReduceMinF(mask, tid, v, log2width));
    r2.z = asuint(pxgWarpReduceMaxF(mask, tid, v, log2width));

    gOut[(base + tid) * 3u + 0u] = r0;
    gOut[(base + tid) * 3u + 1u] = r1;
    gOut[(base + tid) * 3u + 2u] = r2;
}
