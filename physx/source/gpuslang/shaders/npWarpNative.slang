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

// npWarpNative.slang — RUNTIME_PLAN.md P2.1: the NATIVE subgroup path of the
// warp-32 primitive library.  This calls every native-lowering primitive so
// slangc + spirv-val exercise the WaveShuffle / WaveActiveBallot / wave
// arithmetic lowerings that the emulation battery (npWarpLib) cannot reach.
//
// It is COMPILE/lowering validation only: the 32-lane CUDA warp semantics are
// correct only when the pipeline is built with requiredSubgroupSize == 32,
// which CI (llvmpipe, fixed subgroup 8) cannot provide — hardware execution
// is P5.  Building this shader here guarantees the native branches stay
// well-formed as the library evolves.

#define PXG_WARP_NATIVE 1
#define PXG_WARP_BLOCK 32
#include "pxgSlangWarpInc.slang"

struct NwParams { uint numThreads; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<NwParams> gParams;

RWStructuredBuffer<uint> gOut;   // binding 0

[shader("compute")]
[numthreads(32, 1, 1)]
void warpNativeProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    const uint lane = pxgLaneIdx(tid);
    const float fv = float(lane) - 7.0;
    const uint uv = lane * 2654435761u;

    uint acc = 0u;
    acc ^= asuint(pxgShflSyncF(PXG_FULL_MASK, tid, fv, int(lane ^ 1u), 32u));
    acc ^= asuint(pxgShflUpSyncF(PXG_FULL_MASK, tid, fv, 1u, 32u));
    acc ^= asuint(pxgShflDownSyncF(PXG_FULL_MASK, tid, fv, 2u, 32u));
    acc ^= asuint(pxgShflXorSyncF(PXG_FULL_MASK, tid, fv, 4u, 32u));
    acc ^= pxgBallotSync(PXG_FULL_MASK, tid, (lane & 1u) == 0u);
    acc ^= pxgWarpCountAndBroadcast(PXG_FULL_MASK, tid, lane < 10u);
    acc ^= asuint(pxgWarpReduceAddF(PXG_FULL_MASK, tid, fv, 5u));
    acc ^= asuint(pxgWarpReduceMinF(PXG_FULL_MASK, tid, fv, 5u));
    acc ^= asuint(pxgWarpReduceMaxF(PXG_FULL_MASK, tid, fv, 5u));
    acc ^= pxgWarpReduceAddU(PXG_FULL_MASK, tid, uv, 5u);
    acc ^= pxgWarpReduceOrU(PXG_FULL_MASK, tid, uv, 5u);
    acc ^= pxgWarpReduceAndU(PXG_FULL_MASK, tid, uv, 5u);
    acc ^= pxgWarpScanAddInclusiveU(PXG_FULL_MASK, tid, uv, 5u);
    acc ^= pxgWarpScanAddExclusiveU(tid, uv);
    uint winner;
    acc ^= asuint(pxgWarpReduceMaxWinnerF(PXG_FULL_MASK, tid, fv, 5u, winner));
    acc ^= winner;
    acc ^= pxgAnySync(PXG_FULL_MASK, tid, lane == 3u) ? 1u : 0u;
    acc ^= pxgAllSync(PXG_FULL_MASK, tid, lane < 32u) ? 2u : 0u;

    if (tid < gParams.numThreads)
        gOut[tid] = acc;
}
