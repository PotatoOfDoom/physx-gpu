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

// pcmManifoldBda.slang
//
// The PCM gate of midphaseGeneratePairs, wired up as a standalone dispatch so
// it can be tested BEFORE the 200-line traversal half exists.
//
// This is not an engine kernel and is deliberately absent from the manifest:
// when convexTrimeshMidphaseBda lands it will `import pxgSlangPcmManifold` and
// inline this same sequence.  Extracting the gate first is the same move P8.6
// made for the GJK and EPA drivers and P8.15 made for the cloth midphase, taken
// here because the gate is the only part of the kernel whose bugs a
// single-dispatch fixture cannot see (see the module header).
//
// One group per contact manager, [numthreads(32,1,1)] — the port's sanctioned
// 32-lane virtual warp.  The CUDA launches (32, MIDPHASE_WARPS_PER_BLOCK=2, 1)
// and derives cmIdx from threadIdx.y; here one group IS one warp, so cmIdx is
// the group id.  That is a launch-geometry difference, not a behavioural one:
// the CUDA's two warps per block never interact (each has its own
// s_warpScratch slot and its own cmIdx), so splitting them into separate groups
// is exactly equivalent.

import pxgSlangPcmManifold;

static const uint WARP = 32u;

// ---------------------------------------------------------------------------
// invalidateManifold alone — a PURE read, one lane per record.
//   argSizes {8,8,4,4,4,16,12} is NOT the engine pack; this entry exists only
//   for the fixture, so the blob is laid out for convenience:
//     manifolds, out, count, minMargin, ratio, aToB_q, aToB_p
// ---------------------------------------------------------------------------
struct InvalidateParams
{
    uint64_t manifolds; // @0   PxgPersistentContactMultiManifold*
    uint64_t outFlags;  // @8   uint* — one 0/1 per record
    uint     count;     // @16
    float    minMargin; // @20
    float    ratio;     // @24
    float    _pad28;    // @28
    float4   aToB_q;    // @32
    float4   aToB_p;    // @48 (.w unused) -> 64-byte blob
}

[[vk::push_constant]]
ConstantBuffer<InvalidateParams> gInv;

[shader("compute")]
[numthreads(64, 1, 1)]
void pcmInvalidateLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gInv.count) return;

    const uint64_t m = gInv.manifolds + uint64_t(i) * PCM_STRIDE;
    const bool inv = mmInvalidateManifold(m, gInv.aToB_q,
                                          float3(gInv.aToB_p.x, gInv.aToB_p.y, gInv.aToB_p.z),
                                          gInv.minMargin, gInv.ratio);
    Ptr<uint>(gInv.outFlags)[i] = inv ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// refreshManifolds in full — the per-lane slot refresh, the group-wide OR that
// stands in for __any_sync(FULL_MASK, lostContact), and the lane-0
// `mNbManifolds = 0` that the OR guards.
//
// The barrier discipline is the point of this entry.  Both
// GroupMemoryBarrierWithGroupSync calls sit in UNIFORM control flow: the early
// `return` for out-of-range groups happens before either, and mmRefreshSlot's
// own guards are plain value returns, not jumps around a barrier.  The CUDA has
// the same property for the same reason — its __any_sync is placed OUTSIDE the
// `if (curManifoldIdx < numManifolds)` block.
// ---------------------------------------------------------------------------
struct RefreshParams
{
    uint64_t manifolds; // @0
    uint64_t outLost;   // @8   uint* — one 0/1 per record (the __any_sync result)
    uint     count;     // @16
    float    projectBreakingThreshold; // @20
    float    _pad24;    // @24
    float    _pad28;    // @28
    float4   aToB_q;    // @32
    float4   aToB_p;    // @48 -> 64-byte blob
}

[[vk::push_constant]]
ConstantBuffer<RefreshParams> gRef;

groupshared uint sAnyLost;

[shader("compute")]
[numthreads(32, 1, 1)]
void pcmRefreshLaunch(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint cmIdx = gid.x;
    if (cmIdx >= gRef.count) return;   // group-uniform: every lane takes it or none does

    const uint lane = gtid.x;
    const uint64_t m = gRef.manifolds + uint64_t(cmIdx) * PCM_STRIDE;

    if (lane == 0) sAnyLost = 0u;
    GroupMemoryBarrierWithGroupSync();

    // Every lane calls this, including lanes 24..31, which fall out on the slot
    // guards and return false — exactly as the CUDA's lanes 24..31 do.
    const bool lost = mmRefreshSlot(m, gRef.aToB_q,
                                    float3(gRef.aToB_p.x, gRef.aToB_p.y, gRef.aToB_p.z),
                                    gRef.projectBreakingThreshold, lane);
    if (lost) InterlockedOr(sAnyLost, 1u);

    GroupMemoryBarrierWithGroupSync();

    // The write-backs inside mmRefreshSlot must be visible before lane 0 acts on
    // the OR; the barrier above orders both the groupshared flag and the device
    // writes for this group.
    if (lane == 0)
    {
        const uint any = sAnyLost;
        if (any != 0u)
            Ptr<uint>(m + PCM_NBMANIFOLDS)[0] = 0u;   // manifold.cuh:97
        Ptr<uint>(gRef.outLost)[cmIdx] = any;
    }
}
