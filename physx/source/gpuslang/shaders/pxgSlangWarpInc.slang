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

// pxgSlangWarpInc.slang — RUNTIME_PLAN.md P2.1: the warp-32 primitive library.
//
// Replicates the CUDA warp-intrinsic semantics (PxgCommonDefines.h WARP_SIZE
// 32 / FULL_MASK, reduction.cuh, shuffle.cuh) that ~1900 call sites across
// the GPU modules depend on.  A "warp" is a tile of 32 consecutive threads
// of the workgroup (lane = tid & 31); kernels using this library MUST use a
// workgroup size that is a multiple of 32 (every PhysX CUDA block is).
//
// Preprocessor-included (NOT imported) so the path switch and the shared
// scratch declaration land inside each kernel's translation unit:
//
//   PXG_WARP_NATIVE == 0 (default): shared-memory emulation.  Bit-exact CUDA
//     warp semantics at ANY subgroup size (llvmpipe's fixed 8 included; risk
//     R1).  Every primitive is a store/barrier/read/barrier round through a
//     per-workgroup scratch array, so CALL SITES MUST BE WORKGROUP-UNIFORM
//     (a stronger requirement than CUDA's warp-convergence; kernels with
//     warp-divergent primitive calls must restructure on this path, e.g. by
//     launching one warp per workgroup).  Define PXG_WARP_BLOCK to the
//     workgroup size before including (default 64).
//
//   PXG_WARP_NATIVE == 1: native subgroup ops, ONLY valid in pipelines built
//     with requiredSubgroupSize == 32 + REQUIRE_FULL_SUBGROUPS
//     (createPipelineEx) and fully converged calls — the same precondition
//     CUDA places on __*_sync(FULL_MASK).  Mask arguments are honored the
//     CUDA way: ballots are ANDed with the mask, any/all evaluate mask lanes
//     only, shuffles move data irrespective of the mask (matching CUDA when
//     the mask-participation contract holds).  Pending hardware validation
//     (P5): CI cannot execute this path at 32 (llvmpipe window is [8,8]).
//
// CUDA semantics implemented (width = power of two <= 32, segments aligned;
// delta and xorMask are taken mod 32 like PTX shfl.sync's bval = b[4:0]):
//   pxgShflSync(mask,v,srcLane,width)   : src wraps modulo width in-segment
//   pxgShflUpSync(mask,v,delta,width)   : own value when lane-in-seg < delta
//   pxgShflDownSync(mask,v,delta,width) : own value when lane-in-seg+delta >= width
//   pxgShflXorSync(mask,v,xorMask,width): own value when the XOR partner lies
//                                         past the END of the own segment (an
//                                         EARLIER segment's lane IS read)
//   pxgBallotSync(mask,pred)            : bit i iff (mask has i) && pred(lane i)
//   pxgAnySync/pxgAllSync(mask,pred)    : over mask lanes
//   pxgWarpRankInMask(mask,lane)        : popc(mask & ((1<<lane)-1))
//   pxgWarpCountAndBroadcast(mask,pred) : popc(ballot)
//   pxgWarpReduce{Add,Min,Max,Or,And}   : XOR-butterfly, log2width param,
//                                         result broadcast to all lanes
//                                         (reduction.cuh warpReduction<OP>)
//   pxgWarpScanAddInclusive             : up-shuffle inclusive scan
//   pxgWarpScanAddExclusive             : reduction.cuh warpScanExclusive
//   pxgFfs/pxgLowestSetIndex/pxgHighestSetIndex : bit conventions incl. 0

#ifndef PXG_SLANG_WARP_INC
#define PXG_SLANG_WARP_INC

#ifndef PXG_WARP_NATIVE
#define PXG_WARP_NATIVE 0
#endif

#ifndef PXG_WARP_BLOCK
#define PXG_WARP_BLOCK 64
#endif

#define PXG_WARP_SIZE 32u
#define PXG_FULL_MASK 0xffffffffu

uint pxgLaneIdx(uint tid)
{
#if PXG_WARP_NATIVE
    return WaveGetLaneIndex();
#else
    return tid & (PXG_WARP_SIZE - 1u);
#endif
}

// ---- bit conventions (exact CUDA behavior including the zero cases) ----

uint pxgPopc(uint v) { return countbits(v); }

// __ffs: 1-based index of lowest set bit; 0 when v == 0.
int pxgFfs(uint v) { return v == 0u ? 0 : int(firstbitlow(v)) + 1; }

// utils.cuh lowestSetIndex = __ffs(v) - 1  (-1 when v == 0).
int pxgLowestSetIndex(uint v) { return pxgFfs(v) - 1; }

// utils.cuh clearLowestSetBit = v & (v - 1)  (0 when v == 0, since 0 - 1 wraps
// to 0xFFFFFFFF and the AND leaves 0).  The `for (m = mask; m; m = clear(m))`
// idiom that iterates a ballot's set bits depends on that zero case terminating.
uint pxgClearLowestSetBit(uint v) { return v & (v - 1u); }

// utils.cuh highestSetIndex = 31 - __clz(v)  (-1 when v == 0).
int pxgHighestSetIndex(uint v) { return v == 0u ? -1 : int(firstbithigh(v)); }

#if !PXG_WARP_NATIVE

// One scratch array per workgroup; every primitive is
// store / barrier / read / barrier, so the array is reusable immediately.
groupshared uint gPxgWarpScratch[PXG_WARP_BLOCK];

uint pxgWarpScratchExchange(uint tid, uint value, uint srcTid)
{
    gPxgWarpScratch[tid] = value;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gPxgWarpScratch[srcTid];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

#endif

// ---- __syncwarp ----
// Emulation: a workgroup barrier (stronger; workgroup-uniform call sites
// required).  Native: wave ops are implicitly synchronized under the
// required-full-subgroups contract; shared-memory ordering across the warp
// is covered by the workgroup barrier the surrounding kernel already needs.
void pxgSyncWarp(uint mask)
{
#if !PXG_WARP_NATIVE
    GroupMemoryBarrierWithGroupSync();
#endif
}

// ---- shuffles (uint core; float wrappers bit-cast through it) ----

uint pxgShflSyncU(uint mask, uint tid, uint value, int srcLane, uint width)
{
    const uint lane = pxgLaneIdx(tid);
    const uint seg = lane & ~(width - 1u);
    const uint src = seg + (uint(srcLane) & (width - 1u));
#if PXG_WARP_NATIVE
    return WaveShuffle(value, int(src));
#else
    return pxgWarpScratchExchange(tid, value, (tid & ~(PXG_WARP_SIZE - 1u)) + src);
#endif
}

// P9.89: the up/down/xor addressing is PTX shfl.sync's (sm_30_intrinsics.hpp
// builds c = ((32-width)<<8)|0x1f for down/bfly and (32-width)<<8 for up):
//   bval = b[4:0]                          -- delta / xorMask truncated to 5 bits
//   minLane = segment start, maxLane = segment END (down/bfly; up clamps to 0)
//   up:   j = lane - bval, valid iff j >= minLane
//   down: j = lane + bval, valid iff j <= maxLane
//   bfly: j = lane ^ bval, valid iff j <= maxLane
//   invalid -> own value.
// So a bfly partner in an EARLIER segment is read, not clamped (Programming
// Guide: "able to access elements from earlier groups ... later groups ...
// their own value"), and delta/xorMask 33 act as 1.  Before P9.89 the port
// clamped a partner in ANY other segment and used the untruncated
// delta/xorMask.  Latent: every engine caller passes delta < 32 and an xor
// radius < width, where both rules agree.  The addressing is computed ONCE,
// outside the path switch, so the native branch differs only in transport and
// the emulated battery (npWarpLib) pins its addressing too.
uint pxgShflUpSyncU(uint mask, uint tid, uint value, uint delta, uint width)
{
    const uint lane = pxgLaneIdx(tid);
    const uint bval = delta & (PXG_WARP_SIZE - 1u);   // P9.89: PTX bval = b[4:0]
    const uint laneInSeg = lane & (width - 1u);
    const bool clamped = laneInSeg < bval;            // j = lane - bval < minLane
    const uint src = clamped ? lane : lane - bval;
#if PXG_WARP_NATIVE
    const uint r = WaveShuffle(value, int(src));
#else
    const uint r = pxgWarpScratchExchange(tid, value, (tid & ~(PXG_WARP_SIZE - 1u)) + src);
#endif
    return clamped ? value : r;
}

uint pxgShflDownSyncU(uint mask, uint tid, uint value, uint delta, uint width)
{
    const uint lane = pxgLaneIdx(tid);
    const uint bval = delta & (PXG_WARP_SIZE - 1u);   // P9.89: PTX bval = b[4:0]
    const uint laneInSeg = lane & (width - 1u);
    const bool clamped = laneInSeg + bval >= width;   // j = lane + bval > maxLane
    const uint src = clamped ? lane : lane + bval;
#if PXG_WARP_NATIVE
    const uint r = WaveShuffle(value, int(src));
#else
    const uint r = pxgWarpScratchExchange(tid, value, (tid & ~(PXG_WARP_SIZE - 1u)) + src);
#endif
    return clamped ? value : r;
}

uint pxgShflXorSyncU(uint mask, uint tid, uint value, uint xorMask, uint width)
{
    const uint lane = pxgLaneIdx(tid);
    const uint src = lane ^ (xorMask & (PXG_WARP_SIZE - 1u));   // P9.89: j = lane ^ b[4:0]
    const uint maxLane = (lane & ~(width - 1u)) | (width - 1u); // own segment's END
    const bool clamped = src > maxLane;                         // later segment only
#if PXG_WARP_NATIVE
    const uint r = WaveShuffle(value, int(src));
#else
    const uint r = pxgWarpScratchExchange(tid, value, (tid & ~(PXG_WARP_SIZE - 1u)) + src);
#endif
    return clamped ? value : r;
}

float pxgShflSyncF(uint mask, uint tid, float value, int srcLane, uint width)
{
    return asfloat(pxgShflSyncU(mask, tid, asuint(value), srcLane, width));
}
float pxgShflUpSyncF(uint mask, uint tid, float value, uint delta, uint width)
{
    return asfloat(pxgShflUpSyncU(mask, tid, asuint(value), delta, width));
}
float pxgShflDownSyncF(uint mask, uint tid, float value, uint delta, uint width)
{
    return asfloat(pxgShflDownSyncU(mask, tid, asuint(value), delta, width));
}
float pxgShflXorSyncF(uint mask, uint tid, float value, uint xorMask, uint width)
{
    return asfloat(pxgShflXorSyncU(mask, tid, asuint(value), xorMask, width));
}
int pxgShflSyncI(uint mask, uint tid, int value, int srcLane, uint width)
{
    return int(pxgShflSyncU(mask, tid, uint(value), srcLane, width));
}

// ---- ballot / vote ----

uint pxgBallotSync(uint mask, uint tid, bool pred)
{
#if PXG_WARP_NATIVE
    return WaveActiveBallot(pred).x & mask;
#else
    gPxgWarpScratch[tid] = pred ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint base = tid & ~(PXG_WARP_SIZE - 1u);
    uint ballot = 0u;
    for (uint i = 0u; i < PXG_WARP_SIZE; ++i)
        ballot |= ((mask >> i) & 1u) != 0u && gPxgWarpScratch[base + i] != 0u ? (1u << i) : 0u;
    GroupMemoryBarrierWithGroupSync();
    return ballot;
#endif
}

bool pxgAnySync(uint mask, uint tid, bool pred) { return pxgBallotSync(mask, tid, pred) != 0u; }
bool pxgAllSync(uint mask, uint tid, bool pred) { return pxgBallotSync(mask, tid, pred) == mask; }

// reduction.cuh warpScanExclusive(mask, threadIndexInWarp): rank within mask.
uint pxgWarpRankInMask(uint mask, uint lane)
{
    return countbits(mask & ((1u << lane) - 1u));
}

// reduction.cuh warpCountAndBroadcast.
uint pxgWarpCountAndBroadcast(uint mask, uint tid, bool pred)
{
    return countbits(pxgBallotSync(mask, tid, pred));
}

// ---- butterfly reductions (reduction.cuh warpReduction<OP>) ----
// Result broadcast to every lane of each width-sized segment; the shuffle
// order (radius width/2 .. 1) matches the CUDA template exactly, so float
// addition associates identically.

float pxgWarpReduceAddF(uint mask, uint tid, float value, uint log2width)
{
    const uint width = 1u << log2width;
    for (uint radius = width >> 1; radius > 0u; radius >>= 1)
        value = value + pxgShflXorSyncF(mask, tid, value, radius, width);
    return value;
}

float pxgWarpReduceMinF(uint mask, uint tid, float value, uint log2width)
{
    const uint width = 1u << log2width;
    for (uint radius = width >> 1; radius > 0u; radius >>= 1)
        value = min(value, pxgShflXorSyncF(mask, tid, value, radius, width));
    return value;
}

float pxgWarpReduceMaxF(uint mask, uint tid, float value, uint log2width)
{
    const uint width = 1u << log2width;
    for (uint radius = width >> 1; radius > 0u; radius >>= 1)
        value = max(value, pxgShflXorSyncF(mask, tid, value, radius, width));
    return value;
}

uint pxgWarpReduceAddU(uint mask, uint tid, uint value, uint log2width)
{
    const uint width = 1u << log2width;
    for (uint radius = width >> 1; radius > 0u; radius >>= 1)
        value = value + pxgShflXorSyncU(mask, tid, value, radius, width);
    return value;
}

uint pxgWarpReduceMaxU(uint mask, uint tid, uint value, uint log2width)
{
    const uint width = 1u << log2width;
    for (uint radius = width >> 1; radius > 0u; radius >>= 1)
        value = max(value, pxgShflXorSyncU(mask, tid, value, radius, width));
    return value;
}

uint pxgWarpReduceOrU(uint mask, uint tid, uint value, uint log2width)
{
    const uint width = 1u << log2width;
    for (uint radius = width >> 1; radius > 0u; radius >>= 1)
        value = value | pxgShflXorSyncU(mask, tid, value, radius, width);
    return value;
}

uint pxgWarpReduceAndU(uint mask, uint tid, uint value, uint log2width)
{
    const uint width = 1u << log2width;
    for (uint radius = width >> 1; radius > 0u; radius >>= 1)
        value = value & pxgShflXorSyncU(mask, tid, value, radius, width);
    return value;
}

// warpReduction<OP>(..., winnerLaneIndex): broadcast best + lowest lane
// holding it (ballot of equality + ffs), exactly the CUDA composition.
float pxgWarpReduceMaxWinnerF(uint mask, uint tid, float value, uint log2width, out uint winnerLane)
{
    const float best = pxgWarpReduceMaxF(mask, tid, value, log2width);
    winnerLane = uint(pxgLowestSetIndex(pxgBallotSync(mask, tid, best == value)));
    return best;
}

// ---- scans ----

// reduction.cuh warpScan<AddOp> (inclusive, width-parameterized).
uint pxgWarpScanAddInclusiveU(uint mask, uint tid, uint value, uint log2width)
{
    const uint width = 1u << log2width;
    const uint laneInSeg = pxgLaneIdx(tid) & (width - 1u);
    for (uint radius = 1u; radius < width; radius <<= 1)
    {
        const uint n = pxgShflUpSyncU(mask, tid, value, radius, width);
        value = laneInSeg >= radius ? value + n : value;
    }
    return value;
}

// reduction.cuh warpScanExclusive (full warp, FULL_MASK).
uint pxgWarpScanAddExclusiveU(uint tid, uint value)
{
    const uint inclusive = pxgWarpScanAddInclusiveU(PXG_FULL_MASK, tid, value, 5u);
    return inclusive - value;
}

// ---- 64-bit bit conventions (__popcll / __ffsll / __clzll) ----

uint pxgPopcll(uint lo, uint hi) { return countbits(lo) + countbits(hi); }

// __ffsll: 1-based index of lowest set bit; 0 when the value is 0.
uint pxgFfsll(uint lo, uint hi)
{
    if (lo != 0u) return firstbitlow(lo) + 1u;
    if (hi != 0u) return firstbitlow(hi) + 33u;
    return 0u;
}

// __clzll: leading zeros; 64 when the value is 0.
uint pxgClzll(uint lo, uint hi)
{
    if (hi != 0u) return 31u - firstbithigh(hi);
    if (lo != 0u) return 63u - firstbithigh(lo);
    return 64u;
}

// ---- atomics (atomic.cuh parity) ----
// AtomicMin/AtomicMax on float are CAS loops in CUDA itself (no native
// float min/max atomic exists there either).  Buffers carry float BITS as
// uint so one buffer type serves compare-exchange.  AtomicAdd uses the same
// CAS shape: each contribution is applied exactly once; ordering across
// lanes is unspecified exactly as with CUDA's native float atomicAdd.
//
// P9.89: atomic.cuh:104-132 is a SELECT rule, not an fminf/fmaxf combine:
//   int old = *address_as_int, assumed;
//   while (val < __int_as_float(old))          // AtomicMax: val > old
//   { assumed = old; old = atomicCAS(address_as_int, assumed, __float_as_int(val)); }
// The cell is written only on a STRICT improvement, so a NaN already in the
// cell is never replaced, a NaN val is never written, and on a +-0 tie the
// RESIDENT zero stays.  Before P9.89 the port CASed FMin/FMax(value, old):
// on llvmpipe (lower_fminmax_signed_zero, gallivm fmin_nsz) that overwrote a
// NaN cell with the first finite value and turned a resident +0 into -0
// (Min) / a resident -0 into +0 (Max).  Latent: only npAtomicLib calls these;
// the CUDA's return value has no caller here either.

void pxgAtomicMinF(RWStructuredBuffer<uint> buf, uint index, float value)
{
    uint old = buf[index];
    while (value < asfloat(old))
    {
        const uint assumed = old;
        InterlockedCompareExchange(buf[index], assumed, asuint(value), old);
    }
}

void pxgAtomicMaxF(RWStructuredBuffer<uint> buf, uint index, float value)
{
    uint old = buf[index];
    while (value > asfloat(old))
    {
        const uint assumed = old;
        InterlockedCompareExchange(buf[index], assumed, asuint(value), old);
    }
}

float pxgAtomicAddF(RWStructuredBuffer<uint> buf, uint index, float value)
{
    uint assumed = buf[index];
    for (;;)
    {
        const uint newBits = asuint(asfloat(assumed) + value);
        uint old;
        InterlockedCompareExchange(buf[index], assumed, newBits, old);
        if (old == assumed)
            return asfloat(assumed);
        assumed = old;
    }
}

// atomic.cuh AtomicOr(PxU64): two independent 32-bit ORs on the halves.
void pxgAtomicOr64(RWStructuredBuffer<uint> buf, uint indexLo, uint maskLo, uint maskHi)
{
    uint prev;
    InterlockedOr(buf[indexLo], maskLo, prev);
    InterlockedOr(buf[indexLo + 1u], maskHi, prev);
}

#endif // PXG_SLANG_WARP_INC
