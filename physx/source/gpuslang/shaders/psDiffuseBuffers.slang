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

// psDiffuseBuffers.slang -- P9.13c, the three per-BUFFER bookkeeping kernels of
// the diffuse chain.  Ported from
// gpusimulationcontroller/src/CUDA/diffuseParticles.cu, read in full:
//
//   ps_updateUnsortedDiffuseArrayLaunch  (:63)  gather each user buffer's
//       diffuse particles into the system's flat unsorted arrays, at that
//       buffer's running offset
//   ps_diffuseParticleCopy               (:503) publish each buffer's staged
//       count and clear the staging slot
//   ps_diffuseParticleSum                (:528) sum the per-buffer counts into
//       the system's single count
//
// They belong together because they are the only three kernels in the chain
// that walk `mDiffuseSimBuffers`, and because between them they define what the
// per-buffer count actually MEANS -- which the rest of the chain then relies on.
//
// ---------------------------------------------------------------------------
// THE TWO-SLOT COUNT, which is the thing to understand before anything else
// ---------------------------------------------------------------------------
// `PxgParticleDiffuseSimBuffer::mNumDiffuseParticles` points at TWO ints:
//
//     [0]  the PUBLISHED count -- what every consumer reads this frame
//     [1]  the STAGING count -- what ps_diffuseParticleCreate atomically
//          appends into while it is spawning
//
// ps_diffuseParticleCopy is the frame boundary between them: it clamps [1] to
// the buffer's capacity, writes that into [0] AND into the pinned host mirror,
// and then zeroes [1] for the next frame.  Nothing else moves a value between
// the two slots.  A port that treats the pointer as a single int compiles, runs,
// and quietly makes the count monotonic.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// U1 -- THE GRID AXES ARE NOT THE SAME IN ALL THREE.  Copy and UpdateUnsorted
//     take the system from blockIdx.**z** and the buffer from blockIdx.y; Sum
//     takes the system from blockIdx.**x** and has no buffer axis at all.  A
//     port that picks one convention is right for two of the three.
//
// U2 -- `bufferIndex < mNumDiffuseBuffers` is a real guard, not a formality:
//     the grid's y extent is mMaxDiffuseBuffersPerSystem, the MAXIMUM over all
//     systems, so a system with fewer buffers gets blocks it must ignore.
//
// U3 -- the running offset in UpdateUnsorted is a WARP-COOPERATIVE prefix sum
//     over the PRECEDING buffers, `for (i = lane; i < bufferIndex; i += 32)`,
//     reduced across the warp.  Every lane needs the full sum, so the reduction
//     is a broadcast reduction and not a scan.  With bufferIndex < 32 (the only
//     case the engine produces today) exactly one lane iterates per preceding
//     buffer, but the strided loop is transcribed as written.
//
// U4 -- `buffer.mStartIndex = bufferOffset` is written by globalThreadIndex 0
//     only, and it is written AFTER the `globalThreadIndex >= numDiffuseParticles`
//     early return.  So a buffer with ZERO diffuse particles never gets its
//     start index written at all and keeps last frame's value.  That is the
//     CUDA's behaviour, transcribed, not tidied -- see the note in the entry
//     point.
//
// U5 -- Copy clamps with `PxMin(PxI32(mMaxNumParticles), numDiffuseParticles[1])`
//     -- a SIGNED comparison, with the unsigned capacity cast to int.  The
//     staging counter is an int and an over-subscribed frame leaves it larger
//     than the capacity; the signed min is what bounds it.  Done unsigned, a
//     staging value with the top bit set would win the min.
//
// U6 -- Sum's loop is `for (i = threadIdx.x; i < numDiffuseBuffers; i += 32)`
//     over a block of exactly 32 threads, and only thread 0 stores.  It reads
//     slot [0] -- the count Copy just published -- not slot [1].
//
// U7 -- UpdateUnsorted writes into the system's `mDiffusePosition_LifeTime` and
//     `mDiffuseVelocity`, the UNSORTED arrays.  The sorted twins
//     (`mDiffuseSortedPos_LifeTime`, `mDiffuseSortedVel`) are a different pair
//     of pointers that the grid reorder fills later, and the two names differ
//     by one word.

import pxgSlangCommon;
import pxgSlangParticleDesc;

// PXG_WARP_BLOCK sizes the emulation's groupshared scratch (gPxgWarpScratch,
// pxgSlangWarpInc.slang:117) and must therefore be the LARGEST workgroup in this
// translation unit -- 1024, from ps_updateUnsortedDiffuseArrayLaunch.  The
// 32-thread Sum kernel then leaves the upper 992 slots untouched, which costs
// nothing; sizing it at 32 instead would let the 1024-thread kernel index past
// the end of the array.  It is a #define and not an import for the reason
// P9.12e recorded: the value cannot cross an `import`.
#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

// PxgParticleSystemKernelBlockDim: UPDATEBOUND = 1024, and the Sum kernel is
// launched with a block of exactly 32.
static const uint PSD_UPDATEBOUND = 1024u;
static const uint PSD_WARP = 32u;

// All three take (particleSystems, activeParticleSystems) and the last two take
// a PxU32 `count` that neither of them reads -- see D-count below.
struct PsDiffBufPush
{
    uint64_t particleSystems;        // @0
    uint64_t activeParticleSystems;  // @8
    uint count;                      // @16 -> 24 (Copy/Sum only; DEAD in both)
    uint pad0;
}
[[vk::push_constant]]
ConstantBuffer<PsDiffBufPush> gDb;

// ---------------------------------------------------------------------------
// ps_updateUnsortedDiffuseArrayLaunch (diffuseParticles.cu:63)
// ---------------------------------------------------------------------------
// Grid (ceil(maxDiffusePerBuffer/1024), maxDiffuseBuffersPerSystem,
// nbActiveSystems), block (1024,1,1) -- PxgParticleSystemCore.cpp:798.
[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_updateUnsortedDiffuseArrayLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gDb.particleSystems);
    const uint particleId = Ptr<uint>(gDb.activeParticleSystems)[gid.z];   // U1: .z

    const uint bufferIndex = gid.y;
    // U2: the y extent is the maximum over systems, so this can exceed this
    // system's own buffer count.  Block-uniform, so the warp primitives below
    // are still reached by all 32 lanes of every live block or by none.
    if (bufferIndex >= systems[particleId].mNumDiffuseBuffers)
        return;

    const uint tid = gtid.x;
    const uint lane = tid & (PSD_WARP - 1u);

    const uint64_t bufsAddr = systems[particleId].mDiffuseSimBuffers;
    Ptr<PxgParticleDiffuseSimBufferGpu> bufs = Ptr<PxgParticleDiffuseSimBufferGpu>(bufsAddr);

    // U3: the prefix sum over the PRECEDING buffers, strided by the warp.
    uint localSum = 0u;
    for (uint i = lane; i < bufferIndex; i += PSD_WARP)
        localSum += Ptr<uint>(bufs[i].mNumDiffuseParticles)[0];

    const uint bufferOffset = pxgWarpReduceAddU(PXG_FULL_MASK, tid, localSum, 5u);

    const uint numDiffuseParticles = Ptr<uint>(bufs[bufferIndex].mNumDiffuseParticles)[0];

    Ptr<float4> srcPos = Ptr<float4>(bufs[bufferIndex].mDiffusePositions_LifeTime);
    Ptr<float4> srcVel = Ptr<float4>(bufs[bufferIndex].mDiffuseVelocities);

    Ptr<float4> dstPos = Ptr<float4>(systems[particleId].mDiffusePosition_LifeTime);  // U7
    Ptr<float4> dstVel = Ptr<float4>(systems[particleId].mDiffuseVelocity);

    const uint globalThreadIndex = tid + PSD_UPDATEBOUND * gid.x;
    if (globalThreadIndex >= numDiffuseParticles)
        return;

    // U4: after the early return, so a buffer with no diffuse particles keeps
    // whatever start index the previous frame left.  The CUDA does this and the
    // consumers tolerate it because they bound their reads by the count, which
    // is zero in exactly that case.
    if (globalThreadIndex == 0u)
        bufs[bufferIndex].mStartIndex = bufferOffset;

    const uint ind = bufferOffset + globalThreadIndex;
    dstPos[ind] = srcPos[globalThreadIndex];
    dstVel[ind] = srcVel[globalThreadIndex];
}

// ---------------------------------------------------------------------------
// ps_diffuseParticleCopy (diffuseParticles.cu:503)
// ---------------------------------------------------------------------------
// Grid (1, maxDiffuseBuffersPerSystem, nbActiveSystems), block (1,1,1) --
// PxgPBDParticleSystemCore.cpp:1296.  ONE thread per (buffer, system).
[shader("compute")]
[numthreads(1, 1, 1)]
void ps_diffuseParticleCopy(uint3 gid : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gDb.particleSystems);
    const uint particleId = Ptr<uint>(gDb.activeParticleSystems)[gid.z];   // U1: .z

    const uint bufferIndex = gid.y;
    if (bufferIndex >= systems[particleId].mNumDiffuseBuffers)             // U2
        return;

    Ptr<PxgParticleDiffuseSimBufferGpu> bufs =
        Ptr<PxgParticleDiffuseSimBufferGpu>(systems[particleId].mDiffuseSimBuffers);

    Ptr<int> numDiffuseParticles = Ptr<int>(bufs[bufferIndex].mNumDiffuseParticles);

    // U5: PxMin(PxI32(capacity), staging) -- SIGNED.
    const int cap = int(bufs[bufferIndex].mMaxNumParticles);
    const int staged = numDiffuseParticles[1];
    const int numDiffuse = (cap < staged) ? cap : staged;

    Ptr<int>(bufs[bufferIndex].mNumActiveDiffuseParticles)[0] = numDiffuse;  // pinned host
    numDiffuseParticles[0] = numDiffuse;
    numDiffuseParticles[1] = 0;
}

// ---------------------------------------------------------------------------
// ps_diffuseParticleSum (diffuseParticles.cu:528)
// ---------------------------------------------------------------------------
// Grid (nbActiveSystems, 1, 1), block (32,1,1) --
// PxgPBDParticleSystemCore.cpp:1316.  ONE warp per system.
[shader("compute")]
[numthreads(32, 1, 1)]
void ps_diffuseParticleSum(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gDb.particleSystems);
    const uint particleId = Ptr<uint>(gDb.activeParticleSystems)[gid.x];   // U1: .x

    const uint tid = gtid.x;
    const uint numDiffuseBuffers = systems[particleId].mNumDiffuseBuffers;

    Ptr<PxgParticleDiffuseSimBufferGpu> bufs =
        Ptr<PxgParticleDiffuseSimBufferGpu>(systems[particleId].mDiffuseSimBuffers);

    // U6: strided over a 32-thread block, reading the PUBLISHED slot [0].
    uint totalDiffuse = 0u;
    for (uint i = tid; i < numDiffuseBuffers; i += PSD_WARP)
        totalDiffuse += Ptr<uint>(bufs[i].mNumDiffuseParticles)[0];

    totalDiffuse = pxgWarpReduceAddU(PXG_FULL_MASK, tid, totalDiffuse, 5u);

    if (tid == 0u)
        Ptr<uint>(systems[particleId].mNumDiffuseParticles)[0] = totalDiffuse;
}
