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

// psVolumeBoundBda.slang -- P9.80.
//
//   ps_update_volume_bound (particlesystem.cu:683) on the engine launch ABI.
//
// Per PxParticleVolume, reduce the world AABB of the contiguous slice of
// particles it names.  Launched from PxgPBDParticleSystemCore::updateVolumeBound
// (PxgPBDParticleSystemCore.cpp:81) on the finalize stream, once per step.
//
// ---------------------------------------------------------------------------
// WHY THIS WAS NEVER LAUNCHED
// ---------------------------------------------------------------------------
// Its host guard is `if (mMaxVolumesPerBuffer > 0)`, and mMaxVolumesPerBuffer is
// the max over buffers of PxParticleBuffer::getNbParticleVolumes().  Eight
// particle scenes in this harness, and every one creates its buffer with
// maxVolumes = 0.  A kernel that is never LAUNCHED is never reported as
// [pxg-unmapped] either -- the P9.40 machinery only sees launches -- so this one
// was invisible to every instrument the project has.  PXG_SCENE=particlerigid
// (P9.80a) is the first scene that sets a volume, and its first run measured the
// bound still holding the -999 poison it was written with.
//
// ---------------------------------------------------------------------------
// THE ONE DEVIATION FROM CUDA, AND WHY IT IS BIT-EXACT ANYWAY
// ---------------------------------------------------------------------------
// The CUDA does a two-level reduction: a 32-lane __shfl_xor_sync butterfly per
// warp (warpReduction<MinOpFloat>), lane 31 of each warp stores into
// sMin[3][32] / sMax[3][32], then warps 0..2 each reduce ONE COMPONENT over
// those 32 entries.  The engine pins the block size for it:
//
//     UPDATEBOUND = 1024, //can't change this. updateBound kernel is relied on
//                         //numOfWarpPerBlock = 32
//     (PxgParticleSystemCoreKernelIndices.h:39)
//
// so numWarpsPerBlock is exactly 32 and the second-stage reduction reads exactly
// the 32 entries that were written -- no out-of-bounds tail.
//
// This port lowers both levels to ONE portable full-block groupshared tree, for
// the reason psBoundsBda.slang already documents for its sibling
// ps_updateBoundFirstPassLaunch: llvmpipe's subgroup width is NOT 32, so the
// CUDA butterfly cannot be assumed.  That substitution is safe HERE and would not
// be safe for a sum: min and max of a finite float set are associative and
// introduce no rounding, so the reduced AABB is BIT-IDENTICAL whatever the
// association order.  (ps_solveShapes in psShapeMatchBda.slang sums, and there
// the butterfly IS reproduced exactly, because float addition is not
// associative.)
//
// The CUDA's 2D block (32, 32) becomes a 1D block of 1024: the CUDA's
// warpIndex = threadIdx.y and threadIndexInWarp = threadIdx.x linearise to
// tid / 32 and tid % 32, and the work-index arithmetic
// `threadIndexInWarp + warpIndex * WARP_SIZE + i * (blockDim.x * blockDim.y)`
// is exactly `tid + i * 1024`.

import pxgSlangCommon;
import pxgSlangParticleDesc;

static const uint UPDATEBOUND = 1024u;   // PxgParticleSystemKernelBlockDim::UPDATEBOUND
static const float PX_MAX_F32 = 3.4028235e38f;

// PxParticleVolume (PxParticleBuffer.h:63), 16-byte aligned:
//   PxBounds3 bound        @0   (PxVec3 minimum @0, PxVec3 maximum @12)
//   PxU32 particleIndicesOffset @24
//   PxU32 numParticles          @28
//                               -> sizeof 32
// Mirrored as a raw float/uint view rather than a struct so the bound can be
// written component-wise exactly where the CUDA writes it.
static const uint PX_PARTICLE_VOLUME_FLOATS = 8u;   // 32 bytes / 4
static const uint PV_MIN_F = 0u;                    // minimum.x/y/z at float 0,1,2
static const uint PV_MAX_F = 3u;                    // maximum.x/y/z at float 3,4,5
static const uint PV_OFFSET_F = 6u;                 // particleIndicesOffset
static const uint PV_NUMPARTICLES_F = 7u;           // numParticles

static const uint SZ_PARTICLE_SIM_BUFFER = 88u;     // PxgParticleSimBufferGpu stride

// ENGINE launch pack (PxgPBDParticleSystemCore.cpp:92):
//   ps_update_volume_bound(const PxgParticleSystem*, const PxU32*)
// -> argSizes {8,8} = a 16-byte blob.
struct PsVolumeBoundParams
{
    uint64_t particleSystems;             // @0
    uint64_t activeParticleSystemIndices; // @8 -> 16
}

[[vk::push_constant]]
ConstantBuffer<PsVolumeBoundParams> gVB;

// Portable full-block min/max reduction.  Two arrays rather than one reused
// buffer: the CUDA reduces min and max in the same pass, and interleaving them
// through a single scratch array would need twice the barriers for no gain.
groupshared float sVbMin[UPDATEBOUND];
groupshared float sVbMax[UPDATEBOUND];

float vbBlockMin(uint tid, float v)
{
    sVbMin[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = UPDATEBOUND / 2u; s > 0u; s >>= 1)
    {
        if (tid < s)
            sVbMin[tid] = min(sVbMin[tid], sVbMin[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = sVbMin[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float vbBlockMax(uint tid, float v)
{
    sVbMax[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = UPDATEBOUND / 2u; s > 0u; s >>= 1)
    {
        if (tid < s)
            sVbMax[tid] = max(sVbMax[tid], sVbMax[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = sVbMax[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// mirrors ps_update_volume_bound (particlesystem.cu:683).
//
// Engine grid (PxgPBDParticleSystemCore.cpp:98):
//   ( mMaxVolumesPerBuffer, mMaxBuffersPerSystem, numActiveParticleSystems )
// with block (32, 32, 1).  Every one of the three guards below is a real bound,
// because each grid extent is a MAX over buffers or systems and the actual count
// for this block's buffer can be smaller.
//
// NOTE the reduction runs with the WHOLE BLOCK in uniform control flow: the
// CUDA's early-outs are `if (bufferIndex < ...)` / `if (blockIdx.x < ...)`
// wrappers around the entire body, i.e. every thread of the block takes the same
// branch, so no thread can be missing from a __syncthreads().  This port keeps
// that shape -- a `return` inside the reduction would hang it on any backend
// that requires full participation at a barrier.
[shader("compute")]
[numthreads(UPDATEBOUND, 1, 1)]
void ps_update_volume_bound(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gVB.particleSystems);
    const uint particleSystemId = (Ptr<uint>(gVB.activeParticleSystemIndices))[gid.z];

    const uint bufferIndex = gid.y;
    if (bufferIndex >= systems[particleSystemId].mCommonData_NumParticleBuffers)
        return;   // block-uniform: gid.y is the same for every thread

    const uint64_t simBuffers = systems[particleSystemId].mParticleSimBuffers;
    const uint64_t bufferAddr = simBuffers + uint64_t(bufferIndex) * uint64_t(SZ_PARTICLE_SIM_BUFFER);
    const PxgParticleSimBufferGpu buffer = (Ptr<PxgParticleSimBufferGpu>(bufferAddr))[0];

    const uint bufferOffset = (Ptr<uint>(systems[particleSystemId].mParticleBufferRunsum))[bufferIndex];

    if (gid.x >= buffer.mNumVolumes)
        return;   // block-uniform likewise

    Ptr<float> volume = Ptr<float>(buffer.mVolumes + uint64_t(gid.x) * 32u);
    const uint offset = asuint(volume[PV_OFFSET_F]) + bufferOffset;
    const uint numParticles = asuint(volume[PV_NUMPARTICLES_F]);

    Ptr<float4> posInvMass = Ptr<float4>(systems[particleSystemId].mUnsortedPositions_InvMass);

    // The CUDA seeds with +/-PX_MAX_F32 and loops
    //   numIterations = (numParticles + blockDim.x*blockDim.y - 1) / (blockDim.x*blockDim.y)
    // which for a 1024-thread block is exactly the grid-stride below.
    float3 minPos = float3(PX_MAX_F32, PX_MAX_F32, PX_MAX_F32);
    float3 maxPos = float3(-PX_MAX_F32, -PX_MAX_F32, -PX_MAX_F32);
    for (uint workIndex = tid; workIndex < numParticles; workIndex += UPDATEBOUND)
    {
        const float3 pos = posInvMass[offset + workIndex].xyz;
        minPos = min(minPos, pos);
        maxPos = max(maxPos, pos);
    }

    const float mnx = vbBlockMin(tid, minPos.x);
    const float mny = vbBlockMin(tid, minPos.y);
    const float mnz = vbBlockMin(tid, minPos.z);
    const float mxx = vbBlockMax(tid, maxPos.x);
    const float mxy = vbBlockMax(tid, maxPos.y);
    const float mxz = vbBlockMax(tid, maxPos.z);

    // The CUDA has warp 0 write minimum.x/maximum.x, warp 1 the y pair and warp 2
    // the z pair, each from its own lane 31.  One writer per component either way;
    // with a whole-block reduction the natural writer is thread 0.
    if (tid == 0u)
    {
        volume[PV_MIN_F + 0u] = mnx;
        volume[PV_MIN_F + 1u] = mny;
        volume[PV_MIN_F + 2u] = mnz;
        volume[PV_MAX_F + 0u] = mxx;
        volume[PV_MAX_F + 1u] = mxy;
        volume[PV_MAX_F + 2u] = mxz;
    }
}
