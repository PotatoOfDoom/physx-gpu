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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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

// initManifoldsBda.slang
//
// `initializeManifolds` (pairManagement.cu:150) on the Vulkan engine ABI.
//
// P8.21's inventory left exactly two refused launches in every scene once P8.20
// landed: `initializeManifolds` (once per run, at setup) and
// `removeContactManagers_Stage5` (a few times, at teardown).  Both are the P8.20
// pattern — in the manifest, valid SPIR-V, but pointing at a Phase-1 harness
// shader with bound buffers that the engine-ABI dispatcher cannot drive, so no
// ABI row is possible and the shim refuses.  This closes the first of the two.
//
// WHAT IT DOES.  Replicates one `dataSize`-byte record `nbTimesToReplicate`
// times.  The engine uses it to stamp an empty manifold across a freshly grown
// manifold array (`PxgNarrowphaseCore.cpp:8187`).
//
// THE SHARED-MEMORY TRANSPOSE IS THE WHOLE TRICK, and it is easy to "simplify"
// away.  The source is NOT copied verbatim.  Lanes 0..nbThreadsPerElement-1 each
// read one source float4 and SCATTER its four components into four separate
// runs of a groupshared float array — all the x's, then all the y's, then the
// z's, then the w's:
//
//     sourceData[t]                          = value.x
//     sourceData[t +   nbThreadsPerElement]  = value.y
//     sourceData[t + 2*nbThreadsPerElement]  = value.z
//     sourceData[t + 3*nbThreadsPerElement]  = value.w
//
// and every destination lane then GATHERS its element back out with
// `threadReadIdx = threadIdx.x % nbThreadsPerElement`.  A port that just did
// `dst[i] = src[i % n]` produces the same bytes ONLY because the scatter and the
// gather are inverse; break either half and the record is transposed, not
// corrupted-looking.  Transcribed literally so the pairing stays visible.
//
// GRID.  512 blocks x 128 threads (PxgNarrowPhaseGridDims / BlockDims
// ::INITIALIZE_MANIFOLDS).  The block-stride loop `for (a = blockIdx.x;
// a < nbBlocksRequired; a += gridDim.x)` is mirrored on WorkgroupCount().
//
// `MaxStructureSize` is 4096 BYTES, so the groupshared array is 1024 floats —
// and the scatter writes up to `4*nbThreadsPerElement = dataSize/4` of them, so
// the bound holds exactly when dataSize <= 4096.  Upstream does not check it;
// neither does this.

static const uint MAX_STRUCTURE_FLOATS = 1024u;   // MaxStructureSize / sizeof(float)

struct InitManifoldsParams
{
    uint64_t destination;       // @0  float4*
    uint64_t source;            // @8  const float4*
    uint     dataSize;          // @16 bytes per record
    uint     nbTimesToReplicate;// @20 -> 24-byte blob
}

[[vk::push_constant]]
ConstantBuffer<InitManifoldsParams> gIm;

groupshared float sSourceData[MAX_STRUCTURE_FLOATS];

[shader("compute")]
[numthreads(128, 1, 1)]
void initializeManifolds(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;

    const uint nbThreadsPerElement = gIm.dataSize / 16u;   // dataSize / sizeof(float4)
    const uint nbPerBlock = 128u / nbThreadsPerElement;
    const uint maxThreadIdx = nbPerBlock * nbThreadsPerElement;
    const uint nbBlocksRequired = (gIm.nbTimesToReplicate + nbPerBlock - 1u) / nbPerBlock;

    if (tid < nbThreadsPerElement)
    {
        const float4 value = Ptr<float4>(gIm.source)[tid];
        sSourceData[tid] = value.x;
        sSourceData[tid + nbThreadsPerElement] = value.y;
        sSourceData[tid + 2u * nbThreadsPerElement] = value.z;
        sSourceData[tid + 3u * nbThreadsPerElement] = value.w;
    }

    // Computed BEFORE the barrier, exactly as the CUDA does.  Its value does not
    // depend on the shared store, so the placement is not load-bearing — but the
    // barrier below is, and it must be reached by every lane in the group.
    const uint threadReadIdx = tid % nbThreadsPerElement;

    GroupMemoryBarrierWithGroupSync();

    for (uint a = gid.x; a < nbBlocksRequired; a += WorkgroupCount().x)
    {
        const uint startBlockIdx = nbPerBlock * a;
        const uint64_t startPtr = gIm.destination +
                                  uint64_t(startBlockIdx * nbThreadsPerElement) * 16UL;
        const uint idx = startBlockIdx + tid / nbThreadsPerElement;

        if (tid < maxThreadIdx && idx < gIm.nbTimesToReplicate)
        {
            Ptr<float4>(startPtr)[tid] =
                float4(sSourceData[threadReadIdx],
                       sSourceData[threadReadIdx + nbThreadsPerElement],
                       sSourceData[threadReadIdx + 2u * nbThreadsPerElement],
                       sSourceData[threadReadIdx + 3u * nbThreadsPerElement]);
        }
    }
}
