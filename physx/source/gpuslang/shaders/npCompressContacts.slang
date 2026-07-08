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

// npCompressContacts.slang
//
// Slang port of two of the contact-compression kernels from
// gpunarrowphase/src/CUDA/compressOutputContacts.cu:
//   - compressContactStage1: per-block count of the pairs that produced at least
//     one contact (nbContacts != 0), written to gBlockNumPairs — the per-block
//     histogram the cross-block compaction (Stage2, a later tranche) consumes.
//   - updateFrictionPatches: rescales each pair's contact-patch byte offset into
//     the friction-patch stream: frictionPatches = startFriction +
//     (contactPatches - startContact) * sizeof(PxFrictionPatch) /
//     sizeof(PxContactPatch).
//
// PxsContactManagerOutput is 48 bytes = 12 uint words; its device pointers are
// modelled as byte offsets: word 0 = contactPatches, word 6 = frictionPatches,
// word 9 low-16 = nbContacts.  The per-block count is a deterministic function of
// the (fixed) block/iteration work partition, matching the CUDA's warp-ballot
// reduction (BLOCK = 256, NUM_BLOCKS = 32).

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint OUTPUT_WORDS = 12; // sizeof(PxsContactManagerOutput)/4

struct CompressParams
{
    uint numTotalPairs;  // stage1
    uint pairCount;      // updateFrictionPatches
    uint startContact;   // startContactPatches byte offset
    uint startFriction;  // startFrictionPatches byte offset
    uint contactSize;    // sizeof(PxContactPatch)
    uint fricSize;       // sizeof(PxFrictionPatch)
}

[[vk::push_constant]]
ConstantBuffer<CompressParams> gParams;

RWStructuredBuffer<uint> gOutputData;    // binding 0  PxsContactManagerOutput (word view)
RWStructuredBuffer<uint> gBlockNumPairs; // binding 1  [NUM_BLOCKS]

groupshared uint gsBlockCount;

// ---------------------------------------------------------------------------
// compressContactStage1: per-block count of pairs with contacts
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void compressContactStage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.numTotalPairs;

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockCount = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < n)
        {
            const uint nbContacts = gOutputData[workIndex * OUTPUT_WORDS + 9u] & 0xffffu;
            if (nbContacts != 0u)
            {
                uint orig;
                InterlockedAdd(gsBlockCount, 1u, orig);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
        gBlockNumPairs[blockIdx] = gsBlockCount;
}

// ---------------------------------------------------------------------------
// updateFrictionPatches: rescale the contact-patch offset into the friction stream
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void updateFrictionPatches(uint3 dtid : SV_DispatchThreadID)
{
    const uint threadIndex = dtid.x;
    if (threadIndex >= gParams.pairCount)
        return;

    const uint contactPatches = gOutputData[threadIndex * OUTPUT_WORDS + 0u];
    const uint diff = contactPatches - gParams.startContact;
    const uint frictionPatches = gParams.startFriction + diff * gParams.fricSize / gParams.contactSize;
    gOutputData[threadIndex * OUTPUT_WORDS + 6u] = frictionPatches;
}
