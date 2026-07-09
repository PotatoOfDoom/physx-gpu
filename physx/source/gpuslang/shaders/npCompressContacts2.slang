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

// npCompressContacts2.slang
//
// Slang port of compressContactStage2 from
// gpunarrowphase/src/CUDA/compressOutputContacts.cu — the cross-block compaction
// that gathers every pair with at least one contact into the dense
// PxGpuContactPair output list.  Stage1 (npCompressContacts.slang) produced the
// per-block survivor counts (gBlockNumPairs, NUM_BLOCKS entries); Stage2:
//   1. exclusive-prefix-sums those block counts to get each block's start offset
//      (and the grand total numPairs);
//   2. re-walks the same fixed BLOCK/NUM_BLOCKS work partition and, for each pair
//      with nbContacts != 0, assigns it a dense slot = blockStart + within-block
//      running count of prior survivors, then writes the PxGpuContactPair entry
//      (pointer->offset conversions, remap/actor-table lookups, patch/contact
//      counts).
//
// The CUDA warp-decomposes the within-block prefix (ballot/popc + per-warp
// accumulators + a block running total sAccum).  Here one 256-thread block maps
// to one CUDA block; the within-iteration prefix is a block-wide groupshared
// Hillis-Steele scan over the survivor flags — output-identical because it is the
// same stable compaction over the same work order — and the per-iteration block
// total is folded into a running sAccum kept identically in every thread's
// register (no groupshared needed for it).  Pointers are modelled as byte offsets
// into their streams (base pointers = 0): contactForces carries a presence word.

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint OUTPUT_WORDS = 12; // sizeof(PxsContactManagerOutput)/4

struct Compress2Params
{
    uint numTotalPairs;
    uint maxOutputPatches;
    uint contactSize;   // sizeof(PxContactPatch) = 64
    uint fricSize;      // sizeof(PxFrictionPatch) = 52
}

struct GpuPairOut
{
    uint ref0, ref1;
    uint node0, node1;
    uint forceOff;    // 0xFFFFFFFF when contactForces is null
    uint fricOff;
    uint patchOff;
    uint pointOff;
    uint nbPatches, nbContacts;
    uint actor0, actor1;
}

[[vk::push_constant]]
ConstantBuffer<Compress2Params> gParams;

RWStructuredBuffer<uint>       gBlockNumPairs; // binding 0  [NUM_BLOCKS]
RWStructuredBuffer<uint2>      gInputRefs;     // binding 1  (transformCacheRef0, ref1) per pair
RWStructuredBuffer<uint>       gOutputData;    // binding 2  PxsContactManagerOutput word view
RWStructuredBuffer<uint>       gRemap;         // binding 3  shapeToRigidRemapTable[ref] -> nodeIndex
RWStructuredBuffer<uint>       gActorTable;    // binding 4  transformCacheIdToActorTable[ref] -> actor id
RWStructuredBuffer<GpuPairOut> gOut;           // binding 5  compacted output
RWStructuredBuffer<uint>       gNumPairs;      // binding 6  [1] grand total

groupshared uint gsScan[256];

// ---------------------------------------------------------------------------
// compressContactStage2: cross-block PxGpuContactPair compaction
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void compressContactStage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.numTotalPairs;

    // 1. exclusive prefix of the per-block survivor counts (+ grand total)
    uint blockStart = 0u;
    uint grandTotal = 0u;
    for (uint b = 0u; b < NUM_BLOCKS; ++b)
    {
        const uint c = gBlockNumPairs[b];
        if (b < blockIdx) blockStart += c;
        grandTotal += c;
    }
    if (blockIdx == 0u && tid == 0u)
        gNumPairs[0] = grandTotal;

    const uint totalBlockRequired = (n + (BLOCK - 1u)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1u)) / NUM_BLOCKS;

    uint sAccum = 0u; // running survivor count over this block's prior iterations

    for (uint it = 0u; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;

        uint nbContacts = 0u;
        if (workIndex < n)
            nbContacts = gOutputData[workIndex * OUTPUT_WORDS + 9u] & 0xffffu;
        const uint survivor = (nbContacts != 0u) ? 1u : 0u;

        // block-wide inclusive Hillis-Steele scan of the survivor flags
        gsScan[tid] = survivor;
        GroupMemoryBarrierWithGroupSync();
        for (uint off = 1u; off < BLOCK; off <<= 1u)
        {
            const uint add = (tid >= off) ? gsScan[tid - off] : 0u;
            GroupMemoryBarrierWithGroupSync();
            gsScan[tid] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        const uint excl = gsScan[tid] - survivor; // exclusive prefix within iteration
        const uint iterTotal = gsScan[BLOCK - 1u];

        if (survivor != 0u)
        {
            const uint index = blockStart + sAccum + excl;
            if (index < gParams.maxOutputPatches)
            {
                const uint w0 = workIndex * OUTPUT_WORDS;
                const uint patchStartIndex = gOutputData[w0 + 0u]; // contactPatches offset
                const uint contactIndex = gOutputData[w0 + 2u];    // contactPoints offset
                const uint forceOff = gOutputData[w0 + 4u];        // contactForces offset (low word)
                const uint forcePres = gOutputData[w0 + 5u];       // presence (high word); 0 == null
                const uint word8 = gOutputData[w0 + 8u];
                const uint nbPatches = (word8 >> 8u) & 0xffu;

                const uint2 refs = gInputRefs[workIndex];

                GpuPairOut o;
                o.ref0 = refs.x;
                o.ref1 = refs.y;
                o.node0 = gRemap[refs.x];
                o.node1 = gRemap[refs.y];
                o.forceOff = (forcePres != 0u) ? forceOff : 0xFFFFFFFFu;
                o.fricOff = patchStartIndex / gParams.contactSize * gParams.fricSize;
                o.patchOff = patchStartIndex;
                o.pointOff = contactIndex;
                o.nbPatches = nbPatches;
                o.nbContacts = nbContacts;
                o.actor0 = gActorTable[refs.x];
                o.actor1 = gActorTable[refs.y];
                gOut[index] = o;
            }
        }

        sAccum += iterTotal;
        GroupMemoryBarrierWithGroupSync(); // gsScan reuse next iteration
    }
}
