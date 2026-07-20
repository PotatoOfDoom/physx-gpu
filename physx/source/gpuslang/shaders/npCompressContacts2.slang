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
// P5 Step 4 (BDA, engine launch path): compressContactStage2
// (gpunarrowphase/src/CUDA/compressOutputContacts.cu L128-276) on the exact
// engine ABI — 16 arguments,
//   (const PxgContactManagerInput* inputData, const PxsContactManagerOutput*
//    outputData, PxU32 numTotalPairs, PxNodeIndex* shapeToRigidRemapTable,
//    PxActor** transformCacheIdToActorTable, PxU32* gBlockNumPairs,
//    PxU8* cpuCompressedPatchesBase, PxU8* cpuCompressedContactsBase,
//    PxReal* cpuForceBufferBase, PxU8* gpuCompressedPatchesBase,
//    PxU8* gpuCompressedContactsBase, PxReal* gpuForceBufferBase,
//    PxU8* gpuFrictionPatchesBase, PxU32* numPairs,
//    PxGpuContactPair* outputPatchs, PxU32 maxOutputPatches)
//   argSizes {8,8,4,8,8,8,8,8,8,8,8,8,8,8,8,4} -> 128-byte blob.
// Engine grid (PxgNarrowphaseCore.cpp ~L9025): 32 blocks x 256 threads.
//
// The compaction structure is the verified bound-buffer port VERBATIM (the
// CUDA's warp ballot/scan decomposition kept as the block-wide groupshared
// Hillis-Steele scan over the same stable work order — output-identical),
// with data access converted to the real structs and real pointer
// arithmetic:
//   - survivor test reads PxsContactManagerOutput.nbContacts via Ptr<>;
//   - the CPU-stream pointers stored in cmOutput are turned into offsets by
//     REAL 64-bit pointer subtraction against the cpu*Base args (truncated
//     to PxU32 exactly as the CUDA's u32 assignment does; the force index
//     is a PxReal* difference, i.e. byte diff / 4);
//   - the PxGpuContactPair entry gets REAL rebased device pointers
//     (gpu*Base + offset in u64; a null-force pair keeps the CUDA's
//     base + 0xFFFFFFFF * sizeof(PxReal) arithmetic), PxNodeIndex/PxActor*
//     as u64 table loads, and the nbContacts|nbPatches u16 pair as one
//     32-bit word;
//   - every block stores the same grand total to *numPairs in the CUDA
//     (warp-0 scan per block); the port's single block-0 store is kept
//     (identical memory result).

import pxgSlangNpAbi;

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;

// Engine ABI blob (natural alignment): 128 bytes.
struct Compress2Params
{
    uint64_t inputData;                    // @0   const PxgContactManagerInput*
    uint64_t outputData;                   // @8   const PxsContactManagerOutput*
    uint numTotalPairs;                    // @16
    uint64_t shapeToRigidRemapTable;       // @24  PxNodeIndex*
    uint64_t transformCacheIdToActorTable; // @32  PxActor**
    uint64_t gBlockNumPairs;               // @40  PxU32*
    uint64_t cpuCompressedPatchesBase;     // @48  PxU8*
    uint64_t cpuCompressedContactsBase;    // @56  PxU8*
    uint64_t cpuForceBufferBase;           // @64  PxReal*
    uint64_t gpuCompressedPatchesBase;     // @72  PxU8*
    uint64_t gpuCompressedContactsBase;    // @80  PxU8*
    uint64_t gpuForceBufferBase;           // @88  PxReal*
    uint64_t gpuFrictionPatchesBase;       // @96  PxU8*
    uint64_t numPairs;                     // @104 PxU32*
    uint64_t outputPatchs;                 // @112 PxGpuContactPair*
    uint maxOutputPatches;                 // @120
}

[[vk::push_constant]]
ConstantBuffer<Compress2Params> gC2;

groupshared uint gsScan[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void compressContactStage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gC2.numTotalPairs;

    Ptr<PxgContactManagerInputGpu> inputData = Ptr<PxgContactManagerInputGpu>(gC2.inputData);
    Ptr<PxsContactManagerOutputGpu> outputData = Ptr<PxsContactManagerOutputGpu>(gC2.outputData);
    Ptr<uint> gBlockNumPairs = Ptr<uint>(gC2.gBlockNumPairs);

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
        (Ptr<uint>(gC2.numPairs))[0] = grandTotal;

    const uint totalBlockRequired = (n + (BLOCK - 1u)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1u)) / NUM_BLOCKS;

    uint sAccum = 0u; // running survivor count over this block's prior iterations

    for (uint it = 0u; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;

        uint nbContacts = 0u;
        if (workIndex < n)
            nbContacts = outputData[workIndex].nbContactsFlags & 0xffffu;
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
            if (index < gC2.maxOutputPatches)
            {
                const PxgContactManagerInputGpu cmInput = inputData[workIndex];
                const PxsContactManagerOutputGpu cmOutput = outputData[workIndex];

                // real pointer -> offset conversions (u32 truncation as CUDA)
                const uint patchStartIndex = uint(cmOutput.contactPatches - gC2.cpuCompressedPatchesBase);
                const uint contactIndex = uint(cmOutput.contactPoints - gC2.cpuCompressedContactsBase);
                uint forceIndex = 0xFFFFFFFFu;
                if (cmOutput.contactForces != 0)
                    forceIndex = uint((cmOutput.contactForces - gC2.cpuForceBufferBase) >> 2u); // PxReal* diff

                Ptr<PxGpuContactPairGpu> contact =
                    Ptr<PxGpuContactPairGpu>(gC2.outputPatchs + uint64_t(index) * 80u);

                const uint transformCacheRef0 = cmInput.transformCacheRef0;
                const uint transformCacheRef1 = cmInput.transformCacheRef1;
                contact[0].transformCacheRef0 = transformCacheRef0;
                contact[0].transformCacheRef1 = transformCacheRef1;
                contact[0].nodeIndex0 = (Ptr<uint64_t>(gC2.shapeToRigidRemapTable))[transformCacheRef0];
                contact[0].nodeIndex1 = (Ptr<uint64_t>(gC2.shapeToRigidRemapTable))[transformCacheRef1];
                // PxReal* arithmetic: base + forceIndex elements (u32 zero-extends;
                // the CUDA null-force case is base + 0xFFFFFFFF * 4, kept exactly)
                contact[0].contactForces = gC2.gpuForceBufferBase + uint64_t(forceIndex) * 4u;
                contact[0].frictionPatches = gC2.gpuFrictionPatchesBase
                    + (uint64_t(patchStartIndex) / 64u) * 52u; // sizeof(PxContactPatch)=64, sizeof(PxFrictionPatch)=52
                contact[0].contactPatches = gC2.gpuCompressedPatchesBase + uint64_t(patchStartIndex);
                contact[0].contactPoints = gC2.gpuCompressedContactsBase + uint64_t(contactIndex);
                // nbContacts (u16) | nbPatches (u16, from the u8 field) as one word
                contact[0].nbContactsPatches = (cmOutput.nbContactsFlags & 0xffffu)
                    | (((cmOutput.allflagsWord >> 8u) & 0xffu) << 16u);
                contact[0].actor0 = (Ptr<uint64_t>(gC2.transformCacheIdToActorTable))[transformCacheRef0];
                contact[0].actor1 = (Ptr<uint64_t>(gC2.transformCacheIdToActorTable))[transformCacheRef1];
            }
        }

        sAccum += iterTotal;
        GroupMemoryBarrierWithGroupSync(); // gsScan reuse next iteration
    }
}
