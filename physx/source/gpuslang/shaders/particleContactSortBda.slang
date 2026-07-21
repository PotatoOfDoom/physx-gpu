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

// particleContactSortBda.slang
//
// P6.1 — the three particle contact-sort orchestration kernels on the Vulkan
// engine ABI (BDA), ported line-for-line from the CUDA ground truth.  The push
// constant of each entry is the EXACT CUDA __global__ argument pack of device
// pointers (one uint64_t address per pointer), read through Ptr<> exactly as
// the CUDA reads them — the same recipe as radixSortDesc.slang.  These replace
// the Phase-1 resource-binding harness (radixSortGather.slang / clampValues.slang),
// which the engine-ABI dispatcher cannot drive.
//
//   - radixSortDoubleCopy (gpucommon/src/CUDA/radixSortImpl.cu:187): two
//     independent rank-indexed 64-bit gathers in one launch —
//       outValue0[w] = inValue0[rank0[w]];   (full 64-bit)
//       outValue1[w] = inValue1[rank1[w]];   (full 64-bit)
//     The launch site PxgParticleSystemCore.cpp:1268 reorders the contact keys
//     (rigid- and particle-indexed) after the 8-pass radix sort completes.
//   - radixSortDoubleCopyHigh32Bits (radixSortImpl.cu:139): the same twin gather
//     but writing only the HIGH word of each 64-bit key —
//       outValue0[w] = PxU32(inValue0[rank0[w]] >> 32);   (32-bit)
//       outValue1[w] = PxU32(inValue1[rank1[w]] >> 32);   (32-bit)
//     Launched at PxgParticleSystemCore.cpp:1207 to pull the high 32 bits (the
//     rigid/particle id) out of the 64-bit node index into a temp sort key.
//     Same 7-pointer pack as radixSortDoubleCopy; only the OUTPUT width differs
//     (uint2 -> uint2 vs uint2 -> uint).
//   - clampMaxValue (gpucommon/src/CUDA/MemCopyBalanced.cu:88): a single in-place
//     clamp of one device counter —  if (*value > maxValue) *value = maxValue;
//     Launched at PxgParticleSystemCore.cpp:1105 with grid (1,2,1) block (1,1,1)
//     to cap the particle contact count at mMaxContacts.
//
// 64-bit keys are stored as uint2 (x = low word, y = high word), matching the
// radixSortGather.slang harness convention; the high word is member .y, so
// PxU32(key >> 32) == key.y.
//
// Differences from CUDA (recipe policy):
//   - numKeys is dereferenced from its device pointer (const PxU32*) exactly as
//     the CUDA does (`*numKeys`) — it is NOT a host scalar.
//   - the CUDA over-counts its iteration bound (numIterations, from an
//     unparenthesised (n+bd*gd-1)/bd*gd) and bails per-thread with `return` on
//     the first out-of-range workIndex; the monotone grid-stride loop
//     `for (w = dtid.x; w < numElems; w += stride)` visits the identical set of
//     in-range indices, so the memory effect is bit-identical.
//   - clampMaxValue is launched twice (grid.y == 2); the clamp is idempotent, so
//     the two groups writing the same value is harmless — mirrored verbatim, no
//     InterlockedMin (matches the CUDA, which uses a plain compare-and-store).

static const uint RS_BLOCK_DIM = 1024u; // PxgRadixSortKernelBlockDim::RADIX_SORT

// ---------------------------------------------------------------------------
// radixSortDoubleCopy / radixSortDoubleCopyHigh32Bits share the CUDA pack:
//   (const PxU64* inValue0, PxU64/PxU32* outValue0, PxU32* rank0,
//    const PxU64* inValue1, PxU64/PxU32* outValue1, PxU32* rank1,
//    const PxU32* numKeys)  ->  7 device pointers, argSizes {8,8,8,8,8,8,8} (56B).
// Only the output element width differs between the two kernels (interpreted in
// the body), so a single push-constant layout serves both.
// ---------------------------------------------------------------------------
struct RadixDoubleCopyParams
{
    uint64_t inValue0;   // @0  const PxU64*
    uint64_t outValue0;  // @8  PxU64* (DoubleCopy) / PxU32* (High32Bits)
    uint64_t rank0;      // @16 PxU32*
    uint64_t inValue1;   // @24 const PxU64*
    uint64_t outValue1;  // @32 PxU64* (DoubleCopy) / PxU32* (High32Bits)
    uint64_t rank1;      // @40 PxU32*
    uint64_t numKeys;    // @48 const PxU32*  -> 56-byte blob
}

[[vk::push_constant]]
ConstantBuffer<RadixDoubleCopyParams> gDC;

// mirrors radixSortDoubleCopy (radixSortImpl.cu:187): two full 64-bit gathers.
[shader("compute")]
[numthreads(1024, 1, 1)]
void radixSortDoubleCopy(uint3 dtid : SV_DispatchThreadID)
{
    const uint numElems = (Ptr<uint>(gDC.numKeys))[0]; // *numKeys (device pointer)
    const uint stride = WorkgroupCount().x * RS_BLOCK_DIM;

    Ptr<uint2> inValue0 = Ptr<uint2>(gDC.inValue0);
    Ptr<uint2> outValue0 = Ptr<uint2>(gDC.outValue0);
    Ptr<uint>  rank0 = Ptr<uint>(gDC.rank0);
    Ptr<uint2> inValue1 = Ptr<uint2>(gDC.inValue1);
    Ptr<uint2> outValue1 = Ptr<uint2>(gDC.outValue1);
    Ptr<uint>  rank1 = Ptr<uint>(gDC.rank1);

    for (uint w = dtid.x; w < numElems; w += stride)
    {
        outValue0[w] = inValue0[rank0[w]];
        outValue1[w] = inValue1[rank1[w]];
    }
}

// mirrors radixSortDoubleCopyHigh32Bits (radixSortImpl.cu:139): two high-word
// (PxU32(key >> 32) == key.y) gathers.
[shader("compute")]
[numthreads(1024, 1, 1)]
void radixSortDoubleCopyHigh32Bits(uint3 dtid : SV_DispatchThreadID)
{
    const uint numElems = (Ptr<uint>(gDC.numKeys))[0]; // *numKeys (device pointer)
    const uint stride = WorkgroupCount().x * RS_BLOCK_DIM;

    Ptr<uint2> inValue0 = Ptr<uint2>(gDC.inValue0);
    Ptr<uint>  outValue0 = Ptr<uint>(gDC.outValue0);
    Ptr<uint>  rank0 = Ptr<uint>(gDC.rank0);
    Ptr<uint2> inValue1 = Ptr<uint2>(gDC.inValue1);
    Ptr<uint>  outValue1 = Ptr<uint>(gDC.outValue1);
    Ptr<uint>  rank1 = Ptr<uint>(gDC.rank1);

    for (uint w = dtid.x; w < numElems; w += stride)
    {
        outValue0[w] = inValue0[rank0[w]].y; // high 32 bits
        outValue1[w] = inValue1[rank1[w]].y;
    }
}

// ---------------------------------------------------------------------------
// clampMaxValue (MemCopyBalanced.cu:88): CUDA pack (PxU32* value, const PxU32
// maxValue) -> argSizes {8,4}, natural-aligned to a 16-byte blob.
// ---------------------------------------------------------------------------
struct ClampMaxValueParams
{
    uint64_t value;    // @0  PxU32*
    uint     maxValue; // @8  const PxU32 (by value)
    uint     _pad12;   // @12 -> 16-byte blob
}

[[vk::push_constant]]
ConstantBuffer<ClampMaxValueParams> gClamp;

// mirrors clampMaxValue(): single-thread in-place clamp of one device counter.
[shader("compute")]
[numthreads(1, 1, 1)]
void clampMaxValue()
{
    Ptr<uint> value = Ptr<uint>(gClamp.value);
    if (value[0] > gClamp.maxValue)
        value[0] = gClamp.maxValue;
}
