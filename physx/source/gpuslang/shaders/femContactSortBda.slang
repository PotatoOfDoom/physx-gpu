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

// femContactSortBda.slang
//
// P8.20 — the three orchestration kernels PxgSoftBodyCore::sortContacts() and
// PxgFEMClothCore::sortContacts() launch, on the Vulkan engine ABI (BDA).
//
// WHY THIS FILE EXISTS, WHEN THE THREE KERNELS WERE ALREADY "MAPPED".  All
// three appear in the manifest and all three have valid SPIR-V — but pointing
// at the PHASE-1 RESOURCE-BINDING HARNESS (clampValues.slang, radixSort64.slang,
// radixSortGather.slang).  Those take bound RWStructuredBuffers plus indices in
// the push constant; the engine hands them the CUDA argument pack of bare
// device pointers.  There is no ABI row that can describe the harness form, so
// pushConstantSizeFor() returned 0 and the shim REFUSED every launch, silently:
// PxgSoftBodyCore only fprintf's "fail to launch kernel!!" and steps on.
//
// The consequence is not cosmetic.  radixSortCopyHigh32Bits builds the sort KEY
// for the softbody-vs-rigid contact sort and radixSortCopy applies the finished
// rank permutation; with both refused, the 8-pass radix sort in between sorts a
// stale key buffer, and reorderRigidContacts() gathers through a rank array that
// was never written.  PXG_SCENE=softbody catches the cube at the right height
// and then RINGS forever (|dy| 0.0154 on box, 0.0130 on plane, 0.0141 on sphere,
// 0.0117 on mesh, against a 2e-3 / 1e-4 settle bar).  Same three kernels, same
// silent refusal, on the cloth side at PxgFEMClothCore.cpp:826/935/991/1019/1078.
//
// Ported line-for-line from the CUDA ground truth:
//   - clampMaxValues            (gpucommon/src/CUDA/MemCopyBalanced.cu:95)
//   - radixSortCopyHigh32Bits   (gpucommon/src/CUDA/radixSortImpl.cu:119)
//   - radixSortCopy             (gpucommon/src/CUDA/radixSortImpl.cu:163)
//
// 64-bit keys are stored as uint2 (x = low word, y = high word), matching
// radixSortGather.slang and particleContactSortBda.slang; the high word is
// member .y, so PxU32(key >> 32) == key.y.
//
// Differences from CUDA (recipe policy), both inherited verbatim from the P6.1
// twin-gather port in particleContactSortBda.slang:
//   - numKeys is a const PxU32* DEVICE POINTER, dereferenced on-device exactly
//     as the CUDA does (`const PxU32 numElems = *numKeys;`), NOT a host scalar.
//   - the CUDA over-counts its iteration bound (`numIternations`, from an
//     unparenthesised (n + bd*gd - 1) / bd * gd) and bails per-thread with
//     `return` on the first out-of-range workIndex; the monotone grid-stride
//     loop `for (w = dtid.x; w < numElems; w += stride)` visits the identical
//     set of in-range indices, so the memory effect is bit-identical.
//
// clampMaxValues is a SEPARATE ENTRY from clampMaxValue (the singular, already
// BDA in particleContactSortBda.slang): different CUDA kernel, three counters
// instead of one, and a different argument pack ({8,8,8,4} vs {8,4}).

static const uint RS_BLOCK_DIM = 1024u; // PxgRadixSortKernelBlockDim::RADIX_SORT

// ---------------------------------------------------------------------------
// radixSortCopyHigh32Bits / radixSortCopy share the CUDA pack:
//   (const PxU64* inValue, PxU32*/PxU64* outValue, PxU32* rank,
//    const PxU32* numKeys)  ->  4 device pointers, argSizes {8,8,8,8} (32B).
// Only the output element WIDTH differs between the two (interpreted in the
// body), so a single push-constant layout serves both — exactly as the twin
// gathers share RadixDoubleCopyParams.
// ---------------------------------------------------------------------------
struct RadixCopyParams
{
    uint64_t inValue;  // @0  const PxU64*
    uint64_t outValue; // @8  PxU32* (High32Bits) / PxU64* (Copy)
    uint64_t rank;     // @16 PxU32*
    uint64_t numKeys;  // @24 const PxU32*  -> 32-byte blob
}

[[vk::push_constant]]
ConstantBuffer<RadixCopyParams> gRC;

// mirrors radixSortCopyHigh32Bits (radixSortImpl.cu:119): a rank-indexed gather
// that keeps only the HIGH word of each 64-bit key.
//
//     const PxU32 index = rank[workIndex];
//     outValue[workIndex] = PxU32(inValue[index] >> 32);
//
// Launched grid (32,1,1) block (1024,1,1) at PxgSoftBodyCore.cpp:559 and :645
// and PxgFEMClothCore.cpp:935 and :1019, pulling the rigid/particle id out of
// the 64-bit PxNodeIndex into the temp radix key.
[shader("compute")]
[numthreads(1024, 1, 1)]
void radixSortCopyHigh32Bits(uint3 dtid : SV_DispatchThreadID)
{
    const uint numElems = (Ptr<uint>(gRC.numKeys))[0]; // *numKeys (device pointer)
    const uint stride = WorkgroupCount().x * RS_BLOCK_DIM;

    Ptr<uint2> inValue = Ptr<uint2>(gRC.inValue);
    Ptr<uint>  outValue = Ptr<uint>(gRC.outValue);
    Ptr<uint>  rank = Ptr<uint>(gRC.rank);

    for (uint w = dtid.x; w < numElems; w += stride)
        outValue[w] = inValue[rank[w]].y; // high 32 bits
}

// mirrors radixSortCopy (radixSortImpl.cu:163): the same rank-indexed gather,
// writing the FULL 64-bit key.
//
//     const PxU32 index = rank[workIndex];
//     outValue[workIndex] = inValue[index];
//
// Launched grid (32,1,1) block (1024,1,1) at PxgSoftBodyCore.cpp:614 and
// PxgFEMClothCore.cpp:991 and :1078, applying the finished rank permutation to
// the contact key buffer that reorderRigidContacts() then walks.
[shader("compute")]
[numthreads(1024, 1, 1)]
void radixSortCopy(uint3 dtid : SV_DispatchThreadID)
{
    const uint numElems = (Ptr<uint>(gRC.numKeys))[0]; // *numKeys (device pointer)
    const uint stride = WorkgroupCount().x * RS_BLOCK_DIM;

    Ptr<uint2> inValue = Ptr<uint2>(gRC.inValue);
    Ptr<uint2> outValue = Ptr<uint2>(gRC.outValue);
    Ptr<uint>  rank = Ptr<uint>(gRC.rank);

    for (uint w = dtid.x; w < numElems; w += stride)
        outValue[w] = inValue[rank[w]];
}

// ---------------------------------------------------------------------------
// clampMaxValues (MemCopyBalanced.cu:95): CUDA pack
//   (PxU32* value0, PxU32* value1, PxU32* value2, const PxU32 maxValue)
// -> argSizes {8,8,8,4}, natural-aligned to a 32-byte blob (28 rounded up to
// the 8-byte alignment of the pointer members).
// ---------------------------------------------------------------------------
struct ClampMaxValuesParams
{
    uint64_t value0;   // @0  PxU32*
    uint64_t value1;   // @8  PxU32*
    uint64_t value2;   // @16 PxU32*
    uint     maxValue; // @24 const PxU32 (by value)
    uint     _pad28;   // @28 -> 32-byte blob
}

[[vk::push_constant]]
ConstantBuffer<ClampMaxValuesParams> gClamp;

// mirrors clampMaxValues(): a single-thread in-place clamp of three device
// counters.  Launched grid (1,1,1) block (1,1,1) at PxgSoftBodyCore.cpp:474
// (rigid / particle / cloth contact totals) and PxgFEMClothCore.cpp:826
// (fem / rigid / particle contact totals), capping each at mMaxContacts.
//
// Three independent compare-and-stores, NOT an InterlockedMin: the CUDA uses a
// plain `if (*v > maxValue) *v = maxValue;` on a one-thread launch, and the
// shader keeps that verbatim.
[shader("compute")]
[numthreads(1, 1, 1)]
void clampMaxValues()
{
    Ptr<uint> value0 = Ptr<uint>(gClamp.value0);
    Ptr<uint> value1 = Ptr<uint>(gClamp.value1);
    Ptr<uint> value2 = Ptr<uint>(gClamp.value2);

    if (value0[0] > gClamp.maxValue)
        value0[0] = gClamp.maxValue;
    if (value1[0] > gClamp.maxValue)
        value1[0] = gClamp.maxValue;
    if (value2[0] > gClamp.maxValue)
        value2[0] = gClamp.maxValue;
}
