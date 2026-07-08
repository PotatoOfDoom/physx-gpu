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

// radixSortGather.slang
//
// Slang port of the remaining gpucommon radix-sort gather kernels
// (gpucommon/src/CUDA/radixSortImpl.cu) — the rank-indexed copies that reorder
// a 64-bit key array (and, for the "Bits2"/"2" variants, mask aggregate
// sentinels):
//   - radixSortCopy: outValue64[i] = inValue64[rank[i]]
//   - radixSortCopy2: same but rank == 0xffffffff (aggregate) -> 0xffff..ffff
//   - radixSortCopyBits2: outValue32[i] = low or high 32 bits of the gathered
//     key, or 0xffffffff for an aggregate rank
//   - radixSortDoubleCopy: two independent 64-bit gathers in one launch
//   - radixSortDoubleCopyHigh32Bits: two independent high-word gathers
//
// 64-bit keys are stored as uint2 (x = low word, y = high word), matching
// radixSort64.slang.  numKeys is passed as a push constant (the CUDA reads it
// from a device pointer or a value argument; the host provides the count).

struct GatherParams
{
    uint numKeys;
    uint lowBit;   // radixSortCopyBits2: select low (1) or high (0) word
    uint gridDimX;
}

[[vk::push_constant]]
ConstantBuffer<GatherParams> gParams;

RWStructuredBuffer<uint2> gInValue0;  // binding 0  64-bit input 0
RWStructuredBuffer<uint>  gRank0;     // binding 1  ranks for input 0
RWStructuredBuffer<uint2> gOut64_0;   // binding 2  64-bit output 0
RWStructuredBuffer<uint>  gOut32_0;   // binding 3  32-bit output 0
RWStructuredBuffer<uint2> gInValue1;  // binding 4  64-bit input 1 (double copies)
RWStructuredBuffer<uint>  gRank1;     // binding 5  ranks for input 1
RWStructuredBuffer<uint2> gOut64_1;   // binding 6  64-bit output 1
RWStructuredBuffer<uint>  gOut32_1;   // binding 7  32-bit output 1

static const uint AGGREGATE = 0xffffffffu;

// ---------------------------------------------------------------------------
// radixSortCopy: 64-bit rank-indexed gather
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void radixSortCopy(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = gParams.gridDimX * 256u;
    for (uint w = dtid.x; w < gParams.numKeys; w += stride)
        gOut64_0[w] = gInValue0[gRank0[w]];
}

// ---------------------------------------------------------------------------
// radixSortCopy2: 64-bit gather with aggregate sentinel
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void radixSortCopy2(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = gParams.gridDimX * 256u;
    for (uint w = dtid.x; w < gParams.numKeys; w += stride)
    {
        const uint index = gRank0[w];
        gOut64_0[w] = (index == AGGREGATE) ? uint2(AGGREGATE, AGGREGATE) : gInValue0[index];
    }
}

// ---------------------------------------------------------------------------
// radixSortCopyBits2: 32-bit low/high word gather with aggregate sentinel
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void radixSortCopyBits2(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = gParams.gridDimX * 256u;
    const bool lowBit = gParams.lowBit != 0u;
    for (uint w = dtid.x; w < gParams.numKeys; w += stride)
    {
        const uint index = gRank0[w];
        if (index == AGGREGATE)
            gOut32_0[w] = AGGREGATE;
        else
            gOut32_0[w] = lowBit ? gInValue0[index].x : gInValue0[index].y;
    }
}

// ---------------------------------------------------------------------------
// radixSortDoubleCopy: two 64-bit gathers
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void radixSortDoubleCopy(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = gParams.gridDimX * 256u;
    for (uint w = dtid.x; w < gParams.numKeys; w += stride)
    {
        gOut64_0[w] = gInValue0[gRank0[w]];
        gOut64_1[w] = gInValue1[gRank1[w]];
    }
}

// ---------------------------------------------------------------------------
// radixSortDoubleCopyHigh32Bits: two high-word gathers
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void radixSortDoubleCopyHigh32Bits(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = gParams.gridDimX * 256u;
    for (uint w = dtid.x; w < gParams.numKeys; w += stride)
    {
        gOut32_0[w] = gInValue0[gRank0[w]].y; // high 32 bits
        gOut32_1[w] = gInValue1[gRank1[w]].y;
    }
}
