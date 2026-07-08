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

// radixSort64.slang
//
// Slang port of the 64-bit key support kernel of the PhysX GPU radix sort
// (gpucommon/src/CUDA/radixSortImpl.cu):
//   - radixSortCopyHigh32Bits
//
// 64-bit keys are sorted as two chained 32-bit sorts: the low words are
// sorted first (8 passes of radixSort.slang, ranks = original indices), then
// this kernel gathers the *high* words in low-sort rank order, and a second
// stable 8-pass sort of those high words — with the rank payload threading
// through — yields the 64-bit stable order.  The 64-bit keys are bound as a
// uint2 buffer (x = low word, y = high word) so no shader 64-bit integer
// capability is required.
//
// radixSortDoubleCopyHigh32Bits (the CUDA kernel fusing two copies into one
// launch) maps to two dispatches of this kernel.
//
// The element count is read from a device buffer, matching the CUDA kernel
// (sorts frequently run on GPU-computed counts without a CPU round-trip).

static const uint BLOCK_SIZE = 256;

RWStructuredBuffer<uint2> gInValue64; // binding 0: 64-bit keys (low, high)
RWStructuredBuffer<uint> gOutValue;   // binding 1: gathered high words
RWStructuredBuffer<uint> gRank;       // binding 2: ranks from the low-word sort
RWStructuredBuffer<uint> gNumKeys;    // binding 3: single entry, device-side count

// mirrors radixSortCopyHigh32Bits() in radixSortImpl.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortCopyHigh32Bits(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint numElems = gNumKeys[0];
    const uint workIndex = dispatchThreadId.x;

    if (workIndex >= numElems)
        return;

    const uint index = gRank[workIndex];
    gOutValue[workIndex] = gInValue64[index].y;
}
