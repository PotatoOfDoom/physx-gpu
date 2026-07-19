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

// npShimMisalign.slang — RUNTIME_PLAN.md P3 launch-marshalling proof kernel.
//
// This kernel's parameter struct puts a 64-bit value (`mul`) AFTER a 32-bit
// field (`n`), so the value's natural (8-byte-aligned) offset — 24 — differs
// from where a tight, no-padding concatenation of the arguments would place it
// (20).  When the host passes the four fields as SEPARATE PxCudaKernelParam
// entries (the multi-argument CUDA launch shape), PxgVulkanCudaContext must
// reproduce CUDA's natural-alignment .param layout; if it concatenated tightly,
// `mul` would be read from the wrong bytes and every output would be wrong.
//
// The two device pointers sit BEFORE the misalignment (offsets 0 and 8, both
// already 8-aligned in either layout), so a wrong layout corrupts `mul` — a
// data value — not a buffer address: the failure is deterministic wrong data,
// never a wild write.  out[i] = in[i] * mul + i.

struct MisParams
{
    uint64_t inPtr;   // @0   device address of uint[n]
    uint64_t outPtr;  // @8   device address of uint[n]
    uint     n;       // @16
    uint64_t mul;     // @24  natural alignment (tight packing would place it @20)
}

[[vk::push_constant]]
ConstantBuffer<MisParams> gParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void shimMisAdd(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.n)
        return;
    Ptr<uint> in = Ptr<uint>(gParams.inPtr);
    Ptr<uint> out = Ptr<uint>(gParams.outPtr);
    out[i] = in[i] * uint(gParams.mul) + i;
}
