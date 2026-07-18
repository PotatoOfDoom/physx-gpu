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

// npSpikeBda.slang — RUNTIME_PLAN.md P0 spike.
//
// Proves the buffer-device-address argument-marshalling model (P3): a
// CUDA-shaped parameter block whose members are raw device addresses travels
// entirely through push constants — the dispatch binds ZERO descriptors —
// and the kernel dereferences the pointers directly, exactly like a CUDA
// kernel receiving device pointers.

struct BdaParams
{
    uint64_t srcA;   // device address of float[count]
    uint64_t srcB;   // device address of float[count]
    uint64_t dst;    // device address of float[count]
    uint count;
    uint pad;
}

[[vk::push_constant]]
ConstantBuffer<BdaParams> gParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void spikeBda(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= gParams.count) return;

    Ptr<float> a = Ptr<float>(gParams.srcA);
    Ptr<float> b = Ptr<float>(gParams.srcB);
    Ptr<float> d = Ptr<float>(gParams.dst);

    d[dtid.x] = a[dtid.x] + b[dtid.x];
}
