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

// npShimAxpy.slang — RUNTIME_PLAN.md P3 shim proof kernel.
//
// The three device pointers travel NESTED inside a struct, mirroring how
// PhysX kernels receive pointers embedded in aggregate parameter structs
// (PxgSoftBody, PxgParticleSystem, ...).  The whole parameter pack is a
// single push-constant blob the shim forwards verbatim; the kernel walks
// the nested addresses as buffer-device-address pointers.  out = a*x + y.

struct AxpyBuffers
{
    uint64_t x;    // device address of float[count]
    uint64_t y;    // device address of float[count]
    uint64_t out;  // device address of float[count]
}

struct AxpyParams
{
    AxpyBuffers bufs;
    float alpha;
    uint count;
}

[[vk::push_constant]]
ConstantBuffer<AxpyParams> gParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void shimAxpy(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.count)
        return;
    Ptr<float> x = Ptr<float>(gParams.bufs.x);
    Ptr<float> y = Ptr<float>(gParams.bufs.y);
    Ptr<float> out = Ptr<float>(gParams.bufs.out);
    out[i] = gParams.alpha * x[i] + y[i];
}
