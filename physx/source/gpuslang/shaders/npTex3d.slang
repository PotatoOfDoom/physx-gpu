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

// npTex3d.slang — RUNTIME_PLAN.md P2: the tex3D replacement path.
// Samples a single-channel float volume with trilinear filtering and
// clamp-to-edge addressing.  Coordinates arrive in the CUDA tex3D
// convention (unnormalized voxel space, texel i centered at i + 0.5); the
// kernel normalizes by the inverse extents, which is exact for
// power-of-two dimensions.
// P9.96: the engine path (pxgSlangTex3d.slang) no longer samples -- it fetches
// texels and evaluates CUDA's filter itself.  This probe keeps the hardware
// sampler as the record of what the Vulkan sampler does.

struct TxParams
{
    uint numSamples;
    float invW, invH, invD;
}

[[vk::push_constant]]
ConstantBuffer<TxParams> gParams;

RWStructuredBuffer<float4> gPts;  // binding 0: xyz = unnormalized coords
RWStructuredBuffer<float> gOut;   // binding 1
Texture3D<float> gSdf;            // binding 2 (sampled image)
SamplerState gSampler;            // binding 3 (sampler)

[shader("compute")]
[numthreads(64, 1, 1)]
void tex3dSample(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numSamples)
        return;
    const float3 p = gPts[i].xyz;
    const float3 uvw = p * float3(gParams.invW, gParams.invH, gParams.invD);
    gOut[i] = gSdf.SampleLevel(gSampler, uvw, 0.0);
}
