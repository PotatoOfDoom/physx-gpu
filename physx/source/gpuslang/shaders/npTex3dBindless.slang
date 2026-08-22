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

// npTex3dBindless.slang -- P9.12a, the fixture entry over pxgSlangTex3d.
//
// Everything about this entry point exists to make the CUtexObject property
// testable, which is a narrower claim than "sampling works" (npTex3d already
// covers that against a bound descriptor):
//
//   * the handle is READ FROM MEMORY, one per sample, so lanes in one subgroup
//     carry different handles and the NonUniformResourceIndex decoration is
//     exercised rather than merely present;
//   * the extent is QUERIED, not passed, because the sparse subgrid texture's
//     extent is not carried in any struct the kernel can see;
//   * every buffer travels as a DEVICE ADDRESS, so the SPIR-V has to hold
//     PhysicalStorageBufferAddresses and RuntimeDescriptorArray at once -- the
//     combination the SDF kernels need and the one most likely to be missing.

import pxgSlangTex3d;

struct TxbParams
{
    uint64_t handles;   // uint per sample -- the bindless slot
    uint64_t coords;    // float4 per sample, xyz = UNNORMALIZED texel coords
    uint64_t outValue;  // float per sample
    uint64_t outSize;   // uint4 per sample: the queried extent, w = the handle
    uint numSamples;
}

[[vk::push_constant]]
ConstantBuffer<TxbParams> gP;

[shader("compute")]
[numthreads(64, 1, 1)]
void tex3dBindlessSample(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numSamples)
        return;

    const uint handle = Ptr<uint>(gP.handles)[i];
    const float4 p = Ptr<float4>(gP.coords)[i];

    const uint3 dim = pxgTex3DSize(handle);
    Ptr<uint4>(gP.outSize)[i] = uint4(dim.x, dim.y, dim.z, handle);
    Ptr<float>(gP.outValue)[i] = pxgTex3D(handle, p.xyz);
}
