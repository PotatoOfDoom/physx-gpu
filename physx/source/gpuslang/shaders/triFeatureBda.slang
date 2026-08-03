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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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

// triFeatureBda.slang — a dispatch wrapper around pxgSlangTriFeature, so the
// second pass's classifier can be certified before its consumer exists.  Not an
// engine kernel; absent from the manifest, like pcmManifoldBda and
// midphaseAllocBda before it.

import pxgSlangTriFeature;

struct FeatureParams
{
    uint64_t bary;   // @0  float2[] — (v, w) per case
    uint64_t pts;    // @8  float4[4*n] — pa, v0, v1, v2 per case (.w unused)
    uint64_t outB;   // @16 uint[] — mask from the barycentric overload
    uint64_t outP;   // @24 uint[] — mask from the point overload
    uint     count;  // @32
    uint     _pad36; // @36 -> 40-byte blob
}

[[vk::push_constant]]
ConstantBuffer<FeatureParams> gTf;

[shader("compute")]
[numthreads(64, 1, 1)]
void triFeatureLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gTf.count) return;

    const float2 vw = Ptr<float2>(gTf.bary)[i];
    Ptr<uint>(gTf.outB)[i] = tfActiveMaskFromBary(vw.x, vw.y);

    Ptr<float4> p = Ptr<float4>(gTf.pts);
    Ptr<uint>(gTf.outP)[i] = tfActiveMask(p[i*4u + 0u].xyz, p[i*4u + 1u].xyz,
                                          p[i*4u + 2u].xyz, p[i*4u + 3u].xyz);
}
