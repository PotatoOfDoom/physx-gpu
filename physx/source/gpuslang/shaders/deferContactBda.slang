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
//

// deferContactBda.slang — a dispatch wrapper around pxgSlangDeferContact, so the
// deferral predicate can be certified before its consumer exists.  Not an engine
// kernel; absent from the manifest, like pcmManifoldBda, midphaseAllocBda and
// triFeatureBda before it.

import pxgSlangDeferContact;

struct DeferParams
{
    uint64_t minNormal;   // @0  float4[] (.xyz used)
    uint64_t triNormal;   // @8  float4[]
    uint64_t feature;     // @16 uint[]
    uint64_t outFlags;    // @24 uint[]
    uint     count;       // @32
    uint     _pad36;      // @36 -> 40-byte blob
}

[[vk::push_constant]]
ConstantBuffer<DeferParams> gDc;

[shader("compute")]
[numthreads(64, 1, 1)]
void deferContactLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gDc.count) return;
    const bool d = dcDelayGenerateContact(Ptr<float4>(gDc.minNormal)[i].xyz,
                                          Ptr<float4>(gDc.triNormal)[i].xyz,
                                          Ptr<uint>(gDc.feature)[i]);
    Ptr<uint>(gDc.outFlags)[i] = d ? 1u : 0u;
}
