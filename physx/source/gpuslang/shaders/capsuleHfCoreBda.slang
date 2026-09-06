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

// capsuleHfCoreBda.slang
//
// Test scaffolding for pxgSlangCapsuleHfCore -- NOT an engine kernel, absent from
// the manifest.  P9.52b-4.
//
// ONE THREAD PER PAIR, the shape the CUDA core has.  The four record arrays are
// real buffers at their engine strides, so the fixture checks WHERE each write
// landed as well as what it computed -- four of the tail's writes are indexed by
// `pairOffset` and the ordered word by `pairOffsetPadded`.
//
// Output record, 48 bytes:
//   @0   uint4  nbOutputContacts, pairOffset, _pad, _pad
//   @16  float4 contact 0 xyz (in the CAPSULE's frame), separation 0
//   @32  float4 contact 1 xyz, separation 1

import pxgSlangCapsuleHfCore;

struct CapsuleHfCoreParams
{
    uint64_t cmInputs;          // @0
    uint64_t transformCache;    // @8
    uint64_t contactDistance;   // @16
    uint64_t gpuShapes;         // @24
    uint64_t sphereTrimeshPair; // @32
    uint64_t ni;                // @40
    uint64_t maxDepth;          // @48
    uint64_t interm;            // @56
    uint64_t ordered;           // @64
    uint64_t pairs;             // @72
    uint64_t output;            // @80
    uint     numPairs;          // @88
    uint     _pad92;
}

[[vk::push_constant]]
ConstantBuffer<CapsuleHfCoreParams> gChc;

static const uint CHC_OUT_STRIDE = 48u;

[shader("compute")]
[numthreads(64, 1, 1)]
void capsuleHfCoreLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint p = dtid.x;
    if (p >= gChc.numPairs)
        return;

    const uint4 curPair = Ptr<uint4>(gChc.pairs)[p];

    const CapHfOut o = chCapsuleCore(gChc.cmInputs, gChc.transformCache,
                                     gChc.contactDistance, gChc.gpuShapes,
                                     gChc.sphereTrimeshPair,
                                     gChc.ni, gChc.maxDepth, gChc.interm, gChc.ordered,
                                     curPair);

    const uint64_t out = gChc.output + uint64_t(p) * uint64_t(CHC_OUT_STRIDE);
    Ptr<uint4>(out)[0] = uint4(o.nbOutputContacts, o.pairOffset, 0u, 0u);
    Ptr<float4>(out + 16UL)[0] = float4(o.contact[0], o.sep[0]);
    Ptr<float4>(out + 32UL)[0] = float4(o.contact[1], o.sep[1]);
}
