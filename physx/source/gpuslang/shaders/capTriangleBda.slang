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

// capTriangleBda.slang
//
// Test scaffolding for pxgSlangCapTriangle -- NOT an engine kernel, absent from
// the manifest.  P9.52b-3.
//
// THREE entry points:
//
//   capFaceLaunch  -- generateContacts alone, so its own two `if` blocks and the
//                     ideom guard are driven without the edge-edge pass on top.
//   capEELaunch    -- generateEEContacts alone from a per-case STARTING count, so
//                     the `numContacts < 2` guards are reachable directly.  With
//                     only the composed entry point, a start of 1 would need a
//                     face contact to exist first, which couples two independent
//                     decisions.
//   capBothLaunch  -- the composition the cores actually run
//                     (convexHeightfield.cu:770-775): generateContacts, then
//                     generateEEContacts on its result.
//
// Output record, 48 bytes:
//   @0   uint4  n, _pad, _pad, _pad
//   @16  float4 contact 0 xyz, separation 0
//   @32  float4 contact 1 xyz, separation 1

import pxgSlangCapTriangle;

struct CapTriParams
{
    uint64_t cases;     // @0
    uint64_t output;    // @8
    uint     numCases;  // @16
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<CapTriParams> gCt;

// 128 bytes
struct CapTriCase
{
    float4 a, b, c;
    float4 planeNormal, normal;
    float4 p, q;
    float4 cfg;   // x = inflatedRadius, y = sqInflatedRadius, z = starting count
}

static const uint CAP_OUT_STRIDE = 48u;

void capWrite(uint i, CapContacts o)
{
    const uint64_t out = gCt.output + uint64_t(i) * uint64_t(CAP_OUT_STRIDE);
    Ptr<uint4>(out)[0] = uint4(o.n, 0u, 0u, 0u);
    Ptr<float4>(out + 16UL)[0] = float4(o.pt[0], o.sep[0]);
    Ptr<float4>(out + 32UL)[0] = float4(o.pt[1], o.sep[1]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void capFaceLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gCt.numCases) return;
    const CapTriCase c = Ptr<CapTriCase>(gCt.cases)[i];

    capWrite(i, capGenerateContacts(c.a.xyz, c.b.xyz, c.c.xyz,
                                    c.planeNormal.xyz, c.normal.xyz,
                                    c.p.xyz, c.q.xyz, c.cfg.x));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void capEELaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gCt.numCases) return;
    const CapTriCase c = Ptr<CapTriCase>(gCt.cases)[i];

    // A synthetic starting count, with two recognisable slots already filled so a
    // port that overwrote instead of appending is visible in the output rather
    // than merely in the count.
    CapContacts s = capContactsInit();
    s.n = uint(c.cfg.z);
    s.pt[0] = float3(-7.0f, -7.0f, -7.0f);
    s.sep[0] = -7.0f;
    s.pt[1] = float3(-9.0f, -9.0f, -9.0f);
    s.sep[1] = -9.0f;

    capWrite(i, capGenerateEEContacts(c.a.xyz, c.b.xyz, c.c.xyz, c.normal.xyz,
                                      c.p.xyz, c.q.xyz, c.cfg.y, s));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void capBothLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gCt.numCases) return;
    const CapTriCase c = Ptr<CapTriCase>(gCt.cases)[i];

    const CapContacts face = capGenerateContacts(c.a.xyz, c.b.xyz, c.c.xyz,
                                                 c.planeNormal.xyz, c.normal.xyz,
                                                 c.p.xyz, c.q.xyz, c.cfg.x);
    capWrite(i, capGenerateEEContacts(c.a.xyz, c.b.xyz, c.c.xyz, c.normal.xyz,
                                      c.p.xyz, c.q.xyz, c.cfg.y, face));
}
