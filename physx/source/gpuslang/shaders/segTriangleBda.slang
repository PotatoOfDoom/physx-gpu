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

// segTriangleBda.slang
//
// Test scaffolding for pxgSlangSegTriangle -- NOT an engine kernel, absent from
// the manifest.  P9.52b-2.
//
// TWO entry points, because the module has two functions with independent branch
// structure and driving only the outer one would leave the inner one's parallel
// and degenerate arms reachable solely by accident:
//
//   segTriLaunch  -- distanceSegmentTriangleSquared, one case per thread,
//                    reporting (t, u, v) and the squared distance.
//   segSegLaunch  -- distanceSegmentSegmentSquared on its own, reporting (s, t).
//                    Its inputs are a POINT AND A DIRECTION per segment, not two
//                    endpoints, which is the first thing a caller gets wrong.

import pxgSlangSegTriangle;

struct SegTriParams
{
    uint64_t cases;     // @0
    uint64_t output;    // @8
    uint     numCases;  // @16
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<SegTriParams> gSt;

// 80 bytes: p, q, a, b, c as float4 (w unused)
struct SegTriCase { float4 p, q, a, b, c; }
// 64 bytes: p1, d1, p2, d2
struct SegSegCase { float4 p1, d1, p2, d2; }

[shader("compute")]
[numthreads(64, 1, 1)]
void segTriLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gSt.numCases)
        return;

    const SegTriCase c = Ptr<SegTriCase>(gSt.cases)[i];

    float t, u, v;
    const float sq = stSegmentTriangleSquared(c.p.xyz, c.q.xyz, c.a.xyz, c.b.xyz, c.c.xyz,
                                              t, u, v);

    Ptr<float4>(gSt.output)[i] = float4(t, u, v, sq);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void segSegLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gSt.numCases)
        return;

    const SegSegCase c = Ptr<SegSegCase>(gSt.cases)[i];

    float s, t;
    stSegmentSegmentSquared(c.p1.xyz, c.d1.xyz, c.p2.xyz, c.d2.xyz, s, t);

    Ptr<float4>(gSt.output)[i] = float4(s, t, 0.0f, 0.0f);
}
