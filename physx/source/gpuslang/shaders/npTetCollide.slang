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

// npTetCollide.slang
//
// The leaf test for pxgSlangTetCollide -- tetPrimitivesCollide2 and computeInnerSphere.
// GJK and EPA have their own leaf tests (ctGjk, ctEpa) that pin their arithmetic; what this
// one pins is the DRIVER's policy, which is where a plausible-looking port goes wrong
// without any single line looking suspicious: which result short-circuits, when EPA is
// entered at all, and that a degenerate EPA answer forces the whole query to be re-run from
// a different seed.
import pxgSlangGjk;
import pxgSlangTetCollide;

struct TcParams { uint numTests; uint pad0, pad1, pad2; }
struct TcIn
{
    float4 searchDir;      // xyz = search direction, w = contactDistance
    float4 spheres;        // x = inSphereRadius0, y = inSphereRadius1
    uint4 counts;          // x = nbA, y = nbB
}
struct TcOut
{
    uint result; uint degenerate; uint size; uint pad;
    float lowerBound; float upperBound; float innerSphere; float pad2;
    float4 direction; float4 closestPointA; float4 closestPointB; float4 tetCenter;
}

[[vk::push_constant]]
ConstantBuffer<TcParams> gParams;

RWStructuredBuffer<float4> gVA;   // binding 0: rigid cloud, 64 per test
RWStructuredBuffer<float4> gVB;   // binding 1: tet cloud, 64 per test
RWStructuredBuffer<TcIn> gIn;     // binding 2
RWStructuredBuffer<TcOut> gOut;   // binding 3

[shader("compute")]
[numthreads(64, 1, 1)]
void ctTetCollide(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint vbase = t * GJK_MAX_CLOUD;

    const TcIn e = gIn[t];

    GjkCloud a, b;
    a.n = e.counts.x;
    b.n = e.counts.y;
    for (uint i = 0u; i < GJK_MAX_CLOUD; ++i)
    {
        a.v[i] = (i < a.n) ? gVA[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
        b.v[i] = (i < b.n) ? gVB[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
    }

    GjkOut o;
    const uint r = tetPrimitivesCollide2(a, b, e.searchDir.xyz, e.searchDir.w,
                                         e.spheres.x, e.spheres.y, o);

    float3 center;
    const float inner = computeInnerSphere(b.v[0], b.v[1], b.v[2], b.v[3], center);

    TcOut out0;
    out0.result = r;
    out0.degenerate = o.degenerate;
    out0.size = o.size;
    out0.pad = 0u; out0.pad2 = 0.0f;
    out0.lowerBound = o.lowerBound;
    out0.upperBound = o.upperBound;
    out0.innerSphere = inner;
    out0.direction = o.direction;
    out0.closestPointA = o.closestPointA;
    out0.closestPointB = o.closestPointB;
    out0.tetCenter = float4(center, 0.0f);
    gOut[t] = out0;
}
