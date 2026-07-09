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

// npClosest.slang
//
// Slang port of two closest-point primitives from
// gpunarrowphase/src/CUDA/deformableCollision.cuh: closestPtPointTriangle (the
// Ericson 7-Voronoi-region closest point of a point on a triangle, returning the
// closest point and its (s,t) edge parameters) and closestPtPointEdge (the
// clamped closest-point parameter of a point on a segment).  Both the closest
// point and the parameters are continuous functions of the query point (they
// agree across region boundaries), so no measure-zero skip is needed.

struct ClParams { uint numTests; uint pad0, pad1, pad2; }

struct TriIn { float4 p, a, b, c; }
struct TriOut { float4 closest; float4 st; } // closest.xyz, st.xy = (s,t)
struct EdgeIn { float4 v0, ev0, ev1; }

[[vk::push_constant]]
ConstantBuffer<ClParams> gParams;

RWStructuredBuffer<TriIn> gTri;    // binding 0
RWStructuredBuffer<TriOut> gTriOut; // binding 1
RWStructuredBuffer<EdgeIn> gEdge;  // binding 2
RWStructuredBuffer<float> gEdgeOut; // binding 3  (w)

// closestPtPointTriangle: closest point of p on triangle abc; (s,t) are the edge
// parameters (barycentric weights of b and c).
float3 closestPtPointTriangle(float3 p, float3 a, float3 b, float3 c, out float s, out float t)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return a; }

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return b; }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        s = v; t = 0.0f;
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return c; }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        s = 0.0f; t = w;
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        s = 1.0f - w; t = w;
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    s = v; t = w;
    return a + ab * v + ac * w;
}

void closestPtPointEdge(float3 v0, float3 ev0, float3 ev1, out float w)
{
    const float3 r = ev1 - ev0;
    const float d2 = dot(r, r);
    w = (d2 > 0.0f) ? clamp(dot(r, v0 - ev0) / d2, 0.0f, 1.0f) : 0.5f;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctClosestPtTriangle(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const TriIn e = gTri[i];
    float s, t;
    const float3 cp = closestPtPointTriangle(e.p.xyz, e.a.xyz, e.b.xyz, e.c.xyz, s, t);
    TriOut o;
    o.closest = float4(cp, 0.0f);
    o.st = float4(s, t, 0.0f, 0.0f);
    gTriOut[i] = o;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctClosestPtEdge(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const EdgeIn e = gEdge[i];
    float w;
    closestPtPointEdge(e.v0.xyz, e.ev0.xyz, e.ev1.xyz, w);
    gEdgeOut[i] = w;
}
