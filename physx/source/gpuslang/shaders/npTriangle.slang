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

// npTriangle.slang
//
// Slang port of the self-contained triangle geometry primitives shared across
// the narrow-phase mesh/heightfield collision paths (gpunarrowphase/src/CUDA/
// sphereTriangle.cuh + triangle.cuh):
//   - distancePointTriangleSquared: the Ericson closest-point-on-triangle test,
//     returning the squared distance, closest point, the Voronoi-region mask
//     (V0/V1/V2/E01/E02/E12/face) and the edge-convexity-gated generateContact /
//     faceContact flags;
//   - intersectLineTri: the one-sided line-vs-triangle intersection test.

static const uint BOUNDARY = 0xffffffffu;
static const uint NONCONVEX_FLAG = 0x80000000u;
static const uint eV0  = 1u << 29;
static const uint eV1  = 2u << 29;
static const uint eV2  = 3u << 29;
static const uint eE01 = 4u << 29;
static const uint eE02 = 5u << 29;
static const uint eE12 = 6u << 29;

struct TriIn
{
    float4 p, a, b, c; // query point + triangle verts (.xyz)
    uint4 adj;         // adjacency edge indices (x=e01, y=e12, z=e02)
}

struct TriResult
{
    float4 closest; // .xyz closest point, .w squared distance
    uint mask;
    uint gen;       // generateContact
    uint face;      // faceContact
    uint pad;
}

struct LineIn { float4 p, q, a, b, c; }

struct TriParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<TriParams> gParams;

RWStructuredBuffer<TriIn>     gTriIn;    // binding 0
RWStructuredBuffer<TriResult> gTriOut;   // binding 1
RWStructuredBuffer<LineIn>    gLineIn;   // binding 2
RWStructuredBuffer<uint>      gLineOut;  // binding 3

bool isEdgeNonconvex(uint adjEdgeIndex)
{
    return (adjEdgeIndex != BOUNDARY) && ((adjEdgeIndex & NONCONVEX_FLAG) != 0u);
}

// ---------------------------------------------------------------------------
// distancePointTriangleSquared: mirrors sphereTriangle.cuh
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void triPointDistance(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const TriIn inp = gTriIn[t];
    const float3 p = inp.p.xyz, a = inp.a.xyz, b = inp.b.xyz, c = inp.c.xyz;
    const uint4 adj = inp.adj;

    bool faceContact = false;
    bool generateContact;
    uint mask;
    float3 closestP;
    float distSq;

    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);

    const float vc = d1 * d4 - d3 * d2;
    const float vb = d5 * d2 - d1 * d6;
    const float va = d3 * d6 - d5 * d4;
    const float edgeTol = (va + vb + vc) * 1e-5f;

    if (d1 < 0.0f && d2 < 0.0f)
    {
        mask = eV0;
        generateContact = (!isEdgeNonconvex(adj.x)) || (!isEdgeNonconvex(adj.z));
        closestP = a;
        distSq = dot(ap, ap);
    }
    else if (d3 > 0.0f && d4 < d3)
    {
        mask = eV1;
        generateContact = (!isEdgeNonconvex(adj.x)) || (!isEdgeNonconvex(adj.y));
        closestP = b;
        distSq = dot(bp, bp);
    }
    else if (d6 > 0.0f && d5 < d6)
    {
        mask = eV2;
        generateContact = (!isEdgeNonconvex(adj.y)) || (!isEdgeNonconvex(adj.z));
        closestP = c;
        distSq = dot(cp, cp);
    }
    else if (vc < 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        mask = eE01;
        generateContact = !isEdgeNonconvex(adj.x) || vc > -edgeTol;
        const float v = d1 / (d1 - d3);
        closestP = a + v * ab;
        distSq = dot(closestP - p, closestP - p);
    }
    else if (vb < 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        mask = eE02;
        generateContact = !isEdgeNonconvex(adj.z) || vb > -edgeTol;
        const float w = d2 / (d2 - d6);
        closestP = a + w * ac;
        distSq = dot(closestP - p, closestP - p);
    }
    else if (va < 0.0f && d4 >= d3 && d5 >= d6)
    {
        mask = eE12;
        generateContact = !isEdgeNonconvex(adj.y) || va > -edgeTol;
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closestP = b + w * (c - b);
        distSq = dot(closestP - p, closestP - p);
    }
    else
    {
        generateContact = true;
        faceContact = true;
        mask = 0u;
        // P9.36: BOTH halves were wrong here.  GuDistancePointTriangle.h:94-97 is
        //     const float denom = 1.0f / (va + vb + vc);
        //     const float v = vb * denom;  const float w = vc * denom;
        //     return a + ab*v + ac*w;
        // -- a reciprocal, not three divides, AND the two scalars scaled before the
        // vector sum, not after.  The eighteen other copies of this tail in the
        // corpus already spell it this way; this one and the two convex ones did not.
        const float denom = 1.0f / (va + vb + vc);
        closestP = a + ab * (vb * denom) + ac * (vc * denom);
        distSq = dot(closestP - p, closestP - p);
    }

    TriResult r;
    r.closest = float4(closestP, distSq);
    r.mask = mask;
    r.gen = generateContact ? 1u : 0u;
    r.face = faceContact ? 1u : 0u;
    r.pad = 0u;
    gTriOut[t] = r;
}

float scalarTriple(float3 a, float3 b, float3 c) { return dot(cross(a, b), c); }

// ---------------------------------------------------------------------------
// intersectLineTri: mirrors sphereTriangle.cuh intersectLineTri()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void triLineIntersect(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const LineIn s = gLineIn[t];
    const float3 p = s.p.xyz, q = s.q.xyz, a = s.a.xyz, b = s.b.xyz, c = s.c.xyz;

    const float3 pq = q - p;
    const float3 pa = a - p;
    const float3 pb = b - p;
    const float3 pc = c - p;

    const float3 m = cross(pq, pc);
    const float u = dot(pb, m);
    const float v = -dot(pa, m);
    const float w = scalarTriple(pq, pb, pa);

    gLineOut[t] = (u >= 0.0f && v >= 0.0f && w >= 0.0f) ? 1u : 0u;
}
