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

// npSbTet.slang
//
// Slang port of the softbody tetrahedron collision primitives:
//
// - Gu::createPlane / constructTetrahedron
//   (geomutils/include/GuIntersectionTetrahedronTetrahedron.h): the
//   Tetrahedron with four INWARD-oriented face planes (each plane built from
//   three vertices and flipped if the fourth lies on its positive side) and
//   the centroid — which the CUDA deliberately computes as (a+b+c)/3, NOT the
//   true tet centroid.
// - Gu::satIntersect<true> for tetrahedron pairs (same header): the 8 face
//   planes via minProject with early-outs at `sep > tolerance`, then the 6x6
//   edge-cross axes with the magSq > 1e-5 validity gate, running max.
// - closestPtTetrahedron (gpunarrowphase/src/CUDA/softbodySoftbodyMidPhase.cu):
//   the hint-gated per-face closest point of a point on a tetrahedron — for
//   each surface face selected by the PxU8 hint bits (1: abc, 2: abd, 4: acd,
//   8: bcd) run closestPtPointTriangle2 (the Ericson core), keep the running
//   minimum squared distance, and produce the face normal oriented AWAY from
//   the opposite vertex (flip when the opposite-vertex direction has positive
//   dot; face bcd flips on ab.dot > 0).  sqDist is a running input/output —
//   the caller may pre-seed it, in which case farther faces do not update.

static const float PX_MAX_F32 = 3.4028235e38f;

struct StParams { uint numTests; uint pad0, pad1, pad2; }

struct StIn
{
    float4 a0, b0, c0, d0;   // tet 0
    float4 a1, b1, c1, d1;   // tet 1 (SAT)
    float4 point;            // closestPtTetrahedron query point
    float4 misc;             // x = tolerance (SAT), y = sqDist seed (closestPt), z/w unused
    uint4 flags;             // x = hint (closestPt)
}

struct StOut
{
    float4 planes0[4];       // constructTetrahedron(tet0) planes (n.xyz, d)
    float4 centroid_sep;     // xyz = centroid(tet0), w = satIntersect(tet0, tet1, tolerance)
    float4 closest_sq;       // xyz = closestPtTetrahedron closest point, w = final sqDist
    float4 normal;           // closestPtTetrahedron normal
}

[[vk::push_constant]]
ConstantBuffer<StParams> gParams;

RWStructuredBuffer<StIn> gIn;    // binding 0
RWStructuredBuffer<StOut> gOut;  // binding 1

float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// PxPlane(p0, p1, p2)
void makePlane(float3 p0, float3 p1, float3 p2, out float3 n, out float d)
{
    n = getNormalized(cross(p1 - p0, p2 - p0));
    d = -dot(p0, n);
}

// Gu::createPlane: flip so pd lies on the non-positive side
void createPlane(float3 pa, float3 pb, float3 pc, float3 pd, out float3 n, out float d)
{
    makePlane(pa, pb, pc, n, d);
    const float distance = dot(n, pd) + d;
    if (distance > 0.0f)
    {
        n = -n;
        d = -d;
    }
}

// Gu::minProject(plane, tet)
float minProject(float3 n, float d, float3 v0, float3 v1, float3 v2, float3 v3)
{
    return min(dot(n, v0), min(dot(n, v1), min(dot(n, v2), dot(n, v3)))) + d;
}

float testSeparatingAxis(float3 axis, float3 a0[4], float3 a1[4])
{
    float min0, max0, min1, max1;
    min0 = max0 = dot(a0[0], axis);
    min1 = max1 = dot(a1[0], axis);
    for (uint i = 1; i < 4; ++i)
    {
        const float p0 = dot(a0[i], axis);
        const float p1 = dot(a1[i], axis);
        min0 = min(p0, min0);
        max0 = max(p0, max0);
        min1 = min(p1, min1);
        max1 = max(p1, max1);
    }
    return max(min1 - max0, min0 - max1);
}

// closestPtPointTriangle2 (GuDistancePointTriangle.h) — the Ericson core
float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return a + w * ac;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

// closestPtTetrahedron (softbodySoftbodyMidPhase.cu)
void closestPtTetrahedron(float3 point, uint hint, float3 a, float3 b, float3 c, float3 d,
                          inout float3 normal, inout float3 closestP, inout float sqDist)
{
    const float3 ac = c - a;
    const float3 ab = b - a;
    const float3 ad = d - a;

    if ((hint & 1) != 0) // face abc
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, c, ab, ac);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ab));
            if (dot(ad, normal) < 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 2) != 0) // face abd
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, d, ab, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ab, ad));
            if (dot(ac, normal) < 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 4) != 0) // face acd
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, c, d, ac, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ad));
            if (dot(ab, normal) < 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 8) != 0) // face bcd
    {
        const float3 bd = d - b;
        const float3 bc = c - b;
        const float3 tPoint = closestPtPointTriangle2(point, b, c, d, bc, bd);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(bc, bd));
            if (dot(ab, normal) > 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)]
void ctSbTet(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const StIn e = gIn[t];

    float3 v0[4] = { e.a0.xyz, e.b0.xyz, e.c0.xyz, e.d0.xyz };
    float3 v1[4] = { e.a1.xyz, e.b1.xyz, e.c1.xyz, e.d1.xyz };

    StOut o;

    // constructTetrahedron(tet0): planes + centroid ((a+b+c)/3 — as the CUDA does)
    float3 pn[4];
    float pd[4];
    createPlane(v0[0], v0[1], v0[2], v0[3], pn[0], pd[0]);
    createPlane(v0[0], v0[1], v0[3], v0[2], pn[1], pd[1]);
    createPlane(v0[0], v0[2], v0[3], v0[1], pn[2], pd[2]);
    createPlane(v0[1], v0[2], v0[3], v0[0], pn[3], pd[3]);
    for (uint i = 0; i < 4; ++i)
        o.planes0[i] = float4(pn[i], pd[i]);
    const float3 centroid = (v0[0] + v0[1] + v0[2]) * (1.0f / 3.0f);

    // satIntersect<true>(tet0, tet1, tolerance)
    const float tolerance = e.misc.x;
    float3 pn1[4];
    float pd1[4];
    createPlane(v1[0], v1[1], v1[2], v1[3], pn1[0], pd1[0]);
    createPlane(v1[0], v1[1], v1[3], v1[2], pn1[1], pd1[1]);
    createPlane(v1[0], v1[2], v1[3], v1[1], pn1[2], pd1[2]);
    createPlane(v1[1], v1[2], v1[3], v1[0], pn1[3], pd1[3]);

    float sep = -PX_MAX_F32;
    bool done = false;
    for (uint i = 0; i < 4 && !done; ++i)
    {
        sep = max(sep, minProject(pn[i], pd[i], v1[0], v1[1], v1[2], v1[3]));
        if (sep > tolerance) done = true;
    }
    for (uint i = 0; i < 4 && !done; ++i)
    {
        sep = max(sep, minProject(pn1[i], pd1[i], v0[0], v0[1], v0[2], v0[3]));
        if (sep > tolerance) done = true;
    }
    if (!done)
    {
        float3 axes0[6] = { v0[1] - v0[0], v0[2] - v0[0], v0[3] - v0[0], v0[2] - v0[1], v0[3] - v0[1], v0[3] - v0[2] };
        float3 axes1[6] = { v1[1] - v1[0], v1[2] - v1[0], v1[3] - v1[0], v1[2] - v1[1], v1[3] - v1[1], v1[3] - v1[2] };
        for (uint i = 0; i < 6 && !done; ++i)
        {
            for (uint j = 0; j < 6 && !done; ++j)
            {
                float3 sepAxis = cross(axes0[i], axes1[j]);
                const float magSq = dot(sepAxis, sepAxis);
                if (magSq > 1e-5f)
                {
                    sepAxis = sepAxis * (1.0f / sqrt(magSq));
                    sep = max(sep, testSeparatingAxis(sepAxis, v0, v1));
                    if (sep > tolerance) done = true;
                }
            }
        }
    }
    o.centroid_sep = float4(centroid, sep);

    // closestPtTetrahedron(point = a1, hint, tet0) with the sqDist seed
    float3 normal = float3(0, 0, 0);
    float3 closestP = float3(0, 0, 0);
    float sqDist = e.misc.y;
    closestPtTetrahedron(e.point.xyz, e.flags.x, v0[0], v0[1], v0[2], v0[3], normal, closestP, sqDist);
    o.closest_sq = float4(closestP, sqDist);
    o.normal = float4(normal, 0);

    gOut[t] = o;
}
