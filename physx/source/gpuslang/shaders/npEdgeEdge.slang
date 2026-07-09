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

// npEdgeEdge.slang
//
// Slang port of the finite segment-segment closest-point routines from
// gpunarrowphase/src/CUDA/deformableCollision.cuh: closestPtLineLine (the
// closest points of two infinite lines, with degenerate and near-parallel
// early-outs that return false) and closestPtEdgeEdge (the finite-segment
// version — it clamps to the four endpoint/edge projections via
// closestPtPointEdge when the line solution falls outside the [0,1] segment
// range).  Returns the segment parameters (s,t) and the squared distance.

struct EEParams { uint numTests; uint pad0, pad1, pad2; }

struct EEIn { float4 e0v0, e0v1, e1v0, e1v1; }
struct EEOut { float4 res; } // res.x=s, res.y=t, res.z=distSq, res.w=ok(1/0)

[[vk::push_constant]]
ConstantBuffer<EEParams> gParams;

RWStructuredBuffer<EEIn> gIn;        // binding 0
RWStructuredBuffer<EEOut> gLineOut;  // binding 1
RWStructuredBuffer<EEOut> gEdgeOut;  // binding 2

// Clamped closest-point parameter of v0 on segment [ev0,ev1] (see npClosest).
void closestPtPointEdge(float3 v0, float3 ev0, float3 ev1, out float w)
{
    const float3 r = ev1 - ev0;
    const float d2 = dot(r, r);
    w = (d2 > 0.0f) ? clamp(dot(r, v0 - ev0) / d2, 0.0f, 1.0f) : 0.5f;
}

bool closestPtLineLine(float3 e0v0, float3 e0v1, float3 e1v0, float3 e1v1, out float s, out float t, out float distSq)
{
    const float3 r0 = e0v1 - e0v0; // Edge0 direction
    const float3 r1 = e1v1 - e1v0; // Edge1 direction
    const float3 d = e1v0 - e0v0;  // Displacement between starting points

    const float r0Len2 = dot(r0, r0);
    const float r1Len2 = dot(r1, r1);

    // Handle degenerate edges: if either edge has zero length, return midpoints.
    if (r0Len2 < 1.0e-18f || r1Len2 < 1.0e-18f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return false;
    }

    const float r0Len = sqrt(r0Len2);
    const float r1Len = sqrt(r1Len2);

    const float3 u0 = r0 / r0Len;
    const float3 u1 = r1 / r1Len;

    const float cosTheta = abs(dot(u0, u1));

    // Edges are nearly parallel.
    if (cosTheta > 0.9999f)
    {
        const float tProj = dot(e0v0 - e1v0, r1) / r1Len2;
        t = tProj;
        s = 0.0f;

        const float3 pt0 = e0v0;
        const float3 pt1 = e1v0 + r1 * tProj;
        distSq = dot(pt1 - pt0, pt1 - pt0);
        return false;
    }

    const float a01 = -dot(u0, u1);
    const float b0 = dot(u0, d) / r0Len;
    const float b1 = -dot(u1, d) / r1Len;

    const float det = 1.0f - a01 * a01;

    if (det < 1.0e-9f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return false;
    }

    const float invDet = 1.0f / det;
    s = (b0 - b1 * a01) * invDet;
    t = (b1 - a01 * b0) * invDet;

    const float3 pt0 = e0v0 + s * r0;
    const float3 pt1 = e1v0 + t * r1;
    distSq = dot(pt1 - pt0, pt1 - pt0);

    return true;
}

// Closest points on two finite edges.
bool closestPtEdgeEdge(float3 e0v0, float3 e0v1, float3 e1v0, float3 e1v1, out float s, out float t, out float distSq)
{
    if (!closestPtLineLine(e0v0, e0v1, e1v0, e1v1, s, t, distSq))
    {
        return false; // Degenerate / parallel edges
    }

    // Falls back to point-edge distance when the line solution is outside [0,1].
    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f)
    {
        float w0, w1, w2, w3;
        closestPtPointEdge(e0v0, e1v0, e1v1, w0);
        closestPtPointEdge(e0v1, e1v0, e1v1, w1);
        closestPtPointEdge(e1v0, e0v0, e0v1, w2);
        closestPtPointEdge(e1v1, e0v0, e0v1, w3);

        float2 cand[4];
        cand[0] = float2(0.0f, w0);
        cand[1] = float2(1.0f, w1);
        cand[2] = float2(w2, 0.0f);
        cand[3] = float2(w3, 1.0f);

        int minIndex = 0;
        distSq = 3.4028235e38f; // PX_MAX_REAL

        [unroll]
        for (int i = 0; i < 4; ++i)
        {
            const float sC = cand[i].x;
            const float tC = cand[i].y;
            const float3 p0 = e0v0 * (1.0f - sC) + e0v1 * sC;
            const float3 p1 = e1v0 * (1.0f - tC) + e1v1 * tC;
            const float d2 = dot(p0 - p1, p0 - p1);
            minIndex = (d2 < distSq) ? i : minIndex;
            distSq = min(d2, distSq);
        }

        s = cand[minIndex].x;
        t = cand[minIndex].y;
    }

    return true;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctLineLine(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const EEIn e = gIn[i];
    float s, t, distSq;
    const bool ok = closestPtLineLine(e.e0v0.xyz, e.e0v1.xyz, e.e1v0.xyz, e.e1v1.xyz, s, t, distSq);
    EEOut o;
    o.res = float4(s, t, distSq, ok ? 1.0f : 0.0f);
    gLineOut[i] = o;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctEdgeEdge(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const EEIn e = gIn[i];
    float s, t, distSq;
    const bool ok = closestPtEdgeEdge(e.e0v0.xyz, e.e0v1.xyz, e.e1v0.xyz, e.e1v1.xyz, s, t, distSq);
    EEOut o;
    o.res = float4(s, t, distSq, ok ? 1.0f : 0.0f);
    gEdgeOut[i] = o;
}
