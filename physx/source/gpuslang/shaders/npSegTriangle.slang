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

// npSegTriangle.slang
//
// Slang port of distanceSegmentTriangleSquared from
// gpunarrowphase/src/CUDA/capsuleTriangle.cuh — the squared distance between a
// segment (p,q) and a triangle (a,b,c), returning the segment parameter t and
// the triangle barycentric coordinates (u,v) of the closest features (used by
// the capsule/mesh narrow-phase).  The three cases are: the segment crosses the
// triangle plane inside the triangle (distance 0); both endpoints project inside
// the triangle (pick the nearer face projection); otherwise the closest of the
// three segment-vs-edge distances is combined with any interior projection.  The
// helpers isValidTriangleBarycentricCoord and distanceSegmentSegmentSquared
// (from distanceSegmentSegment.cuh) are inlined.

static const float PXG_MAX_F32 = 3.4028235e38f;

struct SegTriIn { float4 p, q, a, b, c; }
struct SegTriParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SegTriParams> gParams;

RWStructuredBuffer<SegTriIn> gIn;    // binding 0
RWStructuredBuffer<float4>   gOut;   // binding 1  (sqDist, t, u, v)

bool isValidBary(float v, float w)
{
    const float eps = 1e-6f;
    const float zero = -eps;
    const float one = 1.0f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

// distanceSegmentSegmentSquared: closest-point parameters s (on seg1) / t (seg2)
void segSeg(float3 p1, float3 d1, float3 p2, float3 d2, out float s, out float t)
{
    const float eps = 1e-6f;
    const float3 r = p1 - p2;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float b = dot(d1, d2);
    const float c = dot(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = dot(d2, r);

    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = clamp(temp / denom, 0.0f, 1.0f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = clamp(tTmp, 0.0f, 1.0f);
    const float comp = (b * t2 - c) * aRecip;
    const float s2 = clamp(comp, 0.0f, 1.0f);
    s = s2;
    t = t2;
}

// ---------------------------------------------------------------------------
// distanceSegmentTriangleSquared: mirrors capsuleTriangle.cuh
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void segTriDistance(uint3 dtid : SV_DispatchThreadID)
{
    const uint idx = dtid.x;
    if (idx >= gParams.numTests) return;
    const SegTriIn s = gIn[idx];
    const float3 p = s.p.xyz, q = s.q.xyz, a = s.a.xyz, b = s.b.xyz, c = s.c.xyz;

    const float3 pq = q - p;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 bc = c - b;
    const float3 ap = p - a;
    const float3 aq = q - a;

    const float d00 = dot(ab, ab);
    const float d01 = dot(ab, ac);
    const float d11 = dot(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : PXG_MAX_F32;

    const float3 n = normalize(cross(ab, ac));

    const float dist3 = dot(ap, n);
    const float sqDist3 = dist3 * dist3;
    const float dist4 = dot(aq, n);
    const float sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;

    float outT = 0.0f, outU = 0.0f, outV = 0.0f, outSq = 0.0f;
    bool resolved = false;

    // plane crossing inside the triangle -> distance 0
    if (dMul <= 0.0f)
    {
        const float nom = -dot(n, ap);
        const float denom = 1.0f / dot(n, pq);
        const float t0 = nom * denom;
        const float3 ip = p + pq * t0;
        const float3 v2 = ip - a;
        const float d20 = dot(v2, ab);
        const float d21 = dot(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom;
        const float w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (isValidBary(v00, w00))
        {
            outT = t0; outU = v00; outV = w00; outSq = 0.0f;
            resolved = true;
        }
    }

    if (!resolved)
    {
        // barycentric of p's and q's plane projections
        const float3 closestP31 = p - n * dist3;
        const float3 pV20 = closestP31 - a;
        const float pD20 = dot(pV20, ab);
        const float pD21 = dot(pV20, ac);
        const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
        const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
        const bool con0 = isValidBary(v0, w0);

        const float3 closestP41 = q - n * dist4;
        const float3 qV20 = closestP41 - a;
        const float qD20 = dot(qV20, ab);
        const float qD21 = dot(qV20, ac);
        const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
        const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
        const bool con1 = isValidBary(v1, w1);

        if (con0 && con1)
        {
            if (sqDist4 > sqDist3) { outT = 0.0f; outU = v0; outV = w0; outSq = sqDist3; }
            else                   { outT = 1.0f; outU = v1; outV = w1; outSq = sqDist4; }
        }
        else
        {
            float t00, t10, t01, t11, t02, t12;
            segSeg(p, pq, a, ab, t00, t10);
            segSeg(p, pq, b, bc, t01, t11);
            segSeg(p, pq, a, ac, t02, t12);

            const float3 cp00 = p + pq * t00; const float3 cp01 = a + ab * t10;
            float3 tv = cp00 - cp01; const float sqDist0 = dot(tv, tv);
            const float3 cp10 = p + pq * t01; const float3 cp11 = b + bc * t11;
            tv = cp10 - cp11; const float sqDist1 = dot(tv, tv);
            const float3 cp20 = p + pq * t02; const float3 cp21 = a + ac * t12;
            tv = cp20 - cp21; const float sqDist2 = dot(tv, tv);

            const float u10 = t10, v10 = 0.0f;         // edge ab
            const float u11 = 1.0f - t11, v11 = t11;   // edge bc
            const float u12 = 0.0f, v12 = t12;         // edge ac

            float sqDistPE, uEdge, vEdge, tSeg;
            if (sqDist1 > sqDist0 && sqDist2 > sqDist0) { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; }
            else if (sqDist2 > sqDist1)                 { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; }
            else                                        { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; }

            if (con0)
            {
                if (sqDistPE > sqDist3) { outT = 0.0f; outU = v0; outV = w0; outSq = sqDist3; }
                else                    { outT = tSeg; outU = uEdge; outV = vEdge; outSq = sqDistPE; }
            }
            else if (con1)
            {
                if (sqDistPE > sqDist4) { outT = 1.0f; outU = v1; outV = w1; outSq = sqDist4; }
                else                    { outT = tSeg; outU = uEdge; outV = vEdge; outSq = sqDistPE; }
            }
            else
            {
                outT = tSeg; outU = uEdge; outV = vEdge; outSq = sqDistPE;
            }
        }
    }

    gOut[idx] = float4(outSq, outT, outU, outV);
}
