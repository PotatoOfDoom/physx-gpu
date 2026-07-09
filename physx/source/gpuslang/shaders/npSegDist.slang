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

// npSegDist.slang
//
// Slang port of two closest-point distance primitives shared across the
// narrow-phase collision routines: distancePointSegmentSquared (from
// sphereCollision.cuh — the squared distance from a point to a segment plus the
// clamped closest-point parameter) and distanceSegmentSegmentSquared (from
// distanceSegmentSegment.cuh — the closest-point parameters (s,t) of two
// segments, clamped into [0,1] with the parallel-segment fallback).

struct SdParams { uint numTests; uint pad0, pad1, pad2; }

struct PsIn { float4 a, b, p; }     // segment [a,b], query point p
struct PsOut { float4 r; }          // x = squared distance, y = param
struct SsIn { float4 p1, d1, p2, d2; } // segment1 = p1 + s*d1, segment2 = p2 + t*d2
struct SsOut { float4 r; }          // x = s, y = t

[[vk::push_constant]]
ConstantBuffer<SdParams> gParams;

RWStructuredBuffer<PsIn> gPs;    // binding 0
RWStructuredBuffer<PsOut> gPsOut; // binding 1
RWStructuredBuffer<SsIn> gSs;    // binding 2
RWStructuredBuffer<SsOut> gSsOut; // binding 3

// distancePointSegmentSquared: squared distance from p to segment [a,b]; param
// is the clamped closest-point parameter along a->b.
float distancePointSegmentSquared(float3 a, float3 b, float3 p, out float param)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    const float tValue = clamp(nom / denom, 0.0f, 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    const float3 v = ap - ab * t;
    param = t;
    return dot(v, v);
}

// distanceSegmentSegmentSquared: closest-point parameters (s,t) of the two
// segments (p1+s*d1) and (p2+t*d2), clamped to [0,1].
void distanceSegmentSegmentSquared(float3 p1, float3 d1, float3 p2, float3 d2, out float s, out float t)
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
[shader("compute")]
[numthreads(256, 1, 1)]
void sdPointSegment(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const PsIn e = gPs[i];
    float param;
    const float sq = distancePointSegmentSquared(e.a.xyz, e.b.xyz, e.p.xyz, param);
    PsOut o;
    o.r = float4(sq, param, 0.0f, 0.0f);
    gPsOut[i] = o;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sdSegmentSegment(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SsIn e = gSs[i];
    float s, t;
    distanceSegmentSegmentSquared(e.p1.xyz, e.d1.xyz, e.p2.xyz, e.d2.xyz, s, t);
    SsOut o;
    o.r = float4(s, t, 0.0f, 0.0f);
    gSsOut[i] = o;
}
