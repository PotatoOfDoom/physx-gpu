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

// npTriMask.slang
//
// Slang port of the triangle barycentric / active-edge classification helpers
// from gpunarrowphase/src/CUDA/convexTriangle.cuh: barycentricCoordinates (the
// signed-area barycentric coordinates (v,w) of a point relative to a triangle),
// getTriangleActiveMask(v,w) (which Voronoi region the barycentric coordinates
// fall in, as a vertex/edge feature mask), getTriangleActiveMaskFromNormal (the
// active feature selected by a direction relative to the triangle's edges), and
// the getTriangleActiveMask(p, v0,v1,v2) point overload.
//
// The feature masks are ConvexTriIntermediateData::e{V0,V1,V2,E01,E02,E12} =
// {1..6} << 29.

struct TmParams { uint numTests; uint pad0, pad1, pad2; }

struct BaryIn { float4 p, a, b, c; }
struct NormIn { float4 v0, v1, v2, n; }
struct PointIn { float4 pa, v0, v1, v2; }

[[vk::push_constant]]
ConstantBuffer<TmParams> gParams;

static const uint M_V0  = 1u << 29;
static const uint M_V1  = 2u << 29;
static const uint M_V2  = 3u << 29;
static const uint M_E01 = 4u << 29;
static const uint M_E02 = 5u << 29;
static const uint M_E12 = 6u << 29;

RWStructuredBuffer<BaryIn> gBaryIn;   // binding 0
RWStructuredBuffer<float4> gBaryOut;  // binding 1  (v, w, 0, 0)
RWStructuredBuffer<float4> gVw;       // binding 2  (v, w) for the mask entry
RWStructuredBuffer<NormIn> gNorm;     // binding 3
RWStructuredBuffer<PointIn> gPoint;   // binding 4
RWStructuredBuffer<uint> gMaskOut;    // binding 5

void barycentricCoordinates(float3 p, float3 a, float3 b, float3 c, out float v, out float w)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 n = cross(ab, ac);

    const float3 ap = a - p;
    const float3 bp = b - p;
    const float3 cp = c - p;

    const float3 bCrossC = cross(bp, cp);
    const float3 cCrossA = cross(cp, ap);
    const float3 aCrossB = cross(ap, bp);

    const float va = dot(n, bCrossC);
    const float vb = dot(n, cCrossA);
    const float vc = dot(n, aCrossB);
    const float totalArea = va + vb + vc;
    const float denom = totalArea == 0.0f ? 0.0f : 1.0f / totalArea;
    v = vb * denom;
    w = vc * denom;
}

uint getTriangleActiveMask(float v, float w)
{
    uint mask = 0u;
    const float upperBound = 0.97f;
    const float u = 1.0f - v - w;

    if (v > upperBound)              mask = M_V1;
    else if (w > upperBound)         mask = M_V2;
    else if (u > upperBound)         mask = M_V0;
    else if ((v + u) > upperBound)   mask = M_E01;
    else if ((u + w) > upperBound)   mask = M_E02;
    else if ((w + v) > upperBound)   mask = M_E12;

    return mask;
}

// PxVec3::getNormalized (PxVec3.h:286-290): m = magnitudeSquared(); m > 0 ?
// v * PxRecipSqrt(m) : zero.  PxRecipSqrt is intrinsics::recipSqrt = 1.0f/sqrtf
// (PxWindowsMathIntrinsics.h:87), so it is a reciprocal of an exactly specified
// square root -- NOT GLSL.std.450 Normalize, which the Vulkan environment spec
// leaves several ULP of room and which is undefined for a zero vector.
float3 tmGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

uint getTriangleActiveMaskFromNormal(float3 v0, float3 v1, float3 v2, float3 n)
{
    // P9.47: convexTriangle.cuh:1602-1604 uses getNormalized(), not a Normalize
    // intrinsic.  The difference is both a last-bit one and a defined-ness one:
    // getNormalized() returns the ZERO vector for a degenerate edge, while
    // GLSL.std.450 Normalize of a zero vector is undefined.
    const float3 e01 = tmGetNormalized(v1 - v0);
    const float3 e02 = tmGetNormalized(v2 - v0);
    const float3 e12 = tmGetNormalized(v2 - v1);
    const float d01 = dot(e01, n);
    const float d02 = dot(e02, n);
    const float d12 = dot(e12, n);
    const float3 tn = tmGetNormalized(cross(e01, e02));   // :1608
    const float eps = 0.0174f; // around 89 degrees

    if (d01 < -eps && d02 < -eps)
        return M_V0;
    if (d01 > eps && d12 < -eps)
        return M_V1;
    if (d02 > eps && d12 > eps)
        return M_V2;
    if (dot(tmGetNormalized(cross(tn, e01)), n) < -eps)   // :1620
        return M_E01;
    if (dot(tmGetNormalized(cross(tn, e02)), n) > eps)    // :1623
        return M_E02;
    if (dot(tmGetNormalized(cross(tn, e12)), n) < -eps)   // :1626
        return M_E12;

    return 0u;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctBarycentric(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const BaryIn e = gBaryIn[i];
    float v, w;
    barycentricCoordinates(e.p.xyz, e.a.xyz, e.b.xyz, e.c.xyz, v, w);
    gBaryOut[i] = float4(v, w, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctActiveMaskVW(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const float4 vw = gVw[i];
    gMaskOut[i] = getTriangleActiveMask(vw.x, vw.y);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctActiveMaskFromNormal(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const NormIn e = gNorm[i];
    gMaskOut[i] = getTriangleActiveMaskFromNormal(e.v0.xyz, e.v1.xyz, e.v2.xyz, e.n.xyz);
}

// ---------------------------------------------------------------------------
// getTriangleActiveMask(p, v0,v1,v2): barycentric + region classification.
[shader("compute")]
[numthreads(256, 1, 1)]
void ctActiveMaskPoint(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const PointIn e = gPoint[i];
    float v, w;
    barycentricCoordinates(e.pa.xyz, e.v0.xyz, e.v1.xyz, e.v2.xyz, v, w);
    gMaskOut[i] = getTriangleActiveMask(v, w);
}
