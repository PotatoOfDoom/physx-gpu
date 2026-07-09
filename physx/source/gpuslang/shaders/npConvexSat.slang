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

// npConvexSat.slang
//
// Slang port of the convex-collision projection / separating-axis primitives
// from gpunarrowphase/src/CUDA/convexTriangle.cuh: supportConvex and
// supportTriangle (min/max projection of a point set onto a direction) and
// satTriangleNormal (the triangle-face-normal separating axis — project the
// convex hull onto the triangle normal, test for separation against the
// triangle's own projection inflated by contactDist, and otherwise advance the
// running-max separation with the triangle-face feature).
//
// The CUDA satTriangleNormal reduces the convex min/max across a warp; min/max
// over a set is order-independent, so the serial loop here is bit-identical.

struct CsParams
{
    uint numTests;
    uint maxVerts; // stride (in float4s) of each test's convex-vertex block
    uint pad0, pad1;
}

struct CsIn
{
    float4 dir;            // support direction (support entries)
    float4 triNormal;      // triangle normal n0 (sat entry)
    float4 misc;           // x = nbVerts, y = contactDist, z = seed separation
    float4 seedFaceNormal; // xyz
    uint seedFeature;
    uint p0, p1, p2;
}

struct CsOut
{
    float4 sepRet;     // support: (minVal, maxVal, -, -);  sat: (separation, returnedBool, -, -)
    float4 faceNormal; // xyz (sat)
    uint feature;      // sat
    uint q0, q1, q2;
}

[[vk::push_constant]]
ConstantBuffer<CsParams> gParams;

static const uint TRI_FEATURE_IDX = 0x7FffFFffu;
static const float CS_MAX_F32 = 3.402823466e38f;

RWStructuredBuffer<float4> gVerts;    // binding 0  convex verts, maxVerts per test (flat)
RWStructuredBuffer<float4> gTriVerts; // binding 1  three triangle verts per test (flat)
RWStructuredBuffer<CsIn> gIn;         // binding 2
RWStructuredBuffer<CsOut> gOut;       // binding 3

// supportConvex: min/max projection of nbVerts convex vertices onto dir.
void supportConvex(uint base, uint nbVerts, float3 dir, out float minVal, out float maxVal)
{
    minVal = CS_MAX_F32;
    maxVal = -CS_MAX_F32;
    for (uint i = 0u; i < nbVerts; ++i)
    {
        const float d = dot(gVerts[base + i].xyz, dir);
        minVal = min(d, minVal);
        maxVal = max(d, maxVal);
    }
}

// supportTriangle: min/max projection of the three triangle vertices onto dir.
void supportTriangle(uint triBase, float3 dir, out float minVal, out float maxVal)
{
    minVal = CS_MAX_F32;
    maxVal = -CS_MAX_F32;
    for (uint i = 0u; i < 3u; ++i)
    {
        const float d = dot(gTriVerts[triBase + i].xyz, dir);
        minVal = min(d, minVal);
        maxVal = max(d, maxVal);
    }
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctSupportConvex(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CsIn e = gIn[t];
    const uint nbVerts = uint(e.misc.x);
    float mn, mx;
    supportConvex(t * gParams.maxVerts, nbVerts, e.dir.xyz, mn, mx);
    CsOut o;
    o.sepRet = float4(mn, mx, 0.0f, 0.0f);
    o.faceNormal = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.feature = 0u;
    o.q0 = 0u; o.q1 = 0u; o.q2 = 0u;
    gOut[t] = o;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctSupportTriangle(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CsIn e = gIn[t];
    float mn, mx;
    supportTriangle(t * 3u, e.dir.xyz, mn, mx);
    CsOut o;
    o.sepRet = float4(mn, mx, 0.0f, 0.0f);
    o.faceNormal = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.feature = 0u;
    o.q0 = 0u; o.q1 = 0u; o.q2 = 0u;
    gOut[t] = o;
}

// ---------------------------------------------------------------------------
// ctSatTriangleNormal: mirrors satTriangleNormal.
[shader("compute")]
[numthreads(256, 1, 1)]
void ctSatTriangleNormal(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CsIn e = gIn[t];
    const uint nbVerts = uint(e.misc.x);
    const float contactDist = e.misc.y;

    const float3 n0 = e.triNormal.xyz;
    const float triProj = dot(n0, gTriVerts[t * 3u].xyz);

    float min1, max1;
    supportConvex(t * gParams.maxVerts, nbVerts, n0, min1, max1);

    float separation = e.misc.z;
    float3 faceNormal = e.seedFaceNormal.xyz;
    uint feature = e.seedFeature;
    float ret = 1.0f;

    const bool isSeparated = (min1 > triProj + contactDist) || (triProj > max1 + contactDist);
    if (isSeparated)
    {
        ret = 0.0f;
    }
    else
    {
        const float _separation = min1 - triProj;
        if (separation < _separation)
        {
            faceNormal = -n0;
            separation = _separation;
            feature = TRI_FEATURE_IDX;
        }
    }

    CsOut o;
    o.sepRet = float4(separation, ret, 0.0f, 0.0f);
    o.faceNormal = float4(faceNormal, 0.0f);
    o.feature = feature;
    o.q0 = 0u; o.q1 = 0u; o.q2 = 0u;
    gOut[t] = o;
}
