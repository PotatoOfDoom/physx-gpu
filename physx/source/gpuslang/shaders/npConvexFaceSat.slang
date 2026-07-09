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

// npConvexFaceSat.slang
//
// Slang port of satConvexFaceNormals from
// gpunarrowphase/src/CUDA/convexTriangle.cuh: the convex-face-normal half of the
// convex-vs-triangle separating-axis test.  Each of the convex hull's face
// polygons contributes a candidate axis (its plane normal, transformed to shape
// space by vertex2ShapeNormalVector so it stays perpendicular under the hull's
// non-uniform scale).  Onto that axis the hull projects to [min0, max0] and the
// triangle to [min1, max1]; if any face separates the two beyond contactDist the
// pair is disjoint (return false), otherwise the deepest face (max min1-max0)
// becomes the running-max separation, its index the feature, and its normal the
// contact normal.  The CUDA warp argmax reduction is serialised here (a serial
// max keeps the first/lowest-index polygon on ties).
//
// vertex2ShapeNormalVector(n, scale, rot) = shape2Vertex(n) =
//   rotateInv( rotate(n) / scale ).
// getMinIndex(polygonDesc) is the low byte of the descriptor.

import pxgSlangCommon;

struct CfParams
{
    uint numTests;
    uint maxPoly;  // stride (in elements) of each test's per-polygon blocks
    uint maxVerts; // stride (in float4s) of each test's convex-vertex block
    uint pad;
}

struct CfIn
{
    float4 convexScale;    // xyz
    float4 convexScaleRot; // quaternion
    float4 tri0, tri1, tri2;
    float4 misc;           // x = nbPolygons, y = contactDist, z = seed separation
    float4 seedFaceNormal; // xyz
    uint seedFeature;
    uint p0, p1, p2;
}

struct CfOut
{
    float4 sepRet;     // (separation, returnedBool, -, -)
    float4 faceNormal; // xyz
    uint feature;
    uint q0, q1, q2;
}

[[vk::push_constant]]
ConstantBuffer<CfParams> gParams;

static const float CF_MAX_F32 = 3.402823466e38f;

RWStructuredBuffer<uint> gPolyDesc;  // binding 0  polygon descriptor, maxPoly per test
RWStructuredBuffer<float4> gPlanes;  // binding 1  plane (xyz normal, w planeDist), maxPoly per test
RWStructuredBuffer<float4> gVerts;   // binding 2  convex verts, maxVerts per test
RWStructuredBuffer<CfIn> gIn;        // binding 3
RWStructuredBuffer<CfOut> gOut;      // binding 4

// vertex2ShapeNormalVector == shape2Vertex(n, scale, rot)
float3 vertex2ShapeNormalVector(float3 n, float3 scale, float4 rot)
{
    float3 t = quatRotate(rot, n);
    t /= scale;
    return quatRotateInv(rot, t);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ctSatConvexFaceNormals(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CfIn e = gIn[t];

    const uint nbPolygons = uint(e.misc.x);
    const float contactDist = e.misc.y;
    const float3 scale = e.convexScale.xyz;
    const float4 rot = e.convexScaleRot;
    const float3 v0 = e.tri0.xyz, v1 = e.tri1.xyz, v2 = e.tri2.xyz;

    const uint polyBase = t * gParams.maxPoly;
    const uint vertBase = t * gParams.maxVerts;

    float _separation = -CF_MAX_F32;
    uint _feature = 0u;
    float3 _faceNormal = e.seedFaceNormal.xyz;
    bool disjoint = false;

    for (uint i = 0u; i < nbPolygons; ++i)
    {
        const uint polygonData = gPolyDesc[polyBase + i];
        const float4 planef4 = gPlanes[polyBase + i];
        const float3 minVertConv = gVerts[vertBase + (polygonData & 0xffu)].xyz;

        const float planeDist = planef4.w;
        const float3 vertexSpacePlaneNormal = planef4.xyz;

        float3 shapeSpacePlaneNormal = vertex2ShapeNormalVector(vertexSpacePlaneNormal, scale, rot);
        const float mag = length(shapeSpacePlaneNormal);
        const float invMagnitude = 1.0f / mag;
        shapeSpacePlaneNormal = shapeSpacePlaneNormal * invMagnitude; // normalize in place

        const float min0 = dot(shapeSpacePlaneNormal, minVertConv);
        const float max0 = -planeDist * invMagnitude;

        const float d0 = dot(v0, shapeSpacePlaneNormal);
        const float d1 = dot(v1, shapeSpacePlaneNormal);
        const float d2 = dot(v2, shapeSpacePlaneNormal);
        const float max1 = max(d0, max(d1, d2));
        const float min1 = min(d0, min(d1, d2));

        if ((min1 > max0 + contactDist) || (min0 > max1 + contactDist))
        {
            disjoint = true;
            break;
        }

        const float tempSeparation = min1 - max0;
        if (_separation < tempSeparation)
        {
            _separation = tempSeparation;
            _feature = i;
            _faceNormal = shapeSpacePlaneNormal;
        }
    }

    float separation = e.misc.z;
    float3 faceNormal = e.seedFaceNormal.xyz;
    uint feature = e.seedFeature;
    float ret = 1.0f;

    if (disjoint)
    {
        ret = 0.0f;
    }
    else if (separation < _separation)
    {
        faceNormal = _faceNormal;
        separation = _separation;
        feature = _feature;
    }

    CfOut o;
    o.sepRet = float4(separation, ret, 0.0f, 0.0f);
    o.faceNormal = float4(faceNormal, 0.0f);
    o.feature = feature;
    o.q0 = 0u; o.q1 = 0u; o.q2 = 0u;
    gOut[t] = o;
}
