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

// npConvexEdgeSat.slang
//
// Slang port of satEdgeCrossDirection2 (convexTriangle.cuh :575-696) — the
// convex-edge x triangle-edge separating-axis driver that completes
// convexMeshSAT above the already-ported satConvexFaceNormals (tranche 65),
// satTriangleNormal (tranche 64) and the support primitives (tranche 64).
//
// The CUDA distributes the numConvexEdges*3 (convex-edge, triangle-edge) pairs
// across a warp; for each pair it
//   - skips inactive triangle edges (isActiveEdge on the adjacency flag),
//   - GAUSS-MAP CULLS the pair unless the triangle normal faces at least one of
//     the convex edge's two adjacent faces (shapeNormal.dot(planeN[f]) < 0),
//   - builds the edge-cross axis v = convexEdge x triEdge (skipped if
//     degenerate, |v|^2 <= 1e-10), normalises it,
//   - projects both shapes onto it (supportTriangle / supportConvex), early-
//     outs the WHOLE test if they are separated beyond contactDist, else
//     advances the running-max separation with the +v / -v orientation,
// then warp-argmax-reduces the separation, and if it beats the seed sets the
// contact normal to the winning axis and the feature to EDGE_FEATURE_IDX.
//
// Serialisation: one thread walks all numConvexEdges*3 pairs.  The max over the
// candidate axes is order-independent, so the serial running-max is identical
// to the warp reduction; the boolean "any pair separated -> whole test fails"
// is likewise order-independent (once separated it stays separated), so an
// immediate early return matches __any_sync.  The seed separation/faceNormal/
// feature are only overwritten when an edge axis strictly beats the seed.
//
// The hull is passed pre-decoded (as the tranche-63/64/65 ports do): scaled
// vertices, plane normals, the per-edge (v0,v1) vertex table and (f0,f1) face
// table, and the vertex/edge counts — the hull-blob bit unpacking is engine
// glue and not re-ported here.
//
// Determinism: with perpendicular integer edges the cross product is axis-
// aligned (exact unit axis) and the integer/power-of-two vertex projections are
// bit-exact; explicit dotX/crossX match the CPU FMA contraction and the
// normalisation uses v*(1/sqrt(m)) (llvmpipe's rsqrt is not correctly rounded).

import pxgSlangCommon;

static const uint EDGE_FEATURE_IDX = 0x80000000u;
static const uint BOUNDARY = 0xffffffffu;
static const uint NONCONVEX_FLAG = 0x80000000u;
static const float ES_MAX_F32 = 3.402823466e38f;

struct EsParams
{
    uint numTests;
    uint maxVerts;  // stride (in float4s) of each test's vertex block
    uint maxEdges;  // stride (in uint2s) of each test's edge block
    uint maxPolys;  // stride (in float4s) of each test's plane-normal block
}

struct EsIn
{
    float4 triV0, triV1, triV2; // triangle local vertices
    float4 triNormal;           // triangleLocNormal (== shapeNormal); w unused
    float4 seedFaceNormal;      // xyz seed face normal
    float4 misc;                // x=nbVerts, y=nbEdges, z=contactDist, w=seedSeparation
    uint4 triAdj;               // x/y/z = triAdjTrisIdx (active-edge flags); w=seedFeature
}

struct EsOut
{
    float4 sepRet;    // x=separation, y=passed(1/0), z/w unused
    float4 faceNormal; // xyz winning axis
    uint feature;
    uint q0, q1, q2;
}

[[vk::push_constant]]
ConstantBuffer<EsParams> gParams;

RWStructuredBuffer<float4> gVerts;   // binding 0: convex scaled verts (maxVerts/test)
RWStructuredBuffer<float4> gPlaneN;  // binding 1: convex plane normals (maxPolys/test)
RWStructuredBuffer<uint2> gEdgeV;    // binding 2: per edge (v0,v1) vertex indices (maxEdges/test)
RWStructuredBuffer<uint2> gEdgeF;    // binding 3: per edge (f0,f1) face indices (maxEdges/test)
RWStructuredBuffer<EsIn> gIn;        // binding 4
RWStructuredBuffer<EsOut> gOut;      // binding 5

// explicit dot/cross so the FMA contraction matches the CPU reference exactly
float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossX(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float3 normalizedX(float3 v)
{
    const float m = dotX(v, v);
    return v * (1.0f / sqrt(m));
}
bool isActiveEdge(uint edgeIndex)
{
    return edgeIndex == BOUNDARY || (edgeIndex & NONCONVEX_FLAG) == 0u;
}

void supportConvex(uint vbase, uint nbVerts, float3 dir, out float minVal, out float maxVal)
{
    minVal = ES_MAX_F32;
    maxVal = -ES_MAX_F32;
    for (uint i = 0u; i < nbVerts; ++i)
    {
        const float d = dotX(gVerts[vbase + i].xyz, dir);
        minVal = min(d, minVal);
        maxVal = max(d, maxVal);
    }
}

void supportTriangle(float3 t0, float3 t1, float3 t2, float3 dir, out float minVal, out float maxVal)
{
    const float d0 = dotX(t0, dir), d1 = dotX(t1, dir), d2 = dotX(t2, dir);
    minVal = min(d0, min(d1, d2));
    maxVal = max(d0, max(d1, d2));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctEdgeCrossSat(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const EsIn e = gIn[t];
    const uint nbVerts = uint(e.misc.x);
    const uint nbEdges = uint(e.misc.y);
    const float contactDist = e.misc.z;
    const float seedSeparation = e.misc.w;

    const uint vbase = t * gParams.maxVerts;
    const uint pbase = t * gParams.maxPolys;
    const uint ebase = t * gParams.maxEdges;

    const float3 tri0 = e.triV0.xyz, tri1 = e.triV1.xyz, tri2 = e.triV2.xyz;
    const float3 triVerts[3] = { tri0, tri1, tri2 };
    const uint triAdj[3] = { e.triAdj.x, e.triAdj.y, e.triAdj.z };
    const float3 shapeNormal = e.triNormal.xyz;

    float3 faceNormalOut = e.seedFaceNormal.xyz;
    uint feature = e.triAdj.w;
    float separation = seedSeparation;

    float3 accFaceNormal = e.seedFaceNormal.xyz;
    float accSeparation = seedSeparation;
    bool passed = true;

    const uint total = nbEdges * 3u;
    for (uint index = 0u; index < total; ++index)
    {
        const uint vStartIndex = index % 3u;
        if (!isActiveEdge(triAdj[vStartIndex]))
            continue;
        const uint vEndIndex = (vStartIndex + 1u) % 3u;
        const uint convexEdgeIdx = index / 3u;

        const uint2 vIdx = gEdgeV[ebase + convexEdgeIdx];
        const uint2 fIdx = gEdgeF[ebase + convexEdgeIdx];

        const float3 nf0 = gPlaneN[pbase + fIdx.x].xyz;
        const float3 nf1 = gPlaneN[pbase + fIdx.y].xyz;
        if (!(dotX(shapeNormal, nf0) < 0.0f || dotX(shapeNormal, nf1) < 0.0f))
            continue;

        const float3 convexEdge = gVerts[vbase + vIdx.x].xyz - gVerts[vbase + vIdx.y].xyz;
        const float3 triEdge = triVerts[vEndIndex] - triVerts[vStartIndex];
        const float3 v = crossX(convexEdge, triEdge);
        const float magSquared = dotX(v, v);
        if (magSquared > 1e-10f)
        {
            const float3 axis = normalizedX(v);
            float minTri, maxTri, minConv, maxConv;
            supportTriangle(tri0, tri1, tri2, axis, minTri, maxTri);
            supportConvex(vbase, nbVerts, axis, minConv, maxConv);

            const float tempOverlap0 = minTri - maxConv;
            const float tempOverlap1 = minConv - maxTri;
            if (minConv > (maxTri + contactDist) || minTri > (maxConv + contactDist))
            {
                passed = false;
                break; // once separated the whole test fails (order-independent)
            }
            if (accSeparation < tempOverlap0)
            {
                accSeparation = tempOverlap0;
                accFaceNormal = axis;
            }
            if (accSeparation < tempOverlap1)
            {
                accSeparation = tempOverlap1;
                accFaceNormal = -axis;
            }
        }
    }

    EsOut o;
    if (!passed)
    {
        o.sepRet = float4(separation, 0.0f, 0.0f, 0.0f);
        o.faceNormal = float4(faceNormalOut, 0.0f);
        o.feature = feature;
        o.q0 = 0u; o.q1 = 0u; o.q2 = 0u;
        gOut[t] = o;
        return;
    }

    if (seedSeparation < accSeparation)
    {
        faceNormalOut = accFaceNormal;
        separation = accSeparation;
        feature = EDGE_FEATURE_IDX;
    }

    o.sepRet = float4(separation, 1.0f, 0.0f, 0.0f);
    o.faceNormal = float4(faceNormalOut, 0.0f);
    o.feature = feature;
    o.q0 = 0u; o.q1 = 0u; o.q2 = 0u;
    gOut[t] = o;
}
