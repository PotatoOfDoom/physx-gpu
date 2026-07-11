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

// npConvexFeature.slang
//
// Slang port of the convex-triangle feature-selection decision layer from
// gpunarrowphase/src/CUDA/convexTriangle.cuh: delayGenerateContact (:1402) and
// selectPolygonFeatureIndex (:1420).  These sit above the SAT primitives
// (satConvexFaceNormals tranche 65, satTriangleNormal tranche 64,
// satEdgeCrossDirection2 tranche 108 — convexMeshSAT is just their early-out
// composition and is not separately re-ported) and below polyClip.
//
// delayGenerateContact: if the SAT winner is a convex FACE (featureIndex is not
// EDGE_FEATURE_IDX / TRI_FEATURE_IDX) the pair is DEFERRED to the second pass
// when the winning face normal is more than 45 degrees from the triangle's
// negated local normal (minNormal.dot(-triNormal) < cos45).  Edge/tri winners
// are never deferred.
//
// selectPolygonFeatureIndex: when the winner is an EDGE/TRI feature it picks the
// convex polygon face(s) to clip against:
//   1. the face whose plane normal best aligns with minNormal (argmax of
//      minNormal.dot(planeN[i]));
//   2. then scans the hull edges for one whose SUMMED-face-normal direction
//      aligns with minNormal even better than that face (dp>0 and
//      dp^2/|edgeN|^2 > minProj^2) — if found, BOTH faces of that edge are
//      returned (the deeper as polyFaceIndex, the other as polyIndex2);
//   3. otherwise the best NEIGHBOUR face of the chosen face (the edge partner
//      with the largest dp^2) is returned as polyIndex2 (PhysX's anti-jitter
//      two-face face-contact path).
// A face winner returns featureIndex unchanged with polyIndex2 = 0xffffffff.
//
// Warp serialisation: the CUDA distributes the face/edge loops across a warp
// (index i owned by lane i % 32) and finishes each argmax with a
// warpReduction<MaxOpFloat> that breaks ties by LOWEST LANE.  The serial port
// reproduces this exactly by bucketing per-lane accumulators (strict-< keeps the
// lowest index within a lane) and taking the lowest-lane maximum across the 32
// buckets (the tranche-108 tie-break doctrine).
//
// Determinism: axis-aligned box plane normals + an axis-aligned minNormal make
// every dot product exact, and exact ties resolve identically on both sides via
// the lowest-lane rule; explicit dotX matches the CPU FMA contraction.

import pxgSlangCommon;

static const uint EDGE_FEATURE_IDX = 0x80000000u;
static const uint TRI_FEATURE_IDX = 0x7FffFFffu;
static const uint INVALID = 0xffffffffu;
static const float FI_MAX_F32 = 3.402823466e38f;
static const float COS45 = 0.707106781f;

struct FiParams
{
    uint numTests;
    uint maxPolys; // stride (in float4s) of each test's plane-normal block
    uint maxEdges; // stride (in uint2s) of each test's edge-face block
    uint pad0;
}

struct FiIn
{
    float4 minNormal;    // xyz; w unused
    float4 triLocNormal; // xyz triangle normal in convex space (delay gate)
    uint4 misc;          // x=featureIndex, y=nbPolygons, z=nbEdges, w unused
}

struct FiOut
{
    uint polyFaceIndex;
    uint polyIndex2;
    uint delay;
    uint pad0;
}

[[vk::push_constant]]
ConstantBuffer<FiParams> gParams;

RWStructuredBuffer<float4> gPlaneN; // binding 0: convex plane normals (maxPolys/test)
RWStructuredBuffer<uint2> gEdgeF;   // binding 1: per edge (f0,f1) face indices (maxEdges/test)
RWStructuredBuffer<FiIn> gIn;       // binding 2
RWStructuredBuffer<FiOut> gOut;     // binding 3

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

[shader("compute")]
[numthreads(64, 1, 1)]
void ctSelectFeature(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const FiIn e = gIn[t];
    const uint featureIndex = e.misc.x;
    const uint nbPolygons = e.misc.y;
    const uint nbEdges = e.misc.z;
    const float3 minNormal = e.minNormal.xyz;
    const uint pbase = t * gParams.maxPolys;
    const uint ebase = t * gParams.maxEdges;

    // ---- delayGenerateContact ----
    uint delay = 0u;
    if (featureIndex != EDGE_FEATURE_IDX && featureIndex != TRI_FEATURE_IDX)
    {
        const float3 triLocNormal = -e.triLocNormal.xyz;
        const float cosTheta = dotX(minNormal, triLocNormal);
        delay = (cosTheta < COS45) ? 1u : 0u;
    }

    // ---- selectPolygonFeatureIndex ----
    uint polyFaceIndex = featureIndex;
    uint polyIndex2 = INVALID;

    if (featureIndex == TRI_FEATURE_IDX || featureIndex == EDGE_FEATURE_IDX)
    {
        // 1) best convex face: argmax minNormal.dot(planeN[i]), lowest-lane tie
        float laneProj[32];
        uint laneFace[32];
        for (uint L = 0u; L < 32u; ++L) { laneProj[L] = -FI_MAX_F32; laneFace[L] = 0u; }
        for (uint i = 0u; i < nbPolygons; ++i)
        {
            const float proj = dotX(minNormal, gPlaneN[pbase + i].xyz);
            const uint L = i % 32u;
            if (laneProj[L] < proj) { laneProj[L] = proj; laneFace[L] = i; }
        }
        float minProj = laneProj[0];
        polyFaceIndex = laneFace[0];
        for (uint L = 1u; L < 32u; ++L)
            if (laneProj[L] > minProj) { minProj = laneProj[L]; polyFaceIndex = laneFace[L]; }

        // 2)/3) edge scan: closest edge (better than the face) + best neighbour
        float laneMaxDpSq[32], laneMaxDpSq2[32];
        uint laneClosestEdge[32], lanePolyIndex2[32];
        for (uint L = 0u; L < 32u; ++L)
        {
            laneMaxDpSq[L] = minProj * minProj;
            laneClosestEdge[L] = INVALID;
            laneMaxDpSq2[L] = -10.0f;
            lanePolyIndex2[L] = INVALID;
        }
        for (uint i = 0u; i < nbEdges; ++i)
        {
            const uint L = i % 32u;
            const uint2 fIdx = gEdgeF[ebase + i];
            const float3 nf0 = gPlaneN[pbase + fIdx.x].xyz;
            const float3 nf1 = gPlaneN[pbase + fIdx.y].xyz;
            const float3 edgeN = nf0 + nf1;
            const float enMagSq = dotX(edgeN, edgeN);
            const float dp = dotX(edgeN, minNormal);
            const float sqDp = dp * dp;

            if ((fIdx.x == polyFaceIndex || fIdx.y == polyFaceIndex) && sqDp > laneMaxDpSq2[L])
            {
                laneMaxDpSq2[L] = sqDp;
                lanePolyIndex2[L] = (fIdx.x == polyFaceIndex) ? fIdx.y : fIdx.x;
            }
            if (dp > 0.0f && sqDp > laneMaxDpSq[L] * enMagSq)
            {
                laneMaxDpSq[L] = sqDp / enMagSq;
                laneClosestEdge[L] = i;
            }
        }
        // reduce closest-edge (lowest-lane winner)
        float maxDpSq = laneMaxDpSq[0];
        uint closestEdge = laneClosestEdge[0];
        for (uint L = 1u; L < 32u; ++L)
            if (laneMaxDpSq[L] > maxDpSq) { maxDpSq = laneMaxDpSq[L]; closestEdge = laneClosestEdge[L]; }

        if (closestEdge != INVALID)
        {
            const uint2 fIdx = gEdgeF[ebase + closestEdge];
            const float dp0 = dotX(gPlaneN[pbase + fIdx.x].xyz, minNormal);
            const float dp1 = dotX(gPlaneN[pbase + fIdx.y].xyz, minNormal);
            if (dp0 > dp1) { polyFaceIndex = fIdx.x; polyIndex2 = fIdx.y; }
            else           { polyFaceIndex = fIdx.y; polyIndex2 = fIdx.x; }
        }
        else
        {
            // reduce best-neighbour (lowest-lane winner)
            float maxDpSq2 = laneMaxDpSq2[0];
            polyIndex2 = lanePolyIndex2[0];
            for (uint L = 1u; L < 32u; ++L)
                if (laneMaxDpSq2[L] > maxDpSq2) { maxDpSq2 = laneMaxDpSq2[L]; polyIndex2 = lanePolyIndex2[L]; }
        }
    }

    FiOut o;
    o.polyFaceIndex = polyFaceIndex;
    o.polyIndex2 = polyIndex2;
    o.delay = delay;
    o.pad0 = 0u;
    gOut[t] = o;
}
