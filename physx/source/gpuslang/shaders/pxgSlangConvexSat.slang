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

// pxgSlangConvexSat.slang
//
// The first two stages of the convex-vs-triangle separating-axis test —
// `satConvexFaceNormals` (convexTriangle.cuh:217-301) and `satTriangleNormal`
// (:303-348) — plus the vertex-scaling prologue that feeds them
// (convexTriangle.cuh:1656-1668).  The third stage, `satEdgeCrossDirection2`
// (:575), is a separate tranche.
//
// ONE WARP PER GROUP, [numthreads(32,1,1)].  Same rule as pxgSlangPolyClip and
// for a sharper reason: `convexMeshSAT` (:867) returns early after each stage,
// and its single `__syncwarp()` sits BEFORE the first early return so that all 32
// lanes reach it whatever the outcome.  Those exits are warp-uniform but not
// group-uniform, and the CUDA block holds two warps on two different pairs.
//
// The warp primitives come from pxgSlangWarpInc rather than being re-invented:
// `pxgWarpReduceMaxWinnerF` already reproduces `warpReduction<MaxOpFloat>(...,
// winnerLane)` including its tie-break, which is the LOWEST lane whose original
// input equals the reduced maximum (reduction.cuh:209-215).  That tie-break is
// observable: the lane->polygon mapping is `i = lane; i += 32`, so two polygons
// with equal separation are resolved by which one a lower lane owned.
//
// THREE TRANSCRIPTION TRAPS, all from the CUDA's use of PxVec3::normalize():
//
//   1. `PxVec3::normalize()` returns the OLD magnitude and divides in place ONLY
//      when that magnitude is > 0 (PxVec3.h:285-300).  So for a zero-length
//      scaled plane normal the vector is left UNCHANGED and `invMagnitude =
//      1.0f / 0.0f` is +inf, which then makes `max0 = -planeDist * inf`.  A port
//      that used `normalize(v)` and `1.0f / length(v)` would produce a NaN
//      normal instead.  Reproduced literally.
//   2. `getNormalized()` (used for the triangle normal upstream) returns the ZERO
//      VECTOR for a degenerate triangle, not NaN.  The triangle normal arrives
//      here already computed, so this module only has to not assume it is unit.
//   3. `min0` is taken against the NORMALIZED normal but `max0` comes from the
//      unnormalised plane distance times `invMagnitude` — the two are only
//      consistent because of that division.  Swapping the order changes nothing
//      algebraically and everything numerically.
//
// THE PER-LANE `break` IS A PURE OPTIMISATION, and the CUDA's own reasoning about
// it does not hold.  It is tempting to argue that turning it into a `continue`
// would let a later polygon owned by the same lane win the argmax — the mutation
// sweep says otherwise, and the reason is structural: if ANY lane sets `disjoint`
// the function returns false, so the argmax and the plane arrays it filled are
// discarded wholesale.  The break therefore only saves work.  It is kept because
// this is a transcription, and it is a divergent break, which is safe here only
// because the loop body contains no barrier — the first one is in the
// `pxgAnySync` that every lane reaches afterwards.

import pxgSlangNpAbi;

#include "pxgSlangWarpInc.slang"
// Re-exported so a consumer that already imports this module can decode a hull
// descriptor without a second import — the P8.25 precedent, where
// pxgSlangClothMidphase re-exports pxgSlangBv32 for the same reason.
__exported import pxgSlangConvexHull;

public static const uint CS_MAX_VERTS_POLYS = 64u;   // CONVEX_MAX_VERTICES_POLYGONS
public static const uint CS_TRI_FEATURE_IDX = 0x7FFFFFFFu;
static const float CS_MAX_F32 = 3.4028234663852886e+38f;

// convexScaledVerts / convexPlaneN / convexPlaneD from ConvexScratch, as separate
// component runs so a lane can index any element.
groupshared float gsCsVertX[CS_MAX_VERTS_POLYS];
groupshared float gsCsVertY[CS_MAX_VERTS_POLYS];
groupshared float gsCsVertZ[CS_MAX_VERTS_POLYS];
groupshared float gsCsPlaneNX[CS_MAX_VERTS_POLYS];
groupshared float gsCsPlaneNY[CS_MAX_VERTS_POLYS];
groupshared float gsCsPlaneNZ[CS_MAX_VERTS_POLYS];
groupshared float gsCsPlaneD[CS_MAX_VERTS_POLYS];

public float3 csScaledVert(uint i) { return float3(gsCsVertX[i], gsCsVertY[i], gsCsVertZ[i]); }
public float3 csPlaneN(uint i) { return float3(gsCsPlaneNX[i], gsCsPlaneNY[i], gsCsPlaneNZ[i]); }
public float csPlaneD(uint i) { return gsCsPlaneD[i]; }

// cudaNpCommon.h:301 — vertex2ShapeNormalVector IS shape2Vertex, which divides.
float3 csVertex2Shape(float3 v, float3 scale, float4 rot)
{
    return npAbiQuatRotateInv(rot, npAbiQuatRotate(rot, v) * scale);
}
float3 csShape2Vertex(float3 v, float3 scale, float4 rot)
{
    return npAbiQuatRotateInv(rot, npAbiQuatRotate(rot, v) / scale);
}

// PxVec3::normalize(): returns the magnitude, divides in place only when it is
// positive, and leaves the vector untouched otherwise.
float csNormalizeInPlace(inout float3 v)
{
    const float m = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f)
        v = v / m;
    return m;
}

// ---------------------------------------------------------------------------
// the prologue: convexScaledVerts[i] = vertex2Shape(hullVertex[i])
// (convexTriangle.cuh:1656-1668).  Ends with the barrier the CUDA's __syncwarp()
// at :1668 provides, so the caller can read the array immediately.
// ---------------------------------------------------------------------------
public void csScaleVerts(uint lane, uint64_t hull, uint hullDesc, float3 scale, float4 scaleRot)
{
    const uint nbVerts = chNbVerts(hullDesc);
    for (uint i = lane; i < nbVerts; i += PXG_WARP_SIZE)
    {
        const float3 v = csVertex2Shape(chVertex(hull, i).xyz, scale, scaleRot);
        gsCsVertX[i] = v.x;
        gsCsVertY[i] = v.y;
        gsCsVertZ[i] = v.z;
    }
    GroupMemoryBarrierWithGroupSync();
}

// ---------------------------------------------------------------------------
// satConvexFaceNormals (convexTriangle.cuh:217)
// ---------------------------------------------------------------------------
public bool csSatConvexFaceNormals(uint lane, uint64_t hull, uint hullDesc,
                                   float3 scale, float4 scaleRot, float contactDist,
                                   float3 t0, float3 t1, float3 t2,
                                   inout uint feature, inout float3 faceNormal,
                                   inout float separation)
{
    float _separation = -CS_MAX_F32;
    uint _feature = 0u;
    float3 _faceNormal = faceNormal;

    const uint nbPolygons0 = chNbPolygons(hullDesc);

    bool disjoint = false;

    for (uint i = lane; i < nbPolygons0; i += PXG_WARP_SIZE)
    {
        const uint polygonData = chPolyDesc(hull, hullDesc, i);
        const float4 polygonf4 = chPlane(hull, hullDesc, i);
        const float3 minVertConv = csScaledVert(chPolyMinIndex(polygonData));

        const float planeDist = polygonf4.w;
        const float3 vertexSpacePlaneNormal = polygonf4.xyz;

        float3 shapeSpacePlaneNormal = csShape2Vertex(vertexSpacePlaneNormal, scale, scaleRot);

        const float invMagnitude = 1.0f / csNormalizeInPlace(shapeSpacePlaneNormal);

        const float min0 = dot(shapeSpacePlaneNormal, minVertConv);
        const float max0 = -planeDist * invMagnitude;

        const float d0 = dot(t0, shapeSpacePlaneNormal);
        const float d1 = dot(t1, shapeSpacePlaneNormal);
        const float d2 = dot(t2, shapeSpacePlaneNormal);

        const float max1 = max(d0, max(d1, d2));
        const float min1 = min(d0, min(d1, d2));

        if ((min1 > max0 + contactDist) || (min0 > max1 + contactDist))
        {
            disjoint = true;
            break;
        }

        gsCsPlaneNX[i] = shapeSpacePlaneNormal.x;
        gsCsPlaneNY[i] = shapeSpacePlaneNormal.y;
        gsCsPlaneNZ[i] = shapeSpacePlaneNormal.z;
        gsCsPlaneD[i] = -max0;

        const float tempSeparation = min1 - max0;

        if ((i < nbPolygons0) && (_separation < tempSeparation))
        {
            _separation = tempSeparation;
            _feature = i;
            _faceNormal = shapeSpacePlaneNormal;
        }
    }

    if (pxgAnySync(PXG_FULL_MASK, lane, disjoint))
        return false;

    uint winnerLane;
    _separation = pxgWarpReduceMaxWinnerF(PXG_FULL_MASK, lane, _separation, 5u, winnerLane);

    _feature = pxgShflSyncU(PXG_FULL_MASK, lane, _feature, int(winnerLane), PXG_WARP_SIZE);

    // The shuffle sits INSIDE the commit test in the CUDA too.  That is safe only
    // because `separation` and `_separation` are both warp-uniform here, so every
    // lane takes the same branch and reaches the same barriers.
    if (separation < _separation)
    {
        _faceNormal = float3(
            pxgShflSyncF(PXG_FULL_MASK, lane, _faceNormal.x, int(winnerLane), PXG_WARP_SIZE),
            pxgShflSyncF(PXG_FULL_MASK, lane, _faceNormal.y, int(winnerLane), PXG_WARP_SIZE),
            pxgShflSyncF(PXG_FULL_MASK, lane, _faceNormal.z, int(winnerLane), PXG_WARP_SIZE));

        faceNormal = _faceNormal;
        separation = _separation;
        feature = _feature;
    }

    return true;
}

// ---------------------------------------------------------------------------
// satTriangleNormal (convexTriangle.cuh:303)
//
// `_faceNormal` exists in the CUDA and is never used — the commit writes `-n0`.
// Not reproduced; a dead local cannot move an output bit.
// ---------------------------------------------------------------------------
public bool csSatTriangleNormal(uint lane, uint hullDesc, float contactDist,
                                float3 triangleLocNormal, float3 t0,
                                inout uint feature, inout float3 faceNormal,
                                inout float separation)
{
    const uint nbConvexVertices = chNbVerts(hullDesc);

    const float3 n0 = triangleLocNormal;
    const float triProj = dot(n0, t0);

    float min1 = CS_MAX_F32;
    float max1 = -CS_MAX_F32;

    for (uint i = lane; i < nbConvexVertices; i += PXG_WARP_SIZE)
    {
        const float dist = dot(csScaledVert(i), n0);
        max1 = max(dist, max1);
        min1 = min(dist, min1);
    }

    min1 = pxgWarpReduceMinF(PXG_FULL_MASK, lane, min1, 5u);
    max1 = pxgWarpReduceMaxF(PXG_FULL_MASK, lane, max1, 5u);

    if ((min1 > triProj + contactDist) || (triProj > max1 + contactDist))
        return false;

    const float _separation = min1 - triProj;

    if (separation < _separation)
    {
        faceNormal = -n0;
        separation = _separation;
        feature = CS_TRI_FEATURE_IDX;
    }

    return true;
}
