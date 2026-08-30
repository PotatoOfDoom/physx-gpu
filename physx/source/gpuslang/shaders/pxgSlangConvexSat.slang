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

static const uint CS_WARP = PXG_WARP_SIZE;
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
    // P9.36: convexTriangle.cuh:246 reaches this through
    // `shapeSpacePlaneNormal.normalize()`, i.e. operator/= (PxVec3.h:260) -- ONE
    // reciprocal and three multiplies.
    const float m = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f)
        v = v * (1.0f / m);
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

// ---------------------------------------------------------------------------
// satEdgeCrossDirection2 (convexTriangle.cuh:575)
//
// The edge-cross axes: for every (convex edge, triangle edge) pair, the cross
// product is a candidate separating axis.  Runs LAST, and it is the only stage
// that READS what stage 1 wrote — `convexPlaneN` is the gate.
//
// FOUR THINGS THE TRANSCRIPTION KEEPS EXACTLY:
//   1. `_separation` starts from the RUNNING BEST, not -PX_MAX_F32, so an edge
//      axis only wins if it strictly beats the face and triangle stages.
//   2. The two hull side-tables are read with the CUDA's little-endian trick: the
//      edge's two vertex indices as one uint, its two face indices as one ushort.
//      Both runs are 4-aligned from the blob base, so the packed loads are legal.
//   3. `magSquared` and `dp` are computed and never used in the CUDA, and
//      `convexScale`/`convexScaleRot` are loaded and never used.  Omitted: dead
//      code cannot move an output bit.  `shapeNormal` is the RAW triangle normal —
//      the shape2Vertex version sits commented out above it at :595.
//   4. The `__any_sync` early-out is INSIDE the iteration loop, so a warp that
//      finds a separating axis leaves before scanning the rest.  The loop bound is
//      warp-uniform, so the barrier inside it is too.
//
// The edge-parallel guard is `v.magnitudeSquared() > 1e-10f`, recomputed rather
// than reusing `magSquared` — same value, kept as written.
// ---------------------------------------------------------------------------

static const uint CS_BOUNDARY = 0xffffffffu;
static const uint CS_NONCONVEX_FLAG = 0x80000000u;
public static const uint CS_EDGE_FEATURE_IDX = 0x80000000u;

// triangle.cuh:556
bool csIsActiveEdge(uint edgeIndex)
{
    return edgeIndex == CS_BOUNDARY || (edgeIndex & CS_NONCONVEX_FLAG) == 0u;
}

// facesByEdges8 is byte-addressed but 4-aligned at its base, so a pair of face
// indices is one aligned ushort.
uint csUshortAt(uint64_t base, uint shortIndex)
{
    const uint64_t addr = base + uint64_t(2u * shortIndex);
    const uint64_t aligned = (addr / 4UL) * 4UL;
    const uint word = Ptr<uint>(aligned)[0];
    return (word >> (8u * uint(addr - aligned))) & 0xffffu;
}

void csSupportTriangle(float3 t0, float3 t1, float3 t2, float3 dir, out float minVal, out float maxVal)
{
    const float d0 = dot(t0, dir), d1 = dot(t1, dir), d2 = dot(t2, dir);
    minVal = min(d0, min(d1, d2));
    maxVal = max(d0, max(d1, d2));
}

void csSupportConvex(uint nbVerts, float3 dir, out float minVal, out float maxVal)
{
    minVal = CS_MAX_F32;
    maxVal = -CS_MAX_F32;
    for (uint i = 0u; i < nbVerts; ++i)
    {
        const float d = dot(csScaledVert(i), dir);
        minVal = min(d, minVal);
        maxVal = max(d, maxVal);
    }
}

public bool csSatEdgeCrossDirection2(uint lane, uint64_t vbeBase, uint64_t fbeBase,
                                     uint nbEdges, uint nbVerts, float contactDist,
                                     uint adj0, uint adj1, uint adj2,
                                     float3 t0, float3 t1, float3 t2, float3 triangleLocNormal,
                                     inout uint feature, inout float3 faceNormal,
                                     inout float separation)
{
    float3 _faceNormal = faceNormal;
    float _separation = separation;

    const float3 shapeNormal = triangleLocNormal;

    bool isPassed = true;
    for (uint it = 0u; it < nbEdges * 3u; it += CS_WARP)
    {
        const uint index = it + lane;

        if (index < nbEdges * 3u)
        {
            const uint vStartIndex = index % 3u;
            const uint adj = (vStartIndex == 0u) ? adj0 : ((vStartIndex == 1u) ? adj1 : adj2);

            if (csIsActiveEdge(adj))
            {
                const uint vEndIndex = (vStartIndex + 1u) % 3u;
                const uint convexEdgeIdx = index / 3u;

                const uint vIndices = Ptr<uint>(vbeBase)[convexEdgeIdx];
                const uint v0idx1 = vIndices & 0xffffu;
                const uint v1idx1 = (vIndices >> 16) & 0xffffu;

                const uint fIndices = csUshortAt(fbeBase, convexEdgeIdx);
                const uint f10 = fIndices & 0xffu;
                const uint f11 = (fIndices >> 8) & 0xffu;

                if (dot(shapeNormal, csPlaneN(f10)) < 0.0f || dot(shapeNormal, csPlaneN(f11)) < 0.0f)
                {
                    const float3 convexEdge = csScaledVert(v0idx1) - csScaledVert(v1idx1);

                    const float3 tS = (vStartIndex == 0u) ? t0 : ((vStartIndex == 1u) ? t1 : t2);
                    const float3 tE = (vEndIndex == 0u) ? t0 : ((vEndIndex == 1u) ? t1 : t2);
                    const float3 triEdge = tE - tS;
                    const float3 v = cross(convexEdge, triEdge);

                    if (dot(v, v) > 1e-10f)
                    {
                        // P9.36: convexTriangle.cuh:639 is `v.getNormalized()`, which is
                        // `v * PxRecipSqrt(magnitudeSquared())` (PxVec3.h:286) and
                        // PxRecipSqrt is 1.0f/sqrtf -- a multiply, never a divide.
                        const float m = sqrt(dot(v, v));
                        const float3 shapeSpaceV = (m > 0.0f) ? (v * (1.0f / m)) : v;

                        float minTri, maxTri, minConv, maxConv;
                        csSupportTriangle(t0, t1, t2, shapeSpaceV, minTri, maxTri);
                        csSupportConvex(nbVerts, shapeSpaceV, minConv, maxConv);

                        const float tempOverlap0 = minTri - maxConv;
                        const float tempOverlap1 = minConv - maxTri;

                        if (minConv > (maxTri + contactDist) || minTri > (maxConv + contactDist))
                        {
                            isPassed = false;
                        }
                        else
                        {
                            if (_separation < tempOverlap0)
                            {
                                _separation = tempOverlap0;
                                _faceNormal = shapeSpaceV;
                            }
                            if (_separation < tempOverlap1)
                            {
                                _separation = tempOverlap1;
                                _faceNormal = -shapeSpaceV;
                            }
                        }
                    }
                }
            }
        }

        if (pxgAnySync(PXG_FULL_MASK, lane, !isPassed))
            return false;
    }

    uint winnerLane;
    _separation = pxgWarpReduceMaxWinnerF(PXG_FULL_MASK, lane, _separation, 5u, winnerLane);

    if (separation < _separation)
    {
        const float nx = pxgShflSyncF(PXG_FULL_MASK, lane, _faceNormal.x, int(winnerLane), CS_WARP);
        const float ny = pxgShflSyncF(PXG_FULL_MASK, lane, _faceNormal.y, int(winnerLane), CS_WARP);
        const float nz = pxgShflSyncF(PXG_FULL_MASK, lane, _faceNormal.z, int(winnerLane), CS_WARP);
        faceNormal = float3(nx, ny, nz);
        separation = _separation;
        feature = CS_EDGE_FEATURE_IDX;
    }

    return true;
}

// ---------------------------------------------------------------------------
// convexMeshSAT (convexTriangle.cuh:868), doEdgeEdge = true
//
// The three-stage driver.  Two things about it are load-bearing and neither is
// obvious from the call sequence:
//
//   1. THE BARRIER SITS BEFORE THE FIRST EARLY RETURN.  `satConvexFaceNormals`
//      writes `convexPlaneN/D`, and the CUDA's `__syncwarp()` at :877 is placed
//      ahead of the `if (!passedSAT) return false`.  `passedSAT` is warp-uniform,
//      so every lane reaches that barrier whatever stage 1 decided — moving it
//      after the return would put a barrier in divergent control flow, which on
//      llvmpipe hangs rather than errors (the P8.38 rule).
//   2. The three seeds are the CALLER's (:1649-1651) and they are folded in here
//      because `convexTriangleContactGen` is the only caller and it always
//      supplies exactly these.  `minNormal` is seeded with +PX_MAX_REAL in ALL
//      THREE components, not with a zero vector.
// ---------------------------------------------------------------------------
public bool csConvexMeshSAT(uint lane, uint64_t hull, uint hullDesc, float3 scale, float4 scaleRot,
                            float contactDist, float3 t0, float3 t1, float3 t2, float3 triangleLocNormal,
                            uint adj0, uint adj1, uint adj2,
                            out uint featureIndex, out float separation, out float3 minNormal)
{
    featureIndex = 0xffffffffu;
    separation = -CS_MAX_F32;
    minNormal = float3(CS_MAX_F32, CS_MAX_F32, CS_MAX_F32);

    bool passedSAT = csSatConvexFaceNormals(lane, hull, hullDesc, scale, scaleRot, contactDist,
                                            t0, t1, t2, featureIndex, minNormal, separation);

    GroupMemoryBarrierWithGroupSync();

    if (!passedSAT)
        return false;

    passedSAT = csSatTriangleNormal(lane, hullDesc, contactDist, triangleLocNormal, t0,
                                    featureIndex, minNormal, separation);

    if (!passedSAT)
        return false;

    return csSatEdgeCrossDirection2(lane, chVerticesByEdges16(hull, hullDesc),
                                    chFacesByEdges8(hull, hullDesc),
                                    chNbEdges(hullDesc), chNbVerts(hullDesc), contactDist,
                                    adj0, adj1, adj2, t0, t1, t2, triangleLocNormal,
                                    featureIndex, minNormal, separation);
}

// ---------------------------------------------------------------------------
// delayGenerateContact (convexTriangle.cuh:1402)
//
// Whether the pair goes to the second-pass buffer instead of generating contacts
// now.  Only a CONVEX FACE axis can be delayed: if the winner was the triangle
// normal or an edge cross, the answer is false whatever the geometry.  The test
// is against the NEGATED triangle normal, and 0.707106781f is the literal in the
// CUDA — not `sqrtf(0.5f)`, which differs in the last bit.
// ---------------------------------------------------------------------------
public bool csDelayGenerateContact(float3 minNormal, float3 triangleLocNormal, uint featureIndex)
{
    if (featureIndex != CS_EDGE_FEATURE_IDX && featureIndex != CS_TRI_FEATURE_IDX)
    {
        const float3 triLocNormal = -triangleLocNormal;
        const float cosTheta = dot(minNormal, triLocNormal);
        const float threshold = 0.707106781f;   // about 45 degrees
        return cosTheta < threshold;
    }

    return false;
}

// ---------------------------------------------------------------------------
// selectPolygonFeatureIndex (convexTriangle.cuh:1420)
//
// Turns the SAT winner into the ONE OR TWO convex faces the clipper runs against.
// If the winner was already a face index there is nothing to do and `polyIndex2`
// keeps the caller's 0xFFFFFFFF.  Otherwise three warp reductions run, and the
// ways they can be got wrong are all quiet:
//
//   * `maxDpSq` is SEEDED WITH `minProj * minProj` — the REDUCED, warp-uniform
//     best face projection, not the lane's own.  An edge only displaces the face
//     if it beats the face's own alignment, so the seed is the comparison's
//     floor and not merely an initial value.
//   * the edge test is `sqDp > maxDpSq * enMagSq` but the STORE is
//     `sqDp / enMagSq`.  Multiplying on one side and dividing on the other is
//     how the CUDA compares normalised alignments without normalising `edgeN`,
//     and the two spellings are not interchangeable once `enMagSq != 1`.
//   * `maxDpSq2` starts at -10.0f, which is below any squared dot product, so
//     the FIRST neighbour of `polyFaceIndex` a lane meets always takes it.
//   * the second reduction is skipped entirely when an edge won, because the two
//     faces then come from the edge itself.  `closestEdge` is warp-uniform after
//     its shuffle, so that branch is uniform and the barrier inside the reduction
//     in the other arm is safe.
//
// `polyIndex2` is `inout`: the caller seeds it with 0xFFFFFFFF (:1698) and the
// no-edge arm broadcasts the winner's value, which may still be that seed if no
// lane ever saw a neighbour.
// ---------------------------------------------------------------------------
public uint csSelectPolygonFeatureIndex(uint lane, uint64_t hull, uint hullDesc, float3 minNormal,
                                        uint featureIndex, inout uint polyIndex2)
{
    uint polyFaceIndex = featureIndex;

    if (featureIndex == CS_TRI_FEATURE_IDX || featureIndex == CS_EDGE_FEATURE_IDX)
    {
        const uint nbPolygons0 = chNbPolygons(hullDesc);

        float minProj = -CS_MAX_F32;
        uint bestConvexFaceIndex = 0u;

        for (uint i = lane; i < nbPolygons0; i += CS_WARP)
        {
            const float proj = dot(minNormal, csPlaneN(i));
            if (minProj < proj)
            {
                minProj = proj;
                bestConvexFaceIndex = i;
            }
        }

        uint winnerLane;
        minProj = pxgWarpReduceMaxWinnerF(PXG_FULL_MASK, lane, minProj, 5u, winnerLane);
        polyFaceIndex = pxgShflSyncU(PXG_FULL_MASK, lane, bestConvexFaceIndex, int(winnerLane), CS_WARP);

        const uint nbEdges = chNbEdges(hullDesc);
        const uint64_t fbeBase = chFacesByEdges8(hull, hullDesc);

        uint closestEdge = 0xFFFFFFFFu;
        float maxDpSq = minProj * minProj;
        float maxDpSq2 = -10.0f;

        for (uint i = lane; i < nbEdges; i += CS_WARP)
        {
            const uint fIndices = csUshortAt(fbeBase, i);
            const uint f0 = fIndices & 0xffu;
            const uint f1 = (fIndices >> 8) & 0xffu;

            const float3 edgeN = csPlaneN(f0) + csPlaneN(f1);
            const float enMagSq = dot(edgeN, edgeN);
            const float dp = dot(edgeN, minNormal);
            const float sqDp = dp * dp;

            if ((f0 == polyFaceIndex || f1 == polyFaceIndex) && (sqDp > maxDpSq2))
            {
                maxDpSq2 = sqDp;
                polyIndex2 = (f0 == polyFaceIndex) ? f1 : f0;
            }

            if (dp > 0.0f && sqDp > (maxDpSq * enMagSq))
            {
                maxDpSq = sqDp / enMagSq;
                closestEdge = i;
            }
        }

        maxDpSq = pxgWarpReduceMaxWinnerF(PXG_FULL_MASK, lane, maxDpSq, 5u, winnerLane);
        closestEdge = pxgShflSyncU(PXG_FULL_MASK, lane, closestEdge, int(winnerLane), CS_WARP);

        if (closestEdge != 0xFFFFFFFFu)
        {
            const uint fIndices = csUshortAt(fbeBase, closestEdge);
            const uint f0 = fIndices & 0xffu;
            const uint f1 = (fIndices >> 8) & 0xffu;

            const float dp0 = dot(csPlaneN(f0), minNormal);
            const float dp1 = dot(csPlaneN(f1), minNormal);

            if (dp0 > dp1)
            {
                polyFaceIndex = f0;
                polyIndex2 = f1;
            }
            else
            {
                polyFaceIndex = f1;
                polyIndex2 = f0;
            }
        }
        else
        {
            maxDpSq2 = pxgWarpReduceMaxWinnerF(PXG_FULL_MASK, lane, maxDpSq2, 5u, winnerLane);
            polyIndex2 = pxgShflSyncU(PXG_FULL_MASK, lane, polyIndex2, int(winnerLane), CS_WARP);
        }
    }

    return polyFaceIndex;
}
