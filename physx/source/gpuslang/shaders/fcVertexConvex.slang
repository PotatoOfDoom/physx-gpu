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

// fcVertexConvex.slang
//
// Slang port of the cloth VERTEX vs convex contact path
// (femClothPrimitives.cu clothVertexConvexCollision +
// closestVertToConvex + computeClosestPolygon + distancePointSegment,
// deformableElementFilter.cuh find(PxgRigidFilterPair*)):
//
// - the LIVE attachment-filter check: find(filterPairs, nbPairs, rigidId,
//   PxEncodeClothIndex(clothId, vertexIdx)) — the PxgRigidFilterPair
//   binary search navigated by the 3-key (index0, index1, index2)
//   lexicographic compare, accepting on exact match OR (ignoreIndex2)
//   index0/index1 equality — a filtered vertex generates nothing;
// - the cloth vertex transformed into the convex SHAPE space
//   (transformInv), then closestVertToConvex:
//   * per-polygon signed distances in VERTEX space, each scaled by
//     |rotate(scaleRot, plane.n) * scale| (scaleCoef) — a per-lane strided
//     argmax with strict-greater update tracks the deepest plane while
//     `outside` accumulates any positive distance;
//   * OUTSIDE: iterate the planes with positive UNSCALED vertex-space
//     distance in ascending index order (the CUDA's chunked ballot +
//     lowestSetIndex walk); for each, computeClosestPolygon: per polygon
//     edge the edge plane (plane.n x (rpA - rpB)).getNormalized(); if the
//     point is outside any edge plane, the closest point is the MIN over
//     those edges of distancePointSegment (warpReduction MinOp, lowest
//     lane on ties) with pen = sqrt(min); otherwise the face projection
//     with pen = plane.distance.  The running polygon min uses strict <
//     (first polygon wins ties).  The final normal is
//     pVertexSpace - closestP and pen = |rotate(normal) * scale|
//     (positive, shape space);
//   * ALL-INSIDE: warpReduction MaxOp over the scaled signed distances
//     (lowest lane on ties) picks the least-negative plane; closestP =
//     plane.project(pVertexSpace), normal = plane.n, pen = the SCALED
//     signed distance (negative);
//   * outputs (lane 0): contact = vertex2Shape(closestP), normal =
//     vertex2ShapeNormalVector(normal) with PxVec3::normalizeSafe (zero
//     below 1e-20), pen as computed;
// - back to world space (transform.transform / rotate + getNormalized),
//   gate pen < contactDist STRICT, then ONE atomicAdd reservation and
//   writeRigidVsDeformableContact: contact.w = restDist, normalPen.w =
//   pen, barycentric = (0, 0, 0, 1) — the 1 in w marks a VERTEX contact —
//   pairInd0 = rigidId, pairInd1 = PxEncodeClothIndex(clothId, vertexIdx),
//   info fourth dword = rigidMaterialIndex with the isInCollision bit
//   cleared.
//
// WARP SERIALIZATION: one warp per pair upstream; every reduction is
// order-fixed (warpReduction winnerLane = lowest lane holding the best
// value after per-lane strided strict-greater/less scans; the
// intersecting-polygon walk is ascending-index) and the only atomic is
// the single per-contact reservation — one CASE per thread with faithful
// two-level loops is output-identical.  Caller glue not ported
// (documented): LoadShapePairWarp/transform caches, PxgFEMCloth loads and
// the pair-stack iteration (cases carry the vertex, poses, scale and ids
// directly).  Hull blob layout identical to fcConvexTri.slang.

import pxgSlangCommon;

static const uint PXG_BITSHIFT_ELEMENT_ID = 20;
static const uint COLLISION_FLAG_MASK = 1u << 31;
static const float PX_NORMALIZATION_EPSILON = 1e-20f;

struct FvcParams { uint numCases; uint maxContacts; uint nbFilterPairs; uint pad0; }

struct FvcCase
{
    float4 clothP;         // world cloth-vertex position; w = contactDist
    float4 rigidP;         // convex transform position; w = restDist
    float4 rigidQ;         // convex transform quaternion
    float4 convexScale;    // xyz
    float4 convexScaleRot; // quaternion
    uint4 ids;             // x = rigidId lo, y = rigidId hi, z = clothId, w = vertexIdx
    uint4 ids2;            // x = rigidMaterialIndex, y = hull offset (uints)
}

// PxgRigidFilterPair (32 bytes): u64 index0, u64 index1, u32 index2, pad
struct FilterPair
{
    uint2 index0;
    uint2 index1;
    uint index2;
    uint pad0, pad1, pad2;
}

[[vk::push_constant]]
ConstantBuffer<FvcParams> gParams;

RWStructuredBuffer<FvcCase> gCase;        // binding 0
RWStructuredBuffer<uint> gHull;           // binding 1
RWStructuredBuffer<FilterPair> gFilter;   // binding 2 (sorted)
// PxgFEMContactWriter (rigid sorting flavour)
RWStructuredBuffer<float4> gFemPoint;     // binding 3
RWStructuredBuffer<float4> gFemNormalPen; // binding 4
RWStructuredBuffer<float4> gFemBary;      // binding 5
RWStructuredBuffer<uint4> gFemInfo;       // binding 6
RWStructuredBuffer<uint> gFemCount;       // binding 7
RWStructuredBuffer<uint2> gFemByX;        // binding 8
RWStructuredBuffer<uint> gFemTempByX;     // binding 9
RWStructuredBuffer<uint> gFemIdxByX;      // binding 10
RWStructuredBuffer<float4> gDebug;        // binding 11: (filtered, pen, wrote, insideBranch)

// ---- hull blob accessors (fcConvexTri.slang) --------------------------------
uint hdr(uint hull) { return gHull[hull + 4]; }
uint nbPolygonsOf(uint h) { return h & 0xffu; }
uint nbVertsOf(uint h) { return (h >> 8) & 0xffu; }
uint nbEdgesOf(uint h) { return h >> 16; }
uint vertsBase(uint hull) { return hull + 12; }
uint planesBase(uint hull) { return hull + 12 + 4 * nbVertsOf(hdr(hull)); }
uint polyDescsBase(uint hull) { return hull + 12 + 4 * nbVertsOf(hdr(hull)) + 4 * nbPolygonsOf(hdr(hull)); }
uint verticesByEdgesByteBase(uint hull)
{
    const uint h = hdr(hull);
    return (hull + 12 + 4 * nbVertsOf(h) + 5 * nbPolygonsOf(h)) * 4;
}
uint facesByEdgesByteBase(uint hull) { return verticesByEdgesByteBase(hull) + 4 * nbEdgesOf(hdr(hull)); }
uint vertexData8ByteBase(uint hull)
{
    const uint h = hdr(hull);
    return facesByEdgesByteBase(hull) + 2 * nbEdgesOf(h) + 3 * nbVertsOf(h);
}
uint readU8(uint byteOfs) { return (gHull[byteOfs >> 2] >> ((byteOfs & 3) * 8)) & 0xffu; }
float4 readF4(uint uintOfs)
{
    return float4(asfloat(gHull[uintOfs]), asfloat(gHull[uintOfs + 1]),
                  asfloat(gHull[uintOfs + 2]), asfloat(gHull[uintOfs + 3]));
}
uint pdNbVerts(uint d) { return (d >> 8) & 0xffu; }
uint pdVRef8(uint d) { return d >> 16; }

// ---- scale/space helpers -----------------------------------------------------
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) / scale);
}
float3 vertex2ShapeNormalVector(float3 n, float3 scale, float4 rot)
{
    return shape2Vertex(n, scale, rot);
}
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}
// PxPlane::distance / project
float planeDistance(float4 pl, float3 p) { return dot(pl.xyz, p) + pl.w; }
float3 planeProject(float4 pl, float3 p) { return p - pl.xyz * planeDistance(pl, p); }

// femClothPrimitives.cu distancePointSegment (returns the segment point)
float3 distancePointSegment(float3 a, float3 b, float3 p)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    const float tValue = clamp(nom / denom, 0.0f, 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    return a + ab * t;
}

// ---- find(PxgRigidFilterPair*) (deformableElementFilter.cuh) ----------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// 3-key lexicographic compare; accept on exact match OR (ignoreIndex2)
// index0/index1 equality
bool findFilter(uint2 rigidId, uint clothMask)
{
    const uint2 q1 = uint2(clothMask, 0);
    uint l = 0, r = gParams.nbFilterPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const FilterPair p = gFilter[mid];
        int cmp;
        if (!u64Eq(rigidId, p.index0))
            cmp = u64Lt(rigidId, p.index0) ? -1 : 1;
        else if (!u64Eq(q1, p.index1))
            cmp = u64Lt(q1, p.index1) ? -1 : 1;
        else if (0u != p.index2)
            cmp = (0u < p.index2) ? -1 : 1;
        else
            cmp = 0;
        if (cmp == 0 || (u64Eq(rigidId, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

// ---- writeRigidVsDeformableContact (PxgFEMCore.h) ---------------------------
void writeRigidVsDeformableContact(uint index, float4 contact, float4 normalPen, float4 bary,
                                   uint2 pairInd0, uint pairInd1, uint rigidMaterialIndex, uint2 rigidId)
{
    if (index >= gParams.maxContacts) return;
    gFemByX[index] = rigidId;
    gFemTempByX[index] = rigidId.x;
    gFemIdxByX[index] = index;
    gFemPoint[index] = contact;
    gFemNormalPen[index] = normalPen;
    gFemBary[index] = bary;
    uint4 info = gFemInfo[index];
    info.x = pairInd0.x;
    info.y = pairInd0.y;
    info.z = pairInd1;
    info.w = (info.w & COLLISION_FLAG_MASK) | (rigidMaterialIndex & ~COLLISION_FLAG_MASK);
    info.w = info.w & ~COLLISION_FLAG_MASK;
    gFemInfo[index] = info;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void ctFcVertexConvex(uint3 tid : SV_DispatchThreadID)
{
    const uint caseIdx = tid.x;
    if (caseIdx >= gParams.numCases) return;
    const FvcCase cs = gCase[caseIdx];
    const uint hull = cs.ids2.y;
    const uint nbPolygons = nbPolygonsOf(hdr(hull));
    const float3 convexScale = cs.convexScale.xyz;
    const float4 convexScaleRot = cs.convexScaleRot;
    const float contactDist = cs.clothP.w;

    float4 dbg = float4(0, 0, 0, 0);

    // LIVE attachment filter (a hit generates nothing)
    const uint2 rigidId = uint2(cs.ids.x, cs.ids.y);
    const uint clothMask = (cs.ids.z << PXG_BITSHIFT_ELEMENT_ID) | cs.ids.w;
    if (findFilter(rigidId, clothMask))
    {
        dbg.x = 1.0f;
        gDebug[caseIdx] = dbg;
        return;
    }

    // vertex into the convex shape space
    const float3 shapeSpaceP = quatRotateInv(cs.rigidQ, cs.clothP.xyz - cs.rigidP.xyz);

    // ---- closestVertToConvex ----------------------------------------------
    const float3 pVertexSpace = shape2Vertex(shapeSpaceP, convexScale, convexScaleRot);

    // per-lane strided argmax over the SCALED vertex-space plane distances
    float laneMax[32];
    uint laneIdx[32];
    bool outside = false;
    for (uint lane = 0; lane < 32; ++lane)
    {
        laneMax[lane] = -PXG_MAX_F32;
        laneIdx[lane] = 0;
        for (uint i = lane; i < nbPolygons; i += 32)
        {
            const float4 plane = readF4(planesBase(hull) + 4 * i);
            const float signDistVertexSpace = planeDistance(plane, pVertexSpace);
            const float scaleCoef = length(quatRotate(convexScaleRot, plane.xyz) * convexScale);
            const float signDist = signDistVertexSpace * scaleCoef;
            outside = outside || signDist > 0.0f;
            if (signDist > laneMax[lane])
            {
                laneMax[lane] = signDist;
                laneIdx[lane] = i;
            }
        }
    }

    float3 closestP;
    float3 normal;
    float maxVal;

    if (outside)
    {
        // walk the planes with positive UNSCALED distance in ascending order
        float minDist = PXG_MAX_F32;
        closestP = float3(0, 0, 0);
        for (uint idx = 0; idx < nbPolygons; ++idx)
        {
            const float4 plane = readF4(planesBase(hull) + 4 * idx);
            if (!(planeDistance(plane, pVertexSpace) > 0.0f)) continue;

            const uint polyDesc = gHull[polyDescsBase(hull) + idx];
            const uint vref = pdVRef8(polyDesc);
            const uint nbVerts = pdNbVerts(polyDesc);

            float3 vert[32];
            for (uint l = 0; l < 32; ++l) vert[l] = float3(0, 0, 0);
            for (uint l = 0; l < nbVerts; ++l)
                vert[l] = readF4(vertsBase(hull) + 4 * readU8(vertexData8ByteBase(hull) + vref + l)).xyz;

            // ---- computeClosestPolygon --------------------------------------
            float3 polyClosestP = float3(0, 0, 0);
            float polyDist;
            {
                float laneSign[32];
                bool anyOutside = false;
                for (uint l = 0; l < 32; ++l)
                {
                    laneSign[l] = -PXG_MAX_F32;
                    if (l < nbVerts)
                    {
                        const float3 rpA = vert[l];
                        const float3 rpB = vert[l + 1 == nbVerts ? 0 : l + 1];
                        const float3 edgePlaneNormal = getNormalized(cross(plane.xyz, rpA - rpB));
                        laneSign[l] = dot(pVertexSpace - rpA, edgePlaneNormal);
                        anyOutside = anyOutside || (laneSign[l] > 0.0f);
                    }
                }
                if (anyOutside)
                {
                    float best = PXG_MAX_F32;
                    uint winner = 0;
                    float dists[32];
                    float3 cps[32];
                    for (uint l = 0; l < 32; ++l)
                    {
                        dists[l] = PXG_MAX_F32;
                        cps[l] = float3(0, 0, 0);
                        if (l < nbVerts && laneSign[l] > 0.0f)
                        {
                            const float3 rpA = vert[l];
                            const float3 rpB = vert[l + 1 == nbVerts ? 0 : l + 1];
                            cps[l] = distancePointSegment(rpA, rpB, pVertexSpace);
                            const float3 d = cps[l] - pVertexSpace;
                            dists[l] = dot(d, d);
                        }
                        if (dists[l] < best) best = dists[l];
                    }
                    for (uint l = 0; l < 32; ++l)
                        if (dists[l] == best) { winner = l; break; }
                    polyClosestP = cps[winner];
                    polyDist = sqrt(best);
                }
                else
                {
                    polyDist = planeDistance(plane, pVertexSpace);
                    polyClosestP = pVertexSpace - plane.xyz * polyDist;
                }
            }

            if (polyDist < minDist)
            {
                minDist = polyDist;
                closestP = polyClosestP;
            }
        }

        normal = pVertexSpace - closestP;
        maxVal = length(quatRotate(convexScaleRot, normal) * convexScale);
    }
    else
    {
        // all inside: least-negative scaled plane (lowest lane on ties)
        dbg.w = 1.0f;
        float best = -PXG_MAX_F32;
        for (uint lane = 0; lane < 32; ++lane) best = max(best, laneMax[lane]);
        uint winner = 0;
        for (uint lane = 0; lane < 32; ++lane)
            if (laneMax[lane] == best) { winner = lane; break; }
        const uint polyFaceIndex = laneIdx[winner];
        const float4 plane = readF4(planesBase(hull) + 4 * polyFaceIndex);
        closestP = planeProject(plane, pVertexSpace);
        normal = plane.xyz;
        maxVal = best;
    }

    // lane-0 outputs
    float3 outContact = vertex2Shape(closestP, convexScale, convexScaleRot);
    float3 outNormal = vertex2ShapeNormalVector(normal, convexScale, convexScaleRot);
    // PxVec3::normalizeSafe: below the epsilon the vector is left UNCHANGED
    // (the function only reports magnitude 0); the world getNormalized below
    // then decides
    {
        const float mag = length(outNormal);
        if (mag >= PX_NORMALIZATION_EPSILON)
            outNormal /= mag;
    }
    const float pen = maxVal;

    const float3 worldP = quatRotate(cs.rigidQ, outContact) + cs.rigidP.xyz;
    const float3 worldNormal = getNormalized(quatRotate(cs.rigidQ, outNormal));

    dbg.y = pen;
    if (pen < contactDist)
    {
        dbg.z = 1.0f;
        const uint pairInd1 = clothMask;
        uint ind;
        InterlockedAdd(gFemCount[0], 1, ind);
        writeRigidVsDeformableContact(ind, float4(worldP, cs.rigidP.w),
            float4(worldNormal, pen),
            float4(0.0f, 0.0f, 0.0f, 1.0f), // 1 in w marks a VERTEX contact
            rigidId, pairInd1, cs.ids2.x, rigidId);
    }
    gDebug[caseIdx] = dbg;
}
