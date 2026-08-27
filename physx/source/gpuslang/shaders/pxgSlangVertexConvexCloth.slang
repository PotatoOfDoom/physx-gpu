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

// pxgSlangVertexConvexCloth.slang -- P9.21
//
// The cloth VERTEX vs convex closest-point core, lifted out of
// fcVertexConvex.slang so the engine-ABI kernel can call it.
//
// Ported from femClothPrimitives.cu, all read in full:
//   closestVertToConvex   (:2027)
//   computeClosestPolygon (:1975)
//   distancePointSegment  (:1961)
// plus the tail of clothVertexConvexCollision (:2206-2240) -- the world-space
// transform, the contact-distance gate and the single-contact write.
//
// WHAT THE CORE COMPUTES
//
// The cloth vertex arrives in the convex's SHAPE space and is moved into its
// VERTEX space (shape2Vertex, i.e. the scale removed in the scale-rotation
// frame).  Then:
//
//   * a strided pass over the hull's planes computes each signed distance in
//     VERTEX space and multiplies it by
//         scaleCoef = |rotate(scaleRot, plane.n) * scale|
//     -- the factor that turns a vertex-space distance into a shape-space one
//     for THAT plane.  It tracks the argmax (strict >) and ORs together whether
//     any SCALED distance is positive.
//
//   * OUTSIDE (any positive): walk the planes whose UNSCALED vertex-space
//     distance is positive, in ascending index order, and for each run
//     computeClosestPolygon -- per polygon edge an edge plane
//     normalize(plane.n x (rpA - rpB)); if the point is outside ANY edge plane
//     the answer is the nearest point on the nearest such edge SEGMENT
//     (minimum of squared distances, lowest lane on ties), otherwise the face
//     projection.  The running minimum across polygons uses strict <, so the
//     FIRST polygon wins ties.  normal = pVertexSpace - closestP and the
//     penetration is |rotate(scaleRot, normal) * scale| -- POSITIVE outside.
//
//   * ALL INSIDE: the least-negative SCALED plane distance wins (lowest lane on
//     ties), closestP = plane.project(p), normal = plane.n, pen = that negative
//     distance.
//
//   * the outputs go back to shape space (vertex2Shape for the point,
//     vertex2ShapeNormalVector == shape2Vertex for the normal, then
//     PxVec3::normalizeSafe), then to world space, and a STRICT pen <
//     contactDist gate reserves one slot and writes the contact with
//     barycentric (0,0,0,1) -- the 1 in w is what marks a VERTEX contact
//     downstream.
//
// WARP SERIALIZATION.  The CUDA runs one warp per pair and uses
// __ballot_sync / warpReduction / __shfl_sync.  Every reduction here is
// order-fixed -- warpReduction's winnerLane is the LOWEST lane holding the best
// value after per-lane strided scans in which only a strictly better candidate
// replaces -- and the only atomic is the single per-contact reservation.  So one
// CASE per thread with faithful two-level (per-lane stride, then lowest-lane
// tie) loops is output-identical, and that is what this module does.
//
// LANE 0 OWNS THE OUTPUT IN THE CUDA TOO.  closestVertToConvex writes
// outContact / outNormal / pen only `if (threadIdx.x == 0)`, and the caller
// gates the write on `threadIdx.x == 0` as well.  Lanes 1..31 read
// uninitialised locals and throw the result away; a per-thread port simply does
// not have those lanes.
//
// ---------------------------------------------------------------------------
// THREE SPELLINGS THAT ARE NOT INTERCHANGEABLE
// ---------------------------------------------------------------------------
// S1 -- `sqrt(dot(v, v))`, NOT `length(v)`.  PxVec3::magnitude() is
//     `PxSqrt(magnitudeSquared())`.  Slang's `length` is free to lower to
//     something else, and P9.19 measured exactly that: a 1-ULP mismatch on five
//     of five vertices from this substitution alone.  Both `magnitude()` calls
//     in this core -- scaleCoef and the outside-branch penetration -- are
//     spelled the long way.
//
// S2 -- normalizeSafe SCALES BY THE RECIPROCAL.  PxVec3::normalizeSafe is
//         `if (mag < PX_NORMALIZATION_EPSILON) return 0;  *this *= 1.0f / mag;`
//     One divide and one vector-times-scalar.  `v /= mag` is three divides and
//     rounds differently.  The guard is spelled `!(mag < eps)` rather than
//     `mag >= eps` for the same reason: the two differ when mag is NaN, and
//     copying the CUDA costs one instruction and no argument.
//
// S3 -- getNormalized() is `m > 0 ? v * rsqrt(m) : 0` on the SQUARED magnitude
//     (PxVec3.h:290) -- a reciprocal square root, not a divide by a sqrt.
//
// ---------------------------------------------------------------------------
// NOT IN HERE, and each is caller glue with a reason
// ---------------------------------------------------------------------------
// * LoadShapePairWarp, the transform caches, the PxgFEMCloth vertex load and
//   the midphase pair-stack iteration -- the case carries the vertex, the
//   poses, the scale and the ids directly.
// * the attachment-filter `find()`.  It is a SINGLE-match skip here (one
//   filtered vertex generates nothing), which is the OPPOSITE quantifier from
//   the triangle sibling's all-three-corners AND -- so it stays visible in the
//   caller rather than buried behind a call.
// ---------------------------------------------------------------------------

import pxgSlangCommon;
import pxgSlangConvexHull;

public static const uint FVC_BITSHIFT_ELEMENT_ID = 20;
public static const float FVC_NORMALIZATION_EPSILON = 1e-20f;

// The per-pair inputs, public so a caller can fill one.
public struct FvcCase
{
    public float4 clothP;         // world cloth-vertex position; w = contactDist
    public float4 rigidP;         // convex transform position;   w = restDist
    public float4 rigidQ;         // convex transform quaternion
    public float4 convexScale;    // xyz
    public float4 convexScaleRot; // quaternion
    public uint4 ids;             // x = rigidId lo, y = rigidId hi, z = clothId, w = vertexIdx
    public uint4 ids2;            // x = rigidMaterialIndex, y = hull offset (uints)
}

// PxgFEMContactWriter, by address.
public struct FvcWriter
{
    public uint64_t point, normalPen, bary, info, count, byX, tempByX, idxByX;
    public uint maxContacts;
}

// ---- scale/space helpers (cudaNpCommon.h:265-305) ---------------------------
float3 fvcVertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
float3 fvcShape2Vertex(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) / scale);
}
// vertex2ShapeNormalVector IS shape2Vertex (cudaNpCommon.h:301) -- covariant.
float3 fvcVertex2ShapeNormalVector(float3 n, float3 scale, float4 rot)
{
    return fvcShape2Vertex(n, scale, rot);
}
// S1: PxVec3::magnitude().
float fvcMagnitude(float3 v) { return sqrt(dot(v, v)); }
// S3: PxVec3::getNormalized().
float3 fvcGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}
// PxPlane::distance / project (PxPlane.h:94, :107)
float fvcPlaneDistance(float4 pl, float3 p) { return dot(pl.xyz, p) + pl.w; }
float3 fvcPlaneProject(float4 pl, float3 p) { return p - pl.xyz * fvcPlaneDistance(pl, p); }

// femClothPrimitives.cu:1961 distancePointSegment -- returns the SEGMENT POINT,
// not the distance, despite the name.
float3 fvcDistancePointSegment(float3 a, float3 b, float3 p)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    const float tValue = clamp(nom / denom, 0.0f, 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    return a + ab * t;
}

// ---- writeRigidVsDeformableContact (PxgFEMCore.h:485) -----------------------
// pairInd0 == rigidId.getInd(); the caller passes them equal, as the plane and
// sphere siblings do.
void fvcWriteRigidVsDeformableContact(FvcWriter w, uint index, float4 contact, float4 normalPen,
                                      float4 bary, uint2 rigidId, uint pairInd1,
                                      uint rigidMaterialIndex)
{
    if (index >= w.maxContacts) return;
    Ptr<uint2>(w.byX)[index] = rigidId;
    Ptr<uint>(w.tempByX)[index] = rigidId.x;
    Ptr<uint>(w.idxByX)[index] = index;
    Ptr<float4>(w.point)[index] = contact;
    Ptr<float4>(w.normalPen)[index] = normalPen;
    Ptr<float4>(w.bary)[index] = bary;
    // PxgFemOtherContactInfo (16B): pairInd0 u64 @0, pairInd1 u32 @8, mat @12
    const uint64_t infoBase = w.info + uint64_t(index) * 16UL;
    Ptr<uint2>(infoBase + 0UL)[0] = rigidId;
    Ptr<uint>(infoBase + 8UL)[0] = pairInd1;
    // setRigidMaterialIndex(mat) + markInCollision(false) == (mat & 0x7fffffff)
    Ptr<uint>(infoBase + 12UL)[0] = rigidMaterialIndex & 0x7fffffffu;
}

// ---------------------------------------------------------------------------
// The core.  Returns the debug quad (unused, pen, wrote, tookTheInsideBranch);
// .x is left to the caller, which owns the filter verdict.
// ---------------------------------------------------------------------------
public float4 fcClothVertexConvexContactGen(FvcCase cs, uint64_t hull, FvcWriter w)
{
    const uint hullDesc = chHullDesc(hull);
    const uint nbPolygons = chNbPolygons(hullDesc);
    const float3 convexScale = cs.convexScale.xyz;
    const float4 convexScaleRot = cs.convexScaleRot;
    const float contactDist = cs.clothP.w;

    float4 dbg = float4(0, 0, 0, 0);

    // the cloth vertex into the convex SHAPE space (PxTransform::transformInv)
    const float3 shapeSpaceP = quatRotateInv(cs.rigidQ, cs.clothP.xyz - cs.rigidP.xyz);

    // ---- closestVertToConvex (:2027) --------------------------------------
    const float3 pVertexSpace = fvcShape2Vertex(shapeSpaceP, convexScale, convexScaleRot);

    // The per-lane strided argmax over the SCALED vertex-space plane distances.
    // `outside` is never reset between lanes: it is the OR over the whole warp,
    // which is what `__ballot_sync(...) != 0` tests one line later.
    float laneMax[32];
    uint laneIdx[32];
    bool outside = false;
    for (uint lane = 0; lane < 32; ++lane)
    {
        laneMax[lane] = -PXG_MAX_F32;
        laneIdx[lane] = 0;
        for (uint i = lane; i < nbPolygons; i += 32)
        {
            const float4 plane = chPlane(hull, hullDesc, i);
            const float signDistVertexSpace = fvcPlaneDistance(plane, pVertexSpace);
            // S1
            const float scaleCoef = fvcMagnitude(quatRotate(convexScaleRot, plane.xyz) * convexScale);
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
        // The gate on the inner walk is the UNSCALED vertex-space distance, not
        // the scaled one the argmax used.  Two different tests over the same
        // planes, four lines apart in the CUDA.
        float minDist = PXG_MAX_F32;
        closestP = float3(0, 0, 0);
        for (uint idx = 0; idx < nbPolygons; ++idx)
        {
            const float4 plane = chPlane(hull, hullDesc, idx);
            if (!(fvcPlaneDistance(plane, pVertexSpace) > 0.0f)) continue;

            const uint polyDesc = chPolyDesc(hull, hullDesc, idx);
            const uint vref = chPolyVRef8(polyDesc);
            const uint nbVerts = chPolyNbVerts(polyDesc);

            float3 vert[32];
            for (uint l = 0; l < 32; ++l) vert[l] = float3(0, 0, 0);
            for (uint l = 0; l < nbVerts; ++l)
                vert[l] = chVertex(hull, chPolyVertex(hull, hullDesc, vref, l)).xyz;

            // ---- computeClosestPolygon (:1975) ------------------------------
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
                        const float3 edgePlaneNormal = fvcGetNormalized(cross(plane.xyz, rpA - rpB));
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
                            cps[l] = fvcDistancePointSegment(rpA, rpB, pVertexSpace);
                            const float3 d = cps[l] - pVertexSpace;
                            dists[l] = dot(d, d);
                        }
                        if (dists[l] < best) best = dists[l];
                    }
                    // warpReduction's winnerLane: the LOWEST lane holding the min.
                    for (uint l = 0; l < 32; ++l)
                        if (dists[l] == best) { winner = l; break; }
                    polyClosestP = cps[winner];
                    polyDist = sqrt(best);
                }
                else
                {
                    polyDist = fvcPlaneDistance(plane, pVertexSpace);
                    polyClosestP = pVertexSpace - plane.xyz * polyDist;
                }
            }

            // strict <: the FIRST polygon wins ties.
            if (polyDist < minDist)
            {
                minDist = polyDist;
                closestP = polyClosestP;
            }
        }

        normal = pVertexSpace - closestP;
        // S1; positive, because the point is exterior.
        maxVal = fvcMagnitude(quatRotate(convexScaleRot, normal) * convexScale);
    }
    else
    {
        dbg.w = 1.0f;
        float best = -PXG_MAX_F32;
        for (uint lane = 0; lane < 32; ++lane) best = max(best, laneMax[lane]);
        uint winner = 0;
        for (uint lane = 0; lane < 32; ++lane)
            if (laneMax[lane] == best) { winner = lane; break; }
        const uint polyFaceIndex = laneIdx[winner];
        const float4 plane = chPlane(hull, hullDesc, polyFaceIndex);
        closestP = fvcPlaneProject(plane, pVertexSpace);
        normal = plane.xyz;
        maxVal = best;                 // already shape-space, and NEGATIVE
    }

    // ---- lane-0 outputs ----------------------------------------------------
    const float3 outContact = fvcVertex2Shape(closestP, convexScale, convexScaleRot);
    float3 outNormal = fvcVertex2ShapeNormalVector(normal, convexScale, convexScaleRot);
    {
        // S2: PxVec3::normalizeSafe -- below the epsilon the vector is left
        // UNCHANGED (the function only reports a magnitude of 0).
        const float mag = fvcMagnitude(outNormal);
        if (!(mag < FVC_NORMALIZATION_EPSILON))
            outNormal = outNormal * (1.0f / mag);
    }
    const float pen = maxVal;

    const float3 worldP = quatRotate(cs.rigidQ, outContact) + cs.rigidP.xyz;
    const float3 worldNormal = fvcGetNormalized(quatRotate(cs.rigidQ, outNormal));

    dbg.y = pen;
    if (pen < contactDist)
    {
        dbg.z = 1.0f;
        const uint2 rigidId = uint2(cs.ids.x, cs.ids.y);
        const uint pairInd1 = (cs.ids.z << FVC_BITSHIFT_ELEMENT_ID) | cs.ids.w;
        uint ind;
        InterlockedAdd(Ptr<uint>(w.count)[0], 1u, ind);
        fvcWriteRigidVsDeformableContact(w, ind, float4(worldP, cs.rigidP.w),
            float4(worldNormal, pen),
            float4(0.0f, 0.0f, 0.0f, 1.0f),   // the 1 in w marks a VERTEX contact
            rigidId, pairInd1, cs.ids2.x);
    }
    return dbg;
}
