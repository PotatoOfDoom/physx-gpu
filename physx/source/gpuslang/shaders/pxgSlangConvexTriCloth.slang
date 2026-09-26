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

// pxgSlangConvexTriCloth.slang -- P9.15c
//
// Slang port of the cloth convex-vs-triangle contact generation
// (femClothPrimitives.cu convexTriangleContactGen + selectPolygonFeatureIndex2,
// convexTriangle.cuh convexMeshSAT<false> / satConvexFaceNormals /
// satTriangleNormal / selectPolygonFeatureIndex, contactReduction.cuh
// contactReduce<false, true, 5>, PxgOutputIndex, and
// PxgFEMContactWriter::writeRigidVsDeformableContact):
//
// - convexScaledVerts precompute (vertex2Shape of every hull vertex);
// - convexMeshSAT<false>: satConvexFaceNormals (per convex face, plane
//   normal to shape space via vertex2ShapeNormalVector with the
//   1/normalize() invMagnitude scaling of the plane d, disjoint when either
//   projection separates beyond contactDist, running max of min1 - max0;
//   also fills the convexPlaneN/convexPlaneD caches) then satTriangleNormal
//   (hull projected onto the triangle normal, separation = min1 - triProj,
//   feature = TRI_FEATURE_IDX, normal = -n0).  The edge-edge stage is
//   compiled out for cloth (doEdgeEdge = false), so EDGE_FEATURE_IDX is
//   unreachable here; selectPolygonFeatureIndex still checks both.
// - selectPolygonFeatureIndex: for TRI/EDGE features pick the convex face
//   whose cached normal best aligns with minNormal, then scan the edges:
//   the best polyIndex2 neighbor of the winning face (max squared dot
//   among edges containing it), and if some edge normal-sum beats the face
//   alignment (dp > 0 and sqDp > maxDpSq * |edgeN|^2, seeded with
//   minProj^2), both of that edge's faces win instead (ordered by dot).
// - the two-face contact loop: reference plane from the vertex-space plane
//   via vertex2ShapeNormalVector + PxPlane::normalize (n /= |n|, d /= |n|),
//   polygon verts gathered through polyDesc vRef8/vertexData8 + vertex2Shape,
//   polyClip4 (tranche 95, inlined here), computeBarycentric VALIDITY gate
//   on each generated contact (the eps window is live in this caller),
//   contactReduce<clustering=false, keepAnotherDeepestForPCM=true,
//   maxPoints=5, dupeTest=true> when more than 5 survive, PxgOutputIndex
//   (ONE atomicAdd of popc(mask) per batch reserving a contiguous range —
//   the reservation happens BEFORE the `mask > 0` write gate, and that gate
//   is a SIGNED int comparison: a kept lane-31 contact makes the whole
//   batch unwritten while the counter still advances — faithful quirk),
//   then writeRigidVsDeformableContact: world-space point (w = restDist)
//   and normal (w = separation), barycentric, pairInd0 = rigidId (u64),
//   pairInd1 = PxEncodeClothIndex(clothId, triangleId), and the info fourth
//   dword = rigidMaterialIndex with the isInCollision bit (31) cleared.
//
// WARP SERIALIZATION: the CUDA runs one warp per pair.  All reductions are
// order-fixed (warpReduction winnerLane = LOWEST lane holding the maximum,
// after per-lane strided scans where a strictly-greater candidate replaces),
// and the only atomic is the single PxgOutputIndex reservation per batch —
// so one CASE per thread with faithful two-level (per-lane stride, then
// lowest-lane tie) argmax loops is output-identical.  Caller glue not
// ported (documented): LoadShapePairWarp/transform caches, the filter-pair
// all-three-verts skip (find() ported in tranche 75), and the triangle
// transform into convex space (cases carry triLocVerts/triangleLocNormal).
//
// Hull blob layout (ConvexScratch getters): centerOfMass float4, header
// uint4 (x = nbEdgesNbHullVerticesNbPolygons: bits 0-7 nbPolygons, 8-15
// nbVerts, 16-31 nbEdges), extents float4, then vertices float4[nbVerts],
// planes float4[nbPolygons], polyDescs u32[nbPolygons] (bits 0-7 minIndex,
// 8-15 nbVerts, 16-31 vRef8), verticesByEdges u16[2*nbEdges], facesByEdges
// u8[2*nbEdges], facesByVertices u8[3*nbVerts], vertexData8 u8[].

import pxgSlangCommon;
import pxgSlangContactReduceSerial;
// P9.15c: the hull is read through the SHARED BDA accessors instead of a bound
// StructuredBuffer.  pxgSlangConvexHull already decodes this exact blob for the
// particle-vs-hull path (P9.14b), and its section offsets were verified to match
// this file's former uint-offset accessors byte for byte:
//     verticesByEdges = hull + 48 + 16*nbVerts + 20*nbPolys   (both)
//     facesByEdges    = verticesByEdges + 4*nbEdges           (both)
//     vertexData8     = facesByEdges + 2*nbEdges + 3*nbVerts  (both)
// Moving to them removes the second spelling AND puts this body one step from
// being callable by the engine-ABI kernel, which cannot bind a StructuredBuffer.
import pxgSlangConvexHull;

// ===========================================================================
// P9.15c -- LIFTED OUT OF fcConvexTri.slang, unchanged.
//
// The body below is the Phase-1 port named in the comment above, moved verbatim
// into a module so a kernel can CALL it.  Nothing in the algebra changed; what
// changed, in the same tranche and one step earlier, is that the hull is read
// through pxgSlangConvexHull's BDA accessors and the contacts are written through
// BDA addresses instead of bound StructuredBuffers -- because an engine-ABI
// kernel has no descriptor set to bind them into.
//
// fcConvexTri.slang keeps its entry point and its battery, and now drives THIS
// code.  That battery is the verification of the lift: it passed unchanged
// across both halves, which is the whole reason for lifting rather than copying.
//
// What the header above calls "caller glue not ported" -- LoadShapePairWarp, the
// transform caches, the filter-pair all-three-verts skip and the triangle
// transform into convex space -- is exactly and only what the engine kernel
// cloth_boxTriangleContactGenLaunch still has to add.  It fills an FctCase and
// calls fcClothConvexTriContactGen.
// ===========================================================================

import pxgSlangCommon;
import pxgSlangConvexHull;

public static const uint NUM_TMP_CONTACTS_PER_PAIR = 32;
public static const uint TRI_FEATURE_IDX = 0x7FffFFffu;
public static const uint EDGE_FEATURE_IDX = 0x80000000u;
public static const uint PXG_BITSHIFT_ELEMENT_ID = 20;
public static const uint COLLISION_FLAG_MASK = 1u << 31;

struct FctParams
{
    uint numCases; uint maxContacts; uint pad0; uint pad1;   // @0
    uint64_t hullAddr;        // @16  base of the hull blob (FctCase.ids2.y is a UINT offset into it)
    // PxgFEMContactWriter, by address (@24..@88)
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX;
}


// The per-pair inputs, public so a caller can fill one.
public struct FctCase
{
    public float4 convexScale;    // xyz; w = contactDist
    public float4 convexScaleRot; // quaternion
    public float4 t0;             // triLocVerts (convex shape space); t0.w = restDist
    public float4 t1;
    public float4 t2;             // t2.w unused
    public float4 triNormal;      // triangleLocNormal
    public float4 convexP;        // convexTransform position
    public float4 convexQ;        // convexTransform quaternion
    public uint4 ids;             // x = rigidId lo, y = rigidId hi, z = clothId, w = triangleId
    public uint4 ids2;            // x = rigidMaterialIndex, y = hull offset (uints)
}

// PxgFEMContactWriter, by address.
public struct FcCtWriter
{
    public uint64_t point, normalPen, bary, info, count, byX, tempByX, idxByX;
    public uint maxContacts;
}

// ---- hull blob accessors: aliases over pxgSlangConvexHull -------------------
// The names below are the ones this file's algebra has always called; only their
// bodies moved.  `hull` is now a BYTE ADDRESS, not a uint offset.
uint hdr(uint64_t hull) { return chHullDesc(hull); }
uint nbPolygonsOf(uint h) { return chNbPolygons(h); }
uint nbVertsOf(uint h) { return chNbVerts(h); }
uint nbEdgesOf(uint h) { return chNbEdges(h); }

// polyDesc accessors
uint pdMinIndex(uint d) { return chPolyMinIndex(d); }
uint pdNbVerts(uint d) { return chPolyNbVerts(d); }
uint pdVRef8(uint d) { return chPolyVRef8(d); }

// ---- scale/space helpers ---------------------------------------------------
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}

// vertex2ShapeNormalVector == shape2Vertex(n, scale, rot)
float3 vertex2ShapeNormalVector(float3 n, float3 scale, float4 rot)
{
    float3 t = quatRotate(rot, n);
    t /= scale;
    return quatRotateInv(rot, t);
}

float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

float3 planeProject(float4 plane, float3 p)
{
    return p - plane.xyz * (dot(plane.xyz, p) + plane.w);
}

uint warpScanExclusive(uint mask, uint lane)
{
    return countbits(mask & ((1u << lane) - 1u));
}

// lowest set bit index
uint lowestSetIndex(uint m) { return firstbitlow(m); }

// ---- polyClip4 core (tranche 95, npPolyClip4.slang) ------------------------
static const float PX_EPS_REAL = 1.192092896e-07f;

struct Mat33 { float3 col0, col1, col2; }
float3 m33Transform(Mat33 m, float3 v) { return m.col0 * v.x + m.col1 * v.y + m.col2 * v.z; }
float3 m33TransformTranspose(Mat33 m, float3 v) { return float3(dot(m.col0, v), dot(m.col1, v), dot(m.col2, v)); }

Mat33 findRotationMatrixFromZ(float3 to)
{
    const float e = to.z;
    const float f = abs(e);
    Mat33 m;
    if (f <= 0.9999f)
    {
        const float vx = -to.y;
        const float vy = to.x;
        const float h = 1.0f / (1.0f + e);
        const float hvx = h * vx;
        const float hvxy = hvx * vy;
        m.col0 = float3(e + hvx * vx, hvxy, vy);
        m.col1 = float3(hvxy, e + h * vy * vy, -vx);
        m.col2 = float3(-vy, vx, e);
    }
    else
    {
        const float3 from = float3(0, 0, 1);
        const float3 absFrom = float3(0, 1, 0);
        const float3 u = absFrom - from;
        const float3 v = absFrom - to;
        const float c1 = -(2.0f / dot(u, u));
        const float c2 = -(2.0f / dot(v, v));
        const float c3 = c1 * c2 * dot(u, v);
        const float3 c1u = u * c1;
        const float3 c2v = v * c2;
        const float3 c3v = v * c3;
        float3 col0 = u * c3v.x + v * c2v.x + u * c1u.x; col0.x += 1.0f;
        float3 col1 = u * c3v.y + v * c2v.y + u * c1u.y; col1.y += 1.0f;
        float3 col2 = u * c3v.z + v * c2v.z + u * c1u.z; col2.z += 1.0f;
        // P9.86: u * c1u.x is the literal (-0, -0, +0) and u * c1u.z the literal (+0, 1, -1);
        // llvmpipe folds their +0 lanes away, but femClothPrimitives.cu:1031/1046 really add
        // them (IEEE (-0) + (+0) = +0): spell those adds as the select they amount to.
        // (The -0 lanes are exact no-ops, x + (-0) == x, so they need nothing.)
        col0.z = (col0.z == 0.0f) ? 0.0f : col0.z;
        col2.x = (col2.x == 0.0f) ? 0.0f : col2.x;
        m.col0 = col0; m.col1 = col1; m.col2 = col2;
    }
    return m;
}

bool containsParallel(float3 verts[32], uint numVerts, float3 p)
{
    const float tx = p.x;
    const float ty = p.y;
    uint numIntersects = 0;
    for (uint tI = 0; tI < numVerts; ++tI)
    {
        const float3 start = verts[tI];
        const float3 endP = verts[tI + 1 == numVerts ? 0 : tI + 1];
        const float jy = endP.y;
        const float iy = start.y;
        const float minY = min(jy, iy);
        const float maxY = max(jy, iy);
        if ((minY != maxY) && ty >= minY && ty <= maxY)
        {
            const float jx = endP.x;
            const float ix = start.x;
            const float jix = ix - jx;
            const float jiy = iy - jy;
            const float jty = ty - jy;
            const float part1 = jty * jix;
            const float part2 = (jx + PX_EPS_REAL) * jiy;
            const float part3 = tx * jiy;
            const bool comp = jiy > 0.0f;
            const float tmp = part1 + part2;
            const float comp1 = comp ? tmp : part3;
            const float comp2 = comp ? part3 : tmp;
            if (comp1 >= comp2) ++numIntersects;
        }
    }
    return (numIntersects % 2) == 1;
}

float signed2DTriArea(float3 a, float3 b, float3 c)
{
    const float3 ca = a - c;
    const float3 cb = b - c;
    return ca.x * cb.y - ca.y * cb.x;
}

bool segmentIntersect(float3 ipA, float3 ipB, float3 rpA, float3 rpB,
                      float3 rMin, float3 rMax, out float3 pAA, out float3 pBB)
{
    pAA = float3(0, 0, 0);
    pBB = float3(0, 0, 0);
    const float3 iMin = min(ipA, ipB);
    const float3 iMax = max(ipA, ipB);
    const bool xCon = iMin.x > rMax.x || rMin.x > iMax.x;
    const bool yCon = iMin.y > rMax.y || rMin.y > iMax.y;
    if (xCon || yCon) return false;
    const float a1 = signed2DTriArea(rpA, rpB, ipA);
    const float a2 = signed2DTriArea(rpA, rpB, ipB);
    if ((a1 * a2) < 0.0f)
    {
        const float a3 = signed2DTriArea(ipA, ipB, rpA);
        const float a4 = signed2DTriArea(ipA, ipB, rpB);
        if ((a3 * a4) < 0.0f)
        {
            const float t0 = a1 / (a2 - a1);
            pAA = ipA - t0 * (ipB - ipA);
            const float t1 = a3 / (a4 - a3);
            pBB = rpA - t1 * (rpB - rpA);
            return true;
        }
    }
    return false;
}

// polyClip4 (scalar serialization, tranche 95); v0/nbVerts0 = reference
// polygon, v1 = triangle verts, ct = 32x8 transposed contact buffer
uint polyClip4(float4 plane0, float3 v0[32], uint nbVerts0, float4 plane1, float3 v1[3],
               float3 axis, float contactDist, inout float ct[NUM_TMP_CONTACTS_PER_PAIR * 8])
{
    const Mat33 rot = findRotationMatrixFromZ(axis);
    float3 triangleP[32];
    float3 polyPoint[32];
    for (uint l = 0; l < 32; ++l) { triangleP[l] = float3(0, 0, 0); polyPoint[l] = float3(0, 0, 0); }
    for (uint l = 0; l < 3; ++l) triangleP[l] = m33Transform(rot, v1[l]);
    for (uint l = 0; l < nbVerts0; ++l) polyPoint[l] = m33Transform(rot, v0[l]);

    const float sqContactDist = contactDist * contactDist;
    uint numContacts = 0;

    // phase 2: polygon verts inside the 2D triangle
    {
        uint mask = 0;
        for (uint j = 0; j < nbVerts0; ++j)
            if (containsParallel(triangleP, 3, polyPoint[j]))
                mask |= 1u << j;

        float dist[32];
        float3 normal[32];
        float3 contactPoint[32];
        uint keepMask = 0;
        for (uint lane = 0; lane < 32; ++lane)
        {
            dist[lane] = PXG_MAX_F32;
            normal[lane] = float3(0, 0, 0);
            contactPoint[lane] = float3(0, 0, 0);
            if ((mask & (1u << lane)) != 0)
            {
                const float3 tProjPoint = planeProject(plane1, v0[lane]);
                const float3 dir = v0[lane] - tProjPoint;
                const float cosTheta = dot(dir, plane0.xyz);
                const float sqDist = dot(dir, dir);
                if (cosTheta < 0.0f)
                {
                    if (sqDist > sqContactDist)
                    {
                        // lane clears its own mask bit (drops from the AND)
                    }
                    else
                    {
                        dist[lane] = sqrt(sqDist);
                        normal[lane] = -getNormalized(dir);
                        contactPoint[lane] = tProjPoint;
                        keepMask |= 1u << lane;
                    }
                }
                else
                {
                    dist[lane] = -sqrt(sqDist);
                    normal[lane] = getNormalized(dir);
                    contactPoint[lane] = tProjPoint;
                    keepMask |= 1u << lane;
                }
            }
        }
        for (uint lane = 0; lane < 32; ++lane)
        {
            if ((keepMask & (1u << lane)) != 0)
            {
                const uint index = warpScanExclusive(keepMask, lane) + numContacts;
                if (index < NUM_TMP_CONTACTS_PER_PAIR)
                {
                    ct[8 * index + 0] = contactPoint[lane].x;
                    ct[8 * index + 1] = contactPoint[lane].y;
                    ct[8 * index + 2] = contactPoint[lane].z;
                    ct[8 * index + 3] = dist[lane];
                    ct[8 * index + 4] = normal[lane].x;
                    ct[8 * index + 5] = normal[lane].y;
                    ct[8 * index + 6] = normal[lane].z;
                    ct[8 * index + 7] = 0.0f;
                }
            }
        }
        numContacts += countbits(keepMask);
    }

    // phase 3: triangle edges vs polygon edges
    for (uint rStart = 0, rEnd = 2; rStart < 3; rEnd = rStart++)
    {
        const float3 rpA = triangleP[rStart];
        const float3 rpB = triangleP[rEnd];
        // faithful upstream quirk: z uses rpA twice (only x/y are consumed)
        const float3 rMin = float3(min(rpA.x, rpB.x), min(rpA.y, rpB.y), min(rpA.z, rpA.z));
        const float3 rMax = float3(max(rpA.x, rpB.x), max(rpA.y, rpB.y), max(rpA.z, rpA.z));

        float dist[32];
        float3 normal[32];
        float3 contactPoint[32];
        uint mask = 0;
        for (uint lane = 0; lane < 32; ++lane)
        {
            dist[lane] = PXG_MAX_F32;
            normal[lane] = float3(0, 0, 0);
            contactPoint[lane] = float3(0, 0, 0);
            const float3 ipA = polyPoint[lane];
            const float3 ipB = polyPoint[(lane + 1 == nbVerts0 ? 0 : lane + 1) & 31];
            float3 pAA, pBB;
            bool hasContacts = (lane < nbVerts0)
                ? segmentIntersect(ipA, ipB, rpA, rpB, rMin, rMax, pAA, pBB) : false;
            if (hasContacts)
            {
                const float3 pA = m33TransformTranspose(rot, pAA);
                contactPoint[lane] = m33TransformTranspose(rot, pBB);
                const float3 dir = pA - contactPoint[lane];
                const float cosTheta = dot(dir, plane0.xyz);
                const float sqDist = dot(dir, dir);
                if (cosTheta < 0.0f)
                {
                    if (sqDist > sqContactDist)
                        hasContacts = false;
                    else
                    {
                        dist[lane] = sqrt(sqDist);
                        normal[lane] = -getNormalized(dir);
                    }
                }
                else
                {
                    dist[lane] = -sqrt(sqDist);
                    normal[lane] = getNormalized(dir);
                }
            }
            if (hasContacts)
                mask |= 1u << lane;
        }
        for (uint lane = 0; lane < 32; ++lane)
        {
            if ((mask & (1u << lane)) != 0)
            {
                const uint index = warpScanExclusive(mask, lane) + numContacts;
                if (index < NUM_TMP_CONTACTS_PER_PAIR)
                {
                    ct[8 * index + 0] = contactPoint[lane].x;
                    ct[8 * index + 1] = contactPoint[lane].y;
                    ct[8 * index + 2] = contactPoint[lane].z;
                    ct[8 * index + 3] = dist[lane];
                    ct[8 * index + 4] = normal[lane].x;
                    ct[8 * index + 5] = normal[lane].y;
                    ct[8 * index + 6] = normal[lane].z;
                    ct[8 * index + 7] = 0.0f;
                }
            }
        }
        numContacts += countbits(mask);
    }
    return numContacts;
}

// ---- computeBarycentric (femClothPrimitives.cu) -----------------------------
bool computeBarycentric(float3 a, float3 b, float3 c, float3 p, out float4 barycentric)
{
    const float minEps = -1e-4f;
    const float maxEps = 1.0f + 1e-4f;
    const float3 v0 = b - a;
    const float3 v1 = c - a;
    const float3 v2 = p - a;
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;
    barycentric = float4(u, v, w, 0.0f);
    return (u >= minEps && u <= maxEps) &&
           (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps);
}

// ---- contactReduce<false, true, 5, true> (contactReduction.cuh; tranche 82)
// contactReduce<clustering=false, keepAnotherDeepestForPCM=true, maxPoints=5,
// dupeTest=true>(pointRaw, separation, normal, allMask, 0) -- distanceAdjustment
// = 0, initial criterion = separation (contactReduction.cuh :172).
//
// P9.83b-3: THE BODY MOVED OUT to pxgSlangContactReduceSerial.  This is the
// clustering=FALSE instantiation, and folding it is what made the module grow
// CUDA's `else` tail (contactReduction.cuh:159-165), which it had been missing.
//
// It was NOT, however, a latent short manifold here, as 5/n claimed: at
// <false, true, 5, true> the fill-out loop the module was missing is provably
// unreachable -- the anchor set is always 4, the PCM pick always fires and
// takes it to 5, and 5 == maxPoints.  Measured over 299920 reductions: zero
// iterations.  The tail belongs in the module for faithfulness, not because
// this call site needs it.  See the note on that else branch.
uint contactReduce5(float3 ptRaw[32], float sep[32], float3 normal, uint allMask)
{
    return crsContactReduce(ptRaw, sep, normal, allMask, 0.0f,
                            crsConfig(false, true, 5, true));
}

// ---- writeRigidVsDeformableContact (PxgFEMCore.h) ---------------------------
public void writeRigidVsDeformableContact(FcCtWriter w, uint index, float4 contact, float4 normalPen, float4 bary,
                                   uint2 pairInd0, uint pairInd1, uint rigidMaterialIndex, uint2 rigidId)
{
    if (index >= w.maxContacts) return;
    Ptr<uint2>(w.byX)[index] = rigidId;
    Ptr<uint>(w.tempByX)[index] = rigidId.x;
    Ptr<uint>(w.idxByX)[index] = index;
    Ptr<float4>(w.point)[index] = contact;
    Ptr<float4>(w.normalPen)[index] = normalPen;
    Ptr<float4>(w.bary)[index] = bary;
    const uint64_t infoBase = w.info + uint64_t(index) * 16UL;
    uint4 info = Ptr<uint4>(infoBase)[0];
    info.x = pairInd0.x;
    info.y = pairInd0.y;
    info.z = pairInd1;
    // setRigidMaterialIndex then markInCollision(false): bit 31 cleared
    info.w = (info.w & COLLISION_FLAG_MASK) | (rigidMaterialIndex & ~COLLISION_FLAG_MASK);
    info.w = info.w & ~COLLISION_FLAG_MASK;
    Ptr<uint4>(infoBase)[0] = info;
}

// ---------------------------------------------------------------------------

// The whole of ctFcConvexTri's former body.  Returns the debug word the Phase-1
// battery reads: (isSATPassed, featureIndex, faceIndex1, dbgFace2).
public uint4 fcClothConvexTriContactGen(FctCase cs, uint64_t hull, FcCtWriter w)
{
    const uint hullHeader = hdr(hull);
    const uint nbVerts = nbVertsOf(hullHeader);
    const uint nbPolygons = nbPolygonsOf(hullHeader);
    const uint nbEdges = nbEdgesOf(hullHeader);
    const float3 convexScale = cs.convexScale.xyz;
    const float4 convexScaleRot = cs.convexScaleRot;
    const float contactDist = cs.convexScale.w;
    const float rest = cs.t0.w;
    float3 triLocVerts[3];
    triLocVerts[0] = cs.t0.xyz;
    triLocVerts[1] = cs.t1.xyz;
    triLocVerts[2] = cs.t2.xyz;
    const float3 triangleLocNormal = cs.triNormal.xyz;

    // convexScaledVerts precompute
    float3 scaledVerts[64];
    for (uint i = 0; i < nbVerts; ++i)
        scaledVerts[i] = vertex2Shape(chVertex(hull, i).xyz, convexScale, convexScaleRot);

    float3 convexPlaneN[64];
    float convexPlaneD[64];

    float separation = -PXG_MAX_F32;
    float3 minNormal = float3(PXG_MAX_F32, PXG_MAX_F32, PXG_MAX_F32);
    uint featureIndex = 0xffffffffu;
    bool isSATPassed = true;

    // ---- satConvexFaceNormals ----------------------------------------------
    {
        // per-lane strided scans (lane holds i, i+32, ...) with strict-greater
        // update, then lowest-lane argmax — faithful warpReduction semantics
        float laneSep[32];
        uint laneFeature[32];
        float3 laneNormal[32];
        bool disjoint = false;
        for (uint lane = 0; lane < 32; ++lane)
        {
            laneSep[lane] = -PXG_MAX_F32;
            laneFeature[lane] = 0;
            laneNormal[lane] = minNormal;
            for (uint i = lane; i < nbPolygons; i += 32)
            {
                const uint polygonData = chPolyDesc(hull, hullHeader, i);
                const float4 polygonf4 = chPlane(hull, hullHeader, i);
                const float3 minVertConv = scaledVerts[pdMinIndex(polygonData)];
                const float planeDist = polygonf4.w;

                float3 shapeSpacePlaneNormal =
                    vertex2ShapeNormalVector(polygonf4.xyz, convexScale, convexScaleRot);
                // P9.47: normalize()'s return value is magnitude() = PxSqrt(msq).
                const float mag = sqrt(dot(shapeSpacePlaneNormal, shapeSpacePlaneNormal));
                const float invMagnitude = 1.0f / mag;
                // P9.36: convexTriangle.cuh:246 is
                // `const PxReal invMagnitude = 1.0f / shapeSpacePlaneNormal.normalize();`
                // -- normalize() scales through operator/= (PxVec3.h:260), ONE reciprocal
                // and three multiplies, and the reciprocal it needs is already on the line
                // above.  The m > 0 guard is normalize()'s and was missing here.
                if (mag > 0.0f)
                    shapeSpacePlaneNormal = shapeSpacePlaneNormal * invMagnitude;

                const float min0 = dot(shapeSpacePlaneNormal, minVertConv);
                const float max0 = -planeDist * invMagnitude;

                const float d0 = dot(triLocVerts[0], shapeSpacePlaneNormal);
                const float d1 = dot(triLocVerts[1], shapeSpacePlaneNormal);
                const float d2 = dot(triLocVerts[2], shapeSpacePlaneNormal);
                const float max1 = max(d0, max(d1, d2));
                const float min1 = min(d0, min(d1, d2));

                if ((min1 > max0 + contactDist) || (min0 > max1 + contactDist))
                {
                    disjoint = true;
                    break;
                }

                convexPlaneN[i] = shapeSpacePlaneNormal;
                convexPlaneD[i] = -max0;

                const float tempSeparation = min1 - max0;
                if (laneSep[lane] < tempSeparation)
                {
                    laneSep[lane] = tempSeparation;
                    laneFeature[lane] = i;
                    laneNormal[lane] = shapeSpacePlaneNormal;
                }
            }
            if (disjoint) break;
        }
        if (disjoint)
            isSATPassed = false;
        else
        {
            float best = -PXG_MAX_F32;
            for (uint lane = 0; lane < 32; ++lane) best = max(best, laneSep[lane]);
            uint winner = 0;
            for (uint lane = 0; lane < 32; ++lane)
                if (laneSep[lane] == best) { winner = lane; break; }
            if (separation < best)
            {
                minNormal = laneNormal[winner];
                separation = best;
                featureIndex = laneFeature[winner];
            }
        }
    }

    // ---- satTriangleNormal (order-independent min/max) ----------------------
    if (isSATPassed)
    {
        const float3 n0 = triangleLocNormal;
        const float triProj = dot(n0, triLocVerts[0]);
        float min1 = PXG_MAX_F32;
        float max1 = -PXG_MAX_F32;
        for (uint i = 0; i < nbVerts; ++i)
        {
            const float dist = dot(scaledVerts[i], n0);
            max1 = max(dist, max1);
            min1 = min(dist, min1);
        }
        if ((min1 > triProj + contactDist) || (triProj > max1 + contactDist))
            isSATPassed = false;
        else
        {
            const float _separation = min1 - triProj;
            if (separation < _separation)
            {
                minNormal = -n0;
                separation = _separation;
                featureIndex = TRI_FEATURE_IDX;
            }
        }
    }
    // convexMeshSAT<false>: NO edge-edge stage for cloth

    uint faceIndex1 = featureIndex;
    uint faceIndex2 = 0xFFFFFFFFu;
    uint dbgFace2 = 0xFFFFFFFFu;

    if (isSATPassed)
    {
        // ---- selectPolygonFeatureIndex (via selectPolygonFeatureIndex2) -----
        if (featureIndex == TRI_FEATURE_IDX || featureIndex == EDGE_FEATURE_IDX)
        {
            // best-aligned convex face (strided argmax, lowest-lane ties)
            float laneProj[32];
            uint laneFace[32];
            for (uint lane = 0; lane < 32; ++lane)
            {
                laneProj[lane] = -PXG_MAX_F32;
                laneFace[lane] = 0;
                for (uint i = lane; i < nbPolygons; i += 32)
                {
                    const float proj = dot(minNormal, convexPlaneN[i]);
                    if (laneProj[lane] < proj)
                    {
                        laneProj[lane] = proj;
                        laneFace[lane] = i;
                    }
                }
            }
            float minProj = -PXG_MAX_F32;
            for (uint lane = 0; lane < 32; ++lane) minProj = max(minProj, laneProj[lane]);
            uint winner = 0;
            for (uint lane = 0; lane < 32; ++lane)
                if (laneProj[lane] == minProj) { winner = lane; break; }
            uint polyFaceIndex = laneFace[winner];

            // edge scan: best neighbor of the winning face + a possibly
            // better edge-normal axis (both faces of that edge win)
            float laneDpSq[32];   // maxDpSq per lane, seeded minProj^2
            uint laneEdge[32];    // closestEdge per lane
            float laneDpSq2[32];  // best neighbor metric per lane
            uint lanePoly2[32];   // per-lane polyIndex2 candidate
            for (uint lane = 0; lane < 32; ++lane)
            {
                laneDpSq[lane] = minProj * minProj;
                laneEdge[lane] = 0xFFFFFFFFu;
                laneDpSq2[lane] = -10.0f;
                lanePoly2[lane] = 0xFFFFFFFFu;
                for (uint i = lane; i < nbEdges; i += 32)
                {
                    const uint f0 = chEdgeFace(hull, hullHeader, i, 0u);
                    const uint f1 = chEdgeFace(hull, hullHeader, i, 1u);
                    const float3 edgeN = convexPlaneN[f0] + convexPlaneN[f1];
                    const float enMagSq = dot(edgeN, edgeN);
                    const float dp = dot(edgeN, minNormal);
                    const float sqDp = dp * dp;
                    if ((f0 == polyFaceIndex || f1 == polyFaceIndex) && (sqDp > laneDpSq2[lane]))
                    {
                        laneDpSq2[lane] = sqDp;
                        lanePoly2[lane] = f0 == polyFaceIndex ? f1 : f0;
                    }
                    if (dp > 0.0f && sqDp > (laneDpSq[lane] * enMagSq))
                    {
                        laneDpSq[lane] = sqDp / enMagSq;
                        laneEdge[lane] = i;
                    }
                }
            }
            float maxDpSq = -PXG_MAX_F32;
            for (uint lane = 0; lane < 32; ++lane) maxDpSq = max(maxDpSq, laneDpSq[lane]);
            uint winner2 = 0;
            for (uint lane = 0; lane < 32; ++lane)
                if (laneDpSq[lane] == maxDpSq) { winner2 = lane; break; }
            const uint closestEdge = laneEdge[winner2];

            if (closestEdge != 0xFFFFFFFFu)
            {
                const uint f0 = chEdgeFace(hull, hullHeader, closestEdge, 0u);
                const uint f1 = chEdgeFace(hull, hullHeader, closestEdge, 1u);
                const float dp0 = dot(convexPlaneN[f0], minNormal);
                const float dp1 = dot(convexPlaneN[f1], minNormal);
                if (dp0 > dp1) { polyFaceIndex = f0; faceIndex2 = f1; }
                else           { polyFaceIndex = f1; faceIndex2 = f0; }
            }
            else
            {
                float maxDpSq2 = -PXG_MAX_F32;
                for (uint lane = 0; lane < 32; ++lane) maxDpSq2 = max(maxDpSq2, laneDpSq2[lane]);
                uint winner3 = 0;
                for (uint lane = 0; lane < 32; ++lane)
                    if (laneDpSq2[lane] == maxDpSq2) { winner3 = lane; break; }
                faceIndex2 = lanePoly2[winner3];
            }
            faceIndex1 = polyFaceIndex;
        }

        dbgFace2 = faceIndex2; // captured before the loop consumes it
        // ---- two-face contact loop ------------------------------------------
        const float4 triPlane = float4(triangleLocNormal, -dot(triLocVerts[0], triangleLocNormal));

        for (uint faceIndex = faceIndex1; faceIndex != 0xFFFFFFFFu;)
        {
            const float4 referencePolygon_plane = chPlane(hull, hullHeader, faceIndex);
            float3 p0n = vertex2ShapeNormalVector(referencePolygon_plane.xyz, convexScale, convexScaleRot);
            float p0d = referencePolygon_plane.w;
            // PxPlane::normalize
            {
                // P9.47: PxPlane::normalize (PxPlane.h:126) divides by n.magnitude().
                const float denom = 1.0f / sqrt(dot(p0n, p0n));
                p0n *= denom;
                p0d *= denom;
            }
            const float4 plane0 = float4(p0n, p0d);

            const uint polyDesc = chPolyDesc(hull, hullHeader, faceIndex);
            const uint vref = pdVRef8(polyDesc);
            const uint nbFaceVerts = pdNbVerts(polyDesc);

            float3 verts0[32];
            for (uint l = 0; l < 32; ++l) verts0[l] = float3(0, 0, 0);
            for (uint l = 0; l < nbFaceVerts; ++l)
                verts0[l] = vertex2Shape(chVertex(hull, chPolyVertex(hull, hullHeader, vref, l)).xyz,
                                         convexScale, convexScaleRot);

            float ct[NUM_TMP_CONTACTS_PER_PAIR * 8];
            for (uint s = 0; s < NUM_TMP_CONTACTS_PER_PAIR * 8; ++s) ct[s] = 0.0f;

            uint numContactsGenerated = polyClip4(plane0, verts0, nbFaceVerts, triPlane, triLocVerts,
                                                  minNormal, contactDist, ct);

            if (numContactsGenerated != 0)
            {
                float3 pa[32];
                float sep[32];
                float4 bary[32];
                uint validMask = 0;
                for (uint lane = 0; lane < 32; ++lane)
                {
                    pa[lane] = float3(0, 0, 0);
                    sep[lane] = 0.0f;
                    bary[lane] = float4(0, 0, 0, 0);
                    if (lane < numContactsGenerated)
                    {
                        pa[lane] = float3(ct[8 * lane + 0], ct[8 * lane + 1], ct[8 * lane + 2]);
                        sep[lane] = ct[8 * lane + 3];
                        float4 b;
                        if (computeBarycentric(triLocVerts[0], triLocVerts[1], triLocVerts[2], pa[lane], b))
                            validMask |= 1u << lane;
                        bary[lane] = b;
                    }
                }

                numContactsGenerated = countbits(validMask);
                if (numContactsGenerated != 0)
                {
                    uint mask = validMask;
                    if (numContactsGenerated > 5)
                        mask = contactReduce5(pa, sep, minNormal, mask);

                    // PxgOutputIndex: ONE reservation for the whole batch,
                    // taken BEFORE the signed mask > 0 write gate (faithful)
                    uint startIndex = 0xFFFFFFFFu;
                    const uint numOut = countbits(mask);
                    if (numOut > 0)
                        InterlockedAdd(Ptr<uint>(w.count)[0], numOut, startIndex);

                    if (int(mask) > 0)
                    {
                        for (uint lane = 0; lane < 32; ++lane)
                        {
                            if ((mask & (1u << lane)) == 0) continue;
                            const uint ind = startIndex + warpScanExclusive(mask, lane);
                            const float3 worldP = quatRotate(cs.convexQ, pa[lane]) + cs.convexP.xyz;
                            const float3 worldNormal = quatRotate(cs.convexQ, float3(ct[8 * lane + 4], ct[8 * lane + 5], ct[8 * lane + 6]));
                            const uint2 pairInd0 = uint2(cs.ids.x, cs.ids.y); // rigidId
                            const uint pairInd1 = (cs.ids.z << PXG_BITSHIFT_ELEMENT_ID) | cs.ids.w;
                            writeRigidVsDeformableContact(w, ind, float4(worldP, rest),
                                float4(worldNormal, sep[lane]), bary[lane],
                                pairInd0, pairInd1, cs.ids2.x, pairInd0);
                        }
                    }
                }
            }
            faceIndex = faceIndex2;
            faceIndex2 = 0xFFFFFFFFu;
        }
    }

    return uint4(isSATPassed ? 1 : 0, featureIndex, faceIndex1, dbgFace2);
}
