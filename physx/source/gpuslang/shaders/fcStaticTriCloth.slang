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

// fcStaticTriCloth.slang
//
// Slang port of the static-triangle vs cloth-triangle contact path of
// femClothPrimitives.cu — the shared core of cloth_meshContactGenLaunch
// and cloth_heightfieldContactGenLaunch — plus the per-vertex box path
// of cloth_boxVertexContactGenLaunch:
//
// MODE 0 (cloth_meshContactGenLaunch per-pair core):
// - the all-three-verts filter skip: the triangle is dropped only when
//   find(filterPairs, rigidId, PxEncodeClothIndex(clothId, v)) hits for
//   ALL THREE cloth triangle vertices (same 3-key PxgRigidFilterPair
//   binary search as fcVertexConvex.slang, query index2 = 0);
// - the static mesh triangle brought into cloth space:
//   vertex2Shape(triV, trimeshScale) then trimeshToClothTransform =
//   clothTransform.transformInv(trimeshToWorld) applied to each vertex;
// - staticTriangleVsClothCollision:
//   * convexPoly[8] seeded with the three CLOTH triangle positions
//     (cloth-local mPosition_InvMass values, used untransformed);
//   * fixNormal = getNormalized((b0-a0) x (c0-a0)) of the STATIC
//     triangle, staticTrianglePlane = PxPlane(localA0, fixNormal)
//     (d = -p.n; distance = n.p + d);
//   * trimTriangleByTriangle (the 5-plane, 8-element variant): its OWN
//     normal n = cross(...) with PxVec3::normalizeSafe (n left UNCHANGED
//     below 1e-20); three edge planes createEdgePlane(n, e0, e1) with
//     n_e = getNormalized((e1-e0) x n), plus the two +-cDistance slab
//     planes PxPlane(triA +- cDistance*n, +-n).  Each trim() clips the
//     loop in place: points STRICTLY outside (distance > 0) are removed;
//     the walk records the first two sign-change edges (changeA/changeB),
//     lerps intersection points by |distA|/(|distA|+|distB|), and
//     rebuilds the loop — including the in-place insert() branch when
//     the intersection lands on the still-unshifted index — loops with
//     < 3 points pass through unchanged, and a crossing-free all-outside
//     loop collapses to 0;
//   * getDeepestPointSimplified with tolerance 0.01*cDistance: a running
//     min over plane.distance with a +-tolerance tie band — a strictly
//     deeper point (d < minDist - tol) RESTARTS the average, a near-tie
//     (d < minDist + tol) ACCUMULATES into the returned average without
//     updating minDist.  (The CUDA stores d in a double; both compare
//     operands are exactly-widened floats, so float compares are
//     bit-identical — ported in float.)  A fully clipped-away polygon
//     leaves minDist at 1e9 => no contact (the 0/count NaN is unused);
//   * hasContact = sep < cDistance STRICT; on contact: computeBarycentric
//     of the deepest point in the CLOTH triangle (bary written even when
//     outside the eps window; the bool is unused here), position =
//     clothTransform.transform(deepestPoint), normal =
//     clothTransform.rotate(fixNormal), separation = sep;
// - the write: globalScanExclusiveSingleWarp — one ballot over the warp,
//   exclusive-popc lane offsets, a SINGLE lane-0 atomicAdd reservation
//   (skipped when the warp has no contacts) — then capacity-gated
//   PxgFEMContactWriter::writeRigidVsDeformableContact32: point
//   (w = restDistance), normalPen (w = separation), barycentric, info
//   pairInd0 = rigidId, pairInd1 = PxEncodeClothIndex(clothId,
//   clothTriangleIdx) — a TRIANGLE element id — info fourth dword =
//   rigidMaterialIndex with the isInCollision bit 31 cleared, and the
//   32-bit sort lane: contactSortedByX[index] = PxU32(rigidId.getInd())
//   (low dword only), contactIndexSortedByX[index] = index.  The 64-bit
//   contactByX/tempContactByX keys are NOT written by this variant.
//
// MODE 1 (cloth_boxVertexContactGenLaunch per-vertex core):
// - single-vertex filter skip (one find());
// - spherebox with sphereRadius = 0 (sphereCollision.cuh, the
//   PxVec3-origin variant): the cloth vertex clamped into the box in box
//   space; inside-the-box picks the min-distance face (x-first <= tie
//   order) with pen = -distToSurface - 0, outside normalizes the
//   center-to-clamp vector with pen = |v|; both rotate the normal to
//   world; gate sqInflatedSum > lengthSq STRICT (inflation = 0 +
//   contactDist);
// - contact point = clothP - pen * normal, w = restDistance, barycentric
//   (0, 0, 0, 1) = VERTEX marker, pairInd1 = PxEncodeClothIndex(clothId,
//   vertexIdx); same ballot + single-reservation slot packing (the CUDA
//   uses an explicit __ballot_sync/warpScanExclusive/lane-0 atomicAdd —
//   identical semantics), but through the 64-bit-key
//   writeRigidVsDeformableContact (contactByX = rigidId u64,
//   tempContactByX = its low dword, contactIndexSortedByX = index).
//
// KNOWN LIMITS (verified textually, not pinnable by tests): the
// +cDistance slab plane in trimTriangleByTriangle is OUTPUT-NEUTRAL
// redundancy — any surviving deepest point already satisfies
// sep <= cDistance, and everything the slab would remove is rejected by
// the sep < cDistance gate (or never enters the +-0.01*cDistance
// averaging band of a deeper point); only cases with sep inside that
// band of cDistance could differ, which is exactly the tainted gate
// boundary.  The -cDistance slab IS live (it clamps deep penetration)
// and is pinned by the deep-penetration fixtures.  The normalizeSafe
// (trimTriangleByTriangle) vs getNormalized (fixNormal) distinction is
// pinnable only for 0 < |cross| < 1e-20 — a subnormal window that cannot
// survive the O(1) vertex transform chain (sub-ulp vertex differences
// collapse to exact zero, where both functions return the same zero) —
// so it is verified textually; the exact-zero degenerate path (zero
// normal, zero planes, centroid contact) IS pinned by fixtures.
//
// WARP SERIALIZATION: both launches are data-parallel one-CASE-per-thread
// kernels whose only cross-lane effect is the slot-packing ballot; a
// batch of 32 consecutive cases is serialized into one thread (lane
// order = ascending case index within the batch, one reservation per
// batch), which is output-identical per the established doctrine.
//
// Engine glue not ported (documented): LoadShapePair/transform caches,
// the shape0/shape1 cloth-identification branch of
// cloth_meshContactGenLaunch (either shape order resolved via
// particleOrSoftbodyId != 0xffffffff — cases carry the resolved roles),
// PxgFEMCloth loads, the trimesh blob walk (uint4 header + BV32 node
// skip to verts/indices — cases carry the triangle verts directly), the
// pair-stack iteration and the stackSizeNeededOnDevice atomicMax.
// The heightfield flavour cloth_heightfieldContactGenLaunch differs from
// mode 0 only upstream of this core: the static triangle comes from
// getTriangle (hf sample decode — ported/tested in npHfTriangle.slang,
// tranche 60), there is NO filter skip and no vertex2Shape, and the
// hf->cloth transform is built the same way — documented as composition
// glue.  Two sibling kernels are DEAD upstream and documented, not
// ported: cloth_meshVertexContactGenLaunch and
// cloth_heightfieldVertexContactGenLaunch both declare
// `bool hasContacts = false;` and the only assignment is commented out
// (`//hasContacts = true;` inside the accept gate), so their ballot is
// always 0 and their writers are unreachable.  (Their bodies also carry
// pre-dead quirks: p0 and p1 load the SAME position array, pen is
// computed via sqrt then immediately redefined as ap1.n, and the
// backface test dots a mesh-VERTEX-space triangle normal against a
// WORLD-space direction.)

import pxgSlangCommon;

static const uint PXG_BITSHIFT_ELEMENT_ID = 20; // PxgSimulationCoreDesc.h
static const uint COLLISION_FLAG_MASK = 1u << 31;
static const float PX_NORMALIZATION_EPSILON = 1e-20f;

struct FstParams { uint numCases; uint maxContacts; uint nbFilterPairs; uint mode; }

struct FstCase
{
    float4 triA;     // mode 0: static tri vert A (trimesh VERTEX space); mode 1: cloth vertex P (world)
    float4 triB;     // mode 0: static tri vert B; mode 1: box half extents (xyz)
    float4 triC;     // mode 0: static tri vert C
    float4 clothQ;   // mode 0: cloth transform quat;   mode 1: rigid (box) transform quat
    float4 clothP;   // mode 0: cloth transform p, w = contactDist; mode 1: rigid p, w = contactDist
    float4 meshQ;    // mode 0: trimesh transform quat
    float4 meshP;    // mode 0: trimesh transform p, w = restDist; mode 1: w = restDist
    float4 scale;    // mode 0: trimesh mesh scale (xyz)
    float4 scaleRot; // mode 0: trimesh scale rotation quat
    uint4 ids;       // x = rigidId lo, y = rigidId hi, z = clothId, w = clothTriangleIdx (mode 0) / vertexIdx (mode 1)
    uint4 ids2;      // x = rigidMaterialIndex, y/z/w = cloth triangle vertex indices (mode 0)
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
ConstantBuffer<FstParams> gParams;

RWStructuredBuffer<FstCase> gCase;        // binding 0
RWStructuredBuffer<float4> gClothPos;     // binding 1: cloth mPosition_InvMass
RWStructuredBuffer<FilterPair> gFilter;   // binding 2 (sorted)
// PxgFEMContactWriter
RWStructuredBuffer<float4> gFemPoint;     // binding 3
RWStructuredBuffer<float4> gFemNormalPen; // binding 4
RWStructuredBuffer<float4> gFemBary;      // binding 5
RWStructuredBuffer<uint4> gFemInfo;       // binding 6
RWStructuredBuffer<uint> gFemCount;       // binding 7
RWStructuredBuffer<uint2> gFemByX;        // binding 8: contactByX (u64 keys, mode 1)
RWStructuredBuffer<uint> gFemTempByX;     // binding 9
RWStructuredBuffer<uint> gFemIdxByX;      // binding 10
RWStructuredBuffer<uint> gFemSorted32;    // binding 11: contactSortedByX (u32, mode 0)
RWStructuredBuffer<float4> gDebug;        // binding 12: (skipped, hasContact, sep|pen, polyCorners)

// ---- shared small helpers ----------------------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}
// PxPlane(point, normal): n, d = -p.n ; distance(p) = n.p + d
float4 makePlane(float3 p, float3 n) { return float4(n, -dot(p, n)); }
float planeDistance(float4 pl, float3 p) { return dot(pl.xyz, p) + pl.w; }

float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
float3 transformPt(float4 q, float3 p, float3 v) { return quatRotate(q, v) + p; }

// femClothPrimitives.cu computeBarycentric (bool result unused here)
void computeBarycentric(float3 a, float3 b, float3 c, float3 p, out float4 barycentric)
{
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
}

// ---- find(PxgRigidFilterPair*) (deformableElementFilter.cuh) ----------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

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

// ---- polygon trim pipeline (femClothPrimitives.cu 3436-3641) ----------------

// Only correct when a and b straddle the plane (one may lie exactly on it)
float3 intersectionPoint(float4 plane, float3 a, float3 b)
{
    const float distA = abs(planeDistance(plane, a));
    const float distB = abs(planeDistance(plane, b));
    const float f = distA / (distA + distB);
    return (1.0f - f) * a + f * b;
}

void insertPt(inout float3 arr[8], int arrCount, int index, float3 element)
{
    for (int i = arrCount; i > index; --i)
        arr[i] = arr[i - 1];
    arr[index] = element;
}

// In-place convex-loop clip: points with distance > 0 (STRICT) are removed
int trimLoop(float4 plane, inout float3 loop[8], int loopCount)
{
    if (loopCount < 3)
        return loopCount;

    float3 intersectionA = float3(0, 0, 0);
    int changeA = -1;
    float3 intersectionB = float3(0, 0, 0);
    int changeB = -1;

    bool keep = false;

    bool prevOutside = planeDistance(plane, loop[0]) > 0.0f;
    for (int i = 0; i < loopCount; ++i)
    {
        const bool outside = planeDistance(plane, loop[(i + 1) % loopCount]) > 0.0f;
        if (outside != prevOutside)
        {
            const float3 inter = intersectionPoint(plane, loop[i], loop[(i + 1) % loopCount]);
            if (changeA < 0)
            {
                changeA = i;
                keep = !prevOutside;
                intersectionA = inter;
            }
            else
            {
                changeB = i;
                intersectionB = inter;
            }
        }
        prevOutside = outside;
    }

    if (changeA >= 0 && changeB >= 0)
    {
        int loopIndexer = -1;
        for (int i = 0; i < loopCount; ++i)
        {
            if (keep)
            {
                ++loopIndexer;
                loop[loopIndexer] = loop[i];
            }
            if (i == changeA || i == changeB)
            {
                const float3 pt = (i == changeA) ? intersectionA : intersectionB;
                if (loopIndexer == i && !keep)
                {
                    ++loopIndexer;
                    insertPt(loop, loopCount, loopIndexer, pt);
                    ++loopCount;
                    ++i;
                    ++changeB;
                }
                else
                {
                    ++loopIndexer;
                    loop[loopIndexer] = pt;
                }
                keep = !keep;
            }
        }
        loopCount = loopIndexer + 1;
    }
    else if (prevOutside)
        loopCount = 0; // no crossing and all points outside

    return loopCount;
}

float4 createEdgePlane(float3 triNormal, float3 edgeStart, float3 edgeEnd)
{
    const float3 n = getNormalized(cross(edgeEnd - edgeStart, triNormal));
    return makePlane(edgeStart, n);
}

// the 5-plane, 8-element variant (3 edge planes + two +-distance slabs)
int trimTriangleByTriangle(inout float3 poly[8], int polyLength, float3 triA, float3 triB, float3 triC,
                           float maxDistanceLimitPosNormalDir, float maxDistanceLimitNegNormalDir)
{
    float3 n = cross(triB - triA, triC - triA);
    // PxVec3::normalizeSafe: n left UNCHANGED below the epsilon
    const float mag = sqrt(dot(n, n));
    if (mag >= PX_NORMALIZATION_EPSILON)
        n = n * (1.0f / mag);

    polyLength = trimLoop(createEdgePlane(n, triA, triB), poly, polyLength);
    polyLength = trimLoop(createEdgePlane(n, triB, triC), poly, polyLength);
    polyLength = trimLoop(createEdgePlane(n, triC, triA), poly, polyLength);

    polyLength = trimLoop(makePlane(triA + maxDistanceLimitPosNormalDir * n, n), poly, polyLength);
    polyLength = trimLoop(makePlane(triA - maxDistanceLimitNegNormalDir * n, -n), poly, polyLength);

    return polyLength;
}

// running-min with a +-tolerance averaging band (see header notes; the
// CUDA double is a no-op over exactly-widened float operands)
float3 getDeepestPointSimplified(float3 poly[8], int polyLength, float4 plane, float tolerance,
                                 out float minDist)
{
    minDist = 1000000000.0f;
    float3 deepestPoint = float3(0, 0, 0);
    int count = 0;
    for (int i = 0; i < polyLength; ++i)
    {
        const float d = planeDistance(plane, poly[i]);
        if (d < minDist - tolerance)
        {
            minDist = d;
            deepestPoint = poly[i];
            count = 1;
        }
        else if (d < minDist + tolerance)
        {
            deepestPoint += poly[i];
            count++;
        }
    }
    deepestPoint /= float(count);
    return deepestPoint;
}

struct FemContact
{
    float3 position;
    float3 normal;
    float separation;
    float4 barycentric;
}

bool staticTriangleVsClothCollision(float3 localA0, float3 localB0, float3 localC0,
                                    float3 a1, float3 b1, float3 c1,
                                    float4 clothQ, float3 clothP, float cDistance,
                                    out FemContact contact, out int dbgCorners)
{
    float3 convexPoly[8];
    convexPoly[0] = a1;
    convexPoly[1] = b1;
    convexPoly[2] = c1;
    for (int k = 3; k < 8; ++k)
        convexPoly[k] = float3(0, 0, 0);

    const float3 fixNormal = getNormalized(cross(localB0 - localA0, localC0 - localA0));
    const float4 staticTrianglePlane = makePlane(localA0, fixNormal);

    const int numPolyCorners = trimTriangleByTriangle(convexPoly, 3, localA0, localB0, localC0, cDistance, cDistance);
    dbgCorners = numPolyCorners;

    float sep;
    const float3 deepestPoint = getDeepestPointSimplified(convexPoly, numPolyCorners, staticTrianglePlane,
                                                          0.01f * cDistance, sep);

    const bool hasContact = sep < cDistance;

    contact.position = float3(0, 0, 0);
    contact.normal = float3(0, 0, 0);
    contact.separation = 0.0f;
    contact.barycentric = float4(0, 0, 0, 0);
    if (hasContact)
    {
        computeBarycentric(a1, b1, c1, deepestPoint, contact.barycentric);
        contact.position = transformPt(clothQ, clothP, deepestPoint);
        contact.normal = quatRotate(clothQ, fixNormal);
        contact.separation = sep;
    }
    return hasContact;
}

// ---- spherebox (sphereCollision.cuh, PxVec3-origin variant), radius 0 -------
uint sphereboxNormalPen(float3 sphereOrigin, float4 boxQ, float3 boxP, float sphereRadius,
                        float3 boxExtents, float contactDist, out float4 outNormalPen)
{
    outNormalPen = float4(0, 0, 0, 0);
    const float3 sphereCenter = quatRotateInv(boxQ, sphereOrigin - boxP);
    const float3 nBoxExtents = -boxExtents;

    const float inflatedSum = sphereRadius + contactDist;
    const float sqInflatedSum = inflatedSum * inflatedSum;

    const float x = clamp(sphereCenter.x, nBoxExtents.x, boxExtents.x);
    const float y = clamp(sphereCenter.y, nBoxExtents.y, boxExtents.y);
    const float z = clamp(sphereCenter.z, nBoxExtents.z, boxExtents.z);
    const float3 p = float3(x, y, z);
    const float3 v = sphereCenter - p;
    const float lengthSq = dot(v, v);

    if (sqInflatedSum > lengthSq)
    {
        const bool bX = boxExtents.x >= abs(sphereCenter.x);
        const bool bY = boxExtents.y >= abs(sphereCenter.y);
        const bool bZ = boxExtents.z >= abs(sphereCenter.z);
        const bool bInsideBox = bX && bY && bZ;
        if (bInsideBox)
        {
            const float3 absP = float3(abs(p.x), abs(p.y), abs(p.z));
            const float3 distToSurface = boxExtents - absP;

            float3 localNorm = p.z >= 0.0f ? float3(0, 0, 1) : float3(0, 0, -1);
            float dist = -distToSurface.z;
            if (distToSurface.x <= distToSurface.y && distToSurface.x <= distToSurface.z)
            {
                localNorm = p.x >= 0.0f ? float3(1, 0, 0) : float3(-1, 0, 0);
                dist = -distToSurface.x;
            }
            else if (distToSurface.y <= distToSurface.z)
            {
                localNorm = p.y >= 0.0f ? float3(0, 1, 0) : float3(0, -1, 0);
                dist = -distToSurface.y;
            }

            const float3 normal = quatRotate(boxQ, localNorm);
            const float penetration = dist - sphereRadius;
            outNormalPen = float4(normal, penetration);
        }
        else
        {
            const float recipLength = rsqrt(lengthSq);
            const float len = 1.0f / recipLength;
            const float3 locNorm = v * recipLength;
            const float penetration = len - sphereRadius;
            const float3 normal = quatRotate(boxQ, locNorm);
            outNormalPen = float4(normal, penetration);
        }
        return 1;
    }
    return 0;
}

// ---- PxgFEMContactWriter variants (PxgFEMCore.h) ----------------------------
// writeRigidVsDeformableContact32: 32-bit sort lane, no u64 key
void writeRigidVsDeformableContact32(uint index, float4 contact, float4 normalPen, float4 bary,
                                     uint2 pairInd0, uint pairInd1, uint rigidMaterialIndex,
                                     uint contactSortedByRigid)
{
    if (index >= gParams.maxContacts) return;
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
    gFemSorted32[index] = contactSortedByRigid;
    gFemIdxByX[index] = index;
}

// writeRigidVsDeformableContact: 64-bit sort key
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
// One thread serializes one 32-case warp batch (lane order = case order);
// the batch does ONE reservation for all its contacts, like the CUDA
// warp's globalScanExclusiveSingleWarp / lane-0 atomicAdd.
[shader("compute")]
[numthreads(64, 1, 1)]
void fcStaticTriClothKernel(uint3 dtid : SV_DispatchThreadID)
{
    const uint batch = dtid.x;
    const uint base = batch * 32;
    if (base >= gParams.numCases) return;

    bool laneHas[32];
    float4 lanePoint[32];
    float4 laneNormalPen[32];
    float4 laneBary[32];
    uint2 lanePair0[32];
    uint lanePair1[32];
    uint laneMat[32];
    uint laneSorted[32];

    uint totalCount = 0;
    for (uint lane = 0; lane < 32; ++lane)
    {
        const uint i = base + lane;
        laneHas[lane] = false;
        lanePoint[lane] = float4(0, 0, 0, 0);
        laneNormalPen[lane] = float4(0, 0, 0, 0);
        laneBary[lane] = float4(0, 0, 0, 0);
        lanePair0[lane] = uint2(0, 0);
        lanePair1[lane] = 0;
        laneMat[lane] = 0;
        laneSorted[lane] = 0;
        if (i >= gParams.numCases)
            continue;

        const FstCase cs = gCase[i];
        const uint2 rigidId = cs.ids.xy;
        const uint clothId = cs.ids.z;
        const float cDistance = cs.clothP.w;
        const float restDist = cs.meshP.w;

        if (gParams.mode == 0)
        {
            const uint clothTriangleIdx = cs.ids.w;
            const uint3 vi = cs.ids2.yzw;

            const uint clothMask0 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | vi.x;
            const uint clothMask1 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | vi.y;
            const uint clothMask2 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | vi.z;

            const bool skipTriangle = findFilter(rigidId, clothMask0) &&
                                      findFilter(rigidId, clothMask1) &&
                                      findFilter(rigidId, clothMask2);
            if (skipTriangle)
            {
                gDebug[i] = float4(1, 0, 0, 0);
                continue;
            }

            // trimeshToClothTransform = clothT.transformInv(trimeshT)
            const PxgTransform28 clothT = makeTransform28(cs.clothQ, cs.clothP.xyz);
            const PxgTransform28 meshT = makeTransform28(cs.meshQ, cs.meshP.xyz);
            const PxgTransform28 meshToCloth = transformInvTransform(clothT, meshT);

            const float3 localA0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                               vertex2Shape(cs.triA.xyz, cs.scale.xyz, cs.scaleRot));
            const float3 localB0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                               vertex2Shape(cs.triB.xyz, cs.scale.xyz, cs.scaleRot));
            const float3 localC0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                               vertex2Shape(cs.triC.xyz, cs.scale.xyz, cs.scaleRot));

            const float3 a1 = gClothPos[vi.x].xyz;
            const float3 b1 = gClothPos[vi.y].xyz;
            const float3 c1 = gClothPos[vi.z].xyz;

            FemContact c;
            int dbgCorners;
            const bool hasContact = staticTriangleVsClothCollision(localA0, localB0, localC0, a1, b1, c1,
                                                                   cs.clothQ, cs.clothP.xyz, cDistance,
                                                                   c, dbgCorners);
            gDebug[i] = float4(0, hasContact ? 1 : 0, c.separation, float(dbgCorners));
            if (hasContact)
            {
                laneHas[lane] = true;
                lanePoint[lane] = float4(c.position, restDist);
                laneNormalPen[lane] = float4(c.normal, c.separation);
                laneBary[lane] = c.barycentric;
                lanePair0[lane] = rigidId;
                lanePair1[lane] = (clothId << PXG_BITSHIFT_ELEMENT_ID) | clothTriangleIdx;
                laneMat[lane] = cs.ids2.x;
                laneSorted[lane] = rigidId.x; // PxU32(rigidId.getInd())
                ++totalCount;
            }
        }
        else
        {
            const uint vertexIdx = cs.ids.w;
            const uint clothMask = (clothId << PXG_BITSHIFT_ELEMENT_ID) | vertexIdx;
            const bool skipVertex = findFilter(rigidId, clothMask);
            if (skipVertex)
            {
                gDebug[i] = float4(1, 0, 0, 0);
                continue;
            }

            float4 normalPen;
            const uint numContacts = sphereboxNormalPen(cs.triA.xyz, cs.clothQ, cs.clothP.xyz, 0.0f,
                                                        cs.triB.xyz, cDistance, normalPen);
            gDebug[i] = float4(0, float(numContacts), normalPen.w, 0);
            if (numContacts != 0)
            {
                const float3 rigidP = cs.triA.xyz - normalPen.w * normalPen.xyz;
                laneHas[lane] = true;
                lanePoint[lane] = float4(rigidP, restDist);
                laneNormalPen[lane] = normalPen;
                laneBary[lane] = float4(0, 0, 0, 1); // vertex contact marker
                lanePair0[lane] = rigidId;
                lanePair1[lane] = clothMask;
                laneMat[lane] = cs.ids2.x;
                ++totalCount;
            }
        }
    }

    // one reservation per batch, skipped when the ballot is empty
    uint startIndex = 0;
    if (totalCount > 0)
        InterlockedAdd(gFemCount[0], totalCount, startIndex);

    uint offset = 0;
    for (uint lane = 0; lane < 32; ++lane)
    {
        if (!laneHas[lane])
            continue;
        const uint index = startIndex + offset;
        ++offset;
        if (gParams.mode == 0)
            writeRigidVsDeformableContact32(index, lanePoint[lane], laneNormalPen[lane], laneBary[lane],
                                            lanePair0[lane], lanePair1[lane], laneMat[lane], laneSorted[lane]);
        else
            writeRigidVsDeformableContact(index, lanePoint[lane], laneNormalPen[lane], laneBary[lane],
                                          lanePair0[lane], lanePair1[lane], laneMat[lane], lanePair0[lane]);
    }
}
