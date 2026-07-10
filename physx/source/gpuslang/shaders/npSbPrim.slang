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

// npSbPrim.slang
//
// Slang port of the non-GJK contact-generation cores of
// softbodyPrimitives.cu (tranche 100):
//
// sbTetPlane — the PLANE branch of sb_primitiveContactGenLaunch:
// the caller's per-pair attachment filter find(PxgRigidFilterPair*,
// rigidId, PxEncodeSoftBodyIndex(sbId, tetIdx)) (3-key compare,
// ignoreIndex2 accept, query index2 = 0), then tetPlaneCollide: each of
// the tet's 4 world verts taken to plane space (transformInv); accept is
// INCLUSIVE `contactDist >= pInPlaneSpace.x`; worldPoint = the
// UNTRANSFORMED vertex, worldNormal = planeTransform.q.getBasisVector0()
// (closed form), pen = the RAW plane-space x (no contactDist/restDist
// subtraction).  One globalScanExclusiveSingleWarp reservation per warp
// (= per pair; ballot + exclusive popc + lane-0 atomicAdd skipped when
// empty), caller-side capacity gate `index < maxNumContacts` STRICT plus
// the method's own `>= maxNumContacts` gate (double gate, both kept).
// writeRigidVsDeformableContactNoBarycentric: contactByX = rigidId u64,
// tempContactByX = its low dword, contactIndexSortedByX = index, point
// (w = restDist), normalPen (w = pen), info pairInd0/pairInd1 and the
// material dword via the two-step RMW (setRigidMaterialIndex preserving
// bit 31, then markInCollision(false) clearing it => net mat &
// 0x7FFFFFFF); the BARYCENTRIC buffer is NOT touched.
//
// sbEmit — writeContactNoBarycentricFullWarp (:297-362), the shared
// emitter of the four (deferred) GJK/EPA kernels, ported and verified now
// against synthetic GjkOutput fixtures so those kernels only await the
// GJK core: ONE UNGATED atomicAdd(+1) per contact (the counter
// intentionally overshoots capacity; downstream clamps), sentinel when
// `index >= maxNumContacts` (INCLUSIVE) => nothing written; pen =
// dot(closestPointB - closestPointA, normal) + penOffset (the CUDA sums
// three per-lane products via shfl — a plain dot in serial form); point =
// (closestPointB, restDist); normalPen = (flipOutputNormal ? -n : n,
// pen); alternativeWriteMode writes ONLY contactSortedByX =
// PxU32(pairInd0 & 0xffffffff) while the normal mode writes contactByX =
// pairInd0 (u64) + tempContactByX = its low dword — the respective other
// lanes stay UNTOUCHED; contactIndexSortedByX = index in both; info via
// the same two-step RMW.  Call-site parameterization to preserve for the
// deferred kernels: :567 penOffset = -radius, flip = false, normal mode;
// :888/:1135 penOffset = 0, flip = true, altMode (with :888 gating on
// direction.dot(n) >= 0 against the reversed-winding triangle normal).
// The CUDA broadcasts the writes from lanes 0-3 of the warp; the
// serialized port collapses the redundant all-lane stores to single
// stores (benign-race removal, output-identical).
//
// sbParticle — sbParticleCollision + sb_psContactGenLaunch (:1962-2128),
// a plain PER-THREAD grid-stride kernel (the "each warp" comment upstream
// is stale): the particle filter chain — the u64 particle mask
// PxEncodeParticleIndex(sysId, UNSORTED index) is passed into a PxU32
// parameter (UPSTREAM IMPLICIT TRUNCATION to the unsorted particle index;
// reproduced deliberately), findRange over mParticleBufferRunsum picks
// the sim-buffer bucket whose mUniqueId becomes the 3rd filter key
// (ignoreIndex2 = false), queried once with tetFullMask =
// PxEncodeSoftBodyIndex(sbId, PX_MAX_NB_DEFORMABLE_VOLUME_TET) and once
// with the exact tet mask; surfaceMask == 0 exits first.  Geometry:
// PointOutsideOfPlane4 (four sign products; INSIDE iff all four >= 0,
// INCLUSIVE), closestPtTetrahedron restricted to the surface-hint faces
// (identical to the npSbTet.slang port), computeTetBarycentric (eps 1e-5,
// per-component CLAMP after the bool), n = -(p - closestP); INSIDE path:
// pen = n.dot(n2) - restDistance with the UNNORMALIZED n, n = n2, and NO
// contact-distance gate (an inside particle beyond cDistance still
// emits); OUTSIDE path: reject sqDist > cDistance^2 STRICT, pen =
// n.normalize() (guarded m > 0 divide) minus restDistance, direction
// replaced by n2 when sqDist < 1e-8 OR n.dot(n2) < 0.8 (both STRICT).
// Per-contact UNGATED atomicAdd + `>= max` drop, then
// PxgFEMContactWriter::writeContact (barycentric overload): contactByX =
// PxEncodeParticleIndex(sysId, SORTED index) u64, tempContactByX = its
// low dword, contactIndexSortedByX = index, point = (p, 0), normalPen =
// (n, pen), barycentric = tet bary, info pairInd0 = the same u64,
// pairInd1 = tet mask — the material/isInCollision dword is NOT written
// (preserved).
//
// sbClothVert — sbClothVertCollision + sb_clothVertContactGenLaunch
// (:2449-2620), per-thread: TRIPLE PxgNonRigidFilterPair filter
// (softBodyFullMask+clothMask, softBodyMask+clothFullMask,
// softBodyMask+clothMask; 2-arg form, index2 = 0, ignoreIndex2 = true),
// surfaceMask != 0 gate, same tet geometry, v = closestP - p with m =
// v.normalize(); INSIDE: n = n2, pen = -m; OUTSIDE: reject sqDist >
// cDistance^2 STRICT, pen = m, n = (m > 1e-10f && v.dot(n2) > 0.707f) ?
// v : n2 (both STRICT; NOTE upstream has NO isValidTet gate — a
// degenerate tet propagates NaN barycentrics into the output).
// Per-contact UNGATED atomicAdd + inclusive `>= max` sentinel;
// PxgSoftBodyContactWriter::writeContact: point = (closestP, 0),
// normalPen = (n, pen), barycentric0 = (0,0,0,1) VERTEX marker,
// barycentric1 = tet bary, info pairInd0 = PxEncodeClothIndex(clothId,
// vertIdx) WIDENED u32->u64, pairInd1 = PxEncodeSoftBodyIndex — cloth is
// pairInd0 (REVERSED vs the PxgFEMCore.h field comment); the auxInd
// dword is untouched, and there are no sort-key buffers in this writer
// at all.
//
// KNOWN LIMITS: the caller-side capacity gate of sbTetPlane is not
// independently pinnable — the writer method re-checks the identical
// bound, so loosening the caller gate is masked (upstream redundancy;
// both gates are kept, verified textually).  sbClothVert's `m > 1e-10f`
// guard is likewise unpinnable: an outside classification with m below
// 1e-10 is reachable only through float rounding noise (verified
// textually).  computeTetBarycentric's post-window clamps are
// transcribed but unobservable through these kernels (closestP always
// lies on the tet surface, so raw barycentrics exceed [0,1] only by
// float noise below the compare tolerance).
//
// WARP SERIALIZATION: sbTetPlane serializes one warp-pair per thread
// (lane order = vert order, one reservation per pair); sbEmit serializes
// the lane-0 reservation + lane-0..3 stores of the warp emitter;
// sbParticle/sbClothVert are genuinely per-thread upstream.
//
// Engine glue not ported (documented): LoadShapePair/transform caches,
// PxgSoftBody/PxgFEMCloth/PxgParticleSystem loads (cases carry verts,
// poses, masks and ids), the 12-lane shared-memory vert staging, the
// pair-stack iteration and stackSizeNeededOnDevice atomicMax.  DEFERRED
// (GJK/EPA, standing): the sphere/capsule/box/convex arm of
// sbPrimitiveCollision (tetPrimitivesCollide2), sbMeshCollision,
// sbHeightfieldCollision and sbClothCollision — their shared emitter is
// sbEmit above.  DEAD upstream, intentionally NOT ported (zero callers,
// grep-verified): isTetrahedronValid (:110), computeSurfaceNormal (:160,
// which also carries stale-`n` reuse bugs and an uninitialized return
// for hint == 0), getClosestPtPointTetrahedron (:1763), checkNormals
// (:1823).

import pxgSlangCommon;

static const uint PXG_BITSHIFT_TET_ID = 20;      // PxgSimulationCoreDesc.h
static const uint PXG_BITSHIFT_ELEMENT_ID = 20;
static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffff;
static const uint PX_MAX_NB_DEFORMABLE_SURFACE_VTX = 0x000fffff;
static const uint COLLISION_FLAG_MASK = 1u << 31;
static const float PXG_MAX_F32_LOCAL = 3.402823466e38f;

struct SbParams { uint numCases; uint maxContacts; uint nbRFilter; uint nbNRFilter; uint nbRunsum; uint pad0, pad1, pad2; }

struct SbCase
{
    float4 a, b, c, d; // tet verts (a.w = contactDist, b.w = restDist) | sbEmit: closestPointA/closestPointB/normal, d = (penOffset, restDist, -, -)
    float4 p;          // particle / cloth-vertex position | sbTetPlane: plane transform position
    float4 q;          // sbTetPlane: plane transform quaternion | sbEmit: unused
    uint4 ids;         // sbTetPlane: (rigidLo, rigidHi, softbodyId, tetIdx) | sbEmit: (pairInd0 lo, pairInd0 hi, pairInd1, materialIndex)
                       // sbParticle: (softbodyId, tetIdx, particleSystemId, sortedParticleIndex) | sbClothVert: (softbodyId, tetIdx, clothId, vertIndex)
    uint4 ids2;        // sbTetPlane: (materialIndex, -, -, -) | sbEmit: (intersect, flip, altMode, -)
                       // sbParticle: (surfaceMask, unsortedParticleIndex, -, -) | sbClothVert: (surfaceMask, -, -, -)
}

// PxgRigidFilterPair (32B): u64 index0, u64 index1, u32 index2, pad
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0, pad1, pad2; }
// PxgNonRigidFilterPair (16B): u64 index0, u32 index1, u32 index2
struct PairNR { uint2 index0; uint index1; uint index2; }

[[vk::push_constant]]
ConstantBuffer<SbParams> gParams;

RWStructuredBuffer<SbCase> gCase;         // binding 0
RWStructuredBuffer<PairR> gRFilter;       // binding 1 (sorted)
RWStructuredBuffer<PairNR> gNRFilter;     // binding 2 (sorted)
RWStructuredBuffer<uint> gRunsum;         // binding 3: mParticleBufferRunsum
RWStructuredBuffer<uint> gBufUid;         // binding 4: per-buffer mUniqueId
// PxgFEMContactWriter
RWStructuredBuffer<float4> gFemPoint;     // binding 5
RWStructuredBuffer<float4> gFemNormalPen; // binding 6
RWStructuredBuffer<float4> gFemBary;      // binding 7
RWStructuredBuffer<uint4> gFemInfo;       // binding 8
RWStructuredBuffer<uint> gFemCount;       // binding 9
RWStructuredBuffer<uint2> gFemByX;        // binding 10
RWStructuredBuffer<uint> gFemTempByX;     // binding 11
RWStructuredBuffer<uint> gFemIdxByX;      // binding 12
RWStructuredBuffer<uint> gFemSorted32;    // binding 13
// PxgSoftBodyContactWriter (no sort keys)
RWStructuredBuffer<float4> gSbPoint;      // binding 14
RWStructuredBuffer<float4> gSbNormalPen;  // binding 15
RWStructuredBuffer<float4> gSbBary0;      // binding 16
RWStructuredBuffer<float4> gSbBary1;      // binding 17
RWStructuredBuffer<uint4> gSbInfo;        // binding 18
RWStructuredBuffer<uint> gSbCount;        // binding 19
RWStructuredBuffer<float4> gDebug;        // binding 20

// ---- small helpers -----------------------------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}
// PxVec3::normalize(): guarded in-place divide, returns the magnitude
float pxNormalize(inout float3 v)
{
    const float m = sqrt(dot(v, v));
    if (m > 0.0f)
        v = v / m;
    return m;
}
// PxQuat::getBasisVector0 closed form
float3 getBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}
uint encodeSoftBody(uint sbId, uint tetId) { return (sbId << PXG_BITSHIFT_TET_ID) | tetId; }
uint encodeCloth(uint clothId, uint elemId) { return (clothId << PXG_BITSHIFT_ELEMENT_ID) | elemId; }

bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// find(PxgRigidFilterPair*) — deformableElementFilter.cuh, query index2 = 0
bool findRigidFilter(uint2 rigidId, uint mask)
{
    const uint2 q1 = uint2(mask, 0);
    uint l = 0, r = gParams.nbRFilter;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const PairR p = gRFilter[mid];
        int cmp;
        if (!u64Eq(rigidId, p.index0))
            cmp = u64Lt(rigidId, p.index0) ? -1 : 1;
        else if (!u64Eq(q1, p.index1))
            cmp = u64Lt(q1, p.index1) ? -1 : 1;
        else if (0u != p.index2)
            cmp = -1;
        else
            cmp = 0;
        if (cmp == 0 || (u64Eq(rigidId, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid;
        else l = mid + 1;
    }
    return false;
}

// PxgNonRigidFilterPair::compare — (index0 u64, index1 u32, index2 u32)
bool findNonRigid(uint2 q0, uint q1, uint q2, bool ignoreIndex2)
{
    uint l = 0, r = gParams.nbNRFilter;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const PairNR p = gNRFilter[mid];
        int cmp;
        if (!u64Eq(q0, p.index0))
            cmp = u64Lt(q0, p.index0) ? -1 : 1;
        else if (q1 != p.index1)
            cmp = (q1 < p.index1) ? -1 : 1;
        else if (q2 != p.index2)
            cmp = (q2 < p.index2) ? -1 : 1;
        else
            cmp = 0;
        if (cmp == 0 || (ignoreIndex2 && u64Eq(q0, p.index0) && q1 == p.index1))
            return true;
        if (cmp < 0) r = mid;
        else l = mid + 1;
    }
    return false;
}

// findRange: upper-bound search returning r-1
uint findRange(uint value, uint length)
{
    if (length == 0)
        return 0;
    uint l = 0, r = length;
    while (l < r)
    {
        const uint m = (l + r) / 2;
        if (gRunsum[m] > value)
            r = m;
        else
            l = m + 1;
    }
    return r - 1;
}

// find(PxgParticleSystem&, ...): the compressedParticleId parameter is PxU32 —
// callers pass the u64 particle mask, which TRUNCATES to the unsorted particle
// index (upstream implicit conversion, reproduced); the bucket's mUniqueId
// becomes the third key with ignoreIndex2 = false
bool findParticleFilter(uint compressedParticleId, uint compressedId1)
{
    const uint particleId = compressedParticleId; // PxGetParticleIndex of the u32
    const uint id = findRange(particleId, gParams.nbRunsum);
    const uint uniqueId = gBufUid[id];
    return findNonRigid(uint2(compressedParticleId, 0), compressedId1, uniqueId, false);
}

// ---- geometry (softbodyPrimitives.cu / deformableCollision.cuh) --------------
static inline float4 PointOutsideOfPlane4(float3 p, float3 a, float3 b, float3 c, float3 d)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ad = d - a;

    const float3 v0 = cross(ab, ac);
    const float signa0 = dot(v0, ap);
    const float signd0 = dot(v0, ad);

    const float3 v1 = cross(ac, ad);
    const float signa1 = dot(v1, ap);
    const float signd1 = dot(v1, ab);

    const float3 v2 = cross(ad, ab);
    const float signa2 = dot(v2, ap);
    const float signd2 = dot(v2, ac);

    const float3 bd = d - b;
    const float3 bc = c - b;
    const float3 v3 = cross(bd, bc);
    const float signd3 = dot(v3, p - b);
    const float signa3 = dot(v3, a - b);

    return float4(signa0 * signd0, signa1 * signd1, signa2 * signd2, signa3 * signd3);
}

// closestPtPointTriangle2 (GuDistancePointTriangle.h) — the Ericson core
float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return a + w * ac;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

// closestPtTetrahedron (softbodyPrimitives.cu :1865 — identical to the
// npSbTet.slang tranche-84 port; hint bit 8 uses the INVERTED flip test)
void closestPtTetrahedron(float3 point, uint hint, float3 a, float3 b, float3 c, float3 d,
                          inout float3 normal, inout float3 closestP, inout float sqDist)
{
    const float3 ac = c - a;
    const float3 ab = b - a;
    const float3 ad = d - a;

    if ((hint & 1) != 0)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, c, ab, ac);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ab));
            if (dot(ad, normal) < 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 2) != 0)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, d, ab, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ab, ad));
            if (dot(ac, normal) < 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 4) != 0)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, c, d, ac, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ad));
            if (dot(ab, normal) < 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 8) != 0)
    {
        const float3 bd = d - b;
        const float3 bc = c - b;
        const float3 tPoint = closestPtPointTriangle2(point, b, c, d, bc, bd);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(bc, bd));
            if (dot(ab, normal) > 0.0f)
                normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
}

// computeTetBarycentric (deformableCollision.cuh :96): bool = eps window,
// per-component clamp applied AFTER the unclamped window test
bool computeTetBarycentric(float3 a, float3 b, float3 c, float3 d, float3 point,
                           out float4 barycentric)
{
    const float eps = 1e-5f;
    const float minEps = -eps;
    const float maxEps = 1.0f + eps;
    const float3 ba = b - a;
    const float3 ca = c - a;
    const float3 da = d - a;
    const float3 pa = point - a;

    const float detBcd = dot(ba, cross(ca, da));
    const float detPcd = dot(pa, cross(ca, da));
    const float v = detPcd / detBcd;
    const float detBpd = dot(ba, cross(pa, da));
    const float w = detBpd / detBcd;
    const float detBcp = dot(ba, cross(ca, pa));
    const float x = detBcp / detBcd;
    const float u = 1.0f - v - w - x;
    barycentric = float4(clamp(u, 0.0f, 1.0f), clamp(v, 0.0f, 1.0f),
                         clamp(w, 0.0f, 1.0f), clamp(x, 0.0f, 1.0f));
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}

// ---- writers (PxgFEMCore.h / PxgSoftBodyCore.h) -------------------------------
// writeRigidVsDeformableContactNoBarycentric (PxgFEMCore.h :446)
void writeRigidVsDeformableContactNoBarycentric(uint index, float4 contact, float4 normalPen,
                                                uint2 pairInd0, uint pairInd1, uint2 rigidId,
                                                uint rigidBodyMaterialId)
{
    if (index >= gParams.maxContacts) return;
    gFemByX[index] = rigidId;
    gFemTempByX[index] = rigidId.x;
    gFemIdxByX[index] = index;
    gFemPoint[index] = contact;
    gFemNormalPen[index] = normalPen;
    uint4 info = gFemInfo[index];
    info.x = pairInd0.x;
    info.y = pairInd0.y;
    info.z = pairInd1;
    info.w = (info.w & COLLISION_FLAG_MASK) | (rigidBodyMaterialId & ~COLLISION_FLAG_MASK);
    info.w = info.w & ~COLLISION_FLAG_MASK;
    gFemInfo[index] = info;
}

// PxgFEMContactWriter::writeContact (barycentric overload, :467): the info
// material dword is NOT written (writeContactNoBarycentric only sets the pair)
void femWriteContactBary(uint index, float4 contact, float4 normalPen, float4 bary,
                         uint2 pairInd0, uint pairInd1, uint2 id)
{
    if (index >= gParams.maxContacts) return;
    gFemByX[index] = id;
    gFemTempByX[index] = id.x;
    gFemIdxByX[index] = index;
    gFemPoint[index] = contact;
    gFemNormalPen[index] = normalPen;
    gFemBary[index] = bary;
    uint4 info = gFemInfo[index];
    info.x = pairInd0.x;
    info.y = pairInd0.y;
    info.z = pairInd1;
    gFemInfo[index] = info; // fourth dword untouched
}

// PxgSoftBodyContactWriter::writeContact (PxgSoftBodyCore.h :277)
void sbWriteContact(uint index, float4 contact, float4 normalPen, float4 bary0, float4 bary1,
                    uint pairId0, uint pairId1)
{
    if (index >= gParams.maxContacts) return;
    gSbPoint[index] = contact;
    gSbNormalPen[index] = normalPen;
    gSbBary0[index] = bary0;
    gSbBary1[index] = bary1;
    uint4 info = gSbInfo[index];
    info.x = pairId0; // u32 -> u64 widen: high dword zero
    info.y = 0;
    info.z = pairId1;
    gSbInfo[index] = info; // auxInd (w) untouched
}

// ============================================================================
// sbTetPlane — one thread serializes one warp-pair (4 verts, one reservation)
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void sbTetPlane(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numCases) return;
    const SbCase cs = gCase[i];
    const uint2 rigidId = cs.ids.xy;
    const uint softbodyId = cs.ids.z;
    const uint tetrahedronIdx = cs.ids.w;
    const float contactDist = cs.a.w;
    const float restDist = cs.b.w;

    const uint softBodyMask = encodeSoftBody(softbodyId, tetrahedronIdx);
    if (findRigidFilter(rigidId, softBodyMask))
    {
        gDebug[i] = float4(1, 0, 0, 0);
        return;
    }

    const float3 verts[4] = { cs.a.xyz, cs.b.xyz, cs.c.xyz, cs.d.xyz };
    bool intersect[4];
    float pen[4];
    uint count = 0;
    for (uint k = 0; k < 4; ++k)
    {
        // planeTransform.transformInv(v): only x is consumed
        const float3 pInPlaneSpace = quatRotateInv(cs.q, verts[k] - cs.p.xyz);
        intersect[k] = contactDist >= pInPlaneSpace.x; // INCLUSIVE
        pen[k] = pInPlaneSpace.x;
        if (intersect[k]) ++count;
    }
    gDebug[i] = float4(0, float(count), 0, 0);

    uint startIndex = 0;
    if (count > 0)
        InterlockedAdd(gFemCount[0], count, startIndex);

    const float3 worldNormal = getBasisVector0(cs.q);
    uint offset = 0;
    for (uint k = 0; k < 4; ++k)
    {
        if (!intersect[k]) continue;
        const uint index = startIndex + offset;
        ++offset;
        if (index < gParams.maxContacts) // caller gate, STRICT
        {
            writeRigidVsDeformableContactNoBarycentric(
                index, float4(verts[k], restDist), float4(worldNormal, pen[k]),
                rigidId, softBodyMask, rigidId, cs.ids2.x);
        }
    }
}

// ============================================================================
// sbEmit — writeContactNoBarycentricFullWarp against synthetic GjkOutput
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void sbEmit(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numCases) return;
    const SbCase cs = gCase[i];
    const bool intersect = cs.ids2.x != 0;
    const bool flipOutputNormal = cs.ids2.y != 0;
    const bool alternativeWriteMode = cs.ids2.z != 0;
    const float penOffset = cs.d.x;
    const float restDist = cs.d.y;
    const uint2 pairInd0 = cs.ids.xy;
    const uint pairInd1 = cs.ids.z;
    const uint materialId = cs.ids.w;

    uint index = 0xffffffffu;
    if (intersect)
    {
        InterlockedAdd(gFemCount[0], 1, index); // UNGATED: counter overshoots
        if (index >= gParams.maxContacts)       // INCLUSIVE sentinel
            index = 0xffffffffu;
    }
    gDebug[i] = float4(intersect ? 1 : 0, asfloat(index), 0, 0);

    if (index != 0xffffffffu)
    {
        if (alternativeWriteMode)
        {
            gFemSorted32[index] = pairInd0.x; // PxU32(pairInd0 & 0xffffffff)
        }
        else
        {
            gFemByX[index] = pairInd0;
            gFemTempByX[index] = pairInd0.x;
        }
        gFemIdxByX[index] = index;

        const float3 witnessA = cs.a.xyz;
        const float3 witnessB = cs.b.xyz;
        const float3 n = cs.c.xyz;
        const float pen = dot(witnessB - witnessA, n) + penOffset;

        gFemPoint[index] = float4(witnessB, restDist);
        gFemNormalPen[index] = float4(flipOutputNormal ? -n : n, pen);

        uint4 info = gFemInfo[index];
        info.x = pairInd0.x;
        info.y = pairInd0.y;
        info.z = pairInd1;
        info.w = (info.w & COLLISION_FLAG_MASK) | (materialId & ~COLLISION_FLAG_MASK);
        info.w = info.w & ~COLLISION_FLAG_MASK;
        gFemInfo[index] = info;
    }
}

// ============================================================================
// sbParticle — sbParticleCollision, per-thread
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void sbParticle(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numCases) return;
    const SbCase cs = gCase[i];
    const uint softbodyId = cs.ids.x;
    const uint tetrahedronIdx = cs.ids.y;
    const uint particleSystemId = cs.ids.z;
    const uint sortedParticleIndex = cs.ids.w;
    const uint surfaceMask = cs.ids2.x;
    const uint unsortedParticleIndex = cs.ids2.y;
    const float cDistance = cs.a.w;
    const float restDistance = cs.b.w;

    // PxEncodeParticleIndex(sysId, unsorted) truncated to PxU32 = the unsorted id
    const uint particleMaskLo = unsortedParticleIndex;
    const uint tetMask = encodeSoftBody(softbodyId, tetrahedronIdx);
    const uint tetFullMask = encodeSoftBody(softbodyId, PX_MAX_NB_DEFORMABLE_VOLUME_TET);

    if (surfaceMask == 0 || findParticleFilter(particleMaskLo, tetFullMask) ||
        findParticleFilter(particleMaskLo, tetMask))
    {
        gDebug[i] = float4(1, 0, 0, 0);
        return;
    }

    const float3 a = cs.a.xyz, b = cs.b.xyz, c = cs.c.xyz, d = cs.d.xyz;
    const float3 p = cs.p.xyz;

    const float4 result = PointOutsideOfPlane4(p, a, b, c, d);

    float4 tBarycentric;
    float3 closestP = float3(0, 0, 0);
    float3 n2 = float3(0, 0, 0);
    float sqDist = PXG_MAX_F32_LOCAL;
    closestPtTetrahedron(p, surfaceMask, a, b, c, d, n2, closestP, sqDist);
    computeTetBarycentric(a, b, c, d, closestP, tBarycentric);

    float3 n = -(p - closestP);
    float pen;
    const bool inside = result.x >= 0.0f && result.y >= 0.0f && result.z >= 0.0f && result.w >= 0.0f;
    if (inside)
    {
        pen = dot(n, n2) - restDistance; // UNNORMALIZED n; no cDistance gate
        n = n2;
    }
    else
    {
        const float sqContactDistance = cDistance * cDistance;
        if (sqDist > sqContactDistance) // STRICT reject
        {
            gDebug[i] = float4(0, 0, sqDist, 0);
            return;
        }
        pen = pxNormalize(n);
        if (sqDist < 1e-8f || dot(n, n2) < 0.8f)
            n = n2;
        pen -= restDistance;
    }
    gDebug[i] = float4(0, 1, pen, inside ? 1 : 0);

    uint index;
    InterlockedAdd(gFemCount[0], 1, index); // UNGATED
    if (index >= gParams.maxContacts)
        return;

    const uint2 compressedParticleIndex = uint2(sortedParticleIndex, particleSystemId); // u64 (SORTED)
    femWriteContactBary(index, float4(p, 0.0f), float4(n, pen), tBarycentric,
                        compressedParticleIndex, tetMask, compressedParticleIndex);
}

// ============================================================================
// sbClothVert — sbClothVertCollision, per-thread
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void sbClothVert(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numCases) return;
    const SbCase cs = gCase[i];
    const uint softbodyId = cs.ids.x;
    const uint tetrahedronIdx = cs.ids.y;
    const uint clothId = cs.ids.z;
    const uint vertIndex = cs.ids.w;
    const uint surfaceMask = cs.ids2.x;
    const float cDistance = cs.a.w;

    const uint softBodyMask = encodeSoftBody(softbodyId, tetrahedronIdx);
    const uint softBodyFullMask = encodeSoftBody(softbodyId, PX_MAX_NB_DEFORMABLE_VOLUME_TET);
    const uint clothMask = encodeCloth(clothId, vertIndex);
    const uint clothFullMask = encodeCloth(clothId, PX_MAX_NB_DEFORMABLE_SURFACE_VTX);

    if (findNonRigid(uint2(softBodyFullMask, 0), clothMask, 0, true) ||
        findNonRigid(uint2(softBodyMask, 0), clothFullMask, 0, true) ||
        findNonRigid(uint2(softBodyMask, 0), clothMask, 0, true))
    {
        gDebug[i] = float4(1, 0, 0, 0);
        return;
    }

    if (surfaceMask == 0)
    {
        gDebug[i] = float4(2, 0, 0, 0);
        return;
    }

    const float3 a = cs.a.xyz, b = cs.b.xyz, c = cs.c.xyz, d = cs.d.xyz;
    const float3 p = cs.p.xyz;

    const float4 result = PointOutsideOfPlane4(p, a, b, c, d);

    float4 tBarycentric;
    float3 closestP = float3(0, 0, 0);
    float3 n2 = float3(0, 0, 0);
    float sqDist = PXG_MAX_F32_LOCAL;
    closestPtTetrahedron(p, surfaceMask, a, b, c, d, n2, closestP, sqDist);
    computeTetBarycentric(a, b, c, d, closestP, tBarycentric);

    float3 v = closestP - p;
    const float m = pxNormalize(v);

    float3 n;
    float pen;
    const float3 contact = closestP;
    const bool inside = result.x >= 0.0f && result.y >= 0.0f && result.z >= 0.0f && result.w >= 0.0f;
    if (inside)
    {
        n = n2;
        pen = -m;
    }
    else
    {
        const float sqContactDistance = cDistance * cDistance;
        if (sqDist > sqContactDistance) // STRICT reject
        {
            gDebug[i] = float4(0, 0, sqDist, 0);
            return;
        }
        pen = m;
        n = (m > 1e-10f && dot(v, n2) > 0.707f) ? v : n2;
    }
    gDebug[i] = float4(0, 1, pen, inside ? 1 : 0);

    uint index;
    InterlockedAdd(gSbCount[0], 1, index); // UNGATED
    if (index >= gParams.maxContacts)
        return;

    sbWriteContact(index, float4(contact, 0.0f), float4(n, pen),
                   float4(0, 0, 0, 1), tBarycentric, clothMask, softBodyMask);
}
