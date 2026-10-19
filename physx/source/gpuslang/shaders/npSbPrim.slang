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
// softbodyPrimitives.cu, PhysX 5.12 (tranche 100, ported to 5.12):
//
// sbTetPlane -- the PLANE branch of sb_primitiveContactGenLaunch: the
// caller's per-pair element filter, ONE exact search of the softbody's
// mRigidVolumeFilterPairs for (PxgEncodeRigidFilterRef(rigidId,
// rigidCacheRef), PxgEncodeDeformableTetrahedron(sbId, tetIdx)) --
// findRigidVolumeFilter, the ENGINE's (pxgSlangDefFilter) -- then
// tetPlaneCollide: each of the tet's 4 world verts taken to plane space
// (transformInv); accept is INCLUSIVE `contactDist >= pInPlaneSpace.x`;
// worldPoint = the UNTRANSFORMED vertex, worldNormal =
// planeTransform.q.getBasisVector0() (closed form), pen = the RAW
// plane-space x (no contactDist/restDist subtraction).  One
// globalScanExclusiveSingleWarp reservation per warp (= per pair; ballot +
// exclusive popc + lane-0 atomicAdd skipped when empty), the caller-side
// capacity gate `index < maxNumContacts` STRICT plus the writer's own
// `>= maxNumContacts` gate (double gate, both kept), then
// PxgRigidDbContactWriter::writeVolumeContact (the ENGINE's,
// pxgSlangDeformableWriters): point (w = restDist), normalPen (w = pen),
// pair info (rigidId.getInd(), tet ref), the PxU16 rigid material, the
// rigid sort key; no barycentric.
//
// sbEmit -- writeVolumeContactFullWarp (5.12), the shared emitter of the
// GJK/EPA kernels against synthetic GjkOutput fixtures: ONE UNGATED
// atomicAdd(+1) per contact (the counter intentionally overshoots
// capacity; downstream clamps), sentinel when `index >= maxNumContacts`
// (INCLUSIVE) => nothing written; pen = dot(closestPointB - closestPointA,
// normal) + penOffset (the CUDA sums three per-lane products via shfl -- a
// plain dot in serial form); point = (closestPointB, restDist); normalPen
// = (flipOutputNormal ? -n : n, pen); then writeVolumeContact's store set
// (sort key, pair info, PxU16 material).  The 5.6.1 alternativeWriteMode
// (a 32-bit contactSortedByX lane) is gone with the PxgFEMContactWriter.
//
// sbParticle -- sbParticleCollision + sb_psContactGenLaunch, a plain
// PER-THREAD grid-stride kernel: surfaceMask == 0 exits first; 5.12 has NO
// particle filter chain and NO rest distance here.  Geometry:
// PointOutsideOfPlane4 (four sign products; INSIDE iff all four >= 0,
// INCLUSIVE), closestPtTetrahedron restricted to the surface-hint faces,
// n = -(p - closestP); INSIDE path: pen = n.dot(n2) with the UNNORMALIZED
// n, n = n2, and NO contact-distance gate; OUTSIDE path: reject sqDist >
// cDistance^2 STRICT, pen = n.normalize() (guarded m > 0 divide), direction
// replaced by n2 when sqDist < 1e-8 OR n.dot(n2) < 0.8 (both STRICT).
// Per-contact UNGATED atomicAdd + `>= max` drop, then
// PxgParticleDbContactWriter::writeVolumeContact: point = (p, 0),
// normalPen = (n, pen), pair info (PxgEncodeParticleRef(sysId, SORTED
// index), tet ref), the particle sort key; no barycentric.
//
// sbClothVert -- sbClothVertCollision + sb_clothVertContactGenLaunch,
// per-thread: the 5.12 element filter -- THREE exact searches of the
// softbody's mVolumeSurfaceFilterPairs, (ANY tet, vertex), (tet, ANY
// vertex) and (tet, vertex), findVolumeSurfaceVertexFilter -- BEFORE the
// surfaceMask != 0 gate; same tet geometry, v = closestP - p with m =
// v.normalize(); INSIDE: n = n2, pen = -m; OUTSIDE: reject sqDist >
// cDistance^2 STRICT, pen = m, n = (m > 1e-10f && v.dot(n2) > 0.707f) ? v :
// n2 (both STRICT).  Per-contact UNGATED atomicAdd + inclusive `>= max`
// sentinel; PxgDbDbContactWriter::writeSurfaceVolumeContact, the VERTEX
// overload: point = (closestP, 0), normalPen = (n, pen), pair info
// (PxgEncodeDeformableVertex(clothId, vertIdx), tet ref); no barycentric
// (sb_fem_contact_remap_to_simLaunch computes the tet's).
//
// KNOWN LIMITS: the caller-side capacity gate of sbTetPlane is not
// independently pinnable -- the writer method re-checks the identical
// bound, so loosening the caller gate is masked (upstream redundancy;
// both gates are kept, verified textually).  sbClothVert's `m > 1e-10f`
// guard is likewise unpinnable: an outside classification with m below
// 1e-10 is reachable only through float rounding noise (verified
// textually).
//
// WARP SERIALIZATION: sbTetPlane serializes one warp-pair per thread
// (lane order = vert order, one reservation per pair); sbEmit serializes
// the lane-0 reservation + lane-0..3 stores of the warp emitter;
// sbParticle/sbClothVert are genuinely per-thread upstream.
//
// Engine glue not ported (documented): LoadShapePair/transform caches,
// PxgSoftBody/PxgFEMCloth/PxgParticleSystem loads (cases carry verts,
// poses and ids; the filter list is passed directly), the 12-lane
// shared-memory vert staging, the pair-stack iteration and
// stackSizeNeededOnDevice atomicMax.  The engine kernels themselves are
// driven by softBodySimPrimitiveContactGen*, softBodyPsContactGen and
// softBodySimClothVertContactGen.

import pxgSlangCommon;
import pxgSlangDeformableRef;       // PxgEncodeDeformable* / PxgEncodeParticleRef, PXG_DREF_ANY_FEATURE
import pxgSlangDefFilter;           // findRigidVolumeFilter / findVolumeSurfaceVertexFilter, the engine's
import pxgSlangDeformableWriters;   // the 5.12 contact writers, the engine's

static const float PXG_MAX_F32_LOCAL = 3.402823466e38f;

// The filter list and every writer stream are device addresses (BDA), the way the
// engine kernels read them from the descriptors and the writers.  One stream set
// serves all four entries; each maps it onto the writer its kernel uses.
struct SbParams
{
    uint numCases; uint maxContacts; uint nbFilter; uint pad0;
    uint64_t filterPairs;       // PxgDeformablePairInfo*, sorted (PxgComparePairInfo)
    uint64_t filterSafeAddr;    // any mapped address, read in place of a NULL list
    uint64_t outPoint;          // float4*
    uint64_t outNormalPen;      // float4*
    uint64_t outBarycentric0;   // float4*  (no entry writes one in 5.12)
    uint64_t outBarycentric1;   // float4*
    uint64_t outContactInfo;    // PxgDeformablePairInfo*
    uint64_t outMaterial;       // PxU16*   (rigid writer only)
    uint64_t totalContactCount; // PxU32*
    uint64_t sortKey;           // PxU64*   contactByRigid / contactByParticle
    uint64_t sortKeyTemp;       // PxU32*
    uint64_t sortIndex;         // PxU32*
}

struct SbCase
{
    float4 a, b, c, d; // tet verts (a.w = contactDist, b.w = restDist) | sbEmit: closestPointA/closestPointB/normal, d = (penOffset, restDist, -, -)
    float4 p;          // particle / cloth-vertex position | sbTetPlane: plane transform position
    float4 q;          // sbTetPlane: plane transform quaternion | sbEmit: unused
    uint4 ids;         // sbTetPlane: (rigidLo, rigidHi, softbodyId, tetIdx) | sbEmit: (rigidLo, rigidHi, tetIdx, materialIndex)
                       // sbParticle: (softbodyId, tetIdx, particleSystemId, sortedParticleIndex) | sbClothVert: (softbodyId, tetIdx, clothId, vertIndex)
    uint4 ids2;        // sbTetPlane: (materialIndex, rigidCacheRef, -, -) | sbEmit: (intersect, flip, softbodyId, -)
                       // sbParticle: (surfaceMask, -, -, -) | sbClothVert: (surfaceMask, -, -, -)
}

[[vk::push_constant]]
ConstantBuffer<SbParams> gParams;

RWStructuredBuffer<SbCase> gCase;         // binding 0
RWStructuredBuffer<float4> gDebug;        // binding 1

// ---- small helpers -----------------------------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}
// PxVec3::normalize() (PxVec3.h:295-301): returns the magnitude and normalises in
// place when m > 0.  The scaling is a RECIPROCAL-MULTIPLY -- normalize() goes
// through operator/= (PxVec3.h:260-267), which forms 1.0/m ONCE and multiplies.
// A componentwise `v / m` differs from it for 53.5% of vectors.  P9.36.
float pxNormalize(inout float3 v)
{
    const float m = sqrt(dot(v, v));
    if (m > 0.0f)
        v = v * (1.0f / m);
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

DfFilterArray caseFilters()
{
    return dfFilterArray(gParams.filterPairs, gParams.nbFilter, gParams.filterSafeAddr);
}

PxgRigidDbContactWriterGpu rigidWriter()
{
    PxgRigidDbContactWriterGpu w;
    w.outPoint = gParams.outPoint;
    w.outNormalPen = gParams.outNormalPen;
    w.outBarycentric = gParams.outBarycentric0;
    w.outContactInfo = gParams.outContactInfo;
    w.outRigidMaterialIndex = gParams.outMaterial;
    w.totalContactCount = gParams.totalContactCount;
    w.contactByRigid = gParams.sortKey;
    w.tempContactByRigid = gParams.sortKeyTemp;
    w.contactIndexSortedByRigid = gParams.sortIndex;
    w.maxNumContacts = gParams.maxContacts;
    return w;
}

PxgParticleDbContactWriterGpu particleWriter()
{
    PxgParticleDbContactWriterGpu w;
    w.outPoint = gParams.outPoint;
    w.outNormalPen = gParams.outNormalPen;
    w.outBarycentric = gParams.outBarycentric0;
    w.outContactInfo = gParams.outContactInfo;
    w.totalContactCount = gParams.totalContactCount;
    w.contactByParticle = gParams.sortKey;
    w.tempContactByParticle = gParams.sortKeyTemp;
    w.contactIndexSortedByParticle = gParams.sortIndex;
    w.maxNumContacts = gParams.maxContacts;
    return w;
}

PxgDbDbContactWriterGpu dbDbWriter()
{
    PxgDbDbContactWriterGpu w;
    w.outPoint = gParams.outPoint;
    w.outNormalPen = gParams.outNormalPen;
    w.outBarycentric0 = gParams.outBarycentric0;
    w.outBarycentric1 = gParams.outBarycentric1;
    w.outContactInfo = gParams.outContactInfo;
    w.totalContactCount = gParams.totalContactCount;
    w.maxContacts = gParams.maxContacts;
    return w;
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

// ============================================================================
// sbTetPlane -- one thread serializes one warp-pair (4 verts, one reservation)
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void sbTetPlane(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numCases) return;
    const SbCase cs = gCase[i];
    const uint64_t rigidId = uint64_t(cs.ids.x) | (uint64_t(cs.ids.y) << 32);
    const uint softbodyId = cs.ids.z;
    const uint tetrahedronIdx = cs.ids.w;
    const uint rigidCacheRef = cs.ids2.y;
    const float contactDist = cs.a.w;
    const float restDist = cs.b.w;

    // A filter pair for this rigid and tetrahedron suppresses contact generation.
    if (findRigidVolumeFilter(caseFilters(), rigidId, rigidCacheRef, softbodyId, tetrahedronIdx))
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
        InterlockedAdd(Ptr<uint>(gParams.totalContactCount)[0], count, startIndex);

    const float3 worldNormal = getBasisVector0(cs.q);
    const uint64_t tetRef = PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronIdx);
    PxgRigidDbContactWriterGpu writer = rigidWriter();
    uint offset = 0;
    for (uint k = 0; k < 4; ++k)
    {
        if (!intersect[k]) continue;
        const uint index = startIndex + offset;
        ++offset;
        if (index < gParams.maxContacts) // caller gate, STRICT
        {
            writer.writeVolumeContact(index, float4(verts[k], restDist), float4(worldNormal, pen[k]),
                                      rigidId, cs.ids2.x, tetRef);
        }
    }
}

// ============================================================================
// sbEmit -- writeVolumeContactFullWarp against synthetic GjkOutput
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
    const float penOffset = cs.d.x;
    const float restDist = cs.d.y;
    const uint64_t rigidRef = uint64_t(cs.ids.x) | (uint64_t(cs.ids.y) << 32);
    const uint64_t tetRef = PxgEncodeDeformableTetrahedron(cs.ids2.z, cs.ids.z);
    const uint materialId = cs.ids.w;

    uint index = 0xffffffffu;
    if (intersect)
    {
        InterlockedAdd(Ptr<uint>(gParams.totalContactCount)[0], 1, index); // UNGATED: counter overshoots
        if (index >= gParams.maxContacts)                                    // INCLUSIVE sentinel
            index = 0xffffffffu;
    }
    gDebug[i] = float4(intersect ? 1 : 0, asfloat(index), 0, 0);

    if (index != 0xffffffffu)
    {
        const float3 witnessA = cs.a.xyz;
        const float3 witnessB = cs.b.xyz;
        const float3 n = cs.c.xyz;
        const float pen = dot(witnessB - witnessA, n) + penOffset;

        // writeSortKey, the point, normal/pen, pair info and PxU16 material:
        // writeVolumeContact's store set, which the full-warp form splits across lanes.
        rigidWriter().writeVolumeContact(index, float4(witnessB, restDist),
                                         float4(flipOutputNormal ? -n : n, pen),
                                         rigidRef, materialId, tetRef);
    }
}

// ============================================================================
// sbParticle -- sbParticleCollision, per-thread
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
    const float cDistance = cs.a.w;

    if (surfaceMask == 0)
    {
        gDebug[i] = float4(1, 0, 0, 0);
        return;
    }

    const float3 a = cs.a.xyz, b = cs.b.xyz, c = cs.c.xyz, d = cs.d.xyz;
    const float3 p = cs.p.xyz;

    const float4 result = PointOutsideOfPlane4(p, a, b, c, d);

    float3 closestP = float3(0, 0, 0);
    float3 n2 = float3(0, 0, 0);
    float sqDist = PXG_MAX_F32_LOCAL;
    closestPtTetrahedron(p, surfaceMask, a, b, c, d, n2, closestP, sqDist);

    float3 n = -(p - closestP);
    float pen;
    const bool inside = result.x >= 0.0f && result.y >= 0.0f && result.z >= 0.0f && result.w >= 0.0f;
    if (inside)
    {
        pen = dot(n, n2); // UNNORMALIZED n; no cDistance gate
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
    }
    gDebug[i] = float4(0, 1, pen, inside ? 1 : 0);

    uint index;
    InterlockedAdd(Ptr<uint>(gParams.totalContactCount)[0], 1, index); // UNGATED
    if (index >= gParams.maxContacts)
        return;

    const uint64_t particleRef = PxgEncodeParticleRef(particleSystemId, sortedParticleIndex);
    const uint64_t tetRef = PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronIdx);
    particleWriter().writeVolumeContact(index, float4(p, 0.0f), float4(n, pen), particleRef, tetRef);
}

// ============================================================================
// sbClothVert -- sbClothVertCollision, per-thread
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

    // A filter pair for this tetrahedron and this cloth vertex suppresses contact
    // generation between them; the first two searches check whether either of the
    // two is filtered against any.
    const DfFilterArray filters = caseFilters();
    if (findVolumeSurfaceVertexFilter(filters, softbodyId, PXG_DREF_ANY_FEATURE, clothId, vertIndex) ||
        findVolumeSurfaceVertexFilter(filters, softbodyId, tetrahedronIdx, clothId, PXG_DREF_ANY_FEATURE) ||
        findVolumeSurfaceVertexFilter(filters, softbodyId, tetrahedronIdx, clothId, vertIndex))
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

    float3 closestP = float3(0, 0, 0);
    float3 n2 = float3(0, 0, 0);
    float sqDist = PXG_MAX_F32_LOCAL;
    closestPtTetrahedron(p, surfaceMask, a, b, c, d, n2, closestP, sqDist);

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
    InterlockedAdd(Ptr<uint>(gParams.totalContactCount)[0], 1, index); // UNGATED
    if (index >= gParams.maxContacts)
        return;

    const uint64_t vertRef = PxgEncodeDeformableVertex(clothId, vertIndex);
    const uint64_t tetRef = PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronIdx);
    dbDbWriter().writeSurfaceVolumeContactVertex(index, float4(contact, 0.0f), float4(n, pen), vertRef, tetRef);
}
