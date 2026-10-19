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

// ===========================================================================
// softBodyClothContactGenBda — sb_clothContactGenLaunch
// (softbodyPrimitives.cu:1586, core sbClothCollision at :1207), on the engine BDA ABI.
//
// The second kernel of the softbody-vs-cloth chain: each midphase pair (cmIdx, tetIdx,
// triIdx) becomes a GJK/EPA query between a TETRAHEDRON and a TRIANGLE, and every surviving
// pair writes ONE contact.  BOTH barycentrics are computed and gate it; since PhysX 5.12 only
// the CLOTH's (barycentric0) is written -- the softbody side is remapped later by
// sb_fem_contact_remap_to_simLaunch from the stored point.
//
// It reuses the driver P8.6 extracted -- pxgSlangGjk + pxgSlangEpa + tetPrimitivesCollide2 --
// so what is genuinely new here is the tet-vs-triangle framing around it:
//
//   * the cloud is 4 tet vertices against 3 triangle vertices.  The triangle's inner-sphere
//     radius is upstream's literal 0.000001f, NOT zero: a triangle is flat and has no inner
//     sphere, and the constant only feeds EPA's convergence epsilon;
//   * BOTH barycentric coordinates are computed and BOTH gate the contact -- the tet one from
//     the witness on the softbody, the triangle one from the witness on the cloth.  Either
//     failing its [-1e-5, 1+1e-5] window drops the pair;
//   * the FACE-MASK rejection.  mTetMeshSurfaceHint is a 4-bit mask of which of the tet's
//     faces are on the surface, and the contact is kept only if at least one SURFACE face
//     points against the GJK direction.  A tet touching the cloth through an interior face is
//     touching it through a neighbour that already owns the contact;
//   * the FILTER (5.12): three exact searches of the softbody's mVolumeSurfaceFilterPairs --
//     (tet, ANY triangle of the cloth), (tet, this triangle) and (ANY tet, this triangle);
//     any hit kills the query.  5.6.1 filtered the triangle's three VERTICES with an `&&`
//     chain; 5.12 filters the triangle as a triangle.
//
// THE NORMAL IS STORED NEGATED, and (5.12) the point is the witness on the SOFTBODY -- "the
// remap converts it into a simulation mesh tetrahedron and the corresponding barycentric"
// (5.6.1 stored the cloth's witness).  The .cu writes `outNormalPen = (-direction, pen)` with
// `pen = dot(witnessB - witnessA, direction)` where A is the softbody and B the cloth.
// The PxgDeformablePairInfo takes ref0 = the CLOTH TRIANGLE and ref1 = the SOFTBODY
// TETRAHEDRON -- the PxgDbDbContactWriter convention (ref1 is always the volume).
//
// ONE LANE, NOT A WARP: as in the rigid GJK port, the driver simulates the 32 CUDA lanes with
// per-thread arrays, so the whole query belongs to a single thread and lane 0 runs it.
//
// GRID.  The engine launches gridDim = (SB_MESHCG = 1024, 1, 1) with blockDim = (32, 2, 1)
// (PxgNarrowphaseCore.cpp:4695), so the stride over midphase pairs is gridDim.x * blockDim.y
// = 2048 and the block's two warps are walked serially here.
// ===========================================================================
module softBodyClothContactGenBda;

import pxgSlangSoftBodyDesc;
import pxgSlangDefFilter;   // 5.12: findVolumeSurfaceTriangleFilter
import pxgSlangDeformableRef;     // PxgEncodeDeformable*, PXG_DREF_ANY_FEATURE
import pxgSlangDeformableWriters; // PxgDbDbContactWriterGpu
import pxgSlangFEMClothDesc;
import pxgSlangGjk;
import pxgSlangTetCollide;

static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
// PxgSoftBodyCoreKernelIndices.h:57 — the engine's fixed grid for SB_CLOTH_CG.
static const uint SB_MESHCG = 1024u;
static const uint SB_CG_STRIDE = SB_MESHCG * MIDPHASE_WARPS_PER_BLOCK;

static const uint GEOM_TETRAHEDRONMESH = 7u;



// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the 5.12 CUDA pack:
// { PxReal tolerenceLength, cmInputs*, transformCache*, contactDistance*, gpuShapes*,
//   softbodies*, clothes*, stackPtr*, midphasePairsNum*, PxU32 stackSizeBytes,
//   stackSizeNeededOnDevice*, PxgDbDbContactWriter writer }.
// 4 @0 (padded to 8), eight pointers @8..@64, PxU32 @72, a pointer realigned to @80, then the
// 56-byte writer at @88 -> 144-byte blob.  ABI row {4,8,8,8,8,8,8,8,8,4,8,56}.
// ---------------------------------------------------------------------------
struct SbClothCgParams
{
    float tolerenceLength;          // @0   (ABI parity only)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t gpuShapes;             // @32
    uint64_t softbodies;            // @40
    uint64_t clothes;               // @48
    uint64_t stackPtr;              // @56
    uint64_t midphasePairsNum;      // @64
    uint stackSizeBytes;            // @72
    uint _pad76;                    // @76
    uint64_t stackSizeNeededOnDevice; // @80 (ABI parity only -- see below)
    // The .cu writes it in the driver's tail, guarded by `globalWarpIdx == 0 &&
    // threadIdx.x == 0` OUTSIDE the capped loop (P9.78b).  The omission here is correct
    // because this kernel is UPSTREAM DEAD: its only launch site sits inside
    // `if (supportClothTriangleCollision)`, a hard-coded local `false` in
    // PxgNarrowphaseCore.cpp (still so in 5.12).  The LIVE half of the pair is the VERTEX
    // generator, which does implement the write (softBodyClothVertContactGenBda.slang).
    // --- PxgDbDbContactWriter, inlined (PxgDeformableContactWriters.h, 5.12) ---
    uint64_t w_outPoint;            // @88
    uint64_t w_outNormalPen;        // @96
    uint64_t w_outBarycentric0;     // @104
    uint64_t w_outBarycentric1;     // @112 (not written by this kernel in 5.12)
    uint64_t w_outContactInfo;      // @120 PxgDeformablePairInfo*
    uint64_t w_totalContactCount;   // @128
    uint w_maxContacts;             // @136
    uint _pad140;                   // @140 -> 144
}

[[vk::push_constant]] ConstantBuffer<SbClothCgParams> gCg;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

struct PxgShape
{
    float scaleX, scaleY, scaleZ;
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW;
    uint materialIndex;
    uint2 hullOrMeshOffset;
    uint type;
    uint particleOrSoftbodyId;
}


// A PxU8 array read through the containing 32-bit word.
uint byteAt(uint64_t base, uint idx)
{
    const uint64_t addr = base + uint64_t(idx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}


// LoadShapePairWarp<eTETRAHEDRONMESH, eTRIANGLEMESH>.
void loadSoftbodyClothPair(uint cmIdx, out PxgShape sbShape, out uint sbCacheRef,
                           out PxgShape clothShape, out uint clothCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gCg.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gCg.gpuShapes);
    uint sRefSb = cin.shapeRef0, cRefSb = cin.transformCacheRef0;
    uint sRefCl = cin.shapeRef1, cRefCl = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cin.shapeRef1; cRefSb = cin.transformCacheRef1;
        sRefCl = cin.shapeRef0; cRefCl = cin.transformCacheRef0;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    clothShape = shapes[sRefCl]; clothCacheRef = cRefCl;
}

// PxQuat::rotate (PxQuat.h:286) — the triangle's vertices go cloth-local -> world.
float3 cgQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// computeTetBarycentric (deformableCollision.cuh:96): the eps window is tested on the
// UNCLAMPED values and the clamp is applied afterwards.
// P9.86: PxClamp(t, 0, 1) = PxMin(1, PxMax(0, t)) = selectMin(1, selectMax(0, t))
// (deformableCollision.cuh:131-134). The selects keep t = -0 (0 > -0 is false) and pass
// NaN through; GLSL FClamp (fmin(fmax(t, 0), 1)) returns +0 for -0 on llvmpipe.
float sbClothPxClamp01(float t) { const float m = (0.0f > t) ? 0.0f : t; return (1.0f < m) ? 1.0f : m; }

bool computeTetBarycentric(float3 a, float3 b, float3 c, float3 d, float3 point,
                           out float4 barycentric)
{
    const float eps = 1e-5f;
    const float minEps = -eps, maxEps = 1.0f + eps;
    const float3 ba = b - a, ca = c - a, da = d - a, pa = point - a;

    const float detBcd = dot(ba, cross(ca, da));
    const float v = dot(pa, cross(ca, da)) / detBcd;
    const float w = dot(ba, cross(pa, da)) / detBcd;
    const float x = dot(ba, cross(ca, pa)) / detBcd;
    const float u = 1.0f - v - w - x;
    barycentric = float4(sbClothPxClamp01(u), sbClothPxClamp01(v),
                         sbClothPxClamp01(w), sbClothPxClamp01(x));   // P9.86: select PxClamp
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}

// computeTriangleBarycentric (softbodyPrimitives.cu:83).  Unlike the tet version this one
// does NOT clamp -- the raw u, v, w are stored, with w-component 0.
bool computeTriangleBarycentric(float3 t0, float3 t1, float3 t2, float3 p, out float4 barycentric)
{
    const float minEps = -1e-5f, maxEps = 1.0f + 1e-5f;
    const float3 v0 = t1 - t0, v1 = t2 - t0, v2 = p - t0;

    const float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
    const float d20 = dot(v2, v0), d21 = dot(v2, v1);
    const float denom = 1.0f / (d00 * d11 - d01 * d01);
    const float v = (d11 * d20 - d01 * d21) * denom;
    const float w = (d00 * d21 - d01 * d20) * denom;
    const float u = 1.0f - v - w;

    barycentric = float4(u, v, w, 0.0f);
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps);
}

// ===========================================================================
// 5.12: findVolumeSurfaceTriangleFilter over the softbody's mVolumeSurfaceFilterPairs.  The
// array goes through dfFilterArray with the softbody descriptor as the safe address
// (llvmpipe speculates the first load of a zero-trip search, see pxgSlangDefFilter).
DfFilterArray sbVolumeSurfaceFilters(uint softbodyId)
{
    return softBodyVolumeSurfaceFilters(gCg.softbodies + uint64_t(softbodyId) * PXG_SOFTBODY_STRIDE);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void sb_clothContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint stackSize = gCg.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gCg.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gCg.softbodies);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCg.clothes);
    Ptr<float> contactDistance = Ptr<float>(gCg.contactDistance);

    // Lane 0 runs the whole query (see the header); the other 31 have nothing to do, but the
    // loop shape is kept so the group's control flow stays uniform.
    if (lane != 0u)
        return;

    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint i = globalWarpIdx; i < numPairs; i += SB_CG_STRIDE)
    {
        const uint4 curPair = Ptr<uint4>(gCg.stackPtr)[i];
        const uint cmIdx = curPair.x;
        const uint tetrahedronInd = curPair.y;
        const uint triangleInd = curPair.z;

        PxgShape sbShape, clothShape;
        uint sbCacheRef, clothCacheRef;
        loadSoftbodyClothPair(cmIdx, sbShape, sbCacheRef, clothShape, clothCacheRef);

        const uint softbodyId = sbShape.particleOrSoftbodyId;
        const uint clothId = clothShape.particleOrSoftbodyId;

        const float cDistance = contactDistance[sbCacheRef] + contactDistance[clothCacheRef];

        // ---- the filter (5.12, see the header) -----------------------------------
        // Filters this tetrahedron against the whole surface, then against the input
        // triangle; also checks whether the triangle is filtered against any tetrahedron.
        // `||` short-circuits as in the CUDA.
        const DfFilterArray filters = sbVolumeSurfaceFilters(softbodyId);
        if (findVolumeSurfaceTriangleFilter(filters, softbodyId, tetrahedronInd, clothId, PXG_DREF_ANY_FEATURE) ||
            findVolumeSurfaceTriangleFilter(filters, softbodyId, tetrahedronInd, clothId, triangleInd) ||
            findVolumeSurfaceTriangleFilter(filters, softbodyId, PXG_DREF_ANY_FEATURE, clothId, triangleInd))
            continue;

        Ptr<uint4> triIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices_ownedEdges);
        const uint4 triVertInds = triIndices[triangleInd];

        // ---- the two point clouds -------------------------------------------------
        const uint4 tetIdx = Ptr<uint4>(softbodies[softbodyId].mTetIndices)[tetrahedronInd];
        Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);

        GjkCloud a;
        for (uint k = 0u; k < GJK_MAX_CLOUD; ++k) a.v[k] = float3(0.0f, 0.0f, 0.0f);
        a.n = 4u;
        a.v[0] = tetVerts[tetIdx.x].xyz;
        a.v[1] = tetVerts[tetIdx.y].xyz;
        a.v[2] = tetVerts[tetIdx.z].xyz;
        a.v[3] = tetVerts[tetIdx.w].xyz;

        float3 tetCenter;
        const float innerSphere = computeInnerSphere(a.v[0], a.v[1], a.v[2], a.v[3], tetCenter);

        // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.  The triangle's vertices are
        // stored in cloth-local space and go to world here (shapeToWorld); the .cu applies no
        // mesh scale at all on this path -- the scale block is commented out at :1416.
        const uint64_t xf = gCg.transformCache + uint64_t(clothCacheRef) * 32UL;
        const float4 clothQ = Ptr<float4>(xf)[0];
        const float3 clothP = float3(Ptr<float>(xf + 16UL)[0],
                                     Ptr<float>(xf + 20UL)[0],
                                     Ptr<float>(xf + 24UL)[0]);
        Ptr<float4> triVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);

        GjkCloud b;
        for (uint k = 0u; k < GJK_MAX_CLOUD; ++k) b.v[k] = float3(0.0f, 0.0f, 0.0f);
        b.n = 3u;
        b.v[0] = cgQuatRotate(clothQ, triVerts[triVertInds.x].xyz) + clothP;
        b.v[1] = cgQuatRotate(clothQ, triVerts[triVertInds.y].xyz) + clothP;
        b.v[2] = cgQuatRotate(clothQ, triVerts[triVertInds.z].xyz) + clothP;

        const float3 triCenter = (b.v[0] + b.v[1] + b.v[2]) * (1.0f / 3.0f);

        GjkOut o;
        // inSphereRadius1 is upstream's literal 1e-6f for a triangle, not 0 (see the header).
        const uint result = tetPrimitivesCollide2(a, b, tetCenter - triCenter, cDistance,
                                                  innerSphere, 0.000001f, o);

        if (!(result == GJK_OVERLAP || result == GJK_CLOSE))
            continue;

        // ---- both barycentrics, both gating ---------------------------------------
        float4 barycentric1;   // on the TET, from the softbody witness
        if (!computeTetBarycentric(a.v[0], a.v[1], a.v[2], a.v[3], o.closestPointA.xyz, barycentric1))
            continue;
        float4 barycentric0;   // on the TRIANGLE, from the cloth witness
        if (!computeTriangleBarycentric(b.v[0], b.v[1], b.v[2], o.closestPointB.xyz, barycentric0))
            continue;

        // ---- the surface-face rejection -------------------------------------------
        // mTetMeshSurfaceHint is a 4-bit face mask.  Face f uses startIndex = (0 - f) & 3 and
        // the corners (startIndex, +1, +2) mod 4, with the fourth as the inside reference:
        // T0: 0,1,2  T1: 3,0,1  T2: 2,3,0  T3: 1,2,3.
        const uint tetMask = byteAt(softbodies[softbodyId].mTetMeshSurfaceHint, tetrahedronInd);
        bool keep = false;
        for (uint f = 0u; f < 4u; ++f)
        {
            if ((tetMask & (1u << f)) == 0u)
                continue;
            const uint s = (0u - f) & 3u;
            const float3 f0 = a.v[s];
            const float3 f1 = a.v[(s + 1u) & 3u];
            const float3 f2 = a.v[(s + 2u) & 3u];
            const float3 otherV = a.v[(s + 3u) & 3u];

            float3 normal = cross(f1 - f0, f2 - f0);
            if (dot(otherV - f0, normal) < 0.0f)
                normal = -normal;

            if (dot(normal, o.direction.xyz) < 0.0f)
                keep = true;
        }
        if (!keep)
            continue;

        uint index;
        InterlockedAdd(Ptr<uint>(gCg.w_totalContactCount)[0], 1u, index);
        if (index >= gCg.w_maxContacts)
            continue;

        // 5.12: the SOFTBODY's witness is the stored point (the remap turns it into a
        // simulation tetrahedron and barycentric); the normal is stored NEGATED and the
        // penetration is the witness separation projected on the un-negated direction.
        // barycentric1 still gates above but is no longer written here.
        const float3 wA = o.closestPointA.xyz;   // softbody
        const float3 wB = o.closestPointB.xyz;   // cloth
        const float3 n = o.direction.xyz;
        const float pen = dot(wB - wA, n);

        Ptr<float4>(gCg.w_outPoint)[index] = float4(wA.x, wA.y, wA.z, 0.0f);
        Ptr<float4>(gCg.w_outNormalPen)[index] = float4(-n.x, -n.y, -n.z, pen);
        Ptr<float4>(gCg.w_outBarycentric0)[index] = barycentric0;

        // PxgDeformablePairInfo: ref0 = the CLOTH triangle, ref1 = the SOFTBODY tetrahedron.
        PxgStorePairInfo(gCg.w_outContactInfo, index,
                         PxgEncodeDeformableTriangle(clothId, triangleInd),
                         PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronInd));
    }
    }
}
