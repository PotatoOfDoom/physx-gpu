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
// pair writes ONE contact with BOTH barycentric sets.
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
//   * the two-level FILTER.  A whole-tet-vs-whole-cloth pair kills the query outright; a
//     per-vertex one kills it only if ALL THREE triangle vertices are filtered, each against
//     both the tet's own mask and the softbody-wide wildcard.  The `&&` chain is upstream's
//     and is transcribed rather than simplified -- an `||` here would drop contacts.
//
// THE NORMAL IS STORED NEGATED, and the point is the witness on the CLOTH.  The .cu writes
// `outNormalPen = (-direction, pen)` with `pen = dot(witnessB - witnessA, direction)` where A
// is the softbody and B the cloth (softbodyPrimitives.cu:1531-1543).  outContactInfo takes
// pairInd0 = the CLOTH index and pairInd1 = the SOFTBODY index -- the opposite order to the
// rigid path, and the solve chain downstream depends on it.
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
import pxgSlangDefFilter;   // P9.38
import pxgSlangFEMClothDesc;
import pxgSlangGjk;
import pxgSlangTetCollide;

static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
// PxgSoftBodyCoreKernelIndices.h:57 — the engine's fixed grid for SB_CLOTH_CG.
static const uint SB_MESHCG = 1024u;
static const uint SB_CG_STRIDE = SB_MESHCG * MIDPHASE_WARPS_PER_BLOCK;

static const uint GEOM_TETRAHEDRONMESH = 7u;

// PxEncodeSoftBodyIndex / PxEncodeClothIndex (PxgSimulationCoreDesc.h:100, :109) are the same
// shape: (id << 20) | elementId, with the same all-ones wildcard.
static const uint PXG_BITSHIFT_ID = 20u;
static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffffu;
static const uint PX_MAX_NB_DEFORMABLE_SURFACE_VTX = 0x000fffffu;


// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:4683):
// { PxReal tolerenceLength, cmInputs*, transformCache*, contactDistance*, gpuShapes*,
//   softbodies*, clothes*, stackPtr*, midphasePairsNum*, PxU32 stackSizeBytes,
//   PxgNonRigidFilterPair* filterPairs, PxU32 nbFilterPairs, stackSizeNeededOnDevice*,
//   PxgSoftBodyContactWriter writer }.
// 4 @0 (padded to 8), eight pointers @8..@64, PxU32 @72, a pointer realigned to @80, PxU32
// @88, a pointer realigned to @96, then the 56-byte writer at @104 -> 160-byte blob.
// ABI row {4,8,8,8,8,8,8,8,8,4,8,4,8,56}.
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
    uint64_t filterPairs;           // @80
    uint nbFilterPairs;             // @88
    uint _pad92;                    // @92
    uint64_t stackSizeNeededOnDevice; // @96 (ABI parity only — see below; the reason is NOT
                                      //      the one this comment used to give)
    // P9.78b CORRECTED THIS.  What stood here was "the .cu writes it in the tail of the
    // driver, which is unreachable here since the stack cap is applied to numPairs
    // instead".  That is FALSE, and the coverage checker is what asked the question.
    // softbodyPrimitives.cu:1648 guards the write with `globalWarpIdx == 0 &&
    // threadIdx.x == 0` -- OUTSIDE the capped loop, and reachable whatever numPairs is.
    // Capping numPairs is precisely WHY the tail write exists: it reports the UNCAPPED
    // requirement so the host can grow the stack next frame.
    //
    // The omission is still correct, for a different reason: this kernel is UPSTREAM
    // DEAD.  Its only launch site (PxgNarrowphaseCore.cpp:4704) sits inside
    // `if (supportClothTriangleCollision)` (:4591), and that is a hard-coded local
    // `bool supportClothTriangleCollision = false;` (:4529) with no other mention in the
    // tree.  The LIVE half of the pair is the VERTEX generator, and it does implement
    // the write -- softBodyClothVertContactGenBda.slang:217.
    //
    // A wrong reason that reaches the right conclusion is the P9.76a shape: it survives
    // because nothing contradicts it.  The reason is now the one that is true.
    // --- PxgSoftBodyContactWriter, inlined (PxgSoftBodyCore.h:242) ---
    uint64_t w_outPoint;            // @104
    uint64_t w_outNormalPen;        // @112
    uint64_t w_outBarycentric0;     // @120
    uint64_t w_outBarycentric1;     // @128
    uint64_t w_outContactInfo;      // @136
    uint64_t w_totalContactCount;   // @144
    uint w_maxContacts;             // @152
    uint _pad156;                   // @156 -> 160
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

uint encodeId(uint id, uint elementId) { return (id << PXG_BITSHIFT_ID) | elementId; }

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
    barycentric = float4(clamp(u, 0.0f, 1.0f), clamp(v, 0.0f, 1.0f),
                         clamp(w, 0.0f, 1.0f), clamp(x, 0.0f, 1.0f));
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
// P9.38: the search itself now lives in pxgSlangDefFilter; only this kernel's own
// null-pointer guard stays here, because only this kernel knows which of its buffers
// is safe to point at when the filter list is null (see the module header).
//
// This site also had a DIFFERENT ALGORITHM -- a lower-bound search rather than the
// engine's probe-and-accept loop.  Equivalent (the module header proves why) but not
// the engine's, and now gone.
bool findFilter(uint64_t pairsAddr, uint nbPairs, uint id0, uint id1)
{
    const bool havePairs = (pairsAddr != 0UL) && (nbPairs != 0u);
    return dfFindNonRigidPair(havePairs ? pairsAddr : gCg.softbodies,
                              havePairs ? nbPairs : 0u, id0, id1);
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

        Ptr<uint4> triIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices);
        const uint4 triVertInds = triIndices[triangleInd];

        // ---- the two-level filter (see the header) -------------------------------
        const uint softBodyMask = encodeId(softbodyId, tetrahedronInd);
        const uint softBodyFullMask = encodeId(softbodyId, PX_MAX_NB_DEFORMABLE_VOLUME_TET);
        const uint clothFullMask = encodeId(clothId, PX_MAX_NB_DEFORMABLE_SURFACE_VTX);

        if (findFilter(gCg.filterPairs, gCg.nbFilterPairs, softBodyMask, clothFullMask))
            continue;

        const uint clothMask0 = encodeId(clothId, triVertInds.x);
        const uint clothMask1 = encodeId(clothId, triVertInds.y);
        const uint clothMask2 = encodeId(clothId, triVertInds.z);

        // ALL THREE vertices must be filtered for the triangle to be skipped.  `&&`, as
        // upstream has it -- an `||` silently loses contacts.
        const bool skipTriangle =
            (findFilter(gCg.filterPairs, gCg.nbFilterPairs, softBodyFullMask, clothMask0) ||
             findFilter(gCg.filterPairs, gCg.nbFilterPairs, softBodyMask, clothMask0)) &&
            (findFilter(gCg.filterPairs, gCg.nbFilterPairs, softBodyFullMask, clothMask1) ||
             findFilter(gCg.filterPairs, gCg.nbFilterPairs, softBodyMask, clothMask1)) &&
            (findFilter(gCg.filterPairs, gCg.nbFilterPairs, softBodyFullMask, clothMask2) ||
             findFilter(gCg.filterPairs, gCg.nbFilterPairs, softBodyMask, clothMask2));
        if (skipTriangle)
            continue;

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

        // The witness on the CLOTH is the stored point; the normal is stored NEGATED and the
        // penetration is the witness separation projected on the un-negated direction.
        const float3 wA = o.closestPointA.xyz;   // softbody
        const float3 wB = o.closestPointB.xyz;   // cloth
        const float3 n = o.direction.xyz;
        const float pen = dot(wB - wA, n);

        Ptr<float4>(gCg.w_outPoint)[index] = float4(wB.x, wB.y, wB.z, 0.0f);
        Ptr<float4>(gCg.w_outNormalPen)[index] = float4(-n.x, -n.y, -n.z, pen);
        Ptr<float4>(gCg.w_outBarycentric0)[index] = barycentric0;
        Ptr<float4>(gCg.w_outBarycentric1)[index] = barycentric1;

        // PxgFemFemContactInfo (PxgFEMCore.h:51), 16 B: pairInd0 PxU64 @0, pairInd1 PxU32 @8,
        // auxInd @12.  pairInd0 is the CLOTH and pairInd1 the SOFTBODY -- the opposite order
        // to the rigid path, and the solve chain depends on it.
        const uint64_t ci = gCg.w_outContactInfo + uint64_t(index) * 16UL;
        Ptr<uint>(ci)[0] = encodeId(clothId, triangleInd);
        Ptr<uint>(ci + 4UL)[0] = 0u;
        Ptr<uint>(ci + 8UL)[0] = encodeId(softbodyId, tetrahedronInd);
    }
    }
}
