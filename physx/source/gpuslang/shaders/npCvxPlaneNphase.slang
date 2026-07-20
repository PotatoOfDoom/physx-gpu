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

// npCvxPlaneNphase.slang
//
// P5 Step 5 tranche C: convexPlaneNphase_Kernel (PxgKernelIds::
// CONVEX_PLANE_KERNEL_MAIN) on the ENGINE launch ABI (BDA, push-constant
// pointer blob).  This is the full convex/box-vs-infinite-PxPlane persistent-
// contact-manifold narrow phase: it REFRESHES the stored manifold under the
// new transform, decides whether to RERUN contact gen (invalidate test),
// GENERATES contacts (face selection + vertex emission + manifold reduction)
// when stale, and emits the contacts to the contact/patch streams.
//
// The per-stage arithmetic is the tranche-155 bound-buffer port VERBATIM,
// factored from the three verified harness shaders into functions here:
//   npCvxPlaneRefresh.slang (cvxPlaneRefresh)  -> planeRefresh()
//   npCvxPlaneGen.slang     (cvxPlaneGen)      -> planeGenerate()
//   npCvxPlaneOut.slang     (cvxPlaneOut)      -> planeOutput()
// Only the parameter plumbing and data access changed: shapes / transforms /
// manifold / contactDistance / materials / streams / counters / cmOutputs are
// read and written through Ptr<> from the raw engine device addresses in the
// 144-byte push blob; the convex cooked-hull blob (header, extents, vertices,
// polygon planes, per-polygon vertex-ref descriptors, facesByEdges8,
// vertexData8) is read at the real GRB byte offsets from shape.hullOrMeshPtr
// instead of arriving pre-extracted.  The contact-output tail runs real
// atomics on PxgPatchAndContactCounters and writes real BDA pointers into
// PxsContactManagerOutput (mirroring boxBoxNphase.slang / the .cu L3072-3181).
//
// llvmpipe portability: the CUDA runs a WARP per contact manager
// (globalWarpIndex = blockIdx.x*blockDim.y + threadIdx.y, threadIdx.x = lane)
// and its face-clipping cooperation is __ballot / __shfl / warpScanExclusive /
// schlock::reduce.  llvmpipe's subgroup width is NOT 32, so — exactly as the
// already-shipped PCM convex path convexConvexNphase_stage1/2 do — this kernel
// SERIALIZES the whole warp on lane 0 (threadIdx.x == 0); lanes 1-31 return
// after the guard.  Every warp reduction becomes the harness's straight-line
// 32-iteration serial loop, so NO subgroup-width-dependent op remains (a
// strictly stronger lowering than a groupshared 32-lane reduction, and it
// needs zero groupshared).  Empty / out-of-range warps return before any BDA
// load, so an empty batch cannot fault.
//
// Engine grid (PxgNarrowphaseCore.cpp ~L776): block (32,
// CONVEX_PLANE_WARPS_PER_BLOCK = 4), numBlocks = ceil(numTests / 4).

import pxgSlangNpAbi;

// ---------------------------------------------------------------------------
static const float PX_MAX_F32 = 3.4028235e38f;
static const uint INVALID = 0xFFFFFFFFu;

static const uint EGT_PLANE = 1u; // PxGeometryType::ePLANE

// PxsContactManagerStatusFlag
static const uint FLAG_HAS_NO_TOUCH = 1u;
static const uint FLAG_HAS_TOUCH = 2u;
static const uint FLAG_TOUCH_KNOWN = 3u;

// PxgPatchAndContactCounters::OverflowError
static const uint OVERFLOW_CONTACT = 1u;
static const uint OVERFLOW_FORCE = 2u;
static const uint OVERFLOW_PATCH = 4u;

static const uint CONTACT_SIZE = 16u; // sizeof(PxContact)
static const uint FORCE_SIZE = 4u;    // sizeof(PxU32)
static const uint PATCH_SIZE = 64u;   // sizeof(PxContactPatch)

// PxCombineMode / PxMaterialFlag
static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint MAT_DISABLE_FRICTION = 1u, MAT_DISABLE_STRONG_FRICTION = 2u, MAT_COMPLIANT_ACC_SPRING = 16u;

// invalidate_BoxConvex thresholds (cudaNpCommon.h): the ConvexPlane quat table
// is the five-0.99996 one; the translational ratios are the convex ones.
static const float thresholdsConvex[5] = { 0.5f, 0.125f, 0.25f, 0.375f, 0.375f };
static const float quatThresholdsConvexPlane[5] = { 0.99996f, 0.99996f, 0.99996f, 0.99996f, 0.99996f };

// ---------------------------------------------------------------------------
// Push constant: mirrors the CUDA __global__ parameter list EXACTLY (order per
// PxgNarrowphaseCore.cpp ~L752).  Natural alignment: numTests(4) @0, then 15
// device pointers (8-aligned, bank starts @8), the three PxU32 byte-limits and
// the PxReal toleranceLength = 144 bytes (maxAlign 8).
// ---------------------------------------------------------------------------
struct ConvexPlaneParams
{
    uint numTests;                    // @0
    uint64_t cmInputs;                // @8   const PxgContactManagerInput*
    uint64_t cmOutputs;               // @16  PxsContactManagerOutput*
    uint64_t shapes;                  // @24  PxgShape*
    uint64_t transformCache;          // @32  const PxsCachedTransform*
    uint64_t contactDistance;         // @40  const PxReal*
    uint64_t materials;               // @48  const PxsMaterialData*
    uint64_t contactStream;           // @56  PxU8*
    uint64_t patchStream;             // @64  PxU8*
    uint64_t cmPersistentManifolds;   // @72  PxgPersistentContactManifold*
    uint64_t patchAndContactCounters; // @80  PxgPatchAndContactCounters*
    uint64_t touchChangeFlags;        // @88  PxU32* (touchLostFlags at the launch site)
    uint64_t patchChangeFlags;        // @96  PxU32* (touchFoundFlags at the launch site)
    uint64_t startContactPatches;     // @104 PxU8*  (baseContactPatches)
    uint64_t startContactPoints;      // @112 PxU8*  (baseContactPoints)
    uint64_t startContactForces;      // @120 PxU8*  (baseContactForces)
    uint patchBytesLimit;             // @128
    uint contactBytesLimit;           // @132
    uint forceBytesLimit;             // @136
    float toleranceLength;            // @140
}                                     // = 144

[[vk::push_constant]]
ConstantBuffer<ConvexPlaneParams> gParams;

// ---------------------------------------------------------------------------
// Shared math (bit-identical to the harness ports).
// ---------------------------------------------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
// covariant vertex->shape NORMAL transform (shape2Vertex: divide)
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    return npAbiQuatRotateInv(rot, npAbiQuatRotate(rot, v) / scale);
}
// contravariant vertex->shape POINT transform (multiply)
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return npAbiQuatRotateInv(rot, npAbiQuatRotate(rot, v) * scale);
}
float3 loadNormal(float4 p, float3 scale, float4 rot)
{
    return getNormalized(shape2Vertex(p.xyz, scale, rot));
}
float quatDot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Raw byte read out of the cooked-hull blob at an arbitrary device address
// (aligned uint load + shift; BDA-safe).
uint loadHullByte(uint64_t addr)
{
    const uint64_t aligned = addr & ~3ull;
    const uint w = (Ptr<uint>(aligned))[0];
    return (w >> (8u * uint(addr & 3ull))) & 0xFFu;
}

// schlock::reduce<false,false,4> with a SPARSE lane mask and the UNIFORM
// normal (1,0,0): lowest-lane argmin/argmax ties (tranche 137/146 semantics).
uint minIdxM(float v[32], uint mask)
{
    float mn = PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u) mn = min(mn, v[L]);
    for (uint L = 0u; L < 32u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mn) return L;
    return 0u;
}
uint maxIdxM(float v[32], uint mask)
{
    float mx = -PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u) mx = max(mx, v[L]);
    for (uint L = 0u; L < 32u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mx) return L;
    return 0u;
}
uint reduce4(float3 pt[32], float sep[32], float3 normal, uint allMask)
{
    const uint i0 = minIdxM(sep, allMask);
    uint mask = (1u << i0);
    float d1[32];
    for (uint L = 0u; L < 32u; ++L) { const float3 df = pt[i0] - pt[L]; d1[L] = dot(df, df); }
    const uint i1 = maxIdxM(d1, allMask & ~mask);
    mask |= (1u << i1);
    const float3 dir = cross(normal, pt[i1] - pt[i0]);
    float d[32];
    for (uint L = 0u; L < 32u; ++L) d[L] = dot(dir, pt[L] - pt[i0]);
    const uint f = maxIdxM(d, allMask & ~mask);
    mask |= (1u << f);
    uint g = minIdxM(d, allMask & ~mask);
    if (d[f] * d[g] > 0.0f) g = maxIdxM(d, allMask & ~mask);
    mask |= (1u << g);
    return mask;
}

// PxsCombineMaterials scalar combine (npInsertPatch combineMaterials).
float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

// ---------------------------------------------------------------------------
// The main entry: full refresh -> gen -> out on one lane per warp.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 4, 1)] // (WARP_SIZE, CONVEX_PLANE_WARPS_PER_BLOCK)
void convexPlaneNphase_Kernel(uint3 groupID : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    // globalWarpIndex = blockIdx.x * blockDim.y + threadIdx.y (blockDim.y = 4)
    const uint globalWarpIndex = groupID.x * 4u + tid.y;
    if (tid.x != 0u)
        return; // serialized warp: lane 0 performs all 32 lanes' work
    if (globalWarpIndex >= gParams.numTests)
        return; // out-of-range / empty batch: never touch a BDA pointer

    // ---- inputs / shapes (identify plane vs convex, LoadShapePairWarp<ePLANE>) ----
    const PxgContactManagerInputGpu input =
        (Ptr<PxgContactManagerInputGpu>(gParams.cmInputs))[globalWarpIndex];

    Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(gParams.shapes);
    const PxgShapeGpu shapeA = shapes[input.shapeRef0];
    const PxgShapeGpu shapeB = shapes[input.shapeRef1];
    const bool aIsPlane = shapeA.type == EGT_PLANE;
    const PxgShapeGpu planeShape = aIsPlane ? shapeA : shapeB;
    const PxgShapeGpu convexShape = aIsPlane ? shapeB : shapeA;
    const uint planeCacheRef = aIsPlane ? input.transformCacheRef0 : input.transformCacheRef1;
    const uint convexCacheRef = aIsPlane ? input.transformCacheRef1 : input.transformCacheRef0;

    Ptr<PxsCachedTransformGpu> cache = Ptr<PxsCachedTransformGpu>(gParams.transformCache);
    const PxsCachedTransformGpu planeTc = cache[planeCacheRef];
    const PxsCachedTransformGpu convexTc = cache[convexCacheRef];
    const float4 planeTq = float4(planeTc.qx, planeTc.qy, planeTc.qz, planeTc.qw);
    const float3 planeTp = float3(planeTc.px, planeTc.py, planeTc.pz);
    const float4 convexTq = float4(convexTc.qx, convexTc.qy, convexTc.qz, convexTc.qw);
    const float3 convexTp = float3(convexTc.px, convexTc.py, convexTc.pz);

    Ptr<float> contactDistance = Ptr<float>(gParams.contactDistance);
    const float contactDist = contactDistance[planeCacheRef] + contactDistance[convexCacheRef];

    // ---- convex cooked-hull blob (GRB byte offsets from hullOrMeshPtr) ----
    const uint64_t hullPtr = convexShape.hullOrMeshPtr;
    const uint64_t convexPtrB = hullPtr + 16u; // + sizeof(float4)
    const uint header = (Ptr<uint>(convexPtrB))[0];
    const uint nbPolygons = header & 0xFFu;
    const uint nbVertices = (header & 0xFFFFu) >> 8u;
    const uint nbEdges = header >> 16u;
    const float4 extents1 = (Ptr<float4>(convexPtrB + 16u))[0];

    const uint64_t vertsBase   = convexPtrB + 32u;
    const uint64_t planesBase  = vertsBase + uint64_t(16u * nbVertices);
    const uint64_t polyDescBase = planesBase + uint64_t(16u * nbPolygons);
    const uint64_t fbeBase     = polyDescBase + uint64_t(4u * nbPolygons + 4u * nbEdges);
    const uint64_t vtxDataBase = fbeBase + uint64_t(2u * nbEdges + 3u * nbVertices);

    const float3 scale = float3(convexShape.scaleX, convexShape.scaleY, convexShape.scaleZ);
    const float4 rot = float4(convexShape.scaleRotX, convexShape.scaleRotY, convexShape.scaleRotZ, convexShape.scaleRotW);

    // ======================================================================
    // REFRESH (npCvxPlaneRefresh.slang) — bToA, working normal, margin, keep.
    // ======================================================================
    // bToA = planeTransform.transformInv(convexTransform)
    const float4 bToAq = npAbiQuatMul(float4(-planeTq.x, -planeTq.y, -planeTq.z, planeTq.w), convexTq);
    const float3 bToAp = npAbiQuatRotateInv(planeTq, convexTp - planeTp);
    // n = bToA.rotateInv(localNormal=(1,0,0)).getNormalized()
    const float3 n = getNormalized(npAbiQuatRotateInv(bToAq, float3(1, 0, 0)));

    const float3 extScaled = float3(extents1.x * scale.x, extents1.y * scale.y, extents1.z * scale.z);
    const float minMargin = min(min(min(extScaled.x, extScaled.y), extScaled.z) * 0.25f, gParams.toleranceLength * 0.05f);

    Ptr<PxgPersistentContactManifoldGpu> manifold =
        Ptr<PxgPersistentContactManifoldGpu>(gParams.cmPersistentManifolds + uint64_t(globalWarpIndex) * 256u);
    const uint initialNb = manifold[0].nbContacts;

    float3 pBstore[4];
    float distL[4];
    for (uint j = 0u; j < 4u; ++j) { pBstore[j] = float3(0, 0, 0); distL[j] = 0.0f; }
    uint keepW = 0u;
    for (uint j = 0u; j < 4u; ++j)
    {
        if (j >= initialNb) break;
        const float3 pointA = manifold[0].localContactA[j].xyz;
        const float3 pointB = manifold[0].localContactB[j].xyz;
        pBstore[j] = pointB;
        const float3 localAInB = npAbiQuatRotateInv(bToAq, pointA - bToAp); // bToA.transformInv(pointA)
        const float3 v = pointB - localAInB; // REVERSED relative to the convex-convex refresh
        const float dist = dot(v, n);
        distL[j] = dist;
        const float3 projectedDifference = v - n * dist;
        const float th = minMargin * 0.8f;
        if (dot(projectedDifference, projectedDifference) <= th * th) keepW |= (1u << j);
    }
    uint nbContacts = countbits(keepW);
    bool bRun = nbContacts != initialNb;

    if (!bRun)
    {
        const float3 preRelPos = manifold[0].relativePos.xyz;
        const float4 preQuatA = manifold[0].quatA;
        const float4 preQuatB = manifold[0].quatB;
        const float radiusB = sqrt(extents1.x * extents1.x + extents1.y * extents1.y + extents1.z * extents1.z); // RAW extents
        const float ratio = thresholdsConvex[nbContacts];
        const float thresholdQ = quatThresholdsConvexPlane[nbContacts];
        const float thresholdP = minMargin * ratio;
        const float3 curRelPos = -bToAp;
        const float3 dpv = curRelPos - preRelPos;
        const float deltaP = max(abs(dpv.x), max(abs(dpv.y), abs(dpv.z)));
        const float deltaQA = quatDot(planeTq, preQuatA);
        const float deltaQB = quatDot(convexTq, preQuatB);
        bool gen = (deltaP > thresholdP) || (thresholdQ > deltaQA) || (thresholdQ > deltaQB);
        if (!gen)
        {
            const float aR = deltaQA < 1.0f ? acos(deltaQA) : 0.0f;
            const float bR = deltaQB < 1.0f ? acos(deltaQB) : 0.0f;
            gen = (aR * radiusB > thresholdP) || (bR * radiusB > thresholdP); // radiusA = radiusB
        }
        bRun = gen;
    }

    // ======================================================================
    // GEN (npCvxPlaneGen.slang) or no-rerun reload (npCvxPlaneRefresh tail).
    // Fills pL[]/sepL[] per-lane and mask; writes the manifold on rerun.
    // ======================================================================
    float3 pL[32];
    float sepL[32];
    for (uint L = 0u; L < 32u; ++L) { pL[L] = float3(0, 0, 0); sepL[L] = 0.0f; }
    uint mask = 0u;

    if (bRun)
    {
        const float3 localNormal = float3(1, 0, 0);
        const float3 selN = -n;

        // getPolygonIndexFromPlaneNormal(nbPolygons, planes, scale, rot, -n, fbe, nbEdges, plane2)
        float laneBest[32]; uint laneIdx[32];
        for (uint L = 0u; L < 32u; ++L) { laneBest[L] = -PX_MAX_F32; laneIdx[L] = 0u; }
        for (uint i = 0u; i < nbPolygons; ++i)
        {
            const uint L = i & 31u;
            const float4 pl = (Ptr<float4>(planesBase + uint64_t(16u * i)))[0];
            const float proj = dot(getNormalized(shape2Vertex(pl.xyz, scale, rot)), selN);
            if (laneBest[L] < proj) { laneIdx[L] = i; laneBest[L] = proj; }
        }
        float bestProjection = -PX_MAX_F32;
        for (uint L = 0u; L < 32u; ++L) bestProjection = max(bestProjection, laneBest[L]);
        uint bestLane = 0u;
        for (uint L = 0u; L < 32u; ++L) if (laneBest[L] == bestProjection) { bestLane = L; break; }
        uint bestPlaneBIdx = laneIdx[bestLane];

        float laneEdge[32]; uint laneEdgeIdx[32];
        for (uint L = 0u; L < 32u; ++L) { laneEdge[L] = -PX_MAX_F32; laneEdgeIdx[L] = INVALID; }
        for (uint i = 0u; i < nbEdges; ++i)
        {
            const uint L = i & 31u;
            const uint f0 = loadHullByte(fbeBase + uint64_t(2u * i));
            const uint f1 = loadHullByte(fbeBase + uint64_t(2u * i + 1u));
            const float4 p0 = (Ptr<float4>(planesBase + uint64_t(16u * f0)))[0];
            const float4 p1 = (Ptr<float4>(planesBase + uint64_t(16u * f1)))[0];
            const float3 edgeN = getNormalized(loadNormal(p0, scale, rot) + loadNormal(p1, scale, rot));
            const float dp = dot(edgeN, selN);
            if (dp > laneEdge[L]) { laneEdge[L] = dp; laneEdgeIdx[L] = i; }
        }
        float bestEdgeMax = -PX_MAX_F32;
        for (uint L = 0u; L < 32u; ++L) bestEdgeMax = max(bestEdgeMax, laneEdge[L]);
        uint bestEdgeLane = 0u;
        for (uint L = 0u; L < 32u; ++L) if (laneEdge[L] == bestEdgeMax) { bestEdgeLane = L; break; }
        const float bestEdge = bestEdgeMax + 5e-4f;
        const uint closestEdge = laneEdgeIdx[bestEdgeLane];

        uint plane2 = INVALID; // PxI32 plane2 = -1
        if (bestEdge > bestProjection)
        {
            const uint f0 = loadHullByte(fbeBase + uint64_t(closestEdge * 2u));
            const uint f1 = loadHullByte(fbeBase + uint64_t(closestEdge * 2u + 1u));
            const float4 p0 = (Ptr<float4>(planesBase + uint64_t(16u * f0)))[0];
            const float4 p1 = (Ptr<float4>(planesBase + uint64_t(16u * f1)))[0];
            const float dp0 = dot(loadNormal(p0, scale, rot), selN);
            const float dp1 = dot(loadNormal(p1, scale, rot), selN);
            if (dp0 > dp1) { bestPlaneBIdx = f0; plane2 = f1; }
            else           { bestPlaneBIdx = f1; plane2 = f0; }
        }

        // vertex emission loop (the polyDesc read is LOOP-INVARIANT: bestPlaneBIdx)
        float3 shPos[32]; float shSep[32];
        uint totalContacts = 0u;
        for (uint planeIndex = bestPlaneBIdx; planeIndex != INVALID;)
        {
            float3 pInPlaneSpace[32]; float signDist[32];
            const uint polyDesc = (Ptr<uint>(polyDescBase + uint64_t(4u * bestPlaneBIdx)))[0]; // cu L2966
            const uint nbVertsPoly = (polyDesc & 0xFFFFu) >> 8u;
            const uint vRef8 = polyDesc >> 16u;
            for (uint L = 0u; L < 32u; ++L)
            {
                pInPlaneSpace[L] = float3(0, 0, 0);
                signDist[L] = PX_MAX_F32;
                if (L < nbVertsPoly)
                {
                    const uint index = loadHullByte(vtxDataBase + uint64_t(vRef8 + L));
                    const float4 vtx = (Ptr<float4>(vertsBase + uint64_t(16u * index)))[0];
                    const float3 pInShapeSpace = vertex2Shape(vtx.xyz, scale, rot);
                    pInPlaneSpace[L] = npAbiQuatRotate(bToAq, pInShapeSpace) + bToAp;
                    signDist[L] = pInPlaneSpace[L].x;
                }
            }
            uint flags = 0u;
            for (uint L = 0u; L < 32u; ++L) if (signDist[L] < contactDist) flags |= (1u << L);
            uint nbC = countbits(flags);
            if (nbC != 0u)
            {
                if (nbC > 16u)
                {
                    flags = reduce4(pInPlaneSpace, signDist, localNormal, flags);
                    nbC = countbits(flags);
                }
                for (uint L = 0u; L < 32u; ++L)
                {
                    if ((flags & (1u << L)) != 0u)
                    {
                        const uint index = countbits(flags & ((1u << L) - 1u));
                        shPos[totalContacts + index] = pInPlaneSpace[L];
                        shSep[totalContacts + index] = signDist[L];
                    }
                }
                totalContacts += nbC;
            }
            planeIndex = plane2;
            plane2 = INVALID;
        }

        nbContacts = totalContacts;
        if (nbContacts != 0u)
        {
            // PTX shl clamps shifts >= 32 (1 << 32 == 0) so CUDA gets 0xFFFFFFFF
            // at totalContacts == 32; SPIR-V wraps mod 32, so clamp explicitly.
            mask = (totalContacts >= 32u) ? INVALID : ((1u << totalContacts) - 1u);
            for (uint L = 0u; L < 32u; ++L) if (L < nbContacts) { pL[L] = shPos[L]; sepL[L] = shSep[L]; }
            if (nbContacts > 4u)
            {
                mask = reduce4(pL, sepL, localNormal, mask);
                nbContacts = countbits(mask);
            }

            for (uint L = 0u; L < 32u; ++L)
            {
                if ((mask & (1u << L)) != 0u)
                {
                    const uint offset = countbits(mask & ((1u << L) - 1u)); // warpScanExclusive
                    const float3 p = pL[L];
                    const float sep = sepL[L];
                    const float3 pInB = npAbiQuatRotateInv(bToAq, p - bToAp); // bToA.transformInv(p)
                    float3 pOnPlane = p;
                    pOnPlane.x -= sep;
                    manifold[0].localContactB[offset] = float4(pInB, 0.0f);
                    manifold[0].localContactA[offset] = float4(pOnPlane, 0.0f);
                    manifold[0].localNormalPen[offset] = float4(n, sep);
                }
            }
            // moreData: relPos = -bToA.p, quatA = planeQ, quatB = convexQ; nbContacts.
            manifold[0].relativePos = float4(-bToAp, 0.0f);
            manifold[0].quatA = planeTq;
            manifold[0].quatB = convexTq;
            manifold[0].nbContacts = nbContacts;
        }
    }
    else
    {
        mask = (nbContacts >= 32u) ? INVALID : ((1u << nbContacts) - 1u);
        for (uint L = 0u; L < 4u; ++L)
        {
            if (L >= nbContacts) break;
            pL[L] = npAbiQuatRotate(bToAq, pBstore[L]) + bToAp; // bToA.transform(pointB)
            sepL[L] = distL[L];
        }
    }

    // ======================================================================
    // OUT (npCvxPlaneOut.slang + boxBox stream/patch write) — real atomics
    // on PxgPatchAndContactCounters + real BDA pointers into cmOutputs.
    // ======================================================================
    Ptr<PxgPatchAndContactCountersGpu> counters = Ptr<PxgPatchAndContactCountersGpu>(gParams.patchAndContactCounters);
    Ptr<PxsContactManagerOutputGpu> cmOutputs = Ptr<PxsContactManagerOutputGpu>(gParams.cmOutputs);
    Ptr<uint> touchChangeFlags = Ptr<uint>(gParams.touchChangeFlags);
    Ptr<uint> patchChangeFlags = Ptr<uint>(gParams.patchChangeFlags);

    uint contactByteOffset = INVALID;
    uint forceByteOffset = INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].contactsBytes, CONTACT_SIZE * nbContacts, contactByteOffset);
        InterlockedAdd(counters[0].forceAndIndiceBytes, FORCE_SIZE * nbContacts, forceByteOffset);
        if ((contactByteOffset + CONTACT_SIZE) > gParams.contactBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_CONTACT);
            contactByteOffset = INVALID;
        }
        if ((forceByteOffset + FORCE_SIZE) > gParams.forceBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_FORCE);
            forceByteOffset = INVALID;
        }
    }

    if (contactByteOffset != INVALID)
    {
        Ptr<float4> baseContactStream = Ptr<float4>(gParams.contactStream + uint64_t(contactByteOffset));
        for (uint L = 0u; L < 32u; ++L)
        {
            if ((mask & (1u << L)) != 0u)
            {
                const uint index = countbits(mask & ((1u << L) - 1u)); // warpScanExclusive
                const float3 worldP = npAbiQuatRotate(planeTq, pL[L]) + planeTp; // planeTransform.transform(p)
                baseContactStream[index] = float4(worldP, sepL[L]);
            }
        }
        cmOutputs[globalWarpIndex].contactForces = gParams.startContactForces + uint64_t(forceByteOffset);
        cmOutputs[globalWarpIndex].contactPoints = gParams.startContactPoints + uint64_t(contactByteOffset);
    }
    else
    {
        cmOutputs[globalWarpIndex].contactForces = 0uL;
        cmOutputs[globalWarpIndex].contactPoints = 0uL;
    }

    const uint allflags = cmOutputs[globalWarpIndex].allflagsWord;
    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint statusFlags = oldStatusFlags & (~FLAG_TOUCH_KNOWN);
    if (nbContacts != 0u) statusFlags |= FLAG_HAS_TOUCH;
    else                  statusFlags |= FLAG_HAS_NO_TOUCH;

    const uint prevPatches = (allflags >> 8u) & 0xFFu;
    const bool previouslyHadTouch = (oldStatusFlags & FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & FLAG_TOUCH_KNOWN) != 0u;
    const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0u;
    const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);

    touchChangeFlags[globalWarpIndex] = change ? 1u : 0u;
    patchChangeFlags[globalWarpIndex] = (prevPatches != numPatches) ? 1u : 0u;

    cmOutputs[globalWarpIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u) | (numPatches << 8u);
    {
        const uint nf = cmOutputs[globalWarpIndex].nbContactsFlags;
        cmOutputs[globalWarpIndex].nbContactsFlags = (nf & 0xFFFF0000u) | (nbContacts & 0xFFFFu);
    }

    uint patchIndex = INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].patchesBytes, PATCH_SIZE, patchIndex);
        if ((patchIndex + PATCH_SIZE) > gParams.patchBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_PATCH);
            patchIndex = INVALID;

            statusFlags = (statusFlags & (~FLAG_TOUCH_KNOWN)) | FLAG_HAS_NO_TOUCH;
            cmOutputs[globalWarpIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u);
            const uint nf = cmOutputs[globalWarpIndex].nbContactsFlags;
            cmOutputs[globalWarpIndex].nbContactsFlags = (nf & 0xFFFF0000u);
            touchChangeFlags[globalWarpIndex] = previouslyHadTouch ? 1u : 0u;
            patchChangeFlags[globalWarpIndex] = (prevPatches != 0u) ? 1u : 0u;
        }
        else
        {
            cmOutputs[globalWarpIndex].contactPatches = gParams.startContactPatches + uint64_t(patchIndex);
        }
    }

    // insertIntoPatchStream(materials, patchStream, planeShape, convexShape,
    //   patchIndex, planeTransform.rotate((1,0,0)), nbContacts)
    if (patchIndex != INVALID)
    {
        Ptr<PxsMaterialDataGpu> materials = Ptr<PxsMaterialDataGpu>(gParams.materials);
        const uint mi0 = planeShape.materialIndex;
        const uint mi1 = convexShape.materialIndex;
        const PxsMaterialDataGpu m0 = materials[mi0];
        const PxsMaterialDataGpu m1 = materials[mi1];

        const float r0f = m0.restitution, r1f = m1.restitution;
        const bool compliant0 = r0f < 0.0f, compliant1 = r1f < 0.0f;
        const bool exactlyOneCompliant = compliant0 != compliant1;
        const bool bothCompliant = compliant0 && compliant1;
        const uint m0flags = m0.flagsModes0 & 0xFFFFu;
        const uint m1flags = m1.flagsModes0 & 0xFFFFu;
        const uint m0fric = (m0.flagsModes0 >> 16u) & 0xFFu;
        const uint m1fric = (m1.flagsModes0 >> 16u) & 0xFFu;
        const uint m0rest = (m0.flagsModes0 >> 24u) & 0xFFu;
        const uint m1rest = (m1.flagsModes0 >> 24u) & 0xFFu;
        const uint m0damp = m0.modes1 & 0xFFu;
        const uint m1damp = m1.modes1 & 0xFFu;
        const bool cAcc0 = (m0flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
        const bool cAcc1 = (m1flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
        const bool exactlyOneAcc = cAcc0 != cAcc1;

        float rest, damping;
        if (bothCompliant && exactlyOneAcc)
            rest = cAcc0 ? r0f : r1f;
        else
        {
            const uint mode = exactlyOneCompliant ? MODE_MIN : max(m0rest, m1rest);
            const float flip = (bothCompliant && (mode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
            rest = flip * combineScalars(r0f, r1f, mode);
        }
        if (bothCompliant && exactlyOneAcc)
            damping = cAcc0 ? m0.damping : m1.damping;
        else
        {
            const uint mode = exactlyOneCompliant ? MODE_MAX : max(m0damp, m1damp);
            damping = combineScalars(m0.damping, m1.damping, mode);
        }

        const uint combineFlags = m0flags | m1flags;
        float sta, dyn;
        uint matFlags;
        if ((combineFlags & MAT_DISABLE_FRICTION) == 0u)
        {
            const uint fm = max(m0fric, m1fric);
            const float dynF = combineScalars(m0.dynamicFriction, m1.dynamicFriction, fm);
            const float staF = combineScalars(m0.staticFriction, m1.staticFriction, fm);
            const float fDyn = max(dynF, 0.0f);
            dyn = fDyn;
            sta = (staF - fDyn) >= 0.0f ? staF : fDyn;
            matFlags = combineFlags;
        }
        else
        {
            matFlags = combineFlags | MAT_DISABLE_STRONG_FRICTION;
            dyn = 0.0f; sta = 0.0f;
        }

        const float3 worldNormal = npAbiQuatRotate(planeTq, float3(1, 0, 0));
        Ptr<uint> patchBase = Ptr<uint>(gParams.patchStream + uint64_t(patchIndex));
        const uint one = asuint(1.0f);
        patchBase[0] = one; patchBase[1] = one; patchBase[2] = one; patchBase[3] = one;
        patchBase[4] = asuint(worldNormal.x);
        patchBase[5] = asuint(worldNormal.y);
        patchBase[6] = asuint(worldNormal.z);
        patchBase[7] = asuint(rest);
        patchBase[8] = asuint(dyn);
        patchBase[9] = asuint(sta);
        patchBase[10] = asuint(damping);
        patchBase[11] = ((nbContacts & 0xFFu) << 16u) | ((matFlags & 0xFFu) << 24u);
        patchBase[12] = (mi0 & 0xFFFFu) << 16u;
        patchBase[13] = (mi1 & 0xFFFFu);
    }
}
