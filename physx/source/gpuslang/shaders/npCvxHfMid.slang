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

// npCvxHfMid.slang
//
// Slang port of the convex/sphere/capsule-vs-heightfield midphase
// (convexHFMidphase.cu heightfieldMidphaseCore) plus the one substantive
// core left in softbodyMidPhase.cu:
//
// chMidphase — heightfieldMidphaseCore, one warp per contact manager:
// 1. margin/ratio selection by rigid shape type: eSPHERE -> minMargin =
//    scale.x (radius), ratio 0.02, breaking 0.05; eCAPSULE -> scale.y,
//    0.02/0.05 (NOTE upstream writes `ratio = 0.02` as a DOUBLE literal
//    there — same value); otherwise calculatePCMConvexMargin
//    (min(extent*scale) * 0.25 vs toleranceLength * 0.05, cudaNpCommon.h
//    :349) with ratio 0.2 / breaking 0.8;
// 2. the persistent-manifold gate: invalidateManifold (deltaP =
//    max-abs-component of the aToB position delta > minMargin*ratio OR
//    quat dot < 0.9998, manifold.cuh:105) and, only when NOT invalidated,
//    refreshManifolds (per cached contact: penetration RECOMPUTED AND
//    WRITTEN BACK, 2D drift off the stored normal vs
//    (minMargin*breakingRatio)^2; ANY drifted contact zeroes mNbManifolds
//    and returns true, manifold.cuh:40) — fullContactGen = invalidate ||
//    lostContacts;
// 3. when fullContactGen: the world bound taken to heightfield space by
//    PxBounds3::transformFast(getInverse) in CUDA op order (inverse
//    materialized first — two rotations plus a vector add), fattened by
//    the summed contact distance, divided per-axis by (|rowScale|,
//    heightScale, |colScale|) — the negative-scale min/max swaps are DEAD
//    upstream (PxHeightFieldGeometry::isValid) and documented, not
//    ported.  UNLIKE the femCloth flavour (fcMidphase.slang), this kernel
//    has the CORRECTLY parenthesised overlap gate, keeps the y-range in
//    FLOAT (no PxU32 conversion quirk), and treats a ZERO-AREA row/column
//    span as "no triangles";
// 4. the surviving quads are walked in 32-lane warp batches (two
//    triangles per quad; con0/con1 height culling with strict compares,
//    PxBitAndByte material bytes masked 0x7f, eHOLE = 127 skipped), each
//    batch ballot-packed into ONE lane-0 atomicAdd(nbPairsFound,
//    validCount) reservation; stores are CAPPED by
//    `prevNbPairs + offset < maxPairs` with payload uint4(cmIdx,
//    triangleIndex, nbPairsPerCM + offset, shapeRef1), and the per-CM
//    counter advances by the CLAMPED validCount = PxMax(maxPairs,
//    prevNbPairs) - prevNbPairs when the batch crosses the cap;
// 5. the epilogue runs UNCONDITIONALLY (also for cached manifolds): two
//    lane-0 reservations — midphasePairsNum += nbPairsPerCM and
//    midphasePairsNumPadded += ((nbPairsPerCM+3)&~3)*2 (radix-sort
//    double space) — both SKIPPED when nbPairsPerCM == 0, leaving the
//    broadcast offsets at 0xFFffFFff, which are then FAITHFULLY written
//    into the ConvexMeshPair record: {aToB, cmIndex, startIndex, count =
//    fullContactGen ? nbPairsPerCM : CONVEX_TRIMESH_CACHED (0xffffffff),
//    roundedStartIndex, materialIndices(convex, heightfield)}.
//
// sbShrink — the collision-pair-update bound-shrink early-out of
// softbodyMidPhase.cu sb_clothVertMidphaseCore (:457-468): when
// nbCollisionPairUpdatesPerTimestep > 1, the cloth bound is SHRUNK by
// (n-1) * clothContactDistance on BOTH sides (minimum += , maximum -=)
// and the pair is dropped when the shrunk cloth bound no longer
// intersects the softbody bound.  Everything else in softbodyMidPhase.cu
// is composition glue over already-ported pieces (documented):
// sb_midphaseGeneratePairsLaunch / sbMidphaseCore drive the tranche-78
// SoftbodyTreeTraverser; sb_psMidphaseGeneratePairsLaunch and
// sb_clothVertMidphaseCore drive the tranche-78 SoftbodyBoxTraverser
// with per-particle / per-cloth-vertex boxes pos +- contactDistance
// (the box construction is the tranche-99 fmSphereTree pattern).
//
// WARP SERIALIZATION: one thread serializes one warp-CM — the manifold
// refresh maps 24 lanes to slots (serial loop, tranche-66 doctrine), the
// quad walk serializes 32-lane batches with one reservation per batch in
// lane order, and the epilogue's lane-0 atomics are single adds.
//
// Engine glue not ported (documented): LoadShapePair/transform-cache and
// shape reads, the hull blob extents fetch (cases carry the extents),
// PxsCachedTransform plumbing, grid-stride distribution and
// ConvexMeshPair_WriteWarp vectorization (plain stores here).

import pxgSlangCommon;

static const uint MULTIMANIFOLD_MAX_MANIFOLDS = 4u;
static const uint SUBMANIFOLD_MAX_CONTACTS = 6u;
static const uint MAX_SLOTS = 24u;
static const uint CONVEX_TRIMESH_CACHED = 0xffffffffu;
static const uint HF_HOLE = 127u;
static const float MESH_MANIFOLD_EPSILON = 0.05f;
static const float BOX_MARGIN_RATIO = 0.25f;

struct ChParams { uint numCases; uint maxPairs; uint nbRows; uint nbCols; }

struct ChCase
{
    float4 aToBq;      // shapeToHeightfieldNoScale rotation
    float4 aToBp;      // shapeToHeightfieldNoScale position; w = toleranceLength
    float4 prevQ;      // cached manifold relative transform rotation
    float4 prevP;      // cached manifold relative transform position
    float4 hfQ;        // heightfield world transform rotation
    float4 hfP;        // heightfield world transform position
    float4 boundMin;   // convex world bound; w = contactDist sum
    float4 boundMax;
    float4 shapeData;  // x/y/z = hull extents (convex) or scale radii; w = shape type (0 sphere, 1 capsule, 2 convex)
    float4 hfScale;    // heightfield mesh scale (xyz)
    float4 cvxScale;   // convex mesh scale (xyz) — feeds calculatePCMConvexMargin
    uint4 ids;         // x = cmIdx (== case), y = shapeRef1, z = convexMaterial, w = hfMaterial
    uint4 counts;      // x = mNbManifolds, y/z/w unused
    uint4 nbContacts;  // per-sub-manifold contact counts
}

// manifold contact slot: pointA, pointB, (normal, penetration)
struct ChContact { float4 pointA; float4 pointB; float4 nPen; }

// ConvexMeshPair mirror (uint-packed for exact checks)
struct ChPairOut
{
    float4 aToBq;
    float4 aToBp;
    uint4 meta;   // x = cmIndex, y = startIndex, z = count, w = roundedStartIndex
    uint4 mats;   // x = convexMaterial, y = hfMaterial
}

[[vk::push_constant]]
ConstantBuffer<ChParams> gParams;

RWStructuredBuffer<ChCase> gCase;         // binding 0
RWStructuredBuffer<ChContact> gManifold;  // binding 1: 24 slots per case (in/out: penetration updated)
RWStructuredBuffer<uint> gSamples;        // binding 2: heightfield samples
RWStructuredBuffer<uint4> gStack;         // binding 3: pair stack (capped at maxPairs)
RWStructuredBuffer<uint> gNbPairsFound;   // binding 4
RWStructuredBuffer<uint> gPairsNum;       // binding 5: midphasePairsNum
RWStructuredBuffer<uint> gPairsNumPad;    // binding 6: midphasePairsNumPadded
RWStructuredBuffer<ChPairOut> gPairInfo;  // binding 7: ConvexMeshPair per case
RWStructuredBuffer<uint> gNbManOut;       // binding 8: resulting mNbManifolds per case
RWStructuredBuffer<float4> gShrink;       // binding 9: sbShrink IO (see entry)
RWStructuredBuffer<float4> gDebug;        // binding 10

float hfHeight(uint vertexIndex)
{
    return float(int(gSamples[vertexIndex] << 16) >> 16);
}
uint hfMaterial0(uint v) { return (gSamples[v] >> 16) & 0x7fu; }
uint hfMaterial1(uint v) { return (gSamples[v] >> 24) & 0x7fu; }

uint getMinRow(float x, uint rows)    { return uint(clamp(int(floor(x)), 0, int(rows) - 2)); }
uint getMaxRow(float x, uint rows)    { return uint(clamp(int(ceil(x)),  0, int(rows) - 1)); }
uint getMinColumn(float z, uint cols) { return uint(clamp(int(floor(z)), 0, int(cols) - 2)); }
uint getMaxColumn(float z, uint cols) { return uint(clamp(int(ceil(z)),  0, int(cols) - 1)); }

// calculatePCMConvexMargin (cudaNpCommon.h :349)
float calculatePCMConvexMargin(float3 extents_, float3 scale, float toleranceLength)
{
    const float3 extents = extents_ * scale;
    const float mn = min(min(extents.x, extents.y), extents.z);
    const float toleranceMargin = toleranceLength * MESH_MANIFOLD_EPSILON;
    return min(mn * BOX_MARGIN_RATIO, toleranceMargin);
}

// PxMat33(PxQuat) columns
void quatBasis(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void chMidphase(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numCases) return;
    const ChCase cs = gCase[i];
    const float toleranceLength = cs.aToBp.w;
    const uint shapeType = uint(cs.shapeData.w);

    float minMargin, ratio, breakingThresholdRatio;
    if (shapeType == 0)      // eSPHERE
    {
        minMargin = cs.shapeData.x; // sphere radius (scale.x)
        ratio = 0.02f;
        breakingThresholdRatio = 0.05f;
    }
    else if (shapeType == 1) // eCAPSULE
    {
        minMargin = cs.shapeData.y; // capsule radius (scale.y)
        ratio = 0.02f;
        breakingThresholdRatio = 0.05f;
    }
    else
    {
        minMargin = calculatePCMConvexMargin(cs.shapeData.xyz, cs.cvxScale.xyz, toleranceLength);
        ratio = 0.2f;
        breakingThresholdRatio = 0.8f;
    }

    // invalidateManifold (manifold.cuh :105)
    const float thresholdP = minMargin * ratio;
    const float thresholdQ = 0.9998f;
    const float3 delta = cs.aToBp.xyz - cs.prevP.xyz;
    const float3 absDelta = abs(delta);
    const float deltaP = max(absDelta.x, max(absDelta.y, absDelta.z));
    const float deltaQ = dot(cs.aToBq, cs.prevQ);
    const bool invalidate = (deltaP > thresholdP) || (deltaQ < thresholdQ);

    // refreshManifolds (manifold.cuh :40): only when not invalidated
    bool lostContacts = false;
    uint nbManifolds = cs.counts.x;
    if (!invalidate)
    {
        const float projectBreakingThreshold = minMargin * breakingThresholdRatio;
        const float sqThreshold = projectBreakingThreshold * projectBreakingThreshold;
        const uint base = i * MAX_SLOTS;
        for (uint s = 0; s < MAX_SLOTS; ++s)
        {
            const uint curManifoldIdx = s / SUBMANIFOLD_MAX_CONTACTS;
            if (curManifoldIdx >= nbManifolds) continue;
            const uint curContactIdx = s % SUBMANIFOLD_MAX_CONTACTS;
            if (curContactIdx >= cs.nbContacts[curManifoldIdx]) continue;
            ChContact c = gManifold[base + s];
            const float3 localAInB = quatRotate(cs.aToBq, c.pointA.xyz) + cs.aToBp.xyz;
            const float3 v = localAInB - c.pointB.xyz;
            const float penNew = dot(v, c.nPen.xyz);
            const float3 projectedPoint = localAInB - c.nPen.xyz * c.nPen.w;
            const float3 projectedDifference = c.pointB.xyz - projectedPoint;
            const float distance2d = dot(projectedDifference, projectedDifference);
            c.nPen.w = penNew; // penetration written back
            gManifold[base + s] = c;
            if (distance2d > sqThreshold)
                lostContacts = true;
        }
        if (lostContacts)
            nbManifolds = 0; // lane-0 write of mNbManifolds = 0
    }
    gNbManOut[i] = nbManifolds;

    const bool fullContactGen = invalidate || lostContacts;
    uint nbPairsPerCM = 0;

    if (fullContactGen)
    {
        const float oneOverHeightScale = 1.0f / cs.hfScale.y;
        const float oneOverRowScale = 1.0f / abs(cs.hfScale.x);
        const float oneOverColScale = 1.0f / abs(cs.hfScale.z);

        // transformFast(hfTransform.getInverse(), worldBound): inverse first
        const float4 qinv = float4(-cs.hfQ.x, -cs.hfQ.y, -cs.hfQ.z, cs.hfQ.w);
        const float3 centerW = (cs.boundMin.xyz + cs.boundMax.xyz) * 0.5f;
        const float3 extentW = (cs.boundMax.xyz - cs.boundMin.xyz) * 0.5f;
        const float3 invP = quatRotateInv(cs.hfQ, -cs.hfP.xyz);
        // P9.94b: inv.transform(c) = conj(q).rotate(c) + inv.p (PxTransform.h:131);
        // equivalent to quatRotateInv(q, c) at every stored output (centre only
        // feeds compares/floor/ceil, nothing derived from it is stored) -- the
        // sweep mutant p994-chmid-center-rotinv survives for exactly that reason.
        const float3 center = quatRotate(qinv, centerW) + invP;
        float3 c0, c1, c2;
        quatBasis(qinv, c0, c1, c2);
        const float3 e0 = c0 * extentW.x;
        const float3 e1 = c1 * extentW.y;
        const float3 e2 = c2 * extentW.z;
        const float3 wext = float3(abs(e0.x) + abs(e1.x) + abs(e2.x),
                                   abs(e0.y) + abs(e1.y) + abs(e2.y),
                                   abs(e0.z) + abs(e1.z) + abs(e2.z));
        float3 lmn = center - wext;
        float3 lmx = center + wext;
        // fattenFast by the summed contact distance
        const float cd = cs.boundMin.w;
        lmn -= float3(cd, cd, cd);
        lmx += float3(cd, cd, cd);
        lmn = float3(lmn.x * oneOverRowScale, lmn.y * oneOverHeightScale, lmn.z * oneOverColScale);
        lmx = float3(lmx.x * oneOverRowScale, lmx.y * oneOverHeightScale, lmx.z * oneOverColScale);
        // negative row/col scale swaps are dead upstream (isValid), documented

        // CORRECTLY parenthesised overlap gate (contrast fcMidphase's quirk)
        const bool boundsDontOverlap =
            (lmn.x > float(gParams.nbRows - 1)) || (lmn.z > float(gParams.nbCols - 1)) ||
            (lmx.x < 0.0f) || (lmx.z < 0.0f);

        if (!boundsDontOverlap)
        {
            const uint minRow = getMinRow(lmn.x, gParams.nbRows);
            const uint maxRow = getMaxRow(lmx.x, gParams.nbRows);
            const uint minColumn = getMinColumn(lmn.z, gParams.nbCols);
            const uint maxColumn = getMaxColumn(lmx.z, gParams.nbCols);

            const bool noTriangles = (2u * (maxColumn - minColumn) * (maxRow - minRow)) == 0u;
            if (!noTriangles)
            {
                const float miny = lmn.y; // FLOAT y-range (no u32 conversion here)
                const float maxy = lmx.y;
                const uint columnSpan = maxColumn - minColumn;
                const uint totalNumProcessed = (maxRow - minRow) * columnSpan * 2u;

                for (uint batch = 0; batch < totalNumProcessed; batch += 32u)
                {
                    bool laneHit[32];
                    uint laneTri[32];
                    uint validCount = 0;
                    for (uint l = 0; l < 32u; ++l)
                    {
                        laneHit[l] = false;
                        laneTri[l] = 0xffffffffu;
                        const uint workIndex = l + batch;
                        if (workIndex >= totalNumProcessed) continue;
                        const uint index = workIndex / 2u;
                        const uint vertexIndex = (minRow + index / columnSpan) * gParams.nbCols +
                                                 (minColumn + index % columnSpan);
                        const float h0 = hfHeight(vertexIndex);
                        const float h1 = hfHeight(vertexIndex + 1u);
                        const float h2 = hfHeight(vertexIndex + gParams.nbCols);
                        const float h3 = hfHeight(vertexIndex + gParams.nbCols + 1u);
                        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
                        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;
                        if (!(con0 || con1))
                        {
                            const bool isMaterial1 = (workIndex & 1u) != 0u;
                            const uint material = isMaterial1 ? hfMaterial1(vertexIndex) : hfMaterial0(vertexIndex);
                            if (material != HF_HOLE)
                            {
                                laneTri[l] = isMaterial1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1);
                                laneHit[l] = true;
                                ++validCount;
                            }
                        }
                    }

                    uint prevNbPairs = 0xffffffffu;
                    if (validCount > 0)
                        InterlockedAdd(gNbPairsFound[0], validCount, prevNbPairs);

                    uint offset = 0;
                    for (uint l = 0; l < 32u; ++l)
                    {
                        if (!laneHit[l]) continue;
                        if (prevNbPairs + offset < gParams.maxPairs)
                            gStack[prevNbPairs + offset] =
                                uint4(cs.ids.x, laneTri[l], nbPairsPerCM + offset, cs.ids.y);
                        ++offset;
                    }

                    if (validCount > 0 && (validCount + prevNbPairs) >= gParams.maxPairs)
                        validCount = max(gParams.maxPairs, prevNbPairs) - prevNbPairs;

                    nbPairsPerCM += validCount;
                }
            }
        }
    }

    // epilogue: reservations skipped when zero, offsets stay 0xFFffFFff
    uint prevIntermArraysOffset = 0xffffffffu;
    uint prevIntermArraysPaddedOffset = 0xffffffffu;
    if (nbPairsPerCM > 0)
    {
        InterlockedAdd(gPairsNum[0], nbPairsPerCM, prevIntermArraysOffset);
        InterlockedAdd(gPairsNumPad[0], ((nbPairsPerCM + 3u) & ~3u) * 2u, prevIntermArraysPaddedOffset);
    }

    ChPairOut po;
    po.aToBq = cs.aToBq;
    po.aToBp = float4(cs.aToBp.xyz, 0);
    po.meta = uint4(cs.ids.x, prevIntermArraysOffset,
                    fullContactGen ? nbPairsPerCM : CONVEX_TRIMESH_CACHED,
                    prevIntermArraysPaddedOffset);
    po.mats = uint4(cs.ids.z, cs.ids.w, 0, 0);
    gPairInfo[i] = po;
    gDebug[i] = float4(fullContactGen ? 1 : 0, float(nbPairsPerCM), invalidate ? 1 : 0, lostContacts ? 1 : 0);
}

// ============================================================================
// sbShrink — sb_clothVertMidphaseCore bound-shrink early-out (:457-468)
// gShrink layout per case: [0] clothMin (w = clothContactDistance),
// [1] clothMax (w = nbCollisionPairUpdatesPerTimestep), [2] sbMin, [3] sbMax;
// result written to gDebug[i].x (1 = pair kept, 0 = early-out)
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void sbShrink(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numCases) return;
    const float4 cmin = gShrink[i * 4 + 0];
    const float4 cmax = gShrink[i * 4 + 1];
    const float4 smin = gShrink[i * 4 + 2];
    const float4 smax = gShrink[i * 4 + 3];
    const float n = cmax.w;
    bool keep = true;
    if (n > 1.0f)
    {
        const float3 shrunkMin = cmin.xyz + (n - 1.0f) * float3(cmin.w, cmin.w, cmin.w);
        const float3 shrunkMax = cmax.xyz - (n - 1.0f) * float3(cmin.w, cmin.w, cmin.w);
        // PxBounds3::intersects
        keep = !(smin.x > shrunkMax.x || shrunkMin.x > smax.x ||
                 smin.y > shrunkMax.y || shrunkMin.y > smax.y ||
                 smin.z > shrunkMax.z || shrunkMin.z > smax.z);
    }
    gDebug[i] = float4(keep ? 1 : 0, 0, 0, 0);
}
