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


// convexHfMidphaseBda.slang -- P9.51a.
//
//   convexHeightFieldMidphase (gpunarrowphase/src/CUDA/convexHFMidphase.cu:350,
//   core at :69) on the engine BDA ABI: one warp per contact manager pairs a
//   convex/sphere/capsule against the heightfield quads its world bound covers,
//   and appends (cmIdx, triangleIndex, perCmSlot, hfShapeRef) to the midphase
//   stack.
//
// WHY THIS IS NOT COSMETIC.  The launch is UNCONDITIONAL:
//
//   PxgNphaseImplementationContext.cpp:125  case GPU_BUCKET_ID::eConvexHeightfield
//     -> PxgNarrowphaseCore.cpp:2795  testSDKConvexHeightfieldGpu
//     -> :2882                        launchKernel(CONVEX_HEIGHTFIELD_MIDPHASE)
//
// so every convex-vs-PxHeightFieldGeometry pair in any scene goes through here.
// While the kernel was unmapped the shim skipped it, no pairs were produced, and
// a box dropped on a heightfield FELL THROUGH IT with nothing reported -- the
// silent-no-op failure mode P9.27/P9.40 exist for.  No previously scored scene
// reached it (all 37 report zero [pxg-unmapped] lines) because none of them puts
// a rigid on a heightfield; the observing scene arrives with the chain.
//
// ---------------------------------------------------------------------------
// THE ARGUMENT PACK IS THE TRIMESH SIBLING'S, PARAMETER FOR PARAMETER
// ---------------------------------------------------------------------------
// convexHeightFieldMidphase and midphaseGeneratePairs (convexTrimeshMidphaseBda)
// have IDENTICAL CUDA signatures -- fifteen arguments, two by value at @0/@4,
// twelve pointers, a PxU32 at @104, a 112-byte blob.  So MidphaseParams below is
// that struct verbatim and the launch-ABI row is the same shape.  Only the middle
// differs: a BV32 tree walk becomes a heightfield row/column rectangle.
//
// ---------------------------------------------------------------------------
// WHERE THE SHARED HEIGHTFIELD MODULE MUST NOT BE USED, AND WHY
// ---------------------------------------------------------------------------
// pxgSlangHeightfield exports hfMakeRect and hfLocalBound, and reaching for them
// here would be wrong in two places.  Both were written from the FEM/softbody
// heightfield midphases, and those differ from THIS .cu:
//
//   1. THE OVERLAP GATE.  femClothHFMidPhase.cu:140 spells it
//
//          if (!(min.x > nbRows-1) || (min.z > nbCols-1) || (max.x<0) || (max.z<0))
//
//      where the `!` binds to the FIRST comparison only -- a quirk hfMakeRect
//      preserves deliberately.  convexHFMidphase.cu:221 instead builds
//      `boundsDontOverlap` from the WHOLE disjunction and negates THAT.  The
//      module's version admits bounds this path rejects.
//
//   2. THE Y RANGE.  HfRect carries minyF/maxyF as float(hfF2uSat(y)) -- the
//      FEM path's saturating u32 conversion.  This .cu compares the RAW FLOAT
//      localBound y against getHeight() (:254-255, :277-278).
//
// So the gate and the rectangle are transcribed here, from this .cu, and only the
// plain sample decoders (hfHeight, hfMaterial0/1, HF_HOLE) come from the module.
// npCvxHfMid.slang -- the descriptor-set port of the same core -- got both of
// these right for the same reason, and its geometry is what this file lifts.
//
// ---------------------------------------------------------------------------
// AND THERE IS NO SHAPE SWAP HERE
// ---------------------------------------------------------------------------
// The trimesh sibling swaps slot 0 and slot 1 when shapeRef0 is the mesh.  This
// .cu does NOT: it reads gpuShapes[shapeRef0] as the convex and
// gpuShapes[shapeRef1] as the heightfield, unconditionally (:113, :165).
// Carrying the swap across would be a plausible-looking addition that silently
// mis-pairs every manager whose slots happen to arrive the other way round.
//
// GRID.  CUDA launches (numBlocks,1,1) x (32, MIDPHASE_WARPS_PER_BLOCK=2, 1) and
// derives cmIdx = threadIdx.y + blockIdx.x * blockDim.y -- one warp per contact
// manager, two per block, never interacting.  The bridge drops blockDim, so a
// [numthreads(32,1,1)] group is ONE warp and the block's two are walked serially,
// exactly as the trimesh sibling and softBodyMeshMidphaseBda do.

// ---------------------------------------------------------------------------
// THE THREE MUTANTS THAT SURVIVE, AND WHY EACH IS NOT A GAP
// ---------------------------------------------------------------------------
// The sweep ran three times: 18/30 killed, then 24/30, then 27/30.  Every jump
// came from sharpening the FIXTURE, never from changing this file -- and the
// second round is worth recording, because moving one placement to kill the gate
// mutants quietly re-hid two others (minRow/minColumn both fell back to zero).
// A sweep run once at the end would have reported that as success.
//
// The three that remain:
//
//   H1  "add the trimesh sibling's slot swap".  It never fires, because slot 0 is
//       never a heightfield on this path -- and that is not an assumption of this
//       port but of the .cu itself, which is precisely why it does not swap.  On
//       every input the engine can present, the mutant is equivalent.  A fixture
//       that fed a heightfield in slot 0 would be testing behaviour upstream does
//       not define.
//
//   F1  "don't materialise the inverse" -- rotInv(q, c - p) instead of
//       rotInv(q, c) + rotInv(q, -p).
//   F2  "build the basis with quatRotate(q, e_i)" instead of the PxMat33 columns.
//
// F1 and F2 are the P9.37 class: the same value in exact arithmetic, at most a
// last-bit difference in float.  For either to change an output, a rectangle
// boundary would have to fall within one ULP of a floor()/ceil() step, which a
// fixture cannot place on purpose and should not pretend to.  They are kept in
// the sweep as a tripwire for a future change that makes them observable, and the
// reason the port spells them the CUDA's way is the transcription, not a mutant.

module convexHfMidphaseBda;

import pxgSlangCommon;
import pxgSlangPcmManifold;
import pxgSlangHeightfield;   // sample decoders ONLY -- see the header

static const uint WARP = 32u;
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;

// PXG_GEOM_SPHERE / PXG_GEOM_CAPSULE come from pxgSlangCommon.

// cudaNpCommon.h:345/346
static const float MESH_MANIFOLD_EPSILON = 0.05f;
static const float BOX_MARGIN_RATIO      = 0.25f;

// ConvexMeshPair (convexNpCommon.h:70), 56 bytes -- the heightfield path fills
// the SAME record as the trimesh path, which is why the shared tail
// (sortTriangleIndices / convexTrimeshCorrelate / convexTrimeshFinishContacts)
// consumes both without knowing which midphase produced it.
static const uint64_t CMP_STRIDE = 56UL;
static const int CONVEX_TRIMESH_CACHED = -1;

struct MidphaseParams
{
    uint     numContactManagers;  // @0
    float    toleranceLength;     // @4
    uint64_t cmInputs;            // @8    const PxgContactManagerInput*
    uint64_t transformCache;      // @16   const PxsCachedTransform*
    uint64_t bounds;              // @24   const PxBounds3*
    uint64_t contactDistance;     // @32   const PxReal*
    uint64_t gpuShapes;           // @40   const PxgShape*
    uint64_t cvxTrimeshPair;      // @48   ConvexMeshPair*        (output)
    uint64_t multiManifolds;      // @56   PxgPersistentContactMultiManifold*
    uint64_t cmOutputs;           // @64   PxsContactManagerOutput*  (unread, ABI parity)
    uint64_t stackPtr;            // @72   PxU8*                  (output)
    uint64_t stackOffset;         // @80   PxU32*  -> the core's `nbPairsFound`
    uint64_t midphasePairsNum;    // @88   PxU32*                 (atomic)
    uint64_t midphasePairsNumPadded; // @96 PxU32*                (atomic)
    uint     stackSizeBytes;      // @104
    uint     _pad108;             // @108 -> 112-byte blob
}

[[vk::push_constant]]
ConstantBuffer<MidphaseParams> gMp;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

// PxgShape (48 B) comes from pxgSlangCommon -- byte-identical to the local copy
// this file first carried, so the copy is gone rather than kept "for clarity".

// ---------------------------------------------------------------------------
// calculateMaxPairs (midphaseAllocate.cuh:119), identical to the trimesh path --
// the two midphases feed the same intermediate arrays, so they must size them
// the same way.  The subtraction is UNSIGNED and upstream does not guard it;
// transcribed as written, and see the stack write for what that costs here.
// ---------------------------------------------------------------------------
static const uint MEM_REQUIRED_PER_PAIR = 56u;

uint calculateMaxPairs(uint stackSizeBytes, uint numContactManagers)
{
    const uint afterPairPadding = 3u * (MEM_REQUIRED_PER_PAIR - 2u * 4u);
    const uint afterTriangleIndexPadding = numContactManagers * 6u * 4u;
    const uint additionalPadding = afterPairPadding + afterTriangleIndexPadding;
    const uint memAvailableAfterPadding = stackSizeBytes - additionalPadding;
    return memAvailableAfterPadding / MEM_REQUIRED_PER_PAIR;
}

// calculatePCMConvexMargin (cudaNpCommon.h:349): 25% of the smallest SCALED
// internal-AABB extent, capped at 5% of the tolerance length.
float calculatePCMConvexMargin(float4 extents_, float3 scale, float toleranceLength)
{
    const float3 extents = float3(extents_.x * scale.x, extents_.y * scale.y, extents_.z * scale.z);
    const float mn = min(min(extents.x, extents.y), extents.z);
    const float toleranceMargin = toleranceLength * MESH_MANIFOLD_EPSILON;
    return min(mn * BOX_MARGIN_RATIO, toleranceMargin);
}

// PxBounds3 is 24 bytes and lands 8-mod-16 on odd indices -- scalars only.
void loadBounds(uint64_t base, uint i, out float3 mn, out float3 mx)
{
    const uint64_t a = base + uint64_t(i) * 24UL;
    mn = float3(Ptr<float>(a)[0], Ptr<float>(a + 4UL)[0], Ptr<float>(a + 8UL)[0]);
    mx = float3(Ptr<float>(a + 12UL)[0], Ptr<float>(a + 16UL)[0], Ptr<float>(a + 20UL)[0]);
}

// ConvexMeshPair_WriteWarp (dataReadWriteHelper.cuh:384) -- a 14 x 4-byte
// scatter, which is what keeps a 56-byte stride safe at odd cmIdx.
void writeConvexMeshPair(uint64_t base, uint cmIdx, float4 aToBq, float3 aToBp,
                         int startIndex, int count, int cmIndex, int roundedStartIndex,
                         uint2 materialIndices)
{
    const uint64_t a = base + uint64_t(cmIdx) * CMP_STRIDE;
    Ptr<float>(a)[0] = aToBq.x;  Ptr<float>(a + 4UL)[0] = aToBq.y;
    Ptr<float>(a + 8UL)[0] = aToBq.z;  Ptr<float>(a + 12UL)[0] = aToBq.w;
    Ptr<float>(a + 16UL)[0] = aToBp.x; Ptr<float>(a + 20UL)[0] = aToBp.y;
    Ptr<float>(a + 24UL)[0] = aToBp.z;
    Ptr<int>(a + 28UL)[0] = startIndex;
    Ptr<int>(a + 32UL)[0] = count;
    Ptr<int>(a + 36UL)[0] = cmIndex;
    Ptr<int>(a + 40UL)[0] = roundedStartIndex;
    Ptr<int>(a + 44UL)[0] = 0;                  // pad, written for parity with the scatter
    Ptr<uint>(a + 48UL)[0] = materialIndices.x;
    Ptr<uint>(a + 52UL)[0] = materialIndices.y;
}

// getMinRow / getMaxRow / getMinColumn / getMaxColumn (heightfieldUtil.cuh).
// The module exports these too, and they are the ONE heightfield helper family
// the two paths do agree on -- but they are spelled out here beside the gate
// they belong to, because the gate above them is exactly where the paths differ.
// PxMat33(PxQuat) -- the three COLUMNS of the rotation matrix, which is what
// PxBounds3::transformFast builds and then takes the absolute value of.  This is
// NOT the same instruction sequence as quatRotate(q, e_i): same value in exact
// arithmetic, different last bits in float, and P9.37 is the tranche that learned
// to transcribe the sequence rather than the identity.
void chQuatBasis(float4 q, out float3 c0, out float3 c1, out float3 c2)
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

uint chMinRow(float x, uint rows)    { return uint(clamp(int(floor(x)), 0, int(rows) - 2)); }
uint chMaxRow(float x, uint rows)    { return uint(clamp(int(ceil(x)),  0, int(rows) - 1)); }
uint chMinColumn(float z, uint cols) { return uint(clamp(int(floor(z)), 0, int(cols) - 2)); }
uint chMaxColumn(float z, uint cols) { return uint(clamp(int(ceil(z)),  0, int(cols) - 1)); }

// ---------------------------------------------------------------------------
// Warp-emulation scratch.  One group == one virtual warp.
// ---------------------------------------------------------------------------
groupshared uint sBallot;
groupshared uint sIndex0;    // the atomicAdd result, broadcast (__shfl_sync(.., 0))
groupshared uint sAnyLost;   // __any_sync for the manifold refresh

uint chBallot(bool pred, uint lane)
{
    if (lane == 0u) sBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sBallot;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void convexHeightFieldMidphase(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint maxPairs = calculateMaxPairs(gMp.stackSizeBytes, gMp.numContactManagers);
    // The stack holds uint4 entries.  maxPairs is normally far below this (56 vs
    // 16 bytes per pair), so this bound is unreachable -- EXCEPT when
    // calculateMaxPairs' unguarded unsigned subtraction wraps, and then it is the
    // only thing between a too-small stack and a wild write.  That is the reason
    // it is here; it is not a second opinion about the CUDA's own limit.
    const uint stackCapacity = gMp.stackSizeBytes / 16u;

    Ptr<PxgShape> shapes = Ptr<PxgShape>(gMp.gpuShapes);

    // The block's two warps, walked serially (see the header on the grid).
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
        const uint cmIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
        GroupMemoryBarrierWithGroupSync();      // scratch is reused per contact manager
        if (cmIdx >= gMp.numContactManagers)
            continue;                            // group-uniform

        const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gMp.cmInputs)[cmIdx];

        // NO SWAP -- convexHFMidphase.cu:113/:165 read slot 0 as the convex and
        // slot 1 as the heightfield unconditionally.  See the header.
        const PxgShape shape = shapes[cin.shapeRef0];
        const PxgShape hfShape = shapes[cin.shapeRef1];

        // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
        const uint64_t xf0 = gMp.transformCache + uint64_t(cin.transformCacheRef0) * 32UL;
        const uint64_t xf1 = gMp.transformCache + uint64_t(cin.transformCacheRef1) * 32UL;
        const float4 q0 = Ptr<float4>(xf0)[0];
        const float3 p0 = float3(Ptr<float>(xf0 + 16UL)[0], Ptr<float>(xf0 + 20UL)[0],
                                 Ptr<float>(xf0 + 24UL)[0]);
        const float4 q1 = Ptr<float4>(xf1)[0];
        const float3 p1 = float3(Ptr<float>(xf1 + 16UL)[0], Ptr<float>(xf1 + 20UL)[0],
                                 Ptr<float>(xf1 + 24UL)[0]);

        // heightfieldTransform.transformInv(shapeTransform): q = conj(q1)*q0,
        // p = rotInv(q1, p0 - p1)   (convexHFMidphase.cu:109)
        const float4 aToBq = quatMul(quatConjugate(q1), q0);
        const float3 aToBp = quatRotateInv(q1, p0 - p1);

        const float3 shapeScale = float3(shape.scaleX, shape.scaleY, shape.scaleZ);
        const uint64_t hullPtr = uint64_t(shape.hullOrMeshOffset.x) |
                                 (uint64_t(shape.hullOrMeshOffset.y) << 32);

        // ---- margin / ratio, per shape type (convexHFMidphase.cu:116) --------
        float minMargin, ratio, breakingThresholdRatio;
        if (shape.type == PXG_GEOM_SPHERE)
        {
            minMargin = shape.scaleX;            // sphere radius
            ratio = 0.02f; breakingThresholdRatio = 0.05f;
        }
        else if (shape.type == PXG_GEOM_CAPSULE)
        {
            // NOTE upstream writes `ratio = 0.02` as a DOUBLE literal here
            // (:126) while the sphere arm above uses 0.02f.  Same value once
            // converted, so the port writes the float; recorded because a reader
            // diffing the two arms will notice and wonder.
            minMargin = shape.scaleY;            // capsule radius
            ratio = 0.02f; breakingThresholdRatio = 0.05f;
        }
        else
        {
            // The convex hull's internal-AABB extents live at hullPtr + sizeof(float4)*2.
            const float4 extents4 = Ptr<float4>(hullPtr + 32UL)[0];
            minMargin = calculatePCMConvexMargin(extents4, shapeScale, gMp.toleranceLength);
            ratio = 0.2f; breakingThresholdRatio = 0.8f;
        }

        // ---- the PCM gate ----------------------------------------------------
        const uint64_t manifold = gMp.multiManifolds + uint64_t(cmIdx) * PCM_STRIDE;
        const bool invalidate = mmInvalidateManifold(manifold, aToBq, aToBp, minMargin, ratio);

        bool lostContacts = false;
        if (!invalidate)
        {
            // Warp-uniform: mmInvalidateManifold reads only the shared record, so
            // every lane agrees and the barriers below stay in uniform flow.
            const float projectBreakingThreshold = minMargin * breakingThresholdRatio;
            if (lane == 0u) sAnyLost = 0u;
            GroupMemoryBarrierWithGroupSync();
            const bool slotLost = mmRefreshSlot(manifold, aToBq, aToBp, projectBreakingThreshold, lane);
            if (slotLost) InterlockedOr(sAnyLost, 1u);
            GroupMemoryBarrierWithGroupSync();
            if (sAnyLost != 0u)
            {
                lostContacts = true;
                if (lane == 0u) Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0] = 0u;
            }
        }

        const bool fullContactGen = invalidate || lostContacts;

        uint nbPairsPerCM = 0u;

        if (fullContactGen)
        {
            // The heightfield blob: nbRows @0, nbCols @4, then the samples.
            const uint64_t hfPtr = uint64_t(hfShape.hullOrMeshOffset.x) |
                                   (uint64_t(hfShape.hullOrMeshOffset.y) << 32);
            const uint nbRows = Ptr<uint>(hfPtr)[0];
            const uint nbCols = Ptr<uint>(hfPtr + 4UL)[0];
            const uint64_t samples = hfPtr + 8UL;

            const float oneOverHeightScale = 1.0f / hfShape.scaleY;
            const float oneOverRowScale    = 1.0f / abs(hfShape.scaleX);
            const float oneOverColScale    = 1.0f / abs(hfShape.scaleZ);

            // The convex's WORLD bound, taken to heightfield space by
            // PxBounds3::transformFast(getInverse(), worldBound) -- and in the
            // CUDA's op order: the inverse is materialised FIRST, so the centre
            // is rotInv(q, c) + rotInv(q, -p) rather than rotInv(q, c - p).
            // Those differ in the last bits, and P9.37 is the tranche that
            // learned to care.
            float3 worldMin, worldMax;
            loadBounds(gMp.bounds, cin.transformCacheRef0, worldMin, worldMax);

            const float4 qinv = float4(-q1.x, -q1.y, -q1.z, q1.w);
            const float3 centerW = (worldMin + worldMax) * 0.5f;
            const float3 extentW = (worldMax - worldMin) * 0.5f;
            const float3 invP = quatRotateInv(q1, -p1);
            const float3 center = quatRotateInv(q1, centerW) + invP;
            float3 c0, c1, c2;
            chQuatBasis(qinv, c0, c1, c2);
            const float3 e0 = c0 * extentW.x;
            const float3 e1 = c1 * extentW.y;
            const float3 e2 = c2 * extentW.z;
            const float3 wext = float3(abs(e0.x) + abs(e1.x) + abs(e2.x),
                                       abs(e0.y) + abs(e1.y) + abs(e2.y),
                                       abs(e0.z) + abs(e1.z) + abs(e2.z));
            float3 lmn = center - wext;
            float3 lmx = center + wext;

            // fattenFast by the SUMMED contact distance of the two shapes (:190).
            const float cd = Ptr<float>(gMp.contactDistance)[cin.transformCacheRef0] +
                             Ptr<float>(gMp.contactDistance)[cin.transformCacheRef1];
            lmn -= float3(cd, cd, cd);
            lmx += float3(cd, cd, cd);

            lmn = float3(lmn.x * oneOverRowScale, lmn.y * oneOverHeightScale, lmn.z * oneOverColScale);
            lmx = float3(lmx.x * oneOverRowScale, lmx.y * oneOverHeightScale, lmx.z * oneOverColScale);

            // Negative row/column scale swaps min and max back (:203, :212).
            if (hfShape.scaleX < 0.0f) { const float t = lmn.x; lmn.x = lmx.x; lmx.x = t; }
            if (hfShape.scaleZ < 0.0f) { const float t = lmn.z; lmn.z = lmx.z; lmx.z = t; }

            // THE GATE, correctly parenthesised -- see the header for why the
            // module's hfMakeRect must not be used in its place.
            const bool boundsDontOverlap =
                (lmn.x > float(nbRows) - 1.0f) || (lmn.z > float(nbCols) - 1.0f) ||
                (lmx.x < 0.0f) || (lmx.z < 0.0f);

            if (!boundsDontOverlap)
            {
                const uint minRow    = chMinRow(lmn.x, nbRows);
                const uint maxRow    = chMaxRow(lmx.x, nbRows);
                const uint minColumn = chMinColumn(lmn.z, nbCols);
                const uint maxColumn = chMaxColumn(lmx.z, nbCols);

                const bool noTriangles = (2u * (maxColumn - minColumn) * (maxRow - minRow)) == 0u;
                if (!noTriangles)
                {
                    // FLOAT y range -- no u32 conversion on this path (:254).
                    const float miny = lmn.y;
                    const float maxy = lmx.y;

                    const uint columnSpan = maxColumn - minColumn;
                    // Two materials per quad, so each vertex index yields two
                    // candidate triangles (:260).
                    const uint totalNumProcessed = (maxRow - minRow) * columnSpan * 2u;

                    for (uint i = 0u; i < totalNumProcessed; i += WARP)
                    {
                        bool result = false;
                        uint triangleIndex = 0xffffffffu;

                        const uint workIndex = lane + i;
                        if (workIndex < totalNumProcessed)
                        {
                            const uint index = workIndex / 2u;
                            const uint vertexIndex = (minRow + index / columnSpan) * nbCols +
                                                     (minColumn + index % columnSpan);
                            const float h0 = hfHeight(samples, vertexIndex);
                            const float h1 = hfHeight(samples, vertexIndex + 1u);
                            const float h2 = hfHeight(samples, vertexIndex + nbCols);
                            const float h3 = hfHeight(samples, vertexIndex + nbCols + 1u);
                            const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
                            const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;

                            if (!(con0 || con1))
                            {
                                const bool isMaterial1 = (workIndex & 1u) != 0u;
                                const uint material = isMaterial1 ? hfMaterial1(samples, vertexIndex)
                                                                  : hfMaterial0(samples, vertexIndex);
                                if (material != HF_HOLE)
                                {
                                    triangleIndex = isMaterial1 ? ((vertexIndex << 1) + 1u)
                                                                : (vertexIndex << 1);
                                    result = true;
                                }
                            }
                        }

                        const uint mask = chBallot(result, lane);
                        const uint offset = countbits(mask & ((1u << lane) - 1u));
                        uint validCount = countbits(mask);

                        // The cursor is stackOffset -- the core's `nbPairsFound`
                        // parameter (:82, wired from gpuStackShift at
                        // PxgNarrowphaseCore.cpp:2876).  midphasePairsNum is a
                        // DIFFERENT counter, bumped once at the end with this
                        // manager's total; crossing them hands the core kernel a
                        // startIndex pointing into another manager's range.
                        if (lane == 0u)
                        {
                            sIndex0 = 0xffffffffu;
                            if (validCount > 0u)
                                InterlockedAdd(Ptr<uint>(gMp.stackOffset)[0], validCount, sIndex0);
                        }
                        GroupMemoryBarrierWithGroupSync();
                        const uint prevNbPairs = sIndex0;

                        if (result && ((prevNbPairs + offset) < maxPairs) &&
                            ((prevNbPairs + offset) < stackCapacity))
                        {
                            Ptr<uint4>(gMp.stackPtr)[prevNbPairs + offset] =
                                uint4(cmIdx, triangleIndex, nbPairsPerCM + offset, cin.shapeRef1);
                        }

                        if ((validCount > 0u) && ((validCount + prevNbPairs) >= maxPairs))
                            validCount = max(maxPairs, prevNbPairs) - prevNbPairs;

                        nbPairsPerCM += validCount;
                    }
                }
            }
        }

        // ---- the per-pair record ---------------------------------------------
        // Both counters are bumped by lane 0 only and broadcast, mirroring the
        // CUDA's `if (idxInWarp == 0 && nbPairsPerCM > 0)` + __shfl_sync pair
        // (:327).  0xFFFFFFFF is the "no pairs" sentinel and goes into the record
        // as-is.  NOTE the record is written for EVERY manager, including the
        // ones that produced nothing and the ones whose cached manifold stands --
        // that is what CONVEX_TRIMESH_CACHED tells the consumer.
        if (lane == 0u)
        {
            uint prevOffset = 0xffffffffu;
            uint prevPaddedOffset = 0xffffffffu;
            if (nbPairsPerCM > 0u)
            {
                InterlockedAdd(Ptr<uint>(gMp.midphasePairsNum)[0], nbPairsPerCM, prevOffset);
                InterlockedAdd(Ptr<uint>(gMp.midphasePairsNumPadded)[0],
                               ((nbPairsPerCM + 3u) & ~3u) * 2u, prevPaddedOffset);
            }
            writeConvexMeshPair(gMp.cvxTrimeshPair, cmIdx, aToBq, aToBp,
                                int(prevOffset),
                                fullContactGen ? int(nbPairsPerCM) : CONVEX_TRIMESH_CACHED,
                                int(cmIdx), int(prevPaddedOffset),
                                uint2(shape.materialIndex, hfShape.materialIndex));
        }
    }
}
