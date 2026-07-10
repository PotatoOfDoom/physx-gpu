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

// fcMidphase.slang
//
// Slang port of the remaining femCloth midphase pair-generation cores of
// femClothMidPhase.cu and femClothHFMidPhase.cu (the ClothTreeTraverser
// itself was ported in npMidphaseDeform.slang):
//
// fmVertexAabb (cloth_midphaseGenerateVertexPairsLaunch): per cloth
// vertex, closest = aabb.closestPoint(v) (min.max(max.min(p)) — a
// componentwise clamp of exact inputs), hasHit = |closest - v|^2 <=
// tolerance (INCLUSIVE, tolerance = contactDist^2; the launch reads the
// rigid's world bound and does NOT fatten it — the commented-out
// fattenFast is upstream-dead), then a ballot pack: lane-0 atomicAdd of
// the popcount (skipped when zero), stack[start + laneOffset] =
// uint4(cmIdx, vertexId, 0, 0).  FAITHFUL QUIRK: unlike
// pushOntoStackFullWarp, this launch writes the stack UNCAPPED — there is
// no stackSize parameter on the CUDA kernel at all.
//
// fmSphereTree (ClothParticleTreeTraverser, used by
// cloth_psMidphaseGeneratePairsLaunch — particle spheres vs the cloth
// BV32 tree in WORLD space — and cloth_midphaseVertexMeshLaunch — cloth
// vertices vs a static trimesh BV32 tree, the query point first taken to
// mesh VERTEX space via shape2Vertex(transformInv(p))): mBox =
// [center - radius, center + radius] (radius = summed contact distance),
// the BV32 walk of npBv32/npMidphaseDeform, leaf test
// triBoundingBox(v0,v1,v2).intersects(mBox), push uint4(cmIdx,
// queryIndex, primitiveIndex, 0) via pushOntoStackFullWarp (capacity-
// capped stores, reservation skipped when the warp ballot is empty).
// FAITHFUL QUIRK: the traverser's isMesh flag is OUTPUT-DEAD — the
// backface plane cull it used to gate is commented out upstream, so both
// branches execute the identical bounding-box test (the flag is not
// ported).
//
// fmHfBound (clothHeightfieldMidphaseCore, both launches of
// femClothHFMidPhase.cu): the per-element WORLD bound — triangle bound
// triBoundingBox(v0,v1,v2).fattenFast(contactDist) for
// cloth_heightfieldMidphaseGeneratePairsLaunch, vertex bound pos +-
// contactDist for cloth_midphaseVertexHeightfieldLaunch — transformed
// into heightfield space by PxBounds3::transformFast(hfTransform
// .getInverse(), bound) (basisExtent: rotated center, abs-sum extent
// matrix) and divided per-axis by (|rowScale|, heightScale, |colScale|).
// The negative row/column scale min/max swaps are DEAD upstream
// (PxHeightFieldGeometry::isValid rejects non-positive scales) and are
// documented, not ported.  Then the FAITHFULLY-TRANSCRIBED culling gate
//   if (!(min.x > nbRows-1) || (min.z > nbCols-1) || (max.x < 0) || (max.z < 0))
// — the upstream `!` binds ONLY to the first comparison, so the gate also
// admits bounds entirely off-grid in z / left of the grid; the clamped
// row/column ranges then produce a zero-area span and zero pairs, which
// keeps the quirk output-neutral for those cases but the transcription
// preserves it exactly.  getMinRow/getMaxRow/getMinColumn/getMaxColumn
// are the tranche-62 clamps (floor→[0, n-2], ceil→[0, n-1]); the area
// gate is 2*(maxColumn-minColumn)*(maxRow-minRow) > 0.  The y range is
// passed into heightfieldComputePairs/heightfieldOutputPairs through
// PxU32 parameters — an UPSTREAM IMPLICIT float→uint conversion (CUDA
// cvt.rzi.u32.f32: truncation, saturating negatives to 0 and huge values
// to 0xffffffff) that the pair predicates then compare back as floats;
// ported as an explicit saturating conversion.  Pair counting/emission
// mirror the tranche-61 port: per covered quad two triangles, culled when
// all four corner heights are strictly above maxy (con0) or strictly
// below miny (con1) or the material byte (masked 0x7f upstream by
// PxBitAndByte) is eHOLE = 127; emitted payload uint4(cmIdx, elementIdx,
// (vertexIndex << 1) + isMaterial1, 0).  The warp reservation is an
// exclusive warp-scan of the PER-LANE pair counts, the lane-31 total, one
// lane-0 atomicAdd (skipped when zero), and each lane's output loop is
// capped by the GLOBAL stack size (writeIndex < stackSize).
//
// KNOWN LIMITS (verified textually, not pinnable by the exact-arithmetic
// fixtures): the abs-sum form of basisExtent is indistinguishable from
// abs-of-sum under the signed-permutation test rotations (each extended
// basis vector has disjoint support, so both forms agree); discriminating
// them needs a non-permutation rotation, which would break the exact
// CPU/GPU comparison.  Also note (found by mutation): scaling only ONE
// side of the local bound with a wrong axis inverts the bound and makes
// the unsigned row/column span wrap to ~2^32 — the pair loop then runs
// essentially forever (the CUDA would hang identically); the faithful
// port keeps min and max on the same per-axis scales.
//
// WARP SERIALIZATION: fmVertexAabb and fmHfBound are one-CASE-per-thread
// kernels whose only cross-lane effect is the slot-packing ballot/scan —
// serialized as one 32-case batch per thread with a single reservation
// (lane order = ascending case index).  fmSphereTree keeps the real
// one-32-thread-group-per-query structure of the npMidphaseDeform port.
//
// Engine glue not ported (documented): LoadShapePair(Warp)/transform
// caches, PxgFEMCloth/PxgParticleSystem loads (incl. the warpCopy of the
// particle system into shared memory), the trimesh/cloth blob walks, the
// nbWarpsPerCm=64 work partitioning and grid-stride loops (cases carry
// verts, bounds, transforms and ids directly), and the shape0/shape1
// cloth-identification branch of cloth_midphaseVertexMeshLaunch.
// FAITHFUL QUIRK (documented): cloth_midphaseVertexHeightfieldLaunch
// loads positions[vertIndex] UNGUARDED — vertIndex can exceed nbVerts on
// the last iteration (an out-of-bounds read whose result is discarded by
// the core's element gate); the port gates the load, which is
// output-identical.

import pxgSlangCommon;

static const uint HF_HOLE = 127;

struct FmParams { uint numCases; uint stackSize; uint nbRows; uint nbCols; }

struct FmCase
{
    float4 a;   // vertexAabb: cloth vertex | sphereTree: query center (world) | hfBound: tri v0 (w = contactDist)
    float4 b;   // vertexAabb: aabb min     | sphereTree: x = radius           | hfBound: tri v1
    float4 c;   // vertexAabb: aabb max (w = tolerance = cd^2)                 | hfBound: tri v2
    float4 d;   // sphereTree: mesh transform quat | hfBound: hf transform quat
    float4 e;   // sphereTree: mesh transform pos  | hfBound: hf transform pos
    float4 f;   // sphereTree: mesh scale (xyz)    | hfBound: hf scale (xyz)
    float4 g;   // sphereTree: mesh scale rotation quat
    uint4 ids;  // x = cmIdx, y = element/vertex/query index, z = doTransform (sphereTree) / kind (hfBound: 0 tri, 1 vertex), w = numElements (hfBound gate)
}

// Gu::BV32DataPacked (1168 bytes)
struct Bv32NodeFm
{
    float4 mMin[32];
    float4 mMax[32];
    uint mData[32];
    uint mNbNodes;
    uint mDepth;
    uint pad0, pad1;
}

[[vk::push_constant]]
ConstantBuffer<FmParams> gParams;

RWStructuredBuffer<FmCase> gCase;        // binding 0
RWStructuredBuffer<float4> gVerts;       // binding 1: tree mesh verts (fmSphereTree)
RWStructuredBuffer<uint4> gPrimIndices;  // binding 2: tree triangle indices (fmSphereTree)
RWStructuredBuffer<Bv32NodeFm> gNodes;   // binding 3: BV32 nodes (fmSphereTree)
RWStructuredBuffer<uint> gSamples;       // binding 4: heightfield samples (fmHfBound)
RWStructuredBuffer<uint4> gStack;        // binding 5: output pair stack
RWStructuredBuffer<uint> gCounter;       // binding 6: midphasePairsNum
RWStructuredBuffer<float4> gDebug;       // binding 7: per-case debug

static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;

// PxBounds3::intersects
bool boundsIntersect(float3 amin, float3 amax, float3 bmin, float3 bmax)
{
    return !(bmin.x > amax.x || amin.x > bmax.x || bmin.y > amax.y || amin.y > bmax.y ||
             bmin.z > amax.z || amin.z > bmax.z);
}

void triBounds(float3 v0, float3 v1, float3 v2, out float3 mn, out float3 mx)
{
    mn = min(min(v0, v1), v2);
    mx = max(max(v0, v1), v2);
}

float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) / scale);
}

// ============================================================================
// fmVertexAabb — cloth_midphaseGenerateVertexPairsLaunch core
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void fmVertexAabb(uint3 dtid : SV_DispatchThreadID)
{
    const uint base = dtid.x * 32;
    if (base >= gParams.numCases) return;

    bool laneHit[32];
    uint4 laneItem[32];
    uint totalCount = 0;
    for (uint lane = 0; lane < 32; ++lane)
    {
        const uint i = base + lane;
        laneHit[lane] = false;
        laneItem[lane] = uint4(0, 0, 0, 0);
        if (i >= gParams.numCases)
            continue;
        const FmCase cs = gCase[i];
        const float3 v = cs.a.xyz;
        // PxBounds3::closestPoint = minimum.maximum(maximum.minimum(p))
        const float3 closest = max(cs.b.xyz, min(cs.c.xyz, v));
        const float3 dv = closest - v;
        const bool hasHit = dot(dv, dv) <= cs.c.w; // inclusive
        gDebug[i] = float4(hasHit ? 1 : 0, dot(dv, dv), 0, 0);
        if (hasHit)
        {
            laneHit[lane] = true;
            laneItem[lane] = uint4(cs.ids.x, cs.ids.y, 0, 0);
            ++totalCount;
        }
    }

    uint startIndex = 0;
    if (totalCount > 0)
        InterlockedAdd(gCounter[0], totalCount, startIndex);
    uint offset = 0;
    for (uint lane = 0; lane < 32; ++lane)
    {
        if (!laneHit[lane]) continue;
        gStack[startIndex + offset] = laneItem[lane]; // UNCAPPED, faithful to the CUDA
        ++offset;
    }
}

// ============================================================================
// fmSphereTree — ClothParticleTreeTraverser BV32 walk (one group per query)
// ============================================================================
groupshared int sBv32Nodes[1024];
groupshared uint sMask;
groupshared uint sStart;

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// pushOntoStackFullWarp (bv32Traversal.cuh)
void pushOntoStackFullWarp(bool threadPushesElement, uint idxInWarp, uint stackSize, uint4 stackItemToPush)
{
    const uint resultWarp = ballot32(threadPushesElement, idxInWarp);
    const uint offset = countbits(resultWarp & ((1u << idxInWarp) - 1u));
    const uint validCount = countbits(resultWarp);
    if (idxInWarp == 0)
    {
        uint startIndex = 0xffffffffu - 32; // -32 to prevent wrap-around
        if (validCount > 0)
            InterlockedAdd(gCounter[0], validCount, startIndex);
        sStart = startIndex;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint stackIndex = sStart + offset;
    if (threadPushesElement && (stackIndex < stackSize))
        gStack[stackIndex] = stackItemToPush;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void fmSphereTree(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint workId = gid.x;
    if (workId >= gParams.numCases) return; // group-uniform
    const uint idxInWarp = gtid.x;
    const FmCase w = gCase[workId];

    float3 center = w.a.xyz;
    if (w.ids.z != 0) // cloth_midphaseVertexMeshLaunch: to mesh VERTEX space
    {
        const float3 local = quatRotateInv(w.d, center - w.e.xyz); // transformInv
        center = shape2Vertex(local, w.f.xyz, w.g);
    }
    const float radius = w.b.x;
    const float3 bmn = center - float3(radius, radius, radius);
    const float3 bmx = center + float3(radius, radius, radius);
    if (idxInWarp == 0)
        gDebug[workId] = float4(center, radius);

    uint currentNodeIndex = 0;
    bool hasNode = true;
    uint nbStack = 0;

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6;

        if (currentNodeIndex & 1)
        {
            const uint data = gNodes[nodeIdx].mData[(currentNodeIndex >> 1) & 31];
            const uint nbPrimitives = (data >> 1) & 63;
            const uint primStartIndex = data >> 7;
            uint primitiveIndex = 0xffffffff;
            if (idxInWarp < nbPrimitives)
                primitiveIndex = idxInWarp + primStartIndex;

            bool intersect = false;
            if (primitiveIndex != 0xffffffff)
            {
                const uint4 triIdx = gPrimIndices[primitiveIndex];
                float3 mn, mx;
                triBounds(gVerts[triIdx.x].xyz, gVerts[triIdx.y].xyz, gVerts[triIdx.z].xyz, mn, mx);
                intersect = boundsIntersect(mn, mx, bmn, bmx);
            }
            pushOntoStackFullWarp(intersect, idxInWarp, gParams.stackSize,
                                  uint4(w.ids.x, w.ids.y, primitiveIndex, 0));
        }
        else
        {
            const uint nbChildren = gNodes[nodeIdx].mNbNodes;

            float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
            const bool hasBox = idxInWarp < nbChildren;
            if (hasBox)
            {
                mn = gNodes[nodeIdx].mMin[idxInWarp].xyz;
                mx = gNodes[nodeIdx].mMax[idxInWarp].xyz;
            }
            const bool intersect = hasBox && boundsIntersect(bmn, bmx, mn, mx);

            uint currentIndex = (currentNodeIndex & 0xFFFFFFC0) | (idxInWarp << 1) | 1;

            bool isValidNode = false;
            if (idxInWarp < nbChildren)
            {
                const uint data = gNodes[nodeIdx].mData[idxInWarp];
                isValidNode = intersect;
                if (isValidNode && (data & 1) == 0)
                    currentIndex = (data >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) << 6;
            }

            const uint resultWarp = ballot32(isValidNode, idxInWarp);
            const uint offset = countbits(resultWarp & ((1u << idxInWarp) - 1u));

            if (isValidNode)
                sBv32Nodes[nbStack + offset] = int(currentIndex);

            nbStack += countbits(resultWarp);
        }

        GroupMemoryBarrierWithGroupSync();

        if (nbStack > 0)
        {
            nbStack -= 1;
            currentNodeIndex = uint(sBv32Nodes[nbStack]);
        }
        else
            hasNode = false;
    }
}

// ============================================================================
// fmHfBound — clothHeightfieldMidphaseCore
// ============================================================================
float hfHeight(uint vertexIndex)
{
    const uint word = gSamples[vertexIndex];
    // signed 16-bit height
    return float(int(word << 16) >> 16);
}
uint hfMaterial0(uint vertexIndex) { return (gSamples[vertexIndex] >> 16) & 0x7fu; }
uint hfMaterial1(uint vertexIndex) { return (gSamples[vertexIndex] >> 24) & 0x7fu; }

// tranche-62 grid clamps (heightfieldUtil.cuh)
uint getMinRow(float x, uint rows)      { return uint(clamp(int(floor(x)), 0, int(rows) - 2)); }
uint getMaxRow(float x, uint rows)      { return uint(clamp(int(ceil(x)),  0, int(rows) - 1)); }
uint getMinColumn(float z, uint cols)   { return uint(clamp(int(floor(z)), 0, int(cols) - 2)); }
uint getMaxColumn(float z, uint cols)   { return uint(clamp(int(ceil(z)),  0, int(cols) - 1)); }

// CUDA cvt.rzi.u32.f32: truncate toward zero, saturate negatives/overflow.
// 4294967040.0f (the largest float below 2^32) still converts normally;
// saturation starts at 2^32.
uint f2uSat(float x)
{
    if (!(x > 0.0f)) return 0u;
    if (x >= 4294967296.0f) return 0xffffffffu;
    return uint(x);
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

struct HfRange
{
    uint minRow, maxRow, minColumn, maxColumn;
    float minyF, maxyF; // float(f2uSat(localBound.y)) as the CUDA predicates see them
    uint nbPairs;
}

uint hfComputePairs(HfRange r)
{
    uint nbPairs = 0;
    const uint columnSpan = r.maxColumn - r.minColumn;
    const uint totalNumProcessed = (r.maxRow - r.minRow) * columnSpan * 2u;
    for (uint i = 0u; i < totalNumProcessed; ++i)
    {
        const uint index = i / 2u;
        const uint vertexIndex = (r.minRow + index / columnSpan) * gParams.nbCols + (r.minColumn + index % columnSpan);
        const float h0 = hfHeight(vertexIndex);
        const float h1 = hfHeight(vertexIndex + 1u);
        const float h2 = hfHeight(vertexIndex + gParams.nbCols);
        const float h3 = hfHeight(vertexIndex + gParams.nbCols + 1u);
        const bool con0 = r.maxyF < h0 && r.maxyF < h1 && r.maxyF < h2 && r.maxyF < h3;
        const bool con1 = r.minyF > h0 && r.minyF > h1 && r.minyF > h2 && r.minyF > h3;
        if (!(con0 || con1))
        {
            const bool isMaterial1 = (i & 1u) != 0u;
            const uint material = isMaterial1 ? hfMaterial1(vertexIndex) : hfMaterial0(vertexIndex);
            if (material != HF_HOLE)
                nbPairs++;
        }
    }
    return nbPairs;
}

void hfOutputPairs(HfRange r, uint cmInd, uint elementInd, uint startOffset, uint size)
{
    uint writeIndex = startOffset;
    const uint columnSpan = r.maxColumn - r.minColumn;
    const uint totalNumProcessed = (r.maxRow - r.minRow) * columnSpan * 2u;
    for (uint i = 0u; i < totalNumProcessed && writeIndex < size; ++i)
    {
        const uint index = i / 2u;
        const uint vertexIndex = (r.minRow + index / columnSpan) * gParams.nbCols + (r.minColumn + index % columnSpan);
        const float h0 = hfHeight(vertexIndex);
        const float h1 = hfHeight(vertexIndex + 1u);
        const float h2 = hfHeight(vertexIndex + gParams.nbCols);
        const float h3 = hfHeight(vertexIndex + gParams.nbCols + 1u);
        const bool con0 = r.maxyF < h0 && r.maxyF < h1 && r.maxyF < h2 && r.maxyF < h3;
        const bool con1 = r.minyF > h0 && r.minyF > h1 && r.minyF > h2 && r.minyF > h3;
        if (!(con0 || con1))
        {
            const bool isMaterial1 = (i & 1u) != 0u;
            const uint material = isMaterial1 ? hfMaterial1(vertexIndex) : hfMaterial0(vertexIndex);
            if (material != HF_HOLE)
            {
                const uint triangleIdx = isMaterial1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1);
                gStack[writeIndex] = uint4(cmInd, elementInd, triangleIdx, 0u);
                writeIndex++;
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fmHfBound(uint3 dtid : SV_DispatchThreadID)
{
    const uint base = dtid.x * 32;
    if (base >= gParams.numCases) return;

    HfRange laneRange[32];
    uint laneCm[32];
    uint laneElem[32];

    uint totalCount = 0;
    uint laneOffsets[32];
    for (uint lane = 0; lane < 32; ++lane)
    {
        const uint i = base + lane;
        HfRange r;
        r.minRow = 0; r.maxRow = 0; r.minColumn = 0; r.maxColumn = 0;
        r.minyF = 0; r.maxyF = 0; r.nbPairs = 0;
        laneCm[lane] = 0;
        laneElem[lane] = 0;
        laneOffsets[lane] = totalCount; // exclusive scan of per-lane pair counts
        if (i >= gParams.numCases)
        {
            laneRange[lane] = r;
            continue;
        }
        const FmCase cs = gCase[i];
        laneCm[lane] = cs.ids.x;
        laneElem[lane] = cs.ids.y;
        const uint elementIdx = cs.ids.y;
        const uint numElements = cs.ids.w;

        if (elementIdx < numElements)
        {
            // element bound (world)
            float3 bmn, bmx;
            if (cs.ids.z == 0)
            {
                triBounds(cs.a.xyz, cs.b.xyz, cs.c.xyz, bmn, bmx);
                bmn -= float3(cs.a.w, cs.a.w, cs.a.w); // fattenFast(contactDist)
                bmx += float3(cs.a.w, cs.a.w, cs.a.w);
            }
            else
            {
                bmn = cs.a.xyz - float3(cs.a.w, cs.a.w, cs.a.w); // vertex: pos +- contactDist
                bmx = cs.a.xyz + float3(cs.a.w, cs.a.w, cs.a.w);
            }

            // PxBounds3::transformFast(hfTransform.getInverse(), bound):
            // basisExtent(inv.transform(center), PxMat33(inv.q), extents).
            // CUDA materializes getInverse() FIRST (inv.p = q.rotateInv(-p)),
            // then inv.transform(c) = conj(q).rotate(c) + inv.p — TWO rotations
            // and a vector add, NOT one rotation of the difference (the forms
            // differ in ULPs and the center feeds floor/ceil grid clamps).
            // conj(q).rotate(x) is bit-identical to q.rotateInv(x) under the
            // PxQuat formula, so quatRotateInv stands in for the conjugate.
            const float4 qinv = float4(-cs.d.x, -cs.d.y, -cs.d.z, cs.d.w);
            const float3 centerW = (bmn + bmx) * 0.5f;
            const float3 extentW = (bmx - bmn) * 0.5f;
            const float3 invP = quatRotateInv(cs.d, -cs.e.xyz);
            const float3 center = quatRotateInv(cs.d, centerW) + invP;
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

            // per-axis inverse scale; negative row/col scale swaps are dead
            // upstream (PxHeightFieldGeometry::isValid requires positive scales)
            const float oneOverHeightScale = 1.0f / cs.f.y;
            const float oneOverRowScale = 1.0f / abs(cs.f.x);
            const float oneOverColScale = 1.0f / abs(cs.f.z);
            lmn.x *= oneOverRowScale;
            lmn.y *= oneOverHeightScale;
            lmn.z *= oneOverColScale;
            lmx.x *= oneOverRowScale;
            lmx.y *= oneOverHeightScale;
            lmx.z *= oneOverColScale;

            // FAITHFUL upstream quirk: the ! binds only to the FIRST comparison
            if (!(lmn.x > float(gParams.nbRows - 1)) || (lmn.z > float(gParams.nbCols - 1)) ||
                (lmx.x < 0.0f) || (lmx.z < 0.0f))
            {
                r.minRow = getMinRow(lmn.x, gParams.nbRows);
                r.maxRow = getMaxRow(lmx.x, gParams.nbRows);
                r.minColumn = getMinColumn(lmn.z, gParams.nbCols);
                r.maxColumn = getMaxColumn(lmx.z, gParams.nbCols);
                r.minyF = float(f2uSat(lmn.y));
                r.maxyF = float(f2uSat(lmx.y));

                if ((2u * (r.maxColumn - r.minColumn) * (r.maxRow - r.minRow)) > 0u)
                    r.nbPairs = hfComputePairs(r);
            }
            gDebug[i] = float4(float(r.nbPairs), lmn.y, lmx.y, float((r.maxColumn - r.minColumn) * (r.maxRow - r.minRow)));
        }
        else
            gDebug[i] = float4(-1, 0, 0, 0);

        totalCount += r.nbPairs;
        laneRange[lane] = r;
    }

    uint startIndex = 0;
    if (totalCount > 0)
        InterlockedAdd(gCounter[0], totalCount, startIndex);

    for (uint lane = 0; lane < 32; ++lane)
    {
        if (laneRange[lane].nbPairs != 0)
            hfOutputPairs(laneRange[lane], laneCm[lane], laneElem[lane],
                          startIndex + laneOffsets[lane], gParams.stackSize);
    }
}
