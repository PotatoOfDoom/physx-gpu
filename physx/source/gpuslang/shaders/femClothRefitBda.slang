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

// femClothRefitBda.slang
//
// P7.2 tranche 3a — the cloth BVH BOUNDS REFIT on the Vulkan engine ABI (BDA).
// Ported line-for-line from gpusimulationcontroller/src/CUDA/FEMCloth.cu:338
// (cloth_refitBoundLaunch, CLOTH_REFIT_BOUND, launch site PxgFEMClothCore.cpp:558).
//
// This is the missing link of the cloth self-collision arc: the tranche-1 midphase
// (femClothSelfCollisionBda) walks the cloth's OWN BV32 triangle tree, but that tree
// is built once at cook time in REST pose.  Without a per-step refit the live midphase
// queries a stale BVH and finds NULL contacts.  This kernel walks the tree bottom-up
// (deepest depth level first) and rewrites, for every packed node, the 32 per-child
// AABBs (mMin/mMax) plus the node's own aggregate bound in mPackedNodeBounds — from
// the CURRENT mPosition_InvMass — then expands the root bound into the broad-phase
// boundArray slot.
//
// Doctrine (identical to the prior cloth BDA tranches femClothSelfCollisionBda /
// femClothContactGenBda):
//   - the PxgFEMCloth descriptor arrives as a device-pointer ARRAY read through Ptr<>
//     (the byte-exact pxgSlangFEMClothDesc mirror); mTriMeshData @0, mPosition_InvMass
//     @16, mTriangleVertexIndices @72, mPackedNodeBounds @272 and mOriginalContactOffset
//     @388 are the fields this kernel touches;
//   - the BV32 tree hangs off mTriMeshData with the SAME layout the midphase decodes:
//     a uint4 header (nbVerts, nbTriangles, maxDepth, nbBv32PackedNodes) @0, then
//     nbBv32PackedNodes x Gu::BV32DataPacked (1168 B: mMin[32] float4 @0, mMax[32]
//     float4 @512, mData[32] uint @1024, mNbNodes @1152), then maxDepth x
//     Gu::BV32DataDepthInfo (8 B: offset, count), then nbBv32PackedNodes x PxU32
//     remapPackedNodeIndex.  Read/written field-by-field by explicit byte offset
//     through Ptr<> (no whole-node copy);
//   - Gu::BV32DataPacked child decode is verbatim: isLeaf(k) = mData[k]&1,
//     getNbReferencedPrimitives(k) = (mData[k]>>1)&63, getPrimitiveStartIndex(k) =
//     mData[k]>>7, getChildOffset(k) = mData[k]>>11;
//   - PxBounds3 is a 24-byte { PxVec3 minimum; PxVec3 maximum; } with NO padding (the
//     CUDA tail writes 6 consecutive floats through `float* r = &root.minimum.x`), so
//     mPackedNodeBounds / boundArray are addressed as stride-24 float runs.
//
// DEVIATIONS from CUDA (recipe policy; each is bit-exactness-preserving, and the
// whole set is exercised by testFemClothSim.cpp's femClothSimRefitBound case):
//   - ONE 32-thread group per cloth (numthreads(32,1,1)) in place of the CUDA
//     (32, SB_REFIT_WAPRS_PER_BLOCK=32, 1) block, i.e. warpIndex is pinned to 0 and
//     the per-depth-level node loop `for (j = warpIndex; j < count; j += 32)` runs
//     SERIALLY over j = 0..count-1 in the single warp.  Required, not cosmetic: the
//     CUDA block synchronizes 32 warps with __syncthreads() between depth levels while
//     each warp independently runs a data-dependent inner loop (the leaf ballot walk
//     `for (k = resultWarp; k; k = clearLowestSetBit(k))` and a j-trip-count that
//     differs by one across warps).  Lowering __ballot_sync/__shfl_xor_sync to
//     groupshared needs a GroupMemoryBarrierWithGroupSync, which must be reached
//     UNIFORMLY by the whole workgroup — impossible with 32 divergent warps in one
//     group.  With group == warp, resultWarp / count / maxDepth are all group-uniform
//     and every barrier is uniform.  Serializing is OUTPUT-IDENTICAL: nodes at the
//     same depth level have no data dependency on each other (each reads only its
//     children — written in a PREVIOUS level, behind the level barrier — and writes
//     only its own mMin/mMax[0..31] and mPackedNodeBounds[nodeIndex]), so the order
//     in which a level's nodes are processed cannot change any result;
//   - __ballot_sync(FULL_MASK, pred) -> groupshared InterlockedOr mask (order-
//     independent); warpShuffleMin/Max (shuffle.cuh:49/62, an XOR butterfly over all
//     32 lanes leaving the result in every lane) -> a groupshared 32-slot exchange
//     reduced identically in every lane.  min/max are exact, associative and
//     commutative on floats (no NaN reaches here: every input is a mesh vertex or a
//     +-PX_MAX_F32 identity), so ANY reduction order gives the SAME bits — the
//     lowering is bit-exact, not merely equivalent.  Both butterflies are fused into
//     one warpMinMax32() call since CUDA issues them back-to-back on independent data;
//   - the per-warp femClothRefitMidphaseScratch (meshVerts / indices / bv32 pointers
//     cached by lane 0 + __syncthreads) is dropped: every lane recomputes those
//     addresses directly from the header through Ptr<> (semantically identical read
//     fan-out; same as femClothSelfCollisionBda);
//   - CUDA computes `offset = warpScanExclusive(resultWarp, idx)` and `validCount =
//     __popc(resultWarp)` inside the node loop and NEVER READS EITHER (dead in the
//     .cu — the leaf walk is driven by the `k = resultWarp` bit loop, not by a
//     compacted slot index).  Omitted here; no observable effect;
//   - the leaf ballot walk `for (k = resultWarp; k; k = clearLowestSetBit(k))` becomes a
//     fixed 32-iteration scan over the same mask.  This is a DRIVER workaround, not a
//     semantic choice: `k &= k-1` is the canonical popcount loop idiom and llvmpipe's
//     LLVM 20.1 backend miscompiles it here (SIGSEGV in the driver worker thread) as soon
//     as the body is non-trivial — bisected to that loop, with the primitive gather and
//     the groupshared reduction EACH reproducing it alone, and an empty-body bit walk
//     running clean.  The scan visits the identical set bits in the identical ascending
//     order, so the emitted bounds are bit-for-bit the same.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS).  The refit
// itself is pure min/max (no rounding freedom at all), but the broad-phase tail
// computes `PxMax(nbCollisionPairUpdatesPerTimestep, 1.0f) * contactDist` and then
// `r[idx] -/+ broadPhaseExpansion`, a mul feeding an add/sub that WOULD fuse into an
// FMA and break bitwise parity with the -ffp-contract=off CPU reference.  There is no
// transcendental anywhere in this kernel (min/max/mul/add only) — this tranche is
// FULLY bit-exact, no tolerance seam.

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mTriMeshData@0, mPosition_InvMass@16, ...)

// Gu::BV32DataPacked (GuBV32.h:88) layout, 1168 bytes — identical constants to
// femClothSelfCollisionBda.slang.
static const uint64_t BV32_STRIDE          = 1168UL;
static const uint64_t BV32_MMIN_OFFSET     = 0UL;
static const uint64_t BV32_MMAX_OFFSET     = 512UL;
static const uint64_t BV32_MDATA_OFFSET    = 1024UL;
static const uint64_t BV32_MNBNODES_OFFSET = 1152UL;
static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11u;

// PxBounds3: { PxVec3 minimum; PxVec3 maximum; } — 24 bytes, no padding.
static const uint64_t BOUNDS3_STRIDE = 24UL;

// Gu::BV32DataDepthInfo (GuBV32.h:114): { PxU32 offset; PxU32 count; } — 8 bytes.
static const uint64_t DEPTHINFO_STRIDE = 8UL;

static const float PX_MAX_F32 = 3.4028234663852886e+38f;
static const uint WARP = 32u;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA refitBound arg pack
// (PxgFEMClothCore.cpp:563): { PxgFEMCloth* gFemClothes, const PxU32* activeFemCloths,
// const PxU32 nbActiveFemCloths (5.12 spelling; was *Clothes), PxReal* contactDists, PxReal* speculativeCCDContactOffset,
// PxBounds3* boundArray, const PxU32* elemIndex, const PxReal nbCollisionPairUpdatesPerTimestep,
// const PxU8* updateContactPairs } -> 72-byte natural-alignment blob.
// (ABI row {8,8,4,8,8,8,8,4,8}.)
struct RefitParams
{
    uint64_t femClothes;                        // @0
    uint64_t activeFemCloths;                  // @8
    uint nbActiveFemCloths;                    // @16
    uint _pad20;                                // @20 (tail-pad realigning the next pointer)
    uint64_t contactDists;                      // @24  (unused; ABI parity — the CUDA body
                                                //       never reads it either)
    uint64_t speculativeCCDContactOffset;       // @32
    uint64_t boundArray;                        // @40  output
    uint64_t elemIndex;                         // @48
    float nbCollisionPairUpdatesPerTimestep;    // @56
    uint _pad60;                                // @60
    uint64_t updateContactPairs;                // @64  -> 72
}

[[vk::push_constant]] ConstantBuffer<RefitParams> gP;

// ---------------------------------------------------------------------------
// BV32DataPacked field accessors (explicit byte offset through Ptr<>).
// treeAddr = mTriMeshData + sizeof(uint4).
// ---------------------------------------------------------------------------
uint64_t bvNodeBase(uint64_t treeAddr, uint nodeIdx) { return treeAddr + uint64_t(nodeIdx) * BV32_STRIDE; }

uint bvData(uint64_t treeAddr, uint nodeIdx, uint k)
{
    return Ptr<uint>(bvNodeBase(treeAddr, nodeIdx) + BV32_MDATA_OFFSET + uint64_t(k) * 4UL)[0];
}
uint bvNbNodes(uint64_t treeAddr, uint nodeIdx)
{
    return Ptr<uint>(bvNodeBase(treeAddr, nodeIdx) + BV32_MNBNODES_OFFSET)[0];
}
// reinterpret_cast<float4&>(currentNode.mMin[idx]) = make_float4(v, 0.f)
void bvStoreMin(uint64_t treeAddr, uint nodeIdx, uint k, float3 v)
{
    Ptr<float4>(bvNodeBase(treeAddr, nodeIdx) + BV32_MMIN_OFFSET + uint64_t(k) * 16UL)[0] = float4(v, 0.0f);
}
void bvStoreMax(uint64_t treeAddr, uint nodeIdx, uint k, float3 v)
{
    Ptr<float4>(bvNodeBase(treeAddr, nodeIdx) + BV32_MMAX_OFFSET + uint64_t(k) * 16UL)[0] = float4(v, 0.0f);
}

// PxBounds3 accessors (stride-24 float runs; NOT float3-aligned).
float3 boundsMin(uint64_t base, uint idx)
{
    const uint64_t a = base + uint64_t(idx) * BOUNDS3_STRIDE;
    return float3(Ptr<float>(a)[0], Ptr<float>(a)[1], Ptr<float>(a)[2]);
}
float3 boundsMax(uint64_t base, uint idx)
{
    const uint64_t a = base + uint64_t(idx) * BOUNDS3_STRIDE + 12UL;
    return float3(Ptr<float>(a)[0], Ptr<float>(a)[1], Ptr<float>(a)[2]);
}
void boundsStore(uint64_t base, uint idx, float3 mn, float3 mx)
{
    const uint64_t a = base + uint64_t(idx) * BOUNDS3_STRIDE;
    Ptr<float>(a)[0] = mn.x; Ptr<float>(a)[1] = mn.y; Ptr<float>(a)[2] = mn.z;
    Ptr<float>(a)[3] = mx.x; Ptr<float>(a)[4] = mx.y; Ptr<float>(a)[5] = mx.z;
}

// PxMin/PxMax (PxMath.h) — `a < b ? a : b`, NOT the HLSL intrinsic (identical for the
// non-NaN inputs here, but the ternary is what the CPU reference transcribes).
float pxMin(float a, float b) { return a < b ? a : b; }
float pxMax(float a, float b) { return a > b ? a : b; }
float3 pxMin3(float3 a, float3 b) { return float3(pxMin(a.x, b.x), pxMin(a.y, b.y), pxMin(a.z, b.z)); }
float3 pxMax3(float3 a, float3 b) { return float3(pxMax(a.x, b.x), pxMax(a.y, b.y), pxMax(a.z, b.z)); }

// ---------------------------------------------------------------------------
// Warp (== group) collectives, groupshared-lowered.
// ---------------------------------------------------------------------------
groupshared uint sMask;
groupshared float sRedMin[WARP * 3];
groupshared float sRedMax[WARP * 3];

// __ballot_sync(FULL_MASK, pred) over the 32-thread group.  The leading barrier orders
// the zero-write after every read of the previous ballot's mask.
uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// P9.86: warpShuffleMin/Max (shuffle.cuh:48-70) reduce with CUDA fminf/fmaxf, NOT the
// PxMin/PxMax selects: NaN-ignoring minNum/maxNum with -0 < +0 (fminf(+0,-0) = -0,
// fmaxf(+0,-0) = +0) regardless of operand order; both-NaN -> canonical NaN.  Spelled
// as selects so neither slangc nor llvmpipe's FMin/FMax semantics apply.
float fcrFminf(float a, float b)
{
    if (isnan(a) && isnan(b)) return asfloat(0x7FFFFFFFu);
    return isnan(a) ? b : (isnan(b) ? a : ((a == b) ? asfloat(asuint(a) | asuint(b)) : (a < b ? a : b)));
}
float fcrFmaxf(float a, float b)
{
    if (isnan(a) && isnan(b)) return asfloat(0x7FFFFFFFu);
    return isnan(a) ? b : (isnan(b) ? a : ((a == b) ? asfloat(asuint(a) & asuint(b)) : (a > b ? a : b)));
}
float3 fcrFmin3(float3 a, float3 b) { return float3(fcrFminf(a.x, b.x), fcrFminf(a.y, b.y), fcrFminf(a.z, b.z)); }
float3 fcrFmax3(float3 a, float3 b) { return float3(fcrFmaxf(a.x, b.x), fcrFmaxf(a.y, b.y), fcrFmaxf(a.z, b.z)); }

// warpShuffleMin(mn) + warpShuffleMax(mx) fused (shuffle.cuh:49/62): both reduce over
// all 32 lanes and leave the result in EVERY lane.  fminf/fmaxf (fcrFmin3/fcrFmax3) are
// exact + associative + commutative (incl. +/-0 and NaN), so the linear scan below is
// bit-identical to CUDA's XOR butterfly.
void warpMinMax32(inout float3 mn, inout float3 mx, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    sRedMin[idx * 3 + 0] = mn.x; sRedMin[idx * 3 + 1] = mn.y; sRedMin[idx * 3 + 2] = mn.z;
    sRedMax[idx * 3 + 0] = mx.x; sRedMax[idx * 3 + 1] = mx.y; sRedMax[idx * 3 + 2] = mx.z;
    GroupMemoryBarrierWithGroupSync();
    float3 rn = float3(sRedMin[0], sRedMin[1], sRedMin[2]);
    float3 rx = float3(sRedMax[0], sRedMax[1], sRedMax[2]);
    for (uint i = 1u; i < WARP; ++i)
    {
        // P9.86: CUDA fminf/fmaxf (shuffle.cuh:48-70, FEMCloth.cu:469-470/487-488), not PxMin/PxMax.
        rn = fcrFmin3(rn, float3(sRedMin[i * 3 + 0], sRedMin[i * 3 + 1], sRedMin[i * 3 + 2]));
        rx = fcrFmax3(rx, float3(sRedMax[i * 3 + 0], sRedMax[i * 3 + 1], sRedMax[i * 3 + 2]));
    }
    GroupMemoryBarrierWithGroupSync();
    mn = rn; mx = rx;
}

// ---------------------------------------------------------------------------
// cloth_refitBoundLaunch (FEMCloth.cu:338).
// Dispatch: grid (nbActiveFemCloths, 1, 1), block (32, 1, 1).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_refitBoundLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // Early exit if contact pairs are not updated.  (PxU8 read as the low byte.)
    if ((Ptr<uint>(gP.updateContactPairs)[0] & 0xFFu) == 0u)
        return;

    const uint idx = gtid.x;
    const uint blockIdx = gid.x;
    if (blockIdx >= gP.nbActiveFemCloths) return;   // group-uniform

    const uint femClothId = Ptr<uint>(gP.activeFemCloths)[blockIdx];

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gP.femClothes);

    const uint64_t triMeshGeomPtr = clothes[femClothId].mTriMeshData;

    // uint4 header: (nbVerts, nbTriangles, maxDepth, nbBv32PackedNodes)
    const uint4 hdr = Ptr<uint4>(triMeshGeomPtr)[0];
    const uint maxDepth = hdr.z;
    const uint nbBv32PackedNodes = hdr.w;

    const uint64_t sPackedNodeBounds = clothes[femClothId].mPackedNodeBounds;

    // The three tree sections, in the order the CUDA scratch init walks them.
    const uint64_t treeAddr = triMeshGeomPtr + 16UL;                             // BV32DataPacked[]
    const uint64_t depthInfoAddr = treeAddr + uint64_t(nbBv32PackedNodes) * BV32_STRIDE; // BV32DataDepthInfo[]
    const uint64_t remapAddr = depthInfoAddr + uint64_t(maxDepth) * DEPTHINFO_STRIDE;    // PxU32[]

    const uint64_t meshVerts = clothes[femClothId].mPosition_InvMass;
    const uint64_t meshVertsIndices = clothes[femClothId].mTriangleVertexIndices_ownedEdges;

    // each warp deals with one node; walk the depth levels bottom-up
    for (uint i = maxDepth; i > 0u; i--)
    {
        const uint2 info = Ptr<uint2>(depthInfoAddr + uint64_t(i - 1u) * DEPTHINFO_STRIDE)[0];
        const uint offset = info.x;
        const uint count = info.y;

        // CUDA: for (j = warpIndex; j < count; j += SB_REFIT_WAPRS_PER_BLOCK).  One warp
        // per group here -> warpIndex == 0, stride 1 (see the header note: same-level
        // nodes are independent, so the serialization is output-identical).
        for (uint j = 0u; j < count; j++)
        {
            const uint nodeIndex = Ptr<uint>(remapAddr + uint64_t(j + offset) * 4UL)[0];
            const uint nodeData = bvData(treeAddr, nodeIndex, idx);
            const uint nbChildren = bvNbNodes(treeAddr, nodeIndex);
            const bool isLeafLane = (nodeData & 1u) != 0u;

            // compute the bitMask for all the leaf nodes
            const uint resultWarp = ballot32(idx < nbChildren && isLeafLane, idx);

            float3 mn = float3(PX_MAX_F32, PX_MAX_F32, PX_MAX_F32);
            float3 mx = float3(-PX_MAX_F32, -PX_MAX_F32, -PX_MAX_F32);

            // using one warp to compute all leaf nodes' (under the same current node)
            // min and max
            // CUDA walks the set bits with `for (k = resultWarp; k; k = clearLowestSetBit(k))`
            // and lowestSetIndex(k).  That is the canonical popcount loop idiom
            // (`k &= k-1`), and llvmpipe's LLVM 20.1 backend MISCOMPILES it here: with a
            // non-trivial body (either the primitive gather or the groupshared reduction —
            // each alone reproduces it) the driver worker thread takes a SIGSEGV.  The
            // equivalent fixed 32-iteration scan below sidesteps the idiom entirely and is
            // OUTPUT-IDENTICAL: it visits exactly the same set bits in the same ascending
            // order, and the skipped slots contribute nothing.  (An empty-body bit walk
            // runs fine, so this is a codegen bug, not a Slang/SPIR-V semantics issue.)
            for (uint kk = 0u; kk < WARP; ++kk)
            {
                if (((resultWarp >> kk) & 1u) == 0u) continue;   // group-uniform: resultWarp is the ballot
                const uint indexInWarp = kk;

                // primitive: triangle
                const uint childData = bvData(treeAddr, nodeIndex, indexInWarp);
                const uint nbPrimitives = (childData >> 1u) & 63u;
                const uint startIndex = childData >> 7u;

                float3 tMin = mn;
                float3 tMax = mx;

                // each thread in a warp deals with one primitive (maximum 32 primitives)
                const uint primitiveIndex = idx + startIndex;

                if (idx < nbPrimitives)
                {
                    const uint4 primitiveIdx = Ptr<uint4>(meshVertsIndices)[primitiveIndex];

                    const float3 worldV0 = Ptr<float4>(meshVerts)[primitiveIdx.x].xyz;
                    const float3 worldV1 = Ptr<float4>(meshVerts)[primitiveIdx.y].xyz;
                    const float3 worldV2 = Ptr<float4>(meshVerts)[primitiveIdx.z].xyz;

                    float tx = pxMin(worldV0.x, worldV1.x);
                    float ty = pxMin(worldV0.y, worldV1.y);
                    float tz = pxMin(worldV0.z, worldV1.z);

                    tx = pxMin(tx, worldV2.x);
                    ty = pxMin(ty, worldV2.y);
                    tz = pxMin(tz, worldV2.z);

                    tMin.x = pxMin(tMin.x, tx);
                    tMin.y = pxMin(tMin.y, ty);
                    tMin.z = pxMin(tMin.z, tz);

                    tx = pxMax(worldV0.x, worldV1.x);
                    ty = pxMax(worldV0.y, worldV1.y);
                    tz = pxMax(worldV0.z, worldV1.z);

                    tx = pxMax(tx, worldV2.x);
                    ty = pxMax(ty, worldV2.y);
                    tz = pxMax(tz, worldV2.z);

                    tMax.x = pxMax(tMax.x, tx);
                    tMax.y = pxMax(tMax.y, ty);
                    tMax.z = pxMax(tMax.z, tz);
                }

                // min = warpShuffleMin(tMin);  max = warpShuffleMax(tMax);
                warpMinMax32(tMin, tMax, idx);
                mn = tMin;
                mx = tMax;
            }

            if (idx < nbChildren && !isLeafLane)
            {
                const uint childOffset = nodeData >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT;

                mn = boundsMin(sPackedNodeBounds, childOffset);
                mx = boundsMax(sPackedNodeBounds, childOffset);
            }

            if (idx < nbChildren)
            {
                // We already updated the bounds in the previous iterations
                bvStoreMin(treeAddr, nodeIndex, idx, mn);
                bvStoreMax(treeAddr, nodeIndex, idx, mx);
            }

            // sPackedNodeBounds[nodeIndex] = { warpShuffleMin(min), warpShuffleMax(max) }
            float3 rn = mn;
            float3 rx = mx;
            warpMinMax32(rn, rx, idx);
            if (idx == 0u)
                boundsStore(sPackedNodeBounds, nodeIndex, rn, rx);
        }

        // __syncthreads() between depth levels.  Group-uniform: maxDepth and count are
        // block-uniform, and the whole group is one warp.
        GroupMemoryBarrierWithGroupSync();
    }

    // update the broad-phase bound (CUDA: `if (warpIndex == 0)`, always true here).
    {
        const uint index = Ptr<uint>(gP.elemIndex)[femClothId];

        // Expand the bound for broad phase.
        // Unlike soft bodies, the contact distance remains unchanged; otherwise, contact
        // buffers easily overflow for cloth.
        const float contactDist = clothes[femClothId].mOriginalContactOffset +
                                  Ptr<float>(gP.speculativeCCDContactOffset)[femClothId];
        const float broadPhaseExpansion = pxMax(gP.nbCollisionPairUpdatesPerTimestep, 1.0f) * contactDist;

        // PxBounds3& root = sPackedNodeBounds[0];  compute min(0-2) max(3-5)
        if (idx < 6u)
        {
            const float r = Ptr<float>(sPackedNodeBounds)[idx];
            const float value = idx < 3u ? (r - broadPhaseExpansion) : (r + broadPhaseExpansion);
            Ptr<float>(gP.boundArray + uint64_t(index) * BOUNDS3_STRIDE)[idx] = value;
        }
    }
}
