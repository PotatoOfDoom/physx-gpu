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

// softBodyRefitBda.slang
//
// P8.0 tranche 2 — the SOFTBODY (FEM-volume) BVH bounds refit on the Vulkan engine ABI
// (BDA).  Ported line-for-line from gpusimulationcontroller/src/CUDA/softBody.cu:99
// (sb_refitBoundLaunch, SB_REFITBOUND, launch site PxgSoftBodyCore.cpp:283).
//
// Second kernel of the softbody free-fall chain (after sb_gm_preIntegrateLaunch): once
// the vertices move, the softbody's cooked BV32 TETRAHEDRON tree and its broad-phase
// bound are stale, so nothing downstream — midphase, contact gen, the broad phase
// itself — sees the body where it actually is.  This walks the tree bottom-up and
// rewrites every node's 32 per-child AABBs from the CURRENT mPosition_InvMass, then
// publishes the root bound into the broad-phase boundArray slot.
//
// This is the direct SIBLING of femClothRefitBda.slang (P7.2 tranche 3a) — the CUDA
// kernels are the same algorithm over a different primitive — and it reuses that port's
// structure, BV32 decode and warp lowering VERBATIM.  Differences, all from the .cu:
//   - the primitive is a TETRAHEDRON: 4 vertices (uint4 tetIdx -> mPosition_InvMass),
//     min/max folded pairwise (v0v1, v2v3, then the two) exactly as softBody.cu does,
//     vs the cloth triangle's 3;
//   - the tree hangs off mTetMeshData (@0) with mTetIndices (@16) and the COLLISION
//     mesh mPosition_InvMass (@64) — NOT the simulation-mesh mSim* fields;
//   - NO `updateContactPairs` early-out gate (that is a cloth-only mechanism);
//   - the broad-phase tail copies the root bound VERBATIM, with no expansion (the
//     comment in the .cu: "we don't need to add on contactDist because the broad phase
//     will add contact dist automatically"), and additionally — only under
//     PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD — republishes the bp/np contact
//     distance as speculativeCCDContactOffset + mOriginalContactOffset.
//
// Doctrine (identical to femClothRefitBda / the prior softbody tranche):
//   - the PxgSoftBody descriptor arrives as a device-pointer ARRAY read through Ptr<>
//     (the byte-exact pxgSlangSoftBodyDesc mirror); mTetMeshData @0, mTetIndices @16,
//     mPosition_InvMass @64, mPackedNodeBounds @280, mFlagsWord @392 (mBodyFlags =
//     byte 1) and mOriginalContactOffset @424 are the fields this kernel touches;
//   - the BV32 tree layout is the SAME one the cloth refit decodes: a uint4 header
//     (nbVerts, nbTets, maxDepth, nbBv32PackedNodes) @0, then nbBv32PackedNodes x
//     Gu::BV32DataPacked (1168 B), then maxDepth x Gu::BV32DataDepthInfo (8 B), then
//     nbBv32PackedNodes x PxU32 remapPackedNodeIndex;
//   - PxBounds3 is a 24-byte { PxVec3 minimum; PxVec3 maximum; } with NO padding.
//
// DEVIATIONS from CUDA — the same three femClothRefitBda documents, for the same
// reasons (see that file's header for the full argument):
//   - ONE 32-thread group per softbody in place of the CUDA (32,
//     SB_REFIT_WAPRS_PER_BLOCK=32) block, so the per-level node loop runs serially in
//     the single warp.  Forced: barriers must be reached uniformly, impossible with 32
//     divergent warps in one group.  Output-identical — same-level nodes are
//     independent, and min/max is exact + associative + commutative;
//   - __ballot_sync -> groupshared InterlockedOr mask; warpShuffleMin/Max (an XOR
//     butterfly over all 32 lanes, result in every lane) -> a groupshared 32-slot
//     exchange reduced identically in every lane.  Bit-exact, not merely equivalent;
//   - the leaf ballot walk `for (k = resultWarp; k; k = clearLowestSetBit(k))` becomes
//     a fixed 32-iteration scan over the same mask: `k &= k-1` is the canonical
//     popcount loop idiom and llvmpipe's LLVM 20.1 backend miscompiles it (SIGSEGV in
//     the driver worker thread) once the body is non-trivial — bisected on the cloth
//     sibling.  Same set bits, same order, same bits out;
//   - the per-warp sbMidphaseScratch (pointers cached by lane 0 + __syncthreads) is
//     dropped: every lane reads the addresses directly through Ptr<>;
//   - CUDA's `offset = warpScanExclusive(...)` / `validCount = __popc(...)` inside the
//     node loop are dead in the .cu (never read) and omitted.
//
// The kernel is pure min/max plus ONE addition (the speculative-CCD contact distance);
// there is no mul feeding an add anywhere, so nothing can contract into an FMA and no
// transcendental appears — FULLY bit-exact, no tolerance seam.  It is stamped
// NoContraction anyway (PXG_NOCONTRACT_SHADERS) to keep the guarantee independent of
// future edits.

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mTetMeshData@0, mTetIndices@16, ...)

// Gu::BV32DataPacked (GuBV32.h:88) layout, 1168 bytes — identical to femClothRefitBda.
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
// (PxgSoftBodyCore.cpp:284): { PxgSoftBody* gSoftbodies, const PxU32* activeSoftbodies,
// const PxU32 nbActiveSoftBodies, PxReal* bpContactDistance, PxReal* npContactDistance,
// PxReal* speculativeCCDContactOffset, PxBounds3* boundArray, const PxU32* elemIndex }
// -> 64-byte natural-alignment blob.  (ABI row {8,8,4,8,8,8,8,8}.)
struct SbRefitParams
{
    uint64_t softBodies;                    // @0
    uint64_t activeSoftBodies;              // @8
    uint nbActiveSoftBodies;                // @16
    uint _pad20;                            // @20 (tail-pad realigning the next pointer)
    uint64_t bpContactDistance;             // @24  output (speculative-CCD only)
    uint64_t npContactDistance;             // @32  output (speculative-CCD only)
    uint64_t speculativeCCDContactOffset;   // @40
    uint64_t boundArray;                    // @48  output
    uint64_t elemIndex;                     // @56  -> 64
}

[[vk::push_constant]] ConstantBuffer<SbRefitParams> gP;

// ---------------------------------------------------------------------------
// BV32DataPacked field accessors (explicit byte offset through Ptr<>).
// treeAddr = mTetMeshData + sizeof(uint4).
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

// PxMin/PxMax (PxMath.h) — `a < b ? a : b`, the exact form the CPU reference transcribes.
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

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// warpShuffleMin(mn) + warpShuffleMax(mx) fused (shuffle.cuh:49/62).
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
        rn = pxMin3(rn, float3(sRedMin[i * 3 + 0], sRedMin[i * 3 + 1], sRedMin[i * 3 + 2]));
        rx = pxMax3(rx, float3(sRedMax[i * 3 + 0], sRedMax[i * 3 + 1], sRedMax[i * 3 + 2]));
    }
    GroupMemoryBarrierWithGroupSync();
    mn = rn; mx = rx;
}

// ---------------------------------------------------------------------------
// sb_refitBoundLaunch (softBody.cu:99).
// Dispatch: grid (nbActiveSoftBodies, 1, 1), block (32, 1, 1).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_refitBoundLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint idx = gtid.x;
    const uint blockIdx = gid.x;
    if (blockIdx >= gP.nbActiveSoftBodies) return;   // group-uniform

    const uint softbodyId = Ptr<uint>(gP.activeSoftBodies)[blockIdx];

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gP.softBodies);

    const uint64_t tetmeshGeomPtr = softbodies[softbodyId].mTetMeshData;

    // uint4 header: (nbVerts, nbTets, maxDepth, nbBv32PackedNodes)
    const uint4 hdr = Ptr<uint4>(tetmeshGeomPtr)[0];
    const uint maxDepth = hdr.z;
    const uint nbBv32PackedNodes = hdr.w;

    const uint64_t sPackedNodeBounds = softbodies[softbodyId].mPackedNodeBounds;

    const uint64_t treeAddr = tetmeshGeomPtr + 16UL;                                     // BV32DataPacked[]
    const uint64_t depthInfoAddr = treeAddr + uint64_t(nbBv32PackedNodes) * BV32_STRIDE; // BV32DataDepthInfo[]
    const uint64_t remapAddr = depthInfoAddr + uint64_t(maxDepth) * DEPTHINFO_STRIDE;    // PxU32[]

    const uint64_t tetmeshVerts = softbodies[softbodyId].mPosition_InvMass;
    const uint64_t tetmeshTetIndices = softbodies[softbodyId].mTetIndices;

    // each warp deals with one node; walk the depth levels bottom-up
    for (uint i = maxDepth; i > 0u; i--)
    {
        const uint2 info = Ptr<uint2>(depthInfoAddr + uint64_t(i - 1u) * DEPTHINFO_STRIDE)[0];
        const uint offset = info.x;
        const uint count = info.y;

        // CUDA: for (j = warpIndex; j < count; j += SB_REFIT_WAPRS_PER_BLOCK).  One warp
        // per group here -> warpIndex == 0, stride 1 (same-level nodes are independent,
        // so the serialization is output-identical).
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
            // min and max.  Fixed 32-iteration scan in place of CUDA's `k &= k-1` bit
            // walk (see the header: llvmpipe miscompiles that idiom).
            for (uint kk = 0u; kk < WARP; ++kk)
            {
                if (((resultWarp >> kk) & 1u) == 0u) continue;   // group-uniform: resultWarp is the ballot
                const uint indexInWarp = kk;

                const uint childData = bvData(treeAddr, nodeIndex, indexInWarp);
                const uint nbTets = (childData >> 1u) & 63u;
                const uint startIndex = childData >> 7u;

                float3 tMin = mn;
                float3 tMax = mx;

                // each thread in a warp deals with one tetrahedron (maximum 32)
                const uint tetIndex = idx + startIndex;

                if (idx < nbTets)
                {
                    const uint4 tetIdx = Ptr<uint4>(tetmeshTetIndices)[tetIndex];

                    const float3 worldV0 = Ptr<float4>(tetmeshVerts)[tetIdx.x].xyz;
                    const float3 worldV1 = Ptr<float4>(tetmeshVerts)[tetIdx.y].xyz;
                    const float3 worldV2 = Ptr<float4>(tetmeshVerts)[tetIdx.z].xyz;
                    const float3 worldV3 = Ptr<float4>(tetmeshVerts)[tetIdx.w].xyz;

                    float tx = pxMin(worldV0.x, worldV1.x);
                    float ty = pxMin(worldV0.y, worldV1.y);
                    float tz = pxMin(worldV0.z, worldV1.z);

                    float tx2 = pxMin(worldV2.x, worldV3.x);
                    float ty2 = pxMin(worldV2.y, worldV3.y);
                    float tz2 = pxMin(worldV2.z, worldV3.z);

                    tx = pxMin(tx, tx2);
                    ty = pxMin(ty, ty2);
                    tz = pxMin(tz, tz2);

                    tMin.x = pxMin(tMin.x, tx);
                    tMin.y = pxMin(tMin.y, ty);
                    tMin.z = pxMin(tMin.z, tz);

                    tx = pxMax(worldV0.x, worldV1.x);
                    ty = pxMax(worldV0.y, worldV1.y);
                    tz = pxMax(worldV0.z, worldV1.z);

                    tx2 = pxMax(worldV2.x, worldV3.x);
                    ty2 = pxMax(worldV2.y, worldV3.y);
                    tz2 = pxMax(worldV2.z, worldV3.z);

                    tx = pxMax(tx, tx2);
                    ty = pxMax(ty, ty2);
                    tz = pxMax(tz, tz2);

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

        // __syncthreads() between depth levels.
        GroupMemoryBarrierWithGroupSync();
    }

    // update the broad phase bound (CUDA: `if (warpIndex == 0)`, always true here).
    {
        const uint index = Ptr<uint>(gP.elemIndex)[softbodyId];

        // we don't need to add on contactDist because the broad phase will add contact
        // dist automatically; we need to update the contact distance to reflect CCD changes
        const uint bodyFlags = (softbodies[softbodyId].mFlagsWord >> 8) & 0xFFu;
        if ((bodyFlags & PXG_SOFTBODY_BODYFLAG_ENABLE_SPECULATIVE_CCD) != 0u && idx == 0u)
        {
            const float contactDist = Ptr<float>(gP.speculativeCCDContactOffset)[softbodyId] +
                                      softbodies[softbodyId].mOriginalContactOffset;
            Ptr<float>(gP.bpContactDistance)[index] = contactDist;
            Ptr<float>(gP.npContactDistance)[index] = contactDist;
        }

        // PxBounds3& root = sPackedNodeBounds[0];  compute min(0-2) max(3-5)
        if (idx < 6u)
        {
            const float r = Ptr<float>(sPackedNodeBounds)[idx];
            Ptr<float>(gP.boundArray + uint64_t(index) * BOUNDS3_STRIDE)[idx] = r;
        }
    }
}
