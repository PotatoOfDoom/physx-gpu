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
//     mPosition_InvMass @64, mPackedNodeBounds @280, mFlagsWord @416 (mBodyFlags =
//     byte 1) and mOriginalContactOffset @424 are the fields this kernel touches;
//   - the BV32 tree layout is the SAME one the cloth refit decodes: a uint4 header
//     (nbVerts, nbTets, maxDepth, nbBv32PackedNodes) @0, then nbBv32PackedNodes x
//     Gu::BV32DataPacked (1168 B), then maxDepth x Gu::BV32DataDepthInfo (8 B), then
//     nbBv32PackedNodes x PxU32 remapPackedNodeIndex;
//   - PxBounds3 is a 24-byte { PxVec3 minimum; PxVec3 maximum; } with NO padding.
//
// SCHEDULE — many one-warp workgroups per softbody, bottom-up "last arriver" (2026-10).
// IDENTICAL to femClothRefitBda.slang's (see that header for the full argument; the
// helpers below are verbatim copies): grid (nbActiveSoftBodies, nbSlices, 1), group = ONE
// warp; group (x, s) owns the nodes at remap positions s, s + nbSlices, ...
// (nbSlices = WorkgroupCount().y; the host launches numSlicesPerSoftbody = 256, see the
// PXG_NO_CUDA_TOOLKIT block in PxgSoftBodyCore::refitBound; any slice count >= 1 is
// correct).  An owned node without internal children is refit at once; after refitting
// node c the group arrives at c's parent with a device-scope atomic, and only the LAST
// arriving child refits the parent and climbs on — nobody waits for another workgroup.
// Each node is refit by exactly one warp with the unchanged per-node body, after all of
// its children published their bounds, reading them in fixed lane order: bit-exact
// with the CUDA level walk.  Ordering: device-scope release/acquire barriers around
// the arrival atomic, and mPackedNodeBounds (the only datum crossing workgroups) moved
// with device-scope atomic loads/stores.  Storage: the arrival counter of node p is p.mMin[0].w (0.0f
// in every PhysX tree and rewritten to 0.0f by p's own refit, so it is 0 at every launch
// entry); the parent-link cache is the node's padding[0] @1160, validated before use.
// Until 2026-10 this kernel was ONE 32-thread group per softbody walking EVERY node of
// the tree serially — one wave on one compute unit per softbody.
//
// DEVIATIONS from CUDA — the same ones femClothRefitBda documents, for the same
// reasons (see that file's header for the full argument):
//   - the schedule above, in place of the CUDA (32, SB_REFIT_WAPRS_PER_BLOCK=32) block
//     walking the depth levels with a __syncthreads between them;
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
static const uint64_t BV32_PADDING_OFFSET  = 1160UL;   // padding[0]: the parent-link cache
static const uint64_t BV32_COUNTER_OFFSET  = 12UL;     // mMin[0].w: the arrival counter
static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11u;

// PxBounds3: { PxVec3 minimum; PxVec3 maximum; } — 24 bytes, no padding.
static const uint64_t BOUNDS3_STRIDE = 24UL;
// Gu::BV32DataDepthInfo (GuBV32.h:114): { PxU32 offset; PxU32 count; } — 8 bytes.
static const uint64_t DEPTHINFO_STRIDE = 8UL;

static const float PX_MAX_F32 = 3.4028234663852886e+38f;
static const uint WARP = 32u;

// "no parent found" / "level unknown".
static const uint NO_LINK = 0xFFFFFFFFu;
static const uint NO_LEVEL = 0xFFFFFFFFu;

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
// mPackedNodeBounds[c] crosses workgroups (child's group -> parent's group), so it moves
// through device-scope relaxed atomics of the float bits: the GLSL450-model Ptr<>
// accesses are not Coherent (see femClothRefitBda.slang).  Same bits in, same bits out.
float boundsWordLoad(uint64_t a) { return asfloat(Ptr<Atomic<uint>>(a)[0].load()); }
void boundsWordStore(uint64_t a, float v) { Ptr<Atomic<uint>>(a)[0].store(asuint(v)); }
float3 boundsMin(uint64_t base, uint idx)
{
    const uint64_t a = base + uint64_t(idx) * BOUNDS3_STRIDE;
    return float3(boundsWordLoad(a), boundsWordLoad(a + 4UL), boundsWordLoad(a + 8UL));
}
float3 boundsMax(uint64_t base, uint idx)
{
    const uint64_t a = base + uint64_t(idx) * BOUNDS3_STRIDE + 12UL;
    return float3(boundsWordLoad(a), boundsWordLoad(a + 4UL), boundsWordLoad(a + 8UL));
}
void boundsStore(uint64_t base, uint idx, float3 mn, float3 mx)
{
    const uint64_t a = base + uint64_t(idx) * BOUNDS3_STRIDE;
    boundsWordStore(a, mn.x); boundsWordStore(a + 4UL, mn.y); boundsWordStore(a + 8UL, mn.z);
    boundsWordStore(a + 12UL, mx.x); boundsWordStore(a + 16UL, mx.y); boundsWordStore(a + 20UL, mx.z);
}

// PxMin/PxMax (PxMath.h) — `a < b ? a : b`, the exact form the CPU reference transcribes.
float pxMin(float a, float b) { return a < b ? a : b; }
float pxMax(float a, float b) { return a > b ? a : b; }
float3 pxMin3(float3 a, float3 b) { return float3(pxMin(a.x, b.x), pxMin(a.y, b.y), pxMin(a.z, b.z)); }
float3 pxMax3(float3 a, float3 b) { return float3(pxMax(a.x, b.x), pxMax(a.y, b.y), pxMax(a.z, b.z)); }

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Warp (== group) collectives, groupshared-lowered.  Every helper below is called by
// all 32 lanes together (group-uniform control flow: every branch around a call
// depends only on ballot results, broadcast words or values every lane loaded from
// the same address).
// ---------------------------------------------------------------------------
groupshared uint sMask;
groupshared uint sWord;            // lane-0 -> group broadcast (parent link, last-arriver flag)
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


// The value `v` of lane `srcLane` in every lane.
uint broadcast32(uint v, uint srcLane, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == srcLane) sWord = v;
    GroupMemoryBarrierWithGroupSync();
    return sWord;
}

// P9.86: warpShuffleMin/Max (shuffle.cuh:49/62) combine with fminf/fmaxf, NOT PxMin/PxMax.
// NVIDIA min/max.f32: a NaN operand is dropped (NaN,NaN -> canonical 0x7FFFFFFF) and a
// zero tie orders -0 < +0 regardless of operand order (so the butterfly is order-free).
// The PxMin select keeps b on a tie -> the serial scan returned the LAST tied lane's sign.
float sbrCuFminf(float a, float b)
{
    if (isnan(a)) return isnan(b) ? asfloat(0x7FFFFFFFu) : b;
    if (isnan(b)) return a;
    if (a == b) return asfloat(asuint(a) | asuint(b));   // +-0 tie -> -0; equal non-zero: same bits
    return a < b ? a : b;
}
float sbrCuFmaxf(float a, float b)
{
    if (isnan(a)) return isnan(b) ? asfloat(0x7FFFFFFFu) : b;
    if (isnan(b)) return a;
    if (a == b) return asfloat(asuint(a) & asuint(b));   // +-0 tie -> +0; equal non-zero: same bits
    return a > b ? a : b;
}
float3 sbrCuFminf3(float3 a, float3 b) { return float3(sbrCuFminf(a.x, b.x), sbrCuFminf(a.y, b.y), sbrCuFminf(a.z, b.z)); }
float3 sbrCuFmaxf3(float3 a, float3 b) { return float3(sbrCuFmaxf(a.x, b.x), sbrCuFmaxf(a.y, b.y), sbrCuFmaxf(a.z, b.z)); }

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
        rn = sbrCuFminf3(rn, float3(sRedMin[i * 3 + 0], sRedMin[i * 3 + 1], sRedMin[i * 3 + 2]));   // P9.86: fminf, shuffle.cuh:53-55
        rx = sbrCuFmaxf3(rx, float3(sRedMax[i * 3 + 0], sRedMax[i * 3 + 1], sRedMax[i * 3 + 2]));   // P9.86: fmaxf, shuffle.cuh:66-68
    }
    GroupMemoryBarrierWithGroupSync();
    mn = rn; mx = rx;
}

// ---------------------------------------------------------------------------
// The CUDA per-node body (softBody.cu:157-263), unchanged: refit packed node
// `nodeIndex` from the current tetrahedron positions (leaf lanes) and the already
// refit mPackedNodeBounds of its children (internal lanes).
// ---------------------------------------------------------------------------
void refitNode(uint64_t treeAddr, uint nodeIndex, uint idx, uint64_t tetmeshVerts, uint64_t tetmeshTetIndices,
               uint64_t sPackedNodeBounds)
{
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

// ---------------------------------------------------------------------------
// The last-arriver schedule (see the header).
// ---------------------------------------------------------------------------

// Depth level (BV32DataDepthInfo index, 0 = root) of remap position j; NO_LEVEL if no
// level covers it.
uint levelOfRemapPos(uint64_t depthInfoAddr, uint maxDepth, uint j)
{
    uint level = NO_LEVEL;
    for (uint i = 0u; i < maxDepth; ++i)
    {
        const uint2 info = Ptr<uint2>(depthInfoAddr + uint64_t(i) * DEPTHINFO_STRIDE)[0];
        if (j >= info.x && j - info.x < info.y)
            level = i;
    }
    return level;
}

// A cached parent link (p << 5) | lane is valid for node c iff p.mData[lane] really is
// an internal child pointing at c.  A tree has exactly one such slot, so a valid link
// IS c's parent, whatever wrote it.  Uniform: every lane loads the same words.
bool parentLinkValid(uint64_t treeAddr, uint nbNodes, uint c, uint link)
{
    if (link == NO_LINK)
        return false;
    const uint p = link >> 5u;
    const uint lane = link & 31u;
    if (p >= nbNodes || p == c)
        return false;
    if (lane >= bvNbNodes(treeAddr, p))
        return false;
    const uint d = bvData(treeAddr, p, lane);
    return (d & 1u) == 0u && (d >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) == c;
}

// Search the nodes at remap positions [begin, end) for c's parent, one candidate per
// lane; returns its link (p << 5) | lane, or NO_LINK.
uint searchParent(uint64_t treeAddr, uint64_t remapAddr, uint begin, uint end, uint c, uint idx)
{
    uint link = NO_LINK;
    for (uint base = begin; base < end && link == NO_LINK; base += WARP)   // group-uniform
    {
        const uint pos = base + idx;
        uint found = NO_LINK;
        if (pos < end)
        {
            const uint p = Ptr<uint>(remapAddr + uint64_t(pos) * 4UL)[0];
            const uint nb = min(bvNbNodes(treeAddr, p), WARP);
            for (uint k = 0u; k < nb; ++k)
            {
                const uint d = bvData(treeAddr, p, k);
                if ((d & 1u) == 0u && (d >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) == c)
                    found = (p << 5u) | k;
            }
        }
        const uint hits = ballot32(found != NO_LINK, idx);
        if (hits != 0u)
            link = broadcast32(found, firstbitlow(hits), idx);
    }
    return link;
}

// Node c's parent link; `level` is c's depth level or NO_LEVEL, `parentLevel` receives
// the parent's when known.  Caches a searched link in c's padding[0].
uint findParent(uint64_t treeAddr, uint64_t depthInfoAddr, uint64_t remapAddr, uint nbNodes, uint c,
                uint level, out uint parentLevel, uint idx)
{
    const uint64_t cacheAddr = bvNodeBase(treeAddr, c) + BV32_PADDING_OFFSET;
    // PhysX trees put a node's parent exactly one level up (BV32Tree::createSOAformatNode);
    // that is only a search hint here, the validation does not rely on it.
    parentLevel = (level != NO_LEVEL && level > 0u) ? level - 1u : NO_LEVEL;

    const uint cached = Ptr<uint>(cacheAddr)[0];
    if (parentLinkValid(treeAddr, nbNodes, c, cached))
        return cached;

    uint link = NO_LINK;
    if (parentLevel != NO_LEVEL)
    {
        const uint2 info = Ptr<uint2>(depthInfoAddr + uint64_t(parentLevel) * DEPTHINFO_STRIDE)[0];
        link = searchParent(treeAddr, remapAddr, info.x, info.x + info.y, c, idx);
    }
    if (link == NO_LINK)
    {
        parentLevel = NO_LEVEL;
        link = searchParent(treeAddr, remapAddr, 0u, nbNodes, c, idx);
    }
    if (link != NO_LINK && idx == 0u)
        Ptr<uint>(cacheAddr)[0] = link;
    return link;
}

// Arrive at parent p after refitting one of its internal children; true in every lane
// iff this group is the LAST of p's internal children to arrive (it then refits p).
bool arriveAtParent(uint64_t treeAddr, uint p, uint idx)
{
    const uint d = bvData(treeAddr, p, idx);
    const uint internalMask = ballot32(idx < bvNbNodes(treeAddr, p) && (d & 1u) == 0u, idx);

    // Release: every lane's stores of the child (mMin/mMax lanes, mPackedNodeBounds)
    // happen before lane 0's arrival, at device scope.
    AllMemoryBarrierWithGroupSync();
    if (idx == 0u)
    {
        uint old;
        InterlockedAdd(Ptr<uint>(bvNodeBase(treeAddr, p) + BV32_MMIN_OFFSET + BV32_COUNTER_OFFSET)[0], 1u, old);
        sWord = (old + 1u == countbits(internalMask)) ? 1u : 0u;
    }
    // Acquire: the last arriver's lanes see every sibling's stores before refitting p.
    AllMemoryBarrierWithGroupSync();
    const bool last = sWord != 0u;
    GroupMemoryBarrierWithGroupSync();   // sWord is reused by the next broadcast
    return last;
}

// ---------------------------------------------------------------------------
// sb_refitBoundLaunch (softBody.cu:73).
// Dispatch: grid (nbActiveSoftBodies, nbSlices >= 1, 1), block (32, 1, 1).
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

    const uint nbSlices = WorkgroupCount().y;
    for (uint j = gid.y; j < nbBv32PackedNodes; j += nbSlices)
    {
        uint c = Ptr<uint>(remapAddr + uint64_t(j) * 4UL)[0];

        // A node with internal children is refit by its last-arriving child, not here.
        const uint cData = bvData(treeAddr, c, idx);
        if (ballot32(idx < bvNbNodes(treeAddr, c) && (cData & 1u) == 0u, idx) != 0u)
            continue;

        uint level = levelOfRemapPos(depthInfoAddr, maxDepth, j);
        bool climbing = true;
        while (climbing)
        {
            refitNode(treeAddr, c, idx, tetmeshVerts, tetmeshTetIndices, sPackedNodeBounds);

            if (c == 0u)
            {
                // The root is done, the whole tree is: update the broad phase bound
                // (CUDA: `if (warpIndex == 0)`).  lane 0's mPackedNodeBounds[0] store
                // must reach lanes 1..5.
                AllMemoryBarrierWithGroupSync();

                const uint index = Ptr<uint>(gP.elemIndex)[softbodyId];

                // we don't need to add on contactDist because the broad phase will add
                // contact dist automatically; we need to update the contact distance to
                // reflect CCD changes
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
                    const float r = boundsWordLoad(sPackedNodeBounds + uint64_t(idx) * 4UL);   // pairs with lane 0's atomic store
                    Ptr<float>(gP.boundArray + uint64_t(index) * BOUNDS3_STRIDE)[idx] = r;
                }
                climbing = false;
            }
            else
            {
                uint parentLevel;
                const uint link = findParent(treeAddr, depthInfoAddr, remapAddr, nbBv32PackedNodes, c, level,
                                             parentLevel, idx);
                // No parent (a malformed tree): nothing above c can be refit.
                bool last = false;
                if (link != NO_LINK)
                    last = arriveAtParent(treeAddr, link >> 5u, idx);
                if (last)
                {
                    c = link >> 5u;
                    level = parentLevel;
                }
                else
                {
                    climbing = false;
                }
            }
        }
    }
}
