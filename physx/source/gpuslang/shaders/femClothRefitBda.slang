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
// Ported from gpusimulationcontroller/src/CUDA/FEMCloth.cu:321
// (cloth_refitBoundLaunch, CLOTH_REFIT_BOUND, launch site PxgFEMClothCore.cpp:558).
//
// This is the missing link of the cloth self-collision arc: the tranche-1 midphase
// (femClothSelfCollisionBda) walks the cloth's OWN BV32 triangle tree, but that tree
// is built once at cook time in REST pose.  Without a per-step refit the live midphase
// queries a stale BVH and finds NULL contacts.  This kernel refits the tree bottom-up
// and rewrites, for every packed node, the 32 per-child AABBs (mMin/mMax) plus the
// node's own aggregate bound in mPackedNodeBounds — from the CURRENT mPosition_InvMass
// — then expands the root bound into the broad-phase boundArray slot.
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
//     float4 @512, mData[32] uint @1024, mNbNodes @1152, mDepth @1156, padding[2]
//     @1160), then maxDepth x Gu::BV32DataDepthInfo (8 B: offset, count), then
//     nbBv32PackedNodes x PxU32 remapPackedNodeIndex.  Read/written field-by-field by
//     explicit byte offset through Ptr<> (no whole-node copy);
//   - Gu::BV32DataPacked child decode is verbatim: isLeaf(k) = mData[k]&1,
//     getNbReferencedPrimitives(k) = (mData[k]>>1)&63, getPrimitiveStartIndex(k) =
//     mData[k]>>7, getChildOffset(k) = mData[k]>>11;
//   - PxBounds3 is a 24-byte { PxVec3 minimum; PxVec3 maximum; } with NO padding (the
//     CUDA tail writes 6 consecutive floats through `float* r = &root.minimum.x`), so
//     mPackedNodeBounds / boundArray are addressed as stride-24 float runs.
//
// SCHEDULE — many one-warp workgroups per cloth, bottom-up "last arriver" (2026-10).
// The CUDA kernel is ONE 1024-thread block per cloth walking the depth levels with a
// __syncthreads between them.  That is one compute unit per cloth: on the RX 9070 XT
// SnippetDeformableSurface (one 79202-triangle sheet, tree levels 1/32/1024) spent ~4 ms
// per launch in it, two launches per step, while the rest of the GPU idled.  Here the
// SAME per-node computation is scheduled across the whole GPU:
//   - grid (nbActiveFemCloths, nbSlices, 1), group = ONE warp (numthreads(32,1,1)).
//     Group (x, s) owns the nodes at remap positions s, s + nbSlices, ... of cloth x
//     (nbSlices = WorkgroupCount().y; the host launches numSlicesPerCloth = 256, see the
//     PXG_NO_CUDA_TOOLKIT block in PxgFEMClothCore::refitBound; ANY slice count >= 1 is
//     correct, the old one-group-per-cloth grid included — fewer slices only serialise);
//   - an owned node with NO internal child (only leaf children: every deepest-level
//     node, and any shallower node of an unbalanced tree) is refit right away.  An owned
//     node WITH internal children is skipped: it is refit by its last-arriving child;
//   - after refitting node c the group ARRIVES at c's parent p: an atomic increment of
//     p's arrival counter.  The group whose increment completes p's count of internal
//     children (the LAST arriver) refits p and climbs on; every other group stops there.
//     Nobody ever waits for another workgroup, so this needs no forward-progress
//     guarantee across workgroups (no spin, no grid barrier);
//   - the group that refits the root (packed node 0, the node the CUDA tail reads)
//     writes the broad-phase bound.
// BIT-EXACTNESS: each node is refit by exactly ONE warp running the unchanged per-node
// body (leaf ballot walk in ascending child order, child bounds read from
// mPackedNodeBounds for internal lanes, the two 32-lane reductions), and a parent is
// refit only after ALL its internal children have published their bounds.  It reads
// the children in fixed lane order, so which child arrived last cannot change a bit.
// That is exactly the CUDA's data dependency (a level reads only the level below it);
// the level barrier only over-constrained it.
// MEMORY ORDERING: a child's stores (its mMin/mMax lanes and mPackedNodeBounds[c]) are
// released by a device-scope AllMemoryBarrierWithGroupSync before lane 0's atomic, and
// the last arriver's lanes acquire with the same device-scope barrier after it, before
// they read any child bound.  The atomic itself is a device-scope InterlockedAdd on
// global memory, so arrivals from different compute units are totally ordered.  The
// handed-over data itself, mPackedNodeBounds[c], is stored and loaded with device-scope
// atomics (boundsStore / boundsMin / boundsMax): the module uses the GLSL450 memory
// model and Ptr<> accesses are not Coherent, so plain accesses would carry no
// cross-workgroup visibility guarantee.  Nothing else crosses workgroups within a launch
// (c's mMin/mMax lanes are not read by the parent; the parent-link cache of c is read
// and written only by the group refitting c).
// STORAGE (no new buffer, no ABI change):
//   - the arrival counter of node p is the w word of p.mMin[0] (as uint bits).  That
//     word is 0.0f in every tree PhysX produces — BV32Tree::createSOAformatNode writes
//     PxVec4(min, 0.f), the cooked stream round-trips it, PxgGeometryManager /
//     PxgSimulationController copy it verbatim — and the refit itself writes
//     make_float4(min, 0.f) into it (CUDA and here alike).  So the counter is 0 at every
//     launch entry, and the consumer resets it with no extra store: the last arriver's
//     refit of p rewrites p.mMin[0] = float4(min, 0) AFTER every increment.  After the
//     launch the tree is bit-identical to the CUDA's, w words included;
//   - the parent link of node c is cached in c's padding[0] (@1160: never serialised,
//     never read by any engine or port kernel) as (p << 5) | lane.  The cache is
//     VALIDATED before use — p in range, lane < p.mNbNodes, p.mData[lane] is an internal
//     child pointing at c — and a tree has exactly one such (p, lane), so stale or
//     garbage padding (a fresh upload) can only miss, never mislead.  A miss searches the
//     level above c (BV32DataDepthInfo; then, if the tree breaks the depth+1 rule, every
//     node) one candidate per lane and caches the result: the first launch on a tree pays
//     the search, later launches pay two loads.
//
// DEVIATIONS from CUDA (recipe policy; each is bit-exactness-preserving, and the whole
// set is exercised by testFemClothSim.cpp's femClothSimRefitBound* cases):
//   - the schedule above, in place of the CUDA (32, SB_REFIT_WAPRS_PER_BLOCK=32) block
//     and its level loop `for (j = warpIndex; j < count; j += 32)` + __syncthreads.
//     Until 115945c this was ONE warp per cloth walking every node serially; 115945c
//     made it the CUDA's 32 warps, still one group per cloth (one CU per cloth);
//   - __ballot_sync(FULL_MASK, pred) -> groupshared InterlockedOr mask (order-
//     independent); warpShuffleMin/Max (shuffle.cuh:49/62, an XOR butterfly over all
//     32 lanes leaving the result in every lane) -> a groupshared 32-slot exchange
//     reduced identically in every lane.  The reducers are CUDA fminf/fmaxf, which are
//     exact, associative and commutative (incl. +/-0 and NaN), so ANY reduction order
//     gives the SAME bits — the lowering is bit-exact, not merely equivalent.  Both
//     butterflies are fused into one warpMinMax32() call since CUDA issues them
//     back-to-back on independent data;
//   - the femClothRefitMidphaseScratch (meshVerts / indices / bv32 pointers cached by
//     lane 0 + __syncthreads) is dropped: every lane recomputes those addresses directly
//     from the header through Ptr<> (semantically identical read fan-out);
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

// Gu::BV32DataPacked (GuBV32.h:65) layout, 1168 bytes — identical constants to
// femClothSelfCollisionBda.slang.
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

// Gu::BV32DataDepthInfo (GuBV32.h:91): { PxU32 offset; PxU32 count; } — 8 bytes.
static const uint64_t DEPTHINFO_STRIDE = 8UL;

static const float PX_MAX_F32 = 3.4028234663852886e+38f;
static const uint WARP = 32u;

// "no parent found" / "level unknown".
static const uint NO_LINK = 0xFFFFFFFFu;
static const uint NO_LEVEL = 0xFFFFFFFFu;

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
// mPackedNodeBounds[c] is the one datum handed from workgroup to workgroup (written by
// the group that refits c, read by the group that refits c's parent), so it is moved
// with device-scope (relaxed) atomic loads/stores of the float bits: the module is on
// the GLSL450 memory model and these Ptr<> accesses carry no Coherent decoration, so a
// plain load in the parent's group is not guaranteed to see another group's plain
// store even after the barriers (it may hit a stale per-CU cache line holding a
// neighbouring node's entry).  Atomics are coherent at their scope; the barriers in
// arriveAtParent order them.  Same bits in, same bits out.
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

// PxMin/PxMax (PxMath.h) — `a < b ? a : b`, NOT the HLSL intrinsic (identical for the
// non-NaN inputs here, but the ternary is what the CPU reference transcribes).
float pxMin(float a, float b) { return a < b ? a : b; }
float pxMax(float a, float b) { return a > b ? a : b; }
float3 pxMin3(float3 a, float3 b) { return float3(pxMin(a.x, b.x), pxMin(a.y, b.y), pxMin(a.z, b.z)); }
float3 pxMax3(float3 a, float3 b) { return float3(pxMax(a.x, b.x), pxMax(a.y, b.y), pxMax(a.z, b.z)); }

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

// __ballot_sync(FULL_MASK, pred).  The leading barrier orders the zero-write after
// every read of the previous ballot's mask.
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
// The CUDA per-node body (FEMCloth.cu:393-474), unchanged: refit packed node
// `nodeIndex` from the current triangle positions (leaf lanes) and the already refit
// mPackedNodeBounds of its children (internal lanes).
// ---------------------------------------------------------------------------
void refitNode(uint64_t treeAddr, uint nodeIndex, uint idx, uint64_t meshVerts, uint64_t meshVertsIndices,
               uint64_t sPackedNodeBounds)
{
    const uint nodeData = bvData(treeAddr, nodeIndex, idx);
    const uint nbChildren = bvNbNodes(treeAddr, nodeIndex);
    const bool isLeafLane = (nodeData & 1u) != 0u;

    // compute the bitMask for all the leaf nodes
    const uint resultWarp = ballot32(idx < nbChildren && isLeafLane, idx);

    float3 mn = float3(PX_MAX_F32, PX_MAX_F32, PX_MAX_F32);
    float3 mx = float3(-PX_MAX_F32, -PX_MAX_F32, -PX_MAX_F32);

    // using one warp to compute all leaf nodes' (under the same current node) min and max.
    // CUDA walks the set bits with `for (k = resultWarp; k; k = clearLowestSetBit(k))`;
    // the fixed 32-iteration scan visits the same bits in the same ascending order (see
    // the header: llvmpipe miscompiles the `k &= k-1` idiom with this body).
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
// cloth_refitBoundLaunch (FEMCloth.cu:321).
// Dispatch: grid (nbActiveFemCloths, nbSlices >= 1, 1), block (32, 1, 1).
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
            refitNode(treeAddr, c, idx, meshVerts, meshVertsIndices, sPackedNodeBounds);

            if (c == 0u)
            {
                // The root is done, the whole tree is: update the broad-phase bound.
                // lane 0's mPackedNodeBounds[0] store must reach lanes 1..5.
                AllMemoryBarrierWithGroupSync();

                const uint index = Ptr<uint>(gP.elemIndex)[femClothId];

                // Expand the bound for broad phase.
                // Unlike soft bodies, the contact distance remains unchanged; otherwise,
                // contact buffers easily overflow for cloth.
                const float contactDist = clothes[femClothId].mOriginalContactOffset +
                                          Ptr<float>(gP.speculativeCCDContactOffset)[femClothId];
                const float broadPhaseExpansion = pxMax(gP.nbCollisionPairUpdatesPerTimestep, 1.0f) * contactDist;

                // PxBounds3& root = sPackedNodeBounds[0];  compute min(0-2) max(3-5)
                if (idx < 6u)
                {
                    const float r = boundsWordLoad(sPackedNodeBounds + uint64_t(idx) * 4UL);   // pairs with lane 0's atomic store
                    const float value = idx < 3u ? (r - broadPhaseExpansion) : (r + broadPhaseExpansion);
                    Ptr<float>(gP.boundArray + uint64_t(index) * BOUNDS3_STRIDE)[idx] = value;
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
