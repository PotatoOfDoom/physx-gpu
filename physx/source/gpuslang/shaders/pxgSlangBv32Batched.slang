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
//
// ===========================================================================
// pxgSlangBv32Batched -- the "one deformable BV32 tree vs ONE rigid AABB"
// midphase walk, spread over a 1024-thread workgroup, emitting EXACTLY the
// pairs, in EXACTLY the order, of the one-warp walk it replaces.
//
// Used by cloth_midphaseGeneratePairsLaunch (femClothContactGenBda) and
// sb_midphaseGeneratePairsLaunch (softBodyContactGenBda): bv32TreeTraversal
// (bv32Traversal.cuh) with ClothTreeTraverser / SoftbodyTreeTraverser.
//
// ---------------------------------------------------------------------------
// WHY (measured on an RX 9070 XT, SnippetDeformableSurface)
// ---------------------------------------------------------------------------
// The CUDA gives each contact manager ONE warp, and that warp pops the BV32
// stack one node at a time: one serial iteration per reached node AND per
// reached leaf.  The 200x200 snippet cloth (79,202 triangles) cooks to 1,057
// nodes over 3,969 leaves (~20 triangles each, depth 3), so the sphere the
// cloth drapes over reaches on the order of a thousand leaves and a few hundred
// nodes, and every leaf iteration is a dependent chain: leaf word -> triangle
// indices -> vertices -> atomicAdd on the pair counter -> next pop.  The port
// reproduced that with one 32-thread group whose ballot is emulated through
// groupshared (three barriers per node): one workgroup on one compute unit,
// 7-13 % of the frame at ~1 launch per step on the RX 9070 XT, i.e. 2-4 ms for
// that single launch.  The cost is the CUDA's own schedule (one warp, serial
// pops, one atomic per leaf); a CUDA GPU pays it too, with shorter latencies.

// ---------------------------------------------------------------------------
// THE ORDER, AND WHY IT CAN BE KEPT
// ---------------------------------------------------------------------------
// The one-warp walk pops a node; an internal node pushes its VALID children in
// ascending child slot (the ballot's exclusive prefix), so they pop in
// DESCENDING slot; a leaf emits its overlapping primitives in ascending lane
// order with one atomicAdd for the run.  So, writing Seq(x) for the leaves a
// stack entry x eventually pops, in pop order:
//
//     Seq(leaf)     = [leaf]
//     Seq(internal) = Seq(valid child 31..0, descending slot) concatenated
//     Seq(stack)    = Seq(top) ++ Seq(next) ++ ... ++ Seq(bottom)
//
// and the pair stream is the overlapping primitives of Seq(root) in that order.
// Which children are "valid" depends only on the node's boxes and the rigid
// box, never on the stack.  Two operations therefore leave
// `emitted ++ Seq(stack)` unchanged, and the walk below uses only these two:
//
//   DRAIN  -- move the run of LEAVES at the top of the stack, in pop order, to
//             the end of the emitted list (here: the leaf batch);
//   EXPAND -- replace the top W entries by their expansions, IN PLACE: entry w
//             (w = 0 is the top) becomes its valid children in the CUDA push
//             order (a leaf entry stays itself), and expansion w lands above
//             expansions w+1..W-1.
//
// The batch then evaluates its leaves one per thread and reserves their pairs
// with ONE atomicAdd, at the exclusive prefix of the per-leaf counts.  Within
// one contact manager that is exactly the slot sequence the per-leaf atomicAdd
// produced: same pairs, same slots, same final counter (M7 below).  ACROSS
// contact managers the CUDA's slots interleave in whatever order the warps'
// atomics land; reserving per batch instead of per leaf is one of those
// interleavings.  A group still walks its MIDPHASE_WARPS_PER_BLOCK managers one
// after the other, so a one-group launch (the snippets' one or two managers
// per deformable) is deterministic and unchanged slot for slot.
//
// ---------------------------------------------------------------------------
// THE SCHEDULE
// ---------------------------------------------------------------------------
//   round:  DRAIN up to (1024 - batch fill) leading leaves, one thread each;
//           batch full -> FLUSH;  stack empty -> done;
//           EXPAND the top W <= 32 entries, one 32-lane slot per entry: lane k
//           tests child k exactly as the CUDA lane k does.
//   FLUSH:  32 leaves per pass, one 32-lane slot each, lane k testing primitive
//           k exactly as CUDA lane k does (so the reads coalesce as in the CUDA);
//           then thread t owns leaf t: counts its hits; chunked exclusive scan
//           of the counts; ONE atomicAdd; scatter in lane order.
//
// For the snippet cloth vs the sphere this is ~10-20 expand rounds (root, the
// 32 mid nodes, the reached leaf-only nodes 32 at a time) and one or two
// flushes of at most 32 passes each, instead of ~1,000-2,500 serial pops.
//
// ---------------------------------------------------------------------------
// THE STACK BOUND
// ---------------------------------------------------------------------------
// The CUDA keeps 192 pending entries (femMidphaseScratch.sBv32Nodes) and writes
// past them unchecked; a tree that needs more is undefined behaviour there.
// Expanding 32 entries at once can grow the stack by up to 31 x 32 in one
// round, so this walk keeps BV32B_STACK = 2048 entries and only widens the
// window while the stack after the round stays at or below
// BV32B_STACK - BV32B_RESERVE, BV32B_RESERVE = 192.  At or above that line it
// expands ONE entry per round (W = 1, the CUDA's own step), whose growth until
// the stack falls back is bounded by what the CUDA's 192 entries had to hold.
// A tree that still overruns 2048 is one the CUDA overran first; the writes
// past the end are dropped (never out of bounds) instead.
//
// M7 (unchanged): the pair COUNTER is not capped, the WRITE is.  Slots at or
// past stackSize are reserved but not written; the overshoot is what sizes the
// next frame's stack (stackSizeNeededOnDevice).
//
// ---------------------------------------------------------------------------
// RULES FOR CALLERS
// ---------------------------------------------------------------------------
//   - [numthreads(BV32B_THREADS, 1, 1)], and the call is group-UNIFORM (every
//     barrier here is reached by the whole group);
//   - IBv32BatchedLeaf::overlaps(prim) gets only a real primitive of a reached
//     leaf (the speculation-safe address IS the primitive index here, see the
//     P9.28 note in the callers' history); it must return what the CUDA
//     traverser's intersectPrimitiveFullWarp computes for that lane;
//   - the BV32 accessors and index rules are pxgSlangBv32's.
// ===========================================================================

module pxgSlangBv32Batched;

import pxgSlangBv32;

public static const uint BV32B_THREADS = 1024u;   // the workgroup every caller declares
static const uint BV32B_SLOTS = 32u;              // 32-lane slots: entries expanded per round
static const uint BV32B_STACK = 2048u;            // pending entries (CUDA: 192)
static const uint BV32B_RESERVE = 192u;           // femMidphaseScratch.sBv32Nodes[192]
static const uint BV32B_BATCH = 1024u;            // leaves per flush, one per thread

groupshared uint sB32Stack[BV32B_STACK];   // pending entries, top = [nbStack - 1]
groupshared uint sB32Leaf[BV32B_BATCH];    // the leaf batch, then its count scan
groupshared uint sB32Mask[BV32B_BATCH];    // the batch's per-leaf hit masks
groupshared uint sB32SlotMask[BV32B_SLOTS];  // __ballot_sync of each expanded entry
groupshared uint sB32SlotCount[BV32B_SLOTS]; // entries each expanded entry becomes
groupshared uint sB32ChunkBase[32];        // scan of 32 chunks of 32 leaf counts
groupshared uint sB32First;                // first non-leaf from the top (DRAIN)
groupshared uint sB32Start;                // the batch's atomicAdd result

// The per-primitive half of a CUDA tree traverser (intersectPrimitiveFullWarp).
public interface IBv32BatchedLeaf
{
    bool overlaps(uint primitiveIndex);
}

// PxBounds3::intersects, the negated six-comparison form of PxBounds3.h.
bool bv32bIntersects(float3 aMin, float3 aMax, float3 bMin, float3 bMax)
{
    return !(bMin.x > aMax.x || aMin.x > bMax.x ||
             bMin.y > aMax.y || aMin.y > bMax.y ||
             bMin.z > aMax.z || aMin.z > bMax.z);
}

// FLUSH: the batch's nbLeaves leaves (sB32Leaf[0..nbLeaves)) in emission order.
void bv32bFlush<L : IBv32BatchedLeaf>(L leaf, uint tid, uint nbLeaves, uint64_t treeAddr,
                                       uint cmIdx, uint64_t pairStack, uint64_t pairCounter,
                                       uint stackSize)
{
    sB32Mask[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();   // the drained entries and the cleared masks are visible

    // TEST: the CUDA's own lane mapping, 32 leaves per pass -- slot (tid >> 5) is a
    // leaf, lane (tid & 31) is its primitive, so a pass issues 1024 independent,
    // coalesced primitive tests.  `lane < nbPrimitives` against the RAW 6-bit count
    // is the CUDA's guard; for lanes 0..31 it clamps a count above 32 bug-for-bug
    // (pxgSlangBv32 header).  The pass loop is group-uniform and has no barrier.
    const uint lane = tid & 31u;
    for (uint first = 0u; first < nbLeaves; first += BV32B_SLOTS)
    {
        const uint j = first + (tid >> 5u);
        // A clamped entry for the slots past the batch: every entry names a real
        // node, so the speculated reads below stay inside the tree (P9.22).
        const uint entry = sB32Leaf[min(j, nbLeaves - 1u)];
        // The leaf's primitive range lives in the PARENT node's mData slot.
        const uint data = bvData(treeAddr, bvNodeIdxOf(entry), bvLeafSlotOf(entry));
        const uint nbPrimitives = bvDataNbPrimitives(data);
        const uint primStartIndex = bvDataPrimStart(data);
        bool hit = false;
        if (j < nbLeaves && lane < nbPrimitives)
        {
            // The clamp never changes a live lane's index; it bounds the address the
            // masked-off SIMD lanes compute (P9.28).
            const uint prim = primStartIndex + min(lane, max(nbPrimitives, 1u) - 1u);
            hit = leaf.overlaps(prim);
        }
        if (hit)
            InterlockedOr(sB32Mask[j], 1u << lane);
    }
    GroupMemoryBarrierWithGroupSync();   // every mask complete, every entry read

    // From here on thread t owns leaf t: its hit mask, then its slot run.
    uint mask = 0u;
    uint primStart = 0u;
    {
        const uint entry = sB32Leaf[min(tid, nbLeaves - 1u)];
        primStart = bvDataPrimStart(bvData(treeAddr, bvNodeIdxOf(entry), bvLeafSlotOf(entry)));
        if (tid < nbLeaves)
            mask = sB32Mask[tid];
    }
    GroupMemoryBarrierWithGroupSync();   // the batch array becomes the scan

    sB32Leaf[tid] = countbits(mask);
    GroupMemoryBarrierWithGroupSync();
    if (tid < 32u)
    {
        // min(): the masked-off lanes' speculated groupshared reads stay in the array.
        const uint chunk = min(tid, 31u) * 32u;
        uint run = 0u;
        for (uint j = 0u; j < 32u; ++j)
        {
            const uint c = sB32Leaf[chunk + j];
            sB32Leaf[chunk + j] = run;
            run += c;
        }
        sB32ChunkBase[tid] = run;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        uint run = 0u;
        for (uint c = 0u; c < 32u; ++c)
        {
            const uint t = sB32ChunkBase[c];
            sB32ChunkBase[c] = run;
            run += t;
        }
        // pushOntoStackFullWarp (bv32Traversal.cuh): reserve, and only when there is
        // something to reserve.  The counter is NOT capped (M7).
        uint start = 0u;
        if (run > 0u)
            InterlockedAdd(Ptr<uint>(pairCounter)[0], run, start);
        sB32Start = start;
    }
    GroupMemoryBarrierWithGroupSync();

    uint slot = sB32Start + sB32ChunkBase[tid >> 5u] + sB32Leaf[tid];
    while (mask != 0u)
    {
        const uint k = firstbitlow(mask);
        mask &= mask - 1u;
        if (slot < stackSize)   // M7: drop the write, keep the reservation
            Ptr<uint4>(pairStack)[slot] = uint4(cmIdx, primStart + k, 0u, 0u);
        ++slot;
    }
    GroupMemoryBarrierWithGroupSync();   // before the next batch reuses the scratch
}

// The whole walk for one contact manager: the BV32 tree at treeAddr (past the
// uint4 header) against the fattened rigid box [boxMin, boxMax].
public void bv32BatchedMidphase<L : IBv32BatchedLeaf>(L leaf, uint tid, uint64_t treeAddr,
                                                      float3 boxMin, float3 boxMax, uint cmIdx,
                                                      uint64_t pairStack, uint64_t pairCounter,
                                                      uint stackSize)
{
    GroupMemoryBarrierWithGroupSync();   // the scratch may still be read by a previous walk
    if (tid == 0u)
        sB32Stack[0] = 0u;               // the root: node 0, internal
    uint nbStack = 1u;
    uint nbLeaves = 0u;
    const uint slotIdx = tid >> 5u;      // 0..31
    const uint lane = tid & 31u;

    for (;;)
    {
        // ---- DRAIN: the leaves at the top, in pop order, into the batch ----
        const uint limit = min(nbStack, BV32B_BATCH - nbLeaves);
        if (tid == 0u)
            sB32First = limit;
        GroupMemoryBarrierWithGroupSync();
        // The index is clamped for the threads past `limit`: llvmpipe issues a guarded
        // GROUPSHARED load for the masked-off lanes too, exactly like the BDA reads of
        // P9.22, and nbStack - 1 - tid wraps to four billion there.  (limit >= 1
        // whenever nbStack > 0: the batch is never full at this point.)
        const uint topAt = (nbStack > 0u) ? nbStack - 1u - min(tid, limit - 1u) : 0u;
        uint top = 0u;
        if (tid < limit)
        {
            top = sB32Stack[topAt];
            if (!bvIsLeaf(top))
                InterlockedMin(sB32First, tid);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint nbDrained = sB32First;
        if (tid < nbDrained)
            sB32Leaf[nbLeaves + tid] = top;
        nbLeaves += nbDrained;
        nbStack -= nbDrained;

        if (nbLeaves == BV32B_BATCH)
        {
            bv32bFlush(leaf, tid, nbLeaves, treeAddr, cmIdx, pairStack, pairCounter, stackSize);
            nbLeaves = 0u;
            continue;
        }
        if (nbStack == 0u)
            break;

        // ---- EXPAND: the top W entries, one 32-lane slot each ----
        // W > 1 only while the stack after the round stays within
        // BV32B_STACK - BV32B_RESERVE (worst case: every entry has 32 valid children).
        const uint room = (BV32B_STACK - BV32B_RESERVE > nbStack)
                              ? (BV32B_STACK - BV32B_RESERVE - nbStack) / (BV32B_SLOTS - 1u)
                              : 0u;
        const uint W = max(1u, min(min(nbStack, BV32B_SLOTS), room));
        const bool inWindow = slotIdx < W;
        const uint entry = sB32Stack[nbStack - 1u - min(slotIdx, W - 1u)];
        if (tid < BV32B_SLOTS)
            sB32SlotMask[tid] = 0u;
        GroupMemoryBarrierWithGroupSync();   // window read: it may now be overwritten

        // The CUDA internal-node step for node `entry`, lane `lane`, verbatim.
        // `entry` always names a real node (a leaf entry names its parent), so the
        // reads below never leave the tree whatever the guards compile to.
        const bool isInternal = inWindow && !bvIsLeaf(entry);
        const uint nodeIdx = bvNodeIdxOf(entry);
        uint nbChildren = 0u;
        if (isInternal)
            nbChildren = bvNbNodes(treeAddr, nodeIdx);
        float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
        const bool hasBox = lane < nbChildren;
        if (hasBox)
        {
            mn = bvMin(treeAddr, nodeIdx, lane);
            mx = bvMax(treeAddr, nodeIdx, lane);
        }
        // intersectBoxFullWarp: false when the lane has no box.
        bool intersect = false;
        if (hasBox)
            intersect = bv32bIntersects(boxMin, boxMax, mn, mx);

        uint childIndex = bvChildIndex(entry, lane);
        bool isValidNode = false;
        if (lane < nbChildren)
        {
            const uint data = bvData(treeAddr, nodeIdx, lane);
            isValidNode = intersect;
            if (isValidNode && bvDataIsInternal(data))
                childIndex = bvDataChildNodeIndex(data);
        }
        if (isValidNode)
            InterlockedOr(sB32SlotMask[slotIdx], 1u << lane);
        GroupMemoryBarrierWithGroupSync();

        const uint mask = sB32SlotMask[slotIdx];
        if (inWindow && lane == 0u)
            sB32SlotCount[slotIdx] = isInternal ? countbits(mask) : 1u;
        GroupMemoryBarrierWithGroupSync();

        // Expansion w sits above expansions w+1..W-1 (they pop after it).
        uint below = 0u, total = 0u;
        for (uint v = 0u; v < W; ++v)
        {
            const uint c = sB32SlotCount[v];
            total += c;
            if (v > slotIdx)
                below += c;
        }
        const uint base = nbStack - W;
        if (isValidNode)
        {
            // CUDA: sBv32Nodes[nbStack + offset], offset = the ballot's exclusive prefix.
            const uint pos = base + below + countbits(mask & ((1u << lane) - 1u));
            if (pos < BV32B_STACK)
                sB32Stack[pos] = childIndex;
        }
        else if (inWindow && !isInternal && lane == 0u)
        {
            const uint pos = base + below;   // a leaf entry stays where its slot lands
            if (pos < BV32B_STACK)
                sB32Stack[pos] = entry;
        }
        nbStack = min(base + total, BV32B_STACK);
        GroupMemoryBarrierWithGroupSync();   // the new top is visible to DRAIN
    }

    if (nbLeaves > 0u)
        bv32bFlush(leaf, tid, nbLeaves, treeAddr, cmIdx, pairStack, pairCounter, stackSize);
}
