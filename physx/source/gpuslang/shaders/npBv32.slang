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

// npBv32.slang
//
// Slang port of the BV32 midphase tree traversal from
// gpunarrowphase/src/CUDA/bv32Traversal.cuh: bv32TreeTraversal (the
// warp-cooperative walk over Gu::BV32DataPacked nodes — packed node index
// nodeIdx<<6, leaf encoding (currentNodeIndex & ~63) | (childLane<<1) | 1 with
// the leaf's primitive range decoded from the PARENT node's mData slot, child
// offset mData>>11 for internal children, per-lane child-AABB tests reduced by
// warp ballot + popcount-prefix into a groupshared LIFO stack) and
// pushOntoStackFullWarp (the warp-aggregated output append: one atomicAdd by
// lane 0 over the ballot count, broadcast, per-lane store dropped past the
// stack capacity — the counter still advances for dropped items).
//
// The traversal callback here is the canonical AABB-overlap query: an internal
// child survives if its box overlaps the query box (pure comparisons on
// uploaded values — bit-exact), and every primitive of a visited leaf is
// appended as uint4(queryId, primitiveIndex, 0, 0).  One 32-thread group per
// query replaces the CUDA warp; __ballot_sync becomes a groupshared OR mask
// (order-independent), warpScanExclusive stays popc(mask & lanemask_lt) and
// __shfl_sync(...,0) becomes a groupshared broadcast, all output-identical.

static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;

struct Bv32Params { uint numQueries; uint stackSize; uint pad0, pad1; }

// Gu::BV32DataPacked (1168 bytes)
struct Bv32Node
{
    float4 mMin[32];
    float4 mMax[32];
    uint mData[32];
    uint mNbNodes;
    uint mDepth;
    uint pad0, pad1;
}

struct QueryBox { float4 mn; float4 mx; }

[[vk::push_constant]]
ConstantBuffer<Bv32Params> gParams;

RWStructuredBuffer<Bv32Node> gNodes;    // binding 0
RWStructuredBuffer<QueryBox> gQueries;  // binding 1
RWStructuredBuffer<uint4> gOutStack;    // binding 2
RWStructuredBuffer<uint> gCounter;      // binding 3 (single element)

groupshared int sBv32Nodes[1024]; // to-be-expanded node stack (CUDA: caller-provided shared memory)
groupshared uint sMask;
groupshared uint sStart;

// __ballot_sync over the 32-thread group.  The leading barrier orders the
// zero-write after every read of the previous ballot's mask.
uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// pushOntoStackFullWarp with the intersectPrimitiveFullWarp callback contract:
// threads with primitiveIndex == 0xffffffff push nothing.
void pushPrimitive(uint queryId, uint primitiveIndex, uint idxInWarp, uint stackSize)
{
    const bool threadPushesElement = primitiveIndex != 0xffffffff;
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
    // If we run out of stacksize, drop contacts
    if (threadPushesElement && (stackIndex < stackSize))
        gOutStack[stackIndex] = uint4(queryId, primitiveIndex, 0, 0);
}

// ---------------------------------------------------------------------------
// One 32-thread group per query box.
[shader("compute")]
[numthreads(32, 1, 1)]
void ctBv32Query(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint queryId = gid.x;
    if (queryId >= gParams.numQueries) return; // group-uniform
    const uint idxInWarp = gtid.x;
    const float3 qmn = gQueries[queryId].mn.xyz;
    const float3 qmx = gQueries[queryId].mx.xyz;

    uint currentNodeIndex = 0; // start at the root
    bool hasNode = true;
    uint nbStack = 0; // number of nodes on the stack of to-be-expanded nodes

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6;

        if (currentNodeIndex & 1)
        {
            // leaf: the primitive range lives in the PARENT node's mData slot
            const uint leafOffset = (currentNodeIndex >> 1) & 31;
            const uint data = gNodes[nodeIdx].mData[leafOffset];
            const uint nbPrimitives = (data >> 1) & 63;   // getNbReferencedPrimitives
            const uint primStartIndex = data >> 7;        // getPrimitiveStartIndex
            uint primitiveIdx = 0xffffffff;
            if (idxInWarp < nbPrimitives)
                primitiveIdx = idxInWarp + primStartIndex;
            pushPrimitive(queryId, primitiveIdx, idxInWarp, gParams.stackSize);
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
            // intersectBoxFullWarp: AABB-overlap query callback
            const bool intersect = hasBox && all(qmn <= mx) && all(mn <= qmx);

            // masking currentNodeIndex with 0xFFFFFFC0 clears the 6 lowest bits,
            // so we can mask in our idxInWarp<<1 | 1 for leaf masks
            uint currentIndex = (currentNodeIndex & 0xFFFFFFC0) | (idxInWarp << 1) | 1;

            bool isValidNode = false;
            if (idxInWarp < nbChildren)
            {
                const uint data = gNodes[nodeIdx].mData[idxInWarp];
                isValidNode = intersect;
                if (isValidNode && (data & 1) == 0) // !isLeaf
                    currentIndex = (data >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) << 6;
            }

            const uint resultWarp = ballot32(isValidNode, idxInWarp);
            const uint offset = countbits(resultWarp & ((1u << idxInWarp) - 1u));
            const uint validCount = countbits(resultWarp);

            if (isValidNode)
                sBv32Nodes[nbStack + offset] = int(currentIndex);

            nbStack += validCount;
        }

        GroupMemoryBarrierWithGroupSync();

        // select the next node from the stack
        if (nbStack > 0)
        {
            nbStack -= 1;
            currentNodeIndex = uint(sBv32Nodes[nbStack]);
        }
        else
            hasNode = false;
    }
}
