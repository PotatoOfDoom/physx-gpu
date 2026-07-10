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

// npMidphaseDeform.slang
//
// Slang port of the three deformable midphase traversers built on the BV32
// walk (bv32Traversal.cuh) and pushOntoStackFullWarp:
//
// - ClothTreeTraverser (femClothMidPhase.cu): fetch the leaf triangle from the
//   cloth's world-space vertices, triBoundingBox(v0,v1,v2).intersects(mBox),
//   push uint4(cmIdx, primitiveIndex, 0, 0).
// - SoftbodyTreeTraverser (softbodyMidPhase.cu): gate on the tet-mesh
//   surface hint (a PxU8 per tetrahedron; only nonzero-hint tets are tested),
//   fetch the tet, isValidTet (signed volume > 1e-9), tetBoundingBox
//   .intersects(mBox), push uint4(cmIdx, primitiveIndex, 0, 0).
// - SoftbodyBoxTraverser (softbodyMidPhase.cu): like the former but with NO
//   hint gate and payload uint4(cmIdx, queryIndex, primitiveIndex, 0).
//
// All box logic (componentwise min/max over uploaded vertices + the
// PxBounds3::intersects negated six-comparison form) selects/compares exact
// input values, so those decisions are bit-exact; isValidTet is the only
// arithmetic gate.  The intersectBoxFullWarp callback of all three is
// mBox.intersects(childBounds).  One 32-thread group per work item replaces
// the CUDA warp exactly as in npBv32.slang / npMidphaseCvx.slang; the PxU8
// hint array is represented as one uint per tetrahedron (the gate is
// `hint != 0` either way).

static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;

struct DmParams { uint numWorkItems; uint stackSize; uint pad0, pad1; }

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

struct DmWork
{
    float4 boxMin, boxMax; // mBox, world space
    uint4 ids;             // x = cmIdx, y = queryIndex
}

[[vk::push_constant]]
ConstantBuffer<DmParams> gParams;

RWStructuredBuffer<Bv32Node> gNodes;      // binding 0
RWStructuredBuffer<uint4> gPrimIndices;   // binding 1 (tri: xyz used; tet: xyzw)
RWStructuredBuffer<float4> gVerts;        // binding 2 (world space)
RWStructuredBuffer<uint> gSurfaceHint;    // binding 3 (PxU8 per tet, widened)
RWStructuredBuffer<DmWork> gWork;         // binding 4
RWStructuredBuffer<uint4> gOutStack;      // binding 5
RWStructuredBuffer<uint> gCounter;        // binding 6 (single element)

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
        gOutStack[stackIndex] = stackItemToPush;
}

// PxBounds3::intersects
bool boundsIntersect(float3 amin, float3 amax, float3 bmin, float3 bmax)
{
    return !(bmin.x > amax.x || amin.x > bmax.x || bmin.y > amax.y || amin.y > bmax.y ||
             bmin.z > amax.z || amin.z > bmax.z);
}

// isValidTet (cudaNpCommon.h): positive signed volume
bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// triBoundingBox / tetBoundingBox (deformableCollision.cuh)
void triBounds(float3 v0, float3 v1, float3 v2, out float3 mn, out float3 mx)
{
    mn = min(min(v0, v1), v2);
    mx = max(max(v0, v1), v2);
}
void tetBounds(float3 v0, float3 v1, float3 v2, float3 v3, out float3 mn, out float3 mx)
{
    mn = min(min(v0, v1), min(v2, v3));
    mx = max(max(v0, v1), max(v2, v3));
}

// ---------------------------------------------------------------------------
// Shared BV32 walk; MODE selects the primitive callback:
// 0 = cloth triangle, 1 = softbody tet (hint gate), 2 = softbody box tet.
void traverse<let MODE : int>(uint3 gtid, uint3 gid)
{
    const uint workId = gid.x;
    if (workId >= gParams.numWorkItems) return; // group-uniform
    const uint idxInWarp = gtid.x;
    const DmWork w = gWork[workId];
    const float3 bmn = w.boxMin.xyz;
    const float3 bmx = w.boxMax.xyz;

    uint currentNodeIndex = 0;
    bool hasNode = true;
    uint nbStack = 0;

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6;

        if (currentNodeIndex & 1)
        {
            const uint leafOffset = (currentNodeIndex >> 1) & 31;
            const uint data = gNodes[nodeIdx].mData[leafOffset];
            const uint nbPrimitives = (data >> 1) & 63;
            const uint primStartIndex = data >> 7;
            uint primitiveIndex = 0xffffffff;
            if (idxInWarp < nbPrimitives)
                primitiveIndex = idxInWarp + primStartIndex;

            bool intersect = false;
            if (primitiveIndex != 0xffffffff)
            {
                if (MODE == 0)
                {
                    const uint4 triIdx = gPrimIndices[primitiveIndex];
                    const float3 v0 = gVerts[triIdx.x].xyz;
                    const float3 v1 = gVerts[triIdx.y].xyz;
                    const float3 v2 = gVerts[triIdx.z].xyz;
                    float3 mn, mx;
                    triBounds(v0, v1, v2, mn, mx);
                    intersect = boundsIntersect(mn, mx, bmn, bmx);
                }
                else
                {
                    bool process = true;
                    if (MODE == 1)
                        process = gSurfaceHint[primitiveIndex] != 0;
                    if (process)
                    {
                        const uint4 tetIdx = gPrimIndices[primitiveIndex];
                        const float3 v0 = gVerts[tetIdx.x].xyz;
                        const float3 v1 = gVerts[tetIdx.y].xyz;
                        const float3 v2 = gVerts[tetIdx.z].xyz;
                        const float3 v3 = gVerts[tetIdx.w].xyz;
                        if (isValidTet(v0, v1, v2, v3))
                        {
                            float3 mn, mx;
                            tetBounds(v0, v1, v2, v3, mn, mx);
                            intersect = boundsIntersect(mn, mx, bmn, bmx);
                        }
                    }
                }
            }

            const uint4 item = (MODE == 2)
                ? uint4(w.ids.x, w.ids.y, primitiveIndex, 0)
                : uint4(w.ids.x, primitiveIndex, 0, 0);
            pushOntoStackFullWarp(intersect, idxInWarp, gParams.stackSize, item);
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
            // intersectBoxFullWarp: mBox.intersects(PxBounds3(min, max))
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
            const uint validCount = countbits(resultWarp);

            if (isValidNode)
                sBv32Nodes[nbStack + offset] = int(currentIndex);

            nbStack += validCount;
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

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void ctClothTri(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    traverse<0>(gtid, gid);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ctSoftbodyTet(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    traverse<1>(gtid, gid);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ctSoftbodyBoxTet(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    traverse<2>(gtid, gid);
}
