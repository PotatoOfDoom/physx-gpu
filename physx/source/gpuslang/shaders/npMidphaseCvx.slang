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

// npMidphaseCvx.slang
//
// Slang port of the convex-vs-trianglemesh midphase traverser from
// gpunarrowphase/src/CUDA/convexMeshMidphase.cu (MidphaseTreeTraverser),
// composed onto the BV32 tree walk ported in npBv32.slang:
//
// - intersectBoxFullWarp: AABB overlap of the query bounds (trimesh vertex
//   space) against each BV32 child box — the CUDA's negated six-comparison
//   form is kept verbatim.
// - intersectPrimitiveFullWarp: fetch the leaf triangle (uint4 index triple +
//   vertex fetch), back-face cull against the convex shape's center of mass
//   (PxPlane(v0,v1,v2).distance(com) >= 0), transform the triangle into world
//   space (vertex2Shape mesh scaling, then the mesh-to-world transform) and
//   run Gu::intersectTriangleBox_RefImpl<true> (the Akenine-Moller
//   triangle-vs-AABB SAT, vertex checks enabled) against the convex's
//   inflated world AABB.  Survivors are ballot-compacted into the global pair
//   stack as uint4(cmIdx, triangleIndex, nbPairsPerCM + offset,
//   shapeMeshIndex), with the maxPairs overflow rule: the atomic counter
//   always advances, out-of-capacity stores are dropped, and the per-CM
//   running count adds the overflow-corrected validCount.
//
// One 32-thread group per work item replaces the CUDA warp (ballot ->
// groupshared OR mask, shuffle-broadcast -> groupshared write), identical to
// npBv32.slang.

import pxgSlangCommon;
import pxgSlangTriBox;   // P8.24: the SAT, extracted so the BDA midphase shares it

static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;

struct MpParams { uint numWorkItems; uint maxPairs; uint pad0, pad1; }

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

struct MpWork
{
    float4 pMin, pMax;               // query AABB, trimesh vertex space
    float4 shapeCoM;                 // convex CoM, trimesh vertex space
    float4 center, inflatedExtents;  // convex AABB, world space
    float4 meshToWorldQ, meshToWorldP;
    float4 scale, scaleRotQ;         // PxMeshScale
    uint4 ids;                       // x = cmIdx, y = shapeMeshIndex
}

[[vk::push_constant]]
ConstantBuffer<MpParams> gParams;

RWStructuredBuffer<Bv32Node> gNodes;       // binding 0
RWStructuredBuffer<uint4> gTriIndices;     // binding 1
RWStructuredBuffer<float4> gVerts;         // binding 2 (vertex space)
RWStructuredBuffer<MpWork> gWork;          // binding 3
RWStructuredBuffer<uint4> gOutPairs;       // binding 4
RWStructuredBuffer<uint> gNbPairs;         // binding 5 (single element)
RWStructuredBuffer<uint> gNbPairsPerCM;    // binding 6 (per work item)

groupshared int sBv32Nodes[1024];
groupshared uint sMask;
groupshared uint sBroadcast;

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// vertex2Shape (cudaNpCommon.h): scale in a rotated frame.
float3 vertex2Shape(float3 v, float3 scale, float4 rotation)
{
    float3 temp = quatRotate(rotation, v);
    temp *= scale;
    return quatRotateInv(rotation, temp);
}

// PxVec3::getNormalized (zero for degenerate input).
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// ---------------------------------------------------------------------------
// One 32-thread group per work item (convex-vs-mesh pair).
[shader("compute")]
[numthreads(32, 1, 1)]
void ctMidphaseCvx(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint workId = gid.x;
    if (workId >= gParams.numWorkItems) return; // group-uniform
    const uint idxInWarp = gtid.x;
    const MpWork w = gWork[workId];
    const uint maxPairs = gParams.maxPairs;

    uint currentNodeIndex = 0;
    bool hasNode = true;
    uint nbStack = 0;
    uint nbPairsPerCM = 0;

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6;

        if (currentNodeIndex & 1)
        {
            // ---- MidphaseTreeTraverser::intersectPrimitiveFullWarp ----
            const uint leafOffset = (currentNodeIndex >> 1) & 31;
            const uint data = gNodes[nodeIdx].mData[leafOffset];
            const uint nbPrimitives = (data >> 1) & 63;
            const uint primStartIndex = data >> 7;
            uint primitiveIndex = 0xffffffff;
            if (idxInWarp < nbPrimitives)
                primitiveIndex = idxInWarp + primStartIndex;

            bool primitiveIntersects = false;
            if (primitiveIndex != 0xffffffff)
            {
                const uint4 triIdx = gTriIndices[primitiveIndex];
                const float3 triV0 = gVerts[triIdx.x].xyz;
                const float3 triV1 = gVerts[triIdx.y].xyz;
                const float3 triV2 = gVerts[triIdx.z].xyz;

                // PxPlane(v0, v1, v2)
                const float3 n = getNormalized(cross(triV1 - triV0, triV2 - triV0));
                const float pd = -dot(triV0, n);

                if (dot(n, w.shapeCoM.xyz) + pd >= 0.0f)
                {
                    const float3 worldSpaceP0 = quatRotate(w.meshToWorldQ, vertex2Shape(triV0, w.scale.xyz, w.scaleRotQ)) + w.meshToWorldP.xyz;
                    const float3 worldSpaceP1 = quatRotate(w.meshToWorldQ, vertex2Shape(triV1, w.scale.xyz, w.scaleRotQ)) + w.meshToWorldP.xyz;
                    const float3 worldSpaceP2 = quatRotate(w.meshToWorldQ, vertex2Shape(triV2, w.scale.xyz, w.scaleRotQ)) + w.meshToWorldP.xyz;

                    primitiveIntersects = tbIntersectTriangleBox(w.center.xyz, w.inflatedExtents.xyz,
                                                               worldSpaceP0, worldSpaceP1, worldSpaceP2);
                }
            }

            const uint resultWarp = ballot32(primitiveIntersects, idxInWarp);
            const uint offset = countbits(resultWarp & ((1u << idxInWarp) - 1u));
            uint validCount = countbits(resultWarp);

            // Allocate only the amount needed for a single warp-wide write.
            if (idxInWarp == 0)
            {
                uint prevNbPairs = 0xffffffffu;
                if (validCount > 0)
                    InterlockedAdd(gNbPairs[0], validCount, prevNbPairs);
                sBroadcast = prevNbPairs;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint prevNbPairs = sBroadcast;

            // only write the pairs that still fit within the max pairs limit
            if (primitiveIntersects && ((prevNbPairs + offset) < maxPairs))
                gOutPairs[prevNbPairs + offset] = uint4(w.ids.x, primitiveIndex, nbPairsPerCM + offset, w.ids.y);

            // correct validCount: subtract the overflowing pairs
            if ((validCount > 0) && ((validCount + prevNbPairs) >= maxPairs))
                validCount = max(maxPairs, prevNbPairs) - prevNbPairs;

            nbPairsPerCM += validCount;
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
            // MidphaseTreeTraverser::intersectBoxFullWarp
            const bool intersect = hasBox &&
                !(mn.x > w.pMax.x || w.pMin.x > mx.x || mn.y > w.pMax.y || w.pMin.y > mx.y || mn.z > w.pMax.z || w.pMin.z > mx.z);

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

    if (idxInWarp == 0)
        gNbPairsPerCM[workId] = nbPairsPerCM;
}
