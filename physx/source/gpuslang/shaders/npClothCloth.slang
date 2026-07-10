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

// npClothCloth.slang
//
// Slang port of the cloth-vs-mesh / cloth-vs-cloth midphase traverser from
// gpunarrowphase/src/CUDA/femClothClothMidPhase.cu:
//
// - computeTriangleBound2: the query triangle's world AABB (triBoundingBox of
//   its three fetched vertices), fattened by contactDistance (fattenFast) as
//   both in-engine callers do before the walk — the box the BV32 walk tests.
// - TriangleLeafBoundMinMaxTraverser::intersectPrimitiveFullWarp: each lane
//   fetches its leaf triangle, builds its PxPlane, and — on the MESH path —
//   back-face culls against the query triangle's centroid
//   (triPlane.distance(centroid) >= 0), or — on the CLOTH-CLOTH path —
//   drops the pair if either PxgNonRigidFilterPair lookup matches (the
//   (fullMask0, mask1) "any triangle of cloth0" filter or the exact
//   (mask0, mask1) filter, both min/max-ordered, where
//   PxEncodeClothIndex(clothId, elementId) = (clothId << 20) | elementId and
//   the full mask uses PX_MAX_NB_DEFORMABLE_SURFACE_TRI = 0xfffff).
//   Surviving triangles run satIntersect (the triangle-triangle SAT from
//   triangletriangle.cuh, tranche 57) with the EVOLVING tolerance
//   min(mBestDistance, contactDistance); a separation below the running best
//   updates the per-lane (mBestDistance, mPrimIndex) — the warp-wide argmin
//   reduction over lanes happens in the caller and is out of scope here.
// - intersectBoxFullWarp: the query triangle bound vs each BV32 child box.
//
// The per-lane best/index state is written out per (workItem, lane).
// NOTE: TriangleSelfCollisionLeafBoundMinMaxTraverser is NOT ported — its
// primitive callback gates on `primitiveIndex == 0xFFFFFFFF` (upstream
// anomaly), which both suppresses every valid lane and indexes the mesh
// arrays with 0xFFFFFFFF on invalid ones.

static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;
static const uint PXG_BITSHIFT_ELEMENT_ID = 20;
static const uint PX_MAX_NB_DEFORMABLE_SURFACE_TRI = 0x000fffff;
static const float PX_MAX_F32 = 3.4028235e38f;

struct CcParams { uint numWorkItems; uint numFilterPairs; uint pad0, pad1; }

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

struct PairNR { uint2 index0; uint index1; uint index2; } // PxgNonRigidFilterPair

struct CcWork
{
    uint4 ids;           // x = triangleIndex (mesh0), y = isMesh, z = id0, w = id1
    float4 centroid_cd;  // xyz = centroid (mesh path), w = contactDistance
}

[[vk::push_constant]]
ConstantBuffer<CcParams> gParams;

RWStructuredBuffer<Bv32Node> gNodes;      // binding 0 (tree over mesh1)
RWStructuredBuffer<uint4> gTris1;         // binding 1
RWStructuredBuffer<float4> gVerts1;       // binding 2
RWStructuredBuffer<uint4> gTris0;         // binding 3 (query cloth)
RWStructuredBuffer<float4> gVerts0;       // binding 4
RWStructuredBuffer<PairNR> gFilterPairs;  // binding 5 (sorted)
RWStructuredBuffer<CcWork> gWork;         // binding 6
RWStructuredBuffer<float> gOutBest;       // binding 7 (workId*32 + lane)
RWStructuredBuffer<uint> gOutPrim;        // binding 8 (workId*32 + lane)

groupshared int sBv32Nodes[1024];
groupshared uint sMask;

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// --- PxgNonRigidFilterPair binary search (deformableElementFilter.cuh, tranche 75) ---
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

int compareNR(uint2 a0, uint a1, uint a2, PairNR p)
{
    if (u64Lt(a0, p.index0)) return -1;
    if (u64Lt(p.index0, a0)) return 1;
    if (a1 < p.index1) return -1;
    if (a1 > p.index1) return 1;
    if (a2 < p.index2) return -1;
    if (a2 > p.index2) return 1;
    return 0;
}

// the 2-arg wrapper: index2 = 0, ignoreIndex2 = true
bool findFilter(uint nbPairs, uint id0, uint id1)
{
    const uint2 q0 = uint2(id0, 0);
    uint l = 0, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const PairNR p = gFilterPairs[mid];
        const int cmp = compareNR(q0, id1, 0u, p);
        if (cmp == 0 || (u64Eq(q0, p.index0) && id1 == p.index1))
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

uint pxEncodeClothIndex(uint clothId, uint elementId)
{
    return (clothId << PXG_BITSHIFT_ELEMENT_ID) | elementId;
}

// --- triangle helpers -------------------------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// PxPlane(p0, p1, p2): n = normalize(cross(p1-p0, p2-p0)), d = -p0.n
void makePlane(float3 p0, float3 p1, float3 p2, out float3 n, out float d)
{
    n = getNormalized(cross(p1 - p0, p2 - p0));
    d = -dot(p0, n);
}

// minProject(plane, triangle) (triangletriangle.cuh)
float minProject(float3 n, float d, float3 v0, float3 v1, float3 v2)
{
    return min(dot(n, v0), min(dot(n, v1), dot(n, v2))) + d;
}

float testSeparatingAxis(float3 axis, float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2)
{
    float min0, max0, min1, max1;
    min0 = max0 = dot(a0, axis);
    min1 = max1 = dot(b0, axis);
    const float pa1 = dot(a1, axis);
    const float pb1 = dot(b1, axis);
    min0 = min(pa1, min0); max0 = max(pa1, max0);
    min1 = min(pb1, min1); max1 = max(pb1, max1);
    const float pa2 = dot(a2, axis);
    const float pb2 = dot(b2, axis);
    min0 = min(pa2, min0); max0 = max(pa2, max0);
    min1 = min(pb2, min1); max1 = max(pb2, max1);
    return max(min1 - max0, min0 - max1);
}

// satIntersect<true> (triangletriangle.cuh): the two face-normal axes with
// early-outs, then the 9 edge-cross axes with the magSq validity gate.
float satIntersect(float3 a0, float3 a1, float3 a2, float3 nA, float dA,
                   float3 b0, float3 b1, float3 b2, float3 nB, float dB,
                   float tolerance)
{
    float sep = minProject(nA, dA, b0, b1, b2);
    if (sep > tolerance) return sep;

    sep = max(sep, minProject(nB, dB, a0, a1, a2));
    if (sep > tolerance) return sep;

    const float3 axes0[3] = { a1 - a0, a2 - a0, a2 - a1 };
    const float3 axes1[3] = { b1 - b0, b2 - b0, b2 - b1 };

    for (uint i = 0u; i < 3u; ++i)
    {
        const float3 axis0 = axes0[i];
        for (uint j = 0u; j < 3u; ++j)
        {
            const float3 axis1 = axes1[j];
            float3 sepAxis = cross(axis0, axis1);
            const float magSq = dot(sepAxis, sepAxis);
            if (magSq > 1e-5f)
            {
                sepAxis = sepAxis * (1.0f / sqrt(magSq));
                const float tSep = testSeparatingAxis(sepAxis, a0, a1, a2, b0, b1, b2);
                sep = max(sep, tSep);
                if (sep > tolerance) return sep;
            }
        }
    }
    return sep;
}

// ---------------------------------------------------------------------------
// One 32-thread group per work item (one query triangle of cloth/mesh 0).
[shader("compute")]
[numthreads(32, 1, 1)]
void ctClothClothQuery(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint workId = gid.x;
    if (workId >= gParams.numWorkItems) return; // group-uniform
    const uint idxInWarp = gtid.x;
    const CcWork w = gWork[workId];
    const uint triangleIndex = w.ids.x;
    const bool isMesh = w.ids.y != 0;
    const uint id0 = w.ids.z;
    const uint id1 = w.ids.w;
    const float contactDistance = w.centroid_cd.w;
    const float3 centroid = w.centroid_cd.xyz;

    // query triangle (mesh 0) + its plane and bound (computeTriangleBound2),
    // fattened by contactDistance (fattenFast) exactly as both in-engine
    // callers do before constructing the traverser
    const uint4 ti0 = gTris0[triangleIndex];
    const float3 a0 = gVerts0[ti0.x].xyz;
    const float3 a1 = gVerts0[ti0.y].xyz;
    const float3 a2 = gVerts0[ti0.z].xyz;
    float3 nA; float dA;
    makePlane(a0, a1, a2, nA, dA);
    const float3 tMin = min(min(a0, a1), a2) - contactDistance;
    const float3 tMax = max(max(a0, a1), a2) + contactDistance;

    // per-lane traverser state
    float mBestDistance = PX_MAX_F32;
    uint mPrimIndex = 0xffffffff;

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

            if (primitiveIndex != 0xffffffff)
            {
                const uint4 ti1 = gTris1[primitiveIndex];
                const float3 b0 = gVerts1[ti1.x].xyz;
                const float3 b1 = gVerts1[ti1.y].xyz;
                const float3 b2 = gVerts1[ti1.z].xyz;
                float3 nB; float dB;
                makePlane(b0, b1, b2, nB, dB);

                if (isMesh)
                {
                    // get rid of double-sided triangle
                    if (dot(nB, centroid) + dB >= 0.0f)
                    {
                        const float separation = satIntersect(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB,
                                                              min(mBestDistance, contactDistance));
                        if (separation < mBestDistance)
                        {
                            mBestDistance = separation;
                            mPrimIndex = primitiveIndex;
                        }
                    }
                }
                else
                {
                    const uint clothMask0 = pxEncodeClothIndex(id0, triangleIndex);
                    const uint clothMask1 = pxEncodeClothIndex(id1, primitiveIndex);
                    const uint clothFullMask0 = pxEncodeClothIndex(id0, PX_MAX_NB_DEFORMABLE_SURFACE_TRI);
                    if (!findFilter(gParams.numFilterPairs, min(clothFullMask0, clothMask1), max(clothFullMask0, clothMask1)) &&
                        !findFilter(gParams.numFilterPairs, min(clothMask0, clothMask1), max(clothMask0, clothMask1)))
                    {
                        const float separation = satIntersect(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB,
                                                              min(mBestDistance, contactDistance));
                        if (separation < mBestDistance)
                        {
                            mBestDistance = separation;
                            mPrimIndex = primitiveIndex;
                        }
                    }
                }
            }
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
            // intersectBoxFullWarp: the query triangle's bound vs the child box
            const bool intersect = hasBox &&
                !(mn.x > tMax.x || tMin.x > mx.x || mn.y > tMax.y || tMin.y > mx.y || mn.z > tMax.z || tMin.z > mx.z);

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

    gOutBest[workId * 32 + idxInWarp] = mBestDistance;
    gOutPrim[workId * 32 + idxInWarp] = mPrimIndex;
}
