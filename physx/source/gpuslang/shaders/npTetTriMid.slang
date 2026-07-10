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

// npTetTriMid.slang
//
// Slang port of TriangleLeafBoundMinMaxTraverser<DoEdgeEdge> and the tet-vs-
// triangle satIntersect + top-4 pair extraction
// (gpunarrowphase/src/CUDA/softbodySoftbodyMidPhase.cu) — the softbody-tet
// vs triangle-mesh midphase on the BV32 walk, one 32-thread group per query
// tetrahedron:
//
// - computeTetBoundWithTetrahedron: isValidTet gate (det > 1e-9), tet AABB,
//   constructTetrahedron (4 createPlane planes + centroid = (a+b+c)/3).
// - intersectBoxFullWarp: per-axis AABB-vs-AABB displacement
//   (min.x > tMax.x -> min.x - tMax.x; max.x < tMin.x -> max.x - tMin.x;
//   else 0), squared magnitude < contactDistance^2.
// - intersectPrimitiveFullWarp, per lane: triangle plane
//   (normalized cross, d = -n.v0); OUTER front-face gate
//   triPlane.distance(centroid) >= 0; satIntersect: sep = min over the TRI
//   verts of the tri plane (identically ~0), then max'd with the min
//   projection of the tri verts onto each of the 4 tet planes, early-out
//   whenever sep > tolerance where tolerance = min(bestDistance,
//   contactDistance) EVOLVES with the running best; strict < running
//   (best, primIndex) update.  NOTE: both engine instantiations (<true> at
//   the softbody-vs-trimesh call site, <false> at the tet-vs-cloth one) sit
//   behind the same OUTER gate, which makes satIntersect's internal
//   BackFaceCull check dead code — one kernel covers both.
// - extraction (driver tail, sb_meshMidphaseGeneratePairsLaunch): ballot of
//   lanes with best < contactDistance, count = min(popc, 4); ONE atomicAdd
//   of count; then count rounds of warp argmin (lowest lane wins ties), the
//   winner writing uint4(workId, primIndex, 0, 0) gated
//   by index + i < capacity and invalidating itself with PX_MAX_F32.
//
// INTEGRATION NOTES: the CUDA driver transforms the tet verts into the
// trimesh vertex space (shape2Vertex/transformInv) and applies the per-tet
// surface-hint gate BEFORE the traverser; work items here receive the
// already-transformed verts and pre-filtered tets.  The real pair written is
// make_uint4(cmIdx, tetrahedronIdx, primIndex, 0); this port writes
// (workId, primIndex) since cmIdx/tetrahedronIdx are driver-side identifiers.

static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;
static const float PX_MAX_F32 = 3.4028235e38f;
static const uint MAX_CONTACTS = 4;

struct TtParams { uint numWorkItems; uint maxPairs; uint pad0; uint pad1; }

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

struct TtWork
{
    float4 v0, v1, v2, v3; // query tet verts (already in trimesh vertex space)
    float4 params;         // x = contactDistance
}

[[vk::push_constant]]
ConstantBuffer<TtParams> gParams;

RWStructuredBuffer<Bv32Node> gNodes;    // binding 0
RWStructuredBuffer<uint4> gTriIndices;  // binding 1
RWStructuredBuffer<float4> gTriVerts;   // binding 2
RWStructuredBuffer<TtWork> gWork;       // binding 3
RWStructuredBuffer<uint4> gPairs;       // binding 4 (x = workId, y = primIndex)
RWStructuredBuffer<uint> gPairCount;    // binding 5 (single element)

groupshared int sBv32Nodes[1024];
groupshared uint sMask;
groupshared float sBest[32];
groupshared uint sIndex0;

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// PxPlane(p0, p1, p2): n = normalize(cross(p1-p0, p2-p0)), d = -p0.dot(n)
void makePlane(float3 p0, float3 p1, float3 p2, out float3 n, out float d)
{
    n = getNormalized(cross(p1 - p0, p2 - p0));
    d = -dot(p0, n);
}

// Gu::createPlane: flip so pd lies on the non-positive side
void createPlane(float3 pa, float3 pb, float3 pc, float3 pd, out float3 n, out float d)
{
    makePlane(pa, pb, pc, n, d);
    const float distance = dot(n, pd) + d;
    if (distance > 0.0f)
    {
        n = -n;
        d = -d;
    }
}

// minProject(plane, v0, v1, v2): min signed distance of the TRI verts
float minProject3(float3 n, float d, float3 v0, float3 v1, float3 v2)
{
    return min(dot(n, v0) + d, min(dot(n, v1) + d, dot(n, v2) + d));
}

// satIntersect<BackFaceCull>(tet, v0, v1, v2, tolerance) — tri plane term
// then the four tet planes, early-out on sep > tolerance.  The internal
// back-face cull is retained for fidelity but is dead under the caller's
// outer gate.
float satIntersectTetTri(float3 pn[4], float pd[4], float3 centroid,
                         float3 v0, float3 v1, float3 v2, float tolerance)
{
    float3 n;
    float d;
    makePlane(v0, v1, v2, n, d);
    if (dot(n, centroid) + d < 0.0f)
        return PX_MAX_F32;

    float sep = minProject3(n, d, v0, v1, v2);
    if (sep > tolerance)
        return sep;
    sep = max(sep, minProject3(pn[0], pd[0], v0, v1, v2));
    if (sep > tolerance)
        return sep;
    sep = max(sep, minProject3(pn[1], pd[1], v0, v1, v2));
    if (sep > tolerance)
        return sep;
    sep = max(sep, minProject3(pn[2], pd[2], v0, v1, v2));
    if (sep > tolerance)
        return sep;
    sep = max(sep, minProject3(pn[3], pd[3], v0, v1, v2));
    if (sep > tolerance)
        return sep;
    return sep;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void ctTetTriMid(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint workId = gid.x;
    if (workId >= gParams.numWorkItems) return; // group-uniform
    const uint idxInWarp = gtid.x;
    const TtWork w = gWork[workId];
    const float3 t0 = w.v0.xyz, t1 = w.v1.xyz, t2 = w.v2.xyz, t3 = w.v3.xyz;
    const float contactDist = w.params.x;

    // computeTetBoundWithTetrahedron
    if (!isValidTet(t0, t1, t2, t3)) return; // group-uniform

    const float3 tMin = min(min(t0, t1), min(t2, t3));
    const float3 tMax = max(max(t0, t1), max(t2, t3));

    // constructTetrahedron: planes + centroid ((a+b+c)/3 — as the CUDA does)
    float3 pn[4];
    float pd[4];
    createPlane(t0, t1, t2, t3, pn[0], pd[0]);
    createPlane(t0, t1, t3, t2, pn[1], pd[1]);
    createPlane(t0, t2, t3, t1, pn[2], pd[2]);
    createPlane(t1, t2, t3, t0, pn[3], pd[3]);
    const float3 centroid = (t0 + t1 + t2) * (1.0f / 3.0f);

    // per-lane traverser state
    float bestDist = PX_MAX_F32;
    uint bestPrimIndex = 0xffffffff;

    uint currentNodeIndex = 0;
    bool hasNode = true;
    uint nbStack = 0;

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6;

        if ((currentNodeIndex & 1) != 0)
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
                const uint4 triIdx = gTriIndices[primitiveIndex];
                const float3 v0 = gTriVerts[triIdx.x].xyz;
                const float3 v1 = gTriVerts[triIdx.y].xyz;
                const float3 v2 = gTriVerts[triIdx.z].xyz;

                float3 tn;
                float td;
                makePlane(v0, v1, v2, tn, td);

                if (dot(tn, centroid) + td >= 0.0f)
                {
                    const float separation = satIntersectTetTri(pn, pd, centroid, v0, v1, v2,
                                                                min(bestDist, contactDist));
                    if (separation < bestDist)
                    {
                        bestDist = separation;
                        bestPrimIndex = primitiveIndex;
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
            // intersectBoxFullWarp: AABB-vs-AABB displacement distance
            bool intersect = false;
            if (hasBox)
            {
                float3 displacement;
                displacement.x = (mn.x > tMax.x) ? (mn.x - tMax.x) : ((mx.x < tMin.x) ? (mx.x - tMin.x) : 0.0f);
                displacement.y = (mn.y > tMax.y) ? (mn.y - tMax.y) : ((mx.y < tMin.y) ? (mx.y - tMin.y) : 0.0f);
                displacement.z = (mn.z > tMax.z) ? (mn.z - tMax.z) : ((mx.z < tMin.z) ? (mx.z - tMin.z) : 0.0f);
                intersect = dot(displacement, displacement) < contactDist * contactDist;
            }

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

    // driver-tail extraction: up to MAX_CONTACTS best pairs per query tet
    sBest[idxInWarp] = bestDist;
    const uint mask = ballot32(bestDist < contactDist, idxInWarp);
    uint count = countbits(mask);
    if (count > MAX_CONTACTS) count = MAX_CONTACTS;

    if (count != 0)
    {
        if (idxInWarp == 0)
        {
            uint idx;
            InterlockedAdd(gPairCount[0], count, idx);
            sIndex0 = idx;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint index = sIndex0;

        float bestSep = -1.0f;
        for (uint i = 0; i < count && bestSep < contactDist; ++i)
        {
            // warp argmin (lowest lane wins ties)
            uint winnerLane = 0;
            {
                float best = PX_MAX_F32;
                for (uint j = 0; j < 32; ++j)
                    if (sBest[j] < best) { best = sBest[j]; winnerLane = j; }
                bestSep = best;
            }
            GroupMemoryBarrierWithGroupSync();
            if (winnerLane == idxInWarp)
            {
                if (sBest[idxInWarp] < contactDist)
                {
                    if ((index + i) < gParams.maxPairs)
                        gPairs[index + i] = uint4(workId, bestPrimIndex, 0, 0);
                    sBest[idxInWarp] = PX_MAX_F32;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
