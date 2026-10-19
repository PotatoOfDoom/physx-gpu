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

// npSbSelf.slang
//
// Slang port of the softbody SELF-collision midphase traverser
// (SbMeshTreeTraverser, gpunarrowphase/src/CUDA/softbodySoftbodyMidPhase.cu,
// PhysX 5.12) on the BV32 walk -- one 32-thread group per query vertex:
//
// - intersectBoxFullWarp: the query point's clamped distance to the child box
//   must be below the contact distance ((clamp(p) - p).magSq < distance^2).
// - intersectPrimitiveFullWarp, per lane: surface-hint gate; skip tets sharing
//   the query vertex; the 5.12 element filter -- two exact searches of the
//   softbody's mVolumeVolumeFilterPairs, (vertex, tet) and (ANY, tet), through
//   the ENGINE's findVolumeVolumeFilter (pxgSlangDefFilter); isValidTet;
//   isPointOutsideOfPlane4 (true = the point is INSIDE all four face
//   half-spaces); closestPtTetrahedron with the surface hint; then the contact
//   decision -- inside: dist = v.dot(-normal); outside: dist = |v|, the normal
//   replaced by v when their dot exceeds 0.707, accepted when dist < distance;
//   accepted contacts re-checked in the REST pose (point outside the rest tet
//   and farther than the self-collision filter distance) and dropped when the
//   tetrahedron stress exceeds the tolerance.  The surviving contact updates
//   the per-lane running best (strict <).
// - finalizeFullWarp: warp argmin over the lane bests (lowest lane wins ties);
//   the winner with bestDist < distance appends ONE contact through the ENGINE's
//   PxgDbDbContactWriter::writeVolumeVolumeContact (pxgSlangDeformableWriters):
//   point = (closestP, 0), normalPen = (normal, pen), pair info
//   (PxgEncodeDeformableVertex(sb, vertIdx), PxgEncodeDeformableTetrahedron(sb,
//   bestTet)), no barycentric (sb_fem_contact_remap_to_simLaunch writes both).
//
// 5.12 vs the 5.6.1 transcription this fixture used to carry: the 32-bit
// PxEncodeSoftBodyIndex ids, the PxgNonRigidFilterPair list and the
// vertToSurfaceTetRemap lookup (pairId0 was the vertex's surface TET) are gone.
//
// INTEGRATION NOTES (kernel-entry responsibilities NOT part of this shader --
// see sb_selfCollisionMidphaseGeneratePairsLaunch in the CUDA):
// - params.x (distance) is the 2x-fattened contact distance
//   (contactDistance[elementInd] * 2.f in the CUDA driver).
// - params.y (filterDistanceSq) is the SQUARE of mSelfCollisionFilterDistance
//   (the CUDA traverser squares it in its constructor).
// - The driver applies two per-VERTEX gates before running the traverser:
//   mSurfaceVertsHint[vertIdx] != 0 and the (vertex, ANY tet) filter search;
//   the host-side work-item builder must replicate both.

import pxgSlangDeformableRef;       // PxgEncodeDeformableVertex / Tetrahedron, PXG_DREF_ANY_FEATURE
import pxgSlangDefFilter;           // findVolumeVolumeFilter, the engine's
import pxgSlangDeformableWriters;   // PxgDbDbContactWriterGpu, the engine's

static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;
static const float PX_MAX_F32 = 3.4028235e38f;

// The filter list and the writer's streams are device addresses (BDA), the way the
// engine kernel reads them from the PxgSoftBody and the PxgDbDbContactWriter.
struct SsParams
{
    uint numWorkItems; uint numFilterPairs; uint maxContacts; uint pad0;
    uint64_t filterPairs;       // PxgDeformablePairInfo*, sorted (PxgComparePairInfo)
    uint64_t filterSafeAddr;    // any mapped address, read in place of a NULL list
    uint64_t outPoint;          // float4*
    uint64_t outNormalPen;      // float4*
    uint64_t outContactInfo;    // PxgDeformablePairInfo*
    uint64_t totalContactCount; // PxU32*
}

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

struct SsWork
{
    float4 point;      // query vertex position (world)
    float4 restP;      // rest-pose position
    uint4 ids;         // x = vertIdx, y = softbodyId, z/w unused
    float4 params;     // x = distance, y = filterDistanceSq, z = stressTolerance
}

[[vk::push_constant]]
ConstantBuffer<SsParams> gParams;

RWStructuredBuffer<Bv32Node> gNodes;        // binding 0
RWStructuredBuffer<uint4> gTetIndices;      // binding 1
RWStructuredBuffer<float4> gTetVerts;       // binding 2 (world)
RWStructuredBuffer<float4> gRestVerts;      // binding 3 (rest pose)
RWStructuredBuffer<uint> gSurfaceHint;      // binding 4 (PxU8 widened)
RWStructuredBuffer<float> gStress;          // binding 5 (per tet)
RWStructuredBuffer<SsWork> gWork;           // binding 6

groupshared int sBv32Nodes[1024];
groupshared uint sMask;
groupshared float sBest[32];

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// --- geometry helpers (npSbTet.slang) ---------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// PointOutsideOfPlane4: per-face product of the point's and the opposite
// vertex's signed distances; all >= 0 means the point is inside.
bool isPointOutsideOfPlane4(float3 p, float3 a, float3 b, float3 c, float3 d)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ad = d - a;

    const float3 v0 = cross(ab, ac);
    const float signa0 = dot(v0, ap);
    const float signd0 = dot(v0, ad);

    const float3 v1 = cross(ac, ad);
    const float signa1 = dot(v1, ap);
    const float signd1 = dot(v1, ab);

    const float3 v2 = cross(ad, ab);
    const float signa2 = dot(v2, ap);
    const float signd2 = dot(v2, ac);

    const float3 bd = d - b;
    const float3 bc = c - b;
    const float3 v3 = cross(bd, bc);
    const float signd3 = dot(v3, p - b);
    const float signa3 = dot(v3, a - b);

    return (signa0 * signd0 >= 0.0f) && (signa1 * signd1 >= 0.0f) &&
           (signa2 * signd2 >= 0.0f) && (signa3 * signd3 >= 0.0f);
}

float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return a + v * ab; }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return a + w * ac; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b); }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

void closestPtTetrahedron(float3 point, uint hint, float3 a, float3 b, float3 c, float3 d,
                          inout float3 normal, inout float3 closestP, inout float sqDist)
{
    const float3 ac = c - a;
    const float3 ab = b - a;
    const float3 ad = d - a;

    if ((hint & 1) != 0)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, c, ab, ac);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ab));
            if (dot(ad, normal) < 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 2) != 0)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, d, ab, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ab, ad));
            if (dot(ac, normal) < 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 4) != 0)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, c, d, ac, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ad));
            if (dot(ab, normal) < 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 8) != 0)
    {
        const float3 bd = d - b;
        const float3 bc = c - b;
        const float3 tPoint = closestPtPointTriangle2(point, b, c, d, bc, bd);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(bc, bd));
            if (dot(ab, normal) > 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void ctSbSelfQuery(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint workId = gid.x;
    if (workId >= gParams.numWorkItems) return; // group-uniform
    const uint idxInWarp = gtid.x;
    const SsWork w = gWork[workId];
    const float3 point = w.point.xyz;
    const float3 restP = w.restP.xyz;
    const uint vertIdx = w.ids.x;
    const uint softbodyId = w.ids.y;
    const float distance = w.params.x;
    const float filterDistanceSq = w.params.y;
    const float stressTolerance = w.params.z;

    // per-lane traverser state
    float bestDist = PX_MAX_F32;
    float3 bestClosestP = float3(0, 0, 0);
    float3 bestNormal = float3(0, 0, 0);
    uint bestPrimIndex = 0;

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
                bool bHasCollision = false;
                bool inside = false;
                float dist = PX_MAX_F32;
                float3 closestP = float3(0, 0, 0);
                float3 normal = float3(0, 0, 0);

                const uint4 tetIdx1 = gTetIndices[primitiveIndex];
                const uint hint = gSurfaceHint[primitiveIndex];

                // don't generate contacts with a neighbouring vertex
                if (hint != 0 && !(tetIdx1.x == vertIdx || tetIdx1.y == vertIdx || tetIdx1.z == vertIdx || tetIdx1.w == vertIdx))
                {
                    // A filter pair for these two elements suppresses contact generation
                    // between them; the second search checks whether the tetrahedron is
                    // filtered against any vertex.
                    const DfFilterArray filters = dfFilterArray(gParams.filterPairs, gParams.numFilterPairs,
                                                                gParams.filterSafeAddr);
                    const bool filtered =
                        findVolumeVolumeFilter(filters, softbodyId, vertIdx, softbodyId, primitiveIndex) ||
                        findVolumeVolumeFilter(filters, softbodyId, PXG_DREF_ANY_FEATURE, softbodyId, primitiveIndex);
                    if (!filtered)
                    {
                        float3 a = gTetVerts[tetIdx1.x].xyz;
                        float3 b = gTetVerts[tetIdx1.y].xyz;
                        float3 c = gTetVerts[tetIdx1.z].xyz;
                        float3 d = gTetVerts[tetIdx1.w].xyz;

                        if (isValidTet(a, b, c, d))
                        {
                            inside = isPointOutsideOfPlane4(point, a, b, c, d);

                            float sqDist2 = PX_MAX_F32;
                            closestPtTetrahedron(point, hint, a, b, c, d, normal, closestP, sqDist2);

                            float3 v = point - closestP;
                            normal = -normal;

                            if (inside)
                            {
                                bHasCollision = true;
                                dist = dot(v, normal);
                            }
                            else if (dot(v, normal) > 0.0f)
                            {
                                // P9.36: `dist = v.normalize();` -- ONE reciprocal then three
                                // multiplies (PxVec3.h:260), and the m > 0 guard restored.  It is
                                // reachable: for denormal-scale v, dot(v, normal) can be positive
                                // while dot(v, v) underflows to zero, and the engine then leaves v
                                // untouched and returns 0 where an unguarded divide gives NaN.
                                dist = sqrt(dot(v, v));   // P9.47: magnitude(), not Length
                                if (dist > 0.0f)
                                    v = v * (1.0f / dist);
                                if (dot(normal, v) > 0.707f)
                                    normal = v;
                                bHasCollision = dist < distance;
                            }

                            if (bHasCollision)
                            {
                                a = gRestVerts[tetIdx1.x].xyz;
                                b = gRestVerts[tetIdx1.y].xyz;
                                c = gRestVerts[tetIdx1.z].xyz;
                                d = gRestVerts[tetIdx1.w].xyz;

                                float3 newNorm = float3(0, 0, 0);
                                float3 newClosestP = float3(0, 0, 0);
                                float newSqDist = PX_MAX_F32;

                                const bool inside2 = isPointOutsideOfPlane4(restP, a, b, c, d);
                                closestPtTetrahedron(restP, 0xf, a, b, c, d, newNorm, newClosestP, newSqDist);

                                bHasCollision = !inside2 && newSqDist > filterDistanceSq;

                                if (bHasCollision)
                                {
                                    const float stress = gStress[primitiveIndex];
                                    if (stress > stressTolerance)
                                        bHasCollision = false;
                                }
                            }
                        }
                    }
                }

                if (bHasCollision && dist < bestDist)
                {
                    bestDist = dist;
                    bestNormal = normal;
                    bestClosestP = closestP;
                    bestPrimIndex = primitiveIndex;
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
            // intersectBoxFullWarp: point-to-box clamped distance vs contact distance
            bool intersect = false;
            if (hasBox)
            {
                float3 closest;
                closest.x = max(mn.x, min(mx.x, point.x));
                closest.y = max(mn.y, min(mx.y, point.y));
                closest.z = max(mn.z, min(mx.z, point.z));
                const float3 dv = closest - point;
                intersect = dot(dv, dv) < distance * distance;
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

    // finalizeFullWarp: warp argmin over the lane bests (lowest lane wins ties)
    sBest[idxInWarp] = bestDist;
    GroupMemoryBarrierWithGroupSync();
    uint winnerLane = 0;
    {
        float best = PX_MAX_F32;
        for (uint j = 0; j < 32; ++j)
            if (sBest[j] < best) { best = sBest[j]; winnerLane = j; }
    }

    if (winnerLane == idxInWarp && bestDist < distance)
    {
        PxgDbDbContactWriterGpu writer;
        writer.outPoint = gParams.outPoint;
        writer.outNormalPen = gParams.outNormalPen;
        writer.outBarycentric0 = 0;   // writeVolumeVolumeContact writes neither barycentric
        writer.outBarycentric1 = 0;
        writer.outContactInfo = gParams.outContactInfo;
        writer.totalContactCount = gParams.totalContactCount;
        writer.maxContacts = gParams.maxContacts;

        uint index;
        InterlockedAdd(Ptr<uint>(gParams.totalContactCount)[0], 1, index);
        const float pen = bestDist;
        const uint64_t vertexRef0 = PxgEncodeDeformableVertex(softbodyId, vertIdx);
        const uint64_t tetRef1 = PxgEncodeDeformableTetrahedron(softbodyId, bestPrimIndex);
        writer.writeVolumeVolumeContact(index, float4(bestClosestP, 0.0f), float4(bestNormal, pen),
                                        vertexRef0, tetRef1);
    }
}
