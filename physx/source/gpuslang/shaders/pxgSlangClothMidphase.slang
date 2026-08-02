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
// pxgSlangClothMidphase — the machinery every FEM-cloth midphase shares.
//
// Extracted from femClothSelfCollisionBda at P8.15, when the cloth-vs-DIFFERENT-cloth midphase
// became a second consumer.  The same move pxgSlangTetClosest got at P8.8 step 4, and for the
// same reason: the alternative is a second copy of `closestPtLineLine` and the BV32 readers,
// and this port has already recorded what duplicated geometry costs — a mutant that deleted an
// outer gate SURVIVED because a duplicated inner copy still culled (see
// softBodyClothMidphaseBda's header).  One copy, one place to mutate.
//
// WHAT IS HERE: only what is genuinely INDEPENDENT of a particular kernel — the BV32 node
// readers, the two closest-point routines, the bounds and edge-activity predicates, and the
// bit layouts.  Deliberately NOT here:
//
//   * `writeContactInfo` and `ballotGroup`, which reference the importing shader's own push
//     constants and groupshared state.  Lifting them would mean passing those in, which buys
//     nothing and obscures where the barriers are;
//   * the rest-pose distance helpers, which exist only on the SELF-collision path.  The
//     different-cloth call passes NULL rest buffers and `distSqInRestPose` then returns
//     PX_MAX_REAL, so the filter collapses there (femClothClothMidPhase.cu:662-668) — putting
//     them in a shared module would invite exactly the null dereference that collapse avoids.
//
// This module is PURE: no push constants, no groupshared, no side effects.  Every function is
// a transcription already proven bit-exact by the self-collision fixtures.
// ===========================================================================
module pxgSlangClothMidphase;

// PXG_BITSHIFT_ELEMENT_ID (PxgSimulationCoreDesc.h:41): PxEncodeClothIndex(clothId, elem) =
// (clothId << 20) | elem.
public static const uint PXG_CLOTH_ID_SHIFT = 20u;

// Gu::BV32DataPacked (GuBV32.h:88) layout, 1168 bytes:
//   PxVec4 mMin[32] @0 (512), PxVec4 mMax[32] @512 (512), PxU32 mData[32] @1024 (128),
//   PxU32 mNbNodes @1152, PxU32 mDepth @1156, + 8 tail pad -> stride 1168.
public static const uint64_t BV32_STRIDE      = 1168UL;
public static const uint64_t BV32_MMAX_OFFSET = 512UL;
public static const uint64_t BV32_MDATA_OFFSET = 1024UL;
public static const uint64_t BV32_MNBNODES_OFFSET = 1152UL;
public static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11u;

// deformableCollision.cuh:37/38.
public static const float DEFORMABLE_BARYCENTRIC_THRESHOLD = 1.0e-6f;
public static const float DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD = 1.0f - 1.0e-6f;

// PxgFemFemContactInfo::auxInd bit packing (PxgFEMCore.h:69-82).
public static const uint AUX_IND0_BITS = 14u;
public static const uint AUX_IND1_BITS = 15u;
public static const uint AUX_IND0_MASK = (1u << AUX_IND0_BITS) - 1u;                       // 0x00003FFF
public static const uint AUX_IND1_MASK = ((1u << AUX_IND1_BITS) - 1u) << AUX_IND0_BITS;    // 0x1FFFC000
public static const uint AUX_IND1_SHIFT = AUX_IND0_BITS;
public static const uint PAIR_TYPE_FLAG_MASK = 1u << 30;                                   // 0 = VT, 1 = EE
public static const uint VALIDITY_FLAG_MASK  = 1u << 31;

public static const uint WARP = 32u;
public static const uint STACK_SIZE = 192u;   // femMidphaseScratch.sBv32Nodes[192] (6-deep BV32)

// PxgNpKernelIndices.h:32 — the engine's warps-per-block for every midphase launch.
public static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;

// ---------------------------------------------------------------------------
// BV32DataPacked field readers (explicit byte offset through Ptr<> — no whole-node copy).
// treeAddr = mTriMeshData + sizeof(uint4).
// ---------------------------------------------------------------------------
public uint64_t bvNodeBase(uint64_t treeAddr, uint nodeIdx) { return treeAddr + uint64_t(nodeIdx) * BV32_STRIDE; }
public float3 bvMin(uint64_t treeAddr, uint nodeIdx, uint k)
{
    return Ptr<float4>(bvNodeBase(treeAddr, nodeIdx) + uint64_t(k) * 16UL)[0].xyz;
}
public float3 bvMax(uint64_t treeAddr, uint nodeIdx, uint k)
{
    return Ptr<float4>(bvNodeBase(treeAddr, nodeIdx) + BV32_MMAX_OFFSET + uint64_t(k) * 16UL)[0].xyz;
}
public uint bvData(uint64_t treeAddr, uint nodeIdx, uint k)
{
    return Ptr<uint>(bvNodeBase(treeAddr, nodeIdx) + BV32_MDATA_OFFSET + uint64_t(k) * 4UL)[0];
}
public uint bvNbNodes(uint64_t treeAddr, uint nodeIdx)
{
    return Ptr<uint>(bvNodeBase(treeAddr, nodeIdx) + BV32_MNBNODES_OFFSET)[0];
}

// ---------------------------------------------------------------------------
// Geometry helpers (proven bit-exact transcriptions).
// ---------------------------------------------------------------------------
public uint pxEncodeClothIndex(uint clothId, uint elem) { return (clothId << PXG_CLOTH_ID_SHIFT) | elem; }

// isType0EdgeActive (deformableCollision.cuh:360): EdgeEncoding::TYPE0_EDGE_BASE_POS == 0.
public bool isType0EdgeActive(uint edgeAuthorship, uint localEdgeIndex)
{
    return (edgeAuthorship & (1u << localEdgeIndex)) != 0u;
}

// PxBounds3::intersects (PxBounds3.h:353) with this == A, b == B.
public bool boundsIntersect(float3 aMin, float3 aMax, float3 bMin, float3 bMax)
{
    return !(bMin.x > aMax.x || aMin.x > bMax.x || bMin.y > aMax.y || aMin.y > bMax.y ||
             bMin.z > aMax.z || aMin.z > bMax.z);
}

// closestPtPointTriangle (deformableCollision.cuh:146 / npClosest.slang:53).
public float3 closestPtPointTriangle(float3 p, float3 a, float3 b, float3 c, out float s, out float t)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return a; }

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return b; }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        s = v; t = 0.0f;
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return c; }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        s = 0.0f; t = w;
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        s = 1.0f - w; t = w;
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    s = v; t = w;
    return a + ab * v + ac * w;
}

// closestPtLineLine (deformableCollision.cuh:244 / npEdgeEdge.slang:57).  The EE traverser
// calls closestPtLineLine (infinite-line closest points), NOT the edge-edge fallback --
// barycentric s,t may fall outside [0,1] and are gated by DEFORMABLE_* at the caller.
public void closestPtLineLine(float3 e0v0, float3 e0v1, float3 e1v0, float3 e1v1,
                              out float s, out float t, out float distSq)
{
    const float3 r0 = e0v1 - e0v0;
    const float3 r1 = e1v1 - e1v0;
    const float3 d = e1v0 - e0v0;

    const float r0Len2 = dot(r0, r0);
    const float r1Len2 = dot(r1, r1);

    if (r0Len2 < 1.0e-18f || r1Len2 < 1.0e-18f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return;
    }

    const float r0Len = sqrt(r0Len2);
    const float r1Len = sqrt(r1Len2);

    const float3 u0 = r0 / r0Len;
    const float3 u1 = r1 / r1Len;

    const float cosTheta = abs(dot(u0, u1));

    if (cosTheta > 0.9999f)
    {
        const float tProj = dot(e0v0 - e1v0, r1) / r1Len2;
        t = tProj;
        s = 0.0f;

        const float3 pt0 = e0v0;
        const float3 pt1 = e1v0 + r1 * tProj;
        distSq = dot(pt1 - pt0, pt1 - pt0);
        return;
    }

    const float a01 = -dot(u0, u1);
    const float b0 = dot(u0, d) / r0Len;
    const float b1 = -dot(u1, d) / r1Len;

    const float det = 1.0f - a01 * a01;

    if (det < 1.0e-9f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return;
    }

    const float invDet = 1.0f / det;
    s = (b0 - b1 * a01) * invDet;
    t = (b1 - a01 * b0) * invDet;

    const float3 pt0 = e0v0 + s * r0;
    const float3 pt1 = e1v0 + t * r1;
    distSq = dot(pt1 - pt0, pt1 - pt0);
}
