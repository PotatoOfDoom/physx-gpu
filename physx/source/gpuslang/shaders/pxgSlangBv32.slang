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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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
// pxgSlangBv32 — the Gu::BV32DataPacked reader, once.
//
// Extracted at P8.25.  Four shaders carried byte-identical private copies of
// these five constants and five accessors — pxgSlangClothMidphase,
// softBodyMeshMidphaseBda, femClothMeshMidphaseBda and
// softBodySoftBodyMidphaseBda — and the convex-vs-trimesh midphase was about to
// make it five.  Same reasoning as P8.24's SAT extraction and P8.15's cloth
// midphase: this port has already had a mutant survive because a DUPLICATED
// inner copy still culled after its outer gate was deleted.  One copy, one place
// to mutate.
//
// Gu::BV32DataPacked (GuBV32.h), 1168 bytes per node:
//   mMin[32]    float4  @0     (.xyz used, .w padding)
//   mMax[32]    float4  @512
//   mData[32]   uint    @1024
//   mNbNodes    uint    @1152
//
// mData encodes a child slot:  bit0 = isLeaf
//                              bits 1..6  = nbPrimitives (6 bits, so up to 63)
//                              bits 7..   = primStartIndex
// and for an INTERNAL child the node offset is mData >> 11
// (GU_BV4_CHILD_OFFSET_SHIFT_COUNT).
//
// NOTE on nbPrimitives: the field is six bits wide and can express 63, while
// bv32Traversal.cuh maps ONE LANE PER PRIMITIVE and therefore silently drops
// anything past 32.  Consumers must clamp to 32 bug-for-bug rather than write
// the obvious loop, or they emit triangles CUDA never emitted.  The clamp lives
// in the consumer because it is a property of the traversal's lane mapping, not
// of the node format.
// ===========================================================================

module pxgSlangBv32;

public static const uint64_t BV32_STRIDE      = 1168UL;
public static const uint64_t BV32_MMAX_OFFSET = 512UL;
public static const uint64_t BV32_MDATA_OFFSET = 1024UL;
public static const uint64_t BV32_MNBNODES_OFFSET = 1152UL;
public static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11u;

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
