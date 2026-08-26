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

// ===========================================================================
// THE INDEX ENCODING (P9.17a) — bv32Traversal.cuh's `currentNodeIndex`.
//
// The accessors above read the node's MEMORY.  These eight rules decode its
// INDICES, and they are where the format's subtlety actually lives: a packed
// node index carries three different things in one uint, and `mData` carries
// three more in another.  Getting one shift wrong yields a tree walk that
// still terminates and still emits plausible pairs.
//
//   currentNodeIndex:  bit0     = this is a LEAF slot
//                      bits 1-5 = which of the 32 child slots (leaves only)
//                      bits 6+  = the node index in the packed array
//
//   mData:             bit0     = the child is a LEAF
//                      bits 1-6 = nbPrimitives (6 bits, so up to 63)
//                      bits 7+  = primStartIndex
//                      internal child's node offset = mData >> 11, then << 6
//                      to become a node index again
//
// MEASURED, NOT ASSUMED (P9.17a): all eight rules appeared, spelled the same
// way, in all TWENTY-ONE shaders in this corpus that walk a BV32 tree.  The
// traversal LOOP around them is not shared and should not be — the leaf
// predicates of those twenty-one measure at 0.27 median similarity, because
// each one asks a different question of a different primitive.  What is shared
// is only the format decode, and that is what lives here.
//
// `bvDataNbPrimitives` returns the RAW six-bit field, deliberately.  The
// traversal maps ONE LANE PER PRIMITIVE, so a consumer that guards with
// `lane < nbPrimitives` needs no clamp at all — for lane <= 31 the guard is
// identical with or without one.  A consumer that instead uses it as a LOOP
// BOUND must clamp to 32 bug-for-bug, or it emits primitives CUDA never
// emitted.  That is a property of the caller's lane mapping, not of the node
// format, which is why the clamp is not applied here.
// ===========================================================================

// --- currentNodeIndex ---
[ForceInline]
public uint bvNodeIdxOf(uint nodeIndex) { return nodeIndex >> 6u; }
[ForceInline]
public bool bvIsLeaf(uint nodeIndex) { return (nodeIndex & 1u) != 0u; }
[ForceInline]
public uint bvLeafSlotOf(uint nodeIndex) { return (nodeIndex >> 1u) & 31u; }

// The child's own packed index: same node, slot k, marked as a leaf.  The
// traversal overwrites it with bvDataChildNodeIndex when mData says the child
// is internal, so this is the LEAF spelling and the default.
[ForceInline]
public uint bvChildIndex(uint nodeIndex, uint k)
{
    return (nodeIndex & 0xFFFFFFC0u) | (k << 1u) | 1u;
}

// --- mData ---
[ForceInline]
// Spelled in the NEGATIVE sense on purpose: all twenty-one traversals ask "is
// this child internal", never "is it a leaf", and `!bvDataIsLeaf(...)` compiles
// to OpINotEqual + OpLogicalNot where this compiles to a single OpIEqual --
// one wasted instruction in the innermost loop of a tree walk, and it costs the
// proof that this refactor changes no code.  Bit 0 still MEANS isLeaf.
public bool bvDataIsInternal(uint data) { return (data & 1u) == 0u; }
[ForceInline]
public uint bvDataNbPrimitives(uint data) { return (data >> 1u) & 63u; }
[ForceInline]
public uint bvDataPrimStart(uint data) { return data >> 7u; }
[ForceInline]
public uint bvDataChildNodeIndex(uint data)
{
    return (data >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) << 6u;
}
