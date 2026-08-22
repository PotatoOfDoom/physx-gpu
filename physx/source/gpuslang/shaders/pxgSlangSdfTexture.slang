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

// pxgSlangSdfTexture.slang -- P9.12c, the SparseSDFTexture layer.
//
// Ports sdfCollision.cuh:43-152 (the SparseSDFTexture struct, its initialize()
// and its Sample()) plus the SDF tail of readTriangleMesh (triangleMesh.cuh:120
// -196) onto the bindless texture layer P9.12a landed.
//
// WHAT THIS LAYER IS.  A PhysX SDF is stored two ways at once.  A coarse
// background grid covers the whole box; where the surface actually is, a
// narrow band of SUBGRIDS carries higher resolution, packed into a second 3D
// texture and addressed through a per-coarse-cell start slot.  Sample() is the
// lookup that decides, per point, which of the two it reads -- and essentially
// all of this tranche's risk is in that addressing, not in the filtering
// (P9.12a already measured the filter).
//
// THE ORACLE, AND IT IS A REAL ONE.  Gu::SDF::decodeSparse (GuSDF.cpp:1340-1412,
// exported PX_PHYSX_COMMON_API) evaluates the same sparse SDF on the HOST at an
// integer fine-grid coordinate, and it makes every one of the same decisions:
// the coarse cell by integer division, the far-edge fix-up, decodeTriple's
// 10-bit unpack, the (subgridSize+1) block stride, and the coarse fall-back over
// a (nbX+1, nbY+1, nbZ+1) grid.  At a texel CENTRE the trilinear weights are 0
// and 1, so the GPU sample must equal it.  The fixture compares against that,
// not against a second transcription of this file.
//
// ---------------------------------------------------------------------------
// THE COOKED-MESH BLOB, past where any earlier shader has read
// (triangleMesh.cuh:120-196, written by PxgGeometryManager.cpp:520-700)
// ---------------------------------------------------------------------------
//   @0                              uint4  (numVerts, numTris, _, nbPackedNodes)
//   @16                             BV32DataPacked[nbPackedNodes]      (1168 each)
//   +16*numVerts                    float4 verts[numVerts]
//   +16*numTris                     uint4  indices[numTris]
//   +16*numTris                     uint4  adjacencies[numTris]
//   +4*numTris                      uint32 faceRemap[numTris]
//   +4*(numVerts+1)                 uint32 accumulatedTrianglesRef[numVerts+1]
//   +4*numTris*3                    uint32 trianglesReferences[numTris*3]
//   ALIGN UP TO 16
//   uint4  sdfDims          .xyz = the FINE grid dimensions,
//                           .w   = getPreferSDFProjection() ? 1 : 0  -- a FLAG,
//                                  not a dimension, and not part of the count
//   if (sdfDims.x * sdfDims.y * sdfDims.z != 0):
//     float4 (meshLower.xyz, spacing)
//     uint4  (subgridSize, bits(subgridsMinSdfValue), bits(subgridsMaxSdfValue),
//             numStartSlots)
//     uint64 mTexObject             <- the coarse texture handle
//     uint64 mTexObjectSparse       <- the subgrid texture handle
//     uint32 subgridStarts[numStartSlots]
//
// B1. The last three sections before the SDF -- faceRemap, accumulatedTrianglesRef
//     and trianglesReferences -- have NEVER been walked by this port before, so
//     every earlier count has to be right for the SDF to be found at all.  Note
//     accumulatedTrianglesRef is numVerts+1 entries, not numVerts.
// B2. The 16-BYTE ALIGNMENT before sdfDims is load-bearing and is easy to miss
//     because it is the only alignment step in the whole blob.
// B3. `sdfDims.w` is NOT a dimension.  numSdfs is x*y*z; a port that multiplies
//     in .w reports "no SDF" for every mesh whose projection flag is 0.
// B4. On the NO-SDF path the CUDA sets ONLY mTexObject = 0 and leaves
//     mTexObjectSparse, subgridStarts, spacing, meshLower, subgridSize and both
//     subgrid value bounds UNINITIALISED (triangleMesh.cuh:193-194).  The struct
//     lives in shared memory, so those are whatever the block last held.  This
//     port returns a bool instead and zeroes the rest -- a DELIBERATE divergence
//     from a transcription, recorded here rather than hidden: the CUDA's
//     behaviour is only safe because the host never routes a non-SDF mesh to the
//     SDF kernel, and see the candidate note at the bottom of this file.
//
// ---------------------------------------------------------------------------
// THE TRAPS in initialize() and Sample()
// ---------------------------------------------------------------------------
//  T1. THE MARGIN IS ASYMMETRIC BETWEEN THE TWO MODES.  `margin` is 0.5 for a
//      DENSE sdf and 0.0 for a sparse one (sdfCollision.cuh:70), and it shifts
//      both ends of the box.  A port that picks either constant unconditionally
//      is right for one mode and off by half a voxel in the other.
//  T2. `coarseSize` divides by `PxMax(1u, subgridSize)` (:80-82) -- the guard is
//      what stops the DENSE path dividing by zero, and it is easy to drop
//      because on the sparse path it does nothing.
//  T3. `fineToCoarse = 1.0f / subgridSize` is NOT guarded (:85).  On the dense
//      path it is +inf.  That is harmless only because Sample's dense branch
//      never touches it -- transcribed as-is, and the fixture pins the inf.
//  T4. `subgridsSdfValueRange` is stored as max - min (:88), and
//      applySubgridSdfScale is `v * range + min` (:107).  Storing max and
//      subtracting at use is NOT the same expression and rounds differently.
//  T5. Sample takes `f` BY VALUE and MUTATES it (:112).  Both mutations below
//      are on that local copy.
//  T6. The dense branch samples at `f + 0.5` (:117) -- the texel-centre
//      alignment, applied on BOTH branches but with different f.
//  T7. The coarse-cell index is `PxU32(f.x * fineToCoarse)` -- a float-to-uint
//      TRUNCATION, then clamped to `coarseSize.x - 1`.  That subtraction is
//      UNSIGNED: a coarseSize of 0 wraps it to 0xFFFFFFFF and the clamp becomes
//      a no-op.  See the candidate note.
//  T8. On the NOT-RESIDENT path `f *= fineToCoarse` (:132) -- the rescale into
//      coarse units happens ONLY there, and the coarse texture is
//      (coarseSize+1)^3, not coarseSize^3 (PxgGeometryManager.cpp:659).
//  T9. On the RESIDENT path the local coordinate is clamped to
//      [0, subgridSize + 1] (:137-139), NOT to subgridSize.  The +1 is the
//      shared face between neighbouring subgrids, and clamping to subgridSize
//      loses the last sample plane of every block.
// T10. applySubgridStart ADDS (`f.x += ...`, :101-107); it does not assign.  Its
//      three fields are 10 bits each, LOW bits first, each scaled by
//      (subgridSize + 1) -- the packed block stride.
// T11. applySubgridSdfScale is applied ONLY on the resident path.  The coarse
//      fall-back returns the raw sample, because the coarse texture is FLOAT and
//      already carries real distances (PxgGeometryManager.cpp:659 vs :663-675).
// T12. idx(x,y,z,width,height) = z*width*height + y*width + x (:95) -- the start
//      slot lookup uses (coarseSize.x, coarseSize.y) as width/height, so a port
//      that passes the FINE dims reads a slot belonging to another cell.
//
// UPSTREAM CANDIDATE 25 -- LATENT, and stated because T7 makes it visible.
//   `coarseSize.x - 1` (sdfCollision.cuh:82-84 via PxClamp at :121-123) is an
//   unsigned decrement of a value that is `sdfDims.x / subgridSize`.  A cooked
//   SDF whose fine dimension is SMALLER than its subgrid size gives coarseSize 0
//   and the clamp upper bound wraps to 0xFFFFFFFF, so `xBase` is used unclamped
//   and indexes subgridStartSlots out of bounds.  Not reachable through the
//   public cooking API as far as this port can tell -- PxSDFDesc's subgrid size
//   is chosen against the resolution -- so it is DOCUMENTED and NOT REPORTED,
//   like the twenty-four before it.

import pxgSlangTex3d;

// sizeof(Gu::BV32DataPacked), GuBV32.h:88 -- 512 + 512 + 128 + 16.  The same
// number every mesh-walking shader in this tree already carries; the layout test
// pins it against the engine header rather than trusting the repetition.
static const uint SDF_BV32_STRIDE = 1168u;

// ---------------------------------------------------------------------------
// The SDF half of PxgTriangleMesh (triangleMesh.cuh:39-66), as this layer needs
// it.  The geometry half (verts, indices, the BV32 tree) is deliberately absent:
// the SDF path never touches it, and carrying it would invite a second, diverging
// copy of the mesh reader that already exists in pxgSlangCvxTriSetup.
// ---------------------------------------------------------------------------
struct PxgSdfMesh
{
    uint4 sdfDims;                 // .w is the projection FLAG (B3)
    float3 meshLower;
    float spacing;
    uint subgridSize;
    float subgridsMinSdfValue;
    float subgridsMaxSdfValue;
    uint numStartSlots;
    uint texObject;                // bindless slot; 0 == none, as CUDA means it
    uint texObjectSparse;
    uint64_t subgridStarts;        // device address of uint[numStartSlots]
};

// Walks the cooked blob to its SDF tail.  Returns false when the mesh carries no
// SDF, with every field zeroed (B4).
bool sdfReadMesh(uint64_t meshPtr, out PxgSdfMesh m)
{
    m.sdfDims = uint4(0u, 0u, 0u, 0u);
    m.meshLower = float3(0.0f, 0.0f, 0.0f);
    m.spacing = 0.0f;
    m.subgridSize = 0u;
    m.subgridsMinSdfValue = 0.0f;
    m.subgridsMaxSdfValue = 0.0f;
    m.numStartSlots = 0u;
    m.texObject = 0u;
    m.texObjectSparse = 0u;
    m.subgridStarts = 0UL;

    const uint4 header = Ptr<uint4>(meshPtr)[0];
    const uint numVerts = header.x;
    const uint numTris = header.y;
    const uint nbPackedNodes = header.w;      // .z is NOT the node count

    uint64_t p = meshPtr + 16UL;
    p += uint64_t(nbPackedNodes) * uint64_t(SDF_BV32_STRIDE);
    p += uint64_t(numVerts) * 16UL;           // float4 verts
    p += uint64_t(numTris) * 16UL;            // uint4 indices
    p += uint64_t(numTris) * 16UL;            // uint4 adjacencies
    p += uint64_t(numTris) * 4UL;             // uint32 faceRemap
    p += uint64_t(numVerts + 1u) * 4UL;       // uint32 accumulatedTrianglesRef  (B1: +1)
    p += uint64_t(numTris) * 12UL;            // uint32 trianglesReferences[3*numTris]
    // B2, and NOTE THE SPELLING.  `& ~15UL` MISCOMPILES: slangc evaluates the
    // complement in 32 bits and zero-extends, so the constant is
    // 0x00000000FFFFFFF0 and the AND clears the TOP HALF OF THE DEVICE ADDRESS.
    // Measured here as `OpConstant %ulong 4294967280` and a segfault on an
    // llvmpipe worker thread with no output at all.  `~uint64_t(15)` and
    // `~15ull` both give the full 0xFFFFFFFFFFFFFFF0; only the `UL` suffix is
    // broken.  femClothStepBda.slang:144-150 hit this before and wrote the rule
    // down locally; it is repeated here because a rule that lives in one
    // shader's comment is a rule nobody finds twice.
    p = (p + 15UL) & ~uint64_t(15);

    const uint4 dims = Ptr<uint4>(p)[0];
    p += 16UL;
    m.sdfDims = dims;

    // B3: the count is x*y*z.  `.w` is a flag and must not enter it.
    if ((dims.x * dims.y * dims.z) == 0u)
        return false;

    const float4 lowerSpacing = Ptr<float4>(p)[0];
    p += 16UL;
    const uint4 sizeInfo = Ptr<uint4>(p)[0];
    p += 16UL;

    m.meshLower = lowerSpacing.xyz;
    m.spacing = lowerSpacing.w;
    m.subgridSize = sizeInfo.x;
    m.subgridsMinSdfValue = asfloat(sizeInfo.y);
    m.subgridsMaxSdfValue = asfloat(sizeInfo.z);
    m.numStartSlots = sizeInfo.w;

    // The two CUtexObjects.  P9.12a made a texture handle a small integer, so the
    // engine's 64-bit slot carries it in its low word and the high word is zero.
    m.texObject = uint(Ptr<uint64_t>(p)[0]);
    p += 8UL;
    m.texObjectSparse = uint(Ptr<uint64_t>(p)[0]);
    p += 8UL;

    m.subgridStarts = p;
    return true;
}

// ---------------------------------------------------------------------------
// SparseSDFTexture (sdfCollision.cuh:43-152), field for field and in order.
// ---------------------------------------------------------------------------
struct SparseSdfTexture
{
    float3 sdfBoxLower;
    float sdfDx;
    float3 sdfBoxHigher;
    float invSdfDx;
    uint texture;
    uint textureSubgrids;
    uint3 coarseSize;
    uint subgridSize;
    uint64_t subgridStartSlots;
    float fineToCoarse;
    uint4 sdfDims;
    float subgridsMinSdfValue;
    float subgridsSdfValueRange;
};

SparseSdfTexture sdfTexInitialize(PxgSdfMesh mesh)
{
    SparseSdfTexture t;
    t.texture = mesh.texObject;

    // T1: 0.5 for dense, 0.0 for sparse, and it moves BOTH ends.
    const float margin = mesh.subgridSize == 0u ? 0.5f : 0.0f;
    t.sdfBoxLower = mesh.meshLower + margin * float3(mesh.spacing, mesh.spacing, mesh.spacing);
    t.sdfBoxHigher = mesh.meshLower + float3((float(mesh.sdfDims.x) - margin) * mesh.spacing,
                                             (float(mesh.sdfDims.y) - margin) * mesh.spacing,
                                             (float(mesh.sdfDims.z) - margin) * mesh.spacing);
    t.invSdfDx = 1.0f / mesh.spacing;
    t.sdfDx = mesh.spacing;

    t.textureSubgrids = mesh.texObjectSparse;
    t.subgridStartSlots = mesh.subgridStarts;
    t.sdfDims = mesh.sdfDims;

    // T2: the PxMax is what keeps the DENSE path from dividing by zero.
    const uint denom = max(1u, mesh.subgridSize);
    t.coarseSize = uint3(mesh.sdfDims.x / denom, mesh.sdfDims.y / denom, mesh.sdfDims.z / denom);
    t.subgridSize = mesh.subgridSize;
    // T3: NOT guarded -- +inf on the dense path, and dead there.
    t.fineToCoarse = 1.0f / float(mesh.subgridSize);

    t.subgridsMinSdfValue = mesh.subgridsMinSdfValue;
    // T4: the RANGE is stored, not the max.
    t.subgridsSdfValueRange = mesh.subgridsMaxSdfValue - mesh.subgridsMinSdfValue;
    return t;
}

// idx(), sdfCollision.cuh:95.  T12: width/height are the COARSE dimensions.
uint sdfTexIdx(uint x, uint y, uint z, uint width, uint height)
{
    return z * width * height + y * width + x;
}

// applySubgridStart(), sdfCollision.cuh:98-106.  T10: three 10-bit fields, low
// first, each scaled by the packed block stride, and it ADDS.
void sdfApplySubgridStart(uint triple, inout float3 f, uint subgridSize)
{
    f.x += float((triple & 1023u) * (subgridSize + 1u));
    triple = triple >> 10;
    f.y += float((triple & 1023u) * (subgridSize + 1u));
    triple = triple >> 10;
    f.z += float((triple & 1023u) * (subgridSize + 1u));
}

// applySubgridSdfScale(), sdfCollision.cuh:108-111.
float sdfApplySubgridScale(SparseSdfTexture t, float rawSdfValue)
{
    return rawSdfValue * t.subgridsSdfValueRange + t.subgridsMinSdfValue;
}

// Sample(), sdfCollision.cuh:113-151.  T5: `f` is the CUDA's by-value parameter,
// mutated locally on both sparse branches.
float sdfTexSample(SparseSdfTexture t, float3 f)
{
    if (t.subgridSize == 0u)
    {
        // Dense sdf.  T6: the +0.5 is the texel-centre alignment.
        return pxgTex3D(t.texture, f + float3(0.5f, 0.5f, 0.5f));
    }

    // T7: a float-to-uint TRUNCATION, then a clamp whose upper bound is an
    // UNSIGNED decrement (candidate 25).
    const uint xBase = clamp(uint(f.x * t.fineToCoarse), 0u, t.coarseSize.x - 1u);
    const uint yBase = clamp(uint(f.y * t.fineToCoarse), 0u, t.coarseSize.y - 1u);
    const uint zBase = clamp(uint(f.z * t.fineToCoarse), 0u, t.coarseSize.z - 1u);

    const uint i = sdfTexIdx(xBase, yBase, zBase, t.coarseSize.x, t.coarseSize.y);
    const uint startSlot = Ptr<uint>(t.subgridStartSlots)[i];

    if (startSlot == 0xFFFFFFFFu)
    {
        // T8: the rescale into coarse units happens ONLY here, and the coarse
        // texture is (coarseSize + 1) on each axis.
        // T11: no subgrid rescale -- the coarse texture is FLOAT and already
        // carries real distances.
        f *= t.fineToCoarse;
        return pxgTex3D(t.texture, f + float3(0.5f, 0.5f, 0.5f));
    }

    // T9: clamped to subgridSize + 1, the shared face between neighbouring
    // blocks -- not to subgridSize.
    f.x = clamp(f.x - float(xBase * t.subgridSize), 0.0f, float(t.subgridSize + 1u));
    f.y = clamp(f.y - float(yBase * t.subgridSize), 0.0f, float(t.subgridSize + 1u));
    f.z = clamp(f.z - float(zBase * t.subgridSize), 0.0f, float(t.subgridSize + 1u));

    sdfApplySubgridStart(startSlot, f, t.subgridSize);

    const float v = pxgTex3D(t.textureSubgrids, f + float3(0.5f, 0.5f, 0.5f));
    return sdfApplySubgridScale(t, v);
}
