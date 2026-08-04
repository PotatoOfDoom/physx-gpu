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

// pxgSlangConvexHull.slang
//
// The cooked convex-hull blob reader — the `ConvexScratch` accessors from
// convexTriangle.cuh:55-136, transcribed as address arithmetic over a Buffer
// Device Address.
//
// WHY THIS IS ITS OWN TRANCHE.  Every remaining kernel of the convex-vs-mesh
// chain re-derives these addresses: the SAT phase walks the planes and the
// edge lists, the clipper walks a face's vertex list, the post-process walks
// faces-by-vertices.  Getting one offset wrong reads plausible-looking floats
// from the neighbouring section, which is the failure mode that is hardest to
// spot downstream.  Same reasoning as P8.25's BV32 reader.
//
// THE BLOB.  `PxgShape::hullOrMeshPtr` points at:
//
//   @0                                    float4  centerOfMass
//   @16                                   uint4   (.x = the packed hull descriptor)
//   @32                                   float4  (not read by this chain)
//   @48                                   float4  vertices[nbVerts]
//   +16*nbVerts                           float4  planes[nbPolygons]      (n.xyz, d)
//   +16*nbPolygons                        uint    polyDescs[nbPolygons]   (vRef8NbVertsMinIndex)
//   +4*nbPolygons                         uint16  verticesByEdges16[2*nbEdges]
//   +4*nbEdges                            uint8   facesByEdges8[2*nbEdges]
//   +2*nbEdges                            uint8   facesByVertices8[3*nbVerts]
//   +3*nbVerts                            uint8   vertexData8[...]
//
// The three fixed float4/uint4 headers before the vertices are why every
// accessor in the CUDA opens with `sizeof(float4) + sizeof(uint4) + sizeof(float4)`.
//
// THE TWO PACKED WORDS are both three fields in one uint, and they are NOT the
// same packing — the hull descriptor's top half is a 16-bit edge count while a
// polygon descriptor's top half is a 16-bit vertex reference:
//
//   hull descriptor   nbEdges = u32High         -> (v >> 16) & 0xffff
//                     nbVerts = u16High(u32Low) -> (v >> 8)  & 0xff
//                  nbPolygons = u16Low(u32Low)  ->  v        & 0xff
//   polygon descriptor  vRef8 = u32High         -> (v >> 16) & 0xffff
//                     nbVerts = u16High(u32Low) -> (v >> 8)  & 0xff
//                    minIndex = u16Low(u32Low)  ->  v        & 0xff
//
// (cudaNpCommon.h:137-172 over cutil_math.h:765-788.)  The returns are PxU8/PxU16
// in the CUDA, so the masks are not decoration: a hull with 300 polygons would
// wrap, and this port wraps identically.
//
// ALIGNMENT.  vertices and planes are float4 runs at 16-byte multiples FROM THE
// BLOB BASE, but the runs after them are not: polyDescs starts at
// 48 + 16*(nbVerts + nbPolygons), which is 16-aligned, while verticesByEdges16
// starts 4*nbPolygons later and the uint8 runs after that are byte-addressed.
// Everything past the planes is therefore read through scalar uint/ushort/byte
// loads assembled by hand, never through a vector pointer.

// ---------------------------------------------------------------------------
// packed-word decoders
// ---------------------------------------------------------------------------
public uint chNbEdges(uint hullDesc) { return (hullDesc >> 16) & 0xffffu; }
public uint chNbVerts(uint hullDesc) { return (hullDesc >> 8) & 0xffu; }
public uint chNbPolygons(uint hullDesc) { return hullDesc & 0xffu; }

public uint chPolyVRef8(uint polyDesc) { return (polyDesc >> 16) & 0xffffu; }
public uint chPolyNbVerts(uint polyDesc) { return (polyDesc >> 8) & 0xffu; }
public uint chPolyMinIndex(uint polyDesc) { return polyDesc & 0xffu; }

// ---------------------------------------------------------------------------
// section addresses
// ---------------------------------------------------------------------------
static const uint CH_HEADER = 48u;   // float4 + uint4 + float4

public float4 chCenterOfMass(uint64_t hull) { return Ptr<float4>(hull)[0]; }
public uint chHullDesc(uint64_t hull) { return Ptr<uint>(hull + 16UL)[0]; }

public uint64_t chVertices(uint64_t hull)
{
    return hull + uint64_t(CH_HEADER);
}

public uint64_t chPlanes(uint64_t hull, uint hullDesc)
{
    return hull + uint64_t(CH_HEADER) + uint64_t(16u * chNbVerts(hullDesc));
}

public uint64_t chPolyDescs(uint64_t hull, uint hullDesc)
{
    return chPlanes(hull, hullDesc) + uint64_t(16u * chNbPolygons(hullDesc));
}

public uint64_t chVerticesByEdges16(uint64_t hull, uint hullDesc)
{
    return chPolyDescs(hull, hullDesc) + uint64_t(4u * chNbPolygons(hullDesc));
}

public uint64_t chFacesByEdges8(uint64_t hull, uint hullDesc)
{
    return chVerticesByEdges16(hull, hullDesc) + uint64_t(4u * chNbEdges(hullDesc));
}

public uint64_t chFacesByVertices8(uint64_t hull, uint hullDesc)
{
    return chFacesByEdges8(hull, hullDesc) + uint64_t(2u * chNbEdges(hullDesc));
}

public uint64_t chVertexData8(uint64_t hull, uint hullDesc)
{
    return chFacesByVertices8(hull, hullDesc) + uint64_t(3u * chNbVerts(hullDesc));
}

// ---------------------------------------------------------------------------
// element reads
//
// The sub-word runs are read by loading the containing uint and shifting.  A
// Ptr<uint16_t> or Ptr<uint8_t> would need the run to be 2- or 1-aligned in a
// way the blob does not promise once nbPolygons is odd.
// ---------------------------------------------------------------------------
public float4 chVertex(uint64_t hull, uint index)
{
    return Ptr<float4>(chVertices(hull))[index];
}

public float4 chPlane(uint64_t hull, uint hullDesc, uint index)
{
    return Ptr<float4>(chPlanes(hull, hullDesc))[index];
}

public uint chPolyDesc(uint64_t hull, uint hullDesc, uint index)
{
    return Ptr<uint>(chPolyDescs(hull, hullDesc))[index];
}

// THE ALIGNMENT IS TAKEN ON THE ABSOLUTE ADDRESS, not on the index, and the
// reason is narrower than it looks.  Only the first three sections start at a
// 16-byte multiple; `facesByVertices8` starts 2*nbEdges past a 4-aligned point and
// `vertexData8` a further 3*nbVerts, so a section base can sit at any byte.
//
// Rounding the INDEX down instead — `Ptr<uint>(base + (byteIndex & ~3))` shifted
// by `byteIndex & 3` — extracts THE SAME BYTE: both spellings land on
// `base + byteIndex`.  What differs is only the address the uint load is issued
// at, which in that spelling inherits the base's misalignment.  Vulkan requires a
// scalar load to be naturally aligned, so that is undefined behaviour even though
// llvmpipe happens to service it.  The mutation sweep records this as an
// equivalent mutant precisely because no fixture on this platform can see it.
//
// The ushort reader has a real constraint rather than a defensive one: it assumes
// the two bytes never straddle a word boundary, which holds because
// `verticesByEdges16` begins at a 4-aligned address (16-aligned polyDescs plus
// 4*nbPolygons) and its elements are two bytes wide.
uint chByteAt(uint64_t base, uint byteIndex)
{
    const uint64_t addr = base + uint64_t(byteIndex);
    const uint64_t aligned = (addr / 4UL) * 4UL;
    const uint word = Ptr<uint>(aligned)[0];
    return (word >> (8u * uint(addr - aligned))) & 0xffu;
}

uint chUshortAt(uint64_t base, uint shortIndex)
{
    const uint64_t addr = base + uint64_t(2u * shortIndex);
    const uint64_t aligned = (addr / 4UL) * 4UL;
    const uint word = Ptr<uint>(aligned)[0];
    return (word >> (8u * uint(addr - aligned))) & 0xffffu;
}

// verticesByEdges16[2*e + 0/1] — the two hull vertices of edge e.
public uint chEdgeVertex(uint64_t hull, uint hullDesc, uint edge, uint which)
{
    return chUshortAt(chVerticesByEdges16(hull, hullDesc), 2u * edge + which);
}

// facesByEdges8[2*e + 0/1] — the two faces meeting at edge e.
public uint chEdgeFace(uint64_t hull, uint hullDesc, uint edge, uint which)
{
    return chByteAt(chFacesByEdges8(hull, hullDesc), 2u * edge + which);
}

// facesByVertices8[3*v + 0/1/2] — three faces incident on vertex v.
public uint chVertexFace(uint64_t hull, uint hullDesc, uint vertex, uint which)
{
    return chByteAt(chFacesByVertices8(hull, hullDesc), 3u * vertex + which);
}

// vertexData8[polygon's vRef8 + i] — the i-th vertex index of a polygon.
public uint chPolyVertex(uint64_t hull, uint hullDesc, uint vRef8, uint i)
{
    return chByteAt(chVertexData8(hull, hullDesc), vRef8 + i);
}
