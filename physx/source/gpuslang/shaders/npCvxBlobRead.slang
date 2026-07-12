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

// npCvxBlobRead.slang
//
// Builds on tranche 141: reads real convex-hull data from a raw uint blob buffer
// using the T141 byte offsets.  The blob layout (per convexPtr in cudaGJKEPA.cu)
// from byte 0:
//   uint4  header (packed nbEdgesNbHullVerticesNbPolygons in header.x)
//   float4 (hull data, e.g. centre/extents)
//   float4[nbVertices]   vertex cloud
//   float4[nbPolygons]   polygon planes
//   PxU32 [nbPolygons]   poly descriptors (getVRef8/getNbVerts/getMinIndex)
//   (PxU16+PxU8)*2*nbEdges edge data
//   PxU8 [3*nbVertices]    per-vertex data
//   PxU8 []                polygon-vertex-index table (indexed getVRef8(desc)+i)
// Reads use asfloat on the uint buffer (bodyUpdate/directApi pattern); the u8
// index table is byte-addressed.

static const uint SZ_HEADER = 32u; // sizeof(uint4)+sizeof(float4)

// T141 bit-field accessors
uint u32HighW(uint v) { return (v >> 16) & 0xffffu; }
uint u32LowW(uint v)  { return v & 0xffffu; }
uint u16HighW(uint v) { return (v >> 8) & 0xffu; }
uint u16LowW(uint v)  { return v & 0xffu; }
uint getVRef8W(uint val)    { return u32HighW(val); }
uint getNbVertsW(uint val)  { return u16HighW(u32LowW(val)); }
uint getMinIndexW(uint val) { return u16LowW(u32LowW(val)); }
uint getNbEdgesW(uint val)     { return u32HighW(val); }
uint getNbPolygonsW(uint val)  { return u16LowW(u32LowW(val)); }

// T141 blob byte offsets
uint planesByteOff(uint nbV) { return SZ_HEADER + 16u * nbV; }
uint polyDescByteOff(uint nbV, uint nbP) { return SZ_HEADER + 16u * nbV + 16u * nbP; }
uint vertexDataByteOff(uint nbV, uint nbP, uint nbE) { return SZ_HEADER + 16u * nbV + 20u * nbP + 6u * nbE + 3u * nbV; }

struct BlobReadIn
{
    uint baseWord;   // start of this test's blob in the shared uint buffer
    uint nbV, nbP, nbE;
    uint vi;         // vertex index to read
    uint pi;         // plane index to read
    uint di;         // poly-descriptor index to read
    uint bi;         // polygon-vertex-index table entry to read (relative to table start)
}

struct BlobReadOut
{
    float4 vertex;
    float4 plane;
    uint polyDesc;
    uint vref8;      // getVRef8(polyDesc)
    uint descNbVerts;// getNbVerts(polyDesc)
    uint descMinIdx; // getMinIndex(polyDesc)
    uint vtxIndex;   // polygon-vertex-index table byte
    uint hdrNbEdges; // getNbEdges(header word)
    uint hdrNbVerts; // getNbVerts(header word)
    uint hdrNbPolys; // getNbPolygons(header word)
    uint pad0, pad1, pad2;
}

struct BlobReadParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BlobReadParams> gParams;

RWStructuredBuffer<BlobReadIn> gIn;   // binding 0
RWStructuredBuffer<uint>       gBlob; // binding 1 : shared uint blob storage
RWStructuredBuffer<BlobReadOut> gOut; // binding 2

// float4 at 16-aligned byte offset within this test's blob
float4 loadFloat4(uint baseWord, uint byteOff)
{
    const uint w = baseWord + (byteOff >> 2);
    return float4(asfloat(gBlob[w]), asfloat(gBlob[w + 1]), asfloat(gBlob[w + 2]), asfloat(gBlob[w + 3]));
}
// u32 at 4-aligned byte offset
uint loadU32(uint baseWord, uint byteOff) { return gBlob[baseWord + (byteOff >> 2)]; }
// u8 at any byte offset
uint loadU8(uint baseWord, uint byteOff)
{
    const uint word = gBlob[baseWord + (byteOff >> 2)];
    return (word >> (8u * (byteOff & 3u))) & 0xffu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxBlobRead(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const BlobReadIn s = gIn[t];
    const uint b = s.baseWord;

    const uint header = loadU32(b, 0u);

    const uint pOff = planesByteOff(s.nbV);
    const uint dOff = polyDescByteOff(s.nbV, s.nbP);
    const uint vdOff = vertexDataByteOff(s.nbV, s.nbP, s.nbE);

    const float4 vertex = loadFloat4(b, SZ_HEADER + 16u * s.vi);
    const float4 plane = loadFloat4(b, pOff + 16u * s.pi);
    const uint polyDesc = loadU32(b, dOff + 4u * s.di);
    const uint vtxIndex = loadU8(b, vdOff + s.bi);

    BlobReadOut r;
    r.vertex = vertex;
    r.plane = plane;
    r.polyDesc = polyDesc;
    r.vref8 = getVRef8W(polyDesc);
    r.descNbVerts = getNbVertsW(polyDesc);
    r.descMinIdx = getMinIndexW(polyDesc);
    r.vtxIndex = vtxIndex;
    r.hdrNbEdges = getNbEdgesW(header);
    r.hdrNbVerts = getNbVertsW(header);
    r.hdrNbPolys = getNbPolygonsW(header);
    r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
    gOut[t] = r;
}
