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

// npCvxBlob.slang
//
// Slang port of the convex-hull blob bit-field accessors (cudaNpCommon.h /
// cutil_math.h) and the blob sub-array byte-offset arithmetic.  These are the
// memory-layout foundation the remaining convex-convex drivers build on.  All
// operations are pure integer, hence bit-deterministic.
//
// Bit primitives (cutil_math.h):
//   u32High(v) = v >> 16          u32Low(v)  = v & 0xffff
//   u16High(v) = (v & 0xffff) >> 8 (of a u16)  u16Low(v) = v & 0xff
//   merge(hi16,lo16) = (hi16<<16)|lo16   merge(hi8,lo8) = (hi8<<8)|lo8
//
// Packed-descriptor accessors (cudaNpCommon.h):
//   getVRef8/getNbEdges(val)      = u32High(val)
//   getNbVerts(val)               = u16High(u32Low(val))
//   getMinIndex/getNbPolygons/getNbAdjVerts(val) = u16Low(u32Low(val))
//   u32ToValency: mCount=u32High, mOffset=u32Low ; valencyToPxU32=merge(count,offset)
//
// Blob byte offsets (from generatedContactsEEContacts, cudaGJKEPA.cu):
//   sizeof(uint4)=16, sizeof(float4)=16, sizeof(PxU32)=4, sizeof(PxU16)=2, sizeof(PxU8)=1
//   pPolygonPlanes  = 16 + 16 + 16*nbV
//   vRef8...MinIndex = pPolygonPlanes + 16*nbP
//   polyData_vertexData8 = 16 + 16 + 16*nbV + (16+4)*nbP + (2+1)*2*nbE + 1*3*nbV

uint u32HighW(uint v) { return (v >> 16) & 0xffffu; }
uint u32LowW(uint v)  { return v & 0xffffu; }
uint u16HighW(uint v) { return (v >> 8) & 0xffu; }   // v already a u16
uint u16LowW(uint v)  { return v & 0xffu; }
uint merge16(uint hi, uint lo) { return ((hi & 0xffffu) << 16) | (lo & 0xffffu); }
uint merge8(uint hi, uint lo)  { return ((hi & 0xffu) << 8) | (lo & 0xffu); }

uint getVRef8W(uint val)     { return u32HighW(val); }
uint getNbVertsW(uint val)   { return u16HighW(u32LowW(val)); }
uint getMinIndexW(uint val)  { return u16LowW(u32LowW(val)); }
// merge(hi16, merge(lohi8, lolo8))
uint merge3(uint hi16, uint lohi8, uint lolo8) { return merge16(hi16, merge8(lohi8, lolo8)); }

struct BlobIn { uint4 wordNvNpNe; } // (packedWord, nbVertices, nbPolygons, nbEdges)

struct BlobOut
{
    uint vref8;        // u32High
    uint nbVerts;      // u16High(u32Low)
    uint minIndex;     // u16Low(u32Low)
    uint lo16;         // u32Low
    uint valCount;     // u32High
    uint valOffset;    // u32Low
    uint valMerged;    // merge(count, offset) == word
    uint word3merged;  // merge3(hi16, midByte, loByte) == word
    uint planesOff;
    uint polyDescOff;
    uint vertexDataOff;
    uint pad0;
}

struct BlobParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BlobParams> gParams;

RWStructuredBuffer<BlobIn> gIn;    // binding 0
RWStructuredBuffer<BlobOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxBlob(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint4 in = gIn[t].wordNvNpNe;
    const uint word = in.x;
    const uint nbV = in.y, nbP = in.z, nbE = in.w;

    const uint lo16 = u32LowW(word);
    const uint hi16 = u32HighW(word);
    const uint midByte = u16HighW(lo16);
    const uint loByte = u16LowW(lo16);

    BlobOut r;
    r.vref8 = getVRef8W(word);
    r.nbVerts = getNbVertsW(word);
    r.minIndex = getMinIndexW(word);
    r.lo16 = lo16;
    r.valCount = u32HighW(word);
    r.valOffset = u32LowW(word);
    r.valMerged = merge16(u32HighW(word), u32LowW(word)); // round-trips to word
    r.word3merged = merge3(hi16, midByte, loByte);        // round-trips to word

    r.planesOff = 32u + 16u * nbV;
    r.polyDescOff = 32u + 16u * nbV + 16u * nbP;
    r.vertexDataOff = 32u + 16u * nbV + 20u * nbP + 6u * nbE + 3u * nbV;
    r.pad0 = 0u;
    gOut[t] = r;
}
