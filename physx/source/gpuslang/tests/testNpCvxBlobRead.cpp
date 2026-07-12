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

// Verification of npCvxBlobRead.slang: reading real convex-hull data from a raw
// uint blob buffer via the tranche-141 byte offsets.  The test builds a blob in
// C++ laid out exactly per the CUDA convex-hull format, then checks the shader
// reads back the exact vertex/plane/descriptor/vertex-index/header values.  All
// reads are exact (no tolerance) — this validates the byte-addressing arithmetic
// (word index, byte shift) and the asfloat bit reinterpretation.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct BlobReadIn { uint32_t baseWord, nbV, nbP, nbE, vi, pi, di, bi; };
struct BlobReadOut
{
    float x, y, z, w;       // vertex
    float px, py, pz, pw;   // plane
    uint32_t polyDesc, vref8, descNbVerts, descMinIdx, vtxIndex, hdrNbEdges, hdrNbVerts, hdrNbPolys, pad0, pad1, pad2, pad3;
    // pad3: BlobReadOut contains float4s, so std430 rounds its stride up to a
    // multiple of 16 (80 bytes); match that here so element addressing lines up.
};

const uint32_t MAXWORDS = 256;
const uint32_t SZ_HEADER = 32;

static uint32_t asuint(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t planesByteOff(uint32_t nbV) { return SZ_HEADER + 16 * nbV; }
static uint32_t polyDescByteOff(uint32_t nbV, uint32_t nbP) { return SZ_HEADER + 16 * nbV + 16 * nbP; }
static uint32_t vertexDataByteOff(uint32_t nbV, uint32_t nbP, uint32_t nbE) { return SZ_HEADER + 16 * nbV + 20 * nbP + 6 * nbE + 3 * nbV; }

PXG_TEST(npCvxBlobRead_read)
{
    std::mt19937 rng(142001);
    const uint32_t N = 10000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<BlobReadIn> in(N);
    std::vector<uint32_t> blob(size_t(N) * MAXWORDS, 0u);
    std::vector<BlobReadOut> ref(N);

    auto setU32 = [&](uint32_t base, uint32_t byteOff, uint32_t v) { blob[base + (byteOff >> 2)] = v; };
    auto setU8 = [&](uint32_t base, uint32_t byteOff, uint32_t v) { blob[base + (byteOff >> 2)] |= (v & 0xffu) << (8 * (byteOff & 3)); };
    auto setF4 = [&](uint32_t base, uint32_t byteOff, float x, float y, float z, float w) {
        const uint32_t wI = base + (byteOff >> 2);
        blob[wI] = asuint(x); blob[wI + 1] = asuint(y); blob[wI + 2] = asuint(z); blob[wI + 3] = asuint(w);
    };

    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nbV = 1 + (rng() % 16);
        const uint32_t nbP = 1 + (rng() % 12);
        const uint32_t nbE = 1 + (rng() % 24);
        const uint32_t base = i * MAXWORDS;

        // header word: getNbEdges=u32High, getNbVerts=u16High(u32Low), getNbPolygons=u16Low(u32Low)
        const uint32_t header = (nbE << 16) | (nbV << 8) | nbP;
        setU32(base, 0, header);

        // vertices
        for (uint32_t v = 0; v < nbV; ++v)
            setF4(base, SZ_HEADER + 16 * v, rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(-1, 1));
        // planes
        const uint32_t pOff = planesByteOff(nbV);
        for (uint32_t p = 0; p < nbP; ++p)
            setF4(base, pOff + 16 * p, rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-2, 2));
        // poly descriptors: desc = (vref8<<16)|(nbVertsOfPoly<<8)|minIdx
        const uint32_t dOff = polyDescByteOff(nbV, nbP);
        for (uint32_t p = 0; p < nbP; ++p)
        {
            const uint32_t vref8 = rng() % 40000;
            const uint32_t nvp = 3 + (rng() % 6);
            const uint32_t minIdx = rng() % 200;
            setU32(base, dOff + 4 * p, (vref8 << 16) | (nvp << 8) | minIdx);
        }
        // vertex-index table (u8), 4*nbP+8 bytes
        const uint32_t vdOff = vertexDataByteOff(nbV, nbP, nbE);
        const uint32_t tblLen = 4 * nbP + 8;
        for (uint32_t j = 0; j < tblLen; ++j)
            setU8(base, vdOff + j, rng() % 256);

        BlobReadIn s;
        s.baseWord = base; s.nbV = nbV; s.nbP = nbP; s.nbE = nbE;
        s.vi = rng() % nbV; s.pi = rng() % nbP; s.di = rng() % nbP; s.bi = rng() % tblLen;
        in[i] = s;

        // reference: read the same locations directly out of the blob we built
        BlobReadOut r{};
        auto getU32 = [&](uint32_t byteOff) { return blob[base + (byteOff >> 2)]; };
        auto getU8 = [&](uint32_t byteOff) { return (blob[base + (byteOff >> 2)] >> (8 * (byteOff & 3))) & 0xffu; };
        auto getF = [&](uint32_t wI) { float f; uint32_t u = blob[wI]; std::memcpy(&f, &u, 4); return f; };
        const uint32_t vW = base + ((SZ_HEADER + 16 * s.vi) >> 2);
        r.x = getF(vW); r.y = getF(vW + 1); r.z = getF(vW + 2); r.w = getF(vW + 3);
        const uint32_t pW = base + ((pOff + 16 * s.pi) >> 2);
        r.px = getF(pW); r.py = getF(pW + 1); r.pz = getF(pW + 2); r.pw = getF(pW + 3);
        r.polyDesc = getU32(dOff + 4 * s.di);
        r.vref8 = (r.polyDesc >> 16) & 0xffffu;
        r.descNbVerts = (r.polyDesc >> 8) & 0xffu;
        r.descMinIdx = r.polyDesc & 0xffu;
        r.vtxIndex = getU8(vdOff + s.bi);
        r.hdrNbEdges = (header >> 16) & 0xffffu;
        r.hdrNbVerts = (header >> 8) & 0xffu;
        r.hdrNbPolys = header & 0xffu;
        ref[i] = r;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BlobReadOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bBlob = createBufferFrom(dev, blob);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct BlobReadParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxBlobRead.spv", "cvxBlobRead",
             { (uint32_t)sizeof(BlobReadIn), (uint32_t)sizeof(uint32_t), (uint32_t)sizeof(BlobReadOut) }, sizeof(BlobReadParams));
    k.dispatch(GROUPS, { bIn, bBlob, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BlobReadOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const BlobReadOut& a = ref[i];
        const BlobReadOut& b = g[i];
        const bool feq = a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w &&
                         a.px == b.px && a.py == b.py && a.pz == b.pz && a.pw == b.pw;
        const bool ieq = a.polyDesc == b.polyDesc && a.vref8 == b.vref8 && a.descNbVerts == b.descNbVerts &&
                         a.descMinIdx == b.descMinIdx && a.vtxIndex == b.vtxIndex &&
                         a.hdrNbEdges == b.hdrNbEdges && a.hdrNbVerts == b.hdrNbVerts && a.hdrNbPolys == b.hdrNbPolys;
        if (!feq || !ieq) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "blobRead: %zu mismatches", mm);

    for (PxgBufferHandle h : { bIn, bBlob, bOut }) dev->destroyBuffer(h);
}

} // namespace
