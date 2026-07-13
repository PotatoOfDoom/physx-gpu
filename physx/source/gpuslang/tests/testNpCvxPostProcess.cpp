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

// Verification of npCvxPostProcess.slang: convexTrimeshPostProcessCore +
// convexHeightfieldPostProcessCore (convexMeshPostProcess.cu L182-365,
// L410-582).  The CPU reference transcribes the CUDA exactly: the
// pairIdx/eMASK word decode, the nbContacts==0 early-out, the trimesh
// byte-offset blob walk (incl. skipping hdr.w * 1168-byte BV32 nodes), the
// flipNormal swaps, the six voronoi branches, the boundary/find logic with
// its verbatim vestigial guards (isBoundaryEdge on a non-0xFFFFFFFF index;
// isActiveEdge on HF adjacencies), the wrongHanded output permutations, the
// clamped 3x3 HF neighbourhood walk, and the final index &= 0x0FFFFFF clear
// (which also zeroes triangle-index bits 24-26).  All outputs integer-exact.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct F4 { float x, y, z, w; };
struct PostIn
{
    U4 hdr;
    U4 pairsGPU[4];
    U4 pairA, pairB;
    F4 scale;
    U4 niIdx[8];
    U4 ordered[8];
    U4 blob[192];
};
struct PostOut { U4 res, aux; };

const uint32_t EV0 = 1u << 29, EV1 = 2u << 29, EV2 = 3u << 29;
const uint32_t EE01 = 4u << 29, EE12 = 5u << 29, EE02 = 6u << 29;
const uint32_t FACE_FLAG = 0x80000000u, EDGE_FLAG = 0xC0000000u;
const uint32_t BOUNDARY = 0xFFFFFFFFu;
const uint32_t BV32_BYTES = 1168;

static uint32_t& word(U4* a, uint32_t idx) { return (&a[idx / 4].x)[idx & 3]; }
static uint32_t cword(const U4* a, uint32_t idx) { return (&a[idx / 4].x)[idx & 3]; }
static uint32_t blobW(const PostIn& s, uint32_t byteOff) { return cword(s.blob, byteOff / 4); }
static uint32_t getNb(uint32_t index) { return (index & 0x7FFFFFFFu) >> 27; }

static uint32_t bsearchOrd(const PostIn& s, uint32_t base, uint32_t count, uint32_t value)
{
    uint32_t left = 0, right = count;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2;
        if (cword(s.ordered, base + pos) <= value) left = pos + 1;
        else right = pos;
    }
    return left != 0 ? left - 1 : 0;
}
static bool findFlag(const PostIn& s, uint32_t base, uint32_t count, uint32_t target, uint32_t flag)
{
    const uint32_t tgt = target | flag;
    const uint32_t idx = bsearchOrd(s, base, count, tgt);
    return idx < count && cword(s.ordered, base + idx) == tgt;
}
static bool isBnd(uint32_t e) { return e == BOUNDARY; }
static bool isActive(uint32_t e) { return e == BOUNDARY || !(e & 0x80000000u); }

static bool hfShared(const PostIn& s, uint32_t v) { return (cword(s.blob, 2 + v) & 0x800000u) != 0; }
static void hfVerts(const PostIn& s, uint32_t tri, uint32_t cols, uint32_t& i0, uint32_t& i1, uint32_t& i2)
{
    const uint32_t cell = tri >> 1;
    const bool first = (tri & 1) == 0;
    if (hfShared(s, cell))
    {
        if (first) { i0 = cell + cols; i1 = cell; i2 = cell + cols + 1; }
        else       { i0 = cell + 1; i1 = cell + cols + 1; i2 = cell; }
    }
    else
    {
        if (first) { i0 = cell; i1 = cell + 1; i2 = cell + cols; }
        else       { i0 = cell + cols + 1; i1 = cell + cols; i2 = cell + 1; }
    }
}
static void hfAdj(const PostIn& s, uint32_t tri, uint32_t cols, uint32_t rows, uint32_t& a0, uint32_t& a1, uint32_t& a2)
{
    const uint32_t cell = tri >> 1;
    const bool first = (tri & 1) == 0;
    if (hfShared(s, cell))
    {
        if (first)
        {
            a0 = BOUNDARY; a1 = tri + 1; a2 = BOUNDARY;
            if (cell % cols != 0)        a0 = tri - 1;
            if (cell / cols != rows - 2) a2 = ((cell + cols) * 2) + 1;
        }
        else
        {
            a0 = BOUNDARY; a1 = tri - 1; a2 = BOUNDARY;
            if (cell % cols < cols - 2)  a0 = tri + 1;
            if (cell >= cols - 1)        a2 = (cell - cols) * 2;
        }
    }
    else
    {
        if (first)
        {
            a0 = BOUNDARY; a1 = tri + 1; a2 = BOUNDARY;
            if (cell >= cols - 1)        a0 = ((cell - cols) * 2) + 1;
            if (cell % cols != 0)        a2 = tri - 1;
        }
        else
        {
            a0 = BOUNDARY; a1 = tri - 1; a2 = BOUNDARY;
            if (cell / cols != rows - 2) a0 = (cell + cols) * 2;
            if (cell % cols < cols - 2)  a2 = tri + 1;
        }
    }
}

static void refPostProcess(const PostIn& s, PostOut& o)
{
    o.res = U4{ 0, 0, 0xFFFFFFFFu, 0xFFFFFFFFu };
    o.aux = U4{ 0, 0, 0, 0 };

    const uint32_t mode = s.hdr.x;
    const uint32_t wordPM = s.hdr.y;
    const uint32_t pairIdx = wordPM & 0x1FFFFFFFu;
    const U4 curPair = s.pairsGPU[pairIdx & 3];
    const U4 pair = (curPair.x == 0) ? s.pairA : s.pairB;
    const uint32_t nbPairsPerCM = pair.x, pairStartIndex = pair.y, orderedStart = pair.z;
    const uint32_t triangleIdx = curPair.y, testOffset = curPair.z;
    const uint32_t off = pairStartIndex + testOffset;
    o.aux.z = off;

    uint32_t niWord = cword(s.niIdx, off);
    o.res.x = niWord;
    if (getNb(niWord) == 0) { o.aux.x = 1; return; }

    const uint32_t mask = wordPM & 0xE0000000u;
    o.aux.y = mask;

    bool needsProcessing = false;
    uint32_t triangleIndex = 0xFFFFFFFFu, vertexIdx = 0xFFFFFFFFu;

    if (mode == 0)
    {
        const uint32_t numVerts = blobW(s, 0), numTris = blobW(s, 4), numBV32 = blobW(s, 12);
        uint32_t bo = 16 + BV32_BYTES * numBV32 + 16 * numVerts;
        const uint32_t triOff = bo; bo += 16 * numTris;
        const uint32_t adjOff = bo; bo += 16 * numTris;
        bo += 4 * numTris;
        const uint32_t accOff = bo; bo += 4 * (numVerts + 1);
        const uint32_t refsOff = bo;

        uint32_t adjX = blobW(s, adjOff + 16 * triangleIdx);
        uint32_t adjY = blobW(s, adjOff + 16 * triangleIdx + 4);
        uint32_t adjZ = blobW(s, adjOff + 16 * triangleIdx + 8);
        uint32_t indX = blobW(s, triOff + 16 * triangleIdx);
        uint32_t indY = blobW(s, triOff + 16 * triangleIdx + 4);
        uint32_t indZ = blobW(s, triOff + 16 * triangleIdx + 8);

        if (s.scale.x * s.scale.y * s.scale.z < 0.f)
        {
            std::swap(adjX, adjZ);
            std::swap(indY, indZ);
        }

        uint32_t a2x = 0, a2y = 0;
        if (mask == EV0) { vertexIdx = indX; a2x = adjX; a2y = adjZ; }
        else if (mask == EV1) { vertexIdx = indY; a2x = adjX; a2y = adjY; }
        else if (mask == EV2) { vertexIdx = indZ; a2x = adjY; a2y = adjZ; }
        else if (mask == EE01) { triangleIndex = adjX; if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true; }
        else if (mask == EE12) { triangleIndex = adjY; if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true; }
        else if (mask == EE02) { triangleIndex = adjZ; if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true; }

        if (triangleIndex != 0xFFFFFFFFu)
        {
            needsProcessing = isBnd(triangleIndex) || !findFlag(s, orderedStart, nbPairsPerCM, triangleIndex, FACE_FLAG);
        }
        else if (vertexIdx != 0xFFFFFFFFu)
        {
            needsProcessing = true;
            if (!(isBnd(a2x) && isBnd(a2y)))
            {
                const uint32_t startIdx = blobW(s, accOff + 4 * vertexIdx);
                const uint32_t endIdx = blobW(s, accOff + 4 * (vertexIdx + 1));
                for (uint32_t idx = startIdx; idx < endIdx; ++idx)
                {
                    const uint32_t triIdx = blobW(s, refsOff + 4 * idx);
                    if (findFlag(s, orderedStart, nbPairsPerCM, triIdx, FACE_FLAG) ||
                        findFlag(s, orderedStart, nbPairsPerCM, triIdx, EDGE_FLAG))
                    {
                        needsProcessing = false;
                        break;
                    }
                }
            }
        }
    }
    else
    {
        const uint32_t nbRows = s.blob[0].x, nbCols = s.blob[0].y;
        bool wrongHanded = false;
        if (s.scale.z < 0.f) wrongHanded = !wrongHanded;
        if (s.scale.x < 0.f) wrongHanded = !wrongHanded;
        const uint32_t wh = wrongHanded ? 1u : 0u;

        uint32_t tv[3], adj[3];
        hfVerts(s, triangleIdx, nbCols, tv[0], tv[1 + wh], tv[2 - wh]);
        hfAdj(s, triangleIdx, nbCols, nbRows, adj[wh ? 2 : 0], adj[1], adj[wh ? 0 : 2]);

        if (mask == EV0) vertexIdx = tv[0];
        else if (mask == EV1) vertexIdx = tv[1];
        else if (mask == EV2) vertexIdx = tv[2];
        else if (mask == EE01) triangleIndex = adj[0];
        else if (mask == EE12) triangleIndex = adj[1];
        else if (mask == EE02) triangleIndex = adj[2];

        if (triangleIndex != 0xFFFFFFFFu)
        {
            needsProcessing = isActive(triangleIndex) && !findFlag(s, orderedStart, nbPairsPerCM, triangleIndex, FACE_FLAG);
        }
        else if (vertexIdx != 0xFFFFFFFFu)
        {
            const uint32_t cellIdx = triangleIdx / 2;
            const uint32_t column = cellIdx % nbCols, row = cellIdx / nbCols;
            const uint32_t startRow = std::max(row, 1u) - 1, endRow = std::min(row, nbRows - 2) + 1;
            const uint32_t startCol = std::max(column, 1u) - 1, endCol = std::min(column, nbCols - 2) + 1;

            needsProcessing = true;
            for (uint32_t r = startRow; r <= endRow && needsProcessing; ++r)
            {
                for (uint32_t c = startCol; c <= endCol; ++c)
                {
                    const uint32_t cell2 = r * nbCols + c;
                    uint32_t triIdx = cell2 * 2;
                    hfVerts(s, triIdx, nbCols, tv[0], tv[1 + wh], tv[2 - wh]);
                    if (tv[0] == vertexIdx || tv[1] == vertexIdx || tv[2] == vertexIdx)
                        if (findFlag(s, orderedStart, nbPairsPerCM, triIdx, FACE_FLAG)) { needsProcessing = false; break; }
                    triIdx = triIdx + 1;
                    hfVerts(s, triIdx, nbCols, tv[0], tv[1 + wh], tv[2 - wh]);
                    if (tv[0] == vertexIdx || tv[1] == vertexIdx || tv[2] == vertexIdx)
                        if (findFlag(s, orderedStart, nbPairsPerCM, triIdx, FACE_FLAG)) { needsProcessing = false; break; }
                }
            }
        }
    }

    if (!needsProcessing)
        niWord &= 0x0FFFFFFu; // seven F's

    o.res = U4{ niWord, needsProcessing ? 1u : 0u, triangleIndex, vertexIdx };
}

PXG_TEST(npCvxPostProcess_cores)
{
    std::mt19937 rng(157001);
    const uint32_t N = 6000;

    std::vector<PostIn> in(N);
    std::vector<PostOut> ref(N);
    size_t cEarly = 0, cClear = 0, cKeep = 0, cHf = 0, cFlip = 0, cWrong = 0, cVtx = 0, cEdge = 0,
           cBndEdge = 0, cNonConvex = 0, cBv32 = 0, cBadMask = 0, cQuirk24 = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PostIn s{};
        const uint32_t mode = rng() & 1;

        const uint32_t maskSel = rng() % 8; // 1..6 valid; 0 and 7 fall through all branches
        const uint32_t vmask = (maskSel == 7) ? (7u << 29) : (maskSel << 29);
        const uint32_t pairIdx = rng() % 4;
        s.hdr = U4{ mode, vmask | pairIdx, 0, 0 };

        uint32_t numTris = 4 + rng() % 5;   // trimesh: 4..8 triangles
        uint32_t numVerts = 6 + rng() % 7;  // 6..12 verts
        uint32_t nbRows = 3 + rng() % 5, nbCols = 3 + rng() % 5;
        const uint32_t hfTris = 2 * (nbRows - 1) * (nbCols - 1);

        uint32_t triangleIdx;
        if (mode == 0) triangleIdx = rng() % numTris;
        else
        {
            const uint32_t r = rng() % (nbRows - 1), c = rng() % (nbCols - 1);
            triangleIdx = (r * nbCols + c) * 2 + (rng() & 1);
        }

        const uint32_t count = 1 + rng() % 8;
        const uint32_t startIndex = rng() % 16;
        const uint32_t roundedStart = 4 * (rng() % 6);
        const uint32_t testOffset = rng() % count;
        const uint32_t cmIdx = rng() & 1;
        for (int p = 0; p < 4; ++p)
            s.pairsGPU[p] = U4{ uint32_t(rng() & 1), (mode == 0) ? uint32_t(rng() % numTris) : triangleIdx, uint32_t(rng() % count), 0 };
        s.pairsGPU[pairIdx] = U4{ cmIdx, triangleIdx, testOffset, 0 };
        const U4 pairFields = U4{ count, startIndex, roundedStart, 0 };
        const U4 otherFields = U4{ uint32_t(1 + rng() % 8), uint32_t(rng() % 16), uint32_t(4 * (rng() % 6)), 0 };
        s.pairA = (cmIdx == 0) ? pairFields : otherFields;
        s.pairB = (cmIdx == 0) ? otherFields : pairFields;

        for (uint32_t j = 0; j < 32; ++j)
        {
            uint32_t nb = rng() % 16;                     // 0..15 (4-bit field)
            if (rng() % 5 == 0) nb = 0;                   // extra early-outs
            const uint32_t tri27 = rng() & 0x07FFFFFFu;   // bits 24-26 sometimes set -> the &0x0FFFFFF quirk shows
            word(s.niIdx, j) = ((rng() & 1u) << 31) | (nb << 27) | tri27;
        }

        // ordered table: sorted flagged triangle indices for this pair's slice
        {
            std::vector<uint32_t> ent(count);
            const uint32_t triDomain = (mode == 0) ? numTris : hfTris;
            for (uint32_t k = 0; k < count; ++k)
            {
                const uint32_t triIdx = rng() % triDomain;
                const uint32_t flag = (mode == 0 && (rng() & 1)) ? EDGE_FLAG : FACE_FLAG;
                ent[k] = triIdx | flag;
            }
            std::sort(ent.begin(), ent.end());
            for (uint32_t k = 0; k < count; ++k) word(s.ordered, roundedStart + k) = ent[k];
            for (uint32_t k = 0; k < 32; ++k)
                if (k < roundedStart || k >= roundedStart + count) word(s.ordered, k) = rng();
        }

        s.scale = F4{ (rng() % 3 == 0) ? -1.5f : 1.5f, 1.f, (rng() % 3 == 0) ? -2.f : 2.f, 0 };

        if (mode == 0)
        {
            const uint32_t numBV32 = (i % 4 == 0) ? 1u : 0u;
            uint32_t bo = 16 + BV32_BYTES * numBV32;
            word(s.blob, 0) = numVerts; word(s.blob, 1) = numTris; word(s.blob, 2) = rng(); word(s.blob, 3) = numBV32;
            bo += 16 * numVerts; // verts skipped (left zero)
            const uint32_t triOff = bo;
            for (uint32_t tj = 0; tj < numTris; ++tj)
                for (int k = 0; k < 3; ++k)
                    word(s.blob, (triOff + 16 * tj + 4 * k) / 4) = rng() % numVerts;
            bo += 16 * numTris;
            const uint32_t adjOff = bo;
            for (uint32_t tj = 0; tj < numTris; ++tj)
                for (int k = 0; k < 3; ++k)
                {
                    uint32_t a = rng() % numTris;
                    const uint32_t sel = rng() % 5;
                    if (sel == 0) a = BOUNDARY;               // boundary edge
                    else if (sel == 1) a |= 0x80000000u;      // nonconvex-flagged neighbour
                    word(s.blob, (adjOff + 16 * tj + 4 * k) / 4) = a;
                }
            bo += 16 * numTris;
            bo += 4 * numTris; // faceRemap (zeros)
            const uint32_t accOff = bo;
            uint32_t acc = 0;
            for (uint32_t v = 0; v <= numVerts; ++v)
            {
                word(s.blob, (accOff + 4 * v) / 4) = acc;
                if (v < numVerts) acc += rng() % 4; // 0..3 refs per vertex
            }
            bo += 4 * (numVerts + 1);
            for (uint32_t k = 0; k < acc; ++k)
                word(s.blob, (bo + 4 * k) / 4) = rng() % numTris;
            if (numBV32) ++cBv32;
        }
        else
        {
            s.blob[0].x = nbRows; s.blob[0].y = nbCols;
            for (uint32_t v = 0; v < nbRows * nbCols; ++v)
                word(s.blob, 2 + v) = (rng() & 0x00FFFFFFu) | ((rng() & 1u) << 23); // random tess bit
        }

        in[i] = s;
        refPostProcess(s, ref[i]);
        if (ref[i].aux.x) ++cEarly;
        else if (ref[i].res.y == 0) { ++cClear; if ((ref[i].aux.y >> 29) == 0 || (ref[i].aux.y >> 29) == 7) ++cBadMask; }
        else ++cKeep;
        if (!ref[i].aux.x && (cword(in[i].niIdx, ref[i].aux.z) & 0x07000000u) && ref[i].res.y == 0) ++cQuirk24;
        if (mode) ++cHf;
        if (mode == 0 && s.scale.x * s.scale.y * s.scale.z < 0) ++cFlip;
        if (mode == 1 && ((s.scale.x < 0) != (s.scale.z < 0))) ++cWrong;
        if (!ref[i].aux.x && ref[i].res.w != 0xFFFFFFFFu) ++cVtx;
        if (!ref[i].aux.x && ref[i].res.z != 0xFFFFFFFFu) ++cEdge;
        if (!ref[i].aux.x && ref[i].res.z == 0xFFFFFFFFu && ref[i].res.w == 0xFFFFFFFFu &&
            ((ref[i].aux.y >> 29) >= 4 && (ref[i].aux.y >> 29) <= 6)) ++cBndEdge; // edge branch hit a 0xFFFFFFFF neighbour
        if (mode == 0 && !ref[i].aux.x && ref[i].res.z != 0xFFFFFFFFu && (ref[i].res.z & 0x80000000u)) ++cNonConvex;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PostOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct PostParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxPostProcess.spv", "cvxPostProcess",
             { (uint32_t)sizeof(PostIn), (uint32_t)sizeof(PostOut) }, sizeof(PostParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PostOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const U4& rr = ref[i].res; const U4& gr = g[i].res;
        const U4& ra = ref[i].aux; const U4& ga = g[i].aux;
        if (rr.x != gr.x || rr.y != gr.y || rr.z != gr.z || rr.w != gr.w ||
            ra.x != ga.x || ra.y != ga.y || ra.z != ga.z) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "postProcess: %zu mismatches", mm);
    PXG_CHECK_MSG(cEarly > N / 20 && cClear > N / 20 && cKeep > N / 20 && cHf > N / 3 &&
                  cFlip > N / 20 && cWrong > N / 20 && cVtx > N / 10 && cEdge > N / 10 &&
                  cBndEdge > N / 50 && cNonConvex > N / 200 && cBv32 > N / 10 && cBadMask > 0 && cQuirk24 > N / 50,
                  "postProcess: coverage thin (early %zu clear %zu keep %zu hf %zu flip %zu wrong %zu vtx %zu edge %zu bnd %zu ncvx %zu bv32 %zu badMask %zu quirk24 %zu)",
                  cEarly, cClear, cKeep, cHf, cFlip, cWrong, cVtx, cEdge, cBndEdge, cNonConvex, cBv32, cBadMask, cQuirk24);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
