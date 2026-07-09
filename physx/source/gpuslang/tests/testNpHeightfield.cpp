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

// Verification of npHeightfield.slang: hfGetTriangle — the heightfield
// triangle-extraction core (getTriangleVertexIndices + getVertex + hf2shapep).
// The three vertex indices and the three shape-space vertices are computed by a
// C++ transcription of the CUDA and compared BIT-EXACTLY: the indices are pure
// integer arithmetic and the vertices are exact integer/i16 -> float values
// scaled by a single IEEE multiply, so no tolerance is required.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct HfOut { U4 idx; F4 v0, v1, v2; };

static float refHeight(uint32_t word)
{
    const int16_t h = (int16_t)(word & 0xffffu);
    return float(h);
}
static void refVertex(uint32_t idx, uint32_t columns, const std::vector<uint32_t>& samples, float& x, float& y, float& z)
{
    const uint32_t row = idx / columns, col = idx % columns;
    x = float(row); y = refHeight(samples[idx]); z = float(col);
}

PXG_TEST(npHeightfield_getTriangle)
{
    std::mt19937 rng(58010);
    const uint32_t rows = 64, columns = 48;
    const float rowScale = 0.5f, heightScale = 0.01f, columnScale = 0.75f;
    const uint32_t NV = rows * columns;

    // build a random sample grid: i16 height, random materialIndex0 (tess flag = 0x80)
    std::vector<uint32_t> samples(NV);
    for (uint32_t i = 0; i < NV; ++i)
    {
        const int16_t h = (int16_t)(rng() % 65536);
        const uint32_t m0 = rng() % 256; // includes the 0x80 tess bit ~50%
        const uint32_t m1 = rng() % 256;
        samples[i] = (uint32_t(uint16_t(h))) | (m0 << 16) | (m1 << 24);
    }

    const uint32_t N = 12000;
    std::vector<uint32_t> triIdx(N);
    std::vector<HfOut> ref(N);
    size_t sharedC = 0, unsharedC = 0, firstC = 0, secondC = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        // pick a valid quad corner (row < rows-1, col < columns-1) and a sub-triangle
        const uint32_t crow = rng() % (rows - 1), ccol = rng() % (columns - 1);
        const uint32_t cell = crow * columns + ccol;
        const uint32_t first = rng() & 1u;
        const uint32_t tri = cell * 2u + first;
        triIdx[t] = tri;

        const bool shared = (samples[cell] & 0x800000u) != 0u;
        const bool firstTri = (tri & 1u) == 0u;
        uint32_t i0, i1, i2;
        if (shared)
        {
            if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1; }
            else          { i0 = cell + 1; i1 = cell + columns + 1; i2 = cell; }
        }
        else
        {
            if (firstTri) { i0 = cell; i1 = cell + 1; i2 = cell + columns; }
            else          { i0 = cell + columns + 1; i1 = cell + columns; i2 = cell + 1; }
        }
        HfOut o{}; o.idx = { i0, i1, i2, 0 };
        float x, y, z;
        refVertex(i0, columns, samples, x, y, z); o.v0 = { x * rowScale, y * heightScale, z * columnScale, 0 };
        refVertex(i1, columns, samples, x, y, z); o.v1 = { x * rowScale, y * heightScale, z * columnScale, 0 };
        refVertex(i2, columns, samples, x, y, z); o.v2 = { x * rowScale, y * heightScale, z * columnScale, 0 };
        ref[t] = o;
        if (shared) ++sharedC; else ++unsharedC;
        if (firstTri) ++firstC; else ++secondC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<HfOut> outInit(N);
    std::vector<U4> adjInit(N);
    PxgBufferHandle bSamples = createBufferFrom(dev, samples);
    PxgBufferHandle bTri = createBufferFrom(dev, triIdx);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bAdj = createBufferFrom(dev, adjInit);

    struct HfParams { uint32_t numTests, columns, rows, pad; float rowScale, heightScale, columnScale, pad2; }
    prm = { N, columns, rows, 0, rowScale, heightScale, columnScale, 0 };
    Kernel k("npHeightfield.spv", "hfGetTriangle", { 4, 4, (uint32_t)sizeof(HfOut), (uint32_t)sizeof(U4) }, sizeof(HfParams));
    k.dispatch(GROUPS, { bSamples, bTri, bOut, bAdj }, prm);
    dev->flush();

    const auto g = downloadBuffer<HfOut>(dev, bOut, N);
    auto eqV = [](const F4& a, const F4& b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
    size_t mmIdx = 0, mmVert = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const HfOut& r = ref[t]; const HfOut& v = g[t];
        if (r.idx.x != v.idx.x || r.idx.y != v.idx.y || r.idx.z != v.idx.z) { ++mmIdx; continue; }
        if (!eqV(r.v0, v.v0) || !eqV(r.v1, v.v1) || !eqV(r.v2, v.v2)) ++mmVert;
    }
    PXG_CHECK_MSG(mmIdx == 0, "hfGetTriangle: %zu vertex-index mismatches", mmIdx);
    PXG_CHECK_MSG(mmVert == 0, "hfGetTriangle: %zu vertex-value mismatches", mmVert);
    PXG_CHECK_MSG(sharedC > N / 4 && unsharedC > N / 4 && firstC > N / 4 && secondC > N / 4,
                  "hfGetTriangle: branch coverage thin (shared=%zu/%zu first=%zu/%zu)", sharedC, unsharedC, firstC, secondC);

    dev->destroyBuffer(bSamples); dev->destroyBuffer(bTri); dev->destroyBuffer(bOut); dev->destroyBuffer(bAdj);
}

// ---------------------------------------------------------------------------
// hfGetAdjacency: the three edge-adjacent triangle indices (getTriangleAdjacencyIndices).
// Pure integer arithmetic with a BOUNDARY sentinel for grid edges — compared
// BIT-EXACTLY.  The grid is chosen so all four diagonal/parity patterns and all
// four grid-edge conditionals (left/right column, top/bottom row) are exercised.
// ---------------------------------------------------------------------------
static const uint32_t HF_BOUNDARY = 0xffffffffu;

PXG_TEST(npHeightfield_getAdjacency)
{
    std::mt19937 rng(59021);
    const uint32_t rows = 64, columns = 48;
    const float rowScale = 0.5f, heightScale = 0.01f, columnScale = 0.75f;
    const uint32_t NV = rows * columns;

    std::vector<uint32_t> samples(NV);
    for (uint32_t i = 0; i < NV; ++i)
    {
        const int16_t h = (int16_t)(rng() % 65536);
        const uint32_t m0 = rng() % 256; // tess flag = 0x80 ~50%
        const uint32_t m1 = rng() % 256;
        samples[i] = (uint32_t(uint16_t(h))) | (m0 << 16) | (m1 << 24);
    }

    const uint32_t N = 12000;
    std::vector<uint32_t> triIdx(N);
    std::vector<U4> ref(N);
    size_t sharedC = 0, unsharedC = 0, firstC = 0, secondC = 0;
    // coverage of the four grid-edge conditionals actually taking the non-BOUNDARY branch
    size_t edgeLeft = 0, edgeRight = 0, edgeTop = 0, edgeBottom = 0, boundaryHits = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t crow = rng() % (rows - 1), ccol = rng() % (columns - 1);
        const uint32_t cell = crow * columns + ccol;
        const uint32_t first = rng() & 1u;
        const uint32_t tri = cell * 2u + first;
        triIdx[t] = tri;

        const bool shared = (samples[cell] & 0x800000u) != 0u;
        const bool firstTri = (tri & 1u) == 0u;
        uint32_t a0, a1, a2;
        if (shared)
        {
            if (firstTri)
            {
                a0 = HF_BOUNDARY; a1 = tri + 1; a2 = HF_BOUNDARY;
                if (cell % columns != 0) a0 = tri - 1;
                if (cell / columns != rows - 2) a2 = ((cell + columns) * 2) + 1;
            }
            else
            {
                a0 = HF_BOUNDARY; a1 = tri - 1; a2 = HF_BOUNDARY;
                if (cell % columns < columns - 2) a0 = tri + 1;
                if (cell >= columns - 1) a2 = (cell - columns) * 2;
            }
        }
        else
        {
            if (firstTri)
            {
                a0 = HF_BOUNDARY; a1 = tri + 1; a2 = HF_BOUNDARY;
                if (cell >= columns - 1) a0 = ((cell - columns) * 2) + 1;
                if (cell % columns != 0) a2 = tri - 1;
            }
            else
            {
                a0 = HF_BOUNDARY; a1 = tri - 1; a2 = HF_BOUNDARY;
                if (cell / columns != rows - 2) a0 = (cell + columns) * 2;
                if (cell % columns < columns - 2) a2 = tri + 1;
            }
        }
        ref[t] = { a0, a1, a2, 0 };
        if (shared) ++sharedC; else ++unsharedC;
        if (firstTri) ++firstC; else ++secondC;
        if (cell % columns == 0) ++edgeLeft;
        if (cell % columns == columns - 2) ++edgeRight;
        if (cell / columns == 0) ++edgeTop;
        if (cell / columns == rows - 2) ++edgeBottom;
        if (a0 == HF_BOUNDARY || a1 == HF_BOUNDARY || a2 == HF_BOUNDARY) ++boundaryHits;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<HfOut> outInit(N);
    std::vector<U4> adjInit(N);
    PxgBufferHandle bSamples = createBufferFrom(dev, samples);
    PxgBufferHandle bTri = createBufferFrom(dev, triIdx);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bAdj = createBufferFrom(dev, adjInit);

    struct HfParams { uint32_t numTests, columns, rows, pad; float rowScale, heightScale, columnScale, pad2; }
    prm = { N, columns, rows, 0, rowScale, heightScale, columnScale, 0 };
    Kernel k("npHeightfield.spv", "hfGetAdjacency", { 4, 4, (uint32_t)sizeof(HfOut), (uint32_t)sizeof(U4) }, sizeof(HfParams));
    k.dispatch(GROUPS, { bSamples, bTri, bOut, bAdj }, prm);
    dev->flush();

    const auto g = downloadBuffer<U4>(dev, bAdj, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const U4& r = ref[t]; const U4& v = g[t];
        if (r.x != v.x || r.y != v.y || r.z != v.z) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "hfGetAdjacency: %zu adjacency mismatches", mm);
    PXG_CHECK_MSG(sharedC > N / 4 && unsharedC > N / 4 && firstC > N / 4 && secondC > N / 4,
                  "hfGetAdjacency: diagonal/parity coverage thin (shared=%zu first=%zu)", sharedC, firstC);
    PXG_CHECK_MSG(edgeLeft > 20 && edgeRight > 20 && edgeTop > 20 && edgeBottom > 20 && boundaryHits > 20,
                  "hfGetAdjacency: grid-edge coverage thin (L=%zu R=%zu T=%zu B=%zu bnd=%zu)",
                  edgeLeft, edgeRight, edgeTop, edgeBottom, boundaryHits);

    dev->destroyBuffer(bSamples); dev->destroyBuffer(bTri); dev->destroyBuffer(bOut); dev->destroyBuffer(bAdj);
}

} // namespace
