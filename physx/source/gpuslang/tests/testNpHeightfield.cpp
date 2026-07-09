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
    PxgBufferHandle bSamples = createBufferFrom(dev, samples);
    PxgBufferHandle bTri = createBufferFrom(dev, triIdx);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct HfParams { uint32_t numTests, columns, rows, pad; float rowScale, heightScale, columnScale, pad2; }
    prm = { N, columns, rows, 0, rowScale, heightScale, columnScale, 0 };
    Kernel k("npHeightfield.spv", "hfGetTriangle", { 4, 4, (uint32_t)sizeof(HfOut) }, sizeof(HfParams));
    k.dispatch(GROUPS, { bSamples, bTri, bOut }, prm);
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

    dev->destroyBuffer(bSamples); dev->destroyBuffer(bTri); dev->destroyBuffer(bOut);
}

} // namespace
