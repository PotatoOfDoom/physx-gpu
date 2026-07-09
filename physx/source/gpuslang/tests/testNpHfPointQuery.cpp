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

// Verification of npHfPointQuery.slang: hfGridRange (getMinRow/getMaxRow/
// getMinColumn/getMaxColumn) and hfTestLocalPoint (testLocalPoint).  The grid
// range is pure integer floor/ceil+clamp, compared BIT-EXACTLY.  testLocalPoint's
// interpolated height is a few float multiplies (compared with almostEqual); the
// discrete outputs (inside/outside decision and the h>y result) are gated on
// float comparisons, so measure-zero cases where a decision variable sits within
// MARG of its boundary are skipped (a GPU-vs-CPU flip there is not a bug).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

static const float MARG = 1e-3f;

static float refHeight(uint32_t word) { return float((int16_t)(word & 0xffffu)); }

static uint32_t refMinRow(float x, uint32_t rows)    { return (uint32_t)std::max(0, std::min((int)std::floor(x), (int)rows - 2)); }
static uint32_t refMaxRow(float x, uint32_t rows)    { return (uint32_t)std::max(0, std::min((int)std::ceil(x),  (int)rows - 1)); }
static uint32_t refMinCol(float z, uint32_t cols)    { return (uint32_t)std::max(0, std::min((int)std::floor(z), (int)cols - 2)); }
static uint32_t refMaxCol(float z, uint32_t cols)    { return (uint32_t)std::max(0, std::min((int)std::ceil(z),  (int)cols - 1)); }

struct HfqParams { uint32_t numTests, rows, columns, pad; };

static void buildGrid(std::mt19937& rng, uint32_t NV, std::vector<uint32_t>& samples)
{
    samples.resize(NV);
    for (uint32_t i = 0; i < NV; ++i)
    {
        const int16_t h = (int16_t)((int)(rng() % 1001) - 500); // -500..500
        samples[i] = (uint32_t(uint16_t(h))) | (rng() << 16); // material bytes irrelevant here
    }
}

PXG_TEST(npHfPointQuery_gridRange)
{
    std::mt19937 rng(62011);
    const uint32_t rows = 50, columns = 40;
    std::vector<uint32_t> samples; buildGrid(rng, rows * columns, samples);

    const uint32_t N = 10000;
    std::vector<F4> pts(N);
    std::vector<U4> ref(N);
    std::uniform_real_distribution<float> ux(-2.0f, float(rows) + 1.0f);
    std::uniform_real_distribution<float> uz(-2.0f, float(columns) + 1.0f);
    size_t clampLo = 0, clampHi = 0, mid = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const float x = ux(rng), z = uz(rng);
        pts[t] = { x, 0.0f, z, 0.0f };
        ref[t] = { refMinRow(x, rows), refMaxRow(x, rows), refMinCol(z, columns), refMaxCol(z, columns) };
        if (x < 0.0f || z < 0.0f) ++clampLo;
        else if (x > float(rows - 1) || z > float(columns - 1)) ++clampHi;
        else ++mid;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<U4> rangeInit(N);
    std::vector<F4> hhDummy(N);
    PxgBufferHandle bSamples = createBufferFrom(dev, samples);
    PxgBufferHandle bPts = createBufferFrom(dev, pts);
    PxgBufferHandle bRange = createBufferFrom(dev, rangeInit);
    PxgBufferHandle bHH = createBufferFrom(dev, hhDummy);

    HfqParams prm = { N, rows, columns, 0 };
    Kernel k("npHfPointQuery.spv", "hfGridRange",
             { 4, (uint32_t)sizeof(F4), (uint32_t)sizeof(U4), (uint32_t)sizeof(F4) }, sizeof(HfqParams));
    k.dispatch(GROUPS, { bSamples, bPts, bRange, bHH }, prm);
    dev->flush();

    const auto g = downloadBuffer<U4>(dev, bRange, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const U4& r = ref[t]; const U4& v = g[t];
        if (r.x != v.x || r.y != v.y || r.z != v.z || r.w != v.w) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "hfGridRange: %zu range mismatches", mm);
    PXG_CHECK_MSG(clampLo > 300 && clampHi > 300 && mid > N / 4,
                  "hfGridRange: clamp coverage thin (lo=%zu hi=%zu mid=%zu)", clampLo, clampHi, mid);

    dev->destroyBuffer(bSamples); dev->destroyBuffer(bPts); dev->destroyBuffer(bRange); dev->destroyBuffer(bHH);
}

PXG_TEST(npHfPointQuery_testLocalPoint)
{
    std::mt19937 rng(62029);
    const uint32_t rows = 50, columns = 40;
    std::vector<uint32_t> samples; buildGrid(rng, rows * columns, samples);

    const uint32_t N = 12000;
    std::vector<F4> pts(N);
    std::uniform_real_distribution<float> ux(-2.0f, float(rows) + 1.0f);
    std::uniform_real_distribution<float> uz(-2.0f, float(columns) + 1.0f);
    std::uniform_real_distribution<float> uy(-600.0f, 600.0f);
    for (uint32_t t = 0; t < N; ++t)
        pts[t] = { ux(rng), uy(rng), uz(rng), 0.0f };

    // CPU reference: inside flag, height, result
    std::vector<uint8_t> refInside(N), refResult(N), skip(N, 0);
    std::vector<float> refH(N, 0.0f);
    size_t insideC = 0, outsideC = 0, hitC = 0, missC = 0, skipped = 0;
    const float rowsM1 = float(rows - 1), colsM1 = float(columns - 1);
    for (uint32_t t = 0; t < N; ++t)
    {
        const float x = pts[t].x, y = pts[t].y, z = pts[t].z;
        // skip near the inside/outside decision boundaries (measure-zero flips)
        if (std::fabs(x) < MARG || std::fabs(x - rowsM1) < MARG ||
            std::fabs(z) < MARG || std::fabs(z - colsM1) < MARG)
        {
            skip[t] = 1; ++skipped; continue;
        }
        const bool inside = !(x < 0.0f || x > rowsM1 || z < 0.0f || z > colsM1);
        refInside[t] = inside ? 1 : 0;
        if (!inside) { refResult[t] = 0; ++outsideC; continue; }
        ++insideC;
        const uint32_t row = refMinRow(x, rows), col = refMinCol(z, columns);
        const uint32_t vtx = row * columns + col;
        const float h0 = refHeight(samples[vtx]);
        const float h1 = refHeight(samples[vtx + 1]);
        const float h2 = refHeight(samples[vtx + columns]);
        const float h3 = refHeight(samples[vtx + columns + 1]);
        const float s = x - std::floor(x);
        const float tt = z - std::floor(z);
        const float h = h0 * (1 - s) * (1 - tt) + h1 * s * (1 - tt) + h2 * (1 - s) * tt + h3 * s * tt;
        refH[t] = h;
        // skip near the h>y result boundary; widened past the height tolerance
        // so a worst-case FMA-vs-non-FMA divergence in h cannot flip the result
        if (std::fabs(h - y) < 2.0f * MARG) { skip[t] = 1; ++skipped; --insideC; continue; }
        refResult[t] = (h > y) ? 1 : 0;
        if (refResult[t]) ++hitC; else ++missC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<U4> rangeDummy(N);
    std::vector<F4> hhInit(N);
    PxgBufferHandle bSamples = createBufferFrom(dev, samples);
    PxgBufferHandle bPts = createBufferFrom(dev, pts);
    PxgBufferHandle bRange = createBufferFrom(dev, rangeDummy);
    PxgBufferHandle bHH = createBufferFrom(dev, hhInit);

    HfqParams prm = { N, rows, columns, 0 };
    Kernel k("npHfPointQuery.spv", "hfTestLocalPoint",
             { 4, (uint32_t)sizeof(F4), (uint32_t)sizeof(U4), (uint32_t)sizeof(F4) }, sizeof(HfqParams));
    k.dispatch(GROUPS, { bSamples, bPts, bRange, bHH }, prm);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bHH, N);
    size_t mmInside = 0, mmResult = 0, mmHeight = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) continue;
        const F4& v = g[t];
        const uint8_t gInside = (v.z != 0.0f) ? 1 : 0;
        const uint8_t gResult = (v.y != 0.0f) ? 1 : 0;
        if (gInside != refInside[t]) { ++mmInside; continue; }
        if (gResult != refResult[t]) { ++mmResult; continue; }
        if (refInside[t] && !almostEqual(v.x, refH[t], 1e-4f, 1e-3f)) ++mmHeight;
    }
    PXG_CHECK_MSG(mmInside == 0, "hfTestLocalPoint: %zu inside-flag mismatches", mmInside);
    PXG_CHECK_MSG(mmResult == 0, "hfTestLocalPoint: %zu result mismatches", mmResult);
    PXG_CHECK_MSG(mmHeight == 0, "hfTestLocalPoint: %zu height mismatches", mmHeight);
    PXG_CHECK_MSG(insideC > N / 4 && outsideC > N / 8 && hitC > N / 8 && missC > N / 8,
                  "hfTestLocalPoint: coverage thin (in=%zu out=%zu hit=%zu miss=%zu skip=%zu)",
                  insideC, outsideC, hitC, missC, skipped);

    dev->destroyBuffer(bSamples); dev->destroyBuffer(bPts); dev->destroyBuffer(bRange); dev->destroyBuffer(bHH);
}

} // namespace
