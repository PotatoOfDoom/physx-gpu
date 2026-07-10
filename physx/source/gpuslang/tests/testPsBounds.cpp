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

// Verification of psBounds.slang (ps_primitivesBoundFirstPass/SecondPass,
// ps_reorderPrimitiveContacts and the calcGridHashInBounds helper from
// cudaParticleSystem.cu).  All outputs are integers, so verification is
// BIT-EXACT: bounds are SANITIZED at generation time (every coordinate of
// the combined-and-fattened overlap box stays > 1e-3 cells away from a cell
// boundary, so the floor() grid coordinates are deterministic on both
// sides — the same safeCoord discipline as testGridCal.cpp; this must hold
// for every row since the scan couples all rows).  Disjoint pairs produce
// inverted overlap boxes whose clamped range size is 0 (counted as
// zero-cell rows).  The two-pass scan is checked against a straight
// exclusive scan in work-index order, the per-block totals against the
// chunk sums implied by the kernel's block/tile mapping, and the reorder
// gather bit-exactly under two independent permutations.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct PbWork { F4 b0min, b0max, b1min, b1max; U4 grid; };
struct PsContact { F4 normal_pen; uint32_t rigidLo, rigidHi, particleLo, particleHi; };
struct PbParams { uint32_t numTests, pad0, pad1, pad2; };

static const uint32_t BLOCK = 512, NUM_BLOCKS = 32;

struct V3 { float x, y, z; };

static int32_t floorDiv(float v, float cw) { return (int32_t)std::floor(v / cw); }

static bool safeCoord(float v, float cw)
{
    const float q = v / cw;
    return std::fabs(q - std::round(q)) > 1e-3f;
}

static uint32_t refCellCount(const PbWork& w)
{
    const float cw = w.b0min.w, cd = w.b0max.w;
    float mn[3] = { std::max(w.b0min.x, w.b1min.x), std::max(w.b0min.y, w.b1min.y), std::max(w.b0min.z, w.b1min.z) };
    float mx[3] = { std::min(w.b0max.x, w.b1max.x), std::min(w.b0max.y, w.b1max.y), std::min(w.b0max.z, w.b1max.z) };
    uint32_t count = 1;
    for (int k = 0; k < 3; ++k)
    {
        mn[k] -= cd;
        mx[k] += cd;
        const int32_t gmin = floorDiv(mn[k], cw), gmax = floorDiv(mx[k], cw);
        const int32_t gs = (int32_t)(&w.grid.x)[k];
        const int32_t r = std::min(std::max(gmax - gmin + 1, 0), gs);
        count *= (uint32_t)r;
    }
    return count;
}

PXG_TEST(psBounds_scanAndReorder)
{
    std::mt19937 rng(88007);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    // NT spans 63 required blocks -> 2 tiles per device block, a partial
    // tail AND a non-empty LAST device block (pins the last term of the
    // second pass's total)
    const uint32_t NT = 32000;
    std::vector<PbWork> work(NT);
    const uint32_t gsizes[3] = { 8, 16, 32 };
    for (uint32_t i = 0; i < NT; ++i)
    {
        PbWork& w = work[i];
        const float cw = 0.3f + 0.7f * u01(rng);
        const float cd = 0.1f + 0.3f * u01(rng);
        w.grid = { gsizes[rng() % 3], gsizes[rng() % 3], gsizes[rng() % 3], 0 };
        for (uint32_t attempt = 0;; ++attempt)
        {
            const bool disjoint = (i % 7 == 5);
            V3 c0 = { 8.0f * uv(rng), 8.0f * uv(rng), 8.0f * uv(rng) };
            V3 h0 = { 0.3f + 1.8f * u01(rng), 0.3f + 1.8f * u01(rng), 0.3f + 1.8f * u01(rng) };
            V3 c1 = { c0.x + 1.5f * uv(rng), c0.y + 1.5f * uv(rng), c0.z + 1.5f * uv(rng) };
            if (disjoint) c1.x = c0.x + h0.x + 4.0f + 2.0f * u01(rng); // x-gap > fatten reach + max cell width
            V3 h1 = { 0.3f + 1.8f * u01(rng), 0.3f + 1.8f * u01(rng), 0.3f + 1.8f * u01(rng) };
            w.b0min = { c0.x - h0.x, c0.y - h0.y, c0.z - h0.z, cw };
            w.b0max = { c0.x + h0.x, c0.y + h0.y, c0.z + h0.z, cd };
            w.b1min = { c1.x - h1.x, c1.y - h1.y, c1.z - h1.z, 0 };
            w.b1max = { c1.x + h1.x, c1.y + h1.y, c1.z + h1.z, 0 };
            // sanitize the fattened overlap coordinates
            bool ok = true;
            const float mn[3] = { std::max(w.b0min.x, w.b1min.x) - cd, std::max(w.b0min.y, w.b1min.y) - cd, std::max(w.b0min.z, w.b1min.z) - cd };
            const float mx[3] = { std::min(w.b0max.x, w.b1max.x) + cd, std::min(w.b0max.y, w.b1max.y) + cd, std::min(w.b0max.z, w.b1max.z) + cd };
            for (int k = 0; k < 3; ++k)
                ok = ok && safeCoord(mn[k], cw) && safeCoord(mx[k], cw);
            if (ok) break;
            PXG_CHECK(attempt < 200);
        }
    }

    // reference: counts, straight exclusive scan, per-chunk block totals
    std::vector<uint32_t> counts(NT), refOffsets(NT);
    uint32_t total = 0;
    size_t zeroRows = 0, bigRows = 0, clampedRows = 0;
    for (uint32_t i = 0; i < NT; ++i)
    {
        counts[i] = refCellCount(work[i]);
        refOffsets[i] = total;
        total += counts[i];
        if (counts[i] == 0) ++zeroRows;
        if (counts[i] > 200) ++bigRows;
        // upper-clamp coverage: some axis spans more cells than the grid
        const PbWork& w = work[i];
        const float cw = w.b0min.w, cd = w.b0max.w;
        for (int k = 0; k < 3; ++k)
        {
            const float mnk = std::max((&w.b0min.x)[k], (&w.b1min.x)[k]) - cd;
            const float mxk = std::min((&w.b0max.x)[k], (&w.b1max.x)[k]) + cd;
            if (floorDiv(mxk, cw) - floorDiv(mnk, cw) + 1 > (int32_t)(&w.grid.x)[k]) { ++clampedRows; break; }
        }
    }
    const uint32_t nbBlocksRequired = (NT + BLOCK - 1) / BLOCK;
    const uint32_t iters = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;
    std::vector<uint32_t> refBlock(NUM_BLOCKS, 0);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
        for (uint32_t j = b * iters * BLOCK; j < std::min((b + 1) * iters * BLOCK, NT); ++j)
            refBlock[b] += counts[j];

    // ---- GPU: pass 1 + pass 2 ----
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> zerosNT(NT, 0), zeros32(NUM_BLOCKS, 0), zero1(1, 0);
    // dummy buffers for the reorder bindings
    std::vector<PsContact> dummyC(1);
    std::vector<uint32_t> dummyU(1, 0);

    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bO = createBufferFrom(dev, zerosNT);
    PxgBufferHandle bB = createBufferFrom(dev, zeros32);
    PxgBufferHandle bT = createBufferFrom(dev, zero1);
    PxgBufferHandle bCin = createBufferFrom(dev, dummyC);
    PxgBufferHandle bRA = createBufferFrom(dev, dummyU);
    PxgBufferHandle bRB = createBufferFrom(dev, dummyU);
    PxgBufferHandle bSA = createBufferFrom(dev, dummyC);
    PxgBufferHandle bSB = createBufferFrom(dev, dummyC);

    const std::vector<uint32_t> strides = { (uint32_t)sizeof(PbWork), 4, 4, 4,
                                            (uint32_t)sizeof(PsContact), 4, 4,
                                            (uint32_t)sizeof(PsContact), (uint32_t)sizeof(PsContact) };
    PbParams prm = { NT, 0, 0, 0 };
    Kernel k1("psBounds.spv", "ctPsBoundFirstPass", strides, sizeof(PbParams));
    k1.dispatch(NUM_BLOCKS, { bW, bO, bB, bT, bCin, bRA, bRB, bSA, bSB }, prm);
    dev->flush();
    const auto gBlocks = downloadBuffer<uint32_t>(dev, bB, NUM_BLOCKS);
    size_t mmB = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
        if (gBlocks[b] != refBlock[b]) ++mmB;
    PXG_CHECK_MSG(mmB == 0, "psBounds: %zu block-total mismatches", mmB);

    Kernel k2("psBounds.spv", "ctPsBoundSecondPass", strides, sizeof(PbParams));
    k2.dispatch(NUM_BLOCKS, { bW, bO, bB, bT, bCin, bRA, bRB, bSA, bSB }, prm);
    dev->flush();

    const auto gOff = downloadBuffer<uint32_t>(dev, bO, NT);
    const uint32_t gTotal = downloadBuffer<uint32_t>(dev, bT, 1)[0];
    size_t mm = 0;
    for (uint32_t i = 0; i < NT; ++i)
        if (gOff[i] != refOffsets[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "psBounds: %zu offset mismatches", mm);
    PXG_CHECK_MSG(gTotal == total, "psBounds: total %u != %u", gTotal, total);
    PXG_CHECK_MSG(zeroRows > 1500 && bigRows > 200 && clampedRows > 0 && total > 100000,
                  "psBounds: coverage thin (zero=%zu big=%zu clamp=%zu total=%u)", zeroRows, bigRows, clampedRows, total);
    dev->destroyBuffer(bW); dev->destroyBuffer(bO); dev->destroyBuffer(bB); dev->destroyBuffer(bT);
    dev->destroyBuffer(bCin); dev->destroyBuffer(bRA); dev->destroyBuffer(bRB);
    dev->destroyBuffer(bSA); dev->destroyBuffer(bSB);

    // ---- reorder gather ----
    {
        const uint32_t NC = 700;
        std::vector<PsContact> contacts(NC);
        for (uint32_t i = 0; i < NC; ++i)
        {
            contacts[i].normal_pen = { uv(rng), uv(rng), uv(rng), uv(rng) };
            contacts[i].rigidLo = rng(); contacts[i].rigidHi = rng();
            contacts[i].particleLo = rng(); contacts[i].particleHi = rng();
        }
        std::vector<uint32_t> remapA(NC), remapB(NC);
        std::iota(remapA.begin(), remapA.end(), 0);
        std::iota(remapB.begin(), remapB.end(), 0);
        std::shuffle(remapA.begin(), remapA.end(), rng);
        std::shuffle(remapB.begin(), remapB.end(), rng);

        std::vector<PbWork> dummyW(1);
        std::vector<uint32_t> dummyO(1, 0), blocks32(NUM_BLOCKS, 0);
        std::vector<uint32_t> totalBuf(1, NC);
        std::vector<PsContact> outA(NC), outB(NC);
        PxgBufferHandle cW = createBufferFrom(dev, dummyW);
        PxgBufferHandle cO = createBufferFrom(dev, dummyO);
        PxgBufferHandle cB = createBufferFrom(dev, blocks32);
        PxgBufferHandle cT = createBufferFrom(dev, totalBuf);
        PxgBufferHandle cCin = createBufferFrom(dev, contacts);
        PxgBufferHandle cRA = createBufferFrom(dev, remapA);
        PxgBufferHandle cRB = createBufferFrom(dev, remapB);
        PxgBufferHandle cSA = createBufferFrom(dev, outA);
        PxgBufferHandle cSB = createBufferFrom(dev, outB);
        PbParams prm2 = { NC, 0, 0, 0 };
        Kernel k3("psBounds.spv", "ctPsReorderContacts", strides, sizeof(PbParams));
        k3.dispatch(32, { cW, cO, cB, cT, cCin, cRA, cRB, cSA, cSB }, prm2);
        dev->flush();
        const auto gA = downloadBuffer<PsContact>(dev, cSA, NC);
        const auto gB = downloadBuffer<PsContact>(dev, cSB, NC);
        size_t mmR = 0;
        for (uint32_t i = 0; i < NC; ++i)
        {
            if (std::memcmp(&gA[i], &contacts[remapA[i]], sizeof(PsContact)) != 0) ++mmR;
            if (std::memcmp(&gB[i], &contacts[remapB[i]], sizeof(PsContact)) != 0) ++mmR;
        }
        PXG_CHECK_MSG(mmR == 0, "psBounds: %zu reorder mismatches", mmR);
        dev->destroyBuffer(cW); dev->destroyBuffer(cO); dev->destroyBuffer(cB); dev->destroyBuffer(cT);
        dev->destroyBuffer(cCin); dev->destroyBuffer(cRA); dev->destroyBuffer(cRB);
        dev->destroyBuffer(cSA); dev->destroyBuffer(cSB);
    }

    // ---- calcGridHashInBounds helper ----
    {
        const uint32_t NH = 2048;
        std::vector<PbWork> hw(NH);
        std::vector<uint32_t> refHash(NH);
        for (uint32_t i = 0; i < NH; ++i)
        {
            PbWork& w = hw[i];
            const float cw = 0.3f + 0.7f * u01(rng);
            const float cd = 0.1f + 0.3f * u01(rng); // fatten is part of the composition
            w.grid = { gsizes[rng() % 3], gsizes[rng() % 3], gsizes[rng() % 3], 0 };
            uint32_t rs[3];
            for (;;)
            {
                const V3 c0 = { 8.0f * uv(rng), 8.0f * uv(rng), 8.0f * uv(rng) };
                const V3 h0 = { 0.3f + 1.5f * u01(rng), 0.3f + 1.5f * u01(rng), 0.3f + 1.5f * u01(rng) };
                const V3 c1 = { c0.x + 0.8f * uv(rng), c0.y + 0.8f * uv(rng), c0.z + 0.8f * uv(rng) };
                const V3 h1 = { 0.3f + 1.5f * u01(rng), 0.3f + 1.5f * u01(rng), 0.3f + 1.5f * u01(rng) };
                w.b0min = { c0.x - h0.x, c0.y - h0.y, c0.z - h0.z, cw };
                w.b0max = { c0.x + h0.x, c0.y + h0.y, c0.z + h0.z, cd };
                w.b1min = { c1.x - h1.x, c1.y - h1.y, c1.z - h1.z, 0 };
                w.b1max = { c1.x + h1.x, c1.y + h1.y, c1.z + h1.z, 0 };
                const float mn[3] = { std::max(w.b0min.x, w.b1min.x) - cd, std::max(w.b0min.y, w.b1min.y) - cd, std::max(w.b0min.z, w.b1min.z) - cd };
                const float mx[3] = { std::min(w.b0max.x, w.b1max.x) + cd, std::min(w.b0max.y, w.b1max.y) + cd, std::min(w.b0max.z, w.b1max.z) + cd };
                bool ok = true;
                for (int kk = 0; kk < 3; ++kk)
                    ok = ok && safeCoord(mn[kk], cw) && safeCoord(mx[kk], cw) && mn[kk] < mx[kk];
                if (!ok) continue;
                bool nonEmpty = true;
                for (int kk = 0; kk < 3; ++kk)
                {
                    const int32_t gmin = floorDiv(mn[kk], cw), gmax = floorDiv(mx[kk], cw);
                    const int32_t gs = (int32_t)(&w.grid.x)[kk];
                    const int32_t r = std::min(std::max(gmax - gmin + 1, 0), gs);
                    rs[kk] = (uint32_t)r;
                    nonEmpty = nonEmpty && r > 0;
                }
                if (nonEmpty) break;
            }
            const uint32_t cellOffset = rng() % (rs[0] * rs[1] * rs[2]);
            w.b1min.w = (float)cellOffset;
            // reference hash (fattened, matching the kernel)
            const float mn[3] = { std::max(w.b0min.x, w.b1min.x) - cd, std::max(w.b0min.y, w.b1min.y) - cd, std::max(w.b0min.z, w.b1min.z) - cd };
            int32_t gmin[3];
            for (int kk = 0; kk < 3; ++kk) gmin[kk] = floorDiv(mn[kk], cw);
            const uint32_t ox = cellOffset % rs[0];
            const uint32_t oy = (cellOffset / rs[0]) % rs[1];
            const uint32_t oz = cellOffset / (rs[0] * rs[1]);
            const int32_t gx = (gmin[0] + (int32_t)ox) & (int32_t)(w.grid.x - 1);
            const int32_t gy = (gmin[1] + (int32_t)oy) & (int32_t)(w.grid.y - 1);
            const int32_t gz = (gmin[2] + (int32_t)oz) & (int32_t)(w.grid.z - 1);
            refHash[i] = (uint32_t)((gz * (int32_t)w.grid.y) * (int32_t)w.grid.x + gy * (int32_t)w.grid.x + gx);
        }
        std::vector<uint32_t> hashOut(NH, 0xffffffff);
        std::vector<PsContact> dC(1);
        std::vector<uint32_t> dU(1, 0), d32(NUM_BLOCKS, 0), d1(1, 0);
        PxgBufferHandle hW = createBufferFrom(dev, hw);
        PxgBufferHandle hO = createBufferFrom(dev, hashOut);
        PxgBufferHandle hB = createBufferFrom(dev, d32);
        PxgBufferHandle hT = createBufferFrom(dev, d1);
        PxgBufferHandle hCin = createBufferFrom(dev, dC);
        PxgBufferHandle hRA = createBufferFrom(dev, dU);
        PxgBufferHandle hRB = createBufferFrom(dev, dU);
        PxgBufferHandle hSA = createBufferFrom(dev, dC);
        PxgBufferHandle hSB = createBufferFrom(dev, dC);
        PbParams prm3 = { NH, 0, 0, 0 };
        Kernel k4("psBounds.spv", "ctPsGridHash", strides, sizeof(PbParams));
        k4.dispatch((NH + 255) / 256, { hW, hO, hB, hT, hCin, hRA, hRB, hSA, hSB }, prm3);
        dev->flush();
        const auto gH = downloadBuffer<uint32_t>(dev, hO, NH);
        size_t mmH = 0;
        for (uint32_t i = 0; i < NH; ++i)
            if (gH[i] != refHash[i]) ++mmH;
        PXG_CHECK_MSG(mmH == 0, "psBounds: %zu hash mismatches", mmH);
        dev->destroyBuffer(hW); dev->destroyBuffer(hO); dev->destroyBuffer(hB); dev->destroyBuffer(hT);
        dev->destroyBuffer(hCin); dev->destroyBuffer(hRA); dev->destroyBuffer(hRB);
        dev->destroyBuffer(hSA); dev->destroyBuffer(hSB);
    }
}

} // namespace
