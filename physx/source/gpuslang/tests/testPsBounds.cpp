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

// P6.4a-1 additions verify the psBoundsBda.slang BDA wrappers against the REAL
// engine descriptors: ps_updateBound{First,Second}PassLaunch (2-pass world AABB
// reduce, driven by a genuine PxgParticleSystem) and
// ps_primitivesBound{First,Second}PassLaunch (per-pair grid cell-range count +
// segmented scan, driven by real PxgContactManagerInput / PxgShape / PxBounds3).
// host_defines.h expands the __device__/__align__ qualifiers before the engine
// headers (nocuda/ is on the include path), matching testParticleSim.cpp.
#include "host_defines.h"
#include "PxgParticleSystem.h"     // real per-system descriptor
#include "PxgContactManager.h"     // PxgContactManagerInput
#include "PxgConvexConvexShape.h"  // PxgShape
#include "PxsTransformCache.h"     // PxsCachedTransform (ABI-parity arg)
#include "geometry/PxGeometry.h"   // PxGeometryType

#include <algorithm>
#include <cmath>
#include <cstddef>
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

// ===========================================================================
// P6.4a-1 BDA wrappers (psBoundsBda.slang) driven by REAL engine descriptors.
// ===========================================================================
using namespace physx;

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
static DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

static PxgParticleSystem makeZeroPs()
{
    PxgParticleSystem ps;
    std::memset(&ps, 0, sizeof(ps));
    return ps;
}

// ---------------------------------------------------------------------------
// ps_updateBound{First,Second}PassLaunch — two-pass world AABB reduce.
// Fills N particles into a genuine PxgParticleSystem, runs the per-block reduce
// then the global reduce+fatten, and asserts the broadphase bound array AND the
// mData.mBoundCenter writeback are BIT-EXACT vs a CPU min/max transcription.
// ---------------------------------------------------------------------------
PXG_TEST(psBounds_updateBoundReduce)
{
    // mirror strides / offsets the BDA kernel bakes in
    static_assert(sizeof(PxgParticleSystem) == 656, "PxgParticleSystem stride");
    static_assert(offsetof(PxgParticleSystem, mData) == 248, "mData offset");
    static_assert(offsetof(PxgParticleSystemData, mBoundCenter) == 0, "mBoundCenter first in mData");

    PxgSlangComputeDevice* dev = testDevice();

    // 2500 -> 3 pass-1 blocks of 1024 (partial last block) -> a 3-bound pass-2 reduce
    const uint32_t N = 2500;
    const uint32_t numBlocks = (N + 1024u - 1u) / 1024u;
    const uint32_t elemIndex = 1u; // exercise non-zero broadphase element index
    const float cd = 0.375f;

    std::mt19937 rng(64041);
    std::uniform_real_distribution<float> uv(-50.f, 50.f);
    std::vector<F4> predicted(N);
    for (uint32_t i = 0; i < N; ++i)
        predicted[i] = { uv(rng), uv(rng), uv(rng), 1.0f };

    // CPU reference: global min/max over xyz, then fatten by cd, then centre.
    float gmin[3] = { 3.4028235e38f, 3.4028235e38f, 3.4028235e38f };
    float gmax[3] = { -3.4028235e38f, -3.4028235e38f, -3.4028235e38f };
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* p = &predicted[i].x;
        for (int k = 0; k < 3; ++k) { gmin[k] = std::min(gmin[k], p[k]); gmax[k] = std::max(gmax[k], p[k]); }
    }
    float refMin[3], refMax[3], refCenter[3];
    for (int k = 0; k < 3; ++k)
    {
        refMin[k] = gmin[k] - cd;
        refMax[k] = gmax[k] + cd;
        refCenter[k] = (refMax[k] + refMin[k]) * 0.5f;
    }

    DevBuf bPred = makeDev(dev, predicted);
    std::vector<int32_t> dummyDiffuseCount = { 0 };
    DevBuf bDiffCount = makeDev(dev, dummyDiffuseCount);

    PxgParticleSystem ps = makeZeroPs();
    ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPred.addr));
    ps.mOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bPred.addr));       // unused (no CCD)
    ps.mDiffuseOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(bPred.addr));
    ps.mDiffusePosition_LifeTime = reinterpret_cast<float4*>(size_t(bPred.addr));
    ps.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(bDiffCount.addr));   // unused (maxDiffuse 0)
    ps.mCommonData.mNumParticles = N;
    ps.mCommonData.mMaxDiffuseParticles = 0;
    ps.mData.mFlags = 0;  // no eENABLE_SPECULATIVE_CCD

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<float> tempBounds(numBlocks * 6u, 0.f);
    DevBuf bTemp = makeDev(dev, tempBounds);
    std::vector<float> boundArray((elemIndex + 1u) * 6u, -123.f);
    DevBuf bBoundArr = makeDev(dev, boundArray);
    std::vector<float> contactDists(elemIndex + 1u, 0.f);
    contactDists[elemIndex] = cd;
    DevBuf bCd = makeDev(dev, contactDists);

    struct Push1 { uint64_t particleSystems; uint32_t id; uint32_t isTGS; uint64_t bounds; };
    Push1 p1 = { bDesc.addr, 0u, 0u, bTemp.addr };
    Kernel k1("psBoundsBda.spv", "ps_updateBoundFirstPassLaunch", {}, sizeof(Push1));
    k1.dispatch(numBlocks, {}, p1, 1u, 1u);
    dev->flush();

    struct Push2
    {
        uint64_t particleSystems; uint32_t id; uint32_t pad12; uint64_t bounds;
        uint32_t numBounds; uint32_t pad28; uint64_t boundArray; uint64_t contactDists;
        uint32_t elemIndex; uint32_t pad52;
    };
    Push2 p2 = { bDesc.addr, 0u, 0u, bTemp.addr, numBlocks, 0u, bBoundArr.addr, bCd.addr, elemIndex, 0u };
    Kernel k2("psBoundsBda.spv", "ps_updateBoundSecondPassLaunch", {}, sizeof(Push2));
    k2.dispatch(1u, {}, p2, 1u, 1u);
    dev->flush();

    const auto gBound = downloadBuffer<float>(dev, bBoundArr.h, (elemIndex + 1u) * 6u);
    const auto gDesc = downloadBuffer<PxgParticleSystem>(dev, bDesc.h, 1);

    size_t mm = 0;
    for (int k = 0; k < 3; ++k)
    {
        if (fbits(gBound[elemIndex * 6u + k]) != fbits(refMin[k])) { ++mm;
            std::printf("  boundMin[%d] gpu=%g ref=%g\n", k, gBound[elemIndex * 6u + k], refMin[k]); }
        if (fbits(gBound[elemIndex * 6u + 3u + k]) != fbits(refMax[k])) { ++mm;
            std::printf("  boundMax[%d] gpu=%g ref=%g\n", k, gBound[elemIndex * 6u + 3u + k], refMax[k]); }
    }
    const float gc[3] = { gDesc[0].mData.mBoundCenter.x, gDesc[0].mData.mBoundCenter.y, gDesc[0].mData.mBoundCenter.z };
    for (int k = 0; k < 3; ++k)
        if (fbits(gc[k]) != fbits(refCenter[k])) { ++mm;
            std::printf("  boundCenter[%d] gpu=%g ref=%g\n", k, gc[k], refCenter[k]); }

    PXG_CHECK_MSG(mm == 0, "ps_updateBound bit-exact AABB reduce vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPred.h); dev->destroyBuffer(bDiffCount.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bTemp.h); dev->destroyBuffer(bBoundArr.h); dev->destroyBuffer(bCd.h);
}

// ---------------------------------------------------------------------------
// P9.86c: signed-zero ties in ps_updateBound{First,Second}PassLaunch.
//
// (1) First pass, per-thread regular/diffuse combine (particlesystem.cu:506-507
//     speculative CCD, :513-514 plain): minPos.minimum(d) / maxPos.maximum(d) are
//     PxMin/PxMax<float> = intrinsics::selectMin/selectMax (PxMath.h:82-101,
//     PxUnixMathIntrinsics.h:116-125), `a < b ? a : b` / `a > b ? a : b`: on a
//     +-0 tie the SECOND operand (the diffuse value) survives.  GLSL FMin/FMax
//     (Slang min/max) are symmetric on llvmpipe (-0 < +0), so the P9.86 fix
//     spells the select (psbSelMin3/psbSelMax3).
// (2) Second pass, the per-iteration accumulator (particlesystem.cu:646-647):
//     sMinMax = PxMin(sMinMax, newMin) / PxMax(sMinMax, newMax), selects with the
//     accumulator FIRST.  It differs from FMin/FMax only when numBounds > 1024
//     (two or more iterations) and two iterations tie on a +-0.
// The block reductions themselves are warpReduction<MinOpFloat/MaxOpFloat> =
// fminf/fmaxf (reduction.cuh:53/83): minNum/maxNum with NVIDIA's zero rule
// (-0 < +0 in either operand order), the same rule as the port's FMin/FMax
// tree -- mirrored by nvFminf/nvFmaxf.
// ---------------------------------------------------------------------------
namespace psbz
{
static const float PXMAX = 3.4028235e38f;

static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static float selMin(float a, float b) { return a < b ? a : b; } // PxMin<float> = selectMin
static float selMax(float a, float b) { return a > b ? a : b; } // PxMax<float> = selectMax

// CUDA fminf/fmaxf on NVIDIA: NaN-ignoring, fminf(+-0, -+0) = -0, fmaxf = +0.
static float nvFminf(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
static float nvFmaxf(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}

// Mutant twins: 0 = the CUDA select, 1 = the reverted FMin/FMax (symmetric).
static int gCombineVariant = 0; // first pass, per-thread regular/diffuse combine
static int gAccVariant = 0;     // second pass, per-iteration accumulator

struct Ties { uint32_t minTie = 0, maxTie = 0; };

// getParticleSpeculativeContactVolume (gridCal.cuh:122-140) with contactDist 0;
// scaleInv = 1 (diffuse, PGS) or 2 (TGS), PxVec3::magnitude = sqrt(x*x+y*y+z*z).
static float specVolume(float c[3], const F4& cur, const F4& pred, bool isDiffuse, bool isTGS)
{
    const float scaleInv = isDiffuse ? 1.0f : (isTGS ? 2.0f : 1.0f);
    const float* pc = &cur.x;
    const float* pp = &pred.x;
    float o[3];
    for (int k = 0; k < 3; ++k)
    {
        o[k] = (pp[k] - pc[k]) * scaleInv * 0.5f;
        c[k] = pc[k] + o[k];
    }
    return std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]) + 0.0f;
}

// ps_updateBoundFirstPassLaunch: one PxBounds3 per 1024-thread block.  Ties
// counts the per-thread combines where select and FMin/FMax differ (min:
// regular -0 vs diffuse +0; max: regular +0 vs diffuse -0).
static std::vector<float> firstPassRef(const std::vector<F4>& rCur, const std::vector<F4>& rPred,
                                       const std::vector<F4>& dCur, const std::vector<F4>& dPred,
                                       uint32_t nReg, uint32_t nDiff, uint32_t numBlocks,
                                       bool ccd, bool isTGS, Ties& ties)
{
    std::vector<float> out(numBlocks * 6u);
    const uint32_t nP = std::max(nReg, nDiff);
    for (uint32_t b = 0; b < numBlocks; ++b)
    {
        float mn[3] = { PXMAX, PXMAX, PXMAX }, mx[3] = { -PXMAX, -PXMAX, -PXMAX };
        for (uint32_t t = 0; t < 1024u; ++t)
        {
            const uint32_t i = b * 1024u + t;
            float tmn[3] = { PXMAX, PXMAX, PXMAX }, tmx[3] = { -PXMAX, -PXMAX, -PXMAX };
            if (i < nP)
            {
                if (i < nReg)
                {
                    if (ccd)
                    {
                        float c[3];
                        const float r = specVolume(c, rCur[i], rPred[i], false, isTGS);
                        for (int k = 0; k < 3; ++k) { tmn[k] = c[k] - r; tmx[k] = c[k] + r; }
                    }
                    else
                        for (int k = 0; k < 3; ++k) tmn[k] = tmx[k] = (&rPred[i].x)[k];
                }
                if (i < nDiff)
                {
                    float dmn[3], dmx[3];
                    if (ccd)
                    {
                        float c[3];
                        const float r = specVolume(c, dCur[i], dPred[i], true, isTGS);
                        for (int k = 0; k < 3; ++k) { dmn[k] = c[k] - r; dmx[k] = c[k] + r; }
                    }
                    else
                        for (int k = 0; k < 3; ++k) dmn[k] = dmx[k] = (&dPred[i].x)[k];
                    for (int k = 0; k < 3; ++k)
                    {
                        if (fbits(tmn[k]) == 0x80000000u && fbits(dmn[k]) == 0u) ++ties.minTie;
                        if (fbits(tmx[k]) == 0u && fbits(dmx[k]) == 0x80000000u) ++ties.maxTie;
                        tmn[k] = gCombineVariant ? nvFminf(tmn[k], dmn[k]) : selMin(tmn[k], dmn[k]);
                        tmx[k] = gCombineVariant ? nvFmaxf(tmx[k], dmx[k]) : selMax(tmx[k], dmx[k]);
                    }
                }
            }
            for (int k = 0; k < 3; ++k) { mn[k] = nvFminf(mn[k], tmn[k]); mx[k] = nvFmaxf(mx[k], tmx[k]); }
        }
        for (int k = 0; k < 3; ++k) { out[b * 6u + k] = mn[k]; out[b * 6u + 3u + k] = mx[k]; }
    }
    return out;
}

// ps_updateBoundSecondPassLaunch: fminf/fmaxf over each 1024-bound iteration,
// folded into the accumulator with PxMin/PxMax (accumulator first), then
// fattened by cd; centre = (max + min) * 0.5 (the __shfl_sync pairing).
struct Pass2 { float bound[6]; float centre[3]; };
static Pass2 secondPassRef(const std::vector<float>& bounds, uint32_t numBounds, float cd, Ties& ties)
{
    float acc[6] = { PXMAX, PXMAX, PXMAX, -PXMAX, -PXMAX, -PXMAX };
    const uint32_t iters = (numBounds + 1023u) / 1024u;
    for (uint32_t it = 0; it < iters; ++it)
    {
        float nmn[3] = { PXMAX, PXMAX, PXMAX }, nmx[3] = { -PXMAX, -PXMAX, -PXMAX };
        for (uint32_t t = 0; t < 1024u; ++t)
        {
            const uint32_t e = it * 1024u + t;
            if (e >= numBounds) continue; // PxBounds3::setEmpty(): the fminf/fmaxf identities
            for (int k = 0; k < 3; ++k)
            {
                nmn[k] = nvFminf(nmn[k], bounds[e * 6u + k]);
                nmx[k] = nvFmaxf(nmx[k], bounds[e * 6u + 3u + k]);
            }
        }
        for (int k = 0; k < 3; ++k)
        {
            if (fbits(acc[k]) == 0x80000000u && fbits(nmn[k]) == 0u) ++ties.minTie;
            if (fbits(acc[3 + k]) == 0u && fbits(nmx[k]) == 0x80000000u) ++ties.maxTie;
            acc[k] = gAccVariant ? nvFminf(acc[k], nmn[k]) : selMin(acc[k], nmn[k]);
            acc[3 + k] = gAccVariant ? nvFmaxf(acc[3 + k], nmx[k]) : selMax(acc[3 + k], nmx[k]);
        }
    }
    Pass2 r;
    for (int k = 0; k < 3; ++k) { r.bound[k] = acc[k] - cd; r.bound[3 + k] = acc[3 + k] + cd; }
    for (int k = 0; k < 3; ++k) r.centre[k] = (r.bound[3 + k] + r.bound[k]) * 0.5f;
    return r;
}

struct Push1 { uint64_t particleSystems; uint32_t id; uint32_t isTGS; uint64_t bounds; };
struct Push2
{
    uint64_t particleSystems; uint32_t id; uint32_t pad12; uint64_t bounds;
    uint32_t numBounds; uint32_t pad28; uint64_t boundArray; uint64_t contactDists;
    uint32_t elemIndex; uint32_t pad52;
};

// Runs the second pass for one elemIndex and compares boundArray[elemIndex] and
// mData.mBoundCenter bitwise with `want`; returns the mismatch count.
static size_t runSecondPass(PxgSlangComputeDevice* dev, const DevBuf& bDesc, const DevBuf& bBounds,
                            uint32_t numBounds, const DevBuf& bBoundArr, uint32_t nElems,
                            const DevBuf& bCd, uint32_t elemIndex, const Pass2& want, const char* tag,
                            float* gotBound)
{
    Push2 p2 = { bDesc.addr, 0u, 0u, bBounds.addr, numBounds, 0u, bBoundArr.addr, bCd.addr, elemIndex, 0u };
    Kernel k2("psBoundsBda.spv", "ps_updateBoundSecondPassLaunch", {}, sizeof(Push2));
    k2.dispatch(1u, {}, p2, 1u, 1u);
    dev->flush();
    const auto gB = downloadBuffer<float>(dev, bBoundArr.h, nElems * 6u);
    const auto gD = downloadBuffer<PxgParticleSystem>(dev, bDesc.h, 1);
    const float gc[3] = { gD[0].mData.mBoundCenter.x, gD[0].mData.mBoundCenter.y, gD[0].mData.mBoundCenter.z };
    size_t mm = 0;
    for (int k = 0; k < 6; ++k)
    {
        gotBound[k] = gB[elemIndex * 6u + k];
        if (fbits(gotBound[k]) != fbits(want.bound[k]))
        {
            ++mm;
            std::printf("  %s boundArray[%u][%d] gpu=%08x ref=%08x\n", tag, elemIndex, k, fbits(gotBound[k]), fbits(want.bound[k]));
        }
    }
    for (int k = 0; k < 3; ++k)
        if (fbits(gc[k]) != fbits(want.centre[k]))
        {
            ++mm;
            std::printf("  %s boundCenter[%d] gpu=%08x ref=%08x\n", tag, k, fbits(gc[k]), fbits(want.centre[k]));
        }
    return mm;
}
} // namespace psbz

// First pass: slot I0 holds BOTH a regular and a diffuse particle whose values
// tie on +-0 in all three axes; every other particle rests in one octant, so
// the tie is the block extreme of min (positive octant) or max (negative octant).
//   run 0, plain, min: regular pred = (-0,-0,-0) vs diffuse (+0,+0,+0):
//                  CUDA select +0, reverted FMin -0.
//   run 1, plain, max: regular pred = (+0,+0,+0) vs diffuse (-0,-0,-0):
//                  CUDA select -0, reverted FMax +0.
//   run 2, speculative CCD, PGS, min: regular cur = (-0,-0,-0), pred = -2^-149
//                  per axis.  IEEE: o = -2^-150 -> -0 (round to even), c = -0,
//                  r = +0, so the regular c - r = -0; the diffuse particle rests
//                  at (+0,+0,+0).  CUDA is built -ftz=true
//                  (compiler/cmakegpu/CMakeLists.txt:100): there pred flushes, the
//                  regular value is +0, and the select gives +0 as well -- the
//                  parity image is model-independent, only the revert
//                  (FMin(-0, +0) = -0) differs.  (The max half of the CCD line
//                  cannot tie: r = sqrt(sum of squares) is never -0, so c + r
//                  never yields -0.)
//   run 3, PLATFORM probe: the run-2 regular particle alone.  Asserts the device
//                  does not flush denormal inputs (else run 2 silently stops
//                  exercising the CCD fix).  Not a CUDA parity check: a flushing
//                  CUDA gives +0 there.
// The first-pass PxBounds3 buffer is compared bitwise; the second pass runs on
// it with contactDists = +0 (elemIndex 0: keeps the min sign, +-0 - +0) and -0
// (elemIndex 1: keeps the max sign, -0 + -0 = -0), boundArray and mBoundCenter
// compared bitwise.
PXG_TEST(psBounds_updateBoundZeroSign)
{
    using namespace psbz;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t NREG = 1500u, NDIFF = 700u, MAXDIFF = 1024u, I0 = 17u;
    // PxgParticleSystemCore.cpp:862: blocks = ceil(max(numParticles, maxDiffuse) / 1024) = 2
    const uint32_t numBlocks = (std::max(NREG, MAXDIFF) + 1023u) / 1024u;
    const float MZ = -0.0f;
    const float DN = u2f(0x80000001u); // -2^-149

    std::mt19937 rng(8612);
    std::uniform_real_distribution<float> upos(0.5f, 50.f);
    std::vector<F4> posR(NREG), posD(NDIFF), negR(NREG), negD(NDIFF);
    for (auto& p : posR) p = F4{ upos(rng), upos(rng), upos(rng), 1.0f };
    for (auto& p : posD) p = F4{ upos(rng), upos(rng), upos(rng), 2.0f };
    for (auto& p : negR) p = F4{ -upos(rng), -upos(rng), -upos(rng), 1.0f };
    for (auto& p : negD) p = F4{ -upos(rng), -upos(rng), -upos(rng), 2.0f };

    Ties tiesRun[3];
    for (int run = 0; run < 4; ++run)
    {
        const bool ccd = run >= 2;
        const bool probe = run == 3;
        const bool maxRun = run == 1;
        const uint32_t nReg = probe ? 1u : NREG;
        const uint32_t nDiff = probe ? 0u : NDIFF;
        const uint32_t maxDiff = probe ? 0u : MAXDIFF;
        const uint32_t nBlk = probe ? 1u : numBlocks;

        // pristine at-rest copies every run (cur == pred everywhere else)
        std::vector<F4> rPred = maxRun ? negR : posR, dPred = maxRun ? negD : posD;
        std::vector<F4> rCur = rPred, dCur = dPred;
        if (probe)
        {
            rCur.assign(1, F4{ MZ, MZ, MZ, 1.0f });
            rPred.assign(1, F4{ DN, DN, DN, 1.0f });
            dCur.assign(1, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
            dPred = dCur;
        }
        else if (run == 0)
        {
            rPred[I0] = F4{ MZ, MZ, MZ, 1.0f };       dPred[I0] = F4{ 0.0f, 0.0f, 0.0f, 2.0f };
            rCur[I0] = rPred[I0];                     dCur[I0] = dPred[I0];
        }
        else if (run == 1)
        {
            rPred[I0] = F4{ 0.0f, 0.0f, 0.0f, 1.0f }; dPred[I0] = F4{ MZ, MZ, MZ, 2.0f };
            rCur[I0] = rPred[I0];                     dCur[I0] = dPred[I0];
        }
        else
        {
            rCur[I0] = F4{ MZ, MZ, MZ, 1.0f };        rPred[I0] = F4{ DN, DN, DN, 1.0f };
            dCur[I0] = F4{ 0.0f, 0.0f, 0.0f, 2.0f };  dPred[I0] = dCur[I0];
        }

        Ties probeTies, twinTies;
        Ties& ties = probe ? probeTies : tiesRun[run];
        gCombineVariant = 0;
        const std::vector<float> want = firstPassRef(rCur, rPred, dCur, dPred, nReg, nDiff, nBlk, ccd, false, ties);
        gCombineVariant = 1;
        const std::vector<float> twin = firstPassRef(rCur, rPred, dCur, dPred, nReg, nDiff, nBlk, ccd, false, twinTies);
        gCombineVariant = 0;

        DevBuf bRC = makeDev(dev, rCur), bRP = makeDev(dev, rPred);
        DevBuf bDC = makeDev(dev, dCur), bDP = makeDev(dev, dPred);
        std::vector<int32_t> diffCount = { int32_t(nDiff), 0 }; // mNumDiffuseParticles -> int[2]
        DevBuf bCnt = makeDev(dev, diffCount);

        PxgParticleSystem ps = makeZeroPs();
        ps.mOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bRC.addr));
        ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bRP.addr));
        ps.mDiffuseOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(bDC.addr));
        ps.mDiffusePosition_LifeTime = reinterpret_cast<float4*>(size_t(bDP.addr));
        ps.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(bCnt.addr));
        ps.mCommonData.mNumParticles = nReg;
        ps.mCommonData.mMaxDiffuseParticles = maxDiff;
        ps.mData.mFlags = ccd ? 8u : 0u; // PxParticleFlag::eENABLE_SPECULATIVE_CCD (PxPBDParticleSystem.h:185)
        std::vector<PxgParticleSystem> descArr = { ps };
        DevBuf bDesc = makeDev(dev, descArr);

        std::vector<float> tempBounds(nBlk * 6u, 123.f);
        DevBuf bTemp = makeDev(dev, tempBounds);

        Push1 p1 = { bDesc.addr, 0u, 0u /*PGS*/, bTemp.addr };
        Kernel k1("psBoundsBda.spv", "ps_updateBoundFirstPassLaunch", {}, sizeof(Push1));
        k1.dispatch(nBlk, {}, p1, 1u, 1u);
        dev->flush();
        const auto gTemp = downloadBuffer<float>(dev, bTemp.h, nBlk * 6u);

        if (probe)
        {
            size_t live = 0;
            for (int k = 0; k < 3; ++k)
                if (fbits(want[k]) == 0x80000000u && fbits(gTemp[k]) == 0x80000000u) ++live;
            PXG_CHECK_MSG(live == 3u,
                          "psBounds zero-sign PLATFORM probe: min gpu=%08x/%08x/%08x -- the device flushes "
                          "denormal inputs, so the CCD tie of run 2 is no longer reachable (a flushing CUDA "
                          "gives +0 here: liveness probe, not a parity check)",
                          fbits(gTemp[0]), fbits(gTemp[1]), fbits(gTemp[2]));
        }
        else
        {
            // the mutant twin must change the first-pass image in every axis
            const uint32_t o = maxRun ? 3u : 0u;
            const uint32_t wantBits = maxRun ? 0x80000000u : 0u;
            for (uint32_t k = 0; k < 3u; ++k)
                PXG_CHECK_MSG(fbits(want[o + k]) == wantBits && fbits(twin[o + k]) == (wantBits ^ 0x80000000u),
                              "psBounds zero-sign run %d: tempBounds[%u] twin gives the same image (want=%08x twin=%08x)",
                              run, o + k, fbits(want[o + k]), fbits(twin[o + k]));

            size_t mm = 0;
            for (uint32_t j = 0; j < nBlk * 6u; ++j)
                if (fbits(gTemp[j]) != fbits(want[j]))
                {
                    ++mm;
                    std::printf("  run %d tempBounds[%u] gpu=%08x ref=%08x twin=%08x\n", run, j,
                                fbits(gTemp[j]), fbits(want[j]), fbits(twin[j]));
                }

            // second pass on the GPU's own first-pass output (== want when mm == 0)
            std::vector<float> boundArray(2u * 6u, -123.f);
            DevBuf bBoundArr = makeDev(dev, boundArray);
            std::vector<float> contactDists = { 0.0f, MZ };
            DevBuf bCd = makeDev(dev, contactDists);
            for (uint32_t e = 0; e < 2u; ++e)
            {
                Ties t2, t2twin;
                const Pass2 w2 = secondPassRef(want, nBlk, contactDists[e], t2);
                const Pass2 v2 = secondPassRef(twin, nBlk, contactDists[e], t2twin);
                char tag[32];
                std::snprintf(tag, sizeof(tag), "run %d cd%s0", run, e ? "-" : "+");
                float got[6];
                mm += runSecondPass(dev, bDesc, bTemp, nBlk, bBoundArr, 2u, bCd, e, w2, tag, got);
                if (e != (maxRun ? 1u : 0u))
                    continue; // cd = +0 shows the min sign, cd = -0 the max sign
                for (uint32_t k = 0; k < 3u; ++k)
                {
                    PXG_CHECK_MSG(fbits(w2.bound[o + k]) == wantBits && fbits(v2.bound[o + k]) == (wantBits ^ 0x80000000u),
                                  "psBounds zero-sign %s: boundArray[%u] twin does not differ", tag, o + k);
                    PXG_CHECK_MSG(fbits(got[o + k]) == wantBits, "psBounds zero-sign %s: boundArray[%u] gpu=%08x, CUDA %08x",
                                  tag, o + k, fbits(got[o + k]), wantBits);
                }
            }
            PXG_CHECK_MSG(mm == 0, "psBounds zero-sign run %d: %zu bitwise mismatches", run, mm);
            dev->destroyBuffer(bBoundArr.h); dev->destroyBuffer(bCd.h);
        }

        dev->destroyBuffer(bRC.h); dev->destroyBuffer(bRP.h); dev->destroyBuffer(bDC.h); dev->destroyBuffer(bDP.h);
        dev->destroyBuffer(bCnt.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bTemp.h);
    }
    PXG_CHECK_MSG(tiesRun[0].minTie >= 3u && tiesRun[1].maxTie >= 3u && tiesRun[2].minTie >= 3u,
                  "psBounds zero-sign coverage: regular/diffuse +-0 ties not exercised in every axis "
                  "(plain min=%u, plain max=%u, ccd min=%u)", tiesRun[0].minTie, tiesRun[1].maxTie, tiesRun[2].minTie);
}

// Second pass alone with numBounds = 1025 (two accumulator iterations), in two
// bound sets.  MIN set: every bound in the positive octant except bound 0
// (iteration 0) with min = (-0,-0,-0) and bound 1024 (iteration 1) with
// min = (+0,+0,+0); CUDA's PxMin(acc, new) keeps the NEW iteration's +0, FMin
// keeps -0; contactDists = +0 carries the sign into boundArray.  MAX set: every
// bound in the negative octant except bound 0 with max = (+0,+0,+0) and bound
// 1024 with max = (-0,-0,-0); PxMax(acc, new) keeps -0, FMax +0; contactDists
// = -0 carries it (-0 + -0 = -0).  (Engine: more than 1024 first-pass blocks,
// i.e. more than 1,048,576 particles or max diffuse particles.)
PXG_TEST(psBounds_updateBoundSecondPassAccZeroSign)
{
    using namespace psbz;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t NB = 1025u;
    const float MZ = -0.0f;
    std::mt19937 rng(8613);
    std::uniform_real_distribution<float> ulo(0.5f, 20.f), uext(0.5f, 30.f);

    PxgParticleSystem ps = makeZeroPs();
    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<float> contactDists = { 0.0f, MZ };
    DevBuf bCd = makeDev(dev, contactDists);

    Ties tiesSet[2];
    for (uint32_t set = 0; set < 2u; ++set)
    {
        const bool maxSet = set == 1u;
        std::vector<float> bounds(NB * 6u);
        for (uint32_t e = 0; e < NB; ++e)
        {
            float* b = &bounds[e * 6u];
            for (int k = 0; k < 3; ++k)
            {
                if (!maxSet) { b[k] = ulo(rng);      b[3 + k] = b[k] + uext(rng); }
                else         { b[3 + k] = -ulo(rng); b[k] = b[3 + k] - uext(rng); }
            }
        }
        const uint32_t o = maxSet ? 3u : 0u;
        for (uint32_t k = 0; k < 3u; ++k)
        {
            bounds[0u * 6u + o + k] = maxSet ? 0.0f : MZ;     // iteration 0
            bounds[1024u * 6u + o + k] = maxSet ? MZ : 0.0f;  // iteration 1
        }
        DevBuf bBounds = makeDev(dev, bounds);
        std::vector<float> boundArray(2u * 6u, -123.f);
        DevBuf bBoundArr = makeDev(dev, boundArray);

        size_t mm = 0;
        for (uint32_t e = 0; e < 2u; ++e)
        {
            Ties tw, tIgnored;
            gAccVariant = 0;
            const Pass2 want = secondPassRef(bounds, NB, contactDists[e], e == set ? tiesSet[set] : tIgnored);
            gAccVariant = 1;
            const Pass2 twin = secondPassRef(bounds, NB, contactDists[e], tw);
            gAccVariant = 0;
            char tag[32];
            std::snprintf(tag, sizeof(tag), "acc %s cd%s0", maxSet ? "max" : "min", e ? "-" : "+");
            float got[6];
            mm += runSecondPass(dev, bDesc, bBounds, NB, bBoundArr, 2u, bCd, e, want, tag, got);
            if (e != set)
                continue; // cd = +0 shows the min sign, cd = -0 the max sign
            const uint32_t wantBits = maxSet ? 0x80000000u : 0u;
            for (uint32_t k = 0; k < 3u; ++k)
            {
                PXG_CHECK_MSG(fbits(want.bound[o + k]) == wantBits && fbits(twin.bound[o + k]) == (wantBits ^ 0x80000000u),
                              "psBounds %s: boundArray[%u] twin gives the same image (want=%08x twin=%08x)",
                              tag, o + k, fbits(want.bound[o + k]), fbits(twin.bound[o + k]));
                PXG_CHECK_MSG(fbits(got[o + k]) == wantBits, "psBounds %s: boundArray[%u] gpu=%08x, CUDA %08x",
                              tag, o + k, fbits(got[o + k]), wantBits);
            }
        }
        PXG_CHECK_MSG(mm == 0, "psBounds acc zero-sign set %u: %zu bitwise mismatches", set, mm);
        dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bBoundArr.h);
    }
    PXG_CHECK_MSG(tiesSet[0].minTie >= 3u && tiesSet[1].maxTie >= 3u,
                  "psBounds acc zero-sign coverage: accumulator +-0 ties not exercised in every axis (min=%u max=%u)",
                  tiesSet[0].minTie, tiesSet[1].maxTie);

    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCd.h);
}

// ---------------------------------------------------------------------------
// ps_primitivesBound{First,Second}PassLaunch — per-pair grid cell-range count +
// segmented scan through real PxgContactManagerInput / PxgShape / PxBounds3.
// Verifies the exclusive scan (offsets), the 32 per-block totals and the grand
// total BIT-EXACT vs a CPU transcription for particle-vs-plane pairs (with the
// LoadShapePair swap exercised by alternating which shapeRef is the particle).
// ---------------------------------------------------------------------------
PXG_TEST(psBounds_primitivesBoundCellRange)
{
    static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput stride");
    static_assert(sizeof(PxgShape) == 48, "PxgShape stride");
    static_assert(offsetof(PxgShape, type) == 40, "PxgShape::type offset");
    static_assert(offsetof(PxgShape, particleOrSoftbodyId) == 44, "PxgShape::particleOrSoftbodyId offset");

    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t BLOCK = 512u, NUM_BLOCKS = 32u;
    const uint32_t NT = 6000u;         // > 512*32/... spans multiple tiles per device block
    const float cw = 0.5f;
    const uint32_t grid[3] = { 16u, 16u, 16u };

    std::mt19937 rng(64042);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    auto safeCoordL = [&](float v) { const float q = v / cw; return std::fabs(q - std::round(q)) > 1e-3f; };
    auto floorDivL = [&](float v) { return (int32_t)std::floor(v / cw); };

    // per-pair boxes + per-cache-ref contact distances
    std::vector<F4> pMin(NT), pMax(NT), rMin(NT), rMax(NT);  // particle / rigid boxes
    std::vector<float> cdP(NT), cdR(NT);
    for (uint32_t i = 0; i < NT; ++i)
    {
        cdP[i] = 0.05f + 0.15f * u01(rng);
        cdR[i] = 0.05f + 0.15f * u01(rng);
        const float cd = cdP[i] + cdR[i];
        for (uint32_t attempt = 0;; ++attempt)
        {
            const bool disjoint = (i % 7 == 5);
            const float c0[3] = { 4.0f * uv(rng), 4.0f * uv(rng), 4.0f * uv(rng) };
            const float h0[3] = { 0.3f + 1.6f * u01(rng), 0.3f + 1.6f * u01(rng), 0.3f + 1.6f * u01(rng) };
            float c1[3] = { c0[0] + 1.3f * uv(rng), c0[1] + 1.3f * uv(rng), c0[2] + 1.3f * uv(rng) };
            if (disjoint) c1[0] = c0[0] + h0[0] + 4.0f + 2.0f * u01(rng);
            const float h1[3] = { 0.3f + 1.6f * u01(rng), 0.3f + 1.6f * u01(rng), 0.3f + 1.6f * u01(rng) };
            float pmn[3], pmx[3], rmn[3], rmx[3];
            bool ok = true;
            for (int k = 0; k < 3; ++k)
            {
                pmn[k] = c0[k] - h0[k]; pmx[k] = c0[k] + h0[k];
                rmn[k] = c1[k] - h1[k]; rmx[k] = c1[k] + h1[k];
                const float mn = std::max(pmn[k], rmn[k]) - cd;
                const float mx = std::min(pmx[k], rmx[k]) + cd;
                ok = ok && safeCoordL(mn) && safeCoordL(mx);
            }
            if (ok)
            {
                pMin[i] = { pmn[0], pmn[1], pmn[2], 0 }; pMax[i] = { pmx[0], pmx[1], pmx[2], 0 };
                rMin[i] = { rmn[0], rmn[1], rmn[2], 0 }; rMax[i] = { rmx[0], rmx[1], rmx[2], 0 };
                break;
            }
            PXG_CHECK(attempt < 200);
        }
    }

    // CPU reference: cell counts, straight exclusive scan, per-block chunk totals
    std::vector<uint32_t> counts(NT), refOffsets(NT);
    uint32_t total = 0;
    size_t zeroRows = 0, multiRows = 0;
    for (uint32_t i = 0; i < NT; ++i)
    {
        const float cd = cdP[i] + cdR[i];
        uint32_t count = 1;
        for (int k = 0; k < 3; ++k)
        {
            const float mn = std::max((&pMin[i].x)[k], (&rMin[i].x)[k]) - cd;
            const float mx = std::min((&pMax[i].x)[k], (&rMax[i].x)[k]) + cd;
            const int32_t r = std::min(std::max(floorDivL(mx) - floorDivL(mn) + 1, 0), (int32_t)grid[k]);
            count *= (uint32_t)r;
        }
        counts[i] = count;
        refOffsets[i] = total;
        total += count;
        if (count == 0) ++zeroRows;
        if (count > 4) ++multiRows;
    }
    const uint32_t nbBlocksRequired = (NT + BLOCK - 1) / BLOCK;
    const uint32_t iters = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;
    std::vector<uint32_t> refBlock(NUM_BLOCKS, 0);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
        for (uint32_t j = b * iters * BLOCK; j < std::min((b + 1) * iters * BLOCK, NT); ++j)
            refBlock[b] += counts[j];

    // ---- build the real engine descriptors ----
    // pair i: cache refs (2i, 2i+1), shape refs (2i, 2i+1); the particle shape
    // is placed in slot 0 for even i and slot 1 for odd i (exercises the swap).
    std::vector<PxgContactManagerInput> cmInputs(NT);
    std::vector<PxgShape> shapes(2 * NT);
    std::vector<float> boundsF(2 * NT * 6, 0.f); // PxBounds3[2*NT] as raw floats
    std::vector<float> contactDistance(2 * NT, 0.f);

    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    for (uint32_t i = 0; i < NT; ++i)
    {
        const uint32_t particleCache = 2u * i, rigidCache = 2u * i + 1u;
        PxgShape particleShape; std::memset(&particleShape, 0, sizeof(particleShape));
        particleShape.type = PxU32(PxGeometryType::ePARTICLESYSTEM);
        particleShape.particleOrSoftbodyId = 0u;
        PxgShape rigidShape; std::memset(&rigidShape, 0, sizeof(rigidShape));
        rigidShape.type = PxU32(PxGeometryType::ePLANE);

        const bool particleInSlot0 = (i % 2u) == 0u;
        const uint32_t sPart = 2u * i, sRig = 2u * i + 1u;
        if (particleInSlot0)
        {
            shapes[sPart] = particleShape; shapes[sRig] = rigidShape;
            cmInputs[i].shapeRef0 = sPart; cmInputs[i].shapeRef1 = sRig;
            cmInputs[i].transformCacheRef0 = particleCache; cmInputs[i].transformCacheRef1 = rigidCache;
        }
        else
        {
            shapes[sPart] = rigidShape; shapes[sRig] = particleShape;
            cmInputs[i].shapeRef0 = sPart; cmInputs[i].shapeRef1 = sRig;
            cmInputs[i].transformCacheRef0 = rigidCache; cmInputs[i].transformCacheRef1 = particleCache;
        }

        // bounds[particleCache] = particle box, bounds[rigidCache] = rigid box
        for (int k = 0; k < 3; ++k)
        {
            boundsF[particleCache * 6 + k] = (&pMin[i].x)[k];
            boundsF[particleCache * 6 + 3 + k] = (&pMax[i].x)[k];
            boundsF[rigidCache * 6 + k] = (&rMin[i].x)[k];
            boundsF[rigidCache * 6 + 3 + k] = (&rMax[i].x)[k];
        }
        contactDistance[particleCache] = cdP[i];
        contactDistance[rigidCache] = cdR[i];
    }

    PxgParticleSystem ps = makeZeroPs();
    ps.mCommonData.mGridCellWidth = cw;
    ps.mCommonData.mGridSizeX = grid[0];
    ps.mCommonData.mGridSizeY = grid[1];
    ps.mCommonData.mGridSizeZ = grid[2];
    std::vector<PxgParticleSystem> descArr = { ps };

    DevBuf bCin = makeDev(dev, cmInputs);
    DevBuf bShapes = makeDev(dev, shapes);
    DevBuf bBounds = makeDev(dev, boundsF);
    DevBuf bCdist = makeDev(dev, contactDistance);
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> blockOffsets(NUM_BLOCKS, 0), offsets(NT, 0), totalBuf(1, 0);
    DevBuf bBlockOff = makeDev(dev, blockOffsets);
    DevBuf bOffsets = makeDev(dev, offsets);
    DevBuf bTotal = makeDev(dev, totalBuf);

    struct PB1
    {
        uint32_t numTests, pad4; uint64_t cmInputs, shapes, transformCache, bounds,
            contactDistance, particleSystems, blockOffsets, offsets;
    };
    PB1 pb1 = { NT, 0u, bCin.addr, bShapes.addr, /*transformCache (unused)*/ bBounds.addr,
                bBounds.addr, bCdist.addr, bDesc.addr, bBlockOff.addr, bOffsets.addr };
    Kernel kp1("psBoundsBda.spv", "ps_primitivesBoundFirstPassLaunch", {}, sizeof(PB1));
    kp1.dispatch(NUM_BLOCKS, {}, pb1, 1u, 1u);
    dev->flush();

    const auto gBlocks = downloadBuffer<uint32_t>(dev, bBlockOff.h, NUM_BLOCKS);
    size_t mmB = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
        if (gBlocks[b] != refBlock[b]) ++mmB;
    PXG_CHECK_MSG(mmB == 0, "psPrimBound: %zu block-total mismatches", mmB);

    struct PB2 { uint32_t numTests, pad4; uint64_t blockOffsets, offsets, totalNumPairs; };
    PB2 pb2 = { NT, 0u, bBlockOff.addr, bOffsets.addr, bTotal.addr };
    Kernel kp2("psBoundsBda.spv", "ps_primitivesBoundSecondPassLaunch", {}, sizeof(PB2));
    kp2.dispatch(NUM_BLOCKS, {}, pb2, 1u, 1u);
    dev->flush();

    const auto gOff = downloadBuffer<uint32_t>(dev, bOffsets.h, NT);
    const uint32_t gTotal = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    size_t mm = 0;
    for (uint32_t i = 0; i < NT; ++i)
        if (gOff[i] != refOffsets[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "psPrimBound: %zu offset mismatches", mm);
    PXG_CHECK_MSG(gTotal == total, "psPrimBound: total %u != %u", gTotal, total);
    PXG_CHECK_MSG(zeroRows > 400 && multiRows > 200 && total > 20000,
                  "psPrimBound: coverage thin (zero=%zu multi=%zu total=%u)", zeroRows, multiRows, total);

    dev->destroyBuffer(bCin.h); dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bBounds.h);
    dev->destroyBuffer(bCdist.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bBlockOff.h); dev->destroyBuffer(bOffsets.h); dev->destroyBuffer(bTotal.h);
}

} // namespace
