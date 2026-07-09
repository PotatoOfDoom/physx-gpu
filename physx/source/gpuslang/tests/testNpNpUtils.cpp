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

// Verification of npNpUtils.slang: next3/prev3 (cyclic index step), the edge-flag
// bit tests (isActiveEdge/isBoundaryEdge/isEdgeNonconvex) and delayGenerateContact
// (the face-normal deferral test).  The index and flag results are pure integer
// and compared BIT-EXACTLY; delayGenerateContact's boolean is gated on an integer
// feature-index check and a float cosTheta<0.707 comparison, so cases within MARG
// of the cosTheta boundary are skipped.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct DelayIn { F4 minNormal, triangleLocNormal; uint32_t featureIndex, p0, p1, p2; };
struct NuParams { uint32_t numTests, pad0, pad1, pad2; };

static const uint32_t BOUNDARY = 0xffffffffu;
static const uint32_t NONCONVEX_FLAG = 0x80000000u;
static const uint32_t EDGE_FEATURE_IDX = 0x80000000u;
static const uint32_t TRI_FEATURE_IDX = 0x7fffffffu;
static const float MARG = 1e-3f;

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 nrm(V3 a) { const float m = std::sqrt(dot3(a, a)); return { a.x / m, a.y / m, a.z / m }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }
static V3 randUnit(std::mt19937& rng) { std::uniform_real_distribution<float> u(-1, 1); V3 v; float m; do { v = { u(rng), u(rng), u(rng) }; m = std::sqrt(dot3(v, v)); } while (m < 1e-2f); return { v.x / m, v.y / m, v.z / m }; }

static int refNext3(int x) { return x == 2 ? 0 : x + 1; }
static int refPrev3(int x) { return x == 0 ? 2 : x - 1; }

PXG_TEST(npNpUtils_next3prev3)
{
    // exhaustively over {0,1,2} plus replicated for a real dispatch size
    const uint32_t N = 3000;
    std::vector<uint32_t> in(N);
    std::mt19937 rng(69011);
    for (uint32_t i = 0; i < N; ++i) in[i] = rng() % 3u;

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> x2Init(N * 2, 9);
    std::vector<uint32_t> eD(N), eoD(N); std::vector<DelayIn> dD(N); std::vector<uint32_t> doD(N);
    PxgBufferHandle bX = createBufferFrom(dev, in);
    PxgBufferHandle bX2 = createBufferFrom(dev, x2Init);
    PxgBufferHandle bE = createBufferFrom(dev, eD);
    PxgBufferHandle bEo = createBufferFrom(dev, eoD);
    PxgBufferHandle bD = createBufferFrom(dev, dD);
    PxgBufferHandle bDo = createBufferFrom(dev, doD);

    NuParams prm = { N, 0, 0, 0 };
    Kernel k("npNpUtils.spv", "nuNext3Prev3", { 4, 8, 4, 4, (uint32_t)sizeof(DelayIn), 4 }, sizeof(NuParams));
    k.dispatch(GROUPS, { bX, bX2, bE, bEo, bD, bDo }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bX2, N * 2);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (g[i * 2] != (uint32_t)refNext3((int)in[i]) || g[i * 2 + 1] != (uint32_t)refPrev3((int)in[i])) ++mm;
    PXG_CHECK_MSG(mm == 0, "nuNext3Prev3: %zu mismatches", mm);

    dev->destroyBuffer(bX); dev->destroyBuffer(bX2); dev->destroyBuffer(bE); dev->destroyBuffer(bEo); dev->destroyBuffer(bD); dev->destroyBuffer(bDo);
}

PXG_TEST(npNpUtils_edgeFlags)
{
    std::mt19937 rng(69023);
    const uint32_t N = 12000;
    std::vector<uint32_t> in(N);
    std::vector<uint32_t> ref(N);
    size_t bnd = 0, ncx = 0, act = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t cat = rng() % 3u;
        uint32_t e;
        if (cat == 0) e = BOUNDARY;                        // boundary sentinel
        else if (cat == 1) e = (rng() & 0x7fffffffu) | NONCONVEX_FLAG; // nonconvex-flagged (not boundary unless all-ones, avoided)
        else e = rng() & 0x7fffffffu;                       // plain index (flag clear)
        if (e == BOUNDARY && cat == 1) e &= ~1u;            // ensure flagged case != BOUNDARY
        in[i] = e;
        const bool active = (e == BOUNDARY) || ((e & NONCONVEX_FLAG) == 0u);
        const bool boundary = (e == BOUNDARY);
        const bool nonconvex = (e != BOUNDARY) && ((e & NONCONVEX_FLAG) != 0u);
        ref[i] = (active ? 1u : 0u) | (boundary ? 2u : 0u) | (nonconvex ? 4u : 0u);
        if (boundary) ++bnd; if (nonconvex) ++ncx; if (active) ++act;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> xD(N), x2D(N * 2); std::vector<DelayIn> dD(N); std::vector<uint32_t> doD(N);
    std::vector<uint32_t> outInit(N, 99);
    PxgBufferHandle bX = createBufferFrom(dev, xD);
    PxgBufferHandle bX2 = createBufferFrom(dev, x2D);
    PxgBufferHandle bE = createBufferFrom(dev, in);
    PxgBufferHandle bEo = createBufferFrom(dev, outInit);
    PxgBufferHandle bD = createBufferFrom(dev, dD);
    PxgBufferHandle bDo = createBufferFrom(dev, doD);

    NuParams prm = { N, 0, 0, 0 };
    Kernel k("npNpUtils.spv", "nuEdgeFlags", { 4, 8, 4, 4, (uint32_t)sizeof(DelayIn), 4 }, sizeof(NuParams));
    k.dispatch(GROUPS, { bX, bX2, bE, bEo, bD, bDo }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bEo, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) if (g[i] != ref[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "nuEdgeFlags: %zu mismatches", mm);
    PXG_CHECK_MSG(bnd > 100 && ncx > 100 && act > 100, "nuEdgeFlags: coverage thin (bnd=%zu ncx=%zu act=%zu)", bnd, ncx, act);

    dev->destroyBuffer(bX); dev->destroyBuffer(bX2); dev->destroyBuffer(bE); dev->destroyBuffer(bEo); dev->destroyBuffer(bD); dev->destroyBuffer(bDo);
}

PXG_TEST(npNpUtils_delayContact)
{
    std::mt19937 rng(69037);
    const uint32_t N = 16000;
    std::vector<DelayIn> in(N);
    std::vector<uint32_t> ref(N);
    std::vector<uint8_t> skip(N, 0);
    const float threshold = 0.707106781f;
    size_t trueC = 0, falseC = 0, faceC = 0, sentinelC = 0, skipped = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const V3 mn = randUnit(rng);
        const V3 tn = randUnit(rng);
        uint32_t fi;
        const uint32_t cat = rng() % 4u;
        if (cat == 0) fi = TRI_FEATURE_IDX;
        else if (cat == 1) fi = EDGE_FEATURE_IDX;
        else fi = rng() % 32u; // a face index
        in[i] = { f4(mn), f4(tn), fi, 0, 0, 0 };

        bool r;
        if (fi != EDGE_FEATURE_IDX && fi != TRI_FEATURE_IDX)
        {
            const float cosTheta = dot3(mn, { -tn.x, -tn.y, -tn.z });
            if (std::fabs(cosTheta - threshold) < MARG) { skip[i] = 1; ++skipped; }
            r = cosTheta < threshold;
            ++faceC;
        }
        else { r = false; ++sentinelC; }
        ref[i] = r ? 1u : 0u;
        if (!skip[i]) { if (r) ++trueC; else ++falseC; }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> xD(N), x2D(N * 2), eD(N), eoD(N);
    std::vector<uint32_t> outInit(N, 9);
    PxgBufferHandle bX = createBufferFrom(dev, xD);
    PxgBufferHandle bX2 = createBufferFrom(dev, x2D);
    PxgBufferHandle bE = createBufferFrom(dev, eD);
    PxgBufferHandle bEo = createBufferFrom(dev, eoD);
    PxgBufferHandle bD = createBufferFrom(dev, in);
    PxgBufferHandle bDo = createBufferFrom(dev, outInit);

    NuParams prm = { N, 0, 0, 0 };
    Kernel k("npNpUtils.spv", "nuDelayContact", { 4, 8, 4, 4, (uint32_t)sizeof(DelayIn), 4 }, sizeof(NuParams));
    k.dispatch(GROUPS, { bX, bX2, bE, bEo, bD, bDo }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bDo, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) { if (skip[i]) continue; if (g[i] != ref[i]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "nuDelayContact: %zu mismatches", mm);
    PXG_CHECK_MSG(trueC > N / 8 && falseC > N / 8 && sentinelC > N / 8,
                  "nuDelayContact: coverage thin (true=%zu false=%zu face=%zu sentinel=%zu skip=%zu)", trueC, falseC, faceC, sentinelC, skipped);

    dev->destroyBuffer(bX); dev->destroyBuffer(bX2); dev->destroyBuffer(bE); dev->destroyBuffer(bEo); dev->destroyBuffer(bD); dev->destroyBuffer(bDo);
}

} // namespace
