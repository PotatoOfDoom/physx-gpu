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

// Verification of npSparseGridCellFromPos.slang (sparseGridCellFromPos): the
// full sparse-grid position-to-cell composition getCellIndexFromPosition
// (sparseGridStandalone.cuh L171-200, haloSize=0), against an exact CPU
// transcription.  Integer cell exact; local position bitwise mod +-0 (eqFZ).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct I4 { int32_t x, y, z, w; };
struct CiIn  { F4 pos; U4 sizes; U4 keys[4]; };
struct CiOut { I4 cell; F4 plocal; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const int32_t MAX_DIM = 1024;
static const int32_t MIN_ID = -512;

static int32_t imax(int32_t a, int32_t b) { return (a < b) ? b : a; }
static int32_t imin(int32_t a, int32_t b) { return (a < b) ? a : b; }
static int32_t iclamp(int32_t v, int32_t lo, int32_t hi) { return imin(hi, imax(lo, v)); }

static uint32_t keyAt(const CiIn& s, uint32_t i) { const uint32_t* k = &s.keys[i >> 2].x; return k[i & 3u]; }

static uint32_t searchSorted(const CiIn& s, uint32_t numElements, uint32_t value)
{
    uint32_t left = 0, right = numElements;
    while ((right - left) > 1u)
    {
        const uint32_t pos = (left + right) >> 1;
        if (keyAt(s, pos) <= value)
            left = pos;
        else
            right = pos;
    }
    return left;
}

// The hash of the subgrid containing p (mirrors the ref's float path exactly).
static uint32_t hashOf(float px, float py, float pz, float dx, int32_t sX, int32_t sY, int32_t sZ)
{
    const float dsx = dx * float(sX), dsy = dx * float(sY), dsz = dx * float(sZ);
    const int32_t idx = int32_t(std::floor(px / dsx));
    const int32_t idy = int32_t(std::floor(py / dsy));
    const int32_t idz = int32_t(std::floor(pz / dsz));
    return uint32_t(MAX_DIM * MAX_DIM * (idz - MIN_ID) + MAX_DIM * (idy - MIN_ID) + (idx - MIN_ID));
}

static void refCi(const CiIn& s, CiOut& o)
{
    float px = s.pos.x, py = s.pos.y, pz = s.pos.z;
    const float dx = s.pos.w;
    const int32_t sX = int32_t(s.sizes.x), sY = int32_t(s.sizes.y), sZ = int32_t(s.sizes.z);
    const uint32_t numSubgrids = s.sizes.w;

    const float dsx = dx * float(sX), dsy = dx * float(sY), dsz = dx * float(sZ);
    const int32_t idx = int32_t(std::floor(px / dsx));
    const int32_t idy = int32_t(std::floor(py / dsy));
    const int32_t idz = int32_t(std::floor(pz / dsz));
    const uint32_t hash = uint32_t(MAX_DIM * MAX_DIM * (idz - MIN_ID) + MAX_DIM * (idy - MIN_ID) + (idx - MIN_ID));

    const uint32_t sortedIdx = searchSorted(s, numSubgrids, hash);
    const bool found = (keyAt(s, sortedIdx) == hash);

    if (!found)
    {
        o.cell = I4{ -1, -1, -1, -1 };
        o.plocal = F4{ px, py, pz, 0.f };
        return;
    }

    const float invDx = 1.0f / dx;
    const float ox = float(idx) * dx * float(sX);
    const float oy = float(idy) * dx * float(sY);
    const float oz = float(idz) * dx * float(sZ);
    px = px - ox; py = py - oy; pz = pz - oz;

    int32_t rx = int32_t(std::floor(px * invDx));
    int32_t ry = int32_t(std::floor(py * invDx));
    int32_t rz = int32_t(std::floor(pz * invDx));
    rx = iclamp(rx, 0, sX - 1);
    ry = iclamp(ry, 0, sY - 1);
    rz = iclamp(rz, 0, sZ - 1);

    o.cell = I4{ rx, ry, rz, int32_t(sortedIdx) };
    o.plocal = F4{ px, py, pz, 0.f };
}

PXG_TEST(npSparseGridCellFromPos)
{
    std::mt19937 rng(249001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto ru = [&](uint32_t lo, uint32_t hi) { return lo + rng() % (hi - lo + 1u); };

    std::vector<CiIn> in(N);
    std::vector<CiOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CiIn s{};
        const float dx = rf(0.1f, 1.5f);
        const uint32_t sX = ru(4u, 32u), sY = ru(4u, 32u), sZ = ru(4u, 32u);
        s.pos = F4{ rf(-30.f, 30.f), rf(-30.f, 30.f), rf(-30.f, 30.f), dx };
        const uint32_t numSubgrids = ru(8u, 16u);
        s.sizes = U4{ sX, sY, sZ, numSubgrids };

        bool stripeRow = false;
        // division-boundary stripe: place pos.x within a few ulps of a subgrid
        // boundary k*domainSize where floor(p/d) != floor(p*(1/d)) -- the two
        // quotient roundings straddle the integer.  The CORRECT code (plain
        // division, both shader and ref) agrees on these rows; a mutant using
        // reciprocal-multiply in calcSubgridId picks a different subgrid ->
        // different hash -> found-status flips -> divergent output.  This makes
        // the plain-division faithfulness observable (it is measure-zero under
        // purely random positions).
        if ((i % 13u) == 5u)
        {
            const float dsx = dx * float(sX);
            bool got = false;
            for (int32_t kk = 1; kk <= 6 && !got; ++kk)
            {
                float cand = float(kk) * dsx;
                for (int j = 0; j < 64 && !got; ++j)
                {
                    if (std::floor(cand / dsx) != std::floor(cand * (1.0f / dsx))) { s.pos.x = cand; got = true; }
                    cand = std::nextafterf(cand, 0.0f);
                }
                cand = std::nextafterf(float(kk) * dsx, 1e30f);
                for (int j = 0; j < 64 && !got; ++j)
                {
                    if (std::floor(cand / dsx) != std::floor(cand * (1.0f / dsx))) { s.pos.x = cand; got = true; }
                    cand = std::nextafterf(cand, 1e30f);
                }
            }
            stripeRow = got;   // fallback: plain random row
        }
        // clamp-hi stripe: place pos.x a few ulps below the TOP of its subgrid so
        // that floor((p - origin) * invDx) lands on sX (fp rounding pushes the
        // local coordinate to the far face) and the [0, sX-1] hi clamp BINDS.
        // The correct code clamps to sX-1; a widened-bound mutant outputs sX.
        if ((i % 13u) == 9u)
        {
            const float dsx = dx * float(sX);
            const float idf = std::floor(s.pos.x / dsx);
            float cand = (idf + 1.0f) * dsx;
            for (int j = 0; j < 96; ++j)
            {
                cand = std::nextafterf(cand, -1e30f);
                if (std::floor(cand / dsx) != idf) continue;        // must stay in the same subgrid
                const float ox = idf * dx * float(sX);              // (id*dx)*sX, left-to-right as in the code
                const int32_t rx = int32_t(std::floor((cand - ox) * (1.0f / dx)));
                if (rx >= int32_t(sX)) { s.pos.x = cand; stripeRow = true; break; }
            }
        }

        if (stripeRow) s.sizes.w = 16u;   // whole table searched -> inserted hash always found

        const uint32_t h = hashOf(s.pos.x, s.pos.y, s.pos.z, dx, int32_t(sX), int32_t(sY), int32_t(sZ));

        // build a sorted unique key table; ~2/3 of rows contain the true hash,
        // and crafted stripe rows ALWAYS contain it (the boundary/clamp behavior
        // is only observable on the found path)
        uint32_t* keys = &s.keys[0].x;
        std::vector<uint32_t> kv;
        while (kv.size() < 16u)
        {
            uint32_t k = rng() % 0x40000000u;
            if (k != h) kv.push_back(k);
        }
        if ((i % 3u) != 2u || stripeRow) kv[ru(0u, 15u)] = h;
        std::sort(kv.begin(), kv.end());
        for (uint32_t j = 0; j < 16u; ++j) keys[j] = kv[j];

        in[i] = s;
        refCi(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CiOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSparseGridCellFromPos.spv", "sparseGridCellFromPos",
             { (uint32_t)sizeof(CiIn), (uint32_t)sizeof(CiOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CiOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = ref[i].cell.x == g[i].cell.x && ref[i].cell.y == g[i].cell.y
               && ref[i].cell.z == g[i].cell.z && ref[i].cell.w == g[i].cell.w
               && eqFZ(ref[i].plocal.x, g[i].plocal.x) && eqFZ(ref[i].plocal.y, g[i].plocal.y)
               && eqFZ(ref[i].plocal.z, g[i].plocal.z);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u cR (%d %d %d %d) cG (%d %d %d %d) pR (%g %g %g) pG (%g %g %g)\n", i,
                            ref[i].cell.x, ref[i].cell.y, ref[i].cell.z, ref[i].cell.w,
                            g[i].cell.x, g[i].cell.y, g[i].cell.z, g[i].cell.w,
                            ref[i].plocal.x, ref[i].plocal.y, ref[i].plocal.z,
                            g[i].plocal.x, g[i].plocal.y, g[i].plocal.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sparseGridCellFromPos outputs identical (cell exact, plocal mod +-0)");
}

}  // namespace
