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

// Verification of npBoxReduce.slang: boxCompactContacts — the Stage-2 contact
// compaction/reduction of doBoxBoxGenerateContacts.  Each group of 4 lanes
// carries 0..3 local-space contacts; the kernel prefix-sums the counts and
// either stores them compacted (total<=6) or reduces to the 4-point manifold
// (deepest / furthest / two perpendicular extrema), all transformed to global
// space.  The C++ reference transcribes both paths.  The reduction's argmin/
// argmax selections turn on float comparisons, so groups where any selection is
// within a small margin of a tie are skipped (a GPU-vs-CPU flip there changes the
// stored contact and is measure-zero for the fixed seed); skips are asserted a
// minority and both total<=6 / total>6 paths are exercised.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct LaneIn { F4 c0, c1, c2, meta; };
struct GroupIn { F4 ntCol0, ntCol1, ntCol2, ntP, fn; };

const float BIG = 3.4028235e38f;
const float MARG = 1e-3f;

static inline V3 xyz(const F4& a) { return V3{ a.x, a.y, a.z }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 crossp(const V3& a, const V3& b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 xform(const GroupIn& g, const V3& v)
{
    return V3{ g.ntCol0.x * v.x + g.ntCol1.x * v.y + g.ntCol2.x * v.z + g.ntP.x,
              g.ntCol0.y * v.x + g.ntCol1.y * v.y + g.ntCol2.y * v.z + g.ntP.y,
              g.ntCol0.z * v.x + g.ntCol1.z * v.y + g.ntCol2.z * v.z + g.ntP.z };
}

// smallest gap between the two extreme (min or max end) values
static bool closeExtreme(std::vector<float> v, bool wantMin)
{
    if (v.size() < 2) return false;
    std::sort(v.begin(), v.end());
    return wantMin ? (v[1] - v[0] < MARG) : (v[v.size() - 1] - v[v.size() - 2] < MARG);
}

// reference for one group.  Returns marginal flag; fills total + out[6] (global).
bool refGroup(const uint32_t count[4], const F4 C[4][3], const GroupIn& g,
              uint32_t& total, F4 out[6])
{
    for (int i = 0; i < 6; ++i) out[i] = F4{ 9, 9, 9, 9 };
    const V3 fn = xyz(g.fn);
    total = count[0] + count[1] + count[2] + count[3];

    if (total <= 6)
    {
        uint32_t pos = 0;
        for (int lane = 0; lane < 4; ++lane)
            for (uint32_t i = 0; i < count[lane]; ++i)
            {
                const V3 gp = xform(g, xyz(C[lane][i]));
                out[pos++] = F4{ gp.x, gp.y, gp.z, C[lane][i].w };
            }
        return false; // case A never marginal
    }

    bool marginal = false;

    // --- deepest (min .w) ---
    float dW[4]; uint32_t dI[4];
    for (int lane = 0; lane < 4; ++lane)
    {
        dW[lane] = BIG; dI[lane] = 0;
        for (uint32_t i = 0; i < count[lane]; ++i) if (C[lane][i].w < dW[lane]) { dW[lane] = C[lane][i].w; dI[lane] = i; }
    }
    if (closeExtreme({ dW[0], dW[1], dW[2], dW[3] }, true)) marginal = true;
    float gm = std::min(std::min(dW[0], dW[1]), std::min(dW[2], dW[3]));
    uint32_t dwin = (dW[0] == gm) ? 0 : ((dW[1] == gm) ? 1 : ((dW[2] == gm) ? 2 : 3));
    { std::vector<float> ws; for (uint32_t i = 0; i < count[dwin]; ++i) ws.push_back(C[dwin][i].w); if (closeExtreme(ws, true)) marginal = true; }
    const F4 slot0 = C[dwin][dI[dwin]];
    const V3 deepest = xyz(slot0);

    // --- furthest from deepest (in the incident-face plane) ---
    const V3 dpp = sub(deepest, V3{ fn.x * dot3(deepest, fn), fn.y * dot3(deepest, fn), fn.z * dot3(deepest, fn) });
    float fW[4]; uint32_t fI[4];
    for (int lane = 0; lane < 4; ++lane)
    {
        fW[lane] = -BIG; fI[lane] = 0;
        for (uint32_t i = 0; i < count[lane]; ++i)
        {
            const V3 d = sub(dpp, xyz(C[lane][i])); const float ds = dot3(d, d);
            if (ds > fW[lane]) { fW[lane] = ds; fI[lane] = i; }
        }
    }
    if (closeExtreme({ fW[0], fW[1], fW[2], fW[3] }, false)) marginal = true;
    float mf = std::max(std::max(fW[0], fW[1]), std::max(fW[2], fW[3]));
    uint32_t fwin = (fW[0] == mf) ? 0 : ((fW[1] == mf) ? 1 : ((fW[2] == mf) ? 2 : 3));
    { std::vector<float> ds; for (uint32_t i = 0; i < count[fwin]; ++i) { const V3 dd = sub(dpp, xyz(C[fwin][i])); ds.push_back(dot3(dd, dd)); } if (closeExtreme(ds, false)) marginal = true; }
    const F4 slot1 = C[fwin][fI[fwin]];
    const V3 furthest = xyz(slot1);

    // --- two perpendicular extrema ---
    const V3 dir = crossp(fn, sub(furthest, deepest));
    float mnV[4], mxV[4]; uint32_t mnI[4], mxI[4];
    for (int lane = 0; lane < 4; ++lane)
    {
        mnV[lane] = BIG; mxV[lane] = -BIG; mnI[lane] = 0; mxI[lane] = 0;
        for (uint32_t i = 0; i < count[lane]; ++i)
        {
            const float dd = dot3(dir, sub(xyz(C[lane][i]), deepest));
            if (dd > mxV[lane]) { mxV[lane] = dd; mxI[lane] = i; }
            if (dd < mnV[lane]) { mnV[lane] = dd; mnI[lane] = i; }
        }
    }
    if (closeExtreme({ mnV[0], mnV[1], mnV[2], mnV[3] }, true)) marginal = true;
    if (closeExtreme({ mxV[0], mxV[1], mxV[2], mxV[3] }, false)) marginal = true;
    float gmn = std::min(std::min(mnV[0], mnV[1]), std::min(mnV[2], mnV[3]));
    uint32_t mnwin = (mnV[0] == gmn) ? 0 : ((mnV[1] == gmn) ? 1 : ((mnV[2] == gmn) ? 2 : 3));
    float gmx = std::max(std::max(mxV[0], mxV[1]), std::max(mxV[2], mxV[3]));
    uint32_t mxwin = (mxV[0] == gmx) ? 0 : ((mxV[1] == gmx) ? 1 : ((mxV[2] == gmx) ? 2 : 3));
    {
        std::vector<float> dmn, dmx;
        for (uint32_t i = 0; i < count[mnwin]; ++i) dmn.push_back(dot3(dir, sub(xyz(C[mnwin][i]), deepest)));
        for (uint32_t i = 0; i < count[mxwin]; ++i) dmx.push_back(dot3(dir, sub(xyz(C[mxwin][i]), deepest)));
        if (closeExtreme(dmn, true)) marginal = true;
        if (closeExtreme(dmx, false)) marginal = true;
    }
    const F4 slot2 = C[mnwin][mnI[mnwin]];
    const F4 slot3 = C[mxwin][mxI[mxwin]];

    const F4 slots[4] = { slot0, slot1, slot2, slot3 };
    for (int i = 0; i < 4; ++i) { const V3 gp = xform(g, xyz(slots[i])); out[i] = F4{ gp.x, gp.y, gp.z, slots[i].w }; }
    total = 4;
    return marginal;
}

void euler(float ax, float ay, float az, V3 col[3])
{
    const float cx = std::cos(ax), sx = std::sin(ax), cy = std::cos(ay), sy = std::sin(ay), cz = std::cos(az), sz = std::sin(az);
    col[0] = V3{ cy * cz, cy * sz, -sy };
    col[1] = V3{ sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy };
    col[2] = V3{ cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy };
}

PXG_TEST(npBoxReduce_compactContacts)
{
    std::mt19937 rng(52010);
    const uint32_t NG = 8000;
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<LaneIn> lin(N);
    std::vector<GroupIn> gin(NG);
    std::vector<uint8_t> skip(NG, 0);
    std::vector<uint32_t> rTotal(NG);
    std::vector<F4> rOut(NG * 6);

    size_t nSkip = 0, caseA = 0, caseB = 0;
    for (uint32_t gg = 0; gg < NG; ++gg)
    {
        uint32_t count[4]; F4 C[4][3];
        for (int lane = 0; lane < 4; ++lane)
        {
            count[lane] = rng() % 4; // 0..3
            LaneIn li{};
            F4* dst[3] = { &li.c0, &li.c1, &li.c2 };
            for (int i = 0; i < 3; ++i)
            {
                F4 c{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-1.0f, 0.1f) };
                C[lane][i] = c; *dst[i] = c;
            }
            li.meta = F4{ float(count[lane]), 0, 0, 0 };
            lin[gg * 4 + lane] = li;
        }
        V3 c0[3]; euler(rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), c0);
        V3 fnv{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        float fl = std::sqrt(dot3(fnv, fnv)) + 1e-6f; fnv = V3{ fnv.x / fl, fnv.y / fl, fnv.z / fl };
        GroupIn g{};
        g.ntCol0 = F4{ c0[0].x, c0[0].y, c0[0].z, 0 }; g.ntCol1 = F4{ c0[1].x, c0[1].y, c0[1].z, 0 }; g.ntCol2 = F4{ c0[2].x, c0[2].y, c0[2].z, 0 };
        g.ntP = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        g.fn = F4{ fnv.x, fnv.y, fnv.z, 0 };
        gin[gg] = g;

        uint32_t total; F4 out[6];
        const bool marg = refGroup(count, C, g, total, out);
        skip[gg] = marg ? 1u : 0u;
        rTotal[gg] = total;
        for (int i = 0; i < 6; ++i) rOut[gg * 6 + i] = out[i];
        if (marg) { ++nSkip; continue; }
        if (count[0] + count[1] + count[2] + count[3] > 6) ++caseB; else ++caseA;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outInit(NG * 6, F4{ 9, 9, 9, 9 });
    std::vector<uint32_t> totInit(NG, 999);
    PxgBufferHandle bLane = createBufferFrom(dev, lin);
    PxgBufferHandle bGroup = createBufferFrom(dev, gin);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bTot = createBufferFrom(dev, totInit);

    struct BoxReduceParams { uint32_t numGroups; } p = { NG };
    Kernel k("npBoxReduce.spv", "boxCompactContacts",
             { (uint32_t)sizeof(LaneIn), (uint32_t)sizeof(GroupIn), 16, 4 }, sizeof(BoxReduceParams));
    k.dispatch(GROUPS, { bLane, bGroup, bOut, bTot }, p);
    dev->flush();

    const auto gT = downloadBuffer<uint32_t>(dev, bTot, NG);
    const auto gC = downloadBuffer<F4>(dev, bOut, NG * 6);
    size_t mmTot = 0, mmContact = 0;
    for (uint32_t gg = 0; gg < NG; ++gg)
    {
        if (skip[gg]) continue;
        if (gT[gg] != rTotal[gg]) { ++mmTot; continue; }
        for (uint32_t i = 0; i < rTotal[gg]; ++i)
        {
            const F4& r = rOut[gg * 6 + i]; const F4& v = gC[gg * 6 + i];
            if (!almostEqual(v.x, r.x, 1e-4f, 1e-5f) || !almostEqual(v.y, r.y, 1e-4f, 1e-5f) ||
                !almostEqual(v.z, r.z, 1e-4f, 1e-5f) || !almostEqual(v.w, r.w, 1e-4f, 1e-5f)) ++mmContact;
        }
    }
    PXG_CHECK_MSG(mmTot == 0, "boxCompactContacts: %zu total mismatches", mmTot);
    PXG_CHECK_MSG(mmContact == 0, "boxCompactContacts: %zu contact-value mismatches", mmContact);
    PXG_CHECK_MSG(nSkip < NG / 4, "boxCompactContacts: too many marginal groups skipped (%zu/%u)", nSkip, NG);
    PXG_CHECK_MSG(caseA > NG / 10 && caseB > NG / 10, "boxCompactContacts: path mix thin (A=%zu B=%zu)", caseA, caseB);

    for (PxgBufferHandle h : { bLane, bGroup, bOut, bTot }) dev->destroyBuffer(h);
}

} // namespace
