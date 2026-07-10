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

// Verification of gridCal.slang (the particle hash-grid helpers from
// gpucommon/src/CUDA/gridCal.cuh).  The grid outputs are integers; the only
// float decision is calcGridPos's floor(pos / cellWidth), whose inputs are
// generated AWAY from cell boundaries (|pos/cw - round| > 1e-3) so the cell
// coordinate is deterministic on both sides — everything integer is compared
// bit-exactly.  The speculative contact volume (midpoint + radius) is
// continuous and compared to tolerance, with the PGS/TGS/diffuse scale
// branches all exercised.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct I4 { int32_t x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct GcIn { F4 pos_cw, bmin, bmax; I4 gridPosIn_flags, gridSize_off, periodSize; F4 curPos, predPos_cd; };
struct GcOut { I4 gridPos_hash, rangeMin_ph, rangeMax; U4 rangeSize, offsetInRange; F4 cVolume; };
struct GcParams { uint32_t numTests, pad0, pad1, pad2; };

struct I3 { int32_t x, y, z; };

static I3 refGridPos(float px, float py, float pz, float cw)
{
    return { (int32_t)std::floor(px / cw), (int32_t)std::floor(py / cw), (int32_t)std::floor(pz / cw) };
}
static uint32_t refHash(I3 gp, uint32_t sx, uint32_t sy, uint32_t sz)
{
    const int32_t x = gp.x & (int32_t)(sx - 1);
    const int32_t y = gp.y & (int32_t)(sy - 1);
    const int32_t z = gp.z & (int32_t)(sz - 1);
    (void)sz;
    return (uint32_t)((z * (int32_t)sy) * (int32_t)sx + y * (int32_t)sx + x);
}
static uint32_t refHashPeriodic(I3 gp, I3 gs, I3 ps)
{
    if (gp.x < 0) gp.x += ps.x; else if (gp.x >= ps.x) gp.x -= ps.x;
    if (gp.y < 0) gp.y += ps.y; else if (gp.y >= ps.y) gp.y -= ps.y;
    if (gp.z < 0) gp.z += ps.z; else if (gp.z >= ps.z) gp.z -= ps.z;
    return (uint32_t)((gp.z * gs.y) * gs.x + gp.y * gs.x + gp.x);
}
static int32_t clampi(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// keep pos/cw away from integers so floor is deterministic across FPUs
static float safeCoord(std::mt19937& rng, float cw)
{
    std::uniform_real_distribution<float> u(-40, 40);
    for (;;)
    {
        const float x = u(rng);
        const float q = x / cw;
        if (std::fabs(q - std::round(q)) > 1e-3f)
            return x;
    }
}

PXG_TEST(gridCal_all)
{
    std::mt19937 rng(81017);
    const uint32_t N = 8192;
    std::vector<GcIn> in(N);
    std::vector<GcOut> ref(N);
    size_t negCells = 0, wrapLo = 0, wrapHi = 0, clamped = 0, diffuse = 0, tgs = 0, pgs = 0, zeroRange = 0, unclampedZ = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        GcIn e;
        const float cw = 0.3f + 0.05f * (rng() % 40);
        e.pos_cw = { safeCoord(rng, cw), safeCoord(rng, cw), safeCoord(rng, cw), cw };

        // bounds with safe corners (min <= max); every 32nd entry INVERTS the x
        // bounds so calcWrappedGridRangeSize's clamp-to-0 branch fires (the
        // offsetInRange decode divides by that zero — undefined on both sides —
        // so those rows skip the offsetInRange comparison)
        const bool inverted = (i % 32 == 0);
        float b0 = safeCoord(rng, cw), b1 = safeCoord(rng, cw);
        if (b0 > b1) std::swap(b0, b1);
        if (inverted)
        {
            // make min's CELL strictly greater than max's
            std::swap(b0, b1);
            while (std::floor(b0 / cw) <= std::floor(b1 / cw)) b0 = safeCoord(rng, cw);
        }
        float c0 = safeCoord(rng, cw), c1 = safeCoord(rng, cw);
        if (c0 > c1) std::swap(c0, c1);
        float d0 = safeCoord(rng, cw), d1 = safeCoord(rng, cw);
        if (d0 > d1) std::swap(d0, d1);
        e.bmin = { b0, c0, d0, 0 };
        e.bmax = { b1, c1, d1, 0 };

        const uint32_t sx = 1u << (3 + rng() % 4), sy = 1u << (3 + rng() % 4), sz = 1u << (3 + rng() % 4);
        const I3 ps = { (int32_t)(6 + rng() % 20), (int32_t)(6 + rng() % 20), (int32_t)(6 + rng() % 20) };
        // periodic gridPos within [-ps, 2*ps): the single-wrap contract
        const I3 gpp = { (int32_t)(rng() % (3 * ps.x)) - ps.x, (int32_t)(rng() % (3 * ps.y)) - ps.y, (int32_t)(rng() % (3 * ps.z)) - ps.z };
        const uint32_t flags = rng() % 4; // 0 = PGS, 1 = diffuse, 2 = TGS, 3 = diffuse+TGS (diffuse wins)
        e.gridPosIn_flags = { gpp.x, gpp.y, gpp.z, (int32_t)flags };
        e.gridSize_off = { (int32_t)sx, (int32_t)sy, (int32_t)sz, (int32_t)(rng() % 4096) };
        e.periodSize = { ps.x, ps.y, ps.z, 0 };

        std::uniform_real_distribution<float> u(-5, 5);
        e.curPos = { u(rng), u(rng), u(rng), 0 };
        e.predPos_cd = { u(rng), u(rng), u(rng), 0.01f + 0.2f * (rng() % 100) / 100.0f };
        in[i] = e;

        // reference
        GcOut r;
        const I3 gp = refGridPos(e.pos_cw.x, e.pos_cw.y, e.pos_cw.z, cw);
        r.gridPos_hash = { gp.x, gp.y, gp.z, (int32_t)refHash(gp, sx, sy, sz) };
        if (gp.x < 0 || gp.y < 0 || gp.z < 0) ++negCells;
        const I3 rMin = refGridPos(e.bmin.x, e.bmin.y, e.bmin.z, cw);
        const I3 rMax = refGridPos(e.bmax.x, e.bmax.y, e.bmax.z, cw);
        r.rangeMin_ph = { rMin.x, rMin.y, rMin.z, (int32_t)refHashPeriodic(gpp, { (int32_t)sx, (int32_t)sy, (int32_t)sz }, ps) };
        r.rangeMax = { rMax.x, rMax.y, rMax.z, 0 };
        if (gpp.x < 0 || gpp.y < 0 || gpp.z < 0) ++wrapLo;
        if (gpp.x >= ps.x || gpp.y >= ps.y || gpp.z >= ps.z) ++wrapHi;
        const uint32_t rsx = (uint32_t)clampi(rMax.x - rMin.x + 1, 0, (int32_t)sx);
        const uint32_t rsy = (uint32_t)clampi(rMax.y - rMin.y + 1, 0, (int32_t)sy);
        const uint32_t rsz = (uint32_t)clampi(rMax.z - rMin.z + 1, 0, (int32_t)sz);
        r.rangeSize = { rsx, rsy, rsz, 0 };
        if (rMax.x - rMin.x + 1 > (int32_t)sx || rMax.y - rMin.y + 1 > (int32_t)sy || rMax.z - rMin.z + 1 > (int32_t)sz) ++clamped;
        const uint32_t off = (uint32_t)e.gridSize_off.w;
        if (rsx == 0)
            r.offsetInRange = { 0, 0, 0, 1 }; // marker: skip (division by zero-range is undefined on both sides)
        else
        {
            r.offsetInRange = { off % rsx, (off / rsx) % rsy, off / (rsx * rsy), 0 };
            if (off / (rsx * rsy) >= rsz) ++unclampedZ;
        }
        if (rsx == 0) ++zeroRange;

        // diffuse (bit 0) takes priority over TGS (bit 1), as in the CUDA
        const float scaleInv = (flags & 1) ? 1.0f : ((flags & 2) ? 2.0f : 1.0f);
        if (flags & 1) ++diffuse; else if (flags & 2) ++tgs; else ++pgs;
        const float ox = (e.predPos_cd.x - e.curPos.x) * scaleInv * 0.5f;
        const float oy = (e.predPos_cd.y - e.curPos.y) * scaleInv * 0.5f;
        const float oz = (e.predPos_cd.z - e.curPos.z) * scaleInv * 0.5f;
        r.cVolume = { e.curPos.x + ox, e.curPos.y + oy, e.curPos.z + oz,
                      std::sqrt(ox * ox + oy * oy + oz * oz) + e.predPos_cd.w };
        ref[i] = r;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<GcOut> outInit(N);
    PxgBufferHandle bI = createBufferFrom(dev, in);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);

    GcParams prm = { N, 0, 0, 0 };
    Kernel k("gridCal.spv", "ctGridCal",
             { (uint32_t)sizeof(GcIn), (uint32_t)sizeof(GcOut) }, sizeof(GcParams));
    k.dispatch(GROUPS, { bI, bO }, prm);
    dev->flush();

    const auto g = downloadBuffer<GcOut>(dev, bO, N);
    size_t mmInt = 0, mmVol = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const GcOut& r = ref[i]; const GcOut& v = g[i];
        if (v.gridPos_hash.x != r.gridPos_hash.x || v.gridPos_hash.y != r.gridPos_hash.y ||
            v.gridPos_hash.z != r.gridPos_hash.z || v.gridPos_hash.w != r.gridPos_hash.w ||
            v.rangeMin_ph.x != r.rangeMin_ph.x || v.rangeMin_ph.y != r.rangeMin_ph.y ||
            v.rangeMin_ph.z != r.rangeMin_ph.z || v.rangeMin_ph.w != r.rangeMin_ph.w ||
            v.rangeMax.x != r.rangeMax.x || v.rangeMax.y != r.rangeMax.y || v.rangeMax.z != r.rangeMax.z ||
            v.rangeSize.x != r.rangeSize.x || v.rangeSize.y != r.rangeSize.y || v.rangeSize.z != r.rangeSize.z ||
            (r.offsetInRange.w == 0 && (v.offsetInRange.x != r.offsetInRange.x || v.offsetInRange.y != r.offsetInRange.y ||
                                        v.offsetInRange.z != r.offsetInRange.z))) ++mmInt;
        if (!almostEqual(v.cVolume.x, r.cVolume.x, 1e-5f, 1e-5f) || !almostEqual(v.cVolume.y, r.cVolume.y, 1e-5f, 1e-5f) ||
            !almostEqual(v.cVolume.z, r.cVolume.z, 1e-5f, 1e-5f) || !almostEqual(v.cVolume.w, r.cVolume.w, 1e-5f, 1e-5f)) ++mmVol;
    }
    PXG_CHECK_MSG(mmInt == 0, "gridCal: %zu integer mismatches", mmInt);
    PXG_CHECK_MSG(mmVol == 0, "gridCal: %zu volume mismatches", mmVol);
    PXG_CHECK_MSG(negCells > 1000 && wrapLo > 1000 && wrapHi > 1000 && clamped > 200 && diffuse > 2000 && tgs > 1000 && pgs > 1000 && zeroRange > 100 && unclampedZ > 100,
                  "gridCal: coverage thin (neg=%zu lo=%zu hi=%zu clamp=%zu d=%zu t=%zu p=%zu z0=%zu uz=%zu)",
                  negCells, wrapLo, wrapHi, clamped, diffuse, tgs, pgs, zeroRange, unclampedZ);

    dev->destroyBuffer(bI); dev->destroyBuffer(bO);
}

} // namespace
