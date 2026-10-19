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

// Verification of npCvxWriteStream.slang: one iteration of
// writeContactsToStream4threads (cudaGJKEPA.cu L1622-1875).  8 pairs x 4 lanes:
// flip/radius/contactDist setup, manifold invalidation-bit handling, group
// average normal (butterfly + 1e-5 fallback), world pointB + keep ballot,
// average-point insertion, warp-wide byte reservation with overflow flags,
// per-lane compaction, cmOutputs offsets and scratch params.  The reference
// transcribes the CUDA; integer outputs exact, floats to tolerance, keep/
// normalize boundary near-ties margin-skipped.

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
struct V3 { float x, y, z; };

struct Pair { U4 meta; F4 scale0, scale1, tq[2], tp[2], normPen[4], pointB[4]; U4 misc; };
struct WsIn { Pair pairs[8]; U4 cfg, counters; };
struct PairOut { F4 avgNorm; U4 outA, material; };
struct WsOut { F4 stream[40]; U4 counters; PairOut pairs[8]; };

const float FLTMAX = 3.4028235e38f;
const float TIE = 2e-4f;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRot(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }

static void refWriteStream(const WsIn& s, WsOut& o, bool& marg)
{
    const bool insertAvg = s.cfg.x != 0;
    const uint32_t cLimit = s.cfg.y, fLimit = s.cfg.z;
    uint32_t cBytes = s.counters.x, fBytes = s.counters.y, oFlags = 0;

    for (int i = 0; i < 40; ++i) o.stream[i] = F4{ 0, 0, 0, 0 };

    bool valid[8], flip[8];
    float radius[8], cd[8];
    F4 t1q[8]; V3 t1p[8];
    uint32_t num0[8], mNbAfter[8], mat0[8], mat1[8];

    for (int g = 0; g < 8; ++g)
    {
        valid[g] = s.pairs[g].misc.y != 0;
        flip[g] = false; radius[g] = 0; cd[g] = 0;
        t1q[g] = F4{ 0, 0, 0, 1 }; t1p[g] = V3{ 0, 0, 0 };
        mat0[g] = 0; mat1[g] = 0;
        num0[g] = 0; mNbAfter[g] = s.pairs[g].misc.x;

        if (valid[g])
        {
            uint32_t type0 = s.pairs[g].meta.x, type1 = s.pairs[g].meta.z;
            mat0[g] = s.pairs[g].meta.y; mat1[g] = s.pairs[g].meta.w;
            V3 sc0 = v3(s.pairs[g].scale0), sc1 = v3(s.pairs[g].scale1);
            uint32_t r0 = 0, r1 = 1;
            flip[g] = type1 < type0;
            if (flip[g]) { std::swap(type0, type1); std::swap(sc0, sc1); std::swap(r0, r1); }
            if (type0 == 0 || type0 == 2) radius[g] = sc0.y; // sphere / capsule
            t1q[g] = s.pairs[g].tq[r1];
            t1p[g] = v3(s.pairs[g].tp[r1]);
            cd[g] = s.pairs[g].tp[r0].w + s.pairs[g].tp[r1].w;

            const uint32_t raw = s.pairs[g].misc.x;
            mNbAfter[g] = raw & ~0x80000000u;
            num0[g] = (raw & 0x80000000u) ? 0u : (raw & ~0x80000000u);
        }
    }

    F4 npL[32], pbL[32];
    for (int L = 0; L < 32; ++L)
    {
        const int g = L / 4, j = L % 4;
        npL[L] = (uint32_t(j) < num0[g]) ? s.pairs[g].normPen[j] : F4{ 0, 0, 0, 0 };
        F4 pb{ FLTMAX, FLTMAX, FLTMAX, FLTMAX };
        if (uint32_t(j) < num0[g])
        {
            const float sep = npL[L].w - radius[g];
            const V3 p = add(qRot(t1q[g], v3(s.pairs[g].pointB[j])), t1p[g]);
            pb = F4{ p.x, p.y, p.z, sep };
        }
        pbL[L] = pb;
    }

    for (int g = 0; g < 8; ++g)
    {
        const V3 sum = add(add(v3(npL[4 * g]), v3(npL[4 * g + 2])), add(v3(npL[4 * g + 1]), v3(npL[4 * g + 3])));
        const float magSq = dot3(sum, sum);
        if (std::fabs(magSq - 1e-5f) < 1e-7f) marg = true;
        V3 avg = (magSq >= 1e-5f) ? mul(sum, 1.f / std::sqrt(magSq)) : v3(npL[4 * g]);
        avg = qRot(t1q[g], avg);
        if (flip[g]) avg = mul(avg, -1.f);
        o.pairs[g].avgNorm = F4{ avg.x, avg.y, avg.z, 0 };
    }

    uint32_t keepW = 0;
    for (int L = 0; L < 32; ++L)
    {
        const int g = L / 4;
        if (pbL[L].w != FLTMAX && std::fabs(pbL[L].w - cd[g]) < TIE) marg = true;
        if (pbL[L].w <= cd[g]) keepW |= (1u << L);
    }

    uint32_t numKept[8];
    for (int g = 0; g < 8; ++g) numKept[g] = popc(keepW & (0xFu << (4 * g)));

    uint32_t extraW = 0;
    for (int g = 0; g < 8; ++g)
        if (insertAvg && numKept[g] > 1) extraW |= (1u << (4 * g));

    uint32_t cOff0 = 0xFFFFFFFFu, fOff0 = 0xFFFFFFFFu;
    const uint32_t total = popc(keepW) + popc(extraW);
    if (total)
    {
        cOff0 = cBytes; cBytes += 16 * total;
        fOff0 = fBytes; fBytes += 4 * total;
        if ((cOff0 + 16 * total) >= cLimit) { oFlags |= 1u; cOff0 = 0xFFFFFFFFu; }
        else if ((fOff0 + 4 * total) >= fLimit) { oFlags |= 2u; fOff0 = 0xFFFFFFFFu; cOff0 = 0xFFFFFFFFu; } // 5.12: cudaGJKEPA.cu:1780
    }

    const uint32_t streamBase = s.counters.x;
    for (int g = 0; g < 8; ++g)
    {
        uint32_t numKeptG = numKept[g];
        uint32_t cbLane0 = cOff0, fbLane0 = fOff0;

        if (cOff0 != 0xFFFFFFFFu)
        {
            // CUDA butterfly association: (p0+p2)+(p1+p3) over lane-index-masked terms
            F4 pv[4];
            for (uint32_t j = 0; j < 4; ++j)
                pv[j] = (j < numKeptG) ? pbL[4 * g + j] : F4{ 0, 0, 0, 0 };
            const F4 avgPt{ (pv[0].x + pv[2].x) + (pv[1].x + pv[3].x),
                            (pv[0].y + pv[2].y) + (pv[1].y + pv[3].y),
                            (pv[0].z + pv[2].z) + (pv[1].z + pv[3].z),
                            (pv[0].w + pv[2].w) + (pv[1].w + pv[3].w) };

            for (uint32_t j = 0; j < 4; ++j)
            {
                const uint32_t L = 4 * g + j;
                const uint32_t mask = (1u << L) - 1u;
                const uint32_t byteOffset = popc(keepW & mask) + popc(extraW & mask);
                const uint32_t cb = cOff0 + 16 * byteOffset;
                const uint32_t fb = fOff0 + 4 * byteOffset;
                if (j == 0) { cbLane0 = cb; fbLane0 = fb; }

                uint32_t offset = 0;
                if (insertAvg && numKeptG > 1)
                {
                    if (j == 0)
                    {
                        const float inv = 1.f / float(numKeptG);
                        o.stream[(cb - streamBase) / 16] = F4{ avgPt.x * inv, avgPt.y * inv, avgPt.z * inv, avgPt.w * inv };
                        offset += 16;
                    }
                }
                if (pbL[L].w <= cd[g])
                    o.stream[(cb + offset - streamBase) / 16] = pbL[L];
            }
        }
        else numKeptG = 0;

        uint32_t cbOut = cbLane0, fbOut = fbLane0;
        if (numKeptG == 0) { cbOut = 0xFFFFFFFFu; fbOut = 0xFFFFFFFFu; }
        o.pairs[g].outA = U4{ 0, 0, 0, mNbAfter[g] };
        if (valid[g]) { o.pairs[g].outA.y = cbOut; o.pairs[g].outA.z = fbOut; }
        o.pairs[g].outA.x = numKeptG + ((numKeptG > 1 && insertAvg) ? 1u : 0u);
        o.pairs[g].material = U4{ mat0[g], mat1[g], 0, 0 };
    }

    o.counters = U4{ cBytes, fBytes, oFlags, 0 };
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 2e-4f, 2e-5f) && almostEqual(a.y, b.y, 2e-4f, 2e-5f) && almostEqual(a.z, b.z, 2e-4f, 2e-5f) && almostEqual(a.w, b.w, 2e-4f, 2e-4f); }

PXG_TEST(npCvxWriteStream_write)
{
    std::mt19937 rng(151001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<WsIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<WsOut> ref(N);
    size_t nSkip = 0, cOvf = 0, fOvf = 0, cInval = 0, cFlip = 0, cAvg = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        WsIn s{};
        for (int g = 0; g < 8; ++g)
        {
            Pair& p = s.pairs[g];
            p.meta = U4{ rng() % 8, rng() % 512, rng() % 8, rng() % 512 };
            p.scale0 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
            p.scale1 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
            for (int k = 0; k < 2; ++k)
            {
                p.tq[k] = randQuat();
                p.tp[k] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.05f, 0.4f) }; // .w = contactDistance
            }
            uint32_t nb = rng() % 5;
            // cancellation fixture: exactly two opposing normals sum to zero,
            // driving the magSq < 1e-5 fallback (avgNorm = lane 0's own normal)
            const bool cancelG = (i % 17 == 0) && (g == int((i / 17) % 8));
            if (cancelG) nb = 2;
            uint32_t raw = nb;
            if (i % 7 == 0 && g == int(i % 8)) { raw |= 0x80000000u; ++cInval; }
            const bool validG = !(i % 11 == 0 && g == int((i / 11) % 8));
            p.misc = U4{ raw, validG ? 1u : 0u, 0, 0 };
            for (int j = 0; j < 4; ++j)
            {
                const V3 n = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
                // seps straddling the keep threshold
                p.normPen[j] = F4{ n.x, n.y, n.z, rf(-0.6f, 0.9f) };
                p.pointB[j] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
            }
            if (cancelG)
                p.normPen[1] = F4{ -p.normPen[0].x, -p.normPen[0].y, -p.normPen[0].z, p.normPen[1].w };
            if (p.meta.z < p.meta.x) ++cFlip;
        }
        const bool insertAvg = (i % 2 == 0);
        if (insertAvg) ++cAvg;
        // limits: mostly ample; ~8% contact overflow; ~8% force overflow
        uint32_t cInit = 16 * (rng() % 64), fInit = 4 * (rng() % 64);
        uint32_t cLimit = cInit + 16 * 64, fLimit = fInit + 4 * 64;
        if (i % 13 == 0) { cLimit = cInit + 16 * (rng() % 12); if (i % 26 == 0) fLimit = fInit + 4 * (rng() % 12); }
        else if (i % 13 == 1) { fLimit = fInit + 4 * (rng() % 12); }
        s.cfg = U4{ insertAvg ? 1u : 0u, cLimit, fLimit, 0 };
        s.counters = U4{ cInit, fInit, 0, 0 };
        in[i] = s;

        bool marg = false;
        refWriteStream(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (ref[i].counters.z & 1u) ++cOvf;
        if (ref[i].counters.z & 2u) ++fOvf;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<WsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct WsParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxWriteStream.spv", "cvxWriteStream",
             { (uint32_t)sizeof(WsIn), (uint32_t)sizeof(WsOut) }, sizeof(WsParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<WsOut>(dev, bOut, N);
    size_t mmS = 0, mmC = 0, mmP = 0, mmN = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        for (int j = 0; j < 40; ++j) if (!eqF4(ref[i].stream[j], g[i].stream[j])) { ++mmS; break; }
        if (g[i].counters.x != ref[i].counters.x || g[i].counters.y != ref[i].counters.y ||
            g[i].counters.z != ref[i].counters.z) ++mmC;
        for (int p = 0; p < 8; ++p)
        {
            const PairOut& rp = ref[i].pairs[p];
            const PairOut& gp = g[i].pairs[p];
            if (gp.outA.x != rp.outA.x || gp.outA.y != rp.outA.y || gp.outA.z != rp.outA.z ||
                gp.outA.w != rp.outA.w || gp.material.x != rp.material.x || gp.material.y != rp.material.y) { ++mmP; break; }
        }
        for (int p = 0; p < 8; ++p)
            if (!eqF4(ref[i].pairs[p].avgNorm, g[i].pairs[p].avgNorm)) { ++mmN; break; }
    }
    PXG_CHECK_MSG(mmS == 0, "writeStream: %zu stream mismatches", mmS);
    PXG_CHECK_MSG(mmC == 0, "writeStream: %zu counter mismatches", mmC);
    PXG_CHECK_MSG(mmP == 0, "writeStream: %zu pair-output mismatches", mmP);
    PXG_CHECK_MSG(mmN == 0, "writeStream: %zu avgNorm mismatches", mmN);
    PXG_CHECK_MSG(nSkip < N / 3, "writeStream: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cOvf > N / 50 && fOvf > N / 50 && cInval > N / 20 && cFlip > N && cAvg > N / 3,
                  "writeStream: coverage thin (cOvf %zu fOvf %zu inval %zu flip %zu avg %zu)",
                  cOvf, fOvf, cInval, cFlip, cAvg);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
