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

// Verification of npCvxFinishContacts.slang: the finishContactsKernel tail
// (status flags, touch/patch change flags, allflags rewrite, patch reservation
// with strict-> overflow, rollback, contactPatches offsets, combineMaterials)
// plus writePatchesToStream4threads (patch payload assembly via the
// patchOffsets table).  One test = one warp (32 pairs).  All outputs exact
// (combine floats are single deterministic ops -> tight tolerance).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct F4 { float x, y, z, w; };
struct MaterialData { float dynamicFriction, staticFriction, restitution, damping; uint32_t flags, fricMode, restMode, dampMode; };
struct FinIn { U4 lane[32]; U4 lane2[32]; U4 cfg; };
struct FinOut { U4 lane[32]; U4 lane2[32]; F4 combined[32]; U4 counters; U4 patches[128]; };

static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static float combineScalars(float a, float b, uint32_t mode)
{
    if (mode == 0) return 0.5f * (a + b);
    if (mode == 1) return std::fmin(a, b);
    if (mode == 2) return a * b;
    if (mode == 3) return std::fmax(a, b);
    return 0.f;
}

static void combineMaterials(const MaterialData& m0, const MaterialData& m1,
                             uint32_t& matFlags, float& staF, float& dynF, float& rest, float& damp)
{
    const float r0 = m0.restitution, r1 = m1.restitution;
    const bool c0 = r0 < 0.f, c1 = r1 < 0.f;
    const bool oneC = c0 != c1, bothC = c0 && c1;
    const bool a0 = (m0.flags & 16u) != 0, a1 = (m1.flags & 16u) != 0;
    const bool oneA = a0 != a1;

    if (bothC && oneA) rest = a0 ? r0 : r1;
    else
    {
        const uint32_t mode = oneC ? 1u : std::max(m0.restMode, m1.restMode);
        const float flip = (bothC && mode == 2u) ? -1.f : 1.f;
        rest = flip * combineScalars(r0, r1, mode);
    }
    if (bothC && oneA) damp = a0 ? m0.damping : m1.damping;
    else
    {
        const uint32_t mode = oneC ? 3u : std::max(m0.dampMode, m1.dampMode);
        damp = combineScalars(m0.damping, m1.damping, mode);
    }
    const uint32_t cf = m0.flags | m1.flags;
    if ((cf & 1u) == 0)
    {
        const uint32_t mode = std::max(m0.fricMode, m1.fricMode);
        const float d = combineScalars(m0.dynamicFriction, m1.dynamicFriction, mode);
        const float s = combineScalars(m0.staticFriction, m1.staticFriction, mode);
        const float fd = std::fmax(d, 0.f);
        dynF = fd; staF = (s - fd) >= 0.f ? s : fd;
        matFlags = cf;
    }
    else { matFlags = cf | 2u; dynF = 0.f; staF = 0.f; }
}

static void refFinish(const FinIn& s, const std::vector<MaterialData>& mats, FinOut& o)
{
    const uint32_t limit = s.cfg.x;
    uint32_t patchesBytes = s.cfg.y, ovf = 0;
    for (int i = 0; i < 128; ++i) o.patches[i] = U4{ 0, 0, 0, 0 };

    uint32_t scratchNb[32], status[32], prevP[32], tc[32], pc[32], afOut[32], nbOut[32], cpOff[32];
    bool prevTouch[32];

    for (int L = 0; L < 32; ++L)
    {
        const bool valid = s.lane[L].z != 0;
        const uint32_t nb = valid ? s.lane[L].x : 0;
        scratchNb[L] = s.lane[L].x;
        const uint32_t af = valid ? s.lane[L].y : 0;
        const uint32_t oldSt = (af >> 16) & 0xFF;
        uint32_t st = (oldSt & ~3u) | (nb != 0 ? 2u : 1u);
        prevP[L] = (af & 0xFFFF) >> 8;
        prevTouch[L] = (oldSt & 2u) != 0;
        const bool prevKnown = (oldSt & 3u) != 0;
        tc[L] = 0; pc[L] = 0; afOut[L] = s.lane[L].y; nbOut[L] = 0; cpOff[L] = 0xFFFFFFFFu;
        if (valid)
        {
            const uint32_t nP = nb ? 1u : 0u;
            tc[L] = ((prevTouch[L] != (nb != 0)) || !prevKnown) ? 1u : 0u;
            pc[L] = (prevP[L] != nP) ? 1u : 0u;
            afOut[L] = (((prevP[L] << 8) | st) << 16) | (nP << 8);
            nbOut[L] = nb;
        }
        status[L] = st;
    }

    uint32_t mask = 0;
    for (int L = 0; L < 32; ++L)
        if ((s.lane[L].z != 0 ? s.lane[L].x : 0) > 0) mask |= (1u << L);

    uint32_t patchIndex = 0xFFFFFFFFu;
    if (mask)
    {
        const uint32_t totalSz = 64 * popc(mask);
        patchIndex = patchesBytes;
        patchesBytes += totalSz;
        if ((patchIndex + totalSz) > limit) { ovf |= 4u; patchIndex = 0xFFFFFFFFu; }
    }

    if (patchIndex == 0xFFFFFFFFu && mask)
    {
        mask = 0;
        for (int L = 0; L < 32; ++L)
        {
            scratchNb[L] = 0;
            if (s.lane[L].z != 0)
            {
                uint32_t st = (status[L] & ~3u) | 1u;
                afOut[L] = ((prevP[L] << 8) | st) << 16;
                nbOut[L] = 0;
                tc[L] = prevTouch[L] ? 1u : 0u;
                pc[L] = (prevP[L] != 0) ? 1u : 0u;
                status[L] = st;
            }
        }
    }

    for (int L = 0; L < 32; ++L)
        if (s.lane[L].z != 0 && patchIndex != 0xFFFFFFFFu)
            cpOff[L] = patchIndex + 64 * popc(mask & ((1u << L) - 1u));

    float rest[32], dynF[32], staF[32], damp[32];
    uint32_t mf[32];
    for (int L = 0; L < 32; ++L)
    {
        rest[L] = dynF[L] = staF[L] = damp[L] = 0.f; mf[L] = 0;
        if (s.lane[L].z != 0)
        {
            uint32_t f; float sf, df, re, da;
            combineMaterials(mats[s.lane[L].w], mats[s.lane2[L].x], f, sf, df, re, da);
            rest[L] = re; dynF[L] = df; staF[L] = sf; damp[L] = da;
            mf[L] = f & 0xFF;
        }
    }

    const uint32_t ONE = f2u(1.0f);
    for (int L = 0; L < 32; ++L)
        if (mask & (1u << L))
        {
            const uint32_t slot = popc(mask & ((1u << L) - 1u)), base = slot * 4;
            o.patches[base + 0] = U4{ ONE, ONE, ONE, ONE };
            o.patches[base + 1] = U4{ s.lane2[L].y, s.lane2[L].z, s.lane2[L].w, f2u(rest[L]) };
            o.patches[base + 2] = U4{ f2u(dynF[L]), f2u(staF[L]), f2u(damp[L]), ((scratchNb[L] & 0xFF) << 16) | (mf[L] << 24) };
            o.patches[base + 3] = U4{ (s.lane[L].w & 0xFFFF) << 16, s.lane2[L].x & 0xFFFF, 0, 0 };
        }

    for (int L = 0; L < 32; ++L)
    {
        o.lane[L] = U4{ tc[L], pc[L], afOut[L], nbOut[L] };
        o.lane2[L] = U4{ cpOff[L], mf[L], 0, 0 };
        o.combined[L] = F4{ rest[L], dynF[L], staF[L], damp[L] };
    }
    o.counters = U4{ patchesBytes, ovf, 0, 0 };
}

static bool eqF(float a, float b) { return almostEqual(a, b, 1e-6f, 1e-7f); }

PXG_TEST(npCvxFinishContacts_finish)
{
    std::mt19937 rng(152001);
    const uint32_t N = 8000, NMAT = 64;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<MaterialData> mats(NMAT);
    for (auto& m : mats)
    {
        m.dynamicFriction = rf(0, 1.5f); m.staticFriction = rf(0, 2.f);
        m.restitution = rf(-1.f, 1.f); // negatives = compliant
        m.damping = rf(0, 2.f);
        m.flags = (rng() % 4 == 0 ? 1u : 0u) | (rng() % 3 == 0 ? 16u : 0u) | (rng() % 5 == 0 ? 256u : 0u); // high bit exercises the u8 cast
        m.fricMode = rng() % 4; m.restMode = rng() % 4; m.dampMode = rng() % 4;
    }

    std::vector<FinIn> in(N);
    std::vector<FinOut> ref(N);
    size_t cOvf = 0, cExactFit = 0, cInvalid = 0, cEmpty = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        FinIn s{};
        uint32_t nPatch = 0;
        const bool emptyWarp = (i % 97 == 0); // no contacts anywhere: no reservation at all
        for (int L = 0; L < 32; ++L)
        {
            const uint32_t nb = emptyWarp ? 0u : ((rng() % 10 < 4) ? 0u : (1u + rng() % 5));
            const bool valid = !(i % 10 == 3 && L == int(i % 32));
            if (!valid) ++cInvalid;
            const uint32_t af = rng(); // random allflags word (status bits vary)
            s.lane[L] = U4{ nb, af, valid ? 1u : 0u, rng() % NMAT };
            s.lane2[L] = U4{ rng() % NMAT, f2u(rf(-1, 1)), f2u(rf(-1, 1)), f2u(rf(-1, 1)) };
            if (valid && nb > 0) ++nPatch;
        }
        const uint32_t init = 64 * (rng() % 32);
        uint32_t limit = init + 64 * 40; // ample
        if (i % 9 == 1) limit = init + 64 * (rng() % 8);        // tight -> overflow likely
        else if (i % 9 == 2) { limit = init + 64 * nPatch; ++cExactFit; } // exact fit: strict > must NOT overflow
        s.cfg = U4{ limit, init, 0, 0 };
        in[i] = s;
        refFinish(s, mats, ref[i]);
        if (ref[i].counters.y & 4u) ++cOvf;
        if (nPatch == 0) ++cEmpty;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<FinOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bMat = createBufferFrom(dev, mats);

    struct FinParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxFinishContacts.spv", "cvxFinishContacts",
             { (uint32_t)sizeof(FinIn), (uint32_t)sizeof(FinOut), (uint32_t)sizeof(MaterialData) }, sizeof(FinParams));
    k.dispatch(GROUPS, { bIn, bOut, bMat }, prm);
    dev->flush();

    const auto g = downloadBuffer<FinOut>(dev, bOut, N);
    size_t mmL = 0, mmP = 0, mmC = 0, mmF = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        for (int L = 0; L < 32; ++L)
        {
            const U4& rl = ref[i].lane[L]; const U4& gl = g[i].lane[L];
            const U4& rl2 = ref[i].lane2[L]; const U4& gl2 = g[i].lane2[L];
            if (rl.x != gl.x || rl.y != gl.y || rl.z != gl.z || rl.w != gl.w ||
                rl2.x != gl2.x || rl2.y != gl2.y) { ++mmL; break; }
        }
        for (int j = 0; j < 128; ++j)
        {
            const U4& rp = ref[i].patches[j]; const U4& gp = g[i].patches[j];
            if (rp.x != gp.x || rp.y != gp.y || rp.z != gp.z || rp.w != gp.w) { ++mmP; break; }
        }
        if (g[i].counters.x != ref[i].counters.x || g[i].counters.y != ref[i].counters.y) ++mmC;
        for (int L = 0; L < 32; ++L)
        {
            const F4& rc = ref[i].combined[L]; const F4& gc = g[i].combined[L];
            if (!eqF(rc.x, gc.x) || !eqF(rc.y, gc.y) || !eqF(rc.z, gc.z) || !eqF(rc.w, gc.w)) { ++mmF; break; }
        }
    }
    PXG_CHECK_MSG(mmL == 0, "finishContacts: %zu lane-output mismatches", mmL);
    PXG_CHECK_MSG(mmP == 0, "finishContacts: %zu patch-payload mismatches", mmP);
    PXG_CHECK_MSG(mmC == 0, "finishContacts: %zu counter mismatches", mmC);
    PXG_CHECK_MSG(mmF == 0, "finishContacts: %zu combined-material mismatches", mmF);
    PXG_CHECK_MSG(cOvf > N / 30 && cExactFit > N / 12 && cInvalid > N / 20 && cEmpty > 0,
                  "finishContacts: coverage thin (ovf %zu exactFit %zu invalid %zu empty %zu)",
                  cOvf, cExactFit, cInvalid, cEmpty);

    for (PxgBufferHandle h : { bIn, bOut, bMat }) dev->destroyBuffer(h);
}

} // namespace
