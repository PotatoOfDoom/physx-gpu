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

// Verification of npArtiJointHeader.slang (ajhPGS / ajhTGS): the jointHeader
// stores of setupArtiSolverConstraintBlockGPU (artiConstraintPrep2.cu
// L1716-1767 PGS, L1770-1830 TGS) against an exact CPU transcription.
// Integer lanes (rowCounts/breakable/writeBackOffset, TGS packed uchar4)
// memcmp; float lanes bitwise modulo the documented +-0 class (eqFZ).

#include "pxgTestFramework.h"

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

static const float PX_MAX_F32 = 3.402823466e+38f;

struct JhIn
{
    F4 invMassScale;
    F4 raWorld;
    F4 rbWorld;
    U4 misc;   // writeBackIndex, numRows, outCount, 0
    F4 scal;   // dt, cfm0, cfm1, 0
};

struct JhPgOut
{
    U4 hdr;    // rowCounts, breakable, writeBackOffset, 0
    F4 b0wo;
    F4 v0;
    F4 v1;
};

struct JhTgOut
{
    U4 hdr;    // packed(rowCounts|breakable<<8), writeBackOffset, 0, 0
    F4 rA;
    F4 rB;
    F4 v0;
    F4 v1;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float pxMaxR(float a, float b) { return (a < b) ? b : a; }

static uint32_t breakableR(float linBreakForce, float angBreakForce)
{
    return ((linBreakForce != PX_MAX_F32) || (angBreakForce != PX_MAX_F32)) ? 1u : 0u;
}

// exact transcription of setupArtiSolverConstraintBlockGPU PGS header stores
static void refAjhPGS(const JhIn& s, JhPgOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const F4 ims = s.invMassScale;
    const float dt = s.scal.x;
    const float linBreakForce = s.raWorld.w;
    const float angBreakForce = s.rbWorld.w;
    const float linBreakImpulse = linBreakForce * dt;
    const float angBreakImpulse = angBreakForce * dt;
    const float cfm = pxMaxR(s.scal.y, s.scal.z);

    o.hdr = U4{ s.misc.z, breakableR(linBreakForce, angBreakForce), s.misc.x, 0u };
    o.b0wo = F4{ s.raWorld.x, s.raWorld.y, s.raWorld.z, linBreakImpulse };
    o.v0 = F4{ angBreakImpulse, -ims.x, ims.z, -ims.y };
    o.v1 = F4{ ims.w, cfm, 0.f, 0.f };
}

// exact transcription of setupArtiSolverConstraintBlockGPU TGS header stores
static void refAjhTGS(const JhIn& s, JhTgOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const F4 ims = s.invMassScale;
    const float dt = s.scal.x;
    const float linBreakForce = s.raWorld.w;
    const float angBreakForce = s.rbWorld.w;
    const float linBreakImpulse = linBreakForce * dt;
    const float angBreakImpulse = angBreakForce * dt;
    const float cfm = pxMaxR(s.scal.y, s.scal.z);

    const uint32_t outCount = s.misc.z & 0xFFu;
    const uint32_t breakable = breakableR(linBreakForce, angBreakForce);
    const uint32_t packed = outCount | (breakable << 8);

    o.hdr = U4{ packed, s.misc.x, 0u, 0u };
    o.rA = F4{ s.raWorld.x, s.raWorld.y, s.raWorld.z, -ims.x };
    o.rB = F4{ s.rbWorld.x, s.rbWorld.y, s.rbWorld.z, ims.z };
    o.v0 = F4{ -ims.y, ims.w, linBreakImpulse, angBreakImpulse };
    o.v1 = F4{ cfm, 0.f, 0.f, 0.f };
}

static void fillFixtures(std::mt19937& rng, std::vector<JhIn>& in, size_t& cBrk0, size_t& cBrkRa,
                         size_t& cBrkAng, size_t& cBrkBoth, size_t& cDegen)
{
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    for (auto& s : in)
    {
        s.invMassScale = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f) };

        // break forces: independently PX_MAX_F32 (not breakable on that lane)
        // or finite, salted across the four disjunction cases
        const int bcase = static_cast<int>(rng() % 4);
        const float lbf = (bcase == 0 || bcase == 2) ? PX_MAX_F32 : rf(1.f, 1e6f);
        const float abf = (bcase == 0 || bcase == 1) ? PX_MAX_F32 : rf(1.f, 1e6f);
        if (bcase == 0) ++cBrk0;
        else if (bcase == 1) ++cBrkRa;   // raW finite, angBF MAX
        else if (bcase == 2) ++cBrkAng;  // raW MAX, angBF finite
        else ++cBrkBoth;
        s.raWorld = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), lbf };
        s.rbWorld = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), abf };

        const uint32_t numRows = rng() % 21u;
        // outCount <= numRows; often strictly less (degenerate rows) so the
        // rowCounts = outCount (not numRows) linkage is observable
        const uint32_t outCount = numRows ? (rng() % (numRows + 1u)) : 0u;
        if (outCount != numRows) ++cDegen;
        s.misc = U4{ rng(), numRows, outCount, 0u };
        s.scal = F4{ rf(0.001f, 0.02f), rf(0.f, 50.f), rf(0.f, 50.f), 0.f };
    }
}

PXG_TEST(npArtiJointHeader_pgs)
{
    std::mt19937 rng(184001);
    const uint32_t N = 2000;
    std::vector<JhIn> in(N);
    std::vector<JhPgOut> ref(N);
    size_t cBrk0 = 0, cBrkRa = 0, cBrkAng = 0, cBrkBoth = 0, cDegen = 0;
    fillFixtures(rng, in, cBrk0, cBrkRa, cBrkAng, cBrkBoth, cDegen);
    for (uint32_t i = 0; i < N; ++i)
        refAjhPGS(in[i], ref[i]);

    std::printf("[ info ] ajhP cov: brk0 %zu brkRa %zu brkAng %zu brkBoth %zu degen %zu\n", cBrk0,
                cBrkRa, cBrkAng, cBrkBoth, cDegen);
    PXG_CHECK(cBrk0 > 300);
    PXG_CHECK(cBrkRa > 300);
    PXG_CHECK(cBrkAng > 300);
    PXG_CHECK(cBrkBoth > 300);
    PXG_CHECK(cDegen > 1500);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<JhTgOut> dummyTg(1);
    std::vector<JhPgOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bPg = createBufferFrom(dev, outInit);
    PxgBufferHandle bTg = createBufferFrom(dev, dummyTg);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiJointHeader.spv", "ajhPGS",
             { (uint32_t)sizeof(JhIn), (uint32_t)sizeof(JhPgOut), (uint32_t)sizeof(JhTgOut) },
             sizeof(P));
    k.dispatch(GROUPS, { bIn, bPg, bTg }, prm);
    dev->flush();

    const auto g = downloadBuffer<JhPgOut>(dev, bPg, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const JhPgOut& r = ref[i];
        const JhPgOut& q = g[i];
        bool ok = std::memcmp(&r.hdr, &q.hdr, sizeof(U4)) == 0;
        const float* rfp = &r.b0wo.x;
        const float* qfp = &q.b0wo.x;
        for (size_t x = 0; x < 12 && ok; ++x)
            ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u rcR %u rcG %u brkR %u brkG %u\n", i, r.hdr.x, q.hdr.x, r.hdr.y,
                            q.hdr.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "ajhPGS header stores bitwise-identical (mod +-0)");
}

PXG_TEST(npArtiJointHeader_tgs)
{
    std::mt19937 rng(184002);
    const uint32_t N = 2000;
    std::vector<JhIn> in(N);
    std::vector<JhTgOut> ref(N);
    size_t cBrk0 = 0, cBrkRa = 0, cBrkAng = 0, cBrkBoth = 0, cDegen = 0;
    fillFixtures(rng, in, cBrk0, cBrkRa, cBrkAng, cBrkBoth, cDegen);
    for (uint32_t i = 0; i < N; ++i)
        refAjhTGS(in[i], ref[i]);

    std::printf("[ info ] ajhT cov: brk0 %zu brkRa %zu brkAng %zu brkBoth %zu degen %zu\n", cBrk0,
                cBrkRa, cBrkAng, cBrkBoth, cDegen);
    PXG_CHECK(cBrk0 > 300);
    PXG_CHECK(cBrkBoth > 300);
    PXG_CHECK(cDegen > 1500);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<JhPgOut> dummyPg(1);
    std::vector<JhTgOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bPg = createBufferFrom(dev, dummyPg);
    PxgBufferHandle bTg = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiJointHeader.spv", "ajhTGS",
             { (uint32_t)sizeof(JhIn), (uint32_t)sizeof(JhPgOut), (uint32_t)sizeof(JhTgOut) },
             sizeof(P));
    k.dispatch(GROUPS, { bIn, bPg, bTg }, prm);
    dev->flush();

    const auto g = downloadBuffer<JhTgOut>(dev, bTg, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const JhTgOut& r = ref[i];
        const JhTgOut& q = g[i];
        bool ok = std::memcmp(&r.hdr, &q.hdr, sizeof(U4)) == 0;
        const float* rfp = &r.rA.x;
        const float* qfp = &q.rA.x;
        for (size_t x = 0; x < 16 && ok; ++x)
            ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u packR %u packG %u\n", i, r.hdr.x, q.hdr.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "ajhTGS header stores bitwise-identical (mod +-0)");
}

}  // namespace
