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

// Verification of npClothDefGrad2D.slang (clothDefGrad2D): the FEM-cloth 2x2
// deformation-gradient primitives queryDeformationGradient_F2x2 and
// queryConstraintGradient_F2x2 (FEMClothUtil.cuh L299-317), against an exact
// CPU transcription.  Outputs bitwise modulo the documented +-0 (eqFZ).

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
struct DgIn  { F4 qinvDef, xp, qinvCon, pcpf; };
struct DgOut { F4 f, grad; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static void refDefGrad(const DgIn& s, DgOut& o)
{
    const F4 QInv = s.qinvDef;
    const float xp01x = s.xp.x, xp01y = s.xp.y, xp02x = s.xp.z, xp02y = s.xp.w;
    const float f0x = QInv.x * xp01x + QInv.y * xp02x;
    const float f0y = QInv.x * xp01y + QInv.y * xp02y;
    const float f1x = QInv.z * xp01x + QInv.w * xp02x;
    const float f1y = QInv.z * xp01y + QInv.w * xp02y;

    const F4 qInv = s.qinvCon;
    const float p0x = s.pcpf.x, p0y = s.pcpf.y, p1x = s.pcpf.z, p1y = s.pcpf.w;
    const float g1x = qInv.x * p0x + qInv.z * p1x;
    const float g1y = qInv.x * p0y + qInv.z * p1y;
    const float g2x = qInv.y * p0x + qInv.w * p1x;
    const float g2y = qInv.y * p0y + qInv.w * p1y;

    o.f = F4{ f0x, f0y, f1x, f1y };
    o.grad = F4{ g1x, g1y, g2x, g2y };
}

PXG_TEST(npClothDefGrad2D)
{
    std::mt19937 rng(208001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), rf(a, b) }; };

    std::vector<DgIn> in(N);
    std::vector<DgOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        DgIn s{};
        s.qinvDef = rv(-3, 3);
        s.xp = rv(-3, 3);
        s.qinvCon = rv(-3, 3);
        s.pcpf = rv(-3, 3);
        in[i] = s;
        refDefGrad(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DgOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothDefGrad2D.spv", "clothDefGrad2D",
             { (uint32_t)sizeof(DgIn), (uint32_t)sizeof(DgOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DgOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].f.x;
        const float* qfp = &g[i].f.x;
        const size_t nf = sizeof(DgOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u fR (%g %g %g %g) G (%g %g %g %g)\n", i, ref[i].f.x, ref[i].f.y, ref[i].f.z,
                            ref[i].f.w, g[i].f.x, g[i].f.y, g[i].f.z, g[i].f.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothDefGrad2D outputs bitwise-identical (mod +-0)");
}

}  // namespace
