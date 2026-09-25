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

// Verification of npSbComputeLambda.slang (sbComputeLambda): the two soft-body
// computeLambda overloads (softBodyGM.cu L606-617), composing
// multiplyStrainByCompliance, against an exact CPU transcription.  Bitwise mod +-0.

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
struct ClIn  { F4 params, params2, flags, gc0, gc1, gc2, ec0, ec1, ec2, ltop, lbot; };
struct ClOut { F4 scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };

static float evalStiffness(uint32_t row, uint32_t col, float invE, float nu)
{
    float result = 0.0f;
    if (row == col)
    {
        result = invE;
        if (row >= 3u) result = result * (1.0f + nu);
    }
    else if (row < 3u && col < 3u)
    {
        result = -invE * nu;
    }
    return result;
}

static float multiplyStrainByCompliance(float invE, float nu, uint32_t si, V3 lt, V3 lb)
{
    return lt.x * evalStiffness(si, 0u, invE, nu)
         + lt.y * evalStiffness(si, 1u, invE, nu)
         + lt.z * evalStiffness(si, 2u, invE, nu)
         + lb.x * evalStiffness(si, 3u, invE, nu)
         + lb.y * evalStiffness(si, 4u, invE, nu)
         + lb.z * evalStiffness(si, 5u, invE, nu);
}

static float comp3(V3 v, uint32_t i) { return (i == 0u) ? v.x : (i == 1u) ? v.y : v.z; }
static float matElem(V3 c0, V3 c1, V3 c2, uint32_t row, uint32_t col)
{
    const V3 c = (col == 0u) ? c0 : (col == 1u) ? c1 : c2;
    return comp3(c, row);
}

static void refCl(const ClIn& s, ClOut& o)
{
    const float gamma = s.params.x, scale = s.params.y, invE = s.params.z, nu = s.params.w;
    const float invDtSq = s.params2.x;
    const uint32_t indexA = f2u(s.params2.y), indexB = f2u(s.params2.z), strainIndex = f2u(s.params2.w);
    const bool useStrain = f2u(s.flags.x) == 1u;

    const V3 gc0 = V3{ s.gc0.x, s.gc0.y, s.gc0.z }, gc1 = V3{ s.gc1.x, s.gc1.y, s.gc1.z }, gc2 = V3{ s.gc2.x, s.gc2.y, s.gc2.z };
    const V3 ec0 = V3{ s.ec0.x, s.ec0.y, s.ec0.z }, ec1 = V3{ s.ec1.x, s.ec1.y, s.ec1.z }, ec2 = V3{ s.ec2.x, s.ec2.y, s.ec2.z };
    const V3 lt = V3{ s.ltop.x, s.ltop.y, s.ltop.z }, lb = V3{ s.lbot.x, s.lbot.y, s.lbot.z };

    const float Gab = matElem(gc0, gc1, gc2, indexA, indexB);
    const float Eab = matElem(ec0, ec1, ec2, indexA, indexB);
    const float mult = multiplyStrainByCompliance(invE, nu, strainIndex, lt, lb);

    float lambda;
    if (useStrain)
        lambda = -(Gab + invDtSq * mult + gamma * Eab) * scale;
    else
        lambda = -(Gab + gamma * Eab) * scale;

    o.scalars = F4{ lambda, mult, 0.f, 0.f };
}

PXG_TEST(npSbComputeLambda)
{
    std::mt19937 rng(235001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ClIn> in(N);
    std::vector<ClOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        ClIn s{};
        s.params = F4{ rf(-2.f, 2.f), rf(0.1f, 3.f), rf(0.01f, 5.f), rf(0.f, 0.49f) };  // gamma, scale, invE, nu
        const uint32_t indexA = i % 3u, indexB = (i / 3u) % 3u, strainIndex = i % 6u;
        s.params2 = F4{ rf(10.f, 200.f), u2f(indexA), u2f(indexB), u2f(strainIndex) };  // invDtSq, a, b, si
        s.flags = F4{ u2f((i % 2u) == 0u ? 1u : 0u), 0.f, 0.f, 0.f };                   // useStrain
        s.gc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.gc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.gc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.ec0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.ec1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.ec2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.ltop = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.lbot = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        in[i] = s;
        refCl(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ClOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbComputeLambda.spv", "sbComputeLambda",
             { (uint32_t)sizeof(ClIn), (uint32_t)sizeof(ClOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ClOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].scalars.x, g[i].scalars.x) && eqFZ(ref[i].scalars.y, g[i].scalars.y);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u lamR %g lamG %g multR %g multG %g\n", i,
                            ref[i].scalars.x, g[i].scalars.x, ref[i].scalars.y, g[i].scalars.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbComputeLambda outputs bitwise-identical (mod +-0)");
}

}  // namespace
