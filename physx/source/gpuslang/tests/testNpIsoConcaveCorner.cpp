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

// Verification of npIsoConcaveCorner.slang (isoConcaveCorner): the isosurface
// concave-corner search directionSign + indexOfConcaveCorner
// (isosurfaceExtraction.cu L741-783), against an exact CPU transcription.  The
// emitted directionSign values are bitwise mod +-0; the integer index is exact.

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
struct I4 { int32_t x, y, z, w; };
struct IcIn  { F4 a, b, c, d, misc; };
struct IcOut { F4 signs; I4 result; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const int projections[3][2] = { {1, 2}, {2, 0}, {0, 1} };

static float comp(const F4& v, int i) { return (i == 0) ? v.x : (i == 1) ? v.y : v.z; }

static float directionSign(int pd, const F4& start, const F4& middle, const F4& end)
{
    const int p0 = projections[pd][0];
    const int p1 = projections[pd][1];
    const float a0 = comp(middle, p0) - comp(start, p0);
    const float a1 = comp(middle, p1) - comp(start, p1);
    const float b0 = comp(end, p0) - comp(middle, p0);
    const float b1 = comp(end, p1) - comp(middle, p1);
    return a0 * b1 - a1 * b0;
}

static void refCorner(const IcIn& s, IcOut& o)
{
    const int pd = int(f2u(s.misc.x));

    const float sABC = directionSign(pd, s.a, s.b, s.c);
    const float sBCD = directionSign(pd, s.b, s.c, s.d);
    const float sCDA = directionSign(pd, s.c, s.d, s.a);
    const float sDAB = directionSign(pd, s.d, s.a, s.b);

    float minimum = 0.f;
    int result = -1;
    if (sABC <= minimum) { minimum = sABC; result = 1; }
    if (sBCD <= minimum) { minimum = sBCD; result = 2; }
    if (sCDA <= minimum) { minimum = sCDA; result = 3; }
    if (sDAB <= minimum) { minimum = sDAB; result = 0; }

    o.signs = F4{ sABC, sBCD, sCDA, sDAB };
    o.result = I4{ result, 0, 0, 0 };
}

PXG_TEST(npIsoConcaveCorner)
{
    std::mt19937 rng(224001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<IcIn> in(N);
    std::vector<IcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        IcIn s{};
        s.a = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.b = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.c = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.d = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        const uint32_t pd = i % 3u;   // cover all three principal directions
        s.misc = F4{ u2f(pd), 0.f, 0.f, 0.f };

        // Convex-quad stripe: a roughly convex planar quad in the projected plane
        // so all four directionSigns share a sign -> result stays -1 (no concave
        // corner) when they are all > 0, exercising the "nothing <= 0" path.
        if ((i % 4u) == 1u)
        {
            // build a CCW quad in the (x,y) plane; use pd=2 so projection is (x,y)
            s.a = F4{ -1.f, -1.f, rf(-3.f, 3.f), 0.f };
            s.b = F4{  1.f, -1.f, rf(-3.f, 3.f), 0.f };
            s.c = F4{  1.f,  1.f, rf(-3.f, 3.f), 0.f };
            s.d = F4{ -1.f,  1.f, rf(-3.f, 3.f), 0.f };
            s.misc = F4{ u2f(2u), 0.f, 0.f, 0.f };
        }
        // Collinear stripe: a,b,c collinear in the (x,y) projection so
        // directionSign(a,b,c) == 0 EXACTLY, with the other three positive.  This
        // pins the `<= minimum` boundary (result=1 at the 0 tie), which a `<`
        // mutant would miss (leaving result=-1), and emits an exact +0 sign.
        else if ((i % 13u) == 7u)
        {
            s.a = F4{ 0.f, 0.f, rf(-3.f, 3.f), 0.f };
            s.b = F4{ 1.f, 0.f, rf(-3.f, 3.f), 0.f };
            s.c = F4{ 2.f, 0.f, rf(-3.f, 3.f), 0.f };
            s.d = F4{ 1.f, 1.f, rf(-3.f, 3.f), 0.f };
            s.misc = F4{ u2f(2u), 0.f, 0.f, 0.f };
        }

        in[i] = s;
        refCorner(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<IcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npIsoConcaveCorner.spv", "isoConcaveCorner",
             { (uint32_t)sizeof(IcIn), (uint32_t)sizeof(IcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<IcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].signs.x, g[i].signs.x) && eqFZ(ref[i].signs.y, g[i].signs.y)
               && eqFZ(ref[i].signs.z, g[i].signs.z) && eqFZ(ref[i].signs.w, g[i].signs.w)
               && ref[i].result.x == g[i].result.x;
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u pd %u resR %d resG %d signsR (%g %g %g %g) signsG (%g %g %g %g)\n", i,
                            f2u(in[i].misc.x), ref[i].result.x, g[i].result.x,
                            ref[i].signs.x, ref[i].signs.y, ref[i].signs.z, ref[i].signs.w,
                            g[i].signs.x, g[i].signs.y, g[i].signs.z, g[i].signs.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "isoConcaveCorner outputs bitwise-identical (mod +-0)");
}

}  // namespace
