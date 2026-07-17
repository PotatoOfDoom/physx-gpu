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

// Verification of npIsoAcosDecode.slang (isoAcosDecode): the isosurface
// approxAcos polynomial arccos (isosurfaceExtraction.cu L884-887) and
// getDecodedId packed-grid-id decode (L552-563), against an exact CPU
// transcription.  approxAcos is bitwise mod +-0; the decoded id is exact.

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
struct AdIn  { F4 misc; };
struct AdOut { F4 out; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float pmax(float a, float b) { return (a > b) ? a : b; }

static float approxAcos(float x)
{
    return pmax(0.0f, (-0.69813170079773212f * x * x - 0.87266462599716477f) * x + 1.5707963267948966f);
}

static uint32_t getDecodedId(uint32_t encoded, uint32_t dim)
{
    if ((encoded & (1u << (29u + dim))) == 0u)
        return 0xFFFFFFFFu;
    uint32_t id = encoded & 0x1FFFFFFFu;
    if (dim > 0u && (encoded & (1u << (29u + 0u))) != 0u)
        id++;
    if (dim > 1u && (encoded & (1u << (29u + 1u))) != 0u)
        id++;
    return id;
}

static void refAd(const AdIn& s, AdOut& o)
{
    const float x = s.misc.x;
    const uint32_t encoded = f2u(s.misc.y);
    const uint32_t dim = f2u(s.misc.z);
    o.out = F4{ approxAcos(x), u2f(getDecodedId(encoded, dim)), 0.f, 0.f };
}

PXG_TEST(npIsoAcosDecode)
{
    std::mt19937 rng(226001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AdIn> in(N);
    std::vector<AdOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AdIn s{};
        // x spans the valid cos range and beyond so the PxMax clamp (poly < 0
        // for |x| > ~1) is exercised at both ends.
        float x = rf(-1.6f, 1.6f);
        const uint32_t dim = i % 3u;
        // encoded: full 32-bit random, so bit 29+dim is set roughly half the time
        // (early-return 0xFFFFFFFF vs decode), and bits 29/30 drive the id++ steps.
        uint32_t encoded = rng();
        // Force-set stripe: guarantee the decode path with both increment bits set
        // for dim==2 so the id++ chain (dim>0 & bit29, dim>1 & bit30) is exercised.
        if ((i % 4u) == 1u)
        {
            encoded |= (1u << 31);            // bit 29+dim for dim=2
            encoded |= (1u << 29) | (1u << 30);
        }
        s.misc = F4{ x, u2f(encoded), u2f(dim), 0.f };
        in[i] = s;
        refAd(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AdOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npIsoAcosDecode.spv", "isoAcosDecode",
             { (uint32_t)sizeof(AdIn), (uint32_t)sizeof(AdOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AdOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // acos: bitwise mod +-0 ; decoded id: exact bit pattern (may be a NaN
        // pattern 0xFFFFFFFF, so compare raw bits, not float value).
        const bool ok = eqFZ(ref[i].out.x, g[i].out.x) && (f2u(ref[i].out.y) == f2u(g[i].out.y));
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dim %u acR %g acG %g idR %08x idG %08x\n", i, f2u(in[i].misc.z),
                            ref[i].out.x, g[i].out.x, f2u(ref[i].out.y), f2u(g[i].out.y));
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "isoAcosDecode outputs bitwise-identical (mod +-0)");
}

}  // namespace
