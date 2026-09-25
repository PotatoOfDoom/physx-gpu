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

// Verification of npSdfHoles.slang (sdfHoles): the remaining SDF-construction
// portable units - bvhCalculateKeyDeltas (int XOR -> float CONVERSION),
// bvhCalculateKeyDeltasSquaredDistance, storeQuantized (u8/u16 truncating
// quantization) and the findHoles 3x3x3 compute core (SDFConstruction.cu
// L85-126, L536-556, L649-696) - against an exact CPU transcription.  The
// 27-value SDF neighborhood block collapses the global gathers.  Outputs
// bitwise (floats mod +-0; keys/quantized/flags exact).

#include "pxgTestFramework.h"

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
struct HoIn  { F4 misc, misc2, misc3, la, ua, lb, ub, sdf0, sdf1, sdf2, sdf3, sdf4, sdf5, sdf6; };
struct HoOut { F4 aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float pabs(float a) { return a < 0.f ? -a : a; }
static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }
static uint32_t uselMin(uint32_t a, uint32_t b) { return a < b ? a : b; }
static uint32_t uselMax(uint32_t a, uint32_t b) { return a < b ? b : a; }
static float psign(float a) { return (a >= 0.f) ? 1.0f : -1.0f; }

static float sdfAt(const HoIn& s, uint32_t idx)
{
    const float* base = &s.sdf0.x;
    return base[idx];
}

static void refHo(const HoIn& s, HoOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t mode = f2u(s.misc.x);

    if (mode == 0u)
    {
        const int32_t a = (int32_t)f2u(s.misc.w);
        const int32_t b = (int32_t)f2u(s.misc2.x);
        const int32_t x = a ^ b;
        o.aux = F4{ float(x), 0.f, 0.f, 0.f };
    }
    else if (mode == 1u)
    {
        const float cax = 0.5f * (s.la.x + s.ua.x), cay = 0.5f * (s.la.y + s.ua.y), caz = 0.5f * (s.la.z + s.ua.z);
        const float cbx = 0.5f * (s.lb.x + s.ub.x), cby = 0.5f * (s.lb.y + s.ub.y), cbz = 0.5f * (s.lb.z + s.ub.z);
        const float dx = cax - cbx, dy = cay - cby, dz = caz - cbz;
        o.aux = F4{ dx * dx + dy * dy + dz * dz, 0.f, 0.f, 0.f };
    }
    else if (mode == 2u)
    {
        const float v = s.misc.y;
        const uint32_t bytesPerPixel = f2u(s.misc.z);
        uint32_t q = 0u;
        if (bytesPerPixel == 1u)
            q = uint32_t(uint8_t(255.0f * selMin(1.0f, selMax(0.0f, v))));
        else if (bytesPerPixel == 2u)
            q = uint32_t(uint16_t(65535.0f * selMin(1.0f, selMax(0.0f, v))));
        o.aux = F4{ u2f(q), 0.f, 0.f, 0.f };
    }
    else
    {
        const uint32_t id = f2u(s.misc2.y);
        const uint32_t width = f2u(s.misc2.z);
        const uint32_t height = f2u(s.misc2.w);
        const uint32_t depth = f2u(s.misc3.x);
        const float csx = s.misc3.y, csy = s.misc3.z, csz = s.misc3.w;

        bool valueChanged = false;
        float newValue = 0.f;

        if (id < width * height * depth)
        {
            const float initialValue = sdfAt(s, 13u);
            newValue = pabs(initialValue);

            uint32_t rem = id;
            const uint32_t px = rem % width; rem /= width;
            const uint32_t py = rem % height;
            const uint32_t pz = rem / height;

            for (uint32_t z = uselMax(1u, pz) - 1u; z <= uselMin(depth - 1u, pz + 1u); ++z)
                for (uint32_t y = uselMax(1u, py) - 1u; y <= uselMin(height - 1u, py + 1u); ++y)
                    for (uint32_t x = uselMax(1u, px) - 1u; x <= uselMin(width - 1u, px + 1u); ++x)
                    {
                        if (x == px && y == py && z == pz)
                            continue;

                        const uint32_t index = (z * height + y) * width + x;
                        if (index >= width * height * depth)
                            continue;

                        const float value = sdfAt(s, (x - px + 1u) + 3u * (y - py + 1u) + 9u * (z - pz + 1u));

                        if (psign(initialValue) != psign(value))
                        {
                            float distance = 0.f;
                            if (x != px)
                                distance += csx * csx;
                            if (y != py)
                                distance += csy * csy;
                            if (z != pz)
                                distance += csz * csz;

                            distance = std::sqrt(distance);

                            const float delta = pabs(value - initialValue);

                            if (0.99f * delta > distance)
                            {
                                const float scaling = distance / delta;
                                const float v = 0.99f * scaling * initialValue;
                                newValue = selMin(newValue, pabs(v));
                            }
                        }
                    }

            if (initialValue < 0.f)
                newValue = -newValue;

            valueChanged = newValue != initialValue;
        }

        o.aux = F4{ newValue, valueChanged ? u2f(1u) : u2f(0u), 0.f, 0.f };
    }
}

PXG_TEST(npSdfHoles)
{
    std::mt19937 rng(262001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<HoIn> in(N);
    std::vector<HoOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        HoIn s{};
        const uint32_t mode = i % 4u;
        s.misc.x = u2f(mode);
        if (mode == 0u)
        {
            // adjacent Morton-style keys, incl. negative (priority) patterns
            uint32_t a = rng() & 0x3fffffffu;
            uint32_t b = rng() & 0x3fffffffu;
            // sign-bit stripe: only ONE key gets the priority bits, so the
            // XOR is negative and float(negative int) is exercised
            if ((i % 7u) == 2u) a |= 0xf8000000u;
            s.misc.w = u2f(a);
            s.misc2.x = u2f(b);
            // identical-keys stripe: XOR = 0 -> float(0)
            if ((i % 13u) == 6u) s.misc2.x = s.misc.w;
        }
        else if (mode == 1u)
        {
            s.la = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.ua = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.lb = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.ub = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            // identical-items stripe: distance 0
            if ((i % 13u) == 6u) { s.lb = s.la; s.ub = s.ua; }
        }
        else if (mode == 2u)
        {
            s.misc.y = rf(-0.5f, 1.5f);                       // both clamp arms
            s.misc.z = u2f(((i / 4u) % 2u) ? 2u : 1u);        // u8 / u16 alternating
            if ((i % 11u) == 4u) s.misc.y = 1.f;              // exact 1.0 (max quant)
            if ((i % 11u) == 8u) s.misc.y = 0.f;              // exact 0.0
        }
        else
        {
            const uint32_t width = 2u + rng() % 5u;
            const uint32_t height = 2u + rng() % 5u;
            const uint32_t depth = 2u + rng() % 5u;
            const uint32_t px = rng() % width;
            const uint32_t py = rng() % height;
            const uint32_t pz = rng() % depth;
            uint32_t id = (pz * height + py) * width + px;
            // out-of-domain stripe: id beyond the grid -> zero outputs
            if ((i % 17u) == 3u) id = width * height * depth + (rng() % 4u);
            s.misc2.y = u2f(id);
            s.misc2.z = u2f(width);
            s.misc2.w = u2f(height);
            s.misc3 = F4{ u2f(depth), rf(0.01f, 0.5f), rf(0.01f, 0.5f), rf(0.01f, 0.5f) };
            float* block = &s.sdf0.x;
            for (int b = 0; b < 27; ++b)
                block[b] = rf(-1.f, 1.f);
            // all-positive stripe: no sign changes, newValue = |initial|
            if ((i % 13u) == 6u)
                for (int b = 0; b < 27; ++b)
                    block[b] = pabs(block[b]);
            // tiny cells + large jumps: the 0.99*delta > distance guard fires
            if ((i % 7u) == 2u)
            {
                s.misc3.y = 0.001f; s.misc3.z = 0.001f; s.misc3.w = 0.001f;
                block[13] = ((i / 7u) % 2u) ? 0.9f : -0.9f;
            }
            // guard boundary window: one sign-changed face neighbor with
            // distance/delta ~ 0.995 in [0.99, 1) - the ONLY ratio band where
            // the 0.99 factor in the guard is observable (rejected by 0.99*
            // delta > distance, admitted by a 1.0* mutant)
            if ((i % 19u) == 5u)
            {
                const uint32_t w3 = 3u;
                s.misc2.y = u2f((1u * w3 + 1u) * w3 + 1u);   // id 13: center of 3x3x3
                s.misc2.z = u2f(w3); s.misc2.w = u2f(w3);
                s.misc3 = F4{ u2f(w3), 0.2f, 0.2f, 0.2f };
                for (int b = 0; b < 27; ++b)
                    block[b] = 0.5f;
                block[13] = 0.1f;
                block[14] = -0.101f;    // delta ~ 0.201, distance 0.2
            }
        }
        in[i] = s;
        refHo(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<HoOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSdfHoles.spv", "sdfHoles",
             { (uint32_t)sizeof(HoIn), (uint32_t)sizeof(HoOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<HoOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const bool ok = eqFZ(ref[i].aux.x, g[i].aux.x) && f2u(ref[i].aux.y) == f2u(g[i].aux.y);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u mode %u auxR (%g %08x) auxG (%g %08x)\n", i, i % 4u,
                            ref[i].aux.x, f2u(ref[i].aux.y), g[i].aux.x, f2u(g[i].aux.y));
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sdfHoles outputs bitwise-identical");
}

}  // namespace
