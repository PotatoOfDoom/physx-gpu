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

// Verification of npWarpVec.slang: the "8 four-vectors in a warp" layout
// primitives (nputils.cuh :66-113).  One thread per test simulates the 32-lane
// warp; each primitive (dot, cross, splatV, splatX, packXFlags/allX/anyX) is
// dispatched over shared integer-valued fixtures and compared exactly against a
// literal CPU transcription (shfl -> arr[s&31], shfl_down -> own-value-if-OOB,
// ballot -> bit loop).  Integer / signed-permutation lane values keep every
// product, dot sum, cross difference and equality exact, so the full 32-lane
// output (and the packed mask) match bit-for-bit.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct WvParams { uint32_t numTests, pad0, pad1, pad2; };

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(npWarpVec_prims)
{
    std::mt19937 rng(1200120);
    const uint32_t N = 900;

    // shared fixtures: 32 lane values a,b per test + scalars (cp0,cp1,index,Ncnt)
    std::vector<float> A(N * 32), B(N * 32);
    std::vector<U4> in(N);
    auto iv = [&]() -> float { return float(int(rng() % 41) - 20); };
    for (uint32_t t = 0; t < N; ++t)
    {
        for (uint32_t L = 0; L < 32; ++L)
        {
            // ~1/4 of x-flag lanes forced to 0 so packXFlags patterns vary
            A[t * 32 + L] = ((L & 3u) == 0u && (rng() % 4u == 0u)) ? 0.0f : iv();
            B[t * 32 + L] = iv();
        }
        uint32_t cp0 = rng() % 34, cp1 = rng() % 34; // include a few >=32 (wrap)
        in[t] = { cp0, cp1, rng() % 8u, 1u + rng() % 8u };
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bA = createBufferFrom(dev, A);
    PxgBufferHandle bB = createBufferFrom(dev, B);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    std::vector<float> outFInit(N * 32, -777.0f);
    std::vector<uint32_t> outUInit(N, 0xEEEEEEEEu);
    PxgBufferHandle bOutF = createBufferFrom(dev, outFInit);
    PxgBufferHandle bOutU = createBufferFrom(dev, outUInit);
    WvParams prm = { N, 0, 0, 0 };
    const std::vector<uint32_t> strides = { 4, 4, (uint32_t)sizeof(U4), 4, 4 };

    // every entry fully overwrites its output block, so no inter-dispatch reset
    // is needed
    auto dispatchF = [&](const char* entry) -> std::vector<float>
    {
        Kernel k("npWarpVec.spv", entry, strides, sizeof(WvParams));
        k.dispatch((N + 63) / 64, { bA, bB, bIn, bOutF, bOutU }, prm);
        dev->flush();
        return downloadBuffer<float>(dev, bOutF, N * 32);
    };

    size_t mm = 0;

    // ---- dot ----
    {
        const auto g = dispatchF("ctWarpDot");
        for (uint32_t t = 0; t < N; ++t)
        {
            float x[32];
            for (uint32_t L = 0; L < 32; ++L) x[L] = A[t * 32 + L] * B[t * 32 + L];
            for (uint32_t L = 0; L < 32; ++L)
            {
                const float s1 = (L + 1 < 32) ? x[L + 1] : x[L];
                const float s2 = (L + 2 < 32) ? x[L + 2] : x[L];
                const float e = x[L] + s1 + s2;
                if (asU32(g[t * 32 + L]) != asU32(e)) ++mm;
            }
        }
    }
    // ---- cross ----
    {
        const auto g = dispatchF("ctWarpCross");
        for (uint32_t t = 0; t < N; ++t)
        {
            const uint32_t cp0 = in[t].x & 31u, cp1 = in[t].y & 31u;
            const float e = A[t * 32 + cp0] * B[t * 32 + cp1] - A[t * 32 + cp1] * B[t * 32 + cp0];
            for (uint32_t L = 0; L < 32; ++L) if (asU32(g[t * 32 + L]) != asU32(e)) ++mm;
        }
    }
    // ---- splatV ----
    {
        const auto g = dispatchF("ctWarpSplatV");
        for (uint32_t t = 0; t < N; ++t)
        {
            const uint32_t index = in[t].z;
            for (uint32_t L = 0; L < 32; ++L)
            {
                const uint32_t src = ((index << 2) | (L & 3u)) & 31u;
                if (asU32(g[t * 32 + L]) != asU32(A[t * 32 + src])) ++mm;
            }
        }
    }
    // ---- splatX ----
    {
        const auto g = dispatchF("ctWarpSplatX");
        for (uint32_t t = 0; t < N; ++t)
            for (uint32_t L = 0; L < 32; ++L)
                if (asU32(g[t * 32 + L]) != asU32(A[t * 32 + (L & 28u)])) ++mm;
    }

    // ---- packXFlags / allX / anyX ----
    size_t nAll = 0, nAny = 0, nNone = 0;
    {
        Kernel k("npWarpVec.spv", "ctWarpPackX", strides, sizeof(WvParams));
        k.dispatch((N + 63) / 64, { bA, bB, bIn, bOutF, bOutU }, prm);
        dev->flush();
        const auto g = downloadBuffer<uint32_t>(dev, bOutU, N);
        for (uint32_t t = 0; t < N; ++t)
        {
            const uint32_t Nc = in[t].w;
            uint32_t mask = 0;
            for (uint32_t i = 0; i < Nc; ++i) if (A[t * 32 + (i << 2)] != 0.0f) mask |= (1u << i);
            mask &= (1u << Nc) - 1u;
            const uint32_t allx = (mask == ((1u << Nc) - 1u)) ? 1u : 0u;
            const uint32_t anyx = (mask != 0u) ? 1u : 0u;
            const uint32_t e = mask | (allx << 8) | (anyx << 9);
            if (g[t] != e) ++mm;
            if (allx) ++nAll; if (!anyx) ++nNone; else ++nAny;
        }
    }

    PXG_CHECK_MSG(mm == 0, "warpVec: %zu mismatches", mm);
    PXG_CHECK_MSG(nAll > 30 && nAny > 200 && nNone > 20,
                  "warpVec packX coverage thin (all=%zu any=%zu none=%zu)", nAll, nAny, nNone);

    for (PxgBufferHandle h : { bA, bB, bIn, bOutF, bOutU }) dev->destroyBuffer(h);
}

} // namespace
