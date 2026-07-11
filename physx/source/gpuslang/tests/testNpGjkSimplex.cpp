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

// Verification of npGjkSimplex.slang: the GJK simplex normal accessors
// setNormal0 / setNormal123 (gjk.cuh :40-50).  One thread per test simulates the
// 32-lane warp; the resulting 32-lane simplex is compared exactly against a CPU
// transcription (splatV lane map / shfl_up-20 / the lane partition).  Pure
// lane-select copies of distinct per-lane payloads, so any mis-partition or
// mis-shuffle is caught bit-for-bit.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct SxParams { uint32_t numTests, pad0, pad1, pad2; };

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(npGjkSimplex_accessors)
{
    std::mt19937 rng(1210121);
    const uint32_t N = 600;

    std::vector<float> Z(N * 32), NV(N * 32);
    std::vector<U4> in(N);
    auto fv = [&]() -> float { return float(int(rng() % 4001) - 2000) * 0.25f; };
    for (uint32_t t = 0; t < N; ++t)
    {
        for (uint32_t L = 0; L < 32; ++L) { Z[t * 32 + L] = fv(); NV[t * 32 + L] = fv(); }
        in[t] = { rng() % 8u, 0, 0, 0 };
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bZ = createBufferFrom(dev, Z);
    PxgBufferHandle bNV = createBufferFrom(dev, NV);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    std::vector<float> outInit(N * 32, -777.0f);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    SxParams prm = { N, 0, 0, 0 };
    const std::vector<uint32_t> strides = { 4, 4, (uint32_t)sizeof(U4), 4 };

    auto dispatch = [&](const char* entry) -> std::vector<float>
    {
        Kernel k("npGjkSimplex.spv", entry, strides, sizeof(SxParams));
        k.dispatch((N + 63) / 64, { bZ, bNV, bIn, bOut }, prm);
        dev->flush();
        return downloadBuffer<float>(dev, bOut, N * 32);
    };

    size_t mm = 0;

    // ---- setNormal0 ----
    {
        const auto g = dispatch("ctGjkSetNormal0");
        for (uint32_t t = 0; t < N; ++t)
        {
            const uint32_t index = in[t].x;
            for (uint32_t L = 0; L < 32; ++L)
            {
                const uint32_t src = ((index << 2) | (L & 3u)) & 31u;
                const float e = (L < 16 || L > 19) ? Z[t * 32 + L] : NV[t * 32 + src];
                if (asU32(g[t * 32 + L]) != asU32(e)) ++mm;
            }
        }
    }

    // ---- setNormal123 ----
    {
        const auto g = dispatch("ctGjkSetNormal123");
        for (uint32_t t = 0; t < N; ++t)
            for (uint32_t L = 0; L < 32; ++L)
            {
                const float tmpV = (L >= 20) ? NV[t * 32 + (L - 20)] : NV[t * 32 + L];
                const float e = (L < 20) ? Z[t * 32 + L] : tmpV;
                if (asU32(g[t * 32 + L]) != asU32(e)) ++mm;
            }
    }

    PXG_CHECK_MSG(mm == 0, "gjkSimplex: %zu mismatches", mm);

    for (PxgBufferHandle h : { bZ, bNV, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
