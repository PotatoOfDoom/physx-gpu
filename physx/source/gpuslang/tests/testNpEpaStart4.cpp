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

// Verification of npEpaStart4.slang: ctEpaStart4 — the EPA initial-tetrahedron
// setup (epa.cuh :254-278).  One thread per test simulates the 32-lane warp; the
// unpacked vertices/indices, the 4 normalised seed-triangle normals + distances
// and the fixed tetrahedron topology are compared bit-exactly against a CPU
// transcription.  Integer simplex data keeps dot(n,n) exact and m=1/sqrt(dot)
// correctly-rounded, so the unit normals and distances are bit-for-bit
// reproducible.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct EsParams { uint32_t numTests, pad0, pad1, pad2; };
struct EsOut { float verts[12]; uint32_t indices[4]; float normal[12]; float dist[4]; uint32_t a[4], v[4]; };

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static void dotW(const float a[32], const float b[32], float o[32])
{
    float x[32]; for (uint32_t L = 0; L < 32; ++L) x[L] = a[L] * b[L];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const float s1 = (L + 1 < 32) ? x[L + 1] : x[L];
        const float s2 = (L + 2 < 32) ? x[L + 2] : x[L];
        o[L] = x[L] + s1 + s2;
    }
}

static EsOut start4Ref(const float input[32])
{
    EsOut o;
    for (uint32_t L = 0; L < 16; ++L)
    {
        if ((L & 3u) < 3u) o.verts[L - L / 4u] = input[L];
        else o.indices[L >> 2] = asU32(input[L]);
    }
    float d[32]; dotW(input, input, d);
    float rs[32]; for (uint32_t L = 0; L < 32; ++L) rs[L] = 1.0f / std::sqrt(d[L]);
    for (uint32_t tI = 0; tI < 4; ++tI)
    {
        const float m = rs[tI * 4u + 16u];
        o.normal[tI * 3u + 0u] = input[tI * 4u + 16u] * m;
        o.normal[tI * 3u + 1u] = input[tI * 4u + 17u] * m;
        o.normal[tI * 3u + 2u] = input[tI * 4u + 18u] * m;
        o.dist[tI] = input[tI * 4u + 19u] * m;
    }
    o.a[0] = 0x010302u; o.a[1] = 0x020300u; o.a[2] = 0x030100u; o.a[3] = 0x010200u;
    o.v[0] = 0x020100u; o.v[1] = 0x000103u; o.v[2] = 0x010203u; o.v[3] = 0x020003u;
    return o;
}

PXG_TEST(npEpaStart4_setup)
{
    std::mt19937 rng(1300130);
    std::vector<float> Data;
    std::vector<EsOut> ref;

    auto ic = [&](int r) -> int { return int(rng() % (2 * r + 1)) - r; };

    for (uint32_t it = 0; it < 1500; ++it)
    {
        float input[32];
        // 4 vertices: xyz integer + index word (encoded 0x3f800000 | b<<8 | a)
        for (uint32_t k = 0; k < 4; ++k)
        {
            input[k * 4 + 0] = float(ic(20));
            input[k * 4 + 1] = float(ic(20));
            input[k * 4 + 2] = float(ic(20));
            input[k * 4 + 3] = asF32(0x3f800000u | ((rng() % 40u) << 8) | (rng() % 40u));
        }
        // 4 normals: xyz integer (nonzero) + distance integer
        for (uint32_t k = 0; k < 4; ++k)
        {
            int nx, ny, nz;
            do { nx = ic(8); ny = ic(8); nz = ic(8); } while (nx == 0 && ny == 0 && nz == 0);
            input[16 + k * 4 + 0] = float(nx);
            input[16 + k * 4 + 1] = float(ny);
            input[16 + k * 4 + 2] = float(nz);
            input[16 + k * 4 + 3] = float(ic(30));
        }
        for (uint32_t L = 0; L < 32; ++L) Data.push_back(input[L]);
        ref.push_back(start4Ref(input));
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bData = createBufferFrom(dev, Data);
    std::vector<EsOut> outInit(N);
    std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(EsOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    EsParams prm = { N, 0, 0, 0 };
    Kernel k("npEpaStart4.spv", "ctEpaStart4", { 4, (uint32_t)sizeof(EsOut) }, sizeof(EsParams));
    k.dispatch((N + 63) / 64, { bData, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<EsOut>(dev, bOut, N);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const EsOut& r = ref[t]; const EsOut& g = gOut[t];
        for (uint32_t i = 0; i < 12; ++i) { if (asU32(g.verts[i]) != asU32(r.verts[i])) ++mm; if (asU32(g.normal[i]) != asU32(r.normal[i])) ++mm; }
        for (uint32_t i = 0; i < 4; ++i)
        {
            if (g.indices[i] != r.indices[i]) ++mm;
            if (asU32(g.dist[i]) != asU32(r.dist[i])) ++mm;
            if (g.a[i] != r.a[i]) ++mm;
            if (g.v[i] != r.v[i]) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "epaStart4: %zu mismatches", mm);

    for (PxgBufferHandle h : { bData, bOut }) dev->destroyBuffer(h);
}

} // namespace
