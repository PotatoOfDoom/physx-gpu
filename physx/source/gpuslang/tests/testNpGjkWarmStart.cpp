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

// Verification of npGjkWarmStart.slang: ctGjkWarmStart — the warm-start
// candidate selection of gjk() (gjk.cuh :282-294).  One thread per test
// simulates the 32-lane warp; the candidate mask, the updated cacheMask, the
// warmStarted flag, the chosen index, the updated simplex and the broadcast
// support vertex are compared bit-exactly against a CPU transcription.  Integer
// dV / simplex payloads keep the warp dot, the `< upperBound` predicate, the
// ballot mask and the splatV broadcast bit-deterministic.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct WsParams { uint32_t numTests, pad0, pad1, pad2; };
struct WsScalars { float upperBound; uint32_t cacheMask; uint32_t size; uint32_t pad; };

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

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
static uint32_t firstbitlow(uint32_t v) { uint32_t i = 0; while (!((v >> i) & 1u)) ++i; return i; }

struct Ref { float z[32], sup[32]; uint32_t candidates, cacheMaskOut, warmStarted, lowIdx; };
static Ref warmRef(const float dVin[32], const float zin[32], const WsScalars& s)
{
    Ref r;
    float dV[32], zV[32]; for (uint32_t L = 0; L < 32; ++L) { dV[L] = dVin[L]; zV[L] = zin[L]; }
    uint32_t cacheMask = s.cacheMask;

    float dd[32]; dotW(dV, zV, dd);
    bool p[32]; for (uint32_t L = 0; L < 32; ++L) p[L] = dd[L] < s.upperBound;

    uint32_t candidates = 0;
    if (cacheMask != 0)
    {
        for (uint32_t L = 0; L < 32; ++L) if (p[(L << 2) & 31u]) candidates |= (1u << L);
        candidates &= cacheMask;
    }

    for (uint32_t L = 0; L < 32; ++L) r.sup[L] = 0.0f;
    uint32_t warmStarted = 0, lowIdx = 0;
    if (candidates != 0)
    {
        warmStarted = 1;
        const uint32_t lowBit = candidates & (0u - candidates);
        cacheMask &= ~(lowBit * 2u - 1u);
        lowIdx = firstbitlow(candidates);
        for (uint32_t L = 0; L < 32; ++L) r.sup[L] = zV[((lowIdx << 2) | (L & 3u)) & 31u];
        for (uint32_t L = 0; L < 32; ++L) if ((L >> 2) == s.size) zV[L] = r.sup[L];
    }

    for (uint32_t L = 0; L < 32; ++L) r.z[L] = zV[L];
    r.candidates = candidates; r.cacheMaskOut = cacheMask; r.warmStarted = warmStarted; r.lowIdx = lowIdx;
    return r;
}

PXG_TEST(npGjkWarmStart_select)
{
    std::mt19937 rng(1280128);
    std::vector<float> DV, Z;
    std::vector<WsScalars> S;
    std::vector<Ref> ref;
    size_t nWarm = 0, nMask0 = 0, nNoCand = 0;

    auto ic = [&](int r) -> int { return int(rng() % (2 * r + 1)) - r; };

    auto emit = [&](WsScalars s, bool wantWarm)
    {
        float dV[32], z[32];
        // replicated integer dV (a search direction), w = 0
        const int dx = ic(4), dy = ic(4), dz = ic(4);
        for (uint32_t sl = 0; sl < 8; ++sl) { dV[sl * 4] = float(dx); dV[sl * 4 + 1] = float(dy); dV[sl * 4 + 2] = float(dz); dV[sl * 4 + 3] = 0.0f; }
        // distinct integer simplex vectors so splatV(lowIdx) is legible
        for (uint32_t k = 0; k < 8; ++k)
        {
            z[k * 4 + 0] = float(int(k) * 3 + 1 + ic(3));
            z[k * 4 + 1] = float(int(k) * 3 + 2 + ic(3));
            z[k * 4 + 2] = float(int(k) * 3 + 3 + ic(3));
            z[k * 4 + 3] = float(700 + k); // sentinel comp3
        }
        for (uint32_t L = 0; L < 32; ++L) { DV.push_back(dV[L]); Z.push_back(z[L]); }
        S.push_back(s);
        Ref r = warmRef(dV, z, s);
        ref.push_back(r);
        if (r.warmStarted) ++nWarm;
        else if (s.cacheMask == 0) ++nMask0;
        else ++nNoCand;
    };

    // family A: warm start likely (large upperBound -> most dots < ub), cacheMask over slots 0-3
    for (uint32_t it = 0; it < 700; ++it)
    {
        WsScalars s{}; s.upperBound = float(50 + rng() % 50); s.cacheMask = 1u + rng() % 15u; s.size = rng() % 5;
        emit(s, true);
    }
    // family B: cacheMask == 0 (never warm)
    for (uint32_t it = 0; it < 300; ++it)
    {
        WsScalars s{}; s.upperBound = float(int(rng() % 100) - 50); s.cacheMask = 0u; s.size = rng() % 5;
        emit(s, false);
    }
    // family C: small upperBound -> few/no dots below it (candidates often empty)
    for (uint32_t it = 0; it < 400; ++it)
    {
        WsScalars s{}; s.upperBound = float(int(rng() % 20) - 60); s.cacheMask = 1u + rng() % 15u; s.size = rng() % 5;
        emit(s, false);
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bDV = createBufferFrom(dev, DV);
    PxgBufferHandle bZ = createBufferFrom(dev, Z);
    PxgBufferHandle bS = createBufferFrom(dev, S);
    std::vector<float> zoInit(N * 32, -777.0f), supInit(N * 32, -777.0f);
    std::vector<uint32_t> outInit(N * 4, 0xEEEEEEEEu);
    PxgBufferHandle bZO = createBufferFrom(dev, zoInit);
    PxgBufferHandle bSup = createBufferFrom(dev, supInit);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    WsParams prm = { N, 0, 0, 0 };
    Kernel k("npGjkWarmStart.spv", "ctGjkWarmStart", { 4, 4, (uint32_t)sizeof(WsScalars), 4, 4, 4 }, sizeof(WsParams));
    k.dispatch((N + 63) / 64, { bDV, bZ, bS, bZO, bSup, bOut }, prm);
    dev->flush();

    const auto gZO = downloadBuffer<float>(dev, bZO, N * 32);
    const auto gSup = downloadBuffer<float>(dev, bSup, N * 32);
    const auto gOut = downloadBuffer<uint32_t>(dev, bOut, N * 4);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const Ref& r = ref[t];
        if (gOut[t * 4 + 0] != r.candidates) ++mm;
        if (gOut[t * 4 + 1] != r.cacheMaskOut) ++mm;
        if (gOut[t * 4 + 2] != r.warmStarted) ++mm;
        if (r.warmStarted && gOut[t * 4 + 3] != r.lowIdx) ++mm;
        for (uint32_t L = 0; L < 32; ++L)
        {
            if (asU32(gZO[t * 32 + L]) != asU32(r.z[L])) ++mm;
            if (r.warmStarted && asU32(gSup[t * 32 + L]) != asU32(r.sup[L])) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "gjkWarmStart: %zu mismatches", mm);
    PXG_CHECK_MSG(nWarm > 200 && nMask0 > 100 && nNoCand > 100,
                  "gjkWarmStart coverage thin (warm=%zu mask0=%zu noCand=%zu)", nWarm, nMask0, nNoCand);

    for (PxgBufferHandle h : { bDV, bZ, bS, bZO, bSup, bOut }) dev->destroyBuffer(h);
}

} // namespace
