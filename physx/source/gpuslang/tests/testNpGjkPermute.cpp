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

// Verification of npGjkPermute.slang: ctGjkPermute — the per-iteration loop-glue
// of gjk() (gjk.cuh :336-363).  One thread per test simulates the 32-lane warp.
// The permuted 32-lane simplex plus (newSize, degenerate, backupSize,
// upperBoundOut) are compared bit-exactly against a CPU transcription.  Distinct
// integer simplex payloads (vector v holds v*10+comp) let the feature-code-driven
// gather be checked exactly; cm/upperBound are integer-valued floats so the
// comparisons and the passthrough of upperBound/cm are bit-deterministic.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct PmParams { uint32_t numTests, pad0, pad1, pad2; };
struct PmScalars { float cm; float upperBound; uint32_t size; uint32_t f; };
static const uint32_t DEGENERATE = 0xffffu;

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct Ref { float z[32]; uint32_t newSize, degenerate, backupSize, ub; };
static Ref permuteRef(const float zin[32], const PmScalars& s)
{
    Ref r;
    float zV[32]; for (uint32_t L = 0; L < 32; ++L) zV[L] = zin[L];
    const uint32_t newSize = s.f & 0xfu;
    const bool degenerate = (s.f == DEGENERATE) || (s.cm >= s.upperBound && newSize <= s.size);

    if (degenerate)
    {
        for (uint32_t L = 0; L < 32; ++L) r.z[L] = zV[L];
        r.newSize = newSize; r.degenerate = 1u; r.backupSize = 0u; r.ub = asU32(s.upperBound);
        return r;
    }

    const uint32_t backupSize = (s.cm >= s.upperBound) ? s.size : 0u;
    const float upperBoundOut = (backupSize != 0u) ? s.upperBound : s.cm;

    float tmp[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t sourceIndex = (s.f >> ((L & 28u) + 4u)) & 0xfu;
        tmp[L] = zV[((sourceIndex << 2) | (L & 3u)) & 31u];
    }
    for (uint32_t L = 0; L < 32; ++L) zV[L] = (L < 12u) ? tmp[L] : zV[L];

    for (uint32_t L = 0; L < 32; ++L) r.z[L] = zV[L];
    r.newSize = newSize; r.degenerate = 0u; r.backupSize = backupSize; r.ub = asU32(upperBoundOut);
    return r;
}

PXG_TEST(npGjkPermute_iteration)
{
    std::mt19937 rng(1260126);
    std::vector<float> Z;
    std::vector<PmScalars> S;
    std::vector<Ref> ref;
    size_t nDegF = 0, nDegCond = 0, nBackup = 0, nAdvance = 0;

    auto emit = [&](const float z[32], PmScalars s)
    {
        for (uint32_t L = 0; L < 32; ++L) Z.push_back(z[L]);
        S.push_back(s);
        Ref r = permuteRef(z, s);
        ref.push_back(r);
        if (r.degenerate) { if (s.f == DEGENERATE) ++nDegF; else ++nDegCond; }
        else if (r.backupSize) ++nBackup; else ++nAdvance;
    };

    auto mkZ = [&](float z[32])
    {
        // vector v (lanes 4v..4v+3) holds v*10 + comp, so the gather source is legible
        for (uint32_t L = 0; L < 32; ++L) z[L] = float((L >> 2) * 10u + (L & 3u));
    };
    auto mkF = [&](uint32_t newSize) -> uint32_t
    {
        const uint32_t i0 = rng() % 8, i1 = rng() % 8, i2 = rng() % 8;
        return (i2 << 12) | (i1 << 8) | (i0 << 4) | (newSize & 0xfu);
    };

    // family A: explicit DEGENERATE feature code
    for (uint32_t it = 0; it < 200; ++it)
    {
        float z[32]; mkZ(z);
        PmScalars s; s.cm = float(rng() % 20); s.upperBound = float(rng() % 20);
        s.size = 1u + rng() % 3; s.f = DEGENERATE;
        emit(z, s);
    }
    // family B: degenerate via cm>=upperBound && newSize<=size
    for (uint32_t it = 0; it < 300; ++it)
    {
        float z[32]; mkZ(z);
        const uint32_t size = 3u;
        const uint32_t newSize = 1u + rng() % 3;    // <= size
        PmScalars s; s.upperBound = float(1 + rng() % 10); s.cm = s.upperBound + float(rng() % 5); // cm>=ub
        s.size = size; s.f = mkF(newSize);
        emit(z, s);
    }
    // family C: non-degenerate backup (cm>=upperBound but newSize>size)
    for (uint32_t it = 0; it < 300; ++it)
    {
        float z[32]; mkZ(z);
        const uint32_t size = 1u + rng() % 2;       // 1 or 2
        const uint32_t newSize = size + 1u + rng() % 2; // > size
        PmScalars s; s.upperBound = float(1 + rng() % 10); s.cm = s.upperBound + float(rng() % 5);
        s.size = size; s.f = mkF(newSize);
        emit(z, s);
    }
    // family D: non-degenerate advance (cm<upperBound -> upperBound=cm)
    for (uint32_t it = 0; it < 400; ++it)
    {
        float z[32]; mkZ(z);
        PmScalars s; s.upperBound = float(10 + rng() % 10); s.cm = float(rng() % 10); // cm<ub
        s.size = 1u + rng() % 3; s.f = mkF(1u + rng() % 4);
        emit(z, s);
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bZ = createBufferFrom(dev, Z);
    PxgBufferHandle bS = createBufferFrom(dev, S);
    std::vector<float> zoInit(N * 32, -777.0f);
    std::vector<uint32_t> outInit(N * 4, 0xEEEEEEEEu);
    PxgBufferHandle bZO = createBufferFrom(dev, zoInit);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    PmParams prm = { N, 0, 0, 0 };
    Kernel k("npGjkPermute.spv", "ctGjkPermute", { 4, (uint32_t)sizeof(PmScalars), 4, 4 }, sizeof(PmParams));
    k.dispatch((N + 63) / 64, { bZ, bS, bZO, bOut }, prm);
    dev->flush();

    const auto gZO = downloadBuffer<float>(dev, bZO, N * 32);
    const auto gOut = downloadBuffer<uint32_t>(dev, bOut, N * 4);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const Ref& r = ref[t];
        if (gOut[t * 4 + 0] != r.newSize) ++mm;
        if (gOut[t * 4 + 1] != r.degenerate) ++mm;
        if (gOut[t * 4 + 2] != r.backupSize) ++mm;
        if (gOut[t * 4 + 3] != r.ub) ++mm;
        for (uint32_t L = 0; L < 32; ++L)
            if (asU32(gZO[t * 32 + L]) != asU32(r.z[L])) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "gjkPermute: %zu mismatches", mm);
    PXG_CHECK_MSG(nDegF > 100 && nDegCond > 100 && nBackup > 100 && nAdvance > 100,
                  "gjkPermute coverage thin (degF=%zu degCond=%zu backup=%zu advance=%zu)",
                  nDegF, nDegCond, nBackup, nAdvance);

    for (PxgBufferHandle h : { bZ, bS, bZO, bOut }) dev->destroyBuffer(h);
}

} // namespace
