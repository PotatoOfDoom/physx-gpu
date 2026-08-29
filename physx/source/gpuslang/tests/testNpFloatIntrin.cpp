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

// P9.33a: a PLATFORM measurement that about forty shaders in this corpus have
// been depending on without anyone taking it.
//
// `PxVec3::getNormalized()` (PxVec3.h:286) is `m > 0 ? v * PxRecipSqrt(m) : 0`,
// and `PxRecipSqrt(float)` (PxMath.h:161) is `intrinsics::recipSqrt`, which on
// unix is `1.0f / ::sqrtf(a)` (PxUnixMathIntrinsics.h:93) -- a correctly-rounded
// square root followed by a correctly-rounded division.  Both operations are
// exactly specified by IEEE 754, so the CPU mirrors in this port's fixtures
// compute a value that is uniquely determined.
//
// The obvious Slang spelling is `v * rsqrt(m)`, and about forty shaders here use
// it.  `rsqrt` lowers to SPIR-V `GLSL.std.450 InverseSqrt`, whose accuracy
// Vulkan specifies as 2 ULP -- NOT as the exact division.  A driver is free to
// answer with a fast approximation.  If this one did, every bit-exact fixture
// running through one of those forty sites would be comparing an approximation
// against an exact reference and would have to fail; they pass, so on THIS
// device the two agree at least on the inputs those fixtures reach.
//
// "At least on the inputs those fixtures reach" is the whole problem.  This test
// measures the two spellings directly, over inputs chosen to be far harder than
// any collision kernel produces, and states the answer as a number instead of an
// inference:
//
//   1. does the device's rsqrt(x) equal its own 1.0f / sqrt(x), bit for bit?
//   2. does the device's 1.0f / sqrt(x) equal the HOST's 1.0f / sqrtf(x), bit
//      for bit?  This is the property the CPU mirrors actually rely on, and it
//      is a separate question from the first.
//
// If (1) ever fails, the corpus's rsqrt sites are not transcriptions of
// getNormalized -- they are a coincidence that held until it did not, and the
// finding names the input that broke it.  If (2) ever fails, no fixture in this
// port can be bit-exact through a normalise at all, whichever spelling is used.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

struct DevBuf
{
    PxgBufferHandle h = 0;
    uint64_t addr = 0;
};

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// Byte-identical to npFloatIntrin.slang's push-constant struct.
struct FloatIntrinPush
{
    uint32_t n, pad4;
    uint64_t inputs, outRsqrt, outDiv, outSqrt;
};
static_assert(sizeof(FloatIntrinPush) == 40, "float-intrinsic blob must be 40 bytes");
static_assert(offsetof(FloatIntrinPush, inputs) == 8, "inputs @8");
static_assert(offsetof(FloatIntrinPush, outSqrt) == 32, "outSqrt @32");

// The inputs.  These are magnitudeSquared values, so they are non-negative; the
// callers in this corpus all guard `m > 0` before dividing, so zero is excluded
// on purpose -- what happens at zero is decided by that guard, not by the
// intrinsic.
static std::vector<float> buildInputs()
{
    std::vector<float> in;
    std::mt19937 rng(90332u);

    // 1. EVERY representable exponent, at several mantissas each.  A fast
    //    reciprocal-sqrt approximation is usually a seed table indexed by the
    //    high mantissa bits plus a Newton step, so its error is a function of
    //    where in the binade the input sits -- sweeping binades is the sweep
    //    that would find it.
    for (int e = -60; e <= 60; ++e)
    {
        const float scale = std::ldexp(1.0f, e);
        for (uint32_t k = 0; k < 64u; ++k)
            in.push_back(scale * (1.0f + float(k) / 64.0f));
    }

    // 2. Exact powers of four (rsqrt is exactly representable), and their
    //    neighbours one ULP either side -- the places where a correctly-rounded
    //    answer and an approximate one are most likely to land in different
    //    buckets.
    for (int e = -30; e <= 30; ++e)
    {
        const float p = std::ldexp(1.0f, 2 * e);
        in.push_back(p);
        in.push_back(u2f(f2u(p) - 1u));
        in.push_back(u2f(f2u(p) + 1u));
    }

    // 3. Uniformly random bit patterns inside a sane exponent window, which is
    //    a far denser probe of mantissa space than any geometric input.
    for (uint32_t i = 0; i < 200000u; ++i)
    {
        const uint32_t exponent = 60u + (rng() % 140u);          // ~2^-67 .. 2^73
        const uint32_t mantissa = rng() & 0x7fffffu;
        in.push_back(u2f((exponent << 23) | mantissa));
    }

    // 4. Denormals and the smallest normals -- reachable from a squared length
    //    of a very short vector, and the region where flush-to-zero behaviour
    //    would show up.
    for (uint32_t i = 1; i <= 512u; ++i)
        in.push_back(u2f(i));
    in.push_back(u2f(0x00800000u));      // FLT_MIN
    in.push_back(u2f(0x007fffffu));      // largest denormal
    in.push_back(3.402823466e+38f);      // FLT_MAX

    // 5. The magnitudes this corpus actually produces: squared lengths of
    //    collision-scale vectors, so the measurement covers the working range
    //    densely as well as the extremes.
    for (uint32_t i = 0; i < 50000u; ++i)
    {
        const float a = (float(rng() % 2000001u) / 1000000.0f) - 1.0f;
        const float b = (float(rng() % 2000001u) / 1000000.0f) - 1.0f;
        const float c = (float(rng() % 2000001u) / 1000000.0f) - 1.0f;
        const float m = a * a + b * b + c * c;
        if (m > 0.0f)
            in.push_back(m);
    }
    return in;
}

PXG_TEST(npFloatIntrinRecipSqrt)
{
    PxgSlangComputeDevice* dev = testDevice();

    const std::vector<float> in = buildInputs();
    const uint32_t N = uint32_t(in.size());

    DevBuf bIn = makeDev(dev, in);
    if (!bIn.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bIn.h); return; }
    const std::vector<float> zero(N, 0.0f);
    DevBuf bR = makeDev(dev, zero), bD = makeDev(dev, zero), bS = makeDev(dev, zero);

    FloatIntrinPush push = {};
    push.n = N;
    push.inputs = bIn.addr;
    push.outRsqrt = bR.addr;
    push.outDiv = bD.addr;
    push.outSqrt = bS.addr;

    Kernel k("npFloatIntrin.spv", "floatIntrinRecipSqrtLaunch", {}, sizeof(FloatIntrinPush));
    k.dispatch((N + 255u) / 256u, {}, push, 1u, 1u);
    dev->flush();

    const auto gR = downloadBuffer<float>(dev, bR.h, N);
    const auto gD = downloadBuffer<float>(dev, bD.h, N);
    const auto gS = downloadBuffer<float>(dev, bS.h, N);

    // ---- question 1: rsqrt(x) vs the device's own 1.0f / sqrt(x) --------------
    uint32_t intrinsicDiff = 0u, worstUlp = 0u;
    float worstInput = 0.0f, worstR = 0.0f, worstDv = 0.0f;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t a = f2u(gR[i]), b = f2u(gD[i]);
        if (a == b) continue;
        ++intrinsicDiff;
        const uint32_t ulp = (a > b) ? (a - b) : (b - a);
        if (ulp > worstUlp)
        { worstUlp = ulp; worstInput = in[i]; worstR = gR[i]; worstDv = gD[i]; }
    }

    // ---- question 2: the device's sqrt/divide vs the host's ------------------
    // This is what every CPU mirror in this port computes, so a difference here
    // would mean no normalise can be compared bit-exactly at all.
    uint32_t hostDiff = 0u, hostSqrtDiff = 0u;
    float worstHostInput = 0.0f;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float hs = std::sqrt(in[i]);
        if (f2u(gS[i]) != f2u(hs)) ++hostSqrtDiff;
        const float hd = 1.0f / hs;
        if (f2u(gD[i]) != f2u(hd))
        {
            if (hostDiff == 0u) worstHostInput = in[i];
            ++hostDiff;
        }
    }

    // ---- the control: can this comparison see a difference at all? ------------
    // Everything above is an equality count, and an equality count over buffers
    // that were never written reads as PERFECT.  Two things answer that.  First,
    // `hostSqrtDiff` is a liveness proof and not merely a third question: it
    // compares every lane against a host sqrt of a strictly positive input, so an
    // unwritten lane (0.0f) or an undispatched lane fails it.  Second, the
    // comparator is run against a deliberately perturbed copy, one ULP off in a
    // single lane, and must report exactly one difference of exactly one ULP.
    // A comparison that cannot fail on demand is not evidence when it passes.
    uint32_t controlDiff = 0u, controlUlp = 0u;
    {
        std::vector<float> perturbed = gR;
        const uint32_t victim = N / 3u;
        perturbed[victim] = u2f(f2u(perturbed[victim]) + 1u);
        for (uint32_t i = 0; i < N; ++i)
        {
            const uint32_t a = f2u(perturbed[i]), b = f2u(gD[i]);
            if (a == b) continue;
            ++controlDiff;
            const uint32_t ulp = (a > b) ? (a - b) : (b - a);
            if (ulp > controlUlp) controlUlp = ulp;
        }
    }
    PXG_CHECK_MSG(controlDiff == 1u && controlUlp == 1u,
                  "the ULP comparison must report exactly one 1-ULP difference when one lane "
                  "is perturbed by one ULP; it reported %u difference(s), worst %u ULP.  "
                  "Until it does, the zero above is not a measurement",
                  controlDiff, controlUlp);

    if (intrinsicDiff)
        std::printf("    rsqrt != 1/sqrt on %u of %u inputs; worst %u ULP at x = %.9g "
                    "(rsqrt %.9g, 1/sqrt %.9g)\n",
                    intrinsicDiff, N, worstUlp, worstInput, worstR, worstDv);
    if (hostDiff)
        std::printf("    device 1/sqrt != host 1/sqrtf on %u of %u inputs; first at x = %.9g\n",
                    hostDiff, N, worstHostInput);

    PXG_CHECK_MSG(hostSqrtDiff == 0u,
                  "the device's sqrt disagrees with the host's on %u of %u inputs -- IEEE 754 "
                  "specifies square root exactly, so nothing in this port can be compared "
                  "bit-exactly through one", hostSqrtDiff, N);
    PXG_CHECK_MSG(hostDiff == 0u,
                  "the device's 1.0f/sqrt(x) disagrees with the host's on %u of %u inputs.  "
                  "Every CPU mirror in this port computes the host form, so this is not a "
                  "question of which spelling a shader uses -- no normalise can be bit-exact "
                  "on this device", hostDiff, N);
    PXG_CHECK_MSG(intrinsicDiff == 0u,
                  "rsqrt(x) and 1.0f/sqrt(x) differ on %u of %u inputs (worst %u ULP).  The "
                  "~40 shaders in this corpus that spell PxVec3::getNormalized as "
                  "`v * rsqrt(m)` are therefore NOT transcriptions of it -- the engine "
                  "computes PxRecipSqrt = 1.0f/sqrtf.  They must be respelled; a fixture that "
                  "passes today does so because its inputs miss the difference",
                  intrinsicDiff, N, worstUlp);

    std::printf("    [ note ] %u inputs (every binade from 2^-60 to 2^60, powers of four and "
                "their ULP neighbours, 200k random bit patterns, denormals, and 50k "
                "collision-scale squared lengths): rsqrt(x) == 1.0f/sqrt(x) == host "
                "1.0f/sqrtf(x), bit for bit, on every one.  All %u lanes verified written "
                "against a host sqrt; the comparator caught a planted 1-ULP perturbation\n",
                N, N);

    for (PxgBufferHandle h : { bIn.h, bR.h, bD.h, bS.h })
        dev->destroyBuffer(h);
}

} // namespace
