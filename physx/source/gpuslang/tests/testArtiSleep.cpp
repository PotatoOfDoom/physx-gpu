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


// P8.95a: npArtiSleep.slang -- sleepCheck1T.
//
// THE ORACLE IS BIT-EXACT AGAIN.  Every operation in the function is engine
// arithmetic with a PX_CUDA_CALLABLE original -- PxQuat::rotateInv,
// PxVec3::multiply/dot/magnitudeSquared -- and there is no transcendental
// anywhere, so the mirror computes with the engine's own functions and the
// comparison is `==` on the raw words.  The function itself is
// `static __device__`, so its STRUCTURE is transcribed; the structure is what
// the sweep pins.
//
// The case table is built so every branch is taken both ways, each on several
// cases, and counted from the same predicates the fill uses:
//   * wc ABOVE both guard conditions (energy never examined), wc below the
//     half-reset-time, and wc >= half-reset but < dt -- the second condition's
//     only independent life;
//   * energy above and below the threshold, and a threshold of EXACTLY ZERO
//     (factor 2);
//   * the factor clamp: energy / threshold beyond 2 and inside it;
//   * invMass zero and non-zero; each inertia component <= 0 on its own case;
//   * wc that decrements to exactly zero (the second accumulator zeroing) and
//     wc that stays positive.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "PxgArticulationLink.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxQuat;
using physx::PxReal;
using physx::PxVec3;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiSleepLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t ASL_CASES = 211;

static float aslV(uint32_t i, uint32_t k)
{
    return (float((i * 37u + k * 19u) % 431u) + 1.0f) * 0.0078125f - 1.6875f;
}

using pxgref::aslEnergy;
using pxgref::aslMirror;

PXG_TEST(artiSleepCheck)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSleep.spv", "artiSleepCheck", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiSleepLoader);

    std::vector<float> inA(size_t(ASL_CASES) * 16);
    std::vector<float> inB(size_t(ASL_CASES) * 12);
    uint32_t wcHigh = 0, wcLowHalf = 0, wcDtOnly = 0, thrZero = 0, overClamp = 0, invMass0 = 0,
             degen = 0, toZero = 0, atBoundary = 0;
    for (uint32_t i = 0; i < ASL_CASES; ++i)
    {
        float* A = inA.data() + size_t(i) * 16;
        float* B = inB.data() + size_t(i) * 12;
        for (uint32_t k = 0; k < 8; ++k)
            A[k] = aslV(i, k) * 0.5f;    // the stored accumulators
        A[3] = aslV(i, 40);              // poison w: the store must overwrite it
        A[7] = aslV(i, 41);

        // dt around 1/60, wc spread across the three guard regimes
        const float dt = 0.01f + float(i % 7) * 0.004f;
        float wc;
        switch (i % 5)
        {
        case 0: wc = 0.25f; ++wcLowHalf; break;                    // < resetTime/2 = 0.2? no: 0.25 > 0.2
        case 1: wc = 0.05f; ++wcLowHalf; break;                    // < 0.2
        case 2: wc = 0.5f; ++wcHigh; break;                        // above both
        case 3: wc = 0.201f + dt * 0.5f; ++wcHigh; break;          // above half, above dt
        default: wc = dt * 0.9f; ++wcLowHalf; break;   // dt*0.9 < 0.2: FIRST condition
        }
        // case 0 is actually ABOVE resetTime/2 (0.25 > 0.2): reclassify
        if (i % 5 == 0) { --wcLowHalf; ++wcHigh; }
        // the decrement-to-zero cases: tiny wc, huge dt, low energy
        if (i % 11 == 7) { wc = dt * 0.5f; ++toZero; }
        float dtUsed = dt;
        // THE SECOND GUARD CONDITION'S ONLY INDEPENDENT LIFE.  With dt <= 0.038
        // the test `wc < dt` can only fire when `wc < resetTime/2 = 0.2`
        // already fired -- the counter proved the "dtOnly" regime empty on the
        // first run.  Only a timestep LARGER than half the reset time makes the
        // second condition matter: wc in [0.2, dt).
        if (i % 23 == 5) { dtUsed = 0.25f; wc = 0.22f; ++wcDtOnly; }
        A[8] = wc;
        A[9] = dtUsed;

        // threshold: zero on some, tiny (over-clamp) on others, large otherwise
        float thr;
        if (i % 13 == 3) { thr = 0.0f; ++thrZero; }
        else if (i % 13 == 8) { thr = 1e-4f; ++overClamp; }
        else thr = 0.4f + float(i % 9) * 0.3f;
        A[10] = thr;
        A[11] = 0.f;

        PxQuat q(aslV(i, 20), aslV(i, 21), aslV(i, 22), aslV(i, 23) + 2.0f);
        q = q.getNormalized();
        A[12] = q.x; A[13] = q.y; A[14] = q.z; A[15] = q.w;

        for (uint32_t k = 0; k < 6; ++k)
            B[k] = aslV(i, 30 + k);
        // low-energy cases so the threshold branch goes both ways
        if (i % 4 == 1)
            for (uint32_t k = 0; k < 6; ++k)
                B[k] *= 0.01f;
        // invInertia: one component <= 0 per residue, invMass 0 on some
        B[6] = (i % 17 == 2) ? 0.0f : std::fabs(aslV(i, 50)) + 0.1f;
        B[7] = (i % 17 == 6) ? -0.5f : std::fabs(aslV(i, 51)) + 0.1f;
        B[8] = (i % 17 == 10) ? 0.0f : std::fabs(aslV(i, 52)) + 0.1f;
        B[9] = (i % 19 == 4) ? 0.0f : std::fabs(aslV(i, 53)) + 0.2f;
        if (i % 17 == 2 || i % 17 == 6 || i % 17 == 10) ++degen;
        if (i % 19 == 4) ++invMass0;
        B[10] = B[11] = 0.f;

        // the >= boundary, hit EXACTLY: threshold := the energy this very case
        // computes, with wc forced under the guard so the comparison runs
        if (i % 29 == 9)
        {
            A[8] = 0.05f;
            const PxQuat bq(A[12], A[13], A[14], A[15]);
            A[10] = float(aslEnergy(A, bq, PxVec3(B[0], B[1], B[2]), PxVec3(B[3], B[4], B[5]),
                                    B + 6));
            ++atBoundary;
        }
    }
    std::printf("[ info ] asl: wc high/lowHalf/dtOnly %u/%u/%u, thr0 %u, overClamp %u, "
                "invMass0 %u, degenInertia %u, toZero %u\n",
                wcHigh, wcLowHalf, wcDtOnly, thrZero, overClamp, invMass0, degen, toZero);
    PXG_CHECK(wcHigh > 10 && wcLowHalf > 10 && wcDtOnly > 5);
    PXG_CHECK(thrZero > 5 && overClamp > 5 && invMass0 > 5 && degen > 10 && toZero > 5);
    std::printf("[ info ] asl: %u cases sit exactly ON the threshold\n", atBoundary);
    PXG_CHECK_MSG(atBoundary > 3, "the >= boundary is not exercised");

    // the expectation
    std::vector<float> want(size_t(ASL_CASES) * 12);
    for (uint32_t i = 0; i < ASL_CASES; ++i)
    {
        const float* A = inA.data() + size_t(i) * 16;
        const float* B = inB.data() + size_t(i) * 12;
        float* W = want.data() + size_t(i) * 12;
        std::memcpy(W, A, 8 * sizeof(float));
        PxReal wc = A[8];
        const PxQuat q(A[12], A[13], A[14], A[15]);
        const float ret = aslMirror(W, wc, q, PxVec3(B[0], B[1], B[2]), PxVec3(B[3], B[4], B[5]),
                                    A[9], A[10], B + 6);
        W[8] = wc;
        W[9] = ret;
        W[10] = 0.f;
        W[11] = 0.f;
    }

    PxgDevicePtr dA = 0, dB = 0, dOut = 0;
    shim.memAlloc(&dA, inA.size() * sizeof(float));
    shim.memAlloc(&dB, inB.size() * sizeof(float));
    shim.memAlloc(&dOut, want.size() * sizeof(float));
    if (!dA || !dB || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dA, inA.data(), inA.size() * sizeof(float));
    shim.memcpyHtoD(dB, inB.data(), inB.size() * sizeof(float));

    struct Params { uint64_t a, b, o; uint32_t n, pad; } prm = { uint64_t(dA), uint64_t(dB),
                                                                 uint64_t(dOut), ASL_CASES, 0 };
    PXG_CHECK_MSG(shim.launchKernel(0, (ASL_CASES + 63) / 64, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiSleepCheck");
    shim.streamSynchronize();

    std::vector<float> got(want.size());
    shim.memcpyDtoH(got.data(), dOut, got.size() * sizeof(float));

    size_t bad = 0, signedZeros = 0;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(want.data());
    const uint32_t* g = reinterpret_cast<const uint32_t*>(got.data());
    for (size_t k = 0; k < want.size(); ++k)
    {
        if (w[k] == g[k]) continue;
        if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
        { ++signedZeros; continue; }
        if (bad < 10)
            std::printf("MM asl case %zu word %zu  ref %08x  gpu %08x\n", k / 12, k % 12, w[k],
                        g[k]);
        ++bad;
    }
    std::printf("[ info ] asl: signed zeros %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "sleepCheck1T is not bit-exact against the engine");

    shim.memFree(dA);
    shim.memFree(dB);
    shim.memFree(dOut);
}

}  // namespace
