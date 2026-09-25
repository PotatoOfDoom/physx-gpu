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

// Verification of npJacobiRotate.slang (jacobiRotate): one Jacobi rotation
// zeroing A(p,q) with rotation accumulated into R (matrixDecomposition.cuh
// L101-136, the IEEE non-fast path), against an exact CPU transcription.
// Outputs bitwise modulo the documented +-0 (eqFZ).

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
struct I4 { int32_t x, y, z, w; };
struct JrIn  { F4 aC0, aC1, aC2, rC0, rC1, rC2; I4 pq; };
struct JrOut { F4 aC0, aC1, aC2, rC0, rC1, rC2; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

// exact transcription of jacobiRotate (row-major A[row][col] = math element)
static void refJacobi(const JrIn& s, JrOut& o)
{
    float A[3][3], R[3][3];
    A[0][0]=s.aC0.x; A[1][0]=s.aC0.y; A[2][0]=s.aC0.z;
    A[0][1]=s.aC1.x; A[1][1]=s.aC1.y; A[2][1]=s.aC1.z;
    A[0][2]=s.aC2.x; A[1][2]=s.aC2.y; A[2][2]=s.aC2.z;
    R[0][0]=s.rC0.x; R[1][0]=s.rC0.y; R[2][0]=s.rC0.z;
    R[0][1]=s.rC1.x; R[1][1]=s.rC1.y; R[2][1]=s.rC1.z;
    R[0][2]=s.rC2.x; R[1][2]=s.rC2.y; R[2][2]=s.rC2.z;

    const int p = s.pq.x, q = s.pq.y;
    if (A[p][q] != 0.0f)
    {
        const float apq = A[p][q];
        const float d = (A[p][p] - A[q][q]) / (2.0f * apq);
        float t = 1.0f / (std::fabs(d) + std::sqrt(d * d + 1.0f));
        if (d < 0.0f) t = -t;
        const float c = 1.0f / std::sqrt(t * t + 1.0f);
        const float sn = t * c;
        A[p][p] += t * apq;
        A[q][q] -= t * apq;
        A[p][q] = 0.0f; A[q][p] = 0.0f;
        for (int k = 0; k < 3; ++k)
        {
            if (k != p && k != q)
            {
                const float Akp = c * A[k][p] + sn * A[k][q];
                const float Akq = -sn * A[k][p] + c * A[k][q];
                A[k][p] = Akp; A[p][k] = Akp;
                A[k][q] = Akq; A[q][k] = Akq;
            }
        }
        for (int k = 0; k < 3; ++k)
        {
            const float Rkp = c * R[k][p] + sn * R[k][q];
            const float Rkq = -sn * R[k][p] + c * R[k][q];
            R[k][p] = Rkp;
            R[k][q] = Rkq;
        }
    }

    o.aC0 = F4{ A[0][0], A[1][0], A[2][0], 0.f };
    o.aC1 = F4{ A[0][1], A[1][1], A[2][1], 0.f };
    o.aC2 = F4{ A[0][2], A[1][2], A[2][2], 0.f };
    o.rC0 = F4{ R[0][0], R[1][0], R[2][0], 0.f };
    o.rC1 = F4{ R[0][1], R[1][1], R[2][1], 0.f };
    o.rC2 = F4{ R[0][2], R[1][2], R[2][2], 0.f };
}

PXG_TEST(npJacobiRotate)
{
    std::mt19937 rng(212001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<JrIn> in(N);
    std::vector<JrOut> ref(N);

    const int pqs[3][2] = { {0,1}, {0,2}, {1,2} };

    for (uint32_t i = 0; i < N; ++i)
    {
        JrIn s{};
        // symmetric A
        float m[3][3];
        m[0][0]=rf(-3,3); m[1][1]=rf(-3,3); m[2][2]=rf(-3,3);
        m[0][1]=m[1][0]=rf(-3,3);
        m[0][2]=m[2][0]=rf(-3,3);
        m[1][2]=m[2][1]=rf(-3,3);
        const int p = pqs[i % 3][0], q = pqs[i % 3][1];
        // early-return stripe: force A(p,q)==0
        if ((i % 13u) == 6u) { m[p][q] = 0.f; m[q][p] = 0.f; }
        s.aC0 = F4{ m[0][0], m[1][0], m[2][0], 0.f };
        s.aC1 = F4{ m[0][1], m[1][1], m[2][1], 0.f };
        s.aC2 = F4{ m[0][2], m[1][2], m[2][2], 0.f };
        // R starts as identity (typical eigenDecomposition seed)
        s.rC0 = F4{ 1, 0, 0, 0 };
        s.rC1 = F4{ 0, 1, 0, 0 };
        s.rC2 = F4{ 0, 0, 1, 0 };
        // Dense-R stripe: a non-identity R (as in eigenDecomposition's 2nd-4th
        // sweeps) so the R-accumulation loop evaluates c*R[k][p]+sn*R[k][q] with
        // BOTH addends simultaneously nonzero, not just one.
        if ((i % 7u) == 3u)
        {
            s.rC0 = F4{ rf(-2,2), rf(-2,2), rf(-2,2), 0.f };
            s.rC1 = F4{ rf(-2,2), rf(-2,2), rf(-2,2), 0.f };
            s.rC2 = F4{ rf(-2,2), rf(-2,2), rf(-2,2), 0.f };
        }
        s.pq = I4{ p, q, 0, 0 };
        in[i] = s;
        refJacobi(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<JrOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npJacobiRotate.spv", "jacobiRotate",
             { (uint32_t)sizeof(JrIn), (uint32_t)sizeof(JrOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<JrOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].aC0.x;
        const float* qfp = &g[i].aC0.x;
        const size_t nf = sizeof(JrOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u pq(%d,%d) aC0 R (%g %g %g) G (%g %g %g)\n", i, in[i].pq.x, in[i].pq.y,
                            ref[i].aC0.x, ref[i].aC0.y, ref[i].aC0.z, g[i].aC0.x, g[i].aC0.y, g[i].aC0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "jacobiRotate outputs bitwise-identical (mod +-0)");
}

}  // namespace
