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

// Verification of npSbCollisionConstraint.slang (sbCollisionConstraint): the
// two soft-body deformable collision constraints tetTriCollisionConstraint
// (softBody.cu L650-715) and tetVertexCollisionConstraint (L717-771), against
// an exact CPU transcription.  The lambdaN slot is an in/out scalar; the
// tetVertex dx outputs accumulate over caller-provided values while the
// tetTri outputs overwrite.  Outputs bitwise mod +-0 (eqFZ).

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
struct ScIn  { F4 misc, normalPen, bcA, bcB, d0, d1, d2, d3, d4, d5, d6, acc0, acc1, acc2, acc3, acc4; };
struct ScOut { F4 o0, o1, o2, o3, o4, o5, o6, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls3(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float magSq3(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float pabs(float a) { return a < 0.f ? -a : a; }
static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }
static V3 xyz(const F4& f) { return V3{ f.x, f.y, f.z }; }

static void refSc(const ScIn& s, ScOut& o)
{
    float lambdaN = s.misc.x;
    const float denomInv = s.misc.y;
    const float frictionCoefficient = s.misc.z;
    const uint32_t ctype = f2u(s.misc.w);
    const float threshold = 1.0e-14f;

    const V3 normal = xyz(s.normalPen);
    const float pen = s.normalPen.w;

    V3 relDelta;
    if (ctype == 0u)
    {
        const V3 cSum = add3(add3(muls3(xyz(s.d0), s.bcA.x), muls3(xyz(s.d1), s.bcA.y)), muls3(xyz(s.d2), s.bcA.z));
        const V3 sSum = add3(add3(add3(muls3(xyz(s.d3), s.bcB.x), muls3(xyz(s.d4), s.bcB.y)),
                                  muls3(xyz(s.d5), s.bcB.z)), muls3(xyz(s.d6), s.bcB.w));
        relDelta = sub3(cSum, sSum);
    }
    else
    {
        const V3 bSum = add3(add3(add3(muls3(xyz(s.d1), s.bcA.x), muls3(xyz(s.d2), s.bcA.y)),
                                  muls3(xyz(s.d3), s.bcA.z)), muls3(xyz(s.d4), s.bcA.w));
        relDelta = sub3(xyz(s.d0), bSum);
    }

    const float CN = pen + dot3(relDelta, normal);

    const float deltaLambdaN = selMax(-CN * denomInv, -lambdaN);
    lambdaN += deltaLambdaN;

    V3 commonVec = muls3(normal, deltaLambdaN);

    V3 tanDir = sub3(relDelta, muls3(normal, dot3(relDelta, normal)));
    const float tanMagSq = magSq3(tanDir);

    if (frictionCoefficient != 0.f && tanMagSq > threshold)
    {
        const float tanMag = std::sqrt(tanMagSq);
        // PxVec3::operator/= : reciprocal-multiply
        const float tanMagInv = 1.0f / tanMag;
        tanDir = V3{ tanDir.x * tanMagInv, tanDir.y * tanMagInv, tanDir.z * tanMagInv };
        const float CT = dot3(relDelta, tanDir);

        float deltaLambdaT = selMax(0.f, CT * denomInv);
        deltaLambdaT = -selMin(deltaLambdaT, frictionCoefficient * pabs(deltaLambdaN));

        const float tanScale = 0.5f;
        commonVec = add3(commonVec, muls3(tanDir, tanScale * deltaLambdaT));
    }

    if (ctype == 0u)
    {
        const V3 c0 = muls3(commonVec, s.d0.w * s.bcA.x);
        const V3 c1 = muls3(commonVec, s.d1.w * s.bcA.y);
        const V3 c2 = muls3(commonVec, s.d2.w * s.bcA.z);
        const V3 s0 = muls3(commonVec, -s.d3.w * s.bcB.x);
        const V3 s1 = muls3(commonVec, -s.d4.w * s.bcB.y);
        const V3 s2 = muls3(commonVec, -s.d5.w * s.bcB.z);
        const V3 s3 = muls3(commonVec, -s.d6.w * s.bcB.w);
        o.o0 = F4{ c0.x, c0.y, c0.z, 0.f };
        o.o1 = F4{ c1.x, c1.y, c1.z, 0.f };
        o.o2 = F4{ c2.x, c2.y, c2.z, 0.f };
        o.o3 = F4{ s0.x, s0.y, s0.z, 0.f };
        o.o4 = F4{ s1.x, s1.y, s1.z, 0.f };
        o.o5 = F4{ s2.x, s2.y, s2.z, 0.f };
        o.o6 = F4{ s3.x, s3.y, s3.z, 0.f };
    }
    else
    {
        const V3 r0 = add3(xyz(s.acc0), muls3(commonVec, s.d0.w));
        const V3 r1 = sub3(xyz(s.acc1), muls3(commonVec, s.d1.w * s.bcA.x));
        const V3 r2 = sub3(xyz(s.acc2), muls3(commonVec, s.d2.w * s.bcA.y));
        const V3 r3 = sub3(xyz(s.acc3), muls3(commonVec, s.d3.w * s.bcA.z));
        const V3 r4 = sub3(xyz(s.acc4), muls3(commonVec, s.d4.w * s.bcA.w));
        o.o0 = F4{ r0.x, r0.y, r0.z, 0.f };
        o.o1 = F4{ r1.x, r1.y, r1.z, 0.f };
        o.o2 = F4{ r2.x, r2.y, r2.z, 0.f };
        o.o3 = F4{ r3.x, r3.y, r3.z, 0.f };
        o.o4 = F4{ r4.x, r4.y, r4.z, 0.f };
        o.o5 = F4{ 0.f, 0.f, 0.f, 0.f };
        o.o6 = F4{ 0.f, 0.f, 0.f, 0.f };
    }
    o.aux = F4{ lambdaN, 0.f, 0.f, 0.f };
}

PXG_TEST(npSbCollisionConstraint)
{
    std::mt19937 rng(258001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ScIn> in(N);
    std::vector<ScOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        ScIn s{};
        const uint32_t ctype = i % 2u;
        s.misc = F4{ rf(-2.f, 2.f), rf(0.1f, 5.f), rf(0.f, 2.f), u2f(ctype) };  // lambdaN, denomInv, friction
        // unit-ish normal (input data; both sides read the same bits)
        {
            float nx = rf(-1.f, 1.f), ny = rf(-1.f, 1.f), nz = rf(-1.f, 1.f);
            const float n = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (n > 1e-3f) { nx /= n; ny /= n; nz /= n; }
            else { nx = 1.f; ny = 0.f; nz = 0.f; }
            s.normalPen = F4{ nx, ny, nz, rf(-0.5f, 0.5f) };
        }
        s.bcA = F4{ rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f) };
        s.bcB = F4{ rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f) };
        F4* ds = &s.d0;
        for (int d = 0; d < 7; ++d)
            ds[d] = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(0.f, 3.f) };
        F4* accs = &s.acc0;
        for (int a = 0; a < 5; ++a)
            accs[a] = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        // zero-friction stripe (branch skipped even with tangential motion)
        if ((i % 7u) == 2u) s.misc.z = 0.f;
        // tangent-free stripe: normal = (1,0,0) exactly and all delta y/z = 0
        // -> tanDir = 0 exactly, friction skipped despite nonzero friction
        if ((i % 11u) == 4u)
        {
            s.normalPen.x = 1.f; s.normalPen.y = 0.f; s.normalPen.z = 0.f;
            for (int d = 0; d < 7; ++d) { ds[d].y = 0.f; ds[d].z = 0.f; }
        }
        // strongly negative lambdaN: the -lambdaN arm of the PxMax clamp wins
        if ((i % 13u) == 6u) s.misc.x = -5.f;
        in[i] = s;
        refSc(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ScOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbCollisionConstraint.spv", "sbCollisionConstraint",
             { (uint32_t)sizeof(ScIn), (uint32_t)sizeof(ScOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ScOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4* rr = &ref[i].o0; const F4* gg = &g[i].o0;
        bool ok = true;
        for (int f = 0; f < 7 && ok; ++f)
            ok = eqFZ(rr[f].x, gg[f].x) && eqFZ(rr[f].y, gg[f].y) && eqFZ(rr[f].z, gg[f].z);
        ok = ok && eqFZ(ref[i].aux.x, g[i].aux.x);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u o0R (%g %g %g) o0G (%g %g %g) lamR %g lamG %g\n", i,
                            ref[i].o0.x, ref[i].o0.y, ref[i].o0.z, g[i].o0.x, g[i].o0.y, g[i].o0.z,
                            ref[i].aux.x, g[i].aux.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbCollisionConstraint outputs bitwise-identical (mod +-0)");
}

}  // namespace
