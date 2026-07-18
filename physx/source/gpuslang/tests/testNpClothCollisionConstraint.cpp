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

// Verification of npClothCollisionConstraint.slang (clothCollisionConstraint):
// computeMassInertiaOffsets (FEMClothExternalSolve.cu L202-240) and the shared
// vertex-triangle/edge-edge triangleCollisionConstraint (L243-286), against an
// exact CPU transcription.  On the denom<threshold false-return the dx output
// slots are zeroed by modeling convention (CUDA leaves the caller values; the
// bool flag distinguishes).  Outputs bitwise mod +-0 (eqFZ).

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
struct CcIn  { F4 misc, a0, a1, a2, b0, b1; };
struct CcOut { F4 o0, o1, o2, o3, o4, o5, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

const float FEMCLOTH_THRESHOLD = 1.0e-14f;

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

static void refCc(const CcIn& s, CcOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t ctype = f2u(s.misc.w);

    if (ctype == 0u)
    {
        const V3 p0 = xyz(s.a0), p1 = xyz(s.a1), p2 = xyz(s.a2);
        const V3 bc = xyz(s.b0);
        const float thickness = s.misc.z;

        const V3 average = muls3(add3(add3(p0, p1), p2), 1.0f / 3.0f);
        const V3 contact = add3(add3(muls3(p0, bc.x), muls3(p1, bc.y)), muls3(p2, bc.z));

        const float m0 = 1.0f / selMax(1e-18f, s.a0.w);
        const float m1 = 1.0f / selMax(1e-18f, s.a1.w);
        const float m2 = 1.0f / selMax(1e-18f, s.a2.w);

        const V3 offset = sub3(contact, average);
        const V3 ra = sub3(p0, average);
        const V3 rb = sub3(p1, average);
        const V3 rc = sub3(p2, average);

        const float thicknessSq = thickness * thickness * (m0 + m1 + m2);

        V3 invInertia;
        invInertia.x = 1.0f / (((ra.y * ra.y + ra.z * ra.z) * m0 + (rb.y * rb.y + rb.z * rb.z) * m1 +
                                (rc.y * rc.y + rc.z * rc.z) * m2 + thicknessSq));
        invInertia.y = 1.0f / (((ra.x * ra.x + ra.z * ra.z) * m0 + (rb.x * rb.x + rb.z * rb.z) * m1 +
                                (rc.x * rc.x + rc.z * rc.z) * m2 + thicknessSq));
        invInertia.z = 1.0f / (((ra.x * ra.x + ra.y * ra.y) * m0 + (rb.x * rb.x + rb.y * rb.y) * m1 +
                                (rc.x * rc.x + rc.y * rc.y) * m2 + thicknessSq));

        const float invMass = 1.0f / (m0 + m1 + m2);

        o.o0 = F4{ contact.x, contact.y, contact.z, 0.f };
        o.o1 = F4{ offset.x, offset.y, offset.z, 0.f };
        o.o2 = F4{ ra.x, ra.y, ra.z, 0.f };
        o.o3 = F4{ rb.x, rb.y, rb.z, 0.f };
        o.o4 = F4{ rc.x, rc.y, rc.z, 0.f };
        o.o5 = F4{ invInertia.x, invInertia.y, invInertia.z, invMass };
        o.aux = F4{ u2f(1u), 0.f, 0.f, 0.f };
    }
    else
    {
        const V3 disp = xyz(s.a0);
        const V3 normal = xyz(s.a1);
        const V3 relDelta = xyz(s.a2);
        const F4 invMass = s.b0;
        const F4 weights = s.b1;
        const float restDistance = s.misc.x;
        const float frictionCoefficient = s.misc.y;

        const float CN = dot3(disp, normal) - restDistance;

        const float denom = invMass.x * weights.x * weights.x + invMass.y * weights.y * weights.y +
                            invMass.z * weights.z * weights.z + invMass.w * weights.w * weights.w;
        if (denom < FEMCLOTH_THRESHOLD)
        {
            o.aux = F4{ u2f(0u), 0.f, 0.f, 0.f };
            return;
        }

        const float invDenom = 1.0f / denom;
        const float deltaLambdaN = -CN * invDenom;

        V3 commonVec = muls3(normal, deltaLambdaN);

        V3 tanDir = sub3(relDelta, muls3(normal, dot3(relDelta, normal)));
        const float tanMagSq = magSq3(tanDir);

        if (frictionCoefficient != 0.f && tanMagSq > FEMCLOTH_THRESHOLD)
        {
            const float invTanMag = 1.0f / std::sqrt(tanMagSq);
            tanDir = muls3(tanDir, invTanMag);
            const float CT = dot3(relDelta, tanDir);

            float deltaLambdaT = selMax(0.f, CT * invDenom);
            deltaLambdaT = -selMin(deltaLambdaT, frictionCoefficient * pabs(deltaLambdaN));

            commonVec = add3(commonVec, muls3(tanDir, deltaLambdaT));
        }

        const V3 r0 = muls3(commonVec, invMass.x * weights.x);
        const V3 r1 = muls3(commonVec, invMass.y * weights.y);
        const V3 r2 = muls3(commonVec, invMass.z * weights.z);
        const V3 r3 = muls3(commonVec, invMass.w * weights.w);
        o.o0 = F4{ r0.x, r0.y, r0.z, 0.f };
        o.o1 = F4{ r1.x, r1.y, r1.z, 0.f };
        o.o2 = F4{ r2.x, r2.y, r2.z, 0.f };
        o.o3 = F4{ r3.x, r3.y, r3.z, 0.f };
        o.aux = F4{ u2f(1u), 0.f, 0.f, 0.f };
    }
}

PXG_TEST(npClothCollisionConstraint)
{
    std::mt19937 rng(259001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CcIn> in(N);
    std::vector<CcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CcIn s{};
        const uint32_t ctype = i % 2u;
        s.misc = F4{ rf(0.f, 0.5f), rf(0.f, 2.f), rf(0.f, 0.1f), u2f(ctype) };  // restDist, friction, thickness
        if (ctype == 0u)
        {
            s.a0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
            s.a1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
            s.a2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
            s.b0 = F4{ rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f), 0.f };
            // exact-zero inverse mass: the PxMax(1e-18, w) clamp arm
            if ((i % 17u) == 3u) s.a0.w = 0.f;
            // zero thickness: thicknessSq = 0
            if ((i % 19u) == 5u) s.misc.z = 0.f;
        }
        else
        {
            s.a0 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };  // disp
            {
                float nx = rf(-1.f, 1.f), ny = rf(-1.f, 1.f), nz = rf(-1.f, 1.f);
                const float n = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (n > 1e-3f) { nx /= n; ny /= n; nz /= n; }
                else { nx = 1.f; ny = 0.f; nz = 0.f; }
                s.a1 = F4{ nx, ny, nz, 0.f };                                // normal
            }
            s.a2 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };  // relDelta
            s.b0 = F4{ rf(0.f, 3.f), rf(0.f, 3.f), rf(0.f, 3.f), rf(0.f, 3.f) };   // invMass
            s.b1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) }; // signed weights
            // degenerate denominator: all inverse masses zero -> false return
            if ((i % 13u) == 6u) s.b0 = F4{ 0.f, 0.f, 0.f, 0.f };
            // zero friction
            if ((i % 7u) == 2u) s.misc.y = 0.f;
            // exact tangent-free row: friction branch skipped despite fc > 0
            if ((i % 11u) == 4u)
            {
                s.a1 = F4{ 1.f, 0.f, 0.f, 0.f };
                s.a0.y = 0.f; s.a0.z = 0.f;
                s.a2.y = 0.f; s.a2.z = 0.f;
            }
        }
        in[i] = s;
        refCc(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothCollisionConstraint.spv", "clothCollisionConstraint",
             { (uint32_t)sizeof(CcIn), (uint32_t)sizeof(CcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4* rr = &ref[i].o0; const F4* gg = &g[i].o0;
        bool ok = true;
        for (int f = 0; f < 6 && ok; ++f)
            ok = eqFZ(rr[f].x, gg[f].x) && eqFZ(rr[f].y, gg[f].y) && eqFZ(rr[f].z, gg[f].z) && eqFZ(rr[f].w, gg[f].w);
        ok = ok && f2u(ref[i].aux.x) == f2u(g[i].aux.x);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u o0R (%g %g %g) o0G (%g %g %g) o5R (%g %g %g %g) o5G (%g %g %g %g)\n", i,
                            ref[i].o0.x, ref[i].o0.y, ref[i].o0.z, g[i].o0.x, g[i].o0.y, g[i].o0.z,
                            ref[i].o5.x, ref[i].o5.y, ref[i].o5.z, ref[i].o5.w,
                            g[i].o5.x, g[i].o5.y, g[i].o5.z, g[i].o5.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothCollisionConstraint outputs bitwise-identical (mod +-0)");
}

}  // namespace
