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

// Verification of npArtiInvertInertia.slang (invertInertia): the block symmetric
// 6x6 spatial-inertia inverse via Schur complement, Dy::SpatialMatrix::
// invertInertia() + invertSym33 (DyFeatherstoneArticulationUtils.h L488-555),
// against an exact CPU transcription.  Outputs bitwise modulo the documented
// +-0 (eqFZ).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };

struct IiIn  { F4 tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2; };
struct IiOut { F4 tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
struct M3 { V3 c0, c1, c2; };  // column-major

static V3 cross3(V3 a, V3 b) { return V3{ a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; }
static float dot3(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static V3 mtx(const M3& m, V3 v) { return V3{ m.c0.x*v.x + m.c1.x*v.y + m.c2.x*v.z,
                                              m.c0.y*v.x + m.c1.y*v.y + m.c2.y*v.z,
                                              m.c0.z*v.x + m.c1.z*v.y + m.c2.z*v.z }; }
static M3 matmul(const M3& a, const M3& b) { M3 r; r.c0 = mtx(a, b.c0); r.c1 = mtx(a, b.c1); r.c2 = mtx(a, b.c2); return r; }
static M3 transpose3(const M3& m) { M3 r; r.c0 = V3{m.c0.x,m.c1.x,m.c2.x}; r.c1 = V3{m.c0.y,m.c1.y,m.c2.y}; r.c2 = V3{m.c0.z,m.c1.z,m.c2.z}; return r; }
static M3 scale3(const M3& m, float s) { M3 r; r.c0 = V3{m.c0.x*s,m.c0.y*s,m.c0.z*s}; r.c1 = V3{m.c1.x*s,m.c1.y*s,m.c1.z*s}; r.c2 = V3{m.c2.x*s,m.c2.y*s,m.c2.z*s}; return r; }
static M3 add3(const M3& a, const M3& b) { M3 r; r.c0 = V3{a.c0.x+b.c0.x,a.c0.y+b.c0.y,a.c0.z+b.c0.z}; r.c1 = V3{a.c1.x+b.c1.x,a.c1.y+b.c1.y,a.c1.z+b.c1.z}; r.c2 = V3{a.c2.x+b.c2.x,a.c2.y+b.c2.y,a.c2.z+b.c2.z}; return r; }
static M3 neg3(const M3& m) { M3 r; r.c0 = V3{-m.c0.x,-m.c0.y,-m.c0.z}; r.c1 = V3{-m.c1.x,-m.c1.y,-m.c1.z}; r.c2 = V3{-m.c2.x,-m.c2.y,-m.c2.z}; return r; }
static M3 identity3() { M3 r; r.c0 = V3{1,0,0}; r.c1 = V3{0,1,0}; r.c2 = V3{0,0,1}; return r; }

static M3 invertSym33(const M3& m)
{
    const V3 v0 = cross3(m.c1, m.c2);
    const V3 v1 = cross3(m.c2, m.c0);
    const V3 v2 = cross3(m.c0, m.c1);
    const float det = dot3(v0, m.c0);
    if (det != 0.f)
    {
        const float recipDet = 1.f / det;
        M3 r;
        r.c0 = V3{ v0.x*recipDet, v0.y*recipDet, v0.z*recipDet };
        r.c1 = V3{ v0.y*recipDet, v1.y*recipDet, v1.z*recipDet };
        r.c2 = V3{ v0.z*recipDet, v1.z*recipDet, v2.z*recipDet };
        return r;
    }
    return identity3();
}

static M3 fromF4(const F4& c0, const F4& c1, const F4& c2) { M3 m; m.c0 = V3{c0.x,c0.y,c0.z}; m.c1 = V3{c1.x,c1.y,c1.z}; m.c2 = V3{c2.x,c2.y,c2.z}; return m; }
static void storeF4(const M3& m, F4& c0, F4& c1, F4& c2) { c0 = F4{m.c0.x,m.c0.y,m.c0.z,0.f}; c1 = F4{m.c1.x,m.c1.y,m.c1.z,0.f}; c2 = F4{m.c2.x,m.c2.y,m.c2.z,0.f}; }

// exact transcription of Dy::SpatialMatrix::invertInertia()
static void refInvertInertia(const IiIn& s, IiOut& o)
{
    M3 aa = fromF4(s.blC0, s.blC1, s.blC2);   // bottomLeft
    M3 ll = fromF4(s.trC0, s.trC1, s.trC2);   // topRight
    const M3 la = fromF4(s.tlC0, s.tlC1, s.tlC2);   // topLeft

    aa = scale3(add3(aa, transpose3(aa)), 0.5f);
    ll = scale3(add3(ll, transpose3(ll)), 0.5f);

    const M3 AAInv = invertSym33(aa);
    const M3 z = matmul(neg3(la), AAInv);
    const M3 S = add3(ll, matmul(z, transpose3(la)));
    const M3 LL = invertSym33(S);
    const M3 LA = matmul(LL, z);
    const M3 AA = add3(AAInv, matmul(transpose3(z), LA));
    const M3 rTL = transpose3(LA);

    storeF4(rTL, o.tlC0, o.tlC1, o.tlC2);
    storeF4(AA, o.trC0, o.trC1, o.trC2);
    storeF4(LL, o.blC0, o.blC1, o.blC2);
}

PXG_TEST(npArtiInvertInertia)
{
    std::mt19937 rng(203001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<IiIn> in(N);
    std::vector<IiOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        IiIn s{};
        // topLeft (coupling): arbitrary
        s.tlC0 = F4{ rf(-2,2), rf(-2,2), rf(-2,2), 0.f };
        s.tlC1 = F4{ rf(-2,2), rf(-2,2), rf(-2,2), 0.f };
        s.tlC2 = F4{ rf(-2,2), rf(-2,2), rf(-2,2), 0.f };
        // topRight (mass block): diagonally dominant so the symmetrised matrix inverts
        const float md = rf(5.f, 8.f);
        s.trC0 = F4{ md, rf(-1,1), rf(-1,1), 0.f };
        s.trC1 = F4{ rf(-1,1), md, rf(-1,1), 0.f };
        s.trC2 = F4{ rf(-1,1), rf(-1,1), md, 0.f };
        // bottomLeft (inertia block): diagonally dominant
        const float ad = rf(5.f, 8.f);
        s.blC0 = F4{ ad, rf(-1,1), rf(-1,1), 0.f };
        s.blC1 = F4{ rf(-1,1), ad, rf(-1,1), 0.f };
        s.blC2 = F4{ rf(-1,1), rf(-1,1), ad, 0.f };

        // Singular stripe: zero bottomLeft so aa==0 and the FIRST invertSym33
        // (on aa) hits the det==0 identity fallback (shader/ref must agree).
        if ((i % 41u) == 13u)
        {
            s.blC0 = F4{ 0,0,0,0 }; s.blC1 = F4{ 0,0,0,0 }; s.blC2 = F4{ 0,0,0,0 };
        }
        // Schur-singular stripe: zero topRight (ll==0) and the whole z-ROW of
        // topLeft (la row 2 == 0).  Then la^T has an exact zero column, so
        // S = z*la^T has an exact zero third column and det(S)==0.0f exactly,
        // driving the SECOND invertSym33 (on the Schur complement) into its
        // det==0 identity fallback (shader/ref must agree).
        else if ((i % 53u) == 17u)
        {
            s.trC0 = F4{ 0,0,0,0 }; s.trC1 = F4{ 0,0,0,0 }; s.trC2 = F4{ 0,0,0,0 };
            s.tlC0.z = 0.f; s.tlC1.z = 0.f; s.tlC2.z = 0.f;  // la row 2 == 0
        }
        in[i] = s;
        refInvertInertia(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<IiOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiInvertInertia.spv", "invertInertia",
             { (uint32_t)sizeof(IiIn), (uint32_t)sizeof(IiOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<IiOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].tlC0.x;
        const float* qfp = &g[i].tlC0.x;
        const size_t nf = sizeof(IiOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u trC0R (%g %g %g) G (%g %g %g)\n", i, ref[i].trC0.x, ref[i].trC0.y,
                            ref[i].trC0.z, g[i].trC0.x, g[i].trC0.y, g[i].trC0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "invertInertia outputs bitwise-identical (mod +-0)");
}

}  // namespace
