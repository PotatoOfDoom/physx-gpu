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

// Verification of npBoxFeature.slang: boxFeatureSetup — the feature->newTransformV
// switch of doBoxBoxGenerateContacts.  Given the SAT feature (0..5) and sign it
// builds mtd, newTransformV, e0/e1 and the derived transform1ToNew + localNormal.
// feature and sign are uploaded inputs (so the single `0>=sign` branch matches
// GPU/CPU bit-for-bit); everything else is deterministic float math compared with
// almostEqual.  All 6 features x both signs are exercised.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct FeatIn { F4 b0e, b1e, c00, c01, c02, p0, c10, c11, c12, p1, misc; };
struct FeatOut { F4 mtd, ntCol0, ntCol1, ntCol2, ntP, e, t1nCol0, t1nCol1, t1nCol2, t1nP, localNormal; };

static inline V3 neg(const V3& v) { return V3{ -v.x, -v.y, -v.z }; }
static inline V3 madd(const V3& a, const V3& b, float s) { return V3{ a.x + b.x * s, a.y + b.y * s, a.z + b.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 tmul(const V3& c0, const V3& c1, const V3& c2, const V3& v)
{ return V3{ dot3(c0, v), dot3(c1, v), dot3(c2, v) }; }
static inline float comp3(const V3& v, uint32_t i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

void refFeature(const V3& box0e, const V3& box1e, const V3 a0[3], const V3& p0,
                const V3 a1[3], const V3& p1, uint32_t feature, float sign,
                V3& mtd, V3& col0, V3& col1, V3& col2, V3& p, float& e0, float& e1,
                V3 t1n[3], V3& t1nP, V3& localNormal)
{
    const bool ng = 0.f >= sign;
    if (feature == 0) {
        if (ng) { mtd = a0[0]; col0 = neg(a0[2]); col1 = a0[1]; col2 = a0[0]; }
        else    { mtd = neg(a0[0]); col0 = a0[2]; col1 = a0[1]; col2 = neg(a0[0]); }
        p = madd(p0, mtd, -box0e.x); e0 = box0e.z; e1 = box0e.y;
    } else if (feature == 1) {
        if (ng) { mtd = a0[1]; col0 = a0[0]; col1 = neg(a0[2]); col2 = a0[1]; }
        else    { mtd = neg(a0[1]); col0 = a0[0]; col1 = a0[2]; col2 = neg(a0[1]); }
        p = madd(p0, mtd, -box0e.y); e0 = box0e.x; e1 = box0e.z;
    } else if (feature == 2) {
        if (ng) { mtd = a0[2]; col0 = a0[0]; col1 = a0[1]; col2 = a0[2]; }
        else    { mtd = neg(a0[2]); col0 = a0[0]; col1 = neg(a0[1]); col2 = neg(a0[2]); }
        p = madd(p0, mtd, -box0e.z); e0 = box0e.x; e1 = box0e.y;
    } else if (feature == 3) {
        if (ng) { mtd = a1[0]; col0 = a1[2]; col1 = a1[1]; col2 = neg(a1[0]); }
        else    { mtd = neg(a1[0]); col0 = neg(a1[2]); col1 = a1[1]; col2 = a1[0]; }
        p = madd(p1, mtd, box1e.x); e0 = box1e.z; e1 = box1e.y;
    } else if (feature == 4) {
        if (ng) { mtd = a1[1]; col0 = a1[0]; col1 = a1[2]; col2 = neg(a1[1]); }
        else    { mtd = neg(a1[1]); col0 = a1[0]; col1 = neg(a1[2]); col2 = a1[1]; }
        p = madd(p1, mtd, box1e.y); e0 = box1e.x; e1 = box1e.z;
    } else {
        if (ng) { mtd = a1[2]; col0 = a1[0]; col1 = neg(a1[1]); col2 = neg(a1[2]); }
        else    { mtd = neg(a1[2]); col0 = a1[0]; col1 = a1[1]; col2 = a1[2]; }
        p = madd(p1, mtd, box1e.z); e0 = box1e.x; e1 = box1e.y;
    }

    const bool useT1 = feature < 3;
    const V3 oc0 = useT1 ? a1[0] : a0[0];
    const V3 oc1 = useT1 ? a1[1] : a0[1];
    const V3 oc2 = useT1 ? a1[2] : a0[2];
    const V3 oP = useT1 ? p1 : p0;
    t1n[0] = tmul(col0, col1, col2, oc0);
    t1n[1] = tmul(col0, col1, col2, oc1);
    t1n[2] = tmul(col0, col1, col2, oc2);
    const V3 diff{ oP.x - p.x, oP.y - p.y, oP.z - p.z };
    t1nP = tmul(col0, col1, col2, diff);
    localNormal = tmul(col0, col1, col2, mtd);
}

void euler(float ax, float ay, float az, V3 col[3])
{
    const float cx = std::cos(ax), sx = std::sin(ax), cy = std::cos(ay), sy = std::sin(ay), cz = std::cos(az), sz = std::sin(az);
    col[0] = V3{ cy * cz, cy * sz, -sy };
    col[1] = V3{ sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy };
    col[2] = V3{ cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy };
}

PXG_TEST(npBoxFeature_featureSetup)
{
    std::mt19937 rng(51010);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<FeatIn> in(N);
    std::vector<FeatOut> ref(N);
    size_t featHist[6] = { 0, 0, 0, 0, 0, 0 }, negCount = 0, posCount = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 b0e{ rf(0.4f, 1.8f), rf(0.4f, 1.8f), rf(0.4f, 1.8f) };
        V3 b1e{ rf(0.4f, 1.8f), rf(0.4f, 1.8f), rf(0.4f, 1.8f) };
        V3 a0[3], a1[3];
        euler(rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), a0);
        euler(rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), a1);
        V3 p0{ rf(-2, 2), rf(-2, 2), rf(-2, 2) }, p1{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        const uint32_t feature = i % 6;
        // nonzero sign, both directions (avoid the measure-zero sign==0 boundary)
        const float sgn = (i & 1) ? rf(0.05f, 2.0f) : -rf(0.05f, 2.0f);
        if (0.f >= sgn) ++negCount; else ++posCount;
        ++featHist[feature];

        FeatIn s{};
        s.b0e = F4{ b0e.x, b0e.y, b0e.z, 0 }; s.b1e = F4{ b1e.x, b1e.y, b1e.z, 0 };
        s.c00 = F4{ a0[0].x, a0[0].y, a0[0].z, 0 }; s.c01 = F4{ a0[1].x, a0[1].y, a0[1].z, 0 }; s.c02 = F4{ a0[2].x, a0[2].y, a0[2].z, 0 };
        s.p0 = F4{ p0.x, p0.y, p0.z, 0 };
        s.c10 = F4{ a1[0].x, a1[0].y, a1[0].z, 0 }; s.c11 = F4{ a1[1].x, a1[1].y, a1[1].z, 0 }; s.c12 = F4{ a1[2].x, a1[2].y, a1[2].z, 0 };
        s.p1 = F4{ p1.x, p1.y, p1.z, 0 };
        s.misc = F4{ float(feature), sgn, 0, 0 };
        in[i] = s;

        V3 mtd, c0, c1, c2, p, t1n[3], t1nP, ln; float e0, e1;
        refFeature(b0e, b1e, a0, p0, a1, p1, feature, sgn, mtd, c0, c1, c2, p, e0, e1, t1n, t1nP, ln);
        FeatOut o{};
        o.mtd = F4{ mtd.x, mtd.y, mtd.z, 0 };
        o.ntCol0 = F4{ c0.x, c0.y, c0.z, 0 }; o.ntCol1 = F4{ c1.x, c1.y, c1.z, 0 }; o.ntCol2 = F4{ c2.x, c2.y, c2.z, 0 };
        o.ntP = F4{ p.x, p.y, p.z, 0 };
        o.e = F4{ e0, e1, 0, 0 };
        o.t1nCol0 = F4{ t1n[0].x, t1n[0].y, t1n[0].z, 0 }; o.t1nCol1 = F4{ t1n[1].x, t1n[1].y, t1n[1].z, 0 }; o.t1nCol2 = F4{ t1n[2].x, t1n[2].y, t1n[2].z, 0 };
        o.t1nP = F4{ t1nP.x, t1nP.y, t1nP.z, 0 };
        o.localNormal = F4{ ln.x, ln.y, ln.z, 0 };
        ref[i] = o;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<FeatOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct BoxFeatParams { uint32_t numPairs; } p = { N };
    Kernel k("npBoxFeature.spv", "boxFeatureSetup", { (uint32_t)sizeof(FeatIn), (uint32_t)sizeof(FeatOut) }, sizeof(BoxFeatParams));
    k.dispatch(GROUPS, { bIn, bOut }, p);
    dev->flush();

    const auto g = downloadBuffer<FeatOut>(dev, bOut, N);
    auto eq = [](const F4& a, const F4& b) {
        return almostEqual(a.x, b.x, 1e-4f, 1e-5f) && almostEqual(a.y, b.y, 1e-4f, 1e-5f) &&
               almostEqual(a.z, b.z, 1e-4f, 1e-5f);
    };
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const FeatOut& r = ref[i]; const FeatOut& v = g[i];
        bool ok = eq(r.mtd, v.mtd) && eq(r.ntCol0, v.ntCol0) && eq(r.ntCol1, v.ntCol1) && eq(r.ntCol2, v.ntCol2) &&
                  eq(r.ntP, v.ntP) && eq(r.t1nCol0, v.t1nCol0) && eq(r.t1nCol1, v.t1nCol1) && eq(r.t1nCol2, v.t1nCol2) &&
                  eq(r.t1nP, v.t1nP) && eq(r.localNormal, v.localNormal) &&
                  almostEqual(r.e.x, v.e.x, 1e-4f, 1e-5f) && almostEqual(r.e.y, v.e.y, 1e-4f, 1e-5f);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "boxFeatureSetup: %zu mismatches", mm);
    size_t featMin = featHist[0];
    for (int f = 1; f < 6; ++f) featMin = std::min(featMin, featHist[f]);
    PXG_CHECK_MSG(featMin > 0 && negCount > N / 4 && posCount > N / 4,
                  "boxFeatureSetup: coverage thin (featMin=%zu neg=%zu pos=%zu)", featMin, negCount, posCount);

    dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

} // namespace
