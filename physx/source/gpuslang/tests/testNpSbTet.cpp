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

// Verification of npSbTet.slang: constructTetrahedron/createPlane and the
// tetrahedron-tetrahedron satIntersect (GuIntersectionTetrahedronTetrahedron.h)
// plus closestPtTetrahedron (softbodySoftbodyMidPhase.cu).  Two independent
// slack-tracked comparison groups: the GEOMETRY group (plane-flip signs feed
// the SAT's minProject, plus the SAT early-outs and the magSq gate) and the
// CLOSEST group (per-face running-min ties, the normal orientation dots).
// Hint gating and the sqDist seed are exact (integers / uploaded floats).
// KNOWN DATA LIMITATION: the 1e-5 magSq validity gate of the edge-cross axes
// is never load-bearing here (all cross products are >> 1e-3 for these tets),
// exactly as in the tri-tri SAT test (tranche 57); a wrong gate constant
// would not be detected by this data.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct StIn { F4 a0, b0, c0, d0, a1, b1, c1, d1, point, misc; U4 flags; };
struct StOut { F4 planes0[4]; F4 centroid_sep, closest_sq, normal; };
struct StParams { uint32_t numTests, pad0, pad1, pad2; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static const float MARG = 1e-4f;
static const float MAXF = 3.4028235e38f;

struct Slack { bool near = false; void note(float l, float r) { if (std::fabs(l - r) < MARG) near = true; } };

static V3 getNormalized(V3 v)
{
    const float m = dot3(v, v);
    return m > 0 ? muls(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
}
static void makePlane(V3 p0, V3 p1, V3 p2, V3& n, float& d)
{
    n = getNormalized(cross3(sub(p1, p0), sub(p2, p0)));
    d = -dot3(p0, n);
}
static void createPlane(V3 pa, V3 pb, V3 pc, V3 pd, V3& n, float& d, Slack& sl)
{
    makePlane(pa, pb, pc, n, d);
    const float distance = dot3(n, pd) + d;
    sl.note(distance, 0.0f);
    if (distance > 0.0f) { n = neg(n); d = -d; }
}
static float minProject(V3 n, float d, const V3* v)
{
    return std::min(std::min(dot3(n, v[0]), dot3(n, v[1])), std::min(dot3(n, v[2]), dot3(n, v[3]))) + d;
}
static float testAxis(V3 axis, const V3* a0, const V3* a1)
{
    float min0 = dot3(a0[0], axis), max0 = min0;
    float min1 = dot3(a1[0], axis), max1 = min1;
    for (int i = 1; i < 4; ++i)
    {
        const float p0 = dot3(a0[i], axis), p1 = dot3(a1[i], axis);
        min0 = std::min(p0, min0); max0 = std::max(p0, max0);
        min1 = std::min(p1, min1); max1 = std::max(p1, max1);
    }
    return std::max(min1 - max0, min0 - max1);
}

static V3 closestPtPointTriangle2(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { const float v = d1 / (d1 - d3); return add(a, muls(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { const float w = d2 / (d2 - d6); return add(a, muls(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return add(b, muls(sub(c, b), w)); }
    const float denom = 1.0f / (va + vb + vc);
    return add(add(a, muls(ab, vb * denom)), muls(ac, vc * denom));
}

struct Ref
{
    V3 pn[4]; float pd[4];
    V3 centroid;
    float sep;
    V3 closest, normal;
    float sqDist;
    bool updated;
};

static Ref refAll(const V3* v0, const V3* v1, float tolerance, V3 point, uint32_t hint, float seed,
                  Slack& slGeo, Slack& slClosest)
{
    Ref r;
    createPlane(v0[0], v0[1], v0[2], v0[3], r.pn[0], r.pd[0], slGeo);
    createPlane(v0[0], v0[1], v0[3], v0[2], r.pn[1], r.pd[1], slGeo);
    createPlane(v0[0], v0[2], v0[3], v0[1], r.pn[2], r.pd[2], slGeo);
    createPlane(v0[1], v0[2], v0[3], v0[0], r.pn[3], r.pd[3], slGeo);
    r.centroid = muls(add(add(v0[0], v0[1]), v0[2]), 1.0f / 3.0f);

    V3 qn[4]; float qd[4];
    createPlane(v1[0], v1[1], v1[2], v1[3], qn[0], qd[0], slGeo);
    createPlane(v1[0], v1[1], v1[3], v1[2], qn[1], qd[1], slGeo);
    createPlane(v1[0], v1[2], v1[3], v1[1], qn[2], qd[2], slGeo);
    createPlane(v1[1], v1[2], v1[3], v1[0], qn[3], qd[3], slGeo);

    float sep = -MAXF;
    bool done = false;
    for (int i = 0; i < 4 && !done; ++i)
    {
        sep = std::max(sep, minProject(r.pn[i], r.pd[i], v1));
        slGeo.note(sep, tolerance);
        if (sep > tolerance) done = true;
    }
    for (int i = 0; i < 4 && !done; ++i)
    {
        sep = std::max(sep, minProject(qn[i], qd[i], v0));
        slGeo.note(sep, tolerance);
        if (sep > tolerance) done = true;
    }
    if (!done)
    {
        const V3 axes0[6] = { sub(v0[1], v0[0]), sub(v0[2], v0[0]), sub(v0[3], v0[0]), sub(v0[2], v0[1]), sub(v0[3], v0[1]), sub(v0[3], v0[2]) };
        const V3 axes1[6] = { sub(v1[1], v1[0]), sub(v1[2], v1[0]), sub(v1[3], v1[0]), sub(v1[2], v1[1]), sub(v1[3], v1[1]), sub(v1[3], v1[2]) };
        for (int i = 0; i < 6 && !done; ++i)
            for (int j = 0; j < 6 && !done; ++j)
            {
                V3 ax = cross3(axes0[i], axes1[j]);
                const float magSq = dot3(ax, ax);
                if (std::fabs(magSq - 1e-5f) < 1e-6f) slGeo.near = true;
                if (magSq > 1e-5f)
                {
                    ax = muls(ax, 1.0f / std::sqrt(magSq));
                    sep = std::max(sep, testAxis(ax, v0, v1));
                    slGeo.note(sep, tolerance);
                    if (sep > tolerance) done = true;
                }
            }
    }
    r.sep = sep;

    // closestPtTetrahedron
    V3 normal = { 0, 0, 0 }, closest = { 0, 0, 0 };
    float sqDist = seed;
    bool updated = false;
    const V3 ac = sub(v0[2], v0[0]), ab = sub(v0[1], v0[0]), ad = sub(v0[3], v0[0]);
    struct Face { V3 fa, fb, fc, e0, e1, cn; V3 opp; bool flipGt; };
    const V3 bd = sub(v0[3], v0[1]), bc = sub(v0[2], v0[1]);
    const Face faces[4] = {
        { v0[0], v0[1], v0[2], ab, ac, cross3(ac, ab), ad, false },
        { v0[0], v0[1], v0[3], ab, ad, cross3(ab, ad), ac, false },
        { v0[0], v0[2], v0[3], ac, ad, cross3(ac, ad), ab, false },
        { v0[1], v0[2], v0[3], bc, bd, cross3(bc, bd), ab, true },
    };
    for (int k = 0; k < 4; ++k)
    {
        if (!((hint >> k) & 1)) continue;
        const Face& f = faces[k];
        const V3 tp = closestPtPointTriangle2(point, f.fa, f.fb, f.fc, f.e0, f.e1);
        const V3 dv = sub(tp, point);
        const float tSq = dot3(dv, dv);
        slClosest.note(tSq, sqDist);
        if (tSq < sqDist)
        {
            V3 n = getNormalized(f.cn);
            const float od = dot3(f.opp, n);
            slClosest.note(od, 0.0f);
            if (f.flipGt ? od > 0.0f : od < 0.0f) n = neg(n);
            normal = n;
            closest = tp;
            sqDist = tSq;
            updated = true;
        }
    }
    r.closest = closest; r.normal = normal; r.sqDist = sqDist; r.updated = updated;
    return r;
}

PXG_TEST(npSbTet_all)
{
    std::mt19937 rng(83021);
    const uint32_t N = 8192;
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    std::vector<StIn> in(N);
    std::vector<Ref> ref(N);
    std::vector<uint8_t> skipGeo(N, 0), skipClosest(N, 0);
    size_t overlapping = 0, separated = 0, updated = 0, seeded = 0, noUpdate = 0, seedBlocked = 0;

    for (uint32_t t = 0; t < N; ++t)
    {
        // non-degenerate tets (|signed volume| > 0.05), tet1 near tet0 half the time
        V3 v0[4], v1[4];
        auto genTet = [&](V3* v, V3 base) {
            for (;;)
            {
                for (int k = 0; k < 4; ++k)
                    v[k] = { base.x + 0.5f * uv(rng), base.y + 0.5f * uv(rng), base.z + 0.5f * uv(rng) };
                const float vol = dot3(sub(v[1], v[0]), cross3(sub(v[2], v[0]), sub(v[3], v[0])));
                if (std::fabs(vol) > 0.05f) return;
            }
        };
        const V3 base0 = { uv(rng), uv(rng), uv(rng) };
        genTet(v0, base0);
        const bool nearBy = (t & 1) != 0;
        const V3 base1 = nearBy ? V3{ base0.x + 0.3f * uv(rng), base0.y + 0.3f * uv(rng), base0.z + 0.3f * uv(rng) }
                                : V3{ base0.x + 3.0f + u01(rng), base0.y + 3.0f, base0.z };
        genTet(v1, base1);

        const float tolerance = 0.005f + 0.04f * u01(rng); // varied: a hardcoded-tolerance port fails
        const V3 point = { base0.x + 0.8f * uv(rng), base0.y + 0.8f * uv(rng), base0.z + 0.8f * uv(rng) };
        const uint32_t hint = t % 16;
        const float seed = (t % 3 == 0) ? (0.002f + 0.05f * u01(rng)) : MAXF;
        if (seed != MAXF) ++seeded;

        StIn e;
        e.a0 = f4(v0[0]); e.b0 = f4(v0[1]); e.c0 = f4(v0[2]); e.d0 = f4(v0[3]);
        e.a1 = f4(v1[0]); e.b1 = f4(v1[1]); e.c1 = f4(v1[2]); e.d1 = f4(v1[3]);
        e.point = f4(point);
        e.misc = { tolerance, seed, 0, 0 };
        e.flags = { hint, 0, 0, 0 };
        in[t] = e;

        Slack sg, sc;
        ref[t] = refAll(v0, v1, tolerance, point, hint, seed, sg, sc);
        skipGeo[t] = sg.near ? 1 : 0;
        skipClosest[t] = sc.near ? 1 : 0;
        if (ref[t].sep > tolerance) ++separated; else ++overlapping;
        if (ref[t].updated) ++updated; else ++noUpdate;
        if (!ref[t].updated && hint != 0 && seed != MAXF) ++seedBlocked; // the seed genuinely blocked faces
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 127) / 128;
    std::vector<StOut> outInit(N);
    PxgBufferHandle bI = createBufferFrom(dev, in);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);

    StParams prm = { N, 0, 0, 0 };
    Kernel k("npSbTet.spv", "ctSbTet",
             { (uint32_t)sizeof(StIn), (uint32_t)sizeof(StOut) }, sizeof(StParams));
    k.dispatch(GROUPS, { bI, bO }, prm);
    dev->flush();

    const auto g = downloadBuffer<StOut>(dev, bO, N);
    size_t mmPlane = 0, mmCent = 0, mmSep = 0, mmClosest = 0, geoSkipped = 0, clSkipped = 0;
    auto eq = [](float a, float b, float rel, float abs) { return almostEqual(a, b, rel, abs); };
    for (uint32_t t = 0; t < N; ++t)
    {
        const Ref& r = ref[t];
        const StOut& v = g[t];
        if (skipGeo[t]) ++geoSkipped;
        else
        {
            for (int i = 0; i < 4; ++i)
                if (!eq(v.planes0[i].x, r.pn[i].x, 1e-4f, 1e-4f) || !eq(v.planes0[i].y, r.pn[i].y, 1e-4f, 1e-4f) ||
                    !eq(v.planes0[i].z, r.pn[i].z, 1e-4f, 1e-4f) || !eq(v.planes0[i].w, r.pd[i], 1e-4f, 1e-4f)) ++mmPlane;
            if (!eq(v.centroid_sep.x, r.centroid.x, 1e-5f, 1e-5f) || !eq(v.centroid_sep.y, r.centroid.y, 1e-5f, 1e-5f) ||
                !eq(v.centroid_sep.z, r.centroid.z, 1e-5f, 1e-5f)) ++mmCent;
            if (!eq(v.centroid_sep.w, r.sep, 1e-4f, 1e-4f)) ++mmSep;
        }
        if (skipClosest[t]) ++clSkipped;
        else
        {
            bool bad = false;
            if (r.updated)
            {
                if (!eq(v.closest_sq.x, r.closest.x, 1e-4f, 1e-4f) || !eq(v.closest_sq.y, r.closest.y, 1e-4f, 1e-4f) ||
                    !eq(v.closest_sq.z, r.closest.z, 1e-4f, 1e-4f) || !eq(v.closest_sq.w, r.sqDist, 1e-4f, 1e-4f) ||
                    !eq(v.normal.x, r.normal.x, 1e-4f, 1e-4f) || !eq(v.normal.y, r.normal.y, 1e-4f, 1e-4f) ||
                    !eq(v.normal.z, r.normal.z, 1e-4f, 1e-4f)) bad = true;
            }
            else
            {
                // untouched: sqDist stays the seed bit-exactly, outputs stay zero
                if (v.closest_sq.w != r.sqDist ||
                    v.closest_sq.x != 0 || v.closest_sq.y != 0 || v.closest_sq.z != 0 ||
                    v.normal.x != 0 || v.normal.y != 0 || v.normal.z != 0) bad = true;
            }
            if (bad) ++mmClosest;
        }
    }
    PXG_CHECK_MSG(mmPlane == 0, "npSbTet: %zu plane mismatches", mmPlane);
    PXG_CHECK_MSG(mmCent == 0, "npSbTet: %zu centroid mismatches", mmCent);
    PXG_CHECK_MSG(mmSep == 0, "npSbTet: %zu separation mismatches", mmSep);
    PXG_CHECK_MSG(mmClosest == 0, "npSbTet: %zu closest mismatches", mmClosest);
    PXG_CHECK_MSG(geoSkipped < N / 4 && clSkipped < N / 4, "npSbTet: too many skips (geo=%zu cl=%zu)", geoSkipped, clSkipped);
    PXG_CHECK_MSG(overlapping > 1000 && separated > 1000 && updated > 3000 && noUpdate > 500 && seeded > 2000 && seedBlocked > 100,
                  "npSbTet: coverage thin (ov=%zu sep=%zu upd=%zu noupd=%zu seed=%zu blocked=%zu)",
                  overlapping, separated, updated, noUpdate, seeded, seedBlocked);

    dev->destroyBuffer(bI); dev->destroyBuffer(bO);
}

} // namespace
