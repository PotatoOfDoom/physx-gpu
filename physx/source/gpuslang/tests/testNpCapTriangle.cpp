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

// Verification of npCapTriangle.slang: generateContacts (endpoint-projection)
// and generateEEContacts (segment vs 3 triangle edges).  Both emit up to 2
// contacts (point + penetration).  The C++ references transcribe the CUDA; the
// contact count and slots are compared exactly (contacts to tolerance) with
// groups whose barycentric / radius / edge-intersection gates sit within a small
// margin of a boundary skipped (a GPU-vs-CPU flip there is measure-zero).

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
struct GenContactsIn { F4 a, b, c, planeNormal, normal, p, q, misc; };
struct GenEEIn { F4 a, b, c, normal, p, q, misc; };
struct OutRes { F4 c0, c1, meta; };

const float MAXF = 3.4028235e38f;
const float MARG = 1e-3f;

static inline V3 xyz(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static inline bool isValidBary(float v, float w) { const float e = 1e-6f; return (v >= -e && v <= 1 + e) && (w >= -e && w <= 1 + e) && (v + w <= 1 + e); }
static inline void baryMarg(float v, float w, bool& m) { if (std::fabs(v) < MARG || std::fabs(v - 1) < MARG || std::fabs(w) < MARG || std::fabs(w - 1) < MARG || std::fabs(v + w - 1) < MARG) m = true; }

uint32_t refGC(const GenContactsIn& s, F4& o0, F4& o1, bool& marg)
{
    const V3 a = xyz(s.a), b = xyz(s.b), c = xyz(s.c), pn = xyz(s.planeNormal), normal = xyz(s.normal), p = xyz(s.p), q = xyz(s.q);
    const float ir = s.misc.x;
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
    const float td = d00 * d11 - d01 * d01, bd = td > 0 ? 1.f / td : MAXF;
    const float ideom = dot3(pn, normal);
    if (std::fabs(ideom) < MARG) marg = true;
    const float inomp = dot3(pn, mul(ap, -1));
    const float ipt = ideom > 0 ? -(inomp / ideom) : MAXF;
    const V3 cp31 = sub(p, mul(normal, ipt)), pV = sub(cp31, a);
    const float v0 = (d11 * dot3(pV, ab) - d01 * dot3(pV, ac)) * bd, w0 = (d00 * dot3(pV, ac) - d01 * dot3(pV, ab)) * bd;
    const bool con0 = isValidBary(v0, w0); baryMarg(v0, w0, marg);
    if (ideom > 0 && std::fabs(ir - ipt) < MARG) marg = true;
    uint32_t count = 0;
    if (con0 && (ir > ipt)) { o0 = F4{ p.x, p.y, p.z, ipt }; count = 1; }
    const float inomq = dot3(pn, mul(aq, -1));
    const float iqt = ideom > 0 ? -(inomq / ideom) : MAXF;
    const V3 cp41 = sub(q, mul(normal, iqt)), qV = sub(cp41, a);
    const float v1 = (d11 * dot3(qV, ab) - d01 * dot3(qV, ac)) * bd, w1 = (d00 * dot3(qV, ac) - d01 * dot3(qV, ab)) * bd;
    const bool con1 = isValidBary(v1, w1); baryMarg(v1, w1, marg);
    if (ideom > 0 && std::fabs(ir - iqt) < MARG) marg = true;
    if (con1 && (ir > iqt)) { F4 hit{ q.x, q.y, q.z, iqt }; if (count == 0) o0 = hit; else o1 = hit; count++; }
    return count;
}

void refEE1(const V3& p, const V3& q, float sqIR, const V3& normal, const V3& a, const V3& b,
            uint32_t& count, F4& o0, F4& o1, bool& marg)
{
    const V3 ab = sub(b, a);
    const V3 n = crossp(ab, normal);
    const float d = dot3(n, a), np = dot3(n, p), nq = dot3(n, q);
    const float signP = np - d, signQ = nq - d;
    if (std::fabs(signP) < MARG || std::fabs(signQ) < MARG) marg = true;
    if (signP * signQ > 0.f) return;
    const V3 pq = sub(q, p);
    const float npq = dot3(n, pq);
    if (std::fabs(npq) < MARG) marg = true;
    if (npq == 0.f) return;
    const float segT = (d - np) / npq;
    const V3 localA = add(p, mul(pq, segT));
    const V3 perNormal = crossp(normal, pq);
    const V3 ap = sub(localA, a);
    const float nom = dot3(perNormal, ap), denom = dot3(perNormal, ab);
    if (std::fabs(denom) < MARG) marg = true;
    const float tValue = nom / denom;
    if (std::fabs(tValue) < MARG || std::fabs(tValue - 1) < MARG) marg = true;
    if (tValue > 1.f || tValue < 0.f) return;
    const V3 v = sub(ap, mul(ab, tValue));
    const float sqDist = dot3(v, v);
    if (std::fabs(sqIR - sqDist) < MARG) marg = true;
    if (sqIR > sqDist)
    {
        const float sd = dot3(v, normal);
        F4 hit{ localA.x, localA.y, localA.z, sd };
        if (count == 0) o0 = hit; else o1 = hit;
        count++;
    }
}

uint32_t refEE(const GenEEIn& s, F4& o0, F4& o1, bool& marg)
{
    const V3 a = xyz(s.a), b = xyz(s.b), c = xyz(s.c), normal = xyz(s.normal), p = xyz(s.p), q = xyz(s.q);
    const float sqIR = s.misc.x;
    uint32_t count = 0;
    if (count < 2) refEE1(p, q, sqIR, normal, a, b, count, o0, o1, marg);
    if (count < 2) refEE1(p, q, sqIR, normal, b, c, count, o0, o1, marg);
    if (count < 2) refEE1(p, q, sqIR, normal, a, c, count, o0, o1, marg);
    return count;
}

static bool eqPt(const F4& r, const F4& v)
{
    return almostEqual(v.x, r.x, 1e-4f, 1e-5f) && almostEqual(v.y, r.y, 1e-4f, 1e-5f) &&
           almostEqual(v.z, r.z, 1e-4f, 1e-5f) && almostEqual(v.w, r.w, 1e-4f, 1e-4f);
}

PXG_TEST(npCapTriangle_generateContacts)
{
    std::mt19937 rng(56010);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<GenContactsIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<OutRes> ref(N);
    size_t nSkip = 0, c0n = 0, c1n = 0, c2n = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        V3 b = add(a, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.f));
        V3 cc = add(a, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.f));
        const V3 N3 = nrm(crossp(sub(b, a), sub(cc, a)));
        const V3 nrmC = nrm(add(N3, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 0.25f))); // perturbed contact normal
        // endpoint projections spanning inside/outside the triangle
        auto proj = [&](float u, float v) { return add(a, add(mul(sub(b, a), u), mul(sub(cc, a), v))); };
        const V3 p = add(proj(rf(-0.3f, 1.0f), rf(-0.3f, 1.0f)), mul(N3, rf(-1.2f, 1.2f)));
        const V3 q = add(proj(rf(-0.3f, 1.0f), rf(-0.3f, 1.0f)), mul(N3, rf(-1.2f, 1.2f)));
        const float ir = rf(0.2f, 1.1f);
        GenContactsIn s{};
        s.a = { a.x, a.y, a.z, 0 }; s.b = { b.x, b.y, b.z, 0 }; s.c = { cc.x, cc.y, cc.z, 0 };
        s.planeNormal = { N3.x, N3.y, N3.z, 0 }; s.normal = { nrmC.x, nrmC.y, nrmC.z, 0 };
        s.p = { p.x, p.y, p.z, 0 }; s.q = { q.x, q.y, q.z, 0 }; s.misc = { ir, 0, 0, 0 };
        in[i] = s;
        OutRes r{}; bool marg = false;
        const uint32_t cnt = refGC(s, r.c0, r.c1, marg); r.meta.x = float(cnt);
        skip[i] = marg ? 1u : 0u; ref[i] = r;
        if (marg) { ++nSkip; continue; }
        if (cnt == 0) ++c0n; else if (cnt == 1) ++c1n; else ++c2n;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<OutRes> outInit(N);
    std::vector<GenEEIn> dummyEE(16);
    std::vector<OutRes> dummyEEOut(16);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bEE = createBufferFrom(dev, dummyEE);
    PxgBufferHandle bEEOut = createBufferFrom(dev, dummyEEOut);

    struct CapTriParams { uint32_t numTests; } prm = { N };
    Kernel k("npCapTriangle.spv", "capTriGenContacts", { (uint32_t)sizeof(GenContactsIn), (uint32_t)sizeof(OutRes), (uint32_t)sizeof(GenEEIn), (uint32_t)sizeof(OutRes) }, sizeof(CapTriParams));
    k.dispatch(GROUPS, { bIn, bOut, bEE, bEEOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<OutRes>(dev, bOut, N);
    size_t mmCount = 0, mmPt = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const uint32_t rc = (uint32_t)(ref[i].meta.x + 0.5f), gc = (uint32_t)(g[i].meta.x + 0.5f);
        if (rc != gc) { ++mmCount; continue; }
        if (rc >= 1 && !eqPt(ref[i].c0, g[i].c0)) ++mmPt;
        if (rc >= 2 && !eqPt(ref[i].c1, g[i].c1)) ++mmPt;
    }
    PXG_CHECK_MSG(mmCount == 0, "generateContacts: %zu count mismatches", mmCount);
    PXG_CHECK_MSG(mmPt == 0, "generateContacts: %zu contact mismatches", mmPt);
    PXG_CHECK_MSG(nSkip < N / 3, "generateContacts: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(c0n > N / 20 && c1n > N / 20 && c2n > N / 20, "generateContacts: count coverage thin (%zu/%zu/%zu)", c0n, c1n, c2n);

    for (PxgBufferHandle h : { bIn, bOut, bEE, bEEOut }) dev->destroyBuffer(h);
}

PXG_TEST(npCapTriangle_generateEE)
{
    std::mt19937 rng(56020);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<GenEEIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<OutRes> ref(N);
    size_t nSkip = 0, hit = 0, miss = 0, ee2 = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        V3 b = add(a, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.f));
        V3 cc = add(a, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.f));
        const V3 N3 = nrm(crossp(sub(b, a), sub(cc, a)));
        V3 p, q;
        if (i % 2 == 0)
        {
            // segment straddling one edge's plane near its span
            const V3 e0 = a, e1 = (i % 4 == 0) ? b : cc;
            const V3 m = mul(add(e0, e1), rf(0.3f, 0.7f)); // point along the edge
            const V3 edge = sub(e1, e0);
            const V3 side = nrm(crossp(edge, N3)); // edge-plane normal direction
            p = add(add(m, mul(side, rf(0.1f, 0.7f))), mul(N3, rf(-0.4f, 0.4f)));
            q = add(sub(m, mul(side, rf(0.1f, 0.7f))), mul(N3, rf(-0.4f, 0.4f)));
        }
        else { p = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) }; q = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) }; }
        const float sqIR = rf(0.05f, 1.2f);
        GenEEIn s{};
        s.a = { a.x, a.y, a.z, 0 }; s.b = { b.x, b.y, b.z, 0 }; s.c = { cc.x, cc.y, cc.z, 0 };
        s.normal = { N3.x, N3.y, N3.z, 0 }; s.p = { p.x, p.y, p.z, 0 }; s.q = { q.x, q.y, q.z, 0 }; s.misc = { sqIR, 0, 0, 0 };
        in[i] = s;
        OutRes r{}; bool marg = false;
        const uint32_t cnt = refEE(s, r.c0, r.c1, marg); r.meta.x = float(cnt);
        skip[i] = marg ? 1u : 0u; ref[i] = r;
        if (marg) { ++nSkip; continue; }
        if (cnt > 0) ++hit; else ++miss;
        if (cnt == 2) ++ee2;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<OutRes> outInit(N);
    std::vector<GenContactsIn> dummyGC(16);
    std::vector<OutRes> dummyGCOut(16);
    PxgBufferHandle bGC = createBufferFrom(dev, dummyGC);
    PxgBufferHandle bGCOut = createBufferFrom(dev, dummyGCOut);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct CapTriParams { uint32_t numTests; } prm = { N };
    Kernel k("npCapTriangle.spv", "capTriGenEE", { (uint32_t)sizeof(GenContactsIn), (uint32_t)sizeof(OutRes), (uint32_t)sizeof(GenEEIn), (uint32_t)sizeof(OutRes) }, sizeof(CapTriParams));
    k.dispatch(GROUPS, { bGC, bGCOut, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<OutRes>(dev, bOut, N);
    size_t mmCount = 0, mmPt = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const uint32_t rc = (uint32_t)(ref[i].meta.x + 0.5f), gc = (uint32_t)(g[i].meta.x + 0.5f);
        if (rc != gc) { ++mmCount; continue; }
        if (rc >= 1 && !eqPt(ref[i].c0, g[i].c0)) ++mmPt;
        if (rc >= 2 && !eqPt(ref[i].c1, g[i].c1)) ++mmPt;
    }
    PXG_CHECK_MSG(mmCount == 0, "generateEE: %zu count mismatches", mmCount);
    PXG_CHECK_MSG(mmPt == 0, "generateEE: %zu contact mismatches", mmPt);
    PXG_CHECK_MSG(nSkip < N / 2, "generateEE: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(hit > N / 20 && miss > N / 20 && ee2 > N / 20, "generateEE: hit/miss/2-contact mix thin (%zu/%zu/%zu)", hit, miss, ee2);

    for (PxgBufferHandle h : { bGC, bGCOut, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
