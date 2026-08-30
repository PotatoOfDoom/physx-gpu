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

// Verification of npTriangle.slang: triPointDistance (Ericson closest-point-on-
// triangle with Voronoi mask + convexity-gated flags) and triLineIntersect.  The
// closest point / squared distance are compared to tolerance; the discrete
// region mask, generateContact and faceContact flags are compared exactly, with
// groups whose Voronoi-region decision variables sit within a small margin of a
// boundary skipped (a GPU-vs-CPU flip there is measure-zero for the fixed seed).

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
struct V3 { float x, y, z; };
struct TriIn { F4 p, a, b, c; U4 adj; };
struct TriResult { F4 closest; uint32_t mask, gen, face, pad; };
struct LineIn { F4 p, q, a, b, c; };

const uint32_t BOUNDARY = 0xffffffffu, NONCONVEX = 0x80000000u;
const uint32_t eV0 = 1u << 29, eV1 = 2u << 29, eV2 = 3u << 29, eE01 = 4u << 29, eE02 = 5u << 29, eE12 = 6u << 29;

static inline V3 xyz(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 madd(const V3& a, const V3& b, float s) { return V3{ a.x + b.x * s, a.y + b.y * s, a.z + b.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline bool edgeNonconvex(uint32_t a) { return a != BOUNDARY && (a & NONCONVEX); }

// mirrors sphereTriangle.cuh distancePointTriangleSquared
float refTri(const V3& p, const V3& a, const V3& b, const V3& c, const U4& adj,
             V3& closestP, uint32_t& mask, bool& gen, bool& face, bool& marginal)
{
    const float MARG = 1e-3f;
    face = false;
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    const V3 bp = sub(p, b); const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    const V3 cp = sub(p, c); const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    const float vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    const float edgeTol = (va + vb + vc) * 1e-5f;

    // any Voronoi-region decision variable near its boundary -> marginal.  The
    // last three are the generateContact edge-tolerance boundaries (va/vb/vc >
    // -edgeTol); edgeTol = (va+vb+vc)*1e-5 = |ab x ac|^2 * 1e-5 can exceed MARG
    // for a large triangle, so it must be guarded explicitly, not via |v|<MARG.
    const float ds[] = { d1, d2, d3, d4, d5, d6, va, vb, vc, d4 - d3, d5 - d6,
                         va + edgeTol, vb + edgeTol, vc + edgeTol };
    marginal = false;
    for (float x : ds) if (std::fabs(x) < MARG) marginal = true;

    if (d1 < 0.f && d2 < 0.f)
    {
        mask = eV0; gen = (!edgeNonconvex(adj.x)) || (!edgeNonconvex(adj.z)); closestP = a; return dot3(ap, ap);
    }
    if (d3 > 0.f && d4 < d3)
    {
        mask = eV1; gen = (!edgeNonconvex(adj.x)) || (!edgeNonconvex(adj.y)); closestP = b; return dot3(bp, bp);
    }
    if (d6 > 0.f && d5 < d6)
    {
        mask = eV2; gen = (!edgeNonconvex(adj.y)) || (!edgeNonconvex(adj.z)); closestP = c; return dot3(cp, cp);
    }
    if (vc < 0.f && d1 >= 0.f && d3 <= 0.f)
    {
        mask = eE01; gen = !edgeNonconvex(adj.x) || vc > -edgeTol;
        const float v = d1 / (d1 - d3); closestP = madd(a, ab, v); const V3 d = sub(closestP, p); return dot3(d, d);
    }
    if (vb < 0.f && d2 >= 0.f && d6 <= 0.f)
    {
        mask = eE02; gen = !edgeNonconvex(adj.z) || vb > -edgeTol;
        const float w = d2 / (d2 - d6); closestP = madd(a, ac, w); const V3 d = sub(closestP, p); return dot3(d, d);
    }
    if (va < 0.f && d4 >= d3 && d5 >= d6)
    {
        mask = eE12; gen = !edgeNonconvex(adj.y) || va > -edgeTol;
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); closestP = madd(b, sub(c, b), w); const V3 d = sub(closestP, p); return dot3(d, d);
    }
    gen = true; face = true; mask = 0u;
    // P9.36: GuDistancePointTriangle.h:94-97 is a RECIPROCAL and the two scalars scaled
    // BEFORE the vector sum -- `denom = 1/(va+vb+vc); v = vb*denom; w = vc*denom;
    // a + ab*v + ac*w`.  This mirror had three divides and the wrong association, and
    // the shader had both too, so the pair agreed.  Note that this fixture compares
    // closestP with almostEqual(1e-4f, 1e-5f) (see below), so it could not have caught
    // the difference even with a correct mirror -- gpuslang.vectorDivide is what does.
    const float denom = 1.0f / (va + vb + vc);
    const float bv = vb * denom, bw = vc * denom;
    closestP = V3{ a.x + ab.x * bv + ac.x * bw, a.y + ab.y * bv + ac.y * bw, a.z + ab.z * bv + ac.z * bw };
    const V3 d = sub(closestP, p); return dot3(d, d);
}

bool refLine(const V3& p, const V3& q, const V3& a, const V3& b, const V3& c)
{
    const V3 pq = sub(q, p), pa = sub(a, p), pb = sub(b, p), pc = sub(c, p);
    const V3 m = V3{ pq.y * pc.z - pq.z * pc.y, pq.z * pc.x - pq.x * pc.z, pq.x * pc.y - pq.y * pc.x };
    const float u = dot3(pb, m), v = -dot3(pa, m);
    const V3 cr = V3{ pq.y * pb.z - pq.z * pb.y, pq.z * pb.x - pq.x * pb.z, pq.x * pb.y - pq.y * pb.x };
    const float w = dot3(cr, pa);
    return u >= 0.f && v >= 0.f && w >= 0.f;
}

PXG_TEST(npTriangle_pointDistance)
{
    std::mt19937 rng(54010);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<TriIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<TriResult> ref(N);
    size_t nSkip = 0, regionHist[7] = { 0, 0, 0, 0, 0, 0, 0 };
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        V3 b = madd(a, V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.0f);
        V3 c = madd(a, V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.0f);
        V3 p{ rf(-4, 4), rf(-4, 4), rf(-4, 4) };
        U4 adj{ (rng() % 3 == 0) ? BOUNDARY : ((rng() % 2) ? (rng() | NONCONVEX) : (rng() & ~NONCONVEX)),
                (rng() % 3 == 0) ? BOUNDARY : ((rng() % 2) ? (rng() | NONCONVEX) : (rng() & ~NONCONVEX)),
                (rng() % 3 == 0) ? BOUNDARY : ((rng() % 2) ? (rng() | NONCONVEX) : (rng() & ~NONCONVEX)), 0 };
        TriIn s{}; s.p = { p.x, p.y, p.z, 0 }; s.a = { a.x, a.y, a.z, 0 }; s.b = { b.x, b.y, b.z, 0 }; s.c = { c.x, c.y, c.z, 0 }; s.adj = adj;
        in[i] = s;

        V3 cp; uint32_t mask; bool gen, face, marg;
        const float dsq = refTri(p, a, b, c, adj, cp, mask, gen, face, marg);
        skip[i] = marg ? 1u : 0u;
        TriResult r{}; r.closest = { cp.x, cp.y, cp.z, dsq }; r.mask = mask; r.gen = gen ? 1u : 0u; r.face = face ? 1u : 0u;
        ref[i] = r;
        if (marg) { ++nSkip; continue; }
        const uint32_t reg = face ? 6u : (mask >> 29) - 1u; // 0..5 for V0..E12, 6 face
        if (reg < 7) ++regionHist[reg];
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<TriResult> outInit(N);
    std::vector<LineIn> dummyLine(16);
    std::vector<uint32_t> dummyU(16, 0);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bLine = createBufferFrom(dev, dummyLine);
    PxgBufferHandle bLineOut = createBufferFrom(dev, dummyU);

    struct TriParams { uint32_t numTests; } p = { N };
    Kernel k("npTriangle.spv", "triPointDistance", { (uint32_t)sizeof(TriIn), (uint32_t)sizeof(TriResult), (uint32_t)sizeof(LineIn), 4 }, sizeof(TriParams));
    k.dispatch(GROUPS, { bIn, bOut, bLine, bLineOut }, p);
    dev->flush();

    const auto g = downloadBuffer<TriResult>(dev, bOut, N);
    size_t mmMask = 0, mmGeom = 0, mmFlag = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const TriResult& r = ref[i]; const TriResult& v = g[i];
        if (r.mask != v.mask || r.face != v.face) { ++mmMask; continue; }
        if (r.gen != v.gen) ++mmFlag;
        if (!almostEqual(v.closest.x, r.closest.x, 1e-4f, 1e-5f) || !almostEqual(v.closest.y, r.closest.y, 1e-4f, 1e-5f) ||
            !almostEqual(v.closest.z, r.closest.z, 1e-4f, 1e-5f) || !almostEqual(v.closest.w, r.closest.w, 1e-4f, 1e-4f)) ++mmGeom;
    }
    PXG_CHECK_MSG(mmMask == 0, "triPointDistance: %zu mask/face mismatches", mmMask);
    PXG_CHECK_MSG(mmFlag == 0, "triPointDistance: %zu generateContact mismatches", mmFlag);
    PXG_CHECK_MSG(mmGeom == 0, "triPointDistance: %zu closest/dist mismatches", mmGeom);
    PXG_CHECK_MSG(nSkip < N / 3, "triPointDistance: too many marginal groups (%zu/%u)", nSkip, N);
    size_t rmin = regionHist[0];
    for (int r = 1; r < 7; ++r) rmin = std::min(rmin, regionHist[r]);
    PXG_CHECK_MSG(rmin > N / 100, "triPointDistance: region coverage thin (%zu %zu %zu %zu %zu %zu %zu)",
                  regionHist[0], regionHist[1], regionHist[2], regionHist[3], regionHist[4], regionHist[5], regionHist[6]);

    for (PxgBufferHandle h : { bIn, bOut, bLine, bLineOut }) dev->destroyBuffer(h);
}

PXG_TEST(npTriangle_lineIntersect)
{
    std::mt19937 rng(54020);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<LineIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<uint8_t> ref(N);
    size_t nSkip = 0, hit = 0, miss = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        V3 b = madd(a, V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.0f);
        V3 c = madd(a, V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.0f);
        V3 p, q;
        if (i % 2 == 0)
        {
            // line through a random interior point (~50% hit depending on winding)
            float w0 = rf(0.1f, 0.9f), w1 = rf(0.05f, 0.85f); if (w0 + w1 > 0.95f) { w0 *= 0.5f; w1 *= 0.5f; }
            const float w2 = 1.f - w0 - w1;
            const V3 inside{ a.x * w0 + b.x * w1 + c.x * w2, a.y * w0 + b.y * w1 + c.y * w2, a.z * w0 + b.z * w1 + c.z * w2 };
            const V3 dir{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            p = madd(inside, dir, -rf(0.5f, 2.f)); q = madd(inside, dir, rf(0.5f, 2.f));
        }
        else { p = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) }; q = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) }; }
        LineIn s{}; s.p = { p.x, p.y, p.z, 0 }; s.q = { q.x, q.y, q.z, 0 }; s.a = { a.x, a.y, a.z, 0 }; s.b = { b.x, b.y, b.z, 0 }; s.c = { c.x, c.y, c.z, 0 };
        in[i] = s;

        // margin: u,v,w near 0 could flip
        const V3 pq = sub(q, p), pa = sub(a, p), pb = sub(b, p), pc = sub(c, p);
        const V3 m = V3{ pq.y * pc.z - pq.z * pc.y, pq.z * pc.x - pq.x * pc.z, pq.x * pc.y - pq.y * pc.x };
        const float u = dot3(pb, m), v = -dot3(pa, m);
        const V3 cr = V3{ pq.y * pb.z - pq.z * pb.y, pq.z * pb.x - pq.x * pb.z, pq.x * pb.y - pq.y * pb.x };
        const float w = dot3(cr, pa);
        const bool marg = std::fabs(u) < 1e-3f || std::fabs(v) < 1e-3f || std::fabs(w) < 1e-3f;
        skip[i] = marg ? 1u : 0u;
        const bool r = refLine(p, q, a, b, c);
        ref[i] = r ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (r) ++hit; else ++miss;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> outInit(N, 9);
    std::vector<TriIn> dummyTri(16);
    std::vector<TriResult> dummyRes(16);
    PxgBufferHandle bTri = createBufferFrom(dev, dummyTri);
    PxgBufferHandle bRes = createBufferFrom(dev, dummyRes);
    PxgBufferHandle bLine = createBufferFrom(dev, in);
    PxgBufferHandle bLineOut = createBufferFrom(dev, outInit);

    struct TriParams { uint32_t numTests; } p = { N };
    Kernel k("npTriangle.spv", "triLineIntersect", { (uint32_t)sizeof(TriIn), (uint32_t)sizeof(TriResult), (uint32_t)sizeof(LineIn), 4 }, sizeof(TriParams));
    k.dispatch(GROUPS, { bTri, bRes, bLine, bLineOut }, p);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bLineOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) { if (skip[i]) continue; if ((g[i] ? 1u : 0u) != ref[i]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "triLineIntersect: %zu mismatches", mm);
    PXG_CHECK_MSG(hit > N / 20 && miss > N / 20, "triLineIntersect: hit/miss mix thin (%zu/%zu)", hit, miss);

    for (PxgBufferHandle h : { bTri, bRes, bLine, bLineOut }) dev->destroyBuffer(h);
}

} // namespace
