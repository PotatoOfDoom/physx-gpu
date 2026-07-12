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

// Verification of npCvxRefFrame.slang: the reference-frame + path-decision core
// of generateConvexContacts (cudaGJKEPA.cu L891-937).  Two hulls' best polygons
// are chosen (T138), their planes built/normalized/transformed, projected onto
// the GJK normal, and the reference-vs-incident decision + ortho gate computed.
// The reference transcribes the CUDA; the plane indices + gate booleans are
// compared exactly, the planes/projections to tolerance, with T138 near-ties,
// the project comparison, the ortho gate and near-degenerate normals skipped.

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
struct RefFrameIn
{
    uint32_t base0, base1, nbP0, nbP1;
    F4 scale0Rad, rot0, scale1Rad, rot1, witnessA, witnessB, normal, aToBq, aToBp;
};
struct RefFrameOut
{
    F4 refN, incNt;
    uint32_t bestPlaneAIdx, bestPlaneBIdx;
    float referenceProject, incidentProject;
    uint32_t orthoGate, refIsReference, pad0, pad1;
};

const uint32_t MAXPLANES = 16;
const float MAXF = 3.4028235e38f;
const float LOWER_EPS = 1e-2f, UPPER_EPS = 5e-2f;
const float TIE = 1.5e-4f;

static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRot(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 shape2Vertex(const V3& v, const V3& sc, const F4& r) { V3 t = qRot(r, v); t.x /= sc.x; t.y /= sc.y; t.z /= sc.z; return qRotInv(r, t); }
static V3 getNormalized(const V3& v) { float m = dot3(v, v); return m > 0 ? mul(v, 1.f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static float clampf(float v, float lo, float hi) { return std::fmin(std::fmax(v, lo), hi); }

static uint32_t refPolyIndex(const std::vector<F4>& planes, uint32_t base, uint32_t nbP, const V3& sc, const F4& rot,
                             const V3& witness, const V3& n, float radius, float tol, bool& marg)
{
    const float margin = radius * 0.01f;
    const float tolerance = clampf(margin, tol * LOWER_EPS, tol * UPPER_EPS);
    const float eps = -tolerance;
    const V3 lw = shape2Vertex(witness, sc, rot);
    float minDist = MAXF; std::vector<float> td(nbP);
    for (uint32_t i = 0; i < nbP; ++i)
    {
        const F4& pl = planes[base + i];
        const V3 pn{ pl.x, pl.y, pl.z };
        const float pd = dot3(pn, lw) + pl.w;
        if (std::fabs(pd - eps) < TIE) marg = true;
        td[i] = pd >= eps ? pd : MAXF;
        minDist = std::fmin(minDist, td[i]);
    }
    if (minDist == MAXF) return 0u;
    float best = MAXF, best2 = MAXF; uint32_t idx = 0;
    for (uint32_t i = 0; i < nbP; ++i)
    {
        const float gate = td[i] - minDist;
        if (td[i] != MAXF && std::fabs(gate - tolerance) < TIE) marg = true;
        if (tolerance > gate)
        {
            const F4& pl = planes[base + i];
            const V3 s2v = shape2Vertex(V3{ pl.x, pl.y, pl.z }, sc, rot);
            if (dot3(s2v, s2v) < 1e-12f) marg = true;
            const float proj = dot3(getNormalized(s2v), n);
            if (proj < best) { best2 = best; best = proj; idx = i; }
            else if (proj < best2) best2 = proj;
        }
    }
    if ((best2 - best) < TIE) marg = true;
    return idx;
}

static bool eqF(float a, float b) { return almostEqual(a, b, 3e-4f, 3e-5f); }
static bool eqP(const F4& r, const F4& v) { return almostEqual(v.x, r.x, 3e-4f, 3e-5f) && almostEqual(v.y, r.y, 3e-4f, 3e-5f) && almostEqual(v.z, r.z, 3e-4f, 3e-5f) && almostEqual(v.w, r.w, 3e-4f, 3e-4f); }

PXG_TEST(npCvxRefFrame_setup)
{
    std::mt19937 rng(144001);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<RefFrameIn> in(N);
    std::vector<F4> pl0(size_t(N) * MAXPLANES, F4{ 0, 0, 0, 0 });
    std::vector<F4> pl1(size_t(N) * MAXPLANES, F4{ 0, 0, 0, 0 });
    std::vector<uint8_t> skip(N, 0);
    std::vector<RefFrameOut> ref(N);
    size_t nSkip = 0, refT = 0, refF = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nbP0 = 3 + (rng() % 12), nbP1 = 3 + (rng() % 12);
        const uint32_t base0 = i * MAXPLANES, base1 = i * MAXPLANES;
        const V3 c0{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, c1{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float R0 = rf(0.6f, 1.4f), R1 = rf(0.6f, 1.4f);
        std::vector<F4> p0(nbP0), p1(nbP1);
        for (uint32_t p = 0; p < nbP0; ++p) { V3 n = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); p0[p] = F4{ n.x, n.y, n.z, -dot3(n, c0) - R0 }; pl0[base0 + p] = p0[p]; }
        for (uint32_t p = 0; p < nbP1; ++p) { V3 n = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); p1[p] = F4{ n.x, n.y, n.z, -dot3(n, c1) - R1 }; pl1[base1 + p] = p1[p]; }

        V3 scale0{ 1, 1, 1 }, scale1{ 1, 1, 1 }; F4 rot0{ 0, 0, 0, 1 }, rot1{ 0, 0, 0, 1 };
        if (i % 3 == 0) { scale0 = V3{ rf(0.8f, 1.3f), rf(0.8f, 1.3f), rf(0.8f, 1.3f) }; rot0 = randQuat(); }
        if (i % 3 == 1) { scale1 = V3{ rf(0.8f, 1.3f), rf(0.8f, 1.3f), rf(0.8f, 1.3f) }; rot1 = randQuat(); }
        const float inS0 = rf(0.5f, 2.5f), inS1 = rf(0.5f, 2.5f), tol = rf(0.9f, 1.1f);
        const V3 normal = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        const V3 wA{ rf(-2, 2), rf(-2, 2), rf(-2, 2) }, wB{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        const F4 aToBq = randQuat();
        const V3 aToBp{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };

        RefFrameIn s{};
        s.base0 = base0; s.base1 = base1; s.nbP0 = nbP0; s.nbP1 = nbP1;
        s.scale0Rad = { scale0.x, scale0.y, scale0.z, inS0 }; s.rot0 = rot0;
        s.scale1Rad = { scale1.x, scale1.y, scale1.z, inS1 }; s.rot1 = rot1;
        s.witnessA = { wA.x, wA.y, wA.z, tol }; s.witnessB = { wB.x, wB.y, wB.z, 0 };
        s.normal = { normal.x, normal.y, normal.z, 0 };
        s.aToBq = aToBq; s.aToBp = { aToBp.x, aToBp.y, aToBp.z, 0 };
        in[i] = s;

        // CPU reference
        bool marg = false;
        const V3 normalIn0 = qRotInv(aToBq, mul(normal, -1));
        const V3 wA_v = qRotInv(aToBq, sub(wA, aToBp));
        const uint32_t bA = refPolyIndex(pl0, base0, nbP0, scale0, rot0, wA_v, normalIn0, inS0, tol, marg);
        const uint32_t bB = refPolyIndex(pl1, base1, nbP1, scale1, rot1, wB, normal, inS1, tol, marg);

        const F4 refPl = p1[bB];
        V3 refN = shape2Vertex(V3{ refPl.x, refPl.y, refPl.z }, scale1, rot1); float refD = refPl.w;
        { float mm = dot3(refN, refN); if (mm < 1e-10f) marg = true; float denom = 1.f / std::sqrt(mm); refN = mul(refN, denom); refD *= denom; }
        const F4 incPl = p0[bA];
        V3 incN = shape2Vertex(V3{ incPl.x, incPl.y, incPl.z }, scale0, rot0); float incD = incPl.w;
        { float mm = dot3(incN, incN); if (mm < 1e-10f) marg = true; float denom = 1.f / std::sqrt(mm); incN = mul(incN, denom); incD *= denom; }

        const float referenceProject = std::fabs(dot3(refN, normal));
        const float incidentProject = std::fabs(dot3(incN, normalIn0));

        V3 incNt = qRot(aToBq, incN); float incDt = incD - dot3(aToBp, incNt);
        { float mm = dot3(incNt, incNt); if (mm < 1e-10f) marg = true; float denom = 1.f / std::sqrt(mm); incNt = mul(incNt, denom); incDt *= denom; }

        const float orthoEps = 1e-6f;
        const float rDot = std::fabs(dot3(normal, refN)), iDot = std::fabs(dot3(normal, incNt));
        if (std::fabs(rDot - orthoEps) < TIE || std::fabs(iDot - orthoEps) < TIE) marg = true;
        const bool orthoGate = rDot >= orthoEps && iDot >= orthoEps;
        if (std::fabs(referenceProject - incidentProject) < TIE) marg = true;
        const bool refIsReference = referenceProject > incidentProject;

        RefFrameOut r{};
        r.refN = { refN.x, refN.y, refN.z, refD };
        r.incNt = { incNt.x, incNt.y, incNt.z, incDt };
        r.bestPlaneAIdx = bA; r.bestPlaneBIdx = bB;
        r.referenceProject = referenceProject; r.incidentProject = incidentProject;
        r.orthoGate = orthoGate ? 1u : 0u; r.refIsReference = refIsReference ? 1u : 0u;
        ref[i] = r;
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (refIsReference) ++refT; else ++refF;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<RefFrameOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bP0 = createBufferFrom(dev, pl0);
    PxgBufferHandle bP1 = createBufferFrom(dev, pl1);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct RefFrameParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxRefFrame.spv", "cvxRefFrame",
             { (uint32_t)sizeof(RefFrameIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(RefFrameOut) }, sizeof(RefFrameParams));
    k.dispatch(GROUPS, { bIn, bP0, bP1, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<RefFrameOut>(dev, bOut, N);
    size_t mmIdx = 0, mmPlane = 0, mmProj = 0, mmGate = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const RefFrameOut& a = ref[i]; const RefFrameOut& b = g[i];
        if (a.bestPlaneAIdx != b.bestPlaneAIdx || a.bestPlaneBIdx != b.bestPlaneBIdx) { ++mmIdx; continue; }
        if (!eqP(a.refN, b.refN) || !eqP(a.incNt, b.incNt)) ++mmPlane;
        if (!eqF(a.referenceProject, b.referenceProject) || !eqF(a.incidentProject, b.incidentProject)) ++mmProj;
        if (a.orthoGate != b.orthoGate || a.refIsReference != b.refIsReference) ++mmGate;
    }
    PXG_CHECK_MSG(mmIdx == 0, "refFrame: %zu polygon-index mismatches", mmIdx);
    PXG_CHECK_MSG(mmPlane == 0, "refFrame: %zu plane mismatches", mmPlane);
    PXG_CHECK_MSG(mmProj == 0, "refFrame: %zu projection mismatches", mmProj);
    PXG_CHECK_MSG(mmGate == 0, "refFrame: %zu gate/decision mismatches", mmGate);
    PXG_CHECK_MSG(nSkip < (size_t)N * 3 / 5, "refFrame: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(refT > N / 20 && refF > N / 20, "refFrame: reference-decision coverage thin (%zu/%zu)", refT, refF);

    for (PxgBufferHandle h : { bIn, bP0, bP1, bOut }) dev->destroyBuffer(h);
}

} // namespace
