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

// Verification of npCvxWitnessPoly.slang: getPolygonIndexFromLocalWitness() from
// cudaGJKEPA.cu.  For each convex polyhedron (sphere-tangent planes in vertex
// space) and a witness point + direction, the shader returns the best-matching
// polygon index.  The C++ reference computes the same selection; because the
// exact lane tie-break only matters on measure-zero ties, the reference uses the
// equivalent global form (all-behind -> index 0; else argmin of plane-normal
// projection over the tolerance-qualifying polygons) and margin-skips any case
// whose pd>=eps gate, tolerance gate, or projection winner sits within a small
// margin of a boundary (a GPU-vs-CPU flip there is measure-zero).

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
struct WitnessIn { F4 scaleNb, rot, witness, nTol; };
struct WitnessOut { uint32_t index, pad0, pad1, pad2; };

const uint32_t MAXPOLYS = 48;
const float MAXF = 3.4028235e38f;
const float LOWER_EPS = 1e-2f, UPPER_EPS = 5e-2f;
const float TIE = 1.5e-4f;

static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }

static V3 quatRotate(const F4& q, const V3& v)
{
    const float vx = 2.f * v.x, vy = 2.f * v.y, vz = 2.f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
              vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
              vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3 quatRotateInv(const F4& q, const V3& v)
{
    const float vx = 2.f * v.x, vy = 2.f * v.y, vz = 2.f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
              vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
              vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3 shape2Vertex(const V3& v, const V3& scale, const F4& rot)
{
    V3 t = quatRotate(rot, v);
    t.x /= scale.x; t.y /= scale.y; t.z /= scale.z;
    return quatRotateInv(rot, t);
}
static V3 vertex2Shape(const V3& u, const V3& scale, const F4& rot)
{
    V3 t = quatRotate(rot, u);
    t.x *= scale.x; t.y *= scale.y; t.z *= scale.z;
    return quatRotateInv(rot, t);
}
static V3 getNormalized(const V3& v) { float m = dot3(v, v); return m > 0 ? mul(v, 1.f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static float clampf(float v, float lo, float hi) { return std::fmin(std::fmax(v, lo), hi); }

// Reference: equivalent global form of getPolygonIndexFromLocalWitness.
static uint32_t refWitness(const WitnessIn& s, const std::vector<F4>& planes, uint32_t nb, bool& marg)
{
    const V3 scale{ s.scaleNb.x, s.scaleNb.y, s.scaleNb.z };
    const V3 witnessPt{ s.witness.x, s.witness.y, s.witness.z };
    const V3 n{ s.nTol.x, s.nTol.y, s.nTol.z };
    const float radius = s.witness.w, toleranceLength = s.nTol.w;

    const float margin = radius * 0.01f;
    const float tolerance = clampf(margin, toleranceLength * LOWER_EPS, toleranceLength * UPPER_EPS);
    const float eps = -tolerance;
    const V3 localWitness = shape2Vertex(witnessPt, scale, s.rot);

    float minDist = MAXF;
    std::vector<float> td(nb);
    for (uint32_t i = 0; i < nb; ++i)
    {
        const V3 pn{ planes[i].x, planes[i].y, planes[i].z };
        const float pd = dot3(pn, localWitness) + planes[i].w;
        if (std::fabs(pd - eps) < TIE) marg = true; // pd>=eps gate
        td[i] = pd >= eps ? pd : MAXF;
        minDist = std::fmin(minDist, td[i]);
    }

    if (minDist == MAXF)
        return 0u; // all behind -> index 0 (deterministic lowest-lane = polygon 0)

    // qualifying: tolerance > (toleranceDist - minDist); argmin projection.
    float best = MAXF, best2 = MAXF;
    uint32_t idx = 0;
    for (uint32_t i = 0; i < nb; ++i)
    {
        const float gate = td[i] - minDist; // finite for in-front planes
        if (td[i] != MAXF && std::fabs(gate - tolerance) < TIE) marg = true; // pass2 gate
        if (tolerance > gate)
        {
            const V3 pn{ planes[i].x, planes[i].y, planes[i].z };
            const V3 s2v = shape2Vertex(pn, scale, s.rot);
            if (dot3(s2v, s2v) < 1e-12f) marg = true;
            const float proj = dot3(getNormalized(s2v), n);
            if (proj < best) { best2 = best; best = proj; idx = i; }
            else if (proj < best2) { best2 = proj; }
        }
    }
    if ((best2 - best) < TIE) marg = true; // projection near-tie -> winner ambiguous
    return idx;
}

PXG_TEST(npCvxWitnessPoly_index)
{
    std::mt19937 rng(138001);
    const uint32_t N = 14000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<WitnessIn> in(N);
    std::vector<F4> planesFlat(size_t(N) * MAXPOLYS, F4{ 0, 0, 0, 0 });
    std::vector<uint8_t> skip(N, 0);
    std::vector<uint32_t> ref(N);
    size_t nSkip = 0, behind = 0, front = 0, multi = 0, nonId = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nb = 4 + (rng() % (MAXPOLYS - 4 + 1)); // 4..48
        const V3 cV{ rf(-2, 2), rf(-2, 2), rf(-2, 2) }; // hull centre in vertex space
        const float R = rf(0.5f, 1.5f);
        std::vector<F4> planes(nb);
        for (uint32_t f = 0; f < nb; ++f)
        {
            const V3 pn = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            planes[f] = F4{ pn.x, pn.y, pn.z, -dot3(pn, cV) - R };
        }
        if (nb > 32) ++multi;

        // transform: identity for most, mild non-identity for coverage of shape2Vertex
        V3 scale{ 1, 1, 1 };
        F4 rot{ 0, 0, 0, 1 };
        if (i % 4 == 0)
        {
            scale = V3{ rf(0.7f, 1.4f), rf(0.7f, 1.4f), rf(0.7f, 1.4f) };
            V3 ax = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            const float ang = rf(-3.1f, 3.1f), hs = std::sin(ang * 0.5f), hc = std::cos(ang * 0.5f);
            rot = F4{ ax.x * hs, ax.y * hs, ax.z * hs, hc };
            ++nonId;
        }

        // witness (shape space): near the surface (front branch) or deep inside (behind)
        V3 witnessV;
        if (i % 5 == 0)
            witnessV = add(cV, mul(nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }), rf(0.0f, 0.4f) * R)); // inside -> behind
        else
            witnessV = add(cV, mul(nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }), R + rf(0.02f, 0.5f))); // near/outside surface
        const V3 witnessShape = vertex2Shape(witnessV, scale, rot);

        const V3 n = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        const float radius = rf(0.5f, 3.0f);
        const float toleranceLength = rf(0.8f, 1.2f);

        WitnessIn s{};
        s.scaleNb = { scale.x, scale.y, scale.z, float(nb) };
        s.rot = rot;
        s.witness = { witnessShape.x, witnessShape.y, witnessShape.z, radius };
        s.nTol = { n.x, n.y, n.z, toleranceLength };
        in[i] = s;
        for (uint32_t f = 0; f < nb; ++f) planesFlat[size_t(i) * MAXPOLYS + f] = planes[f];

        bool marg = false;
        ref[i] = refWitness(s, planes, nb, marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (ref[i] == 0u) {} // (index 0 legitimately covers both branches)
        // branch counting via a quick recompute of minDist would be redundant; use witness placement proxy
        if (i % 5 == 0) ++behind; else ++front;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<WitnessOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bPlanes = createBufferFrom(dev, planesFlat);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct WitnessParams { uint32_t numTests, maxPolys; } prm = { N, MAXPOLYS };
    Kernel k("npCvxWitnessPoly.spv", "cvxWitnessPoly",
             { (uint32_t)sizeof(WitnessIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(WitnessOut) }, sizeof(WitnessParams));
    k.dispatch(GROUPS, { bIn, bPlanes, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<WitnessOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].index != ref[i]) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "witnessPoly: %zu index mismatches", mm);
    PXG_CHECK_MSG(nSkip < (size_t)N * 3 / 5, "witnessPoly: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(behind > N / 20 && front > N / 20, "witnessPoly: branch coverage thin (%zu/%zu)", behind, front);
    PXG_CHECK_MSG(multi > N / 20, "witnessPoly: multi-block coverage thin (%zu)", multi);
    PXG_CHECK_MSG(nonId > N / 10, "witnessPoly: non-identity transform coverage thin (%zu)", nonId);

    for (PxgBufferHandle h : { bIn, bPlanes, bOut }) dev->destroyBuffer(h);
}

} // namespace
