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

// Verification of npCvxCvxEE.slang: the convex-convex edge-edge contact
// generator generateEE() from cudaGJKEPA.cu.  The reference edge (a,b) and the
// colliding edge (p,q) are crossed against a contact normal; if the crossing
// lands within the expanded reference-edge span and within the inflated radius
// (signed distance), a contact is produced: localPointA transformed back through
// aToB.transformInv, localPointB, and (normal, signedDist).  The C++ reference
// transcribes the CUDA exactly; the hit flag and the three output vec4s are
// compared (points to tolerance) with groups whose gate values sit within a
// small margin of a decision boundary skipped (a GPU-vs-CPU flip there is
// measure-zero).

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
struct GenEEIn { F4 p, q, a, b, normal, aToBq, aToBp; }; // aToBp.w = inflatedRadius
struct OutRes { F4 localA, localB, normalPen, meta; };

const float MARG = 1e-3f;

static inline V3 xyz(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }

// mirrors PxQuat::rotateInv()
static V3 quatRotateInv(const F4& q, const V3& v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3{
        (vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2) };
}

// Exact transcription of generateEE() from cudaGJKEPA.cu.
// Returns 1 on hit (fills la/lb/nrmPen), 0 on miss; sets marg for near-boundary cases.
static uint32_t refEE(const GenEEIn& s, F4& la, F4& lb, F4& nrmPen, bool& marg)
{
    const V3 p = xyz(s.p), q = xyz(s.q), a = xyz(s.a), b = xyz(s.b), normal = xyz(s.normal);
    const V3 aToBp = xyz(s.aToBp);
    const float inflatedRadius = s.aToBp.w;
    const float expandedRatio = 0.005f;

    const V3 ab = sub(b, a);
    const V3 n = crossp(ab, normal);
    const float d = dot3(n, a);
    const float np = dot3(n, p);
    const float nq = dot3(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    const float temp = signP * signQ;
    if (std::fabs(signP) < MARG || std::fabs(signQ) < MARG) marg = true;
    if (temp > 0.f) return 0;

    const V3 pq = sub(q, p);
    const float npq = dot3(n, pq);
    if (std::fabs(npq) < MARG) marg = true;
    if (npq == 0.f) return 0;

    const float segTValue = (d - np) / npq;
    const V3 localPointA = add(mul(pq, segTValue), p);

    const V3 perNormal = crossp(normal, pq);
    const V3 ap = sub(localPointA, a);
    const float nom = dot3(perNormal, ap);
    const float denom = dot3(perNormal, ab);
    // denom == npq (scalar triple product identity); when it is small the 2D
    // segment/segment intersection is ill-conditioned and localPointA / v / localB
    // scale like 1/denom, amplifying FMA-contraction rounding past tolerance.
    if (std::fabs(denom) < 0.03f) marg = true;
    const float tValue = nom / denom;

    const float maxT = 1.f + expandedRatio;
    const float minT = -expandedRatio;
    if (std::fabs(tValue - maxT) < MARG || std::fabs(tValue - minT) < MARG) marg = true;
    if (tValue > maxT || minT > tValue) return 0;

    const V3 v = sub(ap, mul(ab, tValue));
    const float signedDist = dot3(v, normal);
    if (std::fabs(inflatedRadius - signedDist) < MARG) marg = true;
    if (inflatedRadius >= signedDist)
    {
        const V3 lpB = sub(localPointA, v);
        const V3 lpA = quatRotateInv(s.aToBq, sub(localPointA, aToBp));
        la = F4{ lpA.x, lpA.y, lpA.z, 0.f };
        lb = F4{ lpB.x, lpB.y, lpB.z, 0.f };
        nrmPen = F4{ normal.x, normal.y, normal.z, signedDist };
        return 1;
    }
    return 0;
}

static bool eqPt(const F4& r, const F4& v)
{
    return almostEqual(v.x, r.x, 1e-4f, 1e-5f) && almostEqual(v.y, r.y, 1e-4f, 1e-5f) &&
           almostEqual(v.z, r.z, 1e-4f, 1e-5f) && almostEqual(v.w, r.w, 1e-4f, 1e-4f);
}

PXG_TEST(npCvxCvxEE_generateEE)
{
    std::mt19937 rng(134001);
    const uint32_t N = 16000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<GenEEIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<OutRes> ref(N);
    size_t nSkip = 0, hit = 0, miss = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // reference edge (a,b)
        V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        V3 b = add(a, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.f));
        // contact normal: random-ish but not degenerate w.r.t. the reference edge
        V3 normal = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });

        V3 p, q;
        if (i % 3 != 0)
        {
            // craft a colliding edge that actually crosses the reference edge:
            // pick a point m along ab, offset it +-normal, and span p,q across the
            // n.(x-a)=0 plane so signP*signQ < 0.
            const V3 ab = sub(b, a);
            const V3 nplane = crossp(ab, normal);           // reference-edge plane normal
            const V3 m = add(a, mul(ab, rf(0.1f, 0.9f)));   // point on ab
            const V3 mm = add(m, mul(normal, rf(-0.6f, 0.6f)));
            const V3 planeDir = nrm(nplane);
            p = add(mm, mul(planeDir, rf(0.05f, 0.9f)));
            q = sub(mm, mul(planeDir, rf(0.05f, 0.9f)));
        }
        else
        {
            p = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) };
            q = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) };
        }

        // random normalized aToB quaternion + translation
        V3 axis = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        const float ang = rf(-3.14159f, 3.14159f);
        const float hs = std::sin(ang * 0.5f), hc = std::cos(ang * 0.5f);
        F4 qrot{ axis.x * hs, axis.y * hs, axis.z * hs, hc };
        V3 tp{ rf(-3, 3), rf(-3, 3), rf(-3, 3) };
        const float inflatedRadius = rf(-0.2f, 1.0f);

        GenEEIn s{};
        s.p = { p.x, p.y, p.z, 0 }; s.q = { q.x, q.y, q.z, 0 };
        s.a = { a.x, a.y, a.z, 0 }; s.b = { b.x, b.y, b.z, 0 };
        s.normal = { normal.x, normal.y, normal.z, 0 };
        s.aToBq = qrot;
        s.aToBp = { tp.x, tp.y, tp.z, inflatedRadius };
        in[i] = s;

        OutRes r{}; bool marg = false;
        const uint32_t cnt = refEE(s, r.localA, r.localB, r.normalPen, marg);
        r.meta.x = float(cnt);
        skip[i] = marg ? 1u : 0u; ref[i] = r;
        if (marg) { ++nSkip; continue; }
        if (cnt > 0) ++hit; else ++miss;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<OutRes> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct CvxCvxEEParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCvxEE.spv", "cvxCvxGenEE", { (uint32_t)sizeof(GenEEIn), (uint32_t)sizeof(OutRes) }, sizeof(CvxCvxEEParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<OutRes>(dev, bOut, N);
    size_t mmCount = 0, mmA = 0, mmB = 0, mmN = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const uint32_t rc = (uint32_t)(ref[i].meta.x + 0.5f), gc = (uint32_t)(g[i].meta.x + 0.5f);
        if (rc != gc) { ++mmCount; continue; }
        if (rc == 1)
        {
            if (!eqPt(ref[i].localA, g[i].localA)) ++mmA;
            if (!eqPt(ref[i].localB, g[i].localB)) ++mmB;
            if (!eqPt(ref[i].normalPen, g[i].normalPen)) ++mmN;
        }
    }
    PXG_CHECK_MSG(mmCount == 0, "generateEE: %zu hit-flag mismatches", mmCount);
    PXG_CHECK_MSG(mmA == 0, "generateEE: %zu localA mismatches", mmA);
    PXG_CHECK_MSG(mmB == 0, "generateEE: %zu localB mismatches", mmB);
    PXG_CHECK_MSG(mmN == 0, "generateEE: %zu normalPen mismatches", mmN);
    PXG_CHECK_MSG(nSkip < N / 2, "generateEE: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(hit > N / 20 && miss > N / 20, "generateEE: hit/miss mix thin (%zu/%zu)", hit, miss);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
