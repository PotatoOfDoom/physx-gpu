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

// Verification of npCvxRayPoly.slang: intersectRayPolyhedron() from
// cudaGJKEPA.cu.  Convex polyhedra are built as rotated boxes (6 faces) and
// random sphere-tangent hulls (many faces, exercising the multi-block warp
// path); rays are aimed to hit or miss.  The C++ reference transcribes the same
// block-of-32 warp reduction (shared pre-block interval -> max tEnter / min
// tExit / OR separation).  hit flag + tEnter/tExit are compared (to tolerance),
// with groups whose face decisions sit within a small margin of a boundary
// (parallel-plane eps, separation distToPlane==0, tTemp==running tEnter/tExit,
// final tEnter==tExit) skipped as measure-zero GPU-vs-CPU flips.

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
struct RayPolyIn { F4 a, dir; };  // a.w = nbPolygons
struct RayPolyOut { F4 meta; };   // meta = (hit, tEnter, tExit, 0)

const uint32_t MAXPOLYS = 48;
const float PXG_MAXF = 3.4028235e38f;
const float MARG = 1e-4f;

static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }

// Exact transcription of intersectRayPolyhedron (block-of-32 warp reduction).
// planes: nbPolygons face planes (n.xyz, d).  Returns 1 (hit) with te/tx, else 0.
static uint32_t refRayPoly(const V3& a, const V3& dir, const std::vector<F4>& planes,
                           uint32_t nbPolygons, float& te, float& tx, bool& marg)
{
    const float eps = 1e-7f;
    float tFirst = 0.f, tLast = PXG_MAXF;
    te = 0.f; tx = 0.f;
    const uint32_t WARP = 32;
    for (uint32_t i = 0; i < nbPolygons; i += WARP)
    {
        bool blockSep = false;
        float newFirst = tFirst, newLast = tLast;
        for (uint32_t lane = 0; lane < WARP; ++lane)
        {
            const uint32_t k = i + lane;
            if (k >= nbPolygons) continue;
            const F4& pl = planes[k];
            const V3 n{ pl.x, pl.y, pl.z };
            const float d = pl.w;
            const float denom = dot3(n, dir);
            const float distToPlane = dot3(n, a) + d;
            if (std::fabs(std::fabs(denom) - eps) < 1e-8f) marg = true;
            if (std::fabs(distToPlane) < MARG && std::fabs(denom) < eps) marg = true;
            if (std::fabs(denom) < eps)
            {
                if (distToPlane > 0.f) blockSep = true;
            }
            else
            {
                const float tTemp = -distToPlane / denom;
                const bool con = 0.f > denom;
                const bool con0 = tTemp > tFirst;
                const bool con1 = tLast > tTemp;
                if (std::fabs(tTemp - tFirst) < MARG) marg = true;
                if (tLast < PXG_MAXF && std::fabs(tLast - tTemp) < MARG) marg = true;
                if (con && con0) newFirst = std::fmax(newFirst, tTemp);
                if (con1 && !con) newLast = std::fmin(newLast, tTemp);
            }
        }
        if (blockSep) return 0;
        tFirst = newFirst; tLast = newLast;
        if (std::fabs(tFirst - tLast) < MARG) marg = true;
        if (tFirst > tLast) return 0;
    }
    te = tFirst; tx = tLast;
    return 1;
}

// build a rotated box's 6 face planes into out; returns 6
static uint32_t boxPlanes(const V3& c, const V3& u, const V3& v, const V3& w,
                          float hu, float hv, float hw, std::vector<F4>& out)
{
    const float cu = dot3(u, c), cv = dot3(v, c), cw = dot3(w, c);
    out.push_back(F4{ u.x, u.y, u.z, -cu - hu });
    out.push_back(F4{ -u.x, -u.y, -u.z, cu - hu });
    out.push_back(F4{ v.x, v.y, v.z, -cv - hv });
    out.push_back(F4{ -v.x, -v.y, -v.z, cv - hv });
    out.push_back(F4{ w.x, w.y, w.z, -cw - hw });
    out.push_back(F4{ -w.x, -w.y, -w.z, cw - hw });
    return 6;
}

static bool close1(float r, float v, float at, float rt) { return almostEqual(v, r, rt, at); }

PXG_TEST(npCvxRayPoly_intersect)
{
    std::mt19937 rng(135001);
    const uint32_t N = 14000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<RayPolyIn> in(N);
    std::vector<F4> planesFlat(size_t(N) * MAXPOLYS, F4{ 0, 0, 0, 0 });
    std::vector<uint8_t> skip(N, 0);
    std::vector<RayPolyOut> ref(N);
    size_t nSkip = 0, hit = 0, miss = 0, multi = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        V3 c{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        std::vector<F4> planes;
        uint32_t nb;
        V3 a0, dir;

        if (i % 7 == 2)
        {
            // Axis-aligned box with a ray that is EXACTLY parallel to the +-x
            // face pair (dir.x == 0) and whose origin lies inside the x-slab, so
            // the parallel faces do NOT separate and the ray transits the box.
            // This deterministically exercises the |denom|<eps parallel branch
            // (with distToPlane < 0), which random rays essentially never hit.
            const V3 ex{ 1, 0, 0 }, ey{ 0, 1, 0 }, ez{ 0, 0, 1 };
            const float hx = rf(0.5f, 1.4f), hy = rf(0.5f, 1.4f), hz = rf(0.5f, 1.4f);
            nb = boxPlanes(c, ex, ey, ez, hx, hy, hz, planes);
            // origin inside the x-slab (parallel faces both have distToPlane<0)
            a0 = V3{ c.x + rf(-0.6f, 0.6f) * hx, c.y + rf(-4, 4), c.z + rf(-4, 4) };
            // direction purely in y/z -> dir.x == 0 exactly -> denom==0 for +-x
            float dy = rf(-1, 1), dz = rf(-1, 1);
            if (std::fabs(dy) < 0.2f) dy = 0.5f;
            V3 raw{ 0, dy, dz };
            dir = nrm(raw);
            dir.x = 0.0f; // guarantee exact parallelism to the x faces
        }
        else
        {
            // random orthonormal frame
            V3 u = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            V3 tmp = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            V3 w = nrm(crossp(u, tmp));
            V3 v = nrm(crossp(w, u));
            if (i % 3 != 0)
            {
                nb = boxPlanes(c, u, v, w, rf(0.4f, 1.4f), rf(0.4f, 1.4f), rf(0.4f, 1.4f), planes);
            }
            else
            {
                // sphere-tangent hull: many faces, exercises multi-block path
                const float R = rf(0.5f, 1.5f);
                nb = 30 + (rng() % 15); // 30..44 faces -> two 32-blocks
                for (uint32_t f = 0; f < nb; ++f)
                {
                    V3 n = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
                    planes.push_back(F4{ n.x, n.y, n.z, -dot3(n, c) - R });
                }
                multi++;
            }

            // ray: aim through or near the center
            a0 = V3{ rf(-6, 6), rf(-6, 6), rf(-6, 6) };
            V3 target = add(c, V3{ rf(-1.6f, 1.6f), rf(-1.6f, 1.6f), rf(-1.6f, 1.6f) });
            dir = (i % 5 == 0) ? nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }) : nrm(sub(target, a0));
        }

        RayPolyIn s{};
        s.a = { a0.x, a0.y, a0.z, float(nb) };
        s.dir = { dir.x, dir.y, dir.z, 0 };
        in[i] = s;
        for (uint32_t f = 0; f < nb; ++f) planesFlat[size_t(i) * MAXPOLYS + f] = planes[f];

        RayPolyOut r{}; bool marg = false;
        float te, tx;
        const uint32_t h = refRayPoly(a0, dir, planes, nb, te, tx, marg);
        r.meta = F4{ float(h), te, tx, 0 };
        skip[i] = marg ? 1u : 0u; ref[i] = r;
        if (marg) { ++nSkip; continue; }
        if (h) ++hit; else ++miss;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<RayPolyOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bPlanes = createBufferFrom(dev, planesFlat);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct RayPolyParams { uint32_t numTests, maxPolys; } prm = { N, MAXPOLYS };
    Kernel k("npCvxRayPoly.spv", "cvxRayPoly",
             { (uint32_t)sizeof(RayPolyIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(RayPolyOut) }, sizeof(RayPolyParams));
    k.dispatch(GROUPS, { bIn, bPlanes, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<RayPolyOut>(dev, bOut, N);
    size_t mmHit = 0, mmT = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const uint32_t rh = (uint32_t)(ref[i].meta.x + 0.5f), gh = (uint32_t)(g[i].meta.x + 0.5f);
        if (rh != gh) { ++mmHit; continue; }
        if (rh == 1)
        {
            if (!close1(ref[i].meta.y, g[i].meta.y, 1e-4f, 1e-4f)) ++mmT;
            if (!close1(ref[i].meta.z, g[i].meta.z, 1e-4f, 1e-4f)) ++mmT;
        }
    }
    PXG_CHECK_MSG(mmHit == 0, "rayPoly: %zu hit-flag mismatches", mmHit);
    PXG_CHECK_MSG(mmT == 0, "rayPoly: %zu tEnter/tExit mismatches", mmT);
    PXG_CHECK_MSG(nSkip < N / 2, "rayPoly: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(hit > N / 20 && miss > N / 20, "rayPoly: hit/miss mix thin (%zu/%zu)", hit, miss);
    PXG_CHECK_MSG(multi > N / 10, "rayPoly: multi-block coverage thin (%zu)", multi);

    for (PxgBufferHandle h : { bIn, bPlanes, bOut }) dev->destroyBuffer(h);
}

} // namespace
