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

// Verification of pxgSlangTriTriBody (P9.83b-8, 2/n) through npTriTriBody.slang.
//
// The host mirror below is convexMeshMidphase.cu:1089-1195 transcribed line for
// line, over the engine's own PxVec3 / PxQuat / PxTransform / PxMath and the
// shared __device__ mirrors in pxgTriTriGeomRef.h.  The SDF call in the middle of
// each pass is NOT mirrored: its outputs (sep, dir, point) are inputs here, chosen
// to reach every branch -- the call itself is certified by
// gpuslang.sdfTriangleCollision, against its own trace.
//
// Bitwise throughout, NaN == NaN (see testNpTriTriGeom.cpp).  Each trap the module
// header names (T1..T9) has a case family and, where it is an ORDER claim, a
// host-side discriminator that fails the test if the fixture could not see the
// wrong order.

#include "pxgTestFramework.h"
#include "pxgTriTriBodyRef.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgtest::tritrigeomref;
using namespace pxgtest::tritribodyref;

struct TtcParams
{
    uint32_t numCases, pad0, pad1, pad2;
    uint64_t cases, outBuf, meshInd, meshVerts;
};
static_assert(sizeof(TtcParams) == 48, "TtcParams layout");

struct VertsCase { F4 scale0, rot0, scale1, rot1, aToBp, aToBq; U4 misc; };
struct PostCase { F4 v0, v1, v2, dirSep, point, boxLo, boxHi, scale, rot, aToBp, aToBq, dists; U4 misc; };
static_assert(sizeof(VertsCase) == 112, "VertsCase layout");
static_assert(sizeof(PostCase) == 208, "PostCase layout");

const float PINF = std::numeric_limits<float>::infinity();
const float QNAN = std::numeric_limits<float>::quiet_NaN();

uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float floatOf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
bool sameBits(uint32_t got, float want)
{
    if (std::isnan(want)) return std::isnan(floatOf(got));
    return got == bitsOf(want);
}
bool sameVec(const U4& got, const PxVec3& want)
{
    return sameBits(got.x, want.x) && sameBits(got.y, want.y) && sameBits(got.z, want.z);
}
bool differs(const PxVec3& p, const PxVec3& q)
{
    return bitsOf(p.x) != bitsOf(q.x) || bitsOf(p.y) != bitsOf(q.y) || bitsOf(p.z) != bitsOf(q.z);
}

// The contact and the two passes' mirrors live in pxgTriTriBodyRef.h since P9.83b-8 (4/n),
// where the kernel fixture replays them too.

// ---- inputs --------------------------------------------------------------------

struct Rng
{
    std::mt19937 g;
    explicit Rng(uint32_t seed) : g(seed) {}
    uint32_t u() { return g(); }
    float f(float lo, float hi) { return lo + (hi - lo) * (float(g() % 1000003u) / 1000003.f); }
    F4 unitQuat()
    {
        float a[4], m2 = 0.f;
        for (int k = 0; k < 4; ++k) { a[k] = f(-1.f, 1.f); m2 += a[k] * a[k]; }
        const float inv = 1.f / std::sqrt(m2 > 1e-4f ? m2 : 1.f);
        return F4{ a[0] * inv, a[1] * inv, a[2] * inv, a[3] * inv };
    }
    F4 point(float r) { return F4{ f(-r, r), f(-r, r), f(-r, r), 0.f }; }
    F4 scale()
    {
        F4 s{ f(0.3f, 3.f), f(0.3f, 3.f), f(0.3f, 3.f), 0.f };
        if (g() % 8u == 3u) s.z = -s.z;   // a mirrored mesh scale
        return s;
    }
};

const F4 IDQ{ 0.f, 0.f, 0.f, 1.f };
const F4 ONE3{ 1.f, 1.f, 1.f, 0.f };

void runKernel(const char* entry, uint32_t numCases, uint64_t cases, uint64_t outBuf,
               uint64_t meshInd = 0, uint64_t meshVerts = 0)
{
    TtcParams p{};
    p.numCases = numCases; p.cases = cases; p.outBuf = outBuf; p.meshInd = meshInd; p.meshVerts = meshVerts;
    Kernel k("npTriTriBody.spv", entry, {}, sizeof(TtcParams));
    PXG_CHECK(k.isValid());
    if (!k.isValid()) return;
    k.dispatch((numCases + 63u) / 64u, {}, p);
    testDevice()->flush();
}

} // namespace

// ---------------------------------------------------------------------------
// The vertices of both passes: getTriangleVertices + shape2Vertex (pass 0), and
// vertex2Shape + transformInv + shape2Vertex (pass 1).
// ---------------------------------------------------------------------------
PXG_TEST(npTriTriBody_vertices)
{
    Rng rng(832001u);
    const uint32_t NV = 83u, NT = 57u;
    std::vector<F4> verts(NV);
    for (uint32_t v = 0; v < NV; ++v) { verts[v] = rng.point(3.f); verts[v].w = 777.f; }
    std::vector<U4> inds(NT);
    for (uint32_t t = 0; t < NT; ++t)
    {
        uint32_t x = rng.u() % NV, y, z, w;
        do y = rng.u() % NV; while (y == x);
        do z = rng.u() % NV; while (z == x || z == y);
        do w = rng.u() % NV; while (w == x || w == y || w == z);
        inds[t] = U4{ x, y, z, w };
    }

    const uint32_t N = 2048u;
    std::vector<VertsCase> cs(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        VertsCase s{};
        s.scale0 = rng.scale(); s.rot0 = rng.unitQuat();
        s.scale1 = rng.scale(); s.rot1 = rng.unitQuat();
        s.aToBp = rng.point(4.f); s.aToBq = rng.unitQuat();
        if (i % 16u == 1u) { s.scale0 = ONE3; s.rot0 = IDQ; s.scale1 = ONE3; s.rot1 = IDQ; }
        if (i % 16u == 2u) { s.aToBp = F4{ 0, 0, 0, 0 }; s.aToBq = IDQ; }
        uint32_t enc = 0u;
        if (i % 3u != 0u)
        {
            const uint32_t level = 1u + rng.u() % 8u;   // the engine's range, maxRefinementLevel = 8
            enc = (level << 26) | (rng.u() & ((1u << (2u * level)) - 1u));
        }
        s.misc = U4{ rng.u() % NT, enc, 0u, 0u };
        cs[i] = s;
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<U4> outInit(N * 6u, U4{ 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu });
    PxgBufferHandle bIn = createBufferFrom(dev, cs), bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bInd = createBufferFrom(dev, inds), bVert = createBufferFrom(dev, verts);
    runKernel("ttcVertsProbe", N, dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut),
              dev->getBufferDeviceAddress(bInd), dev->getBufferDeviceAddress(bVert));
    const std::vector<U4> got = downloadBuffer<U4>(dev, bOut, N * 6u);

    size_t mm0 = 0, mm1 = 0, discFold = 0, discNoS2V = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const VertsCase& s = cs[i];
        const PxTransform aToB(v3(s.aToBp), q4(s.aToBq));
        PxVec3 a0, a1, a2, b0, b1, b2;
        refPass0Vertices(inds, verts, v3(s.scale0), q4(s.rot0), aToB, s.misc.x, s.misc.y,
                         v3(s.scale1), q4(s.rot1), a0, a1, a2);
        refPass1Vertices(inds, verts, v3(s.scale1), q4(s.rot1), aToB, s.misc.x,
                         v3(s.scale0), q4(s.rot0), b0, b1, b2);
        const U4* o = &got[i * 6u];
        if (!sameVec(o[0], a0) || !sameVec(o[1], a1) || !sameVec(o[2], a2))
        {
            if (mm0 < 4) std::printf("  verts case %u pass 0: x got 0x%08x want 0x%08x\n", i, o[0].x, bitsOf(a0.x));
            ++mm0;
        }
        if (!sameVec(o[3], b0) || !sameVec(o[4], b1) || !sameVec(o[5], b2))
        {
            if (mm1 < 4) std::printf("  verts case %u pass 1: x got 0x%08x want 0x%08x\n", i, o[3].x, bitsOf(b0.x));
            ++mm1;
        }
        // T9: pass 1 with aToB.getInverse().transform in place of transformInv.
        const U4 t = inds[s.misc.x];
        const PxVec3 sh = refVertex2Shape(v3(verts[t.x]), v3(s.scale1), q4(s.rot1));
        if (differs(refShape2Vertex(aToB.getInverse().transform(sh), v3(s.scale0), q4(s.rot0)), b0)) ++discFold;
        // pass 0 without the final shape2Vertex (the triangle left in shape space)
        PxVec3 c0, c1, c2;
        refGetTriangleVertices(inds, verts, v3(s.scale0), q4(s.rot0), aToB, s.misc.x, s.misc.y, c0, c1, c2);
        if (differs(c0, a0)) ++discNoS2V;
    }
    std::printf("  vertices: %u cases, pass-1 inverse-fold discriminating %zu, missing-shape2Vertex %zu\n",
                N, discFold, discNoS2V);
    PXG_CHECK_MSG(mm0 == 0, "%zu pass-0 vertex mismatches", mm0);
    PXG_CHECK_MSG(mm1 == 0, "%zu pass-1 vertex mismatches", mm1);
    PXG_CHECK_MSG(discFold > N / 4, "pass-1 fold visible in only %zu cases", discFold);
    PXG_CHECK_MSG(discNoS2V > N / 2, "missing shape2Vertex visible in only %zu cases", discNoS2V);
    for (PxgBufferHandle h : { bIn, bOut, bInd, bVert }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// The post-processing of both passes.
// ---------------------------------------------------------------------------
PXG_TEST(npTriTriBody_post)
{
    Rng rng(832002u);
    const uint32_t N = 6144u;
    std::vector<PostCase> cs;
    cs.reserve(N);

    enum : uint32_t { PASS1 = 1u, HAS_SDF = 2u };

    for (uint32_t i = 0; cs.size() < N; ++i)
    {
        PostCase s{};
        s.v0 = rng.point(2.f); s.v1 = rng.point(2.f); s.v2 = rng.point(2.f);
        const F4 d = rng.point(1.f);
        s.dirSep = F4{ d.x, d.y, d.z, rng.f(-0.5f, 1.5f) };
        s.point = rng.point(2.f);
        s.boxLo = F4{ rng.f(-3.f, -0.5f), rng.f(-3.f, -0.5f), rng.f(-3.f, -0.5f), 0.f };
        s.boxHi = F4{ rng.f(0.5f, 3.f), rng.f(0.5f, 3.f), rng.f(0.5f, 3.f), 0.f };
        s.scale = rng.scale(); s.rot = rng.unitQuat();
        s.aToBp = rng.point(3.f); s.aToBq = rng.unitQuat();
        s.dists = F4{ rng.f(0.2f, 2.f), rng.f(0.05f, 1.f), 256.f, 0.f };   // cullScale, contactDistance, 16*16
        // Pass and SDF state vary by BLOCK of 40, independently of the case family
        // (i % 40), so every family is run in both passes and both SDF states.
        uint32_t flags = ((i / 40u) & 1u) ? PASS1 : 0u;
        if ((i / 80u) % 3u == 0u) flags |= HAS_SDF;

        // Case families (on top of the random bulk).
        switch (i % 40u)
        {
        case 1: s.dirSep.x = s.dirSep.y = s.dirSep.z = 0.f; break;                      // m == 0
        case 2: s.v2 = s.v1; break;                                                      // n == 0: dir2 fails the 1e-8 test
        case 3: { const float k = rng.f(0.004f, 0.03f);                                  // |dir2|^2 around 1e-8
                  s.v1 = F4{ s.v0.x + k, s.v0.y, s.v0.z, 0 }; s.v2 = F4{ s.v0.x, s.v0.y + k, s.v0.z, 0 };
                  s.scale = ONE3; s.rot = IDQ; break; }
        case 4: { // T4: the sign at exactly zero -- v0 AT the box centre
                  s.v0 = F4{ 0.5f * (s.boxHi.x + s.boxLo.x), 0.5f * (s.boxHi.y + s.boxLo.y), 0.5f * (s.boxHi.z + s.boxLo.z), 0 };
                  break; }
        case 5: { // the centre on the triangle's plane but not at v0: dot == +-0 by symmetry
                  s.boxLo = F4{ -1.f, -1.f, -1.f, 0 }; s.boxHi = F4{ 1.f, 1.f, 1.f, 0 };
                  s.v0 = F4{ 0.5f, 0.f, 0.f, 0 }; s.v1 = F4{ -0.25f, 0.5f, 0.f, 0 }; s.v2 = F4{ 0.25f, -0.75f, 0.f, 0 };
                  break; }
        case 6: s.dirSep.w = s.dists.x; break;                                            // sep == cullScale: strict <
        case 7: { s.scale = ONE3; s.rot = IDQ;                                             // |dir| == 1 after shape2Vertex
                  s.dirSep.x = 0.f; s.dirSep.y = 1.f; s.dirSep.z = 0.f;
                  s.dirSep.w = s.dists.y; s.v1 = s.v0; break; }                            // sep*m == contactDistance: strict <
        case 8: { s.dists.x = PINF; s.dists.y = PINF; s.dirSep.w = PX_MAX_F32_;          // T1: the unwritten dir,
                  s.dirSep.x = s.dirSep.y = s.dirSep.z = 0.f; break; }                     // made a CANDIDATE by an
                                                                                           // infinite contactDistance so
                                                                                           // the zero dir reaches the normal
        case 9: { // T6: triRadius2 * 256 == sdfRadius2 exactly (2 * 256 == 512 == 16^2 + 16^2)
                  s.v0 = F4{ 0, 0, 0, 0 }; s.v1 = F4{ 1, 0, 0, 0 }; s.v2 = F4{ 0, 1, 0, 0 };
                  s.boxLo = F4{ -8.f, -8.f, 0.f, 0 }; s.boxHi = F4{ 8.f, 8.f, 0.f, 0 }; break; }
        case 10: { s.v0 = F4{ 0, 0, 0, 0 }; s.v1 = F4{ 1, 0, 0, 0 }; s.v2 = F4{ 0, 1, 0, 0 };   // just over:
                   s.boxLo = F4{ -8.f, -7.9375f, 0.f, 0 }; s.boxHi = F4{ 8.f, 8.f, 0.f, 0 }; break; }   // 512 > 510.00390625
        case 11: s.dirSep.w = rng.f(-3.f, -1.f); break;                                   // deep: always a candidate
        case 12: s.dirSep.w = rng.f(2.5f, 4.f); break;                                    // beyond any cullScale
        case 13: s.v1.y = QNAN; break;                                                     // NaN everywhere downstream
        case 19: { // the 1e-8f threshold ON the boundary: n = (0, -q, p) with |n|^2 == 1e-8f exactly
                   // (host search: fl(fl(q*q) + fl(p*p)) == 1e-8f), identity scale so dir2 = +-n.
                   // `>` keeps the SDF dir, `>=` would take the triangle normal.
                   const float P = 6.00099993e-05f, Q = 7.99924965e-05f;
                   s.v0 = F4{ 0, 0, 0, 0 }; s.v1 = F4{ 1.f, 0, 0, 0 }; s.v2 = F4{ 0, P, Q, 0 };
                   s.scale = ONE3; s.rot = IDQ; s.dirSep.w = rng.f(-3.f, -1.f); break; }
        case 20: { // T4 at -0: box centre exactly 0, v0 = 0, n = (-1, -2, -6) all negative, so
                   // dot = (-0) + (-0) + (-0) = -0 and PxSign(-0) = +1 -- a sign-bit test gives -1.
                   s.boxLo = F4{ -1.5f, -2.5f, -3.5f, 0 }; s.boxHi = F4{ 1.5f, 2.5f, 3.5f, 0 };
                   s.v0 = F4{ 0, 0, 0, 0 }; s.v1 = F4{ -2.f, -2.f, 1.f, 0 }; s.v2 = F4{ -2.f, 1.f, 0.f, 0 };
                   s.dirSep.w = rng.f(-3.f, -1.f); break; }
        case 21: s.dirSep.w = QNAN; break;                                                // NaN sep: every comparison false
        case 22: s.dirSep.y = QNAN; s.dirSep.w = rng.f(-3.f, -1.f); break;                // NaN dir: m is NaN
        case 23: { // NaN made BEHIND the cull test: sep = -inf passes `sep < cullScale`, a dir whose
                   // |dir|^2 overflows gives m = 1/sqrt(inf) = 0, and -inf * 0 = NaN reaches the
                   // candidate test -- the only way a NaN sep meets it.
                   s.dirSep = F4{ 3e19f, -2e19f, 1e19f, -PINF }; break; }
        case 14: { // T4's NaN arm with a FINITE n: a box at +inf makes sdfBoxCenter - v0 = +inf in x,
                   // and a triangle in a z = const plane makes n.x = +0, so dot = inf * 0 + ... = NaN
                   // while dir2 stays finite -- the only way PxSign(NaN) = -1 reaches an output.
                   s.boxHi.x = PINF;
                   s.v0 = F4{ 0.3f, 0.2f, 0.1f, 0 }; s.v1 = F4{ 1.3f, 0.2f, 0.1f, 0 }; s.v2 = F4{ 0.3f, 1.2f, 0.1f, 0 };
                   break; }
        case 15: case 16: case 17: case 18:                                               // a box far larger than
                   s.boxLo = F4{ -40.f, -35.f, -45.f, 0 }; s.boxHi = F4{ 38.f, 42.f, 36.f, 0 }; break;   // the triangle: no refinement
        default: break;
        }
        s.misc = U4{ 0u, 0u, 0u, flags };
        cs.push_back(s);
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<U4> outInit(N * 3u, U4{ 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu });
    PxgBufferHandle bIn = createBufferFrom(dev, cs), bOut = createBufferFrom(dev, outInit);
    runKernel("ttcPostProbe", N, dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut));
    const std::vector<U4> got = downloadBuffer<U4>(dev, bOut, N * 3u);

    size_t mmPos = 0, mmNormal = 0, mmSep = 0, mmFlags = 0;
    size_t nCand[2] = { 0, 0 }, nRefine = 0, nNotRefine = 0, nDir2 = 0, nM0 = 0, nT1 = 0;
    size_t discReuseM = 0, discSignZero = 0, discSignNegZero = 0, discSignNaN = 0, nOnThreshold = 0, nNaNSep = 0, nNaNBehindCull = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const PostCase& s = cs[i];
        const bool pass1 = (s.misc.w & PASS1) != 0u, hasSdf = (s.misc.w & HAS_SDF) != 0u;
        RefContact c;
        c.pos = v3(s.point);
        bool refine = false, usedDir2 = false;
        const PxTransform aToB(v3(s.aToBp), q4(s.aToBq));
        if (!pass1)
            refPass0Post(v3(s.v0), v3(s.v1), v3(s.v2), s.dirSep.w, v3(s.dirSep), v3(s.boxLo), v3(s.boxHi),
                         v3(s.scale), q4(s.rot), hasSdf, s.dists.x, s.dists.y, s.dists.z, c, refine, usedDir2);
        else
            refPass1Post(s.dirSep.w, v3(s.dirSep), v3(s.scale), q4(s.rot), aToB, s.dists.x, s.dists.y, c);

        const U4* o = &got[i * 3u];
        if (!sameVec(o[0], c.pos) || o[0].w != 0u) { if (mmPos < 4) std::printf("  post case %u: pos.x 0x%08x want 0x%08x\n", i, o[0].x, bitsOf(c.pos.x)); ++mmPos; }
        if (!sameVec(o[1], c.normal)) { if (mmNormal < 4) std::printf("  post case %u (pass %u): normal.x 0x%08x want 0x%08x\n", i, pass1 ? 1u : 0u, o[1].x, bitsOf(c.normal.x)); ++mmNormal; }
        if (!sameBits(o[1].w, c.separation)) ++mmSep;
        if (o[2].x != (c.candidateContact ? 1u : 0u) || o[2].y != (refine ? 1u : 0u) || o[2].z != 0u || o[2].w != 0u)
        {
            if (mmFlags < 4) std::printf("  post case %u: cand %u refine %u want %u %u\n", i, o[2].x, o[2].y, c.candidateContact ? 1u : 0u, refine ? 1u : 0u);
            ++mmFlags;
        }

        // Coverage.
        if (c.candidateContact) ++nCand[pass1 ? 1 : 0];
        if (!pass1 && !hasSdf) (refine ? nRefine : nNotRefine)++;
        if (usedDir2) ++nDir2;
        if (s.dirSep.x == 0.f && s.dirSep.y == 0.f && s.dirSep.z == 0.f && s.dirSep.w < s.dists.x) ++nM0;
        if (s.dists.x == PINF && c.candidateContact) ++nT1;   // T1 counted only where it reaches an output

        // Discriminators.
        if (!pass1 && usedDir2 && c.candidateContact)
        {
            // T2: recompute m from the new dir instead of reusing it.
            RefContact alt; alt.pos = v3(s.point);
            PxVec3 dir = refShape2Vertex(v3(s.dirSep), v3(s.scale), q4(s.rot));
            const PxReal m0 = dir.magnitudeSquared();
            PxVec3 n = (v3(s.v1) - v3(s.v0)).cross(v3(s.v2) - v3(s.v0));
            PxVec3 ctr = 0.5f * (v3(s.boxHi) + v3(s.boxLo));
            PxVec3 dir2 = -physx::PxSign((ctr - v3(s.v0)).dot(n)) * refShape2Vertex(n, v3(s.scale), q4(s.rot));
            dir2.normalize();
            dir = dir2 * physx::PxSqrt(m0);
            const PxReal m1 = dir.magnitudeSquared();
            if (m1 > 0.0f && bitsOf(s.dirSep.w * (1.0f / physx::PxSqrt(m1))) != bitsOf(c.separation)) ++discReuseM;
        }
        if (!pass1 && !hasSdf && (i % 40u == 4u || i % 40u == 5u) && s.dirSep.w < s.dists.x) ++discSignZero;
        if (!pass1 && !hasSdf && i % 40u == 14u && usedDir2 && c.candidateContact) ++discSignNaN;
        if (!pass1 && !hasSdf && i % 40u == 20u && usedDir2 && c.candidateContact) ++discSignNegZero;
        if (!pass1 && !hasSdf && i % 40u == 19u && c.candidateContact)
        {
            const PxVec3 n = (v3(s.v1) - v3(s.v0)).cross(v3(s.v2) - v3(s.v0));
            if (bitsOf(n.magnitudeSquared()) == bitsOf(1e-8f) && !usedDir2) ++nOnThreshold;
        }
        if (i % 40u == 21u) ++nNaNSep;
        if (i % 40u == 23u) ++nNaNBehindCull;
    }

    std::printf("  post: %u cases, candidates %zu/%zu (pass 0/1), refine %zu/%zu, dir2 used %zu, m==0 %zu, T1 %zu; "
                "m-reuse discriminating %zu, sign-at-zero cases %zu (-0: %zu), sign-of-NaN cases %zu, "
                "on the 1e-8 boundary %zu, NaN sep %zu\n",
                N, nCand[0], nCand[1], nRefine, nNotRefine, nDir2, nM0, nT1, discReuseM, discSignZero,
                discSignNegZero, discSignNaN, nOnThreshold, nNaNSep);
    PXG_CHECK_MSG(mmPos == 0, "%zu pos mismatches", mmPos);
    PXG_CHECK_MSG(mmNormal == 0, "%zu normal mismatches", mmNormal);
    PXG_CHECK_MSG(mmSep == 0, "%zu separation mismatches", mmSep);
    PXG_CHECK_MSG(mmFlags == 0, "%zu candidate/refinement mismatches", mmFlags);
    PXG_CHECK_MSG(nCand[0] > 500 && nCand[1] > 500, "too few candidates (%zu, %zu)", nCand[0], nCand[1]);
    PXG_CHECK_MSG(nRefine > 100 && nNotRefine > 100, "refinement one-sided (%zu, %zu)", nRefine, nNotRefine);
    PXG_CHECK_MSG(nDir2 > 300, "the floor branch replaced dir only %zu times", nDir2);
    PXG_CHECK_MSG(discReuseM > 50, "reusing m is visible in only %zu cases", discReuseM);
    PXG_CHECK_MSG(nM0 > 20 && nT1 > 20, "m==0 (%zu) or T1 (%zu) under-covered", nM0, nT1);
    PXG_CHECK_MSG(discSignZero > 20, "sign-at-zero reached only %zu times", discSignZero);
    PXG_CHECK_MSG(discSignNaN > 5, "sign-of-NaN reached only %zu times", discSignNaN);
    PXG_CHECK_MSG(discSignNegZero > 5, "sign-at--0 reached only %zu times", discSignNegZero);
    PXG_CHECK_MSG(nOnThreshold > 5, "the 1e-8 boundary reached only %zu times", nOnThreshold);
    PXG_CHECK_MSG(nNaNSep > 50, "NaN sep only %zu times", nNaNSep);
    PXG_CHECK_MSG(nNaNBehindCull > 50, "NaN behind the cull test only %zu times", nNaNBehindCull);
    PXG_CHECK_MSG(nNotRefine > 300, "only %zu cases stay unrefined", nNotRefine);
    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}
