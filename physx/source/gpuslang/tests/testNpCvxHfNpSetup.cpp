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

// Verification of npCvxHfNpSetup.slang: the convexHeightfieldNarrowphaseCore
// setup stage (convexHeightfield.cu L120-269).  Exact CPU transcription: the
// trimeshToConvexTransform composition, getTriangle (wrongHanded from the
// column/row scale signs, the [0, 1+wh, 2-wh] vertex and
// [wh?2:0, 1, wh?0:2] adjacency out-argument permutations, sign-extended
// 16-bit heights, the 0x800000 tessellation bit, hf2shapep component
// multiply), the denormalizedNormal cross over the permuted verts, the
// rotate-only-then-normalize local normal, the full-transform local verts,
// the contact-distance sum, the convex CoM/counts head, and the pair offsets.
// ALL float outputs compare bitwise (identical op sequences; the framework's
// gcc/llvmpipe pairing is proven bit-identical, tranche 159).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };
struct HfsIn
{
    F4 cq, cp, tq, tp;
    F4 cScaleS, cScaleR;
    F4 hfScale;
    F4 dists;
    U4 pairInfo;
    F4 cvxBlob[2];
    U4 hf[17];
};
struct HfsOut
{
    F4 t2cq, t2cp, com;
    U4 adj;
    F4 locVerts[3];
    F4 locNormal;
    U4 offsets;
};

const uint32_t HF_BOUNDARY = 0xFFFFFFFFu;

static uint32_t& word(U4* a, uint32_t idx) { return (&a[idx / 4].x)[idx & 3]; }
static uint32_t cword(const U4* a, uint32_t idx) { return (&a[idx / 4].x)[idx & 3]; }
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static V3 qRot(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3 qRotInv(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static F4 qMul(const F4& a, const F4& b) // PxQuat::operator*, sequential w
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
               a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
               a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static V3 v2s(V3 v, V3 sc, const F4& rot)
{
    const V3 r = qRot(rot, v);
    return qRotInv(rot, V3{ r.x * sc.x, r.y * sc.y, r.z * sc.z });
}
static V3 crossV(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 nrmz(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0) { const float i = 1.f / std::sqrt(m); return V3{ v.x * i, v.y * i, v.z * i }; }
    return V3{ 0, 0, 0 };
}

// ---- Gu heightfield helpers (tranche-59 verified semantics) -----------------
static bool hfShared(const HfsIn& s, uint32_t vertexIndex)
{
    return (cword(s.hf, 2 + vertexIndex) & 0x800000u) != 0;
}
static float hfHeight(const HfsIn& s, uint32_t vertexIndex)
{
    const int32_t h = int32_t(cword(s.hf, 2 + vertexIndex) << 16) >> 16; // sign-extended low 16 bits
    return float(h);
}
static V3 hfGetVertex(const HfsIn& s, uint32_t vertexIndex, uint32_t columns)
{
    return V3{ float(vertexIndex / columns), hfHeight(s, vertexIndex), float(vertexIndex % columns) };
}
static void hfTriVertexIndices(const HfsIn& s, uint32_t triangleIndex, uint32_t columns,
                               uint32_t& i0, uint32_t& i1, uint32_t& i2)
{
    const uint32_t cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 1) == 0;
    if (hfShared(s, cell))
    {
        if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1; }
        else          { i0 = cell + 1; i1 = cell + columns + 1; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1; i2 = cell + columns; }
        else          { i0 = cell + columns + 1; i1 = cell + columns; i2 = cell + 1; }
    }
}
static void hfTriAdjacencyIndices(const HfsIn& s, uint32_t triangleIndex, uint32_t columns, uint32_t rows,
                                  uint32_t& a0, uint32_t& a1, uint32_t& a2)
{
    const uint32_t cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 1) == 0;
    if (hfShared(s, cell))
    {
        if (firstTri)
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex + 1; a2 = HF_BOUNDARY;
            if (cell % columns != 0)          a0 = triangleIndex - 1;
            if (cell / columns != rows - 2)   a2 = ((cell + columns) * 2) + 1;
        }
        else
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex - 1; a2 = HF_BOUNDARY;
            if (cell % columns < columns - 2) a0 = triangleIndex + 1;
            if (cell >= columns - 1)          a2 = (cell - columns) * 2;
        }
    }
    else
    {
        if (firstTri)
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex + 1; a2 = HF_BOUNDARY;
            if (cell >= columns - 1)          a0 = ((cell - columns) * 2) + 1;
            if (cell % columns != 0)          a2 = triangleIndex - 1;
        }
        else
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex - 1; a2 = HF_BOUNDARY;
            if (cell / columns != rows - 2)   a0 = (cell + columns) * 2;
            if (cell % columns < columns - 2) a2 = triangleIndex + 1;
        }
    }
}

// P9.94d twin: the OLD spelling of the translation, q.rotateInv(src.p - p).  The CUDA
// (convexHeightfield.cu:208) is PxTransform::transformInv(PxTransform): qinv = q.getConjugate();
// qinv.rotate(src.p - p) (PxTransform.h:162-169).  The two differ only in the sign of an
// exact zero, which npCvxHfNpSetup_p94dZeroSign must show.
static bool& p94dRotInvTwin() { static bool b = false; return b; }

static void refSetup(const HfsIn& s, HfsOut& o)
{
    const F4 conjC{ -s.cq.x, -s.cq.y, -s.cq.z, s.cq.w };
    const F4 t2cq = qMul(conjC, s.tq);
    const V3 rel{ s.tp.x - s.cp.x, s.tp.y - s.cp.y, s.tp.z - s.cp.z };
    const V3 t2cp = p94dRotInvTwin() ? qRotInv(s.cq, rel) : qRot(conjC, rel);
    const float contactDist = s.dists.x + s.dists.y;
    o.t2cq = t2cq;
    o.t2cp = F4{ t2cp.x, t2cp.y, t2cp.z, contactDist };

    const V3 com = v2s(V3{ s.cvxBlob[0].x, s.cvxBlob[0].y, s.cvxBlob[0].z },
                       V3{ s.cScaleS.x, s.cScaleS.y, s.cScaleS.z }, s.cScaleR);
    o.com = F4{ com.x, com.y, com.z, s.cvxBlob[1].x };

    const uint32_t rows = s.hf[0].x, columns = s.hf[0].y;
    const float rowScale = s.hfScale.x, heightScale = s.hfScale.y, columnScale = s.hfScale.z;

    // getTriangle (cu L83-116): the handedness vector is dead
    bool wrongHanded = false;
    if (columnScale < 0.f) wrongHanded = !wrongHanded;
    if (rowScale < 0.f) wrongHanded = !wrongHanded;
    const uint32_t wh = wrongHanded ? 1u : 0u;

    const uint32_t triangleIdx = s.pairInfo.x;
    uint32_t tv[3];
    hfTriVertexIndices(s, triangleIdx, columns, tv[0], tv[1 + wh], tv[2 - wh]);
    uint32_t adj[3];
    hfTriAdjacencyIndices(s, triangleIdx, columns, rows, adj[wh ? 2 : 0], adj[1], adj[wh ? 0 : 2]);
    o.adj = U4{ adj[0], adj[1], adj[2], wh };

    V3 verts[3];
    for (uint32_t vi = 0; vi < 3; ++vi)
    {
        const V3 v = hfGetVertex(s, tv[vi], columns);
        verts[vi] = V3{ v.x * rowScale, v.y * heightScale, v.z * columnScale }; // hf2shapep
    }

    const V3 triNormal = crossV(V3{ verts[1].x - verts[0].x, verts[1].y - verts[0].y, verts[1].z - verts[0].z },
                                V3{ verts[2].x - verts[0].x, verts[2].y - verts[0].y, verts[2].z - verts[0].z });
    const V3 n = nrmz(qRot(t2cq, triNormal)); // rotate ONLY, then normalize
    o.locNormal = F4{ n.x, n.y, n.z, 0 };

    for (uint32_t vi = 0; vi < 3; ++vi)
    {
        const V3 rp = qRot(t2cq, verts[vi]);
        o.locVerts[vi] = F4{ rp.x + t2cp.x, rp.y + t2cp.y, rp.z + t2cp.z, 0 };
    }

    o.offsets = U4{ s.pairInfo.y + s.pairInfo.w, s.pairInfo.z + s.pairInfo.w, 0, 0 };
}

// bitwise compares throughout (tranche-160 doctrine)
static bool eqF(float a, float b) { return f2u(a) == f2u(b); }
static bool eq4(const F4& a, const F4& b) { return eqF(a.x, b.x) && eqF(a.y, b.y) && eqF(a.z, b.z) && eqF(a.w, b.w); }

PXG_TEST(npCvxHfNpSetup_setup)
{
    std::mt19937 rng(161001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<HfsIn> in(N);
    std::vector<HfsOut> ref(N);
    size_t cWrong = 0, cTess = 0, cBoundary = 0, cInterior = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        HfsIn s{};
        s.cq = randQuat(); s.cp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        s.tq = randQuat(); s.tp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        s.cScaleS = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 }; s.cScaleR = randQuat();
        // row/height/column scales flip sign independently: wrongHanded is the
        // XOR of the row/column signs and must ignore the height sign
        s.hfScale = F4{ (rng() % 2 ? -1.f : 1.f) * rf(0.25f, 2),
                        (rng() % 3 == 0 ? -1.f : 1.f) * rf(0.25f, 2),
                        (rng() % 2 ? -1.f : 1.f) * rf(0.25f, 2), 0 };
        s.dists = F4{ rf(0.01f, 0.5f), rf(0.01f, 0.5f), 0, 0 };
        s.cvxBlob[0] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        s.cvxBlob[1] = F4{ u2f(uint32_t(rng())), 0, 0, 0 };

        const uint32_t rows = 3 + rng() % 5, columns = 3 + rng() % 5; // <= 7x7 = 49 samples, fits hf[17]
        word(s.hf, 0) = rows; word(s.hf, 1) = columns;
        for (uint32_t v = 0; v < rows * columns; ++v)
            word(s.hf, 2 + v) = uint32_t(rng()); // random tess bit 0x800000, random signed 16-bit height

        // a valid cell: r in [0, rows-2], c in [0, columns-2]; edge cells drive
        // the HF_BOUNDARY branches, interior cells the neighbor arithmetic
        const uint32_t r = rng() % (rows - 1), c = rng() % (columns - 1);
        const uint32_t triangleIdx = (r * columns + c) * 2 + (rng() & 1);
        s.pairInfo = U4{ triangleIdx, uint32_t(rng() % 512), uint32_t(4 * (rng() % 128)), uint32_t(rng() % 64) };

        in[i] = s;
        refSetup(s, ref[i]);
        if (ref[i].adj.w) ++cWrong;
        if (hfShared(s, triangleIdx >> 1)) ++cTess;
        const bool bnd = ref[i].adj.x == HF_BOUNDARY || ref[i].adj.y == HF_BOUNDARY || ref[i].adj.z == HF_BOUNDARY;
        bnd ? ++cBoundary : ++cInterior;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<HfsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct HfsParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxHfNpSetup.spv", "cvxHfNpSetup",
             { (uint32_t)sizeof(HfsIn), (uint32_t)sizeof(HfsOut) }, sizeof(HfsParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<HfsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const HfsOut& r = ref[i]; const HfsOut& q = g[i];
        bool ok = eq4(r.t2cq, q.t2cq) && eq4(r.t2cp, q.t2cp) &&
                  eqF(r.com.x, q.com.x) && eqF(r.com.y, q.com.y) && eqF(r.com.z, q.com.z) &&
                  f2u(r.com.w) == f2u(q.com.w) &&
                  r.adj.x == q.adj.x && r.adj.y == q.adj.y && r.adj.z == q.adj.z && r.adj.w == q.adj.w &&
                  r.offsets.x == q.offsets.x && r.offsets.y == q.offsets.y &&
                  eqF(r.locNormal.x, q.locNormal.x) && eqF(r.locNormal.y, q.locNormal.y) && eqF(r.locNormal.z, q.locNormal.z);
        for (int j = 0; j < 3 && ok; ++j) ok = eq4(r.locVerts[j], q.locVerts[j]);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cvxHfNpSetup: %zu mismatches", mm);
    PXG_CHECK_MSG(cWrong > N / 3 && cTess > N / 3 && cBoundary > N / 8 && cInterior > N / 8,
                  "cvxHfNpSetup: coverage thin (wrong %zu tess %zu bnd %zu int %zu)", cWrong, cTess, cBoundary, cInterior);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// P9.94d: t2cp = convexT.transformInv(heightfieldT).p (convexHeightfield.cu:208) is
// conj(cq).rotate(tp - cp), not cq.rotateInv(tp - cp) -- the same witness as
// npCvxTriNpSetup_p94dZeroSign: cq = tq = identity, cp = (1,0,0), tp = (1,-0,-0) gives
// t2cp.y = -0 (CUDA) against +0 (old spelling).  A 3x3 heightfield, positive heights,
// no tessellation bits, unit scales; every output word compares.
PXG_TEST(npCvxHfNpSetup_p94dZeroSign)
{
    const float nz = -0.0f;
    HfsIn s{};
    s.cq = F4{ 0, 0, 0, 1 }; s.cp = F4{ 1, 0, 0, 0 };
    s.tq = F4{ 0, 0, 0, 1 }; s.tp = F4{ 1, nz, nz, 0 };
    s.cScaleS = F4{ 1, 1, 1, 0 }; s.cScaleR = F4{ 0, 0, 0, 1 };
    s.hfScale = F4{ 1, 1, 1, 0 };
    s.dists = F4{ 0.125f, 0.25f, 0, 0 };
    s.cvxBlob[0] = F4{ 0.25f, 0.5f, 0.75f, 0 };
    s.cvxBlob[1] = F4{ u2f(0x00030004u), 0, 0, 0 };
    word(s.hf, 0) = 3u; word(s.hf, 1) = 3u;
    for (uint32_t v = 0; v < 9u; ++v) word(s.hf, 2 + v) = 2u + v;   // heights 2..10, no tess bit
    s.pairInfo = U4{ 0u, 5u, 8u, 1u };

    HfsOut lit{}, twin{};
    refSetup(s, lit);
    p94dRotInvTwin() = true;
    refSetup(s, twin);
    p94dRotInvTwin() = false;
    std::printf("  [p94d] t2cp literal %08x %08x %08x | rotateInv twin %08x %08x %08x\n",
                f2u(lit.t2cp.x), f2u(lit.t2cp.y), f2u(lit.t2cp.z),
                f2u(twin.t2cp.x), f2u(twin.t2cp.y), f2u(twin.t2cp.z));
    PXG_CHECK_MSG(f2u(lit.t2cp.y) == 0x80000000u && f2u(twin.t2cp.y) == 0x00000000u,
                  "p94d: literal t2cp.y 0x%08x (want -0), rotateInv twin 0x%08x (want +0)",
                  f2u(lit.t2cp.y), f2u(twin.t2cp.y));
    PXG_CHECK_MSG(std::memcmp(&lit, &twin, sizeof(HfsOut)) != 0,
                  "p94d: the rotateInv twin stores the same words -- not discriminating");

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<HfsIn> in(1, s);
    std::vector<HfsOut> outInit(1);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct Prm { uint32_t numTests; } prm = { 1u };
    Kernel k("npCvxHfNpSetup.spv", "cvxHfNpSetup",
             { (uint32_t)sizeof(HfsIn), (uint32_t)sizeof(HfsOut) }, sizeof(Prm));
    k.dispatch(1u, { bIn, bOut }, prm);
    dev->flush();
    const HfsOut g = downloadBuffer<HfsOut>(dev, bOut, 1)[0];
    const uint32_t* gw = reinterpret_cast<const uint32_t*>(&g);
    const uint32_t* lw = reinterpret_cast<const uint32_t*>(&lit);
    const uint32_t* tw = reinterpret_cast<const uint32_t*>(&twin);
    for (uint32_t w = 0; w < sizeof(HfsOut) / 4u; ++w)
        PXG_CHECK_MSG(gw[w] == lw[w], "p94d: output word %u GPU 0x%08x != literal 0x%08x (twin 0x%08x)",
                      w, gw[w], lw[w], tw[w]);
    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
