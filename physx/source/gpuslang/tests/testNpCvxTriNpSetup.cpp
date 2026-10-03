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

// Verification of npCvxTriNpSetup.slang: the convexTrimeshNarrowphaseCore
// setup stage (convexMesh.cu L74-215).  Exact CPU transcription: the
// trimeshToConvexTransform composition, the contact-distance sum, the convex
// blob head (CoM through vertex2Shape with the convex scale, the packed
// counts word), the readTriangleMesh byte walk (BV32 skip included), the
// pair's adjacency row and faceRemap entry, the three triangle vertices with
// the negative-determinant writeIndex permutation (tid ^ 3 for threads 1/2),
// the local normal over the permuted winding, and the pair offsets.
// ALL outputs compare bitwise (identical op sequences, proven bit-identical).

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
struct SetIn
{
    F4 cq, cp, tq, tp;
    F4 cScaleS, cScaleR, tScaleS, tScaleR;
    F4 dists;
    U4 pairInfo;
    F4 cvxBlob[2];
    U4 blob[192];
};
struct SetOut
{
    F4 t2cq, t2cp, com;
    U4 adjRemap;
    F4 locVerts[3];
    F4 locNormal;
    U4 offsets;
};

const uint32_t BV32B = 1168;

static uint32_t& word(U4* a, uint32_t idx) { return (&a[idx / 4].x)[idx & 3]; }
static uint32_t cword(const U4* a, uint32_t idx) { return (&a[idx / 4].x)[idx & 3]; }
static float cfloat(const U4* a, uint32_t idx) { float f; const uint32_t u = cword(a, idx); std::memcpy(&f, &u, 4); return f; }
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

// P9.94d twin: the OLD spelling of the translation, q.rotateInv(src.p - p).  The CUDA
// (convexMesh.cu:139) is PxTransform::transformInv(PxTransform): qinv = q.getConjugate();
// qinv.rotate(src.p - p) (PxTransform.h:162-169).  The two differ only in the sign of an
// exact zero, which npCvxTriNpSetup_p94dZeroSign must show.
static bool& p94dRotInvTwin() { static bool b = false; return b; }

static void refSetup(const SetIn& s, SetOut& o)
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

    const uint32_t numVerts = cword(s.blob, 0), numTris = cword(s.blob, 1), nbPacked = cword(s.blob, 3);
    uint32_t off = 16 + BV32B * nbPacked;
    const uint32_t vertsOff = off; off += 16 * numVerts;
    const uint32_t triOff = off; off += 16 * numTris;
    const uint32_t adjOff = off; off += 16 * numTris;
    const uint32_t remapOff = off;

    const uint32_t triangleIdx = s.pairInfo.x;
    o.adjRemap = U4{ cword(s.blob, (adjOff + 16 * triangleIdx) / 4),
                     cword(s.blob, (adjOff + 16 * triangleIdx + 4) / 4),
                     cword(s.blob, (adjOff + 16 * triangleIdx + 8) / 4),
                     cword(s.blob, (remapOff + 4 * triangleIdx) / 4) };

    const bool flip = (s.tScaleS.x * s.tScaleS.y * s.tScaleS.z) < 0.f;
    V3 lv[3];
    for (uint32_t tid = 0; tid < 3; ++tid)
    {
        uint32_t wi = tid;
        if (tid > 0 && flip) wi = tid ^ 3;
        const uint32_t vIdx = cword(s.blob, (triOff + 16 * triangleIdx + 4 * tid) / 4);
        const V3 tv{ cfloat(s.blob, (vertsOff + 16 * vIdx) / 4),
                     cfloat(s.blob, (vertsOff + 16 * vIdx + 4) / 4),
                     cfloat(s.blob, (vertsOff + 16 * vIdx + 8) / 4) };
        const V3 vs = v2s(tv, V3{ s.tScaleS.x, s.tScaleS.y, s.tScaleS.z }, s.tScaleR);
        const V3 rp = qRot(t2cq, vs);
        lv[wi] = V3{ rp.x + t2cp.x, rp.y + t2cp.y, rp.z + t2cp.z };
    }
    for (int j = 0; j < 3; ++j) o.locVerts[j] = F4{ lv[j].x, lv[j].y, lv[j].z, 0 };

    const V3 n = nrmz(crossV(V3{ lv[1].x - lv[0].x, lv[1].y - lv[0].y, lv[1].z - lv[0].z },
                             V3{ lv[2].x - lv[0].x, lv[2].y - lv[0].y, lv[2].z - lv[0].z }));
    o.locNormal = F4{ n.x, n.y, n.z, flip ? 1.f : 0.f };

    o.offsets = U4{ s.pairInfo.y + s.pairInfo.w, s.pairInfo.z + s.pairInfo.w, 0, 0 };
}

// bitwise compares throughout: both sides execute identical op sequences and
// llvmpipe/gcc have proven bit-identical on this framework (tranche 159's GJK)
static bool eqF(float a, float b) { return f2u(a) == f2u(b); }
static bool eq4(const F4& a, const F4& b) { return eqF(a.x, b.x) && eqF(a.y, b.y) && eqF(a.z, b.z) && eqF(a.w, b.w); }

PXG_TEST(npCvxTriNpSetup_setup)
{
    std::mt19937 rng(160001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<SetIn> in(N);
    std::vector<SetOut> ref(N);
    size_t cFlip = 0, cBv32 = 0, cBv32Deep = 0, cBigV = 0, cBigT = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        SetIn s{};
        s.cq = randQuat(); s.cp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        s.tq = randQuat(); s.tp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        s.cScaleS = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 }; s.cScaleR = randQuat();
        const float sx = (rng() % 3 == 0) ? -rf(0.5f, 2) : rf(0.5f, 2);
        s.tScaleS = F4{ sx, (rng() % 4 == 0) ? -rf(0.5f, 2) : rf(0.5f, 2), (rng() % 5 == 0) ? -rf(0.5f, 2) : rf(0.5f, 2), 0 }; // all three axes can go negative: the determinant must use every component
        s.tScaleR = randQuat();
        s.dists = F4{ rf(0.01f, 0.5f), rf(0.01f, 0.5f), 0, 0 };
        s.cvxBlob[0] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        s.cvxBlob[1] = F4{ u2f(uint32_t(rng())), 0, 0, 0 };

        // sizes chosen to defeat cap/saturate defects (review 160): nbPacked
        // reaches 2 (kills min(nbPacked,1)), numVerts reaches 16+ (kills &15),
        // numTris reaches 9+ (kills min(...,8)); large-mesh combos alternate
        // with the BV32 skip so everything still fits blob[192] (3072 bytes)
        const uint32_t nbPacked = (i % 4 == 0) ? ((i % 8 == 0) ? 2u : 1u) : 0u;
        const uint32_t numVerts = nbPacked == 2 ? 6 + rng() % 3 : 6 + rng() % 15;
        const uint32_t numTris = nbPacked == 2 ? 4 + rng() % 3 : 4 + rng() % 17;
        if (nbPacked) ++cBv32;
        if (nbPacked == 2) ++cBv32Deep;
        if (numVerts > 15) ++cBigV;
        if (numTris > 8) ++cBigT;
        word(s.blob, 0) = numVerts; word(s.blob, 1) = numTris;
        word(s.blob, 2) = uint32_t(rng()); word(s.blob, 3) = nbPacked;
        uint32_t off = 16 + BV32B * nbPacked;
        const uint32_t vertsOff = off;
        for (uint32_t v = 0; v < numVerts; ++v)
            for (int c = 0; c < 3; ++c)
                word(s.blob, (vertsOff + 16 * v + 4 * c) / 4) = f2u(rf(-2, 2));
        off += 16 * numVerts;
        const uint32_t triOff = off;
        for (uint32_t tj = 0; tj < numTris; ++tj)
            for (int c = 0; c < 3; ++c)
                word(s.blob, (triOff + 16 * tj + 4 * c) / 4) = rng() % numVerts;
        off += 16 * numTris;
        const uint32_t adjOff = off;
        for (uint32_t tj = 0; tj < numTris; ++tj)
            for (int c = 0; c < 3; ++c)
                word(s.blob, (adjOff + 16 * tj + 4 * c) / 4) = uint32_t(rng());
        off += 16 * numTris;
        for (uint32_t tj = 0; tj < numTris; ++tj)
            word(s.blob, (off + 4 * tj) / 4) = uint32_t(rng());

        s.pairInfo = U4{ uint32_t(rng() % numTris), uint32_t(rng() % 512), uint32_t(4 * (rng() % 128)), uint32_t(rng() % 64) };
        in[i] = s;
        refSetup(s, ref[i]);
        if (ref[i].locNormal.w != 0.f) ++cFlip;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SetOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct SetParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxTriNpSetup.spv", "cvxTriNpSetup",
             { (uint32_t)sizeof(SetIn), (uint32_t)sizeof(SetOut) }, sizeof(SetParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SetOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const SetOut& r = ref[i]; const SetOut& q = g[i];
        bool ok = eq4(r.t2cq, q.t2cq) && eq4(r.t2cp, q.t2cp) &&
                  eqF(r.com.x, q.com.x) && eqF(r.com.y, q.com.y) && eqF(r.com.z, q.com.z) &&
                  f2u(r.com.w) == f2u(q.com.w) &&
                  r.adjRemap.x == q.adjRemap.x && r.adjRemap.y == q.adjRemap.y &&
                  r.adjRemap.z == q.adjRemap.z && r.adjRemap.w == q.adjRemap.w &&
                  r.offsets.x == q.offsets.x && r.offsets.y == q.offsets.y &&
                  r.locNormal.w == q.locNormal.w &&
                  eqF(r.locNormal.x, q.locNormal.x) && eqF(r.locNormal.y, q.locNormal.y) && eqF(r.locNormal.z, q.locNormal.z);
        for (int j = 0; j < 3 && ok; ++j) ok = eq4(r.locVerts[j], q.locVerts[j]);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cvxTriNpSetup: %zu mismatches", mm);
    PXG_CHECK_MSG(cFlip > N / 5 && cBv32 > N / 5 && cBv32Deep > N / 16 && cBigV > N / 8 && cBigT > N / 8,
                  "cvxTriNpSetup: coverage thin (flip %zu bv32 %zu deep %zu bigV %zu bigT %zu)",
                  cFlip, cBv32, cBv32Deep, cBigV, cBigT);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// P9.94d: t2cp = convexT.transformInv(trimeshT).p (convexMesh.cu:139) is
// conj(cq).rotate(tp - cp), not cq.rotateInv(tp - cp).  cq = tq = identity,
// cp = (1,0,0), tp = (1,-0,-0): tp - cp = (+0,-0,-0), and
//   rotate((-0,-0,-0,1), .).y = vy*w2 + (qz*vx - qx*vz)*w + qy*d2 = -0 + -0 + -0 = -0
//   rotateInv((0,0,0,1), .).y = vy*w2 - (qz*vx - qx*vz)*w + qy*d2 = -0 - +0 + +0 = +0
// so the stored t2cp.y is 0x80000000 under the CUDA and 0 under the old spelling.
// A 3-vertex / 1-triangle mesh, no BV32, identity scales; every output word compares.
PXG_TEST(npCvxTriNpSetup_p94dZeroSign)
{
    const float nz = -0.0f;
    SetIn s{};
    s.cq = F4{ 0, 0, 0, 1 }; s.cp = F4{ 1, 0, 0, 0 };
    s.tq = F4{ 0, 0, 0, 1 }; s.tp = F4{ 1, nz, nz, 0 };
    s.cScaleS = F4{ 1, 1, 1, 0 }; s.cScaleR = F4{ 0, 0, 0, 1 };
    s.tScaleS = F4{ 1, 1, 1, 0 }; s.tScaleR = F4{ 0, 0, 0, 1 };
    s.dists = F4{ 0.125f, 0.25f, 0, 0 };
    s.cvxBlob[0] = F4{ 0.25f, 0.5f, 0.75f, 0 };
    s.cvxBlob[1] = F4{ u2f(0x00030004u), 0, 0, 0 };
    word(s.blob, 0) = 3u; word(s.blob, 1) = 1u; word(s.blob, 2) = 0u; word(s.blob, 3) = 0u;
    const float verts[3][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 1 } };
    const uint32_t vertsOff = 16u, triOff = vertsOff + 16u * 3u, adjOff = triOff + 16u, remapOff = adjOff + 16u;
    for (uint32_t v = 0; v < 3; ++v)
        for (uint32_t c = 0; c < 3; ++c)
            word(s.blob, (vertsOff + 16u * v + 4u * c) / 4u) = f2u(verts[v][c]);
    for (uint32_t c = 0; c < 3; ++c)
    {
        word(s.blob, (triOff + 4u * c) / 4u) = c;
        word(s.blob, (adjOff + 4u * c) / 4u) = 0xA0u + c;
    }
    word(s.blob, remapOff / 4u) = 0x77u;
    s.pairInfo = U4{ 0u, 5u, 8u, 1u };

    SetOut lit{}, twin{};
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
    PXG_CHECK_MSG(std::memcmp(&lit, &twin, sizeof(SetOut)) != 0,
                  "p94d: the rotateInv twin stores the same words -- not discriminating");

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<SetIn> in(1, s);
    std::vector<SetOut> outInit(1);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct Prm { uint32_t numTests; } prm = { 1u };
    Kernel k("npCvxTriNpSetup.spv", "cvxTriNpSetup",
             { (uint32_t)sizeof(SetIn), (uint32_t)sizeof(SetOut) }, sizeof(Prm));
    k.dispatch(1u, { bIn, bOut }, prm);
    dev->flush();
    const SetOut g = downloadBuffer<SetOut>(dev, bOut, 1)[0];
    const uint32_t* gw = reinterpret_cast<const uint32_t*>(&g);
    const uint32_t* lw = reinterpret_cast<const uint32_t*>(&lit);
    const uint32_t* tw = reinterpret_cast<const uint32_t*>(&twin);
    for (uint32_t w = 0; w < sizeof(SetOut) / 4u; ++w)
        PXG_CHECK_MSG(gw[w] == lw[w], "p94d: output word %u GPU 0x%08x != literal 0x%08x (twin 0x%08x)",
                      w, gw[w], lw[w], tw[w]);
    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
