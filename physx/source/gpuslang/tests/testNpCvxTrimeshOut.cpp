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

// Verification of npCvxTrimeshOut.slang: convexTrimeshFinishContacts +
// writeCompressedContact (convexMeshOutput.cu L50-82, L116-346).  The CPU
// reference transcribes the CUDA exactly, including: the lane/6 manifold
// layout; the lane-0 reservation with CHAINED else-if full-size overflow
// tests and its rollback; insertAveragePoint IGNORED since 5.12
// (convexMeshOutput.cu L165-171 -- 5.6.1 inflated the count on lane 0 only,
// over-reporting nbContacts and making the faceIndex base per-lane divergent;
// the cases still set the flag and the outputs must not depend on it); the
// no-reservation path that stores base + 0xFFFFFFFF wrapped pointers yet
// passes the early-return gate; the flip swap and patch-normal negation; the
// first-contact worldNormal; the sphere/capsule radius adjustments; and the
// 64-byte PxContactPatch image with combineMaterials (tranche 43).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct MaterialData { float dynamicFriction, staticFriction, restitution, damping; uint32_t flags, fricMode, restMode, dampMode; };
struct TmoIn
{
    U4 hdr, nbC, shapeInfo;
    F4 scaleA, scaleB;
    F4 tAq, tAp, tBq, tBp;
    U4 cfg, counters;
    F4 contacts[72];
};
struct TmoOut
{
    U4 res, outputs, counters;
    U4 patches[16];
    F4 cstream[24];
    U4 fstream[16];
};

const uint32_t NULLS = 0xFFFFFFFEu;
const uint32_t T_SPHERE = 0, T_CAPSULE = 2, T_CONVEX = 5, T_TRIMESH = 8;

static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static V3 qRot(const F4& q, const V3& v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3 nrmz(const V3& v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0) { const float i = 1.f / std::sqrt(m); return V3{ v.x * i, v.y * i, v.z * i }; }
    return V3{ 0, 0, 0 };
}

// P9.86: eMIN/eMAX are PxMin/PxMax(float) = intrinsics::selectMin/selectMax (PxsMaterialCombiner.h:43/47/60/64
// -> PxMath.h:82-101 -> PxUnixMathIntrinsics.h:116-125): on a +-0 tie the SECOND operand survives.  The oracle
// spells the selects; std::fmin/std::fmax are not PxMin/PxMax (their zero-tie sign is libm/platform dependent).
// gCombineVariant is a mutant twin replaying a reverted port: 1 = eMIN as GLSL FMin, 2 = eMAX as FMax, with
// llvmpipe's zero order under the SZINP stamp (P9.86 F4: FMin(+-0,-+0) = -0, FMax = +0, either operand order).
static int gCombineVariant = 0;
static float fminLp(float a, float b) { return (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); }
static float fmaxLp(float a, float b) { return (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); }
static float combineScalars(float a, float b, uint32_t mode)
{
    if (mode == 0) return 0.5f * (a + b);
    if (mode == 1) return (gCombineVariant == 1) ? fminLp(a, b) : ((a < b) ? a : b);   // PxMin = selectMin
    if (mode == 2) return a * b;
    if (mode == 3) return (gCombineVariant == 2) ? fmaxLp(a, b) : ((a > b) ? a : b);   // PxMax = selectMax
    return 0.f;
}
static void combineMats(const MaterialData& m0, const MaterialData& m1,
                        uint32_t& matFlags, float& staF, float& dynF, float& rest, float& damp)
{
    const float r0 = m0.restitution, r1 = m1.restitution;
    const bool c0 = r0 < 0.f, c1 = r1 < 0.f;
    const bool oneC = c0 != c1, bothC = c0 && c1;
    const bool a0 = (m0.flags & 16u) != 0, a1 = (m1.flags & 16u) != 0;
    const bool oneA = a0 != a1;
    if (bothC && oneA) rest = a0 ? r0 : r1;
    else
    {
        const uint32_t mode = oneC ? 1u : std::max(m0.restMode, m1.restMode);
        const float flip = (bothC && mode == 2u) ? -1.f : 1.f;
        rest = flip * combineScalars(r0, r1, mode);
    }
    if (bothC && oneA) damp = a0 ? m0.damping : m1.damping;
    else
    {
        const uint32_t mode = oneC ? 3u : std::max(m0.dampMode, m1.dampMode);
        damp = combineScalars(m0.damping, m1.damping, mode);
    }
    const uint32_t cf = m0.flags | m1.flags;
    if ((cf & 1u) == 0)
    {
        const uint32_t mode = std::max(m0.fricMode, m1.fricMode);
        const float d = combineScalars(m0.dynamicFriction, m1.dynamicFriction, mode);
        const float s = combineScalars(m0.staticFriction, m1.staticFriction, mode);
        const float fd = (d > 0.f) ? d : 0.f;   // PxMax(dynFriction, 0.0f) = selectMax (PxsMaterialCombiner.h:161)
        dynF = fd; staF = (s - fd) >= 0.f ? s : fd;
        matFlags = cf;
    }
    else { matFlags = cf | 2u; dynF = 0.f; staF = 0.f; }
}

// The engine test's patch-stream poison, by word index inside a case's slice.  It varies per word so a
// read-modify-write that reads the WRONG word (14/15, or another slot) is visible, not just one that
// zeroes: convexMeshOutput.cu:65-81 stores the header field by field and never touches bytes 54-63
// (PxContactPatch::pad), and the engine never clears its patch stream (P9.87).
static uint32_t patchPoison(uint32_t sliceWord) { return 0xCDCDCDCDu ^ (sliceWord * 0x00010001u); }

// returns the double-written faceIndex word to mask (0xFFFFFFFF = none).  patchFill, when given, is the
// patch stream's content before the dispatch (by word index inside the slice); without it the unwritten
// bytes are expected as 0, which is the record-model shader's zero-initialised record.
static uint32_t refTrimeshOut(const TmoIn& s, const std::vector<MaterialData>& mats, TmoOut& o,
                              uint32_t (*patchFill)(uint32_t) = nullptr)
{
    o.res = U4{ 0, 0, 0, 0 };
    o.outputs = U4{ NULLS, NULLS, NULLS, 0 };
    for (int j = 0; j < 16; ++j) { o.patches[j] = U4{ 0, 0, 0, 0 }; o.fstream[j] = U4{ 0, 0, 0, 0 }; }
    for (int j = 0; j < 24; ++j) o.cstream[j] = F4{ 0, 0, 0, 0 };

    const uint32_t nbManifolds = s.hdr.x;
    const uint32_t allflags = s.hdr.y;
    // s.hdr.z (insertAveragePoint) is PX_UNUSED since 5.12

    uint32_t contactMask = 0;
    for (int L = 0; L < 32; ++L)
    {
        const uint32_t m = L / 6, ti = L % 6;
        const uint32_t nc = (m >= nbManifolds) ? 0 : (&s.nbC.x)[std::min(m, 3u)];
        if (ti < nc) contactMask |= 1u << L;
    }
    const uint32_t pc = popc(contactMask);

    const uint32_t oldSt = (allflags >> 16) & 0xFF;
    uint32_t st = (oldSt & ~3u) | (pc != 0 ? 2u : 1u);
    const uint32_t prevPatches = (allflags & 0xFFFF) >> 8;

    uint32_t patchesB = s.counters.x, contactsB = s.counters.y, forceB = s.counters.z, ovf = 0;

    uint32_t patchOff = 0xFFFFFFFFu, contactOff = 0xFFFFFFFFu, forceOff = 0xFFFFFFFFu;
    bool overflow = false;
    uint32_t total0 = pc, nbMan0 = nbManifolds;
    {
        if (total0)
        {
            patchOff = patchesB;  patchesB += 64 * nbMan0;
            contactOff = contactsB; contactsB += 16 * total0;
            forceOff = forceB;    forceB += 4 * total0 * 2;
            if ((uint64_t(patchOff) + 64ull * nbMan0) > s.cfg.x) { ovf |= 4u; patchOff = 0xFFFFFFFFu; overflow = true; }
            else if ((uint64_t(contactOff) + 16ull * total0) > s.cfg.y) { ovf |= 1u; contactOff = 0xFFFFFFFFu; overflow = true; }
            else if ((uint64_t(forceOff) + 4ull * total0 * 2) > s.cfg.z) { ovf |= 2u; forceOff = 0xFFFFFFFFu; overflow = true; }
            if (overflow)
            {
                nbMan0 = total0 = 0;
                st = (st & ~3u) | 1u;
            }
        }
        const bool prevTouch = (oldSt & 2u) != 0, prevKnown = (oldSt & 3u) != 0;
        const bool nowTouch = nbMan0 != 0;
        o.res.y = ((prevTouch != nowTouch) || !prevKnown) ? 1u : 0u;
        o.res.z = (prevPatches != nbMan0) ? 1u : 0u;
        o.res.x = (((prevPatches << 8) | st) << 16) | (nbMan0 << 8);
        o.res.w = total0;
        if (!overflow) { o.outputs.x = forceOff; o.outputs.y = patchOff; o.outputs.z = contactOff; }
        else { o.outputs.x = o.outputs.y = o.outputs.z = NULLS; }
    }
    o.counters = U4{ patchesB, contactsB, forceB, ovf };

    // The four compared slots start as the stream's prior content; only the fields cu:66-81 store
    // replace it below.
    if (patchFill && !overflow && patchOff != 0xFFFFFFFFu)
        for (uint32_t j = 0; j < 16; ++j)
        {
            const uint32_t w = patchOff / 4 + j * 4;
            o.patches[j] = U4{ patchFill(w), patchFill(w + 1), patchFill(w + 2), patchFill(w + 3) };
        }

    if (overflow || nbManifolds == 0) { o.outputs.w = 1; return 0xFFFFFFFFu; }

    const bool flip = s.shapeInfo.x == T_TRIMESH;
    const uint32_t type0 = flip ? s.shapeInfo.y : s.shapeInfo.x;
    const F4 scale0 = flip ? s.scaleB : s.scaleA;
    const F4 triQ = flip ? s.tAq : s.tBq;
    const V3 triP = flip ? V3{ s.tAp.x, s.tAp.y, s.tAp.z } : V3{ s.tBp.x, s.tBp.y, s.tBp.z };
    const F4 sphQ = flip ? s.tBq : s.tAq;
    const V3 sphP = flip ? V3{ s.tBp.x, s.tBp.y, s.tBp.z } : V3{ s.tAp.x, s.tAp.y, s.tAp.z };
    const uint32_t mi0 = s.shapeInfo.z, mi1 = s.shapeInfo.w;

    uint32_t conflictWord = 0xFFFFFFFFu;
    for (int L = 0; L < 32; ++L)
    {
        const uint32_t m = L / 6, ti = L % 6;
        const bool hasC = (contactMask & (1u << L)) != 0;
        const uint32_t wI = popc(contactMask & ((1u << L) - 1));
        const uint32_t totalLocal = (L == 0) ? total0 : pc;

        V3 nrm{ 0, 0, 0 };
        const uint32_t nc = (m >= nbManifolds) ? 0 : (&s.nbC.x)[std::min(m, 3u)];
        if (ti < nc && m < 4)
        {
            const F4& n = s.contacts[(m * 6 + 0) * 3 + 2];
            nrm = V3{ n.x, n.y, n.z };
        }
        const V3 wn = nrmz(qRot(triQ, nrm));

        if (!hasC) continue;

        const uint32_t cb = (m * 6 + ti) * 3;
        const V3 pA{ s.contacts[cb].x, s.contacts[cb].y, s.contacts[cb].z };
        float pen = s.contacts[cb].w;
        const V3 pB{ s.contacts[cb + 1].x, s.contacts[cb + 1].y, s.contacts[cb + 1].z };
        const uint32_t triIndex = f2u(s.contacts[cb + 1].w);

        if (ti == 0)
        {
            uint32_t mf; float staF, dynF, rest, damp;
            combineMats(mats[mi0 & 0xFFFFu], mats[mi1 & 0xFFFFu], mf, staF, dynF, rest, damp);   // P9.88: cu:61 PxU16
            const V3 pn = flip ? V3{ -wn.x, -wn.y, -wn.z } : wn;
            const uint32_t pb = m * 4;
            o.patches[pb + 0] = U4{ f2u(1.f), f2u(1.f), f2u(1.f), f2u(1.f) };
            o.patches[pb + 1] = U4{ f2u(pn.x), f2u(pn.y), f2u(pn.z), f2u(rest) };
            o.patches[pb + 2] = U4{ f2u(dynF), f2u(staF), f2u(damp),
                                    (wI & 0xFFFF) | ((nc & 0xFF) << 16) | ((mf & 0xFF) << 24) };
            // cu:75 is a PxU16 store: pad[0] (bytes 54-55) keeps the stream's content, and
            // pad[1..4] (words 14-15) are never stored (P9.87).
            const U4 prior = o.patches[pb + 3];
            o.patches[pb + 3] = U4{ 1u | ((mi0 & 0xFFFF) << 16), (prior.y & 0xFFFF0000u) | (mi1 & 0xFFFF), prior.z, prior.w };
        }

        V3 wp;
        if (type0 == T_SPHERE)
        {
            const float radius = scale0.y;
            pen = pen - radius;
            wp = V3{ sphP.x - wn.x * radius, sphP.y - wn.y * radius, sphP.z - wn.z * radius };
        }
        else if (type0 == T_CAPSULE)
        {
            const float radius = scale0.y;
            pen = pen - radius;
            const V3 rp = qRot(sphQ, pA);
            wp = V3{ rp.x + sphP.x - wn.x * radius, rp.y + sphP.y - wn.y * radius, rp.z + sphP.z - wn.z * radius };
        }
        else
        {
            const V3 rp = qRot(triQ, pB);
            wp = V3{ rp.x + triP.x, rp.y + triP.y, rp.z + triP.z };
        }

        if (contactOff != 0xFFFFFFFFu)
            o.cstream[wI] = F4{ wp.x, wp.y, wp.z, pen };

        if (forceOff != 0xFFFFFFFFu)
        {
            const uint32_t word = totalLocal + wI;
            // 5.12: no inflation, so no lane-0 face-index collision either
            (&o.fstream[word / 4].x)[word & 3] = triIndex;
        }
    }
    return conflictWord;
}

static bool eqF(float a, float b) { return almostEqual(a, b, 2e-4f, 2e-5f); }

PXG_TEST(npCvxTrimeshOut_finish)
{
    std::mt19937 rng(158001);
    const uint32_t N = 6000, NMAT = 64;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<MaterialData> mats(NMAT);
    for (auto& m : mats)
    {
        m.dynamicFriction = rf(0, 1.5f); m.staticFriction = rf(0, 2.f);
        m.restitution = rf(-1.f, 1.f); m.damping = rf(0, 2.f);
        m.flags = (rng() % 4 == 0 ? 1u : 0u) | (rng() % 3 == 0 ? 16u : 0u) | (rng() % 5 == 0 ? 256u : 0u);
        m.fricMode = rng() % 4; m.restMode = rng() % 4; m.dampMode = rng() % 4;
    }
    // P9.88: a second bank at [0x10000 + i] with DIFFERENT materials (its own generator, so the
    // case stream is unchanged), so an index k | 0x10000 tells a lookup that narrows to PxU16
    // first (cu:61 static_cast<PxU16>) from one that does not; both stay in bounds.
    {
        std::mt19937 rng2(88001);
        auto rf2 = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng2() % 1000000) / 1000000.f); };
        mats.resize(0x10000u + NMAT);
        for (uint32_t i = 0; i < NMAT; ++i)
        {
            MaterialData& m = mats[0x10000u + i];
            m.dynamicFriction = rf2(0, 1.5f); m.staticFriction = rf2(0, 2.f);
            m.restitution = rf2(-1.f, 1.f); m.damping = rf2(0, 2.f);
            m.flags = (rng2() % 4 == 0 ? 1u : 0u) | (rng2() % 3 == 0 ? 16u : 0u) | (rng2() % 5 == 0 ? 256u : 0u);
            m.fricMode = rng2() % 4; m.restMode = rng2() % 4; m.dampMode = rng2() % 4;
        }
    }

    std::vector<TmoIn> in(N);
    std::vector<TmoOut> ref(N);
    std::vector<uint32_t> maskWord(N, 0xFFFFFFFFu);
    size_t cOvfP = 0, cOvfC = 0, cOvfF = 0, cZeroMan = 0, cNoRes = 0, cAvg = 0, cFlip = 0,
           cSphere = 0, cCapsule = 0, cConflict = 0, cEmptyMan = 0, cExact = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TmoIn s{};
        const uint32_t nbMan = (i % 13 == 0) ? 0u : (1 + rng() % 4);
        const bool insertAvg = (rng() % 5) < 2;
        s.hdr = U4{ nbMan, uint32_t(rng()), insertAvg ? 1u : 0u, 0 };
        uint32_t pcExpect = 0;
        for (int m = 0; m < 4; ++m)
        {
            uint32_t nc = rng() % 7;                 // 0..6, zeros = empty manifolds
            if (i % 7 == 3) nc = (m == 0) ? 0u : nc; // manifold 0 empty: lane 0 has no contact
            (&s.nbC.x)[m] = nc;
            if (uint32_t(m) < nbMan) pcExpect += nc;
        }
        if (i % 11 == 5) { s.nbC = U4{ 0, 0, 0, 0 }; pcExpect = 0; } // manifolds but zero contacts: no-reservation path

        const bool flip = rng() & 1;
        const uint32_t other = (rng() % 3 == 0) ? T_SPHERE : ((rng() % 2) ? T_CAPSULE : T_CONVEX);
        s.shapeInfo = U4{ flip ? T_TRIMESH : other, flip ? other : T_TRIMESH, uint32_t(rng() % NMAT) | ((i % 4 == 1) ? 0x10000u : 0u), uint32_t(rng() % NMAT) | ((i % 5 == 2) ? 0x10000u : 0u) };   // P9.88
        s.scaleA = F4{ rf(0.5f, 2), rf(0.1f, 1.5f), rf(0.5f, 2), 0 };
        s.scaleB = F4{ rf(0.5f, 2), rf(0.1f, 1.5f), rf(0.5f, 2), 0 };
        s.tAq = randQuat(); s.tAp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        s.tBq = randQuat(); s.tBp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };

        for (int c = 0; c < 24; ++c)
        {
            s.contacts[c * 3 + 0] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-0.5f, 0.5f) };
            s.contacts[c * 3 + 1] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), u2f(rng() & 0x07FFFFFFu) };
            const F4 q = randQuat(); // xyz of a unit quat: genuinely NON-unit, so getNormalized is load-bearing
            s.contacts[c * 3 + 2] = F4{ q.x, q.y, q.z, 0 };
        }

        const uint32_t total0 = pcExpect; // 5.12: no average-point inflation
        const uint32_t initP = 64 * (rng() % 16), initC = 16 * (rng() % 32), initF = 8 * (rng() % 32);
        uint32_t limP = initP + 64 * 20, limC = initC + 16 * 40, limF = initF + 8 * 40;
        const uint32_t v = rng() % 8;
        if (v == 1) limP = initP + 64 * (nbMan ? rng() % nbMan : 0);          // patch overflow
        else if (v == 2 && total0) limC = initC + 16 * (rng() % total0);      // contact overflow
        else if (v == 3 && total0) limF = initF + 8 * (rng() % total0);       // force overflow
        else if (v == 4) { limP = initP + 64 * nbMan; limC = initC + 16 * total0; limF = initF + 8 * total0; if (total0) ++cExact; } // exact fits
        else if (v == 5) { limP = initP + 64 * (nbMan ? rng() % nbMan : 0); limC = initC; } // patch AND contact both tight: the chained else-if must report only the FIRST overflow
        s.cfg = U4{ limP, limC, limF, 0 };
        s.counters = U4{ initP, initC, initF, 0 };

        in[i] = s;
        maskWord[i] = refTrimeshOut(s, mats, ref[i]);
        const uint32_t f = ref[i].counters.w;
        if (f & 4u) ++cOvfP;
        if (f & 1u) ++cOvfC;
        if (f & 2u) ++cOvfF;
        if (nbMan == 0) ++cZeroMan;
        if (nbMan > 0 && pcExpect == 0) ++cNoRes;
        if (insertAvg) ++cAvg;
        if (flip) ++cFlip;
        if (!flip && other == T_SPHERE) ++cSphere;
        if (!flip && other == T_CAPSULE) ++cCapsule;
        if (maskWord[i] != 0xFFFFFFFFu) ++cConflict;
        for (uint32_t m = 0; m < nbMan; ++m) if ((&s.nbC.x)[m] == 0) { ++cEmptyMan; break; }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<TmoOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bMat = createBufferFrom(dev, mats);

    struct TmoParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxTrimeshOut.spv", "cvxTrimeshOut",
             { (uint32_t)sizeof(TmoIn), (uint32_t)sizeof(TmoOut), (uint32_t)sizeof(MaterialData) }, sizeof(TmoParams));
    k.dispatch(GROUPS, { bIn, bOut, bMat }, prm);
    dev->flush();

    const auto g = downloadBuffer<TmoOut>(dev, bOut, N);
    size_t mmR = 0, mmO = 0, mmC = 0, mmP = 0, mmS = 0, mmF = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const U4& rr = ref[i].res; const U4& gr = g[i].res;
        if (rr.x != gr.x || rr.y != gr.y || rr.z != gr.z || rr.w != gr.w) ++mmR;
        const U4& ro = ref[i].outputs; const U4& go = g[i].outputs;
        if (ro.x != go.x || ro.y != go.y || ro.z != go.z || ro.w != go.w) ++mmO;
        const U4& rc = ref[i].counters; const U4& gc = g[i].counters;
        if (rc.x != gc.x || rc.y != gc.y || rc.z != gc.z || rc.w != gc.w) ++mmC;
        for (int p = 0; p < 16; ++p)
        {
            const U4& rp = ref[i].patches[p]; const U4& gp = g[i].patches[p];
            // floats within the patch image compare bitwise except the rotated
            // normal words (p%4==1 xyz) which go through eqF
            if ((p & 3) == 1)
            {
                if (!eqF(u2f(rp.x), u2f(gp.x)) || !eqF(u2f(rp.y), u2f(gp.y)) ||
                    !eqF(u2f(rp.z), u2f(gp.z)) || rp.w != gp.w) { ++mmP; break; }
            }
            else if (rp.x != gp.x || rp.y != gp.y || rp.z != gp.z || rp.w != gp.w) { ++mmP; break; }
        }
        for (int c = 0; c < 24; ++c)
        {
            const F4& rs = ref[i].cstream[c]; const F4& gs = g[i].cstream[c];
            if (!eqF(rs.x, gs.x) || !eqF(rs.y, gs.y) || !eqF(rs.z, gs.z) || !eqF(rs.w, gs.w)) { ++mmS; break; }
        }
        for (int wj = 0; wj < 64; ++wj)
        {
            if (uint32_t(wj) == maskWord[i]) continue; // unordered CUDA write conflict
            const uint32_t rv = (&ref[i].fstream[wj / 4].x)[wj & 3];
            const uint32_t gv = (&g[i].fstream[wj / 4].x)[wj & 3];
            if (rv != gv) { ++mmF; break; }
        }
    }
    PXG_CHECK_MSG(mmR == 0, "trimeshOut: %zu res mismatches", mmR);
    PXG_CHECK_MSG(mmO == 0, "trimeshOut: %zu output-pointer mismatches", mmO);
    PXG_CHECK_MSG(mmC == 0, "trimeshOut: %zu counter mismatches", mmC);
    PXG_CHECK_MSG(mmP == 0, "trimeshOut: %zu patch mismatches", mmP);
    PXG_CHECK_MSG(mmS == 0, "trimeshOut: %zu contact-stream mismatches", mmS);
    PXG_CHECK_MSG(mmF == 0, "trimeshOut: %zu face-index mismatches", mmF);
    PXG_CHECK_MSG(cOvfP > N / 50 && cOvfC > N / 60 && cOvfF > N / 60 && cZeroMan > N / 20 &&
                  cNoRes > N / 100 && cAvg > N / 5 && cFlip > N / 4 && cSphere > N / 30 &&
                  cCapsule > N / 30 && cConflict == 0 && cEmptyMan > N / 10 && cExact > N / 20,
                  "trimeshOut: coverage thin (ovfP %zu ovfC %zu ovfF %zu zeroMan %zu noRes %zu avg %zu flip %zu sph %zu cap %zu conf %zu emptyMan %zu exact %zu)",
                  cOvfP, cOvfC, cOvfF, cZeroMan, cNoRes, cAvg, cFlip, cSphere, cCapsule, cConflict, cEmptyMan, cExact);

    for (PxgBufferHandle h : { bIn, bOut, bMat }) dev->destroyBuffer(h);
}


// P9.86e DIRECTED SIGNED-ZERO MATERIALS through the record-model shader (npCvxTrimeshOut.spv); the same
// directed bank and cases as cvxTriFinish below, which drives the engine BDA twin.  eMIN/eMAX are
// PxMin/PxMax<float> selects: a +-0 tie returns the SECOND operand, so PxMin(-0,+0) = +0 and
// PxMax(+0,-0) = -0, where GLSL FMin/FMax on llvmpipe give -0 / +0 in either order.  The random banks of
// npCvxTrimeshOut_finish never produce -0, so without these cases the patch words are blind to the tie sign.
PXG_TEST(npCvxTrimeshOut_materialZeroSign)
{
    const float PZ = 0.f, NZ = u2f(0x80000000u);
    // Fields: {dyn, sta, rest, damp, flags, fricMode, restMode, dampMode}; modes 1 = eMIN, 3 = eMAX.
    const std::vector<MaterialData> mats = {
        { PZ, NZ, NZ, NZ, 0, 1, 1, 1 },          // D0  -0 under eMIN
        { PZ, PZ, PZ, PZ, 0, 1, 1, 1 },          // D1  +0 under eMIN
        { PZ, PZ, PZ, PZ, 0, 3, 3, 3 },          // D2  +0 under eMAX
        { PZ, NZ, NZ, NZ, 0, 3, 3, 3 },          // D3  -0 under eMAX
        { PZ, PZ, -0.5f, PZ, 0, 0, 0, 0 },       // D4  compliant (rest < 0), damping +0
        { PZ, PZ, NZ, NZ, 0, 0, 0, 0 },          // D5  rigid (rest -0), damping -0
        { NZ, PZ, PZ, PZ, 0, 0, 0, 0 },          // D6  dynamic friction -0 under eAVERAGE
    };
    // One convex/trimesh pair per case, one manifold with one contact, counters 0: patch 0 lands at word 0,
    // so words 7 / 8 / 9 / 10 are restitution / dynF / staF / damping (cu:70-73).  flip puts the trimesh
    // first (shape types swapped); the material order stays shapeInfo.z / .w.  w7..w10 are the CUDA's bits
    // computed by hand from PxsCombineMaterials; killsVariant names the host twin whose image must differ.
    struct DirCase { uint32_t ma, mb; bool flip; uint32_t w7, w8, w9, w10; int killsVariant; };
    const uint32_t NDIR = 8;
    const DirCase dc[NDIR] = {
        { 0, 1, false, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 1 },   // K0 eMIN(-0,+0): PxMin +0, FMin -0
        { 2, 3, false, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 2 },   // K1 eMAX(+0,-0): PxMax -0, FMax +0
        { 4, 5, false, 0xBF000000u, 0u, 0x00000000u, 0x80000000u, 2 },   // K2 one compliant: damping forced to eMAX(+0,-0)
        { 1, 0, false, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 0 },   // K3 control: eMIN(+0,-0) = -0 in both
        { 3, 2, false, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K4 control: eMAX(-0,+0) = +0 in both
        { 6, 6, false, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K5 d = 0.5*(-0 + -0) = -0: PxMax(d, 0) keeps dynF +0
        { 0, 1, true,  0x00000000u, 0u, 0x00000000u, 0x00000000u, 1 },   // K6 = K0, trimesh first
        { 2, 3, true,  0x80000000u, 0u, 0x80000000u, 0x80000000u, 2 },   // K7 = K1, trimesh first
    };
    PXG_CHECK_MSG(std::signbit(0.5f * (mats[6].dynamicFriction + mats[6].dynamicFriction)),
                  "trimeshOut directed: K5 must reach the dynF clamp with -0");

    std::vector<TmoIn> in(NDIR);
    std::vector<TmoOut> ref(NDIR);
    size_t cMinTie = 0, cMaxTie = 0;
    for (uint32_t k = 0; k < NDIR; ++k)
    {
        TmoIn s{};
        s.hdr = U4{ 1, 0, 0, 0 };
        s.nbC = U4{ 1, 0, 0, 0 };
        s.shapeInfo = dc[k].flip ? U4{ T_TRIMESH, T_CONVEX, dc[k].ma, dc[k].mb } : U4{ T_CONVEX, T_TRIMESH, dc[k].ma, dc[k].mb };
        s.scaleA = s.scaleB = F4{ 1, 0.5f, 1, 0 };
        s.tAq = s.tBq = F4{ 0, 0, 0, 1 };
        s.tAp = F4{ 1, 2, 3, 0 };
        s.tBp = F4{ 0, 0, 0, 0 };
        s.contacts[0] = F4{ 0.25f, -0.5f, 1, 0.125f };
        s.contacts[1] = F4{ 1, 2, -0.5f, u2f(5u) };
        s.contacts[2] = F4{ 0, 0.6f, 0.8f, 0 };
        s.cfg = U4{ 64 * 20, 16 * 40, 8 * 40, 0 };
        s.counters = U4{ 0, 0, 0, 0 };
        in[k] = s;
        const uint32_t mw = refTrimeshOut(s, mats, ref[k]);
        PXG_CHECK_MSG(mw == 0xFFFFFFFFu && ref[k].outputs.y == 0u && ref[k].patches[1].w == dc[k].w7 &&
                      ref[k].patches[2].x == dc[k].w8 && ref[k].patches[2].y == dc[k].w9 && ref[k].patches[2].z == dc[k].w10,
                      "trimeshOut directed %u: oracle rest/dynF/staF/damp %08x %08x %08x %08x, CUDA %08x %08x %08x %08x",
                      k, ref[k].patches[1].w, ref[k].patches[2].x, ref[k].patches[2].y, ref[k].patches[2].z,
                      dc[k].w7, dc[k].w8, dc[k].w9, dc[k].w10);
        for (int v = 1; v <= 2; ++v)
        {
            TmoOut tw;
            gCombineVariant = v;
            refTrimeshOut(s, mats, tw);
            gCombineVariant = 0;
            const bool differs = std::memcmp(tw.patches, ref[k].patches, sizeof tw.patches) != 0;
            PXG_CHECK_MSG(differs == (dc[k].killsVariant == v), "trimeshOut directed %u: twin %d %s the image",
                          k, v, differs ? "changes" : "does not change");
            if (differs) ++(v == 1 ? cMinTie : cMaxTie);
        }
    }
    PXG_CHECK_MSG(cMinTie == 2 && cMaxTie == 3, "trimeshOut: signed-zero combine coverage thin (minTie %zu maxTie %zu)",
                  cMinTie, cMaxTie);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<TmoOut> outInit(NDIR);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    struct TmoParams { uint32_t numTests; } prm = { NDIR };
    Kernel kr("npCvxTrimeshOut.spv", "cvxTrimeshOut",
              { (uint32_t)sizeof(TmoIn), (uint32_t)sizeof(TmoOut), (uint32_t)sizeof(MaterialData) }, sizeof(TmoParams));
    kr.dispatch(1, { bIn, bOut, bMat }, prm);
    dev->flush();
    const auto g = downloadBuffer<TmoOut>(dev, bOut, NDIR);
    for (uint32_t k = 0; k < NDIR; ++k)
    {
        // Words 7..10 bitwise, and the whole patch image apart from the rotated normal (words 4..6, eqF).
        const U4* rp = ref[k].patches; const U4* gp = g[k].patches;
        bool ok = true;
        for (int p = 0; p < 16 && ok; ++p)
        {
            if ((p & 3) == 1)
                ok = eqF(u2f(rp[p].x), u2f(gp[p].x)) && eqF(u2f(rp[p].y), u2f(gp[p].y)) &&
                     eqF(u2f(rp[p].z), u2f(gp[p].z)) && rp[p].w == gp[p].w;
            else
                ok = rp[p].x == gp[p].x && rp[p].y == gp[p].y && rp[p].z == gp[p].z && rp[p].w == gp[p].w;
        }
        PXG_CHECK_MSG(ok, "trimeshOut directed %u: GPU rest/dynF/staF/damp %08x %08x %08x %08x, oracle %08x %08x %08x %08x",
                      k, gp[1].w, gp[2].x, gp[2].y, gp[2].z, rp[1].w, rp[2].x, rp[2].y, rp[2].z);
    }

    for (PxgBufferHandle h : { bIn, bOut, bMat }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// cvxTriFinishBda -- the SAME kernel on the engine BDA ABI, with real atomics on
// a real PxgPatchAndContactCounters and real writes into the three streams.
//
// Driven against the SAME refTrimeshOut oracle, so a divergence is attributable
// to the BDA plumbing and the warp emulation rather than to the algorithm.
//
// ONE PAIR PER DISPATCH, AND THAT IS NOT A SHORTCUT.  The reservations are
// InterlockedAdds on a shared cursor, so with several pairs in flight the byte
// offsets each pair receives depend on the order the atomics land -- which is
// exactly what the oracle cannot predict.  Every case therefore gets its own
// counters, its own streams and its own output record (a per-case slice of one
// buffer, addressed through the push blob), and is dispatched on its own with
// numPairs == 1.  All the dispatches are queued and flushed together; they touch
// disjoint memory, so the atomics are deterministic per case.
// ---------------------------------------------------------------------------

struct CfPush
{
    uint64_t pairs, transformCache, gpuShapes, cmInputs, cmOutputs, cmMultiManifold;
    uint32_t numPairs, _pad52;
    uint64_t materials, contactStream, patchStream, forceAndIndiceStream;
    uint32_t insertAveragePointWord, _pad92;
    uint64_t patchAndContactCounters, touchChangeFlags, patchChangeFlags;
    uint64_t startContactPatches, startContactPoints, startContactForces;
    uint32_t patchBytesLimit, contactBytesLimit, forceBytesLimit, _pad156;
};
static_assert(sizeof(CfPush) == 160, "CfParams blob is 160 bytes");

PXG_TEST(cvxTriFinish)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::mt19937 rng(4242001);
    // NRAND random cases, then NDIR directed ones (P9.86) that draw nothing from rng, so the random
    // cases keep their exact inputs.  The directed materials sit at [NMAT, NMAT + NDM).
    const uint32_t NRAND = 220, NDIR = 6, N = NRAND + NDIR, NMAT = 32, NDM = 7;
    const uint32_t STREAM = 2048;   // per-case slice of each of the three streams
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; }
                            const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
                            return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<MaterialData> mats(NMAT);
    for (auto& m : mats)
    {
        m.dynamicFriction = rf(0, 1.5f); m.staticFriction = rf(0, 2.f);
        m.restitution = rf(-1.f, 1.f); m.damping = rf(0, 2.f);
        m.flags = (rng() % 4 == 0 ? 1u : 0u) | (rng() % 3 == 0 ? 16u : 0u) | (rng() % 5 == 0 ? 256u : 0u);
        m.fricMode = rng() % 4; m.restMode = rng() % 4; m.dampMode = rng() % 4;
    }
    // P9.88: a second bank at [0x10000 + i] with DIFFERENT materials (its own generator, so the
    // case stream is unchanged), so an index k | 0x10000 tells a lookup that narrows to PxU16
    // first (cu:61 static_cast<PxU16>) from one that does not; both stay in bounds.
    {
        std::mt19937 rng2(88001);
        auto rf2 = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng2() % 1000000) / 1000000.f); };
        mats.resize(0x10000u + NMAT);
        for (uint32_t i = 0; i < NMAT; ++i)
        {
            MaterialData& m = mats[0x10000u + i];
            m.dynamicFriction = rf2(0, 1.5f); m.staticFriction = rf2(0, 2.f);
            m.restitution = rf2(-1.f, 1.f); m.damping = rf2(0, 2.f);
            m.flags = (rng2() % 4 == 0 ? 1u : 0u) | (rng2() % 3 == 0 ? 16u : 0u) | (rng2() % 5 == 0 ? 256u : 0u);
            m.fricMode = rng2() % 4; m.restMode = rng2() % 4; m.dampMode = rng2() % 4;
        }
    }

    // P9.86 DIRECTED SIGNED-ZERO MATERIALS.  eMIN/eMAX are PxMin/PxMax(float) selects
    // (PxsMaterialCombiner.h:43/47/60/64, PxUnixMathIntrinsics.h:116-125): a +-0 tie returns the SECOND
    // operand, so PxMin(-0,+0) = +0 and PxMax(+0,-0) = -0, where GLSL FMin/FMax on llvmpipe give -0 / +0 in
    // either order (F4).  -0 is a valid material value: NpMaterial only checks finite, <= 1 and >= 0.f, and a
    // restitution of -0 is rigid (-0 < 0 is false).  The random banks never produce -0 (rf = lo + (hi-lo)*k).
    // Fields: {dyn, sta, rest, damp, flags, fricMode, restMode, dampMode}; modes 1 = eMIN, 3 = eMAX.
    {
        const float PZ = 0.f, NZ = u2f(0x80000000u);
        const MaterialData dirMats[NDM] = {
            { PZ, NZ, NZ, NZ, 0, 1, 1, 1 },          // D0  -0 under eMIN
            { PZ, PZ, PZ, PZ, 0, 1, 1, 1 },          // D1  +0 under eMIN
            { PZ, PZ, PZ, PZ, 0, 3, 3, 3 },          // D2  +0 under eMAX
            { PZ, NZ, NZ, NZ, 0, 3, 3, 3 },          // D3  -0 under eMAX
            { PZ, PZ, -0.5f, PZ, 0, 0, 0, 0 },       // D4  compliant (rest < 0), damping +0
            { PZ, PZ, NZ, NZ, 0, 0, 0, 0 },          // D5  rigid (rest -0), damping -0
            { NZ, PZ, PZ, PZ, 0, 0, 0, 0 },          // D6  dynamic friction -0 under eAVERAGE
        };
        for (uint32_t j = 0; j < NDM; ++j) mats[NMAT + j] = dirMats[j];
    }
    // One no-flip convex/trimesh pair per directed case, one manifold with one contact, counters 0: patch 0
    // lands at slice word 0, so words 7 / 8 / 9 / 10 are restitution / dynF / staF / damping (cu:70-73).
    // w7..w10 are the CUDA's bits, computed by hand from PxsCombineMaterials.  killsVariant names the host
    // twin (gCombineVariant) whose patch image must differ: 1 = the eMIN revert, 2 = the eMAX revert.
    struct DirCase { uint32_t ma, mb, w7, w8, w9, w10; int killsVariant; };
    const DirCase dirCases[NDIR] = {
        { 0, 1, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 1 },   // K0 eMIN(-0,+0): PxMin +0, FMin -0
        { 2, 3, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 2 },   // K1 eMAX(+0,-0): PxMax -0, FMax +0
        { 4, 5, 0xBF000000u, 0u, 0x00000000u, 0x80000000u, 2 },   // K2 one compliant: damping forced to eMAX(+0,-0)
        { 1, 0, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 0 },   // K3 control: eMIN(+0,-0) = -0 in both
        { 3, 2, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K4 control: eMAX(-0,+0) = +0 in both
        { 6, 6, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K5 d = 0.5*(-0 + -0) = -0: PxMax(d, 0) keeps dynF +0
    };
    size_t cMinTie = 0, cMaxTie = 0;

    std::vector<TmoIn> in(N);
    std::vector<TmoOut> ref(N);
    std::vector<uint32_t> maskWord(N, 0xFFFFFFFFu);
    size_t cOvfP = 0, cOvfC = 0, cOvfF = 0, cZeroMan = 0, cNoRes = 0, cAvg = 0, cFlip = 0,
           cSphere = 0, cCapsule = 0, cConflict = 0, cExact = 0, cRollbackPatch = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        if (i >= NRAND)
        {
            const DirCase& dc = dirCases[i - NRAND];
            TmoIn s{};
            s.hdr = U4{ 1, 0, 0, 0 };
            s.nbC = U4{ 1, 0, 0, 0 };
            s.shapeInfo = U4{ T_CONVEX, T_TRIMESH, NMAT + dc.ma, NMAT + dc.mb };
            s.scaleA = s.scaleB = F4{ 1, 0.5f, 1, 0 };
            s.tAq = s.tBq = F4{ 0, 0, 0, 1 };
            s.tAp = F4{ 1, 2, 3, 0 };
            s.tBp = F4{ 0, 0, 0, 0 };
            s.contacts[0] = F4{ 0.25f, -0.5f, 1, 0.125f };
            s.contacts[1] = F4{ 1, 2, -0.5f, u2f(5u) };
            s.contacts[2] = F4{ 0, 0.6f, 0.8f, 0 };
            s.cfg = U4{ 64 * 20, 16 * 40, 8 * 40, 0 };
            s.counters = U4{ 0, 0, 0, 0 };
            in[i] = s;
            maskWord[i] = refTrimeshOut(s, mats, ref[i], &patchPoison);
            PXG_CHECK_MSG(ref[i].outputs.y == 0u && ref[i].patches[1].w == dc.w7 && ref[i].patches[2].x == dc.w8 &&
                          ref[i].patches[2].y == dc.w9 && ref[i].patches[2].z == dc.w10,
                          "cvxTriFinish directed %u: oracle rest/dynF/staF/damp %08x %08x %08x %08x, CUDA %08x %08x %08x %08x",
                          i - NRAND, ref[i].patches[1].w, ref[i].patches[2].x, ref[i].patches[2].y, ref[i].patches[2].z,
                          dc.w7, dc.w8, dc.w9, dc.w10);
            for (int v = 1; v <= 2; ++v)
            {
                TmoOut tw;
                gCombineVariant = v;
                refTrimeshOut(s, mats, tw, &patchPoison);
                gCombineVariant = 0;
                const bool differs = std::memcmp(tw.patches, ref[i].patches, sizeof tw.patches) != 0;
                PXG_CHECK_MSG(differs == (dc.killsVariant == v), "cvxTriFinish directed %u: twin %d %s the image",
                              i - NRAND, v, differs ? "changes" : "does not change");
                if (differs) ++(v == 1 ? cMinTie : cMaxTie);
            }
            continue;
        }
        TmoIn s{};
        const uint32_t nbMan = (i % 13 == 0) ? 0u : (1 + rng() % 4);
        const bool insertAvg = (rng() % 5) < 2;
        // allflags carries a random previous status/patch count so the
        // touch-change and patch-change decisions are exercised in both directions.
        s.hdr = U4{ nbMan, uint32_t(rng()), insertAvg ? 1u : 0u, 0 };
        uint32_t pcExpect = 0;
        for (int m = 0; m < 4; ++m)
        {
            uint32_t nc = rng() % 7;
            if (i % 7 == 3) nc = (m == 0) ? 0u : nc;   // manifold 0 empty: lane 0 holds no contact
            (&s.nbC.x)[m] = nc;
            if (uint32_t(m) < nbMan) pcExpect += nc;
        }
        if (i % 11 == 5) { s.nbC = U4{ 0, 0, 0, 0 }; pcExpect = 0; }  // the no-reservation path

        const bool flip = rng() & 1;
        const uint32_t other = (rng() % 3 == 0) ? T_SPHERE : ((rng() % 2) ? T_CAPSULE : T_CONVEX);
        s.shapeInfo = U4{ flip ? T_TRIMESH : other, flip ? other : T_TRIMESH,
                          uint32_t(rng() % NMAT) | ((i % 4 == 1) ? 0x10000u : 0u), uint32_t(rng() % NMAT) | ((i % 5 == 2) ? 0x10000u : 0u) };   // P9.88
        s.scaleA = F4{ rf(0.5f, 2), rf(0.1f, 1.5f), rf(0.5f, 2), 0 };
        s.scaleB = F4{ rf(0.5f, 2), rf(0.1f, 1.5f), rf(0.5f, 2), 0 };
        s.tAq = randQuat(); s.tAp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        s.tBq = randQuat(); s.tBp = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };

        for (int c = 0; c < 24; ++c)
        {
            s.contacts[c * 3 + 0] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-0.5f, 0.5f) };
            s.contacts[c * 3 + 1] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), u2f(rng() & 0x07FFFFFFu) };
            const F4 q = randQuat();   // NON-unit xyz, so getNormalized is load-bearing
            s.contacts[c * 3 + 2] = F4{ q.x, q.y, q.z, 0 };
        }

        const uint32_t total0 = pcExpect; // 5.12: no average-point inflation
        const uint32_t initP = 64 * (rng() % 8), initC = 16 * (rng() % 16), initF = 8 * (rng() % 16);
        uint32_t limP = initP + 64 * 20, limC = initC + 16 * 40, limF = initF + 8 * 40;
        const uint32_t v = rng() % 8;
        if (v == 1) limP = initP + 64 * (nbMan ? rng() % nbMan : 0);
        else if (v == 2 && total0) limC = initC + 16 * (rng() % total0);
        else if (v == 3 && total0) limF = initF + 8 * (rng() % total0);
        else if (v == 4) { limP = initP + 64 * nbMan; limC = initC + 16 * total0; limF = initF + 8 * total0; if (total0) ++cExact; }
        else if (v == 5) { limP = initP + 64 * (nbMan ? rng() % nbMan : 0); limC = initC; }  // chained else-if: only the FIRST overflow is reported
        if (i % 9 == 4 && nbMan > 0)
        {
            // prevPatches == nbManifolds AND a forced overflow.  The patch-change
            // flag is then `prevPatches != 0` (true) if the ROLLED-BACK count is
            // used and `prevPatches != nbManifolds` (false) if the pre-rollback one
            // is -- the only input that separates them.
            s.hdr.y = (s.hdr.y & ~0xFF00u) | ((nbMan & 0xFFu) << 8);
            limP = initP;   // patch reservation cannot fit
            s.cfg = U4{ limP, limC, limF, 0 };
            ++cRollbackPatch;
        }
        else
            s.cfg = U4{ limP, limC, limF, 0 };
        s.counters = U4{ initP, initC, initF, 0 };

        in[i] = s;
        maskWord[i] = refTrimeshOut(s, mats, ref[i], &patchPoison);
        const uint32_t f = ref[i].counters.w;
        if (f & 4u) ++cOvfP;
        if (f & 1u) ++cOvfC;
        if (f & 2u) ++cOvfF;
        if (nbMan == 0) ++cZeroMan;
        if (nbMan > 0 && pcExpect == 0) ++cNoRes;
        if (insertAvg) ++cAvg;
        if (flip) ++cFlip;
        if (!flip && other == T_SPHERE) ++cSphere;
        if (!flip && other == T_CAPSULE) ++cCapsule;
        if (maskWord[i] != 0xFFFFFFFFu) ++cConflict;
    }

    // ---- engine buffers, one slice per case --------------------------------
    // TWO PAIRS PER DISPATCH.  Pair 1 is a DECOY with no manifolds, so under the
    // correct 1216-byte stride it reserves nothing and the counters are the real
    // pair's alone.  A kernel using a 1200-byte stride reads the decoy's header
    // from a different place -- where a live-looking count is planted -- and its
    // reservation moves the counters, which the counter comparison already checks.
    // It also means the constant-trip-2 warp loop is exercised rather than assumed.
    std::vector<uint32_t> pairBuf(N * 2 * 14, 0u);
    std::vector<float> cacheBuf(N * 2 * 8, 0.0f);      // PxsCachedTransform, 32 B
    std::vector<uint32_t> shapeBuf(N * 2 * 12, 0u);    // PxgShape, 48 B
    std::vector<uint32_t> inputBuf(N * 2 * 4, 0u);     // PxgContactManagerInput, 16 B
    std::vector<uint32_t> outBuf(N * 2 * 12, 0u);      // PxsContactManagerOutput, 48 B
    std::vector<uint32_t> manBuf(N * 2 * (1216 / 4), 0u);
    const uint32_t OVF_SEED = 8u;   // not one of PATCH/CONTACT/FORCE_OVF
    std::vector<uint32_t> ctrBuf(N * 4, 0u);
    std::vector<uint32_t> touchBuf(N * 2, 0xDEADBEEFu), patchChgBuf(N * 2, 0xDEADBEEFu);
    std::vector<uint32_t> patchStream(N * (STREAM / 4));   // poisoned per word (patchPoison)
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t w = 0; w < STREAM / 4; ++w) patchStream[i * (STREAM / 4) + w] = patchPoison(w);
    std::vector<uint32_t> contactStream(N * (STREAM / 4), 0u);
    std::vector<uint32_t> forceStream(N * (STREAM / 4), 0u);
    std::vector<uint32_t> matBuf(mats.size() * 6, 0u);   // PxsMaterialData, 24 B (both banks)

    for (uint32_t m = 0; m < uint32_t(mats.size()); m = (m + 1 == NMAT + NDM) ? 0x10000u : m + 1)
    {
        uint32_t* w = &matBuf[m * 6];
        std::memcpy(&w[0], &mats[m].dynamicFriction, 4);
        std::memcpy(&w[1], &mats[m].staticFriction, 4);
        std::memcpy(&w[2], &mats[m].restitution, 4);
        std::memcpy(&w[3], &mats[m].damping, 4);
        w[4] = (mats[m].flags & 0xFFFFu) | ((mats[m].fricMode & 0xFFu) << 16) | ((mats[m].restMode & 0xFFu) << 24);
        w[5] = mats[m].dampMode & 0xFFu;
    }

    // The stored allflags must round-trip: the kernel reads it, rewrites it, and
    // preserves the upper half of nbContactsFlags.
    std::vector<uint32_t> origNbFlags(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const TmoIn& s = in[i];
        pairBuf[i * 28 + 12] = s.shapeInfo.z;   // materialIndices.x @48
        pairBuf[i * 28 + 13] = s.shapeInfo.w;   // materialIndices.y @52

        auto putXf = [&](uint32_t slot, const F4& q, const F4& p) {
            float* w = &cacheBuf[(i * 2 + slot) * 8];
            w[0] = q.x; w[1] = q.y; w[2] = q.z; w[3] = q.w;
            w[4] = p.x; w[5] = p.y; w[6] = p.z;
        };
        putXf(0, s.tAq, s.tAp);
        putXf(1, s.tBq, s.tBp);

        auto putShape = [&](uint32_t slot, const F4& sc, uint32_t type) {
            uint32_t* w = &shapeBuf[(i * 2 + slot) * 12];
            std::memcpy(&w[0], &sc.x, 4); std::memcpy(&w[1], &sc.y, 4); std::memcpy(&w[2], &sc.z, 4);
            w[6] = 0x3F800000u;   // scaleRot.w
            w[10] = type;
        };
        putShape(0, s.scaleA, s.shapeInfo.x);
        putShape(1, s.scaleB, s.shapeInfo.y);

        for (uint32_t pr = 0; pr < 2; ++pr)
        {
            uint32_t* ci = &inputBuf[(i * 2 + pr) * 4];
            ci[0] = 0; ci[1] = 1; ci[2] = 0; ci[3] = 1;   // shapeRef0/1, tcRef0/1
        }

        origNbFlags[i] = 0xABCD0000u | (i & 0xFFFFu);
        uint32_t* co = &outBuf[i * 24];
        co[8] = s.hdr.y;          // allflagsWord @32
        co[9] = origNbFlags[i];   // nbContactsFlags @36
        outBuf[i * 24 + 12 + 8] = 0u;   // the decoy's allflags
        outBuf[i * 24 + 12 + 9] = 0u;

        uint32_t* mm = &manBuf[i * 2 * (1216 / 4)];
        for (uint32_t m = 0; m < 4; ++m)
            for (uint32_t c = 0; c < 6; ++c)
            {
                const uint32_t base = (m * 6 + c) * 3;
                float* cp = reinterpret_cast<float*>(&mm[(m * 6 + c) * 12]);
                cp[0] = s.contacts[base + 0].x; cp[1] = s.contacts[base + 0].y; cp[2] = s.contacts[base + 0].z;
                cp[3] = s.contacts[base + 1].x; cp[4] = s.contacts[base + 1].y; cp[5] = s.contacts[base + 1].z;
                cp[6] = s.contacts[base + 2].x; cp[7] = s.contacts[base + 2].y; cp[8] = s.contacts[base + 2].z;
                cp[9] = s.contacts[base + 0].w;                                  // penetration
                std::memcpy(&mm[(m * 6 + c) * 12 + 10], &s.contacts[base + 1].w, 4); // triIndex
            }
        mm[1152 / 4 + 0] = s.nbC.x; mm[1152 / 4 + 1] = s.nbC.y;
        mm[1152 / 4 + 2] = s.nbC.z; mm[1152 / 4 + 3] = s.nbC.w;
        mm[1200 / 4] = s.hdr.x;   // mNbManifolds

        // The decoy, at slot 1.  Its real header says "no manifolds"; the words a
        // 1200-stride reader would take instead say "two manifolds, six contacts".
        mm[2416 / 4] = 0u;                        // correct mNbManifolds  (1216 + 1200)
        mm[2400 / 4] = 2u;                        // what a 1200 stride reads
        mm[2352 / 4 + 0] = 3u; mm[2352 / 4 + 1] = 3u;   // ... and its mNbContacts
        mm[2352 / 4 + 2] = 0u; mm[2352 / 4 + 3] = 0u;

        uint32_t* ct = &ctrBuf[i * 4];
        // The error word starts with an UNRELATED bit already set.  setOverflowError
        // is an atomicOr, so that bit must survive; a plain store would clobber it,
        // and with one pair per dispatch nothing else could tell the two apart.
        ct[0] = s.counters.x; ct[1] = s.counters.y; ct[2] = s.counters.z; ct[3] = OVF_SEED;
    }

    PxgBufferHandle bPairs = createBufferFrom(dev, pairBuf);
    PxgBufferHandle bCache = createBufferFrom(dev, cacheBuf);
    PxgBufferHandle bShapes = createBufferFrom(dev, shapeBuf);
    PxgBufferHandle bInputs = createBufferFrom(dev, inputBuf);
    PxgBufferHandle bOut = createBufferFrom(dev, outBuf);
    PxgBufferHandle bMan = createBufferFrom(dev, manBuf);
    PxgBufferHandle bCtr = createBufferFrom(dev, ctrBuf);
    PxgBufferHandle bTouch = createBufferFrom(dev, touchBuf);
    PxgBufferHandle bPatchChg = createBufferFrom(dev, patchChgBuf);
    PxgBufferHandle bPS = createBufferFrom(dev, patchStream);
    PxgBufferHandle bCS = createBufferFrom(dev, contactStream);
    PxgBufferHandle bFS = createBufferFrom(dev, forceStream);
    PxgBufferHandle bMat = createBufferFrom(dev, matBuf);

    const uint64_t aPairs = dev->getBufferDeviceAddress(bPairs);
    const uint64_t aCache = dev->getBufferDeviceAddress(bCache);
    const uint64_t aShapes = dev->getBufferDeviceAddress(bShapes);
    const uint64_t aInputs = dev->getBufferDeviceAddress(bInputs);
    const uint64_t aOut = dev->getBufferDeviceAddress(bOut);
    const uint64_t aMan = dev->getBufferDeviceAddress(bMan);
    const uint64_t aCtr = dev->getBufferDeviceAddress(bCtr);
    const uint64_t aTouch = dev->getBufferDeviceAddress(bTouch);
    const uint64_t aPatchChg = dev->getBufferDeviceAddress(bPatchChg);
    const uint64_t aPS = dev->getBufferDeviceAddress(bPS);
    const uint64_t aCS = dev->getBufferDeviceAddress(bCS);
    const uint64_t aFS = dev->getBufferDeviceAddress(bFS);
    const uint64_t aMat = dev->getBufferDeviceAddress(bMat);

    {
        Kernel k("cvxTriFinishBda.spv", "convexTrimeshFinishContacts", {}, sizeof(CfPush));
        for (uint32_t i = 0; i < N; ++i)
        {
            CfPush push{};
            push.pairs = aPairs + uint64_t(i) * 112ull;
            push.transformCache = aCache + uint64_t(i) * 64ull;
            push.gpuShapes = aShapes + uint64_t(i) * 96ull;
            push.cmInputs = aInputs + uint64_t(i) * 32ull;
            push.cmOutputs = aOut + uint64_t(i) * 96ull;
            push.cmMultiManifold = aMan + uint64_t(i) * 2432ull;
            push.numPairs = (i % 5 == 2) ? 1u : 2u;   // an ODD count leaves the
                                                      // loop's second trip out of
                                                      // range, which is the only
                                                      // way to reach the guard
            push.materials = aMat;
            push.contactStream = aCS + uint64_t(i) * STREAM;
            push.patchStream = aPS + uint64_t(i) * STREAM;
            push.forceAndIndiceStream = aFS + uint64_t(i) * STREAM;
            push.insertAveragePointWord = in[i].hdr.z;
            push.patchAndContactCounters = aCtr + uint64_t(i) * 16ull;
            push.touchChangeFlags = aTouch + uint64_t(i) * 8ull;
            push.patchChangeFlags = aPatchChg + uint64_t(i) * 8ull;
            push.startContactPatches = aPS + uint64_t(i) * STREAM;
            push.startContactPoints = aCS + uint64_t(i) * STREAM;
            push.startContactForces = aFS + uint64_t(i) * STREAM;
            push.patchBytesLimit = in[i].cfg.x;
            push.contactBytesLimit = in[i].cfg.y;
            push.forceBytesLimit = in[i].cfg.z;
            k.dispatch(1u, {}, push, 1u, 1u);
        }
        dev->flush();
    }

    const auto gOutB = downloadBuffer<uint32_t>(dev, bOut, outBuf.size());
    const auto gCtr = downloadBuffer<uint32_t>(dev, bCtr, ctrBuf.size());
    const auto gTouch = downloadBuffer<uint32_t>(dev, bTouch, N * 2);
    const auto gPatchChg = downloadBuffer<uint32_t>(dev, bPatchChg, N * 2);
    const auto gPS = downloadBuffer<uint32_t>(dev, bPS, patchStream.size());
    const auto gCS = downloadBuffer<uint32_t>(dev, bCS, contactStream.size());
    const auto gFS = downloadBuffer<uint32_t>(dev, bFS, forceStream.size());

    size_t mmFlags = 0, mmCtr = 0, mmPtr = 0, mmPatch = 0, mmContact = 0, mmFace = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const TmoOut& r = ref[i];
        const uint32_t* co = &gOutB[i * 24];

        if (co[8] != r.res.x) ++mmFlags;
        if ((co[9] & 0xFFFFu) != (r.res.w & 0xFFFFu)) ++mmFlags;
        if ((co[9] & 0xFFFF0000u) != (origNbFlags[i] & 0xFFFF0000u)) ++mmFlags;  // upper half preserved
        if (gTouch[i * 2] != r.res.y) ++mmFlags;
        if (gPatchChg[i * 2] != r.res.z) ++mmFlags;

        // THE DECOY'S OWN OUTPUT.  Its allflags start at zero and it has no
        // manifolds, so the lane-0 block -- which runs whatever the manifold count
        // -- must leave exactly this behind.  Checking it is what pins the
        // constant-trip-2 warp loop: a kernel covering one warp per block would
        // never reach pair 1 and these words would keep their 0xDEADBEEF/0 init.
        const uint32_t* dco = &gOutB[i * 24 + 12];
        if (i % 5 == 2)
        {
            // numPairs == 1: the loop's second trip is OUT OF RANGE and must be
            // skipped, so the decoy's record has to come back exactly as uploaded.
            if (dco[8] != 0u || dco[9] != 0u) ++mmFlags;
            if (gTouch[i * 2 + 1] != 0xDEADBEEFu) ++mmFlags;
            if (gPatchChg[i * 2 + 1] != 0xDEADBEEFu) ++mmFlags;
        }
        else
        {
            if (dco[8] != 0x00010000u) ++mmFlags;    // (prevPatches<<8 | HAS_NO_TOUCH) << 16
            if (dco[9] != 0u) ++mmFlags;
            if (gTouch[i * 2 + 1] != 1u) ++mmFlags;  // touch was never known before
            if (gPatchChg[i * 2 + 1] != 0u) ++mmFlags;
        }

        const uint32_t* ct = &gCtr[i * 4];
        if (ct[0] != r.counters.x || ct[1] != r.counters.y || ct[2] != r.counters.z ||
            ct[3] != (OVF_SEED | r.counters.w)) ++mmCtr;

        // Output pointers: the engine stores base + offset; overflow stores a
        // literal null, which the oracle models as the NULLS sentinel.
        uint64_t pF, pP, pC;
        std::memcpy(&pF, &co[4], 8); std::memcpy(&pP, &co[0], 8); std::memcpy(&pC, &co[2], 8);
        const uint64_t bF = aFS + uint64_t(i) * STREAM, bP = aPS + uint64_t(i) * STREAM,
                       bC = aCS + uint64_t(i) * STREAM;
        auto ptrOk = [&](uint64_t got, uint64_t base, uint32_t expOff) {
            if (expOff == NULLS) return got == 0ull;
            return got == base + uint64_t(expOff);
        };
        if (!ptrOk(pF, bF, r.outputs.x) || !ptrOk(pP, bP, r.outputs.y) || !ptrOk(pC, bC, r.outputs.z))
            ++mmPtr;

        // Stream contents, only where a reservation actually happened.
        const uint32_t po = r.outputs.y, cob = r.outputs.z, fo = r.outputs.x;
        const uint32_t* ps = &gPS[i * (STREAM / 4)];
        const uint32_t* cs = &gCS[i * (STREAM / 4)];
        const uint32_t* fs = &gFS[i * (STREAM / 4)];

        // AN OVERFLOWING PAIR MUST WRITE NOTHING AT ALL.  Skipping the streams here
        // would leave the cu:257 gate untested: on a patch overflow the CONTACT and
        // FORCE offsets keep their reserved values, so a kernel that ignored the
        // overflow ballot would happily fill those two streams.
        if (r.counters.w != 0u)
        {
            for (uint32_t w = 0; w < STREAM / 4; ++w)
                if (ps[w] != patchPoison(w) || cs[w] != 0u || fs[w] != 0u) { ++mmPatch; break; }
        }

        if (po != NULLS && po != 0xFFFFFFFFu)
            for (uint32_t w = 0; w < 16; ++w)
            {
                const U4& e = r.patches[w];
                const uint32_t* g = &ps[po / 4 + w * 4];
                if (g[0] != e.x || g[1] != e.y || g[2] != e.z || g[3] != e.w)
                {
                    // Only the rotated normal (word 1 xyz) gets the scalar test's float
                    // slack.  Restitution (1.w) and dynF/staF/damping (2.xyz) come out of
                    // exact selects, 0.5*(a+b) and a*b, so they compare BITWISE, the sign
                    // of zero included (P9.86; eqF(+0,-0) would pass either sign).
                    const bool normalWord = (w % 4 == 1);
                    if (!normalWord || !eqF(u2f(g[0]), u2f(e.x)) || !eqF(u2f(g[1]), u2f(e.y)) ||
                        !eqF(u2f(g[2]), u2f(e.z)) || g[3] != e.w)
                    { ++mmPatch; break; }
                }
            }

        // Nothing else in the slice may be written: this also measures the no-reservation path
        // (po == 0xFFFFFFFF), which the slot compare above skips.  po <= 448 (initP), so the
        // compared window po/4 .. po/4+64 stays inside the 512-word slice.
        {
            const bool haveP = (po != NULLS && po != 0xFFFFFFFFu);
            for (uint32_t w = 0; w < STREAM / 4; ++w)
            {
                if (haveP && w >= po / 4 && w < po / 4 + 64u) continue;   // compared above
                if (ps[w] != patchPoison(w)) { ++mmPatch; break; }
            }
        }

        if (cob != NULLS && cob != 0xFFFFFFFFu)
            for (uint32_t c = 0; c < 24; ++c)
            {
                const F4& e = r.cstream[c];
                const float* g = reinterpret_cast<const float*>(&cs[cob / 4 + c * 4]);
                if (!eqF(g[0], e.x) || !eqF(g[1], e.y) || !eqF(g[2], e.z) || !eqF(g[3], e.w))
                { ++mmContact; break; }
            }

        if (fo != NULLS && fo != 0xFFFFFFFFu)
            for (uint32_t w = 0; w < 64; ++w)
            {
                if (w == maskWord[i]) continue;   // the documented intra-warp write conflict
                const uint32_t e = (&r.fstream[w / 4].x)[w & 3];
                if (fs[fo / 4 + w] != e) { ++mmFace; break; }
            }
    }

    // P9.86: the directed cases' material words straight against the CUDA's bits (counters start at 0,
    // so patch 0 sits at slice word 0), independent of the oracle.
    for (uint32_t k = 0; k < NDIR; ++k)
    {
        const uint32_t* ps = &gPS[(NRAND + k) * (STREAM / 4)];
        const DirCase& dc = dirCases[k];
        PXG_CHECK_MSG(ps[7] == dc.w7 && ps[8] == dc.w8 && ps[9] == dc.w9 && ps[10] == dc.w10,
                      "cvxTriFinish directed %u: rest/dynF/staF/damp %08x %08x %08x %08x, CUDA %08x %08x %08x %08x",
                      k, ps[7], ps[8], ps[9], ps[10], dc.w7, dc.w8, dc.w9, dc.w10);
    }
    PXG_CHECK_MSG(cMinTie == 1 && cMaxTie == 2,
                  "cvxTriFinish: signed-zero combine coverage thin (minTie %zu maxTie %zu)", cMinTie, cMaxTie);

    for (PxgBufferHandle h : { bPairs, bCache, bShapes, bInputs, bOut, bMan, bCtr, bTouch,
                               bPatchChg, bPS, bCS, bFS, bMat })
        dev->destroyBuffer(h);

    PXG_CHECK_MSG(mmFlags == 0, "cvxTriFinish: %zu status/flag mismatches", mmFlags);
    PXG_CHECK_MSG(mmCtr == 0, "cvxTriFinish: %zu counter mismatches", mmCtr);
    PXG_CHECK_MSG(mmPtr == 0, "cvxTriFinish: %zu output-pointer mismatches", mmPtr);
    PXG_CHECK_MSG(mmPatch == 0, "cvxTriFinish: %zu patch-stream mismatches", mmPatch);
    PXG_CHECK_MSG(mmContact == 0, "cvxTriFinish: %zu contact-stream mismatches", mmContact);
    PXG_CHECK_MSG(mmFace == 0, "cvxTriFinish: %zu face-index mismatches", mmFace);
    PXG_CHECK_MSG(cOvfP > 0 && cOvfC > 0 && cOvfF > 0 && cZeroMan > 0 && cNoRes > 0 &&
                  cAvg > 0 && cFlip > 0 && cSphere > 0 && cCapsule > 0 && cExact > 0 && cRollbackPatch > 0,
                  "cvxTriFinish: coverage thin (ovfP %zu ovfC %zu ovfF %zu zeroMan %zu noRes %zu "
                  "avg %zu flip %zu sphere %zu capsule %zu exact %zu conflict %zu)",
                  cOvfP, cOvfC, cOvfF, cZeroMan, cNoRes, cAvg, cFlip, cSphere, cCapsule,
                  cExact, cConflict);
}

} // namespace
