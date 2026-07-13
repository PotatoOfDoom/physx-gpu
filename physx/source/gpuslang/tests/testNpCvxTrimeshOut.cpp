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
// tests and its rollback; the insertAveragePoint count inflation that lives
// on lane 0 only (making the faceIndex region base per-lane divergent -- the
// one word where lane 0's slot can collide with another lane's is a genuine
// unordered CUDA write conflict and is masked from comparison); the
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

static float combineScalars(float a, float b, uint32_t mode)
{
    if (mode == 0) return 0.5f * (a + b);
    if (mode == 1) return std::fmin(a, b);
    if (mode == 2) return a * b;
    if (mode == 3) return std::fmax(a, b);
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
        const float fd = std::fmax(d, 0.f);
        dynF = fd; staF = (s - fd) >= 0.f ? s : fd;
        matFlags = cf;
    }
    else { matFlags = cf | 2u; dynF = 0.f; staF = 0.f; }
}

// returns the double-written faceIndex word to mask (0xFFFFFFFF = none)
static uint32_t refTrimeshOut(const TmoIn& s, const std::vector<MaterialData>& mats, TmoOut& o)
{
    o.res = U4{ 0, 0, 0, 0 };
    o.outputs = U4{ NULLS, NULLS, NULLS, 0 };
    for (int j = 0; j < 16; ++j) { o.patches[j] = U4{ 0, 0, 0, 0 }; o.fstream[j] = U4{ 0, 0, 0, 0 }; }
    for (int j = 0; j < 24; ++j) o.cstream[j] = F4{ 0, 0, 0, 0 };

    const uint32_t nbManifolds = s.hdr.x;
    const uint32_t allflags = s.hdr.y;
    const bool insertAvg = s.hdr.z != 0;

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
        total0 += insertAvg ? nbMan0 : 0;
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
            combineMats(mats[mi0], mats[mi1], mf, staF, dynF, rest, damp);
            const V3 pn = flip ? V3{ -wn.x, -wn.y, -wn.z } : wn;
            const uint32_t pb = m * 4;
            o.patches[pb + 0] = U4{ f2u(1.f), f2u(1.f), f2u(1.f), f2u(1.f) };
            o.patches[pb + 1] = U4{ f2u(pn.x), f2u(pn.y), f2u(pn.z), f2u(rest) };
            o.patches[pb + 2] = U4{ f2u(dynF), f2u(staF), f2u(damp),
                                    (wI & 0xFFFF) | ((nc & 0xFF) << 16) | ((mf & 0xFF) << 24) };
            o.patches[pb + 3] = U4{ 1u | ((mi0 & 0xFFFF) << 16), mi1 & 0xFFFF, 0, 0 };
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
            // lane 0 collides with lane word==pc+nbManifolds when insertAvg inflates:
            if (L == 0 && insertAvg && total0 != pc && word >= pc + 1 && word <= pc + pc - 1)
                conflictWord = word;
            if (L != 0 && insertAvg && total0 != pc && (contactMask & 1u) && word == total0)
                conflictWord = word;
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
        s.shapeInfo = U4{ flip ? T_TRIMESH : other, flip ? other : T_TRIMESH, uint32_t(rng() % NMAT), uint32_t(rng() % NMAT) };
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

        const uint32_t total0 = pcExpect + (insertAvg ? nbMan : 0);
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
        if (nbMan > 0 && pcExpect == 0 && !insertAvg) ++cNoRes;
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
                  cCapsule > N / 30 && cConflict > N / 100 && cEmptyMan > N / 10 && cExact > N / 20,
                  "trimeshOut: coverage thin (ovfP %zu ovfC %zu ovfF %zu zeroMan %zu noRes %zu avg %zu flip %zu sph %zu cap %zu conf %zu emptyMan %zu exact %zu)",
                  cOvfP, cOvfC, cOvfF, cZeroMan, cNoRes, cAvg, cFlip, cSphere, cCapsule, cConflict, cEmptyMan, cExact);

    for (PxgBufferHandle h : { bIn, bOut, bMat }) dev->destroyBuffer(h);
}

} // namespace
