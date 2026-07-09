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

// Verification of npManifold.slang: mfRefresh (refreshManifolds) and mfInvalidate
// (invalidateManifold) — the persistent-contact-manifold maintenance helpers.
// The recomputed penetrations are continuous (compared with almostEqual); the
// discrete outputs (the lost-contact / invalidate booleans) are gated on float
// comparisons against thresholds, so cases within MARG of a threshold boundary
// are skipped.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct Contact { F4 pointA, pointB, nPen; };
struct RefIn { F4 aToBp, aToBq, misc; uint32_t nbContacts[4]; };
struct RefOut { uint32_t ret, nbMan, p0, p1; };
struct InvIn { F4 aToBp, aToBq, prevP, prevQ, misc; };
struct MfParams { uint32_t numTests, pad0, pad1, pad2; };

static const uint32_t MAX_SLOTS = 24;
static const uint32_t SUB_MAX = 6;
static const float MARG = 1e-3f;

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float dot4(F4 a, F4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

// quatRotate, byte-identical to pxgSlangCommon
static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = (q.x * vx + q.y * vy + q.z * vz);
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3 transformPoint(F4 p, F4 q, V3 v)
{
    const V3 r = quatRotate(q, v);
    return { p.x + r.x, p.y + r.y, p.z + r.z };
}
static F4 randQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    F4 q; float m;
    do { q = { u(rng), u(rng), u(rng), u(rng) }; m = std::sqrt(dot4(q, q)); } while (m < 1e-3f);
    return { q.x / m, q.y / m, q.z / m, q.w / m };
}

PXG_TEST(npManifold_refresh)
{
    std::mt19937 rng(66011);
    const uint32_t N = 8000;
    std::vector<Contact> contacts(N * MAX_SLOTS);
    std::vector<RefIn> in(N);
    std::vector<float> refPen(N * MAX_SLOTS);
    std::vector<uint8_t> refRet(N), skip(N, 0);
    std::vector<uint32_t> refNbMan(N);

    std::uniform_real_distribution<float> uc(-1.5f, 1.5f);
    std::uniform_real_distribution<float> up(-0.4f, 0.4f);
    std::uniform_real_distribution<float> ubt(0.05f, 0.6f);
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> unm(1, 4);
    std::uniform_int_distribution<uint32_t> unc(0, 6);
    size_t lostC = 0, keptC = 0, skipped = 0;
    auto randUnit3 = [&](std::mt19937& r) { V3 n; float m; do { n = { u11(r), u11(r), u11(r) }; m = std::sqrt(dot3(n, n)); } while (m < 1e-2f); return V3{ n.x / m, n.y / m, n.z / m }; };

    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t numManifolds = unm(rng);
        uint32_t nb[4] = { 0, 0, 0, 0 };
        for (uint32_t m = 0; m < 4; ++m) nb[m] = unc(rng);
        const F4 p = { up(rng), up(rng), up(rng), 0 };
        const F4 q = randQuat(rng);
        const float bt = ubt(rng);
        std::uniform_real_distribution<float> udrift(0.0f, 2.0f * bt); // drift magnitude straddles bt
        // build cached contacts that are geometrically consistent with (p,q) up to a
        // controlled 2D drift, so ~half exceed the breaking threshold
        for (uint32_t i = 0; i < MAX_SLOTS; ++i)
        {
            const V3 normal = randUnit3(rng);
            const float penOld = up(rng);
            const V3 pointA = { uc(rng), uc(rng), uc(rng) };
            const V3 localAInB = transformPoint(p, q, pointA);
            const V3 projPt = { localAInB.x - normal.x * penOld, localAInB.y - normal.y * penOld, localAInB.z - normal.z * penOld };
            const V3 dd = randUnit3(rng);
            const float dm = udrift(rng);
            const V3 pointB = { projPt.x + dd.x * dm, projPt.y + dd.y * dm, projPt.z + dd.z * dm };
            contacts[t * MAX_SLOTS + i] = { { pointA.x, pointA.y, pointA.z, 0 }, { pointB.x, pointB.y, pointB.z, 0 },
                                            { normal.x, normal.y, normal.z, penOld } };
        }
        in[t] = { p, q, { bt, float(numManifolds), 0, 0 }, { nb[0], nb[1], nb[2], nb[3] } };

        // reference
        const float sqThr = bt * bt;
        bool lost = false, doSkip = false;
        for (uint32_t i = 0; i < MAX_SLOTS; ++i)
        {
            const Contact& c = contacts[t * MAX_SLOTS + i];
            float penOut = c.nPen.w;
            const uint32_t mIdx = i / SUB_MAX, cIdx = i % SUB_MAX;
            if (mIdx < numManifolds && cIdx < nb[mIdx])
            {
                const V3 pointA = { c.pointA.x, c.pointA.y, c.pointA.z };
                const V3 pointB = { c.pointB.x, c.pointB.y, c.pointB.z };
                const V3 normal = { c.nPen.x, c.nPen.y, c.nPen.z };
                const float penOld = c.nPen.w;
                const V3 localAInB = transformPoint(p, q, pointA);
                const V3 v = sub3(localAInB, pointB);
                const float penNew = dot3(v, normal);
                const V3 projPt = { localAInB.x - normal.x * penOld, localAInB.y - normal.y * penOld, localAInB.z - normal.z * penOld };
                const V3 projDiff = sub3(pointB, projPt);
                const float distance2d = dot3(projDiff, projDiff);
                penOut = penNew;
                if (std::fabs(distance2d - sqThr) < 1e-4f) doSkip = true; // breaking-threshold boundary
                if (distance2d > sqThr) lost = true;
            }
            refPen[t * MAX_SLOTS + i] = penOut;
        }
        refRet[t] = lost ? 1 : 0;
        refNbMan[t] = lost ? 0 : numManifolds;
        if (lost) ++lostC; else ++keptC;
        if (doSkip) { skip[t] = 1; ++skipped; }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<RefOut> outInit(N);
    std::vector<float> penInit(N * MAX_SLOTS, -999.0f);
    std::vector<InvIn> invDummy(N);
    std::vector<uint32_t> invoDummy(N);
    PxgBufferHandle bC = createBufferFrom(dev, contacts);
    PxgBufferHandle bRi = createBufferFrom(dev, in);
    PxgBufferHandle bRo = createBufferFrom(dev, outInit);
    PxgBufferHandle bP = createBufferFrom(dev, penInit);
    PxgBufferHandle bIi = createBufferFrom(dev, invDummy);
    PxgBufferHandle bIo = createBufferFrom(dev, invoDummy);

    MfParams prm = { N, 0, 0, 0 };
    Kernel k("npManifold.spv", "mfRefresh",
             { (uint32_t)sizeof(Contact), (uint32_t)sizeof(RefIn), (uint32_t)sizeof(RefOut), 4, (uint32_t)sizeof(InvIn), 4 }, sizeof(MfParams));
    k.dispatch(GROUPS, { bC, bRi, bRo, bP, bIi, bIo }, prm);
    dev->flush();

    const auto go = downloadBuffer<RefOut>(dev, bRo, N);
    const auto gp = downloadBuffer<float>(dev, bP, N * MAX_SLOTS);
    size_t mmRet = 0, mmNb = 0, mmPen = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) continue;
        if (go[t].ret != refRet[t]) { ++mmRet; continue; }
        if (go[t].nbMan != refNbMan[t]) { ++mmNb; continue; }
        for (uint32_t i = 0; i < MAX_SLOTS; ++i)
            if (!almostEqual(gp[t * MAX_SLOTS + i], refPen[t * MAX_SLOTS + i], 1e-4f, 1e-4f)) { ++mmPen; break; }
    }
    PXG_CHECK_MSG(mmRet == 0, "mfRefresh: %zu ret mismatches", mmRet);
    PXG_CHECK_MSG(mmNb == 0, "mfRefresh: %zu nbManifolds mismatches", mmNb);
    PXG_CHECK_MSG(mmPen == 0, "mfRefresh: %zu penetration mismatches", mmPen);
    PXG_CHECK_MSG(lostC > 300 && keptC > 300, "mfRefresh: coverage thin (lost=%zu kept=%zu skip=%zu)", lostC, keptC, skipped);

    dev->destroyBuffer(bC); dev->destroyBuffer(bRi); dev->destroyBuffer(bRo); dev->destroyBuffer(bP); dev->destroyBuffer(bIi); dev->destroyBuffer(bIo);
}

PXG_TEST(npManifold_invalidate)
{
    std::mt19937 rng(66029);
    const uint32_t N = 12000;
    std::vector<InvIn> in(N);
    std::vector<uint8_t> refInv(N), skip(N, 0);
    std::uniform_real_distribution<float> up(-0.3f, 0.3f);
    std::uniform_real_distribution<float> umm(0.05f, 0.4f);
    std::uniform_real_distribution<float> ur(0.2f, 0.6f);
    std::uniform_real_distribution<float> uposn(-0.15f, 0.15f); // position perturbation straddles thresholdP
    const float thresholdQ = 0.9998f;
    size_t invC = 0, validC = 0, skipped = 0;

    for (uint32_t t = 0; t < N; ++t)
    {
        const F4 p = { up(rng), up(rng), up(rng), 0 };
        const F4 q = randQuat(rng);
        // prev pose: position perturbed to straddle thresholdP; rotation is either
        // identical (deltaQ==1, Q-branch inactive) or a clearly different rotation
        // (deltaQ well below 0.9998), so the Q boundary is never sat on.
        const F4 pp = { p.x + uposn(rng), p.y + uposn(rng), p.z + uposn(rng), 0 };
        const F4 pq = (rng() % 5u == 0u) ? randQuat(rng) : q;
        const float minMargin = umm(rng), ratio = ur(rng);
        in[t] = { p, q, pp, pq, { minMargin, ratio, 0, 0 } };

        const float thresholdP = minMargin * ratio;
        const float dx = std::fabs(p.x - pp.x), dy = std::fabs(p.y - pp.y), dz = std::fabs(p.z - pp.z);
        const float deltaP = std::max(dx, std::max(dy, dz));
        const float deltaQ = dot4(q, pq);
        if (std::fabs(deltaP - thresholdP) < 1e-4f || std::fabs(deltaQ - thresholdQ) < 1e-4f) { skip[t] = 1; ++skipped; }
        const bool invalid = (deltaP > thresholdP) || (deltaQ < thresholdQ);
        refInv[t] = invalid ? 1 : 0;
        if (invalid) ++invC; else ++validC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<Contact> cDummy(N * MAX_SLOTS);
    std::vector<RefIn> riDummy(N);
    std::vector<RefOut> roDummy(N);
    std::vector<float> pDummy(N * MAX_SLOTS);
    std::vector<uint32_t> outInit(N, 7);
    PxgBufferHandle bC = createBufferFrom(dev, cDummy);
    PxgBufferHandle bRi = createBufferFrom(dev, riDummy);
    PxgBufferHandle bRo = createBufferFrom(dev, roDummy);
    PxgBufferHandle bP = createBufferFrom(dev, pDummy);
    PxgBufferHandle bIi = createBufferFrom(dev, in);
    PxgBufferHandle bIo = createBufferFrom(dev, outInit);

    MfParams prm = { N, 0, 0, 0 };
    Kernel k("npManifold.spv", "mfInvalidate",
             { (uint32_t)sizeof(Contact), (uint32_t)sizeof(RefIn), (uint32_t)sizeof(RefOut), 4, (uint32_t)sizeof(InvIn), 4 }, sizeof(MfParams));
    k.dispatch(GROUPS, { bC, bRi, bRo, bP, bIi, bIo }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bIo, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t) { if (skip[t]) continue; if ((g[t] ? 1 : 0) != refInv[t]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "mfInvalidate: %zu mismatches", mm);
    PXG_CHECK_MSG(invC > N / 4 && validC > N / 4, "mfInvalidate: coverage thin (inv=%zu valid=%zu skip=%zu)", invC, validC, skipped);

    dev->destroyBuffer(bC); dev->destroyBuffer(bRi); dev->destroyBuffer(bRo); dev->destroyBuffer(bP); dev->destroyBuffer(bIi); dev->destroyBuffer(bIo);
}

} // namespace
