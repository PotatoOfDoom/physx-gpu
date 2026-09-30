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

// Verification of npTriPlaneNp.slang (trimeshPlaneNarrowphase,
// trimeshCollision.cu :61-434) against an exact CPU transcription of the
// whole 32-warp block: per-lane inclusive plane test, per-warp
// contactReduce<clustering, keepDeepestPCM, 6, noDupeTest>, the block
// prefix pack behind the carried retainedCount, and the warp-0 readMask
// batching loop with its faithful `ind += popc(readMask)` overshoot.
// 1/64-grid vertices, signed-permutation quaternions and pow2 scales
// keep every decision bit-deterministic (worldNormal is a signed axis,
// so the in-plane projection stays on the grid; length()/sqrt values
// are IEEE-identical on both sides and argmax/argmin tiebreaks are
// transcribed).  Families: invalidate-by-position / by-rotation with
// DRIFTED cached contacts (a refresh-despite-invalidate mutant loses
// the manifold), CACHED (prev == trimeshToPlane exactly, undrifted
// contacts, post-refresh pens reloaded), refresh-LOST (one contact with
// 2D drift), small full-gen with the deepest-accepted vertex EXACTLY on
// cDistance (pins the inclusive gate), all-rejected (zero contacts ->
// poisoned output record), empty-cached (mNbManifolds == 0 seed),
// multi-iteration (> 1024 verts: wide first iteration + clustered
// second pins the retainedCount carry), threshold-window probes
// (deltaP one ulp above / exactly at minMargin*0.1), and a two-manifold
// cached case (refresh write-backs beyond manifold 0, reload from
// manifold 0 only; three seeded manifolds pin the manifold-1/2 refresh
// arms).  A second test pins the contact-stream overflow
// tail: contact-buffer overflow poisons the offset and zeroes
// nbContacts, while a force-buffer-only overflow FAITHFULLY keeps the
// contacts and the count, carrying the poisoned force offset;
// reservations are ungated; single-case runs with limits exactly at /
// one short of the reservation pin the strict > gates and the combined
// overflow flag.
//
// KNOWN LIMITS: the 0.9998 quat gate strictness is unpinnable with
// exact quats (dots land on quarters); manifold 3 refresh write-backs
// are unpinnable because the uint4 meta carries three counts (the
// kernel itself only ever writes manifold 0).

#include "pxgTestFramework.h"

#include <algorithm>
#include <bitset>
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
struct TpCase
{
    F4 meshQ, meshP, planeQ, planeP, scale, scaleRot;
    U4 verts, rootMeta;
};
struct TpParams { uint32_t numCases, contactBytesLimit, forceBytesLimit, pad0; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static float dot4(F4 a, F4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 quatRotateInv(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static F4 quatMul(F4 a, F4 b)
{
    return { a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
             a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
             a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
             a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static F4 quatConj(F4 q) { return { -q.x, -q.y, -q.z, q.w }; }

// P9.94a mutant twin: true selects the port's PRE-FIX trimeshToPlane.p spelling
// q.rotateInv(v) instead of the CUDA's planeTransform.transformInv(trimeshTransform)
// = q.getConjugate().rotate(v) (PxTransform.h:162-169).  Same bits for every
// nonzero component; an exact-zero component can take the other sign.  Used
// only to prove that the directed zero-sign cases below separate the two forms.
static bool gP94TinvRotInv = false;
static V3 transformInvP(F4 q, V3 v)
{
    return gP94TinvRotInv ? quatRotateInv(q, v) : quatRotate(quatConj(q), v);
}
static V3 basis0(F4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return { (q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2 };
}
static V3 vertex2ShapeRef(V3 v, V3 scale, F4 rot)
{
    V3 t = quatRotate(rot, v);
    t.x *= scale.x; t.y *= scale.y; t.z *= scale.z;
    return quatRotateInv(rot, t);
}
static F4 exactQuat(std::mt19937& rng)
{
    if (rng() % 4 == 0) return { 0, 0, 0, 1 };
    const uint32_t s = rng() % 16;
    return { (s & 1) ? -0.5f : 0.5f, (s & 2) ? -0.5f : 0.5f, (s & 4) ? -0.5f : 0.5f, (s & 8) ? -0.5f : 0.5f };
}
static float grid64(std::mt19937& rng, float lo, float hi)
{
    const int n = int((hi - lo) * 64.0f);
    return lo + float(int(rng() % uint32_t(n + 1))) / 64.0f;
}
static uint32_t popc(uint32_t m) { return uint32_t(std::bitset<32>(m).count()); }
static uint32_t ctz(uint32_t m)
{
    for (uint32_t b = 0; b < 32; ++b)
        if ((m >> b) & 1) return b;
    return 32;
}

static const float FLT_BIG = 3.402823466e38f;
static const uint32_t OVERFLOW_CONTACT = 1u, OVERFLOW_FORCE = 2u;

static uint32_t lowestArgMaxRef(const float vals[32], uint32_t mask, float& best)
{
    best = -FLT_BIG;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
static uint32_t lowestArgMinRef(const float vals[32], uint32_t mask, float& best)
{
    best = FLT_BIG;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// contactReduce<true, true, 6, false> on lane arrays (contactReduction.cuh)
static uint32_t contactReduce6Ref(const V3 ptRaw[32], const float sep[32], V3 normal,
                                  uint32_t allMask, float clusterBias)
{
    V3 point[32];
    for (uint32_t j = 0; j < 32; ++j)
        point[j] = sub(ptRaw[j], mul(normal, dot3(ptRaw[j], normal)));
    if (popc(allMask) <= 6u)
        return allMask;

    float v;
    float vals[32];
    for (uint32_t j = 0; j < 32; ++j) vals[j] = sep[j];
    const uint32_t i0 = lowestArgMaxRef(vals, allMask, v);
    uint32_t mask = 1u << i0;
    for (uint32_t j = 0; j < 32; ++j)
    {
        const V3 d = sub(point[i0], point[j]);
        vals[j] = std::sqrt(dot3(d, d));
    }
    const uint32_t i1 = lowestArgMaxRef(vals, allMask & ~mask, v);
    mask |= 1u << i1;
    const V3 dir = cross3(normal, sub(point[i1], point[i0]));
    float dval[32];
    for (uint32_t j = 0; j < 32; ++j) dval[j] = dot3(dir, sub(point[j], point[i0]));
    const uint32_t f = lowestArgMaxRef(dval, allMask & ~mask, v);
    mask |= 1u << f;
    const uint32_t g = lowestArgMinRef(dval, allMask & ~mask, v);
    mask |= 1u << g;
    if (popc(mask) == 4)
    {
        const uint32_t i4 = lowestArgMinRef(sep, allMask & ~mask, v);
        mask |= 1u << i4;
    }
    float sepB[32];
    for (uint32_t j = 0; j < 32; ++j)
        sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);
    int label[32];
    int nbClusters = 0;
    {
        float tbest[32];
        for (uint32_t j = 0; j < 32; ++j) { tbest[j] = FLT_BIG; label[j] = -1; }
        for (uint32_t m = mask; m != 0; ++nbClusters, m &= (m - 1))
        {
            const uint32_t anchor = ctz(m);
            for (uint32_t j = 0; j < 32; ++j)
            {
                const V3 dv = sub(point[j], point[anchor]);
                const float d = dot3(dv, dv);
                if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
            }
        }
    }
    mask = 0;
    for (int i = 0; i < nbClusters; ++i)
    {
        uint32_t cluster = 0;
        for (uint32_t j = 0; j < 32; ++j)
            if (label[j] == i) cluster |= 1u << j;
        cluster &= allMask;
        if (cluster != 0)
            mask |= 1u << lowestArgMinRef(sepB, cluster, v);
    }
    for (int i = nbClusters; i < 6; ++i)
        mask |= 1u << lowestArgMinRef(sepB, allMask & ~mask, v);
    return mask;
}

struct TpRef
{
    bool fullGen = false, lost = false, reduced = false;
    uint32_t retained = 0, iters = 0;
    V3 pts[6];
    float seps[6];
    uint32_t manX = 0;
    float penOut[24]; // NaN = untouched by the refresh
    F4 ttpQ{};
    V3 ttpP{};
    V3 worldNormal{};
    float thresholdP = 0;
};

static TpRef refTp(const TpCase& cs, const std::vector<F4>& verts, const std::vector<F4>& root,
                   const F4* mA, const F4* mB, const F4* mN, U4 meta, F4 prevQ, V3 prevP)
{
    TpRef ro;
    for (int s = 0; s < 24; ++s) ro.penOut[s] = std::nanf("");
    const float tol = cs.meshP.w, cd = cs.planeP.w, clusterTol = cs.scale.w;

    V3 mn = { FLT_BIG, FLT_BIG, FLT_BIG }, mx = { -FLT_BIG, -FLT_BIG, -FLT_BIG };
    for (uint32_t s = 0; s < cs.rootMeta.x; ++s)
    {
        const V3 bmn = v3(root[(cs.rootMeta.y + s) * 2 + 0]);
        const V3 bmx = v3(root[(cs.rootMeta.y + s) * 2 + 1]);
        mn = { std::min(mn.x, bmn.x), std::min(mn.y, bmn.y), std::min(mn.z, bmn.z) };
        mx = { std::max(mx.x, bmx.x), std::max(mx.y, bmx.y), std::max(mx.z, bmx.z) };
    }
    const V3 ext = mul(sub(mx, mn), 0.5f);
    const V3 es = { ext.x * cs.scale.x, ext.y * cs.scale.y, ext.z * cs.scale.z };
    const float mnE = std::min(std::min(es.x, es.y), es.z);
    const float minMargin = std::min(mnE * 0.25f, tol * 0.05f) * 0.5f;

    ro.ttpQ = quatMul(quatConj(cs.planeQ), cs.meshQ);
    // trimeshCollision.cu:147 planeTransform.transformInv(trimeshTransform) (P9.94a: literal conj().rotate)
    ro.ttpP = transformInvP(cs.planeQ, sub(v3(cs.meshP), v3(cs.planeP)));
    ro.thresholdP = minMargin * 0.1f;
    const V3 dP = sub(ro.ttpP, prevP);
    const float deltaP = std::max(std::fabs(dP.x), std::max(std::fabs(dP.y), std::fabs(dP.z)));
    const float deltaQ = dot4(ro.ttpQ, prevQ);
    const bool invalidate = (deltaP > ro.thresholdP) || (deltaQ < 0.9998f);

    uint32_t manX = meta.x;
    if (!invalidate)
    {
        const float thr = minMargin * 0.5f;
        const float sqThr = thr * thr;
        for (uint32_t s = 0; s < 24; ++s)
        {
            const uint32_t mi = s / 6;
            if (mi >= manX) continue;
            const uint32_t nb = mi == 0 ? meta.y : mi == 1 ? meta.z : mi == 2 ? meta.w : 0;
            if (s % 6 >= nb) continue;
            const V3 pointA = v3(mA[s]), pointB = v3(mB[s]);
            const V3 nrm = v3(mN[s]);
            const float penOld = mN[s].w;
            const V3 lab = add(quatRotate(ro.ttpQ, pointA), ro.ttpP);
            const V3 v = sub(lab, pointB);
            const float penNew = dot3(v, nrm);
            const V3 proj = sub(lab, mul(nrm, penOld));
            const V3 pd = sub(pointB, proj);
            ro.penOut[s] = penNew;
            if (dot3(pd, pd) > sqThr) ro.lost = true;
        }
        if (ro.lost) manX = 0;
    }
    ro.fullGen = invalidate || ro.lost;
    ro.worldNormal = basis0(cs.planeQ);

    V3 contacts[198];
    float seps[198];
    uint32_t retained = 0;
    if (ro.fullGen)
    {
        const uint32_t numVerts = cs.verts.y;
        ro.iters = (numVerts + 1023u) / 1024u;
        for (uint32_t iter = 0; iter < ro.iters; ++iter)
        {
            V3 wPts[32][6];
            float wSeps[32][6];
            uint32_t counters[32];
            for (uint32_t w = 0; w < 32; ++w)
            {
                V3 lanePt[32];
                float laneSep[32];
                uint32_t mask = 0;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    lanePt[l] = { 0, 0, 0 };
                    laneSep[l] = FLT_BIG;
                    const uint32_t i = iter * 1024u + w * 32u + l;
                    if (i >= numVerts) continue;
                    const V3 tp = v3(verts[cs.verts.x + i]);
                    const V3 p = vertex2ShapeRef(tp, v3(cs.scale), cs.scaleRot);
                    const V3 pip = add(quatRotate(ro.ttpQ, p), ro.ttpP);
                    if (cd >= pip.x)
                    {
                        laneSep[l] = pip.x;
                        lanePt[l] = add(quatRotate(cs.meshQ, p), v3(cs.meshP));
                        mask |= 1u << l;
                    }
                }
                if (popc(mask) > 6u) ro.reduced = true;
                mask = contactReduce6Ref(lanePt, laneSep, ro.worldNormal, mask, clusterTol);
                counters[w] = popc(mask);
                uint32_t k = 0;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    if (((mask >> l) & 1) == 0) continue;
                    wPts[w][k] = lanePt[l];
                    wSeps[w][k] = laneSep[l];
                    ++k;
                }
            }
            uint32_t offset = 0;
            for (uint32_t w = 0; w < 32; ++w)
            {
                for (uint32_t k = 0; k < counters[w]; ++k)
                {
                    contacts[retained + offset + k] = wPts[w][k];
                    seps[retained + offset + k] = wSeps[w][k];
                }
                offset += counters[w];
            }
            const uint32_t total = offset + retained;
            V3 lanePt[32];
            float laneSep[32];
            bool has[32];
            for (uint32_t l = 0; l < 32; ++l) { has[l] = false; lanePt[l] = { 0, 0, 0 }; laneSep[l] = FLT_BIG; }
            for (uint32_t ind = 0; ind < total;)
            {
                uint32_t readMask = 0;
                for (uint32_t l = 0; l < 32; ++l)
                    if (!has[l]) readMask |= 1u << l;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    if (has[l]) continue;
                    const uint32_t ri = popc(readMask & ((1u << l) - 1u)) + ind;
                    if (ri < total)
                    {
                        lanePt[l] = contacts[ri];
                        laneSep[l] = seps[ri];
                        has[l] = true;
                    }
                }
                uint32_t mask = 0;
                for (uint32_t l = 0; l < 32; ++l)
                    if (has[l]) mask |= 1u << l;
                ind += popc(readMask);
                if (popc(mask) > 6u) ro.reduced = true;
                mask = contactReduce6Ref(lanePt, laneSep, ro.worldNormal, mask, clusterTol);
                for (uint32_t l = 0; l < 32; ++l)
                    has[l] = ((mask >> l) & 1) != 0;
            }
            uint32_t k2 = 0;
            for (uint32_t l = 0; l < 32; ++l)
            {
                if (!has[l]) continue;
                contacts[k2] = lanePt[l];
                seps[k2] = laneSep[l];
                ++k2;
            }
            retained = k2;
        }
        manX = retained != 0 ? 1u : 0u;
    }
    else if (manX != 0)
    {
        for (uint32_t k = 0; k < meta.y; ++k)
        {
            contacts[k] = add(quatRotate(cs.meshQ, v3(mA[k])), v3(cs.meshP));
            seps[k] = ro.penOut[k];
        }
        retained = meta.y;
    }
    ro.manX = manX;
    ro.retained = retained;
    for (uint32_t k = 0; k < retained && k < 6; ++k)
    {
        ro.pts[k] = contacts[k];
        ro.seps[k] = seps[k];
    }
    return ro;
}

struct TpRun
{
    std::vector<F4> manA, manB, manN, rel, dbg, stream;
    std::vector<U4> meta, rec;
    uint32_t counters[3];
};
static TpRun runTp(const std::vector<TpCase>& cases, const std::vector<F4>& verts,
                   const std::vector<F4>& root, const std::vector<F4>& manA,
                   const std::vector<F4>& manB, const std::vector<F4>& manN,
                   const std::vector<U4>& meta, const std::vector<F4>& rel,
                   uint32_t streamSlots, uint32_t contactLimit, uint32_t forceLimit)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = (uint32_t)cases.size();
    std::vector<uint32_t> ctrInit = { 0, 0, 0 };
    std::vector<F4> streamInit(streamSlots, { -777.0f, -777.0f, -777.0f, -777.0f });
    std::vector<U4> recInit(n, { 0xEDEDEDEDu, 0xEDEDEDEDu, 0xEDEDEDEDu, 0xEDEDEDEDu });
    std::vector<F4> dbgInit(n, { -7, -7, -7, -7 });

    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bR = createBufferFrom(dev, root);
    PxgBufferHandle bA = createBufferFrom(dev, manA);
    PxgBufferHandle bB = createBufferFrom(dev, manB);
    PxgBufferHandle bN = createBufferFrom(dev, manN);
    PxgBufferHandle bM = createBufferFrom(dev, meta);
    PxgBufferHandle bRel = createBufferFrom(dev, rel);
    PxgBufferHandle bCt = createBufferFrom(dev, ctrInit);
    PxgBufferHandle bS = createBufferFrom(dev, streamInit);
    PxgBufferHandle bRec = createBufferFrom(dev, recInit);
    PxgBufferHandle bD = createBufferFrom(dev, dbgInit);
    TpParams prm = { n, contactLimit, forceLimit, 0 };
    Kernel k("npTriPlaneNp.spv", "tpNarrowphase",
             { (uint32_t)sizeof(TpCase), 16, 16, 16, 16, 16, 16, 16, 4, 16, 16, 16 },
             sizeof(TpParams));
    k.dispatch((n + 63) / 64, { bC, bV, bR, bA, bB, bN, bM, bRel, bCt, bS, bRec, bD }, prm);
    dev->flush();
    TpRun r;
    r.manA = downloadBuffer<F4>(dev, bA, (uint32_t)manA.size());
    r.manB = downloadBuffer<F4>(dev, bB, (uint32_t)manB.size());
    r.manN = downloadBuffer<F4>(dev, bN, (uint32_t)manN.size());
    r.meta = downloadBuffer<U4>(dev, bM, n);
    r.rel = downloadBuffer<F4>(dev, bRel, 2 * n);
    const auto ctr = downloadBuffer<uint32_t>(dev, bCt, 3);
    r.counters[0] = ctr[0]; r.counters[1] = ctr[1]; r.counters[2] = ctr[2];
    r.stream = downloadBuffer<F4>(dev, bS, streamSlots);
    r.rec = downloadBuffer<U4>(dev, bRec, n);
    r.dbg = downloadBuffer<F4>(dev, bD, n);
    for (PxgBufferHandle h : { bC, bV, bR, bA, bB, bN, bM, bRel, bCt, bS, bRec, bD })
        dev->destroyBuffer(h);
    return r;
}

struct TpFixture
{
    std::vector<TpCase> cases;
    std::vector<F4> verts, root, manA, manB, manN, rel;
    std::vector<U4> meta;
    std::vector<TpRef> ref;
};

// family = i % 8: 0 invalidate-pos, 1 invalidate-rot (i%32==9 threshold-over
// probe, i%32==25 threshold-equal probe -> cached), 2/5 cached (i%16==5 two
// manifolds, i%16==13 empty), 3 refresh-lost, 4 small full gen with the
// deepest vertex EXACTLY on cDistance, 6 all-rejected, 7 generic full gen
// (i%16==15 multi-iteration)
static TpFixture buildFixture(uint32_t NC, uint32_t seed)
{
    TpFixture fx;
    std::mt19937 rng(seed);
    fx.cases.resize(NC);
    fx.manA.resize(NC * 24);
    fx.manB.resize(NC * 24);
    fx.manN.resize(NC * 24);
    fx.meta.resize(NC);
    fx.rel.resize(NC * 2);
    fx.ref.resize(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        TpCase& cs = fx.cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const uint32_t fam = i % 8;
        const float tol = 0.5f * float(1 + i % 3);
        cs.meshQ = exactQuat(rng);
        cs.meshP = { grid64(rng, -1, 1), grid64(rng, -1, 1), grid64(rng, -1, 1), tol };
        cs.planeQ = exactQuat(rng);
        const float sc[3] = { 0.5f, 1.0f, 2.0f };
        cs.scale = { sc[rng() % 3], sc[rng() % 3], sc[rng() % 3], 0.0625f };
        cs.scaleRot = exactQuat(rng);

        uint32_t numVerts = 40 + i % 50;
        if (fam == 4) numVerts = 3 + i % 4;
        if (fam == 6) numVerts = 10;
        if (i % 16 == 15) numVerts = (i == 31) ? 2112 : 1200;
        const bool probe = (i % 32 == 9) || (i % 32 == 25);
        // probes keep retained (4) below the seeded count (6) so refresh
        // write-backs on slots 4..5 stay visible after a full gen
        if (probe) numVerts = 4;

        // vertex pool with 2 poison pads before each range (off-by-one reads
        // land on a distinctively deep vertex and change anchor picks)
        fx.verts.push_back({ 99, 99, 99, 0 });
        fx.verts.push_back({ 99, 99, 99, 0 });
        cs.verts = { (uint32_t)fx.verts.size(), numVerts, 0, 0 };
        for (uint32_t k = 0; k < numVerts; ++k)
        {
            const bool cluster = (i % 16 == 15) && k >= 1024;
            const float lo = cluster ? -0.25f : -2.0f, hi = cluster ? 0.25f : 2.0f;
            fx.verts.push_back({ grid64(rng, lo, hi), grid64(rng, lo, hi), grid64(rng, lo, hi), 0 });
        }

        // root bounds pool (one pad slot); nNodes in 1..32
        fx.root.push_back({ 88, 88, 88, 0 });
        fx.root.push_back({ 88, 88, 88, 0 });
        const uint32_t nNodes = 1 + (i * 7) % 32;
        cs.rootMeta = { nNodes, (uint32_t)(fx.root.size() / 2), 0, 0 };
        for (uint32_t s = 0; s < nNodes; ++s)
        {
            const V3 bmn = { grid64(rng, -3, -1), grid64(rng, -3, -1), grid64(rng, -3, -1) };
            const V3 bmx = add(bmn, V3{ 0.5f + grid64(rng, 0, 2), 0.5f + grid64(rng, 0, 2), 0.5f + grid64(rng, 0, 2) });
            fx.root.push_back({ bmn.x, bmn.y, bmn.z, 0 });
            fx.root.push_back({ bmx.x, bmx.y, bmx.z, 0 });
        }

        // plane placement drives acceptance deterministically: shifting planeP
        // by +K*basis0 lowers every plane-space x by K
        const V3 b0q = basis0(cs.planeQ);
        V3 pp = { grid64(rng, -1, 1), grid64(rng, -1, 1), grid64(rng, -1, 1) };
        float cd = 0.25f;
        if (probe)
        {
            // zero trimeshToPlane.p().x (basis0 is a signed axis: match the
            // corresponding planeP component to meshP) so the probe deltas
            // below are EXACT float subtractions against thresholdP
            if (b0q.x != 0.0f) pp.x = cs.meshP.x;
            else if (b0q.y != 0.0f) pp.y = cs.meshP.y;
            else pp.z = cs.meshP.z;
        }
        else if (fam == 0 || fam == 1 || fam == 3 || fam == 7)
            pp = add(pp, mul(b0q, 16.0f)); // all verts accepted
        else if (fam == 6)
            pp = sub(pp, mul(b0q, 16.0f)); // all verts rejected
        cs.planeP = { pp.x, pp.y, pp.z, cd };

        // trimeshToPlane (exact) for seeds and probes
        const F4 ttpQ = quatMul(quatConj(cs.planeQ), cs.meshQ);
        const V3 ttpP = transformInvP(cs.planeQ, sub(v3(cs.meshP), v3(cs.planeP))); // P9.94a

        if (fam == 4)
        {
            // cd = the exact max plane-space x over the verts: everything is
            // accepted with the shallowest vertex EXACTLY on the gate
            float mxX = -FLT_BIG;
            for (uint32_t k = 0; k < numVerts; ++k)
            {
                const V3 p = vertex2ShapeRef(v3(fx.verts[cs.verts.x + k]), v3(cs.scale), cs.scaleRot);
                mxX = std::max(mxX, add(quatRotate(ttpQ, p), ttpP).x);
            }
            cs.planeP.w = mxX;
        }

        // margin threshold for the window probes
        V3 mnB = { FLT_BIG, FLT_BIG, FLT_BIG }, mxB = { -FLT_BIG, -FLT_BIG, -FLT_BIG };
        for (uint32_t s = 0; s < nNodes; ++s)
        {
            const V3 a = v3(fx.root[(cs.rootMeta.y + s) * 2 + 0]);
            const V3 b = v3(fx.root[(cs.rootMeta.y + s) * 2 + 1]);
            mnB = { std::min(mnB.x, a.x), std::min(mnB.y, a.y), std::min(mnB.z, a.z) };
            mxB = { std::max(mxB.x, b.x), std::max(mxB.y, b.y), std::max(mxB.z, b.z) };
        }
        const V3 extB = mul(sub(mxB, mnB), 0.5f);
        const float mnE = std::min(std::min(extB.x * cs.scale.x, extB.y * cs.scale.y), extB.z * cs.scale.z);
        const float thresholdP = std::min(mnE * 0.25f, tol * 0.05f) * 0.5f * 0.1f;

        // previous relative transform (gManRel seed)
        F4 prevQ = ttpQ;
        V3 prevP = ttpP;
        bool cachedIntent = false;
        switch (fam)
        {
        case 0: case 4: case 6: case 7:
            prevP.x += 4.0f;
            break;
        case 1:
            if (i % 32 == 9)
            {
                // ttpP.x == 0: one ulp ABOVE the threshold, below 2x of it
                // -> invalidate under the correct margin only
                prevP.x = -std::nextafterf(thresholdP, 2.0f);
            }
            else if (i % 32 == 25)
            {
                // ttpP.x == 0: deltaP == thresholdP EXACTLY -> the strict >
                // keeps the cache; a >= mutant invalidates
                prevP.x = -thresholdP;
                cachedIntent = true;
            }
            else
            {
                F4 pq = exactQuat(rng);
                while (std::fabs(dot4(pq, ttpQ)) > 0.9f) pq = exactQuat(rng);
                prevQ = pq;
            }
            break;
        default: // 2, 3, 5: prev == current exactly
            cachedIntent = fam != 3;
            break;
        }
        fx.rel[i * 2 + 0] = prevQ;
        fx.rel[i * 2 + 1] = { prevP.x, prevP.y, prevP.z, 5.0f };

        // manifold seeds: nPen normal (1,0,0) as the plane kernel writes it;
        // pointB = localAInB - n*penOld + n*t keeps 2D drift exactly 0 while
        // penNew = penOld - t pins the refresh write-back
        const uint32_t seedNb = probe ? 6 : 3 + i % 4;
        U4 meta = { 1, seedNb, 0, 0 };
        // three manifolds pin the mi==1 AND mi==2 refresh arms (write-backs
        // on slots 6..7 and 12..15) plus the mi >= nbManifolds cut at 3
        if (i % 16 == 5) meta = { 3, seedNb, 2, 4 };
        if (i % 16 == 13) meta = { 0, seedNb, 0, 0 };
        fx.meta[i] = meta;
        // t must stay below the breaking threshold (>= minMargin*0.5 >=
        // 0.00625 for tol 0.5): 1/512 keeps pd*pd = t*t well under sqThr
        const float penOld = 0.25f, t = 0.001953125f;
        // i%16==2 (fam 2): pure-NORMAL drift far beyond the breaking
        // threshold -> LOST; a mutant projecting with penNew instead of
        // penOld sees zero drift and stays cached
        const float tEff = (i % 16 == 2) ? 0.0625f : t;
        for (uint32_t s = 0; s < 24; ++s)
        {
            const V3 pA = { grid64(rng, -1, 1), grid64(rng, -1, 1), grid64(rng, -1, 1) };
            const V3 lab = add(quatRotate(ttpQ, pA), ttpP);
            V3 pB = { lab.x - (penOld - tEff), lab.y, lab.z };
            if ((fam == 0 || fam == 1) && !cachedIntent)
                pB.y += 3.0f; // invalidate families carry DRIFTED contacts
            if (fam == 3 && s == 1)
                pB.y += 5.0f; // 2D drift past any breaking threshold -> lost
            fx.manA[i * 24 + s] = { pA.x, pA.y, pA.z, 3.0f };
            fx.manB[i * 24 + s] = { pB.x, pB.y, pB.z, 0 };
            fx.manN[i * 24 + s] = { 1, 0, 0, penOld };
        }
    }
    for (uint32_t i = 0; i < NC; ++i)
        fx.ref[i] = refTp(fx.cases[i], fx.verts, fx.root, &fx.manA[i * 24], &fx.manB[i * 24],
                          &fx.manN[i * 24], fx.meta[i], fx.rel[i * 2], v3(fx.rel[i * 2 + 1]));
    return fx;
}

static uint32_t asU32(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

PXG_TEST(npTriPlaneNp_narrowphase)
{
    const uint32_t NC = 256;
    TpFixture fx = buildFixture(NC, 1030701);

    uint32_t totalContacts = 0;
    size_t nFullGen = 0, nCached = 0, nEmptyCached = 0, nLost = 0, nZero = 0, nReduce = 0, nMulti = 0,
           nProbeOver = 0, nProbeEq = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const TpRef& ro = fx.ref[i];
        totalContacts += ro.retained;
        if (ro.fullGen) ++nFullGen; else if (ro.retained > 0) ++nCached; else ++nEmptyCached;
        if (ro.lost) ++nLost;
        if (ro.fullGen && ro.retained == 0) ++nZero;
        if (ro.reduced) ++nReduce;
        if (ro.iters > 1) ++nMulti;
        if (i % 32 == 9 && ro.fullGen) ++nProbeOver;
        if (i % 32 == 25 && !ro.fullGen && ro.retained > 0) ++nProbeEq;
    }
    PXG_CHECK_MSG(nFullGen > 120 && nCached > 30 && nEmptyCached > 10 && nLost > 20 && nZero > 20 &&
                  nReduce > 100 && nMulti > 10 && nProbeOver == 8 && nProbeEq == 8,
                  "triPlane coverage thin (%zu %zu %zu %zu %zu %zu %zu %zu %zu)",
                  nFullGen, nCached, nEmptyCached, nLost, nZero, nReduce, nMulti, nProbeOver, nProbeEq);

    const uint32_t streamSlots = totalContacts + 32;
    const TpRun r = runTp(fx.cases, fx.verts, fx.root, fx.manA, fx.manB, fx.manN, fx.meta, fx.rel,
                          streamSlots, 0x10000000u, 0x10000000u);

    PXG_CHECK_MSG(r.counters[0] == 16 * totalContacts && r.counters[1] == 4 * totalContacts &&
                  r.counters[2] == 0,
                  "triPlane counters %u/%u/%u (want %u/%u/0)", r.counters[0], r.counters[1],
                  r.counters[2], 16 * totalContacts, 4 * totalContacts);

    size_t mm = 0;
    std::vector<std::pair<uint32_t, uint32_t>> rangesC, rangesF;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const TpRef& ro = fx.ref[i];
        const U4 rec = r.rec[i];
        if (rec.w != (ro.fullGen ? 1u : 0u)) ++mm;
        if (ro.retained == 0)
        {
            if (rec.x != 0xFFFFFFFFu || rec.y != 0xFFFFFFFFu || rec.z != 0) ++mm;
        }
        else
        {
            if (rec.z != ro.retained || rec.x % 16 != 0 || rec.y % 4 != 0 ||
                rec.x + 16 * ro.retained > 16 * totalContacts) { ++mm; continue; }
            rangesC.push_back({ rec.x, 16 * ro.retained });
            rangesF.push_back({ rec.y, 4 * ro.retained });
            for (uint32_t k = 0; k < ro.retained; ++k)
            {
                const F4 got = r.stream[rec.x / 16 + k];
                if (got.x != ro.pts[k].x || got.y != ro.pts[k].y || got.z != ro.pts[k].z ||
                    got.w != ro.seps[k]) ++mm;
            }
        }
        // manifold meta: full gen writes x and y only; other lanes keep seeds
        const U4 ms = fx.meta[i], mo = r.meta[i];
        if (ro.fullGen && ro.retained > 0)
        {
            if (mo.x != 1 || mo.y != ro.retained || mo.z != ms.z || mo.w != ms.w) ++mm;
        }
        else if (mo.x != ro.manX || mo.y != ms.y || mo.z != ms.z || mo.w != ms.w) ++mm;
        // manifold contacts
        for (uint32_t s = 0; s < 24; ++s)
        {
            const F4 gA = r.manA[i * 24 + s], gB = r.manB[i * 24 + s], gN = r.manN[i * 24 + s];
            const F4 sA = fx.manA[i * 24 + s], sB = fx.manB[i * 24 + s], sN = fx.manN[i * 24 + s];
            if (ro.fullGen && ro.retained > 0 && s < ro.retained)
            {
                const V3 eA = quatRotateInv(fx.cases[i].meshQ, sub(ro.pts[s], v3(fx.cases[i].meshP)));
                V3 eB = quatRotateInv(fx.cases[i].planeQ, sub(ro.pts[s], v3(fx.cases[i].planeP)));
                eB.x = 0.0f;
                if (gA.x != eA.x || gA.y != eA.y || gA.z != eA.z || asU32(gA.w) != 0xFFFFFFFFu) ++mm;
                if (gB.x != eB.x || gB.y != eB.y || gB.z != eB.z || gB.w != 0.0f) ++mm;
                if (gN.x != 1.0f || gN.y != 0.0f || gN.z != 0.0f || gN.w != ro.seps[s]) ++mm;
            }
            else
            {
                if (std::memcmp(&gA, &sA, 16) != 0 || std::memcmp(&gB, &sB, 16) != 0) ++mm;
                if (gN.x != sN.x || gN.y != sN.y || gN.z != sN.z) ++mm;
                const float expPen = std::isnan(ro.penOut[s]) ? sN.w : ro.penOut[s];
                if (gN.w != expPen) ++mm;
            }
        }
        // relative transform: written only on full gen with retained contacts
        // (P9.94a: BITWISE, so a zero-sign slip in trimeshToPlane is visible)
        const F4 rq = r.rel[i * 2 + 0], rp = r.rel[i * 2 + 1];
        if (ro.fullGen && ro.retained > 0)
        {
            if (asU32(rq.x) != asU32(ro.ttpQ.x) || asU32(rq.y) != asU32(ro.ttpQ.y) ||
                asU32(rq.z) != asU32(ro.ttpQ.z) || asU32(rq.w) != asU32(ro.ttpQ.w) ||
                asU32(rp.x) != asU32(ro.ttpP.x) || asU32(rp.y) != asU32(ro.ttpP.y) ||
                asU32(rp.z) != asU32(ro.ttpP.z) || rp.w != 0.0f) ++mm;
        }
        else if (std::memcmp(&rq, &fx.rel[i * 2 + 0], 16) != 0 ||
                 std::memcmp(&rp, &fx.rel[i * 2 + 1], 16) != 0) ++mm;
        // debug: worldNormal + retainedCount
        if (r.dbg[i].x != ro.worldNormal.x || r.dbg[i].y != ro.worldNormal.y ||
            r.dbg[i].z != ro.worldNormal.z || r.dbg[i].w != float(ro.retained)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "triPlane: %zu mismatches", mm);

    // reservation ranges partition [0, bytes) exactly (contacts and forces)
    auto checkPartition = [](std::vector<std::pair<uint32_t, uint32_t>>& v, uint32_t total, const char* what)
    {
        std::sort(v.begin(), v.end());
        uint32_t cursor = 0;
        size_t bad = 0;
        for (const auto& pr : v)
        {
            if (pr.first != cursor) ++bad;
            cursor = pr.first + pr.second;
        }
        if (cursor != total) ++bad;
        PXG_CHECK_MSG(bad == 0, "triPlane: %s ranges do not partition [0, %u)", what, total);
    };
    checkPartition(rangesC, 16 * totalContacts, "contact");
    checkPartition(rangesF, 4 * totalContacts, "force");

    size_t dirty = 0;
    for (uint32_t s = totalContacts; s < streamSlots; ++s)
        if (r.stream[s].x != -777.0f) ++dirty;
    PXG_CHECK_MSG(dirty == 0, "triPlane: %zu dirty stream slots beyond total", dirty);
}

PXG_TEST(npTriPlaneNp_overflow)
{
    // all-full-gen fleet; overflow of the CONTACT buffer poisons the pair
    // (offset 0xFFFFFFFF, nbContacts zeroed, nothing written); overflow of
    // the FORCE buffer alone keeps contacts and count (faithful quirk)
    const uint32_t NC = 24;
    TpFixture fx = buildFixture(NC, 1030702);
    // force every case into full gen with contacts: rebuild prev far for all
    for (uint32_t i = 0; i < NC; ++i)
    {
        fx.rel[i * 2 + 1].x += 8.0f;
        const V3 b0q = basis0(fx.cases[i].planeQ);
        V3 pp = v3(fx.cases[i].planeP);
        pp = add(pp, mul(b0q, 32.0f)); // all verts accepted regardless of family shifts
        fx.cases[i].planeP = { pp.x, pp.y, pp.z, 0.25f };
    }
    for (uint32_t i = 0; i < NC; ++i)
        fx.ref[i] = refTp(fx.cases[i], fx.verts, fx.root, &fx.manA[i * 24], &fx.manB[i * 24],
                          &fx.manN[i * 24], fx.meta[i], fx.rel[i * 2], v3(fx.rel[i * 2 + 1]));
    uint32_t total = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        PXG_CHECK_MSG(fx.ref[i].fullGen && fx.ref[i].retained > 0, "overflow fixture: case %u empty", i);
        total += fx.ref[i].retained;
    }
    const uint32_t streamSlots = total + 16;

    // run A: contact buffer holds roughly half the fleet
    {
        const uint32_t limit = 16 * (total / 2);
        const TpRun r = runTp(fx.cases, fx.verts, fx.root, fx.manA, fx.manB, fx.manN, fx.meta, fx.rel,
                              streamSlots, limit, 0x10000000u);
        PXG_CHECK_MSG(r.counters[0] == 16 * total && r.counters[2] == OVERFLOW_CONTACT,
                      "overflow A: counters %u/%u", r.counters[0], r.counters[2]);
        size_t bad = 0, nOk = 0, nPoison = 0;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (uint32_t i = 0; i < NC; ++i)
        {
            const TpRef& ro = fx.ref[i];
            const U4 rec = r.rec[i];
            if (rec.x == 0xFFFFFFFFu)
            {
                ++nPoison;
                if (rec.y != 0xFFFFFFFFu || rec.z != 0) ++bad;
            }
            else
            {
                ++nOk;
                if (rec.x + 16 * ro.retained > limit || rec.z != ro.retained) { ++bad; continue; }
                ranges.push_back({ rec.x, 16 * ro.retained });
                for (uint32_t k = 0; k < ro.retained; ++k)
                {
                    const F4 got = r.stream[rec.x / 16 + k];
                    if (got.x != ro.pts[k].x || got.y != ro.pts[k].y || got.z != ro.pts[k].z ||
                        got.w != ro.seps[k]) ++bad;
                }
            }
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t j = 1; j < ranges.size(); ++j)
            if (ranges[j].first < ranges[j - 1].first + ranges[j - 1].second) ++bad;
        PXG_CHECK_MSG(bad == 0 && nOk > 0 && nPoison > 0,
                      "overflow A: %zu bad, %zu ok, %zu poisoned", bad, nOk, nPoison);
    }

    // run B: force buffer alone overflows -> contacts and count survive
    {
        const uint32_t limit = 4 * (total / 2);
        const TpRun r = runTp(fx.cases, fx.verts, fx.root, fx.manA, fx.manB, fx.manN, fx.meta, fx.rel,
                              streamSlots, 0x10000000u, limit);
        PXG_CHECK_MSG(r.counters[1] == 4 * total && r.counters[2] == OVERFLOW_FORCE,
                      "overflow B: counters %u/%u", r.counters[1], r.counters[2]);
        size_t bad = 0, nPoisonF = 0;
        for (uint32_t i = 0; i < NC; ++i)
        {
            const TpRef& ro = fx.ref[i];
            const U4 rec = r.rec[i];
            if (rec.x == 0xFFFFFFFFu || rec.z != ro.retained) { ++bad; continue; }
            if (rec.y == 0xFFFFFFFFu) ++nPoisonF;
            else if (rec.y + 4 * ro.retained > limit) ++bad;
            for (uint32_t k = 0; k < ro.retained; ++k)
            {
                const F4 got = r.stream[rec.x / 16 + k];
                if (got.x != ro.pts[k].x || got.y != ro.pts[k].y || got.z != ro.pts[k].z ||
                    got.w != ro.seps[k]) ++bad;
            }
        }
        PXG_CHECK_MSG(bad == 0 && nPoisonF > 0, "overflow B: %zu bad, %zu force-poisoned", bad, nPoisonF);
    }

    // runs C/D: single case -> deterministic offsets pin the STRICT > gates
    {
        const std::vector<TpCase> one = { fx.cases[0] };
        const std::vector<F4> mA(fx.manA.begin(), fx.manA.begin() + 24);
        const std::vector<F4> mB(fx.manB.begin(), fx.manB.begin() + 24);
        const std::vector<F4> mN(fx.manN.begin(), fx.manN.begin() + 24);
        const std::vector<U4> mt = { fx.meta[0] };
        const std::vector<F4> rl(fx.rel.begin(), fx.rel.begin() + 2);
        const TpRef& ro = fx.ref[0];
        const uint32_t nb = ro.retained;
        // C: reservation ends EXACTLY at both limits -> kept under strict >
        {
            const TpRun r = runTp(one, fx.verts, fx.root, mA, mB, mN, mt, rl,
                                  nb + 4, 16 * nb, 4 * nb);
            size_t bad = 0;
            if (r.counters[2] != 0) ++bad;
            if (r.rec[0].x != 0 || r.rec[0].y != 0 || r.rec[0].z != nb) ++bad;
            for (uint32_t k = 0; k < nb; ++k)
            {
                const F4 got = r.stream[k];
                if (got.x != ro.pts[k].x || got.y != ro.pts[k].y || got.z != ro.pts[k].z ||
                    got.w != ro.seps[k]) ++bad;
            }
            PXG_CHECK_MSG(bad == 0, "overflow C (exact-limit fit): %zu bad", bad);
        }
        // D: both buffers one contact short -> both flags, poisoned record
        {
            const TpRun r = runTp(one, fx.verts, fx.root, mA, mB, mN, mt, rl,
                                  nb + 4, 16 * nb - 16, 4 * nb - 4);
            size_t bad = 0;
            if (r.counters[2] != (OVERFLOW_CONTACT | OVERFLOW_FORCE)) ++bad;
            if (r.rec[0].x != 0xFFFFFFFFu || r.rec[0].y != 0xFFFFFFFFu || r.rec[0].z != 0) ++bad;
            for (uint32_t k = 0; k < nb + 4; ++k)
                if (r.stream[k].x != -777.0f) ++bad;
            PXG_CHECK_MSG(bad == 0, "overflow D (both overflow): %zu bad", bad);
        }
    }
}

// P9.94a: directed zero-sign cases for trimeshToPlane = planeTransform.transformInv(
// trimeshTransform) (trimeshCollision.cu:147).  Identity rotations and -0 position
// components make p a sum of exact zeros in the y/z (case A) or x (case B) lane, where
// the CUDA's conj().rotate and the pre-fix rotateInv give opposite signs.  Both cases
// force a full gen with 3 retained contacts, so gManRel[1].xyz is written.
//   A: meshP (-0,-0,-0), planeP (16,0,0): CUDA (-16,+0,+0), rotateInv (-16,-0,-0)
//   B: meshP (-0,-0,1),  planeP (0,0,0):  CUDA (-0,+0,1),   rotateInv (+0,+0,1)
PXG_TEST(npTriPlaneNp_p94ZeroSign)
{
    const uint32_t NC = 2;
    TpFixture fx;
    fx.cases.resize(NC);
    fx.manA.assign(NC * 24, F4{ 0, 0, 0, 0 });
    fx.manB.assign(NC * 24, F4{ 0, 0, 0, 0 });
    fx.manN.assign(NC * 24, F4{ 1, 0, 0, 0.25f });
    fx.meta.assign(NC, U4{ 0, 0, 0, 0 });
    fx.rel.resize(NC * 2);
    fx.ref.resize(NC);
    fx.verts = { { -0.5f, 0.25f, 0.75f, 0 }, { -0.25f, 0.5f, 0.25f, 0 }, { 0.25f, -0.75f, 0.5f, 0 } };
    fx.root = { { -1, -1, -1, 0 }, { 1, 1, 1, 0 } };
    const F4 meshPs[NC] = { { -0.0f, -0.0f, -0.0f, 0.5f }, { -0.0f, -0.0f, 1.0f, 0.5f } };
    const F4 planePs[NC] = { { 16.0f, 0.0f, 0.0f, 0.25f }, { 0.0f, 0.0f, 0.0f, 0.25f } };
    for (uint32_t i = 0; i < NC; ++i)
    {
        TpCase& cs = fx.cases[i];
        std::memset(&cs, 0, sizeof(cs));
        cs.meshQ = { 0, 0, 0, 1 };
        cs.meshP = meshPs[i];
        cs.planeQ = { 0, 0, 0, 1 };
        cs.planeP = planePs[i];
        cs.scale = { 1, 1, 1, 0.0625f };
        cs.scaleRot = { 0, 0, 0, 1 };
        cs.verts = { 0, 3, 0, 0 };
        cs.rootMeta = { 1, 0, 0, 0 };
        // far previous transform -> invalidate -> full gen
        fx.rel[i * 2 + 0] = { 0, 0, 0, 1 };
        fx.rel[i * 2 + 1] = { 1000.0f, 0, 0, 5.0f };
    }

    std::vector<TpRef> twin(NC);
    size_t nTwinDiff = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        fx.ref[i] = refTp(fx.cases[i], fx.verts, fx.root, &fx.manA[i * 24], &fx.manB[i * 24],
                          &fx.manN[i * 24], fx.meta[i], fx.rel[i * 2], v3(fx.rel[i * 2 + 1]));
        gP94TinvRotInv = true;
        twin[i] = refTp(fx.cases[i], fx.verts, fx.root, &fx.manA[i * 24], &fx.manB[i * 24],
                        &fx.manN[i * 24], fx.meta[i], fx.rel[i * 2], v3(fx.rel[i * 2 + 1]));
        gP94TinvRotInv = false;
        const TpRef& ro = fx.ref[i];
        PXG_CHECK_MSG(ro.fullGen && ro.retained == 3, "p94 case %u: want full gen with 3 contacts (%d %u)",
                      i, int(ro.fullGen), ro.retained);
        std::printf("  p94 case %c: CUDA ttpP %08x %08x %08x, rotateInv twin %08x %08x %08x\n", 'A' + i,
                    asU32(ro.ttpP.x), asU32(ro.ttpP.y), asU32(ro.ttpP.z),
                    asU32(twin[i].ttpP.x), asU32(twin[i].ttpP.y), asU32(twin[i].ttpP.z));
        if (std::memcmp(&ro.ttpP, &twin[i].ttpP, sizeof(V3)) != 0) ++nTwinDiff;
    }
    // the cases must discriminate: A in the y/z lanes, B in the x lane
    PXG_CHECK_MSG(asU32(fx.ref[0].ttpP.y) != asU32(twin[0].ttpP.y) &&
                  asU32(fx.ref[0].ttpP.z) != asU32(twin[0].ttpP.z) &&
                  asU32(fx.ref[1].ttpP.x) != asU32(twin[1].ttpP.x) && nTwinDiff == NC,
                  "p94: directed cases do not separate conj().rotate from rotateInv");
    // hard-coded CUDA bits
    PXG_CHECK_MSG(asU32(fx.ref[0].ttpP.x) == 0xC1800000u && asU32(fx.ref[0].ttpP.y) == 0u &&
                  asU32(fx.ref[0].ttpP.z) == 0u && asU32(fx.ref[1].ttpP.x) == 0x80000000u &&
                  asU32(fx.ref[1].ttpP.y) == 0u && asU32(fx.ref[1].ttpP.z) == 0x3F800000u,
                  "p94: literal oracle bits differ from the hand-derived CUDA bits");

    const TpRun r = runTp(fx.cases, fx.verts, fx.root, fx.manA, fx.manB, fx.manN, fx.meta, fx.rel,
                          6 + 32, 0x10000000u, 0x10000000u);
    size_t mm = 0, mmTwin = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const TpRef& ro = fx.ref[i];
        const F4 rq = r.rel[i * 2 + 0], rp = r.rel[i * 2 + 1];
        if (r.rec[i].z != ro.retained) ++mm;
        if (asU32(rq.x) != asU32(ro.ttpQ.x) || asU32(rq.y) != asU32(ro.ttpQ.y) ||
            asU32(rq.z) != asU32(ro.ttpQ.z) || asU32(rq.w) != asU32(ro.ttpQ.w) ||
            asU32(rp.x) != asU32(ro.ttpP.x) || asU32(rp.y) != asU32(ro.ttpP.y) ||
            asU32(rp.z) != asU32(ro.ttpP.z) || rp.w != 0.0f) ++mm;
        if (asU32(rp.x) == asU32(twin[i].ttpP.x) && asU32(rp.y) == asU32(twin[i].ttpP.y) &&
            asU32(rp.z) == asU32(twin[i].ttpP.z)) ++mmTwin;
        std::printf("  p94 case %c: GPU gManRel[1] %08x %08x %08x\n", 'A' + i, asU32(rp.x), asU32(rp.y), asU32(rp.z));
    }
    PXG_CHECK_MSG(mm == 0, "p94: %zu cases differ bitwise from the literal CUDA oracle "
                  "(%zu equal the rotateInv twin: transformInvTransform reverted?)", mm, mmTwin);
}

} // namespace
