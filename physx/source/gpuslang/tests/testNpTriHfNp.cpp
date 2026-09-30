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

// Verification of npTriHfNp.slang (trimeshHeightfieldNarrowphase,
// trimeshCollision.cu :435-682) against an exact CPU transcription.
// The upstream kernel treats the heightfield LIKE A PLANE (local-x test,
// worldNormal = hf basis vector 0, no samples read) and has NO manifold
// gate — always full gen.  Three faithful upstream quirks are pinned:
// 1. contactReduce runs with the DEFAULT dupe test (the plane kernel
//    disables it): duplicate-vertex families verify that coincident RAW
//    points cull (equal separations -> the HIGHEST lane survives), and
//    that a culled duplicate changes nbContacts vs a no-dupe-test mutant;
// 2. the warp-0 batching loop reads from the buffer FRONT (readIndex
//    omits + ind): a >32-packed-contacts family (many accepted warps in
//    one iteration) pins that slots >= 32 never participate — a
//    "corrected" readIndex + ind mutant reads them and diverges;
// 3. the overflow tail checks a SINGLE element (offset + 16 / + 4, not
//    * nbContacts), the force check is an ELSE-IF (unreachable when the
//    contact buffer overflows), and nbContacts is NEVER zeroed: exact
//    single-case boundary runs pin all three (a 16-byte contact limit
//    accepts a 6-contact case and writes 96 bytes; a 15-byte limit
//    poisons the pair but leaves the force flag unraised and the count
//    raw).
// Exact-arithmetic fixtures as established: 1/64-grid vertices,
// signed-permutation quaternions, pow2 scales; the deepest-accepted
// vertex sits EXACTLY on cDistance in one family (inclusive gate);
// disjoint-partition checks cover the nondeterministic reservations.
//
// The dupe test's 1e-8 threshold is a NEAR-coincidence test: probe
// families place vertex pairs 2^-14 apart along the plane normal
// (d = 2^-28 < 1e-8, unequal separations -> the strictly-deeper arm
// decides) and 2^-11 apart (d = 2^-22, between 1e-8 and 1e-6 -> kept
// under the correct threshold, culled by a loosened one), all exact.
// A crafted 8-vertex warp with two anchors sharing a projected point
// (they differ only along the plane normal) pins the EMPTY-CLUSTER
// guard and the strict-< label tiebreak of the clustering stage.
//
// UPSTREAM DIVERGENCE (deliberate, documented in the shader header):
// the CUDA stage-1 pack gates on the PRE-reduce hasContact (the plane
// kernel refreshes it from the reduced mask; this kernel does not),
// which in CUDA is a shared-memory data race with an OOB write in the
// saturated case — the port packs reduce SURVIVORS only, the only
// race-free reading, so that upstream behavior is intentionally NOT
// transcribed and cannot be pinned by this test.

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
struct ThCase
{
    F4 meshQ, meshP, hfQ, hfP, scale, scaleRot;
    U4 verts;
};
struct ThParams { uint32_t numCases, contactBytesLimit, forceBytesLimit, pad0; };

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

// contactReduce<true, true, 6, DUPE=true> on lane arrays; sets dupeCulled
// when the pre-cull actually removed a lane
static uint32_t contactReduce6DupeRef(const V3 ptRaw[32], const float sep[32], V3 normal,
                                      uint32_t allMask, float clusterBias, bool& dupeCulled)
{
    {
        uint32_t dupes = 0;
        for (uint32_t l = 0; l < 32; ++l)
        {
            if (((allMask >> l) & 1) == 0) continue;
            bool imADupe = false;
            for (uint32_t m = allMask; m != 0; m &= (m - 1))
            {
                const uint32_t i = ctz(m);
                const V3 dv = sub(ptRaw[i], ptRaw[l]);
                const float d = dot3(dv, dv);
                if (d < 1e-8f && (sep[i] < sep[l] || (sep[i] == sep[l] && i > l)))
                    imADupe = true;
            }
            if (imADupe) dupes |= 1u << l;
        }
        if (dupes != 0) dupeCulled = true;
        allMask &= ~dupes;
    }

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

struct ThRef
{
    uint32_t retained = 0, iters = 0;
    V3 pts[6];
    float seps[6];
    V3 worldNormal{};
    bool reduced = false, dupeCulled = false, bigPack = false;
};

// P9.94a mutant twin: true selects the port's PRE-FIX trimeshToHf.p spelling q.rotateInv(v)
// instead of the CUDA's heightfieldTransform.transformInv(trimeshTransform) (trimeshCollision.cu
// :502) = q.getConjugate().rotate(v).  Used only to prove that npTriHfNp_p94ZeroSign separates
// the two forms.
static bool gP94TinvRotInv = false;

// P9.94a: PxTransform::transformInv(PxTransform).p (PxTransform.h:162-169) LITERALLY:
// qinv = q.getConjugate(); qinv.rotate(src.p - p).  NOT q.rotateInv(src.p - p): same bits for
// every nonzero component, but an exact-zero component can take the other sign.
static V3 ttpPRef(const ThCase& cs)
{
    const V3 v = sub(v3(cs.meshP), v3(cs.hfP));
    return gP94TinvRotInv ? quatRotateInv(cs.hfQ, v) : quatRotate(quatConj(cs.hfQ), v);
}

static ThRef refTh(const ThCase& cs, const std::vector<F4>& verts)
{
    ThRef ro;
    const float cd = cs.hfP.w, clusterBias = cs.scale.w;
    const F4 ttpQ = quatMul(quatConj(cs.hfQ), cs.meshQ);
    const V3 ttpP = ttpPRef(cs); // P9.94a: literal conj().rotate
    ro.worldNormal = basis0(cs.hfQ);

    V3 contacts[198];
    float seps[198];
    uint32_t retained = 0;
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
                const V3 pip = add(quatRotate(ttpQ, p), ttpP);
                if (cd >= pip.x)
                {
                    laneSep[l] = pip.x;
                    lanePt[l] = add(quatRotate(cs.meshQ, p), v3(cs.meshP));
                    mask |= 1u << l;
                }
            }
            if (popc(mask) > 6u) ro.reduced = true;
            mask = contactReduce6DupeRef(lanePt, laneSep, ro.worldNormal, mask, clusterBias, ro.dupeCulled);
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
        if (total > 32) ro.bigPack = true;
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
                const uint32_t ri = popc(readMask & ((1u << l) - 1u));
                if (ind + ri < total)
                {
                    lanePt[l] = contacts[ri]; // FAITHFUL: front re-read, no + ind
                    laneSep[l] = seps[ri];
                    has[l] = true;
                }
            }
            uint32_t mask = 0;
            for (uint32_t l = 0; l < 32; ++l)
                if (has[l]) mask |= 1u << l;
            ind += popc(readMask);
            mask = contactReduce6DupeRef(lanePt, laneSep, ro.worldNormal, mask, clusterBias, ro.dupeCulled);
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
    ro.retained = retained;
    for (uint32_t k = 0; k < retained && k < 6; ++k)
    {
        ro.pts[k] = contacts[k];
        ro.seps[k] = seps[k];
    }
    return ro;
}

struct ThRun
{
    std::vector<F4> dbg, stream;
    std::vector<U4> rec;
    uint32_t counters[3];
};
static ThRun runTh(const std::vector<ThCase>& cases, const std::vector<F4>& verts,
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
    PxgBufferHandle bCt = createBufferFrom(dev, ctrInit);
    PxgBufferHandle bS = createBufferFrom(dev, streamInit);
    PxgBufferHandle bRec = createBufferFrom(dev, recInit);
    PxgBufferHandle bD = createBufferFrom(dev, dbgInit);
    ThParams prm = { n, contactLimit, forceLimit, 0 };
    Kernel k("npTriHfNp.spv", "thNarrowphase",
             { (uint32_t)sizeof(ThCase), 16, 4, 16, 16, 16 },
             sizeof(ThParams));
    k.dispatch((n + 63) / 64, { bC, bV, bCt, bS, bRec, bD }, prm);
    dev->flush();
    ThRun r;
    const auto ctr = downloadBuffer<uint32_t>(dev, bCt, 3);
    r.counters[0] = ctr[0]; r.counters[1] = ctr[1]; r.counters[2] = ctr[2];
    r.stream = downloadBuffer<F4>(dev, bS, streamSlots);
    r.rec = downloadBuffer<U4>(dev, bRec, n);
    r.dbg = downloadBuffer<F4>(dev, bD, n);
    for (PxgBufferHandle h : { bC, bV, bCt, bS, bRec, bD })
        dev->destroyBuffer(h);
    return r;
}

struct ThFixture
{
    std::vector<ThCase> cases;
    std::vector<F4> verts;
    std::vector<ThRef> ref;
};

// family = i % 8: 0/1/7 accept-all (reduce; i%4==1 adds duplicate verts;
// i%16==15 multi-iteration 1200/2112), 2 boundary-exact small (deepest
// vertex EXACTLY on cDistance), 3 all-rejected, 4 front-quirk (>32 packed
// contacts in one iteration), 5 mixed acceptance (+ dupes on i%4==1),
// 6 tiny accept-all (2..6 verts, no reduction)
static ThFixture buildFixture(uint32_t NC, uint32_t seed)
{
    ThFixture fx;
    std::mt19937 rng(seed);
    fx.cases.resize(NC);
    fx.ref.resize(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        ThCase& cs = fx.cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const uint32_t fam = i % 8;
        cs.meshQ = exactQuat(rng);
        cs.meshP = { grid64(rng, -1, 1), grid64(rng, -1, 1), grid64(rng, -1, 1), 0 };
        cs.hfQ = exactQuat(rng);
        const float sc[3] = { 0.5f, 1.0f, 2.0f };
        cs.scale = { sc[rng() % 3], sc[rng() % 3], sc[rng() % 3], 0.0625f };
        cs.scaleRot = exactQuat(rng);
        // probe cases use identity scaling so engineered offsets survive
        // the vertex->world chain exactly
        const bool probe = (i % 16 == 6) || (i % 16 == 7) || (i % 16 == 14) || (i == 40);
        if (probe)
        {
            cs.scale = { 1, 1, 1, 0.0625f };
            cs.scaleRot = { 0, 0, 0, 1 };
        }

        uint32_t numVerts = 40 + i % 50;
        if (fam == 2) numVerts = 4 + i % 4;
        if (fam == 3) numVerts = 10;
        if (fam == 4) numVerts = 300 + (i % 3) * 32;
        if (fam == 6) numVerts = 2 + i % 5;
        if (i % 16 == 6) numVerts = 5;   // near-dupe probe (2^-14 pair)
        if (i % 16 == 15) numVerts = (i == 31) ? 2112 : 1200;
        if (i == 40) numVerts = 8;       // crafted empty-cluster warp
        if (i == 48) numVerts = 0;       // zero verts: no iterations
        if (i == 80) numVerts = 1024;    // exact iteration multiple

        fx.verts.push_back({ 99, 99, 99, 0 });
        fx.verts.push_back({ 99, 99, 99, 0 });
        cs.verts = { (uint32_t)fx.verts.size(), numVerts, 0, 0 };
        for (uint32_t k = 0; k < numVerts; ++k)
        {
            const bool cluster = (i % 16 == 15) && k >= 1024;
            const float lo = cluster ? -0.25f : -2.0f, hi = cluster ? 0.25f : 2.0f;
            fx.verts.push_back({ grid64(rng, lo, hi), grid64(rng, lo, hi), grid64(rng, lo, hi), 0 });
        }
        if (i % 4 == 1)
        {
            // duplicate verts INSIDE one warp: coincident world points with
            // equal separations -> the dupe pre-cull removes the lower lane
            for (uint32_t k = 2; k + 1 < numVerts && k < 40; k += 9)
                if ((k % 32) != 31)
                    fx.verts[cs.verts.x + k + 1] = fx.verts[cs.verts.x + k];
        }

        const V3 b0q = basis0(cs.hfQ);
        V3 pp = { grid64(rng, -1, 1), grid64(rng, -1, 1), grid64(rng, -1, 1) };
        float cd = 0.25f;
        if (fam == 0 || fam == 1 || fam == 4 || fam == 6 || fam == 7)
            pp = add(pp, mul(b0q, 16.0f)); // all verts accepted
        else if (fam == 3)
            pp = sub(pp, mul(b0q, 16.0f)); // all verts rejected
        cs.hfP = { pp.x, pp.y, pp.z, cd };

        if (fam == 2)
        {
            // cd = the exact max heightfield-space x over the verts
            const F4 ttpQ = quatMul(quatConj(cs.hfQ), cs.meshQ);
            const V3 ttpP = ttpPRef(cs); // P9.94a: literal conj().rotate
            float mxX = -FLT_BIG;
            for (uint32_t k = 0; k < numVerts; ++k)
            {
                const V3 p = vertex2ShapeRef(v3(fx.verts[cs.verts.x + k]), v3(cs.scale), cs.scaleRot);
                mxX = std::max(mxX, add(quatRotate(ttpQ, p), ttpP).x);
            }
            cs.hfP.w = mxX;
        }

        // near-dupe probes: a vertex pair offset by h along the WORLD plane
        // normal (identity scaling -> vertex offset = rotInv(meshQ, h*n),
        // exact): d = h*h against the 1e-8 threshold, separations differ by
        // exactly h so the strictly-deeper dupe arm decides
        if (i % 16 == 6 || i % 16 == 7 || i % 16 == 14)
        {
            // i%16==14 places the 2^-11 pair in a TINY family (<= 6 accepted,
            // no reduction) so keeping vs culling it changes nbContacts
            // directly — a loosened 1e-6 threshold is output-equivalent in
            // reduced warps (the pair shares a cluster) but not here
            const float h = (i % 16 == 6) ? 0.00006103515625f /* 2^-14 */
                                          : 0.00048828125f;   /* 2^-11 */
            const V3 n = basis0(cs.hfQ);
            const V3 vo = quatRotateInv(cs.meshQ, mul(n, h));
            const uint32_t a = (i % 16 == 6) ? 1u : (i % 16 == 7) ? 9u : 0u;
            const F4 base = fx.verts[cs.verts.x + a];
            fx.verts[cs.verts.x + a + 1] = { base.x + vo.x, base.y + vo.y, base.z + vo.z, 0 };
        }

        // crafted empty-cluster warp: hf-LOCAL points (x = separation);
        // lanes 1/2 share (y,z) and differ only in x -> identical projected
        // points; lane 1 is the strict argmax-sep anchor, lane 3 the strict
        // farthest-projection anchor, every other point sits strictly on one
        // side of the anchor line while lane 2's cross-dir value is exactly
        // 0 -> lane 2 becomes the fourth anchor and its cluster is EMPTY
        // (its own point ties lane 1's anchor and strict < keeps the
        // earlier one)
        if (i == 40)
        {
            const V3 local[8] = {
                { -0.5f,   2.0f,  4.5f }, { 0.0f, 4.0f, 4.0f }, { -0.25f, 4.0f, 4.0f },
                { -0.5f,   4.0f, 12.0f }, { -0.75f, 1.0f, 5.0f }, { -1.0f, 3.0f, 7.0f },
                { -0.5f,   2.5f,  9.5f }, { -0.625f, 3.5f, 10.0f } };
            cs.hfP.w = 0.25f;
            for (uint32_t k = 0; k < 8; ++k)
            {
                const V3 world = add(quatRotate(cs.hfQ, local[k]), v3(cs.hfP));
                const V3 vert = quatRotateInv(cs.meshQ, sub(world, v3(cs.meshP)));
                fx.verts[cs.verts.x + k] = { vert.x, vert.y, vert.z, 0 };
            }
        }
    }
    for (uint32_t i = 0; i < NC; ++i)
        fx.ref[i] = refTh(fx.cases[i], fx.verts);
    return fx;
}

PXG_TEST(npTriHfNp_narrowphase)
{
    const uint32_t NC = 256;
    ThFixture fx = buildFixture(NC, 1040701);

    uint32_t totalContacts = 0;
    size_t nReduce = 0, nDupe = 0, nBigPack = 0, nZero = 0, nMulti = 0, nBoundary = 0, nTiny = 0,
           nBadProbe = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const ThRef& ro = fx.ref[i];
        totalContacts += ro.retained;
        if (ro.reduced) ++nReduce;
        if (ro.dupeCulled) ++nDupe;
        if (ro.bigPack) ++nBigPack;
        if (ro.retained == 0) ++nZero;
        if (ro.iters > 1) ++nMulti;
        if (i % 8 == 2) ++nBoundary;
        if (i % 8 == 6 && ro.retained == fx.cases[i].verts.y) ++nTiny;
        // near-dupe deep probes must actually cull (d = 2^-28 < 1e-8)
        if (i % 16 == 6 && !ro.dupeCulled) ++nBadProbe;
        // zero-vert case must emit the empty record
        if (i == 48 && ro.retained != 0) ++nBadProbe;
    }
    PXG_CHECK_MSG(nReduce > 120 && nDupe > 40 && nBigPack > 25 && nZero > 25 && nMulti > 10 &&
                  nBoundary == 32 && nTiny > 15 && nBadProbe == 0,
                  "triHf coverage thin (%zu %zu %zu %zu %zu %zu %zu %zu)",
                  nReduce, nDupe, nBigPack, nZero, nMulti, nBoundary, nTiny, nBadProbe);

    const uint32_t streamSlots = totalContacts + 32;
    const ThRun r = runTh(fx.cases, fx.verts, streamSlots, 0x10000000u, 0x10000000u);

    PXG_CHECK_MSG(r.counters[0] == 16 * totalContacts && r.counters[1] == 4 * totalContacts &&
                  r.counters[2] == 0,
                  "triHf counters %u/%u/%u (want %u/%u/0)", r.counters[0], r.counters[1],
                  r.counters[2], 16 * totalContacts, 4 * totalContacts);

    size_t mm = 0;
    std::vector<std::pair<uint32_t, uint32_t>> rangesC, rangesF;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const ThRef& ro = fx.ref[i];
        const U4 rec = r.rec[i];
        if (rec.z != ro.retained || rec.w != 0) ++mm; // count RAW in all paths
        if (ro.retained == 0)
        {
            if (rec.x != 0xFFFFFFFFu || rec.y != 0xFFFFFFFFu) ++mm;
        }
        else
        {
            if (rec.x % 16 != 0 || rec.y % 4 != 0 ||
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
        if (r.dbg[i].x != ro.worldNormal.x || r.dbg[i].y != ro.worldNormal.y ||
            r.dbg[i].z != ro.worldNormal.z || r.dbg[i].w != float(ro.retained)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "triHf: %zu mismatches", mm);

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
        PXG_CHECK_MSG(bad == 0, "triHf: %s ranges do not partition [0, %u)", what, total);
    };
    checkPartition(rangesC, 16 * totalContacts, "contact");
    checkPartition(rangesF, 4 * totalContacts, "force");

    size_t dirty = 0;
    for (uint32_t s = totalContacts; s < streamSlots; ++s)
        if (r.stream[s].x != -777.0f) ++dirty;
    PXG_CHECK_MSG(dirty == 0, "triHf: %zu dirty stream slots beyond total", dirty);
}

PXG_TEST(npTriHfNp_overflow)
{
    // 16 accept-all cases (retained 6 each): the pass criterion is
    // offset + 16 <= limit (SINGLE element), so with offsets partitioning
    // multiples of 96 the number of surviving cases is deterministic
    const uint32_t NC = 16;
    ThFixture fx = buildFixture(NC, 1040702);
    for (uint32_t i = 0; i < NC; ++i)
    {
        const V3 b0q = basis0(fx.cases[i].hfQ);
        V3 pp = v3(fx.cases[i].hfP);
        pp = add(pp, mul(b0q, 32.0f)); // all verts accepted regardless of family
        fx.cases[i].hfP = { pp.x, pp.y, pp.z, 0.25f };
        fx.cases[i].verts = fx.cases[0].verts; // shared 40-vert range: retained 6 everywhere
    }
    for (uint32_t i = 0; i < NC; ++i)
        fx.ref[i] = refTh(fx.cases[i], fx.verts);
    uint32_t total = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        PXG_CHECK_MSG(fx.ref[i].retained == 6, "overflow fixture: case %u retained %u != 6",
                      i, fx.ref[i].retained);
        total += fx.ref[i].retained;
    }
    const uint32_t streamSlots = total + 16;

    // run A: contact limit passes exactly 8 of 16 (offset <= 96*8 - 16)
    {
        const uint32_t limit = 96 * 8;
        const ThRun r = runTh(fx.cases, fx.verts, streamSlots, limit, 0x10000000u);
        PXG_CHECK_MSG(r.counters[0] == 16 * total && r.counters[2] == OVERFLOW_CONTACT,
                      "overflow A: counters %u/%u", r.counters[0], r.counters[2]);
        size_t bad = 0, nOk = 0, nPoison = 0;
        for (uint32_t i = 0; i < NC; ++i)
        {
            const ThRef& ro = fx.ref[i];
            const U4 rec = r.rec[i];
            if (rec.z != 6) ++bad; // count stays RAW even when poisoned
            if (rec.x == 0xFFFFFFFFu)
            {
                ++nPoison;
                if (rec.y != 0xFFFFFFFFu) ++bad;
            }
            else
            {
                ++nOk;
                if (rec.x + 16 > limit) ++bad; // pass criterion is one element
                for (uint32_t k = 0; k < ro.retained; ++k)
                {
                    const F4 got = r.stream[rec.x / 16 + k];
                    if (got.x != ro.pts[k].x || got.y != ro.pts[k].y || got.z != ro.pts[k].z ||
                        got.w != ro.seps[k]) ++bad;
                }
            }
        }
        PXG_CHECK_MSG(bad == 0 && nOk == 8 && nPoison == 8,
                      "overflow A: %zu bad, %zu ok, %zu poisoned", bad, nOk, nPoison);
    }

    // run B: force-only overflow (8 of 16): contacts still written, count raw
    {
        const uint32_t limit = 24 * 8;
        const ThRun r = runTh(fx.cases, fx.verts, streamSlots, 0x10000000u, limit);
        PXG_CHECK_MSG(r.counters[1] == 4 * total && r.counters[2] == OVERFLOW_FORCE,
                      "overflow B: counters %u/%u", r.counters[1], r.counters[2]);
        size_t bad = 0, nPoisonF = 0;
        for (uint32_t i = 0; i < NC; ++i)
        {
            const ThRef& ro = fx.ref[i];
            const U4 rec = r.rec[i];
            if (rec.x == 0xFFFFFFFFu || rec.z != 6) { ++bad; continue; }
            if (rec.y == 0xFFFFFFFFu) ++nPoisonF;
            else if (rec.y + 4 > limit) ++bad;
            for (uint32_t k = 0; k < ro.retained; ++k)
            {
                const F4 got = r.stream[rec.x / 16 + k];
                if (got.x != ro.pts[k].x || got.y != ro.pts[k].y || got.z != ro.pts[k].z ||
                    got.w != ro.seps[k]) ++bad;
            }
        }
        PXG_CHECK_MSG(bad == 0 && nPoisonF == 8, "overflow B: %zu bad, %zu force-poisoned", bad, nPoisonF);
    }

    // runs C/D: single case pins the SINGLE-ELEMENT checks and the else-if
    {
        const std::vector<ThCase> one = { fx.cases[0] };
        const ThRef& ro = fx.ref[0];
        // C: contact limit of ONE PxContact accepts a 6-contact case and
        //    writes all 96 bytes past the limit (faithful quirk); force
        //    limit of ONE PxU32 likewise passes
        {
            const ThRun r = runTh(one, fx.verts, 6 + 4, 16, 4);
            size_t bad = 0;
            if (r.counters[2] != 0) ++bad;
            if (r.rec[0].x != 0 || r.rec[0].y != 0 || r.rec[0].z != 6) ++bad;
            for (uint32_t k = 0; k < 6; ++k)
            {
                const F4 got = r.stream[k];
                if (got.x != ro.pts[k].x || got.y != ro.pts[k].y || got.z != ro.pts[k].z ||
                    got.w != ro.seps[k]) ++bad;
            }
            PXG_CHECK_MSG(bad == 0, "overflow C (single-element fit): %zu bad", bad);
        }
        // D: contact limit one byte short -> poisoned pair; the force check
        //    is SKIPPED (else-if), so its flag stays unraised even though
        //    the force buffer would also overflow; count stays raw
        {
            const ThRun r = runTh(one, fx.verts, 6 + 4, 15, 3);
            size_t bad = 0;
            if (r.counters[2] != OVERFLOW_CONTACT) ++bad;
            if (r.rec[0].x != 0xFFFFFFFFu || r.rec[0].y != 0xFFFFFFFFu || r.rec[0].z != 6) ++bad;
            for (uint32_t k = 0; k < 6 + 4; ++k)
                if (r.stream[k].x != -777.0f) ++bad;
            PXG_CHECK_MSG(bad == 0, "overflow D (else-if skip): %zu bad", bad);
        }
    }
}

// P9.94a: directed zero-sign cases for trimeshToHf = hfT.transformInv(meshT).  One vertex
// (0, 1, -1), meshQ = 180 deg about z, identity hf: the composed rotation gives x = -0 for the
// vertex, so sep = -0 + ttpP.x keeps ttpP.x's sign.  ttpP.x = conj(hfQ).rotate(meshP - hfP).x
// is -0 (A) / +0 (B) under CUDA and +0 (A) / -0 (B) under the old rotateInv spelling.  No RNG.
PXG_TEST(npTriHfNp_p94ZeroSign)
{
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    const float nz = -0.0f;
    std::vector<F4> verts = { { 0.0f, 1.0f, -1.0f, 0.0f } };
    std::vector<ThCase> cases(2);
    for (ThCase& cs : cases)
    {
        std::memset(&cs, 0, sizeof(cs));
        cs.meshQ = { 0, 0, 1, 0 };
        cs.hfQ = { 0, 0, 0, 1 };
        cs.hfP = { 0, 0, 0, 0.25f };
        cs.scale = { 1, 1, 1, 0.0625f };
        cs.scaleRot = { 0, 0, 0, 1 };
        cs.verts = { 0, 1, 0, 0 };
    }
    cases[0].meshP = { nz, nz, 0.5f, 0 }; // A: CUDA sep -0
    cases[1].meshP = { nz, nz, nz, 0 };   // B: CUDA sep +0
    const uint32_t wantSep[2] = { 0x80000000u, 0x00000000u };

    ThRef lit[2];
    for (uint32_t i = 0; i < 2; ++i)
    {
        lit[i] = refTh(cases[i], verts);
        gP94TinvRotInv = true;
        const ThRef twin = refTh(cases[i], verts);
        gP94TinvRotInv = false;
        PXG_CHECK_MSG(lit[i].retained == 1 && twin.retained == 1,
                      "p94 case %u: retained %u/%u != 1", i, lit[i].retained, twin.retained);
        PXG_CHECK_MSG(u32(lit[i].seps[0]) == wantSep[i],
                      "p94 case %u: literal sep 0x%08x != 0x%08x", i, u32(lit[i].seps[0]), wantSep[i]);
        PXG_CHECK_MSG(u32(twin.seps[0]) != u32(lit[i].seps[0]),
                      "p94 case %u: twin sep 0x%08x does not discriminate", i, u32(twin.seps[0]));
    }

    const ThRun r = runTh(cases, verts, 2 + 4, 0x10000000u, 0x10000000u);
    for (uint32_t i = 0; i < 2; ++i)
    {
        const U4 rec = r.rec[i];
        PXG_CHECK_MSG(rec.z == 1 && rec.x != 0xFFFFFFFFu && rec.x % 16 == 0 && rec.x / 16 < 2,
                      "p94 case %u: record %u/%u", i, rec.x, rec.z);
        if (rec.z != 1 || rec.x == 0xFFFFFFFFu || rec.x / 16 >= 2) continue;
        const F4 got = r.stream[rec.x / 16];
        PXG_CHECK_MSG(u32(got.w) == u32(lit[i].seps[0]) && u32(got.x) == u32(lit[i].pts[0].x) &&
                      u32(got.y) == u32(lit[i].pts[0].y) && u32(got.z) == u32(lit[i].pts[0].z),
                      "p94 case %u: GPU sep 0x%08x != literal 0x%08x", i, u32(got.w), u32(lit[i].seps[0]));
    }
}

} // namespace
