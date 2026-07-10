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

// Verification of npSbPrim.slang (softbodyPrimitives.cu non-GJK cores).
// sbTetPlane and sbEmit compare EXACTLY (1/64-grid vertices,
// signed-permutation plane quaternions, integer emitter fixtures) with
// deliberate on-threshold cases pinning the inclusive gates; sbParticle
// and sbClothVert use transcribed CPU references with Slack taints on the
// sqrt/normalize-driven gates (sqDist vs cDistance^2, dot(n,n2) vs 0.8 /
// 0.707, sqDist vs 1e-8, m vs 1e-10, cross-face running-min ties with the
// benign-same-point exclusion).  Filter tables pin the three
// PxgNonRigidFilterPair query shapes, the PxgRigidFilterPair
// ignoreIndex2 accept, the findRange bucket chain (a wrong mUniqueId
// entry must NOT match, ignoreIndex2 = false) and the deliberate
// u64->u32 particle-mask truncation (the filter is keyed by the UNSORTED
// index while positions/sort keys use the SORTED index).  Writer checks
// cover both PxgFEMContactWriter flavours (NoBarycentric with the
// material RMW+clear vs the barycentric overload that must leave the
// material dword untouched) and the PxgSoftBodyContactWriter (no sort
// keys, u32->u64 pairInd0 widening, auxInd untouched), plus capacity
// runs (ungated counters overshoot, stores stop).

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

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct SbCase { F4 a, b, c, d, p, q; U4 ids, ids2; };
struct SbParams { uint32_t numCases, maxContacts, nbRFilter, nbNRFilter, nbRunsum, pad0, pad1, pad2; };
struct PairR { uint64_t index0, index1; uint32_t index2, pad0, pad1, pad2; };
struct PairNR { uint64_t index0; uint32_t index1, index2; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static V3 getNorm0(V3 v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static float pxNormalizeRef(V3& v)
{
    const float m = std::sqrt(dot3(v, v));
    if (m > 0.0f) v = mul(v, 1.0f / m);
    return m;
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
static V3 basisVector0(F4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return { (q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2 };
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

struct Slack
{
    bool near = false;
    void note(float margin, float eps) { if (margin != 0.0f && std::fabs(margin) < eps) near = true; }
};

// ---- filter references --------------------------------------------------------
static bool findRigidRef(const std::vector<PairR>& t, uint64_t rigid, uint32_t mask)
{
    size_t l = 0, r = t.size();
    while (l < r)
    {
        const size_t mid = (l + r) / 2;
        const PairR& p = t[mid];
        int cmp;
        if (rigid != p.index0) cmp = rigid < p.index0 ? -1 : 1;
        else if (uint64_t(mask) != p.index1) cmp = uint64_t(mask) < p.index1 ? -1 : 1;
        else if (0u != p.index2) cmp = -1;
        else cmp = 0;
        if (cmp == 0 || (rigid == p.index0 && uint64_t(mask) == p.index1)) return true;
        if (cmp < 0) r = mid; else l = mid + 1;
    }
    return false;
}
static bool findNonRigidRef(const std::vector<PairNR>& t, uint64_t q0, uint32_t q1, uint32_t q2, bool ignore2)
{
    size_t l = 0, r = t.size();
    while (l < r)
    {
        const size_t mid = (l + r) / 2;
        const PairNR& p = t[mid];
        int cmp;
        if (q0 != p.index0) cmp = q0 < p.index0 ? -1 : 1;
        else if (q1 != p.index1) cmp = q1 < p.index1 ? -1 : 1;
        else if (q2 != p.index2) cmp = q2 < p.index2 ? -1 : 1;
        else cmp = 0;
        if (cmp == 0 || (ignore2 && q0 == p.index0 && q1 == p.index1)) return true;
        if (cmp < 0) r = mid; else l = mid + 1;
    }
    return false;
}
static uint32_t findRangeRef(uint32_t value, const std::vector<uint32_t>& runsum)
{
    if (runsum.empty()) return 0;
    uint32_t l = 0, r = (uint32_t)runsum.size();
    while (l < r)
    {
        const uint32_t m = (l + r) / 2;
        if (runsum[m] > value) r = m; else l = m + 1;
    }
    return r - 1;
}

// ---- geometry references (transcribed) -----------------------------------------
static F4 pointOutsideOfPlane4Ref(V3 p, V3 a, V3 b, V3 c, V3 d)
{
    const V3 ap = sub(p, a), ab = sub(b, a), ac = sub(c, a), ad = sub(d, a);
    const V3 v0 = crossp(ab, ac);
    const float sa0 = dot3(v0, ap), sd0 = dot3(v0, ad);
    const V3 v1 = crossp(ac, ad);
    const float sa1 = dot3(v1, ap), sd1 = dot3(v1, ab);
    const V3 v2 = crossp(ad, ab);
    const float sa2 = dot3(v2, ap), sd2 = dot3(v2, ac);
    const V3 bd = sub(d, b), bc = sub(c, b);
    const V3 v3_ = crossp(bd, bc);
    const float sd3 = dot3(v3_, sub(p, b)), sa3 = dot3(v3_, sub(a, b));
    return { sa0 * sd0, sa1 * sd1, sa2 * sd2, sa3 * sd3 };
}

static V3 closestPtPointTriangle2Ref(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return add(a, mul(ab, d1 / (d1 - d3)));
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return add(a, mul(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, mul(sub(c, b), w));
    }
    const float denom = 1.0f / (va + vb + vc);
    return add(a, add(mul(ab, vb * denom), mul(ac, vc * denom)));
}

static void closestPtTetrahedronRef(V3 point, uint32_t hint, V3 a, V3 b, V3 c, V3 d,
                                    V3& normal, V3& closestP, float& sqDist, Slack& sl)
{
    const V3 ac = sub(c, a), ab = sub(b, a), ad = sub(d, a);
    struct Face { uint32_t bit; V3 p0, p1, p2, e0, e1, nc0, nc1, flipRef; bool flipGT; };
    // face list transcribed in order; flip test operand + comparison direction
    const Face faces[4] = {
        { 1, a, b, c, ab, ac, ac, ab, ad, false },
        { 2, a, b, d, ab, ad, ab, ad, ac, false },
        { 4, a, c, d, ac, ad, ac, ad, ab, false },
        { 8, b, c, d, sub(c, b), sub(d, b), sub(c, b), sub(d, b), ab, true },
    };
    for (int f = 0; f < 4; ++f)
    {
        if (!(hint & faces[f].bit)) continue;
        const V3 tPoint = closestPtPointTriangle2Ref(point, faces[f].p0, faces[f].p1, faces[f].p2, faces[f].e0, faces[f].e1);
        const V3 v = sub(tPoint, point);
        const float tSqDist = dot3(v, v);
        if (tSqDist < sqDist)
        {
            V3 n = getNorm0(crossp(faces[f].nc0, faces[f].nc1));
            // faces 1/2/4 build cross(ac,ab)/cross(ab,ad)/cross(ac,ad); face 1 order is (ac,ab)
            if (f == 0) n = getNorm0(crossp(ac, ab));
            const float flipDot = dot3(faces[f].flipRef, n);
            if (faces[f].flipGT ? (flipDot > 0.0f) : (flipDot < 0.0f)) n = neg(n);
            normal = n;
            closestP = tPoint;
            sqDist = tSqDist;
        }
        else
        {
            // cross-face running-min tie: output-relevant when the tied
            // candidate differs in closest point OR face normal (n2 feeds the
            // output even when the points coincide)
            V3 nc = getNorm0(crossp(faces[f].nc0, faces[f].nc1));
            if (f == 0) nc = getNorm0(crossp(ac, ab));
            const float fd = dot3(faces[f].flipRef, nc);
            if (faces[f].flipGT ? (fd > 0.0f) : (fd < 0.0f)) nc = neg(nc);
            const V3 dcp = sub(tPoint, closestP);
            const V3 dcn = sub(nc, normal);
            if (dot3(dcp, dcp) > 1e-6f || dot3(dcn, dcn) > 1e-6f)
                sl.note(tSqDist - sqDist, 1e-4f);
        }
    }
}

static bool computeTetBarycentricRef(V3 a, V3 b, V3 c, V3 d, V3 point, F4& bary)
{
    const float eps = 1e-5f;
    const V3 ba = sub(b, a), ca = sub(c, a), da = sub(d, a), pa = sub(point, a);
    const float detBcd = dot3(ba, crossp(ca, da));
    const float v = dot3(pa, crossp(ca, da)) / detBcd;
    const float w = dot3(ba, crossp(pa, da)) / detBcd;
    const float x = dot3(ba, crossp(ca, pa)) / detBcd;
    const float u = 1.0f - v - w - x;
    auto cl = [](float t) { return std::min(std::max(t, 0.0f), 1.0f); };
    bary = { cl(u), cl(v), cl(w), cl(x) };
    return (u >= -eps && u <= 1 + eps) && (v >= -eps && v <= 1 + eps) &&
           (w >= -eps && w <= 1 + eps) && (x >= -eps && x <= 1 + eps);
}

static const uint32_t POISON = 0xCDCDCDCDu;
static const uint32_t INFOPOISON = 0xDEADBEEFu;

// shared GPU runner ------------------------------------------------------------
struct SbRun
{
    std::vector<F4> pt, np, bc, sbPt, sbNp, sbB0, sbB1, dbg;
    std::vector<U4> in, sbIn;
    std::vector<uint32_t> by, tb, ix, s32;
    uint32_t femWc, sbWc;
};
static SbRun runSb(const char* entry, const std::vector<SbCase>& cases,
                   const std::vector<PairR>& rf, const std::vector<PairNR>& nrf,
                   const std::vector<uint32_t>& runsum, const std::vector<uint32_t>& bufUid,
                   uint32_t maxContacts, uint32_t CAP)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = (uint32_t)cases.size();
    std::vector<F4> f4Init(CAP, { -9, -9, -9, -9 });
    std::vector<U4> infoInit(CAP, { INFOPOISON, INFOPOISON, INFOPOISON, INFOPOISON });
    std::vector<uint32_t> u1Init(CAP, POISON), u2Init(CAP * 2, POISON), wc0(1, 0);
    std::vector<F4> dbgInit(n, { -7, -7, -7, -7 });
    std::vector<uint32_t> rfRaw(std::max<size_t>(rf.size(), 1) * 8, 0);
    if (!rf.empty()) std::memcpy(rfRaw.data(), rf.data(), rf.size() * sizeof(PairR));
    std::vector<uint32_t> nrfRaw(std::max<size_t>(nrf.size(), 1) * 4, 0);
    if (!nrf.empty()) std::memcpy(nrfRaw.data(), nrf.data(), nrf.size() * sizeof(PairNR));
    std::vector<uint32_t> rsBuf = runsum.empty() ? std::vector<uint32_t>{ 0 } : runsum;
    std::vector<uint32_t> uidBuf = bufUid.empty() ? std::vector<uint32_t>{ 0 } : bufUid;

    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bRF = createBufferFrom(dev, rfRaw);
    PxgBufferHandle bNF = createBufferFrom(dev, nrfRaw);
    PxgBufferHandle bRS = createBufferFrom(dev, rsBuf);
    PxgBufferHandle bUID = createBufferFrom(dev, uidBuf);
    PxgBufferHandle bPt = createBufferFrom(dev, f4Init);
    PxgBufferHandle bNp = createBufferFrom(dev, f4Init);
    PxgBufferHandle bBc = createBufferFrom(dev, f4Init);
    PxgBufferHandle bIn = createBufferFrom(dev, infoInit);
    PxgBufferHandle bWc = createBufferFrom(dev, wc0);
    PxgBufferHandle bBy = createBufferFrom(dev, u2Init);
    PxgBufferHandle bTb = createBufferFrom(dev, u1Init);
    PxgBufferHandle bIx = createBufferFrom(dev, u1Init);
    PxgBufferHandle bS3 = createBufferFrom(dev, u1Init);
    PxgBufferHandle bSP = createBufferFrom(dev, f4Init);
    PxgBufferHandle bSN = createBufferFrom(dev, f4Init);
    PxgBufferHandle bB0 = createBufferFrom(dev, f4Init);
    PxgBufferHandle bB1 = createBufferFrom(dev, f4Init);
    PxgBufferHandle bSI = createBufferFrom(dev, infoInit);
    PxgBufferHandle bSW = createBufferFrom(dev, wc0);
    PxgBufferHandle bDb = createBufferFrom(dev, dbgInit);

    SbParams prm = { n, maxContacts, (uint32_t)rf.size(), (uint32_t)nrf.size(), (uint32_t)runsum.size(), 0, 0, 0 };
    Kernel k("npSbPrim.spv", entry,
             { (uint32_t)sizeof(SbCase), (uint32_t)sizeof(PairR), (uint32_t)sizeof(PairNR), 4, 4,
               16, 16, 16, 16, 4, 8, 4, 4, 4,
               16, 16, 16, 16, 16, 4, 16 },
             sizeof(SbParams));
    k.dispatch((n + 63) / 64, { bC, bRF, bNF, bRS, bUID, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx, bS3,
                                bSP, bSN, bB0, bB1, bSI, bSW, bDb }, prm);
    dev->flush();
    SbRun r;
    r.pt = downloadBuffer<F4>(dev, bPt, CAP);
    r.np = downloadBuffer<F4>(dev, bNp, CAP);
    r.bc = downloadBuffer<F4>(dev, bBc, CAP);
    r.in = downloadBuffer<U4>(dev, bIn, CAP);
    r.by = downloadBuffer<uint32_t>(dev, bBy, CAP * 2);
    r.tb = downloadBuffer<uint32_t>(dev, bTb, CAP);
    r.ix = downloadBuffer<uint32_t>(dev, bIx, CAP);
    r.s32 = downloadBuffer<uint32_t>(dev, bS3, CAP);
    r.sbPt = downloadBuffer<F4>(dev, bSP, CAP);
    r.sbNp = downloadBuffer<F4>(dev, bSN, CAP);
    r.sbB0 = downloadBuffer<F4>(dev, bB0, CAP);
    r.sbB1 = downloadBuffer<F4>(dev, bB1, CAP);
    r.sbIn = downloadBuffer<U4>(dev, bSI, CAP);
    r.dbg = downloadBuffer<F4>(dev, bDb, n);
    r.femWc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
    r.sbWc = downloadBuffer<uint32_t>(dev, bSW, 1)[0];
    for (PxgBufferHandle h : { bC, bRF, bNF, bRS, bUID, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx, bS3,
                               bSP, bSN, bB0, bB1, bSI, bSW, bDb })
        dev->destroyBuffer(h);
    return r;
}

// ============================================================================
// sbTetPlane — EXACT
// ============================================================================
PXG_TEST(npSbPrim_tetPlane)
{
    const uint32_t NC = 1024, CAP = 3200, SBID = 4;
    std::mt19937 rng(1000701);
    std::vector<SbCase> cases(NC);
    std::vector<PairR> rfilter;
    size_t boundaryN = 0, filteredN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const F4 q = exactQuat(rng);
        const V3 tp = { float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f };
        cs.q = q;
        cs.p = { tp.x, tp.y, tp.z, 0 };
        const float cd = 0.25f + float(i % 4) * 0.25f;
        const float rd = 0.0625f * float(1 + i % 5);

        // choose plane-space x values per vert, then map back to world exactly:
        // v = q.rotate(planeSpace) + p (signed permutation => exact)
        float px[4];
        for (int k = 0; k < 4; ++k)
        {
            switch ((i + k) % 5)
            {
            case 0: px[k] = cd; ++boundaryN; break;                 // EXACTLY on the inclusive gate
            case 1: px[k] = cd - grid64(rng, 0.25f, 1.5f); break;   // accepted (may be negative = penetrating)
            case 2: px[k] = cd + grid64(rng, 0.25f, 2.0f); break;   // rejected
            case 3: px[k] = -grid64(rng, 0.5f, 2.0f); break;        // deep accept
            default: px[k] = cd + 0.015625f; break;                 // just rejected (exact)
            }
        }
        V3 verts[4];
        for (int k = 0; k < 4; ++k)
        {
            const V3 ps = { px[k], grid64(rng, -2, 2), grid64(rng, -2, 2) };
            // rotate by q (forward): use rotateInv of conjugate = rotate
            const F4 qc = { -q.x, -q.y, -q.z, q.w };
            verts[k] = add(quatRotateInv(qc, ps), tp);
        }
        cs.a = { verts[0].x, verts[0].y, verts[0].z, cd };
        cs.b = { verts[1].x, verts[1].y, verts[1].z, rd };
        cs.c = { verts[2].x, verts[2].y, verts[2].z, 0 };
        cs.d = { verts[3].x, verts[3].y, verts[3].z, 0 };
        const uint32_t rlo = 0x300u + i % 13, rhi = 1 + i % 3;
        cs.ids = { rlo, rhi, SBID, i };
        cs.ids2 = { (0x21u + i % 9) | ((i % 5 == 1) ? 0x80000000u : 0u), 0, 0, 0 };

        const uint64_t r64 = (uint64_t(rhi) << 32) | rlo;
        if (i % 9 == 2) { rfilter.push_back({ r64, uint64_t((SBID << 20) | i), 0, 0, 0, 0 }); ++filteredN; }
        else if (i % 9 == 5) rfilter.push_back({ r64, uint64_t((SBID << 20) | (i + 1)), 0, 0, 0, 0 }); // near miss
    }
    std::sort(rfilter.begin(), rfilter.end(), [](const PairR& a, const PairR& b)
              { return a.index0 != b.index0 ? a.index0 < b.index0 : a.index1 < b.index1; });

    // exact reference
    struct Ref { bool filtered; int hits[4]; float pen[4]; V3 n; };
    std::vector<Ref> ref(NC);
    uint32_t total = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        Ref& ro = ref[i];
        const uint64_t r64 = (uint64_t(cs.ids.y) << 32) | cs.ids.x;
        ro.filtered = findRigidRef(rfilter, r64, (SBID << 20) | i);
        if (ro.filtered) { ro.hits[0] = ro.hits[1] = ro.hits[2] = ro.hits[3] = 0; continue; }
        const V3 verts[4] = { v3(cs.a), v3(cs.b), v3(cs.c), v3(cs.d) };
        for (int k = 0; k < 4; ++k)
        {
            const V3 ps = quatRotateInv(cs.q, sub(verts[k], v3(cs.p)));
            ro.hits[k] = cs.a.w >= ps.x ? 1 : 0;
            ro.pen[k] = ps.x;
            if (ro.hits[k]) ++total;
        }
        ro.n = basisVector0(cs.q);
    }

    const SbRun r = runSb("sbTetPlane", cases, rfilter, {}, {}, {}, CAP, CAP);
    PXG_CHECK_MSG(r.femWc == total, "tetPlane: counter %u != %u", r.femWc, total);

    // slot map: pairInd1 unique per case; per-case runs contiguous in vert order
    std::vector<std::vector<uint32_t>> slots(NC);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.femWc, CAP); ++s)
    {
        const uint32_t caseIdx = r.in[s].z & 0xFFFFFu;
        if ((r.in[s].z >> 20) != SBID || caseIdx >= NC) { ++mm; continue; }
        slots[caseIdx].push_back(s);
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Ref& ro = ref[i];
        const uint32_t want = ro.filtered ? 0 : uint32_t(ro.hits[0] + ro.hits[1] + ro.hits[2] + ro.hits[3]);
        if ((r.dbg[i].x != 0.0f) != ro.filtered) ++mm;
        if (slots[i].size() != want) { ++mm; continue; }
        int vi = 0;
        for (uint32_t j = 0; j < want; ++j)
        {
            while (!ro.hits[vi]) ++vi;
            const uint32_t s = slots[i][j];
            if (j > 0 && s != slots[i][j - 1] + 1) ++mm; // contiguous run
            const V3 vert = vi == 0 ? v3(cases[i].a) : vi == 1 ? v3(cases[i].b) : vi == 2 ? v3(cases[i].c) : v3(cases[i].d);
            if (r.pt[s].x != vert.x || r.pt[s].y != vert.y || r.pt[s].z != vert.z || r.pt[s].w != cases[i].b.w) ++mm;
            if (r.np[s].x != ro.n.x || r.np[s].y != ro.n.y || r.np[s].z != ro.n.z || r.np[s].w != ro.pen[vi]) ++mm;
            if (r.in[s].x != cases[i].ids.x || r.in[s].y != cases[i].ids.y) ++mm;
            if (r.in[s].w != (cases[i].ids2.x & 0x7FFFFFFFu)) ++mm;
            if (r.by[s * 2] != cases[i].ids.x || r.by[s * 2 + 1] != cases[i].ids.y || r.tb[s] != cases[i].ids.x || r.ix[s] != s) ++mm;
            if (r.bc[s].x != -9.0f) ++mm; // barycentric buffer untouched
            ++vi;
        }
    }
    PXG_CHECK_MSG(mm == 0, "tetPlane: %zu mismatches", mm);
    PXG_CHECK_MSG(r.sbWc == 0 && r.sbPt[0].x == -9.0f, "tetPlane: softbody writer touched");
    PXG_CHECK_MSG(boundaryN > 500 && filteredN > 80, "tetPlane coverage thin (b=%zu f=%zu)", boundaryN, filteredN);

    // capacity: counter unchanged, stores gated
    const uint32_t TINY = total / 3;
    const SbRun rc = runSb("sbTetPlane", cases, rfilter, {}, {}, {}, TINY, CAP);
    size_t bad = rc.femWc != total ? 1 : 0;
    for (uint32_t s = TINY; s < CAP; ++s)
        if (rc.ix[s] != POISON || rc.pt[s].x != -9.0f || rc.in[s].x != INFOPOISON) ++bad;
    PXG_CHECK_MSG(bad == 0, "tetPlane cap: %zu leaks", bad);
}

// ============================================================================
// sbEmit — EXACT
// ============================================================================
PXG_TEST(npSbPrim_emit)
{
    const uint32_t NC = 512, CAP = 700;
    std::mt19937 rng(1000702);
    std::vector<SbCase> cases(NC);
    size_t altN = 0, flipN = 0, offN = 0, dropN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const V3 A = { float(int(rng() % 9)) - 4, float(int(rng() % 9)) - 4, float(int(rng() % 9)) - 4 };
        const V3 B = add(A, V3{ float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2 });
        const V3 n = { float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2 };
        const float penOffset = (i % 3 == 1) ? -float(1 + i % 4) * 0.25f : 0.0f; // -radius style
        if (penOffset != 0) ++offN;
        cs.a = { A.x, A.y, A.z, 0 };
        cs.b = { B.x, B.y, B.z, 0 };
        cs.c = { n.x, n.y, n.z, 0 };
        cs.d = { penOffset, 0.125f * float(1 + i % 4), 0, 0 };
        const bool intersect = (i % 6) != 4;
        if (!intersect) ++dropN;
        const bool flip = (i % 4) == 2;
        if (flip) ++flipN;
        const bool alt = (i % 2) == 1;
        if (alt) ++altN;
        cs.ids = { 0x1000u + i, 5 + i % 7, 9000 + i, (0x31u + i % 5) | ((i % 7 == 3) ? 0x80000000u : 0u) };
        cs.ids2 = { intersect ? 1u : 0u, flip ? 1u : 0u, alt ? 1u : 0u, 0 };
    }

    uint32_t total = 0;
    for (uint32_t i = 0; i < NC; ++i) if (cases[i].ids2.x) ++total;

    const SbRun r = runSb("sbEmit", cases, {}, {}, {}, {}, CAP, CAP);
    PXG_CHECK_MSG(r.femWc == total, "emit: counter %u != %u", r.femWc, total);

    std::vector<int> slotOf(NC, -1);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.femWc, CAP); ++s)
    {
        const uint32_t caseIdx = r.in[s].z - 9000;
        if (caseIdx >= NC || slotOf[caseIdx] != -1) { ++mm; continue; }
        slotOf[caseIdx] = (int)s;
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        if (!cs.ids2.x) { if (slotOf[i] != -1) ++mm; continue; }
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        const V3 A = v3(cs.a), B = v3(cs.b), n = v3(cs.c);
        const float pen = dot3(sub(B, A), n) + cs.d.x;
        const bool alt = cs.ids2.z != 0, flip = cs.ids2.y != 0;
        if (r.pt[s].x != B.x || r.pt[s].y != B.y || r.pt[s].z != B.z || r.pt[s].w != cs.d.y) ++mm;
        const V3 en = flip ? neg(n) : n;
        if (r.np[s].x != en.x || r.np[s].y != en.y || r.np[s].z != en.z || r.np[s].w != pen) ++mm;
        if (r.in[s].x != cs.ids.x || r.in[s].y != cs.ids.y || r.in[s].w != (cs.ids.w & 0x7FFFFFFFu)) ++mm;
        if (r.ix[s] != s) ++mm;
        if (alt)
        {
            // ONLY contactSortedByX written; the u64 lanes stay poisoned
            if (r.s32[s] != cs.ids.x) ++mm;
            if (r.by[s * 2] != POISON || r.by[s * 2 + 1] != POISON || r.tb[s] != POISON) ++mm;
        }
        else
        {
            if (r.by[s * 2] != cs.ids.x || r.by[s * 2 + 1] != cs.ids.y || r.tb[s] != cs.ids.x) ++mm;
            if (r.s32[s] != POISON) ++mm;
        }
        if (r.bc[s].x != -9.0f) ++mm; // no barycentric
    }
    PXG_CHECK_MSG(mm == 0, "emit: %zu mismatches", mm);
    PXG_CHECK_MSG(r.sbWc == 0 && r.sbPt[0].x == -9.0f, "emit: softbody writer touched");

    // capacity: UNGATED counter overshoots; sentinel drops all writes >= max
    const uint32_t TINY = total / 2;
    const SbRun rc = runSb("sbEmit", cases, {}, {}, {}, {}, TINY, CAP);
    size_t bad = rc.femWc != total ? 1 : 0; // counter intentionally overshoots the cap
    for (uint32_t s = TINY; s < CAP; ++s)
        if (rc.ix[s] != POISON || rc.pt[s].x != -9.0f || rc.in[s].x != INFOPOISON ||
            rc.s32[s] != POISON || rc.tb[s] != POISON ||
            rc.by[s * 2] != POISON || rc.by[s * 2 + 1] != POISON) ++bad;
    PXG_CHECK_MSG(bad == 0, "emit cap: %zu leaks", bad);
    PXG_CHECK_MSG(altN > 200 && flipN > 100 && offN > 140 && dropN > 60,
                  "emit coverage thin (a=%zu f=%zu o=%zu d=%zu)", altN, flipN, offN, dropN);
}

// ============================================================================
// shared tet fixture builder for sbParticle / sbClothVert
// ============================================================================
struct TetFix
{
    V3 a, b, c, d, p;
    uint32_t surfaceMask;
    bool wantInside, bigCd;
};
static TetFix makeTetFix(std::mt19937& rng, uint32_t i)
{
    TetFix f;
    std::uniform_real_distribution<float> uf(0.0f, 1.0f);
    const V3 base = { grid64(rng, -3, 3), grid64(rng, -3, 3), grid64(rng, -3, 3) };
    // well-conditioned tet on the 1/64 grid
    f.a = base;
    f.b = add(base, V3{ 1.0f + grid64(rng, 0, 0.5f), grid64(rng, 0, 0.25f), grid64(rng, 0, 0.25f) });
    f.c = add(base, V3{ grid64(rng, 0, 0.25f), 1.0f + grid64(rng, 0, 0.5f), grid64(rng, 0, 0.25f) });
    f.d = add(base, V3{ grid64(rng, 0, 0.25f), grid64(rng, 0, 0.25f), 1.0f + grid64(rng, 0, 0.5f) });
    f.surfaceMask = 1 + rng() % 15;
    f.wantInside = false;
    f.bigCd = false;
    const V3 cen = mul(add(add(f.a, f.b), add(f.c, f.d)), 0.25f);
    switch (i % 10)
    {
    case 0:
    case 5: // inside: blend toward center
        f.p = add(mul(cen, 0.75f), mul(f.a, 0.25f));
        f.wantInside = true;
        f.surfaceMask = 0xf;
        break;
    case 1: // outside near a face, within contact distance
        if ((i / 10) % 2 == 1)
        {
            // GRAZING: p a few 1e-5 below the interior of face abc — sqDist
            // lands under the 1e-8 rescue clause, which must replace the
            // normalized-noise direction by n2
            f.p = add(mul(add(add(f.a, f.b), f.c), 1.0f / 3.0f), V3{ 0, 0, -5e-5f - 3e-5f * uf(rng) });
            f.surfaceMask = 1;
            break;
        }
        f.p = add(cen, V3{ -0.4f - 0.2f * uf(rng), -0.4f, -0.4f });
        f.surfaceMask = 0xf;
        break;
    case 6: // FACE-OPPOSED: p straight below face abc (projection inside the
            // triangle) — v points up into the tet, and closestPtTetrahedron's
            // normals are flipped INWARD (toward the remaining vertex), so
            // dot(v, n2) ~ +1 feeds the v-direction / keep-n paths
        f.p = { cen.x, cen.y, f.a.z - 0.5f - 0.3f * uf(rng) };
        f.surfaceMask = 1;
        f.bigCd = true;
        break;
    case 2:
    case 7: // outside far (reject)
        f.p = add(cen, V3{ 3.0f + uf(rng), 3.0f, 3.0f });
        break;
    case 3: // ON-BOUNDARY inside: p exactly on edge ab (two products exactly 0,
            // inclusive >= keeps it INSIDE) while the hint mask excludes the
            // adjacent faces — the inside/outside paths then produce DIFFERENT
            // pen formulas, pinning the inclusive inside test
        if ((i / 10) % 2 == 0)
        {
            f.p = mul(add(f.a, f.b), 0.5f);
            f.surfaceMask = 8;
            f.bigCd = true;
            f.wantInside = true;
            break;
        }
        f.p = add(f.b, V3{ 0.3f + 0.2f * uf(rng), -0.3f, -0.3f });
        break;
    case 8: // outside near an edge/corner
        f.p = add(f.b, V3{ 0.3f + 0.2f * uf(rng), -0.3f, -0.3f });
        break;
    default: // exactly on vertex a: sqDist == 0 path (n -> n2)
        f.p = f.a;
        f.surfaceMask = 0xf;
        f.wantInside = true; // on the corner: all four products are exactly 0 => inside
        break;
    }
    return f;
}

// ============================================================================
// sbParticle — taint-based
// ============================================================================
PXG_TEST(npSbPrim_particle)
{
    const uint32_t NC = 1024, CAP = 1400, SBID = 6, SYSID = 3;
    std::mt19937 rng(1000703);
    std::vector<SbCase> cases(NC);
    std::vector<PairNR> nrf;
    // three particle sim buffers: runsum = exclusive starts, uniqueIds
    const std::vector<uint32_t> runsum = { 0, 400, 800 };
    const std::vector<uint32_t> bufUid = { 0x50, 0x60, 0x70 };
    std::vector<uint32_t> unsortedOf(NC);
    for (uint32_t i = 0; i < NC; ++i) unsortedOf[i] = (i * 733) % 1200; // scrambled mapping

    size_t filtN = 0, missN = 0, maskZeroN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const TetFix f = makeTetFix(rng, i);
        // on-edge-inside family: SMALL cd pins that the inside path has NO
        // cDistance gate (closest hint face lies far beyond cd, contact must
        // still be emitted); face-opposed family needs the big cd
        cs.a = { f.a.x, f.a.y, f.a.z, f.bigCd ? (f.wantInside ? 0.125f : 2.5f) : 0.5f + 0.25f * float(i % 3) };
        cs.b = { f.b.x, f.b.y, f.b.z, 0.0625f * float(1 + i % 4) };
        cs.c = { f.c.x, f.c.y, f.c.z, 0 };
        cs.d = { f.d.x, f.d.y, f.d.z, 0 };
        cs.p = { f.p.x, f.p.y, f.p.z, 0 };
        cs.ids = { SBID, i, SYSID, (i * 5 + 29) % NC /*sorted, decoupled from tetIdx*/ };
        cs.ids2 = { (i % 13 == 7) ? 0u : f.surfaceMask, unsortedOf[i], 0, 0 };
        if (i % 13 == 7) ++maskZeroN;

        const uint32_t uid = bufUid[findRangeRef(unsortedOf[i], runsum)];
        if (i % 7 == 2) { nrf.push_back({ uint64_t(unsortedOf[i]), (SBID << 20) | i, uid }); ++filtN; }
        else if (i % 11 == 3) { nrf.push_back({ uint64_t(unsortedOf[i]), (SBID << 20) | 0x000fffffu, uid }); ++filtN; }
        else if (i % 11 == 5) { nrf.push_back({ uint64_t(unsortedOf[i]), (SBID << 20) | i, uid ^ 0x5u }); ++missN; } // wrong uniqueId: must NOT match
    }
    std::sort(nrf.begin(), nrf.end(), [](const PairNR& a, const PairNR& b)
              { return a.index0 != b.index0 ? a.index0 < b.index0
                     : a.index1 != b.index1 ? a.index1 < b.index1 : a.index2 < b.index2; });

    struct Ref { bool skipped = false, wrote = false, inside = false; V3 p, n; float pen; F4 bary; };
    std::vector<Ref> ref(NC);
    std::vector<int> taint(NC, 0);
    size_t insideN = 0, outsideN = 0, rejN = 0, keepN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        Slack sl;
        Ref& ro = ref[i];
        const uint32_t surfaceMask = cs.ids2.x;
        const uint32_t uid = bufUid[findRangeRef(cs.ids2.y, runsum)];
        ro.skipped = surfaceMask == 0 ||
                     findNonRigidRef(nrf, cs.ids2.y, (SBID << 20) | 0x000fffffu, uid, false) ||
                     findNonRigidRef(nrf, cs.ids2.y, (SBID << 20) | i, uid, false);
        if (ro.skipped) continue;

        const V3 a = v3(cs.a), b = v3(cs.b), c = v3(cs.c), d = v3(cs.d), p = v3(cs.p);
        const F4 res = pointOutsideOfPlane4Ref(p, a, b, c, d);
        F4 bary;
        V3 closestP = {}, n2 = {};
        float sqDist = 3.402823466e38f;
        closestPtTetrahedronRef(p, surfaceMask, a, b, c, d, n2, closestP, sqDist, sl);
        computeTetBarycentricRef(a, b, c, d, closestP, bary);

        V3 n = neg(sub(p, closestP));
        float pen;
        const bool inside = res.x >= 0 && res.y >= 0 && res.z >= 0 && res.w >= 0;
        // exact sign products on grid inputs: no taint on the inside test itself
        if (inside)
        {
            pen = dot3(n, n2) - cs.b.w;
            n = n2;
        }
        else
        {
            const float cd2 = cs.a.w * cs.a.w;
            sl.note(sqDist - cd2, 2e-4f);
            if (sqDist > cd2) { ro.wrote = false; taint[i] = sl.near; if (!taint[i]) ++rejN; continue; }
            pen = pxNormalizeRef(n);
            sl.note(sqDist - 1e-8f, 1e-9f);
            sl.note(dot3(n, n2) - 0.8f, 2e-4f);
            if (sqDist < 1e-8f || dot3(n, n2) < 0.8f)
                n = n2;
            else
                ++keepN; // normalized contact direction kept (not replaced by n2)
            pen -= cs.b.w;
        }
        ro.wrote = true;
        ro.inside = inside;
        ro.p = p;
        ro.n = n;
        ro.pen = pen;
        ro.bary = bary;
        taint[i] = sl.near;
        if (!taint[i]) { if (inside) ++insideN; else ++outsideN; }
    }

    uint32_t sumClean = 0, numTainted = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (ref[i].skipped) continue;
        if (taint[i]) { ++numTainted; continue; }
        if (ref[i].wrote) ++sumClean;
    }

    const SbRun r = runSb("sbParticle", cases, {}, nrf, runsum, bufUid, CAP, CAP);
    std::vector<int> slotOf(NC, -1);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.femWc, CAP); ++s)
    {
        const uint32_t caseIdx = r.in[s].z & 0xFFFFFu;
        if ((r.in[s].z >> 20) != SBID || caseIdx >= NC || slotOf[caseIdx] != -1) { ++mm; continue; }
        slotOf[caseIdx] = (int)s;
        // contactByX = (SORTED index, systemId) u64; info.x/y the same; info.w untouched
        const uint32_t sorted = (caseIdx * 5 + 29) % NC;
        if (r.by[s * 2] != sorted || r.by[s * 2 + 1] != SYSID || r.tb[s] != sorted || r.ix[s] != s) ++mm;
        if (r.in[s].x != sorted || r.in[s].y != SYSID || r.in[s].w != INFOPOISON) ++mm;
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Ref& ro = ref[i];
        if (ro.skipped)
        {
            if (r.dbg[i].x != 1.0f || slotOf[i] != -1) ++mm;
            continue;
        }
        if (taint[i]) continue;
        if ((r.dbg[i].y != 0.0f) != ro.wrote) { ++mm; continue; }
        if (!ro.wrote) { if (slotOf[i] != -1) ++mm; continue; }
        if ((r.dbg[i].w != 0.0f) != ro.inside) ++mm; // inclusive inside test
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        if (r.pt[s].x != ro.p.x || r.pt[s].y != ro.p.y || r.pt[s].z != ro.p.z || r.pt[s].w != 0.0f) ++mm;
        if (!almostEqual(r.np[s].x, ro.n.x, 2e-3f, 2e-3f) || !almostEqual(r.np[s].y, ro.n.y, 2e-3f, 2e-3f) ||
            !almostEqual(r.np[s].z, ro.n.z, 2e-3f, 2e-3f) || !almostEqual(r.np[s].w, ro.pen, 2e-3f, 2e-3f)) ++mm;
        if (!almostEqual(r.bc[s].x, ro.bary.x, 5e-3f, 5e-3f) || !almostEqual(r.bc[s].y, ro.bary.y, 5e-3f, 5e-3f) ||
            !almostEqual(r.bc[s].z, ro.bary.z, 5e-3f, 5e-3f) || !almostEqual(r.bc[s].w, ro.bary.w, 5e-3f, 5e-3f)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "particle: %zu mismatches", mm);
    PXG_CHECK_MSG(r.sbWc == 0 && r.sbPt[0].x == -9.0f, "particle: softbody writer touched");
    PXG_CHECK_MSG(r.femWc >= sumClean && r.femWc <= sumClean + numTainted,
                  "particle: counter %u outside [%u, %u]", r.femWc, sumClean, sumClean + numTainted);
    PXG_CHECK_MSG(insideN > 200 && outsideN > 150 && rejN > 100 && filtN > 150 && missN > 50 && maskZeroN > 50 && keepN > 30,
                  "particle coverage thin (i=%zu o=%zu r=%zu f=%zu m=%zu z=%zu k=%zu)",
                  insideN, outsideN, rejN, filtN, missN, maskZeroN, keepN);
}

// ============================================================================
// sbClothVert — taint-based
// ============================================================================
PXG_TEST(npSbPrim_clothVert)
{
    const uint32_t NC = 1024, CAP = 1400, SBID = 9, CLOTHID = 5;
    std::mt19937 rng(1000704);
    std::vector<SbCase> cases(NC);
    std::vector<PairNR> nrf;
    size_t f1N = 0, f2N = 0, f3N = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const TetFix f = makeTetFix(rng, i);
        cs.a = { f.a.x, f.a.y, f.a.z, f.bigCd ? (f.wantInside ? 0.125f : 2.5f) : 0.5f + 0.25f * float(i % 3) };
        cs.b = { f.b.x, f.b.y, f.b.z, 0 };
        cs.c = { f.c.x, f.c.y, f.c.z, 0 };
        cs.d = { f.d.x, f.d.y, f.d.z, 0 };
        cs.p = { f.p.x, f.p.y, f.p.z, 0 };
        cs.ids = { SBID, i, CLOTHID, (i * 7 + 13) % NC /*vertIndex, decoupled from tetIdx*/ };
        cs.ids2 = { (i % 17 == 11) ? 0u : f.surfaceMask, 0, 0, 0 };

        const uint32_t vIdx = (i * 7 + 13) % NC;
        const uint32_t sbMask = (SBID << 20) | i;
        const uint32_t sbFull = (SBID << 20) | 0x000fffffu;
        const uint32_t clMask = (CLOTHID << 20) | vIdx;
        const uint32_t clFull = (CLOTHID << 20) | 0x000fffffu;
        if (i % 9 == 2) { nrf.push_back({ uint64_t(sbFull), clMask, 0 }); ++f1N; }
        else if (i % 9 == 4) { nrf.push_back({ uint64_t(sbMask), clFull, 0 }); ++f2N; }
        else if (i % 9 == 6) { nrf.push_back({ uint64_t(sbMask), clMask, 7 }); ++f3N; } // index2 != 0 but ignoreIndex2 accepts
        else if (i % 9 == 8) nrf.push_back({ uint64_t(sbMask), (CLOTHID << 20) | (vIdx + 1), 0 }); // near miss
    }
    std::sort(nrf.begin(), nrf.end(), [](const PairNR& a, const PairNR& b)
              { return a.index0 != b.index0 ? a.index0 < b.index0
                     : a.index1 != b.index1 ? a.index1 < b.index1 : a.index2 < b.index2; });

    struct Ref { bool skipped = false, wrote = false, inside = false; V3 cp, n; float pen; F4 bary; };
    std::vector<Ref> ref(NC);
    std::vector<int> taint(NC, 0);
    size_t insideN = 0, outsideN = 0, rejN = 0, vDirN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        Slack sl;
        Ref& ro = ref[i];
        const uint32_t sbMask = (SBID << 20) | i;
        const uint32_t sbFull = (SBID << 20) | 0x000fffffu;
        const uint32_t clMask = (CLOTHID << 20) | ((i * 7 + 13) % NC);
        const uint32_t clFull = (CLOTHID << 20) | 0x000fffffu;
        ro.skipped = findNonRigidRef(nrf, sbFull, clMask, 0, true) ||
                     findNonRigidRef(nrf, sbMask, clFull, 0, true) ||
                     findNonRigidRef(nrf, sbMask, clMask, 0, true);
        if (!ro.skipped && cs.ids2.x == 0) ro.skipped = true; // surfaceMask gate (dbg code 2)
        if (ro.skipped) continue;

        const V3 a = v3(cs.a), b = v3(cs.b), c = v3(cs.c), d = v3(cs.d), p = v3(cs.p);
        const F4 res = pointOutsideOfPlane4Ref(p, a, b, c, d);
        F4 bary;
        V3 closestP = {}, n2 = {};
        float sqDist = 3.402823466e38f;
        closestPtTetrahedronRef(p, cs.ids2.x, a, b, c, d, n2, closestP, sqDist, sl);
        computeTetBarycentricRef(a, b, c, d, closestP, bary);

        V3 v = sub(closestP, p);
        const float m = pxNormalizeRef(v);
        V3 n;
        float pen;
        const bool inside = res.x >= 0 && res.y >= 0 && res.z >= 0 && res.w >= 0;
        if (inside)
        {
            n = n2;
            pen = -m;
        }
        else
        {
            const float cd2 = cs.a.w * cs.a.w;
            sl.note(sqDist - cd2, 2e-4f);
            if (sqDist > cd2) { taint[i] = sl.near; if (!taint[i]) ++rejN; continue; }
            pen = m;
            sl.note(m - 1e-10f, 1e-11f);
            sl.note(dot3(v, n2) - 0.707f, 2e-4f);
            // unlike sbParticleCollision there is NO sqDist < 1e-8 rescue here:
            // a grazing contact keeps the normalized micro-vector as its
            // direction, which is ill-conditioned -- taint it
            sl.note(m, 2e-4f);
            const bool useV = m > 1e-10f && dot3(v, n2) > 0.707f;
            n = useV ? v : n2;
            if (useV) ++vDirN;
        }
        ro.wrote = true;
        ro.inside = inside;
        ro.cp = closestP;
        ro.n = n;
        ro.pen = pen;
        ro.bary = bary;
        taint[i] = sl.near;
        if (!taint[i]) { if (inside) ++insideN; else ++outsideN; }
    }

    uint32_t sumClean = 0, numTainted = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (ref[i].skipped) continue;
        if (taint[i]) { ++numTainted; continue; }
        if (ref[i].wrote) ++sumClean;
    }

    const SbRun r = runSb("sbClothVert", cases, {}, nrf, {}, {}, CAP, CAP);
    std::vector<int> slotOf(NC, -1);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.sbWc, CAP); ++s)
    {
        const uint32_t caseIdx = r.sbIn[s].z & 0xFFFFFu;
        if ((r.sbIn[s].z >> 20) != SBID || caseIdx >= NC || slotOf[caseIdx] != -1) { ++mm; continue; }
        slotOf[caseIdx] = (int)s;
        // pairInd0 = clothMask (VERTEX index, decoupled from tetIdx) widened
        // u32->u64 (high dword 0); auxInd untouched
        if (r.sbIn[s].x != ((CLOTHID << 20) | ((caseIdx * 7 + 13) % NC)) || r.sbIn[s].y != 0 || r.sbIn[s].w != INFOPOISON) ++mm;
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Ref& ro = ref[i];
        if (ro.skipped)
        {
            if (r.dbg[i].x == 0.0f || slotOf[i] != -1) ++mm;
            continue;
        }
        if (taint[i]) continue;
        if ((r.dbg[i].y != 0.0f) != ro.wrote) { ++mm; continue; }
        if (!ro.wrote) { if (slotOf[i] != -1) ++mm; continue; }
        if ((r.dbg[i].w != 0.0f) != ro.inside) ++mm; // inclusive inside test
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        if (!almostEqual(r.sbPt[s].x, ro.cp.x, 2e-3f, 2e-3f) || !almostEqual(r.sbPt[s].y, ro.cp.y, 2e-3f, 2e-3f) ||
            !almostEqual(r.sbPt[s].z, ro.cp.z, 2e-3f, 2e-3f) || r.sbPt[s].w != 0.0f) ++mm;
        if (!almostEqual(r.sbNp[s].x, ro.n.x, 2e-3f, 2e-3f) || !almostEqual(r.sbNp[s].y, ro.n.y, 2e-3f, 2e-3f) ||
            !almostEqual(r.sbNp[s].z, ro.n.z, 2e-3f, 2e-3f) || !almostEqual(r.sbNp[s].w, ro.pen, 2e-3f, 2e-3f)) ++mm;
        if (r.sbB0[s].x != 0.0f || r.sbB0[s].y != 0.0f || r.sbB0[s].z != 0.0f || r.sbB0[s].w != 1.0f) ++mm;
        if (!almostEqual(r.sbB1[s].x, ro.bary.x, 5e-3f, 5e-3f) || !almostEqual(r.sbB1[s].y, ro.bary.y, 5e-3f, 5e-3f) ||
            !almostEqual(r.sbB1[s].z, ro.bary.z, 5e-3f, 5e-3f) || !almostEqual(r.sbB1[s].w, ro.bary.w, 5e-3f, 5e-3f)) ++mm;
        // FEM writer buffers must be completely untouched by this entry
        if (r.pt[s].x != -9.0f || r.ix[s] != POISON) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "clothVert: %zu mismatches", mm);
    PXG_CHECK_MSG(r.femWc == 0 && r.pt[0].x == -9.0f && r.ix[0] == POISON, "clothVert: FEM writer touched");
    PXG_CHECK_MSG(r.sbWc >= sumClean && r.sbWc <= sumClean + numTainted,
                  "clothVert: counter %u outside [%u, %u]", r.sbWc, sumClean, sumClean + numTainted);
    PXG_CHECK_MSG(insideN > 200 && outsideN > 120 && rejN > 100 && vDirN > 40 &&
                  f1N > 80 && f2N > 80 && f3N > 80,
                  "clothVert coverage thin (i=%zu o=%zu r=%zu v=%zu f=%zu/%zu/%zu)",
                  insideN, outsideN, rejN, vDirN, f1N, f2N, f3N);
}

} // namespace
