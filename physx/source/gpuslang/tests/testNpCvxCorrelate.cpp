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

// Verification of npCvxCorrelate.slang: correlate (convexMeshCorrelate.cu
// L95-344) with its purgeEmptyTriangles prelude.  The CPU reference is an
// exact transcription of the CUDA warp code (serialized), including:
//   - the tempCounts encoding mismatch (purge stores FOUR nbContacts bits,
//     the patch loop reads THREE: index & 7);
//   - the pairIndex-vs-cmIndex output slot quirk of the zero-triangle path;
//   - the deepest-triangle scan (per-lane strided strict <, lowest-lane exact
//     min), acceptance strict > PATCH_ACCEPTANCE_EPS, rejected lanes keeping
//     their startIndex, the pull loop with last-index binarySearch over the
//     exclusive run-sums, c += popc(readMask), and
//     contactReduce<true,false,6,false> only at popc > 6;
//   - per-patch output at scan slots, empty patches included, and the
//     nbPatches>0-gated mRelativeTransform write.
// Near-ties (depth mins, acceptance dots at EPS, reduce argmin/argmax picks,
// cluster labeling) are margin-flagged SYMMETRICALLY and skipped; exact ties
// are deterministic on both sides and compared.

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
struct CorrIn
{
    F4 aToBq, aToBp;
    U4 hdr;
    F4 depth[16];
    F4 nrm[64];
    U4 nicIdx[16];
    U4 cbuf[16];
    F4 tempC[128];
};
struct CorrOut
{
    F4 contacts[72];
    U4 nbC;
    F4 relQ, relP;
    U4 misc;
    U4 tempOut[16];
    F4 depthOut[16];
};

const float FLTMAX = 3.4028235e38f;
const float EPS_ACC = 0.996194698f; // PATCH_ACCEPTANCE_EPS
const float TIE = 2e-4f;
const uint32_t CACHED = 0xFFFFFFFFu;
const uint32_t UNTOUCHED = 0xFFFFFFFFu;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }
static uint32_t u32At(const U4* a, uint32_t j) { return (&a[j / 4].x)[j & 3]; }
static float fAt(const F4* a, uint32_t j) { return (&a[j / 4].x)[j & 3]; }

static V3 qRot(const F4& q, const V3& v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}

static uint32_t getNb(uint32_t index) { return (index & 0x7FFFFFFFu) >> 27; }
static uint32_t getTri(uint32_t index) { return index & ((1u << 27) - 1); }

static uint32_t argMaxM(const float v[32], uint32_t mask, bool& marg)
{
    float best = -FLTMAX; uint32_t idx = 0;
    for (int j = 0; j < 32; ++j) if ((mask & (1u << j)) && v[j] > best) { best = v[j]; idx = (uint32_t)j; }
    for (int j = 0; j < 32; ++j) { const float d = std::fabs(v[j] - best); if ((mask & (1u << j)) && d != 0.f && d < TIE) marg = true; }
    return idx;
}
static uint32_t argMinM(const float v[32], uint32_t mask, bool& marg)
{
    float best = FLTMAX; uint32_t idx = 0;
    for (int j = 0; j < 32; ++j) if ((mask & (1u << j)) && v[j] < best) { best = v[j]; idx = (uint32_t)j; }
    for (int j = 0; j < 32; ++j) { const float d = std::fabs(v[j] - best); if ((mask & (1u << j)) && d != 0.f && d < TIE) marg = true; }
    return idx;
}

// contactReduce<true, false, 6, false>: exact transcription of the tranche-82
// port specialization (criterion = separation, distanceAdjustment = 0)
static uint32_t contactReduce6(const V3 ptRaw[32], const float sep[32], const V3& normal,
                               uint32_t allMask, float clusterBias, bool& marg)
{
    V3 point[32];
    for (int j = 0; j < 32; ++j) point[j] = sub(ptRaw[j], mul(normal, dot3(ptRaw[j], normal)));

    float vals[32];
    for (int j = 0; j < 32; ++j) vals[j] = sep[j];
    const uint32_t i0 = argMaxM(vals, allMask, marg);
    uint32_t mask = 1u << i0;

    for (int j = 0; j < 32; ++j) { const V3 d = sub(point[i0], point[j]); vals[j] = std::sqrt(dot3(d, d)); }
    const uint32_t i1 = argMaxM(vals, allMask & ~mask, marg);
    mask |= 1u << i1;

    const V3 dir = crossp(normal, sub(point[i1], point[i0]));
    float dval[32];
    for (int j = 0; j < 32; ++j) dval[j] = dot3(dir, sub(point[j], point[i0]));

    const uint32_t f = argMaxM(dval, allMask & ~mask, marg);
    mask |= 1u << f;
    const uint32_t g = argMinM(dval, allMask & ~mask, marg);
    mask |= 1u << g;

    float sepB[32];
    for (int j = 0; j < 32; ++j) sepB[j] = sep[j] - ((mask & (1u << j)) ? clusterBias : 0.f);

    int label[32], nbClusters = 0;
    {
        float tbest[32];
        for (int j = 0; j < 32; ++j) { tbest[j] = FLTMAX; label[j] = -1; }
        for (uint32_t m = mask; m != 0; ++nbClusters, m &= (m - 1))
        {
            uint32_t anchor = 0; while (!(m & (1u << anchor))) ++anchor; // firstbitlow
            for (int j = 0; j < 32; ++j)
            {
                const V3 dv = sub(point[j], point[anchor]);
                const float d = dot3(dv, dv);
                const float dd = std::fabs(d - tbest[j]);
                if (tbest[j] != FLTMAX && dd != 0.f && dd < 1e-4f) marg = true; // label near-tie
                if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
            }
        }
    }

    mask = 0;
    for (int i = 0; i < nbClusters; ++i)
    {
        uint32_t cluster = 0;
        for (int j = 0; j < 32; ++j) if (label[j] == i) cluster |= 1u << j;
        cluster &= allMask;
        if (cluster) mask |= 1u << argMinM(sepB, cluster, marg);
    }
    for (int i = nbClusters; i < 6; ++i)
        mask |= 1u << argMinM(sepB, allMask & ~mask, marg);
    return mask;
}

static uint32_t binarySearch32(const uint32_t data[32], uint32_t value)
{
    uint32_t left = 0, right = 32;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2;
        if (data[pos] <= value) left = pos + 1;
        else right = pos;
    }
    return left != 0 ? left - 1 : 0;
}

static void refCorrelate(const CorrIn& s, CorrOut& o, bool& marg, bool& reduceRan)
{
    for (int j = 0; j < 72; ++j) o.contacts[j] = F4{ 0, 0, 0, 0 };
    o.nbC = U4{ UNTOUCHED, UNTOUCHED, UNTOUCHED, UNTOUCHED }; // stale-buffer sentinel: the per-patch write must show
    o.relQ = F4{ 0, 0, 0, 0 }; o.relP = F4{ 0, 0, 0, 0 };
    o.misc = U4{ UNTOUCHED, UNTOUCHED, 0, 0 };

    float depth[64];
    uint32_t tc[64], nicSlot[64];
    for (uint32_t j = 0; j < 64; ++j) { depth[j] = fAt(s.depth, j); tc[j] = 0; nicSlot[j] = j; }

    const uint32_t count = s.hdr.x, start = s.hdr.y, cmIndex = s.hdr.z;
    const float clusterBias = s.aToBp.w;
    uint32_t nbTriangles = 0;

    if (count != CACHED)
    {
        // purge
        float dNew[64]; uint32_t tcNew[64], srcNew[64];
        for (int j = 0; j < 64; ++j) { dNew[j] = 0; tcNew[j] = 0; srcNew[j] = 0; }
        for (uint32_t loaded = 0; loaded < count; loaded += 32)
        {
            const uint32_t nbToLoad = std::min(count - loaded, 32u);
            uint32_t validBits = 0;
            for (uint32_t L = 0; L < nbToLoad; ++L)
            {
                const float d = depth[start + loaded + L];
                const uint32_t nb = getNb(u32At(s.nicIdx, start + loaded + L));
                if (d != FLTMAX && nb > 0) validBits |= 1u << L;
            }
            for (uint32_t L = 0; L < 32; ++L)
                if (validBits & (1u << L))
                {
                    const uint32_t dest = popc(validBits & ((1u << L) - 1)) + nbTriangles;
                    const uint32_t src = loaded + L;
                    dNew[dest] = depth[start + src];
                    tcNew[dest] = (src << 4) | getNb(u32At(s.nicIdx, start + src));
                    srcNew[dest] = start + src;
                }
            nbTriangles += popc(validBits);
        }
        for (uint32_t j = 0; j < nbTriangles; ++j)
        {
            depth[start + j] = dNew[j];
            tc[start + j] = tcNew[j];
            nicSlot[start + j] = srcNew[j];
        }

        if (nbTriangles == 0)
        {
            o.misc.x = 0; // outBuffer[pairIndex] slot (pairIndex modeled as slot 0)
        }
        else
        {
            uint32_t nbPatches = 0, remaining = nbTriangles;
            for (; nbPatches < 4 && remaining != 0; nbPatches++)
            {
                // deepest triangle
                float laneBest[32]; uint32_t laneIdx[32];
                for (int L = 0; L < 32; ++L) { laneBest[L] = FLTMAX; laneIdx[L] = 0; }
                for (uint32_t i = 0; i < nbTriangles; ++i)
                {
                    const int L = i & 31;
                    if (tc[start + i])
                    {
                        const float d = depth[start + i];
                        const float dd = std::fabs(d - laneBest[L]);
                        if (laneBest[L] != FLTMAX && dd != 0.f && dd < TIE) marg = true;
                        if (d < laneBest[L]) { laneBest[L] = d; laneIdx[L] = start + i; }
                    }
                }
                float minD = FLTMAX;
                for (int L = 0; L < 32; ++L) minD = std::fmin(minD, laneBest[L]);
                uint32_t i0 = 0;
                for (int L = 0; L < 32; ++L) if (laneBest[L] == minD) { i0 = (uint32_t)L; break; }
                for (int L = 0; L < 32; ++L)
                { const float dd = std::fabs(laneBest[L] - minD); if (laneBest[L] != FLTMAX && dd != 0.f && dd < TIE) marg = true; }
                const uint32_t bestIndex = laneIdx[i0];
                const V3 bestNormal = v3(s.nrm[nicSlot[bestIndex]]);

                uint32_t curMask = 0;
                V3 pA[32]; float sepL[32]; uint32_t triL[32]; bool hasC[32];
                for (int L = 0; L < 32; ++L) { pA[L] = V3{ 0, 0, 0 }; sepL[L] = 0; triL[L] = 0; hasC[L] = false; }

                for (uint32_t i = 0; i < nbTriangles; i += 32)
                {
                    V3 nrmL[32]; uint32_t runs[32], startS[32], inds[32], nbCts[32];
                    for (uint32_t L = 0; L < 32; ++L)
                    {
                        nrmL[L] = V3{ 0, 0, 0 }; inds[L] = 0; nbCts[L] = 0; startS[L] = 0;
                        if ((i + L) < nbTriangles)
                        {
                            const uint32_t index = tc[start + i + L];
                            nbCts[L] = index & 7;
                            startS[L] = index >> 4;
                            if (nbCts[L])
                            {
                                const uint32_t slot = nicSlot[start + i + L];
                                nrmL[L] = v3(s.nrm[slot]);
                                inds[L] = getTri(u32At(s.nicIdx, slot));
                            }
                        }
                    }
                    uint32_t acceptMask = 0;
                    for (int L = 0; L < 32; ++L)
                    {
                        const float dp = dot3(nrmL[L], bestNormal);
                        const float de = std::fabs(dp - EPS_ACC);
                        if (de != 0.f && de < TIE) marg = true; // exact equality is deterministic (rejected by strict >)
                        const bool accept = dp > EPS_ACC;
                        if (!accept) nbCts[L] = 0;
                        else acceptMask |= 1u << L;
                    }
                    remaining -= popc(acceptMask);

                    uint32_t run = 0;
                    for (int L = 0; L < 32; ++L) { runs[L] = run; run += nbCts[L]; }
                    const uint32_t total = run;

                    for (uint32_t c = 0; c < total;)
                    {
                        const uint32_t readMask = ~curMask;
                        for (uint32_t L = 0; L < 32; ++L)
                        {
                            const bool needs = (readMask & (1u << L)) != 0;
                            const uint32_t readIndex = c + popc(readMask & ((1u << L) - 1));
                            if (needs && readIndex < total)
                            {
                                const uint32_t idx = binarySearch32(runs, readIndex);
                                const uint32_t offset = readIndex - runs[idx];
                                const uint32_t startIdx = startS[idx] + start;
                                const uint32_t cs = u32At(s.cbuf, startIdx);
                                const F4 v = s.tempC[cs + offset];
                                pA[L] = v3(v); sepL[L] = v.w; triL[L] = inds[idx]; hasC[L] = true;
                            }
                        }
                        uint32_t nm = 0;
                        for (int L = 0; L < 32; ++L) if (hasC[L]) nm |= 1u << L;
                        curMask = nm;
                        c += popc(readMask);
                        if (popc(curMask) > 6)
                        {
                            reduceRan = true;
                            curMask = contactReduce6(pA, sepL, bestNormal, curMask, clusterBias, marg);
                        }
                        for (int L = 0; L < 32; ++L) hasC[L] = (curMask & (1u << L)) != 0;
                    }

                    for (uint32_t L = 0; L < 32; ++L)
                        if (acceptMask & (1u << L)) tc[start + i + L] = 0;
                }

                const uint32_t nb = popc(curMask);
                (&o.nbC.x)[nbPatches] = nb;
                for (uint32_t L = 0; L < 32; ++L)
                    if (curMask & (1u << L))
                    {
                        const uint32_t slot = popc(curMask & ((1u << L) - 1));
                        const V3 nor = qRot(s.aToBq, bestNormal);
                        const V3 pointB = add(add(qRot(s.aToBq, pA[L]), v3(s.aToBp)), mul(nor, sepL[L]));
                        const uint32_t base = (nbPatches * 6 + slot) * 3;
                        o.contacts[base + 0] = F4{ pA[L].x, pA[L].y, pA[L].z, sepL[L] };
                        float triF; std::memcpy(&triF, &triL[L], 4);
                        o.contacts[base + 1] = F4{ pointB.x, pointB.y, pointB.z, triF };
                        o.contacts[base + 2] = F4{ -nor.x, -nor.y, -nor.z, 0.f };
                    }
            }

            if (nbPatches > 0)
            {
                o.relQ = s.aToBq;
                o.relP = F4{ s.aToBp.x, s.aToBp.y, s.aToBp.z, 0.f };
            }
            if (cmIndex == 0) o.misc.x = nbPatches; else o.misc.y = nbPatches;
            o.misc.w = remaining;
        }
    }

    o.misc.z = nbTriangles;
    for (int j = 0; j < 16; ++j)
    {
        o.tempOut[j] = U4{ tc[j * 4], tc[j * 4 + 1], tc[j * 4 + 2], tc[j * 4 + 3] };
        o.depthOut[j] = F4{ depth[j * 4], depth[j * 4 + 1], depth[j * 4 + 2], depth[j * 4 + 3] };
    }
}

static bool eqF(float a, float b) { return almostEqual(a, b, 2e-4f, 2e-5f); }
static bool bitEq(float a, float b) { uint32_t u, v; std::memcpy(&u, &a, 4); std::memcpy(&v, &b, 4); return u == v; }

PXG_TEST(npCvxCorrelate_correlate)
{
    std::mt19937 rng(156001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<CorrIn> in(N);
    std::vector<CorrOut> ref(N);
    std::vector<uint8_t> skip(N, 0);
    size_t nSkip = 0, cCached = 0, cZeroTri = 0, cCm1 = 0, cReduce = 0, cMultiPatch = 0, cQuirk8 = 0, cTwoBlocks = 0, cEmptyPatch = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CorrIn s{};
        s.aToBq = randQuat();
        s.aToBp = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.01f, 0.1f) };

        const bool cached = (i % 29 == 0);
        uint32_t start = (i % 3 == 0) ? 8u : 0u;
        uint32_t count = 1 + rng() % 48;
        if (i % 5 == 2) { start = 0; count = 56 + rng() % 8; } // dense: purge survivors span two 32-lane blocks
        if (start + count > 64) count = 64 - start;
        const uint32_t cmIndex = (i % 3 == 1) ? 1u : 0u;
        s.hdr = U4{ cached ? CACHED : count, start, cmIndex, 0 };

        // 2-3 well-separated cluster normals; triangle normals are bitwise
        // copies (exact accept/reject, no near-EPS dots except perturbed tests)
        V3 cn[3];
        cn[0] = unit(V3{ 1, rf(-0.2f, 0.2f), rf(-0.2f, 0.2f) });
        cn[1] = unit(V3{ rf(-0.2f, 0.2f), 1, rf(-0.2f, 0.2f) });
        cn[2] = unit(V3{ rf(-0.2f, 0.2f), rf(-0.2f, 0.2f), -1 });

        for (uint32_t jj = 0; jj < 64; ++jj)
        {
            V3 n = cn[rng() % 3];
            if (i % 13 == 5 && (rng() % 8 == 0))
            {
                // perturbed normals drive near-EPS dots -> symmetric marg skip
                n = unit(add(n, V3{ rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), rf(-0.1f, 0.1f) }));
            }
            s.nrm[jj] = F4{ n.x, n.y, n.z, 0 };

            uint32_t nb = rng() % 8;                       // 0..7
            if (i % 17 == 4 && (jj & 7) == 3) nb = 8 + rng() % 8; // 4-bit purge vs 3-bit read quirk
            const uint32_t tri = rng() & 0x07FFFFFFu;
            const uint32_t defBit = (rng() & 1u) << 31;
            (&s.nicIdx[jj / 4].x)[jj & 3] = defBit | (nb << 27) | tri;

            (&s.cbuf[jj / 4].x)[jj & 3] = rng() % 112;     // contact pool start (112 + 15 < 128)
            (&s.depth[jj / 4].x)[jj & 3] = (rng() % 4 == 0) ? FLTMAX : rf(-1.f, 1.f);
        }
        for (int jj = 0; jj < 128; ++jj)
            s.tempC[jj] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-0.5f, 0.5f) };
        if (i % 23 == 7) // force the zero-triangle early-out
            for (uint32_t jj = 0; jj < 64; ++jj) (&s.depth[jj / 4].x)[jj & 3] = FLTMAX;
        else if ((i % 31 == 6 || i % 31 == 17) && !cached)
        {
            // exact depth ties pinning the deterministic tie-breaks.  All
            // slots valid (identity purge, depths in [-1,1], nb 1..7), then
            // the tied pair gets the strict global minimum and DIFFERENT
            // cluster normals, so a wrong tie-break changes the first patch.
            start = 0; count = 56;
            s.hdr = U4{ count, 0, cmIndex, 0 };
            for (uint32_t jj = 0; jj < 64; ++jj)
            {
                float& d = (&s.depth[jj / 4].x)[jj & 3];
                if (d == FLTMAX) d = rf(-1.f, 1.f);
                uint32_t& w = (&s.nicIdx[jj / 4].x)[jj & 3];
                const uint32_t nb = getNb(w);
                if (nb == 0 || nb > 7) w = (w & ~(15u << 27)) | ((1 + rng() % 7) << 27);
            }
            if (i % 31 == 6)
            {
                // slots 2 and 37: tie ACROSS lanes (2 vs 5) -- lowest lane wins the warp min
                (&s.depth[0].x)[2] = -2.0f; s.depth[9].y = -2.0f;
                s.nrm[2] = F4{ cn[0].x, cn[0].y, cn[0].z, 0 };
                s.nrm[37] = F4{ cn[1].x, cn[1].y, cn[1].z, 0 };
            }
            else
            {
                // slots 3 and 35: tie WITHIN lane 3 -- first strided index wins the strict <
                (&s.depth[0].x)[3] = -2.0f; s.depth[8].w = -2.0f;
                s.nrm[3] = F4{ cn[1].x, cn[1].y, cn[1].z, 0 };
                s.nrm[35] = F4{ cn[2].x, cn[2].y, cn[2].z, 0 };
            }
        }
        else if (i % 41 == 13 && !cached)
        {
            // exact-EPS acceptance boundary: bestNormal is bitwise (1,0,0), so
            // slot 5's dot is exactly its normal.x == PATCH_ACCEPTANCE_EPS.
            // Strict > must REJECT it (dd == 0 is not a near-tie: deterministic
            // on both sides).
            start = 0; count = 24;
            s.hdr = U4{ count, 0, cmIndex, 0 };
            const V3 ex{ 1.f, 0.f, 0.f };
            for (uint32_t jj = 0; jj < 64; ++jj)
            {
                float& d = (&s.depth[jj / 4].x)[jj & 3];
                if (d == FLTMAX) d = rf(-1.f, 1.f);
                uint32_t& w = (&s.nicIdx[jj / 4].x)[jj & 3];
                const uint32_t nb = getNb(w);
                if (nb == 0 || nb > 7) w = (w & ~(15u << 27)) | ((1 + rng() % 7) << 27);
                if (rng() % 2) s.nrm[jj] = F4{ ex.x, ex.y, ex.z, 0 };
            }
            (&s.depth[0].x)[2] = -2.0f;                         // global min -> bestNormal = (1,0,0)
            s.nrm[2] = F4{ ex.x, ex.y, ex.z, 0 };
            s.nrm[5] = F4{ EPS_ACC, 0.08715574f, 0.f, 0 };      // dot == EPS_ACC bitwise
            s.depth[1].y = -1.0f;                                // slot 5 stays off the min
        }
        else if (i % 41 == 27 && !cached)
        {
            // exactly-6-held state with duplicate contact points: two 3-contact
            // triangles sharing one normal AND one contact pool range.  No
            // reduce may run at exactly 6 (strict >); a reduce here would
            // collapse the duplicate anchors' clusters and drop contacts.
            start = 0; count = 2;
            s.hdr = U4{ count, 0, cmIndex, 0 };
            (&s.depth[0].x)[0] = -0.5f; (&s.depth[0].x)[1] = -0.6f;
            s.nrm[0] = F4{ cn[0].x, cn[0].y, cn[0].z, 0 };
            s.nrm[1] = F4{ cn[0].x, cn[0].y, cn[0].z, 0 };
            for (int jj = 0; jj < 2; ++jj)
            {
                uint32_t& w = (&s.nicIdx[0].x)[jj];
                w = (w & ~(15u << 27) & 0x87FFFFFFu) | (3u << 27);
                (&s.cbuf[0].x)[jj] = 0;                          // identical pool base
            }
        }

        in[i] = s;
        bool marg = false, reduceRan = false;
        refCorrelate(s, ref[i], marg, reduceRan);
        skip[i] = marg ? 1 : 0;
        if (marg) { ++nSkip; continue; }
        if (cached) ++cCached;
        else if (ref[i].misc.z == 0) ++cZeroTri;
        if (cmIndex == 1) ++cCm1;
        if (ref[i].misc.z > 32) ++cTwoBlocks;
        if (reduceRan) ++cReduce;
        // exercised 4-bit/3-bit quirk: a purged slot whose stored count bits
        // exceed 7 is never accepted/cleared, so it survives in the final state
        for (int jj = 0; jj < 64; ++jj)
            if ((u32At(ref[i].tempOut, (uint32_t)jj) & 15u) > 7u) { ++cQuirk8; break; }
        const uint32_t nbMan = (cmIndex == 0) ? ref[i].misc.x : ref[i].misc.y;
        if (!cached && ref[i].misc.z > 0 && nbMan != UNTOUCHED)
        {
            if (nbMan > 1) ++cMultiPatch;
            for (uint32_t p = 0; p < nbMan && p < 4; ++p)
            {
                const uint32_t nb = (&ref[i].nbC.x)[p];
                if (nb == 0) ++cEmptyPatch;
            }
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CorrOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct CorrParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCorrelate.spv", "cvxCorrelate",
             { (uint32_t)sizeof(CorrIn), (uint32_t)sizeof(CorrOut) }, sizeof(CorrParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CorrOut>(dev, bOut, N);
    size_t mmC = 0, mmN = 0, mmR = 0, mmM = 0, mmT = 0, mmD = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        for (int j = 0; j < 72; ++j)
        {
            const F4& rc = ref[i].contacts[j]; const F4& gc = g[i].contacts[j];
            const bool triWord = (j % 3) == 1; // w of the second float4 is a bit-cast triIndex
            if (!eqF(rc.x, gc.x) || !eqF(rc.y, gc.y) || !eqF(rc.z, gc.z) ||
                (triWord ? !bitEq(rc.w, gc.w) : !eqF(rc.w, gc.w))) { ++mmC; break; }
        }
        const U4& rn = ref[i].nbC; const U4& gn = g[i].nbC;
        if (rn.x != gn.x || rn.y != gn.y || rn.z != gn.z || rn.w != gn.w) ++mmN;
        if (!bitEq(ref[i].relQ.x, g[i].relQ.x) || !bitEq(ref[i].relQ.y, g[i].relQ.y) ||
            !bitEq(ref[i].relQ.z, g[i].relQ.z) || !bitEq(ref[i].relQ.w, g[i].relQ.w) ||
            !bitEq(ref[i].relP.x, g[i].relP.x) || !bitEq(ref[i].relP.y, g[i].relP.y) ||
            !bitEq(ref[i].relP.z, g[i].relP.z)) ++mmR;
        const U4& rm = ref[i].misc; const U4& gm = g[i].misc;
        if (rm.x != gm.x || rm.y != gm.y || rm.z != gm.z || rm.w != gm.w) ++mmM;
        for (int j = 0; j < 16; ++j)
        {
            const U4& rt = ref[i].tempOut[j]; const U4& gt = g[i].tempOut[j];
            if (rt.x != gt.x || rt.y != gt.y || rt.z != gt.z || rt.w != gt.w) { ++mmT; break; }
        }
        for (int j = 0; j < 16; ++j)
        {
            const F4& rd = ref[i].depthOut[j]; const F4& gd = g[i].depthOut[j];
            if (!bitEq(rd.x, gd.x) || !bitEq(rd.y, gd.y) || !bitEq(rd.z, gd.z) || !bitEq(rd.w, gd.w)) { ++mmD; break; }
        }
    }
    PXG_CHECK_MSG(mmC == 0, "correlate: %zu contact mismatches", mmC);
    PXG_CHECK_MSG(mmN == 0, "correlate: %zu nbContacts mismatches", mmN);
    PXG_CHECK_MSG(mmR == 0, "correlate: %zu relTransform mismatches", mmR);
    PXG_CHECK_MSG(mmM == 0, "correlate: %zu misc mismatches", mmM);
    PXG_CHECK_MSG(mmT == 0, "correlate: %zu tempCounts mismatches", mmT);
    PXG_CHECK_MSG(mmD == 0, "correlate: %zu depth mismatches", mmD);
    PXG_CHECK_MSG(nSkip < N / 2, "correlate: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cCached > N / 50 && cZeroTri > N / 50 && cCm1 > N / 5 && cReduce > N / 60 &&
                  cMultiPatch > N / 10 && cQuirk8 > N / 150 && cTwoBlocks > N / 30 && cEmptyPatch > 0,
                  "correlate: coverage thin (cached %zu zeroTri %zu cm1 %zu reduce %zu multi %zu quirk8 %zu blocks2 %zu emptyP %zu)",
                  cCached, cZeroTri, cCm1, cReduce, cMultiPatch, cQuirk8, cTwoBlocks, cEmptyPatch);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}


// ---------------------------------------------------------------------------
// cvxTriCorrelateBda -- the SAME core on the engine BDA ABI, as a real 32-lane
// warp writing real PxgPersistentContactMultiManifold records.
//
// Driven against the SAME refCorrelate above, which is the point: the scalar
// shader proves the algorithm, so a divergence here is attributable to the BDA
// plumbing and the warp emulation and nothing else.
//
// THE LAYOUT MAPPING.  refCorrelate works on one self-contained record with
// 64 depth/nic/cbuf slots and 128 temp contacts.  The engine's buffers are shared
// across contact managers and indexed globally, so every per-case index is
// rebased: pair i owns depth/nic/tempCounts slots [i*64, i*64+64), contact-pool
// entries [i*128, i*128+128), and its ConvexMeshPair carries `startIndex = i*64 +
// localStart`.  The cbuf VALUES are rebased too, since they index the contact pool.
//
// THE TWO OUTPUT SLOTS.  The zero-triangle early-out writes mNbManifolds of
// outBuffer[pairIndex] while everything else writes outBuffer[cmIndex] -- so the
// oracle models "slot 0" (pairIndex) and "slot cmIndex".  Here pairIndex is the
// case index, and the engine cmIndex is set to the SAME slot when the oracle's
// cmIndex is 0 (the aliasing case the engine really hits) and to a distinct slot
// N+i when it is 1.  Every manifold word starts at 0xFFFFFFFF, so "not written"
// is observable rather than assumed.
// ---------------------------------------------------------------------------

struct CcPush
{
    uint64_t pairs, depthsPtr, normalAndIndexPtr, contactsPtr, output, cmOutputs;
    uint64_t tempCountAndOffsetBuffer;
    uint32_t numPairs;
    float clusterBias;
    uint64_t tempConvexTriContacts, pTempContactIndex;
};

const uint32_t MANIFOLD_WORDS = 1216 / 4;
const uint32_t NBC_W = 1152 / 4;    // mNbContacts[0]
const uint32_t RELXF_W = 1168 / 4;  // mRelativeTransform.q
const uint32_t NBMAN_W = 1200 / 4;  // mNbManifolds

PXG_TEST(cvxTriCorrelate)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::mt19937 rng(90210017);
    const uint32_t N = 240;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f);
                            return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<CorrIn> in(N);
    std::vector<CorrOut> ref(N);
    std::vector<uint8_t> skip(N, 0);
    size_t nSkip = 0, cCached = 0, cZeroTri = 0, cCm1 = 0, cReduce = 0, cMultiPatch = 0,
           cQuirk8 = 0, cTwoBlocks = 0, cTie = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CorrIn s{};
        s.aToBq = randQuat();
        s.aToBp = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.01f, 0.1f) };

        const bool cached = (i % 29 == 0);
        uint32_t start = (i % 3 == 0) ? 8u : 0u;
        uint32_t count = 1 + rng() % 48;
        if (i % 5 == 2) { start = 0; count = 56 + rng() % 8; }   // survivors span two 32-lane blocks
        if (start + count > 64) count = 64 - start;
        const uint32_t cmIndex = (i % 3 == 1) ? 1u : 0u;
        s.hdr = U4{ cached ? CACHED : count, start, cmIndex, 0 };

        V3 cn[3];
        cn[0] = unit(V3{ 1, rf(-0.2f, 0.2f), rf(-0.2f, 0.2f) });
        cn[1] = unit(V3{ rf(-0.2f, 0.2f), 1, rf(-0.2f, 0.2f) });
        cn[2] = unit(V3{ rf(-0.2f, 0.2f), rf(-0.2f, 0.2f), -1 });

        for (uint32_t jj = 0; jj < 64; ++jj)
        {
            const V3 n = cn[rng() % 3];
            s.nrm[jj] = F4{ n.x, n.y, n.z, 0 };
            uint32_t nb = rng() % 8;
            if (i % 17 == 4 && (jj & 7) == 3) nb = 8 + rng() % 8;   // the 4-bit/3-bit quirk
            (&s.nicIdx[jj / 4].x)[jj & 3] = ((rng() & 1u) << 31) | (nb << 27) | (rng() & 0x07FFFFFFu);
            (&s.cbuf[jj / 4].x)[jj & 3] = rng() % 112;              // 112 + 15 < 128
            (&s.depth[jj / 4].x)[jj & 3] = (rng() % 4 == 0) ? FLTMAX : rf(-1.f, 1.f);
        }
        for (int jj = 0; jj < 128; ++jj)
            s.tempC[jj] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-0.5f, 0.5f) };

        if (i % 23 == 7)
        {
            for (uint32_t jj = 0; jj < 64; ++jj) (&s.depth[jj / 4].x)[jj & 3] = FLTMAX;  // zero-triangle
        }
        else if ((i % 31 == 6 || i % 31 == 17) && !cached)
        {
            // The two DEPTH TIES the warp decides and a scalar loop cannot: across
            // lanes (lowest lane wins the exact min) and within one lane (the first
            // strided index wins the strict <).  Both pick a different bestNormal,
            // so a wrong tie-break changes the whole first patch.
            start = 0; count = 56;
            s.hdr = U4{ count, 0, cmIndex, 0 };
            for (uint32_t jj = 0; jj < 64; ++jj)
            {
                float& d = (&s.depth[jj / 4].x)[jj & 3];
                if (d == FLTMAX) d = rf(-1.f, 1.f);
                uint32_t& w = (&s.nicIdx[jj / 4].x)[jj & 3];
                const uint32_t nb = getNb(w);
                if (nb == 0 || nb > 7) w = (w & ~(15u << 27)) | ((1 + rng() % 7) << 27);
            }
            if (i % 31 == 6)
            {
                (&s.depth[0].x)[2] = -2.0f; s.depth[9].y = -2.0f;   // slots 2 and 37: lanes 2 vs 5
                s.nrm[2] = F4{ cn[0].x, cn[0].y, cn[0].z, 0 };
                s.nrm[37] = F4{ cn[1].x, cn[1].y, cn[1].z, 0 };
            }
            else
            {
                (&s.depth[0].x)[3] = -2.0f; s.depth[8].w = -2.0f;   // slots 3 and 35: both lane 3
                s.nrm[3] = F4{ cn[1].x, cn[1].y, cn[1].z, 0 };
                s.nrm[35] = F4{ cn[2].x, cn[2].y, cn[2].z, 0 };
            }
            ++cTie;
        }
        else if (i % 41 == 13 && !cached)
        {
            // The acceptance test sitting EXACTLY on PATCH_ACCEPTANCE_EPS.  Strict
            // `>` must reject; the dot is bitwise equal on both sides, so this is a
            // decision, not a near-tie.
            start = 0; count = 24;
            s.hdr = U4{ count, 0, cmIndex, 0 };
            const V3 ex{ 1.f, 0.f, 0.f };
            for (uint32_t jj = 0; jj < 64; ++jj)
            {
                float& d = (&s.depth[jj / 4].x)[jj & 3];
                if (d == FLTMAX) d = rf(-1.f, 1.f);
                uint32_t& w = (&s.nicIdx[jj / 4].x)[jj & 3];
                const uint32_t nb = getNb(w);
                if (nb == 0 || nb > 7) w = (w & ~(15u << 27)) | ((1 + rng() % 7) << 27);
                if (rng() % 2) s.nrm[jj] = F4{ ex.x, ex.y, ex.z, 0 };
            }
            (&s.depth[0].x)[2] = -2.0f;
            s.nrm[2] = F4{ ex.x, ex.y, ex.z, 0 };
            s.nrm[5] = F4{ EPS_ACC, 0.08715574f, 0.f, 0 };
            s.depth[1].y = -1.0f;
        }

        in[i] = s;
        bool marg = false, reduceRan = false;
        refCorrelate(s, ref[i], marg, reduceRan);
        skip[i] = marg ? 1 : 0;
        if (marg) { ++nSkip; continue; }
        if (cached) ++cCached;
        else if (ref[i].misc.z == 0) ++cZeroTri;
        if (cmIndex == 1) ++cCm1;
        if (ref[i].misc.z > 32) ++cTwoBlocks;
        if (reduceRan) ++cReduce;
        for (int jj = 0; jj < 64; ++jj)
            if ((u32At(ref[i].tempOut, (uint32_t)jj) & 15u) > 7u) { ++cQuirk8; break; }
        const uint32_t nbMan = (cmIndex == 0) ? ref[i].misc.x : ref[i].misc.y;
        if (!cached && ref[i].misc.z > 0 && nbMan != UNTOUCHED && nbMan > 1) ++cMultiPatch;
    }

    // ---- lay the cases out as real engine buffers ---------------------------
    std::vector<uint32_t> pairBuf(N * 14, 0u);
    std::vector<float> depthBuf(N * 64, 0.0f);
    std::vector<float> nicBuf(N * 64 * 4, 0.0f);
    std::vector<uint32_t> cbufBuf(N * 64, 0u);
    std::vector<uint32_t> tempCounts(N * 64, 0u);      // the oracle models these as zero
    std::vector<float> tempC(N * 128 * 4, 0.0f);
    std::vector<uint32_t> manifolds(2 * N * MANIFOLD_WORDS, 0xFFFFFFFFu);

    std::vector<uint32_t> cmSlot(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const CorrIn& s = in[i];
        // cmIndex 0 means the two writes ALIAS, which is what the engine does when
        // a pair's contact-manager index equals its pair index.
        cmSlot[i] = (s.hdr.z == 0) ? i : (N + i);

        uint32_t* w = &pairBuf[i * 14];
        std::memcpy(&w[0], &s.aToBq, 16);                 // PxQuat q @0
        std::memcpy(&w[4], &s.aToBp, 12);                 // PxVec3 p @16
        w[7] = i * 64 + s.hdr.y;                          // startIndex @28, rebased
        w[8] = s.hdr.x;                                   // count @32
        w[9] = cmSlot[i];                                 // cmIndex @36
        w[10] = 0xAAAA0000u | i;                          // roundedStartIndex @40, unread here
        w[11] = 0xBBBB0000u | i;

        for (uint32_t j = 0; j < 64; ++j)
        {
            depthBuf[i * 64 + j] = fAt(s.depth, j);
            const F4& n = s.nrm[j];
            nicBuf[(i * 64 + j) * 4 + 0] = n.x;
            nicBuf[(i * 64 + j) * 4 + 1] = n.y;
            nicBuf[(i * 64 + j) * 4 + 2] = n.z;
            const uint32_t idx = u32At(s.nicIdx, j);
            std::memcpy(&nicBuf[(i * 64 + j) * 4 + 3], &idx, 4);
            cbufBuf[i * 64 + j] = i * 128 + u32At(s.cbuf, j);   // values index the pool
        }
        for (uint32_t j = 0; j < 128; ++j)
        {
            tempC[(i * 128 + j) * 4 + 0] = s.tempC[j].x;
            tempC[(i * 128 + j) * 4 + 1] = s.tempC[j].y;
            tempC[(i * 128 + j) * 4 + 2] = s.tempC[j].z;
            tempC[(i * 128 + j) * 4 + 3] = s.tempC[j].w;
        }
    }

    PxgBufferHandle bPairs = createBufferFrom(dev, pairBuf);
    PxgBufferHandle bDepth = createBufferFrom(dev, depthBuf);
    PxgBufferHandle bNic = createBufferFrom(dev, nicBuf);
    PxgBufferHandle bCbuf = createBufferFrom(dev, cbufBuf);
    PxgBufferHandle bTemp = createBufferFrom(dev, tempCounts);
    PxgBufferHandle bTempC = createBufferFrom(dev, tempC);
    PxgBufferHandle bOut = createBufferFrom(dev, manifolds);

    std::vector<uint64_t> pDepth(1, dev->getBufferDeviceAddress(bDepth));
    std::vector<uint64_t> pNic(1, dev->getBufferDeviceAddress(bNic));
    std::vector<uint64_t> pCbuf(1, dev->getBufferDeviceAddress(bCbuf));
    PxgBufferHandle bPDepth = createBufferFrom(dev, pDepth);
    PxgBufferHandle bPNic = createBufferFrom(dev, pNic);
    PxgBufferHandle bPCbuf = createBufferFrom(dev, pCbuf);

    CcPush push{};
    push.pairs = dev->getBufferDeviceAddress(bPairs);
    push.depthsPtr = dev->getBufferDeviceAddress(bPDepth);
    push.normalAndIndexPtr = dev->getBufferDeviceAddress(bPNic);
    push.contactsPtr = dev->getBufferDeviceAddress(bPCbuf);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.cmOutputs = 0ull;
    push.tempCountAndOffsetBuffer = dev->getBufferDeviceAddress(bTemp);
    push.numPairs = N;
    push.clusterBias = 0.0f;   // per case, overwritten below
    push.tempConvexTriContacts = dev->getBufferDeviceAddress(bTempC);
    push.pTempContactIndex = 0ull;

    // The bias is a scalar in the push blob, so it is uniform across a dispatch.
    // Every case therefore uses the SAME bias, and the oracle is fed the same
    // value through aToBp.w -- which the case builder above already wrote.
    const float BIAS = in[0].aToBp.w;
    for (uint32_t i = 0; i < N; ++i)
    {
        CorrIn s = in[i];
        s.aToBp.w = BIAS;
        if (s.aToBp.w != in[i].aToBp.w)
        {
            in[i] = s;
            bool marg = false, rr = false;
            refCorrelate(s, ref[i], marg, rr);
            skip[i] = marg ? 1 : 0;
        }
        std::memcpy(&pairBuf[i * 14 + 4], &s.aToBp, 12);
    }
    push.clusterBias = BIAS;
    dev->destroyBuffer(bPairs);
    bPairs = createBufferFrom(dev, pairBuf);
    push.pairs = dev->getBufferDeviceAddress(bPairs);

    {
        Kernel k("cvxTriCorrelateBda.spv", "convexTrimeshCorrelate", {}, sizeof(CcPush));
        k.dispatch((N + 1) / 2, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto gMan = downloadBuffer<uint32_t>(dev, bOut, manifolds.size());
    const auto gTemp = downloadBuffer<uint32_t>(dev, bTemp, tempCounts.size());
    const auto gDepth = downloadBuffer<float>(dev, bDepth, depthBuf.size());

    size_t mmC = 0, mmN = 0, mmR = 0, mmM = 0, mmT = 0, mmD = 0, checked = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        ++checked;
        const CorrOut& r = ref[i];
        const uint32_t* man = &gMan[cmSlot[i] * MANIFOLD_WORDS];
        const uint32_t* pairMan = &gMan[i * MANIFOLD_WORDS];

        // mNbManifolds: the oracle's misc.x is the pairIndex slot, misc.y the
        // cmIndex slot; UNTOUCHED means the kernel must not have written it.
        if (pairMan[NBMAN_W] != r.misc.x && !(in[i].hdr.z == 0 && man[NBMAN_W] == r.misc.x)) ++mmM;
        if (in[i].hdr.z == 1 && man[NBMAN_W] != r.misc.y) ++mmM;

        for (uint32_t p = 0; p < 4; ++p)
        {
            const uint32_t expNb = (&r.nbC.x)[p];
            if (man[NBC_W + p] != expNb) { ++mmN; break; }
        }

        const uint32_t nbMan = (in[i].hdr.z == 0) ? r.misc.x : r.misc.y;
        if (nbMan != UNTOUCHED && nbMan > 0)
        {
            float q[4], pp[4];
            std::memcpy(q, &man[RELXF_W], 16);
            std::memcpy(pp, &man[RELXF_W + 4], 16);
            if (!bitEq(q[0], r.relQ.x) || !bitEq(q[1], r.relQ.y) || !bitEq(q[2], r.relQ.z) ||
                !bitEq(q[3], r.relQ.w) || !bitEq(pp[0], r.relP.x) || !bitEq(pp[1], r.relP.y) ||
                !bitEq(pp[2], r.relP.z))
                ++mmR;

            for (uint32_t p = 0; p < nbMan && p < 4; ++p)
                for (uint32_t c = 0; c < (&r.nbC.x)[p]; ++c)
                {
                    const uint32_t base = (p * 6 + c) * 3;
                    const float* cp = reinterpret_cast<const float*>(&man[(p * 6 + c) * 12]);
                    const F4& e0 = r.contacts[base + 0];   // pointA, penetration
                    const F4& e1 = r.contacts[base + 1];   // pointB, asfloat(triIndex)
                    const F4& e2 = r.contacts[base + 2];   // normal
                    uint32_t gotTri; std::memcpy(&gotTri, &man[(p * 6 + c) * 12 + 10], 4);
                    uint32_t expTri; std::memcpy(&expTri, &e1.w, 4);
                    if (!eqF(cp[0], e0.x) || !eqF(cp[1], e0.y) || !eqF(cp[2], e0.z) ||
                        !eqF(cp[3], e1.x) || !eqF(cp[4], e1.y) || !eqF(cp[5], e1.z) ||
                        !eqF(cp[6], e2.x) || !eqF(cp[7], e2.y) || !eqF(cp[8], e2.z) ||
                        !eqF(cp[9], e0.w) || gotTri != expTri)
                    { ++mmC; break; }
                }
        }

        for (uint32_t j = 0; j < 64; ++j)
        {
            if (gTemp[i * 64 + j] != u32At(r.tempOut, j)) { ++mmT; break; }
            if (!bitEq(gDepth[i * 64 + j], fAt(r.depthOut, j))) { ++mmD; break; }
        }
    }

    for (PxgBufferHandle h : { bPairs, bDepth, bNic, bCbuf, bTemp, bTempC, bOut,
                               bPDepth, bPNic, bPCbuf })
        dev->destroyBuffer(h);

    PXG_CHECK_MSG(mmC == 0, "cvxTriCorrelate: %zu contact mismatches", mmC);
    PXG_CHECK_MSG(mmN == 0, "cvxTriCorrelate: %zu nbContacts mismatches", mmN);
    PXG_CHECK_MSG(mmR == 0, "cvxTriCorrelate: %zu relTransform mismatches", mmR);
    PXG_CHECK_MSG(mmM == 0, "cvxTriCorrelate: %zu mNbManifolds mismatches", mmM);
    PXG_CHECK_MSG(mmT == 0, "cvxTriCorrelate: %zu tempCounts mismatches", mmT);
    PXG_CHECK_MSG(mmD == 0, "cvxTriCorrelate: %zu depth mismatches", mmD);
    PXG_CHECK_MSG(checked > N / 2, "cvxTriCorrelate: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cCached > 0 && cZeroTri > 0 && cCm1 > N / 5 && cReduce > 0 &&
                  cMultiPatch > 0 && cQuirk8 > 0 && cTwoBlocks > 0 && cTie > 0,
                  "cvxTriCorrelate: coverage thin (cached %zu zeroTri %zu cm1 %zu reduce %zu "
                  "multi %zu quirk8 %zu blocks2 %zu tie %zu)",
                  cCached, cZeroTri, cCm1, cReduce, cMultiPatch, cQuirk8, cTwoBlocks, cTie);
}

} // namespace
