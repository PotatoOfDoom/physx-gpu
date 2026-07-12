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

// Verification of npCvxEEDriver.slang: generatedContactsEEContacts() from
// cudaGJKEPA.cu -- the convex-convex edge-edge contact driver composing
// getPolygonIndexFromLocalWitness (T138), the blob accessors (T141/T142) and
// generateEE (T134).  The test builds a real hull blob + vB cloud, runs the CPU
// T138 to find the selected polygon, then aims the A-edge to cross its edges.
// The output manifold (points to tolerance, count exact) is compared, with the
// T138 selection near-ties and each edge's generateEE decision-boundaries
// margin-skipped.  Caller contract: numContactsIn in {0,1}.

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
struct V3 { float x, y, z; };
struct EEDriverIn
{
    uint32_t baseWord, baseVB, nbV, nbP, nbE, numContactsIn, pad0;
    uint32_t pad_align; // keep float4s below 16-aligned
    F4 scaleRadius, rot, witnessTol, normalCD, vA0, vA1, aToBq, aToBp, seedA, seedB, seedNp;
};
struct EEDriverOut { F4 mA[4], mB[4], mNp[4]; uint32_t numContacts, pad0, pad1, pad2; };

const uint32_t MAXWORDS = 512;
const uint32_t MAXVB = 32;
const uint32_t SZ_HEADER = 32;
const float MAXF = 3.4028235e38f;
const float LOWER_EPS = 1e-2f, UPPER_EPS = 5e-2f;
const float TIE = 1.5e-4f;

static uint32_t asuint(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 quatRotate(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 quatRotateInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 shape2Vertex(const V3& v, const V3& sc, const F4& r) { V3 t = quatRotate(r, v); t.x /= sc.x; t.y /= sc.y; t.z /= sc.z; return quatRotateInv(r, t); }
static V3 getNormalized(const V3& v) { float m = dot3(v, v); return m > 0 ? mul(v, 1.f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static float clampf(float v, float lo, float hi) { return std::fmin(std::fmax(v, lo), hi); }

// T138 (global form + margin) reading planes from an array.
static uint32_t refPolyIndex(const std::vector<F4>& planes, uint32_t nbP, const V3& sc, const F4& rot,
                             const V3& witness, const V3& n, float radius, float tol, bool& marg)
{
    const float margin = radius * 0.01f;
    const float tolerance = clampf(margin, tol * LOWER_EPS, tol * UPPER_EPS);
    const float eps = -tolerance;
    const V3 lw = shape2Vertex(witness, sc, rot);
    float minDist = MAXF;
    std::vector<float> td(nbP);
    for (uint32_t i = 0; i < nbP; ++i)
    {
        const V3 pn{ planes[i].x, planes[i].y, planes[i].z };
        const float pd = dot3(pn, lw) + planes[i].w;
        if (std::fabs(pd - eps) < TIE) marg = true;
        td[i] = pd >= eps ? pd : MAXF;
        minDist = std::fmin(minDist, td[i]);
    }
    if (minDist == MAXF) return 0u;
    float best = MAXF, best2 = MAXF; uint32_t idx = 0;
    for (uint32_t i = 0; i < nbP; ++i)
    {
        const float gate = td[i] - minDist;
        if (td[i] != MAXF && std::fabs(gate - tolerance) < TIE) marg = true;
        if (tolerance > gate)
        {
            const V3 pn{ planes[i].x, planes[i].y, planes[i].z };
            const V3 s2v = shape2Vertex(pn, sc, rot);
            if (dot3(s2v, s2v) < 1e-12f) marg = true;
            const float proj = dot3(getNormalized(s2v), n);
            if (proj < best) { best2 = best; best = proj; idx = i; }
            else if (proj < best2) best2 = proj;
        }
    }
    if ((best2 - best) < TIE) marg = true;
    return idx;
}

// T134 convex-convex generateEE + margin.
static bool refGenEE(const V3& p, const V3& q, const V3& normal, const V3& a, const V3& b,
                     const F4& aToBq, const V3& aToBp, float inflatedRadius,
                     V3& lA, V3& lB, F4& nrmPen, bool& marg)
{
    const float er = 0.005f;
    const V3 ab = sub(b, a), n = crossp(ab, normal);
    const float d = dot3(n, a), np = dot3(n, p), nq = dot3(n, q);
    const float signP = np - d, signQ = nq - d, temp = signP * signQ;
    if (std::fabs(signP) < TIE || std::fabs(signQ) < TIE) marg = true;
    if (temp > 0.f) return false;
    const V3 pq = sub(q, p); const float npq = dot3(n, pq);
    if (std::fabs(npq) < TIE) marg = true;
    if (npq == 0.f) return false;
    const float segT = (d - np) / npq;
    const V3 lpa = add(mul(pq, segT), p);
    const V3 perN = crossp(normal, pq), ap = sub(lpa, a);
    const float nom = dot3(perN, ap), denom = dot3(perN, ab);
    if (std::fabs(denom) < 0.03f) marg = true;
    const float tV = nom / denom;
    const float maxT = 1.f + er, minT = -er;
    if (std::fabs(tV - maxT) < TIE || std::fabs(tV - minT) < TIE) marg = true;
    if (tV > maxT || minT > tV) return false;
    const V3 v = sub(ap, mul(ab, tV));
    const float sd = dot3(v, normal);
    if (std::fabs(inflatedRadius - sd) < TIE) marg = true;
    if (inflatedRadius >= sd)
    {
        lB = sub(lpa, v);
        lA = quatRotateInv(aToBq, sub(lpa, aToBp));
        nrmPen = F4{ normal.x, normal.y, normal.z, sd };
        return true;
    }
    return false;
}

static bool eqPt(const F4& r, const F4& v) { return almostEqual(v.x, r.x, 2e-4f, 2e-5f) && almostEqual(v.y, r.y, 2e-4f, 2e-5f) && almostEqual(v.z, r.z, 2e-4f, 2e-5f) && almostEqual(v.w, r.w, 2e-4f, 2e-4f); }

PXG_TEST(npCvxEEDriver_generate)
{
    std::mt19937 rng(143001);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<EEDriverIn> in(N);
    std::vector<uint32_t> blob(size_t(N) * MAXWORDS, 0u);
    std::vector<F4> vbBuf(size_t(N) * MAXVB, F4{ 0, 0, 0, 0 });
    std::vector<uint8_t> skip(N, 0);
    std::vector<EEDriverOut> ref(N);
    size_t nSkip = 0, hits = 0, nc0 = 0, nc1 = 0;

    auto setU32 = [&](uint32_t base, uint32_t off, uint32_t v) { blob[base + (off >> 2)] = v; };
    auto setU8 = [&](uint32_t base, uint32_t off, uint32_t v) { blob[base + (off >> 2)] |= (v & 0xffu) << (8 * (off & 3)); };
    auto setF4 = [&](uint32_t base, uint32_t off, const F4& f) { const uint32_t w = base + (off >> 2); blob[w] = asuint(f.x); blob[w + 1] = asuint(f.y); blob[w + 2] = asuint(f.z); blob[w + 3] = asuint(f.w); };

    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nbP = 3 + (rng() % 8);          // 3..10 polygons
        const uint32_t nbVB = 6 + (rng() % 14);        // 6..19 hull vertices
        const uint32_t nbE = 3 + (rng() % 18);
        const uint32_t nbVfake = nbVB;                 // blob vertex region size (unused by this fn)
        const uint32_t base = i * MAXWORDS;
        const uint32_t vbBase = i * MAXVB;

        // vB cloud
        std::vector<V3> vb(nbVB);
        for (uint32_t v = 0; v < nbVB; ++v)
        {
            vb[v] = V3{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            vbBuf[vbBase + v] = F4{ vb[v].x, vb[v].y, vb[v].z, 0 };
        }
        // hull centre for plane generation
        const V3 c{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float R = rf(0.6f, 1.4f);
        std::vector<F4> planes(nbP);
        for (uint32_t p = 0; p < nbP; ++p)
        {
            const V3 pn = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            planes[p] = F4{ pn.x, pn.y, pn.z, -dot3(pn, c) - R };
        }
        // descriptors + vertex-index table
        std::vector<uint32_t> desc(nbP);
        std::vector<uint8_t> table;
        std::vector<uint32_t> vref(nbP), nvp(nbP);
        for (uint32_t p = 0; p < nbP; ++p)
        {
            const uint32_t nv = 3 + (rng() % 5); // 3..7 verts per polygon
            nvp[p] = nv;
            vref[p] = (uint32_t)table.size();
            for (uint32_t j = 0; j < nv; ++j) table.push_back((uint8_t)(rng() % nbVB));
            const uint32_t minIdx = rng() % 100;
            desc[p] = (vref[p] << 16) | (nv << 8) | minIdx;
        }

        // transform + witness + normal
        V3 scale{ 1, 1, 1 }; F4 rot{ 0, 0, 0, 1 };
        if (i % 4 == 0)
        {
            scale = V3{ rf(0.8f, 1.3f), rf(0.8f, 1.3f), rf(0.8f, 1.3f) };
            V3 ax = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            const float ang = rf(-3.1f, 3.1f); rot = F4{ ax.x * std::sin(ang * .5f), ax.y * std::sin(ang * .5f), ax.z * std::sin(ang * .5f), std::cos(ang * .5f) };
        }
        const V3 normal = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        // witness in vertex space near the surface, mapped to shape space
        const V3 wV = add(c, mul(nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }), R + rf(-0.1f, 0.3f)));
        // vertex2Shape
        auto vertex2Shape = [&](const V3& u) { V3 tt = quatRotate(rot, u); tt.x *= scale.x; tt.y *= scale.y; tt.z *= scale.z; return quatRotateInv(rot, tt); };
        const V3 witness = vertex2Shape(wV);
        const float radius = rf(0.5f, 2.5f), tol = rf(0.9f, 1.1f);
        F4 aToBq{ 0, 0, 0, 1 };
        { V3 ax = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float ang = rf(-3.1f, 3.1f); aToBq = F4{ ax.x * std::sin(ang * .5f), ax.y * std::sin(ang * .5f), ax.z * std::sin(ang * .5f), std::cos(ang * .5f) }; }
        const V3 aToBp{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        const float inflatedRadius = rf(0.05f, 0.9f);

        // find selected polygon (CPU T138)
        bool marg = false;
        const uint32_t bIdx = refPolyIndex(planes, nbP, scale, rot, witness, normal, radius, tol, marg);
        const uint32_t selNv = nvp[bIdx], selVref = vref[bIdx];
        // selected polygon's vertices
        std::vector<V3> pv(selNv);
        for (uint32_t j = 0; j < selNv; ++j) pv[j] = vb[table[selVref + j]];

        // aim the A-edge to cross one edge of the selected polygon
        const uint32_t ek = rng() % selNv;
        const V3 e0 = pv[ek], e1 = pv[(ek + 1) % selNv];
        const V3 edge = sub(e1, e0);
        const V3 planeDir = nrm(crossp(edge, normal));
        const V3 m = add(e0, mul(edge, rf(0.3f, 0.7f)));
        const V3 mm = add(m, mul(normal, rf(-0.3f, 0.3f)));
        V3 vA0 = add(mm, mul(planeDir, rf(0.05f, 0.6f)));
        V3 vA1 = sub(mm, mul(planeDir, rf(0.05f, 0.6f)));

        const uint32_t numContactsIn = (i % 2 == 0) ? 0u : 1u;
        V3 seedA{ rf(-2, 2), rf(-2, 2), rf(-2, 2) }, seedB{ rf(-2, 2), rf(-2, 2), rf(-2, 2) }, seedN = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        const float seedS = rf(-1, 0);

        // build the blob
        setU32(base, 0, (nbE << 16) | ((nbVfake & 0xff) << 8) | (nbP & 0xff));
        const uint32_t pOff = SZ_HEADER + 16 * nbVfake;
        const uint32_t dOff = pOff + 16 * nbP;
        const uint32_t vdOff = SZ_HEADER + 16 * nbVfake + 20 * nbP + 6 * nbE + 3 * nbVfake;
        for (uint32_t p = 0; p < nbP; ++p) setF4(base, pOff + 16 * p, planes[p]);
        for (uint32_t p = 0; p < nbP; ++p) setU32(base, dOff + 4 * p, desc[p]);
        for (uint32_t j = 0; j < table.size(); ++j) setU8(base, vdOff + j, table[j]);

        EEDriverIn s{};
        s.baseWord = base; s.baseVB = vbBase; s.nbV = nbVfake; s.nbP = nbP; s.nbE = nbE; s.numContactsIn = numContactsIn;
        s.scaleRadius = { scale.x, scale.y, scale.z, radius };
        s.rot = rot;
        s.witnessTol = { witness.x, witness.y, witness.z, tol };
        s.normalCD = { normal.x, normal.y, normal.z, inflatedRadius };
        s.vA0 = { vA0.x, vA0.y, vA0.z, 0 }; s.vA1 = { vA1.x, vA1.y, vA1.z, 0 };
        s.aToBq = aToBq; s.aToBp = { aToBp.x, aToBp.y, aToBp.z, 0 };
        s.seedA = { seedA.x, seedA.y, seedA.z, 0 }; s.seedB = { seedB.x, seedB.y, seedB.z, 0 };
        s.seedNp = { seedN.x, seedN.y, seedN.z, seedS };
        in[i] = s;

        // CPU reference: replicate the driver
        EEDriverOut r{};
        V3 mA[4], mB[4], mN[4]; float mS[4];
        for (uint32_t j = 0; j < 4; ++j) { mA[j] = V3{ 0, 0, 0 }; mB[j] = V3{ 0, 0, 0 }; mN[j] = V3{ 0, 0, 0 }; mS[j] = 0; }
        if (numContactsIn >= 1) { mA[0] = seedA; mB[0] = seedB; mN[0] = seedN; mS[0] = seedS; }
        uint32_t numContacts = numContactsIn;
        size_t nHit = 0;
        for (uint32_t bi = 0; bi < selNv; bi += 32)
        {
            uint32_t offset = 0; const uint32_t blockNC = numContacts;
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint32_t start = bi + lane;
                if (start >= selNv) continue;
                const uint32_t endv = (start + 1) % selNv;
                const V3 v0 = vb[table[selVref + start]];
                const V3 v1 = vb[table[selVref + endv]];
                V3 lA, lB; F4 lNp;
                const bool hit = refGenEE(vA0, vA1, normal, v0, v1, aToBq, aToBp, inflatedRadius, lA, lB, lNp, marg);
                if (hit)
                {
                    if (offset < 4)
                    {
                        const uint32_t idx = blockNC + offset;
                        if (idx < 4) { mA[idx] = lA; mB[idx] = lB; mN[idx] = V3{ -lNp.x, -lNp.y, -lNp.z }; mS[idx] = -lNp.w; }
                    }
                    offset++; nHit++;
                }
            }
            numContacts = std::min(numContacts + offset, 4u);
        }
        for (uint32_t j = 0; j < 4; ++j) { r.mA[j] = F4{ mA[j].x, mA[j].y, mA[j].z, 0 }; r.mB[j] = F4{ mB[j].x, mB[j].y, mB[j].z, 0 }; r.mNp[j] = F4{ mN[j].x, mN[j].y, mN[j].z, mS[j] }; }
        r.numContacts = numContacts;
        ref[i] = r;
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (nHit > 0) ++hits;
        if (numContactsIn == 0) ++nc0; else ++nc1;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<EEDriverOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bBlob = createBufferFrom(dev, blob);
    PxgBufferHandle bVB = createBufferFrom(dev, vbBuf);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct EEDriverParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxEEDriver.spv", "cvxEEDriver",
             { (uint32_t)sizeof(EEDriverIn), (uint32_t)sizeof(uint32_t), (uint32_t)sizeof(F4), (uint32_t)sizeof(EEDriverOut) }, sizeof(EEDriverParams));
    k.dispatch(GROUPS, { bIn, bBlob, bVB, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<EEDriverOut>(dev, bOut, N);
    size_t mmN = 0, mmSlot = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].numContacts != ref[i].numContacts) { ++mmN; continue; }
        for (uint32_t j = 0; j < ref[i].numContacts; ++j)
            if (!eqPt(ref[i].mA[j], g[i].mA[j]) || !eqPt(ref[i].mB[j], g[i].mB[j]) || !eqPt(ref[i].mNp[j], g[i].mNp[j])) { ++mmSlot; break; }
    }
    PXG_CHECK_MSG(mmN == 0, "eeDriver: %zu numContacts mismatches", mmN);
    PXG_CHECK_MSG(mmSlot == 0, "eeDriver: %zu manifold-slot mismatches", mmSlot);
    PXG_CHECK_MSG(nSkip < (size_t)N * 4 / 5, "eeDriver: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(hits > N / 20, "eeDriver: EE-hit coverage thin (%zu)", hits);
    PXG_CHECK_MSG(nc0 > N / 20 && nc1 > N / 20, "eeDriver: numContactsIn coverage thin (%zu/%zu)", nc0, nc1);

    for (PxgBufferHandle h : { bIn, bBlob, bVB, bOut }) dev->destroyBuffer(h);
}

} // namespace
