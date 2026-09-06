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

// Verification of pxgSlangCapsuleHfCore -- capsuleHeightfieldNarrowphaseCore
// (convexHeightfield.cu:636-789), the CAPSULE arm of
// sphereHeightfieldNarrowphase.  P9.52b-4.
//
// The mirror is written from the CUDA and composes the same four pieces the module
// does.  It is a FOURTH copy of mirror code that now exists in three other
// fixtures; consolidating all four onto one shared reference header is the next
// tranche, kept separate so this one's evidence is not entangled with a refactor.
//
// ===========================================================================
// WHAT THIS FIXTURE EXISTS TO PIN
// ===========================================================================
//
// 1. UPSTREAM CANDIDATE 50 -- the capsule axis that stays in WORLD space.  Cases
//    drive a ROTATED heightfield actor, which is the only configuration where
//    `capsuleCenterInMesh + tmp` differs from `capsuleCenterInMesh +
//    rotateInv(meshQ, tmp)`.  With an axis-aligned heightfield the two agree and
//    the transcription would be accidentally right.
// 2. `getBasisVector0()` as its own formula rather than a rotate: the capsule
//    transforms carry non-identity rotations so the x component's `2w^2 - 1 + 2x^2`
//    is exercised against the generic `(w^2 - u.u) + 2x^2`.
// 3. The THREE patch-normal branches, including the only one that sets `deferred`.
// 4. The record tail, with sentinels on every slot no case claims.
//
// ===========================================================================
// WHY HALF THE CASES BELOW LOOK OVER-SPECIFIED
// ===========================================================================
//
// The module's first mutation sweep killed 46 of 56 and left TEN survivors, and all
// ten were one finding wearing ten costumes: THE QUANTITY THE MUTANT CHANGES DOES NOT
// REACH THE OUTPUT FROM ANY ORDINARY INPUT.  Specifically --
//
//   * `triFlags` reaches the output ONLY through `selectNormal`, and its answer ONLY
//     when `sqDist == 0`, because the other two arms both answer `triNormal`.  So
//     every flag-affecting mutant was blind until a case reached the deferred arm
//     with the right bit in play.
//   * `dist >= 0` and `dist > 0` differ only at EXACTLY zero.
//   * `sqInflatedRadius > sqDist` and `>=` differ only at bitwise equality.
//   * Comparing against the UNsquared radius passes unless sqDist lands between the
//     two.
//   * Dropping the second `contactDistance` term is invisible while both terms are
//     the same size.
//
// The later cases are therefore constructed, not chosen, and each names the mutant it
// exists for.  A boundary case with no stated purpose is the first thing a later
// reader deletes as arbitrary -- and this file's boundary cases are the only reason
// most of this module is measured at all.
//
// Set PXG_CAPHF_DUMP=1 to print, per case, the branch taken, dist, sqDist, u, v, the
// edge flags and both normals.  That dump is how each of the eight new cases was
// aimed; without it they would have been guesses.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct V4 { float x, y, z, w; };

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

static V3 quatRotate(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) + c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) + c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) + c.z * (w * 2.0f) };
}
static V3 quatRotateInv(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) - c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) - c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) - c.z * (w * 2.0f) };
}
static V4 quatMul(V4 a, V4 b)
{
    return { a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
             a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
             a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
             a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static V4 quatConj(V4 q) { return { -q.x, -q.y, -q.z, q.w }; }

// PxQuat::getBasisVector0 (PxQuat.h:260-265) -- ITS OWN FORMULA, not a rotate.
static V3 basisVector0(V4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return V3{ (q.w * w2) - 1.0f + q.x * x2,
               (q.z * w2) + q.y * x2,
               (-q.y * w2) + q.z * x2 };
}

static V3 normalized(V3 v)
{
    const float m = dot3(v, v);
    if (!(m > 0.0f)) return V3{ 0, 0, 0 };
    const float r = 1.0f / std::sqrt(m);
    return V3{ v.x * r, v.y * r, v.z * r };
}

static float pxMax(float a, float b) { return a < b ? b : a; }
static float pxMin(float a, float b) { return a < b ? a : b; }
static float pxClamp(float v, float lo, float hi) { return pxMin(hi, pxMax(lo, v)); }

static const float MAX_F32 = 3.4028234663852886e+38f;
static const uint32_t HF_BOUNDARY = 0xFFFFFFFFu;
static const uint32_t ETD_01 = 1u << 3, ETD_12 = 1u << 4, ETD_20 = 1u << 5;
static const uint32_t NB_CONTACTS_SHIFT = 27u;
static const uint32_t DEFER_MASK = 0x80000000u;

// ---------------------------------------------------------------------------
// the heightfield blob (heightfieldUtil.cuh, convexHeightfield.cu:83-116)
// ---------------------------------------------------------------------------
struct Hf { uint32_t rows = 0, cols = 0; std::vector<uint32_t> samples, blob; };

static uint32_t sampleWord(int16_t h, uint8_t m0, bool tess, uint8_t m1)
{
    return uint32_t(uint16_t(h)) | (uint32_t(m0 & 0x7f) << 16)
         | (tess ? (1u << 23) : 0u) | (uint32_t(m1 & 0x7f) << 24);
}
static void buildHf(Hf& h)
{ h.blob.clear(); h.blob.push_back(h.rows); h.blob.push_back(h.cols);
  for (uint32_t s : h.samples) h.blob.push_back(s); }
static bool zerothShared(const Hf& h, uint32_t cell)
{ return (h.samples[cell] & 0x800000u) != 0u; }
static float hfHeight(const Hf& h, uint32_t vi)
{ return float(int16_t(uint16_t(h.samples[vi] & 0xffffu))); }
static V3 hfVertex(const Hf& h, uint32_t vi)
{ return V3{ float(vi / h.cols), hfHeight(h, vi), float(vi % h.cols) }; }

static void triVertexIndices(const Hf& h, uint32_t t, uint32_t& i0, uint32_t& i1, uint32_t& i2)
{
    const uint32_t cell = t >> 1, c = h.cols;
    const bool first = (t & 1u) == 0u;
    if (zerothShared(h, cell))
    {
        if (first) { i0 = cell + c; i1 = cell; i2 = cell + c + 1u; }
        else       { i0 = cell + 1u; i1 = cell + c + 1u; i2 = cell; }
    }
    else
    {
        if (first) { i0 = cell; i1 = cell + 1u; i2 = cell + c; }
        else       { i0 = cell + c + 1u; i1 = cell + c; i2 = cell + 1u; }
    }
}
static void triAdjacency(const Hf& h, uint32_t t, uint32_t& a0, uint32_t& a1, uint32_t& a2)
{
    const uint32_t cell = t >> 1, c = h.cols, r = h.rows;
    const bool first = (t & 1u) == 0u;
    a0 = HF_BOUNDARY; a2 = HF_BOUNDARY;
    if (zerothShared(h, cell))
    {
        if (first) { a1 = t + 1u;
                     if (cell % c != 0u) a0 = t - 1u;
                     if (cell / c != r - 2u) a2 = ((cell + c) * 2u) + 1u; }
        else       { a1 = t - 1u;
                     if (cell % c < c - 2u) a0 = t + 1u;
                     if (cell >= c - 1u) a2 = (cell - c) * 2u; }
    }
    else
    {
        if (first) { a1 = t + 1u;
                     if (cell >= c - 1u) a0 = ((cell - c) * 2u) + 1u;
                     if (cell % c != 0u) a2 = t - 1u; }
        else       { a1 = t - 1u;
                     if (cell / c != r - 2u) a0 = (cell + c) * 2u;
                     if (cell % c < c - 2u) a2 = t + 1u; }
    }
}
static void getTriangle(const Hf& h, uint32_t t, V3 scale,
                        V3& v0, V3& v1, V3& v2, uint32_t adj[3])
{
    bool wh = false;
    if (scale.z < 0.0f) wh = !wh;
    if (scale.x < 0.0f) wh = !wh;
    uint32_t i0, i1, i2, a0, a1, a2;
    triVertexIndices(h, t, i0, i1, i2);
    triAdjacency(h, t, a0, a1, a2);
    adj[0] = wh ? a2 : a0; adj[1] = a1; adj[2] = wh ? a0 : a2;
    const uint32_t tv[3] = { i0, wh ? i2 : i1, wh ? i1 : i2 };
    V3* out[3] = { &v0, &v1, &v2 };
    for (uint32_t k = 0; k < 3; ++k)
    {
        const V3 g = hfVertex(h, tv[k]);
        *out[k] = V3{ g.x * scale.x, g.y * scale.y, g.z * scale.z };
    }
}

// computeTriangleFlags (convexHeightfield.cu:565-614), boundaryCollisions == true
static uint32_t triangleFlags(const Hf& h, const uint32_t adj[3],
                              V3 c0, V3 c1, V3 c2, V3 triNormal, V3 scale)
{
    const uint32_t nextInd[3] = { 2u, 0u, 1u };
    uint32_t f = 0u;
    for (uint32_t a = 0; a < 3; ++a)
    {
        if (adj[a] == HF_BOUNDARY) continue;
        V3 av0, av1, av2; uint32_t ig[3];
        getTriangle(h, adj[a], scale, av0, av1, av2, ig);
        V3 adjNormal = cross3(sub(av1, av0), sub(av2, av0));
        const uint32_t oi = nextInd[a];
        const V3 other = (oi == 0u) ? c0 : ((oi == 1u) ? c1 : c2);
        const float projD = dot3(adjNormal, sub(other, av0));
        if (projD < 0.0f)
        {
            const float m2 = dot3(adjNormal, adjNormal);
            if (m2 > 0.0f)
            { const float r = 1.0f / std::sqrt(m2);
              adjNormal = V3{ adjNormal.x * r, adjNormal.y * r, adjNormal.z * r }; }
            if (dot3(adjNormal, triNormal) < 0.997f) f |= (1u << (a + 3u));
        }
        else f |= (1u << (a + 3u));
    }
    return f;
}

// selectNormal (capsuleTriangle.cuh:38-98)
static bool selectNormal(float u, float v, uint32_t f)
{
    const float zero = 1e-6f, one = 0.999999f;
    if (zero > u)
    {
        if (zero > v) { if ((f & (ETD_01 | ETD_20)) == 0u) return true; }
        else if (v > one) { if ((f & (ETD_12 | ETD_20)) == 0u) return true; }
        else { if ((f & ETD_20) == 0u) return true; }
    }
    else if (u > one)
    {
        if (zero > v) { if ((f & (ETD_01 | ETD_12)) == 0u) return true; }
    }
    else
    {
        if (zero > v) { if ((f & ETD_01) == 0u) return true; }
        else
        {
            const float threshold = 0.9999f;
            if ((u + v) >= threshold) { if ((f & ETD_12) == 0u) return true; }
            else return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// distanceSegmentSegmentSquared / distanceSegmentTriangleSquared
// ---------------------------------------------------------------------------
static bool validBary(float v, float w)
{
    const float eps = 1e-6f, zero = -eps, one = 1.0f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

static void segSeg(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const V3 r = sub(p1, p2);
    const float a = dot3(d1, d1), e = dot3(d2, d2);
    const float b = dot3(d1, d2), c = dot3(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = dot3(d2, r);
    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = pxClamp(temp / denom, 0.0f, 1.0f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = pxClamp(tTmp, 0.0f, 1.0f);
    const float comp = (b * t2 - c) * aRecip;
    s = pxClamp(comp, 0.0f, 1.0f);
    t = t2;
}

static float segTri(V3 p, V3 q, V3 a, V3 b, V3 c, float& t, float& u, float& v)
{
    const V3 pq = sub(q, p), ab = sub(b, a), ac = sub(c, a), bc = sub(c, b);
    const V3 ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : MAX_F32;
    const V3 n = normalized(cross3(ab, ac));
    const float dist3 = dot3(ap, n), sqDist3 = dist3 * dist3;
    const float dist4 = dot3(aq, n), sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;

    if (dMul <= 0.0f)
    {
        const float nom = -dot3(n, ap);
        const float denom = 1.0f / dot3(n, pq);
        const float t0 = nom * denom;
        const V3 ip = add(p, mul(pq, t0));
        const V3 v2 = sub(ip, a);
        const float d20 = dot3(v2, ab), d21 = dot3(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom;
        const float w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (validBary(v00, w00)) { t = t0; u = v00; v = w00; return 0.0f; }
    }

    const V3 cp31 = sub(p, mul(n, dist3));
    const V3 pV20 = sub(cp31, a);
    const float pD20 = dot3(pV20, ab), pD21 = dot3(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = validBary(v0, w0);

    const V3 cp41 = sub(q, mul(n, dist4));
    const V3 qV20 = sub(cp41, a);
    const float qD20 = dot3(qV20, ab), qD21 = dot3(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = validBary(v1, w1);

    if (con0 && con1)
    {
        if (sqDist4 > sqDist3) { t = 0.0f; u = v0; v = w0; return sqDist3; }
        t = 1.0f; u = v1; v = w1; return sqDist4;
    }

    float t00, t01, t02, t10, t11, t12;
    segSeg(p, pq, a, ab, t00, t10);
    segSeg(p, pq, b, bc, t01, t11);
    segSeg(p, pq, a, ac, t02, t12);

    V3 tv = sub(add(p, mul(pq, t00)), add(a, mul(ab, t10)));
    const float sqDist0 = dot3(tv, tv);
    tv = sub(add(p, mul(pq, t01)), add(b, mul(bc, t11)));
    const float sqDist1 = dot3(tv, tv);
    tv = sub(add(p, mul(pq, t02)), add(a, mul(ac, t12)));
    const float sqDist2 = dot3(tv, tv);

    const float u10 = t10, v10 = 0.0f;
    const float u11 = 1.0f - t11, v11 = t11;
    const float u12 = 0.0f, v12 = t12;

    float sqDistPE, uEdge, vEdge, tSeg;
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0)
    { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; }
    else if (sqDist2 > sqDist1)
    { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; }
    else
    { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; }

    if (con0)
    {
        if (sqDistPE > sqDist3) { t = 0.0f; u = v0; v = w0; return sqDist3; }
        t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
    }
    if (con1)
    {
        if (sqDistPE > sqDist4) { t = 1.0f; u = v1; v = w1; return sqDist4; }
        t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
    }
    t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
}

// ---------------------------------------------------------------------------
// the two contact generators
// ---------------------------------------------------------------------------
struct Contacts { V3 pt[2]; float sep[2]; uint32_t n; };
static Contacts contactsInit()
{ Contacts o{}; o.pt[0] = V3{0,0,0}; o.pt[1] = V3{0,0,0};
  o.sep[0] = 0.0f; o.sep[1] = 0.0f; o.n = 0; return o; }

static Contacts genContacts(V3 a, V3 b, V3 c, V3 planeNormal, V3 normal,
                            V3 p, V3 q, float inflatedRadius)
{
    Contacts o = contactsInit();
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
    const float tdenom = (d00 * d11 - d01 * d01);
    const float bdenom = (tdenom > 0.0f) ? (1.0f / tdenom) : MAX_F32;

    const float inomp = dot3(planeNormal, neg(ap));
    const float ideom = dot3(planeNormal, normal);
    const float ipt = ideom > 0.0f ? -(inomp / ideom) : MAX_F32;
    const V3 cp31 = sub(p, mul(normal, ipt));
    const V3 pV20 = sub(cp31, a);
    const float v0 = (d11 * dot3(pV20, ab) - d01 * dot3(pV20, ac)) * bdenom;
    const float w0 = (d00 * dot3(pV20, ac) - d01 * dot3(pV20, ab)) * bdenom;
    if (validBary(v0, w0) && (inflatedRadius > ipt))
    { o.pt[o.n] = p; o.sep[o.n] = ipt; o.n++; }

    const float inomq = dot3(planeNormal, neg(aq));
    const float iqt = ideom > 0.0f ? -(inomq / ideom) : MAX_F32;
    const V3 cp41 = sub(q, mul(normal, iqt));
    const V3 qV20 = sub(cp41, a);
    const float v1 = (d11 * dot3(qV20, ab) - d01 * dot3(qV20, ac)) * bdenom;
    const float w1 = (d00 * dot3(qV20, ac) - d01 * dot3(qV20, ab)) * bdenom;
    if (validBary(v1, w1) && (inflatedRadius > iqt))
    { o.pt[o.n] = q; o.sep[o.n] = iqt; o.n++; }

    return o;
}

static Contacts genEE(V3 p, V3 q, float sqR, V3 normal, V3 a, V3 b, Contacts o)
{
    const V3 ab = sub(b, a);
    const V3 n = cross3(ab, normal);
    const float d = dot3(n, a), np = dot3(n, p), nq = dot3(n, q);
    if ((np - d) * (nq - d) > 0.0f) return o;
    const V3 pq = sub(q, p);
    const float npq = dot3(n, pq);
    if (npq == 0.0f) return o;
    const float segT = (d - np) / npq;
    const V3 L = add(p, mul(pq, segT));
    const V3 perNormal = cross3(normal, pq);
    const V3 ap = sub(L, a);
    const float tValue = dot3(perNormal, ap) / dot3(perNormal, ab);
    if (tValue > 1.0f || tValue < 0.0f) return o;
    const V3 v = sub(ap, mul(ab, tValue));
    const float sqDist = dot3(v, v);
    if (sqR > sqDist)
    { o.pt[o.n] = L; o.sep[o.n] = dot3(v, normal); o.n++; }
    return o;
}

static Contacts genEEContacts(V3 a, V3 b, V3 c, V3 normal, V3 p, V3 q,
                              float sqR, Contacts o)
{
    if (o.n < 2) o = genEE(p, q, sqR, normal, a, b, o);
    if (o.n < 2) o = genEE(p, q, sqR, normal, b, c, o);
    if (o.n < 2) o = genEE(p, q, sqR, normal, a, c, o);
    return o;
}

// ---------------------------------------------------------------------------
// the core itself
// ---------------------------------------------------------------------------
enum PnBranch { PN_SELECTED = 0, PN_FACE_FALLBACK, PN_DEFERRED, PN_NONE };

struct CoreOut
{
    uint32_t nb, pairOffset, pairOffsetPadded;
    V3 pt[2];
    float sep[2];
    float maxDepth;
    uint32_t interm, ordered, niIndex;
    V3 niNormal;
    PnBranch branch;
    bool accepted, distNegative, tooFar;
    // Diagnostics.  Every survivor of this module's first mutation sweep was a
    // fixture gap of the same shape -- an input that never approached the decision
    // it was supposed to exercise -- and every one of them was invisible from the
    // outputs alone, because `triFlags` reaches the output ONLY through
    // `selectNormal`, whose answer in turn matters ONLY when `sqDist == 0`.  These
    // are printed, not asserted, so that constructing the next boundary case is a
    // reading exercise rather than a guessing one.
    float dist, sqDist, u, v, inflatedRadius, sqInflatedRadius;
    uint32_t triFlags;
    V3 triNormal, patchNormal;
};

struct GpuShape
{
    float sx, sy, sz, rx, ry, rz, rw;
    uint32_t materialIndex, hullLo, hullHi, type, particleOrSoftbodyId;
};
static_assert(sizeof(GpuShape) == 48, "PxgShape mirror");
struct CmInput { uint32_t shapeRef0, shapeRef1, tcRef0, tcRef1; };

static CoreOut capsuleCore(const Hf& hf, const GpuShape& capsuleShape,
                           const GpuShape& hfShape, V4 capQ, V3 capP, V4 meshQ, V3 meshP,
                           float contactDist, uint32_t triangleIdx,
                           uint32_t startIndex, uint32_t rounded, uint32_t testOffset)
{
    CoreOut r{};
    r.pairOffset = startIndex + testOffset;
    r.pairOffsetPadded = rounded + testOffset;

    const V3 hfScale = { hfShape.sx, hfShape.sy, hfShape.sz };
    V3 v0, v1, v2; uint32_t adj[3];
    getTriangle(hf, triangleIdx, hfScale, v0, v1, v2, adj);

    const float capsuleRadius = capsuleShape.sy;
    const float capsuleHalfHeight = capsuleShape.sx;

    const V4 toCapQ = quatMul(quatConj(capQ), meshQ);
    const V3 toCapP = quatRotateInv(capQ, sub(meshP, capP));

    const V3 tmp = mul(basisVector0(capQ), capsuleHalfHeight);
    const V3 capsuleCenterInMesh = quatRotateInv(meshQ, sub(capP, meshP));
    const V3 p0 = add(capsuleCenterInMesh, tmp);
    const V3 p1 = sub(capsuleCenterInMesh, tmp);

    const float inflatedRadius = capsuleRadius + contactDist;
    const V3 triNormal = normalized(cross3(sub(v1, v0), sub(v2, v0)));
    const float dist = dot3(sub(capsuleCenterInMesh, v0), triNormal);

    float tSeg, u, v;
    const float sqDist = segTri(p0, p1, v0, v1, v2, tSeg, u, v);
    const float sqInflatedRadius = inflatedRadius * inflatedRadius;

    float separation = MAX_F32;
    bool deferred = false;
    uint32_t nb = 0;
    V3 patchNormal = { 0, 0, 0 };
    Contacts cc = contactsInit();
    r.branch = PN_NONE;
    r.distNegative = dist < 0.0f;
    r.tooFar = !(sqInflatedRadius > sqDist);
    r.dist = dist; r.sqDist = sqDist; r.u = u; r.v = v;
    r.inflatedRadius = inflatedRadius; r.sqInflatedRadius = sqInflatedRadius;
    r.triNormal = triNormal;
    r.triFlags = triangleFlags(hf, adj, v0, v1, v2, triNormal, hfScale);

    if (dist >= 0.0f && sqInflatedRadius > sqDist)
    {
        const uint32_t flags = r.triFlags;
        if (selectNormal(u, v, flags)) { patchNormal = triNormal; r.branch = PN_SELECTED; }
        else if (sqDist > 0.0f)        { patchNormal = triNormal; r.branch = PN_FACE_FALLBACK; }
        else
        {
            const V3 pq = sub(p1, p0);
            const V3 pointOnSegment = add(p0, mul(pq, tSeg));
            const float w = 1.0f - (u + v);
            const V3 pointOnTriangle = add(add(mul(v0, w), mul(v1, u)), mul(v2, v));
            patchNormal = normalized(sub(pointOnSegment, pointOnTriangle));
            deferred = true;
            r.branch = PN_DEFERRED;
        }

        cc = genContacts(v0, v1, v2, triNormal, patchNormal, p0, p1, inflatedRadius);
        cc = genEEContacts(v0, v1, v2, patchNormal, p0, p1, sqInflatedRadius, cc);
        nb = cc.n;

        for (uint32_t i = 0; i < 2; ++i)
            if (i < nb)
            {
                cc.pt[i] = add(quatRotate(toCapQ, cc.pt[i]), toCapP);
                separation = pxMin(separation, cc.sep[i]);
            }
    }

    r.nb = nb;
    r.pt[0] = cc.pt[0]; r.pt[1] = cc.pt[1];
    r.sep[0] = cc.sep[0]; r.sep[1] = cc.sep[1];
    r.maxDepth = separation;
    r.interm = triangleIdx;
    r.ordered = ((nb != 0 && !deferred) ? DEFER_MASK : 0u) | triangleIdx;
    r.niNormal = quatRotate(toCapQ, neg(patchNormal));
    r.niIndex = (nb << NB_CONTACTS_SHIFT) + triangleIdx;
    r.accepted = nb != 0;
    r.patchNormal = patchNormal;
    return r;
}

// ---------------------------------------------------------------------------
struct Push
{
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, sphereTrimeshPair;
    uint64_t ni, maxDepth, interm, ordered, pairs, output;
    uint32_t numPairs, pad;
};

static bool sameBits(float g, float e)
{ uint32_t x, y; std::memcpy(&x, &g, 4); std::memcpy(&y, &e, 4); return x == y; }
static bool sameBitsU(uint32_t g, float e)
{ uint32_t y; std::memcpy(&y, &e, 4); return g == y; }

// Hand-picked world transforms put the capsule where the ARITHMETIC happens to land
// it, which for a ridged heightfield read through a rotated actor is almost always
// "nowhere near the triangle" -- the first draft of this fixture had eight of twelve
// cases never enter the body at all.  Each case therefore names a point on its
// TARGET TRIANGLE in barycentric coordinates and an axis direction, and the capsule's
// world transform is solved for backwards.  That makes every branch addressable:
//
//   * the segment MIDPOINT sits at `a + ab*bu + ac*bv + N*h`, so `h` alone decides
//     `dist` and `bu, bv` alone decide the barycentrics `selectNormal` is asked
//     about -- a displacement along N cannot change them (P9.52a's lever again);
//   * the axis is a UNIT direction read as a mesh-space vector, which is exactly
//     what upstream candidate 50 makes of it, so the placement is solved in the same
//     frame the defect leaves it in.
//
// The quaternion is the shortest arc from local +x to that direction, TWISTED about
// the local x axis afterwards.  The twist leaves `getBasisVector0` untouched -- it is
// the image of local x -- but it is not cosmetic: the shortest arc alone has q.x == 0
// exactly, which would zero the `q.x*x2` term in both of the formula's first two
// components and let a mutation on either survive.
enum AxisMode { AX_NORMAL = 0, AX_EDGE_AB, AX_TILTED };

// TUNE_TIE_SQ: solve this case's `contactDistance` so that `sqInflatedRadius` comes
// out BITWISE EQUAL to its `sqDist`.  `sqInflatedRadius > sqDist` and
// `sqInflatedRadius >= sqDist` differ on exactly that input and on no other, and no
// amount of hand-chosen geometry lands there -- the radius has to be solved for the
// distance rather than the other way round.  See the search below.
enum Tune { TUNE_NONE = 0, TUNE_TIE_SQ };

struct Spec
{
    uint32_t capsuleShape, hfShape, meshXf, tri, testOffset;
    float bu, bv;      // where on the triangle the segment's midpoint projects
    float h;           // its signed height above the triangle plane
    AxisMode axis;
    float twist;       // rotation about the capsule's own axis -- see above
    float tiltAmt;     // AX_TILTED only: 0 -> along ab, 1 -> along the normal
    Tune tune;
    const char* why;
};

PXG_TEST(capsuleHfCore)
{
    PxgSlangComputeDevice* dev = testDevice();

    // A gently ridged 6x5 heightfield: flat rows, one shallow step and one steep
    // one, so the edge classification produces mixed flags and `selectNormal` is
    // asked real questions rather than uniform ones.
    Hf hf;
    hf.rows = 6; hf.cols = 5;
    {
        const int16_t hts[30] = {  0,  0,  0,  0,  0,
                                   0,  0,  0,  0,  0,
                                   0,  8,  8,  0,  0,
                                   0, 24, 24,  0,  0,
                                   0,  0,  0,  0,  0,
                                   0,  0,  0,  0,  0 };
        for (uint32_t i = 0; i < 30; ++i)
            hf.samples.push_back(sampleWord(hts[i], uint8_t(1 + (i % 5)),
                                            (i % 4u) == 1u, uint8_t(2 + (i % 7))));
    }
    buildHf(hf);

    PxgBufferHandle bHf = createBufferFrom(dev, hf.blob);
    const uint64_t addrHf = dev->getBufferDeviceAddress(bHf);

    std::vector<GpuShape> shapes(5);
    auto setShape = [](GpuShape& s, V3 sc, uint64_t ptr, uint32_t type) {
        s.sx = sc.x; s.sy = sc.y; s.sz = sc.z;
        s.rx = 0; s.ry = 0; s.rz = 0; s.rw = 1;
        s.materialIndex = 0;
        s.hullLo = uint32_t(ptr & 0xffffffffu); s.hullHi = uint32_t(ptr >> 32);
        s.type = type; s.particleOrSoftbodyId = 0;
    };
    // 0, 1: heightfields, one with a NEGATIVE column scale so the winding swap runs.
    setShape(shapes[0], { 1.0f, 0.0625f, 1.0f }, addrHf, 9u);   // eHEIGHTFIELD
    setShape(shapes[1], { 1.0f, 0.0625f, -1.0f }, addrHf, 9u);
    // 2, 3: capsules.  PxgShape carries (halfHeight, radius, radius), so `.y` is the
    // radius and `.x` the half height -- the two DIFFER here, which is what pins
    // which component each is read from.
    setShape(shapes[2], { 0.6f, 0.25f, 0.25f }, 0, 2u);         // eCAPSULE
    setShape(shapes[3], { 0.35f, 0.5f, 0.5f }, 0, 2u);
    // 4: a third heightfield, whose scales are NOT binary fractions.
    //
    // This one exists for a single mutant.  `dist-from-v1` measures the triangle
    // plane from v1 instead of v0, and the difference is `dot(v0 - v1, N)`, which is
    // identically zero in exact arithmetic.  Measured over all forty triangles of
    // both heightfields above, it is zero BITWISE as well -- with vertex coordinates
    // that are small integers times a power of two, every product in
    // `ab.x*N.x + ab.y*N.y + ab.z*N.z` is exact and the sum cancels to a clean zero
    // (tri 18 even produces -0.0f, which is not `< 0.0f`).  So on axis-aligned
    // integer terrain the two spellings are the SAME FUNCTION, and no placement on
    // such terrain can tell them apart.  Scales of 0.7 and 0.3 break the exactness:
    // the cross product rounds, the normalization rounds, and the cancellation leaves
    // a residue whose sign is a property of the triangle.  The guard below picks a
    // triangle whose residue is negative, which is what turns `dist >= 0` from true
    // into false under the mutant.
    setShape(shapes[4], { 0.7f, 0.08125f, 0.3f }, addrHf, 9u);

    // Slots 0 and 1 are the heightfield frames -- 0 ROTATED, the configuration
    // upstream candidate 50 needs to be visible, 1 the identity.  The capsule slots
    // follow, one per case, all solved below.
    const uint32_t NUM_MESH_XF = 2;
    struct Xf { V4 q; V3 p; };
    Xf meshXfs[NUM_MESH_XF];
    {
        const float a = 0.65f;
        meshXfs[0] = { { 0.0f, std::sin(a * 0.5f), 0.0f, std::cos(a * 0.5f) },
                       { -0.6f, 0.2f, 0.4f } };
    }
    meshXfs[1] = { { 0, 0, 0, 1 }, { 0.0f, 0.0f, 0.0f } };

    // `h` is the midpoint height, so with AX_NORMAL the NEAR endpoint sits at
    // `h - halfHeight`: capsule 2 has halfHeight 0.6 and inflatedRadius ~0.26, so
    // h = 0.7 puts one endpoint 0.1 above the face (inside the radius, one contact)
    // and the other 1.3 above it (outside, no second contact and no edge crossing) --
    // the only way to reach `nbOutputContacts == 1`.
    //
    // bv = 0.49995 with bu = 0.5 puts the closest feature at u + v = 0.99995: past
    // `selectNormal`'s 0.9999 threshold, so the answer turns on ETD_12, and inside
    // `validBary`'s 1 + 1e-6, so the point still counts as on the triangle.  ETD_12
    // is set on essentially every triangle here -- the in-cell diagonal neighbour is
    // coplanar, which makes `projD` exactly 0, which is NOT `< 0`, which takes the
    // `else if (boundaryCollisions)` arm.
    const std::vector<Spec> specs = {
        { 2, 0, 0, 18, 0, 1.0f/3, 1.0f/3, 0.70f, AX_NORMAL,   0.7f, 0.0f, TUNE_NONE,
          "rotated hf, axis along the normal -- ONE endpoint inside the radius" },
        { 3, 0, 0, 18, 1, 0.50f, 0.49995f, 0.02f, AX_NORMAL,  -1.1f, 0.0f, TUNE_NONE,
          "rotated hf, segment CROSSES the face at u+v=0.99995 -> DEFERRED" },
        { 3, 0, 0, 18, 2, 0.50f, 0.49995f, 0.45f, AX_NORMAL,   0.4f, 0.0f, TUNE_NONE,
          "same barycentrics but clear of the face -> face fallback" },
        { 2, 0, 0, 18, 3, 1.0f/3, 1.0f/3, -0.50f, AX_NORMAL,   0.2f, 0.0f, TUNE_NONE,
          "capsule BELOW the plane -> dist < 0" },
        { 2, 0, 0, 18, 4, 1.0f/3, 1.0f/3,  5.00f, AX_NORMAL,   1.3f, 0.0f, TUNE_NONE,
          "capsule far above -> outside the inflated radius" },
        { 3, 0, 0, 18, 0, 0.30f, 0.30f,  0.12f, AX_EDGE_AB,  -0.6f, 0.0f, TUNE_NONE,
          "axis along edge ab, lying over the face -> edge-edge contacts" },
        { 3, 0, 0, 19, 1, 0.25f, 0.40f,  0.10f, AX_TILTED,    0.9f, 0.66f, TUNE_NONE,
          "the cell's OTHER triangle, tilted axis" },
        { 2, 1, 1, 18, 2, 1.0f/3, 1.0f/3,  0.15f, AX_EDGE_AB,   1.7f, 0.0f, TUNE_NONE,
          "UNROTATED hf with a NEGATIVE column scale (wrongHanded)" },
        { 3, 1, 1, 19, 3, 0.50f, 0.49995f, 0.03f, AX_NORMAL,  -0.3f, 0.0f, TUNE_NONE,
          "wrongHanded, crossing the face -> deferred on a swapped winding" },
        { 2, 0, 0,  8, 4, 0.30f, 0.35f,  0.14f, AX_TILTED,   -1.5f, 0.66f, TUNE_NONE,
          "a rim triangle (BOUNDARY adjacencies, no flags from those edges)" },
        { 3, 0, 0, 26, 0, 0.35f, 0.30f,  0.10f, AX_EDGE_AB,   0.5f, 0.0f, TUNE_NONE,
          "the steep ridge" },
        { 2, 0, 0, 27, 1, 1.0f/3, 1.0f/3,  0.66f, AX_NORMAL,   2.2f, 0.0f, TUNE_NONE,
          "the steep ridge, second triangle, one endpoint inside" },
        { 2, 0, 0, 12, 2, 0.05f, 0.90f,  0.09f, AX_TILTED,    0.3f, 0.66f, TUNE_NONE,
          "a flat interior triangle, closest feature near vertex c" },
        { 3, 0, 0, 24, 3, 0.45f, 0.10f,  0.05f, AX_EDGE_AB,  -2.0f, 0.0f, TUNE_NONE,
          "the ridge's flat neighbour, axis across the shallow step" },

        // ------------------------------------------------------------------
        // The cases below were written to kill named survivors of this module's
        // FIRST mutation sweep.  Each names the mutant it exists for, because a
        // boundary case with no stated purpose is the first thing a later reader
        // deletes as arbitrary.
        // ------------------------------------------------------------------

        // `selnormal-uv` (the two arguments of htfSelectNormal swapped).  Its answer
        // reaches the output ONLY when sqDist == 0 -- both other arms set
        // patchNormal = triNormal, so a wrong answer there is invisible -- and it is
        // symmetric under the swap unless u and v fall in DIFFERENT regions of the
        // classifier AND the two edge flags it then consults disagree.  bu = 0 puts
        // the closest feature on edge ac, where the edge path returns u EXACTLY 0;
        // tri 19 read through the wrongHanded scale has ETD_20 clear and ETD_01 set,
        // so the original consults the clear bit and the swap the set one.
        { 3, 1, 1, 19, 4, 0.00f, 0.50f,  0.02f, AX_NORMAL,   0.35f, 0.0f, TUNE_NONE,
          "u=0 with ETD_20 clear and ETD_01 set -> the argument ORDER is observable" },

        // `rows-cols-swap`.  `rows` reaches nothing but `triAdjacency`, and there
        // only through `cell / c != r - 2`, so it is invisible unless a case sits in
        // the LAST-BUT-ONE cell row: cell 15 is row 3, and r-2 is 4 with the real
        // row count and 3 with the column count in its place, which turns adj[0]
        // into BOUNDARY and clears ETD_01.  bv = 0 puts the closest feature on edge
        // ab, which is the one selectNormal answers with ETD_01.
        { 2, 0, 0, 31, 0, 0.50f, 0.00f,  0.02f, AX_NORMAL,  -0.75f, 0.0f, TUNE_NONE,
          "cell row 3, closest feature on edge ab -> ETD_01, hence `rows`, is read" },

        // `gate-dist-strict` (`dist >= 0` weakened to `> 0`) and `dist-from-v1`
        // (the plane's reference vertex changed).  Both need `dist` at EXACTLY zero:
        // bu = bv = h = 0 puts the segment midpoint on vertex a itself, so
        // `capsuleCenterInMesh - t.v0` is the exact zero vector and its dot product
        // with any normal is exactly 0.0f -- but only through the IDENTITY mesh
        // transform, where ctQuatRotate's `v*(w*w - dot(u,u))` degenerates to `v*1`
        // and the world round trip is bit-exact.  `dist-from-v1` then measures from
        // v1 instead, whose offset from the plane is the rounding residue of
        // `dot(ab, N)`; on a SLOPED triangle that residue is not zero, and the guard
        // below refuses the case if it does not come out negative.
        { 2, 4, 1, 25, 1, 0.00f, 0.00f,  0.00f, AX_NORMAL,   1.1f, 0.0f, TUNE_NONE,
          "midpoint EXACTLY on vertex a, on non-binary scales -> dist == 0 and the "
          "v1 residue is negative" },
        { 3, 4, 1, 30, 2, 0.00f, 0.00f,  0.00f, AX_NORMAL,  -1.9f, 0.0f, TUNE_NONE,
          "the same, on a second such triangle" },

        // `gate-unsquared` (sqInflatedRadius replaced by inflatedRadius in the gate).
        // Needs sqDist strictly between the two: 0.4 above the face gives 0.16,
        // against 0.069 squared and 0.262 unsquared.  The real code skips the body
        // and leaves patchNormal zero; the mutant enters and writes -triNormal.
        { 2, 0, 0, 18, 3, 1.0f/3, 1.0f/3,  1.00f, AX_NORMAL,   0.85f, 0.0f, TUNE_NONE,
          "0.4 above: outside the SQUARED radius, inside the unsquared one" },

        // `cdist-one-side` (the second contactDistance term dropped).  The two terms
        // differ by 0.05 here -- see the explicit contactDistance below -- and the
        // near endpoint sits between the truncated and the full inflated radius.
        { 2, 1, 1, 18, 4, 1.0f/3, 1.0f/3,  0.885f, AX_NORMAL, -0.45f, 0.0f, TUNE_NONE,
          "near endpoint between the one-sided and the two-sided contact distance" },

        // `gate-sq-nonstrict` (`>` weakened to `>=`).  Differs on an exact tie only;
        // the contact distance is solved for it.
        { 2, 1, 1, 12, 0, 0.25f, 0.25f,  0.85f, AX_NORMAL,   1.55f, 0.0f, TUNE_TIE_SQ,
          "sqInflatedRadius solved to equal sqDist BITWISE -> `>` vs `>=`" },

        // `ee-gets-unsq` (the edge generator handed the unsquared radius).  Needs an
        // edge-edge candidate whose distance is outside the squared radius but
        // inside its square root, while the SEGMENT still reaches the triangle --
        // a steeply tilted capsule whose low end touches the face and whose high end
        // crosses an edge's plane well above it.
        { 2, 0, 0, 18, 1, 0.50f, 0.20f,  0.30f, AX_TILTED,   1.15f, 0.42f, TUNE_NONE,
          "tilted: face contact low, edge-plane crossing high -> the EE radius shows" },
    };

    const uint32_t n = uint32_t(specs.size());

    // ---- solve each capsule's world transform from its placement ---------------
    std::vector<Xf> capXfs(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const Spec& sp = specs[i];
        const GpuShape& cs = shapes[sp.capsuleShape];
        const GpuShape& hs = shapes[sp.hfShape];

        V3 a, b, c; uint32_t ig[3];
        getTriangle(hf, sp.tri, V3{ hs.sx, hs.sy, hs.sz }, a, b, c, ig);
        const V3 ab = sub(b, a), ac = sub(c, a);
        const V3 nrm = normalized(cross3(ab, ac));

        V3 dir;
        if (sp.axis == AX_NORMAL)        dir = nrm;
        else if (sp.axis == AX_EDGE_AB)  dir = normalized(ab);
        else                             dir = normalized(add(mul(normalized(ab), 1.0f - sp.tiltAmt),
                                                              mul(nrm, sp.tiltAmt)));

        // The shortest arc is undefined for the antipode of +x, and heightfield edges
        // run along the axes, so `ab` hits it exactly rather than nearly.  Flipping
        // the direction there only swaps which endpoint is p0 and which is p1 -- the
        // mirror and the shader see the same quaternion either way.
        if (dir.x < -0.99f) dir = neg(dir);

        // shortest arc from local +x to `dir`, then a twist about local +x
        V4 q;
        {
            const V3 ax = cross3(V3{ 1, 0, 0 }, dir);
            const float w = 1.0f + dir.x;
            const float len = std::sqrt(ax.x * ax.x + ax.y * ax.y + ax.z * ax.z + w * w);
            q = V4{ ax.x / len, ax.y / len, ax.z / len, w / len };
            const V4 tw = { std::sin(sp.twist * 0.5f), 0.0f, 0.0f, std::cos(sp.twist * 0.5f) };
            q = quatMul(q, tw);
        }

        // the midpoint, in the HEIGHTFIELD's frame, then pushed out to the world
        const V3 mid = add(add(a, add(mul(ab, sp.bu), mul(ac, sp.bv))), mul(nrm, sp.h));
        const Xf& mx = meshXfs[sp.meshXf];
        capXfs[i] = { q, add(mx.p, quatRotate(mx.q, mid)) };

        // The solve only makes sense if `getBasisVector0` really returns `dir`; a
        // near-degenerate arc (dir.x close to -1) would quietly break it.
        const V3 got = basisVector0(q);
        const float err = std::fabs(got.x - dir.x) + std::fabs(got.y - dir.y)
                        + std::fabs(got.z - dir.z);
        PXG_CHECK_MSG(err < 1e-4f,
                      "case %u: the solved quaternion's basis vector 0 is off by %g -- the "
                      "placement below it is then meaningless", i, double(err));
        PXG_CHECK_MSG(std::fabs(q.x) > 1e-3f,
                      "case %u: q.x is ~0, which zeroes the `q.x*x2` term in two of "
                      "getBasisVector0's three components", i);
        (void)cs;
    }

    std::vector<float> xf((NUM_MESH_XF + capXfs.size()) * 8, 0.0f);
    auto setXf = [&](uint32_t i, V4 q, V3 p) {
        float* r = xf.data() + size_t(i) * 8;
        r[0] = q.x; r[1] = q.y; r[2] = q.z; r[3] = q.w;
        r[4] = p.x; r[5] = p.y; r[6] = p.z;
    };
    for (uint32_t i = 0; i < NUM_MESH_XF; ++i) setXf(i, meshXfs[i].q, meshXfs[i].p);
    for (uint32_t i = 0; i < capXfs.size(); ++i)
        setXf(NUM_MESH_XF + i, capXfs[i].q, capXfs[i].p);

    std::vector<float> cdist(NUM_MESH_XF + capXfs.size(), 0.0f);
    for (uint32_t i = 0; i < cdist.size(); ++i) cdist[i] = 0.004f + 0.0021f * float(i);
    // The engine sums the two shapes' contact distances.  With both terms the same
    // size, dropping one moves the inflated radius by well under a per-mille and no
    // case notices; mesh slot 1 therefore carries a term an order of magnitude
    // larger than any capsule's, which is what makes the sum's SECOND operand
    // observable at all.
    cdist[1] = 0.05f;
    // ... and the case written for `cdist-one-side` needs its own term small enough
    // that the mesh's is what carries the gate over the line.  Its slot is read by
    // nobody else -- there is one transform per case -- so pinning it here does not
    // disturb any other placement.
    for (uint32_t i = 0; i < n; ++i)
        if (specs[i].meshXf == 1u && specs[i].tune == TUNE_NONE
            && specs[i].h > 0.88f && specs[i].h < 0.89f)
            cdist[NUM_MESH_XF + i] = 0.005f;

    std::vector<CmInput> cms(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        cms[i].shapeRef0 = specs[i].capsuleShape;
        cms[i].shapeRef1 = specs[i].hfShape;
        cms[i].tcRef0 = NUM_MESH_XF + i;   // one capsule frame per case
        cms[i].tcRef1 = specs[i].meshXf;
    }

    std::vector<uint8_t> pairBlob(size_t(n) * 56, 0);
    std::vector<uint32_t> startIdx(n), roundedIdx(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        startIdx[i] = 10u * i;
        roundedIdx[i] = 300u + 7u * i;
        uint8_t* r = pairBlob.data() + size_t(i) * 56;
        const uint32_t count = 3u + i, cmIndex = 40u + i;
        std::memcpy(r + 28, &startIdx[i], 4);
        std::memcpy(r + 32, &count, 4);
        std::memcpy(r + 36, &cmIndex, 4);
        std::memcpy(r + 40, &roundedIdx[i], 4);
    }

    struct Pair { uint32_t cm, tri, off, pad; };
    std::vector<Pair> pairs(n);
    for (uint32_t i = 0; i < n; ++i)
        pairs[i] = { i, specs[i].tri, specs[i].testOffset, 0u };

    // ---- solve the tie cases' contact distance -------------------------------
    //
    // `sqInflatedRadius > sqDist` and `>=` part company on exactly one input, and
    // hand-chosen geometry never lands on it: `sqDist` is a sum of squares of
    // differences of rotated coordinates, so it is not a number one can aim at.  It
    // is, however, INDEPENDENT of the contact distance -- the radius enters the core
    // only through the comparison -- so the tie can be solved from the other side:
    // run the mirror once, read the sqDist it produced, and walk this case's own
    // contactDistance slot (each case has one nobody else reads) through the floats
    // around `sqrt(sqDist) - radius - meshTerm` until the squared inflated radius is
    // BITWISE equal.  The sum is spelled in the engine's order -- cd[ref0] + cd[ref1],
    // then radius + that -- because a differently associated sum lands on a different
    // float and the tie would be lost.
    auto runMirror = [&](uint32_t i) {
        const CmInput& cm = cms[i];
        const float* r0 = xf.data() + size_t(cm.tcRef0) * 8;
        const float* r1 = xf.data() + size_t(cm.tcRef1) * 8;
        return capsuleCore(hf, shapes[cm.shapeRef0], shapes[cm.shapeRef1],
                           V4{ r0[0], r0[1], r0[2], r0[3] }, V3{ r0[4], r0[5], r0[6] },
                           V4{ r1[0], r1[1], r1[2], r1[3] }, V3{ r1[4], r1[5], r1[6] },
                           cdist[cm.tcRef0] + cdist[cm.tcRef1],
                           specs[i].tri, startIdx[i], roundedIdx[i], specs[i].testOffset);
    };
    for (uint32_t i = 0; i < n; ++i)
    {
        if (specs[i].tune != TUNE_TIE_SQ) continue;
        const float sqDist = runMirror(i).sqDist;
        PXG_CHECK_MSG(sqDist > 0.0f, "case %u: a tie case needs a POSITIVE sqDist", i);

        const float radius = shapes[specs[i].capsuleShape].sy;
        const float meshTerm = cdist[specs[i].meshXf];
        const float seed = std::sqrt(sqDist) - radius - meshTerm;
        auto ties = [&](float c) {
            const float infl = radius + (c + meshTerm);
            return infl * infl == sqDist;
        };
        bool solved = ties(seed);
        if (solved) cdist[NUM_MESH_XF + i] = seed;
        float up = seed, dn = seed;
        for (int k = 0; k < 4096 && !solved; ++k)
        {
            up = std::nextafterf(up, 1e30f);
            dn = std::nextafterf(dn, -1e30f);
            if (ties(up))      { cdist[NUM_MESH_XF + i] = up; solved = true; }
            else if (ties(dn)) { cdist[NUM_MESH_XF + i] = dn; solved = true; }
        }
        PXG_CHECK_MSG(solved,
                      "case %u: no contact distance within 4096 ulps makes "
                      "sqInflatedRadius bitwise equal to sqDist (%.9g) -- `>` vs `>=` "
                      "is then untested", i, double(sqDist));
        const CoreOut chk = runMirror(i);
        PXG_CHECK_MSG(chk.sqInflatedRadius == chk.sqDist,
                      "case %u: the solved tie did not hold (%.9g vs %.9g)", i,
                      double(chk.sqInflatedRadius), double(chk.sqDist));
        PXG_CHECK_MSG(chk.branch == PN_NONE,
                      "case %u: an exact tie must FAIL the strict gate -- branch is %d",
                      i, int(chk.branch));
    }

    // ---- mirror ------------------------------------------------------------
    std::vector<CoreOut> exp(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const CmInput& cm = cms[i];
        const float* r0 = xf.data() + size_t(cm.tcRef0) * 8;
        const float* r1 = xf.data() + size_t(cm.tcRef1) * 8;
        exp[i] = capsuleCore(hf, shapes[cm.shapeRef0], shapes[cm.shapeRef1],
                             V4{ r0[0], r0[1], r0[2], r0[3] }, V3{ r0[4], r0[5], r0[6] },
                             V4{ r1[0], r1[1], r1[2], r1[3] }, V3{ r1[4], r1[5], r1[6] },
                             cdist[cm.tcRef0] + cdist[cm.tcRef1],
                             specs[i].tri, startIdx[i], roundedIdx[i], specs[i].testOffset);
    }

    uint32_t maxOff = 0, maxPad = 0;
    for (uint32_t i = 0; i < n; ++i)
    { maxOff = std::max(maxOff, exp[i].pairOffset); maxPad = std::max(maxPad, exp[i].pairOffsetPadded); }
    const uint32_t slots = std::max(maxOff, maxPad) + 1u;

    const uint32_t U_SENT = 0xdeadbeefu;
    const float F_SENT = -98765.0f;
    std::vector<uint32_t> niInit(size_t(slots) * 4, U_SENT);
    std::vector<float> mdInit(slots, F_SENT);
    std::vector<uint32_t> imInit(slots, U_SENT), ordInit(slots, U_SENT);

    PxgBufferHandle bShapes = createBufferFrom(dev, shapes);
    PxgBufferHandle bXf = createBufferFrom(dev, xf);
    PxgBufferHandle bCd = createBufferFrom(dev, cdist);
    PxgBufferHandle bCm = createBufferFrom(dev, cms);
    PxgBufferHandle bPairBlob = createBufferFrom(dev, pairBlob);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairs);
    PxgBufferHandle bNi = createBufferFrom(dev, niInit);
    PxgBufferHandle bMd = createBufferFrom(dev, mdInit);
    PxgBufferHandle bIm = createBufferFrom(dev, imInit);
    PxgBufferHandle bOrd = createBufferFrom(dev, ordInit);
    std::vector<uint32_t> outInit(size_t(n) * 12, U_SENT);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    Push push{};
    push.cmInputs = dev->getBufferDeviceAddress(bCm);
    push.transformCache = dev->getBufferDeviceAddress(bXf);
    push.contactDistance = dev->getBufferDeviceAddress(bCd);
    push.gpuShapes = dev->getBufferDeviceAddress(bShapes);
    push.sphereTrimeshPair = dev->getBufferDeviceAddress(bPairBlob);
    push.ni = dev->getBufferDeviceAddress(bNi);
    push.maxDepth = dev->getBufferDeviceAddress(bMd);
    push.interm = dev->getBufferDeviceAddress(bIm);
    push.ordered = dev->getBufferDeviceAddress(bOrd);
    push.pairs = dev->getBufferDeviceAddress(bPairs);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numPairs = n;

    {
        Kernel k("capsuleHfCoreBda.spv", "capsuleHfCoreLaunch", {}, sizeof(Push));
        k.dispatch((n + 63u) / 64u, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto gotOut = downloadBuffer<uint32_t>(dev, bOut, outInit.size());
    const auto gotNi = downloadBuffer<uint32_t>(dev, bNi, niInit.size());
    const auto gotMd = downloadBuffer<uint32_t>(dev, bMd, slots);
    const auto gotIm = downloadBuffer<uint32_t>(dev, bIm, slots);
    const auto gotOrd = downloadBuffer<uint32_t>(dev, bOrd, slots);

    size_t bad = 0;
    size_t branchCount[4] = { 0, 0, 0, 0 };
    size_t nbCount[3] = { 0, 0, 0 };
    size_t sawDistNeg = 0, sawTooFar = 0, sawWrongHanded = 0, sawRotatedHf = 0;
    size_t sawRadiusNotHalfHeight = 0;

    for (uint32_t i = 0; i < n; ++i)
    {
        const CoreOut& e = exp[i];
        const uint32_t* r = gotOut.data() + size_t(i) * 12;
        auto chkU = [&](const char* w, uint32_t g, uint32_t x) {
            if (g != x) { ++bad; if (bad <= 24)
                std::printf("  [case %u %s] %s: got=0x%x exp=0x%x\n", i, specs[i].why, w, g, x); } };
        auto chkF = [&](const char* w, uint32_t gBits, float x) {
            if (!sameBitsU(gBits, x)) { float g; std::memcpy(&g, &gBits, 4); ++bad; if (bad <= 24)
                std::printf("  [case %u %s] %s: got=%.9g exp=%.9g\n", i, specs[i].why, w,
                            double(g), double(x)); } };

        chkU("nb", r[0], e.nb);
        chkU("pairOffset", r[1], e.pairOffset);
        for (uint32_t k = 0; k < 2; ++k)
        {
            const uint32_t* f = r + 4 + k * 4;
            const float x[4] = { e.pt[k].x, e.pt[k].y, e.pt[k].z, e.sep[k] };
            static const char* nm[4] = { "pt.x", "pt.y", "pt.z", "sep" };
            for (uint32_t j = 0; j < 4; ++j) chkF(nm[j], f[j], x[j]);
        }
        chkF("maxDepth", gotMd[e.pairOffset], e.maxDepth);
        chkU("interm", gotIm[e.pairOffset], e.interm);
        chkU("ordered", gotOrd[e.pairOffsetPadded], e.ordered);
        chkF("ni.x", gotNi[4 * e.pairOffset + 0], e.niNormal.x);
        chkF("ni.y", gotNi[4 * e.pairOffset + 1], e.niNormal.y);
        chkF("ni.z", gotNi[4 * e.pairOffset + 2], e.niNormal.z);
        chkU("ni.index", gotNi[4 * e.pairOffset + 3], e.niIndex);

        branchCount[e.branch]++;
        if (e.nb < 3) nbCount[e.nb]++;
        if (e.distNegative) ++sawDistNeg;
        if (!e.distNegative && e.tooFar) ++sawTooFar;
        {
            const GpuShape& hs = shapes[specs[i].hfShape];
            if (hs.sx < 0.0f || hs.sz < 0.0f) ++sawWrongHanded;
            const float* r1 = xf.data() + size_t(cms[i].tcRef1) * 8;
            if (r1[0] != 0.0f || r1[1] != 0.0f || r1[2] != 0.0f) ++sawRotatedHf;
            const GpuShape& cs = shapes[specs[i].capsuleShape];
            if (cs.sx != cs.sy) ++sawRadiusNotHalfHeight;
        }
    }

    {
        std::vector<bool> claimed(slots, false), claimedOrd(slots, false);
        for (uint32_t i = 0; i < n; ++i)
        { claimed[exp[i].pairOffset] = true; claimedOrd[exp[i].pairOffsetPadded] = true; }
        size_t stray = 0;
        for (uint32_t k = 0; k < slots; ++k)
        {
            if (!claimed[k])
            {
                if (!sameBitsU(gotMd[k], F_SENT)) ++stray;
                if (gotIm[k] != U_SENT) ++stray;
                for (uint32_t w = 0; w < 4; ++w) if (gotNi[4 * k + w] != U_SENT) ++stray;
            }
            if (!claimedOrd[k] && gotOrd[k] != U_SENT) ++stray;
        }
        if (stray) { bad += stray;
            std::printf("  %zu record slots written outside their pair offset\n", stray); }
    }

    dev->destroyBuffer(bHf); dev->destroyBuffer(bShapes); dev->destroyBuffer(bXf);
    dev->destroyBuffer(bCd); dev->destroyBuffer(bCm); dev->destroyBuffer(bPairBlob);
    dev->destroyBuffer(bPairs); dev->destroyBuffer(bNi); dev->destroyBuffer(bMd);
    dev->destroyBuffer(bIm); dev->destroyBuffer(bOrd); dev->destroyBuffer(bOut);

    if (std::getenv("PXG_CAPHF_DUMP"))
    {
        for (uint32_t hs = 0; hs < 5; ++hs) if (shapes[hs].type == 9u)
            for (uint32_t t = 0; t < 2u * (hf.rows - 1u) * hf.cols; ++t)
            {
                V3 a, b, c; uint32_t ig[3];
                const GpuShape& s = shapes[hs];
                getTriangle(hf, t, V3{ s.sx, s.sy, s.sz }, a, b, c, ig);
                const V3 nn = normalized(cross3(sub(b, a), sub(c, a)));
                const float res = dot3(sub(a, b), nn);
                if (res < 0.0f)
                    std::printf("  residual hfShape %u tri %2u : %.9g\n", hs, t, double(res));
            }
        static const char* bn[4] = { "selected", "faceFallb", "DEFERRED ", "none     " };
        for (uint32_t i = 0; i < n; ++i)
        {
            const CoreOut& e = exp[i];
            std::printf("  [%2u] tri %2u %s nb=%u  dist=%- 12.6g sqDist=%- 12.6g "
                        "sqInflR=%- 10.6g u=%- 11.6g v=%- 11.6g u+v=%- 11.6g "
                        "flags=%c%c%c  N=(%.4f %.4f %.4f) pN=(%.4f %.4f %.4f)\n",
                        i, specs[i].tri, bn[e.branch], e.nb,
                        double(e.dist), double(e.sqDist), double(e.sqInflatedRadius),
                        double(e.u), double(e.v), double(e.u + e.v),
                        (e.triFlags & ETD_01) ? '1' : '.',
                        (e.triFlags & ETD_12) ? '2' : '.',
                        (e.triFlags & ETD_20) ? '3' : '.',
                        double(e.triNormal.x), double(e.triNormal.y), double(e.triNormal.z),
                        double(e.patchNormal.x), double(e.patchNormal.y),
                        double(e.patchNormal.z));
        }
    }

    std::printf("  branches: selected %zu, face-fallback %zu, deferred %zu, none %zu; "
                "contacts 0/%zu 1/%zu 2/%zu\n",
                branchCount[PN_SELECTED], branchCount[PN_FACE_FALLBACK],
                branchCount[PN_DEFERRED], branchCount[PN_NONE],
                nbCount[0], nbCount[1], nbCount[2]);

    PXG_CHECK_MSG(bad == 0, "capsuleHfCore: %zu mismatches", bad);
    PXG_CHECK_MSG(sawRotatedHf > 0,
                  "no case has a ROTATED heightfield actor -- upstream candidate 50, the "
                  "capsule axis left in world space, is then invisible");
    PXG_CHECK_MSG(sawRadiusNotHalfHeight > 0,
                  "no capsule has halfHeight != radius -- which component each is read from "
                  "is then untested");
    PXG_CHECK_MSG(sawWrongHanded > 0, "no case uses a negative scale (wrongHanded untested)");
    PXG_CHECK_MSG(sawDistNeg > 0, "no case sits BEHIND the triangle plane");
    PXG_CHECK_MSG(sawTooFar > 0, "no case is outside the inflated radius");
    PXG_CHECK_MSG(branchCount[PN_SELECTED] > 0, "selectNormal never accepted the face normal");
    PXG_CHECK_MSG(branchCount[PN_FACE_FALLBACK] > 0,
                  "no case takes the sqDist > 0 fallback -- the middle arm of the "
                  "patch-normal choice is then untested");
    PXG_CHECK_MSG(branchCount[PN_DEFERRED] > 0,
                  "no case reaches the DEFERRED arm -- which is the only writer of "
                  "`deferred`, so bit 31 of the ordered word is stuck at one value and "
                  "upstream candidate 52's contrast with the sphere arm's NI mask is "
                  "unobserved");
    PXG_CHECK_MSG(nbCount[1] > 0,
                  "no case emits exactly ONE contact -- the loop that transforms the "
                  "contacts back into the capsule's frame then never runs a partial "
                  "trip, and slot 1 keeping its initial value is untested");
    PXG_CHECK_MSG(nbCount[2] > 0, "no case emits TWO contacts");
    PXG_CHECK_MSG(nbCount[0] > 0, "no case emits none");

    // ---- the boundary cases the first sweep's survivors asked for -------------
    //
    // Each of these restates, as a property of the INPUTS, what the corresponding
    // mutant needs in order to be distinguishable.  Stated on the outputs they would
    // be vacuous: the whole reason those mutants survived is that the quantities they
    // change do not reach the outputs from any ordinary input.
    {
        size_t exactZeroDist = 0, negV1Residual = 0, tieExact = 0, betweenSqAndLin = 0;
        size_t cdistSplit = 0, uvAsymmetric = 0, lastButOneRow = 0;
        for (uint32_t i = 0; i < n; ++i)
        {
            const CoreOut& e = exp[i];
            const Spec& sp = specs[i];

            if (e.dist == 0.0f) ++exactZeroDist;

            // `dist-from-v1`: the mutant measures the plane from v1, whose distance
            // from it is the rounding residue of dot(ab, N).  It only diverges where
            // the residue's SIGN flips the gate, i.e. where dist is exactly 0 and the
            // residue is negative.
            {
                V3 a, b, c; uint32_t ig[3];
                const GpuShape& hs = shapes[sp.hfShape];
                getTriangle(hf, sp.tri, V3{ hs.sx, hs.sy, hs.sz }, a, b, c, ig);
                const V3 nrm = normalized(cross3(sub(b, a), sub(c, a)));
                const float residual = dot3(sub(a, b), nrm);
                if (e.dist == 0.0f && residual < 0.0f) ++negV1Residual;
            }

            if (e.sqInflatedRadius == e.sqDist) ++tieExact;
            if (e.sqDist > e.sqInflatedRadius && e.sqDist < e.inflatedRadius)
                ++betweenSqAndLin;

            // `cdist-one-side`: the gate must accept with both contact-distance terms
            // and reject with only the first.
            {
                const float one = shapes[sp.capsuleShape].sy + cdist[cms[i].tcRef0];
                if (e.dist >= 0.0f && e.sqInflatedRadius > e.sqDist && !(one * one > e.sqDist))
                    ++cdistSplit;
            }

            // `selnormal-uv`: u and v must land in DIFFERENT regions of the
            // classifier, and it must be the deferred arm that reads the answer.
            {
                const float zero = 1e-6f, one = 0.999999f;
                auto region = [&](float x) { return zero > x ? 0 : (x > one ? 2 : 1); };
                if (e.branch != PN_NONE && e.sqDist == 0.0f
                    && region(e.u) != region(e.v)) ++uvAsymmetric;
            }

            // `rows-cols-swap`: `rows` is read only as `cell / c != r - 2`.
            if (e.branch != PN_NONE && (sp.tri >> 1) / hf.cols == hf.rows - 3u)
                ++lastButOneRow;
        }

        PXG_CHECK_MSG(exactZeroDist > 0,
                      "no case has `dist` EXACTLY 0.0f -- `dist >= 0` and `dist > 0` are "
                      "then the same predicate");
        PXG_CHECK_MSG(negV1Residual > 0,
                      "no exactly-zero-dist case sits on a triangle whose dot(ab, N) "
                      "residue is negative -- measuring the plane from v1 instead of v0 "
                      "is then indistinguishable");
        PXG_CHECK_MSG(tieExact > 0,
                      "no case has sqInflatedRadius bitwise equal to sqDist -- `>` and "
                      "`>=` are then the same predicate");
        PXG_CHECK_MSG(betweenSqAndLin > 0,
                      "no case has sqDist between the squared and the unsquared inflated "
                      "radius -- comparing against the unsquared one would pass");
        PXG_CHECK_MSG(cdistSplit > 0,
                      "no case is accepted by the two-term contact distance and rejected "
                      "by its first term alone");
        PXG_CHECK_MSG(uvAsymmetric > 0,
                      "no case reaches selectNormal with u and v in different regions "
                      "AND sqDist == 0 -- its argument ORDER is then unobservable, "
                      "because the other two arms both answer triNormal");
        PXG_CHECK_MSG(lastButOneRow > 0,
                      "no case enters the body from the last-but-one cell row -- `rows` "
                      "reaches the output only through `cell / c != r - 2`");
    }
}

} // namespace
