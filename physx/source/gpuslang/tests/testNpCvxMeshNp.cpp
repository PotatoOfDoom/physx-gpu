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

// Verification of npCvxMeshNp.slang (sphereTrimeshNarrowphaseCore /
// capsuleTrimeshNarrowphaseCore + the warp output stage of
// sphereTrimeshNarrowphase, convexMesh.cu, tranche 106) against exact
// CPU transcriptions.  Fixtures follow the exact-arithmetic doctrine
// (grid geometry, signed-permutation quaternions, dyadic radii); the
// only rounding ops are sqrt / rsqrt / normalize / divisions, mirrored
// with the tranche-54/55/56/68 conventions (getNormalized = v*(1/sqrt),
// the mask-from-normal path normalizes per-component division).
//
// Pinned per lane: the mesh-scaling identity fast path AND the
// negative-determinant winding/adjacency flip; the one-sided plane
// gate; the accept gates (sqInflatedRadius > sqDist STRICT — a
// vertex-region family places the center at EXACTLY radius+cd from the
// vertex, excluded, and 1/64 closer, included); face vs vertex/edge
// patch normals; the 0.996 deferral with voronoi masks (vertex
// deferral is NOT an edge contact, edge deferral is); the nonconvex
// adjacency block; the SphereTriNormalAndIndex packing (deferred bit
// 31, count << 27) with the sphere core using the REMAPPED cpu index
// while the capsule core FAITHFULLY uses the raw gpu index and the
// meshToCapsule-rotated normal; the ordered-index search flags
// (eFACE_CONTACT for undeferred hits, eEDGE_CONTACT for deferred edge
// hits, plain otherwise) at the PADDED offset; capsule contacts from
// endpoint projections + segment-edge tests (cap 2) transformed into
// capsule space with separation = min.
//
// Pinned per warp batch: the single second-pass and temp-contact
// reservations with exclusive-scan lane offsets (per-batch contiguity,
// disjoint-partition of both streams), ConvexTriContacts.index
// linkage, and the FAITHFUL batch-drop quirk — when NO lane of a batch
// produced a contact, the batch's deferred outMask entries are dropped
// (a dedicated all-deferred-zero-contact batch pins it).
//
// Index spaces are decoupled: pair offsets come from permuted per-cm
// startIndex / roundedStartIndex tables (different permutations, so a
// port reading the wrong field collides with poison), triangleIdx != 0
// with pool pads, gti is a distinct tagged value (bit 28 set), face
// remap carries distinctive values.
//
// KNOWN LIMIT: the deferral boundary (cosTheta <= 0.996 vs < 0.996) is
// unpinnable with exact arithmetic — the fixtures produce cosTheta of
// exactly 1 (parallel, not deferred) or exactly 0 (perpendicular,
// deferred), never the irrational 0.996 threshold, so the strict/loose
// distinction is an equivalent mutant (cf. the 0.9998 quat gate of
// tranche 103).

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
struct CmCase
{
    F4 aQ, aP, mQ, mP, scale, scaleRot;
    U4 pair, mesh;
};
struct CmParams { uint32_t numCases, pad0, pad1, pad2; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
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

static const float FLT_BIG = 3.402823466e38f;
static const float FLT_EPS = 1.192092896e-07f;
static const uint32_t BOUNDARY = 0xffffffffu;
static const uint32_t NONCONVEX = 0x80000000u;
static const uint32_t eV0 = 1u << 29, eV1 = 2u << 29, eV2 = 3u << 29;
static const uint32_t eE01 = 4u << 29, eE02 = 5u << 29, eE12 = 6u << 29;
static const uint32_t DEFER = 0x80000000u;
static const uint32_t FACE_FLAG = 0x80000000u, EDGE_FLAG = 0xc0000000u;
static const uint32_t POISON = 0xCDCDCDCDu;

// getNormalized (rsqrt convention: v * (1/sqrt(m)), zero fallback)
static V3 getNormalizedRef(V3 v)
{
    const float m = dot3(v, v);
    if (!(m > 0.0f)) return { 0, 0, 0 };
    const float r = 1.0f / std::sqrt(m);
    return mul(v, r);
}
// normalize() intrinsic convention (per-component division)
static V3 nrmDiv(V3 a)
{
    const float m = std::sqrt(dot3(a, a));
    return { a.x / m, a.y / m, a.z / m };
}
static V3 vertex2ShapeRef(V3 v, V3 scale, F4 rot)
{
    V3 t = quatRotate(rot, v);
    t.x *= scale.x; t.y *= scale.y; t.z *= scale.z;
    return quatRotateInv(rot, t);
}
// inverse of vertex2Shape (exact for |scale| a power of two): divides by scale
static V3 shape2VertexRef(V3 v, V3 scale, F4 rot)
{
    V3 t = quatRotate(rot, v);
    t.x /= scale.x; t.y /= scale.y; t.z /= scale.z;
    return quatRotateInv(rot, t);
}
static bool isEdgeNonconvexRef(uint32_t a) { return (a != BOUNDARY) && ((a & NONCONVEX) != 0u); }
static bool isValidBaryRef(float v, float w)
{
    const float eps = 1e-6f;
    return (v >= -eps && v <= 1.0f + eps) && (w >= -eps && w <= 1.0f + eps) && ((v + w) <= 1.0f + eps);
}

// P9.36: how many FACE-region cases would actually separate the engine's association
// (scalars scaled first) from the sum-then-scale spelling this mirror used to carry.
// Reset by the caller before each sweep; asserted afterwards.
static size_t gFaceDistinguish = 0;
static size_t gFaceMixedComponent = 0;
static uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static float distPointTriRef(V3 p, V3 a, V3 b, V3 c, U4 adj, V3& closestP,
                             bool& generateContact, bool& faceContact, uint32_t& mask)
{
    faceContact = false;
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    const float vc = d1 * d4 - d3 * d2;
    const float vb = d5 * d2 - d1 * d6;
    const float va = d3 * d6 - d5 * d4;
    // sphereTriangle.cuh:96's literal `1e-5` carries NO SUFFIX, so the multiply
    // happens in DOUBLE and only the assignment narrows -- see the twin note in
    // the shader.  `double(1.0f)/double(100000.0f)` is the C++ literal bit for bit.
    const double D1E_5 = double(1.0) / double(100000.0);
    const float edgeTol = float(double(va + vb + vc) * D1E_5);
    if (d1 < 0.0f && d2 < 0.0f)
    {
        mask = eV0;
        generateContact = (!isEdgeNonconvexRef(adj.x)) || (!isEdgeNonconvexRef(adj.z));
        closestP = a;
        return dot3(ap, ap);
    }
    if (d3 > 0.0f && d4 < d3)
    {
        mask = eV1;
        generateContact = (!isEdgeNonconvexRef(adj.x)) || (!isEdgeNonconvexRef(adj.y));
        closestP = b;
        return dot3(bp, bp);
    }
    if (d6 > 0.0f && d5 < d6)
    {
        mask = eV2;
        generateContact = (!isEdgeNonconvexRef(adj.y)) || (!isEdgeNonconvexRef(adj.z));
        closestP = c;
        return dot3(cp, cp);
    }
    if (vc < 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        mask = eE01;
        generateContact = !isEdgeNonconvexRef(adj.x) || vc > -edgeTol;
        const float v = d1 / (d1 - d3);
        closestP = add(a, mul(ab, v));
        const V3 d = sub(closestP, p);
        return dot3(d, d);
    }
    if (vb < 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        mask = eE02;
        generateContact = !isEdgeNonconvexRef(adj.z) || vb > -edgeTol;
        const float w = d2 / (d2 - d6);
        closestP = add(a, mul(ac, w));
        const V3 d = sub(closestP, p);
        return dot3(d, d);
    }
    if (va < 0.0f && d4 >= d3 && d5 >= d6)
    {
        mask = eE12;
        generateContact = !isEdgeNonconvexRef(adj.y) || va > -edgeTol;
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closestP = add(b, mul(sub(c, b), w));
        const V3 d = sub(closestP, p);
        return dot3(d, d);
    }
    generateContact = true;
    faceContact = true;
    mask = 0u;
    // THE MIRROR IS WRITTEN FROM THE CUDA (P9.53) -- see the twin note in
    // testNpCvxHfNp.cpp.  sphereTriangle.cuh:138 sums the two scaled edge vectors
    // and scales the sum once, then adds `a` once.
    const float denom = 1.0f / (va + vb + vc);
    closestP = add(a, mul(add(mul(ab, vb), mul(ac, vc)), denom));
    // ...and this counts how many of these cases could tell the CUDA's association
    // apart from the one P9.36 imposed -- i.e. how much of this branch the fixture
    // is able to police at all.  On this fixture's axis-aligned geometry the answer
    // is ZERO, and that is asserted below rather than assumed.
    {
        // The spelling P9.36 put here and P9.53 removed: GuDistancePointTriangle.h
        // :93-97, two scaled scalars and two adds.
        const V3 p936Form = add(add(a, mul(ab, vb * denom)), mul(ac, vc * denom));
        // diagnosis probe: the two spellings coincide IDENTICALLY when, in every
        // component, at most one of ab/ac is non-zero -- then ab.x*vb + ac.x*vc has a
        // single term and (t)*denom == ab.x*(vb*denom) trivially.  Axis-aligned grid
        // triangles are exactly that shape.
        {
            int mixed = 0;
            if (ab.x != 0.0f && ac.x != 0.0f) ++mixed;
            if (ab.y != 0.0f && ac.y != 0.0f) ++mixed;
            if (ab.z != 0.0f && ac.z != 0.0f) ++mixed;
            if (mixed > 0) ++gFaceMixedComponent;
        }
        if (bitsOf(p936Form.x) != bitsOf(closestP.x) || bitsOf(p936Form.y) != bitsOf(closestP.y) ||
            bitsOf(p936Form.z) != bitsOf(closestP.z))
            ++gFaceDistinguish;
    }
    const V3 d = sub(closestP, p);
    return dot3(d, d);
}

static void segSegRef(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const V3 r = sub(p1, p2);
    const float a = dot3(d1, d1), e = dot3(d2, d2), b = dot3(d1, d2);
    const float c = dot3(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = dot3(d2, r);
    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = std::min(std::max(temp / denom, 0.0f), 1.0f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = std::min(std::max(tTmp, 0.0f), 1.0f);
    const float comp = (b * t2 - c) * aRecip;
    s = std::min(std::max(comp, 0.0f), 1.0f);
    t = t2;
}

static float distSegTriRef(V3 p, V3 q, V3 a, V3 b, V3 c, float& outT, float& outU, float& outV)
{
    const V3 pq = sub(q, p), ab = sub(b, a), ac = sub(c, a), bc = sub(c, b);
    const V3 ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : FLT_BIG;
    const V3 n = getNormalizedRef(cross3(ab, ac));
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
        if (isValidBaryRef(v00, w00)) { outT = t0; outU = v00; outV = w00; return 0.0f; }
    }
    const V3 closestP31 = sub(p, mul(n, dist3));
    const V3 pV20 = sub(closestP31, a);
    const float pD20 = dot3(pV20, ab), pD21 = dot3(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = isValidBaryRef(v0, w0);
    const V3 closestP41 = sub(q, mul(n, dist4));
    const V3 qV20 = sub(closestP41, a);
    const float qD20 = dot3(qV20, ab), qD21 = dot3(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = isValidBaryRef(v1, w1);
    if (con0 && con1)
    {
        if (sqDist4 > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        outT = 1.0f; outU = v1; outV = w1; return sqDist4;
    }
    float t00, t10, t01, t11, t02, t12;
    segSegRef(p, pq, a, ab, t00, t10);
    segSegRef(p, pq, b, bc, t01, t11);
    segSegRef(p, pq, a, ac, t02, t12);
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
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0) { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; }
    else if (sqDist2 > sqDist1)                 { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; }
    else                                        { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; }
    if (con0)
    {
        if (sqDistPE > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    if (con1)
    {
        if (sqDistPE > sqDist4) { outT = 1.0f; outU = v1; outV = w1; return sqDist4; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    outT = tSeg; outU = uEdge; outV = vEdge;
    return sqDistPE;
}

static uint32_t maskFromNormalRef(V3 v0, V3 v1, V3 v2, V3 n)
{
    const V3 e01 = nrmDiv(sub(v1, v0));
    const V3 e02 = nrmDiv(sub(v2, v0));
    const V3 e12 = nrmDiv(sub(v2, v1));
    const float d01 = dot3(e01, n), d02 = dot3(e02, n), d12 = dot3(e12, n);
    const V3 tn = nrmDiv(cross3(e01, e02));
    const float eps = 0.0174f;
    if (d01 < -eps && d02 < -eps) return eV0;
    if (d01 > eps && d12 < -eps)  return eV1;
    if (d02 > eps && d12 > eps)   return eV2;
    if (dot3(nrmDiv(cross3(tn, e01)), n) < -eps) return eE01;
    if (dot3(nrmDiv(cross3(tn, e02)), n) > eps)  return eE02;
    if (dot3(nrmDiv(cross3(tn, e12)), n) < -eps) return eE12;
    return 0u;
}

static uint32_t genContactsRef(V3 a, V3 b, V3 c, V3 planeNormal, V3 normal, V3 p, V3 q,
                               float inflatedRadius, V3 cts[2], float seps[2])
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
    const float tdenom = d00 * d11 - d01 * d01;
    const float bdenom = tdenom > 0.0f ? 1.0f / tdenom : FLT_BIG;
    const float inomp = dot3(planeNormal, neg(ap));
    const float ideom = dot3(planeNormal, normal);
    const float ipt = ideom > 0.0f ? -(inomp / ideom) : FLT_BIG;
    const V3 cp31 = sub(p, mul(normal, ipt));
    const V3 pV20 = sub(cp31, a);
    const float pD20 = dot3(pV20, ab), pD21 = dot3(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    uint32_t count = 0;
    if (isValidBaryRef(v0, w0) && (inflatedRadius > ipt)) { cts[count] = p; seps[count] = ipt; count++; }
    const float inomq = dot3(planeNormal, neg(aq));
    const float iqt = ideom > 0.0f ? -(inomq / ideom) : FLT_BIG;
    const V3 cp41 = sub(q, mul(normal, iqt));
    const V3 qV20 = sub(cp41, a);
    const float qD20 = dot3(qV20, ab), qD21 = dot3(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    if (isValidBaryRef(v1, w1) && (inflatedRadius > iqt)) { cts[count] = q; seps[count] = iqt; count++; }
    return count;
}

static void genEERef(V3 p, V3 q, float sqIR, V3 normal, V3 a, V3 b,
                     V3 cts[2], float seps[2], uint32_t& count)
{
    const V3 ab = sub(b, a);
    const V3 n = cross3(ab, normal);
    const float d = dot3(n, a), np = dot3(n, p), nq = dot3(n, q);
    const float signP = np - d, signQ = nq - d;
    if (signP * signQ > 0.0f) return;
    const V3 pq = sub(q, p);
    const float npq = dot3(n, pq);
    if (npq == 0.0f) return;
    const float segT = (d - np) / npq;
    const V3 localA = add(p, mul(pq, segT));
    const V3 perNormal = cross3(normal, pq);
    const V3 ap = sub(localA, a);
    const float nom = dot3(perNormal, ap);
    const float denom = dot3(perNormal, ab);
    const float tValue = nom / denom;
    if (tValue > 1.0f || tValue < 0.0f) return;
    const V3 v = sub(ap, mul(ab, tValue));
    const float sqDist = dot3(v, v);
    if (sqIR > sqDist)
    {
        cts[count] = localA;
        seps[count] = dot3(v, normal);
        count++;
    }
}
static void genEEContactsRef(V3 a, V3 b, V3 c, V3 normal, V3 p, V3 q, float sqIR,
                             V3 cts[2], float seps[2], uint32_t& count)
{
    if (count < 2) genEERef(p, q, sqIR, normal, a, b, cts, seps, count);
    if (count < 2) genEERef(p, q, sqIR, normal, b, c, cts, seps, count);
    if (count < 2) genEERef(p, q, sqIR, normal, a, c, cts, seps, count);
}

// ---------------------------------------------------------------------------
struct CmMesh
{
    std::vector<F4> verts;
    std::vector<U4> tris, adj;
    std::vector<uint32_t> remap;
};

struct CmRef
{
    uint32_t nb = 0, om = 0;
    V3 cts[2];
    float seps[2] = { 0, 0 };
    float maxDepth = FLT_BIG;
    V3 niNormal{};
    uint32_t niIndex = 0, ordered = 0, pairOffset = 0, padOffset = 0;
    bool delay = false, edge = false;
    // P9.36: does this case land in the FACE region of distPointTriRef?  That is the
    // one branch whose arithmetic (the barycentric tail of
    // GuDistancePointTriangle.h:94-97) this fixture was silently not policing --
    // correcting the shader's association there left the suite green.  Counted so a
    // coverage assertion can say whether the branch is reached at all.
    bool faceRegion = false;
};

static void getShapeSpaceVertsRef(const CmCase& cs, const CmMesh& mesh, uint32_t triangleIdx,
                                  V3& v0, V3& v1, V3& v2, U4& adj, bool& flip)
{
    const V3 sc = v3(cs.scale);
    const bool idt = (sc.x == 1.0f && sc.y == 1.0f && sc.z == 1.0f);
    flip = (sc.x * sc.y * sc.z) < 0.0f;
    const U4 tri = mesh.tris[cs.mesh.y + triangleIdx];
    V3 tv0 = v3(mesh.verts[cs.mesh.x + tri.x]);
    V3 tv1 = v3(mesh.verts[cs.mesh.x + tri.y]);
    V3 tv2 = v3(mesh.verts[cs.mesh.x + tri.z]);
    if (idt) { v0 = tv0; v1 = tv1; v2 = tv2; }
    else
    {
        tv0 = vertex2ShapeRef(tv0, sc, cs.scaleRot);
        tv1 = vertex2ShapeRef(tv1, sc, cs.scaleRot);
        tv2 = vertex2ShapeRef(tv2, sc, cs.scaleRot);
        v0 = tv0;
        v1 = flip ? tv2 : tv1;
        v2 = flip ? tv1 : tv2;
    }
    adj = mesh.adj[cs.mesh.y + triangleIdx];
    if (flip) std::swap(adj.x, adj.z);
}

static CmRef refSphere(const CmCase& cs, const CmMesh& mesh, const std::vector<U4>& pairRec)
{
    CmRef ro;
    const uint32_t triangleIdx = cs.pair.y;
    const U4 rec = pairRec[cs.pair.x];
    ro.pairOffset = rec.x + cs.pair.z;
    ro.padOffset = rec.y + cs.pair.z;
    const float contactDist = cs.aP.w;
    const float radius = cs.scale.w;
    const V3 center = quatRotateInv(cs.mQ, sub(v3(cs.aP), v3(cs.mP)));
    V3 v0, v1, v2;
    U4 adj;
    bool flip;
    getShapeSpaceVertsRef(cs, mesh, triangleIdx, v0, v1, v2, adj, flip);
    const V3 n = getNormalizedRef(cross3(sub(v1, v0), sub(v2, v0)));
    const uint32_t remapCpu = mesh.remap[cs.mesh.y + triangleIdx];
    const float inflated = radius + contactDist;
    const float d = dot3(v0, n);
    const float dist0 = dot3(center, n) - d;
    float separation = FLT_BIG;
    V3 patchNormal = { 0, 0, 0 };
    if (dist0 >= 0.0f)
    {
        V3 closestP;
        bool gen = false, face = false;
        uint32_t mask;
        const float sqDist = distPointTriRef(center, v0, v1, v2, adj, closestP, gen, face, mask);
        const float sqInflated = inflated * inflated;
        if (sqInflated > sqDist && gen)
        {
            ro.faceRegion = face;                                     // P9.36
            patchNormal = n;
            if (!face) patchNormal = getNormalizedRef(sub(center, closestP));
            const float cosTheta = dot3(patchNormal, n);
            ro.delay = cosTheta <= 0.996f;
            if (ro.delay)
            {
                ro.om = mask | cs.pair.w;
                ro.edge = (mask & 0x80000000u) != 0u;
            }
            ro.nb = 1;
            separation = std::sqrt(sqDist);
            ro.seps[0] = separation;
            ro.cts[0] = { 0, 0, 0 };
        }
    }
    ro.maxDepth = separation;
    ro.ordered = triangleIdx;
    if (ro.nb) { if (!ro.delay) ro.ordered |= FACE_FLAG; else if (ro.edge) ro.ordered |= EDGE_FLAG; }
    const V3 worldNormal = quatRotate(cs.mQ, neg(patchNormal));
    ro.niNormal = quatRotateInv(cs.aQ, worldNormal);
    ro.niIndex = (ro.delay ? DEFER : 0u) + (ro.nb << 27) + remapCpu;
    return ro;
}

static CmRef refCapsule(const CmCase& cs, const CmMesh& mesh, const std::vector<U4>& pairRec)
{
    CmRef ro;
    const uint32_t triangleIdx = cs.pair.y;
    const U4 rec = pairRec[cs.pair.x];
    ro.pairOffset = rec.x + cs.pair.z;
    ro.padOffset = rec.y + cs.pair.z;
    const float contactDist = cs.aP.w;
    const float radius = cs.scale.w;
    const float halfHeight = cs.mP.w;
    const F4 mtcQ = quatMul(quatConj(cs.aQ), cs.mQ);
    const V3 mtcP = quatRotateInv(cs.aQ, sub(v3(cs.mP), v3(cs.aP)));
    const V3 tmp = mul(basis0(cs.aQ), halfHeight);
    const V3 center = quatRotateInv(cs.mQ, sub(v3(cs.aP), v3(cs.mP)));
    const V3 dir = quatRotateInv(cs.mQ, tmp);
    const V3 p0 = add(center, dir), p1 = sub(center, dir);
    V3 v0, v1, v2;
    U4 adj;
    bool flip;
    getShapeSpaceVertsRef(cs, mesh, triangleIdx, v0, v1, v2, adj, flip);
    const V3 triNormal = getNormalizedRef(cross3(sub(v1, v0), sub(v2, v0)));
    const float inflated = radius + contactDist;
    const float dist = dot3(sub(center, v0), triNormal);
    float separation = FLT_BIG;
    V3 patchNormal = { 0, 0, 0 };
    if (dist >= 0.0f)
    {
        float t, u, v;
        const float sqDist = distSegTriRef(p0, p1, v0, v1, v2, t, u, v);
        const float sqInflated = inflated * inflated;
        if (sqInflated > sqDist)
        {
            if (sqDist < FLT_EPS)
                patchNormal = triNormal;
            else
            {
                const V3 pointOnSegment = add(p0, mul(sub(p1, p0), t));
                const V3 pointOnTriangle = add(add(mul(v0, 1.0f - u - v), mul(v1, u)), mul(v2, v));
                patchNormal = getNormalizedRef(sub(pointOnSegment, pointOnTriangle));
                const float cosTheta = dot3(patchNormal, triNormal);
                ro.delay = cosTheta <= 0.996f;
            }
            if (ro.delay)
            {
                const uint32_t mask = maskFromNormalRef(v0, v1, v2, patchNormal);
                ro.om = mask | cs.pair.w;
                ro.edge = (mask & 0x80000000u) != 0u;
            }
            ro.nb = genContactsRef(v0, v1, v2, triNormal, patchNormal, p0, p1, inflated, ro.cts, ro.seps);
            genEEContactsRef(v0, v1, v2, patchNormal, p0, p1, sqInflated, ro.cts, ro.seps, ro.nb);
            for (uint32_t i = 0; i < ro.nb; ++i)
            {
                ro.cts[i] = add(quatRotate(mtcQ, ro.cts[i]), mtcP);
                separation = std::min(separation, ro.seps[i]);
            }
        }
    }
    ro.maxDepth = separation;
    ro.ordered = triangleIdx;
    if (ro.nb) { if (!ro.delay) ro.ordered |= FACE_FLAG; else if (ro.edge) ro.ordered |= EDGE_FLAG; }
    ro.niNormal = quatRotate(mtcQ, neg(patchNormal));
    ro.niIndex = (ro.delay ? DEFER : 0u) + (ro.nb << 27) + triangleIdx; // RAW gpu index
    return ro;
}

struct CmRun
{
    std::vector<F4> ni, temp, dbg;
    std::vector<float> depth;
    std::vector<uint32_t> interm, ordered, second, ctsIdx;
    uint32_t counters[2];
};
static CmRun runCm(const char* entry, const std::vector<CmCase>& cases, const CmMesh& mesh,
                   const std::vector<U4>& pairRec, uint32_t poolN, uint32_t spCap, uint32_t tcCap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = (uint32_t)cases.size();
    std::vector<F4> niInit(poolN, { -7, -7, -7, -7 });
    std::vector<float> depthInit(poolN, -7.0f);
    std::vector<uint32_t> intermInit(poolN, POISON), orderedInit(poolN, POISON);
    std::vector<uint32_t> spInit(spCap, POISON), ctsIdxInit(poolN, POISON);
    std::vector<F4> tcInit(tcCap, { -777, -777, -777, -777 });
    std::vector<uint32_t> ctrInit = { 0, 0 };
    std::vector<F4> dbgInit(n, { -7, -7, -7, -7 });

    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bV = createBufferFrom(dev, mesh.verts);
    PxgBufferHandle bT = createBufferFrom(dev, mesh.tris);
    PxgBufferHandle bA = createBufferFrom(dev, mesh.adj);
    PxgBufferHandle bR = createBufferFrom(dev, mesh.remap);
    PxgBufferHandle bP = createBufferFrom(dev, pairRec);
    PxgBufferHandle bNI = createBufferFrom(dev, niInit);
    PxgBufferHandle bD = createBufferFrom(dev, depthInit);
    PxgBufferHandle bI = createBufferFrom(dev, intermInit);
    PxgBufferHandle bO = createBufferFrom(dev, orderedInit);
    PxgBufferHandle bS = createBufferFrom(dev, spInit);
    PxgBufferHandle bTC = createBufferFrom(dev, tcInit);
    PxgBufferHandle bCI = createBufferFrom(dev, ctsIdxInit);
    PxgBufferHandle bCt = createBufferFrom(dev, ctrInit);
    PxgBufferHandle bDbg = createBufferFrom(dev, dbgInit);
    CmParams prm = { n, 0, 0, 0 };
    Kernel k("npCvxMeshNp.spv", entry,
             { (uint32_t)sizeof(CmCase), 16, 16, 16, 4, 16, 16, 4, 4, 4, 4, 16, 4, 4, 16 },
             sizeof(CmParams));
    const uint32_t batches = (n + 31) / 32;
    k.dispatch((batches + 63) / 64,
               { bC, bV, bT, bA, bR, bP, bNI, bD, bI, bO, bS, bTC, bCI, bCt, bDbg }, prm);
    dev->flush();
    CmRun r;
    r.ni = downloadBuffer<F4>(dev, bNI, poolN);
    r.depth = downloadBuffer<float>(dev, bD, poolN);
    r.interm = downloadBuffer<uint32_t>(dev, bI, poolN);
    r.ordered = downloadBuffer<uint32_t>(dev, bO, poolN);
    r.second = downloadBuffer<uint32_t>(dev, bS, spCap);
    r.temp = downloadBuffer<F4>(dev, bTC, tcCap);
    r.ctsIdx = downloadBuffer<uint32_t>(dev, bCI, poolN);
    const auto ctr = downloadBuffer<uint32_t>(dev, bCt, 2);
    r.counters[0] = ctr[0]; r.counters[1] = ctr[1];
    r.dbg = downloadBuffer<F4>(dev, bDbg, n);
    for (PxgBufferHandle h : { bC, bV, bT, bA, bR, bP, bNI, bD, bI, bO, bS, bTC, bCI, bCt, bDbg })
        dev->destroyBuffer(h);
    return r;
}

static uint32_t asU32(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

struct CmFixture
{
    std::vector<CmCase> cases;
    CmMesh mesh;
    std::vector<U4> pairRec;
    std::vector<CmRef> ref;
    uint32_t poolN = 0;
};

// the 16 exact quaternions (identity + signed-permutation halves)
static std::vector<F4> exactQuatSet()
{
    std::vector<F4> q = { { 0, 0, 0, 1 } };
    for (uint32_t s = 0; s < 16; ++s)
        q.push_back({ (s & 1) ? -0.5f : 0.5f, (s & 2) ? -0.5f : 0.5f,
                      (s & 4) ? -0.5f : 0.5f, (s & 8) ? -0.5f : 0.5f });
    return q;
}
// choose a SIGNED-PERMUTATION aQ so the capsule axis rotateInv(mQ, basis0(aQ))
// is +-e_x (wantZ=false) or +-e_z (wantZ=true).  For x, aQ = mQ works exactly
// (basis0(mQ) = mQ.rotate(e_x), rotateInv(mQ, .) = e_x, and mtcQ collapses to
// identity).  For z, pick the candidate whose local axis is most z-aligned;
// signed-permutation rotations map axes to axes exactly, so it is +-e_z.
static F4 findAxisQuat(F4 mQ, bool wantZ)
{
    if (!wantZ) return mQ;
    static const std::vector<F4> cand = exactQuatSet();
    F4 best = mQ;
    float bestZ = -1.0f;
    for (const F4& q : cand)
    {
        const V3 d = quatRotateInv(mQ, basis0(q));
        if (std::fabs(d.z) > bestZ) { bestZ = std::fabs(d.z); best = q; }
    }
    return best;
}

// Shape-space authoring: the canonical triangle V0=(0,0,0), V1=(2,0,0),
// V2=(0,2,0) with normal N=(0,0,1) is placed in the mesh's LOCAL frame;
// mesh verts are stored via shape2Vertex (with the per-flip winding
// arrangement) so getShapeSpaceVerts recovers the SAME triangle for
// identity, positive-scale AND negative-determinant (flip) cases — the
// flip cases then reproduce identical results, exercising the flip path
// without changing the geometry.  Every family's patch normal is either
// N or an EXACT in-plane unit vector, so the NI normal is bit-exact.
// i%4==1 -> flip scale; i%8==7 -> positive non-identity scale.  Capsule
// batch 2 (cases 64..95) is the all-deferred-zero-contact DROP batch.
static CmFixture buildFixture(uint32_t NC, uint32_t seed, bool capsule)
{
    CmFixture fx;
    std::mt19937 rng(seed);
    fx.cases.resize(NC);
    const uint32_t NCM = NC / 4;
    fx.pairRec.resize(NCM);
    for (uint32_t cm = 0; cm < NCM; ++cm)
    {
        const uint32_t px = ((cm * 7 + 3) % NCM) * 4;
        const uint32_t py = ((cm * 11 + 5) % NCM) * 4;
        fx.pairRec[cm] = { px, py, 0, 0 };
    }
    fx.poolN = NC + 16;
    const V3 V0 = { 0, 0, 0 }, V1 = { 2, 0, 0 }, V2 = { 0, 2, 0 };
    const float cd = 0.25f, radius = 0.5f, hh = 0.5f;
    for (uint32_t i = 0; i < NC; ++i)
    {
        CmCase& cs = fx.cases[i];
        std::memset(&cs, 0, sizeof(cs));
        cs.mQ = exactQuat(rng);
        cs.mP = { grid64(rng, -2, 2), grid64(rng, -2, 2), grid64(rng, -2, 2), 0 };
        cs.scale = { 1, 1, 1, radius };
        cs.scaleRot = { 0, 0, 0, 1 };
        const bool flip = (i % 4 == 1);
        if (flip)
        {
            cs.scale = { -1, 1, 1, radius }; // negative determinant: flip path
            cs.scaleRot = exactQuat(rng);
        }
        else if (i % 8 == 7)
        {
            cs.scale = { 2, 1, 0.5f, radius }; // positive non-identity scale (pow2)
            cs.scaleRot = exactQuat(rng);
        }
        if (capsule) cs.mP.w = hh; // half height in mP.w (scaleRot stays a unit quat)

        // store mesh verts so getShapeSpaceVerts recovers (V0,V1,V2):
        //   vert0 = shape2Vertex(V0), and (vert1,vert2) = shape2Vertex of
        //   (V2,V1) when flipping, else (V1,V2)
        const uint32_t vertBase = (uint32_t)fx.mesh.verts.size();
        const uint32_t triBase = (uint32_t)fx.mesh.tris.size();
        const uint32_t tIdx = 1 + i % 2;
        fx.mesh.verts.push_back({ 99, 99, 99, 0 });
        for (uint32_t t = 0; t < 3; ++t)
        {
            const uint32_t vb = (uint32_t)fx.mesh.verts.size() - vertBase;
            V3 s0, s1, s2;
            if (t == tIdx)
            {
                s0 = shape2VertexRef(V0, v3(cs.scale), cs.scaleRot);
                s1 = shape2VertexRef(flip ? V2 : V1, v3(cs.scale), cs.scaleRot);
                s2 = shape2VertexRef(flip ? V1 : V2, v3(cs.scale), cs.scaleRot);
            }
            else
            {
                s0 = { float(int(rng() % 9) + 8), float(int(rng() % 9) + 8), float(int(rng() % 9) + 8) };
                s1 = add(s0, V3{ 1, 0, 0 });
                s2 = add(s0, V3{ 0, 1, 0 });
            }
            fx.mesh.verts.push_back({ s0.x, s0.y, s0.z, 0 });
            fx.mesh.verts.push_back({ s1.x, s1.y, s1.z, 0 });
            fx.mesh.verts.push_back({ s2.x, s2.y, s2.z, 0 });
            fx.mesh.tris.push_back({ vb, vb + 1, vb + 2, 0 });
            U4 adj = { BOUNDARY, BOUNDARY, BOUNDARY, 0 };
            if (t == tIdx && !capsule && i % 8 == 5)
            {
                // fam5 is a flip family (i%4==1) in an eE01 edge region.
                //  i%16==5:  edge e01 nonconvex on x AND z -> blocked either
                //            way (correct nb==0), pins the nonconvex block;
                //  i%16==13: nonconvex only on adj.x, convex on adj.z -> with
                //            the winding flip the swap moves the CONVEX edge
                //            onto e01 so the contact IS generated (nb==1); a
                //            mutant that drops the adj.x<->adj.z swap reads the
                //            nonconvex edge and blocks it (nb==0) -> caught
                if (i % 16 == 5)
                    adj = { NONCONVEX | 7u, BOUNDARY, NONCONVEX | 9u, 0 };
                else
                    adj = { NONCONVEX | 7u, BOUNDARY, BOUNDARY, 0 };
            }
            fx.mesh.adj.push_back(adj);
            fx.mesh.remap.push_back(0x300u + (uint32_t)fx.mesh.remap.size() * 3u + 1u);
        }
        cs.mesh = { vertBase, triBase, 0, 0 };

        const uint32_t cm = i / 4;
        cs.pair = { cm, tIdx, i % 4, (1u << 28) | ((i * 11 + 3) % 4096) };

        // C = the sphere / capsule-segment center in shape (mesh-local) space;
        // wantZ picks a perpendicular capsule axis (segment pierces / stands
        // beside) vs an in-plane axis (parallel to the triangle)
        // dropBatch (batch 2): ALL lanes deferred with zero contacts -> the
        // whole warp is contact-less and dropped.  mixedBatch (batch 3): odd
        // lanes are deferred-zero-contact, even lanes are face contacts, so
        // the warp is LIVE (totalContacts>0) yet contains zero-contact
        // deferred lanes whose outMask must still be emitted — pins that the
        // second-pass write gates on outMask alone, not on nbContacts.
        const bool dropBatch = capsule && (i / 32 == 2);
        const bool mixedBatch = capsule && (i / 32 == 3);
        V3 C;
        bool wantZ = false;
        if (dropBatch || (mixedBatch && (i % 2 == 1)))
        {
            C = { 0.5f, -0.5f, 0 }; wantZ = false; // coplanar beside e01 -> 0 contacts
        }
        else if (mixedBatch)
        {
            C = { 0.5f, 0.4f, 0.25f }; wantZ = false; // face contact (nb==2, om==0)
        }
        else
        {
            switch (i % 8)
            {
            case 0: C = { 0.5f, 0.4f, 0.25f }; wantZ = false; break; // face
            case 1: C = capsule ? V3{ 0.6f, 0.6f, 0.25f } : V3{ -0.3f, -0.4f, 0 };
                    wantZ = false; break;                            // face-2 / vertex defer
            case 2: C = { 1.0f, -0.5f, 0 }; wantZ = true; break;     // edge e01 deferral
            case 3: C = { 0.5f, 0.5f, capsule ? -0.125f : -1.0f }; wantZ = false; break; // below (capsule: CLOSE below -> only the gate rejects, pins it)
            case 4:
                // capsule accept-gate boundary: an in-plane segment at height
                // 0.75 gives sqDist == (radius+cd)^2 == 0.5625 EXACTLY, so the
                // STRICT sqInflatedRadius > sqDist excludes it (nb==0,
                // patchNormal stays 0); a >= mutant enters the block and
                // writes a non-zero patchNormal -> ni.normal mismatch
                C = (capsule && i % 16 == 12) ? V3{ 0.5f, 0.4f, 0.75f } : V3{ 0.5f, 0.5f, 4.0f };
                wantZ = false; break;                                // too far / accept boundary
            case 5: C = capsule ? V3{ 0.5f, 0.5f, 0 } : V3{ 1.0f, -0.375f, 0 };
                    wantZ = true; break;                             // pierce / eE01 blocked+swap
            case 6:
                if (capsule)
                {
                    // vertex (V0) deferral: a perpendicular segment beside the
                    // V0 corner; closest = V0, patch normal (-0.6,-0.8,0)
                    // classifies as eV0 via getTriangleActiveMaskFromNormal
                    C = { -0.3f, -0.4f, 0 }; wantZ = true;
                }
                else
                {
                    // sphere boundary probe: C on the -y axis, closest to V0
                    // along edge e01 (v==0), |C - V0| == radius+cd (0.75)
                    // EXACTLY (0.75^2 is a single multiply, fma-immune) so the
                    // strict > excludes it; 47/64 (i%16==14) is included
                    C = { 0, (i % 16 == 6) ? -0.75f : -0.734375f, 0 };
                    wantZ = false;
                }
                break;
            default: C = { 0.6f, 0.5f, 0.375f }; wantZ = false; break; // face2
            }
        }

        // capsule axis: force segment along +-x (wantZ=false) or +-z
        cs.aQ = exactQuat(rng);
        if (capsule)
            cs.aQ = findAxisQuat(cs.mQ, wantZ);
        // aP so that rotateInv(mQ, aP - mP) == C exactly
        const V3 world = add(quatRotate(cs.mQ, C), v3(cs.mP));
        cs.aP = { world.x, world.y, world.z, cd };
    }
    fx.ref.resize(NC);
    for (uint32_t i = 0; i < NC; ++i)
        fx.ref[i] = capsule ? refCapsule(fx.cases[i], fx.mesh, fx.pairRec)
                            : refSphere(fx.cases[i], fx.mesh, fx.pairRec);
    return fx;
}

static void checkKind(const char* entry, bool capsule, uint32_t seed)
{
    const uint32_t NC = 1024;
    CmFixture fx = buildFixture(NC, seed, capsule);

    // batch aggregation (32 cases per batch) with the drop quirk
    const uint32_t NB = NC / 32;
    std::vector<uint32_t> batchContacts(NB, 0), batchMasks(NB, 0);
    uint32_t totalSP = 0, totalTC = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        batchContacts[i / 32] += fx.ref[i].nb;
        if (fx.ref[i].om != 0) batchMasks[i / 32]++;
    }
    size_t droppedMasks = 0;
    for (uint32_t b = 0; b < NB; ++b)
    {
        if (batchContacts[b] == 0) { droppedMasks += batchMasks[b]; continue; }
        totalSP += batchMasks[b];
        totalTC += batchContacts[b];
    }

    size_t nHit = 0, nDelay = 0, nEdgeD = 0, nVertD = 0, nZero = 0, nFlip = 0, nOmZeroNb = 0,
           nBoundaryEx = 0, nBoundaryIn = 0, nTwoCts = 0, nFaceRegion = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const CmRef& ro = fx.ref[i];
        if (ro.nb) ++nHit; else ++nZero;
        if (ro.delay) ++nDelay;
        if (ro.delay && ro.edge) ++nEdgeD;
        if (ro.delay && !ro.edge && ro.om) ++nVertD;
        if (i % 4 == 1 && ro.nb) ++nFlip;
        if (ro.om && ro.nb == 0) ++nOmZeroNb;
        if (i % 16 == 6 && ro.nb == 0) ++nBoundaryEx;
        if (i % 16 == 14 && ro.nb != 0) ++nBoundaryIn;
        if (ro.nb == 2) ++nTwoCts;
        if (ro.faceRegion) ++nFaceRegion;                             // P9.36
    }
    if (capsule)
        PXG_CHECK_MSG(nOmZeroNb > 20 && droppedMasks > 10 && nTwoCts > 100,
                      "%s coverage: omZeroNb=%zu dropped=%zu two=%zu", entry, nOmZeroNb, droppedMasks, nTwoCts);
    else
        PXG_CHECK_MSG(nBoundaryEx == 64 && nBoundaryIn == 64,
                      "%s boundary probes: ex=%zu in=%zu (want 64/64)", entry, nBoundaryEx, nBoundaryIn);
    // P9.36: the face region must actually be entered, or the barycentric tail of
    // distPointTriRef is dead weight and the fixture proves nothing about it.
    // A NAMED LIMITATION, measured rather than assumed -- AND THE REASON P9.36'S
    // ASSOCIATION CHANGE SURVIVED HERE FOR TWO TRANCHES.
    //
    // The face region of distPointTriRef is entered often on the sphere path (256 of
    // 1024) and NEVER on the capsule path.  Yet swapping the barycentric tail of
    // npCvxMeshNp.slang between the CUDA's `(ab*vb + ac*vc) * denom`
    // (sphereTriangle.cuh:138) and the CPU header's `ab*(vb*denom) + ac*(vc*denom)`
    // (GuDistancePointTriangle.h:93-97) leaves this fixture green EITHER WAY, and the
    // counters say why: in ZERO of those 256 cases does any component have BOTH ab and
    // ac non-zero.  This mesh's triangles are axis-aligned, so every component of
    // `ab.c*vb + ac.c*vc` has at most one live term, and the two associations are then
    // IDENTICAL as expressions -- not close, identical.  No tolerance and no sample
    // count could change that; it follows from the geometry.
    //
    // This is a fourth way a fixture goes blind, distinct from a wrong mirror, a loose
    // tolerance and an uncovered path: the exact-arithmetic doctrine that makes
    // bit-exact comparison possible here is the same thing that puts rounding-order
    // defects out of reach.  Separating them needs a triangle whose two edge vectors
    // share a non-zero component, which means reshaping the shared mesh that every
    // other expectation in this file is pinned against.
    //
    // WHAT ACTUALLY POLICES IT is the sibling: gpuslang.npCvxHfNp ports the SAME
    // function (sphereTriangle.cuh:41-140) over heightfield triangles, whose two edge
    // vectors DO share a live y component, and it separates the two spellings on 123
    // cases -- measured three times now, twice in P9.36 and once in P9.53.
    // gpuslang.vectorDivide covers the divide half.
    PXG_CHECK_MSG(!capsule || nFaceRegion == 0,
                  "%s: the capsule path now reaches the face region (%zu cases) -- the "
                  "limitation note above is stale, re-measure it", entry, nFaceRegion);
    PXG_CHECK_MSG(capsule || nFaceRegion > 0,
                  "%s: the sphere path no longer reaches the face region", entry);
    std::printf("    [ note ] %s: face region reached in %zu of %zu cases; %zu separate the "
                "CUDA's association from the CPU header's (%zu have a component where both "
                "edge vectors are non-zero)\n",
                entry, nFaceRegion, fx.cases.size(), gFaceDistinguish, gFaceMixedComponent);
    PXG_CHECK_MSG(nHit > 300 && nDelay > 100 && nEdgeD > 30 && nVertD > 30 && nZero > 100 && nFlip > 50,
                  "%s coverage thin (hit=%zu delay=%zu edgeD=%zu vertD=%zu zero=%zu flip=%zu)",
                  entry, nHit, nDelay, nEdgeD, nVertD, nZero, nFlip);

    const uint32_t spCap = totalSP + 32, tcCap = totalTC + 32;
    const CmRun r = runCm(entry, fx.cases, fx.mesh, fx.pairRec, fx.poolN, spCap, tcCap);

    PXG_CHECK_MSG(r.counters[0] == totalSP && r.counters[1] == totalTC,
                  "%s counters %u/%u (want %u/%u)", entry, r.counters[0], r.counters[1], totalSP, totalTC);

    // per-pair outputs at DECOUPLED offsets; everything else stays poisoned
    std::vector<bool> usedPo(fx.poolN, false), usedPad(fx.poolN, false);
    size_t mm = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const CmRef& ro = fx.ref[i];
        usedPo[ro.pairOffset] = true;
        usedPad[ro.padOffset] = true;
        const F4 ni = r.ni[ro.pairOffset];
        if (ni.x != ro.niNormal.x || ni.y != ro.niNormal.y || ni.z != ro.niNormal.z ||
            asU32(ni.w) != ro.niIndex) ++mm;
        if (r.depth[ro.pairOffset] != ro.maxDepth) ++mm;
        if (r.interm[ro.pairOffset] != fx.cases[i].pair.y) ++mm;
        if (r.ordered[ro.padOffset] != ro.ordered) ++mm;
        if (r.dbg[i].x != float(ro.nb) || r.dbg[i].y != ro.maxDepth ||
            r.dbg[i].z != (ro.delay ? 1.0f : 0.0f) || r.dbg[i].w != (ro.edge ? 1.0f : 0.0f)) ++mm;
        // temp contacts + linkage
        if (ro.nb)
        {
            const uint32_t idx = r.ctsIdx[ro.pairOffset];
            if (idx == POISON || idx + ro.nb > totalTC) { ++mm; continue; }
            for (uint32_t k = 0; k < ro.nb; ++k)
            {
                const F4 got = r.temp[idx + k];
                if (got.x != ro.cts[k].x || got.y != ro.cts[k].y || got.z != ro.cts[k].z ||
                    got.w != ro.seps[k]) ++mm;
            }
        }
        else if (r.ctsIdx[ro.pairOffset] != POISON) ++mm;
    }
    for (uint32_t s = 0; s < fx.poolN; ++s)
    {
        if (!usedPo[s] && (asU32(r.ni[s].x) != asU32(-7.0f) || r.depth[s] != -7.0f ||
                           r.interm[s] != POISON || r.ctsIdx[s] != POISON)) ++mm;
        if (!usedPad[s] && r.ordered[s] != POISON) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "%s: %zu mismatches", entry, mm);

    // temp-contact ranges partition [0, totalTC)
    {
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (uint32_t i = 0; i < NC; ++i)
            if (fx.ref[i].nb)
                ranges.push_back({ r.ctsIdx[fx.ref[i].pairOffset], fx.ref[i].nb });
        std::sort(ranges.begin(), ranges.end());
        uint32_t cursor = 0;
        size_t bad = 0;
        for (const auto& pr : ranges)
        {
            if (pr.first != cursor) ++bad;
            cursor = pr.first + pr.second;
        }
        if (cursor != totalTC) ++bad;
        PXG_CHECK_MSG(bad == 0, "%s: temp ranges do not partition [0, %u)", entry, totalTC);
    }

    // second-pass entries: multiset per batch, per-batch contiguity in lane
    // order, batch blocks partition [0, totalSP); dropped batches absent
    {
        std::vector<std::vector<uint32_t>> expByBatch(NB);
        for (uint32_t i = 0; i < NC; ++i)
            if (fx.ref[i].om != 0 && batchContacts[i / 32] != 0)
                expByBatch[i / 32].push_back(fx.ref[i].om);
        // find each batch's entries in the stream by value (gti tags are unique)
        std::vector<int> ownerOf(totalSP, -1);
        size_t bad = 0;
        std::vector<uint32_t> stream(r.second.begin(), r.second.begin() + totalSP);
        for (uint32_t b = 0; b < NB; ++b)
        {
            if (expByBatch[b].empty()) continue;
            // locate the first element
            auto it = std::find(stream.begin(), stream.end(), expByBatch[b][0]);
            if (it == stream.end()) { ++bad; continue; }
            const size_t pos = size_t(it - stream.begin());
            for (size_t j = 0; j < expByBatch[b].size(); ++j)
            {
                if (pos + j >= stream.size() || stream[pos + j] != expByBatch[b][j]) { ++bad; break; }
                ownerOf[pos + j] = int(b);
            }
        }
        for (uint32_t s = 0; s < totalSP; ++s)
            if (ownerOf[s] < 0) ++bad;
        for (uint32_t s = totalSP; s < spCap; ++s)
            if (r.second[s] != POISON) ++bad;
        PXG_CHECK_MSG(bad == 0, "%s: %zu second-pass stream errors", entry, bad);
    }
    for (uint32_t s = totalTC; s < tcCap; ++s)
        PXG_CHECK_MSG(r.temp[s].x == -777.0f, "%s: dirty temp slot %u", entry, s);
}

PXG_TEST(npCvxMeshNp_sphere)
{
    checkKind("cmSphereTri", false, 1060701);
}

PXG_TEST(npCvxMeshNp_capsule)
{
    checkKind("cmCapsuleTri", true, 1060702);
}

} // namespace
