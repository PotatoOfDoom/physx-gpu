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

// pxgCapsuleTriRef.h -- the capsule/triangle geometric core and its two contact
// generators, transcribed from the ENGINE.
//
// Sources: distanceSegmentSegmentSquared (distanceSegmentSegment.cuh:33-88),
// distanceSegmentTriangleSquared (capsuleTriangle.cuh:117-327), generateContacts
// (:329-406), generateEE (:408-461) and generateEEContacts (:463-474).
//
// WHY IT IS A HEADER (P9.52b-5).  testSegTriangleBda, testCapTriangleBda and
// testCapsuleHfCore each carried a copy.  Diffed expression by expression before
// the move; no arithmetic and no control-flow divergence, so the bodies are the
// surviving copies moved VERBATIM.
//
// TWO THINGS THIS FILE MUST KEEP SAYING OUT LOUD, because both have already cost a
// tranche:
//
//   * `generateContacts` compares an UNSQUARED radius against a SIGNED penetration,
//     `generateEE` a SQUARED radius against a squared distance.  Handing either the
//     other's radius compiles, runs, and is wrong by one squaring.
//   * `generateContacts` emits the SEGMENT ENDPOINT, not the projected point.  The
//     projection exists only to compute the barycentrics that decide validity.
//
// The trace out-parameters are how the probe fixtures assert which internal arm ran;
// the trace-free overloads at the bottom are for callers that only want the answer.

#ifndef PXG_CAPSULE_TRI_REF_H
#define PXG_CAPSULE_TRI_REF_H

#include "pxgNpRefMath.h"

#include <cstdint>

namespace pxgtest
{
namespace captriref
{
using npref::V3;
using npref::sub;
using npref::add;
using npref::mul;
using npref::neg;
using npref::dot3;
using npref::cross3;
using npref::normalized;
using npref::pxClamp;
using npref::validBarycentric;
using npref::MAX_F32;

// distanceSegmentSegmentSquared (distanceSegmentSegment.cuh:33-88).
// `parallelArm` and the two reciprocal guards are reported so the fixture can
// require them.
struct SegSegTrace { bool parallelArm; bool aDegenerate; bool eDegenerate;
                     bool sClampedLo, sClampedHi, tClampedLo, tClampedHi;
                     // P9.86: the s0 clamp kept a -0 (selectMax, not the generic
                     // a < b ? b : a) AND that sign reaches t: recomputing tTmp with
                     // the flattened +0 changes tTmp's zero sign.
                     bool s0SignReachesT; };

inline void segSegSquared(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t, SegSegTrace& tr)
{
    const float eps = 1e-6f;
    const V3 r = sub(p1, p2);
    const float a = dot3(d1, d1);
    const float e = dot3(d2, d2);
    const float b = dot3(d1, d2);
    const float c = dot3(d1, r);
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
    const float s2 = pxClamp(comp, 0.0f, 1.0f);

    tr.parallelArm = eps > denom;
    tr.aDegenerate = !(a > eps);
    tr.eDegenerate = !(e > eps);
    tr.tClampedLo = tTmp < 0.0f;
    tr.tClampedHi = tTmp > 1.0f;
    tr.sClampedLo = comp < 0.0f;
    tr.sClampedHi = comp > 1.0f;
    tr.s0SignReachesT = !(eps > denom) && s0 == 0.0f && std::signbit(s0)
                        && std::signbit(tTmp) != std::signbit((b * 0.0f + f) * eRecip);

    s = s2;
    t = t2;
}

// distanceSegmentTriangleSquared (capsuleTriangle.cuh:117-327).
enum StWay { WAY_PLANE = 0, WAY_BOTH_INSIDE, WAY_CON0, WAY_CON1, WAY_EDGE_ONLY };
enum StEdge { EDGE_NONE = 0, EDGE_AB, EDGE_BC, EDGE_AC };

struct StTrace
{
    StWay way;
    StEdge edge;
    bool endpointWon;     // in WAY_CON0 / WAY_CON1: the endpoint beat the edge
    bool tookQ;           // in WAY_BOTH_INSIDE: t == 1 rather than 0
    bool degenerateTri;   // bdenom fell back to PX_MAX_F32
    bool dMulZero;        // an endpoint sits exactly on the plane
};

inline float segTriSquared(V3 p, V3 q, V3 a, V3 b, V3 c,
                           float& t, float& u, float& v, StTrace& tr)
{
    const V3 pq = sub(q, p);
    const V3 ab = sub(b, a);
    const V3 ac = sub(c, a);
    const V3 bc = sub(c, b);
    const V3 ap = sub(p, a);
    const V3 aq = sub(q, a);

    const float d00 = dot3(ab, ab);
    const float d01 = dot3(ab, ac);
    const float d11 = dot3(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;

    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : MAX_F32;
    tr.degenerateTri = !(tDenom > 0.0f);

    const V3 n = normalized(cross3(ab, ac));

    const float dist3 = dot3(ap, n);
    const float sqDist3 = dist3 * dist3;
    const float dist4 = dot3(aq, n);
    const float sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;
    tr.dMulZero = (dMul == 0.0f);

    tr.edge = EDGE_NONE;
    tr.endpointWon = false;
    tr.tookQ = false;

    if (dMul <= 0.0f)
    {
        const float nom = -dot3(n, ap);
        const float denom = 1.0f / dot3(n, pq);
        const float t0 = nom * denom;
        const V3 ip = add(p, mul(pq, t0));
        const V3 v2 = sub(ip, a);
        const float d20 = dot3(v2, ab);
        const float d21 = dot3(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom;
        const float w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (validBarycentric(v00, w00))
        {
            t = t0; u = v00; v = w00;
            tr.way = WAY_PLANE;
            return 0.0f;
        }
    }

    const V3 closestP31 = sub(p, mul(n, dist3));
    const V3 pV20 = sub(closestP31, a);
    const float pD20 = dot3(pV20, ab);
    const float pD21 = dot3(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = validBarycentric(v0, w0);

    const V3 closestP41 = sub(q, mul(n, dist4));
    const V3 qV20 = sub(closestP41, a);
    const float qD20 = dot3(qV20, ab);
    const float qD21 = dot3(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = validBarycentric(v1, w1);

    if (con0 && con1)
    {
        tr.way = WAY_BOTH_INSIDE;
        if (sqDist4 > sqDist3) { t = 0.0f; u = v0; v = w0; return sqDist3; }
        else                   { t = 1.0f; u = v1; v = w1; tr.tookQ = true; return sqDist4; }
    }

    float t00, t01, t02, t10, t11, t12;
    SegSegTrace ig;
    segSegSquared(p, pq, a, ab, t00, t10, ig);
    segSegSquared(p, pq, b, bc, t01, t11, ig);
    segSegSquared(p, pq, a, ac, t02, t12, ig);

    const V3 closestP00 = add(p, mul(pq, t00));
    const V3 closestP01 = add(a, mul(ab, t10));
    V3 tempV = sub(closestP00, closestP01);
    const float sqDist0 = dot3(tempV, tempV);

    const V3 closestP10 = add(p, mul(pq, t01));
    const V3 closestP11 = add(b, mul(bc, t11));
    tempV = sub(closestP10, closestP11);
    const float sqDist1 = dot3(tempV, tempV);

    const V3 closestP20 = add(p, mul(pq, t02));
    const V3 closestP21 = add(a, mul(ac, t12));
    tempV = sub(closestP20, closestP21);
    const float sqDist2 = dot3(tempV, tempV);

    const float u10 = t10, v10 = 0.0f;
    const float u11 = 1.0f - t11, v11 = t11;
    const float u12 = 0.0f, v12 = t12;

    float sqDistPE, uEdge, vEdge, tSeg;
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0)
    { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; tr.edge = EDGE_AB; }
    else if (sqDist2 > sqDist1)
    { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; tr.edge = EDGE_BC; }
    else
    { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; tr.edge = EDGE_AC; }

    if (con0)
    {
        tr.way = WAY_CON0;
        if (sqDistPE > sqDist3)
        { t = 0.0f; u = v0; v = w0; tr.endpointWon = true; return sqDist3; }
        t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
    }
    if (con1)
    {
        tr.way = WAY_CON1;
        if (sqDistPE > sqDist4)
        { t = 1.0f; u = v1; v = w1; tr.endpointWon = true; return sqDist4; }
        t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
    }
    tr.way = WAY_EDGE_ONLY;
    t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
}

// ---------------------------------------------------------------------------
// generateContacts / generateEE / generateEEContacts (capsuleTriangle.cuh:329-474)
// ---------------------------------------------------------------------------
struct Contacts
{
    V3 pt[2];
    float sep[2];
    uint32_t n;
};

inline Contacts contactsInit()
{
    Contacts o{};
    o.pt[0] = V3{ 0, 0, 0 }; o.pt[1] = V3{ 0, 0, 0 };
    o.sep[0] = 0.0f; o.sep[1] = 0.0f;
    o.n = 0;
    return o;
}

// ---------------------------------------------------------------------------
// generateContacts (capsuleTriangle.cuh:329-406)
// ---------------------------------------------------------------------------
struct FaceTrace { bool ideomGuard; bool con0, con1, took0, took1; };

inline Contacts genContacts(V3 a, V3 b, V3 c, V3 planeNormal, V3 normal,
                            V3 p, V3 q, float inflatedRadius, FaceTrace& tr)
{
    Contacts o = contactsInit();

    const V3 ab = sub(b, a);
    const V3 ac = sub(c, a);
    const V3 ap = sub(p, a);
    const V3 aq = sub(q, a);

    const float d00 = dot3(ab, ab);
    const float d01 = dot3(ab, ac);
    const float d11 = dot3(ac, ac);
    const float tdenom = (d00 * d11 - d01 * d01);
    const float bdenom = (tdenom > 0.0f) ? (1.0f / tdenom) : MAX_F32;

    const float inomp = dot3(planeNormal, neg(ap));
    const float ideom = dot3(planeNormal, normal);
    tr.ideomGuard = !(ideom > 0.0f);

    const float ipt = ideom > 0.0f ? -(inomp / ideom) : MAX_F32;

    const V3 closestP31 = sub(p, mul(normal, ipt));
    const V3 pV20 = sub(closestP31, a);
    const float pD20 = dot3(pV20, ab);
    const float pD21 = dot3(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = validBarycentric(v0, w0);
    tr.con0 = con0;

    tr.took0 = false; tr.took1 = false;
    if (con0 && (inflatedRadius > ipt))
    {
        o.pt[o.n] = p; o.sep[o.n] = ipt; o.n++;
        tr.took0 = true;
    }

    const float inomq = dot3(planeNormal, neg(aq));
    const float iqt = ideom > 0.0f ? -(inomq / ideom) : MAX_F32;

    const V3 closestP41 = sub(q, mul(normal, iqt));
    const V3 qV20 = sub(closestP41, a);
    const float qD20 = dot3(qV20, ab);
    const float qD21 = dot3(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = validBarycentric(v1, w1);
    tr.con1 = con1;

    if (con1 && (inflatedRadius > iqt))
    {
        o.pt[o.n] = q; o.sep[o.n] = iqt; o.n++;
        tr.took1 = true;
    }

    return o;
}

// ---------------------------------------------------------------------------
// generateEE (capsuleTriangle.cuh:408-461)
// ---------------------------------------------------------------------------
enum EeExit { EE_SAME_SIDE = 0, EE_PARALLEL, EE_T_HIGH, EE_T_LOW, EE_TOO_FAR, EE_EMIT };

struct EeTrace { EeExit exits[3]; uint32_t nRun; float npq[3], denom[3]; };

inline Contacts genEE(V3 p, V3 q, float sqInflatedRadius, V3 normal, V3 a, V3 b,
                      Contacts o, EeExit& exit, float& npqOut, float& denomOut)
{
    const V3 ab = sub(b, a);
    const V3 n = cross3(ab, normal);
    const float d = dot3(n, a);
    const float np = dot3(n, p);
    const float nq = dot3(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    const float temp = signP * signQ;
    npqOut = 0.0f; denomOut = 0.0f;
    if (temp > 0.0f) { exit = EE_SAME_SIDE; return o; }

    const V3 pq = sub(q, p);
    const float npq = dot3(n, pq);
    npqOut = npq;
    if (npq == 0.0f) { exit = EE_PARALLEL; return o; }

    const float segTValue = (d - np) / npq;
    const V3 localPointA = add(p, mul(pq, segTValue));

    const V3 perNormal = cross3(normal, pq);
    const V3 ap = sub(localPointA, a);
    const float nom = dot3(perNormal, ap);
    const float denom = dot3(perNormal, ab);
    denomOut = denom;

    const float tValue = nom / denom;
    if (tValue > 1.0f) { exit = EE_T_HIGH; return o; }
    if (tValue < 0.0f) { exit = EE_T_LOW; return o; }

    const V3 v = sub(ap, mul(ab, tValue));
    const float sqDist = dot3(v, v);

    if (sqInflatedRadius > sqDist)
    {
        const float signedDist = dot3(v, normal);
        o.pt[o.n] = localPointA; o.sep[o.n] = signedDist; o.n++;
        exit = EE_EMIT;
        return o;
    }
    exit = EE_TOO_FAR;
    return o;
}

inline Contacts genEEContacts(V3 a, V3 b, V3 c, V3 normal, V3 p, V3 q,
                              float sqInflatedRadius, Contacts o, EeTrace& tr)
{
    tr.nRun = 0;
    for (uint32_t k = 0; k < 3; ++k) { tr.exits[k] = EE_SAME_SIDE; tr.npq[k] = 0; tr.denom[k] = 0; }

    if (o.n < 2) { o = genEE(p, q, sqInflatedRadius, normal, a, b, o,
                             tr.exits[0], tr.npq[0], tr.denom[0]); tr.nRun++; }
    if (o.n < 2) { o = genEE(p, q, sqInflatedRadius, normal, b, c, o,
                             tr.exits[1], tr.npq[1], tr.denom[1]); tr.nRun++; }
    if (o.n < 2) { o = genEE(p, q, sqInflatedRadius, normal, a, c, o,
                             tr.exits[2], tr.npq[2], tr.denom[2]); tr.nRun++; }
    return o;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trace-free overloads.
// ---------------------------------------------------------------------------
inline void segSegSquared(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t)
{
    SegSegTrace sink;
    segSegSquared(p1, d1, p2, d2, s, t, sink);
}

inline float segTriSquared(V3 p, V3 q, V3 a, V3 b, V3 c, float& t, float& u, float& v)
{
    StTrace sink;
    return segTriSquared(p, q, a, b, c, t, u, v, sink);
}

inline Contacts genContacts(V3 a, V3 b, V3 c, V3 planeNormal, V3 normal,
                            V3 p, V3 q, float inflatedRadius)
{
    FaceTrace sink;
    return genContacts(a, b, c, planeNormal, normal, p, q, inflatedRadius, sink);
}

inline Contacts genEEContacts(V3 a, V3 b, V3 c, V3 normal, V3 p, V3 q,
                              float sqInflatedRadius, Contacts o)
{
    EeTrace sink;
    return genEEContacts(a, b, c, normal, p, q, sqInflatedRadius, o, sink);
}

}  // namespace captriref
}  // namespace pxgtest

#endif  // PXG_CAPSULE_TRI_REF_H
