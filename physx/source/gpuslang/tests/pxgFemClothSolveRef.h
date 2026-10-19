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

// pxgFemClothSolveRef.h -- the host oracle of the PhysX 5.12 cloth-vs-cloth CONTACT kernels
// (FEMClothConstraintPrep.cu cloth_clothContactPrepareLaunch, FEMClothExternalSolve.cu
// computeClothClothVTContact / computeClothClothEEContact / computeClothClothInitPen and the
// count / solve kernels), transcribed from the CUDA, NOT from the Slang port.  The closest-
// point leaves are deformableCollision.cuh's (closestPtPointTriangle, closestPtLineLine,
// closestPtPointEdge, closestPtEdgeEdge) and Gu::closestPtPointTriangle2.
//
// It builds on pxgSoftBodySolveRef.h (HostCloth, DbPart, solveDbDbContact, writeCloth /
// writeClothEdge) and pxgParticleSolveRef.h (V3 spellings).  PxVec3 dot is (x + y) + z,
// PxMin / PxMax / PxClamp on floats are the selects, a vector divided by a scalar is one
// reciprocal and three multiplies.  The test target builds with -ffp-contract=off.

#ifndef PXG_FEMCLOTH_SOLVE_REF_H
#define PXG_FEMCLOTH_SOLVE_REF_H

#include "pxgSoftBodySolveRef.h"

#include <cmath>
#include <cstdint>

namespace pxgfcref
{
using namespace pxgpsref;
using namespace pxgsbref;

static const uint32_t REF_eTRIANGLE_EDGE = 3u;
inline uint32_t refLocalEdge(uint64_t r) { return (uint32_t(r) >> 27) & 3u; }
inline uint64_t encodeEdgeRef(uint32_t clothId, uint32_t tri, uint32_t localEdge)
{
    return (uint64_t(clothId) << 34) | (uint64_t(REF_eTRIANGLE_EDGE) << 29) | (uint64_t(localEdge) << 27) | uint64_t(tri);
}

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;
static const float FEMCLOTH_NORMAL_FALLBACK_DISTSQ = 1.0e-10f;
static const float BARY_THR = 1.0e-6f;
static const float ONE_MINUS_BARY_THR = 1.0f - 1.0e-6f;

inline V3 scaleInv(V3 a, float s) { const float r = 1.0f / s; return v3(a.x * r, a.y * r, a.z * r); }

// deformableCollision.cuh closestPtPointTriangle (s, t out; the point is not needed).
inline void closestPtPointTriangle(V3 p, V3 a, V3 b, V3 c, float& s, float& t)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return; }
    const V3 bp = sub(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { s = d1 / (d1 - d3); t = 0.0f; return; }
    const V3 cp = sub(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return; }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { s = 0.0f; t = d2 / (d2 - d6); return; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        s = 1.0f - w; t = w;
        return;
    }
    const float denom = 1.0f / (va + vb + vc);
    s = vb * denom;
    t = vc * denom;
}

// Gu::closestPtPointTriangle2 (GuDistancePointTriangle.h).
inline V3 closestPtPointTriangle2(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return add(a, mul(ab, d1 / (d1 - d3)));
    const V3 cp = sub(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return add(a, mul(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
        return add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    return add(add(a, mul(ab, v)), mul(ac, w));
}

// deformableCollision.cuh closestPtLineLine, as written: its s / t mix the two edge lengths
// (b1 * a01 carries 1 / r1Len, a01 * b0 1 / r0Len), exact closest points for equal lengths only.
inline bool closestPtLineLine(V3 e0v0, V3 e0v1, V3 e1v0, V3 e1v1, float& s, float& t, float& distSq)
{
    const V3 r0 = sub(e0v1, e0v0), r1 = sub(e1v1, e1v0), d = sub(e1v0, e0v0);
    const float r0Len2 = dot(r0, r0), r1Len2 = dot(r1, r1);
    if (r0Len2 < 1.0e-18f || r1Len2 < 1.0e-18f) { s = t = 0.0f; distSq = dot(d, d); return false; }
    const float r0Len = std::sqrt(r0Len2), r1Len = std::sqrt(r1Len2);
    const V3 u0 = scaleInv(r0, r0Len), u1 = scaleInv(r1, r1Len);
    const float cosTheta = std::fabs(dot(u0, u1));
    if (cosTheta > 0.9999f)
    {
        const float tProj = dot(sub(e0v0, e1v0), r1) / r1Len2;
        t = tProj; s = 0.0f;
        const V3 pt1 = add(e1v0, mul(r1, tProj));
        distSq = dot(sub(pt1, e0v0), sub(pt1, e0v0));
        return false;
    }
    const float a01 = -dot(u0, u1);
    const float b0 = dot(u0, d) / r0Len, b1 = -dot(u1, d) / r1Len;
    const float det = 1.0f - a01 * a01;
    if (det < 1.0e-9f) { s = t = 0.0f; distSq = dot(d, d); return false; }
    const float invDet = 1.0f / det;
    s = (b0 - b1 * a01) * invDet;
    t = (b1 - a01 * b0) * invDet;
    const V3 pt0 = add(e0v0, mul(r0, s)), pt1 = add(e1v0, mul(r1, t));
    distSq = dot(sub(pt1, pt0), sub(pt1, pt0));
    return true;
}

// closestPtPointEdge: PxClamp(x, 0, 1) = PxMax(0, PxMin(1, x)).
inline float closestPtPointEdge(V3 v0, V3 ev0, V3 ev1)
{
    const V3 r = sub(ev1, ev0);
    const float d2 = dot(r, r);
    return (d2 > 0.0f) ? selectMax(0.0f, selectMin(1.0f, dot(r, sub(v0, ev0)) / d2)) : 0.5f;
}

// closestPtEdgeEdge.
inline bool closestPtEdgeEdge(V3 e0v0, V3 e0v1, V3 e1v0, V3 e1v1, float& s, float& t, float& distSq)
{
    if (!closestPtLineLine(e0v0, e0v1, e1v0, e1v1, s, t, distSq))
        return false;
    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f)
    {
        const float w[4] = { closestPtPointEdge(e0v0, e1v0, e1v1), closestPtPointEdge(e0v1, e1v0, e1v1),
                             closestPtPointEdge(e1v0, e0v0, e0v1), closestPtPointEdge(e1v1, e0v0, e0v1) };
        const float cand[4][2] = { { 0.0f, w[0] }, { 1.0f, w[1] }, { w[2], 0.0f }, { w[3], 1.0f } };
        int minIndex = 0;
        distSq = 3.40282347e+38f;
        for (int i = 0; i < 4; ++i)
        {
            const float sC = cand[i][0], tC = cand[i][1];
            const V3 p0 = add(mul(e0v0, 1.0f - sC), mul(e0v1, sC));
            const V3 p1 = add(mul(e1v0, 1.0f - tC), mul(e1v1, tC));
            const float d2 = dot(sub(p0, p1), sub(p0, p1));
            minIndex = (d2 < distSq) ? i : minIndex;
            distSq = selectMin(d2, distSq);
        }
        s = cand[minIndex][0];
        t = cand[minIndex][1];
    }
    return true;
}

// weights[0] * xx0 + weights[1] * xx1 + weights[2] * xx2 + weights[3] * xx3, left to right.
inline V3 weightedSum(F4 w, V3 a, V3 b, V3 c, V3 d)
{
    return add(add(add(mul(a, w.x), mul(b, w.y)), mul(c, w.z)), mul(d, w.w));
}

inline bool vtContact(const HostCloth& c0, const HostCloth& c1, F4 x0, F4 x1, F4 x2, F4 x3, F4& weights, V3& n, V3& disp)
{
    const V3 xx0 = xyz(x0), xx1 = xyz(x1), xx2 = xyz(x2), xx3 = xyz(x3);
    float s, t;
    closestPtPointTriangle(xx0, xx1, xx2, xx3, s, t);
    const float r = 1.0f - s - t;
    weights = F4{ 1.0f, -r, -s, -t };
    disp = weightedSum(weights, xx0, xx1, xx2, xx3);
    const float dispSq = dot(disp, disp);
    const float restDist = c0.restOffset + c1.restOffset;
    if (dispSq > restDist * restDist)
        return false;
    if (dispSq < FEMCLOTH_NORMAL_FALLBACK_DISTSQ)
    {
        n = cross(sub(xx2, xx1), sub(xx3, xx1));
        const float nLenSq = dot(n, n);
        if (nLenSq < FEMCLOTH_THRESHOLD)
            return false;
        n = mul(n, 1.0f / std::sqrt(nLenSq));
        if (dot(n, sub(xx0, xx1)) < 0.0f)
            n = neg(n);
    }
    else
        n = mul(disp, 1.0f / std::sqrt(dispSq));
    return true;
}

inline bool eeContact(const HostCloth& c0, const HostCloth& c1, F4 x0, F4 x1, F4 x2, F4 x3, F4& weights, V3& n, V3& disp)
{
    const V3 xx0 = xyz(x0), xx1 = xyz(x1), xx2 = xyz(x2), xx3 = xyz(x3);
    float s, t, distSq;
    const bool isValid = closestPtLineLine(xx0, xx1, xx2, xx3, s, t, distSq);
    const bool isOutside = (s < BARY_THR) || (s > ONE_MINUS_BARY_THR) || (t < BARY_THR) || (t > ONE_MINUS_BARY_THR);
    if (isOutside)
        return false;
    weights = F4{ 1.0f - s, s, -(1.0f - t), -t };
    disp = weightedSum(weights, xx0, xx1, xx2, xx3);
    const float dispSq = dot(disp, disp);
    const float restDist = c0.restOffset + c1.restOffset;
    if (dispSq > restDist * restDist)
        return false;
    if (distSq < FEMCLOTH_NORMAL_FALLBACK_DISTSQ)
    {
        if (!isValid)
            return false;
        n = cross(sub(xx1, xx0), sub(xx3, xx2));
        const float nLenSq = dot(n, n);
        if (nLenSq < FEMCLOTH_THRESHOLD)
            return false;
        n = mul(n, 1.0f / std::sqrt(nLenSq));
    }
    else
        n = mul(disp, 1.0f / std::sqrt(distSq));
    if (!isValid)
        weights = F4{ 0.5f, 0.5f, -0.5f, -0.5f };
    return true;
}

// float4 operators: w * (x - prevX) (scalar first), summed left to right.
inline F4 sub4(F4 a, F4 b) { return F4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
inline F4 smul4(float s, F4 a) { return F4{ s * a.x, s * a.y, s * a.z, s * a.w }; }

inline float initPen(const HostCloth& c0, const HostCloth& c1, F4 x0, F4 x1, F4 x2, F4 x3, F4 p0, F4 p1, F4 p2, F4 p3,
                     F4 w, V3 n, V3 disp)
{
    const F4 rd = add4(add4(add4(smul4(w.x, sub4(x0, p0)), smul4(w.y, sub4(x1, p1))), smul4(w.z, sub4(x2, p2))),
                       smul4(w.w, sub4(x3, p3)));
    return dot(sub(disp, xyz(rd)), n) - (c0.restOffset + c1.restOffset);
}

} // namespace pxgfcref

#endif // PXG_FEMCLOTH_SOLVE_REF_H
