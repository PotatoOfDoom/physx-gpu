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

// Verification of npD6JointPrep.slang: D6JointSolverPrep
// (gpusolver/src/CUDA/constraintBlockPrePrep.cu L516-737) + the full
// PxgConstraintHelper / swing-twist limit helper stack, verified against an
// exact CPU transcription of the CUDA source.  Compare doctrine: the trig
// bits (counts.y), cA2wOut/cB2wOut and every non-trig case verify BITWISE
// (modulo the documented +-0 class); cases whose configuration executed
// atan2/acos/asin/tan (deterministically flagged) compare float lanes to
// tolerance with flags/hints/counts bitwise; a trig-value flip of a HARD
// anglePair gate (angle vs lower/upper) can change the emitted row set --
// such cases are excluded only when the trig bits match and both frame
// outputs are bitwise-equal, counted, and CAPPED (the tranche-171b
// trig-unstable class).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxgSolverCoreDesc.h"      // PxgPrePrepDesc, PxgSolverSharedDesc, IterativeSolveData
#include "PxgSolverConstraintDesc.h"
#include "PxgConstraintPrep.h"      // PxgConstraintData, PxgConstraintPrePrep, PxgD6JointData
#include "PxgSolverBody.h"          // PxgSolverBodyData
#include "PxConstraintDesc.h"       // Px1DConstraint

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };
struct Q4 { float x, y, z, w; };

struct D6In
{
    F4 c2bAq; F4 c2bAp;   // c2bAp.w = asuint(useExtendedLimits)
    F4 c2bBq; F4 c2bBp;   // c2bBp.w = asuint(use bits)
    F4 bA2wq; F4 bA2wp;   // bA2wp.w = asuint(locked)
    F4 bB2wq; F4 bB2wp;   // bB2wp.w = asuint(limited)
    F4 drvPq; F4 drvPp;   // drvPp.w = asuint(driving)
    F4 linVel;            // w = distanceMinDist
    F4 angVel;
    F4 drives[6];         // stiffness, damping, forceLimit, asuint(flags)
    F4 distP;  F4 distV;
    F4 linXP;  F4 linXV;  // x = upper, y = lower
    F4 linYP;  F4 linYV;
    F4 linZP;  F4 linZV;
    F4 twistP; F4 twistV;
    F4 swingP; F4 swingV; // x = yAngle, y = zAngle
    F4 pyrP;   F4 pyrV;   // yMin, yMax, zMin, zMax
};

const uint32_t MAX_ROWS = 24;

struct D6Out
{
    U4 counts;   // x = numConstraints, y = trig bits
    F4 cAoutW;
    F4 cBoutW;
    U4 nrfs;     // c1dPrePrep: mNumRows_Flags_StartIndex image
    F4 invMass;  // c1dPrePrep: mInvMassScale passthrough
    F4 raW;      // c1dPrePrep: mRaWorld_linBreakForceW
    F4 rbW;      // c1dPrePrep: mRbWorld_angBreakForceW
    F4 rows[120];
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static bool eqBits(float a, float b) { return f2u(a) == f2u(b); }
static bool eqTol(float a, float b)
{
    if (std::isnan(a) && std::isnan(b)) return true;
    const float diff = std::fabs(a - b);
    return diff <= 2e-3f + 1e-3f * std::fmax(std::fabs(a), std::fabs(b));
}
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }
static bool eqF4FZ(const F4& a, const F4& b)
{ return eqFZ(a.x, b.x) && eqFZ(a.y, b.y) && eqFZ(a.z, b.z) && eqFZ(a.w, b.w); }

// ---- exact CPU transcription (namespace d6ref) ----
namespace d6ref
{
const float FLT_MAXQ = 3.402823466e+38f;

const uint32_t E_SPRING = 1, E_ACCELERATION_SPRING = 2, E_RESTITUTION = 4, E_KEEPBIAS = 8,
               E_OUTPUT_FORCE = 16, E_HAS_DRIVE_LIMIT = 32, E_ANGULAR = 64;
const uint32_t H_NONE = 0, H_SLERP_SPRING = 258, H_EQUALITY = 2048, H_INEQUALITY = 2049;

inline V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
inline V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
inline V3 neg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
inline V3 mul(V3 a, float f) { return V3{ a.x * f, a.y * f, a.z * f }; }
inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline float dot4(Q4 a, Q4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline Q4 qneg(Q4 q) { return Q4{ -q.x, -q.y, -q.z, -q.w }; }
inline Q4 qconj(Q4 q) { return Q4{ -q.x, -q.y, -q.z, q.w }; }

// PxQuat::rotate
inline V3 qrot(Q4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}

// PxQuat::operator*
inline Q4 qmul(Q4 a, Q4 b)
{
    return Q4{ a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
               a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
               a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

inline V3 basis0(Q4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return V3{ (q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2 };
}
inline V3 basis1(Q4 q)
{
    const float y2 = q.y * 2.0f, w2 = q.w * 2.0f;
    return V3{ (-q.z * w2) + q.x * y2, (q.w * w2) - 1.0f + q.y * y2, (q.x * w2) + q.z * y2 };
}
inline V3 basis2(Q4 q)
{
    const float z2 = q.z * 2.0f, w2 = q.w * 2.0f;
    return V3{ (q.y * w2) + q.x * z2, (-q.x * w2) + q.y * z2, (q.w * w2) - 1.0f + q.z * z2 };
}

inline void quatToMat33(Q4 q, V3& c0, V3& c1, V3& c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = V3{ 1.0f - yy - zz, xy + zw, xz - yw };
    c1 = V3{ xy - zw, 1.0f - xx - zz, yz + xw };
    c2 = V3{ xz + yw, yz - xw, 1.0f - xx - yy };
}

// computeJacobianAxes (PxgConstraintHelper.h L43-58)
inline void jacobianAxes(Q4 qa, Q4 qb, V3 row[3])
{
    const float wa = qa.w, wb = qb.w;
    const V3 va{ qa.x, qa.y, qa.z }, vb{ qb.x, qb.y, qb.z };
    const V3 c = add(mul(vb, wa), mul(va, wb));
    const float d = wa * wb - dot3(va, vb);
    row[0] = mul(add(add(mul(va, vb.x), mul(vb, va.x)), V3{ d, c.z, -c.y }), 0.5f);
    row[1] = mul(add(add(mul(va, vb.y), mul(vb, va.y)), V3{ -c.z, d, c.x }), 0.5f);
    row[2] = mul(add(add(mul(va, vb.z), mul(vb, va.z)), V3{ c.y, -c.x, d }), 0.5f);
}

// PxSeparateSwingTwist
inline void separateSwingTwist(Q4 q, Q4& swing, Q4& twist)
{
    if (q.x != 0.0f)
    {
        const float s = 1.0f / std::sqrt(q.x * q.x + 0.0f * 0.0f + 0.0f * 0.0f + q.w * q.w);
        twist = Q4{ q.x * s, 0.0f * s, 0.0f * s, q.w * s };
    }
    else
        twist = Q4{ 0, 0, 0, 1 };
    swing = qmul(q, qconj(twist));
}

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// computeSwingAngle (constraintBlockPrePrep.cu L102-105)
inline float computeSwingAngle(float swingYZ, float swingW, uint32_t& trig)
{
    trig |= 1u;
    return 4.0f * std::atan2(swingYZ, 1.0f + swingW);
}

// computePhi (L107-116)
inline float computePhi(Q4 q, uint32_t& trig)
{
    Q4 t = q;
    const float mag = std::sqrt(dot4(t, t));
    if (mag != 0.0f)
    {
        const float imag = 1.0f / mag;
        t = Q4{ t.x * imag, t.y * imag, t.z * imag, t.w * imag };
    }
    trig |= 2u;
    float angle = std::acos(clampf(t.w, -1.0f, 1.0f)) * 2.0f;
    if (t.x < 0.0f)
        angle = -angle;
    return angle;
}

// PxEllipseClamp (PxMathUtils.h L204-262)
inline V3 ellipseClamp(V3 pt, V3 radii)
{
    V3 q{ 0.0f, std::fabs(pt.y), std::fabs(pt.z) };
    const float tinyEps = 1e-6f;
    if (radii.y >= radii.z)
    {
        if (q.z < tinyEps) return V3{ 0.0f, pt.y > 0.0f ? radii.y : -radii.y, 0.0f };
    }
    else
    {
        if (q.y < tinyEps) return V3{ 0.0f, 0.0f, pt.z > 0.0f ? radii.z : -radii.z };
    }
    V3 denom{ 0, 0, 0 };
    const V3 e2{ radii.x * radii.x, radii.y * radii.y, radii.z * radii.z };
    const V3 eq{ radii.x * q.x, radii.y * q.y, radii.z * q.z };
    const float ty = eq.y - e2.y, tz = eq.z - e2.z;
    float t = (ty < tz) ? tz : ty;  // PxMax
    for (uint32_t i = 0; i < 20; ++i)
    {
        denom = V3{ 0.0f, 1.0f / (t + e2.y), 1.0f / (t + e2.z) };
        const V3 denom2{ eq.x * denom.x, eq.y * denom.y, eq.z * denom.z };
        const V3 fv{ denom2.x * denom2.x, denom2.y * denom2.y, denom2.z * denom2.z };
        const float f = fv.y + fv.z - 1.0f;
        if (f < 1e-4f)
            return V3{ e2.x * pt.x * denom.x, e2.y * pt.y * denom.y, e2.z * pt.z * denom.z };
        const float df = dot3(fv, denom) * -2.0f;
        t = t - f / df;
    }
    const V3 r{ e2.x * pt.x * denom.x, e2.y * pt.y * denom.y, e2.z * pt.z * denom.z };
    const float ry = r.y / radii.y, rz = r.z / radii.z;
    return mul(r, 1.0f / std::sqrt(ry * ry + rz * rz));
}

// Cm::computeAxisAndError (CmConeLimitHelper.h L50-69)
inline float coneAxisAndError(V3 r, V3 d, V3 twistAxis, V3& axis)
{
    const float r2 = dot3(r, r), a = 1.0f - r2, b = 1.0f / (1.0f + r2), b2 = b * b;
    const float v1 = 2.0f * a * b2;
    const V3 v2{ a, 2.0f * r.z, -2.0f * r.y };
    const V3 coneLine = sub(mul(v2, v1), V3{ 1.0f, 0.0f, 0.0f });

    const float rd = dot3(r, d);
    const float dv1 = -4.0f * rd * (3.0f - r2) * b2 * b;
    const V3 dv2{ -2.0f * rd, 2.0f * d.z, -2.0f * d.y };

    const V3 coneNormal = add(mul(dv2, v1), mul(v2, dv1));

    const float cnMag = std::sqrt(dot3(coneNormal, coneNormal));
    axis = mul(cross(coneLine, coneNormal), 1.0f / cnMag);  // PxVec3 operator/: reciprocal mult
    return dot3(cross(coneLine, axis), twistAxis);
}

// Cm::ConeLimitHelperTanLess::getLimit (CmConeLimitHelper.h L163-178)
inline void coneGetLimit(float yMax, float zMax, Q4 swing, V3& axis, float& error, uint32_t& trig)
{
    trig |= 8u;
    const V3 twistAxis = basis0(swing);
    const V3 swingAngle{ 0.0f, 4.0f * std::atan2(swing.y, 1.0f + swing.w),
                         4.0f * std::atan2(swing.z, 1.0f + swing.w) };
    const V3 clamped = ellipseClamp(swingAngle, V3{ 0.0f, yMax, zMax });
    const V3 normal{ 0.0f, clamped.y / (yMax * yMax), clamped.z / (zMax * zMax) };
    const V3 r{ 0.0f, std::tan(clamped.y / 4.0f), std::tan(clamped.z / 4.0f) };
    const V3 d{ 0.0f, normal.y, normal.z };
    error = coneAxisAndError(r, d, twistAxis, axis);
}

struct C1D
{
    V3 lin0; float geo;
    V3 ang0; float vt;
    V3 lin1; float minI;
    V3 ang1; float maxI;
    float m0, m1;
    uint32_t flags, hint;
};

struct RowBuf { C1D r[MAX_ROWS]; uint32_t n; };

inline C1D chLinear(V3 mRa, V3 mRb, V3 axis, float posErr, uint32_t hint)
{
    C1D c;
    c.hint = hint;
    c.lin0 = axis; c.ang0 = cross(mRa, axis);
    c.lin1 = axis; c.ang1 = cross(mRb, axis);
    c.geo = posErr;
    c.flags = 0;
    c.minI = -FLT_MAXQ; c.maxI = FLT_MAXQ;
    c.m1 = 0.0f; c.m0 = 0.0f;
    c.vt = 0.0f;
    return c;
}

inline C1D chLinearRa(V3 ra, V3 rb, V3 axis, float posErr, uint32_t hint, uint32_t flags)
{
    C1D c = chLinear(ra, rb, axis, posErr, hint);
    c.flags = flags;
    return c;
}

inline C1D chAngular(V3 axis, float posErr, uint32_t hint, uint32_t flags)
{
    C1D c;
    c.hint = hint;
    c.lin0 = V3{ 0, 0, 0 }; c.ang0 = axis;
    c.lin1 = V3{ 0, 0, 0 }; c.ang1 = axis;
    c.geo = posErr;
    c.flags = flags | E_ANGULAR;
    c.minI = -FLT_MAXQ; c.maxI = FLT_MAXQ;
    c.m1 = 0.0f; c.m0 = 0.0f;
    c.vt = 0.0f;
    return c;
}

inline bool isSoft(const F4& lp) { return lp.w > 0.0f || lp.z > 0.0f; }

inline void chAddLimit(C1D& c, const F4& lp)
{
    uint32_t flags = c.flags | E_OUTPUT_FORCE;
    if (isSoft(lp))
    {
        flags |= E_SPRING;
        c.m0 = lp.z; c.m1 = lp.w;
    }
    else
    {
        c.hint = H_INEQUALITY;
        c.m0 = lp.x; c.m1 = lp.y;
        if (c.geo > 0.0f) flags |= E_KEEPBIAS;
        if (lp.x > 0.0f) flags |= E_RESTITUTION;
    }
    c.flags = flags;
    c.minI = 0.0f;
}

inline void chAddDrive(C1D& c, float velTarget, const F4& drv)
{
    c.vt = velTarget;
    uint32_t flags = c.flags | E_SPRING | E_HAS_DRIVE_LIMIT;
    const uint32_t df = f2u(drv.w);
    if (df & 1u) flags |= E_ACCELERATION_SPRING;
    if (df & 2u) flags |= E_OUTPUT_FORCE;
    c.flags = flags;
    c.m0 = drv.x; c.m1 = drv.y;
    c.minI = -drv.z; c.maxI = drv.z;
}

inline void chLinearLimit(RowBuf& buf, V3 mRa, V3 mRb, V3 axis, float ordinate, float limitValue,
                          const F4& lp)
{
    if (!isSoft(lp) || ordinate > limitValue)
    {
        C1D c = chLinear(mRa, mRb, axis, limitValue - ordinate, H_NONE);
        chAddLimit(c, lp);
        buf.r[buf.n++] = c;
    }
}

inline void chAngularLimit(RowBuf& buf, V3 axis, float error, const F4& lp)
{
    C1D c = chAngular(axis, error, H_NONE, 0);
    chAddLimit(c, lp);
    buf.r[buf.n++] = c;
}

inline void chAnglePair(RowBuf& buf, float angle, float lower, float upper, V3 axis, const F4& lp)
{
    const bool soft = isSoft(lp);
    if (!soft || angle < lower)
        chAngularLimit(buf, neg(axis), -(lower - angle), lp);
    if (!soft || angle > upper)
        chAngularLimit(buf, axis, upper - angle, lp);
}

inline V3 getNormalized(V3 v)
{
    const float m = dot3(v, v);
    if (m > 0.0f)
        return mul(v, 1.0f / std::sqrt(m));
    return V3{ 0, 0, 0 };
}

// D6JointSolverPrep (constraintBlockPrePrep.cu L516-737)
void solverPrep(const D6In& s, D6Out& o)
{
    std::memset(&o, 0, sizeof(o));
    uint32_t trig = 0;
    RowBuf buf; buf.n = 0;

    const uint32_t useExtendedLimits = f2u(s.c2bAp.w);
    const uint32_t useBits = f2u(s.c2bBp.w);
    const bool useDistanceLimit = (useBits & 1u) != 0;
    const bool useNewLinearLimits = (useBits & 2u) != 0;
    const bool useConeLimit = (useBits & 4u) != 0;
    const bool usePyramidLimits = (useBits & 8u) != 0;

    uint32_t locked = f2u(s.bA2wp.w);
    const uint32_t limited = f2u(s.bB2wp.w);
    const uint32_t driving = f2u(s.drvPp.w);

    const Q4 bA2wq{ s.bA2wq.x, s.bA2wq.y, s.bA2wq.z, s.bA2wq.w };
    const V3 bA2wp{ s.bA2wp.x, s.bA2wp.y, s.bA2wp.z };
    const Q4 bB2wq{ s.bB2wq.x, s.bB2wq.y, s.bB2wq.z, s.bB2wq.w };
    const V3 bB2wp{ s.bB2wp.x, s.bB2wp.y, s.bB2wp.z };

    Q4 cA2wq = qmul(bA2wq, Q4{ s.c2bAq.x, s.c2bAq.y, s.c2bAq.z, s.c2bAq.w });
    V3 cA2wp = add(qrot(bA2wq, V3{ s.c2bAp.x, s.c2bAp.y, s.c2bAp.z }), bA2wp);
    Q4 cB2wq = qmul(bB2wq, Q4{ s.c2bBq.x, s.c2bBq.y, s.c2bBq.z, s.c2bBq.w });
    V3 cB2wp = add(qrot(bB2wq, V3{ s.c2bBp.x, s.c2bBp.y, s.c2bBp.z }), bB2wp);

    const V3 mRa = sub(cB2wp, bA2wp);
    const V3 mRb = sub(cB2wp, bB2wp);

    const uint32_t SWING1_FLAG = 1u << 4, SWING2_FLAG = 1u << 5, TWIST_FLAG = 1u << 3;
    const uint32_t ANGULAR_MASK = SWING1_FLAG | SWING2_FLAG | TWIST_FLAG;
    const uint32_t LINEAR_MASK = 7u;

    if (!useExtendedLimits)
    {
        if (dot4(cA2wq, cB2wq) < 0.0f)
            cB2wq = qneg(cB2wq);
    }

    const Q4 cA2wqInv = qconj(cA2wq);
    const V3 cB2cAp = qrot(cA2wqInv, sub(cB2wp, cA2wp));
    const Q4 cB2cAq = qmul(cA2wqInv, cB2wq);

    V3 am[3], bm[3];
    quatToMat33(cA2wq, am[0], am[1], am[2]);
    quatToMat33(cB2wq, bm[0], bm[1], bm[2]);

    const V3 bX = bm[0], aY = am[1], aZ = am[2];

    // linear drives
    if (driving & 7u)
    {
        const V3 posErr = sub(V3{ s.drvPp.x, s.drvPp.y, s.drvPp.z }, cB2cAp);
        const float lv[3] = { s.linVel.x, s.linVel.y, s.linVel.z };
        const float pe[3] = { posErr.x, posErr.y, posErr.z };
        for (uint32_t i = 0; i < 3; ++i)
        {
            if (driving & (1u << i))
            {
                C1D c = chLinear(mRa, mRb, am[i], pe[i], H_NONE);
                chAddDrive(c, -lv[i], s.drives[i]);
                buf.r[buf.n++] = c;
            }
        }
    }

    // angular drives
    if (driving & ((1u << 5) | (1u << 3) | (1u << 4) | (1u << 6) | (1u << 7)))
    {
        const Q4 drvq{ s.drvPq.x, s.drvPq.y, s.drvPq.z, s.drvPq.w };
        const Q4 d2cA_q = dot4(cB2cAq, drvq) > 0.0f ? drvq : qneg(drvq);
        const V3 v{ s.angVel.x, s.angVel.y, s.angVel.z };
        const Q4 delta = qmul(qconj(d2cA_q), cB2cAq);

        if (driving & (1u << 5))  // SLERP
        {
            const V3 velTarget = neg(qrot(cA2wq, v));
            V3 axis[3] = { V3{ 1, 0, 0 }, V3{ 0, 1, 0 }, V3{ 0, 0, 1 } };
            if (s.drives[5].x != 0.0f)
                jacobianAxes(qmul(cA2wq, d2cA_q), cB2wq, axis);
            const float de[3] = { delta.x, delta.y, delta.z };
            for (uint32_t i = 0; i < 3; ++i)
            {
                C1D c = chAngular(axis[i], -de[i], H_SLERP_SPRING, 0);
                chAddDrive(c, dot3(axis[i], velTarget), s.drives[5]);
                buf.r[buf.n++] = c;
            }
        }
        else
        {
            if (driving & (1u << 4))  // TWIST
            {
                C1D c = chAngular(am[0], -2.0f * delta.x, H_NONE, 0);
                chAddDrive(c, v.x, s.drives[4]);
                buf.r[buf.n++] = c;
            }
            if (driving & (1u << 3))  // SWING
            {
                const V3 err = basis0(delta);
                if (!(locked & SWING1_FLAG))
                {
                    C1D c = chAngular(aY, err.z, H_NONE, 0);
                    chAddDrive(c, v.y, s.drives[3]);
                    buf.r[buf.n++] = c;
                }
                if (!(locked & SWING2_FLAG))
                {
                    C1D c = chAngular(aZ, -err.y, H_NONE, 0);
                    chAddDrive(c, v.z, s.drives[3]);
                    buf.r[buf.n++] = c;
                }
            }
            else if (driving & ((1u << 6) | (1u << 7)))  // SWING1 / SWING2
            {
                const V3 err = basis0(delta);
                if (driving & (1u << 6))
                {
                    C1D c = chAngular(aY, err.z, H_NONE, 0);
                    chAddDrive(c, v.y, s.drives[3]);  // gDriveSwing1DataIndex = eSWING entry
                    buf.r[buf.n++] = c;
                }
                if (driving & (1u << 7))
                {
                    C1D c = chAngular(aZ, -err.y, H_NONE, 0);
                    chAddDrive(c, v.z, s.drives[5]);  // gDriveSwing2DataIndex = eSLERP entry
                    buf.r[buf.n++] = c;
                }
            }
        }
    }

    // angular limits
    if (limited & ANGULAR_MASK)
    {
        Q4 swing, twist;
        separateSwingTwist(cB2cAq, swing, twist);

        if ((limited & SWING1_FLAG) && (limited & SWING2_FLAG))
        {
            if (useConeLimit)
            {
                V3 axis; float error;
                coneGetLimit(s.swingV.x, s.swingV.y, swing, axis, error, trig);
                chAngularLimit(buf, qrot(cA2wq, axis), error, s.swingP);
            }
            if (usePyramidLimits)
            {
                const Q4 q = qmul(cA2wq, swing);
                chAnglePair(buf, computeSwingAngle(swing.y, swing.w, trig), s.pyrV.x, s.pyrV.y,
                            basis1(q), s.pyrP);
                chAnglePair(buf, computeSwingAngle(swing.z, swing.w, trig), s.pyrV.z, s.pyrV.w,
                            basis2(q), s.pyrP);
            }
        }
        else
        {
            if (limited & SWING1_FLAG)
            {
                if (locked & SWING2_FLAG)
                {
                    if (usePyramidLimits)
                    {
                        const Q4 q = qmul(cA2wq, swing);
                        chAnglePair(buf, computeSwingAngle(swing.y, swing.w, trig), s.pyrV.x,
                                    s.pyrV.y, basis1(q), s.pyrP);
                    }
                    else
                        chAnglePair(buf, computeSwingAngle(swing.y, swing.w, trig), -s.swingV.x,
                                    s.swingV.x, aY, s.swingP);
                }
                else
                {
                    trig |= 4u;
                    chAnglePair(buf, std::asin(clampf(-dot3(aZ, bX), -1.0f, 1.0f)), -s.swingV.x,
                                s.swingV.x, getNormalized(cross(aZ, bX)), s.swingP);
                }
            }
            if (limited & SWING2_FLAG)
            {
                if (locked & SWING1_FLAG)
                {
                    if (usePyramidLimits)
                    {
                        const Q4 q = qmul(cA2wq, swing);
                        chAnglePair(buf, computeSwingAngle(swing.z, swing.w, trig), s.pyrV.z,
                                    s.pyrV.w, basis2(q), s.pyrP);
                    }
                    else
                        chAnglePair(buf, computeSwingAngle(swing.z, swing.w, trig), -s.swingV.y,
                                    s.swingV.y, aZ, s.swingP);
                }
                else
                {
                    trig |= 4u;
                    chAnglePair(buf, std::asin(clampf(dot3(aY, bX), -1.0f, 1.0f)), -s.swingV.y,
                                s.swingV.y, getNormalized(neg(cross(aY, bX))), s.swingP);
                }
            }
        }

        if (limited & TWIST_FLAG)
            chAnglePair(buf, computePhi(twist, trig), s.twistV.y, s.twistV.x, bm[0], s.twistP);
    }

    // linear limits
    if (limited & LINEAR_MASK)
    {
        if (useDistanceLimit)
        {
            V3 limitDir{ 0, 0, 0 };
            const float pc[3] = { cB2cAp.x, cB2cAp.y, cB2cAp.z };
            for (uint32_t i = 0; i < 3; ++i)
            {
                if (limited & (1u << i))
                    limitDir = add(limitDir, mul(am[i], pc[i]));
            }
            const float distance = std::sqrt(dot3(limitDir, limitDir));
            if (distance > s.linVel.w)
                chLinearLimit(buf, mRa, mRb, mul(limitDir, 1.0f / distance), distance, s.distV.x,
                              s.distP);
        }

        if (useNewLinearLimits)
        {
            if ((limited & 1u) && s.linXV.y <= s.linXV.x)
            {
                chLinearLimit(buf, mRa, mRb, am[0], cB2cAp.x, s.linXV.x, s.linXP);
                chLinearLimit(buf, mRa, mRb, neg(am[0]), -cB2cAp.x, -s.linXV.y, s.linXP);
            }
            if ((limited & 2u) && s.linYV.y <= s.linYV.x)
            {
                chLinearLimit(buf, mRa, mRb, am[1], cB2cAp.y, s.linYV.x, s.linYP);
                chLinearLimit(buf, mRa, mRb, neg(am[1]), -cB2cAp.y, -s.linYV.y, s.linYP);
            }
            if ((limited & 4u) && s.linZV.y <= s.linZV.x)
            {
                chLinearLimit(buf, mRa, mRb, am[2], cB2cAp.z, s.linZV.x, s.linZP);
                chLinearLimit(buf, mRa, mRb, neg(am[2]), -cB2cAp.z, -s.linZV.y, s.linZP);
            }
        }
    }

    // single locked swing special case
    const uint32_t angularLocked = locked & ANGULAR_MASK;
    if (angularLocked == SWING1_FLAG)
    {
        C1D c = chAngular(cross(bX, aZ), -dot3(bX, aZ), H_EQUALITY, 0);
        c.flags |= E_OUTPUT_FORCE;
        buf.r[buf.n++] = c;
        locked &= ~SWING1_FLAG;
    }
    else if (angularLocked == SWING2_FLAG)
    {
        locked &= ~SWING2_FLAG;
        C1D c = chAngular(cross(bX, aY), -dot3(bX, aY), H_EQUALITY, 0);
        c.flags |= E_OUTPUT_FORCE;
        buf.r[buf.n++] = c;
    }

    // prepareLockedAxes
    const uint32_t lin = locked & 7u;
    const uint32_t ang = locked >> 3u;
    V3 ra = mRa;
    if (lin)
    {
        V3 ax[3];
        quatToMat33(cA2wq, ax[0], ax[1], ax[2]);
        V3 errorVector{ 0, 0, 0 };
        if (lin & 1u) errorVector = sub(errorVector, mul(ax[0], cB2cAp.x));
        if (lin & 2u) errorVector = sub(errorVector, mul(ax[1], cB2cAp.y));
        if (lin & 4u) errorVector = sub(errorVector, mul(ax[2], cB2cAp.z));
        ra = add(ra, errorVector);
        const float pc[3] = { cB2cAp.x, cB2cAp.y, cB2cAp.z };
        for (uint32_t i = 0; i < 3; ++i)
            if (lin & (1u << i))
                buf.r[buf.n++] = chLinearRa(ra, mRb, ax[i], -pc[i], H_EQUALITY, E_OUTPUT_FORCE);
    }
    if (ang)
    {
        const Q4 qB2qA = qmul(qconj(cA2wq), cB2wq);
        V3 row[3];
        jacobianAxes(cA2wq, cB2wq, row);
        const float imp[3] = { qB2qA.x, qB2qA.y, qB2qA.z };
        for (uint32_t i = 0; i < 3; ++i)
            if (ang & (1u << i))
                buf.r[buf.n++] = chAngular(row[i], -imp[i], H_EQUALITY, E_OUTPUT_FORCE);
    }

    PXG_CHECK_MSG(buf.n <= MAX_ROWS, "ref row overflow %u", buf.n);

    o.counts = U4{ buf.n, trig, 0, 0 };
    const V3 cAout = add(ra, bA2wp);
    o.cAoutW = F4{ cAout.x, cAout.y, cAout.z, 0.0f };
    o.cBoutW = F4{ cB2wp.x, cB2wp.y, cB2wp.z, 0.0f };
    for (uint32_t i = 0; i < buf.n; ++i)
    {
        const C1D& c = buf.r[i];
        o.rows[i * 5 + 0] = F4{ c.lin0.x, c.lin0.y, c.lin0.z, c.geo };
        o.rows[i * 5 + 1] = F4{ c.ang0.x, c.ang0.y, c.ang0.z, c.vt };
        o.rows[i * 5 + 2] = F4{ c.lin1.x, c.lin1.y, c.lin1.z, c.minI };
        o.rows[i * 5 + 3] = F4{ c.ang1.x, c.ang1.y, c.ang1.z, c.maxI };
        float packed; const uint32_t pv = c.flags | (c.hint << 16);
        std::memcpy(&packed, &pv, 4);
        o.rows[i * 5 + 4] = F4{ c.m0, c.m1, packed, 0.0f };
    }
}

} // namespace d6ref

static float rfS(std::mt19937& rng, float lo, float hi)
{ return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); }
static F4 randQuatS(std::mt19937& rng)
{
    float a[4]; float m2 = 0;
    for (int k = 0; k < 4; ++k) { a[k] = rfS(rng, -1, 1); m2 += a[k] * a[k]; }
    const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
    return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i };
}
static float asF(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static F4 limitParamsS(std::mt19937& rng, bool hard)
{
    // (restitution, bounceThreshold, stiffness, damping)
    if (hard) return F4{ (rng() % 2) ? rfS(rng, 0.f, 1.f) : 0.f, rfS(rng, 0.f, 1.f), 0.f, 0.f };
    return F4{ 0.f, 0.f, (rng() % 4) ? rfS(rng, 1.f, 500.f) : 0.f, rfS(rng, 0.5f, 50.f) };
}

// shared fixture generator (both suites); returns the hard/soft pick
static bool genFixture(std::mt19937& rng, uint32_t i, D6In& s)
{
    auto rf = [&](float lo, float hi) { return rfS(rng, lo, hi); };
    auto randQuat = [&]() { return randQuatS(rng); };
    auto limitParams = [&](bool h) { return limitParamsS(rng, h); };

    {
        // joint-frame and body poses
        s.c2bAq = (i % 6 == 0) ? F4{ 0, 0, 0, 1 } : randQuat();
        s.c2bAp = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), 0 };
        s.c2bBq = (i % 6 == 1) ? F4{ 0, 0, 0, 1 } : randQuat();
        s.c2bBp = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), 0 };
        s.bA2wq = randQuat();
        s.bA2wp = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.bB2wq = (i % 9 == 2) ? s.bA2wq : randQuat();
        s.bB2wp = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        if (i % 29 == 13)
        {
            // identity everything: swing = twist = identity, exercises the
            // q.x == 0 branch of PxSeparateSwingTwist and zero errors
            s.c2bAq = s.c2bBq = s.bA2wq = s.bB2wq = F4{ 0, 0, 0, 1 };
            s.c2bAp = s.c2bBp = F4{ 0, 0, 0, 0 };
            s.bA2wp = s.bB2wp = F4{ 0, 0, 0, 0 };
        }
        if (i % 9 == 4)
        {
            // near-aligned pair: small relative rotation keeps hard swing
            // angles inside the limits (the no-row anglePair path)
            s.bB2wq = s.bA2wq;
            s.c2bBq = s.c2bAq;
        }

        // per-axis motion: 0 = locked, 1 = limited, 2 = free
        uint32_t motion[6];
        for (int a = 0; a < 6; ++a)
        {
            const uint32_t r = rng() % 10;
            motion[a] = r < 4 ? 0u : (r < 7 ? 1u : 2u);
        }
        if (i % 13 == 7)
            for (int a = 0; a < 6; ++a) motion[a] = 0;   // fully locked joint
        if (i % 17 == 5)
        {
            // exactly one swing locked (the angularHard special case)
            motion[3] = 1 + rng() % 2;                   // twist not locked
            if (rng() % 2) { motion[4] = 0; motion[5] = 1 + rng() % 2; }
            else           { motion[5] = 0; motion[4] = 1 + rng() % 2; }
        }
        if (i % 11 == 3)
        {
            // both swings limited (cone / pyramid paths)
            motion[4] = 1; motion[5] = 1;
        }
        uint32_t locked = 0, limited = 0;
        for (int a = 0; a < 6; ++a)
        {
            if (motion[a] == 0) locked |= (1u << a);
            else if (motion[a] == 1) limited |= (1u << a);
        }

        // driving bitmap
        uint32_t driving = 0;
        for (int a = 0; a < 3; ++a)
            if (motion[a] != 0 && rng() % 3 == 0) driving |= (1u << a);
        const uint32_t angDrvKind = rng() % 5;
        if (angDrvKind == 1)
        {
            driving |= (1u << 5);                                        // SLERP
            // review-174 gap: SLERP together with SWING/TWIST bits pins the
            // if/else precedence (SLERP wins, cu L579-591)
            if (rng() % 2) driving |= (1u << 4) | (1u << 3);
        }
        else if (angDrvKind == 2) driving |= (1u << 4) | (rng() % 2 ? (1u << 3) : 0u);
        else if (angDrvKind == 3) driving |= (1u << 3);                  // SWING
        else if (angDrvKind == 4)
            driving |= (rng() % 2 ? (1u << 6) : 0u) | (rng() % 2 ? (1u << 7) : 0u) |
                       (rng() % 3 == 0 ? (1u << 4) : 0u);                // SWING1/SWING2 (+TWIST)

        if (i % 43 == 21)
        {
            // review-174 gap: all-free undriven joints -> zero rows
            locked = 0; limited = 0; driving = 0;
        }

        const uint32_t useExt = (i % 4 == 2) ? 1u : 0u;
        uint32_t useBits = rng() % 16u;
        if (i % 11 == 3) useBits |= (rng() % 2) ? 4u : 12u;  // force cone (and maybe pyramid)
        if (i % 19 == 9) useBits |= 1u;                      // force distance limit

        s.c2bAp.w = asF(useExt);
        s.c2bBp.w = asF(useBits);
        s.bA2wp.w = asF(locked);
        s.bB2wp.w = asF(limited);

        // drive target pose + velocities
        F4 dq = (rng() % 4 == 0) ? F4{ 0, 0, 0, 1 } : randQuat();
        s.drvPq = dq;
        s.drvPp = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), asF(driving) };
        s.linVel = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), (i % 19 == 9) ? rf(0.f, 0.02f) : rf(0.f, 0.3f) };
        s.angVel = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

        for (int d = 0; d < 6; ++d)
        {
            const bool zeroStiff = (d == 5) && (rng() % 2 == 0);  // slerp identity-axes branch
            s.drives[d] = F4{ zeroStiff ? 0.f : rf(0.f, 300.f), rf(0.f, 30.f), rf(1.f, 1e6f),
                              asF(rng() % 4u) };
        }

        const bool hard = (rng() % 2) == 0;
        s.distP = limitParams(hard);
        s.distV = F4{ rf(0.01f, 1.5f), 0, 0, 0 };
        s.linXP = limitParams(hard);
        s.linYP = limitParams(hard);
        s.linZP = limitParams(hard);
        auto linPair = [&]() {
            float lo = rf(-1.f, 0.5f), hi = lo + rf(0.f, 1.5f);
            if (rng() % 8 == 0) hi = lo;                 // lower == upper (locked-style pair)
            if (i % 23 == 11 && rng() % 2) { const float t = lo; lo = hi + 0.1f; hi = t; } // skip gate
            return F4{ hi, lo, 0, 0 };                   // x = upper, y = lower
        };
        s.linXV = linPair(); s.linYV = linPair(); s.linZV = linPair();
        s.twistP = limitParams(hard);
        { float lo = rf(-2.8f, 1.f); s.twistV = F4{ lo + rf(0.05f, 1.8f), lo, 0, 0 }; }
        s.swingP = limitParams(hard);
        s.swingV = F4{ rf(0.05f, 2.9f), rf(0.05f, 2.9f), 0, 0 };
        s.pyrP = limitParams(hard);
        { float ylo = rf(-2.5f, 0.5f), zlo = rf(-2.5f, 0.5f);
          s.pyrV = F4{ ylo, ylo + rf(0.05f, 2.f), zlo, zlo + rf(0.05f, 2.f) }; }

        return hard;
    }
}

PXG_TEST(npD6JointPrep_prep)
{
    std::mt19937 rng(174001);
    const uint32_t N = 4000;

    std::vector<D6In> in(N);
    std::vector<D6Out> ref(N);
    size_t cTrig = 0, cSlerp = 0, cCone = 0, cPyr = 0, cDual = 0, cSingle = 0, cTwistL = 0,
           cDist = 0, cNewLin = 0, cSpecial = 0, cLockRows = 0, cHard = 0, cSoft = 0,
           cRows = 0, cZeroRows = 0, cLinDrv = 0, cAngDrv = 0, cExt = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        D6In s{};
        const bool hard = genFixture(rng, i, s);
        if (hard) ++cHard; else ++cSoft;

        const uint32_t locked = f2u(s.bA2wp.w);
        const uint32_t limited = f2u(s.bB2wp.w);
        const uint32_t driving = f2u(s.drvPp.w);
        const uint32_t useBits = f2u(s.c2bBp.w);
        const uint32_t useExt = f2u(s.c2bAp.w);

        in[i] = s;
        d6ref::solverPrep(s, ref[i]);

        const uint32_t trig = ref[i].counts.y;
        if (trig) ++cTrig;
        if (trig & 8u) ++cCone;
        if (trig & 4u) ++cDual;
        if (trig & 2u) ++cTwistL;
        if ((limited & 0x30u) && (useBits & 8u)) ++cPyr;
        if ((driving & (1u << 5))) ++cSlerp;
        if (driving & 7u) ++cLinDrv;
        if (driving & 0xF8u) ++cAngDrv;
        if ((limited & 0x30u) == 0x10u || (limited & 0x30u) == 0x20u)
            if (!(useBits & 8u) && ((locked >> 4) & 3u)) ++cSingle;
        if ((limited & 7u) && (useBits & 1u)) ++cDist;
        if ((limited & 7u) && (useBits & 2u)) ++cNewLin;
        const uint32_t angLock = locked & 0x38u;
        if (angLock == 0x10u || angLock == 0x20u) ++cSpecial;
        if (locked) ++cLockRows;
        cRows += ref[i].counts.x;
        if (ref[i].counts.x == 0) ++cZeroRows;
        if (useExt) ++cExt;
    }

    std::printf("[ info ] d6 cov: rows %zu trig %zu cone %zu pyr %zu dual %zu single %zu twistL %zu "
                "dist %zu newlin %zu special %zu locked %zu slerp %zu linDrv %zu angDrv %zu "
                "hard %zu soft %zu zero %zu ext %zu\n",
                cRows, cTrig, cCone, cPyr, cDual, cSingle, cTwistL, cDist, cNewLin, cSpecial,
                cLockRows, cSlerp, cLinDrv, cAngDrv, cHard, cSoft, cZeroRows, cExt);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<D6Out> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npD6JointPrep.spv", "d6jPrep",
             { (uint32_t)sizeof(D6In), (uint32_t)sizeof(D6Out) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<D6Out>(dev, bOut, N);
    size_t mm = 0, unstable = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const D6Out& r = ref[i]; const D6Out& q = g[i];

        // the trig bits and both frame outputs are trig-free: always bitwise
        bool meta = r.counts.y == q.counts.y && eqF4FZ(r.cAoutW, q.cAoutW) &&
                    eqF4FZ(r.cBoutW, q.cBoutW);
        if (!meta)
        {
            if (mm < 5)
                std::printf("MM %u meta: tR %u tG %u cA (%g %g %g)/(%g %g %g)\n", i, r.counts.y,
                            q.counts.y, r.cAoutW.x, r.cAoutW.y, r.cAoutW.z, q.cAoutW.x, q.cAoutW.y,
                            q.cAoutW.z);
            ++mm;
            continue;
        }

        const bool trig = r.counts.y != 0;

        // structure: row count and per-row flags|hint words
        bool structOk = r.counts.x == q.counts.x;
        if (structOk)
            for (uint32_t j = 0; j < r.counts.x; ++j)
                structOk = structOk && eqBits(r.rows[j * 5 + 4].z, q.rows[j * 5 + 4].z);
        if (!structOk)
        {
            if (trig)
            {
                // a trig deviation flipped a hard anglePair gate or the
                // KEEPBIAS sign of a trig-derived error: the CUDA original is
                // not bit-portable here either.  Counted and capped.
                ++unstable;
                continue;
            }
            if (mm < 5)
                std::printf("MM %u struct: nR %u nG %u\n", i, r.counts.x, q.counts.x);
            ++mm;
            continue;
        }

        bool ok = true;
        for (uint32_t j = 0; j < r.counts.x * 5 && ok; ++j)
        {
            const F4& a = r.rows[j]; const F4& b = q.rows[j];
            if ((j % 5) == 4)
            {
                ok = trig ? (eqTol(a.x, b.x) && eqTol(a.y, b.y))
                          : (eqFZ(a.x, b.x) && eqFZ(a.y, b.y));
                ok = ok && eqBits(a.z, b.z);
            }
            else
                ok = trig ? (eqTol(a.x, b.x) && eqTol(a.y, b.y) && eqTol(a.z, b.z) && eqTol(a.w, b.w))
                          : eqF4FZ(a, b);
        }
        if (!ok && mm < 5)
        {
            std::printf("MM %u rows: n %u trig %u\n", i, r.counts.x, r.counts.y);
            for (uint32_t j = 0; j < r.counts.x * 5; ++j)
                if (!(eqTol(r.rows[j].x, q.rows[j].x) && eqTol(r.rows[j].y, q.rows[j].y) &&
                      eqTol(r.rows[j].z, q.rows[j].z) && eqTol(r.rows[j].w, q.rows[j].w)))
                    std::printf("  row %u.%u R (%g %g %g %g) G (%g %g %g %g)\n", j / 5, j % 5,
                                r.rows[j].x, r.rows[j].y, r.rows[j].z, r.rows[j].w, q.rows[j].x,
                                q.rows[j].y, q.rows[j].z, q.rows[j].w);
        }
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "d6jPrep: %zu mismatches", mm);
    PXG_CHECK_MSG(unstable <= 8, "d6jPrep: %zu trig-unstable cases (cap 8)", unstable);
    if (unstable)
        std::printf("[ info ] d6 trig-unstable excluded: %zu of %u\n", unstable, N);

    bool cov = cRows > N && cTrig > N / 8 && cCone > 50 && cPyr > 50 && cDual > 50 &&
               cSingle > 20 && cTwistL > N / 8 && cDist > 50 && cNewLin > 50 && cSpecial > 50 &&
               cLockRows > N / 4 && cSlerp > N / 10 && cLinDrv > N / 10 && cAngDrv > N / 4 &&
               cHard > N / 4 && cSoft > N / 4 && cExt > N / 8 && cZeroRows > 50;
    PXG_CHECK_MSG(cov, "d6jPrep: coverage thin (rows %zu trig %zu cone %zu pyr %zu dual %zu "
                       "single %zu special %zu)",
                  cRows, cTrig, cCone, cPyr, cDual, cSingle, cSpecial);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// ---- constraint1DPrePrep wrapper (constraintBlockPrePrep.cu L1059-1166,
// tranche 175): removed-joint gate (NO output write), eDISABLE_CONSTRAINT
// branch, useExtendedLimits derived from the prePrep flags, ra/rb =
// cX2w - poseX.p add-then-subtract roundtrip, PxgConstraintData image ----

PXG_TEST(npD6JointPrep_c1dPrePrep)
{
    std::mt19937 rng(175001);
    const uint32_t N = 2000;
    const uint32_t F_EXT = 1u << 9, F_DISABLE = 1u << 12;  // PxConstraintFlag

    std::vector<D6In> in(N);
    std::vector<D6Out> ref(N);
    std::vector<uint8_t> removed(N);
    size_t cRem = 0, cDis = 0, cExtF = 0, cRun = 0, cTrig = 0, cRows = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        D6In s{};
        genFixture(rng, i, s);

        // prePrep flags: irrelevant bits + independent extended-limits pick
        // (deliberately decoupled from the fixture's c2bAp.w, which the
        // wrapper entry must IGNORE)
        uint32_t ppFlags = ((rng() % 2) ? (1u << 3) : 0u) | ((rng() % 2) ? (1u << 5) : 0u);
        if (rng() % 3 == 0) ppFlags |= F_EXT;
        if (i % 5 == 1) ppFlags |= F_DISABLE;
        const bool rem = (i % 7 == 3);

        s.angVel.w = asF(ppFlags);
        s.swingV.z = asF(rem ? 1u : 0u);
        s.distV.y = rfS(rng, 0.f, 1e5f);   // mLinBreakForce
        s.distV.z = rfS(rng, 0.f, 1e5f);   // mAngBreakForce
        s.distV.w = asF(i * 3u + 7u);      // jointDataIndex, DECOUPLED from the
                                           // thread id (review-175 x5: tid-tied
                                           // fixtures let a hard-wired index pass)
        s.linXV.z = rfS(rng, 0.f, 2.f); s.linXV.w = rfS(rng, 0.f, 2.f);  // invMassScale
        s.linYV.z = rfS(rng, 0.f, 2.f); s.linYV.w = rfS(rng, 0.f, 2.f);

        in[i] = s;
        removed[i] = rem ? 1 : 0;

        D6Out& o = ref[i];
        std::memset(&o, 0, sizeof(o));
        if (rem)
        {
            ++cRem;
        }
        else
        {
            uint32_t numRows = 0;
            F4 raW{ 0, 0, 0, s.distV.y }, rbW{ 0, 0, 0, s.distV.z };
            if (!(ppFlags & F_DISABLE))
            {
                D6In t = s;
                t.c2bAp.w = asF((ppFlags & F_EXT) ? 1u : 0u);
                d6ref::solverPrep(t, o);
                numRows = o.counts.x;
                // ra = ca2w - pose0.p; rb = cb2w - pose1.p (cu L1150-1151)
                raW.x = o.cAoutW.x - s.bA2wp.x; raW.y = o.cAoutW.y - s.bA2wp.y;
                raW.z = o.cAoutW.z - s.bA2wp.z;
                rbW.x = o.cBoutW.x - s.bB2wp.x; rbW.y = o.cBoutW.y - s.bB2wp.y;
                rbW.z = o.cBoutW.z - s.bB2wp.z;
                ++cRun;
                if (o.counts.y) ++cTrig;
                cRows += numRows;
            }
            else
            {
                ++cDis;
            }
            o.counts = U4{ numRows, (ppFlags & F_DISABLE) ? 0u : o.counts.y, 0, 0 };
            o.cAoutW = F4{ 0, 0, 0, 0 };
            o.cBoutW = F4{ 0, 0, 0, 0 };
            o.nrfs = U4{ numRows, ppFlags, (i * 3u + 7u) * 20u, 0 };  // Dy::MAX_CONSTRAINT_ROWS
            o.invMass = F4{ s.linXV.z, s.linXV.w, s.linYV.z, s.linYV.w };
            o.raW = raW;
            o.rbW = rbW;
        }
        if (ppFlags & F_EXT) ++cExtF;
    }

    std::printf("[ info ] c1d cov: run %zu dis %zu rem %zu extF %zu trig %zu rows %zu\n",
                cRun, cDis, cRem, cExtF, cTrig, cRows);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<D6Out> outInit(N);
    std::memset(outInit.data(), 0xCD, N * sizeof(D6Out));  // sentinel: removed = untouched
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npD6JointPrep.spv", "c1dPrePrep",
             { (uint32_t)sizeof(D6In), (uint32_t)sizeof(D6Out) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<D6Out>(dev, bOut, N);
    D6Out sentinel;
    std::memset(&sentinel, 0xCD, sizeof(sentinel));
    size_t mm = 0, unstable = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const D6Out& r = ref[i]; const D6Out& q = g[i];

        if (removed[i])
        {
            // the removed-joint gate writes NOTHING
            if (std::memcmp(&q, &sentinel, sizeof(D6Out)) != 0)
            {
                if (mm < 5) std::printf("MM %u removed case was written\n", i);
                ++mm;
            }
            continue;
        }

        // trig-FREE meta (review-175 hardening: never excludable, even under
        // trig): trig bits, ra/rb, invMassScale, flags/startIndex words
        const bool metaHard = r.counts.y == q.counts.y && r.nrfs.y == q.nrfs.y &&
                              r.nrfs.z == q.nrfs.z && r.nrfs.w == q.nrfs.w &&
                              eqF4FZ(r.invMass, q.invMass) && eqF4FZ(r.raW, q.raW) &&
                              eqF4FZ(r.rbW, q.rbW);
        if (!metaHard)
        {
            if (mm < 5)
                std::printf("MM %u c1d meta: tR %u tG %u raR (%g %g %g) raG (%g %g %g)\n", i,
                            r.counts.y, q.counts.y, r.raW.x, r.raW.y, r.raW.z, q.raW.x, q.raW.y,
                            q.raW.z);
            ++mm;
            continue;
        }

        // row-count structure: only THIS may fall under the capped
        // trig-unstable class (a trig deviation flipping a hard anglePair gate)
        if (!(r.counts.x == q.counts.x && r.nrfs.x == q.nrfs.x))
        {
            if (r.counts.y != 0)
            {
                ++unstable;
                continue;
            }
            if (mm < 5)
                std::printf("MM %u c1d struct: nR %u nG %u\n", i, r.nrfs.x, q.nrfs.x);
            ++mm;
            continue;
        }

        const bool trig = r.counts.y != 0;
        bool ok = true;
        for (uint32_t j = 0; j < r.counts.x * 5 && ok; ++j)
        {
            const F4& a = r.rows[j]; const F4& b = q.rows[j];
            if ((j % 5) == 4)
                ok = (trig ? (eqTol(a.x, b.x) && eqTol(a.y, b.y))
                           : (eqFZ(a.x, b.x) && eqFZ(a.y, b.y))) && eqBits(a.z, b.z);
            else
                ok = trig ? (eqTol(a.x, b.x) && eqTol(a.y, b.y) && eqTol(a.z, b.z) && eqTol(a.w, b.w))
                          : eqF4FZ(a, b);
        }
        if (!ok)
        {
            if (mm < 5) std::printf("MM %u c1d rows: n %u trig %u\n", i, r.counts.x, r.counts.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "c1dPrePrep: %zu mismatches", mm);
    PXG_CHECK_MSG(unstable <= 8, "c1dPrePrep: %zu trig-unstable cases (cap 8)", unstable);

    const bool cov = cRun > N / 2 && cDis > 100 && cRem > 100 && cExtF > N / 5 && cTrig > 100 &&
                     cRows > 2000;
    PXG_CHECK_MSG(cov, "c1dPrePrep: coverage thin (run %zu dis %zu rem %zu extF %zu trig %zu)",
                  cRun, cDis, cRem, cExtF, cTrig);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// constraint1DPrePrepLaunch on the ENGINE ABI (constraintBlockPrePrep.cu L1168)
// ---------------------------------------------------------------------------
// The full per-joint kernel driven exactly as PxgSolverCore::constraintPrePrep-
// Parallel launches it: REAL PxgD6JointData / PxgConstraintPrePrep /
// PxgSolverBodyData device arrays reached through a live PxgPrePrepDesc, params
// { PxgPrePrepDesc*, PxgSolverSharedDesc<IterativeSolveData>* } -> {8,8}, grid =
// ceil(nbD6Joints / 128), block 128, thread-per-joint.  The GPU builds its D6In
// from the real PxgD6JointData mirror (pinning every field offset end-to-end);
// the reference reuses the proven d6ref::solverPrep on the same fixture.  Rigid
// joints only (the arti branch is a documented deferred no-op).

using namespace physx;

// materialise a random D6In fixture into the REAL PxgD6JointData struct.  The
// two body poses (bA2w/bB2w) do NOT live here — they come from the solver-body
// data — so this stores only the joint-frame / limit / drive / bitmap fields.
static void materializeJointData(const D6In& s, PxgD6JointData& d)
{
    std::memset(&d, 0, sizeof(d));
    d.invMassScale.linear0 = 1.0f + 0.1f * s.linXP.x;   // arbitrary known values;
    d.invMassScale.angular0 = 1.0f + 0.1f * s.linXP.y;  // passed through verbatim
    d.invMassScale.linear1 = 1.0f + 0.1f * s.linYP.x;
    d.invMassScale.angular1 = 1.0f + 0.1f * s.linYP.y;

    d.c2b[0].q = PxQuat(s.c2bAq.x, s.c2bAq.y, s.c2bAq.z, s.c2bAq.w);
    d.c2b[0].p = PxVec3(s.c2bAp.x, s.c2bAp.y, s.c2bAp.z);
    d.c2b[1].q = PxQuat(s.c2bBq.x, s.c2bBq.y, s.c2bBq.z, s.c2bBq.w);
    d.c2b[1].p = PxVec3(s.c2bBp.x, s.c2bBp.y, s.c2bBp.z);

    auto setParams = [](PxgJointLimitParameters& p, const F4& lp) {
        p.restitution = lp.x; p.bounceThreshold = lp.y; p.stiffness = lp.z; p.damping = lp.w;
    };
    setParams(d.distanceLimit, s.distP);   d.distanceLimit.value = s.distV.x;
    setParams(d.linearLimitX, s.linXP);    d.linearLimitX.upper = s.linXV.x; d.linearLimitX.lower = s.linXV.y;
    setParams(d.linearLimitY, s.linYP);    d.linearLimitY.upper = s.linYV.x; d.linearLimitY.lower = s.linYV.y;
    setParams(d.linearLimitZ, s.linZP);    d.linearLimitZ.upper = s.linZV.x; d.linearLimitZ.lower = s.linZV.y;
    setParams(d.twistLimit, s.twistP);     d.twistLimit.upper = s.twistV.x; d.twistLimit.lower = s.twistV.y;
    setParams(d.swingLimit, s.swingP);     d.swingLimit.yAngle = s.swingV.x; d.swingLimit.zAngle = s.swingV.y;
    setParams(d.pyramidSwingLimit, s.pyrP);
    d.pyramidSwingLimit.yAngleMin = s.pyrV.x; d.pyramidSwingLimit.yAngleMax = s.pyrV.y;
    d.pyramidSwingLimit.zAngleMin = s.pyrV.z; d.pyramidSwingLimit.zAngleMax = s.pyrV.w;

    for (int k = 0; k < 6; ++k)
    {
        d.drive[k].stiffness = s.drives[k].x;
        d.drive[k].damping = s.drives[k].y;
        d.drive[k].forceLimit = s.drives[k].z;
        d.drive[k].flags = PxgD6JointDriveFlags(PxgD6JointDriveFlag::Enum(f2u(s.drives[k].w)));
    }

    d.drivePosition.q = PxQuat(s.drvPq.x, s.drvPq.y, s.drvPq.z, s.drvPq.w);
    d.drivePosition.p = PxVec3(s.drvPp.x, s.drvPp.y, s.drvPp.z);
    d.driveLinearVelocity = PxVec3(s.linVel.x, s.linVel.y, s.linVel.z);
    d.driveAngularVelocity = PxVec3(s.angVel.x, s.angVel.y, s.angVel.z);

    d.locked = f2u(s.bA2wp.w);
    d.limited = f2u(s.bB2wp.w);
    d.driving = f2u(s.drvPp.w);
    d.distanceMinDist = s.linVel.w;

    const uint32_t useBits = f2u(s.c2bBp.w);
    d.mUseDistanceLimit = (useBits & 1u) != 0;
    d.mUseNewLinearLimits = (useBits & 2u) != 0;
    d.mUseConeLimit = (useBits & 4u) != 0;
    d.mUsePyramidLimits = (useBits & 8u) != 0;
    d.angularDriveConfig = 0;
    for (int a = 0; a < 6; ++a) d.motion[a] = PxgD6Motion::eLOCKED;   // unread by the kernel
}

static std::vector<uint8_t> d6ShimLoader(const char* file) { return pxgtest::loadShaderBlob(file); }

PXG_TEST(npD6JointPrep_engineKernel)
{
    std::mt19937 rng(175701);
    const uint32_t N = 1500;                              // rigid joints
    const uint32_t F_EXT = 1u << 9, F_DISABLE = 1u << 12; // PxConstraintFlag
    const uint32_t PX_INVALID = 0xFFFFFFFFu;
    const uint32_t MAX_ROWS_PER_JOINT = 20;               // Dy::MAX_CONSTRAINT_ROWS

    // static anchor (solver body 0, referenced by every PX_INVALID node)
    const PxTransform anchor(PxVec3(0.3f, -0.7f, 1.1f), PxQuat(0.1f, 0.2f, 0.3f, 0.927f).getNormalized());

    const uint32_t numNodes = 2 * N + 2;    // node 0 unused; joints use 1..2N
    const uint32_t numBodies = numNodes;    // identity solverBodyIndices

    std::vector<PxgD6JointData> jointData(N);
    std::vector<PxgConstraintPrePrep> prePrep(N);
    std::vector<PxgSolverBodyData> bodyData(numBodies);
    std::memset(bodyData.data(), 0, bodyData.size() * sizeof(PxgSolverBodyData));
    bodyData[0].body2World = PxAlignedTransform(anchor);
    std::vector<uint32_t> solverBodyIndices(numNodes);
    for (uint32_t k = 0; k < numNodes; ++k) solverBodyIndices[k] = k;

    // references
    std::vector<D6Out> ref(N);
    std::vector<PxgConstraintData> refCData(N);
    std::vector<uint8_t> removed(N);
    std::vector<float> refUseExt(N);
    size_t cRem = 0, cDis = 0, cExtF = 0, cRun = 0, cTrig = 0, cRows = 0, cStatA = 0, cStatB = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        D6In s{};
        genFixture(rng, i, s);

        // node topology: mostly dynamic-dynamic; a few static / removed
        const bool rem = (i % 41 == 7);
        const bool staticA = !rem && (i % 13 == 3);
        const bool staticB = !rem && !staticA && (i % 17 == 5);

        const uint32_t nodeA = (rem || staticA) ? PX_INVALID : (2 * i + 1);
        const uint32_t nodeB = (rem || staticB) ? PX_INVALID : (2 * i + 2);

        // place body poses: PX_INVALID -> solver body 0 (anchor); the fixture's
        // bA2w/bB2w must then equal the anchor so the reference sees the same pose.
        if (nodeA == PX_INVALID)
        {
            s.bA2wq = F4{ anchor.q.x, anchor.q.y, anchor.q.z, anchor.q.w };
            s.bA2wp = F4{ anchor.p.x, anchor.p.y, anchor.p.z, s.bA2wp.w };
            if (staticA) ++cStatA;
        }
        else
            bodyData[solverBodyIndices[nodeA]].body2World =
                PxAlignedTransform(PxTransform(PxVec3(s.bA2wp.x, s.bA2wp.y, s.bA2wp.z),
                                               PxQuat(s.bA2wq.x, s.bA2wq.y, s.bA2wq.z, s.bA2wq.w)));
        if (nodeB == PX_INVALID)
        {
            s.bB2wq = F4{ anchor.q.x, anchor.q.y, anchor.q.z, anchor.q.w };
            s.bB2wp = F4{ anchor.p.x, anchor.p.y, anchor.p.z, s.bB2wp.w };
            if (staticB) ++cStatB;
        }
        else
            bodyData[solverBodyIndices[nodeB]].body2World =
                PxAlignedTransform(PxTransform(PxVec3(s.bB2wp.x, s.bB2wp.y, s.bB2wp.z),
                                               PxQuat(s.bB2wq.x, s.bB2wq.y, s.bB2wq.z, s.bB2wq.w)));

        materializeJointData(s, jointData[i]);

        uint32_t ppFlags = ((rng() % 2) ? (1u << 3) : 0u) | ((rng() % 2) ? (1u << 5) : 0u);
        if (rng() % 3 == 0) ppFlags |= F_EXT;
        if (i % 5 == 1) ppFlags |= F_DISABLE;

        prePrep[i].mNodeIndexA = PxNodeIndex(nodeA);
        prePrep[i].mNodeIndexB = PxNodeIndex(nodeB);
        prePrep[i].mFlags = ppFlags;
        prePrep[i].mLinBreakForce = rfS(rng, 0.f, 1e5f);
        prePrep[i].mAngBreakForce = rfS(rng, 0.f, 1e5f);

        removed[i] = rem ? 1 : 0;
        refUseExt[i] = (ppFlags & F_EXT) ? 1.f : 0.f;

        // reference
        D6Out& o = ref[i];
        std::memset(&o, 0, sizeof(o));
        PxgConstraintData& cd = refCData[i];
        std::memset(&cd, 0, sizeof(cd));
        if (rem) { ++cRem; continue; }

        uint32_t numRows = 0;
        V3 ra{ 0, 0, 0 }, rb{ 0, 0, 0 };
        if (!(ppFlags & F_DISABLE))
        {
            D6In t = s;
            t.c2bAp.w = asF((ppFlags & F_EXT) ? 1u : 0u);   // engine derives useExt from prePrep
            d6ref::solverPrep(t, o);
            numRows = o.counts.x;
            ra = V3{ o.cAoutW.x - s.bA2wp.x, o.cAoutW.y - s.bA2wp.y, o.cAoutW.z - s.bA2wp.z };
            rb = V3{ o.cBoutW.x - s.bB2wp.x, o.cBoutW.y - s.bB2wp.y, o.cBoutW.z - s.bB2wp.z };
            ++cRun;
            if (o.counts.y) ++cTrig;
            cRows += numRows;
        }
        else
            ++cDis;

        cd.mInvMassScale = jointData[i].invMassScale;
        cd.mRaWorld_linBreakForceW = make_float4(ra.x, ra.y, ra.z, prePrep[i].mLinBreakForce);
        cd.mRbWorld_angBreakForceW = make_float4(rb.x, rb.y, rb.z, prePrep[i].mAngBreakForce);
        cd.mNumRows_Flags_StartIndex = make_uint4(numRows, ppFlags, i * MAX_ROWS_PER_JOINT, 0);
        if (ppFlags & F_EXT) ++cExtF;
    }

    std::printf("[ info ] eng cov: run %zu dis %zu rem %zu extF %zu trig %zu rows %zu statA %zu statB %zu\n",
                cRun, cDis, cRem, cExtF, cTrig, cRows, cStatA, cStatB);

    PxgSlangComputeDevice* dev = testDevice();
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;
    const physx::gpuslang::PxgKernelDef manifest[] = {
        { "npD6JointPrep.spv", "constraint1DPrePrepLaunch", 16u, 0 },
    };
    physx::gpuslang::PxgCudaShim shim(dev, manifest, 1, &d6ShimLoader);

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data) shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    // outputs: sentinel-filled so removed joints prove "no write"
    std::vector<PxgConstraintData> cDataInit(N);
    std::memset(cDataInit.data(), 0xCD, cDataInit.size() * sizeof(PxgConstraintData));
    std::vector<Px1DConstraint> rowInit(N * MAX_ROWS_PER_JOINT);
    std::memset(rowInit.data(), 0xCD, rowInit.size() * sizeof(Px1DConstraint));

    const PxgDevicePtr dJoint = alloc(jointData.data(), jointData.size() * sizeof(PxgD6JointData));
    const PxgDevicePtr dPre = alloc(prePrep.data(), prePrep.size() * sizeof(PxgConstraintPrePrep));
    const PxgDevicePtr dBody = alloc(bodyData.data(), bodyData.size() * sizeof(PxgSolverBodyData));
    const PxgDevicePtr dSBI = alloc(solverBodyIndices.data(), solverBodyIndices.size() * 4);
    const PxgDevicePtr dCData = alloc(cDataInit.data(), cDataInit.size() * sizeof(PxgConstraintData));
    const PxgDevicePtr dRows = alloc(rowInit.data(), rowInit.size() * sizeof(Px1DConstraint));
    if (!dJoint) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgPrePrepDesc desc;
    std::memset(&desc, 0xCD, sizeof(desc));
    desc.rigidJointData = reinterpret_cast<PxgD6JointData*>(dJoint);
    desc.rigidConstraintPrePrep = reinterpret_cast<PxgConstraintPrePrep*>(dPre);
    desc.constraintData = reinterpret_cast<PxgConstraintData*>(dCData);
    desc.constraintRows = reinterpret_cast<Px1DConstraint*>(dRows);
    desc.artiJointData = NULL;
    desc.artiConstraintPrePrep = NULL;
    desc.artiConstraintData = NULL;
    desc.artiConstraintRows = NULL;
    desc.mSolverBodyData = reinterpret_cast<PxgSolverBodyData*>(dBody);
    desc.solverBodyIndices = reinterpret_cast<PxU32*>(dSBI);
    desc.nbD6RigidJoints = N;
    desc.nbD6ArtiJoints = 0;
    desc.nbTotalArtiJoints = 0;

    PxgSolverSharedDesc<IterativeSolveData> sh;
    std::memset(&sh, 0xCD, sizeof(sh));
    sh.mBodySimBufferDeviceData = NULL;

    const PxgDevicePtr dDesc = alloc(&desc, sizeof(desc));
    const PxgDevicePtr dSh = alloc(&sh, sizeof(sh));

    const uint32_t nbBlocks = (N + 127) / 128;   // CONSTRAINT_PREPREP_BLOCK = 128
    struct { uint64_t pre, sh; } prm = { dDesc, dSh };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch constraint1DPrePrepLaunch");
    shim.streamSynchronize();

    std::vector<PxgConstraintData> gCData(N);
    std::vector<Px1DConstraint> gRows(N * MAX_ROWS_PER_JOINT);
    shim.memcpyDtoH(gCData.data(), dCData, gCData.size() * sizeof(PxgConstraintData));
    shim.memcpyDtoH(gRows.data(), dRows, gRows.size() * sizeof(Px1DConstraint));

    PxgConstraintData sentinel;
    std::memset(&sentinel, 0xCD, sizeof(sentinel));

    // Px1DConstraint field views for the row comparison
    auto rowLin0 = [](const Px1DConstraint& c) { return F4{ c.linear0.x, c.linear0.y, c.linear0.z, c.geometricError }; };
    auto rowAng0 = [](const Px1DConstraint& c) { return F4{ c.angular0.x, c.angular0.y, c.angular0.z, c.velocityTarget }; };
    auto rowLin1 = [](const Px1DConstraint& c) { return F4{ c.linear1.x, c.linear1.y, c.linear1.z, c.minImpulse }; };
    auto rowAng1 = [](const Px1DConstraint& c) { return F4{ c.angular1.x, c.angular1.y, c.angular1.z, c.maxImpulse }; };

    size_t mm = 0, unstable = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (removed[i])
        {
            if (std::memcmp(&gCData[i], &sentinel, sizeof(PxgConstraintData)) != 0)
            {
                if (mm < 5) std::printf("MM %u removed case was written\n", i);
                ++mm;
            }
            continue;
        }

        const PxgConstraintData& r = refCData[i];
        const PxgConstraintData& q = gCData[i];

        // trig-free meta: invMassScale, ra/rb, flags/startIndex (never excludable)
        const F4* rInv = reinterpret_cast<const F4*>(&r.mInvMassScale);
        const F4* qInv = reinterpret_cast<const F4*>(&q.mInvMassScale);
        const F4 rRa{ r.mRaWorld_linBreakForceW.x, r.mRaWorld_linBreakForceW.y, r.mRaWorld_linBreakForceW.z, r.mRaWorld_linBreakForceW.w };
        const F4 qRa{ q.mRaWorld_linBreakForceW.x, q.mRaWorld_linBreakForceW.y, q.mRaWorld_linBreakForceW.z, q.mRaWorld_linBreakForceW.w };
        const bool metaHard = eqF4FZ(*rInv, *qInv) && eqF4FZ(rRa, qRa) &&
                              eqFZ(r.mRbWorld_angBreakForceW.x, q.mRbWorld_angBreakForceW.x) &&
                              eqFZ(r.mRbWorld_angBreakForceW.y, q.mRbWorld_angBreakForceW.y) &&
                              eqFZ(r.mRbWorld_angBreakForceW.z, q.mRbWorld_angBreakForceW.z) &&
                              eqFZ(r.mRbWorld_angBreakForceW.w, q.mRbWorld_angBreakForceW.w) &&
                              r.mNumRows_Flags_StartIndex.y == q.mNumRows_Flags_StartIndex.y &&
                              r.mNumRows_Flags_StartIndex.z == q.mNumRows_Flags_StartIndex.z &&
                              r.mNumRows_Flags_StartIndex.w == q.mNumRows_Flags_StartIndex.w;
        if (!metaHard)
        {
            if (mm < 5)
                std::printf("MM %u eng meta: raR (%g %g %g) raG (%g %g %g) flagsR %u flagsG %u\n", i,
                            r.mRaWorld_linBreakForceW.x, r.mRaWorld_linBreakForceW.y,
                            r.mRaWorld_linBreakForceW.z, q.mRaWorld_linBreakForceW.x,
                            q.mRaWorld_linBreakForceW.y, q.mRaWorld_linBreakForceW.z,
                            r.mNumRows_Flags_StartIndex.y, q.mNumRows_Flags_StartIndex.y);
            ++mm;
            continue;
        }

        const uint32_t rN = r.mNumRows_Flags_StartIndex.x, qN = q.mNumRows_Flags_StartIndex.x;
        const bool trig = ref[i].counts.y != 0;
        if (rN != qN)
        {
            if (trig) { ++unstable; continue; }   // trig-flipped hard anglePair gate (capped)
            if (mm < 5) std::printf("MM %u eng numRows: R %u G %u\n", i, rN, qN);
            ++mm;
            continue;
        }

        bool ok = true;
        for (uint32_t j = 0; j < rN && ok; ++j)
        {
            // reference row j comes from the flat D6Out image (5 float4s / row)
            const F4 rl0 = ref[i].rows[j * 5 + 0], ra0 = ref[i].rows[j * 5 + 1],
                     rl1 = ref[i].rows[j * 5 + 2], ra1 = ref[i].rows[j * 5 + 3],
                     rmod = ref[i].rows[j * 5 + 4];
            const Px1DConstraint& g = gRows[i * MAX_ROWS_PER_JOINT + j];
            const F4 gl0 = rowLin0(g), ga0 = rowAng0(g), gl1 = rowLin1(g), ga1 = rowAng1(g);
            uint32_t gpackFlags; std::memcpy(&gpackFlags, &g.flags, 4);
            const uint32_t rpackFlags = f2u(rmod.z);

            auto cmp4 = [&](const F4& A, const F4& B) {
                return trig ? (eqTol(A.x, B.x) && eqTol(A.y, B.y) && eqTol(A.z, B.z) && eqTol(A.w, B.w))
                            : eqF4FZ(A, B);
            };
            ok = cmp4(rl0, gl0) && cmp4(ra0, ga0) && cmp4(rl1, gl1) && cmp4(ra1, ga1);
            const bool modOk = trig ? (eqTol(rmod.x, g.mods.spring.stiffness) && eqTol(rmod.y, g.mods.spring.damping))
                                    : (eqFZ(rmod.x, g.mods.spring.stiffness) && eqFZ(rmod.y, g.mods.spring.damping));
            ok = ok && modOk && (gpackFlags == rpackFlags);
        }
        if (!ok)
        {
            if (mm < 5) std::printf("MM %u eng rows: n %u trig %u\n", i, rN, ref[i].counts.y);
            ++mm;
        }
    }

    PXG_CHECK_MSG(mm == 0, "constraint1DPrePrepLaunch: %zu mismatches", mm);
    PXG_CHECK_MSG(unstable <= 8, "constraint1DPrePrepLaunch: %zu trig-unstable cases (cap 8)", unstable);
    if (unstable)
        std::printf("[ info ] eng trig-unstable excluded: %zu of %u\n", unstable, N);

    const bool cov = cRun > N / 2 && cDis > 100 && cRem > 20 && cExtF > N / 6 && cTrig > 100 &&
                     cRows > 1500 && cStatA > 20 && cStatB > 20;
    PXG_CHECK_MSG(cov, "constraint1DPrePrepLaunch: coverage thin (run %zu dis %zu rem %zu extF %zu "
                       "trig %zu statA %zu statB %zu)",
                  cRun, cDis, cRem, cExtF, cTrig, cStatA, cStatB);

    for (PxgDevicePtr p : { dJoint, dPre, dBody, dSBI, dCData, dRows, dDesc, dSh })
        shim.memFree(p);
}

} // namespace
