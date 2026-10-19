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

// npD6JointPrep.slang
//
// Slang port of D6JointSolverPrep from
// gpusolver/src/CUDA/constraintBlockPrePrep.cu (L516-737) together with its
// full helper stack:
//   - PxgConstraintHelper (PxgConstraintHelper.h): computeJointFrames,
//     linear/angular row setup, addLimit/addDrive, linearLimit/angularLimit,
//     anglePair, angularHard, prepareLockedAxes (incl. computeJacobianAxes);
//   - the swing/twist limit helpers (constraintBlockPrePrep.cu L446-490):
//     setupConeSwingLimits (Cm::ConeLimitHelperTanLess of CmConeLimitHelper.h
//     with PxEllipseClamp of PxMathUtils.h), setupPyramidSwingLimits
//     (computeSwingAngle), setupSingleSwingLimit, setupDualConeSwingLimits
//     (PxAsin), setupLinearLimit, applyNeighborhoodOperator, computePhi,
//     PxSeparateSwingTwist (PxMathUtils.h L272-276);
//   - the exact PxQuat forms (rotate, operator*, getConjugate,
//     getBasisVector0/1/2, getNormalized, normalize, getAngle = acos(w)*2)
//     and PxMat33-from-quat columns.
// Trig class (deterministically flagged, tolerance-compared on the CPU
// side): atan2 (computeSwingAngle / cone helper swing angles), acos
// (computePhi via PxQuat::getAngle), asin (dual-cone limits, input clamped
// per PxAsin), tan (cone helper rotation vector).  Everything else
// verifies bitwise (modulo the documented +-0 class); this shader is in
// PXG_NOCONTRACT_SHADERS, so all float arithmetic carries NoContraction.

// P5 Step 5 (BDA, ENGINE launch ABI): the engine entry
// constraint1DPrePrepLaunch (constraintBlockPrePrep.cu L1168) is appended at
// the end of this file.  It reuses the proven d6SolveQ core verbatim, feeding
// it a D6In assembled from the REAL PxgD6JointData mirror + the two solver-body
// poses reached through the BDA desc.  These imports supply the PxgPrePrepDesc /
// PxgSolverBodyData / PxNodeIndex mirrors (proven by testPrepDescLayout /
// pxgSlangCommon).
import pxgSlangPrepDesc;
import pxgSlangCommon;
import pxgSlangArticulationDesc; // PXG_ARTI_STRIDE, PXG_ARTI_LINK_BODY2WORLDS, artiPtrField

static const float FLT_MAXQ = 3.402823466e+38f;
static const uint MAX_ROWS = 24u;

// Px1DConstraintFlag
static const uint E_SPRING = 1u;
static const uint E_ACCELERATION_SPRING = 2u;
static const uint E_RESTITUTION = 4u;
static const uint E_KEEPBIAS = 8u;
static const uint E_OUTPUT_FORCE = 16u;
static const uint E_HAS_DRIVE_LIMIT = 32u;
static const uint E_ANGULAR_CONSTRAINT = 64u;

// PxConstraintSolveHint
static const uint H_NONE = 0u;
static const uint H_SLERP_SPRING = 258u;
static const uint H_EQUALITY = 2048u;
static const uint H_INEQUALITY = 2049u;

// PxgD6Axis bit positions (locked/limited bitmaps)
static const uint AX_X = 0u, AX_Y = 1u, AX_Z = 2u, AX_TWIST = 3u, AX_SWING1 = 4u, AX_SWING2 = 5u;
// PxgD6Drive bit positions (driving bitmap).  5.12 removed the legacy eSWING
// drive: eTWIST = 3, eSWING1 = 4, eSWING2 = 5, eSLERP = 6 (5.6.1: eSWING = 3,
// eTWIST = 4, eSLERP = 5, eSWING1 = 6, eSWING2 = 7).
static const uint DR_X = 0u, DR_Y = 1u, DR_Z = 2u, DR_TWIST = 3u, DR_SWING1 = 4u, DR_SWING2 = 5u,
                  DR_SLERP = 6u;
// drive data entries (constraintBlockPrePrep.cu gDrive*DataIndex): the slerp
// drive shares the eSWING2 entry (5.6.1: swing1 shared the eSWING entry 3,
// swing2 the eSLERP entry 5, twist was 4)
static const uint DRIVE_DATA_TWIST = 3u, DRIVE_DATA_SWING1 = 4u, DRIVE_DATA_SWING2 = 5u,
                  DRIVE_DATA_SLERP = DRIVE_DATA_SWING2;

struct D6In
{
    float4 c2bAq; float4 c2bAp;   // c2bAp.w = asuint(useExtendedLimits 0/1)
    float4 c2bBq; float4 c2bBp;   // c2bBp.w = asuint(use bits: dist|newLin<<1|cone<<2|pyr<<3)
    float4 bA2wq; float4 bA2wp;   // bA2wp.w = asuint(locked)
    float4 bB2wq; float4 bB2wp;   // bB2wp.w = asuint(limited)
    float4 drvPq; float4 drvPp;   // drvPp.w = asuint(driving)
    float4 linVel;                // driveLinearVelocity, w = distanceMinDist
    float4 angVel;                // driveAngularVelocity
    float4 drives[6];             // per drive: stiffness, damping, forceLimit, asuint(flags)
    float4 distP;  float4 distV;  // distanceLimit params (rest, bounce, stiff, damp); x = value
    float4 linXP;  float4 linXV;  // linearLimitX params; x = upper, y = lower
    float4 linYP;  float4 linYV;
    float4 linZP;  float4 linZV;
    float4 twistP; float4 twistV; // twistLimit params; x = upper, y = lower
    float4 swingP; float4 swingV; // swingLimit params; x = yAngle, y = zAngle
    float4 pyrP;   float4 pyrV;   // pyramid params; x = yMin, y = yMax, z = zMin, w = zMax
}

struct D6Out
{
    uint4 counts;                 // x = numConstraints, y = trig bits
    float4 cAoutW;                // cA2wOut (ra + bA2w.p)
    float4 cBoutW;                // cB2wOut (cB2w.p)
    uint4 nrfs;                   // c1dPrePrep: mNumRows_Flags_StartIndex image
    float4 invMass;               // c1dPrePrep: mInvMassScale passthrough
    float4 raW;                   // c1dPrePrep: mRaWorld_linBreakForceW
    float4 rbW;                   // c1dPrePrep: mRbWorld_angBreakForceW
    float4 rows[120];             // MAX_ROWS x 5 float4s per Px1DConstraint:
                                  //  (linear0, geometricError) (angular0, velocityTarget)
                                  //  (linear1, minImpulse) (angular1, maxImpulse)
                                  //  (mods.x, mods.y, asfloat(flags | solveHint<<16), 0)
}

struct D6Params { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<D6Params> gParams;

RWStructuredBuffer<D6In> gIn;    // binding 0
RWStructuredBuffer<D6Out> gOut;  // binding 1

float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float dot4Q(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float magSqQ(float3 v) { return dot3Q(v, v); }
float magQ(float3 v) { return sqrt(magSqQ(v)); }

// P9.86: PxMax<float> is intrinsics::selectMax (PxMath.h:81-85, PxUnixMathIntrinsics.h:122):
// b wins a +-0 tie and a NaN compare -- not the generic template's `a < b ? b : a`
float pxMaxQ(float a, float b) { return (a > b) ? a : b; }

// PxVec3::getNormalized: m > 0 ? v * PxRecipSqrt(m) : 0
float3 getNormalizedQ(float3 v)
{
    const float m = magSqQ(v);
    if (m > 0.0f)
        return v * (1.0f / sqrt(m));
    return float3(0, 0, 0);
}

// PxQuat::rotate (PxQuat.h L286-295)
float3 quatRotQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// PxQuat::operator* (PxQuat.h L388-392)
float4 quatMulQ(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
                  a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
                  a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float4 quatConjQ(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }

// PxQuat::getBasisVector0/1/2 (PxQuat.h L260-281)
float3 basisVector0Q(float4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2);
}
float3 basisVector1Q(float4 q)
{
    const float y2 = q.y * 2.0f, w2 = q.w * 2.0f;
    return float3((-q.z * w2) + q.x * y2, (q.w * w2) - 1.0f + q.y * y2, (q.x * w2) + q.z * y2);
}
float3 basisVector2Q(float4 q)
{
    const float z2 = q.z * 2.0f, w2 = q.w * 2.0f;
    return float3((q.y * w2) + q.x * z2, (-q.x * w2) + q.y * z2, (q.w * w2) - 1.0f + q.z * z2);
}

// columns of PxMat33(quat) (PxMat33.h L136-163)
// IEEE `0 - a` per component: +0 for either zero, -a otherwise (P9.86, see the
// errorVector in prepareLockedAxes).
float3 d6ZeroMinus(float3 a)
{
    return float3((a.x == 0.0f) ? 0.0f : -a.x, (a.y == 0.0f) ? 0.0f : -a.y, (a.z == 0.0f) ? 0.0f : -a.z);
}

void quatToMat33(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
}

// computeJacobianAxes (PxgConstraintHelper.h L43-58)
void computeJacobianAxesQ(float4 qa, float4 qb, out float3 row0, out float3 row1, out float3 row2)
{
    const float wa = qa.w, wb = qb.w;
    const float3 va = qa.xyz, vb = qb.xyz;
    const float3 c = vb * wa + va * wb;
    const float d = wa * wb - dot3Q(va, vb);
    row0 = (va * vb.x + vb * va.x + float3(d, c.z, -c.y)) * 0.5f;
    row1 = (va * vb.y + vb * va.y + float3(-c.z, d, c.x)) * 0.5f;
    row2 = (va * vb.z + vb * va.z + float3(c.y, -c.x, d)) * 0.5f;
}

// PxSeparateSwingTwist (PxMathUtils.h L272-276); twist getNormalized includes
// the zero y/z terms of the 4-component magnitude
void separateSwingTwistQ(float4 q, out float4 swing, out float4 twist)
{
    if (q.x != 0.0f)
    {
        const float s = 1.0f / sqrt(q.x * q.x + 0.0f * 0.0f + 0.0f * 0.0f + q.w * q.w);
        twist = float4(q.x * s, 0.0f * s, 0.0f * s, q.w * s);
    }
    else
        twist = float4(0, 0, 0, 1);
    swing = quatMulQ(q, quatConjQ(twist));
}

// computeSwingAngle (constraintBlockPrePrep.cu L102-105)
float computeSwingAngleQ(float swingYZ, float swingW, inout uint trig)
{
    trig |= 1u;
    return 4.0f * atan2(swingYZ, 1.0f + swingW);
}

// computePhi (constraintBlockPrePrep.cu L107-116): PxQuat::normalize (guarded)
// then getAngle = PxAcos(w) * 2 with the PxAcos clamp
float computePhiQ(float4 q, inout uint trig)
{
    float4 t = q;
    const float mag = sqrt(dot4Q(t, t));
    if (mag != 0.0f)
    {
        const float imag = 1.0f / mag;
        t = float4(t.x * imag, t.y * imag, t.z * imag, t.w * imag);
    }
    trig |= 2u;
    float angle = acos(clamp(t.w, -1.0f, 1.0f)) * 2.0f;
    if (t.x < 0.0f)
        angle = -angle;
    return angle;
}

// PxEllipseClamp (PxMathUtils.h L204-262)
float3 ellipseClampQ(float3 pt, float3 radii)
{
    float3 q = float3(0.0f, abs(pt.y), abs(pt.z));
    const float tinyEps = 1e-6f;
    if (radii.y >= radii.z)
    {
        if (q.z < tinyEps)
            return float3(0.0f, pt.y > 0.0f ? radii.y : -radii.y, 0.0f);
    }
    else
    {
        if (q.y < tinyEps)
            return float3(0.0f, 0.0f, pt.z > 0.0f ? radii.z : -radii.z);
    }

    float3 denom = float3(0, 0, 0);
    const float3 e2 = radii * radii;
    const float3 eq = radii * q;

    float t = pxMaxQ(eq.y - e2.y, eq.z - e2.z);

    for (uint i = 0u; i < 20u; ++i)
    {
        denom = float3(0.0f, 1.0f / (t + e2.y), 1.0f / (t + e2.z));
        const float3 denom2 = eq * denom;
        const float3 fv = denom2 * denom2;
        const float f = fv.y + fv.z - 1.0f;
        if (f < 1e-4f)
            return e2 * pt * denom;
        const float df = dot3Q(fv, denom) * -2.0f;
        t = t - f / df;
    }
    const float3 r = e2 * pt * denom;
    const float ry = r.y / radii.y, rz = r.z / radii.z;
    return r * (1.0f / sqrt(ry * ry + rz * rz));
}

// Cm::computeAxisAndError (CmConeLimitHelper.h L50-69)
float coneAxisAndErrorQ(float3 r, float3 d, float3 twistAxis, out float3 axis)
{
    const float r2 = dot3Q(r, r), a = 1.0f - r2, b = 1.0f / (1.0f + r2), b2 = b * b;
    const float v1 = 2.0f * a * b2;
    const float3 v2 = float3(a, 2.0f * r.z, -2.0f * r.y);
    const float3 coneLine = v1 * v2 - float3(1.0f, 0.0f, 0.0f);

    const float rd = dot3Q(r, d);
    const float dv1 = -4.0f * rd * (3.0f - r2) * b2 * b;
    const float3 dv2 = float3(-2.0f * rd, 2.0f * d.z, -2.0f * d.y);

    const float3 coneNormal = v1 * dv2 + dv1 * v2;

    // PxVec3::operator/ = reciprocal multiply
    axis = crossQ(coneLine, coneNormal) * (1.0f / magQ(coneNormal));
    return dot3Q(crossQ(coneLine, axis), twistAxis);
}

// Cm::ConeLimitHelperTanLess::getLimit (CmConeLimitHelper.h L163-178)
void coneGetLimitQ(float yMax, float zMax, float4 swing, out float3 axis, out float error,
                   inout uint trig)
{
    trig |= 8u;
    const float3 twistAxis = basisVector0Q(swing);
    const float3 swingAngle = float3(0.0f, 4.0f * atan2(swing.y, 1.0f + swing.w),
                                     4.0f * atan2(swing.z, 1.0f + swing.w));

    const float3 clamped = ellipseClampQ(swingAngle, float3(0.0f, yMax, zMax));
    const float3 normal = float3(0.0f, clamped.y / (yMax * yMax), clamped.z / (zMax * zMax));

    const float3 r = float3(0.0f, tan(clamped.y / 4.0f), tan(clamped.z / 4.0f));
    const float3 d = float3(0.0f, normal.y, normal.z);

    error = coneAxisAndErrorQ(r, d, twistAxis, axis);
}

// ---- Px1DConstraint row + PxgConstraintHelper ----

struct C1D
{
    float3 lin0; float geo;
    float3 ang0; float vt;
    float3 lin1; float minI;
    float3 ang1; float maxI;
    float m0; float m1;      // mods.spring.{stiffness,damping} == mods.bounce.{restitution,velocityThreshold}
    uint flags; uint hint;
};

struct RowBuf
{
    C1D r[MAX_ROWS];
    uint n;
};

// PxgConstraintHelper::linear (mRa/mRb form, PxgConstraintHelper.h L207-225)
C1D chLinear(float3 mRa, float3 mRb, float3 axis, float posErr, uint hint)
{
    C1D c;
    c.hint = hint;
    c.lin0 = axis; c.ang0 = crossQ(mRa, axis);
    c.lin1 = axis; c.ang1 = crossQ(mRb, axis);
    c.geo = posErr;
    c.flags = 0u;
    c.minI = -FLT_MAXQ; c.maxI = FLT_MAXQ;
    c.m1 = 0.0f; c.m0 = 0.0f;
    c.vt = 0.0f;
    return c;
}

// the explicit ra/rb overload used by prepareLockedAxes (L227-246)
C1D chLinearRa(float3 ra, float3 rb, float3 axis, float posErr, uint hint, uint flags)
{
    C1D c;
    c.hint = hint;
    c.lin0 = axis; c.ang0 = crossQ(ra, axis);
    c.lin1 = axis; c.ang1 = crossQ(rb, axis);
    c.geo = posErr;
    c.flags = flags;
    c.minI = -FLT_MAXQ; c.maxI = FLT_MAXQ;
    c.m1 = 0.0f; c.m0 = 0.0f;
    c.vt = 0.0f;
    return c;
}

// PxgConstraintHelper::angular (L248-262)
C1D chAngular(float3 axis, float posErr, uint hint, uint flags)
{
    C1D c;
    c.hint = hint;
    c.lin0 = float3(0, 0, 0); c.ang0 = axis;
    c.lin1 = float3(0, 0, 0); c.ang1 = axis;
    c.geo = posErr;
    c.flags = flags | E_ANGULAR_CONSTRAINT;
    c.minI = -FLT_MAXQ; c.maxI = FLT_MAXQ;
    c.m1 = 0.0f; c.m0 = 0.0f;
    c.vt = 0.0f;
    return c;
}

bool limitIsSoft(float4 lp) { return lp.w > 0.0f || lp.z > 0.0f; }  // damping>0 || stiffness>0

// PxgConstraintHelper::addLimit (L264-287); lp = (restitution, bounceThreshold, stiffness, damping)
void chAddLimit(inout C1D c, float4 lp)
{
    uint flags = c.flags | E_OUTPUT_FORCE;
    if (limitIsSoft(lp))
    {
        flags |= E_SPRING;
        c.m0 = lp.z;   // stiffness
        c.m1 = lp.w;   // damping
    }
    else
    {
        c.hint = H_INEQUALITY;
        c.m0 = lp.x;   // bounce.restitution
        c.m1 = lp.y;   // bounce.velocityThreshold
        // 5.12: minImpulse = 0 only for hard (inequality) limits; 5.6.1 set it
        // after the branch, so soft limits lost their -FLT_MAX lower bound
        c.minI = 0.0f;
        if (c.geo > 0.0f)
            flags |= E_KEEPBIAS;
        if (lp.x > 0.0f)
            flags |= E_RESTITUTION;
    }
    c.flags = flags;
}

// PxgConstraintHelper::addDrive (L289-310); drv = (stiffness, damping, forceLimit, asuint(flags))
void chAddDrive(inout C1D c, float velTarget, float4 drv)
{
    c.vt = velTarget;
    uint flags = c.flags | E_SPRING | E_HAS_DRIVE_LIMIT;
    const uint df = asuint(drv.w);
    if ((df & 1u) != 0u)   // PxgD6JointDriveFlag::eACCELERATION
        flags |= E_ACCELERATION_SPRING;
    if ((df & 2u) != 0u)   // PxgD6JointDriveFlag::eOUTPUT_FORCE
        flags |= E_OUTPUT_FORCE;
    c.flags = flags;
    c.m0 = drv.x;
    c.m1 = drv.y;
    c.minI = -drv.z;
    c.maxI = drv.z;
}

// linearLimit (L102-111)
void chLinearLimit(inout RowBuf buf, float3 mRa, float3 mRb, float3 axis, float ordinate,
                   float limitValue, float4 lp)
{
    if (!limitIsSoft(lp) || ordinate > limitValue)
    {
        C1D c = chLinear(mRa, mRb, axis, limitValue - ordinate, H_NONE);
        chAddLimit(c, lp);
        buf.r[buf.n] = c; buf.n += 1u;
    }
}

// angularLimit single-row form (L127-131)
void chAngularLimit(inout RowBuf buf, float3 axis, float error, float4 lp)
{
    C1D c = chAngular(axis, error, H_NONE, 0u);
    chAddLimit(c, lp);
    buf.r[buf.n] = c; buf.n += 1u;
}

// anglePair (L133-144)
void chAnglePair(inout RowBuf buf, float angle, float lower, float upper, float3 axis, float4 lp)
{
    const bool softLimit = limitIsSoft(lp);
    if (!softLimit || angle < lower)
        chAngularLimit(buf, -axis, -(lower - angle), lp);
    if (!softLimit || angle > upper)
        chAngularLimit(buf, axis, upper - angle, lp);
}

// ---- D6JointSolverPrep driver (constraintBlockPrePrep.cu L516-737) ----
// raOut is the internal ra of prepareLockedAxes (cA2wOut = raOut + bA2w.p);
// cB2wpOut = cB2w.p

void d6SolveQ(const D6In s, uint useExtendedLimits, inout RowBuf buf, inout uint trig,
              out float3 raOut, out float3 cB2wpOut)
{
    const uint useBits = asuint(s.c2bBp.w);
    const bool useDistanceLimit = (useBits & 1u) != 0u;
    const bool useNewLinearLimits = (useBits & 2u) != 0u;
    const bool useConeLimit = (useBits & 4u) != 0u;
    const bool usePyramidLimits = (useBits & 8u) != 0u;

    uint locked = asuint(s.bA2wp.w);
    const uint limited = asuint(s.bB2wp.w);
    const uint driving = asuint(s.drvPp.w);

    // computeJointFrames: cX2w = bX2w.transform(c2b[X])
    float4 cA2wq = quatMulQ(s.bA2wq, s.c2bAq);
    float3 cA2wp = quatRotQ(s.bA2wq, s.c2bAp.xyz) + s.bA2wp.xyz;
    float4 cB2wq = quatMulQ(s.bB2wq, s.c2bBq);
    float3 cB2wp = quatRotQ(s.bB2wq, s.c2bBp.xyz) + s.bB2wp.xyz;

    const float3 mRa = cB2wp - s.bA2wp.xyz;
    const float3 mRb = cB2wp - s.bB2wp.xyz;

    const uint SWING1_FLAG = 1u << AX_SWING1, SWING2_FLAG = 1u << AX_SWING2,
               TWIST_FLAG = 1u << AX_TWIST;
    const uint ANGULAR_MASK = SWING1_FLAG | SWING2_FLAG | TWIST_FLAG;
    const uint LINEAR_MASK = (1u << AX_X) | (1u << AX_Y) | (1u << AX_Z);

    if (useExtendedLimits == 0u)
    {
        // applyNeighborhoodOperator
        if (dot4Q(cA2wq, cB2wq) < 0.0f)
            cB2wq = -cB2wq;
    }

    // cB2cA = cA2w.transformInv(cB2w)
    const float4 cA2wqInv = quatConjQ(cA2wq);
    const float3 cB2cAp = quatRotQ(cA2wqInv, cB2wp - cA2wp);
    const float4 cB2cAq = quatMulQ(cA2wqInv, cB2wq);

    float3 amc0, amc1, amc2, bmc0, bmc1, bmc2;
    quatToMat33(cA2wq, amc0, amc1, amc2);
    quatToMat33(cB2wq, bmc0, bmc1, bmc2);

    const float3 bX = bmc0, aY = amc1, aZ = amc2;

    // ---- linear drives ----
    if ((driving & ((1u << DR_X) | (1u << DR_Y) | (1u << DR_Z))) != 0u)
    {
        const float3 posErr = s.drvPp.xyz - cB2cAp;
        for (uint i = 0u; i < 3u; ++i)
        {
            if ((driving & (1u << (DR_X + i))) != 0u)
            {
                const float3 axis = i == 0u ? amc0 : (i == 1u ? amc1 : amc2);
                const float vt = -(i == 0u ? s.linVel.x : (i == 1u ? s.linVel.y : s.linVel.z));
                const float pe = i == 0u ? posErr.x : (i == 1u ? posErr.y : posErr.z);
                C1D c = chLinear(mRa, mRb, axis, pe, H_NONE);
                chAddDrive(c, vt, s.drives[i]);
                buf.r[buf.n] = c; buf.n += 1u;
            }
        }
    }

    // ---- angular drives ----
    if ((driving & ((1u << DR_SLERP) | (1u << DR_TWIST) | (1u << DR_SWING1) | (1u << DR_SWING2))) != 0u)
    {
        const float4 d2cA_q = dot4Q(cB2cAq, s.drvPq) > 0.0f ? s.drvPq : -s.drvPq;

        if ((driving & (1u << DR_SLERP)) != 0u)
        {
            const float4 delta = quatMulQ(quatConjQ(d2cA_q), cB2cAq);

            const float3 velTarget = -quatRotQ(cA2wq, s.angVel.xyz);
            float3 ax0 = float3(1, 0, 0), ax1 = float3(0, 1, 0), ax2 = float3(0, 0, 1);
            if (s.drives[DRIVE_DATA_SLERP].x != 0.0f)  // slerp drive stiffness
                computeJacobianAxesQ(quatMulQ(cA2wq, d2cA_q), cB2wq, ax0, ax1, ax2);
            for (uint i = 0u; i < 3u; ++i)
            {
                const float3 axis = i == 0u ? ax0 : (i == 1u ? ax1 : ax2);
                const float de = i == 0u ? delta.x : (i == 1u ? delta.y : delta.z);
                C1D c = chAngular(axis, -de, H_SLERP_SPRING, 0u);
                chAddDrive(c, dot3Q(axis, velTarget), s.drives[DRIVE_DATA_SLERP]);
                buf.r[buf.n] = c; buf.n += 1u;
            }
        }
        else
        {
            const float3 v = s.angVel.xyz;

            // 5.12 (CHANGELOG v5.7, drive targets beyond 90 degrees): the twist /
            // swing error uses delta = cB2cA * conj(drivePose) (5.6.1 and the SLERP
            // branch above: conj(drivePose) * cB2cA)
            const float4 delta = quatMulQ(cB2cAq, quatConjQ(d2cA_q));

            if ((driving & (1u << DR_TWIST)) != 0u)
            {
                const float errX = -2.0f * delta.x;
                C1D c = chAngular(amc0, errX, H_NONE, 0u);
                chAddDrive(c, v.x, s.drives[DRIVE_DATA_TWIST]);
                buf.r[buf.n] = c; buf.n += 1u;
            }

            if ((driving & ((1u << DR_SWING1) | (1u << DR_SWING2))) != 0u)
            {
                const float3 err = basisVector0Q(delta);
                if ((driving & (1u << DR_SWING1)) != 0u)
                {
                    const float errY = err.z;
                    C1D c = chAngular(aY, errY, H_NONE, 0u);
                    chAddDrive(c, v.y, s.drives[DRIVE_DATA_SWING1]);
                    buf.r[buf.n] = c; buf.n += 1u;
                }
                if ((driving & (1u << DR_SWING2)) != 0u)
                {
                    const float errZ = -err.y;
                    C1D c = chAngular(aZ, errZ, H_NONE, 0u);
                    chAddDrive(c, v.z, s.drives[DRIVE_DATA_SWING2]);
                    buf.r[buf.n] = c; buf.n += 1u;
                }
            }
        }
    }

    // ---- angular limits ----
    if ((limited & ANGULAR_MASK) != 0u)
    {
        float4 swing, twist;
        separateSwingTwistQ(cB2cAq, swing, twist);

        if ((limited & SWING1_FLAG) != 0u && (limited & SWING2_FLAG) != 0u)
        {
            if (useConeLimit)
            {
                // setupConeSwingLimits
                float3 axis; float error;
                coneGetLimitQ(s.swingV.x, s.swingV.y, swing, axis, error, trig);
                chAngularLimit(buf, quatRotQ(cA2wq, axis), error, s.swingP);
            }
            if (usePyramidLimits)
            {
                // setupPyramidSwingLimits(useY = true, useZ = true)
                const float4 q = quatMulQ(cA2wq, swing);
                chAnglePair(buf, computeSwingAngleQ(swing.y, swing.w, trig), s.pyrV.x, s.pyrV.y,
                            basisVector1Q(q), s.pyrP);
                chAnglePair(buf, computeSwingAngleQ(swing.z, swing.w, trig), s.pyrV.z, s.pyrV.w,
                            basisVector2Q(q), s.pyrP);
            }
        }
        else
        {
            if ((limited & SWING1_FLAG) != 0u)
            {
                if ((locked & SWING2_FLAG) != 0u)
                {
                    if (usePyramidLimits)
                    {
                        // setupPyramidSwingLimits(useY = true, useZ = false)
                        const float4 q = quatMulQ(cA2wq, swing);
                        chAnglePair(buf, computeSwingAngleQ(swing.y, swing.w, trig), s.pyrV.x,
                                    s.pyrV.y, basisVector1Q(q), s.pyrP);
                    }
                    else
                    {
                        // setupSingleSwingLimit: swing Y limited, swing Z locked
                        chAnglePair(buf, computeSwingAngleQ(swing.y, swing.w, trig), -s.swingV.x,
                                    s.swingV.x, aY, s.swingP);
                    }
                }
                else
                {
                    // setupDualConeSwingLimits: swing Y limited, swing Z free
                    trig |= 4u;
                    chAnglePair(buf, asin(clamp(-dot3Q(aZ, bX), -1.0f, 1.0f)), -s.swingV.x,
                                s.swingV.x, getNormalizedQ(crossQ(aZ, bX)), s.swingP);
                }
            }
            if ((limited & SWING2_FLAG) != 0u)
            {
                if ((locked & SWING1_FLAG) != 0u)
                {
                    if (usePyramidLimits)
                    {
                        // setupPyramidSwingLimits(useY = false, useZ = true)
                        const float4 q = quatMulQ(cA2wq, swing);
                        chAnglePair(buf, computeSwingAngleQ(swing.z, swing.w, trig), s.pyrV.z,
                                    s.pyrV.w, basisVector2Q(q), s.pyrP);
                    }
                    else
                    {
                        // setupSingleSwingLimit: swing Z limited, swing Y locked
                        chAnglePair(buf, computeSwingAngleQ(swing.z, swing.w, trig), -s.swingV.y,
                                    s.swingV.y, aZ, s.swingP);
                    }
                }
                else
                {
                    // setupDualConeSwingLimits: swing Z limited, swing Y free
                    trig |= 4u;
                    chAnglePair(buf, asin(clamp(dot3Q(aY, bX), -1.0f, 1.0f)), -s.swingV.y,
                                s.swingV.y, getNormalizedQ(-crossQ(aY, bX)), s.swingP);
                }
            }
        }

        if ((limited & TWIST_FLAG) != 0u)
            chAnglePair(buf, computePhiQ(twist, trig), s.twistV.y, s.twistV.x, bmc0, s.twistP);
    }

    // ---- linear limits ----
    if ((limited & LINEAR_MASK) != 0u)
    {
        if (useDistanceLimit)
        {
            float3 limitDir = float3(0, 0, 0);
            for (uint i = 0u; i < 3u; ++i)
            {
                if ((limited & (1u << (AX_X + i))) != 0u)
                {
                    const float3 col = i == 0u ? amc0 : (i == 1u ? amc1 : amc2);
                    const float pc = i == 0u ? cB2cAp.x : (i == 1u ? cB2cAp.y : cB2cAp.z);
                    limitDir += col * pc;
                }
            }
            // P9.86: `PxVec3 limitDir(0); limitDir += cA2w_m[i] * cB2cA.p[i]` (constraintBlockPrePrep.cu:683-689)
            // is only ever += from a +0 seed, so no IEEE lane can be -0.  llvmpipe unrolls the constant-trip
            // loop and folds the forwarded seed into the first add, so a -0 product (+0 * negative cB2cAp.x)
            // survived.  Canonicalise -0 -> +0 with a select (identity for every nonzero, NaN, inf).
            limitDir = float3((limitDir.x == 0.0f) ? 0.0f : limitDir.x,
                              (limitDir.y == 0.0f) ? 0.0f : limitDir.y,
                              (limitDir.z == 0.0f) ? 0.0f : limitDir.z);
            const float distance = magQ(limitDir);
            if (distance > s.linVel.w)  // distanceMinDist
                chLinearLimit(buf, mRa, mRb, limitDir * (1.0f / distance), distance, s.distV.x,
                              s.distP);
        }

        if (useNewLinearLimits)
        {
            // setupLinearLimit per axis: (+axis, +origin, upper), (-axis, -origin, -lower)
            if ((limited & (1u << AX_X)) != 0u && s.linXV.y <= s.linXV.x)
            {
                chLinearLimit(buf, mRa, mRb, amc0, cB2cAp.x, s.linXV.x, s.linXP);
                chLinearLimit(buf, mRa, mRb, -amc0, -cB2cAp.x, -s.linXV.y, s.linXP);
            }
            if ((limited & (1u << AX_Y)) != 0u && s.linYV.y <= s.linYV.x)
            {
                chLinearLimit(buf, mRa, mRb, amc1, cB2cAp.y, s.linYV.x, s.linYP);
                chLinearLimit(buf, mRa, mRb, -amc1, -cB2cAp.y, -s.linYV.y, s.linYP);
            }
            if ((limited & (1u << AX_Z)) != 0u && s.linZV.y <= s.linZV.x)
            {
                chLinearLimit(buf, mRa, mRb, amc2, cB2cAp.z, s.linZV.x, s.linZP);
                chLinearLimit(buf, mRa, mRb, -amc2, -cB2cAp.z, -s.linZV.y, s.linZP);
            }
        }
    }

    // ---- single locked swing DOF special case ----
    const uint angularLocked = locked & ANGULAR_MASK;
    if (angularLocked == SWING1_FLAG)
    {
        C1D c = chAngular(crossQ(bX, aZ), -dot3Q(bX, aZ), H_EQUALITY, 0u);
        c.flags |= E_OUTPUT_FORCE;  // angularHard
        buf.r[buf.n] = c; buf.n += 1u;
        locked &= ~SWING1_FLAG;
    }
    else if (angularLocked == SWING2_FLAG)
    {
        locked &= ~SWING2_FLAG;
        C1D c = chAngular(crossQ(bX, aY), -dot3Q(bX, aY), H_EQUALITY, 0u);
        c.flags |= E_OUTPUT_FORCE;
        buf.r[buf.n] = c; buf.n += 1u;
    }

    // ---- prepareLockedAxes ----
    const uint lin = locked & 7u;
    const uint ang = locked >> 3u;
    float3 ra = mRa;
    {
        if (lin != 0u)
        {
            float3 axc0, axc1, axc2;
            quatToMat33(cA2wq, axc0, axc1, axc2);
            // P9.86: `PxVec3 errorVector(0); errorVector -= ...` (PxgConstraintHelper.h:176-178).
            // Spelled literally, slangc emits the first subtraction as OpFSub(0, a), which
            // llvmpipe rewrites to -a even under SignedZeroInfNanPreserve -- so a = +0 (a
            // zero offset, the common joint) gave -0 where IEEE gives +0.  The first term
            // goes through d6ZeroMinus (a select: no arithmetic identity to exploit); an
            // unlocked axis contributes +0, and x - (+0) is the identity for every x.
            const float3 t0 = ((lin & 1u) != 0u) ? axc0 * cB2cAp.x : float3(0, 0, 0);
            const float3 t1 = ((lin & 2u) != 0u) ? axc1 * cB2cAp.y : float3(0, 0, 0);
            const float3 t2 = ((lin & 4u) != 0u) ? axc2 * cB2cAp.z : float3(0, 0, 0);
            float3 errorVector = d6ZeroMinus(t0);
            errorVector -= t1;
            errorVector -= t2;
            ra += errorVector;
            if ((lin & 1u) != 0u)
            {
                buf.r[buf.n] = chLinearRa(ra, mRb, axc0, -cB2cAp.x, H_EQUALITY, E_OUTPUT_FORCE);
                buf.n += 1u;
            }
            if ((lin & 2u) != 0u)
            {
                buf.r[buf.n] = chLinearRa(ra, mRb, axc1, -cB2cAp.y, H_EQUALITY, E_OUTPUT_FORCE);
                buf.n += 1u;
            }
            if ((lin & 4u) != 0u)
            {
                buf.r[buf.n] = chLinearRa(ra, mRb, axc2, -cB2cAp.z, H_EQUALITY, E_OUTPUT_FORCE);
                buf.n += 1u;
            }
        }
        if (ang != 0u)
        {
            const float4 qB2qA = quatMulQ(quatConjQ(cA2wq), cB2wq);
            float3 row0, row1, row2;
            computeJacobianAxesQ(cA2wq, cB2wq, row0, row1, row2);
            const float3 imp = qB2qA.xyz;
            if ((ang & 1u) != 0u)
            {
                buf.r[buf.n] = chAngular(row0, -imp.x, H_EQUALITY, E_OUTPUT_FORCE);
                buf.n += 1u;
            }
            if ((ang & 2u) != 0u)
            {
                buf.r[buf.n] = chAngular(row1, -imp.y, H_EQUALITY, E_OUTPUT_FORCE);
                buf.n += 1u;
            }
            if ((ang & 4u) != 0u)
            {
                buf.r[buf.n] = chAngular(row2, -imp.z, H_EQUALITY, E_OUTPUT_FORCE);
                buf.n += 1u;
            }
        }
    }

    raOut = ra;
    cB2wpOut = cB2wp;
}

void packRows(const RowBuf buf, inout D6Out o)
{
    for (uint i = 0u; i < MAX_ROWS; ++i)
    {
        if (i < buf.n)
        {
            const C1D c = buf.r[i];
            o.rows[i * 5u + 0u] = float4(c.lin0, c.geo);
            o.rows[i * 5u + 1u] = float4(c.ang0, c.vt);
            o.rows[i * 5u + 2u] = float4(c.lin1, c.minI);
            o.rows[i * 5u + 3u] = float4(c.ang1, c.maxI);
            o.rows[i * 5u + 4u] = float4(c.m0, c.m1, asfloat(c.flags | (c.hint << 16u)), 0.0f);
        }
        else
        {
            o.rows[i * 5u + 0u] = float4(0, 0, 0, 0);
            o.rows[i * 5u + 1u] = float4(0, 0, 0, 0);
            o.rows[i * 5u + 2u] = float4(0, 0, 0, 0);
            o.rows[i * 5u + 3u] = float4(0, 0, 0, 0);
            o.rows[i * 5u + 4u] = float4(0, 0, 0, 0);
        }
    }
}

// ---- entry: bare D6JointSolverPrep (tranche 174) ----

[shader("compute")]
[numthreads(64, 1, 1)]
void d6jPrep(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const D6In s = gIn[tid];

    uint trig = 0u;
    RowBuf buf;
    buf.n = 0u;
    float3 ra, cB2wp;
    d6SolveQ(s, asuint(s.c2bAp.w), buf, trig, ra, cB2wp);

    D6Out o;
    o.counts = uint4(buf.n, trig, 0u, 0u);
    o.cAoutW = float4(ra + s.bA2wp.xyz, 0.0f);
    o.cBoutW = float4(cB2wp, 0.0f);
    o.nrfs = uint4(0, 0, 0, 0);
    o.invMass = float4(0, 0, 0, 0);
    o.raW = float4(0, 0, 0, 0);
    o.rbW = float4(0, 0, 0, 0);
    packRows(buf, o);
    gOut[tid] = o;
}

// ---- entry: constraint1DPrePrep wrapper (constraintBlockPrePrep.cu
// L1059-1166, tranche 175) ----
// The pose fetch (solver body data / articulation link poses,
// PxAlignedTransform::getTransform = arithmetic-free repack) is data
// plumbing; the ported compute: the removed-joint gate (both node indices
// PX_INVALID_NODE -> NO output write), the eDISABLE_CONSTRAINT branch
// (numRows = 0, ra = rb = 0), useExtendedLimits derived from the prePrep
// flags, the D6JointSolverPrep call, ra/rb = cX2w - poseX.p (the exact
// add-then-subtract roundtrip through cA2wOut = ra + bA2w.p), and the
// PxgConstraintData image (invMassScale passthrough,
// mRaWorld_linBreakForceW / mRbWorld_angBreakForceW,
// mNumRows_Flags_StartIndex with startIndex = jointDataIndex *
// Dy::MAX_CONSTRAINT_ROWS = 20).
// Extra input lanes: angVel.w = asuint(prePrep mFlags), swingV.z =
// asuint(removed), distV.y/z = lin/ang break force, distV.w =
// asuint(jointDataIndex), linXV.zw + linYV.zw = invMassScale.

static const uint E_DISABLE_CONSTRAINT = 4096u;    // PxConstraintFlag 1<<12
static const uint E_EXTENDED_LIMITS = 512u;        // PxConstraintFlag 1<<9
static const uint DY_MAX_CONSTRAINT_ROWS = 20u;    // DyConstraintPrep.h L55

[shader("compute")]
[numthreads(64, 1, 1)]
void c1dPrePrep(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const D6In s = gIn[tid];

    // removed joint: both node indices PX_INVALID_NODE -> nothing written
    if (asuint(s.swingV.z) != 0u)
        return;

    const uint prePrepFlags = asuint(s.angVel.w);
    const bool disableConstraints = (prePrepFlags & E_DISABLE_CONSTRAINT) != 0u;
    const uint useExtendedLimits = (prePrepFlags & E_EXTENDED_LIMITS) != 0u ? 1u : 0u;
    const uint jointDataIndex = asuint(s.distV.w);

    uint trig = 0u;
    RowBuf buf;
    buf.n = 0u;

    float3 ra, rb;
    if (!disableConstraints)
    {
        float3 raInner, cB2wp;
        d6SolveQ(s, useExtendedLimits, buf, trig, raInner, cB2wp);
        const float3 ca2w = raInner + s.bA2wp.xyz;  // cA2wOut of D6JointSolverPrep
        const float3 cb2w = cB2wp;                  // cB2wOut
        ra = ca2w - s.bA2wp.xyz;                    // pose0.p
        rb = cb2w - s.bB2wp.xyz;                    // pose1.p
    }
    else
    {
        ra = float3(0, 0, 0);
        rb = float3(0, 0, 0);
    }

    const uint constraintsStartIndex = jointDataIndex * DY_MAX_CONSTRAINT_ROWS;

    D6Out o;
    o.counts = uint4(buf.n, trig, 0u, 0u);
    o.cAoutW = float4(0, 0, 0, 0);
    o.cBoutW = float4(0, 0, 0, 0);
    o.nrfs = uint4(buf.n, prePrepFlags, constraintsStartIndex, 0u);
    o.invMass = float4(s.linXV.z, s.linXV.w, s.linYV.z, s.linYV.w);
    o.raW = float4(ra, s.distV.y);   // mLinBreakForce
    o.rbW = float4(rb, s.distV.z);   // mAngBreakForce
    packRows(buf, o);
    gOut[tid] = o;
}

// ---------------------------------------------------------------------------
// constraint1DPrePrepLaunch (P5 Step 5 BDA engine ABI)
// ---------------------------------------------------------------------------
// Port of constraint1DPrePrepLaunch() in constraintBlockPrePrep.cu L1168 — the
// per-joint D6 data-prep kernel.  The engine launches it from
// PxgSolverCore::constraintPrePrepParallel (PxgSolverCore.cpp L600) with the
// SAME two-pointer pack as the block twin:
//   params { PxgPrePrepDesc*, PxgSolverSharedDesc<IterativeSolveData>* } -> {8,8},
//   grid = ceil(nbD6Joints / CONSTRAINT_PREPREP_BLOCK=128), block = 128,
//   thread-per-joint (threadIndex = threadIdx.x + blockIdx.x*blockDim.x).  It is
//   launched once per step regardless of solver (PxgPrePrepDesc is
//   solver-agnostic and constraintPrePrepParallel is shared by the PGS and TGS
//   cores), so BOTH solver paths hit this kernel.  nbD6Joints spans rigid AND
//   arti joints; threads with threadIndex >= nbRigidJoints take the articulation
//   arm, constraint1DPrePrep<true> (L1203-1212): the artiJointData /
//   artiConstraintPrePrep inputs, the artiConstraintData / artiConstraintRows
//   outputs, and a link's pose from
//   articulations[bodySims[node].articulationRemapId].linkBody2Worlds[linkId]
//   (L1083-1113).  It was a deferred no-op until a D6 joint touching a link was
//   measured to have EXACTLY zero effect: every articulation D6 batch read 0 rows
//   from the never-written arti pools (SnippetDirectGPUAPIArticulation's scissor
//   loops; link-to-body and link-to-world joints alike).
//
// The rigid arm is constraint1DPrePrep<false> transcribed line-for-line:
//   - the removed-joint gate (both node indices PX_INVALID_NODE -> NO write);
//   - poses from solverBodyData[solverBodyIndices[node]].body2World.getTransform()
//     (PX_INVALID_NODE -> solver body 0), an arithmetic-free repack;
//   - the eDISABLE_CONSTRAINT branch (numRows = 0, ra = rb = 0);
//   - useExtendedLimits from the prePrep mFlags;
//   - the D6JointSolverPrep call (d6SolveQ) with ra = ca2w - pose0.p (= raInner)
//     and rb = cb2w - pose1.p;
//   - the PxgConstraintData image (invMassScale passthrough from the joint data,
//     mRaWorld_linBreakForceW / mRbWorld_angBreakForceW,
//     mNumRows_Flags_StartIndex with startIndex = jointDataIndex * 20).
// All D6 row arithmetic is the proven d6SolveQ; only the desc-driven data plumbing
// and the real PxgD6JointData -> D6In assembly are new here.

static const uint PX_INVALID_NODE_J = 0xFFFFFFFFu;

struct C1dLaunchParams
{
    uint64_t prePrepDesc; // @0 PxgPrePrepDesc*
    uint64_t sharedDesc;  // @8 PxgSolverSharedDesc<IterativeSolveData>*
}

[[vk::push_constant]]
ConstantBuffer<C1dLaunchParams> gC1dLaunch;

// mirrors PxgD6JointData (PxgD6JointData.h + PxgD6JointLimit.h + PxgD6Joint.h),
// 448 bytes.  Offsets verified against the C++ layout (PxgJointData base:
// PxgConstraintInvMassScale @0 [PX_ALIGN 16], PxTransform32 c2b[2] @16..80;
// then PxgD6Motion motion[6] @80, the PxgJointLimitParameters-derived limits
// [restitution,bounceThreshold,stiffness,damping first], the 16-byte
// PxgD6JointDrive drive[6], PxTransform drivePosition, the two PxVec3 drive
// velocities, the locked/limited/driving bitmaps, distanceMinDist, the four
// mUse* bools packed as bytes, and angularDriveConfig).  The end-to-end row
// comparison in testNpD6JointPrep pins every offset empirically.
struct PxgD6JointDataGpu
{
    float4 invMassScale;   // @0   lin0,ang0,lin1,ang1
    float4 c2bAq;          // @16  c2b[0].q
    float4 c2bAp;          // @32  c2b[0].p.xyz + pad
    float4 c2bBq;          // @48  c2b[1].q
    float4 c2bBp;          // @64  c2b[1].p.xyz + pad
    uint motion0; uint motion1; uint motion2; uint motion3; uint motion4; uint motion5; // @80..104
    float dist_rest; float dist_bounce; float dist_stiff; float dist_damp; float dist_value; // @104..124
    float linX_rest; float linX_bounce; float linX_stiff; float linX_damp; float linX_upper; float linX_lower; // @124..148
    float linY_rest; float linY_bounce; float linY_stiff; float linY_damp; float linY_upper; float linY_lower; // @148..172
    float linZ_rest; float linZ_bounce; float linZ_stiff; float linZ_damp; float linZ_upper; float linZ_lower; // @172..196
    float twist_rest; float twist_bounce; float twist_stiff; float twist_damp; float twist_upper; float twist_lower; // @196..220
    float swing_rest; float swing_bounce; float swing_stiff; float swing_damp; float swing_yAngle; float swing_zAngle; // @220..244
    float pyr_rest; float pyr_bounce; float pyr_stiff; float pyr_damp; float pyr_yMin; float pyr_yMax; float pyr_zMin; float pyr_zMax; // @244..276
    float d0_s; float d0_d; float d0_f; uint d0_flags; // @276..292
    float d1_s; float d1_d; float d1_f; uint d1_flags; // @292..308
    float d2_s; float d2_d; float d2_f; uint d2_flags; // @308..324
    float d3_s; float d3_d; float d3_f; uint d3_flags; // @324..340
    float d4_s; float d4_d; float d4_f; uint d4_flags; // @340..356
    float d5_s; float d5_d; float d5_f; uint d5_flags; // @356..372
    float drvQ_x; float drvQ_y; float drvQ_z; float drvQ_w; // @372..388  drivePosition.q
    float drvP_x; float drvP_y; float drvP_z;              // @388..400  drivePosition.p
    float dlv_x; float dlv_y; float dlv_z;                 // @400..412  driveLinearVelocity
    float dav_x; float dav_y; float dav_z;                 // @412..424  driveAngularVelocity
    uint locked;             // @424
    uint limited;            // @428
    uint driving;            // @432
    float distanceMinDist;   // @436
    uint useBools;           // @440  byte0=useDistance,1=useNewLinear,2=useCone,3=usePyramid
    uint angularDriveConfig; // @444  (byte0 only; rest pad) -> size 448
}

// mirrors PxgConstraintPrePrep (PxgConstraintPrep.h), 32 bytes (PxNodeIndex is
// PX_ALIGN 8; mNodeIndexA/B = PxU64 {index, linkID})
struct PxgConstraintPrePrepGpu
{
    uint2 mNodeIndexA;       // @0  (x = index)
    uint2 mNodeIndexB;       // @8
    uint mFlags;             // @16 PxConstraintFlag
    float mLinBreakForce;    // @20
    float mAngBreakForce;    // @24
    uint pad;                // @28 -> size 32
}

// mirrors PxgConstraintData (PxgConstraintPrep.h), 64 bytes (output)
struct PxgConstraintDataGpu
{
    float4 mInvMassScale;              // @0
    float4 mRaWorld_linBreakForceW;    // @16
    float4 mRbWorld_angBreakForceW;    // @32
    uint4 mNumRows_Flags_StartIndex;   // @48
}

// mirrors Px1DConstraint (PxConstraintDesc.h), 80 bytes PX_ALIGN(16) (output row)
struct Px1DConstraintGpu
{
    float4 linear0XYZ_geometricErrorW; // @0
    float4 angular0XYZ_velocityTargetW;// @16
    float4 linear1XYZ_minImpulseW;     // @32
    float4 angular1XYZ_maxImpulseW;    // @48
    float mods_restitution;            // @64 (mods.spring.stiffness / bounce.restitution)
    float mods_velocityThreshold;      // @68 (mods.spring.damping / bounce.velocityThreshold)
    uint flags_solveHint;              // @72 PxU16 flags | PxU16 solveHint << 16
    uint pad;                          // @76 -> size 80
}

// Assemble the flat D6In (the proven d6SolveQ input) from the REAL joint data +
// the two body poses.  The mapping mirrors the D6In field comments and the
// PxgConstraintHelper / D6JointSolverPrep field reads (constraintBlockPrePrep.cu
// L516-737).  useBits is packed into c2bBp.w exactly as d6SolveQ expects; the
// locked/limited/driving bitmaps into bA2wp.w / bB2wp.w / drvPp.w.
D6In d6InFromReal(const PxgD6JointDataGpu d, float4 pose0q, float3 pose0p, float4 pose1q,
                  float3 pose1p)
{
    const bool useDistance = (d.useBools & 0xFFu) != 0u;
    const bool useNewLinear = ((d.useBools >> 8u) & 0xFFu) != 0u;
    const bool useCone = ((d.useBools >> 16u) & 0xFFu) != 0u;
    const bool usePyramid = ((d.useBools >> 24u) & 0xFFu) != 0u;
    const uint useBits = (useDistance ? 1u : 0u) | (useNewLinear ? 2u : 0u) |
                         (useCone ? 4u : 0u) | (usePyramid ? 8u : 0u);

    D6In s;
    s.c2bAq = d.c2bAq;
    s.c2bAp = float4(d.c2bAp.xyz, 0.0f);            // .w unused by d6SolveQ
    s.c2bBq = d.c2bBq;
    s.c2bBp = float4(d.c2bBp.xyz, asfloat(useBits));
    s.bA2wq = pose0q;
    s.bA2wp = float4(pose0p, asfloat(d.locked));
    s.bB2wq = pose1q;
    s.bB2wp = float4(pose1p, asfloat(d.limited));
    s.drvPq = float4(d.drvQ_x, d.drvQ_y, d.drvQ_z, d.drvQ_w);
    s.drvPp = float4(d.drvP_x, d.drvP_y, d.drvP_z, asfloat(d.driving));
    s.linVel = float4(d.dlv_x, d.dlv_y, d.dlv_z, d.distanceMinDist);
    s.angVel = float4(d.dav_x, d.dav_y, d.dav_z, 0.0f);
    s.drives[0] = float4(d.d0_s, d.d0_d, d.d0_f, asfloat(d.d0_flags));
    s.drives[1] = float4(d.d1_s, d.d1_d, d.d1_f, asfloat(d.d1_flags));
    s.drives[2] = float4(d.d2_s, d.d2_d, d.d2_f, asfloat(d.d2_flags));
    s.drives[3] = float4(d.d3_s, d.d3_d, d.d3_f, asfloat(d.d3_flags));
    s.drives[4] = float4(d.d4_s, d.d4_d, d.d4_f, asfloat(d.d4_flags));
    s.drives[5] = float4(d.d5_s, d.d5_d, d.d5_f, asfloat(d.d5_flags));
    s.distP = float4(d.dist_rest, d.dist_bounce, d.dist_stiff, d.dist_damp);
    s.distV = float4(d.dist_value, 0.0f, 0.0f, 0.0f);
    s.linXP = float4(d.linX_rest, d.linX_bounce, d.linX_stiff, d.linX_damp);
    s.linXV = float4(d.linX_upper, d.linX_lower, 0.0f, 0.0f);
    s.linYP = float4(d.linY_rest, d.linY_bounce, d.linY_stiff, d.linY_damp);
    s.linYV = float4(d.linY_upper, d.linY_lower, 0.0f, 0.0f);
    s.linZP = float4(d.linZ_rest, d.linZ_bounce, d.linZ_stiff, d.linZ_damp);
    s.linZV = float4(d.linZ_upper, d.linZ_lower, 0.0f, 0.0f);
    s.twistP = float4(d.twist_rest, d.twist_bounce, d.twist_stiff, d.twist_damp);
    s.twistV = float4(d.twist_upper, d.twist_lower, 0.0f, 0.0f);
    s.swingP = float4(d.swing_rest, d.swing_bounce, d.swing_stiff, d.swing_damp);
    s.swingV = float4(d.swing_yAngle, d.swing_zAngle, 0.0f, 0.0f);
    s.pyrP = float4(d.pyr_rest, d.pyr_bounce, d.pyr_stiff, d.pyr_damp);
    s.pyrV = float4(d.pyr_yMin, d.pyr_yMax, d.pyr_zMin, d.pyr_zMax);
    return s;
}

// One side's pose for constraint1DPrePrep (constraintBlockPrePrep.cu:1083-1129).
// The articulation arm takes a LINK's pose from
// articulations[bodySims[node].articulationRemapId].linkBody2Worlds[linkId]
// (PxgSolverSharedDescBase: mBodySimBufferDeviceData @32, articulations @40 --
// the base, so the PGS and TGS template instances agree); every other side, and
// every side of a rigid joint, reads solverBodyData[solverBodyIndices[node]]
// (PX_INVALID_NODE -> solver body 0) and getTransform()s the aligned pose.
void d6JointPose(uint2 igNodeIndex, bool hasArticulationLink, Ptr<uint> solverBodyIndices,
                 Ptr<PxgSolverBodyData> solverBodyData, out float4 q, out float3 p)
{
    const uint nodeIndex = igNodeIndex.x;   // PxNodeIndex::index()
    if (hasArticulationLink && nodeIndexIsArticulation(igNodeIndex))
    {
        Ptr<uint64_t> shared = Ptr<uint64_t>(gC1dLaunch.sharedDesc);
        Ptr<PxgBodySim> bodySims = Ptr<PxgBodySim>(shared[4]);   // @32
        const uint64_t articulations = shared[5];                 // @40
        const uint articulationId = bodySims[nodeIndex].articulationRemapId;
        const uint linkId = nodeIndexArticulationLinkId(igNodeIndex);
        const uint64_t a = articulations + uint64_t(articulationId) * PXG_ARTI_STRIDE;
        // linkBody2Worlds: PxTransform (28 B) -- q at 0, p at 16
        Ptr<float> t = Ptr<float>(artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS) + uint64_t(linkId) * 28u);
        q = float4(t[0], t[1], t[2], t[3]);
        p = float3(t[4], t[5], t[6]);
        return;
    }
    const uint sbIndex = nodeIndex == PX_INVALID_NODE_J ? 0u : solverBodyIndices[nodeIndex];
    const PxgSolverBodyData body = solverBodyData[sbIndex];
    q = body.body2World.q;       // PxAlignedTransform::getTransform()
    p = body.body2World.p.xyz;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void constraint1DPrePrepLaunch(uint3 dtid : SV_DispatchThreadID)
{
    // llvmpipe zero-trip guard on a null descriptor pointer
    if (gC1dLaunch.prePrepDesc == 0)
        return;

    Ptr<PxgPrePrepDescGpu> desc = Ptr<PxgPrePrepDescGpu>(gC1dLaunch.prePrepDesc);

    const uint nbRigidJoints = desc[0].nbD6RigidJoints;
    const uint nbArtiJoints = desc[0].nbD6ArtiJoints;
    const uint nbTotalJoints = nbRigidJoints + nbArtiJoints;
    const uint threadIndex = dtid.x;

    if (threadIndex >= nbTotalJoints)
        return;

    // threadIndex >= nbRigidJoints: the articulation arm (L1201-1213), index
    // threadIndex - nbRigidJoints into the arti pools.
    const bool hasArticulationLink = threadIndex >= nbRigidJoints;
    const uint jointDataIndex = hasArticulationLink ? threadIndex - nbRigidJoints : threadIndex;

    Ptr<PxgConstraintPrePrepGpu> prePrep = Ptr<PxgConstraintPrePrepGpu>(
        hasArticulationLink ? desc[0].artiConstraintPrePrep : desc[0].rigidConstraintPrePrep);
    const PxgConstraintPrePrepGpu pp = prePrep[jointDataIndex];

    const uint nodeIndexA = pp.mNodeIndexA.x;   // PxNodeIndex::index()
    const uint nodeIndexB = pp.mNodeIndexB.x;

    // removed joint: both node indices PX_INVALID_NODE -> nothing written
    if (nodeIndexA == PX_INVALID_NODE_J && nodeIndexB == PX_INVALID_NODE_J)
        return;

    const uint prePrepFlags = pp.mFlags;
    const bool disableConstraints = (prePrepFlags & E_DISABLE_CONSTRAINT) != 0u;
    const uint useExtendedLimits = (prePrepFlags & E_EXTENDED_LIMITS) != 0u ? 1u : 0u;

    Ptr<uint> solverBodyIndices = Ptr<uint>(desc[0].solverBodyIndices);
    Ptr<PxgSolverBodyData> solverBodyData = Ptr<PxgSolverBodyData>(desc[0].mSolverBodyData);

    float4 pose0q, pose1q;
    float3 pose0p, pose1p;
    d6JointPose(pp.mNodeIndexA, hasArticulationLink, solverBodyIndices, solverBodyData, pose0q, pose0p);
    d6JointPose(pp.mNodeIndexB, hasArticulationLink, solverBodyIndices, solverBodyData, pose1q, pose1p);

    Ptr<PxgD6JointDataGpu> jointData = Ptr<PxgD6JointDataGpu>(
        hasArticulationLink ? desc[0].artiJointData : desc[0].rigidJointData);
    const PxgD6JointDataGpu d = jointData[jointDataIndex];

    uint trig = 0u;
    RowBuf buf;
    buf.n = 0u;
    float3 ra, rb;
    if (!disableConstraints)
    {
        const D6In s = d6InFromReal(d, pose0q, pose0p, pose1q, pose1p);
        float3 raInner, cB2wp;
        d6SolveQ(s, useExtendedLimits, buf, trig, raInner, cB2wp);
        // exact CUDA roundtrip: D6JointSolverPrep returns cA2wOut = raInner +
        // bA2w.p, then ra = cA2wOut - pose0.p — NOT algebraically raInner under
        // NoContraction float rounding, so the add-then-subtract must be kept.
        const float3 ca2w = raInner + pose0p; // cA2wOut
        ra = ca2w - pose0p;                   // cA2wOut - pose0.p
        rb = cB2wp - pose1p;                  // cB2wOut - pose1.p
    }
    else
    {
        ra = float3(0, 0, 0);
        rb = float3(0, 0, 0);
    }

    const uint numRows = buf.n;
    const uint constraintsStartIndex = jointDataIndex * DY_MAX_CONSTRAINT_ROWS;

    Ptr<Px1DConstraintGpu> constraintRows = Ptr<Px1DConstraintGpu>(
        hasArticulationLink ? desc[0].artiConstraintRows : desc[0].constraintRows);
    for (uint i = 0u; i < numRows; ++i)
    {
        const C1D c = buf.r[i];
        Px1DConstraintGpu row;
        row.linear0XYZ_geometricErrorW = float4(c.lin0, c.geo);
        row.angular0XYZ_velocityTargetW = float4(c.ang0, c.vt);
        row.linear1XYZ_minImpulseW = float4(c.lin1, c.minI);
        row.angular1XYZ_maxImpulseW = float4(c.ang1, c.maxI);
        row.mods_restitution = c.m0;
        row.mods_velocityThreshold = c.m1;
        row.flags_solveHint = c.flags | (c.hint << 16u);
        row.pad = 0u;
        constraintRows[constraintsStartIndex + i] = row;
    }

    Ptr<PxgConstraintDataGpu> constraintData = Ptr<PxgConstraintDataGpu>(
        hasArticulationLink ? desc[0].artiConstraintData : desc[0].constraintData);
    PxgConstraintDataGpu cd;
    cd.mInvMassScale = d.invMassScale;             // passthrough from jointData
    cd.mRaWorld_linBreakForceW = float4(ra, pp.mLinBreakForce);
    cd.mRbWorld_angBreakForceW = float4(rb, pp.mAngBreakForce);
    cd.mNumRows_Flags_StartIndex = uint4(numRows, prePrepFlags, constraintsStartIndex, 0u);
    constraintData[jointDataIndex] = cd;
}
