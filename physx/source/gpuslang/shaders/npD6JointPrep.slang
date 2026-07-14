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
// PxgD6Drive bit positions (driving bitmap)
static const uint DR_X = 0u, DR_Y = 1u, DR_Z = 2u, DR_SWING = 3u, DR_TWIST = 4u, DR_SLERP = 5u,
                  DR_SWING1 = 6u, DR_SWING2 = 7u;

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

float pxMaxQ(float a, float b) { return (a < b) ? b : a; }

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
        if (c.geo > 0.0f)
            flags |= E_KEEPBIAS;
        if (lp.x > 0.0f)
            flags |= E_RESTITUTION;
    }
    c.flags = flags;
    c.minI = 0.0f;
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

// ---- entry ----

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

    const uint useExtendedLimits = asuint(s.c2bAp.w);
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
    if ((driving & ((1u << DR_SLERP) | (1u << DR_SWING) | (1u << DR_TWIST) |
                    (1u << DR_SWING1) | (1u << DR_SWING2))) != 0u)
    {
        const float4 d2cA_q = dot4Q(cB2cAq, s.drvPq) > 0.0f ? s.drvPq : -s.drvPq;
        const float3 v = s.angVel.xyz;
        const float4 delta = quatMulQ(quatConjQ(d2cA_q), cB2cAq);

        if ((driving & (1u << DR_SLERP)) != 0u)
        {
            const float3 velTarget = -quatRotQ(cA2wq, s.angVel.xyz);
            float3 ax0 = float3(1, 0, 0), ax1 = float3(0, 1, 0), ax2 = float3(0, 0, 1);
            if (s.drives[5].x != 0.0f)  // slerp drive stiffness
                computeJacobianAxesQ(quatMulQ(cA2wq, d2cA_q), cB2wq, ax0, ax1, ax2);
            for (uint i = 0u; i < 3u; ++i)
            {
                const float3 axis = i == 0u ? ax0 : (i == 1u ? ax1 : ax2);
                const float de = i == 0u ? delta.x : (i == 1u ? delta.y : delta.z);
                C1D c = chAngular(axis, -de, H_SLERP_SPRING, 0u);
                chAddDrive(c, dot3Q(axis, velTarget), s.drives[5]);
                buf.r[buf.n] = c; buf.n += 1u;
            }
        }
        else
        {
            if ((driving & (1u << DR_TWIST)) != 0u)
            {
                C1D c = chAngular(amc0, -2.0f * delta.x, H_NONE, 0u);
                chAddDrive(c, v.x, s.drives[4]);
                buf.r[buf.n] = c; buf.n += 1u;
            }

            if ((driving & (1u << DR_SWING)) != 0u)
            {
                const float3 err = basisVector0Q(delta);
                if ((locked & SWING1_FLAG) == 0u)
                {
                    C1D c = chAngular(aY, err.z, H_NONE, 0u);
                    chAddDrive(c, v.y, s.drives[3]);
                    buf.r[buf.n] = c; buf.n += 1u;
                }
                if ((locked & SWING2_FLAG) == 0u)
                {
                    C1D c = chAngular(aZ, -err.y, H_NONE, 0u);
                    chAddDrive(c, v.z, s.drives[3]);
                    buf.r[buf.n] = c; buf.n += 1u;
                }
            }
            else if ((driving & ((1u << DR_SWING1) | (1u << DR_SWING2))) != 0u)
            {
                const float3 err = basisVector0Q(delta);
                if ((driving & (1u << DR_SWING1)) != 0u)
                {
                    C1D c = chAngular(aY, err.z, H_NONE, 0u);
                    chAddDrive(c, v.y, s.drives[3]);  // gDriveSwing1DataIndex = eSWING entry
                    buf.r[buf.n] = c; buf.n += 1u;
                }
                if ((driving & (1u << DR_SWING2)) != 0u)
                {
                    C1D c = chAngular(aZ, -err.y, H_NONE, 0u);
                    chAddDrive(c, v.z, s.drives[5]);  // gDriveSwing2DataIndex = eSLERP entry
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
            float3 errorVector = float3(0, 0, 0);
            if ((lin & 1u) != 0u) errorVector -= axc0 * cB2cAp.x;
            if ((lin & 2u) != 0u) errorVector -= axc1 * cB2cAp.y;
            if ((lin & 4u) != 0u) errorVector -= axc2 * cB2cAp.z;
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

    // ---- outputs ----
    D6Out o;
    o.counts = uint4(buf.n, trig, 0u, 0u);
    o.cAoutW = float4(ra + s.bA2wp.xyz, 0.0f);
    o.cBoutW = float4(cB2wp, 0.0f);
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
    gOut[tid] = o;
}
