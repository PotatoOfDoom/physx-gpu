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

// npSpherePrimitives.slang
//
// Shared, single-source contact-generation math for the sphere/capsule/plane/box
// primitive narrow phase (gpunarrowphase/src/CUDA/sphereCollision.cuh, used by
// cudaSphere.cu's sphereNphase_Kernel).  The six routines below produce at most
// one (sphere-*) or two (plane-capsule / capsule-capsule) contacts each, in world
// space, exactly as the CUDA helpers do.
//
// These functions are the single source of truth for the sphere-primitive math:
// the bound-buffer harness entry points (npSphereCollision.slang /
// npCapsuleCollision.slang) call them for their unit tests, and the engine-ABI
// dispatcher sphereNphase_Kernel (npSphereNphase.slang) calls them per contact
// manager.  Transforms are passed as raw (quaternion, position) pairs so both the
// harness Xform and the engine PxsCachedTransform can drive them.

module npSpherePrimitives;

import pxgSlangCommon; // quatRotate, quatRotateInv, quatBasisVector0, quatConjugate, quatMul

public static const float PX_EPS_REAL = 1.192092896e-07f;

// Result of a primitive-primitive contact query.  pointPen0/1 are float4(worldPoint,
// penetration); a routine that yields a single contact leaves nbContacts <= 1 and
// only writes pointPen0.  normal is a shared world-space contact normal.
public struct PrimContact
{
    public uint   nbContacts;
    public float4 pointPen0;
    public float4 pointPen1;
    public float3 normal;
}

PrimContact emptyContact()
{
    PrimContact c;
    c.nbContacts = 0u;
    c.pointPen0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    c.pointPen1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    c.normal = float3(0.0f, 0.0f, 0.0f);
    return c;
}

// PxTransform::transformInv / transform on a (q, p) pair.
float3 xformInv(float4 q, float3 p, float3 pt) { return quatRotateInv(q, pt - p); }
float3 xformPt(float4 q, float3 p, float3 pt) { return quatRotate(q, pt) + p; }
float3 normalize3(float3 v) { return v * rsqrt(dot(v, v)); }

// mirrors sphereCollision.cuh distancePointSegmentSquared
float distancePointSegmentSquared(float3 a, float3 b, float3 p, out float param)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    // P9.86f: sphereCollision.cuh:98 PxClamp(nom / denom, 0, 1) = PxMin(1, PxMax(0, .)) with the float
    // selects of PxUnixMathIntrinsics.h:116-125: nom = -0 keeps the param -0 (and NaN stays NaN), where
    // GLSL FClamp gives +0 on llvmpipe.  (sphereCapsuleContact, the only caller, cannot show the sign:
    // a -0 quotient needs every lane product of ap.ab to be -0, and then p = segS + ab*t changes sign
    // only in a lane with segS = -0 and ab = +0 -- whose ap lane is +0 or nonzero, so dir keeps its bits.)
    const float tQ = nom / denom;
    const float tLo = (0.0f > tQ) ? 0.0f : tQ;
    const float tValue = (1.0f < tLo) ? 1.0f : tLo;
    const float t = denom > 0.0f ? tValue : 0.0f;
    const float3 v = ap - (ab * t);
    param = t;
    return dot(v, v);
}

// mirrors distanceSegmentSegment.cuh distanceSegmentSegmentSquared.  Public since
// 5.12: boxBoxNphase's edge-edge fallback (cudaBox.cu computeEdgeEdgeContactWorld)
// calls the same helper, so it imports this body rather than retyping it.
public void distanceSegmentSegmentSquared(float3 p1, float3 d1, float3 p2, float3 d2, out float s, out float t)
{
    const float eps = 1e-6f;
    const float3 r = p1 - p2;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float b = dot(d1, d2);
    const float c = dot(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = dot(d2, r);

    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    // P9.86f: distanceSegmentSegment.cuh:62 PxClamp(temp / denom, 0, 1) = PxMin(1, PxMax(0, .)) with the
    // float selects: temp = -0 (b >= +0, f = -0, c = +0) keeps s0 = -0, where FClamp gives +0.
    // (capsuleCapsuleContact cannot show it: c = +0 and f = -0 put r = s0 - s1 orthogonal to both axes, and a
    // zero lane of vv that could carry the flipped s/t forces r = 0 -- the eps branch, whose point adds the
    // +0 posOff lane and whose normal ignores s/t.)
    const float sQ = temp / denom;
    const float s0Lo = (0.0f > sQ) ? 0.0f : sQ;
    const float s0 = (1.0f < s0Lo) ? 1.0f : s0Lo;
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    // P9.86: distanceSegmentSegment.cuh:76 PxClamp(tTmp, 0, 1) = PxMin(1, PxMax(0, tTmp)) with the
    // float selects (selectMax = a > b ? a : b, selectMin = a < b ? a : b): tTmp = -0 (eRecip == 0,
    // negative bracket) stays -0 as in the CUDA, where FClamp would give +0.
    const float t2Lo = (0.0f > tTmp) ? 0.0f : tTmp;
    const float t2 = (1.0f < t2Lo) ? 1.0f : t2Lo;
    const float comp = (b * t2 - c) * aRecip;
    // P9.86: distanceSegmentSegment.cuh:81 PxClamp(comp, 0.f, 1.f) = PxMin(1, PxMax(0, comp)) with the
    // selects of PxUnixMathIntrinsics.h:116-125 -- keeps comp = -0 (aRecip = 0 and a negative
    // bracket) and NaN, which FClamp on llvmpipe turns into +0 / 0.
    const float s2Lo = (0.0f > comp) ? 0.0f : comp;
    const float s2 = (1.0f < s2Lo) ? 1.0f : s2Lo;
    s = s2;
    t = t2;
}

// mirrors sphereCollision.cuh pcmDistancePointSegmentTValue22
void pcmDistancePointSegmentTValue22(float3 a0, float3 b0, float3 a1, float3 b1,
                                     float3 p0, float3 p1, float3 p2, float3 p3,
                                     out float t0, out float t1, out float t2, out float t3)
{
    const float3 ap00 = p0 - a0;
    const float3 ap10 = p1 - a0;
    const float3 ap01 = p2 - a1;
    const float3 ap11 = p3 - a1;
    const float3 ab0 = b0 - a0;
    const float3 ab1 = b1 - a1;
    const float nom00 = dot(ap00, ab0);
    const float nom10 = dot(ap10, ab0);
    const float nom01 = dot(ap01, ab1);
    const float nom11 = dot(ap11, ab1);
    const float denom0 = dot(ab0, ab0);
    const float denom1 = dot(ab1, ab1);
    const float eps = 1e-6f;
    t0 = (abs(denom0) < eps) ? 0.0f : (nom00 / denom0);
    t1 = (abs(denom0) < eps) ? 0.0f : (nom10 / denom0);
    t2 = (abs(denom1) < eps) ? 0.0f : (nom01 / denom1);
    t3 = (abs(denom1) < eps) ? 0.0f : (nom11 / denom1);
}

// ---------------------------------------------------------------------------
// 0 = sphere (q0,p0,radius0), 1 = sphere (q1,p1,radius1)
// ---------------------------------------------------------------------------
public PrimContact sphereSphereContact(float4 q0, float3 p0, float radius0,
                                       float4 q1, float3 p1, float radius1, float contactDist)
{
    PrimContact c = emptyContact();
    const float3 delta = p0 - p1;
    const float distanceSq = dot(delta, delta);
    const float radiusSum = radius0 + radius1;
    const float inflatedSum = radiusSum + contactDist;

    if ((inflatedSum * inflatedSum) > distanceSq)
    {
        const float eps = 1e-4f;
        const float dist = sqrt(distanceSq);
        const float3 normal = (eps >= dist) ? float3(1.0f, 0.0f, 0.0f) : (delta * (1.0f / dist));
        const float3 point = p1 + (normal * radius1);
        const float pen = dist - radiusSum;
        c.pointPen0 = float4(point, pen);
        c.normal = normal;
        c.nbContacts = 1u;
    }
    return c;
}

// ---------------------------------------------------------------------------
// 0 = sphere (q0,p0,radius0), 1 = plane (q1,p1)
// ---------------------------------------------------------------------------
public PrimContact spherePlaneContact(float4 q0, float3 p0, float radius0,
                                      float4 q1, float3 p1, float contactDist)
{
    PrimContact c = emptyContact();
    const float3 sphereCenterInPlaneSpace = xformInv(q1, p1, p0);
    const float separation = sphereCenterInPlaneSpace.x - radius0;

    if (contactDist >= separation)
    {
        const float3 worldNormal = quatBasisVector0(q1);
        const float3 worldPoint = p0 - worldNormal * radius0;
        c.pointPen0 = float4(worldPoint, separation);
        c.normal = worldNormal;
        c.nbContacts = 1u;
    }
    return c;
}

// ---------------------------------------------------------------------------
// 0 = sphere (q0,p0,sphereRadius), 1 = capsule (q1,p1,capsuleRadius,halfHeight)
// ---------------------------------------------------------------------------
public PrimContact sphereCapsuleContact(float4 q0, float3 p0, float sphereRadius,
                                        float4 q1, float3 p1, float capsuleRadius, float halfHeight,
                                        float contactDist)
{
    PrimContact c = emptyContact();
    const float3 sphereCenter = p0;
    const float3 basisVector0 = quatBasisVector0(q1);
    const float3 tmp0 = basisVector0 * halfHeight;
    const float3 segS = p1 + tmp0;
    const float3 segE = p1 - tmp0;

    const float radiusSum = sphereRadius + capsuleRadius;
    const float inflatedSum = radiusSum + contactDist;

    float t;
    const float squareDist = distancePointSegmentSquared(segS, segE, sphereCenter, t);
    const float sqInflatedSum = inflatedSum * inflatedSum;

    if (sqInflatedSum > squareDist)
    {
        const float3 p = segS + (segE - segS) * t;
        const float3 dir = sphereCenter - p;
        const float length = sqrt(dot(dir, dir));
        const float3 normal = length > PX_EPS_REAL ? (dir * (1.0f / length)) : float3(1.0f, 0.0f, 0.0f);
        const float3 point = sphereCenter - normal * sphereRadius;
        const float pen = sqrt(squareDist) - radiusSum;
        c.pointPen0 = float4(point, pen);
        c.normal = normal;
        c.nbContacts = 1u;
    }
    return c;
}

// ---------------------------------------------------------------------------
// 0 = sphere (q0,p0,sphereRadius), 1 = box (q1,p1,boxExtents)
// ---------------------------------------------------------------------------
public PrimContact sphereBoxContact(float4 q0, float3 p0, float sphereRadius,
                                    float4 q1, float3 p1, float3 boxExtents, float contactDist)
{
    PrimContact c = emptyContact();
    const float3 sphereOrigin = p0;
    const float3 sphereCenter = xformInv(q1, p1, sphereOrigin);
    const float3 nBoxExtents = -boxExtents;

    const float inflatedSum = sphereRadius + contactDist;
    const float sqInflatedSum = inflatedSum * inflatedSum;

    const float px = clamp(sphereCenter.x, nBoxExtents.x, boxExtents.x);
    const float py = clamp(sphereCenter.y, nBoxExtents.y, boxExtents.y);
    const float pz = clamp(sphereCenter.z, nBoxExtents.z, boxExtents.z);
    const float3 p = float3(px, py, pz);
    const float3 v = sphereCenter - p;
    const float lengthSq = dot(v, v);

    if (sqInflatedSum > lengthSq)
    {
        const bool bX = boxExtents.x >= abs(sphereCenter.x);
        const bool bY = boxExtents.y >= abs(sphereCenter.y);
        const bool bZ = boxExtents.z >= abs(sphereCenter.z);
        const bool bInsideBox = bX && bY && bZ;

        if (bInsideBox)
        {
            const float3 absP = float3(abs(p.x), abs(p.y), abs(p.z));
            const float3 distToSurface = boxExtents - absP;

            float3 localNorm = p.z >= 0.0f ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 0.0f, -1.0f);
            float dist = -distToSurface.z;
            if (distToSurface.x <= distToSurface.y && distToSurface.x <= distToSurface.z)
            {
                localNorm = p.x >= 0.0f ? float3(1.0f, 0.0f, 0.0f) : float3(-1.0f, 0.0f, 0.0f);
                dist = -distToSurface.x;
            }
            else if (distToSurface.y <= distToSurface.z)
            {
                localNorm = p.y >= 0.0f ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, -1.0f, 0.0f);
                dist = -distToSurface.y;
            }

            const float3 normal = quatRotate(q1, localNorm);
            const float penetration = dist - sphereRadius;
            const float3 point = sphereOrigin - normal * dist;
            c.pointPen0 = float4(point, penetration);
            c.normal = normal;
        }
        else
        {
            const float recipLength = rsqrt(lengthSq);
            const float length = 1.0f / recipLength;
            const float3 locNorm = v * recipLength;
            const float penetration = length - sphereRadius;
            const float3 normal = quatRotate(q1, locNorm);
            const float3 point = xformPt(q1, p1, p);
            c.pointPen0 = float4(point, penetration);
            c.normal = normal;
        }
        c.nbContacts = 1u;
    }
    return c;
}

// ---------------------------------------------------------------------------
// 0 = plane (q0,p0), 1 = capsule (q1,p1,radius,halfHeight).  Up to two contacts.
// ---------------------------------------------------------------------------
public PrimContact planeCapsuleContact(float4 q0, float3 p0, float4 q1, float3 p1,
                                       float radius, float halfHeight, float contactDist)
{
    PrimContact c = emptyContact();

    // bToA = transform0.transformInv(transform1)  (sphereCollision.cuh:527).
    // P9.94b: PxTransform::transformInv (PxTransform.h:162-168) is
    // `qinv = q.getConjugate(); (qinv.rotate(src.p - p), qinv * src.q)` -- the
    // conjugate's FORWARD rotate, not q.rotateInv: the two differ in the sign
    // of a zero lane (the cross term enters negated vs. with negated factors).
    const float4 qinv = quatConjugate(q0);
    const float4 bToAq = quatMul(qinv, q1);
    const float3 bToAp = quatRotate(qinv, p1 - p0);

    const float3 planeNormal = normalize3(quatBasisVector0(q0));
    const float3 outN = -planeNormal;

    const float3 basisVector = quatBasisVector0(bToAq);
    const float3 tmp = basisVector * halfHeight;
    const float3 segS = bToAp + tmp;
    const float3 segE = bToAp - tmp;

    const float inflatedRadius = radius + contactDist;
    uint numContacts = 0u;

    const float signDist0 = segS.x;
    if (inflatedRadius >= signDist0)
    {
        const float3 worldPoint = xformPt(q0, p0, segS) - planeNormal * radius;
        c.pointPen0 = float4(worldPoint, signDist0 - radius);
        numContacts++;
    }
    const float signDist1 = segE.x;
    if (inflatedRadius >= signDist1)
    {
        const float3 worldPoint = xformPt(q0, p0, segE) - planeNormal * radius;
        if (numContacts == 0u)
            c.pointPen0 = float4(worldPoint, signDist1 - radius);
        else
            c.pointPen1 = float4(worldPoint, signDist1 - radius);
        numContacts++;
    }

    c.normal = outN;
    c.nbContacts = numContacts;
    return c;
}

// ---------------------------------------------------------------------------
// 0 = capsule (q0,p0,radius0,halfHeight0), 1 = capsule (q1,p1,radius1,halfHeight1).
// Up to two contacts.
// ---------------------------------------------------------------------------
public PrimContact capsuleCapsuleContact(float4 q0, float3 p0, float radius0, float halfHeight0,
                                         float4 q1, float3 p1, float radius1, float halfHeight1,
                                         float contactDist)
{
    PrimContact c = emptyContact();

    const float3 positionOffset = (p0 + p1) * 0.5f;
    const float3 pp0 = p0 - positionOffset;
    const float3 pp1 = p1 - positionOffset;

    const float3 tmp0 = quatBasisVector0(q0) * halfHeight0;
    const float3 s0 = pp0 + tmp0;
    const float3 e0 = pp0 - tmp0;
    const float3 d0 = e0 - s0;

    const float3 tmp1 = quatBasisVector0(q1) * halfHeight1;
    const float3 s1 = pp1 + tmp1;
    const float3 e1 = pp1 - tmp1;
    const float3 d1 = e1 - s1;

    const float sumRadius = radius0 + radius1;
    const float inflatedSum = sumRadius + contactDist;
    const float inflatedSumSquared = inflatedSum * inflatedSum;
    const float a = dot(d0, d0);
    const float e = dot(d1, d1);
    const float eps = 1e-6f;

    float t0v, t1v;
    distanceSegmentSegmentSquared(s0, d0, s1, d1, t0v, t1v);

    const float3 closestA = s0 + d0 * t0v;
    const float3 closestB = s1 + d1 * t1v;
    const float3 vv = closestA - closestB;
    const float sqDist0 = dot(vv, vv);

    uint numContacts = 0u;
    float3 outN = float3(0.0f, 0.0f, 0.0f);

    if (eps > sqDist0)
    {
        const float3 normal = normalize3(quatRotateInv(q0, float3(0.0f, 1.0f, 0.0f)));
        const float3 pt = (closestA - normal * radius0) + positionOffset;
        c.pointPen0 = float4(pt, -radius0);
        outN = normal;
        numContacts = 1u;
    }
    else if (inflatedSumSquared >= sqDist0)
    {
        const float parallelTolerance = 0.9998f;
        const float3 dir0 = (eps > a) ? float3(0.0f, 0.0f, 0.0f) : (d0 * (1.0f / sqrt(a)));
        const float3 dir1 = (eps > e) ? float3(0.0f, 0.0f, 0.0f) : (d1 * (1.0f / sqrt(e)));
        const float cosv = abs(dot(dir0, dir1));

        if (cosv > parallelTolerance)
        {
            float tV0, tV1, tV2, tV3;
            pcmDistancePointSegmentTValue22(s0, e0, s1, e1, s1, e1, s0, e0, tV0, tV1, tV2, tV3);

            if (tV0 >= 0.0f && tV0 <= 1.0f)
            {
                const float3 projS1 = s0 + d0 * tV0;
                const float3 vc = projS1 - s1;
                const float dist = sqrt(dot(vc, vc));
                // P9.36: sphereCollision.cuh:733/755/778/799 are `const PxVec3 normal =
                // v / dist;`, annotated `V3ScaleInv(v, dist)` -- PxVec3::operator/
                // (PxVec3.h:219) is ONE reciprocal and three multiplies.
                const float3 normal = vc * (1.0f / dist);
                const float3 pt = (projS1 - normal * radius0) + positionOffset;
                if (numContacts == 0u) c.pointPen0 = float4(pt, dist - sumRadius);
                else                   c.pointPen1 = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
            if (tV1 >= 0.0f && tV1 <= 1.0f)
            {
                const float3 projE1 = s0 + d0 * tV1;
                const float3 vc = projE1 - e1;
                const float dist = sqrt(dot(vc, vc));
                // P9.36: sphereCollision.cuh:733/755/778/799 are `const PxVec3 normal =
                // v / dist;`, annotated `V3ScaleInv(v, dist)` -- PxVec3::operator/
                // (PxVec3.h:219) is ONE reciprocal and three multiplies.
                const float3 normal = vc * (1.0f / dist);
                const float3 pt = (projE1 - normal * radius0) + positionOffset;
                if (numContacts == 0u) c.pointPen0 = float4(pt, dist - sumRadius);
                else                   c.pointPen1 = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
            if (numContacts < 2u && (tV2 >= 0.0f && tV2 <= 1.0f))
            {
                const float3 projS0 = s1 + d1 * tV2;
                const float3 vc = s0 - projS0;
                const float dist = sqrt(dot(vc, vc));
                // P9.36: sphereCollision.cuh:733/755/778/799 are `const PxVec3 normal =
                // v / dist;`, annotated `V3ScaleInv(v, dist)` -- PxVec3::operator/
                // (PxVec3.h:219) is ONE reciprocal and three multiplies.
                const float3 normal = vc * (1.0f / dist);
                const float3 pt = (s0 - normal * radius0) + positionOffset;
                if (numContacts == 0u) c.pointPen0 = float4(pt, dist - sumRadius);
                else                   c.pointPen1 = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
            if (numContacts < 2u && (tV3 >= 0.0f && tV3 <= 1.0f))
            {
                const float3 projE0 = s1 + d1 * tV3;
                const float3 vc = e0 - projE0;
                const float dist = sqrt(dot(vc, vc));
                // P9.36: sphereCollision.cuh:733/755/778/799 are `const PxVec3 normal =
                // v / dist;`, annotated `V3ScaleInv(v, dist)` -- PxVec3::operator/
                // (PxVec3.h:219) is ONE reciprocal and three multiplies.
                const float3 normal = vc * (1.0f / dist);
                const float3 pt = (e0 - normal * radius0) + positionOffset;
                if (numContacts == 0u) c.pointPen0 = float4(pt, dist - sumRadius);
                else                   c.pointPen1 = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
        }

        if (numContacts == 0u)
        {
            const float3 normal = normalize3(vv);
            const float3 pt = (closestA - normal * radius0) + positionOffset;
            const float dist = sqrt(sqDist0);
            c.pointPen0 = float4(pt, dist - sumRadius);
            outN = normal;
            numContacts = 1u;
        }
    }

    c.normal = outN;
    c.nbContacts = numContacts;
    return c;
}
