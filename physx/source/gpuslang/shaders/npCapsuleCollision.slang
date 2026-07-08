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

// npCapsuleCollision.slang
//
// Slang port of the two-contact capsule collision routines from
// gpunarrowphase/src/CUDA/sphereCollision.cuh (used by cudaSphere.cu's
// sphereNphase_Kernel): planeCapsule and capsuleCapsule, plus the shared
// segment-segment distance helpers (distanceSegmentSegmentSquared and
// pcmDistancePointSegmentTValue22 from distanceSegmentSegment.cuh).  Each can
// produce up to two contacts (world point + penetration per contact in
// gOutPointPen[2*i + {0,1}], a shared world normal in gOutNormal[i]).
//
// capsuleCapsule closest-features: the general case is one contact at the
// closest points of the two axis segments; near-parallel capsules (|cos| >
// 0.9998) fall back to projecting each capsule's endpoints onto the other axis
// (pcmDistancePointSegmentTValue22) to recover up to two contacts; degenerate
// (inline) segments emit a single -radius0 contact.

import pxgSlangCommon;

struct Xform
{
    float4 p; // .xyz position
    float4 q; // quaternion (x,y,z,w)
}

struct CapsuleTest
{
    float radius0;     // planeCapsule: capsule radius
    float halfHeight0; // planeCapsule: capsule half height
    float radius1;
    float halfHeight1;
    float contactDist;
    float pad0, pad1, pad2;
}

struct CapsuleParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CapsuleParams> gParams;

RWStructuredBuffer<Xform>       gXform0;      // binding 0
RWStructuredBuffer<Xform>       gXform1;      // binding 1
RWStructuredBuffer<CapsuleTest> gTests;       // binding 2
RWStructuredBuffer<float4>      gOutPointPen; // binding 3  (2 per test)
RWStructuredBuffer<float4>      gOutNormal;   // binding 4  (1 per test)
RWStructuredBuffer<uint>        gNbContacts;  // binding 5

float3 xformPt(Xform t, float3 pt) { return quatRotate(t.q, pt) + t.p.xyz; }
float3 normalize3(float3 v) { return v * rsqrt(dot(v, v)); }

// mirrors distanceSegmentSegment.cuh distanceSegmentSegmentSquared
void distanceSegmentSegmentSquared(float3 p1, float3 d1, float3 p2, float3 d2, out float s, out float t)
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
    const float s0 = clamp(temp / denom, 0.0f, 1.0f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = clamp(tTmp, 0.0f, 1.0f);
    const float comp = (b * t2 - c) * aRecip;
    const float s2 = clamp(comp, 0.0f, 1.0f);
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
// planeCapsule (0 = plane at transform0, 1 = capsule at transform1)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void planeCapsule(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const CapsuleTest ct = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];
    const float radius = ct.radius0;
    const float halfHeight = ct.halfHeight0;

    // bToA = transform0.transformInv(transform1)
    const float4 qinv = quatConjugate(t0.q);
    const float4 bToAq = quatMul(qinv, t1.q);
    const float3 bToAp = quatRotateInv(t0.q, t1.p.xyz - t0.p.xyz);

    const float3 planeNormal = normalize3(quatBasisVector0(t0.q));
    const float3 outN = -planeNormal;

    const float3 basisVector = quatBasisVector0(bToAq);
    const float3 tmp = basisVector * halfHeight;
    const float3 segS = bToAp + tmp;
    const float3 segE = bToAp - tmp;

    const float inflatedRadius = radius + ct.contactDist;
    uint numContacts = 0;

    const float signDist0 = segS.x;
    if (inflatedRadius >= signDist0)
    {
        const float3 worldPoint = xformPt(t0, segS) - planeNormal * radius;
        gOutPointPen[2u * i + numContacts] = float4(worldPoint, signDist0 - radius);
        numContacts++;
    }
    const float signDist1 = segE.x;
    if (inflatedRadius >= signDist1)
    {
        const float3 worldPoint = xformPt(t0, segE) - planeNormal * radius;
        gOutPointPen[2u * i + numContacts] = float4(worldPoint, signDist1 - radius);
        numContacts++;
    }

    if (numContacts > 0)
        gOutNormal[i] = float4(outN, 0.0f);
    gNbContacts[i] = numContacts;
}

// ---------------------------------------------------------------------------
// capsuleCapsule
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void capsuleCapsule(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const CapsuleTest ct = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];
    const float radius0 = ct.radius0, halfHeight0 = ct.halfHeight0;
    const float radius1 = ct.radius1, halfHeight1 = ct.halfHeight1;

    const float3 positionOffset = (t0.p.xyz + t1.p.xyz) * 0.5f;
    const float3 p0 = t0.p.xyz - positionOffset;
    const float3 p1 = t1.p.xyz - positionOffset;

    const float3 tmp0 = quatBasisVector0(t0.q) * halfHeight0;
    const float3 s0 = p0 + tmp0;
    const float3 e0 = p0 - tmp0;
    const float3 d0 = e0 - s0;

    const float3 tmp1 = quatBasisVector0(t1.q) * halfHeight1;
    const float3 s1 = p1 + tmp1;
    const float3 e1 = p1 - tmp1;
    const float3 d1 = e1 - s1;

    const float sumRadius = radius0 + radius1;
    const float inflatedSum = sumRadius + ct.contactDist;
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

    uint numContacts = 0;
    float3 outN = float3(0.0f, 0.0f, 0.0f);

    if (eps > sqDist0)
    {
        const float3 normal = normalize3(quatRotateInv(t0.q, float3(0.0f, 1.0f, 0.0f)));
        const float3 pt = (closestA - normal * radius0) + positionOffset;
        gOutPointPen[2u * i + 0u] = float4(pt, -radius0);
        outN = normal;
        numContacts = 1;
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
                const float3 v = projS1 - s1;
                const float dist = sqrt(dot(v, v));
                const float3 normal = v / dist;
                const float3 pt = (projS1 - normal * radius0) + positionOffset;
                gOutPointPen[2u * i + numContacts] = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
            if (tV1 >= 0.0f && tV1 <= 1.0f)
            {
                const float3 projE1 = s0 + d0 * tV1;
                const float3 v = projE1 - e1;
                const float dist = sqrt(dot(v, v));
                const float3 normal = v / dist;
                const float3 pt = (projE1 - normal * radius0) + positionOffset;
                gOutPointPen[2u * i + numContacts] = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
            if (numContacts < 2u && (tV2 >= 0.0f && tV2 <= 1.0f))
            {
                const float3 projS0 = s1 + d1 * tV2;
                const float3 v = s0 - projS0;
                const float dist = sqrt(dot(v, v));
                const float3 normal = v / dist;
                const float3 pt = (s0 - normal * radius0) + positionOffset;
                gOutPointPen[2u * i + numContacts] = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
            if (numContacts < 2u && (tV3 >= 0.0f && tV3 <= 1.0f))
            {
                const float3 projE0 = s1 + d1 * tV3;
                const float3 v = e0 - projE0;
                const float dist = sqrt(dot(v, v));
                const float3 normal = v / dist;
                const float3 pt = (e0 - normal * radius0) + positionOffset;
                gOutPointPen[2u * i + numContacts] = float4(pt, dist - sumRadius);
                outN = normal;
                numContacts++;
            }
        }

        if (numContacts == 0u)
        {
            const float3 normal = normalize3(vv);
            const float3 pt = (closestA - normal * radius0) + positionOffset;
            const float dist = sqrt(sqDist0);
            gOutPointPen[2u * i + 0u] = float4(pt, dist - sumRadius);
            outN = normal;
            numContacts = 1;
        }
    }

    if (numContacts > 0u)
        gOutNormal[i] = float4(outN, 0.0f);
    gNbContacts[i] = numContacts;
}
