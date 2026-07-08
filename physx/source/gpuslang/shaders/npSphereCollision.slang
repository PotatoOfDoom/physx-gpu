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

// npSphereCollision.slang
//
// Slang port of the single-contact sphere-primitive collision routines from
// gpunarrowphase/src/CUDA/sphereCollision.cuh (used by cudaSphere.cu's
// sphereNphase_Kernel): spheresphere, sphereplane, spherebox and spherecapsule.
// Each produces at most one contact (world point + penetration in outPointPen,
// world normal in outNormal, returning 0/1).  Here each routine is exposed as a
// batch compute entry point that runs one test per thread so the geometry can be
// verified directly against a C++ reference; the contact-stream output machinery
// (atomic patch/point allocation, material combining) is a later tranche.
//
// Shapes: sphere at transform0.p with radius sphereRadius; the second shape is at
// transform1 (a sphere of radius1, an x-axis plane, a box of half-extents
// boxExtents.xyz, or an x-axis capsule of radius1/halfHeight).

import pxgSlangCommon;

struct Xform
{
    float4 p; // .xyz position
    float4 q; // quaternion (x,y,z,w)
}

struct SphereTest
{
    float sphereRadius;
    float contactDist;
    float radius1;    // sphere r1 / capsule radius
    float halfHeight; // capsule half height
    float4 boxExtents;
}

struct SphereParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SphereParams> gParams;

RWStructuredBuffer<Xform>       gXform0;      // binding 0
RWStructuredBuffer<Xform>       gXform1;      // binding 1
RWStructuredBuffer<SphereTest>  gTests;       // binding 2
RWStructuredBuffer<float4>      gOutPointPen; // binding 3  (world point, penetration)
RWStructuredBuffer<float4>      gOutNormal;   // binding 4  (world normal, w unused)
RWStructuredBuffer<uint>        gNbContacts;  // binding 5

static const float PX_EPS_REAL = 1.192092896e-07f;

float3 xformInv(Xform t, float3 pt) { return quatRotateInv(t.q, pt - t.p.xyz); }
float3 xformPt(Xform t, float3 pt) { return quatRotate(t.q, pt) + t.p.xyz; }

// mirrors sphereCollision.cuh distancePointSegmentSquared
float distancePointSegmentSquared(float3 a, float3 b, float3 p, out float param)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    const float tValue = clamp(nom / denom, 0.0f, 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    const float3 v = ap - (ab * t);
    param = t;
    return dot(v, v);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_sphereSphere(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const float3 p0 = gXform0[i].p.xyz;
    const float3 p1 = gXform1[i].p.xyz;

    const float3 delta = p0 - p1;
    const float distanceSq = dot(delta, delta);
    const float radiusSum = s.sphereRadius + s.radius1;
    const float inflatedSum = radiusSum + s.contactDist;

    uint nb = 0;
    if ((inflatedSum * inflatedSum) > distanceSq)
    {
        const float eps = 1e-4f;
        const float dist = sqrt(distanceSq);
        const float3 normal = (eps >= dist) ? float3(1.0f, 0.0f, 0.0f) : (delta * (1.0f / dist));
        const float3 point = p1 + (normal * s.radius1);
        const float pen = dist - radiusSum;
        gOutPointPen[i] = float4(point, pen);
        gOutNormal[i] = float4(normal, 0.0f);
        nb = 1;
    }
    gNbContacts[i] = nb;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_spherePlane(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const float3 sphereCenterInPlaneSpace = xformInv(t1, t0.p.xyz);
    const float separation = sphereCenterInPlaneSpace.x - s.sphereRadius;

    uint nb = 0;
    if (s.contactDist >= separation)
    {
        const float3 worldNormal = quatBasisVector0(t1.q);
        const float3 worldPoint = t0.p.xyz - worldNormal * s.sphereRadius;
        gOutPointPen[i] = float4(worldPoint, separation);
        gOutNormal[i] = float4(worldNormal, 0.0f);
        nb = 1;
    }
    gNbContacts[i] = nb;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_sphereCapsule(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const float3 sphereCenter = t0.p.xyz;
    const float3 basisVector0 = quatBasisVector0(t1.q);
    const float3 tmp0 = basisVector0 * s.halfHeight;
    const float3 segS = t1.p.xyz + tmp0;
    const float3 segE = t1.p.xyz - tmp0;

    const float radiusSum = s.sphereRadius + s.radius1;
    const float inflatedSum = radiusSum + s.contactDist;

    float t;
    const float squareDist = distancePointSegmentSquared(segS, segE, sphereCenter, t);
    const float sqInflatedSum = inflatedSum * inflatedSum;

    uint nb = 0;
    if (sqInflatedSum > squareDist)
    {
        const float3 p = segS + (segE - segS) * t;
        const float3 dir = sphereCenter - p;
        const float length = sqrt(dot(dir, dir));
        const float3 normal = length > PX_EPS_REAL ? (dir * (1.0f / length)) : float3(1.0f, 0.0f, 0.0f);
        const float3 point = sphereCenter - normal * s.sphereRadius;
        const float pen = sqrt(squareDist) - radiusSum;
        gOutPointPen[i] = float4(point, pen);
        gOutNormal[i] = float4(normal, 0.0f);
        nb = 1;
    }
    gNbContacts[i] = nb;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_sphereBox(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];
    const float3 boxExtents = s.boxExtents.xyz;

    const float3 sphereOrigin = t0.p.xyz;
    const float3 sphereCenter = xformInv(t1, sphereOrigin);
    const float3 nBoxExtents = -boxExtents;

    const float inflatedSum = s.sphereRadius + s.contactDist;
    const float sqInflatedSum = inflatedSum * inflatedSum;

    const float px = clamp(sphereCenter.x, nBoxExtents.x, boxExtents.x);
    const float py = clamp(sphereCenter.y, nBoxExtents.y, boxExtents.y);
    const float pz = clamp(sphereCenter.z, nBoxExtents.z, boxExtents.z);
    const float3 p = float3(px, py, pz);
    const float3 v = sphereCenter - p;
    const float lengthSq = dot(v, v);

    uint nb = 0;
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

            const float3 normal = quatRotate(t1.q, localNorm);
            const float penetration = dist - s.sphereRadius;
            const float3 point = sphereOrigin - normal * dist;
            gOutPointPen[i] = float4(point, penetration);
            gOutNormal[i] = float4(normal, 0.0f);
        }
        else
        {
            const float recipLength = rsqrt(lengthSq);
            const float length = 1.0f / recipLength;
            const float3 locNorm = v * recipLength;
            const float penetration = length - s.sphereRadius;
            const float3 normal = quatRotate(t1.q, locNorm);
            const float3 point = xformPt(t1, p);
            gOutPointPen[i] = float4(point, penetration);
            gOutNormal[i] = float4(normal, 0.0f);
        }
        nb = 1;
    }
    gNbContacts[i] = nb;
}
