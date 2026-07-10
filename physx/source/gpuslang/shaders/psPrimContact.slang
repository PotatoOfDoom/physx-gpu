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

// psPrimContact.slang
//
// Slang port of the particle-vs-primitive contact core
// (gpunarrowphase/src/CUDA/particleCollision.cuh + the
// particlePrimitiveCollision dispatcher of cudaParticleSystem.cu):
//
// - contactPointBox: transform into box space, clamp to the extents, the
//   maxDist rejection (only when maxDist < PX_MAX_F32), the inside branch
//   selecting the closest FACE (z default, then x if smallest, else y if
//   smaller than z) with the negative embedded distance, the outside branch
//   normal = rel - closest with dist = normalizeFast() (= magnitude, no
//   zero guard), and the rotation of the local normal back to world.
// - contactPointSphere: center distance vs maxDist + radius, normal from
//   the center delta guarded by PX_EPS_REAL (fallback (1,0,0)),
//   dist = centerDist - radius.
// - contactPointCapsule: segment param via distancePointSegmentSquared
//   (t = clamp(nom/denom) if denom > 0 else 0), distSegmentVec = rel -
//   axis*(2t - 1) with axis = dir * halfHeight, same eps-guarded normal,
//   dist = segmentDist - radius.
// - particleTriangleTest: triangle normal (normalized cross), the contact
//   volume tests (plane distance < cVolumeRadius, then squared closest
//   distance <= cVolumeRadius^2 via closestPtPointTriangle2), back-face
//   cull against the CURRENT particle position (> 0), the CCD re-evaluation
//   against the particle position, normal = distVec/sqrt(distSq) or the
//   triangle normal at distSq == 0, distance = sqrt(distSq).
// - the dispatcher's PLANE case: PxPlane(1,0,0,0).transform(pose) — normal
//   = pose rotation's basis vector 0, d = -n.p — gated on plane distance of
//   the contact-volume position <= cVolumeRadius, with the CCD distance
//   replacement.
// - CCD semantics throughout: the contact-volume test decides intersection,
//   then dist/normal are recomputed from the CURRENT particle position with
//   maxDist = PX_MAX_F32 (which disables the early rejection: the CUDA
//   checks maxDist < PX_MAX_F32).
//
// The dispatcher's eCONVEXCORE case (particleConvexCore) rides on
// Gu::RefGjkEpa::computeGjkDistance/computeEpaDepth (the scalar per-thread
// reference GJK/EPA) and is DEFERRED alongside the GJK/EPA cores (see
// MIGRATION_STATUS.md).
//
// Shape types here: 0 = plane, 1 = sphere, 2 = capsule, 3 = box,
// 4 = triangle (the engine's PxGeometryType dispatch is driver glue).

import pxgSlangCommon;

static const float PX_MAX_F32 = 3.4028235e38f;
static const float PX_EPS_REAL = 1.192092896e-07f;

struct PcParams { uint numCases; uint pad0; uint pad1; uint pad2; }

struct PcWork
{
    float4 particlePos; // xyz current particle position
    float4 cVolume;     // xyz contact-volume position, w radius
    float4 shapeP;      // pose position | triangle v0
    float4 shapeQ;      // pose quaternion | triangle v1 in xyz + w unused
    float4 scale;       // shape scale (sphere r=x, capsule hh=x r=y, box he=xyz) | triangle v2
    uint4 ids;          // x = type (0 plane, 1 sphere, 2 capsule, 3 box, 4 triangle), y = enableCCD
}

struct PcOut
{
    float4 normal_dist; // xyz normal, w dist
    uint4 flags;        // x = intersect
}

[[vk::push_constant]]
ConstantBuffer<PcParams> gParams;

RWStructuredBuffer<PcWork> gWork;  // binding 0
RWStructuredBuffer<PcOut> gOut;    // binding 1

// PxQuat::getBasisVector0 (the specialized form, not quatRotate(q, (1,0,0)))
float3 getBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return a + v * ab; }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return a + w * ac; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b); }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

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

float distancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, out float3 closestP)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    closestP = closestPtPointTriangle2(p, a, b, c, ab, ac);
    const float3 v = p - closestP;
    return dot(v, v);
}

bool contactPointBox(inout float3 normal, inout float dist,
                     float3 pointPos, float3 boxP, float4 boxQ, float3 halfBoxExtents, float maxDist)
{
    const float3 relPointPos = quatRotateInv(boxQ, pointPos - boxP);

    const float3 closestPos = clamp(relPointPos, -halfBoxExtents, halfBoxExtents);

    if (maxDist < PX_MAX_F32)
    {
        const float maxDistSq = maxDist * maxDist;
        const float3 dv = relPointPos - closestPos;
        const float distSq = dot(dv, dv);
        if (distSq > maxDistSq)
            return false;
    }

    const bool bX = halfBoxExtents.x >= abs(relPointPos.x);
    const bool bY = halfBoxExtents.y >= abs(relPointPos.y);
    const bool bZ = halfBoxExtents.z >= abs(relPointPos.z);

    const bool isInside = bX && bY && bZ;
    if (isInside)
    {
        const float3 distToSurface = halfBoxExtents - abs(closestPos);

        normal = closestPos.z >= 0.0f ? float3(0, 0, 1) : float3(0, 0, -1);
        dist = -distToSurface.z;
        if (distToSurface.x <= distToSurface.y && distToSurface.x <= distToSurface.z)
        {
            normal = closestPos.x >= 0.0f ? float3(1, 0, 0) : float3(-1, 0, 0);
            dist = -distToSurface.x;
        }
        else if (distToSurface.y <= distToSurface.z)
        {
            normal = closestPos.y >= 0.0f ? float3(0, 1, 0) : float3(0, -1, 0);
            dist = -distToSurface.y;
        }
    }
    else
    {
        normal = relPointPos - closestPos;
        const float mag = sqrt(dot(normal, normal)); // PxVec3::normalizeFast
        normal = normal * (1.0f / mag);
        dist = mag;
    }
    normal = quatRotate(boxQ, normal);
    return true;
}

bool contactPointSphere(inout float3 normal, inout float dist,
                        float3 pointPos, float3 spherePos, float sphereRadius, float maxDist)
{
    const float3 relPointPos = pointPos - spherePos;
    const float distCenterSq = dot(relPointPos, relPointPos);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistCenter = maxDist + sphereRadius;
        const float maxDistCenterSq = maxDistCenter * maxDistCenter;
        if (distCenterSq > maxDistCenterSq)
            return false;
    }
    const float distCenter = sqrt(distCenterSq);
    normal = (distCenter > PX_EPS_REAL) ? relPointPos * (1.0f / distCenter) : float3(1, 0, 0);
    dist = distCenter - sphereRadius;
    return true;
}

bool contactPointCapsule(inout float3 normal, inout float dist,
                         float3 pointPos, float3 capsulePos, float3 capsuleDir,
                         float capsuleRadius, float capsuleHalfHeight, float maxDist)
{
    const float3 capsuleAxisVec = capsuleDir * capsuleHalfHeight;
    const float3 relPointPos = pointPos - capsulePos;
    float t;
    const float distSegmentSq = distancePointSegmentSquared(-capsuleAxisVec, capsuleAxisVec, relPointPos, t);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistSegment = maxDist + capsuleRadius;
        const float maxDistSegmentSq = maxDistSegment * maxDistSegment;
        if (distSegmentSq > maxDistSegmentSq)
            return false;
    }
    const float3 distSegmentVec = relPointPos - capsuleAxisVec * (2.0f * t - 1.0f);
    const float distSegment = sqrt(distSegmentSq);
    normal = (distSegment > PX_EPS_REAL) ? (distSegmentVec * (1.0f / distSegment)) : float3(1, 0, 0);
    dist = distSegment - capsuleRadius;
    return true;
}

bool particleTriangleTest(inout float3 normal, inout float distance,
                          float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                          float3 triV0, float3 triV1, float3 triV2, bool enableCCD)
{
    const float3 cn = cross(triV1 - triV0, triV2 - triV0);
    const float m = dot(cn, cn);
    const float3 triNormal = m > 0.0f ? cn * rsqrt(m) : float3(0, 0, 0);
    const float cVolumeRadiusSq = cVolumeRadius * cVolumeRadius;

    const float cVolumePosPlaneDist = dot(cVolumePos - triV0, triNormal);
    float3 closestPos;
    if (cVolumePosPlaneDist < cVolumeRadius)
    {
        float distSq = distancePointTriangleSquared(cVolumePos, triV0, triV1, triV2, closestPos);
        if (distSq <= cVolumeRadiusSq)
        {
            // back-face culling with respect to the current particle position
            const float particlePosPlaneDist = dot(particlePos - triV0, triNormal);
            if (particlePosPlaneDist > 0.0f)
            {
                if (enableCCD)
                    distSq = distancePointTriangleSquared(particlePos, triV0, triV1, triV2, closestPos);
                // assumes particlePos == cVolumePos if enableCCD == false
                const float3 distVec = particlePos - closestPos;
                normal = (distSq > 0.0f) ? distVec * rsqrt(distSq) : triNormal;
                distance = sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctPsPrimContact(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gParams.numCases) return;
    const PcWork w = gWork[i];
    const float3 particlePos = w.particlePos.xyz;
    const float3 cVolumePos = w.cVolume.xyz;
    const float cVolumeRadius = w.cVolume.w;
    const bool enableCCD = w.ids.y != 0;

    bool intersect = false;
    float3 normal = float3(0, 0, 0);
    float dist = 0.0f;

    switch (w.ids.x)
    {
    case 0: // ePLANE: PxPlane(1,0,0,0).transform(shapeTransform)
    {
        const float3 pn = quatRotate(w.shapeQ, float3(1, 0, 0));
        const float pdd = -dot(pn, w.shapeP.xyz);
        dist = dot(pn, cVolumePos) + pdd;
        if (dist <= cVolumeRadius)
        {
            if (enableCCD)
                dist = dot(pn, particlePos) + pdd;
            normal = pn;
            intersect = true;
        }
        break;
    }
    case 1: // eSPHERE
    {
        const float3 spherePos = w.shapeP.xyz;
        const float sphereRadius = w.scale.x;
        intersect = contactPointSphere(normal, dist, cVolumePos, spherePos, sphereRadius, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointSphere(normal, dist, particlePos, spherePos, sphereRadius, PX_MAX_F32);
        break;
    }
    case 2: // eCAPSULE
    {
        const float capsuleRadius = w.scale.y;
        const float capsuleHalfHeight = w.scale.x;
        const float3 capsuleDir = getBasisVector0(w.shapeQ);
        const float3 capsulePos = w.shapeP.xyz;
        intersect = contactPointCapsule(normal, dist, cVolumePos, capsulePos, capsuleDir,
                                        capsuleRadius, capsuleHalfHeight, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointCapsule(normal, dist, particlePos, capsulePos, capsuleDir,
                                capsuleRadius, capsuleHalfHeight, PX_MAX_F32);
        break;
    }
    case 3: // eBOX
    {
        const float3 boxHalfExtents = w.scale.xyz;
        intersect = contactPointBox(normal, dist, cVolumePos, w.shapeP.xyz, w.shapeQ, boxHalfExtents, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointBox(normal, dist, particlePos, w.shapeP.xyz, w.shapeQ, boxHalfExtents, PX_MAX_F32);
        break;
    }
    default: // 4: triangle (particleTriangleTest)
    {
        intersect = particleTriangleTest(normal, dist, particlePos, cVolumePos, cVolumeRadius,
                                         w.shapeP.xyz, w.shapeQ.xyz, w.scale.xyz, enableCCD);
        break;
    }
    }

    PcOut o;
    o.normal_dist = float4(normal, dist);
    o.flags = uint4(intersect ? 1u : 0u, 0, 0, 0);
    gOut[i] = o;
}
