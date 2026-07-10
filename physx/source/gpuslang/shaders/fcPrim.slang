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

// fcPrim.slang
//
// Slang port of the FEM-cloth vertex/triangle primitive contact cores
// (gpunarrowphase/src/CUDA/femClothPrimitives.cu):
//
// - vertPlaneCollide: cloth vertex vs plane — transform the vertex into
//   plane space, gate contactDist >= x, normal = the plane pose's basis
//   vector 0, penetration = the plane-space x coordinate, and the caller's
//   contact-point derivation rigidP = clothP - pen * n
//   (cloth_planeVertContactGenLaunch) folded in.
// - triSphereCollide (the live path of fcSphereCollision; the vertex-vs-
//   sphere prelude is compiled out with `if (0)` upstream): closest point on
//   the triangle to the sphere center via the barycentric
//   closestPtPointTriangle (deformableCollision.cuh, s/t out), gate
//   sqInflatedRadius > sqDist with inflatedRadius = radius + contactDist,
//   normal = v/dist guarded at 1e-8 with the triangle surface normal as the
//   degenerate fallback, contact = sphereCenter + n * radius,
//   pen = dist - radius, barycentric = (1-s-t, s, t, 0).
// - vertCapsuleCollide: cloth vertex vs capsule — segment from
//   pose.p - dir*halfHeight spanned along basis vector 0 with
//   len = 2*halfHeight (closestPtPointSegment clamps the projection),
//   nor = a - closest with dist = PxVec3::normalize() (m > 0 guarded),
//   gate dist < capsuleRadius + contactDist, pen = dist - capsuleRadius,
//   point = the vertex itself.
//
// The fcSphereCollision / driver wrappers (PxgFEMCloth loads, transform
// caches) are glue; work items carry the triangle/vertex and shape poses
// directly.  Modes: 0 = vertPlane, 1 = triSphere, 2 = vertCapsule.

import pxgSlangCommon;

struct FcParams { uint numCases; uint pad0; uint pad1; uint pad2; }

struct FcWork
{
    float4 a;      // vertex / triangle vert 0 (w = contactDist)
    float4 b;      // triangle vert 1
    float4 c;      // triangle vert 2
    float4 shapeP; // shape pose position (w = radius / halfHeight)
    float4 shapeQ; // shape pose quaternion
    uint4 ids;     // x = mode, y: capsule radius bits (asfloat) for mode 2
}

struct FcOut
{
    float4 point;
    float4 normalPen;
    float4 barycentric;
    uint4 count; // x = contact count
}

[[vk::push_constant]]
ConstantBuffer<FcParams> gParams;

RWStructuredBuffer<FcWork> gWork; // binding 0
RWStructuredBuffer<FcOut> gOut;   // binding 1

// PxQuat::getBasisVector0 (specialized form)
float3 getBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// deformableCollision.cuh closestPtPointTriangle with barycentric s/t out
float3 closestPtPointTriangleST(float3 p, float3 a, float3 b, float3 c, out float s, out float t)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return a; }

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return b; }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        s = v; t = 0.0f;
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return c; }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        s = 0.0f; t = w;
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        s = 1.0f - w; t = w;
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    s = v; t = w;
    return a + ab * v + ac * w;
}

float3 closestPtPointSegment(float3 p0, float3 dir, float len, float3 point)
{
    float diff = dot(point - p0, dir);
    diff = clamp(diff, 0.0f, len);
    return p0 + dir * diff;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctFcPrim(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gParams.numCases) return;
    const FcWork w = gWork[i];
    const float contactDist = w.a.w;

    FcOut o;
    o.point = float4(0, 0, 0, 0);
    o.normalPen = float4(0, 0, 0, 0);
    o.barycentric = float4(0, 0, 0, 0);
    uint count = 0;

    switch (w.ids.x)
    {
    case 0: // vertPlaneCollide
    {
        const float3 pInPlaneSpace = quatRotateInv(w.shapeQ, w.a.xyz - w.shapeP.xyz);
        const float3 normal = getBasisVector0(w.shapeQ);
        if (contactDist >= pInPlaneSpace.x)
        {
            o.normalPen = float4(normal, pInPlaneSpace.x);
            // the caller derives the rigid contact point by projecting the
            // vertex onto the plane along the world normal
            // (cloth_planeVertContactGenLaunch: rigidP = clothP - pen * n)
            o.point = float4(w.a.xyz - pInPlaneSpace.x * normal, 0.0f);
            count = 1;
        }
        break;
    }
    case 1: // triSphereCollide (radius = shapeP.w, center = shapeP.xyz)
    {
        const float3 sphereCenter = w.shapeP.xyz;
        const float radius = w.shapeP.w;
        const float inflatedRadius = radius + contactDist;
        const float sqInflatedRadius = inflatedRadius * inflatedRadius;

        float s, t;
        const float3 closestPt = closestPtPointTriangleST(sphereCenter, w.a.xyz, w.b.xyz, w.c.xyz, s, t);
        const float3 v = closestPt - sphereCenter;
        const float sqDist = dot(v, v);

        if (sqInflatedRadius > sqDist)
        {
            float3 n;
            const float dist = sqrt(sqDist);
            if (dist > 1e-8f)
            {
                const float recipMag = 1.0f / dist;
                n = v * recipMag;
            }
            else
            {
                const float3 ab = w.b.xyz - w.a.xyz;
                const float3 ac = w.c.xyz - w.a.xyz;
                n = getNormalized(cross(ab, ac)); // triangle surface normal
            }
            const float3 contact = sphereCenter + n * radius;
            o.point = float4(contact, 0.0f);
            o.normalPen = float4(n, dist - radius);
            o.barycentric = float4(1.0f - s - t, s, t, 0.0f);
            count = 1;
        }
        break;
    }
    default: // 2: vertCapsuleCollide (halfHeight = shapeP.w, radius = asfloat(ids.y))
    {
        const float halfHeight = w.shapeP.w;
        const float capsuleRadius = asfloat(w.ids.y);
        const float3 dir = getBasisVector0(w.shapeQ);
        const float3 tmp = dir * halfHeight;
        const float3 q = w.shapeP.xyz - tmp;
        const float inflatedRadius = capsuleRadius + contactDist;
        const float len = halfHeight * 2.0f;
        const float3 closest0 = closestPtPointSegment(q, dir, len, w.a.xyz);

        float3 nor0 = w.a.xyz - closest0;
        // PxVec3::normalize (m > 0 guarded); returns the length
        const float m = dot(nor0, nor0);
        const float dist0 = sqrt(m);
        if (dist0 > 0.0f)
            nor0 = nor0 * (1.0f / dist0);

        if (dist0 < inflatedRadius)
        {
            o.point = float4(w.a.xyz, 0.0f);
            o.normalPen = float4(nor0, dist0 - capsuleRadius);
            count = 1;
        }
        break;
    }
    }

    o.count = uint4(count, 0, 0, 0);
    gOut[i] = o;
}
