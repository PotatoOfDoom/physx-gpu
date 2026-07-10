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

// fcPrim2.slang
//
// Slang port of the second femClothPrimitives.cu tranche
// (gpunarrowphase/src/CUDA/femClothPrimitives.cu):
//
// - fcCapsuleCollision (mode 0): capsule vs cloth triangle.  Only the live
//   `if(1)` path is ported — the two `if(0)` blocks (triangle vertices vs
//   capsule, triangle vs the two capsule axis points) are compiled out
//   upstream, and the meshToCapsule transform is computed but unused (dead).
//   Segment p/q = pose.p +- basisVector0 * halfHeight (halfHeight =
//   scale.x, radius = scale.y), distanceSegmentTriangleSquared (the
//   capsuleTriangle.cuh routine already ported in npSegTriangle.slang),
//   gate sqInflatedRadius > sqDist STRICT, pointOnSegment = p + pq*t,
//   pointOnTriangle from the (w,u,v) barycentrics, normal =
//   (pointOnTriangle - pointOnSegment).getNormalized() (zero fallback),
//   pen = sqrt(sqDist) - radius, contact point = pointOnSegment +
//   radius * normal, barycentric = (1-u-v, u, v, 0).
//
// - clothParticleCollision core (mode 1): particle vs cloth triangle
//   (cloth_psContactGenLaunch).  closestPtPointTriangle2
//   (GuDistancePointTriangle.h) on the particle position, gate
//   sqDist <= cDistance^2 INCLUSIVE, n = v / sqrt(sqDist) with NO zero
//   guard (faithful), pen = dist - restDistance, contact = p - n *
//   restDistance, computeBarycentric of the closest point (the returned
//   bool is only asserted upstream — the barycentric is written
//   regardless; its eps window becomes gate-relevant in the cloth-convex
//   path, ported later).  The contact goes through PxgFEMContactWriter
//   ::writeContact (PxgFEMCore.h): an UNGATED atomicAdd reservation on
//   totalContactCount, then capacity-gated stores of contactByX (the u64
//   sort key = pairInd0 here), tempContactByX (its low 32 bits),
//   contactIndexSortedByX (= index), point, normalPen (normal NEGATED:
//   -n), PxgFemOtherContactInfo{pairInd0, pairInd1} — whose fourth dword
//   (rigidMatInd_isInCollision) is NOT written (preserved) — and the
//   barycentric.  pairInd0 = PxEncodeParticleIndex(systemId, particleId)
//   = (u64(systemId) << 32) | particleId; pairInd1 =
//   PxEncodeClothIndex(clothId, triangleIdx) = (clothId << 20) | triIdx.
//
// Engine glue not ported (documented): LoadShapePair/transform-cache and
// PxgFEMCloth/PxgParticleSystem loads, the midphase pair-stack iteration
// (work items carry vertices/poses/ids directly).  Mode 1 echoes its
// contact count into gOut purely for test observability; the CUDA core
// returns void and writes only through the writer.

import pxgSlangCommon;

static const uint PXG_BITSHIFT_ELEMENT_ID = 20; // PxgSimulationCoreDesc.h

struct Fc2Params { uint numCases; uint maxContacts; uint pad0; uint pad1; }

struct Fc2Work
{
    float4 a;      // triangle vert 0 (w = contactDist)
    float4 b;      // triangle vert 1 (w = restDistance, mode 1)
    float4 c;      // triangle vert 2
    float4 shapeP; // capsule pose position (w = halfHeight) / particle position (mode 1)
    float4 shapeQ; // capsule pose quaternion (mode 0)
    uint4 ids;     // x = mode, y = capsule radius bits (mode 0) / particleId (mode 1),
                   // z = particleSystemId, w = clothId (mode 1)
    uint4 ids2;    // x = triangleIdx (mode 1)
}

struct Fc2Out
{
    float4 point;
    float4 normalPen;
    float4 barycentric;
    uint4 count; // x = contact count
}

[[vk::push_constant]]
ConstantBuffer<Fc2Params> gParams;

RWStructuredBuffer<Fc2Work> gWork;      // binding 0
RWStructuredBuffer<Fc2Out> gOut;        // binding 1
// PxgFEMContactWriter buffers (particle sorting flavour)
RWStructuredBuffer<float4> gFemPoint;   // binding 2 (outPoint)
RWStructuredBuffer<float4> gFemNormalPen; // binding 3 (outNormalPen)
RWStructuredBuffer<float4> gFemBary;    // binding 4 (outBarycentric)
RWStructuredBuffer<uint4> gFemInfo;     // binding 5 (PxgFemOtherContactInfo, 16B)
RWStructuredBuffer<uint> gFemCount;     // binding 6 (totalContactCount)
RWStructuredBuffer<uint2> gFemByX;      // binding 7 (contactByX, u64)
RWStructuredBuffer<uint> gFemTempByX;   // binding 8 (tempContactByX)
RWStructuredBuffer<uint> gFemIdxByX;    // binding 9 (contactIndexSortedByX)

// PxQuat::getBasisVector0 (specialized form)
float3 getBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

// PxVec3::getNormalized (zero fallback)
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// ---- distanceSegmentTriangleSquared (capsuleTriangle.cuh; npSegTriangle
// port) -----------------------------------------------------------------
bool isValidBary(float v, float w)
{
    const float eps = 1e-6f;
    const float zero = -eps;
    const float one = 1.0f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

void segSeg(float3 p1, float3 d1, float3 p2, float3 d2, out float s, out float t)
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
    s = clamp(comp, 0.0f, 1.0f);
    t = t2;
}

float segTriDistSq(float3 p, float3 q, float3 a, float3 b, float3 c,
                   out float outT, out float outU, out float outV)
{
    const float3 pq = q - p;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 bc = c - b;
    const float3 ap = p - a;
    const float3 aq = q - a;

    const float d00 = dot(ab, ab);
    const float d01 = dot(ab, ac);
    const float d11 = dot(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : PXG_MAX_F32;

    const float3 n = getNormalized(cross(ab, ac)); // PxVec3::getNormalized zero fallback

    const float dist3 = dot(ap, n);
    const float sqDist3 = dist3 * dist3;
    const float dist4 = dot(aq, n);
    const float sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;

    outT = 0.0f; outU = 0.0f; outV = 0.0f;

    if (dMul <= 0.0f)
    {
        const float nom = -dot(n, ap);
        const float denom = 1.0f / dot(n, pq);
        const float t0 = nom * denom;
        const float3 ip = p + pq * t0;
        const float3 v2 = ip - a;
        const float d20 = dot(v2, ab);
        const float d21 = dot(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom;
        const float w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (isValidBary(v00, w00))
        {
            outT = t0; outU = v00; outV = w00;
            return 0.0f;
        }
    }

    const float3 closestP31 = p - n * dist3;
    const float3 pV20 = closestP31 - a;
    const float pD20 = dot(pV20, ab);
    const float pD21 = dot(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = isValidBary(v0, w0);

    const float3 closestP41 = q - n * dist4;
    const float3 qV20 = closestP41 - a;
    const float qD20 = dot(qV20, ab);
    const float qD21 = dot(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = isValidBary(v1, w1);

    if (con0 && con1)
    {
        if (sqDist4 > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        outT = 1.0f; outU = v1; outV = w1; return sqDist4;
    }

    float t00, t10, t01, t11, t02, t12;
    segSeg(p, pq, a, ab, t00, t10);
    segSeg(p, pq, b, bc, t01, t11);
    segSeg(p, pq, a, ac, t02, t12);

    const float3 cp00 = p + pq * t00; const float3 cp01 = a + ab * t10;
    float3 tv = cp00 - cp01; const float sqDist0 = dot(tv, tv);
    const float3 cp10 = p + pq * t01; const float3 cp11 = b + bc * t11;
    tv = cp10 - cp11; const float sqDist1 = dot(tv, tv);
    const float3 cp20 = p + pq * t02; const float3 cp21 = a + ac * t12;
    tv = cp20 - cp21; const float sqDist2 = dot(tv, tv);

    const float u10 = t10, v10 = 0.0f;         // edge ab
    const float u11 = 1.0f - t11, v11 = t11;   // edge bc
    const float u12 = 0.0f, v12 = t12;         // edge ac

    float sqDistPE, uEdge, vEdge, tSeg;
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0) { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; }
    else if (sqDist2 > sqDist1)                 { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; }
    else                                        { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; }

    if (con0)
    {
        if (sqDistPE > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    if (con1)
    {
        if (sqDistPE > sqDist4) { outT = 1.0f; outU = v1; outV = w1; return sqDist4; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    outT = tSeg; outU = uEdge; outV = vEdge;
    return sqDistPE;
}

// Gu::closestPtPointTriangle2 (GuDistancePointTriangle.h)
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
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

// femClothPrimitives.cu computeBarycentric: plain Cramer barycentric with an
// eps acceptance window; the barycentric is produced regardless of the bool
bool computeBarycentric(float3 a, float3 b, float3 c, float3 p, out float4 barycentric)
{
    const float minEps = -1e-4f;
    const float maxEps = 1.0f + 1e-4f;
    const float3 v0 = b - a;
    const float3 v1 = c - a;
    const float3 v2 = p - a;

    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);

    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;

    barycentric = float4(u, v, w, 0.0f);

    return (u >= minEps && u <= maxEps) &&
           (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps);
}

// PxgFEMContactWriter::writeContact (PxgFEMCore.h): capacity-gated stores;
// the reservation happens at the caller via the UNGATED atomicAdd
void femWriteContact(uint index, float4 contact, float4 normalPen, float4 bary,
                     uint2 pairInd0, uint pairInd1, uint2 id)
{
    if (index >= gParams.maxContacts) return;
    gFemByX[index] = id;
    gFemTempByX[index] = id.x;
    gFemIdxByX[index] = index;
    gFemPoint[index] = contact;
    gFemNormalPen[index] = normalPen;
    // PxgFemOtherContactInfo: pairInd0 (u64) + pairInd1; the fourth dword
    // (rigidMatInd_isInCollision) is NOT touched by writeContact
    uint4 info = gFemInfo[index];
    info.x = pairInd0.x;
    info.y = pairInd0.y;
    info.z = pairInd1;
    gFemInfo[index] = info;
    gFemBary[index] = bary;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctFcPrim2(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gParams.numCases) return;
    const Fc2Work w = gWork[i];
    const float contactDist = w.a.w;
    const float3 a = w.a.xyz;
    const float3 b = w.b.xyz;
    const float3 c = w.c.xyz;

    Fc2Out o;
    o.point = float4(0, 0, 0, 0);
    o.normalPen = float4(0, 0, 0, 0);
    o.barycentric = float4(0, 0, 0, 0);
    uint count = 0;

    if (w.ids.x == 0) // fcCapsuleCollision (live path)
    {
        const float halfHeight = w.shapeP.w;
        const float capsuleRadius = asfloat(w.ids.y);
        const float3 dir = getBasisVector0(w.shapeQ);
        const float3 tmp = dir * halfHeight;
        const float3 p = w.shapeP.xyz + tmp;
        const float3 q = w.shapeP.xyz - tmp;
        const float inflatedRadius = capsuleRadius + contactDist;
        const float sqInflatedRadius = inflatedRadius * inflatedRadius;

        float t, u, v;
        const float sqDist = segTriDistSq(p, q, a, b, c, t, u, v);

        if (sqInflatedRadius > sqDist)
        {
            const float3 pq = q - p;
            const float3 pointOnSegment = p + pq * t;
            const float wBary = 1.0f - u - v;
            const float3 pointOnTriangle = a * wBary + b * u + c * v;
            const float3 normal = getNormalized(pointOnTriangle - pointOnSegment);
            const float dist = sqrt(sqDist);
            const float pen = dist - capsuleRadius;
            const float3 pointOnCapsule = pointOnSegment + capsuleRadius * normal;
            o.point = float4(pointOnCapsule, 0.0f);
            o.normalPen = float4(normal, pen);
            o.barycentric = float4(wBary, u, v, 0.0f);
            count = 1;
        }
    }
    else // clothParticleCollision core
    {
        const float restDistance = w.b.w;
        const float3 p = w.shapeP.xyz;
        const float3 ab = b - a;
        const float3 ac = c - a;

        const float3 closestPt = closestPtPointTriangle2(p, a, b, c, ab, ac);
        const float3 v = p - closestPt;
        const float sqDist = dot(v, v);
        const float sqContactDistance = contactDist * contactDist;

        if (sqDist <= sqContactDistance)
        {
            const float m = sqrt(sqDist);
            const float3 n = v / m; // faithful: no zero guard
            const float pen = m - restDistance;
            const float3 contact = p - n * restDistance;

            float4 tBarycentric;
            computeBarycentric(a, b, c, closestPt, tBarycentric); // bool assert-only upstream

            uint index;
            InterlockedAdd(gFemCount[0], 1, index); // ungated reservation

            const uint2 pairInd0 = uint2(w.ids.y, w.ids.z); // PxEncodeParticleIndex(sys, particle)
            const uint pairInd1 = (w.ids.w << PXG_BITSHIFT_ELEMENT_ID) | w.ids2.x; // PxEncodeClothIndex

            femWriteContact(index, float4(contact, 0.0f), float4(-n, pen), tBarycentric,
                            pairInd0, pairInd1, pairInd0);
            // test observability only — the CUDA core returns void
            o.point = float4(contact, 0.0f);
            o.normalPen = float4(-n, pen);
            o.barycentric = tBarycentric;
            count = 1;
        }
    }

    o.count = uint4(count, 0, 0, 0);
    gOut[i] = o;
}
