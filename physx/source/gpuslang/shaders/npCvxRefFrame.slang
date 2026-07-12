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

// npCvxRefFrame.slang
//
// Slang port of the reference-frame + path-decision core of
// generateConvexContacts() from cudaGJKEPA.cu (L891-937).  It picks the best
// polygon of each hull (getPolygonIndexFromLocalWitness, T138), builds the
// reference (hull B) and incident (hull A) polygon planes in shape/B space,
// computes their projections onto the GJK normal, transforms the incident plane
// into B space, and decides which polygon is the reference face.
//
// New PxPlane ops:
//   normalize(): denom = 1/n.magnitude(); n *= denom; d *= denom;
//   transform(pose): n' = pose.rotate(n); d' = d - pose.p.dot(n');
// PxTransform: rotate = q.rotate, rotateInv = q.rotateInv,
//   transformInv(v) = q.rotateInv(v - p).

static const uint WARP_SIZE = 32u;
static const float PXG_MAX_F32 = 3.4028235e38f;
static const float PCM_LOWER_EPS = 1e-2f;
static const float PCM_UPPER_EPS = 5e-2f;

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    float3 t = quatRotate(rot, v);
    t.x /= scale.x; t.y /= scale.y; t.z /= scale.z;
    return quatRotateInv(rot, t);
}
float3 getNormalized(float3 v) { const float m = dot(v, v); return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0); }

struct RefFrameIn
{
    uint base0, base1;   // plane-buffer bases (float4 units)
    uint nbP0, nbP1;
    float4 scale0Rad;    // scale0.xyz ; .w = inSphereRadius0
    float4 rot0;
    float4 scale1Rad;    // scale1.xyz ; .w = inSphereRadius1
    float4 rot1;
    float4 witnessA;     // closestPointA.xyz ; .w = toleranceLength
    float4 witnessB;     // closestPointB.xyz
    float4 normal;       // gjk direction .xyz
    float4 aToBq;
    float4 aToBp;        // .xyz translation
}

struct RefFrameOut
{
    float4 refN;         // reference plane (n.xyz, d), normalized
    float4 incNt;        // incident plane transformed into B space (n.xyz, d), normalized
    uint bestPlaneAIdx;
    uint bestPlaneBIdx;
    float referenceProject;
    float incidentProject;
    uint orthoGate;      // 1 if both |normal.n| >= 1e-6
    uint refIsReference; // 1 if referenceProject > incidentProject
    uint pad0, pad1;
}

struct RefFrameParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<RefFrameParams> gParams;

RWStructuredBuffer<RefFrameIn> gIn;      // binding 0
RWStructuredBuffer<float4>     gPlanes0; // binding 1 : hull-A planes
RWStructuredBuffer<float4>     gPlanes1; // binding 2 : hull-B planes
RWStructuredBuffer<RefFrameOut> gOut;    // binding 3

// getPolygonIndexFromLocalWitness (T138); planes read from a plane buffer.
uint getPolygonIndex(RWStructuredBuffer<float4> planes, uint base, uint nbPolygons,
                     float3 scale, float4 rot, float3 witnessPt, float3 n, float radius, float toleranceLength)
{
    const float margin = radius * 0.01f;
    const float tolerance = clamp(margin, toleranceLength * PCM_LOWER_EPS, toleranceLength * PCM_UPPER_EPS);
    const float3 localWitness = shape2Vertex(witnessPt, scale, rot);
    const float eps = -tolerance;

    float laneMin[32]; float laneMax[32]; uint laneIdx[32];
    for (uint L = 0u; L < WARP_SIZE; ++L)
    {
        float minDist = PXG_MAX_F32, maxDist = -PXG_MAX_F32; uint index = 0u;
        for (uint i = L; i < nbPolygons; i += WARP_SIZE)
        {
            const float4 pl = planes[base + i];
            const float pd = dot(float3(pl.x, pl.y, pl.z), localWitness) + pl.w;
            const float td = pd >= eps ? pd : PXG_MAX_F32;
            minDist = min(td, minDist);
            if (td > maxDist) { maxDist = td; index = i; }
        }
        laneMin[L] = minDist; laneMax[L] = maxDist; laneIdx[L] = index;
    }
    float minDist = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) minDist = min(minDist, laneMin[L]);
    if (minDist == PXG_MAX_F32)
    {
        float maxDist = -PXG_MAX_F32;
        for (uint L = 0u; L < WARP_SIZE; ++L) maxDist = max(maxDist, laneMax[L]);
        for (uint L = 0u; L < WARP_SIZE; ++L) if (laneMax[L] == maxDist) return laneIdx[L];
        return 0u;
    }
    float laneProj[32]; uint laneProjIdx[32];
    for (uint L = 0u; L < WARP_SIZE; ++L)
    {
        float bestProjection = PXG_MAX_F32; uint index = laneIdx[L];
        for (uint i = L; i < nbPolygons; i += WARP_SIZE)
        {
            const float4 pl = planes[base + i];
            const float3 pn = float3(pl.x, pl.y, pl.z);
            const float pd = dot(pn, localWitness) + pl.w;
            const float td = pd >= eps ? pd : PXG_MAX_F32;
            if (tolerance > (td - minDist))
            {
                const float3 planeN = getNormalized(shape2Vertex(pn, scale, rot));
                const float proj = dot(planeN, n);
                if (bestProjection > proj) { index = i; bestProjection = proj; }
            }
        }
        laneProj[L] = bestProjection; laneProjIdx[L] = index;
    }
    float bestProjection = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) bestProjection = min(bestProjection, laneProj[L]);
    for (uint L = 0u; L < WARP_SIZE; ++L) if (laneProj[L] == bestProjection) return laneProjIdx[L];
    return 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxRefFrame(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const RefFrameIn s = gIn[t];

    const float3 normal = s.normal.xyz;
    const float3 scale0 = s.scale0Rad.xyz, scale1 = s.scale1Rad.xyz;
    const float inSphere0 = s.scale0Rad.w, inSphere1 = s.scale1Rad.w;
    const float3 witnessA = s.witnessA.xyz, witnessB = s.witnessB.xyz;
    const float tol = s.witnessA.w;
    const float3 aToBp = s.aToBp.xyz;
    const float4 aToBq = s.aToBq;

    // normalIn0 = aToB.rotateInv(-normal)
    const float3 normalIn0 = quatRotateInv(aToBq, -normal);

    // hull A: witness = aToB.transformInv(closestPointA), direction = normalIn0
    const float3 witnessA_v = quatRotateInv(aToBq, witnessA - aToBp);
    const uint bestPlaneAIdx = getPolygonIndex(gPlanes0, s.base0, s.nbP0, scale0, s.rot0, witnessA_v, normalIn0, inSphere0, tol);
    // hull B: witness = closestPointB, direction = normal
    const uint bestPlaneBIdx = getPolygonIndex(gPlanes1, s.base1, s.nbP1, scale1, s.rot1, witnessB, normal, inSphere1, tol);

    // reference plane (hull B)
    const float4 refPl = gPlanes1[s.base1 + bestPlaneBIdx];
    float3 refN = shape2Vertex(float3(refPl.x, refPl.y, refPl.z), scale1, s.rot1);
    float refD = refPl.w;
    { const float denom = 1.0f / sqrt(dot(refN, refN)); refN = refN * denom; refD = refD * denom; }

    // incident plane (hull A)
    const float4 incPl = gPlanes0[s.base0 + bestPlaneAIdx];
    float3 incN = shape2Vertex(float3(incPl.x, incPl.y, incPl.z), scale0, s.rot0);
    float incD = incPl.w;
    { const float denom = 1.0f / sqrt(dot(incN, incN)); incN = incN * denom; incD = incD * denom; }

    const float referenceProject = abs(dot(refN, normal));
    const float incidentProject = abs(dot(incN, normalIn0)); // pre-transform incident normal

    // transform incident plane into B space: n' = rotate(n); d' = d - p.n'
    float3 incNt = quatRotate(aToBq, incN);
    float incDt = incD - dot(aToBp, incNt);
    { const float denom = 1.0f / sqrt(dot(incNt, incNt)); incNt = incNt * denom; incDt = incDt * denom; }

    const float orthoEps = 1e-6f;
    const bool orthoGate = (abs(dot(normal, refN)) >= orthoEps) && (abs(dot(normal, incNt)) >= orthoEps);
    const bool refIsReference = referenceProject > incidentProject;

    RefFrameOut r;
    r.refN = float4(refN, refD);
    r.incNt = float4(incNt, incDt);
    r.bestPlaneAIdx = bestPlaneAIdx;
    r.bestPlaneBIdx = bestPlaneBIdx;
    r.referenceProject = referenceProject;
    r.incidentProject = incidentProject;
    r.orthoGate = orthoGate ? 1u : 0u;
    r.refIsReference = refIsReference ? 1u : 0u;
    r.pad0 = 0u; r.pad1 = 0u;
    gOut[t] = r;
}
