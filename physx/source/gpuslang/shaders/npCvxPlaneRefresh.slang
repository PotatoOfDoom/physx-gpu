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

// npCvxPlaneRefresh.slang
//
// Slang port of the convexPlaneNphase_kernel_core manifold refresh + rerun
// decision + no-rerun contact reload (cudaGJKEPA.cu L2856-2950 and the
// !bRunNarrowphase branch L3061-3070).  This path differs from the
// convex-convex stage1 refresh in several load-bearing ways, all ported
// verbatim:
//   - bToA = planeTransform.transformInv(convexTransform); the working normal
//     n = bToA.rotateInv((1,0,0)).getNormalized() (the plane's +x axis brought
//     to convex space);
//   - the refresh transforms the stored PLANE-space pointA back with
//     bToA.transformInv (not transform), and v = localBInB - localAInB
//     (REVERSED relative to stage1); dist = v.dot(n); keep iff
//     |v - n*dist|^2 <= (0.8*minMargin)^2 with minMargin =
//     calculatePCMConvexMargin(extents*scale, toleranceLength);
//   - NO compaction: only the ballot count matters here (nbContacts = popc);
//   - bRunNarrowphase = (nbContacts != initialNbContacts), else
//     invalidate_BoxConvex(-bToA.p, planeQ, convexQ, prevRelPos, prevQuatA,
//     prevQuatB, minMargin, radiusB, radiusB, nbContacts,
//     local_invalidateThresholdsConvex, local_invalidateQuatThresholdsConvexPlane)
//     -- BOTH radius arguments are |RAW blob extents| (radiusA = radiusB; the
//     CUDA never scales extents1 here, unlike the stage1 path which
//     pre-scales -- only calculatePCMConvexMargin scales internally) and the
//     quat table is the ConvexPlane one (five 0.99996 entries);
//   - when not rerunning, each lane < nbContacts reloads p =
//     bToA.transform(pointB) and sep = dist (its refreshed separation), and
//     mask = (1 << nbContacts) - 1 -- these feed the output stage.
//
// One pair per test (32 lanes serialized).  extents arrive as direct inputs
// (blob reads are tranche 142).

static const float thresholdsConvex[5] = { 0.5f, 0.125f, 0.25f, 0.375f, 0.375f };
static const float quatThresholdsConvexPlane[5] = { 0.99996f, 0.99996f, 0.99996f, 0.99996f, 0.99996f };

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
float4 quatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
                  a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
                  a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}
float quatDot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

struct PlaneRefreshIn
{
    float4 planeTq, planeTp;   // plane transform; planeTp.w = contactDistance sum (unused here)
    float4 convexTq, convexTp; // convex transform
    float4 extents;            // blob extents (.xyz raw); .w unused
    float4 scale;              // .xyz mesh scale; .w = toleranceLength
    float4 mA[4];              // manifold A points (plane space)
    float4 mB[4];              // manifold B points (convex space)
    uint4 moreData0;           // mRelativePos bits
    uint4 moreData1;           // mQuatA bits (previous plane quat)
    uint4 moreData2;           // mQuatB bits (previous convex quat)
    uint4 misc;                // x = initialNbContacts, y/z/w pad
}

struct PlaneRefreshOut
{
    float4 n;          // working normal in convex space (.xyz); .w = minMargin
    float4 bToAq;
    float4 bToAp;
    float4 p[32];      // no-rerun reload: p (.xyz) and sep (.w) per lane (< nbContacts)
    uint4 result;      // x = bRunNarrowphase, y = nbContacts (refresh count), z = mask, w = keep ballot
}

struct PlaneRefreshParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PlaneRefreshParams> gParams;

RWStructuredBuffer<PlaneRefreshIn> gIn;    // binding 0
RWStructuredBuffer<PlaneRefreshOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxPlaneRefresh(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PlaneRefreshIn s = gIn[t];

    const float4 planeTq = s.planeTq;
    const float3 planeTp = s.planeTp.xyz;
    const float4 convexTq = s.convexTq;
    const float3 convexTp = s.convexTp.xyz;
    const float toleranceLength = s.scale.w;

    // bToA = planeTransform.transformInv(convexTransform)
    const float4 bToAq = quatMul(float4(-planeTq.x, -planeTq.y, -planeTq.z, planeTq.w), convexTq);
    const float3 bToAp = quatRotateInv(planeTq, convexTp - planeTp);

    // n = bToA.rotateInv(localNormal=(1,0,0)).getNormalized()
    const float3 n = getNormalized(quatRotateInv(bToAq, float3(1, 0, 0)));

    // minMargin = calculatePCMConvexMargin(extents, scale, toleranceLength)
    const float3 ext = float3(s.extents.x * s.scale.x, s.extents.y * s.scale.y, s.extents.z * s.scale.z);
    const float minMargin = min(min(min(ext.x, ext.y), ext.z) * 0.25f, toleranceLength * 0.05f);

    const uint initialNb = s.misc.x;

    // refresh (no compaction): count survivors, remember per-lane dist for the reload
    uint keepW = 0u;
    float distL[32];
    for (uint L = 0u; L < 32u; ++L) distL[L] = 0.0f;
    for (uint L = 0u; L < 4u; ++L)
    {
        if (L >= initialNb) break;
        const float3 pointA = s.mA[L].xyz;
        const float3 pointB = s.mB[L].xyz;
        // localAInB = bToA.transformInv(pointA)
        const float3 localAInB = quatRotateInv(bToAq, pointA - bToAp);
        const float3 v = pointB - localAInB; // REVERSED relative to the convex-convex refresh
        const float dist = dot(v, n);
        distL[L] = dist;
        const float3 projectedDifference = v - n * dist;
        const float distance2d = dot(projectedDifference, projectedDifference);
        const float th = minMargin * 0.8f;
        if (distance2d <= th * th) keepW |= (1u << L);
    }
    const uint nbContacts = countbits(keepW);

    bool bRun = nbContacts != initialNb;

    if (!bRun)
    {
        // invalidate_BoxConvex(-bToA.p, planeQ, convexQ, preRelPos, preQuatA, preQuatB,
        //                      minMargin, radiusB, radiusB, nbContacts, thresholdsConvex,
        //                      quatThresholdsConvexPlane)
        const float3 preRelPos = float3(asfloat(s.moreData0.x), asfloat(s.moreData0.y), asfloat(s.moreData0.z));
        const float4 preQuatA = float4(asfloat(s.moreData1.x), asfloat(s.moreData1.y), asfloat(s.moreData1.z), asfloat(s.moreData1.w));
        const float4 preQuatB = float4(asfloat(s.moreData2.x), asfloat(s.moreData2.y), asfloat(s.moreData2.z), asfloat(s.moreData2.w));

        // RAW blob extents (cu L2946) -- NOT the scaled ones used for the margin
        const float3 re = s.extents.xyz;
        const float radiusB = sqrt(re.x * re.x + re.y * re.y + re.z * re.z);

        const float ratio = thresholdsConvex[nbContacts];
        const float thresholdQ = quatThresholdsConvexPlane[nbContacts];
        const float thresholdP = minMargin * ratio;

        const float3 curRelPos = -bToAp;
        const float3 dpv = curRelPos - preRelPos;
        const float deltaP = max(abs(dpv.x), max(abs(dpv.y), abs(dpv.z)));
        const float deltaQA = quatDot(planeTq, preQuatA);
        const float deltaQB = quatDot(convexTq, preQuatB);

        bool generateContacts = (deltaP > thresholdP) || (thresholdQ > deltaQA) || (thresholdQ > deltaQB);
        if (!generateContacts)
        {
            const float aRadian = deltaQA < 1.0f ? acos(deltaQA) : 0.0f;
            const float bRadian = deltaQB < 1.0f ? acos(deltaQB) : 0.0f;
            const float travelDistA = aRadian * radiusB; // radiusA = radiusB here
            const float travelDistB = bRadian * radiusB;
            generateContacts = (travelDistA > thresholdP) || (travelDistB > thresholdP);
        }
        bRun = generateContacts;
    }

    PlaneRefreshOut r;
    for (uint L = 0u; L < 32u; ++L) r.p[L] = float4(0, 0, 0, 0);
    uint mask = 0u;
    if (!bRun)
    {
        mask = (1u << nbContacts) - 1u;
        for (uint L = 0u; L < 4u; ++L)
        {
            if (L >= nbContacts) break;
            // p = bToA.transform(pointB); sep = dist (this lane's refreshed separation)
            const float3 p = quatRotate(bToAq, s.mB[L].xyz) + bToAp;
            r.p[L] = float4(p, distL[L]);
        }
    }

    r.n = float4(n, minMargin);
    r.bToAq = bToAq;
    r.bToAp = float4(bToAp, 0.0f);
    r.result = uint4(bRun ? 1u : 0u, nbContacts, mask, keepW);
    gOut[t] = r;
}
