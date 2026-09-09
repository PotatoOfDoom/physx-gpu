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

// npCvxCorePlaneGen.slang
//
// Slang port of Gu::generateContacts(PxPlane, ConvexShape) -- the
// plane-vs-convex contact driver (GuConvexSupport.cpp L41-75):
//   normal = -plane.n; point1 = convex.support(normal) (WITH margin);
//   dist = plane.distance(point1) = point1.dot(n) + d;
//   if dist < contactDist: contactFace(normal, point1) via the ConvexShape
//   wrapper (local dir/ref prologue, world rotate + margin push epilogue);
//   numPoints == 0 -> the single midpoint contact point1 + normal*dist*0.5;
//   else each facePoint p1 contributes p1 + normal*d*0.5 with d =
//   plane.distance(p1).
// The contactFace family is the tranche-165/166 transcription verbatim
// (all 7 core types incl. the ePOINTS hull-face reduce).  The cylinder/cone
// cap branches pass through rotatePoints' acos/sin/cos, which are NOT
// bit-exact-portable: those outputs verify to tolerance behind the
// deterministic trig flag; every branch decision and all other outputs
// (counts, normal, non-trig points, dists) verify BITWISE (modulo the
// documented +-0 class).

import pxgSlangCvxCoreShape;   // P9.67b: the shared shape layer, moved out verbatim


struct PgIn
{
    float4 plane;     // PxPlane (n, d)
    uint4 meta;       // x = coreType
    uint4 core[3];
    float4 poseQ, poseP;  // poseP.w = margin
    float4 cfg;       // x = contactDist
    float4 pts[12];   // ePOINTS pool
}

struct PgOut
{
    uint4 counts;     // x = numContacts, y = trig flag
    float4 normal;
    float4 points[4]; // w = dists[i]
}

struct PgParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PgParams> gParams;

RWStructuredBuffer<PgIn> gIn;    // binding 0
RWStructuredBuffer<PgOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cpgContacts(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PgIn s = gIn[t];

    ShapeRef sh;
    sh.type = s.meta.x; sh.core = s.core; sh.poseQ = s.poseQ; sh.poseP = s.poseP.xyz;
    const float margin = s.poseP.w;
    const float contactDist = s.cfg.x;
    const float3 pn = s.plane.xyz;
    const float pd = s.plane.w;

    // generateContacts (GuConvexSupport.cpp L41-75)
    const float3 normal = -pn;

    // ConvexShape::support = pose.transform(supportLocal(pose.rotateInv(dir))) + dir * margin
    const float3 point1 = poseTransform(sh, supportLocalQ(sh, s.pts, quatRotInvQ(sh.poseQ, normal))) + normal * margin;
    const float dist = dot3Q(point1, pn) + pd; // PxPlane::distance

    uint numContacts = 0u;
    uint trig = 0u;
    float3 pointsOut[4] = { float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0) };
    float distsOut[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    if (dist < contactDist)
    {
        // wrapper prologue (GuConvexSupport.h L545)
        const float3 dirL = quatRotInvQ(sh.poseQ, normal);
        const float3 refL = quatRotInvQ(sh.poseQ, point1 - sh.poseP); // pose.transformInv

        float3 faceNormal;
        float3 fp[4];
        const uint numPoints = contactFaceQ(sh, s.pts, dirL, refL, faceNormal, fp, trig);

        // wrapper epilogue: world rotate + margin push
        const float3 worldNormal = quatRotQ(sh.poseQ, faceNormal);

        if (numPoints == 0u)
        {
            const float3 point = point1 + normal * dist * 0.5f;
            pointsOut[numContacts] = point;
            distsOut[numContacts] = dist;
            ++numContacts;
        }

        for (uint i = 0u; i < numPoints; ++i)
        {
            const float3 p1 = (quatRotQ(sh.poseQ, fp[i]) + sh.poseP) + worldNormal * margin;
            const float d = dot3Q(p1, pn) + pd;
            pointsOut[numContacts] = p1 + normal * d * 0.5f;
            distsOut[numContacts] = d;
            ++numContacts;
        }
    }

    PgOut r;
    r.counts = uint4(numContacts, trig, 0u, 0u);
    r.normal = float4(normal, 0);
    for (uint i = 0u; i < 4u; ++i)
        r.points[i] = i < numContacts ? float4(pointsOut[i], distsOut[i]) : float4(0, 0, 0, 0);
    gOut[t] = r;
}
