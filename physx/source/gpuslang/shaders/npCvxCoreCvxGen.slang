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

// npCvxCoreCvxGen.slang
//
// The Phase-1 FIXTURE ENTRY for Gu::generateContacts(ConvexShape, ConvexShape)
// -- the full convex-vs-convex contact driver (GuConvexSupport.h L1162-1207):
//   maxDist = margin0 + margin1 + contactDist;
//   dist = computeGjkDistance(...); if dist < FLT_EPSILON -> computeEpaDepth;
//   dist > maxDist -> 0 contacts; cullDir.dot(axis) < -1e-5 -> 0 contacts;
//   point0 -= axis*margin0; point1 += axis*margin1; dist -= margins;
//   generateContactPatch = FaceClipper(contactFace(-axis, point0),
//   contactFace(axis, point1)).clip().getContact().
//
// P9.69 MOVED THE ARITHMETIC OUT.  What is left here is the descriptor-set
// binding, the CvxIn/CvxOut record layout and the entry point; the shape layer
// now comes from the module pxgSlangCvxCoreShape (P9.67b) and the GJK/EPA/clip
// layer from the INCLUDE pxgSlangCvxCoreCvx (P9.69 -- see that file's header for
// why an include and not a module: a module boundary took this file's compile from
// 1.07 seconds to over forty minutes without finishing).  Nothing about the numbers changed, and that is
// checkable: this fixture's bitwise compare and its mutation sweep are exactly
// the ones that ran before the move, against exactly the same instructions.
//
// The 553 lines of shape layer this file used to carry were a
// character-for-character duplicate of pxgSlangCvxCoreShape.
//
// Compare doctrine: counts/normal/non-trig outputs BITWISE (modulo the
// documented +-0 class); contactFace cap points that passed rotatePoints'
// acos/sin/cos verify to tolerance behind the deterministic trig flags.
// SPIR-V gets the NoContraction post-pass.

import pxgSlangCvxCoreShape;          // ShapeRef, supportLocalQ, contactFaceQ, ...
#include "pxgSlangCvxCoreCvx.slang"   // gjkDistanceQ / epaDepthQ / faceClipQ

struct CvxIn
{
    uint4 metaA;
    uint4 coreA[3];
    float4 poseAq, poseAp;  // poseAp.w = margin0
    uint4 metaB;
    uint4 coreB[3];
    float4 poseBq, poseBp;  // poseBp.w = margin1
    float4 cfg;             // x = contactDist, yzw = cullDir
    // P9.69b: the ePOINTS hulls reach the shader as DEVICE ADDRESSES now, because
    // the shared shape layer reads them through Ptr<float4> instead of taking a
    // float4[12] by value -- a cooked hull carries up to 255 points and the pool
    // capped it at 12.  The inline arrays stay in the record because the host
    // mirror memcpy's this struct's leading bytes into gjkref::GjkIn and
    // eparef::EpaTIn, whose own shaders still take a pool; the addresses point at
    // a separate device buffer holding the same points.
    float4 ptsA[12];
    float4 ptsB[12];
    uint4 ptsAddrA;    // lo/hi in (x,y) -- see npCvxCorePlaneGen on why not uint64_t
    uint4 ptsAddrB;
}

struct CvxOut
{
    uint4 counts;   // x = numPoints, y = trig flags, z = path tag (0 none, 1 contacts)
    float4 normal;
    float4 points[4]; // w = dists[i]
}

struct CvxParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CvxParams> gParams;

RWStructuredBuffer<CvxIn> gIn;    // binding 0
RWStructuredBuffer<CvxOut> gOut;  // binding 1

[shader("compute")]
[numthreads(32, 1, 1)]
void cvgContacts(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CvxIn s = gIn[t];

    // lo/hi -> the 64-bit device addresses of this test's two hulls
    const uint64_t ptsAddrA = (uint64_t(s.ptsAddrA.y) << 32) | uint64_t(s.ptsAddrA.x);
    const uint64_t ptsAddrB = (uint64_t(s.ptsAddrB.y) << 32) | uint64_t(s.ptsAddrB.x);

    ShapeRef shA, shB;
    shA.type = s.metaA.x; shA.core = s.coreA; shA.poseQ = s.poseAq; shA.poseP = s.poseAp.xyz;
    shB.type = s.metaB.x; shB.core = s.coreB; shB.poseQ = s.poseBq; shB.poseP = s.poseBp.xyz;
    const float margin0 = s.poseAp.w, margin1 = s.poseBp.w;
    const float contactDist = s.cfg.x;
    const float3 cullDir = float3(s.cfg.y, s.cfg.z, s.cfg.w);

    // generateContacts (GuConvexSupport.h L1169-1201)
    const float maxDist = margin0 + margin1 + contactDist;

    float dist; uint tagG;
    float3 point0, point1, axis;
    gjkDistanceQ(shA, shB, ptsAddrA, ptsAddrB, maxDist, dist, tagG, point0, point1, axis);

    if (dist < FLT_EPS)
    {
        uint tagE;
        epaDepthQ(shA, shB, ptsAddrA, ptsAddrB, dist, tagE, point0, point1, axis);
    }

    uint numPoints = 0u;
    uint trig = 0u;
    float3 normal = float3(0, 0, 0);
    float4 patch[4] = { float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0) };
    bool emit = true;

    if (dist > maxDist)
        emit = false;

    const float testEps = 1e-5f;
    if (emit && dot3Q(cullDir, axis) < -testEps)
        emit = false;

    if (emit)
    {
        point0 -= axis * margin0;
        point1 += axis * margin1;
        dist -= margin0 + margin1;

        // generateContactPatch (L872-882): FaceClipper ctor's two
        // ConvexShape::contactFace wrapper calls, then clip + getContact
        uint trig0, trig1;
        float3 fn0, fn1;
        float3 fpl[4];
        float3 wfp0[4], wfp1[4];

        const uint np0 = contactFaceQ(shA, ptsAddrA, quatRotInvQ(shA.poseQ, -axis),
                                      quatRotInvQ(shA.poseQ, point0 - shA.poseP), fn0, fpl, trig0);
        const float3 wn0 = quatRotQ(shA.poseQ, fn0);
        for (uint i = 0u; i < 4u; ++i)
            wfp0[i] = i < np0 ? (quatRotQ(shA.poseQ, fpl[i]) + shA.poseP) + wn0 * margin0 : float3(0, 0, 0);

        const uint np1 = contactFaceQ(shB, ptsAddrB, quatRotInvQ(shB.poseQ, axis),
                                      quatRotInvQ(shB.poseQ, point1 - shB.poseP), fn1, fpl, trig1);
        const float3 wn1 = quatRotQ(shB.poseQ, fn1);
        for (uint i = 0u; i < 4u; ++i)
            wfp1[i] = i < np1 ? (quatRotQ(shB.poseQ, fpl[i]) + shB.poseP) + wn1 * margin1 : float3(0, 0, 0);

        numPoints = faceClipQ(wfp0, np0, wn0, wfp1, np1, wn1, point0, point1, axis, normal, patch);
        trig = trig0 | (trig1 << 1u);
    }

    CvxOut r;
    r.counts = uint4(numPoints, trig, emit ? 1u : 0u, 0u);
    r.normal = float4(normal, 0);
    for (uint i = 0u; i < 4u; ++i)
        r.points[i] = i < numPoints ? patch[i] : float4(0, 0, 0, 0);
    gOut[t] = r;
}
