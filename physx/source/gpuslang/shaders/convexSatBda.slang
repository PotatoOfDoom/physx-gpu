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

// convexSatBda.slang
//
// Test scaffolding for pxgSlangConvexSat — NOT an engine kernel, absent from the
// manifest.  One workgroup of 32 threads per case, which is the module's standing
// requirement.  It runs the vertex-scaling prologue, then stage 1, then stage 2 in
// the order convexMeshSAT does, and reports every output the later stages consume.
//
// Input record, 112 bytes:
//   @0   uint64 hullAddr | uint probeIndex | uint _pad
//   @16  float4 scale.xyz, contactDist
//   @32  float4 scaleRot (quat)
//   @48  float4 triLocVerts[0]
//   @64  float4 triLocVerts[1]
//   @80  float4 triLocVerts[2]
//   @96  float4 triangleLocNormal
//
// Output record, 64 bytes:
//   @0   uint stage1Passed | uint stage2Passed | uint feature | uint _pad
//   @16  float4 faceNormal.xyz, separation
//   @32  float4 convexPlaneN[probeIndex].xyz, convexPlaneD[probeIndex]
//   @48  float4 convexScaledVerts[probeIndex].xyz, _pad
//
// The stage-1 plane arrays are reported because satEdgeCrossDirection2 and
// selectPolygonFeatureIndex read them: they are an OUTPUT of this tranche, not an
// internal.

import pxgSlangConvexSat;

struct ConvexSatParams
{
    uint64_t input;
    uint64_t output;
    uint     numCases;
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<ConvexSatParams> gCs;

static const uint CSB_IN_STRIDE = 128u;   // @112 uint3 triAdjTrisIdx
static const uint CSB_OUT_STRIDE = 64u;

[shader("compute")]
[numthreads(32, 1, 1)]
void convexSatLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    // ONE WORKGROUP PER CASE, no grid-stride loop.  A grid-stride loop around a
    // CHAINED buffer-device-address load -- an address read out of memory and then
    // dereferenced, which is what `hull` is here -- segfaults llvmpipe inside the
    // JIT-compiled shader.  The identical body without the loop runs.  P8.40 in
    // RUNTIME_STATUS has the standalone reproducer; the caller simply dispatches
    // one group per case, which it already did.
    const uint c = gid.x;
    if (c < gCs.numCases)
    {
        const uint64_t inBase = gCs.input + uint64_t(c) * uint64_t(CSB_IN_STRIDE);
        const uint64_t outBase = gCs.output + uint64_t(c) * uint64_t(CSB_OUT_STRIDE);

        const uint64_t hull = Ptr<uint64_t>(inBase)[0];
        const uint probeIndex = Ptr<uint>(inBase + 8UL)[0];
        const float4 scaleDist = Ptr<float4>(inBase + 16UL)[0];
        const float4 scaleRot = Ptr<float4>(inBase + 32UL)[0];
        const float3 t0 = Ptr<float4>(inBase + 48UL)[0].xyz;
        const float3 t1 = Ptr<float4>(inBase + 64UL)[0].xyz;
        const float3 t2 = Ptr<float4>(inBase + 80UL)[0].xyz;
        const float3 triN = Ptr<float4>(inBase + 96UL)[0].xyz;
        const uint4 adj = Ptr<uint4>(inBase + 112UL)[0];

        const uint desc = chHullDesc(hull);

        csScaleVerts(lane, hull, desc, scaleDist.xyz, scaleRot);

        uint feature = 0xffffffffu;
        float3 faceNormal = float3(0.0f, 0.0f, 0.0f);
        float separation = -3.4028234663852886e+38f;

        const bool p1 = csSatConvexFaceNormals(lane, hull, desc, scaleDist.xyz, scaleRot,
                                               scaleDist.w, t0, t1, t2,
                                               feature, faceNormal, separation);

        // convexMeshSAT's single barrier sits here, BEFORE the first early return,
        // so every lane reaches it whatever stage 1 decided.
        GroupMemoryBarrierWithGroupSync();

        bool p2 = false;
        if (p1)
            p2 = csSatTriangleNormal(lane, desc, scaleDist.w, triN, t0,
                                     feature, faceNormal, separation);

        // Stage 3 reads convexPlaneN, which stage 1 wrote, so it only runs when
        // stage 1 accepted.  Section bases resolved by the caller, per P8.40.
        bool p3 = false;
        if (p1 && p2)
            p3 = csSatEdgeCrossDirection2(lane, chVerticesByEdges16(hull, desc),
                                          chFacesByEdges8(hull, desc),
                                          chNbEdges(desc), chNbVerts(desc), scaleDist.w,
                                          adj.x, adj.y, adj.z, t0, t1, t2, triN,
                                          feature, faceNormal, separation);

        if (lane == 0u)
        {
            Ptr<uint>(outBase)[0] = p1 ? 1u : 0u;
            Ptr<uint>(outBase)[1] = p2 ? 1u : 0u;
            Ptr<uint>(outBase)[2] = feature;
            Ptr<uint>(outBase)[3] = p3 ? 1u : 0u;
            Ptr<float4>(outBase + 16UL)[0] = float4(faceNormal, separation);
            Ptr<float4>(outBase + 32UL)[0] = float4(csPlaneN(probeIndex), csPlaneD(probeIndex));
            Ptr<float4>(outBase + 48UL)[0] = float4(csScaledVert(probeIndex), 0.0f);
        }

        GroupMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// convexSelectLaunch — the same input record, driving the OTHER half of the
// tranche: convexMeshSAT as a whole, then delayGenerateContact, then
// selectPolygonFeatureIndex.  Separate entry point rather than more output
// fields on the first, because the first one's 64-byte record is full and
// because the two are then independently dispatchable from the fixture.
//
// Output record, 32 bytes:
//   @0   uint satPassed | uint featureIndex | uint delayContacts | uint faceIndex1
//   @16  uint faceIndex2 | float separation | float2 minNormal.xy
//
// minNormal.z is dropped rather than padded to 48: the fixture compares x and y
// bitwise, and a third component that agrees adds no discrimination the first
// two do not already give.
// ---------------------------------------------------------------------------
static const uint CSB_SEL_OUT_STRIDE = 32u;

[shader("compute")]
[numthreads(32, 1, 1)]
void convexSelectLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint c = gid.x;
    if (c < gCs.numCases)
    {
        const uint64_t inBase = gCs.input + uint64_t(c) * uint64_t(CSB_IN_STRIDE);
        const uint64_t outBase = gCs.output + uint64_t(c) * uint64_t(CSB_SEL_OUT_STRIDE);

        const uint64_t hull = Ptr<uint64_t>(inBase)[0];
        const float4 scaleDist = Ptr<float4>(inBase + 16UL)[0];
        const float4 scaleRot = Ptr<float4>(inBase + 32UL)[0];
        const float3 t0 = Ptr<float4>(inBase + 48UL)[0].xyz;
        const float3 t1 = Ptr<float4>(inBase + 64UL)[0].xyz;
        const float3 t2 = Ptr<float4>(inBase + 80UL)[0].xyz;
        const float3 triN = Ptr<float4>(inBase + 96UL)[0].xyz;
        const uint4 adj = Ptr<uint4>(inBase + 112UL)[0];

        const uint desc = chHullDesc(hull);

        csScaleVerts(lane, hull, desc, scaleDist.xyz, scaleRot);

        uint featureIndex;
        float separation;
        float3 minNormal;
        const bool passed = csConvexMeshSAT(lane, hull, desc, scaleDist.xyz, scaleRot, scaleDist.w,
                                            t0, t1, t2, triN, adj.x, adj.y, adj.z,
                                            featureIndex, separation, minNormal);

        // convexTriangleContactGen's __syncwarp() at :1683, between convexMeshSAT
        // and the reads of convexPlaneN/D below it.
        GroupMemoryBarrierWithGroupSync();

        bool delayContacts = false;
        uint faceIndex1 = 0xFFFFFFFFu;
        uint faceIndex2 = 0xFFFFFFFFu;

        if (passed)
        {
            delayContacts = csDelayGenerateContact(minNormal, triN, featureIndex);
            faceIndex1 = csSelectPolygonFeatureIndex(lane, hull, desc, minNormal,
                                                     featureIndex, faceIndex2);
        }

        if (lane == 0u)
        {
            Ptr<uint>(outBase)[0] = passed ? 1u : 0u;
            Ptr<uint>(outBase)[1] = featureIndex;
            Ptr<uint>(outBase)[2] = delayContacts ? 1u : 0u;
            Ptr<uint>(outBase)[3] = faceIndex1;
            Ptr<uint>(outBase)[4] = faceIndex2;
            Ptr<float>(outBase + 20UL)[0] = separation;
            Ptr<float>(outBase + 24UL)[0] = minNormal.x;
            Ptr<float>(outBase + 28UL)[0] = minNormal.y;
        }
    }
}

// ---------------------------------------------------------------------------
// convexSupportLaunch -- P9.89: the two stage-3 support folds, csSupportTriangle
// (supportTriangle, convexTriangle.cuh:507-518) and csSupportConvex
// (supportConvex, :494-505), driven directly, one (triangle, direction) pair per
// lane against the group's hull.
//
// Why a probe of its own: in satEdgeCrossDirection2 the four results only reach
// an output through `minConv > maxTri + contactDist` style comparisons, which
// cannot see the sign of a zero, and through tempOverlap0/1 -> `separation`,
// which only shows the sign when BOTH operands are zeros.  Two of the four lines
// (the two MIN folds) can be pinned that way and are, in convexSatStages; the
// two MAX folds, the fold order and the +-PX_MAX_F32 seeds against a +inf
// projection cannot, so they are read out here, per lane and bit for bit.
//
// Input record, CSB_SUP_IN_STRIDE bytes per group:
//   @0   uint64 hullAddr | uint _pad[2]
//   @16  float4 scale.xyz, _pad
//   @32  float4 scaleRot (quat)
//   @48  _pad[16]
//   @64  32 x { float4 t0, float4 t1, float4 t2, float4 dir }
// Output: 32 x float4 (minTri, maxTri, minConv, maxConv) per group.
// ---------------------------------------------------------------------------
static const uint CSB_SUP_IN_STRIDE = 64u + 32u * 64u;
static const uint CSB_SUP_OUT_STRIDE = 32u * 16u;

[shader("compute")]
[numthreads(32, 1, 1)]
void convexSupportLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    // One group per hull and no grid-stride loop, for the reason convexSatLaunch
    // gives: `hull` is a chained BDA load.
    const uint g = gid.x;
    if (g < gCs.numCases)
    {
        const uint64_t inBase = gCs.input + uint64_t(g) * uint64_t(CSB_SUP_IN_STRIDE);
        const uint64_t hull = Ptr<uint64_t>(inBase)[0];
        const float4 scale = Ptr<float4>(inBase + 16UL)[0];
        const float4 scaleRot = Ptr<float4>(inBase + 32UL)[0];
        const uint desc = chHullDesc(hull);

        // fills convexScaledVerts and ends with the group barrier
        csScaleVerts(lane, hull, desc, scale.xyz, scaleRot);

        const uint64_t laneIn = inBase + 64UL + uint64_t(lane) * 64UL;
        const float3 t0 = Ptr<float4>(laneIn)[0].xyz;
        const float3 t1 = Ptr<float4>(laneIn + 16UL)[0].xyz;
        const float3 t2 = Ptr<float4>(laneIn + 32UL)[0].xyz;
        const float3 dir = Ptr<float4>(laneIn + 48UL)[0].xyz;

        float minTri, maxTri, minConv, maxConv;
        csSupportTriangle(t0, t1, t2, dir, minTri, maxTri);
        csSupportConvex(chNbVerts(desc), dir, minConv, maxConv);

        const uint64_t outBase = gCs.output + uint64_t(g) * uint64_t(CSB_SUP_OUT_STRIDE);
        Ptr<float4>(outBase + uint64_t(lane) * 16UL)[0] = float4(minTri, maxTri, minConv, maxConv);
    }
}
