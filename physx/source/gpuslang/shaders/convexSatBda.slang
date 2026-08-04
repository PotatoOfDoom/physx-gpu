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

static const uint CSB_IN_STRIDE = 112u;
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

        if (lane == 0u)
        {
            Ptr<uint>(outBase)[0] = p1 ? 1u : 0u;
            Ptr<uint>(outBase)[1] = p2 ? 1u : 0u;
            Ptr<uint>(outBase)[2] = feature;
            Ptr<uint>(outBase)[3] = 0u;
            Ptr<float4>(outBase + 16UL)[0] = float4(faceNormal, separation);
            Ptr<float4>(outBase + 32UL)[0] = float4(csPlaneN(probeIndex), csPlaneD(probeIndex));
            Ptr<float4>(outBase + 48UL)[0] = float4(csScaledVert(probeIndex), 0.0f);
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
