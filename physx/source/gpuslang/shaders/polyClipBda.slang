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

// polyClipBda.slang
//
// Test scaffolding for pxgSlangPolyClip — NOT an engine kernel, and deliberately
// absent from the manifest.  One workgroup per pair, 32 threads, which is the
// module's standing requirement (its early-outs are warp-uniform but not
// group-uniform, and a divergent barrier HANGS llvmpipe rather than erroring).
//
// Input record, 1088 bytes per pair:
//   @0    float4 plane0            (n.xyz, d)  — the convex face
//   @16   float4 plane1            (n.xyz, d)  — the triangle
//   @32   float4 axisAndMaxSep     (axis.xyz, maxSep)
//   @48   uint4  counts            (nbVerts0, initContacts, triEdgeMask, _)
//   @64   float4 v0[32]            poly0 vertex per lane (.xyz)
//   @576  float4 v1[32]            poly1 vertex per lane (.xyz; only 0..2 matter)
//
// Output record, 528 bytes per pair:
//   @0    uint   nbContacts (the return value)
//   @16   float4 contacts[32]      (x, y, z, sep) as the module staged them
//
// Lanes at or past nbVerts0 still get a DEFINED v0 from the fixture.  In the
// kernel they hold whatever the vertex load left there and the algorithm masks
// their contribution out (in1/out1 forced to 0, in0/out0 never assigned, the
// second addContacts masked by (1<<nbVerts0)-1) — but "masked out" is not
// "unread", so the fixture keeps them deterministic to stay comparable with the
// CPU oracle.

import pxgSlangPolyClip;

struct PolyClipParams
{
    uint64_t input;
    uint64_t output;
    uint     numPairs;
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<PolyClipParams> gPc;

static const uint PC_IN_STRIDE = 1088u;
static const uint PC_OUT_STRIDE = 528u;

[shader("compute")]
[numthreads(32, 1, 1)]
void polyClipLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    for (uint pair = gid.x; pair < gPc.numPairs; pair += WorkgroupCount().x)
    {
        const uint64_t inBase = gPc.input + uint64_t(pair) * uint64_t(PC_IN_STRIDE);
        const uint64_t outBase = gPc.output + uint64_t(pair) * uint64_t(PC_OUT_STRIDE);

        const float4 plane0 = Ptr<float4>(inBase)[0];
        const float4 plane1 = Ptr<float4>(inBase + 16UL)[0];
        const float4 axisMax = Ptr<float4>(inBase + 32UL)[0];
        const uint4 counts = Ptr<uint4>(inBase + 48UL)[0];
        const float3 v0 = Ptr<float4>(inBase + 64UL)[lane].xyz;
        const float3 v1 = Ptr<float4>(inBase + 576UL)[lane].xyz;

        const uint nb = pcPolyClip(lane, plane0, v0, counts.x, plane1, v1,
                                   axisMax.xyz, axisMax.w, counts.z, counts.y);

        // The module's staging buffer is groupshared; make the writes visible
        // before any lane reads them back out.
        GroupMemoryBarrierWithGroupSync();

        if (lane == 0u)
            Ptr<uint>(outBase)[0] = nb;

        Ptr<float4>(outBase + 16UL)[lane] = pcContact(lane);

        // The next pair re-uses the same groupshared staging.
        GroupMemoryBarrierWithGroupSync();
    }
}
