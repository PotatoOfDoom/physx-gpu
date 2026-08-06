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

// contactReduceWarpBda.slang
//
// Test scaffolding for pxgSlangContactReduce — NOT an engine kernel, absent from
// the manifest.  One workgroup of 32 threads per case, which is the module's
// standing requirement; lane i carries contact i.
//
// Input record, 544 bytes:
//   @0    float4 normal.xyz, _pad
//   @16   uint4  allMask, _pad, _pad, _pad
//   @32   float4 [32] pointRaw.xyz, separation
//
// Output record, 4 bytes: the surviving lane mask.
//
// One group per case with an `if (c < numCases)` guard rather than a grid-stride
// loop.  Two reasons, and only the second is about this shader: the P8.40 platform
// rule forbids a chained buffer-device-address load inside such a loop (there is
// none here — the record address comes straight from the push constant), and the
// module's barriers make a looping group unsafe the moment the trip count stops
// being uniform across groups.

import pxgSlangContactReduce;

struct ContactReduceWarpParams
{
    uint64_t input;
    uint64_t output;
    uint     numCases;
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<ContactReduceWarpParams> gCr;

static const uint CRB_IN_STRIDE = 544u;

[shader("compute")]
[numthreads(32, 1, 1)]
void contactReduceWarpLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint c = gid.x;

    // Group-uniform, so every lane leaves together and no barrier is skipped by
    // some lanes only.
    if (c >= gCr.numCases)
        return;

    const uint64_t inBase = gCr.input + uint64_t(c) * uint64_t(CRB_IN_STRIDE);

    const float3 normal = Ptr<float4>(inBase)[0].xyz;
    const uint allMask = Ptr<uint>(inBase + 16UL)[0];
    const float4 pt = Ptr<float4>(inBase + 32UL)[lane];

    const uint mask = crContactReduce5(lane, pt.xyz, pt.w, normal, allMask);

    if (lane == 0u)
        Ptr<uint>(gCr.output)[c] = mask;
}

// The SECOND instantiation, `contactReduce<true, false, 6, false>`, which
// convexTrimeshCorrelate reaches.  Same record layout with one addition: the
// cluster bias, which is a live value here rather than the zero the emit tail
// passes, rides in the normal's unused .w slot.
[shader("compute")]
[numthreads(32, 1, 1)]
void contactReduceWarpCluster6Launch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint c = gid.x;

    if (c >= gCr.numCases)
        return;

    const uint64_t inBase = gCr.input + uint64_t(c) * uint64_t(CRB_IN_STRIDE);

    const float4 nrm = Ptr<float4>(inBase)[0];
    const uint allMask = Ptr<uint>(inBase + 16UL)[0];
    const float4 pt = Ptr<float4>(inBase + 32UL)[lane];

    const uint mask = crContactReduce6Cluster(lane, pt.xyz, pt.w, nrm.xyz, allMask, nrm.w);

    if (lane == 0u)
        Ptr<uint>(gCr.output)[c] = mask;
}
