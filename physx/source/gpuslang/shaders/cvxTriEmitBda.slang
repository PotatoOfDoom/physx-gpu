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

// cvxTriEmitBda.slang
//
// Test scaffolding for pxgSlangCvxTriEmit — NOT an engine kernel, absent from the
// manifest.  One workgroup of 32 threads per pair, which is the module's standing
// requirement; lane i carries the clipper's staging slot i.
//
// ONE GROUP PER PAIR, deliberately NOT a group-stride loop.  The pairs share the
// two atomic cursors and the temp contact stream, so a looping group would make
// the reservation order depend on the dispatch shape; and the module's barriers
// are only safe while every group runs the same number of iterations.
//
// Input record, 640 bytes:
//   @0    uint4  isSATPassed | delayContactsIn | numContactsGenerated | tempContactSize
//   @16   uint4  convexTriPairOffset | convexTriPairOffsetPadded | remapCpuTriangleIdx | triangleIdx
//   @32   uint4  globalWarpIndex | _pad x3
//   @48   float4 minNormal.xyz, _pad
//   @64   float4 triVert0.xyz, _pad
//   @80   float4 triVert1.xyz, _pad
//   @96   float4 triVert2.xyz, _pad
//   @112  float4 _pad (keeps the staging run 128-aligned)
//   @128  float4 [32] staged contacts: pa.xyz, sep
//
// The six output buffers and the two atomic cursors are separate allocations, so
// the fixture can preset each cursor and read the slices back independently.

import pxgSlangCvxTriEmit;

struct CvxTriEmitParams
{
    uint64_t input;                  // @0
    uint64_t cvxTriNI;               // @8
    uint64_t cvxTriContacts;         // @16
    uint64_t cvxTriMaxDepth;         // @24
    uint64_t cvxTriInterm;           // @32
    uint64_t orderedCvxTriInterm;    // @40
    uint64_t convTriSecondPassPairs; // @48
    uint64_t nbSecondPassPairs;      // @56
    uint64_t tempConvexTriContacts;  // @64
    uint64_t pTempContactIndex;      // @72
    uint     numPairs;               // @80
    uint     _pad84;
}

[[vk::push_constant]]
ConstantBuffer<CvxTriEmitParams> gCte;

static const uint CTEB_IN_STRIDE = 640u;

[shader("compute")]
[numthreads(32, 1, 1)]
void cvxTriEmitLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint p = gid.x;

    // Group-uniform, so the whole group leaves together and no barrier is skipped
    // by some lanes only.
    if (p >= gCte.numPairs)
        return;

    const uint64_t inBase = gCte.input + uint64_t(p) * uint64_t(CTEB_IN_STRIDE);

    const uint4 w0 = Ptr<uint4>(inBase)[0];
    const uint4 w1 = Ptr<uint4>(inBase + 16UL)[0];
    const uint4 w2 = Ptr<uint4>(inBase + 32UL)[0];
    const float3 minNormal = Ptr<float4>(inBase + 48UL)[0].xyz;
    const float3 triVert0 = Ptr<float4>(inBase + 64UL)[0].xyz;
    const float3 triVert1 = Ptr<float4>(inBase + 80UL)[0].xyz;
    const float3 triVert2 = Ptr<float4>(inBase + 96UL)[0].xyz;

    // Every lane index is a valid staging slot, so this read is in bounds even
    // when the lane holds no contact — the module's `lane < count` guard is what
    // makes the value irrelevant.  Slot 0 is read separately because the
    // second-pass classifier needs the RAW first contact, which the reduction may
    // have discarded.
    const float4 staged = Ptr<float4>(inBase + 128UL)[lane];
    const float3 rawContact0 = Ptr<float4>(inBase + 128UL)[0].xyz;

    cteEmitTail(lane,
                w0.x != 0u, w0.y != 0u, w0.z, staged, rawContact0, minNormal,
                triVert0, triVert1, triVert2,
                w1.x, w1.y, w1.z, w1.w, w2.x,
                gCte.cvxTriNI, gCte.cvxTriContacts, gCte.cvxTriMaxDepth,
                gCte.cvxTriInterm, gCte.orderedCvxTriInterm,
                gCte.convTriSecondPassPairs, gCte.nbSecondPassPairs,
                gCte.tempConvexTriContacts, w0.w, gCte.pTempContactIndex);
}
