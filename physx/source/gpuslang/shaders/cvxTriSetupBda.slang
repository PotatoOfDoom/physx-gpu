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

// cvxTriSetupBda.slang
//
// Test scaffolding for pxgSlangCvxTriSetup — NOT an engine kernel, absent from
// the manifest.  One workgroup of 32 threads per pair; only lane 0 reports,
// because the module is warp-uniform by construction and a per-lane report would
// only re-measure that.
//
// The pair records come from a device array so the fixture can drive several
// pairs of one dispatch, exactly as the engine kernel will.
//
// Output record, 128 bytes:
//   @0    uint2  hull address | uint remapCpuTriangleIdx | uint _pad
//   @16   float4 convexScale.xyz, contactDist
//   @32   float4 convexScaleRot
//   @48   float4 triLocVerts0.xyz, _pad
//   @64   float4 triLocVerts1.xyz, _pad
//   @80   float4 triLocVerts2.xyz, _pad
//   @96   float4 triangleLocNormal.xyz, _pad
//   @112  uint4  triAdjTrisIdx.xyz | convexTriPairOffset
//   (convexTriPairOffsetPadded rides in the _pad word at @12)

import pxgSlangCvxTriSetup;

struct CvxTriSetupParams
{
    uint64_t cmInputs;        // @0
    uint64_t transformCache;  // @8
    uint64_t contactDistance; // @16
    uint64_t gpuShapes;       // @24
    uint64_t cvxTrimeshPair;  // @32
    uint64_t pairs;           // @40
    uint64_t output;          // @48
    uint     numPairs;        // @56
    uint     _pad60;
}

[[vk::push_constant]]
ConstantBuffer<CvxTriSetupParams> gCs;

static const uint CTS_OUT_STRIDE = 128u;

[shader("compute")]
[numthreads(32, 1, 1)]
void cvxTriSetupLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint p = gid.x;

    if (p >= gCs.numPairs)
        return;

    const uint4 curPair = Ptr<uint4>(gCs.pairs)[p];

    const CtSetup s = ctSetup(gCs.cmInputs, gCs.transformCache, gCs.contactDistance,
                              gCs.gpuShapes, gCs.cvxTrimeshPair, curPair);

    if (lane != 0u)
        return;

    const uint64_t o = gCs.output + uint64_t(p) * uint64_t(CTS_OUT_STRIDE);
    Ptr<uint>(o)[0] = uint(s.hull & 0xffffffffUL);
    Ptr<uint>(o)[1] = uint(s.hull >> 32);
    Ptr<uint>(o)[2] = s.remapCpuTriangleIdx;
    Ptr<uint>(o)[3] = s.convexTriPairOffsetPadded;
    Ptr<float4>(o + 16UL)[0] = float4(s.convexScale, s.contactDist);
    Ptr<float4>(o + 32UL)[0] = s.convexScaleRot;
    Ptr<float4>(o + 48UL)[0] = float4(s.triLocVerts0, 0.0f);
    Ptr<float4>(o + 64UL)[0] = float4(s.triLocVerts1, 0.0f);
    Ptr<float4>(o + 80UL)[0] = float4(s.triLocVerts2, 0.0f);
    Ptr<float4>(o + 96UL)[0] = float4(s.triangleLocNormal, 0.0f);
    Ptr<uint4>(o + 112UL)[0] = uint4(s.triAdjTrisIdx, s.convexTriPairOffset);
}
