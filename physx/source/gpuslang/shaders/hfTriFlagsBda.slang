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

// hfTriFlagsBda.slang
//
// Test scaffolding for pxgSlangHfTriFlags -- NOT an engine kernel, absent from the
// manifest.  P9.52b-1.
//
// It drives the WHOLE path the capsule arm will take: fetch the current triangle
// out of the sample grid, form its normal the way the .cu does
// (`denormalizedNormal().getNormalized()`), classify the three edges against their
// neighbours, and then ask `selectNormal` what that classification means for a
// given barycentric pair.  Splitting those would let the fetch and the
// classification agree with each other about a triangle neither of them fetched
// correctly.
//
// Output record, 32 bytes:
//   @0   uint  triFlags
//   @4   uint  selectNormal (0/1)
//   @8   uint  triangleIdx  (echoed, so a mis-indexed case is visible)
//   @12  uint  _pad
//   @16  uint4 adj.xyz, _pad -- the permuted adjacency the classification saw

import pxgSlangHfTriFlags;
import pxgSlangHeightfield;
import pxgSlangPointTriangle;

struct HfTriFlagsParams
{
    uint64_t samples;   // @0   the sample grid (blob + 8)
    uint64_t cases;     // @8
    uint64_t output;    // @16
    uint     numCases;  // @24
    uint     rows;      // @28
    uint     columns;   // @32
    uint     _pad36;
}

// One case: the scale, the triangle, and the barycentric pair for selectNormal.
struct HfTriFlagsCase
{
    float4 scale;   // xyz = (rowScale, heightScale, columnScale)
    float4 uv;      // x = u, y = v
    uint4  cfg;     // x = triangleIdx
}

[[vk::push_constant]]
ConstantBuffer<HfTriFlagsParams> gHtf;

static const uint HTF_OUT_STRIDE = 32u;

[shader("compute")]
[numthreads(64, 1, 1)]
void hfTriFlagsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint c = dtid.x;
    if (c >= gHtf.numCases)
        return;

    const HfTriFlagsCase cs = Ptr<HfTriFlagsCase>(gHtf.cases)[c];
    const float3 scale = cs.scale.xyz;
    const uint triangleIdx = cs.cfg.x;

    const HfTriangle t = hfGetTriangleAdj(gHtf.samples, triangleIdx, scale,
                                          gHtf.rows, gHtf.columns);
    const float3 triNormal = ptNormalized(ptCross(t.v1 - t.v0, t.v2 - t.v0));

    const uint triFlags = htfTriangleFlags(gHtf.samples, t.adj, t.v0, t.v1, t.v2,
                                           triNormal, scale, gHtf.columns);
    const bool sel = htfSelectNormal(cs.uv.x, cs.uv.y, triFlags);

    const uint64_t o = gHtf.output + uint64_t(c) * uint64_t(HTF_OUT_STRIDE);
    Ptr<uint>(o)[0] = triFlags;
    Ptr<uint>(o)[1] = sel ? 1u : 0u;
    Ptr<uint>(o)[2] = triangleIdx;
    Ptr<uint>(o)[3] = 0u;
    Ptr<uint4>(o + 16UL)[0] = uint4(t.adj, 0u);
}
