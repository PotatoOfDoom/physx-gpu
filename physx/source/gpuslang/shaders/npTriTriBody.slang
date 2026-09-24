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

// npTriTriBody.slang -- P9.83b-8 (2/n).
//
//   Fixture entries for pxgSlangTriTriBody: serial lanes, no texture.  The SDF
//   call between the two halves of each pass is certified elsewhere
//   (gpuslang.sdfTriangleCollision); here its outputs -- sep, dir, point -- are
//   INPUTS, chosen by the host to reach every branch of the post-processing.
//
//   ttcVertsProbe   ttcPass0Vertices and ttcPass1Vertices over one BDA mesh
//   ttcPostProbe    ttcPass0Post (misc.w bit 0 clear) or ttcPass1Post (set)
//
// Stamped NoContraction (see npTriTriGeom.slang for why).

import pxgSlangTriTriBody;

struct TtcParams
{
    uint numCases;
    uint pad0, pad1, pad2;
    uint64_t cases;
    uint64_t outBuf;
    uint64_t meshInd;
    uint64_t meshVerts;
}

[[vk::push_constant]] ConstantBuffer<TtcParams> gP;

// case: float4 scale0, rot0, scale1, rot1, aToBp, aToBq; uint4 (ind, subInd, 0, 0) = 112 B
// out:  float4 p0v0, p0v1, p0v2, p1v0, p1v1, p1v2                            = 96 B
struct TtcVertsCase
{
    float4 scale0, rot0, scale1, rot1, aToBp, aToBq;
    uint4 misc;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ttcVertsProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numCases) return;
    const TtcVertsCase cs = Ptr<TtcVertsCase>(gP.cases)[i];

    float3 a0, a1, a2, b0, b1, b2;
    ttcPass0Vertices(gP.meshInd, gP.meshVerts, cs.scale0.xyz, cs.rot0, cs.aToBp.xyz, cs.aToBq,
                     cs.misc.x, cs.misc.y, cs.scale1.xyz, cs.rot1, a0, a1, a2);
    ttcPass1Vertices(gP.meshInd, gP.meshVerts, cs.scale1.xyz, cs.rot1, cs.aToBp.xyz, cs.aToBq,
                     cs.misc.x, cs.scale0.xyz, cs.rot0, b0, b1, b2);

    Ptr<uint4> o = Ptr<uint4>(gP.outBuf) + i * 6u;
    o[0] = uint4(asuint(a0), 0u); o[1] = uint4(asuint(a1), 0u); o[2] = uint4(asuint(a2), 0u);
    o[3] = uint4(asuint(b0), 0u); o[4] = uint4(asuint(b1), 0u); o[5] = uint4(asuint(b2), 0u);
}

// case: float4 v0, v1, v2 (xyz); float4 dir (xyz), w = sep; float4 point (xyz);
//       float4 boxLo, boxHi; float4 scale (xyz), rot; float4 aToBp, aToBq;
//       float4 (cullScale, contactDistance, refinementRatioSquared, 0);
//       uint4 (0, 0, 0, flags: bit0 = pass 1, bit1 = mesh0HasSdf)          = 208 B
// out:  float4 pos (w = 0); float4 normal (w = separation);
//       uint4 (candidate, needsRefinement, 0, 0)                            = 48 B
struct TtcPostCase
{
    float4 v0, v1, v2;
    float4 dirSep;
    float4 point;
    float4 boxLo, boxHi;
    float4 scale, rot;
    float4 aToBp, aToBq;
    float4 dists;
    uint4 misc;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ttcPostProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numCases) return;
    const TtcPostCase cs = Ptr<TtcPostCase>(gP.cases)[i];

    // The CUDA's fresh `Contact contact;` (:1070, :755-758), with pos overwritten
    // by the SDF call's point.
    float3 pos = cs.point.xyz;
    float separation = TTC_FLT_MAX;
    float3 normal = float3(0.0f, 0.0f, 0.0f);
    bool candidate = false;
    bool needsRefinement = false;

    if ((cs.misc.w & 1u) == 0u)
        ttcPass0Post(cs.v0.xyz, cs.v1.xyz, cs.v2.xyz, cs.dirSep.w, cs.dirSep.xyz,
                     cs.boxLo.xyz, cs.boxHi.xyz, cs.scale.xyz, cs.rot, (cs.misc.w & 2u) != 0u,
                     cs.dists.x, cs.dists.y, cs.dists.z,
                     pos, separation, normal, candidate, needsRefinement);
    else
        ttcPass1Post(cs.dirSep.w, cs.dirSep.xyz, cs.scale.xyz, cs.rot, cs.aToBp.xyz, cs.aToBq,
                     cs.dists.x, cs.dists.y, pos, separation, normal, candidate);

    Ptr<uint4> o = Ptr<uint4>(gP.outBuf) + i * 3u;
    o[0] = uint4(asuint(pos), 0u);
    o[1] = uint4(asuint(normal), asuint(separation));
    o[2] = uint4(candidate ? 1u : 0u, needsRefinement ? 1u : 0u, 0u, 0u);
}
