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

// npSparseGridDomain.slang
//
// Slang port of getSubgridDomainSize + calcSubgridId (sparseGridStandalone.cuh
// L51-59): the sparse-grid subgrid domain size and the subgrid id of a point.
//   getSubgridDomainSize(params, haloSize):
//     dx = gridSpacing
//     return (dx*(subgridSizeX - 2*haloSize),
//             dx*(subgridSizeY - 2*haloSize),
//             dx*(subgridSizeZ - 2*haloSize))
//   calcSubgridId(pos, domainSize):
//     return ((int)floor(pos.x/domainSize.x),
//             (int)floor(pos.y/domainSize.y),
//             (int)floor(pos.z/domainSize.z))
//
// The (subgridSize - 2*haloSize) is SIGNED integer arithmetic in CUDA (the
// PxU16 fields promote to int), converted to float before the dx multiply; we
// mirror that with int(subX) - 2*int(haloSize) so the transcription matches
// CUDA's integer model exactly (bit-identical to the unsigned form for the
// non-negative values that arise, since float(v) agrees for v >= 0).  PxFloor
// is IEEE floor; the (int) cast truncates a floor-integral float exactly.
// Float mul/div NoContraction-stamped; +-0 eqFZ on domainSize; exact int id.

struct GdIn
{
    float4 params;   // gridSpacing, asfloat(haloSize), _, _
    float4 sub;      // asfloat(subX), asfloat(subY), asfloat(subZ), _
    float4 pos;      // .xyz query point
}

struct GdOut
{
    float4 domainSize;   // .xyz
    int4 subgridId;      // .xyz
}

struct GdParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<GdParams> gParams;

RWStructuredBuffer<GdIn> gIn;    // binding 0
RWStructuredBuffer<GdOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sparseGridDomain(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const GdIn s = gIn[tid];

    const float dx = s.params.x;
    const uint haloSize = asuint(s.params.y);
    const uint subX = asuint(s.sub.x), subY = asuint(s.sub.y), subZ = asuint(s.sub.z);
    const float3 pos = s.pos.xyz;

    // getSubgridDomainSize
    const int halo = int(haloSize);
    const float3 domainSize = float3(dx * float(int(subX) - 2 * halo),
                                     dx * float(int(subY) - 2 * halo),
                                     dx * float(int(subZ) - 2 * halo));

    // calcSubgridId
    const int3 subgridId = int3(int(floor(pos.x / domainSize.x)),
                                int(floor(pos.y / domainSize.y)),
                                int(floor(pos.z / domainSize.z)));

    GdOut o;
    o.domainSize = float4(domainSize, 0.0f);
    o.subgridId = int4(subgridId, 0);
    gOut[tid] = o;
}
