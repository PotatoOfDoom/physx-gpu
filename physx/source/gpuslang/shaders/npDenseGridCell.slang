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

// npDenseGridCell.slang
//
// Slang port of getCellIndexFromParticleAndTransformToLocalCoordinates
// (denseGridStandalone.cuh L123-131): translate a particle into dense-grid-local
// coordinates and return its integer cell index.
//   p    = p - origin                 // p is modified in place (local coords out)
//   invDx= 1/gridSpacing
//   cxi  = (int)floor(p.x * invDx)
//   cyi  = (int)floor(p.y * invDx)
//   czi  = (int)floor(p.z * invDx)
//   return int4(cxi, cyi, czi, -1)
//
// PxFloor is IEEE floor; the (int) cast truncates the floor-integral float
// exactly.  The returned .w sentinel is -1.  Float sub/mul NoContraction-
// stamped; +-0 eqFZ on the local coords; exact match on the int cell.

struct DcIn
{
    float4 p;        // .xyz particle
    float4 origin;   // .xyz grid origin
    float4 params;   // gridSpacing, _, _, _
}

struct DcOut
{
    float4 pLocal;   // .xyz p - origin
    int4 cell;       // cell index (.w = -1)
}

struct DcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<DcParams> gParams;

RWStructuredBuffer<DcIn> gIn;    // binding 0
RWStructuredBuffer<DcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void denseGridCell(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const DcIn s = gIn[tid];

    float3 p = s.p.xyz;
    const float3 origin = s.origin.xyz;
    const float gridSpacing = s.params.x;

    p = p - origin;
    const float invDx = 1.0f / gridSpacing;
    const int cxi = int(floor(p.x * invDx));
    const int cyi = int(floor(p.y * invDx));
    const int czi = int(floor(p.z * invDx));

    DcOut o;
    o.pLocal = float4(p, 0.0f);
    o.cell = int4(cxi, cyi, czi, -1);
    gOut[tid] = o;
}
