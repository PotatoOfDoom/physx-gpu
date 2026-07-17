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

// npGridLocation.slang
//
// Slang port of the two grid cell-location functions that map an integer cell
// index to a world-space position:
//
//   denseGridStandalone.cuh getLocation (L118-121):
//     origin + PxVec3(index.x, index.y, index.z) * gridSpacing
//
//   sparseGridStandalone.cuh getLocationFromHashkey (L202-211), with the
//   compile-time haloSize = 0:
//     subgridId   = subgridHashToId(hash)   (decode; see npSparseGridHash)
//     subgridOrigin = ( subgridId.x * gridSpacing * subgridSizeX,
//                       subgridId.y * gridSpacing * subgridSizeY,
//                       subgridId.z * gridSpacing * subgridSizeZ )
//     return subgridOrigin + PxVec3(index.x, index.y, index.z) * gridSpacing
//
// PxVec3(int,int,int) converts each component to float; PxVec3::operator*(s) is
// a direct per-component multiply and operator+ a per-component add, so each
// output component is origin_i + float(index_i) * dx (one mul, one add) and for
// the sparse origin ((float(id_i) * dx) * float(size_i)) evaluated left to
// right.  subgridSize{X,Y,Z} are PxU16 and promote to int in the CUDA
// subtraction (subgridSize - 2*haloSize), so they enter the float product as
// float(int(size)).  All float mul/add NoContraction-stamped (the origin + idx*dx
// forms are prime FMA-fusion candidates); +-0 compared with eqFZ.

static const int MAX_SPARSEGRID_DIM = 1024;
static const int MIN_SPARSEGRID_ID = -512;

// sparseGridStandalone.cuh subgridHashToId (L114-121).
int3 subgridHashToId(uint hashKey)
{
    const int ihashKey = int(hashKey);
    return int3(ihashKey % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
                (ihashKey / MAX_SPARSEGRID_DIM) % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
                ihashKey / MAX_SPARSEGRID_DIM / MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID);
}

struct GlIn
{
    float4 origin;   // dense origin.xyz, _
    float4 params;   // denseSpacing, sparseSpacing, asfloat(hash), _
    uint4 sizes;     // subgridSizeX, subgridSizeY, subgridSizeZ, _
    int4 index;      // index.xyz (shared by both grids), _
}

struct GlOut
{
    float4 denseLoc;    // dense getLocation.xyz
    float4 sparseLoc;   // sparse getLocationFromHashkey.xyz
}

struct GlParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<GlParams> gParams;

RWStructuredBuffer<GlIn> gIn;    // binding 0
RWStructuredBuffer<GlOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void gridLocation(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const GlIn s = gIn[tid];

    const float3 idxf = float3(float(s.index.x), float(s.index.y), float(s.index.z));

    // dense getLocation
    const float ddx = s.params.x;
    const float3 denseLoc = s.origin.xyz + idxf * ddx;

    // sparse getLocationFromHashkey (haloSize = 0)
    const float sdx = s.params.y;
    const int3 sid = subgridHashToId(asuint(s.params.z));
    const int szx = int(s.sizes.x), szy = int(s.sizes.y), szz = int(s.sizes.z);
    const float3 subgridOrigin = float3(float(sid.x) * sdx * float(szx),
                                        float(sid.y) * sdx * float(szy),
                                        float(sid.z) * sdx * float(szz));
    const float3 sparseLoc = subgridOrigin + idxf * sdx;

    GlOut o;
    o.denseLoc = float4(denseLoc, 0.0f);
    o.sparseLoc = float4(sparseLoc, 0.0f);
    gOut[tid] = o;
}
