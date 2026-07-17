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

// npSparseGridHash.slang
//
// Slang port of the sparse-grid subgrid hash/index arithmetic from
// sparseGridStandalone.cuh: calcSubgridHash (L62-66), subgridHashToId
// (L114-121), subgridNeighborIndex (L76-79), isSubgridInsideRange (L106-112),
// mod (L128-132) and sparseGridAccess (L134-139).
//   calcSubgridHash(id): shifted = id - MIN_ID; DIM*DIM*sz + DIM*sy + sx
//   subgridHashToId(h): (h%DIM, (h/DIM)%DIM, h/DIM/DIM) + MIN_ID  [round-trip]
//   subgridNeighborIndex(x,y,z) = (x+1) + 3*(y+1) + 9*(z+1)
//   isSubgridInsideRange(v) = all of v in [MIN_ID, MAX_ID]
//   mod(a,b) = (a+b) % b
//   sparseGridAccess(i,j,k,si) = i + sX*(j + sY*(k + sZ*si))
//
// Pure integer arithmetic (exact by construction).

static const int MAX_SPARSEGRID_DIM = 1024;
static const int MIN_SPARSEGRID_ID = -512;
static const int MAX_SPARSEGRID_ID = 511;

uint calcSubgridHash(int3 id)
{
    int3 shifted = int3(id.x - MIN_SPARSEGRID_ID, id.y - MIN_SPARSEGRID_ID, id.z - MIN_SPARSEGRID_ID);
    return uint(MAX_SPARSEGRID_DIM * MAX_SPARSEGRID_DIM * shifted.z + MAX_SPARSEGRID_DIM * shifted.y + shifted.x);
}

int4 subgridHashToId(uint hashKey)
{
    int ih = int(hashKey);
    return int4(
        ih % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
        (ih / MAX_SPARSEGRID_DIM) % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
        ih / MAX_SPARSEGRID_DIM / MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID, 0);
}

int subgridNeighborIndex(int x, int y, int z) { return ((x + 1) + 3 * (y + 1) + 9 * (z + 1)); }

bool isSubgridInsideRange(int3 v)
{
    return v.x >= MIN_SPARSEGRID_ID && v.x <= MAX_SPARSEGRID_ID &&
           v.y >= MIN_SPARSEGRID_ID && v.y <= MAX_SPARSEGRID_ID &&
           v.z >= MIN_SPARSEGRID_ID && v.z <= MAX_SPARSEGRID_ID;
}

int imod(int a, int b) { return (a + b) % b; }

uint sparseGridAccess(int i, int j, int k, int si, int sX, int sY, int sZ)
{
    return uint(i + sX * (j + sY * (k + sZ * si)));
}

struct SgIn
{
    int4 id;      // subgrid id x,y,z, _
    int4 nb;      // neighbor offsets x,y,z, _
    int4 modab;   // mod a, b, _, _
    int4 acc;     // access i,j,k,si
    int4 sizes;   // subgrid sX,sY,sZ, _
}

struct SgOut
{
    uint4 hash;     // calcSubgridHash(id), _, _, _
    int4 rtId;      // subgridHashToId(hash)
    int4 misc;      // subgridNeighborIndex, imod, isInside(0/1), _
    uint4 access;   // sparseGridAccess, _, _, _
}

struct SgParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SgParams> gParams;

RWStructuredBuffer<SgIn> gIn;    // binding 0
RWStructuredBuffer<SgOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sparseGridHash(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SgIn s = gIn[tid];

    const int3 id = int3(s.id.x, s.id.y, s.id.z);
    const uint h = calcSubgridHash(id);
    const int4 rt = subgridHashToId(h);
    const int ni = subgridNeighborIndex(s.nb.x, s.nb.y, s.nb.z);
    const int md = imod(s.modab.x, s.modab.y);
    const bool inside = isSubgridInsideRange(id);
    const uint acc = sparseGridAccess(s.acc.x, s.acc.y, s.acc.z, s.acc.w, s.sizes.x, s.sizes.y, s.sizes.z);

    SgOut o;
    o.hash = uint4(h, 0u, 0u, 0u);
    o.rtId = rt;
    o.misc = int4(ni, md, inside ? 1 : 0, 0);
    o.access = uint4(acc, 0u, 0u, 0u);
    gOut[tid] = o;
}
