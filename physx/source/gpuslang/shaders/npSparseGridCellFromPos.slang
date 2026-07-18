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

// npSparseGridCellFromPos.slang
//
// Slang port of the full sparse-grid position-to-cell composition
// getCellIndexFromPosition (sparseGridStandalone.cuh L171-200, compile-time
// haloSize = 0), composing the previously ported pieces end-to-end:
//
//   domainSize = getSubgridDomainSize(params, 0)      // dx * subgridSize_i
//   subgridId  = calcSubgridId(p, domainSize)         // (int)PxFloor(p_i / domain_i)
//                -- NOTE: plain per-component DIVISION, not reciprocal-multiply
//   hash       = calcSubgridHash(subgridId)           // DIM^2*(z+512) + DIM*(y+512) + (x+512)
//   found      = tryFindHashkey(keys, numSubgrids, hash, sortedIdx)
//   if (!found) return (-1,-1,-1, OUT_OF_BOUNDS=-1)   // p left UNCHANGED
//   invDx = 1/dx
//   subgridOrigin_i = (subgridId_i * dx) * subgridSize_i    // left-to-right
//   p = p - subgridOrigin                              // p becomes cell-local
//   result_i = PxI32(PxFloor(p_i * invDx)) clamped to [0, subgridSize_i - 1]
//   return (result.xyz, sortedIdx)
//
// PxClamp(v,lo,hi) = PxMin(hi, PxMax(lo, v)) with the generic integer templates
// PxMax(a,b) = a<b?b:a and PxMin(a,b) = a<b?a:b.  PxFloor = floorf (IEEE).
// The guarded `1/dx` is one rounding; the calcSubgridId divisions are three
// independent single-rounding divides.  subgridSize are PxU16 (promote to int).
// Float div/mul/sub/floor NoContraction-stamped; +-0 eqFZ on the local p.

static const int MAX_SPARSEGRID_DIM = 1024;
static const int MIN_SPARSEGRID_ID = -512;

int imax(int a, int b) { return (a < b) ? b : a; }   // generic PxMax
int imin(int a, int b) { return (a < b) ? a : b; }   // generic PxMin
int iclamp(int v, int lo, int hi) { return imin(hi, imax(lo, v)); }   // PxClamp

struct CiIn
{
    float4 pos;      // p.xyz, gridSpacing (dx)
    uint4 sizes;     // subgridSizeX, subgridSizeY, subgridSizeZ, numSubgridsInUse
    uint4 keys[4];   // 16 sorted unique subgrid hash keys
}

struct CiOut
{
    int4 cell;       // result.xyz, sortedIdx  (or -1,-1,-1,-1 when hash not found)
    float4 plocal;   // p after the call (cell-local when found, original otherwise)
}

struct CiParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CiParams> gParams;

RWStructuredBuffer<CiIn> gIn;    // binding 0
RWStructuredBuffer<CiOut> gOut;  // binding 1

uint keyAt(CiIn s, uint i) { return s.keys[i >> 2][i & 3u]; }

// sparseGridStandalone.cuh searchSorted (L82-98).
uint searchSorted(CiIn s, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while ((right - left) > 1u)
    {
        const uint pos = (left + right) >> 1;
        if (keyAt(s, pos) <= value)
            left = pos;
        else
            right = pos;
    }
    return left;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sparseGridCellFromPos(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CiIn s = gIn[tid];

    float3 p = s.pos.xyz;
    const float dx = s.pos.w;
    const int sX = int(s.sizes.x), sY = int(s.sizes.y), sZ = int(s.sizes.z);
    const uint numSubgrids = s.sizes.w;

    // getSubgridDomainSize (haloSize = 0): dx * size_i
    const float3 domainSize = float3(dx * float(sX), dx * float(sY), dx * float(sZ));

    // calcSubgridId: (int)PxFloor(p_i / domain_i) -- plain per-component division
    const int3 subgridId = int3(int(floor(p.x / domainSize.x)),
                                int(floor(p.y / domainSize.y)),
                                int(floor(p.z / domainSize.z)));

    // calcSubgridHash
    const int shx = subgridId.x - MIN_SPARSEGRID_ID;
    const int shy = subgridId.y - MIN_SPARSEGRID_ID;
    const int shz = subgridId.z - MIN_SPARSEGRID_ID;
    const uint subgridHash = uint(MAX_SPARSEGRID_DIM * MAX_SPARSEGRID_DIM * shz + MAX_SPARSEGRID_DIM * shy + shx);

    // tryFindHashkey
    const uint sortedIdx = searchSorted(s, numSubgrids, subgridHash);
    const bool hashFound = (keyAt(s, sortedIdx) == subgridHash);

    CiOut o;
    if (!hashFound)
    {
        o.cell = int4(-1, -1, -1, -1);            // OUT_OF_BOUNDS = -1
        o.plocal = float4(p, 0.0f);               // p unchanged
        gOut[tid] = o;
        return;
    }

    const float invDx = 1.0f / dx;

    // subgridOrigin_i = (subgridId_i * dx) * size_i, left-to-right
    const float3 subgridOrigin = float3(float(subgridId.x) * dx * float(sX),
                                        float(subgridId.y) * dx * float(sY),
                                        float(subgridId.z) * dx * float(sZ));
    p = p - subgridOrigin;

    int rx = int(floor(p.x * invDx));
    int ry = int(floor(p.y * invDx));
    int rz = int(floor(p.z * invDx));
    rx = iclamp(rx, 0, sX - 1);
    ry = iclamp(ry, 0, sY - 1);
    rz = iclamp(rz, 0, sZ - 1);

    o.cell = int4(rx, ry, rz, int(sortedIdx));
    o.plocal = float4(p, 0.0f);
    gOut[tid] = o;
}
