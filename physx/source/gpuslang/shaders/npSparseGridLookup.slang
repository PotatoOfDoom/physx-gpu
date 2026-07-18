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

// npSparseGridLookup.slang
//
// Slang port of the sparse-grid sorted-hash lookup and thread-to-cell
// decomposition from sparseGridStandalone.cuh (pure integer, no float arith):
//
//   searchSorted (L82-98):    left = 0; right = numElements;
//                             while (right - left > 1):
//                               pos = (left + right) >> 1
//                               if (data[pos] <= value) left = pos else right = pos
//                             return left
//   tryFindHashkey (L100-104): result = searchSorted(keys, num, hash)
//                              found  = (keys[result] == hash)
//   getGridCoordinates (L213-237, compile-time haloSize = 0):
//     numSubgridCells = sX*sY*sZ            (PxU16 sizes promote to int)
//     si = threadIndex / numSubgridCells    (int / PxU32 -> unsigned division)
//     local = threadIndex - si*numSubgridCells
//     xi = local % sX;  yi = (local / sX) % sY;  zi = local / (sX*sY)
//     return (xi, yi, zi, si)
//
// The candidate key array is fixed at M=16 slots (numElements <= 16); the
// binary search is a data-dependent but bounded loop.  All arithmetic is
// integer, so this shader is NOT in the NoContraction list.

static const uint M = 16u;

struct LkIn
{
    uint4 keys[4];   // 16 sorted hash keys, keys[i>>2][i&3]
    uint4 misc;      // numElements (1..16), search value, threadIndex, _
    uint4 sizes;     // subgridSizeX, subgridSizeY, subgridSizeZ, _
}

struct LkOut
{
    uint4 res;       // searchSorted index, found (1/0), _, _
    int4 coords;     // xi, yi, zi, si
}

struct LkParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<LkParams> gParams;

RWStructuredBuffer<LkIn> gIn;    // binding 0
RWStructuredBuffer<LkOut> gOut;  // binding 1

uint keyAt(LkIn s, uint i) { return s.keys[i >> 2][i & 3u]; }

// sparseGridStandalone.cuh searchSorted (L82-98).
uint searchSorted(LkIn s, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while ((right - left) > 1u)
    {
        const uint pos = (left + right) >> 1;
        const uint element = keyAt(s, pos);
        if (element <= value)
            left = pos;
        else
            right = pos;
    }
    return left;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sparseGridLookup(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const LkIn s = gIn[tid];

    const uint numElements = s.misc.x;
    const uint value = s.misc.y;
    const uint threadIndex = s.misc.z;

    // tryFindHashkey
    const uint result = searchSorted(s, numElements, value);
    const uint found = (keyAt(s, result) == value) ? 1u : 0u;

    // getGridCoordinates (haloSize = 0)
    const int sX = int(s.sizes.x), sY = int(s.sizes.y), sZ = int(s.sizes.z);
    const uint numSubgridCells = uint(sX * sY * sZ);
    const uint si = threadIndex / numSubgridCells;
    const int localThreadIndex = int(threadIndex - si * numSubgridCells);
    const int xi = localThreadIndex % sX;
    const int yi = (localThreadIndex / sX) % sY;
    const int zi = localThreadIndex / (sX * sY);

    LkOut o;
    o.res = uint4(result, found, 0u, 0u);
    o.coords = int4(xi, yi, zi, int(si));
    gOut[tid] = o;
}
