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

// npPairRemoveSwap.slang
//
// Slang port of the sparse-remove data swap (Stage 5) from
// gpunarrowphase/src/CUDA/pairManagement.cu — removeContactManagers_Stage5 and
// removeContactManagers_Stage5_CvxTri.  Given the run-sum buffer built by the
// keep/drop stages (npPairManagement.slang), this moves each kept element that
// still lives in the removed tail down into the hole left by a removed element
// in the kept region, compacting the surviving pairs into [0, N-K):
//
//   nbToSwap = runSum[N - K]                          (getNbSwapsRequired)
//   for each swap i in [0, nbToSwap):
//     dstIndex = binarySearch(runSum, N, i)           (a removed slot in [0,N-K))
//     srcIndex = binarySearch(runSum, N, i+nbToSwap)  (a kept slot in [N-K,N))
//     payload[dstIndex] = payload[srcIndex]           (the six CM arrays)
//     manifold[dstIndex] = manifold[srcIndex]         (if copyManifold)
//
// binarySearch (gpucommon reduction.cuh): largest index s.t. data[index] <= value.
// The two entry points differ only in cooperation granularity and manifold size:
// Stage5 uses 16 threads/swap (persistent single manifold), Stage5_CvxTri uses 32
// (persistent multi-manifold).  Each swap is independent (dst in the kept region,
// src in the tail, all distinct), so mapping one thread group per swap is
// output-identical to the CUDA packing.  The six typed contact-manager arrays are
// swapped identically; a single representative uint payload array stands in for
// them here.

struct RemoveSwapParams
{
    uint totalCount;   // N
    uint nbToRemove;   // K
    uint nbFloat4;     // manifold size in float4
    uint copyManifold; // 0/1
    uint numGroups;    // dispatch group count (grid stride over swaps)
}

[[vk::push_constant]]
ConstantBuffer<RemoveSwapParams> gParams;

RWStructuredBuffer<uint>   gRunSum;    // binding 0  exclusive-prefix run-sum
RWStructuredBuffer<uint>   gPayload;   // binding 1  representative CM array
RWStructuredBuffer<float4> gManifolds; // binding 2  manifold blocks (nbFloat4 each)

// gpucommon reduction.cuh binarySearch(): largest index s.t. data[index] <= value,
// or 0 if data[0] > value
uint binarySearchRunSum(uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (gRunSum[pos] <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// ---------------------------------------------------------------------------
// removeContactManagers_Stage5 (16 threads per swap)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(16, 1, 1)]
void removeContactManagers_Stage5(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint n = gParams.totalCount;
    const uint nbToSwap = gRunSum[n - gParams.nbToRemove];
    const uint lane = gtid.x; // [0, 16)

    for (uint i = gid.x; i < nbToSwap; i += gParams.numGroups)
    {
        const uint dstIndex = binarySearchRunSum(n, i);
        const uint srcIndex = binarySearchRunSum(n, i + nbToSwap);

        if (lane == 0u)
            gPayload[dstIndex] = gPayload[srcIndex];

        if (gParams.copyManifold != 0u)
        {
            const uint nbFloat4 = gParams.nbFloat4;
            for (uint idx = lane; idx < nbFloat4; idx += 16u)
                gManifolds[dstIndex * nbFloat4 + idx] = gManifolds[srcIndex * nbFloat4 + idx];
        }
    }
}

// ---------------------------------------------------------------------------
// removeContactManagers_Stage5_CvxTri (32 threads per swap)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void removeContactManagers_Stage5_CvxTri(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint n = gParams.totalCount;
    const uint nbToSwap = gRunSum[n - gParams.nbToRemove];
    const uint lane = gtid.x; // [0, 32)

    for (uint i = gid.x; i < nbToSwap; i += gParams.numGroups)
    {
        const uint dstIndex = binarySearchRunSum(n, i);
        const uint srcIndex = binarySearchRunSum(n, i + nbToSwap);

        if (lane == 0u)
            gPayload[dstIndex] = gPayload[srcIndex];

        if (gParams.copyManifold != 0u)
        {
            const uint nbFloat4 = gParams.nbFloat4;
            for (uint idx = lane; idx < nbFloat4; idx += 32u)
                gManifolds[dstIndex * nbFloat4 + idx] = gManifolds[srcIndex * nbFloat4 + idx];
        }
    }
}
