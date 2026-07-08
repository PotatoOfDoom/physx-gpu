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

// radixFourBit.slang
//
// Slang port of the PxGpuRadixSort kernels from
// gpusimulationcontroller/src/CUDA/algorithms.cu:
//   - radixFourBitCountPerBlockKernel
//   - radixFourBitReorderKernel
//   - reorderKernel (float4 gather; separate bindings, see reorderData.slang)
//
// Together with scan4x4.slang these implement PxGpuRadixSort::sort
// (PxgAlgorithms.cpp): per pass,
//   1. count      -> per-element within-block slot ranks (gOffsets) +
//                    per-block slot histograms (gPartialSums) + last-block
//                    inclusive totals (gTotalSum)
//   2. scan4x4    -> exclusive scan of the per-block histograms + slot
//                    cumulative sums (computeBlockSum composition)
//   3. reorder    -> stable scatter using cumulative[slot] + blockBase[slot]
//                    + within-block rank
//
// Differences from CUDA:
//   - The CUDA count kernel ranks slots via PxU64-packed 10-bit counters
//     (3 scan rounds x 6 slots).  The port scans uint4 lanes (4 rounds x 4
//     slots), which avoids the shaderInt64 requirement and produces the
//     identical ranks/histograms.
//   - The PxU16 offsets buffer is widened to 32-bit (16-bit storage buffers
//     are an optional Vulkan feature).
//   - Workgroup size fixed at 512 (the blockDim PxGpuRadixSort launches
//     with); the reorder kernel is grid-size independent.

import pxgSlangScan;

static const uint BLOCK_SIZE = 512;

struct Int4x4
{
    uint4 d0, d1, d2, d3;
};

struct RadixFourBitParams
{
    uint length;
    uint passIndex;
    uint writeTotalSum; // count: totalSum buffer valid
    uint hasDependent;  // reorder: dependent data buffers valid
};

[[vk::push_constant]]
ConstantBuffer<RadixFourBitParams> gParams;

RWStructuredBuffer<uint> gData;               // binding 0: keys (current pass input)
RWStructuredBuffer<uint> gOffsets;            // binding 1: within-block slot ranks (PxU16 in CUDA)
RWStructuredBuffer<uint> gReordered;          // binding 2: reordered keys
RWStructuredBuffer<Int4x4> gPartialSums;      // binding 3: per-block slot histograms / scanned bases
RWStructuredBuffer<Int4x4> gCumulativeSum;    // binding 4: slot cumulative sums (single entry)
RWStructuredBuffer<uint> gDependentData;      // binding 5: reorder tracking (input)
RWStructuredBuffer<uint> gDependentReordered; // binding 6: reorder tracking (output)

uint int4x4At(Int4x4 v, uint i)
{
    const uint4 q = i < 4 ? v.d0 : (i < 8 ? v.d1 : (i < 12 ? v.d2 : v.d3));
    return q[i & 3];
}

// mirrors radixFourBitCountPerBlock<PxU32>() in algorithms.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixFourBitCountPerBlockKernel(uint3 dispatchThreadId : SV_DispatchThreadID,
                                     uint3 groupThreadId : SV_GroupThreadID,
                                     uint3 groupId : SV_GroupID)
{
    const uint id = dispatchThreadId.x;
    const uint tid = groupThreadId.x;
    const uint length = gParams.length;

    uint slot = 0xFFFFFFFFu;
    if (id < length)
        slot = (gData[id] >> (gParams.passIndex * 4)) & 15u;

    // rank the 16 slots in 4 rounds of 4 (uint4 lanes), mirroring the CUDA
    // 3-round PxU64 packing
    for (uint group = 0; group < 16; group += 4)
    {
        uint4 v = uint4(0);
        const uint rel = slot - group;
        if (rel < 4)
            v[rel] = 1;

        const uint4 incl = blockInclusiveScanU4<BLOCK_SIZE>(tid, v);

        if (tid == BLOCK_SIZE - 1)
        {
            if (group == 0)       gPartialSums[groupId.x].d0 = incl;
            else if (group == 4)  gPartialSums[groupId.x].d1 = incl;
            else if (group == 8)  gPartialSums[groupId.x].d2 = incl;
            else                  gPartialSums[groupId.x].d3 = incl;
        }

        if (gParams.writeTotalSum != 0 && id == length - 1)
        {
            if (group == 0)       gCumulativeSum[0].d0 = incl;
            else if (group == 4)  gCumulativeSum[0].d1 = incl;
            else if (group == 8)  gCumulativeSum[0].d2 = incl;
            else                  gCumulativeSum[0].d3 = incl;
        }

        if (rel < 4 && id < length)
            gOffsets[id] = incl[rel] - 1;
    }
}

// mirrors radixFourBitReorder<PxU32, PxU32>() in algorithms.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixFourBitReorderKernel(uint3 dispatchThreadId : SV_DispatchThreadID,
                               uint3 groupId : SV_GroupID)
{
    const uint id = dispatchThreadId.x;
    const uint length = gParams.length;

    if (id >= length)
        return;

    const uint slot = (gData[id] >> (gParams.passIndex * 4)) & 15u;

    const uint newIndex = int4x4At(gCumulativeSum[0], slot) + gOffsets[id] +
                          int4x4At(gPartialSums[groupId.x], slot);

    // This condition should always be met but in case everything goes wrong,
    // it ensures that no out of bounds access happens
    if (newIndex < length)
    {
        gReordered[newIndex] = gData[id];

        if (gParams.hasDependent != 0)
            gDependentReordered[newIndex] = gParams.passIndex == 0 ? id : gDependentData[id];
    }
}
