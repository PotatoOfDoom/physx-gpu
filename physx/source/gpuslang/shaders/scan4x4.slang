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

// scan4x4.slang
//
// Slang port of the int4x4 (16-counter) prefix-sum kernels from
// gpusimulationcontroller/src/CUDA/algorithms.cu:
//   - scanPerBlockKernel4x4
//   - addBlockSumsKernel4x4
//
// These back PxGpuRadixSort (PxgAlgorithms.cpp): the radix pipeline scans
// per-block histograms of 16 radix slots as one int4x4 element per block.
//
// Differences from CUDA:
//   - The workgroup size is fixed at 512 (the launch-time blockDim of the
//     CUDA version); grid math on the host must match.
//   - gridDim.x-dependent logic (exclusiveSumInt16 on the total in the last
//     block / single-block case) uses the numGroups push constant.
//   - Warp shuffles are replaced by the portable groupshared scans.

import pxgSlangScan;

static const uint BLOCK_SIZE = 512;

// int4x4: 4 x int4 = 64 bytes
struct Int4x4
{
    uint4 d0, d1, d2, d3;
};

struct Scan4x4Params
{
    uint length;
    uint exclusiveScan;
    uint writeResult;   // result/data buffer valid (CUDA passed NULL to skip)
    uint writeTotalSum; // totalSum buffer valid
    uint numGroups;     // gridDim.x of this dispatch
};

[[vk::push_constant]]
ConstantBuffer<Scan4x4Params> gParams;

RWStructuredBuffer<Int4x4> gData;        // binding 0
RWStructuredBuffer<Int4x4> gResult;      // binding 1
RWStructuredBuffer<Int4x4> gPartialSums; // binding 2
RWStructuredBuffer<Int4x4> gTotalSum;    // binding 3 (single entry, viewed as int[16])

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void scanPerBlockKernel4x4(uint3 dispatchThreadId : SV_DispatchThreadID,
                           uint3 groupThreadId : SV_GroupThreadID,
                           uint3 groupId : SV_GroupID)
{
    const uint id = dispatchThreadId.x;
    const uint tid = groupThreadId.x;
    const uint length = gParams.length;
    const bool exclusive = gParams.exclusiveScan != 0;

    // inclusive block scan of the 16 counters (4 x uint4)
    Int4x4 v;
    const bool valid = id < length;
    v.d0 = valid ? gData[id].d0 : uint4(0);
    v.d1 = valid ? gData[id].d1 : uint4(0);
    v.d2 = valid ? gData[id].d2 : uint4(0);
    v.d3 = valid ? gData[id].d3 : uint4(0);

    Int4x4 incl;
    incl.d0 = blockInclusiveScanU4<BLOCK_SIZE>(tid, v.d0);
    incl.d1 = blockInclusiveScanU4<BLOCK_SIZE>(tid, v.d1);
    incl.d2 = blockInclusiveScanU4<BLOCK_SIZE>(tid, v.d2);
    incl.d3 = blockInclusiveScanU4<BLOCK_SIZE>(tid, v.d3);

    if (tid == BLOCK_SIZE - 1)
        gPartialSums[groupId.x] = incl;

    if (gParams.writeTotalSum != 0 && id == length - 1)
        gTotalSum[0] = incl;

    if (valid && gParams.writeResult != 0)
    {
        if (!exclusive)
        {
            gResult[id] = incl;
        }
        else
        {
            if (tid + 1 < BLOCK_SIZE && id + 1 < length)
                gResult[id + 1] = incl;
            if (tid == 0)
            {
                Int4x4 zero;
                zero.d0 = zero.d1 = zero.d2 = zero.d3 = uint4(0);
                gResult[id] = zero;
            }
        }
    }

    // mirrors: if (totalSum && gridDim.x == 1) exclusiveSumInt16(totalSum)
    // (CUDA __syncthreads orders the buffer write above within the block; the
    // Slang equivalent is an all-memory group barrier)
    if (gParams.writeTotalSum != 0 && gParams.numGroups == 1)
    {
        AllMemoryBarrierWithGroupSync();
        if (tid == 0)
            exclusiveScanTotal16();
    }
}

// exclusive scan of the 16 counters of gTotalSum[0] in place (serial by one
// thread — order-equivalent to the CUDA 16-lane warp scan)
void exclusiveScanTotal16()
{
    Int4x4 t = gTotalSum[0];
    uint vals[16] = { t.d0.x, t.d0.y, t.d0.z, t.d0.w, t.d1.x, t.d1.y, t.d1.z, t.d1.w,
                      t.d2.x, t.d2.y, t.d2.z, t.d2.w, t.d3.x, t.d3.y, t.d3.z, t.d3.w };
    uint running = 0;
    uint scanned[16];
    for (uint i = 0; i < 16; ++i)
    {
        scanned[i] = running;
        running += vals[i];
    }
    Int4x4 outT;
    outT.d0 = uint4(scanned[0], scanned[1], scanned[2], scanned[3]);
    outT.d1 = uint4(scanned[4], scanned[5], scanned[6], scanned[7]);
    outT.d2 = uint4(scanned[8], scanned[9], scanned[10], scanned[11]);
    outT.d3 = uint4(scanned[12], scanned[13], scanned[14], scanned[15]);
    gTotalSum[0] = outT;
}

// mirrors addBlockSumsKernel4x4() in algorithms.cu.  gPartialSums must hold
// the exclusively scanned per-block sums; gData is updated in place.  In the
// last group the slot totals in gTotalSum are exclusively scanned across the
// 16 slots (producing the radix "cumulativeSum").
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void addBlockSumsKernel4x4(uint3 dispatchThreadId : SV_DispatchThreadID,
                           uint3 groupThreadId : SV_GroupThreadID,
                           uint3 groupId : SV_GroupID)
{
    const uint id = dispatchThreadId.x;
    const uint len = gParams.length;

    if (gParams.writeTotalSum != 0 && id == len - 1)
    {
        Int4x4 t = gTotalSum[0];
        const Int4x4 p = gPartialSums[groupId.x];
        t.d0 += p.d0;
        t.d1 += p.d1;
        t.d2 += p.d2;
        t.d3 += p.d3;
        gTotalSum[0] = t;
    }

    if (gParams.writeResult != 0 && id < len)
    {
        Int4x4 d = gData[id];
        const Int4x4 p = gPartialSums[groupId.x];
        d.d0 += p.d0;
        d.d1 += p.d1;
        d.d2 += p.d2;
        d.d3 += p.d3;
        gData[id] = d;
    }

    // mirrors: if (totalSum && blockIdx.x == gridDim.x - 1) exclusiveSumInt16
    // (the id == len-1 accumulation above always lives in this last group)
    if (gParams.writeTotalSum != 0 && groupId.x == gParams.numGroups - 1)
    {
        AllMemoryBarrierWithGroupSync();
        if (groupThreadId.x == 0)
            exclusiveScanTotal16();
    }
}
