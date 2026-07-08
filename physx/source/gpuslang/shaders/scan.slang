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

// scan.slang
//
// Slang port of the device-wide prefix-sum kernels from
// gpusimulationcontroller/src/CUDA/algorithms.cu:
//   - scanPerBlockKernel   (PxU32 variant)
//   - addBlockSumsKernel   (PxU32 variant)
//
// Device-wide scans compose exactly like the CUDA pipeline:
//   1. scanPerBlock over the input        -> per-block results + partialSums
//   2. scanPerBlock over the partialSums  -> exclusively scanned partialSums
//   3. addBlockSums                       -> adds scanned partialSums to (1)
//
// Differences from CUDA:
//   - The kernel signature scalars (length, exclusiveScan) arrive as push
//     constants instead of kernel arguments.
//   - The workgroup size is 1024, identical to the CUDA __launch_bounds__.
//   - Warp shuffles are replaced by the portable groupshared scan in
//     pxgSlangScan.slang.

import pxgSlangScan;

static const uint BLOCK_SIZE = 1024;

struct ScanParams
{
    uint length;
    uint exclusiveScan;
    uint writeResult;   // result buffer valid (CUDA passed NULL to skip)
    uint writeTotalSum; // totalSum buffer valid
};

[[vk::push_constant]]
ConstantBuffer<ScanParams> gParams;

RWStructuredBuffer<uint> gData;        // binding 0: input values
RWStructuredBuffer<uint> gResult;      // binding 1: scanned output
RWStructuredBuffer<uint> gPartialSums; // binding 2: one entry per block
RWStructuredBuffer<uint> gTotalSum;    // binding 3: single entry

// Mirrors scanPerBlockKernel/scanPerBlockKernelShared<PxU32>.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void scanPerBlockKernel(uint3 dispatchThreadId : SV_DispatchThreadID,
                        uint3 groupThreadId : SV_GroupThreadID,
                        uint3 groupId : SV_GroupID)
{
    const uint id = dispatchThreadId.x;
    const uint tid = groupThreadId.x;
    const uint length = gParams.length;

    uint value = id < length ? gData[id] : 0u;

    // scanPerBlock(): inclusive scan across the workgroup, block total to
    // partialSums[blockIdx.x].
    value = blockInclusiveScan<BLOCK_SIZE>(tid, value);

    if (tid == BLOCK_SIZE - 1)
        gPartialSums[groupId.x] = value;

    if (gParams.writeTotalSum != 0 && id == length - 1)
        gTotalSum[0] = value;

    if (id < length && gParams.writeResult != 0)
    {
        if (gParams.exclusiveScan == 0)
        {
            gResult[id] = value;
        }
        else
        {
            if (tid + 1 < BLOCK_SIZE && id + 1 < length)
                gResult[id + 1] = value;
            if (tid == 0)
                gResult[id] = 0u;
        }
    }
}

// Mirrors addBlockSumsKernel/addBlockSumsKernelShared<PxU32>.
// gPartialSums must contain the *exclusively scanned* per-block sums
// (produced by running scanPerBlockKernel over the raw partial sums with
// exclusiveScan = 1), and gData is updated in place — identical contract to
// the CUDA kernel.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void addBlockSumsKernel(uint3 dispatchThreadId : SV_DispatchThreadID,
                        uint3 groupId : SV_GroupID)
{
    const uint id = dispatchThreadId.x;

    if (id >= gParams.length)
        return;

    if (gParams.writeTotalSum != 0 && id == gParams.length - 1)
        gTotalSum[0] = gTotalSum[0] + gPartialSums[groupId.x];

    if (gParams.writeResult != 0)
        gData[id] = gData[id] + gPartialSums[groupId.x];
}
