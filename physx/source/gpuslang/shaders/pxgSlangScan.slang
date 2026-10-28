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

// pxgSlangScan.slang
//
// Portable workgroup scan primitives replacing the CUDA warp-shuffle based
// scans in gpucommon/src/CUDA/reduction.cuh and
// gpusimulationcontroller/src/CUDA/algorithms.cu.
//
// The CUDA originals rely on __shfl_sync/__shfl_up_sync with an assumed
// 32-wide warp.  Vulkan and D3D12 make no such guarantee (subgroup sizes of
// 8/16/32/64 all exist in the wild, and llvmpipe/WARP software devices differ
// again), so these implementations use groupshared memory and workgroup
// barriers.  Semantically they compute the same inclusive/exclusive prefix
// sums.  A wave-intrinsic specialization can be layered on top later without
// changing any call sites.

module pxgSlangScan;

// Maximum workgroup size any kernel in the port uses.
public static const uint PXG_SCAN_MAX_BLOCK = 1024;

groupshared uint gsScanTemp[PXG_SCAN_MAX_BLOCK];

// Inclusive prefix sum over the whole workgroup (Hillis-Steele).
// BLOCK_SIZE must equal the dispatch workgroup size.
// All threads of the workgroup must call this (uniform control flow).
public uint blockInclusiveScan<let BLOCK_SIZE : uint>(uint tid, uint value)
{
    gsScanTemp[tid] = value;
    GroupMemoryBarrierWithGroupSync();

    [ForceUnroll]
    for (uint offset = 1; offset < BLOCK_SIZE; offset <<= 1)
    {
        const uint n = (tid >= offset) ? gsScanTemp[tid - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsScanTemp[tid] += n;
        GroupMemoryBarrierWithGroupSync();
    }

    return gsScanTemp[tid];
}

// Inclusive prefix sum within 32-lane segments of the workgroup ("virtual
// warps" — replacement for the CUDA warpScan/warpScanAdd helpers, which are
// hard-wired to 32-thread warps regardless of the physical subgroup width).
// The scanned value may be a packed multi-counter word (e.g. 4 x 8-bit
// counters), since only additions are performed.
// All threads of the workgroup must call this (uniform control flow).
public uint warp32InclusiveScan(uint tid, uint value)
{
    const uint lane = tid & 31u;

    gsScanTemp[tid] = value;
    GroupMemoryBarrierWithGroupSync();

    [ForceUnroll]
    for (uint offset = 1; offset < 32; offset <<= 1)
    {
        const uint n = (lane >= offset) ? gsScanTemp[tid - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsScanTemp[tid] += n;
        GroupMemoryBarrierWithGroupSync();
    }

    return gsScanTemp[tid];
}

// The same inclusive 32-lane segment scan with ONE barrier instead of eleven.
//
// warp32InclusiveScan above is a Hillis-Steele scan: five rounds of read, barrier,
// add, barrier, every one a WORKGROUP barrier because Vulkan gives a 32-lane segment
// no lockstep.  In the radix-sort rank kernels (radixSortDesc, radixSort,
// thresholdStreamBda) it runs four times per 4096-key iteration of a 1024-thread
// group, i.e. 44 full-group barriers per iteration where the CUDA's __shfl_up_sync
// scan needs none -- and those kernels are the most launched and, under
// PXG_KERNEL_TIMES, the most expensive ones in every rigid, cloth, softbody and
// particle scene.  Here every thread publishes its value, one barrier, and then sums
// its own segment up to itself: 32 reads per thread, all in range, unconditional
// (no masked groupshared read for llvmpipe to speculate), and every lane of a
// segment reads the same word at the same step, which the hardware broadcasts.
// Integer addition, so the result is the same bits as warp32InclusiveScan for any
// input, packed multi-counter words included.
//
// CALLER CONTRACT (the price of the eleven barriers saved): this function returns
// while other threads may still be READING gsScanTemp, so the caller must pass a
// workgroup barrier before anything writes gsScanTemp again -- including the next
// call of any scan in this module.  warp32ScanTotal is not valid after it (use the
// returned value of lane 31 instead).  All threads must call it (uniform control).
public uint warp32InclusiveScanOneBarrier(uint tid, uint value)
{
    const uint lane = tid & 31u;
    const uint segment = tid & ~31u;

    gsScanTemp[tid] = value;
    GroupMemoryBarrierWithGroupSync();

    uint sum = 0u;
    [ForceUnroll]
    for (uint j = 0u; j < 32u; ++j)
    {
        const uint v = gsScanTemp[segment + j];
        if (j <= lane)
            sum += v;
    }
    return sum;
}

// Reads the inclusive-scan total of the 32-lane segment containing `tid`.
// Only valid immediately after warp32InclusiveScan (before gsScanTemp is
// overwritten by another scan).
public uint warp32ScanTotal(uint tid)
{
    return gsScanTemp[(tid & ~31u) | 31u];
}

groupshared uint4 gsScanTemp4[PXG_SCAN_MAX_BLOCK / 2];

// Inclusive prefix sum of uint4 values over the whole workgroup (component
// wise) — used by the int4/int4x4 scans of algorithms.cu.  BLOCK_SIZE must
// be <= PXG_SCAN_MAX_BLOCK / 2 (the shared array budget).
public uint4 blockInclusiveScanU4<let BLOCK_SIZE : uint>(uint tid, uint4 value)
{
    gsScanTemp4[tid] = value;
    GroupMemoryBarrierWithGroupSync();

    [ForceUnroll]
    for (uint offset = 1; offset < BLOCK_SIZE; offset <<= 1)
    {
        const uint4 n = (tid >= offset) ? gsScanTemp4[tid - offset] : uint4(0);
        GroupMemoryBarrierWithGroupSync();
        gsScanTemp4[tid] += n;
        GroupMemoryBarrierWithGroupSync();
    }

    return gsScanTemp4[tid];
}
