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

// Reads the inclusive-scan total of the 32-lane segment containing `tid`.
// Only valid immediately after warp32InclusiveScan (before gsScanTemp is
// overwritten by another scan).
public uint warp32ScanTotal(uint tid)
{
    return gsScanTemp[(tid & ~31u) | 31u];
}
