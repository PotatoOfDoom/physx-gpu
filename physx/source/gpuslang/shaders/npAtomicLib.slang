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

// npAtomicLib.slang — RUNTIME_PLAN.md P2.2 battery for the atomic layer of
// the warp library (atomic.cuh parity: CAS-loop float min/max/add, split
// 64-bit OR) plus the 64-bit bit conventions (__popcll/__ffsll/__clzll).
// Contended cells are verified with ORDER-INVARIANT designs: min/max are
// order-invariant by construction; the float adds use integer-valued
// contributions whose exact sums are associative, so the final cell values
// are bitwise deterministic regardless of interleaving.

#define PXG_WARP_BLOCK 64
#include "pxgSlangWarpInc.slang"

struct AlParams
{
    uint numThreads;
    uint numCells;    // contended float cells per op kind
    uint numOrCells;  // contended 64-bit OR cells (2 uints each)
    uint pad;
}

[[vk::push_constant]]
ConstantBuffer<AlParams> gParams;

RWStructuredBuffer<uint> gIn;    // binding 0: per-thread float bits
RWStructuredBuffer<uint> gCells; // binding 1: [0,C) min | [C,2C) max | [2C,3C) add | [3C,3C+2*C64) or64
RWStructuredBuffer<uint> gOut;   // binding 2: per-thread packed 64-bit helper results

[shader("compute")]
[numthreads(64, 1, 1)]
void alAtomics(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numThreads)
        return;

    const uint C = gParams.numCells;
    const float v = asfloat(gIn[t]);

    pxgAtomicMinF(gCells, t % C, v);
    pxgAtomicMaxF(gCells, C + t % C, v);
    // Integer-valued contribution: exact float addition in any order.
    pxgAtomicAddF(gCells, 2u * C + t % C, float(int(t % 97u) - 48));

    const uint orCell = t % gParams.numOrCells;
    pxgAtomicOr64(gCells, 3u * C + 2u * orCell, 1u << (t & 31u), 1u << ((t * 7u) & 31u));

    // 64-bit bit helpers on a value assembled from this and the paired
    // thread's input bits.
    const uint lo = gIn[t];
    const uint hi = gIn[t ^ 1u];
    gOut[t] = pxgPopcll(lo, hi) | (pxgFfsll(lo, hi) << 8) | (pxgClzll(lo, hi) << 16);
}

// P9.89 battery for the SELECT rule of atomic.cuh:104-132 AtomicMin/AtomicMax
// on exactly the inputs where it differs from an fminf/fmaxf combine: a
// resident +0 fed -0 (Min) / a resident -0 fed +0 (Max), NaN-seeded cells fed
// finite values or other NaN payloads, and NaN contributions to finite cells.
// Every scripted cell's CUDA outcome is order-independent, so it is compared
// bitwise.  Same three bindings as alAtomics; binding 2 is READ here as the
// per-thread selector: bit 31 = AtomicMax (else AtomicMin), bits 0..30 = cell.
[shader("compute")]
[numthreads(64, 1, 1)]
void alAtomicSelect(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numThreads)
        return;

    const uint sel = gOut[t];
    const uint cell = sel & 0x7fffffffu;
    const float v = asfloat(gIn[t]);
    if ((sel >> 31) != 0u)
        pxgAtomicMaxF(gCells, cell, v);
    else
        pxgAtomicMinF(gCells, cell, v);
}
