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

// npTriTriTail.slang -- P9.83b-8 T2.
//
//   Fixture entry for the mesh-mesh kernel's OUTPUT TAIL, ttbWritePatchesAndContacts
//   (pxgSlangTriTriTail.slang): writePatchesAndContacts<32> and the block-wide
//   reservation it calls, run by a whole 1024-thread workgroup per case against
//   REAL counters, a REAL contact-manager record, real flag arrays and real
//   streams -- the atomics and the stores no earlier fixture reached.
//
// ONE WORKGROUP PER CASE, and every case owns its counters, record block, flag
// words and stream regions, so the atomics are deterministic: the offsets they
// return are the counters the host preset.
//
// THE PATCHES ARE LOADED WITHOUT A BARRIER.  Threads 0..128 each copy one record
// into gTtbPatches, and the only barrier between those writes and thread 0's
// reads is the tail's own B0.  That is deliberate: B0 is the port's addition, and
// this is the one place its job is observable -- without it thread 0 reads
// records other threads have not written yet.
//
// FULLY BDA.  Nothing is bound, for the reason npTriTriWrite.slang gives.
//
// STAMPED NoContraction, like every tri-tri fixture since T1: the header's normal
// and the points are quatRotate chains compared bit for bit against a mirror built
// with -ffp-contract=off.

import pxgSlangContactReduceSerial;   // pxgSlangTriTriBlock's warp reduce needs it
import pxgSlangTriTriPatch;
import pxgSlangNpAbi;

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"
// The poisoned guard record past the CUDA's 128: a tail that read record 128
// would read poison rather than undefined storage.
#define TTB_PATCH_GUARD 1
#include "pxgSlangTriTriBlock.slang"
#include "pxgSlangTriTriTail.slang"

struct TttPush
{
    uint64_t cases;      // @0   TttCase*
    uint     numCases;   // @8
    uint     _pad12;
}

[[vk::push_constant]]
ConstantBuffer<TttPush> gTt;

// The case record, TTT_CASE_STRIDE bytes (tests/testNpTriTriTail.cpp mirrors it):
//   @0    uint4   pairIndex, numPatchesToKeep, isFlipped, -
//   @16   uint4   localMaterialIndex0, localMaterialIndex1, patchBytesLimit, contactBytesLimit
//   @32   uint4   forceBytesLimit, -, -, -
//   @48   float4  localQ1
//   @64   float4  localP1 (xyz)
//   @80   uint64  cmOutputs, materials, contactStream, patchStream, counters,
//                 touchChangeFlags, patchChangeFlags, startContactPatches,
//                 startContactPoints, startContactForces            (80 bytes)
//   @160  -       (16 bytes)
//   @176  129 patch records of 128 bytes: keepPoints[6] @0, normal/deepest @96,
//         pointCount @112 -- 128 CUDA slots and the guard.
static const uint TTT_CASE_STRIDE = 16688u;
static const uint64_t TTT_PATCHES_AT = 176UL;

[shader("compute")]
[numthreads(1024, 1, 1)]
void tttTail(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint c = gid.x;
    if (c >= gTt.numCases) return;          // block-uniform: one case per workgroup
    const uint tid = gtid.x;

    const uint64_t base = gTt.cases + uint64_t(c) * uint64_t(TTT_CASE_STRIDE);

    if (tid < TTP_MAX_MESH_MESH_PATCHES + TTB_PATCH_GUARD)
    {
        const uint64_t pb = base + TTT_PATCHES_AT + uint64_t(tid) * 128UL;
        TtpPatch p;
        for (uint k = 0u; k < TTP_MESH_MESH_CONTACT_LIMIT; ++k)
            p.keepPoints[k] = Ptr<float4>(pb + uint64_t(k) * 16UL)[0];
        const float4 n = Ptr<float4>(pb + 96UL)[0];
        p.normal = n.xyz;
        p.deepest = n.w;
        p.pointCount = Ptr<uint>(pb + 112UL)[0];
        gTtbPatches[tid] = p;
    }
    // No barrier: see the header.  The tail's B0 publishes these writes.

    const uint4 a = Ptr<uint4>(base)[0];
    const uint4 b = Ptr<uint4>(base + 16UL)[0];
    const uint4 f = Ptr<uint4>(base + 32UL)[0];
    const float4 q = Ptr<float4>(base + 48UL)[0];
    const float4 t = Ptr<float4>(base + 64UL)[0];
    Ptr<uint64_t> ptrs = Ptr<uint64_t>(base + 80UL);

    // Member by member, as the kernel will build it from its push block: a struct
    // loaded whole would hide an unread field from check_push_block.py (the critic's
    // N5), and the fixture should be written the way the kernel must be.
    TtbTailArgs args;
    args.cmOutputs = ptrs[0];
    args.materials = ptrs[1];
    args.contactStream = ptrs[2];
    args.patchStream = ptrs[3];
    args.counters = ptrs[4];
    args.touchChangeFlags = ptrs[5];
    args.patchChangeFlags = ptrs[6];
    args.startContactPatches = ptrs[7];
    args.startContactPoints = ptrs[8];
    args.startContactForces = ptrs[9];
    args.patchBytesLimit = b.z;
    args.contactBytesLimit = b.w;
    args.forceBytesLimit = f.x;

    ttbWritePatchesAndContacts(tid, a.x, a.y, q, t.xyz, a.z != 0u, b.x, b.y, args);
}
