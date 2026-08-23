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

// npSdfFind.slang -- P9.12e, the fixture entry over pxgSlangSdfFind.
//
// ONE entry point, and it drives the WHOLE resume loop rather than a single
// call.  That is the point: findInterestingVertices on its own is a block scan
// with a shared buffer, but what ps_sdfMeshCollisonLaunch actually depends on is
// the protocol BETWEEN calls -- the resume index one past the vertex that filled
// the buffer, the saturating count, and the "offset unchanged means done" test.
// A fixture that called it once would leave all three untested.
//
// The loop is the kernel's own (particleSystemMeshMidphase.cu:478), bounded here
// by maxRounds so a mutant that fails to advance startIndex is reported as a
// wrong round count instead of hanging the GPU for the sweep's timeout.

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;
import pxgSlangSdfTexture;
import pxgSlangSdfCollision;

#include "pxgSlangSdfFind.slang"

// 128 bytes exactly -- the size Vulkan guarantees every implementation offers.
// (llvmpipe reports maxPushConstantsSize = 256, but the guarantee is what this
// is sized against.)
struct SdfFindParams
{
    uint64_t meshPtr;        //   0  the cooked-mesh blob
    uint64_t positions;      //   8  float4 per particle
    uint64_t echoOut;        //  16  float4 per particle: xyz = vertex-space pt, w = dist
    uint64_t roundsOut;      //  24  4 uints per round
    uint64_t bufOut;         //  32  SDFF_TARGET_COUNT uints per round
    uint64_t numRoundsOut;   //  40  1 uint
    float4 aToBq;            //  48
    float4 scaleRot;         //  64
    float3 aToBp;            //  80
    float tolerance;         //  92
    float3 scale;            //  96
    uint numVertices;        // 108
    uint maxRounds;          // 112
    uint pad0;               // 116
    uint pad1;               // 120
    uint pad2;               // 124
}

[[vk::push_constant]]
ConstantBuffer<SdfFindParams> gP;

[shader("compute")]
[numthreads(1024, 1, 1)]
void sdfFindRounds(uint3 gtid : SV_GroupThreadID)
{
    const uint tid = gtid.x;

    PxgSdfMesh m;
    const bool hasSdf = sdfReadMesh(gP.meshPtr, m);
    const SparseSdfTexture t = sdfTexInitialize(m);

    if (!hasSdf)
    {
        if (tid == 0u)
            Ptr<uint>(gP.numRoundsOut)[0] = 0xFFFFFFFFu;
        return;
    }

    uint startIndex = 0u;
    uint round = 0u;

    // Workgroup-uniform: startIndex and round are computed identically by every
    // thread, which is what the emulated warp primitives require (their
    // barriers are workgroup barriers).
    while (startIndex < gP.numVertices && round < gP.maxRounds)
    {
        const uint before = startIndex;
        const uint count = sdfFindInterestingVerticesEx(tid, gP.numVertices, gP.positions,
                                                        gP.scale, gP.scaleRot, gP.tolerance,
                                                        t, gP.aToBp, gP.aToBq,
                                                        startIndex, gP.echoOut);

        if (tid == 0u)
        {
            Ptr<uint> r = Ptr<uint>(gP.roundsOut) + round * 4u;
            r[0] = count;
            r[1] = before;
            r[2] = startIndex;      // the resume point this round produced
            r[3] = 0xC0FFEEu;       // a canary, so a short write is visible
        }

        // One thread per slot: the buffer is exactly SDFF_TARGET_COUNT and the
        // block is exactly that wide, so this is a straight copy.
        if (tid < count)
            Ptr<uint>(gP.bufOut)[round * SDFF_TARGET_COUNT + tid] = gSdfIndices[tid];

        // Before the next round overwrites gSdfIndices.
        GroupMemoryBarrierWithGroupSync();
        ++round;
    }

    if (tid == 0u)
        Ptr<uint>(gP.numRoundsOut)[0] = round;
}
