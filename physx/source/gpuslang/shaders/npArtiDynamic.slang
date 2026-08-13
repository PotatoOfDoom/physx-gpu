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

// npArtiDynamic.slang
//
// P8.90: the table-driven harness over pxgSlangArticulationDynamic.slang.  One
// thread per case, one opcode per case, and both directions compared bit-for-bit
// against the ENGINE on the host (tests/testArtiDynamic.cpp) --
// `translateSpatialVector` is PX_CUDA_CALLABLE, as are `innerProduct`, `cross`
// and `dot`, so only the six-line structure of each function is transcribed.
//
// The output record is not written whole: each thread loads the poison image the
// fixture uploaded, writes only the slots its opcode owns, and stores it back, so
// an operation that writes a slot it does not own fails as loudly as a wrong
// number.  `adPropagateAccelerationW` must leave `qOut` poisoned; only the
// impulse direction produces one.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

static const uint AD_OP_IMPULSE = 0u;
static const uint AD_OP_ACCEL = 1u;

struct AdCase
{
    float4 vTop, vBot;      // Z (impulse) / parentLinkAccelerationW (accel)
    float4 off;             // childToParent (impulse) / parentToChild (accel)
    float4 mmTop[3];
    float4 mmBot[3];
    float4 aTop[3];         // isInvD (impulse) / IsW (accel)
    float4 aBot[3];
    float4 invS[3];         // invStISW -- accel only
    float4 qIn;             // qstZ seed (impulse) / QMinusSTZ (accel)
    uint4 cfg;              // .x = dofCount, .y = opcode
};

struct AdOut
{
    float4 rTop, rBot;
    float4 qOut;            // impulse only; the accel direction must leave it poison
};

struct AdParams { uint numCases; };

[[vk::push_constant]]
ConstantBuffer<AdParams> gAd;

RWStructuredBuffer<AdCase> gIn;   // binding 0
RWStructuredBuffer<AdOut> gOut;   // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void artiDynamicOps(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gAd.numCases)
        return;

    const AdCase c = gIn[tid];
    const uint dofCount = c.cfg.x;

    AbSpatialVector mm[3];
    AbSpatialVector a[3];
    float3 invS[3];
    for (uint k = 0u; k < 3u; ++k)
    {
        mm[k] = asmSVec(c.mmTop[k].xyz, c.mmBot[k].xyz);
        a[k] = asmSVec(c.aTop[k].xyz, c.aBot[k].xyz);
        invS[k] = c.invS[k].xyz;
    }
    const AbSpatialVector v = asmSVec(c.vTop.xyz, c.vBot.xyz);
    float q[3] = { c.qIn.x, c.qIn.y, c.qIn.z };

    AdOut o = gOut[tid];   // the poison image; untouched slots must survive

    if (c.cfg.y == AD_OP_IMPULSE)
    {
        const AbSpatialVector r = adPropagateImpulseW(a, c.off.xyz, mm, v, dofCount, q);
        o.rTop.xyz = r.top;
        o.rBot.xyz = r.bottom;
        o.qOut.xyz = float3(q[0], q[1], q[2]);
    }
    else
    {
        const AbSpatialVector r =
            adPropagateAccelerationW(c.off.xyz, invS, mm, v, dofCount, a, q);
        o.rTop.xyz = r.top;
        o.rBot.xyz = r.bottom;
    }

    gOut[tid] = o;
}
