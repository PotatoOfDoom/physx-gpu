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

// npCvxStage2SetupDbg.slang — HARNESS-ONLY sibling of npCvxStage2Setup.slang.
// Shares npCvxStage2Core.stage2Body but takes the engine blob plus a trailing
// debug pointer (64-byte push) and dumps every scratch value the .cu computes
// (one Stage2Debug record per test) so the collide()-elided setup arithmetic
// stays bit-verified in the BDA form.  Not a registry kernel; never launched
// by the engine.  Kept in its own file because two [[vk::push_constant]]
// blocks cannot coexist in one Slang module (they collide at offset 0).

import npCvxStage2Core;

struct Stage2DebugParams
{
    uint numTests;               // @0
    float toleranceLength;       // @4
    uint64_t tempKeepPairsArray; // @8
    uint64_t cmInputs;           // @16
    uint64_t cmManifolds;        // @24
    uint64_t shapes;             // @32
    uint64_t transformCache;     // @40
    uint64_t contactDistance;    // @48
    uint64_t debugOut;           // @56
}

[[vk::push_constant]]
ConstantBuffer<Stage2DebugParams> gS2d;

[shader("compute")]
[numthreads(32, 1, 1)]
void convexConvexNphase_stage2KernelDebug(uint3 groupID : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    const uint globalWarpIndex = groupID.x * 1u + tid.y;
    if (tid.x != 0u)
        return;
    stage2Body(globalWarpIndex, gS2d.numTests, gS2d.toleranceLength, gS2d.tempKeepPairsArray,
               gS2d.cmInputs, gS2d.cmManifolds, gS2d.shapes, gS2d.transformCache,
               gS2d.contactDistance, gS2d.debugOut);
}
