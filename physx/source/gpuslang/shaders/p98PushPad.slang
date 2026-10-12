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


// p98PushPad.slang -- P9.98 test-only probe (gpuslang.pipelineRefusal), never in the
// engine manifest.
//
// Its push block is 64 bytes and the fixture creates the pipeline with a 16-byte
// caller size: the shape the validation layer flagged as VUID-07987 on three engine
// entries (ps_primitivesDiffuseCollisionLaunch / ps_convexDiffuseCollisionLaunch push
// 104 of a 176-byte block, ps_updateUnsortedDiffuseArrayLaunch 16 of 24).  Unlike
// those, this entry READS the tail and writes the whole block out, so the fixture can
// see what the device pushes past the caller's bytes: since P9.98 the layout's range
// covers the block and dispatch zero-fills the surplus.
//
// pushPadDimY / pushPadDimZ (P9.98 check round 0): workgroups of (1, 2, 1) and (1, 1, 2),
// so the per-DIMENSION maxComputeWorkGroupSize check is pinned on dimensions 1 and 2 (a
// device with every dimension lowered to 1 must refuse them and build pushPadEcho's
// (1, 1, 1)).  Two invocations fit every invocation limit, so only that check refuses.

struct PushPadParams
{
    uint64_t out;   //  0  device address of uint[16]
    uint head0;     //  8
    uint head1;     // 12
    uint4 tail0;    // 16  -- past the 16 bytes the narrow caller pushes
    uint4 tail1;    // 32
    uint4 tail2;    // 48 -> 64
}

[[vk::push_constant]]
ConstantBuffer<PushPadParams> gP;

[shader("compute")]
[numthreads(1, 1, 1)]
void pushPadEcho(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<uint> o = Ptr<uint>(gP.out);
    o[0] = gP.head0;
    o[1] = gP.head1;
    o[2] = gP.tail0.x;  o[3] = gP.tail0.y;  o[4] = gP.tail0.z;  o[5] = gP.tail0.w;
    o[6] = gP.tail1.x;  o[7] = gP.tail1.y;  o[8] = gP.tail1.z;  o[9] = gP.tail1.w;
    o[10] = gP.tail2.x; o[11] = gP.tail2.y; o[12] = gP.tail2.z; o[13] = gP.tail2.w;
}

[shader("compute")]
[numthreads(1, 2, 1)]
void pushPadDimY(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<uint> o = Ptr<uint>(gP.out);
    o[dtid.y] = gP.head0 + dtid.y;
}

[shader("compute")]
[numthreads(1, 1, 2)]
void pushPadDimZ(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<uint> o = Ptr<uint>(gP.out);
    o[dtid.z] = gP.head1 + dtid.z;
}
