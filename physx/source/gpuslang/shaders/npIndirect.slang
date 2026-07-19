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

// npIndirect.slang — RUNTIME_PLAN.md P6 plumbing: indirect dispatch.
// writeArgs computes a workgroup count on device and stores it as a
// VkDispatchIndirectCommand (x, y, z); fillGrid is then launched with
// dispatchIndirect reading that command, so the grid is device-decided
// (no host readback).  If the indirect grid is wrong, fillGrid leaves part
// of the output unwritten and the test catches it.

struct IndParams { uint count; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<IndParams> gParams;

RWStructuredBuffer<uint> gBuf;   // binding 0: args for writeArgs, output for fillGrid

[shader("compute")]
[numthreads(64, 1, 1)]
void writeArgs(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x == 0)
    {
        gBuf[0] = (gParams.count + 63u) / 64u;   // workgroup count x
        gBuf[1] = 1u;                            // y
        gBuf[2] = 1u;                            // z
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fillGrid(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x < gParams.count)
        gBuf[dtid.x] = dtid.x + 1u;
}
