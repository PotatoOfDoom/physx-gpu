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

// npGjk.slang
//
// The standalone GJK leaf test.  The DRIVER itself now lives in pxgSlangGjk, so that this
// test and the engine-path contact-gen kernels exercise the same implementation rather than
// two copies of it -- see that module's header for why.  What remains here is the
// descriptor-set ABI this test's fixture speaks: the two vertex clouds, the per-test inputs
// and the results, loaded into GjkCloud values and handed to gjkRun.
//
// This shader used to hold the whole do-while.  Its ctest (gpuslang.ctGjk) is unchanged and
// is what pins the extraction: the same fixture, the same expected values, the same
// bit-exact comparison against the CPU reference.
import pxgSlangGjk;

struct GjkParams { uint numTests; uint maxV; uint pad0, pad1; }
struct GjkIn { float4 initialDir; uint4 cfg; float4 params; } // cfg.x=nbA cfg.y=nbB; params.x=minSep params.y=convergenceRatio

[[vk::push_constant]]
ConstantBuffer<GjkParams> gParams;

RWStructuredBuffer<float4> gVA;   // binding 0: convex A cloud (maxV/test)
RWStructuredBuffer<float4> gVB;   // binding 1: convex B cloud (maxV/test)
RWStructuredBuffer<GjkIn> gIn;    // binding 2
RWStructuredBuffer<GjkOut> gOut;  // binding 3

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjk(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint vbase = t * gParams.maxV;

    const GjkIn e = gIn[t];

    GjkCloud a, b;
    a.n = e.cfg.x;
    b.n = e.cfg.y;
    for (uint i = 0u; i < GJK_MAX_CLOUD; ++i)
    {
        a.v[i] = (i < a.n) ? gVA[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
        b.v[i] = (i < b.n) ? gVB[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
    }

    GjkOut o;
    gjkRun(a, b, e.initialDir.xyz, e.params.x, e.params.y, o);
    gOut[t] = o;
}
