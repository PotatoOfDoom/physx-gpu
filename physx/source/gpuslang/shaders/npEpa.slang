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

// npEpa.slang
//
// The standalone EPA leaf test.  The polytope expansion itself now lives in pxgSlangEpa so
// that this test and the engine-path contact-gen kernels run one implementation rather than
// two -- see pxgSlangGjk's header.  What remains is the descriptor-set ABI this test's
// fixture speaks.  Its ctest (gpuslang.ctEpa) is unchanged and is what pins the extraction.
import pxgSlangGjk;
import pxgSlangEpa;

struct EpaParams { uint numTests; uint maxV; uint pad0, pad1; }
struct EpaScalars { uint size; uint nbA; uint nbB; uint pad; float convergenceRatio; float relEpsBase; float p0, p1; }

[[vk::push_constant]]
ConstantBuffer<EpaParams> gParams;

RWStructuredBuffer<float> gData;   // binding 0: GjkCachedData 32 floats/test (the overlap simplex)
RWStructuredBuffer<float4> gVA;    // binding 1: convex A cloud (maxV/test)
RWStructuredBuffer<float4> gVB;    // binding 2: convex B cloud (maxV/test)
RWStructuredBuffer<EpaScalars> gS; // binding 3: uniforms (per test)
RWStructuredBuffer<EpaOut> gOut;   // binding 4

[shader("compute")]
[numthreads(64, 1, 1)]
void ctEpa(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    const uint vbase = t * gParams.maxV;
    const EpaScalars sc = gS[t];

    GjkCloud a, b;
    a.n = sc.nbA;
    b.n = sc.nbB;
    for (uint i = 0u; i < GJK_MAX_CLOUD; ++i)
    {
        a.v[i] = (i < a.n) ? gVA[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
        b.v[i] = (i < b.n) ? gVB[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
    }

    float input[32];
    for (uint i = 0u; i < 32u; ++i) input[i] = gData[base + i];

    EpaOut o;
    epaRun(input, sc.size, sc.relEpsBase * (1.0f - sc.convergenceRatio), a, b, o);
    gOut[t] = o;
}
