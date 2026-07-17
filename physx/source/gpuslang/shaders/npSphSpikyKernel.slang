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

// npSphSpikyKernel.slang
//
// Slang port of the SPH spiky-kernel scalar helpers sqr/W/dWdx
// (particleSystem.cuh L58-68):
//   sqr(x)  = x * x
//   W(x)    = kSpiky1 * sqr(1 - x*kInvRadius)
//   dWdx(x) = -kSpiky2 * (1 - x*kInvRadius)
//
// Pure scalar float arithmetic.  NoContraction-stamped; +-0 eqFZ.

float sqr(float x) { return x * x; }
float W(float x, float kSpiky1, float kInvRadius) { return kSpiky1 * sqr(1.0f - x * kInvRadius); }
float dWdx(float x, float kSpiky2, float kInvRadius) { return -kSpiky2 * (1.0f - x * kInvRadius); }

struct SkIn
{
    float4 in;   // x, kSpiky1, kSpiky2, kInvRadius
}

struct SkOut
{
    float4 out;  // sqr(x), W(x), dWdx(x), _
}

struct SkParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SkParams> gParams;

RWStructuredBuffer<SkIn> gIn;    // binding 0
RWStructuredBuffer<SkOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sphSpikyKernel(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SkIn s = gIn[tid];

    const float x = s.in.x;
    const float kSpiky1 = s.in.y, kSpiky2 = s.in.z, kInvRadius = s.in.w;

    SkOut o;
    o.out = float4(sqr(x), W(x, kSpiky1, kInvRadius), dWdx(x, kSpiky2, kInvRadius), 0.0f);
    gOut[tid] = o;
}
