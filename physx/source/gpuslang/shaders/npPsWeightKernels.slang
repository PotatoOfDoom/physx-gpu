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

// npPsWeightKernels.slang
//
// Slang port of the particle-system polynomial weight helpers from
// particlesystem.cu:
//   cube (L110)  = x*x*x
//   quart (L111) = x*x*x*x
//   oct (L112-117): x2 = x*x; x4 = x2*x2; return x4*x4     (i.e. x^8, staged)
//   WCohesion (L3397-3401): w = x*kInvRadius;
//                           return kCohesion1*cube(w) + kCohesion2*sqr(w) - 1
//                           (sqr(x) = x*x)
//
// The oct staging (x2=x*x; x4=x2*x2; x4*x4 = ((x^2)^2)^2) is preserved exactly
// because it is NOT bit-identical to a flat left-fold of eight multiplies
// (((((((x*x)*x)*x)*x)*x)*x)*x): the two round differently for most inputs
// (~68% of x in [-3,3] differ bitwise), so the staged squaring form must be
// transcribed literally to match the CUDA source.  All float mul/add/sub
// NoContraction-stamped; +-0 compared with eqFZ.

float cube(float x) { return x * x * x; }
float quart(float x) { return x * x * x * x; }
float sqr(float x) { return x * x; }
float oct(float x)
{
    float x2 = x * x;
    float x4 = x2 * x2;
    return x4 * x4;
}
float WCohesion(float x, float kCohesion1, float kCohesion2, float kInvRadius)
{
    float w = x * kInvRadius;
    return kCohesion1 * cube(w) + kCohesion2 * sqr(w) - 1.0f;
}

struct WkIn
{
    float4 a;    // x (for cube/quart/oct), _, _, _
    float4 wc;   // WCohesion x, kCohesion1, kCohesion2, kInvRadius
}

struct WkOut
{
    float4 r;    // cube(x), quart(x), oct(x), WCohesion(...)
}

struct WkParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<WkParams> gParams;

RWStructuredBuffer<WkIn> gIn;    // binding 0
RWStructuredBuffer<WkOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psWeightKernels(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const WkIn s = gIn[tid];

    const float x = s.a.x;
    const float c = cube(x);
    const float q = quart(x);
    const float o = oct(x);
    const float wc = WCohesion(s.wc.x, s.wc.y, s.wc.z, s.wc.w);

    WkOut out;
    out.r = float4(c, q, o, wc);
    gOut[tid] = out;
}
