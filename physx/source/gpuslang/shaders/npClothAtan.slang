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

// npClothAtan.slang
//
// Slang port of the FEM-cloth approximate arctangent from FEMClothUtil.cuh:
// atanApprox (L153-157) and atan2Approx (L159-173).
//   atanApprox(x) = x*(c0 + x2*(c1 + x2*(c2 + x2*(c3 + x2*(c4 + x2*c5)))))  [x2=x*x]
//   atan2Approx(y,x): swap = |x|<|y|; input = swap ? x/y : y/x;
//     output = atanApprox(input);
//     output = swap ? ((input>=0 ? HALF_PI : -HALF_PI) - output) : output;
//     if (x<0) output += (y>=0 ? PI : -PI); return output;
//
// The Horner polynomial is the sharpest FMA-contraction canary in the set, so
// every mul/add is NoContraction-stamped.  +-0 eqFZ.

static const float FEMCLOTH_PI = 3.14159265358979323846f;
static const float FEMCLOTH_HALF_PI = 1.57079632679489661923f;

float atanApprox(float x)
{
    float x2 = x * x;
    return x * (0.99997726f + x2 * (-0.33262347f + x2 * (0.19354346f + x2 * (-0.11643287f + x2 * (0.05265332f + x2 * (-0.01172120f))))));
}

float atan2Approx(float y, float x)
{
    bool swap = abs(x) < abs(y);
    float input = swap ? (x / y) : (y / x);
    float output = atanApprox(input);

    output = swap ? ((input >= 0.0f ? FEMCLOTH_HALF_PI : -FEMCLOTH_HALF_PI) - output) : output;

    if (x < 0.0f)
    {
        output += (y >= 0.0f ? FEMCLOTH_PI : -FEMCLOTH_PI);
    }
    return output;
}

struct AtIn
{
    float4 misc;   // x (atanApprox), y2, x2 (atan2Approx args), _
}

struct AtOut
{
    float4 res;    // atanApprox(x), atan2Approx(y2,x2), _, _
}

struct AtParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AtParams> gParams;

RWStructuredBuffer<AtIn> gIn;    // binding 0
RWStructuredBuffer<AtOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothAtan(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AtIn s = gIn[tid];

    const float a1 = atanApprox(s.misc.x);
    const float a2 = atan2Approx(s.misc.y, s.misc.z);

    AtOut o;
    o.res = float4(a1, a2, 0.0f, 0.0f);
    gOut[tid] = o;
}
