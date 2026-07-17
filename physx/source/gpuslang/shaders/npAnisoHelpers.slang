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

// npAnisoHelpers.slang
//
// Slang port of the anisotropy small math helpers from anisotropy.cu:
// cube (L45), Wa (L46-49), Lerp (L50-54) and Clamp (L55-60).
//   cube(x)        = x*x*x
//   Wa(x, invr)    = 1 - cube(x*invr)
//   Lerp(a, b, t)  = a + (b - a) * t          (PxVec3, scalar t)
//   Clamp(a, s, t) = (PxMin(t, PxMax(s, a[i])))_i   (PxVec3 a, scalar s, t)
//
// PxMin(a,b)=(a<b)?a:b, PxMax(a,b)=(a>b)?a:b.  NoContraction-stamped; +-0 eqFZ.

float pmin(float a, float b) { return (a < b) ? a : b; }
float pmax(float a, float b) { return (a > b) ? a : b; }

float cube(float x) { return x * x * x; }
float Wa(float x, float invr) { return 1.0f - cube(x * invr); }
float3 lerp3(float3 a, float3 b, float t) { return a + (b - a) * t; }
float3 clamp3(float3 a, float s, float t)
{
    return float3(pmin(t, pmax(s, a.x)), pmin(t, pmax(s, a.y)), pmin(t, pmax(s, a.z)));
}

struct AnIn
{
    float4 misc;    // cubeX, waX, waInvr, lerpT
    float4 start;   // lerp start .xyz
    float4 end;     // lerp end .xyz
    float4 a;       // clamp a .xyz
    float4 st;      // clamp s, clamp t, _, _
}

struct AnOut
{
    float4 scal;    // cube(x), Wa, _, _
    float4 lerp;    // lerp .xyz
    float4 clamp;   // clamp .xyz
}

struct AnParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AnParams> gParams;

RWStructuredBuffer<AnIn> gIn;    // binding 0
RWStructuredBuffer<AnOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void anisoHelpers(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AnIn s = gIn[tid];

    const float cx = cube(s.misc.x);
    const float wa = Wa(s.misc.y, s.misc.z);
    const float3 lp = lerp3(s.start.xyz, s.end.xyz, s.misc.w);
    const float3 cl = clamp3(s.a.xyz, s.st.x, s.st.y);

    AnOut o;
    o.scal = float4(cx, wa, 0.0f, 0.0f);
    o.lerp = float4(lp, 0.0f);
    o.clamp = float4(cl, 0.0f);
    gOut[tid] = o;
}
