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

// npGjkBarycentric.slang
//
// Slang port of schlock::barycentrics (schlockShared.h :127-141) — tranche 118,
// the barycentric-coordinate helper the GJK/EPA closest-point stages use
// (gjk.cuh:402, epa.cuh:518) to express the closest point p on triangle (a,b,c).
//
//   m  = (b-a) x (c-a);   ma = |m|;                 // triangle normal (unscaled)
//   a1 = (b-p) x (c-p);   a2 = (c-p) x (a-p);       // sub-triangle area vectors
//   // ratios taken in the plane where the normal has its largest component
//   if      (ma.x >= ma.y && ma.x >= ma.z) d = 1/m.x, u = a1.x*d, v = a2.x*d;
//   else if (ma.y >= ma.z)                 d = 1/m.y, u = a1.y*d, v = a2.y*d;
//   else                                   d = 1/m.z, u = a1.z*d, v = a2.z*d;
//   w = 1 - u - v;
//
// This is a DIFFERENT formulation from barycentricCoordinates in npConvexTriOut
// (which uses Cramer dot-products); both are ported.  The axis with the largest
// |m| component is the best-conditioned projection plane.
//
// Determinism: explicit crossX matches the CPU FMA contraction; for integer /
// signed-permutation coordinates the cross products and |m| are exact, so the
// three-way component selection is deterministic, and d = 1.0f/m.[axis] is a
// correctly-rounded IEEE division (llvmpipe's OpFDiv, unlike OpInverseSqrt, is
// correctly rounded) followed by a correctly-rounded multiply — so the shader
// and the CPU reference agree bit-for-bit.  Fixtures use non-degenerate
// triangles (m != 0), so the selected component is non-zero and d is finite.

struct BcParams { uint numTests; uint pad0, pad1, pad2; }

struct BcIn { float4 a, b, c, p; } // xyz used

[[vk::push_constant]]
ConstantBuffer<BcParams> gParams;

RWStructuredBuffer<BcIn> gIn;    // binding 0
RWStructuredBuffer<float4> gOut; // binding 1: xyz = (u, v, w)

float3 crossX(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float3 absX(float3 a) { return float3(abs(a.x), abs(a.y), abs(a.z)); }

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkBarycentric(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const BcIn e = gIn[t];
    const float3 a = e.a.xyz, b = e.b.xyz, c = e.c.xyz, p = e.p.xyz;

    const float3 m = crossX(b - a, c - a);
    const float3 ma = absX(m);
    const float3 a1 = crossX(b - p, c - p);
    const float3 a2 = crossX(c - p, a - p);

    float d, u, v;
    if (ma.x >= ma.y && ma.x >= ma.z)
        { d = 1.0f / m.x; u = a1.x * d; v = a2.x * d; }
    else if (ma.y >= ma.z)
        { d = 1.0f / m.y; u = a1.y * d; v = a2.y * d; }
    else
        { d = 1.0f / m.z; u = a1.z * d; v = a2.z * d; }

    const float w = 1.0f - u - v;

    gOut[t] = float4(u, v, w, 0.0f);
}
