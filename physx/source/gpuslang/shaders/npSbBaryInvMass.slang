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

// npSbBaryInvMass.slang
//
// Slang port of computeBarycentric (softBody.cu L532-575) composed with the
// p-based computeInvMass (softBody.cu L577-606): the tetrahedron barycentric
// coordinates of a point p via scalar-triple-product determinants, then the
// barycentric-weighted inverse mass with a [0,1]-clamped barycentric output.
//   ba=b-a, ca=c-a, da=d-a, pa=p-a
//   detBcd = ba . (ca x da)
//   v = (pa . (ca x da)) / detBcd
//   w = (ba . (pa x da)) / detBcd
//   x = (ba . (ca x pa)) / detBcd
//   u = 1 - v - w - x
//   invMass = a.w*u + b.w*v + c.w*w + d.w*x        // uses UNCLAMPED u,v,w,x
//   barycentric = clamp01((u,v,w,x))               // clamped AFTER invMass
//
// a/b/c/d carry position in .xyz and inverse mass in .w.  PxClamp(t,0,1) is the
// nested-ternary min(1,max(0,t)) (not the Slang clamp intrinsic).  All float
// arith NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float clamp01(float t) { const float hi = (1.0f < t) ? 1.0f : t; return (0.0f > hi) ? 0.0f : hi; }

struct BcIn
{
    float4 a, b, c, d;   // .xyz position, .w inverse mass
    float4 p;            // .xyz query point
}

struct BcOut
{
    float4 bary;         // clamped (u, v, w, x)
    float4 invMass;      // invMass in .x
}

struct BcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BcParams> gParams;

RWStructuredBuffer<BcIn> gIn;    // binding 0
RWStructuredBuffer<BcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbBaryInvMass(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const BcIn s = gIn[tid];

    const float3 a = s.a.xyz, b = s.b.xyz, c = s.c.xyz, d = s.d.xyz;
    const float3 p = s.p.xyz;

    const float3 ba = b - a;
    const float3 ca = c - a;
    const float3 da = d - a;
    const float3 pa = p - a;

    const float detBcd = dot3(ba, cross3(ca, da));
    const float detPcd = dot3(pa, cross3(ca, da));
    const float v = detPcd / detBcd;
    const float detBpd = dot3(ba, cross3(pa, da));
    const float w = detBpd / detBcd;
    const float detBcp = dot3(ba, cross3(ca, pa));
    const float x = detBcp / detBcd;

    const float u = 1.0f - v - w - x;
    const float invMass = s.a.w * u + s.b.w * v + s.c.w * w + s.d.w * x;

    BcOut o;
    o.bary = float4(clamp01(u), clamp01(v), clamp01(w), clamp01(x));
    o.invMass = float4(invMass, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
