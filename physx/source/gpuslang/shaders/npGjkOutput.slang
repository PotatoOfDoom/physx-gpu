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

// npGjkOutput.slang
//
// Slang port of the gjk() final-output stage (gjk.cuh :383-417) — tranche 125.
// Once the GJK loop has converged, this stage classifies the result and, when
// the objects are CLOSE, produces the witness points.  It is the pure lane-0
// scalar tail of the driver (the only warp op is reading the converged closest
// point as a vec3, which we supply directly), so it ports as a per-thread scalar
// shader on top of the tranche-118 barycentrics.
//
//   if (upperBound == 0)      return eOVERLAP;          // provably overlapping
//   if (lowerBound > minSep)  return eDISTANT;          // provably separated
//   closestPointDir = -closestPoint.getNormalized();
//   size==3: barycentrics(v0,v1,v2, p,q,r, closestPoint);
//            closestPointA = p*vA[a0] + q*vA[a1] + r*vA[a2];
//   size==2: d = v1-v0; q = -v0.dot(d)/d.magnitudeSquared();
//            closestPointA = (1-q)*vA[a0] + q*vA[a1];
//   else:    closestPointA = vA[a0];
//   closestPointB = closestPointA + upperBound*closestPointDir;
//   return eCLOSE;
//
// The cached vertices carry their A-index in the low byte of the encoded index
// word (encodeIndices: a = indices & 0xff), which selects vA[a{0,1,2}].
//
// Determinism: getNormalized is m>0 ? v*(1/sqrt(m)) : 0 with the reciprocal
// formed as 1.0f/sqrt (PxRecipSqrt -> 1.0f/::sqrtf; llvmpipe's OpFDiv and OpSqrt
// are both correctly rounded, unlike OpInverseSqrt).  With integer inputs every
// dot / cross / magnitudeSquared is exact and the divisions (1/m.[axis],
// v0.dot(d)/magSq, 1/sqrt) are correctly-rounded IEEE, so the shader and the CPU
// reference agree bit-for-bit.

// GjkResult::Enum
static const uint GJK_DISTANT = 0u;
static const uint GJK_CLOSE   = 1u;
static const uint GJK_OVERLAP = 2u;

struct GoParams { uint numTests; uint pad0, pad1, pad2; }

// per test: v0/v1/v2 xyz = cached vertex positions, w = asfloat(encoded index);
// cp.xyz = converged closest point; bounds = (upperBound, lowerBound, minSep, asfloat(size))
struct GoIn { float4 v0, v1, v2, cp, bounds; }
struct GoOut { uint ret; uint p0, p1, p2; float4 dir, cpA, cpB; }

[[vk::push_constant]]
ConstantBuffer<GoParams> gParams;

RWStructuredBuffer<GoIn> gIn;    // binding 0
RWStructuredBuffer<float4> gVA;  // binding 1: the A-convex vertex cloud (xyz used)
RWStructuredBuffer<GoOut> gOut;  // binding 2

float3 crossX(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float3 absX(float3 a) { return float3(abs(a.x), abs(a.y), abs(a.z)); }
float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxVec3::getNormalized(): m>0 ? *this * PxRecipSqrt(m) : 0, PxRecipSqrt = 1/sqrt
float3 getNormalized(float3 v)
{
    const float m = dotX(v, v);
    if (m > 0.0f)
    {
        const float r = 1.0f / sqrt(m);
        return float3(v.x * r, v.y * r, v.z * r);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkOutput(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const GoIn e = gIn[t];
    const float upperBound = e.bounds.x, lowerBound = e.bounds.y, minSep = e.bounds.z;
    const uint size = asuint(e.bounds.w);

    GoOut o;
    o.p0 = 0u; o.p1 = 0u; o.p2 = 0u;
    o.dir = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.cpA = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.cpB = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (upperBound == 0.0f) { o.ret = GJK_OVERLAP; gOut[t] = o; return; }
    if (lowerBound > minSep) { o.ret = GJK_DISTANT; gOut[t] = o; return; }

    const uint a0 = asuint(e.v0.w) & 0xffu;
    const uint a1 = asuint(e.v1.w) & 0xffu;
    const uint a2 = asuint(e.v2.w) & 0xffu;

    const float3 cp = e.cp.xyz;
    const float3 dir = -getNormalized(cp);
    o.dir = float4(dir, 0.0f);

    float3 cpA;
    if (size == 3u)
    {
        const float3 A = e.v0.xyz, B = e.v1.xyz, C = e.v2.xyz;
        const float3 m = crossX(B - A, C - A);
        const float3 ma = absX(m);
        const float3 s1 = crossX(B - cp, C - cp);
        const float3 s2 = crossX(C - cp, A - cp);
        float d, u, v;
        if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; }
        else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; }
        else                              { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; }
        const float w = 1.0f - u - v;
        cpA = u * gVA[a0].xyz + v * gVA[a1].xyz + w * gVA[a2].xyz;
    }
    else if (size == 2u)
    {
        const float3 V0 = e.v0.xyz, V1 = e.v1.xyz;
        const float3 dv = V1 - V0;
        const float q = -dotX(V0, dv) / dotX(dv, dv);
        cpA = (1.0f - q) * gVA[a0].xyz + q * gVA[a1].xyz;
    }
    else
    {
        cpA = gVA[a0].xyz;
    }

    o.cpA = float4(cpA, 0.0f);
    o.cpB = float4(cpA + upperBound * dir, 0.0f);
    o.ret = GJK_CLOSE;
    gOut[t] = o;
}
