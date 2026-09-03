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

// npConvexReduce.slang
//
// Slang port of reducePoints from gpunarrowphase/src/CUDA/convexCoreCollision.cu:
// the 4-point contact-manifold reduction that keeps, from a set of candidate
// contacts, the deepest one (p0, min penetration d), the one farthest from p0
// (p1), and the two contacts farthest to either side of the p0->p1 axis measured
// along cross(normal, p1-p0).  The four chosen point indices are OR-ed into a
// bitmask (which may hold fewer than 4 bits when the picks coincide).
//
// The CUDA distributes one warp lane per candidate and uses the warp helpers
// minIndex/maxIndex, which return the LOWEST lane index achieving the min/max
// value over the masked lanes; a serial argmin/argmax with a strict comparison
// (first/lowest index wins on ties) is bit-identical.

struct CrParams { uint numTests; uint maxPts; uint pad0, pad1; }

[[vk::push_constant]]
ConstantBuffer<CrParams> gParams;

static const float CR_MAX_F32 = 3.402823466e38f;

RWStructuredBuffer<float4> gPts;    // binding 0  candidate (xyz = point, w = penetration d), maxPts per test
RWStructuredBuffer<float4> gNorm;   // binding 1  contact normal per test (xyz)
RWStructuredBuffer<uint> gMask;     // binding 2  active-candidate bitmask per test
RWStructuredBuffer<uint> gOut;      // binding 3  reduced-point bitmask

[shader("compute")]
[numthreads(256, 1, 1)]
void ctReducePoints(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint maxPts = gParams.maxPts;
    const uint base = t * maxPts;
    const uint mask = gMask[t];
    const float3 n = gNorm[t].xyz;

    // p0 = deepest (min penetration d), lowest index on ties
    float minDist = CR_MAX_F32;
    uint pi0 = 0u;
    for (uint i = 0u; i < maxPts; ++i)
    {
        if ((mask & (1u << i)) == 0u) continue;
        const float d = gPts[base + i].w;
        if (d < minDist) { minDist = d; pi0 = i; }
    }
    const float3 p0 = gPts[base + pi0].xyz;

    // p1 = farthest from p0
    float maxDist = -CR_MAX_F32;
    uint pi1 = 0u;
    for (uint i = 0u; i < maxPts; ++i)
    {
        if ((mask & (1u << i)) == 0u) continue;
        // P9.47: convexCoreCollision.cu:813 is `(p0 - p).magnitude()`.
        const float3 d0 = p0 - gPts[base + i].xyz;
        const float dist = sqrt(dot(d0, d0));
        if (dist > maxDist) { maxDist = dist; pi1 = i; }
    }
    const float3 p1 = gPts[base + pi1].xyz;

    // p2, p3 = the two farthest to either side of the p0->p1 axis
    const float3 sideAxis = cross(n, p1 - p0);
    float minSide = CR_MAX_F32, maxSide = -CR_MAX_F32;
    uint pi2 = 0u, pi3 = 0u;
    for (uint i = 0u; i < maxPts; ++i)
    {
        if ((mask & (1u << i)) == 0u) continue;
        const float s = dot(sideAxis, gPts[base + i].xyz - p0);
        if (s < minSide) { minSide = s; pi2 = i; }
        if (s > maxSide) { maxSide = s; pi3 = i; }
    }

    gOut[t] = (1u << pi0) | (1u << pi1) | (1u << pi2) | (1u << pi3);
}
