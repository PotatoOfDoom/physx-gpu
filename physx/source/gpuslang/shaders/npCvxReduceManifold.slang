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

// npCvxReduceManifold.slang
//
// Slang port of the schlock::reduce() manifold contact reduction from
// gpunarrowphase/src/CUDA/cudaGJKEPA.cu (the active #else variant ~L415,
// instantiated reduce<false,false,4> by addManifoldPoint).  Given nbContacts
// candidate contacts (one per warp lane 0..nbContacts-1), it selects the four
// extreme points that best preserve the contact manifold:
//   i0 = deepest (min separation);
//   i1 = furthest from i0 (max squared distance);
//   f  = point maximizing d = [normal x (p_i1 - p_i0)] . (p - p_i0);
//   g  = point minimizing d; if d[f] and d[g] have the same sign (both on one
//        side) g is re-picked as the second maximizer.
// Returns the 4-bit lane mask of the chosen contacts.
//
// The CUDA runs one warp; minIndex/maxIndex are butterfly argmin/argmax over a
// lane mask returning the LOWEST lane holding the extreme value.  This port runs
// one thread per problem and serialises the warp with per-lane arrays.
//
// Callers only invoke reduce with popc(allMask) > 4 (addManifoldPoint passes 5),
// so every allMask&~mask below is non-empty; the port mirrors that contract.

static const float PXG_FLT_MAX = 3.4028235e38f;

struct ReduceIn
{
    float4 pt[32];   // per-lane contact point (.xyz); .w unused
    float4 nrm[32];  // per-lane normal (.xyz)
    float4 sep[32];  // per-lane separation in .x
    uint nbContacts; // active lanes 0..nbContacts-1
    uint pad0, pad1, pad2;
}

struct ReduceOut { uint mask; uint count; uint pad0, pad1; }

struct ReduceParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ReduceParams> gParams;

RWStructuredBuffer<ReduceIn> gIn;    // binding 0
RWStructuredBuffer<ReduceOut> gOut;  // binding 1

// lowest set bit index (0..31); matches CUDA lowestSetIndex on a non-zero mask
uint lowestSetIndex(uint mask) { return firstbitlow(mask); }

// minIndex(v, mask): lowest lane in mask whose v equals min over masked lanes.
uint minIndexArr(float v[32], uint mask)
{
    float mn = PXG_FLT_MAX;
    for (uint L = 0u; L < 32u; ++L)
        if ((mask & (1u << L)) != 0u)
            mn = min(mn, v[L]);
    // ballot(mn == v) & mask, lowest set
    uint hit = 0u;
    for (uint L = 0u; L < 32u; ++L)
        if (((mask & (1u << L)) != 0u) && (v[L] == mn))
            hit |= (1u << L);
    return lowestSetIndex(hit);
}

// maxIndex(v, mask): lowest lane in mask whose v equals max over masked lanes.
uint maxIndexArr(float v[32], uint mask)
{
    float mx = -PXG_FLT_MAX;
    for (uint L = 0u; L < 32u; ++L)
        if ((mask & (1u << L)) != 0u)
            mx = max(mx, v[L]);
    uint hit = 0u;
    for (uint L = 0u; L < 32u; ++L)
        if (((mask & (1u << L)) != 0u) && (v[L] == mx))
            hit |= (1u << L);
    return lowestSetIndex(hit);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxReduceManifold(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const ReduceIn s = gIn[t];

    float3 pt[32];
    float3 nrm[32];
    float sep[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        pt[L] = s.pt[L].xyz;
        nrm[L] = s.nrm[L].xyz;
        sep[L] = s.sep[L].x;
    }

    const uint nbContacts = s.nbContacts;
    const uint allMask = (nbContacts >= 32u) ? 0xffffffffu : ((1u << nbContacts) - 1u);

    // i0 = deepest (min separation)
    const uint i0 = minIndexArr(sep, allMask);
    uint mask = (1u << i0);

    // i1 = furthest from i0 (max squared distance)
    float d1[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float3 diff = pt[i0] - pt[L];
        d1[L] = dot(diff, diff);
    }
    const uint i1 = maxIndexArr(d1, allMask & ~mask);
    mask |= (1u << i1);

    // per-lane d = [normal x (p_i1 - p_i0)] . (p - p_i0)
    float d[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float3 dir = cross(nrm[L], pt[i1] - pt[i0]);
        d[L] = dot(dir, pt[L] - pt[i0]);
    }

    const uint f = maxIndexArr(d, allMask & ~mask);
    mask |= (1u << f);

    uint g = minIndexArr(d, allMask & ~mask);
    if (d[f] * d[g] > 0.0f)
        g = maxIndexArr(d, allMask & ~mask);
    mask |= (1u << g);

    ReduceOut r;
    r.mask = mask;
    r.count = countbits(mask);
    r.pad0 = 0u; r.pad1 = 0u;
    gOut[t] = r;
}
