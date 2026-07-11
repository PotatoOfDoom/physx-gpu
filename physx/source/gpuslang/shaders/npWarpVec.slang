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

// npWarpVec.slang
//
// Slang port of the "8 four-vectors in a warp" layout primitives from
// nputils.cuh (:66-113) — tranche 120.  In this layout a 32-lane warp holds 8
// vec4s: lane L carries component (L & 3) of vector (L >> 2).  These pure
// shuffle/ballot register primitives are the algebra the GJK/EPA simplex core
// (closestPointTri / closestPointTet) is written on, and are unported.
//
//   dot(a,b)     = a*b + shfl_down(a*b,1) + shfl_down(a*b,2)   // vec3 dot, kept in comp-0 lane
//   cross(a,b,cp0,cp1) = shfl(a,cp0)*shfl(b,cp1) - shfl(a,cp1)*shfl(b,cp0)  // scalar, all lanes
//   splatV(index,v)    = shfl(v, (index<<2) | (lane & 3))      // broadcast vector `index`
//   splatX(a)          = shfl(a, lane & 28)                    // broadcast each vec's x to yzw
//   packXFlags<N>(a)   = ballot(shfl(a, lane<<2)) & ((1<<N)-1) // x-flags of first N vecs
//   allX<N>(a) = packXFlags<N>==(1<<N)-1;  anyX<N>(a) = packXFlags<N>!=0
//
// Serialisation: one thread simulates the whole 32-lane warp with per-lane
// arrays.  __shfl_sync(m, x, s) -> x[s & 31]; __shfl_down_sync(m, x, d) returns
// the caller's own value when s+d >= 32; __ballot_sync -> a bit loop.  normalize
// and magnitude are intentionally NOT ported (rsqrt / sqrt).
//
// Determinism: with integer / signed-permutation lane values every product, the
// three-term dot sum, the cross difference, the max/min and every equality are
// exact, so the shader and the CPU reference agree bit-for-bit.  For `dot` the
// meaningful result lives in the comp-0 lanes (L & 3 == 0); the other lanes carry
// shuffle residue that the shader and reference reproduce identically, so the
// full 32-lane output is still compared position-for-position.

struct WvParams { uint numTests; uint pad0, pad1, pad2; }

struct WvIn { uint4 s; } // x = cp0, y = cp1, z = index, w = N

[[vk::push_constant]]
ConstantBuffer<WvParams> gParams;

RWStructuredBuffer<float> gA;    // binding 0: lane value a (32/test)
RWStructuredBuffer<float> gB;    // binding 1: lane value b (32/test)
RWStructuredBuffer<WvIn> gIn;    // binding 2: per-test scalars
RWStructuredBuffer<float> gOutF; // binding 3: 32-lane float output
RWStructuredBuffer<uint> gOutU;  // binding 4: packed uint output (packXFlags/all/any)

// dot: x + shfl_down(x,1) + shfl_down(x,2), x = a*b; shfl_down returns own value
// when the source lane is out of range.
[shader("compute")]
[numthreads(64, 1, 1)]
void ctWarpDot(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    float x[32];
    for (uint L = 0u; L < 32u; ++L) x[L] = gA[base + L] * gB[base + L];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float s1 = (L + 1u < 32u) ? x[L + 1u] : x[L];
        const float s2 = (L + 2u < 32u) ? x[L + 2u] : x[L];
        gOutF[base + L] = x[L] + s1 + s2;
    }
}

// cross: a[cp0]*b[cp1] - a[cp1]*b[cp0], broadcast to all lanes.
[shader("compute")]
[numthreads(64, 1, 1)]
void ctWarpCross(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    const uint cp0 = gIn[t].s.x & 31u;
    const uint cp1 = gIn[t].s.y & 31u;
    const float val = gA[base + cp0] * gB[base + cp1] - gA[base + cp1] * gB[base + cp0];
    for (uint L = 0u; L < 32u; ++L) gOutF[base + L] = val;
}

// splatV: broadcast vector `index` across all 8 vector slots.
[shader("compute")]
[numthreads(64, 1, 1)]
void ctWarpSplatV(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    const uint index = gIn[t].s.z;
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint src = ((index << 2) | (L & 3u)) & 31u;
        gOutF[base + L] = gA[base + src];
    }
}

// splatX: broadcast each vector's x-component (comp-0 lane) into its y,z,w lanes.
[shader("compute")]
[numthreads(64, 1, 1)]
void ctWarpSplatX(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    for (uint L = 0u; L < 32u; ++L) gOutF[base + L] = gA[base + (L & 28u)];
}

// packXFlags<N> and its allX/anyX wrappers.  gA holds the per-lane boolean as
// 0.0/nonzero; only the x-flag lanes (i<<2) are read.
uint packXFlags(uint base, uint N)
{
    uint mask = 0u;
    for (uint i = 0u; i < N; ++i)
        if (gA[base + (i << 2)] != 0.0f) mask |= (1u << i);
    return mask & ((1u << N) - 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctWarpPackX(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    const uint N = gIn[t].s.w;
    const uint mask = packXFlags(base, N);
    const uint allx = (mask == ((1u << N) - 1u)) ? 1u : 0u;
    const uint anyx = (mask != 0u) ? 1u : 0u;
    gOutU[t] = mask | (allx << 8) | (anyx << 9);
}
