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

// npGjkWarmStart.slang
//
// Slang port of the warm-start candidate selection of gjk() (gjk.cuh :282-294) —
// tranche 128.  On a warm start (cacheMask != 0) the loop looks for a cached
// simplex vertex that is closer to the origin along the current search direction
// than the current closest feature, and folds it back into the simplex:
//
//   candidates = __ballot_sync(__shfl_sync(dot(dV,simplexV) < upperBound,
//                                          threadIdx.x<<2)) & cacheMask;
//   if (candidates) {
//       warmStarted = true;
//       cacheMask &= ~(lowestSetBit(candidates)*2 - 1);      // clear up to lowest
//       supportVertexV = splatV(lowestSetIndex(candidates), simplexV);
//       if (threadIdx.x>>2 == size) simplexV = supportVertexV;
//   }
//
// dot(dV,simplexV) leaves each vector's answer in its comp-0 lane; the
// __shfl_sync(·, lane<<2) gathers vector k's predicate into lane k (lanes 0..7
// read comp-0 of vectors 0..7; higher lanes wrap mod 32), so __ballot_sync builds
// a per-vector candidate mask, then `& cacheMask` keeps only cached slots.
// lowestSetBit(x) = x & -x; lowestSetIndex(x) = ffs(x)-1 = firstbitlow(x).
//
// One thread simulates the 32-lane warp with per-lane arrays.  dV is a replicated
// vec3, simplexV holds the cached vertices; with integer payloads the warp dot,
// the `< upperBound` comparison, the ballot mask and the splatV broadcast are all
// bit-deterministic, so the shader and the CPU reference agree bit-for-bit.

struct WsParams { uint numTests; uint pad0, pad1, pad2; }
struct WsScalars { float upperBound; uint cacheMask; uint size; uint pad; }

[[vk::push_constant]]
ConstantBuffer<WsParams> gParams;

RWStructuredBuffer<float> gDV;         // binding 0: search dir dV (32/test) — in
RWStructuredBuffer<float> gZ;          // binding 1: simplexV (32/test) — in
RWStructuredBuffer<WsScalars> gS;      // binding 2: uniforms (per test)
RWStructuredBuffer<float> gZOut;       // binding 3: simplexV (32/test) — out
RWStructuredBuffer<float> gSupport;    // binding 4: supportVertexV (32/test) — out
RWStructuredBuffer<uint> gOut;         // binding 5: (candidates, cacheMaskOut, warmStarted, lowestSetIndex) x test

void dotW(float a[32], float b[32], out float o[32])
{
    float x[32];
    for (uint L = 0u; L < 32u; ++L) x[L] = a[L] * b[L];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float s1 = (L + 1u < 32u) ? x[L + 1u] : x[L];
        const float s2 = (L + 2u < 32u) ? x[L + 2u] : x[L];
        o[L] = x[L] + s1 + s2;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkWarmStart(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;

    float dV[32], zV[32];
    for (uint L = 0u; L < 32u; ++L) { dV[L] = gDV[base + L]; zV[L] = gZ[base + L]; }

    const WsScalars s = gS[t];
    const float upperBound = s.upperBound;
    uint cacheMask = s.cacheMask;
    const uint size = s.size;

    // dot(dV, simplexV) < upperBound, gathered per-vector then ballot & cacheMask
    float dd[32]; dotW(dV, zV, dd);
    bool p[32]; for (uint L = 0u; L < 32u; ++L) p[L] = dd[L] < upperBound;

    uint candidates = 0u;
    if (cacheMask != 0u)
    {
        for (uint L = 0u; L < 32u; ++L)
            if (p[(L << 2) & 31u]) candidates |= (1u << L);
        candidates &= cacheMask;
    }

    float supportVertexV[32];
    for (uint L = 0u; L < 32u; ++L) supportVertexV[L] = 0.0f;

    uint warmStarted = 0u, lowIdx = 0u;
    if (candidates != 0u)
    {
        warmStarted = 1u;
        const uint lowBit = candidates & (0u - candidates); // lowestSetBit
        cacheMask &= ~(lowBit * 2u - 1u);
        lowIdx = firstbitlow(candidates);                   // lowestSetIndex
        for (uint L = 0u; L < 32u; ++L) supportVertexV[L] = zV[((lowIdx << 2) | (L & 3u)) & 31u];
        for (uint L = 0u; L < 32u; ++L) if ((L >> 2) == size) zV[L] = supportVertexV[L];
    }

    for (uint L = 0u; L < 32u; ++L) { gZOut[base + L] = zV[L]; gSupport[base + L] = supportVertexV[L]; }
    gOut[t * 4u + 0u] = candidates;
    gOut[t * 4u + 1u] = cacheMask;
    gOut[t * 4u + 2u] = warmStarted;
    gOut[t * 4u + 3u] = lowIdx;
}
