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

// npGjkPermute.slang
//
// Slang port of the per-iteration loop-glue of gjk() (gjk.cuh :336-363) —
// tranche 126.  After a simplex closest-point core returns a feature code f and
// a candidate closest point, this glue closes the iteration:
//
//   newSize   = f & 0xf;                                        // low nibble
//   degenerate= f == DEGENERATE || (cm >= upperBound && newSize <= size);
//   if (degenerate) { ... break / restart ... }                // no state update
//   backupSize = cm >= upperBound ? size : 0;                  // keep-best guard
//   if (backupSize) { backup the simplex }  else upperBound = cm;
//   size = newSize;
//   // feature-code-driven simplex permute: retained vector k (k=0,1,2) is
//   // gathered from source vector nibble (k+1) of f.
//   sourceIndex = f >> ((threadIdx.x & 28) + 4) & 0xf;
//   tmp = __shfl(simplexV, sourceIndex<<2 | (threadIdx.x&3));
//   if (threadIdx.x < 12) simplexV = tmp;
//
// The feature code packs f = (i2<<12 | i1<<8 | i0<<4 | size); vectors 0,1,2 pull
// their new contents from source vectors i0,i1,i2 respectively (lanes 12-31 are
// left untouched).  This is data movement plus integer/uniform comparisons, so
// with integer simplex payloads the shader and the CPU reference agree bit-for-
// bit.  cm, upperBound, size and f are warp-uniform; one thread simulates the
// 32-lane warp with per-lane arrays (lane L = component L&3 of vector L>>2).

static const uint DEGENERATE = 0xffffu;

struct PmParams { uint numTests; uint pad0, pad1, pad2; }

// per test: cm, upperBound (uniform scalars); size, f (uniform)
struct PmScalars { float cm; float upperBound; uint size; uint f; }

[[vk::push_constant]]
ConstantBuffer<PmParams> gParams;

RWStructuredBuffer<float> gZ;       // binding 0: simplex zV (32/test) — in
RWStructuredBuffer<PmScalars> gS;   // binding 1: uniform scalars (per test)
RWStructuredBuffer<float> gZOut;    // binding 2: permuted simplex (32/test) — out
RWStructuredBuffer<uint> gOut;      // binding 3: (newSize, degenerate, backupSize, asuint(upperBoundOut)) x test

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkPermute(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;

    float zV[32];
    for (uint L = 0u; L < 32u; ++L) zV[L] = gZ[base + L];

    const PmScalars s = gS[t];
    const float cm = s.cm, upperBound = s.upperBound;
    const uint size = s.size, f = s.f;

    const uint newSize = f & 0xfu;
    const bool degenerate = (f == DEGENERATE) || (cm >= upperBound && newSize <= size);

    if (degenerate)
    {
        // no state update: simplex, upperBound left unchanged (caller breaks/restarts)
        for (uint L = 0u; L < 32u; ++L) gZOut[base + L] = zV[L];
        gOut[t * 4u + 0u] = newSize;
        gOut[t * 4u + 1u] = 1u;
        gOut[t * 4u + 2u] = 0u;
        gOut[t * 4u + 3u] = asuint(upperBound);
        return;
    }

    const uint backupSize = (cm >= upperBound) ? size : 0u;
    const float upperBoundOut = (backupSize != 0u) ? upperBound : cm;

    // feature-code-driven simplex permute (retained vectors 0,1,2 <- source nibbles)
    float tmp[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint sourceIndex = (f >> ((L & 28u) + 4u)) & 0xfu;
        tmp[L] = zV[((sourceIndex << 2) | (L & 3u)) & 31u];
    }
    for (uint L = 0u; L < 32u; ++L) zV[L] = (L < 12u) ? tmp[L] : zV[L];

    for (uint L = 0u; L < 32u; ++L) gZOut[base + L] = zV[L];
    gOut[t * 4u + 0u] = newSize;
    gOut[t * 4u + 1u] = 0u;
    gOut[t * 4u + 2u] = backupSize;
    gOut[t * 4u + 3u] = asuint(upperBoundOut);
}
