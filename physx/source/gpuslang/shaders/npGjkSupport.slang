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

// npGjkSupport.slang
//
// Slang port of supportIndex (epa.cuh :281-302) — tranche 116, the first piece
// of the GJK/EPA module.  The support mapping: over a convex vertex list, return
// the index of the vertex that maximises dot(dir, v[i]), with ties broken to the
// LOWEST global index.
//
// CUDA structure (one 32-lane warp): walk the vertex list in 32-vertex chunks;
// each lane holds d = (n < nbVertices) ? dir.dot(v[n]) : -FLT_MAX.  A five-step
// __shfl_xor butterfly of fmaxf broadcasts the chunk maximum e to every lane;
// when e > bestDist (STRICT) the chunk wins and bestN = i +
// __ffs(__ballot(e == d)) - 1 (the lowest lane whose d equals the chunk max).
// The strict cross-chunk compare means that when two chunks tie on the maximum,
// the EARLIER chunk keeps it — so the overall winner is the lowest global index
// achieving the global maximum.
//
// Serialisation: one thread simulates the whole warp.  The butterfly fmaxf
// reduction is order-independent for non-NaN inputs, so a plain max over the 32
// lane values reproduces e exactly; __ffs(__ballot(e == d)) - 1 becomes the first
// lane with d == e; the cross-chunk strict `>` is kept verbatim.  With the
// integer/exact-arithmetic fixtures the dot products are exact, so every max and
// equality — hence the tie-break lane — is bit-for-bit reproducible.

static const float GS_MAX_F32 = 3.402823466e38f; // FLT_MAX

struct GsParams
{
    uint numTests;
    uint maxV; // stride (in float4s) of each test's vertex block
    uint pad0, pad1;
}

struct GsIn
{
    float4 dir;  // xyz = support direction (w unused)
    uint4 cfg;   // x = nbVertices, y/z/w unused
}

[[vk::push_constant]]
ConstantBuffer<GsParams> gParams;

RWStructuredBuffer<float4> gVerts; // binding 0: vertices xyz (maxV per test)
RWStructuredBuffer<GsIn> gIn;      // binding 1
RWStructuredBuffer<uint> gOut;     // binding 2: winning index

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkSupportIndex(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const GsIn e = gIn[t];
    const float3 dir = e.dir.xyz;
    const uint nbVertices = e.cfg.x;
    const uint vbase = t * gParams.maxV;

    uint bestN = 0xFFFFFFFFu;
    float bestDist = -GS_MAX_F32;

    for (uint i = 0u; i < nbVertices; i += 32u)
    {
        // per-lane projection (lanes past the tail contribute -FLT_MAX)
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nbVertices) ? dotX(dir, gVerts[vbase + n].xyz) : -GS_MAX_F32;
        }

        // chunk maximum (butterfly fmaxf == plain max for non-NaN)
        float chunkMax = -GS_MAX_F32;
        for (uint lane = 0u; lane < 32u; ++lane) chunkMax = max(chunkMax, d[lane]);

        if (chunkMax > bestDist)
        {
            // lowest lane whose d equals the chunk max (__ffs(__ballot(e==d))-1)
            uint lowestLane = 0u;
            for (uint lane = 0u; lane < 32u; ++lane)
                if (d[lane] == chunkMax) { lowestLane = lane; break; }
            bestDist = chunkMax;
            bestN = i + lowestLane;
        }
    }

    gOut[t] = bestN;
}
