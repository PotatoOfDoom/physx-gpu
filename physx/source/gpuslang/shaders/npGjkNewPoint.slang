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

// npGjkNewPoint.slang
//
// Slang port of newPoint (epa.cuh :306-325) — tranche 117.  The EPA rare-case
// "find a new support point in either the +ve or -ve direction furthest from the
// reference point": take the Minkowski-difference support in +dir and in -dir,
// and keep whichever is further from `ref` measured along `dir`.
//
//   indexAp = supportIndex(vA, dir);   indexBp = supportIndex(vB, -dir);
//   indexAn = supportIndex(vA, -dir);  indexBn = supportIndex(vB,  dir);
//   vp = vA[indexAp] - vB[indexBp];    vn = vA[indexAn] - vB[indexBn];
//   if (|(vp-ref).dot(dir)| > |(vn-ref).dot(dir)|)   // STRICT; else -> vn
//        vertex = vp; indices = (indexBp << 8) | indexAp;
//   else vertex = vn; indices = (indexBn << 8) | indexAn;
//
// supportIndex is the tranche-116 warp argmax (lowest global index at the max of
// dot(dir, v[i])), reused here as a serialized device function over a combined
// vA|vB vertex buffer.  The four support calls are warp-collective in the CUDA
// (every lane computes the same indices); the single writing lane (threadIdx==0)
// stores the result.  Serialised: one thread computes the four indices and the
// pick.
//
// Determinism: supportIndex is exact for integer coordinates; the Minkowski
// subtraction, the (v-ref).dot(dir) projections and the abs are all exact
// integer arithmetic, and the pick is a single strict `>` comparison, so the
// chosen vertex and packed indices are bit-for-bit reproducible.
//
// Note (equivalent mutant): the abs() on the vp side is provably redundant and
// unobservable.  vp is the +dir Minkowski support and vn the -dir support, so
// vp.dot(dir) >= vn.dot(dir); writing r = ref.dot(dir), the pick |vp'|>|vn'|
// reduces to r < (maxMink+minMink)/2, and in that regime vp' = maxMink - r > 0,
// so abs(vp') == vp' whenever vp wins.  A 3,000,000-config brute force confirms
// no input diverges.  The abs is retained verbatim (faithful to the CUDA); only
// the vn-side abs is load-bearing.

static const float GS_MAX_F32 = 3.402823466e38f; // FLT_MAX

struct NpParams
{
    uint numTests;
    uint maxA; // stride (in float4s) of vA within a test block
    uint maxB; // stride of vB within a test block (block stride = maxA + maxB)
    uint pad0;
}

struct NpIn
{
    float4 dir;  // xyz = support direction
    float4 ref;  // xyz = reference point
    uint4 cfg;   // x = nbA, y = nbB, z/w unused
}

struct NpOut
{
    float4 vertex; // xyz = chosen Minkowski support point
    uint indices;  // (indexB << 8) | indexA
    uint pad0, pad1, pad2;
}

[[vk::push_constant]]
ConstantBuffer<NpParams> gParams;

RWStructuredBuffer<float4> gVerts; // binding 0: [maxA verts of A][maxB verts of B] per test
RWStructuredBuffer<NpIn> gIn;      // binding 1
RWStructuredBuffer<NpOut> gOut;    // binding 2

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// supportIndex (epa.cuh :281-302), serialized: lowest global index maximizing
// dot(dir, gVerts[base + i]) over [0, nb), per-32-chunk with strict cross-chunk.
uint supportIndex(uint base, uint nb, float3 dir)
{
    uint bestN = 0xFFFFFFFFu;
    float bestDist = -GS_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nb) ? dotX(dir, gVerts[base + n].xyz) : -GS_MAX_F32;
        }
        float chunkMax = -GS_MAX_F32;
        for (uint lane = 0u; lane < 32u; ++lane) chunkMax = max(chunkMax, d[lane]);
        if (chunkMax > bestDist)
        {
            uint lowestLane = 0u;
            for (uint lane = 0u; lane < 32u; ++lane)
                if (d[lane] == chunkMax) { lowestLane = lane; break; }
            bestDist = chunkMax;
            bestN = i + lowestLane;
        }
    }
    return bestN;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkNewPoint(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const NpIn e = gIn[t];
    const float3 dir = e.dir.xyz;
    const float3 ref = e.ref.xyz;
    const uint nbA = e.cfg.x;
    const uint nbB = e.cfg.y;
    const uint vAbase = t * (gParams.maxA + gParams.maxB);
    const uint vBbase = vAbase + gParams.maxA;

    const uint indexAp = supportIndex(vAbase, nbA, dir);
    const uint indexBp = supportIndex(vBbase, nbB, -dir);
    const uint indexAn = supportIndex(vAbase, nbA, -dir);
    const uint indexBn = supportIndex(vBbase, nbB, dir);

    const float3 vp = gVerts[vAbase + indexAp].xyz - gVerts[vBbase + indexBp].xyz;
    const float3 vn = gVerts[vAbase + indexAn].xyz - gVerts[vBbase + indexBn].xyz;

    NpOut o;
    o.pad0 = 0u; o.pad1 = 0u; o.pad2 = 0u;
    if (abs(dotX(vp - ref, dir)) > abs(dotX(vn - ref, dir)))
    {
        o.vertex = float4(vp, 0.0f);
        o.indices = (indexBp << 8) | indexAp;
    }
    else
    {
        o.vertex = float4(vn, 0.0f);
        o.indices = (indexBn << 8) | indexAn;
    }
    gOut[t] = o;
}
