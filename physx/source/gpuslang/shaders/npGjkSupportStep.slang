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

// npGjkSupportStep.slang
//
// Slang port of the cold-path front half of one gjk() iteration (gjk.cuh :272 +
// :296-310) — tranche 127.  Given the current closest point, it generates a new
// support vertex against both convex point clouds and updates the provable lower
// bound:
//
//   dV = closestPointV / magnitude(closestPointV);            // unit search dir
//   indexA = supportIndex(vA, nbA, -dV);                      // support mappings
//   indexB = supportIndex(vB, nbB,  dV);
//   supportVertexV = loadV3Unsafe(vA+indexA) - loadV3Unsafe(vB+indexB);
//   if (lane>>2 == size)                                      // add to simplex slot
//       simplexV = (lane&3)<3 ? supportVertexV : asfloat(encodeIndices(iA,iB));
//   lb = splatX(dot(dV, supportVertexV));
//   if (lb > lowerBound) {
//       output.direction = -dV;  lowerBound = lb;
//       if (lowerBound > minSep || lowerBound >= upperBound*convergenceRatio)
//           converged;                                        // (CUDA: break)
//   }
//
// closestPointV is a single vec3 replicated across all 8 vector-slots, so dV and
// supportVertexV are likewise replicated; supportIndex reads the direction from
// lanes 0,1,2.  magnitude = splatX(sqrt(dot)) and the division dV = cpv/cm are
// correctly-rounded IEEE (sqrt/OpFDiv, not OpInverseSqrt).  With integer cloud
// vertices the support dot products, the argmax tie-break, supportVertexV and the
// lb comparison are all bit-deterministic, so the shader and the CPU reference
// agree bit-for-bit.  One thread simulates the 32-lane warp with per-lane arrays.

static const float SS_MAX_F32 = 3.402823466e38f;

struct SsParams { uint numTests; uint maxV; uint pad0, pad1; }

// per test uniforms
struct SsScalars
{
    float lowerBound; float upperBound; float minSep; float convergenceRatio;
    uint nbA; uint nbB; uint size; uint pad;
}
struct SsResult
{
    uint indexA; uint indexB; uint converged; uint directionWritten;
    float lowerBoundOut; float dir0; float dir1; float dir2;
}

[[vk::push_constant]]
ConstantBuffer<SsParams> gParams;

RWStructuredBuffer<float> gCPV;        // binding 0: closestPointV (32/test) — in
RWStructuredBuffer<float> gZ;          // binding 1: simplexV (32/test) — in
RWStructuredBuffer<float4> gVA;        // binding 2: convex A cloud (maxV/test)
RWStructuredBuffer<float4> gVB;        // binding 3: convex B cloud (maxV/test)
RWStructuredBuffer<SsScalars> gS;      // binding 4: uniforms (per test)
RWStructuredBuffer<float> gZOut;       // binding 5: simplexV (32/test) — out
RWStructuredBuffer<float> gSupport;    // binding 6: supportVertexV (32/test) — out
RWStructuredBuffer<SsResult> gResult;  // binding 7: per-test result

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// warp dot over the 8-vec4 layout: answer in comp-0 lanes
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

// supportIndex over a cloud: argmax dot(dir, v[i]) with lowest-index tie-break
uint supportIdxA(float3 dir, uint vbase, uint nb)
{
    uint bestN = 0xFFFFFFFFu;
    float bestDist = -SS_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nb) ? dotX(dir, gVA[vbase + n].xyz) : -SS_MAX_F32;
        }
        float chunkMax = -SS_MAX_F32;
        for (uint lane = 0u; lane < 32u; ++lane) chunkMax = max(chunkMax, d[lane]);
        if (chunkMax > bestDist)
        {
            uint lowestLane = 0u;
            for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == chunkMax) { lowestLane = lane; break; }
            bestDist = chunkMax;
            bestN = i + lowestLane;
        }
    }
    return bestN;
}
uint supportIdxB(float3 dir, uint vbase, uint nb)
{
    uint bestN = 0xFFFFFFFFu;
    float bestDist = -SS_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nb) ? dotX(dir, gVB[vbase + n].xyz) : -SS_MAX_F32;
        }
        float chunkMax = -SS_MAX_F32;
        for (uint lane = 0u; lane < 32u; ++lane) chunkMax = max(chunkMax, d[lane]);
        if (chunkMax > bestDist)
        {
            uint lowestLane = 0u;
            for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == chunkMax) { lowestLane = lane; break; }
            bestDist = chunkMax;
            bestN = i + lowestLane;
        }
    }
    return bestN;
}

// encodeIndices(a,b) = 0x3f800000 | b<<8 | a
uint encodeIndices(uint a, uint b) { return 0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu); }

[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkSupportStep(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    const uint vbase = t * gParams.maxV;

    float cpv[32], zV[32];
    for (uint L = 0u; L < 32u; ++L) { cpv[L] = gCPV[base + L]; zV[L] = gZ[base + L]; }

    const SsScalars s = gS[t];

    // cm = magnitude(closestPointV) = splatX(sqrt(dot(cpv,cpv)))
    float dcp[32]; dotW(cpv, cpv, dcp);
    float cm[32]; for (uint L = 0u; L < 32u; ++L) cm[L] = sqrt(dcp[L & 28u]);
    // dV = closestPointV / cm
    float dV[32]; for (uint L = 0u; L < 32u; ++L) dV[L] = cpv[L] / cm[L];

    const float3 dvVec = float3(dV[0], dV[1], dV[2]);
    const uint indexA = supportIdxA(float3(-dvVec.x, -dvVec.y, -dvVec.z), vbase, s.nbA);
    const uint indexB = supportIdxB(dvVec, vbase, s.nbB);

    // supportVertexV = vA[indexA] - vB[indexB], replicated across all 8 slots
    const float3 vA = gVA[vbase + indexA].xyz;
    const float3 vB = gVB[vbase + indexB].xyz;
    const float3 sv = vA - vB;
    float supportVertexV[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint c = L & 3u;
        supportVertexV[L] = (c == 0u) ? sv.x : (c == 1u) ? sv.y : (c == 2u) ? sv.z : 0.0f;
    }

    // add to simplex slot `size`
    const float encoded = asfloat(encodeIndices(indexA, indexB));
    for (uint L = 0u; L < 32u; ++L)
    {
        if ((L >> 2) == s.size)
            zV[L] = ((L & 3u) < 3u) ? supportVertexV[L] : encoded;
    }

    // lb = splatX(dot(dV, supportVertexV))
    float dds[32]; dotW(dV, supportVertexV, dds);
    const float lb = dds[0]; // splatX broadcasts comp-0; lane 0 is the representative

    float lowerBoundOut = s.lowerBound;
    uint converged = 0u, directionWritten = 0u;
    float dir0 = 0.0f, dir1 = 0.0f, dir2 = 0.0f;
    if (lb > s.lowerBound)
    {
        dir0 = -dV[0]; dir1 = -dV[1]; dir2 = -dV[2];
        directionWritten = 1u;
        lowerBoundOut = lb;
        if (lowerBoundOut > s.minSep || lowerBoundOut >= s.upperBound * s.convergenceRatio)
            converged = 1u;
    }

    for (uint L = 0u; L < 32u; ++L) { gZOut[base + L] = zV[L]; gSupport[base + L] = supportVertexV[L]; }
    SsResult r;
    r.indexA = indexA; r.indexB = indexB; r.converged = converged; r.directionWritten = directionWritten;
    r.lowerBoundOut = lowerBoundOut; r.dir0 = dir0; r.dir1 = dir1; r.dir2 = dir2;
    gResult[t] = r;
}
