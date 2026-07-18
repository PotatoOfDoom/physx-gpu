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

// npPsDiffuseAvg.slang
//
// Slang port of the diffuse-particle PBF neighborhood velocity averaging in
// ps_diffuseParticleUpdatePBF (diffuseParticles.cu L215-271).  The surrounding
// grid-cell traversal is out of scope; this models the per-particle reduction
// over a candidate neighbor list (fixed here at K=8, below the maxNeighbors=16
// early-out which therefore never triggers), preserving the exact sequential
// accumulation order:
//
//   velAvg = 0; weightSum = 0; velocitySum = 0; numNeighbors = 0
//   for each neighbor xj/vj in order:
//     xij = pos - xj;  dSq = xij.dot(xij)
//     if (dSq < contactDistanceSq):
//       w = WDiffuse(sqrt(dSq), invContactDistance)   // WDiffuse(h,invR) = 1 - h*invR
//       weightSum   += w
//       velocitySum += vj * w
//       ++numNeighbors
//   if (weightSum > 0): velAvg = velocitySum / weightSum
//   out = (velAvg, numNeighbors)
//
// CRITICAL: `velocitySum / weightSum` is PxVec3::operator/, a reciprocal-multiply
// (inv = 1/weightSum; velocitySum.i * inv), NOT per-component division.  The
// accumulation is a strict left-to-right fold (each += is one rounding), so the
// neighbor order is significant.  All float mul/add/sub/div NoContraction-stamped;
// +-0 compared with eqFZ.

static const uint K = 8u;

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float WDiffuse(float h, float invR) { return 1.0f - h * invR; }

struct AvIn
{
    float4 pos;      // pos.xyz, contactDistanceSq
    float4 prm;      // invContactDistance, _, _, _
    float4 xj[8];    // neighbor positions .xyz
    float4 vj[8];    // neighbor velocities .xyz
}

struct AvOut
{
    float4 velAvg;   // velAvg.xyz, float(numNeighbors)
}

struct AvParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AvParams> gParams;

RWStructuredBuffer<AvIn> gIn;    // binding 0
RWStructuredBuffer<AvOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psDiffuseAvg(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AvIn s = gIn[tid];

    const float3 pos = s.pos.xyz;
    const float contactDistanceSq = s.pos.w;
    const float invContactDistance = s.prm.x;

    float3 velAvg = float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    float3 velocitySum = float3(0.0f, 0.0f, 0.0f);
    uint numNeighbors = 0u;

    for (uint k = 0u; k < K; ++k)
    {
        const float3 xj = s.xj[k].xyz;
        const float3 xij = pos - xj;
        const float dSq = dot3(xij, xij);
        if (dSq < contactDistanceSq)
        {
            const float3 vj = s.vj[k].xyz;
            const float w = WDiffuse(sqrt(dSq), invContactDistance);
            weightSum += w;
            velocitySum += vj * w;
            ++numNeighbors;
        }
    }

    if (weightSum > 0.0f)
        velAvg = velocitySum * (1.0f / weightSum);   // velocitySum / weightSum (reciprocal-multiply)

    AvOut o;
    o.velAvg = float4(velAvg, float(numNeighbors));
    gOut[tid] = o;
}
