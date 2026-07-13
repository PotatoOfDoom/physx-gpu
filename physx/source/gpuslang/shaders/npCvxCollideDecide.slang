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

// npCvxCollideDecide.slang
//
// Slang port of the collide() post-GJK decision glue and the sphere-path
// witness store (cudaGJKEPA.cu L1354-1445).
//
// Decision glue (GJK/EPA results injected — the full gjk()/epa() drivers are
// separate verified shaders):
//   - gjkResult == eDISTANT (0)  ->  return false (no contact processing);
//   - anomaly   = gjkResult == eCLOSE && closestPointDir.dot(direction) < 0.999
//     (PxVec3::dot associates (x+y)+z);
//   - separated = gjkResult == eCLOSE;
//   - EPA runs iff (!separated || anomaly); then separated = epaResult1==eCLOSE;
//   - if EPA flagged gjkOutput.degenerate: cachedData.size = 0 and GJK+EPA
//     re-run with the same initialDir; separated = epaResult2==eCLOSE.
//   Emits: proceed (1 unless eDISTANT), ranEpa, ranRetry, separated.
//
// Sphere-path witness store (type0 == eSPHERE):
//   lanes 0..2 compute one component product sep_c = (witnessB_c - witnessA_c)
//   * direction_c; sep = shfl(sep,0)+shfl(sep,1)+shfl(sep,2) = (x+y)+z;
//   manifold slot 0 (written by lanes 0..3 across the float4s):
//     pointA = 0 (sphere centre), pointB = witnessB,
//     normalPen = (-direction, sep); numManifoldContacts = 1.

static const uint GJK_DISTANT = 0u; // GjkResult::eDISTANT
static const uint GJK_CLOSE = 1u;   // GjkResult::eCLOSE
static const uint GJK_OVERLAP = 2u; // GjkResult::eOVERLAP

struct CollideDecideIn
{
    float4 closestPointDir; // gjkOutput.closestPointDir (.xyz) after the FIRST GJK
    float4 direction;       // gjkOutput.direction (.xyz) after the FIRST GJK (anomaly gate)
    float4 directionFinal;  // gjkOutput.direction (.xyz) after the final GJK/EPA -- EPA
                            // overwrites output.direction, so the sphere store reads a
                            // potentially different vector than the anomaly gate
    float4 witnessA;        // gjkOutput.closestPointA (.xyz), post final GJK/EPA
    float4 witnessB;        // gjkOutput.closestPointB (.xyz)
    uint gjkResult;         // first GJK result
    uint epaResult1;        // EPA result if it runs
    uint degenerate;        // gjkOutput.degenerate after first EPA
    uint epaResult2;        // EPA result of the degenerate re-run
}

struct CollideDecideOut
{
    float4 mA0;   // sphere-path manifold slot 0 A (.xyz = 0, .w = 0)
    float4 mB0;   // sphere-path manifold slot 0 B
    float4 mNp0;  // sphere-path normalPen (-direction, sep)
    uint proceed;     // 0 iff gjkResult == eDISTANT
    uint ranEpa;      // 1 iff EPA gate fired
    uint ranRetry;    // 1 iff degenerate re-run fired
    uint separated;   // final separated flag
    uint numManifold; // sphere path: 1 (0 when !proceed)
    uint pad0, pad1, pad2;
}

struct CollideDecideParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CollideDecideParams> gParams;

RWStructuredBuffer<CollideDecideIn> gIn;    // binding 0
RWStructuredBuffer<CollideDecideOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxCollideDecide(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CollideDecideIn s = gIn[t];

    CollideDecideOut r;
    r.mA0 = float4(0, 0, 0, 0);
    r.mB0 = float4(0, 0, 0, 0);
    r.mNp0 = float4(0, 0, 0, 0);
    r.proceed = 0u; r.ranEpa = 0u; r.ranRetry = 0u; r.separated = 0u; r.numManifold = 0u;
    r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;

    if (s.gjkResult == GJK_DISTANT)
    {
        gOut[t] = r; // return false: nothing else runs
        return;
    }
    r.proceed = 1u;

    // anomaly test: PxVec3::dot associates (x*vx + y*vy) + z*vz
    const float3 cpd = s.closestPointDir.xyz;
    const float3 dir = s.direction.xyz;
    const float dp = (cpd.x * dir.x + cpd.y * dir.y) + cpd.z * dir.z;
    const bool anomaly = (s.gjkResult == GJK_CLOSE) && (dp < 0.999f);
    bool separated = (s.gjkResult == GJK_CLOSE);

    if (!separated || anomaly)
    {
        r.ranEpa = 1u;
        separated = (s.epaResult1 == GJK_CLOSE);
        if (s.degenerate != 0u)
        {
            r.ranRetry = 1u; // cachedData.size = 0; GJK + EPA re-run
            separated = (s.epaResult2 == GJK_CLOSE);
        }
    }
    r.separated = separated ? 1u : 0u;

    // sphere-path witness store: sep = ((x + y) + z) of per-component products.
    // Uses the FINAL direction (post-EPA when EPA ran), not the anomaly-gate one.
    const float3 dirF = s.directionFinal.xyz;
    const float3 wA = s.witnessA.xyz;
    const float3 wB = s.witnessB.xyz;
    const float sepX = (wB.x - wA.x) * dirF.x;
    const float sepY = (wB.y - wA.y) * dirF.y;
    const float sepZ = (wB.z - wA.z) * dirF.z;
    const float sep = (sepX + sepY) + sepZ;

    r.mA0 = float4(0, 0, 0, 0); // sphere centre
    r.mB0 = float4(wB, 0.0f);
    r.mNp0 = float4(-dirF, sep);
    r.numManifold = 1u;

    gOut[t] = r;
}
