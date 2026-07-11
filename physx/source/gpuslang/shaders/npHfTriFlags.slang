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

// npHfTriFlags.slang
//
// Slang port of the edge-classification core of computeTriangleFlags
// (convexHeightfield.cu :565-614) — tranche 119.  For a heightfield triangle it
// marks each of the 3 edges as an active convex edge, an active boundary edge,
// or a silhouette edge, by comparing the edge's adjacent triangle's normal to
// the current triangle's normal:
//
//   nextInd = {2, 0, 1}
//   for a in 0..2, if the adjacent triangle exists:
//     adjNormal = (av1-av0) x (av2-av0);                  // denormalizedNormal
//     projD = adjNormal.dot(cur[nextInd[a]] - av0);
//     if (projD < 0) {                                    // convex fold
//         adjNormal.normalize();
//         if (adjNormal.dot(triNormal) < 0.997) triFlags |= 1 << (a+3);   // non-coplanar
//     } else if (boundaryCollisions) triFlags |= 1 << (a+3);              // boundary active
//     else                           triFlags |= 1 << a;                  // silhouette
//
// boundaryCollisions = !(flags & PxHeightFieldFlag::eNO_BOUNDARY_EDGES (1<<0)).
//
// The getTriangle heightfield-sample decode that the CUDA calls to obtain each
// adjacent triangle is a separately-verified concern (tranches 58/60,
// npHeightfield), so here the 3 adjacent triangles and the adjacency-valid mask
// are supplied pre-decoded — this isolates the NEW classification logic.
//
// Determinism: for integer triangle coordinates projD is an exact integer dot,
// so the `projD < 0` branch is deterministic; the normalize uses the established
// v*(1.0f/sqrt(dot)) discipline (PxVec3::normalize is `*this /= magnitude()`, i.e.
// reciprocal-then-multiply), and explicit dotX controls the FMA contraction, so
// the shader and the CPU reference compute proj bit-for-bit identically and the
// `proj < 0.997f` compare is reproducible.
//
// Note (equivalent guard): the `m2 > 0` test before the normalize (mirroring
// PxVec3::normalize's `if (m > 0)`) is unreachable-when-false here — it sits
// inside the `projD < 0` branch, and projD = dot(adjNormal, other-av0) < 0 forces
// adjNormal != 0, hence m2 = dot(adjNormal, adjNormal) > 0.  It is retained
// verbatim (faithful to the CUDA), but no fixture can exercise its false arm.

struct HfParams { uint numTests; uint pad0, pad1, pad2; }

struct HfIn
{
    float4 cur[3];      // current triangle verts (xyz)
    float4 triNormal;   // current triangle normal (xyz, already unit)
    float4 adj[9];      // adjacent triangles: adj[a*3 + vi]
    uint4 cfg;          // x = adjValidMask (bits 0..2), y = boundaryCollisions (0/1)
}

[[vk::push_constant]]
ConstantBuffer<HfParams> gParams;

RWStructuredBuffer<HfIn> gIn;  // binding 0
RWStructuredBuffer<uint> gOut; // binding 1: triFlags

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossX(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctHfTriFlags(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const HfIn e = gIn[t];
    const uint adjValidMask = e.cfg.x;
    const bool boundaryCollisions = e.cfg.y != 0u;
    const float3 triNormal = e.triNormal.xyz;
    const float3 cur0 = e.cur[0].xyz, cur1 = e.cur[1].xyz, cur2 = e.cur[2].xyz;

    const uint nextInd[3] = { 2u, 0u, 1u };
    uint triFlags = 0u;

    for (uint a = 0u; a < 3u; ++a)
    {
        if (((adjValidMask >> a) & 1u) == 0u) continue;

        const float3 av0 = e.adj[a * 3u + 0u].xyz;
        const float3 av1 = e.adj[a * 3u + 1u].xyz;
        const float3 av2 = e.adj[a * 3u + 2u].xyz;

        float3 adjNormal = crossX(av1 - av0, av2 - av0);
        const uint otherIndex = nextInd[a];
        const float3 other = (otherIndex == 0u) ? cur0 : ((otherIndex == 1u) ? cur1 : cur2);
        const float projD = dotX(adjNormal, other - av0);

        if (projD < 0.0f)
        {
            // adjNormal.normalize(): m = magnitude(); if (m>0) *this *= 1/m
            const float m2 = dotX(adjNormal, adjNormal);
            if (m2 > 0.0f) adjNormal = adjNormal * (1.0f / sqrt(m2));
            const float proj = dotX(adjNormal, triNormal);
            if (proj < 0.997f)
                triFlags |= (1u << (a + 3u));
        }
        else if (boundaryCollisions)
        {
            triFlags |= (1u << (a + 3u));
        }
        else
        {
            triFlags |= (1u << a);
        }
    }

    gOut[t] = triFlags;
}
