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

// npBoxSat.slang
//
// Slang port of the separating-axis test + feature/sign selection half of
// doBoxBoxGenerateContacts() from gpunarrowphase/src/CUDA/cudaBox.cu
// (lines 444-620).  For a box pair it evaluates the 15 OBB separating axes —
// the 3 face axes of box A (feature 0..2), the 3 face axes of box B
// (feature 3..5) and the 9 edge-edge cross axes (which can only reject) —
// each with a `contactDist`-inflated radius test, and if the boxes overlap
// selects the minimum-penetration face feature and its sign.
//
// The CUDA runs one 4-lane group per pair (lanes 0/1/2 own the three axis
// families, lane 3 idle) with __any_sync separation reductions and __shfl to
// broadcast the minimum-overlap feature.  Here the 4-lane group lives in a
// 256-thread block (64 groups): each lane computes its axis family, publishes
// (minOverlap, feature, sign, separated) to groupshared, one block-wide barrier,
// and every lane then reduces (separation OR + minimum pick).  Because
// separation is monotonic, the CUDA's early `return false`s become a single OR
// over all axes (no early return keeps the barrier uniform).

static const float PXG_MAX_F32 = 3.4028235e38f;

struct SatIn
{
    float4 b0e;                 // box0Extent (.xyz)
    float4 b1e;                 // box1Extent (.xyz)
    float4 c00, c01, c02, p0;   // transform0: rotation columns + translation
    float4 c10, c11, c12, p1;   // transform1
    float4 misc;                // (contactDist, unused...)
}

struct BoxSatParams { uint numGroups; }

[[vk::push_constant]]
ConstantBuffer<BoxSatParams> gParams;

RWStructuredBuffer<SatIn>  gIn;    // binding 0  one per group of 4
RWStructuredBuffer<float4> gOut;   // binding 1  one per group: (overlap, feature, sign, minOverlap)

groupshared float gsMin[256];
groupshared uint gsFeat[256];
groupshared float gsSign[256];
groupshared uint gsSep[256];

float comp3(float3 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : v.z); }

// ---------------------------------------------------------------------------
// boxSat: mirrors the SAT + feature selection of doBoxBoxGenerateContacts()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void boxSat(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = dtid.x;
    const uint lt = gtid.x;
    const uint lane = lt & 3u;
    const uint gbase = lt & ~3u;
    const uint groupIdx = t >> 2;
    const uint numGroups = gParams.numGroups;
    const bool valid = groupIdx < numGroups;
    const uint gi = valid ? groupIdx : (numGroups > 0u ? numGroups - 1u : 0u);

    const SatIn s = gIn[gi];
    const float3 box0e = s.b0e.xyz, box1e = s.b1e.xyz;
    const float3 c00 = s.c00.xyz, c01 = s.c01.xyz, c02 = s.c02.xyz;
    const float3 c10 = s.c10.xyz, c11 = s.c11.xyz, c12 = s.c12.xyz;
    const float3 dP = s.p1.xyz - s.p0.xyz;
    const float contactDist = s.misc.x;
    const float uEps = 1e-6f;

    // transform1To0 = transform0.transformTranspose(transform1):
    //   rot1To0.column_k = transform0.m^T * transform1.column_k
    //   tp               = transform0.m^T * (p1 - p0)
    const float3 r0 = float3(dot(c00, c10), dot(c01, c10), dot(c02, c10));
    const float3 r1 = float3(dot(c00, c11), dot(c01, c11), dot(c02, c11));
    const float3 r2 = float3(dot(c00, c12), dot(c01, c12), dot(c02, c12));
    const float3 tp = float3(dot(c00, dP), dot(c01, dP), dot(c02, dP));

    // abs0To1.column_k = |rot1To0.row_k| + eps  (rot0To1 = rot1To0^T)
    const float3 ab0 = float3(abs(r0.x), abs(r1.x), abs(r2.x)) + uEps;
    const float3 ab1 = float3(abs(r0.y), abs(r1.y), abs(r2.y)) + uEps;
    const float3 ab2 = float3(abs(r0.z), abs(r1.z), abs(r2.z)) + uEps;

    float minO = PXG_MAX_F32;
    uint feat = 0u;
    float sgn = 0.0f;
    bool sep = false;

    if (lane < 3u)
    {
        const uint tid = lane;
        const float3 col = (tid == 0u) ? r0 : ((tid == 1u) ? r1 : r2);   // rot1To0.column_tid
        const float3 abCol = abs(col) + uEps;                            // abs1To0Col
        const float3 abT = (tid == 0u) ? ab0 : ((tid == 1u) ? ab1 : ab2); // abs0To1.column_tid
        const float tpc = comp3(tp, tid);
        const float ea = comp3(box0e, tid);
        const float eb = comp3(box1e, tid);

        // --- face axis of A (feature tid) ---
        const float rbA = dot(abT, box1e);
        const float oA = (ea + rbA) - abs(tpc) + contactDist;
        const float sA = tpc;

        // --- face axis of B (feature tid+3) ---
        const float sB = dot(tp, col);
        const float raB = dot(abCol, box0e);
        const float oB = (raB + eb) - abs(sB) + contactDist;

        minO = oA; feat = tid; sgn = sA;
        if (minO > oB) { minO = oB; sgn = sB; feat = tid + 3u; }

        bool sepFace = (0.0f > oA) || (0.0f > oB);

        // --- 9 edge-edge cross axes (reject only) ---
        const uint iHi = (tid == 2u) ? 1u : 2u; // abs0To1.column_k[tid==2?1:2]
        const uint iLo = (tid == 0u) ? 1u : 0u; // abs0To1.column_k[tid==0?1:0]
        const float b1Lo = comp3(box1e, iLo);   // box1Extent[tid==0?1:0]
        const float b1Hi = comp3(box1e, iHi);   // box1Extent[tid==2?1:2]

        // ua_tid x ub0
        const float as1 = abs(col.y * tp.z - col.z * tp.y);
        const float ra1 = abCol.z * box0e.y + abCol.y * box0e.z;
        const float rb1 = comp3(ab0, iHi) * b1Lo + comp3(ab0, iLo) * b1Hi;
        const bool sep1 = as1 > (ra1 + rb1 + contactDist);

        // ua_tid x ub1
        const float as2 = abs(col.z * tp.x - col.x * tp.z);
        const float ra2 = abCol.z * box0e.x + abCol.x * box0e.z;
        const float rb2 = comp3(ab1, iHi) * b1Lo + comp3(ab1, iLo) * b1Hi;
        const bool sep2 = as2 > (ra2 + rb2 + contactDist);

        // ua_tid x ub2
        const float as3 = abs(col.x * tp.y - col.y * tp.x);
        const float ra3 = abCol.y * box0e.x + abCol.x * box0e.y;
        const float rb3 = comp3(ab2, iHi) * b1Lo + comp3(ab2, iLo) * b1Hi;
        const bool sep3 = as3 > (ra3 + rb3 + contactDist);

        sep = sepFace || sep1 || sep2 || sep3;
    }

    gsMin[lt] = minO;
    gsFeat[lt] = feat;
    gsSign[lt] = sgn;
    gsSep[lt] = sep ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    const bool separated = (gsSep[gbase] | gsSep[gbase + 1u] | gsSep[gbase + 2u]) != 0u;

    const float min0 = gsMin[gbase], min1 = gsMin[gbase + 1u], min2 = gsMin[gbase + 2u];
    uint pick;
    if (min0 <= min1 && min0 <= min2) pick = gbase;
    else if (min1 <= min2)            pick = gbase + 1u;
    else                              pick = gbase + 2u;

    if (valid && lane == 0u)
    {
        const float overlap = separated ? 0.0f : 1.0f;
        gOut[groupIdx] = float4(overlap, float(gsFeat[pick]), gsSign[pick], gsMin[pick]);
    }
}
