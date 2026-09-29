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

// npBoxIncident.slang
//
// Slang port of the two warp-cooperative (width-4) box-box setup primitives from
// gpunarrowphase/src/CUDA/cudaBox.cu.  Each works over a group of 4 lanes (one
// per corner of a box face); the original uses __shfl_sync / __shfl_xor_sync
// within that 4-lane group, reproduced here with a 32-bit groupshared exchange
// (256-thread block = 64 independent groups of 4).
//
//   - getIncidentPolygon4: picks box B's most anti-aligned ("incident") face
//     w.r.t. a separating axis (broadcast __shfl_sync of the |column.dot(axis)|
//     from lanes 0/1/2 of the group), then each lane emits one of the 4
//     face-corner vertices transformed into box A's local space, plus the
//     (group-uniform) incident-face normal.
//   - calculateBounds4: a width-4 __shfl_xor_sync butterfly reduction computing
//     the AABB of the 4 per-lane points; every lane ends holding the group AABB
//     (bit for bit only up to the selectMin/selectMax tie and NaN rule, P9.89).
//
// These feed the box-box clipping core (calculateContacts / doBoxBoxGenerate-
// Contacts), a later tranche.

static const uint OUTPUT_UNUSED = 0u;

struct IncidentIn
{
    float4 col0;  // PxMat34.m.column0
    float4 col1;  // PxMat34.m.column1
    float4 col2;  // PxMat34.m.column2
    float4 col3;  // PxMat34.p  (translation)
    float4 axis;  // separating axis (.xyz)
    float4 ext;   // box B extents (.xyz)
}

struct BoxIncParams { uint numGroups; }

[[vk::push_constant]]
ConstantBuffer<BoxIncParams> gParams;

RWStructuredBuffer<IncidentIn> gIncIn;      // binding 0  one per group of 4
RWStructuredBuffer<float4>     gIncVerts;    // binding 1  one per lane (4 per group)
RWStructuredBuffer<float4>     gIncNormal;   // binding 2  one per group
RWStructuredBuffer<float4>     gPts;         // binding 3  bounds input (one per lane)
RWStructuredBuffer<float4>     gBoundsMin;   // binding 4  one per lane (group min, per-lane +-0/NaN winner)
RWStructuredBuffer<float4>     gBoundsMax;   // binding 5  one per lane (group max, per-lane +-0/NaN winner)

groupshared float gsD[256];
groupshared float gsAbsD[256];
groupshared float3 gsExch[256];

// ---------------------------------------------------------------------------
// getIncidentPolygon4: mirrors cudaBox.cu getIncidentPolygon4()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void boxIncidentPolygon4(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = dtid.x;
    const uint lt = gtid.x;
    const uint lane = lt & 3u;                 // threadIndexInGroup
    const uint gbase = lt & ~3u;               // groupStartThreadIndex (local)
    const uint groupIdx = t >> 2;
    const uint numGroups = gParams.numGroups;

    // Broadcast the group's input to all 4 lanes (clamp the index so tail lanes
    // never read out of bounds; their outputs are masked off below).
    const uint gi = (groupIdx < numGroups) ? groupIdx : (numGroups > 0u ? numGroups - 1u : 0u);
    const IncidentIn in = gIncIn[gi];
    const float3 axis = in.axis.xyz;
    const float3 col0 = in.col0.xyz, col1 = in.col1.xyz, col2 = in.col2.xyz, col3 = in.col3.xyz;
    const float3 ext = in.ext.xyz;

    // d = column[lane].dot(axis) for lanes 0/1/2 only.
    float d = 0.0f;
    if (lane < 3u)
    {
        const float3 col = (lane == 0u) ? col0 : ((lane == 1u) ? col1 : col2);
        d = dot(col, axis);
    }
    gsD[lt] = d;
    gsAbsD[lt] = abs(d);
    GroupMemoryBarrierWithGroupSync();

    const float absd0 = gsAbsD[gbase + 0u];
    const float absd1 = gsAbsD[gbase + 1u];
    const float absd2 = gsAbsD[gbase + 2u];
    const float d0 = gsD[gbase + 0u];
    const float d1 = gsD[gbase + 1u];
    const float d2 = gsD[gbase + 2u];

    const float s0 = (lane == 0u || lane == 3u) ? -1.0f : 1.0f;
    const float s1 = (lane & 2u) != 0u ? 1.0f : -1.0f;

    float3 sgn;
    float3 faceNormal;
    if (absd0 >= absd1 && absd0 >= absd2)
    {
        sgn.x = (d0 > 0.0f) ? -1.0f : 1.0f;
        sgn.y = s0;
        sgn.z = s1;
        faceNormal = col0 * sgn.x;
    }
    else if (absd1 >= absd2)
    {
        sgn.y = (d1 > 0.0f) ? -1.0f : 1.0f;
        sgn.x = s0;
        sgn.z = s1;
        faceNormal = col1 * sgn.y;
    }
    else
    {
        sgn.z = (d2 > 0.0f) ? -1.0f : 1.0f;
        sgn.x = s0;
        sgn.y = s1;
        faceNormal = col2 * sgn.z;
    }

    const float3 vert = ext * sgn;                         // extents.multiply(sign)
    const float3 outV = col0 * vert.x + col1 * vert.y + col2 * vert.z + col3; // transf1To0.transform

    if (groupIdx < numGroups)
    {
        gIncVerts[t] = float4(outV, 0.0f);
        if (lane == 0u)
            gIncNormal[groupIdx] = float4(faceNormal, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// calculateBounds4: mirrors cudaBox.cu calculateBounds4()
// width-4 __shfl_xor_sync butterfly:  xor 1 (include), then xor 2 (min / max).
//
// P9.89: every combine of the butterfly (cudaBox.cu L199-220: bounds.include(val)
// at L208, bounds.minimum.minimum(...) at L210, bounds.maximum.maximum(...) at
// L214) is PxVec3::minimum/maximum (PxBounds3.h L333-338, PxVec3.h L339/L355) =
// componentwise PxMin/PxMax(float) = intrinsics::selectMin/selectMax (PxMath.h
// L82-101, PxUnixMathIntrinsics.h L116-125): own < partner ? own : partner /
// own > partner ? own : partner, the lane's OWN value first.  On a +-0 tie the
// PARTNER survives, so the lanes of one group need not end with the same zero
// sign; a NaN partner propagates and a NaN own value is replaced.  Slang
// min()/max() (GLSL.std.450 FMin/FMax) run zero-ordered (-0 / +0 in either
// order) and NaN-ignoring on the SZINP-stamped llvmpipe build (P9.86 F4), which
// is not the CUDA's combine.  Transcribed per component as ternaries.
// ---------------------------------------------------------------------------
float bxSelMin(float own, float partner) { return (own < partner) ? own : partner; }  // selectMin
float bxSelMax(float own, float partner) { return (own > partner) ? own : partner; }  // selectMax

[shader("compute")]
[numthreads(256, 1, 1)]
void boxCalculateBounds4(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = dtid.x;
    const uint lt = gtid.x;
    const uint numLanes = gParams.numGroups * 4u;

    const float3 pt = (t < numLanes) ? gPts[t].xyz : float3(0.0f, 0.0f, 0.0f);
    float3 bmin = pt;
    float3 bmax = pt;

    // __shfl_xor_sync(mask, pt, 1, 4): swap with lane^1, then include().
    gsExch[lt] = pt;
    GroupMemoryBarrierWithGroupSync();
    const float3 val = gsExch[lt ^ 1u];
    GroupMemoryBarrierWithGroupSync();
    // P9.89: bounds.include(val) (cudaBox.cu L208) = selectMin/selectMax(own, partner)
    bmin = float3(bxSelMin(bmin.x, val.x), bxSelMin(bmin.y, val.y), bxSelMin(bmin.z, val.z));
    bmax = float3(bxSelMax(bmax.x, val.x), bxSelMax(bmax.y, val.y), bxSelMax(bmax.z, val.z));

    // __shfl_xor_sync(..., 2, 4) on the running minimum.
    gsExch[lt] = bmin;
    GroupMemoryBarrierWithGroupSync();
    const float3 omin = gsExch[lt ^ 2u];
    GroupMemoryBarrierWithGroupSync();
    // P9.89: bounds.minimum.minimum(partner) (cudaBox.cu L210-212) = selectMin(own, partner)
    bmin = float3(bxSelMin(bmin.x, omin.x), bxSelMin(bmin.y, omin.y), bxSelMin(bmin.z, omin.z));

    // __shfl_xor_sync(..., 2, 4) on the running maximum.
    gsExch[lt] = bmax;
    GroupMemoryBarrierWithGroupSync();
    const float3 omax = gsExch[lt ^ 2u];
    GroupMemoryBarrierWithGroupSync();
    // P9.89: bounds.maximum.maximum(partner) (cudaBox.cu L214-216) = selectMax(own, partner)
    bmax = float3(bxSelMax(bmax.x, omax.x), bxSelMax(bmax.y, omax.y), bxSelMax(bmax.z, omax.z));

    if (t < numLanes)
    {
        gBoundsMin[t] = float4(bmin, 0.0f);
        gBoundsMax[t] = float4(bmax, 0.0f);
    }
}
