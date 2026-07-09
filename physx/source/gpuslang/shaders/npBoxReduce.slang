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

// npBoxReduce.slang
//
// Slang port of the Stage-2 tail of doBoxBoxGenerateContacts() from
// gpunarrowphase/src/CUDA/cudaBox.cu (lines 806-940): the per-lane contacts
// produced by calculateContacts are compacted into the box-box contact patch.
// Each of the 4 lanes carries 0..3 local-space contacts (xyz + penetration in
// .w).  The group:
//   1. computes the width-4 inclusive prefix sum of the per-lane counts and the
//      group total;
//   2. if total <= 6 (PXG_MAX_NUM_POINTS_PER_CONTACT_PATCH): stores every contact
//      compacted at its prefix offset, transformed to global space;
//   3. else runs the 4-point manifold reduction — keep the deepest contact, the
//      one furthest from it (in the incident-face plane), and the two extrema of
//      the perpendicular direction — each chosen with a minIndex4 argmin/argmax
//      (lowest-lane tie-break), then transforms the 4 to global space.
//
// The CUDA uses __shfl / minIndex4 / __syncwarp within the 4-lane group; here the
// group lives in a 256-thread block (64 groups) with a groupshared exchange.  The
// total<=6 vs >6 decision is group-uniform, so the reduction's barriers are kept
// unconditional (executed by every thread) and only the data writes are
// predicated — this keeps all block-wide barriers in uniform control flow.

static const uint MAXP = 6u;         // PXG_MAX_NUM_POINTS_PER_CONTACT_PATCH
static const float BIG = 3.4028235e38f;

struct LaneIn
{
    float4 c0, c1, c2; // up to 3 local-space contacts (xyz, .w = penetration)
    float4 meta;       // (count, unused...)
}

struct GroupIn
{
    float4 ntCol0, ntCol1, ntCol2, ntP; // newTransformV
    float4 fn;                          // incidentFaceNormalInNew (.xyz)
}

struct BoxReduceParams { uint numGroups; }

[[vk::push_constant]]
ConstantBuffer<BoxReduceParams> gParams;

RWStructuredBuffer<LaneIn>  gLaneIn;      // binding 0  4 per group
RWStructuredBuffer<GroupIn> gGroupIn;     // binding 1  1 per group
RWStructuredBuffer<float4>  gOutContacts; // binding 2  6 per group (global space)
RWStructuredBuffer<uint>    gOutTotal;    // binding 3  1 per group

groupshared uint   gsCount[256];
groupshared float  gsA[256];
groupshared float  gsB[256];
groupshared float4 gsLC[384]; // 6 slots per group

float3 xform(GroupIn g, float3 v)
{
    return g.ntCol0.xyz * v.x + g.ntCol1.xyz * v.y + g.ntCol2.xyz * v.z + g.ntP.xyz;
}

// ---------------------------------------------------------------------------
// boxCompactContacts: mirrors the Stage-2 tail of doBoxBoxGenerateContacts()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void boxCompactContacts(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = dtid.x;
    const uint lt = gtid.x;
    const uint lane = lt & 3u;
    const uint gbase = lt & ~3u;
    const uint groupIdx = t >> 2;
    const uint numGroups = gParams.numGroups;
    const uint numLanes = numGroups * 4u;
    const bool valid = groupIdx < numGroups;
    const uint safeT = t < numLanes ? t : (numLanes > 0u ? numLanes - 1u : 0u);
    const uint lbase = (lt >> 2) * 6u;

    const LaneIn li = gLaneIn[safeT];
    const GroupIn gin = gGroupIn[valid ? groupIdx : (numGroups > 0u ? numGroups - 1u : 0u)];
    const float3 fn = gin.fn.xyz;
    const uint count = (uint)(li.meta.x + 0.5f);
    float4 origC[3] = { li.c0, li.c1, li.c2 };

    // --- 1. inclusive prefix sum of counts ---
    gsCount[lt] = count;
    GroupMemoryBarrierWithGroupSync();
    const uint n0 = gsCount[gbase], n1 = gsCount[gbase + 1u], n2 = gsCount[gbase + 2u], n3 = gsCount[gbase + 3u];
    const uint total = n0 + n1 + n2 + n3;
    uint rc = n0;
    if (lane >= 1u) rc += n1;
    if (lane >= 2u) rc += n2;
    if (lane >= 3u) rc += n3;
    const uint exclStart = rc - count;
    const bool caseB = total > MAXP;

    // --- 2. total <= 6: compacted store (no barrier, distinct positions) ---
    if (valid && !caseB)
    {
        for (uint i = 0u; i < count; ++i)
        {
            const float3 gp = xform(gin, origC[i].xyz);
            gOutContacts[groupIdx * 6u + exclStart + i] = float4(gp, origC[i].w);
        }
    }

    // --- 3. total > 6: 4-point manifold reduction (barriers unconditional) ---
    // deepest contact (min penetration .w)
    float dW = BIG; uint dI = 0u;
    for (uint i = 0u; i < count; ++i) { if (origC[i].w < dW) { dW = origC[i].w; dI = i; } }
    gsA[lt] = caseB ? dW : BIG;
    GroupMemoryBarrierWithGroupSync();
    {
        const float m = min(gsA[gbase], min(gsA[gbase + 1u], min(gsA[gbase + 2u], gsA[gbase + 3u])));
        const uint win = (gsA[gbase] == m) ? 0u : ((gsA[gbase + 1u] == m) ? 1u : ((gsA[gbase + 2u] == m) ? 2u : 3u));
        if (caseB && lane == win) gsLC[lbase + 0u] = origC[dI];
    }
    GroupMemoryBarrierWithGroupSync();
    const float3 deepestPt = gsLC[lbase + 0u].xyz;

    // furthest from deepest, measured in the incident-face plane
    const float3 dpp = deepestPt - fn * dot(deepestPt, fn);
    float fW = -BIG; uint fI = 0u;
    for (uint i = 0u; i < count; ++i)
    {
        const float3 d = dpp - origC[i].xyz;
        const float ds = dot(d, d);
        if (ds > fW) { fW = ds; fI = i; }
    }
    gsA[lt] = caseB ? -fW : BIG; // minIndex4(-fW) == argmax(fW)
    GroupMemoryBarrierWithGroupSync();
    {
        const float m = min(gsA[gbase], min(gsA[gbase + 1u], min(gsA[gbase + 2u], gsA[gbase + 3u])));
        const uint win = (gsA[gbase] == m) ? 0u : ((gsA[gbase + 1u] == m) ? 1u : ((gsA[gbase + 2u] == m) ? 2u : 3u));
        if (caseB && lane == win) gsLC[lbase + 1u] = origC[fI];
    }
    GroupMemoryBarrierWithGroupSync();
    const float3 furthestPt = gsLC[lbase + 1u].xyz;

    // two extrema perpendicular to the deepest->furthest line
    const float3 dir = cross(fn, furthestPt - deepestPt);
    float mxV = -BIG, mnV = BIG; uint mxI = 0u, mnI = 0u;
    for (uint i = 0u; i < count; ++i)
    {
        const float dd = dot(dir, origC[i].xyz - deepestPt);
        if (dd > mxV) { mxV = dd; mxI = i; }
        if (dd < mnV) { mnV = dd; mnI = i; }
    }
    gsA[lt] = caseB ? mnV : BIG;   // minIndex4(mnV)  == argmin
    gsB[lt] = caseB ? -mxV : BIG;  // minIndex4(-mxV) == argmax
    GroupMemoryBarrierWithGroupSync();
    {
        const float ma = min(gsA[gbase], min(gsA[gbase + 1u], min(gsA[gbase + 2u], gsA[gbase + 3u])));
        const uint winA = (gsA[gbase] == ma) ? 0u : ((gsA[gbase + 1u] == ma) ? 1u : ((gsA[gbase + 2u] == ma) ? 2u : 3u));
        const float mb = min(gsB[gbase], min(gsB[gbase + 1u], min(gsB[gbase + 2u], gsB[gbase + 3u])));
        const uint winB = (gsB[gbase] == mb) ? 0u : ((gsB[gbase + 1u] == mb) ? 1u : ((gsB[gbase + 2u] == mb) ? 2u : 3u));
        if (caseB && lane == winA) gsLC[lbase + 2u] = origC[mnI];
        if (caseB && lane == winB) gsLC[lbase + 3u] = origC[mxI];
    }
    GroupMemoryBarrierWithGroupSync();

    // output the reduced 4 (one per lane), transformed to global space
    if (valid && caseB)
    {
        const float4 lc = gsLC[lbase + lane];
        gOutContacts[groupIdx * 6u + lane] = float4(xform(gin, lc.xyz), lc.w);
    }

    if (valid && lane == 0u)
        gOutTotal[groupIdx] = caseB ? 4u : total;
}
