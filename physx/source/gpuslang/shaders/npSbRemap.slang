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

// npSbRemap.slang
//
// Slang port of the collision-mesh -> simulation-mesh contact remap and
// the contact reorder tail of softbodyPrimitives.cu (tranche 101):
//
// rmOther — sb_other_contact_remap_to_simLaunch (:2133-2273): one warp
// per contact.  The contact count is CLAMPED to maxNumContacts and the
// clamp is written back to *totalContactCount by warp 0 lane 0; an
// optional *prevContactCount offsets the processed range.  For each
// contact, the tet id in pairInd1 selects a remap range
// [accumRemap[tet-1], accumRemap[tet]) of candidate SIM tets, walked in
// 32-candidate warp batches: a candidate CONTAINING the contact point
// (computeTetBarycentric's eps window, 1e-5) sets intersect and the
// batch's ballot breaks the walk — the winner is the LOWEST intersecting
// lane of the FIRST batch with any hit (NOT the globally best
// candidate; a containing tet in a later batch is never seen).
// Non-containing candidates keep a PER-LANE running minimum of the
// squared distance to Gu::closestPtPointTetrahedron (with the
// barycentric of the CLOSEST POINT, recomputed).  If no batch hit, a
// warpReduction<MinOpFloat> picks the global minimum with the LOWEST
// lane winning float-equal ties, and that lane writes the barycentric
// CLAMPED per component plus pairInd1 = PxEncodeSoftBodyIndex(sbId,
// simTetInd).  NOTE: the fallback bary clamps exist only in THIS kernel;
// the fem variant has them commented out.
//
// rmFem — sb_fem_contact_remap_to_simLaunch (:2276-2444): same skeleton
// with a side-select — blockIdx.y == 0 remaps pairInd1/barycentric1,
// == 1 remaps pairInd0/barycentric0 (a mode parameter here).  The
// CONTACT POINT is stashed in the remapped barycentric array itself
// (pre-remap convention).  Candidates additionally pass isValidTet
// (signed volume > 1e-9, cudaNpCommon.h:180); a contact whose ballot of
// hasTet over ALL lanes and batches is zero falls back to
// PxEncodeSoftBodyIndex(sbId, 0) with a ZERO barycentric.  QUIRKS kept:
// the intersect path does NOT reset sqDist (`//sqDist = 0.f;` commented
// upstream — unreachable difference since the batch breaks), the
// fallback bary is NOT clamped (commented upstream), the count clamp is
// written back by BOTH blockIdx.y slices (benign duplicate; single
// write per mode here), and pairInd values move through u64<->u32
// widening/truncation (pairInd1 read as u32 from the u64 field).
//
// rmReorder — fem_reorderRigidContactsLaunch (:1654-1693) and
// sb_reorderPSContactsLaunch (:1695-1726), selected by a mode flag:
// gather contacts/normalPens/barycentrics/contactInfos through
// remapByRigid, with the rigid flavour additionally extracting
// sortedRigidIds[i] = (PxU64)contactInfo.pairInd0.  FAITHFUL QUIRK
// (documented, behaviourally neutral): the upstream iteration count is
// the mis-parenthesised `(total + N - 1) / blockDim.x * gridDim.x` —
// an over-count saved by the `index >= totalNumContacts` early return;
// the port loops one thread per index under the same bound guard.
//
// BINDING NOTE for the eventual launch glue: the CUDA `barycentrics`
// input of fem_reorderRigidContactsLaunch is the SAME device buffer that
// sb_other_contact_remap_to_simLaunch writes as `outBarycentric` — in
// this port that logical buffer binds to gBary0 for rmReorder but to
// gBary1 for rmOther (per-entry slot assignment differs deliberately).
//
// KNOWN LIMITS (verified textually, not pinnable): the rmOther fallback's
// per-component bary clamps are UPSTREAM-REDUNDANT — computeTetBarycentric
// already clamps its own output, so the second clamp is output-neutral
// (both kept).  Gu::closestPtPointTetrahedron's face ORDER is pinnable
// only through exact cross-face ties, where the tied faces return the
// SAME point (an edge/vertex of the tet) up to ULPs — the function
// returns only the position, so order swaps are observationally
// equivalent within tolerance.  A CONSISTENT shift typo in the
// PxGetSoftBodyId/PxEncodeSoftBodyIndex pair (e.g. >>19 with <<19)
// round-trips bit-identically for element indices below 2^19 and is
// unobservable with test-sized meshes; only mismatched shifts are
// caught (verified textually against PxgSimulationCoreDesc.h).
//
// WARP SERIALIZATION: rmOther/rmFem serialize one warp-contact per
// thread — per-lane running minima are tracked in lane arrays, the
// batch ballot is the serialized any-of-32, and the min reduction is
// best-of-32 with the lowest equal lane winning, exactly the
// warpReduction<MinOpFloat> contract.
//
// Engine glue not ported (documented): PxgSoftBody loads (tables are
// bound directly), grid-stride distribution, and the launch dimensions.
// Gu::closestPtPointTetrahedron (GuDistancePointTetrahedron.h:77) is
// ported here for the first time — its face order (a,b,c), (a,c,d),
// (a,d,b), (b,d,c) and strict-< first-wins ties are NOT interchangeable
// with softbodyPrimitives' closestPtTetrahedron (different enumeration,
// no hint mask, no normals).

import pxgSlangCommon;

static const uint PXG_BITSHIFT_TET_ID = 20;
static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffff;
static const float PX_MAX_F32_LOCAL = 3.402823466e38f;

struct SrParams
{
    uint maxContacts;  // clamp bound
    uint mode;         // rmFem: pairIndex (0/1); rmReorder: 1 = rigid flavour (sortedRigidIds)
    uint hasPrev;      // rmOther: read gPrev[0] as start offset
    uint pad0;
}

[[vk::push_constant]]
ConstantBuffer<SrParams> gParams;

RWStructuredBuffer<float4> gPoint;        // binding 0: contact points (rmOther) / reorder input contacts
RWStructuredBuffer<float4> gBary0;        // binding 1
RWStructuredBuffer<float4> gBary1;        // binding 2
RWStructuredBuffer<uint4> gInfo;          // binding 3: PxgFemOther/FemFem contact info
RWStructuredBuffer<uint4> gSimTets;       // binding 4
RWStructuredBuffer<float4> gSimVerts;     // binding 5
RWStructuredBuffer<uint> gRemapColToSim;  // binding 6
RWStructuredBuffer<uint> gAccumRemap;     // binding 7
RWStructuredBuffer<uint> gCount;          // binding 8: totalContactCount
RWStructuredBuffer<uint> gPrev;           // binding 9: prevContactCount
RWStructuredBuffer<uint> gRemapByRigid;   // binding 10
RWStructuredBuffer<float4> gNormalPen;    // binding 11: reorder input
RWStructuredBuffer<float4> gSortedPoint;  // binding 12
RWStructuredBuffer<float4> gSortedNp;     // binding 13
RWStructuredBuffer<float4> gSortedBary;   // binding 14
RWStructuredBuffer<uint4> gSortedInfo;    // binding 15
RWStructuredBuffer<uint2> gSortedRigidIds;// binding 16
RWStructuredBuffer<float4> gDebug;        // binding 17

uint encodeSoftBody(uint sbId, uint tetId) { return (sbId << PXG_BITSHIFT_TET_ID) | tetId; }
uint getSoftBodyId(uint compressedId) { return compressedId >> PXG_BITSHIFT_TET_ID; }
uint getSoftBodyElementIndex(uint compressedId) { return PX_MAX_NB_DEFORMABLE_VOLUME_TET & compressedId; }

// isValidTet (cudaNpCommon.h:180): signed volume STRICTLY > 1e-9
bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// closestPtPointTriangle2 (GuDistancePointTriangle.h) — the Ericson core
float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return a + w * ac;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

// Gu::closestPtPointTetrahedron (GuDistancePointTetrahedron.h:77):
// face order (a,b,c), (a,c,d), (a,d,b), (b,d,c); strict < keeps the first
float3 guClosestPtPointTetrahedron(float3 p, float3 a, float3 b, float3 c, float3 d)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ad = d - a;
    const float3 bc = c - b;
    const float3 bd = d - b;

    float3 bestClosestPt = closestPtPointTriangle2(p, a, b, c, ab, ac);
    float3 diff = bestClosestPt - p;
    float bestSqDist = dot(diff, diff);

    float3 closestPt = closestPtPointTriangle2(p, a, c, d, ac, ad);
    diff = closestPt - p;
    float sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    closestPt = closestPtPointTriangle2(p, a, d, b, ad, ab);
    diff = closestPt - p;
    sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    closestPt = closestPtPointTriangle2(p, b, d, c, bd, bc);
    diff = closestPt - p;
    sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    return bestClosestPt;
}

// computeTetBarycentric (deformableCollision.cuh :96)
bool computeTetBarycentric(float3 a, float3 b, float3 c, float3 d, float3 point,
                           out float4 barycentric)
{
    const float eps = 1e-5f;
    const float minEps = -eps;
    const float maxEps = 1.0f + eps;
    const float3 ba = b - a;
    const float3 ca = c - a;
    const float3 da = d - a;
    const float3 pa = point - a;

    const float detBcd = dot(ba, cross(ca, da));
    const float detPcd = dot(pa, cross(ca, da));
    const float v = detPcd / detBcd;
    const float detBpd = dot(ba, cross(pa, da));
    const float w = detBpd / detBcd;
    const float detBcp = dot(ba, cross(ca, pa));
    const float x = detBcp / detBcd;
    const float u = 1.0f - v - w - x;
    barycentric = float4(clamp(u, 0.0f, 1.0f), clamp(v, 0.0f, 1.0f),
                         clamp(w, 0.0f, 1.0f), clamp(x, 0.0f, 1.0f));
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}

// ============================================================================
// rmOther — one thread serializes one warp-contact
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void rmOther(uint3 dtid : SV_DispatchThreadID)
{
    uint numContacts = gCount[0];
    if (numContacts > gParams.maxContacts)
    {
        numContacts = gParams.maxContacts;
        if (dtid.x == 0)
            gCount[0] = gParams.maxContacts; // clamp write-back (warp 0 lane 0)
    }
    const uint startIndex = gParams.hasPrev != 0 ? gPrev[0] : 0;
    const uint contactIndex = dtid.x + startIndex;
    if (contactIndex >= numContacts) return;

    const float3 contact = gPoint[contactIndex].xyz;
    const uint pairInd1 = gInfo[contactIndex].z;
    const uint softbodyId = getSoftBodyId(pairInd1);
    const uint tetInd = getSoftBodyElementIndex(pairInd1);

    const uint endInd = gAccumRemap[tetInd];
    const uint startInd = (tetInd != 0) ? gAccumRemap[tetInd - 1] : 0;
    const uint range = endInd - startInd;
    const uint nbIterations = (range + 31) / 32;

    // per-lane running state (persists across batches, like the CUDA registers)
    uint laneSimTet[32];
    float4 laneBary[32];
    float laneSqDist[32];
    for (uint l = 0; l < 32; ++l)
    {
        laneSimTet[l] = 0xffffffffu;
        laneBary[l] = float4(0, 0, 0, 0);
        laneSqDist[l] = PX_MAX_F32_LOCAL;
    }

    bool valid = false;
    for (uint j = 0; j < nbIterations && !valid; ++j)
    {
        uint hitLane = 0xffffffffu;
        uint hitTet = 0;
        float4 hitBary = float4(0, 0, 0, 0);
        for (uint l = 0; l < 32; ++l)
        {
            const uint workIndex = l + j * 32;
            if (workIndex >= range) continue;
            const uint tmpSimTetInd = gRemapColToSim[startInd + workIndex];
            const uint4 simTet = gSimTets[tmpSimTetInd];
            const float3 a = gSimVerts[simTet.x].xyz;
            const float3 b = gSimVerts[simTet.y].xyz;
            const float3 c = gSimVerts[simTet.z].xyz;
            const float3 d = gSimVerts[simTet.w].xyz;

            float4 tmpBarycentric;
            const bool intersect = computeTetBarycentric(a, b, c, d, contact, tmpBarycentric);
            if (!intersect)
            {
                const float3 tmpClosest = guClosestPtPointTetrahedron(contact, a, b, c, d);
                const float3 v = contact - tmpClosest;
                const float tmpDist = dot(v, v);
                if (tmpDist < laneSqDist[l])
                {
                    laneSqDist[l] = tmpDist;
                    laneSimTet[l] = tmpSimTetInd;
                    float4 cb;
                    computeTetBarycentric(a, b, c, d, tmpClosest, cb);
                    laneBary[l] = cb;
                }
            }
            else if (hitLane == 0xffffffffu) // ballot: lowest intersecting lane wins
            {
                hitLane = l;
                hitTet = tmpSimTetInd;
                hitBary = tmpBarycentric;
            }
        }
        if (hitLane != 0xffffffffu)
        {
            uint4 info = gInfo[contactIndex];
            info.z = encodeSoftBody(softbodyId, hitTet);
            gInfo[contactIndex] = info;
            gBary1[contactIndex] = hitBary;
            valid = true;
        }
    }

    if (!valid)
    {
        // warpReduction<MinOpFloat>: global min, LOWEST float-equal lane wins
        float best = PX_MAX_F32_LOCAL;
        for (uint l = 0; l < 32; ++l) best = min(best, laneSqDist[l]);
        uint winner = 0;
        for (uint l = 0; l < 32; ++l)
            if (laneSqDist[l] == best) { winner = l; break; }
        float4 bary = laneBary[winner];
        bary = clamp(bary, float4(0, 0, 0, 0), float4(1, 1, 1, 1));
        uint4 info = gInfo[contactIndex];
        info.z = encodeSoftBody(softbodyId, laneSimTet[winner]);
        gInfo[contactIndex] = info;
        gBary1[contactIndex] = bary;
    }
    gDebug[contactIndex] = float4(valid ? 1 : 0, 0, 0, 0);
}

// ============================================================================
// rmFem — side-selected remap with isValidTet gate and the tet0 fallback
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void rmFem(uint3 dtid : SV_DispatchThreadID)
{
    uint numContacts = gCount[0];
    if (numContacts > gParams.maxContacts)
    {
        numContacts = gParams.maxContacts;
        if (dtid.x == 0)
            gCount[0] = gParams.maxContacts;
    }
    const uint startOffset = gPrev[0];
    const uint contactIndex = dtid.x + startOffset;
    if (contactIndex >= numContacts) return;

    const uint4 contactInfo = gInfo[contactIndex];
    const uint pairIndex = gParams.mode;
    // pairInd1 is the u32 field; pairInd0 the u64 (low dword carries the id)
    uint pairInd;
    if (pairIndex == 1)
        pairInd = contactInfo.x; // pairInd0 low dword (u64 -> u32 truncation)
    else
        pairInd = contactInfo.z; // pairInd1

    // the contact point is stashed in the barycentric array being remapped
    const float4 tBarycentric = (pairIndex == 1) ? gBary0[contactIndex] : gBary1[contactIndex];
    const float3 contact = tBarycentric.xyz;

    const uint softbodyId = getSoftBodyId(pairInd);
    const uint tetInd = getSoftBodyElementIndex(pairInd);

    const uint endInd = gAccumRemap[tetInd];
    const uint startInd = (tetInd != 0) ? gAccumRemap[tetInd - 1] : 0;
    const uint range = endInd - startInd;
    const uint nbIterations = (range + 31) / 32;

    uint laneSimTet[32];
    float4 laneBary[32];
    float laneSqDist[32];
    for (uint l = 0; l < 32; ++l)
    {
        laneSimTet[l] = 0xffffffffu;
        laneBary[l] = float4(0, 0, 0, 0);
        laneSqDist[l] = PX_MAX_F32_LOCAL;
    }

    bool valid = false;
    bool anyTet = false;
    for (uint j = 0; j < nbIterations && !valid; ++j)
    {
        uint hitLane = 0xffffffffu;
        uint hitTet = 0;
        float4 hitBary = float4(0, 0, 0, 0);
        for (uint l = 0; l < 32; ++l)
        {
            const uint workIndex = l + j * 32;
            if (workIndex >= range) continue;
            const uint tmpSimTetInd = gRemapColToSim[startInd + workIndex];
            const uint4 simTet = gSimTets[tmpSimTetInd];
            const float3 a = gSimVerts[simTet.x].xyz;
            const float3 b = gSimVerts[simTet.y].xyz;
            const float3 c = gSimVerts[simTet.z].xyz;
            const float3 d = gSimVerts[simTet.w].xyz;

            if (!isValidTet(a, b, c, d))
                continue;
            anyTet = true;

            float4 tmpBarycentric;
            const bool intersect = computeTetBarycentric(a, b, c, d, contact, tmpBarycentric);
            if (!intersect)
            {
                const float3 tmpClosest = guClosestPtPointTetrahedron(contact, a, b, c, d);
                const float3 v = contact - tmpClosest;
                const float tmpDist = dot(v, v);
                if (tmpDist < laneSqDist[l])
                {
                    laneSqDist[l] = tmpDist;
                    laneSimTet[l] = tmpSimTetInd;
                    float4 cb;
                    computeTetBarycentric(a, b, c, d, tmpClosest, cb);
                    laneBary[l] = cb;
                }
            }
            else if (hitLane == 0xffffffffu)
            {
                hitLane = l;
                hitTet = tmpSimTetInd;
                hitBary = tmpBarycentric;
                // NOTE: sqDist deliberately NOT reset (upstream `//sqDist = 0.f;`)
            }
        }
        if (hitLane != 0xffffffffu)
        {
            uint4 info = gInfo[contactIndex];
            const uint newPairInd = encodeSoftBody(softbodyId, hitTet);
            if (pairIndex == 1) { info.x = newPairInd; info.y = 0; } // u32 -> u64 widen
            else info.z = newPairInd;
            gInfo[contactIndex] = info;
            if (pairIndex == 1) gBary0[contactIndex] = hitBary;
            else gBary1[contactIndex] = hitBary;
            valid = true;
        }
    }

    if (!anyTet)
    {
        uint4 info = gInfo[contactIndex];
        const uint newPairInd = encodeSoftBody(softbodyId, 0);
        if (pairIndex == 1) { info.x = newPairInd; info.y = 0; }
        else info.z = newPairInd;
        gInfo[contactIndex] = info;
        if (pairIndex == 1) gBary0[contactIndex] = float4(0, 0, 0, 0);
        else gBary1[contactIndex] = float4(0, 0, 0, 0);
        gDebug[contactIndex] = float4(2, 0, 0, 0);
        return;
    }
    if (!valid)
    {
        float best = PX_MAX_F32_LOCAL;
        for (uint l = 0; l < 32; ++l) best = min(best, laneSqDist[l]);
        uint winner = 0;
        for (uint l = 0; l < 32; ++l)
            if (laneSqDist[l] == best) { winner = l; break; }
        uint4 info = gInfo[contactIndex];
        const uint newPairInd = encodeSoftBody(softbodyId, laneSimTet[winner]);
        if (pairIndex == 1) { info.x = newPairInd; info.y = 0; }
        else info.z = newPairInd;
        gInfo[contactIndex] = info;
        // fallback bary NOT clamped in this kernel (clamps commented upstream)
        if (pairIndex == 1) gBary0[contactIndex] = laneBary[winner];
        else gBary1[contactIndex] = laneBary[winner];
    }
    gDebug[contactIndex] = float4(valid ? 1 : 0, 0, 0, 0);
}

// ============================================================================
// rmReorder — gather through remapByRigid; mode 1 also extracts rigid u64s
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void rmReorder(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    const uint totalNumContacts = gCount[0];
    if (index >= totalNumContacts) return; // the upstream over-count is bound-guarded

    const uint remapIndex = gRemapByRigid[index];
    gSortedPoint[index] = gPoint[remapIndex];
    gSortedNp[index] = gNormalPen[remapIndex];
    gSortedBary[index] = gBary0[remapIndex];
    const uint4 contactInfo = gInfo[remapIndex];
    gSortedInfo[index] = contactInfo;
    if (gParams.mode == 1)
        gSortedRigidIds[index] = contactInfo.xy; // (PxU64)pairInd0
}
