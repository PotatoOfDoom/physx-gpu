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

// npTriHfNp.slang
//
// Slang port of trimeshHeightfieldNarrowphase (trimeshCollision.cu
// :435-682), the block-level (32-warp) triangle-mesh-vs-heightfield
// narrowphase.  The upstream kernel treats the heightfield LIKE A PLANE:
// vertices are tested against local x with worldNormal = the
// heightfield rotation's basis vector 0 — no heightfield samples are
// read, and there is NO persistent-manifold gate (always full gen).
//
// Three FAITHFUL upstream quirks, all deliberately preserved:
// 1. contactReduce runs with the DEFAULT TDoDupeTest = true (the plane
//    kernel passes false): a duplicate pre-cull removes lanes whose RAW
//    point sits within 1e-8 squared distance of a deeper lane (or an
//    equal-separation lane with a HIGHER index, so the highest lane of
//    an equal group survives) before the anchor selection;
// 2. the warp-0 batching loop reads contacts[readIndex] — WITHOUT
//    adding ind (the plane kernel reads readIndex + ind): every round
//    re-reads the FRONT of the shared buffer, so packed contacts beyond
//    slot 31 of an iteration never participate in the second reduction
//    (the bound check `ind + readIndex < totalContacts` and the
//    `ind += popc(readMask)` advance still walk ind forward, so the
//    loop terminates); the re-read duplicates are then culled by the
//    dupe test against the surviving lanes;
// 3. the overflow checks test a SINGLE element (`offset +
//    sizeof(PxContact) > limit`, not nbContacts * 16), the force check
//    is an ELSE-IF (skipped entirely when the contact buffer overflows,
//    so its flag cannot be raised in that case), and nbContacts is
//    NEVER zeroed — a poisoned pair only NULLs the output pointers
//    while the patch side keeps the raw count.
//
// FOURTH divergence — DELIBERATELY RESOLVED, NOT TRANSCRIBED: unlike the
// plane kernel (:220), the HF kernel never refreshes hasContact from the
// reduced mask before the stage-1 pack (:556 still tests the PRE-reduce
// acceptance while the slot index :554/:558 comes from the POST-reduce
// mask): in CUDA every accepted-but-culled lane also stores, racing
// survivors on the same shared slot, spilling into the next warp's slot
// 0, and — with retainedCount 6 and all 32 warps saturated — writing one
// element past the 198-slot shared arrays.  That is a nondeterministic
// data race / OOB in upstream 5.6.1 which no deterministic port can
// reproduce; this port packs REDUCE SURVIVORS ONLY (the plane kernel's
// well-defined behavior, and the only race-free reading).  Upstream's
// launch site (PxgNarrowphaseCore.cpp:1324-1350) also passes this
// kernel's arguments in an order that mismatches its signature, so the
// path appears dead/broken as shipped and the race is latent.
//
// WARP SERIALIZATION: one thread serializes the whole 32-warp block —
// same structure as npTriPlaneNp.slang (tranche 103).  All decisions
// compare exactly-computed values on the exact-arithmetic fixtures, so
// the port is bit-deterministic against the CPU reference — REQUIRING a
// correctly-rounded sqrt on both sides (CUDA sqrtf and std::sqrt are;
// Vulkan only mandates a ULP bound, and the fixtures contain projected
// distances within 1 ULP of each other after sqrt).
//
// Engine glue not ported (documented): LoadShapePairWarp/transform
// caches, readTriangleMesh blob walk (cases carry vert ranges), and the
// registerContactPatch + insertIntoPatchStream tail (tranches 44/45;
// this port emits nbContacts + worldNormal, the inputs that tail
// consumes).  toleranceLength is a kernel parameter but unused by the
// upstream body (no margins without the manifold gate).

import pxgSlangCommon;

static const uint PLANE_TRI_MAX_CONTACTS = 6u;
static const float FLT_MAX_LOCAL = 3.402823466e38f;
static const uint OVERFLOW_CONTACT = 1u;
static const uint OVERFLOW_FORCE = 2u;

struct ThParams
{
    uint numCases;
    uint contactBytesLimit;
    uint forceBytesLimit;
    uint pad0;
}

struct ThCase
{
    float4 meshQ;     // trimesh transform rotation
    float4 meshP;     // trimesh transform position
    float4 hfQ;       // heightfield transform rotation
    float4 hfP;       // heightfield transform position; w = cDistance
    float4 scale;     // trimesh scale (xyz); w = clusterBias
    float4 scaleRot;  // trimesh scale rotation quaternion
    uint4 verts;      // x = vert offset, y = numVerts
}

[[vk::push_constant]]
ConstantBuffer<ThParams> gParams;

RWStructuredBuffer<ThCase> gCase;         // binding 0
RWStructuredBuffer<float4> gVerts;        // binding 1: mesh vertices (vertex space)
RWStructuredBuffer<uint> gCounters;       // binding 2: [0] contactsBytes, [1] forceAndIndiceBytes, [2] overflowError
RWStructuredBuffer<float4> gContactStream;// binding 3: byte offset / 16 indexing
RWStructuredBuffer<uint4> gOutRec;        // binding 4: per case (contactPointsOfs, contactForcesOfs, nbContacts, 0)
RWStructuredBuffer<float4> gDebug;        // binding 5: per case (worldNormal, retainedCount)

// ---- helpers -----------------------------------------------------------------
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}

// NOTE: with an empty mask (or all masked values at the sentinel) these
// return idx 0, unlike CUDA maxIndex/minIndex (lowest MASKED index); every
// call site below provably passes a nonempty mask with finite values
uint lowestArgMax(float vals[32], uint mask, out float best)
{
    best = -FLT_MAX_LOCAL;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
uint lowestArgMin(float vals[32], uint mask, out float best)
{
    best = FLT_MAX_LOCAL;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// contactReduce<clustering=true, keepDeepestPCM=true, maxPoints=6,
// dupeTest=TRUE (default)> (contactReduction.cuh), on lane arrays
uint contactReduce6Dupe(float3 ptRaw[32], float sep[32], float3 normal, uint allMask, float clusterBias)
{
    const int maxPoints = int(PLANE_TRI_MAX_CONTACTS);

    // duplicate pre-cull on the RAW points: lane l dies to a strictly
    // deeper coincident lane, or to an equal-separation coincident lane
    // with a HIGHER index
    {
        uint dupes = 0;
        for (uint l = 0; l < 32; ++l)
        {
            if (((allMask >> l) & 1) == 0) continue;
            bool imADupe = false;
            for (uint m = allMask; m != 0; m &= (m - 1))
            {
                const uint i = firstbitlow(m);
                const float3 dv = ptRaw[i] - ptRaw[l];
                const float d = dot(dv, dv);
                if (d < 1e-8f && (sep[i] < sep[l] || (sep[i] == sep[l] && i > l)))
                    imADupe = true;
            }
            if (imADupe) dupes |= 1u << l;
        }
        allMask &= ~dupes;
    }

    float3 point[32];
    for (uint j = 0; j < 32; ++j)
        point[j] = ptRaw[j] - normal * dot(ptRaw[j], normal);

    const uint newCount = countbits(allMask);
    if (newCount <= uint(maxPoints))
        return allMask;

    float v;
    float vals[32];

    for (uint j = 0; j < 32; ++j) vals[j] = sep[j];
    const uint i0 = lowestArgMax(vals, allMask, v);
    uint mask = 1u << i0;

    for (uint j = 0; j < 32; ++j)
    {
        const float3 d = point[i0] - point[j];
        vals[j] = sqrt(dot(d, d)); // PxVec3::magnitude(); sqrt(dot) keeps the
                                   // IEEE-sqrt assumption minimal vs OpLength
    }
    const uint i1 = lowestArgMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;

    const float3 dir = cross(normal, point[i1] - point[i0]);
    float dval[32];
    for (uint j = 0; j < 32; ++j) dval[j] = dot(dir, point[j] - point[i0]);

    for (uint j = 0; j < 32; ++j) vals[j] = dval[j];
    const uint f = lowestArgMax(vals, allMask & ~mask, v);
    mask |= 1u << f;

    for (uint j = 0; j < 32; ++j) vals[j] = dval[j];
    const uint g = lowestArgMin(vals, allMask & ~mask, v);
    mask |= 1u << g;

    if (countbits(mask) == 4 && maxPoints > 4)
    {
        const uint i4 = lowestArgMin(sep, allMask & ~mask, v);
        mask |= 1u << i4;
    }

    // clustering
    {
        float sepB[32];
        for (uint j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);

        int label[32];
        int nbClusters = 0;
        {
            uint m = mask;
            float tbest[32];
            for (uint j = 0; j < 32; ++j) { tbest[j] = FLT_MAX_LOCAL; label[j] = -1; }
            for (; m != 0; ++nbClusters, m &= (m - 1))
            {
                const uint anchor = firstbitlow(m);
                for (uint j = 0; j < 32; ++j)
                {
                    const float3 dv = point[j] - point[anchor];
                    const float d = dot(dv, dv);
                    if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
                }
            }
        }

        mask = 0;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint cluster = 0;
            for (uint j = 0; j < 32; ++j)
                if (label[j] == i) cluster |= 1u << j;
            cluster &= allMask;
            if (cluster != 0)
                mask |= 1u << lowestArgMin(sepB, cluster, v);
        }
        for (int i = nbClusters; i < maxPoints; ++i)
            mask |= 1u << lowestArgMin(sepB, allMask & ~mask, v);
    }
    return mask;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void thNarrowphase(uint3 dtid : SV_DispatchThreadID)
{
    const uint ci = dtid.x;
    if (ci >= gParams.numCases) return;
    const ThCase cs = gCase[ci];
    const float cDistance = cs.hfP.w;
    const float clusterBias = cs.scale.w;

    // trimeshToHeightfield = heightfieldTransform.transformInv(trimeshTransform)
    const PxgTransform28 hfT = makeTransform28(cs.hfQ, cs.hfP.xyz);
    const PxgTransform28 meshT = makeTransform28(cs.meshQ, cs.meshP.xyz);
    const PxgTransform28 trimeshToHf = transformInvTransform(hfT, meshT);

    const float3 worldNormal = quatBasisVector0(cs.hfQ);

    // shared contacts buffer: (32 + 1) * 6 slots
    float3 contacts[198];
    float seps[198];
    uint retainedCount = 0;

    const uint numVerts = cs.verts.y;
    const uint numIterationsRequired = (numVerts + 1023) / 1024;
    for (uint iter = 0; iter < numIterationsRequired; ++iter)
    {
        // stage 1: 32 warps, each reducing its 32 lanes
        float3 warpPts[32][6];
        float warpSeps[32][6];
        uint counters[32];
        for (uint w = 0; w < 32; ++w)
        {
            float3 lanePt[32];
            float laneSep[32];
            uint mask = 0;
            for (uint l = 0; l < 32; ++l)
            {
                lanePt[l] = float3(0, 0, 0);
                laneSep[l] = FLT_MAX_LOCAL;
                const uint i = iter * 1024 + w * 32 + l;
                if (i >= numVerts) continue;
                const float3 tp = gVerts[cs.verts.x + i].xyz;
                const float3 p = vertex2Shape(tp, cs.scale.xyz, cs.scaleRot);
                const float3 pInHfSpace = quatRotate(trimeshToHf.q(), p) + trimeshToHf.p();
                if (cDistance >= pInHfSpace.x) // INCLUSIVE
                {
                    laneSep[l] = pInHfSpace.x;
                    lanePt[l] = quatRotate(cs.meshQ, p) + cs.meshP.xyz;
                    mask |= 1u << l;
                }
            }
            mask = contactReduce6Dupe(lanePt, laneSep, worldNormal, mask, clusterBias);
            counters[w] = countbits(mask);
            uint k = 0;
            for (uint l = 0; l < 32; ++l)
            {
                if (((mask >> l) & 1) == 0) continue;
                warpPts[w][k] = lanePt[l];
                warpSeps[w][k] = laneSep[l];
                ++k;
            }
        }
        // block scan: exclusive prefix of warp counters, contacts behind retainedCount
        uint offset = 0;
        for (uint w = 0; w < 32; ++w)
        {
            for (uint k = 0; k < counters[w]; ++k)
            {
                contacts[retainedCount + offset + k] = warpPts[w][k];
                seps[retainedCount + offset + k] = warpSeps[w][k];
            }
            offset += counters[w];
        }
        const uint totalContacts = offset + retainedCount;

        // stage 2: warp-0 batching loop; FAITHFUL upstream quirk: the read
        // index omits `+ ind`, so every round refills from the BUFFER FRONT
        // (slots beyond 31 never participate); the re-read duplicates are
        // culled by the dupe test inside the reduce
        float3 lanePt[32];
        float laneSep[32];
        bool laneHas[32];
        for (uint l = 0; l < 32; ++l) { laneHas[l] = false; lanePt[l] = float3(0, 0, 0); laneSep[l] = FLT_MAX_LOCAL; }
        for (uint ind = 0; ind < totalContacts;)
        {
            uint readMask = 0;
            for (uint l = 0; l < 32; ++l)
                if (!laneHas[l]) readMask |= 1u << l;
            for (uint l = 0; l < 32; ++l)
            {
                if (laneHas[l]) continue;
                const uint readIndex = countbits(readMask & ((1u << l) - 1u));
                if (ind + readIndex < totalContacts)
                {
                    lanePt[l] = contacts[readIndex]; // NOT readIndex + ind
                    laneSep[l] = seps[readIndex];
                    laneHas[l] = true;
                }
            }
            uint mask = 0;
            for (uint l = 0; l < 32; ++l)
                if (laneHas[l]) mask |= 1u << l;
            ind += countbits(readMask); // advances by FREE lanes, read or not
            mask = contactReduce6Dupe(lanePt, laneSep, worldNormal, mask, clusterBias);
            for (uint l = 0; l < 32; ++l)
                laneHas[l] = ((mask >> l) & 1) != 0;
        }
        // compact survivors to the buffer front
        uint k2 = 0;
        for (uint l = 0; l < 32; ++l)
        {
            if (!laneHas[l]) continue;
            contacts[k2] = lanePt[l];
            seps[k2] = laneSep[l];
            ++k2;
        }
        retainedCount = k2;
    }

    // contact-stream tail; FAITHFUL upstream quirks: the overflow checks
    // test ONE element (not nbContacts * 16 / * 4), the force check is an
    // ELSE-IF (skipped when the contact buffer overflowed), and nbContacts
    // is never zeroed — a poisoned pair only NULLs the output pointers
    const uint nbContacts = retainedCount;
    uint contactByteOffset = 0xFFFFFFFFu;
    uint forceAndIndiceByteOffset = 0xFFFFFFFFu;
    if (nbContacts != 0)
    {
        InterlockedAdd(gCounters[0], 16u * nbContacts, contactByteOffset);
        InterlockedAdd(gCounters[1], 4u * nbContacts, forceAndIndiceByteOffset);
        if ((contactByteOffset + 16u) > gParams.contactBytesLimit)
        {
            InterlockedOr(gCounters[2], OVERFLOW_CONTACT);
            contactByteOffset = 0xFFFFFFFFu;
        }
        else if ((forceAndIndiceByteOffset + 4u) > gParams.forceBytesLimit)
        {
            InterlockedOr(gCounters[2], OVERFLOW_FORCE);
            forceAndIndiceByteOffset = 0xFFFFFFFFu;
        }
    }
    uint4 rec;
    if (contactByteOffset != 0xFFFFFFFFu)
    {
        // CUDA writes with one warp (threadIndexInWarp < nbContacts): at
        // most 32 float4s (nbContacts <= 6 structurally here)
        for (uint k = 0; k < nbContacts && k < 32u; ++k)
            gContactStream[contactByteOffset / 16u + k] = float4(contacts[k], seps[k]);
        rec = uint4(contactByteOffset, forceAndIndiceByteOffset, nbContacts, 0);
    }
    else
    {
        // NULL pointers; nbContacts stays RAW (the patch side keeps it)
        rec = uint4(0xFFFFFFFFu, 0xFFFFFFFFu, nbContacts, 0);
    }
    gOutRec[ci] = rec;
    gDebug[ci] = float4(worldNormal, float(retainedCount));
}
