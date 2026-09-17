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

// npCvxCorrelate.slang
//
// Slang port of correlate (convexMeshCorrelate.cu L95-344), the convex-trimesh
// multi-manifold patch builder, including its purgeEmptyTriangles prelude
// (L58-87, separately verified in tranche 114).  One warp per pair, serialized
// here as 32-lane arrays in a single thread.  Quirks ported verbatim:
//   - purge writes tempCounts = ((origSlot << 4) | nbContacts) with nbContacts
//     from getNbContacts (FOUR bits, (index & 0x7FFFFFFF) >> 27), but the
//     patch loop reads back only THREE (index & 7): a triangle with 8..15
//     contacts purges as valid yet contributes 8&7.. contacts, and at exactly
//     8 keeps its nonzero tempCounts forever (never accepted, never cleared),
//     so remainingTriangles stays nonzero and the patch loop runs out its
//     MULTIMANIFOLD_MAX_MANIFOLDS = 4 iterations;
//   - the zero-triangle early-out writes mNbManifolds of outBuffer[pairIndex]
//     while every other output goes to outBuffer[pair.cmIndex] -- two
//     different slots when cmIndex != pairIndex (modeled as slot 0 = pairIndex
//     and slot cmIndex);
//   - deepest-triangle scan: per-lane strided i += 32 with strict <
//     (first-strided-index wins per lane), then exact-min ballot across lanes,
//     lowest lane wins; candidates are triangles with tempCounts != 0 (the
//     WHOLE word, slot bits included);
//   - acceptance: nic.normal.dot(bestNormal) > PATCH_ACCEPTANCE_EPS
//     (0.996194698f, strict >); rejected lanes zero nbContacts but KEEP their
//     startIndex in shared triStartIndex;
//   - the contact pull loop: readMask = ~currentContactMask over all 32 lanes,
//     readIndex = c + warpScanExclusive(readMask), binarySearch over the
//     32-entry exclusive run-sums (last index with runsum <= readIndex -- with
//     zero-contact duplicates the contact-bearing lane is always the last of
//     an equal-runsum group), c += popc(readMask) (free lanes, not contacts
//     read), contactReduce<clustering, no PCM-deepest, 6, no dupe cull> only
//     when popc > 6 (strict); lane state (pA/separation/triIndex) persists
//     across 32-triangle blocks and pull rounds within a patch;
//   - output per patch at warpScanExclusive slots into mContacts[patch] of the
//     cmIndex slot: normal = -aToB.rotate(bestNormal), pointA = pA, pointB =
//     aToB.transform(pA) + aToB.rotate(bestNormal) * separation, penetration,
//     triIndex; mNbContacts[patch] written even for empty patches;
//   - numContactsTotal (cu L301) is accumulated but never used -- not modeled;
//     cmOutputs and pTempContactIndex are unused parameters.
// mRelativeTransform (q then p) is written only when nbPatches > 0;
// mNbManifolds of the cmIndex slot is always written on the non-early-out
// path.  count == CONVEX_TRIMESH_CACHED (0xFFFFFFFF) returns with NOTHING
// written (both slots keep their 0xFFFFFFFF sentinel here).

import pxgSlangContactReduceSerial;   // P9.83b-2: the reduction body, shared

static const float PX_MAX_F32 = 3.4028235e38f;
static const float PATCH_ACCEPTANCE_EPS = 0.996194698f;
static const uint CONVEX_TRIMESH_CACHED = 0xFFFFFFFFu;
static const uint MULTIMANIFOLD_MAX_MANIFOLDS = 4u;
static const uint SUBMANIFOLD_MAX_CONTACTS = 6u;
static const uint UNTOUCHED = 0xFFFFFFFFu;

struct CorrIn
{
    float4 aToBq;         // pair.aToB.q
    float4 aToBp;         // pair.aToB.p (xyz), w = clusterBias
    uint4 hdr;            // x = count (or CONVEX_TRIMESH_CACHED), y = startIndex, z = cmIndex (0/1), w pad
    float4 depth[16];     // depthBuffer[64]
    float4 nrm[64];       // nicBuffer normals (w pad)
    uint4 nicIdx[16];     // nicBuffer index words [64]
    uint4 cbuf[16];       // contactBuffer[..].index [64]
    float4 tempC[128];    // tempConvexTriContacts[..].contact_sepW
}

struct CorrOut
{
    float4 contacts[72];  // [patch][slot][3]: (pA, pen), (pB, asfloat(triIndex)), (normal, 0)
    uint4 nbC;            // mNbContacts[0..3]
    float4 relQ, relP;    // mRelativeTransform (written when nbPatches > 0)
    uint4 misc;           // x = mNbManifolds slot0(pairIndex), y = slot1, z = nbTriangles after purge, w = remainingTriangles final
    uint4 tempOut[16];    // final tempCounts[64]
    float4 depthOut[16];  // depthBuffer[64] after purge compaction
}

struct CorrParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CorrParams> gParams;

RWStructuredBuffer<CorrIn> gIn;    // binding 0
RWStructuredBuffer<CorrOut> gOut;  // binding 1

uint getNbContactsNic(uint index)
{
    return (index & 0x7FFFFFFFu) >> 27u; // (index & ~DeferredContactMask) >> NbContactsShift
}

uint getTriangleIndexNic(uint index)
{
    return index & ((1u << 27u) - 1u);
}

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
// P9.83b-2: THE BODY AND ITS TWO ARGMAX/ARGMIN HELPERS MOVED OUT.
// contactReduceShared is one CUDA template and this file carried the
// <true, false, 6, false> instantiation of it as its own transcription.
// pxgSlangContactReduceSerial holds the body once, with the template arguments as
// a CrsConfig; this is now a thin adapter and its fixture certifies the shared
// body rather than a private copy of it.
//
// THE EARLY-OUT IS NEW HERE AND IS A NO-OP.  The shared body returns allMask when
// newCount <= maxPoints (contactReduction.cuh:85); this file's transcription had
// omitted it, which is safe because the CALL SITE hoists exactly that test
// (`if (countbits(currentContactMask) > SUBMANIFOLD_MAX_CONTACTS)`, :412).  The
// test now happens twice and can never fire the second time.
uint contactReduce6(float3 ptRaw[32], float sep[32], float3 normal, uint allMask, float clusterBias)
{
    return crsContactReduce(ptRaw, sep, normal, allMask, clusterBias,
                            crsConfig(true, false, int(SUBMANIFOLD_MAX_CONTACTS), false));
}

// nputils binarySearch: last index with data[idx] <= value (0 if none)
uint binarySearch32(uint data[32], uint value)
{
    uint left = 0u, right = 32u;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (data[pos] <= value) left = pos + 1u;
        else right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxCorrelate(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CorrIn s = gIn[t];

    CorrOut r;
    for (uint j = 0; j < 72; ++j) r.contacts[j] = float4(0, 0, 0, 0);
    // sentinel-init: CUDA's mNbContacts buffer holds stale data, so the
    // unconditional per-patch write (0 for empty patches) must be observable
    r.nbC = uint4(UNTOUCHED, UNTOUCHED, UNTOUCHED, UNTOUCHED);
    r.relQ = float4(0, 0, 0, 0); r.relP = float4(0, 0, 0, 0);
    r.misc = uint4(UNTOUCHED, UNTOUCHED, 0, 0);

    // local copies of the mutable buffers
    float depth[64];
    uint tc[64];
    for (uint j = 0; j < 64; ++j)
    {
        depth[j] = s.depth[j / 4][j & 3];
        tc[j] = 0u;
    }

    const uint count = s.hdr.x;
    const uint start = s.hdr.y;
    const uint cmIndex = s.hdr.z;
    const float4 aToBq = s.aToBq;
    const float3 aToBp = s.aToBp.xyz;
    const float clusterBias = s.aToBp.w;

    uint nbTriangles = 0;
    uint nicSlot[64]; // compacted nic view: nic[j] = original nic[nicSlot[j]]
    for (uint j = 0; j < 64; ++j) nicSlot[j] = j;

    if (count != CONVEX_TRIMESH_CACHED)
    {
        // ---- purgeEmptyTriangles (cu L58-87), strided loads of 32 ----
        float dNew[64];
        uint tcNew[64], srcNew[64];
        for (uint j = 0; j < 64; ++j) { dNew[j] = 0.0f; tcNew[j] = 0u; srcNew[j] = 0u; }

        for (uint loaded = 0; loaded < count; loaded += 32)
        {
            const uint nbToLoad = min(count - loaded, 32u);
            uint validBits = 0;
            for (uint L = 0; L < 32; ++L)
            {
                if (L >= nbToLoad) continue; // d = FLT_MAX -> never valid
                const float d = depth[start + loaded + L];
                const uint nb = getNbContactsNic(s.nicIdx[(start + loaded + L) / 4][(start + loaded + L) & 3]);
                if (d != PX_MAX_F32 && nb > 0u) validBits |= 1u << L;
            }
            for (uint L = 0; L < 32; ++L)
            {
                if ((validBits & (1u << L)) != 0u)
                {
                    const uint dest = countbits(validBits & ((1u << L) - 1u)) + nbTriangles;
                    const uint src = loaded + L;
                    const uint nb = getNbContactsNic(s.nicIdx[(start + src) / 4][(start + src) & 3]);
                    dNew[dest] = depth[start + src];
                    tcNew[dest] = (src << 4u) | nb;
                    srcNew[dest] = start + src;
                }
            }
            nbTriangles += countbits(validBits);
        }

        for (uint j = 0; j < nbTriangles; ++j)
        {
            depth[start + j] = dNew[j];
            tc[start + j] = tcNew[j];
            nicSlot[start + j] = srcNew[j];
        }

        if (nbTriangles == 0u)
        {
            r.misc.x = 0u; // outBuffer[pairIndex].mNbManifolds = 0 (pairIndex slot!)
        }
        else
        {
            // ---- patch loop (cu L157-325) ----
            uint nbPatches = 0;
            uint remainingTriangles = nbTriangles;

            for (; nbPatches < MULTIMANIFOLD_MAX_MANIFOLDS && remainingTriangles != 0u; nbPatches++)
            {
                // (1) deepest triangle: per-lane strided strict <, then lowest-lane exact min
                float laneBest[32];
                uint laneIdx[32];
                for (uint L = 0; L < 32; ++L) { laneBest[L] = PX_MAX_F32; laneIdx[L] = 0u; }
                for (uint i = 0; i < nbTriangles; ++i)
                {
                    const uint L = i & 31u;
                    if (tc[start + i] != 0u)
                    {
                        const float d = depth[start + i];
                        if (d < laneBest[L]) { laneBest[L] = d; laneIdx[L] = start + i; }
                    }
                }
                float minD = PX_MAX_F32;
                for (uint L = 0; L < 32; ++L) minD = min(minD, laneBest[L]);
                uint i0 = 0;
                for (uint L = 0; L < 32; ++L) if (laneBest[L] == minD) { i0 = L; break; }
                const uint bestIndex = laneIdx[i0];

                const float3 bestNormal = s.nrm[nicSlot[bestIndex]].xyz;

                uint currentContactMask = 0u;
                float3 pA[32];
                float sepL[32];
                uint triIdxL[32];
                bool hasContact[32];
                for (uint L = 0; L < 32; ++L) { pA[L] = float3(0, 0, 0); sepL[L] = 0.0f; triIdxL[L] = 0u; hasContact[L] = false; }

                for (uint i = 0; i < nbTriangles; i += 32)
                {
                    float3 nrmL[32];
                    uint triRunSums[32], triStartIndex[32], triInds[32];
                    uint nbCts[32];
                    for (uint L = 0; L < 32; ++L)
                    {
                        nrmL[L] = float3(0, 0, 0); triInds[L] = 0u; nbCts[L] = 0u; triStartIndex[L] = 0u;
                        if ((i + L) < nbTriangles)
                        {
                            const uint index = tc[start + i + L];
                            nbCts[L] = index & 7u;      // THREE bits (purge stored four)
                            triStartIndex[L] = index >> 4u;
                            if (nbCts[L] != 0u)
                            {
                                const uint slot = nicSlot[start + i + L];
                                nrmL[L] = s.nrm[slot].xyz;
                                triInds[L] = getTriangleIndexNic(s.nicIdx[slot / 4][slot & 3]);
                            }
                        }
                    }

                    uint acceptMask = 0u;
                    for (uint L = 0; L < 32; ++L)
                    {
                        const bool accept = dot(nrmL[L], bestNormal) > PATCH_ACCEPTANCE_EPS;
                        if (!accept) nbCts[L] = 0u;
                        else acceptMask |= 1u << L;
                    }
                    remainingTriangles -= countbits(acceptMask);

                    uint run = 0;
                    for (uint L = 0; L < 32; ++L)
                    {
                        triRunSums[L] = run; // warpScan inclusive minus own = exclusive
                        run += nbCts[L];
                    }
                    const uint totalContacts = run;

                    for (uint c = 0; c < totalContacts;)
                    {
                        const uint readMask = ~currentContactMask;
                        for (uint L = 0; L < 32; ++L)
                        {
                            const bool needsContact = (readMask & (1u << L)) != 0u;
                            const uint readIndex = c + countbits(readMask & ((1u << L) - 1u));
                            if (needsContact && readIndex < totalContacts)
                            {
                                const uint idx = binarySearch32(triRunSums, readIndex);
                                const uint offset = readIndex - triRunSums[idx];
                                const uint startIdx = triStartIndex[idx] + start;
                                const uint contactStartIdx = s.cbuf[startIdx / 4][startIdx & 3];
                                const float4 v = s.tempC[contactStartIdx + offset];
                                pA[L] = v.xyz;
                                sepL[L] = v.w;
                                triIdxL[L] = triInds[idx];
                                hasContact[L] = true;
                            }
                        }
                        uint newMask = 0u;
                        for (uint L = 0; L < 32; ++L) if (hasContact[L]) newMask |= 1u << L;
                        currentContactMask = newMask;

                        c += countbits(readMask);
                        if (countbits(currentContactMask) > SUBMANIFOLD_MAX_CONTACTS)
                            currentContactMask = contactReduce6(pA, sepL, bestNormal, currentContactMask, clusterBias);
                        for (uint L = 0; L < 32; ++L) hasContact[L] = (currentContactMask & (1u << L)) != 0u;
                    }

                    for (uint L = 0; L < 32; ++L)
                        if ((acceptMask & (1u << L)) != 0u)
                            tc[start + i + L] = 0u; // triangle done
                }

                // ---- output this patch (cmIndex slot) ----
                const uint nbContacts = countbits(currentContactMask);
                r.nbC[nbPatches] = nbContacts;
                for (uint L = 0; L < 32; ++L)
                {
                    if ((currentContactMask & (1u << L)) != 0u)
                    {
                        const uint slot = countbits(currentContactMask & ((1u << L) - 1u));
                        const float3 nor = quatRotate(aToBq, bestNormal);
                        const float3 pointB = quatRotate(aToBq, pA[L]) + aToBp + nor * sepL[L];
                        const uint base = (nbPatches * 6u + slot) * 3u;
                        r.contacts[base + 0] = float4(pA[L], sepL[L]);
                        r.contacts[base + 1] = float4(pointB, asfloat(triIdxL[L]));
                        r.contacts[base + 2] = float4(-nor, 0.0f);
                    }
                }
            }

            if (nbPatches > 0u)
            {
                r.relQ = aToBq;
                r.relP = float4(aToBp, 0.0f);
            }
            // outBuffer[pair.cmIndex].mNbManifolds = nbPatches
            if (cmIndex == 0u) r.misc.x = nbPatches; else r.misc.y = nbPatches;
            r.misc.w = remainingTriangles;
        }
    }

    r.misc.z = nbTriangles;
    for (uint j = 0; j < 16; ++j)
    {
        r.tempOut[j] = uint4(tc[j * 4], tc[j * 4 + 1], tc[j * 4 + 2], tc[j * 4 + 3]);
        r.depthOut[j] = float4(depth[j * 4], depth[j * 4 + 1], depth[j * 4 + 2], depth[j * 4 + 3]);
    }
    gOut[t] = r;
}
