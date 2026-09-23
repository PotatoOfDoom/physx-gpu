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

// pxgTriTriRef.h -- host mirrors of the mesh-mesh PATCH machinery of
// doTriangleTriangleCollision (convexMeshMidphase.cu), written FROM THE CUDA.
//
// P9.83b-5 (15/n) gathered these here instead of letting a third fixture grow a
// third copy: refReduce and refMinIndex lived in testNpTriTriReduce.cpp, and
// testNpTriTriBlock.cpp carried its own minIndex.  They move VERBATIM (static ->
// inline); the block-level mirror of tryToAssignContactToExistingPatch and
// createNewPatches is new and is built from them.
//
// THE BLOCK MIRROR IS SEQUENTIAL, and the CUDA is not -- so the only thing it may
// depend on is what the CUDA's barriers make deterministic.  Every value that
// crosses warps in the CUDA crosses a barrier first (bar.sync 1-5, :813, :838,
// :944, :953 and the ones inside the block collectives), so the mirror computes
// each phase for all 1024 threads before the next phase reads any of it.

#ifndef PXG_TRITRI_REF_H
#define PXG_TRITRI_REF_H

#include "pxgCrsRef.h"

#include <cstdint>
#include <vector>

namespace pxgtest
{
namespace tritriref
{
using namespace pxgtest::crsref;

static const float PATCH_ACCEPTANCE_EPS2 = 0.85f;     // convexMeshMidphase.cu:422
static const float PATCH_ACCEPTANCE_EPS3 = 0.9995f;   // :423
static const uint32_t MAX_PATCHES = 128;              // MAX_MESH_MESH_PATCHES2
static const uint32_t TT_NO_INDEX = 0xFFFFFFFFu;   // lowestSetIndex(0)

struct RefResult
{
    V4 keepPoints[6];
    uint32_t pointCount;
    float deepest;
    uint32_t survivorMask;
    uint32_t passes;
};

// reduceContactsSingleWarp (convexMeshMidphase.cu:440-496), serialised.
inline RefResult refReduce(const V4* tempPoints, uint32_t totalContacts, V3 nor, float clusterBias)
{
    V3 pA[32];
    float sep[32];
    for (uint32_t j = 0; j < 32; ++j) { pA[j] = { 0, 0, 0 }; sep[j] = FLT_MAX_LOCAL; }

    uint32_t currentContactMask = 0, hasMask = 0;
    const uint32_t maxCounter = 100;
    uint32_t counter2 = 0;

    for (uint32_t c = 0; c < totalContacts && counter2 < maxCounter; )
    {
        ++counter2;                                        // :455
        const uint32_t readMask = ~currentContactMask;     // :457
        for (uint32_t l = 0; l < 32; ++l)
        {
            const bool needsContact = ((readMask >> l) & 1u) != 0u;   // :458, both arms
            const uint32_t readIndex = c + popcount32(readMask & ((1u << l) - 1u));  // :459
            if (needsContact && readIndex < totalContacts)  // :461
            {
                pA[l] = { tempPoints[readIndex].x, tempPoints[readIndex].y, tempPoints[readIndex].z };
                sep[l] = tempPoints[readIndex].w;
                hasMask |= 1u << l;                        // :466
            }
        }
        c += popcount32(readMask);                         // :469, LANES OFFERED
        currentContactMask = hasMask;                      // :471
        currentContactMask = refWrapper2(pA, sep, nor, currentContactMask, clusterBias); // :472
        hasMask = currentContactMask;                      // :473
    }

    RefResult r{};
    r.passes = counter2;

    float deepest = FLT_MAX_LOCAL;                         // :477-478
    for (uint32_t j = 0; j < 32; ++j)
        if (((currentContactMask >> j) & 1u) != 0u && sep[j] < deepest) deepest = sep[j];
    r.deepest = deepest;

    r.pointCount = popcount32(currentContactMask);         // :481
    r.survivorMask = currentContactMask;
    for (uint32_t k = 0; k < 6; ++k) r.keepPoints[k] = { 0, 0, 0, 0 };

    for (uint32_t l = 0; l < 32; ++l)                      // :491-495
    {
        if (((currentContactMask >> l) & 1u) == 0u) continue;
        const uint32_t off = popcount32(currentContactMask & ((1u << l) - 1u));
        if (off < 6) r.keepPoints[off] = { pA[l].x, pA[l].y, pA[l].z, sep[l] };
    }
    return r;
}

// minIndex (nputils.cuh:151-164).  Written from the CUDA: the ballot compares
// against the lane's ORIGINAL value and the mask is applied AFTER it, so an
// unmasked lane equal to the masked minimum sets a bit only `& mask` removes.
// lowestSetIndex(0) = __ffs(0) - 1 = 0xFFFFFFFF.
inline uint32_t refMinIndex(const float* v, uint32_t mask, float& minV)
{
    minV = FLT_MAX_LOCAL;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1u) != 0u && v[j] < minV) minV = v[j];
    uint32_t ballot = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (v[j] == minV) ballot |= 1u << j;
    ballot &= mask;
    if (ballot == 0u) return TT_NO_INDEX;
    uint32_t i = 0; while (((ballot >> i) & 1u) == 0u) ++i;
    return i;
}

// ---------------------------------------------------------------------------
// The block mirror
// ---------------------------------------------------------------------------

// struct Contact, convexMeshMidphase.cu:753-766
struct RefContact
{
    V3 pos;
    float separation;
    V3 normal;
    bool candidate;
};

// markAsProcessed (:760-765): pos is NOT reset.
inline void refMarkProcessed(RefContact& c)
{
    c.candidate = false;
    c.separation = FLT_MAX_LOCAL;
    c.normal = { 0, 0, 0 };
}

// Patch (:432-438)
struct RefPatch
{
    V4 keepPoints[6];
    V3 normal;
    float deepest;
    uint32_t pointCount;
};

// reduceContactsSingleWarp's three writes into patches[index] (:479, :486-489,
// :491-495).  Only the SURVIVING KeepPoints are written; the slots behind them
// keep whatever they held, and so does Normal.
inline void refStoreReduce(RefPatch& patch, const RefResult& r)
{
    patch.deepest = r.deepest;
    patch.pointCount = r.pointCount;
    for (uint32_t k = 0; k < r.pointCount && k < 6; ++k)
        patch.keepPoints[k] = r.keepPoints[k];
}

// threadBlockScanExclusive<32> over 1024 0/1 contributions (reduction.cuh:598-621)
inline uint32_t refBlockScan(const bool* emits, uint32_t* offset)
{
    uint32_t run = 0;
    for (uint32_t t = 0; t < 1024; ++t) { offset[t] = run; run += emits[t] ? 1u : 0u; }
    return run;
}

// tryToAssignContactToExistingPatch (:768-858)
inline void refTryAssignExisting(RefContact* c, RefPatch* patches, uint32_t p, float clusterBias,
                                 int& remainingContacts)
{
    RefPatch& patch = patches[p];
    bool accept[1024], accept2[1024];
    uint32_t reduceMask[32];
    bool anyKeep = false;
    for (uint32_t t = 0; t < 1024; ++t)
    {
        const float proj = dot3(c[t].normal, patch.normal);           // :778
        accept2[t] = proj > PATCH_ACCEPTANCE_EPS2;                     // :781, UNGATED
        accept[t] = proj > PATCH_ACCEPTANCE_EPS3;                      // :782
    }
    for (uint32_t w = 0; w < 32; ++w)
    {
        reduceMask[w] = 0;
        for (uint32_t l = 0; l < 32; ++l)
            if (accept[w * 32 + l]) reduceMask[w] |= 1u << l;          // :788
        if (reduceMask[w] != 0u) anyKeep = true;                       // :790-793
    }
    if (!anyKeep)                                                      // :808
        return;

    for (uint32_t w = 0; w < 32; ++w)                                  // :810, per warp
    {
        V3 pts[32];
        float seps[32];
        for (uint32_t l = 0; l < 32; ++l) { pts[l] = c[w * 32 + l].pos; seps[l] = c[w * 32 + l].separation; }
        reduceMask[w] = refWrapper2(pts, seps, patch.normal, reduceMask[w], clusterBias);
    }

    uint32_t nbUsed = 0;                                               // :812
    for (uint32_t t = 0; t < 1024; ++t) nbUsed += accept2[t] ? 1u : 0u;
    remainingContacts -= int(nbUsed);                                  // :815

    bool emits[1024];
    uint32_t offset[1024];
    for (uint32_t t = 0; t < 1024; ++t) emits[t] = ((reduceMask[t >> 5] >> (t & 31u)) & 1u) != 0u;
    uint32_t totalContacts = refBlockScan(emits, offset);             // :820

    std::vector<V4> temp(198, V4{ 0, 0, 0, 0 });
    for (uint32_t k = 0; k < patch.pointCount; ++k) temp[k] = patch.keepPoints[k];   // :828-829
    for (uint32_t t = 0; t < 1024; ++t)
        if (emits[t])
            temp[offset[t] + patch.pointCount] = V4{ c[t].pos.x, c[t].pos.y, c[t].pos.z, c[t].separation };  // :832-835
    totalContacts += patch.pointCount;                                 // :837

    const RefResult r = refReduce(temp.data(), totalContacts, patch.normal, clusterBias);   // :843
    refStoreReduce(patch, r);

    for (uint32_t t = 0; t < 1024; ++t)
        if (accept2[t]) refMarkProcessed(c[t]);                        // :845-846
}

// createNewPatches (:860-970)
inline void refCreateNewPatches(RefContact* c, RefPatch* patches, float clusterBias,
                                int remainingContacts, uint32_t& numPatchesToKeep)
{
    bool hasAnyContacts = remainingContacts > 0;                       // :872
    while (hasAnyContacts && numPatchesToKeep < MAX_PATCHES)           // :874
    {
        hasAnyContacts = false;

        float shMinSep[32];
        uint32_t counters[32], minI[32];
        for (uint32_t w = 0; w < 32; ++w)                              // :879-891
        {
            float sep[32];
            uint32_t candidateMask = 0;
            for (uint32_t l = 0; l < 32; ++l)
            {
                sep[l] = c[w * 32 + l].separation;
                if (c[w * 32 + l].candidate) candidateMask |= 1u << l;
            }
            counters[w] = popcount32(candidateMask);
            minI[w] = refMinIndex(sep, candidateMask, shMinSep[w]);
        }
        float blockMin;
        const uint32_t minIBlock = refMinIndex(shMinSep, 0xFFFFFFFFu, blockMin);   // :897-899

        if (counters[minIBlock] == 0u)                                 // :902
            continue;   // hasAnyContacts stays false: the loop ends

        hasAnyContacts = true;
        RefPatch& patch = patches[numPatchesToKeep];
        const uint32_t seed = minIBlock * 32u + minI[minIBlock];
        patch.normal = c[seed].normal;                                 // :908-912

        bool accept[1024], accept2[1024];
        for (uint32_t t = 0; t < 1024; ++t)                            // :919-926
        {
            accept[t] = accept2[t] = false;
            if (c[t].candidate)
            {
                const float proj = dot3(c[t].normal, patch.normal);
                accept2[t] = proj > PATCH_ACCEPTANCE_EPS2;
                accept[t] = proj > PATCH_ACCEPTANCE_EPS3 || t == seed;
            }
        }

        uint32_t reduceMask[32];
        for (uint32_t w = 0; w < 32; ++w)                              // :930-931
        {
            reduceMask[w] = 0;
            V3 pts[32];
            float seps[32];
            for (uint32_t l = 0; l < 32; ++l)
            {
                if (accept[w * 32 + l]) reduceMask[w] |= 1u << l;
                pts[l] = c[w * 32 + l].pos;
                seps[l] = c[w * 32 + l].separation;
            }
            reduceMask[w] = refWrapper2(pts, seps, patch.normal, reduceMask[w], clusterBias);
        }

        bool emits[1024];
        uint32_t offset[1024];
        for (uint32_t t = 0; t < 1024; ++t) emits[t] = ((reduceMask[t >> 5] >> (t & 31u)) & 1u) != 0u;
        const uint32_t totalContacts = refBlockScan(emits, offset);   // :936

        if (totalContacts != 0u)                                       // :939
        {
            std::vector<V4> temp(198, V4{ 0, 0, 0, 0 });
            for (uint32_t t = 0; t < 1024; ++t)
                if (emits[t])
                    temp[offset[t]] = V4{ c[t].pos.x, c[t].pos.y, c[t].pos.z, c[t].separation };  // :941-942
            const RefResult r = refReduce(temp.data(), totalContacts, patch.normal, clusterBias);   // :951
            refStoreReduce(patch, r);
            if (patch.pointCount != 0u)                                // :955-956
                ++numPatchesToKeep;
        }

        for (uint32_t t = 0; t < 1024; ++t)
            if (accept2[t]) refMarkProcessed(c[t]);                    // :959-960
    }
}

// :1207-1220
inline int refAssignContacts(RefContact* c, RefPatch* patches, float clusterBias, int generatedThisIter,
                             uint32_t& numPatchesToKeep)
{
    int remainingContacts = generatedThisIter;
    for (uint32_t p = 0; p < numPatchesToKeep && remainingContacts > 0; ++p)
        refTryAssignExisting(c, patches, p, clusterBias, remainingContacts);
    refCreateNewPatches(c, patches, clusterBias, remainingContacts, numPatchesToKeep);
    return remainingContacts;
}

} // namespace tritriref
} // namespace pxgtest

#endif
