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

#include <algorithm>
#include <cstdint>
#include <cstring>
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

// ===========================================================================
// THE OUTPUT TAIL: the reservation and the patch writer (P9.83b-8 T2)
// ===========================================================================
// createPatchWriterFullThreadBlock's decisions (:504-641) lived in
// testNpTriTriReduce.cpp, the header/point writer's combine and rotation in
// testNpTriTriWrite.cpp.  The block-tail fixture composes both, so they move here
// VERBATIM (static -> inline) rather than growing a third copy.

// cutil_math.h:765-788, transcribed: the CUDA reads the flag bytes through these.
inline uint32_t u32High(uint32_t v) { return (v >> 16) & 0xFFFFu; }
inline uint32_t u32Low(uint32_t v) { return v & ((1u << 16u) - 1u); }
inline uint32_t u16High(uint32_t v) { return (v >> 8) & 0xFFu; }
inline uint32_t u16Low(uint32_t v) { return v & ((1u << 8u) - 1u); }

static const uint32_t TTP_PATCH_SIZE = 64, TTP_CONTACT_SIZE = 16, TTP_FORCE_SIZE = 4;
static const uint32_t FLAG_HAS_NO_TOUCH = 1, FLAG_HAS_TOUCH = 2, FLAG_TOUCH_KNOWN = 3;
static const uint32_t OVF_PATCH = 4, OVF_CONTACT = 1, OVF_FORCE = 2;

struct RefRes
{
    uint32_t patchOfs, contactOfs, forceOfs, allFlags;
    uint32_t nPatches, nContacts, statusFlags, ovfBits;
    bool touchChange, patchChange, overflow, wPatchNull, wContactNull;
};

// createPatchWriterFullThreadBlock (:504-629) with the three atomic RESULTS as
// inputs, so the decisions are testable without a race.
//   - the checks are 64-bit and the reservations are 32-bit (:560, :567, :573);
//   - the force check tests 2N where the reservation added N (:558 vs :573);
//   - the zero-contact path leaves all three offsets at 0xFFFFFFFF with
//     `overflow` FALSE, so the output pointers become base + 0xFFFFFFFF while the
//     writer members become null (:607-611 vs :625-626).
inline RefRes refReserve(uint32_t oldAllFlags, uint32_t numPatchesToKeep, uint32_t totalNumContacts,
                         uint32_t patchAdd, uint32_t contactAdd, uint32_t forceAdd,
                         uint32_t patchLimit, uint32_t contactLimit, uint32_t forceLimit)
{
    RefRes r{};
    r.patchOfs = r.contactOfs = r.forceOfs = 0xFFFFFFFFu;
    r.ovfBits = 0; r.overflow = false;

    // :537 and :547, in the CUDA's own spelling.  P9.83b-8 T0: this oracle used to
    // read byte 3 (`>> 24`), agreeing with the port's defect instead of the CUDA.
    const uint32_t oldStatus = u16Low(u32High(oldAllFlags));
    const uint32_t prevPatches = u16High(u32Low(oldAllFlags));

    uint32_t statusFlags = oldStatus & (~FLAG_TOUCH_KNOWN);
    if (numPatchesToKeep != 0) statusFlags |= FLAG_HAS_TOUCH;
    else                       statusFlags |= FLAG_HAS_NO_TOUCH;

    uint32_t nP = numPatchesToKeep, nC = totalNumContacts;

    if (totalNumContacts != 0)                                       // :554
    {
        r.patchOfs = patchAdd; r.contactOfs = contactAdd; r.forceOfs = forceAdd;
        if ((uint64_t)r.patchOfs + (uint64_t)TTP_PATCH_SIZE * numPatchesToKeep > (uint64_t)patchLimit)
        { r.ovfBits |= OVF_PATCH; r.patchOfs = 0xFFFFFFFFu; r.overflow = true; }
        if ((uint64_t)r.contactOfs + (uint64_t)TTP_CONTACT_SIZE * totalNumContacts > (uint64_t)contactLimit)
        { r.ovfBits |= OVF_CONTACT; r.contactOfs = 0xFFFFFFFFu; r.overflow = true; }
        if ((uint64_t)r.forceOfs + (uint64_t)TTP_FORCE_SIZE * totalNumContacts * 2 > (uint64_t)forceLimit)
        { r.ovfBits |= OVF_FORCE; r.forceOfs = 0xFFFFFFFFu; r.overflow = true; }
        if (r.overflow)                                              // :580-586
        {
            nP = 0; nC = 0;
            statusFlags = (statusFlags & (~FLAG_TOUCH_KNOWN)) | FLAG_HAS_NO_TOUCH;
        }
    }

    const bool prevHadTouch = (oldStatus & FLAG_HAS_TOUCH) != 0;
    const bool prevKnown = (oldStatus & FLAG_TOUCH_KNOWN) != 0;
    const bool nowHasTouch = nP != 0;
    r.touchChange = (prevHadTouch != nowHasTouch) || (!prevKnown);
    r.patchChange = (prevPatches != nP);
    r.allFlags = (prevPatches << 24) | (statusFlags << 16) | (nP << 8);
    r.statusFlags = statusFlags;
    r.nPatches = nP; r.nContacts = nC;
    r.wPatchNull = (r.patchOfs == 0xFFFFFFFFu);
    r.wContactNull = (r.contactOfs == 0xFFFFFFFFu);
    return r;
}

// PxQuat::rotate (PxQuat.h:286-295), the expression pxgSlangCommon::quatRotate
// transcribes.  NOT npref::quatRotate, which uses the `w*w - u.u` spelling: the
// two agree algebraically on unit quaternions and not bitwise, and the oracle has
// to be written from the same upstream line the shader was.
inline V3 quatRotatePx(V4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}

// PxsCombineMaterials, transcribed for the header words.  Already the oracle in
// testNpInsertPatch.cpp against the same shader body (P9.83b-7 folded that fixture
// onto pxgSlangNpPatchTail), so a mistake here would show there first.
struct MaterialH
{
    float dynamicFriction, staticFriction, restitution, damping;
    uint32_t flagsModes0, modes1;
};
static_assert(sizeof(MaterialH) == 24, "PxsMaterialDataGpu is 24 bytes");

static const uint32_t MODE_MIN = 1, MODE_MULTIPLY = 2, MODE_MAX = 3;
static const uint32_t FLAG_DISABLE_FRICTION = 1, FLAG_DISABLE_STRONG_FRICTION = 2;
static const uint32_t FLAG_COMPLIANT_ACC_SPRING = 16;

inline float combineScalars(float a, float b, uint32_t mode)
{
    switch (mode)
    {
    case 0: return 0.5f * (a + b);
    case 1: return std::min(a, b);
    case 2: return a * b;
    case 3: return std::max(a, b);
    }
    return 0.f;
}

struct CombinedH { float sta, dyn, rest, damping; uint32_t flags; };

inline CombinedH combine(const MaterialH& p0, const MaterialH& p1)
{
    const uint32_t f0 = p0.flagsModes0 & 0xFFFFu, f1 = p1.flagsModes0 & 0xFFFFu;
    const uint32_t fric0 = (p0.flagsModes0 >> 16) & 0xFFu, fric1 = (p1.flagsModes0 >> 16) & 0xFFu;
    const uint32_t rest0 = (p0.flagsModes0 >> 24) & 0xFFu, rest1 = (p1.flagsModes0 >> 24) & 0xFFu;
    const uint32_t damp0 = p0.modes1 & 0xFFu, damp1 = p1.modes1 & 0xFFu;

    CombinedH c{};
    const float r0 = p0.restitution, r1 = p1.restitution;
    const bool c0 = r0 < 0.f, c1 = r1 < 0.f, one = c0 != c1, both = c0 && c1;
    const bool a0 = (f0 & FLAG_COMPLIANT_ACC_SPRING) != 0, a1 = (f1 & FLAG_COMPLIANT_ACC_SPRING) != 0;
    const bool oneAcc = a0 != a1;

    if (both && oneAcc) c.rest = a0 ? r0 : r1;
    else
    {
        const uint32_t m = one ? MODE_MIN : std::max(rest0, rest1);
        const float f = (both && m == MODE_MULTIPLY) ? -1.f : 1.f;
        c.rest = f * combineScalars(r0, r1, m);
    }
    if (both && oneAcc) c.damping = a0 ? p0.damping : p1.damping;
    else
    {
        const uint32_t m = one ? MODE_MAX : std::max(damp0, damp1);
        c.damping = combineScalars(p0.damping, p1.damping, m);
    }
    const uint32_t cf = f0 | f1;
    if ((cf & FLAG_DISABLE_FRICTION) == 0)
    {
        const uint32_t fm = std::max(fric0, fric1);
        const float d = combineScalars(p0.dynamicFriction, p1.dynamicFriction, fm);
        const float s = combineScalars(p0.staticFriction, p1.staticFriction, fm);
        const float fd = std::max(d, 0.f);
        c.dyn = fd;
        c.sta = (s - fd) >= 0.f ? s : fd;
        c.flags = cf;
    }
    else { c.flags = cf | FLAG_DISABLE_STRONG_FRICTION; c.dyn = 0.f; c.sta = 0.f; }
    return c;
}

inline uint32_t bitsOfF(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// writeContactPatch (:649-672) through writePatchesAndContacts' :727-734: the
// fourteen words one header receives.  Words 14 and 15 (pad[1..4]) are never
// written.  mi0/mi1 are the LOCAL shapes' indices; the :729 swap happens here.
inline void refPatchHeader(const MaterialH* mats, uint32_t mi0, uint32_t mi1, bool isFlipped,
                           V4 q, V3 nLocal, uint32_t pointCount, uint32_t startIndex,
                           uint32_t word[14])
{
    const uint32_t emi0 = isFlipped ? mi1 : mi0;
    const uint32_t emi1 = isFlipped ? mi0 : mi1;
    const CombinedH cm = combine(mats[emi0], mats[emi1]);

    V3 nWorld = quatRotatePx(q, nLocal);
    if (!isFlipped) nWorld = { -nWorld.x, -nWorld.y, -nWorld.z };

    const uint32_t one = bitsOfF(1.0f);
    word[0] = one; word[1] = one; word[2] = one; word[3] = one;
    word[4] = bitsOfF(nWorld.x);
    word[5] = bitsOfF(nWorld.y);
    word[6] = bitsOfF(nWorld.z);
    word[7] = bitsOfF(cm.rest);
    word[8] = bitsOfF(cm.dyn);
    word[9] = bitsOfF(cm.sta);
    word[10] = bitsOfF(cm.damping);
    word[11] = (startIndex & 0xFFFFu) | ((pointCount & 0xFFu) << 16) | ((cm.flags & 0xFFu) << 24);
    word[12] = (emi0 & 0xFFFFu) << 16;
    word[13] = (emi1 & 0xFFFFu);
}

// :745-748: a point is rotated AND translated; the separation rides through.
inline V4 refPatchPoint(V4 q, V3 p, V4 local)
{
    const V3 pw = quatRotatePx(q, V3{ local.x, local.y, local.z });
    return V4{ pw.x + p.x, pw.y + p.y, pw.z + p.z, local.w };
}

} // namespace tritriref
} // namespace pxgtest

#endif
