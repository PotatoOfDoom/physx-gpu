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

// Verification of the tail of writePatchesAndContacts (convexMeshMidphase.cu
// :673-751): ttpPatchStartIndices and ttpWritePatchAndContacts.
//
// THE STREAMS ARE CHECKED WHOLE, NOT AT THE WRITTEN WORDS.  Both buffers start
// filled with a poison word and every byte is compared: a word the mirror says is
// untouched must still be poison.  Checking only the addresses the mirror expects
// would pass a port that wrote twice as much -- and the two most plausible bugs
// here (a point loop that runs to MESH_MESH_CONTACT_LIMIT instead of PointCount,
// a header write that ignores its sentinel) are exactly that shape.
//
// THE MATERIAL COMBINE IS SYMMETRIC, SO THE SWAP SHOWS IN TWO WORDS ONLY.
// Every branch of PxsCombineMaterials is symmetric in its two arguments -- min,
// max, average and multiply all are, and the two compliant-acc-spring picks
// select by flag rather than by position.  So words 7..11 of the header are
// identical whichever way :729 orders the pair, and materialIndex0/1 (words 12
// and 13) are the ONLY observable of the swap.  That is stated here because it
// decides what kills the `swap-dropped` mutant: nothing but those two words can.

#include "pxgTestFramework.h"
#include "pxgTriTriRef.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

using namespace pxgtest::tritriref;   // quatRotatePx, MaterialH, combine -- P9.83b-8 T2

static uint32_t bits(float f) { return bitsOfF(f); }

// ---------------------------------------------------------------------------
// The case record, 704 bytes, matching npTriTriWrite.slang's explicit offsets.
// ---------------------------------------------------------------------------
struct TtwCaseH
{
    float keepPoints[6][4];   // @0
    float normal[4];          // @96  xyz = patch normal, w = deepest
    float rotQ[4];            // @112
    float rotP[4];            // @128 xyz = translation
    uint32_t a[4];            // @144 pointCount, patchIndex, patchStartIndex, isFlipped
    uint32_t b[4];            // @160 patchByteOffset, contactByteOffset, mi0, mi1
    uint32_t c[4];            // @176 numPatchesToKeep
    uint32_t counts[128];     // @192
};
static_assert(sizeof(TtwCaseH) == 704, "case stride");

struct TtwPushH
{
    uint64_t cases, patchStream, contactStream, materials, scanOut;
    uint32_t numCases, pad;
};

const uint32_t POISON = 0xCDCDCDCDu;
const uint32_t SENTINEL = 0xFFFFFFFFu;
const uint32_t PATCH_SLOTS_PER_CASE = 2;    // 128 bytes
const uint32_t CONTACT_SLOTS_PER_CASE = 16; // 256 bytes

} // namespace

// ---------------------------------------------------------------------------
// The scan (convexMeshMidphase.cu:698-705).
// ---------------------------------------------------------------------------
PXG_TEST(npTriTriWrite_scan)
{
    std::mt19937 rng(90210);
    const uint32_t N = 2000;

    std::vector<TtwCaseH> in(N);
    std::memset(in.data(), 0, in.size() * sizeof(TtwCaseH));

    std::vector<std::vector<uint32_t>> refStart(N, std::vector<uint32_t>(128));
    std::vector<uint32_t> refTotal(N);

    size_t sawZeroPatches = 0, sawFull = 0, sawZeroCount = 0, sawMaxCount = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // One case in ten keeps nothing -- the shape createNewPatches produces
        // when every candidate was rejected, and the one that makes totalSum 0.
        uint32_t nKeep;
        if (i % 10 == 0) nKeep = 0;
        else if (i % 10 == 1) nKeep = 128;
        else nKeep = 1 + (rng() % 128);
        in[i].c[0] = nKeep;
        if (nKeep == 0) ++sawZeroPatches;
        if (nKeep == 128) ++sawFull;

        for (uint32_t j = 0; j < 128; ++j)
        {
            // Counts past nKeep are DELIBERATELY non-zero.  The CUDA zeroes them
            // by never loading them (:698-700); a port that scanned the raw array
            // would agree with a mirror that also scanned the raw array.
            const uint32_t v = rng() % 7;
            in[i].counts[j] = v;
            if (j < nKeep && v == 0) ++sawZeroCount;
            if (j < nKeep && v == 6) ++sawMaxCount;
        }

        uint32_t run = 0;
        for (uint32_t j = 0; j < 128; ++j)
        {
            refStart[i][j] = run;
            if (j < nKeep) run += in[i].counts[j];
        }
        refTotal[i] = run;
    }

    PXG_CHECK_MSG(sawZeroPatches > 100 && sawFull > 100,
                  "npTriTriWrite_scan: %zu empty and %zu full cases -- both ends of "
                  "numPatchesToKeep are needed", sawZeroPatches, sawFull);
    PXG_CHECK_MSG(sawZeroCount > 1000 && sawMaxCount > 1000,
                  "npTriTriWrite_scan: %zu zero and %zu six-point live patches; an "
                  "empty patch is what makes the start index stall",
                  sawZeroCount, sawMaxCount);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> scanInit(size_t(N) * 132, POISON);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bScan = createBufferFrom(dev, scanInit);

    TtwPushH push{};
    push.cases = dev->getBufferDeviceAddress(bIn);
    push.scanOut = dev->getBufferDeviceAddress(bScan);
    push.numCases = N;

    Kernel k("npTriTriWrite.spv", "ttrScan", {}, sizeof(TtwPushH));
    k.dispatch((N + 63) / 64, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<uint32_t>(dev, bScan, scanInit.size());

    size_t mmStart = 0, mmTotal = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t* g = &got[size_t(i) * 132];
        for (uint32_t j = 0; j < 128; ++j)
            if (g[j] != refStart[i][j])
            {
                if (mmStart < 4)
                    std::printf("  case %u patch %u: start got %u want %u (nKeep %u)\n",
                                i, j, g[j], refStart[i][j], in[i].c[0]);
                ++mmStart;
            }
        if (g[128] != refTotal[i])
        {
            if (mmTotal < 4)
                std::printf("  case %u: total got %u want %u\n", i, g[128], refTotal[i]);
            ++mmTotal;
        }
    }

    PXG_CHECK_MSG(mmStart == 0, "npTriTriWrite_scan: %zu start-index mismatches", mmStart);
    PXG_CHECK_MSG(mmTotal == 0, "npTriTriWrite_scan: %zu total mismatches", mmTotal);

    std::printf("  npTriTriWrite_scan: %u cases, %zu empty / %zu full, "
                "%zu zero-count and %zu six-count live patches\n",
                N, sawZeroPatches, sawFull, sawZeroCount, sawMaxCount);

    for (PxgBufferHandle h : { bIn, bScan }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// The header and point writes (:727-750).
// ---------------------------------------------------------------------------
PXG_TEST(npTriTriWrite_patch)
{
    std::mt19937 rng(90211);
    const uint32_t N = 3000;
    const uint32_t nbMaterials = 24;

    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<MaterialH> mats(nbMaterials);
    for (uint32_t i = 0; i < nbMaterials; ++i)
    {
        mats[i].dynamicFriction = rf(-0.2f, 1.5f);
        mats[i].staticFriction = rf(-0.2f, 1.5f);
        // A third of the materials carry a NEGATIVE restitution, which is what
        // selects the compliant-contact arm of the combine.
        mats[i].restitution = (i % 3 == 0) ? rf(-4.0f, -0.01f) : rf(0.f, 1.f);
        mats[i].damping = rf(0.f, 40.f);
        const uint32_t flags = ((i % 5 == 0) ? FLAG_DISABLE_FRICTION : 0u)
                             | ((i % 4 == 0) ? FLAG_COMPLIANT_ACC_SPRING : 0u);
        mats[i].flagsModes0 = flags | ((i % 4) << 16) | (((i + 1) % 4) << 24);
        mats[i].modes1 = (i + 2) % 4;
    }
    // P9.88: a second bank at [0x10000 + i] with DIFFERENT materials, so an index k | 0x10000
    // tells a lookup that narrows to PxU16 first (materialCombiner.cuh:35) from one that does
    // not, and both stay in bounds.
    mats.resize(0x10000u + nbMaterials);
    for (uint32_t i = 0; i < nbMaterials; ++i)
    {
        MaterialH& m = mats[0x10000u + i];
        m.dynamicFriction = rf(-0.2f, 1.5f);
        m.staticFriction = rf(-0.2f, 1.5f);
        m.restitution = (i % 3 == 1) ? rf(-4.0f, -0.01f) : rf(0.f, 1.f);
        m.damping = rf(0.f, 40.f);
        const uint32_t flags = ((i % 5 == 2) ? FLAG_DISABLE_FRICTION : 0u)
                             | ((i % 4 == 1) ? FLAG_COMPLIANT_ACC_SPRING : 0u);
        m.flagsModes0 = flags | (((i + 2) % 4) << 16) | (((i + 3) % 4) << 24);
        m.modes1 = (i + 1) % 4;
    }

    std::vector<TtwCaseH> in(N);
    std::memset(in.data(), 0, in.size() * sizeof(TtwCaseH));

    struct RefH
    {
        bool headerWritten;
        uint32_t headerWord[16];
        uint32_t headerBase;        // word index into the patch stream
        bool pointsWritten;
        uint32_t nPoints;
        uint32_t pointBase;         // float index into the contact stream
        uint32_t pointStart;        // writeIndex of the first point
        float point[6][4];
    };
    std::vector<RefH> ref(N);

    size_t sawPatchSentinelOnly = 0, sawContactSentinelOnly = 0, sawBothSentinel = 0;
    size_t sawFlipped = 0, sawUnflipped = 0, sawZeroPoints = 0, sawSixPoints = 0;
    size_t sawStartOffGrid = 0, sawSameMaterial = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TtwCaseH& c = in[i];

        const uint32_t pointCount = rng() % 7;                 // 0..6
        const uint32_t patchIndex = rng() % PATCH_SLOTS_PER_CASE;
        // NOT patchIndex * 6: the start index comes from a scan of real counts, and
        // a port that recomputed it from the index would agree on every full patch.
        const uint32_t patchStartIndex = rng() % 10;
        const bool isFlipped = (rng() & 1) != 0;

        // Every combination of the two independent sentinels, by construction.
        const uint32_t mode = i % 8;   // 0..4 both live, 5 patch dead, 6 contact dead, 7 both
        const bool patchDead = (mode == 5 || mode == 7);
        const bool contactDead = (mode == 6 || mode == 7);
        if (patchDead && !contactDead) ++sawPatchSentinelOnly;
        if (contactDead && !patchDead) ++sawContactSentinelOnly;
        if (patchDead && contactDead) ++sawBothSentinel;

        const uint32_t patchByteOffset = patchDead ? SENTINEL : i * PATCH_SLOTS_PER_CASE * 64u;
        const uint32_t contactByteOffset = contactDead ? SENTINEL : i * CONTACT_SLOTS_PER_CASE * 16u;

        uint32_t mi0 = rng() % nbMaterials;
        uint32_t mi1 = rng() % nbMaterials;
        // One case in twelve uses the SAME material on both sides: the swap is
        // invisible there, so those cases are the control, not the evidence.
        if (rng() % 4 == 0) mi0 |= 0x10000u;   // P9.88: second bank
        if (rng() % 4 == 0) mi1 |= 0x10000u;
        if (i % 12 == 0) { mi1 = mi0; ++sawSameMaterial; }

        c.a[0] = pointCount; c.a[1] = patchIndex; c.a[2] = patchStartIndex;
        c.a[3] = isFlipped ? 1u : 0u;
        c.b[0] = patchByteOffset; c.b[1] = contactByteOffset; c.b[2] = mi0; c.b[3] = mi1;

        if (isFlipped) ++sawFlipped; else ++sawUnflipped;
        if (pointCount == 0) ++sawZeroPoints;
        if (pointCount == 6) ++sawSixPoints;
        if (patchStartIndex % 6 != 0) ++sawStartOffGrid;

        // A unit quaternion that is NOT near identity, and a translation that is
        // not zero -- :732 rotates only, :745 rotates and translates.
        const float ang = rf(0.4f, 2.8f);
        V3 axis = { rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) };
        float al = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
        if (al < 1e-4f) { axis = { 0.f, 0.f, 1.f }; al = 1.f; }
        const float s = std::sin(ang * 0.5f) / al;
        const V4 q = { axis.x * s, axis.y * s, axis.z * s, std::cos(ang * 0.5f) };
        const V3 tp = { rf(-9.f, 9.f), rf(-9.f, 9.f), rf(-9.f, 9.f) };
        c.rotQ[0] = q.x; c.rotQ[1] = q.y; c.rotQ[2] = q.z; c.rotQ[3] = q.w;
        c.rotP[0] = tp.x; c.rotP[1] = tp.y; c.rotP[2] = tp.z; c.rotP[3] = 0.f;

        const V3 nLocal = { rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) };
        c.normal[0] = nLocal.x; c.normal[1] = nLocal.y; c.normal[2] = nLocal.z;
        c.normal[3] = rf(-1.f, 0.f);   // deepest, unread by this body

        for (uint32_t k = 0; k < 6; ++k)
        {
            c.keepPoints[k][0] = rf(-6.f, 6.f);
            c.keepPoints[k][1] = rf(-6.f, 6.f);
            c.keepPoints[k][2] = rf(-6.f, 6.f);
            c.keepPoints[k][3] = rf(-0.6f, 0.15f);   // separation
        }

        // ---- the mirror ----
        RefH& r = ref[i];
        r.headerWritten = !patchDead;
        r.headerBase = (patchByteOffset / 4u) + patchIndex * 16u;
        r.pointsWritten = !contactDead && pointCount > 0;
        r.nPoints = pointCount;
        r.pointBase = contactDead ? 0u : (contactByteOffset / 4u);
        r.pointStart = patchStartIndex;

        // :729 -- the swap that cancels the kernel's argument exchange.
        const uint32_t emi0 = isFlipped ? mi1 : mi0;
        const uint32_t emi1 = isFlipped ? mi0 : mi1;
        const CombinedH cm = combine(mats[emi0 & 0xFFFFu], mats[emi1 & 0xFFFFu]);   // P9.88

        V3 nWorld = quatRotatePx(q, nLocal);
        if (!isFlipped) nWorld = { -nWorld.x, -nWorld.y, -nWorld.z };

        const uint32_t one = bits(1.0f);
        r.headerWord[0] = one; r.headerWord[1] = one; r.headerWord[2] = one; r.headerWord[3] = one;
        r.headerWord[4] = bits(nWorld.x);
        r.headerWord[5] = bits(nWorld.y);
        r.headerWord[6] = bits(nWorld.z);
        r.headerWord[7] = bits(cm.rest);
        r.headerWord[8] = bits(cm.dyn);
        r.headerWord[9] = bits(cm.sta);
        r.headerWord[10] = bits(cm.damping);
        r.headerWord[11] = (patchStartIndex & 0xFFFFu) | ((pointCount & 0xFFu) << 16)
                         | ((cm.flags & 0xFFu) << 24);
        r.headerWord[12] = (emi0 & 0xFFFFu) << 16;
        // :663 stores materialIndex1 as a PxU16: pad[0] (bits 16-31) keeps the stream's word.
        r.headerWord[13] = (POISON & 0xFFFF0000u) | (emi1 & 0xFFFFu);
        r.headerWord[14] = POISON;   // pad[1..4] -- NEVER written
        r.headerWord[15] = POISON;

        for (uint32_t k = 0; k < pointCount; ++k)
        {
            const V3 pl = { c.keepPoints[k][0], c.keepPoints[k][1], c.keepPoints[k][2] };
            const V3 pw = quatRotatePx(q, pl);
            r.point[k][0] = pw.x + tp.x;
            r.point[k][1] = pw.y + tp.y;
            r.point[k][2] = pw.z + tp.z;
            r.point[k][3] = c.keepPoints[k][3];
        }
    }

    PXG_CHECK_MSG(sawPatchSentinelOnly > 200 && sawContactSentinelOnly > 200 && sawBothSentinel > 200,
                  "npTriTriWrite_patch: %zu patch-only, %zu contact-only and %zu combined "
                  "sentinel cases -- the two guards are independent in the CUDA "
                  "(:651 vs :746) and a case set where they always agree cannot see it",
                  sawPatchSentinelOnly, sawContactSentinelOnly, sawBothSentinel);
    PXG_CHECK_MSG(sawFlipped > N / 4 && sawUnflipped > N / 4,
                  "npTriTriWrite_patch: %zu flipped / %zu unflipped", sawFlipped, sawUnflipped);
    PXG_CHECK_MSG(sawZeroPoints > 100 && sawSixPoints > 100,
                  "npTriTriWrite_patch: %zu zero-point and %zu six-point patches; the "
                  "zero-point header is what createNewPatches:955 can emit",
                  sawZeroPoints, sawSixPoints);
    PXG_CHECK_MSG(sawStartOffGrid > N / 2,
                  "npTriTriWrite_patch: only %zu cases have a start index that is not a "
                  "multiple of six -- the scan is over real counts, not over the index",
                  sawStartOffGrid);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> patchInit(size_t(N) * PATCH_SLOTS_PER_CASE * 16, POISON);
    std::vector<uint32_t> contactInit(size_t(N) * CONTACT_SLOTS_PER_CASE * 4, POISON);

    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    PxgBufferHandle bPatch = createBufferFrom(dev, patchInit);
    PxgBufferHandle bContact = createBufferFrom(dev, contactInit);

    TtwPushH push{};
    push.cases = dev->getBufferDeviceAddress(bIn);
    push.patchStream = dev->getBufferDeviceAddress(bPatch);
    push.contactStream = dev->getBufferDeviceAddress(bContact);
    push.materials = dev->getBufferDeviceAddress(bMat);
    push.scanOut = dev->getBufferDeviceAddress(bPatch);   // unused by ttrWrite
    push.numCases = N;

    Kernel k("npTriTriWrite.spv", "ttrWrite", {}, sizeof(TtwPushH));
    k.dispatch((N + 63) / 64, {}, push, 1u, 1u);
    dev->flush();

    const auto gotPatch = downloadBuffer<uint32_t>(dev, bPatch, patchInit.size());
    const auto gotContact = downloadBuffer<uint32_t>(dev, bContact, contactInit.size());

    // The expected image of BOTH streams, poison everywhere the mirror says nothing
    // was written.  Compared word for word, so a stray write is a failure.
    std::vector<uint32_t> wantPatch(patchInit.size(), POISON);
    std::vector<uint32_t> wantContact(contactInit.size(), POISON);
    for (uint32_t i = 0; i < N; ++i)
    {
        const RefH& r = ref[i];
        if (r.headerWritten)
            for (uint32_t w = 0; w < 14; ++w)     // 14 and 15 stay poison
                wantPatch[r.headerBase + w] = r.headerWord[w];
        if (r.pointsWritten)
            for (uint32_t kk = 0; kk < r.nPoints; ++kk)
                for (uint32_t comp = 0; comp < 4; ++comp)
                    wantContact[r.pointBase + (r.pointStart + kk) * 4 + comp] =
                        bits(r.point[kk][comp]);
    }

    size_t mmPatch = 0, mmContact = 0, strayPatch = 0, strayContact = 0;
    for (size_t w = 0; w < wantPatch.size(); ++w)
        if (gotPatch[w] != wantPatch[w])
        {
            if (wantPatch[w] == POISON) ++strayPatch;
            if (mmPatch < 6)
                std::printf("  patch word %zu (case %zu, word %zu): got 0x%08x want 0x%08x\n",
                            w, w / (PATCH_SLOTS_PER_CASE * 16), w % 16, gotPatch[w], wantPatch[w]);
            ++mmPatch;
        }
    for (size_t w = 0; w < wantContact.size(); ++w)
        if (gotContact[w] != wantContact[w])
        {
            if (wantContact[w] == POISON) ++strayContact;
            if (mmContact < 6)
                std::printf("  contact word %zu (case %zu): got 0x%08x want 0x%08x\n",
                            w, w / (CONTACT_SLOTS_PER_CASE * 4), gotContact[w], wantContact[w]);
            ++mmContact;
        }

    PXG_CHECK_MSG(mmPatch == 0, "npTriTriWrite_patch: %zu patch-stream word mismatches "
                                "(%zu of them writes into words the CUDA leaves alone)",
                  mmPatch, strayPatch);
    PXG_CHECK_MSG(mmContact == 0, "npTriTriWrite_patch: %zu contact-stream word mismatches "
                                  "(%zu of them stray writes)", mmContact, strayContact);

    std::printf("  npTriTriWrite_patch: %u cases; %zu patch-sentinel, %zu contact-sentinel, "
                "%zu both; %zu flipped; %zu zero-point, %zu six-point; %zu off-grid starts; "
                "%zu same-material controls\n",
                N, sawPatchSentinelOnly, sawContactSentinelOnly, sawBothSentinel,
                sawFlipped, sawZeroPoints, sawSixPoints, sawStartOffGrid, sawSameMaterial);

    for (PxgBufferHandle h : { bIn, bMat, bPatch, bContact }) dev->destroyBuffer(h);
}
