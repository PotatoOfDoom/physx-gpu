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

// testNpTriTriTail.cpp -- P9.83b-8 T2: the mesh-mesh kernel's output tail as one
// block (pxgSlangTriTriTail.slang, entry npTriTriTail.slang:tttTail).
//
// THE ORACLE IS COMPOSED, NOT NEW.  Every decision the tail makes is already
// mirrored: the reservation (refReserve), the header words (refPatchHeader) and
// the points (refPatchPoint), all in pxgTriTriRef.h and each already the oracle
// of its own fixture.  What only this fixture can see is the part between them:
// the atomics and what they return, the overflow OR, the two flag stores, the
// contact-manager record (a 32-bit word, a 16-bit field inside another, three
// 64-bit pointers), and which stream base each pointer is built from.
//
// WHAT THE CASES CONSTRUCT, over and above random draws:
//
//   * PRESET COUNTERS, so an offset is never 0 and a reservation that started
//     from 0 is visible; and PRESET overflow bits, so a store where the CUDA ORs
//     loses them.
//   * EACH STREAM OVERFLOWING ALONE.  The CUDA's writes are gated per stream
//     (:651, :746) while the counts are zeroed for all (:580-586): a patch-only
//     overflow still writes the points, a contact-only overflow still writes the
//     headers.  Every overflow that spares the patch or the contact stream writes
//     under the caller's N while the record says 0 -- the single-stream ones are
//     the CONSTRUCTED cases; only an overflow of both streams writes nothing.
//   * LIMITS EXACTLY ON `offset + reservation`, per stream, and force limits
//     strictly between N and 2N (candidate 56).
//   * RESERVATIONS PAST 2^32.  A counter within one reservation of 2^32: the
//     64-bit check overflows where a 32-bit one would wrap and pass, and the
//     counter itself wraps (the atomics are 32-bit).
//   * THE FLAG WORD with byte 1 ON the new count (patchChange = 0) and byte 3 on
//     it elsewhere, and a nbContactsFlags word whose high half must survive.
//   * THE START BASES ARE REAL, SEPARATE BUFFERS ("decoys") -- the published
//     pointers are built from them and the writes must NOT go through them.  A
//     tail that wrote through a start base would write into a decoy, which is
//     compared against its poison; with arbitrary integers there it would be
//     undefined behaviour, and a kill by UB proves nothing.
//   * N = 0, N = 1, N = 128 all full (768 points), and records past N poisoned
//     with non-zero point counts, plus the guard record.
//   * isFlipped both ways with DIFFERENT materials, pairIndex anywhere in a block
//     of four records, and neighbours poisoned.
//   * FUNCTION-LEVEL ONLY: N > 0 with every point count 0.  The kernel cannot
//     produce it (createNewPatches keeps a patch only with a nonzero count, :955),
//     but it is the only input that tells the CUDA's gate on the scan TOTAL (:554)
//     from a gate on N.
//   * FUNCTION-LEVEL ONLY: point counts above 6, up to 455.  The CUDA asserts
//     count <= 6 (:483, :703), but its code is defined past that: the scan takes
//     the raw count (:700-701), the point writes stop at idx = tid % 6 (:740-741)
//     and the header keeps the count's low byte (PxU8, :656).  The port's
//     min(pointCount, 6) (pxgSlangTriTriPatch) and `& 0xFF` are those three
//     readings, and without a count past 6 neither was ever exercised.

#include "pxgTestFramework.h"
#include "pxgTriTriRef.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgtest::tritriref;

struct PatchH
{
    float keep[6][4];     // @0   w = separation
    float normal[4];      // @96  xyz, w = deepest
    uint32_t pointCount;  // @112
    uint32_t pad[3];
};
static_assert(sizeof(PatchH) == 128, "patch record");

struct TttCaseH
{
    uint32_t a[4];        // @0   pairIndex, numPatchesToKeep, isFlipped, -
    uint32_t b[4];        // @16  mi0, mi1, patchLimit, contactLimit
    uint32_t f[4];        // @32  forceLimit
    float q[4];           // @48
    float p[4];           // @64
    uint64_t ptr[10];     // @80  cmOutputs .. startContactForces
    uint32_t pad[4];      // @160
    PatchH patches[129];  // @176
};
static_assert(sizeof(TttCaseH) == 16688, "npTriTriTail.slang TTT_CASE_STRIDE");
static_assert(offsetof(TttCaseH, ptr) == 80 && offsetof(TttCaseH, patches) == 176, "case layout");

struct TttPushH { uint64_t cases; uint32_t numCases, pad; };

const uint32_t POISON = 0xCDCDCDCDu;
const uint32_t FLAG_POISON = 0xA5A5A5A5u;
const uint64_t PTR_POISON = 0xD00DFEEDBAADF00Dull;
const uint32_t SENTINEL = 0xFFFFFFFFu;

const uint32_t RECORDS_PER_CASE = 4;                 // pairIndex in [0, 4)
const uint32_t RECORD_WORDS = 12;                    // 48 bytes
const uint32_t PATCH_WORDS_PER_CASE = 128 * 16;      // 128 headers
const uint32_t CONTACT_WORDS_PER_CASE = 768 * 4;     // 768 points
const uint32_t DECOY_WORDS_PER_CASE = 3072;          // a whole contact region: what a stray write reaches

} // namespace

PXG_TEST(npTriTriTail_block)
{
    std::mt19937 rng(551207);
    const uint32_t NC = 480;
    const uint32_t nbMaterials = 24;

    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    // The same material table as npTriTriWrite_patch: a third compliant, some with
    // friction disabled, every combine mode.
    std::vector<MaterialH> mats(nbMaterials);
    for (uint32_t i = 0; i < nbMaterials; ++i)
    {
        mats[i].dynamicFriction = rf(-0.2f, 1.5f);
        mats[i].staticFriction = rf(-0.2f, 1.5f);
        mats[i].restitution = (i % 3 == 0) ? rf(-4.0f, -0.01f) : rf(0.f, 1.f);
        mats[i].damping = rf(0.f, 40.f);
        const uint32_t flags = ((i % 5 == 0) ? FLAG_DISABLE_FRICTION : 0u)
                             | ((i % 4 == 0) ? FLAG_COMPLIANT_ACC_SPRING : 0u);
        mats[i].flagsModes0 = flags | ((i % 4) << 16) | (((i + 1) % 4) << 24);
        mats[i].modes1 = (i + 2) % 4;
    }

    std::vector<TttCaseH> cs(NC);
    std::memset(cs.data(), 0, cs.size() * sizeof(TttCaseH));

    // Device-side images, host-initialised: counters, records, flags, streams, decoys.
    std::vector<uint32_t> counters(size_t(NC) * 4, 0);
    std::vector<uint32_t> records(size_t(NC) * RECORDS_PER_CASE * RECORD_WORDS, POISON);
    std::vector<uint32_t> touchFlags(size_t(NC) * RECORDS_PER_CASE, FLAG_POISON);
    std::vector<uint32_t> patchFlags(size_t(NC) * RECORDS_PER_CASE, FLAG_POISON);
    std::vector<uint32_t> patchStream(size_t(NC) * PATCH_WORDS_PER_CASE, POISON);
    std::vector<uint32_t> contactStream(size_t(NC) * CONTACT_WORDS_PER_CASE, POISON);
    std::vector<uint32_t> decoy(size_t(NC) * 3 * DECOY_WORDS_PER_CASE, POISON);

    // Per-case presets the oracle needs back.
    struct Pre { uint32_t cP, cC, cF, err; };
    std::vector<Pre> pre(NC);

    size_t sawZeroN = 0, sawFull = 0, sawOne = 0, sawZeroTotal = 0;
    size_t sawOvfOnly[3] = { 0, 0, 0 }, sawExact[3] = { 0, 0, 0 }, sawWrap[3] = { 0, 0, 0 };
    size_t sawNoOvfWritten = 0, sawForceBetween = 0, sawByte1 = 0, sawErrPreset = 0;
    size_t sawFlipped = 0, sawCounterWrap = 0, sawNoPatchChange = 0, sawOverCount = 0, sawByteWrap = 0;

    for (uint32_t i = 0; i < NC; ++i)
    {
        TttCaseH& c = cs[i];
        const uint32_t shape = i % 16u;

        // ---- N and the patches ----
        uint32_t N;
        if (shape == 0u) N = 0;
        else if (shape == 1u) N = 128;
        else if (shape == 2u) N = 1;
        else if (shape == 12u) N = ((i / 16u) % 3u == 0u) ? 1u : 1u + rng() % 32u;   // <= 768 points
        else N = 1 + rng() % 128u;
        const bool overCount = shape == 12u;              // FUNCTION-LEVEL: counts past 6

        for (uint32_t pi = 0; pi < 129; ++pi)
        {
            PatchH& p = c.patches[pi];
            const bool live = pi < N;
            for (uint32_t k = 0; k < 6; ++k)
            {
                p.keep[k][0] = rf(-6.f, 6.f); p.keep[k][1] = rf(-6.f, 6.f);
                p.keep[k][2] = rf(-6.f, 6.f); p.keep[k][3] = rf(-0.6f, 0.15f);
            }
            p.normal[0] = rf(-1.f, 1.f); p.normal[1] = rf(-1.f, 1.f); p.normal[2] = rf(-1.f, 1.f);
            p.normal[3] = rf(-1.f, 0.f);
            if (!live) p.pointCount = 5u;                      // poison: a live-looking count
            else if (overCount && N == 1u) p.pointCount = 256u + rng() % 200u;   // the PxU8 header byte
            else if (overCount) p.pointCount = (pi % 2u == 0u) ? 7u + rng() % 18u : 1u + rng() % 6u;
            else if (shape == 1u) p.pointCount = 6u;
            else if (shape == 3u) p.pointCount = 0u;           // FUNCTION-LEVEL ONLY (see header)
            else p.pointCount = 1u + rng() % 6u;
        }
        uint32_t total = 0;
        bool hasOver = false, hasByteWrap = false;
        for (uint32_t pi = 0; pi < N; ++pi)
        {
            total += c.patches[pi].pointCount;
            hasOver |= c.patches[pi].pointCount > 6u;
            hasByteWrap |= c.patches[pi].pointCount > 255u;
        }
        if (N == 0) ++sawZeroN;
        if (N == 128 && total == 768) ++sawFull;
        if (N == 1) ++sawOne;
        if (N != 0 && total == 0) ++sawZeroTotal;

        // ---- counters ----
        Pre& pr = pre[i];
        pr.cP = 64u * (1u + rng() % 20000u);
        pr.cC = 16u * (1u + rng() % 50000u);
        pr.cF = 4u * (1u + rng() % 200000u);
        pr.err = (i % 3u == 0u) ? ((rng() % 8u) << 3) | (rng() % 8u) : 0u;   // bits past the three, and the three
        if (pr.err) ++sawErrPreset;
        // RESERVATIONS PAST 2^32, one stream at a time.
        if (shape == 5u && N > 0) pr.cP = 0u - 64u * (1u + (N - 1u) / 2u);    // offset + 64N > 2^32
        if (shape == 6u && total > 0) pr.cC = 0u - 16u * (1u + (total - 1u) / 2u);
        if (shape == 7u && total > 0) pr.cF = 0u - 4u * (1u + (total - 1u) / 2u);

        // ---- limits ----
        const uint64_t needP = uint64_t(pr.cP) + 64ull * N;
        const uint64_t needC = uint64_t(pr.cC) + 16ull * total;
        const uint64_t needF = uint64_t(pr.cF) + 8ull * total;     // tests 2N (:573)
        // Per stream: exact 3 in 10, above 5 in 10, below 2 in 10 -- mostly fitting,
        // so the writes run (the first draft's 3/4/3 left 81 of 480 cases writing).
        auto place = [&](uint64_t need, uint32_t which) -> uint32_t {
            const uint32_t pick = rng() % 10u;
            uint64_t lim;
            (void)which;
            if (pick < 3u) lim = need;
            else if (pick < 8u) lim = need + 1u + rng() % 4096u;
            else { const uint64_t s = 1u + rng() % 4096u; lim = need > s ? need - s : 0u; }
            return uint32_t(lim > 0xFFFFFFFFull ? 0xFFFFFFFFull : lim);
        };
        uint32_t limP = place(needP, 0), limC = place(needC, 1), limF = place(needF, 2);
        // One case in eight whose force limit sits strictly between N and 2N.
        if (i % 8u == 4u && total > 0)
        {
            limF = uint32_t(uint64_t(pr.cF) + 4ull * total + 2ull * total);
            ++sawForceBetween;
        }
        // Single-stream overflows, constructed: shapes 8/9/10 fit two streams and
        // put the third one byte below its need.
        auto fitAll = [&]() { limP = uint32_t(std::min<uint64_t>(needP, 0xFFFFFFFFull));
                              limC = uint32_t(std::min<uint64_t>(needC, 0xFFFFFFFFull));
                              limF = uint32_t(std::min<uint64_t>(needF, 0xFFFFFFFFull)); };
        if (shape == 8u && N > 0 && total > 0) { fitAll(); limP = uint32_t(needP - 1u); }
        if (shape == 9u && total > 0) { fitAll(); limC = uint32_t(needC - 1u); }
        if (shape == 10u && total > 0) { fitAll(); limF = uint32_t(needF - 1u); }
        if (shape == 11u) fitAll();
        // Two of the four byte-1 shapes fit every limit, so the unchanged count --
        // patchChange = 0, which only the byte-1 read produces -- is not erased by an
        // overflow zeroing it (the first draft kept 22 such cases).
        if (shape == 2u || shape == 14u) fitAll();
        // Counted on the FINAL limits: the overrides above move some placed limits off
        // their need, and fitAll puts others on it.
        if (needP <= 0xFFFFFFFFull && limP == uint32_t(needP)) ++sawExact[0];
        if (needC <= 0xFFFFFFFFull && limC == uint32_t(needC)) ++sawExact[1];
        if (needF <= 0xFFFFFFFFull && limF == uint32_t(needF)) ++sawExact[2];

        // ---- the record and its flag word ----
        const uint32_t pairIndex = rng() % RECORDS_PER_CASE;
        uint32_t oldAllFlags = rng();
        if (i % 4u == 2u) { oldAllFlags = (oldAllFlags & ~0x0000FF00u) | ((N & 0xFFu) << 8); ++sawByte1; }
        else if (i % 8u == 3u) oldAllFlags = (oldAllFlags & ~0xFF000000u) | ((N & 0xFFu) << 24);
        const uint32_t oldNbContactsFlags = rng();

        uint32_t* rec = &records[(size_t(i) * RECORDS_PER_CASE + pairIndex) * RECORD_WORDS];
        for (uint32_t w = 0; w < 8; w += 2) { rec[w] = uint32_t(PTR_POISON); rec[w + 1] = uint32_t(PTR_POISON >> 32); }
        rec[8] = oldAllFlags;
        rec[9] = oldNbContactsFlags;
        rec[10] = rng(); rec[11] = rng();

        // ---- transform, flip, materials ----
        const bool isFlipped = (rng() & 1u) != 0u;
        if (isFlipped) ++sawFlipped;
        uint32_t mi0 = rng() % nbMaterials, mi1 = rng() % nbMaterials;
        if (mi1 == mi0) mi1 = (mi0 + 1u) % nbMaterials;
        const float ang = rf(0.4f, 2.8f);
        V3 axis = { rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) };
        float al = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
        if (al < 1e-4f) { axis = { 0.f, 0.f, 1.f }; al = 1.f; }
        const float s = std::sin(ang * 0.5f) / al;
        c.q[0] = axis.x * s; c.q[1] = axis.y * s; c.q[2] = axis.z * s; c.q[3] = std::cos(ang * 0.5f);
        c.p[0] = rf(-9.f, 9.f); c.p[1] = rf(-9.f, 9.f); c.p[2] = rf(-9.f, 9.f); c.p[3] = 0.f;

        c.a[0] = pairIndex; c.a[1] = N; c.a[2] = isFlipped ? 1u : 0u; c.a[3] = 0u;
        c.b[0] = mi0; c.b[1] = mi1; c.b[2] = limP; c.b[3] = limC;
        c.f[0] = limF;

        counters[size_t(i) * 4 + 0] = pr.cP;
        counters[size_t(i) * 4 + 1] = pr.cC;
        counters[size_t(i) * 4 + 2] = pr.cF;
        counters[size_t(i) * 4 + 3] = pr.err;

        const RefRes r = refReserve(oldAllFlags, N, total, pr.cP, pr.cC, pr.cF, limP, limC, limF);
        if (!r.patchChange) ++sawNoPatchChange;
        if (total != 0)
        {
            const uint32_t nOvf = (r.ovfBits & OVF_PATCH ? 1u : 0u) + (r.ovfBits & OVF_CONTACT ? 1u : 0u)
                                + (r.ovfBits & OVF_FORCE ? 1u : 0u);
            if (nOvf == 1u)
            {
                if (r.ovfBits & OVF_PATCH) ++sawOvfOnly[0];
                if (r.ovfBits & OVF_CONTACT) ++sawOvfOnly[1];
                if (r.ovfBits & OVF_FORCE) ++sawOvfOnly[2];
            }
            if (!r.overflow) ++sawNoOvfWritten;
            if (hasOver && r.contactOfs != SENTINEL) ++sawOverCount;
            if (hasByteWrap && r.patchOfs != SENTINEL) ++sawByteWrap;
            if (needP > 0xFFFFFFFFull) ++sawWrap[0];
            if (needC > 0xFFFFFFFFull) ++sawWrap[1];
            if (needF > 0xFFFFFFFFull) ++sawWrap[2];
            if (uint64_t(pr.cP) + 64ull * N > 0xFFFFFFFFull || uint64_t(pr.cC) + 16ull * total > 0xFFFFFFFFull ||
                uint64_t(pr.cF) + 4ull * total > 0xFFFFFFFFull)
                ++sawCounterWrap;
        }
    }

    PXG_CHECK_MSG(sawZeroN > 20 && sawFull > 20 && sawOne > 20 && sawZeroTotal > 20,
                  "npTriTriTail_block: %zu N=0, %zu full, %zu N=1, %zu N>0 with no points",
                  sawZeroN, sawFull, sawOne, sawZeroTotal);
    PXG_CHECK_MSG(sawOvfOnly[0] > 20 && sawOvfOnly[1] > 20 && sawOvfOnly[2] > 20,
                  "npTriTriTail_block: single-stream overflows %zu patch / %zu contact / %zu force -- "
                  "the constructed cases that write under the caller's N while the record says 0",
                  sawOvfOnly[0], sawOvfOnly[1], sawOvfOnly[2]);
    PXG_CHECK_MSG(sawOverCount > 10 && sawByteWrap > 3,
                  "npTriTriTail_block: point counts past 6 written %zu times, past 255 %zu times",
                  sawOverCount, sawByteWrap);
    PXG_CHECK_MSG(sawExact[0] > 50 && sawExact[1] > 50 && sawExact[2] > 50,
                  "npTriTriTail_block: limits exactly on the need %zu / %zu / %zu",
                  sawExact[0], sawExact[1], sawExact[2]);
    PXG_CHECK_MSG(sawWrap[0] > 15 && sawWrap[1] > 15 && sawWrap[2] > 15 && sawCounterWrap > 45,
                  "npTriTriTail_block: reservations past 2^32 %zu / %zu / %zu, counter wraps %zu",
                  sawWrap[0], sawWrap[1], sawWrap[2], sawCounterWrap);
    PXG_CHECK_MSG(sawNoOvfWritten > NC / 5 && sawForceBetween > 30 && sawByte1 > NC / 5 &&
                  sawErrPreset > NC / 5 && sawFlipped > NC / 4 && sawFlipped < 3 * NC / 4,
                  "npTriTriTail_block: %zu written without overflow, %zu force limits between N and 2N, "
                  "%zu byte-1 cases, %zu preset error words, %zu flipped",
                  sawNoOvfWritten, sawForceBetween, sawByte1, sawErrPreset, sawFlipped);
    PXG_CHECK_MSG(sawNoPatchChange > NC / 20,
                  "npTriTriTail_block: only %zu cases keep the patch count (patchChange = 0) -- the "
                  "one outcome only the byte-1 read gives", sawNoPatchChange);

    // ---- buffers and addresses ----
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    PxgBufferHandle bCnt = createBufferFrom(dev, counters);
    PxgBufferHandle bRec = createBufferFrom(dev, records);
    PxgBufferHandle bTouch = createBufferFrom(dev, touchFlags);
    PxgBufferHandle bPFlag = createBufferFrom(dev, patchFlags);
    PxgBufferHandle bPatch = createBufferFrom(dev, patchStream);
    PxgBufferHandle bContact = createBufferFrom(dev, contactStream);
    PxgBufferHandle bDecoy = createBufferFrom(dev, decoy);

    const uint64_t aMat = dev->getBufferDeviceAddress(bMat);
    const uint64_t aCnt = dev->getBufferDeviceAddress(bCnt);
    const uint64_t aRec = dev->getBufferDeviceAddress(bRec);
    const uint64_t aTouch = dev->getBufferDeviceAddress(bTouch);
    const uint64_t aPFlag = dev->getBufferDeviceAddress(bPFlag);
    const uint64_t aPatch = dev->getBufferDeviceAddress(bPatch);
    const uint64_t aContact = dev->getBufferDeviceAddress(bContact);
    const uint64_t aDecoy = dev->getBufferDeviceAddress(bDecoy);

    // Each case's bases are chosen so that base + (the offset its atomic returns)
    // lands at the start of its own region.  The arithmetic is modulo 2^64, and a
    // base below the region is never dereferenced by itself.
    auto region = [](uint64_t a, uint32_t i, uint32_t words) { return a + uint64_t(i) * words * 4u; };
    for (uint32_t i = 0; i < NC; ++i)
    {
        TttCaseH& c = cs[i];
        const Pre& pr = pre[i];
        c.ptr[0] = region(aRec, i, RECORDS_PER_CASE * RECORD_WORDS);
        c.ptr[1] = aMat;
        c.ptr[2] = region(aContact, i, CONTACT_WORDS_PER_CASE) - uint64_t(pr.cC);
        c.ptr[3] = region(aPatch, i, PATCH_WORDS_PER_CASE) - uint64_t(pr.cP);
        c.ptr[4] = aCnt + uint64_t(i) * 16u;
        c.ptr[5] = region(aTouch, i, RECORDS_PER_CASE);
        c.ptr[6] = region(aPFlag, i, RECORDS_PER_CASE);
        // The decoys: start bases that are real memory, one per stream.
        c.ptr[7] = region(aDecoy, 3 * i + 0, DECOY_WORDS_PER_CASE) - uint64_t(pr.cP);
        c.ptr[8] = region(aDecoy, 3 * i + 1, DECOY_WORDS_PER_CASE) - uint64_t(pr.cC);
        c.ptr[9] = region(aDecoy, 3 * i + 2, DECOY_WORDS_PER_CASE) - uint64_t(pr.cF);
    }
    PxgBufferHandle bIn = createBufferFrom(dev, cs);

    TttPushH push{};
    push.cases = dev->getBufferDeviceAddress(bIn);
    push.numCases = NC;
    Kernel k("npTriTriTail.spv", "tttTail", {}, sizeof(TttPushH));
    k.dispatch(NC, {}, push, 1u, 1u);
    dev->flush();

    const auto gCnt = downloadBuffer<uint32_t>(dev, bCnt, counters.size());
    const auto gRec = downloadBuffer<uint32_t>(dev, bRec, records.size());
    const auto gTouch = downloadBuffer<uint32_t>(dev, bTouch, touchFlags.size());
    const auto gPFlag = downloadBuffer<uint32_t>(dev, bPFlag, patchFlags.size());
    const auto gPatch = downloadBuffer<uint32_t>(dev, bPatch, patchStream.size());
    const auto gContact = downloadBuffer<uint32_t>(dev, bContact, contactStream.size());
    const auto gDecoy = downloadBuffer<uint32_t>(dev, bDecoy, decoy.size());

    // ---- the oracle: the expected image of every buffer ----
    std::vector<uint32_t> wCnt = counters, wRec = records, wTouch = touchFlags, wPFlag = patchFlags;
    std::vector<uint32_t> wPatch = patchStream, wContact = contactStream;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const TttCaseH& c = cs[i];
        const Pre& pr = pre[i];
        const uint32_t pairIndex = c.a[0], N = c.a[1];
        const bool isFlipped = c.a[2] != 0u;

        uint32_t starts[128], total = 0;                          // :699-706
        for (uint32_t pi = 0; pi < 128; ++pi)
        {
            starts[pi] = total;
            if (pi < N) total += c.patches[pi].pointCount;
        }

        uint32_t* cnt = &wCnt[size_t(i) * 4];
        if (total != 0)                                           // :554-558, 32-bit adds
        {
            cnt[0] = pr.cP + 64u * N;
            cnt[1] = pr.cC + 16u * total;
            cnt[2] = pr.cF + 4u * total;
        }
        uint32_t* rec = &wRec[(size_t(i) * RECORDS_PER_CASE + pairIndex) * RECORD_WORDS];
        const RefRes r = refReserve(rec[8], N, total, pr.cP, pr.cC, pr.cF, c.b[2], c.b[3], c.f[0]);
        cnt[3] = pr.err | r.ovfBits;                              // :562/:569/:575, an OR

        wTouch[size_t(i) * RECORDS_PER_CASE + pairIndex] = r.touchChange ? 1u : 0u;
        wPFlag[size_t(i) * RECORDS_PER_CASE + pairIndex] = r.patchChange ? 1u : 0u;

        rec[8] = r.allFlags;
        rec[9] = (rec[9] & 0xFFFF0000u) | (r.nContacts & 0xFFFFu);
        uint64_t pPatches = 0, pPoints = 0, pForces = 0;
        if (!r.overflow)                                          // :607-612, sentinel kept
        {
            pForces = c.ptr[9] + uint64_t(r.forceOfs);
            pPatches = c.ptr[7] + uint64_t(r.patchOfs);
            pPoints = c.ptr[8] + uint64_t(r.contactOfs);
        }
        rec[0] = uint32_t(pPatches); rec[1] = uint32_t(pPatches >> 32);
        rec[2] = uint32_t(pPoints);  rec[3] = uint32_t(pPoints >> 32);
        rec[4] = uint32_t(pForces);  rec[5] = uint32_t(pForces >> 32);
        // rec[6..7] frictionPatches and rec[10..11] pad: untouched.

        const V4 q = { c.q[0], c.q[1], c.q[2], c.q[3] };
        const V3 t = { c.p[0], c.p[1], c.p[2] };
        for (uint32_t pi = 0; pi < N; ++pi)                       // the CALLER's N
        {
            const PatchH& p = c.patches[pi];
            if (r.patchOfs != SENTINEL)
            {
                uint32_t word[14];
                refPatchHeader(mats.data(), c.b[0], c.b[1], isFlipped, q,
                               V3{ p.normal[0], p.normal[1], p.normal[2] }, p.pointCount, starts[pi], word);
                refStorePatchHeader(&wPatch[size_t(i) * PATCH_WORDS_PER_CASE + size_t(pi) * 16], word);
            }
            if (r.contactOfs != SENTINEL)
            {
                const uint32_t n = std::min(p.pointCount, 6u);
                for (uint32_t kk = 0; kk < n; ++kk)
                {
                    const V4 w = refPatchPoint(q, t, V4{ p.keep[kk][0], p.keep[kk][1], p.keep[kk][2], p.keep[kk][3] });
                    float* d = reinterpret_cast<float*>(&wContact[size_t(i) * CONTACT_WORDS_PER_CASE +
                                                                   size_t(starts[pi] + kk) * 4]);
                    d[0] = w.x; d[1] = w.y; d[2] = w.z; d[3] = w.w;
                }
            }
        }
    }

    // ---- compare, whole buffers, poison included ----
    auto cmp = [&](const char* what, const std::vector<uint32_t>& got, const std::vector<uint32_t>& want,
                   uint32_t perCase) -> size_t {
        size_t mm = 0;
        for (size_t w = 0; w < want.size(); ++w)
            if (got[w] != want[w])
            {
                if (mm < 4)
                    std::printf("  %s word %zu (case %zu, word %zu): got 0x%08x want 0x%08x\n",
                                what, w, w / perCase, w % perCase, got[w], want[w]);
                ++mm;
            }
        return mm;
    };
    const size_t mmCnt = cmp("counters", gCnt, wCnt, 4);
    const size_t mmRec = cmp("record", gRec, wRec, RECORDS_PER_CASE * RECORD_WORDS);
    const size_t mmTouch = cmp("touchChange", gTouch, wTouch, RECORDS_PER_CASE);
    const size_t mmPFlag = cmp("patchChange", gPFlag, wPFlag, RECORDS_PER_CASE);
    const size_t mmPatch = cmp("patch stream", gPatch, wPatch, PATCH_WORDS_PER_CASE);
    const size_t mmContact = cmp("contact stream", gContact, wContact, CONTACT_WORDS_PER_CASE);
    const size_t mmDecoy = cmp("decoy", gDecoy, decoy, DECOY_WORDS_PER_CASE);
    PXG_CHECK_MSG(mmCnt == 0 && mmRec == 0 && mmTouch == 0 && mmPFlag == 0,
                  "npTriTriTail_block: %zu counter, %zu record, %zu touch-flag, %zu patch-flag mismatches",
                  mmCnt, mmRec, mmTouch, mmPFlag);
    PXG_CHECK_MSG(mmPatch == 0 && mmContact == 0 && mmDecoy == 0,
                  "npTriTriTail_block: %zu patch-stream, %zu contact-stream, %zu decoy mismatches",
                  mmPatch, mmContact, mmDecoy);

    std::printf("  npTriTriTail_block: %u cases; N=0 %zu, full %zu, N=1 %zu, N>0 without points %zu; "
                "single-stream overflows %zu/%zu/%zu, exact limits %zu/%zu/%zu, past 2^32 %zu/%zu/%zu "
                "(counter wraps %zu); %zu written without overflow, %zu force limits in (N, 2N), "
                "%zu byte-1 cases (%zu keep the count), %zu preset error words, %zu flipped; point counts "
                "past 6 written %zu times, past 255 %zu times\n",
                NC, sawZeroN, sawFull, sawOne, sawZeroTotal, sawOvfOnly[0], sawOvfOnly[1], sawOvfOnly[2],
                sawExact[0], sawExact[1], sawExact[2], sawWrap[0], sawWrap[1], sawWrap[2], sawCounterWrap,
                sawNoOvfWritten, sawForceBetween, sawByte1, sawNoPatchChange, sawErrPreset, sawFlipped,
                sawOverCount, sawByteWrap);

    for (PxgBufferHandle h : { bIn, bMat, bCnt, bRec, bTouch, bPFlag, bPatch, bContact, bDecoy })
        dev->destroyBuffer(h);
}
