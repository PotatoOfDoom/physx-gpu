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
// Verification of npDefFilterBda.slang, the BDA fixture for pxgSlangDefFilter's
// ENGINE-PATH searches (P9.39b).
//
// testNpDefFilter.cpp next door proves the same lookups in their Phase-1 form, over
// descriptor-set bindings.  It cannot reach the MODULE, because the module takes
// device addresses and a StructuredBuffer binding is not one; that is the limitation
// pxgSlangDefFilter's own header records, and this file is the other half of it.
//
// Three things here are not reachable from the Phase-1 rig:
//
//  B1. dfFindRange's UNDERFLOW.  `return r - 1` wraps to 0xffffffff for a value below
//      the first entry.  Transcribed rather than "fixed", so the fixture asks and
//      pins the wrapped answer.
//
//      IT IS UNREACHABLE ON THE ENGINE PATH, and that is the reason it is asked here
//      SEPARATELY rather than through the particle chain.  The runsum is built at
//      PxgPBDParticleSystemCore.cpp:1588-1611 as `PxU32 runSumCount = 0;` followed by
//      `mParticleBufferRunSum[idx++] = runSumCount;` BEFORE the first increment, so
//      entry 0 is always 0.  Every particleId is a PxU32 and therefore >= 0 == v[0],
//      so the search takes at least one `l = m + 1` step and r >= 1 on exit; r - 1
//      cannot wrap.  My first version of this fixture only asked findRange through an
//      engine-shaped runsum, the coverage assertion below went red, and that red is
//      what produced this proof.
//
//      The wrap is pinned anyway, because dfFindRange is a MODULE function and the
//      next caller need not have this precondition -- and because a reader who meets
//      `r - 1u` without the proof is likely to "fix" it to `r ? r - 1u : 0u`, which
//      would be a silent behaviour change at any future call site that can pass a
//      value below v[0].  So mode 0 is aimed at its OWN array (`values`), whose first
//      entry is deliberately non-zero; the particle chain keeps the engine-shaped
//      runsum {0, 4, 9} reached through the descriptor.
//
//  B2. ignoreIndex2 = FALSE.  Every caller before P9.39b passed true, so the disjunct
//      was dead corpus-wide and hardcoding it passed.  The pairs below include an
//      entry whose (index0, index1) matches a query but whose index2 does not, and it
//      must be FOUND under true and MISSED under false.
//
//  B3. THE PARTICLE CHAIN, and that it is not the rigid one.  Three buffers with
//      distinct mUniqueId values, so a rig using one buffer could not tell the two
//      chains apart.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct DfbParams
{
    uint32_t numTests, numPairs;
    uint64_t queries, results, pairsNR, particleSystem, values;
    uint32_t numValues, pad52;
};
static_assert(sizeof(DfbParams) == 56, "npDefFilterBda push blob must be 56 bytes");
static_assert(offsetof(DfbParams, pairsNR) == 24, "pairsNR @24");
static_assert(offsetof(DfbParams, numValues) == 48, "numValues @48");

struct DfbQuery { uint32_t mode, id0, id1, id2, ignoreIndex2, value, pad0, pad1; };
static_assert(sizeof(DfbQuery) == 32, "DfbQuery must be 32 bytes");

// PxgNonRigidFilterPair (PxgSimulationCoreDesc.h:235), 16 B.
struct PairNRHost { uint32_t i0lo, i0hi, i1, i2; };
static_assert(sizeof(PairNRHost) == 16, "PxgNonRigidFilterPair is 16 bytes");

struct DevBufB { PxgBufferHandle h = 0; uint64_t addr = 0; };
template <class T> static DevBufB mk(PxgSlangComputeDevice* dev, const std::vector<T>& v)
{
    DevBufB b;
    b.h = createBufferFrom(dev, v);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// ---- the host mirror, straight from deformableElementFilter.cuh ------------------
static bool u64Lt(uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi)
{ return (ahi != bhi) ? (ahi < bhi) : (alo < blo); }

static bool findNR3(const std::vector<PairNRHost>& p, uint32_t id0, uint32_t id1,
                    uint32_t id2, bool ignoreIndex2)
{
    uint32_t l = 0, r = uint32_t(p.size());
    while (l < r)
    {
        const uint32_t mid = (l + r) / 2u;
        const PairNRHost& e = p[mid];
        int cmp = 0;
        if (u64Lt(id0, 0u, e.i0lo, e.i0hi)) cmp = -1;
        else if (u64Lt(e.i0lo, e.i0hi, id0, 0u)) cmp = 1;
        else if (id1 < e.i1) cmp = -1;
        else if (id1 > e.i1) cmp = 1;
        else if (id2 < e.i2) cmp = -1;
        else if (id2 > e.i2) cmp = 1;
        if (cmp == 0 || (ignoreIndex2 && id0 == e.i0lo && e.i0hi == 0u && id1 == e.i1))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}

// findRange, INCLUDING the r - 1 wrap (B1).
static uint32_t findRangeRef(const std::vector<uint32_t>& v, uint32_t length, uint32_t value)
{
    if (length == 0u) return 0u;
    uint32_t l = 0, r = length;
    while (l < r) { const uint32_t m = (l + r) / 2u; if (v[m] > value) r = m; else l = m + 1u; }
    return r - 1u;   // wraps deliberately
}

PXG_TEST(npDefFilterBda)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the runsum: three particle buffers, boundaries at 0, 4, 9 ---------------
    const std::vector<uint32_t> runsum = { 0u, 4u, 9u };
    DevBufB bRun = mk(dev, runsum);
    if (!bRun.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the direct findRange array, first entry NON-ZERO (B1) -------------------
    // Not an engine shape -- see the B1 note in the header: the engine's runsum always
    // starts at 0, so the wrap is unreachable there.  dfFindRange is a module function
    // and this pins its transcription independently of its one current caller.
    const std::vector<uint32_t> values = { 3u, 4u, 9u };
    DevBufB bVal = mk(dev, values);

    // ---- the particle system: three sim buffers with DISTINCT unique ids (B3) ----
    // PxgParticleSimBuffer is 88 B with mUniqueId @80; PxgParticleSystem carries
    // mCommonData.mNumParticleBuffers @132, mParticleSimBuffers @400,
    // mParticleBufferRunsum @408.  Built as raw words so the fixture pins the OFFSETS
    // as well as the values -- a descriptor whose fields moved would read garbage here
    // rather than quietly agreeing with a mirror built from the same wrong offsets.
    const uint32_t SIMBUF_STRIDE = 88u;
    std::vector<uint32_t> simBufs(3u * SIMBUF_STRIDE / 4u, 0u);
    const uint32_t UID[3] = { 700u, 800u, 900u };
    for (uint32_t b = 0; b < 3u; ++b)
        simBufs[b * (SIMBUF_STRIDE / 4u) + 80u / 4u] = UID[b];   // mUniqueId @80
    DevBufB bSim = mk(dev, simBufs);

    std::vector<uint32_t> psWords(512u / 4u, 0u);
    psWords[132u / 4u] = 3u;                                       // mNumParticleBuffers
    std::memcpy(&psWords[400u / 4u], &bSim.addr, 8);               // mParticleSimBuffers
    std::memcpy(&psWords[408u / 4u], &bRun.addr, 8);               // mParticleBufferRunsum
    DevBufB bPs = mk(dev, psWords);

    // ---- the pair array, sorted by (index0, index1, index2) ----------------------
    // Entry 2 shares (index0, index1) with entry 1 but carries a DIFFERENT index2 --
    // that is the pair B2 turns on.
    //
    // B1c -- THE HIGH WORD OF index0.  PxgNonRigidFilterPair::index0 is a PxU64 and its
    // compare (PxgSimulationCoreDesc.h:235) is a 64-bit one, but the query this module
    // builds is `uint2(id0, 0u)` -- a WIDENED PxU32, high word always zero.  With every
    // stored pair also carrying a zero high word, dropping the high-word load entirely
    // is unobservable, and the P9.39b sweep proved that by leaving `C-hi-word-ignored`
    // green.  The last two entries carry NON-ZERO high words so the comparison has to
    // load and use them: both must be MISSED, and they must not disturb the ordering
    // that the searches below rely on.
    //
    // B1d -- index1 MUST DISCRIMINATE.  In the first version every distinct index0 had
    // exactly one index1, so the index1 comparison never chose between two entries: it
    // fired only when index0 was already equal, and then index1 was equal too.
    // Reversing its two branches was therefore invisible, and the sweep proved it by
    // leaving `C-index1-order-swap` green.  index0 = 7 now carries THREE index1 values
    // (15, 22, 30) so a search for the lowest has to step down past the highest --
    // exactly the step a reversed comparison gets wrong.
    std::vector<PairNRHost> pairs = {
        { 5u, 0u, 11u, 0u   },
        { 7u, 0u, 15u, 0u   },   // B1d: same index0 as the pair below, lower index1
        { 7u, 0u, 22u, 700u },
        { 7u, 0u, 22u, 900u },
        { 7u, 0u, 30u, 0u   },   // B1d: same index0, higher index1
        { 9u, 0u, 33u, 800u },
        { 12u, 0u, 44u, 0u  },
        { 7u, 1u, 22u, 700u },   // same low word as (7,22,700), high word 1 -> other pair
        { 0u, 2u, 11u, 0u   },   // low word 0, high word 2 -> sorts after everything
    };
    std::sort(pairs.begin(), pairs.end(), [](const PairNRHost& a, const PairNRHost& b) {
        if (a.i0hi != b.i0hi) return a.i0hi < b.i0hi;
        if (a.i0lo != b.i0lo) return a.i0lo < b.i0lo;
        if (a.i1 != b.i1) return a.i1 < b.i1;
        return a.i2 < b.i2;
    });
    DevBufB bPairs = mk(dev, pairs);

    // ---- the queries -------------------------------------------------------------
    std::vector<DfbQuery> qs;
    auto range = [&](uint32_t v) { qs.push_back(DfbQuery{ 0u, 0u, 0u, 0u, 0u, v, 0u, 0u }); };
    auto nr3   = [&](uint32_t a, uint32_t b, uint32_t c, bool ign)
                 { qs.push_back(DfbQuery{ 1u, a, b, c, ign ? 1u : 0u, 0u, 0u, 0u }); };
    auto nr2   = [&](uint32_t a, uint32_t b) { qs.push_back(DfbQuery{ 2u, a, b, 0u, 0u, 0u, 0u, 0u }); };
    auto part  = [&](uint32_t a, uint32_t b) { qs.push_back(DfbQuery{ 3u, a, b, 0u, 0u, 0u, 0u, 0u }); };

    // B1: below the first entry -> the WRAP; then on each boundary; then past the end.
    // Against values = {3, 4, 9}: 0 and 1 wrap, 3 -> 0, 4 -> 1, 8 -> 1, 9 -> 2, 100 -> 2.
    range(0u); range(1u); range(3u); range(4u); range(8u); range(9u); range(100u);

    // B2: the same (index0, index1) with the RIGHT and the WRONG index2, each asked
    // both ways.  700 is entry 1's index2, 900 is entry 2's, 555 is neither.
    nr3(7u, 22u, 700u, false);   // exact -> found
    nr3(7u, 22u, 900u, false);   // exact -> found
    nr3(7u, 22u, 555u, false);   // index2 wrong, ignore FALSE -> MISSED
    nr3(7u, 22u, 555u, true);    // index2 wrong, ignore TRUE  -> found
    nr3(5u, 11u, 0u,   false);
    nr3(4u, 11u, 0u,   false);   // absent
    nr2(7u, 22u);                // the two-key wrapper: index2 = 0, ignore = true
    nr2(12u, 44u);
    nr2(12u, 45u);               // absent

    // B1c: the query's index0 high word is ALWAYS zero, so these must MISS.  Entry 6 is
    // (lo 0, hi 2, index1 11): drop the high-word load and it becomes (0, 0, 11), which
    // matches this query exactly and turns both into hits.  This is the only place the
    // high word changes an answer.
    nr2(0u, 11u);
    nr3(0u, 11u, 0u, false);

    // B1d: (7, 15) is the LOWEST index1 under index0 = 7, and the search's first probe
    // lands on (7, 30).  Getting there requires index1 to compare in the right
    // direction: reverse it and the search walks away from the entry and reports a miss.
    // (7, 30) itself is useless as a probe -- the first comparison hits it exactly, so
    // both directions agree.
    nr2(7u, 15u);
    nr2(7u, 30u);

    // B3: the particle chain.  particle 2 -> buffer 0 (uid 700), particle 6 -> buffer
    // 1 (uid 800), particle 11 -> buffer 2 (uid 900).  The pair array was built so
    // that (7,22) matches only under uid 700 or 900, and (9,33) only under 800.
    part(7u, 22u);    // particleId 7 -> runsum{0,4,9} -> buffer 1 -> uid 800 -> MISS
    part(9u, 33u);    // particleId 9 -> buffer 2 -> uid 900 -> MISS (entry has 800)
    part(5u, 11u);    // particleId 5 -> buffer 1 -> uid 800; entry has index2 0 -> MISS

    const uint32_t N = uint32_t(qs.size());
    DevBufB bQ = mk(dev, qs);
    std::vector<uint32_t> outInit(N, 0xABABABABu);
    DevBufB bOut = mk(dev, outInit);

    DfbParams p = {};
    p.numTests = N; p.numPairs = uint32_t(pairs.size());
    p.queries = bQ.addr; p.results = bOut.addr; p.pairsNR = bPairs.addr;
    p.particleSystem = bPs.addr; p.values = bVal.addr; p.numValues = uint32_t(values.size());

    Kernel k("npDefFilterBda.spv", "dfbRun", {}, sizeof(DfbParams));
    k.dispatch((N + 63u) / 64u, {}, p, 1u, 1u);
    dev->flush();

    const std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bOut.h, N);

    // ---- the mirror ---------------------------------------------------------------
    uint32_t bad = 0u, wrapped = 0u, ignoreSeparated = 0u;
    for (uint32_t i = 0; i < N; ++i)
    {
        const DfbQuery& q = qs[i];
        uint32_t want = 0u;
        if (q.mode == 0u)
        {
            want = findRangeRef(values, uint32_t(values.size()), q.value);
            if (want == 0xffffffffu) ++wrapped;
        }
        else if (q.mode == 1u)
            want = findNR3(pairs, q.id0, q.id1, q.id2, q.ignoreIndex2 != 0u) ? 1u : 0u;
        else if (q.mode == 2u)
            want = findNR3(pairs, q.id0, q.id1, 0u, true) ? 1u : 0u;
        else
        {
            const uint32_t id = findRangeRef(runsum, uint32_t(runsum.size()), q.id0);
            want = findNR3(pairs, q.id0, q.id1, UID[id], false) ? 1u : 0u;
        }
        if (got[i] != want)
        {
            ++bad;
            std::printf("  query %u (mode %u, id0 %u, id1 %u, id2 %u, ign %u, value %u): "
                        "got %u, mirror %u\n", i, q.mode, q.id0, q.id1, q.id2,
                        q.ignoreIndex2, q.value, got[i], want);
        }
    }
    // The two ignoreIndex2 spellings of the SAME query must DISAGREE, or B2 is not
    // being exercised and a port that hardcoded the flag still passes.
    for (uint32_t i = 0; i + 1u < N; ++i)
        if (qs[i].mode == 1u && qs[i + 1u].mode == 1u &&
            qs[i].id0 == qs[i + 1u].id0 && qs[i].id1 == qs[i + 1u].id1 &&
            qs[i].id2 == qs[i + 1u].id2 && qs[i].ignoreIndex2 != qs[i + 1u].ignoreIndex2 &&
            got[i] != got[i + 1u])
            ++ignoreSeparated;

    PXG_CHECK_MSG(bad == 0u, "%u of %u queries disagree with the mirror", bad, N);
    PXG_CHECK_MSG(wrapped > 0u,
                  "no findRange query fell below values[0], so the deliberate r - 1 WRAP "
                  "(B1) is never exercised -- note this needs the SEPARATE `values` array, "
                  "since the engine's runsum starts at 0 and cannot underflow");
    PXG_CHECK_MSG(ignoreSeparated > 0u,
                  "no pair of queries differs ONLY in ignoreIndex2 and in its answer, so "
                  "hardcoding the flag (B2) would still pass");

    // B1c coverage: there must be a stored pair whose index0 high word is non-zero AND
    // whose low word plus index1 match some query -- otherwise the 64-bit comparison is
    // indistinguishable from a 32-bit one.
    uint32_t hiWordDiscriminating = 0u;
    for (const PairNRHost& e : pairs)
        if (e.i0hi != 0u)
            for (const DfbQuery& q : qs)
                if ((q.mode == 1u || q.mode == 2u) && q.id0 == e.i0lo && q.id1 == e.i1)
                    ++hiWordDiscriminating;
    // B1d coverage: some index0 must carry at least two DIFFERENT index1 values, and a
    // query must ask for one of them -- otherwise index1 never chooses between entries
    // and reversing its comparison stays invisible.
    uint32_t idx1Discriminating = 0u;
    for (const PairNRHost& e : pairs)
        for (const PairNRHost& f : pairs)
            if (e.i0lo == f.i0lo && e.i0hi == f.i0hi && e.i1 != f.i1)
                for (const DfbQuery& q : qs)
                    if ((q.mode == 1u || q.mode == 2u) && q.id0 == e.i0lo && q.id1 == e.i1)
                        ++idx1Discriminating;
    PXG_CHECK_MSG(idx1Discriminating > 0u,
                  "no index0 carries two different index1 values with a query asking for "
                  "one of them, so index1 never chooses between entries and reversing its "
                  "comparison (B1d) would still pass");

    PXG_CHECK_MSG(hiWordDiscriminating > 0u,
                  "no query matches a pair in (low word, index1) while differing in the "
                  "index0 HIGH word, so ignoring the high word entirely (B1c) would still "
                  "pass -- the compare would be 32-bit and nothing would notice");

    // ---- B1b: the EMPTY array, in its own dispatch --------------------------------
    // findRange's first statement is `if (length == 0) return 0;`
    // (deformableElementFilter.cuh:101).  It is a real branch of the engine and NOT the
    // same answer as the rest of the function would give: with length == 0 the loop
    // body never runs, r stays 0, and `return r - 1` would hand back 0xffffffff.  The
    // guard is what makes the empty case 0 instead.
    //
    // The P9.39b mutation sweep found this: changing that `0u` to `0xffffffffu` left
    // the fixture GREEN, because every query above is asked with numValues == 3.  The
    // gap is closed here rather than allowed, and it needs a SECOND DISPATCH because
    // numValues is one push-constant shared by every query in a dispatch.
    {
        std::vector<DfbQuery> eq = { DfbQuery{ 0u, 0u, 0u, 0u, 0u, 7u, 0u, 0u },
                                     DfbQuery{ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u } };
        DevBufB bEQ = mk(dev, eq);
        std::vector<uint32_t> eInit(eq.size(), 0xABABABABu);
        DevBufB bEOut = mk(dev, eInit);

        DfbParams pe = p;
        pe.numTests = uint32_t(eq.size());
        pe.queries = bEQ.addr;
        pe.results = bEOut.addr;
        pe.numValues = 0u;               // the branch under test

        k.dispatch(1u, {}, pe, 1u, 1u);
        dev->flush();

        const std::vector<uint32_t> eGot = downloadBuffer<uint32_t>(dev, bEOut.h, eq.size());
        for (size_t i = 0; i < eq.size(); ++i)
            PXG_CHECK_MSG(eGot[i] == 0u,
                          "findRange over an EMPTY array returned %u for value %u, but the "
                          "engine's length == 0 guard returns 0 (a port without the guard "
                          "would return 0xffffffff here)", eGot[i], eq[i].value);
    }

    std::printf("    [ note ] %u queries over %zu pairs and %zu buffers; %u findRange "
                "wraps, %u ignoreIndex2 separations, plus 2 empty-array queries\n",
                N, pairs.size(), runsum.size(), wrapped, ignoreSeparated);
}
} // namespace
