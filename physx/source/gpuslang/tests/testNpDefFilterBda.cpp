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
    std::vector<PairNRHost> pairs = {
        { 5u, 0u, 11u, 0u   },
        { 7u, 0u, 22u, 700u },
        { 7u, 0u, 22u, 900u },
        { 9u, 0u, 33u, 800u },
        { 12u, 0u, 44u, 0u  },
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
    std::printf("    [ note ] %u queries over %zu pairs and %zu buffers; %u findRange "
                "wraps, %u ignoreIndex2 separations\n",
                N, pairs.size(), runsum.size(), wrapped, ignoreSeparated);
}
} // namespace
