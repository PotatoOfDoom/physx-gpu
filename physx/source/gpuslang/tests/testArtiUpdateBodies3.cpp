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

// P9.43a: `updateBodiesLaunch_Part3` (forwardDynamic2.cu:3205-3249), the
// per-articulation sleep roll-up, on the engine ABI: block (32, 2, 1), grid
// ceil(68 / 64) = 2, the same {desc, dt, bool} arg pack as Part2 with the two
// unread members poisoned.
//
// The oracle is a transcription of the .cu -- a max over the link counters with
// PxMax<float>'s select, two float compares, a masked flags word -- and the comparison
// is `==` on the raw words of the sleep-data array with NO signed-zero tolerance.
// PxMax<float> is intrinsics::selectMax, `a > b ? a : b` (PxMath.h:81-85,
// PxUnixMathIntrinsics.h:122), not the generic template `a < b ? b : a` (P9.86: the
// port and this oracle both used the generic form before): on an all -0.0 record the
// +0 seed survives, where the generic template would return -0.0.  The blocks, the records and the link counters must all
// come back byte-identical (the kernel writes eight bytes per articulation and
// nothing else).
//
// Shaped to see:
//   * 68 articulations -> two X blocks, a half-empty second warp, and the
//     articulation index a PERMUTATION of the fleet slot (the sleep-data entry and
//     the record are addressed by the permuted index, the block by the slot);
//   * link counts 0..6: every seventh articulation has NO links and a NULL counter
//     array, so a hoisted first-iteration load faults;
//   * old wake counters in {+0, -0, positive} against new maxima in {+0, -0,
//     positive}, so all three flag outcomes occur and are counted;
//   * every other bit of internalFlags set and stale transition bits present, both
//     required back exactly.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> aub3Loader(const char* file) { return loadShaderBlob(file); }

const uint32_t A3_ARTS = 68;       // grid X = ceil(68/64) = 2; warp 2 has 4 live lanes
const uint32_t A3_WARPS = 4;       // 2 blocks x 2 warps
const uint32_t A3_MAX_LINKS = 6;
const float A3_DT = 0.021484375f;  // unread by Part3; poisoned padding rides with it

// PxsRigidBody.h:52-53
static const uint32_t A3_ACTIVATE = 1u << 3;
static const uint32_t A3_DEACTIVATE = 1u << 4;

// PxgSolverBodySleepData (PxgSolverBody.h:141-145)
struct A3SleepData { float wakeCounter; uint32_t internalFlags; };
static_assert(sizeof(A3SleepData) == 8, "PxgSolverBodySleepData is 8 bytes");

// link count: every seventh articulation has none (zero-trip loop, NULL counters)
static uint32_t a3NumLinks(uint32_t a) { return (a % 7u == 6u) ? 0u : 2u + (a % 5u); }
// the record / sleep-data index is a PERMUTATION of the fleet slot, never the identity
static uint32_t a3ArtiIndex(uint32_t a) { return (a * 29u + 7u) % A3_ARTS; }

// per-link wake counters across the regimes the roll-up must separate: positive,
// +0.0, -0.0 -- and every ninth record ENTIRELY -0.0, whose maximum selectMax keeps
// as the +0 seed (the generic template would return -0.0)
static float a3Wake(uint32_t rec, uint32_t link)
{
    if (rec % 9u == 4u) return -0.0f;
    const uint32_t k = (rec * 13u + link * 7u) % 11u;
    if (k < 3u) return 0.0f;
    if (k == 3u) return -0.0f;
    return float(k) * 0.03125f + float(rec % 5u) * 0.5f;
}

static A3SleepData a3OldSleep(uint32_t rec)
{
    A3SleepData s;
    const uint32_t k = rec % 4u;
    s.wakeCounter = (k == 0u) ? 0.0f : (k == 1u) ? -0.0f : (k == 2u) ? 0.4f : 0.0078125f * float(rec + 1u);
    // every other flag bit in a distinct pattern (pass-through) plus stale transition
    // bits the kernel must clear before deciding
    s.internalFlags = ((0x3E7u ^ (rec & 0x7u)) & ~(A3_ACTIVATE | A3_DEACTIVATE))
                    | ((rec & 1u) ? A3_ACTIVATE : 0u) | ((rec & 2u) ? A3_DEACTIVATE : 0u);
    return s;
}

// forwardDynamic2.cu:3229-3247
static A3SleepData a3Mirror(const A3SleepData& old, const float* lwc, uint32_t linkCount)
{
    const float oldWc = old.wakeCounter;
    float maxWc = 0.f;
    for (uint32_t linkID = 0; linkID < linkCount; ++linkID)
        maxWc = (lwc[linkID] > maxWc) ? lwc[linkID] : maxWc; // PxMax<float>(lwc, maxWc) = selectMax
    uint32_t newFlags = old.internalFlags & ~(A3_ACTIVATE | A3_DEACTIVATE);
    if (oldWc == 0.0f && maxWc != 0.f)
        newFlags |= A3_ACTIVATE;
    else if (maxWc == 0.f && oldWc != 0.f)
        newFlags |= A3_DEACTIVATE;
    A3SleepData out;
    out.internalFlags = newFlags;
    out.wakeCounter = maxWc;
    return out;
}

static uint32_t a3Bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(artiUpdateBodies3_sleepRollup)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiUpdateBodies.spv", "updateBodiesLaunch_Part3", 16u, 0 }, // {desc, dt, bool} -> 16 B
    };
    PxgCudaShim shim(dev, manifest, 1, &aub3Loader);

    // blocks: only mNumLinks / mArticulationIndex are live; everything else is poison
    // and must come back
    std::vector<physx::PxgArticulationBlockData> ab(A3_WARPS);
    std::memset(ab.data(), 0xAB, ab.size() * sizeof(physx::PxgArticulationBlockData));
    for (uint32_t a = 0; a < A3_ARTS; ++a)
    {
        ab[a / 32u].mNumLinks[a & 31u] = a3NumLinks(a);
        ab[a / 32u].mArticulationIndex[a & 31u] = a3ArtiIndex(a);
    }

    // records: a poisoned PxgArticulation each, with only linkWakeCounters planted --
    // NULL for the zero-link ones
    std::vector<physx::PxgArticulation> arts(A3_ARTS);
    std::memset(arts.data(), 0xAB, arts.size() * sizeof(physx::PxgArticulation));
    std::vector<std::vector<float>> wake(A3_ARTS);
    std::vector<PxgDevicePtr> dWake(A3_ARTS, 0);
    std::vector<uint32_t> linksOfRec(A3_ARTS, 0);
    for (uint32_t a = 0; a < A3_ARTS; ++a)
    {
        const uint32_t rec = a3ArtiIndex(a);
        const uint32_t nl = a3NumLinks(a);
        linksOfRec[rec] = nl;
        wake[rec].assign(A3_MAX_LINKS, 0.f);
        for (uint32_t i = 0; i < A3_MAX_LINKS; ++i) wake[rec][i] = a3Wake(rec, i);
        if (nl)
        {
            shim.memAlloc(&dWake[rec], A3_MAX_LINKS * 4);
            if (dWake[rec]) shim.memcpyHtoD(dWake[rec], wake[rec].data(), A3_MAX_LINKS * 4);
        }
        arts[rec].linkWakeCounters = reinterpret_cast<physx::PxReal*>(dWake[rec]);
    }

    // sleep data per record, plus one canary entry past the fleet
    std::vector<A3SleepData> sleep(A3_ARTS + 1);
    for (uint32_t rec = 0; rec < A3_ARTS; ++rec) sleep[rec] = a3OldSleep(rec);
    sleep[A3_ARTS].internalFlags = 0xABABABABu;
    std::memcpy(&sleep[A3_ARTS].wakeCounter, &sleep[A3_ARTS].internalFlags, 4);

    PxgDevicePtr dAb = 0, dArts = 0, dSleep = 0, dDesc = 0;
    shim.memAlloc(&dAb, ab.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dArts, arts.size() * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dSleep, sleep.size() * sizeof(A3SleepData));
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    if (!dAb || !dArts || !dSleep || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dAb, ab.data(), ab.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dArts, arts.data(), arts.size() * sizeof(physx::PxgArticulation));
    shim.memcpyHtoD(dSleep, sleep.data(), sleep.size() * sizeof(A3SleepData));

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc)); // every other pointer, incl. the link blocks, is poison
    desc.nbArticulations = A3_ARTS;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.articulations = reinterpret_cast<physx::PxgArticulation*>(dArts);
    desc.articulationSleepData = reinterpret_cast<physx::PxgSolverBodySleepData*>(dSleep);
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // the expectation
    std::vector<A3SleepData> want = sleep;
    uint32_t nActivate = 0, nDeactivate = 0, nNeither = 0, nNegZero = 0, nZeroLinks = 0;
    for (uint32_t a = 0; a < A3_ARTS; ++a)
    {
        const uint32_t rec = a3ArtiIndex(a);
        want[rec] = a3Mirror(sleep[rec], wake[rec].data(), a3NumLinks(a));
        const uint32_t f = want[rec].internalFlags;
        if (f & A3_ACTIVATE) ++nActivate;
        else if (f & A3_DEACTIVATE) ++nDeactivate;
        else ++nNeither;
        {
            // records where the generic template `a < b ? b : a` would differ (-0.0)
            float g = 0.f;
            for (uint32_t l = 0; l < a3NumLinks(a); ++l)
                g = (wake[rec][l] < g) ? g : wake[rec][l];
            if (a3Bits(g) != a3Bits(want[rec].wakeCounter)) ++nNegZero;
        }
        if (a3NumLinks(a) == 0u) ++nZeroLinks;
    }
    // the seed covers what the kernel can decide
    PXG_CHECK_MSG(nActivate > 0 && nDeactivate > 0 && nNeither > 0,
                  "seed covers all three flag outcomes (activate %u, deactivate %u, neither %u)", nActivate,
                  nDeactivate, nNeither);
    PXG_CHECK_MSG(nNegZero > 0, "seed has articulations where selectMax and the generic PxMax differ (%u)", nNegZero);
    PXG_CHECK_MSG(nZeroLinks > 0, "seed has zero-link articulations (%u)", nZeroLinks);

    // the arg pack: dt and the bool byte are unread by Part3, so the padding AND the
    // bool are poisoned -- a shader that read them could only be wrong
    struct Params { uint64_t desc; float dt; uint32_t integrateWord; } prm = { uint64_t(dDesc), A3_DT, 0xB6B4B2A7u };
    PXG_CHECK_MSG(shim.launchKernel(0, 2, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch updateBodiesLaunch_Part3 (grid 2, block 32x2)");
    shim.streamSynchronize();

    // sleep data: word-exact, no signed-zero tolerance
    std::vector<A3SleepData> got(sleep.size());
    shim.memcpyDtoH(got.data(), dSleep, got.size() * sizeof(A3SleepData));
    size_t bad = 0;
    for (uint32_t rec = 0; rec <= A3_ARTS; ++rec)
    {
        const bool okWc = a3Bits(got[rec].wakeCounter) == a3Bits(want[rec].wakeCounter);
        const bool okFl = got[rec].internalFlags == want[rec].internalFlags;
        if (okWc && okFl) continue;
        if (bad < 10)
            std::printf("MM aub3 rec %u (%u links): wake ref %08x gpu %08x | flags ref %08x gpu %08x\n", rec,
                        rec < A3_ARTS ? linksOfRec[rec] : 0u, a3Bits(want[rec].wakeCounter),
                        a3Bits(got[rec].wakeCounter), want[rec].internalFlags, got[rec].internalFlags);
        ++bad;
    }
    PXG_CHECK_MSG(bad == 0, "%zu sleep-data mismatches over %u articulations + canary", bad, A3_ARTS);

    // nothing else moved
    std::vector<physx::PxgArticulationBlockData> gotAb(ab.size());
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    PXG_CHECK_MSG(std::memcmp(gotAb.data(), ab.data(), ab.size() * sizeof(physx::PxgArticulationBlockData)) == 0,
                  "the articulation blocks are untouched");
    std::vector<physx::PxgArticulation> gotArts(arts.size());
    shim.memcpyDtoH(gotArts.data(), dArts, gotArts.size() * sizeof(physx::PxgArticulation));
    PXG_CHECK_MSG(std::memcmp(gotArts.data(), arts.data(), arts.size() * sizeof(physx::PxgArticulation)) == 0,
                  "the articulation records are untouched");
    size_t wakeBad = 0;
    std::vector<float> tmp(A3_MAX_LINKS);
    for (uint32_t rec = 0; rec < A3_ARTS; ++rec)
    {
        if (!dWake[rec]) continue;
        shim.memcpyDtoH(tmp.data(), dWake[rec], A3_MAX_LINKS * 4);
        if (std::memcmp(tmp.data(), wake[rec].data(), A3_MAX_LINKS * 4) != 0) ++wakeBad;
    }
    PXG_CHECK_MSG(wakeBad == 0, "the link wake counters are untouched (%zu arrays changed)", wakeBad);

    std::printf("  articulations %u: activate %u, deactivate %u, neither %u; generic-PxMax -0.0 maxima %u; zero-link %u\n",
                A3_ARTS, nActivate, nDeactivate, nNeither, nNegZero, nZeroLinks);

    for (PxgDevicePtr p : dWake) if (p) shim.memFree(p);
    for (PxgDevicePtr p : { dAb, dArts, dSleep, dDesc }) shim.memFree(p);
}

} // namespace
