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

// P8.37 — the contact-manager removal chain (npPairMgmtBda.spv) on the engine ABI.
//
// Runs all five stages exactly as removeLostPairsGpuInternal does — same struct,
// same argument pack, grid 32 — and compares the compacted arrays against a CPU
// transcription of SparseRemove.cuh.  The chain is only meaningful end to end: the
// first four stages exist to build the run-sum that stage 5 binary-searches.
//
// Suite (ctest filter "pairMgmt"):
//   pairMgmtLayout   — the PxgPairManagementData offsets the shader bakes in
//   pairMgmtChain    — stage 1..5, single manifold (16 threads/swap)
//   pairMgmtChainCvx — stage 1..5_CvxTri, multi-manifold (32 threads/swap)

#include "pxgTestFramework.h"

#include "PxgContactManager.h"
#include "PxgPersistentContactManifold.h"
#include "PxsContactManagerState.h"
#include "cudaNpCommon.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

// The push blob the engine builds: {const PxgPairManagementData*, bool}.  The shim
// writes 9 bytes, so the three bytes above the bool are whatever the push-constant
// range already held — the test writes garbage there on purpose, which is what the
// shader's `& 0xff` has to survive.
struct PmPush
{
    uint64_t pairData;
    uint32_t copyManifoldWord;
};

const uint32_t GRID = 32u;   // PxgNarrowPhaseGridDims::REMOVE_CONTACT_MANAGERS

// reduction.cuh:729
uint32_t cpuBinarySearch(const std::vector<uint32_t>& data, uint32_t n, uint32_t value)
{
    uint32_t left = 0u, right = n;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2u;
        if (data[pos] <= value) left = pos + 1u; else right = pos;
    }
    return left ? left - 1u : 0u;
}

struct Payload
{
    std::vector<uint32_t> input;      // 4 words / pair  (PxgContactManagerInput)
    std::vector<uint32_t> output;     // 12 words / pair (PxsContactManagerOutput)
    std::vector<uint64_t> cms;        // 1 pointer / pair
    std::vector<uint64_t> sis;        // 1 pointer / pair
    std::vector<uint32_t> rest;       // 1 word / pair   (PxReal, compared as bits)
    std::vector<uint32_t> torsional;  // 2 words / pair
    std::vector<uint32_t> manifold;   // nbF4*4 words / pair
};

void fill(Payload& p, uint32_t n, uint32_t nbF4)
{
    p.input.resize(size_t(n) * 4u);
    p.output.resize(size_t(n) * 12u);
    p.cms.resize(n);
    p.sis.resize(n);
    p.rest.resize(n);
    p.torsional.resize(size_t(n) * 2u);
    p.manifold.resize(size_t(n) * nbF4 * 4u);
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t w = 0; w < 4u; ++w)  p.input[size_t(i) * 4u + w]  = 0x1000000u + i * 16u + w;
        for (uint32_t w = 0; w < 12u; ++w) p.output[size_t(i) * 12u + w] = 0x2000000u + i * 16u + w;
        p.cms[i] = 0xC000000000000000ull + uint64_t(i) * 0x51ull;
        p.sis[i] = 0x5000000000000000ull + uint64_t(i) * 0x71ull;
        // A rest distance that is a signalling-NaN bit pattern for half the pairs:
        // it moves as raw bits, and a float-typed copy would be free to canonicalise
        // it.  Same reason the manifold words below are not floats.
        p.rest[i] = (i & 1u) ? (0x7F800001u + i) : (0x3F800000u + i);
        for (uint32_t w = 0; w < 2u; ++w) p.torsional[size_t(i) * 2u + w] = 0x3000000u + i * 4u + w;
        for (uint32_t w = 0; w < nbF4 * 4u; ++w)
            p.manifold[size_t(i) * nbF4 * 4u + w] = 0xCD000000u + i * 4096u + w;
    }
}

void movePair(Payload& p, uint32_t dst, uint32_t src, uint32_t nbF4, bool copyManifold)
{
    for (uint32_t w = 0; w < 4u; ++w)  p.input[size_t(dst) * 4u + w]  = p.input[size_t(src) * 4u + w];
    for (uint32_t w = 0; w < 12u; ++w) p.output[size_t(dst) * 12u + w] = p.output[size_t(src) * 12u + w];
    p.cms[dst] = p.cms[src];
    p.sis[dst] = p.sis[src];
    p.rest[dst] = p.rest[src];
    for (uint32_t w = 0; w < 2u; ++w) p.torsional[size_t(dst) * 2u + w] = p.torsional[size_t(src) * 2u + w];
    if (copyManifold)
        for (uint32_t w = 0; w < nbF4 * 4u; ++w)
            p.manifold[size_t(dst) * nbF4 * 4u + w] = p.manifold[size_t(src) * nbF4 * 4u + w];
}

template <typename T>
size_t cmpVec(const char* what, const std::vector<T>& got, const std::vector<T>& exp, size_t& printed)
{
    size_t bad = 0;
    for (size_t i = 0; i < exp.size(); ++i)
        if (got[i] != exp[i])
        {
            ++bad;
            if (printed < 6) { ++printed; std::printf("  %s[%zu] differs\n", what, i); }
        }
    return bad;
}

// Runs the five stages for one (N, K) case and returns the mismatch count.
size_t runCase(const char* stage5Entry, uint32_t nbF4, bool copyManifold,
               uint32_t n, const std::vector<uint32_t>& removeIndices)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t k = uint32_t(removeIndices.size());

    Payload host;
    fill(host, n, nbF4);
    Payload expect = host;

    // ---- CPU reference -----------------------------------------------------
    const uint32_t newSize = n - k;
    std::vector<uint32_t> mark(n);
    for (uint32_t i = 0; i < n; ++i) mark[i] = (i < newSize) ? 0u : 1u;
    for (uint32_t r : removeIndices) mark[r] = (r < newSize) ? 1u : 0u;
    std::vector<uint32_t> runSum(n);
    { uint32_t acc = 0; for (uint32_t i = 0; i < n; ++i) { runSum[i] = acc; acc += mark[i]; } }
    const uint32_t nbToSwap = runSum[newSize];
    for (uint32_t i = 0; i < nbToSwap; ++i)
    {
        const uint32_t dst = cpuBinarySearch(runSum, n, i);
        const uint32_t src = cpuBinarySearch(runSum, n, i + nbToSwap);
        movePair(expect, dst, src, nbF4, copyManifold);
    }

    // ---- device buffers ----------------------------------------------------
    std::vector<uint32_t> zeroRun(n, 0xEEEEEEEEu);
    std::vector<uint32_t> zeroBlk(GRID, 0xEEEEEEEEu);
    PxgBufferHandle bRun = createBufferFrom(dev, zeroRun);
    PxgBufferHandle bBlk = createBufferFrom(dev, zeroBlk);
    PxgBufferHandle bRem = createBufferFrom(dev, removeIndices);
    PxgBufferHandle bIn = createBufferFrom(dev, host.input);
    PxgBufferHandle bOut = createBufferFrom(dev, host.output);
    PxgBufferHandle bCms = createBufferFrom(dev, host.cms);
    PxgBufferHandle bSis = createBufferFrom(dev, host.sis);
    PxgBufferHandle bRest = createBufferFrom(dev, host.rest);
    PxgBufferHandle bTor = createBufferFrom(dev, host.torsional);
    PxgBufferHandle bMf = createBufferFrom(dev, host.manifold);

    // PxgPairManagementData, 88 bytes, built exactly as removeLostPairsGpuInternal
    // builds it (PxgNarrowphaseCore.cpp:8504).
    std::vector<uint64_t> pmd(11, 0);
    pmd[0] = dev->getBufferDeviceAddress(bIn);    // mContactManagerInputData
    pmd[1] = dev->getBufferDeviceAddress(bOut);   // mContactManagerOutputData
    pmd[2] = dev->getBufferDeviceAddress(bCms);   // mCpuContactManagerMapping
    pmd[3] = dev->getBufferDeviceAddress(bSis);   // mShapeInteractions
    pmd[4] = dev->getBufferDeviceAddress(bRest);  // mRestDistances
    pmd[5] = dev->getBufferDeviceAddress(bTor);   // mTorsionalData
    pmd[6] = dev->getBufferDeviceAddress(bRun);   // mTempAccumulator
    pmd[7] = dev->getBufferDeviceAddress(bBlk);   // mBlockSharedAccumulator
    pmd[8] = dev->getBufferDeviceAddress(bRem);   // mRemoveIndices
    { const uint32_t two[2] = { n, k }; std::memcpy(&pmd[9], two, sizeof(two)); }  // @72, @76
    pmd[10] = dev->getBufferDeviceAddress(bMf);   // mPersistentContactManagers
    PxgBufferHandle bPmd = createBufferFrom(dev, pmd);

    PmPush push{};
    push.pairData = dev->getBufferDeviceAddress(bPmd);
    push.copyManifoldWord = 0xA5B6C700u | (copyManifold ? 1u : 0u);

    {
        Kernel s1("npPairMgmtBda.spv", "removeContactManagers_Stage1", {}, sizeof(PmPush));
        Kernel s2("npPairMgmtBda.spv", "removeContactManagers_Stage2", {}, sizeof(PmPush));
        Kernel s3("npPairMgmtBda.spv", "removeContactManagers_Stage3", {}, sizeof(PmPush));
        Kernel s4("npPairMgmtBda.spv", "removeContactManagers_Stage4", {}, sizeof(PmPush));
        Kernel s5("npPairMgmtBda.spv", stage5Entry, {}, sizeof(PmPush));
        s1.dispatch(GRID, {}, push, 1u, 1u);
        s2.dispatch(GRID, {}, push, 1u, 1u);
        s3.dispatch(GRID, {}, push, 1u, 1u);
        s4.dispatch(GRID, {}, push, 1u, 1u);
        s5.dispatch(GRID, {}, push, 1u, 1u);
        dev->flush();
    }

    size_t bad = 0, printed = 0;

    // The run-sum is an intermediate, but it is the one the binary search reads,
    // so a wrong scan is worth naming rather than only seeing as moved payload.
    const auto gotRun = downloadBuffer<uint32_t>(dev, bRun, n);
    bad += cmpVec("runSum", gotRun, runSum, printed);

    Payload got;
    got.input = downloadBuffer<uint32_t>(dev, bIn, host.input.size());
    got.output = downloadBuffer<uint32_t>(dev, bOut, host.output.size());
    got.cms = downloadBuffer<uint64_t>(dev, bCms, host.cms.size());
    got.sis = downloadBuffer<uint64_t>(dev, bSis, host.sis.size());
    got.rest = downloadBuffer<uint32_t>(dev, bRest, host.rest.size());
    got.torsional = downloadBuffer<uint32_t>(dev, bTor, host.torsional.size());
    got.manifold = downloadBuffer<uint32_t>(dev, bMf, host.manifold.size());

    bad += cmpVec("input", got.input, expect.input, printed);
    bad += cmpVec("output", got.output, expect.output, printed);
    bad += cmpVec("cms", got.cms, expect.cms, printed);
    bad += cmpVec("sis", got.sis, expect.sis, printed);
    bad += cmpVec("rest", got.rest, expect.rest, printed);
    bad += cmpVec("torsional", got.torsional, expect.torsional, printed);
    bad += cmpVec("manifold", got.manifold, expect.manifold, printed);

    if (bad)
        std::printf("  case N=%u K=%u nbToSwap=%u entry=%s copyManifold=%d: %zu mismatches\n",
                    n, k, nbToSwap, stage5Entry, int(copyManifold), bad);

    for (PxgBufferHandle h : { bRun, bBlk, bRem, bIn, bOut, bCms, bSis, bRest, bTor, bMf, bPmd })
        dev->destroyBuffer(h);
    return bad;
}

// The remove list the engine passes is PxSort'ed ascending (PxgNarrowphaseCore.cpp:8456).
std::vector<uint32_t> spread(uint32_t n, uint32_t k, uint32_t seed)
{
    std::vector<uint8_t> chosen(n, 0);
    uint32_t s = seed | 1u, taken = 0;
    while (taken < k)
    {
        s = s * 1664525u + 1013904223u;
        const uint32_t idx = s % n;
        if (!chosen[idx]) { chosen[idx] = 1; ++taken; }
    }
    std::vector<uint32_t> out;
    out.reserve(k);
    for (uint32_t i = 0; i < n; ++i) if (chosen[i]) out.push_back(i);
    return out;
}

PXG_TEST(pairMgmtLayout)
{
    using namespace physx;
    static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput size");
    static_assert(sizeof(PxsContactManagerOutput) == 48, "PxsContactManagerOutput size");
    static_assert(sizeof(PxsTorsionalFrictionData) == 8, "PxsTorsionalFrictionData size");
    static_assert(sizeof(PxgPersistentContactManifold) == 256, "manifold size");
    static_assert(sizeof(PxgPersistentContactMultiManifold) == 1216, "multi-manifold size");

    // The offsets npPairMgmtBda.slang bakes in.  mPersistentContactManagers is the
    // one that is not obvious: the two PxU32 at 72/76 do NOT let a pointer follow
    // at 80 by accident — it re-aligns there, and the struct is 88, not 84.
    size_t bad = 0;
    auto chk = [&](const char* what, size_t got, size_t want) {
        if (got != want) { ++bad; std::printf("  %s at %zu, expected %zu\n", what, got, want); }
    };
    chk("mContactManagerInputData", offsetof(PxgPairManagementData, mContactManagerInputData), 0);
    chk("mContactManagerOutputData", offsetof(PxgPairManagementData, mContactManagerOutputData), 8);
    chk("mCpuContactManagerMapping", offsetof(PxgPairManagementData, mCpuContactManagerMapping), 16);
    chk("mShapeInteractions", offsetof(PxgPairManagementData, mShapeInteractions), 24);
    chk("mRestDistances", offsetof(PxgPairManagementData, mRestDistances), 32);
    chk("mTorsionalData", offsetof(PxgPairManagementData, mTorsionalData), 40);
    chk("mTempAccumulator", offsetof(PxgPairManagementData, mTempAccumulator), 48);
    chk("mBlockSharedAccumulator", offsetof(PxgPairManagementData, mBlockSharedAccumulator), 56);
    chk("mRemoveIndices", offsetof(PxgPairManagementData, mRemoveIndices), 64);
    chk("mNbPairs", offsetof(PxgPairManagementData, mNbPairs), 72);
    chk("mNbToRemove", offsetof(PxgPairManagementData, mNbToRemove), 76);
    chk("mPersistentContactManagers", offsetof(PxgPairManagementData, mPersistentContactManagers), 80);
    chk("sizeof", sizeof(PxgPairManagementData), 88);

    PXG_CHECK_MSG(bad == 0, "PxgPairManagementData layout matches the shader (%zu off)", bad);
}

PXG_TEST(pairMgmtChain)
{
    size_t bad = 0;
    // K entirely inside the kept region: every removal is a hole, so nbToSwap == K.
    bad += runCase("removeContactManagers_Stage5", 16u, true, 40u, { 0u, 1u, 7u, 30u });
    // Removals straddling the boundary, and one case where every removal is
    // already in the tail so nbToSwap is 0 and stage 5 must do nothing at all.
    bad += runCase("removeContactManagers_Stage5", 16u, true, 40u, { 36u, 37u, 38u, 39u });
    bad += runCase("removeContactManagers_Stage5", 16u, true, 40u, { 3u, 36u, 38u, 39u });
    // A single pair removed from a single-pair array (newSize 0).
    bad += runCase("removeContactManagers_Stage5", 16u, true, 1u, { 0u });
    // copyManifold false: the manifold array must come back untouched even though
    // the payload moves.  This is the `no manifold` bucket loop at :8486.
    bad += runCase("removeContactManagers_Stage5", 16u, false, 40u, { 0u, 1u, 7u, 30u });
    // Past one scan block (512) and past the whole grid's worth (512*32 = 16384),
    // so stage 3/4 run several iterations per block and the cross-block prefix
    // actually carries.
    bad += runCase("removeContactManagers_Stage5", 16u, true, 1000u, spread(1000u, 137u, 7u));
    bad += runCase("removeContactManagers_Stage5", 16u, true, 20000u, spread(20000u, 4321u, 11u));
    PXG_CHECK_MSG(bad == 0, "removeContactManagers chain matches SparseRemove (%zu mismatches)", bad);
}

PXG_TEST(pairMgmtChainCvx)
{
    size_t bad = 0;
    bad += runCase("removeContactManagers_Stage5_CvxTri", 76u, true, 40u, { 0u, 1u, 7u, 30u });
    bad += runCase("removeContactManagers_Stage5_CvxTri", 76u, true, 40u, { 36u, 37u, 38u, 39u });
    bad += runCase("removeContactManagers_Stage5_CvxTri", 76u, false, 40u, { 3u, 36u, 38u, 39u });
    bad += runCase("removeContactManagers_Stage5_CvxTri", 76u, true, 1000u, spread(1000u, 137u, 7u));
    bad += runCase("removeContactManagers_Stage5_CvxTri", 76u, true, 20000u, spread(20000u, 4321u, 11u));
    PXG_CHECK_MSG(bad == 0, "removeContactManagers_CvxTri chain matches SparseRemove (%zu mismatches)", bad);
}

} // namespace
