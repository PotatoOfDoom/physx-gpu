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

// P5 Step 4 (BDA, engine launch path): verification of
// broadphaseIncrementalSap.slang (performIncrementalSAP) driven through
// PxgCudaShim EXACTLY as the engine launches it — a REAL PxgBroadPhaseDesc
// filled with live device addresses, ONE bpDesc pointer packed into the
// {8}-byte push blob, and the engine grid (256 x 3, grid-strided; gid.y = axis).
// useEnvIds is derived by the kernel from updateData_envIDs != null (the .cu's
// `envIds ?`).  The kernel writes found/lost pairs in a nondeterministic
// (atomic-compacted) order, so the pair sets and counts (read back from the
// desc's inline sharedFoundPairIndex/sharedLostPairIndex) are compared
// order-independently against a transcription of the exact kernel logic.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t INVALID = 0xffffffffu;

static std::vector<uint8_t> bpShimLoader(const char* file) { return loadShaderBlob(file); }

struct BpDescRig
{
    physx::gpuslang::PxgCudaShim* shim;
    physx::PxgBroadPhaseDesc desc;
    explicit BpDescRig(physx::gpuslang::PxgCudaShim* s) : shim(s) { std::memset(&desc, 0, sizeof(desc)); }
    physx::gpuslang::PxgDevicePtr alloc(const void* data, size_t bytes)
    {
        physx::gpuslang::PxgDevicePtr p = 0;
        shim->memAlloc(&p, bytes);
        if (p && data) shim->memcpyHtoD(p, data, bytes);
        return p;
    }
    physx::gpuslang::PxgDevicePtr upload() { return alloc(&desc, sizeof(desc)); }
};

bool isStart(uint32_t sh) { return (sh & 1u) != 0u; }
uint32_t getHandle(uint32_t sh) { return sh >> 3; }
bool iaIntersects(const uint32_t* a, const uint32_t* b)
{
    return !(b[0] > a[3] || a[0] > b[3] || b[1] > a[4] || a[1] > b[4] || b[2] > a[5] || a[2] > b[5]);
}
bool iaIntersects1D(const uint32_t* a, const uint32_t* b, uint32_t axis)
{
    const uint32_t mx = axis + 3;
    return !(b[axis] > a[mx] || a[axis] > b[mx]);
}
bool iaSeparatedBeforeAxis(uint32_t axis, const uint32_t* a0, const uint32_t* a1)
{
    if (axis == 2)
    {
        if (!iaIntersects1D(a0, a1, 1)) return true;
        if (!iaIntersects1D(a0, a1, 0)) return true;
    }
    else if (axis == 1)
    {
        if (!iaIntersects1D(a0, a1, 0)) return true;
    }
    return false;
}
bool iaFiltering(uint32_t g, uint32_t og, uint32_t e, uint32_t oe)
{
    return (g != og) && ((e == oe) || (e == INVALID) || (oe == INVALID));
}

PXG_TEST(broadphaseIncrementalSap_foundLostPairs)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(16000);

    const uint32_t N = 600;          // #boxes / handles
    const uint32_t nbProj = N * 2;   // projections per axis
    const uint32_t maxPairs = 300000;
    const bool useEnvIds = true;

    std::vector<uint32_t> newB(N * 6), oldB(N * 6);
    std::vector<uint32_t> groups(N), envs(N);
    auto genAabb = [&](uint32_t* dst) {
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            const uint32_t mn = rng() % 900;
            const uint32_t ext = 40 + rng() % 200;
            dst[ax] = mn;
            dst[ax + 3] = mn + ext;
        }
    };
    for (uint32_t b = 0; b < N; ++b)
    {
        genAabb(&newB[b * 6]);
        genAabb(&oldB[b * 6]);
        groups[b] = rng() % 200;
        envs[b] = (rng() % 100 < 15) ? INVALID : (rng() % 8);
    }

    auto makeSorted = [&]() {
        std::vector<uint32_t> proj;
        proj.reserve(nbProj);
        for (uint32_t h = 0; h < N; ++h)
        {
            proj.push_back((h << 3) | 1u);
            proj.push_back((h << 3) | 0u);
        }
        std::shuffle(proj.begin(), proj.end(), rng);
        return proj;
    };
    std::vector<std::vector<uint32_t>> boxHandles0(3), boxHandles1(3);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        boxHandles0[ax] = makeSorted();
        boxHandles1[ax] = makeSorted();
    }

    struct FrameHist
    {
        std::vector<uint32_t> startHist, endHist, startHandles, endHandles;
    };
    auto buildHist = [&](const std::vector<uint32_t>& handles) {
        FrameHist f;
        f.startHist.resize(nbProj);
        f.endHist.resize(nbProj);
        uint32_t accS = 0, accE = 0;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            f.startHist[i] = accS;
            f.endHist[i] = accE;
            const uint32_t sh = handles[i];
            if (isStart(sh)) { f.startHandles.push_back(getHandle(sh)); ++accS; }
            else { f.endHandles.push_back(getHandle(sh)); ++accE; }
        }
        return f;
    };
    std::vector<std::vector<FrameHist>> hist(2, std::vector<FrameHist>(3));
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        hist[0][ax] = buildHist(boxHandles0[ax]);
        hist[1][ax] = buildHist(boxHandles1[ax]);
    }

    std::vector<std::vector<uint32_t>> ranks(3), incComp(3);
    std::vector<uint32_t> total(3, 0);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        ranks[ax].resize(nbProj);
        incComp[ax].resize(nbProj);
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            int32_t delta = int32_t(rng() % 241) - 120;
            int64_t r = int64_t(i) + delta;
            r = std::max<int64_t>(0, std::min<int64_t>(int64_t(nbProj) - 1, r));
            ranks[ax][i] = uint32_t(r);
        }
        uint32_t acc = 0;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            const uint32_t rank = ranks[ax][i];
            const uint32_t sh = boxHandles0[ax][i];
            const bool isStartHandle = isStart(sh);
            const bool isDownPass = i <= rank;
            const uint32_t start = isDownPass ? i : (rank + 1);
            const uint32_t end = isDownPass ? std::min(rank, nbProj - 1) : i;
            const uint32_t frame = isDownPass ? 1u : 0u;
            const std::vector<uint32_t>& accum = isStartHandle ? hist[frame][ax].endHist : hist[frame][ax].startHist;
            const uint32_t c = accum[end] - accum[start];
            incComp[ax][i] = acc;
            acc += c;
        }
        total[ax] = acc;
    }
    std::printf("    [sap] totals per axis: %u %u %u\n", total[0], total[1], total[2]);

    // ---- device: fill a real PxgBroadPhaseDesc ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseIncrementalSap.spv", "performIncrementalSAP", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    std::vector<uint32_t> foundInit(maxPairs * 2, 0xcdcdcdcdu), lostInit(maxPairs * 2, 0xcdcdcdcdu);

    BpDescRig rig(&shim);
    PxgDevicePtr dInc[3], dRanks[3], dBox[3];
    PxgDevicePtr dSH[2][3], dEH[2][3], dSHand[2][3], dEHand[2][3];
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        dInc[ax] = rig.alloc(incComp[ax].data(), nbProj * 4);
        dRanks[ax] = rig.alloc(ranks[ax].data(), nbProj * 4);
        dBox[ax] = rig.alloc(boxHandles0[ax].data(), nbProj * 4);
        for (uint32_t fr = 0; fr < 2; ++fr)
        {
            dSH[fr][ax] = rig.alloc(hist[fr][ax].startHist.data(), nbProj * 4);
            dEH[fr][ax] = rig.alloc(hist[fr][ax].endHist.data(), nbProj * 4);
            dSHand[fr][ax] = rig.alloc(hist[fr][ax].startHandles.data(), std::max<size_t>(1, hist[fr][ax].startHandles.size()) * 4);
            dEHand[fr][ax] = rig.alloc(hist[fr][ax].endHandles.data(), std::max<size_t>(1, hist[fr][ax].endHandles.size()) * 4);
        }
    }
    const PxgDevicePtr dNew = rig.alloc(newB.data(), N * 6 * 4);
    const PxgDevicePtr dOld = rig.alloc(oldB.data(), N * 6 * 4);
    const PxgDevicePtr dGroups = rig.alloc(groups.data(), N * 4);
    const PxgDevicePtr dEnvs = rig.alloc(envs.data(), N * 4);
    const PxgDevicePtr dFound = rig.alloc(foundInit.data(), foundInit.size() * 4);
    const PxgDevicePtr dLost = rig.alloc(lostInit.data(), lostInit.size() * 4);
    if (!dInc[0]) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.numPreviousHandles = N;
    rig.desc.max_found_lost_pairs = maxPairs;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        rig.desc.totalIncrementalComparisons[ax] = total[ax]; // inline desc input
        rig.desc.incrementalComparisons[ax] = reinterpret_cast<PxU32*>(dInc[ax]);
        rig.desc.boxProjectionRanks[ax] = reinterpret_cast<PxU32*>(dRanks[ax]);
        rig.desc.boxHandles[0][ax] = reinterpret_cast<PxU32*>(dBox[ax]);
        for (uint32_t fr = 0; fr < 2; ++fr)
        {
            rig.desc.startPtHistogram[fr][ax] = reinterpret_cast<PxU32*>(dSH[fr][ax]);
            rig.desc.endPtHistogram[fr][ax] = reinterpret_cast<PxU32*>(dEH[fr][ax]);
            rig.desc.startPointHandles[fr][ax] = reinterpret_cast<PxU32*>(dSHand[fr][ax]);
            rig.desc.endPointHandles[fr][ax] = reinterpret_cast<PxU32*>(dEHand[fr][ax]);
        }
    }
    rig.desc.newIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dNew);
    rig.desc.oldIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dOld);
    rig.desc.updateData_groups = reinterpret_cast<PxU32*>(dGroups);
    rig.desc.updateData_envIDs = useEnvIds ? reinterpret_cast<PxU32*>(dEnvs) : nullptr;
    rig.desc.foundPairReport = reinterpret_cast<PxgBroadPhasePair*>(dFound);
    rig.desc.lostPairReport = reinterpret_cast<PxgBroadPhasePair*>(dLost);
    // sharedFoundPairIndex / sharedLostPairIndex left 0 (memset)
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grid: BP_INCREMENTAL_SAP = 256 blocks x (gridY = 3 axes) x 256 threads
    PXG_CHECK_MSG(shim.launchKernel(0, 256, 3, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch performIncrementalSAP");
    shim.streamSynchronize();

    uint32_t gFound = 0, gLost = 0;
    shim.memcpyDtoH(&gFound, dDesc + offsetof(PxgBroadPhaseDesc, sharedFoundPairIndex), 4);
    shim.memcpyDtoH(&gLost, dDesc + offsetof(PxgBroadPhaseDesc, sharedLostPairIndex), 4);
    std::vector<uint32_t> gFoundPairs(std::min(gFound, maxPairs) * 2), gLostPairs(std::min(gLost, maxPairs) * 2);
    if (!gFoundPairs.empty()) shim.memcpyDtoH(gFoundPairs.data(), dFound, gFoundPairs.size() * 4);
    if (!gLostPairs.empty()) shim.memcpyDtoH(gLostPairs.data(), dLost, gLostPairs.size() * 4);

    // ---- CPU transcription of the exact kernel logic ----
    auto binarySearch = [](const std::vector<uint32_t>& data, uint32_t n, uint32_t value) -> uint32_t {
        uint32_t left = 0, right = n;
        while (left < right)
        {
            uint32_t pos = (left + right) / 2;
            if (data[pos] <= value) left = pos + 1;
            else right = pos;
        }
        return left ? left - 1 : 0;
    };
    std::vector<uint64_t> refFound, refLost;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        for (uint32_t workId = 0; workId < total[ax]; ++workId)
        {
            const uint32_t index = binarySearch(incComp[ax], nbProj, workId);
            const uint32_t sh = boxHandles0[ax][index];
            const uint32_t isStartHandle = isStart(sh) ? 1u : 0u;
            const uint32_t handle = getHandle(sh);
            const uint32_t groupId = groups[handle];
            const uint32_t envId = useEnvIds ? envs[handle] : INVALID;
            const uint32_t offset = workId - incComp[ax][index];
            const uint32_t prevIndex = ranks[ax][index];
            const uint32_t downPass = prevIndex > index ? 1u : 0u;
            const uint32_t startIndex = downPass ? index : (prevIndex + 1);
            const uint32_t willFindPair = (1u - isStartHandle) ^ downPass;
            const uint32_t frame = downPass;
            const std::vector<uint32_t>& accum = isStartHandle ? hist[frame][ax].endHist : hist[frame][ax].startHist;
            const std::vector<uint32_t>& hnd = isStartHandle ? hist[frame][ax].endHandles : hist[frame][ax].startHandles;
            const uint32_t handleIndex = accum[startIndex] + offset;
            const uint32_t otherHandle = hnd[handleIndex];
            const uint32_t otherGroupId = groups[otherHandle];
            const uint32_t otherEnvId = useEnvIds ? envs[otherHandle] : INVALID;
            if (!iaFiltering(groupId, otherGroupId, envId, otherEnvId)) continue;
            const bool wfp = willFindPair != 0u;
            const uint32_t* aabb = &(wfp ? newB : oldB)[handle * 6];
            const uint32_t* oldAABB = &(wfp ? oldB : newB)[handle * 6];
            const uint32_t* otherAABB = &(wfp ? newB : oldB)[otherHandle * 6];
            const uint32_t* otherOld = &(wfp ? oldB : newB)[otherHandle * 6];
            if (iaIntersects(aabb, otherAABB) && !iaIntersects1D(oldAABB, otherOld, ax) &&
                !iaSeparatedBeforeAxis(ax, oldAABB, otherOld))
            {
                const uint32_t id = 1u - willFindPair;
                const uint32_t lo = std::min(handle, otherHandle), hi = std::max(handle, otherHandle);
                const uint64_t key = (uint64_t(lo) << 32) | hi;
                (id == 0 ? refFound : refLost).push_back(key);
            }
        }
    }

    PXG_CHECK_MSG(gFound == refFound.size(), "found count: gpu=%u ref=%zu", gFound, refFound.size());
    PXG_CHECK_MSG(gLost == refLost.size(), "lost count: gpu=%u ref=%zu", gLost, refLost.size());
    PXG_CHECK_MSG(gFound > 0 && gLost > 0, "expected nonzero found (%u) and lost (%u) pairs", gFound, gLost);

    auto toKeys = [](const std::vector<uint32_t>& flat, uint32_t count) {
        std::vector<uint64_t> k(count);
        for (uint32_t i = 0; i < count; ++i) k[i] = (uint64_t(flat[i * 2]) << 32) | flat[i * 2 + 1];
        return k;
    };
    std::vector<uint64_t> gF = toKeys(gFoundPairs, std::min(gFound, maxPairs));
    std::vector<uint64_t> gL = toKeys(gLostPairs, std::min(gLost, maxPairs));
    std::sort(gF.begin(), gF.end());
    std::sort(gL.begin(), gL.end());
    std::sort(refFound.begin(), refFound.end());
    std::sort(refLost.begin(), refLost.end());
    PXG_CHECK_MSG(gF == refFound, "found pair set mismatch (gpu %zu vs ref %zu)", gF.size(), refFound.size());
    PXG_CHECK_MSG(gL == refLost, "lost pair set mismatch (gpu %zu vs ref %zu)", gL.size(), refLost.size());

    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        for (PxgDevicePtr p : { dInc[ax], dRanks[ax], dBox[ax] }) shim.memFree(p);
        for (uint32_t fr = 0; fr < 2; ++fr)
        {
            shim.memFree(dSH[fr][ax]); shim.memFree(dEH[fr][ax]);
            shim.memFree(dSHand[fr][ax]); shim.memFree(dEHand[fr][ax]);
        }
    }
    for (PxgDevicePtr p : { dNew, dOld, dGroups, dEnvs, dFound, dLost, dDesc }) shim.memFree(p);
}

} // namespace
