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

// Verification of broadphaseNewBoundsPairs.slang (generateFoundPairsForNew
// BoundsRegion), the final region kernel — P5 Step 4: driven through
// PxgCudaShim on the ENGINE ABI.  The whole one-shot region pipeline is built
// on the CPU (the same reference logic verified by tranches 17-20:
// createRegions counts, the region prefix scans, the active-region flag/scan/
// scatter, and the overlap-check counts + scan) so every input is consistent
// and in bounds; a REAL engine PxgBroadPhaseDesc is filled with the live
// device addresses (including the u64 overlapChecksTotalRegion VALUE and
// sharedFoundPairIndex = 0) and the kernel launches with the one bpDesc
// pointer on the engine's 64-block grid.  Its atomic-compacted found-pair
// output (nondeterministic order) is compared order-independently against a
// transcription of the exact kernel logic; the atomic pair count is verified
// by reading the desc back (the kernel InterlockedAdds the desc's
// sharedFoundPairIndex field).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t TOTAL_REGION_SIZE = 64;
const uint32_t INVALID = 0xffffffffu;

uint32_t getHandle(uint32_t sh) { return sh >> 3; }
uint32_t isNewP(uint32_t sh) { return (sh >> 1) & 1u; }

uint32_t bsearch32(const std::vector<uint32_t>& d, uint32_t n, uint32_t v)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) { uint32_t p = (lo + hi) / 2; if (d[p] <= v) lo = p + 1; else hi = p; }
    return lo ? lo - 1 : 0;
}
uint32_t bsearch64(const std::vector<uint64_t>& d, uint32_t n, uint64_t v)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) { uint32_t p = (lo + hi) / 2; if (d[p] <= v) lo = p + 1; else hi = p; }
    return lo ? lo - 1 : 0;
}

static std::vector<uint8_t> nbpShimLoader(const char* file) { return loadShaderBlob(file); }

// Fills a REAL engine PxgBroadPhaseDesc with live device addresses (same rig
// as testBroadphaseSetup.cpp).
struct BpDescRig
{
    physx::gpuslang::PxgCudaShim* shim;
    physx::PxgBroadPhaseDesc desc;

    explicit BpDescRig(physx::gpuslang::PxgCudaShim* s) : shim(s) { std::memset(&desc, 0, sizeof(desc)); }

    physx::gpuslang::PxgDevicePtr alloc(const void* data, size_t bytes)
    {
        physx::gpuslang::PxgDevicePtr p = 0;
        shim->memAlloc(&p, bytes);
        if (p && data)
            shim->memcpyHtoD(p, data, bytes);
        return p;
    }
    physx::gpuslang::PxgDevicePtr upload()
    {
        physx::gpuslang::PxgDevicePtr p = alloc(&desc, sizeof(desc));
        return p;
    }
};

PXG_TEST(broadphaseNewBoundsPairs_foundPairs)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(21010);
    const uint32_t N = 500;             // created boxes
    const uint32_t nbProj = N * 2;
    const uint32_t maxPairs = 500000;
    const uint32_t useEnvIds = 1;       // desc.updateData_envIDs non-null (the .cu pointer branch)

    // ---- box set: region spans [0,3], X arrangement, bounds, groups, envs ----
    std::vector<uint32_t> regionRange(N * 8, 0), sapBoxX(N * 2), handlesX(nbProj);
    std::vector<uint32_t> newBounds(N * 6), groups(N), envs(N);
    std::vector<uint32_t> yzCount(N), fullCount(N);
    {
        std::vector<uint32_t> perm(nbProj);
        for (uint32_t i = 0; i < nbProj; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t b = 0; b < N; ++b)
        {
            auto span = [&](uint32_t& s, uint32_t& e) { s = rng() % 4; e = s + rng() % (4 - s); };
            uint32_t sX, eX, sY, eY, sZ, eZ;
            span(sX, eX); span(sY, eY); span(sZ, eZ);
            regionRange[b * 8 + 0] = sX; regionRange[b * 8 + 1] = sY; regionRange[b * 8 + 2] = sZ;
            regionRange[b * 8 + 4] = eX; regionRange[b * 8 + 5] = eY; regionRange[b * 8 + 6] = eZ;
            yzCount[b] = (eY - sY + 1) * (eZ - sZ + 1);
            fullCount[b] = (eX - sX + 1) * yzCount[b];
            uint32_t mn = std::min(perm[2 * b], perm[2 * b + 1]);
            uint32_t mx = std::max(perm[2 * b], perm[2 * b + 1]);
            sapBoxX[b * 2 + 0] = mn; sapBoxX[b * 2 + 1] = mx;
            const uint32_t newBit = (rng() % 100 < 50) ? 1u : 0u;
            handlesX[mn] = (b << 3) | 1u | (newBit << 1);
            handlesX[mx] = (b << 3) | 0u;
            // integer bounds in a small space so many pairs intersect
            for (uint32_t ax = 0; ax < 3; ++ax)
            {
                uint32_t lo = rng() % 200, ext = 30 + rng() % 120;
                newBounds[b * 6 + ax] = lo;
                newBounds[b * 6 + ax + 3] = lo + ext;
            }
            groups[b] = rng() % 64;
            envs[b] = (rng() % 100 < 20) ? INVALID : (rng() % 6);
        }
    }

    // ---- createRegions counts + region prefix scans ----
    std::vector<uint32_t> startRegionAccum(nbProj), regionAccum(nbProj);
    uint32_t sAcc = 0, rAcc = 0;
    for (uint32_t p = 0; p < nbProj; ++p)
    {
        startRegionAccum[p] = sAcc; regionAccum[p] = rAcc;
        const uint32_t sh = handlesX[p];
        if (sh & 1u) { sAcc += yzCount[getHandle(sh)]; rAcc += fullCount[getHandle(sh)]; }
    }
    const uint32_t startRegionAccumTotal = sAcc;
    const uint32_t regionAccumTotal = rAcc;

    // ---- active-region: flags -> offset tables -> ordered handles ----
    std::vector<uint32_t> startHist(TOTAL_REGION_SIZE * nbProj, 0), activeHist(TOTAL_REGION_SIZE * nbProj, 0);
    std::vector<uint32_t> ordStart(TOTAL_REGION_SIZE * nbProj, INVALID), ordActive(TOTAL_REGION_SIZE * nbProj, INVALID);
    // fixed-X (y,z) region decode used by the active-region kernels
    auto decodeYZ = [&](uint32_t pos, uint32_t offset) {
        const uint32_t h = getHandle(handlesX[pos]);
        const uint32_t sX = regionRange[h * 8 + 0], sY = regionRange[h * 8 + 1], sZ = regionRange[h * 8 + 2];
        const uint32_t eZ = regionRange[h * 8 + 6];
        const uint32_t nbZ = (eZ - sZ) + 1;
        const uint32_t zR = (offset % nbZ) + sZ, yR = (offset / nbZ) + sY;
        return (sX + yR * 4 + zR * 16) * nbProj;
    };
    for (uint32_t wi = 0; wi < startRegionAccumTotal; ++wi)
    {
        const uint32_t pos = bsearch32(startRegionAccum, nbProj, wi);
        const uint32_t rb = decodeYZ(pos, wi - startRegionAccum[pos]);
        startHist[rb + pos] = 1;
        activeHist[rb + pos] = isNewP(handlesX[pos]);
    }
    for (uint32_t r = 0; r < TOTAL_REGION_SIZE; ++r)
    {
        uint32_t sa = 0, aa = 0;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            uint32_t s = startHist[r * nbProj + i], a = activeHist[r * nbProj + i];
            startHist[r * nbProj + i] = sa; activeHist[r * nbProj + i] = aa;
            sa += s; aa += a;
        }
    }
    for (uint32_t wi = 0; wi < startRegionAccumTotal; ++wi)
    {
        const uint32_t pos = bsearch32(startRegionAccum, nbProj, wi);
        const uint32_t rb = decodeYZ(pos, wi - startRegionAccum[pos]);
        const uint32_t h = getHandle(handlesX[pos]);
        const uint32_t spi = sapBoxX[h * 2 + 0];
        ordStart[rb + startHist[rb + spi]] = h;
        if (isNewP(handlesX[pos])) ordActive[rb + activeHist[rb + spi]] = h;
    }

    // ---- overlap-check: counts + handle/region records + 64-bit scan ----
    std::vector<uint64_t> overlapChecks(regionAccumTotal);
    std::vector<uint32_t> handleReg(regionAccumTotal * 2);
    // full x/y/z region decode used by the overlap-check kernel
    auto decodeXYZ = [&](uint32_t bs, uint32_t offset, uint32_t& outHandle) {
        const uint32_t h = getHandle(handlesX[bs]);
        outHandle = h;
        const uint32_t sX = regionRange[h * 8 + 0], sY = regionRange[h * 8 + 1], sZ = regionRange[h * 8 + 2];
        const uint32_t eY = regionRange[h * 8 + 5], eZ = regionRange[h * 8 + 6];
        const uint32_t nbZ = (eZ - sZ) + 1, nbY = (eY - sY) + 1;
        const uint32_t zR = (offset % nbZ) + sZ, yR = ((offset / nbZ) % nbY) + sY, xR = (offset / (nbZ * nbY)) + sX;
        return (xR + yR * 4 + zR * 16) * nbProj;
    };
    for (uint32_t wi = 0; wi < regionAccumTotal; ++wi)
    {
        uint32_t h;
        const uint32_t bs = bsearch32(regionAccum, nbProj, wi);
        const uint32_t rb = decodeXYZ(bs, wi - regionAccum[bs], h);
        const uint32_t be = sapBoxX[h * 2 + 1];
        const std::vector<uint32_t>& tbl = isNewP(handlesX[bs]) ? startHist : activeHist;
        overlapChecks[wi] = uint64_t(tbl[rb + be] - tbl[rb + bs + 1]);
        handleReg[wi * 2 + 0] = h;
        handleReg[wi * 2 + 1] = rb;
    }
    uint64_t oAcc = 0;
    for (uint32_t wi = 0; wi < regionAccumTotal; ++wi) { uint64_t c = overlapChecks[wi]; overlapChecks[wi] = oAcc; oAcc += c; }
    const uint64_t overlapChecksTotalRegion = oAcc;

    std::printf("    [nbp] startTotal=%u regionAccumTotal=%u overlapTotal=%llu\n", startRegionAccumTotal,
                regionAccumTotal, (unsigned long long) overlapChecksTotalRegion);

    // ---- reference: transcribe generateFoundPairs ----
    std::vector<uint64_t> refPairs;
    for (uint64_t wi = 0; wi < overlapChecksTotalRegion; ++wi)
    {
        const uint32_t index = bsearch64(overlapChecks, regionAccumTotal, wi);
        const uint32_t handle = handleReg[index * 2 + 0];
        const uint32_t regionIndex = handleReg[index * 2 + 1];
        const uint32_t bsi = sapBoxX[handle * 2 + 0], bei = sapBoxX[handle * 2 + 1];
        const uint32_t isNew = isNewP(handlesX[bsi]);
        const uint64_t offset = wi - overlapChecks[index];
        const std::vector<uint32_t>& hist = isNew ? startHist : activeHist;
        const std::vector<uint32_t>& ord = isNew ? ordStart : ordActive;
        const uint32_t rStart = hist[regionIndex + bsi + 1];
        const uint32_t rEnd = hist[regionIndex + bei];
        if (rEnd > rStart)
        {
            const uint64_t oi = uint64_t(rStart) + offset;
            if (oi < uint64_t(rEnd))
            {
                const uint32_t otherHandle = ord[regionIndex + uint32_t(oi)];
                const uint32_t x = std::max(regionRange[handle * 8 + 0], regionRange[otherHandle * 8 + 0]);
                const uint32_t y = std::max(regionRange[handle * 8 + 1], regionRange[otherHandle * 8 + 1]);
                const uint32_t z = std::max(regionRange[handle * 8 + 2], regionRange[otherHandle * 8 + 2]);
                const uint32_t otherRegionIndex = (x + y * 4 + z * 16) * nbProj;
                if (regionIndex == otherRegionIndex)
                {
                    const uint32_t g = groups[handle], og = groups[otherHandle];
                    const uint32_t e = useEnvIds ? envs[handle] : INVALID, oe = useEnvIds ? envs[otherHandle] : INVALID;
                    const bool filt = (g != og) && ((e == oe) || e == INVALID || oe == INVALID);
                    if (filt)
                    {
                        const uint32_t* a = &newBounds[handle * 6];
                        const uint32_t* bb = &newBounds[otherHandle * 6];
                        const bool inter = !(bb[0] > a[3] || a[0] > bb[3] || bb[1] > a[4] || a[1] > bb[4] ||
                                             bb[2] > a[5] || a[2] > bb[5]);
                        if (inter)
                        {
                            // PxgBroadPhasePair canonicalizes to (min, max)
                            const uint32_t lo = std::min(handle, otherHandle), hi = std::max(handle, otherHandle);
                            refPairs.push_back((uint64_t(lo) << 32) | hi);
                        }
                    }
                }
            }
        }
    }

    // ---- device: fill a REAL PxgBroadPhaseDesc and launch on the engine ABI ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseNewBoundsPairs.spv", "generateFoundPairsForNewBoundsRegion", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &nbpShimLoader);

    std::vector<uint32_t> foundInit(maxPairs * 2, 0xcdcdcdcdu);

    BpDescRig rig(&shim);
    const PxgDevicePtr dOv = rig.alloc(overlapChecks.data(), overlapChecks.size() * 8);
    const PxgDevicePtr dHR = rig.alloc(handleReg.data(), handleReg.size() * 4);
    const PxgDevicePtr dRange = rig.alloc(regionRange.data(), regionRange.size() * 4);
    const PxgDevicePtr dSap = rig.alloc(sapBoxX.data(), sapBoxX.size() * 4);
    const PxgDevicePtr dH = rig.alloc(handlesX.data(), handlesX.size() * 4);
    const PxgDevicePtr dSH = rig.alloc(startHist.data(), startHist.size() * 4);
    const PxgDevicePtr dAH = rig.alloc(activeHist.data(), activeHist.size() * 4);
    const PxgDevicePtr dOS = rig.alloc(ordStart.data(), ordStart.size() * 4);
    const PxgDevicePtr dOA = rig.alloc(ordActive.data(), ordActive.size() * 4);
    const PxgDevicePtr dNB = rig.alloc(newBounds.data(), newBounds.size() * 4);
    const PxgDevicePtr dG = rig.alloc(groups.data(), groups.size() * 4);
    const PxgDevicePtr dE = rig.alloc(envs.data(), envs.size() * 4);
    const PxgDevicePtr dFound = rig.alloc(foundInit.data(), foundInit.size() * 4);
    if (!dOv) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // numHandles = numPreviousHandles + numCreatedHandles - numRemovedHandles = N
    rig.desc.numPreviousHandles = N;
    rig.desc.overlapChecksRegion = reinterpret_cast<regionOverlapType*>(dOv);
    rig.desc.overlapChecksHandleRegiones = reinterpret_cast<PxgHandleRegion*>(dHR);
    rig.desc.regionRange = reinterpret_cast<PxgIntegerRegion*>(dRange);
    rig.desc.boxSapBox1D[0] = reinterpret_cast<PxgSapBox1D*>(dSap);
    rig.desc.boxHandles[0][0] = reinterpret_cast<PxU32*>(dH);
    rig.desc.startRegionsHistogram = reinterpret_cast<PxU32*>(dSH);
    rig.desc.activeRegionsHistogram = reinterpret_cast<PxU32*>(dAH);
    rig.desc.orderedStartRegionHandles = reinterpret_cast<PxU32*>(dOS);
    rig.desc.orderedActiveRegionHandles = reinterpret_cast<PxU32*>(dOA);
    rig.desc.newIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dNB);
    rig.desc.updateData_groups = reinterpret_cast<PxU32*>(dG);
    rig.desc.updateData_envIDs = useEnvIds ? reinterpret_cast<PxU32*>(dE) : NULL;
    rig.desc.foundPairReport = reinterpret_cast<PxgBroadPhasePair*>(dFound);
    rig.desc.sharedFoundPairIndex = 0;
    rig.desc.max_found_lost_pairs = maxPairs;
    rig.desc.regionAccumTotal = regionAccumTotal;
    rig.desc.overlapChecksTotalRegion = overlapChecksTotalRegion; // u64 VALUE in the desc
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grid: BP_GENERATE_FOUNDPAIR_NEWBOUNDS = 64 blocks x 256
    PXG_CHECK_MSG(shim.launchKernel(0, 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch generateFoundPairs");
    shim.streamSynchronize();

    // the atomic pair count lives in the desc itself — read it back
    PxgBroadPhaseDesc back;
    std::memset(&back, 0, sizeof(back));
    PXG_CHECK_MSG(shim.memcpyDtoH(&back, dDesc, sizeof(back)) == PxgResult::Success, "DtoH desc");
    const uint32_t gFound = back.sharedFoundPairIndex;

    std::vector<uint32_t> gPairsFlat(std::min(gFound, maxPairs) * 2);
    if (!gPairsFlat.empty())
        shim.memcpyDtoH(gPairsFlat.data(), dFound, gPairsFlat.size() * 4);

    PXG_CHECK_MSG(gFound == refPairs.size(), "found count: gpu=%u ref=%zu", gFound, refPairs.size());
    PXG_CHECK_MSG(gFound > 0, "expected nonzero found pairs");

    std::vector<uint64_t> gPairs(std::min(gFound, maxPairs));
    for (uint32_t i = 0; i < gPairs.size(); ++i)
        gPairs[i] = (uint64_t(gPairsFlat[i * 2]) << 32) | gPairsFlat[i * 2 + 1];
    std::sort(gPairs.begin(), gPairs.end());
    std::sort(refPairs.begin(), refPairs.end());
    PXG_CHECK_MSG(gPairs == refPairs, "found pair set mismatch (gpu %zu vs ref %zu)", gPairs.size(), refPairs.size());

    for (PxgDevicePtr pfree : { dOv, dHR, dRange, dSap, dH, dSH, dAH, dOS, dOA, dNB, dG, dE, dFound, dDesc })
        shim.memFree(pfree);
}

} // namespace
