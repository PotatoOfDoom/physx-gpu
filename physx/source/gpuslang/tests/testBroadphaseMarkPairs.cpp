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

// P5 Step 4 (BDA, engine launch path): verification of broadphaseMarkPairs.slang
// (markRemovedPairsLaunch, markRemovedPairsProjectionsLaunch,
// markUpdatedPairsLaunch, markUpdatedPairsLaunch2) driven through PxgCudaShim
// EXACTLY as the engine launches them — a REAL PxgBroadPhaseDesc filled with
// live device addresses, ONE bpDesc pointer packed into the {8}-byte push blob,
// and the engine grid (32 x 256, grid-strided).  Every output is compared
// bit-exactly against a C++ transcription of the CUDA kernels; a mutation check
// confirms the kernel actually wrote (guards against a no-op/unbound launch).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t INVALID = 0xffffffffu;

struct IntegerAABB
{
    uint32_t mMinMax[6];
};
struct SapBox1D
{
    uint32_t mMinMax[2];
};

static std::vector<uint8_t> bpShimLoader(const char* file) { return loadShaderBlob(file); }

// Fills a REAL PxgBroadPhaseDesc with live device addresses and uploads it, so
// the kernel is launched exactly as the engine launches it (one bpDesc ptr).
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
    physx::gpuslang::PxgDevicePtr upload() { return alloc(&desc, sizeof(desc)); }
};

// per-axis SAP arrangement: sapBox[axis][b]={minPos,maxPos} partitioning [0,2N)
void genArrangement(std::mt19937& rng, uint32_t N, std::vector<uint32_t> sapBox[3])
{
    const uint32_t nbProj = N * 2;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        sapBox[ax].assign(N * 2, 0);
        std::vector<uint32_t> perm(nbProj);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t b = 0; b < N; ++b)
        {
            sapBox[ax][b * 2 + 0] = std::min(perm[2 * b], perm[2 * b + 1]);
            sapBox[ax][b * 2 + 1] = std::max(perm[2 * b], perm[2 * b + 1]);
        }
    }
}

std::vector<uint32_t> genBounds(std::mt19937& rng, uint32_t N)
{
    std::vector<uint32_t> b(N * 6);
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t lo = rng() % 100000, ext = 1 + rng() % 50000;
            b[i * 6 + ax] = lo;
            b[i * 6 + ax + 3] = lo + ext;
        }
    return b;
}

PXG_TEST(broadphaseMarkPairs_removed)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(22010);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox[3];
    genArrangement(rng, N, sapBox);

    // previous-frame handles (boxHandles[1][ax]): (handle<<3)|isStart at positions
    std::vector<uint32_t> handles1[3];
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        handles1[ax].assign(nbProj, 0);
        for (uint32_t b = 0; b < N; ++b)
        {
            uint32_t mn = sapBox[ax][b * 2 + 0], mx = sapBox[ax][b * 2 + 1];
            handles1[ax][mn] = (b << 3) | 1u;
            handles1[ax][mx] = (b << 3) | 0u;
        }
    }
    std::vector<uint32_t> oldBounds = genBounds(rng, N);

    std::vector<uint32_t> removed;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 30) removed.push_back(b);
    const uint32_t numRem = uint32_t(removed.size());

    // reference
    std::vector<uint32_t> rHandles[3] = { handles1[0], handles1[1], handles1[2] };
    std::vector<uint32_t> rOld = oldBounds;
    for (uint32_t h : removed)
    {
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[ax][h * 2 + 0], mx = sapBox[ax][h * 2 + 1];
            rHandles[ax][mn] |= (1u << 2);
            rHandles[ax][mx] |= (1u << 2);
        }
        rOld[h * 6 + 0] = rOld[h * 6 + 1] = rOld[h * 6 + 2] = 0xff7fffffu;
        rOld[h * 6 + 3] = rOld[h * 6 + 4] = rOld[h * 6 + 5] = 0x00800000u;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseMarkPairs.spv", "markRemovedPairsLaunch", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    BpDescRig rig(&shim);
    const PxgDevicePtr dRem = rig.alloc(removed.data(), std::max<size_t>(1, removed.size()) * 4);
    const PxgDevicePtr dSap[3] = { rig.alloc(sapBox[0].data(), N * 2 * 4), rig.alloc(sapBox[1].data(), N * 2 * 4),
                                   rig.alloc(sapBox[2].data(), N * 2 * 4) };
    const PxgDevicePtr dH1[3] = { rig.alloc(handles1[0].data(), nbProj * 4), rig.alloc(handles1[1].data(), nbProj * 4),
                                  rig.alloc(handles1[2].data(), nbProj * 4) };
    const PxgDevicePtr dOld = rig.alloc(oldBounds.data(), N * 6 * 4);
    if (!dRem) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.numRemovedHandles = numRem;
    rig.desc.updateData_removedHandles = reinterpret_cast<PxU32*>(dRem);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        rig.desc.boxSapBox1D[ax] = reinterpret_cast<PxgSapBox1D*>(dSap[ax]);
        rig.desc.boxHandles[1][ax] = reinterpret_cast<PxU32*>(dH1[ax]);
    }
    rig.desc.oldIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dOld);
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch markRemovedPairs");
    shim.streamSynchronize();

    size_t hM = 0, oM = 0, mutated = 0;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> got(nbProj);
        shim.memcpyDtoH(got.data(), dH1[ax], nbProj * 4);
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            if (got[i] != rHandles[ax][i]) ++hM;
            if (got[i] != handles1[ax][i]) ++mutated;
        }
    }
    std::vector<uint32_t> gOld(N * 6);
    shim.memcpyDtoH(gOld.data(), dOld, N * 6 * 4);
    for (uint32_t i = 0; i < N * 6; ++i) if (gOld[i] != rOld[i]) ++oM;
    PXG_CHECK_MSG(hM == 0, "%zu handle-delete mismatches (numRem=%u)", hM, numRem);
    PXG_CHECK_MSG(oM == 0, "%zu old-bounds mismatches", oM);
    PXG_CHECK_MSG(numRem == 0 || mutated > 0, "kernel did not mutate any handle (unbound launch?)");

    for (PxgDevicePtr p : { dRem, dSap[0], dSap[1], dSap[2], dH1[0], dH1[1], dH1[2], dOld, dDesc }) shim.memFree(p);
}

PXG_TEST(broadphaseMarkPairs_removedProjections)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(22020);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox[3];
    genArrangement(rng, N, sapBox);
    std::vector<uint32_t> proj[3];
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        proj[ax].resize(nbProj);
        for (uint32_t i = 0; i < nbProj; ++i) proj[ax][i] = rng();
    }

    std::vector<uint32_t> removed;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 30) removed.push_back(b);
    const uint32_t numRem = uint32_t(removed.size());

    std::vector<uint32_t> rProj[3] = { proj[0], proj[1], proj[2] };
    std::vector<uint32_t> rSap[3] = { sapBox[0], sapBox[1], sapBox[2] };
    for (uint32_t h : removed)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[ax][h * 2 + 0], mx = sapBox[ax][h * 2 + 1];
            rProj[ax][mn] = INVALID;
            rProj[ax][mx] = INVALID;
            rSap[ax][h * 2 + 0] = INVALID;
            rSap[ax][h * 2 + 1] = INVALID;
        }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseMarkPairs.spv", "markRemovedPairsProjectionsLaunch", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    BpDescRig rig(&shim);
    const PxgDevicePtr dRem = rig.alloc(removed.data(), std::max<size_t>(1, removed.size()) * 4);
    const PxgDevicePtr dSap[3] = { rig.alloc(sapBox[0].data(), N * 2 * 4), rig.alloc(sapBox[1].data(), N * 2 * 4),
                                   rig.alloc(sapBox[2].data(), N * 2 * 4) };
    const PxgDevicePtr dPr[3] = { rig.alloc(proj[0].data(), nbProj * 4), rig.alloc(proj[1].data(), nbProj * 4),
                                  rig.alloc(proj[2].data(), nbProj * 4) };
    if (!dRem) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.numRemovedHandles = numRem;
    rig.desc.updateData_removedHandles = reinterpret_cast<PxU32*>(dRem);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        rig.desc.boxSapBox1D[ax] = reinterpret_cast<PxgSapBox1D*>(dSap[ax]);
        rig.desc.boxProjections[ax] = reinterpret_cast<PxU32*>(dPr[ax]);
    }
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch markRemovedProj");
    shim.streamSynchronize();

    size_t pM = 0, sM = 0, mutated = 0;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> gp(nbProj), gs(N * 2);
        shim.memcpyDtoH(gp.data(), dPr[ax], nbProj * 4);
        shim.memcpyDtoH(gs.data(), dSap[ax], N * 2 * 4);
        for (uint32_t i = 0; i < nbProj; ++i) { if (gp[i] != rProj[ax][i]) ++pM; if (gp[i] != proj[ax][i]) ++mutated; }
        for (uint32_t i = 0; i < N * 2; ++i) if (gs[i] != rSap[ax][i]) ++sM;
    }
    PXG_CHECK_MSG(pM == 0, "%zu projection-invalidate mismatches (numRem=%u)", pM, numRem);
    PXG_CHECK_MSG(sM == 0, "%zu sapbox-invalidate mismatches", sM);
    PXG_CHECK_MSG(numRem == 0 || mutated > 0, "kernel did not mutate any projection (unbound launch?)");

    for (PxgDevicePtr p : { dRem, dSap[0], dSap[1], dSap[2], dPr[0], dPr[1], dPr[2], dDesc }) shim.memFree(p);
}

PXG_TEST(broadphaseMarkPairs_updated)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(22030);
    const uint32_t words = 40, N = words * 32, nbProj = N * 2; // numElements == N
    std::vector<uint32_t> sapBox[3];
    genArrangement(rng, N, sapBox);
    std::vector<uint32_t> newB = genBounds(rng, N);
    std::vector<uint32_t> changed(words), added(words), removedM(words);
    for (uint32_t w = 0; w < words; ++w) { changed[w] = rng(); added[w] = rng(); removedM[w] = rng(); }

    const uint32_t SENT = 0xdeadbeefu;
    std::vector<uint32_t> proj[3] = { std::vector<uint32_t>(nbProj, SENT), std::vector<uint32_t>(nbProj, SENT),
                                      std::vector<uint32_t>(nbProj, SENT) };
    std::vector<uint32_t> rProj[3] = { proj[0], proj[1], proj[2] };
    for (uint32_t handle = 0; handle < N; ++handle)
    {
        uint32_t w = handle / 32;
        if (((changed[w] & ~(added[w] | removedM[w])) & (1u << (handle & 31))) != 0u)
            for (uint32_t ax = 0; ax < 3; ++ax)
            {
                uint32_t mn = sapBox[ax][handle * 2 + 0], mx = sapBox[ax][handle * 2 + 1];
                rProj[ax][mn] = newB[handle * 6 + ax];
                rProj[ax][mx] = newB[handle * 6 + ax + 3];
            }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseMarkPairs.spv", "markUpdatedPairsLaunch", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    BpDescRig rig(&shim);
    const PxgDevicePtr dSap[3] = { rig.alloc(sapBox[0].data(), N * 2 * 4), rig.alloc(sapBox[1].data(), N * 2 * 4),
                                   rig.alloc(sapBox[2].data(), N * 2 * 4) };
    const PxgDevicePtr dPr[3] = { rig.alloc(proj[0].data(), nbProj * 4), rig.alloc(proj[1].data(), nbProj * 4),
                                  rig.alloc(proj[2].data(), nbProj * 4) };
    const PxgDevicePtr dNew = rig.alloc(newB.data(), N * 6 * 4);
    const PxgDevicePtr dCh = rig.alloc(changed.data(), words * 4);
    const PxgDevicePtr dAdd = rig.alloc(added.data(), words * 4);
    const PxgDevicePtr dRm = rig.alloc(removedM.data(), words * 4);
    if (!dSap[0]) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.aabbMngr_changedHandleBitMapWordCounts = words;
    rig.desc.aabbMngr_changedHandleMap = reinterpret_cast<PxU32*>(dCh);
    rig.desc.aabbMngr_addedHandleMap = reinterpret_cast<PxU32*>(dAdd);
    rig.desc.aabbMngr_removedHandleMap = reinterpret_cast<PxU32*>(dRm);
    rig.desc.newIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dNew);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        rig.desc.boxSapBox1D[ax] = reinterpret_cast<PxgSapBox1D*>(dSap[ax]);
        rig.desc.boxProjections[ax] = reinterpret_cast<PxU32*>(dPr[ax]);
    }
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch markUpdatedPairs");
    shim.streamSynchronize();

    size_t pM = 0, mutated = 0;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> gp(nbProj);
        shim.memcpyDtoH(gp.data(), dPr[ax], nbProj * 4);
        for (uint32_t i = 0; i < nbProj; ++i) { if (gp[i] != rProj[ax][i]) ++pM; if (gp[i] != SENT) ++mutated; }
    }
    PXG_CHECK_MSG(pM == 0, "%zu updated-projection mismatches", pM);
    PXG_CHECK_MSG(mutated > 0, "kernel did not mutate any projection (unbound launch?)");

    for (PxgDevicePtr p : { dSap[0], dSap[1], dSap[2], dPr[0], dPr[1], dPr[2], dNew, dCh, dAdd, dRm, dDesc })
        shim.memFree(p);
}

PXG_TEST(broadphaseMarkPairs_updated2)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(22040);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox[3];
    genArrangement(rng, N, sapBox);
    std::vector<uint32_t> newB = genBounds(rng, N);

    std::vector<uint32_t> updated;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 35) updated.push_back(b);
    const uint32_t numUpd = uint32_t(updated.size());

    const uint32_t SENT = 0xdeadbeefu;
    std::vector<uint32_t> proj[3] = { std::vector<uint32_t>(nbProj, SENT), std::vector<uint32_t>(nbProj, SENT),
                                      std::vector<uint32_t>(nbProj, SENT) };
    std::vector<uint32_t> rProj[3] = { proj[0], proj[1], proj[2] };
    for (uint32_t h : updated)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[ax][h * 2 + 0], mx = sapBox[ax][h * 2 + 1];
            rProj[ax][mn] = newB[h * 6 + ax];
            rProj[ax][mx] = newB[h * 6 + ax + 3];
        }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseMarkPairs.spv", "markUpdatedPairsLaunch2", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    BpDescRig rig(&shim);
    const PxgDevicePtr dUpd = rig.alloc(updated.data(), std::max<size_t>(1, updated.size()) * 4);
    const PxgDevicePtr dSap[3] = { rig.alloc(sapBox[0].data(), N * 2 * 4), rig.alloc(sapBox[1].data(), N * 2 * 4),
                                   rig.alloc(sapBox[2].data(), N * 2 * 4) };
    const PxgDevicePtr dPr[3] = { rig.alloc(proj[0].data(), nbProj * 4), rig.alloc(proj[1].data(), nbProj * 4),
                                  rig.alloc(proj[2].data(), nbProj * 4) };
    const PxgDevicePtr dNew = rig.alloc(newB.data(), N * 6 * 4);
    if (!dUpd) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.numUpdatedHandles = numUpd;
    rig.desc.updateData_updatedHandles = reinterpret_cast<PxU32*>(dUpd);
    rig.desc.newIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dNew);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        rig.desc.boxSapBox1D[ax] = reinterpret_cast<PxgSapBox1D*>(dSap[ax]);
        rig.desc.boxProjections[ax] = reinterpret_cast<PxU32*>(dPr[ax]);
    }
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch markUpdatedPairs2");
    shim.streamSynchronize();

    size_t pM = 0, mutated = 0;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> gp(nbProj);
        shim.memcpyDtoH(gp.data(), dPr[ax], nbProj * 4);
        for (uint32_t i = 0; i < nbProj; ++i) { if (gp[i] != rProj[ax][i]) ++pM; if (gp[i] != SENT) ++mutated; }
    }
    PXG_CHECK_MSG(pM == 0, "%zu updated2-projection mismatches (numUpd=%u)", pM, numUpd);
    PXG_CHECK_MSG(numUpd == 0 || mutated > 0, "kernel did not mutate any projection (unbound launch?)");

    for (PxgDevicePtr p : { dUpd, dSap[0], dSap[1], dSap[2], dPr[0], dPr[1], dPr[2], dNew, dDesc }) shim.memFree(p);
}

} // namespace
