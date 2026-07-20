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

// P5 Step 4: clearNewFlagLaunch + bpSignalComplete on the ENGINE ABI — the
// kernels are driven through PxgCudaShim exactly as the engine launches them:
// clearNewFlag receives ONE device pointer to a REAL PxgBroadPhaseDesc whose
// nested pointers (createdHandles, per-axis boxSapBox1D / boxHandles[0]) are
// live device addresses (interior pointers included), walked via the proven
// BDA mirror; bpSignalComplete receives the raw signal pointer.  Bit-exact.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

const uint32_t GRID_X = 32; // engine: PxgBPKernelGridDim::BP_CLEAR_NEWFLAG blocks

static std::vector<uint8_t> cnfLoader(const char* file) { return loadShaderBlob(file); }

// createHandle(handle, isStart, isNew) — PxgBroadPhaseDesc.h
uint32_t refCreateHandle(uint32_t h, bool isStart, bool isNew)
{
    return (h << 3) | (isStart ? 1u : 0u) | (isNew ? 2u : 0u);
}

// per-axis SAP arrangement: sapBox[axis][b]={minPos,maxPos} partitioning [0,2N)
void genArrangement(std::mt19937& rng, uint32_t N, std::vector<uint32_t>& sapBox)
{
    const uint32_t nbProj = N * 2;
    sapBox.assign(3 * N * 2, 0);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> perm(nbProj);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t b = 0; b < N; ++b)
        {
            sapBox[(ax * N + b) * 2 + 0] = std::min(perm[2 * b], perm[2 * b + 1]);
            sapBox[(ax * N + b) * 2 + 1] = std::max(perm[2 * b], perm[2 * b + 1]);
        }
    }
}

PXG_TEST(broadphaseClearNewFlag_clear)
{
    using namespace physx;

    std::mt19937 rng(23010);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox;
    genArrangement(rng, N, sapBox);

    // current-frame handles pre-filled with the "new" flag SET at every box's
    // start/end positions, so the kernel must clear it for created handles.
    std::vector<uint32_t> handles0(3 * nbProj, 0);
    for (uint32_t ax = 0; ax < 3; ++ax)
        for (uint32_t b = 0; b < N; ++b)
        {
            uint32_t mn = sapBox[(ax * N + b) * 2 + 0], mx = sapBox[(ax * N + b) * 2 + 1];
            handles0[ax * nbProj + mn] = refCreateHandle(b, true, true);
            handles0[ax * nbProj + mx] = refCreateHandle(b, false, true);
        }

    // created handles: ~40% distinct
    std::vector<uint32_t> created;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 40) created.push_back(b);
    const uint32_t numCreated = uint32_t(created.size());

    // reference: clear the "new" flag (isNew=false) for created handles
    std::vector<uint32_t> rHandles = handles0;
    for (uint32_t h : created)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[(ax * N + h) * 2 + 0], mx = sapBox[(ax * N + h) * 2 + 1];
            rHandles[ax * nbProj + mn] = refCreateHandle(h, true, false);
            rHandles[ax * nbProj + mx] = refCreateHandle(h, false, false);
        }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseClearNewFlag.spv", "clearNewFlagLaunch", 8u, 0 }, // { u64 bpDesc }
    };
    PxgCudaShim shim(dev, manifest, 1, &cnfLoader);

    PxgDevicePtr dCreated = 0, dSap = 0, dH = 0, dDesc = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dCreated, created.size() * 4) == PxgResult::Success, "memAlloc created");
    PXG_CHECK_MSG(shim.memAlloc(&dSap, sapBox.size() * 4) == PxgResult::Success, "memAlloc sap");
    PXG_CHECK_MSG(shim.memAlloc(&dH, handles0.size() * 4) == PxgResult::Success, "memAlloc handles");
    PXG_CHECK_MSG(shim.memAlloc(&dDesc, sizeof(PxgBroadPhaseDesc)) == PxgResult::Success, "memAlloc desc");
    if (!dCreated || !dSap || !dH || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dCreated, created.data(), created.size() * 4);
    shim.memcpyHtoD(dSap, sapBox.data(), sapBox.size() * 4);
    shim.memcpyHtoD(dH, handles0.data(), handles0.size() * 4);

    // The REAL engine aggregate, with per-axis interior device pointers —
    // exactly what the engine DMAs up before launching.
    PxgBroadPhaseDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.numCreatedHandles = numCreated;
    desc.updateData_createdHandles = reinterpret_cast<PxU32*>(dCreated);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        desc.boxSapBox1D[ax] = reinterpret_cast<PxgSapBox1D*>(dSap + uint64_t(ax) * N * 2 * 4);
        desc.boxHandles[0][ax] = reinterpret_cast<PxU32*>(dH + uint64_t(ax) * nbProj * 4);
    }
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    struct { uint64_t bpDesc; } prm = { dDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, GRID_X, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch clearNewFlag with the engine bpDesc aggregate");
    shim.streamSynchronize();

    std::vector<uint32_t> gH(3 * nbProj, 0);
    shim.memcpyDtoH(gH.data(), dH, gH.size() * 4);
    size_t hM = 0;
    for (uint32_t i = 0; i < 3 * nbProj; ++i) if (gH[i] != rHandles[i]) ++hM;
    PXG_CHECK_MSG(hM == 0, "%zu clear-new-flag mismatches (numCreated=%u)", hM, numCreated);

    shim.memFree(dCreated);
    shim.memFree(dSap);
    shim.memFree(dH);
    shim.memFree(dDesc);
}

PXG_TEST(broadphaseClearNewFlag_signal)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseClearNewFlag.spv", "bpSignalComplete", 8u, 0 }, // { u64 signal }
    };
    PxgCudaShim shim(dev, manifest, 1, &cnfLoader);

    // signal[0] must become 1; the neighbouring words must stay untouched.
    const uint32_t init[4] = { 0u, 0xdeadbeefu, 0xcafef00du, 0x12345678u };
    PxgDevicePtr dSig = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dSig, sizeof(init)) == PxgResult::Success, "memAlloc signal");
    if (!dSig) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memcpyHtoD(dSig, init, sizeof(init));

    struct { uint64_t signal; } prm = { dSig };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch bpSignalComplete");
    shim.streamSynchronize();

    uint32_t gS[4] = {};
    shim.memcpyDtoH(gS, dSig, sizeof(gS));
    PXG_CHECK_MSG(gS[0] == 1u, "signal[0]=%u expected 1", gS[0]);
    PXG_CHECK_MSG(gS[1] == 0xdeadbeefu && gS[2] == 0xcafef00du && gS[3] == 0x12345678u,
                  "signal spilled past word 0: %u %u %u", gS[1], gS[2], gS[3]);
    shim.memFree(dSig);
}

} // namespace
