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


// P8.96a: npArtiSumBatches.slang -- artiSumInternalContactAndJointBatches1Launch.
//
// Pure integer reductions, so the oracle is trivially bit-exact: the mirror
// replicates the CUDA's block/iteration/warp-chunk structure as host loops and
// the comparison is == on every word of all eight 32-entry output arrays --
// including the blocks BEYOND the data, which the kernel still writes (zeros),
// and which the fixture poisons so a kernel that skips them fails.
//
// The count seeding makes the four source arrays and the two aggregations
// separable: each array gets its own value pattern (so a crossed source is
// visible), the per-warp-chunk maxima differ from the chunk sums (so a
// header/unique swap is visible), and nbArticulations = 700 puts real data in
// exactly TWO of the 32 blocks with a partial warp chunk at the tail --
// blocks 2..31 must write zeros over their poison.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxgArticulationCoreDesc.h"
#include "PxgSolverCoreDesc.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxU32;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> asbLoader(const char* file) { return loadShaderBlob(file); }

// TWO sizes, and the second is what makes numIter exceed one.  With any
// nbArticulations <= 16384 (32 blocks x 512), numIter == 1 and the mutants
// "the block slice strides one block-span" (numIter*b*512 -> b*512) and "the
// iteration count floors" are IDENTITIES -- the first sweep proved both by
// surviving.  17000 articulations force numIter = 2, where each block owns TWO
// consecutive 512-spans and both mutants change the answer.
const uint32_t ASB_SIZES[2] = { 700, 17000 };
const uint32_t ASB_BLOCKS = 32;    // the fixed grid (PxgArticulationCore.cpp:305)
const uint32_t ASB_BLOCK_DIM = 512;
const uint32_t ASB_WARPS = 16;

// four DISTINCT count patterns, one per source array
static uint32_t asbSC(uint32_t i) { return (i * 7u + 3u) % 11u; }
static uint32_t asbSJ(uint32_t i) { return (i * 5u + 1u) % 7u; }
static uint32_t asbFC(uint32_t i) { return (i * 13u + 5u) % 17u; }
static uint32_t asbFJ(uint32_t i) { return (i * 3u + 2u) % 5u; }

static size_t asbRun(PxgCudaShim& shim, uint32_t ASB_ARTS)
{
    std::vector<uint32_t> sc(ASB_ARTS), sj(ASB_ARTS), fc(ASB_ARTS), fj(ASB_ARTS);
    for (uint32_t i = 0; i < ASB_ARTS; ++i)
    {
        sc[i] = asbSC(i);
        sj[i] = asbSJ(i);
        fc[i] = asbFC(i);
        fj[i] = asbFJ(i);
    }

    // the mirror: the CUDA's structure as host loops
    const uint32_t totalBlockRequired = (ASB_ARTS + ASB_BLOCK_DIM - 1) / ASB_BLOCK_DIM;
    const uint32_t numIter = (totalBlockRequired + 31) / 32;
    std::vector<uint32_t> want[8];
    for (auto& w : want) w.assign(ASB_BLOCKS, 0u);
    for (uint32_t b = 0; b < ASB_BLOCKS; ++b)
    {
        uint32_t sums[4] = { 0, 0, 0, 0 }, hdrs[4] = { 0, 0, 0, 0 };
        for (uint32_t i = 0; i < numIter; ++i)
            for (uint32_t w = 0; w < ASB_WARPS; ++w)
            {
                uint32_t wmax[4] = { 0, 0, 0, 0 };
                for (uint32_t l = 0; l < 32; ++l)
                {
                    const uint32_t idx = i * ASB_BLOCK_DIM + (w * 32 + l)
                                       + numIter * b * ASB_BLOCK_DIM;
                    const uint32_t v[4] = { idx < ASB_ARTS ? sc[idx] : 0u,
                                            idx < ASB_ARTS ? sj[idx] : 0u,
                                            idx < ASB_ARTS ? fc[idx] : 0u,
                                            idx < ASB_ARTS ? fj[idx] : 0u };
                    for (uint32_t k = 0; k < 4; ++k)
                    {
                        sums[k] += v[k];
                        wmax[k] = v[k] > wmax[k] ? v[k] : wmax[k];
                    }
                }
                for (uint32_t k = 0; k < 4; ++k)
                    hdrs[k] += wmax[k];
            }
        want[0][b] = sums[0]; want[1][b] = sums[1];
        want[2][b] = hdrs[0]; want[3][b] = hdrs[1];
        want[4][b] = sums[2]; want[5][b] = sums[3];
        want[6][b] = hdrs[2]; want[7][b] = hdrs[3];
    }

    // device buffers
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        return dp;
    };
    PxgDevicePtr dSc = up(sc.data(), sc.size() * 4), dSj = up(sj.data(), sj.size() * 4);
    PxgDevicePtr dFc = up(fc.data(), fc.size() * 4), dFj = up(fj.data(), fj.size() * 4);
    std::vector<uint32_t> poison(ASB_BLOCKS, 0xABABABABu);
    PxgDevicePtr dOut[8];
    for (auto& dp : dOut) dp = up(poison.data(), poison.size() * 4);
    PxgDevicePtr dDesc = 0, dPre = 0;
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    shim.memAlloc(&dPre, sizeof(physx::PxgPrePrepDesc));
    if (!dSc || !dSj || !dFc || !dFj || !dDesc || !dPre)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.mTempContactUniqueIndicesBlock = reinterpret_cast<PxU32*>(dOut[0]);
    desc.mTempConstraintUniqueIndicesBlock = reinterpret_cast<PxU32*>(dOut[1]);
    desc.mTempContactHeaderBlock = reinterpret_cast<PxU32*>(dOut[2]);
    desc.mTempConstraintHeaderBlock = reinterpret_cast<PxU32*>(dOut[3]);
    desc.mTempSelfContactUniqueIndicesBlock = reinterpret_cast<PxU32*>(dOut[4]);
    desc.mTempSelfConstraintUniqueIndicesBlock = reinterpret_cast<PxU32*>(dOut[5]);
    desc.mTempSelfContactHeaderBlock = reinterpret_cast<PxU32*>(dOut[6]);
    desc.mTempSelfConstraintHeaderBlock = reinterpret_cast<PxU32*>(dOut[7]);
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    physx::PxgPrePrepDesc pre;
    std::memset(&pre, 0xAB, sizeof(pre));
    pre.mArtiStaticContactCounts = reinterpret_cast<PxU32*>(dSc);
    pre.mArtiStaticConstraintCounts = reinterpret_cast<PxU32*>(dSj);
    pre.mArtiSelfContactCounts = reinterpret_cast<PxU32*>(dFc);
    pre.mArtiSelfConstraintCounts = reinterpret_cast<PxU32*>(dFj);
    shim.memcpyHtoD(dPre, &pre, sizeof(pre));

    struct Params { uint64_t scDesc, prePrep; uint32_t nb, pad; } prm = { uint64_t(dDesc),
                                                                          uint64_t(dPre),
                                                                          ASB_ARTS, 0 };
    PXG_CHECK_MSG(shim.launchKernel(0, ASB_BLOCKS, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiSumInternalContactAndJointBatches1Launch");
    shim.streamSynchronize();

    static const char* names[8] = { "staticContactUnique", "staticJointUnique",
                                    "staticContactHeader", "staticJointHeader",
                                    "selfContactUnique", "selfJointUnique",
                                    "selfContactHeader", "selfJointHeader" };
    size_t bad = 0;
    std::vector<uint32_t> got(ASB_BLOCKS);
    for (uint32_t k = 0; k < 8; ++k)
    {
        shim.memcpyDtoH(got.data(), dOut[k], got.size() * 4);
        for (uint32_t b = 0; b < ASB_BLOCKS; ++b)
            if (got[b] != want[k][b])
            {
                if (bad < 10)
                    std::printf("MM asb %s block %u  ref %u  gpu %u\n", names[k], b, want[k][b],
                                got[b]);
                ++bad;
            }
    }
    for (auto dp : dOut) shim.memFree(dp);
    shim.memFree(dSc); shim.memFree(dSj); shim.memFree(dFc); shim.memFree(dFj);
    shim.memFree(dDesc); shim.memFree(dPre);
    return bad;
}

PXG_TEST(artiSumBatches1)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSumBatches.spv", "artiSumInternalContactAndJointBatches1Launch", 24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &asbLoader);

    size_t bad = 0;
    for (uint32_t nb : ASB_SIZES)
    {
        const uint32_t iters = (((nb + 511) / 512) + 31) / 32;
        std::printf("[ info ] asb: %u articulations -> numIter %u\n", nb, iters);
        bad += asbRun(shim, nb);
    }
    PXG_CHECK_MSG(bad == 0, "the block reductions do not match the engine");
}

}  // namespace
