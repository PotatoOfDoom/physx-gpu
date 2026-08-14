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

// P8.93: dmaBackArticulationDataLaunch.
//
// ===========================================================================
// THE ABI, GENERATED AND PINNED
// ===========================================================================
//
// Every offset the shader uses is asserted here against the engine's own
// headers.  That is the P8.49a discipline: a shader and a fixture sharing one
// wrong number agree perfectly, and only generation closes it.
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
// The kernel uses TWO index conventions — one THREAD per articulation for the
// sleep data, one WARP per articulation for the state buffer — and each resolves
// its own island-node / bodySim chain.  So:
//
//   * the island indices are a permutation and the remap ids are a DIFFERENT
//     permutation, so neither convention can be confused with the other or with
//     the identity;
//   * `nbArticulations` is 37 — more than one warp's worth, so the thread-indexed
//     half genuinely spans two warps and a port that assumes it fits in warp 0 is
//     visibly wrong;
//   * link and dof counts differ per articulation and are all BELOW the
//     descriptor's maxima, so the destination stride (maxima) and the copy length
//     (own counts) cannot be swapped without showing;
//   * every destination starts poisoned and is compared whole, so the bytes
//     between one articulation's copy length and the next one's stride must come
//     back poisoned;
//   * both accumulator pointers are exercised null and non-null, in two
//     dispatches.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "PxgBodySim.h"
#include "PxgSolverBody.h"
#include "PxNodeIndex.h"

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

using physx::PxgArticulation;
using physx::PxgArticulationCoreDesc;
using physx::PxgArticulationOutputDesc;
using physx::PxgArticulationLinkJointRootStateData;

// ---- the ABI, pinned ----
static_assert(sizeof(PxgArticulationOutputDesc) == 32, "PXG_ARTI_OUT_DESC_SIZE");
static_assert(offsetof(PxgArticulationOutputDesc, linkAndJointAndRootStateData) == 0,
              "PXG_ARTI_OUT_STATE_DATA");
static_assert(offsetof(PxgArticulationOutputDesc, sleepData) == 8, "PXG_ARTI_OUT_SLEEP_DATA");
static_assert(offsetof(PxgArticulationOutputDesc, errorAccumulator) == 16,
              "PXG_ARTI_OUT_ERROR_ACC");
static_assert(offsetof(PxgArticulationOutputDesc, contactResidualAccumulator) == 24,
              "PXG_ARTI_OUT_CONTACT_ACC");
static_assert(offsetof(PxgArticulationCoreDesc, mContactErrorAccumulator) == 344,
              "PXG_ARTI_CORE_CONTACT_ERROR_ACC");
static_assert(offsetof(PxgArticulation, linkJointRootStateDataBuffer) == 192,
              "PXG_ARTI_LINK_JOINT_ROOT_STATE");
static_assert(sizeof(physx::PxgSolverBodySleepData) == 8, "PXG_ARTI_SLEEP_DATA_SIZE");
// 28, not 32: a quat and a vec3 with no padding.
static_assert(sizeof(physx::PxTransform) == 28, "PXG_ARTI_STATE_TRANSFORM_SIZE");
static_assert(sizeof(physx::Cm::UnAlignedSpatialVector) == 24, "PXG_ARTI_STATE_SVEC_SIZE");

static std::vector<uint8_t> artiDmaBackLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t DB_NB_ARTI = 37;      // > 32, so the thread-indexed half spans two warps
const uint32_t DB_MAX_LINKS = 7;
const uint32_t DB_MAX_DOFS = 11;
const uint32_t DB_BODY_SIM_SLOTS = 192;
const uint32_t DB_ARTI_SLOTS = 64;   // > DB_NB_ARTI: the extra records stay poisoned
const uint32_t DB_ARTI_OFFSET = 5;

static uint32_t dbNumLinks(uint32_t k) { return 1u + (k % DB_MAX_LINKS); }
static uint32_t dbNumDofs(uint32_t k) { return (k * 3u) % (DB_MAX_DOFS + 1u); }

PXG_TEST(artiDmaBack)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // engine ABI { 8, 8 } -- the core descriptor and the output descriptor
        { "npArtiLinkAccel.spv", "dmaBackArticulationDataLaunch", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiDmaBackLoader);

    // Two DIFFERENT permutations, so the two index conventions cannot be confused.
    //
    // `REMAP` must be a permutation of [0, nbArticulations) and NOT merely of the
    // record slots: `articulationRemapId` is the engine's COMPACT articulation
    // index, and the error accumulator is addressed at `index` and at
    // `index + nbArticulations`.  Letting the remap ids run past
    // `nbArticulations` makes those two ranges overlap, and the fixture then
    // "fails" on a collision the engine cannot produce.  DB_NB_ARTI is prime, so
    // a stride-13 walk over it is a permutation.
    std::vector<uint32_t> NODE(DB_NB_ARTI), REMAP(DB_NB_ARTI);
    for (uint32_t k = 0; k < DB_NB_ARTI; ++k)
    {
        NODE[k] = (k * 7u + 11u) % DB_BODY_SIM_SLOTS;
        REMAP[k] = (k * 13u + 3u) % DB_NB_ARTI;
    }
    for (uint32_t k = 0; k < DB_NB_ARTI; ++k)
        for (uint32_t j = k + 1; j < DB_NB_ARTI; ++j)
        {
            PXG_CHECK_MSG(NODE[k] != NODE[j], "node index collision at %u/%u", k, j);
            PXG_CHECK_MSG(REMAP[k] != REMAP[j], "remap id collision at %u/%u", k, j);
        }

    std::vector<physx::PxNodeIndex> island(DB_ARTI_OFFSET + DB_NB_ARTI + 3);
    for (size_t i = 0; i < island.size(); ++i)
        island[i] = physx::PxNodeIndex(0x0BAD00u + PxU32(i));
    for (uint32_t k = 0; k < DB_NB_ARTI; ++k)
        island[DB_ARTI_OFFSET + k] = physx::PxNodeIndex(NODE[k]);

    std::vector<physx::PxgBodySim> bodySims(DB_BODY_SIM_SLOTS);
    std::memset(bodySims.data(), 0xEE, bodySims.size() * sizeof(physx::PxgBodySim));
    for (size_t i = 0; i < bodySims.size(); ++i)
        bodySims[i].articulationRemapId = 0xDEAD0000u + PxU32(i);
    for (uint32_t k = 0; k < DB_NB_ARTI; ++k)
        bodySims[NODE[k]].articulationRemapId = REMAP[k];

    // the per-articulation SOURCE buffers, one contiguous allocation sliced up
    const uint32_t srcStride =
        PxgArticulationLinkJointRootStateData::computeStateDataBufferByteSizeAligned16(
            DB_MAX_LINKS, DB_MAX_DOFS, 1);
    std::vector<uint32_t> srcImg(size_t(DB_ARTI_SLOTS) * srcStride / 4);
    for (size_t j = 0; j < srcImg.size(); ++j)
        srcImg[j] = 0x51000000u | uint32_t(j & 0xFFFFFFu);

    // the sleep-data input, one per articulation SLOT (indexed by the remap id)
    std::vector<uint32_t> sleepIn(size_t(DB_ARTI_SLOTS) * 2);
    for (size_t j = 0; j < sleepIn.size(); ++j)
        sleepIn[j] = 0x5E000000u | uint32_t(j & 0xFFFFFFu);

    const uint32_t outStride =
        PxgArticulationLinkJointRootStateData::computeStateDataBufferByteSizeAligned16(
            DB_MAX_LINKS, DB_MAX_DOFS, 1);
    std::vector<uint32_t> outPoison(size_t(DB_NB_ARTI + 3) * outStride / 4);
    for (size_t j = 0; j < outPoison.size(); ++j)
        outPoison[j] = 0xAB000000u | uint32_t(j & 0xFFFFFFu);
    std::vector<uint32_t> sleepPoison(size_t(DB_NB_ARTI + 5) * 2);
    for (size_t j = 0; j < sleepPoison.size(); ++j)
        sleepPoison[j] = 0xAC000000u | uint32_t(j & 0xFFFFFFu);
    std::vector<uint32_t> errPoison(size_t(DB_ARTI_SLOTS * 2 + 4) * 3);
    for (size_t j = 0; j < errPoison.size(); ++j)
        errPoison[j] = 0xAD000000u | uint32_t(j & 0xFFFFFFu);
    std::vector<uint32_t> conPoison(6);
    for (size_t j = 0; j < conPoison.size(); ++j)
        conPoison[j] = 0xAE000000u | uint32_t(j);

    PxgDevicePtr dIsland = 0, dBody = 0, dArtis = 0, dSrc = 0, dSleepIn = 0;
    PxgDevicePtr dOut = 0, dSleepOut = 0, dErr = 0, dCon = 0, dDesc = 0, dOutDesc = 0;
    shim.memAlloc(&dIsland, island.size() * sizeof(physx::PxNodeIndex));
    shim.memAlloc(&dBody, bodySims.size() * sizeof(physx::PxgBodySim));
    shim.memAlloc(&dArtis, size_t(DB_ARTI_SLOTS) * sizeof(PxgArticulation));
    shim.memAlloc(&dSrc, srcImg.size() * 4);
    shim.memAlloc(&dSleepIn, sleepIn.size() * 4);
    shim.memAlloc(&dOut, outPoison.size() * 4);
    shim.memAlloc(&dSleepOut, sleepPoison.size() * 4);
    shim.memAlloc(&dErr, errPoison.size() * 4);
    shim.memAlloc(&dCon, conPoison.size() * 4);
    shim.memAlloc(&dDesc, sizeof(PxgArticulationCoreDesc));
    shim.memAlloc(&dOutDesc, sizeof(PxgArticulationOutputDesc));
    if (!dIsland || !dBody || !dArtis || !dSrc || !dSleepIn || !dOut || !dSleepOut || !dErr
        || !dCon || !dDesc || !dOutDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<PxgArticulation> recs(DB_ARTI_SLOTS);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t k = 0; k < DB_NB_ARTI; ++k)
    {
        PxgArticulation& r = recs[REMAP[k]];
        // BELOW the descriptor's maxima, so stride and length cannot be swapped
        r.data.numLinks = physx::PxU16(dbNumLinks(k));
        r.data.numJointDofs = physx::PxU16(dbNumDofs(k));
        r.linkJointRootStateDataBuffer =
            reinterpret_cast<physx::PxU8*>(dSrc + PxgDevicePtr(size_t(REMAP[k]) * srcStride));
        r.internalResidualAccumulator.mErrorSumOfSquares = 1.0f + float(k);
        r.internalResidualAccumulator.mMaxError = 2.0f + float(k);
        r.internalResidualAccumulator.mCounter = int(100 + k);
        r.contactResidualAccumulator.mErrorSumOfSquares = 3.0f + float(k);
        r.contactResidualAccumulator.mMaxError = 4.0f + float(k);
        r.contactResidualAccumulator.mCounter = int(200 + k);
    }

    shim.memcpyHtoD(dIsland, island.data(), island.size() * sizeof(physx::PxNodeIndex));
    shim.memcpyHtoD(dBody, bodySims.data(), bodySims.size() * sizeof(physx::PxgBodySim));
    shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));
    shim.memcpyHtoD(dSrc, srcImg.data(), srcImg.size() * 4);
    shim.memcpyHtoD(dSleepIn, sleepIn.data(), sleepIn.size() * 4);

    size_t bad = 0;
    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool withAccumulators = (pass == 1);

        shim.memcpyHtoD(dOut, outPoison.data(), outPoison.size() * 4);
        shim.memcpyHtoD(dSleepOut, sleepPoison.data(), sleepPoison.size() * 4);
        shim.memcpyHtoD(dErr, errPoison.data(), errPoison.size() * 4);
        shim.memcpyHtoD(dCon, conPoison.data(), conPoison.size() * 4);

        PxgArticulationCoreDesc desc;
        std::memset(&desc, 0x5A, sizeof(desc));
        desc.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(dBody);
        desc.islandNodeIndices = reinterpret_cast<physx::PxNodeIndex*>(dIsland);
        desc.articulations = reinterpret_cast<PxgArticulation*>(dArtis);
        desc.articulationSleepData = reinterpret_cast<physx::PxgSolverBodySleepData*>(dSleepIn);
        desc.articulationOffset = DB_ARTI_OFFSET;
        desc.nbArticulations = DB_NB_ARTI;
        desc.mMaxLinksPerArticulation = DB_MAX_LINKS;
        desc.mMaxDofsPerArticulation = DB_MAX_DOFS;
        desc.mContactErrorAccumulator.mErrorSumOfSquares = 7.5f;
        desc.mContactErrorAccumulator.mMaxError = 8.25f;
        desc.mContactErrorAccumulator.mCounter = 4242;
        shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

        PxgArticulationOutputDesc outDesc;
        std::memset(&outDesc, 0, sizeof(outDesc));
        outDesc.linkAndJointAndRootStateData = reinterpret_cast<physx::PxU8*>(dOut);
        outDesc.sleepData = reinterpret_cast<physx::PxgSolverBodySleepData*>(dSleepOut);
        outDesc.errorAccumulator =
            withAccumulators ? reinterpret_cast<physx::Dy::ErrorAccumulator*>(dErr) : NULL;
        outDesc.contactResidualAccumulator =
            withAccumulators ? reinterpret_cast<physx::Dy::ErrorAccumulator*>(dCon) : NULL;
        shim.memcpyHtoD(dOutDesc, &outDesc, sizeof(outDesc));

        struct Params { uint64_t desc, outDesc; } prm = { uint64_t(dDesc), uint64_t(dOutDesc) };
        const uint32_t numBlocks = (DB_NB_ARTI + 1u) / 2u;
        PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &prm, sizeof(prm))
                          == PxgResult::Success,
                      "launch dmaBackArticulationDataLaunch");
        shim.streamSynchronize();

        // ---- the expectation ----
        std::vector<uint32_t> wantOut = outPoison, wantSleep = sleepPoison, wantErr = errPoison,
                              wantCon = conPoison;
        for (uint32_t k = 0; k < DB_NB_ARTI; ++k)
        {
            // ONE THREAD per articulation: the sleep data, indexed by the remap id
            wantSleep[size_t(k) * 2 + 0] = sleepIn[size_t(REMAP[k]) * 2 + 0];
            wantSleep[size_t(k) * 2 + 1] = sleepIn[size_t(REMAP[k]) * 2 + 1];

            // ONE WARP per articulation: the state buffer.  Destination stride from
            // the MAXIMA, copy length from this articulation's OWN counts.
            const uint32_t len =
                PxgArticulationLinkJointRootStateData::computeStateDataBufferByteSizeAligned16(
                    dbNumLinks(k), dbNumDofs(k), 1);
            for (uint32_t w = 0; w < len / 4; ++w)
                wantOut[size_t(k) * outStride / 4 + w] = srcImg[size_t(REMAP[k]) * srcStride / 4 + w];

            if (withAccumulators)
            {
                const float* ir = &recs[REMAP[k]].internalResidualAccumulator.mErrorSumOfSquares;
                const float* cr = &recs[REMAP[k]].contactResidualAccumulator.mErrorSumOfSquares;
                std::memcpy(&wantErr[size_t(REMAP[k]) * 3], ir, 12);
                std::memcpy(&wantErr[size_t(REMAP[k] + DB_NB_ARTI) * 3], cr, 12);
            }
        }
        if (withAccumulators)
            std::memcpy(wantCon.data(), &desc.mContactErrorAccumulator, 12);

        std::vector<uint32_t> gotOut(outPoison.size()), gotSleep(sleepPoison.size()),
            gotErr(errPoison.size()), gotCon(conPoison.size());
        shim.memcpyDtoH(gotOut.data(), dOut, gotOut.size() * 4);
        shim.memcpyDtoH(gotSleep.data(), dSleepOut, gotSleep.size() * 4);
        shim.memcpyDtoH(gotErr.data(), dErr, gotErr.size() * 4);
        shim.memcpyDtoH(gotCon.data(), dCon, gotCon.size() * 4);

        auto cmp = [&](const char* what, const std::vector<uint32_t>& w,
                       const std::vector<uint32_t>& g) {
            for (size_t k = 0; k < w.size(); ++k)
            {
                if (w[k] == g[k])
                    continue;
                if (bad < 12)
                    std::printf("MM db pass %u %s word %zu  ref %08x  gpu %08x\n", pass, what, k,
                                w[k], g[k]);
                ++bad;
            }
        };
        cmp("state", wantOut, gotOut);
        cmp("sleep", wantSleep, gotSleep);
        cmp("error acc", wantErr, gotErr);
        cmp("contact acc", wantCon, gotCon);
    }

    PXG_CHECK_MSG(bad == 0, "the DMA-back image matches the engine's own layout");

    shim.memFree(dIsland);
    shim.memFree(dBody);
    shim.memFree(dArtis);
    shim.memFree(dSrc);
    shim.memFree(dSleepIn);
    shim.memFree(dOut);
    shim.memFree(dSleepOut);
    shim.memFree(dErr);
    shim.memFree(dCon);
    shim.memFree(dDesc);
    shim.memFree(dOutDesc);
}

}  // namespace
