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

// Verification of broadphaseEndPtHistogram.slang on the ENGINE ABI (P5 Step 4):
// a REAL PxgBroadPhaseDesc is filled with live device addresses, uploaded, and
// both kernels are launched through PxgCudaShim exactly as the engine does —
// (bpDesc*, bool isIncremental) packed by the bridge's natural-alignment rule
// into a 16-byte blob, grid (32, 3, 1) with blockIdx.y = axis.  Checked against
// a C++ transcription of the CUDA passes: per-projection start/end/end2
// classification, the device-wide (two-pass) exclusive prefix producing
// start/end/totalEnd point histograms and their per-block totals, and the
// compacting scatter of non-deleted handles into the ordered start/end point
// handle arrays.  Covers incremental (with and without removals — the removal
// path exercises the .cu's NULL total-end second pass) and full-rebuild
// (non-incremental) modes across the 32-block boundary.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t NUM_BLOCKS = 32;
const uint32_t BLOCK = 256;
const uint32_t SENT = 0xdeadbeefu;

bool isStart(uint32_t sh) { return (sh & 1u) != 0u; }
bool isDeleted(uint32_t sh) { return (sh & (1u << 2)) != 0u; }
uint32_t getHandle(uint32_t sh) { return sh >> 3; }

static std::vector<uint8_t> bpShimLoader(const char* file) { return loadShaderBlob(file); }

// Mirrors the bridge's marshalling: place each argument at the natural alignment
// inferred from its size, then round the blob to the largest alignment seen
// (a compiled struct's trailing padding).
struct EngineBlob
{
    std::vector<uint8_t> bytes;
    size_t maxAlign = 1;

    static size_t alignFor(size_t size)
    {
        if (size >= 16) return 16;
        if (size >= 8)  return 8;
        if (size >= 4)  return 4;
        if (size >= 2)  return 2;
        return 1;
    }
    template <typename T>
    void add(const T& v)
    {
        const size_t a = alignFor(sizeof(T));
        if (a > maxAlign) maxAlign = a;
        bytes.resize((bytes.size() + (a - 1)) & ~(a - 1));
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + (maxAlign - 1)) & ~(maxAlign - 1)); }
};

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

// Runs one dispatch configuration end-to-end on the engine ABI and asserts
// every produced array against a CPU reference.  Failures are recorded via
// PXG_CHECK_MSG.
void runEndPtCase(const char* tag, uint32_t seed, uint32_t numPrev, uint32_t numCreated,
                  uint32_t numRemoved, uint32_t isIncremental)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(seed);

    const uint32_t nH = isIncremental ? numPrev : (numPrev + numCreated - numRemoved);
    const uint32_t P = nH * 2;
    const bool doFrame1 = isIncremental && (numRemoved != 0);
    const uint32_t numFrames = doFrame1 ? 2u : 1u;

    const uint32_t nbBlocksRequired = (P + BLOCK - 1) / BLOCK;
    const uint32_t numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    // ---- inputs: boxHandles[2][3][P] flat (fill both frames/all axes) ----
    // Real broad-phase projections come in exact start/end pairs (one of each per
    // box), so every (frame,axis) has exactly nH start and nH end projections;
    // the ordered-handle scatter stride depends on that invariant. Build nH
    // pairs per (frame,axis) and shuffle them into a plausible sorted order,
    // with an independent "new" bit (~must not gate the scatter) and ~10% deleted.
    std::vector<uint32_t> boxHandles(2 * 3 * P);
    for (uint32_t f = 0; f < 2; ++f)
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            std::vector<uint32_t> proj;
            proj.reserve(P);
            for (uint32_t k = 0; k < nH; ++k)
            {
                const uint32_t h = rng() % 100000u;
                const uint32_t nS = rng() & 1u, nE = rng() & 1u; // "new" bit
                const uint32_t dS = (rng() % 100 < 10) ? 1u : 0u;
                const uint32_t dE = (rng() % 100 < 10) ? 1u : 0u;
                proj.push_back((h << 3) | 1u | (nS << 1) | (dS << 2)); // start projection (bit0 = 1)
                proj.push_back((h << 3) | 0u | (nE << 1) | (dE << 2)); // end projection   (bit0 = 0)
            }
            std::shuffle(proj.begin(), proj.end(), rng);
            for (uint32_t i = 0; i < P; ++i)
                boxHandles[(f * 3 + axis) * P + i] = proj[i];
        }

    // ---- shim + engine desc with live device addresses ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseEndPtHistogram.spv", "computeEndPtsHistogram", 16u, 0 },
        { "broadphaseEndPtHistogram.spv", "outputEndPtsHistogram", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &bpShimLoader);

    BpDescRig rig(&shim);

    const std::vector<uint32_t> sentP(P, SENT);
    const std::vector<uint32_t> sentH(nH, SENT);
    const std::vector<uint32_t> sentB(NUM_BLOCKS, SENT);

    PxgDevicePtr dBox[2][3], dEnd[2][3], dStart[2][3], dBlockEnd[2][3], dBlockStart[2][3];
    PxgDevicePtr dEndHand[2][3], dStartHand[2][3];
    PxgDevicePtr dTotal[3], dBlockTotal[3];
    for (uint32_t f = 0; f < 2; ++f)
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            dBox[f][axis] = rig.alloc(&boxHandles[(f * 3 + axis) * P], P * 4);
            dEnd[f][axis] = rig.alloc(sentP.data(), P * 4);
            dStart[f][axis] = rig.alloc(sentP.data(), P * 4);
            dBlockEnd[f][axis] = rig.alloc(sentB.data(), NUM_BLOCKS * 4);
            dBlockStart[f][axis] = rig.alloc(sentB.data(), NUM_BLOCKS * 4);
            dEndHand[f][axis] = rig.alloc(sentH.data(), nH * 4);
            dStartHand[f][axis] = rig.alloc(sentH.data(), nH * 4);
        }
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        dTotal[axis] = rig.alloc(sentP.data(), P * 4);
        dBlockTotal[axis] = rig.alloc(sentB.data(), NUM_BLOCKS * 4);
    }
    if (!dBox[0][0])
    {
        std::printf("  NOTE: shim has no device-address allocations (no BDA) -> skip\n");
        return;
    }

    rig.desc.numPreviousHandles = numPrev;
    rig.desc.numCreatedHandles = numCreated;
    rig.desc.numRemovedHandles = numRemoved;
    for (uint32_t f = 0; f < 2; ++f)
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            rig.desc.boxHandles[f][axis] = reinterpret_cast<PxU32*>(dBox[f][axis]);
            rig.desc.endPtHistogram[f][axis] = reinterpret_cast<PxU32*>(dEnd[f][axis]);
            rig.desc.startPtHistogram[f][axis] = reinterpret_cast<PxU32*>(dStart[f][axis]);
            rig.desc.blockEndPtHistogram[f][axis] = reinterpret_cast<PxU32*>(dBlockEnd[f][axis]);
            rig.desc.blockStartPtHistogram[f][axis] = reinterpret_cast<PxU32*>(dBlockStart[f][axis]);
            rig.desc.endPointHandles[f][axis] = reinterpret_cast<PxU32*>(dEndHand[f][axis]);
            rig.desc.startPointHandles[f][axis] = reinterpret_cast<PxU32*>(dStartHand[f][axis]);
        }
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        rig.desc.totalEndPtHistogram[axis] = reinterpret_cast<PxU32*>(dTotal[axis]);
        rig.desc.blockTotalEndPtHistogram[axis] = reinterpret_cast<PxU32*>(dBlockTotal[axis]);
    }
    const PxgDevicePtr dDesc = rig.upload();

    // Engine ABI blob: (bpDesc*, bool) — argSizes {8,1}; the bool lands at
    // byte 8 and the blob rounds to 16 (bytes 9..15 zero-filled).
    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.add<uint8_t>(uint8_t(isIncremental ? 1u : 0u));
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 16, "[%s] endpt blob is 16 bytes (got %zu)", tag, blob.bytes.size());

    // Engine grid: (BP_COMPUTE/OUTPUT_ENDPT_HISTOGRAM = 32, 3, 1), 256 threads;
    // blockIdx.y selects the axis exactly like the CUDA launch.
    PXG_CHECK_MSG(shim.launchKernel(0, NUM_BLOCKS, 3, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "[%s] launch computeEndPtsHistogram", tag);
    shim.streamSynchronize();
    PXG_CHECK_MSG(shim.launchKernel(1, NUM_BLOCKS, 3, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "[%s] launch outputEndPtsHistogram", tag);
    shim.streamSynchronize();

    // ---- download into flat host mirrors (same indexing as the reference) ----
    std::vector<uint32_t> gEnd(2 * 3 * P, SENT), gStart(2 * 3 * P, SENT), gTotal(3 * P, SENT);
    std::vector<uint32_t> gBlockEnd(2 * 3 * NUM_BLOCKS, SENT), gBlockStart(2 * 3 * NUM_BLOCKS, SENT),
        gBlockTotal(3 * NUM_BLOCKS, SENT);
    std::vector<uint32_t> gEndHand(2 * 3 * nH, SENT), gStartHand(2 * 3 * nH, SENT);
    for (uint32_t f = 0; f < 2; ++f)
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const uint32_t fa = f * 3 + axis;
            shim.memcpyDtoH(&gEnd[fa * P], dEnd[f][axis], P * 4);
            shim.memcpyDtoH(&gStart[fa * P], dStart[f][axis], P * 4);
            shim.memcpyDtoH(&gBlockEnd[fa * NUM_BLOCKS], dBlockEnd[f][axis], NUM_BLOCKS * 4);
            shim.memcpyDtoH(&gBlockStart[fa * NUM_BLOCKS], dBlockStart[f][axis], NUM_BLOCKS * 4);
            shim.memcpyDtoH(&gEndHand[fa * nH], dEndHand[f][axis], nH * 4);
            shim.memcpyDtoH(&gStartHand[fa * nH], dStartHand[f][axis], nH * 4);
        }
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        shim.memcpyDtoH(&gTotal[axis * P], dTotal[axis], P * 4);
        shim.memcpyDtoH(&gBlockTotal[axis * NUM_BLOCKS], dBlockTotal[axis], NUM_BLOCKS * 4);
    }

    // ---- reference + assertions, per processed frame/axis ----
    size_t histMism = 0, blockMism = 0, handMism = 0;
    uint32_t firstBadHist = 0xffffffffu, badGpu = 0, badRef = 0;

    for (uint32_t frame = 0; frame < numFrames; ++frame)
    {
        const bool writeTotal = (frame == 0);
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const uint32_t faBase = (frame * 3 + axis) * P;
            const uint32_t taBase = axis * P;
            const uint32_t blkBase = (frame * 3 + axis) * NUM_BLOCKS;
            const uint32_t tblkBase = axis * NUM_BLOCKS;
            const uint32_t hBase = (frame * 3 + axis) * nH;

            std::vector<uint32_t> refEnd(P), refStart(P), refTotal(P);
            std::vector<uint32_t> refEndHand, refStartHand;
            uint32_t accE = 0, accS = 0, accT = 0;
            for (uint32_t i = 0; i < P; ++i)
            {
                const uint32_t sortedHandle = boxHandles[faBase + i];
                const uint32_t end2 = 1u - (isStart(sortedHandle) ? 1u : 0u);
                const uint32_t st = isDeleted(sortedHandle) ? 0u : (1u - end2);
                const uint32_t en = isDeleted(sortedHandle) ? 0u : end2;
                refEnd[i] = accE;   // exclusive prefix (global — matches the two-pass result)
                refStart[i] = accS;
                refTotal[i] = accT;
                if (!isDeleted(sortedHandle))
                {
                    if (isStart(sortedHandle))
                        refStartHand.push_back(getHandle(sortedHandle));
                    else
                        refEndHand.push_back(getHandle(sortedHandle));
                }
                accE += en;
                accS += st;
                accT += end2;
            }

            for (uint32_t i = 0; i < P; ++i)
            {
                if (gEnd[faBase + i] != refEnd[i])
                {
                    if (firstBadHist == 0xffffffffu) { firstBadHist = faBase + i; badGpu = gEnd[faBase + i]; badRef = refEnd[i]; }
                    ++histMism;
                }
                if (gStart[faBase + i] != refStart[i]) ++histMism;
                if (writeTotal && gTotal[taBase + i] != refTotal[i]) ++histMism;
            }

            // block totals over each block's contiguous range
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
            {
                const uint32_t s0 = b * numIterationPerBlock * BLOCK;
                const uint32_t e0 = std::min((b + 1) * numIterationPerBlock * BLOCK, P);
                uint32_t bE = 0, bS = 0, bT = 0;
                for (uint32_t i = s0; i < e0; ++i)
                {
                    const uint32_t sortedHandle = boxHandles[faBase + i];
                    const uint32_t end2 = 1u - (isStart(sortedHandle) ? 1u : 0u);
                    bT += end2;
                    if (!isDeleted(sortedHandle))
                    {
                        if (end2) ++bE; else ++bS;
                    }
                }
                if (gBlockEnd[blkBase + b] != bE) ++blockMism;
                if (gBlockStart[blkBase + b] != bS) ++blockMism;
                if (writeTotal && gBlockTotal[tblkBase + b] != bT) ++blockMism;
            }

            // ordered (compacted) handles
            for (size_t j = 0; j < refStartHand.size(); ++j)
                if (gStartHand[hBase + j] != refStartHand[j]) ++handMism;
            for (size_t j = 0; j < refEndHand.size(); ++j)
                if (gEndHand[hBase + j] != refEndHand[j]) ++handMism;
        }
    }

    if (histMism)
        PXG_CHECK_MSG(histMism == 0, "[%s] %zu histogram mismatches (first at %u: gpu=%u ref=%u)", tag, histMism,
                      firstBadHist, badGpu, badRef);
    else
        PXG_CHECK_MSG(histMism == 0, "[%s] %zu histogram mismatches", tag, histMism);
    PXG_CHECK_MSG(blockMism == 0, "[%s] %zu block-total mismatches", tag, blockMism);
    PXG_CHECK_MSG(handMism == 0, "[%s] %zu ordered-handle mismatches", tag, handMism);

    for (uint32_t f = 0; f < 2; ++f)
        for (uint32_t axis = 0; axis < 3; ++axis)
            for (PxgDevicePtr pfree : { dBox[f][axis], dEnd[f][axis], dStart[f][axis], dBlockEnd[f][axis],
                                        dBlockStart[f][axis], dEndHand[f][axis], dStartHand[f][axis] })
                shim.memFree(pfree);
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        shim.memFree(dTotal[axis]);
        shim.memFree(dBlockTotal[axis]);
    }
    shim.memFree(dDesc);
}

PXG_TEST(broadphaseEndPtHistogram_incrementalNoRemoval)
{
    runEndPtCase("incremental-no-removal", 15100, 6000, 0, 0, 1);
}

PXG_TEST(broadphaseEndPtHistogram_incrementalWithRemoval)
{
    runEndPtCase("incremental-with-removal", 15200, 6000, 0, 500, 1);
}

PXG_TEST(broadphaseEndPtHistogram_nonIncremental)
{
    runEndPtCase("non-incremental", 15300, 4000, 2500, 500, 0);
}

} // namespace
