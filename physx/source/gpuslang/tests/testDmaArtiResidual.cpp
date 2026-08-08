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

// P8.66: engine-ABI fixture for dmaArticulationResidual (dmaArtiResidualBda.slang).
//
// The kernel walks a three-hop index chain and the fixture makes every hop
// distinguishable, because a chain where the hops agree cannot detect a shader
// that skips one:
//
//   islandNodeIndices[tid + articulationOffset].index()  -> nodeIndex
//   bodySimPool[nodeIndex].articulationRemapId           -> poolIndex
//   articulations[poolIndex].{internal,contact}Residual  -> the payload
//
// articulationOffset is 2 (so ignoring it reads the wrong island slots), the
// node indices are 7, 3 and 11 (so using the dispatch index as a node index is
// wrong), and the remap sends them to pool slots 1, 2 and 0 — neither the
// identity nor the node index.  The output is indexed by the POOL index, not
// the dispatch index, exactly as the CUDA does, so a remap bug lands the value
// in the wrong host slot instead of merely reading the wrong source.
//
// The destination is a REAL pinned-host allocation (memHostAlloc), so the
// launch-ABI row's hostPtrMask bit is exercised end to end: the fixture passes
// the CPU address, the bridge substitutes the device address, and the fixture
// reads the result back through the CPU mapping.  Without the mask bit the
// shader writes to an unrelated address and every slot stays poisoned.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "PxgBodySim.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> dmaArtiResidualLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// Compile-time pins for the constants this kernel addresses with.
// ---------------------------------------------------------------------------
namespace pin
{
using namespace physx;
static_assert(sizeof(PxgArticulationCoreDesc) == 360, "PxgArticulationCoreDesc size");
static_assert(offsetof(PxgArticulationCoreDesc, mBodySimBufferDeviceData) == 0, "..BODY_SIM_POOL");
static_assert(offsetof(PxgArticulationCoreDesc, articulationSleepData) == 8, "..SLEEP_DATA");
static_assert(offsetof(PxgArticulationCoreDesc, islandNodeIndices) == 16, "..ISLAND_NODE_INDICES");
static_assert(offsetof(PxgArticulationCoreDesc, solverBodyIndices) == 24, "..SOLVER_BODY_INDICES");
static_assert(offsetof(PxgArticulationCoreDesc, articulations) == 32, "..ARTICULATIONS");
static_assert(offsetof(PxgArticulationCoreDesc, articulationOffset) == 40, "..ARTICULATION_OFFSET");
static_assert(offsetof(PxgArticulationCoreDesc, nbArticulations) == 44, "..NB_ARTICULATIONS");
static_assert(offsetof(PxgArticulation, internalResidualAccumulator) == 512, "..INTERNAL_RESIDUAL");
static_assert(offsetof(PxgArticulation, contactResidualAccumulator) == 524, "..CONTACT_RESIDUAL");
static_assert(sizeof(Dy::ErrorAccumulator) == 12, "..ERROR_ACCUMULATOR_SIZE");
static_assert(sizeof(PxNodeIndex) == 8, "PXG_NODE_INDEX_SIZE");
static_assert(sizeof(PxgBodySim) == 224, "PXG_BODY_SIM_SIZE");
static_assert(offsetof(PxgBodySim, articulationRemapId) == 176, "..ARTICULATION_REMAP_ID");
} // namespace pin

PXG_TEST(dmaArtiResidual)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "dmaArtiResidualBda.spv", "dmaArticulationResidual", 16u, 0 }, // { u64 desc, u64 out }
    };
    PxgCudaShim shim(dev, manifest, 1, &dmaArtiResidualLoader);

    const uint32_t NB_ARTI = 3;
    const uint32_t ARTI_OFFSET = 2;          // the articulations start at island slot 2
    const uint32_t NODE[NB_ARTI] = { 7, 3, 11 };
    // The remap must be a PERMUTATION of [0, nbArticulations) -- the CUDA asserts
    // `index < nbArticulations` (forwardDynamic2.cu:3786), and that invariant is
    // exactly what keeps the two output ranges disjoint: internal residuals land
    // in [0, nbArti) and contact residuals in [nbArti, 2*nbArti).  A first draft
    // of this fixture used pool index 4 with nbArticulations = 3, which made one
    // articulation's contact slot collide with another's internal slot -- and the
    // fixture, not the shader, was wrong.  Still not the identity and still not
    // the node index, so all three hops stay distinguishable.
    const uint32_t POOL[NB_ARTI] = { 1, 2, 0 };
    const uint32_t ISLAND_SLOTS = ARTI_OFFSET + NB_ARTI + 2;  // trailing slots stay decoys
    const uint32_t BODY_SIM_SLOTS = 16;
    const uint32_t POOL_SLOTS = 6;   // 3 live + 3 decoy records the kernel must not read
    // 2*nbArti live slots plus three the kernel must leave alone, so an
    // over-write past the contact range is caught rather than assumed absent
    const uint32_t OUT_SLOTS = 2 * NB_ARTI + 3;

    // ---- island node indices: real PxNodeIndex, decoys outside the range ----
    std::vector<PxNodeIndex> island(ISLAND_SLOTS);
    for (uint32_t i = 0; i < ISLAND_SLOTS; ++i)
        island[i] = PxNodeIndex(0x0BADu + i);           // decoy
    for (uint32_t a = 0; a < NB_ARTI; ++a)
        island[ARTI_OFFSET + a] = PxNodeIndex(NODE[a]);

    // ---- body sims: only the three named slots carry a real remap ----
    std::vector<PxgBodySim> bodySims(BODY_SIM_SLOTS);
    std::memset(bodySims.data(), 0xEE, bodySims.size() * sizeof(PxgBodySim));
    for (uint32_t i = 0; i < BODY_SIM_SLOTS; ++i)
        bodySims[i].articulationRemapId = 0xDEAD0000u + i;   // decoy remap
    for (uint32_t a = 0; a < NB_ARTI; ++a)
        bodySims[NODE[a]].articulationRemapId = POOL[a];

    // ---- the articulation pool: distinct accumulators per slot ----
    auto acc = [](uint32_t seed) {
        Dy::ErrorAccumulator e;
        std::memset(&e, 0, sizeof(e));
        uint8_t* b = reinterpret_cast<uint8_t*>(&e);
        for (size_t i = 0; i < sizeof(e); ++i)
            b[i] = uint8_t(seed * 41u + uint32_t(i) * 7u + 5u);
        return e;
    };
    std::vector<PxgArticulation> pool(POOL_SLOTS);
    std::memset(pool.data(), 0xEE, pool.size() * sizeof(PxgArticulation));
    for (uint32_t s = 0; s < POOL_SLOTS; ++s)
    {
        pool[s].internalResidualAccumulator = acc(100u + s);
        pool[s].contactResidualAccumulator = acc(200u + s);
    }

    // ---- device side ----
    PxgDevicePtr dIsland = 0, dBodySims = 0, dPool = 0, dDesc = 0;
    shim.memAlloc(&dIsland, island.size() * sizeof(PxNodeIndex));
    shim.memAlloc(&dBodySims, bodySims.size() * sizeof(PxgBodySim));
    shim.memAlloc(&dPool, pool.size() * sizeof(PxgArticulation));
    shim.memAlloc(&dDesc, sizeof(PxgArticulationCoreDesc));

    // the OUTPUT is real pinned host memory, exactly as the solver core passes it
    void* hostOut = nullptr;
    shim.memHostAlloc(&hostOut, OUT_SLOTS * sizeof(Dy::ErrorAccumulator));

    if (!dIsland || !dBodySims || !dPool || !dDesc || !hostOut)
    {
        std::printf("  NOTE: no BDA / no pinned host memory -> skip\n");
        return;
    }
    std::memset(hostOut, 0xAB, OUT_SLOTS * sizeof(Dy::ErrorAccumulator));

    PxgArticulationCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dBodySims);
    desc.islandNodeIndices = reinterpret_cast<PxNodeIndex*>(dIsland);
    desc.articulations = reinterpret_cast<PxgArticulation*>(dPool);
    desc.articulationOffset = ARTI_OFFSET;
    desc.nbArticulations = NB_ARTI;

    shim.memcpyHtoD(dIsland, island.data(), island.size() * sizeof(PxNodeIndex));
    shim.memcpyHtoD(dBodySims, bodySims.data(), bodySims.size() * sizeof(PxgBodySim));
    shim.memcpyHtoD(dPool, pool.data(), pool.size() * sizeof(PxgArticulation));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // ---- dispatch: the engine's geometry, ceil(nbArticulations / 256) blocks ----
    struct Params { uint64_t desc, out; } prm = {
        uint64_t(dDesc), uint64_t(reinterpret_cast<uintptr_t>(hostOut))
    };
    const uint32_t numBlocks = (NB_ARTI + 255u) / 256u;
    PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch dmaArticulationResidual");
    shim.streamSynchronize();

    // ---- the host array, read through its CPU mapping ----
    const uint8_t* out = static_cast<const uint8_t*>(hostOut);
    size_t bad = 0;
    auto slotIs = [&](const char* what, uint32_t slot, const void* want) {
        const uint8_t* w = static_cast<const uint8_t*>(want);
        for (size_t i = 0; i < sizeof(Dy::ErrorAccumulator); ++i)
            if (out[slot * sizeof(Dy::ErrorAccumulator) + i] != w[i])
            {
                if (bad < 8)
                    std::printf("  %s slot %u byte %zu: got 0x%02X want 0x%02X\n", what, slot, i,
                                out[slot * sizeof(Dy::ErrorAccumulator) + i], w[i]);
                ++bad;
                return;
            }
    };

    std::vector<bool> written(OUT_SLOTS, false);
    for (uint32_t a = 0; a < NB_ARTI; ++a)
    {
        const uint32_t p = POOL[a];
        slotIs("internal", p, &pool[p].internalResidualAccumulator);
        slotIs("contact", p + NB_ARTI, &pool[p].contactResidualAccumulator);
        written[p] = true;
        written[p + NB_ARTI] = true;
    }
    // every slot the kernel must not touch is still poison -- this is what makes
    // a wrong remap visible rather than merely unlucky
    const std::vector<uint8_t> poison(sizeof(Dy::ErrorAccumulator), 0xAB);
    for (uint32_t s = 0; s < OUT_SLOTS; ++s)
        if (!written[s])
            slotIs("untouched", s, poison.data());

    PXG_CHECK_MSG(bad == 0, "%zu slot(s) wrong", bad);
    std::printf("  dmaArticulationResidual: 3 articulations, island offset 2, nodes 7/3/11 "
                "-> pool slots 1/2/0, written through a real pinned-host pointer\n");

    shim.memFreeHost(hostOut);
}

} // namespace
