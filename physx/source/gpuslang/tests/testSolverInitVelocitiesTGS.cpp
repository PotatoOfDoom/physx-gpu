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

// Verification of solverInitVelocitiesTGS.slang: initializeSolverVelocitiesTGS
// on the ENGINE ABI (P5 Step 4).  A synthetic set of pre-solve body velocities
// + per-(body,slab) remap references (with ~15% invalid) is seeded; the
// kernel's scatter of the repacked vel0/vel1/vel2 triples into the remapped
// slab slots and the accumulation-offset copy are compared bit-exactly against
// a C++ reference.  Remap targets are spaced 96 apart so the [r, r+32, r+64]
// triples never alias, and the output-offset region sits above them, so a full
// float4 comparison of the whole body-velocity buffer (touched slots vs
// preserved sentinel) is exact.
//
// Engine plumbing: a REAL PxgSolverCoreDesc + PxgIslandContext +
// PxgSolverSharedDesc<IterativeSolveDataTGS> carry every count/offset/pool
// pointer; the kernel is launched through PxgCudaShim with the engine's exact
// (desc*, sharedDesc*) pack and the engine's warp-structured grid arithmetic
// (nbBlocks = ceil(32*(mBodyCount+mBodyStartIndex)/256), blockDim (32,8,1) ->
// dispatch nbBlocks*8 one-warp workgroups); numThreadsPerBody is derived from
// numSlabs on-device exactly like the .cu.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

struct F4 { float x, y, z, w; };

const uint32_t INVALID = 0xffffffffu;
const float SENT = -9.0f;

uint32_t nextPow2(uint32_t v) { uint32_t p = 1; while (p < v) p <<= 1; return p; }

static std::vector<uint8_t> initVelLoader(const char* file) { return loadShaderBlob(file); }

void runInitVelocities(uint32_t seed, uint32_t numSlabs)
{
    using namespace physx;

    std::mt19937 rng(seed);
    const uint32_t allBodiesCount = 200;
    const uint32_t bodyOffset = 40;
    const uint32_t numThreadsPerBody = std::min(nextPow2(numSlabs), 32u);
    const uint32_t numBodiesPerWarp = 32u / numThreadsPerBody;
    // remap targets are spaced 96*i, so the slab region ends at 96*nPairs;
    // the accumulation-offset region sits above it
    const uint32_t nPairsTotal = numSlabs * (allBodiesCount - bodyOffset);
    const uint32_t outputOffset = 96u * nPairsTotal + 128u;
    const uint32_t bodyVelElems = outputOffset + 3u * allBodiesCount + 256u;

    // input velocities: lin at [bodyId], ang at [bodyId + allBodiesCount]
    std::vector<F4> outVel(2 * allBodiesCount);
    auto rf = [&]() { return float(int(rng() % 20000) - 10000) * 0.01f; };
    for (auto& v : outVel) v = { rf(), rf(), rf(), rf() };

    // references: one uint per (body,slab) for dynamic bodies [bodyOffset, allBodiesCount)
    const uint32_t nPairs = numSlabs * (allBodiesCount - bodyOffset);
    std::vector<uint32_t> refs(nPairs);
    for (uint32_t i = 0; i < nPairs; ++i)
        refs[i] = (rng() % 100 < 15) ? INVALID : (96u * i); // spaced so triples never alias

    std::vector<F4> bodyVelInit(bodyVelElems, F4{ SENT, SENT, SENT, SENT });

    // ---- C++ reference ----
    std::vector<F4> refBody = bodyVelInit;
    auto vpack = [&](uint32_t bodyId, F4& v0, F4& v1, F4& v2) {
        const F4 lin = outVel[bodyId];
        const F4 ang = outVel[bodyId + allBodiesCount];
        v0 = { lin.x, lin.y, lin.z, ang.x };
        v1 = { ang.y, ang.z, 0.f, 0.f };
        v2 = { 0.f, 0.f, 0.f, 0.f };
    };
    for (uint32_t bodyId = 0; bodyId < allBodiesCount; ++bodyId)
    {
        F4 v0, v1, v2; vpack(bodyId, v0, v1, v2);
        if (bodyId >= bodyOffset)
        {
            for (uint32_t b = 0; b < numSlabs; ++b)
            {
                const uint32_t r = refs[numSlabs * (bodyId - bodyOffset) + b];
                if (r != INVALID) { refBody[r] = v0; refBody[r + 32] = v1; refBody[r + 64] = v2; }
            }
        }
        refBody[outputOffset + bodyId] = v0;
        refBody[outputOffset + bodyId + allBodiesCount] = v1;
        refBody[outputOffset + bodyId + 2 * allBodiesCount] = v2;
    }
    (void)numBodiesPerWarp;

    // ---- GPU (engine ABI through the shim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "solverInitVelocitiesTGS.spv", "initializeSolverVelocitiesTGS", 16u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &initVelLoader);

    PxgDevicePtr dRefs = 0, dBody = 0, dOut = 0, dIsland = 0, dDesc = 0, dShared = 0;
    shim.memAlloc(&dRefs, refs.size() * 4);
    shim.memAlloc(&dBody, bodyVelInit.size() * sizeof(F4));
    shim.memAlloc(&dOut, outVel.size() * sizeof(F4));
    if (!dRefs || !dBody || !dOut) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memcpyHtoD(dRefs, refs.data(), refs.size() * 4);
    shim.memcpyHtoD(dBody, bodyVelInit.data(), bodyVelInit.size() * sizeof(F4));
    shim.memcpyHtoD(dOut, outVel.data(), outVel.size() * sizeof(F4));

    // the REAL engine structs
    PxgIslandContext island;
    std::memset(&island, 0, sizeof(island));
    island.mBodyStartIndex = bodyOffset;
    island.mBodyCount = allBodiesCount - bodyOffset;
    shim.memAlloc(&dIsland, sizeof(island));
    shim.memcpyHtoD(dIsland, &island, sizeof(island));

    PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<PxgIslandContext*>(dIsland);
    desc.solverBodyReferences = reinterpret_cast<PxgSolverReferences*>(dRefs);
    desc.outSolverVelocity = reinterpret_cast<float4*>(dOut);
    desc.numSlabs = numSlabs;
    desc.numSolverBodies = allBodiesCount;
    desc.accumulatedBodyDeltaVOffset = outputOffset;
    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    PxgSolverSharedDesc<IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dBody);
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    // engine grid arithmetic (PxgTGSCudaSolverCore.cpp:1292):
    // nbThreadsRequired = 32 * (mBodyCount + mBodyStartIndex);
    // nbBlocks = ceil(nbThreadsRequired / 256); blockDim (32, 8, 1)
    // -> one-warp workgroups: dispatch nbBlocks * 8
    const uint32_t nbThreadsRequired = 32 * (island.mBodyCount + island.mBodyStartIndex);
    const uint32_t nbBlocksRequired = (nbThreadsRequired + 255) / 256;

    struct { uint64_t desc, shared; } prm = { dDesc, dShared };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocksRequired * 8, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch initializeSolverVelocitiesTGS");
    shim.streamSynchronize();

    std::vector<F4> gBody(bodyVelElems);
    shim.memcpyDtoH(gBody.data(), dBody, gBody.size() * sizeof(F4));
    size_t mm = 0;
    for (uint32_t e = 0; e < bodyVelElems; ++e)
    {
        const F4& a = gBody[e];
        const F4& r = refBody[e];
        if (a.x != r.x || a.y != r.y || a.z != r.z || a.w != r.w) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "initializeSolverVelocitiesTGS(slabs=%u): %zu mismatches", numSlabs, mm);

    for (PxgDevicePtr pfree : { dRefs, dBody, dOut, dIsland, dDesc, dShared })
        shim.memFree(pfree);
}

PXG_TEST(solverInitVelocitiesTGS_scatter)
{
    runInitVelocities(36010, 4);
    runInitVelocities(36011, 1);  // numThreadsPerBody = 1 (32 bodies/warp)
    runInitVelocities(36012, 12); // non-pow2 -> numThreadsPerBody = 16
}

} // namespace
