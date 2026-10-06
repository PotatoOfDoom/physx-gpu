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
// + per-(body,slab) remap references (with ~15% invalid, plus directed bodies
// with NO valid reference) is seeded; the kernel's scatter of the repacked
// vel0/vel1/vel2 triples into the remapped slab slots and the
// accumulation-offset copy are compared BITWISE (memcmp, sign of zero counts)
// against a C++ oracle.  Remap targets are spaced 96 apart so the
// [r, r+32, r+64] triples never alias, and the output-offset region sits above
// them, so a full float4 comparison of the whole body-velocity buffer (touched
// slots vs preserved sentinel) is exact.
//
// Engine plumbing: a REAL PxgSolverCoreDesc + PxgIslandContext +
// PxgSolverSharedDesc<IterativeSolveDataTGS> carry every count/offset/pool
// pointer; the kernel is launched through PxgCudaShim with the engine's exact
// (desc*, sharedDesc*) pack and the engine's grid, UNSCALED:
// nbBlocks = ceil(32*(mBodyCount+mBodyStartIndex)/256) (PxgTGSCudaSolverCore.cpp:1295-1306).
// The bridge drops blockDim (32, 8, 1) and dispatches that grid as it is, so
// the shader's (32, 8) workgroup IS the CUDA block.  Up to P9.97a this file
// dispatched nbBlocks*8 one-warp workgroups -- a multiplication nothing in the
// engine performs -- and so could not see that the shipped [numthreads(32,1,1)]
// reached one eighth of the CUDA's warps (every body past ceil(T/8)*bpw was
// never seeded for numSlabs >= 5).  The oracle now emulates the CUDA launch
// literally (blockIdx, threadIdx.y, threadIdx.x) and counts the accumulation
// copies it writes; numThreadsPerBody is derived from numSlabs on-device
// exactly like the .cu.

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

// allBodiesCount = T = mBodyCount + mBodyStartIndex, bodyOffset = O = mBodyStartIndex
void runInitVelocities(uint32_t seed, uint32_t numSlabs, uint32_t allBodiesCount, uint32_t bodyOffset)
{
    using namespace physx;

    std::mt19937 rng(seed);
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
    // DIRECTED: every 7th dynamic body has no valid slab reference at all.  It gets no
    // slab triple, but its accumulation-offset copy must still be written.
    uint32_t noRefBodies = 0;
    for (uint32_t bodyId = bodyOffset + 3; bodyId < allBodiesCount; bodyId += 7, ++noRefBodies)
        for (uint32_t b = 0; b < numSlabs; ++b)
            refs[numSlabs * (bodyId - bodyOffset) + b] = INVALID;
    PXG_CHECK_MSG(noRefBodies >= 1, "T=%u O=%u: no reference-less body seeded", allBodiesCount, bodyOffset);

    std::vector<F4> bodyVelInit(bodyVelElems, F4{ SENT, SENT, SENT, SENT });

    // ---- oracle: the CUDA launch, emulated literally ----
    // PxgTGSCudaSolverCore.cpp:1295-1306: grid G = ceil(32*T/256), block (32, 8, 1).
    // solverMultiBlockTGS.cu:689-710: warpIndex = blockIdx.x*blockDim.y + threadIdx.y,
    // bodyId = warpIndex*numBodiesPerWarp + threadIdx.x/numThreadsPerBody, guard bodyId < T.
    const uint32_t nbThreadsRequired = 32 * allBodiesCount;
    const uint32_t nbBlocksRequired = (nbThreadsRequired + 255) / 256;
    std::vector<F4> refBody = bodyVelInit;
    uint32_t written = 0;
    for (uint32_t blockIdx = 0; blockIdx < nbBlocksRequired; ++blockIdx)
    for (uint32_t threadIdxY = 0; threadIdxY < 8; ++threadIdxY)
    for (uint32_t threadIdxX = 0; threadIdxX < 32; ++threadIdxX)
    {
        const uint32_t warpIndex = blockIdx * 8 + threadIdxY;
        const uint32_t threadIdxInWorkUnit = threadIdxX & (numThreadsPerBody - 1);
        const uint32_t bodyId = warpIndex * numBodiesPerWarp + threadIdxX / numThreadsPerBody;
        if (bodyId >= allBodiesCount)
            continue;
        const F4 lin = outVel[bodyId];
        const F4 ang = outVel[bodyId + allBodiesCount];
        const F4 v0 = { lin.x, lin.y, lin.z, ang.x };
        const F4 v1 = { ang.y, ang.z, 0.f, 0.f };
        const F4 v2 = { 0.f, 0.f, 0.f, 0.f };
        if (bodyId >= bodyOffset)
        {
            for (uint32_t b = threadIdxInWorkUnit; b < numSlabs; b += numThreadsPerBody)
            {
                const uint32_t r = refs[numSlabs * (bodyId - bodyOffset) + b];
                if (r != INVALID) { refBody[r] = v0; refBody[r + 32] = v1; refBody[r + 64] = v2; }
            }
        }
        if (threadIdxInWorkUnit == 0)
        {
            refBody[outputOffset + bodyId] = v0;
            refBody[outputOffset + bodyId + allBodiesCount] = v1;
            refBody[outputOffset + bodyId + 2 * allBodiesCount] = v2;
            ++written;
        }
    }
    // the CUDA's init grid reaches 8*G*bpw >= T bodies: every body is seeded
    PXG_CHECK_MSG(written == allBodiesCount, "oracle(slabs=%u T=%u): %u accumulation copies, want %u",
                  numSlabs, allBodiesCount, written, allBodiesCount);

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

    // the engine's grid, dispatched unscaled (see the header)
    PXG_CHECK_MSG(nbThreadsRequired == 32 * (island.mBodyCount + island.mBodyStartIndex), "grid arithmetic");
    struct { uint64_t desc, shared; } prm = { dDesc, dShared };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocksRequired, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch initializeSolverVelocitiesTGS");
    shim.streamSynchronize();

    std::vector<F4> gBody(bodyVelElems);
    shim.memcpyDtoH(gBody.data(), dBody, gBody.size() * sizeof(F4));
    size_t mm = 0, accMissing = 0;
    uint32_t firstMissing = INVALID;
    for (uint32_t e = 0; e < bodyVelElems; ++e)
    {
        if (std::memcmp(&gBody[e], &refBody[e], sizeof(F4)) == 0)
            continue;
        ++mm;
        if (e >= outputOffset && e < outputOffset + allBodiesCount)
        {
            ++accMissing;
            if (firstMissing == INVALID) firstMissing = e - outputOffset;
        }
    }
    PXG_CHECK_MSG(mm == 0, "initializeSolverVelocitiesTGS(slabs=%u T=%u O=%u G=%u): %zu float4 mismatches, "
                  "%zu accumulation copies wrong (first body %d)", numSlabs, allBodiesCount, bodyOffset,
                  nbBlocksRequired, mm, accMissing, firstMissing == INVALID ? -1 : int(firstMissing));

    for (PxgDevicePtr pfree : { dRefs, dBody, dOut, dIsland, dDesc, dShared })
        shim.memFree(pfree);
}

PXG_TEST(solverInitVelocitiesTGS_scatter)
{
    runInitVelocities(36010, 4, 200, 40);
    runInitVelocities(36011, 1, 200, 40);  // numThreadsPerBody = 1 (32 bodies/warp)
    runInitVelocities(36012, 12, 200, 40); // non-pow2 -> numThreadsPerBody = 16
}

// P9.97a: the engine grid at every numThreadsPerBody band.  The shipped one-warp
// group reached G*bpw bodies, so numSlabs >= 5 left 1/2, 3/4 or 7/8 of them
// unseeded; (200, 40) is the tight case (G = 25, 8G = T exactly at bpw 1), (13, 1)
// is below one block, (201, 1) puts the last body alone in a new block.
PXG_TEST(solverInitVelocitiesTGS_engineSlabCoverage)
{
    const uint32_t slabs[] = { 1u, 2u, 4u, 5u, 12u, 32u, 33u };
    const uint32_t shapes[][2] = { { 13u, 1u }, { 200u, 40u }, { 201u, 1u } };
    uint32_t seed = 97100;
    for (uint32_t ns : slabs)
        for (const auto& tc : shapes)
            runInitVelocities(seed++, ns, tc[0], tc[1]);
}

} // namespace
