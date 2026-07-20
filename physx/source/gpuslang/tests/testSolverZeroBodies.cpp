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

// Verification of solverZeroBodies.slang: the ZeroBodies (PGS) and ZeroBodiesTGS
// pre-solve reset kernels.  Both float4 buffers are pre-filled with a nonzero
// sentinel; after the launch the kernel-defined ranges must be exactly zero and
// everything outside them must retain the sentinel (catching over- and
// under-zeroing).  A short grid stride forces the grid-stride loop to iterate.
//
// ZeroBodiesTGS runs on the ENGINE ABI (P5 Step 4): a REAL PxgSolverCoreDesc +
// PxgSolverSharedDesc<IterativeSolveDataTGS> pair is filled with live device
// addresses and counts, and the kernel is launched through PxgCudaShim with
// the engine's exact (desc*, sharedDesc*) param pack and fixed
// ZERO_BODIES = 64x256 grid, deriving both zeroed ranges on-device.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct ZeroBodiesParams { uint32_t totalNumBodies, totalNumBodiesConstraints, offset, gridStride; };

const float SENT = 7.5f;

// verify buffer[4*e .. 4*e+3] is zero if inZeroed(e) else SENT, over [0,nElems)
template <class Pred>
size_t checkFloat4(const std::vector<float>& buf, uint32_t nElems, Pred inZeroed)
{
    size_t mm = 0;
    for (uint32_t e = 0; e < nElems; ++e)
    {
        const float want = inZeroed(e) ? 0.0f : SENT;
        for (uint32_t c = 0; c < 4; ++c)
            if (buf[4 * e + c] != want) { ++mm; break; }
    }
    return mm;
}

PXG_TEST(solverZeroBodies_pgs)
{
    const uint32_t numSolverBodies = 500, numBatches = 20, numArticBatches = 4;
    const uint32_t totalNumBodies = numSolverBodies * 2;                                 // 1000
    const uint32_t totalNumBodiesConstraints = (numBatches + numArticBatches) * 32 * 2 * 2; // 3072
    const uint32_t offset = 5000;                                                        // gap [3072,5000) preserved
    const uint32_t motionElems = 2000;   // zero [0,1000), preserve [1000,2000)
    const uint32_t bodyElems = 8000;     // zero [0,3072) U [5000,6000)

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = 4;
    const uint32_t gridStride = GROUPS * 256; // 1024 < ranges -> loop iterates

    std::vector<float> motionInit(4 * motionElems, SENT), bodyInit(4 * bodyElems, SENT);
    PxgBufferHandle bMotion = createBufferFrom(dev, motionInit);
    PxgBufferHandle bBody = createBufferFrom(dev, bodyInit);

    Kernel k("solverZeroBodies.spv", "ZeroBodies", { 16, 16 }, sizeof(ZeroBodiesParams));
    ZeroBodiesParams p = { totalNumBodies, totalNumBodiesConstraints, offset, gridStride };
    k.dispatch(GROUPS, { bMotion, bBody }, p);
    dev->flush();

    const auto gMotion = downloadBuffer<float>(dev, bMotion, 4 * motionElems);
    const auto gBody = downloadBuffer<float>(dev, bBody, 4 * bodyElems);

    const size_t mmMotion = checkFloat4(gMotion, motionElems,
        [&](uint32_t e) { return e < totalNumBodies; });
    PXG_CHECK_MSG(mmMotion == 0, "ZeroBodies motion: %zu mismatches", mmMotion);

    const size_t mmBody = checkFloat4(gBody, bodyElems, [&](uint32_t e) {
        return e < totalNumBodiesConstraints || (e >= offset && e < offset + totalNumBodies);
    });
    PXG_CHECK_MSG(mmBody == 0, "ZeroBodies body: %zu mismatches", mmBody);

    dev->destroyBuffer(bMotion);
    dev->destroyBuffer(bBody);
}

static std::vector<uint8_t> zeroBodiesLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(solverZeroBodies_tgs)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    const uint32_t numSolverBodies = 500, numBatches = 20, numArticBatches = 4;
    const uint32_t totalNumBodies = numSolverBodies * 2;                                    // 1000
    const uint32_t totalNumBodiesConstraints =
        (numArticBatches + numBatches) * 32 * 3 * 2 + numSolverBodies * 3;                  // 6108
    const uint32_t motionElems = 2000;   // zero [0,1000)
    const uint32_t bodyElems = 8000;     // zero [0,6108), preserve [6108,8000)

    PxgSlangComputeDevice* dev = testDevice();
    // engine ABI: (const PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveDataTGS>*)
    const PxgKernelDef manifest[] = { { "solverZeroBodies.spv", "ZeroBodiesTGS", 16u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &zeroBodiesLoader);

    std::vector<float> motionInit(4 * motionElems, SENT), bodyInit(4 * bodyElems, SENT);
    PxgDevicePtr dMotion = 0, dBody = 0, dDesc = 0, dShared = 0;
    shim.memAlloc(&dMotion, motionInit.size() * 4);
    shim.memAlloc(&dBody, bodyInit.size() * 4);
    if (!dMotion || !dBody) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memcpyHtoD(dMotion, motionInit.data(), motionInit.size() * 4);
    shim.memcpyHtoD(dBody, bodyInit.data(), bodyInit.size() * 4);

    // the REAL engine descs, with live device addresses + counts
    PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.motionVelocityArray = reinterpret_cast<float4*>(dMotion);
    desc.numSolverBodies = numSolverBodies;
    desc.numBatches = numBatches;
    desc.numArticBatches = numArticBatches;

    PxgSolverSharedDesc<IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dBody);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    // PxgSolverCore::resetVelocities: fixed grid ZERO_BODIES = 64 blocks x 256
    struct { uint64_t desc, shared; } prm = { dDesc, dShared };
    PXG_CHECK_MSG(shim.launchKernel(0, 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch ZeroBodiesTGS");
    shim.streamSynchronize();

    std::vector<float> gMotion(4 * motionElems), gBody(4 * bodyElems);
    shim.memcpyDtoH(gMotion.data(), dMotion, gMotion.size() * 4);
    shim.memcpyDtoH(gBody.data(), dBody, gBody.size() * 4);

    const size_t mmMotion = checkFloat4(gMotion, motionElems,
        [&](uint32_t e) { return e < totalNumBodies; });
    PXG_CHECK_MSG(mmMotion == 0, "ZeroBodiesTGS motion: %zu mismatches", mmMotion);

    const size_t mmBody = checkFloat4(gBody, bodyElems,
        [&](uint32_t e) { return e < totalNumBodiesConstraints; });
    PXG_CHECK_MSG(mmBody == 0, "ZeroBodiesTGS body: %zu mismatches", mmBody);

    for (PxgDevicePtr pfree : { dMotion, dBody, dDesc, dShared })
        shim.memFree(pfree);
}

} // namespace
