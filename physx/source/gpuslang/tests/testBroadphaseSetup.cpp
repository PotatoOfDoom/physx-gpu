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

// Verification of broadphaseSetup.slang (translateAABBsLaunch,
// markCreatedPairsLaunch, initializeSapBox1DLaunch) against C++ transcriptions
// of the CUDA broad-phase foundation kernels.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
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

const uint32_t PX_SIGN = 0x80000000u;
const uint32_t INVALID_U32 = 0xffffffffu;
const uint32_t INVALID_PROJ = 0xffffffffu;
const uint32_t INVALID_HANDLE = 0x3fffffffu;
const uint32_t ENC_MIN = 0x01800000u;
const uint32_t ENC_MAX = 0xfe7fffffu;

struct IntegerAABB
{
    uint32_t mMinMax[6];
};
struct SapBox1D
{
    uint32_t mMinMax[2];
};

struct BpParams
{
    uint32_t boxesCapacity;
    uint32_t shiftX, shiftY, shiftZ;
    uint32_t envShiftX, envShiftY, envShiftZ;
    uint32_t hasEnvIds;
    uint32_t numCreatedHandles;
    uint32_t numPreviousHandles;
    uint32_t numHandles;
    uint32_t numProjections;
};

uint32_t encodeFloat(float f)
{
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return (i & PX_SIGN) ? ~i : (i | PX_SIGN);
}
uint32_t createHandle(uint32_t h, bool isStart, bool isNew)
{
    return (h << 3) | (isStart ? 1u : 0u) | (isNew ? 2u : 0u);
}

// ---------------------------------------------------------------------------
// Test 1: translateAABBsLaunch — P5 Step 4: driven through PxgCudaShim on the
// ENGINE ABI.  The param blob is assembled from a CUDA-shaped argument list
// (4 device pointers, PxU32 capacity, six PxU8 shifts) by the SAME natural-
// alignment rule PxgVulkanCudaContext::launchKernel uses, so this validates the
// blob layout (u8 tail at bytes 36..41, rounded to 48) against the compiled
// push-constant struct, the BDA pointer walk, and the 8-threads-per-box grid
// arithmetic the engine launches with.
// ---------------------------------------------------------------------------

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

static std::vector<uint8_t> bpShimLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(broadphaseSetup_translateAABBs)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(6100);
    std::uniform_real_distribution<float> pos(-500.f, 500.f);
    std::uniform_real_distribution<float> ext(0.01f, 50.f);
    std::uniform_real_distribution<float> cd(0.f, 2.f);

    PxgSlangComputeDevice* dev = testDevice();
    // Engine ABI blob: 4x u64 + u32 + 6x u8 = 42 -> rounded to 48.
    const PxgKernelDef manifest[] = {
        { "broadphaseSetup.spv", "translateAABBsLaunch", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    for (uint32_t variant = 0; variant < 3; ++variant)
    {
        const uint32_t capacity = 300;
        const bool hasEnvIds = variant >= 1;
        const uint8_t shiftX = variant == 2 ? 1u : 0u;
        const uint8_t shiftY = variant == 2 ? 2u : 0u;
        const uint8_t shiftZ = 0u;
        const uint8_t envShiftX = hasEnvIds ? 3u : 0u;
        const uint8_t envShiftY = hasEnvIds ? 3u : 0u;
        const uint8_t envShiftZ = hasEnvIds ? 0u : 0u; // exercise the envShift==0 fallthrough

        std::vector<float> inBounds(capacity * 6);
        std::vector<float> contactDist(capacity);
        std::vector<uint32_t> envIds(capacity);
        for (uint32_t i = 0; i < capacity; ++i)
        {
            const float mnx = pos(rng), mny = pos(rng), mnz = pos(rng);
            inBounds[i * 6 + 0] = mnx;
            inBounds[i * 6 + 1] = mny;
            inBounds[i * 6 + 2] = mnz;
            inBounds[i * 6 + 3] = mnx + ext(rng);
            inBounds[i * 6 + 4] = mny + ext(rng);
            inBounds[i * 6 + 5] = mnz + ext(rng);
            contactDist[i] = cd(rng);
            envIds[i] = (rng() % 5 == 0) ? INVALID_U32 : (rng() % 16);
        }

        PxgDevicePtr dIn = 0, dOut = 0, dCd = 0, dEnv = 0;
        PXG_CHECK_MSG(shim.memAlloc(&dIn, inBounds.size() * 4) == PxgResult::Success, "memAlloc in");
        PXG_CHECK_MSG(shim.memAlloc(&dOut, capacity * sizeof(IntegerAABB)) == PxgResult::Success, "memAlloc out");
        PXG_CHECK_MSG(shim.memAlloc(&dCd, contactDist.size() * 4) == PxgResult::Success, "memAlloc cd");
        if (hasEnvIds)
            PXG_CHECK_MSG(shim.memAlloc(&dEnv, envIds.size() * 4) == PxgResult::Success, "memAlloc env");
        if (!dIn || !dOut || !dCd)
        {
            std::printf("  NOTE: shim has no device-address allocations (no BDA) -> skip\n");
            return;
        }
        shim.memcpyHtoD(dIn, inBounds.data(), inBounds.size() * 4);
        shim.memcpyHtoD(dCd, contactDist.data(), contactDist.size() * 4);
        if (hasEnvIds)
            shim.memcpyHtoD(dEnv, envIds.data(), envIds.size() * 4);
        shim.memsetD32(dOut, 0xcccccccc, capacity * 6);

        // The CUDA argument list, packed by the bridge's rule.  envIds == 0 (null)
        // when absent — the kernel branches on the pointer, as the .cu does.
        EngineBlob blob;
        blob.add<uint64_t>(dIn);
        blob.add<uint64_t>(dOut);
        blob.add<uint64_t>(dCd);
        blob.add<uint64_t>(hasEnvIds ? dEnv : 0);
        blob.add<uint32_t>(capacity);
        blob.add<uint8_t>(shiftX);
        blob.add<uint8_t>(shiftY);
        blob.add<uint8_t>(shiftZ);
        blob.add<uint8_t>(envShiftX);
        blob.add<uint8_t>(envShiftY);
        blob.add<uint8_t>(envShiftZ);
        blob.finish();
        PXG_CHECK_MSG(blob.bytes.size() == 48, "engine blob is 48 bytes (got %zu)", blob.bytes.size());

        // Engine grid arithmetic: 8 threads/box, blockDim 256 -> 32 boxes/block.
        const uint32_t aabbsPerBlock = 256 / 8;
        const uint32_t nbBlocks = (capacity + aabbsPerBlock - 1) / aabbsPerBlock;
        PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) == PxgResult::Success,
                      "launchKernel translateAABBs through engine-ABI blob");
        shim.streamSynchronize();

        // ---- reference ----
        const uint32_t shifts[6] = { shiftX, shiftY, shiftZ, shiftX, shiftY, shiftZ };
        const uint32_t envShifts[6] = { envShiftX, envShiftY, envShiftZ, envShiftX, envShiftY, envShiftZ };
        std::vector<IntegerAABB> ref(capacity);
        for (uint32_t i = 0; i < capacity; ++i)
        {
            const float contactDistance = contactDist[i];
            for (uint32_t a = 0; a < 6; ++a)
            {
                const float in = inBounds[i * 6 + a];
                const uint32_t mask = a < 3 ? 0u : 1u;
                const float projection = a < 3 ? in - contactDistance : in + contactDistance;
                const uint32_t b = (encodeFloat(projection) >> shifts[a]) & (~1u);
                uint32_t out = b | mask;
                if (hasEnvIds && envShifts[a])
                {
                    const uint32_t envId = envIds[i];
                    if (envId != INVALID_U32)
                        out |= envId << (32 - envShifts[a]);
                    else
                        out = mask ? ENC_MAX : ENC_MIN;
                }
                ref[i].mMinMax[a] = out;
            }
        }

        std::vector<IntegerAABB> g(capacity);
        PXG_CHECK_MSG(shim.memcpyDtoH(g.data(), dOut, capacity * sizeof(IntegerAABB)) == PxgResult::Success,
                      "DtoH out");
        size_t mism = 0;
        for (uint32_t i = 0; i < capacity && mism < 8; ++i)
            for (uint32_t a = 0; a < 6; ++a)
                if (g[i].mMinMax[a] != ref[i].mMinMax[a])
                {
                    if (mism < 8)
                        PXG_CHECK_MSG(false, "variant %u box %u comp %u: gpu=%08x ref=%08x", variant, i, a,
                                      g[i].mMinMax[a], ref[i].mMinMax[a]);
                    ++mism;
                }
        PXG_CHECK_MSG(mism == 0, "variant %u: %zu translate mismatches", variant, mism);

        // sanity: min components even, max components odd (encoding invariant),
        // only when env packing did not override them
        if (!hasEnvIds)
        {
            for (uint32_t i = 0; i < capacity; ++i)
            {
                PXG_CHECK_MSG((g[i].mMinMax[0] & 1) == 0 && (g[i].mMinMax[3] & 1) == 1,
                              "min/max parity broken at box %u", i);
            }
        }

        shim.memFree(dIn);
        shim.memFree(dOut);
        shim.memFree(dCd);
        if (dEnv)
            shim.memFree(dEnv);
    }
}

// ---------------------------------------------------------------------------
// Tests 2-5: the bpDesc-aggregate kernels on the ENGINE ABI — each fills a
// REAL PxgBroadPhaseDesc with live device addresses and launches through
// PxgCudaShim exactly as the engine does (one bpDesc pointer, void** form).
// ---------------------------------------------------------------------------

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

PXG_TEST(broadphaseSetup_markCreatedPairs)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(6200);
    const uint32_t numPrevious = 37; // odd on purpose so total may be odd
    const uint32_t numCreated = 90;
    const uint32_t numBounds = 256;

    std::vector<IntegerAABB> newBounds(numBounds);
    for (uint32_t h = 0; h < numBounds; ++h)
        for (uint32_t a = 0; a < 6; ++a)
            newBounds[h].mMinMax[a] = (h << 8) | (a << 4) | 0x3;

    std::vector<uint32_t> createdHandles(numCreated);
    std::vector<uint32_t> perm(numBounds);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (uint32_t i = 0; i < numCreated; ++i)
        createdHandles[i] = perm[i];

    const uint32_t numExisting = numPrevious + numCreated;
    const uint32_t total = numExisting + (numExisting & 1);
    const uint32_t projLen = total * 2;

    std::vector<uint32_t> projX(projLen, 0x11111111u), projY(projLen, 0x22222222u), projZ(projLen, 0x33333333u);
    std::vector<uint32_t> handX(projLen, 0x44444444u), handY(projLen, 0x55555555u), handZ(projLen, 0x66666666u);
    std::vector<IntegerAABB> oldBounds(numBounds);
    std::memset(oldBounds.data(), 0, oldBounds.size() * sizeof(IntegerAABB));

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseSetup.spv", "markCreatedPairsLaunch", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    BpDescRig rig(&shim);
    const PxgDevicePtr dCh = rig.alloc(createdHandles.data(), createdHandles.size() * 4);
    const PxgDevicePtr dPX = rig.alloc(projX.data(), projX.size() * 4);
    const PxgDevicePtr dPY = rig.alloc(projY.data(), projY.size() * 4);
    const PxgDevicePtr dPZ = rig.alloc(projZ.data(), projZ.size() * 4);
    const PxgDevicePtr dHX = rig.alloc(handX.data(), handX.size() * 4);
    const PxgDevicePtr dHY = rig.alloc(handY.data(), handY.size() * 4);
    const PxgDevicePtr dHZ = rig.alloc(handZ.data(), handZ.size() * 4);
    const PxgDevicePtr dNB = rig.alloc(newBounds.data(), newBounds.size() * sizeof(IntegerAABB));
    const PxgDevicePtr dOB = rig.alloc(oldBounds.data(), oldBounds.size() * sizeof(IntegerAABB));
    if (!dCh) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.numCreatedHandles = numCreated;
    rig.desc.numPreviousHandles = numPrevious;
    rig.desc.updateData_createdHandles = reinterpret_cast<PxU32*>(dCh);
    rig.desc.boxProjections[0] = reinterpret_cast<PxU32*>(dPX);
    rig.desc.boxProjections[1] = reinterpret_cast<PxU32*>(dPY);
    rig.desc.boxProjections[2] = reinterpret_cast<PxU32*>(dPZ);
    rig.desc.boxHandles[0][0] = reinterpret_cast<PxU32*>(dHX);
    rig.desc.boxHandles[0][1] = reinterpret_cast<PxU32*>(dHY);
    rig.desc.boxHandles[0][2] = reinterpret_cast<PxU32*>(dHZ);
    rig.desc.newIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dNB);
    rig.desc.oldIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(dOB);
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grid: BP_UPDATE_CREATEDPAIRS = 32 blocks x 256 (grid-stride)
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch markCreatedPairs");
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<uint32_t> rPX = projX, rPY = projY, rPZ = projZ, rHX = handX, rHY = handY, rHZ = handZ;
    std::vector<IntegerAABB> rOld = oldBounds;
    for (uint32_t i = 0; i < numCreated; ++i)
    {
        const uint32_t handle = createdHandles[i];
        const uint32_t index = numPrevious + i;
        const uint32_t sh = createHandle(handle, true, true), eh = createHandle(handle, false, true);
        rPX[index * 2 + 0] = newBounds[handle].mMinMax[0]; rPX[index * 2 + 1] = newBounds[handle].mMinMax[3];
        rPY[index * 2 + 0] = newBounds[handle].mMinMax[1]; rPY[index * 2 + 1] = newBounds[handle].mMinMax[4];
        rPZ[index * 2 + 0] = newBounds[handle].mMinMax[2]; rPZ[index * 2 + 1] = newBounds[handle].mMinMax[5];
        rHX[index * 2 + 0] = sh; rHX[index * 2 + 1] = eh;
        rHY[index * 2 + 0] = sh; rHY[index * 2 + 1] = eh;
        rHZ[index * 2 + 0] = sh; rHZ[index * 2 + 1] = eh;
        for (uint32_t a = 0; a < 3; ++a) rOld[handle].mMinMax[a] = 0xff7fffffu;
        for (uint32_t a = 3; a < 6; ++a) rOld[handle].mMinMax[a] = 0x00800000u;
    }
    if (numExisting & 1)
    {
        const uint32_t index = numExisting;
        rPX[index * 2 + 0] = rPX[index * 2 + 1] = INVALID_PROJ;
        rPY[index * 2 + 0] = rPY[index * 2 + 1] = INVALID_PROJ;
        rPZ[index * 2 + 0] = rPZ[index * 2 + 1] = INVALID_PROJ;
        rHX[index * 2 + 0] = rHX[index * 2 + 1] = INVALID_HANDLE;
        rHY[index * 2 + 0] = rHY[index * 2 + 1] = INVALID_HANDLE;
        rHZ[index * 2 + 0] = rHZ[index * 2 + 1] = INVALID_HANDLE;
    }

    std::vector<uint32_t> gPX(projLen), gPY(projLen), gPZ(projLen), gHX(projLen), gHY(projLen), gHZ(projLen);
    std::vector<IntegerAABB> gOld(numBounds);
    shim.memcpyDtoH(gPX.data(), dPX, projLen * 4);
    shim.memcpyDtoH(gPY.data(), dPY, projLen * 4);
    shim.memcpyDtoH(gPZ.data(), dPZ, projLen * 4);
    shim.memcpyDtoH(gHX.data(), dHX, projLen * 4);
    shim.memcpyDtoH(gHY.data(), dHY, projLen * 4);
    shim.memcpyDtoH(gHZ.data(), dHZ, projLen * 4);
    shim.memcpyDtoH(gOld.data(), dOB, numBounds * sizeof(IntegerAABB));

    size_t mism = 0;
    for (uint32_t i = 0; i < projLen; ++i)
    {
        if (gPX[i] != rPX[i] || gPY[i] != rPY[i] || gPZ[i] != rPZ[i]) ++mism;
        if (gHX[i] != rHX[i] || gHY[i] != rHY[i] || gHZ[i] != rHZ[i]) ++mism;
    }
    for (uint32_t h = 0; h < numBounds; ++h)
        for (uint32_t a = 0; a < 6; ++a)
            if (gOld[h].mMinMax[a] != rOld[h].mMinMax[a]) ++mism;
    PXG_CHECK_MSG(mism == 0, "%zu markCreated mismatches", mism);

    for (PxgDevicePtr pfree : { dCh, dPX, dPY, dPZ, dHX, dHY, dHZ, dNB, dOB, dDesc })
        shim.memFree(pfree);
}

PXG_TEST(broadphaseSetup_initializeSapBox1D)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(6300);
    const uint32_t numHandles = 200;
    const uint32_t numProjections = numHandles * 2;

    auto buildAxis = [&](std::vector<uint32_t>& handles) {
        std::vector<uint32_t> order(numProjections);
        std::iota(order.begin(), order.end(), 0u);
        std::shuffle(order.begin(), order.end(), rng);
        handles.resize(numProjections);
        for (uint32_t j = 0; j < numProjections; ++j)
        {
            const uint32_t box = order[j] >> 1;
            const bool isStart = (order[j] & 1) == 0;
            handles[j] = createHandle(box, isStart, false);
        }
    };
    std::vector<uint32_t> handX, handY, handZ;
    buildAxis(handX);
    buildAxis(handY);
    buildAxis(handZ);

    std::vector<SapBox1D> sapInit(numHandles);
    std::memset(sapInit.data(), 0xff, sapInit.size() * sizeof(SapBox1D));

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseSetup.spv", "initializeSapBox1DLaunch", 16u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    // both variants: isNew=false -> boxSapBox1D, isNew=true -> boxNewSapBox1D
    for (uint32_t variant = 0; variant < 2; ++variant)
    {
        const bool isNew = variant == 1;
        BpDescRig rig(&shim);
        const PxgDevicePtr dHX = rig.alloc(handX.data(), handX.size() * 4);
        const PxgDevicePtr dHY = rig.alloc(handY.data(), handY.size() * 4);
        const PxgDevicePtr dHZ = rig.alloc(handZ.data(), handZ.size() * 4);
        const PxgDevicePtr dSX = rig.alloc(sapInit.data(), sapInit.size() * sizeof(SapBox1D));
        const PxgDevicePtr dSY = rig.alloc(sapInit.data(), sapInit.size() * sizeof(SapBox1D));
        const PxgDevicePtr dSZ = rig.alloc(sapInit.data(), sapInit.size() * sizeof(SapBox1D));
        if (!dHX) { std::printf("  NOTE: no BDA -> skip\n"); return; }
        rig.desc.boxHandles[0][0] = reinterpret_cast<PxU32*>(dHX);
        rig.desc.boxHandles[0][1] = reinterpret_cast<PxU32*>(dHY);
        rig.desc.boxHandles[0][2] = reinterpret_cast<PxU32*>(dHZ);
        PxgSapBox1D** sapArr = isNew ? rig.desc.boxNewSapBox1D : rig.desc.boxSapBox1D;
        sapArr[0] = reinterpret_cast<PxgSapBox1D*>(dSX);
        sapArr[1] = reinterpret_cast<PxgSapBox1D*>(dSY);
        sapArr[2] = reinterpret_cast<PxgSapBox1D*>(dSZ);
        const PxgDevicePtr dDesc = rig.upload();

        // engine ABI: (bpDesc*, PxU32 numHandles, bool isNew) via the bridge blob rule
        EngineBlob blob;
        blob.add<uint64_t>(dDesc);
        blob.add<uint32_t>(numHandles);
        blob.add<uint8_t>(isNew ? 1u : 0u);
        blob.finish();
        PXG_CHECK_MSG(blob.bytes.size() == 16, "sapbox blob is 16 bytes (got %zu)", blob.bytes.size());
        PXG_CHECK_MSG(shim.launchKernel(0, (numProjections + 255) / 256, 1, 1, blob.bytes.data(),
                                        uint32_t(blob.bytes.size())) == PxgResult::Success, "launch sapbox init");
        shim.streamSynchronize();

        auto refAxis = [&](const std::vector<uint32_t>& handles, std::vector<SapBox1D>& sap) {
            for (uint32_t i = 0; i < numProjections; ++i)
            {
                const uint32_t sh = handles[i];
                const uint32_t handle = sh >> 3;
                const bool isStart = (sh & 1) != 0;
                sap[handle].mMinMax[isStart ? 0 : 1] = i;
            }
        };
        std::vector<SapBox1D> rX = sapInit, rY = sapInit, rZ = sapInit;
        refAxis(handX, rX);
        refAxis(handY, rY);
        refAxis(handZ, rZ);

        std::vector<SapBox1D> gX(numHandles), gY(numHandles), gZ(numHandles);
        shim.memcpyDtoH(gX.data(), dSX, numHandles * sizeof(SapBox1D));
        shim.memcpyDtoH(gY.data(), dSY, numHandles * sizeof(SapBox1D));
        shim.memcpyDtoH(gZ.data(), dSZ, numHandles * sizeof(SapBox1D));

        size_t mism = 0;
        for (uint32_t h = 0; h < numHandles; ++h)
        {
            if (gX[h].mMinMax[0] != rX[h].mMinMax[0] || gX[h].mMinMax[1] != rX[h].mMinMax[1]) ++mism;
            if (gY[h].mMinMax[0] != rY[h].mMinMax[0] || gY[h].mMinMax[1] != rY[h].mMinMax[1]) ++mism;
            if (gZ[h].mMinMax[0] != rZ[h].mMinMax[0] || gZ[h].mMinMax[1] != rZ[h].mMinMax[1]) ++mism;
        }
        PXG_CHECK_MSG(mism == 0, "variant %u: %zu SAP box mismatches", variant, mism);

        for (PxgDevicePtr pfree : { dHX, dHY, dHZ, dSX, dSY, dSZ, dDesc })
            shim.memFree(pfree);
    }
}

PXG_TEST(broadphaseSetup_initializeRadixRanks)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    const uint32_t numPrevious = 151; // odd; numProjections = 302, padded to 304
    const uint32_t numProjections = numPrevious * 2;
    const uint32_t padded = ((numProjections + 3) / 4) * 4;

    std::vector<uint32_t> init(padded, 0xeeeeeeeeu);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseSetup.spv", "initializeRadixRanks", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    BpDescRig rig(&shim);
    const PxgDevicePtr dRX = rig.alloc(init.data(), init.size() * 4);
    const PxgDevicePtr dRY = rig.alloc(init.data(), init.size() * 4);
    const PxgDevicePtr dRZ = rig.alloc(init.data(), init.size() * 4);
    if (!dRX) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    rig.desc.numPreviousHandles = numPrevious;
    rig.desc.boxProjectionRanks[0] = reinterpret_cast<PxU32*>(dRX);
    rig.desc.boxProjectionRanks[1] = reinterpret_cast<PxU32*>(dRY);
    rig.desc.boxProjectionRanks[2] = reinterpret_cast<PxU32*>(dRZ);
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grid: BP_INITIALIZE_RANKS = 64 blocks x 256 (grid-stride over N/4)
    PXG_CHECK_MSG(shim.launchKernel(0, 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch radix ranks");
    shim.streamSynchronize();

    std::vector<uint32_t> gX(padded), gY(padded), gZ(padded);
    shim.memcpyDtoH(gX.data(), dRX, padded * 4);
    shim.memcpyDtoH(gY.data(), dRY, padded * 4);
    shim.memcpyDtoH(gZ.data(), dRZ, padded * 4);
    size_t mism = 0;
    for (uint32_t i = 0; i < padded; ++i)
        if (gX[i] != i || gY[i] != i || gZ[i] != i) ++mism;
    PXG_CHECK_MSG(mism == 0, "%zu radix-rank mismatches", mism);

    for (PxgDevicePtr pfree : { dRX, dRY, dRZ, dDesc })
        shim.memFree(pfree);
}

PXG_TEST(broadphaseSetup_updateHandles)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(6400);
    const uint32_t numPrevious = 220;
    const uint32_t numProjections = numPrevious * 2;

    auto buildPerm = [&]() {
        std::vector<uint32_t> perm(numProjections);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        return perm;
    };
    std::vector<uint32_t> ranksX = buildPerm(), ranksY = buildPerm(), ranksZ = buildPerm();
    std::vector<uint32_t> tmpX(numProjections), tmpY(numProjections), tmpZ(numProjections);
    for (uint32_t i = 0; i < numProjections; ++i)
    {
        tmpX[i] = 0x1000 + i;
        tmpY[i] = 0x2000 + i;
        tmpZ[i] = 0x3000 + i;
    }
    std::vector<uint32_t> zero(numProjections, 0);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseSetup.spv", "updateHandles", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    BpDescRig rig(&shim);
    const PxgDevicePtr dRX = rig.alloc(ranksX.data(), ranksX.size() * 4);
    const PxgDevicePtr dRY = rig.alloc(ranksY.data(), ranksY.size() * 4);
    const PxgDevicePtr dRZ = rig.alloc(ranksZ.data(), ranksZ.size() * 4);
    const PxgDevicePtr dTX = rig.alloc(tmpX.data(), tmpX.size() * 4);
    const PxgDevicePtr dTY = rig.alloc(tmpY.data(), tmpY.size() * 4);
    const PxgDevicePtr dTZ = rig.alloc(tmpZ.data(), tmpZ.size() * 4);
    const PxgDevicePtr dCX = rig.alloc(zero.data(), zero.size() * 4);
    const PxgDevicePtr dCY = rig.alloc(zero.data(), zero.size() * 4);
    const PxgDevicePtr dCZ = rig.alloc(zero.data(), zero.size() * 4);
    if (!dRX) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    rig.desc.numPreviousHandles = numPrevious;
    rig.desc.boxProjectionRanks[0] = reinterpret_cast<PxU32*>(dRX);
    rig.desc.boxProjectionRanks[1] = reinterpret_cast<PxU32*>(dRY);
    rig.desc.boxProjectionRanks[2] = reinterpret_cast<PxU32*>(dRZ);
    rig.desc.boxHandles[0][0] = reinterpret_cast<PxU32*>(dCX);
    rig.desc.boxHandles[0][1] = reinterpret_cast<PxU32*>(dCY);
    rig.desc.boxHandles[0][2] = reinterpret_cast<PxU32*>(dCZ);
    rig.desc.boxHandles[1][0] = reinterpret_cast<PxU32*>(dTX);
    rig.desc.boxHandles[1][1] = reinterpret_cast<PxU32*>(dTY);
    rig.desc.boxHandles[1][2] = reinterpret_cast<PxU32*>(dTZ);
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grid: BP_UDPATE_HANDLES = 64 blocks x 256 (grid-stride)
    PXG_CHECK_MSG(shim.launchKernel(0, 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch updateHandles");
    shim.streamSynchronize();

    std::vector<uint32_t> gX(numProjections), gY(numProjections), gZ(numProjections);
    shim.memcpyDtoH(gX.data(), dCX, numProjections * 4);
    shim.memcpyDtoH(gY.data(), dCY, numProjections * 4);
    shim.memcpyDtoH(gZ.data(), dCZ, numProjections * 4);

    size_t mism = 0;
    for (uint32_t i = 0; i < numProjections; ++i)
    {
        if (gX[i] != tmpX[ranksX[i]]) ++mism;
        if (gY[i] != tmpY[ranksY[i]]) ++mism;
        if (gZ[i] != tmpZ[ranksZ[i]]) ++mism;
    }
    PXG_CHECK_MSG(mism == 0, "%zu updateHandles mismatches", mism);

    for (PxgDevicePtr pfree : { dRX, dRY, dRZ, dTX, dTY, dTZ, dCX, dCY, dCZ, dDesc })
        shim.memFree(pfree);
}

} // namespace
