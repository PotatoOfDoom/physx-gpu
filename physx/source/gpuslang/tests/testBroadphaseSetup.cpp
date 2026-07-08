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
// Test 1: translateAABBsLaunch
// ---------------------------------------------------------------------------
PXG_TEST(broadphaseSetup_translateAABBs)
{
    std::mt19937 rng(6100);
    std::uniform_real_distribution<float> pos(-500.f, 500.f);
    std::uniform_real_distribution<float> ext(0.01f, 50.f);
    std::uniform_real_distribution<float> cd(0.f, 2.f);

    for (uint32_t variant = 0; variant < 3; ++variant)
    {
        const uint32_t capacity = 300;
        const bool hasEnvIds = variant >= 1;
        const uint32_t shiftX = variant == 2 ? 1u : 0u;
        const uint32_t shiftY = variant == 2 ? 2u : 0u;
        const uint32_t shiftZ = 0u;
        const uint32_t envShiftX = hasEnvIds ? 3u : 0u;
        const uint32_t envShiftY = hasEnvIds ? 3u : 0u;
        const uint32_t envShiftZ = hasEnvIds ? 0u : 0u; // exercise the envShift==0 fallthrough

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

        PxgSlangComputeDevice* dev = testDevice();
        std::vector<IntegerAABB> outInit(capacity);
        std::memset(outInit.data(), 0xcc, outInit.size() * sizeof(IntegerAABB));

        PxgBufferHandle inBuf = createBufferFrom(dev, inBounds);
        PxgBufferHandle outBuf = createBufferFrom(dev, outInit);
        PxgBufferHandle cdBuf = createBufferFrom(dev, contactDist);
        PxgBufferHandle envBuf = createBufferFrom(dev, envIds);

        Kernel k("broadphaseSetup.spv", "translateAABBsLaunch", { 4, 24, 4, 4 }, sizeof(BpParams));
        BpParams p = {};
        p.boxesCapacity = capacity;
        p.shiftX = shiftX; p.shiftY = shiftY; p.shiftZ = shiftZ;
        p.envShiftX = envShiftX; p.envShiftY = envShiftY; p.envShiftZ = envShiftZ;
        p.hasEnvIds = hasEnvIds ? 1u : 0u;
        k.dispatch((capacity + 255) / 256, { inBuf, outBuf, cdBuf, envBuf }, p);
        dev->flush();

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

        const auto g = downloadBuffer<IntegerAABB>(dev, outBuf, capacity);
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

        for (PxgBufferHandle h : { inBuf, outBuf, cdBuf, envBuf })
            dev->destroyBuffer(h);
    }
}

// ---------------------------------------------------------------------------
// Test 2: markCreatedPairsLaunch
// ---------------------------------------------------------------------------
PXG_TEST(broadphaseSetup_markCreatedPairs)
{
    std::mt19937 rng(6200);
    const uint32_t numPrevious = 37; // odd on purpose so total may be odd
    const uint32_t numCreated = 90;
    const uint32_t numBounds = 256;

    // new integer bounds (arbitrary but distinct so gathers are checkable)
    std::vector<IntegerAABB> newBounds(numBounds);
    for (uint32_t h = 0; h < numBounds; ++h)
        for (uint32_t a = 0; a < 6; ++a)
            newBounds[h].mMinMax[a] = (h << 8) | (a << 4) | 0x3; // even/odd not required here

    // created handles: distinct box indices
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
    PxgBufferHandle chBuf = createBufferFrom(dev, createdHandles);
    PxgBufferHandle pxBuf = createBufferFrom(dev, projX);
    PxgBufferHandle pyBuf = createBufferFrom(dev, projY);
    PxgBufferHandle pzBuf = createBufferFrom(dev, projZ);
    PxgBufferHandle hxBuf = createBufferFrom(dev, handX);
    PxgBufferHandle hyBuf = createBufferFrom(dev, handY);
    PxgBufferHandle hzBuf = createBufferFrom(dev, handZ);
    PxgBufferHandle nbBuf = createBufferFrom(dev, newBounds);
    PxgBufferHandle obBuf = createBufferFrom(dev, oldBounds);
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);

    Kernel k("broadphaseSetup.spv", "markCreatedPairsLaunch", { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 24, 24 },
             sizeof(BpParams));
    BpParams p = {};
    p.numCreatedHandles = numCreated;
    p.numPreviousHandles = numPrevious;
    // bindings 4..12
    k.dispatch((numCreated + 255) / 256, { dummyBuf, dummyBuf, dummyBuf, dummyBuf, chBuf, pxBuf, pyBuf, pzBuf,
                                           hxBuf, hyBuf, hzBuf, nbBuf, obBuf }, p);
    dev->flush();

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

    const auto gPX = downloadBuffer<uint32_t>(dev, pxBuf, projLen);
    const auto gPY = downloadBuffer<uint32_t>(dev, pyBuf, projLen);
    const auto gPZ = downloadBuffer<uint32_t>(dev, pzBuf, projLen);
    const auto gHX = downloadBuffer<uint32_t>(dev, hxBuf, projLen);
    const auto gHY = downloadBuffer<uint32_t>(dev, hyBuf, projLen);
    const auto gHZ = downloadBuffer<uint32_t>(dev, hzBuf, projLen);
    const auto gOld = downloadBuffer<IntegerAABB>(dev, obBuf, numBounds);

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

    for (PxgBufferHandle h : { chBuf, pxBuf, pyBuf, pzBuf, hxBuf, hyBuf, hzBuf, nbBuf, obBuf, dummyBuf })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// Test 3: initializeSapBox1DLaunch
// ---------------------------------------------------------------------------
PXG_TEST(broadphaseSetup_initializeSapBox1D)
{
    std::mt19937 rng(6300);
    const uint32_t numHandles = 200;
    const uint32_t numProjections = numHandles * 2;

    // build per-axis sorted handle arrays: each box contributes a start and an
    // end handle; place them in a random order per axis
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

    std::vector<SapBox1D> sapX(numHandles), sapY(numHandles), sapZ(numHandles);
    std::memset(sapX.data(), 0xff, sapX.size() * sizeof(SapBox1D));
    std::memset(sapY.data(), 0xff, sapY.size() * sizeof(SapBox1D));
    std::memset(sapZ.data(), 0xff, sapZ.size() * sizeof(SapBox1D));

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle hxBuf = createBufferFrom(dev, handX);
    PxgBufferHandle hyBuf = createBufferFrom(dev, handY);
    PxgBufferHandle hzBuf = createBufferFrom(dev, handZ);
    PxgBufferHandle sxBuf = createBufferFrom(dev, sapX);
    PxgBufferHandle syBuf = createBufferFrom(dev, sapY);
    PxgBufferHandle szBuf = createBufferFrom(dev, sapZ);
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);

    Kernel k("broadphaseSetup.spv", "initializeSapBox1DLaunch", { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8 },
             sizeof(BpParams));
    BpParams p = {};
    p.numHandles = numHandles;
    // bindings 8,9,10 (handles) + 13,14,15 (sapbox)
    k.dispatch((numProjections + 255) / 256, { dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf,
                                               dummyBuf, dummyBuf, hxBuf, hyBuf, hzBuf, dummyBuf, dummyBuf,
                                               sxBuf, syBuf, szBuf }, p);
    dev->flush();

    // ---- reference ----
    auto refAxis = [&](const std::vector<uint32_t>& handles, std::vector<SapBox1D>& sap) {
        for (uint32_t i = 0; i < numProjections; ++i)
        {
            const uint32_t sh = handles[i];
            const uint32_t handle = sh >> 3;
            const bool isStart = (sh & 1) != 0;
            sap[handle].mMinMax[isStart ? 0 : 1] = i;
        }
    };
    std::vector<SapBox1D> rX(numHandles), rY(numHandles), rZ(numHandles);
    std::memset(rX.data(), 0xff, rX.size() * sizeof(SapBox1D));
    std::memset(rY.data(), 0xff, rY.size() * sizeof(SapBox1D));
    std::memset(rZ.data(), 0xff, rZ.size() * sizeof(SapBox1D));
    refAxis(handX, rX);
    refAxis(handY, rY);
    refAxis(handZ, rZ);

    const auto gX = downloadBuffer<SapBox1D>(dev, sxBuf, numHandles);
    const auto gY = downloadBuffer<SapBox1D>(dev, syBuf, numHandles);
    const auto gZ = downloadBuffer<SapBox1D>(dev, szBuf, numHandles);

    size_t mism = 0;
    for (uint32_t h = 0; h < numHandles; ++h)
    {
        if (gX[h].mMinMax[0] != rX[h].mMinMax[0] || gX[h].mMinMax[1] != rX[h].mMinMax[1]) ++mism;
        if (gY[h].mMinMax[0] != rY[h].mMinMax[0] || gY[h].mMinMax[1] != rY[h].mMinMax[1]) ++mism;
        if (gZ[h].mMinMax[0] != rZ[h].mMinMax[0] || gZ[h].mMinMax[1] != rZ[h].mMinMax[1]) ++mism;
    }
    PXG_CHECK_MSG(mism == 0, "%zu SAP box mismatches", mism);

    for (PxgBufferHandle h : { hxBuf, hyBuf, hzBuf, sxBuf, syBuf, szBuf, dummyBuf })
        dev->destroyBuffer(h);
}

} // namespace
