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

// P5 Step 4: empirical layout proof for the simulation-controller descriptor
// mirrors (pxgSlangSimCoreDesc.slang).  Fills the REAL engine
// PxgSimulationCoreDesc / PxgNewBodiesDesc (the exact structs the engine
// DMAs to the device) with distinct sentinels, uploads the raw bytes, and
// the simCoreDescProbe kernel reads every mirror field back through the
// Slang mirrors.  Element [1] of REAL PxgShapeSim / PxgNewShapeSim /
// PxgBodySim / PxsCachedTransform / PxBounds3 / PxgSolverBodySleepData /
// PxgShape arrays is probed through live desc pointers to pin the array
// STRIDES on top of the field offsets, and the raw PxgArticulation
// stride/offset constants are checked against host-side sizeof/offsetof
// plus a functional pointer walk (pool[1].linkBody2Worlds[2]).  The
// updateTransformCacheAndBoundArrayLaunch / updateBodiesLaunch /
// updateShapesLaunch conversions rest on this proof.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "foundation/PxBounds3.h"
#include "foundation/PxTransform.h"
#include "PxgSimulationCoreDesc.h"
#include "PxgShapeSim.h"
#include "PxgBodySim.h"
#include "PxgSolverBody.h"
#include "PxsTransformCache.h"
#include "PxgConvexConvexShape.h"
#include "PxgArticulation.h"

#include <cstddef>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> simCoreDescLoader(const char* file) { return loadShaderBlob(file); }

// Distinct 64-bit sentinel per probed pointer field.
static uint64_t sent(uint32_t idx) { return 0xC000000000000400ull + uint64_t(idx) * 0x700013ull; }
// Distinct 32-bit sentinel per probed uint field.
static uint32_t sent32(uint32_t idx) { return 0x50000047u + idx * 0x3000Du; }
// Distinct float sentinel (compared by bit pattern).
static float sentF(uint32_t idx) { return 2.25f + float(idx) * 0.015625f; }
static uint64_t bitsOf(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

PXG_TEST(simCoreDescLayout)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npSimCoreDescProbe.spv", "simCoreDescProbe", 40u, 0 }, // { u64 x5 }
    };
    PxgCudaShim shim(dev, manifest, 1, &simCoreDescLoader);

    uint32_t p = 0, u = 0, f = 0;
    std::vector<uint64_t> expect;

    // ---- A) the REAL engine PxgSimulationCoreDesc, sentinel-filled (0xEE
    // background catches any field the fill misses) ----
    PxgSimulationCoreDesc d;
    std::memset(&d, 0xEE, sizeof(d));
    d.mChangedAABBMgrHandles = reinterpret_cast<PxU32*>(sent(p++));
    d.mFrozen = reinterpret_cast<PxU32*>(sent(p++));
    d.mUnfrozen = reinterpret_cast<PxU32*>(sent(p++));
    d.mFrozenBlockAndRes = reinterpret_cast<PxU32*>(sent(p++));
    d.mUnfrozenBlockAndRes = reinterpret_cast<PxU32*>(sent(p++));
    d.mUpdated = reinterpret_cast<PxU32*>(sent(p++));
    d.mActivate = reinterpret_cast<PxU32*>(sent(p++));
    d.mDeactivate = reinterpret_cast<PxU32*>(sent(p++));
    d.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(sent(p++));
    d.mShapeSimsBufferDeviceData = reinterpret_cast<const PxgShapeSim*>(sent(p++));
    d.mArticulationPool = reinterpret_cast<PxgArticulation*>(sent(p++));
    d.mArticulationSleepDataPool = reinterpret_cast<PxgSolverBodySleepData*>(sent(p++));
    d.mTransformCache = reinterpret_cast<PxsCachedTransform*>(sent(p++));
    d.mBounds = reinterpret_cast<PxBounds3*>(sent(p++));
    d.mBodyDataIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mSleepData = reinterpret_cast<PxgSolverBodySleepData*>(sent(p++));
    d.mShapes = reinterpret_cast<PxgShape*>(sent(p++));
    d.mNbTotalShapes = sent32(u++);
    d.mBitMapWordCounts = sent32(u++);
    d.mTotalFrozenShapes = sent32(u++);
    d.mTotalUnfrozenShapes = sent32(u++);

    expect.push_back(uint64_t(d.mChangedAABBMgrHandles));
    expect.push_back(uint64_t(d.mFrozen));
    expect.push_back(uint64_t(d.mUnfrozen));
    expect.push_back(uint64_t(d.mFrozenBlockAndRes));
    expect.push_back(uint64_t(d.mUnfrozenBlockAndRes));
    expect.push_back(uint64_t(d.mUpdated));
    expect.push_back(uint64_t(d.mActivate));
    expect.push_back(uint64_t(d.mDeactivate));
    expect.push_back(uint64_t(d.mBodySimBufferDeviceData));
    expect.push_back(uint64_t(d.mShapeSimsBufferDeviceData));
    expect.push_back(uint64_t(d.mArticulationPool));
    expect.push_back(uint64_t(d.mArticulationSleepDataPool));
    expect.push_back(uint64_t(d.mTransformCache));
    expect.push_back(uint64_t(d.mBounds));
    expect.push_back(uint64_t(d.mBodyDataIndices));
    expect.push_back(uint64_t(d.mSleepData));
    expect.push_back(uint64_t(d.mShapes));
    expect.push_back(d.mNbTotalShapes);
    expect.push_back(d.mBitMapWordCounts);
    expect.push_back(d.mTotalFrozenShapes);
    expect.push_back(d.mTotalUnfrozenShapes);

    // ---- B) the REAL engine PxgNewBodiesDesc ----
    PxgNewBodiesDesc nb;
    std::memset(&nb, 0xEE, sizeof(nb));
    nb.mNewBodySim = reinterpret_cast<const PxgBodySim*>(sent(p++));
    nb.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(sent(p++));
    nb.mPrevVelocitiesBuffer = reinterpret_cast<PxgBodySimVelocities*>(sent(p++));
    nb.mNbNewBodies = sent32(u++);
    // 5.12 inserted mPrevVelocitiesBuffer before mNbNewBodies (24 -> 32 bytes)
    static_assert(sizeof(PxgNewBodiesDesc) == 32, "5.12 PxgNewBodiesDesc size");
    static_assert(offsetof(PxgNewBodiesDesc, mNbNewBodies) == 24, "mNbNewBodies @24");

    expect.push_back(uint64_t(nb.mNewBodySim));
    expect.push_back(uint64_t(nb.mBodySimBufferDeviceData));
    expect.push_back(uint64_t(nb.mPrevVelocitiesBuffer));
    expect.push_back(nb.mNbNewBodies);

    // ---- C) REAL element arrays (element [1] is probed -> stride proof) ----

    PxgShapeSim ss[2];
    std::memset(ss, 0xEE, sizeof(ss));
    ss[1].mTransform.q = PxQuat(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
    ss[1].mTransform.p = PxVec3(sentF(f + 4), sentF(f + 5), sentF(f + 6));
    f += 7;
    ss[1].mLocalBounds.minimum = PxVec3(sentF(f), sentF(f + 1), sentF(f + 2));
    ss[1].mLocalBounds.maximum = PxVec3(sentF(f + 3), sentF(f + 4), sentF(f + 5));
    f += 6;
    ss[1].mBodySimIndex = PxNodeIndex(PxU64(0xA1B2C3D45E6F7089ull));
    ss[1].mHullDataIndex = sent32(u++);
    ss[1].mShapeFlags = 0x1357;
    ss[1].mShapeType = 0x9BDF;

    expect.push_back(bitsOf(ss[1].mTransform.q.x));
    expect.push_back(bitsOf(ss[1].mTransform.p.z));
    expect.push_back(bitsOf(ss[1].mLocalBounds.minimum.x));
    expect.push_back(bitsOf(ss[1].mLocalBounds.maximum.z));
    expect.push_back(uint64_t(uint32_t(ss[1].mBodySimIndex.getInd() & 0xFFFFFFFFu)));
    expect.push_back(uint64_t(uint32_t(ss[1].mBodySimIndex.getInd() >> 32)));
    expect.push_back(ss[1].mHullDataIndex);
    expect.push_back(uint64_t(ss[1].mShapeFlags) | (uint64_t(ss[1].mShapeType) << 16));

    PxgNewShapeSim ns[2];
    std::memset(ns, 0xEE, sizeof(ns));
    ns[1].mTransform.q = PxQuat(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
    ns[1].mTransform.p = PxVec3(sentF(f + 4), sentF(f + 5), sentF(f + 6));
    f += 7;
    ns[1].mLocalBounds.minimum = PxVec3(sentF(f), sentF(f + 1), sentF(f + 2));
    ns[1].mLocalBounds.maximum = PxVec3(sentF(f + 3), sentF(f + 4), sentF(f + 5));
    f += 6;
    ns[1].mBodySimIndex = PxNodeIndex(PxU64(0x1122334455667788ull));
    ns[1].mHullDataIndex = sent32(u++);
    ns[1].mShapeFlags = 0x2468;
    ns[1].mShapeType = 0xACE0;
    ns[1].mElementIndex = sent32(u++);

    expect.push_back(bitsOf(ns[1].mTransform.q.y));
    expect.push_back(bitsOf(ns[1].mLocalBounds.maximum.y));
    expect.push_back(uint64_t(uint32_t(ns[1].mBodySimIndex.getInd() & 0xFFFFFFFFu)));
    expect.push_back(ns[1].mHullDataIndex);
    expect.push_back(uint64_t(ns[1].mShapeFlags) | (uint64_t(ns[1].mShapeType) << 16));
    expect.push_back(ns[1].mElementIndex);

    PxgBodySim bs[2];
    std::memset(bs, 0xEE, sizeof(bs));
    bs[1].linearVelocityXYZ_inverseMassW = make_float4(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
    f += 4;
    bs[1].freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex =
        make_float4(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
    f += 4;
    bs[1].body2World.q.q = make_float4(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
    bs[1].body2World.p = make_float4(sentF(f + 4), sentF(f + 5), sentF(f + 6), sentF(f + 7));
    f += 8;
    bs[1].body2Actor_maxImpulseW.q.q = make_float4(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
    bs[1].body2Actor_maxImpulseW.p = make_float4(sentF(f + 4), sentF(f + 5), sentF(f + 6), sentF(f + 7));
    f += 8;
    bs[1].articulationRemapId = sent32(u++);
    bs[1].internalFlags = sent32(u++);

    expect.push_back(bitsOf(bs[1].linearVelocityXYZ_inverseMassW.x));
    expect.push_back(bitsOf(bs[1].freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.w));
    expect.push_back(bitsOf(bs[1].body2World.q.q.x));
    expect.push_back(bitsOf(bs[1].body2World.p.z));
    expect.push_back(bitsOf(bs[1].body2Actor_maxImpulseW.q.q.y));
    expect.push_back(bitsOf(bs[1].body2Actor_maxImpulseW.p.x));
    expect.push_back(bs[1].articulationRemapId);
    expect.push_back(bs[1].internalFlags);

    PxgSolverBodySleepData sd[2];
    std::memset(sd, 0xEE, sizeof(sd));
    sd[1].wakeCounter = sentF(f++);
    sd[1].internalFlags = sent32(u++);
    PxgSolverBodySleepData asd[2];
    std::memset(asd, 0xEE, sizeof(asd));
    asd[1].wakeCounter = sentF(f++);
    asd[1].internalFlags = sent32(u++);

    expect.push_back(bitsOf(sd[1].wakeCounter));
    expect.push_back(sd[1].internalFlags);
    expect.push_back(asd[1].internalFlags);

    PxsCachedTransform ct[2];
    std::memset(ct, 0xEE, sizeof(ct));
    ct[1].transform.q = PxQuat(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
    ct[1].transform.p = PxVec3(sentF(f + 4), sentF(f + 5), sentF(f + 6));
    f += 7;
    ct[1].flags = sent32(u++);

    expect.push_back(bitsOf(ct[1].transform.q.x));
    expect.push_back(bitsOf(ct[1].transform.p.z));
    expect.push_back(ct[1].flags);

    PxBounds3 bd[2];
    std::memset(bd, 0xEE, sizeof(bd));
    bd[1].minimum = PxVec3(sentF(f), sentF(f + 1), sentF(f + 2));
    bd[1].maximum = PxVec3(sentF(f + 3), sentF(f + 4), sentF(f + 5));
    f += 6;

    expect.push_back(bitsOf(bd[1].minimum.x));
    expect.push_back(bitsOf(bd[1].maximum.z));

    PxgShape sh[2];
    std::memset(sh, 0xEE, sizeof(sh));
    sh[1].scale.scale = PxVec3(sentF(f), sentF(f + 1), sentF(f + 2));
    sh[1].scale.rotation = PxQuat(sentF(f + 3), sentF(f + 4), sentF(f + 5), sentF(f + 6));
    f += 7;
    sh[1].materialIndex = sent32(u++);
    sh[1].hullOrMeshPtr = size_t(sent(p++));
    sh[1].type = sent32(u++);
    sh[1].particleOrSoftbodyId = sent32(u++);

    expect.push_back(bitsOf(sh[1].scale.scale.x));
    expect.push_back(bitsOf(sh[1].scale.rotation.w));
    expect.push_back(sh[1].materialIndex);
    expect.push_back(uint64_t(sh[1].hullOrMeshPtr));
    expect.push_back(sh[1].type);
    expect.push_back(sh[1].particleOrSoftbodyId);

    PxU32 bdi[2] = { 0xEEEEEEEEu, sent32(u++) };
    expect.push_back(bdi[1]);

    // ---- D) raw stride/offset constants (host sizeof/offsetof ground truth) ----
    expect.push_back(sizeof(PxgBodySim) / 16);
    expect.push_back(offsetof(PxgBodySim, freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex) / 16);
    expect.push_back(sizeof(PxgArticulation));
    expect.push_back(offsetof(PxgArticulation, linkBody2Worlds));
    expect.push_back(offsetof(PxgArticulation, linkBody2Actors));

    // articulation walk: pool[1].linkBody2Worlds[2] / .linkBody2Actors[2]
    float linkB2W[3 * 7];
    float linkB2A[3 * 7];
    for (int k = 0; k < 21; ++k)
    {
        linkB2W[k] = sentF(f + uint32_t(k));
        linkB2A[k] = sentF(f + 32 + uint32_t(k));
    }
    f += 64;
    // PxTransform[2] fields in probe order: qx, qw, px, pz / qy, py
    expect.push_back(bitsOf(linkB2W[14 + 0]));
    expect.push_back(bitsOf(linkB2W[14 + 3]));
    expect.push_back(bitsOf(linkB2W[14 + 4]));
    expect.push_back(bitsOf(linkB2W[14 + 6]));
    expect.push_back(bitsOf(linkB2A[14 + 1]));
    expect.push_back(bitsOf(linkB2A[14 + 5]));

    // ---- upload ----
    PxgDevicePtr dSs = 0, dNs = 0, dBs = 0, dSd = 0, dAsd = 0, dCt = 0, dBd = 0, dSh = 0, dBdi = 0;
    PxgDevicePtr dB2W = 0, dB2A = 0, dArts = 0;
    shim.memAlloc(&dSs, sizeof(ss));
    shim.memAlloc(&dNs, sizeof(ns));
    shim.memAlloc(&dBs, sizeof(bs));
    shim.memAlloc(&dSd, sizeof(sd));
    shim.memAlloc(&dAsd, sizeof(asd));
    shim.memAlloc(&dCt, sizeof(ct));
    shim.memAlloc(&dBd, sizeof(bd));
    shim.memAlloc(&dSh, sizeof(sh));
    shim.memAlloc(&dBdi, sizeof(bdi));
    shim.memAlloc(&dB2W, sizeof(linkB2W));
    shim.memAlloc(&dB2A, sizeof(linkB2A));
    shim.memAlloc(&dArts, 2 * sizeof(PxgArticulation));
    if (!dSs || !dNs || !dBs || !dSd || !dAsd || !dCt || !dBd || !dSh || !dBdi || !dB2W || !dB2A || !dArts)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dSs, ss, sizeof(ss));
    shim.memcpyHtoD(dNs, ns, sizeof(ns));
    shim.memcpyHtoD(dBs, bs, sizeof(bs));
    shim.memcpyHtoD(dSd, sd, sizeof(sd));
    shim.memcpyHtoD(dAsd, asd, sizeof(asd));
    shim.memcpyHtoD(dCt, ct, sizeof(ct));
    shim.memcpyHtoD(dBd, bd, sizeof(bd));
    shim.memcpyHtoD(dSh, sh, sizeof(sh));
    shim.memcpyHtoD(dBdi, bdi, sizeof(bdi));
    shim.memcpyHtoD(dB2W, linkB2W, sizeof(linkB2W));
    shim.memcpyHtoD(dB2A, linkB2A, sizeof(linkB2A));

    // REAL PxgArticulation[2] raw pool: [0] is a 0xEE decoy, [1] carries the
    // live link-transform array pointers at the engine's field offsets
    PxgArticulation arts[2];
    std::memset(arts, 0xEE, sizeof(arts));
    arts[1].linkBody2Worlds = reinterpret_cast<PxTransform*>(dB2W);
    arts[1].linkBody2Actors = reinterpret_cast<PxTransform*>(dB2A);
    shim.memcpyHtoD(dArts, arts, sizeof(arts));

    // live desc: pointer fields carry the real device addresses
    PxgSimulationCoreDesc dl;
    std::memset(&dl, 0, sizeof(dl));
    dl.mShapeSimsBufferDeviceData = reinterpret_cast<const PxgShapeSim*>(dSs);
    dl.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dBs);
    dl.mSleepData = reinterpret_cast<PxgSolverBodySleepData*>(dSd);
    dl.mArticulationSleepDataPool = reinterpret_cast<PxgSolverBodySleepData*>(dAsd);
    dl.mTransformCache = reinterpret_cast<PxsCachedTransform*>(dCt);
    dl.mBounds = reinterpret_cast<PxBounds3*>(dBd);
    dl.mShapes = reinterpret_cast<PxgShape*>(dSh);
    dl.mBodyDataIndices = reinterpret_cast<PxU32*>(dBdi);
    dl.mArticulationPool = reinterpret_cast<PxgArticulation*>(dArts);

    PxgDevicePtr dDesc = 0, dDescLive = 0, dNbDesc = 0, dOut = 0;
    shim.memAlloc(&dDesc, sizeof(d));
    shim.memAlloc(&dDescLive, sizeof(dl));
    shim.memAlloc(&dNbDesc, sizeof(nb));
    shim.memAlloc(&dOut, expect.size() * 8);
    if (!dDesc || !dDescLive || !dNbDesc || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dDesc, &d, sizeof(d));
    shim.memcpyHtoD(dDescLive, &dl, sizeof(dl));
    shim.memcpyHtoD(dNbDesc, &nb, sizeof(nb));
    shim.memsetD32(dOut, 0, expect.size() * 2);

    struct ProbeParams { uint64_t descSent, descLive, nbDesc, newShapeSims, out; } prm = {
        dDesc, dDescLive, dNbDesc, dNs, dOut
    };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch probe");
    shim.streamSynchronize();

    std::vector<uint64_t> got(expect.size(), 0);
    shim.memcpyDtoH(got.data(), dOut, got.size() * 8);

    size_t bad = 0;
    for (size_t i = 0; i < expect.size(); ++i)
        if (got[i] != expect[i])
        {
            if (bad < 10)
                PXG_CHECK_MSG(false, "probe[%zu]: gpu=%016llx expected=%016llx", i,
                              (unsigned long long)got[i], (unsigned long long)expect[i]);
            ++bad;
        }
    PXG_CHECK_MSG(bad == 0, "sim-core desc mirrors are byte-exact (%zu bad of %zu)", bad, expect.size());

    for (PxgDevicePtr h : { dSs, dNs, dBs, dSd, dAsd, dCt, dBd, dSh, dBdi, dB2W, dB2A, dArts,
                            dDesc, dDescLive, dNbDesc, dOut })
        shim.memFree(h);
}

} // namespace
