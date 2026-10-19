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


// P8.95b: the assembled `updateBodiesLaunch_Part2` kernel, driven over the REAL
// engine ABI with the articulation chain's first Z-dimensioned launch:
// grid (blocks, 1, ceil(maxLinks/4)), block (32, 2, 4).
//
// THE ORACLE IS BIT-EXACT: every operation is engine arithmetic with a
// PX_CUDA_CALLABLE original -- Dy::SpatialMatrix::operator*, PxTransform
// products, PxQuat::rotate/rotateInv, translateSpatialVectorInPlace (hoisted in
// P8.91c), and the P8.95a sleep mirror (hoisted for this fixture) -- and there
// is no transcendental anywhere.  The comparison is `==` on the raw words of
// every output array AND every block byte (this kernel writes no block field;
// the whole block region must come back at its seed).
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//   * 68 articulations -> grid (2, 1, 2): both X blocks and both Z blocks
//     live, so a Z-indexing error lands on a link the other Z block owns;
//   * link counts 2..6 with maxLinks 6: links 4 and 5 exist only for some
//     articulations, so the SECOND guard (linkCount) is distinguishable from
//     the FIRST (maxLinks);
//   * `integrate` BOTH ways, two dispatches, with posMotionVelocity seeds that
//     make the invDt scaling visible;
//   * articulationIndex != the articulation's fleet slot (a permutation), so
//     the record lookup through mArticulationIndex is distinguishable from
//     globalThreadIndex;
//   * every record array allocated at its packed engine stride -- 28-byte
//     PxTransforms, 24-byte spatial vectors, 80-byte links, 32-byte sleep
//     data -- with poison beyond every live element, and the link records'
//     .w lanes (invMass, penBiasClamp) seeded distinct and required BACK.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxQuat;
using physx::PxReal;
using physx::PxTransform;
using physx::PxU32;
using physx::PxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::Cm::SpatialVectorF;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;
using physx::loadSpatialVector;
using physx::storeSpatialVector;
using physx::loadSpatialVectorF;
using physx::loadSpatialTransform;
using physx::storeSpatialTransform;
using physx::storeSpatialMatrix;
using physx::loadSpatialMatrix;

static std::vector<uint8_t> aubLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t AUB_ARTS = 68;        // grid X = ceil(68/64) = 2
const uint32_t AUB_WARPS = 4;        // 2 blocks x 2 warps; warp 2 has 4 live lanes
const uint32_t AUB_MAX_LINKS = 6;    // grid Z = ceil(6/4) = 2
const uint32_t AUB_MAX_DOFS = 13;
const float AUB_DT = 0.021484375f;   // 11/512

static uint32_t aubNumLinks(uint32_t a) { return 2u + (a % 5u); }
// the record index is a PERMUTATION of the fleet slots, never the identity
static uint32_t aubArtiIndex(uint32_t a) { return (a * 29u + 7u) % AUB_ARTS; }

static float aubV(uint32_t a, uint32_t f)
{
    return (float((a * 41u + f * 17u) % 439u) + 1.0f) * 0.0078125f - 1.71875f;
}

static void aubFillBlocks(physx::PxgArticulationBlockData* ab,
                          physx::PxgArticulationBlockLinkData* lk, uint32_t a, uint32_t lane)
{
    const uint32_t nl = aubNumLinks(a);
    auto v = [&](uint32_t i, uint32_t f) { return aubV(a, i * 67u + f); };
    auto unitQ = [&](uint32_t i, uint32_t f) {
        PxQuat q(v(i, f), v(i, f + 1), v(i, f + 2), v(i, f + 3) + 2.25f);
        return q.getNormalized();
    };
    auto sv = [&](uint32_t i, uint32_t f) {
        return UnAlignedSpatialVector(PxVec3(v(i, f), v(i, f + 1), v(i, f + 2)),
                                      PxVec3(v(i, f + 3), v(i, f + 4), v(i, f + 5)));
    };

    ab->mNumLinks[lane] = nl;
    ab->mArticulationIndex[lane] = aubArtiIndex(a);
    ab->mSleepThreshold[lane] = 0.05f + float(a % 7) * 0.4f;

    for (uint32_t i = 0; i < nl; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        storeSpatialVector(L.mMotionVelocity, sv(i, 0), lane);
        storeSpatialVector(L.mPosMotionVelocity, sv(i, 6), lane);
        storeSpatialVector(L.mSolverSpatialDeltaVel, sv(i, 12), lane);
        storeSpatialVector(L.mSolverSpatialImpulse, sv(i, 18), lane);
        storeSpatialVector(L.mZAVector, sv(i, 24), lane);
        storeSpatialVector(L.mZAIntVector, sv(i, 30), lane);
        storeSpatialVector(L.mMotionAcceleration, sv(i, 36), lane);
        storeSpatialVector(L.mMotionAccelerationInternal, sv(i, 42), lane);
        storeSpatialTransform(L.mAccumulatedPose, lane,
                              PxTransform(PxVec3(v(i, 48), v(i, 49), v(i, 50)), unitQ(i, 51)));
        storeSpatialTransform(L.mChildPose, lane,
                              PxTransform(PxVec3(v(i, 55), v(i, 56), v(i, 57)), unitQ(i, 58)));
        physx::Dy::SpatialMatrix m;
        m.topLeft = physx::PxMat33(PxVec3(v(i, 70), v(i, 71), v(i, 72)),
                                   PxVec3(v(i, 73), v(i, 74), v(i, 75)),
                                   PxVec3(v(i, 76), v(i, 77), v(i, 78)));
        m.topRight = physx::PxMat33(PxVec3(v(i, 80), v(i, 81), v(i, 82)),
                                    PxVec3(v(i, 83), v(i, 84), v(i, 85)),
                                    PxVec3(v(i, 86), v(i, 87), v(i, 88)));
        m.bottomLeft = physx::PxMat33(PxVec3(v(i, 90), v(i, 91), v(i, 92)),
                                      PxVec3(v(i, 93), v(i, 94), v(i, 95)),
                                      PxVec3(v(i, 96), v(i, 97), v(i, 98)));
        storeSpatialMatrix(L.mSpatialArticulatedInertia, lane, m);
        { float4 f; f.x = std::fabs(v(i, 100)) + 0.05f; f.y = std::fabs(v(i, 101)) + 0.05f;
          f.z = ((a + i) % 9 == 3) ? 0.0f : std::fabs(v(i, 102)) + 0.05f;
          f.w = ((a + i) % 8 == 5) ? 0.0f : std::fabs(v(i, 103)) + 0.1f;
          L.mInvInertiaXYZ_invMassW[lane] = f; }
    }
}

// the record-side arrays for ONE articulation, in one contiguous poisoned blob
struct AubRecord
{
    std::vector<uint8_t> links;        // 80 * maxLinks
    std::vector<uint8_t> sleepData;    // 32 * maxLinks
    std::vector<float> wakeCounters;   // maxLinks
    std::vector<uint8_t> body2World;   // 28 * (maxLinks + 1)   (+1: overrun canary)
    std::vector<uint8_t> motionVel;    // 24 * (maxLinks + 1)
    std::vector<uint8_t> accel;        // 24 * (maxLinks + 1)
    std::vector<uint8_t> incoming;     // 24 * (maxLinks + 1)
    std::vector<uint8_t> zAForces;     // 24 * maxLinks: 5.12 external-Z chain (read only)
};

static void aubFillRecord(AubRecord& r, uint32_t recIdx)
{
    auto v = [&](uint32_t i, uint32_t f) { return aubV(recIdx, i * 53u + f + 7u); };
    r.links.assign(80 * AUB_MAX_LINKS, 0xAB);
    r.sleepData.assign(32 * AUB_MAX_LINKS, 0xAB);
    r.wakeCounters.assign(AUB_MAX_LINKS, 0.f);
    r.body2World.assign(28 * (AUB_MAX_LINKS + 1), 0xAB);
    r.motionVel.assign(24 * (AUB_MAX_LINKS + 1), 0xAB);
    r.accel.assign(24 * (AUB_MAX_LINKS + 1), 0xAB);
    r.incoming.assign(24 * (AUB_MAX_LINKS + 1), 0xAB);
    r.zAForces.assign(24 * AUB_MAX_LINKS, 0xAB);
    for (uint32_t i = 0; i < AUB_MAX_LINKS; ++i)
    {
        float* za = reinterpret_cast<float*>(r.zAForces.data() + 24 * i);
        for (uint32_t k = 0; k < 6; ++k)
            za[k] = v(i, 30 + k) * 0.5f;
        float* rec = reinterpret_cast<float*>(r.links.data() + 80 * i);
        // the two float4s: angular @0 (w = penBiasClamp), linear @16 (w = invMass)
        rec[0] = v(i, 0); rec[1] = v(i, 1); rec[2] = v(i, 2); rec[3] = v(i, 3) + 3.0f;
        rec[4] = v(i, 4); rec[5] = v(i, 5); rec[6] = v(i, 6); rec[7] = v(i, 7) + 5.0f;
        float* sd = reinterpret_cast<float*>(r.sleepData.data() + 32 * i);
        for (uint32_t k = 0; k < 8; ++k)
            sd[k] = v(i, 10 + k) * 0.25f;
        // wake counters across the three sleep regimes
        r.wakeCounters[i] = (i % 3 == 0) ? 0.5f : ((i % 3 == 1) ? 0.05f : AUB_DT * 0.4f);
    }
}

// ---------------------------------------------------------------------------
// the mirror
// ---------------------------------------------------------------------------
static void aubMirror(physx::PxgArticulationBlockData* ab,
                      physx::PxgArticulationBlockLinkData* lk, AubRecord& r, uint32_t lane,
                      uint32_t linkIndex, float dt, bool integrate, bool extEvery,
                      bool sleepingDisabled)
{
    const uint32_t linkCount = ab->mNumLinks[lane];
    if (linkIndex >= linkCount)
        return;
    const float invDt = 1.f / dt;
    const float sleepThreshold = ab->mSleepThreshold[lane];

    physx::PxgArticulationBlockLinkData& L = lk[linkIndex];

    if (linkIndex == 0)
    {
        const SpatialVectorF acc = loadSpatialVectorF(L.mMotionAcceleration, lane);
        const SpatialVectorF dv = loadSpatialVectorF(L.mSolverSpatialDeltaVel, lane);
        const SpatialVectorF accW = acc + dv * invDt;
        float* out = reinterpret_cast<float*>(r.accel.data());
        out[0] = accW.top.x; out[1] = accW.top.y; out[2] = accW.top.z;
        out[3] = accW.bottom.x; out[4] = accW.bottom.y; out[5] = accW.bottom.z;
        float* inc = reinterpret_cast<float*>(r.incoming.data());
        for (uint32_t k = 0; k < 6; ++k) inc[k] = 0.f;
    }
    else
    {
        SpatialVectorF zaExt = loadSpatialVectorF(L.mZAVector, lane);
        if (extEvery)
        {
            // 5.12: the external-Z chain updateBodiesLaunch1T accumulated into the
            // record's zAForces is added back for the readback
            const float* za = reinterpret_cast<const float*>(r.zAForces.data() + 24 * linkIndex);
            zaExt.top += PxVec3(za[0], za[1], za[2]);
            zaExt.bottom += PxVec3(za[3], za[4], za[5]);
        }
        const SpatialVectorF zaInt = loadSpatialVectorF(L.mZAIntVector, lane);
        const SpatialVectorF accExt = loadSpatialVectorF(L.mMotionAcceleration, lane);
        const SpatialVectorF accInt = loadSpatialVectorF(L.mMotionAccelerationInternal, lane);
        physx::Dy::SpatialMatrix inertia;
        loadSpatialMatrix(L.mSpatialArticulatedInertia, lane, inertia);
        const SpatialVectorF dv = loadSpatialVectorF(L.mSolverSpatialDeltaVel, lane);
        const SpatialVectorF imp = loadSpatialVectorF(L.mSolverSpatialImpulse, lane);
        const PxTransform Gc = loadSpatialTransform(L.mAccumulatedPose, lane);
        const PxTransform Lc = loadSpatialTransform(L.mChildPose, lane);
        const PxTransform GcLc = Gc * Lc;
        const PxVec3 dW = Gc.rotate(Lc.p);

        const SpatialVectorF accW = accExt + accInt + dv * invDt;
        SpatialVectorF incoming = inertia * accW + (zaExt + zaInt + imp * invDt);
        // translateSpatialVectorInPlace is a TEMPLATE in the engine; its body,
        // instantiated at SpatialVectorF, is exactly this line (the header's
        // ref helper is pinned to the UnAligned type)
        incoming.bottom += (-dW).cross(incoming.top);

        float* out = reinterpret_cast<float*>(r.accel.data() + 24 * linkIndex);
        out[0] = accW.top.x; out[1] = accW.top.y; out[2] = accW.top.z;
        out[3] = accW.bottom.x; out[4] = accW.bottom.y; out[5] = accW.bottom.z;
        const PxVec3 it = GcLc.rotateInv(incoming.top);
        const PxVec3 ib = GcLc.rotateInv(incoming.bottom);
        float* inc = reinterpret_cast<float*>(r.incoming.data() + 24 * linkIndex);
        inc[0] = it.x; inc[1] = it.y; inc[2] = it.z;
        inc[3] = ib.x; inc[4] = ib.y; inc[5] = ib.z;
    }

    const UnAlignedSpatialVector motionV = loadSpatialVector(L.mMotionVelocity, lane);
    float* rec = reinterpret_cast<float*>(r.links.data() + 80 * linkIndex);
    const float invMassW = rec[7];
    const float penBiasW = rec[3];
    float lwc = r.wakeCounters[linkIndex];
    const PxTransform pose = loadSpatialTransform(L.mAccumulatedPose, lane);
    UnAlignedSpatialVector posMotionV = loadSpatialVector(L.mPosMotionVelocity, lane);
    const float4 invInertia = L.mInvInertiaXYZ_invMassW[lane];

    float* mv = reinterpret_cast<float*>(r.motionVel.data() + 24 * linkIndex);
    mv[0] = motionV.top.x; mv[1] = motionV.top.y; mv[2] = motionV.top.z;
    mv[3] = motionV.bottom.x; mv[4] = motionV.bottom.y; mv[5] = motionV.bottom.z;
    rec[4] = motionV.bottom.x; rec[5] = motionV.bottom.y; rec[6] = motionV.bottom.z;
    rec[7] = invMassW;
    rec[0] = motionV.top.x; rec[1] = motionV.top.y; rec[2] = motionV.top.z;
    rec[3] = penBiasW;

    float* b2w = reinterpret_cast<float*>(r.body2World.data() + 28 * linkIndex);
    b2w[0] = pose.q.x; b2w[1] = pose.q.y; b2w[2] = pose.q.z; b2w[3] = pose.q.w;
    b2w[4] = pose.p.x; b2w[5] = pose.p.y; b2w[6] = pose.p.z;

    if (!integrate)
        posMotionV *= invDt;

    // 5.12: no sleep check (and no wake-counter write) when sleeping is disabled
    if (!sleepingDisabled)
    {
        const float inv4[4] = { invInertia.x, invInertia.y, invInertia.z, invInertia.w };
        pxgref::aslMirror(reinterpret_cast<float*>(r.sleepData.data() + 32 * linkIndex), lwc, pose.q,
                          posMotionV.bottom, posMotionV.top, dt, sleepThreshold, inv4);
        r.wakeCounters[linkIndex] = lwc;
    }
}

static void aubRun(PxgCudaShim& shim, bool integrate, bool extEvery, bool sleepingDisabled,
                   size_t& bad, size_t& signedZeros)
{
    std::vector<physx::PxgArticulationBlockData> abImg(AUB_WARPS);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(AUB_WARPS) * AUB_MAX_LINKS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));

    std::vector<AubRecord> recs(AUB_ARTS);
    std::vector<physx::PxgArticulation> arts(AUB_ARTS);
    std::memset(arts.data(), 0xAB, arts.size() * sizeof(physx::PxgArticulation));

    for (uint32_t a = 0; a < AUB_ARTS; ++a)
    {
        aubFillBlocks(&abImg[a / 32u], &lkImg[size_t(a / 32u) * AUB_MAX_LINKS], a, a & 31u);
        aubFillRecord(recs[aubArtiIndex(a)], aubArtiIndex(a));
    }

    // device: allocate every record array at its packed stride
    PxgDevicePtr dAb = 0, dLk = 0, dArts = 0, dDesc = 0;
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dArts, arts.size() * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    std::vector<PxgDevicePtr> devBufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp) shim.memcpyHtoD(dp, p, n);
        devBufs.push_back(dp);
        return dp;
    };
    if (!dAb || !dLk || !dArts || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    struct RecPtrs { PxgDevicePtr links, sleep, wake, b2w, mv, acc, inc, za; };
    std::vector<RecPtrs> rp(AUB_ARTS);
    for (uint32_t k = 0; k < AUB_ARTS; ++k)
    {
        AubRecord& r = recs[k];
        rp[k].links = up(r.links.data(), r.links.size());
        rp[k].sleep = up(r.sleepData.data(), r.sleepData.size());
        rp[k].wake = up(r.wakeCounters.data(), r.wakeCounters.size() * 4);
        rp[k].b2w = up(r.body2World.data(), r.body2World.size());
        rp[k].mv = up(r.motionVel.data(), r.motionVel.size());
        rp[k].acc = up(r.accel.data(), r.accel.size());
        rp[k].inc = up(r.incoming.data(), r.incoming.size());
        rp[k].za = up(r.zAForces.data(), r.zAForces.size());
        physx::PxgArticulation& A = arts[k];
        A.links = reinterpret_cast<physx::PxgArticulationLink*>(rp[k].links);
        A.linkSleepData = reinterpret_cast<physx::PxgArticulationLinkSleepData*>(rp[k].sleep);
        A.linkWakeCounters = reinterpret_cast<PxReal*>(rp[k].wake);
        A.linkBody2Worlds = reinterpret_cast<PxTransform*>(rp[k].b2w);
        A.motionVelocities = reinterpret_cast<UnAlignedSpatialVector*>(rp[k].mv);
        A.motionAccelerations = reinterpret_cast<UnAlignedSpatialVector*>(rp[k].acc);
        A.linkIncomingJointForces = reinterpret_cast<UnAlignedSpatialVector*>(rp[k].inc);
        A.zAForces = reinterpret_cast<UnAlignedSpatialVector*>(rp[k].za);
    }
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dArts, arts.data(), arts.size() * sizeof(physx::PxgArticulation));

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.nbArticulations = AUB_ARTS;
    desc.mMaxLinksPerArticulation = AUB_MAX_LINKS;
    desc.mMaxDofsPerArticulation = AUB_MAX_DOFS;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks = reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.articulations = reinterpret_cast<physx::PxgArticulation*>(dArts);
    // 5.12: the two flag bytes this kernel reads (the memset poison would read true)
    desc.isExternalForcesEveryTgsIterationEnabled = extEvery;
    desc.isSleepingDisabled = sleepingDisabled;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // the expectation
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<AubRecord> wantRecs = recs;
    for (uint32_t a = 0; a < AUB_ARTS; ++a)
        for (uint32_t li = 0; li < AUB_MAX_LINKS; ++li)
            aubMirror(&wantAb[a / 32u], &wantLk[size_t(a / 32u) * AUB_MAX_LINKS],
                      wantRecs[aubArtiIndex(a)], a & 31u, li, AUB_DT, integrate, extEvery,
                      sleepingDisabled);

    // The CUDA arg pack carries the bool in ONE byte at offset 12; the three
    // bytes above it are padding with UNSPECIFIED content, so the shader must
    // mask.  The fixture poisons them -- a whole-word read gives the WRONG
    // truth value for the false case and the wrong bits for the true case.
    struct Params { uint64_t desc; float dt; uint32_t integrateWord; } prm = {
        uint64_t(dDesc), AUB_DT, integrate ? 0xA7A5A301u : 0xB6B4B200u };
    // grid (ceil(68/64)=2, 1, ceil(6/4)=2), block (32, 2, 4) from the shader
    PXG_CHECK_MSG(shim.launchKernel(0, 2, 1, 2, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch updateBodiesLaunch_Part2");
    shim.streamSynchronize();

    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == g[k]) continue;
            if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
            { ++signedZeros; continue; }
            if (bad < 10)
                std::printf("MM aub int=%d %s word %zu  ref %08x  gpu %08x\n", int(integrate),
                            what, k, w[k], g[k]);
            ++bad;
        }
    };

    // the blocks must be UNTOUCHED (trap 1)
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    cmp("arti(block)", wantAb.data(), gotAb.data(),
        wantAb.size() * sizeof(physx::PxgArticulationBlockData));
    cmp("link(block)", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData));

    std::vector<uint8_t> tmp;
    for (uint32_t k = 0; k < AUB_ARTS; ++k)
    {
        AubRecord& wr = wantRecs[k];
        auto pull = [&](PxgDevicePtr dp, const std::vector<uint8_t>& wantBuf, const char* what) {
            tmp.resize(wantBuf.size());
            shim.memcpyDtoH(tmp.data(), dp, tmp.size());
            cmp(what, wantBuf.data(), tmp.data(), wantBuf.size());
        };
        pull(rp[k].links, wr.links, "links");
        pull(rp[k].sleep, wr.sleepData, "sleep");
        { tmp.resize(wr.wakeCounters.size() * 4);
          shim.memcpyDtoH(tmp.data(), rp[k].wake, tmp.size());
          cmp("wake", wr.wakeCounters.data(), tmp.data(), tmp.size()); }
        pull(rp[k].b2w, wr.body2World, "body2World");
        pull(rp[k].mv, wr.motionVel, "motionVel");
        pull(rp[k].acc, wr.accel, "accel");
        pull(rp[k].inc, wr.incoming, "incoming");
        pull(rp[k].za, wr.zAForces, "zAForces");
    }

    for (PxgDevicePtr dp : devBufs)
        if (dp) shim.memFree(dp);
    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dArts);
    shim.memFree(dDesc);
}

PXG_TEST(artiUpdateBodies2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiUpdateBodies.spv", "updateBodiesLaunch_Part2", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &aubLoader);

    uint32_t nlSeen[5] = { 0, 0, 0, 0, 0 };
    uint32_t identity = 0;
    for (uint32_t a = 0; a < AUB_ARTS; ++a)
    {
        nlSeen[aubNumLinks(a) - 2u] += 1u;
        if (aubArtiIndex(a) == a) ++identity;
    }
    std::printf("[ info ] aub: link counts 2..6 on %u/%u/%u/%u/%u lanes, %u identity remaps\n",
                nlSeen[0], nlSeen[1], nlSeen[2], nlSeen[3], nlSeen[4], identity);
    for (uint32_t c = 0; c < 5; ++c)
        PXG_CHECK(nlSeen[c] > 8);
    PXG_CHECK_MSG(identity <= 2, "the articulation remap is nearly the identity (%u fixed points)",
                  identity);

    size_t bad = 0, signedZeros = 0;
    aubRun(shim, true, false, false, bad, signedZeros);
    aubRun(shim, false, false, false, bad, signedZeros);
    // 5.12: the external-Z chain readback and the sleeping-disabled skip
    aubRun(shim, true, true, false, bad, signedZeros);
    aubRun(shim, false, false, true, bad, signedZeros);
    std::printf("[ info ] aub: signed zeros %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "updateBodiesLaunch_Part2 is not bit-exact against the engine");
}

}  // namespace
