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

// P7.0 FEMCloth bring-up test: the three cloth orchestration kernels
// (femClothSimBda.spv) on the engine ABI (BDA), driven by the REAL engine
// descriptor.  A genuine PxgFEMCloth (gpusimulationcontroller header) is filled
// with device buffer addresses + scalars, DMA'd to the device, and read back
// through the Slang pxgSlangFEMClothDesc mirror — the first-ever PxgFEMCloth BDA
// round-trip.  static_asserts / offsetof checks pin the mirror's byte offsets
// against the real engine struct (the highest-risk item, sizeof == 400), and each
// kernel's output is compared BIT-EXACT against a CPU transcription of the
// FEMCloth.cu math.
//
// Suites (substring-matched by the "femClothSim" ctest filter):
//   femClothSimDescLayout        — byte-offset proof of the descriptor mirror
//   femClothSimPreIntegratePGS   — cloth_preIntegrateLaunch PGS predict+integrate
//   femClothSimPreIntegrateTGS   — cloth_preIntegrateLaunch TGS branch (vel + zero posDelta)
//   femClothSimFinalize          — cloth_finalizeVelocitiesLaunch settling + wake flag
//   femClothSimSleeping          — cloth_sleeping wake-counter bookkeeping

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers the engine
// descriptor header decorates its inline members with; must precede PxgFEMCloth.h.
#include "host_defines.h"
#include "PxgFEMCloth.h"   // real engine per-cloth descriptor

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };

// A device buffer + its buffer-device-address (BDA).
struct DevBuf
{
    PxgBufferHandle h = 0;
    uint64_t addr = 0;
};

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

static const float PX_MAX_REAL_F = 3.40282347e+38f;

// The three push-constant blobs, byte-identical to femClothSimBda.slang's
// [[vk::push_constant]] structs (which the shim reproduces from the CUDA arg pack).
struct PrePush
{
    uint64_t femCloths, activeIds;
    float gx, gy, gz, dt;
    uint32_t isTGSWord, pad36;
    uint64_t speculativeCCD;
    uint32_t extFlagsWord, pad52;
    uint64_t updateContactPairs, vtCounts, eeCounts;
};
static_assert(sizeof(PrePush) == 80, "PreIntegrate push-constant blob must be 80 bytes");

struct FinPush
{
    uint64_t femCloths, activeId;
    float invDt, dt;
    uint32_t alwaysAvgWord, pad28;
};
static_assert(sizeof(FinPush) == 32, "Finalize push-constant blob must be 32 bytes");

struct SleepPush
{
    uint64_t femCloths;
    uint32_t numActive, pad12;
    uint64_t activeId;
    float dt, resetCounter;
    uint64_t wakeCounters, stateChangedMask;
};
static_assert(sizeof(SleepPush) == 48, "Sleeping push-constant blob must be 48 bytes");

// Zeroed real PxgFEMCloth (no vtable — memset-safe, like PxgParticleSystem).
static PxgFEMCloth makeZeroDesc()
{
    PxgFEMCloth c;
    std::memset(&c, 0, sizeof(c));
    return c;
}

// ---------------------------------------------------------------------------
// Descriptor mirror layout proof — pins pxgSlangFEMClothDesc against the real
// engine PxgFEMCloth (offsets the three kernels dereference).  sizeof == 400.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimDescLayout)
{
    // sizeof / array stride: the cloth kernels index femCloths[id], so the STRIDE
    // must match (verified in SPIR-V as ArrayStride 400).
    PXG_CHECK_MSG(sizeof(PxgFEMCloth) == 400, "PxgFEMCloth sizeof == 400 (got %zu)", sizeof(PxgFEMCloth));

    // device-pointer members
    PXG_CHECK(offsetof(PxgFEMCloth, mVelocity_InvMass) == 8);
    PXG_CHECK(offsetof(PxgFEMCloth, mPosition_InvMass) == 16);
    PXG_CHECK(offsetof(PxgFEMCloth, mRestPosition) == 24);
    PXG_CHECK(offsetof(PxgFEMCloth, mPrevPositionInContactOffset) == 32);
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriPair_hasActiveBending) == 160);
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedRestEdge0_edge1) == 168); // +8 => 7 pad after the bool
    PXG_CHECK(offsetof(PxgFEMCloth, mAccumulatedDeltaPos) == 256);

    // scalar tail
    PXG_CHECK(offsetof(PxgFEMCloth, mLinearDamping) == 288);
    PXG_CHECK(offsetof(PxgFEMCloth, mMaxLinearVelocity) == 292);
    PXG_CHECK(offsetof(PxgFEMCloth, mSettlingThreshold) == 300);
    PXG_CHECK(offsetof(PxgFEMCloth, mSleepThreshold) == 304);
    PXG_CHECK(offsetof(PxgFEMCloth, mSettlingDamping) == 308);
    PXG_CHECK(offsetof(PxgFEMCloth, mNbVerts) == 316);
    PXG_CHECK(offsetof(PxgFEMCloth, mActorFlags) == 376);
    PXG_CHECK(offsetof(PxgFEMCloth, mBodyFlags) == 377);
    PXG_CHECK(offsetof(PxgFEMCloth, mSurfaceFlags) == 378);
    PXG_CHECK(offsetof(PxgFEMCloth, mIsActive) == 380);
    PXG_CHECK(offsetof(PxgFEMCloth, mRestDistance) == 384);

    // the byte-packed flag word (@376): low byte == mActorFlags (eDISABLE_GRAVITY test)
    PXG_CHECK(PxActorFlag::eDISABLE_GRAVITY == (1 << 1));
}

// ---------------------------------------------------------------------------
// Shared preIntegrate driver: dispatch + bit-exact CPU compare for a given
// isTGS mode.  invMass encoded in .w; vertex 3 has w==0 (skip branch).
// ---------------------------------------------------------------------------
static void runPreIntegrate(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const float3 g = { 0.0f, -9.81f, 0.0f };
    const float linearDamping = 0.1f;
    const float maxLinVel = 1.0e6f;    // finite but never reached -> no clamp
    const float restDistance = 0.05f;
    const uint32_t actorFlags = 0u;    // gravity enabled

    std::vector<F4> positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 3) ? 0.0f : 1.0f;
        positions[i] = { float(i) * 1.0f, 5.0f + float(i) * 0.25f, -float(i) * 0.5f, w };
        vels[i] = { 0.1f * float(i), -0.2f * float(i), 0.05f * float(i), 0.0f };
    }
    std::vector<F4> posDelta(N, { 9.f, 9.f, 9.f, 9.f });
    std::vector<F4> prevContact(N, { -1.f, -1.f, -1.f, -1.f });
    std::vector<uint32_t> vtCounts = { 123u }, eeCounts = { 456u };
    std::vector<float> speculativeCCD = { 7.0f };
    std::vector<uint8_t> updatePairs = { 9u };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, posDelta);
    DevBuf bPrev = makeDev(dev, prevContact);
    DevBuf bVt = makeDev(dev, vtCounts);
    DevBuf bEe = makeDev(dev, eeCounts);
    DevBuf bCcd = makeDev(dev, speculativeCCD);
    DevBuf bUpd = makeDev(dev, updatePairs);

    PxgFEMCloth c = makeZeroDesc();
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(bPrev.addr));
    c.mLinearDamping = linearDamping;
    c.mMaxLinearVelocity = maxLinVel;
    c.mRestDistance = restDistance;
    c.mNbVerts = N;
    c.mActorFlags = PxU8(actorFlags);

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    PrePush push = {};
    push.femCloths = bDesc.addr;
    push.activeIds = bActive.addr;
    push.gx = g.x; push.gy = g.y; push.gz = g.z; push.dt = dt;
    push.isTGSWord = isTGS ? 1u : 0u;
    push.speculativeCCD = bCcd.addr;
    push.extFlagsWord = 0u;                 // extForces=false, adaptivePairUpd=false
    push.updateContactPairs = bUpd.addr;
    push.vtCounts = bVt.addr;
    push.eeCounts = bEe.addr;

    Kernel k("femClothSimBda.spv", "cloth_preIntegrateLaunch", {}, sizeof(PrePush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference (bit-exact expression order, matching FEMCloth.cu:48).
    std::vector<F4> refPos = positions, refVel = vels, refDelta = posDelta, refPrev = prevContact;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4 tPos = positions[i];
        float vx = vels[i].x, vy = vels[i].y, vz = vels[i].z;
        if (tPos.w != 0.0f) // !externalForces && w!=0
        {
            vx += g.x * dt; vy += g.y * dt; vz += g.z * dt;   // gravity (actorFlags has no DISABLE_GRAVITY)
            const float damping = 1.0f - std::fmin(1.0f, linearDamping * dt);
            vx *= damping; vy *= damping; vz *= damping;
        }
        if (isTGS)
        {
            refVel[i] = { vx, vy, vz, tPos.w };
            refDelta[i] = { 0.f, 0.f, 0.f, tPos.w };
            // positions + prevContact unchanged
        }
        else
        {
            const float maxVel = (maxLinVel == PX_MAX_REAL_F) ? PX_MAX_REAL_F
                                 : (maxLinVel < 0.0f) ? (0.8f * restDistance / dt) : maxLinVel;
            const float velMagSq = vx * vx + vy * vy + vz * vz;
            if (maxVel != PX_MAX_REAL_F && maxVel * maxVel < velMagSq)
            {
                const float s = maxVel / std::sqrt(velMagSq);
                vx *= s; vy *= s; vz *= s;
            }
            const float dx = vx * dt, dy = vy * dt, dz = vz * dt;
            refVel[i] = { vx, vy, vz, tPos.w };
            refDelta[i] = { dx, dy, dz, tPos.w };
            refPos[i] = { tPos.x + dx, tPos.y + dy, tPos.z + dz, tPos.w };
            refPrev[i] = { tPos.x + dx, tPos.y + dy, tPos.z + dz, 1.0f };
        }
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);
    const auto gPrev = downloadBuffer<F4>(dev, bPrev.h, N);
    const auto gVt = downloadBuffer<uint32_t>(dev, bVt.h, 1);
    const auto gEe = downloadBuffer<uint32_t>(dev, bEe.h, 1);

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8)
                std::printf("  %s MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", what, i,
                            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp4(refPos[i], gPos[i], "pos", i);
        cmp4(refVel[i], gVel[i], "vel", i);
        cmp4(refDelta[i], gDelta[i], "delta", i);
        cmp4(refPrev[i], gPrev[i], "prevContact", i);
    }
    // globalThreadIndex==0, groupId.y==0, non-adaptive -> VT/EE counts reset to 0.
    if (gVt[0] != 0u) { ++mm; std::printf("  VTContactCounts MM: %u (expected 0)\n", gVt[0]); }
    if (gEe[0] != 0u) { ++mm; std::printf("  EEContactCounts MM: %u (expected 0)\n", gEe[0]); }

    PXG_CHECK_MSG(mm == 0, "cloth_preIntegrateLaunch (%s) bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bPrev.h); dev->destroyBuffer(bVt.h); dev->destroyBuffer(bEe.h);
    dev->destroyBuffer(bCcd.h); dev->destroyBuffer(bUpd.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bActive.h);
}

PXG_TEST(femClothSimPreIntegratePGS) { runPreIntegrate(/*isTGS*/ false); }
PXG_TEST(femClothSimPreIntegrateTGS) { runPreIntegrate(/*isTGS*/ true); }

// ---------------------------------------------------------------------------
// cloth_finalizeVelocitiesLaunch — velocity settling + wake flag
// ---------------------------------------------------------------------------
static void runFinalize(bool alwaysAvg)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const float settlingThreshold = 2.0f;
    const float sleepThreshold = 0.5f;
    const float settlingDamping = 4.0f;

    // Mixed deltas: some large (awake, scaling 1), some tiny (asleep, scaled),
    // vertex 5 has w==0 (skipped).  posDelta.w carries invMass (unused here).
    std::vector<F4> posDelta(N), positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 5) ? 0.0f : 1.0f;
        const float dmag = (i % 3 == 0) ? 0.2f : (i % 3 == 1) ? 0.0005f : 0.02f;
        posDelta[i] = { dmag, dmag * 0.5f, -dmag, w };
        positions[i] = { float(i), float(i) + 1.0f, float(i) + 2.0f, w };
        vels[i] = { 0.3f * float(i) + 0.1f, -0.2f * float(i) - 0.05f, 0.15f, w };
    }

    DevBuf bDelta = makeDev(dev, posDelta);
    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);

    PxgFEMCloth c = makeZeroDesc();
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mSettlingThreshold = settlingThreshold;
    c.mSleepThreshold = sleepThreshold;
    c.mSettlingDamping = settlingDamping;
    c.mNbVerts = N;
    c.mIsActive = 0u;

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    FinPush push = {};
    push.femCloths = bDesc.addr;
    push.activeId = bActive.addr;
    push.invDt = 1.0f / dt;
    push.dt = dt;
    push.alwaysAvgWord = alwaysAvg ? 1u : 0u;

    Kernel k("femClothSimBda.spv", "cloth_finalizeVelocitiesLaunch", {}, sizeof(FinPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference (FEMCloth.cu:1227).
    std::vector<F4> refVel = vels;
    bool anyAwake = false;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float settleTolerance = settlingThreshold * dt;
        const float tolerance = sleepThreshold * dt;
        const float sleepDamping = 1.0f - std::fmin(1.0f, settlingDamping * dt);

        const F4 delta = posDelta[i];
        const float tdx = delta.x, tdy = delta.y, tdz = delta.z;
        const float dvx = tdx / dt, dvy = tdy / dt, dvz = tdz / dt;
        const F4 pos = positions[i];

        if (pos.w != 0.0f)
        {
            float velocityScaling = 1.0f;
            const float magSq = tdx * tdx + tdy * tdy + tdz * tdz;
            bool awake;
            if (magSq < settleTolerance * settleTolerance)
            {
                awake = magSq >= tolerance * tolerance;
                velocityScaling = sleepDamping;
            }
            else
            {
                awake = true;
            }
            anyAwake = anyAwake || awake;

            if (alwaysAvg || velocityScaling != 1.0f)
            {
                const F4 vel = vels[i];
                float tvx = vel.x, tvy = vel.y, tvz = vel.z;
                const float deltaVelMagSqr = dvx * dvx + dvy * dvy + dvz * dvz;
                const float velMagSqr = tvx * tvx + tvy * tvy + tvz * tvz;
                if (alwaysAvg && deltaVelMagSqr < (tvx * tvx + tvy * tvy + tvz * tvz))
                {
                    const float s = std::sqrt(deltaVelMagSqr / velMagSqr);
                    tvx *= s; tvy *= s; tvz *= s;
                }
                refVel[i] = { velocityScaling * tvx, velocityScaling * tvy, velocityScaling * tvz, vel.w };
            }
        }
    }

    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDesc = downloadBuffer<PxgFEMCloth>(dev, bDesc.h, 1);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4& a = refVel[i]; const F4& b = gVel[i];
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8) std::printf("  vel MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", i,
                                    a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    }
    const uint32_t expectActive = anyAwake ? 1u : 0u;
    if (gDesc[0].mIsActive != expectActive)
    { ++mm; std::printf("  mIsActive MM: %u (expected %u)\n", gDesc[0].mIsActive, expectActive); }

    PXG_CHECK_MSG(mm == 0, "cloth_finalizeVelocitiesLaunch (alwaysAvg=%d) bit-exact vs CPU (%zu mismatches)",
                  int(alwaysAvg), mm);

    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(femClothSimFinalize)
{
    runFinalize(/*alwaysAvg*/ false); // the PGS free-fall path (velocities untouched when awake)
    runFinalize(/*alwaysAvg*/ true);  // the TGS averaging path
}

// ---------------------------------------------------------------------------
// cloth_sleeping — wake-counter bookkeeping + state-change ballot
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimSleeping)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float resetCounter = 0.4f;
    const uint32_t numActive = 3;

    // Three cloths exercising each transition:
    //  c0: active this frame (mIsActive=1), was asleep (wake=0)   -> reset->0.4, pred=true (bit0)
    //  c1: inactive, still awake (wake=0.1 -> 0.1-dt>0)           -> decay,     pred=false
    //  c2: inactive, falls asleep (wake=0.005 < dt -> 0)          -> decay->0,  pred=true (bit2)
    std::vector<uint32_t> isActive = { 1u, 0u, 0u };
    std::vector<float> wake = { 0.0f, 0.1f, 0.005f };

    std::vector<PxgFEMCloth> descArr(numActive);
    for (uint32_t i = 0; i < numActive; ++i)
    {
        descArr[i] = makeZeroDesc();
        descArr[i].mIsActive = isActive[i];
    }
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<uint32_t> active = { 0u, 1u, 2u };
    DevBuf bActive = makeDev(dev, active);
    DevBuf bWake = makeDev(dev, wake);
    // stateChangedMask: cloth_sleeping writes CLOTH_STEP/32 == 32 words per block.
    std::vector<uint32_t> stateMask(32, 0xDEADBEEFu);
    DevBuf bMask = makeDev(dev, stateMask);

    SleepPush push = {};
    push.femCloths = bDesc.addr;
    push.numActive = numActive;
    push.activeId = bActive.addr;
    push.dt = dt;
    push.resetCounter = resetCounter;
    push.wakeCounters = bWake.addr;
    push.stateChangedMask = bMask.addr;

    Kernel k("femClothSimBda.spv", "cloth_sleeping", {}, sizeof(SleepPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference.
    std::vector<float> refWake = wake;
    uint32_t refMask0 = 0u;
    for (uint32_t i = 0; i < numActive; ++i)
    {
        const bool reset = isActive[i] != 0u;
        float counter = wake[i];
        const bool wasActive = counter > 0.0f;
        counter = reset ? resetCounter : std::fmax(0.0f, counter - dt);
        const bool nowActive = counter > 0.0f;
        refWake[i] = counter;
        if (nowActive != wasActive)
            refMask0 |= (1u << i); // laneId == i, warpId == 0
    }

    const auto gWake = downloadBuffer<float>(dev, bWake.h, numActive);
    const auto gMask = downloadBuffer<uint32_t>(dev, bMask.h, 32);

    size_t mm = 0;
    for (uint32_t i = 0; i < numActive; ++i)
        if (f2u(refWake[i]) != f2u(gWake[i]))
        { ++mm; std::printf("  wake MM %u: ref %g got %g\n", i, refWake[i], gWake[i]); }

    if (gMask[0] != refMask0)
    { ++mm; std::printf("  stateMask[0] MM: got 0x%x expected 0x%x\n", gMask[0], refMask0); }
    for (uint32_t w = 1; w < 32; ++w) // warps with no active thread -> 0
        if (gMask[w] != 0u) { ++mm; std::printf("  stateMask[%u] MM: got 0x%x expected 0\n", w, gMask[w]); }

    PXG_CHECK_MSG(mm == 0, "cloth_sleeping bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bWake.h); dev->destroyBuffer(bMask.h);
}

} // namespace
