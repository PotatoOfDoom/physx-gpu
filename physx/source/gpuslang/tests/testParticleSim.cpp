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

// P6.0 particle bring-up test: the three PBD particle orchestration kernels
// (particleSim.spv) on the engine ABI (BDA), driven by the REAL engine
// descriptor.  A genuine PxgParticleSystem (gpusimulationcontroller header) is
// filled with device buffer addresses + scalars, DMA'd to the device, and read
// back through the Slang pxgSlangParticleDesc mirror — the first-ever
// PxgParticleSystem BDA round-trip.  static_asserts pin the mirror's byte
// offsets against the real engine struct (the highest-risk item), and each
// kernel's output is compared BIT-EXACT against a CPU transcription of the
// particlesystem.cu math.
//
// Suites (substring-matched by the "particleSim" ctest filter):
//   particleSimDescLayout  — static byte-offset proof of the descriptor mirror
//   particleSimPreIntegrate — ps_preIntegrateLaunch gravity predict
//   particleSimIntegrate    — ps_integrateLaunch post-solve writeback
//   particleSimUpdateBuffer — ps_updateUserBufferLaunch user-buffer copy
//   particleSimApplyBufferData — applyParticleBufferDataDEPRECATED (P9.95)

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in that expands the __device__/__align__ qualifiers the
// engine descriptor header decorates its inline members with (nocuda/ is on the
// test include path); must precede PxgParticleSystem.h so its device-qualified
// ctors parse.
#include "host_defines.h"
#include "PxgParticleSystem.h"   // real engine per-system descriptor

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
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

// The particlesystem.cu forward-projection scales.
static const float FWD_TGS = 0.5f;
static const float FWD_PGS = 1.0f;

// ---------------------------------------------------------------------------
// Descriptor mirror layout proof — pins pxgSlangParticleDesc against the real
// engine PxgParticleSystem (offsets the three kernels dereference).
// ---------------------------------------------------------------------------
PXG_TEST(particleSimDescLayout)
{
    // sizeof / array stride: the ps_* kernels index particleSystems[particleId],
    // so the STRIDE must match (verified in SPIR-V as ArrayStride 656 / 88).
    PXG_CHECK_MSG(sizeof(PxgParticleSystem) == 656, "PxgParticleSystem sizeof == 656 (got %zu)",
                  sizeof(PxgParticleSystem));
    PXG_CHECK_MSG(sizeof(PxgParticleSimBuffer) == 88, "PxgParticleSimBuffer sizeof == 88 (got %zu)",
                  sizeof(PxgParticleSimBuffer));

    // base PxGpuParticleSystem members
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedPositions_InvMass) == 0);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedVelocities) == 8);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedPhaseArray) == 16);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPositions_InvMass) == 24);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedVelocities) == 32);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPhaseArray) == 40);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedToUnsortedMapping) == 56);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mParticleMaterials) == 80);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) == 88);
    PXG_CHECK(offsetof(PxGpuParticleData, mNumParticles) == 28);        // -> struct off 116
    PXG_CHECK(offsetof(PxGpuParticleData, mNumParticleBuffers) == 44);  // -> struct off 132

    // derived PxgParticleSystem members
    PXG_CHECK(offsetof(PxgParticleSystem, mOriginPos_InvMass) == 136);
    PXG_CHECK(offsetof(PxgParticleSystem, mAccumDeltaP) == 152);
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedDeltaP) == 160);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayForces) == 208);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactCount) == 232);
    PXG_CHECK(offsetof(PxgParticleSystem, mData) == 248);
    PXG_CHECK(offsetof(PxgParticleSystem, mDelta) == 368);
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedOriginPos_InvMass) == 392);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleSimBuffers) == 400);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleMaterialStride) == 632);
    PXG_CHECK(offsetof(PxgParticleSystem, mPhaseGroupToMaterialHandle) == 640);
    // mData members (mLockFlags @320 rel-72, mMaxVelocity @316 rel-68)
    PXG_CHECK(offsetof(PxgParticleSystemData, mMaxVelocity) == 68);
    PXG_CHECK(offsetof(PxgParticleSystemData, mLockFlags) == 72);

    // PxgParticleSimBuffer members the ps_updateUserBufferLaunch kernel touches
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mPositionInvMasses) == 0);
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mVelocities) == 8);
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mNumActiveParticles) == 56);
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mFlags) == 72);
}

// Zeroes and returns a real PxgParticleSystem (no vtable — memset-safe).
static PxgParticleSystem makeZeroDesc()
{
    PxgParticleSystem ps;
    std::memset(&ps, 0, sizeof(ps));
    return ps;
}

// ---------------------------------------------------------------------------
// ps_preIntegrateLaunch — gravity predict
// ---------------------------------------------------------------------------
PXG_TEST(particleSimPreIntegrate)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;

    // Inputs.  Particle 3 has w==0 (infinite mass) to exercise the skip branch.
    std::vector<F4> positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 3) ? 0.0f : 1.0f;
        positions[i] = { float(i) * 1.0f, float(i) * 2.0f + 0.5f, -float(i) * 0.5f, w };
        vels[i] = { 0.1f * float(i), -0.2f * float(i), 0.05f * float(i), 0.0f };
    }
    std::vector<uint32_t> phases(N, 0u);
    std::vector<F4> oriPos(N, { -1.f, -1.f, -1.f, -1.f });
    std::vector<F4> accumDeltaP(N, { 9.f, 9.f, 9.f, 9.f });
    std::vector<uint32_t> contactCounts(N, 7u);
    // oneWayForces is a float2 array (12 * N entries).
    std::vector<float> oneWayForces(2 * 12 * N, 5.0f);

    // material: friction, damping, adhesion, gravityScale, adhesionRadiusScale
    const float matDamping = 0.25f, matGravityScale = 2.0f;
    std::vector<float> material = { 0.5f, matDamping, 0.0f, matGravityScale, 0.0f };
    const uint32_t materialStride = 20u; // sizeof our 5-float material
    std::vector<uint16_t> phaseToMat = { 0u, 0u };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bPhase = makeDev(dev, phases);
    DevBuf bOri = makeDev(dev, oriPos);
    DevBuf bAccum = makeDev(dev, accumDeltaP);
    DevBuf bCounts = makeDev(dev, contactCounts);
    DevBuf bForces = makeDev(dev, oneWayForces);
    DevBuf bMat = makeDev(dev, material);
    DevBuf bP2M = makeDev(dev, phaseToMat);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(bVel.addr));
    ps.mUnsortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bPhase.addr));
    ps.mOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bOri.addr));
    ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(bAccum.addr));
    ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(bCounts.addr));
    ps.mOneWayForces = reinterpret_cast<float2*>(size_t(bForces.addr));
    ps.mParticleMaterials = reinterpret_cast<PxsParticleMaterialData*>(size_t(bMat.addr));
    ps.mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(bP2M.addr));
    ps.mParticleMaterialStride = materialStride;
    ps.mCommonData.mNumParticles = N;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct PrePush
    {
        uint64_t particleSystems, activeParticleSystems;
        float gx, gy, gz, dt;
        uint32_t flagsWord, pad;
    };
    const float dt = 1.0f / 60.0f;
    const bool isTGS = true, extForces = false;
    PrePush push = { bDesc.addr, bActive.addr, 0.0f, -9.81f, 0.0f, dt,
                     (isTGS ? 1u : 0u) | (extForces ? 0x100u : 0u), 0u };

    Kernel k("particleSim.spv", "ps_preIntegrateLaunch", {}, sizeof(PrePush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference (bit-exact expression order).
    std::vector<F4> refPos = positions, refVel = vels, refOri(N);
    std::vector<F4> refAccum(N, { 0.f, 0.f, 0.f, 0.f });
    std::vector<uint32_t> refCounts(N, 0u);
    std::vector<float> refForces(2 * 12 * N, 5.0f);
    const float3 g = { 0.0f, -9.81f, 0.0f };
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4 tPos = positions[i];
        refOri[i] = tPos;
        const float damping = 1.0f - std::fmin(1.0f, matDamping * dt);
        const float tgx = g.x * matGravityScale, tgy = g.y * matGravityScale, tgz = g.z * matGravityScale;
        float vx = 0.f, vy = 0.f, vz = 0.f;
        if (tPos.w != 0.0f)
        {
            const F4 tv = vels[i];
            vx = (tv.x + tgx * dt) * damping;
            vy = (tv.y + tgy * dt) * damping;
            vz = (tv.z + tgz * dt) * damping;
            const float fwd = isTGS ? FWD_TGS : FWD_PGS;
            refPos[i] = { tPos.x + vx * dt * fwd, tPos.y + vy * dt * fwd, tPos.z + vz * dt * fwd, tPos.w };
        }
        if (!extForces)
            refVel[i] = { vx, vy, vz, tPos.w };
        for (uint32_t j = i; j < 12 * N; j += N) { refForces[2 * j] = 0.f; refForces[2 * j + 1] = 0.f; }
    }
    (void)refForces;

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gOri = downloadBuffer<F4>(dev, bOri.h, N);
    const auto gAccum = downloadBuffer<F4>(dev, bAccum.h, N);
    const auto gCounts = downloadBuffer<uint32_t>(dev, bCounts.h, N);
    const auto gForces = downloadBuffer<float>(dev, bForces.h, 2 * 12 * N);

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
        cmp4(refOri[i], gOri[i], "ori", i);
        cmp4(refAccum[i], gAccum[i], "accum", i);
        if (gCounts[i] != refCounts[i]) { ++mm; std::printf("  count MM %u\n", i); }
    }
    for (uint32_t j = 0; j < 2 * 12 * N; ++j)
        if (f2u(refForces[j]) != f2u(gForces[j])) { if (mm < 12) std::printf("  force MM %u\n", j); ++mm; }

    PXG_CHECK_MSG(mm == 0, "ps_preIntegrateLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bPhase.h);
    dev->destroyBuffer(bOri.h); dev->destroyBuffer(bAccum.h); dev->destroyBuffer(bCounts.h);
    dev->destroyBuffer(bForces.h); dev->destroyBuffer(bMat.h); dev->destroyBuffer(bP2M.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// ps_preDiffuseIntegrateLaunch — the DIFFUSE gravity predict (P9.13b)
// ---------------------------------------------------------------------------
// Two dispatches, because the CUDA's reset of mDiffuseOneWayContactCount is
// guarded on the POINTER being non-null (trap D6) and a single dispatch can only
// be on one side of that.
//
// Everything the solid twin has and this one does not is checked by CONSTRUCTION
// rather than by inspection: the descriptor's material, phase and damping fields
// are left at zero and the arrays for them are NOT BOUND AT ALL.  A port that
// copied the solid body across would dereference a null material pointer and
// fault, which is a louder failure than a wrong number.
//
// THE GEOMETRY, and every part of it is load-bearing -- the first sweep lost two
// mutants to a fixture that had one system and one block:
//
//   * MAXD = 1300 > 1024, so the launch is TWO blocks and the
//     `groupId.x * blockDim` term is live.  LIVE = 1100 crosses the block
//     boundary, so both blocks carry a mix of live and dead slots.
//   * TWO particle systems, with DIFFERENT arrays and DIFFERENT live counts, and
//     an active list that is a PERMUTATION ({1, 0}).  That separates three
//     things a single system cannot: groupId.y from groupId.x, and the ACTIVE
//     index from the DESCRIPTOR index.
//   * a poison tail past the allocation in every array.
static void runPreDiffuseIntegrate(PxgSlangComputeDevice* dev, const char* label, bool bindCounts)
{
    const uint32_t MAXD = 1300u;   // > 1024: two blocks
    const uint32_t PAD  = 37u;     // poison past the allocation
    const uint32_t TOT  = MAXD + PAD;
    // Per system, and DIFFERENT: 1100 crosses the 1024 block boundary, 700 does
    // not, so a kernel that took the wrong system's count is wrong in both
    // directions at once.
    const uint32_t LIVE[2] = { 1100u, 700u };

    // sys 0 and sys 1 get different data as well as different counts.
    std::vector<F4> positions[2], vels[2];
    for (uint32_t sysi = 0; sysi < 2; ++sysi)
    {
        positions[sysi].resize(TOT);
        vels[sysi].resize(TOT);
        const float bias = sysi == 0 ? 0.0f : 100.0f;
        for (uint32_t i = 0; i < TOT; ++i)
        {
            // w is the LIFETIME.  Slot 2 is exactly 0 and slot 4 negative: a
            // diffuse particle on (or past) its last frame must STILL be
            // integrated -- trap D2, there is no `w != 0` guard.
            const float life = (i == 2u) ? 0.0f
                             : (i == 4u) ? -1.5f
                             : (0.5f + 0.001f * float(i) + 0.03f * float(sysi));
            positions[sysi][i] = { bias + float(i) * 0.017f - 3.0f,
                                   bias + 2.0f + float(i) * 0.005f,
                                   bias - 0.75f + 0.011f * float(i), life };
            // A w nothing may ever read (trap D5: the velocity's w is written
            // FROM the position's w).
            vels[sysi][i] = { 0.002f * float(i) + 0.1f * float(sysi),
                              -0.0035f * float(i) + 0.1f,
                              0.0007f * float(i) - 0.05f,
                              -99.0f - float(i) - 1000.0f * float(sysi) };
        }
    }
    std::vector<F4> oriPos(TOT, { -1.f, -1.f, -1.f, -1.f });
    std::vector<uint32_t> counts(TOT, 7u);

    DevBuf bPos[2], bVel[2], bOri[2], bCounts[2], bLive[2];
    for (uint32_t sysi = 0; sysi < 2; ++sysi)
    {
        bPos[sysi] = makeDev(dev, positions[sysi]);
        bVel[sysi] = makeDev(dev, vels[sysi]);
        bOri[sysi] = makeDev(dev, oriPos);
        bCounts[sysi] = makeDev(dev, counts);
        const std::vector<uint32_t> liveCount = { LIVE[sysi], 0xEEEEEEEEu }; // int[2]
        bLive[sysi] = makeDev(dev, liveCount);
    }

    std::vector<PxgParticleSystem> descArr(2);
    for (uint32_t sysi = 0; sysi < 2; ++sysi)
    {
        PxgParticleSystem ps = makeZeroDesc();
        ps.mDiffusePosition_LifeTime = reinterpret_cast<float4*>(size_t(bPos[sysi].addr));
        ps.mDiffuseVelocity = reinterpret_cast<float4*>(size_t(bVel[sysi].addr));
        ps.mDiffuseOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(bOri[sysi].addr));
        ps.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(bLive[sysi].addr));
        ps.mDiffuseOneWayContactCount = bindCounts
            ? reinterpret_cast<PxU32*>(size_t(bCounts[sysi].addr)) : nullptr;
        // D1: this is the ALLOCATION and the kernel must NOT read it as the count.
        ps.mCommonData.mMaxDiffuseParticles = MAXD;
        // ...and the SOLID count, which is not this kernel's business either.
        ps.mCommonData.mNumParticles = TOT + 11u;
        descArr[sysi] = ps;
    }
    DevBuf bDesc = makeDev(dev, descArr);
    // A PERMUTATION: block y=0 works on descriptor 1, y=1 on descriptor 0.
    const std::vector<uint32_t> active = { 1u, 0u };
    DevBuf bActive = makeDev(dev, active);

    struct PreDPush
    {
        uint64_t particleSystems, activeParticleSystems;
        float gx, gy, gz, dt;
    };
    static_assert(sizeof(PreDPush) == 32, "the diffuse predict's blob is 32 bytes, not 40");
    const float dt = 1.0f / 90.0f;                       // not 1/60, and not 1
    const float gx = 0.7f, gy = -9.81f, gz = -0.35f;     // three distinct, none 0
    PreDPush push = { bDesc.addr, bActive.addr, gx, gy, gz, dt };

    {
        Kernel k("particleSim.spv", "ps_preDiffuseIntegrateLaunch", {}, sizeof(PreDPush));
        PXG_CHECK_MSG(k.isValid(), "%s: the pipeline must build", label);
        if (!k.isValid()) { return; }
        // grid = (ceil(MAXD/1024), nbActiveSystems, 1) = (2, 2, 1)
        k.dispatch((MAXD + 1023u) / 1024u, {}, push, /*gridY*/ 2u, /*gridZ*/ 1u);
    }
    dev->flush();

    size_t mm = 0;
    for (uint32_t sysi = 0; sysi < 2; ++sysi)
    {
        const auto gPos = downloadBuffer<F4>(dev, bPos[sysi].h, TOT);
        const auto gVel = downloadBuffer<F4>(dev, bVel[sysi].h, TOT);
        const auto gOri = downloadBuffer<F4>(dev, bOri[sysi].h, TOT);
        const auto gCnt = downloadBuffer<uint32_t>(dev, bCounts[sysi].h, TOT);
        const uint32_t live = LIVE[sysi];

        for (uint32_t i = 0; i < TOT; ++i)
        {
            const bool alive = i < live;
            // The host mirror: raw gravity, no damping, no forward-projection scale.
            const float vxr = alive ? vels[sysi][i].x + gx * dt : vels[sysi][i].x;
            const float vyr = alive ? vels[sysi][i].y + gy * dt : vels[sysi][i].y;
            const float vzr = alive ? vels[sysi][i].z + gz * dt : vels[sysi][i].z;
            const float pxr = alive ? positions[sysi][i].x + vxr * dt : positions[sysi][i].x;
            const float pyr = alive ? positions[sysi][i].y + vyr * dt : positions[sysi][i].y;
            const float pzr = alive ? positions[sysi][i].z + vzr * dt : positions[sysi][i].z;
            // D5: the velocity's w becomes the POSITION's w.
            const float vwr = alive ? positions[sysi][i].w : vels[sysi][i].w;

            if (f2u(gPos[i].x) != f2u(pxr) || f2u(gPos[i].y) != f2u(pyr) ||
                f2u(gPos[i].z) != f2u(pzr) || f2u(gPos[i].w) != f2u(positions[sysi][i].w))
            {
                if (mm < 4)
                    std::printf("  %s sys%u pos MM %u: got (%.9g,%.9g,%.9g,%.9g) want "
                                "(%.9g,%.9g,%.9g,%.9g)\n", label, sysi, i, gPos[i].x, gPos[i].y,
                                gPos[i].z, gPos[i].w, pxr, pyr, pzr, positions[sysi][i].w);
                ++mm;
            }
            if (f2u(gVel[i].x) != f2u(vxr) || f2u(gVel[i].y) != f2u(vyr) ||
                f2u(gVel[i].z) != f2u(vzr) || f2u(gVel[i].w) != f2u(vwr))
            {
                if (mm < 4)
                    std::printf("  %s sys%u vel MM %u: got (%.9g,%.9g,%.9g,%.9g) want "
                                "(%.9g,%.9g,%.9g,%.9g)\n", label, sysi, i, gVel[i].x, gVel[i].y,
                                gVel[i].z, gVel[i].w, vxr, vyr, vzr, vwr);
                ++mm;
            }
            // The ORIGIN buffer takes the PRE-integration position, lifetime
            // included; past the live count it must still be poison.
            const F4 oriWant = alive ? positions[sysi][i] : F4{ -1.f, -1.f, -1.f, -1.f };
            if (f2u(gOri[i].x) != f2u(oriWant.x) || f2u(gOri[i].y) != f2u(oriWant.y) ||
                f2u(gOri[i].z) != f2u(oriWant.z) || f2u(gOri[i].w) != f2u(oriWant.w))
            {
                if (mm < 4)
                    std::printf("  %s sys%u ori MM %u: got (%.9g,%.9g,%.9g,%.9g) want "
                                "(%.9g,%.9g,%.9g,%.9g)\n", label, sysi, i, gOri[i].x, gOri[i].y,
                                gOri[i].z, gOri[i].w, oriWant.x, oriWant.y, oriWant.z, oriWant.w);
                ++mm;
            }
            const uint32_t cntWant = (bindCounts && alive) ? 0u : 7u;
            if (gCnt[i] != cntWant)
            {
                if (mm < 4)
                    std::printf("  %s sys%u cnt MM %u: got %u want %u\n",
                                label, sysi, i, gCnt[i], cntWant);
                ++mm;
            }
        }
    }
    PXG_CHECK_MSG(mm == 0, "%s: ps_preDiffuseIntegrateLaunch bit-exact (%zu mismatches)",
                  label, mm);

    for (uint32_t sysi = 0; sysi < 2; ++sysi)
    {
        dev->destroyBuffer(bPos[sysi].h); dev->destroyBuffer(bVel[sysi].h);
        dev->destroyBuffer(bOri[sysi].h); dev->destroyBuffer(bCounts[sysi].h);
        dev->destroyBuffer(bLive[sysi].h);
    }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(particleSimPreDiffuseIntegrate)
{
    PxgSlangComputeDevice* dev = testDevice();
    runPreDiffuseIntegrate(dev, "counts-bound", true);
    // D6: with the pointer null the kernel must NOT write, and must not fault.
    // The array is still uploaded and read back, so "did not write" is checked
    // rather than assumed.
    runPreDiffuseIntegrate(dev, "counts-null", false);
}

// ---------------------------------------------------------------------------
// ps_integrateLaunch — post-solve writeback (populated sorted arrays)
// ---------------------------------------------------------------------------
PXG_TEST(particleSimIntegrate)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float eps = 1e-6f;
    const float maxVel = 5.0f;

    // sorted->unsorted reverse permutation (all in-range).
    std::vector<uint32_t> mapping(N);
    for (uint32_t i = 0; i < N; ++i) mapping[i] = N - 1 - i;

    std::vector<F4> sortedDeltaP(N), sortedOrigin(N), sortedPos(N), sortedVel(N), delta(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        // particle 0: zero delta -> locked; particle 5: newPos.w == 0 -> skip.
        sortedDeltaP[i] = (i == 0) ? F4{ 0, 0, 0, 0 } : F4{ 0.1f, 0.2f, 0.3f, 0.f };
        sortedOrigin[i] = { float(i), float(i) + 1.f, float(i) + 2.f, 1.f };
        const float w = (i == 5) ? 0.0f : 1.0f;
        sortedPos[i] = { 10.f + float(i), 20.f + float(i), 30.f + float(i), w };
        sortedVel[i] = { 0.5f * float(i), -0.3f * float(i), 0.2f, 0.f };
        delta[i] = { 0.01f * float(i), 0.02f * float(i), 0.03f * float(i), float(i) + 1.f };
    }
    // unsorted position / velocity buffers pre-seeded with a sentinel.
    std::vector<F4> position(N, { -100.f, -100.f, -100.f, -100.f });
    std::vector<F4> originalVels(N, { -50.f, -50.f, -50.f, -50.f });

    DevBuf bPos = makeDev(dev, position);
    DevBuf bNewPos = makeDev(dev, sortedPos);
    DevBuf bOrig = makeDev(dev, sortedOrigin);
    DevBuf bDeltaP = makeDev(dev, sortedDeltaP);
    DevBuf bOrigVel = makeDev(dev, originalVels);
    DevBuf bNewVel = makeDev(dev, sortedVel);
    DevBuf bDelta = makeDev(dev, delta);
    DevBuf bMap = makeDev(dev, mapping);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bNewPos.addr));
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bOrig.addr));
    ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(bDeltaP.addr));
    ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(bOrigVel.addr));
    ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(bNewVel.addr));
    ps.mDelta = reinterpret_cast<float4*>(size_t(bDelta.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bMap.addr));
    ps.mData.mMaxVelocity = maxVel;
    ps.mData.mLockFlags = 0;
    ps.mCommonData.mNumParticles = N;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct IntPush { uint64_t particleSystems, activeParticleSystems; float dt, eps; };
    IntPush push = { bDesc.addr, bActive.addr, 1.0f / 60.0f, eps };

    Kernel k("particleSim.spv", "ps_integrateLaunch", {}, sizeof(IntPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference.
    std::vector<F4> refPos = position, refVel = originalVels, refDelta = delta;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t sortedIndex = mapping[i];
        const float dsq = sortedDeltaP[i].x * sortedDeltaP[i].x + sortedDeltaP[i].y * sortedDeltaP[i].y +
                          sortedDeltaP[i].z * sortedDeltaP[i].z;
        const bool locked = dsq <= eps;
        const F4 origPos4 = sortedOrigin[i];
        F4 pos4 = locked ? origPos4 : sortedPos[i];
        if (pos4.w > 0.0f)
        {
            const float mv = locked ? 0.0f : maxVel;
            const F4 nv = sortedVel[i];
            const F4 j = delta[i];
            const float scale = 1.0f / std::fmax(j.w, 1.0f);
            float vx = nv.x + j.x * scale, vy = nv.y + j.y * scale, vz = nv.z + j.z * scale;
            const float magSq = vx * vx + vy * vy + vz * vz;
            if (magSq > mv * mv)
            {
                // Only hit for the locked (mv==0) particle -> ratio 0 -> zero (rsqrt-independent).
                const float ratio = mv * (1.0f / std::sqrt(magSq));
                vx *= ratio; vy *= ratio; vz *= ratio;
            }
            refPos[sortedIndex] = pos4;
            refVel[sortedIndex] = { vx, vy, vz, 0.0f };
            refDelta[i] = { 0.f, 0.f, 0.f, 0.f };
        }
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bOrigVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);

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
    }
    PXG_CHECK_MSG(mm == 0, "ps_integrateLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bNewPos.h); dev->destroyBuffer(bOrig.h);
    dev->destroyBuffer(bDeltaP.h); dev->destroyBuffer(bOrigVel.h); dev->destroyBuffer(bNewVel.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bMap.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// ps_updateUnsortedArrayLaunch — ingest user buffer -> unsorted (flag-gated)
// ---------------------------------------------------------------------------
PXG_TEST(particleSimIngest)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const uint32_t numActive = 6;
    const uint32_t bufOffset = 2;
    const uint32_t flags = 1u | 2u | 4u; // POSITION | VELOCITY | PHASE (no RESTPOSITION)

    std::vector<F4> userPos(numActive), userVel(numActive);
    std::vector<uint32_t> userPhase(numActive);
    for (uint32_t j = 0; j < numActive; ++j)
    {
        userPos[j] = { float(j) + 0.5f, 100.f + float(j), -float(j) * 2.f, 1.f };
        userVel[j] = { -0.2f * float(j), 0.1f, 0.4f * float(j), 0.f };
        userPhase[j] = 0x40000000u | j;
    }
    // unsorted arrays pre-seeded with a sentinel so untouched slots are visible.
    std::vector<F4> unsortedPos(N, { -1.f, -1.f, -1.f, -1.f });
    std::vector<F4> unsortedVel(N, { -2.f, -2.f, -2.f, -2.f });
    std::vector<uint32_t> unsortedPhase(N, 0xffffffffu);
    std::vector<F4> restArray(N, { -3.f, -3.f, -3.f, -3.f });

    DevBuf bUPos = makeDev(dev, unsortedPos);
    DevBuf bUVel = makeDev(dev, unsortedVel);
    DevBuf bUPhase = makeDev(dev, unsortedPhase);
    DevBuf bRest = makeDev(dev, restArray);
    DevBuf bUserPos = makeDev(dev, userPos);
    DevBuf bUserVel = makeDev(dev, userVel);
    DevBuf bUserPhase = makeDev(dev, userPhase);
    std::vector<uint32_t> runsum = { bufOffset };
    DevBuf bRunsum = makeDev(dev, runsum);

    PxgParticleSimBuffer simBuf;
    std::memset(&simBuf, 0, sizeof(simBuf));
    simBuf.mPositionInvMasses = reinterpret_cast<float4*>(size_t(bUserPos.addr));
    simBuf.mVelocities = reinterpret_cast<float4*>(size_t(bUserVel.addr));
    simBuf.mPhases = reinterpret_cast<PxU32*>(size_t(bUserPhase.addr));
    simBuf.mRestPositions = nullptr; // exercise the restPose null guard
    simBuf.mNumActiveParticles = numActive;
    simBuf.mFlags = flags;
    std::vector<PxgParticleSimBuffer> simArr = { simBuf };
    DevBuf bSim = makeDev(dev, simArr);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bUPos.addr));
    ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(bUVel.addr));
    ps.mUnsortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bUPhase.addr));
    ps.mRestArray = reinterpret_cast<float4*>(size_t(bRest.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSim.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
    ps.mCommonData.mNumParticleBuffers = 1;
    ps.mCommonData.mNumParticles = N;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct IngestPush { uint64_t particleSystems, activeParticleSystems; };
    IngestPush push = { bDesc.addr, bActive.addr };

    Kernel k("particleSim.spv", "ps_updateUnsortedArrayLaunch", {}, sizeof(IngestPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference: unsorted[offset+j] = user[j] for j < numActive; rest untouched.
    std::vector<F4> refPos = unsortedPos, refVel = unsortedVel, refRest = restArray;
    std::vector<uint32_t> refPhase = unsortedPhase;
    for (uint32_t j = 0; j < numActive; ++j)
    {
        refPos[bufOffset + j] = userPos[j];
        refVel[bufOffset + j] = userVel[j];
        refPhase[bufOffset + j] = userPhase[j];
    }

    const auto gPos = downloadBuffer<F4>(dev, bUPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bUVel.h, N);
    const auto gPhase = downloadBuffer<uint32_t>(dev, bUPhase.h, N);
    const auto gRest = downloadBuffer<F4>(dev, bRest.h, N);

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        { if (mm < 8) std::printf("  %s MM %u\n", what, i); ++mm; }
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp4(refPos[i], gPos[i], "pos", i);
        cmp4(refVel[i], gVel[i], "vel", i);
        cmp4(refRest[i], gRest[i], "rest", i); // must be untouched (restPose null)
        if (gPhase[i] != refPhase[i]) { ++mm; if (mm < 8) std::printf("  phase MM %u\n", i); }
    }
    PXG_CHECK_MSG(mm == 0, "ps_updateUnsortedArrayLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bUPos.h); dev->destroyBuffer(bUVel.h); dev->destroyBuffer(bUPhase.h);
    dev->destroyBuffer(bRest.h); dev->destroyBuffer(bUserPos.h); dev->destroyBuffer(bUserVel.h);
    dev->destroyBuffer(bUserPhase.h); dev->destroyBuffer(bRunsum.h); dev->destroyBuffer(bSim.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// applyParticleBufferDataDEPRECATED (P9.95) -- direct-GPU-API ingest by UNIQUE ID
// (particlesystem.cu:5831-5866; helper particleSystem.cuh:99-117; host pack
// PxgPBDParticleSystemCore.cpp:2279-2285).
//
// The oracle is the CUDA spelled literally on the REAL types (PxgParticleSystem,
// PxgParticleSimBuffer, PxGpuParticleBufferIndexPair, PxParticleBufferFlag) over HOST
// shadow descriptors whose pointers are host pointers, looped as by / bx / tx < 1024.
// The device descriptors are the same records with device addresses.  Every region is
// compared word by word.
//
// The data is chosen so each wrong spelling changes a word (asserted on the host by
// the variant oracles below, before the GPU runs):
//   * TWO systems.  System 0 registers ids {7, 2, 12} (original index order), system 1
//     ids {20, 7}: the shared id 7 resolves to buffer 0 in system 0 and buffer 1 in
//     system 1, so the search table must come from pair.systemIndex.
//   * indices = {3, 0, 5, 4}: blockIdx.y -> slot is never the identity, and the slots
//     blockIdx.y names directly (0..3) all differ from the named ones in id, and for
//     blockIdx.y = 1 and 2 in system as well.
//   * id 2 is an in-range buffer index that resolves to original index 1, so a
//     missing lookup reads a live buffer, not only out of bounds.
//   * every source array carries 4 slack slots of distinct data past its active
//     count, so `<=` for `<` writes a visible word over the sentinel at offset+active.
//   * the lanes include -0, +0, qNaN 0x7fc12345, the sNaN pattern 0x7f812345, +-inf
//     and denormals, so a float-path copy that canonicalises is visible.
//   * launch 2 applies the UNREGISTERED id 9 (upstream resolves it to its predecessor
//     7, original buffer 0) with VEL alone; launch 3 applies to a system whose
//     mNumParticleBuffers is 0 (the early return gives buffer 0).
// ---------------------------------------------------------------------------
static_assert(sizeof(PxGpuParticleBufferIndexPair) == 8, "index pair is 8 bytes");
static_assert(offsetof(PxGpuParticleBufferIndexPair, systemIndex) == 0, "systemIndex @0");
static_assert(offsetof(PxGpuParticleBufferIndexPair, bufferIndex) == 4, "bufferIndex @4");
static_assert(sizeof(PxParticleBufferFlags) == 4, "PxParticleBufferFlags is PxFlags<,PxU32>");
// The kernel's parameter is `const PxParticleBufferFlag*`, a 1-byte (empty) struct, but
// it reads the array through `const PxU32*` (:5854): this pins the trap.
static_assert(sizeof(PxParticleBufferFlag) == 1, "the declared element type of the kernel's dirtyFlags");
static_assert(PxParticleBufferFlag::eUPDATE_POSITION == 1 && PxParticleBufferFlag::eUPDATE_VELOCITY == 2 &&
              PxParticleBufferFlag::eUPDATE_PHASE == 4 && PxParticleBufferFlag::eUPDATE_RESTPOSITION == 8,
              "PxParticleSystemFlag.h:51-54");
static_assert(offsetof(PxgParticleSystem, mParticleSimBuffers) == 400, "@400");
static_assert(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408, "@408");
static_assert(offsetof(PxgParticleSystem, mParticleBufferSortedUniqueIds) == 416, "@416");
static_assert(offsetof(PxgParticleSystem, mParticleBufferSortedUniqueIdsOriginalIndex) == 424, "@424");
static_assert(offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mNumParticleBuffers) == 132,
              "@132");
static_assert(sizeof(PxgParticleSimBuffer) == 88, "sim buffer stride 88");
static_assert(offsetof(PxgParticleSimBuffer, mNumActiveParticles) == 56, "@56");

// The push block == the host pack (:2279-2285): four 8-byte device pointers.
struct ApplyPush { uint64_t ps, indices, pairs, flags; };
static_assert(sizeof(ApplyPush) == 32, "applyParticleBufferDataDEPRECATED push block is 32 bytes");

// Wrong spellings the oracle can be asked for (the self-check).
enum ApplyVariant
{
    AV_CUDA = 0,
    AV_FLAGS_BY_Y,      // (a) flags[blockIdx.y]
    AV_LOWER_BOUND,     // (b) values[m] >= id
    AV_NO_OFFSET,       // (c) ind = gtid
    AV_PAIR_BY_Y,       // (d) indexPairs[blockIdx.y]
    AV_SEARCH_SYS0,     // (e) the search done in system 0's table
};

// particleSystem.cuh:99-117, verbatim.
static PxU32 refFindBufferIndexFromUniqueId(const PxgParticleSystem& particleSystem, PxU32 uniqueBufferId,
                                            ApplyVariant v)
{
    const PxU32 length = particleSystem.mCommonData.mNumParticleBuffers;
    if (length == 0)
        return 0;

    const PxU32* values = particleSystem.mParticleBufferSortedUniqueIds;

    PxU32 l = 0, r = length;
    while (l < r)
    {
        PxU32 m = (l + r) / 2;
        if (v == AV_LOWER_BOUND ? values[m] >= uniqueBufferId : values[m] > uniqueBufferId)
            r = m;
        else
            l = m + 1;
    }
    // Self-check only: the lower bound of the SMALLEST registered id leaves r == 0, and the
    // r - 1 read would leave the table; the host clamps it rather than fault.  The CUDA
    // spelling never takes this branch for a registered id.
    if (v == AV_LOWER_BOUND && r == 0)
        r = 1;
    return particleSystem.mParticleBufferSortedUniqueIdsOriginalIndex[r - 1];
}

// particlesystem.cu:5831-5866, verbatim, one CUDA thread per (by, bx, tx).
static void refApplyParticleBufferData(const PxgParticleSystem* particleSystems, const PxU32* indices,
                                       const PxGpuParticleBufferIndexPair* indexPairs,
                                       const PxParticleBufferFlag* dirtyFlags, PxU32 gridX, PxU32 gridY,
                                       ApplyVariant v)
{
    const PxU32 blockDimX = 1024;
    for (PxU32 by = 0; by < gridY; ++by)
        for (PxU32 bx = 0; bx < gridX; ++bx)
            for (PxU32 tx = 0; tx < blockDimX; ++tx)
            {
                const PxU32 index = indices[by];
                const PxGpuParticleBufferIndexPair indexPair = indexPairs[v == AV_PAIR_BY_Y ? by : index];
                const PxgParticleSystem& particleSystem = particleSystems[indexPair.systemIndex];

                float4* unsortedPositions = reinterpret_cast<float4*>(particleSystem.mUnsortedPositions_InvMass);
                float4* unsortedVels = reinterpret_cast<float4*>(particleSystem.mUnsortedVelocities);

                PxU32 bufferId = refFindBufferIndexFromUniqueId(
                    v == AV_SEARCH_SYS0 ? particleSystems[0] : particleSystem, indexPair.bufferIndex, v);

                PxgParticleSimBuffer& buffer = particleSystem.mParticleSimBuffers[bufferId];
                const PxU32 offset = particleSystem.mParticleBufferRunsum[bufferId];

                const float4* particles = buffer.mPositionInvMasses;
                const float4* velocities = buffer.mVelocities;

                const PxU32 globalThreadIndex = tx + blockDimX * bx;

                const PxU32* flagsPtr = reinterpret_cast<const PxU32*>(dirtyFlags);
                const PxU32 flags = flagsPtr[v == AV_FLAGS_BY_Y ? by : index];

                if (globalThreadIndex < buffer.mNumActiveParticles)
                {
                    const PxU32 ind = (v == AV_NO_OFFSET ? 0u : offset) + globalThreadIndex;
                    if (flags & PxParticleBufferFlag::eUPDATE_POSITION)
                        std::memcpy(&unsortedPositions[ind], &particles[globalThreadIndex], 16);

                    if (flags & PxParticleBufferFlag::eUPDATE_VELOCITY)
                        std::memcpy(&unsortedVels[ind], &velocities[globalThreadIndex], 16);
                }
            }
}

// One particle system's arrays, host side, with the matching device buffers.
struct ApplySys
{
    uint32_t numUnsorted = 0;
    std::vector<uint32_t> uPos, uVel, uPhase, uRest;      // 4 words per float4 (phase: 1)
    std::vector<std::vector<uint32_t>> userPos, userVel;  // per buffer, active + 4 slack slots
    std::vector<uint32_t> ids, sorted, origIdx, runsum, active;
    std::vector<PxgParticleSimBuffer> simHost;            // host pointers (oracle)
    std::vector<PxgParticleSimBuffer> simDev;             // device addresses (uploaded)
    DevBuf bUPos, bUVel, bUPhase, bURest, bSorted, bOrig, bRunsum, bSim;
    std::vector<DevBuf> bUserPos, bUserVel;
    PxgParticleSystem descHost, descDev;
};

static const uint32_t kApplySpecials[8] = { 0x80000000u, 0x00000000u, 0x7fc12345u, 0x7f812345u,
                                            0x7f800000u, 0xff800000u, 0x00000001u, 0x807fffffu };

static void applyBuildSys(PxgSlangComputeDevice* dev, ApplySys& s, uint32_t sysId, uint32_t numUnsorted,
                          const std::vector<uint32_t>& ids, const std::vector<uint32_t>& runsum,
                          const std::vector<uint32_t>& active)
{
    const uint32_t nb = uint32_t(ids.size());
    s.numUnsorted = numUnsorted;
    s.ids = ids; s.runsum = runsum; s.active = active;
    // The sorted table and its original-index permutation, as the host builds them
    // (PxgPBDParticleSystemCore.cpp:1785-1788).
    std::vector<uint32_t> perm(nb);
    for (uint32_t i = 0; i < nb; ++i) perm[i] = i;
    for (uint32_t i = 0; i < nb; ++i)
        for (uint32_t j = i + 1; j < nb; ++j)
            if (ids[perm[j]] < ids[perm[i]]) std::swap(perm[i], perm[j]);
    s.sorted.resize(nb); s.origIdx.resize(nb);
    for (uint32_t i = 0; i < nb; ++i) { s.sorted[i] = ids[perm[i]]; s.origIdx[i] = perm[i]; }

    s.uPos.resize(numUnsorted * 4); s.uVel.resize(numUnsorted * 4);
    s.uRest.resize(numUnsorted * 4); s.uPhase.resize(numUnsorted);
    for (uint32_t i = 0; i < numUnsorted; ++i)
    {
        for (uint32_t k = 0; k < 4; ++k)
        {
            s.uPos[i * 4 + k] = 0xBADF00D0u | k;
            s.uVel[i * 4 + k] = 0xBADF00E0u | k;
            s.uRest[i * 4 + k] = 0xBADF00F0u | k;
        }
        s.uPhase[i] = 0xBADF0000u | i;
    }
    s.userPos.resize(nb); s.userVel.resize(nb);
    for (uint32_t b = 0; b < nb; ++b)
    {
        const uint32_t n = active[b] + 4u;
        s.userPos[b].resize(n * 4); s.userVel[b].resize(n * 4);
        for (uint32_t j = 0; j < n; ++j)
            for (uint32_t k = 0; k < 4; ++k)
            {
                // Distinct per (system, buffer, slot, lane, array); every third slot carries
                // the special patterns in a rotating lane order.
                const uint32_t tag = (sysId << 28) | (b << 24) | ((j & 0xFFFFu) << 4) | (k << 1);
                const bool special = (j % 3u) == 0u;
                s.userPos[b][j * 4 + k] = special ? kApplySpecials[(j + k + b) & 7u] : (0x3f000000u ^ tag);
                s.userVel[b][j * 4 + k] = special ? kApplySpecials[(j + 2u * k + b + 3u) & 7u]
                                                  : (0x40000001u ^ tag);
            }
    }

    s.bUPos = makeDev(dev, s.uPos); s.bUVel = makeDev(dev, s.uVel);
    s.bUPhase = makeDev(dev, s.uPhase); s.bURest = makeDev(dev, s.uRest);
    s.bSorted = makeDev(dev, s.sorted); s.bOrig = makeDev(dev, s.origIdx); s.bRunsum = makeDev(dev, s.runsum);
    s.simHost.resize(nb); s.simDev.resize(nb);
    for (uint32_t b = 0; b < nb; ++b)
    {
        s.bUserPos.push_back(makeDev(dev, s.userPos[b]));
        s.bUserVel.push_back(makeDev(dev, s.userVel[b]));
        PxgParticleSimBuffer sb;
        std::memset(&sb, 0, sizeof(sb));
        sb.mNumActiveParticles = active[b];
        sb.mFlags = 0xF00Du;               // the kernel must NOT read the buffer's own flags
        sb.mUniqueId = ids[b];
        s.simHost[b] = sb; s.simDev[b] = sb;
        s.simHost[b].mPositionInvMasses = reinterpret_cast<float4*>(s.userPos[b].data());
        s.simHost[b].mVelocities = reinterpret_cast<float4*>(s.userVel[b].data());
        s.simDev[b].mPositionInvMasses = reinterpret_cast<float4*>(size_t(s.bUserPos[b].addr));
        s.simDev[b].mVelocities = reinterpret_cast<float4*>(size_t(s.bUserVel[b].addr));
    }
    s.bSim = makeDev(dev, s.simDev);

    s.descHost = makeZeroDesc();
    s.descHost.mCommonData.mNumParticleBuffers = nb;
    s.descHost.mCommonData.mNumParticles = numUnsorted;
    s.descDev = s.descHost;
    s.descHost.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(s.uPos.data());
    s.descHost.mUnsortedVelocities = reinterpret_cast<float4*>(s.uVel.data());
    s.descHost.mUnsortedPhaseArray = s.uPhase.data();
    s.descHost.mRestArray = reinterpret_cast<float4*>(s.uRest.data());
    s.descHost.mParticleSimBuffers = s.simHost.data();
    s.descHost.mParticleBufferRunsum = s.runsum.data();
    s.descHost.mParticleBufferSortedUniqueIds = s.sorted.data();
    s.descHost.mParticleBufferSortedUniqueIdsOriginalIndex = s.origIdx.data();
    s.descDev.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s.bUPos.addr));
    s.descDev.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(s.bUVel.addr));
    s.descDev.mUnsortedPhaseArray = reinterpret_cast<PxU32*>(size_t(s.bUPhase.addr));
    s.descDev.mRestArray = reinterpret_cast<float4*>(size_t(s.bURest.addr));
    s.descDev.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(s.bSim.addr));
    s.descDev.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(s.bRunsum.addr));
    s.descDev.mParticleBufferSortedUniqueIds = reinterpret_cast<PxU32*>(size_t(s.bSorted.addr));
    s.descDev.mParticleBufferSortedUniqueIdsOriginalIndex = reinterpret_cast<PxU32*>(size_t(s.bOrig.addr));
}

static void applyFreeSys(PxgSlangComputeDevice* dev, ApplySys& s)
{
    for (DevBuf* b : { &s.bUPos, &s.bUVel, &s.bUPhase, &s.bURest, &s.bSorted, &s.bOrig, &s.bRunsum, &s.bSim })
        dev->destroyBuffer(b->h);
    for (DevBuf& b : s.bUserPos) dev->destroyBuffer(b.h);
    for (DevBuf& b : s.bUserVel) dev->destroyBuffer(b.h);
}

// Word-by-word comparison of one region; prints up to 8 mismatches.
static size_t applyCmpWords(const char* what, const std::vector<uint32_t>& want, const std::vector<uint32_t>& got)
{
    size_t mm = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (i >= got.size() || want[i] != got[i])
        {
            if (mm < 8)
                std::printf("  %s MM word %zu: want 0x%08x got 0x%08x\n", what, i, want[i],
                            i < got.size() ? got[i] : 0u);
            ++mm;
        }
    return mm;
}

template <typename T>
static std::vector<uint32_t> applyWords(const std::vector<T>& v)
{
    std::vector<uint32_t> w(v.size() * sizeof(T) / 4);
    std::memcpy(w.data(), v.data(), w.size() * 4);
    return w;
}

// Every word the launch may or may not touch, against the oracle's host shadows.
static size_t applyCompareAll(PxgSlangComputeDevice* dev, const char* launch, ApplySys* const* sys, uint32_t nSys,
                              const std::vector<PxgParticleSystem>& descDev, const DevBuf& bDesc)
{
    size_t mm = 0;
    char what[96];
    for (uint32_t si = 0; si < nSys; ++si)
    {
        ApplySys& s = *sys[si];
        std::snprintf(what, sizeof(what), "%s sys%u unsortedPos", launch, si);
        mm += applyCmpWords(what, s.uPos, downloadBuffer<uint32_t>(dev, s.bUPos.h, s.uPos.size()));
        std::snprintf(what, sizeof(what), "%s sys%u unsortedVel", launch, si);
        mm += applyCmpWords(what, s.uVel, downloadBuffer<uint32_t>(dev, s.bUVel.h, s.uVel.size()));
        std::snprintf(what, sizeof(what), "%s sys%u unsortedPhase", launch, si);
        mm += applyCmpWords(what, s.uPhase, downloadBuffer<uint32_t>(dev, s.bUPhase.h, s.uPhase.size()));
        std::snprintf(what, sizeof(what), "%s sys%u rest", launch, si);
        mm += applyCmpWords(what, s.uRest, downloadBuffer<uint32_t>(dev, s.bURest.h, s.uRest.size()));
        std::snprintf(what, sizeof(what), "%s sys%u sorted/orig/runsum", launch, si);
        mm += applyCmpWords(what, s.sorted, downloadBuffer<uint32_t>(dev, s.bSorted.h, s.sorted.size()));
        mm += applyCmpWords(what, s.origIdx, downloadBuffer<uint32_t>(dev, s.bOrig.h, s.origIdx.size()));
        mm += applyCmpWords(what, s.runsum, downloadBuffer<uint32_t>(dev, s.bRunsum.h, s.runsum.size()));
        std::snprintf(what, sizeof(what), "%s sys%u simBuffers", launch, si);
        mm += applyCmpWords(what, applyWords(s.simDev),
                            applyWords(downloadBuffer<PxgParticleSimBuffer>(dev, s.bSim.h, s.simDev.size())));
        for (size_t b = 0; b < s.userPos.size(); ++b)
        {
            std::snprintf(what, sizeof(what), "%s sys%u user%zu pos/vel", launch, si, b);
            mm += applyCmpWords(what, s.userPos[b], downloadBuffer<uint32_t>(dev, s.bUserPos[b].h, s.userPos[b].size()));
            mm += applyCmpWords(what, s.userVel[b], downloadBuffer<uint32_t>(dev, s.bUserVel[b].h, s.userVel[b].size()));
        }
    }
    std::snprintf(what, sizeof(what), "%s descriptors", launch);
    mm += applyCmpWords(what, applyWords(descDev),
                        applyWords(downloadBuffer<PxgParticleSystem>(dev, bDesc.h, descDev.size())));
    return mm;
}

PXG_TEST(particleSimApplyBufferData)
{
    PxgSlangComputeDevice* dev = testDevice();
    PXG_CHECK(sizeof(PxParticleBufferFlags) == 4);

    // System 0: ids {7, 2, 12} -> sorted {2, 7, 12}, OriginalIndex {1, 0, 2}.
    // System 1: ids {20, 7}    -> sorted {7, 20},    OriginalIndex {1, 0}.
    ApplySys s0, s1;
    applyBuildSys(dev, s0, 0u, 1600u, { 7u, 2u, 12u }, { 3u, 14u, 40u }, { 9u, 5u, 1500u });
    applyBuildSys(dev, s1, 1u, 64u, { 20u, 7u }, { 5u, 30u }, { 6u, 4u });
    PXG_CHECK(s0.origIdx[0] == 1u && s0.origIdx[1] == 0u && s0.origIdx[2] == 2u);
    PXG_CHECK(s1.origIdx[0] == 1u && s1.origIdx[1] == 0u);

    std::vector<PxgParticleSystem> descHost = { s0.descHost, s1.descHost };
    std::vector<PxgParticleSystem> descDev = { s0.descDev, s1.descDev };
    DevBuf bDesc = makeDev(dev, descDev);

    typedef PxParticleBufferFlag F;
    const PxParticleBufferFlags POSVEL = PxParticleBufferFlags(F::eUPDATE_POSITION) | F::eUPDATE_VELOCITY;
    // Launch 1, grid (ceil(1500/1024) = 2, 4, 1).
    std::vector<PxGpuParticleBufferIndexPair> pairs1 = {
        { 0u, 12u },   // [0] POS|VEL                 named by by=1 -> system 0 buffer 2
        { 1u, 20u },   // [1] POS|VEL                 DECOY, never named
        { 0u, 2u },    // [2] POS|VEL                 DECOY, never named
        { 0u, 2u },    // [3] POS|PHASE|RESTPOSITION  named by by=0 -> only POS may land (buffer 1)
        { 0u, 7u },    // [4] PHASE|RESTPOSITION      named by by=3 -> nothing may land
        { 1u, 7u } };  // [5] POS|VEL                 named by by=2 -> system 1 buffer 1
    std::vector<PxParticleBufferFlags> flags1 = {
        POSVEL, POSVEL, POSVEL,
        PxParticleBufferFlags(F::eUPDATE_POSITION) | F::eUPDATE_PHASE | F::eUPDATE_RESTPOSITION,
        PxParticleBufferFlags(F::eUPDATE_PHASE) | F::eUPDATE_RESTPOSITION,
        POSVEL };
    std::vector<uint32_t> indices1 = { 3u, 0u, 5u, 4u };

    // The self-check: every wrong spelling of the oracle must change launch 1.
    {
        const ApplyVariant variants[5] = { AV_FLAGS_BY_Y, AV_LOWER_BOUND, AV_NO_OFFSET, AV_PAIR_BY_Y, AV_SEARCH_SYS0 };
        const char* names[5] = { "(a) flags[by]", "(b) lower bound", "(c) offset dropped", "(d) pairs[by]",
                                 "(e) search in system 0" };
        const std::vector<uint32_t> p0 = s0.uPos, v0 = s0.uVel, p1 = s1.uPos, v1 = s1.uVel;
        // In place: the host shadow descriptors point at these vectors' storage.
        auto restore = [&]() {
            std::copy(p0.begin(), p0.end(), s0.uPos.begin()); std::copy(v0.begin(), v0.end(), s0.uVel.begin());
            std::copy(p1.begin(), p1.end(), s1.uPos.begin()); std::copy(v1.begin(), v1.end(), s1.uVel.begin());
        };
        refApplyParticleBufferData(descHost.data(), indices1.data(), pairs1.data(),
                                   reinterpret_cast<const PxParticleBufferFlag*>(flags1.data()), 2u, 4u, AV_CUDA);
        const std::vector<uint32_t> rp0 = s0.uPos, rv0 = s0.uVel, rp1 = s1.uPos, rv1 = s1.uVel;
        for (int vi = 0; vi < 5; ++vi)
        {
            restore();
            refApplyParticleBufferData(descHost.data(), indices1.data(), pairs1.data(),
                                       reinterpret_cast<const PxParticleBufferFlag*>(flags1.data()), 2u, 4u,
                                       variants[vi]);
            size_t diff = 0;
            for (size_t i = 0; i < rp0.size(); ++i) diff += (rp0[i] != s0.uPos[i]) + (rv0[i] != s0.uVel[i]);
            for (size_t i = 0; i < rp1.size(); ++i) diff += (rp1[i] != s1.uPos[i]) + (rv1[i] != s1.uVel[i]);
            PXG_CHECK_MSG(diff > 0, "self-check %s changes launch 1 (%zu words)", names[vi], diff);
        }
        restore();
    }

    Kernel k("particleSim.spv", "applyParticleBufferDataDEPRECATED", {}, sizeof(ApplyPush));
    ApplySys* both[2] = { &s0, &s1 };

    // ---- launch 1 ----
    DevBuf bIdx1 = makeDev(dev, indices1), bPairs1 = makeDev(dev, pairs1), bFlags1 = makeDev(dev, flags1);
    k.dispatch(2u, {}, ApplyPush{ bDesc.addr, bIdx1.addr, bPairs1.addr, bFlags1.addr }, /*gridY*/ 4u, 1u);
    dev->flush();
    refApplyParticleBufferData(descHost.data(), indices1.data(), pairs1.data(),
                               reinterpret_cast<const PxParticleBufferFlag*>(flags1.data()), 2u, 4u, AV_CUDA);
    size_t mm = applyCompareAll(dev, "L1", both, 2u, descDev, bDesc);
    mm += applyCmpWords("L1 indices", indices1, downloadBuffer<uint32_t>(dev, bIdx1.h, indices1.size()));
    mm += applyCmpWords("L1 pairs", applyWords(pairs1), downloadBuffer<uint32_t>(dev, bPairs1.h, pairs1.size() * 2));
    mm += applyCmpWords("L1 flags", applyWords(flags1), downloadBuffer<uint32_t>(dev, bFlags1.h, flags1.size()));
    PXG_CHECK_MSG(mm == 0, "applyParticleBufferDataDEPRECATED launch 1 bit-exact vs the CUDA oracle (%zu mismatches)", mm);

    // ---- launch 2: the unregistered id 9 resolves to its predecessor 7 (buffer 0), VEL ----
    std::vector<PxGpuParticleBufferIndexPair> pairs2 = { { 0u, 9u } };
    std::vector<PxParticleBufferFlags> flags2 = { PxParticleBufferFlags(F::eUPDATE_VELOCITY) };
    std::vector<uint32_t> indices2 = { 0u };
    PXG_CHECK(refFindBufferIndexFromUniqueId(descHost[0], 9u, AV_CUDA) == 0u);
    DevBuf bIdx2 = makeDev(dev, indices2), bPairs2 = makeDev(dev, pairs2), bFlags2 = makeDev(dev, flags2);
    k.dispatch(1u, {}, ApplyPush{ bDesc.addr, bIdx2.addr, bPairs2.addr, bFlags2.addr }, 1u, 1u);
    dev->flush();
    refApplyParticleBufferData(descHost.data(), indices2.data(), pairs2.data(),
                               reinterpret_cast<const PxParticleBufferFlag*>(flags2.data()), 1u, 1u, AV_CUDA);
    mm = applyCompareAll(dev, "L2", both, 2u, descDev, bDesc);
    PXG_CHECK_MSG(mm == 0, "applyParticleBufferDataDEPRECATED launch 2 (unregistered id) bit-exact (%zu mismatches)", mm);

    // ---- launch 3: a system with mNumParticleBuffers == 0 -> the early return, buffer 0 ----
    std::vector<PxgParticleSystem> desc3Host = { s1.descHost }, desc3Dev = { s1.descDev };
    desc3Host[0].mCommonData.mNumParticleBuffers = 0u;
    desc3Dev[0].mCommonData.mNumParticleBuffers = 0u;
    // The original-index table of this system starts ONE WORD INTO a buffer whose word
    // before it is 1: the early return never reads the table, but a search without it
    // reads OriginalIndex[r - 1] at r == 0, i.e. that word (or, with a 64-bit index, far
    // out of bounds) -- either way not the buffer 0 the CUDA returns.
    std::vector<uint32_t> orig3 = { 1u, 0u, 1u };
    DevBuf bOrig3 = makeDev(dev, orig3);
    desc3Host[0].mParticleBufferSortedUniqueIdsOriginalIndex = orig3.data() + 1;
    desc3Dev[0].mParticleBufferSortedUniqueIdsOriginalIndex = reinterpret_cast<PxU32*>(size_t(bOrig3.addr + 4u));
    DevBuf bDesc3 = makeDev(dev, desc3Dev);
    std::vector<PxGpuParticleBufferIndexPair> pairs3 = { { 0u, 7u } };
    std::vector<PxParticleBufferFlags> flags3 = { POSVEL };
    std::vector<uint32_t> indices3 = { 0u };
    DevBuf bIdx3 = makeDev(dev, indices3), bPairs3 = makeDev(dev, pairs3), bFlags3 = makeDev(dev, flags3);
    k.dispatch(1u, {}, ApplyPush{ bDesc3.addr, bIdx3.addr, bPairs3.addr, bFlags3.addr }, 1u, 1u);
    dev->flush();
    refApplyParticleBufferData(desc3Host.data(), indices3.data(), pairs3.data(),
                               reinterpret_cast<const PxParticleBufferFlag*>(flags3.data()), 1u, 1u, AV_CUDA);
    ApplySys* one[1] = { &s1 };
    mm = applyCompareAll(dev, "L3", one, 1u, desc3Dev, bDesc3);
    mm += applyCompareAll(dev, "L3", both, 1u, descDev, bDesc);   // system 0 untouched
    mm += applyCmpWords("L3 orig3", orig3, downloadBuffer<uint32_t>(dev, bOrig3.h, orig3.size()));
    PXG_CHECK_MSG(mm == 0, "applyParticleBufferDataDEPRECATED launch 3 (empty table) bit-exact (%zu mismatches)", mm);

    for (DevBuf* b : { &bDesc, &bIdx1, &bPairs1, &bFlags1, &bIdx2, &bPairs2, &bFlags2, &bDesc3, &bIdx3, &bPairs3,
                       &bFlags3, &bOrig3 })
        dev->destroyBuffer(b->h);
    applyFreeSys(dev, s0);
    applyFreeSys(dev, s1);
}

// ---------------------------------------------------------------------------
// ps_updateUserBufferLaunch — copy unsorted state to user buffer
// ---------------------------------------------------------------------------
PXG_TEST(particleSimUpdateBuffer)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const uint32_t numActive = 6;
    const uint32_t bufOffset = 2; // runsum[0]

    std::vector<F4> unsortedPos(N), unsortedVel(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        unsortedPos[i] = { float(i) + 0.25f, float(i) * 3.f, -float(i), 1.f };
        unsortedVel[i] = { 0.7f * float(i), 0.3f, -0.1f * float(i), 0.f };
    }
    std::vector<F4> userPos(numActive, { -9.f, -9.f, -9.f, -9.f });
    std::vector<F4> userVel(numActive, { -8.f, -8.f, -8.f, -8.f });

    DevBuf bUPos = makeDev(dev, unsortedPos);
    DevBuf bUVel = makeDev(dev, unsortedVel);
    DevBuf bUserPos = makeDev(dev, userPos);
    DevBuf bUserVel = makeDev(dev, userVel);
    std::vector<uint32_t> runsum = { bufOffset };
    DevBuf bRunsum = makeDev(dev, runsum);

    // one real PxgParticleSimBuffer with mFlags pre-set to a sentinel.
    PxgParticleSimBuffer simBuf;
    std::memset(&simBuf, 0, sizeof(simBuf));
    simBuf.mPositionInvMasses = reinterpret_cast<float4*>(size_t(bUserPos.addr));
    simBuf.mVelocities = reinterpret_cast<float4*>(size_t(bUserVel.addr));
    simBuf.mNumActiveParticles = numActive;
    simBuf.mFlags = 0xdeadbeefu;
    std::vector<PxgParticleSimBuffer> simArr = { simBuf };
    DevBuf bSim = makeDev(dev, simArr);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bUPos.addr));
    ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(bUVel.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSim.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
    ps.mCommonData.mNumParticleBuffers = 1;
    ps.mCommonData.mNumParticles = N;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct UpdPush { uint64_t particleSystems, activeParticleSystems; uint32_t count; };
    UpdPush push = { bDesc.addr, bActive.addr, 42u };

    Kernel k("particleSim.spv", "ps_updateUserBufferLaunch", {}, sizeof(UpdPush));
    // grid = (numBlocks, mMaxBuffersPerSystem, nbActiveSystems) = (1, 1, 1)
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto gUserPos = downloadBuffer<F4>(dev, bUserPos.h, numActive);
    const auto gUserVel = downloadBuffer<F4>(dev, bUserVel.h, numActive);
    const auto gSim = downloadBuffer<PxgParticleSimBuffer>(dev, bSim.h, 1);

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8) std::printf("  %s MM %u\n", what, i);
            ++mm;
        }
    };
    for (uint32_t j = 0; j < numActive; ++j)
    {
        cmp4(unsortedPos[bufOffset + j], gUserPos[j], "userPos", j);
        cmp4(unsortedVel[bufOffset + j], gUserVel[j], "userVel", j);
    }
    PXG_CHECK_MSG(gSim[0].mFlags == 0u, "ps_updateUserBufferLaunch cleared mFlags (got 0x%x)", gSim[0].mFlags);
    PXG_CHECK_MSG(mm == 0, "ps_updateUserBufferLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bUPos.h); dev->destroyBuffer(bUVel.h); dev->destroyBuffer(bUserPos.h);
    dev->destroyBuffer(bUserVel.h); dev->destroyBuffer(bRunsum.h); dev->destroyBuffer(bSim.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

}  // namespace
