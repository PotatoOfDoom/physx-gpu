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
//   particleSimIngestGuard / particleSimUpdateBufferGuard — the two buffer-copy
//                             kernels on the engine grid, every array before a
//                             PROT_NONE guard page (P9.97)
//   particleSimPreIntegrate_p99TwoSystems / particleSimIntegrate_p99TwoSystems —
//                             P9.99b: the same kernels over a pool of three with
//                             act = {2, 0} (per-system material map, cap, lock)

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
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

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
// P9.99b -- the TWO-SYSTEM twin of particleSimPreIntegrate.
//
// particleSimPreIntegrate drives one system at act = {0}, so a kernel that took the
// system from anywhere but act[blockIdx.y], or read a per-system parameter out of
// the wrong descriptor, was invisible there: the P9.99b force-system-0 mutants of
// this kernel (`particleId = act[0]`, `systems[0].mPhaseGroupToMaterialHandle`)
// survived every fixture.  The frame every p99 twin shares:
//   * a pool of THREE descriptors, act = {2, 0}, gridY = 2 -- not a permutation, so
//     a kernel that indexes the pool by blockIdx.y hits the DECOY at slot 1 instead
//     of permuting the work;
//   * systems 2 and 0 differ in everything the kernel reads: N, the data, and the
//     material, which each system reaches through its OWN phase->material map into
//     ONE shared material table (as in the engine);
//   * the decoy has a LARGER count and sentinel data in every array, and all of it
//     must survive bit for bit.
// The oracle is particleSimPreIntegrate's CPU transcription, per system.
// ---------------------------------------------------------------------------
PXG_TEST(particleSimPreIntegrate_p99TwoSystems)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NS[3] = { 8u, 21u, 13u };   // slot 1 is the decoy, with the largest count
    // material: friction, damping, adhesion, gravityScale, adhesionRadiusScale (stride 20)
    const std::vector<float> material = {
        0.5f, 0.25f, 0.0f, 2.0f, 0.0f,    // material 0: system 0
        0.1f, 0.6f,  0.0f, 0.5f, 0.0f,    // material 1: system 2
        0.0f, 0.9f,  0.0f, 3.0f, 0.0f };  // material 2: the decoy
    const uint32_t materialStride = 20u;
    const uint16_t matOf[3] = { 0u, 2u, 1u };
    DevBuf bMat = makeDev(dev, material);

    struct Sys
    {
        std::vector<F4> pos, vel, ori, accum;
        std::vector<uint32_t> phase, counts;
        std::vector<float> forces;
        std::vector<uint16_t> p2m;
        DevBuf bPos, bVel, bPhase, bOri, bAccum, bCounts, bForces, bP2M;
    };
    Sys s[3];
    std::vector<PxgParticleSystem> descArr(3);
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t N = NS[k];
        const float fk = float(k);
        s[k].pos.resize(N); s[k].vel.resize(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            const float w = (i == 3) ? 0.0f : (k == 2 ? 0.5f : 1.0f);   // particle 3: infinite mass
            s[k].pos[i] = { float(i) * 1.0f + 10.0f * fk, float(i) * 2.0f + 0.5f - fk, -float(i) * 0.5f + 0.25f * fk, w };
            s[k].vel[i] = { 0.1f * float(i) - 0.3f * fk, -0.2f * float(i) + 0.7f * fk, 0.05f * float(i), 0.0f };
        }
        s[k].phase.assign(N, 0u);
        s[k].ori.assign(N, { -1.f, -1.f, -1.f, -1.f });
        s[k].accum.assign(N, { 9.f, 9.f, 9.f, 9.f });
        s[k].counts.assign(N, 7u);
        s[k].forces.assign(2 * 12 * N, 5.0f);
        s[k].p2m = { matOf[k], matOf[k] };
        s[k].bPos = makeDev(dev, s[k].pos);
        s[k].bVel = makeDev(dev, s[k].vel);
        s[k].bPhase = makeDev(dev, s[k].phase);
        s[k].bOri = makeDev(dev, s[k].ori);
        s[k].bAccum = makeDev(dev, s[k].accum);
        s[k].bCounts = makeDev(dev, s[k].counts);
        s[k].bForces = makeDev(dev, s[k].forces);
        s[k].bP2M = makeDev(dev, s[k].p2m);

        PxgParticleSystem ps = makeZeroDesc();
        ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s[k].bPos.addr));
        ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(s[k].bVel.addr));
        ps.mUnsortedPhaseArray = reinterpret_cast<PxU32*>(size_t(s[k].bPhase.addr));
        ps.mOriginPos_InvMass = reinterpret_cast<float4*>(size_t(s[k].bOri.addr));
        ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(s[k].bAccum.addr));
        ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(s[k].bCounts.addr));
        ps.mOneWayForces = reinterpret_cast<float2*>(size_t(s[k].bForces.addr));
        ps.mParticleMaterials = reinterpret_cast<PxsParticleMaterialData*>(size_t(bMat.addr));
        ps.mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(s[k].bP2M.addr));
        ps.mParticleMaterialStride = materialStride;
        ps.mCommonData.mNumParticles = N;
        descArr[k] = ps;
    }
    DevBuf bDesc = makeDev(dev, descArr);
    const std::vector<uint32_t> active = { 2u, 0u };
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
    {
        Kernel k("particleSim.spv", "ps_preIntegrateLaunch", {}, sizeof(PrePush));
        k.dispatch(1u, {}, push, /*gridY = nbActive*/ 2u, /*gridZ*/ 1u);
    }
    dev->flush();

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t k, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8)
                std::printf("  sys%u %s MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", k, what, i,
                            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    };
    const float3 g = { 0.0f, -9.81f, 0.0f };
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t N = NS[k];
        const bool live = k != 1u;   // the decoy is never named by act
        const float matDamping = material[matOf[k] * 5u + 1u], matGravityScale = material[matOf[k] * 5u + 3u];
        std::vector<F4> refPos = s[k].pos, refVel = s[k].vel, refOri = s[k].ori, refAccum = s[k].accum;
        std::vector<uint32_t> refCounts = s[k].counts;
        std::vector<float> refForces = s[k].forces;
        if (live)
            for (uint32_t i = 0; i < N; ++i)
            {
                const F4 tPos = s[k].pos[i];
                refOri[i] = tPos;
                const float damping = 1.0f - std::fmin(1.0f, matDamping * dt);
                const float tgx = g.x * matGravityScale, tgy = g.y * matGravityScale, tgz = g.z * matGravityScale;
                float vx = 0.f, vy = 0.f, vz = 0.f;
                if (tPos.w != 0.0f)
                {
                    const F4 tv = s[k].vel[i];
                    vx = (tv.x + tgx * dt) * damping;
                    vy = (tv.y + tgy * dt) * damping;
                    vz = (tv.z + tgz * dt) * damping;
                    const float fwd = isTGS ? FWD_TGS : FWD_PGS;
                    refPos[i] = { tPos.x + vx * dt * fwd, tPos.y + vy * dt * fwd, tPos.z + vz * dt * fwd, tPos.w };
                }
                if (!extForces)
                    refVel[i] = { vx, vy, vz, tPos.w };
                refAccum[i] = { 0.f, 0.f, 0.f, 0.f };
                refCounts[i] = 0u;
                for (uint32_t j = i; j < 12 * N; j += N) { refForces[2 * j] = 0.f; refForces[2 * j + 1] = 0.f; }
            }
        const auto gPos = downloadBuffer<F4>(dev, s[k].bPos.h, N);
        const auto gVel = downloadBuffer<F4>(dev, s[k].bVel.h, N);
        const auto gOri = downloadBuffer<F4>(dev, s[k].bOri.h, N);
        const auto gAccum = downloadBuffer<F4>(dev, s[k].bAccum.h, N);
        const auto gCounts = downloadBuffer<uint32_t>(dev, s[k].bCounts.h, N);
        const auto gForces = downloadBuffer<float>(dev, s[k].bForces.h, 2 * 12 * N);
        for (uint32_t i = 0; i < N; ++i)
        {
            cmp4(refPos[i], gPos[i], "pos", k, i);
            cmp4(refVel[i], gVel[i], "vel", k, i);
            cmp4(refOri[i], gOri[i], "ori", k, i);
            cmp4(refAccum[i], gAccum[i], "accum", k, i);
            if (gCounts[i] != refCounts[i]) { ++mm; std::printf("  sys%u count MM %u\n", k, i); }
        }
        for (uint32_t j = 0; j < 2 * 12 * N; ++j)
            if (f2u(refForces[j]) != f2u(gForces[j])) { if (mm < 12) std::printf("  sys%u force MM %u\n", k, j); ++mm; }
    }
    PXG_CHECK_MSG(mm == 0, "ps_preIntegrateLaunch, systems 2 and 0 via act {2,0}, decoy untouched: bit-exact "
                  "vs CPU (%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k)
    {
        dev->destroyBuffer(s[k].bPos.h); dev->destroyBuffer(s[k].bVel.h); dev->destroyBuffer(s[k].bPhase.h);
        dev->destroyBuffer(s[k].bOri.h); dev->destroyBuffer(s[k].bAccum.h); dev->destroyBuffer(s[k].bCounts.h);
        dev->destroyBuffer(s[k].bForces.h); dev->destroyBuffer(s[k].bP2M.h);
    }
    dev->destroyBuffer(bMat.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
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
// P9.99b -- the TWO-SYSTEM twin of particleSimIntegrate (frame: see
// particleSimPreIntegrate_p99TwoSystems).  What differs per system is what the
// kernel reads per system:
//   * mData.mMaxVelocity: 3 for system 0, 5 for system 2.  System 2's unlocked
//     particles move at 3.2-4.4, under ITS cap and over system 0's, so reading system
//     0's cap clamps them (and only then is rsqrt reached for a live particle -- the
//     oracle never needs it);
//   * mData.mLockFlags: none for system 0, eLOCK_Z for system 2 (z reset to the
//     origin, v.z = 0);
//   * N, the permutation and every array.
// The oracle is particleSimIntegrate's CPU transcription plus the lock branch.
// ---------------------------------------------------------------------------
PXG_TEST(particleSimIntegrate_p99TwoSystems)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NS[3] = { 8u, 20u, 11u };          // slot 1 is the decoy, with the largest count
    const float MAXV[3] = { 3.0f, 2.0f, 5.0f };
    const uint32_t LOCK[3] = { 0u, 0u, 4u };          // PxParticleLockFlag::eLOCK_Z = 4
    const float eps = 1e-6f;
    const float dt = 1.0f / 60.0f;

    struct Sys
    {
        std::vector<uint32_t> mapping;
        std::vector<F4> sortedDeltaP, sortedOrigin, sortedPos, sortedVel, delta, position, originalVels;
        DevBuf bPos, bNewPos, bOrig, bDeltaP, bOrigVel, bNewVel, bDelta, bMap;
    };
    Sys s[3];
    std::vector<PxgParticleSystem> descArr(3);
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t N = NS[k];
        const float fk = float(k);
        s[k].mapping.resize(N);
        for (uint32_t i = 0; i < N; ++i) s[k].mapping[i] = (i * 3u + k) % N;   // a permutation (gcd(3, N) = 1)
        s[k].sortedDeltaP.resize(N); s[k].sortedOrigin.resize(N); s[k].sortedPos.resize(N);
        s[k].sortedVel.resize(N); s[k].delta.resize(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            const float fi = float(i);
            // particle 0: zero delta -> locked; particle 5: newPos.w == 0 -> skipped.
            s[k].sortedDeltaP[i] = (i == 0) ? F4{ 0, 0, 0, 0 } : F4{ 0.1f, 0.2f + 0.1f * fk, 0.3f, 0.f };
            s[k].sortedOrigin[i] = { fi + fk, fi + 1.f, fi + 2.f - fk, 1.f };
            const float w = (i == 5) ? 0.0f : (k == 2 ? 0.5f : 1.0f);
            s[k].sortedPos[i] = { 10.f + fi, 20.f + fi - fk, 30.f + fi + 0.5f * fk, w };
            if (k == 2)
                s[k].sortedVel[i] = { 3.0f + 0.1f * fi, -1.0f - 0.05f * fi, 0.2f, 0.f };   // |v| in (3.2, 4.4)
            else
                s[k].sortedVel[i] = { 0.3f * fi, -0.2f * fi, 0.2f + 0.1f * fk, 0.f };    // |v| < 2.6
            s[k].delta[i] = { 0.01f * fi, 0.02f * fi, 0.03f * fi, fi + 1.f };
        }
        s[k].position.assign(N, { -100.f, -100.f, -100.f, -100.f });
        s[k].originalVels.assign(N, { -50.f, -50.f, -50.f, -50.f });
        s[k].bPos = makeDev(dev, s[k].position);
        s[k].bNewPos = makeDev(dev, s[k].sortedPos);
        s[k].bOrig = makeDev(dev, s[k].sortedOrigin);
        s[k].bDeltaP = makeDev(dev, s[k].sortedDeltaP);
        s[k].bOrigVel = makeDev(dev, s[k].originalVels);
        s[k].bNewVel = makeDev(dev, s[k].sortedVel);
        s[k].bDelta = makeDev(dev, s[k].delta);
        s[k].bMap = makeDev(dev, s[k].mapping);

        PxgParticleSystem ps = makeZeroDesc();
        ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s[k].bPos.addr));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s[k].bNewPos.addr));
        ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(s[k].bOrig.addr));
        ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(s[k].bDeltaP.addr));
        ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(s[k].bOrigVel.addr));
        ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(s[k].bNewVel.addr));
        ps.mDelta = reinterpret_cast<float4*>(size_t(s[k].bDelta.addr));
        ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(s[k].bMap.addr));
        ps.mData.mMaxVelocity = MAXV[k];
        ps.mData.mLockFlags = PxU16(LOCK[k]);
        ps.mCommonData.mNumParticles = N;
        descArr[k] = ps;
    }
    DevBuf bDesc = makeDev(dev, descArr);
    const std::vector<uint32_t> active = { 2u, 0u };
    DevBuf bActive = makeDev(dev, active);

    struct IntPush { uint64_t particleSystems, activeParticleSystems; float dt, eps; };
    IntPush push = { bDesc.addr, bActive.addr, dt, eps };
    {
        Kernel k("particleSim.spv", "ps_integrateLaunch", {}, sizeof(IntPush));
        k.dispatch(1u, {}, push, /*gridY = nbActive*/ 2u, 1u);
    }
    dev->flush();

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t k, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8)
                std::printf("  sys%u %s MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", k, what, i,
                            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    };
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t N = NS[k];
        const Sys& c = s[k];
        std::vector<F4> refPos = c.position, refVel = c.originalVels, refDelta = c.delta;
        if (k != 1u)   // the decoy is never named by act
            for (uint32_t i = 0; i < N; ++i)
            {
                const uint32_t sortedIndex = c.mapping[i];
                const float dsq = c.sortedDeltaP[i].x * c.sortedDeltaP[i].x + c.sortedDeltaP[i].y * c.sortedDeltaP[i].y +
                                  c.sortedDeltaP[i].z * c.sortedDeltaP[i].z;
                const bool locked = dsq <= eps;
                const F4 origPos4 = c.sortedOrigin[i];
                F4 pos4 = locked ? origPos4 : c.sortedPos[i];
                if (pos4.w > 0.0f)
                {
                    const float mv = locked ? 0.0f : MAXV[k];
                    const F4 nv = c.sortedVel[i];
                    const F4 j = c.delta[i];
                    const float scale = 1.0f / std::fmax(j.w, 1.0f);
                    float vx = nv.x + j.x * scale, vy = nv.y + j.y * scale, vz = nv.z + j.z * scale;
                    const float magSq = vx * vx + vy * vy + vz * vz;
                    if (magSq > mv * mv)
                    {
                        // Only the locked (mv == 0) particle -> ratio 0 -> zero (rsqrt-independent).
                        const float ratio = mv * (1.0f / std::sqrt(magSq));
                        vx *= ratio; vy *= ratio; vz *= ratio;
                    }
                    if (LOCK[k] & 4u) { pos4.z = origPos4.z; vz = 0.0f; }
                    refPos[sortedIndex] = pos4;
                    refVel[sortedIndex] = { vx, vy, vz, 0.0f };
                    refDelta[i] = { 0.f, 0.f, 0.f, 0.f };
                }
            }
        const auto gPos = downloadBuffer<F4>(dev, c.bPos.h, N);
        const auto gVel = downloadBuffer<F4>(dev, c.bOrigVel.h, N);
        const auto gDelta = downloadBuffer<F4>(dev, c.bDelta.h, N);
        for (uint32_t i = 0; i < N; ++i)
        {
            cmp4(refPos[i], gPos[i], "pos", k, i);
            cmp4(refVel[i], gVel[i], "vel", k, i);
            cmp4(refDelta[i], gDelta[i], "delta", k, i);
        }
    }
    // System 2's live particles really sit between the two caps (else the max-velocity
    // read is not discriminated).
    uint32_t between = 0;
    for (uint32_t i = 1; i < NS[2]; ++i)
    {
        const F4 nv = s[2].sortedVel[i], j = s[2].delta[i];
        const float sc = 1.0f / std::fmax(j.w, 1.0f);
        const float vx = nv.x + j.x * sc, vy = nv.y + j.y * sc, vz = nv.z + j.z * sc;
        const float m2 = vx * vx + vy * vy + vz * vz;
        if (s[2].sortedPos[i].w > 0.0f && m2 > MAXV[0] * MAXV[0] && m2 < MAXV[2] * MAXV[2]) ++between;
    }
    PXG_CHECK_MSG(between == NS[2] - 2u, "system 2: %u of %u live particles between the two caps", between, NS[2] - 2u);
    PXG_CHECK_MSG(mm == 0, "ps_integrateLaunch, systems 2 and 0 via act {2,0} (caps 5/3, lock Z/none), decoy "
                  "untouched: bit-exact vs CPU (%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k)
    {
        dev->destroyBuffer(s[k].bPos.h); dev->destroyBuffer(s[k].bNewPos.h); dev->destroyBuffer(s[k].bOrig.h);
        dev->destroyBuffer(s[k].bDeltaP.h); dev->destroyBuffer(s[k].bOrigVel.h); dev->destroyBuffer(s[k].bNewVel.h);
        dev->destroyBuffer(s[k].bDelta.h); dev->destroyBuffer(s[k].bMap.h);
    }
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

// (PhysX 5.12 removed applyParticleBufferDataDEPRECATED; its particleSimApplyBufferData
// fixture was deleted with the kernel.)

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

// ---------------------------------------------------------------------------
// P9.97 -- particleSimIngestGuard / particleSimUpdateBufferGuard: the two buffer-copy
// kernels on the ENGINE's grid, with every array ending at a PROT_NONE guard page.
//
// The engine launches both kernels on (ceil(mMaxParticlesPerBuffer/1024),
// mMaxBuffersPerSystem, nbActive) (PxgParticleSystemCore.cpp:753/764, :2520/2532):
// both maxima are over every buffer of every system and only grow, so lanes run past
// a buffer's user arrays (ingest) and past the unsorted arrays (egress, offset + x),
// and workgroups run past a system's buffer count.  The CUDA guards every such
// access; the port adds value-identical clamps (UA:clamp, UA:restSel, UB:clamp) for
// back ends that hoist a guarded load.  On llvmpipe none of the unclamped reads is
// issued (masked gathers; the y-guarded body is a real branch), so the clamps
// cannot be made visible here -- what this fixture pins instead:
//   * every word of every array, decoy system, descriptors, records, runsum and active
//     list included, against a CUDA-literal host oracle (particlesystem.cu:218-301);
//   * with the guard arena, a load moved out of its guard FAULTS deterministically
//     (the arrays end on the last byte before a PROT_NONE page -- no layout lottery);
//   * the CUDA's own guards: y at nb == 0 (L2/E2: an active system with NO buffers and
//     NULL record/runsum pointers, standing for the engine's null, stale or
//     uninitialised nb == 0 pointers -- PxgSimulationController.cpp:1148-1150), the
//     lane guard, the null-rest test (S0 b0 has REST set and a NULL rest pointer,
//     as every rigid buffer has in the engine, PxgPBDParticleSystemCore.cpp:1411/1519).
//
// The arena hands the shader plain mmap memory by its host VA.  That is valid only
// where a buffer device address IS the host VA -- llvmpipe (P8.46a), measured here
// through a host-visible buffer, never assumed -- so it is used only when that
// identity holds AND the device is llvmpipe; elsewhere every array is a host-visible
// device buffer and the fixture is bitwise-only (a hoisted load then reads slack
// instead of faulting).  Neither mode is asserted.
//
// Layout (the engine formula gives grid (3, 3, 2)):
//   S0 active (z = 1): b0 n=3 flags 0xF rest NULL; b1 n=2100 flags 0xD (no VEL) rest valid
//   S1 DECOY, never active: b0..b2 n = 5, 6, 7 flags 0xF -- makes mMaxBuffersPerSystem 3
//   S2 active (z = 0): b0 n=700 flags 0x2 rest NULL; b1 n=0 (capacity 1) flags 0xF rest
//      valid -- or, for L2/E2, NO buffers at all (null record and runsum pointers)
// active list {2, 0}: not a permutation of all systems, so dropping the indirection shows.
// The unsorted arrays are sized to the sum of ACTIVE counts, tighter than the engine's
// (at least the sum of capacities, grow-only): every CUDA store index is below it.
// ---------------------------------------------------------------------------
struct GuardArr
{
    uint8_t* p = nullptr;      // host view of the first byte
    uint64_t dev = 0;          // device address (0 for an absent array)
    size_t bytes = 0;
    void* map = nullptr;       // guard mode: the mmap (array + guard page)
    size_t mapBytes = 0;
    PxgBufferHandle h = 0;     // otherwise: the host-visible buffer
};

// Does this device address a host-visible buffer AT its host VA?
static bool guardIdentity(PxgSlangComputeDevice* dev)
{
    void* m = nullptr;
    const PxgBufferHandle h = dev->createHostVisibleBuffer(256, &m);
    const bool id = h != 0 && m != nullptr && dev->getBufferDeviceAddress(h) == uint64_t(uintptr_t(m));
    if (h)
        dev->destroyBuffer(h);
    return id;
}

static GuardArr guardAlloc(PxgSlangComputeDevice* dev, bool guard, size_t bytes)
{
    GuardArr a;
    a.bytes = bytes;
    if (bytes == 0)
        return a;                                   // an absent (NULL) array
#if defined(__linux__)
    if (guard)
    {
        const size_t pg = size_t(sysconf(_SC_PAGESIZE));
        const size_t np = (bytes + pg - 1) / pg;
        void* m = mmap(nullptr, (np + 1) * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED)
            return a;
        uint8_t* base = static_cast<uint8_t*>(m);
        if (mprotect(base + np * pg, pg, PROT_NONE) != 0)
        {
            munmap(m, (np + 1) * pg);
            return a;
        }
        std::memset(base, 0xA5, np * pg);            // poison in front of the array
        a.map = m;
        a.mapBytes = (np + 1) * pg;
        a.p = base + np * pg - bytes;               // the LAST byte is the last before the guard
        a.dev = uint64_t(uintptr_t(a.p));
        return a;
    }
#else
    (void)guard;
#endif
    void* m = nullptr;
    a.h = dev->createHostVisibleBuffer(bytes, &m);
    a.p = static_cast<uint8_t*>(m);
    a.dev = a.h ? dev->getBufferDeviceAddress(a.h) : 0;
    return a;
}

static void guardFree(PxgSlangComputeDevice* dev, GuardArr& a)
{
#if defined(__linux__)
    if (a.map)
        munmap(a.map, a.mapBytes);
#endif
    if (a.h)
        dev->destroyBuffer(a.h);
    a = GuardArr();
}

static uint32_t guardPattern(uint32_t salt, uint32_t i)
{
    // every third word a P9.95b special value: -0, +0, quiet/signalling NaN payloads,
    // +-inf, the smallest denormal, the largest negative denormal
    static const uint32_t special[] = { 0x80000000u, 0x00000000u, 0x7fc12345u, 0x7f812345u,
                                        0x7f800000u, 0xff800000u, 0x00000001u, 0x807fffffu };
    const uint32_t h = (i * 2654435761u) ^ (salt * 40503u + 0x9e37u);
    return (i % 3u == 0u) ? special[(h >> 7) % 8u] : h;
}

struct GuardBuf { uint32_t active, flags; bool rest; GuardArr pos, vel, restA, phaseA; };
struct GuardSys { std::vector<GuardBuf> bufs; GuardArr upos, uvel, uphase, urest, simbuf, runsum; uint32_t total = 0; };

// Oracle variants: the CUDA (None) and four deliberately wrong ones, each of which the
// layout must be able to tell apart (checked on the host before the launch).
enum class GuardVariant { None, SysDirect, FlagsBuf0, OffsetDrop, NoFlagsClear };

struct GuardWorld
{
    PxgSlangComputeDevice* dev = nullptr;
    bool guard = false, ok = true;
    std::vector<GuardSys> sys;
    std::vector<uint32_t> active;
    GuardArr desc, act;
    std::vector<GuardArr*> all;          // every array, in a fixed order
    std::vector<std::string> names;

    GuardArr alloc(size_t bytes)
    {
        GuardArr a = guardAlloc(dev, guard, bytes);
        if (bytes && !a.p)
            ok = false;
        return a;
    }

    void build(bool emptyS2, bool egress)
    {
        sys.resize(3);
        sys[0].bufs = { { 3u, 0xFu, false, {}, {}, {}, {} }, { 2100u, 0xDu, true, {}, {}, {}, {} } };
        sys[1].bufs = { { 5u, 0xFu, true, {}, {}, {}, {} }, { 6u, 0xFu, true, {}, {}, {}, {} },
                        { 7u, 0xFu, true, {}, {}, {}, {} } };
        if (!emptyS2)
            sys[2].bufs = { { 700u, 0x2u, false, {}, {}, {}, {} }, { 0u, 0xFu, true, {}, {}, {}, {} } };
        active = { 2u, 0u };
        uint32_t salt = 1;
        for (size_t si = 0; si < sys.size(); ++si)
        {
            GuardSys& s = sys[si];
            for (size_t k = 0; k < s.bufs.size(); ++k)
            {
                GuardBuf& b = s.bufs[k];
                const uint32_t cap = b.active ? b.active : 1u;      // the buffer's capacity
                b.pos = alloc(16u * cap);
                b.vel = alloc(16u * cap);
                b.restA = b.rest ? alloc(16u * cap) : GuardArr();
                b.phaseA = alloc(4u * cap);
                if (!ok)
                    return;
                for (uint32_t i = 0; i < 4u * cap; ++i)
                {
                    put32(b.pos.p, i, guardPattern(salt, i));
                    put32(b.vel.p, i, guardPattern(salt + 100u, i));
                    if (b.rest)
                        put32(b.restA.p, i, guardPattern(salt + 200u, i));
                }
                for (uint32_t i = 0; i < cap; ++i)
                    put32(b.phaseA.p, i, guardPattern(salt + 300u, i));
                ++salt;
                s.total += b.active;
            }
            const uint32_t tot = s.total ? s.total : 1u;
            s.upos = alloc(16u * tot);
            s.uvel = alloc(16u * tot);
            s.urest = alloc(16u * tot);
            s.uphase = alloc(4u * tot);
            s.simbuf = alloc(sizeof(PxgParticleSimBuffer) * s.bufs.size());   // NULL when nb == 0
            s.runsum = alloc(4u * s.bufs.size());
            if (!ok)
                return;
            for (uint32_t i = 0; i < 4u * tot; ++i)
            {
                put32(s.upos.p, i, guardPattern(salt + 400u, i));
                put32(s.uvel.p, i, guardPattern(salt + 500u, i));
                put32(s.urest.p, i, guardPattern(salt + 600u, i));
            }
            for (uint32_t i = 0; i < tot; ++i)
                put32(s.uphase.p, i, guardPattern(salt + 700u, i));
            uint32_t run = 0;
            for (size_t k = 0; k < s.bufs.size(); ++k)
            {
                const GuardBuf& b = s.bufs[k];
                PxgParticleSimBuffer r;
                std::memset(&r, 0, sizeof(r));
                r.mPositionInvMasses = reinterpret_cast<float4*>(size_t(b.pos.dev));
                r.mVelocities = reinterpret_cast<float4*>(size_t(b.vel.dev));
                r.mRestPositions = reinterpret_cast<float4*>(size_t(b.restA.dev));
                r.mPhases = reinterpret_cast<PxU32*>(size_t(b.phaseA.dev));
                r.mNumActiveParticles = b.active;
                // egress: a record-unique sentinel, so a clear of the wrong record shows
                r.mFlags = egress ? (0xF00D0000u | uint32_t(si << 8) | uint32_t(k)) : b.flags;
                r.mUniqueId = 0x5EED0000u | uint32_t(si << 8) | uint32_t(k);
                std::memcpy(s.simbuf.p + sizeof(PxgParticleSimBuffer) * k, &r, sizeof(r));
                put32(s.runsum.p, uint32_t(k), run);
                run += b.active;
            }
        }
        desc = alloc(sizeof(PxgParticleSystem) * sys.size());
        act = alloc(4u * active.size());
        if (!ok)
            return;
        for (size_t si = 0; si < sys.size(); ++si)
        {
            PxgParticleSystem ps = makeZeroDesc();
            ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(sys[si].upos.dev));
            ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(sys[si].uvel.dev));
            ps.mUnsortedPhaseArray = reinterpret_cast<PxU32*>(size_t(sys[si].uphase.dev));
            ps.mRestArray = reinterpret_cast<float4*>(size_t(sys[si].urest.dev));
            ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(sys[si].simbuf.dev));
            ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(sys[si].runsum.dev));
            ps.mCommonData.mNumParticleBuffers = PxU32(sys[si].bufs.size());
            ps.mCommonData.mNumParticles = sys[si].total;
            std::memcpy(desc.p + sizeof(PxgParticleSystem) * si, &ps, sizeof(ps));
        }
        for (size_t i = 0; i < active.size(); ++i)
            put32(act.p, uint32_t(i), active[i]);

        char nm[64];
        for (size_t si = 0; si < sys.size(); ++si)
        {
            GuardSys& s = sys[si];
            const char* sn[] = { "upos", "uvel", "uphase", "urest", "records", "runsum" };
            GuardArr* sa[] = { &s.upos, &s.uvel, &s.uphase, &s.urest, &s.simbuf, &s.runsum };
            for (int j = 0; j < 6; ++j)
            {
                std::snprintf(nm, sizeof(nm), "S%zu.%s", si, sn[j]);
                all.push_back(sa[j]);
                names.push_back(nm);
            }
            for (size_t k = 0; k < s.bufs.size(); ++k)
            {
                const char* bn[] = { "pos", "vel", "rest", "phase" };
                GuardArr* ba[] = { &s.bufs[k].pos, &s.bufs[k].vel, &s.bufs[k].restA, &s.bufs[k].phaseA };
                for (int j = 0; j < 4; ++j)
                {
                    std::snprintf(nm, sizeof(nm), "S%zu.b%zu.%s", si, k, bn[j]);
                    all.push_back(ba[j]);
                    names.push_back(nm);
                }
            }
        }
        all.push_back(&desc);
        names.push_back("descriptors");
        all.push_back(&act);
        names.push_back("active");
    }

    static void put32(uint8_t* p, uint32_t word, uint32_t v) { std::memcpy(p + 4u * word, &v, 4); }

    std::vector<std::vector<uint8_t>> snapshot() const
    {
        std::vector<std::vector<uint8_t>> r(all.size());
        for (size_t i = 0; i < all.size(); ++i)
            if (all[i]->bytes)
                r[i].assign(all[i]->p, all[i]->p + all[i]->bytes);
        return r;
    }

    // The CUDA, literally, over host SHADOWS: device pointers read from the shadow
    // descriptors / records are resolved to the shadow of the array they point into.
    void oracle(std::vector<std::vector<uint8_t>>& sh, bool ingest, GuardVariant v,
                uint32_t gx, uint32_t gy, uint32_t gz) const
    {
        auto at = [&](uint64_t addr) -> uint8_t* {
            for (size_t i = 0; i < all.size(); ++i)
                if (all[i]->bytes && addr >= all[i]->dev && addr < all[i]->dev + all[i]->bytes)
                    return sh[i].data() + (addr - all[i]->dev);
            return nullptr;   // NULL, or past an end: the caller must not dereference it
        };
        auto rd32 = [](const uint8_t* p) { uint32_t x; std::memcpy(&x, p, 4); return x; };
        auto rd64 = [](const uint8_t* p) { uint64_t x; std::memcpy(&x, p, 8); return x; };
        const uint64_t SP = sizeof(PxgParticleSystem), SB = sizeof(PxgParticleSimBuffer);
        for (uint32_t bz = 0; bz < gz; ++bz)
            for (uint32_t by = 0; by < gy; ++by)
                for (uint32_t bx = 0; bx < gx; ++bx)
                    for (uint32_t tx = 0; tx < 1024u; ++tx)
                    {
                        const uint32_t particleId = v == GuardVariant::SysDirect ? bz : rd32(at(act.dev + 4u * bz));
                        const uint8_t* ps = at(desc.dev + SP * particleId);
                        const uint32_t bufferIndex = by;
                        if (!(bufferIndex < rd32(ps + offsetof(PxgParticleSystem, mCommonData) +
                                                 offsetof(PxGpuParticleData, mNumParticleBuffers))))
                            continue;
                        const uint64_t uPos = rd64(ps + offsetof(PxgParticleSystem, mUnsortedPositions_InvMass));
                        const uint64_t uVel = rd64(ps + offsetof(PxgParticleSystem, mUnsortedVelocities));
                        const uint64_t uRest = rd64(ps + offsetof(PxgParticleSystem, mRestArray));
                        const uint64_t uPhase = rd64(ps + offsetof(PxgParticleSystem, mUnsortedPhaseArray));
                        const uint64_t recs = rd64(ps + offsetof(PxgParticleSystem, mParticleSimBuffers));
                        const uint64_t runs = rd64(ps + offsetof(PxgParticleSystem, mParticleBufferRunsum));
                        uint8_t* buffer = at(recs + SB * bufferIndex);
                        const uint32_t offset = v == GuardVariant::OffsetDrop ? 0u : rd32(at(runs + 4u * bufferIndex));
                        const uint64_t particles = rd64(buffer + offsetof(PxgParticleSimBuffer, mPositionInvMasses));
                        const uint64_t velocities = rd64(buffer + offsetof(PxgParticleSimBuffer, mVelocities));
                        const uint64_t restPose = rd64(buffer + offsetof(PxgParticleSimBuffer, mRestPositions));
                        const uint64_t phases = rd64(buffer + offsetof(PxgParticleSimBuffer, mPhases));
                        const uint32_t n = rd32(buffer + offsetof(PxgParticleSimBuffer, mNumActiveParticles));
                        const uint32_t globalThreadIndex = tx + 1024u * bx;
                        const uint8_t* flagRec = v == GuardVariant::FlagsBuf0 ? at(recs) : buffer;
                        const uint32_t flags = rd32(flagRec + offsetof(PxgParticleSimBuffer, mFlags));
                        if (globalThreadIndex < n)
                        {
                            const uint32_t ind = offset + globalThreadIndex;
                            if (ingest)
                            {
                                if (flags & 1u)
                                    std::memcpy(at(uPos + 16u * ind), at(particles + 16u * globalThreadIndex), 16);
                                if (flags & 2u)
                                    std::memcpy(at(uVel + 16u * ind), at(velocities + 16u * globalThreadIndex), 16);
                                if (flags & 4u)
                                    std::memcpy(at(uPhase + 4u * ind), at(phases + 4u * globalThreadIndex), 4);
                                if ((flags & 8u) && restPose)
                                    std::memcpy(at(uRest + 16u * ind), at(restPose + 16u * globalThreadIndex), 16);
                            }
                            else
                            {
                                std::memcpy(at(particles + 16u * globalThreadIndex), at(uPos + 16u * ind), 16);
                                std::memcpy(at(velocities + 16u * globalThreadIndex), at(uVel + 16u * ind), 16);
                            }
                        }
                        if (!ingest && globalThreadIndex == 0u && v != GuardVariant::NoFlagsClear)
                        {
                            const uint32_t zero = 0u;
                            std::memcpy(buffer + offsetof(PxgParticleSimBuffer, mFlags), &zero, 4);
                        }
                    }
    }

    // Words differing between two snapshots, printing up to 8 per array when `say`.
    size_t diff(const std::vector<std::vector<uint8_t>>& want, const std::vector<std::vector<uint8_t>>& got,
                bool say) const
    {
        size_t mm = 0;
        for (size_t i = 0; i < all.size(); ++i)
        {
            size_t here = 0;
            for (size_t w = 0; w + 4 <= want[i].size(); w += 4)
                if (std::memcmp(want[i].data() + w, got[i].data() + w, 4) != 0)
                {
                    if (say && here < 8)
                    {
                        uint32_t a, b;
                        std::memcpy(&a, got[i].data() + w, 4);
                        std::memcpy(&b, want[i].data() + w, 4);
                        std::printf("    %s word %zu: got 0x%08x want 0x%08x\n", names[i].c_str(), w / 4, a, b);
                    }
                    ++here;
                }
            mm += here;
        }
        return mm;
    }

    // Walks the members, not `all`: build() registers the arrays in `all` only after every
    // allocation succeeded, so a failure part-way through would leak those already made.
    void release()
    {
        for (GuardSys& s : sys)
        {
            for (GuardBuf& b : s.bufs)
                for (GuardArr* a : { &b.pos, &b.vel, &b.restA, &b.phaseA })
                    guardFree(dev, *a);
            for (GuardArr* a : { &s.upos, &s.uvel, &s.uphase, &s.urest, &s.simbuf, &s.runsum })
                guardFree(dev, *a);
        }
        guardFree(dev, desc);
        guardFree(dev, act);
        all.clear();
        names.clear();
    }
};

// One launch: build the world, check the oracle can see the four wrong variants (on
// the L1/E1 layout), run the real kernel on the engine grid, compare every word.
static void runGuardLaunch(bool ingest, bool emptyS2, const char* tag)
{
    PxgSlangComputeDevice* dev = testDevice();
    const bool llvmpipe = std::strstr(dev->getDeviceName(), "llvmpipe") != nullptr;
    const bool identity = guardIdentity(dev);
    GuardWorld wld;
    wld.dev = dev;
    wld.guard = identity && llvmpipe;
    static bool said = false;
    if (!said)
    {
        said = true;
        if (wld.guard)
            std::printf("  NOTE: guard pages active (llvmpipe, device address == host VA)\n");
        else
            std::printf("  NOTE: no guard pages (%s) -> bitwise checks only\n",
                        !identity ? "device addresses are not host VAs here" : "not llvmpipe");
    }
    wld.build(emptyS2, !ingest);
    PXG_CHECK_MSG(wld.ok, "%s: could not allocate the arena", tag);
    if (!wld.ok)
    {
        wld.release();
        return;
    }
    // engine grid: x = ceil(max active / 1024) over EVERY buffer of every system (the
    // decoy's included), y = max buffer count of any system, z = active systems
    uint32_t maxActive = 0, maxBufs = 0;
    for (const GuardSys& s : wld.sys)
    {
        maxBufs = std::max<uint32_t>(maxBufs, uint32_t(s.bufs.size()));
        for (const GuardBuf& b : s.bufs)
            maxActive = std::max(maxActive, b.active);
    }
    const uint32_t gx = (maxActive + 1023u) / 1024u, gy = maxBufs, gz = uint32_t(wld.active.size());
    PXG_CHECK_MSG(gx == 3u && gy == 3u && gz == 2u, "%s: grid (%u, %u, %u), want (3, 3, 2)", tag, gx, gy, gz);

    const std::vector<std::vector<uint8_t>> before = wld.snapshot();
    std::vector<std::vector<uint8_t>> want = before;
    wld.oracle(want, ingest, GuardVariant::None, gx, gy, gz);
    // The oracle must change something, and must tell every wrong variant from the CUDA.
    PXG_CHECK_MSG(wld.diff(before, want, false) > 0, "%s: the oracle changes nothing", tag);
    if (!emptyS2)
    {
        const GuardVariant vs[] = { GuardVariant::SysDirect, GuardVariant::FlagsBuf0, GuardVariant::OffsetDrop,
                                    GuardVariant::NoFlagsClear };
        const char* vn[] = { "particleId = blockIdx.z", "flags from record 0", "offset dropped", "no mFlags clear" };
        for (int i = 0; i < 4; ++i)
        {
            if (ingest && vs[i] == GuardVariant::NoFlagsClear)
                continue;   // the ingest has no clear
            if (!ingest && vs[i] == GuardVariant::FlagsBuf0)
                continue;   // the egress reads no flags
            std::vector<std::vector<uint8_t>> alt = before;
            wld.oracle(alt, ingest, vs[i], gx, gy, gz);
            PXG_CHECK_MSG(wld.diff(want, alt, false) > 0, "%s: the layout cannot see the wrong variant '%s'", tag,
                          vn[i]);
        }
    }

    struct IngestPush { uint64_t particleSystems, activeParticleSystems; };
    struct UpdPush { uint64_t particleSystems, activeParticleSystems; uint32_t count; };
    if (ingest)
    {
        Kernel k("particleSim.spv", "ps_updateUnsortedArrayLaunch", {}, sizeof(IngestPush));
        k.dispatch(gx, {}, IngestPush{ wld.desc.dev, wld.act.dev }, gy, gz);
    }
    else
    {
        Kernel k("particleSim.spv", "ps_updateUserBufferLaunch", {}, sizeof(UpdPush));
        k.dispatch(gx, {}, UpdPush{ wld.desc.dev, wld.act.dev, 42u /* the dead launch argument */ }, gy, gz);
    }
    dev->flush();

    const std::vector<std::vector<uint8_t>> got = wld.snapshot();
    size_t words = 0;
    for (const auto& a : got)
        words += a.size() / 4;
    const size_t mm = wld.diff(want, got, true);
    PXG_CHECK_MSG(mm == 0, "%s: %zu of %zu words differ from the CUDA-literal oracle", tag, mm, words);
    std::printf("  %s: grid (%u, %u, %u), %zu words compared, %zu differ%s\n", tag, gx, gy, gz, words, mm,
                wld.guard ? ", 0 faults" : "");
    wld.release();
}

PXG_TEST(particleSimIngestGuard)
{
    runGuardLaunch(true, false, "L1 ingest");
    runGuardLaunch(true, true, "L2 ingest, active system with no buffers");
}

PXG_TEST(particleSimUpdateBufferGuard)
{
    runGuardLaunch(false, false, "E1 egress");
    runGuardLaunch(false, true, "E2 egress, active system with no buffers");
}

}  // namespace
