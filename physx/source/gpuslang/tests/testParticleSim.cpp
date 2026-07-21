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

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in that expands the __device__/__align__ qualifiers the
// engine descriptor header decorates its inline members with (nocuda/ is on the
// test include path); must precede PxgParticleSystem.h so its device-qualified
// ctors parse.
#include "host_defines.h"
#include "PxgParticleSystem.h"   // real engine per-system descriptor

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
