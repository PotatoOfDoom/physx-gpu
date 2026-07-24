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

// P8.0 SOFTBODY (FEM-volume) bring-up test: the softbody free-fall
// predict/integrate kernel (softBodySimBda.spv) on the engine ABI (BDA), driven by
// the REAL engine descriptor.  A genuine PxgSoftBody (gpusimulationcontroller
// header) is filled with device buffer addresses + scalars, DMA'd to the device,
// and read back through the Slang pxgSlangSoftBodyDesc mirror — the first-ever
// PxgSoftBody BDA round-trip.  static_asserts / offsetof checks pin the mirror's
// byte offsets against the real engine struct (the highest-risk item, sizeof ==
// 432), and the kernel's output is compared BIT-EXACT against a CPU transcription
// of the softBodyGM.cu math.
//
// Suites (substring-matched by the "softBodySim" ctest filter):
//   softBodySimDescLayout        — byte-offset proof of the descriptor mirror
//   softBodySimPreIntegratePGS   — sb_gm_preIntegrateLaunch PGS predict+integrate
//   softBodySimPreIntegrateTGS   — sb_gm_preIntegrateLaunch TGS branch (vel + zero posDelta)

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers the engine
// descriptor header decorates its inline members with; must precede PxgSoftBody.h.
#include "host_defines.h"
#include "PxgSoftBody.h"                     // real engine per-softbody descriptor
#include "PxActor.h"                         // PxActorFlag::eDISABLE_GRAVITY
#include "PxDeformableBodyFlag.h"            // PxDeformableBodyFlag::eKINEMATIC / eENABLE_SPECULATIVE_CCD
#include "PxDeformableVolumeFlag.h"          // PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC

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

// Same literal the shader uses for PX_MAX_REAL; keeping both sides identical makes
// the max-velocity clamp `maxVel != PX_MAX_REAL` short-circuit false (clamp dead),
// so the 1/sqrt transcendental never feeds the output (no bit-exactness seam).
static const float PX_MAX_REAL_F = 3.40282347e+38f;

// The push-constant blob, byte-identical to softBodySimBda.slang's
// [[vk::push_constant]] PreIntegrateParams (which the shim reproduces from the CUDA
// arg pack).  ABI row {8,8,12,4,1,8,1} -> 56 bytes.
struct PrePush
{
    uint64_t softbodies, activeId;
    float gx, gy, gz, dt;
    uint32_t isTGSWord, pad36;
    uint64_t speculativeCCD;
    uint32_t extForcesWord, pad52;
};
static_assert(sizeof(PrePush) == 56, "PreIntegrate push-constant blob must be 56 bytes");

// Zeroed real PxgSoftBody (no vtable — memset-safe, like PxgFEMCloth / PxgParticleSystem).
static PxgSoftBody makeZeroDesc()
{
    PxgSoftBody c;
    std::memset(&c, 0, sizeof(c));
    return c;
}

// ---------------------------------------------------------------------------
// Descriptor mirror layout proof — pins pxgSlangSoftBodyDesc against the real
// engine PxgSoftBody (offsets the kernel dereferences).  sizeof == 432.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimDescLayout)
{
    // sizeof / array stride: the kernel indexes softbodies[id], so the STRIDE must
    // match (verified in SPIR-V as ArrayStride 432).
    PXG_CHECK_MSG(sizeof(PxgSoftBody) == 432, "PxgSoftBody sizeof == 432 (got %zu)", sizeof(PxgSoftBody));

    // device-pointer members the pre-integrate driver reads/writes
    PXG_CHECK(offsetof(PxgSoftBody, mSimVelocity_InvMass) == 88);
    PXG_CHECK(offsetof(PxgSoftBody, mSimPosition_InvMass) == 96);
    PXG_CHECK(offsetof(PxgSoftBody, mSimKinematicTarget) == 104);
    PXG_CHECK(offsetof(PxgSoftBody, mVertsAreDeformed) == 112);
    PXG_CHECK(offsetof(PxgSoftBody, mVertsCanNotDeform) == 120);
    PXG_CHECK(offsetof(PxgSoftBody, mSimDeltaPos) == 128);

    // scalar tail
    PXG_CHECK(offsetof(PxgSoftBody, mLinearDamping) == 316);
    PXG_CHECK(offsetof(PxgSoftBody, mMaxLinearVelocity) == 320);
    PXG_CHECK(offsetof(PxgSoftBody, mPenBiasClamp) == 324);
    PXG_CHECK(offsetof(PxgSoftBody, mNumVertsGM) == 372);
    PXG_CHECK(offsetof(PxgSoftBody, mElementIndex) == 384);
    PXG_CHECK(offsetof(PxgSoftBody, mGpuRemapIndex) == 388);

    // the byte-packed flag triple (@392): byte0 == mActorFlags, byte1 == mBodyFlags,
    // bytes2..3 == mVolumeFlags -> the shader's mFlagsWord fold.
    PXG_CHECK(offsetof(PxgSoftBody, mActorFlags) == 392);
    PXG_CHECK(offsetof(PxgSoftBody, mBodyFlags) == 393);
    PXG_CHECK(offsetof(PxgSoftBody, mVolumeFlags) == 394);
    PXG_CHECK(offsetof(PxgSoftBody, mSimPullIndices) == 400); // realigned 8-byte ptr after the flag word
    PXG_CHECK(offsetof(PxgSoftBody, mAwake) == 408);
    PXG_CHECK(offsetof(PxgSoftBody, mRestDistance) == 420);

    // flag bit values the mirror hard-codes (pxgSlangSoftBodyDesc constants).
    PXG_CHECK(PxActorFlag::eDISABLE_GRAVITY == (1 << 1));                     // 2
    PXG_CHECK(PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD == (1 << 1));     // 2
    PXG_CHECK(PxDeformableBodyFlag::eKINEMATIC == (1 << 2));                  // 4
    PXG_CHECK(PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC == (1 << 1));      // 2
}

// ---------------------------------------------------------------------------
// Shared preIntegrate driver: dispatch + bit-exact CPU compare for a given
// isTGS mode.  invMass encoded in .w; vertex 3 has w==0 (fixed -> no delta),
// the rest are free (invMass != 0 -> fall).  Non-kinematic softbody (mBodyFlags
// == 0, mVolumeFlags == 0), gravity enabled (mActorFlags == 0).
// ---------------------------------------------------------------------------
static void runPreIntegrate(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const float3 g = { 0.0f, -9.81f, 0.0f };
    const float linearDamping = 0.1f;
    const float maxLinVel = PX_MAX_REAL_F;   // default -> clamp dead, no sqrt seam
    const uint32_t actorFlags = 0u;          // gravity enabled

    std::vector<F4> positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 3) ? 0.0f : 1.0f; // vertex 3 is fixed (invMass == 0)
        positions[i] = { float(i) * 1.0f, 5.0f + float(i) * 0.25f, -float(i) * 0.5f, w };
        vels[i] = { 0.1f * float(i), -0.2f * float(i), 0.05f * float(i), 0.0f };
    }
    std::vector<F4> posDelta(N, { 9.f, 9.f, 9.f, 9.f });
    std::vector<float> speculativeCCD = { 7.0f };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, posDelta);
    DevBuf bCcd = makeDev(dev, speculativeCCD);

    PxgSoftBody c = makeZeroDesc();
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mSimKinematicTarget = nullptr;         // no kinematic target on the free-fall scene
    c.mLinearDamping = linearDamping;
    c.mMaxLinearVelocity = maxLinVel;
    c.mNumVertsGM = N;
    c.mActorFlags = PxU8(actorFlags);
    c.mBodyFlags = 0u;                        // not eKINEMATIC / not eENABLE_SPECULATIVE_CCD
    c.mVolumeFlags = 0u;                      // not ePARTIALLY_KINEMATIC
    c.mAwake = 1u;                            // gThreadIdx==0 must reset this to 0

    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    PrePush push = {};
    push.softbodies = bDesc.addr;
    push.activeId = bActive.addr;
    push.gx = g.x; push.gy = g.y; push.gz = g.z; push.dt = dt;
    push.isTGSWord = isTGS ? 1u : 0u;
    push.speculativeCCD = bCcd.addr;
    push.extForcesWord = 0u;                  // externalForcesEveryTgsIterationEnabled = false

    Kernel k("softBodySimBda.spv", "sb_gm_preIntegrateLaunch", {}, sizeof(PrePush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference (bit-exact expression order, matching softBodyGM.cu:72 else-branch:
    // non-kinematic softbody, no partially-kinematic target).
    std::vector<F4> refPos = positions, refVel = vels, refDelta = posDelta;
    for (uint32_t i = 0; i < N; ++i)
    {
        F4 tPos = positions[i];
        float vx = 0.f, vy = 0.f, vz = 0.f;
        const float ox = tPos.x, oy = tPos.y, oz = tPos.z;

        // kinematic == false (mVolumeFlags has no ePARTIALLY_KINEMATIC)
        if (tPos.w != 0.0f)
        {
            vx = vels[i].x; vy = vels[i].y; vz = vels[i].z;
            // !externalForcesEveryTgsIterationEnabled
            // actorFlags has no eDISABLE_GRAVITY -> apply gravity
            vx += g.x * dt; vy += g.y * dt; vz += g.z * dt;
            const float damping = 1.0f - std::fmin(dt * linearDamping, 1.0f);
            vx *= damping; vy *= damping; vz *= damping;
        }

        // Max velocity clamping (dead: maxVel == PX_MAX_REAL -> ternary picks vel).
        const float maxVel = maxLinVel;
        const float velMagSq = vx * vx + vy * vy + vz * vz;
        if (maxVel != PX_MAX_REAL_F && maxVel * maxVel < velMagSq)
        {
            const float s = maxVel / std::sqrt(velMagSq);
            vx *= s; vy *= s; vz *= s;
        }

        float dx = 0.f, dy = 0.f, dz = 0.f;
        if (!isTGS) { dx += vx * dt; dy += vy * dt; dz += vz * dt; }

        refPos[i] = { ox + dx, oy + dy, oz + dz, tPos.w };
        refVel[i] = { vx, vy, vz, tPos.w };
        refDelta[i] = { dx, dy, dz, tPos.w };
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);
    const auto gDesc = downloadBuffer<PxgSoftBody>(dev, bDesc.h, 1);

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
    // globalThreadIndex==0 resets mAwake to 0 (PxU32 store).
    if (gDesc[0].mAwake != 0u) { ++mm; std::printf("  mAwake MM: %u (expected 0)\n", gDesc[0].mAwake); }

    PXG_CHECK_MSG(mm == 0, "sb_gm_preIntegrateLaunch (%s) bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm);

    // Sanity beacons the bit-exact compare already implies, kept human-readable:
    //  - free vertex 0 (invMass != 0) falls: PGS y-position strictly below its start.
    if (!isTGS)
        PXG_CHECK_MSG(gPos[0].y < positions[0].y,
                      "free vertex 0 should fall (y %g -> %g)", positions[0].y, gPos[0].y);
    //  - fixed vertex 3 (invMass == 0) never moves and gets a zero delta.
    PXG_CHECK_MSG(f2u(gPos[3].x) == f2u(positions[3].x) && f2u(gPos[3].y) == f2u(positions[3].y) &&
                  f2u(gPos[3].z) == f2u(positions[3].z),
                  "fixed vertex 3 must not move");
    PXG_CHECK_MSG(gDelta[3].x == 0.f && gDelta[3].y == 0.f && gDelta[3].z == 0.f,
                  "fixed vertex 3 delta must be zero");
    //  - TGS: delta stays zero everywhere but velocity still received gravity.
    if (isTGS)
    {
        PXG_CHECK_MSG(gDelta[0].x == 0.f && gDelta[0].y == 0.f && gDelta[0].z == 0.f,
                      "TGS free vertex 0 delta must be zero");
        PXG_CHECK_MSG(gVel[0].y < vels[0].y, "TGS free vertex 0 velocity must receive gravity");
    }

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bCcd.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimPreIntegratePGS) { runPreIntegrate(/*isTGS*/ false); }
PXG_TEST(softBodySimPreIntegrateTGS) { runPreIntegrate(/*isTGS*/ true); }

} // namespace
