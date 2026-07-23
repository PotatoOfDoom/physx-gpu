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
#include "PxgFEMCloth.h"                     // real engine per-cloth descriptor
#include "PxsDeformableSurfaceMaterialCore.h" // real engine surface material data

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

// P7.1: the in-plane membrane energy-solve push-constant blob, byte-identical to
// femClothSolveBda.slang's SolveNonSharedTriParams (ABI row {8,8,8,4,4,1}).
struct SolvePush
{
    uint64_t femCloths, activeFEMCloths, materials;
    float dt;
    uint32_t partitionId, isTGSWord, pad36;
};
static_assert(sizeof(SolvePush) == 40, "SolveNonSharedTri push-constant blob must be 40 bytes");

// ---------------------------------------------------------------------------
// P7.1: bit-exact CPU reference for the in-plane membrane energy solve.  The
// membrane leaf (refMembrane) + its 2x2 constraints (extractRotation2D /
// arapConstraint2D / areaConstraint2D) are lifted VERBATIM from the Phase-1-
// verified testNpClothSharedEnergy.cpp — which already pins them bit-exact vs the
// SAME Slang membraneSolve this tranche imports (pxgSlangClothMembrane, a byte-
// for-byte copy of npClothSharedEnergy's).  Kept in a private namespace so it
// cannot collide with the file-scope helpers / F4.
// ---------------------------------------------------------------------------
namespace p71ref
{
struct F4 { float x, y, z, w; };
struct V2 { float x, y; };
struct V3 { float x, y, z; };
const float FEMCLOTH_THRESHOLD = 1.0e-14f;

static V2 add2(V2 a, V2 b) { return V2{ a.x + b.x, a.y + b.y }; }
static V2 sub2(V2 a, V2 b) { return V2{ a.x - b.x, a.y - b.y }; }
static V2 muls2(V2 a, float s) { return V2{ a.x * s, a.y * s }; }
static V2 neg2(V2 a) { return V2{ -a.x, -a.y }; }
static float dot2(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
static float magSq2(V2 v) { return v.x * v.x + v.y * v.y; }
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls3(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static float magSq3(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

static V3 getNormalized3(V3 v)
{
    const float m = magSq3(v);
    return m > 0.f ? muls3(v, 1.0f / std::sqrt(m)) : V3{ 0.f, 0.f, 0.f };
}

static void extractRotation2D(V2& r0, V2& r1, V2 f0, V2 f1)
{
    const float s2x = dot2(f0, f0), s2y = dot2(f1, f1), s2z = dot2(f0, f1);
    const float det = s2x * s2y - s2z * s2z;
    if (det < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }

    const float s0 = std::sqrt(det);
    const float t = std::sqrt(s2x + s2y + 2.0f * s0);
    if (t < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float tInv = 1.0f / t;

    const float Sx = (s2x + s0) * tInv, Sy = (s2y + s0) * tInv, Sz = s2z * tInv;

    const float sDet = Sx * Sy - Sz * Sz;
    if (sDet < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float sDetInv = 1.0f / sDet;

    const float ix = Sy * sDetInv, iy = Sx * sDetInv, iz = (-Sz) * sDetInv;

    r0 = add2(muls2(f0, ix), muls2(f1, iz));
    r1 = add2(muls2(f0, iz), muls2(f1, iy));
}

static float queryDeltaLambda3(float C, V2 g0, V2 g1, V2 g2, float alphaTilde, float lambda,
                               float m0, float m1, float m2)
{
    const float denom = (m0 * magSq2(g0) + m1 * magSq2(g1) + m2 * magSq2(g2)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD)
        return 0.0f;
    return (-C - alphaTilde * lambda) / denom;
}

static void arapConstraint2D(float& lambda, V2& dx0, V2& dx1, V2& dx2, float alphaTilde,
                             const F4& QInv, V2 x01, V2 x02, float m0, float m1, float m2)
{
    const V2 f0 = add2(muls2(x01, QInv.x), muls2(x02, QInv.y));
    const V2 f1 = add2(muls2(x01, QInv.z), muls2(x02, QInv.w));

    V2 r0, r1;
    extractRotation2D(r0, r1, f0, f1);

    const V2 FMinusR0 = sub2(f0, r0);
    const V2 FMinusR1 = sub2(f1, r1);

    const float C = std::sqrt(dot2(FMinusR0, FMinusR0) + dot2(FMinusR1, FMinusR1));

    if (C > FEMCLOTH_THRESHOLD)
    {
        const float CInv = 1.0f / C;

        const V2 pC_pF0 = muls2(FMinusR0, CInv);
        const V2 pC_pF1 = muls2(FMinusR1, CInv);

        const V2 grad1 = add2(muls2(pC_pF0, QInv.x), muls2(pC_pF1, QInv.z));
        const V2 grad2 = add2(muls2(pC_pF0, QInv.y), muls2(pC_pF1, QInv.w));
        const V2 grad0 = sub2(neg2(grad1), grad2);

        const float deltaLambda = queryDeltaLambda3(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
        lambda = lambda + deltaLambda;

        dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
        dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
        dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
    }
}

static void areaConstraint2D(float& lambda, V2& dx0, V2& dx1, V2& dx2, float alphaTilde,
                             V2 x01, V2 x02, float m0, float m1, float m2, float area)
{
    const float x01CrossX02 = x01.x * x02.y - x01.y * x02.x;
    const float undeformedAreaInv = 1.0f / area;

    const float C = 0.5f * x01CrossX02 * undeformedAreaInv - 1.0f;

    const V2 grad1 = V2{ 0.5f * undeformedAreaInv * x02.y, -(0.5f * undeformedAreaInv * x02.x) };
    const V2 grad2 = V2{ -(0.5f * undeformedAreaInv * x01.y), 0.5f * undeformedAreaInv * x01.x };
    const V2 grad0 = sub2(neg2(grad1), grad2);

    const float deltaLambda = queryDeltaLambda3(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
    lambda = lambda + deltaLambda;

    dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
    dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
    dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
}

// membraneEnergySolvePerTriangle (FEMClothUtil.cuh L436-527).
static void refMembrane(F4& xx0, F4& xx1, F4& xx2, float dt, float youngs, float poissons,
                        float thickness, const F4& QInv, float vs0, float vs1, float vs2,
                        float& slotX, float& slotY, bool isTGS)
{
    if (youngs < FEMCLOTH_THRESHOLD)
        return;

    V3 x0 = V3{ xx0.x, xx0.y, xx0.z };
    V3 x1 = V3{ xx1.x, xx1.y, xx1.z };
    V3 x2 = V3{ xx2.x, xx2.y, xx2.z };

    const V3 x01 = sub3(x1, x0);
    const V3 x02 = sub3(x2, x0);

    const V3 axis0 = getNormalized3(x01);
    const V3 normal = cross3(x01, x02);
    const V3 axis1 = getNormalized3(cross3(normal, axis0));

    const float dt2 = dt * dt;

    const float det = QInv.x * QInv.w - QInv.y * QInv.z;
    const float area = 1.0f / (2.0f * det);
    const float volume = area * thickness;

    V2 dx0 = V2{ 0.f, 0.f }, dx1 = V2{ 0.f, 0.f }, dx2 = V2{ 0.f, 0.f };
    float lambda0 = 0.f, lambda1 = 0.f;

    if (!isTGS)
    {
        lambda0 = slotX;
        lambda1 = slotY;
    }

    const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
    const float mu = youngs / (2.0f * (1.0f + poissons));

    V2 xp01 = V2{ dot3(axis0, x01), dot3(axis1, x01) };
    V2 xp02 = V2{ dot3(axis0, x02), dot3(axis1, x02) };

    const float alphaTilde0 = 1.0f / (2.0f * mu * volume * dt2);

    arapConstraint2D(lambda0, dx0, dx1, dx2, alphaTilde0, QInv, xp01, xp02,
                     vs0 * xx0.w, vs1 * xx1.w, vs2 * xx2.w);

    if (poissons > FEMCLOTH_THRESHOLD)
    {
        float alphaTilde1 = 0.0f;

        if (poissons < 0.5f - FEMCLOTH_THRESHOLD)
        {
            alphaTilde1 = 1.0f / (lamL * volume * dt2);
        }

        xp01 = add2(xp01, sub2(dx1, dx0));
        xp02 = add2(xp02, sub2(dx2, dx0));

        areaConstraint2D(lambda1, dx0, dx1, dx2, alphaTilde1, xp01, xp02,
                         vs0 * xx0.w, vs1 * xx1.w, vs2 * xx2.w, area);
    }

    x0 = add3(x0, add3(muls3(axis0, dx0.x), muls3(axis1, dx0.y)));
    x1 = add3(x1, add3(muls3(axis0, dx1.x), muls3(axis1, dx1.y)));
    x2 = add3(x2, add3(muls3(axis0, dx2.x), muls3(axis1, dx2.y)));

    if (!isTGS)
    {
        slotX = lambda0;
        slotY = lambda1;
    }

    xx0.x = x0.x; xx0.y = x0.y; xx0.z = x0.z;
    xx1.x = x1.x; xx1.y = x1.y; xx1.z = x1.z;
    xx2.x = x2.x; xx2.y = x2.y; xx2.z = x2.z;
}
} // namespace p71ref

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

    // P7.1: the in-plane membrane solve (cloth_solveNonSharedTriangleEnergy*) reads
    // these fields through the pxgSlangFEMClothDesc mirror — pin their offsets.
    PXG_CHECK(offsetof(PxgFEMCloth, mMaterialIndices) == 56);                                // PxU16*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleVertexIndices_triIndex) == 80); // uint4*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleLambdas) == 96);                // float2*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleRestPoseInv) == 104);           // float4* (QInv)
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriAccumulatedPartitionsCP) == 192);           // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mNbNonSharedTriPartitions) == 344);                      // PxU32

    // the byte-packed flag word (@376): low byte == mActorFlags (eDISABLE_GRAVITY test)
    PXG_CHECK(PxActorFlag::eDISABLE_GRAVITY == (1 << 1));

    // P7.1: PxsDeformableSurfaceMaterialGpu mirror (Slang) vs the real engine
    // PxsDeformableSurfaceMaterialData (32-byte array element read per triangle).
    PXG_CHECK_MSG(sizeof(PxsDeformableSurfaceMaterialData) == 32,
                  "PxsDeformableSurfaceMaterialData sizeof == 32 (got %zu)",
                  sizeof(PxsDeformableSurfaceMaterialData));
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, youngs) == 0);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, poissons) == 4);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, dynamicFriction) == 8);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, thickness) == 12);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, bendingStiffness) == 16);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, elasticityDamping) == 20);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, bendingDamping) == 24);
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

// ---------------------------------------------------------------------------
// P7.1: cloth_solveNonSharedTriangleEnergyLaunch — the in-plane membrane energy
// solve.  One STRETCHED triangle (edge x01 pulled to 1.5x rest) in a single
// non-shared partition [0,1); dispatch, then bit-exact compare positions /
// velocities / accumulated-delta / lambdas vs a CPU transcription of
// updateNonSharedTriangle (FEMCloth.cu:610) driving the verified refMembrane.
// ---------------------------------------------------------------------------
static void runSolveNonShared(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 3;                 // one triangle
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;

    struct F2 { float x, y; };
    struct U4 { uint32_t x, y, z, w; };

    // Stretched current positions (rest edges are unit; x01 pulled to 1.5).
    std::vector<F4> positions = {
        { 0.0f, 0.0f, 0.0f, 1.0f },
        { 1.5f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f, 1.0f },
    };
    std::vector<F4> vels = {
        { 0.10f, -0.20f, 0.05f, 0.0f },
        { -0.03f, 0.07f, -0.11f, 0.0f },
        { 0.02f, -0.09f, 0.13f, 0.0f },
    };
    std::vector<F4> accum = {
        { 0.30f, -0.10f, 0.20f, 0.0f },
        { -0.40f, 0.50f, -0.60f, 0.0f },
        { 0.70f, -0.80f, 0.90f, 0.0f },
    };
    // Rest-pose inverse: Q = [[1,0],[0,1]] -> QInv = (1,0,0,1); det=1, area=0.5.
    std::vector<F4> qinv = { { 1.0f, 0.0f, 0.0f, 1.0f } };
    std::vector<F2> lambdas = { { 0.0f, 0.0f } };
    std::vector<U4> triVtx = { { 0u, 1u, 2u, 0u } };   // verts (0,1,2), triIndex 0
    std::vector<uint32_t> matIdx = { 0u };              // mMaterialIndices[0] = 0
    std::vector<uint32_t> partCP = { 1u };              // partition 0 -> [0,1)

    PxsDeformableSurfaceMaterialData mat;
    std::memset(&mat, 0, sizeof(mat));
    mat.youngs = 1.0e5f;
    mat.poissons = 0.3f;
    mat.thickness = 0.01f;
    std::vector<PxsDeformableSurfaceMaterialData> mats = { mat };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bAcc = makeDev(dev, accum);
    DevBuf bQ = makeDev(dev, qinv);
    DevBuf bLam = makeDev(dev, lambdas);
    DevBuf bTri = makeDev(dev, triVtx);
    DevBuf bMatIdx = makeDev(dev, matIdx);
    DevBuf bPart = makeDev(dev, partCP);
    DevBuf bMat = makeDev(dev, mats);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mMaterialIndices = reinterpret_cast<decltype(c.mMaterialIndices)>(size_t(bMatIdx.addr));
    c.mOrderedNonSharedTriangleVertexIndices_triIndex =
        reinterpret_cast<decltype(c.mOrderedNonSharedTriangleVertexIndices_triIndex)>(size_t(bTri.addr));
    c.mOrderedNonSharedTriangleLambdas =
        reinterpret_cast<decltype(c.mOrderedNonSharedTriangleLambdas)>(size_t(bLam.addr));
    c.mOrderedNonSharedTriangleRestPoseInv =
        reinterpret_cast<decltype(c.mOrderedNonSharedTriangleRestPoseInv)>(size_t(bQ.addr));
    c.mNonSharedTriAccumulatedPartitionsCP =
        reinterpret_cast<decltype(c.mNonSharedTriAccumulatedPartitionsCP)>(size_t(bPart.addr));
    c.mNbNonSharedTriPartitions = 1u;
    c.mNbVerts = N;

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    SolvePush push = {};
    push.femCloths = bDesc.addr;
    push.activeFEMCloths = bActive.addr;
    push.materials = bMat.addr;
    push.dt = dt;
    push.partitionId = 0u;
    push.isTGSWord = isTGS ? 1u : 0u;

    Kernel k("femClothSolveBda.spv", "cloth_solveNonSharedTriangleEnergyLaunch", {}, sizeof(SolvePush));
    k.dispatch(/*gridX numBlocks*/ 1u, {}, push, /*gridY nbActive*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference: updateNonSharedTriangle (FEMCloth.cu:610) wiring + refMembrane.
    p71ref::F4 rx0 = { positions[0].x, positions[0].y, positions[0].z, positions[0].w };
    p71ref::F4 rx1 = { positions[1].x, positions[1].y, positions[1].z, positions[1].w };
    p71ref::F4 rx2 = { positions[2].x, positions[2].y, positions[2].z, positions[2].w };
    const p71ref::F4 rQ = { qinv[0].x, qinv[0].y, qinv[0].z, qinv[0].w };
    float slotX = isTGS ? 0.0f : lambdas[0].x;
    float slotY = isTGS ? 0.0f : lambdas[0].y;

    p71ref::refMembrane(rx0, rx1, rx2, dt, mat.youngs, mat.poissons, mat.thickness, rQ,
                        1.0f, 1.0f, 1.0f, slotX, slotY, isTGS);

    std::vector<F4> refPos = positions, refVel = vels, refAcc = accum;
    const p71ref::F4 nx[3] = { rx0, rx1, rx2 };
    const F4 xcp[3] = { positions[0], positions[1], positions[2] };
    for (uint32_t v = 0; v < 3; ++v)
    {
        const F4 newPos = { nx[v].x, nx[v].y, nx[v].z, xcp[v].w }; // membrane preserves .w
        const F4 d = { newPos.x - xcp[v].x, newPos.y - xcp[v].y, newPos.z - xcp[v].z, newPos.w - xcp[v].w };
        refVel[v] = { vels[v].x + d.x * invDt, vels[v].y + d.y * invDt, vels[v].z + d.z * invDt, vels[v].w + d.w * invDt };
        refAcc[v] = { accum[v].x + d.x, accum[v].y + d.y, accum[v].z + d.z, accum[v].w + d.w };
        refPos[v] = newPos;
    }
    const F2 refLam = { slotX, slotY };

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gAcc = downloadBuffer<F4>(dev, bAcc.h, N);
    const auto gLam = downloadBuffer<F2>(dev, bLam.h, 1);

    // The solve must actually move the stretched vertex (teeth: not a no-op).
    PXG_CHECK_MSG(f2u(gPos[1].x) != f2u(positions[1].x),
                  "energy solve should correct the stretched triangle (x1.x moved)");

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, uint32_t i, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm, i,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    for (uint32_t v = 0; v < N; ++v)
    {
        cmp4("pos", v, refPos[v], gPos[v]);
        cmp4("vel", v, refVel[v], gVel[v]);
        cmp4("acc", v, refAcc[v], gAcc[v]);
    }
    if (!isTGS && (f2u(refLam.x) != f2u(gLam[0].x) || f2u(refLam.y) != f2u(gLam[0].y)))
    { ++mm; std::printf("  lambda MM: ref(%g,%g) got(%g,%g)\n", refLam.x, refLam.y, gLam[0].x, gLam[0].y); }

    PXG_CHECK_MSG(mm == 0, "cloth_solveNonSharedTriangleEnergy bit-exact vs CPU (%zu mismatches, isTGS=%d)",
                  mm, int(isTGS));

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bQ.h); dev->destroyBuffer(bLam.h); dev->destroyBuffer(bTri.h);
    dev->destroyBuffer(bMatIdx.h); dev->destroyBuffer(bPart.h); dev->destroyBuffer(bMat.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(femClothSimSolveNonSharedPGS) { runSolveNonShared(/*isTGS*/ false); }
PXG_TEST(femClothSimSolveNonSharedTGS) { runSolveNonShared(/*isTGS*/ true); }

} // namespace
