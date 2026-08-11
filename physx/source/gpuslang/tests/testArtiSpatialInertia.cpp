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

// P8.84: computeSpatialInertiaW (forwardDynamic2.cu:1114-1310), and
// P8.85: computeUnconstrainedSpatialInertiaLaunchPartial1T, the wrapper over it.
//
// Two tests in one translation unit, sharing ONE mirror (`asiMirror` below).  A
// second copy of that mirror would be the transcription this chain exists to
// avoid, and a header for two callers in the same directory would be ceremony.
//
// ===========================================================================
// THE ORACLE, AND WHY IT IS STRONGER THAN A TRANSCRIBED EXPECTATION
// ===========================================================================
//
// `computeSpatialInertiaW` is `static __device__` and cannot be called from the
// host.  But every primitive it is built from IS `PX_CUDA_CALLABLE`:
// `Cm::transformInertiaTensor`, `PxMat33(PxQuat)`, `PxVec3::cross`,
// `PxVec3::normalize`, and — this is the part that matters most — the SoA
// writers `storeSpatialMatrix`, `storePxMat33` and `storeSpatialVector`.
//
// So the expectation below is a mirror of the CUDA's *control flow* in which
// every arithmetic expression and every store is the engine's own code, writing
// into a real `PxgArticulationBlockLinkData` on the host.  That host image IS
// the expected device image, byte for byte.  Nothing about the maths and
// nothing about the packing is retyped into a literal — which is the failure
// mode P8.49a was: a shader and a fixture sharing one wrong number.
//
// The residual gap is honest and stated: the loop structure and the branch
// conditions are transcribed, so a mis-transcribed *structure* would be mirrored
// in both. That is what the mutation sweep is for, and it is why the sweep for
// this tranche attacks the structure (loop bounds, branch order, which slot a
// value lands in) rather than the arithmetic.
//
// ===========================================================================
// WHAT THE FIXTURE HAS TO CONTAIN, AND WHY
// ===========================================================================
//
// Five lanes with link counts 1, 3, 5, 6, 2 — all distinct from each other and
// from the lane index, which is the standing rule five earlier tranches paid
// for.  MAX_LINKS is 7, so every lane leaves unused link blocks that must come
// back untouched at 0xAB.
//
// The per-link cases are generated from a mixing function and then COUNTED, and
// the test fails if any mode stops occurring.  A fixture that silently stops
// covering a case after an edit is worse than one that never did.  Three
// booleans, five damping modes and six velocity modes:
//
//   * `zeroMass`             -> the `w == 0 ? 0 : 1/w` guards; a static link
//   * `disableGravity`       both ways
//   * `retainsAccelerations` both ways -> the record-side CLEAR
//   * DAMP_NONE / _LIN / _ANG / _BOTH / _HUGE -- one coefficient at a time,
//     because setting both together makes `||` and `&&` the same predicate, and
//     _HUGE puts them above `invDt` so the `min(coeff, invDt)` clamp is live
//   * VEL_SMALL / _OVER_ANG / _OVER_LIN / _OVER_BOTH / _ZERO_ANG / _RECIP --
//     _OVER_ANG's magnitude lies BETWEEN the two caps so the angular test's
//     threshold is pinned, _ZERO_ANG exercises `normalize()`'s `m > 0` guard,
//     and _RECIP is a SEARCHED magnitude where `v * (1/m)` and `v / m` round
//     differently (they agree for every round-numbered velocity)
//
// The first sweep left six mutants alive on exactly the coincidences that list
// now rules out.
//
// Two dispatches, for `isExternalForcesEveryTgsIterationEnabled` off and on:
// it is a descriptor bool that reroutes zExt into the record's `zAForces`, so a
// single pass cannot see it at all.  `dt` is 1/60, not 1, so `invDt` is
// observable in every term that clamps against it.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "PxgArticulationCoreDesc.h"
#include "PxgBodySim.h"
#include "PxNodeIndex.h"
#include "CmUtils.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiSpatialInertiaLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;
static const uint32_t MAX_LINKS = 7;
static const uint32_t MAX_DOFS = 13;
static const uint32_t NUM_LINKS[LANES] = { 1u, 3u, 5u, 6u, 2u };

// a dense index over the 17 real links, so the mode cycles below cover every
// mode instead of skipping with the per-lane stride
static uint32_t linkOrdinal(uint32_t L, uint32_t i)
{
    uint32_t n = 0;
    for (uint32_t l = 0; l < L; ++l)
        n += NUM_LINKS[l];
    return n + i;
}

// The velocity caps.  Deliberately NOT perfect squares and deliberately
// different from each other: the angular cap is `.w` and the linear cap is `.z`
// of one float4, so a fixture where the two are interchangeable cannot tell a
// kernel that reads the wrong one.  Non-squares because the engine computes
// `sqrt(cap)/sqrt(mag)` and a port that writes `sqrt(cap/mag)` is only visible
// when the two roundings differ.
// The values are SEARCHED, not chosen by eye: with the first pair (4.3 / 9.7)
// `sqrt(cap)/sqrt(mag)` and `sqrt(cap/mag)` rounded to the same float for every
// magnitude in the fixture, and the mutant that swaps the two spellings lived.
// 0.65 against a magnitude-squared of 1.625 is the first pair that separates
// them while keeping the over-angular link under the linear cap.
static const float ANG_CAP = 0.65f;
static const float LIN_CAP = 9.7f;

// Five damping modes and five velocity modes, because the first draft of this
// fixture had `damped` as a single bool with both coefficients set together --
// which made `linDamping > 0 || angDamping > 0` and `&&` the same predicate, and
// left four mutants alive on the coincidence.
enum DampMode
{
    DAMP_NONE = 0,   // both coefficients zero: the guard must not fire
    DAMP_LIN,        // linear only  -> separates || from &&
    DAMP_ANG,        // angular only -> the other direction
    DAMP_BOTH,
    DAMP_HUGE        // both above invDt -> the min(coeff, invDt) clamp is live
};

enum VelMode
{
    VEL_SMALL = 0,   // under both caps
    VEL_OVER_ANG,    // angMag BETWEEN the two caps -> pins which one the
                     // angular test reads
    VEL_OVER_LIN,    // linMag over the linear cap only
    VEL_OVER_BOTH,
    VEL_ZERO_ANG,    // zero angular velocity -> normalize()'s m > 0 guard
    VEL_RECIP        // a magnitude where `v * (1/m)` and `v / m` round
                     // DIFFERENTLY -- searched for, because they agree for every
                     // round-numbered velocity and the mutant that divides three
                     // times instead of multiplying by the reciprocal lived
};
static const uint32_t NB_VEL_MODES = 6;

struct LinkCfg
{
    bool zeroMass;
    bool disableGravity;
    bool retains;
    DampMode damp;
    VelMode vel;
};

static uint32_t linkOrdinal(uint32_t L, uint32_t i);

static LinkCfg cfgOf(uint32_t L, uint32_t i)
{
    const uint32_t n = linkOrdinal(L, i);
    LinkCfg c;
    c.zeroMass = (n % 7u) == 3u;
    c.disableGravity = ((n >> 1) & 1u) != 0u;
    c.retains = ((n >> 2) & 1u) != 0u;
    // 3 is coprime with 5, so both cycles cover all five modes over the 17 links
    c.damp = DampMode(n % 5u);
    // 5 is coprime with 6, so this cycles through all six modes
    c.vel = VelMode((n * 5u) % NB_VEL_MODES);
    return c;
}

// dofs per link: 0 for the root, then 1..3 varying, and never equal to the link
// index (so a loop that uses one for the other is visible)
static uint32_t dofsOf(uint32_t L, uint32_t i)
{
    if (i == 0u)
        return 0u;
    return 1u + ((L + i) % 3u);
}

using namespace physx;
using namespace physx::Dy;

// =========================================================================
// THE MIRROR, hoisted (P8.85)
// =========================================================================
//
// The CUDA's control flow with the ENGINE's arithmetic and the ENGINE's stores,
// writing into a host copy of the block image.  It is a file-static function
// rather than a lambda so that BOTH tests in this file use it -- the per-link
// body test below and the engine-ABI wrapper test after it.  One mirror, two
// users: a second copy would be the transcription this chain exists to avoid.
//
// The arrays arrive as per-warp / per-articulation SLICES, because the wrapper
// hands out a different slice of each to every warp.
static void asiMirror(PxgArticulationBlockLinkData* lk,
                      PxgArticulationBlockDofData* df,
                      PxgArticulationBlockData& ab,
                      Cm::UnAlignedSpatialVector* ea,
                      Cm::UnAlignedSpatialVector* za,
                      const PxgArticulationLinkProp* props,
                      uint32_t numLinks, uint32_t lane,
                      const PxVec3& gravity, float dt, bool extEveryIter)
{
    const PxReal invDt = 1.0f / dt;
    PxVec3 COM(0.0f);
    PxReal totalMass = 0.0f;

    SpatialMatrix spatialInertia;
    spatialInertia.topLeft = PxMat33(PxZero);
    spatialInertia.topRight = PxMat33(PxZero);

    for (PxU32 linkID = 0; linkID < numLinks; ++linkID)
    {
        const float4 invInertiaXYZ_invMass =
            make_float4(props[linkID].invInertia.x, props[linkID].invInertia.y, props[linkID].invInertia.z,
                        props[linkID].invMass);

        PxgArticulationBlockLinkData& linkData = lk[linkID];
        const PxTransform t = loadSpatialTransform(linkData.mAccumulatedPose, lane);

        const PxReal m = invInertiaXYZ_invMass.w == 0.f ? 0.f : (1.f / invInertiaXYZ_invMass.w);
        const PxVec3 inertiaTensor(
            invInertiaXYZ_invMass.x == 0.f ? 0.f : (1.f / invInertiaXYZ_invMass.x),
            invInertiaXYZ_invMass.y == 0.f ? 0.f : (1.f / invInertiaXYZ_invMass.y),
            invInertiaXYZ_invMass.z == 0.f ? 0.f : (1.f / invInertiaXYZ_invMass.z));

        spatialInertia.topRight[0][0] = spatialInertia.topRight[1][1] =
            spatialInertia.topRight[2][2] = m;

        Cm::transformInertiaTensor(inertiaTensor, PxMat33(t.q), spatialInertia.bottomLeft);
        storeSpatialMatrix(linkData.mSpatialArticulatedInertia, lane, spatialInertia);
        storePxMat33(linkData.mIsolatedInertia, lane, spatialInertia.bottomLeft);

        COM += t.p * m;
        totalMass += m;

        const Cm::UnAlignedSpatialVector vel = loadSpatialVector(linkData.mMotionVelocity, lane);
        const float4 damping = linkData.mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW[lane];

        PxVec3 force(0.f), torque(0.f);
        if (linkID != 0)
        {
            const PxU32 parent = linkData.mParents[lane];
            const Cm::UnAlignedSpatialVector pVel =
                loadSpatialVector(lk[parent].mMotionVelocity, lane);
            const PxVec3 rw(linkData.mRw_x[lane], linkData.mRw_y[lane], linkData.mRw_z[lane]);

            torque = pVel.top.cross(pVel.top.cross(rw));

            const PxU32 jointOffset = linkData.mJointOffset[lane];
            const PxU32 nbDofs = linkData.mDofs[lane];
            if (nbDofs)
            {
                Cm::UnAlignedSpatialVector deltaV = Cm::UnAlignedSpatialVector::Zero();
                for (PxU32 ind = 0; ind < nbDofs; ++ind)
                {
                    const PxReal jVel = df[jointOffset + ind].mJointVelocities[lane];
                    deltaV += loadSpatialVector(df[jointOffset + ind].mWorldMotionMatrix, lane) * jVel;
                }
                const PxVec3 aVec = deltaV.top;
                force = pVel.top.cross(aVec);
                const PxVec3 lVel = deltaV.bottom;
                const PxVec3 temp1 = 2.f * pVel.top.cross(lVel);
                const PxVec3 temp2 = aVec.cross(lVel);
                torque += temp1 + temp2;
            }
        }

        storeSpatialVector(linkData.mCoriolis, Cm::UnAlignedSpatialVector(force, torque), lane);
        linkData.mInvInertiaXYZ_invMassW[lane] = invInertiaXYZ_invMass;

        PxVec3 vA = vel.top;
        PxReal mag = vA.normalize();
        mag = PxMin(mag, invDt);
        vA *= mag;

        PxVec3 gravLinAccel(0.f);
        if (!linkData.mDisableGravity[lane])
            gravLinAccel = -gravity;

        Cm::UnAlignedSpatialVector zExt(gravLinAccel * m, PxVec3(0.f));
        Cm::UnAlignedSpatialVector zDamp = Cm::UnAlignedSpatialVector::Zero();
        const Cm::UnAlignedSpatialVector zInt(PxVec3(0.f),
                                              vA.cross(spatialInertia.bottomLeft * vA));

        {
            const Cm::UnAlignedSpatialVector externalAccel = ea[linkID];
            zExt.top += (-externalAccel.top * m);
            zExt.bottom += spatialInertia.bottomLeft * (-externalAccel.bottom);
            if (!linkData.mRetainsAcceleration[lane])
                ea[linkID] = Cm::UnAlignedSpatialVector::Zero();
        }

        if (damping.x > 0.f || damping.y > 0.f)
        {
            const PxReal linDamp = PxMin(damping.x, invDt);
            const PxReal angDamp = PxMin(damping.y, invDt);
            zDamp.top = (vel.bottom * linDamp * m) - zExt.top * linDamp * dt;
            zDamp.bottom =
                spatialInertia.bottomLeft * (vel.top * angDamp) - zExt.bottom * angDamp * dt;
        }

        const PxReal angMag = vel.top.magnitudeSquared();
        const PxReal linMag = vel.bottom.magnitudeSquared();
        if (angMag > damping.w || linMag > damping.z)
        {
            if (angMag > damping.w)
            {
                const PxReal scale = 1.f - PxSqrt(damping.w) / PxSqrt(angMag);
                const PxVec3 tmpAng = (spatialInertia.bottomLeft * vel.top) * scale;
                zDamp.bottom += tmpAng * invDt;
            }
            if (linMag > damping.z)
            {
                const PxReal scale = 1.f - (PxSqrt(damping.z) / PxSqrt(linMag));
                const PxVec3 tmpLin = (vel.bottom * m * scale);
                zDamp.top += tmpLin * invDt;
            }
        }

        Cm::UnAlignedSpatialVector zExtToStore;
        if (extEveryIter)
        {
            za[linkID] = zExt;
            zExtToStore = zDamp;
        }
        else
        {
            zExtToStore = zDamp + zExt;
        }

        storeSpatialVector(linkData.mZAVector, zExtToStore, lane);
        storeSpatialVector(linkData.mZAIntVector, zInt, lane);
        storeSpatialVector(linkData.mBiasForce, zExtToStore + zInt, lane);
        linkData.mMass[lane] = m;
    }

    const PxReal invMass = 1.f / totalMass;
    COM *= invMass;
    ab.mCOM_TotalInvMassW[lane] = make_float4(COM.x, COM.y, COM.z, invMass);
}

PXG_TEST(artiSpatialInertia)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64 x4, f32 x4, u32 x4 } -> 64 bytes
        { "npArtiSpatialInertia.spv", "artiSpatialInertiaW", 64u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiSpatialInertiaLoader);

    const PxVec3 GRAVITY(0.75f, -9.8125f, 0.5f);   // exact in binary32
    const PxReal DT = 1.0f / 60.0f;

    auto fval = [](uint32_t L, uint32_t i, uint32_t f) {
        return float(10 * (L + 1) + (i + 1)) * 0.125f + float(f) * 0.03125f;
    };

    // ---- the record side: link props and the two spatial-vector arrays ----
    std::vector<PxgArticulationLinkProp> props(size_t(LANES) * MAX_LINKS);
    std::memset(props.data(), 0, props.size() * sizeof(PxgArticulationLinkProp));
    std::vector<Cm::UnAlignedSpatialVector> extAccels(size_t(LANES) * MAX_LINKS);
    std::vector<Cm::UnAlignedSpatialVector> zaForces(size_t(LANES) * MAX_LINKS);
    std::memset(extAccels.data(), 0, extAccels.size() * sizeof(Cm::UnAlignedSpatialVector));
    std::memset(zaForces.data(), 0xCD, zaForces.size() * sizeof(Cm::UnAlignedSpatialVector));

    uint32_t seen[3] = { 0, 0, 0 };
    uint32_t seenNot[3] = { 0, 0, 0 };
    uint32_t seenDamp[5] = { 0, 0, 0, 0, 0 };
    uint32_t seenVel[NB_VEL_MODES] = { 0, 0, 0, 0, 0, 0 };

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t i = 0; i < NUM_LINKS[L]; ++i)
        {
            const LinkCfg c = cfgOf(L, i);
            const size_t x = size_t(L) * MAX_LINKS + i;
            const bool flags[3] = { c.zeroMass, c.disableGravity, c.retains };
            for (uint32_t b = 0; b < 3; ++b)
                (flags[b] ? seen[b] : seenNot[b]) += 1u;
            seenDamp[uint32_t(c.damp)] += 1u;
            seenVel[uint32_t(c.vel)] += 1u;

            if (c.zeroMass)
            {
                // a static link: every inverse is zero, which is the case the
                // `w == 0 ? 0 : 1/w` guards exist for
                props[x].invInertia = PxVec3(0.0f, 0.0f, 0.0f);
                props[x].invMass = 0.0f;
            }
            else
            {
                props[x].invInertia = PxVec3(fval(L, i, 1), fval(L, i, 2), fval(L, i, 3));
                props[x].invMass = fval(L, i, 4);
            }

            extAccels[x] = Cm::UnAlignedSpatialVector(
                PxVec3(fval(L, i, 10), fval(L, i, 11), fval(L, i, 12)),
                PxVec3(fval(L, i, 13), fval(L, i, 14), fval(L, i, 15)));
        }
    }

    // every case must actually occur, both ways where that makes sense
    static const char* const CASE_NAME[3] = { "zeroMass", "disableGravity",
                                             "retainsAccelerations" };
    for (uint32_t b = 0; b < 3; ++b)
    {
        PXG_CHECK_MSG(seen[b] > 0, "fixture never exercises %s", CASE_NAME[b]);
        PXG_CHECK_MSG(seenNot[b] > 0, "fixture never exercises !%s", CASE_NAME[b]);
    }
    static const char* const DAMP_NAME[5] = { "DAMP_NONE", "DAMP_LIN", "DAMP_ANG", "DAMP_BOTH",
                                             "DAMP_HUGE" };
    static const char* const VEL_NAME[NB_VEL_MODES] = { "VEL_SMALL", "VEL_OVER_ANG",
                                                       "VEL_OVER_LIN", "VEL_OVER_BOTH",
                                                       "VEL_ZERO_ANG", "VEL_RECIP" };
    for (uint32_t b = 0; b < 5; ++b)
        PXG_CHECK_MSG(seenDamp[b] > 0, "fixture never exercises %s", DAMP_NAME[b]);
    for (uint32_t b = 0; b < NB_VEL_MODES; ++b)
        PXG_CHECK_MSG(seenVel[b] > 0, "fixture never exercises %s", VEL_NAME[b]);

    // ---- the block images.  These are REAL engine structs, so every offset the
    // fixture touches is the header's, and the expectation is produced by the
    // engine's own writers into a second copy of the same image.
    std::vector<PxgArticulationBlockLinkData> lkImg(MAX_LINKS);
    std::vector<PxgArticulationBlockDofData> dfImg(MAX_DOFS);
    std::vector<PxgArticulationBlockData> abImg(1);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(PxgArticulationBlockData));

    // ---- seed exactly the inputs the kernel reads ----
    for (uint32_t L = 0; L < LANES; ++L)
    {
        uint32_t dofCursor = 0;
        for (uint32_t i = 0; i < NUM_LINKS[L]; ++i)
        {
            const LinkCfg c = cfgOf(L, i);
            PxgArticulationBlockLinkData& b = lkImg[i];

            const PxQuat q = PxQuat(fval(L, i, 20), fval(L, i, 21), fval(L, i, 22),
                                    fval(L, i, 23)).getNormalized();
            const PxVec3 p(fval(L, i, 30) * 3.0f + 0.75f * float(i * i),
                           fval(L, i, 31) * 11.0f - 0.4f * float(i),
                           fval(L, i, 32) * 7.0f + 1.5f * float(i));
            storeSpatialTransform(b.mAccumulatedPose, L, PxTransform(p, q));

            // VEL_OVER_ANG puts angMag BETWEEN the two caps (5.25 is above 4.3 and
            // below 9.7), which is the only shape that separates a kernel reading
            // the angular threshold from one reading the linear one.
            PxVec3 angVel(0.25f, -0.5f, 0.125f);          // angMag 0.328125
            PxVec3 linVel(0.5f, 0.75f, -0.25f);           // linMag 0.875
            switch (c.vel)
            {
            case VEL_OVER_ANG:   angVel = PxVec3(0.0f, -0.25f, 1.25f); break;      // 1.625
            case VEL_OVER_LIN:   linVel = PxVec3(3.0f, -1.5f, 1.25f); break;       // 12.8125
            case VEL_OVER_BOTH:  angVel = PxVec3(3.0f, 2.0f, 1.5f);                // 15.25
                                 linVel = PxVec3(4.0f, -3.5f, 2.25f); break;       // 33.3125
            case VEL_ZERO_ANG:   angVel = PxVec3(0.0f, 0.0f, 0.0f); break;
            case VEL_RECIP:      angVel = PxVec3(0.125f, -0.125f, 0.75f); break;   // 0.59375
            default: break;
            }
            storeSpatialVector(b.mMotionVelocity, Cm::UnAlignedSpatialVector(angVel, linVel), L);

            // DAMP_HUGE is above invDt (60) so the min(coeff, invDt) clamp is live;
            // DAMP_LIN / DAMP_ANG set exactly one coefficient so the guard's `||`
            // is not interchangeable with `&&`.
            PxReal linDamp = 0.0f, angDamp = 0.0f;
            switch (c.damp)
            {
            case DAMP_LIN:  linDamp = 0.5f; break;
            case DAMP_ANG:  angDamp = 0.25f; break;
            case DAMP_BOTH: linDamp = 0.5f; angDamp = 0.25f; break;
            case DAMP_HUGE: linDamp = 137.0f; angDamp = 91.5f; break;
            default: break;
            }
            b.mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW[L] =
                make_float4(linDamp, angDamp, LIN_CAP, ANG_CAP);

            b.mParents[L] = (i == 0u) ? 0u : (i - 1u);
            b.mJointOffset[L] = dofCursor;
            b.mDofs[L] = PxU8(dofsOf(L, i));
            b.mDisableGravity[L] = PxU8(c.disableGravity ? 1 : 0);
            b.mRetainsAcceleration[L] = PxU8(c.retains ? 1 : 0);
            b.mRw_x[L] = fval(L, i, 40) * 2.0f;
            b.mRw_y[L] = fval(L, i, 41) * 5.0f - 0.25f;
            b.mRw_z[L] = fval(L, i, 42) * 3.0f + 0.5f;

            for (uint32_t d = 0; d < dofsOf(L, i); ++d)
            {
                PxgArticulationBlockDofData& dd = dfImg[dofCursor + d];
                dd.mJointVelocities[L] = fval(L, i, 50 + d);
                storeSpatialVector(dd.mWorldMotionMatrix,
                                   Cm::UnAlignedSpatialVector(
                                       PxVec3(fval(L, i, 60 + d), fval(L, i, 61 + d),
                                              fval(L, i, 62 + d)),
                                       PxVec3(fval(L, i, 63 + d), fval(L, i, 64 + d),
                                              fval(L, i, 65 + d))),
                                   L);
            }
            dofCursor += dofsOf(L, i);
        }
        PXG_CHECK_MSG(dofCursor <= MAX_DOFS, "lane %u needs %u dofs", L, dofCursor);
    }

    // ---- device side ----
    PxgDevicePtr dArtis = 0, dProps = 0, dExt = 0, dZa = 0, dLk = 0, dDf = 0, dAb = 0;
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dProps, props.size() * sizeof(PxgArticulationLinkProp));
    shim.memAlloc(&dExt, extAccels.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dZa, zaForces.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(PxgArticulationBlockData));
    if (!dArtis || !dProps || !dExt || !dZa || !dLk || !dDf || !dAb)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<PxgArticulation> recs(LANES);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t L = 0; L < LANES; ++L)
    {
        const size_t off = size_t(L) * MAX_LINKS;
        recs[L].data.numLinks = PxU16(NUM_LINKS[L]);
        recs[L].linkProps = reinterpret_cast<PxgArticulationLinkProp*>(
            dProps + PxgDevicePtr(off * sizeof(PxgArticulationLinkProp)));
        recs[L].externalAccelerations = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dExt + PxgDevicePtr(off * sizeof(Cm::UnAlignedSpatialVector)));
        recs[L].zAForces = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dZa + PxgDevicePtr(off * sizeof(Cm::UnAlignedSpatialVector)));
    }
    shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));
    shim.memcpyHtoD(dProps, props.data(), props.size() * sizeof(PxgArticulationLinkProp));

    // =====================================================================
    // The mirror: the CUDA's control flow with the ENGINE's arithmetic and the
    // ENGINE's stores, writing into a host copy of the same block image.
    // =====================================================================


    size_t bad = 0;
    size_t signedZeros = 0;

    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool extEveryIter = (pass == 1);

        // fresh device state, and a host copy of the very same image
        shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(PxgArticulationBlockData));
        shim.memcpyHtoD(dExt, extAccels.data(),
                        extAccels.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyHtoD(dZa, zaForces.data(), zaForces.size() * sizeof(Cm::UnAlignedSpatialVector));

        std::vector<PxgArticulationBlockLinkData> wantLk = lkImg;
        std::vector<PxgArticulationBlockDofData> wantDf = dfImg;
        std::vector<PxgArticulationBlockData> wantAb = abImg;
        std::vector<Cm::UnAlignedSpatialVector> wantExt = extAccels;
        std::vector<Cm::UnAlignedSpatialVector> wantZa = zaForces;
        for (uint32_t L = 0; L < LANES; ++L)
            asiMirror(wantLk.data(), wantDf.data(), wantAb[0],
                      wantExt.data() + size_t(L) * MAX_LINKS,
                      wantZa.data() + size_t(L) * MAX_LINKS,
                      props.data() + size_t(L) * MAX_LINKS,
                      NUM_LINKS[L], L, GRAVITY, DT, extEveryIter);

        struct Params
        {
            uint64_t artis, artiBlock, linkBlocks, dofBlocks;
            float gx, gy, gz, dt;
            uint32_t nbArticulations, extEveryIter, pad0, pad1;
        } prm = { uint64_t(dArtis), uint64_t(dAb), uint64_t(dLk), uint64_t(dDf),
                  GRAVITY.x,        GRAVITY.y,     GRAVITY.z,     DT,
                  LANES,            extEveryIter ? 1u : 0u, 0u, 0u };
        PXG_CHECK_MSG(sizeof(prm) == 64, "push-constant blob is %zu bytes", sizeof(prm));

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiSpatialInertiaW");
        shim.streamSynchronize();

        std::vector<PxgArticulationBlockLinkData> gotLk(MAX_LINKS);
        std::vector<PxgArticulationBlockDofData> gotDf(MAX_DOFS);
        std::vector<PxgArticulationBlockData> gotAb(1);
        std::vector<Cm::UnAlignedSpatialVector> gotExt(extAccels.size());
        std::vector<Cm::UnAlignedSpatialVector> gotZa(zaForces.size());
        shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(PxgArticulationBlockData));
        shim.memcpyDtoH(gotExt.data(), dExt, gotExt.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyDtoH(gotZa.data(), dZa, gotZa.size() * sizeof(Cm::UnAlignedSpatialVector));

        // ===============================================================
        // THE ONE TOLERANCE, AND EXACTLY HOW WIDE IT IS
        // ===============================================================
        //
        // Compared 4-byte word by 4-byte word, and a mismatch is forgiven ONLY
        // when both words are a zero of some sign -- 0x00000000 against
        // 0x80000000.  Nothing else is tolerated: 0xABABABAB poison against a
        // written zero still fails, and so does every real value.
        //
        // This is not a rounding fudge, it is a measured platform limit.  The
        // engine's accumulators start at a literal zero and add, and IEEE says
        // `(-0) + (+0)` is `+0`.  slangc folds that add away entirely (measured:
        // no OpFAdd survives at -O0), which the shader works around by seeding
        // its accumulators from a runtime zero -- and the emitted SPIR-V then
        // DOES contain the OpFAdd, verified with spirv-dis.  The result is still
        // `-0`, which proves the second fold happens below SPIR-V, in the
        // driver's algebraic pass, where this port has no reach at all.
        //
        // A signed zero survives no arithmetic that matters here: any addition
        // with a non-zero erases it, `-0 == +0` is true, and the only operations
        // that could observe it (a reciprocal's sign, a signbit test) are not
        // applied to these fields downstream.  The count is printed so a change
        // in how many of them occur is visible rather than silent.
        auto cmp = [&](const char* what, const void* g, const void* w, size_t n, size_t stride) {
            const uint8_t* gb = static_cast<const uint8_t*>(g);
            const uint8_t* wb = static_cast<const uint8_t*>(w);
            for (size_t i = 0; i < n; ++i)
            {
                if (gb[i] == wb[i])
                    continue;
                // is the enclosing aligned word a signed zero on both sides?
                const size_t word = (i / 4) * 4;
                bool bothZero = false;
                if (word + 4 <= n)
                {
                    uint32_t gv, wv;
                    std::memcpy(&gv, gb + word, 4);
                    std::memcpy(&wv, wb + word, 4);
                    bothZero = ((gv == 0u || gv == 0x80000000u) &&
                                (wv == 0u || wv == 0x80000000u));
                }
                if (bothZero)
                {
                    ++signedZeros;
                    continue;
                }
                if (bad < 12)
                    std::printf("  [%s] %s +%zu (element %zu, +%zu): got 0x%02X want 0x%02X\n",
                                extEveryIter ? "extEveryIter" : "plain", what, i,
                                stride ? i / stride : 0, stride ? i % stride : i, gb[i], wb[i]);
                ++bad;
            }
        };
        cmp("linkBlocks", gotLk.data(), wantLk.data(),
            gotLk.size() * sizeof(PxgArticulationBlockLinkData),
            sizeof(PxgArticulationBlockLinkData));
        cmp("dofBlocks", gotDf.data(), wantDf.data(),
            gotDf.size() * sizeof(PxgArticulationBlockDofData),
            sizeof(PxgArticulationBlockDofData));
        cmp("articulationBlock", gotAb.data(), wantAb.data(),
            gotAb.size() * sizeof(PxgArticulationBlockData), 0);
        cmp("externalAccelerations", gotExt.data(), wantExt.data(),
            gotExt.size() * sizeof(Cm::UnAlignedSpatialVector),
            sizeof(Cm::UnAlignedSpatialVector));
        cmp("zAForces", gotZa.data(), wantZa.data(),
            gotZa.size() * sizeof(Cm::UnAlignedSpatialVector),
            sizeof(Cm::UnAlignedSpatialVector));
    }

    shim.memFree(dArtis);
    shim.memFree(dProps);
    shim.memFree(dExt);
    shim.memFree(dZa);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  computeSpatialInertiaW: %u lanes x 1/3/5/6/2 links, 5 damping modes x 5 "
                "velocity modes x 2 external-force modes -- bit-exact against the engine's own "
                "transformInertiaTensor and SoA writers (%zu signed-zero byte(s) forgiven, "
                "all on static links)\n", LANES, signedZeros);
}

// ===========================================================================
// P8.85: computeUnconstrainedSpatialInertiaLaunchPartial1T on the ENGINE ABI
// ===========================================================================
//
// The wrapper is twenty-five lines, and this test exists for three things that
// only it can check — plus one that P8.84 could not:
//
//  1. **The descriptor offsets.**  P8.84 handed `gravity`, `dt` and
//     `isExternalForcesEveryTgsIterationEnabled` to the function through push
//     constants, so 52 / 48 / 68 were pinned by `static_assert` and by nothing
//     else.  Here they are read out of a real `PxgArticulationCoreDesc`.
//     `gravity` gets three DISTINCT, non-round components so reading `.y` for
//     `.x` shows; the bool's three padding bytes at 69..71 are poisoned, so a
//     word read where a byte read belongs is caught rather than merely lucky.
//  2. **The per-warp bases**, over five live warp slices in three blocks, with
//     two further slices allocated and left poisoned.
//  3. **Three differences from the velocity kernel's wrapper**: this one does
//     NOT write `solverBodyIndices`; `numLinks` comes from the RECORD with no
//     block fallback; and `artiBlock` advances by ONE block per warp while the
//     two arrays beside it advance by a pitch.
//
// THREE blocks and not two, for the reason P8.83 measured: at two blocks a
// warps-per-block of 1 maps block 1's first warp onto a slice block 0 has
// already filled with the same data, so the duplicated work is idempotent and
// the mutant survives.  At three it leaves a hole — articulations 128 and 129
// are never processed.
//
// The expectation is `asiMirror`, the same function the per-link test above
// uses; it is why that function is file-static rather than a lambda.

static const uint32_t WNB_ARTI = 130;
static const uint32_t WARPS_ALLOC = 7;    // five carry work, two stay poison
static const uint32_t WARPS_USED = 5;
static const uint32_t WMAX_LINKS = 7;     // > every per-articulation count
static const uint32_t WMAX_DOFS = 13;

PXG_TEST(artiSpatialInertiaLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // engine ABI { 8 } -> the descriptor pointer alone
        { "npArtiSpatialInertia.spv", "computeUnconstrainedSpatialInertiaLaunchPartial1T", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiSpatialInertiaLoader);

    // three components that are distinct, non-round and of different signs, so a
    // swapped or shifted read is visible in the first link's gravity term
    const PxVec3 GRAVITY(0.703125f, -9.828125f, 0.390625f);
    const PxReal DT = 1.0f / 50.0f;

    auto fval = [](uint32_t k, uint32_t i, uint32_t f) {
        return float(10 * (k % 23 + 1) + (i + 1)) * 0.125f + float(f) * 0.03125f;
    };

    // ---- the index chain: three hops, none of them the identity ----
    const uint32_t ARTI_OFFSET = 5;
    const uint32_t ISLAND_SLOTS = ARTI_OFFSET + WNB_ARTI + 3;
    const uint32_t BODY_SIM_SLOTS = 192;   // > WNB_ARTI and coprime with 7
    std::vector<uint32_t> NODE(WNB_ARTI), POOL(WNB_ARTI);
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
    {
        NODE[k] = (k * 7u + 11u) % BODY_SIM_SLOTS;   // distinct, never == k
        POOL[k] = (k + 13u) % WNB_ARTI;              // a permutation, never the identity
    }

    std::vector<PxNodeIndex> island(ISLAND_SLOTS);
    for (uint32_t i = 0; i < ISLAND_SLOTS; ++i)
        island[i] = PxNodeIndex(0x0BAD00u + i);
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
        island[ARTI_OFFSET + k] = PxNodeIndex(NODE[k]);

    std::vector<PxgBodySim> bodySims(BODY_SIM_SLOTS);
    std::memset(bodySims.data(), 0xEE, bodySims.size() * sizeof(PxgBodySim));
    for (uint32_t i = 0; i < BODY_SIM_SLOTS; ++i)
        bodySims[i].articulationRemapId = 0xDEAD0000u + i;
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
        bodySims[NODE[k]].articulationRemapId = POOL[k];

    // ---- per-articulation record data, varied as in the per-link test ----
    // link counts cycle over 1..6 and are never equal to the lane index
    auto numLinksOf = [](uint32_t k) { return 1u + ((k * 5u) % 6u); };

    std::vector<PxgArticulationLinkProp> props(size_t(WNB_ARTI) * WMAX_LINKS);
    std::memset(props.data(), 0, props.size() * sizeof(PxgArticulationLinkProp));
    std::vector<Cm::UnAlignedSpatialVector> extAccels(size_t(WNB_ARTI) * WMAX_LINKS);
    std::vector<Cm::UnAlignedSpatialVector> zaForces(size_t(WNB_ARTI) * WMAX_LINKS);
    std::memset(extAccels.data(), 0, extAccels.size() * sizeof(Cm::UnAlignedSpatialVector));
    std::memset(zaForces.data(), 0xCD, zaForces.size() * sizeof(Cm::UnAlignedSpatialVector));

    for (uint32_t k = 0; k < WNB_ARTI; ++k)
        for (uint32_t i = 0; i < numLinksOf(k); ++i)
        {
            const size_t x = size_t(k) * WMAX_LINKS + i;
            if (((k * 7u + i) % 11u) == 3u)   // a static link, as in the per-link test
            {
                props[x].invInertia = PxVec3(0.0f, 0.0f, 0.0f);
                props[x].invMass = 0.0f;
            }
            else
            {
                props[x].invInertia = PxVec3(fval(k, i, 1), fval(k, i, 2), fval(k, i, 3));
                props[x].invMass = fval(k, i, 4);
            }
            extAccels[x] = Cm::UnAlignedSpatialVector(
                PxVec3(fval(k, i, 10), fval(k, i, 11), fval(k, i, 12)),
                PxVec3(fval(k, i, 13), fval(k, i, 14), fval(k, i, 15)));
        }

    // ---- the block images, one slice per allocated warp ----
    std::vector<PxgArticulationBlockLinkData> lkImg(size_t(WARPS_ALLOC) * WMAX_LINKS);
    std::vector<PxgArticulationBlockDofData> dfImg(size_t(WARPS_ALLOC) * WMAX_DOFS);
    std::vector<PxgArticulationBlockData> abImg(WARPS_ALLOC);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(PxgArticulationBlockData));

    for (uint32_t k = 0; k < WNB_ARTI; ++k)
    {
        const uint32_t warp = k / 32u, lane = k % 32u;
        PxgArticulationBlockLinkData* lk = lkImg.data() + size_t(warp) * WMAX_LINKS;
        PxgArticulationBlockDofData* df = dfImg.data() + size_t(warp) * WMAX_DOFS;

        // the block header says a DIFFERENT link count from the record.  The
        // wrapper must ignore it: unlike the velocity kernel there is no refill
        // guard here and numLinks comes from the record, always.
        abImg[warp].mNumLinks[lane] = 0xFEEDu;

        uint32_t dofCursor = 0;
        for (uint32_t i = 0; i < numLinksOf(k); ++i)
        {
            PxgArticulationBlockLinkData& b = lk[i];
            const PxQuat q = PxQuat(fval(k, i, 20), fval(k, i, 21), fval(k, i, 22),
                                    fval(k, i, 23)).getNormalized();
            const PxVec3 p(fval(k, i, 30) * 3.0f + 0.75f * float(i * i),
                           fval(k, i, 31) * 11.0f - 0.4f * float(i),
                           fval(k, i, 32) * 7.0f + 1.5f * float(i));
            storeSpatialTransform(b.mAccumulatedPose, lane, PxTransform(p, q));

            const uint32_t vm = (k * 5u + i) % NB_VEL_MODES;
            PxVec3 angVel(0.25f, -0.5f, 0.125f), linVel(0.5f, 0.75f, -0.25f);
            switch (VelMode(vm))
            {
            case VEL_OVER_ANG:   angVel = PxVec3(0.0f, -0.25f, 1.25f); break;
            case VEL_OVER_LIN:   linVel = PxVec3(3.0f, -1.5f, 1.25f); break;
            case VEL_OVER_BOTH:  angVel = PxVec3(3.0f, 2.0f, 1.5f);
                                 linVel = PxVec3(4.0f, -3.5f, 2.25f); break;
            case VEL_ZERO_ANG:   angVel = PxVec3(0.0f, 0.0f, 0.0f); break;
            case VEL_RECIP:      angVel = PxVec3(0.125f, -0.125f, 0.75f); break;
            default: break;
            }
            storeSpatialVector(b.mMotionVelocity, Cm::UnAlignedSpatialVector(angVel, linVel), lane);

            PxReal linDamp = 0.0f, angDamp = 0.0f;
            switch (DampMode((k * 3u + i) % 5u))
            {
            case DAMP_LIN:  linDamp = 0.5f; break;
            case DAMP_ANG:  angDamp = 0.25f; break;
            case DAMP_BOTH: linDamp = 0.5f; angDamp = 0.25f; break;
            case DAMP_HUGE: linDamp = 137.0f; angDamp = 91.5f; break;
            default: break;
            }
            b.mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW[lane] =
                make_float4(linDamp, angDamp, LIN_CAP, ANG_CAP);

            b.mParents[lane] = (i == 0u) ? 0u : (i - 1u);
            b.mJointOffset[lane] = dofCursor;
            const uint32_t nd = (i == 0u) ? 0u : (1u + ((k + i) % 3u));
            b.mDofs[lane] = PxU8(nd);
            b.mDisableGravity[lane] = PxU8(((k + i) & 1u) ? 1 : 0);
            b.mRetainsAcceleration[lane] = PxU8(((k + i) & 2u) ? 1 : 0);
            b.mRw_x[lane] = fval(k, i, 40) * 2.0f;
            b.mRw_y[lane] = fval(k, i, 41) * 5.0f - 0.25f;
            b.mRw_z[lane] = fval(k, i, 42) * 3.0f + 0.5f;

            for (uint32_t d = 0; d < nd; ++d)
            {
                PxgArticulationBlockDofData& dd = df[dofCursor + d];
                dd.mJointVelocities[lane] = fval(k, i, 50 + d);
                storeSpatialVector(dd.mWorldMotionMatrix,
                                   Cm::UnAlignedSpatialVector(
                                       PxVec3(fval(k, i, 60 + d), fval(k, i, 61 + d),
                                              fval(k, i, 62 + d)),
                                       PxVec3(fval(k, i, 63 + d), fval(k, i, 64 + d),
                                              fval(k, i, 65 + d))),
                                   lane);
            }
            dofCursor += nd;
        }
        PXG_CHECK_MSG(dofCursor <= WMAX_DOFS, "articulation %u needs %u dofs", k, dofCursor);
    }

    // ---- device side ----
    PxgDevicePtr dIsland = 0, dBodySims = 0, dSolverBodyIdx = 0, dDesc = 0, dArtis = 0;
    PxgDevicePtr dProps = 0, dExt = 0, dZa = 0, dLk = 0, dDf = 0, dAb = 0;
    shim.memAlloc(&dIsland, island.size() * sizeof(PxNodeIndex));
    shim.memAlloc(&dBodySims, bodySims.size() * sizeof(PxgBodySim));
    shim.memAlloc(&dSolverBodyIdx, size_t(BODY_SIM_SLOTS) * sizeof(PxU32));
    shim.memAlloc(&dDesc, sizeof(PxgArticulationCoreDesc));
    shim.memAlloc(&dArtis, size_t(WNB_ARTI) * sizeof(PxgArticulation));
    shim.memAlloc(&dProps, props.size() * sizeof(PxgArticulationLinkProp));
    shim.memAlloc(&dExt, extAccels.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dZa, zaForces.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(PxgArticulationBlockData));
    if (!dIsland || !dBodySims || !dSolverBodyIdx || !dDesc || !dArtis || !dProps || !dExt ||
        !dZa || !dLk || !dDf || !dAb)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<PxgArticulation> recs(WNB_ARTI);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
    {
        PxgArticulation& r = recs[POOL[k]];
        const size_t off = size_t(k) * WMAX_LINKS;
        r.data.numLinks = PxU16(numLinksOf(k));
        r.linkProps = reinterpret_cast<PxgArticulationLinkProp*>(
            dProps + PxgDevicePtr(off * sizeof(PxgArticulationLinkProp)));
        r.externalAccelerations = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dExt + PxgDevicePtr(off * sizeof(Cm::UnAlignedSpatialVector)));
        r.zAForces = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dZa + PxgDevicePtr(off * sizeof(Cm::UnAlignedSpatialVector)));
    }

    const std::vector<PxU32> sbiImg(BODY_SIM_SLOTS, 0xABABABABu);

    size_t bad = 0, signedZeros = 0;

    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool extEveryIter = (pass == 1);

        shim.memcpyHtoD(dIsland, island.data(), island.size() * sizeof(PxNodeIndex));
        shim.memcpyHtoD(dBodySims, bodySims.data(), bodySims.size() * sizeof(PxgBodySim));
        shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));
        shim.memcpyHtoD(dProps, props.data(), props.size() * sizeof(PxgArticulationLinkProp));
        shim.memcpyHtoD(dExt, extAccels.data(),
                        extAccels.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyHtoD(dZa, zaForces.data(), zaForces.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(PxgArticulationBlockData));
        shim.memcpyHtoD(dSolverBodyIdx, sbiImg.data(), sbiImg.size() * sizeof(PxU32));

        // ---- the descriptor.  0x5A everywhere first, so the three bytes after
        // the bool (69..71, padding before `impulses` at 72) are NOT zero: a word
        // read where a byte read belongs then yields a non-zero flag when the
        // flag is false, and the pass-0 expectation catches it.
        PxgArticulationCoreDesc desc;
        std::memset(&desc, 0x5A, sizeof(desc));
        desc.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dBodySims);
        desc.islandNodeIndices = reinterpret_cast<PxNodeIndex*>(dIsland);
        desc.solverBodyIndices = reinterpret_cast<PxU32*>(dSolverBodyIdx);
        desc.articulations = reinterpret_cast<PxgArticulation*>(dArtis);
        desc.articulationOffset = ARTI_OFFSET;
        desc.nbArticulations = WNB_ARTI;
        desc.dt = DT;
        desc.gravity = GRAVITY;
        desc.isExternalForcesEveryTgsIterationEnabled = extEveryIter;
        desc.mArticulationBlocks = reinterpret_cast<PxgArticulationBlockData*>(dAb);
        desc.mArticulationLinkBlocks = reinterpret_cast<PxgArticulationBlockLinkData*>(dLk);
        desc.mArticulationDofBlocks = reinterpret_cast<PxgArticulationBlockDofData*>(dDf);
        desc.mMaxLinksPerArticulation = WMAX_LINKS;
        desc.mMaxDofsPerArticulation = WMAX_DOFS;
        shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

        // ---- the expectation: the same mirror the per-link test uses ----
        std::vector<PxgArticulationBlockLinkData> wantLk = lkImg;
        std::vector<PxgArticulationBlockDofData> wantDf = dfImg;
        std::vector<PxgArticulationBlockData> wantAb = abImg;
        std::vector<Cm::UnAlignedSpatialVector> wantExt = extAccels;
        std::vector<Cm::UnAlignedSpatialVector> wantZa = zaForces;
        for (uint32_t k = 0; k < WNB_ARTI; ++k)
        {
            const uint32_t warp = k / 32u, lane = k % 32u;
            asiMirror(wantLk.data() + size_t(warp) * WMAX_LINKS,
                      wantDf.data() + size_t(warp) * WMAX_DOFS,
                      wantAb[warp],
                      wantExt.data() + size_t(k) * WMAX_LINKS,
                      wantZa.data() + size_t(k) * WMAX_LINKS,
                      props.data() + size_t(k) * WMAX_LINKS,
                      numLinksOf(k), lane, GRAVITY, DT, extEveryIter);
        }

        struct Params { uint64_t desc; } prm = { uint64_t(dDesc) };
        const uint32_t numBlocks = (WNB_ARTI + 63u) / 64u;
        PXG_CHECK_MSG(numBlocks == 3u, "fixture assumes three blocks, got %u", numBlocks);
        PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch computeUnconstrainedSpatialInertiaLaunchPartial1T");
        shim.streamSynchronize();

        std::vector<PxgArticulationBlockLinkData> gotLk(lkImg.size());
        std::vector<PxgArticulationBlockDofData> gotDf(dfImg.size());
        std::vector<PxgArticulationBlockData> gotAb(abImg.size());
        std::vector<Cm::UnAlignedSpatialVector> gotExt(extAccels.size());
        std::vector<Cm::UnAlignedSpatialVector> gotZa(zaForces.size());
        std::vector<PxU32> gotSbi(BODY_SIM_SLOTS);
        shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(PxgArticulationBlockData));
        shim.memcpyDtoH(gotExt.data(), dExt, gotExt.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyDtoH(gotZa.data(), dZa, gotZa.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyDtoH(gotSbi.data(), dSolverBodyIdx, gotSbi.size() * sizeof(PxU32));

        auto cmp = [&](const char* what, const void* g, const void* w, size_t n, size_t stride) {
            const uint8_t* gb = static_cast<const uint8_t*>(g);
            const uint8_t* wb = static_cast<const uint8_t*>(w);
            for (size_t i = 0; i < n; ++i)
            {
                if (gb[i] == wb[i])
                    continue;
                const size_t word = (i / 4) * 4;
                bool bothZero = false;
                if (word + 4 <= n)
                {
                    uint32_t gv, wv;
                    std::memcpy(&gv, gb + word, 4);
                    std::memcpy(&wv, wb + word, 4);
                    bothZero = ((gv == 0u || gv == 0x80000000u) &&
                                (wv == 0u || wv == 0x80000000u));
                }
                if (bothZero) { ++signedZeros; continue; }
                if (bad < 12)
                    std::printf("  [%s] %s +%zu (element %zu, +%zu): got 0x%02X want 0x%02X\n",
                                extEveryIter ? "extEveryIter" : "plain", what, i,
                                stride ? i / stride : 0, stride ? i % stride : i, gb[i], wb[i]);
                ++bad;
            }
        };
        cmp("linkBlocks", gotLk.data(), wantLk.data(),
            gotLk.size() * sizeof(PxgArticulationBlockLinkData),
            sizeof(PxgArticulationBlockLinkData));
        cmp("dofBlocks", gotDf.data(), wantDf.data(),
            gotDf.size() * sizeof(PxgArticulationBlockDofData),
            sizeof(PxgArticulationBlockDofData));
        cmp("articulationBlocks", gotAb.data(), wantAb.data(),
            gotAb.size() * sizeof(PxgArticulationBlockData), sizeof(PxgArticulationBlockData));
        cmp("externalAccelerations", gotExt.data(), wantExt.data(),
            gotExt.size() * sizeof(Cm::UnAlignedSpatialVector),
            sizeof(Cm::UnAlignedSpatialVector));
        cmp("zAForces", gotZa.data(), wantZa.data(),
            gotZa.size() * sizeof(Cm::UnAlignedSpatialVector),
            sizeof(Cm::UnAlignedSpatialVector));

        // this kernel does NOT touch solverBodyIndices -- unlike the velocity
        // kernel's wrapper, which writes it on every dispatch
        for (uint32_t i = 0; i < BODY_SIM_SLOTS; ++i)
            if (gotSbi[i] != 0xABABABABu)
            {
                if (bad < 12)
                    std::printf("  [%s] solverBodyIndices[%u] written: 0x%08X\n",
                                extEveryIter ? "extEveryIter" : "plain", i, gotSbi[i]);
                ++bad;
            }
    }

    shim.memFree(dIsland);
    shim.memFree(dBodySims);
    shim.memFree(dSolverBodyIdx);
    shim.memFree(dDesc);
    shim.memFree(dArtis);
    shim.memFree(dProps);
    shim.memFree(dExt);
    shim.memFree(dZa);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  computeUnconstrainedSpatialInertiaLaunchPartial1T: %u articulations over %u "
                "warp slices in 3 blocks (32x2), %u allocated-and-untouched, descriptor gravity/dt/"
                "bool read for real, solverBodyIndices untouched (%zu signed-zero byte(s) "
                "forgiven)\n",
                WNB_ARTI, WARPS_USED, WARPS_ALLOC - WARPS_USED, signedZeros);
}

} // namespace
