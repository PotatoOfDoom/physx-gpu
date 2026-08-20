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

// testPsFluidVelocity.cpp -- P9.1d fixture for the PBD velocity pair,
// ps_vorticityConfinementLaunch (particlesystem.cu:3757) and
// ps_solveVelocityLaunch (:3891), ported as npPsFluidVelocity.slang.
//
// The two are tested CHAINED, in the order the engine dispatches them: vorticity
// writes mCurl, then solveVelocity reads that same mCurl.  The host expectation is
// chained the same way -- refSolveVelocity consumes the curl array refVorticity
// produced -- so the handoff between the two kernels is part of what is verified,
// not assumed.
//
// ORACLE: host transcription, carried by generated offsets, whole-region 0xAB
// poison compare over both outputs, and the sweep.
//
// GEOMETRY: four systems, THREE active with active = { 2, 0, 3 } (not a
// permutation) and a DECOY at pool index 1 whose outputs must stay entirely
// poisoned.  A has 500 particles so a halved block stride leaves a hole; every
// per-particle array is PAD = 544 with a poisoned tail, and the padded particles
// are fluid with a positive vorticityConfinement and ZERO contacts, so a mutant
// that widens the `p < numParticles` guard writes zeros into the tail and dies.
// NOTE these two kernels have NO eDISABLE_SELF_COLLISION guard, unlike the density
// pair, so there is no disabled-system case to carry here.
//
// MATERIALS carry the three vorticityConfinement regimes that trap V1 turns on:
//   m0: vc == 0    -- vorticity early-outs (its test is `== 0`), solveVelocity
//                     early-outs too (`<= 0`)
//   m1: vc == 0.8  -- both proceed
//   m2: vc == -0.5 -- vorticity PROCEEDS (== 0 is false!) while solveVelocity
//                     early-outs.  Without a negative material the two predicates
//                     are indistinguishable.
//
// PER-PARTICLE WITNESSES in A:
//   0  zero contacts                 -- curl written as all-zero, delta all-zero
//   1  a neighbour at EXACTLY the    -- the guard is `<=` here, the opposite of the
//      contact distance                density kernels' strict `<`, so it must be
//                                      INCLUDED
//   2  a COINCIDENT neighbour        -- dSq == 0, excluded by the `> 0` half
//   3  a NON-FLUID neighbour         -- vorticity `continue`s past it AFTER its
//                                      prefetch, which is what keeps the velocity
//                                      lag indexed by raw neighbour rather than by
//                                      accepted neighbour
//   4  same group, DIFFERENT phase   -- solveVelocity requires phase == phasej on
//      word (bit 20 set)               the WHOLE word, so this pair contributes
//                                      nothing to the gradient
//   5  non-fluid with vc > 0         -- solveVelocity skips the whole loop and
//                                      stores a zero impulse (trap V11)
//
// THE BUG ASSERTION for upstream candidate 12, the same shape P9.1b used: the
// expectation is recomputed with the CORRECTED velocity index and the test requires
// (a) that the two differ at all and (b) that no GPU word matches the corrected
// form where they differ.  A port that silently fixed the lag fails.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"

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
struct V3 { float x, y, z; };

static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 mul3(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

struct VcPush { uint64_t particleSystems, activeParticleSystems; };
static_assert(sizeof(VcPush) == 16, "kTwoPtr blob is 16 bytes");
struct SvPush { uint64_t particleSystems, activeParticleSystems; float dt; uint32_t _pad; };
static_assert(sizeof(SvPush) == 24, "kPsUpdateUserBuffer blob is 24 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;
static const uint32_t PS_PHASE_FLUID = 0x00400000u;
static const uint32_t PS_PHASE_SELF_COLLIDE = 0x00100000u;
static const uint32_t PS_PHASE_GROUP_MASK = 0x000fffffu;

static float dWdxH(float x, float k2, float kInv) { return -k2 * (1.0f - x * kInv); }

enum : uint32_t { MF_VORTICITY = 7, MF_COUNT = 15 };

struct Sys
{
    uint32_t numParticles = 0;
    float spiky2 = 0.f, invRestDensity = 0.f;
    float contactDistInv = 0.f, contactDistSq = 0.f;
    uint32_t materialStride = 0;

    std::vector<F4> pos, vel;
    std::vector<uint32_t> phase, collisionIndex, contactCount;
    std::vector<float> materials;
    std::vector<uint16_t> phaseToMat;

    DevBuf bPos, bVel, bPhase, bColl, bCount, bMaterials, bP2m;
    DevBuf bCurl, bDelta;
};

// particlesystem.cu:3790-3887
static void refVorticity(const Sys& s, std::vector<uint32_t>& outCurl,
                         std::vector<F4>& curlValues, bool forceCorrectVelocityIndex)
{
    const uint32_t N = s.numParticles;
    for (uint32_t p = 0; p < N; ++p)
    {
        const uint32_t phase = s.phase[p];
        const uint32_t group = phase & PS_PHASE_GROUP_MASK;
        const float vc = s.materials[size_t(s.phaseToMat[group]) * MF_COUNT + MF_VORTICITY];

        if (((phase & PS_PHASE_FLUID) == 0u) || vc == 0.0f)      // V1: strict ==
        {
            for (uint32_t k = 0; k < 4u; ++k) outCurl[p * 4u + k] = f2u(0.0f);
            curlValues[p] = { 0.f, 0.f, 0.f, 0.f };
            continue;
        }

        const V3 xi = { s.pos[p].x, s.pos[p].y, s.pos[p].z };
        const V3 vi = { s.vel[p].x, s.vel[p].y, s.vel[p].z };
        const uint32_t cc = s.contactCount[p];

        V3 vorticity = { 0.f, 0.f, 0.f };
        for (uint32_t c = 0; c < cc; ++c)
        {
            const uint32_t qPos = s.collisionIndex[p + c * N];
            const uint32_t iVel = forceCorrectVelocityIndex ? c : ((c > 0u) ? (c - 1u) : 0u);
            const uint32_t qVel = s.collisionIndex[p + iVel * N];

            const uint32_t phasej = s.phase[qPos];
            if ((phasej & PS_PHASE_FLUID) == 0u) continue;        // V4

            const V3 xj = { s.pos[qPos].x, s.pos[qPos].y, s.pos[qPos].z };
            const V3 xij = sub3(xi, xj);
            const float dSq = dot3(xij, xij);

            if (dSq <= s.contactDistSq && dSq > 0.0f)             // V3: inclusive
            {
                const V3 vj = { s.vel[qVel].x, s.vel[qVel].y, s.vel[qVel].z };
                const V3 vij = sub3(vj, vi);                      // V6
                const float d = std::sqrt(dSq);
                const V3 dw = mul3(mul3(xij, 1.0f / d),
                                   dWdxH(d, s.spiky2, s.contactDistInv));   // V5
                const V3 cr = cross3(vij, dw);
                vorticity.x += cr.x; vorticity.y += cr.y; vorticity.z += cr.z;
            }
        }
        const float mag = std::sqrt(dot3(vorticity, vorticity));  // V6
        outCurl[p * 4u + 0u] = f2u(vorticity.x);
        outCurl[p * 4u + 1u] = f2u(vorticity.y);
        outCurl[p * 4u + 2u] = f2u(vorticity.z);
        outCurl[p * 4u + 3u] = f2u(mag);
        curlValues[p] = { vorticity.x, vorticity.y, vorticity.z, mag };
    }
}

// particlesystem.cu:3925-4040
static void refSolveVelocity(const Sys& s, float dt, const std::vector<F4>& curls,
                             std::vector<uint32_t>& outDelta)
{
    const uint32_t N = s.numParticles;
    for (uint32_t p = 0; p < N; ++p)
    {
        const uint32_t phase = s.phase[p];
        const uint32_t group = phase & PS_PHASE_GROUP_MASK;
        const float vc = s.materials[size_t(s.phaseToMat[group]) * MF_COUNT + MF_VORTICITY];

        if (vc <= 0.0f)                                           // V1: `<=` here
        {
            for (uint32_t k = 0; k < 4u; ++k) outDelta[p * 4u + k] = f2u(0.0f);
            continue;
        }

        const V3 xi = { s.pos[p].x, s.pos[p].y, s.pos[p].z };
        const F4 curl4 = curls[p];

        V3 impulse = { 0.f, 0.f, 0.f };
        V3 vorticityGrad = { 0.f, 0.f, 0.f };
        float weight = 0.f;

        if ((phase & PS_PHASE_FLUID) != 0u)                       // V11
        {
            for (uint32_t c = 0; c < s.contactCount[p]; ++c)
            {
                const uint32_t q = s.collisionIndex[p + c * N];
                const uint32_t phasej = s.phase[q];
                const F4 curlN = curls[q];

                const V3 xj = { s.pos[q].x, s.pos[q].y, s.pos[q].z };
                const V3 xij = sub3(xi, xj);
                const float dSq = dot3(xij, xij);

                if (dSq <= s.contactDistSq && dSq > 0.0f)         // V3
                {
                    if (phase == phasej)                          // V8: whole word
                    {
                        const float d = std::sqrt(dSq);
                        const V3 n = mul3(xij, 1.0f / d);         // V5
                        const V3 dw = mul3(n, dWdxH(d, s.spiky2, s.contactDistInv));
                        vorticityGrad.x += curlN.w * dw.x;        // V7: the .w
                        vorticityGrad.y += curlN.w * dw.y;
                        vorticityGrad.z += curlN.w * dw.z;
                        weight += 1.0f;
                    }
                }
            }

            const V3 curl3 = { curl4.x, curl4.y, curl4.z };
            const float gradMagSq = dot3(vorticityGrad, vorticityGrad);
            V3 nGrad = { 0.f, 0.f, 0.f };
            if (gradMagSq > 0.0f)                                 // V9
                nGrad = mul3(vorticityGrad, 1.0f / std::sqrt(gradMagSq));

            const V3 cr = cross3(nGrad, curl3);
            const float k = dt * s.invRestDensity * vc;           // V10, left to right
            impulse.x += k * cr.x; impulse.y += k * cr.y; impulse.z += k * cr.z;
        }

        outDelta[p * 4u + 0u] = f2u(impulse.x);
        outDelta[p * 4u + 1u] = f2u(impulse.y);
        outDelta[p * 4u + 2u] = f2u(impulse.z);
        outDelta[p * 4u + 3u] = f2u(weight);
    }
}

static float jitter(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return float(h % 20011u) * (1.0f / 20011.0f);
}

PXG_TEST(psFluidVelocity)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t BLOCK = 256u, MN = 6u, PAD = 544u;
    const uint32_t NUM_GROUPS = 3u, NUM_MATERIALS = 3u;
    const float CONTACT_DIST = 0.5f;

    auto buildSys = [&](uint32_t N, float spiky2, float invRestDensity,
                        uint32_t salt, bool withSpecials)
    {
        Sys s;
        s.numParticles = N; s.spiky2 = spiky2; s.invRestDensity = invRestDensity;
        s.contactDistInv = 1.0f / CONTACT_DIST;
        s.contactDistSq = CONTACT_DIST * CONTACT_DIST;
        s.materialStride = MF_COUNT * 4u;

        s.pos.resize(PAD); s.vel.resize(PAD); s.phase.resize(PAD);
        s.contactCount.resize(PAD);
        s.collisionIndex.assign(size_t(N) * MN, 0u);

        for (uint32_t p = 0; p < PAD; ++p)
        {
            s.pos[p] = { jitter(p, salt + 1u) * 1.2f, jitter(p, salt + 2u) * 1.2f,
                         jitter(p, salt + 3u) * 1.2f, 0.5f + jitter(p, salt + 4u) };
            s.vel[p] = { jitter(p, salt + 5u) * 2.f - 1.f, jitter(p, salt + 6u) * 2.f - 1.f,
                         jitter(p, salt + 7u) * 2.f - 1.f, 0.f };
            const uint32_t grp = p % NUM_GROUPS;
            const bool fluid = (p % 11u != 5u);
            s.phase[p] = grp | (fluid ? PS_PHASE_FLUID : 0u);
            // The padded tail is fluid with a POSITIVE-vorticity material (group 1
            // maps to m1) and zero contacts, so a widened guard writes zeros there.
            s.contactCount[p] = (p < N) ? (1u + (p % (MN - 1u))) : 0u;
            if (p >= N) s.phase[p] = 1u | PS_PHASE_FLUID;
        }
        for (uint32_t p = 0; p < N; ++p)
            for (uint32_t i = 0; i < MN; ++i)
                s.collisionIndex[p + i * N] = (p * 31u + i * 67u + 7u) % N;

        if (withSpecials)
        {
            s.phase[0] = 1u | PS_PHASE_FLUID;  s.contactCount[0] = 0u;
            // 1: a neighbour at EXACTLY the contact distance (0.75-0.25 == 0.5,
            //    0.5*0.5 == 0.25 == contactDistSq, all exact) -- the `<=` guard
            //    must INCLUDE it.
            s.phase[1] = 1u | PS_PHASE_FLUID;  s.contactCount[1] = 1u;
            s.collisionIndex[1 + 0u * N] = 40u;
            s.phase[40] = 1u | PS_PHASE_FLUID;
            s.pos[1] = { 0.75f, 0.4f, 0.5f, 0.8f };
            s.pos[40] = { 0.25f, 0.4f, 0.5f, 0.8f };
            // 2: a COINCIDENT neighbour -- dSq == 0, excluded by the `> 0` half
            s.phase[2] = 1u | PS_PHASE_FLUID;  s.contactCount[2] = 2u;
            s.collisionIndex[2 + 0u * N] = 41u;
            s.phase[41] = 1u | PS_PHASE_FLUID;
            s.pos[41] = s.pos[2];
            // 3: a NON-FLUID neighbour, which vorticity `continue`s past
            s.phase[3] = 1u | PS_PHASE_FLUID;  s.contactCount[3] = 3u;
            s.collisionIndex[3 + 0u * N] = 42u;
            s.phase[42] = 1u;                              // non-fluid
            s.pos[42] = { s.pos[3].x + 0.15f, s.pos[3].y, s.pos[3].z, 0.8f };
            // 4: same GROUP but a different PHASE WORD (bit 20) -- solveVelocity's
            //    phase == phasej must reject it
            s.phase[4] = 1u | PS_PHASE_FLUID | PS_PHASE_SELF_COLLIDE;
            s.contactCount[4] = 1u;
            s.collisionIndex[4 + 0u * N] = 43u;
            s.phase[43] = 1u | PS_PHASE_FLUID;
            s.pos[43] = { s.pos[4].x + 0.16f, s.pos[4].y, s.pos[4].z, 0.8f };
            // 5: NON-fluid with a positive-vorticity material (trap V11), AND an
            // in-range neighbour carrying the IDENTICAL phase word.  Both halves
            // are needed: without the matching neighbour, dropping the fluid gate
            // around solveVelocity's loop is an identity, because the loop's own
            // `phase == phasej` test then rejects every pair anyway.  With it, the
            // mutant reaches `weight += 1` and the stored weight separates them --
            // the gradient itself cannot, since a non-fluid particle's curl is
            // zeroed by the vorticity kernel and 0 * dw contributes nothing.
            s.phase[5] = 1u;                               // group 1 -> m1, vc > 0
            s.contactCount[5] = 1u;
            s.collisionIndex[5 + 0u * N] = 44u;
            s.phase[44] = 1u;                              // the SAME phase word
            s.pos[44] = { s.pos[5].x + 0.17f, s.pos[5].y, s.pos[5].z, 0.8f };
        }

        s.materials.assign(NUM_MATERIALS * MF_COUNT, 0.0f);
        for (uint32_t m = 0; m < NUM_MATERIALS; ++m)
        {
            float* mm = s.materials.data() + size_t(m) * MF_COUNT;
            for (uint32_t f = 0; f < MF_COUNT; ++f) mm[f] = -800.0f - float(m) * 40.f - float(f);
            // V1's three regimes.  A NEGATIVE value is what separates the two
            // kernels' predicates (`== 0` vs `<= 0`); without it they agree.
            mm[MF_VORTICITY] = (m == 0u) ? 0.0f : ((m == 1u) ? 0.8f : -0.5f);
        }
        s.phaseToMat.assign(NUM_GROUPS, 0u);
        s.phaseToMat[0] = 2u;   // group 0 -> the NEGATIVE material
        s.phaseToMat[1] = 1u;   // group 1 -> the positive one
        s.phaseToMat[2] = 0u;   // group 2 -> zero
        return s;
    };

    Sys sysA = buildSys(500u, 2.4f, 0.85f, 100u, true);
    Sys sysD = buildSys(200u, 1.9f, 1.15f, 200u, false);
    Sys sysB = buildSys(130u, 3.1f, 0.65f, 300u, false);
    Sys sysC = buildSys(90u, 1.5f, 1.35f, 400u, false);

    const uint32_t GRID_X = (sysA.numParticles + BLOCK - 1u) / BLOCK;
    PXG_CHECK_MSG(GRID_X == 2u, "two blocks needed (got %u)", GRID_X);
    PXG_CHECK_MSG(sysA.numParticles > 128u * (GRID_X - 1u) + 255u, "A must outreach a halved stride");
    PXG_CHECK_MSG(PAD > GRID_X * BLOCK, "the tail must outreach the grid");

    Sys* order[4] = { &sysA, &sysD, &sysB, &sysC };
    for (Sys* s : order)
    {
        s->bPos = makeDev(dev, s->pos);
        s->bVel = makeDev(dev, s->vel);
        s->bPhase = makeDev(dev, s->phase);
        s->bColl = makeDev(dev, s->collisionIndex);
        s->bCount = makeDev(dev, s->contactCount);
        s->bMaterials = makeDev(dev, s->materials);
        s->bP2m = makeDev(dev, s->phaseToMat);
        std::vector<uint32_t> c(size_t(PAD) * 4u, POISON_U32);
        std::vector<uint32_t> d(size_t(PAD) * 4u, POISON_U32);
        s->bCurl = makeDev(dev, c);
        s->bDelta = makeDev(dev, d);
    }

    std::vector<uint32_t> active = { 2u, 0u, 3u };
    DevBuf bActive = makeDev(dev, active);

    std::vector<PxgParticleSystem> descArr;
    for (Sys* s : order)
    {
        PxgParticleSystem ps;
        std::memset(&ps, 0xEE, sizeof(ps));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s->bPos.addr));
        ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(s->bVel.addr));
        ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(s->bPhase.addr));
        ps.mCollisionIndex = reinterpret_cast<PxU32*>(size_t(s->bColl.addr));
        ps.mParticleSelfCollisionCount = reinterpret_cast<PxU32*>(size_t(s->bCount.addr));
        ps.mParticleMaterials = reinterpret_cast<PxsParticleMaterialData*>(size_t(s->bMaterials.addr));
        ps.mParticleMaterialStride = s->materialStride;
        ps.mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(s->bP2m.addr));
        ps.mCurl = reinterpret_cast<float4*>(size_t(s->bCurl.addr));
        ps.mDelta = reinterpret_cast<float4*>(size_t(s->bDelta.addr));
        ps.mCommonData.mNumParticles = s->numParticles;
        ps.mCommonData.mParticleContactDistanceInv = s->contactDistInv;
        ps.mCommonData.mParticleContactDistanceSq = s->contactDistSq;
        ps.mData.mSpiky2 = s->spiky2;
        ps.mData.mInvRestDensity = s->invRestDensity;
        descArr.push_back(ps);
    }
    DevBuf bDesc = makeDev(dev, descArr);

    const float dt = 1.0f / 75.0f;

    // CHAINED, in the engine's dispatch order.
    {
        VcPush vp = { bDesc.addr, bActive.addr };
        Kernel kv("npPsFluidVelocity.spv", "ps_vorticityConfinementLaunch", {}, sizeof(VcPush));
        kv.dispatch(GRID_X, {}, vp, uint32_t(active.size()), 1u);
        dev->flush();
    }
    {
        SvPush sp = {}; sp.particleSystems = bDesc.addr;
        sp.activeParticleSystems = bActive.addr; sp.dt = dt;
        Kernel ks("npPsFluidVelocity.spv", "ps_solveVelocityLaunch", {}, sizeof(SvPush));
        ks.dispatch(GRID_X, {}, sp, uint32_t(active.size()), 1u);
        dev->flush();
    }

    size_t mm = 0;
    bool sawBugWitness = false;
    for (Sys* s : order)
    {
        std::vector<uint32_t> rc(size_t(PAD) * 4u, POISON_U32);
        std::vector<uint32_t> rd(size_t(PAD) * 4u, POISON_U32);
        std::vector<F4> curlValues(PAD, F4{ 0.f, 0.f, 0.f, 0.f });

        const bool isActive = (s != &sysD);
        if (isActive)
        {
            refVorticity(*s, rc, curlValues, /*forceCorrect*/ false);
            refSolveVelocity(*s, dt, curlValues, rd);
        }

        const auto gc = downloadBuffer<uint32_t>(dev, s->bCurl.h, PAD * 4u);
        const auto gd = downloadBuffer<uint32_t>(dev, s->bDelta.h, PAD * 4u);

        auto cmp = [&](const std::vector<uint32_t>& r, const std::vector<uint32_t>& g,
                       const char* what)
        {
            for (size_t w = 0; w < r.size(); ++w)
                if (r[w] != g[w])
                {
                    if (mm < 12)
                    {
                        float rf, gf;
                        std::memcpy(&rf, &r[w], 4); std::memcpy(&gf, &g[w], 4);
                        std::printf("  sys %p %s MM %zu (particle %zu comp %zu): ref %g got %g\n",
                                    (void*)s, what, w, w / 4u, w % 4u, rf, gf);
                    }
                    ++mm;
                }
        };
        cmp(rc, gc, "curl");
        cmp(rd, gd, "delta");

        // --- the bug assertion for candidate 12, on A -----------------------
        if (s == &sysA)
        {
            std::vector<uint32_t> cc(size_t(PAD) * 4u, POISON_U32);
            std::vector<F4> ccVals(PAD, F4{ 0.f, 0.f, 0.f, 0.f });
            refVorticity(*s, cc, ccVals, /*forceCorrect*/ true);
            size_t agreesWithFixed = 0;
            for (size_t w = 0; w < cc.size(); ++w)
            {
                if (cc[w] != rc[w])
                {
                    sawBugWitness = true;
                    if (cc[w] == gc[w]) ++agreesWithFixed;
                }
            }
            PXG_CHECK_MSG(sawBugWitness,
                          "the fixture must contain a particle where the upstream "
                          "velocity-index lag changes the curl (candidate 12)");
            PXG_CHECK_MSG(agreesWithFixed == 0,
                          "the port must REPRODUCE the upstream lag, not fix it "
                          "(%zu words match the corrected form)", agreesWithFixed);
        }
    }

    PXG_CHECK_MSG(mm == 0, "the PBD velocity pair is bit-exact (%zu mismatches)", mm);

    for (Sys* s : order)
    {
        dev->destroyBuffer(s->bPos.h); dev->destroyBuffer(s->bVel.h);
        dev->destroyBuffer(s->bPhase.h); dev->destroyBuffer(s->bColl.h);
        dev->destroyBuffer(s->bCount.h); dev->destroyBuffer(s->bMaterials.h);
        dev->destroyBuffer(s->bP2m.h);
        dev->destroyBuffer(s->bCurl.h); dev->destroyBuffer(s->bDelta.h);
    }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

} // namespace
