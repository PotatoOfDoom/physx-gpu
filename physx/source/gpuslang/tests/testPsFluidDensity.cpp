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

// testPsFluidDensity.cpp -- P9.1b fixture for ps_calculateDensityAndPotentialLaunch
// (particlesystem.cu:3222-3384 template, :3386 dispatcher), ported as
// npPsFluidDensity.slang.
//
// ORACLE.  The P8.73-P8.80 arrangement again: the function is __device__-only, so
// the expectation is a host transcription.  What carries it: every offset
// generated from the engine headers and pinned in testPsMaterials' layout test,
// whole-region 0xAB poison compare over THREE output arrays per system, and the
// sweep.  The one genuinely strong check is structural rather than numerical --
// see "the bug assertion" below.
//
// GEOMETRY.  Four systems in the pool, THREE active, active = { 2, 0, 3 } so the
// list is not a permutation of { 0..gridY-1 } (the P9.1a lesson: a permuted list
// does not kill the mutant that indexes the pool by blockIdx.y).
//
//   pool[0] = A: 500 particles, mRestDensityBoundary > 0 (rho seeds from
//                mStaticDensity), mMaxDiffuseParticles > 0 (so hasDiffuse
//                follows the push bool).  500 > 383 so a halved block stride
//                leaves a hole rather than being an idempotent identity.
//   pool[1] = DECOY: never active; all three outputs must stay fully poisoned.
//   pool[2] = B: 130 particles, mRestDensityBoundary == 0 (rho starts at zero),
//                mMaxDiffuseParticles == 0 (hasDiffuse false in BOTH rounds, so
//                potentials are never written and the velocity stream is dead).
//   pool[3] = C: 90 particles WITH PxParticleFlag::eDISABLE_SELF_COLLISION.  It
//                is active and fully populated, so a mutant that drops the
//                per-system early-out writes real values into C's outputs and
//                dies; with the guard intact all three stay poisoned.
//
// Every per-particle array is PAD (544) entries long while the descriptor reports
// only numParticles, and the tail must come back poisoned.  The first sweep found
// that without such a tail there is nothing beyond numParticles to observe, so the
// mutant replacing the `p < numParticles` guard with the grid's own extent
// SURVIVES.  The padded particles are fluid with zero contacts, so that mutant
// deterministically writes into the tail; PAD > GRID_X * BLOCK is asserted.
//
// TWO ROUNDS over shouldComputePotentials (true, false).  With A carrying
// mMaxDiffuseParticles > 0, round 1 exercises hasDiffuse == true (potentials
// written, the velocity stream live, the upstream bug observable) and round 2
// hasDiffuse == false (potentials poisoned, velocity stream dead).  That is trap
// T10 from both sides, and it is also what proves the AND rather than an OR.
//
// PER-PARTICLE CASES, placed at known indices of A so each trap has a witness:
//   0  contactCount == 0            -- the loop never runs; rho stays at its seed
//   1  contactCount == 1            -- the ONLY case where the CUDA reads the
//                                      never-written collisionIndex slot 1
//   2  a COINCIDENT neighbour       -- dSq == 0, so rd == inf and d == 0; W(0) is
//                                      spiky1 so it DOES enter rho, while the
//                                      d > 0 half of the gradient guard excludes
//                                      it from the normal (traps T4 and T6)
//   3  a neighbour at EXACTLY the   -- dSq == cellSizeSq, which the strict `<`
//      contact distance               must EXCLUDE (trap T5)
//   4  non-fluid                    -- nothing written at all (trap T2)
//   5  fluid in group 1, whose      -- normals[p] must stay poisoned (trap T9)
//      surfaceTensionDerived == 0
//   6  invMass (pos.w) == 0         -- the div normalisation is skipped (T11)
//   7  no neighbour in range        -- rho stays at its seed; with A's seed 0 for
//                                      this particle, rho == 0 skips the other
//                                      half of T11
// every other particle is generic, with a contact count cycling 1..7.
//
// THE BUG ASSERTION.  Reproducing upstream candidate 13 is the point of this
// tranche, so the fixture does not merely encode it in the expectation -- it
// asserts that the fixture can SEE it: there must exist a particle for which the
// buggy pairing (pos[idx[0]] with vel[idx[1]]) and the corrected pairing
// (pos[idx[0]] with vel[idx[0]]) give DIFFERENT div, and the test recomputes the
// corrected expectation and requires the GPU to disagree with it.  Without that,
// a port that silently "fixed" the bug would pass.

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
struct F2 { float x, y; };

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

struct CdPush
{
    uint64_t particleSystems;
    uint64_t activeParticleSystems;
    uint32_t shouldComputePotentialsWord;
    uint32_t _pad20;
};
static_assert(sizeof(CdPush) == 24, "kTwoPtrBool blob is 24 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;

static const uint32_t PS_FLAG_DISABLE_SELF_COLLISION = 1u;
static const uint32_t PS_PHASE_FLUID = 0x00400000u;
static const uint32_t PS_PHASE_GROUP_MASK = 0x000fffffu;

// --- the host mirror of the shader's scalar helpers ------------------------
static float sqrH(float x) { return x * x; }
static float WH(float x, float k1, float kInv) { return k1 * sqrH(1.0f - x * kInv); }
static float dWdxH(float x, float k2, float kInv) { return -k2 * (1.0f - x * kInv); }
static float pxMaxH(float a, float b) { return a > b ? a : b; }

// One particle system's inputs and outputs, host-side.
struct Sys
{
    uint32_t numParticles = 0;
    uint32_t flags = 0;
    uint32_t maxDiffuse = 0;
    float restDensityBoundary = 0.f, restDensity = 0.f, lambdaScale = 0.f;
    float spiky1 = 0.f, spiky2 = 0.f, contactDistInv = 0.f, contactDistSq = 0.f;

    std::vector<F4> pos, vel;
    std::vector<uint32_t> phase, collisionIndex, contactCount;
    std::vector<float> staticDensity;
    std::vector<float> derived;   // 4 floats per group

    DevBuf bPos, bVel, bPhase, bColl, bCount, bStatic, bDerived;
    DevBuf bDensity, bNormal, bPotential;
};

// The transcription of particlesystem.cu:3259-3382.  `forceCorrectVelocityIndex`
// exists ONLY so the test can compute what a bug-fixed port would produce and
// require the GPU to differ from it.
static void refSystem(const Sys& s, bool shouldComputePotentials,
                      std::vector<uint32_t>& density,
                      std::vector<uint32_t>& normal,
                      std::vector<uint32_t>& potential,
                      bool forceCorrectVelocityIndex)
{
    if ((s.flags & PS_FLAG_DISABLE_SELF_COLLISION) != 0u)
        return;

    const uint32_t N = s.numParticles;
    const bool hasDiffuse = (s.maxDiffuse > 0u) && shouldComputePotentials;

    for (uint32_t p = 0; p < N; ++p)
    {
        const uint32_t phase = s.phase[p];
        if ((phase & PS_PHASE_FLUID) == 0u)
            continue;                       // trap T2: writes nothing

        const uint32_t group = phase & PS_PHASE_GROUP_MASK;
        const float std_ = s.derived[group * 4u + 0u];

        const F4 xi4 = s.pos[p];
        const float xix = xi4.x, xiy = xi4.y, xiz = xi4.z;
        const F4 vi4 = s.vel[p];

        float rho = 0.0f;
        if (s.restDensityBoundary > 0.0f)
            rho = s.staticDensity[p];

        float nx = 0.f, ny = 0.f, nz = 0.f;
        float weight = 0.f, div = 0.f;

        const uint32_t cc = s.contactCount[p];

        for (uint32_t i = 0; i < cc; ++i)
        {
            const uint32_t qPos = s.collisionIndex[p + i * N];
            const F4 xj4 = s.pos[qPos];

            float vjx = 0.f, vjy = 0.f, vjz = 0.f;
            if (hasDiffuse)
            {
                const uint32_t iVel = forceCorrectVelocityIndex ? i : ((i > 0u) ? i : 1u);
                const uint32_t qRaw = s.collisionIndex[p + iVel * N];
                const uint32_t qVel = (qRaw < N) ? qRaw : 0u;
                const F4 vj4 = s.vel[qVel];
                vjx = vj4.x; vjy = vj4.y; vjz = vj4.z;
            }

            const float xijx = xix - xj4.x, xijy = xiy - xj4.y, xijz = xiz - xj4.z;
            const float dSq = xijx * xijx + xijy * xijy + xijz * xijz;

            if (dSq < s.contactDistSq)                       // trap T5: strict <
            {
                const float rd = 1.0f / std::sqrt(dSq);      // trap T4: the
                const float d = 1.0f / rd;                   // double reciprocal

                const float mass = 1.0f;
                rho += mass * WH(d, s.spiky1, s.contactDistInv);
                weight += mass;

                if ((std_ > 0.0f || hasDiffuse) && d > 0.0f) // trap T6
                {
                    const float g = dWdxH(d, s.spiky2, s.contactDistInv);
                    const float dwx = (g * xijx) * rd;       // trap T7
                    const float dwy = (g * xijy) * rd;
                    const float dwz = (g * xijz) * rd;
                    nx += dwx; ny += dwy; nz += dwz;

                    if (hasDiffuse)
                    {
                        const float vijx = vjx - vi4.x;
                        const float vijy = vjy - vi4.y;
                        const float vijz = vjz - vi4.z;
                        div += vijx * dwx + vijy * dwy + vijz * dwz;
                    }
                }
            }
        }

        float c = pxMaxH(rho - s.restDensity, -s.restDensity * 0.005f);   // T8
        c *= s.lambdaScale;
        density[p] = f2u(c);

        if (std_ > 0.0f)                                                  // T9
        {
            nx *= std_; ny *= std_; nz *= std_;
            normal[p * 4u + 0u] = f2u(nx);
            normal[p * 4u + 1u] = f2u(ny);
            normal[p * 4u + 2u] = f2u(nz);
            normal[p * 4u + 3u] = f2u(weight);
        }

        if (hasDiffuse)
        {
            if (xi4.w > 0.0f && rho > 0.0f)                               // T11
                div /= (xi4.w * rho);
            potential[p * 2u + 0u] = f2u(div);
            potential[p * 2u + 1u] = f2u(pxMaxH(1.0f - 2.0f * rho, 0.0f));
        }
    }
}

// A small deterministic hash so positions/velocities vary without <random>.
static float jitter(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return float(h % 20011u) * (1.0f / 20011.0f);   // [0,1)
}

PXG_TEST(psFluidDensity)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t BLOCK = 256u;
    const uint32_t MN = 8u;          // neighbour columns per particle
    const uint32_t NUM_GROUPS = 3u;
    // Every per-particle array is allocated PAD entries long while the descriptor
    // reports only numParticles, and the tail must come back poisoned.  Without
    // that tail there is nothing beyond numParticles to observe, and the mutant
    // that replaces the `p < numParticles` guard with the grid's own extent
    // SURVIVES -- which is exactly what the first sweep found.  PAD must exceed
    // the whole grid's reach (GRID_X * BLOCK = 512), and the padded particles are
    // made FLUID with ZERO contacts so such a mutant deterministically writes
    // into the tail (fluid => it does not early-out; zero contacts => it does not
    // walk a neighbour list that is only sized for the real particles).
    const uint32_t PAD = 544u;

    // The exact-boundary case needs a separation whose square is representable
    // and reproducible in both the fixture and the shader: 0.5 * 0.5 = 0.25.
    const float SEP_EXACT = 0.5f;
    const float CONTACT_DIST = SEP_EXACT;                 // so cellSizeSq == 0.25
    const float CONTACT_DIST_SQ = CONTACT_DIST * CONTACT_DIST;
    const float CONTACT_DIST_INV = 1.0f / CONTACT_DIST;

    auto buildSys = [&](uint32_t N, uint32_t flags, uint32_t maxDiffuse,
                        float restDensityBoundary, float restDensity,
                        float lambdaScale, float spiky1, float spiky2,
                        uint32_t salt, bool withSpecials)
    {
        Sys s;
        s.numParticles = N;
        s.flags = flags;
        s.maxDiffuse = maxDiffuse;
        s.restDensityBoundary = restDensityBoundary;
        s.restDensity = restDensity;
        s.lambdaScale = lambdaScale;
        s.spiky1 = spiky1;
        s.spiky2 = spiky2;
        s.contactDistInv = CONTACT_DIST_INV;
        s.contactDistSq = CONTACT_DIST_SQ;

        s.pos.resize(PAD); s.vel.resize(PAD); s.phase.resize(PAD);
        s.contactCount.resize(PAD); s.staticDensity.resize(PAD);
        s.collisionIndex.assign(size_t(N) * MN, 0u);

        for (uint32_t p = 0; p < N; ++p)
        {
            // A loose cloud: most neighbours land inside the contact distance,
            // some outside, so trap T5 is exercised in both directions.
            s.pos[p] = { jitter(p, salt + 1u) * 1.6f,
                         jitter(p, salt + 2u) * 1.6f,
                         jitter(p, salt + 3u) * 1.6f,
                         (p % 11u == 6u) ? 0.0f : (0.5f + jitter(p, salt + 4u)) };
            s.vel[p] = { jitter(p, salt + 5u) * 2.0f - 1.0f,
                         jitter(p, salt + 6u) * 2.0f - 1.0f,
                         jitter(p, salt + 7u) * 2.0f - 1.0f,
                         0.0f };
            const uint32_t grp = p % NUM_GROUPS;
            const bool fluid = (p % 13u != 4u);
            s.phase[p] = grp | (fluid ? PS_PHASE_FLUID : 0u);
            s.staticDensity[p] = jitter(p, salt + 8u) * 0.4f;
            s.contactCount[p] = 1u + (p % 7u);
            for (uint32_t i = 0; i < MN; ++i)
                s.collisionIndex[p + i * N] = (p * 37u + i * 101u + 5u) % N;
        }

        if (withSpecials)
        {
            s.contactCount[0] = 0u;                       // no contacts
            s.contactCount[1] = 1u;                       // the stale-slot read
            s.phase[4] = (4u % NUM_GROUPS);               // non-fluid
            s.phase[5] = 1u | PS_PHASE_FLUID;             // group 1: std == 0
            s.pos[6].w = 0.0f;                            // invMass 0

            // p == 2: a COINCIDENT first neighbour (dSq == 0).
            s.contactCount[2] = 3u;
            s.collisionIndex[2 + 0u * N] = 2u;            // itself
            // p == 3: a first neighbour at EXACTLY the contact distance.
            s.contactCount[3] = 2u;
            s.collisionIndex[3 + 0u * N] = 9u;
            s.pos[9] = s.pos[3];
            s.pos[9].x = s.pos[3].x - SEP_EXACT;          // dSq == 0.25 exactly
            // p == 7: every neighbour far away, so rho keeps only its seed.
            s.contactCount[7] = 2u;
            s.collisionIndex[7 + 0u * N] = 10u;
            s.collisionIndex[7 + 1u * N] = 11u;
            s.pos[10] = { s.pos[7].x + 40.0f, s.pos[7].y, s.pos[7].z, 1.0f };
            s.pos[11] = { s.pos[7].x, s.pos[7].y + 60.0f, s.pos[7].z, 1.0f };
            s.staticDensity[7] = 0.0f;                    // and rho == 0
            s.phase[7] = 0u | PS_PHASE_FLUID;

            // Whatever p == 9/10/11 were reused for above, keep them fluid so
            // they are not silently skipped when visited as particles.
            s.phase[9] = 0u | PS_PHASE_FLUID;
            s.phase[10] = 0u | PS_PHASE_FLUID;
            s.phase[11] = 0u | PS_PHASE_FLUID;
        }

        // The tail past numParticles: fluid, zero contacts, so a guard mutant
        // writes here and is caught, while the correct kernel never reaches it.
        for (uint32_t q = N; q < PAD; ++q)
        {
            s.pos[q] = { jitter(q, salt + 21u), jitter(q, salt + 22u),
                         jitter(q, salt + 23u), 1.0f };
            s.vel[q] = { jitter(q, salt + 24u), jitter(q, salt + 25u),
                         jitter(q, salt + 26u), 0.0f };
            s.phase[q] = 0u | PS_PHASE_FLUID;
            s.contactCount[q] = 0u;
            s.staticDensity[q] = jitter(q, salt + 27u);
        }

        // group 0 and 2 have surface tension, group 1 has none (trap T9).
        s.derived.assign(NUM_GROUPS * 4u, 0.0f);
        const float st[NUM_GROUPS] = { 0.75f, 0.0f, 1.25f };
        for (uint32_t g = 0; g < NUM_GROUPS; ++g)
        {
            s.derived[g * 4u + 0u] = st[g];
            s.derived[g * 4u + 1u] = -1000.0f - float(g);   // cohesion: unread here
            s.derived[g * 4u + 2u] = -2000.0f - float(g);
            s.derived[g * 4u + 3u] = -3000.0f - float(g);
        }
        return s;
    };

    Sys sysA = buildSys(500u, 0u, 64u, 0.6f, 1.1f, 0.37f, 1.9f, 2.7f, 100u, true);
    Sys sysD = buildSys(200u, 0u, 0u, 0.0f, 0.9f, 0.21f, 1.3f, 3.1f, 200u, false);
    Sys sysB = buildSys(130u, 0u, 0u, 0.0f, 1.3f, 0.53f, 2.3f, 1.7f, 300u, false);
    Sys sysC = buildSys(90u, PS_FLAG_DISABLE_SELF_COLLISION, 32u,
                        0.5f, 1.0f, 0.44f, 1.5f, 2.1f, 400u, false);

    const uint32_t GRID_X = (sysA.numParticles + BLOCK - 1u) / BLOCK;
    PXG_CHECK_MSG(GRID_X == 2u, "the fixture needs two blocks (got %u)", GRID_X);
    PXG_CHECK_MSG(sysA.numParticles > 128u * (GRID_X - 1u) + 255u,
                  "A must exceed a halved block stride's reach (%u vs %u)",
                  sysA.numParticles, 128u * (GRID_X - 1u) + 255u);
    PXG_CHECK_MSG(PAD > GRID_X * BLOCK,
                  "the poisoned tail must outreach the whole grid (%u vs %u)",
                  PAD, GRID_X * BLOCK);

    Sys* order[4] = { &sysA, &sysD, &sysB, &sysC };
    for (Sys* s : order)
    {
        s->bPos = makeDev(dev, s->pos);
        s->bVel = makeDev(dev, s->vel);
        s->bPhase = makeDev(dev, s->phase);
        s->bColl = makeDev(dev, s->collisionIndex);
        s->bCount = makeDev(dev, s->contactCount);
        s->bStatic = makeDev(dev, s->staticDensity);
        s->bDerived = makeDev(dev, s->derived);
    }

    std::vector<uint32_t> active = { 2u, 0u, 3u };
    DevBuf bActive = makeDev(dev, active);

    size_t mm = 0;
    bool sawBugWitness = false;

    for (int round = 0; round < 2; ++round)
    {
        const bool shouldComputePotentials = (round == 0);

        // fresh, poisoned outputs every round
        for (Sys* s : order)
        {
            std::vector<uint32_t> d(PAD, POISON_U32);
            std::vector<uint32_t> nrm(size_t(PAD) * 4u, POISON_U32);
            std::vector<uint32_t> pot(size_t(PAD) * 2u, POISON_U32);
            s->bDensity = makeDev(dev, d);
            s->bNormal = makeDev(dev, nrm);
            s->bPotential = makeDev(dev, pot);
        }

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
            ps.mStaticDensity = reinterpret_cast<PxReal*>(size_t(s->bStatic.addr));
            ps.mDerivedPBDMaterialData =
                reinterpret_cast<PxgPBDParticleMaterialDerived*>(size_t(s->bDerived.addr));
            ps.mDensity = reinterpret_cast<PxReal*>(size_t(s->bDensity.addr));
            ps.mSurfaceNormal = reinterpret_cast<float4*>(size_t(s->bNormal.addr));
            ps.mDiffusePotentials = reinterpret_cast<float2*>(size_t(s->bPotential.addr));
            ps.mCommonData.mNumParticles = s->numParticles;
            ps.mCommonData.mMaxDiffuseParticles = s->maxDiffuse;
            ps.mCommonData.mParticleContactDistance = CONTACT_DIST;
            ps.mCommonData.mParticleContactDistanceInv = s->contactDistInv;
            ps.mCommonData.mParticleContactDistanceSq = s->contactDistSq;
            ps.mData.mFlags = s->flags;
            ps.mData.mRestDensityBoundary = s->restDensityBoundary;
            ps.mData.mRestDensity = s->restDensity;
            ps.mData.mLambdaScale = s->lambdaScale;
            ps.mData.mSpiky1 = s->spiky1;
            ps.mData.mSpiky2 = s->spiky2;
            descArr.push_back(ps);
        }
        DevBuf bDesc = makeDev(dev, descArr);

        CdPush push = {};
        push.particleSystems = bDesc.addr;
        push.activeParticleSystems = bActive.addr;
        push.shouldComputePotentialsWord = shouldComputePotentials ? 1u : 0u;

        Kernel k("npPsFluidDensity.spv", "ps_calculateDensityAndPotentialLaunch",
                 {}, sizeof(CdPush));
        k.dispatch(GRID_X, {}, push, uint32_t(active.size()), 1u);
        dev->flush();

        for (Sys* s : order)
        {
            std::vector<uint32_t> rd(PAD, POISON_U32);
            std::vector<uint32_t> rn(size_t(PAD) * 4u, POISON_U32);
            std::vector<uint32_t> rp(size_t(PAD) * 2u, POISON_U32);

            const bool isActive = (s == &sysA || s == &sysB || s == &sysC);
            if (isActive)
                refSystem(*s, shouldComputePotentials, rd, rn, rp, false);

            const auto gd = downloadBuffer<uint32_t>(dev, s->bDensity.h, PAD);
            const auto gn = downloadBuffer<uint32_t>(dev, s->bNormal.h, PAD * 4u);
            const auto gp = downloadBuffer<uint32_t>(dev, s->bPotential.h, PAD * 2u);

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
                            std::printf("  round %d sys %p %s MM %zu: ref %g got %g\n",
                                        round, (void*)s, what, w, rf, gf);
                        }
                        ++mm;
                    }
            };
            cmp(rd, gd, "density");
            cmp(rn, gn, "normal");
            cmp(rp, gp, "potential");

            // --- the bug assertion, round 0 on A only --------------------
            // Recompute with the CORRECTED velocity index and require the GPU to
            // DISAGREE, so a port that silently fixed the upstream lag fails.
            if (s == &sysA && shouldComputePotentials)
            {
                std::vector<uint32_t> cd(PAD, POISON_U32);
                std::vector<uint32_t> cn(size_t(PAD) * 4u, POISON_U32);
                std::vector<uint32_t> cp(size_t(PAD) * 2u, POISON_U32);
                refSystem(*s, true, cd, cn, cp, /*forceCorrect*/ true);
                for (size_t w = 0; w < cp.size(); ++w)
                    if (cp[w] != rp[w]) { sawBugWitness = true; break; }
                PXG_CHECK_MSG(sawBugWitness,
                              "the fixture must contain a particle where the upstream "
                              "velocity-index lag changes div (candidate 13)");
                size_t agreesWithFixed = 0;
                for (size_t w = 0; w < cp.size(); ++w)
                    if (cp[w] != rp[w] && cp[w] == gp[w]) ++agreesWithFixed;
                PXG_CHECK_MSG(agreesWithFixed == 0,
                              "the port must REPRODUCE the upstream lag, not fix it "
                              "(%zu words match the corrected form)", agreesWithFixed);
            }
        }

        for (Sys* s : order)
        {
            dev->destroyBuffer(s->bDensity.h);
            dev->destroyBuffer(s->bNormal.h);
            dev->destroyBuffer(s->bPotential.h);
        }
        dev->destroyBuffer(bDesc.h);
    }

    PXG_CHECK_MSG(mm == 0, "ps_calculateDensityAndPotentialLaunch bit-exact (%zu mismatches)", mm);

    for (Sys* s : order)
    {
        dev->destroyBuffer(s->bPos.h); dev->destroyBuffer(s->bVel.h);
        dev->destroyBuffer(s->bPhase.h); dev->destroyBuffer(s->bColl.h);
        dev->destroyBuffer(s->bCount.h); dev->destroyBuffer(s->bStatic.h);
        dev->destroyBuffer(s->bDerived.h);
    }
    dev->destroyBuffer(bActive.h);
}

} // namespace
