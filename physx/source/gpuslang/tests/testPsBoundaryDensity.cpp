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

// testPsBoundaryDensity.cpp -- P9.8 fixture for ps_accumulateStaticDensity and
// ps_accumulateRigidDensity, ported as npPsBoundaryDensity.slang.
//
// ORACLE: host transcription, carried by generated-and-pinned offsets, whole-region
// sentinel/poison compare, and the sweep.  Neither kernel body has a
// PX_CUDA_CALLABLE original, so this is the P8.73-P8.80 arrangement, not a true
// oracle -- and this file says so rather than presenting the sweep as one.  What it
// does have that most of those tranches did not is that both kernels are pure
// gather-then-assign: each thread sums into a register and stores once, so unlike the
// P9.5 aerodynamics pair there are NO float atomics and no summation-order ambiguity.
// Every check in this file is therefore BIT-EXACT and there is not one tolerance
// anywhere -- not even a relative epsilon.
//
// The single place a check admits more than one value is NOT a tolerance and NOT a
// property of the port: on the doubly-covered work indices of psBoundaryDensityWorkIndex
// the ENGINE's own kernel races with itself (see UPSTREAM CANDIDATE 16 below), and the
// test accepts either of the two EXACT outcomes that race can produce, counting which
// one it saw.
//
// The two kernels are CHAINED -- ps_accumulateRigidDensity seeds its accumulator
// from `mStaticDensity[tParticleId]`, i.e. from what ps_accumulateStaticDensity just
// wrote -- so psBoundaryDensity dispatches them in the host's order and compares the
// composed result.  A mutant that seeds from zero dies on that.
//
// THREE TESTS:
//   psBoundaryDensityLayout -- every constant the shader hard-codes, pinned against
//     the real engine headers, including the PhysX 5.12 PxgParticleRef packing.
//   psBoundaryDensity -- the pair, over a four-system pool with a decoy, an active
//     list that is not a permutation, and a gate-closed system.
//   psBoundaryDensityWorkIndex -- 262145 work indices, the smallest size at which
//     nbIterationsPerBlock exceeds one and UPSTREAM CANDIDATE 16 becomes OBSERVABLE.
//     Below that size the buggy `i * ACCUMULATE_DELTA` term is multiplied by zero and
//     any port of it is an identity; above it, half the work indices are covered
//     twice and the other half never.  This port reproduces the arithmetic exactly,
//     so the fixture asserts the SKIPPED HALF is skipped -- a "fixed" port dies here.
//
// THE DECOY IS LOAD-BEARING, and differently from every other particle fixture in
// this port.  Pool index 1 is never named by the active list, so
// ps_accumulateStaticDensity must never touch it -- the usual check.  But
// ps_accumulateRigidDensity does NOT consult the active list at all: it recovers the
// system from the contact's packed PxU64.  So one rigid work index deliberately
// names the decoy, and the decoy's density must come back written by the rigid
// kernel and only by the rigid kernel.  A port that adds an active-list indirection
// to the rigid kernel dies on that one entry.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxgSimulationCoreDesc.h"
#include "PxgParticleSystemCoreKernelIndices.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// PxMax/PxMin for float are ternaries (intrinsics::selectMax), not fmaxf/fminf.
static float hMax(float a, float b) { return a > b ? a : b; }
static float hMin(float a, float b) { return a < b ? a : b; }

// particleSystem.cuh:58-63, verbatim.  NOT clamped at x > radius: beyond the
// radius the bracket goes negative and the square brings it back positive.
static float hSqr(float x) { return x * x; }
static float hW(float x, float kSpiky1, float kInvRadius)
{
    return kSpiky1 * hSqr(1.0f - x * kInvRadius);
}

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

struct StaticPush { uint64_t particleSystems, activeParticleSystems; };
static_assert(sizeof(StaticPush) == 16, "{8,8} = kTwoPtr");

// PhysX 5.12: the SoA PxgDeformablePairInfo + float4 normalPen streams replace the
// 32-byte PxgParticlePrimitiveContact, so the rigid kernel takes six pointers.
struct RigidPush { uint64_t particleSystems, sortedContactInfos, sortedContactNormalPens, totalNumPairs, rangeStart, rangeEnd; };
static_assert(sizeof(RigidPush) == 48, "{8,8,8,8,8,8} = kPsAccumulateRigidDensity");

// The fixtures below still lay each contact out as the 5.6.1 32-byte record (normal_pen
// @0, rigid id @16, particle ref @24) -- it keeps their construction readable -- and this
// splits it into the two 5.12 streams the kernel reads.
struct SoaContacts { DevBuf infos, normalPens; };
static SoaContacts splitContacts(PxgSlangComputeDevice* dev, const std::vector<uint8_t>& bytes)
{
    const size_t n = bytes.size() / 32u;
    std::vector<PxgDeformablePairInfo> infos(n);
    std::vector<float> normalPens(n * 4u);
    for (size_t i = 0; i < n; ++i)
    {
        std::memcpy(&normalPens[i * 4u], &bytes[i * 32u], 16);
        std::memcpy(&infos[i].ref0, &bytes[i * 32u + 16u], 8);
        std::memcpy(&infos[i].ref1, &bytes[i * 32u + 24u], 8);
    }
    SoaContacts c;
    c.infos = makeDev(dev, infos);
    c.normalPens = makeDev(dev, normalPens);
    return c;
}

static const uint32_t POISON_U32 = 0xABABABABu;

// PxgParticleSystemKernelBlockDim::PS_COLLISION -- the LAUNCH width of both kernels.
static const uint32_t BLOCK = 256u;
// PxgParticleSystemKernelGridDim::PS_COLLISION -- the rigid kernel's FIXED grid.
static const uint32_t RIGID_GRID = 1024u;
// PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA -- what the rigid kernel's work
// index strides by, which is NOT its launch width.  Candidate 16.
static const uint32_t ACCUM_DELTA = 512u;
static const uint32_t MAX_STATIC_CONTACTS = 12u;

// A per-particle sentinel for mStaticDensity, used instead of poison over the live
// range.  It is distinctive (no kernel in this pair can produce a negative density),
// distinct per particle, and -- unlike poison -- a legitimate float, so the rigid
// kernel's `rho = staticDensity[tParticleId]` seed is observable on a system the
// static kernel never wrote.
static float densitySentinel(uint32_t i) { return -7.0f - float(i) * 0.011f; }

static uint32_t hashu(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

// ---------------------------------------------------------------------------
// Layout proof -- every constant npPsBoundaryDensity.slang hard-codes.
// ---------------------------------------------------------------------------
PXG_TEST(psBoundaryDensityLayout)
{
    // The contact-count clamp.
    PXG_CHECK_MSG(PxgParticleContactInfo::MaxStaticContactsPerParticle == MAX_STATIC_CONTACTS,
                  "MaxStaticContactsPerParticle == 12 (got %u)",
                  unsigned(PxgParticleContactInfo::MaxStaticContactsPerParticle));

    // PxgParticleContactInfo is a BARE float4 -- the shader reads mOneWayContactInfos
    // as Ptr<float4> with no struct stride of its own.
    PXG_CHECK_MSG(sizeof(PxgParticleContactInfo) == 16,
                  "sizeof(PxgParticleContactInfo) == 16 (got %zu)", sizeof(PxgParticleContactInfo));
    PXG_CHECK(offsetof(PxgParticleContactInfo, mNormal_PenW) == 0);

    // PxgDeformablePairInfo (PhysX 5.12): two PxU64, the particle ref in ref1.
    PXG_CHECK_MSG(sizeof(PxgDeformablePairInfo) == 16,
                  "sizeof(PxgDeformablePairInfo) == 16 (got %zu)", sizeof(PxgDeformablePairInfo));
    PXG_CHECK(offsetof(PxgDeformablePairInfo, ref0) == 0);
    PXG_CHECK(offsetof(PxgDeformablePairInfo, ref1) == 8);

    // The PxgParticleRef the rigid kernel decodes (PxgGetParticleSystemId /
    // PxgGetParticleIndex): the LOW word is the particle index, the system id sits
    // above the two flag bits at bit 34.  Checked by calling the engine's own encoder
    // rather than by restating the shift.
    {
        const PxU32 sysId = 0x12345678u, partId = 0x9ABCDEF0u;
        const PxU64 packed = PxgEncodeParticleRef(sysId, partId);
        PXG_CHECK(PxgGetParticleSystemId(packed) == sysId);
        PXG_CHECK(PxgGetParticleIndex(packed) == partId);
        PXG_CHECK_MSG(uint32_t(packed & 0xFFFFFFFFull) == partId,
                      "the low word must be the particle index");
        PXG_CHECK_MSG(uint32_t(packed >> 34) == sysId,
                      "the system id must sit at bit 34");
    }

    // The descriptor fields both kernels walk.
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mParticleContactDistanceInv) == 108);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mNumParticles) == 116);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactInfos) == 200);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactCount) == 232);
    PXG_CHECK(offsetof(PxgParticleSystem, mStaticDensity) == 352);
    PXG_CHECK_MSG(offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mSpiky1) == 276,
                  "mData.mSpiky1 must sit at 276 (got %zu)",
                  offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mSpiky1));
    PXG_CHECK_MSG(offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mRestDensityBoundary) == 288,
                  "mData.mRestDensityBoundary must sit at 288 (got %zu)",
                  offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mRestDensityBoundary));
    PXG_CHECK_MSG(offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mFluidRestOffset) == 304,
                  "mData.mFluidRestOffset must sit at 304 (got %zu)",
                  offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mFluidRestOffset));
    PXG_CHECK_MSG(sizeof(PxgParticleSystem) == 608,
                  "sizeof(PxgParticleSystem) == 608 (PhysX 5.12; got %zu)", sizeof(PxgParticleSystem));

    // The launch shapes.  The rigid kernel's grid is a COMPILE-TIME constant on our
    // side (it is a fixed host constant, not a dispatch parameter), so the shader
    // spells 1024 and this is what pins it.
    PXG_CHECK_MSG(PxgParticleSystemKernelBlockDim::PS_COLLISION == BLOCK,
                  "PS_COLLISION block dim == 256 (got %u)",
                  unsigned(PxgParticleSystemKernelBlockDim::PS_COLLISION));
    PXG_CHECK_MSG(PxgParticleSystemKernelGridDim::PS_COLLISION == RIGID_GRID,
                  "PS_COLLISION grid dim == 1024 (got %u)",
                  unsigned(PxgParticleSystemKernelGridDim::PS_COLLISION));
    // Candidate 16: the two are DIFFERENT, which is the whole bug.
    PXG_CHECK_MSG(PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA == ACCUM_DELTA,
                  "ACCUMULATE_DELTA block dim == 512 (got %u)",
                  unsigned(PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA));
    PXG_CHECK_MSG(PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA
                      != PxgParticleSystemKernelBlockDim::PS_COLLISION,
                  "candidate 16 rests on ACCUMULATE_DELTA != PS_COLLISION");
}

// ---------------------------------------------------------------------------
// The pair.
// ---------------------------------------------------------------------------

struct Sys
{
    uint32_t numParticles = 0;
    float restDensityBoundary = 0.0f;
    float spiky1 = 0.0f;
    float contactDistInv = 0.0f;
    float restDistance = 0.0f;

    std::vector<uint32_t> counts;       // STORED counts, some above the clamp
    std::vector<float> pen;             // [column * numParticles + particle]
    DevBuf bContacts, bCounts, bDensity;
};

PXG_TEST(psBoundaryDensity)
{
    PxgSlangComputeDevice* dev = testDevice();

    // Four systems.  Pool index 1 is the DECOY: never in the active list, so the
    // static kernel must never touch it -- but the rigid kernel reaches it anyway
    // through a contact, because it does not use the active list.
    const uint32_t NUM_SYS = 4u;
    const uint32_t DECOY = 1u;
    // Every particle count DISTINCT, none a multiple of the block width, and the
    // largest well past one block so the blockIdx.x term is under test.
    const uint32_t NP[NUM_SYS] = { 700u, 337u, 300u, 520u };
    const uint32_t PAD = 24u;           // poisoned tail on every density array
    // More columns than the clamp allows, so a dropped clamp reads REAL contacts
    // rather than falling off the allocation.
    const uint32_t COLUMNS = 16u;

    std::vector<Sys> sys(NUM_SYS);
    // Every scalar distinct across systems, so reading the wrong system is visible
    // in the value and not only in the index.
    //
    // SYSTEM D'S BOUNDARY IS HUGE ON PURPOSE, and the first sweep is why.  With
    // every active system's rest density small, the store's min() saturated 912 of
    // 1220 particles, and a saturated particle cannot show a difference in how many
    // contacts were summed -- so dropping the twelve-contact clamp entirely, and
    // moving it to eleven or thirteen, all SURVIVED.  D's boundary is above any
    // reachable rho (16 contacts * spiky1 4.25 = 68), so on D the sum is fully
    // observable and the clamp is under test; A and the decoy keep small boundaries
    // so the min() itself stays under test.  Both properties are asserted below
    // rather than assumed.
    const float RDB[NUM_SYS] = { 3.5f, 100.0f, 0.0f, 1.0e6f };
    //                            A    decoy   GATE CLOSED   D (min never binds)
    const float SPIKY[NUM_SYS] = { 2.75f, 1.875f, 1.5f, 4.25f };
    const float CDI[NUM_SYS] = { 1.0f / 0.35f, 1.0f / 0.27f, 1.0f / 0.5f, 1.0f / 0.22f };
    const float RESTD[NUM_SYS] = { 0.06f, 0.09f, 0.11f, 0.03f };

    uint32_t maxParticles = 0;
    for (uint32_t s = 0; s < NUM_SYS; ++s)
    {
        Sys& S = sys[s];
        S.numParticles = NP[s];
        S.restDensityBoundary = RDB[s];
        S.spiky1 = SPIKY[s];
        S.contactDistInv = CDI[s];
        S.restDistance = RESTD[s];
        maxParticles = PxMax(maxParticles, S.numParticles);

        S.counts.resize(S.numParticles);
        S.pen.assign(size_t(S.numParticles) * COLUMNS, 0.0f);
        for (uint32_t i = 0; i < S.numParticles; ++i)
        {
            // Counts cycle through 0, 1, 5, 11, 12, 13 and 16.  The last two are
            // ABOVE MaxStaticContactsPerParticle, which is what puts the clamp
            // under test; 12 sits exactly on it, so an off-by-one clamp dies too.
            static const uint32_t kCounts[7] = { 0u, 1u, 5u, 11u, 12u, 13u, 16u };
            S.counts[i] = kCounts[i % 7u];

            for (uint32_t c = 0; c < COLUMNS; ++c)
            {
                // Penetrations spread across all four regimes of
                // W(PxMax(0, pen + restDistance)):
                //   pen + restDistance <  0  -> clamped to zero, W = spiky1
                //   0 <= x < radius          -> the ordinary case
                //   x  = radius exactly      -> W = 0
                //   x  > radius              -> the bracket goes NEGATIVE and the
                //                               square brings W back POSITIVE, which
                //                               is the un-clamped behaviour B4 pins
                const uint32_t h = hashu(i * COLUMNS + c, s + 11u);
                const float radius = 1.0f / S.contactDistInv;
                float x;
                switch (h % 5u)
                {
                case 0: x = -radius * 2.0f; break;                       // -> clamped
                case 1: x = radius * (0.05f + 0.9f * float(h % 97u) / 97.0f); break;
                case 2: x = radius; break;                               // -> W == 0
                case 3: x = radius * (1.3f + float(h % 31u) * 0.05f); break;  // past it
                default: x = radius * 0.5f; break;
                }
                S.pen[size_t(c) * S.numParticles + i] = x - S.restDistance;
            }
        }
    }

    // Sanity on the design rather than on the port: the clamp and the two regimes
    // must actually be reachable, or the traps they guard are untested.
    {
        uint32_t nOverClamp = 0, nClampedPen = 0, nPastRadius = 0;
        for (uint32_t s = 0; s < NUM_SYS; ++s)
        {
            for (uint32_t i = 0; i < sys[s].numParticles; ++i)
                if (sys[s].counts[i] > MAX_STATIC_CONTACTS) ++nOverClamp;
            for (size_t k = 0; k < sys[s].pen.size(); ++k)
            {
                const float x = sys[s].pen[k] + sys[s].restDistance;
                if (x < 0.0f) ++nClampedPen;
                if (x * sys[s].contactDistInv > 1.0f) ++nPastRadius;
            }
        }
        PXG_CHECK_MSG(nOverClamp > 0, "no particle exceeds the twelve-contact clamp");
        PXG_CHECK_MSG(nClampedPen > 0, "no contact drives pen + restDistance negative");
        PXG_CHECK_MSG(nPastRadius > 0, "no contact reaches past the interaction radius");
    }

    for (uint32_t s = 0; s < NUM_SYS; ++s)
    {
        Sys& S = sys[s];
        // The contact array is a bare float4 array; only .w is read, so xyz carry a
        // DIFFERENT, plausible value per entry -- a mutant that reads .x, .y or .z
        // lands on a real number rather than on a zero that happens to work.
        std::vector<float> contacts(size_t(S.numParticles) * COLUMNS * 4u, 0.0f);
        for (uint32_t c = 0; c < COLUMNS; ++c)
        {
            for (uint32_t i = 0; i < S.numParticles; ++i)
            {
                const size_t k = size_t(c) * S.numParticles + i;
                contacts[k * 4u + 0u] = 0.25f + 0.001f * float(i);
                contacts[k * 4u + 1u] = -0.5f - 0.002f * float(c);
                contacts[k * 4u + 2u] = 0.75f + 0.003f * float(s);
                contacts[k * 4u + 3u] = S.pen[k];
            }
        }
        S.bContacts = makeDev(dev, contacts);
        S.bCounts = makeDev(dev, S.counts);

        // mStaticDensity: a distinctive per-particle sentinel over the live range, a
        // POISON tail past it.  The sentinel proves an untouched entry is untouched
        // just as poison would, and additionally makes the rigid kernel's seed
        // observable on the decoy, which the static kernel never writes.
        std::vector<uint32_t> dens(size_t(S.numParticles + PAD), POISON_U32);
        for (uint32_t i = 0; i < S.numParticles; ++i) dens[i] = f2u(densitySentinel(i));
        S.bDensity = makeDev(dev, dens);
    }

    // The rigid contacts.  numTests is deliberately > BLOCK so more than one block
    // does real work, and < BLOCK * RIGID_GRID so nbIterationsPerBlock is ONE --
    // which is the shape every real scene has.  Candidate 16 is invisible at this
    // size by construction; psBoundaryDensityWorkIndex is where it is observed.
    const uint32_t NUM_TESTS = 400u;
    PXG_CHECK(NUM_TESTS > BLOCK && NUM_TESTS < BLOCK * RIGID_GRID);

    // Work index w -> (system, particle).  Distinct particles per work index, as the
    // producer (ps_findStartEndParticleSecondLaunch) guarantees: two work indices on
    // one particle would be a read-modify-write race, which the engine never has and
    // a fixture must not invent.
    // A TRIPWIRE past the end of the range arrays.  The kernel's guard is
    // `workIndex >= numTests`, and workIndex == numTests IS reachable (400 = 1*256 +
    // 144), so an off-by-one guard reads rangeStart[numTests].  With the arrays
    // sized exactly numTests that read landed in allocation slack full of zeros,
    // which decoded to system 0 / particle 0 with an EMPTY contact range -- the
    // store rewrote the value it had just read and the mutant survived.  The range
    // arrays therefore carry RANGE_PAD extra entries that name a real, non-empty
    // contact range on a particle no legitimate work index touches.  Cross the
    // guard and that particle changes; stay inside it and it keeps its sentinel.
    //
    // The tripwire lives in SYSTEM 3, not system 0, and that is the same saturation
    // lesson the twelve-contact clamp taught above: system 0's boundary is 3.5 and
    // its particles are already sitting on that clamp after the static kernel, so a
    // tripwire write there computes min(3.5 + something, 3.5) = 3.5 and changes
    // NOTHING.  System 3's boundary is above any reachable rho, so the write lands.
    // The mirror asserts below that it really would land, rather than assuming it.
    const uint32_t RANGE_PAD = 8u;
    const uint32_t TRIPWIRE_SYS = 3u;
    const uint32_t TRIPWIRE_PARTICLE = 510u;   // system 3 has 520; nothing else uses it

    struct Work { uint32_t sysId, particleId, start, end; };
    std::vector<Work> work(NUM_TESTS);
    std::vector<float> rigidPen;            // flat, appended in range order
    std::vector<uint32_t> rangeStart(NUM_TESTS + RANGE_PAD), rangeEnd(NUM_TESTS + RANGE_PAD);
    std::vector<uint8_t> contactBytes;

    {
        // Round-robin over the three systems that are NOT the decoy, then one
        // deliberate decoy entry, so the active list plays no part.  Systems 0, 2
        // and 3 are the active ones; system 2's gate is CLOSED (rdb == 0), so both
        // sides of B7 run on the rigid path too.
        static const uint32_t kSysCycle[3] = { 0u, 2u, 3u };
        std::vector<uint32_t> nextParticle(NUM_SYS, 0u);
        uint32_t cursor = 0u;
        for (uint32_t w = 0; w < NUM_TESTS; ++w)
        {
            // Work index 137 names the DECOY.  Not the first and not the last, so a
            // port that special-cases either end does not stumble into it.
            const uint32_t s = (w == 137u) ? DECOY : kSysCycle[w % 3u];
            const uint32_t p = nextParticle[s]++;
            PXG_CHECK_MSG(p < sys[s].numParticles, "ran out of particles for system %u", s);

            // Contact counts per work index cycle 1..5, so the inner loop runs a
            // different number of times per index and an off-by-one on either end
            // shows up.
            const uint32_t n = 1u + (w % 5u);
            work[w].sysId = s;
            work[w].particleId = p;
            work[w].start = cursor;
            work[w].end = cursor + n;
            rangeStart[w] = work[w].start;
            rangeEnd[w] = work[w].end;

            for (uint32_t k = 0; k < n; ++k)
            {
                const uint32_t h = hashu(w * 8u + k, 907u);
                const float radius = 1.0f / sys[s].contactDistInv;
                float x;
                switch (h % 4u)
                {
                case 0: x = -radius * 1.5f; break;                  // clamps to zero
                case 1: x = radius * (0.1f + 0.8f * float(h % 61u) / 61.0f); break;
                case 2: x = radius * (1.2f + float(h % 17u) * 0.07f); break;
                default: x = radius * 0.4f; break;
                }
                rigidPen.push_back(x - sys[s].restDistance);
            }
            cursor += n;
        }

        // The tripwire's own contacts, appended past every legitimate range.
        const uint32_t tripStart = cursor;
        const uint32_t tripCount = 3u;
        for (uint32_t k = 0; k < tripCount; ++k)
            rigidPen.push_back(0.5f / sys[TRIPWIRE_SYS].contactDistInv - sys[TRIPWIRE_SYS].restDistance);
        cursor += tripCount;
        for (uint32_t k = 0; k < RANGE_PAD; ++k)
        {
            rangeStart[NUM_TESTS + k] = tripStart;
            rangeEnd[NUM_TESTS + k] = tripStart + tripCount;
        }

        // Build the 32-byte records.  The rigid kernel reads normal_pen.w at byte 12
        // and particleId at byte 24; rigidId at 16 gets a DISTINCT non-zero pattern
        // so a port that reads the wrong PxU64 lands on garbage that is visibly not
        // a valid (system, particle) pair rather than on zero.
        contactBytes.assign(size_t(cursor) * 32u, 0);
        for (uint32_t w = 0; w < NUM_TESTS; ++w)
        {
            const Work& W = work[w];
            for (uint32_t c = W.start; c < W.end; ++c)
            {
                uint8_t* rec = &contactBytes[size_t(c) * 32u];
                const float nx = 0.6f + 0.0007f * float(c);
                const float ny = -0.8f + 0.0009f * float(c);
                const float nz = 0.1f - 0.0003f * float(c);
                const uint32_t u0 = f2u(nx), u1 = f2u(ny), u2 = f2u(nz), u3 = f2u(rigidPen[c]);
                std::memcpy(rec + 0, &u0, 4);
                std::memcpy(rec + 4, &u1, 4);
                std::memcpy(rec + 8, &u2, 4);
                std::memcpy(rec + 12, &u3, 4);
                const PxU64 rigidId = 0xDEAD0000ull | PxU64(c);
                std::memcpy(rec + 16, &rigidId, 8);
                const PxU64 pid = PxgEncodeParticleRef(W.sysId, W.particleId);
                std::memcpy(rec + 24, &pid, 8);
            }
        }
        for (uint32_t c = tripStart; c < tripStart + tripCount; ++c)
        {
            uint8_t* rec = &contactBytes[size_t(c) * 32u];
            const uint32_t u3 = f2u(rigidPen[c]);
            std::memcpy(rec + 12, &u3, 4);
            const PxU64 rigidId = 0xF00D0000ull | PxU64(c);
            std::memcpy(rec + 16, &rigidId, 8);
            const PxU64 pid = PxgEncodeParticleRef(TRIPWIRE_SYS, TRIPWIRE_PARTICLE);
            std::memcpy(rec + 24, &pid, 8);
        }
        // The tripwire must be genuinely unreached by the real work indices, or it
        // proves nothing.
        for (uint32_t w = 0; w < NUM_TESTS; ++w)
            PXG_CHECK_MSG(!(work[w].sysId == TRIPWIRE_SYS && work[w].particleId == TRIPWIRE_PARTICLE),
                          "work index %u already uses the tripwire particle", w);
    }

    SoaContacts bContactsR = splitContacts(dev, contactBytes);
    DevBuf bRangeStart = makeDev(dev, rangeStart);
    DevBuf bRangeEnd = makeDev(dev, rangeEnd);
    std::vector<uint32_t> totalPairs = { NUM_TESTS };
    DevBuf bTotal = makeDev(dev, totalPairs);

    // The active list is NOT a permutation and never names the decoy.
    std::vector<uint32_t> active = { 2u, 0u, 3u };
    DevBuf bActive = makeDev(dev, active);

    std::vector<PxgParticleSystem> descArr;
    for (uint32_t s = 0; s < NUM_SYS; ++s)
    {
        Sys& S = sys[s];
        PxgParticleSystem ps;
        std::memset(&ps, 0xEE, sizeof(ps));
        ps.mCommonData.mNumParticles = S.numParticles;
        ps.mCommonData.mParticleContactDistanceInv = S.contactDistInv;
        ps.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(S.bContacts.addr));
        ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(S.bCounts.addr));
        ps.mStaticDensity = reinterpret_cast<PxReal*>(size_t(S.bDensity.addr));
        ps.mData.mSpiky1 = S.spiky1;
        ps.mData.mRestDensityBoundary = S.restDensityBoundary;
        ps.mData.mFluidRestOffset = S.restDistance;
        descArr.push_back(ps);
    }
    DevBuf bDesc = makeDev(dev, descArr);

    StaticPush pushS = { bDesc.addr, bActive.addr };
    RigidPush pushR = { bDesc.addr, bContactsR.infos.addr, bContactsR.normalPens.addr, bTotal.addr,
                        bRangeStart.addr, bRangeEnd.addr };

    // The host's grid: ceil(mMaxParticles / PS_COLLISION) blocks, one Y slice per
    // ACTIVE system.  The system is on gridDim.Y, not Z -- every cloth kernel in
    // this port uses Z and this one does not.
    const uint32_t gridStatic = (maxParticles + BLOCK - 1u) / BLOCK;
    PXG_CHECK_MSG(gridStatic > 1u, "the static grid must span more than one block");
    {
        Kernel k("npPsBoundaryDensity.spv", "ps_accumulateStaticDensity", {}, sizeof(StaticPush));
        k.dispatch(gridStatic, {}, pushS, uint32_t(active.size()), 1u);
        dev->flush();
    }
    {
        Kernel k("npPsBoundaryDensity.spv", "ps_accumulateRigidDensity", {}, sizeof(RigidPush));
        k.dispatch(RIGID_GRID, {}, pushR, 1u, 1u);
        dev->flush();
    }

    // ---------------------------------------------------------------------
    // The host mirror, replaying both kernels in the host's order.
    // ---------------------------------------------------------------------
    std::vector<std::vector<float>> ref(NUM_SYS);
    std::vector<std::vector<uint8_t>> written(NUM_SYS);
    for (uint32_t s = 0; s < NUM_SYS; ++s)
    {
        ref[s].resize(sys[s].numParticles);
        written[s].assign(sys[s].numParticles, 0u);
        for (uint32_t i = 0; i < sys[s].numParticles; ++i) ref[s][i] = densitySentinel(i);
    }

    size_t nStaticClamped = 0, nStaticBelow = 0, nClampObservable = 0;
    for (uint32_t a = 0; a < active.size(); ++a)
    {
        const uint32_t s = active[a];
        Sys& S = sys[s];
        if (!(S.restDensityBoundary > 0.0f)) continue;      // B7
        for (uint32_t i = 0; i < S.numParticles; ++i)
        {
            const uint32_t count = PxMin(S.counts[i], MAX_STATIC_CONTACTS);   // B3
            float rho = 0.0f;
            for (uint32_t c = 0; c < count; ++c)                              // B2
            {
                const float x = hMax(0.0f, S.pen[size_t(c) * S.numParticles + i] + S.restDistance);
                rho += hW(x, S.spiky1, S.contactDistInv);                     // B4/B5
            }
            if (rho > S.restDensityBoundary) ++nStaticClamped; else ++nStaticBelow;
            ref[s][i] = hMin(rho, S.restDensityBoundary);                     // B6

            // Is the twelve-contact clamp OBSERVABLE at this particle?  It is only
            // observable when the particle stores more than twelve contacts AND the
            // store's min() does not saturate the difference away.  The first sweep
            // survived three clamp mutants because that count was zero; it is
            // asserted now rather than hoped for.
            if (S.counts[i] > MAX_STATIC_CONTACTS)
            {
                float rhoUnclamped = rho;
                for (uint32_t c = count; c < S.counts[i]; ++c)
                {
                    const float x = hMax(0.0f, S.pen[size_t(c) * S.numParticles + i] + S.restDistance);
                    rhoUnclamped += hW(x, S.spiky1, S.contactDistInv);
                }
                if (hMin(rhoUnclamped, S.restDensityBoundary) != ref[s][i]) ++nClampObservable;
            }
            written[s][i] = 1u;
        }
    }
    PXG_CHECK_MSG(nStaticClamped > 0 && nStaticBelow > 0,
                  "the static min() clamp must bind on some particles and not others "
                  "(clamped %zu, below %zu)", nStaticClamped, nStaticBelow);
    PXG_CHECK_MSG(nClampObservable > 0,
                  "no particle makes the twelve-contact clamp observable -- the store's "
                  "min() saturates every over-count particle");

    // The rigid kernel, simulated exactly as the launch shape runs it.
    size_t nRigidGateClosed = 0, nRigidDecoy = 0, nRigidClamped = 0;
    {
        const uint32_t nbBlocksRequired = (NUM_TESTS + BLOCK - 1u) / BLOCK;
        const uint32_t nbIterationsPerBlock = (nbBlocksRequired + RIGID_GRID - 1u) / RIGID_GRID;
        PXG_CHECK_MSG(nbIterationsPerBlock == 1u,
                      "this test is deliberately in the one-iteration regime (got %u)",
                      nbIterationsPerBlock);
        for (uint32_t bx = 0; bx < RIGID_GRID; ++bx)
        {
            for (uint32_t idx = 0; idx < BLOCK; ++idx)
            {
                for (uint32_t it = 0; it < nbIterationsPerBlock; ++it)
                {
                    const uint32_t w = it * ACCUM_DELTA + idx + nbIterationsPerBlock * bx * BLOCK;
                    if (w >= NUM_TESTS) break;                                // B10
                    const Work& W = work[w];
                    Sys& S = sys[W.sysId];
                    if (!(S.restDensityBoundary > 0.0f)) { ++nRigidGateClosed; continue; }
                    if (W.sysId == DECOY) ++nRigidDecoy;

                    float rho = ref[W.sysId][W.particleId];                   // THE CHAIN
                    for (uint32_t c = W.start; c < W.end; ++c)
                    {
                        const float x = hMax(0.0f, rigidPen[c] + S.restDistance);
                        rho += hW(x, S.spiky1, S.contactDistInv);
                    }
                    if (rho > S.restDensityBoundary) ++nRigidClamped;
                    ref[W.sysId][W.particleId] = hMin(rho, S.restDensityBoundary);
                    written[W.sysId][W.particleId] = 1u;
                }
            }
        }
    }
    PXG_CHECK_MSG(nRigidGateClosed > 0, "no rigid work index lands on a gate-closed system");
    PXG_CHECK_MSG(nRigidDecoy == 1u, "exactly one rigid work index must name the decoy (got %zu)",
                  nRigidDecoy);
    PXG_CHECK_MSG(nRigidClamped > 0, "the rigid min() clamp must bind somewhere");

    // The tripwire must be able to CHANGE its particle, or a loosened guard writes a
    // value indistinguishable from the one already there and the mutant survives.
    // Saturation is what makes that fail silently, so it is asserted, not assumed.
    {
        Sys& T = sys[TRIPWIRE_SYS];
        float tripRho = ref[TRIPWIRE_SYS][TRIPWIRE_PARTICLE];
        for (uint32_t k = 0; k < 3u; ++k)
        {
            const float x = hMax(0.0f, (0.5f / T.contactDistInv - T.restDistance) + T.restDistance);
            tripRho += hW(x, T.spiky1, T.contactDistInv);
        }
        PXG_CHECK_MSG(hMin(tripRho, T.restDensityBoundary) != ref[TRIPWIRE_SYS][TRIPWIRE_PARTICLE],
                      "the tripwire cannot change its particle (%.9g -> %.9g): a loosened "
                      "guard would be invisible here",
                      ref[TRIPWIRE_SYS][TRIPWIRE_PARTICLE], hMin(tripRho, T.restDensityBoundary));
    }

    // ---------------------------------------------------------------------
    // Compare.  Bit-exact everywhere; there is no tolerance in this file.
    // ---------------------------------------------------------------------
    size_t mm = 0, nTouched = 0, nUntouched = 0;
    for (uint32_t s = 0; s < NUM_SYS; ++s)
    {
        const auto got = downloadBuffer<uint32_t>(dev, sys[s].bDensity.h, sys[s].numParticles + PAD);
        for (uint32_t i = 0; i < sys[s].numParticles; ++i)
        {
            if (written[s][i]) ++nTouched; else ++nUntouched;
            const uint32_t want = f2u(ref[s][i]);
            if (got[i] != want)
            {
                if (mm < 12)
                    std::printf("  sys %u particle %u: got %.9g (0x%08x) want %.9g (0x%08x)%s\n",
                                s, i, u2f(got[i]), got[i], ref[s][i], want,
                                written[s][i] ? "" : "  [must be UNTOUCHED]");
                ++mm;
            }
        }
        for (uint32_t i = 0; i < PAD; ++i)
        {
            if (got[sys[s].numParticles + i] != POISON_U32)
            {
                if (mm < 12)
                    std::printf("  sys %u PAD slot %u overwritten: 0x%08x\n",
                                s, i, got[sys[s].numParticles + i]);
                ++mm;
            }
        }
    }

    // The decoy must be untouched by the static kernel and touched exactly once by
    // the rigid one, which is the whole point of B8.
    {
        uint32_t decoyWritten = 0;
        for (uint32_t i = 0; i < sys[DECOY].numParticles; ++i) decoyWritten += written[DECOY][i];
        PXG_CHECK_MSG(decoyWritten == 1u,
                      "the decoy must carry exactly one written particle (got %u)", decoyWritten);
    }

    std::printf("psBoundaryDensity: %u systems, %u active, %u rigid work indices; "
                "static clamped %zu / below %zu, rigid gate-closed %zu, clamped %zu; "
                "clamp observable at %zu; touched %zu, untouched %zu\n",
                NUM_SYS, unsigned(active.size()), NUM_TESTS, nStaticClamped, nStaticBelow,
                nRigidGateClosed, nRigidClamped, nClampObservable, nTouched, nUntouched);
    PXG_CHECK_MSG(mm == 0, "%zu boundary-density mismatches", mm);
}

// ---------------------------------------------------------------------------
// UPSTREAM CANDIDATE 16, made observable.
//
// ps_accumulateRigidDensity strides its work index by
// PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA == 512 while the kernel is
// LAUNCHED with blockDim.x == PxgParticleSystemKernelBlockDim::PS_COLLISION == 256.
// While nbIterationsPerBlock == 1 the `i * 512` term is multiplied by zero and the
// discrepancy cannot be seen.  It takes numTests > PS_COLLISION_GRID * PS_COLLISION
// = 262144 for a second iteration to exist, and then:
//
//   block bx, iteration i covers [512*bx + 512*i, 512*bx + 512*i + 256)
//
// so every work index whose remainder mod 512 is at least 256 is covered by NO
// block at all, while [512(bx+1), 512(bx+1)+256) is covered by both block bx's
// second iteration and block bx+1's first.
//
// BOTH halves are wrong, and writing this fixture is what established the second
// one.  The first draft asserted a single bit-exact value everywhere and claimed
// the double cover was harmless "because each thread recomputes the same value and
// the store is an assignment".  IT IS NOT: `rho` is seeded from
// mStaticDensity[tParticleId], the very array the kernel then stores to, so two
// threads in different blocks with no ordering between them form a READ-MODIFY-
// WRITE RACE.  The admissible outcomes are `static + sum` (both read before either
// wrote) and `static + 2*sum` (the second read saw the first write).  On llvmpipe
// this fixture observes the doubled value at a ratio of exactly 2.0.
//
// So the three regions are checked differently, and deliberately:
//   SKIPPED        -- must equal the sentinel EXACTLY.  This is the assertion a
//                     port that "fixes" the 512 stride fails.
//   SINGLY COVERED -- bit-exact against the mirror.
//   DOUBLY COVERED -- one of the two admissible race outcomes, counted and
//                     reported rather than averaged away.  That is the only place
//                     in this file where a single expected value is not demanded,
//                     and it is because the kernel is racy, not because the port
//                     is approximate.
//
// Candidate 16 is documented and NOT reported upstream.
// ---------------------------------------------------------------------------
static void runWorkIndexCase(PxgSlangComputeDevice* dev, uint32_t NUM_TESTS, uint32_t expectNbIter,
                             bool expectDoubleCover)
{
    const uint32_t nbBlocksRequired = (NUM_TESTS + BLOCK - 1u) / BLOCK;
    const uint32_t nbIter = (nbBlocksRequired + RIGID_GRID - 1u) / RIGID_GRID;
    PXG_CHECK_MSG(nbIter == expectNbIter, "this case needs exactly %u iterations per block (got %u)",
                  expectNbIter, nbIter);

    // One system, one contact per work index, one particle per work index.  The
    // inner loop and the material scalars are the previous test's job; this one is
    // about the work index and nothing else.
    const float SPIKY = 3.125f, CDI = 1.0f / 0.4f, RESTD = 0.05f, RDB = 1.0e9f;
    // RDB is huge so min() never binds and every write is fully observable.

    // The same off-by-one tripwire as psBoundaryDensity: workIndex == numTests is
    // reachable here too (262145 = 512*512 + 1), and the range arrays carry extra
    // entries naming a particle in the SKIPPED half, so a loosened guard trips the
    // skipped-half assertion below rather than reading zeroed slack.
    // The tripwire particle must be one this size actually SKIPS, and which index that
    // is depends on nbIter -- at 2 the skipped set is "remainder mod 512 >= 256", at 3
    // it is a different band pattern entirely.  So it is DERIVED from the coverage the
    // mirror computes below rather than hard-coded, and asserted to be skipped.
    const uint32_t RANGE_PAD = 8u;

    // The coverage the faithful loop produces, computed FIRST so the tripwire can be
    // placed on an index this size genuinely skips.
    std::vector<uint8_t> covered(NUM_TESTS, 0u);
    for (uint32_t bx = 0; bx < RIGID_GRID; ++bx)
        for (uint32_t idx = 0; idx < BLOCK; ++idx)
            for (uint32_t it = 0; it < nbIter; ++it)
            {
                const uint32_t w = it * ACCUM_DELTA + idx + nbIter * bx * BLOCK;
                if (w >= NUM_TESTS) break;
                covered[w] = covered[w] < 255u ? uint8_t(covered[w] + 1u) : covered[w];
            }
    uint32_t TRIPWIRE_PARTICLE = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < NUM_TESTS; ++w)
        if (!covered[w]) { TRIPWIRE_PARTICLE = w; break; }
    PXG_CHECK_MSG(TRIPWIRE_PARTICLE != 0xFFFFFFFFu,
                  "no skipped index at numTests=%u -- the tripwire has nowhere to live", NUM_TESTS);

    std::vector<uint32_t> rangeStart(NUM_TESTS + RANGE_PAD), rangeEnd(NUM_TESTS + RANGE_PAD);
    std::vector<uint8_t> contactBytes(size_t(NUM_TESTS + 1u) * 32u, 0);
    std::vector<float> pen(NUM_TESTS);
    for (uint32_t w = 0; w < NUM_TESTS; ++w)
    {
        rangeStart[w] = w;
        rangeEnd[w] = w + 1u;
        // A penetration that varies per work index, so a wrong index gives a wrong
        // value and not merely a wrong slot.
        pen[w] = (0.4f * float(hashu(w, 5501u) % 1000u) / 1000.0f) - RESTD;

        uint8_t* rec = &contactBytes[size_t(w) * 32u];
        const uint32_t u3 = f2u(pen[w]);
        std::memcpy(rec + 12, &u3, 4);
        const PxU64 rigidId = 0xBEEF0000ull | PxU64(w);
        std::memcpy(rec + 16, &rigidId, 8);
        const PxU64 pid = PxgEncodeParticleRef(0u, w);     // system 0, particle w
        std::memcpy(rec + 24, &pid, 8);
    }
    {   // the tripwire record, one past the live contacts
        uint8_t* rec = &contactBytes[size_t(NUM_TESTS) * 32u];
        const uint32_t u3 = f2u(0.5f / CDI - RESTD);
        std::memcpy(rec + 12, &u3, 4);
        const PxU64 pid = PxgEncodeParticleRef(0u, TRIPWIRE_PARTICLE);
        std::memcpy(rec + 24, &pid, 8);
        for (uint32_t k = 0; k < RANGE_PAD; ++k)
        {
            rangeStart[NUM_TESTS + k] = NUM_TESTS;
            rangeEnd[NUM_TESTS + k] = NUM_TESTS + 1u;
        }
    }

    // mStaticDensity carries one distinct sentinel per particle.  A skipped work
    // index must leave its particle EXACTLY at the sentinel.
    const uint32_t PAD = 16u;
    std::vector<uint32_t> dens(size_t(NUM_TESTS + PAD), POISON_U32);
    for (uint32_t i = 0; i < NUM_TESTS; ++i) dens[i] = f2u(densitySentinel(i));
    DevBuf bDensity = makeDev(dev, dens);
    SoaContacts bContacts = splitContacts(dev, contactBytes);
    DevBuf bStart = makeDev(dev, rangeStart);
    DevBuf bEnd = makeDev(dev, rangeEnd);
    std::vector<uint32_t> totalPairs = { NUM_TESTS };
    DevBuf bTotal = makeDev(dev, totalPairs);

    std::vector<PxgParticleSystem> descArr(1);
    std::memset(descArr.data(), 0xEE, sizeof(PxgParticleSystem));
    descArr[0].mCommonData.mNumParticles = NUM_TESTS;
    descArr[0].mCommonData.mParticleContactDistanceInv = CDI;
    descArr[0].mStaticDensity = reinterpret_cast<PxReal*>(size_t(bDensity.addr));
    descArr[0].mData.mSpiky1 = SPIKY;
    descArr[0].mData.mRestDensityBoundary = RDB;
    descArr[0].mData.mFluidRestOffset = RESTD;
    // mOneWayContactInfos / mOneWayContactCount stay 0xEE: the rigid kernel must
    // never read them, and a port that does faults rather than passing quietly.
    DevBuf bDesc = makeDev(dev, descArr);

    RigidPush push = { bDesc.addr, bContacts.infos.addr, bContacts.normalPens.addr, bTotal.addr, bStart.addr, bEnd.addr };
    {
        Kernel k("npPsBoundaryDensity.spv", "ps_accumulateRigidDensity", {}, sizeof(RigidPush));
        k.dispatch(RIGID_GRID, {}, push, 1u, 1u);
        dev->flush();
    }

    // The mirror: the loop as written, block by block.  `once` is the value a
    // single pass produces; `twice` is what a doubly-covered index produces when the
    // second thread's read sees the first thread's write.  Both are computed for
    // every index; which one is admissible depends on the coverage count.
    std::vector<float> once(NUM_TESTS), twice(NUM_TESTS);
    std::fill(covered.begin(), covered.end(), uint8_t(0));   // recounted alongside the values
    for (uint32_t i = 0; i < NUM_TESTS; ++i) { once[i] = densitySentinel(i); twice[i] = once[i]; }
    for (uint32_t bx = 0; bx < RIGID_GRID; ++bx)
    {
        for (uint32_t idx = 0; idx < BLOCK; ++idx)
        {
            for (uint32_t it = 0; it < nbIter; ++it)
            {
                const uint32_t w = it * ACCUM_DELTA + idx + nbIter * bx * BLOCK;
                if (w >= NUM_TESTS) break;
                const float x = hMax(0.0f, pen[w] + RESTD);
                const float contribution = hW(x, SPIKY, CDI);
                once[w] = hMin(densitySentinel(w) + contribution, RDB);
                // The racy composition: seed from the ALREADY-STORED value, i.e.
                // from once[w], and clamp again.
                twice[w] = hMin(once[w] + contribution, RDB);
                covered[w] = covered[w] < 255u ? uint8_t(covered[w] + 1u) : covered[w];
            }
        }
    }

    // The coverage pattern is itself an assertion.  It used to be spelled as the
    // closed form "remainder mod 512 below 256", which is the nbIter == 2
    // SPECIALISATION -- the general block stride is nbIter*256, not 512, and at
    // nbIter == 3 that closed form is simply false.  It is now compared against a
    // second, independently written enumeration of the same loop, so the check still
    // fires if a constant changes but no longer encodes a formula that only holds at
    // one size.
    size_t nCovered = 0, nSkipped = 0, nDouble = 0;
    {
        std::vector<uint8_t> expect(NUM_TESTS, 0u);
        for (uint32_t bx = 0; bx < RIGID_GRID; ++bx)
            for (uint32_t it = 0; it < nbIter; ++it)
            {
                const uint32_t base = it * ACCUM_DELTA + nbIter * bx * BLOCK;
                for (uint32_t idx = 0; idx < BLOCK; ++idx)
                {
                    const uint32_t w = base + idx;
                    if (w >= NUM_TESTS) break;
                    expect[w] = expect[w] < 255u ? uint8_t(expect[w] + 1u) : expect[w];
                }
            }
        for (uint32_t w = 0; w < NUM_TESTS; ++w)
            PXG_CHECK_MSG(covered[w] == expect[w],
                          "work index %u: mirror counted %u, enumeration counted %u",
                          w, unsigned(covered[w]), unsigned(expect[w]));
    }
    for (uint32_t w = 0; w < NUM_TESTS; ++w)
    {
        if (covered[w]) ++nCovered; else ++nSkipped;
        if (covered[w] > 1u) ++nDouble;
    }
    // Every size must SKIP something -- that is candidate 16's first half and the
    // deterministic discriminator.  Double cover is the second half and is a property
    // of nbIter == 2 specifically (there the block and iteration strides coincide);
    // at nbIter == 3 the faithful loop double-covers nothing at all.  So the caller
    // states which of the two this size is here to demonstrate, rather than the
    // helper assuming one shape fits both.
    PXG_CHECK_MSG(nSkipped > 0, "candidate 16 must skip something at numTests=%u", NUM_TESTS);
    PXG_CHECK_MSG((nDouble > 0) == expectDoubleCover,
                  "numTests=%u: double-covered %zu, expected %s", NUM_TESTS, nDouble,
                  expectDoubleCover ? "some" : "none");

    const auto got = downloadBuffer<uint32_t>(dev, bDensity.h, NUM_TESTS + PAD);
    size_t mm = 0, mmSkipped = 0, nRaceOnce = 0, nRaceTwice = 0;
    for (uint32_t w = 0; w < NUM_TESTS; ++w)
    {
        bool ok;
        if (covered[w] == 0u)
        {
            // The skipped half: the sentinel, untouched.  This is the assertion a
            // port which "fixes" the 512 stride fails.
            ok = (got[w] == f2u(densitySentinel(w)));
            if (!ok) ++mmSkipped;
        }
        else if (covered[w] == 1u)
        {
            ok = (got[w] == f2u(once[w]));                       // bit-exact
        }
        else
        {
            // Doubly covered: either race outcome, counted so the report says which
            // one this run actually produced rather than hiding the race.
            const bool isOnce = (got[w] == f2u(once[w]));
            const bool isTwice = (got[w] == f2u(twice[w]));
            if (isOnce) ++nRaceOnce;
            if (isTwice) ++nRaceTwice;
            ok = isOnce || isTwice;
        }
        if (!ok)
        {
            if (mm < 12)
                std::printf("  work %u (cover %u): got %.9g want %.9g%s\n", w,
                            unsigned(covered[w]), u2f(got[w]),
                            covered[w] == 0u ? densitySentinel(w) : once[w],
                            covered[w] > 1u ? " or the doubled value" : "");
            ++mm;
        }
    }
    for (uint32_t i = 0; i < PAD; ++i)
        if (got[NUM_TESTS + i] != POISON_U32) { ++mm; std::printf("  PAD %u overwritten\n", i); }

    std::printf("  workIndex[%u tests, %u iter/block]: covered %zu, SKIPPED %zu (candidate 16), "
                "double-covered %zu (race resolved once %zu / twice %zu)\n",
                NUM_TESTS, nbIter, nCovered, nSkipped, nDouble, nRaceOnce, nRaceTwice);
    if (mmSkipped)
        std::printf("  NOTE: %zu mismatches are on SKIPPED indices -- that is exactly what a "
                    "port which 'fixes' the 512 stride looks like\n", mmSkipped);
    PXG_CHECK_MSG(mm == 0, "%zu work-index mismatches at numTests=%u", mm, NUM_TESTS);
}

PXG_TEST(psBoundaryDensityWorkIndex)
{
    PxgSlangComputeDevice* dev = testDevice();

    // TWO sizes, and the second one is not padding -- it closes a hole the first
    // cannot see.
    //
    // At nbIter == 2 the block stride (nbIter*256 = 512) coincides with the iteration
    // stride (ACCUMULATE_DELTA = 512), and that coincidence makes the iteration stride
    // UNDER-DETERMINED: replacing `it * 512` with `it * 1024` in the shader yields a
    // byte-identical COVERED SET at this size and differs only in the MULTIPLICITY of
    // one 512-band (indices 512..767 drop from doubly to singly covered).  Since the
    // doubly-covered branch below accepts either race outcome, that mutant produced
    // zero failures -- the stride constant this test exists to pin was unconstrained
    // anywhere between 512 and 1024.  Found by adversarial review of the landed
    // fixture, then reproduced by enumeration before being fixed.
    //
    // At nbIter == 3 the two strides no longer coincide (block stride 768, iteration
    // stride 512), so the covered SET itself separates: `it*1024` leaves 512 indices
    // uncovered that the faithful loop covers, and `it*256` covers 256 the faithful
    // loop skips.  Both die on the skipped-half assertion, which is deterministic and
    // does not lean on how the race resolves.
    runWorkIndexCase(dev, BLOCK * RIGID_GRID + 1u, 2u, /*doubleCover*/ true);       // 262145
    runWorkIndexCase(dev, 2u * BLOCK * RIGID_GRID + 1u, 3u, /*doubleCover*/ false); // 524289
}

} // namespace
