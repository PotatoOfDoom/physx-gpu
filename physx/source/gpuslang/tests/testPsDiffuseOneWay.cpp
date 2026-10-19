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

// testPsDiffuseOneWay.cpp -- P9.13f, the fixture for
// ps_diffuseParticleOneWayCollision.
//
// ---------------------------------------------------------------------------
// THE ARRANGEMENT
// ---------------------------------------------------------------------------
// The kernel is a strided gather and a float fold; there is no engine primitive
// to call, so the oracle is a host mirror of the CUDA's control flow.  What
// carries it is (1) every offset generated from the engine headers and pinned by
// static_assert below, closing the P8.49a failure mode by construction, and
// (2) whole-region poison comparison over EVERY array the kernel can reach, so
// an extra write fails as loudly as a wrong one.
//
// ---------------------------------------------------------------------------
// WHAT THE GEOMETRY IS BUILT TO DISCRIMINATE
// ---------------------------------------------------------------------------
//   * TWO systems with DIFFERENT particle counts (700 and 173, neither a
//     multiple of the 256 block nor of the 12 contact cap), reached through an
//     activeParticleSystems indirection that is NOT the identity;
//   * FOUR active slots against a grid y of 2, the extra two pointing at TRAP
//     systems with a published count of zero -- so a port that takes the system
//     from blockIdx.x (which runs 0..2 here) lands on a system that corrects
//     nothing and leaves a third of system A untouched, rather than reading out
//     of bounds;
//   * a STAGING slot [1] seeded LARGER than the published slot [0] on both
//     systems, so a port that reads the wrong slot of the two-slot pair walks
//     past the live count into poisoned contacts (O3);
//   * contact counts covering 0 (the no-store case), 12 EXACTLY (the clamp
//     boundary, which separates min(12,.) from min(11,.)), and 13 and 20 (the
//     only counts at which the clamp does anything);
//   * SIXTEEN contact planes allocated where twelve are legal, the upper four
//     filled with 1e6-scale traps -- so dropping the clamp does not read zeroes
//     and quietly agree (the P9.13d/P9.13e lesson: poisoned slack behind every
//     bound);
//   * penetrations of BOTH signs, so O6's double negation is exercised in both
//     directions rather than only on the physical case;
//   * a CONSTRUCTED ORDER SENSITIVITY: one particle whose three corrections are
//     +1e8, +1, -1e8, which a left-to-right fold and any other order round
//     differently.  Float data does not land on that by accident, so it is
//     built, and the fixture DEMANDS that the two orders differ bitwise before
//     it will accept the run.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

// ---------------------------------------------------------------------------
// The ABI, generated from the engine headers and pinned here.
// ---------------------------------------------------------------------------
static_assert(sizeof(PxgParticleContactInfo) == 16, "PxgParticleContactInfo is one float4");
static_assert(PxgParticleContactInfo::MaxStaticContactsPerParticle == 12,
              "the static-contact cap the kernel clamps to");
static_assert(offsetof(PxgParticleSystem, mNumDiffuseParticles) == 448, "@448");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) == 512, "@512");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) == 520, "@520");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactInfos) == 552, "@552");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayForces) == 560, "@560");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayNodeIndex) == 568, "@568");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactCount) == 576, "@576");

struct F4 { float x, y, z, w; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

// psDiffuseOneWay.slang's PsDiffOneWayPush, byte for byte.
struct PsOneWayBlob
{
    uint64_t particleSystems;        // @0
    uint64_t activeParticleSystems;  // @8
    uint32_t count;                  // @16   (DEAD in the kernel)
    uint32_t pad0;                   // -> 24
};
static_assert(sizeof(PsOneWayBlob) == 24, "engine ABI blob size");

static const uint32_t MAXC = 12u;        // MaxStaticContactsPerParticle
static const uint32_t ALLOC_C = 16u;     // four TRAP planes past the legal twelve
static const uint32_t SLACK = 137u;      // poisoned margin past every live count
static const uint32_t NB_SYS_SLOTS = 6u;
static const uint32_t SYS_A = 4u;        // neither 0 nor 1: a dropped indirection shows
static const uint32_t SYS_B = 1u;
static const uint32_t TRAP_S0 = 2u;
static const uint32_t TRAP_S1 = 5u;
static const uint32_t BLOCK = 256u;      // PxgParticleSystemKernelBlockDim::PS_SOLVE
static const uint32_t GRID_Y = 2u;

static const float POISON_F = -7.5e3f;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;

// The particle whose three corrections are engineered to be order-sensitive.
static const uint32_t ORDER_PI = 137u;   // in system A, inside block 0

struct SysSpec
{
    const char* label;
    uint32_t slot;        // index into the descriptor array
    uint32_t published;   // mNumDiffuseParticles[0]
    int32_t  staged;      // mNumDiffuseParticles[1]  -- a trap, larger than published
    uint32_t tag;         // seeds every value in this system's arrays
};

static const SysSpec SPEC_A{ "sysA", SYS_A, 700u, 900,  1u };
static const SysSpec SPEC_B{ "sysB", SYS_B, 173u,   4,  2u };

// Contact counts, cycled per particle.  Both cycles contain 0, exactly 12, and
// values above 12; the two cycles differ so a system mix-up is visible.
static uint32_t rawCountFor(const SysSpec& s, uint32_t pi)
{
    static const uint32_t CYCLE_A[10] = { 0u, 1u, 2u, 3u, 5u, 12u, 13u, 20u, 7u, 4u };
    static const uint32_t CYCLE_B[6]  = { 2u, 0u, 12u, 6u, 1u, 15u };
    if (s.tag == 1u)
    {
        if (pi == ORDER_PI) return 3u;   // the constructed order sensitivity
        return CYCLE_A[pi % 10u];
    }
    return CYCLE_B[pi % 6u];
}

// The contact at (system, particle, slot).  Slots >= MAXC are TRAPS: reading one
// moves the particle by ~1e6 and cannot be mistaken for a rounding difference.
static F4 contactFor(const SysSpec& s, uint32_t pi, uint32_t c)
{
    if (c >= MAXC)
        return F4{ 1.0e6f, -1.0e6f, 1.0e6f, 1.0e6f };

    if (s.tag == 1u && pi == ORDER_PI)
    {
        // deltaP = -normal * pen, so these give +1e8, -1e8, +1 on x -- and that
        // ORDER matters.  Forward the +1 lands on a value near zero and
        // survives; reversed it is added first, is rounded away by the 1e8 pair
        // (whose ulp is 8), and the fold ends one lower.  The +1e8, +1, -1e8
        // spelling looks equally hostile and is NOT: it is symmetric under
        // reversal and both orders agree, which cost this fixture a run.
        if (c == 0u) return F4{ -1.0f, 0.0f, 0.0f, 1.0e8f };
        if (c == 1u) return F4{  1.0f, 0.0f, 0.0f, 1.0e8f };
        return          F4{ -1.0f, 0.0f, 0.0f, 1.0f };
    }

    // Non-unit, non-axis-aligned, all three components distinct, and derived
    // from (system, particle, slot) so a wrong index reads a different number.
    const float k = float(1000 * int(s.tag) + int(pi) % 97 + 1);
    const float n0 = 0.125f + 0.001f * float(c) + 1.0e-4f * k;
    const float n1 = -0.375f - 0.002f * float(c) - 2.0e-4f * k;
    const float n2 = 0.625f + 0.003f * float(c) + 3.0e-4f * k;
    // Every third contact SEPARATES (positive penetration), the rest penetrate.
    const float pen = ((pi + c) % 3u == 0u) ? (0.011f + 0.0007f * float(c))
                                            : -(0.023f + 0.0013f * float(c) + 1.0e-5f * k);
    return F4{ n0, n1, n2, pen };
}

// The seeded predicted position.  The .w is a distinct lifetime per particle so
// O10's pass-through is checkable.
static F4 seedPos(const SysSpec& s, uint32_t pi)
{
    return F4{ 10.0f * float(s.tag) + 0.5f * float(pi),
               -3.0f * float(s.tag) - 0.25f * float(pi),
                7.0f * float(s.tag) + 0.125f * float(pi),
                0.5f + 0.001f * float(pi) + float(s.tag) };
}

struct Side
{
    DevBuf bPos, bOrigin, bContacts, bCounts, bTwo, bForces, bNodes;
};

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bDesc, bActive, bZeroCount;
    Side A, B;
    PsOneWayBlob blob{};

    template <typename T>
    DevBuf mk(PxgSlangComputeDevice* dev, const std::vector<T>& d)
    {
        DevBuf b;
        b.h = createBufferFrom(dev, d);
        b.addr = dev->getBufferDeviceAddress(b.h);
        owned.push_back(b.h);
        return b;
    }
    void destroy(PxgSlangComputeDevice* dev)
    {
        for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
        owned.clear();
    }
};

static uint32_t posSlots(const SysSpec& s) { return s.published + SLACK; }
static uint32_t contactSlots(const SysSpec& s) { return s.published * ALLOC_C + SLACK; }

static void buildSide(PxgSlangComputeDevice* dev, Rig& R, const SysSpec& s, Side& S)
{
    // The predicted positions, with a poisoned margin past the live count so an
    // over-bound write lands somewhere the comparison looks.
    std::vector<F4> pos(posSlots(s), F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    for (uint32_t i = 0; i < s.published; ++i) pos[i] = seedPos(s, i);
    S.bPos = R.mk(dev, pos);

    // The ORIGIN array, the neighbour this kernel must not touch (O9).  Its
    // contents differ from the predicted array everywhere.
    std::vector<F4> origin(posSlots(s));
    for (uint32_t i = 0; i < origin.size(); ++i)
        origin[i] = F4{ -1.0f - float(i), -2.0f - float(i), -3.0f - float(i), -4.0f - float(i) };
    S.bOrigin = R.mk(dev, origin);

    // The contact list, COLUMN-MAJOR: contact c of particle pi at
    // pi + c * published.  Slots at c >= MAXC carry traps.
    std::vector<F4> contacts(contactSlots(s), F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    for (uint32_t c = 0; c < ALLOC_C; ++c)
        for (uint32_t pi = 0; pi < s.published; ++pi)
            contacts[size_t(pi) + size_t(c) * s.published] = contactFor(s, pi, c);
    S.bContacts = R.mk(dev, contacts);

    std::vector<uint32_t> counts(posSlots(s), POISON_U32);
    for (uint32_t i = 0; i < s.published; ++i) counts[i] = rawCountFor(s, i);
    S.bCounts = R.mk(dev, counts);

    // The two-slot count pair: [0] published, [1] a LARGER trap (O3).
    S.bTwo = R.mk(dev, std::vector<int32_t>{ int32_t(s.published), s.staged });

    // The producer's other two outputs.  This kernel reads neither and writes
    // neither; both are checked back unchanged.
    S.bForces = R.mk(dev, std::vector<float>(contactSlots(s), POISON_F));
    S.bNodes = R.mk(dev, std::vector<uint64_t>(contactSlots(s), 0xDEADBEEFCAFEF00Dull));
}

static void fillDesc(PxgParticleSystem& d, const Side& S)
{
    d.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(S.bTwo.addr));
    d.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(S.bPos.addr));
    d.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(S.bOrigin.addr));
    d.mDiffuseOneWayContactInfos =
        reinterpret_cast<PxgParticleContactInfo*>(size_t(S.bContacts.addr));
    d.mDiffuseOneWayForces = reinterpret_cast<float*>(size_t(S.bForces.addr));
    d.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(S.bNodes.addr));
    d.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(S.bCounts.addr));
}

static void buildRig(PxgSlangComputeDevice* dev, Rig& R)
{
    buildSide(dev, R, SPEC_A, R.A);
    buildSide(dev, R, SPEC_B, R.B);

    // The trap systems publish ZERO particles, so a port that takes the system
    // from the wrong grid axis corrects nothing for those blocks instead of
    // reading out of bounds -- a visible hole, not undefined behaviour.
    R.bZeroCount = R.mk(dev, std::vector<int32_t>{ 0, 0 });

    std::vector<PxgParticleSystem> descArr(NB_SYS_SLOTS);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));
    fillDesc(descArr[SYS_A], R.A);
    fillDesc(descArr[SYS_B], R.B);
    for (uint32_t t : { TRAP_S0, TRAP_S1 })
    {
        fillDesc(descArr[t], R.A);   // real, reachable pointers ...
        descArr[t].mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bZeroCount.addr));
    }                                 // ... but a published count of zero
    R.bDesc = R.mk(dev, descArr);

    // NOT the identity, and four entries against a grid y of 2.
    R.bActive = R.mk(dev, std::vector<uint32_t>{ SYS_A, SYS_B, TRAP_S0, TRAP_S1 });

    R.blob.particleSystems = R.bDesc.addr;
    R.blob.activeParticleSystems = R.bActive.addr;
    // O2: DEAD.  Deliberately SMALLER than either particle count, so a port that
    // mistakes it for the bound leaves all but the first seven particles alone.
    R.blob.count = 7u;
}

// ---------------------------------------------------------------------------
// The host mirror: the CUDA's control flow, transcribed once.
// ---------------------------------------------------------------------------
static std::vector<F4> predict(const SysSpec& s)
{
    std::vector<F4> out(posSlots(s), F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    for (uint32_t i = 0; i < s.published; ++i) out[i] = seedPos(s, i);

    for (uint32_t pi = 0; pi < s.published; ++pi)
    {
        const uint32_t raw = rawCountFor(s, pi);
        const uint32_t n = raw < MAXC ? raw : MAXC;   // O5
        if (n == 0u) continue;                        // O8: no store at all

        const F4 p0 = out[pi];
        float cx = p0.x, cy = p0.y, cz = p0.z;
        for (uint32_t c = 0; c < n; ++c)              // O7: strict left-to-right
        {
            const F4 k = contactFor(s, pi, c);        // O4: (pi, c), column-major
            cx += -k.x * k.w;                         // O6
            cy += -k.y * k.w;
            cz += -k.z * k.w;
        }
        out[pi] = F4{ cx, cy, cz, p0.w };             // O10
    }
    return out;
}

static void checkSide(PxgSlangComputeDevice* dev, const SysSpec& s, const Side& S)
{
    const std::vector<F4> E = predict(s);
    const auto pos = downloadBuffer<F4>(dev, S.bPos.h, posSlots(s));
    const auto origin = downloadBuffer<F4>(dev, S.bOrigin.h, posSlots(s));
    const auto contacts = downloadBuffer<F4>(dev, S.bContacts.h, contactSlots(s));
    const auto counts = downloadBuffer<uint32_t>(dev, S.bCounts.h, posSlots(s));
    const auto two = downloadBuffer<int32_t>(dev, S.bTwo.h, 2);
    const auto forces = downloadBuffer<float>(dev, S.bForces.h, contactSlots(s));
    const auto nodes = downloadBuffer<uint64_t>(dev, S.bNodes.h, contactSlots(s));

    size_t bad = 0;
    for (uint32_t i = 0; i < posSlots(s); ++i)
    {
        if (std::memcmp(&pos[i], &E[i], sizeof(F4)) != 0)
        {
            if (bad < 5)
                std::printf("    %s particle %u (raw count %u): got (%g,%g,%g,%g) "
                            "expected (%g,%g,%g,%g)\n", s.label, i,
                            i < s.published ? rawCountFor(s, i) : 0u,
                            pos[i].x, pos[i].y, pos[i].z, pos[i].w,
                            E[i].x, E[i].y, E[i].z, E[i].w);
            ++bad;
        }
    }
    PXG_CHECK_MSG(bad == 0, "%s: %zu predicted-position slots wrong (the %u-slot poisoned "
                  "margin past the live count is compared too)", s.label, bad, SLACK);

    // Everything else the kernel can reach must come back untouched.
    size_t badOrigin = 0;
    for (uint32_t i = 0; i < posSlots(s); ++i)
    {
        const F4 want{ -1.0f - float(i), -2.0f - float(i), -3.0f - float(i), -4.0f - float(i) };
        if (std::memcmp(&origin[i], &want, sizeof(F4)) != 0) ++badOrigin;
    }
    PXG_CHECK_MSG(badOrigin == 0, "%s: %zu slots of the SORTED ORIGIN array were written -- "
                  "this kernel writes the PREDICTED array (O9)", s.label, badOrigin);

    size_t badContacts = 0, badCounts = 0, badSide = 0;
    for (uint32_t c = 0; c < ALLOC_C; ++c)
        for (uint32_t pi = 0; pi < s.published; ++pi)
        {
            const F4 want = contactFor(s, pi, c);
            if (std::memcmp(&contacts[size_t(pi) + size_t(c) * s.published], &want,
                            sizeof(F4)) != 0) ++badContacts;
        }
    for (uint32_t i = 0; i < posSlots(s); ++i)
    {
        const uint32_t want = i < s.published ? rawCountFor(s, i) : POISON_U32;
        if (counts[i] != want) ++badCounts;
    }
    for (uint32_t i = 0; i < contactSlots(s); ++i)
    {
        if (std::memcmp(&forces[i], &POISON_F, sizeof(float)) != 0) ++badSide;
        if (nodes[i] != 0xDEADBEEFCAFEF00Dull) ++badSide;
    }
    PXG_CHECK_MSG(badContacts == 0, "%s: %zu contact records were modified", s.label, badContacts);
    PXG_CHECK_MSG(badCounts == 0, "%s: %zu contact counts were modified -- this kernel is the "
                  "CONSUMER and clears nothing", s.label, badCounts);
    PXG_CHECK_MSG(badSide == 0, "%s: %zu impulse/node slots were modified", s.label, badSide);
    PXG_CHECK_MSG(two[0] == int32_t(s.published) && two[1] == s.staged,
                  "%s: the two-slot count pair changed (%d,%d), expected (%d,%d)",
                  s.label, two[0], two[1], int32_t(s.published), s.staged);
}

static void runAll(PxgSlangComputeDevice* dev)
{
    Rig R;
    buildRig(dev, R);

    {
        Kernel k("psDiffuseOneWay.spv", "ps_diffuseParticleOneWayCollision", {},
                 sizeof(PsOneWayBlob));
        PXG_CHECK_MSG(k.isValid(), "the one-way collision pipeline must build");
        if (!k.isValid()) { R.destroy(dev); return; }
        // Grid (ceil(maxDiffuseParticles/256), nbActiveSystems, 1), block 256.
        const uint32_t gridX = (SPEC_A.published + BLOCK - 1u) / BLOCK;
        k.dispatch(gridX, {}, R.blob, GRID_Y, 1u);
    }
    dev->flush();

    checkSide(dev, SPEC_A, R.A);
    checkSide(dev, SPEC_B, R.B);

    // ---- coverage, demanded rather than designed ----
    {
        uint32_t zero = 0, atCap = 0, overCap = 0, posPen = 0, negPen = 0;
        for (const SysSpec* s : { &SPEC_A, &SPEC_B })
            for (uint32_t pi = 0; pi < s->published; ++pi)
            {
                const uint32_t raw = rawCountFor(*s, pi);
                if (raw == 0u) ++zero;
                if (raw == MAXC) ++atCap;
                if (raw > MAXC) ++overCap;
                for (uint32_t c = 0; c < (raw < MAXC ? raw : MAXC); ++c)
                    (contactFor(*s, pi, c).w > 0.0f ? posPen : negPen)++;
            }
        PXG_CHECK_MSG(zero > 0, "no particle had zero contacts -- O8 is untested");
        PXG_CHECK_MSG(atCap > 0, "no particle had EXACTLY %u contacts -- min(%u,.) and "
                      "min(%u,.) are indistinguishable", MAXC, MAXC, MAXC - 1u);
        PXG_CHECK_MSG(overCap > 0, "no particle exceeded the cap -- O5's clamp is dead");
        PXG_CHECK_MSG(posPen > 0 && negPen > 0, "penetrations were all one sign (%u positive, "
                      "%u negative) -- O6 is only half exercised", posPen, negPen);

        // The CONSTRUCTED order sensitivity, verified on the host before it is
        // relied on: folding the same three corrections right-to-left must give
        // a DIFFERENT bit pattern, or a reordering mutant would be equivalent.
        const F4 p0 = seedPos(SPEC_A, ORDER_PI);
        float fwd = p0.x, rev = p0.x;
        for (uint32_t c = 0; c < 3u; ++c)
        {
            const F4 k = contactFor(SPEC_A, ORDER_PI, c);
            fwd += -k.x * k.w;
        }
        for (int c = 2; c >= 0; --c)
        {
            const F4 k = contactFor(SPEC_A, ORDER_PI, uint32_t(c));
            rev += -k.x * k.w;
        }
        PXG_CHECK_MSG(std::memcmp(&fwd, &rev, sizeof(float)) != 0,
                      "particle %u's contact fold is order-INSENSITIVE (%g both ways) -- "
                      "a reordering mutant cannot be killed", ORDER_PI, fwd);
        std::printf("    [ note ] %u zero-contact, %u at the cap, %u over it; %u separating "
                    "and %u penetrating contacts applied; particle %u folds to %g forward "
                    "and %g reversed\n", zero, atCap, overCap, posPen, negPen,
                    ORDER_PI, double(fwd), double(rev));
    }

    R.destroy(dev);
}

} // namespace

PXG_TEST(psDiffuseOneWay)
{
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}
