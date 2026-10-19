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

// testPsDiffuseCreate.cpp -- P9.13d, the fixture for ps_diffuseParticleCreate.
//
// ---------------------------------------------------------------------------
// WHAT IS DETERMINISTIC HERE, AND WHAT IS A RACE
// ---------------------------------------------------------------------------
// The slot a spawning particle gets comes from an atomicAdd, so WHICH index a
// given particle lands on is a race and is not asserted (the P9.11 rule).  What
// IS determined, and is:
//
//   * WHICH particles qualify -- `r0 * intensity > threshold` is a pure function
//     of the inputs, so the host computes the exact qualifier set;
//   * the STAGING COUNTER's final value -- the atomic runs for every qualifier,
//     capacity or not (C6), so it is exactly the qualifier count;
//   * the number of records actually WRITTEN -- min(qualifiers, capacity);
//   * the SET of records written -- every written record must be a member of the
//     predicted set, bit for bit, and when the capacity is not exceeded the two
//     sets must be equal.
//
// The comparison is therefore a multiset match on a bitwise key, not an
// index-by-index one.  Nothing here depends on the order the atomic hands out.
//
// ---------------------------------------------------------------------------
// WHAT THE GEOMETRY IS BUILT TO DISCRIMINATE
// ---------------------------------------------------------------------------
//   * TWO systems reached through a NON-IDENTITY active list with a DECOY
//     descriptor between them, so a kernel that uses the grid axis directly
//     visits the decoy and misses a live system (the P9.13b lesson);
//   * particle-buffer counts that cross the 256-thread block boundary, so
//     `gid.x * 256` is live rather than dead code (also P9.13b);
//   * a particle buffer with mDiffuseParticleBufferIndex == 0xffffffff -- the
//     common path in any scene that mixes plain and diffuse buffers;
//   * a particle->diffuse buffer mapping that is NOT the identity, so the two
//     index spaces cannot be confused for one;
//   * a NON-ZERO mParticleBufferRunsum and a NON-IDENTITY unsorted->sorted
//     mapping, so all three index spaces (buffer-local, flat, sorted) differ;
//   * a random table SHORTER than the particle count and of a size that divides
//     none of the counts, so the `% randomTableSize` wrap is exercised;
//   * one diffuse buffer whose capacity is SMALLER than its qualifier count
//     (C6/C7: the counter overshoots, the writes stop), one with capacity 0
//     (C12), and one comfortable;
//   * fluid and non-fluid phases interleaved, so the phase test is not
//     satisfied by the whole array;
//   * every diffuse array allocated LARGER than its capacity, with the slack
//     required to come back poison.  Sizing it at the capacity is what let a
//     dropped capacity guard survive the first sweep: the over-capacity writes
//     landed outside the region the check read, and llvmpipe neither faulted nor
//     showed them.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxParticleBuffer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

struct F4 { float x, y, z, w; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// psDiffuseCreate.slang's PsDiffCreatePush, byte for byte.
struct PsDiffCreateBlob
{
    uint64_t particleSystems;        // @0
    uint64_t activeParticleSystems;  // @8
    uint64_t randomTable;            // @16
    uint32_t randomTableSize;        // @24
    float dt;                        // @28 -> 32
};
static_assert(sizeof(PsDiffCreateBlob) == 32, "engine ABI blob size");

static_assert(offsetof(PxgParticleSimBuffer, mDiffuseParticleBufferIndex) == 40,
              "mDiffuseParticleBufferIndex @40");
static_assert(offsetof(PxgParticleSimBuffer, mNumActiveParticles) == 32, "@32");
static_assert(offsetof(PxgParticleSystem, mSortedPositions_InvMass) == 24, "@24");
static_assert(offsetof(PxgParticleSystem, mSortedVelocities) == 32, "@32");
static_assert(offsetof(PxgParticleSystem, mSortedPhaseArray) == 40, "@40");
static_assert(offsetof(PxgParticleSystem, mUnsortedToSortedMapping) == 48, "@48");
static_assert(offsetof(PxgParticleSystem, mDiffusePotentials) == 472, "@472");
static_assert(offsetof(PxgParticleSystem, mParticleSimBuffers) == 400, "@400");
static_assert(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408, "@408");

// ---------------------------------------------------------------------------
// The scene.
// ---------------------------------------------------------------------------
static const uint32_t NB_SYS_SLOTS = 4u;
static const uint32_t SYS_A = 2u;         // the active list is { 2, 0 }: a decoy at 1
static const uint32_t SYS_B = 0u;
static const uint32_t GRID_Y = 5u;        // > either system's particle-buffer count
static const uint32_t RANDOM_TABLE_SIZE = 97u;   // prime, and < every particle count
static const float DT = 1.0f / 47.0f;     // not 1/60: a wrong dt is visible
static const float REST_OFFSET_A = 0.037f;
static const float REST_OFFSET_B = 0.061f;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;
static const float POISON_F = -9.0f;
static const uint32_t PHASE_FLUID_BIT = 0x00400000u;
// The slack past every diffuse buffer's capacity -- see checkSide.
static const uint32_t DIFFUSE_ALLOC_MARGIN = 137u;

// A particle buffer: how many particles, and which diffuse buffer it feeds
// (0xffffffff for none).
struct PBufSpec { uint32_t numActive; uint32_t diffuseIndex; };
// A diffuse buffer: its capacity and its spawn parameters.
struct DBufSpec { uint32_t capacity; float threshold, lifetime, kew, pw, dw; };

// System A: three particle buffers, two of them feeding diffuse buffers, and the
// mapping is not the identity (buffer 0 -> diffuse 1, buffer 2 -> diffuse 0).
static const PBufSpec A_PBUFS[3] = {
    { 700u, 1u },            // > 256: three blocks
    { 410u, 0xffffffffu },   // NO diffuse side -- the common path
    { 300u, 0u },            // > 256: two blocks
};
static DBufSpec A_DBUFS[2] = {
    //  cap  threshold lifetime  kew    pw     dw
    {   64u,     0.35f,   4.0f, 0.60f, 1.30f, 0.90f },  // small: the capacity bites
    { 4096u,     0.80f,   7.0f, 0.25f, 2.10f, 0.40f },  // comfortable
};
// System B: two particle buffers; one feeds a ZERO-capacity diffuse buffer.
static const PBufSpec B_PBUFS[2] = {
    { 520u, 0u },            // -> capacity 0 (C12)
    { 330u, 1u },
};
static DBufSpec B_DBUFS[2] = {
    {    0u,     0.50f,   3.0f, 0.40f, 1.00f, 0.70f },  // capacity 0
    { 2048u,     0.20f,   5.5f, 0.90f, 0.75f, 1.60f },
};

struct SysSpec
{
    uint32_t nbP, nbD;
    const PBufSpec* p;
    const DBufSpec* d;
    float restOffset;
    uint32_t tag;
};

// A spawned record, compared as bit patterns.
struct Rec
{
    uint32_t px, py, pz, pw, vx, vy, vz, vw;
    bool operator<(const Rec& o) const
    {
        if (px != o.px) return px < o.px;
        if (py != o.py) return py < o.py;
        if (pz != o.pz) return pz < o.pz;
        if (pw != o.pw) return pw < o.pw;
        if (vx != o.vx) return vx < o.vx;
        if (vy != o.vy) return vy < o.vy;
        if (vz != o.vz) return vz < o.vz;
        return vw < o.vw;
    }
    bool operator==(const Rec& o) const
    {
        return px == o.px && py == o.py && pz == o.pz && pw == o.pw
            && vx == o.vx && vy == o.vy && vz == o.vz && vw == o.vw;
    }
};

struct SideArrays
{
    std::vector<F4> sortedPos, sortedVel;
    std::vector<uint32_t> sortedPhase, u2s, runsum;
    std::vector<float> potentials;   // float2 per particle, flat
    uint32_t total = 0;
};

// The particle data.  Everything is a deterministic function of (system tag,
// flat index) so a wrong index space produces a different number rather than a
// shifted one.
static void buildArrays(const SysSpec& S, SideArrays& A)
{
    uint32_t total = 0;
    A.runsum.resize(S.nbP);
    for (uint32_t b = 0; b < S.nbP; ++b) { A.runsum[b] = total; total += S.p[b].numActive; }
    A.total = total;

    A.sortedPos.resize(total);
    A.sortedVel.resize(total);
    A.sortedPhase.resize(total);
    A.u2s.resize(total);
    A.potentials.resize(size_t(total) * 2u);

    // A PERMUTATION, and not the identity: unsorted i -> sorted (5i + 7) mod n.
    // gcd(5, n) == 1 for both totals below, checked by the coverage block.
    for (uint32_t i = 0; i < total; ++i)
        A.u2s[i] = (5u * i + 7u) % total;

    for (uint32_t s = 0; s < total; ++s)
    {
        const float t = float(s) * 0.013f + float(S.tag) * 3.7f;
        A.sortedPos[s] = F4{ std::sin(t) * 2.0f, std::cos(t * 1.7f) * 1.5f,
                             std::sin(t * 0.6f) * 3.0f, 0.5f + 0.001f * float(s) };
        A.sortedVel[s] = F4{ std::cos(t) * 0.9f, std::sin(t * 2.1f) * 1.1f,
                             std::cos(t * 0.3f) * 0.7f, -1.0f };
        // Fluid on ~3 of every 4, with the non-fluid ones spread rather than
        // clustered, and a second bit set so a `== FLUID` test would fail where
        // a `& FLUID` test passes.
        A.sortedPhase[s] = ((s % 4u) == 3u) ? 0x00000011u : (PHASE_FLUID_BIT | 0x00000011u);
        A.potentials[size_t(s) * 2 + 0] = std::sin(t * 0.9f) * 0.8f;         // divergence
        A.potentials[size_t(s) * 2 + 1] = 0.6f + std::cos(t * 1.3f) * 0.9f;  // pressure
    }
}

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bDesc, bActive, bRandom;
    // per side
    DevBuf bPos[2], bVel[2], bPhase[2], bU2s[2], bRunsum[2], bPot[2];
    DevBuf bPBufs[2], bDBufs[2], bTwo[2], bPinned[2];
    DevBuf bDPos[2][2], bDVel[2][2];
    PsDiffCreateBlob blob{};

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

static const SysSpec SPEC_A{ 3u, 2u, A_PBUFS, A_DBUFS, REST_OFFSET_A, 0u };
static const SysSpec SPEC_B{ 2u, 2u, B_PBUFS, B_DBUFS, REST_OFFSET_B, 1u };

// Set in runAll: the exact `r0 * intensity` of one particle, installed as its
// buffer's threshold so `>` and `>=` disagree on it.  See the note there.
static uint32_t gExactTieCount = 0;

static std::vector<float> gRandom;

static void buildSide(PxgSlangComputeDevice* dev, Rig& R, uint32_t side, const SysSpec& S,
                      const SideArrays& A, PxgParticleSystem& desc)
{
    R.bPos[side] = R.mk(dev, A.sortedPos);
    R.bVel[side] = R.mk(dev, A.sortedVel);
    R.bPhase[side] = R.mk(dev, A.sortedPhase);
    R.bU2s[side] = R.mk(dev, A.u2s);
    R.bRunsum[side] = R.mk(dev, A.runsum);
    R.bPot[side] = R.mk(dev, A.potentials);

    // The two-slot counts per diffuse buffer, both slots seeded non-zero so a
    // kernel that writes the wrong one is visible.
    std::vector<int32_t> two(size_t(S.nbD) * 2u);
    for (uint32_t d = 0; d < S.nbD; ++d)
    {
        two[size_t(d) * 2 + 0] = 111 + int32_t(d);   // published: must NOT be touched
        two[size_t(d) * 2 + 1] = 0;                  // staging: the kernel appends here
    }
    R.bTwo[side] = R.mk(dev, two);
    R.bPinned[side] = R.mk(dev, std::vector<int32_t>(S.nbD, int32_t(POISON_U32)));

    std::vector<PxgParticleDiffuseSimBuffer> dbufs(S.nbD);
    std::memset(dbufs.data(), 0, dbufs.size() * sizeof(PxgParticleDiffuseSimBuffer));
    for (uint32_t d = 0; d < S.nbD; ++d)
    {
        const uint32_t cap = S.d[d].capacity;
        // THE ALLOCATION IS BIGGER THAN THE CAPACITY, and that is load-bearing.
        // The kernel may only ever write below mMaxNumParticles; sizing the
        // array AT the capacity puts every over-capacity write outside the
        // region the check reads, and on llvmpipe those writes neither fault nor
        // show up -- which is exactly how a dropped capacity guard survived the
        // first sweep.  The margin is odd so it cannot coincide with a count.
        const uint32_t alloc = cap + DIFFUSE_ALLOC_MARGIN;
        R.bDPos[side][d] = R.mk(dev, std::vector<F4>(alloc,
                                 F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
        R.bDVel[side][d] = R.mk(dev, std::vector<F4>(alloc,
                                 F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
        PxDiffuseParticleParams pr;
        pr.threshold = S.d[d].threshold;
        pr.lifetime = S.d[d].lifetime;
        pr.kineticEnergyWeight = S.d[d].kew;
        pr.pressureWeight = S.d[d].pw;
        pr.divergenceWeight = S.d[d].dw;
        dbufs[d].mParams = pr;
        dbufs[d].mDiffusePositions_LifeTime =
            reinterpret_cast<float4*>(size_t(R.bDPos[side][d].addr));
        dbufs[d].mDiffuseVelocities = reinterpret_cast<float4*>(size_t(R.bDVel[side][d].addr));
        dbufs[d].mMaxNumParticles = cap;
        dbufs[d].mNumDiffuseParticles =
            reinterpret_cast<int*>(size_t(R.bTwo[side].addr + uint64_t(d) * 2u * sizeof(int32_t)));
        dbufs[d].mNumActiveDiffuseParticles =
            reinterpret_cast<int*>(size_t(R.bPinned[side].addr + uint64_t(d) * sizeof(int32_t)));
        dbufs[d].mStartIndex = 4242u + d;
        dbufs[d].mFlags = 0xCAFE0000u | d;
    }
    R.bDBufs[side] = R.mk(dev, dbufs);

    // The array runs to GRID_Y, not to the live count, and every entry past the
    // count is a TRAP rather than a zero: a large particle count and a VALID
    // diffuse index, so a kernel that reads past mNumParticleBuffers spawns into
    // a real buffer instead of quietly reading zeroes and returning.  Zero-filled
    // slack is what let `g-pbuf-guard-dropped` survive the first sweep.
    std::vector<PxgParticleSimBuffer> pbufs(GRID_Y);
    std::memset(pbufs.data(), 0, pbufs.size() * sizeof(PxgParticleSimBuffer));
    for (uint32_t b = 0; b < GRID_Y; ++b)
    {
        const bool live = b < S.nbP;
        pbufs[b].mNumActiveParticles = live ? S.p[b].numActive : 512u;
        pbufs[b].mDiffuseParticleBufferIndex = live ? S.p[b].diffuseIndex : (S.nbD - 1u);
        pbufs[b].mUniqueId = 900u + b;
    }
    R.bPBufs[side] = R.mk(dev, pbufs);

    std::memset(&desc, 0, sizeof(PxgParticleSystem));
    desc.mCommonData.mNumParticleBuffers = S.nbP;
    desc.mNumDiffuseBuffers = S.nbD;
    desc.mData.mRestOffset = S.restOffset;
    desc.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bPos[side].addr));
    desc.mSortedVelocities = reinterpret_cast<float4*>(size_t(R.bVel[side].addr));
    desc.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(R.bPhase[side].addr));
    desc.mUnsortedToSortedMapping = reinterpret_cast<PxU32*>(size_t(R.bU2s[side].addr));
    desc.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(R.bRunsum[side].addr));
    desc.mDiffusePotentials = reinterpret_cast<float2*>(size_t(R.bPot[side].addr));
    desc.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(R.bPBufs[side].addr));
    desc.mDiffuseSimBuffers =
        reinterpret_cast<PxgParticleDiffuseSimBuffer*>(size_t(R.bDBufs[side].addr));
}

// ---------------------------------------------------------------------------
// The host mirror: the CUDA's arithmetic, in the CUDA's order.
// ---------------------------------------------------------------------------
struct Predicted
{
    // per diffuse buffer
    std::vector<uint32_t> qualifiers;   // the atomic's final value
    std::vector<std::vector<Rec>> recs; // the records those qualifiers would write
};

static Predicted predict(const SysSpec& S, const SideArrays& A)
{
    Predicted P;
    P.qualifiers.assign(S.nbD, 0u);
    P.recs.assign(S.nbD, {});

    for (uint32_t b = 0; b < S.nbP; ++b)
    {
        const uint32_t dbi = S.p[b].diffuseIndex;
        if (dbi == 0xffffffffu) continue;
        const uint32_t cap = S.d[dbi].capacity;
        if (cap == 0u) continue;                        // C12

        for (uint32_t pi = 0; pi < S.p[b].numActive; ++pi)
        {
            const uint32_t sortedInd = A.u2s[pi + A.runsum[b]];
            if ((A.sortedPhase[sortedInd] & PHASE_FLUID_BIT) == 0u) continue;   // C11

            const float ptntsX = A.potentials[size_t(sortedInd) * 2 + 0];
            const float ptntsY = A.potentials[size_t(sortedInd) * 2 + 1];
            const F4 v = A.sortedVel[sortedInd];

            const float threshold = S.d[dbi].threshold;
            const float kineticEnergy = (v.x * v.x + v.y * v.y + v.z * v.z) * S.d[dbi].kew;
            const float divergence = S.d[dbi].dw * ptntsX;
            const float pressure = S.d[dbi].pw * ptntsY;
            const float intensity = pressure - divergence + kineticEnergy;   // C4

            const float r0 = gRandom[(sortedInd + 0u) % RANDOM_TABLE_SIZE];  // C3
            if (!(r0 * intensity > threshold)) continue;                     // C5

            ++P.qualifiers[dbi];                                             // C6

            const F4 x = A.sortedPos[sortedInd];
            const float r1 = gRandom[(sortedInd + 1u) % RANDOM_TABLE_SIZE];
            const float r2 = gRandom[(sortedInd + 2u) % RANDOM_TABLE_SIZE];
            const float r3 = gRandom[(sortedInd + 3u) % RANDOM_TABLE_SIZE];

            const float lifeMin = 1.0f;                                      // C8
            const float lifeMax = S.d[dbi].lifetime;
            const float ratio = intensity / threshold;
            const float lifeScale = ((ratio < 1.0f) ? ratio : 1.0f) * r1;
            const float lifetime = lifeMin + lifeScale * (lifeMax - lifeMin);

            // C9: exactly the shader's association order.
            const float bx = x.x - r2 * v.x * DT + r1 * S.restOffset * 0.25f;
            const float by = x.y - r2 * v.y * DT + r2 * S.restOffset * 0.25f;
            const float bz = x.z - r2 * v.z * DT + r3 * S.restOffset * 0.25f;

            Rec rec;
            rec.px = f2u(bx); rec.py = f2u(by); rec.pz = f2u(bz); rec.pw = f2u(lifetime);
            rec.vx = f2u(v.x); rec.vy = f2u(v.y); rec.vz = f2u(v.z); rec.vw = f2u(0.0f);
            P.recs[dbi].push_back(rec);
        }
    }
    return P;
}

// ---------------------------------------------------------------------------
static void checkSide(PxgSlangComputeDevice* dev, const char* label, Rig& R, uint32_t side,
                      const SysSpec& S, const SideArrays& A, size_t& nOver, size_t& nZero)
{
    const Predicted P = predict(S, A);
    const auto two = downloadBuffer<int32_t>(dev, R.bTwo[side].h, size_t(S.nbD) * 2u);
    const auto pinned = downloadBuffer<int32_t>(dev, R.bPinned[side].h, S.nbD);

    for (uint32_t d = 0; d < S.nbD; ++d)
    {
        const uint32_t cap = S.d[d].capacity;
        const uint32_t alloc = cap + DIFFUSE_ALLOC_MARGIN;
        const auto pos = downloadBuffer<F4>(dev, R.bDPos[side][d].h, alloc);
        const auto vel = downloadBuffer<F4>(dev, R.bDVel[side][d].h, alloc);

        // C6: the staging counter is the qualifier count, capacity or not.
        PXG_CHECK_MSG(uint32_t(two[size_t(d) * 2 + 1]) == P.qualifiers[d],
                      "%s/dbuf%u: staging counter %d, expected %u qualifiers",
                      label, d, two[size_t(d) * 2 + 1], P.qualifiers[d]);
        // Nothing in this kernel publishes or touches the pinned mirror.
        PXG_CHECK_MSG(two[size_t(d) * 2 + 0] == int32_t(111 + d),
                      "%s/dbuf%u: the PUBLISHED slot was written (%d)", label, d,
                      two[size_t(d) * 2 + 0]);
        PXG_CHECK_MSG(uint32_t(pinned[d]) == POISON_U32,
                      "%s/dbuf%u: the pinned mirror was written (%d)", label, d, pinned[d]);

        // The records actually written, in slot order, and the poison tail.
        std::vector<Rec> got;
        size_t badTail = 0;
        for (uint32_t i = 0; i < alloc; ++i)
        {
            const bool written = f2u(pos[i].x) != f2u(POISON_F);
            if (written)
            {
                Rec r;
                r.px = f2u(pos[i].x); r.py = f2u(pos[i].y);
                r.pz = f2u(pos[i].z); r.pw = f2u(pos[i].w);
                r.vx = f2u(vel[i].x); r.vy = f2u(vel[i].y);
                r.vz = f2u(vel[i].z); r.vw = f2u(vel[i].w);
                got.push_back(r);
            }
            else if (f2u(vel[i].x) != f2u(POISON_F))
            {
                ++badTail;   // a velocity written without a position
            }
        }
        PXG_CHECK_MSG(badTail == 0, "%s/dbuf%u: %zu slots hold a velocity but no position",
                      label, d, badTail);

        // Not one write at or past the capacity, however many qualifiers there
        // were.  This is what the margin above exists for.
        size_t pastCap = 0;
        for (uint32_t i = cap; i < alloc; ++i)
            if (f2u(pos[i].x) != f2u(POISON_F) || f2u(vel[i].x) != f2u(POISON_F)) ++pastCap;
        PXG_CHECK_MSG(pastCap == 0, "%s/dbuf%u: %zu slots past the capacity (%u) were written "
                      "-- the allocation runs to %u", label, d, pastCap, cap, alloc);

        const size_t expectWritten = std::min<size_t>(P.qualifiers[d], cap);
        PXG_CHECK_MSG(got.size() == expectWritten,
                      "%s/dbuf%u: %zu records written, expected %zu (qualifiers %u, capacity %u)",
                      label, d, got.size(), expectWritten, P.qualifiers[d], cap);
        if (cap > 0u && P.qualifiers[d] > cap) ++nOver;
        if (cap == 0u) ++nZero;

        // Every written record must be a predicted one; when the capacity was not
        // exceeded, the two multisets must be equal.  Never index-by-index: the
        // slot a qualifier gets is a race.
        std::vector<Rec> want = P.recs[d];
        std::sort(want.begin(), want.end());
        std::vector<Rec> mine = got;
        std::sort(mine.begin(), mine.end());
        size_t badRec = 0;
        for (const Rec& r : mine)
            if (!std::binary_search(want.begin(), want.end(), r)) ++badRec;
        PXG_CHECK_MSG(badRec == 0, "%s/dbuf%u: %zu written records are not in the predicted set",
                      label, d, badRec);
        if (P.qualifiers[d] <= cap)
            PXG_CHECK_MSG(mine == want,
                          "%s/dbuf%u: the written set differs from the predicted set "
                          "(%zu vs %zu records)", label, d, mine.size(), want.size());
    }
    std::printf("    [ note ] %s: qualifiers", label);
    for (uint32_t d = 0; d < S.nbD; ++d)
        std::printf(" dbuf%u=%u(cap %u)", d, P.qualifiers[d], S.d[d].capacity);
    std::printf("\n");
}

static void runAll(PxgSlangComputeDevice* dev)
{
    gRandom.resize(RANDOM_TABLE_SIZE);
    for (uint32_t i = 0; i < RANDOM_TABLE_SIZE; ++i)
        gRandom[i] = 0.05f + 0.9f * float((i * 37u + 11u) % RANDOM_TABLE_SIZE)
                                  / float(RANDOM_TABLE_SIZE);

    SideArrays AA, AB;
    buildArrays(SPEC_A, AA);
    buildArrays(SPEC_B, AB);

    // AN EXACT TIE.  `r0 * intensity > threshold` and `>= threshold` differ on
    // exactly one input: a particle whose product equals the threshold bit for
    // bit.  Float data never lands there by accident -- which is why `c5-test-ge`
    // survived the first sweep -- so one is CONSTRUCTED: take a fluid particle
    // that feeds system A's diffuse buffer 1, compute its product under that
    // buffer's own weights, and install the result as that buffer's threshold.
    // Every other particle's verdict shifts with it, and the prediction follows,
    // because the mirror reads the same field.
    {
        const uint32_t b = 0u;                       // A_PBUFS[0] -> diffuse buffer 1
        const uint32_t d = A_PBUFS[b].diffuseIndex;
        bool done = false;
        for (uint32_t pi = 0; pi < A_PBUFS[b].numActive && !done; ++pi)
        {
            const uint32_t si = AA.u2s[pi + AA.runsum[b]];
            if ((AA.sortedPhase[si] & PHASE_FLUID_BIT) == 0u) continue;
            const F4 v = AA.sortedVel[si];
            const float ke = (v.x * v.x + v.y * v.y + v.z * v.z) * A_DBUFS[d].kew;
            const float dv = A_DBUFS[d].dw * AA.potentials[size_t(si) * 2 + 0];
            const float pr = A_DBUFS[d].pw * AA.potentials[size_t(si) * 2 + 1];
            const float intensity = pr - dv + ke;
            const float r0 = gRandom[(si + 0u) % RANDOM_TABLE_SIZE];
            const float product = r0 * intensity;
            if (product > 0.0f) { A_DBUFS[d].threshold = product; done = true; }
        }
        PXG_CHECK_MSG(done, "no fluid particle with a positive product to build the tie from");
        // Count how many particles now sit exactly on the threshold.
        for (uint32_t pi = 0; pi < A_PBUFS[b].numActive; ++pi)
        {
            const uint32_t si = AA.u2s[pi + AA.runsum[b]];
            if ((AA.sortedPhase[si] & PHASE_FLUID_BIT) == 0u) continue;
            const F4 v = AA.sortedVel[si];
            const float ke = (v.x * v.x + v.y * v.y + v.z * v.z) * A_DBUFS[d].kew;
            const float dv = A_DBUFS[d].dw * AA.potentials[size_t(si) * 2 + 0];
            const float pr = A_DBUFS[d].pw * AA.potentials[size_t(si) * 2 + 1];
            const float r0 = gRandom[(si + 0u) % RANDOM_TABLE_SIZE];
            if (f2u(r0 * (pr - dv + ke)) == f2u(A_DBUFS[d].threshold)) ++gExactTieCount;
        }
    }

    Rig R;
    R.bRandom = R.mk(dev, gRandom);

    std::vector<PxgParticleSystem> descArr(NB_SYS_SLOTS);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));
    buildSide(dev, R, 0u, SPEC_A, AA, descArr[SYS_A]);
    buildSide(dev, R, 1u, SPEC_B, AB, descArr[SYS_B]);
    R.bDesc = R.mk(dev, descArr);
    // { 2, 0 }: index 1 is a DECOY descriptor, all zeroes -- a kernel that takes
    // the grid axis for the descriptor index walks into it.
    R.bActive = R.mk(dev, std::vector<uint32_t>{ SYS_A, SYS_B });

    R.blob.particleSystems = R.bDesc.addr;
    R.blob.activeParticleSystems = R.bActive.addr;
    R.blob.randomTable = R.bRandom.addr;
    R.blob.randomTableSize = RANDOM_TABLE_SIZE;
    R.blob.dt = DT;

    {
        Kernel k("psDiffuseCreate.spv", "ps_diffuseParticleCreate", {}, sizeof(PsDiffCreateBlob));
        PXG_CHECK_MSG(k.isValid(), "the create pipeline must build");
        if (!k.isValid()) { R.destroy(dev); return; }
        // Grid (ceil(maxParticlesPerBuffer/256), GRID_Y, nbActiveSystems).
        uint32_t maxPer = 0;
        for (uint32_t b = 0; b < SPEC_A.nbP; ++b) maxPer = std::max(maxPer, A_PBUFS[b].numActive);
        for (uint32_t b = 0; b < SPEC_B.nbP; ++b) maxPer = std::max(maxPer, B_PBUFS[b].numActive);
        k.dispatch((maxPer + 255u) / 256u, {}, R.blob, GRID_Y, 2u);
    }
    dev->flush();

    size_t nOver = 0, nZero = 0;
    checkSide(dev, "sysA", R, 0u, SPEC_A, AA, nOver, nZero);
    checkSide(dev, "sysB", R, 1u, SPEC_B, AB, nOver, nZero);

    // The DECOY descriptor must be untouched -- it is all zeroes, so a kernel
    // that visited it would have faulted; this records the intent.
    const auto descBack = downloadBuffer<PxgParticleSystem>(dev, R.bDesc.h, NB_SYS_SLOTS);
    PXG_CHECK_MSG(descBack[1].mCommonData.mNumParticleBuffers == 0u,
                  "the decoy descriptor was written");

    // Coverage, demanded rather than designed.
    PXG_CHECK_MSG(nOver > 0, "no diffuse buffer over-subscribed its capacity -- the C6/C7 "
                  "overshoot is untested");
    PXG_CHECK_MSG(nZero > 0, "no diffuse buffer had capacity 0 -- C12 is untested");
    PXG_CHECK_MSG(gExactTieCount > 0, "no particle sits exactly on its threshold -- `>` and "
                  "`>=` are indistinguishable");
    {
        uint32_t maxPer = 0;
        for (uint32_t b = 0; b < SPEC_A.nbP; ++b) maxPer = std::max(maxPer, A_PBUFS[b].numActive);
        PXG_CHECK_MSG(maxPer > 256u, "no particle buffer crosses the block boundary -- "
                      "gid.x * 256 is dead code");
        bool hasNoDiffuse = false;
        for (uint32_t b = 0; b < SPEC_A.nbP; ++b)
            if (A_PBUFS[b].diffuseIndex == 0xffffffffu) hasNoDiffuse = true;
        PXG_CHECK_MSG(hasNoDiffuse, "no particle buffer without a diffuse side");
        PXG_CHECK_MSG(AA.total % RANDOM_TABLE_SIZE != 0u,
                      "the random table divides the particle count -- the wrap is untested");
    }
    R.destroy(dev);
}

} // namespace

PXG_TEST(psDiffuseCreate)
{
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}
