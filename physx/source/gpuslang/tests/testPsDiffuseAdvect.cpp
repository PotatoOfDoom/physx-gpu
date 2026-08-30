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

// testPsDiffuseAdvect.cpp -- P9.13e, the fixture for the two kernels that advect
// and age diffuse particles.
//
// ---------------------------------------------------------------------------
// THE ARRANGEMENT
// ---------------------------------------------------------------------------
// The two run in the engine's order over ONE arena: UpdatePBF writes the
// velocity average into the OVERLOADED origin array, and Compact reads exactly
// that.  So UpdatePBF's output is genuinely Compact's input, and a mistake in
// the first shows up in the second rather than being checked twice in isolation.
//
// UpdatePBF is fully deterministic: one thread per diffuse particle, no atomics,
// bit-comparable against a host mirror of the CUDA's arithmetic in the CUDA's
// order.
//
// Compact is warp-cooperative.  Within one warp the survivors' order is FIXED
// (an exclusive scan over the ballot), but the per-warp base offset comes from
// an atomic and is a race.  So the check is: the survivor COUNT is exact, every
// written record is a predicted one, and -- because the capacity is never
// exceeded here -- the two multisets are equal.  The P9.11 rule again: never the
// index, only the set.
//
// ---------------------------------------------------------------------------
// WHAT THE GEOMETRY IS BUILT TO DISCRIMINATE
// ---------------------------------------------------------------------------
//   * a FLUID neighbourhood that actually varies: a grid whose cells hold
//     0, 1, a few and MORE THAN 16 fluid particles, so the neighbour cap (V7)
//     and the empty-cell path are both taken;
//   * both advection modes -- one system with eFULL_DIFFUSE_ADVECTION set (the
//     3x3x3 gather) and one without (own cell only), which is the only way V5
//     and the offset ORDER of V6 are observable;
//   * neighbour counts spread across all three classification bands (< 4,
//     < 8, >= 8), demanded rather than hoped for;
//   * lifetimes that straddle dt: some survive, some land exactly on zero (K10
//     kills those), some are already below it;
//   * a NON-IDENTITY diffuse unsorted->sorted mapping and a NON-ZERO
//     mStartIndex, so K4's three index spaces are three different numbers;
//   * a speed above mMaxVelocity so K8's clamp bites, and a zero-velocity
//     particle so its guard is taken;
//   * two systems through a non-identity active list with a decoy between them.

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
struct V3 { float x, y, z; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// psDiffuseAdvect.slang's PsDiffAdvectPush.
struct PsDiffAdvectBlob
{
    uint64_t particleSystems;
    uint64_t activeParticleSystems;
    float gx, gy, gz;
    float dt;
};
static_assert(sizeof(PsDiffAdvectBlob) == 32, "engine ABI blob size");

static_assert(offsetof(PxgParticleSystem, mCellStart) == 168, "@168");
static_assert(offsetof(PxgParticleSystem, mCellEnd) == 176, "@176");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) == 552, "@552");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) == 560, "@560");
static_assert(offsetof(PxgParticleSystem, mDiffuseOriginPos_LifeTime) == 544, "@544");
static_assert(offsetof(PxgParticleSystem, mDiffuseUnsortedToSortedMapping) == 584, "@584");

// ---------------------------------------------------------------------------
static const uint32_t NB_SYS_SLOTS = 4u;
static const uint32_t SYS_A = 2u;     // active list { 2, 0 }: a decoy at index 1
static const uint32_t SYS_B = 0u;
static const uint32_t GRID_Y_COMPACT = 4u;   // > either system's diffuse-buffer count
static const float DT = 1.0f / 53.0f;
static const V3 GRAVITY{ 0.31f, -9.71f, 0.17f };   // not axis-aligned: a dropped
                                                   // component is visible
static const float POISON_F = -9.0f;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;
// NOT `EMPTY_CELL`: PxgParticleSystem.h:52 defines that as a MACRO, and a
// same-named constant here does not shadow it, it breaks the header.
static const uint32_t CELL_EMPTY = 0xffffffffu;
static const uint32_t FLAG_FULL_ADVECTION = 4u;   // 1<<2
static const uint32_t GRID_SX = 8u, GRID_SY = 8u, GRID_SZ = 8u;   // powers of two
static const float CELL_WIDTH = 0.25f;
// EXACTLY REPRESENTABLE, and smaller than the cell width, so the tie pair below
// can sit inside ONE cell and be seen in both advection modes.
static const float CONTACT_DIST = 0.125f;
// The tie: a diffuse particle and a fluid particle exactly CONTACT_DIST apart
// along x, at binary-exact coordinates, so dSq == contactDistanceSq bit for bit
// and `<` and `<=` disagree on it.  Float data never lands there by accident --
// which is why `v-dist-test-le` survived the first sweep.
static const float TIE_X = 2.0f, TIE_Y = 2.0f, TIE_Z = 2.0f;

struct SysSpec
{
    uint32_t nbDiffuse;      // the SYSTEM's diffuse count (what Sum published)
    uint32_t nbFluid;
    bool fullAdvection;
    float maxVelocity;
    uint32_t startIndex;     // the single diffuse buffer's mStartIndex
    uint32_t capacity;
    float airDrag, buoyancy, bubbleDrag;
    uint32_t tag;
};

static const SysSpec SPEC_A{ 400u, 900u, true,  6.0f,  0u,  1024u, 0.35f, 0.80f, 0.55f, 0u };
static const SysSpec SPEC_B{ 260u, 640u, false, 2.5f, 37u,  1024u, 0.10f, 1.40f, 0.20f, 1u };

struct SideArrays
{
    std::vector<F4> fluidPos, fluidVel;
    std::vector<uint32_t> cellStart, cellEnd;
    std::vector<F4> dSortedPos;     // diffuse, SORTED: position + lifetime
    std::vector<F4> dOriginPos;     // diffuse, UNSORTED/flat: the previous position
    std::vector<uint32_t> d_u2s;    // diffuse unsorted -> sorted
    uint32_t numCells = GRID_SX * GRID_SY * GRID_SZ;
};

// The fluid particles are laid out CELL BY CELL so the per-cell population is
// exactly what the fixture intends: 0, 1, 3, 9 or 20 (> the 16 cap).
static void buildArrays(const SysSpec& S, SideArrays& A)
{
    A.cellStart.assign(A.numCells, CELL_EMPTY);
    A.cellEnd.assign(A.numCells, 0u);
    A.fluidPos.clear();
    A.fluidVel.clear();

    const uint32_t pattern[5] = { 0u, 1u, 3u, 9u, 20u };
    uint32_t next = 0;
    for (uint32_t c = 0; c < A.numCells && next < S.nbFluid; ++c)
    {
        const uint32_t want = pattern[(c + S.tag) % 5u];
        const uint32_t n = (next + want <= S.nbFluid) ? want : (S.nbFluid - next);
        // Cell 0 is never skipped, whatever the pattern says: it carries the
        // exact-tie neighbour below, and both systems need it.  (sysA's pattern
        // gives cell 0 a population of zero, which is how the tie went missing
        // on the first attempt.)
        if (n == 0u && c != 0u) continue;            // stays CELL_EMPTY
        A.cellStart[c] = next;
        // Cell 0 also carries the exact-tie neighbour.  (TIE_X, TIE_Y, TIE_Z)
        // hashes to (0,0,0) because 8 & 7 == 0, so this particle belongs to
        // cell 0 even though it sits nowhere near its geometric centre -- the
        // kernel reads cellStart/cellEnd and tests distance, nothing else.
        if (c == 0u)
        {
            A.fluidPos.push_back(F4{ TIE_X + CONTACT_DIST, TIE_Y, TIE_Z, 1.0f });
            A.fluidVel.push_back(F4{ 1.25f, -0.5f, 0.75f, 0.0f });
            ++next;
        }
        // The cell's own centre, so a particle placed in cell c really hashes to c.
        const uint32_t cx = c % GRID_SX, cy = (c / GRID_SX) % GRID_SY, cz = c / (GRID_SX * GRID_SY);
        for (uint32_t k = 0; k < n; ++k)
        {
            const float t = float(next) * 0.017f + float(S.tag) * 2.3f;
            // Spread along +x from the cell centre out to just inside the +x
            // face.  THIS IS WHAT MAKES THE ADVECTION MODE OBSERVABLE: a
            // diffuse particle placed just across that face (below) is within
            // the contact distance of these but in the NEXT cell, so only the
            // 3x3x3 gather finds them.  With everything at cell centres the two
            // modes give identical answers and the V5 mutants are equivalent --
            // which is exactly what happened on the first attempt.
            const float along = (n > 1u) ? (float(k) / float(n - 1u)) : 0.0f;
            A.fluidPos.push_back(F4{ (float(cx) + 0.5f) * CELL_WIDTH + along * 0.11f,
                                     (float(cy) + 0.5f) * CELL_WIDTH + 0.02f * std::cos(t),
                                     (float(cz) + 0.5f) * CELL_WIDTH + 0.02f * std::sin(t * 1.7f),
                                     1.0f });
            A.fluidVel.push_back(F4{ std::sin(t * 1.1f) * 2.0f, std::cos(t * 0.7f) * 2.0f,
                                     std::sin(t * 0.4f) * 2.0f, 0.0f });
            ++next;
        }
        A.cellEnd[c] = next;
    }

    A.cellEnd[0] = A.cellEnd[0];   // (the tie was counted into `next` above)

    // The diffuse particles: sorted array, and a non-identity unsorted->sorted map.
    const uint32_t n = S.nbDiffuse;
    A.dSortedPos.resize(n);
    A.dOriginPos.resize(n + S.startIndex);
    A.d_u2s.resize(n + S.startIndex);
    for (uint32_t i = 0; i < n + S.startIndex; ++i)
        A.d_u2s[i] = (7u * i + 3u) % n;          // into the SORTED range

    for (uint32_t s = 0; s < n; ++s)
    {
        const float t = float(s) * 0.011f + float(S.tag) * 5.1f;
        // Spread across the grid so the neighbour counts vary, including cells
        // with 20 fluid particles (the cap) and empty ones.
        const uint32_t c = (s * 13u + S.tag * 7u) % A.numCells;
        const uint32_t cx = c % GRID_SX, cy = (c / GRID_SX) % GRID_SY, cz = c / (GRID_SX * GRID_SY);
        // Lifetimes that straddle dt: some > dt, some EXACTLY dt (die at zero),
        // some already below.
        float life;
        switch (s % 4u)
        {
            case 0:  life = DT * 5.0f; break;
            case 1:  life = DT;        break;    // -> exactly 0 -> dies (K10)
            case 2:  life = DT * 0.5f; break;    // -> clamped to 0 -> dies
            default: life = DT * 2.0f; break;
        }
        // Every third one sits just inside its cell's -x face, so its nearest
        // fluid neighbours are the ones spread toward the +x face of the
        // PREVIOUS cell -- a cross-cell neighbourhood that only the 3x3x3
        // gather can see.
        const float offX = (s % 3u == 0u) ? -0.11f : (0.03f * std::cos(t));
        A.dSortedPos[s] = F4{ (float(cx) + 0.5f) * CELL_WIDTH + offX,
                              (float(cy) + 0.5f) * CELL_WIDTH + 0.03f * std::sin(t),
                              (float(cz) + 0.5f) * CELL_WIDTH + 0.03f * std::cos(t * 1.3f),
                              life };
    }
    // Sorted slot 0 is the tie's partner, placed at binary-exact coordinates.
    A.dSortedPos[0] = F4{ TIE_X, TIE_Y, TIE_Z, DT * 5.0f };

    // The origin positions are indexed by the UNSORTED index, and one particle
    // gets an origin EQUAL to its position so its reconstructed velocity is zero
    // (K8's guard), while another is far away so its speed exceeds maxVelocity.
    for (uint32_t i = 0; i < n + S.startIndex; ++i)
    {
        const uint32_t s = A.d_u2s[i];
        const F4 p = A.dSortedPos[s];
        if (i % 37u == 0u)
            A.dOriginPos[i] = p;                                   // zero velocity
        else if (i % 41u == 0u)
            A.dOriginPos[i] = F4{ p.x - 3.0f, p.y + 2.5f, p.z - 1.5f, p.w };  // very fast
        else
        {
            const float t = float(i) * 0.019f;
            A.dOriginPos[i] = F4{ p.x - 0.01f * std::sin(t), p.y - 0.01f * std::cos(t),
                                  p.z - 0.01f * std::sin(t * 0.6f), p.w };
        }
    }
}

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bDesc, bActive;
    DevBuf bFluidPos[2], bFluidVel[2], bCellStart[2], bCellEnd[2];
    DevBuf bDSorted[2], bDOrigin[2], bDU2s[2], bVelAvg[2], bNumDiffuse[2];
    DevBuf bDBufs[2], bTwo[2], bOutPos[2], bOutVel[2];
    PsDiffAdvectBlob blob{};

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

static void buildSide(PxgSlangComputeDevice* dev, Rig& R, uint32_t side, const SysSpec& S,
                      const SideArrays& A, PxgParticleSystem& desc)
{
    R.bFluidPos[side] = R.mk(dev, A.fluidPos);
    R.bFluidVel[side] = R.mk(dev, A.fluidVel);
    R.bCellStart[side] = R.mk(dev, A.cellStart);
    R.bCellEnd[side] = R.mk(dev, A.cellEnd);
    R.bDSorted[side] = R.mk(dev, A.dSortedPos);
    R.bDOrigin[side] = R.mk(dev, A.dOriginPos);
    R.bDU2s[side] = R.mk(dev, A.d_u2s);
    // The OVERLOADED array: poisoned, so an unwritten slot is visible.
    R.bVelAvg[side] = R.mk(dev, std::vector<F4>(S.nbDiffuse,
                                F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    R.bNumDiffuse[side] = R.mk(dev, std::vector<uint32_t>(1u, S.nbDiffuse));

    R.bOutPos[side] = R.mk(dev, std::vector<F4>(S.capacity,
                              F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    R.bOutVel[side] = R.mk(dev, std::vector<F4>(S.capacity,
                              F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    // [0] published (what Compact reads), [1] staging (what it appends into).
    R.bTwo[side] = R.mk(dev, std::vector<int32_t>{ int32_t(S.nbDiffuse), 0 });

    std::vector<PxgParticleDiffuseSimBuffer> dbufs(1);
    std::memset(dbufs.data(), 0, sizeof(PxgParticleDiffuseSimBuffer));
    PxDiffuseParticleParams pr;
    pr.airDrag = S.airDrag;
    pr.buoyancy = S.buoyancy;
    pr.bubbleDrag = S.bubbleDrag;
    dbufs[0].mParams = pr;
    dbufs[0].mDiffusePositions_LifeTime = reinterpret_cast<float4*>(size_t(R.bOutPos[side].addr));
    dbufs[0].mDiffuseVelocities = reinterpret_cast<float4*>(size_t(R.bOutVel[side].addr));
    dbufs[0].mMaxNumParticles = S.capacity;
    dbufs[0].mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bTwo[side].addr));
    dbufs[0].mStartIndex = S.startIndex;
    R.bDBufs[side] = R.mk(dev, dbufs);

    std::memset(&desc, 0, sizeof(PxgParticleSystem));
    desc.mNumDiffuseBuffers = 1u;
    desc.mCommonData.mGridCellWidth = CELL_WIDTH;
    desc.mCommonData.mGridSizeX = GRID_SX;
    desc.mCommonData.mGridSizeY = GRID_SY;
    desc.mCommonData.mGridSizeZ = GRID_SZ;
    desc.mCommonData.mParticleContactDistanceSq = CONTACT_DIST * CONTACT_DIST;
    desc.mCommonData.mParticleContactDistanceInv = 1.0f / CONTACT_DIST;
    desc.mData.mFlags = S.fullAdvection ? FLAG_FULL_ADVECTION : 0u;
    desc.mData.mMaxVelocity = S.maxVelocity;
    desc.mCellStart = reinterpret_cast<PxU32*>(size_t(R.bCellStart[side].addr));
    desc.mCellEnd = reinterpret_cast<PxU32*>(size_t(R.bCellEnd[side].addr));
    desc.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bFluidPos[side].addr));
    desc.mSortedVelocities = reinterpret_cast<float4*>(size_t(R.bFluidVel[side].addr));
    desc.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bNumDiffuse[side].addr));
    desc.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDSorted[side].addr));
    desc.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bVelAvg[side].addr));
    desc.mDiffuseOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDOrigin[side].addr));
    desc.mDiffuseUnsortedToSortedMapping = reinterpret_cast<PxU32*>(size_t(R.bDU2s[side].addr));
    desc.mDiffuseSimBuffers =
        reinterpret_cast<PxgParticleDiffuseSimBuffer*>(size_t(R.bDBufs[side].addr));
}

// ---------------------------------------------------------------------------
// The host mirror.
// ---------------------------------------------------------------------------
struct Rec
{
    uint32_t px, py, pz, pw, vx, vy, vz, vw;
    bool operator<(const Rec& o) const
    { return std::memcmp(this, &o, sizeof(Rec)) < 0; }
    bool operator==(const Rec& o) const
    { return std::memcmp(this, &o, sizeof(Rec)) == 0; }
};

struct Expect
{
    std::vector<F4> velAvg;      // UpdatePBF's output, per SORTED diffuse index
    std::vector<Rec> survivors;  // Compact's output, as a multiset
    uint32_t bandLow = 0, bandMid = 0, bandHigh = 0;   // coverage of K6
    uint32_t nCapped = 0, nZeroVel = 0, nClamped = 0, nExactTie = 0;
};


static Expect predict(const SysSpec& S, const SideArrays& A)
{
    Expect E;
    E.velAvg.assign(S.nbDiffuse, F4{ 0, 0, 0, 0 });

    const uint32_t end = S.fullAdvection ? 3u : 1u;
    const int offs[3] = { 0, -1, 1 };                                   // V6
    const float contactDistanceSq = CONTACT_DIST * CONTACT_DIST;
    const float invContactDistance = 1.0f / CONTACT_DIST;

    // ---- UpdatePBF ----
    for (uint32_t pi = 0; pi < S.nbDiffuse; ++pi)
    {
        const F4 xi4 = A.dSortedPos[pi];
        const float px = xi4.x, py = xi4.y, pz = xi4.z;
        const int gx = int(std::floor(xi4.x / CELL_WIDTH));
        const int gy = int(std::floor(xi4.y / CELL_WIDTH));
        const int gz = int(std::floor(xi4.z / CELL_WIDTH));

        float weightSum = 0.0f;
        float vsx = 0.0f, vsy = 0.0f, vsz = 0.0f;
        uint32_t numNeighbors = 0;
        bool full = false;
        for (uint32_t z = 0; z < end && !full; ++z)
            for (uint32_t y = 0; y < end && !full; ++y)
                for (uint32_t x = 0; x < end && !full; ++x)
                {
                    const uint32_t hx = uint32_t(gx + offs[x]) & (GRID_SX - 1u);
                    const uint32_t hy = uint32_t(gy + offs[y]) & (GRID_SY - 1u);
                    const uint32_t hz = uint32_t(gz + offs[z]) & (GRID_SZ - 1u);
                    const uint32_t hash = ((hz * GRID_SY) * GRID_SX) + (hy * GRID_SX) + hx;
                    const uint32_t si = A.cellStart[hash];
                    if (si == CELL_EMPTY) continue;
                    const uint32_t ei = A.cellEnd[hash];
                    for (uint32_t q = si; q < ei; ++q)
                    {
                        const F4 xj = A.fluidPos[q];
                        const float dx = px - xj.x, dy = py - xj.y, dz = pz - xj.z;
                        const float dSq = dx * dx + dy * dy + dz * dz;
                        if (f2u(dSq) == f2u(contactDistanceSq)) ++E.nExactTie;
                        if (dSq < contactDistanceSq)
                        {
                            const F4 vj = A.fluidVel[q];
                            const float w = 1.0f - std::sqrt(dSq) * invContactDistance;  // V8
                            weightSum += w;
                            vsx += vj.x * w; vsy += vj.y * w; vsz += vj.z * w;
                            ++numNeighbors;
                            if (numNeighbors == 16u) { full = true; ++E.nCapped; break; }
                        }
                    }
                }
        F4 avg{ 0, 0, 0, float(numNeighbors) };
        // P9.36: diffuseParticles.cu:269 is `velAvg = velocitySum / weightSum` on a
        // PxVec3 -- PxVec3::operator/ (PxVec3.h:219), one reciprocal then three
        // multiplies, not three divides.
        if (weightSum > 0.0f)
        { const float r = 1.0f / weightSum; avg.x = vsx * r; avg.y = vsy * r; avg.z = vsz * r; }
        E.velAvg[pi] = avg;
    }

    // ---- Compact ----
    for (uint32_t pi = 0; pi < S.nbDiffuse; ++pi)
    {
        const uint32_t index = pi + S.startIndex;                        // K3
        const uint32_t sortedInd = A.d_u2s[index];
        const F4 xi4 = A.dSortedPos[sortedInd];
        const F4 vi4Old = A.dOriginPos[index];
        const F4 xiva4 = E.velAvg[sortedInd];                            // K4

        const float velx = (xi4.x - vi4Old.x) * (1.0f / DT);             // K5
        const float vely = (xi4.y - vi4Old.y) * (1.0f / DT);
        const float velz = (xi4.z - vi4Old.z) * (1.0f / DT);

        float nvx, nvy, nvz;
        if (xiva4.w < 4.0f)                                              // K6
        {
            ++E.bandLow;
            const float k = 1.0f - S.airDrag * DT;
            nvx = velx * k; nvy = vely * k; nvz = velz * k;
        }
        else if (xiva4.w < 8.0f)
        {
            ++E.bandMid;
            nvx = xiva4.x; nvy = xiva4.y; nvz = xiva4.z;
        }
        else
        {
            ++E.bandHigh;                                                // K7
            const float b = 1.0f + S.buoyancy;
            nvx = velx - b * GRAVITY.x * DT + S.bubbleDrag * (xiva4.x - velx);
            nvy = vely - b * GRAVITY.y * DT + S.bubbleDrag * (xiva4.y - vely);
            nvz = velz - b * GRAVITY.z * DT + S.bubbleDrag * (xiva4.z - velz);
        }

        const float magSq = nvx * nvx + nvy * nvy + nvz * nvz;
        if (magSq > 0.0f)                                                // K8
        {
            const float mag = std::sqrt(magSq);
            const float scale = (mag < S.maxVelocity) ? mag : S.maxVelocity;
            if (mag > S.maxVelocity) ++E.nClamped;
            const float inv = 1.0f / std::sqrt(magSq);
            nvx = scale * (nvx * inv); nvy = scale * (nvy * inv); nvz = scale * (nvz * inv);
        }
        else ++E.nZeroVel;

        const float qx = xi4.x + (nvx - velx) * DT;                      // K9
        const float qy = xi4.y + (nvy - vely) * DT;
        const float qz = xi4.z + (nvz - velz) * DT;
        const float lifeTime = std::max(xi4.w - DT, 0.0f);               // K10

        if (lifeTime > 0.0f)
        {
            Rec r;
            r.px = f2u(qx); r.py = f2u(qy); r.pz = f2u(qz); r.pw = f2u(lifeTime);
            r.vx = f2u(nvx); r.vy = f2u(nvy); r.vz = f2u(nvz); r.vw = f2u(0.0f);
            E.survivors.push_back(r);
        }
    }
    return E;
}

// ---------------------------------------------------------------------------
static void checkSide(PxgSlangComputeDevice* dev, const char* label, Rig& R, uint32_t side,
                      const SysSpec& S, const SideArrays& A, Expect& E)
{
    // ---- UpdatePBF: fully deterministic, compared slot by slot ----
    const auto velAvg = downloadBuffer<F4>(dev, R.bVelAvg[side].h, S.nbDiffuse);
    size_t badAvg = 0;
    for (uint32_t i = 0; i < S.nbDiffuse; ++i)
    {
        const F4 g = velAvg[i], w = E.velAvg[i];
        if (f2u(g.x) != f2u(w.x) || f2u(g.y) != f2u(w.y) ||
            f2u(g.z) != f2u(w.z) || f2u(g.w) != f2u(w.w))
        {
            if (badAvg < 4)
                std::printf("    %s velAvg[%u]: (%.9g,%.9g,%.9g,%.9g) expected "
                            "(%.9g,%.9g,%.9g,%.9g)\n", label, i, g.x, g.y, g.z, g.w,
                            w.x, w.y, w.z, w.w);
            ++badAvg;
        }
    }
    PXG_CHECK_MSG(badAvg == 0, "%s: %zu velocity averages wrong", label, badAvg);

    // ---- Compact: the survivor set, never the index ----
    const auto two = downloadBuffer<int32_t>(dev, R.bTwo[side].h, 2);
    PXG_CHECK_MSG(uint32_t(two[1]) == E.survivors.size(),
                  "%s: the staging counter is %d, expected %zu survivors",
                  label, two[1], E.survivors.size());
    PXG_CHECK_MSG(uint32_t(two[0]) == S.nbDiffuse,
                  "%s: the PUBLISHED slot was written (%d)", label, two[0]);

    const auto outPos = downloadBuffer<F4>(dev, R.bOutPos[side].h, S.capacity);
    const auto outVel = downloadBuffer<F4>(dev, R.bOutVel[side].h, S.capacity);
    std::vector<Rec> got;
    size_t badTail = 0;
    for (uint32_t i = 0; i < S.capacity; ++i)
    {
        if (f2u(outPos[i].x) != f2u(POISON_F))
        {
            Rec r;
            r.px = f2u(outPos[i].x); r.py = f2u(outPos[i].y);
            r.pz = f2u(outPos[i].z); r.pw = f2u(outPos[i].w);
            r.vx = f2u(outVel[i].x); r.vy = f2u(outVel[i].y);
            r.vz = f2u(outVel[i].z); r.vw = f2u(outVel[i].w);
            got.push_back(r);
            if (i >= E.survivors.size()) ++badTail;   // the survivors are contiguous
        }
        else if (f2u(outVel[i].x) != f2u(POISON_F)) ++badTail;
    }
    PXG_CHECK_MSG(badTail == 0, "%s: %zu writes outside the contiguous survivor range",
                  label, badTail);

    std::vector<Rec> want = E.survivors;
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    PXG_CHECK_MSG(got.size() == want.size(), "%s: %zu records written, expected %zu",
                  label, got.size(), want.size());
    PXG_CHECK_MSG(got == want, "%s: the written survivor set differs from the predicted one",
                  label);

    std::printf("    [ note ] %s: %u diffuse -> %zu survivors; bands %u/%u/%u; "
                "%u neighbour-capped, %u clamped, %u zero-velocity\n",
                label, S.nbDiffuse, E.survivors.size(), E.bandLow, E.bandMid, E.bandHigh,
                E.nCapped, E.nClamped, E.nZeroVel);
}

static void runAll(PxgSlangComputeDevice* dev)
{
    SideArrays AA, AB;
    buildArrays(SPEC_A, AA);
    buildArrays(SPEC_B, AB);

    Rig R;
    std::vector<PxgParticleSystem> descArr(NB_SYS_SLOTS);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));
    buildSide(dev, R, 0u, SPEC_A, AA, descArr[SYS_A]);
    buildSide(dev, R, 1u, SPEC_B, AB, descArr[SYS_B]);
    R.bDesc = R.mk(dev, descArr);
    R.bActive = R.mk(dev, std::vector<uint32_t>{ SYS_A, SYS_B });

    R.blob.particleSystems = R.bDesc.addr;
    R.blob.activeParticleSystems = R.bActive.addr;
    R.blob.gx = GRAVITY.x; R.blob.gy = GRAVITY.y; R.blob.gz = GRAVITY.z;
    R.blob.dt = DT;

    const uint32_t maxDiffuse = std::max(SPEC_A.nbDiffuse, SPEC_B.nbDiffuse);
    {
        Kernel k("psDiffuseAdvect.spv", "ps_diffuseParticleUpdatePBF", {}, sizeof(PsDiffAdvectBlob));
        PXG_CHECK_MSG(k.isValid(), "the UpdatePBF pipeline must build");
        if (!k.isValid()) { R.destroy(dev); return; }
        // Grid (ceil(maxDiffuse/256), nbActiveSystems, 1) -- V1: the system on .y.
        k.dispatch((maxDiffuse + 255u) / 256u, {}, R.blob, 2u, 1u);
    }
    dev->flush();
    {
        Kernel k("psDiffuseAdvect.spv", "ps_diffuseParticleCompact", {}, sizeof(PsDiffAdvectBlob));
        PXG_CHECK_MSG(k.isValid(), "the Compact pipeline must build");
        if (!k.isValid()) { R.destroy(dev); return; }
        // Grid (x, maxDiffuseBuffersPerSystem, nbActiveSystems) -- K1: the system on .z.
        k.dispatch((maxDiffuse + 255u) / 256u, {}, R.blob, GRID_Y_COMPACT, 2u);
    }
    dev->flush();

    Expect EA = predict(SPEC_A, AA);
    Expect EB = predict(SPEC_B, AB);
    checkSide(dev, "sysA(full)", R, 0u, SPEC_A, AA, EA);
    checkSide(dev, "sysB(own-cell)", R, 1u, SPEC_B, AB, EB);

    // Coverage, demanded rather than designed.
    PXG_CHECK_MSG(EA.bandLow > 0 && EA.bandMid > 0 && EA.bandHigh > 0,
                  "sysA misses a classification band (%u/%u/%u) -- K6 is not fully exercised",
                  EA.bandLow, EA.bandMid, EA.bandHigh);
    PXG_CHECK_MSG(EA.nCapped > 0, "no diffuse particle reached the 16-neighbour cap -- V7 is dead");
    PXG_CHECK_MSG(EA.nExactTie > 0 && EB.nExactTie > 0,
                  "no fluid neighbour sits exactly at the contact distance (%u/%u) -- `<` and "
                  "`<=` are indistinguishable", EA.nExactTie, EB.nExactTie);
    PXG_CHECK_MSG(EA.nClamped > 0 || EB.nClamped > 0,
                  "no particle exceeded mMaxVelocity -- K8's clamp is dead");
    PXG_CHECK_MSG(EA.nZeroVel > 0 || EB.nZeroVel > 0,
                  "no particle had zero velocity -- K8's guard is untested");
    PXG_CHECK_MSG(EA.survivors.size() < SPEC_A.nbDiffuse,
                  "nothing expired -- K10 is untested");
    PXG_CHECK_MSG(EA.survivors.size() > 0, "everything expired -- the survivor path is untested");
    PXG_CHECK_MSG(SPEC_B.startIndex != 0u, "mStartIndex is zero -- K3's offset is invisible");
    R.destroy(dev);
}

} // namespace

PXG_TEST(psDiffuseAdvect)
{
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}
