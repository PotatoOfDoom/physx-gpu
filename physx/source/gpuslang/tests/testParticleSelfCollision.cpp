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

// P6.3 particle self-collision + one-way contact solve test: the four kernels
// (particleSelfCollisionBda.spv) on the engine ABI (BDA).
//   (1) particleSelfCollisionDescLayout — offsetof proof pinning every
//       pxgSlangParticleDesc mirror field the four kernels dereference against
//       the real engine PxgParticleSystem (the 3-way P6.0 validation: the
//       656-byte stride == SPIR-V ArrayStride, these offsetof checks, the device
//       round-trips in the kernel tests below).  Every field was already present
//       from P6.0/P6.2, so no mirror extension was required for P6.3.
//   (2) particleSelfCollisionNeighborSearch — ps_selfCollisionLaunch.  A valid
//       grid with two OVERLAPPING particles (closer than the cell width, same
//       cell) must have the 3x3x3 neighbour search record each as the other's
//       neighbour (mCollisionIndex) + count it (mParticleSelfCollisionCount) and
//       zero mDensityCollisionImpulses, matching a CPU transcription of
//       collideCell.
//   (3) particleSelfCollisionSolveOneWay — ps_solveOneWayContactDeltaVLaunch.  A
//       penetrating one-way (static) contact must resolve to a SEPARATING
//       accumDeltaP, bit-exact against a CPU transcription of
//       solvePCOutputDeltaVTGS.
//   (4) particleSelfCollisionApplyDelta — ps_applyDeltaLaunch in isolation.
//   (5) particleSelfCollisionUpdateParticle — ps_updateParticleLaunch in
//       isolation (position adjust + sorted-velocity recompute).
//   (6) particleSelfCollisionSeparates — the FULL chain (selfCollision ->
//       solveOneWayContact -> applyDelta -> updateParticle) over two overlapping
//       particles, each given the one-way contact that resolves the overlap,
//       must (a) detect the overlap and (b) move the two particles APART past
//       2*solidRestOffset, bit-exact against the CPU chain.  Because the uniform
//       free-fall scene keeps particles at rest spacing (self-collision finds no
//       penetration -> the kernels are no-ops there), this is the primary proof
//       the port resolves an overlap.
//
// NOTE: these four kernels own self-collision DETECTION (the neighbour search)
// and one-way / static contact RESOLUTION (solve -> apply -> update).  Particle-
// particle contact RESOLUTION is a separate later kernel; here the separating
// delta is produced by the one-way solve chain these kernels own, driven by the
// contact that models the resolved overlap.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

// no-CUDA-toolkit stand-in expanding the __device__/__align__ qualifiers the
// engine descriptor header decorates its inline members with; must precede
// PxgParticleSystem.h so its device-qualified ctors parse.
#include "host_defines.h"
#include "PxgParticleSystem.h"   // real engine per-system descriptor

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

static PxgParticleSystem makeZeroDesc()
{
    PxgParticleSystem ps;
    std::memset(&ps, 0, sizeof(ps));
    return ps;
}

// Phase-flag bits (PxParticleSystemFlag.h) — group 0, self-collide on.
static const uint32_t PHASE_SELF_COLLIDE = 0x00100000u; // 1<<20

// ---------------------------------------------------------------------------
// CPU transcription of solvePCOutputDeltaVTGS (particlesystem.cu:1473) for the
// no-adhesion, no-friction path exercised here (mat.friction ==
// mat.adhesion == mat.adhesionRadiusScale == 0 -> deltaFr terms vanish).
// ---------------------------------------------------------------------------
static float cpuSolveDeltaF(const float normal[3], float error, const float dp[3],
                            float velMultiplier, float dt, bool isVelIter, float appliedForceX)
{
    const float projectionDt = isVelIter ? 0.0f : dt;
    const float dotND = normal[0] * dp[0] + normal[1] * dp[1] + normal[2] * dp[2];
    const float separation = (error - dotND + 0.0f) + 0.0f * projectionDt;
    const float clamp = -appliedForceX;
    const float deltaF = std::fmax(clamp, -separation * velMultiplier);
    return deltaF;
}

// Full CPU transcription of the ps_solveOneWayContactDeltaVLaunch inner body for
// one particle with one contact (friction/adhesion zeroed), returning the
// accumDeltaP increment.
static void cpuSolveOneWay(const float normalPen[4], const float dp4[4], float bias,
                           float dt, bool isVelIter, float out[3])
{
    const float invMass = dp4[3];
    const float velMultiplier = 1.0f / invMass;
    const float normal[3] = { -normalPen[0], -normalPen[1], -normalPen[2] };
    const float dp[3] = { dp4[0], dp4[1], dp4[2] };
    const float deltaF = cpuSolveDeltaF(normal, normalPen[3], dp, velMultiplier, dt, isVelIter, 0.0f);
    // friction terms are 0 (frictionCoefficient == 0), so deltaLinVel is purely
    // the normal push: -(normal*deltaF) * invMass * bias.
    float deltaLin[3] = { -(normal[0] * deltaF), -(normal[1] * deltaF), -(normal[2] * deltaF) };
    const float scale = (deltaF != 0.0f) ? 1.0f : 0.0f;
    float thisDeltaP[3] = { deltaLin[0] * invMass * bias, deltaLin[1] * invMass * bias,
                            deltaLin[2] * invMass * bias };
    const float denom = scale;
    if (denom != 0.0f)
    {
        const float inv = 1.0f / denom;
        thisDeltaP[0] *= inv; thisDeltaP[1] *= inv; thisDeltaP[2] *= inv;
    }
    out[0] = thisDeltaP[0]; out[1] = thisDeltaP[1]; out[2] = thisDeltaP[2];
}

// ---------------------------------------------------------------------------
// (1) Descriptor mirror layout proof.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionDescLayout)
{
    PXG_CHECK_MSG(sizeof(PxgParticleSystem) == 656, "PxgParticleSystem sizeof == 656 (got %zu)",
                  sizeof(PxgParticleSystem));

    // ps_selfCollisionLaunch fields
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPositions_InvMass) == 24);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPhaseArray) == 40);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedToUnsortedMapping) == 56);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mParticleSelfCollisionCount) == 64);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCollisionIndex) == 72);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mParticleMaterials) == 80);
    PXG_CHECK(offsetof(PxGpuParticleData, mGridCellWidth) == 0);      // -> struct off 88
    PXG_CHECK(offsetof(PxGpuParticleData, mMaxNeighborhood) == 36);   // -> struct off 124
    PXG_CHECK(offsetof(PxgParticleSystem, mCellStart) == 168);
    PXG_CHECK(offsetof(PxgParticleSystem, mCellEnd) == 176);
    PXG_CHECK(offsetof(PxgParticleSystem, mDensityCollisionImpulses) == 192);
    PXG_CHECK(offsetof(PxgParticleSystem, mRestArray) == 240);

    // ps_solveOneWayContactDeltaVLaunch fields
    PXG_CHECK(offsetof(PxgParticleSystem, mAccumDeltaP) == 152);
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedDeltaP) == 160);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactInfos) == 200);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayForces) == 208);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactCount) == 232);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleMaterialStride) == 632);
    PXG_CHECK(offsetof(PxgParticleSystem, mPhaseGroupToMaterialHandle) == 640);
    PXG_CHECK(offsetof(PxgParticleSystem, mData) == 248);
    PXG_CHECK(offsetof(PxgParticleSystemData, mRestOffset) == 20);        // -> struct off 268
    PXG_CHECK(offsetof(PxgParticleSystemData, mSolidRestOffset) == 24);   // -> struct off 272
    PXG_CHECK(offsetof(PxgParticleSystemData, mRelaxationFactor) == 52);  // -> struct off 300

    // ps_updateParticleLaunch fields
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedVelocities) == 32);
    PXG_CHECK(offsetof(PxgParticleSystem, mDelta) == 368);

    // PxsParticleMaterialData layout the solve indexes by raw byte offset.
    PXG_CHECK(offsetof(PxsParticleMaterialData, friction) == 0);
    PXG_CHECK(offsetof(PxsParticleMaterialData, adhesion) == 8);
    PXG_CHECK(offsetof(PxsParticleMaterialData, adhesionRadiusScale) == 16);
}

// ---------------------------------------------------------------------------
// (2) ps_selfCollisionLaunch — overlap neighbour search.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionNeighborSearch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 4;
    const float cellWidth = 0.3f;           // cellWidthSq = 0.09
    const uint32_t gs = 4;                   // power-of-two grid
    const uint32_t numCells = gs * gs * gs;  // 64
    const uint32_t EMPTY = 0xffffffffu;

    // Two overlapping pairs, already in sorted (cell-hash) order.  Positions are
    // chosen so calcGridPos(x/cellWidth) actually lands each pair in the cell its
    // cellStart/cellEnd points at: floor(x/0.3) == 0 for pair A, == 2 for pair B.
    //   sorted 0,1 in cell (0,0,0) hash 0; sorted 2,3 in cell (2,0,0) hash 2.
    std::vector<F4> sortedPos(N);
    sortedPos[0] = { 0.00f, 0.05f, 0.05f, 1.0f };
    sortedPos[1] = { 0.15f, 0.05f, 0.05f, 1.0f }; // floor(0.15/0.3)=0; 0.15 < cellWidth from #0
    sortedPos[2] = { 0.65f, 0.05f, 0.05f, 1.0f }; // floor(0.65/0.3)=2
    sortedPos[3] = { 0.70f, 0.05f, 0.05f, 1.0f }; // floor(0.70/0.3)=2; 0.05 from #2

    std::vector<uint32_t> sortedPhase(N, PHASE_SELF_COLLIDE); // group 0, self-collide, no filter
    std::vector<uint32_t> mapping = { 0u, 1u, 2u, 3u };       // sortedToUnsorted (identity)

    std::vector<uint32_t> cellStart(numCells, EMPTY);
    std::vector<uint32_t> cellEnd(numCells, 0u);
    cellStart[0] = 0u; cellEnd[0] = 2u; // cell 0 spans sorted [0,2)
    cellStart[2] = 2u; cellEnd[2] = 4u; // cell 2 spans sorted [2,4)

    // outputs seeded with sentinels; mCollisionIndex/mDensityCollisionImpulses
    // sized maxNeighborhood*N.
    const uint32_t maxNeighborhood = 8u;
    std::vector<uint32_t> selfCount(N, 0xdeadbeefu);
    std::vector<uint32_t> collisionIndex(N * maxNeighborhood, 0xccccccccu);
    std::vector<F2> collisionForces(N * maxNeighborhood, { 7.0f, 7.0f });

    DevBuf bPos = makeDev(dev, sortedPos);
    DevBuf bPhase = makeDev(dev, sortedPhase);
    DevBuf bMap = makeDev(dev, mapping);
    DevBuf bStart = makeDev(dev, cellStart);
    DevBuf bEnd = makeDev(dev, cellEnd);
    DevBuf bCount = makeDev(dev, selfCount);
    DevBuf bColIdx = makeDev(dev, collisionIndex);
    DevBuf bColForce = makeDev(dev, collisionForces);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bPhase.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bMap.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bEnd.addr));
    ps.mParticleSelfCollisionCount = reinterpret_cast<PxU32*>(size_t(bCount.addr));
    ps.mCollisionIndex = reinterpret_cast<PxU32*>(size_t(bColIdx.addr));
    ps.mDensityCollisionImpulses = reinterpret_cast<float2*>(size_t(bColForce.addr));
    ps.mCommonData.mNumParticles = N;
    ps.mCommonData.mGridCellWidth = cellWidth;
    ps.mCommonData.mGridSizeX = gs;
    ps.mCommonData.mGridSizeY = gs;
    ps.mCommonData.mGridSizeZ = gs;
    ps.mCommonData.mMaxNeighborhood = maxNeighborhood;
    ps.mData.mSolidRestOffset = 0.1f; // only used under the (disabled) filter branch

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);

    struct SelfPush { uint64_t particleSystems; uint32_t id; };
    SelfPush push = { bDesc.addr, 0u };

    Kernel k("particleSelfCollisionBda.spv", "ps_selfCollisionLaunch", {}, sizeof(SelfPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u); // grid (numBlocks,1,1)
    dev->flush();

    // CPU reference: expected one neighbour per particle (its overlapping pair).
    std::vector<uint32_t> refCount = { 1u, 1u, 1u, 1u };
    std::vector<uint32_t> refIdx = collisionIndex;   // sentinels except entries written
    refIdx[0] = 1u; // particle 0 -> neighbour 1  (slot k=0 -> index gtid + 0*N)
    refIdx[1] = 0u; // particle 1 -> neighbour 0
    refIdx[2] = 3u; // particle 2 -> neighbour 3
    refIdx[3] = 2u; // particle 3 -> neighbour 2

    const auto gCount = downloadBuffer<uint32_t>(dev, bCount.h, N);
    const auto gIdx = downloadBuffer<uint32_t>(dev, bColIdx.h, N * maxNeighborhood);
    const auto gForce = downloadBuffer<F2>(dev, bColForce.h, N * maxNeighborhood);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (gCount[i] != refCount[i]) { ++mm; std::printf("  count MM %u: ref %u got %u\n", i, refCount[i], gCount[i]); }
    for (uint32_t i = 0; i < N * maxNeighborhood; ++i)
        if (gIdx[i] != refIdx[i]) { ++mm; std::printf("  collisionIndex MM %u: ref %u got %u\n", i, refIdx[i], gIdx[i]); }
    // collisionForces: only slot k=0 of each particle written to (0,0); rest sentinel.
    for (uint32_t i = 0; i < N; ++i)
    {
        if (f2u(gForce[i].x) != f2u(0.0f) || f2u(gForce[i].y) != f2u(0.0f))
        { ++mm; std::printf("  force MM %u: got(%g %g)\n", i, gForce[i].x, gForce[i].y); }
    }
    PXG_CHECK_MSG(mm == 0, "ps_selfCollisionLaunch neighbour search matches CPU (%zu mismatches)", mm);
    PXG_CHECK_MSG(gCount[0] == 1u && gIdx[0] == 1u, "particle 0 detects overlapping neighbour 1");

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bPhase.h); dev->destroyBuffer(bMap.h);
    dev->destroyBuffer(bStart.h); dev->destroyBuffer(bEnd.h); dev->destroyBuffer(bCount.h);
    dev->destroyBuffer(bColIdx.h); dev->destroyBuffer(bColForce.h); dev->destroyBuffer(bDesc.h);
}

// ---------------------------------------------------------------------------
// (2') P9.99b -- particleSelfCollisionNeighborSearch with push id 2.  The host runs
// ps_selfCollisionLaunch once per active system with `id` = its remap index
// (PxgParticleSystemCore.cpp:1041ff); the single-system fixture only ever passed 0.
// Pool of three: slots 0 and 1 are decoys (6 particles on a 4^3 grid, maxNeighborhood
// 8, a valid cell structure, sentinel outputs), slot 2 is the real system -- the
// fixture's two overlapping pairs on an 8^3 grid with maxNeighborhood 6.
// ---------------------------------------------------------------------------
struct SelfSys
{
    uint32_t N = 0, gs = 0, maxNbr = 0;
    std::vector<F4> sortedPos;
    std::vector<uint32_t> sortedPhase, mapping, cellStart, cellEnd, selfCount, collisionIndex;
    std::vector<F2> collisionForces;
    DevBuf bPos, bPhase, bMap, bStart, bEnd, bCount, bColIdx, bColForce;
};

PXG_TEST(particleSelfCollisionNeighborSearch_p99Id2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float cellWidth = 0.3f;
    const uint32_t EMPTY = 0xffffffffu;
    SelfSys s[3];
    std::vector<PxgParticleSystem> descArr(3);
    for (uint32_t k = 0; k < 3; ++k)
    {
        SelfSys& c = s[k];
        const bool real = k == 2u;
        c.N = real ? 4u : 6u;
        c.gs = real ? 8u : 4u;
        c.maxNbr = real ? 6u : 8u;
        const uint32_t numCells = c.gs * c.gs * c.gs;
        // sorted 0,1 in cell (0,0,0); 2,3 in cell (2,0,0); decoys add 4,5 in cell (3,0,0).
        c.sortedPos = { { 0.00f, 0.05f, 0.05f, 1.0f }, { 0.15f, 0.05f, 0.05f, 1.0f },
                        { 0.65f, 0.05f, 0.05f, 1.0f }, { 0.70f, 0.05f, 0.05f, 1.0f } };
        if (!real) { c.sortedPos.push_back({ 0.95f, 0.05f, 0.05f, 1.0f }); c.sortedPos.push_back({ 1.00f, 0.05f, 0.05f, 1.0f }); }
        c.sortedPhase.assign(c.N, PHASE_SELF_COLLIDE);
        c.mapping.resize(c.N);
        for (uint32_t i = 0; i < c.N; ++i) c.mapping[i] = i;
        c.cellStart.assign(numCells, EMPTY);
        c.cellEnd.assign(numCells, 0u);
        c.cellStart[0] = 0u; c.cellEnd[0] = 2u;
        c.cellStart[2] = 2u; c.cellEnd[2] = 4u;
        if (!real) { c.cellStart[3] = 4u; c.cellEnd[3] = 6u; }
        c.selfCount.assign(c.N, 0xdeadbeefu);
        c.collisionIndex.assign(c.N * c.maxNbr, 0xccccccccu);
        c.collisionForces.assign(c.N * c.maxNbr, { 7.0f, 7.0f });
        c.bPos = makeDev(dev, c.sortedPos); c.bPhase = makeDev(dev, c.sortedPhase); c.bMap = makeDev(dev, c.mapping);
        c.bStart = makeDev(dev, c.cellStart); c.bEnd = makeDev(dev, c.cellEnd); c.bCount = makeDev(dev, c.selfCount);
        c.bColIdx = makeDev(dev, c.collisionIndex); c.bColForce = makeDev(dev, c.collisionForces);

        PxgParticleSystem ps = makeZeroDesc();
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(c.bPos.addr));
        ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(c.bPhase.addr));
        ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(c.bMap.addr));
        ps.mCellStart = reinterpret_cast<PxU32*>(size_t(c.bStart.addr));
        ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(c.bEnd.addr));
        ps.mParticleSelfCollisionCount = reinterpret_cast<PxU32*>(size_t(c.bCount.addr));
        ps.mCollisionIndex = reinterpret_cast<PxU32*>(size_t(c.bColIdx.addr));
        ps.mDensityCollisionImpulses = reinterpret_cast<float2*>(size_t(c.bColForce.addr));
        ps.mCommonData.mNumParticles = c.N;
        ps.mCommonData.mGridCellWidth = cellWidth;
        ps.mCommonData.mGridSizeX = c.gs;
        ps.mCommonData.mGridSizeY = c.gs;
        ps.mCommonData.mGridSizeZ = c.gs;
        ps.mCommonData.mMaxNeighborhood = c.maxNbr;
        ps.mData.mSolidRestOffset = real ? 0.15f : 0.1f; // only used under the (disabled) filter branch
        descArr[k] = ps;
    }
    DevBuf bDesc = makeDev(dev, descArr);

    struct SelfPush { uint64_t particleSystems; uint32_t id; };
    SelfPush push = { bDesc.addr, 2u };
    {
        Kernel k("particleSelfCollisionBda.spv", "ps_selfCollisionLaunch", {}, sizeof(SelfPush));
        k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    }
    dev->flush();

    size_t mm = 0;
    for (uint32_t k = 0; k < 3; ++k)
    {
        const SelfSys& c = s[k];
        std::vector<uint32_t> refCount = c.selfCount, refIdx = c.collisionIndex;
        std::vector<F2> refForce = c.collisionForces;
        if (k == 2u)
        {
            // one neighbour per particle (its overlapping pair), output slot 0 = index gtid
            const uint32_t nb[4] = { 1u, 0u, 3u, 2u };
            for (uint32_t i = 0; i < c.N; ++i) { refCount[i] = 1u; refIdx[i] = nb[i]; refForce[i] = { 0.0f, 0.0f }; }
        }
        const auto gCount = downloadBuffer<uint32_t>(dev, c.bCount.h, c.N);
        const auto gIdx = downloadBuffer<uint32_t>(dev, c.bColIdx.h, c.N * c.maxNbr);
        const auto gForce = downloadBuffer<F2>(dev, c.bColForce.h, c.N * c.maxNbr);
        for (uint32_t i = 0; i < c.N; ++i)
            if (gCount[i] != refCount[i]) { if (mm < 8) std::printf("  slot%u count MM %u: ref %u got %u\n", k, i, refCount[i], gCount[i]); ++mm; }
        for (uint32_t i = 0; i < c.N * c.maxNbr; ++i)
        {
            if (gIdx[i] != refIdx[i]) { if (mm < 8) std::printf("  slot%u collisionIndex MM %u: ref %u got %u\n", k, i, refIdx[i], gIdx[i]); ++mm; }
            if (f2u(gForce[i].x) != f2u(refForce[i].x) || f2u(gForce[i].y) != f2u(refForce[i].y))
            { if (mm < 8) std::printf("  slot%u force MM %u: got(%g %g)\n", k, i, gForce[i].x, gForce[i].y); ++mm; }
        }
    }
    PXG_CHECK_MSG(mm == 0, "ps_selfCollisionLaunch, push id 2, decoys at slots 0/1 untouched: neighbour search "
                  "matches CPU (%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k)
    {
        const SelfSys& c = s[k];
        for (const DevBuf* b : { &c.bPos, &c.bPhase, &c.bMap, &c.bStart, &c.bEnd, &c.bCount, &c.bColIdx, &c.bColForce })
            dev->destroyBuffer(b->h);
    }
    dev->destroyBuffer(bDesc.h);
}

// ---------------------------------------------------------------------------
// DIVERGENT neighbour counts inside one subgroup.
//
// particleSelfCollisionNeighborSearch above gives every particle exactly ONE neighbour, so
// its output-writing loop runs the same number of iterations in every lane.  That makes it
// blind to a defect this port hit for real in softBodyContactRemapBda: llvmpipe drops a
// loop's per-lane exit mask when the bound was redefined inside a conditional block that
// also stores, and then every lane of the subgroup runs the trip count of the longest one.
// With uniform counts that is invisible; the shape is present here, because
// `totalCollidedParticles` is assigned inside the `if (globalThreadIndex < numParticles)`
// block that also writes particleSelfCollisionCounts, and is then the bound of the output
// loop below it.
//
// Three particles in ONE cell, arranged so the counts differ: particle 0 is within the
// contact distance of both others, and the other two are NOT within it of each other.
// cellWidthSq = 0.09, and the test is `distanceSq <= contactDistanceSq`:
//   |p0-p1|^2 = 0.04, |p0-p2|^2 = 0.08  -> both overlap particle 0
//   |p1-p2|^2 = 0.12                    -> p1 and p2 do not overlap
// so the counts are (2, 1, 1) and lanes 0..2 of subgroup 0 have different trip counts.
// The load-bearing assertion is on the SECOND output slot: only particle 0 may write it.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionDivergentCounts)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 3;
    const float cellWidth = 0.3f;            // cellWidthSq = 0.09
    const uint32_t gs = 4;
    const uint32_t numCells = gs * gs * gs;
    const uint32_t EMPTY = 0xffffffffu;

    // all three inside cell (0,0,0): every coordinate < cellWidth.
    std::vector<F4> sortedPos(N);
    sortedPos[0] = { 0.05f, 0.05f, 0.05f, 1.0f };
    sortedPos[1] = { 0.25f, 0.05f, 0.05f, 1.0f };
    sortedPos[2] = { 0.05f, 0.25f, 0.25f, 1.0f };

    std::vector<uint32_t> sortedPhase(N, PHASE_SELF_COLLIDE);
    std::vector<uint32_t> mapping = { 0u, 1u, 2u };
    std::vector<uint32_t> cellStart(numCells, EMPTY);
    std::vector<uint32_t> cellEnd(numCells, 0u);
    cellStart[0] = 0u; cellEnd[0] = N;

    const uint32_t maxNeighborhood = 8u;
    std::vector<uint32_t> selfCount(N, 0xdeadbeefu);
    std::vector<uint32_t> collisionIndex(N * maxNeighborhood, 0xccccccccu);
    std::vector<F2> collisionForces(N * maxNeighborhood, { 7.0f, 7.0f });

    DevBuf bPos = makeDev(dev, sortedPos);
    DevBuf bPhase = makeDev(dev, sortedPhase);
    DevBuf bMap = makeDev(dev, mapping);
    DevBuf bStart = makeDev(dev, cellStart);
    DevBuf bEnd = makeDev(dev, cellEnd);
    DevBuf bCount = makeDev(dev, selfCount);
    DevBuf bColIdx = makeDev(dev, collisionIndex);
    DevBuf bColForce = makeDev(dev, collisionForces);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bPhase.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bMap.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bEnd.addr));
    ps.mParticleSelfCollisionCount = reinterpret_cast<PxU32*>(size_t(bCount.addr));
    ps.mCollisionIndex = reinterpret_cast<PxU32*>(size_t(bColIdx.addr));
    ps.mDensityCollisionImpulses = reinterpret_cast<float2*>(size_t(bColForce.addr));
    ps.mCommonData.mNumParticles = N;
    ps.mCommonData.mGridCellWidth = cellWidth;
    ps.mCommonData.mGridSizeX = gs;
    ps.mCommonData.mGridSizeY = gs;
    ps.mCommonData.mGridSizeZ = gs;
    ps.mCommonData.mMaxNeighborhood = maxNeighborhood;
    ps.mData.mSolidRestOffset = 0.1f;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);

    struct SelfPush { uint64_t particleSystems; uint32_t id; };
    SelfPush push = { bDesc.addr, 0u };
    Kernel("particleSelfCollisionBda.spv", "ps_selfCollisionLaunch", {}, sizeof(SelfPush))
        .dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const auto gCount = downloadBuffer<uint32_t>(dev, bCount.h, N);
    const auto gIdx = downloadBuffer<uint32_t>(dev, bColIdx.h, N * maxNeighborhood);
    const auto gForce = downloadBuffer<F2>(dev, bColForce.h, N * maxNeighborhood);
    std::printf("  counts = (%u, %u, %u)\n", gCount[0], gCount[1], gCount[2]);

    PXG_CHECK_MSG(gCount[0] == 2u && gCount[1] == 1u && gCount[2] == 1u,
                  "the fixture must produce DIVERGENT counts (2, 1, 1) or it tests nothing -- "
                  "got (%u, %u, %u)", gCount[0], gCount[1], gCount[2]);
    // slot k of particle p lives at p + k * numParticles.
    PXG_CHECK_MSG(gIdx[0] == 1u && gIdx[0 + N] == 2u,
                  "particle 0 must record both neighbours, got (%u, %u)", gIdx[0], gIdx[0 + N]);
    PXG_CHECK_MSG(gIdx[1] == 0u && gIdx[2] == 0u,
                  "particles 1 and 2 must each record particle 0, got (%u, %u)", gIdx[1], gIdx[2]);
    // THE mask check: slot 1 belongs to particle 0 alone.  Every lane running particle 0's
    // trip count would write slots 1 + N and 2 + N as well.
    for (uint32_t p = 1; p < N; ++p)
    {
        PXG_CHECK_MSG(gIdx[p + N] == 0xccccccccu,
                      "particle %u has ONE neighbour, so its second collisionIndex slot must "
                      "stay untouched (got %u) -- the output loop ran a longer lane's trip "
                      "count", p, gIdx[p + N]);
        PXG_CHECK_MSG(f2u(gForce[p + N].x) == f2u(7.0f) && f2u(gForce[p + N].y) == f2u(7.0f),
                      "particle %u has ONE neighbour, so its second collisionForces slot must "
                      "stay at the sentinel (got %g, %g)", p, gForce[p + N].x, gForce[p + N].y);
    }
    // and the slots that SHOULD be written are zeroed
    PXG_CHECK_MSG(f2u(gForce[0].x) == f2u(0.0f) && f2u(gForce[0 + N].x) == f2u(0.0f) &&
                  f2u(gForce[1].x) == f2u(0.0f) && f2u(gForce[2].x) == f2u(0.0f),
                  "every in-range collisionForces slot must be zeroed");

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bPhase.h); dev->destroyBuffer(bMap.h);
    dev->destroyBuffer(bStart.h); dev->destroyBuffer(bEnd.h); dev->destroyBuffer(bCount.h);
    dev->destroyBuffer(bColIdx.h); dev->destroyBuffer(bColForce.h); dev->destroyBuffer(bDesc.h);
}

// Helper: build a descriptor + material/phase-handle buffers for the solve path.
struct SolveBufs
{
    DevBuf mat, phaseHandle;
};
static SolveBufs makeMaterialBufs(PxgSlangComputeDevice* dev, PxgParticleSystem& ps,
                                  std::vector<DevBuf>& keep)
{
    SolveBufs sb;
    // one PxsParticleMaterialData (5 floats) with friction/adhesion/adhScale = 0.
    std::vector<float> material = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    sb.mat = makeDev(dev, material);
    // phaseGroupToMaterialHandle: group 0 -> material index 0 (low 16 bits of word0).
    std::vector<uint32_t> handle = { 0u };
    sb.phaseHandle = makeDev(dev, handle);
    ps.mParticleMaterials = reinterpret_cast<PxsParticleMaterialData*>(size_t(sb.mat.addr));
    ps.mParticleMaterialStride = uint32_t(sizeof(PxsParticleMaterialData)); // 20
    ps.mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(sb.phaseHandle.addr));
    keep.push_back(sb.mat);
    keep.push_back(sb.phaseHandle);
    return sb;
}

// ---------------------------------------------------------------------------
// (3) ps_solveOneWayContactDeltaVLaunch — penetrating contact -> separating dP.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionSolveOneWay)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 1;
    const float dt = 1.0f / 60.0f;
    const float bias = 1.0f;
    const float invMass = 1.0f;
    // Contact normal points +y, particle penetrating a floor by 0.1 (pen < 0).
    const float pen = -0.1f;
    F4 normalPen = { 0.0f, 1.0f, 0.0f, pen };

    std::vector<F4> oneWayContacts = { normalPen };        // PxgParticleContactInfo == float4
    std::vector<uint32_t> oneWayCount = { 1u };
    std::vector<F2> oneWayForces(PxgParticleContactInfo::MaxStaticContactsPerParticle * N, { 0.0f, 0.0f });
    std::vector<F4> sortedDeltaP = { { 0.0f, 0.0f, 0.0f, invMass } };
    std::vector<F4> accumDeltaP = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    std::vector<uint32_t> sortedPhase = { PHASE_SELF_COLLIDE };

    DevBuf bContacts = makeDev(dev, oneWayContacts);
    DevBuf bCount = makeDev(dev, oneWayCount);
    DevBuf bForces = makeDev(dev, oneWayForces);
    DevBuf bDeltaP = makeDev(dev, sortedDeltaP);
    DevBuf bAccum = makeDev(dev, accumDeltaP);
    DevBuf bPhase = makeDev(dev, sortedPhase);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(bContacts.addr));
    ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(bCount.addr));
    ps.mOneWayForces = reinterpret_cast<float2*>(size_t(bForces.addr));
    ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(bDeltaP.addr));
    ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(bAccum.addr));
    ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bPhase.addr));
    ps.mCommonData.mNumParticles = N;
    ps.mData.mRestOffset = 0.05f;

    std::vector<DevBuf> keep;
    makeMaterialBufs(dev, ps, keep);

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct SolvePush { uint64_t systems, active; float invDt, dt, bias; uint32_t velIter; };
    SolvePush push = { bDesc.addr, bActive.addr, 1.0f / dt, dt, bias, 0u };

    Kernel k("particleSelfCollisionBda.spv", "ps_solveOneWayContactDeltaVLaunch", {}, sizeof(SolvePush));
    k.dispatch(1u, {}, push, /*gridY (nbActive)*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    float ref[3];
    cpuSolveOneWay(&normalPen.x, &sortedDeltaP[0].x, bias, dt, false, ref);

    const auto gAccum = downloadBuffer<F4>(dev, bAccum.h, N);
    size_t mm = 0;
    if (f2u(gAccum[0].x) != f2u(ref[0]) || f2u(gAccum[0].y) != f2u(ref[1]) || f2u(gAccum[0].z) != f2u(ref[2]))
    {
        ++mm;
        std::printf("  accumDeltaP MM: ref(%g %g %g) got(%g %g %g)\n", ref[0], ref[1], ref[2],
                    gAccum[0].x, gAccum[0].y, gAccum[0].z);
    }
    PXG_CHECK_MSG(mm == 0, "ps_solveOneWayContactDeltaVLaunch bit-exact vs CPU");
    PXG_CHECK_MSG(gAccum[0].y > 0.0f, "penetrating contact -> separating (+y) delta (got %g)", gAccum[0].y);

    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bCount.h); dev->destroyBuffer(bForces.h);
    dev->destroyBuffer(bDeltaP.h); dev->destroyBuffer(bAccum.h); dev->destroyBuffer(bPhase.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
    for (auto& b : keep) dev->destroyBuffer(b.h);
}

// ---------------------------------------------------------------------------
// (3') P9.99b -- the TWO-SYSTEM twin of particleSelfCollisionSolveOneWay: a pool of
// three descriptors, act = {2, 0}, gridY = 2.  Systems 2 and 0 differ in N, contacts,
// penetrations, inverse masses and rest offset; the decoy at slot 1 has the largest
// count, a contact on every particle and sentinel accumulators that must survive.
// Oracle: cpuSolveOneWay (one contact per particle, friction/adhesion zero).
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionSolveOneWay_p99TwoSystems)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float bias = 0.75f;
    const uint32_t NS[3] = { 2u, 6u, 3u };
    const uint32_t MAXC = PxgParticleContactInfo::MaxStaticContactsPerParticle;

    struct Sys
    {
        std::vector<F4> contacts, sortedDeltaP, accumDeltaP;
        std::vector<uint32_t> count, phase;
        std::vector<F2> forces;
        DevBuf bContacts, bCount, bForces, bDeltaP, bAccum, bPhase;
    };
    Sys s[3];
    std::vector<DevBuf> keep;
    std::vector<PxgParticleSystem> descArr(3);
    for (uint32_t k = 0; k < 3; ++k)
    {
        Sys& c = s[k];
        const uint32_t N = NS[k];
        c.contacts.assign(MAXC * N, { 0.0f, 0.0f, 0.0f, 0.0f });
        c.count.assign(N, 1u);
        c.forces.assign(MAXC * N, { 0.0f, 0.0f });
        c.sortedDeltaP.resize(N);
        c.accumDeltaP.resize(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            const float fi = float(i), fk = float(k);
            // per system a different normal and depth: +y floors for 0, a tilted wall for 2
            const F4 n = (k == 2u) ? F4{ 0.6f, 0.8f, 0.0f, -0.2f + 0.05f * fi } : F4{ 0.0f, 1.0f, 0.0f, -0.1f - 0.03f * fi };
            c.contacts[i] = n;
            // dp lies along z, orthogonal to both normals, so normal.dot(dp) is an exact 0 in any
            // association; .w = inverse mass
            c.sortedDeltaP[i] = { 0.0f, 0.0f, 0.005f * fi + 0.01f * fk, (k == 2u) ? 0.5f : 1.0f };
            c.accumDeltaP[i] = { 0.1f * fi + fk, -0.25f, 0.5f * fk, 0.0f };
        }
        if (k == 2u) c.count[1] = 0u;   // a particle without contact keeps its accumulator
        c.phase.assign(N, PHASE_SELF_COLLIDE);
        c.bContacts = makeDev(dev, c.contacts); c.bCount = makeDev(dev, c.count); c.bForces = makeDev(dev, c.forces);
        c.bDeltaP = makeDev(dev, c.sortedDeltaP); c.bAccum = makeDev(dev, c.accumDeltaP); c.bPhase = makeDev(dev, c.phase);

        PxgParticleSystem ps = makeZeroDesc();
        ps.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(c.bContacts.addr));
        ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(c.bCount.addr));
        ps.mOneWayForces = reinterpret_cast<float2*>(size_t(c.bForces.addr));
        ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(c.bDeltaP.addr));
        ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(c.bAccum.addr));
        ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(c.bPhase.addr));
        ps.mCommonData.mNumParticles = N;
        ps.mData.mRestOffset = (k == 2u) ? 0.15f : 0.05f;
        makeMaterialBufs(dev, ps, keep);
        descArr[k] = ps;
    }
    DevBuf bDesc = makeDev(dev, descArr);
    const std::vector<uint32_t> active = { 2u, 0u };
    DevBuf bActive = makeDev(dev, active);

    struct SolvePush { uint64_t systems, active; float invDt, dt, bias; uint32_t velIter; };
    SolvePush push = { bDesc.addr, bActive.addr, 1.0f / dt, dt, bias, 0u };
    {
        Kernel k("particleSelfCollisionBda.spv", "ps_solveOneWayContactDeltaVLaunch", {}, sizeof(SolvePush));
        k.dispatch(1u, {}, push, /*gridY (nbActive)*/ 2u, /*gridZ*/ 1u);
    }
    dev->flush();

    size_t mm = 0;
    for (uint32_t k = 0; k < 3; ++k)
    {
        const Sys& c = s[k];
        const uint32_t N = NS[k];
        std::vector<F4> ref = c.accumDeltaP;
        if (k != 1u)
            for (uint32_t i = 0; i < N; ++i)
            {
                if (c.count[i] == 0u) continue;
                float d[3];
                cpuSolveOneWay(&c.contacts[i].x, &c.sortedDeltaP[i].x, bias, dt, false, d);
                ref[i].x += d[0]; ref[i].y += d[1]; ref[i].z += d[2];
            }
        const auto g = downloadBuffer<F4>(dev, c.bAccum.h, N);
        for (uint32_t i = 0; i < N; ++i)
            if (f2u(g[i].x) != f2u(ref[i].x) || f2u(g[i].y) != f2u(ref[i].y) || f2u(g[i].z) != f2u(ref[i].z) ||
                f2u(g[i].w) != f2u(ref[i].w))
            {
                if (mm < 8)
                    std::printf("  sys%u accumDeltaP MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", k, i, ref[i].x, ref[i].y,
                                ref[i].z, ref[i].w, g[i].x, g[i].y, g[i].z, g[i].w);
                ++mm;
            }
    }
    PXG_CHECK_MSG(mm == 0, "ps_solveOneWayContactDeltaVLaunch, systems 2 and 0 via act {2,0}, decoy untouched: "
                  "bit-exact vs CPU (%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k)
    {
        const Sys& c = s[k];
        for (const DevBuf* b : { &c.bContacts, &c.bCount, &c.bForces, &c.bDeltaP, &c.bAccum, &c.bPhase })
            dev->destroyBuffer(b->h);
    }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
    for (auto& b : keep) dev->destroyBuffer(b.h);
}

// ---------------------------------------------------------------------------
// (4) ps_applyDeltaLaunch — SOR-average mDelta into mAccumDeltaP.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionApplyDelta)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 3;
    const float relaxation = 1.5f;

    std::vector<F4> deltas(N), accum(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        deltas[i] = { 0.1f * float(i + 1), -0.2f * float(i + 1), 0.05f, 2.0f + float(i) }; // .w = weight
        accum[i] = { float(i), -float(i), 0.5f * float(i), 3.0f };
    }

    DevBuf bDeltas = makeDev(dev, deltas);
    DevBuf bAccum = makeDev(dev, accum);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mDelta = reinterpret_cast<float4*>(size_t(bDeltas.addr));
    ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(bAccum.addr));
    ps.mCommonData.mNumParticles = N;
    ps.mData.mRelaxationFactor = relaxation;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct ApplyPush { uint64_t systems, active; float dt; };
    ApplyPush push = { bDesc.addr, bActive.addr, 1.0f / 60.0f };

    Kernel k("particleSelfCollisionBda.spv", "ps_applyDeltaLaunch", {}, sizeof(ApplyPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto gAccum = downloadBuffer<F4>(dev, bAccum.h, N);
    const auto gDeltas = downloadBuffer<F4>(dev, bDeltas.h, N);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4 j = deltas[i];
        const F4 a = accum[i];
        const float scale = 1.0f / std::fmax(j.w * relaxation, 1.0f);
        const float ax = a.x + j.x * scale, ay = a.y + j.y * scale, az = a.z + j.z * scale;
        if (f2u(gAccum[i].x) != f2u(ax) || f2u(gAccum[i].y) != f2u(ay) ||
            f2u(gAccum[i].z) != f2u(az) || f2u(gAccum[i].w) != f2u(0.0f))
        { ++mm; std::printf("  accum MM %u\n", i); }
        if (f2u(gDeltas[i].x) != f2u(0.0f) || f2u(gDeltas[i].y) != f2u(0.0f) ||
            f2u(gDeltas[i].z) != f2u(0.0f) || f2u(gDeltas[i].w) != f2u(0.0f))
        { ++mm; std::printf("  delta-clear MM %u\n", i); }
    }
    PXG_CHECK_MSG(mm == 0, "ps_applyDeltaLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bDeltas.h); dev->destroyBuffer(bAccum.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// (4') P9.99b -- the TWO-SYSTEM twin of particleSelfCollisionApplyDelta: act = {2, 0},
// gridY = 2; systems 2 and 0 differ in N (5 / 3), data and relaxation factor
// (0.8 / 1.5); the decoy at slot 1 (N 7, relaxation 2.5) must survive bit for bit.
// Oracle: particleSelfCollisionApplyDelta's, per system.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionApplyDelta_p99TwoSystems)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NS[3] = { 3u, 7u, 5u };
    const float RELAX[3] = { 1.5f, 2.5f, 0.8f };
    std::vector<F4> deltas[3], accum[3];
    DevBuf bDeltas[3], bAccum[3];
    std::vector<PxgParticleSystem> descArr(3);
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t N = NS[k];
        const float fk = float(k);
        deltas[k].resize(N); accum[k].resize(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            deltas[k][i] = { 0.1f * float(i + 1) + fk, -0.2f * float(i + 1), 0.05f - 0.01f * fk, 2.0f + float(i) - 0.5f * fk }; // .w = weight
            accum[k][i] = { float(i) + 0.5f * fk, -float(i), 0.5f * float(i), 3.0f + fk };
        }
        bDeltas[k] = makeDev(dev, deltas[k]);
        bAccum[k] = makeDev(dev, accum[k]);
        PxgParticleSystem ps = makeZeroDesc();
        ps.mDelta = reinterpret_cast<float4*>(size_t(bDeltas[k].addr));
        ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(bAccum[k].addr));
        ps.mCommonData.mNumParticles = N;
        ps.mData.mRelaxationFactor = RELAX[k];
        descArr[k] = ps;
    }
    DevBuf bDesc = makeDev(dev, descArr);
    const std::vector<uint32_t> active = { 2u, 0u };
    DevBuf bActive = makeDev(dev, active);

    struct ApplyPush { uint64_t systems, active; float dt; };
    ApplyPush push = { bDesc.addr, bActive.addr, 1.0f / 60.0f };
    {
        Kernel k("particleSelfCollisionBda.spv", "ps_applyDeltaLaunch", {}, sizeof(ApplyPush));
        k.dispatch(1u, {}, push, /*gridY*/ 2u, /*gridZ*/ 1u);
    }
    dev->flush();

    size_t mm = 0;
    for (uint32_t k = 0; k < 3; ++k)
    {
        const auto gAccum = downloadBuffer<F4>(dev, bAccum[k].h, NS[k]);
        const auto gDeltas = downloadBuffer<F4>(dev, bDeltas[k].h, NS[k]);
        for (uint32_t i = 0; i < NS[k]; ++i)
        {
            F4 wantA = accum[k][i], wantD = deltas[k][i];
            if (k != 1u)
            {
                const F4 j = deltas[k][i];
                const F4 a = accum[k][i];
                const float scale = 1.0f / std::fmax(j.w * RELAX[k], 1.0f);
                wantA = { a.x + j.x * scale, a.y + j.y * scale, a.z + j.z * scale, 0.0f };
                wantD = { 0.0f, 0.0f, 0.0f, 0.0f };
            }
            if (f2u(gAccum[i].x) != f2u(wantA.x) || f2u(gAccum[i].y) != f2u(wantA.y) ||
                f2u(gAccum[i].z) != f2u(wantA.z) || f2u(gAccum[i].w) != f2u(wantA.w))
            { if (mm < 8) std::printf("  sys%u accum MM %u\n", k, i); ++mm; }
            if (f2u(gDeltas[i].x) != f2u(wantD.x) || f2u(gDeltas[i].y) != f2u(wantD.y) ||
                f2u(gDeltas[i].z) != f2u(wantD.z) || f2u(gDeltas[i].w) != f2u(wantD.w))
            { if (mm < 8) std::printf("  sys%u delta MM %u\n", k, i); ++mm; }
        }
    }
    PXG_CHECK_MSG(mm == 0, "ps_applyDeltaLaunch, systems 2 and 0 via act {2,0} (relaxation 0.8/1.5), decoy "
                  "untouched: bit-exact vs CPU (%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k) { dev->destroyBuffer(bDeltas[k].h); dev->destroyBuffer(bAccum[k].h); }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// (5) ps_updateParticleLaunch — apply accumDeltaP + recompute sorted velocity.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionUpdateParticle)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 3;
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;

    std::vector<F4> accum(N), vel(N), deltaP(N), pos(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        accum[i] = { 0.01f * float(i + 1), 0.02f, -0.03f * float(i), 0.0f };
        vel[i] = { float(i), -0.5f * float(i), 0.25f, 0.0f };
        deltaP[i] = { 0.001f, 0.002f, 0.003f, 1.0f + float(i) };
        pos[i] = { 5.0f + float(i), 2.0f, -1.0f * float(i), 1.0f + 0.1f * float(i) };
    }

    DevBuf bAccum = makeDev(dev, accum);
    DevBuf bVel = makeDev(dev, vel);
    DevBuf bDeltaP = makeDev(dev, deltaP);
    DevBuf bPos = makeDev(dev, pos);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(bAccum.addr));
    ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(bVel.addr));
    ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(bDeltaP.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mCommonData.mNumParticles = N;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct UpdatePush { uint64_t systems, active; float invDt; uint32_t skip; };
    UpdatePush push = { bDesc.addr, bActive.addr, invDt, 0u };

    Kernel k("particleSelfCollisionBda.spv", "ps_updateParticleLaunch", {}, sizeof(UpdatePush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gAccum = downloadBuffer<F4>(dev, bAccum.h, N);
    const auto gDeltaP = downloadBuffer<F4>(dev, bDeltaP.h, N);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4 a = accum[i], v = vel[i], dp = deltaP[i], p = pos[i];
        const F4 nv = { v.x + a.x * invDt, v.y + a.y * invDt, v.z + a.z * invDt, v.w + a.w * invDt };
        const F4 np = { p.x + a.x, p.y + a.y, p.z + a.z, p.w }; // w passes through sortedPos.w
        const F4 ndp = { dp.x + a.x, dp.y + a.y, dp.z + a.z, dp.w + a.w };
        if (f2u(gVel[i].x) != f2u(nv.x) || f2u(gVel[i].y) != f2u(nv.y) ||
            f2u(gVel[i].z) != f2u(nv.z) || f2u(gVel[i].w) != f2u(nv.w))
        { ++mm; std::printf("  vel MM %u\n", i); }
        if (f2u(gPos[i].x) != f2u(np.x) || f2u(gPos[i].y) != f2u(np.y) ||
            f2u(gPos[i].z) != f2u(np.z) || f2u(gPos[i].w) != f2u(np.w))
        { ++mm; std::printf("  pos MM %u\n", i); }
        if (f2u(gDeltaP[i].x) != f2u(ndp.x) || f2u(gDeltaP[i].y) != f2u(ndp.y) ||
            f2u(gDeltaP[i].z) != f2u(ndp.z) || f2u(gDeltaP[i].w) != f2u(ndp.w))
        { ++mm; std::printf("  sortedDeltaP MM %u\n", i); }
        if (f2u(gAccum[i].x) != f2u(0.0f) || f2u(gAccum[i].y) != f2u(0.0f) ||
            f2u(gAccum[i].z) != f2u(0.0f) || f2u(gAccum[i].w) != f2u(0.0f))
        { ++mm; std::printf("  accum-clear MM %u\n", i); }
    }
    PXG_CHECK_MSG(mm == 0, "ps_updateParticleLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bAccum.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDeltaP.h);
    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// (5') P9.99b -- the TWO-SYSTEM twin of particleSelfCollisionUpdateParticle: act =
// {2, 0}, gridY = 2; systems 2 and 0 differ in N (4 / 3) and every array; the decoy
// at slot 1 (N 6) must survive bit for bit.  Oracle: particleSelfCollisionUpdateParticle's.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionUpdateParticle_p99TwoSystems)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NS[3] = { 3u, 6u, 4u };
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;
    std::vector<F4> accum[3], vel[3], deltaP[3], pos[3];
    DevBuf bAccum[3], bVel[3], bDeltaP[3], bPos[3];
    std::vector<PxgParticleSystem> descArr(3);
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t N = NS[k];
        const float fk = float(k);
        accum[k].resize(N); vel[k].resize(N); deltaP[k].resize(N); pos[k].resize(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            const float fi = float(i);
            accum[k][i] = { 0.01f * float(i + 1) + 0.002f * fk, 0.02f - 0.004f * fk, -0.03f * fi, 0.0f };
            vel[k][i] = { fi - fk, -0.5f * fi, 0.25f + fk, 0.0f };
            deltaP[k][i] = { 0.001f, 0.002f * (fk + 1.0f), 0.003f, 1.0f + fi };
            pos[k][i] = { 5.0f + fi + 10.0f * fk, 2.0f - fk, -1.0f * fi, 1.0f + 0.1f * fi };
        }
        bAccum[k] = makeDev(dev, accum[k]); bVel[k] = makeDev(dev, vel[k]);
        bDeltaP[k] = makeDev(dev, deltaP[k]); bPos[k] = makeDev(dev, pos[k]);
        PxgParticleSystem ps = makeZeroDesc();
        ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(bAccum[k].addr));
        ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(bVel[k].addr));
        ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(bDeltaP[k].addr));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos[k].addr));
        ps.mCommonData.mNumParticles = N;
        descArr[k] = ps;
    }
    DevBuf bDesc = makeDev(dev, descArr);
    const std::vector<uint32_t> active = { 2u, 0u };
    DevBuf bActive = makeDev(dev, active);

    struct UpdatePush { uint64_t systems, active; float invDt; uint32_t skip; };
    UpdatePush push = { bDesc.addr, bActive.addr, invDt, 0u };
    {
        Kernel k("particleSelfCollisionBda.spv", "ps_updateParticleLaunch", {}, sizeof(UpdatePush));
        k.dispatch(1u, {}, push, /*gridY*/ 2u, /*gridZ*/ 1u);
    }
    dev->flush();

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t k, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        { if (mm < 8) std::printf("  sys%u %s MM %u\n", k, what, i); ++mm; }
    };
    for (uint32_t k = 0; k < 3; ++k)
    {
        const auto gVel = downloadBuffer<F4>(dev, bVel[k].h, NS[k]);
        const auto gPos = downloadBuffer<F4>(dev, bPos[k].h, NS[k]);
        const auto gAccum = downloadBuffer<F4>(dev, bAccum[k].h, NS[k]);
        const auto gDeltaP = downloadBuffer<F4>(dev, bDeltaP[k].h, NS[k]);
        for (uint32_t i = 0; i < NS[k]; ++i)
        {
            const F4 a = accum[k][i], v = vel[k][i], dp = deltaP[k][i], p = pos[k][i];
            F4 nv = v, np = p, ndp = dp, na = a;
            if (k != 1u)
            {
                nv = { v.x + a.x * invDt, v.y + a.y * invDt, v.z + a.z * invDt, v.w + a.w * invDt };
                np = { p.x + a.x, p.y + a.y, p.z + a.z, p.w };   // w passes through sortedPos.w
                ndp = { dp.x + a.x, dp.y + a.y, dp.z + a.z, dp.w + a.w };
                na = { 0.0f, 0.0f, 0.0f, 0.0f };
            }
            cmp4(nv, gVel[i], "vel", k, i);
            cmp4(np, gPos[i], "pos", k, i);
            cmp4(ndp, gDeltaP[i], "sortedDeltaP", k, i);
            cmp4(na, gAccum[i], "accum", k, i);
        }
    }
    PXG_CHECK_MSG(mm == 0, "ps_updateParticleLaunch, systems 2 and 0 via act {2,0}, decoy untouched: bit-exact "
                  "vs CPU (%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k)
    { dev->destroyBuffer(bAccum[k].h); dev->destroyBuffer(bVel[k].h); dev->destroyBuffer(bDeltaP[k].h); dev->destroyBuffer(bPos[k].h); }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// (6) Full chain: selfCollision -> solveOneWay -> applyDelta -> updateParticle
//     over two OVERLAPPING particles, each given the one-way contact that
//     resolves the overlap.  Proves both detection and separation.
// ---------------------------------------------------------------------------
PXG_TEST(particleSelfCollisionSeparates)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 2;
    const float dt = 1.0f / 60.0f;
    const float bias = 1.0f;
    const float invMass = 1.0f;
    const float relaxation = 1.0f;
    const float cellWidth = 0.3f;
    const uint32_t gs = 4;
    const uint32_t numCells = gs * gs * gs;
    const uint32_t EMPTY = 0xffffffffu;
    const float solidRestOffset = 0.1f;      // 2*solidRestOffset = 0.2 (overlap threshold)

    // Two particles overlapping along x: 0.0 and 0.15 (< 0.2 apart).
    std::vector<F4> pos(N);
    pos[0] = { 0.00f, 0.5f, 0.5f, 1.0f };
    pos[1] = { 0.15f, 0.5f, 0.5f, 1.0f };
    std::vector<uint32_t> phase(N, PHASE_SELF_COLLIDE);
    std::vector<uint32_t> mapping = { 0u, 1u };

    std::vector<uint32_t> cellStart(numCells, EMPTY);
    std::vector<uint32_t> cellEnd(numCells, 0u);
    cellStart[0] = 0u; cellEnd[0] = 2u; // both in cell (0,0,0)

    const uint32_t maxNeighborhood = 8u;
    std::vector<uint32_t> selfCount(N, 0u);
    std::vector<uint32_t> collisionIndex(N * maxNeighborhood, 0xffffffffu);
    std::vector<F2> collisionForces(N * maxNeighborhood, { 9.0f, 9.0f });

    // One-way contact per particle, resolving the overlap: push #0 in -x, #1 in +x.
    // deltaLinVel = normalPen.xyz * deltaF * invMass * bias (friction 0), so set
    // normalPen.xyz toward the desired push direction.  penetration = -(overlap).
    const float overlap = 2.0f * solidRestOffset - 0.15f; // 0.05
    std::vector<F4> oneWayContacts(N);
    oneWayContacts[0] = { -1.0f, 0.0f, 0.0f, -overlap }; // push #0 toward -x
    oneWayContacts[1] = {  1.0f, 0.0f, 0.0f, -overlap }; // push #1 toward +x
    std::vector<uint32_t> oneWayCount = { 1u, 1u };
    std::vector<F2> oneWayForces(PxgParticleContactInfo::MaxStaticContactsPerParticle * N, { 0.0f, 0.0f });

    std::vector<F4> sortedDeltaP(N, { 0.0f, 0.0f, 0.0f, invMass });
    std::vector<F4> accumDeltaP(N, { 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<F4> sortedVel(N, { 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<F4> mDelta(N, { 0.0f, 0.0f, 0.0f, 0.0f });

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bPhase = makeDev(dev, phase);
    DevBuf bMap = makeDev(dev, mapping);
    DevBuf bStart = makeDev(dev, cellStart);
    DevBuf bEnd = makeDev(dev, cellEnd);
    DevBuf bCount = makeDev(dev, selfCount);
    DevBuf bColIdx = makeDev(dev, collisionIndex);
    DevBuf bColForce = makeDev(dev, collisionForces);
    DevBuf bContacts = makeDev(dev, oneWayContacts);
    DevBuf bOwCount = makeDev(dev, oneWayCount);
    DevBuf bOwForces = makeDev(dev, oneWayForces);
    DevBuf bSDeltaP = makeDev(dev, sortedDeltaP);
    DevBuf bAccum = makeDev(dev, accumDeltaP);
    DevBuf bVel = makeDev(dev, sortedVel);
    DevBuf bMDelta = makeDev(dev, mDelta);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bPhase.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bMap.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bEnd.addr));
    ps.mParticleSelfCollisionCount = reinterpret_cast<PxU32*>(size_t(bCount.addr));
    ps.mCollisionIndex = reinterpret_cast<PxU32*>(size_t(bColIdx.addr));
    ps.mDensityCollisionImpulses = reinterpret_cast<float2*>(size_t(bColForce.addr));
    ps.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(bContacts.addr));
    ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(bOwCount.addr));
    ps.mOneWayForces = reinterpret_cast<float2*>(size_t(bOwForces.addr));
    ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(bSDeltaP.addr));
    ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(bAccum.addr));
    ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(bVel.addr));
    ps.mDelta = reinterpret_cast<float4*>(size_t(bMDelta.addr));
    ps.mCommonData.mNumParticles = N;
    ps.mCommonData.mGridCellWidth = cellWidth;
    ps.mCommonData.mGridSizeX = gs;
    ps.mCommonData.mGridSizeY = gs;
    ps.mCommonData.mGridSizeZ = gs;
    ps.mCommonData.mMaxNeighborhood = maxNeighborhood;
    ps.mData.mSolidRestOffset = solidRestOffset;
    ps.mData.mRestOffset = 0.05f;
    ps.mData.mRelaxationFactor = relaxation;

    std::vector<DevBuf> keep;
    makeMaterialBufs(dev, ps, keep);

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct SelfPush { uint64_t systems; uint32_t id; };
    struct SolvePush { uint64_t systems, active; float invDt, dt, bias; uint32_t velIter; };
    struct ApplyPush { uint64_t systems, active; float dt; };
    struct UpdatePush { uint64_t systems, active; float invDt; uint32_t skip; };

    {
        SelfPush push = { bDesc.addr, 0u };
        Kernel k("particleSelfCollisionBda.spv", "ps_selfCollisionLaunch", {}, sizeof(SelfPush));
        k.dispatch(1u, {}, push, 1u, 1u);
        dev->flush();
    }
    {
        SolvePush push = { bDesc.addr, bActive.addr, 1.0f / dt, dt, bias, 0u };
        Kernel k("particleSelfCollisionBda.spv", "ps_solveOneWayContactDeltaVLaunch", {}, sizeof(SolvePush));
        k.dispatch(1u, {}, push, 1u, 1u);
        dev->flush();
    }
    {
        ApplyPush push = { bDesc.addr, bActive.addr, dt };
        Kernel k("particleSelfCollisionBda.spv", "ps_applyDeltaLaunch", {}, sizeof(ApplyPush));
        k.dispatch(1u, {}, push, 1u, 1u);
        dev->flush();
    }
    {
        UpdatePush push = { bDesc.addr, bActive.addr, 1.0f / dt, 0u };
        Kernel k("particleSelfCollisionBda.spv", "ps_updateParticleLaunch", {}, sizeof(UpdatePush));
        k.dispatch(1u, {}, push, 1u, 1u);
        dev->flush();
    }

    // CPU reference for the chain.
    // (a) selfCollision: each detects the other.
    // (b) solveOneWay -> accumDeltaP.
    float acc0[3], acc1[3];
    cpuSolveOneWay(&oneWayContacts[0].x, &sortedDeltaP[0].x, bias, dt, false, acc0);
    cpuSolveOneWay(&oneWayContacts[1].x, &sortedDeltaP[1].x, bias, dt, false, acc1);
    // (c) applyDelta with mDelta==0 keeps accumDeltaP.xyz (w -> 0).
    // (d) updateParticle: pos += accumDeltaP.
    const F4 refPos0 = { pos[0].x + acc0[0], pos[0].y + acc0[1], pos[0].z + acc0[2], pos[0].w };
    const F4 refPos1 = { pos[1].x + acc1[0], pos[1].y + acc1[1], pos[1].z + acc1[2], pos[1].w };

    const auto gCount = downloadBuffer<uint32_t>(dev, bCount.h, N);
    const auto gIdx = downloadBuffer<uint32_t>(dev, bColIdx.h, N * maxNeighborhood);
    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);

    size_t mm = 0;
    // detection
    PXG_CHECK_MSG(gCount[0] == 1u && gCount[1] == 1u, "both particles detect one neighbour (%u,%u)",
                  gCount[0], gCount[1]);
    PXG_CHECK_MSG(gIdx[0] == 1u && gIdx[1] == 0u, "collisionIndex records the overlap (%u,%u)",
                  gIdx[0], gIdx[1]);
    // bit-exact final positions
    if (f2u(gPos[0].x) != f2u(refPos0.x) || f2u(gPos[0].y) != f2u(refPos0.y) ||
        f2u(gPos[0].z) != f2u(refPos0.z) || f2u(gPos[0].w) != f2u(refPos0.w)) { ++mm; std::printf("  pos0 MM\n"); }
    if (f2u(gPos[1].x) != f2u(refPos1.x) || f2u(gPos[1].y) != f2u(refPos1.y) ||
        f2u(gPos[1].z) != f2u(refPos1.z) || f2u(gPos[1].w) != f2u(refPos1.w)) { ++mm; std::printf("  pos1 MM\n"); }
    PXG_CHECK_MSG(mm == 0, "self-collision chain final positions bit-exact vs CPU (%zu mismatches)", mm);

    // separation: distance increased past the overlap threshold.
    const float d0 = pos[1].x - pos[0].x;                 // 0.15 (overlapping)
    const float d1 = gPos[1].x - gPos[0].x;               // separated
    PXG_CHECK_MSG(d1 > d0, "particles moved apart (%g -> %g)", d0, d1);
    PXG_CHECK_MSG(d1 >= 2.0f * solidRestOffset, "separation reaches 2*solidRestOffset (%g >= %g)",
                  d1, 2.0f * solidRestOffset);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bPhase.h); dev->destroyBuffer(bMap.h);
    dev->destroyBuffer(bStart.h); dev->destroyBuffer(bEnd.h); dev->destroyBuffer(bCount.h);
    dev->destroyBuffer(bColIdx.h); dev->destroyBuffer(bColForce.h); dev->destroyBuffer(bContacts.h);
    dev->destroyBuffer(bOwCount.h); dev->destroyBuffer(bOwForces.h); dev->destroyBuffer(bSDeltaP.h);
    dev->destroyBuffer(bAccum.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bMDelta.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
    for (auto& b : keep) dev->destroyBuffer(b.h);
}

}  // namespace
