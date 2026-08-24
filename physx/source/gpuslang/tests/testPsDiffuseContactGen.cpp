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

// testPsDiffuseContactGen.cpp -- P9.13g, the fixture for
// ps_primitivesDiffuseCollisionLaunch (psContactGenBda.slang's second entry
// point), driven through the REAL 14-argument engine ABI blob.
//
// ---------------------------------------------------------------------------
// THE ARRANGEMENT
// ---------------------------------------------------------------------------
// The contact core (particlePrimitiveCollision and the quat/box/sphere/capsule
// helpers) is SHARED with ps_primitivesCollisionLaunch, which testPsContactGen
// already pins bit-exactly against a CPU transcription of the ePLANE branch.
// This fixture therefore does not re-verify that core.  What it verifies is
// everything the DIFFUSE entry point does differently -- D1..D8 in the shader
// header -- plus the per-particle output layout P9.13f's consumer depends on.
//
// The oracle is a host mirror of the CUDA's control flow over the same ePLANE
// arithmetic, compared BIT-EXACTLY (the tests build with -ffp-contract=off,
// matching the shader's NoContraction).
//
// ---------------------------------------------------------------------------
// WHAT THE GEOMETRY IS BUILT TO DISCRIMINATE
// ---------------------------------------------------------------------------
//   * FOURTEEN planes at different heights against ONE particle system, so a
//     single diffuse particle accumulates FOURTEEN contacts and the twelve-slot
//     cap actually bites (D7).  With one plane it never could.
//   * the FLUID cell range of the same system covers a DIFFERENT particle span
//     (0..24) than the diffuse one (3..17), so a port that reads mCellStart /
//     mCellEnd instead of mDiffuseCellStart / mDiffuseCellEnd processes a
//     different set of particles (D1).
//   * speculative CCD ON with origin != predicted, and a per-particle offset
//     that is distinct and non-zero, so swapping the two arrays (D2) or
//     dropping isDiffuse (D3) changes the contact volume.
//   * BOTH the PGS and the TGS dispatch, whose results must be IDENTICAL.  That
//     is the whole point of D3: PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE
//     is 1.0, so a diffuse particle is NOT halved in a TGS step the way a fluid
//     particle is.  A port that passes isTGS through makes the two runs differ.
//   * a SECOND particle system with mMaxDiffuseParticles == 0 but a fully
//     populated diffuse cell range and particles placed to contact -- so the D5
//     early-out is the only thing keeping its outputs poisoned.
//   * a RIGID FILTER PAIR seeded to match one (particle, plane) combination.
//     The diffuse path has no filter (D4); a port that borrows the fluid
//     kernel's findPs suppresses exactly that one contact.
//   * per-pair rest distances that are all DIFFERENT, so a wrong pair index
//     lands on a different penetration.
//   * output arrays running SIXTEEN planes deep where twelve are legal, plus a
//     poisoned margin, so an over-cap write is visible rather than silent.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgSimulationCoreDesc.h"
#include "PxsTransformCache.h"
#include "PxNodeIndex.h"
#include "geometry/PxGeometry.h"

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

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

// ---------------------------------------------------------------------------
// The ABI, pinned from the engine headers.
// ---------------------------------------------------------------------------
static_assert(sizeof(PxgParticleSystem) == 656, "PxgParticleSystem stride");
static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput stride");
static_assert(sizeof(PxgShape) == 48, "PxgShape stride");
static_assert(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform stride");
static_assert(sizeof(PxgRigidFilterPair) == 32, "PxgRigidFilterPair stride");
static_assert(sizeof(PxgParticleContactInfo) == 16, "PxgParticleContactInfo is one float4");
static_assert(PxgParticleContactInfo::MaxStaticContactsPerParticle == 12, "the static-contact cap");
static_assert(offsetof(PxgParticleSystem, mCommonData) == 88, "mCommonData @88");
static_assert(offsetof(PxgParticleSystem, mDiffuseCellStart) == 520, "@520");
static_assert(offsetof(PxgParticleSystem, mDiffuseCellEnd) == 528, "@528");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) == 552, "@552");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) == 560, "@560");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactInfos) == 592, "@592");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactCount) == 616, "@616");

// The 104-byte engine ABI blob: kPsPrimCollide's head with no writer.
struct PsDiffCollideBlob
{
    uint32_t isTGSWord;         // @0
    uint32_t numTests;          // @4
    float toleranceLength;      // @8
    uint32_t pad12;             // @12
    uint64_t cmInputs;          // @16
    uint64_t shapes;            // @24
    uint64_t transformCache;    // @32
    uint64_t bounds;            // @40
    uint64_t contactDistance;   // @48
    uint64_t restDistances;     // @56
    uint64_t materials;         // @64
    uint64_t startIndices;      // @72
    uint64_t totalNumPairs;     // @80
    uint64_t particleSystems;   // @88
    uint64_t shapeToRigidRemap; // @96 -> 104
};
static_assert(sizeof(PsDiffCollideBlob) == 104, "engine ABI blob size");

// THIRTY planes, not fourteen, and the reason is a mutant that fourteen could
// not kill.  `d7-atomic-capped-too` replaces the unconditional atomicAdd with a
// non-atomic read plus a guarded add -- the counter would then STOP at twelve
// instead of overshooting, which is precisely what D7 forbids.  With fourteen
// contacting work items and llvmpipe's subgroup of 8, the lanes read 0 and then
// 8, both below twelve, so every lane still incremented and the mutant was
// indistinguishable.  With thirty the third subgroup reads 16, the guard bites,
// and the count stops at 16 instead of 30.  Any serialisation at all now shows
// it; only 19+ lanes reading simultaneously could hide it, which a subgroup of
// 8 cannot do.
static const uint32_t NB_PLANES = 30u;      // > 12, so the cap bites -- see above
// The pair table is deliberately WIDER than the work the kernel may do:
//   pairs 0..13  main system vs plane j -- and pair 0 spans TWO cells, so the
//                per-pair run-sum is NOT the identity and `pos` differs from
//                `workIndex` (without that, restDistances[pos] and
//                restDistances[workIndex] are the same expression);
//   pair 14      the mMaxDiffuseParticles == 0 trap system;
//   pair 15      main vs plane 0 again, EXCLUDED by totalNumPairs -- a valid,
//                in-bounds pair that would produce contacts, so a port whose
//                work-item guard is `>` instead of `>=` reaches it and is caught
//                by real data rather than by an out-of-bounds read.
static const uint32_t NB_PAIRS_TABLE = NB_PLANES + 2u;   // 16 rows
static const uint32_t TOTAL_WORK = NB_PLANES + 2u;       // 16 items: pair 0 twice
static const uint32_t EXCLUDED_PAIR = NB_PLANES + 1u;    // 15
static const uint32_t MAXC = 12u;
static const uint32_t ALLOC_C = 16u;        // four planes past the legal twelve
static const uint32_t SLACK = 137u;
static const uint32_t NUM_DIFFUSE = 24u;    // published count for the main system
static const uint32_t D_START = 3u;         // diffuse cell 0 range
static const uint32_t D_END = 17u;          // -> 14 live particles
static const uint32_t TRAP_DIFFUSE = 8u;    // the mMaxDiffuseParticles == 0 system
static const float CONTACT_OFFSET = 0.3f;
static const float POISON_F = -7.5e3f;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;
static const uint32_t FILTERED_PARTICLE = 5u;   // matched by the seeded filter pair
static const uint32_t FILTERED_PLANE = 0u;
static const uint32_t NODE_BASE = 100u;

// Plane j sits at y = PLANE_Y0 + PLANE_DY * j, normal +Y.
static float planeHeight(uint32_t j) { return -0.5f + 0.05f * float(j); }
// Diffuse particle i's ORIGIN and PREDICTED heights.  The offsets are distinct
// and non-zero so the CCD midpoint is neither endpoint.
static float originY(uint32_t i) { return -0.45f + 0.10f * float(i - D_START); }
// THE SIGN OF THE CCD OFFSET IS WHAT MAKES CCD OBSERVABLE AT ALL.
// The plane test is `dot(n, cVolumePos) + pdd <= cVolumeRadius`, i.e.
// `oy + off - h <= |off| + cDist`.  For off > 0 the two `off` terms CANCEL and
// the condition collapses to `oy - h <= cDist` -- exactly the no-CCD condition.
// A fixture whose particles all move UP therefore cannot see the contact volume
// at all: the first sweep of this fixture had every offset positive and lost
// five mutants to it (both position-array swaps, both isDiffuse selectors and
// the scale constant).  Every third particle now moves DOWN, where the terms add
// instead and the threshold genuinely widens.
static float predY(uint32_t i)
{
    const float d = ((i % 3u) == 0u) ? (-0.021f - 0.005f * float(i))
                                     : (0.017f + 0.004f * float(i));
    return originY(i) + d;
}
static float restDistanceFor(uint32_t j) { return 0.001f * float(j + 1u); }

// The shader's quatRotate (== PxQuat::rotate), transcribed once.
static void cpuQuatRotate(const float q[4], const float v[3], float out[3])
{
    const float vx = 2.0f * v[0], vy = 2.0f * v[1], vz = 2.0f * v[2];
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    out[0] = (vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2);
    out[1] = (vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2);
    out[2] = (vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2);
}

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bDesc, bCin, bShapes, bTc, bBounds, bCd, bRd, bMat, bStart, bTotal, bRemap;
    DevBuf bDOrigin, bDPred, bDCellStart, bDCellEnd, bCellStart, bCellEnd, bTwo;
    DevBuf bInfos, bForces, bNodes, bCounts;
    DevBuf bRunsum, bSimBuf, bFilterPairs;
    DevBuf tOrigin, tPred, tCellStart, tCellEnd, tTwo, tInfos, tForces, tNodes, tCounts;
    PsDiffCollideBlob blob{};

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

static uint32_t infoSlots(uint32_t n) { return n * ALLOC_C + SLACK; }

// The plane rotation: PxPlane(1,0,0,0)'s +X normal onto +Y.
static void planeQuat(float q[4])
{
    const float s = std::sin(0.78539816339744830961f);
    const float c = std::cos(0.78539816339744830961f);
    q[0] = 0.0f; q[1] = 0.0f; q[2] = s; q[3] = c;
}

// ---------------------------------------------------------------------------
// The host mirror.  One entry per (particle, plane) contact the kernel must
// produce.  isDiffuse is TRUE, so scaleInv is 1.0 in BOTH memory modes.
// ---------------------------------------------------------------------------
struct Expected { uint32_t particle; uint32_t plane; F4 normalPen; uint64_t node; };

static std::vector<Expected> predict()
{
    float q[4]; planeQuat(q);
    float pn[3]; cpuQuatRotate(q, (const float[3]){ 1.0f, 0.0f, 0.0f }, pn);

    std::vector<Expected> out;
    for (uint32_t i = D_START; i < D_END; ++i)
    {
        const float oy = originY(i), py = predY(i);
        // getParticleSpeculativeContactVolume with isDiffuse = true:
        // scaleInv = 1/PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0.
        const float offY = (py - oy) * 1.0f * 0.5f;
        const float cvY = oy + offY;
        const float cvR = std::fabs(offY) + CONTACT_OFFSET;   // length of (0,offY,0)

        for (uint32_t j = 0; j < NB_PLANES; ++j)
        {
            const float ph = planeHeight(j);
            const float pdd = -(pn[0] * 0.0f + pn[1] * ph + pn[2] * 0.0f);
            float dist = pn[0] * 0.0f + pn[1] * cvY + pn[2] * 0.0f + pdd;
            if (dist > cvR) continue;
            // enableCCD -> dist recomputed from the ORIGIN position
            dist = pn[0] * 0.0f + pn[1] * oy + pn[2] * 0.0f + pdd;
            Expected e;
            e.particle = i;
            e.plane = j;
            e.normalPen = F4{ pn[0], pn[1], pn[2], dist - restDistanceFor(j) };
            e.node = uint64_t(NODE_BASE + j);
            // Pair 0 owns TWO work items (two cells over a 1x1x1 grid, both
            // resolving to hash 0), so its contact is produced TWICE.  That is
            // what makes the run-sum non-trivial; the expectation follows it.
            out.push_back(e);
            if (j == 0u) out.push_back(e);
        }
    }
    return out;
}

static void buildRig(PxgSlangComputeDevice* dev, Rig& R)
{
    float q[4]; planeQuat(q);

    // ---- the MAIN system's diffuse arrays ----
    std::vector<F4> dOrigin(NUM_DIFFUSE + SLACK, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    std::vector<F4> dPred(NUM_DIFFUSE + SLACK, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    for (uint32_t i = 0; i < NUM_DIFFUSE; ++i)
    {
        // Particles outside [D_START, D_END) exist only so the FLUID cell range
        // reaches them: they sit far below every plane, so a port that walks the
        // wrong range produces contacts the mirror does not predict.
        const bool live = (i >= D_START && i < D_END);
        const float oy = live ? originY(i) : (-40.0f - float(i));
        const float py = live ? predY(i) : (-40.0f - float(i));
        dOrigin[i] = F4{ 0.0f, oy, 0.0f, 1.0f };
        dPred[i] = F4{ 0.0f, py, 0.0f, 1.0f };
    }
    R.bDOrigin = R.mk(dev, dOrigin);
    R.bDPred = R.mk(dev, dPred);

    // D1: the two grids of the SAME system cover different spans.
    R.bDCellStart = R.mk(dev, std::vector<uint32_t>{ D_START });
    R.bDCellEnd = R.mk(dev, std::vector<uint32_t>{ D_END });
    R.bCellStart = R.mk(dev, std::vector<uint32_t>{ 0u });
    R.bCellEnd = R.mk(dev, std::vector<uint32_t>{ NUM_DIFFUSE });

    // D8: the two-slot count pair; [1] is a trap, larger than [0].
    R.bTwo = R.mk(dev, std::vector<int32_t>{ int32_t(NUM_DIFFUSE), int32_t(NUM_DIFFUSE) * 3 });

    R.bInfos = R.mk(dev, std::vector<F4>(infoSlots(NUM_DIFFUSE),
                                         F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    R.bForces = R.mk(dev, std::vector<float>(infoSlots(NUM_DIFFUSE), POISON_F));
    R.bNodes = R.mk(dev, std::vector<uint64_t>(infoSlots(NUM_DIFFUSE), 0xDEADBEEFCAFEF00Dull));
    std::vector<uint32_t> counts(NUM_DIFFUSE + SLACK, POISON_U32);
    for (uint32_t i = 0; i < NUM_DIFFUSE; ++i) counts[i] = 0u;
    R.bCounts = R.mk(dev, counts);

    // D4: the filter machinery, seeded so a borrowed findPs would bite.
    R.bRunsum = R.mk(dev, std::vector<uint32_t>{ 0u });
    std::vector<PxgRigidFilterPair> pairs(1);
    std::memset(pairs.data(), 0, sizeof(PxgRigidFilterPair));
    pairs[0].index0 = uint64_t(NODE_BASE + FILTERED_PLANE);
    pairs[0].index1 = uint64_t(FILTERED_PARTICLE);
    pairs[0].index2 = 0u;
    R.bFilterPairs = R.mk(dev, pairs);
    std::vector<PxgParticleSimBuffer> simBufs(1);
    std::memset(simBufs.data(), 0, sizeof(PxgParticleSimBuffer));
    simBufs[0].mFilterPairs = reinterpret_cast<PxParticleRigidFilterPair*>(size_t(R.bFilterPairs.addr));
    simBufs[0].mNumFilterPairs = 1u;
    R.bSimBuf = R.mk(dev, simBufs);

    // ---- the TRAP system: mMaxDiffuseParticles == 0, everything else live ----
    std::vector<F4> tO(TRAP_DIFFUSE + SLACK, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    std::vector<F4> tP = tO;
    for (uint32_t i = 0; i < TRAP_DIFFUSE; ++i)
    {
        tO[i] = F4{ 0.0f, -0.45f, 0.0f, 1.0f };   // deep inside every plane
        tP[i] = F4{ 0.0f, -0.44f, 0.0f, 1.0f };
    }
    R.tOrigin = R.mk(dev, tO);
    R.tPred = R.mk(dev, tP);
    R.tCellStart = R.mk(dev, std::vector<uint32_t>{ 0u });
    R.tCellEnd = R.mk(dev, std::vector<uint32_t>{ TRAP_DIFFUSE });
    R.tTwo = R.mk(dev, std::vector<int32_t>{ int32_t(TRAP_DIFFUSE), 0 });
    R.tInfos = R.mk(dev, std::vector<F4>(infoSlots(TRAP_DIFFUSE),
                                         F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    R.tForces = R.mk(dev, std::vector<float>(infoSlots(TRAP_DIFFUSE), POISON_F));
    R.tNodes = R.mk(dev, std::vector<uint64_t>(infoSlots(TRAP_DIFFUSE), 0xDEADBEEFCAFEF00Dull));
    R.tCounts = R.mk(dev, std::vector<uint32_t>(TRAP_DIFFUSE + SLACK, POISON_U32));

    // ---- the two descriptors ----
    std::vector<PxgParticleSystem> descArr(2);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));

    PxgParticleSystem& m = descArr[0];
    m.mCommonData.mGridCellWidth = 1.0f;
    m.mCommonData.mGridSizeX = 1u;   // 1x1x1 -> gridHash is always 0
    m.mCommonData.mGridSizeY = 1u;
    m.mCommonData.mGridSizeZ = 1u;
    m.mCommonData.mNumParticles = NUM_DIFFUSE;
    m.mCommonData.mMaxDiffuseParticles = 4096u;
    m.mCommonData.mNumParticleBuffers = 1u;
    m.mData.mFlags = 8u;             // PxParticleFlag::eENABLE_SPECULATIVE_CCD
    m.mCellStart = reinterpret_cast<PxU32*>(size_t(R.bCellStart.addr));
    m.mCellEnd = reinterpret_cast<PxU32*>(size_t(R.bCellEnd.addr));
    m.mDiffuseCellStart = reinterpret_cast<PxU32*>(size_t(R.bDCellStart.addr));
    m.mDiffuseCellEnd = reinterpret_cast<PxU32*>(size_t(R.bDCellEnd.addr));
    m.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDOrigin.addr));
    m.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDPred.addr));
    m.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bTwo.addr));
    m.mDiffuseOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.bInfos.addr));
    m.mDiffuseOneWayForces = reinterpret_cast<float*>(size_t(R.bForces.addr));
    m.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.bNodes.addr));
    m.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.bCounts.addr));
    m.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(R.bRunsum.addr));
    m.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(R.bSimBuf.addr));

    PxgParticleSystem& t = descArr[1];
    t = m;                                   // same grid, same flags ...
    t.mCommonData.mMaxDiffuseParticles = 0u; // ... and the ONE difference (D5)
    t.mCellStart = reinterpret_cast<PxU32*>(size_t(R.tCellStart.addr));
    t.mCellEnd = reinterpret_cast<PxU32*>(size_t(R.tCellEnd.addr));
    t.mDiffuseCellStart = reinterpret_cast<PxU32*>(size_t(R.tCellStart.addr));
    t.mDiffuseCellEnd = reinterpret_cast<PxU32*>(size_t(R.tCellEnd.addr));
    t.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.tOrigin.addr));
    t.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(R.tPred.addr));
    t.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.tTwo.addr));
    t.mDiffuseOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.tInfos.addr));
    t.mDiffuseOneWayForces = reinterpret_cast<float*>(size_t(R.tForces.addr));
    t.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.tNodes.addr));
    t.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.tCounts.addr));
    R.bDesc = R.mk(dev, descArr);

    // ---- narrowphase descriptors ----
    // shapes: 0 = main particle system, 1 = trap system, 2.. = the planes.
    const uint32_t NB_SHAPES = 2u + NB_PLANES;
    std::vector<PxgShape> shapes(NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[1].particleOrSoftbodyId = 1u;
    for (uint32_t j = 0; j < NB_PLANES; ++j)
        shapes[2u + j].type = PxU32(PxGeometryType::ePLANE);
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> tcache(NB_SHAPES);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    for (uint32_t j = 0; j < NB_PLANES; ++j)
        tcache[2u + j].transform = PxTransform(PxVec3(0.0f, planeHeight(j), 0.0f),
                                               PxQuat(q[0], q[1], q[2], q[3]));
    R.bTc = R.mk(dev, tcache);

    // See NB_PAIRS_TABLE above for what each row is for.
    std::vector<PxgContactManagerInput> cin(NB_PAIRS_TABLE);
    std::memset(cin.data(), 0, cin.size() * sizeof(PxgContactManagerInput));
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        cin[j].shapeRef0 = 0u;  cin[j].transformCacheRef0 = 0u;
        cin[j].shapeRef1 = 2u + j;  cin[j].transformCacheRef1 = 2u + j;
    }
    cin[NB_PLANES].shapeRef0 = 1u;  cin[NB_PLANES].transformCacheRef0 = 1u;
    cin[NB_PLANES].shapeRef1 = 2u;  cin[NB_PLANES].transformCacheRef1 = 2u;
    // The excluded row: main system vs plane 0 again, reachable only by a port
    // whose work-item guard lets workIndex == totalNumPairs through.
    cin[EXCLUDED_PAIR].shapeRef0 = 0u;  cin[EXCLUDED_PAIR].transformCacheRef0 = 0u;
    cin[EXCLUDED_PAIR].shapeRef1 = 2u;  cin[EXCLUDED_PAIR].transformCacheRef1 = 2u;
    R.bCin = R.mk(dev, cin);

    std::vector<float> boundsF(size_t(NB_SHAPES) * 6u, 0.0f);
    for (uint32_t s = 0; s < NB_SHAPES; ++s)
        for (int k = 0; k < 3; ++k)
        {
            boundsF[size_t(s) * 6 + k] = -8.0f;
            boundsF[size_t(s) * 6 + 3 + k] = 8.0f;
        }
    R.bBounds = R.mk(dev, boundsF);

    // cDistance = contactDistance[particleCacheRef] + contactDistance[rigidCacheRef]
    std::vector<float> cdist(NB_SHAPES, 0.0f);
    cdist[0] = CONTACT_OFFSET;
    cdist[1] = CONTACT_OFFSET;
    R.bCd = R.mk(dev, cdist);

    std::vector<float> rdist(NB_PAIRS_TABLE);
    for (uint32_t j = 0; j < NB_PAIRS_TABLE; ++j) rdist[j] = restDistanceFor(j);
    R.bRd = R.mk(dev, rdist);

    R.bMat = R.mk(dev, std::vector<uint32_t>(4u, POISON_U32));   // ABI parity only

    // The per-pair cell run-sum.  Pair 0 owns TWO cells, everything after it one,
    // so pos != workIndex from work item 1 onward -- restDistances[pos] and
    // restDistances[workIndex] are then different expressions.
    std::vector<uint32_t> startIdx(NB_PAIRS_TABLE);
    startIdx[0] = 0u;
    for (uint32_t j = 1; j < NB_PAIRS_TABLE; ++j) startIdx[j] = j + 1u;
    R.bStart = R.mk(dev, startIdx);
    // TOTAL_WORK covers pairs 0..NB_PLANES only; the excluded row's work item
    // sits at index TOTAL_WORK and must never run.
    R.bTotal = R.mk(dev, std::vector<uint32_t>{ TOTAL_WORK });

    std::vector<uint64_t> remap(NB_SHAPES, 0ull);
    for (uint32_t j = 0; j < NB_PLANES; ++j) remap[2u + j] = uint64_t(NODE_BASE + j);
    R.bRemap = R.mk(dev, remap);

    std::memset(&R.blob, 0, sizeof(R.blob));
    R.blob.numTests = NB_PAIRS_TABLE;
    R.blob.toleranceLength = 1.0f;
    R.blob.cmInputs = R.bCin.addr;
    R.blob.shapes = R.bShapes.addr;
    R.blob.transformCache = R.bTc.addr;
    R.blob.bounds = R.bBounds.addr;
    R.blob.contactDistance = R.bCd.addr;
    R.blob.restDistances = R.bRd.addr;
    R.blob.materials = R.bMat.addr;
    R.blob.startIndices = R.bStart.addr;
    R.blob.totalNumPairs = R.bTotal.addr;
    R.blob.particleSystems = R.bDesc.addr;
    R.blob.shapeToRigidRemap = R.bRemap.addr;
}

// The framework exposes createBufferFrom/downloadBuffer but no typed re-upload;
// dev->upload is the primitive underneath both.
template <typename T>
static void reupload(PxgSlangComputeDevice* dev, PxgBufferHandle h, const std::vector<T>& d)
{
    dev->upload(h, d.data(), d.size() * sizeof(T));
}


static void checkMain(PxgSlangComputeDevice* dev, const Rig& R, const char* label)
{
    const std::vector<Expected> E = predict();

    const auto infos = downloadBuffer<F4>(dev, R.bInfos.h, infoSlots(NUM_DIFFUSE));
    const auto forces = downloadBuffer<float>(dev, R.bForces.h, infoSlots(NUM_DIFFUSE));
    const auto nodes = downloadBuffer<uint64_t>(dev, R.bNodes.h, infoSlots(NUM_DIFFUSE));
    const auto counts = downloadBuffer<uint32_t>(dev, R.bCounts.h, NUM_DIFFUSE + SLACK);

    size_t badCount = 0, badSlot = 0, badPoison = 0, capped = 0, deepest = 0;
    for (uint32_t i = 0; i < NUM_DIFFUSE; ++i)
    {
        std::vector<Expected> want;
        for (const Expected& e : E) if (e.particle == i) want.push_back(e);

        if (counts[i] != uint32_t(want.size()))
        {
            if (badCount < 4)
                std::printf("    %s particle %u: count %u, expected %zu\n",
                            label, i, counts[i], want.size());
            ++badCount;
        }
        deepest = std::max(deepest, want.size());
        const uint32_t written = uint32_t(std::min<size_t>(want.size(), MAXC));
        if (want.size() > MAXC) ++capped;

        // WHICH contact landed in WHICH slot is a race (the atomicAdd order), so
        // the written entries are compared as a SET against the expected ones and
        // required to be pairwise distinct -- never index by index.
        // Claim by INDEX into `want`, not by plane: pair 0 legitimately appears
        // twice (its two cells), so a set keyed on the plane would reject the
        // second copy.
        std::vector<bool> claimed(want.size(), false);
        for (uint32_t c = 0; c < written; ++c)
        {
            const size_t at = size_t(i) + size_t(c) * NUM_DIFFUSE;
            bool matched = false;
            for (size_t w = 0; w < want.size(); ++w)
            {
                const Expected& e = want[w];
                if (claimed[w]) continue;
                if (fbits(infos[at].x) == fbits(e.normalPen.x) &&
                    fbits(infos[at].y) == fbits(e.normalPen.y) &&
                    fbits(infos[at].z) == fbits(e.normalPen.z) &&
                    fbits(infos[at].w) == fbits(e.normalPen.w) &&
                    nodes[at] == e.node && forces[at] == 0.0f)
                {
                    matched = true;
                    claimed[w] = true;
                    break;
                }
            }
            if (!matched)
            {
                if (badSlot < 4)
                    std::printf("    %s particle %u slot %u: (%g,%g,%g,%g) node %llu force %g "
                                "matches no unclaimed expected contact\n", label, i, c,
                                infos[at].x, infos[at].y, infos[at].z, infos[at].w,
                                (unsigned long long)nodes[at], forces[at]);
                ++badSlot;
            }
        }

        // Everything past the written slots must still be poison, including the
        // four planes' worth allocated beyond the twelve-slot cap.
        for (uint32_t c = written; c < ALLOC_C; ++c)
        {
            const size_t at = size_t(i) + size_t(c) * NUM_DIFFUSE;
            if (fbits(infos[at].x) != fbits(POISON_F) || nodes[at] != 0xDEADBEEFCAFEF00Dull ||
                fbits(forces[at]) != fbits(POISON_F))
                ++badPoison;
        }
    }
    for (uint32_t i = NUM_DIFFUSE; i < NUM_DIFFUSE + SLACK; ++i)
        if (counts[i] != POISON_U32) ++badPoison;

    PXG_CHECK_MSG(badCount == 0, "%s: %zu per-particle contact counts wrong", label, badCount);
    PXG_CHECK_MSG(badSlot == 0, "%s: %zu written contacts matched no expected contact", label, badSlot);
    PXG_CHECK_MSG(badPoison == 0, "%s: %zu slots past the written contacts were not poison "
                  "(the four planes' worth past the cap and the %u-slot count margin are "
                  "compared too)", label, badPoison, SLACK);
    PXG_CHECK_MSG(capped > 0 && deepest > MAXC,
                  "%s: no particle exceeded the %u-slot cap (deepest %zu) -- D7 is dead",
                  label, MAXC, deepest);
}

static void checkTrap(PxgSlangComputeDevice* dev, const Rig& R, const char* label)
{
    const auto infos = downloadBuffer<F4>(dev, R.tInfos.h, infoSlots(TRAP_DIFFUSE));
    const auto nodes = downloadBuffer<uint64_t>(dev, R.tNodes.h, infoSlots(TRAP_DIFFUSE));
    const auto counts = downloadBuffer<uint32_t>(dev, R.tCounts.h, TRAP_DIFFUSE + SLACK);
    size_t bad = 0;
    for (uint32_t i = 0; i < infoSlots(TRAP_DIFFUSE); ++i)
    {
        if (fbits(infos[i].x) != fbits(POISON_F)) ++bad;
        if (nodes[i] != 0xDEADBEEFCAFEF00Dull) ++bad;
    }
    for (uint32_t i = 0; i < TRAP_DIFFUSE + SLACK; ++i)
        if (counts[i] != POISON_U32) ++bad;
    PXG_CHECK_MSG(bad == 0, "%s: the mMaxDiffuseParticles == 0 system produced %zu writes -- "
                  "its cell range and particles are live, so only the D5 early-out keeps it "
                  "silent", label, bad);
}

static void resetOutputs(PxgSlangComputeDevice* dev, Rig& R)
{
    reupload(dev, R.bInfos.h, std::vector<F4>(infoSlots(NUM_DIFFUSE),
                                                  F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    reupload(dev, R.bForces.h, std::vector<float>(infoSlots(NUM_DIFFUSE), POISON_F));
    reupload(dev, R.bNodes.h, std::vector<uint64_t>(infoSlots(NUM_DIFFUSE),
                                                        0xDEADBEEFCAFEF00Dull));
    std::vector<uint32_t> counts(NUM_DIFFUSE + SLACK, POISON_U32);
    for (uint32_t i = 0; i < NUM_DIFFUSE; ++i) counts[i] = 0u;
    reupload(dev, R.bCounts.h, counts);
}

static void runAll(PxgSlangComputeDevice* dev)
{
    Rig R;
    buildRig(dev, R);

    Kernel k("psContactGenBda.spv", "ps_primitivesDiffuseCollisionLaunch", {},
             sizeof(PsDiffCollideBlob));
    PXG_CHECK_MSG(k.isValid(), "the diffuse contact-gen pipeline must build");
    if (!k.isValid()) { R.destroy(dev); return; }

    // D3: PGS and TGS must agree, because isDiffuse wins over isTGS.
    for (int mode = 0; mode < 2; ++mode)
    {
        const char* label = mode ? "TGS" : "PGS";
        resetOutputs(dev, R);
        dev->flush();
        R.blob.isTGSWord = mode ? 1u : 0u;
        k.dispatch(1024u, {}, R.blob, 1u, 1u);
        dev->flush();
        checkMain(dev, R, label);
        checkTrap(dev, R, label);
    }

    // ---- coverage, demanded rather than designed ----
    {
        const std::vector<Expected> E = predict();
        uint32_t maxPer = 0, minPer = 0xffffffffu, filtered = 0;
        for (uint32_t i = D_START; i < D_END; ++i)
        {
            uint32_t n = 0;
            for (const Expected& e : E)
            {
                if (e.particle != i) continue;
                ++n;
                if (i == FILTERED_PARTICLE && e.plane == FILTERED_PLANE) ++filtered;
            }
            maxPer = std::max(maxPer, n);
            minPer = std::min(minPer, n);
        }
        uint32_t up = 0, down = 0;
        for (uint32_t i = D_START; i < D_END; ++i)
            (predY(i) > originY(i) ? up : down)++;
        PXG_CHECK_MSG(up > 0u && down > 0u,
                      "the CCD offsets are all one sign (%u up, %u down) -- for an UPWARD "
                      "offset the contact-volume terms cancel and the speculative volume is "
                      "unobservable", up, down);
        PXG_CHECK_MSG(maxPer > MAXC, "the deepest particle has only %u contacts -- the cap "
                      "never bites", maxPer);
        PXG_CHECK_MSG(minPer > 0u && minPer < MAXC,
                      "every particle has the same depth band (min %u, max %u) -- a wrong "
                      "count would not be visible", minPer, maxPer);
        PXG_CHECK_MSG(filtered > 0u, "the seeded filter pair matches no contact -- a port "
                      "that borrowed the fluid kernel's findPs could not be caught");
        std::printf("    [ note ] %zu contacts over %u particles; depth band %u..%u against a "
                    "cap of %u; %u trap particles held silent by mMaxDiffuseParticles == 0\n",
                    E.size(), D_END - D_START, minPer, maxPer, MAXC, TRAP_DIFFUSE);
    }

    R.destroy(dev);
}

} // namespace

PXG_TEST(psDiffuseContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}
