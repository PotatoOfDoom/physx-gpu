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
//   * (5.6.1 also seeded a RIGID FILTER PAIR here, to catch a port borrowing the
//     fluid kernel's findPs.  PhysX 5.12 removed that filter from every particle
//     kernel, so there is nothing left to borrow and the pair is gone.)
//   * per-pair rest distances that are all DIFFERENT, so a wrong pair index
//     lands on a different penetration.
//   * output arrays running SIXTEEN planes deep where twelve are legal, plus a
//     poisoned margin, so an over-cap write is visible rather than silent.

#include "pxgTestFramework.h"
#include "pxgGridCalRef.h"

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
static_assert(sizeof(PxgParticleSystem) == 608, "PxgParticleSystem stride");
static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput stride");
static_assert(sizeof(PxgShape) == 48, "PxgShape stride");
static_assert(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform stride");
static_assert(sizeof(PxgParticleContactInfo) == 16, "PxgParticleContactInfo is one float4");
static_assert(PxgParticleContactInfo::MaxStaticContactsPerParticle == 12, "the static-contact cap");
static_assert(offsetof(PxgParticleSystem, mCommonData) == 88, "mCommonData @88");
static_assert(offsetof(PxgParticleSystem, mDiffuseCellStart) == 480, "@480");
static_assert(offsetof(PxgParticleSystem, mDiffuseCellEnd) == 488, "@488");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) == 512, "@512");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) == 520, "@520");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactInfos) == 552, "@552");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactCount) == 576, "@576");

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
// TOTAL_WORK is now COMPUTED (totalWorkItems()): each pair owns one work item
// per cell its overlap spans, and the three bounds profiles give 4, 2 and 1.
// It used to be a hand-written NB_PLANES + 2 with pair 0 owning two items --
// which was the only reason `pos` differed from `workIndex` at all.  With the
// profiles the run-sum is non-trivial everywhere, so that property survives
// the change and stops depending on one hand-placed duplicate.
static const uint32_t EXCLUDED_PAIR = NB_PLANES + 1u;    // 15
static const uint32_t MAXC = 12u;
static const uint32_t ALLOC_C = 16u;        // four planes past the legal twelve
static const uint32_t SLACK = 137u;
static const uint32_t NUM_DIFFUSE = 44u;    // published count for the main system
// P9.24: the live range has to hold the DERIVED cell spans, which are 1, 2, 3,
// 4, 5 and 6 over the six cells some pair reads -- 21 particles.  The
// invariant check below is what said so, in those words, when the range was
// still the old 14.
static const uint32_t D_START = 3u;         // first live diffuse particle
static const uint32_t D_END = 39u;          // EXACTLY the derived spans: 1+2+..+8
static const uint32_t TRAP_DIFFUSE = 8u;    // the mMaxDiffuseParticles == 0 system
static const float CONTACT_OFFSET = 0.3f;
static const float POISON_F = -7.5e3f;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;
static const uint32_t NODE_BASE = 100u;

// ---------------------------------------------------------------------------
// P9.24 -- THE GRID, which until this tranche was switched off.
//
// This fixture ran `mGridSizeX/Y/Z = 1` and said so: "1x1x1 -> gridHash is
// always 0".  That is not a simplification, it is a disabling.  With one cell
// calcWrappedGridRangeSize clamps the range to (1,1,1), calcGridOffsetInRange
// returns (0,0,0) for every cell offset, and calcGridHash masks each component
// with (1-1) == 0 -- so ANY implementation that returns zero passes, and
// `ps_primitivesDiffuseCollisionLaunch`'s call to calcGridHashInBounds was
// verified by nothing.
//
// THE BOUNDS ARE THE ONLY THING THAT MOVED.  A shape's bounds feed the grid hash
// and nothing else -- the plane contact math takes the TRANSFORM -- so the cell
// structure below is free to be redesigned without disturbing a single one of
// the thirty planes' expectations.
//
// ---------------------------------------------------------------------------
// WHY 4x2x1 AND NOT 2x2x1, which is what the first sweep asked for
// ---------------------------------------------------------------------------
// The first attempt used 2x2x1 (P9.18's shape) and killed 6 of 11 mutants.  The
// five survivors were not five accidents; three of them are STRUCTURALLY
// invisible at gridSize 2, and working out why is what fixed the rig:
//
//   g3-hash-from-max-corner  needs an axis where 1 < rs < S.  At S == 2 that
//       window is empty: rs == 1 forces gpMin == gpMax (min and max corner are
//       the same cell), and rs == 2 makes the offsets cover BOTH residues of the
//       mask, so the min-corner and max-corner cell SETS coincide.  At S == 4
//       with rs == 2 the sets are {p, p+1} and {p+2, p+3} -- different.
//
//   g8-gridpos-trunc-not-floor  needs a NEGATIVE gpMin whose masked value moves.
//       At S == 2, floor gives -1 -> 1 and trunc gives 0 -> 0, but every pair
//       with rs == 2 covers both residues anyway.  At S == 4, -1 -> 3 and
//       0 -> 0, and a pair with rs == 2 reads {3, 0} instead of {0, 1}.
//
//   g11-cellwidth-one  needs an axis with rs == 1 where floor(v/4) and floor(v/1)
//       land on different residues.  Near the origin they do not (0.2 gives 0
//       either way); at v = 4.2 they give 1 and 4, which differ under mask 3.
//       Hence PROFILE_D, which sits a pair's whole x-extent inside cell 1.
//
//   g7-offset-xy-swapped  is invariant for ANY pair that reads its full
//       rs.x * rs.y block -- swapping the two offset components just permutes
//       the same cells.  It needs a pair reading a NON-SYMMETRIC SUBSET, which
//       is what PROFILE_E is: the same overlap as A, but the fixture declares
//       only TWO work items for it, so the kernel visits offsets 0 and 1 --
//       (0,0) and (1,0), which the swap turns into (0,0) and (0,1).
//       Declaring fewer work items than the span is a FIXTURE liberty: the
//       engine allocates exactly one per cell.  The kernel's behaviour for a
//       prefix of the offsets is the same code on the same path, and P9.18 used
//       the same device on the convex sibling.
//
//   g5-mask-unsigned  is the one genuine EQUIVALENT: `g.x & int(s.x - 1u)` and
//       `int(uint(g.x) & (s.x - 1u))` are the same bits.  A bitwise AND does not
//       care how its operands are signed, and the mask is non-negative.  It stays
//       in the sweep as a recorded equivalent, not as an open survivor.
//
// Cell width FOUR: with a cell that wide, whether a pair's overlap crosses a
// boundary is decided by numbers far from any boundary, which is what keeps the
// oracle exact rather than a knife edge.
static const float GRID_CELL_WIDTH = 4.0f;
static const uint32_t GRID_SIZE[3] = { 4u, 2u, 1u };
static const uint32_t NB_CELLS = 8u;

// Five bounds profiles, so different pairs read different CELL SETS.  A pair
// that reads every cell has a union invariant under any relabelling, so a
// fixture in which every pair reads everything cannot see a relabelling at all.
enum BoundsProfile { PROFILE_A = 0, PROFILE_B, PROFILE_C, PROFILE_D, PROFILE_E,
                     PROFILE_F, NB_PROFILES };
static BoundsProfile profileOf(uint32_t pair) { return BoundsProfile(pair % NB_PROFILES); }

// The particle system's own bounds; every pair's overlap is this intersected
// with the plane's and grown by the contact distance.
static const float PS_BMN[3] = { -1.0f, -1.0f, -1.0f };
// x runs to 17, not 5, and that is g6-range-not-clamped's doing.  The clamp in
// calcWrappedGridRangeSize only matters where the RAW range exceeds the grid
// size, and it only becomes VISIBLE at an offset that reaches past the clamped
// size.  With x spanning 6 raw cells clamped to 4 and y spanning 3 clamped to 2,
// the span is 8 and offset 6 lands on cell 1 clamped versus cell 3 unclamped --
// so the pair reaches the difference without the fixture having to declare more
// work items than the span.  At the earlier x extent of 5 nothing was clamped in
// x at all, no offset reached the y difference either, and the mutant was
// output-identical.
static const float PS_BMX[3] = { 17.0f, 5.0f, 1.0f };

static void planeBoundsFor(uint32_t pair, float bmn[3], float bmx[3])
{
    // +-40, not +-8: the overlap is the INTERSECTION, so a plane bound of 8 caps
    // the region at 8 no matter how far the particle system reaches -- and with
    // the region capped there, the raw x range is exactly 4, the clamp never
    // bites, and g6-range-not-clamped is output-identical.  That cost a sweep
    // round to find: the check that was supposed to guarantee the clamp bites
    // was reading the PARTICLE system's extent and not the overlap's.
    for (int k = 0; k < 3; ++k) { bmn[k] = -40.0f; bmx[k] = 40.0f; }
    switch (profileOf(pair))
    {
    case PROFILE_A: case PROFILE_E: break;                     // the full overlap
    case PROFILE_B: bmn[1] = 0.5f; bmx[1] = 2.0f; break;       // y inside one cell
    case PROFILE_C: bmn[0] = 0.5f; bmx[0] = 2.0f;              // x and y inside cell 0
                    bmn[1] = 0.5f; bmx[1] = 2.0f; break;
    case PROFILE_D: bmn[0] = 4.5f; bmx[0] = 5.0f;              // x inside cell 1 -- g11
                    bmn[1] = 0.5f; bmx[1] = 2.0f; break;
    // F is the only profile with 1 < rs < gridSize on an axis (rs.x == 2 of 4),
    // which is the window g3-hash-from-max-corner needs: at rs == 1 the min and
    // max corner ARE the same cell, and at rs == gridSize the offsets cover every
    // residue of the mask, so both ends give the same set either way.
    case PROFILE_F: bmn[0] = 0.5f; bmx[0] = 5.0f; break;
    default: break;
    }
}

// Plane j sits at y = PLANE_Y0 + PLANE_DY * j, normal +Y.
static float planeHeight(uint32_t j) { return -0.5f + 0.05f * float(j); }
// Diffuse particle i's ORIGIN and PREDICTED heights.  The offsets are distinct
// and non-zero so the CCD midpoint is neither endpoint.
// P9.24: the step tightened from 0.10 to 0.065 when the live range grew from
// 14 particles to 21.  At 0.10 the last particles sat above every plane and
// got NO contacts, which the depth-band check below rejects: a particle that
// never contacts cannot show a wrong count.
static float originY(uint32_t i) { return -0.45f + 0.037f * float(i - D_START); }
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

// ---------------------------------------------------------------------------
// The grid oracle for THIS rig, over the shared engine mirror (pxgGridCalRef.h).
// ---------------------------------------------------------------------------
static void pairOverlap(uint32_t pair, float omn[3], float omx[3])
{
    float pmn[3], pmx[3];
    planeBoundsFor(pair, pmn, pmx);
    // cDistance = contactDistance[particleCacheRef] + contactDistance[rigidCacheRef];
    // only the particle system carries one here.
    pxgtest::gridcal::overlapBound(PS_BMN, PS_BMX, pmn, pmx, CONTACT_OFFSET, omn, omx);
}

static pxgtest::gridcal::U3 gridU3()
{
    return pxgtest::gridcal::U3{ GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2] };
}

// The number of cells the pair's overlap SPANS -- what the engine would allocate.
static uint32_t spanOfPair(uint32_t pair)
{
    float omn[3], omx[3];
    pairOverlap(pair, omn, omx);
    return pxgtest::gridcal::cellsInBounds(omn, omx, GRID_CELL_WIDTH, gridU3());
}

// The number of work items this FIXTURE declares for the pair.  Equal to the
// span everywhere except PROFILE_E, which declares two so the kernel visits a
// non-symmetric PREFIX of the offsets -- see the note on g7 above.
static const uint32_t PROFILE_E_ITEMS = 2u;
static uint32_t cellsOfPair(uint32_t pair)
{
    const uint32_t span = spanOfPair(pair);
    if (profileOf(pair) == PROFILE_E) return std::min(PROFILE_E_ITEMS, span);
    return span;
}

// The cell a given work item of a given pair reads.  `cellOffset` is
// workIndex - startIndices[pair], exactly as the kernel spells it.
static uint32_t cellOfWorkItem(uint32_t pair, uint32_t cellOffset)
{
    float omn[3], omx[3];
    pairOverlap(pair, omn, omx);
    return pxgtest::gridcal::calcGridHashInBounds(omn, omx, GRID_CELL_WIDTH,
                                                  cellOffset, gridU3());
}

// The per-pair run-sum the kernel binary-searches, and the total it stops at.
static uint32_t pairStartIndex(uint32_t pair)
{
    uint32_t acc = 0;
    for (uint32_t j = 0; j < pair; ++j) acc += cellsOfPair(j);
    return acc;
}
static uint32_t totalWorkItems() { return pairStartIndex(NB_PLANES + 1u); }

// ---------------------------------------------------------------------------
// THE CELL CONTENTS ARE DERIVED, NOT WRITTEN DOWN.
//
// Every cell some pair actually reads gets a DISTINCT particle count (1, 2, 3,
// ... in ascending cell order); every cell nothing reads is left EMPTY.  Two
// cells holding the same number of particles are interchangeable, so a swap
// between them is invisible -- P9.18 hit exactly that with even eighths and had
// to re-cut its boundaries by hand.  Deriving the spans from the oracle keeps
// the property true when the profiles change, and it is safe to derive them
// from the oracle because the KERNEL computes its hashes independently: if the
// two disagree the kernel reads different cells and the contacts diverge.
// ---------------------------------------------------------------------------
struct CellTable
{
    uint32_t lo[NB_CELLS], hi[NB_CELLS];
    bool read[NB_CELLS];
    uint32_t nbRead = 0, live = 0;
};

static const CellTable& cells()
{
    static CellTable t = [] {
        CellTable c{};
        for (uint32_t i = 0; i < NB_CELLS; ++i) { c.lo[i] = c.hi[i] = 0u; c.read[i] = false; }
        for (uint32_t j = 0; j <= NB_PLANES; ++j)
        {
            const uint32_t n = cellsOfPair(j);
            for (uint32_t o = 0; o < n; ++o) c.read[cellOfWorkItem(j, o) % NB_CELLS] = true;
        }
        uint32_t cursor = D_START, span = 1u;
        for (uint32_t i = 0; i < NB_CELLS; ++i)
        {
            c.lo[i] = cursor;
            if (c.read[i]) { cursor += span; ++span; ++c.nbRead; }
            c.hi[i] = cursor;
        }
        c.live = cursor - D_START;
        return c;
    }();
    return t;
}

static uint32_t cellOfParticle(uint32_t i)
{
    const CellTable& c = cells();
    for (uint32_t k = 0; k < NB_CELLS; ++k)
        if (c.hi[k] > c.lo[k] && i >= c.lo[k] && i < c.hi[k]) return k;
    return 0xFFFFFFFFu;
}

// Does this pair reach this particle at all?  Only through the cell that holds
// it, and only if one of the pair's work items resolves to that cell.
static bool pairReadsParticle(uint32_t pair, uint32_t i)
{
    const uint32_t want = cellOfParticle(i);
    if (want == 0xFFFFFFFFu) return false;
    const uint32_t n = cellsOfPair(pair);
    for (uint32_t o = 0; o < n; ++o)
        if ((cellOfWorkItem(pair, o) % NB_CELLS) == want) return true;
    return false;
}

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
    DevBuf bRunsum, bSimBuf;
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
            // P9.24: the pair only reaches this particle through the CELL that
            // holds it, and only if one of the pair's work items resolves to
            // that cell.  Before this tranche the grid was 1x1x1 and every pair
            // reached every particle, so this line had nothing to say.
            if (!pairReadsParticle(j, i)) continue;
            const float ph = planeHeight(j);
            // PxPlane::transform: `0 - p.dot(n)` (PxPlane.h:137), not a unary minus
            const float pdd = 0.0f - (pn[0] * 0.0f + pn[1] * ph + pn[2] * 0.0f);
            float dist = pn[0] * 0.0f + pn[1] * cvY + pn[2] * 0.0f + pdd;
            if (dist > cvR) continue;
            // enableCCD -> dist recomputed from the ORIGIN position
            dist = pn[0] * 0.0f + pn[1] * oy + pn[2] * 0.0f + pdd;
            Expected e;
            e.particle = i;
            e.plane = j;
            e.normalPen = F4{ pn[0], pn[1], pn[2], dist - restDistanceFor(j) };
            e.node = uint64_t(NODE_BASE + j);
            // One work item per cell, and the cells are disjoint, so a pair
            // produces each of its particles' contacts exactly ONCE.  The
            // doubling that used to live here existed only to make the run-sum
            // non-trivial; the per-pair cell counts (4, 2, 1) do that now.
            out.push_back(e);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// P9.24 -- WHAT THE RIG HAS TO BE, checked instead of asserted in a comment.
//
// Each of these was forced by a mutant that survived without it.  A property
// that merely happens to hold today is a property a later edit removes silently.
// ---------------------------------------------------------------------------
static void checkGridInvariants()
{
    const CellTable& c = cells();

    // 1. at least four cells must be READ, and at least one left unread: a cell
    //    nothing resolves to is a place a relabelling can be caught VANISHING into.
    PXG_CHECK_MSG(c.nbRead >= 4u,
                  "only %u of %u cells are read -- a cell no work item resolves to is "
                  "not under test", c.nbRead, NB_CELLS);

    // 2. no two READ cells may hold the same number of particles, or swapping
    //    them is unobservable.  P9.18 hit exactly this with even eighths.
    uint32_t dup = 0;
    for (uint32_t a2 = 0; a2 < NB_CELLS; ++a2)
        for (uint32_t b2 = a2 + 1u; b2 < NB_CELLS; ++b2)
            if (c.read[a2] && c.read[b2] &&
                (c.hi[a2] - c.lo[a2]) == (c.hi[b2] - c.lo[b2])) ++dup;
    PXG_CHECK_MSG(dup == 0u, "%u read-cell pairs hold the same particle count -- those "
                  "cells are interchangeable and a swap is invisible", dup);

    // 3. the live particles must fit the published diffuse range.
    // EXACTLY, not "at most": a particle inside the published diffuse range but
    // outside every cell contacts nothing, and a particle that never contacts
    // cannot show a wrong count -- which is what the depth-band check downstream
    // rejects, in those words.
    PXG_CHECK_MSG(D_START + c.live == D_END,
                  "the derived cells hold %u particles from %u, so the diffuse range "
                  "must be [%u, %u) -- it is [%u, %u)",
                  c.live, D_START, D_START, D_START + c.live, D_START, D_END);

    // 4. the pairs must read at least four DIFFERENT cell sets, and one of them
    //    must be a PROPER SUBSET of its own block -- g7-offset-xy-swapped is
    //    invariant for any pair that reads its full rs.x * rs.y block.
    uint32_t masks[NB_PLANES], distinct = 0, subsetPairs = 0;
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        uint32_t m = 0;
        const uint32_t n = cellsOfPair(j);
        for (uint32_t o = 0; o < n; ++o) m |= 1u << (cellOfWorkItem(j, o) % NB_CELLS);
        masks[j] = m;
        if (n < spanOfPair(j)) ++subsetPairs;
    }
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        bool seen = false;
        for (uint32_t k = 0; k < j; ++k) if (masks[k] == masks[j]) seen = true;
        if (!seen) ++distinct;
    }
    PXG_CHECK_MSG(distinct >= 4u, "the pairs read only %u distinct cell sets; a "
                  "relabelling is invisible to a pair that reads all of them", distinct);
    PXG_CHECK_MSG(subsetPairs > 0u,
                  "no pair reads a PROPER SUBSET of its block -- swapping the two offset "
                  "components then only permutes cells the pair reads anyway");

    // 5. some axis must have 1 < rs < gridSize, or the min-corner and the
    //    max-corner produce the SAME cell set under the mask and
    //    g3-hash-from-max-corner cannot be seen.
    uint32_t windowed = 0;
    for (uint32_t j = 0; j <= NB_PLANES; ++j)
    {
        float omn[3], omx[3];
        pairOverlap(j, omn, omx);
        const pxgtest::gridcal::U3 rs = pxgtest::gridcal::calcWrappedGridRangeSize(
            pxgtest::gridcal::calcGridPos(omn, GRID_CELL_WIDTH),
            pxgtest::gridcal::calcGridPos(omx, GRID_CELL_WIDTH), gridU3());
        if ((rs.x > 1u && rs.x < GRID_SIZE[0]) || (rs.y > 1u && rs.y < GRID_SIZE[1]))
            ++windowed;
    }
    PXG_CHECK_MSG(windowed > 0u,
                  "no pair has 1 < rs < gridSize on any axis -- the min and max corner "
                  "then hash to the same set and reading the wrong one is invisible");

    // 6. some pair's overlap must start at a NEGATIVE grid position, or trunc
    //    and floor agree everywhere and g8 is invisible.
    uint32_t negative = 0;
    for (uint32_t j = 0; j <= NB_PLANES; ++j)
    {
        float omn[3], omx[3];
        pairOverlap(j, omn, omx);
        const pxgtest::gridcal::I3 g = pxgtest::gridcal::calcGridPos(omn, GRID_CELL_WIDTH);
        if (g.x < 0 || g.y < 0) ++negative;
    }
    PXG_CHECK_MSG(negative > 0u, "every overlap starts at a non-negative grid position -- "
                  "truncation and flooring then agree and cannot be told apart");

    // 7. some pair must sit ENTIRELY inside a cell that is not cell 0 on an axis
    //    where floor(v / cellWidth) and floor(v) land on different residues, or
    //    the cell width itself is never tested.
    uint32_t widthSensitive = 0;
    for (uint32_t j = 0; j <= NB_PLANES; ++j)
    {
        float omn[3], omx[3];
        pairOverlap(j, omn, omx);
        const int32_t withW = int32_t(std::floor(omn[0] / GRID_CELL_WIDTH));
        const int32_t withOne = int32_t(std::floor(omn[0]));
        if ((uint32_t(withW) & (GRID_SIZE[0] - 1u)) != (uint32_t(withOne) & (GRID_SIZE[0] - 1u)))
            ++widthSensitive;
    }
    PXG_CHECK_MSG(widthSensitive > 0u,
                  "no pair's x origin distinguishes cellWidth from 1 under the mask -- "
                  "the cell width is then not under test");

    // 8. no overlap bound may sit near a cell boundary -- otherwise the floor is
    //    a knife edge and the oracle stops being exact.
    float worst = 1.0f;
    for (uint32_t j = 0; j <= NB_PLANES; ++j)
    {
        float omn[3], omx[3];
        pairOverlap(j, omn, omx);
        for (int k = 0; k < 3; ++k)
            for (float v : { omn[k], omx[k] })
            {
                const float t = v / GRID_CELL_WIDTH;
                worst = std::min(worst, std::min(t - std::floor(t), std::ceil(t) - t));
            }
    }
    PXG_CHECK_MSG(worst > 0.02f,
                  "an overlap bound sits %.4f of a cell from a boundary -- too close for "
                  "the floor to be an exact oracle", worst);

    // 9. the TRAP pair reuses plane 0's bounds, so its profile has to match.
    PXG_CHECK_MSG(profileOf(NB_PLANES) == profileOf(0u),
                  "the trap pair takes plane 0's bounds, so profileOf(%u) must equal "
                  "profileOf(0)", NB_PLANES);

    // 10b. THE CLAMP MUST BE OBSERVABLE, and this check now says that by
    //      SIMULATING the mutant rather than by reasoning about it.
    //
    //      The first version asked whether some pair's raw range exceeded the
    //      grid size and whether it declared more offsets than the clamped x
    //      size.  It passed -- and g6-range-not-clamped survived anyway, because
    //      the property it was checking was not the property that matters.  The
    //      overlap is an INTERSECTION, so a plane bound of 8 capped the region
    //      long before the particle system's extent of 17 could make the raw x
    //      range exceed 4; and the y clamp (3 -> 2) was unreachable because
    //      off / rs.x never got to 2 inside a span of 8.
    //
    //      A check that reasons about a mutant can be wrong about it.  A check
    //      that RUNS the mutant cannot.
    uint32_t clampVisible = 0;
    for (uint32_t j = 0; j <= NB_PLANES; ++j)
    {
        float omn[3], omx[3];
        pairOverlap(j, omn, omx);
        const pxgtest::gridcal::I3 mn = pxgtest::gridcal::calcGridPos(omn, GRID_CELL_WIDTH);
        const pxgtest::gridcal::I3 mx = pxgtest::gridcal::calcGridPos(omx, GRID_CELL_WIDTH);
        const pxgtest::gridcal::U3 clamped =
            pxgtest::gridcal::calcWrappedGridRangeSize(mn, mx, gridU3());
        const pxgtest::gridcal::U3 raw{ uint32_t(std::max(mx.x - mn.x + 1, 0)),
                                        uint32_t(std::max(mx.y - mn.y + 1, 0)),
                                        uint32_t(std::max(mx.z - mn.z + 1, 0)) };
        const uint32_t n = cellsOfPair(j);
        for (uint32_t o = 0; o < n; ++o)
        {
            const pxgtest::gridcal::U3 oc = pxgtest::gridcal::calcGridOffsetInRange(clamped, o);
            const pxgtest::gridcal::U3 orw = pxgtest::gridcal::calcGridOffsetInRange(raw, o);
            const uint32_t hc = pxgtest::gridcal::calcGridHash(
                pxgtest::gridcal::I3{ mn.x + int32_t(oc.x), mn.y + int32_t(oc.y),
                                      mn.z + int32_t(oc.z) }, gridU3());
            const uint32_t hr = pxgtest::gridcal::calcGridHash(
                pxgtest::gridcal::I3{ mn.x + int32_t(orw.x), mn.y + int32_t(orw.y),
                                      mn.z + int32_t(orw.z) }, gridU3());
            if (hc != hr) { ++clampVisible; break; }
        }
    }
    PXG_CHECK_MSG(clampVisible > 0u,
                  "dropping the clamp in calcWrappedGridRangeSize changes NO cell any "
                  "pair reads -- the clamp is not under test");

    // 10. the run-sum must be non-trivial, or restDistances[pos] and
    //     restDistances[workIndex] are the same expression.
    uint32_t offset = 0;
    for (uint32_t j = 0; j < NB_PLANES; ++j) if (pairStartIndex(j) != j) ++offset;
    PXG_CHECK_MSG(offset > 0u, "pos == workIndex for every pair -- the per-pair run-sum "
                  "is the identity and nothing distinguishes the two indices");

    std::printf("    [ grid ] %ux%ux%u cells, width %.1f; %u work items over %u pairs; "
                "%u cells read (%u live particles), %u distinct cell sets, %u subset pair(s)\n",
                GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2], GRID_CELL_WIDTH,
                totalWorkItems(), NB_PLANES + 1u, c.nbRead, c.live, distinct, subsetPairs);
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
    // P9.24: FOUR cells, with the spans of CELL_BOUNDS.  Their union is still
    // exactly [D_START, D_END) -- the same fourteen live particles as before --
    // but which pair sees which of them is now decided by the grid hash.
    std::vector<uint32_t> dcs(NB_CELLS), dce(NB_CELLS);
    for (uint32_t c = 0; c < NB_CELLS; ++c)
    { dcs[c] = cells().lo[c]; dce[c] = cells().hi[c]; }
    R.bDCellStart = R.mk(dev, dcs);
    R.bDCellEnd = R.mk(dev, dce);
    R.bCellStart = R.mk(dev, std::vector<uint32_t>(NB_CELLS, 0u));
    R.bCellEnd = R.mk(dev, std::vector<uint32_t>(NB_CELLS, NUM_DIFFUSE));

    // D8: the two-slot count pair; [1] is a trap, larger than [0].
    R.bTwo = R.mk(dev, std::vector<int32_t>{ int32_t(NUM_DIFFUSE), int32_t(NUM_DIFFUSE) * 3 });

    R.bInfos = R.mk(dev, std::vector<F4>(infoSlots(NUM_DIFFUSE),
                                         F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    R.bForces = R.mk(dev, std::vector<float>(infoSlots(NUM_DIFFUSE), POISON_F));
    R.bNodes = R.mk(dev, std::vector<uint64_t>(infoSlots(NUM_DIFFUSE), 0xDEADBEEFCAFEF00Dull));
    std::vector<uint32_t> counts(NUM_DIFFUSE + SLACK, POISON_U32);
    for (uint32_t i = 0; i < NUM_DIFFUSE; ++i) counts[i] = 0u;
    R.bCounts = R.mk(dev, counts);

    // D4: PhysX 5.12 removed the particle-rigid element filter altogether (with
    // particle rigids, 5.9), so there is no fluid-kernel findPs left to borrow and no
    // filter pair to seed; the sim buffer is a plain one.
    R.bRunsum = R.mk(dev, std::vector<uint32_t>{ 0u });
    std::vector<PxgParticleSimBuffer> simBufs(1);
    std::memset(simBufs.data(), 0, sizeof(PxgParticleSimBuffer));
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
    m.mCommonData.mGridCellWidth = GRID_CELL_WIDTH;
    m.mCommonData.mGridSizeX = GRID_SIZE[0];   // P9.24: 2x2x1 -- see the note above
    m.mCommonData.mGridSizeY = GRID_SIZE[1];
    m.mCommonData.mGridSizeZ = GRID_SIZE[2];
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

    // P9.24: the bounds are what drive the grid hash, and nothing else.  Both
    // particle systems carry PS_BMN/PS_BMX; each plane carries the bounds of its
    // pair's profile, which is what makes different pairs read different cells.
    std::vector<float> boundsF(size_t(NB_SHAPES) * 6u, 0.0f);
    for (uint32_t sh = 0; sh < 2u; ++sh)
        for (int k = 0; k < 3; ++k)
        {
            boundsF[size_t(sh) * 6 + k] = PS_BMN[k];
            boundsF[size_t(sh) * 6 + 3 + k] = PS_BMX[k];
        }
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        float pmn[3], pmx[3];
        planeBoundsFor(j, pmn, pmx);
        for (int k = 0; k < 3; ++k)
        {
            boundsF[size_t(2u + j) * 6 + k] = pmn[k];
            boundsF[size_t(2u + j) * 6 + 3 + k] = pmx[k];
        }
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
    for (uint32_t j = 0; j < NB_PAIRS_TABLE; ++j) startIdx[j] = pairStartIndex(j);
    R.bStart = R.mk(dev, startIdx);
    // TOTAL_WORK covers pairs 0..NB_PLANES only; the excluded row's work item
    // sits at index TOTAL_WORK and must never run.
    R.bTotal = R.mk(dev, std::vector<uint32_t>{ totalWorkItems() });

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
        uint32_t maxPer = 0, minPer = 0xffffffffu;
        for (uint32_t i = D_START; i < D_END; ++i)
        {
            uint32_t n = 0;
            for (const Expected& e : E)
            {
                if (e.particle != i) continue;
                ++n;
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
        std::printf("    [ note ] %zu contacts over %u particles; depth band %u..%u against a "
                    "cap of %u; %u trap particles held silent by mMaxDiffuseParticles == 0\n",
                    E.size(), D_END - D_START, minPer, maxPer, MAXC, TRAP_DIFFUSE);
    }

    R.destroy(dev);
}

} // namespace

PXG_TEST(psDiffuseContactGen)
{
    checkGridInvariants();
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}
