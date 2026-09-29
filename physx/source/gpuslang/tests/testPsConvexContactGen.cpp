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

// testPsConvexContactGen.cpp -- P9.14b, the fixture for
// ps_convexCollisionLaunch (psContactGenBda.slang's third entry point).
//
// ---------------------------------------------------------------------------
// THE ORACLE, STATED HONESTLY
// ---------------------------------------------------------------------------
// The contact test here is GJK/EPA, which is iterative: there is no closed form
// to compare its output against bit for bit, and no host-callable engine
// original (squawk::gjk is __device__ only).  What CAN be pinned exactly is the
// GEOMETRY the query is asked about -- so the hull is a BOX with identity
// rotations, for which point-vs-box separation IS closed form:
//
//     d = p - centre;  q = clamp(d, -h, h);  separation = |d - q|
//
// and the fixture asserts, per particle:
//
//   * WHICH particles produce a contact -- the analytic set, exactly;
//   * that the stored normal is a UNIT vector;
//   * that the stored normal points OUT of the box.  For a particle above the
//     top face that is +Y, and it is what pins X7's negation PHYSICALLY rather
//     than by copying a sign out of a measured run: the solver has to push the
//     particle up, and a port that stores +direction pushes it down;
//   * that the stored penetration is separation - restDistance, to a tight
//     tolerance, with the worst error printed rather than hidden.
//
// The GJK/EPA algebra itself is NOT re-verified here: it is pxgSlangGjk /
// pxgSlangEpa / pxgSlangTetCollide, already swept under the softbody-vs-rigid
// path, and this kernel composes them.  What this fixture verifies is
// everything AROUND that -- the hull decode, the cell walk, the filter, the
// contact-volume policy and the writer.
//
// ---------------------------------------------------------------------------
// WHAT THE GEOMETRY IS BUILT TO DISCRIMINATE
// ---------------------------------------------------------------------------
//   * a SYNTHETIC hull blob laid out by hand at the offsets pxgSlangConvexHull
//     documents (float4 com, uint4 desc, float4 extents, float4 verts[]), so a
//     wrong section offset reads a neighbour rather than plausible garbage;
//   * particles spread ACROSS the top face, off the side, and INSIDE the box,
//     so the separated, the touching and the penetrating branches all run;
//   * the FLUID cell range covering a different span than the particles that
//     can possibly touch, so a wrong cell walk is visible;
//   * BOTH the PGS and the TGS dispatch, which must agree -- X1: this kernel
//     passes isDiffuse = TRUE, so the TGS halving never applies, and a port
//     that passes isTGS through makes the two runs differ;
//   * a RIGID FILTER PAIR matching one particle, keyed on its UNSORTED index
//     (X9), so a port that keys on the sorted one suppresses the wrong particle
//     -- or none;
//   * a second work item whose cell is EMPTY, and a third pointing at a
//     particle system with a different grid, so the per-work-item hull decode
//     and cell walk cannot be hoisted;
//   * output arrays with a poisoned margin past the contact capacity.

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

struct F4 { float x, y, z, w; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

static_assert(sizeof(PxgParticleSystem) == 656, "PxgParticleSystem stride");
static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput stride");
static_assert(sizeof(PxgShape) == 48, "PxgShape stride");
static_assert(offsetof(PxgShape, hullOrMeshPtr) == 32, "PxgShape::hullOrMeshPtr @32");
static_assert(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform stride");
static_assert(sizeof(PxgRigidFilterPair) == 32, "PxgRigidFilterPair stride");
static_assert(sizeof(PxgParticlePrimitiveContact) == 32, "contact stride");

// psContactGenBda.slang's PsCollideParams, byte for byte (the 176-byte blob the
// engine builds from 23 individual arguments -- see kPsConvexCollide).
struct PsCollideBlob
{
    uint32_t isTGSWord;                        // @0
    uint32_t numTests;                         // @4
    float toleranceLength;                     // @8
    uint32_t pad12;                            // @12
    uint64_t cmInputs;                         // @16
    uint64_t shapes;                           // @24
    uint64_t transformCache;                   // @32
    uint64_t bounds;                           // @40
    uint64_t contactDistance;                  // @48
    uint64_t restDistances;                    // @56
    uint64_t materials;                        // @64
    uint64_t startIndices;                     // @72
    uint64_t totalNumPairs;                    // @80
    uint64_t particleSystems;                  // @88
    uint64_t shapeToRigidRemap;                // @96
    uint64_t w_particleContacts;               // @104
    uint64_t w_numTotalContacts;               // @112
    uint64_t w_contactSortedByParticle;        // @120
    uint64_t w_tempContactByParticle;          // @128
    uint64_t w_contactIndexSortedByParticle;   // @136
    uint64_t w_contactByRigid;                 // @144
    uint64_t w_tempContactByRigid;             // @152
    uint64_t w_contactIndexSortedByRigid;      // @160
    uint32_t w_maxContacts;                    // @168
    uint32_t pad172;                           // @172 -> 176
};
static_assert(sizeof(PsCollideBlob) == 176, "engine ABI blob size");
static_assert(offsetof(PsCollideBlob, w_numTotalContacts) == 112, "@112");
static_assert(offsetof(PsCollideBlob, w_maxContacts) == 168, "@168");

struct HostContact { F4 normal_pen; uint64_t rigidId; uint64_t particleId; };

// ---------------------------------------------------------------------------
// The scene.
// ---------------------------------------------------------------------------
// Shape bounds, at file scope because the ORACLE needs them too: the cell a work
// item reads is calcGridHashInBounds of the pair's fattened OVERLAP region, so
// the expected set cannot be written without them.
static const float SHAPE_BMN[3][3] = { { -9.0f, -3.0f, -7.0f },   // shape 0, the particles
                                       { -4.0f, -8.0f, -5.0f },   // shape 1, the hull
                                       { -9.0f, -3.0f, -7.0f } }; // shape 2, the CCD particles
static const float SHAPE_BMX[3][3] = { {  6.0f,  9.0f,  4.0f },
                                       {  8.0f,  5.0f,  9.0f },
                                       {  6.0f,  9.0f,  4.0f } };
static const float GRID_CELL_WIDTH = 1.0f;
// 2x2x1.  The Y AXIS IS NOT DECORATION: with gridSize.y == 1 the wrapped range
// size is 1 in y whatever the bounds, so calcGridOffsetInRange's y term is always
// zero and both its y arithmetic AND the clamp in calcWrappedGridRangeSize are
// unobservable.  Two mutants survived the first P9.18 sweep on exactly that.
static constexpr uint32_t GRID_SIZE[3] = { 2u, 2u, 1u };   // power of two per axis
// THE PAIR TABLE, AND IT IS DELIBERATELY LOPSIDED -- 4, 3 and 2 work items.
//
// Every count in that list was forced by a mutant that survived without it:
//
//   * FOUR (pair 0) -- full coverage, so every cell is reached and the contact
//     set is the same one the fixture always compared.
//   * THREE (pair 1) -- symmetric coverage cannot see a PERMUTATION of the cell
//     mapping: when a pair reads every cell, relabelling which work item gets
//     which leaves its union, and therefore the compared multiset, identical.
//     Two mutants lived on that (hashing from the max corner, and truncating
//     instead of flooring; both merely relabel).  Missing one cell makes the
//     relabelling visible because the MISSED cell changes.
//   * TWO (pair 2) -- three is still not enough for one of them.  Swapping the
//     x and y terms of calcGridOffsetInRange exchanges offsets 1 and 2, and a
//     pair reading the prefix {0,1,2} is CLOSED under that swap -- same set,
//     different order, invisible.  A pair reading {0,1} is not.
//
// Pair 2 runs system A against the hull again, with its own rest distance, so it
// also gives restDistances[pos] a third value to be wrong about.
static constexpr uint32_t CVX_NB_PAIRS = 3u;
static constexpr uint32_t CVX_WORK_COUNT[CVX_NB_PAIRS] = { 4u, 3u, 2u };
static constexpr uint32_t CVX_WORK_OFFSETS[CVX_NB_PAIRS] = { 0u, 4u, 7u };
static constexpr uint32_t CVX_TOTAL_WORK = 9u;
// Which particle system each pair drives, and which two shapes bound it.
static constexpr uint32_t CVX_PAIR_SYS[CVX_NB_PAIRS] = { 0u, 1u, 0u };
static constexpr uint32_t CVX_PAIR_SHAPE_A[CVX_NB_PAIRS] = { 0u, 1u, 0u };
static constexpr uint32_t CVX_PAIR_SHAPE_B[CVX_NB_PAIRS] = { 1u, 2u, 1u };

static const uint32_t NB_PARTICLES = 40u;
// TWO CELLS, and that is the whole point of P9.18.
//
// Until this tranche every particle contact-gen fixture ran a 1x1x1 grid, and
// said so: "gridHash is always 0".  With one cell calcWrappedGridRangeSize
// clamps the range to (1,1,1), calcGridOffsetInRange returns (0,0,0) whatever
// the cell offset, and calcGridHash masks every component with (1-1) == 0 --
// so ANY implementation that returns zero passes.  The helper has FOUR live
// call sites in psContactGenBda and was pinned by none of them.
//
// The cells are laid out in CELL_BOUNDS below.
static const uint32_t CELL_START = 4u;
static const uint32_t CELL_END = 36u;
// Four cells of eight, and slots 0..3 and 36..39 outside all of them as the
// decoys they always were.  CELL_START stays 4 so particlePos/particlePredicted
// are untouched: this tranche changes which particles a work item READS, not
// where any of them is.
// UNEVEN ON PURPOSE.  Even eighths gave cells 1 and 2 the same accepted count,
// and two cells that accept the same number are interchangeable -- swapping them
// is then unobservable and the check below says so.  These four accept 3, 8, 9
// and 6.  The split also keeps both marginal CCD particles (slots 5 and 9) in
// cell 0 and the filtered slot (30) in cell 3, the two cells BOTH pairs read.
static constexpr uint32_t CELL_BOUNDS[5] = { 4u, 12u, 20u, 29u, 36u };
static constexpr uint32_t NB_CELLS = 4u;

// ---------------------------------------------------------------------------
// THE GRID-HASH ORACLE lives in pxgGridCalRef.h (P9.24).
//
// P9.18 wrote it here, and wrote down at the same time that the other two
// particle contact-gen fixtures still ran a 1x1x1 grid and that the mirror
// "should be SHARED rather than transcribed again".  P9.24 is that: the engine
// transcription -- calcGridPos, calcGridHash, calcWrappedGridRangeSize,
// calcGridOffsetInRange, calcGridHashInBounds, overlapBound -- moved to the
// header unchanged, and the three fixture-specific wrappers stayed here because
// they close over THIS rig's grid size, cell width and cell boundaries.
// ---------------------------------------------------------------------------
using pxgtest::gridcal::I3;
using pxgtest::gridcal::U3;

// The overlap bound of two of THIS rig's shapes.
static void gcOverlapBound(uint32_t shapeA, uint32_t shapeB, float cDistance,
                           float omn[3], float omx[3])
{
    pxgtest::gridcal::overlapBound(SHAPE_BMN[shapeA], SHAPE_BMX[shapeA],
                                   SHAPE_BMN[shapeB], SHAPE_BMX[shapeB],
                                   cDistance, omn, omx);
}

// The cell a given work item of a given pair reads.  `cellOffset` is
// workIndex - offsets[pair], exactly as the kernel spells it.
static uint32_t gcCellOfWorkItem(uint32_t shapeA, uint32_t shapeB, float cDistance,
                                 uint32_t cellOffset)
{
    float omn[3], omx[3];
    gcOverlapBound(shapeA, shapeB, cDistance, omn, omx);
    return pxgtest::gridcal::calcGridHashInBounds(
        omn, omx, GRID_CELL_WIDTH, cellOffset,
        U3{ GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2] });
}

// The particle span of cell h, as uploaded to mCellStart / mCellEnd.
static void gcCellSpan(uint32_t h, uint32_t& lo, uint32_t& hi)
{
    lo = CELL_BOUNDS[h];
    hi = CELL_BOUNDS[h + 1u];
}
// Just BELOW the number of contacts the three work items produce, so the writer's
// capacity gate fires -- and the COUNTER must still overshoot it, the same
// contract as the diffuse one-way count in P9.13f/g.
//
// JUST below, and that margin is the point.  The first draft capped at 40 of 82,
// and WHICH 40 survive is the atomicAdd's race: two mutants that had been killed
// started surviving because the contacts they corrupted lost that race and were
// never compared.  A cap that hides half the output hides half the errors.  The
// check below also demands that the written set covers BOTH particle systems.
// P9.18 re-cut this from 76.  The work items of a pair used to read the SAME cell
// and produce every contact twice (82 total); now they read four DIFFERENT cells
// and each particle is produced once per pair that reaches it (48).  The margin is
// what matters, not the number: 54 of 58 leaves the gate firing on four contacts,
// the same ~7% it hid before.  A cap that hides half the output hides half the
// errors.
static const uint32_t MAX_CONTACTS = 54u;
static const uint32_t SLACK = 64u;
// THE HULL IS NOT AXIS-IDENTITY, AND THAT IS THE POINT.
//
// The first draft gave the shape a unit scale, an identity scale rotation and an
// identity rigid rotation.  Three mutants survived on that alone -- dropping the
// whole vertex transform, and swapping vertex2Shape's two rotations, are both
// the IDENTITY when every frame is the identity.
//
// The fix keeps the oracle exact.  A 90-degree rotation about z maps the vertex
// CUBE to itself, so composing
//     vertex -> quatRotateInv(R, scale * quatRotate(R, vertex)) -> rigidQ -> +p
// with R = rigidQ = Rz(90) leaves an AXIS-ALIGNED BOX, only with its half-extents
// PERMUTED twice:
//     vertex cube      (h,   h,   h  )
//     * scale in R     (h*sx, h*sy, h*sz)          = (2.0, 1.2, 2.8)
//     back through R   (h*sy, h*sx, h*sz)          = (1.2, 2.0, 2.8)
//     through rigidQ   (h*sx, h*sy, h*sz)          = (2.0, 1.2, 2.8)
// so the world hull is the box below -- and swapping the two rotations inside
// vertex2Shape yields (1.2, 2.0, 2.8) instead, a different solid.
static const float VERT_H = 2.0f;             // the vertex cube's half-extent
static const float SCALE_X = 1.0f, SCALE_Y = 0.6f, SCALE_Z = 1.4f;
// The WORLD half-extents the composition above must produce.  Written out rather
// than recomputed from the same formula the shader uses -- an oracle that reuses
// the code under test is not an oracle.
static const float BOX_HX = 2.0f;
static const float BOX_HY = 1.2f;
static const float BOX_HZ = 2.8f;
static const float BOX_CY = -1.2f;            // centre y -> top face at y = 0
static const float CONTACT_OFFSET = 0.35f;
static constexpr float REST_DISTANCE = 0.02f;
static constexpr float REST_DISTANCE_B = 0.037f;   // pair 1, deliberately different
static constexpr float REST_DISTANCE_C = 0.061f;   // pair 2, different again
// Per pair, so a port that indexes restDistances by workIndex instead of by the
// pair the binary search found lands on a value no pair owns.
static constexpr float CVX_PAIR_REST[3] = { REST_DISTANCE, REST_DISTANCE_B, REST_DISTANCE_C };
static const uint32_t NODE_INDEX = 77u;
static const uint32_t FILTERED_UNSORTED = 903u;   // the unsorted id the pair matches
static const uint32_t FILTERED_SORTED = 30u;      // which sorted slot carries it
                                                 // (cell 3, which BOTH pairs read)
static const uint32_t MARGINAL_ISDIFFUSE = 5u;    // see particlePos
static const uint32_t MARGINAL_MINSEP = 9u;
static const float POISON_F = -8.25e3f;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;
static const uint64_t POISON_U64 = 0xDEADBEEFCAFEF00Dull;

// Particle i's world position.
//
// The band walks from above the top face (y = 0) down through it and deep into
// the box, and every fourth particle is moved off the +x SIDE so the closest
// feature is that face rather than the top one.
//
// NO SEPARATION MAY LAND ON THE CONTACT DISTANCE.  The first draft stepped y by
// 0.05 from 0.60, which put one particle at a separation of exactly 0.35 = the
// contact offset -- and there GJK's iterative lower bound and the fixture's
// closed form disagree by an ulp, so the kernel emitted 19 contacts where the
// oracle predicted 20.  That was the FIXTURE's knife edge, not a port defect:
// the CUDA's test is `lowerBound > minSep`, which no analytic predicate can
// track at the boundary.  The band now starts at 0.62, which leaves every
// particle at least 0.02 clear of the threshold on one side or the other.
static void particlePos(uint32_t i, float out[3])
{
    const float t = float(int(i) - int(CELL_START));
    // TWO CONSTRUCTED MARGINAL PARTICLES, straight above the top face.  The CCD
    // acceptance test is two nested inequalities, and a band of ordinary
    // particles never lands inside the SHELL where a mis-scaled contact volume
    // or a mis-sized second-pass minSep changes the answer -- the first sweep
    // lost four mutants to exactly that.  These two are placed by solving the
    // inequalities (cDistance = 0.35, top face at y = 0, moving straight down by
    // D, off = D * scaleInv / 2, cVolumeRadius = |off| + 0.35):
    //
    //   MARGINAL_ISDIFFUSE  y = 0.80, D = 0.30
    //       pass 1 accepts iff y <= D*scaleInv/2 + cVolumeRadius, i.e.
    //       y <= D*scaleInv + 0.35 -> 0.65 with scaleInv 1 (REJECT)
    //                              -> 0.95 with scaleInv 2 (ACCEPT)
    //       so isDiffuse = false in a TGS step accepts what the engine rejects.
    //
    //   MARGINAL_MINSEP     y = 1.30, D = 1.00
    //       pass 1 accepts (1.30 <= 1.35); pass 2's bound is
    //       cDistance + 2*cVolumeRadius = 2.05 (ACCEPT), and with the doubling
    //       dropped it is cDistance + cVolumeRadius = 1.20 (REJECT).
    if (i == MARGINAL_ISDIFFUSE) { out[0] = 0.0f; out[1] = 0.80f; out[2] = 0.0f; return; }
    if (i == MARGINAL_MINSEP)    { out[0] = 0.0f; out[1] = 1.30f; out[2] = 0.0f; return; }
    if ((i % 4u) == 3u)
    {
        // Off the +x face by 0.18 -- inside the contact distance, and its nearest
        // feature is the +x FACE, not the top one.
        out[0] = BOX_HX + 0.18f;
        out[1] = -1.0f;
        out[2] = 0.20f * float((i % 3u)) - 0.20f;
        return;
    }
    out[0] = -1.0f + 0.05f * t;
    out[1] = 0.62f - 0.05f * t;      // +0.62 down to about -0.93
    out[2] = 0.25f * float((i % 3u)) - 0.25f;
}

// ---------------------------------------------------------------------------
// THE SECOND PARTICLE SYSTEM, WITH SPECULATIVE CCD ON.
//
// Without it the whole contact-volume policy is dead code in this fixture:
// getSpecVolume is only called when eENABLE_SPECULATIVE_CCD is set, so X1's
// isDiffuse selector and X6's two-pass query would both have been untested and
// their mutants equivalent.  System B is identical to A except for that flag and
// for a per-particle PREDICTED position that differs from the origin.
//
// The oracle still holds, and that is the useful part: under CCD the values the
// kernel REPORTS come from the second pass, which queries the ORIGIN position --
// so normal and penetration are still the closed form at the origin.  Only
// ACCEPTANCE changes, and it becomes two analytic tests:
//     off  = (pred - orig) * scaleInv * 0.5,  scaleInv = 1 (isDiffuse)
//     (a)  signedDistance(orig + off) <= |off| + cDistance
//     (b)  signedDistance(orig)       <= cDistance + 2*(|off| + cDistance)
// A port that passes isDiffuse = false makes scaleInv 2 in a TGS step, which
// moves the midpoint and widens the radius -- a different set.
static void particlePredicted(uint32_t i, float out[3])
{
    float p[3]; particlePos(i, p);
    if (i == MARGINAL_ISDIFFUSE)
    { out[0] = p[0]; out[1] = p[1] - 0.30f; out[2] = p[2]; return; }
    if (i == MARGINAL_MINSEP)
    { out[0] = p[0]; out[1] = p[1] - 1.00f; out[2] = p[2]; return; }
    // Distinct per particle, both signs, and large enough that doubling the
    // scale moves the midpoint by more than the contact distance.
    const float t = float(int(i) - int(CELL_START));
    out[0] = p[0] + 0.02f * t - 0.15f;
    out[1] = p[1] + ((i % 2u) ? 0.55f : -0.45f);
    out[2] = p[2] + 0.01f * t;
}

// The UNSORTED id of sorted slot i -- deliberately far from i so the two cannot
// be confused, and slot FILTERED_SORTED carries the id the filter pair names.
static uint32_t unsortedIdOf(uint32_t i)
{
    return (i == FILTERED_SORTED) ? FILTERED_UNSORTED : (500u + 7u * i);
}

// The SIGNED point-vs-box distance for the fixture's identity-rotation hull:
// positive outside (the euclidean separation), NEGATIVE inside (minus the
// distance to the nearest face).  Both branches are exact for this box, which is
// what lets the fixture check EPA's penetration depth analytically and not only
// GJK's separation.  `outDir` comes back as the unit direction that pushes the
// particle OUT -- away from the surface when separated, toward the nearest face
// when penetrating.
static float boxSignedDistance(const float p[3], float outDir[3])
{
    const float c[3] = { 0.0f, BOX_CY, 0.0f };
    const float h[3] = { BOX_HX, BOX_HY, BOX_HZ };
    float d[3], q[3], e[3];
    float acc = 0.0f;
    for (int k = 0; k < 3; ++k)
    {
        d[k] = p[k] - c[k];
        q[k] = d[k] < -h[k] ? -h[k] : (d[k] > h[k] ? h[k] : d[k]);
        e[k] = d[k] - q[k];
        acc += e[k] * e[k];
    }
    const float outside = std::sqrt(acc);
    if (outside > 0.0f)
    {
        for (int k = 0; k < 3; ++k) outDir[k] = e[k] / outside;
        return outside;
    }
    // Inside: the nearest face wins, and this fixture's band keeps that
    // unambiguous (see particlePos).
    int axis = 0; float best = h[0] - std::fabs(d[0]); float sgn = d[0] < 0.0f ? -1.0f : 1.0f;
    for (int k = 1; k < 3; ++k)
    {
        const float cand = h[k] - std::fabs(d[k]);
        if (cand < best) { best = cand; axis = k; sgn = d[k] < 0.0f ? -1.0f : 1.0f; }
    }
    outDir[0] = outDir[1] = outDir[2] = 0.0f;
    outDir[axis] = sgn;
    return -best;
}

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bDesc, bCin, bShapes, bTc, bBounds, bCd, bRd, bMat, bStart, bTotal, bRemap;
    DevBuf bHull, bCellStart, bCellEnd, bOrigin, bPred, bS2U, bRunsum, bSimBuf, bFilterPairs;
    DevBuf bPredB;   // system B's predicted positions (CCD on)
    DevBuf bContacts, bCount, bSbP, bTbP, bIbP, bBr, bTbr, bIbr;
    PsCollideBlob blob{};

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

// The cooked-hull blob, laid out by hand at the offsets pxgSlangConvexHull
// documents.  A real cooked hull carries planes, polygon descriptors and the
// edge/vertex tables after the vertices; this kernel reads NONE of them, so the
// blob stops after the vertices and the tail is poison -- which is also what
// catches a port that walks past the vertex run.
static std::vector<F4> makeHullBlob()
{
    std::vector<F4> blob;
    blob.push_back(F4{ 0.0f, 0.0f, 0.0f, 0.0f });              // @0  centre of mass
    // @16 uint4: .x is the packed descriptor (nbEdges<<16 | nbVerts<<8 | nbPolys).
    F4 desc{};
    const uint32_t packed = (12u << 16) | (8u << 8) | 6u;      // 12 edges, 8 verts, 6 polys
    std::memcpy(&desc.x, &packed, 4);
    desc.y = 0.0f; desc.z = 0.0f; desc.w = 0.0f;
    blob.push_back(desc);
    // @32 extents; .w is the inner-sphere radius IN VERTEX SPACE.
    blob.push_back(F4{ VERT_H, VERT_H, VERT_H, VERT_H });
    // @48 the eight vertices, in VERTEX space (the shape transform is applied by
    // the kernel).  Centre at (0, BOX_CY, 0) is folded into the shape TRANSFORM,
    // not the vertices, so the vertices are symmetric about the origin.
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2)
                blob.push_back(F4{ VERT_H * float(sx), VERT_H * float(sy),
                                   VERT_H * float(sz), 0.0f });
    // Poisoned tail where a real hull's planes would start.
    for (int i = 0; i < 16; ++i)
        blob.push_back(F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    return blob;
}

static void buildRig(PxgSlangComputeDevice* dev, Rig& R)
{
    R.bHull = R.mk(dev, makeHullBlob());

    // ---- the particle system ----
    std::vector<F4> origin(NB_PARTICLES + SLACK, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    std::vector<F4> pred = origin;
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
    {
        float p[3]; particlePos(i, p);
        origin[i] = F4{ p[0], p[1], p[2], 1.0f };
        pred[i] = F4{ p[0], p[1], p[2], 1.0f };   // CCD off in this fixture
    }
    R.bOrigin = R.mk(dev, origin);
    R.bPred = R.mk(dev, pred);

    // System B shares the origin array and the cell arrays with A; only its
    // PREDICTED positions and its CCD flag differ.
    std::vector<F4> predB(NB_PARTICLES + SLACK, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
    {
        float q[3]; particlePredicted(i, q);
        predB[i] = F4{ q[0], q[1], q[2], 1.0f };
    }
    R.bPredB = R.mk(dev, predB);

    std::vector<uint32_t> s2u(NB_PARTICLES + SLACK, POISON_U32);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i) s2u[i] = unsortedIdOf(i);
    R.bS2U = R.mk(dev, s2u);

    // A SINGLE-CELL grid, so gridHash is 0 by construction.
    //
    // A 2x2x2 grid with cell 0 empty was tried and reverted, and the reason is
    // worth recording: the cell a work item reads is calcGridHashInBounds of the
    // pair's OVERLAP REGION, so the moment the two shapes' bounds differ the
    // fixture can no longer say which cell each work item lands in without
    // reimplementing that helper -- an oracle that transcribes the code under
    // test a second time.  With one cell the question does not arise.
    //
    // P9.18 IS that tranche, and this is what it changed.  The paragraph above
    // used to end "the grid-hash path is not pinned by this fixture", and the fix
    // it named -- a host mirror of gridCal.cuh, an ENGINE header rather than a
    // second transcription of ours -- is now in place above.  So the grid is TWO
    // cells wide and the two work items of a pair land on DIFFERENT ones: the
    // oracle no longer needs to assume which cell a work item reads, it computes
    // it the way the engine does.
    R.bCellStart = R.mk(dev, std::vector<uint32_t>{ CELL_BOUNDS[0], CELL_BOUNDS[1],
                                                    CELL_BOUNDS[2], CELL_BOUNDS[3] });
    R.bCellEnd = R.mk(dev, std::vector<uint32_t>{ CELL_BOUNDS[1], CELL_BOUNDS[2],
                                                  CELL_BOUNDS[3], CELL_BOUNDS[4] });

    // The filter machinery: one pair naming (node, UNSORTED id).
    R.bRunsum = R.mk(dev, std::vector<uint32_t>{ 0u });
    std::vector<PxgRigidFilterPair> pairs(1);
    std::memset(pairs.data(), 0, sizeof(PxgRigidFilterPair));
    pairs[0].index0 = uint64_t(NODE_INDEX);
    pairs[0].index1 = uint64_t(FILTERED_UNSORTED);
    pairs[0].index2 = 0u;
    R.bFilterPairs = R.mk(dev, pairs);
    std::vector<PxgParticleSimBuffer> simBufs(1);
    std::memset(simBufs.data(), 0, sizeof(PxgParticleSimBuffer));
    simBufs[0].mFilterPairs = reinterpret_cast<PxParticleRigidFilterPair*>(size_t(R.bFilterPairs.addr));
    simBufs[0].mNumFilterPairs = 1u;
    R.bSimBuf = R.mk(dev, simBufs);

    std::vector<PxgParticleSystem> descArr(2);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));
    PxgParticleSystem& m = descArr[0];
    m.mCommonData.mGridCellWidth = GRID_CELL_WIDTH;
    m.mCommonData.mGridSizeX = GRID_SIZE[0];   // 2x1x1 -- see the two-cell note above
    m.mCommonData.mGridSizeY = GRID_SIZE[1];
    m.mCommonData.mGridSizeZ = GRID_SIZE[2];
    m.mCommonData.mNumParticles = NB_PARTICLES;
    m.mCommonData.mNumParticleBuffers = 1u;
    m.mData.mFlags = 0u;             // no eENABLE_SPECULATIVE_CCD
    m.mCellStart = reinterpret_cast<PxU32*>(size_t(R.bCellStart.addr));
    m.mCellEnd = reinterpret_cast<PxU32*>(size_t(R.bCellEnd.addr));
    m.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(R.bOrigin.addr));
    m.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bPred.addr));
    m.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(R.bS2U.addr));
    m.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(R.bRunsum.addr));
    m.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(R.bSimBuf.addr));

    PxgParticleSystem& n = descArr[1];
    n = m;                                    // same grid, same cells, same filter ...
    n.mData.mFlags = 8u;                      // ... and eENABLE_SPECULATIVE_CCD
    n.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bPredB.addr));
    R.bDesc = R.mk(dev, descArr);

    // ---- narrowphase descriptors: one pair, particle vs the hull ----
    std::vector<PxgShape> shapes(3);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[2].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[2].particleOrSoftbodyId = 1u;      // the CCD system
    shapes[1].type = PxU32(PxGeometryType::eCONVEXMESH);
    // A NON-UNIFORM scale in a ROTATED frame, and a rotated rigid transform: see
    // the derivation at VERT_H.  Both are 90 degrees about z, which keeps the
    // world hull axis-aligned (so the oracle stays a closed form) while making
    // every frame in the composition distinguishable.
    shapes[1].scale = PxMeshScale(PxVec3(SCALE_X, SCALE_Y, SCALE_Z),
                                  PxQuat(PxPi * 0.5f, PxVec3(0.0f, 0.0f, 1.0f)));
    shapes[1].hullOrMeshPtr = size_t(R.bHull.addr);
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> tcache(3);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(0.0f, BOX_CY, 0.0f),
                                      PxQuat(PxPi * 0.5f, PxVec3(0.0f, 0.0f, 1.0f)));
    R.bTc = R.mk(dev, tcache);

    // Pair 0: system A (no CCD) vs the hull.  Pair 1: system B (CCD) vs the same
    // hull, with the particle shape in slot 1 so the LoadShapePair swap runs too.
    std::vector<PxgContactManagerInput> cin(CVX_NB_PAIRS);
    std::memset(cin.data(), 0, cin.size() * sizeof(PxgContactManagerInput));
    cin[0].shapeRef0 = 0u;  cin[0].transformCacheRef0 = 0u;
    cin[0].shapeRef1 = 1u;  cin[0].transformCacheRef1 = 1u;
    cin[1].shapeRef0 = 1u;  cin[1].transformCacheRef0 = 1u;
    cin[1].shapeRef1 = 2u;  cin[1].transformCacheRef1 = 2u;
    // Pair 2: system A against the hull again, so the only thing that separates
    // it from pair 0 is its work-item count and its rest distance.
    cin[2].shapeRef0 = 0u;  cin[2].transformCacheRef0 = 0u;
    cin[2].shapeRef1 = 1u;  cin[2].transformCacheRef1 = 1u;
    R.bCin = R.mk(dev, cin);

    // DIFFERENT bounds per shape.  With one box for everything the overlap region
    // and the union are the same solid and X2's `combine` mutant is the identity;
    // these two overlap in a strictly smaller box than either.
    std::vector<float> boundsF(3 * 6, 0.0f);
    for (uint32_t sh = 0; sh < 3; ++sh)
        for (int k = 0; k < 3; ++k)
        { boundsF[size_t(sh) * 6 + k] = SHAPE_BMN[sh][k];
          boundsF[size_t(sh) * 6 + 3 + k] = SHAPE_BMX[sh][k]; }
    R.bBounds = R.mk(dev, boundsF);

    std::vector<float> cdist(3, 0.0f);
    cdist[0] = CONTACT_OFFSET;   // cDistance = 0.35 + 0 for pair 0 ...
    cdist[2] = CONTACT_OFFSET;   // ... and for pair 1
    R.bCd = R.mk(dev, cdist);
    // DIFFERENT rest distances, so a port that reads the wrong pair index lands
    // on a different penetration.
    R.bRd = R.mk(dev, std::vector<float>{ CVX_PAIR_REST[0], CVX_PAIR_REST[1], CVX_PAIR_REST[2] });
    R.bMat = R.mk(dev, std::vector<uint32_t>(4u, POISON_U32));
    // BOTH pairs own TWO cells, so `pos` and `workIndex` still part company from
    // work item 1 onward -- without that, restDistances[pos] and
    // restDistances[workIndex] are the same expression.  What changed in P9.18 is
    // that the two work items of a pair no longer resolve to the SAME cell: they
    // resolve to cells 1 and 0 respectively (the oracle computes which), so each
    // particle is produced ONCE and reading the wrong cell reads different
    // particles instead of the same ones.
    R.bStart = R.mk(dev, std::vector<uint32_t>{ CVX_WORK_OFFSETS[0], CVX_WORK_OFFSETS[1],
                                                CVX_WORK_OFFSETS[2] });
    R.bTotal = R.mk(dev, std::vector<uint32_t>{ CVX_TOTAL_WORK });
    static_assert(CVX_WORK_OFFSETS[1] == CVX_WORK_COUNT[0], "pair 1 starts where pair 0 ends");
    static_assert(CVX_WORK_OFFSETS[2] == CVX_WORK_COUNT[0] + CVX_WORK_COUNT[1], "pair 2 likewise");
    static_assert(CVX_TOTAL_WORK == CVX_WORK_COUNT[0] + CVX_WORK_COUNT[1] + CVX_WORK_COUNT[2],
                  "the dispatched total must be the sum of the per-pair counts");
    R.bRemap = R.mk(dev, std::vector<uint64_t>{ 0ull, uint64_t(NODE_INDEX), 0ull });

    // ---- writer outputs, poisoned past the live contacts ----
    std::vector<HostContact> contacts(MAX_CONTACTS + SLACK);
    for (auto& c : contacts)
    { c.normal_pen = F4{ POISON_F, POISON_F, POISON_F, POISON_F };
      c.rigidId = POISON_U64; c.particleId = POISON_U64; }
    R.bContacts = R.mk(dev, contacts);
    R.bCount = R.mk(dev, std::vector<uint32_t>{ 0u });
    R.bSbP = R.mk(dev, std::vector<uint64_t>(MAX_CONTACTS + SLACK, POISON_U64));
    R.bTbP = R.mk(dev, std::vector<uint32_t>(MAX_CONTACTS + SLACK, POISON_U32));
    R.bIbP = R.mk(dev, std::vector<uint32_t>(MAX_CONTACTS + SLACK, POISON_U32));
    R.bBr = R.mk(dev, std::vector<uint64_t>(MAX_CONTACTS + SLACK, POISON_U64));
    R.bTbr = R.mk(dev, std::vector<uint32_t>(MAX_CONTACTS + SLACK, POISON_U32));
    R.bIbr = R.mk(dev, std::vector<uint32_t>(MAX_CONTACTS + SLACK, POISON_U32));

    std::memset(&R.blob, 0, sizeof(R.blob));
    R.blob.numTests = CVX_NB_PAIRS;
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
    R.blob.w_particleContacts = R.bContacts.addr;
    R.blob.w_numTotalContacts = R.bCount.addr;
    R.blob.w_contactSortedByParticle = R.bSbP.addr;
    R.blob.w_tempContactByParticle = R.bTbP.addr;
    R.blob.w_contactIndexSortedByParticle = R.bIbP.addr;
    R.blob.w_contactByRigid = R.bBr.addr;
    R.blob.w_tempContactByRigid = R.bTbr.addr;
    R.blob.w_contactIndexSortedByRigid = R.bIbr.addr;
    R.blob.w_maxContacts = MAX_CONTACTS;
}

static void resetOutputs(PxgSlangComputeDevice* dev, Rig& R)
{
    std::vector<HostContact> contacts(MAX_CONTACTS + SLACK);
    for (auto& c : contacts)
    { c.normal_pen = F4{ POISON_F, POISON_F, POISON_F, POISON_F };
      c.rigidId = POISON_U64; c.particleId = POISON_U64; }
    dev->upload(R.bContacts.h, contacts.data(), contacts.size() * sizeof(HostContact));
    const uint32_t zero = 0u;
    dev->upload(R.bCount.h, &zero, sizeof(uint32_t));
    std::vector<uint64_t> p64(MAX_CONTACTS + SLACK, POISON_U64);
    std::vector<uint32_t> p32(MAX_CONTACTS + SLACK, POISON_U32);
    dev->upload(R.bSbP.h, p64.data(), p64.size() * sizeof(uint64_t));
    dev->upload(R.bBr.h, p64.data(), p64.size() * sizeof(uint64_t));
    dev->upload(R.bTbP.h, p32.data(), p32.size() * sizeof(uint32_t));
    dev->upload(R.bIbP.h, p32.data(), p32.size() * sizeof(uint32_t));
    dev->upload(R.bTbr.h, p32.data(), p32.size() * sizeof(uint32_t));
    dev->upload(R.bIbr.h, p32.data(), p32.size() * sizeof(uint32_t));
}

// One expected contact: which system it came from, which particle, and the rest
// distance of that system's pair.
struct Want { uint32_t sys; uint32_t particle; float rest; };

// The CCD acceptance test, PARAMETERISED by the two things the mutants change:
// the forward-projection scale (1 for isDiffuse, 2 for a TGS non-diffuse) and the
// factor on cVolumeRadius in the second pass's minSep (2 in the engine).  The
// oracle calls it with the engine's values; the coverage block calls it with the
// mutants' and DEMANDS the two answers differ at the constructed probes -- which
// is what "this probe discriminates" means, stated rather than hoped.
static bool ccdAccepted(uint32_t i, float scaleInv, float minSepFactor)
{
    float p[3], dir[3]; particlePos(i, p);
    const float sdOrigin = boxSignedDistance(p, dir);
    float q[3]; particlePredicted(i, q);
    float off[3], mid[3];
    float m2 = 0.0f;
    for (int k = 0; k < 3; ++k)
    { off[k] = (q[k] - p[k]) * scaleInv * 0.5f; mid[k] = p[k] + off[k]; m2 += off[k] * off[k]; }
    const float cVolumeRadius = std::sqrt(m2) + CONTACT_OFFSET;
    float midDir[3];
    if (boxSignedDistance(mid, midDir) > cVolumeRadius) return false;              // pass 1
    if (sdOrigin > CONTACT_OFFSET + minSepFactor * cVolumeRadius) return false;    // pass 2
    return true;
}

// Which pair owns work item `w` -- the binary search the kernel does over the
// per-pair offsets, spelled as a linear scan because three is three.
static uint32_t cvxPairOfWork(uint32_t w)
{
    uint32_t pair = 0u;
    for (uint32_t k = 0; k < CVX_NB_PAIRS; ++k) if (w >= CVX_WORK_OFFSETS[k]) pair = k;
    return pair;
}

// Which cell work item `w` reads, computed the way the ENGINE computes it.
static uint32_t cvxCellOfWork(uint32_t w)
{
    const uint32_t pair = cvxPairOfWork(w);
    return gcCellOfWorkItem(CVX_PAIR_SHAPE_A[pair], CVX_PAIR_SHAPE_B[pair],
                            CONTACT_OFFSET, w - CVX_WORK_OFFSETS[pair]);
}

// Which (system, particle) pairs must produce a contact, analytically.
//   system 0: no CCD -> accepted when signedDistance(origin) <= cDistance.
//   system 1: CCD    -> the two-pass test spelled out at particlePredicted.
//
// P9.18 REWROTE THIS TO ITERATE WORK ITEMS, not systems.  A work item reads ONE
// cell -- the one calcGridHashInBounds names -- and the old version could ignore
// that only because every work item read cell 0.  Now the cell is derived, so a
// port that mis-decodes the hash lands on the other cell's particles and the
// multiset stops matching.
static std::vector<Want> expectedSet()
{
    std::vector<Want> out;
    for (uint32_t w = 0; w < CVX_TOTAL_WORK; ++w)
    {
        const uint32_t pair = cvxPairOfWork(w);
        const uint32_t sys = CVX_PAIR_SYS[pair];
        const float rest = CVX_PAIR_REST[pair];
        uint32_t lo = 0u, hi = 0u;
        gcCellSpan(cvxCellOfWork(w), lo, hi);
        for (uint32_t i = lo; i < hi; ++i)
        {
            if (i == FILTERED_SORTED) continue;              // X9, on BOTH systems
            float p[3], dir[3]; particlePos(i, p);
            const float sdOrigin = boxSignedDistance(p, dir);
            if (sys == 0u)
            {
                if (sdOrigin <= CONTACT_OFFSET) out.push_back(Want{ sys, i, rest });
                continue;
            }
            // isDiffuse = TRUE -> scaleInv 1; the engine's second-pass factor is 2.
            if (!ccdAccepted(i, 1.0f, 2.0f)) continue;
            out.push_back(Want{ sys, i, rest });
        }
    }
    return out;
}

static void checkRun(PxgSlangComputeDevice* dev, const Rig& R, const char* label,
                     double& worstPenErr, double& worstNormErr)
{
    const std::vector<Want> want = expectedSet();
    const auto count = downloadBuffer<uint32_t>(dev, R.bCount.h, 1);
    const auto contacts = downloadBuffer<HostContact>(dev, R.bContacts.h, MAX_CONTACTS + SLACK);
    const auto sbp = downloadBuffer<uint64_t>(dev, R.bSbP.h, MAX_CONTACTS + SLACK);
    const auto tbp = downloadBuffer<uint32_t>(dev, R.bTbP.h, MAX_CONTACTS + SLACK);
    const auto ibp = downloadBuffer<uint32_t>(dev, R.bIbP.h, MAX_CONTACTS + SLACK);
    const auto br = downloadBuffer<uint64_t>(dev, R.bBr.h, MAX_CONTACTS + SLACK);

    // The COUNTER counts every attempt and is NOT capped; only the WRITE is.
    PXG_CHECK_MSG(count[0] == uint32_t(want.size()),
                  "%s: the contact counter reads %u, expected %zu -- it must count every "
                  "attempt, including the ones past maxContacts",
                  label, count[0], want.size());
    if (count[0] != uint32_t(want.size())) return;
    PXG_CHECK_MSG(count[0] > MAX_CONTACTS,
                  "%s: only %u contacts were attempted against a cap of %u -- the capacity "
                  "gate never fires and a port that caps the counter too is invisible",
                  label, count[0], MAX_CONTACTS);
    const uint32_t written = std::min(count[0], MAX_CONTACTS);

    // The slot each contact lands in is the atomicAdd's order, i.e. a race, so the
    // emitted contacts are matched as a SET against the expected particles.
    std::vector<bool> claimed(want.size(), false);
    size_t bad = 0;
    for (uint32_t c = 0; c < written; ++c)
    {
        // PxEncodeParticleIndex packs (systemId << 32) | particleIndex, so the
        // record names BOTH halves and the two systems cannot be confused.
        const uint32_t pid = uint32_t(contacts[c].particleId & 0xffffffffull);
        const uint32_t sys = uint32_t(contacts[c].particleId >> 32);
        size_t at = want.size();
        for (size_t w = 0; w < want.size(); ++w)
            if (!claimed[w] && want[w].particle == pid && want[w].sys == sys) { at = w; break; }
        if (at == want.size())
        { if (bad < 4) std::printf("    %s contact %u: (system %u, particle %u) is not an "
                                   "unclaimed expected contact\n", label, c, sys, pid);
          ++bad; continue; }
        claimed[at] = true;
        const float rest = want[at].rest;

        float p[3], outDir[3]; particlePos(pid, p);
        const float sd = boxSignedDistance(p, outDir);
        const F4 np = contacts[c].normal_pen;

        // 1. the normal is a unit vector
        const double nlen = std::sqrt(double(np.x) * np.x + double(np.y) * np.y +
                                      double(np.z) * np.z);
        worstNormErr = std::max(worstNormErr, std::fabs(nlen - 1.0));
        if (std::fabs(nlen - 1.0) > 1e-4)
        { if (bad < 4) std::printf("    %s particle %u: |normal| = %g\n", label, pid, nlen);
          ++bad; continue; }

        // 2. it points OUT of the box -- away from the surface when the particle is
        //    outside, and toward the NEAREST FACE when it is inside (which is the
        //    direction EPA's minimum translation must take).  This is what pins
        //    X7's negation PHYSICALLY: a port that stores +direction pushes the
        //    particle deeper in, and the check does not depend on any sign read
        //    out of a measured run.
        const double dotOut = double(np.x) * outDir[0] + double(np.y) * outDir[1] +
                              double(np.z) * outDir[2];
        if (dotOut < 0.9)
        { if (bad < 4) std::printf("    %s particle %u (signed dist %g): normal (%g,%g,%g) is "
                                   "not the OUTWARD direction (%g,%g,%g), cos = %g -- the "
                                   "correction would push it further in\n", label, pid, sd,
                                   np.x, np.y, np.z, outDir[0], outDir[1], outDir[2], dotOut);
          ++bad; continue; }

        // 3. the penetration is the SIGNED distance less the rest distance -- which
        //    checks GJK's separation for the outside particles and EPA's depth for
        //    the inside ones, both against the closed form.
        const double err = std::fabs(double(np.w) - (double(sd) - rest));
        worstPenErr = std::max(worstPenErr, err);
        if (err > 2e-4)
        { if (bad < 4) std::printf("    %s sys %u particle %u: pen %g, expected %g "
                                   "(signed dist %g)\n", label, sys, pid, np.w, sd - rest, sd);
          ++bad; continue; }

        // 4. the identities and the six staging arrays
        if (contacts[c].rigidId != uint64_t(NODE_INDEX)) ++bad;
        if (sbp[c] != contacts[c].particleId) ++bad;
        if (tbp[c] != pid) ++bad;
        if (ibp[c] != c) ++bad;
        if (br[c] != uint64_t(NODE_INDEX)) ++bad;
    }
    PXG_CHECK_MSG(bad == 0, "%s: %zu contacts failed the geometric or identity checks",
                  label, bad);

    // The cap drops whichever contacts lose the atomicAdd race, so the WRITTEN set
    // must still be shown to span both systems -- otherwise a whole family could
    // be wrong and never compared.
    uint32_t wroteA = 0, wroteB = 0;
    for (uint32_t c = 0; c < written; ++c)
        ((contacts[c].particleId >> 32) ? wroteB : wroteA)++;
    PXG_CHECK_MSG(wroteA > 0 && wroteB > 0,
                  "%s: the written contacts came from one system only (%u from the plain "
                  "system, %u from the CCD one) -- the cap is hiding a whole family",
                  label, wroteA, wroteB);

    // Nothing past the CAP may have been touched -- which is also where the
    // attempts past maxContacts would have landed had the gate not held.
    size_t badPoison = 0;
    for (uint32_t c = written; c < MAX_CONTACTS + SLACK; ++c)
    {
        if (contacts[c].rigidId != POISON_U64 || contacts[c].particleId != POISON_U64) ++badPoison;
        if (sbp[c] != POISON_U64 || tbp[c] != POISON_U32 || ibp[c] != POISON_U32) ++badPoison;
    }
    PXG_CHECK_MSG(badPoison == 0, "%s: %zu slots past the emitted contacts were written "
                  "(the %u-slot margin past maxContacts is compared too)",
                  label, badPoison, SLACK);
}

static void runAll(PxgSlangComputeDevice* dev)
{
    Rig R;
    buildRig(dev, R);

    Kernel k("psContactGenBda.spv", "ps_convexCollisionLaunch", {}, sizeof(PsCollideBlob));
    PXG_CHECK_MSG(k.isValid(), "the convex contact-gen pipeline must build");
    if (!k.isValid()) { R.destroy(dev); return; }

    double worstPenErr = 0.0, worstNormErr = 0.0;
    // X1: PGS and TGS must agree, because this kernel passes isDiffuse = true.
    for (int mode = 0; mode < 2; ++mode)
    {
        const char* label = mode ? "TGS" : "PGS";
        resetOutputs(dev, R);
        dev->flush();
        R.blob.isTGSWord = mode ? 1u : 0u;
        k.dispatch(1024u, {}, R.blob, 1u, 1u);   // grid (1024,1,1), block (32,8,1)
        dev->flush();
        checkRun(dev, R, label, worstPenErr, worstNormErr);
    }

    // ---- coverage, demanded rather than designed ----
    {
        const std::vector<Want> want = expectedSet();
        uint32_t separated = 0, penetrating = 0, sideFace = 0, filtered = 0;
        uint32_t sysA = 0, sysB = 0;
        for (const Want& w : want) (w.sys ? sysB : sysA)++;
        // The CCD system must accept a DIFFERENT set than the non-CCD one, or the
        // whole contact-volume policy is untested.
        // Each constructed probe must SEPARATE the engine's policy from the mutant
        // it was built for.  Which side of the line it falls on does not matter;
        // that the two sides disagree is the whole point.
        const bool a1 = ccdAccepted(MARGINAL_ISDIFFUSE, 1.0f, 2.0f);
        const bool a2 = ccdAccepted(MARGINAL_ISDIFFUSE, 2.0f, 2.0f);
        PXG_CHECK_MSG(a1 != a2, "particle %u does not separate isDiffuse=true (accepted: %d) "
                      "from isDiffuse=false in a TGS step (accepted: %d) -- the contact-volume "
                      "scale is unprobed", MARGINAL_ISDIFFUSE, int(a1), int(a2));
        const bool b1 = ccdAccepted(MARGINAL_MINSEP, 1.0f, 2.0f);
        const bool b2 = ccdAccepted(MARGINAL_MINSEP, 1.0f, 1.0f);
        PXG_CHECK_MSG(b1 != b2, "particle %u does not separate the second pass's "
                      "cDistance + 2*cVolumeRadius (accepted: %d) from cDistance + "
                      "cVolumeRadius (accepted: %d)", MARGINAL_MINSEP, int(b1), int(b2));
        PXG_CHECK_MSG(sysA > 0 && sysB > 0 && sysA != sysB,
                      "the two systems accept the same number of contacts (%u and %u) -- the "
                      "CCD contact-volume policy is not being exercised", sysA, sysB);
        for (uint32_t i = CELL_START; i < CELL_END; ++i)
        {
            float p[3], dir[3]; particlePos(i, p);
            const float sd = boxSignedDistance(p, dir);
            if (i == FILTERED_SORTED) { if (sd <= CONTACT_OFFSET) ++filtered; continue; }
            if (sd > CONTACT_OFFSET) continue;
            if (sd > 0.0f) ++separated; else ++penetrating;
            if ((i % 4u) == 3u) ++sideFace;
        }
        PXG_CHECK_MSG(!want.empty(), "the analytic contact set is EMPTY -- the fixture "
                      "measures nothing");
        PXG_CHECK_MSG(separated > 0 && penetrating > 0,
                      "the contact set is all one kind (%u separated, %u penetrating) -- GJK "
                      "and EPA are not both exercised", separated, penetrating);
        PXG_CHECK_MSG(sideFace > 0, "no particle is off the box's side -- every contact is "
                      "against the top face and a wrong closest-feature is invisible");
        PXG_CHECK_MSG(filtered == 1u, "the seeded filter pair suppresses %u contacts, not 1 "
                      "-- X9's unsorted-index key is untested", filtered);

        // ---- P9.18: the grid hash has to be OBSERVABLE, not merely computed ----
        {
            const uint32_t h0 = cvxCellOfWork(0u), h1 = cvxCellOfWork(1u);
            PXG_CHECK_MSG(h0 < 2u && h1 < 2u,
                          "a work item resolves to cell %u/%u, outside the two this fixture "
                          "uploads -- the oracle would be reading past mCellStart", h0, h1);
            PXG_CHECK_MSG(h0 != h1,
                          "both of pair 0's work items land on cell %u -- with one cell for "
                          "the whole pair, returning a constant hash is indistinguishable "
                          "and calcGridHashInBounds is unprobed (this is exactly the state "
                          "P9.18 found)", h0);
            // Every cell must be REACHED, and no two may look alike -- a cell no
            // work item reads is dead weight, and two cells with the same accepted
            // count make swapping them invisible.
            bool reached[NB_CELLS] = { false, false, false, false };
            for (uint32_t w = 0; w < CVX_TOTAL_WORK; ++w) reached[cvxCellOfWork(w)] = true;
            uint32_t acc[NB_CELLS] = { 0u, 0u, 0u, 0u };
            for (uint32_t h = 0; h < NB_CELLS; ++h)
            {
                PXG_CHECK_MSG(reached[h], "cell %u is read by no work item -- it cannot "
                              "discriminate anything", h);
                uint32_t lo, hi; gcCellSpan(h, lo, hi);
                for (uint32_t i = lo; i < hi; ++i)
                {
                    if (i == FILTERED_SORTED) continue;
                    float p[3], dir[3]; particlePos(i, p);
                    if (boxSignedDistance(p, dir) <= CONTACT_OFFSET) acc[h]++;
                }
            }
            for (uint32_t h = 0; h < NB_CELLS; ++h)
                for (uint32_t k = h + 1u; k < NB_CELLS; ++k)
                    PXG_CHECK_MSG(acc[h] != acc[k],
                                  "cells %u and %u both accept %u contacts -- exchanging them "
                                  "would be unobservable", h, k, acc[h]);
            // The three pairs must read three DIFFERENT cell sets.  If any two
            // agree, the asymmetry that makes a relabelling of the mapping
            // visible is gone for that pair, and a permuting mutant lives.
            {
                uint32_t mask[CVX_NB_PAIRS] = { 0u, 0u, 0u };
                for (uint32_t w = 0; w < CVX_TOTAL_WORK; ++w)
                    mask[cvxPairOfWork(w)] |= (1u << cvxCellOfWork(w));
                for (uint32_t a = 0; a < CVX_NB_PAIRS; ++a)
                {
                    PXG_CHECK_MSG(mask[a] != 0xFu || a == 0u,
                                  "pair %u reads every cell -- a pure relabelling of the cell "
                                  "mapping leaves its union unchanged and cannot be seen", a);
                    for (uint32_t b = a + 1u; b < CVX_NB_PAIRS; ++b)
                        PXG_CHECK_MSG(mask[a] != mask[b],
                                      "pairs %u and %u read the same cell set (0x%x) -- one of "
                                      "them is not adding discrimination", a, b, mask[a]);
                }
            }
            // The oracle floors positions to cells; if any bound sits on a cell
            // boundary the floor is a knife edge and the oracle stops being exact.
            float omn[3], omx[3];
            gcOverlapBound(0u, 1u, CONTACT_OFFSET, omn, omx);
            for (int k = 0; k < 3; ++k)
            {
                const float fr[2] = { omn[k] / GRID_CELL_WIDTH, omx[k] / GRID_CELL_WIDTH };
                for (int e = 0; e < 2; ++e)
                {
                    const float d = std::fabs(fr[e] - std::floor(fr[e]));
                    PXG_CHECK_MSG(d > 0.05f && d < 0.95f,
                                  "overlap bound component %d edge %d sits %.3g of a cell from "
                                  "a boundary -- the floor is a knife edge and this oracle is "
                                  "no longer exact", k, e, double(d));
                }
            }
            std::printf("    [ note ] grid %ux%ux%u, cell width %.2f, %u cells; work items "
                        "read cells", GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2],
                        double(GRID_CELL_WIDTH), NB_CELLS);
            for (uint32_t w = 0; w < CVX_TOTAL_WORK; ++w)
            {
                bool brk = false;
                for (uint32_t k = 1; k < CVX_NB_PAIRS; ++k) brk |= (w == CVX_WORK_OFFSETS[k]);
                std::printf("%s%u", brk ? " | " : " ", cvxCellOfWork(w));
            }
            std::printf("; cells accept %u %u %u %u\n", acc[0], acc[1], acc[2], acc[3]);
        }
        std::printf("    [ note ] %zu contacts (%u no-CCD, %u CCD) over %u cell particles; "
                    "%u separated, %u penetrating, %u off the side, 1 filtered per system; "
                    "worst |normal|-1 %.3g, worst penetration error %.3g\n",
                    want.size(), sysA, sysB, CELL_END - CELL_START,
                    separated, penetrating, sideFace, worstNormErr, worstPenErr);
    }

    R.destroy(dev);
}

// ===========================================================================
// P9.14c -- ps_convexDiffuseCollisionLaunch, the DIFFUSE twin.
//
// A second PXG_TEST in the SAME translation unit, and deliberately so: the
// geometry, the hull blob, the closed-form box oracle and the CCD acceptance
// predicate above are EXACTLY the ones this kernel needs, and a second file
// would mean a second transcription of the thing that has to be right.  One
// oracle, two users -- the P8.85 arrangement.
//
// What the diffuse rig adds, and what each addition is there to kill:
//
//   * the DIFFUSE arrays (XD1/XD2) -- cell start/end, sorted origin/predicted --
//     wired to buffers the FLUID fields do not point at, so swapping a field for
//     its fluid neighbour reads a different solid entirely;
//   * the per-particle OUTPUT (XD4) with a COLUMN-MAJOR stride of
//     *mNumDiffuseParticles, DISTINCT per system (47 / 53 / 41 / 59) and
//     distinct from every particle count, cell bound and work-item count in the
//     fixture, so a wrong stride lands on a poisoned slot;
//   * pair 0 owning FOURTEEN work items over the same cell, so every accepted
//     particle's counter reads 14 against a cap of 12 -- the overshoot contract
//     (D7 / X10 / XD4) measured directly, with slots 12 and 13 required back at
//     poison.  All fourteen contacts for a particle are IDENTICAL, so unlike
//     P9.14b's global writer the cap here hides nothing;
//   * pair 1 the CCD system, THREE work items, entirely below the cap, so the
//     uncapped path is measured in the same run;
//   * NO FILTER (XD3): the rigid filter pair, the sorted-to-unsorted map and the
//     sim-buffer chain are all wired up and the pair still names
//     particle FILTERED_SORTED -- and this kernel must produce a contact for it
//     anyway.  A port that copies the fluid twin's findPs call suppresses it and
//     dies;
//   * a TRAP system with mMaxDiffuseParticles == 0 and everything else live --
//     live cell arrays, live positions, live output arrays -- so the guard is
//     pinned by its OUTPUT staying poison rather than by a null dereference;
//   * and the shape that makes XD6 measurable at all: 8193 work items, so
//     nbIterationsPerWarps is 2 and each warp owns a CONTIGUOUS PAIR.  The trap
//     pair's run ends on an EVEN index and the last system's begins on the odd
//     one directly after it, so warp 4095 meets the trap first.  The engine's
//     `return` abandons its partner; a `continue` would process it.  System C is
//     therefore reached exactly ONCE, by warp 4096, and its counter must read 1.
//     Two is the mutant.  Without nbIter > 1 the two spellings are literally
//     indistinguishable, which is why the fixture pays for the extra work items.
// ===========================================================================
namespace diffuse
{
static_assert(offsetof(PxgParticleSystem, mDiffuseCellStart) == 520, "@520");
static_assert(offsetof(PxgParticleSystem, mDiffuseCellEnd) == 528, "@528");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) == 552, "@552");
static_assert(offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) == 560, "@560");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactInfos) == 592, "@592");
static_assert(offsetof(PxgParticleSystem, mDiffuseOneWayContactCount) == 616, "@616");

// PxgParticleContactInfo::MaxStaticContactsPerParticle
static const uint32_t MAX_STATIC = 12u;

// The work-item layout.  PS_COLLISION's grid x warps-per-block is 1024 * 8 =
// 8192, so 8193 items is the SMALLEST total for which nbIterationsPerWarps is 2
// -- and two is all XD6 needs.
static const uint32_t TOTAL_WORK = 8193u;
static const uint32_t RUN_A = 0u;        // pair 0: system A, items 0..13   (14)
static const uint32_t RUN_B = 14u;       // pair 1: system B, items 14..16  (3)
static const uint32_t RUN_Z = 17u;       // pair 2: the trap, items 17..8190
static const uint32_t RUN_C = 8191u;     // pair 3: system C, items 8191..8192
static const uint32_t WORK_A = RUN_B - RUN_A;   // 14, ABOVE the cap of 12
static const uint32_t WORK_B = RUN_Z - RUN_B;   // 3,  below it
// XD6: item 8190 is the trap's LAST and it is even, so warp 4095 owns
// { 8190, 8191 } and the engine's `return` throws 8191 away.  Warp 4096 owns
// { 8192, 8193 }; 8193 is past the end.  So C is reached exactly once.
static const uint32_t WORK_C = 1u;
static_assert((RUN_C - 1u) % 2u == 0u, "the trap's last item must be even");

// The published diffuse counts -- the output STRIDE.  Four distinct values, none
// equal to any particle count, cell bound, work-item count or slot cap above.
static const uint32_t NUM_A = 47u;
static const uint32_t NUM_B = 53u;
static const uint32_t NUM_Z = 41u;
static const uint32_t NUM_C = 59u;

static const float REST_C = 0.061f;      // pair 3, a third distinct rest distance
static const float REST_Z = 0.05f;

static inline uint32_t infoSlots(uint32_t num) { return MAX_STATIC * num + SLACK; }

struct Sys
{
    const char* name;
    uint32_t id;          // psId
    uint32_t num;         // *mNumDiffuseParticles
    uint32_t work;        // how many work items reach it
    float rest;
    bool ccd;
    bool trap;
    DevBuf bInfos, bForces, bNodes, bCounts, bNum, bPred;
};

struct DRig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bHull, bCin, bShapes, bTc, bBounds, bCd, bRd, bMat, bStart, bTotal, bRemap;
    DevBuf bCellStart, bCellEnd, bOrigin, bS2U, bRunsum, bSimBuf, bFilterPairs, bDesc;
    DevBuf bFluidCellStart, bFluidCellEnd, bFluidPos;
    Sys sys[4];
    PsCollideBlob blob{};

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

static void makeOutputs(PxgSlangComputeDevice* dev, DRig& R, Sys& s)
{
    s.bInfos = R.mk(dev, std::vector<F4>(infoSlots(s.num),
                                         F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    s.bForces = R.mk(dev, std::vector<float>(infoSlots(s.num), POISON_F));
    s.bNodes = R.mk(dev, std::vector<uint64_t>(infoSlots(s.num), POISON_U64));
    // The counters START at zero -- they are atomicAdd targets, and a poisoned
    // one would return a slot past the cap and hide every write.  What stands in
    // for poison here is that the set of NON-ZERO counters must be exactly the
    // accepted particles: a wrong index lands on a zero that must have stayed
    // zero.
    s.bCounts = R.mk(dev, std::vector<uint32_t>(s.num + SLACK, 0u));
    s.bNum = R.mk(dev, std::vector<int32_t>{ int32_t(s.num), int32_t(s.num) * 3 });
}

static void resetOutputs(PxgSlangComputeDevice* dev, DRig& R)
{
    for (Sys& s : R.sys)
    {
        std::vector<F4> infos(infoSlots(s.num), F4{ POISON_F, POISON_F, POISON_F, POISON_F });
        std::vector<float> forces(infoSlots(s.num), POISON_F);
        std::vector<uint64_t> nodes(infoSlots(s.num), POISON_U64);
        std::vector<uint32_t> counts(s.num + SLACK, 0u);
        dev->upload(s.bInfos.h, infos.data(), infos.size() * sizeof(F4));
        dev->upload(s.bForces.h, forces.data(), forces.size() * sizeof(float));
        dev->upload(s.bNodes.h, nodes.data(), nodes.size() * sizeof(uint64_t));
        dev->upload(s.bCounts.h, counts.data(), counts.size() * sizeof(uint32_t));
    }
}

static void buildRig(PxgSlangComputeDevice* dev, DRig& R)
{
    R.bHull = R.mk(dev, makeHullBlob());

    R.sys[0] = Sys{ "A",     0u, NUM_A, WORK_A, REST_DISTANCE,   false, false, {}, {}, {}, {}, {}, {} };
    R.sys[1] = Sys{ "B/CCD", 1u, NUM_B, WORK_B, REST_DISTANCE_B, true,  false, {}, {}, {}, {}, {}, {} };
    R.sys[2] = Sys{ "trap",  2u, NUM_Z, 0u,     REST_Z,          false, true,  {}, {}, {}, {}, {}, {} };
    R.sys[3] = Sys{ "C",     3u, NUM_C, WORK_C, REST_C,          false, false, {}, {}, {}, {}, {}, {} };

    // ---- the DIFFUSE position arrays, shared by every system ----
    std::vector<F4> origin(NB_PARTICLES + SLACK, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    std::vector<F4> flat = origin, ccd = origin;
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
    {
        float p[3]; particlePos(i, p);
        origin[i] = F4{ p[0], p[1], p[2], 3.5f };   // .w is the LIFETIME here
        flat[i] = origin[i];                        // CCD off -> predicted == origin
        float q[3]; particlePredicted(i, q);
        ccd[i] = F4{ q[0], q[1], q[2], 3.5f };
    }
    R.bOrigin = R.mk(dev, origin);
    const DevBuf bFlat = R.mk(dev, flat);
    const DevBuf bCcd = R.mk(dev, ccd);
    for (Sys& s : R.sys) s.bPred = s.ccd ? bCcd : bFlat;

    // The same 1x1x1 single-cell grid P9.14b uses, and for the same reason (see
    // bCellStart there): with more than one cell the fixture cannot say which
    // cell a work item reads without reimplementing calcGridHashInBounds.
    R.bCellStart = R.mk(dev, std::vector<uint32_t>{ CELL_START });
    R.bCellEnd = R.mk(dev, std::vector<uint32_t>{ CELL_END });

    // THE FLUID FIELDS ARE WIRED TOO, TO A DIFFERENT AND HARMLESS CONFIGURATION.
    //
    // The first draft left mCellStart / mSortedOriginPos_InvMass NULL so a port
    // that read them would fault.  A fault IS a kill, but it is the weakest kind:
    // it depends on the allocator, it can be flaky under llvmpipe, and it says
    // nothing about which field was read.  These arrays are live instead -- a
    // three-particle cell of particles parked far from the hull -- so a port that
    // reads the fluid grid or the fluid positions produces a DETERMINISTIC
    // DIFFERENT contact set and dies cleanly.
    //
    // The fluid cell's bounds matter: the first draft used [0, 3), and a port
    // that read the fluid END against the diffuse START got range = 3 - 4, which
    // UNDERFLOWS to 4294967295 and walks off the array -- a crash, i.e. the weak
    // kind of kill again.  [0, 6) straddles CELL_START instead, so every mix of
    // the two grids is a small, valid, WRONG particle span.
    R.bFluidCellStart = R.mk(dev, std::vector<uint32_t>{ 0u });
    R.bFluidCellEnd = R.mk(dev, std::vector<uint32_t>{ 6u });
    R.bFluidPos = R.mk(dev, std::vector<F4>(NB_PARTICLES + SLACK,
                                            F4{ 100.0f, 100.0f, 100.0f, 1.0f }));

    std::vector<uint32_t> s2u(NB_PARTICLES + SLACK, POISON_U32);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i) s2u[i] = unsortedIdOf(i);
    R.bS2U = R.mk(dev, s2u);

    // XD3: the whole filter chain is wired and the pair still names
    // FILTERED_SORTED's unsorted id -- and this kernel must IGNORE it.
    R.bRunsum = R.mk(dev, std::vector<uint32_t>{ 0u });
    std::vector<PxgRigidFilterPair> pairs(1);
    std::memset(pairs.data(), 0, sizeof(PxgRigidFilterPair));
    pairs[0].index0 = uint64_t(NODE_INDEX);
    pairs[0].index1 = uint64_t(FILTERED_UNSORTED);
    pairs[0].index2 = 0u;
    R.bFilterPairs = R.mk(dev, pairs);
    std::vector<PxgParticleSimBuffer> simBufs(1);
    std::memset(simBufs.data(), 0, sizeof(PxgParticleSimBuffer));
    simBufs[0].mFilterPairs = reinterpret_cast<PxParticleRigidFilterPair*>(size_t(R.bFilterPairs.addr));
    simBufs[0].mNumFilterPairs = 1u;
    R.bSimBuf = R.mk(dev, simBufs);

    for (Sys& s : R.sys) makeOutputs(dev, R, s);

    std::vector<PxgParticleSystem> descArr(4);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));
    for (uint32_t k = 0; k < 4u; ++k)
    {
        PxgParticleSystem& m = descArr[k];
        Sys& s = R.sys[k];
        m.mCommonData.mGridCellWidth = 1.0f;
        m.mCommonData.mGridSizeX = 1u;
        m.mCommonData.mGridSizeY = 1u;
        m.mCommonData.mGridSizeZ = 1u;
        m.mCommonData.mNumParticles = NB_PARTICLES;
        m.mCommonData.mNumParticleBuffers = 1u;
        // XD6: the trap's ONE difference.  Everything below it is live.
        m.mCommonData.mMaxDiffuseParticles = s.trap ? 0u : 4096u;
        m.mData.mFlags = s.ccd ? 8u : 0u;   // eENABLE_SPECULATIVE_CCD
        // XD1/XD2: the DIFFUSE fields, beside live FLUID ones that point somewhere
        // else entirely (see bFluidCellStart) -- so reading a field's fluid twin
        // is a clean empty result rather than a fault or a lucky coincidence.
        m.mCellStart = reinterpret_cast<PxU32*>(size_t(R.bFluidCellStart.addr));
        m.mCellEnd = reinterpret_cast<PxU32*>(size_t(R.bFluidCellEnd.addr));
        m.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(R.bFluidPos.addr));
        m.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bFluidPos.addr));
        m.mDiffuseCellStart = reinterpret_cast<PxU32*>(size_t(R.bCellStart.addr));
        m.mDiffuseCellEnd = reinterpret_cast<PxU32*>(size_t(R.bCellEnd.addr));
        m.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bOrigin.addr));
        m.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(s.bPred.addr));
        m.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(s.bNum.addr));
        m.mDiffuseOneWayContactInfos =
            reinterpret_cast<PxgParticleContactInfo*>(size_t(s.bInfos.addr));
        m.mDiffuseOneWayForces = reinterpret_cast<float*>(size_t(s.bForces.addr));
        m.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(s.bNodes.addr));
        m.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(s.bCounts.addr));
        m.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(R.bS2U.addr));
        m.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(R.bRunsum.addr));
        m.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(R.bSimBuf.addr));
    }
    R.bDesc = R.mk(dev, descArr);

    // ---- shapes: four particle systems and one hull, all with distinct cache
    //      refs so contactDistance / transformCache / the remap cannot be
    //      confused ----
    std::vector<PxgShape> shapes(5);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    const uint32_t psShape[4] = { 0u, 2u, 3u, 4u };
    for (uint32_t k = 0; k < 4u; ++k)
    {
        shapes[psShape[k]].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
        shapes[psShape[k]].particleOrSoftbodyId = R.sys[k].id;
    }
    shapes[1].type = PxU32(PxGeometryType::eCONVEXMESH);
    shapes[1].scale = PxMeshScale(PxVec3(SCALE_X, SCALE_Y, SCALE_Z),
                                  PxQuat(PxPi * 0.5f, PxVec3(0.0f, 0.0f, 1.0f)));
    shapes[1].hullOrMeshPtr = size_t(R.bHull.addr);
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> tcache(5);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(0.0f, BOX_CY, 0.0f),
                                      PxQuat(PxPi * 0.5f, PxVec3(0.0f, 0.0f, 1.0f)));
    R.bTc = R.mk(dev, tcache);

    // Pairs 0 and 2 put the particle shape in slot 0; pairs 1 and 3 put it in
    // slot 1, so the LoadShapePair swap runs on both halves of the fixture.
    std::vector<PxgContactManagerInput> cin(4);
    std::memset(cin.data(), 0, cin.size() * sizeof(PxgContactManagerInput));
    cin[0].shapeRef0 = 0u; cin[0].transformCacheRef0 = 0u;
    cin[0].shapeRef1 = 1u; cin[0].transformCacheRef1 = 1u;
    cin[1].shapeRef0 = 1u; cin[1].transformCacheRef0 = 1u;
    cin[1].shapeRef1 = 2u; cin[1].transformCacheRef1 = 2u;
    cin[2].shapeRef0 = 3u; cin[2].transformCacheRef0 = 3u;
    cin[2].shapeRef1 = 1u; cin[2].transformCacheRef1 = 1u;
    cin[3].shapeRef0 = 1u; cin[3].transformCacheRef0 = 1u;
    cin[3].shapeRef1 = 4u; cin[3].transformCacheRef1 = 4u;
    R.bCin = R.mk(dev, cin);

    std::vector<float> boundsF(5 * 6, 0.0f);
    const float bmn[3] = { -9.0f, -3.0f, -7.0f };
    const float bmx[3] = { 6.0f, 9.0f, 4.0f };
    const float hmn[3] = { -4.0f, -8.0f, -5.0f };
    const float hmx[3] = { 8.0f, 5.0f, 9.0f };
    for (uint32_t sh = 0; sh < 5u; ++sh)
        for (int k = 0; k < 3; ++k)
        {
            const bool hull = (sh == 1u);
            boundsF[size_t(sh) * 6 + k] = hull ? hmn[k] : bmn[k];
            boundsF[size_t(sh) * 6 + 3 + k] = hull ? hmx[k] : bmx[k];
        }
    R.bBounds = R.mk(dev, boundsF);

    std::vector<float> cdist(5, CONTACT_OFFSET);
    cdist[1] = 0.0f;                       // the hull contributes nothing
    R.bCd = R.mk(dev, cdist);
    R.bRd = R.mk(dev, std::vector<float>{ REST_DISTANCE, REST_DISTANCE_B, REST_Z, REST_C });
    R.bMat = R.mk(dev, std::vector<uint32_t>(4u, POISON_U32));
    R.bStart = R.mk(dev, std::vector<uint32_t>{ RUN_A, RUN_B, RUN_Z, RUN_C });
    R.bTotal = R.mk(dev, std::vector<uint32_t>{ TOTAL_WORK });
    R.bRemap = R.mk(dev, std::vector<uint64_t>{ 0ull, uint64_t(NODE_INDEX), 0ull, 0ull, 0ull });

    std::memset(&R.blob, 0, sizeof(R.blob));
    R.blob.numTests = 4u;
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
    // The writer half of the blob stays NULL: this kernel has no global contact
    // stream, and a port that reaches for one faults rather than scribbling.
}

// Which particles system `s` must accept, analytically.  Identical to the fluid
// oracle EXCEPT that the filtered particle is NOT removed (XD3).
static std::vector<uint32_t> acceptedFor(const Sys& s)
{
    std::vector<uint32_t> out;
    if (s.trap) return out;
    for (uint32_t i = CELL_START; i < CELL_END; ++i)
    {
        if (s.ccd) { if (ccdAccepted(i, 1.0f, 2.0f)) out.push_back(i); continue; }
        float p[3], dir[3]; particlePos(i, p);
        if (boxSignedDistance(p, dir) <= CONTACT_OFFSET) out.push_back(i);
    }
    return out;
}

static void checkSys(PxgSlangComputeDevice* dev, const Sys& s, const char* label,
                     double& worstPenErr, double& worstNormErr)
{
    const std::vector<uint32_t> accept = acceptedFor(s);
    const auto counts = downloadBuffer<uint32_t>(dev, s.bCounts.h, s.num + SLACK);
    const auto infos = downloadBuffer<F4>(dev, s.bInfos.h, infoSlots(s.num));
    const auto forces = downloadBuffer<float>(dev, s.bForces.h, infoSlots(s.num));
    const auto nodes = downloadBuffer<uint64_t>(dev, s.bNodes.h, infoSlots(s.num));

    // ---- 1. the counters.  Every accepted particle's counter must read the
    //         number of work items that reached its system -- INCLUDING the
    //         attempts past the cap (XD4) -- and every other counter must be
    //         untouched at zero.
    std::vector<uint32_t> wantCount(s.num + SLACK, 0u);
    for (uint32_t pi : accept) wantCount[pi] = s.work;
    size_t badCount = 0;
    for (uint32_t pi = 0; pi < s.num + SLACK; ++pi)
        if (counts[pi] != wantCount[pi])
        {
            if (badCount < 4)
                std::printf("    %s/%s: counts[%u] = %u, expected %u\n",
                            label, s.name, pi, counts[pi], wantCount[pi]);
            ++badCount;
        }
    PXG_CHECK_MSG(badCount == 0, "%s/%s: %zu contact counters are wrong",
                  label, s.name, badCount);

    // ---- 2. the written slots, column-major with a stride of s.num.
    const uint32_t written = std::min(s.work, MAX_STATIC);
    std::vector<bool> live(infoSlots(s.num), false);
    size_t bad = 0;
    for (uint32_t pi : accept)
    {
        float p[3], outDir[3]; particlePos(pi, p);
        const float sd = boxSignedDistance(p, outDir);
        for (uint32_t slot = 0; slot < written; ++slot)
        {
            const uint32_t oi = pi + slot * s.num;
            live[oi] = true;
            const F4 np = infos[oi];
            const double nlen = std::sqrt(double(np.x) * np.x + double(np.y) * np.y +
                                          double(np.z) * np.z);
            worstNormErr = std::max(worstNormErr, std::fabs(nlen - 1.0));
            if (std::fabs(nlen - 1.0) > 1e-4)
            { if (bad < 4) std::printf("    %s/%s particle %u slot %u: |normal| = %g\n",
                                       label, s.name, pi, slot, nlen);
              ++bad; continue; }
            // X7's negation, pinned PHYSICALLY: the stored normal is the one the
            // one-way consumer (P9.13f) pushes ALONG, so it has to point out.
            const double dotOut = double(np.x) * outDir[0] + double(np.y) * outDir[1] +
                                  double(np.z) * outDir[2];
            if (dotOut < 0.9)
            { if (bad < 4) std::printf("    %s/%s particle %u (signed dist %g): normal "
                                       "(%g,%g,%g) is not the OUTWARD direction "
                                       "(%g,%g,%g), cos = %g\n", label, s.name, pi, sd,
                                       np.x, np.y, np.z, outDir[0], outDir[1], outDir[2],
                                       dotOut);
              ++bad; continue; }
            const double err = std::fabs(double(np.w) - (double(sd) - s.rest));
            worstPenErr = std::max(worstPenErr, err);
            if (err > 2e-4)
            { if (bad < 4) std::printf("    %s/%s particle %u slot %u: pen %g, expected %g "
                                       "(signed dist %g, rest %g)\n", label, s.name, pi,
                                       slot, np.w, sd - s.rest, sd, s.rest);
              ++bad; continue; }
            if (forces[oi] != 0.0f)
            { if (bad < 4) std::printf("    %s/%s particle %u slot %u: impulse %g, must be "
                                       "zeroed\n", label, s.name, pi, slot, forces[oi]);
              ++bad; continue; }
            if (nodes[oi] != uint64_t(NODE_INDEX))
            { if (bad < 4) std::printf("    %s/%s particle %u slot %u: node %llu, expected "
                                       "%u\n", label, s.name, pi, slot,
                                       (unsigned long long)nodes[oi], NODE_INDEX);
              ++bad; }
        }
    }
    PXG_CHECK_MSG(bad == 0, "%s/%s: %zu written contacts failed the geometric or identity "
                  "checks", label, s.name, bad);

    // ---- 3. EVERY other slot must still be poison -- which is where the
    //         attempts past the cap would have landed, and where a wrong stride
    //         lands.
    size_t badPoison = 0;
    for (uint32_t oi = 0; oi < infoSlots(s.num); ++oi)
    {
        if (live[oi]) continue;
        if (infos[oi].x != POISON_F || infos[oi].y != POISON_F ||
            infos[oi].z != POISON_F || infos[oi].w != POISON_F ||
            forces[oi] != POISON_F || nodes[oi] != POISON_U64)
        {
            if (badPoison < 4)
                std::printf("    %s/%s: slot %u (particle %u, contact %u) was written and "
                            "must not have been\n", label, s.name, oi, oi % s.num,
                            oi / s.num);
            ++badPoison;
        }
    }
    PXG_CHECK_MSG(badPoison == 0, "%s/%s: %zu slots outside the expected column-major set "
                  "were written", label, s.name, badPoison);
}

static void runAll(PxgSlangComputeDevice* dev)
{
    DRig R;
    buildRig(dev, R);

    Kernel k("psContactGenBda.spv", "ps_convexDiffuseCollisionLaunch", {}, sizeof(PsCollideBlob));
    PXG_CHECK_MSG(k.isValid(), "the convex diffuse contact-gen pipeline must build");
    if (!k.isValid()) { R.destroy(dev); return; }

    double worstPenErr = 0.0, worstNormErr = 0.0;
    // XD7: isDiffuse = TRUE, so the TGS halving never applies and the two runs
    // must produce IDENTICAL sets.
    for (int mode = 0; mode < 2; ++mode)
    {
        const char* label = mode ? "TGS" : "PGS";
        resetOutputs(dev, R);
        dev->flush();
        R.blob.isTGSWord = mode ? 1u : 0u;
        k.dispatch(1024u, {}, R.blob, 1u, 1u);   // grid (1024,1,1), block (32,8,1)
        dev->flush();
        for (const Sys& s : R.sys) checkSys(dev, s, label, worstPenErr, worstNormErr);
    }

    // ---- coverage, demanded rather than designed ----
    {
        const std::vector<uint32_t> accA = acceptedFor(R.sys[0]);
        const std::vector<uint32_t> accB = acceptedFor(R.sys[1]);
        PXG_CHECK_MSG(!accA.empty() && !accB.empty(),
                      "one of the analytic contact sets is EMPTY (%zu, %zu) -- the fixture "
                      "measures nothing", accA.size(), accB.size());
        PXG_CHECK_MSG(accA.size() != accB.size(),
                      "the plain and the CCD system accept the same number of particles "
                      "(%zu) -- the contact-volume policy is not exercised", accA.size());
        PXG_CHECK_MSG(WORK_A > MAX_STATIC,
                      "pair 0 owns %u work items against a cap of %u -- the capacity gate "
                      "never fires and a port that caps the COUNTER too is invisible",
                      WORK_A, MAX_STATIC);
        PXG_CHECK_MSG(WORK_B > 0u && WORK_B < MAX_STATIC,
                      "pair 1 owns %u work items -- the UNCAPPED path is not measured",
                      WORK_B);
        // XD3, stated as a demand: the filter pair names this particle and the
        // kernel must produce a contact for it anyway.
        const bool filteredAccepted =
            std::find(accA.begin(), accA.end(), FILTERED_SORTED) != accA.end();
        PXG_CHECK_MSG(filteredAccepted,
                      "particle %u, the one the rigid filter pair names, is not in the "
                      "analytic set -- XD3 (there is NO filter on this path) is unprobed",
                      FILTERED_SORTED);
        // XD6, stated as a demand: without nbIter > 1 the `return` and a
        // `continue` are the same statement.
        const uint32_t nbIter = (TOTAL_WORK + 8191u) / 8192u;
        PXG_CHECK_MSG(nbIter == 2u,
                      "nbIterationsPerWarps is %u, not 2 -- with one iteration per warp the "
                      "engine's `return` and a `continue` are indistinguishable and XD6 is "
                      "untested", nbIter);
        PXG_CHECK_MSG(WORK_C == 1u,
                      "system C is reached %u times -- the whole point of the 8193-item run "
                      "is that the trap's `return` costs it exactly one of its two", WORK_C);
        const bool a1 = ccdAccepted(MARGINAL_ISDIFFUSE, 1.0f, 2.0f);
        const bool a2 = ccdAccepted(MARGINAL_ISDIFFUSE, 2.0f, 2.0f);
        PXG_CHECK_MSG(a1 != a2, "particle %u does not separate isDiffuse=true (accepted: %d) "
                      "from isDiffuse=false in a TGS step (accepted: %d)",
                      MARGINAL_ISDIFFUSE, int(a1), int(a2));
        const bool b1 = ccdAccepted(MARGINAL_MINSEP, 1.0f, 2.0f);
        const bool b2 = ccdAccepted(MARGINAL_MINSEP, 1.0f, 1.0f);
        PXG_CHECK_MSG(b1 != b2, "particle %u does not separate the second pass's "
                      "cDistance + 2*cVolumeRadius (accepted: %d) from cDistance + "
                      "cVolumeRadius (accepted: %d)", MARGINAL_MINSEP, int(b1), int(b2));
        std::printf("    [ note ] %u work items, nbIter %u; accepted %zu (x%u, capped at %u) "
                    "/ %zu CCD (x%u) / %zu after the trap's return (x%u); strides %u/%u/%u; "
                    "worst |normal|-1 %.3g, worst penetration error %.3g\n",
                    TOTAL_WORK, nbIter, accA.size(), WORK_A, MAX_STATIC, accB.size(), WORK_B,
                    acceptedFor(R.sys[3]).size(), WORK_C, NUM_A, NUM_B, NUM_C,
                    worstNormErr, worstPenErr);
    }

    R.destroy(dev);
}

} // namespace diffuse

// ===========================================================================
// P9.89 -- THE PENETRATION'S THREE-LANE SUM, BIT FOR BIT.
//
// cudaParticleSystem.cu:963-964 (psConvexCollision) and :1116-1117
// (psConvexDiffuseCollision) assemble the penetration across lanes.  Lane k < 3
// holds (witnessB[k] - witnessA[k]) * normal[k], rounded as it enters the
// shuffle, and every lane computes
//     pen = shfl(pen, 0) + shfl(pen, 1) + shfl(pen, 2) - restDistance
// which C++ evaluates as ((p0 + p1) + p2) - restDistance.  The port spells it
// `dot(closestPointB - closestPointA, direction) - restDistance`.  In an
// UNDECORATED .spv llvmpipe evaluates that OpDot as p0 + (p1 + p2).  That is
// NOT a contraction into fmas; it is a different association.  Measured on
// 65536 random cases: the model misses 0 of them, and the port's value differs
// from the CUDA's on 19346.  psContactGenBda was unstamped until P9.89.
//
// The two batteries above could not see this.  Their box is axis-aligned and
// their particles sit nearest a FACE, so at most one lane product is ever
// non-negligible.  They also compare the penetration against a closed form to
// 2e-4, not bit for bit.
//
// This battery is exact instead.  The hull is a ROTATED, non-uniformly scaled
// cloud of twelve non-round vertices.  The particles are separated from it in
// every direction, so the closest feature is a vertex, an edge or a face at an
// off-axis angle.  The oracle is a host transcription of pxgSlangGjk's gjkRun,
// the same cold-start driver testNpGjk's gjkRef mirrors, run on the hull staged
// exactly as psStageHull stages it.  Only particles that end CLOSE with no
// anomaly (tetPrimitivesCollide2's EPA gate) are admitted, together with some
// that are DISTANT.  So what the kernel stores is the GJK witness pair and
// direction, and the penetration is assembled in the CUDA's lane order.  The
// twin assembles it as p0 + (p1 + p2) and must differ on some particles.
// ===========================================================================
namespace p89cvx
{
struct V3 { float x, y, z; };
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float floatOf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ---- pxgSlangGjk.slang, transcribed (cold start, per-thread 32-lane arrays) ----
static const uint32_t DEGENERATE = 0xffffu;
static const float MAXF = 3.4028234663852886e38f;
static const uint32_t G_DISTANT = 0u, G_CLOSE = 1u, G_OVERLAP = 2u;
static uint32_t next3(uint32_t x) { return x == 2u ? 0u : x + 1u; }
static uint32_t prev3(uint32_t x) { return x == 0u ? 2u : x - 1u; }
static uint32_t ffs(uint32_t m) { if (!m) return 0; uint32_t i = 0; while (!((m >> i) & 1u)) ++i; return i + 1; }
static void dotW(const float a[32], const float b[32], float o[32])
{
    float x[32]; for (uint32_t L = 0; L < 32; ++L) x[L] = a[L] * b[L];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const float s1 = (L + 1 < 32) ? x[L + 1] : x[L];
        const float s2 = (L + 2 < 32) ? x[L + 2] : x[L];
        o[L] = x[L] + s1 + s2;
    }
}
static void crossW(const float a[32], const float b[32], const uint32_t c0[32], const uint32_t c1[32], float o[32])
{
    for (uint32_t L = 0; L < 32; ++L) o[L] = a[c0[L]] * b[c1[L]] - a[c1[L]] * b[c0[L]];
}
static void splatXW(const float a[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[L & 28u]; }
static void splatVW(uint32_t index, const float a[32], float o[32])
{
    for (uint32_t L = 0; L < 32; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u];
}
static void magW(const float v[32], float cm[32])
{
    float dv[32]; dotW(v, v, dv);
    for (uint32_t L = 0; L < 32; ++L) cm[L] = std::sqrt(dv[L & 28u]);
}
static uint32_t encodeIndices(uint32_t a, uint32_t b) { return 0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu); }
static V3 crossX(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 absX(V3 a) { return { std::fabs(a.x), std::fabs(a.y), std::fabs(a.z) }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 getNormalized(V3 v)
{
    const float m = dot3(v, v);
    if (m > 0.0f) { const float r = 1.0f / std::sqrt(m); return { v.x * r, v.y * r, v.z * r }; }
    return { 0, 0, 0 };
}

static uint32_t closestPointTriW(float zV[32], const float cV[32], float closestPoint[32])
{
    uint32_t cp0I[32], cp1I[32];
    for (uint32_t L = 0; L < 32; ++L) { cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u; }
    for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = 0.0f;
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f)) return DEGENERATE;
    float shortEdgeV[32]; for (uint32_t L = 0; L < 32; ++L) shortEdgeV[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];
    float d1[32]; for (uint32_t L = 0; L < 32; ++L) d1[L] = zV[L ^ 4u] - zV[L];
    float mV[32]; crossW(d1, shortEdgeV, cp0I, cp1I, mV);
    float pS[32]; dotW(eV, cV, pS);
    float cme[32]; crossW(mV, eV, cp0I, cp1I, cme);
    float dcme[32]; dotW(cme, cV, dcme);
    bool uS[32]; for (uint32_t L = 0; L < 32; ++L) uS[L] = dcme[L] <= 0.0f;
    if (!(uS[0] || uS[4]))
    {
        float dmc[32]; dotW(mV, cV, dmc);
        const float projD = dmc[0];
        float dmm[32]; dotW(mV, mV, dmm);
        float sxdmm[32]; splatXW(dmm, sxdmm);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = mV[L] * projD / sxdmm[L];
        splatVW(0u, tmp, closestPoint);
        const bool flip = projD < 0.0f;
        for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) mV[L] = projD;
        const uint32_t index = flip ? 1u : 0u;
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((index << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : mV[src]; }
        return flip ? 0x0123u : 0x1023u;
    }
    uint32_t emask = 0u;
    if (uS[0] && (pS[0] > 0.0f)) emask |= 1u;
    if (uS[4] && (pS[4] > 0.0f)) emask |= 2u;
    uint32_t eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        float sxpS[32]; splatXW(pS, sxpS);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = cV[L] - eV[L] * sxpS[L] / lD[L];
        splatVW(eid, tmp, closestPoint);
        return (eid << 8) | 0x22u;
    }
    for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = cV[L];
    return 0x0021u;
}

static uint32_t closestPointTetW(float zV[32], const float dV[32], float closestPoint[32])
{
    for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = 0.0f;
    float n0V[32]; splatVW(4u, zV, n0V);
    float dn0d[32]; dotW(n0V, dV, dn0d);
    float dn0z[32]; dotW(n0V, zV, dn0z);
    if (dn0d[0] > dn0z[0]) return DEGENERATE;
    uint32_t cp0I[32], cp1I[32], vNextI[32], vPrevI[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u;
        vNextI[L] = (L > 7u) ? (L - 8u) : (L + 4u); vPrevI[L] = (L < 4u) ? (L + 8u) : (L - 4u);
    }
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = dV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f)) return DEGENERATE;
    float shortEdgeV[32]; for (uint32_t L = 0; L < 32; ++L) shortEdgeV[L] = (lD[vNextI[L]] < lD[L]) ? eV[vNextI[L]] : eV[L];
    float d2[32]; for (uint32_t L = 0; L < 32; ++L) d2[L] = zV[vNextI[L]] - zV[L];
    float nV[32]; crossW(shortEdgeV, d2, cp0I, cp1I, nV);
    float aS[32]; dotW(nV, dV, aS);
    float zvp[32]; for (uint32_t L = 0; L < 32; ++L) zvp[L] = zV[vPrevI[L]];
    float gS[32]; dotW(zvp, nV, gS);
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8])) return DEGENERATE;
    for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) nV[L] = aS[L - 3u];
    if ((aS[0] > 0.0f) && (aS[4] > 0.0f) && (aS[8] > 0.0f))
    {
        for (uint32_t L = 0; L < 32; ++L) if (L >= 20u) zV[L] = nV[L - 20u];
        return 0x2104u;
    }
    float pS[32]; dotW(eV, dV, pS);
    if ((pS[0] <= 0.0f) && (pS[4] <= 0.0f) && (pS[8] <= 0.0f))
    {
        for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = dV[L];
        return 0x2131u;
    }
    float cen[32]; crossW(eV, nV, cp0I, cp1I, cen);
    float dcen[32]; dotW(cen, dV, dcen);
    bool uiS[32]; for (uint32_t L = 0; L < 32; ++L) uiS[L] = dcen[L] <= 0.0f;
    float evn[32]; for (uint32_t L = 0; L < 32; ++L) evn[L] = eV[vNextI[L]];
    float cevn[32]; crossW(evn, nV, cp0I, cp1I, cevn);
    float dcevn[32]; dotW(cevn, dV, dcevn);
    bool ujS[32]; for (uint32_t L = 0; L < 32; ++L) ujS[L] = dcevn[L] > 0.0f;
    uint32_t emask = 0u;
    if (uiS[0] && ujS[vPrevI[0]] && (pS[0] >= 0.0f)) emask |= 1u;
    if (uiS[4] && ujS[vPrevI[4]] && (pS[4] >= 0.0f)) emask |= 2u;
    if (uiS[8] && ujS[vPrevI[8]] && (pS[8] >= 0.0f)) emask |= 4u;
    uint32_t eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        float pl[32]; for (uint32_t L = 0; L < 32; ++L) pl[L] = pS[L] / lD[L];
        float spl[32]; splatXW(pl, spl);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = dV[L] - eV[L] * spl[L];
        splatVW(eid, tmp, closestPoint);
        return (eid << 8) | 0x32u;
    }
    uint32_t fmask = 0u;
    if (!uiS[0] && !ujS[0] && (aS[0] <= 0.0f)) fmask |= 1u;
    if (!uiS[4] && !ujS[4] && (aS[4] <= 0.0f)) fmask |= 2u;
    if (!uiS[8] && !ujS[8] && (aS[8] <= 0.0f)) fmask |= 4u;
    uint32_t fid = ffs(fmask);
    if (fid != 0u)
    {
        fid = fid - 1u;
        float dnn[32]; dotW(nV, nV, dnn);
        float ad[32]; for (uint32_t L = 0; L < 32; ++L) ad[L] = aS[L] / dnn[L];
        float sad[32]; splatXW(ad, sad);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = nV[L] * sad[L];
        splatVW(fid, tmp, closestPoint);
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((fid << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : -nV[src]; }
        return (next3(fid) << 12) | (fid << 8) | 0x33u;
    }
    return DEGENERATE;
}

static uint32_t supportIdx(V3 dir, const std::vector<V3>& V)
{
    const uint32_t nb = uint32_t(V.size());
    uint32_t bestN = 0xFFFFFFFFu; float bestDist = -MAXF;
    for (uint32_t i = 0; i < nb; i += 32)
    {
        float d[32];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t n = i + lane;
            d[lane] = (n < nb) ? dot3(dir, V[n]) : -MAXF;
        }
        float chunkMax = -MAXF; for (uint32_t lane = 0; lane < 32; ++lane) chunkMax = std::fmax(chunkMax, d[lane]);
        if (chunkMax > bestDist) { uint32_t ll = 0; for (uint32_t lane = 0; lane < 32; ++lane) if (d[lane] == chunkMax) { ll = lane; break; } bestDist = chunkMax; bestN = i + ll; }
    }
    return bestN;
}

struct GjkRes { uint32_t ret; V3 direction, cpA, cpB, cpDir; };

static GjkRes gjkRun(const std::vector<V3>& A, const std::vector<V3>& B, V3 initialUnitDir, float minSep,
                     float convergenceRatio)
{
    float simplexV[32]; for (uint32_t L = 0; L < 32; ++L) simplexV[L] = 0.0f;
    float closestPointV[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t c = L & 3u;
        closestPointV[L] = (c == 0) ? initialUnitDir.x : (c == 1) ? initialUnitDir.y : (c == 2) ? initialUnitDir.z : MAXF;
    }
    float upperBound = MAXF, lowerBound = -MAXF;
    float cmA[32]; magW(closestPointV, cmA);
    bool degenerate = false;
    int size = 0, backupSize = 0;
    float backupSimplexV[32]; for (uint32_t L = 0; L < 32; ++L) backupSimplexV[L] = 0.0f;
    float backupClosestV[32]; for (uint32_t L = 0; L < 32; ++L) backupClosestV[L] = 0.0f;
    V3 directionOut = { 0, 0, 0 };
    bool run = true;
    while (run)
    {
        float dV[32]; for (uint32_t L = 0; L < 32; ++L) dV[L] = closestPointV[L] / cmA[L];
        const V3 dvVec = { dV[0], dV[1], dV[2] };
        const uint32_t indexA = supportIdx({ -dvVec.x, -dvVec.y, -dvVec.z }, A);
        const uint32_t indexB = supportIdx(dvVec, B);
        const V3 sv = sub3(A[indexA], B[indexB]);
        float supportVertexV[32];
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t c = L & 3u; supportVertexV[L] = (c == 0) ? sv.x : (c == 1) ? sv.y : (c == 2) ? sv.z : 0.0f; }
        const float encoded = floatOf(encodeIndices(indexA, indexB));
        for (uint32_t L = 0; L < 32; ++L) if ((L >> 2) == (uint32_t)size) simplexV[L] = ((L & 3u) < 3u) ? supportVertexV[L] : encoded;
        float dds[32]; dotW(dV, supportVertexV, dds);
        const float lb = dds[0];
        if (lb > lowerBound)
        {
            directionOut = { -dV[0], -dV[1], -dV[2] };
            lowerBound = lb;
            if (lowerBound > minSep || lowerBound >= upperBound * convergenceRatio) { run = false; break; }
        }
        uint32_t f = 0u;
        float pointV[32];
        if (size == 0) { f = 0x2101u; for (uint32_t L = 0; L < 32; ++L) pointV[L] = supportVertexV[L]; }
        else if (size == 1)
        {
            float aV[32]; splatVW(0u, simplexV, aV);
            float lV[32]; for (uint32_t L = 0; L < 32; ++L) lV[L] = supportVertexV[L] - aV[L];
            float dll[32]; dotW(lV, lV, dll);
            float mm[32]; splatXW(dll, mm);
            float dal[32]; dotW(aV, lV, dal);
            float xx[32]; splatXW(dal, xx); for (uint32_t L = 0; L < 32; ++L) xx[L] = -xx[L];
            if (mm[0] == 0.0f) { f = DEGENERATE; for (uint32_t L = 0; L < 32; ++L) pointV[L] = 0.0f; }
            else if (xx[0] < mm[0]) { f = 0x2102u; for (uint32_t L = 0; L < 32; ++L) pointV[L] = aV[L] + lV[L] * xx[L] / mm[L]; }
            else { f = 0x2111u; for (uint32_t L = 0; L < 32; ++L) pointV[L] = supportVertexV[L]; }
        }
        else if (size == 2) { f = closestPointTriW(simplexV, supportVertexV, pointV); }
        else { f = closestPointTetW(simplexV, supportVertexV, pointV); }
        magW(pointV, cmA);
        const int newSize = (int)(f & 0xfu);
        degenerate = (f == DEGENERATE) || (cmA[0] >= upperBound && newSize <= size);
        if (degenerate) { run = false; break; }
        backupSize = (cmA[0] >= upperBound) ? size : 0;
        if (backupSize != 0) { for (uint32_t L = 0; L < 32; ++L) { backupSimplexV[L] = simplexV[L]; backupClosestV[L] = closestPointV[L]; } }
        else upperBound = cmA[0];
        for (uint32_t L = 0; L < 32; ++L) closestPointV[L] = pointV[L];
        size = newSize;
        float tmpP[32];
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t sourceIndex = (f >> ((L & 28u) + 4u)) & 0xfu; tmpP[L] = simplexV[((sourceIndex << 2) | (L & 3u)) & 31u]; }
        for (uint32_t L = 0; L < 32; ++L) if (L < 12u) simplexV[L] = tmpP[L];
        if (!(upperBound > 0.0f && lowerBound < upperBound * convergenceRatio)) run = false;
    }
    if (backupSize > 0) { for (uint32_t L = 0; L < 32; ++L) { simplexV[L] = backupSimplexV[L]; closestPointV[L] = backupClosestV[L]; } size = backupSize; }

    GjkRes o;
    std::memset(&o, 0, sizeof(o));
    o.direction = directionOut;
    if (upperBound == 0.0f) { o.ret = G_OVERLAP; return o; }
    if (lowerBound > minSep) { o.ret = G_DISTANT; return o; }
    const V3 cp = { closestPointV[0], closestPointV[1], closestPointV[2] };
    const V3 gn = getNormalized(cp);
    const V3 dir = { -gn.x, -gn.y, -gn.z };
    o.cpDir = dir;
    const uint32_t a0 = bitsOf(simplexV[3]) & 0xffu, a1 = bitsOf(simplexV[7]) & 0xffu, a2 = bitsOf(simplexV[11]) & 0xffu;
    V3 cpA;
    if (size == 3)
    {
        const V3 Av = { simplexV[0], simplexV[1], simplexV[2] }, Bv = { simplexV[4], simplexV[5], simplexV[6] }, Cv = { simplexV[8], simplexV[9], simplexV[10] };
        const V3 m = crossX(sub3(Bv, Av), sub3(Cv, Av));
        const V3 ma = absX(m);
        const V3 s1 = crossX(sub3(Bv, cp), sub3(Cv, cp));
        const V3 s2 = crossX(sub3(Cv, cp), sub3(Av, cp));
        float d, u, v;
        if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; }
        else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; }
        else                              { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; }
        const float w = 1.0f - u - v;
        const V3 p0 = A[a0], p1 = A[a1], p2 = A[a2];
        cpA = { u * p0.x + v * p1.x + w * p2.x, u * p0.y + v * p1.y + w * p2.y, u * p0.z + v * p1.z + w * p2.z };
    }
    else if (size == 2)
    {
        const V3 V0 = { simplexV[0], simplexV[1], simplexV[2] }, V1 = { simplexV[4], simplexV[5], simplexV[6] };
        const V3 dv = sub3(V1, V0);
        const float q = -dot3(V0, dv) / dot3(dv, dv);
        const V3 p0 = A[a0], p1 = A[a1];
        cpA = { (1.0f - q) * p0.x + q * p1.x, (1.0f - q) * p0.y + q * p1.y, (1.0f - q) * p0.z + q * p1.z };
    }
    else cpA = A[a0];
    o.cpA = cpA;
    o.cpB = { cpA.x + upperBound * dir.x, cpA.y + upperBound * dir.y, cpA.z + upperBound * dir.z };
    o.ret = G_CLOSE;
    return o;
}

// psContactGenBda's quatRotate / quatRotateInv (== PxQuat::rotate / rotateInv).
static V3 qRot(const float q[4], V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    return V3{ (vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2),
               (vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2),
               (vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2) };
}
static V3 qRotInv(const float q[4], V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    return V3{ (vx * w2 - (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2),
               (vy * w2 - (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2),
               (vz * w2 - (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2) };
}

// The rig: one hull, one particle system (fluid AND diffuse arrays over the same
// particles), one pair, one work item, one grid cell.
static const uint32_t NB_VERTS = 12u;
static const uint32_t NB_PART = 96u;
static const float PART_CD = 0.2371f, HULL_CD = 0.0173f, REST = 0.0137f;
static const uint32_t NODE = 411u;
static const float RIGID_Q[4] = { 0.2311f, -0.3877f, 0.1523f, 0.8795f };   // normalised below
static const V3 RIGID_P = { 0.3713f, -0.2129f, 0.5237f };
static const V3 SCALE = { 1.13f, 0.87f, 1.21f };
static const float SCALE_Q[4] = { -0.1417f, 0.2963f, 0.3319f, 0.8839f };    // normalised below

static void normQ(const float in[4], float q[4])
{
    const float m = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2] + in[3] * in[3]);
    for (int k = 0; k < 4; ++k) q[k] = in[k] / m;
}

struct Case { V3 p; bool contact; F4 np; float penOld; };

struct Rig
{
    float rq[4], sq[4];
    std::vector<V3> vertsLocal, hullWorld;
    std::vector<Case> cases;
    uint32_t nbDistant = 0, nbSize[4] = { 0, 0, 0, 0 };
};

static bool buildRig(Rig& R)
{
    normQ(RIGID_Q, R.rq);
    normQ(SCALE_Q, R.sq);
    uint64_t s = 0x89c0ffeeull;
    auto u = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return float(uint32_t(s >> 40)) / 16777216.0f; };
    // twelve vertices on a lumpy ellipsoid, every coordinate non-round
    for (uint32_t v = 0; v < NB_VERTS; ++v)
    {
        V3 d{ u() * 2.0f - 1.0f, u() * 2.0f - 1.0f, u() * 2.0f - 1.0f };
        const float m = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        const float r = 0.55f + 0.25f * u();
        R.vertsLocal.push_back(V3{ d.x / m * r * 1.3f, d.y / m * r * 0.9f, d.z / m * r });
    }
    // psStageHull: vertex -> psVertex2Shape -> quatRotate(rigidQ) + rigidP
    for (const V3& v : R.vertsLocal)
    {
        V3 t = qRot(R.sq, v);
        t.x *= SCALE.x; t.y *= SCALE.y; t.z *= SCALE.z;
        R.hullWorld.push_back(add3(qRot(R.rq, qRotInv(R.sq, t)), RIGID_P));
    }
    V3 c{ 0, 0, 0 };
    for (const V3& w : R.hullWorld) c = add3(c, w);
    c = V3{ c.x / float(NB_VERTS), c.y / float(NB_VERTS), c.z / float(NB_VERTS) };

    const float cd = PART_CD + HULL_CD;
    const float ratio = 1.0f - 0.000225f;
    for (int trial = 0; trial < 100000 && R.cases.size() < NB_PART; ++trial)
    {
        V3 d{ u() * 2.0f - 1.0f, u() * 2.0f - 1.0f, u() * 2.0f - 1.0f };
        const float m = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        if (m < 0.2f) continue;
        const float r = 0.4f + 1.6f * u();
        const V3 p = V3{ c.x + d.x / m * r, c.y + d.y / m * r, c.z + d.z / m * r };
        // particleConvexCollision, CCD off: initial direction p - rigidP
        const V3 aToB = sub3(p, RIGID_P);
        V3 dir0{ 1.0f, 0.0f, 0.0f };
        if (dot3(aToB, aToB) > 0.0f) dir0 = aToB;
        const GjkRes o = gjkRun(std::vector<V3>{ p }, R.hullWorld, dir0, cd, ratio);
        Case k{};
        k.p = p;
        if (o.ret == G_DISTANT)
        {
            if (R.nbDistant >= NB_PART / 8u) continue;   // a few, not most
            ++R.nbDistant;
            k.contact = false;
            R.cases.push_back(k);
            continue;
        }
        if (o.ret != G_CLOSE) continue;                  // OVERLAP would run EPA
        // tetPrimitivesCollide2's anomaly gate (pxgSlangTetCollide.slang:93,
        // cudaParticleSystem.cu:751): keep well clear of 0.999 in BOTH orders
        const float ax = o.cpDir.x * o.direction.x, ay = o.cpDir.y * o.direction.y, az = o.cpDir.z * o.direction.z;
        if (!((ax + ay) + az > 0.9995f) || !(ax + (ay + az) > 0.9995f)) continue;
        // cu:959-964: lane k holds (witnessB - witnessA) * normal, rounded; lanes
        // 0 + 1 + 2 left to right, then - restDistance
        const float p0 = (o.cpB.x - o.cpA.x) * o.direction.x;
        const float p1 = (o.cpB.y - o.cpA.y) * o.direction.y;
        const float p2 = (o.cpB.z - o.cpA.z) * o.direction.z;
        k.contact = true;
        k.np = F4{ -o.direction.x, -o.direction.y, -o.direction.z, ((p0 + p1) + p2) - REST };
        k.penOld = (p0 + (p1 + p2)) - REST;                  // the pre-P9.89 port on llvmpipe
        R.cases.push_back(k);
    }
    return R.cases.size() == NB_PART;
}

// The hull blob at pxgSlangConvexHull's offsets; the kernel reads only the
// header, the extents and the vertex run.
static std::vector<F4> makeBlob(const Rig& R)
{
    std::vector<F4> blob;
    blob.push_back(F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    F4 desc{};
    const uint32_t packed = (0u << 16) | (NB_VERTS << 8) | 0u;
    std::memcpy(&desc.x, &packed, 4);
    blob.push_back(desc);
    blob.push_back(F4{ 1.0f, 1.0f, 1.0f, 0.4f });   // .w: inner-sphere radius (EPA only)
    for (const V3& v : R.vertsLocal) blob.push_back(F4{ v.x, v.y, v.z, 0.0f });
    for (int i = 0; i < 8; ++i) blob.push_back(F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    return blob;
}

static void run(bool diffuse)
{
    const char* lbl = diffuse ? "diffuse" : "fluid";
    PxgSlangComputeDevice* dev = testDevice();
    Rig R;
    const bool ok = buildRig(R);
    PXG_CHECK_MSG(ok, "p89 %s: only %zu admissible particles around the hull", lbl, R.cases.size());
    if (!ok) return;

    uint32_t nContact = 0, nOrder = 0;
    for (const Case& k : R.cases)
        if (k.contact) { ++nContact; if (bitsOf(k.penOld) != bitsOf(k.np.w)) ++nOrder; }
    PXG_CHECK_MSG(nContact >= NB_PART / 2u, "p89 %s: only %u of %u particles are in contact", lbl, nContact, NB_PART);
    PXG_CHECK_MSG(nOrder > 0u, "p89 %s: the pre-P9.89 lane order p0 + (p1 + p2) gives the same bits as the CUDA's "
                  "(p0 + p1) + p2 on every contact -- the penetration's association is not under test", lbl);
    std::printf("    [p89 %s] %u contacts, %u DISTANT; the old lane order changes %u penetrations\n",
                lbl, nContact, R.nbDistant, nOrder);

    std::vector<PxgBufferHandle> owned;
    auto mk = [&](auto vec) { DevBuf b; b.h = createBufferFrom(dev, vec); b.addr = dev->getBufferDeviceAddress(b.h); owned.push_back(b.h); return b; };

    std::vector<F4> pos(NB_PART);
    for (uint32_t i = 0; i < NB_PART; ++i) pos[i] = F4{ R.cases[i].p.x, R.cases[i].p.y, R.cases[i].p.z, 1.0f };
    const DevBuf bPos = mk(pos);
    const DevBuf bHull = mk(makeBlob(R));
    const DevBuf bCellStart = mk(std::vector<uint32_t>{ 0u });
    const DevBuf bCellEnd = mk(std::vector<uint32_t>{ NB_PART });
    std::vector<uint32_t> s2u(NB_PART);
    for (uint32_t i = 0; i < NB_PART; ++i) s2u[i] = 7000u + i;
    const DevBuf bS2U = mk(s2u);
    const DevBuf bRunsum = mk(std::vector<uint32_t>{ 0u });
    std::vector<PxgParticleSimBuffer> simBufs(1);
    std::memset(simBufs.data(), 0, sizeof(PxgParticleSimBuffer));
    const DevBuf bSimBuf = mk(simBufs);
    const uint32_t MAXC = 12u;
    const DevBuf bInfos = mk(std::vector<F4>(size_t(NB_PART) * MAXC, F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    const DevBuf bForces = mk(std::vector<float>(size_t(NB_PART) * MAXC, POISON_F));
    const DevBuf bNodes = mk(std::vector<uint64_t>(size_t(NB_PART) * MAXC, POISON_U64));
    const DevBuf bCounts = mk(std::vector<uint32_t>(NB_PART, 0u));
    const DevBuf bNum = mk(std::vector<int32_t>{ int32_t(NB_PART), 0 });

    std::vector<PxgParticleSystem> descs(1);
    std::memset(descs.data(), 0, sizeof(PxgParticleSystem));
    PxgParticleSystem& m = descs[0];
    m.mCommonData.mGridCellWidth = 1.0f;
    m.mCommonData.mGridSizeX = 1u; m.mCommonData.mGridSizeY = 1u; m.mCommonData.mGridSizeZ = 1u;
    m.mCommonData.mNumParticles = NB_PART;
    m.mCommonData.mMaxDiffuseParticles = NB_PART;
    m.mCommonData.mNumParticleBuffers = 1u;
    m.mData.mFlags = 0u;                                   // CCD off
    m.mCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
    m.mCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
    m.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    m.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    m.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bS2U.addr));
    m.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
    m.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSimBuf.addr));
    m.mDiffuseCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
    m.mDiffuseCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
    m.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(bPos.addr));
    m.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(bPos.addr));
    m.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(bNum.addr));
    m.mDiffuseOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(bInfos.addr));
    m.mDiffuseOneWayForces = reinterpret_cast<float*>(size_t(bForces.addr));
    m.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(bNodes.addr));
    m.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(bCounts.addr));
    const DevBuf bDesc = mk(descs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::eCONVEXMESH);
    shapes[1].scale = PxMeshScale(PxVec3(SCALE.x, SCALE.y, SCALE.z), PxQuat(R.sq[0], R.sq[1], R.sq[2], R.sq[3]));
    shapes[1].hullOrMeshPtr = size_t(bHull.addr);
    const DevBuf bShapes = mk(shapes);
    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(RIGID_P.x, RIGID_P.y, RIGID_P.z), PxQuat(R.rq[0], R.rq[1], R.rq[2], R.rq[3]));
    const DevBuf bTc = mk(tcache);
    std::vector<PxgContactManagerInput> cin(1);
    std::memset(cin.data(), 0, sizeof(PxgContactManagerInput));
    cin[0].shapeRef0 = 1u; cin[0].transformCacheRef0 = 1u;   // the hull in slot 0: LoadShapePair swaps
    cin[0].shapeRef1 = 0u; cin[0].transformCacheRef1 = 0u;
    const DevBuf bCin = mk(cin);
    const DevBuf bBounds = mk(std::vector<float>{ -9, -9, -9, 9, 9, 9, -9, -9, -9, 9, 9, 9 });
    const DevBuf bCd = mk(std::vector<float>{ PART_CD, HULL_CD });
    const DevBuf bRd = mk(std::vector<float>{ REST });
    const DevBuf bMat = mk(std::vector<uint32_t>(4u, POISON_U32));
    const DevBuf bStart = mk(std::vector<uint32_t>{ 0u });
    const DevBuf bTotal = mk(std::vector<uint32_t>{ 1u });
    const DevBuf bRemap = mk(std::vector<uint64_t>{ 0ull, uint64_t(NODE) });
    const uint32_t MAX_CONTACTS = 256u;
    std::vector<HostContact> contacts(MAX_CONTACTS);
    for (auto& ct : contacts) { ct.normal_pen = F4{ POISON_F, POISON_F, POISON_F, POISON_F }; ct.rigidId = POISON_U64; ct.particleId = POISON_U64; }
    const DevBuf bContacts = mk(contacts);
    const DevBuf bCount = mk(std::vector<uint32_t>{ 0u });
    const DevBuf bSbP = mk(std::vector<uint64_t>(MAX_CONTACTS, POISON_U64));
    const DevBuf bTbP = mk(std::vector<uint32_t>(MAX_CONTACTS, POISON_U32));
    const DevBuf bIbP = mk(std::vector<uint32_t>(MAX_CONTACTS, POISON_U32));
    const DevBuf bBr = mk(std::vector<uint64_t>(MAX_CONTACTS, POISON_U64));
    const DevBuf bTbr = mk(std::vector<uint32_t>(MAX_CONTACTS, POISON_U32));
    const DevBuf bIbr = mk(std::vector<uint32_t>(MAX_CONTACTS, POISON_U32));

    PsCollideBlob blob;
    std::memset(&blob, 0, sizeof(blob));
    blob.numTests = 1u;
    blob.toleranceLength = 1.0f;
    blob.cmInputs = bCin.addr; blob.shapes = bShapes.addr; blob.transformCache = bTc.addr;
    blob.bounds = bBounds.addr; blob.contactDistance = bCd.addr; blob.restDistances = bRd.addr;
    blob.materials = bMat.addr; blob.startIndices = bStart.addr; blob.totalNumPairs = bTotal.addr;
    blob.particleSystems = bDesc.addr; blob.shapeToRigidRemap = bRemap.addr;
    blob.w_particleContacts = bContacts.addr; blob.w_numTotalContacts = bCount.addr;
    blob.w_contactSortedByParticle = bSbP.addr; blob.w_tempContactByParticle = bTbP.addr;
    blob.w_contactIndexSortedByParticle = bIbP.addr; blob.w_contactByRigid = bBr.addr;
    blob.w_tempContactByRigid = bTbr.addr; blob.w_contactIndexSortedByRigid = bIbr.addr;
    blob.w_maxContacts = MAX_CONTACTS;

    Kernel k("psContactGenBda.spv", diffuse ? "ps_convexDiffuseCollisionLaunch" : "ps_convexCollisionLaunch", {},
             sizeof(PsCollideBlob));
    // The convex kernels are one warp (32 x 8 threads per block) per work item.
    k.dispatch(1024u, {}, blob, 1u, 1u);
    dev->flush();

    // Per particle: the stored normal/penetration, or none.
    std::vector<int> gotAt(NB_PART, -1);
    std::vector<F4> gotNp(NB_PART);
    size_t bad = 0, asOld = 0, extra = 0;
    if (!diffuse)
    {
        const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
        PXG_CHECK_MSG(gCount == nContact, "p89 %s: %u contacts, the mirror predicts %u", lbl, gCount, nContact);
        const auto gC = downloadBuffer<HostContact>(dev, bContacts.h, std::min(gCount, MAX_CONTACTS));
        for (size_t a = 0; a < gC.size(); ++a)
        {
            const uint64_t pid = gC[a].particleId;
            if ((pid >> 32) != 0ull || uint32_t(pid) >= NB_PART || gotAt[uint32_t(pid)] >= 0 || gC[a].rigidId != uint64_t(NODE))
            { ++extra; continue; }
            gotAt[uint32_t(pid)] = int(a);
            gotNp[uint32_t(pid)] = gC[a].normal_pen;
        }
    }
    else
    {
        const auto counts = downloadBuffer<uint32_t>(dev, bCounts.h, NB_PART);
        const auto infos = downloadBuffer<F4>(dev, bInfos.h, NB_PART);
        const auto nodes = downloadBuffer<uint64_t>(dev, bNodes.h, NB_PART);
        const auto forces = downloadBuffer<float>(dev, bForces.h, NB_PART);
        for (uint32_t i = 0; i < NB_PART; ++i)
        {
            if (counts[i] == 0u) continue;
            if (counts[i] != 1u || nodes[i] != uint64_t(NODE) || forces[i] != 0.0f) { ++extra; continue; }
            gotAt[i] = int(i);
            gotNp[i] = infos[i];
        }
    }
    for (uint32_t i = 0; i < NB_PART; ++i)
    {
        const Case& c = R.cases[i];
        if ((gotAt[i] >= 0) != c.contact) { ++bad; continue; }
        if (!c.contact) continue;
        const F4& g = gotNp[i];
        const bool same = bitsOf(g.x) == bitsOf(c.np.x) && bitsOf(g.y) == bitsOf(c.np.y) &&
                          bitsOf(g.z) == bitsOf(c.np.z) && bitsOf(g.w) == bitsOf(c.np.w);
        if (!same)
        {
            if (bad < 4)
                std::printf("    p89 %s particle %u: gpu (%08x %08x %08x | %08x) mirror (%08x %08x %08x | %08x) old-order pen %08x\n",
                            lbl, i, bitsOf(g.x), bitsOf(g.y), bitsOf(g.z), bitsOf(g.w), bitsOf(c.np.x), bitsOf(c.np.y),
                            bitsOf(c.np.z), bitsOf(c.np.w), bitsOf(c.penOld));
            ++bad;
            if (bitsOf(g.w) == bitsOf(c.penOld)) ++asOld;
        }
    }
    PXG_CHECK_MSG(extra == 0u, "p89 %s: %zu malformed or duplicated outputs", lbl, extra);
    PXG_CHECK_MSG(bad == 0u, "p89 %s: %zu particles differ from the CUDA lane-order mirror (%zu of them carry the "
                  "pre-P9.89 p0 + (p1 + p2) penetration)", lbl, bad, asOld);

    for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
}
} // namespace p89cvx

} // namespace

PXG_TEST(psConvexContactGen_p89PenLaneOrder)
{
    p89cvx::run(false);
}

PXG_TEST(psConvexDiffuseContactGen_p89PenLaneOrder)
{
    p89cvx::run(true);
}

PXG_TEST(psConvexContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}

PXG_TEST(psConvexDiffuseContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();
    diffuse::runAll(dev);
}
