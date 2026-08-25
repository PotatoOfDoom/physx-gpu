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
static const uint32_t NB_PARTICLES = 40u;     // in the live cell
static const uint32_t CELL_START = 4u;        // the live cell's span ...
static const uint32_t CELL_END = 36u;         // ... 32 particles
// Just BELOW the number of contacts the three work items produce, so the writer's
// capacity gate fires -- and the COUNTER must still overshoot it, the same
// contract as the diffuse one-way count in P9.13f/g.
//
// JUST below, and that margin is the point.  The first draft capped at 40 of 82,
// and WHICH 40 survive is the atomicAdd's race: two mutants that had been killed
// started surviving because the contacts they corrupted lost that race and were
// never compared.  A cap that hides half the output hides half the errors.  The
// check below also demands that the written set covers BOTH particle systems.
static const uint32_t MAX_CONTACTS = 76u;
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
static const float REST_DISTANCE = 0.02f;
static const float REST_DISTANCE_B = 0.037f;   // pair 1, deliberately different
static const uint32_t NODE_INDEX = 77u;
static const uint32_t FILTERED_UNSORTED = 903u;   // the unsorted id the pair matches
static const uint32_t FILTERED_SORTED = 20u;      // which sorted slot carries it
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
    // The cost is stated rather than hidden: `cell-hash-zero` (reading cell 0
    // instead of cell gridHash) is EQUIVALENT here, and the grid-hash path is not
    // pinned by this fixture or by the primitives one, which uses the same
    // single-cell trick.  Pinning it wants its own tranche with a host mirror of
    // gridCal.cuh -- an ENGINE header, so not a second transcription of ours.
    R.bCellStart = R.mk(dev, std::vector<uint32_t>{ CELL_START });
    R.bCellEnd = R.mk(dev, std::vector<uint32_t>{ CELL_END });

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
    m.mCommonData.mGridCellWidth = 1.0f;
    m.mCommonData.mGridSizeX = 1u;   // 1x1x1 -> gridHash is always 0; see bCellStart
    m.mCommonData.mGridSizeY = 1u;
    m.mCommonData.mGridSizeZ = 1u;
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
    std::vector<PxgContactManagerInput> cin(2);
    std::memset(cin.data(), 0, cin.size() * sizeof(PxgContactManagerInput));
    cin[0].shapeRef0 = 0u;  cin[0].transformCacheRef0 = 0u;
    cin[0].shapeRef1 = 1u;  cin[0].transformCacheRef1 = 1u;
    cin[1].shapeRef0 = 1u;  cin[1].transformCacheRef0 = 1u;
    cin[1].shapeRef1 = 2u;  cin[1].transformCacheRef1 = 2u;
    R.bCin = R.mk(dev, cin);

    // DIFFERENT bounds per shape.  With one box for everything the overlap region
    // and the union are the same solid and X2's `combine` mutant is the identity;
    // these two overlap in a strictly smaller box than either.
    std::vector<float> boundsF(3 * 6, 0.0f);
    const float bmn[3][3] = { { -9.0f, -3.0f, -7.0f },     // shape 0, the particles
                              { -4.0f, -8.0f, -5.0f },     // shape 1, the hull
                              { -9.0f, -3.0f, -7.0f } };   // shape 2, the CCD particles
    const float bmx[3][3] = { {  6.0f,  9.0f,  4.0f },
                              {  8.0f,  5.0f,  9.0f },
                              {  6.0f,  9.0f,  4.0f } };
    for (uint32_t sh = 0; sh < 3; ++sh)
        for (int k = 0; k < 3; ++k)
        { boundsF[size_t(sh) * 6 + k] = bmn[sh][k];
          boundsF[size_t(sh) * 6 + 3 + k] = bmx[sh][k]; }
    R.bBounds = R.mk(dev, boundsF);

    std::vector<float> cdist(3, 0.0f);
    cdist[0] = CONTACT_OFFSET;   // cDistance = 0.35 + 0 for pair 0 ...
    cdist[2] = CONTACT_OFFSET;   // ... and for pair 1
    R.bCd = R.mk(dev, cdist);
    // DIFFERENT rest distances, so a port that reads the wrong pair index lands
    // on a different penetration.
    R.bRd = R.mk(dev, std::vector<float>{ REST_DISTANCE, REST_DISTANCE_B });
    R.bMat = R.mk(dev, std::vector<uint32_t>(4u, POISON_U32));
    // Pair 0 owns TWO cells, so `pos` and `workIndex` part company from work item
    // 1 onward -- without that, restDistances[pos] and restDistances[workIndex]
    // are the same expression.  Both of pair 0's work items resolve to a live
    // cell, so its contacts are produced TWICE and the oracle follows.
    R.bStart = R.mk(dev, std::vector<uint32_t>{ 0u, 2u });
    R.bTotal = R.mk(dev, std::vector<uint32_t>{ 3u });
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
    R.blob.numTests = 2u;
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

// Which (system, particle) pairs must produce a contact, analytically.
//   system 0: no CCD -> accepted when signedDistance(origin) <= cDistance.
//   system 1: CCD    -> the two-pass test spelled out at particlePredicted.
static std::vector<Want> expectedSet()
{
    std::vector<Want> out;
    for (uint32_t sys = 0; sys < 2u; ++sys)
    {
        const float rest = sys ? REST_DISTANCE_B : REST_DISTANCE;
        for (uint32_t i = CELL_START; i < CELL_END; ++i)
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
        // Pair 0 owns TWO work items over the same cell, so system A's contacts
        // are produced twice.  Emitting the duplicate here is what keeps the
        // multiset match honest.
        if (sys == 0u)
        {
            const size_t n = out.size();
            for (size_t w = 0; w < n; ++w) out.push_back(out[w]);
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
        std::printf("    [ note ] %zu contacts (%u no-CCD, %u CCD) over %u cell particles; "
                    "%u separated, %u penetrating, %u off the side, 1 filtered per system; "
                    "worst |normal|-1 %.3g, worst penetration error %.3g\n",
                    want.size(), sysA, sysB, CELL_END - CELL_START,
                    separated, penetrating, sideFace, worstNormErr, worstPenErr);
    }

    R.destroy(dev);
}

} // namespace

PXG_TEST(psConvexContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}
