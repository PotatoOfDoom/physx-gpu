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

// P6.4a-2 — verification of psContactGenBda.slang (ps_primitivesCollisionLaunch),
// the particle-vs-primitive contact GENERATOR, driven through the REAL 15-arg
// engine ABI blob (both PGS and TGS modes) against REAL engine descriptors
// (PxgParticleSystem, PxgContactManagerInput, PxgShape, PxsCachedTransform,
// PxgRigidFilterPair, PxgParticleContactWriter).
//
// THE KEY PROOF (no scene observable this tranche): ONE particle penetrating a
// static PLANE.  A y=0 plane (a PxgShape of type ePLANE with a transformCache
// whose quaternion rotates the canonical PxPlane(1,0,0,0) normal onto +Y, and a
// cmInput pairing the particle system with it); a single-cell grid whose
// tempCellsHistogram/totalPairs make the cell loop find exactly the one
// particle; contactOffset ~0.3; particle at y=-0.1.  Running the kernel must emit
// EXACTLY ONE PxgParticlePrimitiveContact with normal (0,1,0) and penetration
// dist - restDistance = -0.1, BIT-EXACT vs a CPU transcription of the ePLANE
// branch (the tests build with -ffp-contract=off, matching the shader's
// NoContraction, so the separately-rounded quatRotate + plane-distance chain is
// bitwise reproducible).  The six radix staging arrays and the counter are
// checked too.
//
// P9.86c: the same rig also runs particles resting EXACTLY on the plane
// through the origin, with y = -0 (a negated zero) and restDistance +0.  The
// CUDA's plane offset is PxPlane::transform's `d - p.dot(n)` with d = 0
// (PxPlane.h:137, cudaParticleSystem.cu:285), an IEEE `0 - x` that is +0 for
// either zero, so dist = (-0) + (+0) = +0 and the stored pen is +0.  A unary
// minus would give -0 + -0 = -0.  The CPU mirror spells the CUDA's `0 - x`.

#include "pxgTestFramework.h"
#include "pxgGridCalRef.h"

// host_defines.h expands __device__/__align__ before the engine headers
// (nocuda/ is on the include path), matching testPsBounds.cpp.
#include "host_defines.h"
#include "PxgParticleSystem.h"     // PxgParticleSystem, PxgParticleSimBuffer, PxgParticleContactWriter, PxgParticlePrimitiveContact
#include "PxgContactManager.h"     // PxgContactManagerInput
#include "PxgConvexConvexShape.h"  // PxgShape
#include "PxgSimulationCoreDesc.h" // PxgRigidFilterPair
#include "PxsTransformCache.h"     // PxsCachedTransform
#include "PxNodeIndex.h"           // PxNodeIndex
#include "geometry/PxGeometry.h"   // PxGeometryType

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
static DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

static PxgParticleSystem makeZeroPs()
{
    PxgParticleSystem ps;
    std::memset(&ps, 0, sizeof(ps));
    return ps;
}

// CPU transcription of psContactGenBda's quatRotate (== PxQuat::rotate).
static void cpuQuatRotate(const float q[4], const float v[3], float out[3])
{
    const float vx = 2.0f * v[0], vy = 2.0f * v[1], vz = 2.0f * v[2];
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    out[0] = (vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2);
    out[1] = (vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2);
    out[2] = (vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2);
}

// The 176-byte engine ABI blob (15 args + the by-value 72-byte writer @104).
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
    // ---- PxgParticleContactWriter (by value, 72B) @104 ----
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

// PxgParticlePrimitiveContact readback mirror (32 bytes).
struct HostContact { F4 normal_pen; uint64_t rigidId; uint64_t particleId; };


// ===========================================================================
// P9.25 -- THE GRID, at the LAST of calcGridHashInBounds' four live call sites.
//
// The battery above drives ONE particle against ONE plane on a 1x1x1 grid and
// says so.  That is the right shape for what it proves -- a bit-exact ePLANE
// branch -- and it is left exactly as it was.  What it cannot prove is the grid:
// with one cell, calcWrappedGridRangeSize clamps the range to (1,1,1),
// calcGridOffsetInRange returns (0,0,0) for every offset, and calcGridHash masks
// every component with (1-1) == 0, so ANY implementation returning zero passes.
//
// P9.18 pinned :1215 and :1448, P9.24 pinned :888.  This is :656,
// ps_primitivesCollisionLaunch, the last one -- and the only one that reads the
// FLUID cell arrays (mCellStart / mCellEnd) rather than the diffuse ones.
//
// The rig is P9.24's, and the reasoning behind every number in it is written out
// there: the grid is 4 wide in x because three of the mutants are structurally
// invisible at gridSize 2; one profile sits inside cell 1 because otherwise the
// cell width is not under test; one profile declares fewer work items than its
// span because swapping the two offset components is invisible to any pair that
// reads its whole block; and the plane bounds are +-40 rather than +-8 because
// the overlap is an INTERSECTION and a bound of 8 caps the region before the
// clamp can bite.
// ===========================================================================
namespace psgrid
{
static const float GRID_CELL_WIDTH = 4.0f;
static const uint32_t GRID_SIZE[3] = { 4u, 2u, 1u };
static const uint32_t NB_CELLS = 8u;
static const uint32_t NB_PLANES = 12u;          // two full profile cycles
static const float CONTACT_OFFSET = 0.3f;
static const uint32_t NODE_BASE = 100u;

static const float PS_BMN[3] = { -1.0f, -1.0f, -1.0f };
static const float PS_BMX[3] = { 17.0f, 5.0f, 1.0f };

enum Profile { P_A = 0, P_B, P_C, P_D, P_E, P_F, NB_PROFILES };
static Profile profileOf(uint32_t pair) { return Profile(pair % NB_PROFILES); }

static void planeBoundsFor(uint32_t pair, float bmn[3], float bmx[3])
{
    for (int k = 0; k < 3; ++k) { bmn[k] = -40.0f; bmx[k] = 40.0f; }
    switch (profileOf(pair))
    {
    case P_A: case P_E: break;
    case P_B: bmn[1] = 0.5f; bmx[1] = 2.0f; break;
    case P_C: bmn[0] = 0.5f; bmx[0] = 2.0f; bmn[1] = 0.5f; bmx[1] = 2.0f; break;
    case P_D: bmn[0] = 4.5f; bmx[0] = 5.0f; bmn[1] = 0.5f; bmx[1] = 2.0f; break;
    case P_F: bmn[0] = 0.5f; bmx[0] = 5.0f; break;
    default: break;
    }
}

static pxgtest::gridcal::U3 gridU3()
{ return pxgtest::gridcal::U3{ GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2] }; }

static void pairOverlap(uint32_t pair, float omn[3], float omx[3])
{
    float pmn[3], pmx[3];
    planeBoundsFor(pair, pmn, pmx);
    pxgtest::gridcal::overlapBound(PS_BMN, PS_BMX, pmn, pmx, CONTACT_OFFSET, omn, omx);
}

static uint32_t spanOfPair(uint32_t pair)
{
    float omn[3], omx[3];
    pairOverlap(pair, omn, omx);
    return pxgtest::gridcal::cellsInBounds(omn, omx, GRID_CELL_WIDTH, gridU3());
}

static const uint32_t PROFILE_E_ITEMS = 2u;
static uint32_t cellsOfPair(uint32_t pair)
{
    const uint32_t span = spanOfPair(pair);
    return profileOf(pair) == P_E ? std::min(PROFILE_E_ITEMS, span) : span;
}

static uint32_t cellOfWorkItem(uint32_t pair, uint32_t off)
{
    float omn[3], omx[3];
    pairOverlap(pair, omn, omx);
    return pxgtest::gridcal::calcGridHashInBounds(omn, omx, GRID_CELL_WIDTH, off, gridU3())
           % NB_CELLS;
}

static uint32_t pairStartIndex(uint32_t pair)
{
    uint32_t acc = 0;
    for (uint32_t j = 0; j < pair; ++j) acc += cellsOfPair(j);
    return acc;
}
static uint32_t totalWorkItems() { return pairStartIndex(NB_PLANES); }

// The cell contents are DERIVED: every cell some pair reads gets a distinct
// particle count, unread cells stay empty.  See P9.24 for why two cells holding
// the same number are interchangeable and therefore untestable.
struct CellTable { uint32_t lo[NB_CELLS], hi[NB_CELLS]; bool read[NB_CELLS];
                   uint32_t nbRead = 0, live = 0; };

static const CellTable& cells()
{
    static CellTable t = [] {
        CellTable c{};
        for (uint32_t i = 0; i < NB_CELLS; ++i) { c.lo[i] = c.hi[i] = 0u; c.read[i] = false; }
        for (uint32_t j = 0; j < NB_PLANES; ++j)
        {
            const uint32_t n = cellsOfPair(j);
            for (uint32_t o = 0; o < n; ++o) c.read[cellOfWorkItem(j, o)] = true;
        }
        uint32_t cursor = 0u, span = 1u;
        for (uint32_t i = 0; i < NB_CELLS; ++i)
        {
            c.lo[i] = cursor;
            if (c.read[i]) { cursor += span; ++span; ++c.nbRead; }
            c.hi[i] = cursor;
        }
        c.live = cursor;
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

static bool pairReadsParticle(uint32_t pair, uint32_t i)
{
    const uint32_t want = cellOfParticle(i);
    if (want == 0xFFFFFFFFu) return false;
    const uint32_t n = cellsOfPair(pair);
    for (uint32_t o = 0; o < n; ++o) if (cellOfWorkItem(pair, o) == want) return true;
    return false;
}

// Plane j sits at y = -0.5 + 0.05 j; particle i at y = -0.45 + 0.037 i.
static float planeHeight(uint32_t j) { return -0.5f + 0.05f * float(j); }
static float particleY(uint32_t i) { return -0.45f + 0.037f * float(i); }
static float restDistanceFor(uint32_t j) { return 0.001f * float(j + 1u); }
} // namespace psgrid

// particlePosIn / planePIn: the particle's sorted origin position and the plane
// pose's translation.  approxPen: the tolerant sanity value of the penetration.
// expectPenBits: 0xffffffff = no pin, otherwise the bits the CUDA-faithful
// reference pen must have.  *signCases counts runs on which the old port's unary
// minus (`pdd = -dot(n, p)`) would store a different pen than the CUDA's `0 - x`.
static void runPlaneContact(bool isTGS, const float particlePosIn[3], const float planePIn[3],
                            float approxPen, uint32_t expectPenBits, uint32_t* signCases)
{
    // ---- pin every byte layout the ABI blob + shader bake in ----
    static_assert(sizeof(PxgParticleSystem) == 656, "PxgParticleSystem stride");
    static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput stride");
    static_assert(sizeof(PxgShape) == 48, "PxgShape stride");
    static_assert(offsetof(PxgShape, type) == 40, "PxgShape::type");
    static_assert(offsetof(PxgShape, particleOrSoftbodyId) == 44, "PxgShape::particleOrSoftbodyId");
    static_assert(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform stride");
    static_assert(sizeof(PxgRigidFilterPair) == 32, "PxgRigidFilterPair stride");
    static_assert(sizeof(PxgParticleSimBuffer) == 88, "PxgParticleSimBuffer stride");
    static_assert(offsetof(PxgParticleSimBuffer, mFilterPairs) == 40, "mFilterPairs offset");
    static_assert(offsetof(PxgParticleSimBuffer, mNumFilterPairs) == 64, "mNumFilterPairs offset");
    // The by-value writer must be exactly 72 bytes so it fits the ABI blob tail.
    static_assert(sizeof(PxgParticleContactWriter) == 72, "PxgParticleContactWriter by-value size");
    static_assert(sizeof(PsCollideBlob) == 176, "engine ABI blob size");
    static_assert(offsetof(PsCollideBlob, cmInputs) == 16, "cmInputs @16");
    static_assert(offsetof(PsCollideBlob, w_particleContacts) == 104, "writer @104");
    static_assert(offsetof(PsCollideBlob, w_maxContacts) == 168, "maxContacts @168");
    static_assert(sizeof(PxgParticlePrimitiveContact) == 32, "PxgParticlePrimitiveContact stride");

    PxgSlangComputeDevice* dev = testDevice();

    // ---- the static plane: rotate PxPlane(1,0,0,0) normal (+X) onto +Y ----
    // q = rotation of +90 deg about z = (0, 0, sin(pi/4), cos(pi/4)); p = origin.
    const float s = std::sin(0.78539816339744830961f); // pi/4
    const float c = std::cos(0.78539816339744830961f);
    const float planeQ[4] = { 0.0f, 0.0f, s, c };
    const float planeP[3] = { planePIn[0], planePIn[1], planePIn[2] };

    const float particlePos[3] = { particlePosIn[0], particlePosIn[1], particlePosIn[2] };
    const float contactOffset = 0.3f;
    const float restDistance = 0.0f;
    const uint32_t rigidNodeIndex = 7u;

    // ---- CPU transcription of the ePLANE branch (bit-exact ground truth) ----
    float refN[3];
    const float planeCanonicalNormal[3] = { 1.0f, 0.0f, 0.0f };
    cpuQuatRotate(planeQ, planeCanonicalNormal, refN);
    // P9.86c: PxPlane(1,0,0,0).transform (PxPlane.h:137) is `d - pose.p.dot(n)` with
    // d = 0 -- an IEEE `0 - x` (+0 for either zero), NOT a unary minus.
    const float pdd = 0.0f - (refN[0] * planeP[0] + refN[1] * planeP[1] + refN[2] * planeP[2]);
    // no CCD -> cVolumePos == particlePos, cVolumeRadius == contactOffset
    const float refDist = refN[0] * particlePos[0] + refN[1] * particlePos[1] + refN[2] * particlePos[2] + pdd;
    const float refPen = refDist - restDistance;
    PXG_CHECK_MSG(refDist <= contactOffset, "reference: particle must be within the contact offset");
    // the old port's twin: pdd = -dot(n, p)
    const float pddPortOld = -(refN[0] * planeP[0] + refN[1] * planeP[1] + refN[2] * planeP[2]);
    const float refPenPortOld = (refN[0] * particlePos[0] + refN[1] * particlePos[1] +
                                 refN[2] * particlePos[2] + pddPortOld) - restDistance;
    if (fbits(refPenPortOld) != fbits(refPen)) ++*signCases;
    if (expectPenBits != 0xffffffffu)
        PXG_CHECK_MSG(fbits(refPen) == expectPenBits,
                      "%s: reference pen %08x != expected %08x (CUDA `0 - x` plane offset)",
                      isTGS ? "TGS" : "PGS", fbits(refPen), expectPenBits);

    // ---- particle system descriptor: 1 cell, 1 particle, no CCD, no filters ----
    std::vector<F4> sortedOrigin = { { particlePos[0], particlePos[1], particlePos[2], 1.0f } };
    std::vector<F4> sortedPred   = sortedOrigin; // unused (no CCD), present for ABI parity
    std::vector<uint32_t> cellStart = { 0u };            // cell 0 -> particle 0
    std::vector<uint32_t> cellEnd   = { 1u };
    std::vector<uint32_t> sortedToUnsorted = { 0u };
    std::vector<uint32_t> bufferRunsum = { 0u };
    DevBuf bOrigin = makeDev(dev, sortedOrigin);
    DevBuf bPred   = makeDev(dev, sortedPred);
    DevBuf bCellStart = makeDev(dev, cellStart);
    DevBuf bCellEnd   = makeDev(dev, cellEnd);
    DevBuf bS2U       = makeDev(dev, sortedToUnsorted);
    DevBuf bRunsum    = makeDev(dev, bufferRunsum);

    // one sim buffer with NO rigid filter pairs -> find() always false
    std::vector<PxgParticleSimBuffer> simBuffers(1);
    std::memset(simBuffers.data(), 0, sizeof(PxgParticleSimBuffer));
    simBuffers[0].mNumFilterPairs = 0u;
    simBuffers[0].mFilterPairs = nullptr;
    DevBuf bSimBuf = makeDev(dev, simBuffers);

    PxgParticleSystem ps = makeZeroPs();
    ps.mCommonData.mGridCellWidth = 1.0f;
    ps.mCommonData.mGridSizeX = 1u; // power-of-two 1x1x1 -> gridHash is always 0
    ps.mCommonData.mGridSizeY = 1u;
    ps.mCommonData.mGridSizeZ = 1u;
    ps.mCommonData.mNumParticles = 1u;
    ps.mCommonData.mNumParticleBuffers = 1u;
    ps.mData.mFlags = 0u; // no eENABLE_SPECULATIVE_CCD
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bOrigin.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPred.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bS2U.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSimBuf.addr));
    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- narrowphase descriptors: pair 0 = (particle slot0, plane slot1) ----
    std::vector<PxgContactManagerInput> cmInputs(1);
    cmInputs[0].shapeRef0 = 0u;          // particle shape
    cmInputs[0].shapeRef1 = 1u;          // plane shape
    cmInputs[0].transformCacheRef0 = 0u; // particle cache
    cmInputs[0].transformCacheRef1 = 1u; // rigid (plane) cache
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::ePLANE);
    DevBuf bShapes = makeDev(dev, shapes);

    // transformCache: index 1 (rigid) = plane transform; index 0 unused.
    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(planeP[0], planeP[1], planeP[2]),
                                      PxQuat(planeQ[0], planeQ[1], planeQ[2], planeQ[3]));
    DevBuf bTc = makeDev(dev, tcache);

    // bounds[0]=particle box, bounds[1]=rigid box (min<max so overlap is valid;
    // with a 1x1x1 grid the exact box is irrelevant, gridHash resolves to 0).
    std::vector<float> boundsF(2 * 6, 0.0f);
    const float pmn[3] = { particlePos[0] - 0.2f, particlePos[1] - 0.2f, particlePos[2] - 0.2f };
    const float pmx[3] = { particlePos[0] + 0.2f, particlePos[1] + 0.2f, particlePos[2] + 0.2f };
    for (int k = 0; k < 3; ++k)
    {
        boundsF[0 * 6 + k] = pmn[k]; boundsF[0 * 6 + 3 + k] = pmx[k];
        boundsF[1 * 6 + k] = pmn[k]; boundsF[1 * 6 + 3 + k] = pmx[k];
    }
    DevBuf bBounds = makeDev(dev, boundsF);

    std::vector<float> contactDist = { contactOffset, 0.0f }; // cDistance = 0.3 + 0
    DevBuf bCd = makeDev(dev, contactDist);
    std::vector<float> restDist = { restDistance };
    DevBuf bRd = makeDev(dev, restDist);
    std::vector<uint32_t> materials = { 0u }; // unused (ABI parity)
    DevBuf bMat = makeDev(dev, materials);
    std::vector<uint32_t> startIndices = { 0u }; // per-pair cell run-sum
    DevBuf bStart = makeDev(dev, startIndices);
    std::vector<uint32_t> totalPairs = { 1u };
    DevBuf bTotal = makeDev(dev, totalPairs);
    std::vector<uint64_t> remap = { 0ull, uint64_t(rigidNodeIndex) }; // shapeToRigidRemap[1] = node 7
    DevBuf bRemap = makeDev(dev, remap);

    // ---- writer output buffers ----
    const uint32_t MAX_CONTACTS = 64u;
    std::vector<HostContact> contacts(MAX_CONTACTS);
    std::memset(contacts.data(), 0, contacts.size() * sizeof(HostContact));
    std::vector<uint32_t> count = { 0u };
    std::vector<uint64_t> sortedByParticle(MAX_CONTACTS, 0xffffffffffffffffull);
    std::vector<uint32_t> tempByParticle(MAX_CONTACTS, 0xffffffffu);
    std::vector<uint32_t> idxByParticle(MAX_CONTACTS, 0xffffffffu);
    std::vector<uint64_t> byRigid(MAX_CONTACTS, 0xffffffffffffffffull);
    std::vector<uint32_t> tempByRigid(MAX_CONTACTS, 0xffffffffu);
    std::vector<uint32_t> idxByRigid(MAX_CONTACTS, 0xffffffffu);
    DevBuf bContacts = makeDev(dev, contacts);
    DevBuf bCount = makeDev(dev, count);
    DevBuf bSbP = makeDev(dev, sortedByParticle);
    DevBuf bTbP = makeDev(dev, tempByParticle);
    DevBuf bIbP = makeDev(dev, idxByParticle);
    DevBuf bBr = makeDev(dev, byRigid);
    DevBuf bTbr = makeDev(dev, tempByRigid);
    DevBuf bIbr = makeDev(dev, idxByRigid);

    // ---- assemble the 176-byte ABI blob ----
    PsCollideBlob blob;
    std::memset(&blob, 0, sizeof(blob));
    blob.isTGSWord = isTGS ? 1u : 0u;
    blob.numTests = 1u;
    blob.toleranceLength = 1.0f;
    blob.cmInputs = bCin.addr;
    blob.shapes = bShapes.addr;
    blob.transformCache = bTc.addr;
    blob.bounds = bBounds.addr;
    blob.contactDistance = bCd.addr;
    blob.restDistances = bRd.addr;
    blob.materials = bMat.addr;
    blob.startIndices = bStart.addr;
    blob.totalNumPairs = bTotal.addr;
    blob.particleSystems = bDesc.addr;
    blob.shapeToRigidRemap = bRemap.addr;
    blob.w_particleContacts = bContacts.addr;
    blob.w_numTotalContacts = bCount.addr;
    blob.w_contactSortedByParticle = bSbP.addr;
    blob.w_tempContactByParticle = bTbP.addr;
    blob.w_contactIndexSortedByParticle = bIbP.addr;
    blob.w_contactByRigid = bBr.addr;
    blob.w_tempContactByRigid = bTbr.addr;
    blob.w_contactIndexSortedByRigid = bIbr.addr;
    blob.w_maxContacts = MAX_CONTACTS;

    // Dispatch the fixed CUDA grid (PS_COLLISION grid = 1024 blocks of 256).
    Kernel k("psContactGenBda.spv", "ps_primitivesCollisionLaunch", {}, sizeof(PsCollideBlob));
    k.dispatch(1024u, {}, blob, 1u, 1u);
    dev->flush();

    // ---- verify: EXACTLY ONE contact, bit-exact plane normal + penetration ----
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == 1u, "%s: writer emitted %u contacts (expected 1)", isTGS ? "TGS" : "PGS", gCount);

    const auto gC = downloadBuffer<HostContact>(dev, bContacts.h, 1);
    const HostContact& ct = gC[0];

    size_t mm = 0;
    if (fbits(ct.normal_pen.x) != fbits(refN[0])) { ++mm; std::printf("  normal.x gpu=%g ref=%g\n", ct.normal_pen.x, refN[0]); }
    if (fbits(ct.normal_pen.y) != fbits(refN[1])) { ++mm; std::printf("  normal.y gpu=%g ref=%g\n", ct.normal_pen.y, refN[1]); }
    if (fbits(ct.normal_pen.z) != fbits(refN[2])) { ++mm; std::printf("  normal.z gpu=%g ref=%g\n", ct.normal_pen.z, refN[2]); }
    if (fbits(ct.normal_pen.w) != fbits(refPen)) { ++mm; std::printf("  pen gpu=%g ref=%g\n", ct.normal_pen.w, refPen); }
    PXG_CHECK_MSG(mm == 0, "%s: contact normal_pen bit-exact vs CPU ePLANE transcription (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm);

    // human-readable sanity: normal ~ (0,1,0), penetration ~ approxPen
    PXG_CHECK_MSG(almostEqual(ct.normal_pen.x, 0.0f, 1e-5f, 1e-6f) &&
                  almostEqual(ct.normal_pen.y, 1.0f, 1e-5f, 1e-6f) &&
                  almostEqual(ct.normal_pen.z, 0.0f, 1e-5f, 1e-6f),
                  "%s: contact normal ~ (0,1,0) (got %g %g %g)", isTGS ? "TGS" : "PGS",
                  ct.normal_pen.x, ct.normal_pen.y, ct.normal_pen.z);
    PXG_CHECK_MSG(almostEqual(ct.normal_pen.w, approxPen, 1e-5f, 1e-6f),
                  "%s: penetration ~ %g (got %g)", isTGS ? "TGS" : "PGS", approxPen, ct.normal_pen.w);

    // identities + staging arrays
    PXG_CHECK_MSG(ct.rigidId == uint64_t(rigidNodeIndex), "%s: contact rigidId %llu != %u",
                  isTGS ? "TGS" : "PGS", (unsigned long long)ct.rigidId, rigidNodeIndex);
    PXG_CHECK_MSG(ct.particleId == 0ull, "%s: contact particleId %llu != 0 (systemId 0, particle 0)",
                  isTGS ? "TGS" : "PGS", (unsigned long long)ct.particleId);

    const uint64_t gSbP = downloadBuffer<uint64_t>(dev, bSbP.h, 1)[0];
    const uint32_t gTbP = downloadBuffer<uint32_t>(dev, bTbP.h, 1)[0];
    const uint32_t gIbP = downloadBuffer<uint32_t>(dev, bIbP.h, 1)[0];
    const uint64_t gBr = downloadBuffer<uint64_t>(dev, bBr.h, 1)[0];
    const uint32_t gTbr = downloadBuffer<uint32_t>(dev, bTbr.h, 1)[0];
    const uint32_t gIbr = downloadBuffer<uint32_t>(dev, bIbr.h, 1)[0];
    PXG_CHECK_MSG(gSbP == 0ull && gTbP == 0u && gIbP == 0u,
                  "%s: by-particle staging (sorted=%llu temp=%u idx=%u)", isTGS ? "TGS" : "PGS",
                  (unsigned long long)gSbP, gTbP, gIbP);
    PXG_CHECK_MSG(gBr == uint64_t(rigidNodeIndex) && gTbr == rigidNodeIndex && gIbr == 0u,
                  "%s: by-rigid staging (node=%llu temp=%u idx=%u)", isTGS ? "TGS" : "PGS",
                  (unsigned long long)gBr, gTbr, gIbr);

    PxgBufferHandle all[] = { bOrigin.h, bPred.h, bCellStart.h, bCellEnd.h, bS2U.h, bRunsum.h, bSimBuf.h,
                              bDesc.h, bCin.h, bShapes.h, bTc.h, bBounds.h, bCd.h, bRd.h, bMat.h,
                              bStart.h, bTotal.h, bRemap.h, bContacts.h, bCount.h, bSbP.h, bTbP.h,
                              bIbP.h, bBr.h, bTbr.h, bIbr.h };
    for (PxgBufferHandle h : all)
        dev->destroyBuffer(h);
}


// ---------------------------------------------------------------------------
// P9.25 -- the grid battery.  Many particles over eight cells, twelve planes
// over six bounds profiles, and one contact per (particle, plane) pair whose
// work items resolve to the cell holding that particle.
// ---------------------------------------------------------------------------
static void runGridContact(bool isTGS)
{
    using namespace psgrid;
    PxgSlangComputeDevice* dev = testDevice();
    const CellTable& CT = cells();

    // ---- the rig's own invariants, checked rather than asserted ----
    PXG_CHECK_MSG(CT.nbRead >= 4u, "only %u of %u cells are read", CT.nbRead, NB_CELLS);
    uint32_t dup = 0;
    for (uint32_t a = 0; a < NB_CELLS; ++a)
        for (uint32_t b = a + 1u; b < NB_CELLS; ++b)
            if (CT.read[a] && CT.read[b] && (CT.hi[a] - CT.lo[a]) == (CT.hi[b] - CT.lo[b])) ++dup;
    PXG_CHECK_MSG(dup == 0u, "%u read-cell pairs hold the same particle count -- those "
                  "cells are interchangeable and a swap is invisible", dup);
    // The clamp must be OBSERVABLE, and this says so by running the mutant.
    uint32_t clampVisible = 0, subsetPairs = 0, distinct = 0;
    uint32_t masks[NB_PLANES];
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        float omn[3], omx[3];
        pairOverlap(j, omn, omx);
        const pxgtest::gridcal::I3 mn = pxgtest::gridcal::calcGridPos(omn, GRID_CELL_WIDTH);
        const pxgtest::gridcal::I3 mx = pxgtest::gridcal::calcGridPos(omx, GRID_CELL_WIDTH);
        const pxgtest::gridcal::U3 cl = pxgtest::gridcal::calcWrappedGridRangeSize(mn, mx, gridU3());
        const pxgtest::gridcal::U3 raw{ uint32_t(std::max(mx.x - mn.x + 1, 0)),
                                        uint32_t(std::max(mx.y - mn.y + 1, 0)),
                                        uint32_t(std::max(mx.z - mn.z + 1, 0)) };
        const uint32_t n = cellsOfPair(j);
        if (n < spanOfPair(j)) ++subsetPairs;
        uint32_t m = 0;
        for (uint32_t o = 0; o < n; ++o)
        {
            m |= 1u << cellOfWorkItem(j, o);
            const pxgtest::gridcal::U3 oc = pxgtest::gridcal::calcGridOffsetInRange(cl, o);
            const pxgtest::gridcal::U3 orw = pxgtest::gridcal::calcGridOffsetInRange(raw, o);
            const uint32_t hc = pxgtest::gridcal::calcGridHash(
                pxgtest::gridcal::I3{ mn.x + int32_t(oc.x), mn.y + int32_t(oc.y),
                                      mn.z + int32_t(oc.z) }, gridU3());
            const uint32_t hr = pxgtest::gridcal::calcGridHash(
                pxgtest::gridcal::I3{ mn.x + int32_t(orw.x), mn.y + int32_t(orw.y),
                                      mn.z + int32_t(orw.z) }, gridU3());
            if (hc != hr) ++clampVisible;
        }
        masks[j] = m;
    }
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        bool seen = false;
        for (uint32_t k = 0; k < j; ++k) if (masks[k] == masks[j]) seen = true;
        if (!seen) ++distinct;
    }
    PXG_CHECK_MSG(clampVisible > 0u, "dropping the clamp in calcWrappedGridRangeSize changes "
                  "NO cell any pair reads -- the clamp is not under test");
    PXG_CHECK_MSG(subsetPairs > 0u, "no pair reads a PROPER SUBSET of its block");
    PXG_CHECK_MSG(distinct >= 4u, "the pairs read only %u distinct cell sets", distinct);

    const uint32_t NB_PARTICLES = CT.live;
    PXG_CHECK_MSG(NB_PARTICLES > 0u, "the derived cell table is empty");

    // ---- the plane transform, as in the battery above ----
    const float sq = std::sin(0.78539816339744830961f);
    const float cq = std::cos(0.78539816339744830961f);
    const float planeQ[4] = { 0.0f, 0.0f, sq, cq };
    float refN[3];
    const float canonical[3] = { 1.0f, 0.0f, 0.0f };
    cpuQuatRotate(planeQ, canonical, refN);

    // ---- the host mirror: one contact per (particle, plane) the grid admits ----
    struct Want { uint32_t particle, plane; float pen; };
    std::vector<Want> want;
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
        for (uint32_t j = 0; j < NB_PLANES; ++j)
        {
            if (!pairReadsParticle(j, i)) continue;
            const float ph = planeHeight(j);
            // PxPlane::transform: `0 - p.dot(n)` (PxPlane.h:137), not a unary minus
            const float pdd = 0.0f - (refN[0] * 0.0f + refN[1] * ph + refN[2] * 0.0f);
            const float py = particleY(i);
            const float dist = refN[0] * 0.0f + refN[1] * py + refN[2] * 0.0f + pdd;
            if (dist > CONTACT_OFFSET) continue;     // cVolumeRadius == cDistance, no CCD
            want.push_back(Want{ i, j, dist - restDistanceFor(j) });
        }
    PXG_CHECK_MSG(!want.empty(), "the rig predicts no contacts at all");

    // ---- descriptors ----
    std::vector<F4> pos(NB_PARTICLES);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i) pos[i] = F4{ 0.0f, particleY(i), 0.0f, 1.0f };
    std::vector<uint32_t> cellStart(NB_CELLS), cellEnd(NB_CELLS), s2u(NB_PARTICLES);
    for (uint32_t c = 0; c < NB_CELLS; ++c) { cellStart[c] = CT.lo[c]; cellEnd[c] = CT.hi[c]; }
    for (uint32_t i = 0; i < NB_PARTICLES; ++i) s2u[i] = i;
    DevBuf bPos = makeDev(dev, pos);
    DevBuf bCellStart = makeDev(dev, cellStart);
    DevBuf bCellEnd = makeDev(dev, cellEnd);
    DevBuf bS2U = makeDev(dev, s2u);
    DevBuf bRunsum = makeDev(dev, std::vector<uint32_t>{ 0u });

    std::vector<PxgParticleSimBuffer> simBuffers(1);
    std::memset(simBuffers.data(), 0, sizeof(PxgParticleSimBuffer));
    DevBuf bSimBuf = makeDev(dev, simBuffers);

    PxgParticleSystem ps = makeZeroPs();
    ps.mCommonData.mGridCellWidth = GRID_CELL_WIDTH;
    ps.mCommonData.mGridSizeX = GRID_SIZE[0];
    ps.mCommonData.mGridSizeY = GRID_SIZE[1];
    ps.mCommonData.mGridSizeZ = GRID_SIZE[2];
    ps.mCommonData.mNumParticles = NB_PARTICLES;
    ps.mCommonData.mNumParticleBuffers = 1u;
    ps.mData.mFlags = 0u;                                    // no CCD
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bS2U.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSimBuf.addr));
    DevBuf bDesc = makeDev(dev, std::vector<PxgParticleSystem>{ ps });

    const uint32_t NB_SHAPES = 1u + NB_PLANES;
    std::vector<PxgContactManagerInput> cin(NB_PLANES);
    std::memset(cin.data(), 0, cin.size() * sizeof(PxgContactManagerInput));
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        cin[j].shapeRef0 = 0u;        cin[j].transformCacheRef0 = 0u;
        cin[j].shapeRef1 = 1u + j;    cin[j].transformCacheRef1 = 1u + j;
    }
    DevBuf bCin = makeDev(dev, cin);

    std::vector<PxgShape> shapes(NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[0].particleOrSoftbodyId = 0u;
    for (uint32_t j = 0; j < NB_PLANES; ++j) shapes[1u + j].type = PxU32(PxGeometryType::ePLANE);
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<PxsCachedTransform> tcache(NB_SHAPES);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    for (uint32_t j = 0; j < NB_PLANES; ++j)
        tcache[1u + j].transform = PxTransform(PxVec3(0.0f, planeHeight(j), 0.0f),
                                               PxQuat(planeQ[0], planeQ[1], planeQ[2], planeQ[3]));
    DevBuf bTc = makeDev(dev, tcache);

    // P9.25: the bounds are what drive the grid hash, and nothing else.
    std::vector<float> boundsF(size_t(NB_SHAPES) * 6u, 0.0f);
    for (int k = 0; k < 3; ++k)
    { boundsF[k] = PS_BMN[k]; boundsF[3 + k] = PS_BMX[k]; }
    for (uint32_t j = 0; j < NB_PLANES; ++j)
    {
        float pmn[3], pmx[3];
        planeBoundsFor(j, pmn, pmx);
        for (int k = 0; k < 3; ++k)
        { boundsF[size_t(1u + j) * 6 + k] = pmn[k]; boundsF[size_t(1u + j) * 6 + 3 + k] = pmx[k]; }
    }
    DevBuf bBounds = makeDev(dev, boundsF);

    std::vector<float> cdist(NB_SHAPES, 0.0f);
    cdist[0] = CONTACT_OFFSET;
    DevBuf bCd = makeDev(dev, cdist);
    std::vector<float> rdist(NB_PLANES);
    for (uint32_t j = 0; j < NB_PLANES; ++j) rdist[j] = restDistanceFor(j);
    DevBuf bRd = makeDev(dev, rdist);
    DevBuf bMat = makeDev(dev, std::vector<uint32_t>(4u, 0u));

    std::vector<uint32_t> startIdx(NB_PLANES);
    for (uint32_t j = 0; j < NB_PLANES; ++j) startIdx[j] = pairStartIndex(j);
    DevBuf bStart = makeDev(dev, startIdx);
    DevBuf bTotal = makeDev(dev, std::vector<uint32_t>{ totalWorkItems() });

    std::vector<uint64_t> remap(NB_SHAPES, 0ull);
    for (uint32_t j = 0; j < NB_PLANES; ++j) remap[1u + j] = uint64_t(NODE_BASE + j);
    DevBuf bRemap = makeDev(dev, remap);

    const uint32_t MAX_CONTACTS = 2048u;
    DevBuf bContacts = makeDev(dev, std::vector<HostContact>(MAX_CONTACTS));
    DevBuf bCount = makeDev(dev, std::vector<uint32_t>{ 0u });
    DevBuf bSbP = makeDev(dev, std::vector<uint64_t>(MAX_CONTACTS, ~0ull));
    DevBuf bTbP = makeDev(dev, std::vector<uint32_t>(MAX_CONTACTS, ~0u));
    DevBuf bIbP = makeDev(dev, std::vector<uint32_t>(MAX_CONTACTS, ~0u));
    DevBuf bBr = makeDev(dev, std::vector<uint64_t>(MAX_CONTACTS, ~0ull));
    DevBuf bTbr = makeDev(dev, std::vector<uint32_t>(MAX_CONTACTS, ~0u));
    DevBuf bIbr = makeDev(dev, std::vector<uint32_t>(MAX_CONTACTS, ~0u));

    PsCollideBlob blob;
    std::memset(&blob, 0, sizeof(blob));
    blob.isTGSWord = isTGS ? 1u : 0u;
    blob.numTests = NB_PLANES;
    blob.toleranceLength = 1.0f;
    blob.cmInputs = bCin.addr;             blob.shapes = bShapes.addr;
    blob.transformCache = bTc.addr;        blob.bounds = bBounds.addr;
    blob.contactDistance = bCd.addr;       blob.restDistances = bRd.addr;
    blob.materials = bMat.addr;            blob.startIndices = bStart.addr;
    blob.totalNumPairs = bTotal.addr;      blob.particleSystems = bDesc.addr;
    blob.shapeToRigidRemap = bRemap.addr;
    blob.w_particleContacts = bContacts.addr;
    blob.w_numTotalContacts = bCount.addr;
    blob.w_contactSortedByParticle = bSbP.addr;
    blob.w_tempContactByParticle = bTbP.addr;
    blob.w_contactIndexSortedByParticle = bIbP.addr;
    blob.w_contactByRigid = bBr.addr;
    blob.w_tempContactByRigid = bTbr.addr;
    blob.w_contactIndexSortedByRigid = bIbr.addr;
    blob.w_maxContacts = MAX_CONTACTS;

    Kernel k("psContactGenBda.spv", "ps_primitivesCollisionLaunch", {}, sizeof(PsCollideBlob));
    k.dispatch(1024u, {}, blob, 1u, 1u);
    dev->flush();

    // ---- compare as a MULTISET: which contact lands in which slot is a race ----
    const char* lbl = isTGS ? "TGS" : "PGS";
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == uint32_t(want.size()), "%s: writer emitted %u contacts, the grid "
                  "oracle predicts %zu", lbl, gCount, want.size());
    if (gCount == uint32_t(want.size()) && gCount > 0u)
    {
        const auto gC = downloadBuffer<HostContact>(dev, bContacts.h, std::min(gCount, MAX_CONTACTS));
        std::vector<bool> used(want.size(), false);
        size_t mm = 0;
        for (uint32_t a = 0; a < gCount && a < MAX_CONTACTS; ++a)
        {
            bool found = false;
            for (size_t b = 0; b < want.size() && !found; ++b)
            {
                if (used[b]) continue;
                if (gC[a].particleId != uint64_t(want[b].particle)) continue;
                if (gC[a].rigidId != uint64_t(NODE_BASE + want[b].plane)) continue;
                if (fbits(gC[a].normal_pen.x) != fbits(refN[0]) ||
                    fbits(gC[a].normal_pen.y) != fbits(refN[1]) ||
                    fbits(gC[a].normal_pen.z) != fbits(refN[2])) continue;
                if (fbits(gC[a].normal_pen.w) != fbits(want[b].pen)) continue;
                used[b] = true; found = true;
            }
            if (!found)
            {
                if (mm < 4)
                    std::printf("    %s contact %u (particle %llu rigid %llu pen %g) has no "
                                "match in the grid oracle\n", lbl, a,
                                (unsigned long long)gC[a].particleId,
                                (unsigned long long)gC[a].rigidId, gC[a].normal_pen.w);
                ++mm;
            }
        }
        PXG_CHECK_MSG(mm == 0, "%s: %zu contacts differ from the grid oracle", lbl, mm);
    }

    std::printf("    [ grid ] %ux%ux%u cells, width %.1f; %u work items over %u pairs; "
                "%u cells read (%u particles), %u distinct cell sets, %u subset pair(s); "
                "%zu contacts\n",
                GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2], GRID_CELL_WIDTH,
                totalWorkItems(), NB_PLANES, CT.nbRead, CT.live, distinct, subsetPairs,
                want.size());

    for (PxgBufferHandle h : { bPos.h, bCellStart.h, bCellEnd.h, bS2U.h, bRunsum.h, bSimBuf.h,
                               bDesc.h, bCin.h, bShapes.h, bTc.h, bBounds.h, bCd.h, bRd.h,
                               bMat.h, bStart.h, bTotal.h, bRemap.h, bContacts.h, bCount.h,
                               bSbP.h, bTbP.h, bIbP.h, bBr.h, bTbr.h, bIbr.h })
        dev->destroyBuffer(h);
}

static void runPlaneContactCases(bool isTGS)
{
    uint32_t signCases = 0;
    const float origin[3] = { 0.0f, 0.0f, 0.0f };
    // the original case: particle 0.1 below the y=0 plane
    const float deep[3] = { 0.0f, -0.1f, 0.0f };
    runPlaneContact(isTGS, deep, origin, -0.1f, 0xffffffffu, &signCases);
    // P9.86c: a particle resting EXACTLY on the plane through the origin, its y a -0
    // from negating a zero, restDistance +0.  n = (-2^-24, 0x1.fffffep-1, +0), so
    // n.p = +0 and n.x*(+0) + n.y*(-0) + n.z*(-1) = -0: CUDA pdd = 0 - (+0) = +0,
    // dist = -0 + +0 = +0, pen = +0; the old port's unary minus stored -0.
    const float onPlane[3] = { 0.0f, -0.0f, -1.0f };
    runPlaneContact(isTGS, onPlane, origin, 0.0f, 0x00000000u, &signCases);
    // The same, with the plane pose at x = -0: n.p = (-2^-24)*(-0) + ... = +0 again,
    // and a negation pushed into the products ((-n).p) is still -0 here.
    const float negZeroOrigin[3] = { -0.0f, 0.0f, 0.0f };
    runPlaneContact(isTGS, onPlane, negZeroOrigin, 0.0f, 0x00000000u, &signCases);
    PXG_CHECK_MSG(signCases == 2u, "%s: the plane `0 - x` zero-sign case ran %u times (expected 2)",
                  isTGS ? "TGS" : "PGS", signCases);
}

PXG_TEST(psContactGen_planeContactPGS) { runPlaneContactCases(false); }
PXG_TEST(psContactGen_planeContactTGS) { runPlaneContactCases(true); }
PXG_TEST(psContactGen_gridHashPGS) { runGridContact(false); }
PXG_TEST(psContactGen_gridHashTGS) { runGridContact(true); }

} // namespace
