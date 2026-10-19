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
#include "PxgSimulationCoreDesc.h" // PxgDeformablePairInfo, PxgEncodeParticleRef
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

// The 192-byte engine ABI blob (15 args; the by-value 80-byte PxgRigidParticleContactWriter
// lands at @112 because the bridge aligns an 80-byte argument to 16 -- PhysX 5.12).
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
    uint64_t pad104;                           // @104 (paramAlign(80) = 16)
    // ---- PxgRigidParticleContactWriter (by value, 80B) @112 ----
    uint64_t w_particleContactInfos;           // @112
    uint64_t w_particleContactNormalPens;      // @120
    uint64_t w_numTotalContacts;               // @128
    uint64_t w_contactSortedByParticle;        // @136
    uint64_t w_tempContactByParticle;          // @144
    uint64_t w_contactIndexSortedByParticle;   // @152
    uint64_t w_contactByRigid;                 // @160
    uint64_t w_tempContactByRigid;             // @168
    uint64_t w_contactIndexSortedByRigid;      // @176
    uint32_t w_maxContacts;                    // @184
    uint32_t pad188;                           // @188 -> 192
};

// The contact as the fixtures compare it (5.6.1's PxgParticlePrimitiveContact shape).
struct HostContact { F4 normal_pen; uint64_t rigidId; uint64_t particleId; };

// PhysX 5.12: PxgRigidParticleContactWriter writes the particle-rigid contact as two
// SoA streams -- PxgDeformablePairInfo {ref0 = rigid PxNodeIndex, ref1 =
// PxgEncodeParticleRef(system, sortedIndex)} and a float4 normalPen -- where 5.6.1
// wrote one 32-byte PxgParticlePrimitiveContact.  The fixtures read both streams back
// and reassemble the old record shape (rigidId = ref0, particleId = ref1).
struct ContactStreams : DevBuf { PxgBufferHandle h2 = 0; uint64_t addr2 = 0; };
static ContactStreams makeContacts(PxgSlangComputeDevice* dev, uint32_t n)
{
    ContactStreams c;
    const DevBuf infos = makeDev(dev, std::vector<PxgDeformablePairInfo>(n, PxgDeformablePairInfo{ ~0ull, ~0ull }));
    const DevBuf normalPens = makeDev(dev, std::vector<F4>(n, F4{ 0.0f, 0.0f, 0.0f, 0.0f }));
    c.h = infos.h; c.addr = infos.addr;
    c.h2 = normalPens.h; c.addr2 = normalPens.addr;
    return c;
}
static std::vector<HostContact> downloadContacts(PxgSlangComputeDevice* dev, const ContactStreams& c, size_t n)
{
    const std::vector<PxgDeformablePairInfo> infos = downloadBuffer<PxgDeformablePairInfo>(dev, c.h, n);
    const std::vector<F4> normalPens = downloadBuffer<F4>(dev, c.h2, n);
    std::vector<HostContact> out(n);
    for (size_t i = 0; i < n; ++i)
    {
        out[i].normal_pen = normalPens[i];
        out[i].rigidId = infos[i].ref0;
        out[i].particleId = infos[i].ref1;
    }
    return out;
}


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
    static_assert(sizeof(PxgParticleSystem) == 608, "PxgParticleSystem stride");
    static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput stride");
    static_assert(sizeof(PxgShape) == 48, "PxgShape stride");
    static_assert(offsetof(PxgShape, type) == 40, "PxgShape::type");
    static_assert(offsetof(PxgShape, particleOrSoftbodyId) == 44, "PxgShape::particleOrSoftbodyId");
    static_assert(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform stride");
    static_assert(sizeof(PxgParticleSimBuffer) == 48, "PxgParticleSimBuffer stride");
    // The by-value writer is 80 bytes; the bridge aligns it to 16 (@112).
    static_assert(sizeof(PxgRigidParticleContactWriter) == 80, "PxgRigidParticleContactWriter by-value size");
    static_assert(sizeof(PsCollideBlob) == 192, "engine ABI blob size");
    static_assert(offsetof(PsCollideBlob, cmInputs) == 16, "cmInputs @16");
    static_assert(offsetof(PsCollideBlob, w_particleContactInfos) == 112, "writer @112");
    static_assert(offsetof(PsCollideBlob, w_maxContacts) == 184, "maxContacts @184");
    static_assert(sizeof(PxgDeformablePairInfo) == 16, "PxgDeformablePairInfo stride");

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

    // one sim buffer (5.12 has no particle-rigid filter pairs any more)
    std::vector<PxgParticleSimBuffer> simBuffers(1);
    std::memset(simBuffers.data(), 0, sizeof(PxgParticleSimBuffer));
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
        std::vector<uint32_t> count = { 0u };
    std::vector<uint64_t> sortedByParticle(MAX_CONTACTS, 0xffffffffffffffffull);
    std::vector<uint32_t> tempByParticle(MAX_CONTACTS, 0xffffffffu);
    std::vector<uint32_t> idxByParticle(MAX_CONTACTS, 0xffffffffu);
    std::vector<uint64_t> byRigid(MAX_CONTACTS, 0xffffffffffffffffull);
    std::vector<uint32_t> tempByRigid(MAX_CONTACTS, 0xffffffffu);
    std::vector<uint32_t> idxByRigid(MAX_CONTACTS, 0xffffffffu);
    ContactStreams bContacts = makeContacts(dev, MAX_CONTACTS);
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
    blob.w_particleContactInfos = bContacts.addr; blob.w_particleContactNormalPens = bContacts.addr2;
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

    const auto gC = downloadContacts(dev, bContacts, 1);
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
                              bStart.h, bTotal.h, bRemap.h, bContacts.h, bContacts.h2, bCount.h, bSbP.h, bTbP.h,
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
    ContactStreams bContacts = makeContacts(dev, MAX_CONTACTS);
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
    blob.w_particleContactInfos = bContacts.addr; blob.w_particleContactNormalPens = bContacts.addr2;
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
        const auto gC = downloadContacts(dev, bContacts, std::min(gCount, MAX_CONTACTS));
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
                               bMat.h, bStart.h, bTotal.h, bRemap.h, bContacts.h, bContacts.h2, bCount.h,
                               bSbP.h, bTbP.h, bIbP.h, bBr.h, bTbr.h, bIbr.h })
        dev->destroyBuffer(h);
}

// P9.86f -- the eCAPSULE branch's distancePointSegmentSquared (sphereCollision.cuh:91-103) clamps its
// parameter with PxClamp = PxMin(1, PxMax(0, .)) over the float selects, which keep a NaN.  A particle
// at (0, +inf, 0) against a capsule along x (axis lane y = +0): nom = 0.5*1 + inf*(+0) + 0 = NaN.
// CUDA: t = NaN, distSegmentSq = NaN, the maxDist gate `NaN > x` is false, normal falls back to
// (1,0,0), dist = NaN -- ONE contact with a NaN penetration.  GLSL FClamp (the pre-P9.86f port) turns
// t into 0: distSegmentSq = |ap|^2 = inf and the gate rejects -- no contact.  The 1x1x1 grid puts the
// particle in cell 0 whatever its position (the same rig as runPlaneContact).
static bool capsuleRefP86f(bool fclampTwin, float rel[3], float halfHeight, float radius, float maxDist,
                           float n[3], float& dist)
{
    const float axis[3] = { halfHeight, 0.0f, 0.0f };      // capsuleDir (1,0,0) * halfHeight
    const float a[3] = { -axis[0], -axis[1], -axis[2] };
    float ap[3], ab[3];
    for (int k = 0; k < 3; ++k) { ap[k] = rel[k] - a[k]; ab[k] = axis[k] - a[k]; }
    const float nom = ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2];
    const float denom = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    const float q = nom / denom;
    float tv;
    if (fclampTwin) { const float m = (q > 0.0f) ? q : 0.0f; tv = (m < 1.0f) ? m : 1.0f; }   // FClamp: NaN -> 0
    else { const float lo = (0.0f > q) ? 0.0f : q; tv = (1.0f < lo) ? 1.0f : lo; }        // PxClamp selects
    const float t = denom > 0.0f ? tv : 0.0f;
    float v[3];
    for (int k = 0; k < 3; ++k) v[k] = ap[k] - ab[k] * t;
    const float dsSq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const float mds = maxDist + radius;
    if (dsSq > mds * mds) return false;
    const float ds = std::sqrt(dsSq);
    float dv[3];
    for (int k = 0; k < 3; ++k) dv[k] = rel[k] - axis[k] * (2.0f * t - 1.0f);
    for (int k = 0; k < 3; ++k) n[k] = (ds > 1.192092896e-07f) ? dv[k] * (1.0f / ds) : (k == 0 ? 1.0f : 0.0f);
    dist = ds - radius;
    return true;
}

static void runCapsuleNaNContact(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float particlePos[3] = { 0.0f, INFINITY, 0.0f };
    const float halfHeight = 0.5f, radius = 0.25f, contactOffset = 0.3f;
    const uint32_t rigidNodeIndex = 7u;

    float rel[3] = { particlePos[0], particlePos[1], particlePos[2] };   // capsule at the origin
    float refN[3], refDist = 0.0f, twinN[3], twinDist = 0.0f;
    const bool refHit = capsuleRefP86f(false, rel, halfHeight, radius, contactOffset, refN, refDist);
    const bool twinHit = capsuleRefP86f(true, rel, halfHeight, radius, contactOffset, twinN, twinDist);
    PXG_CHECK_MSG(refHit && std::isnan(refDist) && refN[0] == 1.0f && !twinHit,
                  "capsule NaN rig: CUDA mirror hit %d dist %g n.x %g, FClamp twin hit %d", int(refHit), refDist,
                  refN[0], int(twinHit));

    std::vector<F4> sortedOrigin = { { particlePos[0], particlePos[1], particlePos[2], 1.0f } };
    std::vector<F4> sortedPred = sortedOrigin;
    std::vector<uint32_t> cellStart = { 0u }, cellEnd = { 1u }, sortedToUnsorted = { 0u }, bufferRunsum = { 0u };
    DevBuf bOrigin = makeDev(dev, sortedOrigin);
    DevBuf bPred = makeDev(dev, sortedPred);
    DevBuf bCellStart = makeDev(dev, cellStart);
    DevBuf bCellEnd = makeDev(dev, cellEnd);
    DevBuf bS2U = makeDev(dev, sortedToUnsorted);
    DevBuf bRunsum = makeDev(dev, bufferRunsum);
    std::vector<PxgParticleSimBuffer> simBuffers(1);
    std::memset(simBuffers.data(), 0, sizeof(PxgParticleSimBuffer));
    DevBuf bSimBuf = makeDev(dev, simBuffers);

    PxgParticleSystem ps = makeZeroPs();
    ps.mCommonData.mGridCellWidth = 1.0f;
    ps.mCommonData.mGridSizeX = 1u;
    ps.mCommonData.mGridSizeY = 1u;
    ps.mCommonData.mGridSizeZ = 1u;
    ps.mCommonData.mNumParticles = 1u;
    ps.mCommonData.mNumParticleBuffers = 1u;
    ps.mData.mFlags = 0u;
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bOrigin.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPred.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bS2U.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSimBuf.addr));
    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);
    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::eCAPSULE);
    shapes[1].scale.scale = PxVec3(halfHeight, radius, 0.0f);   // scale.x = half height, scale.y = radius
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(0.0f, 0.0f, 0.0f), PxQuat(0.0f, 0.0f, 0.0f, 1.0f));
    DevBuf bTc = makeDev(dev, tcache);
    // finite pair bounds: on the 1x1x1 grid every work item reads cell 0 anyway
    std::vector<float> boundsF = { -1, -1, -1, 1, 1, 1,  -1, -1, -1, 1, 1, 1 };
    DevBuf bBounds = makeDev(dev, boundsF);
    std::vector<float> contactDist = { contactOffset, 0.0f };
    DevBuf bCd = makeDev(dev, contactDist);
    std::vector<float> restDist = { 0.0f };
    DevBuf bRd = makeDev(dev, restDist);
    std::vector<uint32_t> materials = { 0u };
    DevBuf bMat = makeDev(dev, materials);
    std::vector<uint32_t> startIndices = { 0u };
    DevBuf bStart = makeDev(dev, startIndices);
    std::vector<uint32_t> totalPairs = { 1u };
    DevBuf bTotal = makeDev(dev, totalPairs);
    std::vector<uint64_t> remap = { 0ull, uint64_t(rigidNodeIndex) };
    DevBuf bRemap = makeDev(dev, remap);

    const uint32_t MAX_CONTACTS = 64u;
        std::vector<uint32_t> count = { 0u };
    std::vector<uint64_t> u64s(MAX_CONTACTS, 0xffffffffffffffffull);
    std::vector<uint32_t> u32s(MAX_CONTACTS, 0xffffffffu);
    ContactStreams bContacts = makeContacts(dev, MAX_CONTACTS);
    DevBuf bCount = makeDev(dev, count);
    DevBuf bSbP = makeDev(dev, u64s), bTbP = makeDev(dev, u32s), bIbP = makeDev(dev, u32s);
    DevBuf bBr = makeDev(dev, u64s), bTbr = makeDev(dev, u32s), bIbr = makeDev(dev, u32s);

    PsCollideBlob blob;
    std::memset(&blob, 0, sizeof(blob));
    blob.isTGSWord = isTGS ? 1u : 0u;
    blob.numTests = 1u;
    blob.toleranceLength = 1.0f;
    blob.cmInputs = bCin.addr; blob.shapes = bShapes.addr; blob.transformCache = bTc.addr;
    blob.bounds = bBounds.addr; blob.contactDistance = bCd.addr; blob.restDistances = bRd.addr;
    blob.materials = bMat.addr; blob.startIndices = bStart.addr; blob.totalNumPairs = bTotal.addr;
    blob.particleSystems = bDesc.addr; blob.shapeToRigidRemap = bRemap.addr;
    blob.w_particleContactInfos = bContacts.addr; blob.w_particleContactNormalPens = bContacts.addr2; blob.w_numTotalContacts = bCount.addr;
    blob.w_contactSortedByParticle = bSbP.addr; blob.w_tempContactByParticle = bTbP.addr;
    blob.w_contactIndexSortedByParticle = bIbP.addr; blob.w_contactByRigid = bBr.addr;
    blob.w_tempContactByRigid = bTbr.addr; blob.w_contactIndexSortedByRigid = bIbr.addr;
    blob.w_maxContacts = MAX_CONTACTS;

    Kernel k("psContactGenBda.spv", "ps_primitivesCollisionLaunch", {}, sizeof(PsCollideBlob));
    k.dispatch(1024u, {}, blob, 1u, 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == 1u, "%s: capsule vs particle at infinity emitted %u contacts (CUDA: 1, NaN penetration)",
                  isTGS ? "TGS" : "PGS", gCount);
    if (gCount == 1u)
    {
        const HostContact ct = downloadContacts(dev, bContacts, 1)[0];
        PXG_CHECK_MSG(fbits(ct.normal_pen.x) == fbits(refN[0]) && fbits(ct.normal_pen.y) == fbits(refN[1]) &&
                      fbits(ct.normal_pen.z) == fbits(refN[2]) && std::isnan(ct.normal_pen.w),
                      "%s: capsule NaN contact normal/pen %08x %08x %08x %08x, CUDA (1,0,0) / NaN", isTGS ? "TGS" : "PGS",
                      fbits(ct.normal_pen.x), fbits(ct.normal_pen.y), fbits(ct.normal_pen.z), fbits(ct.normal_pen.w));
        PXG_CHECK_MSG(ct.rigidId == uint64_t(rigidNodeIndex) && ct.particleId == 0ull,
                      "%s: capsule NaN contact ids %llu / %llu", isTGS ? "TGS" : "PGS",
                      (unsigned long long)ct.rigidId, (unsigned long long)ct.particleId);
    }
    for (PxgBufferHandle h : { bOrigin.h, bPred.h, bCellStart.h, bCellEnd.h, bS2U.h, bRunsum.h, bSimBuf.h,
                               bDesc.h, bCin.h, bShapes.h, bTc.h, bBounds.h, bCd.h, bRd.h, bMat.h, bStart.h,
                               bTotal.h, bRemap.h, bContacts.h, bContacts.h2, bCount.h, bSbP.h, bTbP.h, bIbP.h, bBr.h, bTbr.h, bIbr.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P9.89 -- THE ORDER OF EVERY THREE-TERM SUM IN THE PRIMITIVE CONTACT CORE.
//
// PxVec3::dot and ::magnitudeSquared are `x*v.x + y*v.y + z*v.z` (PxVec3.h:173,
// :274).  C++ evaluates that as (x + y) + z, and each product is rounded on its
// own (the P9.54 reading of the CUDA as IEEE C++; this file builds with
// -ffp-contract=off).  The port spells every one of them `dot(a, b)`, and SPIR-V
// leaves OpDot's summation order to the driver.  llvmpipe (Mesa's lower_fdot)
// evaluates an UNDECORATED OpDot as x + (y + z) and a NoContraction-decorated one
// as (x + y) + z.  Both were measured on 65536 random cases with 0 misses.
// psContactGenBda was unstamped until P9.89, so every plane, sphere, capsule
// and box distance, and the speculative contact-volume radius, came out
// re-associated.  None of the batteries above could see it.  Their plane normal
// is (-2^-24, ~1, 0) and their particles sit at x = z = 0, so at most one
// product is ever non-zero.
//
// This battery is built so that each of the ten dot() sites of the primitive
// core, taken ALONE in the old order, changes at least one compared bit.  The
// twin is run to show it:
//   * rotated planes, spheres, capsules and boxes with non-round transforms, and
//     particles whose offsets to them have three non-zero components;
//   * a CCD-off system, where the plane distance at cVolumePos IS the stored
//     value, and a CCD-on system, where the stored value comes from particlePos
//     and getSpecVolume's radius only gates;
//   * two PLANTED knife edges for the two sites that only gate.  The first is a
//     CCD particle whose plane distance sits between the two possible
//     contact-volume radii (gridCal.cuh:139).  The second is a box whose contact
//     distance makes maxDist^2 sit between the two possible squared corner
//     distances (particleCollision.cuh:56);
//   * one plane offset and one capsule axis whose own dot differs between the two
//     orders, because those two sites are one number per shape, not per particle.
// The DIFFUSE twin (ps_primitivesDiffuseCollisionLaunch) runs the same
// particlePrimitiveCollision on the same rig, with isDiffuse = true, and is
// compared per particle as well.
// ===========================================================================
namespace p89
{
static const float PX_MAX = 3.4028235e38f;
static const float EPS_REAL = 1.192092896e-07f;

// One bit per dot() site of psContactGenBda.slang's primitive core.  A set bit
// evaluates that site in the pre-P9.89 port's order, x + (y + z).
enum : uint32_t
{
    S_SPEC = 1u << 0,      // :314 sqrt(dot(off, off))  gridCal.cuh:139 cVolumeOffset.magnitude()
    S_CAP_NOM = 1u << 1,   // :369 dot(ap, ab)          sphereCollision.cuh:97 ap.dot(ab)
    S_CAP_DEN = 1u << 2,   // :370 dot(ab, ab)          sphereCollision.cuh:98 ab.dot(ab)
    S_CAP_VV = 1u << 3,    // :383 dot(v, v)            sphereCollision.cuh:103 v.dot(v)
    S_BOX_GATE = 1u << 4,  // :395 dot(dv, dv)          particleCollision.cuh:56 magnitudeSquared()
    S_BOX_MAG = 1u << 5,   // :421 dot(n, n)            particleCollision.cuh:92 normalizeFast()
    S_SPH = 1u << 6,       // :433 dot(rel, rel)        particleCollision.cuh:107 magnitudeSquared()
    S_PL_P = 1u << 7,      // :484 dot(pn, shapeP)      PxPlane.h:137 `d - p.dot(n)` (cudaParticleSystem.cu:285)
    S_PL_CV = 1u << 8,     // :486 dot(pn, cVolumePos)  cudaParticleSystem.cu:286 plane.distance()
    S_PL_PP = 1u << 9,     // :490 dot(pn, particlePos) cudaParticleSystem.cu:294 plane.distance()
    NB_SITES = 10u,
    S_ALL = (1u << NB_SITES) - 1u
};
static const char* const SITE_NAME[NB_SITES] = {
    "getSpecVolume :314", "capsule nom :369", "capsule denom :370", "capsule v.v :383",
    "box gate :395", "box magnitude :421", "sphere :433", "plane offset :484",
    "plane dist(cVolumePos) :486", "plane dist(particlePos) :490" };

struct V { float x, y, z; };
static V vsub(V a, V b) { return V{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V vadd(V a, V b) { return V{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V vmul(V a, float s) { return V{ a.x * s, a.y * s, a.z * s }; }

// THE site under test.  CUDA: PxVec3::dot, (x + y) + z.  Old port: x + (y + z).
static float dotO(uint32_t oldMask, uint32_t site, V a, V b)
{
    const float x = a.x * b.x, y = a.y * b.y, z = a.z * b.z;
    return (oldMask & site) ? (x + (y + z)) : ((x + y) + z);
}

static V qRot(const float q[4], V v)
{
    const float in[3] = { v.x, v.y, v.z };
    float o[3];
    cpuQuatRotate(q, in, o);
    return V{ o[0], o[1], o[2] };
}
// PxQuat::rotateInv (PxQuat.h:324) == psContactGenBda's quatRotateInv.
static V qRotInv(const float q[4], V v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    return V{ (vx * w2 - (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2),
              (vy * w2 - (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2),
              (vz * w2 - (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2) };
}
// PxQuat::getBasisVector0 (PxQuat.h:260).
static V basis0(const float q[4])
{
    const float x2 = q[0] * 2.0f, w2 = q[3] * 2.0f;
    return V{ (q[3] * w2) - 1.0f + q[0] * x2, (q[2] * w2) + q[1] * x2, (-q[1] * w2) + q[2] * x2 };
}

// gridCal.cuh:122-140 with isDiffuse selecting the scale first.
static float specVolume(uint32_t om, V& cvp, V cur, V pred, float contactDist, bool isDiffuse, bool isTGS)
{
    const float scaleInv = isDiffuse ? (1.0f / 1.0f) : (isTGS ? (1.0f / 0.5f) : (1.0f / 1.0f));
    const V off = V{ (pred.x - cur.x) * scaleInv * 0.5f, (pred.y - cur.y) * scaleInv * 0.5f,
                     (pred.z - cur.z) * scaleInv * 0.5f };
    cvp = vadd(cur, off);
    return std::sqrt(dotO(om, S_SPEC, off, off)) + contactDist;
}

// particleCollision.cuh:101-121.
static bool sphereC(uint32_t om, V point, V c, float radius, float maxDist, V& n, float& d)
{
    const V rel = vsub(point, c);
    const float dsq = dotO(om, S_SPH, rel, rel);
    if (maxDist < PX_MAX)
    {
        const float m = maxDist + radius;
        if (dsq > m * m) return false;
    }
    const float dc = std::sqrt(dsq);
    n = (dc > EPS_REAL) ? vmul(rel, 1.0f / dc) : V{ 1.0f, 0.0f, 0.0f };
    d = dc - radius;
    return true;
}

// sphereCollision.cuh:91-104 + particleCollision.cuh:128-150.
static bool capsuleC(uint32_t om, V point, V c, V dir, float radius, float hh, float maxDist, V& n, float& d)
{
    const V axis = vmul(dir, hh);
    const V rel = vsub(point, c);
    const V a = V{ -axis.x, -axis.y, -axis.z };
    const V ap = vsub(rel, a), ab = vsub(axis, a);
    const float nom = dotO(om, S_CAP_NOM, ap, ab);
    const float den = dotO(om, S_CAP_DEN, ab, ab);
    const float q = nom / den;
    const float lo = (0.0f > q) ? 0.0f : q;          // PxClamp = PxMin(1, PxMax(0, .)), float selects
    const float tv = (1.0f < lo) ? 1.0f : lo;
    const float t = den > 0.0f ? tv : 0.0f;
    const V v = vsub(ap, vmul(ab, t));
    const float dsq = dotO(om, S_CAP_VV, v, v);
    if (maxDist < PX_MAX)
    {
        const float m = maxDist + radius;
        if (dsq > m * m) return false;
    }
    const V dsv = vsub(rel, vmul(axis, 2.0f * t - 1.0f));
    const float ds = std::sqrt(dsq);
    n = (ds > EPS_REAL) ? vmul(dsv, 1.0f / ds) : V{ 1.0f, 0.0f, 0.0f };
    d = ds - radius;
    return true;
}

static float clampF(float x, float lo, float hi) { return std::min(std::max(x, lo), hi); }

// particleCollision.cuh:42-95.
static bool boxC(uint32_t om, V point, V bp, const float q[4], V h, float maxDist, V& n, float& d)
{
    const V rel = qRotInv(q, vsub(point, bp));
    const V cl = V{ clampF(rel.x, -h.x, h.x), clampF(rel.y, -h.y, h.y), clampF(rel.z, -h.z, h.z) };
    if (maxDist < PX_MAX)
    {
        const float m = maxDist * maxDist;
        const V dv = vsub(rel, cl);
        if (dotO(om, S_BOX_GATE, dv, dv) > m) return false;
    }
    const bool inside = h.x >= std::fabs(rel.x) && h.y >= std::fabs(rel.y) && h.z >= std::fabs(rel.z);
    V nn;
    float dd;
    if (inside)
    {
        const V ds = V{ h.x - std::fabs(cl.x), h.y - std::fabs(cl.y), h.z - std::fabs(cl.z) };
        nn = cl.z >= 0.0f ? V{ 0, 0, 1 } : V{ 0, 0, -1 };
        dd = -ds.z;
        if (ds.x <= ds.y && ds.x <= ds.z) { nn = cl.x >= 0.0f ? V{ 1, 0, 0 } : V{ -1, 0, 0 }; dd = -ds.x; }
        else if (ds.y <= ds.z) { nn = cl.y >= 0.0f ? V{ 0, 1, 0 } : V{ 0, -1, 0 }; dd = -ds.y; }
    }
    else
    {
        nn = vsub(rel, cl);
        const float mag = std::sqrt(dotO(om, S_BOX_MAG, nn, nn));
        nn = vmul(nn, 1.0f / mag);
        dd = mag;
    }
    n = qRot(q, nn);
    d = dd;
    return true;
}

// PxGeometryType: eSPHERE 0, ePLANE 1, eCAPSULE 2, eBOX 3.
struct Shape { uint32_t type; float q[4]; V p; V scale; float cd; };

// particlePrimitiveCollision (cudaParticleSystem.cu:268-376).
static bool collide(uint32_t om, V pp, V cvp, float r, const Shape& s, bool ccd, V& n, float& d)
{
    switch (s.type)
    {
    case 1u:
    {
        const V pn = qRot(s.q, V{ 1.0f, 0.0f, 0.0f });
        const float pdd = 0.0f - dotO(om, S_PL_P, pn, s.p);       // PxPlane::transform, d = 0
        float dist = dotO(om, S_PL_CV, pn, cvp) + pdd;
        if (!(dist <= r)) return false;
        if (ccd) dist = dotO(om, S_PL_PP, pn, pp) + pdd;
        n = pn;
        d = dist;
        return true;
    }
    case 0u:
    {
        if (!sphereC(om, cvp, s.p, s.scale.x, r, n, d)) return false;
        if (ccd) sphereC(om, pp, s.p, s.scale.x, PX_MAX, n, d);
        return true;
    }
    case 2u:
    {
        const V dir = basis0(s.q);
        if (!capsuleC(om, cvp, s.p, dir, s.scale.y, s.scale.x, r, n, d)) return false;
        if (ccd) capsuleC(om, pp, s.p, dir, s.scale.y, s.scale.x, PX_MAX, n, d);
        return true;
    }
    case 3u:
    {
        if (!boxC(om, cvp, s.p, s.q, s.scale, r, n, d)) return false;
        if (ccd) boxC(om, pp, s.p, s.q, s.scale, PX_MAX, n, d);
        return true;
    }
    default: return false;
    }
}

struct Rng
{
    uint64_t s;
    float u() { s = s * 6364136223846793005ull + 1442695040888963407ull; return float(uint32_t(s >> 40)) / 16777216.0f; }
    float r(float a, float b) { return a + (b - a) * u(); }
    // a unit direction whose three components are all at least 0.2 in magnitude
    V dir()
    {
        for (;;)
        {
            const V v = V{ r(-1.0f, 1.0f), r(-1.0f, 1.0f), r(-1.0f, 1.0f) };
            const float m = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            if (m < 0.3f) continue;
            const V n = vmul(v, 1.0f / m);
            if (std::fabs(n.x) > 0.2f && std::fabs(n.y) > 0.2f && std::fabs(n.z) > 0.2f) return n;
        }
    }
    void quat(float q[4])
    {
        const V a = dir();
        const float ang = r(0.4f, 2.6f);
        const float s = std::sin(0.5f * ang), c = std::cos(0.5f * ang);
        const float m = std::sqrt(a.x * a.x * s * s + a.y * a.y * s * s + a.z * a.z * s * s + c * c);
        q[0] = a.x * s / m; q[1] = a.y * s / m; q[2] = a.z * s / m; q[3] = c / m;
    }
};

static const uint32_t NB_SYS = 2u;          // system 0: CCD off; system 1: CCD on
static const uint32_t NB_RIGID = 8u;
static const uint32_t NB_PART = 160u;
static const uint32_t NODE_BASE = 300u;
static const float SYS_CD[NB_SYS] = { 0.2113f, 0.1837f };
static const uint32_t K_SPEC_PARTICLE = 0u;   // system 1, planted against rigid 0
static const uint32_t K_BOX_PARTICLE = 6u;    // system 0, planted against rigid 6

struct Rig
{
    Shape shape[NB_RIGID];
    V cur[NB_SYS][NB_PART], pred[NB_SYS][NB_PART];
    float rest[NB_SYS * NB_RIGID];
};

static void placeNear(Rng& g, const Shape& s, V& p)
{
    switch (s.type)
    {
    case 1u:
    {
        const V pn = qRot(s.q, V{ 1.0f, 0.0f, 0.0f });
        // two tangents: pn x e_z and pn x (pn x e_z), unnormalised is fine for placement
        const V t1 = V{ pn.y, -pn.x, 0.0f };
        const V t2 = V{ pn.x * pn.z, pn.y * pn.z, -(pn.x * pn.x + pn.y * pn.y) };
        p = vadd(vadd(s.p, vadd(vmul(t1, g.r(-1.2f, 1.2f)), vmul(t2, g.r(-1.2f, 1.2f)))),
                 vmul(pn, g.r(-0.2f, 0.33f)));
        return;
    }
    case 0u:
        p = vadd(s.p, vmul(g.dir(), s.scale.x + g.r(-0.15f, 0.3f)));
        return;
    case 2u:
    {
        const V ax = basis0(s.q);
        const V rd = g.dir();
        V perp = V{ ax.y * rd.z - ax.z * rd.y, ax.z * rd.x - ax.x * rd.z, ax.x * rd.y - ax.y * rd.x };
        const float m = std::sqrt(perp.x * perp.x + perp.y * perp.y + perp.z * perp.z);
        perp = vmul(perp, 1.0f / m);
        p = vadd(vadd(s.p, vmul(ax, s.scale.x * g.r(-1.3f, 1.3f))), vmul(perp, s.scale.y + g.r(-0.12f, 0.3f)));
        return;
    }
    default:
    {
        const V l = V{ (g.u() < 0.5f ? -1.0f : 1.0f) * (s.scale.x + g.r(-0.15f, 0.2f)),
                       (g.u() < 0.5f ? -1.0f : 1.0f) * (s.scale.y + g.r(-0.15f, 0.2f)),
                       (g.u() < 0.5f ? -1.0f : 1.0f) * (s.scale.z + g.r(-0.15f, 0.2f)) };
        p = vadd(s.p, qRot(s.q, l));
        return;
    }
    }
}

static float nextUp(float x, int k)
{
    float y = x;
    for (int i = 0; i < k; ++i) y = std::nextafter(y, PX_MAX);
    for (int i = 0; i > k; --i) y = std::nextafter(y, -PX_MAX);
    return y;
}

// Builds the rig for one isTGS mode.  Returns false (and says why) if a planted
// knife edge could not be found -- the fixture then fails rather than silently
// testing less.
static bool buildRig(Rig& R, bool isTGS)
{
    Rng g{ 0x5eed89u };
    // ---- rigid 0: plane through the origin, normal ~+y tilted off every axis ----
    for (;;)
    {
        g.quat(R.shape[0].q);
        const V pn = qRot(R.shape[0].q, V{ 1, 0, 0 });
        if (pn.y > 0.75f && std::fabs(pn.x) > 0.2f && std::fabs(pn.z) > 0.2f) break;
    }
    R.shape[0].type = 1u; R.shape[0].p = V{ 0.0f, 0.0f, 0.0f }; R.shape[0].scale = V{ 1, 1, 1 };
    // ---- rigid 1: plane with normal ~+x through an off-axis point whose p.dot(n)
    //      differs between the two orders (S_PL_P is ONE number per plane) ----
    for (;;)
    {
        g.quat(R.shape[1].q);
        const V pn = qRot(R.shape[1].q, V{ 1, 0, 0 });
        if (!(pn.x > 0.75f && std::fabs(pn.y) > 0.2f && std::fabs(pn.z) > 0.2f)) continue;
        R.shape[1].p = V{ g.r(-3.3f, -2.9f), g.r(0.2f, 0.6f), g.r(0.3f, 0.7f) };
        if (dotO(0u, S_PL_P, pn, R.shape[1].p) != dotO(S_PL_P, S_PL_P, pn, R.shape[1].p)) break;
    }
    R.shape[1].type = 1u; R.shape[1].scale = V{ 1, 1, 1 };
    // ---- spheres ----
    R.shape[2] = Shape{ 0u, { 0, 0, 0, 1 }, V{ -1.31f, 1.62f, 0.43f }, V{ 0.4137f, 0.0f, 0.0f }, 0.0f };
    R.shape[3] = Shape{ 0u, { 0, 0, 0, 1 }, V{ 0.93f, 1.91f, -0.71f }, V{ 0.2671f, 0.0f, 0.0f }, 0.0f };
    // ---- capsules (scale.x = half height, scale.y = radius); capsule 4's |ab|^2
    //      differs between the orders (S_CAP_DEN is ONE number per capsule) ----
    R.shape[4] = Shape{ 2u, { 0, 0, 0, 1 }, V{ 2.13f, 1.73f, 0.31f }, V{ 0.5311f, 0.2219f, 0.0f }, 0.0f };
    for (;;)
    {
        g.quat(R.shape[4].q);
        const V ax = vmul(basis0(R.shape[4].q), R.shape[4].scale.x);
        const V ab = vsub(ax, V{ -ax.x, -ax.y, -ax.z });
        if (dotO(0u, S_CAP_DEN, ab, ab) != dotO(S_CAP_DEN, S_CAP_DEN, ab, ab)) break;
    }
    R.shape[5] = Shape{ 2u, { 0, 0, 0, 1 }, V{ -0.42f, 2.31f, -1.23f }, V{ 0.7723f, 0.1543f, 0.0f }, 0.0f };
    g.quat(R.shape[5].q);
    // ---- boxes (scale = half extents) ----
    R.shape[6] = Shape{ 3u, { 0, 0, 0, 1 }, V{ 1.21f, 1.43f, 1.52f }, V{ 0.4317f, 0.2861f, 0.3593f }, 0.0f };
    g.quat(R.shape[6].q);
    R.shape[7] = Shape{ 3u, { 0, 0, 0, 1 }, V{ -2.23f, 1.52f, -0.91f }, V{ 0.6121f, 0.3347f, 0.2518f }, 0.0f };
    g.quat(R.shape[7].q);
    for (uint32_t j = 0; j < NB_RIGID; ++j) R.shape[j].cd = 0.0071f + 0.0123f * float(j);
    for (uint32_t k = 0; k < NB_SYS * NB_RIGID; ++k) R.rest[k] = 0.0013f + 0.0021f * float(k);

    // ---- particles: particle i of either system sits near rigid i % NB_RIGID ----
    for (uint32_t s = 0; s < NB_SYS; ++s)
        for (uint32_t i = 0; i < NB_PART; ++i)
        {
            placeNear(g, R.shape[i % NB_RIGID], R.cur[s][i]);
            R.pred[s][i] = (s == 1u) ? vadd(R.cur[s][i], vmul(g.dir(), g.r(0.04f, 0.3f))) : R.cur[s][i];
        }

    // ---- K_BOX: a system-0 particle in a corner region of box 6 whose |dv|^2
    //      differs between the orders, and box 6's contact distance chosen so
    //      maxDist^2 = (SYS_CD[0] + cd)^2 falls between the two values ----
    {
        const Shape& b = R.shape[6];
        bool found = false;
        for (int trial = 0; trial < 20000 && !found; ++trial)
        {
            const V l = V{ (g.u() < 0.5f ? -1.0f : 1.0f) * (b.scale.x + g.r(0.13f, 0.2f)),
                           (g.u() < 0.5f ? -1.0f : 1.0f) * (b.scale.y + g.r(0.13f, 0.2f)),
                           (g.u() < 0.5f ? -1.0f : 1.0f) * (b.scale.z + g.r(0.13f, 0.2f)) };
            const V p = vadd(b.p, qRot(b.q, l));
            const V rel = qRotInv(b.q, vsub(p, b.p));
            const V cl = V{ clampF(rel.x, -b.scale.x, b.scale.x), clampF(rel.y, -b.scale.y, b.scale.y),
                            clampF(rel.z, -b.scale.z, b.scale.z) };
            const V dv = vsub(rel, cl);
            if (dv.x == 0.0f || dv.y == 0.0f || dv.z == 0.0f) continue;
            const float dF = dotO(0u, S_BOX_GATE, dv, dv), dR = dotO(S_BOX_GATE, S_BOX_GATE, dv, dv);
            if (dF == dR) continue;
            const float lo = std::min(dF, dR), hi = std::max(dF, dR);
            const float base = float(std::sqrt(double(lo)) - double(SYS_CD[0]));
            for (int k = -64; k <= 64 && !found; ++k)
            {
                const float c = nextUp(base, k);
                if (!(c > 0.0f)) continue;
                const float md = SYS_CD[0] + c;
                const float m2 = md * md;
                if (m2 >= lo && m2 < hi)
                {
                    R.shape[6].cd = c;
                    R.cur[0][K_BOX_PARTICLE] = p;
                    R.pred[0][K_BOX_PARTICLE] = p;
                    found = true;
                }
            }
        }
        if (!found) { std::printf("    p89: no K_BOX knife edge found\n"); return false; }
    }

    // ---- K_SPEC: a system-1 (CCD) particle whose plane-0 distance at cVolumePos
    //      sits between the two possible contact-volume radii ----
    {
        const V pn = qRot(R.shape[0].q, V{ 1, 0, 0 });
        const float cd = SYS_CD[1] + R.shape[0].cd;
        bool found = false;
        for (int trial = 0; trial < 20000 && !found; ++trial)
        {
            const V D = vmul(g.dir(), g.r(0.08f, 0.25f));
            V c = V{ g.r(-0.5f, 0.5f), g.r(-0.5f, 0.5f), g.r(-0.5f, 0.5f) };
            V cvp;
            float rF = specVolume(0u, cvp, c, vadd(c, D), cd, false, isTGS);
            // move c along the normal so the distance lands near the radius
            const float dist0 = dotO(0u, S_PL_CV, pn, cvp);
            c = vadd(c, vmul(pn, rF - dist0));
            for (int k = -400; k <= 400 && !found; ++k)
            {
                V ck = c;
                ck.y = nextUp(c.y, k);
                const V pk = vadd(ck, D);
                V cv;
                rF = specVolume(0u, cv, ck, pk, cd, false, isTGS);
                const float rR = specVolume(S_SPEC, cv, ck, pk, cd, false, isTGS);
                const float dist = dotO(0u, S_PL_CV, pn, cv) + (0.0f - dotO(0u, S_PL_P, pn, R.shape[0].p));
                if ((dist <= rF) != (dist <= rR))
                {
                    R.cur[1][K_SPEC_PARTICLE] = ck;
                    R.pred[1][K_SPEC_PARTICLE] = pk;
                    found = true;
                }
            }
        }
        if (!found) { std::printf("    p89: no K_SPEC knife edge found\n"); return false; }
    }
    return true;
}

struct Hit { uint32_t sys, particle, rigid; F4 np; };

// The contact stream of the fluid kernel, under a site mask.
static std::vector<Hit> predict(const Rig& R, uint32_t om, bool isDiffuse, bool isTGS)
{
    std::vector<Hit> out;
    for (uint32_t s = 0; s < NB_SYS; ++s)
        for (uint32_t j = 0; j < NB_RIGID; ++j)
        {
            const float cd = SYS_CD[s] + R.shape[j].cd;
            const float rest = R.rest[s * NB_RIGID + j];
            for (uint32_t i = 0; i < NB_PART; ++i)
            {
                V cvp = R.cur[s][i];
                float r = cd;
                const bool ccd = (s == 1u);
                if (ccd) r = specVolume(om, cvp, R.cur[s][i], R.pred[s][i], cd, isDiffuse, isTGS);
                V n{ 0, 0, 0 };
                float d = 0.0f;
                if (!collide(om, R.cur[s][i], cvp, r, R.shape[j], ccd, n, d)) continue;
                out.push_back(Hit{ s, i, j, F4{ n.x, n.y, n.z, d - rest } });
            }
        }
    return out;
}

static bool sameHit(const Hit& a, const Hit& b)
{
    return a.sys == b.sys && a.particle == b.particle && a.rigid == b.rigid &&
           fbits(a.np.x) == fbits(b.np.x) && fbits(a.np.y) == fbits(b.np.y) &&
           fbits(a.np.z) == fbits(b.np.z) && fbits(a.np.w) == fbits(b.np.w);
}

// How many hits of `a` have no bit-identical hit in `b`.
static size_t unmatched(const std::vector<Hit>& a, const std::vector<Hit>& b)
{
    size_t diff = 0;
    for (const Hit& h : a)
    {
        bool found = false;
        for (const Hit& k : b)
            if (sameHit(h, k)) { found = true; break; }
        if (!found) ++diff;
    }
    return diff;
}
// Both directions, so a duplicated hit cannot stand in for a missing one.
static size_t streamDiff(const std::vector<Hit>& a, const std::vector<Hit>& b)
{
    return unmatched(a, b) + unmatched(b, a);
}
} // namespace p89

static void runP89DotOrder(bool isTGS)
{
    using namespace p89;
    const char* lbl = isTGS ? "TGS" : "PGS";
    PxgSlangComputeDevice* dev = testDevice();

    Rig R;
    const bool rigOk = buildRig(R, isTGS);
    PXG_CHECK_MSG(rigOk, "%s: the P9.89 rig could not plant its knife edges", lbl);
    if (!rigOk) return;

    // ---- the fixture's own strength: every site ALONE in the old order must
    //      change the compared stream, for BOTH kernels ----
    const std::vector<Hit> want = predict(R, 0u, false, isTGS);
    const std::vector<Hit> wantD = predict(R, 0u, true, isTGS);
    for (uint32_t b = 0; b < NB_SITES; ++b)
    {
        const size_t d = streamDiff(predict(R, 1u << b, false, isTGS), want);
        PXG_CHECK_MSG(d > 0u, "%s: site %s alone in the old order changes NOTHING in the fluid stream -- "
                      "that site is not under test", lbl, SITE_NAME[b]);
        std::printf("    [p89 %s] %-28s old order alone changes %zu of %zu fluid contacts\n",
                    lbl, SITE_NAME[b], d, want.size());
    }
    const size_t allD = streamDiff(predict(R, S_ALL, true, isTGS), wantD);
    PXG_CHECK_MSG(allD > 0u, "%s: the old port's order changes nothing in the diffuse stream", lbl);
    const size_t allF = streamDiff(predict(R, S_ALL, false, isTGS), want);
    std::printf("    [p89 %s] the whole pre-P9.89 order changes %zu of %zu fluid and %zu of %zu diffuse contacts\n",
                lbl, allF, want.size(), allD, wantD.size());

    // ---- device data ----
    std::vector<PxgBufferHandle> owned;
    auto mk = [&](auto vec) { DevBuf b = makeDev(dev, vec); owned.push_back(b.h); return b; };

    DevBuf bCur[NB_SYS], bPred[NB_SYS], bInfos[NB_SYS], bForces[NB_SYS], bNodes[NB_SYS], bCounts[NB_SYS],
        bNum[NB_SYS];
    const uint32_t MAXC = 12u;   // PxgParticleContactInfo::MaxStaticContactsPerParticle
    for (uint32_t s = 0; s < NB_SYS; ++s)
    {
        std::vector<F4> c(NB_PART), p(NB_PART);
        for (uint32_t i = 0; i < NB_PART; ++i)
        {
            c[i] = F4{ R.cur[s][i].x, R.cur[s][i].y, R.cur[s][i].z, 1.0f };
            p[i] = F4{ R.pred[s][i].x, R.pred[s][i].y, R.pred[s][i].z, 1.0f };
        }
        bCur[s] = mk(c);
        bPred[s] = mk(p);
        bInfos[s] = mk(std::vector<F4>(size_t(NB_PART) * MAXC, F4{ -8.25e3f, -8.25e3f, -8.25e3f, -8.25e3f }));
        bForces[s] = mk(std::vector<float>(size_t(NB_PART) * MAXC, -8.25e3f));
        bNodes[s] = mk(std::vector<uint64_t>(size_t(NB_PART) * MAXC, ~0ull));
        bCounts[s] = mk(std::vector<uint32_t>(NB_PART, 0u));
        bNum[s] = mk(std::vector<int32_t>{ int32_t(NB_PART), 0 });
    }
    DevBuf bCellStart = mk(std::vector<uint32_t>{ 0u });
    DevBuf bCellEnd = mk(std::vector<uint32_t>{ NB_PART });
    std::vector<uint32_t> s2u(NB_PART);
    for (uint32_t i = 0; i < NB_PART; ++i) s2u[i] = 1000u + i;
    DevBuf bS2U = mk(s2u);
    DevBuf bRunsum = mk(std::vector<uint32_t>{ 0u });
    std::vector<PxgParticleSimBuffer> simBuffers(1);
    std::memset(simBuffers.data(), 0, sizeof(PxgParticleSimBuffer));
    DevBuf bSimBuf = mk(simBuffers);

    std::vector<PxgParticleSystem> descs(NB_SYS);
    for (uint32_t s = 0; s < NB_SYS; ++s)
    {
        PxgParticleSystem ps = makeZeroPs();
        ps.mCommonData.mGridCellWidth = 1.0f;
        ps.mCommonData.mGridSizeX = 1u;   // 1x1x1: every work item reads cell 0
        ps.mCommonData.mGridSizeY = 1u;
        ps.mCommonData.mGridSizeZ = 1u;
        ps.mCommonData.mNumParticles = NB_PART;
        ps.mCommonData.mMaxDiffuseParticles = NB_PART;
        ps.mCommonData.mNumParticleBuffers = 1u;
        ps.mData.mFlags = (s == 1u) ? 8u : 0u;          // eENABLE_SPECULATIVE_CCD on system 1
        ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bCur[s].addr));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPred[s].addr));
        ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
        ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
        ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bS2U.addr));
        ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
        ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSimBuf.addr));
        // the diffuse side reads the SAME particles
        ps.mDiffuseCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
        ps.mDiffuseCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
        ps.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(bCur[s].addr));
        ps.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(bPred[s].addr));
        ps.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(bNum[s].addr));
        ps.mDiffuseOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(bInfos[s].addr));
        ps.mDiffuseOneWayForces = reinterpret_cast<float*>(size_t(bForces[s].addr));
        ps.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(bNodes[s].addr));
        ps.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(bCounts[s].addr));
        descs[s] = ps;
    }
    DevBuf bDesc = mk(descs);

    // shapes / caches: 0..1 = the particle systems, 2.. = the rigids.  Pair k =
    // (system k / NB_RIGID, rigid k % NB_RIGID), one work item each.
    const uint32_t NB_SHAPES = NB_SYS + NB_RIGID, NB_PAIRS = NB_SYS * NB_RIGID;
    std::vector<PxgShape> shapes(NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    std::vector<PxsCachedTransform> tcache(NB_SHAPES);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    std::vector<float> cdist(NB_SHAPES, 0.0f);
    std::vector<uint64_t> remap(NB_SHAPES, 0ull);
    for (uint32_t s = 0; s < NB_SYS; ++s)
    {
        shapes[s].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
        shapes[s].particleOrSoftbodyId = s;
        cdist[s] = SYS_CD[s];
    }
    for (uint32_t j = 0; j < NB_RIGID; ++j)
    {
        const Shape& sh = R.shape[j];
        shapes[NB_SYS + j].type = sh.type;
        shapes[NB_SYS + j].scale.scale = PxVec3(sh.scale.x, sh.scale.y, sh.scale.z);
        tcache[NB_SYS + j].transform = PxTransform(PxVec3(sh.p.x, sh.p.y, sh.p.z),
                                                   PxQuat(sh.q[0], sh.q[1], sh.q[2], sh.q[3]));
        cdist[NB_SYS + j] = sh.cd;
        remap[NB_SYS + j] = uint64_t(NODE_BASE + j);
    }
    std::vector<PxgContactManagerInput> cin(NB_PAIRS);
    std::vector<uint32_t> startIdx(NB_PAIRS);
    for (uint32_t k = 0; k < NB_PAIRS; ++k)
    {
        const uint32_t s = k / NB_RIGID, j = k % NB_RIGID;
        // odd pairs put the rigid in slot 0, so LoadShapePair's swap runs too
        const bool swap = (k & 1u) != 0u;
        cin[k].shapeRef0 = swap ? NB_SYS + j : s;
        cin[k].transformCacheRef0 = swap ? NB_SYS + j : s;
        cin[k].shapeRef1 = swap ? s : NB_SYS + j;
        cin[k].transformCacheRef1 = swap ? s : NB_SYS + j;
        startIdx[k] = k;
    }
    DevBuf bShapes = mk(shapes), bTc = mk(tcache), bCd = mk(cdist), bRemap = mk(remap), bCin = mk(cin);
    DevBuf bStart = mk(startIdx);
    DevBuf bTotal = mk(std::vector<uint32_t>{ NB_PAIRS });
    DevBuf bRd = mk(std::vector<float>(R.rest, R.rest + NB_PAIRS));
    DevBuf bMat = mk(std::vector<uint32_t>(4u, 0u));
    std::vector<float> boundsF(size_t(NB_SHAPES) * 6u);
    for (uint32_t sh = 0; sh < NB_SHAPES; ++sh)
        for (int k = 0; k < 3; ++k) { boundsF[sh * 6u + k] = -20.0f; boundsF[sh * 6u + 3 + k] = 20.0f; }
    DevBuf bBounds = mk(boundsF);

    const uint32_t MAX_CONTACTS = 4096u;
    ContactStreams bContacts = makeContacts(dev, MAX_CONTACTS);
    DevBuf bCount = mk(std::vector<uint32_t>{ 0u });
    DevBuf bSbP = mk(std::vector<uint64_t>(MAX_CONTACTS, ~0ull));
    DevBuf bTbP = mk(std::vector<uint32_t>(MAX_CONTACTS, ~0u));
    DevBuf bIbP = mk(std::vector<uint32_t>(MAX_CONTACTS, ~0u));
    DevBuf bBr = mk(std::vector<uint64_t>(MAX_CONTACTS, ~0ull));
    DevBuf bTbr = mk(std::vector<uint32_t>(MAX_CONTACTS, ~0u));
    DevBuf bIbr = mk(std::vector<uint32_t>(MAX_CONTACTS, ~0u));

    PsCollideBlob blob;
    std::memset(&blob, 0, sizeof(blob));
    blob.isTGSWord = isTGS ? 1u : 0u;
    blob.numTests = NB_PAIRS;
    blob.toleranceLength = 1.0f;
    blob.cmInputs = bCin.addr;             blob.shapes = bShapes.addr;
    blob.transformCache = bTc.addr;        blob.bounds = bBounds.addr;
    blob.contactDistance = bCd.addr;       blob.restDistances = bRd.addr;
    blob.materials = bMat.addr;            blob.startIndices = bStart.addr;
    blob.totalNumPairs = bTotal.addr;      blob.particleSystems = bDesc.addr;
    blob.shapeToRigidRemap = bRemap.addr;
    blob.w_particleContactInfos = bContacts.addr; blob.w_particleContactNormalPens = bContacts.addr2;
    blob.w_numTotalContacts = bCount.addr;
    blob.w_contactSortedByParticle = bSbP.addr;
    blob.w_tempContactByParticle = bTbP.addr;
    blob.w_contactIndexSortedByParticle = bIbP.addr;
    blob.w_contactByRigid = bBr.addr;
    blob.w_tempContactByRigid = bTbr.addr;
    blob.w_contactIndexSortedByRigid = bIbr.addr;
    blob.w_maxContacts = MAX_CONTACTS;

    // ---- the fluid kernel: one global stream, compared as a multiset ----
    {
        Kernel k("psContactGenBda.spv", "ps_primitivesCollisionLaunch", {}, sizeof(PsCollideBlob));
        k.dispatch(1024u, {}, blob, 1u, 1u);
        dev->flush();
        const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
        PXG_CHECK_MSG(gCount == uint32_t(want.size()), "%s: fluid kernel emitted %u contacts, the CUDA-order "
                      "mirror predicts %zu", lbl, gCount, want.size());
        const uint32_t n = std::min(gCount, MAX_CONTACTS);
        const auto gC = downloadContacts(dev, bContacts, n);
        std::vector<Hit> got;
        for (uint32_t a = 0; a < n; ++a)
        {
            const uint32_t sys = PxgGetParticleSystemId(gC[a].particleId), part = PxgGetParticleIndex(gC[a].particleId);
            const uint32_t rig = uint32_t(gC[a].rigidId - NODE_BASE);
            got.push_back(Hit{ sys, part, rig, gC[a].normal_pen });
        }
        const size_t mm = streamDiff(got, want);
        const size_t asOld = (allF > 0u) ? streamDiff(got, predict(R, S_ALL, false, isTGS)) : 1u;
        PXG_CHECK_MSG(mm == 0u, "%s: %zu fluid contacts differ bitwise from the CUDA-order mirror "
                      "(%zu from the pre-P9.89 order twin)", lbl, mm, asOld);
    }

    // ---- the diffuse kernel: per-particle slots, compared per particle ----
    {
        Kernel k("psContactGenBda.spv", "ps_primitivesDiffuseCollisionLaunch", {}, sizeof(PsCollideBlob));
        k.dispatch(1024u, {}, blob, 1u, 1u);
        dev->flush();
        size_t mm = 0, badCount = 0;
        for (uint32_t s = 0; s < NB_SYS; ++s)
        {
            const auto counts = downloadBuffer<uint32_t>(dev, bCounts[s].h, NB_PART);
            const auto infos = downloadBuffer<F4>(dev, bInfos[s].h, size_t(NB_PART) * MAXC);
            const auto nodes = downloadBuffer<uint64_t>(dev, bNodes[s].h, size_t(NB_PART) * MAXC);
            for (uint32_t i = 0; i < NB_PART; ++i)
            {
                std::vector<Hit> exp, got;
                for (const Hit& h : wantD) if (h.sys == s && h.particle == i) exp.push_back(h);
                if (counts[i] != uint32_t(exp.size())) { ++badCount; continue; }
                for (uint32_t slot = 0; slot < std::min(counts[i], MAXC); ++slot)
                {
                    const size_t at = size_t(i) + size_t(slot) * NB_PART;
                    got.push_back(Hit{ s, i, uint32_t(nodes[at] - NODE_BASE), infos[at] });
                }
                if (exp.size() <= MAXC) mm += streamDiff(got, exp);
            }
        }
        PXG_CHECK_MSG(badCount == 0u, "%s: %zu diffuse per-particle counts differ from the mirror", lbl, badCount);
        PXG_CHECK_MSG(mm == 0u, "%s: %zu diffuse contacts differ bitwise from the CUDA-order mirror", lbl, mm);
    }

    for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
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
PXG_TEST(psContactGen_capsuleNaNPGS) { runCapsuleNaNContact(false); }
PXG_TEST(psContactGen_capsuleNaNTGS) { runCapsuleNaNContact(true); }
PXG_TEST(psContactGen_p89DotOrderPGS) { runP89DotOrder(false); }
PXG_TEST(psContactGen_p89DotOrderTGS) { runP89DotOrder(true); }


// The work-item COMPACTION (cudaParticleSystem.cu:1189-1320): each round a block
// appends the work items whose cell holds particles to a shared queue and processes
// it when full; items that do not fit go to the front of the next queue (overflow),
// and a partly filled queue is processed after the last round.  A 4x1x1 grid with
// one empty cell makes three of four work items non-empty, so the queue fills
// unevenly and overflows from the second round on; over 3 x PS_BLOCK x PS_GRID work
// items every block runs four rounds.  Every non-empty work item must yield exactly
// its cell's contact: the per-particle contact counts pin that no queued item is
// lost or processed twice.
static void runCompaction(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const char* lbl = isTGS ? "TGS" : "PGS";
    const uint32_t NB_ITEMS = 3u * 256u * 1024u + 777u;
    const uint32_t EMPTY = 0xffffffffu;

    const float s = std::sin(0.78539816339744830961f);
    const float c = std::cos(0.78539816339744830961f);
    const float planeQ[4] = { 0.0f, 0.0f, s, c };   // PxPlane(1,0,0,0) normal +X onto +Y

    // particles 0..2 just below the y = 0 plane, one per cell 0..2; cell 3 empty
    std::vector<F4> pos = { { 0.25f, -0.1f, 0.0f, 1.0f }, { 1.25f, -0.05f, 0.0f, 1.0f }, { 2.25f, -0.2f, 0.0f, 1.0f } };
    DevBuf bOrigin = makeDev(dev, pos);
    DevBuf bPred = makeDev(dev, pos);
    DevBuf bCellStart = makeDev(dev, std::vector<uint32_t>{ 0u, 1u, 2u, EMPTY });
    DevBuf bCellEnd = makeDev(dev, std::vector<uint32_t>{ 1u, 2u, 3u, 0u });
    DevBuf bS2U = makeDev(dev, std::vector<uint32_t>{ 0u, 1u, 2u });
    DevBuf bRunsum = makeDev(dev, std::vector<uint32_t>{ 0u });
    std::vector<PxgParticleSimBuffer> simBuffers(1);
    std::memset(simBuffers.data(), 0, sizeof(PxgParticleSimBuffer));
    DevBuf bSimBuf = makeDev(dev, simBuffers);

    PxgParticleSystem ps = makeZeroPs();
    ps.mCommonData.mGridCellWidth = 1.0f;
    ps.mCommonData.mGridSizeX = 4u;
    ps.mCommonData.mGridSizeY = 1u;
    ps.mCommonData.mGridSizeZ = 1u;
    ps.mCommonData.mNumParticles = 3u;
    ps.mCommonData.mNumParticleBuffers = 1u;
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bOrigin.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPred.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bS2U.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(bRunsum.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(bSimBuf.addr));
    DevBuf bDesc = makeDev(dev, std::vector<PxgParticleSystem>{ ps });

    std::vector<PxgContactManagerInput> cin(1);
    cin[0].shapeRef0 = 0u; cin[0].transformCacheRef0 = 0u;
    cin[0].shapeRef1 = 1u; cin[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cin);
    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[1].type = PxU32(PxGeometryType::ePLANE);
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(0.0f, 0.0f, 0.0f), PxQuat(planeQ[0], planeQ[1], planeQ[2], planeQ[3]));
    DevBuf bTc = makeDev(dev, tcache);
    // [-20, 20]^3 for both: the overlap spans more than four cells in x, so the
    // wrapped range is the whole 4-wide grid and work item k lands in cell
    // (calcGridPos(-20.3).x + k % 4) & 3 = (k + 3) % 4.
    std::vector<float> boundsF(12);
    for (int k = 0; k < 3; ++k) { boundsF[k] = boundsF[6 + k] = -20.0f; boundsF[3 + k] = boundsF[9 + k] = 20.0f; }
    DevBuf bBounds = makeDev(dev, boundsF);
    DevBuf bCd = makeDev(dev, std::vector<float>{ 0.3f, 0.0f });
    DevBuf bRd = makeDev(dev, std::vector<float>{ 0.0f });
    DevBuf bMat = makeDev(dev, std::vector<uint32_t>{ 0u });
    DevBuf bStart = makeDev(dev, std::vector<uint32_t>{ 0u });
    DevBuf bTotal = makeDev(dev, std::vector<uint32_t>{ NB_ITEMS });
    DevBuf bRemap = makeDev(dev, std::vector<uint64_t>{ 0ull, 7ull });

    // expected: one contact per work item whose cell holds a particle
    uint32_t want[3] = { 0u, 0u, 0u };
    for (uint32_t k = 0; k < NB_ITEMS; ++k)
    {
        const uint32_t cell = (k + 3u) % 4u;
        if (cell < 3u)
            ++want[cell];
    }
    const uint32_t wantTotal = want[0] + want[1] + want[2];

    const uint32_t MAX_CONTACTS = wantTotal + 64u;
    ContactStreams bContacts = makeContacts(dev, MAX_CONTACTS);
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
    blob.numTests = 1u;
    blob.toleranceLength = 1.0f;
    blob.cmInputs = bCin.addr;             blob.shapes = bShapes.addr;
    blob.transformCache = bTc.addr;        blob.bounds = bBounds.addr;
    blob.contactDistance = bCd.addr;       blob.restDistances = bRd.addr;
    blob.materials = bMat.addr;            blob.startIndices = bStart.addr;
    blob.totalNumPairs = bTotal.addr;      blob.particleSystems = bDesc.addr;
    blob.shapeToRigidRemap = bRemap.addr;
    blob.w_particleContactInfos = bContacts.addr; blob.w_particleContactNormalPens = bContacts.addr2;
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

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == wantTotal, "%s: %u contacts from %u work items, expected %u", lbl, gCount, NB_ITEMS,
                  wantTotal);
    const uint32_t n = std::min(gCount, MAX_CONTACTS);
    const auto gC = downloadContacts(dev, bContacts, n);
    uint32_t got[3] = { 0u, 0u, 0u }, bad = 0u;
    F4 first[3] = {};
    bool seen[3] = { false, false, false };
    for (uint32_t a = 0; a < n; ++a)
    {
        const uint64_t part = gC[a].particleId & 0xffffffffull;
        if (part >= 3u || gC[a].rigidId != 7ull) { ++bad; continue; }
        ++got[part];
        if (!seen[part]) { first[part] = gC[a].normal_pen; seen[part] = true; }
        else if (std::memcmp(&first[part], &gC[a].normal_pen, sizeof(F4)) != 0) ++bad;
    }
    PXG_CHECK_MSG(bad == 0u, "%s: %u contacts with a wrong identity or a differing normal/penetration", lbl, bad);
    for (uint32_t p = 0; p < 3u; ++p)
        PXG_CHECK_MSG(got[p] == want[p], "%s: particle %u got %u contacts, expected %u (one per work item of "
                      "its cell)", lbl, p, got[p], want[p]);
    for (const DevBuf* b : std::initializer_list<const DevBuf*>{ &bOrigin, &bPred, &bCellStart, &bCellEnd, &bS2U, &bRunsum, &bSimBuf, &bDesc, &bCin,
                             &bShapes, &bTc, &bBounds, &bCd, &bRd, &bMat, &bStart, &bTotal, &bRemap,
                             static_cast<const DevBuf*>(&bContacts), &bCount, &bSbP, &bTbP, &bIbP, &bBr, &bTbr, &bIbr })
        dev->destroyBuffer(b->h);
    dev->destroyBuffer(bContacts.h2);
}

PXG_TEST(psContactGen_compactionPGS) { runCompaction(false); }
PXG_TEST(psContactGen_compactionTGS) { runCompaction(true); }

} // namespace
