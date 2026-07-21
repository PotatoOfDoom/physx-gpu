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

#include "pxgTestFramework.h"

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

#include <cmath>
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

static void runPlaneContact(bool isTGS)
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
    const float planeP[3] = { 0.0f, 0.0f, 0.0f };

    const float particlePos[3] = { 0.0f, -0.1f, 0.0f };
    const float contactOffset = 0.3f;
    const float restDistance = 0.0f;
    const uint32_t rigidNodeIndex = 7u;

    // ---- CPU transcription of the ePLANE branch (bit-exact ground truth) ----
    float refN[3];
    const float planeCanonicalNormal[3] = { 1.0f, 0.0f, 0.0f };
    cpuQuatRotate(planeQ, planeCanonicalNormal, refN);
    const float pdd = -(refN[0] * planeP[0] + refN[1] * planeP[1] + refN[2] * planeP[2]);
    // no CCD -> cVolumePos == particlePos, cVolumeRadius == contactOffset
    const float refDist = refN[0] * particlePos[0] + refN[1] * particlePos[1] + refN[2] * particlePos[2] + pdd;
    const float refPen = refDist - restDistance;
    PXG_CHECK_MSG(refDist <= contactOffset, "reference: particle must be within the contact offset");

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

    // human-readable sanity: normal ~ (0,1,0), penetration ~ -0.1
    PXG_CHECK_MSG(almostEqual(ct.normal_pen.x, 0.0f, 1e-5f, 1e-6f) &&
                  almostEqual(ct.normal_pen.y, 1.0f, 1e-5f, 1e-6f) &&
                  almostEqual(ct.normal_pen.z, 0.0f, 1e-5f, 1e-6f),
                  "%s: contact normal ~ (0,1,0) (got %g %g %g)", isTGS ? "TGS" : "PGS",
                  ct.normal_pen.x, ct.normal_pen.y, ct.normal_pen.z);
    PXG_CHECK_MSG(almostEqual(ct.normal_pen.w, -0.1f, 1e-5f, 1e-6f),
                  "%s: penetration ~ -0.1 (got %g)", isTGS ? "TGS" : "PGS", ct.normal_pen.w);

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

PXG_TEST(psContactGen_planeContactPGS) { runPlaneContact(false); }
PXG_TEST(psContactGen_planeContactTGS) { runPlaneContact(true); }

} // namespace
