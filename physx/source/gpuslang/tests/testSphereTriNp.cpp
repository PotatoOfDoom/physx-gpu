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

// testSphereTriNp.cpp -- P9.66c.
//
// THE ONE THING NEITHER THE SCENES NOR THE PHASE-1 FIXTURE CAN CHECK.
//
// `sphereTriNpBda.slang` (P9.66b) put sphereTrimeshNarrowphase on the engine ABI,
// and PXG_SCENE=spheremesh / capsulemesh proved the chain runs: both rest at
// y = 0.5000, the number the meshfloor box control gives over the same floor.
// `testNpCvxMeshNp.cpp` proves the ARITHMETIC of both cores, including this exact
// asymmetry (its :573 expects `+ remapCpu`, its :638 `+ triangleIdx`).
//
// Neither covers the BDA shader's OWN copy of that choice.  The two Slang files
// are separate transcriptions of one CUDA kernel, and the engine one is the one
// that ships:
//
//     convexMesh.cu:392  (sphere)   ... + remapCpuTriangleIdx;   five-arg readTriangleMesh
//     convexMesh.cu:613  (capsule)  ... + triangleIdx;           four-arg overload
//
// Getting it backwards is SILENT.  `getTriangleIndex()` (convexNpCommon.h:90-93)
// masks to 27 bits, so both candidates are small non-negative integers: nothing
// faults, no bounds check fires, and contact positions, normals and separations
// are all identical.  Only the reported face identity is wrong.
//
// AND THE SCENE PROVABLY CANNOT SEE IT.  `addMeshFloor` cooks a two-triangle quad,
// whose face remap is the identity -- `remapCpuTriangleIdx == triangleIdx` -- so
// the two spellings coincide and both scenes pass either way.  That is why this
// fixture exists and why it builds its own mesh: the remap here is
// {700, 701, 702, 703}, which shares no value with any triangle index, so the two
// expected numbers cannot be confused for one another.
//
// The shape of the test is the whole point: ONE mesh, TWO primitives, the SAME
// field, TWO different expected values.  A port that reuses one arm's index
// expression for both fails on exactly one of the two rows.

#include "pxgTestFramework.h"

#include "geometry/PxGeometry.h"
#include "GuBV32.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct V4 { float x, y, z, w; };

static const uint32_t BV32_STRIDE = uint32_t(sizeof(physx::Gu::BV32DataPacked)); // 1168
static_assert(sizeof(physx::Gu::BV32DataPacked) == 1168,
              "BV32DataPacked changed size; sphereTriNpBda.slang carries this number too");

// P9.57: from the ENGINE ENUM, never a literal.
static const uint32_t GEOM_SPHERE = uint32_t(physx::PxGeometryType::eSPHERE);
static const uint32_t GEOM_CAPSULE = uint32_t(physx::PxGeometryType::eCAPSULE);
static const uint32_t GEOM_TRIANGLEMESH = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);

// PxgShape, 48 B
struct GpuShape
{
    float sx, sy, sz;
    float rx, ry, rz, rw;
    uint32_t materialIndex;
    uint32_t hullLo, hullHi;
    uint32_t type;
    uint32_t particleOrSoftbodyId;
};
static_assert(sizeof(GpuShape) == 48, "PxgShape mirror");

struct CmInput { uint32_t shapeRef0, shapeRef1, tcRef0, tcRef1; };

// PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16 (+4 pad).
struct CachedXf { float qx, qy, qz, qw, px, py, pz, pad; };
static_assert(sizeof(CachedXf) == 32, "PxsCachedTransform mirror");

// The 21-argument blob of convexMesh.cu:763-788 -- must match SphereTriNpParams
// in the shader field for field.
struct StnPush
{
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes;
    uint64_t sphereTrimeshPair;
    uint64_t niPtr, contactsPtr, maxDepthPtr, intermPtr, orderedPtr, secondPassPtr;
    uint64_t stackPtr, nbPairsGlobal, nbPaddedPairsGlobal, nbSecondPassPairs;
    uint32_t stackSizeBytes, pad124;
    uint64_t tempContacts, pTempContactIndex, maxTempMemRequirement, midphasePairsNeeded;
    uint32_t nbContactManagers, pad164;
};
static_assert(sizeof(StnPush) == 168, "the shared kCvxTriCore blob is 168 bytes");

// ---------------------------------------------------------------------------
// the cooked mesh, by hand
// ---------------------------------------------------------------------------
struct Mesh
{
    std::vector<uint8_t> blob;
    std::vector<V4> verts;
    std::vector<uint32_t> idx;      // 4 per triangle (uint4, .w unused)
    std::vector<uint32_t> adj;      // 4 per triangle
    std::vector<uint32_t> remap;    // 1 per triangle
    uint32_t numTris = 0;
};

static void buildMesh(Mesh& m, uint32_t nbNodes)
{
    const uint32_t numVerts = uint32_t(m.verts.size());
    m.numTris = uint32_t(m.idx.size() / 4);

    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) m.blob.push_back(uint8_t(v >> (8 * i))); };
    auto f32 = [&](float v) { uint32_t u; std::memcpy(&u, &v, 4); u32(u); };

    u32(numVerts); u32(m.numTris); u32(0xA5A5A5A5u); u32(nbNodes);   // .z is NOT a node count
    // A recognisable tree pattern: a mis-sized walk lands IN it and reports
    // nonsense rather than zeros that could pass for data.
    for (uint32_t i = 0; i < nbNodes * BV32_STRIDE / 4; ++i) u32(0xDEAD0000u | (i & 0xffffu));
    for (const V4& v : m.verts) { f32(v.x); f32(v.y); f32(v.z); f32(v.w); }
    for (uint32_t v : m.idx) u32(v);
    for (uint32_t v : m.adj) u32(v);
    for (uint32_t v : m.remap) u32(v);
    for (int i = 0; i < 64; ++i) u32(0);   // trailing sections this kernel never reads
}

// maCompute (pxgSlangMidphaseAlloc.slang:90-105), mirrored so the fixture knows
// where the shader put each section.  The allocation ORDER is not the order of
// the CUDA function's out-parameters, and `ordered` alone uses the padded count.
struct StackOffsets { uint64_t pairs, ni, contacts, ordered, maxDepth, interm, secondPass, total; };
static StackOffsets stackLayout(uint32_t nbPairs, uint32_t nbPaddedPairs)
{
    const uint64_t n = uint64_t((nbPairs + 3u) & ~3u);
    const uint64_t nPadded = uint64_t(nbPaddedPairs);
    StackOffsets s{};
    uint64_t p = 0;
    s.pairs = p;      p += n * 16;
    s.ni = p;         p += n * 16;
    s.contacts = p;   p += n * 4;
    s.ordered = p;    p += nPadded * 4;
    s.maxDepth = p;   p += n * 4;
    s.interm = p;     p += n * 4;
    s.secondPass = p; p += n * 4;
    s.total = p;
    return s;
}

// ConvexTriNormalAndIndex::getTriangleIndex (convexNpCommon.h:90-93): the low
// 27 bits, below the count field and the deferred bit.
static uint32_t triangleIndexOf(uint32_t niIndex) { return niIndex & ((1u << 27) - 1u); }
static uint32_t nbContactsOf(uint32_t niIndex) { return (niIndex >> 27) & 0xfu; }

PXG_TEST(sphereTriNp_faceRemapAsymmetry)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- one mesh, one triangle, a remap that shares no value with an index --
    // The triangle is wound so its normal is +y: cross(v1-v0, v2-v0) =
    // cross((0,0,4),(4,0,0)) = (0,16,0).  Both primitives sit above it, which is
    // the side the one-sided gate (`dist0 >= 0`) admits.
    Mesh mesh;
    mesh.verts = { { 0, 0, 0, 0 }, { 0, 0, 4, 0 }, { 4, 0, 0, 0 } };
    mesh.idx = { 0, 1, 2, 0 };
    // BOUNDARY on every edge: `isEdgeNonconvex` is false for the sentinel, so the
    // primitive's convexity verdict is true and a face contact is generated.
    mesh.adj = { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0 };
    mesh.remap = { 700 };
    buildMesh(mesh, 2);

    PxgBufferHandle bMesh = createBufferFrom(dev, mesh.blob);
    const uint64_t addrMesh = dev->getBufferDeviceAddress(bMesh);

    // ---- shapes: 0 sphere, 1 mesh, 2 capsule --------------------------------
    std::vector<GpuShape> shapes(3);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    // PxSphereGeometry packs the radius into scale.x (PxgNarrowphaseCore.cpp:7508).
    shapes[0].sx = 0.5f; shapes[0].sy = 0.5f; shapes[0].sz = 0.5f;
    shapes[0].rw = 1.0f; shapes[0].type = GEOM_SPHERE;
    // the mesh: identity scale, so no winding flip and no vertex2Shape movement
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f;
    shapes[1].rw = 1.0f; shapes[1].type = GEOM_TRIANGLEMESH;
    shapes[1].hullLo = uint32_t(addrMesh & 0xffffffffu);
    shapes[1].hullHi = uint32_t(addrMesh >> 32);
    // PxCapsuleGeometry packs (halfHeight, radius, radius) (:7515).
    shapes[2].sx = 0.5f; shapes[2].sy = 0.5f; shapes[2].sz = 0.5f;
    shapes[2].rw = 1.0f; shapes[2].type = GEOM_CAPSULE;

    // ---- transforms: identity rotations, the primitives 0.4 above the face ---
    std::vector<CachedXf> xf(3);
    std::memset(xf.data(), 0, xf.size() * sizeof(CachedXf));
    xf[0].qw = 1.0f; xf[0].px = 1.0f; xf[0].py = 0.4f; xf[0].pz = 1.0f;   // sphere
    xf[1].qw = 1.0f;                                                        // mesh at origin
    xf[2].qw = 1.0f; xf[2].px = 1.0f; xf[2].py = 0.4f; xf[2].pz = 1.0f;   // capsule
    std::vector<float> contactDist(3, 0.0f);

    // ---- two contact managers: (sphere, mesh) and (capsule, mesh) -----------
    // Slot 0 is the primitive in both, so the kernel's mesh-in-slot-0 swap does
    // NOT fire here; the swap has its own coverage in the scenes.
    std::vector<CmInput> cms(2);
    cms[0] = { 0u, 1u, 0u, 1u };
    cms[1] = { 2u, 1u, 2u, 1u };

    // ---- ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40 --------
    // DIFFERENT permutations for the two fields, so a port reading the wrong one
    // lands somewhere visible rather than on the same number.
    std::vector<uint8_t> pairRecs(2 * 56, 0);
    auto setPair = [&](uint32_t cm, uint32_t startIndex, uint32_t rounded) {
        std::memcpy(&pairRecs[cm * 56 + 28], &startIndex, 4);
        std::memcpy(&pairRecs[cm * 56 + 40], &rounded, 4);
    };
    setPair(0, 0u, 0u);
    setPair(1, 1u, 3u);

    // ---- the midphase stack -------------------------------------------------
    const uint32_t nbPairs = 2, nbPaddedPairs = 8;
    const StackOffsets so = stackLayout(nbPairs, nbPaddedPairs);
    std::vector<uint8_t> stack(size_t(so.total) + 256, 0);
    // the pair records the kernel reads: (cmIdx, triangleIdx, testOffset, 0)
    auto setStackPair = [&](uint32_t i, uint32_t cm, uint32_t tri, uint32_t off) {
        uint32_t v[4] = { cm, tri, off, 0u };
        std::memcpy(&stack[size_t(so.pairs) + i * 16], v, 16);
    };
    setStackPair(0, 0u, 0u, 0u);   // sphere  x triangle 0
    setStackPair(1, 1u, 0u, 0u);   // capsule x triangle 0

    PxgBufferHandle bShapes = createBufferFrom(dev, shapes);
    PxgBufferHandle bXf = createBufferFrom(dev, xf);
    PxgBufferHandle bCd = createBufferFrom(dev, contactDist);
    PxgBufferHandle bCms = createBufferFrom(dev, cms);
    PxgBufferHandle bPairRecs = createBufferFrom(dev, pairRecs);
    PxgBufferHandle bStack = createBufferFrom(dev, stack);

    std::vector<uint32_t> counters(4, 0);
    counters[0] = nbPairs;
    counters[1] = nbPaddedPairs;
    PxgBufferHandle bNbPairs = createBufferFrom(dev, counters);
    std::vector<uint32_t> scratch(64, 0);
    PxgBufferHandle bScratch = createBufferFrom(dev, scratch);
    std::vector<uint64_t> outPtrs(8, 0);
    PxgBufferHandle bOutPtrs = createBufferFrom(dev, outPtrs);
    std::vector<float> tempContacts(64, 0.0f);
    PxgBufferHandle bTemp = createBufferFrom(dev, tempContacts);

    const uint64_t aNbPairs = dev->getBufferDeviceAddress(bNbPairs);
    const uint64_t aScratch = dev->getBufferDeviceAddress(bScratch);
    const uint64_t aOutPtrs = dev->getBufferDeviceAddress(bOutPtrs);

    StnPush p{};
    p.cmInputs = dev->getBufferDeviceAddress(bCms);
    p.transformCache = dev->getBufferDeviceAddress(bXf);
    p.contactDistance = dev->getBufferDeviceAddress(bCd);
    p.gpuShapes = dev->getBufferDeviceAddress(bShapes);
    p.sphereTrimeshPair = dev->getBufferDeviceAddress(bPairRecs);
    p.niPtr = aOutPtrs;
    p.contactsPtr = aOutPtrs + 8;
    p.maxDepthPtr = aOutPtrs + 16;
    p.intermPtr = aOutPtrs + 24;
    p.orderedPtr = aOutPtrs + 32;
    p.secondPassPtr = aOutPtrs + 40;
    p.stackPtr = dev->getBufferDeviceAddress(bStack);
    p.nbPairsGlobal = aNbPairs;
    p.nbPaddedPairsGlobal = aNbPairs + 4;
    p.nbSecondPassPairs = aScratch;
    p.stackSizeBytes = uint32_t(stack.size());
    p.tempContacts = dev->getBufferDeviceAddress(bTemp);
    p.pTempContactIndex = aScratch + 4;
    p.maxTempMemRequirement = aScratch + 8;
    p.midphasePairsNeeded = aScratch + 12;
    p.nbContactManagers = 2u;

    // ONE workgroup of one warp: the shader is [numthreads(32,1,1)] and two pairs
    // fit in a single pass, so the grid-stride loop runs exactly once.
    Kernel k("sphereTriNpBda.spv", "sphereTrimeshNarrowphase", {}, sizeof(StnPush));
    k.dispatch(1u, {}, p, 1u, 1u);
    dev->flush();

    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bStack, stack.size());

    auto niIndexAt = [&](uint32_t pairOffset) {
        uint32_t v; std::memcpy(&v, &got[size_t(so.ni) + pairOffset * 16 + 12], 4); return v;
    };

    // pairOffset = startIndex + testOffset, from the ConvexMeshPair record.
    const uint32_t sphereIdx = niIndexAt(0);   // cm 0: startIndex 0 + offset 0
    const uint32_t capsuleIdx = niIndexAt(1);  // cm 1: startIndex 1 + offset 0

    // Both must actually have generated a contact -- otherwise the index field is
    // trivially the bare triangle index for both arms and the test proves nothing.
    PXG_CHECK_MSG(nbContactsOf(sphereIdx) != 0u,
                  "sphere generated no contact (niIndex=0x%08x); the remap comparison "
                  "below would be vacuous", sphereIdx);
    PXG_CHECK_MSG(nbContactsOf(capsuleIdx) != 0u,
                  "capsule generated no contact (niIndex=0x%08x); the raw-index "
                  "comparison below would be vacuous", capsuleIdx);

    // THE ASSERTION THIS FILE EXISTS FOR.  Same field, two arms, two values.
    PXG_CHECK_MSG(triangleIndexOf(sphereIdx) == mesh.remap[0],
                  "SPHERE arm must pack the REMAPPED cpu triangle index "
                  "(convexMesh.cu:392): expected %u, got %u (niIndex=0x%08x)",
                  mesh.remap[0], triangleIndexOf(sphereIdx), sphereIdx);
    PXG_CHECK_MSG(triangleIndexOf(capsuleIdx) == 0u,
                  "CAPSULE arm must pack the RAW gpu triangle index "
                  "(convexMesh.cu:613): expected 0, got %u (niIndex=0x%08x)",
                  triangleIndexOf(capsuleIdx), capsuleIdx);

    for (PxgBufferHandle h : { bMesh, bShapes, bXf, bCd, bCms, bPairRecs, bStack,
                               bNbPairs, bScratch, bOutPtrs, bTemp })
        dev->destroyBuffer(h);
}

} // namespace
