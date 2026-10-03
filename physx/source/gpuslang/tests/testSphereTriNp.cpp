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
//
// P9.86c: a THIRD pair (the capsule, rotated 180 deg about y, lying IN the plane
// of a second triangle) pins the capsule arm's `separation = PxMin(separation,
// pen)` (convexMesh.cu:575) on a signed-zero tie.  Its endpoint separations are
// -0 then +0; PxMin is selectMin (a < b ? a : b) and keeps the SECOND, so the
// stored maxDepth (:603) is +0.  GLSL FMin on llvmpipe gives -0 there.

#include "pxgTestFramework.h"
#include "pxgNpRefMath.h"
#include "pxgCapsuleTriRef.h"

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

    // ---- one mesh, two triangles, a remap that shares no value with an index --
    // The triangle is wound so its normal is +y: cross(v1-v0, v2-v0) =
    // cross((0,0,4),(4,0,0)) = (0,16,0).  Both primitives sit above it, which is
    // the side the one-sided gate (`dist0 >= 0`) admits.
    Mesh mesh;
    // Triangle 1 (P9.86c): (0,0,0), (4,0,-4), (-4,0,0).  cross((4,+0,-4),
    // (-4,+0,+0)) = (+0,16,+0), so its normal is exactly (+0,1,+0) and its region
    // reaches x < 0, where the in-plane capsule of pair 2 lies.
    mesh.verts = { { 0, 0, 0, 0 }, { 0, 0, 4, 0 }, { 4, 0, 0, 0 }, { 4, 0, -4, 0 }, { -4, 0, 0, 0 } };
    mesh.idx = { 0, 1, 2, 0, 0, 3, 4, 0 };
    // BOUNDARY on every edge: `isEdgeNonconvex` is false for the sentinel, so the
    // primitive's convexity verdict is true and a face contact is generated.
    mesh.adj = { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0 };
    mesh.remap = { 700, 701 };
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
    std::vector<CachedXf> xf(4);
    std::memset(xf.data(), 0, xf.size() * sizeof(CachedXf));
    xf[0].qw = 1.0f; xf[0].px = 1.0f; xf[0].py = 0.4f; xf[0].pz = 1.0f;   // sphere
    xf[1].qw = 1.0f;                                                        // mesh at origin
    xf[2].qw = 1.0f; xf[2].px = 1.0f; xf[2].py = 0.4f; xf[2].pz = 1.0f;   // capsule
    // P9.86c: capsule at q = (0,1,0,0) (180 deg about y), p = (-0.5,0,0).
    // getBasisVector0 = (-1,+0,+0), so p0 = (-1,+0,+0) and p1 = (+0,+0,+0) = v0:
    // both endpoints lie in triangle 1's plane.  capGenerateContacts:
    // inomp = (+0*1)+(1*-0)+(+0*-0) = +0 -> sep0 = -0;  inomq = -0 -> sep1 = +0.
    xf[3].qy = 1.0f; xf[3].px = -0.5f;
    std::vector<float> contactDist(4, 0.0f);

    // ---- two contact managers: (sphere, mesh) and (capsule, mesh) -----------
    // Slot 0 is the primitive in both, so the kernel's mesh-in-slot-0 swap does
    // NOT fire here; the swap has its own coverage in the scenes.
    std::vector<CmInput> cms(3);
    cms[0] = { 0u, 1u, 0u, 1u };
    cms[1] = { 2u, 1u, 2u, 1u };
    cms[2] = { 2u, 1u, 3u, 1u };   // P9.86c: in-plane capsule x mesh

    // ---- ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40 --------
    // DIFFERENT permutations for the two fields, so a port reading the wrong one
    // lands somewhere visible rather than on the same number.
    std::vector<uint8_t> pairRecs(3 * 56, 0);
    auto setPair = [&](uint32_t cm, uint32_t startIndex, uint32_t rounded) {
        std::memcpy(&pairRecs[cm * 56 + 28], &startIndex, 4);
        std::memcpy(&pairRecs[cm * 56 + 40], &rounded, 4);
    };
    setPair(0, 0u, 0u);
    setPair(1, 1u, 3u);
    setPair(2, 2u, 5u);

    // ---- the midphase stack -------------------------------------------------
    const uint32_t nbPairs = 3, nbPaddedPairs = 8;
    const StackOffsets so = stackLayout(nbPairs, nbPaddedPairs);
    std::vector<uint8_t> stack(size_t(so.total) + 256, 0);
    // the pair records the kernel reads: (cmIdx, triangleIdx, testOffset, 0)
    auto setStackPair = [&](uint32_t i, uint32_t cm, uint32_t tri, uint32_t off) {
        uint32_t v[4] = { cm, tri, off, 0u };
        std::memcpy(&stack[size_t(so.pairs) + i * 16], v, 16);
    };
    setStackPair(0, 0u, 0u, 0u);   // sphere  x triangle 0
    setStackPair(1, 1u, 0u, 0u);   // capsule x triangle 0
    setStackPair(2, 2u, 1u, 0u);   // in-plane capsule x triangle 1 (P9.86c)
    // maxDepth of pair 2 is expected to be +0: a zero-filled stack would pass
    // vacuously, so the slot carries a sentinel the kernel must overwrite.
    const uint32_t kDepthSentinel = 0x7fbadbadu;
    std::memcpy(&stack[size_t(so.maxDepth) + 2 * 4], &kDepthSentinel, 4);

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
    std::vector<float> tempContacts(64, -99.0f);
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
    p.nbContactManagers = 3u;

    // ONE workgroup of one warp: the shader is [numthreads(32,1,1)] and three pairs
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

    // ---- P9.86c: the PxMin signed-zero tie (convexMesh.cu:575, :603) --------
    const uint32_t tieIdx = niIndexAt(2);
    PXG_CHECK_MSG(nbContactsOf(tieIdx) == 2u && triangleIndexOf(tieIdx) == 1u,
                  "in-plane capsule must give 2 contacts on raw triangle 1 "
                  "(niIndex=0x%08x)", tieIdx);
    uint32_t cs = 0xffffffffu;
    std::memcpy(&cs, &got[size_t(so.contacts) + 2 * 4], 4);
    PXG_CHECK_MSG(cs == 3u, "in-plane capsule contactStart: expected 3 (sphere 1 + "
                  "capsule 2), got %u", cs);
    if (cs + 1u < 16u)
    {
        const std::vector<float> tc = downloadBuffer<float>(dev, bTemp, tempContacts.size());
        uint32_t b0, b1;
        std::memcpy(&b0, &tc[4 * cs + 3], 4);
        std::memcpy(&b1, &tc[4 * (cs + 1) + 3], 4);
        int zeroTieSeen = 0;
        if (b0 == 0x80000000u && b1 == 0x00000000u) ++zeroTieSeen;
        PXG_CHECK_MSG(zeroTieSeen == 1,
                      "in-plane capsule must give sep0=-0 (p0=(-1,+0,+0): inomp=+0) and "
                      "sep1=+0 (p1=v0: inomq=-0); got 0x%08x 0x%08x -- the PxMin zero "
                      "tie is not exercised", b0, b1);
        float s0, s1;
        std::memcpy(&s0, &b0, 4);
        std::memcpy(&s1, &b1, 4);
        // PxMin<float> = intrinsics::selectMin (PxMath.h:97-101,
        // PxUnixMathIntrinsics.h:116-119): a < b ? a : b, a tie keeps b.  NOT
        // std::min (keeps a) and NOT std::fmin.
        auto pxSelectMin = [](float a, float b) { return a < b ? a : b; };
        const float expectDepth = pxSelectMin(pxSelectMin(3.4028234663852886e+38f, s0), s1);
        uint32_t gBits, eBits;
        std::memcpy(&gBits, &got[size_t(so.maxDepth) + 2 * 4], 4);
        std::memcpy(&eBits, &expectDepth, 4);
        PXG_CHECK_MSG(gBits == eBits && eBits == 0x00000000u,
                      "capsule maxDepth (convexMesh.cu:575 PxMin, :603 store): expected "
                      "0x%08x, got 0x%08x", eBits, gBits);
    }

    for (PxgBufferHandle h : { bMesh, bShapes, bXf, bCd, bCms, bPairRecs, bStack,
                               bNbPairs, bScratch, bOutPtrs, bTemp })
        dev->destroyBuffer(h);
}

// P9.94d: ctQuatRotate / ctQuatRotateInv (pxgSlangCvxTriSetup) are PxQuat::rotate /
// rotateInv LITERALLY.  This kernel is their busiest importer: the sphere arm's NI
// normal is `sphereQ.rotateInv(meshQ.rotate(-patchNormal))` (convexMesh.cu:387-388),
// and the sphere centre reaches mesh space through meshQ.rotateInv (:262).
//
// 32 spheres over triangle 0 of the faceRemapAsymmetry mesh, each with its own random
// mesh and sphere rotation.  The centre sits 0.4 above the face in mesh space, so every
// pair is a FACE contact and patchNormal is the triangle normal: identity mesh scale,
// v1 - v0 = (0,0,4), v2 - v0 = (4,0,0), cross = (+0,16,+0), normalized EXACTLY to
// (+0,1,+0) (16 * (1/16) is exact), so -patchNormal = (-0,-1,-0) with no rounding of
// its own.  The stored NI normal must equal the literal oracle BITWISE on every pair,
// and the old Rodrigues spelling (npref's twin) must differ on some -- otherwise this
// test cannot tell the formulas apart.
PXG_TEST(sphereTriNp_p94dLiteralRotate)
{
    PxgSlangComputeDevice* dev = testDevice();
    namespace nr = pxgtest::npref;
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };

    uint32_t seed = 0x94d0003u;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float(seed >> 8) * (1.0f / 16777216.0f); };
    auto sym = [&](float m) { return (rnd() * 2.0f - 1.0f) * m; };
    auto unitQ = [&]() {
        nr::V4 q = { sym(1), sym(1), sym(1), sym(1) };
        const float r = 1.0f / std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        return nr::V4{ q.x * r, q.y * r, q.z * r, q.w * r };
    };

    Mesh mesh;
    mesh.verts = std::vector<V4>{ { 0, 0, 0, 0 }, { 0, 0, 4, 0 }, { 4, 0, 0, 0 } };
    mesh.idx = { 0, 1, 2, 0 };
    mesh.adj = { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0 };
    mesh.remap = { 700 };
    buildMesh(mesh, 1);
    PxgBufferHandle bMesh = createBufferFrom(dev, mesh.blob);
    const uint64_t addrMesh = dev->getBufferDeviceAddress(bMesh);

    const uint32_t N = 32;
    std::vector<GpuShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    shapes[0].sx = 0.5f; shapes[0].sy = 0.5f; shapes[0].sz = 0.5f;
    shapes[0].rw = 1.0f; shapes[0].type = GEOM_SPHERE;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f;
    shapes[1].rw = 1.0f; shapes[1].type = GEOM_TRIANGLEMESH;
    shapes[1].hullLo = uint32_t(addrMesh & 0xffffffffu);
    shapes[1].hullHi = uint32_t(addrMesh >> 32);

    std::vector<CachedXf> xf(2 * N);
    std::memset(xf.data(), 0, xf.size() * sizeof(CachedXf));
    std::vector<CmInput> cms(N);
    std::vector<uint8_t> pairRecs(size_t(N) * 56, 0);
    for (uint32_t i = 0; i < N; ++i)
    {
        const nr::V4 mq = unitQ(), sq = unitQ();
        const nr::V3 mp = { sym(3), sym(3), sym(3) };
        // the centre 0.4 above the face, inside it, pushed out through the mesh pose
        const nr::V3 world = nr::add(mp, nr::quatRotate(mq, nr::V3{ 1.0f, 0.4f, 1.0f }));
        CachedXf& xs = xf[2 * i];
        CachedXf& xm = xf[2 * i + 1];
        xs.qx = sq.x; xs.qy = sq.y; xs.qz = sq.z; xs.qw = sq.w; xs.px = world.x; xs.py = world.y; xs.pz = world.z;
        xm.qx = mq.x; xm.qy = mq.y; xm.qz = mq.z; xm.qw = mq.w; xm.px = mp.x; xm.py = mp.y; xm.pz = mp.z;
        cms[i] = { 0u, 1u, 2 * i, 2 * i + 1 };
        std::memcpy(&pairRecs[size_t(i) * 56 + 28], &i, 4);   // startIndex
        std::memcpy(&pairRecs[size_t(i) * 56 + 40], &i, 4);   // roundedStartIndex
    }
    std::vector<float> contactDist(2 * N, 0.0f);

    const uint32_t nbPairs = N, nbPaddedPairs = N;
    const StackOffsets so = stackLayout(nbPairs, nbPaddedPairs);
    std::vector<uint8_t> stack(size_t(so.total) + 256, 0);
    for (uint32_t i = 0; i < N; ++i)
    {
        uint32_t v[4] = { i, 0u, 0u, 0u };
        std::memcpy(&stack[size_t(so.pairs) + i * 16], v, 16);
    }

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
    std::vector<float> tempContacts(4 * 2 * N, -99.0f);
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
    p.nbContactManagers = N;

    Kernel k("sphereTriNpBda.spv", "sphereTrimeshNarrowphase", {}, sizeof(StnPush));
    k.dispatch(1u, {}, p, 1u, 1u);
    dev->flush();
    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bStack, stack.size());

    const nr::V3 negN = { -0.0f, -1.0f, -0.0f };
    size_t bad = 0, rodriguesDiffers = 0, noContact = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const CachedXf& xs = xf[2 * i];
        const CachedXf& xm = xf[2 * i + 1];
        const nr::V4 sq = { xs.qx, xs.qy, xs.qz, xs.qw }, mq = { xm.qx, xm.qy, xm.qz, xm.qw };
        const nr::V3 lit = nr::quatRotateInv(sq, nr::quatRotate(mq, negN));
        nr::p94dRodriguesTwin() = true;
        const nr::V3 twin = nr::quatRotateInv(sq, nr::quatRotate(mq, negN));
        nr::p94dRodriguesTwin() = false;
        const uint32_t a[3] = { u32(lit.x), u32(lit.y), u32(lit.z) };
        const uint32_t b[3] = { u32(twin.x), u32(twin.y), u32(twin.z) };
        if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2]) ++rodriguesDiffers;

        uint32_t g[4];
        std::memcpy(g, &got[size_t(so.ni) + size_t(i) * 16], 16);
        if (nbContactsOf(g[3]) != 1u || triangleIndexOf(g[3]) != mesh.remap[0]) ++noContact;
        for (uint32_t c = 0; c < 3; ++c)
            if (g[c] != a[c] && ++bad <= 12)
                std::printf("  [pair %u] ni.normal[%u]: GPU 0x%08x literal 0x%08x twin 0x%08x\n",
                            i, c, g[c], a[c], b[c]);
    }
    std::printf("  sphereTriNp_p94dLiteralRotate: %zu/%u NI normals differ under the Rodrigues twin\n",
                rodriguesDiffers, N);
    PXG_CHECK_MSG(noContact == 0, "p94d: %zu pairs did not record exactly one face contact on "
                  "remapped triangle 700 -- the normal comparison would be vacuous", noContact);
    PXG_CHECK_MSG(bad == 0, "p94d: %zu NI normal words differ from the literal PxQuat oracle", bad);
    PXG_CHECK_MSG(rodriguesDiffers > 0,
                  "p94d: the Rodrigues twin gives the same NI normal on every pair -- the literal "
                  "formula is unobserved");

    for (PxgBufferHandle h : { bMesh, bShapes, bXf, bCd, bCms, bPairRecs, bStack,
                               bNbPairs, bScratch, bOutPtrs, bTemp })
        dev->destroyBuffer(h);
}


// ---------------------------------------------------------------------------
// P9.94d unit B: a one-launch runner shared by the two probes below.  Pair i is
// contact manager i (startIndex = roundedStartIndex = i) against triangle tri[i];
// contact distances are zero.
// ---------------------------------------------------------------------------
struct StnRun
{
    StackOffsets so;
    std::vector<uint8_t> stack;
    std::vector<float> temp;
    uint32_t niWord(uint32_t pair, uint32_t w) const
    { uint32_t v; std::memcpy(&v, &stack[size_t(so.ni) + pair * 16 + w * 4], 4); return v; }
    uint32_t contactStart(uint32_t pair) const
    { uint32_t v; std::memcpy(&v, &stack[size_t(so.contacts) + pair * 4], 4); return v; }
    uint32_t tempWord(uint32_t contact, uint32_t w) const
    { uint32_t v; std::memcpy(&v, &temp[size_t(contact) * 4 + w], 4); return v; }
};

static StnRun runStn(PxgSlangComputeDevice* dev, const std::vector<GpuShape>& shapes,
                     const std::vector<CachedXf>& xf, const std::vector<CmInput>& cms,
                     const std::vector<uint32_t>& tri)
{
    const uint32_t N = uint32_t(cms.size());
    StnRun r;
    r.so = stackLayout(N, N);
    r.stack.assign(size_t(r.so.total) + 256, 0);
    std::vector<uint8_t> pairRecs(size_t(N) * 56, 0);
    for (uint32_t i = 0; i < N; ++i)
    {
        uint32_t v[4] = { i, tri[i], 0u, 0u };
        std::memcpy(&r.stack[size_t(r.so.pairs) + i * 16], v, 16);
        std::memcpy(&pairRecs[size_t(i) * 56 + 28], &i, 4);
        std::memcpy(&pairRecs[size_t(i) * 56 + 40], &i, 4);
    }
    std::vector<float> contactDist(xf.size(), 0.0f);
    std::vector<uint32_t> counters(4, 0);
    counters[0] = N;
    counters[1] = N;
    std::vector<uint32_t> scratch(64, 0);
    std::vector<uint64_t> outPtrs(8, 0);
    r.temp.assign(size_t(N) * 2 * 4 + 64, -99.0f);

    PxgBufferHandle bShapes = createBufferFrom(dev, shapes);
    PxgBufferHandle bXf = createBufferFrom(dev, xf);
    PxgBufferHandle bCd = createBufferFrom(dev, contactDist);
    PxgBufferHandle bCms = createBufferFrom(dev, cms);
    PxgBufferHandle bPairRecs = createBufferFrom(dev, pairRecs);
    PxgBufferHandle bStack = createBufferFrom(dev, r.stack);
    PxgBufferHandle bNbPairs = createBufferFrom(dev, counters);
    PxgBufferHandle bScratch = createBufferFrom(dev, scratch);
    PxgBufferHandle bOutPtrs = createBufferFrom(dev, outPtrs);
    PxgBufferHandle bTemp = createBufferFrom(dev, r.temp);
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
    p.stackSizeBytes = uint32_t(r.stack.size());
    p.tempContacts = dev->getBufferDeviceAddress(bTemp);
    p.pTempContactIndex = aScratch + 4;
    p.maxTempMemRequirement = aScratch + 8;
    p.midphasePairsNeeded = aScratch + 12;
    p.nbContactManagers = N;

    Kernel k("sphereTriNpBda.spv", "sphereTrimeshNarrowphase", {}, sizeof(StnPush));
    k.dispatch(1u, {}, p, 1u, 1u);
    dev->flush();
    r.stack = downloadBuffer<uint8_t>(dev, bStack, r.stack.size());
    r.temp = downloadBuffer<float>(dev, bTemp, r.temp.size());
    for (PxgBufferHandle h : { bShapes, bXf, bCd, bCms, bPairRecs, bStack, bNbPairs, bScratch,
                               bOutPtrs, bTemp })
        dev->destroyBuffer(h);
    return r;
}

// ---------------------------------------------------------------------------
// P9.94d: MeshScaling::getShapeSpaceVert's IDENTITY-SCALE BYPASS (cudaNpCommon.h:
// 241-259, `idtMeshScale` set at :197 from the SCALE ALONE).  With scale (1,1,1) the
// raw cooked vertices are used as they are, whatever the scale rotation -- and
// vertex2Shape's `rotateInv(rot, rotate(rot, v) * 1)` is NOT the identity in float,
// so a port without the bypass moves the vertices by ulps and with them the
// triangle normal the sphere arm stores.
//
// 32 triangles, each read twice: through shape A (scale (1,1,1), scale rotation
// normalised (0.36, 0.48, 0.8, 0.53) -- the bypass) and through one of shapes B
// (scale (1,2,1)), C (2,1,1), D (1,1,2) and E (2,0.5,1), same rotation -- NOT the
// bypass.  B kills an `idt = x==1 || y==1 || z==1` mutant; B, C and D together kill
// a test that drops any ONE of the three comparisons; E (determinant exactly 1)
// kills one keyed on the product.  Identity poses, the sphere 0.3 along the oracle's own normal
// above the oracle's centroid, so every pair is one face contact and the stored NI
// normal is `-n` through two literal identity rotations.  Oracles: A on the RAW
// vertices, B through vertex2Shape (PxQuat).  Twins: A through the round trip
// (must differ on some triangle), B..E on the raw vertices (each must differ).
// ---------------------------------------------------------------------------
PXG_TEST(sphereTriNp_p94dIdtScaleRot)
{
    PxgSlangComputeDevice* dev = testDevice();
    namespace nr = pxgtest::npref;
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };

    uint32_t seed = 0x94d0b01u;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float(seed >> 8) * (1.0f / 16777216.0f); };
    auto sym = [&](float m) { return (rnd() * 2.0f - 1.0f) * m; };

    nr::V4 rot = { 0.36f, 0.48f, 0.8f, 0.53f };
    {
        const float r = 1.0f / std::sqrt(rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w);
        rot = nr::V4{ rot.x * r, rot.y * r, rot.z * r, rot.w * r };
    }
    // MeshScaling::vertex2Shape (cudaNpCommon.h:201-211): rotate, scale per axis,
    // rotateInv -- PxQuat's own rotate/rotateInv via npref.
    auto vertex2Shape = [&](nr::V3 v, nr::V3 sc) {
        nr::V3 t = nr::quatRotate(rot, v);
        t.x *= sc.x; t.y *= sc.y; t.z *= sc.z;
        return nr::quatRotateInv(rot, t);
    };

    const uint32_t T = 32;
    Mesh mesh;
    for (uint32_t t = 0; t < T; ++t)
    {
        const V4 a = { sym(3.0f), sym(1.0f), sym(3.0f), 0.0f };
        const V4 b = { a.x + sym(0.4f), a.y + sym(0.3f), a.z + 1.7f + sym(0.3f), 0.0f };
        const V4 c = { a.x + 1.9f + sym(0.3f), a.y + sym(0.3f), a.z + sym(0.4f), 0.0f };
        mesh.verts.push_back(a); mesh.verts.push_back(b); mesh.verts.push_back(c);
        mesh.idx.insert(mesh.idx.end(), { 3 * t, 3 * t + 1, 3 * t + 2, 0u });
        mesh.adj.insert(mesh.adj.end(), { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0u });
        mesh.remap.push_back(500u + t);
    }
    buildMesh(mesh, 1);
    PxgBufferHandle bMesh = createBufferFrom(dev, mesh.blob);
    const uint64_t addrMesh = dev->getBufferDeviceAddress(bMesh);

    const nr::V3 scA = { 1.0f, 1.0f, 1.0f };
    const nr::V3 scNon[4] = { { 1.0f, 2.0f, 1.0f }, { 2.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 2.0f },
                              { 2.0f, 0.5f, 1.0f } };
    std::vector<GpuShape> shapes(6);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    shapes[0].sx = 0.5f; shapes[0].sy = 0.5f; shapes[0].sz = 0.5f;
    shapes[0].rw = 1.0f; shapes[0].type = GEOM_SPHERE;
    for (uint32_t k = 1; k < 6; ++k)
    {
        const nr::V3 sc = k == 1 ? scA : scNon[k - 2];
        shapes[k].sx = sc.x; shapes[k].sy = sc.y; shapes[k].sz = sc.z;
        shapes[k].rx = rot.x; shapes[k].ry = rot.y; shapes[k].rz = rot.z; shapes[k].rw = rot.w;
        shapes[k].type = GEOM_TRIANGLEMESH;
        shapes[k].hullLo = uint32_t(addrMesh & 0xffffffffu);
        shapes[k].hullHi = uint32_t(addrMesh >> 32);
    }

    auto rawVert = [&](uint32_t i) { const V4& v = mesh.verts[i]; return nr::V3{ v.x, v.y, v.z }; };
    // the triangle as the shader should see it: `scaled` false = raw, true = vertex2Shape
    auto triVerts = [&](uint32_t t, bool scaled, nr::V3 sc, nr::V3 out[3]) {
        for (uint32_t k = 0; k < 3; ++k)
            out[k] = scaled ? vertex2Shape(rawVert(3 * t + k), sc) : rawVert(3 * t + k);
    };
    auto normalOf = [](const nr::V3 v[3]) {
        return nr::normalized(nr::cross3(nr::sub(v[1], v[0]), nr::sub(v[2], v[0])));
    };
    const nr::V4 idq = { 0.0f, 0.0f, 0.0f, 1.0f };
    auto niOf = [&](nr::V3 n) { return nr::quatRotateInv(idq, nr::quatRotate(idq, nr::neg(n))); };

    // pairs 0..T-1 through shape A, T..2T-1 through B, C, D, E in turn
    const uint32_t N = 2 * T;
    std::vector<CachedXf> xf(2 * N);
    std::memset(xf.data(), 0, xf.size() * sizeof(CachedXf));
    std::vector<CmInput> cms(N);
    std::vector<uint32_t> tri(N);
    std::vector<nr::V3> expNi(N), twinNi(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t t = i % T;
        const bool isB = i >= T;
        const uint32_t variant = t % 4u;
        const nr::V3 sc = isB ? scNon[variant] : scA;
        nr::V3 v[3], w[3];
        triVerts(t, isB, sc, v);          // literal
        triVerts(t, !isB, sc, w);         // twin: A round-trips, B..E stay raw
        const nr::V3 n = normalOf(v);
        const nr::V3 centroid = nr::mul(nr::add(nr::add(v[0], v[1]), v[2]), 1.0f / 3.0f);
        const nr::V3 centre = nr::add(centroid, nr::mul(n, 0.3f));
        expNi[i] = niOf(n);
        twinNi[i] = niOf(normalOf(w));
        CachedXf& xs = xf[2 * i];
        CachedXf& xm = xf[2 * i + 1];
        xs.qw = 1.0f; xs.px = centre.x; xs.py = centre.y; xs.pz = centre.z;
        xm.qw = 1.0f;
        cms[i] = { 0u, isB ? 2u + variant : 1u, 2 * i, 2 * i + 1 };
        tri[i] = t;
    }

    const StnRun r = runStn(dev, shapes, xf, cms, tri);

    size_t bad = 0, noContact = 0, twinDiffA = 0, twinDiffB[4] = { 0, 0, 0, 0 };
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t idx = r.niWord(i, 3);
        if (nbContactsOf(idx) != 1u || triangleIndexOf(idx) != mesh.remap[i % T] || (idx >> 31) != 0u)
            ++noContact;
        const uint32_t e[3] = { u32(expNi[i].x), u32(expNi[i].y), u32(expNi[i].z) };
        const uint32_t tw[3] = { u32(twinNi[i].x), u32(twinNi[i].y), u32(twinNi[i].z) };
        if (e[0] != tw[0] || e[1] != tw[1] || e[2] != tw[2])
            ++(i < T ? twinDiffA : twinDiffB[(i % T) % 4u]);
        for (uint32_t c = 0; c < 3; ++c)
            if (r.niWord(i, c) != e[c] && ++bad <= 12)
                std::printf("  [pair %u %s] ni.normal[%u]: GPU 0x%08x oracle 0x%08x twin 0x%08x\n",
                            i, i < T ? "unit scale" : "non-unit scale", c, r.niWord(i, c), e[c], tw[c]);
    }
    std::printf("  sphereTriNp_p94dIdtScaleRot: twin differs on %zu/%u bypass pairs; scaled "
                "pairs (1,2,1) %zu, (2,1,1) %zu, (1,1,2) %zu, (2,0.5,1) %zu\n", twinDiffA, T,
                twinDiffB[0], twinDiffB[1], twinDiffB[2], twinDiffB[3]);
    PXG_CHECK_MSG(noContact == 0, "p94d: %zu pairs did not record exactly one undeferred face "
                  "contact -- the normal comparison would be vacuous", noContact);
    PXG_CHECK_MSG(bad == 0, "p94d: %zu NI normal words differ from the oracle", bad);
    PXG_CHECK_MSG(twinDiffA > 0, "p94d: the vertex2Shape round trip at scale (1,1,1) never moves "
                  "a stored normal -- the identity-scale bypass is unobserved");
    PXG_CHECK_MSG(twinDiffB[0] > 0 && twinDiffB[1] > 0 && twinDiffB[2] > 0 && twinDiffB[3] > 0,
                  "p94d: some non-unit scale stores the raw-vertex normal -- a bypass keyed on "
                  "ANY unit component, on two of the three, or on the product would pass");
    dev->destroyBuffer(bMesh);
}

// ---------------------------------------------------------------------------
// P9.94d: the capsule arm's meshToCapsule is `capsuleTransform.transformInv(
// trimeshTransform)` (convexMesh.cu:443 -> PxTransform.h:162-169), whose position is
// `conj(capQ).rotate(meshP - capP)` -- not `capQ.rotateInv(meshP - capP)`, which it
// matches in every nonzero bit and NOT in the sign of an exact zero.  Same search as
// capsuleHfCore_p94dToCapPZeroSign: the 56 signed-zero identity / 180-degree
// quaternions for both frames, both signs of a zero z offset in each, a capsule
// lying along x 0.125 above a large +y triangle at mesh-local z = 0 (and with one
// endpoint at local x = 0).  The stored contact is `rotate(meshToCapQ, pt) +
// meshToCapP`; the candidates where a stored contact word differs between the two
// spellings are the witnesses, and the GPU must match the literal oracle on them.
// ---------------------------------------------------------------------------
PXG_TEST(sphereTriNp_p94dCapsuleMtcZeroSign)
{
    PxgSlangComputeDevice* dev = testDevice();
    namespace nr = pxgtest::npref;
    namespace ct = pxgtest::captriref;
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };

    Mesh mesh;
    mesh.verts = { { -4, 0, -4, 0 }, { -4, 0, 4, 0 }, { 4, 0, 0, 0 } };   // normal exactly +y
    mesh.idx = { 0, 1, 2, 0 };
    mesh.adj = { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0 };
    mesh.remap = { 900 };
    buildMesh(mesh, 1);
    PxgBufferHandle bMesh = createBufferFrom(dev, mesh.blob);
    const uint64_t addrMesh = dev->getBufferDeviceAddress(bMesh);

    std::vector<GpuShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    shapes[0].sx = 0.5f; shapes[0].sy = 0.5f; shapes[0].sz = 0.5f;   // (halfHeight, radius, radius)
    shapes[0].rw = 1.0f; shapes[0].type = GEOM_CAPSULE;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f;
    shapes[1].rw = 1.0f; shapes[1].type = GEOM_TRIANGLEMESH;
    shapes[1].hullLo = uint32_t(addrMesh & 0xffffffffu);
    shapes[1].hullHi = uint32_t(addrMesh >> 32);

    std::vector<nr::V4> qs;
    for (uint32_t sgn = 0; sgn < 8; ++sgn)
        qs.push_back(nr::V4{ (sgn & 1) ? -0.0f : 0.0f, (sgn & 2) ? -0.0f : 0.0f, (sgn & 4) ? -0.0f : 0.0f, 1.0f });
    for (uint32_t axis = 0; axis < 3; ++axis)
        for (uint32_t sgn = 0; sgn < 16; ++sgn)
        {
            float c[4];
            uint32_t z = 0;
            for (uint32_t k = 0; k < 4; ++k)
            {
                if (k == axis) c[k] = (sgn & 1) ? -1.0f : 1.0f;
                else { c[k] = (sgn & (2u << z)) ? -0.0f : 0.0f; ++z; }
            }
            qs.push_back(nr::V4{ c[0], c[1], c[2], c[3] });
        }

    const nr::V3 v0 = { -4, 0, -4 }, v1 = { -4, 0, 4 }, v2 = { 4, 0, 0 };
    struct ArmOut { uint32_t nb; nr::V3 pt[2]; float sep[2]; };
    // stCapsuleArm (convexMesh.cu:400-578), identity mesh scale, zero contact distance --
    // only what reaches the stored contacts.
    auto arm = [&](nr::V4 capQ, nr::V3 capP, nr::V4 meshQ, nr::V3 meshP, bool twin) {
        ArmOut o{};
        const float radius = shapes[0].sy, halfHeight = shapes[0].sx;
        const nr::V4 capQinv = nr::quatConj(capQ);
        const nr::V4 mtcQ = nr::quatMul(capQinv, meshQ);
        const nr::V3 d = nr::sub(meshP, capP);
        const nr::V3 mtcP = twin ? nr::quatRotateInv(capQ, d) : nr::quatRotate(capQinv, d);
        const nr::V3 tmp = nr::mul(nr::basisVector0(capQ), halfHeight);
        const nr::V3 centre = nr::quatRotateInv(meshQ, nr::sub(capP, meshP));
        const nr::V3 dir = nr::quatRotateInv(meshQ, tmp);
        const nr::V3 p0 = nr::add(centre, dir), p1 = nr::sub(centre, dir);
        const nr::V3 triNormal = nr::normalized(nr::cross3(nr::sub(v1, v0), nr::sub(v2, v0)));
        const float inflated = radius + 0.0f;
        const float dist = nr::dot3(nr::sub(centre, v0), triNormal);
        if (!(dist >= 0.0f)) return o;
        float t, u, v;
        const float sqDist = ct::segTriSquared(p0, p1, v0, v1, v2, t, u, v);
        const float sqInflated = inflated * inflated;
        if (!(sqInflated > sqDist)) return o;
        nr::V3 patchNormal;
        if (sqDist < 1.1920928955078125e-07f) patchNormal = triNormal;
        else
        {
            const nr::V3 onSeg = nr::add(p0, nr::mul(nr::sub(p1, p0), t));
            const nr::V3 onTri = nr::add(nr::add(nr::mul(v0, 1.0f - u - v), nr::mul(v1, u)), nr::mul(v2, v));
            patchNormal = nr::normalized(nr::sub(onSeg, onTri));
        }
        ct::Contacts cc = ct::genContacts(v0, v1, v2, triNormal, patchNormal, p0, p1, inflated);
        cc = ct::genEEContacts(v0, v1, v2, patchNormal, p0, p1, sqInflated, cc);
        o.nb = cc.n;
        for (uint32_t i = 0; i < 2 && i < cc.n; ++i)
        {
            o.pt[i] = nr::add(nr::quatRotate(mtcQ, cc.pt[i]), mtcP);
            o.sep[i] = cc.sep[i];
        }
        return o;
    };
    auto sameOut = [&](const ArmOut& a, const ArmOut& b) {
        if (a.nb != b.nb) return false;
        for (uint32_t i = 0; i < a.nb && i < 2; ++i)
            if (u32(a.pt[i].x) != u32(b.pt[i].x) || u32(a.pt[i].y) != u32(b.pt[i].y)
                || u32(a.pt[i].z) != u32(b.pt[i].z) || u32(a.sep[i]) != u32(b.sep[i]))
                return false;
        return true;
    };

    struct Cand { nr::V4 capQ, meshQ; nr::V3 capP, meshP; };
    std::vector<Cand> witnesses, controls;
    size_t searched = 0, entered = 0;
    for (const nr::V4& capQ : qs)
        for (const nr::V4& meshQ : qs)
            for (uint32_t zs = 0; zs < 4; ++zs)
            {
                Cand c;
                c.capQ = capQ; c.meshQ = meshQ;
                c.meshP = nr::V3{ 1.0f, 2.0f, (zs & 1) ? -0.0f : 0.0f };
                const nr::V3 off = nr::quatRotate(meshQ, nr::V3{ 0.5f, 0.125f, 0.0f });
                c.capP = nr::V3{ c.meshP.x + off.x, c.meshP.y + off.y, (zs & 2) ? -0.0f : 0.0f };
                ++searched;
                const ArmOut lit = arm(c.capQ, c.capP, c.meshQ, c.meshP, false);
                if (lit.nb == 0) continue;
                ++entered;
                const ArmOut tw = arm(c.capQ, c.capP, c.meshQ, c.meshP, true);
                if (!sameOut(lit, tw)) { if (witnesses.size() < 24) witnesses.push_back(c); }
                else if (controls.size() < 8) controls.push_back(c);
            }
    std::printf("  sphereTriNp_p94dCapsuleMtcZeroSign: %zu candidates, %zu with contacts, "
                "%zu+ witnesses\n", searched, entered, witnesses.size());
    PXG_CHECK_MSG(entered > 0, "p94d: no candidate reaches the contact loop -- the search is vacuous");
    PXG_CHECK_MSG(!witnesses.empty(),
                  "p94d: no candidate stores a different contact under rotateInv(capQ) -- the "
                  "conjugate spelling of meshToCapsule.p is unobserved");

    std::vector<Cand> cases = witnesses;
    cases.insert(cases.end(), controls.begin(), controls.end());
    const uint32_t N = uint32_t(cases.size());
    std::vector<CachedXf> xf(2 * N);
    std::memset(xf.data(), 0, xf.size() * sizeof(CachedXf));
    std::vector<CmInput> cms(N);
    std::vector<uint32_t> tri(N, 0u);
    for (uint32_t i = 0; i < N; ++i)
    {
        const Cand& c = cases[i];
        CachedXf& xc = xf[2 * i];
        CachedXf& xm = xf[2 * i + 1];
        xc.qx = c.capQ.x; xc.qy = c.capQ.y; xc.qz = c.capQ.z; xc.qw = c.capQ.w;
        xc.px = c.capP.x; xc.py = c.capP.y; xc.pz = c.capP.z;
        xm.qx = c.meshQ.x; xm.qy = c.meshQ.y; xm.qz = c.meshQ.z; xm.qw = c.meshQ.w;
        xm.px = c.meshP.x; xm.py = c.meshP.y; xm.pz = c.meshP.z;
        cms[i] = { 0u, 1u, 2 * i, 2 * i + 1 };
    }
    const StnRun r = runStn(dev, shapes, xf, cms, tri);

    size_t bad = 0, gpuDiffersFromTwin = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const Cand& c = cases[i];
        const ArmOut e = arm(c.capQ, c.capP, c.meshQ, c.meshP, false);
        const ArmOut tw = arm(c.capQ, c.capP, c.meshQ, c.meshP, true);
        const uint32_t idx = r.niWord(i, 3);
        if (nbContactsOf(idx) != e.nb || triangleIndexOf(idx) != 0u)
        {
            if (++bad <= 16) std::printf("  [case %u] niIndex 0x%08x, oracle nb %u\n", i, idx, e.nb);
            continue;
        }
        const uint32_t cs = r.contactStart(i);
        for (uint32_t k = 0; k < e.nb && k < 2; ++k)
        {
            const uint32_t ew[4] = { u32(e.pt[k].x), u32(e.pt[k].y), u32(e.pt[k].z), u32(e.sep[k]) };
            const uint32_t tw4[4] = { u32(tw.pt[k].x), u32(tw.pt[k].y), u32(tw.pt[k].z), u32(tw.sep[k]) };
            bool differsFromTwin = false;
            for (uint32_t w = 0; w < 4; ++w)
            {
                const uint32_t g = r.tempWord(cs + k, w);
                if (g != ew[w] && ++bad <= 16)
                    std::printf("  [case %u] contact %u word %u: GPU 0x%08x literal 0x%08x twin 0x%08x\n",
                                i, k, w, g, ew[w], tw4[w]);
                if (g != tw4[w]) differsFromTwin = true;
            }
            if (differsFromTwin) ++gpuDiffersFromTwin;
        }
    }
    std::printf("  %u cases on the GPU (%zu witnesses), %zu stored contacts differ from the twin\n",
                N, witnesses.size(), gpuDiffersFromTwin);
    PXG_CHECK_MSG(bad == 0, "p94d: %zu stored words differ from the literal oracle", bad);
    PXG_CHECK_MSG(gpuDiffersFromTwin > 0,
                  "p94d: no GPU contact differs from the rotateInv twin on a witness");
    dev->destroyBuffer(bMesh);
}

} // namespace
