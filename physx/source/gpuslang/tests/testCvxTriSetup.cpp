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

// Verification of pxgSlangCvxTriSetup — the prologue of
// convexTrimeshNarrowphaseCore (convexMesh.cu:99-210).
//
// The fixture builds a COOKED MESH BLOB by hand, because the two sections this
// stage needs — the per-triangle adjacency words and the face-remap table — sit
// past the BV32 tree, the vertices and the indices, so every earlier count has to
// be right for them to be right.  The tree is filled with a recognisable pattern
// rather than zeros: a reader that mis-sizes it lands IN the pattern and reports
// garbage instead of silently reading zeros that could be mistaken for real data.

#include "pxgTestFramework.h"

#include "geometry/PxGeometry.h"
#include "foundation/PxQuat.h"

// The BV32 node stride is taken from the ENGINE HEADER, never typed by hand.  It
// was typed by hand once (160) and the shader carried the same wrong number, so
// the fixture and the shader agreed with each other and disagreed with the
// engine — a mutation sweep cannot catch that, because the fixture IS the oracle
// for a layout it also defines.  Anchoring it here makes the compiler the oracle.
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

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 mul(V3 a, V3 b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }

// P9.94d: the RODRIGUES spelling the shader's ctQuatRotate/ctQuatRotateInv carried
// before P9.94d (and this mirror with them): `v*(w*w - u.u) + u*(2 v.u) +- (u x v)*2w`.
// Not PxQuat's formula -- the same rotation in exact arithmetic, a different float
// expression.  Kept ONLY as the vacuity twin selected by gP94dRodrigues.
static V3 quatRotateRodrigues(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) + c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) + c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) + c.z * (w * 2.0f) };
}
static V3 quatRotateInvRodrigues(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) - c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) - c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) - c.z * (w * 2.0f) };
}
// P9.94d mutant twins, false except inside a vacuity probe:
//   gP94dRodrigues    -- both rotations take the old Rodrigues spelling above;
//   gP94dToCvxPRotInv -- toCvxP takes the old q0.rotateInv(p1 - p0) spelling instead
//                        of the CUDA's q0.getConjugate().rotate(p1 - p0).
static bool gP94dRodrigues = false;
static bool gP94dToCvxPRotInv = false;

// PxQuat::rotate / rotateInv (PxQuat.h:286-295 / :324-333), the engine header's own
// code called rather than retyped (this TU is -ffp-contract=off).  P9.94d.
static V3 quatRotate(V4 q, V3 v)
{
    if (gP94dRodrigues) return quatRotateRodrigues(q, v);
    const physx::PxVec3 r = physx::PxQuat(q.x, q.y, q.z, q.w).rotate(physx::PxVec3(v.x, v.y, v.z));
    return { r.x, r.y, r.z };
}
static V3 quatRotateInv(V4 q, V3 v)
{
    if (gP94dRodrigues) return quatRotateInvRodrigues(q, v);
    const physx::PxVec3 r = physx::PxQuat(q.x, q.y, q.z, q.w).rotateInv(physx::PxVec3(v.x, v.y, v.z));
    return { r.x, r.y, r.z };
}
static V4 quatMul(V4 a, V4 b)
{
    return { a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
             a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
             a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
             a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static V4 quatConj(V4 q) { return { -q.x, -q.y, -q.z, q.w }; }
static V3 vertex2Shape(V3 v, V3 scale, V4 rot)
{ return quatRotateInv(rot, mul(quatRotate(rot, v), scale)); }
// P9.36: PxVec3::getNormalized (PxVec3.h:286) is `v * PxRecipSqrt(magnitudeSquared())`,
// and PxRecipSqrt is 1.0f/sqrtf (PxMath.h:161) -- a reciprocal and three multiplies.
static V3 normalized(V3 v)
{
    const float m = std::sqrt(dot3(v, v));
    if (!(m > 0.0f)) return V3{ 0, 0, 0 };
    const float r = 1.0f / m;
    return V3{ v.x * r, v.y * r, v.z * r };
}

static const uint32_t BV32_STRIDE = uint32_t(sizeof(physx::Gu::BV32DataPacked)); // 1168
static_assert(sizeof(physx::Gu::BV32DataPacked) == 1168,
              "BV32DataPacked changed size; the cooked-mesh walk in "
              "pxgSlangCvxTriSetup.slang carries this number too");
// P9.57: FROM THE ENGINE ENUM, never a literal.  `PxGeometryType` grew
// eCONVEXCORE at position 4, which shifted eCONVEXMESH to 5, eTRIANGLEMESH to 8
// and eHEIGHTFIELD to 9; three fixtures and two shaders still carried the old
// numbers, and because a fixture and its shader carried the SAME wrong number
// the differential could not see it.  Taking the value from the header makes
// the compiler the oracle, exactly as this suite already does for the BV32
// stride.  tools/check_geom_type_constants.py keeps the Slang side honest.
static const uint32_t GEOM_TRIANGLEMESH = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
static const uint32_t GEOM_CONVEXMESH = uint32_t(physx::PxGeometryType::eCONVEXMESH);

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

struct SetupPush
{
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, cvxTrimeshPair, pairs, output;
    uint32_t numPairs, pad;
};

// A cooked mesh: header, a BV32 tree filled with a recognisable pattern, verts,
// indices, adjacencies, face remap.
struct Mesh
{
    std::vector<uint8_t> blob;
    std::vector<V4> verts;
    std::vector<uint32_t> idx;      // 4 per triangle (uint4, .w unused)
    std::vector<uint32_t> adj;      // 4 per triangle
    std::vector<uint32_t> remap;    // 1 per triangle
    uint32_t numTris = 0, nbNodes = 0;
};

static void buildMesh(Mesh& m, uint32_t nbNodes)
{
    m.nbNodes = nbNodes;
    const uint32_t numVerts = uint32_t(m.verts.size());
    m.numTris = uint32_t(m.idx.size() / 4);

    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) m.blob.push_back(uint8_t(v >> (8 * i))); };
    auto f32 = [&](float v) { uint32_t u; std::memcpy(&u, &v, 4); u32(u); };

    u32(numVerts); u32(m.numTris); u32(0xA5A5A5A5u); u32(nbNodes);   // .z is NOT a node count
    // The tree: a recognisable pattern, so a mis-sized walk reports nonsense
    // rather than zeros that could pass for data.
    for (uint32_t i = 0; i < nbNodes * BV32_STRIDE / 4; ++i) u32(0xDEAD0000u | (i & 0xffffu));
    for (const V4& v : m.verts) { f32(v.x); f32(v.y); f32(v.z); f32(v.w); }
    for (uint32_t v : m.idx) u32(v);
    for (uint32_t v : m.adj) u32(v);
    for (uint32_t v : m.remap) u32(v);
    for (int i = 0; i < 64; ++i) u32(0);   // trailing sections this stage never reads
}

struct Pair { uint32_t cm, tri, off, pad; };

// convexTransform.transformInv(trimeshTransform).p (convexMesh.cu:139) =
// PxTransform::transformInv(PxTransform) (PxTransform.h:162-169) LITERALLY:
// qinv = q0.getConjugate(); qinv.rotate(p1 - p0).  P9.94d.
static V3 toCvxPRef(V4 q0, V3 p0, V3 p1)
{
    const V3 d = sub(p1, p0);
    return gP94dToCvxPRotInv ? quatRotateInv(q0, d) : quatRotate(quatConj(q0), d);
}

// The three triangle vertices in convex space and the normal, as the stage stores
// them (convexMesh.cu:139, :176-196 and the normal of :204-206).
struct TriRef { V3 e0, e1, e2, en; bool flip; };
static TriRef refTri(const Mesh& mesh, uint32_t tri, V3 triScale, V4 triRot,
                     V4 q0, V3 p0, V4 q1, V3 p1)
{
    const V4 toCvxQ = quatMul(quatConj(q0), q1);
    const V3 toCvxP = toCvxPRef(q0, p0, p1);
    TriRef r;
    r.flip = (triScale.x * triScale.y * triScale.z) < 0.0f;
    V3 loc[3];
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t vi = mesh.idx[4 * tri + k];
        const V4 v4 = mesh.verts[vi];
        const V3 v = { v4.x, v4.y, v4.z };
        loc[k] = add(quatRotate(toCvxQ, vertex2Shape(v, triScale, triRot)), toCvxP);
    }
    r.e0 = loc[0];
    r.e1 = r.flip ? loc[2] : loc[1];
    r.e2 = r.flip ? loc[1] : loc[2];
    r.en = normalized(cross3(sub(r.e1, r.e0), sub(r.e2, r.e0)));
    return r;
}

// The twelve stored float words a TriRef predicts, and where they sit in the
// 128-byte output record (in floats).
static const uint32_t kRefWordAt[12] = { 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26 };
static void refWords(const TriRef& r, uint32_t w[12])
{
    const float f[12] = { r.e0.x, r.e0.y, r.e0.z, r.e1.x, r.e1.y, r.e1.z,
                          r.e2.x, r.e2.y, r.e2.z, r.en.x, r.en.y, r.en.z };
    std::memcpy(w, f, sizeof(f));
}

// Uploads one fixture, runs cvxTriSetupLaunch over `pairs` and returns the
// 128-byte-per-pair output.  The shapes already carry the device addresses.
static std::vector<float> runSetupKernel(PxgSlangComputeDevice* dev,
                                         const std::vector<GpuShape>& shapes,
                                         const std::vector<float>& xf,
                                         const std::vector<float>& cdist,
                                         const std::vector<CmInput>& cms,
                                         const std::vector<uint8_t>& pairBlob,
                                         const std::vector<Pair>& pairs)
{
    const uint32_t n = uint32_t(pairs.size());
    PxgBufferHandle bShapes = createBufferFrom(dev, shapes);
    PxgBufferHandle bXf = createBufferFrom(dev, xf);
    PxgBufferHandle bCd = createBufferFrom(dev, cdist);
    PxgBufferHandle bCm = createBufferFrom(dev, cms);
    PxgBufferHandle bPairBlob = createBufferFrom(dev, pairBlob);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairs);
    std::vector<float> outInit(size_t(n) * (128 / 4), -12345.0f);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    SetupPush push{};
    push.cmInputs = dev->getBufferDeviceAddress(bCm);
    push.transformCache = dev->getBufferDeviceAddress(bXf);
    push.contactDistance = dev->getBufferDeviceAddress(bCd);
    push.gpuShapes = dev->getBufferDeviceAddress(bShapes);
    push.cvxTrimeshPair = dev->getBufferDeviceAddress(bPairBlob);
    push.pairs = dev->getBufferDeviceAddress(bPairs);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numPairs = n;

    {
        Kernel k("cvxTriSetupBda.spv", "cvxTriSetupLaunch", {}, sizeof(SetupPush));
        k.dispatch(n, {}, push, 1u, 1u);
        dev->flush();
    }

    std::vector<float> got = downloadBuffer<float>(dev, bOut, outInit.size());
    dev->destroyBuffer(bShapes); dev->destroyBuffer(bXf); dev->destroyBuffer(bCd);
    dev->destroyBuffer(bCm); dev->destroyBuffer(bPairBlob); dev->destroyBuffer(bPairs);
    dev->destroyBuffer(bOut);
    return got;
}

PXG_TEST(cvxTriSetupPrologue)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- two meshes: one with a positive scale, one the fixture will scale
    //      negatively, and different node counts so a mis-sized tree walk cannot
    //      accidentally agree between them ----------------------------------
    Mesh meshA, meshB;
    meshA.verts = { { 0, 0, 0, 0 }, { 2, 0, 0, 0 }, { 0, 3, 0, 0 }, { 1, 1, 4, 0 }, { -2, 1, 1, 0 } };
    // Triangle 3 is DEGENERATE (two coincident vertices), so the cross product is
    // the zero vector and getNormalized() has to take its zero branch rather than
    // divide.  Without it that guard is never exercised.
    meshA.idx = { 0, 1, 2, 0,   1, 3, 2, 0,   4, 0, 2, 0,   1, 1, 3, 0 };
    meshA.adj = { 11, 12, 13, 0,   0xffffffffu, 21, 0x80000000u, 0,
                  31, 32, 33, 0,   51, 52, 53, 0 };
    meshA.remap = { 700, 701, 702, 703 };
    buildMesh(meshA, 3);

    meshB.verts = { { 1, -1, 0, 0 }, { 4, 0, 1, 0 }, { 0, 5, -2, 0 }, { 2, 2, 2, 0 } };
    meshB.idx = { 0, 1, 2, 0,   3, 1, 0, 0 };
    meshB.adj = { 41, 42, 43, 0,   0xffffffffu, 0xffffffffu, 0xffffffffu, 0 };
    meshB.remap = { 900, 901 };
    buildMesh(meshB, 1);

    // A stand-in hull blob; the setup stage only forwards its address.
    std::vector<uint32_t> hullBlob(64, 0x1234u);

    PxgBufferHandle bMeshA = createBufferFrom(dev, meshA.blob);
    PxgBufferHandle bMeshB = createBufferFrom(dev, meshB.blob);
    PxgBufferHandle bHull = createBufferFrom(dev, hullBlob);
    const uint64_t addrMeshA = dev->getBufferDeviceAddress(bMeshA);
    const uint64_t addrMeshB = dev->getBufferDeviceAddress(bMeshB);
    const uint64_t addrHull = dev->getBufferDeviceAddress(bHull);

    // ---- shapes ------------------------------------------------------------
    // 0: convex.  1: mesh A, positive scale.  2: mesh B, NEGATIVE determinant.
    // 3: a second convex with a different scale rotation.
    std::vector<GpuShape> shapes(5);
    auto setShape = [](GpuShape& s, V3 sc, V4 rot, uint64_t ptr, uint32_t type) {
        s.sx = sc.x; s.sy = sc.y; s.sz = sc.z;
        s.rx = rot.x; s.ry = rot.y; s.rz = rot.z; s.rw = rot.w;
        s.materialIndex = 0;
        s.hullLo = uint32_t(ptr & 0xffffffffu);
        s.hullHi = uint32_t(ptr >> 32);
        s.type = type;
        s.particleOrSoftbodyId = 0;
    };
    setShape(shapes[0], { 1.3f, 0.8f, 1.1f }, { 0, 0, 0, 1 }, addrHull, GEOM_CONVEXMESH);
    {
        // A NON-IDENTITY MESH SCALE ROTATION.  With every scale rotation the
        // identity, vertex2Shape degenerates to a componentwise multiply and both
        // quaternion rotations inside it become no-ops -- so dropping either one
        // is invisible.
        const float a = 0.9f;
        setShape(shapes[1], { 1.0f, 1.6f, 0.7f },
                 { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, addrMeshA, GEOM_TRIANGLEMESH);
    }
    setShape(shapes[2], { 1.5f, -0.7f, 1.2f }, { 0, 0, 0, 1 }, addrMeshB, GEOM_TRIANGLEMESH);
    {
        const float a = 0.4f;
        setShape(shapes[3], { 0.9f, 1.4f, 0.6f },
                 { 0.0f, std::sin(a * 0.5f), 0.0f, std::cos(a * 0.5f) }, addrHull, GEOM_CONVEXMESH);
    }
    // TWO negative scale components: the determinant is POSITIVE, so the winding
    // must NOT flip.  A guard written as "any component is negative" agrees with
    // the determinant whenever exactly one is negative, which is the only shape
    // the fixture had.
    setShape(shapes[4], { -1.5f, -0.7f, 1.2f }, { 0, 0, 0, 1 }, addrMeshB, GEOM_TRIANGLEMESH);

    // ---- transforms (PxsCachedTransform: quat @0, pos @16, 32 B) -----------
    std::vector<float> xf(6 * 8, 0.0f);
    auto setXf = [&](uint32_t i, V4 q, V3 p) {
        float* r = xf.data() + i * 8;
        r[0] = q.x; r[1] = q.y; r[2] = q.z; r[3] = q.w;
        r[4] = p.x; r[5] = p.y; r[6] = p.z;
    };
    // THE CONVEX BODY MUST ROTATE.  The composition is conj(q0)*q1 with
    // conj(q0).rotate(p1 - p0); with q0 the identity the conjugate is a no-op and the
    // translation needs no rotating, so three separate mistakes become invisible.
    {
        const float a = 0.55f;
        setXf(0, { 0.0f, 0.0f, std::sin(a * 0.5f), std::cos(a * 0.5f) }, { 0.5f, -0.25f, 1.0f });
    }
    {
        const float a = 0.7f;
        setXf(1, { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, { -1.0f, 2.0f, 0.5f });
    }
    {
        const float a = -1.1f;
        setXf(2, { 0.0f, 0.0f, std::sin(a * 0.5f), std::cos(a * 0.5f) }, { 3.0f, 0.0f, -1.5f });
    }
    {
        const float a = -0.8f;
        setXf(3, { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, { 1.5f, -2.0f, 0.75f });
    }
    setXf(4, { 0, 0, 0, 1 }, { -0.5f, 1.25f, 2.0f });
    setXf(5, { 0, 0, 0, 1 }, { 2.25f, 0.5f, -1.0f });

    std::vector<float> cdist = { 0.02f, 0.031f, 0.047f, 0.013f, 0.005f, 0.061f };

    // ---- contact managers --------------------------------------------------
    // 0: convex first.  1: MESH FIRST, so the swap has to fire.  2: the negative
    // scale.  3: the rotated convex.
    // THE TRANSFORM REFS MUST NOT EQUAL THE SHAPE REFS.  While they match, the
    // contact distance -- which is indexed by the TRANSFORM refs -- reads the same
    // slots either way, so indexing it by the shape refs is invisible.
    std::vector<CmInput> cms = {
        { 0, 1, 4, 5 },
        { 1, 0, 5, 4 },   // shapeRef0 IS the mesh -> swap
        { 0, 2, 0, 2 },
        { 3, 1, 3, 1 },
        { 0, 4, 4, 2 },   // two negative scale components: no flip
    };

    // ---- ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40 -------
    // Deliberately different values so a port reading `count`/`cmIndex` instead
    // (the offsets the declaration's trailing comment numbers suggest) is caught.
    std::vector<uint8_t> pairBlob(8 * 56, 0);
    auto setPair = [&](uint32_t cm, uint32_t startIndex, uint32_t count, uint32_t cmIndex,
                       uint32_t rounded) {
        uint8_t* r = pairBlob.data() + size_t(cm) * 56;
        std::memcpy(r + 28, &startIndex, 4);
        std::memcpy(r + 32, &count, 4);
        std::memcpy(r + 36, &cmIndex, 4);
        std::memcpy(r + 40, &rounded, 4);
    };
    setPair(0, 100, 7, 55, 200);
    setPair(1, 300, 9, 66, 400);
    setPair(2, 500, 3, 77, 600);
    setPair(3, 800, 5, 88, 900);
    setPair(4, 1100, 6, 99, 1200);

    // ---- the pairs to run --------------------------------------------------
    std::vector<Pair> pairs = {
        { 0, 0, 0, 0 },
        { 0, 2, 3, 0 },     // a different triangle of the same mesh
        { 1, 1, 1, 0 },     // the swapped pair
        { 2, 1, 2, 0 },     // the negative determinant
        { 3, 1, 4, 0 },     // a rotated convex scale
        { 0, 3, 5, 0 },     // the DEGENERATE triangle
        { 4, 0, 6, 0 },     // two negative scale components -> determinant positive
    };

    const uint32_t n = uint32_t(pairs.size());
    const auto got = runSetupKernel(dev, shapes, xf, cdist, cms, pairBlob, pairs);

    size_t bad = 0;
    size_t sawSwap = 0, sawFlip = 0, sawBoundaryAdj = 0, sawRotatedScale = 0;
    size_t sawDegenerate = 0, sawNegativePairNoFlip = 0, sawRodriguesDiffers = 0;
    auto sameF = [](float a, float b)
    { uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y; };

    for (uint32_t i = 0; i < n; ++i)
    {
        const Pair& pr = pairs[i];
        const CmInput& cm = cms[pr.cm];

        uint32_t sr0 = cm.shapeRef0, sr1 = cm.shapeRef1, tc0 = cm.tcRef0, tc1 = cm.tcRef1;
        if (shapes[sr0].type == GEOM_TRIANGLEMESH)
        { std::swap(sr0, sr1); std::swap(tc0, tc1); ++sawSwap; }

        const GpuShape& cvx = shapes[sr0];
        const GpuShape& tri = shapes[sr1];
        const Mesh& mesh = (tri.hullLo == uint32_t(addrMeshA & 0xffffffffu) &&
                            tri.hullHi == uint32_t(addrMeshA >> 32)) ? meshA : meshB;

        const V4 q0 = { xf[tc0 * 8 + 0], xf[tc0 * 8 + 1], xf[tc0 * 8 + 2], xf[tc0 * 8 + 3] };
        const V3 p0 = { xf[tc0 * 8 + 4], xf[tc0 * 8 + 5], xf[tc0 * 8 + 6] };
        const V4 q1 = { xf[tc1 * 8 + 0], xf[tc1 * 8 + 1], xf[tc1 * 8 + 2], xf[tc1 * 8 + 3] };
        const V3 p1 = { xf[tc1 * 8 + 4], xf[tc1 * 8 + 5], xf[tc1 * 8 + 6] };
        const V3 triScale = { tri.sx, tri.sy, tri.sz };
        const V4 triRot = { tri.rx, tri.ry, tri.rz, tri.rw };
        const bool flip = (triScale.x * triScale.y * triScale.z) < 0.0f;
        if (flip) ++sawFlip;
        if (!flip && (triScale.x < 0.0f || triScale.y < 0.0f || triScale.z < 0.0f))
            ++sawNegativePairNoFlip;
        if ((cvx.rx != 0.0f) || (cvx.ry != 0.0f) || (cvx.rz != 0.0f)) ++sawRotatedScale;

        const TriRef ref = refTri(mesh, pr.tri, triScale, triRot, q0, p0, q1, p1);
        const V3 e0 = ref.e0, e1 = ref.e1, e2 = ref.e2, en = ref.en;
        // P9.94d vacuity: the same pair under the OLD Rodrigues rotations.  At least
        // one pair must store a different word, or this fixture never saw the
        // shader's switch to the literal PxQuat formula.
        {
            gP94dRodrigues = true;
            const TriRef twin = refTri(mesh, pr.tri, triScale, triRot, q0, p0, q1, p1);
            gP94dRodrigues = false;
            uint32_t a[12], b[12];
            refWords(ref, a); refWords(twin, b);
            for (uint32_t k = 0; k < 12; ++k) if (a[k] != b[k]) { ++sawRodriguesDiffers; break; }
        }

        if (dot3(cross3(sub(e1, e0), sub(e2, e0)), cross3(sub(e1, e0), sub(e2, e0))) == 0.0f)
            ++sawDegenerate;
        const uint32_t adjX = mesh.adj[4 * pr.tri + 0];
        if (adjX == 0xffffffffu) ++sawBoundaryAdj;

        const float* r = got.data() + size_t(i) * (128 / 4);
        uint32_t u[4];
        std::memcpy(u, r, 16);
        const uint64_t gotHull = uint64_t(u[0]) | (uint64_t(u[1]) << 32);
        const uint64_t expHull = uint64_t(cvx.hullLo) | (uint64_t(cvx.hullHi) << 32);

        auto chkU = [&](const char* w, uint32_t g, uint32_t e)
        { if (g != e) { ++bad; if (bad <= 12) std::printf("  [pair %u] %s: got=%u exp=%u\n", i, w, g, e); } };
        auto chkF = [&](const char* w, float g, float e)
        { if (!sameF(g, e)) { ++bad; if (bad <= 12) std::printf("  [pair %u] %s: got=%.9g exp=%.9g\n", i, w, g, e); } };

        if (gotHull != expHull)
        { ++bad; std::printf("  [pair %u] hull address mismatch\n", i); }
        chkU("remap", u[2], mesh.remap[pr.tri]);
        {
            uint32_t startIndex, rounded;
            std::memcpy(&startIndex, pairBlob.data() + size_t(pr.cm) * 56 + 28, 4);
            std::memcpy(&rounded, pairBlob.data() + size_t(pr.cm) * 56 + 40, 4);
            chkU("pairOffsetPadded", u[3], rounded + pr.off);
            uint32_t adjs[4];
            std::memcpy(adjs, r + 28, 16);
            chkU("adj.x", adjs[0], mesh.adj[4 * pr.tri + 0]);
            chkU("adj.y", adjs[1], mesh.adj[4 * pr.tri + 1]);
            chkU("adj.z", adjs[2], mesh.adj[4 * pr.tri + 2]);
            chkU("pairOffset", adjs[3], startIndex + pr.off);
        }
        chkF("scale.x", r[4], cvx.sx);
        chkF("scale.y", r[5], cvx.sy);
        chkF("scale.z", r[6], cvx.sz);
        chkF("contactDist", r[7], cdist[tc0] + cdist[tc1]);
        chkF("scaleRot.x", r[8], cvx.rx);
        chkF("scaleRot.w", r[11], cvx.rw);
        chkF("v0.x", r[12], e0.x); chkF("v0.y", r[13], e0.y); chkF("v0.z", r[14], e0.z);
        chkF("v1.x", r[16], e1.x); chkF("v1.y", r[17], e1.y); chkF("v1.z", r[18], e1.z);
        chkF("v2.x", r[20], e2.x); chkF("v2.y", r[21], e2.y); chkF("v2.z", r[22], e2.z);
        chkF("n.x", r[24], en.x); chkF("n.y", r[25], en.y); chkF("n.z", r[26], en.z);
    }

    dev->destroyBuffer(bMeshA); dev->destroyBuffer(bMeshB); dev->destroyBuffer(bHull);

    PXG_CHECK_MSG(bad == 0, "cvxTriSetup: %zu mismatches", bad);
    PXG_CHECK_MSG(sawSwap > 0, "no case has the mesh in slot 0 (the swap is untested)");
    PXG_CHECK_MSG(sawFlip > 0, "no case has a negative scale determinant (the winding flip is untested)");
    PXG_CHECK_MSG(sawBoundaryAdj > 0, "no case carries a BOUNDARY adjacency word");
    PXG_CHECK_MSG(sawRotatedScale > 0, "no case has a non-identity convex scale rotation");
    PXG_CHECK_MSG(sawDegenerate > 0, "no case has a degenerate triangle (the zero-normal branch)");
    PXG_CHECK_MSG(sawNegativePairNoFlip > 0,
                  "no case has negative scale components WITHOUT a negative determinant");
    PXG_CHECK_MSG(sawRodriguesDiffers > 0,
                  "no case stores a different word under the old Rodrigues rotation "
                  "(the P9.94d literal PxQuat formula is unobserved)");
}

// P9.94d: toCvxP = convexTransform.transformInv(trimeshTransform).p (convexMesh.cu:139)
// is q0.getConjugate().rotate(p1 - p0) (PxTransform.h:162-169), NOT q0.rotateInv(p1 - p0).
// The two agree on every nonzero component; this case makes them disagree on a zero.
//
// q0 = identity, p0 = +0, p1 = (-0,-0,-0): d = p1 - p0 = (-0,-0,-0).  conj(q0) is
// (-0,-0,-0,1), and its rotate gives, per axis, vx*w2 + (y*vz - z*vy)*w + x*dot2 =
// (-0) + (+0 - +0 = +0) + (-0 * +0 = -0) = +0; q0.rotateInv gives
// vx*w2 - (y*vz - z*vy)*w + x*dot2 = (-0) - (-0 - -0 = +0) + (+0 * -0 = -0) = -0.  So
// toCvxP is (+0,+0,+0) under the CUDA and (-0,-0,-0) under the old spelling.  q1 = (1,0,0,0) (180 deg about x) turns a vertex with y = +0 into a
// rotated y of -0, and -0 + toCvxP.y keeps toCvxP.y's sign: the stored vertex y is +0
// (CUDA) against -0 (old).  The mesh scale is the identity, so vertex2Shape leaves the
// coordinates' magnitudes alone.  No RNG.
PXG_TEST(cvxTriSetup_p94dToCvxPZeroSign)
{
    PxgSlangComputeDevice* dev = testDevice();
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    const float nz = -0.0f;

    Mesh mesh;
    mesh.verts = { { -1, 0, 1, 0 }, { 2, 0, 1, 0 }, { 0, 0, -3, 0 } };
    mesh.idx = { 0, 1, 2, 0 };
    mesh.adj = { 5, 6, 7, 0 };
    mesh.remap = { 400 };
    buildMesh(mesh, 1);
    std::vector<uint32_t> hullBlob(64, 0x1234u);
    PxgBufferHandle bMesh = createBufferFrom(dev, mesh.blob);
    PxgBufferHandle bHull = createBufferFrom(dev, hullBlob);
    const uint64_t addrMesh = dev->getBufferDeviceAddress(bMesh);
    const uint64_t addrHull = dev->getBufferDeviceAddress(bHull);

    std::vector<GpuShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    shapes[0].sx = 1.0f; shapes[0].sy = 1.0f; shapes[0].sz = 1.0f; shapes[0].rw = 1.0f;
    shapes[0].hullLo = uint32_t(addrHull & 0xffffffffu); shapes[0].hullHi = uint32_t(addrHull >> 32);
    shapes[0].type = GEOM_CONVEXMESH;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].rw = 1.0f;
    shapes[1].hullLo = uint32_t(addrMesh & 0xffffffffu); shapes[1].hullHi = uint32_t(addrMesh >> 32);
    shapes[1].type = GEOM_TRIANGLEMESH;

    const V4 q0 = { 0, 0, 0, 1 }, q1 = { 1, 0, 0, 0 };
    const V3 p0 = { 0, 0, 0 }, p1 = { nz, nz, nz };
    std::vector<float> xf(2 * 8, 0.0f);
    xf[0] = q0.x; xf[1] = q0.y; xf[2] = q0.z; xf[3] = q0.w; xf[4] = p0.x; xf[5] = p0.y; xf[6] = p0.z;
    xf[8] = q1.x; xf[9] = q1.y; xf[10] = q1.z; xf[11] = q1.w; xf[12] = p1.x; xf[13] = p1.y; xf[14] = p1.z;
    std::vector<float> cdist = { 0.01f, 0.02f };
    std::vector<CmInput> cms = { { 0, 1, 0, 1 } };
    std::vector<uint8_t> pairBlob(56, 0);
    std::vector<Pair> pairs = { { 0, 0, 0, 0 } };

    const V3 one = { 1, 1, 1 };
    const V4 idq = { 0, 0, 0, 1 };
    const TriRef lit = refTri(mesh, 0, one, idq, q0, p0, q1, p1);
    gP94dToCvxPRotInv = true;
    const TriRef twin = refTri(mesh, 0, one, idq, q0, p0, q1, p1);
    gP94dToCvxPRotInv = false;
    uint32_t a[12], b[12];
    refWords(lit, a); refWords(twin, b);
    PXG_CHECK_MSG(u32(lit.e0.y) == 0x00000000u && u32(twin.e0.y) == 0x80000000u,
                  "p94d: literal v0.y 0x%08x (want +0), twin 0x%08x (want -0)",
                  u32(lit.e0.y), u32(twin.e0.y));
    uint32_t differ = 0;
    for (uint32_t k = 0; k < 12; ++k) if (a[k] != b[k]) ++differ;
    PXG_CHECK_MSG(differ > 0, "p94d: the rotateInv twin stores the same words -- not discriminating");

    const auto got = runSetupKernel(dev, shapes, xf, cdist, cms, pairBlob, pairs);
    for (uint32_t k = 0; k < 12; ++k)
    {
        const uint32_t g = u32(got[kRefWordAt[k]]);
        PXG_CHECK_MSG(g == a[k], "p94d: stored word %u (float %u) GPU 0x%08x != literal 0x%08x (twin 0x%08x)",
                      k, kRefWordAt[k], g, a[k], b[k]);
    }
    dev->destroyBuffer(bMesh); dev->destroyBuffer(bHull);
}

// P9.94d: ctQuatRotateInv must keep PxQuat::rotateInv's OPERAND ORDER
// (PxQuat.h:324-333), not merely its value.  Its z lane is
// `vz*w2 - (x*vy - y*vx)*w + z*dot2`; the respelling `vz*w2 + (y*vx - x*vy)*w + z*dot2`
// is the same number whenever the cross term is nonzero, but when x*vy == y*vx the
// bracket is +0 in BOTH spellings, and `a - (+0*w)` against `a + (+0*w)` differs on
// a = -0.  The P9.94d sweep left that respelling (R12) alive: no fixture reached the
// zero.  A host search over 180-degree quaternions, signed-zero vertices and +-1 scales
// found this witness, which reaches it through vertex2Shape's rotateInv (convexMesh.cu
// :176-196, cudaNpCommon.h:265-276):
//   mesh scale (1,-1,1), scale rotation (1,0,0,0), vertex (-1,-0,-0),
//   q0 = (1,0,0,0), q1 = (-1,0,-0,0), p0 = 0, p1 = (-0,0,0)
// stores v0 = (-1,+0,-0) under the literal formula and (-1,+0,+0) under R12.  No RNG.
PXG_TEST(cvxTriSetup_p94dRotateInvZeroSign)
{
    PxgSlangComputeDevice* dev = testDevice();
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    const float nz = -0.0f;

    Mesh mesh;
    mesh.verts = { { -1, nz, nz, 0 }, { 2, 1, 0.5f, 0 }, { 0.5f, 3, -1, 0 } };
    mesh.idx = { 0, 1, 2, 0 };
    mesh.adj = { 5, 6, 7, 0 };
    mesh.remap = { 400 };
    buildMesh(mesh, 1);
    std::vector<uint32_t> hullBlob(64, 0x1234u);
    PxgBufferHandle bMesh = createBufferFrom(dev, mesh.blob);
    PxgBufferHandle bHull = createBufferFrom(dev, hullBlob);
    const uint64_t addrMesh = dev->getBufferDeviceAddress(bMesh);
    const uint64_t addrHull = dev->getBufferDeviceAddress(bHull);

    const V3 sc = { 1.0f, -1.0f, 1.0f };
    const V4 rq = { 1, 0, 0, 0 };
    std::vector<GpuShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    shapes[0].sx = 1.0f; shapes[0].sy = 1.0f; shapes[0].sz = 1.0f; shapes[0].rw = 1.0f;
    shapes[0].hullLo = uint32_t(addrHull & 0xffffffffu); shapes[0].hullHi = uint32_t(addrHull >> 32);
    shapes[0].type = GEOM_CONVEXMESH;
    shapes[1].sx = sc.x; shapes[1].sy = sc.y; shapes[1].sz = sc.z;
    shapes[1].rx = rq.x; shapes[1].ry = rq.y; shapes[1].rz = rq.z; shapes[1].rw = rq.w;
    shapes[1].hullLo = uint32_t(addrMesh & 0xffffffffu); shapes[1].hullHi = uint32_t(addrMesh >> 32);
    shapes[1].type = GEOM_TRIANGLEMESH;

    const V4 q0 = { 1, 0, 0, 0 }, q1 = { -1, 0, nz, 0 };
    const V3 p0 = { 0, 0, 0 }, p1 = { nz, 0, 0 };
    std::vector<float> xf(2 * 8, 0.0f);
    xf[0] = q0.x; xf[1] = q0.y; xf[2] = q0.z; xf[3] = q0.w; xf[4] = p0.x; xf[5] = p0.y; xf[6] = p0.z;
    xf[8] = q1.x; xf[9] = q1.y; xf[10] = q1.z; xf[11] = q1.w; xf[12] = p1.x; xf[13] = p1.y; xf[14] = p1.z;
    std::vector<float> cdist = { 0.01f, 0.02f };
    std::vector<CmInput> cms = { { 0, 1, 0, 1 } };
    std::vector<uint8_t> pairBlob(56, 0);
    std::vector<Pair> pairs = { { 0, 0, 0, 0 } };

    const TriRef lit = refTri(mesh, 0, sc, rq, q0, p0, q1, p1);
    PXG_CHECK_MSG(u32(lit.e0.x) == u32(-1.0f) && u32(lit.e0.y) == 0x00000000u &&
                  u32(lit.e0.z) == 0x80000000u,
                  "p94d: literal v0 (0x%08x 0x%08x 0x%08x), want (-1, +0, -0)",
                  u32(lit.e0.x), u32(lit.e0.y), u32(lit.e0.z));
    uint32_t a[12];
    refWords(lit, a);
    const auto got = runSetupKernel(dev, shapes, xf, cdist, cms, pairBlob, pairs);
    for (uint32_t k = 0; k < 12; ++k)
    {
        const uint32_t g = u32(got[kRefWordAt[k]]);
        PXG_CHECK_MSG(g == a[k], "p94d: stored word %u (float %u) GPU 0x%08x != literal 0x%08x",
                      k, kRefWordAt[k], g, a[k]);
    }
    dev->destroyBuffer(bMesh); dev->destroyBuffer(bHull);
}

// P9.94d: ctQuatRotate/ctQuatRotateInv are PxQuat::rotate/rotateInv LITERALLY.  Random
// poses, random mesh scales and scale rotations, both slot orders: every stored vertex
// and normal word must equal the literal oracle BITWISE, and the old Rodrigues spelling
// must store a different word on some of them -- otherwise this test cannot tell the
// two formulas apart and proves nothing about the switch.
PXG_TEST(cvxTriSetup_p94dLiteralRotate)
{
    PxgSlangComputeDevice* dev = testDevice();
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };

    uint32_t seed = 0x94d0001u;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float(seed >> 8) * (1.0f / 16777216.0f); };
    auto sym = [&](float m) { return (rnd() * 2.0f - 1.0f) * m; };
    auto unitQ = [&]() {
        V4 q = { sym(1), sym(1), sym(1), sym(1) };
        const float r = 1.0f / std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        return V4{ q.x * r, q.y * r, q.z * r, q.w * r };
    };

    Mesh mesh;
    for (uint32_t i = 0; i < 9; ++i) mesh.verts.push_back({ sym(3), sym(3), sym(3), 0 });
    mesh.idx = { 0, 1, 2, 0,   3, 4, 5, 0,   6, 7, 8, 0,   0, 4, 8, 0 };
    mesh.adj = { 1, 2, 3, 0,   0, 2, 3, 0,   0, 1, 3, 0,   0, 1, 2, 0 };
    mesh.remap = { 10, 11, 12, 13 };
    buildMesh(mesh, 2);
    std::vector<uint32_t> hullBlob(64, 0x1234u);
    PxgBufferHandle bMesh = createBufferFrom(dev, mesh.blob);
    PxgBufferHandle bHull = createBufferFrom(dev, hullBlob);
    const uint64_t addrMesh = dev->getBufferDeviceAddress(bMesh);
    const uint64_t addrHull = dev->getBufferDeviceAddress(bHull);

    const uint32_t N = 128;
    std::vector<GpuShape> shapes(1 + N);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    shapes[0].sx = 1.2f; shapes[0].sy = 0.9f; shapes[0].sz = 1.1f; shapes[0].rw = 1.0f;
    shapes[0].hullLo = uint32_t(addrHull & 0xffffffffu); shapes[0].hullHi = uint32_t(addrHull >> 32);
    shapes[0].type = GEOM_CONVEXMESH;
    std::vector<float> xf(size_t(2 * N) * 8, 0.0f);
    std::vector<float> cdist(2 * N, 0.01f);
    std::vector<CmInput> cms(N);
    std::vector<uint8_t> pairBlob(size_t(N) * 56, 0);
    std::vector<Pair> pairs(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        GpuShape& m = shapes[1 + i];
        auto mag = [&]() { const float s = 0.5f + rnd() * 1.5f; return rnd() < 0.2f ? -s : s; };
        m.sx = mag(); m.sy = mag(); m.sz = mag();
        const V4 r = unitQ();
        m.rx = r.x; m.ry = r.y; m.rz = r.z; m.rw = r.w;
        m.hullLo = uint32_t(addrMesh & 0xffffffffu); m.hullHi = uint32_t(addrMesh >> 32);
        m.type = GEOM_TRIANGLEMESH;
        for (uint32_t t = 0; t < 2; ++t)
        {
            const V4 q = unitQ();
            float* x = xf.data() + size_t(2 * i + t) * 8;
            x[0] = q.x; x[1] = q.y; x[2] = q.z; x[3] = q.w;
            x[4] = sym(4); x[5] = sym(4); x[6] = sym(4);
        }
        // odd pairs put the mesh in slot 0, so the swap path runs the same helpers
        cms[i] = (i & 1u) ? CmInput{ 1 + i, 0, 2 * i + 1, 2 * i } : CmInput{ 0, 1 + i, 2 * i, 2 * i + 1 };
        pairs[i] = { i, i % 4u, 0, 0 };
    }

    const auto got = runSetupKernel(dev, shapes, xf, cdist, cms, pairBlob, pairs);

    size_t bad = 0, rodriguesDiffers = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const GpuShape& m = shapes[1 + i];
        const float* x0 = xf.data() + size_t(2 * i) * 8;   // the convex's transform
        const float* x1 = x0 + 8;                          // the mesh's
        const V4 q0 = { x0[0], x0[1], x0[2], x0[3] }, q1 = { x1[0], x1[1], x1[2], x1[3] };
        const V3 p0 = { x0[4], x0[5], x0[6] }, p1 = { x1[4], x1[5], x1[6] };
        const V3 sc = { m.sx, m.sy, m.sz };
        const V4 rq = { m.rx, m.ry, m.rz, m.rw };
        const TriRef lit = refTri(mesh, pairs[i].tri, sc, rq, q0, p0, q1, p1);
        gP94dRodrigues = true;
        const TriRef twin = refTri(mesh, pairs[i].tri, sc, rq, q0, p0, q1, p1);
        gP94dRodrigues = false;
        uint32_t a[12], b[12];
        refWords(lit, a); refWords(twin, b);
        bool differs = false;
        for (uint32_t k = 0; k < 12; ++k)
        {
            if (a[k] != b[k]) differs = true;
            const uint32_t g = u32(got[size_t(i) * 32 + kRefWordAt[k]]);
            if (g != a[k])
            {
                if (++bad <= 12)
                    std::printf("  [pair %u] word %u: GPU 0x%08x literal 0x%08x twin 0x%08x\n",
                                i, k, g, a[k], b[k]);
            }
        }
        if (differs) ++rodriguesDiffers;
    }
    std::printf("  cvxTriSetup_p94dLiteralRotate: %zu/%u pairs store a different word under the "
                "Rodrigues twin\n", rodriguesDiffers, N);
    PXG_CHECK_MSG(bad == 0, "p94d: %zu stored words differ from the literal PxQuat oracle", bad);
    PXG_CHECK_MSG(rodriguesDiffers > 0,
                  "p94d: the Rodrigues twin stores the same words on every pair -- the literal "
                  "formula is unobserved");
    dev->destroyBuffer(bMesh); dev->destroyBuffer(bHull);
}

} // namespace
