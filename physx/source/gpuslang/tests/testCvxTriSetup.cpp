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

static V3 quatRotate(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) + c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) + c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) + c.z * (w * 2.0f) };
}
static V3 quatRotateInv(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) - c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) - c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) - c.z * (w * 2.0f) };
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
static V3 normalized(V3 v)
{
    const float m = std::sqrt(dot3(v, v));
    return m > 0.0f ? V3{ v.x / m, v.y / m, v.z / m } : V3{ 0, 0, 0 };
}

static const uint32_t BV32_STRIDE = 160;
static const uint32_t GEOM_TRIANGLEMESH = 4;
static const uint32_t GEOM_CONVEXMESH = 5;

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

PXG_TEST(cvxTriSetupPrologue)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- two meshes: one with a positive scale, one the fixture will scale
    //      negatively, and different node counts so a mis-sized tree walk cannot
    //      accidentally agree between them ----------------------------------
    Mesh meshA, meshB;
    meshA.verts = { { 0, 0, 0, 0 }, { 2, 0, 0, 0 }, { 0, 3, 0, 0 }, { 1, 1, 4, 0 }, { -2, 1, 1, 0 } };
    meshA.idx = { 0, 1, 2, 0,   1, 3, 2, 0,   4, 0, 2, 0 };
    meshA.adj = { 11, 12, 13, 0,   0xffffffffu, 21, 0x80000000u, 0,   31, 32, 33, 0 };
    meshA.remap = { 700, 701, 702 };
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
    std::vector<GpuShape> shapes(4);
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
    setShape(shapes[1], { 1.0f, 1.0f, 1.0f }, { 0, 0, 0, 1 }, addrMeshA, GEOM_TRIANGLEMESH);
    setShape(shapes[2], { 1.5f, -0.7f, 1.2f }, { 0, 0, 0, 1 }, addrMeshB, GEOM_TRIANGLEMESH);
    {
        const float a = 0.4f;
        setShape(shapes[3], { 0.9f, 1.4f, 0.6f },
                 { 0.0f, std::sin(a * 0.5f), 0.0f, std::cos(a * 0.5f) }, addrHull, GEOM_CONVEXMESH);
    }

    // ---- transforms (PxsCachedTransform: quat @0, pos @16, 32 B) -----------
    std::vector<float> xf(4 * 8, 0.0f);
    auto setXf = [&](uint32_t i, V4 q, V3 p) {
        float* r = xf.data() + i * 8;
        r[0] = q.x; r[1] = q.y; r[2] = q.z; r[3] = q.w;
        r[4] = p.x; r[5] = p.y; r[6] = p.z;
    };
    setXf(0, { 0, 0, 0, 1 }, { 0.5f, -0.25f, 1.0f });
    {
        const float a = 0.7f;
        setXf(1, { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, { -1.0f, 2.0f, 0.5f });
    }
    {
        const float a = -1.1f;
        setXf(2, { 0.0f, 0.0f, std::sin(a * 0.5f), std::cos(a * 0.5f) }, { 3.0f, 0.0f, -1.5f });
    }
    setXf(3, { 0, 0, 0, 1 }, { 0.0f, 0.0f, 0.0f });

    std::vector<float> cdist = { 0.02f, 0.031f, 0.047f, 0.013f };

    // ---- contact managers --------------------------------------------------
    // 0: convex first.  1: MESH FIRST, so the swap has to fire.  2: the negative
    // scale.  3: the rotated convex.
    std::vector<CmInput> cms = {
        { 0, 1, 0, 1 },
        { 1, 0, 1, 0 },   // shapeRef0 IS the mesh -> swap
        { 0, 2, 0, 2 },
        { 3, 1, 3, 1 },
    };

    // ---- ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40 -------
    // Deliberately different values so a port reading `count`/`cmIndex` instead
    // (the offsets the declaration's trailing comment numbers suggest) is caught.
    std::vector<uint8_t> pairBlob(cms.size() * 56, 0);
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

    // ---- the pairs to run --------------------------------------------------
    struct Pair { uint32_t cm, tri, off, pad; };
    std::vector<Pair> pairs = {
        { 0, 0, 0, 0 },
        { 0, 2, 3, 0 },     // a different triangle of the same mesh
        { 1, 1, 1, 0 },     // the swapped pair
        { 2, 1, 2, 0 },     // the negative determinant
        { 3, 1, 4, 0 },     // a rotated convex scale
    };

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

    const auto got = downloadBuffer<float>(dev, bOut, outInit.size());

    size_t bad = 0;
    size_t sawSwap = 0, sawFlip = 0, sawBoundaryAdj = 0, sawRotatedScale = 0;
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
        const V4 toCvxQ = quatMul(quatConj(q0), q1);
        const V3 toCvxP = quatRotateInv(q0, sub(p1, p0));

        const V3 triScale = { tri.sx, tri.sy, tri.sz };
        const V4 triRot = { tri.rx, tri.ry, tri.rz, tri.rw };
        const bool flip = (triScale.x * triScale.y * triScale.z) < 0.0f;
        if (flip) ++sawFlip;
        if ((cvx.rx != 0.0f) || (cvx.ry != 0.0f) || (cvx.rz != 0.0f)) ++sawRotatedScale;

        V3 loc[3];
        for (uint32_t k = 0; k < 3; ++k)
        {
            const uint32_t vi = mesh.idx[4 * pr.tri + k];
            const V4 v4 = mesh.verts[vi];
            const V3 v = { v4.x, v4.y, v4.z };
            loc[k] = add(quatRotate(toCvxQ, vertex2Shape(v, triScale, triRot)), toCvxP);
        }
        const V3 e0 = loc[0];
        const V3 e1 = flip ? loc[2] : loc[1];
        const V3 e2 = flip ? loc[1] : loc[2];
        const V3 en = normalized(cross3(sub(e1, e0), sub(e2, e0)));

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
    dev->destroyBuffer(bShapes); dev->destroyBuffer(bXf); dev->destroyBuffer(bCd);
    dev->destroyBuffer(bCm); dev->destroyBuffer(bPairBlob); dev->destroyBuffer(bPairs);
    dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(bad == 0, "cvxTriSetup: %zu mismatches", bad);
    PXG_CHECK_MSG(sawSwap > 0, "no case has the mesh in slot 0 (the swap is untested)");
    PXG_CHECK_MSG(sawFlip > 0, "no case has a negative scale determinant (the winding flip is untested)");
    PXG_CHECK_MSG(sawBoundaryAdj > 0, "no case carries a BOUNDARY adjacency word");
    PXG_CHECK_MSG(sawRotatedScale > 0, "no case has a non-identity convex scale rotation");
}

} // namespace
