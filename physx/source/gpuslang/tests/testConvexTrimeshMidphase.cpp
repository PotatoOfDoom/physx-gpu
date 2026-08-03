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

// P8.27 — midphaseGeneratePairs (convexTrimeshMidphaseBda.spv) on the engine ABI.
//
// Suites (substring-matched by the "convexTrimeshMidphase" ctest filter):
//   convexTrimeshMidphasePairs  — the emitted triangle SET, the two counters and the
//                                 ConvexMeshPair record, vs a CPU transcription
//   convexTrimeshMidphaseCached — the PCM gate says "cached": no traversal, no pairs,
//                                 and count == CONVEX_TRIMESH_CACHED (-1)
//   convexTrimeshMidphaseCull   — the front-face gate, isolated on down-facing triangles
//
// The second and third fixtures exist because the first cannot see them.  A port
// that ignored the gate entirely would pass `Pairs` — it emits the same triangles
// when fullContactGen is true, which it always would be.  And a port that dropped
// the plane-distance cull would also pass `Pairs`, because every triangle there
// faces the box.  Each of those two failure modes gets a scene that only it fails.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct V3 { float x, y, z; };
static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

// --- the 112-byte engine blob, byte-identical to MidphaseParams -------------
struct MidphasePush
{
    uint32_t numContactManagers; float toleranceLength;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes;
    uint64_t cvxTrimeshPair, multiManifolds, cmOutputs, stackPtr, stackOffset;
    uint64_t midphasePairsNum, midphasePairsNumPadded;
    uint32_t stackSizeBytes, pad108;
};
static_assert(sizeof(MidphasePush) == 112, "midphase blob must be 112 bytes");
static_assert(offsetof(MidphasePush, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(MidphasePush, stackSizeBytes) == 104, "stackSizeBytes @104");

// --- PxgShape, 48 bytes (PxgConvexConvexShape.h) -----------------------------
struct ShapeHost
{
    float scaleX, scaleY, scaleZ;                      // 0
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW;  // 12
    uint32_t materialIndex;                            // 28
    uint32_t hullLo, hullHi;                           // 32
    uint32_t type;                                     // 40
    uint32_t particleOrSoftbodyId;                     // 44
};
static_assert(sizeof(ShapeHost) == 48, "PxgShape is 48 bytes");

struct CmInputHost { uint32_t shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; };

static const uint32_t GEOM_BOX = 3u, GEOM_TRIANGLEMESH = 8u;

// --- Gu::intersectTriangleBox_RefImpl<true>, CPU transcription ---------------
// Transcribed from pxgSlangTriBox (P8.24), which was itself proven opcode-identical
// to the Phase-1 harness copy.  Built with the same asymmetric plane test.
static bool planeBoxOverlap(V3 n, float d, V3 maxbox)
{
    V3 vmin, vmax;
    vmin.x = n.x > 0.0f ? -maxbox.x : maxbox.x;  vmax.x = n.x > 0.0f ? maxbox.x : -maxbox.x;
    vmin.y = n.y > 0.0f ? -maxbox.y : maxbox.y;  vmax.y = n.y > 0.0f ? maxbox.y : -maxbox.y;
    vmin.z = n.z > 0.0f ? -maxbox.z : maxbox.z;  vmax.z = n.z > 0.0f ? maxbox.z : -maxbox.z;
    if (dot3(n, vmin) + d > 0.0f) return false;      // near corner: strictly greater
    if (dot3(n, vmax) + d >= 0.0f) return true;      // far corner: greater OR EQUAL
    return false;
}

static bool intersectTriangleBox(V3 c, V3 e, V3 tp0, V3 tp1, V3 tp2)
{
    const V3 v0 = sub3(tp0, c), v1 = sub3(tp1, c), v2 = sub3(tp2, c);
    auto inBox = [&](V3 v) {
        return std::fabs(v.x) <= e.x && std::fabs(v.y) <= e.y && std::fabs(v.z) <= e.z;
    };
    if (inBox(v0) || inBox(v1) || inBox(v2)) return true;

    const V3 ed[3] = { sub3(v1, v0), sub3(v2, v1), sub3(v0, v2) };
    const V3 vv[3] = { v0, v1, v2 };
    // The nine edge-cross axes, in the CUDA's evaluation order.  Each AXISTEST uses a
    // specific PAIR of the three vertices; the third is the one the edge starts from.
    struct T { int e, a, b, ax; };
    const T tests[9] = {
        {0,0,2,0},{0,0,2,1},{0,1,2,2},
        {1,0,2,0},{1,0,2,1},{1,0,1,2},
        {2,0,1,0},{2,0,1,1},{2,1,2,2},
    };
    for (const T& t : tests)
    {
        const V3 ee = ed[t.e];
        const float fex = std::fabs(ee.x), fey = std::fabs(ee.y), fez = std::fabs(ee.z);
        float pa, pb, rad;
        if (t.ax == 0) {           // X: p = e.z*v.y - e.y*v.z
            pa = ee.z * vv[t.a].y - ee.y * vv[t.a].z;
            pb = ee.z * vv[t.b].y - ee.y * vv[t.b].z;
            rad = fez * e.y + fey * e.z;
        } else if (t.ax == 1) {    // Y: p = -e.z*v.x + e.x*v.z
            pa = -ee.z * vv[t.a].x + ee.x * vv[t.a].z;
            pb = -ee.z * vv[t.b].x + ee.x * vv[t.b].z;
            rad = fez * e.x + fex * e.z;
        } else {                   // Z: p = e.y*v.x - e.x*v.y
            pa = ee.y * vv[t.a].x - ee.x * vv[t.a].y;
            pb = ee.y * vv[t.b].x - ee.x * vv[t.b].y;
            rad = fey * e.x + fex * e.y;
        }
        const float mn = std::fmin(pa, pb), mx = std::fmax(pa, pb);
        if (mn > rad || mx < -rad) return false;
    }
    // the three AABB axes
    if (std::fmin(std::fmin(v0.x, v1.x), v2.x) > e.x || std::fmax(std::fmax(v0.x, v1.x), v2.x) < -e.x) return false;
    if (std::fmin(std::fmin(v0.y, v1.y), v2.y) > e.y || std::fmax(std::fmax(v0.y, v1.y), v2.y) < -e.y) return false;
    if (std::fmin(std::fmin(v0.z, v1.z), v2.z) > e.z || std::fmax(std::fmax(v0.z, v1.z), v2.z) < -e.z) return false;
    // the triangle plane
    const V3 n = cross3(ed[0], ed[1]);
    return planeBoxOverlap(n, -dot3(n, v0), e);
}

// --- the scene the fixtures share -------------------------------------------
// A unit box (half-extent 0.5) hovering at y = 0.4 over a mesh of triangles laid in
// the plane y = 0.  Everything is identity-transformed and unit-scaled, so mesh
// VERTEX space, mesh LOCAL space and WORLD space coincide — which is deliberate:
// it keeps this fixture about the traversal, the gate and the emission, and leaves
// the scale/rotation plumbing to the engine-path scenes.
struct Scene
{
    std::vector<V3> verts;
    std::vector<uint32_t> tri;   // 3 indices per triangle
    std::vector<uint8_t> up;     // 1 = wound so the normal points +y
};

static Scene makeScene(bool includeDownFacing)
{
    Scene s;
    // A 3 x 3 grid of quads over [-1.5, 1.5]^2 in x/z at y = 0, two triangles each
    // = 18 triangles.  The count matters: a BV32 leaf is walked with ONE LANE PER
    // PRIMITIVE, so anything past 32 in a single leaf is never visited at all.  The
    // first version of this fixture used a 4 x 4 grid (34 triangles) and the two
    // back-facing triangles below sat at indices 32 and 33 -- they were dropped by
    // the lane mapping, not by the front-face gate, and a mutant that DELETED THE
    // GATE survived because of it.
    const int N = 4;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            s.verts.push_back(V3{ -1.5f + float(i), 0.0f, -1.5f + float(j) });
    auto vid = [&](int i, int j) { return uint32_t(i * N + j); };
    for (int i = 0; i + 1 < N; ++i)
        for (int j = 0; j + 1 < N; ++j)
        {
            // wound so cross(v1-v0, v2-v0) points +y
            s.tri.push_back(vid(i, j));     s.tri.push_back(vid(i, j + 1));     s.tri.push_back(vid(i + 1, j));
            s.up.push_back(1);
            s.tri.push_back(vid(i + 1, j)); s.tri.push_back(vid(i, j + 1));     s.tri.push_back(vid(i + 1, j + 1));
            s.up.push_back(1);
        }
    if (includeDownFacing)
    {
        // Two triangles directly under the box but wound the OTHER way, so their
        // plane distance to the box centre of mass is negative and the front-face
        // gate must reject them even though they overlap the query box.
        const uint32_t base = uint32_t(s.verts.size());
        s.verts.push_back(V3{ -0.3f, 0.0f, -0.3f });
        s.verts.push_back(V3{  0.3f, 0.0f, -0.3f });
        s.verts.push_back(V3{ -0.3f, 0.0f,  0.3f });
        s.tri.push_back(base); s.tri.push_back(base + 1); s.tri.push_back(base + 2);
        s.up.push_back(0);
        s.verts.push_back(V3{ 0.3f, 0.0f, 0.3f });
        s.tri.push_back(base + 3); s.tri.push_back(base + 2); s.tri.push_back(base + 1);
        s.up.push_back(0);
    }
    // A triangle ABOVE the box, separated from its face by 0.02 -- inside the
    // 0.54 half-extent the contact distance produces, outside the 0.50 the box
    // alone gives.  Wound DOWN-facing so the front-face gate keeps it (its normal
    // must point back toward the centre of mass, which is below it).
    {
        const uint32_t base = uint32_t(s.verts.size());
        const float y = 0.92f;   // box centre 0.4 + half 0.5 + 0.02
        s.verts.push_back(V3{ -0.2f, y, -0.2f });
        s.verts.push_back(V3{  0.2f, y, -0.2f });
        s.verts.push_back(V3{ -0.2f, y,  0.2f });
        s.tri.push_back(base); s.tri.push_back(base + 1); s.tri.push_back(base + 2);
        s.up.push_back(0);
    }
    return s;
}

// Gu::BV32DataPacked: a root with THREE leaf children, each boxed TIGHTLY around
// its own primitive range.
//
// The first version used ONE child with a [-1000, 1000] box, and two mutants
// survived because of it.  A node box that CONTAINS the query box makes the
// overlap test agree with a swapped-operand version of itself, so a mutant that
// exchanged the query box's min and max was unobservable; and with a single leaf
// the atomic base offset is always zero, so a mutant that wrote pairs at the bare
// lane offset was unobservable too.  Tight boxes across three leaves fix both:
// the boxes are SMALLER than the query box, and the second and third leaves write
// at a non-zero base.
struct Child { uint32_t primStart, nbPrims; V3 lo, hi; };

static std::vector<uint32_t> makeBv32Root(const std::vector<Child>& kids)
{
    std::vector<uint32_t> w(292, 0u);     // 1168 bytes
    for (size_t k = 0; k < kids.size(); ++k)
    {
        w[4*k + 0] = f2u(kids[k].lo.x); w[4*k + 1] = f2u(kids[k].lo.y); w[4*k + 2] = f2u(kids[k].lo.z);
        w[128 + 4*k + 0] = f2u(kids[k].hi.x);
        w[128 + 4*k + 1] = f2u(kids[k].hi.y);
        w[128 + 4*k + 2] = f2u(kids[k].hi.z);
        w[256 + k] = (kids[k].primStart << 7) | (kids[k].nbPrims << 1) | 1u;   // leaf
    }
    w[288] = uint32_t(kids.size());       // mNbNodes @1152
    return w;
}

// Split the triangles into three contiguous ranges and box each one tightly.
static std::vector<Child> splitIntoChildren(const Scene& sc)
{
    const uint32_t n = uint32_t(sc.tri.size() / 3);
    const uint32_t cut0 = n / 3, cut1 = (2 * n) / 3;
    const uint32_t starts[3] = { 0u, cut0, cut1 };
    const uint32_t ends[3]   = { cut0, cut1, n };
    std::vector<Child> kids;
    for (int c = 0; c < 3; ++c)
    {
        Child k{}; k.primStart = starts[c]; k.nbPrims = ends[c] - starts[c];
        k.lo = V3{ 1e30f, 1e30f, 1e30f }; k.hi = V3{ -1e30f, -1e30f, -1e30f };
        for (uint32_t t = starts[c]; t < ends[c]; ++t)
            for (int v = 0; v < 3; ++v)
            {
                const V3 q = sc.verts[sc.tri[3*t + v]];
                k.lo.x = std::fmin(k.lo.x, q.x); k.lo.y = std::fmin(k.lo.y, q.y); k.lo.z = std::fmin(k.lo.z, q.z);
                k.hi.x = std::fmax(k.hi.x, q.x); k.hi.y = std::fmax(k.hi.y, q.y); k.hi.z = std::fmax(k.hi.z, q.z);
            }
        kids.push_back(k);
    }
    return kids;
}

struct Built
{
    DevBuf mesh, hull, cmIn, xf, bnd, cdist, shapes;
    DevBuf pair, manifold, stack, stackOff, pairsNum, pairsNumPadded;
    uint32_t nbTris;
};

// boxY: the box centre height.  cached: build the manifold so the gate reports
// "the cached manifold stands" instead of asking for a full contact gen.
static Built build(PxgSlangComputeDevice* dev, const Scene& sc, float boxY, bool cached)
{
    Built b;
    b.nbTris = uint32_t(sc.tri.size() / 3);

    // --- the cooked-mesh blob: uint4 header, one BV32 node, float4 verts, uint4 idx
    std::vector<uint32_t> blob;
    blob.push_back(uint32_t(sc.verts.size()));  // .x nbVerts
    blob.push_back(b.nbTris);                   // .y nbTris
    blob.push_back(1u);                         // .z (a different node count; unused here)
    blob.push_back(1u);                         // .w nbPackedNodes -> sizes the node array
    { const auto n = makeBv32Root(splitIntoChildren(sc)); blob.insert(blob.end(), n.begin(), n.end()); }
    for (const V3& v : sc.verts)
    { blob.push_back(f2u(v.x)); blob.push_back(f2u(v.y)); blob.push_back(f2u(v.z)); blob.push_back(0u); }
    for (uint32_t t = 0; t < b.nbTris; ++t)
    { blob.push_back(sc.tri[3*t]); blob.push_back(sc.tri[3*t+1]); blob.push_back(sc.tri[3*t+2]); blob.push_back(0u); }
    b.mesh = makeDev(dev, blob);

    // --- the convex hull blob: CoM at 0, internal-AABB extents at 32 ----------
    std::vector<float> hull(12, 0.0f);
    hull[0] = 0.0f; hull[1] = 0.0f; hull[2] = 0.0f; hull[3] = 0.0f;   // CoM
    hull[8] = 0.5f; hull[9] = 0.5f; hull[10] = 0.5f; hull[11] = 0.0f; // extents @32
    b.hull = makeDev(dev, hull);

    // Two managers describing the SAME pair, the second with the shape order
    // REVERSED.  midphaseCore swaps slot 0 and slot 1 when shape 0 is the mesh, and
    // with only the already-correct ordering present that swap never fires -- a
    // mutant that deleted it survived.  Both managers must emit the same set.
    std::vector<CmInputHost> cmIn = { { 0u, 1u, 0u, 1u }, { 1u, 0u, 1u, 0u } };
    b.cmIn = makeDev(dev, cmIn);

    // PxsCachedTransform, 32 B: quat @0, pos @16.
    std::vector<float> xf = { 0,0,0,1,  0,boxY,0,0,      // the box
                              0,0,0,1,  0,0,0,0 };       // the mesh
    b.xf = makeDev(dev, xf);

    // PxBounds3, 24 B each: the box's world AABB, then the mesh's (unused).
    std::vector<float> bnd = { -0.5f, boxY - 0.5f, -0.5f,  0.5f, boxY + 0.5f, 0.5f,
                               -2.0f, -0.01f, -2.0f,       2.0f, 0.01f, 2.0f };
    b.bnd = makeDev(dev, bnd);

    std::vector<float> cd = { 0.02f, 0.02f };
    b.cdist = makeDev(dev, cd);

    ShapeHost s0{}, s1{};
    s0.scaleX = s0.scaleY = s0.scaleZ = 0.5f;      // box half-extents live in scale
    s0.scaleRotW = 1.0f; s0.materialIndex = 7u; s0.type = GEOM_BOX;
    s0.particleOrSoftbodyId = 0xFFFFFFFFu;
    s0.hullLo = uint32_t(b.hull.addr & 0xFFFFFFFFull);
    s0.hullHi = uint32_t(b.hull.addr >> 32);
    s1.scaleX = s1.scaleY = s1.scaleZ = 1.0f;
    s1.scaleRotW = 1.0f; s1.materialIndex = 11u; s1.type = GEOM_TRIANGLEMESH;
    s1.particleOrSoftbodyId = 0xFFFFFFFFu;
    s1.hullLo = uint32_t(b.mesh.addr & 0xFFFFFFFFull);
    s1.hullHi = uint32_t(b.mesh.addr >> 32);
    std::vector<ShapeHost> shapes = { s0, s1 };
    b.shapes = makeDev(dev, shapes);

    // --- the persistent manifold ---------------------------------------------
    std::vector<uint8_t> man(1216, 0u);
    auto putf = [&](size_t off, float v) { std::memcpy(man.data() + off, &v, 4); };
    auto putu = [&](size_t off, uint32_t v) { std::memcpy(man.data() + off, &v, 4); };
    if (cached)
    {
        // mRelativeTransform EXACTLY equals aToB (mesh is identity, so aToB is the
        // box transform), and zero manifolds so the refresh finds nothing.  Both
        // gates then report false and the kernel must skip the traversal.
        // deltaP is 0.005 — UNDER the 0.01 threshold that minMargin*ratio gives
        // (minMargin = min(0.5*0.25, 1.0*0.05) = 0.05, ratio 0.2).  Chosen rather
        // than an exact match so the margin itself is load-bearing: a port that
        // read the hull extents from the wrong offset gets minMargin 0, threshold
        // 0, and invalidates on the same 0.005.
        putf(1168, 0.0f); putf(1172, 0.0f); putf(1176, 0.0f); putf(1180, 1.0f);  // q
        putf(1184, 0.0f); putf(1188, boxY - 0.005f); putf(1192, 0.0f); putf(1196, 0.0f);
        putu(1200, 0u);
    }
    else
    {
        // A stored transform far from aToB: deltaP is enormous, so invalidate fires.
        putf(1168, 0.0f); putf(1172, 0.0f); putf(1176, 0.0f); putf(1180, 1.0f);
        putf(1184, 100.0f); putf(1188, 100.0f); putf(1192, 100.0f); putf(1196, 0.0f);
        putu(1200, 0u);
    }
    // ONE RECORD PER CONTACT MANAGER.  The kernel indexes multiManifolds by cmIdx,
    // so a single-record buffer sends manager 1 reading past the end -- which showed
    // up as manager 1 invalidating on garbage while manager 0 stayed cached.
    std::vector<uint8_t> man2(man);
    man2.insert(man2.end(), man.begin(), man.end());
    b.manifold = makeDev(dev, man2);

    std::vector<uint8_t> pair(2 * 56, 0xEE);
    b.pair = makeDev(dev, pair);
    std::vector<uint32_t> stack(4u * 1024u, 0xFFFFFFFFu);
    b.stack = makeDev(dev, stack);
    std::vector<uint32_t> zero1 = { 0u };
    b.stackOff = makeDev(dev, zero1);
    b.pairsNum = makeDev(dev, zero1);
    b.pairsNumPadded = makeDev(dev, zero1);
    return b;
}

static void destroy(PxgSlangComputeDevice* dev, Built& b)
{
    for (PxgBufferHandle h : { b.mesh.h, b.hull.h, b.cmIn.h, b.xf.h, b.bnd.h, b.cdist.h,
                               b.shapes.h, b.pair.h, b.manifold.h, b.stack.h,
                               b.stackOff.h, b.pairsNum.h, b.pairsNumPadded.h })
        dev->destroyBuffer(h);
}

// The CPU model of what the kernel must emit.
static std::vector<uint32_t> expectedTriangles(const Scene& sc, float boxY)
{
    const float cd = 0.04f;                               // 0.02 + 0.02
    const V3 center{ 0.0f, boxY, 0.0f };
    const V3 ext{ 0.5f + cd, 0.5f + cd, 0.5f + cd };
    const V3 com{ 0.0f, boxY, 0.0f };                     // hull CoM at the box origin
    std::vector<uint32_t> out;
    for (uint32_t t = 0; t < sc.tri.size() / 3; ++t)
    {
        const V3 a = sc.verts[sc.tri[3*t]], b2 = sc.verts[sc.tri[3*t+1]], c = sc.verts[sc.tri[3*t+2]];
        const V3 raw = cross3(sub3(b2, a), sub3(c, a));
        const float m = dot3(raw, raw);
        const V3 n = m > 0.0f ? V3{ raw.x / std::sqrt(m), raw.y / std::sqrt(m), raw.z / std::sqrt(m) }
                              : V3{ 0, 0, 0 };
        const float d = -dot3(n, a);
        if (dot3(n, com) + d < 0.0f) continue;            // the front-face gate
        if (intersectTriangleBox(center, ext, a, b2, c)) out.push_back(t);
    }
    return out;
}

static void dispatchMidphase(PxgSlangComputeDevice* dev, Built& b)
{
    MidphasePush p{};
    p.numContactManagers = 2u;
    p.toleranceLength = 1.0f;
    p.cmInputs = b.cmIn.addr; p.transformCache = b.xf.addr; p.bounds = b.bnd.addr;
    p.contactDistance = b.cdist.addr; p.gpuShapes = b.shapes.addr;
    p.cvxTrimeshPair = b.pair.addr; p.multiManifolds = b.manifold.addr;
    p.cmOutputs = 0ull;                                   // never touched, see the shader header
    p.stackPtr = b.stack.addr; p.stackOffset = b.stackOff.addr;
    p.midphasePairsNum = b.pairsNum.addr; p.midphasePairsNumPadded = b.pairsNumPadded.addr;
    p.stackSizeBytes = 16u * 1024u;

    // grid ((numTests+1)/2, 1, 1) mirrors PxgNarrowphaseCore.cpp:1973.
    Kernel k("convexTrimeshMidphaseBda.spv", "midphaseGeneratePairs", {}, sizeof(MidphasePush));
    k.dispatch(1u, {}, p, 1u, 1u);
    dev->flush();
}

// ---------------------------------------------------------------------------
PXG_TEST(convexTrimeshMidphasePairs)
{
    PxgSlangComputeDevice* dev = testDevice();
    const Scene sc = makeScene(/*includeDownFacing*/ false);
    const float boxY = 0.4f;

    Built b = build(dev, sc, boxY, /*cached*/ false);
    dispatchMidphase(dev, b);

    const auto exp = expectedTriangles(sc, boxY);
    const uint32_t total = 2u * uint32_t(exp.size());   // two managers, same pair
    const auto gotNum = downloadBuffer<uint32_t>(dev, b.pairsNum.h, 1);
    const auto gotOff = downloadBuffer<uint32_t>(dev, b.stackOff.h, 1);
    const auto gotStack = downloadBuffer<uint32_t>(dev, b.stack.h, 4u * (total + 4u));
    const auto gotPair = downloadBuffer<uint8_t>(dev, b.pair.h, 112);

    size_t mm = 0;
    if (gotNum[0] != total) { ++mm; std::printf("  pairsNum got=%u exp=%u\n", gotNum[0], total); }
    if (gotOff[0] != total) { ++mm; std::printf("  stackOffset got=%u exp=%u\n", gotOff[0], total); }

    // The emitted set PER MANAGER, order-insensitive: the slot a pair lands in
    // comes from a device-wide atomic and is nondeterministic on CUDA too.  Both
    // managers describe the same pair with the shape order reversed, so both must
    // produce the identical triangle set.
    std::vector<uint32_t> perCm[2];
    for (uint32_t i = 0; i < total; ++i)
    {
        const uint32_t cm = gotStack[4*i + 0];
        if (cm > 1u) { ++mm; std::printf("  pair %u: cmIdx got=%u, out of range\n", i, cm); continue; }
        perCm[cm].push_back(gotStack[4*i + 1]);
    }
    for (int cm = 0; cm < 2; ++cm)
    {
        std::sort(perCm[cm].begin(), perCm[cm].end());
        if (perCm[cm] != exp)
        {
            ++mm;
            std::printf("  cm %d: emitted set differs (got %zu, exp %zu)\n",
                        cm, perCm[cm].size(), exp.size());
            for (size_t i = 0; i < perCm[cm].size() && i < 8; ++i)
                std::printf("    got[%zu]=%u\n", i, perCm[cm][i]);
            for (size_t i = 0; i < exp.size() && i < 8; ++i)
                std::printf("    exp[%zu]=%u\n", i, exp[i]);
        }
    }

    // The ConvexMeshPair record: aToB, count, cmIndex, materials.
    auto pf = [&](size_t o) { float v; std::memcpy(&v, gotPair.data() + o, 4); return v; };
    auto pi = [&](size_t o) { int32_t v; std::memcpy(&v, gotPair.data() + o, 4); return v; };
    auto pu = [&](size_t o) { uint32_t v; std::memcpy(&v, gotPair.data() + o, 4); return v; };
    if (pf(0) != 0.0f || pf(4) != 0.0f || pf(8) != 0.0f || pf(12) != 1.0f)
    { ++mm; std::printf("  record aToB.q wrong\n"); }
    if (pf(16) != 0.0f || pf(20) != boxY || pf(24) != 0.0f)
    { ++mm; std::printf("  record aToB.p got=(%.4f,%.4f,%.4f)\n", pf(16), pf(20), pf(24)); }
    if (pi(32) != int32_t(exp.size()))
    { ++mm; std::printf("  record count got=%d exp=%zu\n", pi(32), exp.size()); }
    if (pi(36) != 0) { ++mm; std::printf("  record cmIndex got=%d exp=0\n", pi(36)); }
    if (pi(28) != 0) { ++mm; std::printf("  record startIndex got=%d exp=0\n", pi(28)); }
    if (pu(48) != 7u || pu(52) != 11u)
    { ++mm; std::printf("  record materials got=(%u,%u) exp=(7,11)\n", pu(48), pu(52)); }

    PXG_CHECK_MSG(mm == 0, "midphaseGeneratePairs matches CPU (%zu mismatches, %zu triangles)",
                  mm, exp.size());
    destroy(dev, b);
}

// ---------------------------------------------------------------------------
// The PCM gate integrated: a record whose stored relative transform already
// matches aToB and which holds no manifolds.  invalidateManifold is false,
// refreshManifolds finds nothing, so fullContactGen is false — no traversal, no
// pairs, and the record carries CONVEX_TRIMESH_CACHED instead of a count.
// ---------------------------------------------------------------------------
PXG_TEST(convexTrimeshMidphaseCached)
{
    PxgSlangComputeDevice* dev = testDevice();
    const Scene sc = makeScene(false);
    const float boxY = 0.4f;

    Built b = build(dev, sc, boxY, /*cached*/ true);
    dispatchMidphase(dev, b);

    const auto gotNum = downloadBuffer<uint32_t>(dev, b.pairsNum.h, 1);
    const auto gotOff = downloadBuffer<uint32_t>(dev, b.stackOff.h, 1);
    const auto gotPair = downloadBuffer<uint8_t>(dev, b.pair.h, 112);

    size_t mm = 0;
    if (gotNum[0] != 0u) { ++mm; std::printf("  cached: pairsNum got=%u exp=0\n", gotNum[0]); }
    if (gotOff[0] != 0u) { ++mm; std::printf("  cached: stackOffset got=%u exp=0\n", gotOff[0]); }
    for (int cm = 0; cm < 2; ++cm)
    {
        int32_t count; std::memcpy(&count, gotPair.data() + 56 * cm + 32, 4);
        if (count != -1) { ++mm; std::printf("  cached: cm %d count got=%d exp=-1\n", cm, count); }
    }

    // Sanity: the SAME scene with the gate open emits a non-empty set, so this
    // fixture is measuring the gate and not an empty query box.
    const auto exp = expectedTriangles(sc, boxY);
    if (exp.empty()) { ++mm; std::printf("  cached: the control set is empty — fixture is vacuous\n"); }

    PXG_CHECK_MSG(mm == 0, "the PCM gate suppresses the traversal (%zu mismatches)", mm);
    destroy(dev, b);
}

// ---------------------------------------------------------------------------
// The front-face gate, isolated.  Two extra triangles sit directly under the box
// and overlap the query box, but are wound so their plane normal points AWAY from
// the convex's centre of mass.  They must not be emitted.
// ---------------------------------------------------------------------------
PXG_TEST(convexTrimeshMidphaseCull)
{
    PxgSlangComputeDevice* dev = testDevice();
    const Scene sc = makeScene(/*includeDownFacing*/ true);
    const float boxY = 0.4f;

    Built b = build(dev, sc, boxY, false);
    dispatchMidphase(dev, b);

    const auto exp = expectedTriangles(sc, boxY);
    const uint32_t total = 2u * uint32_t(exp.size());
    const auto gotNum = downloadBuffer<uint32_t>(dev, b.pairsNum.h, 1);
    const auto gotStack = downloadBuffer<uint32_t>(dev, b.stack.h, 4u * (total + 4u));

    size_t mm = 0;
    if (gotNum[0] != total) { ++mm; std::printf("  cull: pairsNum got=%u exp=%u\n", gotNum[0], total); }

    std::vector<uint32_t> gotTris;
    for (uint32_t i = 0; i < total; ++i) gotTris.push_back(gotStack[4*i + 1]);
    std::sort(gotTris.begin(), gotTris.end());
    std::vector<uint32_t> expTwice;
    for (int r = 0; r < 2; ++r) expTwice.insert(expTwice.end(), exp.begin(), exp.end());
    std::sort(expTwice.begin(), expTwice.end());
    if (gotTris != expTwice) { ++mm; std::printf("  cull: emitted set differs\n"); }

    // The two down-facing triangles are the LAST two, and the CPU model must
    // exclude both — if it did not, this fixture would prove nothing.
    // The two back-facing grid triangles are at nbTris-3 and nbTris-2; the last
    // triangle is the near-boundary one ABOVE the box, which is also down-facing
    // but whose normal DOES point at the centre of mass, so it must be kept.
    const uint32_t nbTris = uint32_t(sc.tri.size() / 3);
    for (uint32_t t = nbTris - 3; t < nbTris - 1; ++t)
    {
        if (std::find(exp.begin(), exp.end(), t) != exp.end())
        { ++mm; std::printf("  cull: CPU model kept down-facing triangle %u — fixture is vacuous\n", t); }
        if (std::find(gotTris.begin(), gotTris.end(), t) != gotTris.end())
        { ++mm; std::printf("  cull: shader emitted down-facing triangle %u\n", t); }
    }

    PXG_CHECK_MSG(mm == 0, "the front-face gate rejects back-facing triangles (%zu mismatches)", mm);
    destroy(dev, b);
}

} // namespace
