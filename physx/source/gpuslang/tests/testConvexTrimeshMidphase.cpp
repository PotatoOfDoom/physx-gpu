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
//   convexTrimeshMidphaseQueryBox — P9.94: a rotated, non-uniformly scaled mesh whose
//                                 BV32 children sit ON and one ULP OUTSIDE each face of
//                                 the vertex-space query box, so the emitted set pins
//                                 every face bit of PxBounds3::transformFast
//
// The second and third fixtures exist because the first cannot see them.  A port
// that ignored the gate entirely would pass `Pairs` — it emits the same triangles
// when fullContactGen is true, which it always would be.  And a port that dropped
// the plane-distance cull would also pass `Pairs`, because every triangle there
// faces the box.  Each of those two failure modes gets a scene that only it fails.

#include "pxgTestFramework.h"
#include "pxgRefMath.h"

#include "geometry/PxGeometry.h"

#include <algorithm>
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
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

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

// P9.57: FROM THE ENGINE ENUM, never a literal.  `PxGeometryType` grew
// eCONVEXCORE at position 4, which shifted eCONVEXMESH to 5, eTRIANGLEMESH to 8
// and eHEIGHTFIELD to 9; three fixtures and two shaders still carried the old
// numbers, and because a fixture and its shader carried the SAME wrong number
// the differential could not see it.  Taking the value from the header makes
// the compiler the oracle, exactly as this suite already does for the BV32
// stride.  tools/check_geom_type_constants.py keeps the Slang side honest.
static const uint32_t GEOM_BOX = uint32_t(physx::PxGeometryType::eBOX), GEOM_TRIANGLEMESH = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);

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

// P9.86f: AXISTEST_* / FINDMINMAX are intrinsics::selectMin/selectMax (GuIntersectionTriangleBoxRef.h:58-61,
// :108-156; PxUnixMathIntrinsics.h:116-125): a NaN SECOND operand survives.  std::fmin/std::fmax drop it.
static float selMin(float a, float b) { return (a < b) ? a : b; }
static float selMax(float a, float b) { return (a > b) ? a : b; }
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
        const float mn = selMin(pa, pb), mx = selMax(pa, pb);
        if (mn > rad || mx < -rad) return false;
    }
    // the three AABB axes
    if (selMin(selMin(v0.x, v1.x), v2.x) > e.x || selMax(selMax(v0.x, v1.x), v2.x) < -e.x) return false;
    if (selMin(selMin(v0.y, v1.y), v2.y) > e.y || selMax(selMax(v0.y, v1.y), v2.y) < -e.y) return false;
    if (selMin(selMin(v0.z, v1.z), v2.z) > e.z || selMax(selMax(v0.z, v1.z), v2.z) < -e.z) return false;
    // the triangle plane
    const V3 n = cross3(ed[0], ed[1]);
    return planeBoxOverlap(n, -dot3(n, v0), e);
}

// --- P9.94: literal CUDA transcriptions of the pose / query-box math ---------
// PxTransform::transformInv(PxTransform) (PxTransform.h:162-169), spelled locally:
// qinv = q.getConjugate(); (qinv * src.q, qinv.rotate(src.p - p)).  NOT
// TransformV::transformInv, whose callers elsewhere must not move with this suite.
static void refAToB(const QuatV& q0, const Vec3& p0, const QuatV& q1, const Vec3& p1,
                    QuatV& outQ, Vec3& outP)
{
    const QuatV qinv = q1.getConjugate();
    outQ = qinv * q0;
    outP = qinv.rotate(p0 - p1);
}

// PxMeshScale::toMat33 (PxMeshScale.h:120-128): rot^T, column k scaled by s[k], times rot.
// (testUpdateBounds has its own meshScaleToMat33; distinct name on purpose.)
static Mat33 refMeshScaleToMat33(const Vec3& s, const QuatV& r)
{
    const Mat33 rot(r);
    Mat33 t = rot.getTranspose();
    t.col0 = t.col0 * s.x;
    t.col1 = t.col1 * s.y;
    t.col2 = t.col2 * s.z;
    return t * rot;
}

// shape2Vertex / vertex2Shape (cudaNpCommon.h:270-297).
static Vec3 refShape2Vertex(const Vec3& v, const Vec3& s, const QuatV& r)
{
    Vec3 t = r.rotate(v);
    t.x /= s.x; t.y /= s.y; t.z /= s.z;
    return r.rotateInv(t);
}
static Vec3 refVertex2Shape(const Vec3& v, const Vec3& s, const QuatV& r)
{
    Vec3 t = r.rotate(v);
    t.x *= s.x; t.y *= s.y; t.z *= s.z;
    return r.rotateInv(t);
}
static Vec3 absV(const Vec3& v) { return Vec3(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)); }

// The vertex-space query box (convexMeshMidphase.cu:321-327):
//   aabb -= trimeshTransf.p; fattenFast(cd);
//   transformFast(getInverse().toMat33() * PxMat33(q.getConjugate()), aabb)
//   = basisExtent(M * getCenter(), M, getExtents())   (PxBounds3.h:460-464, :301-314).
// preFix = the pre-P9.94 port spelling (per-axis rotateInv + shape2Vertex), kept ONLY
// to prove the directed inputs separate the two.
static void refQueryBox(const Vec3& aMin, const Vec3& aMax, const Vec3& p1, float cd,
                        const QuatV& q1, const Vec3& s, const QuatV& r, bool preFix,
                        Vec3& vMin, Vec3& vMax)
{
    const Vec3 mn = (aMin - p1) - Vec3(cd, cd, cd);
    const Vec3 mx = (aMax - p1) + Vec3(cd, cd, cd);
    const Vec3 c = (mn + mx) * 0.5f;
    const Vec3 e = (mx - mn) * 0.5f;
    Vec3 cV, w;
    if (!preFix)
    {
        const Mat33 M = refMeshScaleToMat33(Vec3(1.0f / s.x, 1.0f / s.y, 1.0f / s.z), r) *
                        Mat33(q1.getConjugate());
        cV = M.transform(c);
        const Vec3 c0 = M.col0 * e.x, c1 = M.col1 * e.y, c2 = M.col2 * e.z;
        w = Vec3(std::fabs(c0.x) + std::fabs(c1.x) + std::fabs(c2.x),
                 std::fabs(c0.y) + std::fabs(c1.y) + std::fabs(c2.y),
                 std::fabs(c0.z) + std::fabs(c1.z) + std::fabs(c2.z));
    }
    else
    {
        cV = refShape2Vertex(q1.rotateInv(c), s, r);
        const Vec3 ex = absV(refShape2Vertex(q1.rotateInv(Vec3(e.x, 0.0f, 0.0f)), s, r));
        const Vec3 ey = absV(refShape2Vertex(q1.rotateInv(Vec3(0.0f, e.y, 0.0f)), s, r));
        const Vec3 ez = absV(refShape2Vertex(q1.rotateInv(Vec3(0.0f, 0.0f, e.z)), s, r));
        w = (ex + ey) + ez;
    }
    vMin = cV - w;
    vMax = cV + w;
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

// P9.94: the poses / mesh scale / BV32 children, defaulting to the identity scene.
struct BuildOpts
{
    QuatV boxQ;                                   // identity
    Vec3 boxP;                                    // default (0, boxY, 0)
    QuatV meshQ;                                  // identity
    Vec3 meshP;                                   // origin
    Vec3 meshScale = Vec3(1.0f, 1.0f, 1.0f);
    QuatV meshScaleRot;                           // identity
    const std::vector<Child>* kids = nullptr;     // default splitIntoChildren(sc)
};
static BuildOpts defaultOpts(float boxY)
{
    BuildOpts o;
    o.boxP = Vec3(0.0f, boxY, 0.0f);
    return o;
}

// boxY: the box centre height (and its world AABB).  cached: build the manifold so
// the gate reports "the cached manifold stands" instead of asking for a full contact gen.
static Built build(PxgSlangComputeDevice* dev, const Scene& sc, float boxY, bool cached,
                   const BuildOpts* optsIn = nullptr)
{
    const BuildOpts opts = optsIn ? *optsIn : defaultOpts(boxY);
    Built b;
    b.nbTris = uint32_t(sc.tri.size() / 3);

    // --- the cooked-mesh blob: uint4 header, one BV32 node, float4 verts, uint4 idx
    std::vector<uint32_t> blob;
    blob.push_back(uint32_t(sc.verts.size()));  // .x nbVerts
    blob.push_back(b.nbTris);                   // .y nbTris
    blob.push_back(1u);                         // .z (a different node count; unused here)
    blob.push_back(1u);                         // .w nbPackedNodes -> sizes the node array
    { const auto n = makeBv32Root(opts.kids ? *opts.kids : splitIntoChildren(sc)); blob.insert(blob.end(), n.begin(), n.end()); }
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

    // PxsCachedTransform, 32 B: quat @0, pos @16.  Written from the option floats so a
    // -0.0f survives (an integer 0 in the initializer would be +0).
    std::vector<float> xf = { opts.boxQ.x, opts.boxQ.y, opts.boxQ.z, opts.boxQ.w,
                              opts.boxP.x, opts.boxP.y, opts.boxP.z, 0.0f,      // the box
                              opts.meshQ.x, opts.meshQ.y, opts.meshQ.z, opts.meshQ.w,
                              opts.meshP.x, opts.meshP.y, opts.meshP.z, 0.0f }; // the mesh
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
    s1.scaleX = opts.meshScale.x; s1.scaleY = opts.meshScale.y; s1.scaleZ = opts.meshScale.z;
    s1.scaleRotX = opts.meshScaleRot.x; s1.scaleRotY = opts.meshScaleRot.y;
    s1.scaleRotZ = opts.meshScaleRot.z; s1.scaleRotW = opts.meshScaleRot.w;
    s1.materialIndex = 11u; s1.type = GEOM_TRIANGLEMESH;
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
        // P9.36: PxPlane(v0,v1,v2) normalizes through getNormalized (PxVec3.h:286) =
        // `v * PxRecipSqrt(magnitudeSquared())` -- a reciprocal and three MULTIPLIES.
        // The shader spells it `nRaw * rsqrt(nl)`; this mirror divided three times, so
        // the pair was mismatched and merely happened to agree on the cases run.  Found
        // by sweeping the host mirrors, which gpuslang.vectorDivide cannot reach.
        const float rm = 1.0f / std::sqrt(m);
        const V3 n = m > 0.0f ? V3{ raw.x * rm, raw.y * rm, raw.z * rm }
                              : V3{ 0, 0, 0 };
        const float d = -dot3(n, a);
        if (dot3(n, com) + d < 0.0f) continue;            // the front-face gate
        if (intersectTriangleBox(center, ext, a, b2, c)) out.push_back(t);
    }
    return out;
}

static void dispatchMidphase(PxgSlangComputeDevice* dev, Built& b, uint32_t groups = 2u)
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

    // grid ((numTests+1)/2, 1, 1) x block (32, 2, 1) mirrors PxgNarrowphaseCore.cpp; the port's
    // launch site (PXG_NO_CUDA_TOOLKIT) launches one 32-thread group per CUDA warp, i.e.
    // ((numTests+1)/2) * 2 groups, one contact manager each.
    Kernel k("convexTrimeshMidphaseBda.spv", "midphaseGeneratePairs", {}, sizeof(MidphasePush));
    k.dispatch(groups, {}, p, 1u, 1u);
    dev->flush();
}

// ---------------------------------------------------------------------------
// `groups`: 2 is the port's launch (one group per CUDA warp, i.e. per manager); 1 is the
// CUDA's own grid, where one group takes both managers through the group-stride loop.
static void runPairs(uint32_t groups)
{
    PxgSlangComputeDevice* dev = testDevice();
    const Scene sc = makeScene(/*includeDownFacing*/ false);
    const float boxY = 0.4f;

    // P9.94: the box sits at (-0, boxY, -0).  PxTransform::transformInv's conjugate
    // rotate then gives aToB.p.z = -0 where the pre-fix rotateInv spelling gave +0;
    // nothing else in this scene moves (the record is where the kill is observed --
    // the manifold refresh is a second consumer none of these fixtures reach, all of
    // them take the invalidate path).
    BuildOpts opts = defaultOpts(boxY);
    opts.boxP = Vec3(-0.0f, boxY, -0.0f);
    Built b = build(dev, sc, boxY, /*cached*/ false, &opts);
    dispatchMidphase(dev, b, groups);

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
    auto pi = [&](size_t o) { int32_t v; std::memcpy(&v, gotPair.data() + o, 4); return v; };
    auto pu = [&](size_t o) { uint32_t v; std::memcpy(&v, gotPair.data() + o, 4); return v; };
    // P9.94: aToB BITWISE (a `!=` on floats is blind to the sign of zero), both records.
    QuatV expQ; Vec3 expP;
    refAToB(opts.boxQ, opts.boxP, opts.meshQ, opts.meshP, expQ, expP);
    {
        const Vec3 pre = opts.meshQ.rotateInv(opts.boxP - opts.meshP);
        if (f2u(pre.x) == f2u(expP.x) && f2u(pre.y) == f2u(expP.y) && f2u(pre.z) == f2u(expP.z))
        { ++mm; std::printf("  rotateInv and conjugate-rotate agree on aToB.p -- fixture is vacuous\n"); }
    }
    const uint32_t expA[7] = { f2u(expQ.x), f2u(expQ.y), f2u(expQ.z), f2u(expQ.w),
                               f2u(expP.x), f2u(expP.y), f2u(expP.z) };
    for (int cm = 0; cm < 2; ++cm)
    {
        const size_t r = size_t(56 * cm);
        for (int k = 0; k < 7; ++k)
            if (pu(r + 4 * k) != expA[k])
            { ++mm; std::printf("  record %d aToB[%d] got=0x%08X exp=0x%08X\n", cm, k, pu(r + 4 * k), expA[k]); }
        if (pi(r + 32) != int32_t(exp.size()))
        { ++mm; std::printf("  record %d count got=%d exp=%zu\n", cm, pi(r + 32), exp.size()); }
        if (pi(r + 36) != cm) { ++mm; std::printf("  record %d cmIndex got=%d\n", cm, pi(r + 36)); }
        if (pu(r + 48) != 7u || pu(r + 52) != 11u)
        { ++mm; std::printf("  record %d materials got=(%u,%u) exp=(7,11)\n", cm, pu(r + 48), pu(r + 52)); }
    }
    // startIndex: the two managers' ranges partition [0, total), in EITHER order.  Each manager
    // has its own 32-thread group (one per CUDA warp), and the two race for midphasePairsNum
    // exactly as the CUDA's two warps of the block do; which one allocates first is not
    // specified, and the consumers address a manager's pairs through its own record.
    {
        const int32_t s0 = pi(28), s1 = pi(56 + 28), n = int32_t(exp.size());
        if (!((s0 == 0 && s1 == n) || (s1 == 0 && s0 == n)))
        { ++mm; std::printf("  record startIndex got=(%d,%d) exp={0,%d}\n", s0, s1, n); }
    }

    PXG_CHECK_MSG(mm == 0, "midphaseGeneratePairs matches CPU at %u group(s) (%zu mismatches, %zu triangles)",
                  groups, mm, exp.size());
    destroy(dev, b);
}

PXG_TEST(convexTrimeshMidphasePairs)
{
    runPairs(2u);
}

PXG_TEST(convexTrimeshMidphasePairsCudaGrid)
{
    runPairs(1u);
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

// ---------------------------------------------------------------------------
// P9.94: the vertex-space query box, bit for bit.  convexMeshMidphase.cu:321-327
// builds worldToTrimeshVertex = getInverse().toMat33() * PxMat33(conj q) and applies
// PxBounds3::transformFast; the pre-fix port mapped the centre and the three axis
// extents through rotateInv + shape2Vertex instead -- equal in exact arithmetic,
// ULPs apart in float, and those ULPs decide which BV32 children are visited.
//
// The mesh is rotated and non-uniformly scaled (with a scale rotation), and the
// root's twelve leaf children are SLABS placed on each face of the CUDA box: three
// touching the max faces, three one ULP beyond them, three touching the min faces,
// three one ULP beyond those.  Every leaf holds the same triangle, which sits inside
// the box and faces its centre of mass, so the emitted set is exactly the set of
// visited children: {0,1,2,6,7,8} iff every face is bit-exact.  The box itself is
// rotated (q0 non-identity) so the record's aToB.q pins the quaternion product order.
// ---------------------------------------------------------------------------
PXG_TEST(convexTrimeshMidphaseQueryBox)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float boxY = 0.4f;
    // normalize(0.2, -0.1, 0.05, 0.97), literal bits
    const QuatV q0(u2f(0x3E4D7AADu), u2f(0xBDCD7AADu), u2f(0x3D4D7AADu), u2f(0x3F7924BFu));
    const Vec3 p0(0.0f, boxY, 0.0f);
    const Vec3 p1(0.25f, -0.5f, 0.125f);
    const Vec3 scale(2.0f, 3.0f, 0.7f);

    struct Case { const char* name; uint32_t q1[4]; uint32_t sr[4]; };
    const Case cases[2] = {
        // q1 = (0, sqrt(1/2), 0, sqrt(1/2)); scaleRot = normalize(0.3, 0, 0, 0.95)
        { "A", { 0x00000000u, 0x3F3504F3u, 0x00000000u, 0x3F3504F3u },
               { 0x3E9A2DE4u, 0x00000000u, 0x00000000u, 0x3F741DFDu } },
        // q1 = normalize(0.1, 0.2, 0.3, 0.9); scaleRot = normalize(0.3, 0.1, 0, 0.95)
        { "B", { 0x3DD21ED0u, 0x3E521ED0u, 0x3E9D971Cu, 0x3F6C62AAu },
               { 0x3E99688Bu, 0x3DCC8B63u, 0x00000000u, 0x3F72E586u } },
    };

    size_t mm = 0;
    for (const Case& cs : cases)
    {
        const QuatV q1(u2f(cs.q1[0]), u2f(cs.q1[1]), u2f(cs.q1[2]), u2f(cs.q1[3]));
        const QuatV sr(u2f(cs.sr[0]), u2f(cs.sr[1]), u2f(cs.sr[2]), u2f(cs.sr[3]));

        // The exact floats build() uploads: world AABB and contact distance.
        const Vec3 aMin(-0.5f, boxY - 0.5f, -0.5f), aMax(0.5f, boxY + 0.5f, 0.5f);
        const float cd = 0.02f + 0.02f;
        Vec3 V, Vp, preV, preVp;
        refQueryBox(aMin, aMax, p1, cd, q1, scale, sr, /*preFix*/ false, V, Vp);
        refQueryBox(aMin, aMax, p1, cd, q1, scale, sr, /*preFix*/ true, preV, preVp);

        // One triangle (+y wound, 0.2 below the box CoM, inside the box), twelve times.
        Scene sc;
        const Vec3 W[3] = { Vec3(-0.2f, 0.2f, -0.2f), Vec3(-0.2f, 0.2f, 0.2f), Vec3(0.2f, 0.2f, -0.2f) };
        for (const Vec3& w : W)
        {
            const Vec3 v = refShape2Vertex(q1.rotateInv(w - p1), scale, sr);
            sc.verts.push_back(V3{ v.x, v.y, v.z });
        }
        for (int t = 0; t < 12; ++t) { sc.tri.push_back(0u); sc.tri.push_back(1u); sc.tri.push_back(2u); sc.up.push_back(1); }

        auto comp = [](const Vec3& v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); };
        auto setc = [](V3& v, int a, float f) { if (a == 0) v.x = f; else if (a == 1) v.y = f; else v.z = f; };
        std::vector<Child> kids;
        for (int k = 0; k < 12; ++k)
        {
            Child c{}; c.primStart = uint32_t(k); c.nbPrims = 1u;
            c.lo = V3{ -1e30f, -1e30f, -1e30f }; c.hi = V3{ 1e30f, 1e30f, 1e30f };
            const int a = k % 3;
            if (k < 3)       setc(c.lo, a, comp(Vp, a));                                  // max face, touching
            else if (k < 6)  setc(c.lo, a, std::nextafterf(comp(Vp, a), INFINITY));       // max face, 1 ULP out
            else if (k < 9)  setc(c.hi, a, comp(V, a));                                   // min face, touching
            else             setc(c.hi, a, std::nextafterf(comp(V, a), -INFINITY));       // min face, 1 ULP out
            kids.push_back(c);
        }

        // The oracle: root walk (cu:160 predicate), then the gate and the triangle test.
        QuatV aQ; Vec3 aP;
        refAToB(q0, p0, q1, p1, aQ, aP);
        const Vec3 boxScale(0.5f, 0.5f, 0.5f);
        const QuatV boxRot;
        const Vec3 com = refShape2Vertex(aQ.rotate(refVertex2Shape(Vec3(0.0f, 0.0f, 0.0f), boxScale, boxRot)) + aP,
                                         scale, sr);
        const V3 center = { (aMin.x + aMax.x) * 0.5f, (aMin.y + aMax.y) * 0.5f, (aMin.z + aMax.z) * 0.5f };
        const V3 ext = { (aMax.x - aMin.x) * 0.5f + cd, (aMax.y - aMin.y) * 0.5f + cd, (aMax.z - aMin.z) * 0.5f + cd };
        auto walk = [&](const Vec3& qMin, const Vec3& qMax) {
            std::vector<uint32_t> out;
            for (const Child& c : kids)
            {
                const bool hit = !(c.lo.x > qMax.x || qMin.x > c.hi.x || c.lo.y > qMax.y ||
                                   qMin.y > c.hi.y || c.lo.z > qMax.z || qMin.z > c.hi.z);
                if (!hit) continue;
                const Vec3 a(sc.verts[0].x, sc.verts[0].y, sc.verts[0].z);
                const Vec3 b2(sc.verts[1].x, sc.verts[1].y, sc.verts[1].z);
                const Vec3 c3(sc.verts[2].x, sc.verts[2].y, sc.verts[2].z);
                // PxPlane(v0,v1,v2) (PxPlane.h:80-84), getNormalized = v * (1/sqrt(m)).
                const Vec3 raw = (b2 - a).cross(c3 - a);
                const float m = raw.dot(raw);
                const Vec3 n = m > 0.0f ? raw * (1.0f / std::sqrt(m)) : Vec3(0.0f, 0.0f, 0.0f);
                const float d = -n.dot(a);
                if (n.dot(com) + d < 0.0f) continue;
                auto world = [&](const Vec3& v) {
                    const Vec3 w = q1.rotate(refVertex2Shape(v, scale, sr)) + p1;
                    return V3{ w.x, w.y, w.z };
                };
                if (intersectTriangleBox(center, ext, world(a), world(b2), world(c3)))
                    out.push_back(c.primStart);
            }
            return out;
        };
        const std::vector<uint32_t> exp = walk(V, Vp);
        const std::vector<uint32_t> want = { 0u, 1u, 2u, 6u, 7u, 8u };
        if (exp != want)
        { ++mm; std::printf("  case %s: CPU expected set is not {0,1,2,6,7,8} (%zu) -- fixture is vacuous\n", cs.name, exp.size()); }
        if (walk(preV, preVp) == exp)
        { ++mm; std::printf("  case %s: the pre-fix query box gives the same set -- fixture is vacuous\n", cs.name); }

        BuildOpts opts = defaultOpts(boxY);
        opts.boxQ = q0; opts.boxP = p0;
        opts.meshQ = q1; opts.meshP = p1;
        opts.meshScale = scale; opts.meshScaleRot = sr;
        opts.kids = &kids;
        Built b = build(dev, sc, boxY, /*cached*/ false, &opts);
        dispatchMidphase(dev, b);

        const uint32_t total = 2u * uint32_t(exp.size());
        const auto gotNum = downloadBuffer<uint32_t>(dev, b.pairsNum.h, 1);
        const auto gotOff = downloadBuffer<uint32_t>(dev, b.stackOff.h, 1);
        const auto gotStack = downloadBuffer<uint32_t>(dev, b.stack.h, 4u * 32u);
        const auto gotPair = downloadBuffer<uint8_t>(dev, b.pair.h, 112);

        if (gotNum[0] != total) { ++mm; std::printf("  case %s: pairsNum got=%u exp=%u\n", cs.name, gotNum[0], total); }
        if (gotOff[0] != total) { ++mm; std::printf("  case %s: stackOffset got=%u exp=%u\n", cs.name, gotOff[0], total); }
        std::vector<uint32_t> perCm[2];
        for (uint32_t i = 0; i < gotOff[0] && i < 32u; ++i)
        {
            const uint32_t cm = gotStack[4*i + 0];
            if (cm > 1u) { ++mm; std::printf("  case %s: pair %u cmIdx got=%u\n", cs.name, i, cm); continue; }
            perCm[cm].push_back(gotStack[4*i + 1]);
        }
        for (int cm = 0; cm < 2; ++cm)
        {
            std::sort(perCm[cm].begin(), perCm[cm].end());
            if (perCm[cm] != exp)
            {
                ++mm;
                std::printf("  case %s cm %d: emitted set differs, got {", cs.name, cm);
                for (uint32_t t : perCm[cm]) std::printf(" %u", t);
                std::printf(" }\n");
            }
        }

        auto pi = [&](size_t o) { int32_t v; std::memcpy(&v, gotPair.data() + o, 4); return v; };
        auto pu = [&](size_t o) { uint32_t v; std::memcpy(&v, gotPair.data() + o, 4); return v; };
        const uint32_t expA[7] = { f2u(aQ.x), f2u(aQ.y), f2u(aQ.z), f2u(aQ.w),
                                   f2u(aP.x), f2u(aP.y), f2u(aP.z) };
        for (int cm = 0; cm < 2; ++cm)
        {
            const size_t r = size_t(56 * cm);
            for (int k = 0; k < 7; ++k)
                if (pu(r + 4 * k) != expA[k])
                { ++mm; std::printf("  case %s record %d aToB[%d] got=0x%08X exp=0x%08X\n", cs.name, cm, k, pu(r + 4 * k), expA[k]); }
            if (pi(r + 32) != int32_t(exp.size()))
            { ++mm; std::printf("  case %s record %d count got=%d exp=%zu\n", cs.name, cm, pi(r + 32), exp.size()); }
            if (pi(r + 36) != cm) { ++mm; std::printf("  case %s record %d cmIndex got=%d\n", cs.name, cm, pi(r + 36)); }
            if (pu(r + 48) != 7u || pu(r + 52) != 11u)
            { ++mm; std::printf("  case %s record %d materials got=(%u,%u)\n", cs.name, cm, pu(r + 48), pu(r + 52)); }
        }
        destroy(dev, b);
    }

    PXG_CHECK_MSG(mm == 0, "the vertex-space query box is PxBounds3::transformFast bit for bit (%zu mismatches)", mm);
}

} // namespace
