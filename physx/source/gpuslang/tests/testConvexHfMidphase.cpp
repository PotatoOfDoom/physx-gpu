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
// P9.51a — convexHeightFieldMidphase (convexHfMidphaseBda.spv) on the engine ABI.
//
// Suites (substring-matched by the "convexHfMidphase" ctest filter):
//   convexHfMidphasePairs   — the emitted triangle SET, both counters and the
//                             ConvexMeshPair record, vs a CPU transcription of
//                             convexHFMidphase.cu:69
//   convexHfMidphaseCached  — the PCM gate says "cached": no walk, no pairs, and
//                             count == CONVEX_TRIMESH_CACHED (-1)
//   convexHfMidphaseRotated — the same walk with a NON-IDENTITY heightfield
//                             transform, which is the only fixture that can see
//                             the bound-to-heightfield-space conversion at all
//
// WHY THREE.  `Pairs` alone cannot see two of the three things most likely to be
// got wrong.  A port that ignored the PCM gate passes it (fullContactGen is true
// there anyway), and a port whose transform conversion is wrong passes it too,
// because with an identity heightfield transform almost any spelling agrees.
// Each of those failure modes gets a fixture only it fails.
//
// TWO SHAPES, BOTH IN SLOT 0.  convexHFMidphase.cu does NOT swap slot 0 and slot 1
// the way its trimesh sibling does — it reads slot 0 as the convex and slot 1 as
// the heightfield unconditionally (:113, :165).  So both managers here put the
// convex first, and they differ in TYPE instead: manager 0 is a box, which takes
// the calculatePCMConvexMargin arm, and manager 1 is a sphere, which takes the
// `minMargin = scale.x` arm.  A port that collapsed the two arms into one passes
// neither.

#include "pxgTestFramework.h"

#include "geometry/PxGeometry.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

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

struct V3 { float x, y, z; };

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
static const uint32_t GEOM_SPHERE = uint32_t(physx::PxGeometryType::eSPHERE), GEOM_BOX = uint32_t(physx::PxGeometryType::eBOX),
                      GEOM_HEIGHTFIELD = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
static const uint32_t HF_HOLE = 127u;
static const int CONVEX_TRIMESH_CACHED = -1;

// ---------------------------------------------------------------------------
// The heightfield.  nbRows @0, nbCols @4, then nbRows*nbCols PxHeightFieldSample:
// PxI16 height @0, materialIndex0 @2, materialIndex1 @3.
//
// The grid is 5x5 and deliberately NOT flat: a ridge along row 2 puts some quads
// entirely above the query box and some entirely below it, so both `con0` and
// `con1` (convexHFMidphase.cu:277-278) reject at least one candidate each.  If
// every height were equal, neither predicate would ever fire and a port that
// dropped both would still pass.
// ---------------------------------------------------------------------------
static const uint32_t HF_ROWS = 8u, HF_COLS = 5u;

static uint32_t hfSample(int16_t height, uint8_t mat0, uint8_t mat1)
{
    return uint32_t(uint16_t(height)) | (uint32_t(mat0) << 16) | (uint32_t(mat1) << 24);
}

static std::vector<uint32_t> makeHeightfield()
{
    std::vector<uint32_t> blob;
    blob.push_back(HF_ROWS);
    blob.push_back(HF_COLS);
    for (uint32_t r = 0; r < HF_ROWS; ++r)
        for (uint32_t c = 0; c < HF_COLS; ++c)
        {
            // Every height here is doing a job; the first version of this
            // fixture had a coarse +/-20000 profile and SEVEN mutants survived it.
            //
            //   rows 0-1   200      both above maxy -> quad row 0 fires con0
            //   rows 2-3   c / 2+c  varies with the COLUMN, so h3 (row+1,col+1)
            //                       differs from h2 (row+1,col) -- without that a
            //                       mutant that dropped the diagonal survives
            //   rows 4-5   -1-c/-2  SMALL negatives: the y range's sign and scale
            //                       decide these, which is what makes the u32
            //                       saturation and the height scale observable
            //   rows 6-7   -20000   both below miny -> quad row 6 fires con1
            int16_t h = 0;
            if (r <= 1u) h = 200;
            // ONE low cell inside the plateau, placed so that for quad (r=0,c=2)
            // the DIAGONAL corner h3 = (r+1,c+1) is the only one below maxy.
            // Without it con0 never depends on h3 and a mutant that reads h2 twice
            // survives -- which is exactly what the first two sweeps showed.
            if (r == 1u && c == 3u) h = -100;
            else if (r <= 1u) h = 200;
            else if (r == 2u) h = int16_t(c);
            else if (r == 3u) h = int16_t(2 + c);
            else if (r == 4u) h = int16_t(-1 - int(c));
            else if (r == 5u) h = -2;
            else h = -20000;
            // One quad carries a HOLE in material slot 0 and a live material in
            // slot 1, so the two halves of a quad are decided SEPARATELY.  A port
            // that read one material byte for both triangles emits one pair too
            // many here, or one too few.
            const bool hole = (r == 2u && c == 1u);
            blob.push_back(hfSample(h, hole ? uint8_t(HF_HOLE) : uint8_t(3),
                                    uint8_t(5)));
        }
    return blob;
}

static float hfHeightAt(const std::vector<uint32_t>& blob, uint32_t vertexIndex)
{
    const uint32_t w = blob[2 + vertexIndex];
    return float(int32_t(w << 16) >> 16);
}
static uint32_t hfMat0At(const std::vector<uint32_t>& blob, uint32_t vertexIndex)
{ return (blob[2 + vertexIndex] >> 16) & 0x7fu; }
static uint32_t hfMat1At(const std::vector<uint32_t>& blob, uint32_t vertexIndex)
{ return (blob[2 + vertexIndex] >> 24) & 0x7fu; }

// ---------------------------------------------------------------------------
// The CPU model, transcribed from convexHFMidphase.cu:176-321.
//
// Written from the .cu, NOT from the shader -- a mirror derived from the port
// agrees with the port by construction and proves nothing.  Everything here has
// a line number beside it for that reason.
// ---------------------------------------------------------------------------
struct HfXform { float qx, qy, qz, qw, px, py, pz; };

static V3 quatRotateInvH(const HfXform& t, V3 v)
{
    // PxQuat::rotateInv: v*(2w^2 - 1) - cross(axis,v)*2w + axis*(dot(axis,v)*2)
    const float w = t.qw;
    const V3 a{ t.qx, t.qy, t.qz };
    const float d = a.x * v.x + a.y * v.y + a.z * v.z;
    const V3 cr{ a.y * v.z - a.z * v.y, a.z * v.x - a.x * v.z, a.x * v.y - a.y * v.x };
    const float k = 2.0f * w * w - 1.0f;
    return V3{ v.x * k - cr.x * 2.0f * w + a.x * d * 2.0f,
               v.y * k - cr.y * 2.0f * w + a.y * d * 2.0f,
               v.z * k - cr.z * 2.0f * w + a.z * d * 2.0f };
}

// PxMat33(PxQuat) columns -- what PxBounds3::transformFast takes the absolute
// value of.  NOT quatRotate(q, e_i): the same value in exact arithmetic and
// different in the last bits (P9.37).
static void quatBasisH(float x, float y, float z, float w, V3& c0, V3& c1, V3& c2)
{
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = V3{ 1.0f - yy - zz, xy + zw, xz - yw };
    c1 = V3{ xy - zw, 1.0f - xx - zz, yz + xw };
    c2 = V3{ xz + yw, yz - xw, 1.0f - xx - yy };
}

static uint32_t minRowH(float x, uint32_t rows)
{ int v = int(std::floor(x)); v = v < 0 ? 0 : v; const int hi = int(rows) - 2; return uint32_t(v > hi ? hi : v); }
static uint32_t maxRowH(float x, uint32_t rows)
{ int v = int(std::ceil(x));  v = v < 0 ? 0 : v; const int hi = int(rows) - 1; return uint32_t(v > hi ? hi : v); }
static uint32_t minColH(float z, uint32_t cols)
{ int v = int(std::floor(z)); v = v < 0 ? 0 : v; const int hi = int(cols) - 2; return uint32_t(v > hi ? hi : v); }
static uint32_t maxColH(float z, uint32_t cols)
{ int v = int(std::ceil(z));  v = v < 0 ? 0 : v; const int hi = int(cols) - 1; return uint32_t(v > hi ? hi : v); }

// The triangle indices this manager must emit, in the order the warp produces
// them (ascending workIndex), which is also the order they land on the stack.
static std::vector<uint32_t> expectedTriangles(const std::vector<uint32_t>& hf,
                                               const HfXform& hfT, V3 hfScale,
                                               V3 worldMin, V3 worldMax, float contactDist)
{
    std::vector<uint32_t> out;

    const float oneOverHeightScale = 1.0f / hfScale.y;
    const float oneOverRowScale = 1.0f / std::fabs(hfScale.x);
    const float oneOverColScale = 1.0f / std::fabs(hfScale.z);

    // PxBounds3::transformFast(getInverse(), worldBound) -- the inverse is
    // materialised FIRST (:191), so the centre is rotInv(q,c) + rotInv(q,-p).
    const V3 centerW{ (worldMin.x + worldMax.x) * 0.5f, (worldMin.y + worldMax.y) * 0.5f,
                      (worldMin.z + worldMax.z) * 0.5f };
    const V3 extentW{ (worldMax.x - worldMin.x) * 0.5f, (worldMax.y - worldMin.y) * 0.5f,
                      (worldMax.z - worldMin.z) * 0.5f };
    const V3 negP{ -hfT.px, -hfT.py, -hfT.pz };
    const V3 invP = quatRotateInvH(hfT, negP);
    const V3 rc = quatRotateInvH(hfT, centerW);
    const V3 center{ rc.x + invP.x, rc.y + invP.y, rc.z + invP.z };

    V3 c0, c1, c2;
    quatBasisH(-hfT.qx, -hfT.qy, -hfT.qz, hfT.qw, c0, c1, c2);
    const V3 e0{ c0.x * extentW.x, c0.y * extentW.x, c0.z * extentW.x };
    const V3 e1{ c1.x * extentW.y, c1.y * extentW.y, c1.z * extentW.y };
    const V3 e2{ c2.x * extentW.z, c2.y * extentW.z, c2.z * extentW.z };
    const V3 wext{ std::fabs(e0.x) + std::fabs(e1.x) + std::fabs(e2.x),
                   std::fabs(e0.y) + std::fabs(e1.y) + std::fabs(e2.y),
                   std::fabs(e0.z) + std::fabs(e1.z) + std::fabs(e2.z) };

    V3 lmn{ center.x - wext.x, center.y - wext.y, center.z - wext.z };
    V3 lmx{ center.x + wext.x, center.y + wext.y, center.z + wext.z };
    lmn.x -= contactDist; lmn.y -= contactDist; lmn.z -= contactDist;   // fattenFast (:190)
    lmx.x += contactDist; lmx.y += contactDist; lmx.z += contactDist;

    lmn = V3{ lmn.x * oneOverRowScale, lmn.y * oneOverHeightScale, lmn.z * oneOverColScale };
    lmx = V3{ lmx.x * oneOverRowScale, lmx.y * oneOverHeightScale, lmx.z * oneOverColScale };
    if (hfScale.x < 0.0f) { const float t = lmn.x; lmn.x = lmx.x; lmx.x = t; }
    if (hfScale.z < 0.0f) { const float t = lmn.z; lmn.z = lmx.z; lmx.z = t; }

    // The gate, with the WHOLE disjunction negated (:221) -- not the FEM path's
    // `!`-binding quirk, which is why pxgSlangHeightfield::hfMakeRect is not used
    // by the shader either.
    const bool boundsDontOverlap =
        (lmn.x > float(HF_ROWS) - 1.0f) || (lmn.z > float(HF_COLS) - 1.0f) ||
        (lmx.x < 0.0f) || (lmx.z < 0.0f);
    if (boundsDontOverlap) return out;

    const uint32_t minRow = minRowH(lmn.x, HF_ROWS);
    const uint32_t maxRow = maxRowH(lmx.x, HF_ROWS);
    const uint32_t minColumn = minColH(lmn.z, HF_COLS);
    const uint32_t maxColumn = maxColH(lmx.z, HF_COLS);
    if ((2u * (maxColumn - minColumn) * (maxRow - minRow)) == 0u) return out;

    const float miny = lmn.y;    // FLOAT y range (:254) -- no u32 conversion
    const float maxy = lmx.y;
    const uint32_t columnSpan = maxColumn - minColumn;
    const uint32_t totalNumProcessed = (maxRow - minRow) * columnSpan * 2u;

    for (uint32_t workIndex = 0; workIndex < totalNumProcessed; ++workIndex)
    {
        const uint32_t index = workIndex / 2u;
        const uint32_t vertexIndex = (minRow + index / columnSpan) * HF_COLS +
                                     (minColumn + index % columnSpan);
        const float h0 = hfHeightAt(hf, vertexIndex);
        const float h1 = hfHeightAt(hf, vertexIndex + 1u);
        const float h2 = hfHeightAt(hf, vertexIndex + HF_COLS);
        const float h3 = hfHeightAt(hf, vertexIndex + HF_COLS + 1u);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;
        if (con0 || con1) continue;
        const bool isMaterial1 = (workIndex & 1u) != 0u;
        const uint32_t material = isMaterial1 ? hfMat1At(hf, vertexIndex)
                                              : hfMat0At(hf, vertexIndex);
        if (material == HF_HOLE) continue;
        out.push_back(isMaterial1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1));
    }
    return out;
}

// ---------------------------------------------------------------------------
// The scene.  Two managers, both with the convex in slot 0 (see the header).
//
//   manager 0  a BOX spanning the whole grid          -> the convex-margin arm,
//              72 candidates, so the warp loop runs THREE times (32/32/8).  A
//              port that dropped the `i += WARP` stride passes a one-batch
//              fixture and fails this one.
//   manager 1  a SPHERE over the far corner            -> the sphere-margin arm,
//              and a rectangle whose minRow/minColumn are NOT zero, which is the
//              only way to see the `minRow + index / columnSpan` term.
// ---------------------------------------------------------------------------
struct Built
{
    DevBuf hf, hull, cmIn, xf, bnd, cdist, shapes, pair, manifold, stack,
           stackOff, pairsNum, pairsNumPadded;
    std::vector<uint32_t> hfBlob;
};

static const float HF_SCALE_Y = 0.1f;    // int16 heights -> world units
static const float CDIST = 0.02f;        // per shape; the kernel sums the two

static Built build(PxgSlangComputeDevice* dev, bool cached)
{
    Built b;
    b.hfBlob = makeHeightfield();
    b.hf = makeDev(dev, b.hfBlob);

    // A convex hull blob: the internal-AABB extents live at +sizeof(float4)*2.
    std::vector<float> hull(12, 0.0f);
    hull[8] = 0.5f; hull[9] = 0.5f; hull[10] = 0.5f;      // extents at @32
    b.hull = makeDev(dev, hull);

    // THREE managers, all with the convex in slot 0 (see the header).
    std::vector<CmInputHost> cmIn = { { 0u, 3u, 0u, 3u }, { 1u, 3u, 1u, 3u },
                                      { 2u, 3u, 2u, 3u } };
    b.cmIn = makeDev(dev, cmIn);

    // PxsCachedTransform, 32 B: quat @0, pos @16.
    std::vector<float> xf = { 0,0,0,1,  3.5f, 0.3f, 2.0f, 0,      // the box
                              0,0,0,1,  2.0f, 0.3f, 2.0f, 0,      // the sphere
                              0,0,0,1,  2.0f, 0.3f, 11.0f, 0,     // the off-grid box
                              0,0,0,1,  0,0,0,0 };                // the heightfield
    b.xf = makeDev(dev, xf);

    // PxBounds3, 24 B each, in the same order.
    //
    //   manager 0  spans the whole grid, y from -0.4 to 1.0.  The NEGATIVE miny is
    //              what makes the u32-saturating y conversion observable at all --
    //              max(y,0) only differs from y when y is below zero.
    //   manager 1  a sphere starting at 2.01 -- a NON-ZERO minRow and minColumn,
    //              without which the `minRow +` and `minColumn +` terms are dead --
    //              and ending at 4.98/3.98.  The summed contact
    //              distance 0.04 pushes that to 3.02 and moves ceil() from 3 to 4,
    //              so dropping the fatten, or summing only one shape's distance,
    //              changes the rectangle.  Chosen deliberately: at a round
    //              placement both mutants are invisible.
    //   manager 2  INSIDE the grid in x, far outside it in z.  This is the only
    //              manager any gate variant can see: the correctly-bracketed gate
    //              rejects it outright, while the FEM path's `!`-binding quirk --
    //              and a gate that tests rows only -- would admit it and emit
    //              pairs against a clamped column range.
    std::vector<float> bnd = { -0.5f, -0.4f, -0.5f,   7.5f, 1.0f, 4.5f,    // box
                                2.01f, -0.4f, 2.01f,  4.98f, 1.0f, 3.98f,  // sphere
                                1.00f, -0.4f, 10.0f,  3.00f, 1.0f, 12.0f,  // off-grid
                               -1.0f, -1.0f, -1.0f,   8.0f, 1.0f, 5.0f };  // hf (unread)
    b.bnd = makeDev(dev, bnd);

    std::vector<float> cd = { CDIST, CDIST, CDIST, CDIST };
    b.cdist = makeDev(dev, cd);

    ShapeHost box{}, sph{}, hfs{};
    box.scaleX = box.scaleY = box.scaleZ = 0.5f;   // box half-extents live in scale
    box.scaleRotW = 1.0f; box.materialIndex = 7u; box.type = GEOM_BOX;
    box.particleOrSoftbodyId = 0xFFFFFFFFu;
    box.hullLo = uint32_t(b.hull.addr & 0xFFFFFFFFull);
    box.hullHi = uint32_t(b.hull.addr >> 32);

    sph.scaleX = 1.0f; sph.scaleY = sph.scaleZ = 1.0f;   // sphere radius in scale.x
    sph.scaleRotW = 1.0f; sph.materialIndex = 9u; sph.type = GEOM_SPHERE;
    sph.particleOrSoftbodyId = 0xFFFFFFFFu;
    sph.hullLo = uint32_t(b.hull.addr & 0xFFFFFFFFull);
    sph.hullHi = uint32_t(b.hull.addr >> 32);

    hfs.scaleX = 1.0f; hfs.scaleY = HF_SCALE_Y; hfs.scaleZ = 1.0f;
    hfs.scaleRotW = 1.0f; hfs.materialIndex = 11u; hfs.type = GEOM_HEIGHTFIELD;
    hfs.particleOrSoftbodyId = 0xFFFFFFFFu;
    hfs.hullLo = uint32_t(b.hf.addr & 0xFFFFFFFFull);
    hfs.hullHi = uint32_t(b.hf.addr >> 32);

    ShapeHost box2 = box;
    box2.materialIndex = 13u;
    std::vector<ShapeHost> shapes = { box, sph, box2, hfs };
    b.shapes = makeDev(dev, shapes);

    // --- the persistent manifold, ONE RECORD PER CONTACT MANAGER -------------
    auto makeManifold = [&](float py) {
        std::vector<uint8_t> man(1216, 0u);
        auto putf = [&](size_t off, float v) { std::memcpy(man.data() + off, &v, 4); };
        auto putu = [&](size_t off, uint32_t v) { std::memcpy(man.data() + off, &v, 4); };
        putf(1168, 0.0f); putf(1172, 0.0f); putf(1176, 0.0f); putf(1180, 1.0f);   // q
        putf(1184, 3.0f); putf(1188, py); putf(1192, 3.0f); putf(1196, 0.0f);     // p
        putu(1200, 0u);                                                          // mNbManifolds
        return man;
    };
    auto putf = [&](std::vector<uint8_t>& m, size_t off, float v)
                { std::memcpy(m.data() + off, &v, 4); };
    std::vector<uint8_t> man2;
    if (cached)
    {
        // mRelativeTransform close enough to aToB that invalidateManifold reports
        // false, and zero manifolds so the refresh finds nothing: the walk is
        // skipped and the record carries the CACHED sentinel.
        //
        // THE SPHERE'S DELTA IS 0.015, AND THAT NUMBER IS THE POINT.  Its margin
        // arm gives minMargin = radius = 1.0 and ratio 0.02, so its threshold is
        // 0.02 and 0.015 stays cached.  The CONVEX arm would give
        // min(0.5*0.25, 1.0*0.05) = 0.05 at ratio 0.2, a threshold of 0.01, and
        // 0.015 would invalidate and walk.  Without a delta BETWEEN the two
        // thresholds the margin never decides anything and a mutant that collapses
        // the two arms survives -- which is exactly what happened first time.
        std::vector<uint8_t> m0 = makeManifold(0.3f - 0.0005f);
        putf(m0, 1184, 3.5f); putf(m0, 1192, 2.0f);
        std::vector<uint8_t> m1 = makeManifold(0.3f);
        putf(m1, 1184, 2.0f + 0.015f); putf(m1, 1192, 2.0f);
        std::vector<uint8_t> m2 = makeManifold(0.3f);
        putf(m2, 1184, 2.0f); putf(m2, 1192, 11.0f);
        man2.insert(man2.end(), m0.begin(), m0.end());
        man2.insert(man2.end(), m1.begin(), m1.end());
        man2.insert(man2.end(), m2.begin(), m2.end());
    }
    else
    {
        std::vector<uint8_t> far = makeManifold(100.0f);
        for (int i = 0; i < 3; ++i) man2.insert(man2.end(), far.begin(), far.end());
    }
    b.manifold = makeDev(dev, man2);

    std::vector<uint8_t> pair(3 * 56, 0xEE);
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
    for (PxgBufferHandle h : { b.hf.h, b.hull.h, b.cmIn.h, b.xf.h, b.bnd.h, b.cdist.h,
                               b.shapes.h, b.pair.h, b.manifold.h, b.stack.h,
                               b.stackOff.h, b.pairsNum.h, b.pairsNumPadded.h })
        dev->destroyBuffer(h);
}

static void dispatchMidphase(PxgSlangComputeDevice* dev, Built& b)
{
    MidphasePush p{};
    p.numContactManagers = 3u;
    p.toleranceLength = 1.0f;
    p.cmInputs = b.cmIn.addr; p.transformCache = b.xf.addr; p.bounds = b.bnd.addr;
    p.contactDistance = b.cdist.addr; p.gpuShapes = b.shapes.addr;
    p.cvxTrimeshPair = b.pair.addr; p.multiManifolds = b.manifold.addr;
    p.cmOutputs = 0ull;                                   // never touched, see the shader header
    p.stackPtr = b.stack.addr; p.stackOffset = b.stackOff.addr;
    p.midphasePairsNum = b.pairsNum.addr; p.midphasePairsNumPadded = b.pairsNumPadded.addr;
    p.stackSizeBytes = 16u * 1024u;

    // grid ((numTests + warpsPerBlock - 1) / warpsPerBlock, 1, 1), the engine's
    // own arithmetic at PxgNarrowphaseCore.cpp:2856.
    Kernel k("convexHfMidphaseBda.spv", "convexHeightFieldMidphase", {}, sizeof(MidphasePush));
    k.dispatch(2u, {}, p, 1u, 1u);
    dev->flush();
}

// The two managers' expected sets, from the CPU model.
static const V3 WMIN[3] = { { -0.5f, -0.4f, -0.5f }, { 2.01f, -0.4f, 2.01f },
                            { 1.00f, -0.4f, 10.0f } };
static const V3 WMAX[3] = { { 7.5f, 1.0f, 4.5f },   { 4.98f, 1.0f, 3.98f },
                            { 3.00f, 1.0f, 12.0f } };

static std::vector<uint32_t> expectFor(const Built& b, uint32_t manager)
{
    const HfXform hfT{ 0, 0, 0, 1, 0, 0, 0 };            // identity heightfield
    const V3 hfScale{ 1.0f, HF_SCALE_Y, 1.0f };
    return expectedTriangles(b.hfBlob, hfT, hfScale, WMIN[manager], WMAX[manager],
                             2.0f * CDIST);
}

// ---------------------------------------------------------------------------
PXG_TEST(convexHfMidphasePairs)
{
    PxgSlangComputeDevice* dev = testDevice();
    Built b = build(dev, /*cached*/ false);
    dispatchMidphase(dev, b);

    const std::vector<uint32_t> e0 = expectFor(b, 0u);
    const std::vector<uint32_t> e1 = expectFor(b, 1u);
    const std::vector<uint32_t> e2 = expectFor(b, 2u);
    PXG_CHECK_MSG(!e0.empty() && !e1.empty(),
                  "managers 0 and 1 must produce pairs (got %zu and %zu) -- an empty "
                  "expectation would make every check below vacuous",
                  e0.size(), e1.size());
    PXG_CHECK_MSG(e2.empty(),
                  "manager 2 sits outside the grid in z and the correctly-bracketed "
                  "gate must reject it outright; the model says %zu pairs, so the model "
                  "and the gate disagree about what this fixture is for", e2.size());

    const auto stack = downloadBuffer<uint32_t>(dev, b.stack.h, 4u * 1024u);
    const auto nPairs = downloadBuffer<uint32_t>(dev, b.pairsNum.h, 1);
    const auto nPadded = downloadBuffer<uint32_t>(dev, b.pairsNumPadded.h, 1);
    const auto nStack = downloadBuffer<uint32_t>(dev, b.stackOff.h, 1);
    const auto pair = downloadBuffer<uint8_t>(dev, b.pair.h, 3 * 56);

    // The stack holds uint4 (cmIdx, triangleIndex, perCmSlot, hfShapeRef).  Both
    // managers write into one global cursor, so collect per manager.
    std::vector<uint32_t> got0, got1;
    std::vector<uint32_t> slot0, slot1;
    for (uint32_t i = 0; i < nStack[0]; ++i)
    {
        const uint32_t cm = stack[4 * i], tri = stack[4 * i + 1],
                       slot = stack[4 * i + 2], hfRef = stack[4 * i + 3];
        PXG_CHECK_MSG(hfRef == 3u, "entry %u carries hfShapeRef %u, expected 3 -- the "
                      "fourth word is cin.shapeRef1, not the convex's", i, hfRef);
        if (cm == 0u) { got0.push_back(tri); slot0.push_back(slot); }
        else if (cm == 1u) { got1.push_back(tri); slot1.push_back(slot); }
        else PXG_CHECK_MSG(false, "entry %u has cmIdx %u -- manager 2 is off the grid "
                           "and must emit nothing", i, cm);
    }
    std::sort(got0.begin(), got0.end());
    std::sort(got1.begin(), got1.end());
    std::vector<uint32_t> w0 = e0, w1 = e1;
    std::sort(w0.begin(), w0.end());
    std::sort(w1.begin(), w1.end());

    PXG_CHECK_MSG(got0 == w0, "manager 0 emitted %zu triangles, expected %zu",
                  got0.size(), w0.size());
    PXG_CHECK_MSG(got1 == w1, "manager 1 emitted %zu triangles, expected %zu",
                  got1.size(), w1.size());

    // The per-manager slot indices must be a permutation of 0..n-1: they are what
    // the core kernel uses to find this pair's intermediate arrays, so a duplicate
    // or a gap silently aliases two pairs onto one slot.
    std::sort(slot0.begin(), slot0.end());
    std::sort(slot1.begin(), slot1.end());
    for (size_t i = 0; i < slot0.size(); ++i)
        PXG_CHECK_MSG(slot0[i] == uint32_t(i), "manager 0 slot %zu is %u", i, slot0[i]);
    for (size_t i = 0; i < slot1.size(); ++i)
        PXG_CHECK_MSG(slot1[i] == uint32_t(i), "manager 1 slot %zu is %u", i, slot1[i]);

    PXG_CHECK_MSG(nPairs[0] == uint32_t(w0.size() + w1.size()),
                  "midphasePairsNum is %u, expected %zu", nPairs[0], w0.size() + w1.size());
    PXG_CHECK_MSG(nStack[0] == nPairs[0],
                  "stackOffset (%u) and midphasePairsNum (%u) must agree once every "
                  "manager has contributed -- they are different counters and crossing "
                  "them is the trap the shader header names", nStack[0], nPairs[0]);

    const uint32_t pad0 = ((uint32_t(w0.size()) + 3u) & ~3u) * 2u;
    const uint32_t pad1 = ((uint32_t(w1.size()) + 3u) & ~3u) * 2u;
    PXG_CHECK_MSG(nPadded[0] == pad0 + pad1,
                  "midphasePairsNumPadded is %u, expected %u -- the radix sort needs "
                  "2x the 4-rounded count", nPadded[0], pad0 + pad1);

    // The ConvexMeshPair records: count is the per-manager pair count, and the
    // material pair is (convex, heightfield).
    auto readI = [&](uint32_t cm, size_t off) {
        int v; std::memcpy(&v, pair.data() + cm * 56 + off, 4); return v; };
    auto readU = [&](uint32_t cm, size_t off) {
        uint32_t v; std::memcpy(&v, pair.data() + cm * 56 + off, 4); return v; };
    PXG_CHECK_MSG(readI(0, 32) == int(w0.size()), "record 0 count %d", readI(0, 32));
    PXG_CHECK_MSG(readI(1, 32) == int(w1.size()), "record 1 count %d", readI(1, 32));
    PXG_CHECK_MSG(readI(0, 36) == 0 && readI(1, 36) == 1 && readI(2, 36) == 2,
                  "record cmIndex must be cmIdx");
    PXG_CHECK_MSG(readI(2, 32) == 0,
                  "manager 2 walked nothing, so its record count must be 0 (got %d) -- the record is still WRITTEN, which is what tells the consumer there is nothing here",
                  readI(2, 32));
    PXG_CHECK_MSG(readU(0, 48) == 7u && readU(0, 52) == 11u,
                  "record 0 materials (%u,%u), expected the BOX's 7 then the "
                  "heightfield's 11 -- in that order", readU(0, 48), readU(0, 52));
    PXG_CHECK_MSG(readU(1, 48) == 9u && readU(1, 52) == 11u,
                  "record 1 materials (%u,%u), expected the SPHERE's 9 then 11",
                  readU(1, 48), readU(1, 52));

    std::printf("    [ note ] %zu + %zu pairs over a %ux%u heightfield; manager 1's "
                "rectangle is decided by the fatten, manager 2 is rejected by the gate\n",
                w0.size(), w1.size(), HF_ROWS, HF_COLS);
    destroy(dev, b);
}

// ---------------------------------------------------------------------------
PXG_TEST(convexHfMidphaseCached)
{
    PxgSlangComputeDevice* dev = testDevice();
    Built b = build(dev, /*cached*/ true);
    dispatchMidphase(dev, b);

    const auto nPairs = downloadBuffer<uint32_t>(dev, b.pairsNum.h, 1);
    const auto nStack = downloadBuffer<uint32_t>(dev, b.stackOff.h, 1);
    const auto pair = downloadBuffer<uint8_t>(dev, b.pair.h, 3 * 56);

    PXG_CHECK_MSG(nPairs[0] == 0u && nStack[0] == 0u,
                  "the PCM gate must suppress the walk entirely (pairs %u, stack %u)",
                  nPairs[0], nStack[0]);

    auto readI = [&](uint32_t cm, size_t off) {
        int v; std::memcpy(&v, pair.data() + cm * 56 + off, 4); return v; };
    for (uint32_t cm = 0; cm < 3u; ++cm)
    {
        PXG_CHECK_MSG(readI(cm, 32) == CONVEX_TRIMESH_CACHED,
                      "record %u count is %d, expected the CACHED sentinel %d -- the "
                      "record is written even when nothing was walked, and that value "
                      "is how the consumer knows the stored manifold still stands",
                      cm, readI(cm, 32), CONVEX_TRIMESH_CACHED);
        PXG_CHECK_MSG(uint32_t(readI(cm, 28)) == 0xFFFFFFFFu,
                      "record %u startIndex is %d, expected the no-pairs sentinel",
                      cm, readI(cm, 28));
    }
    std::printf("    [ note ] both managers cached: no walk, no pairs, count == -1\n");
    destroy(dev, b);
}

// ---------------------------------------------------------------------------
PXG_TEST(convexHfMidphaseRotated)
{
    PxgSlangComputeDevice* dev = testDevice();
    Built b = build(dev, /*cached*/ false);

    // Turn the heightfield 90 degrees about y and re-upload the transform cache.
    // WITH AN IDENTITY HEIGHTFIELD every spelling of the bound conversion agrees,
    // so `Pairs` cannot see it: rotInv(q, c - p) and rotInv(q,c) + rotInv(q,-p)
    // are the same when q is identity, and quatRotate(q, e_i) equals the PxMat33
    // columns exactly.  This fixture is the one that separates them.
    // A 90-degree turn about y maps local (x,y,z) to world (z,y,-x), so the grid's
    // local x in [0,6] / z in [0,6] lands at world X in [1.5,7.5] and Z in
    // [-6.75,-0.75].  The bounds move with it -- the walk reads
    // bounds[transformCacheRef0], not the transform, so leaving them where they
    // were is what made the first attempt at this fixture assert nothing.
    const float s = 0.70710678f;
    std::vector<float> xf = { 0,0,0,1,  2.0f, 0.3f, -3.5f, 0,
                              0,0,0,1,  2.0f, 0.3f, -2.0f, 0,
                              0,0,0,1,  11.0f, 0.3f, -2.0f, 0,
                              0,s,0,s,  1.5f, 0.25f, -0.75f, 0 };
    dev->upload(b.xf.h, xf.data(), xf.size() * sizeof(float));

    const V3 boxMin{ 1.5f, -0.4f, -7.75f }, boxMax{ 5.5f, 1.0f, 0.25f };
    const V3 sphMin{ 1.51f, -0.4f, -3.74f }, sphMax{ 3.48f, 1.0f, -1.77f };
    const V3 offMin{ 10.0f, -0.4f, -3.0f }, offMax{ 12.0f, 1.0f, -1.0f };
    std::vector<float> bnd = { boxMin.x, boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z,
                               sphMin.x, sphMin.y, sphMin.z, sphMax.x, sphMax.y, sphMax.z,
                               offMin.x, offMin.y, offMin.z, offMax.x, offMax.y, offMax.z,
                               -1.0f, -1.0f, -1.0f, 13.0f, 1.0f, 1.0f };
    dev->upload(b.bnd.h, bnd.data(), bnd.size() * sizeof(float));

    dispatchMidphase(dev, b);

    const HfXform hfT{ 0, s, 0, s, 1.5f, 0.25f, -0.75f };
    const V3 hfScale{ 1.0f, HF_SCALE_Y, 1.0f };
    const std::vector<uint32_t> e0 =
        expectedTriangles(b.hfBlob, hfT, hfScale, boxMin, boxMax, 2.0f * CDIST);
    const std::vector<uint32_t> e1 =
        expectedTriangles(b.hfBlob, hfT, hfScale, sphMin, sphMax, 2.0f * CDIST);
    PXG_CHECK_MSG(!e0.empty(),
                  "the rotated placement must still overlap the grid, or this fixture "
                  "asserts nothing -- it went empty, so the transform under test is "
                  "untested");

    const auto stack = downloadBuffer<uint32_t>(dev, b.stack.h, 4u * 1024u);
    const auto nStack = downloadBuffer<uint32_t>(dev, b.stackOff.h, 1);
    std::vector<uint32_t> got0, got1;
    for (uint32_t i = 0; i < nStack[0]; ++i)
    {
        if (stack[4 * i] == 0u) got0.push_back(stack[4 * i + 1]);
        else got1.push_back(stack[4 * i + 1]);
    }
    std::sort(got0.begin(), got0.end());
    std::sort(got1.begin(), got1.end());
    std::vector<uint32_t> w0 = e0, w1 = e1;
    std::sort(w0.begin(), w0.end());
    std::sort(w1.begin(), w1.end());

    PXG_CHECK_MSG(got0 == w0, "rotated manager 0 emitted %zu triangles, expected %zu",
                  got0.size(), w0.size());
    PXG_CHECK_MSG(got1 == w1, "rotated manager 1 emitted %zu triangles, expected %zu",
                  got1.size(), w1.size());

    std::printf("    [ note ] heightfield turned 90 degrees about y: %zu + %zu pairs, "
                "bit-exact against the CUDA op order\n", w0.size(), w1.size());
    destroy(dev, b);
}

}  // namespace
