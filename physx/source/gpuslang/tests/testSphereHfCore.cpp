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

// Verification of pxgSlangSphereHfCore -- sphereHeightfieldNarrowphaseCore
// (convexHeightfield.cu:359-562), the SPHERE arm of sphereHeightfieldNarrowphase.
// P9.52a.
//
// The mirror below is written from the CUDA: getTriangle (heightfieldUtil.cuh:262,
// which the core calls at :400 and which is effect-identical to the
// convexHeightfield.cu:83 overload the P9.51b fixture was written from),
// distancePointTriangleSquared (sphereTriangle.cuh:41-140), the edge-concavity
// switch at :466-516 and the record tail at :539-556.
//
// ===========================================================================
// HOW THE CASES AIM AT A VORONOI REGION
// ===========================================================================
//
// Every branch of distancePointTriangleSquared is selected by d1..d6, and every
// one of those is a dot product with `ab` or `ac` -- both IN THE TRIANGLE'S PLANE.
// Displacing the sphere centre along the triangle NORMAL therefore cannot change
// which region is selected.  Each case names a region and gives two numbers: a
// lateral factor that pushes an in-plane point out of the triangle towards that
// region, and a height along the normal.  The height is what makes `dist >= 0`
// (:453) true or false and what sets how oblique the patch normal ends up, so it
// alone decides the 0.996 deferral -- which is exactly the separation of concerns
// the two knobs are for.
//
// The centre is then placed in WORLD space as `meshP + rotate(meshQ, centreLocal)`
// so that the shader's `rotInv(meshQ, sphereP - meshP)` reproduces it.  That
// round trip is not bit-exact, and it is not meant to be: the MIRROR reads the
// same world position out of the same transform cache and does the same rounding.
// What the construction buys is that the case lands in the region it claims.
//
// ===========================================================================
// THE TWO SYNTHETIC INPUTS, AND WHY THEY ARE HONEST
// ===========================================================================
//
// 1. AN ANISOTROPIC SPHERE SCALE.  The core reads `scale.scale.x` (:428); the
//    output stage of the same chain reads `scale.scale.y` for the same sphere
//    (convexMeshOutput.cu:314).  Both are right, because PxgNarrowphaseCore.cpp
//    :7508 fills a sphere's scale as `PxVec3(radius)` -- three equal components --
//    and the output stage is shared with the capsule, whose scale is
//    `(halfHeight, radius, radius)`.  With three equal components NOTHING can tell
//    the two readings apart, so shape 2 below carries x != y and two cases sit in
//    the gap between the two radii, where reading .y accepts a contact that .x
//    rejects.
//
// 2. A NEGATIVE COLUMN SCALE.  `PxHeightFieldGeometry::isValid` rejects it, so no
//    scene produces `wrongHanded` -- and the winding swap it drives is live code
//    reached by two different permutations.  Same argument as the P9.51b fixture's
//    shape 2.
//
// ===========================================================================
// WHERE THE WRITES LAND IS CHECKED, NOT ONLY WHAT THEY ARE
// ===========================================================================
//
// The four record arrays are seeded with sentinels and EVERY slot no case claims
// is re-checked afterwards.  Four of the tail's five writes are indexed by
// `pairOffset` and the fifth by `pairOffsetPadded`, the two differ per case, and
// a port that confuses them writes plausible values into the wrong records --
// which a value-only check cannot see.

#include "pxgTestFramework.h"
#include "pxgHeightfieldRef.h"
#include "pxgNpRefMath.h"

#include "geometry/PxGeometry.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

// The mirrors this fixture shares with the other heightfield fixtures live in
// pxgNpRefMath.h and pxgHeightfieldRef.h.  What stays here is what only this
// fixture has: the sphere-arm constants and distancePointTriangleSquared.
using namespace pxgtest::npref;
using namespace pxgtest::hfref;

// P9.57: FROM THE ENGINE ENUM, never a literal.  `PxGeometryType` grew
// eCONVEXCORE at position 4, which shifted eCONVEXMESH to 5, eTRIANGLEMESH to 8
// and eHEIGHTFIELD to 9; three fixtures and two shaders still carried the old
// numbers, and because a fixture and its shader carried the SAME wrong number
// the differential could not see it.  Taking the value from the header makes
// the compiler the oracle, exactly as this suite already does for the BV32
// stride.  tools/check_geom_type_constants.py keeps the Slang side honest.
static const uint32_t GEOM_SPHERE = uint32_t(physx::PxGeometryType::eSPHERE);
static const uint32_t GEOM_HEIGHTFIELD = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
static const uint32_t eV0 = 1u << 29, eV1 = 2u << 29, eV2 = 3u << 29;
static const uint32_t eE01 = 4u << 29, eE02 = 5u << 29, eE12 = 6u << 29;
static const uint32_t DEFERRED_CONTACT_MASK = 0x80000000u;
static const uint32_t NB_CONTACTS_SHIFT = 27u;
static const uint32_t FACE_CONTACT_FLAG = 0x80000000u;

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

struct CorePush
{
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, sphereTrimeshPair;
    uint64_t ni, maxDepth, interm, ordered, pairs, output;
    uint32_t numPairs, pad;
};

// ---------------------------------------------------------------------------
// distancePointTriangleSquared (sphereTriangle.cuh:41-140)
//
// STRICT comparisons in the three vertex regions and in `vc < 0`, the branch order
// V0, V1, V2 then E01, E02, E12, and a face region that SUMS the two scaled edge
// vectors and THEN divides -- PxVec3::operator/ being one reciprocal and three
// multiplies (PxVec3.h:218).  `edgeTol` is computed in DOUBLE because `1e-5` is an
// unsuffixed double literal (P9.55); it is dead on this path, because heightfield
// adjacency indices never carry bit 31 except as the BOUNDARY sentinel, which
// isEdgeNonconvex excludes -- so the `||` short-circuits before it is reached.
// ---------------------------------------------------------------------------
static float distancePointTriangleSquared(V3 p, V3 a, V3 b, V3 c, const uint32_t adj[3],
                                          V3& closestP, bool& generateContact,
                                          bool& faceContact, uint32_t& mask)
{
    faceContact = false;

    const V3 ab = sub(b, a);
    const V3 ac = sub(c, a);
    const V3 ap = sub(p, a);
    const float d1 = dot3(ab, ap);
    const float d2 = dot3(ac, ap);
    if (d1 < 0.0f && d2 < 0.0f)
    {
        mask = eV0;
        generateContact = (!isEdgeNonconvex(adj[0])) || (!isEdgeNonconvex(adj[2]));
        closestP = a;
        return dot3(ap, ap);
    }
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp);
    const float d4 = dot3(ac, bp);
    if (d3 > 0.0f && d4 < d3)
    {
        mask = eV1;
        generateContact = (!isEdgeNonconvex(adj[0])) || (!isEdgeNonconvex(adj[1]));
        closestP = b;
        return dot3(bp, bp);
    }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp);
    const float d6 = dot3(ac, cp);
    if (d6 > 0.0f && d5 < d6)
    {
        mask = eV2;
        generateContact = (!isEdgeNonconvex(adj[1])) || (!isEdgeNonconvex(adj[2]));
        closestP = c;
        return dot3(cp, cp);
    }

    const float vc = d1 * d4 - d3 * d2;
    const float vb = d5 * d2 - d1 * d6;
    const float va = d3 * d6 - d5 * d4;

    const float edgeTol = float(double(va + vb + vc) * 1e-5);

    if (vc < 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        mask = eE01;
        generateContact = !isEdgeNonconvex(adj[0]) || vc > -edgeTol;
        const float v = d1 / (d1 - d3);
        closestP = add(a, mul(ab, v));
        return dot3(sub(closestP, p), sub(closestP, p));
    }

    if (vb < 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        mask = eE02;
        generateContact = !isEdgeNonconvex(adj[2]) || vb > -edgeTol;
        const float w = d2 / (d2 - d6);
        closestP = add(a, mul(ac, w));
        return dot3(sub(closestP, p), sub(closestP, p));
    }

    if (va < 0.0f && d4 >= d3 && d5 >= d6)
    {
        mask = eE12;
        generateContact = !isEdgeNonconvex(adj[1]) || va > -edgeTol;
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closestP = add(b, mul(sub(c, b), w));
        return dot3(sub(closestP, p), sub(closestP, p));
    }

    generateContact = true;
    faceContact = true;
    mask = 0;

    // `a + (ab * vb + ac * vc) / (va + vb + vc)` -- SUM, then one reciprocal and
    // three multiplies.
    const float r = 1.0f / (va + vb + vc);
    closestP = add(a, mul(add(mul(ab, vb), mul(ac, vc)), r));
    return dot3(sub(closestP, p), sub(closestP, p));
}

// ---------------------------------------------------------------------------
// the case table
// ---------------------------------------------------------------------------
// R_AT_V0 puts the centre EXACTLY on vertex 0, which is the only way to reach
// `dist == 0` bit-exactly and therefore the only input that tells `dist >= 0`
// (:453) apart from `dist > 0`.  It needs an IDENTITY mesh transform: with
// q = (0,0,0,1) and p = 0 both quatRotate and quatRotateInv reduce to
// `v * (1 - 0) + u * 0 +- 0`, so the world round trip is exact and the shader's
// `centre - v0` is the true zero vector rather than a few ulp of one.
// (`a + n * 0` is also exact: hf2shapep can produce +0 but never -0, and
// x + (-0) == x for every such x.)
enum Region { R_FACE = 0, R_V0, R_V1, R_V2, R_E01, R_E02, R_E12, R_AT_V0 };

struct Spec
{
    uint32_t sphereShape;   // index into `shapes`
    uint32_t hfShape;       // index into `shapes`
    uint32_t meshXf;        // transform-cache slot of the heightfield
    uint32_t cdSphere;      // contact-distance slot for the sphere (== its xf slot)
    uint32_t tri;
    Region region;
    float lateral;          // how far out of the triangle, as a fraction
    float height;           // along the triangle normal; negative puts the centre behind it
    uint32_t gti;           // the globalThreadIndex the deferred record must carry
    const char* why;
};

struct Expected
{
    uint32_t nb, outMask, pairOffset, pairOffsetPadded;
    V3 contact;
    float sep;
    float maxDepth;
    uint32_t interm, ordered;
    V3 niNormal;
    uint32_t niIndex;
    // coverage bookkeeping
    uint32_t mask;
    bool faceContact, delayContacts, accepted, switchTaken, boundarySkip;
    float dist, sqDist, sqInflated, contactDist, cosTheta;
    // WHICH arm of the switch ran and what it answered.  The three arms have to be
    // covered SEPARATELY: each pairs a different vertex with a different adjacency
    // slot, and a `kept` answer in one arm says nothing about the other two.
    uint32_t armSlot;     // 0 none, 1 E01, 2 E12, 3 E02
    bool armKept;
};

PXG_TEST(sphereHfCore)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- two heightfields -------------------------------------------------
    // hfA: 5 rows x 4 columns, a RIDGED surface.  Coplanar neighbours would make
    // every edge arm of the switch answer the same way (a coplanar neighbour gives
    // dot == 0, which is not < 0, so the contact is REJECTED); the ridges make some
    // edges convex and keep both answers reachable.
    Hf hfA, hfB;
    hfA.rows = 5; hfA.cols = 4;
    {
        const int16_t hts[20] = {  0,  40,  40,   0,
                                  16,  96,  96,  16,
                                   0,  40,  40,   0,
                                 -24, -80, -80, -24,
                                   8,  24,  24,   8 };
        for (uint32_t i = 0; i < 20; ++i)
            hfA.samples.push_back(sampleWord(hts[i], uint8_t(1 + (i % 5)), (i % 3u) == 1u,
                                             uint8_t(2 + (i % 7))));
    }
    buildHf(hfA);

    // hfB: 4 rows x 5 columns -- rows and columns TRANSPOSED, so a port that reads
    // them the wrong way round indexes different vertices here.  Flat-ish with one
    // step, and driven through the NEGATIVE column scale.
    hfB.rows = 4; hfB.cols = 5;
    {
        const int16_t hts[20] = {  0,   0,   0,   0,  0,
                                   0,  64,  64,   0,  0,
                                   0,  64,  64,   0,  0,
                                   0,   0,   0,   0,  0 };
        for (uint32_t i = 0; i < 20; ++i)
            hfB.samples.push_back(sampleWord(hts[i], uint8_t(3 + (i % 4)), (i % 2u) == 0u,
                                             uint8_t(5 + (i % 3))));
    }
    buildHf(hfB);

    PxgBufferHandle bHfA = createBufferFrom(dev, hfA.blob);
    PxgBufferHandle bHfB = createBufferFrom(dev, hfB.blob);
    const uint64_t addrHfA = dev->getBufferDeviceAddress(bHfA);
    const uint64_t addrHfB = dev->getBufferDeviceAddress(bHfB);

    // ---- shapes ------------------------------------------------------------
    // 0,1: heightfields.  2,3,4: spheres.
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
    // 0: hfA.  Three different scale components and a NON-IDENTITY scale rotation
    //    the heightfield path must IGNORE (only scale.scale.xyz reach getTriangle).
    {
        const float a = -0.6f;
        setShape(shapes[0], { 1.0f, 0.03125f, 1.5f },
                 { 0.0f, std::sin(a * 0.5f), 0.0f, std::cos(a * 0.5f) }, addrHfA, GEOM_HEIGHTFIELD);
    }
    // 1: hfB with a NEGATIVE COLUMN SCALE -> wrongHanded, so both permutations fire.
    setShape(shapes[1], { 1.25f, 0.015625f, -0.8f }, { 0, 0, 0, 1 }, addrHfB, GEOM_HEIGHTFIELD);
    // 2: THE ANISOTROPIC SPHERE -- x != y, see the header.  A sphere's blob pointer
    //    is never dereferenced on this path; only the scale is read.
    setShape(shapes[2], { 0.35f, 0.75f, 0.55f }, { 0, 0, 0, 1 }, 0, GEOM_SPHERE);
    // 3, 4: ordinary isotropic spheres, as PxgNarrowphaseCore.cpp:7508 fills them.
    setShape(shapes[3], { 0.6f, 0.6f, 0.6f }, { 0, 0, 0, 1 }, 0, GEOM_SPHERE);
    setShape(shapes[4], { 0.28f, 0.28f, 0.28f }, { 0, 0, 0, 1 }, 0, GEOM_SPHERE);

    // ---- the case table ----------------------------------------------------
    // `meshXf` picks the heightfield's transform slot; the sphere gets a slot of
    // its own (see below), because its position is derived per case.
    const std::vector<Spec> specs = {
        // face contacts: patchNormal == triNormal, so cosTheta is 1 and nothing is
        // deferred; the ordered word gets its bit 31.
        { 3, 0, 0, 0, 10, R_FACE, 0.0f,  0.30f, 0x00010001u, "hfA interior face" },
        { 3, 0, 1, 0,  0, R_FACE, 0.0f,  0.05f, 0x00010002u, "hfA corner cell face, second transform" },
        { 4, 0, 0, 0, 21, R_FACE, 0.0f,  0.12f, 0x00010003u, "hfA face, small sphere" },
        // the same face from BEHIND: dist < 0 rejects however close it is.
        { 3, 0, 0, 0, 10, R_FACE, 0.0f, -0.30f, 0x00010004u, "behind the plane -> dist < 0" },
        // too far: sqDist >= sqInflatedRadius.
        { 4, 0, 0, 0, 10, R_FACE, 0.0f,  1.90f, 0x00010005u, "outside the inflated radius" },
        // vertex regions -- no switch arm, patchNormal is the centre-to-closest
        // direction, and the obliqueness defers the contact.
        { 3, 0, 0, 0, 10, R_V0,  0.45f, 0.10f, 0x00010006u, "vertex region V0" },
        { 3, 0, 0, 0, 10, R_V1,  0.22f, 0.10f, 0x00010007u, "vertex region V1" },
        { 3, 0, 0, 0, 10, R_V2,  0.22f, 0.10f, 0x00010008u, "vertex region V2" },
        // edge regions on an INTERIOR cell, so all three adjacency slots are real
        // triangles and the switch actually runs.
        { 3, 0, 0, 0, 10, R_E01, 0.30f, 0.10f, 0x00010009u, "edge E01, real adjacency" },
        { 3, 0, 0, 0, 10, R_E02, 0.30f, 0.10f, 0x0001000au, "edge E02, real adjacency" },
        { 3, 0, 0, 0, 10, R_E12, 0.30f, 0.10f, 0x0001000bu, "edge E12, real adjacency" },
        { 3, 0, 0, 0, 11, R_E01, 0.30f, 0.10f, 0x0001000cu, "edge E01 on the cell's second triangle" },
        { 3, 0, 0, 0,  9, R_E12, 0.30f, 0.10f, 0x0001000du, "edge E12 across a ridge" },
        { 3, 0, 0, 0, 17, R_E02, 0.30f, 0.10f, 0x0001000eu, "edge E02 across the valley" },
        // edge regions on the RIM, where the switch is skipped on a BOUNDARY
        // adjacency and the contact survives.
        { 3, 0, 0, 0,  0, R_E01, 0.30f, 0.10f, 0x0001000fu, "edge E01, BOUNDARY adjacency" },
        { 3, 0, 0, 0,  0, R_E02, 0.30f, 0.10f, 0x00010010u, "edge E02, BOUNDARY adjacency" },
        // a near-normal approach to an edge: the lateral offset is small enough
        // that cosTheta stays above 0.996 and the contact is NOT deferred.
        { 3, 0, 0, 0,  0, R_E01, 0.01f, 0.40f, 0x00010011u, "edge contact NOT deferred" },
        // the wrongHanded heightfield, driven through the second mesh transform.
        { 3, 1, 2, 0, 12, R_FACE, 0.0f, 0.20f, 0x00010012u, "wrongHanded face" },
        { 3, 1, 2, 0, 13, R_E01,  0.30f, 0.10f, 0x00010013u, "wrongHanded edge E01" },
        { 3, 1, 2, 0,  6, R_V1,   0.45f, 0.10f, 0x00010014u, "wrongHanded vertex V1" },
        // THE RADIUS PROBE.  Shape 2 has scale.x = 0.35 and scale.y = 0.75; these
        // two sit in the gap, so a port reading .y accepts what .x rejects.
        { 2, 0, 0, 0, 10, R_FACE, 0.0f, 0.50f, 0x00010015u, "radius probe: .x rejects, .y accepts" },
        { 2, 0, 0, 0, 21, R_FACE, 0.0f, 0.60f, 0x00010016u, "radius probe, second cell" },
        // and one the small radius DOES accept, so the probe is not one-sided.
        { 2, 0, 0, 0, 10, R_FACE, 0.0f, 0.20f, 0x00010017u, "radius probe: both accept" },
        // --- P9.52a sweep round 2: the cases the first sweep proved missing -----
        { 3, 0, 0, 0,  0, R_E12, 0.30f, 0.10f, 0x00010019u, "E12 with a BOUNDARY adj.z" },
        { 3, 0, 0, 0, 16, R_E02, 0.30f, 0.10f, 0x0001001au, "E02 with a real adj.x, BOUNDARY adj.z" },
        // Triangle 22 is the one E12 edge of hfA's tent whose neighbour is CONVEX, so
        // the arm ANSWERS KEEP.  Without it the E12 arm only ever dropped, and
        // `C6-eE12-tests-v2` survived the first sweep: v2 is shared with the
        // neighbour across E12, so the mutant's dot is zero to within rounding and
        // agrees with the real answer on every DROP.
        { 3, 0, 0, 0, 22, R_E12, 0.30f, 0.10f, 0x0001001fu, "E12 across a CONVEX ridge -> keep" },
        // an approach oblique enough to defer but far from perpendicular: the only
        // input that pins the 0.996 tolerance itself rather than its two extremes.
        { 3, 0, 0, 0,  0, R_E01, 0.10f, 0.10f, 0x00010021u, "cosTheta in the mid band" },
        // THE EXACT-ZERO PROBE: the centre sits on vertex 0 of triangle 10 through the
        // identity mesh transform, so `dist` is a true zero.  The point is then in the
        // FACE region (d1 = d2 = d3 = d4 = 0 makes vc and vb zero and va positive), so
        // it also drives sqDist == 0 and a face contact at zero separation.
        { 3, 0, 3, 0, 10, R_AT_V0, 0.0f, 0.0f, 0x00010018u, "centre exactly on v0: dist == 0" },
    };
    const uint32_t n = uint32_t(specs.size());

    // ---- transforms --------------------------------------------------------
    // Slots 0..2 are the three heightfield transforms; slot 3 + i is case i's
    // sphere.  EVERY rotation is non-identity: with the identity the mesh's
    // `transformInv` needs no rotating and the sphere's `rotateInv` on the output
    // normal is a no-op, and three separate mistakes become invisible.
    const uint32_t NUM_MESH_XF = 4;
    std::vector<float> xf((NUM_MESH_XF + n) * 8, 0.0f);
    auto setXf = [&](uint32_t i, V4 q, V3 p) {
        float* r = xf.data() + size_t(i) * 8;
        r[0] = q.x; r[1] = q.y; r[2] = q.z; r[3] = q.w;
        r[4] = p.x; r[5] = p.y; r[6] = p.z;
    };
    auto getXf = [&](uint32_t i, V4& q, V3& p) {
        const float* r = xf.data() + size_t(i) * 8;
        q = V4{ r[0], r[1], r[2], r[3] };
        p = V3{ r[4], r[5], r[6] };
    };
    { const float a = 0.7f;
      setXf(0, { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, { -1.0f, 2.0f, 0.5f }); }
    { const float a = -1.1f;
      setXf(1, { 0.0f, 0.0f, std::sin(a * 0.5f), std::cos(a * 0.5f) }, { 3.0f, 0.0f, -1.5f }); }
    { const float a = 1.3f;
      setXf(2, { 0.0f, std::sin(a * 0.5f), 0.0f, std::cos(a * 0.5f) }, { -0.5f, 1.25f, 2.0f }); }
    // Slot 3 is the ONE identity mesh transform, and only R_AT_V0 uses it: see the
    // comment on that region.  Every other case keeps a rotating, translating mesh,
    // because with the identity `transformInv` needs no rotating and three separate
    // mistakes in the prologue become invisible.
    setXf(3, { 0, 0, 0, 1 }, { 0.0f, 0.0f, 0.0f });

    // Place every sphere: pick the in-plane point for the case's region, lift it
    // along the triangle normal, and push it out to world space through the mesh
    // transform.  See the header for why the normal lift cannot move the region.
    for (uint32_t i = 0; i < n; ++i)
    {
        const Spec& s = specs[i];
        const GpuShape& hfs = shapes[s.hfShape];
        const Hf& hf = (hfs.hullLo == uint32_t(addrHfA & 0xffffffffu)) ? hfA : hfB;

        V3 a, b, c; uint32_t adj[3];
        getTriangle(hf, s.tri, V3{ hfs.sx, hfs.sy, hfs.sz }, a, b, c, adj);
        const V3 nrm = normalized(cross3(sub(b, a), sub(c, a)));

        V3 q;
        switch (s.region)
        {
        case R_FACE: q = mul(add(add(a, b), c), 1.0f / 3.0f); break;
        case R_V0:   q = add(a, mul(sub(a, mul(add(b, c), 0.5f)), s.lateral)); break;
        case R_V1:   q = add(b, mul(sub(b, mul(add(a, c), 0.5f)), s.lateral)); break;
        case R_V2:   q = add(c, mul(sub(c, mul(add(a, b), 0.5f)), s.lateral)); break;
        case R_E01:  { const V3 m = mul(add(a, b), 0.5f); q = add(m, mul(sub(m, c), s.lateral)); } break;
        case R_E02:  { const V3 m = mul(add(a, c), 0.5f); q = add(m, mul(sub(m, b), s.lateral)); } break;
        case R_AT_V0: q = a; break;
        default:     { const V3 m = mul(add(b, c), 0.5f); q = add(m, mul(sub(m, a), s.lateral)); } break;
        }
        const V3 centreLocal = add(q, mul(nrm, s.height));

        V4 mq; V3 mp;
        getXf(s.meshXf, mq, mp);
        const V3 world = add(mp, quatRotate(mq, centreLocal));

        // A rotation of its own per case, so no two spheres share an output frame.
        const float ang = 0.31f * float(i + 1) - 1.0f;
        const V4 sq = { std::sin(ang * 0.5f) * 0.6f, std::sin(ang * 0.5f) * 0.8f, 0.0f,
                        std::cos(ang * 0.5f) };
        setXf(NUM_MESH_XF + i, sq, world);
    }

    // ---- contact distances -------------------------------------------------
    // Every slot different, so indexing the array by the SHAPE refs instead of the
    // TRANSFORM refs picks up a different number.
    std::vector<float> cdist(NUM_MESH_XF + n, 0.0f);
    for (uint32_t i = 0; i < cdist.size(); ++i)
        cdist[i] = 0.003f + 0.0017f * float(i);

    // ---- contact managers, one per case ------------------------------------
    // THE TRANSFORM REFS ARE NOT THE SHAPE REFS: while they match, indexing the
    // contact distance by the wrong one is invisible.
    std::vector<CmInput> cms(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        cms[i].shapeRef0 = specs[i].sphereShape;
        cms[i].shapeRef1 = specs[i].hfShape;
        cms[i].tcRef0 = NUM_MESH_XF + i;
        cms[i].tcRef1 = specs[i].meshXf;
    }

    // ---- ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40 -------
    // @32 (`count`) and @36 (`cmIndex`) carry decoys: a port that reads the
    // declaration's trailing RUNNING-END offsets as field offsets lands on them,
    // and both are small non-negative integers, so nothing faults.
    std::vector<uint8_t> pairBlob(size_t(n) * 56, 0);
    std::vector<uint32_t> startIdx(n), roundedIdx(n), testOff(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        startIdx[i] = 10u * i;
        roundedIdx[i] = 200u + 7u * i;
        testOff[i] = i % 5u;
        uint8_t* r = pairBlob.data() + size_t(i) * 56;
        const uint32_t count = 3u + i, cmIndex = 40u + i;
        std::memcpy(r + 28, &startIdx[i], 4);
        std::memcpy(r + 32, &count, 4);
        std::memcpy(r + 36, &cmIndex, 4);
        std::memcpy(r + 40, &roundedIdx[i], 4);
    }

    struct Pair { uint32_t cm, tri, off, gti; };
    std::vector<Pair> pairs(n);
    for (uint32_t i = 0; i < n; ++i)
        pairs[i] = { i, specs[i].tri, testOff[i], specs[i].gti };

    // ---- the mirror --------------------------------------------------------
    std::vector<Expected> exp(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const Spec& s = specs[i];
        const CmInput& cm = cms[i];
        const GpuShape& sphereShape = shapes[cm.shapeRef0];
        const GpuShape& hfs = shapes[cm.shapeRef1];
        const Hf& hf = (hfs.hullLo == uint32_t(addrHfA & 0xffffffffu)) ? hfA : hfB;
        const V3 hfScale = { hfs.sx, hfs.sy, hfs.sz };

        V4 sphereQ, meshQ; V3 sphereP, meshP;
        getXf(cm.tcRef0, sphereQ, sphereP);
        getXf(cm.tcRef1, meshQ, meshP);

        V3 v0, v1, v2; uint32_t adj[3];
        getTriangle(hf, s.tri, hfScale, v0, v1, v2, adj);
        const V3 triNormal = normalized(cross3(sub(v1, v0), sub(v2, v0)));

        const uint32_t pairOffset = startIdx[i] + testOff[i];
        const uint32_t pairOffsetPadded = roundedIdx[i] + testOff[i];

        const float contactDist = cdist[cm.tcRef0] + cdist[cm.tcRef1];
        const float sphereRadius = sphereShape.sx;     // .x, not .y
        const V3 centre = quatRotateInv(meshQ, sub(sphereP, meshP));
        const float inflatedRadius = sphereRadius + contactDist;
        const float dist = dot3(sub(centre, v0), triNormal);

        float separation = MAX_F32;
        uint32_t nb = 0;
        bool delayContacts = false;
        V3 patchNormal = { 0, 0, 0 };
        uint32_t outMask = 0;
        V3 outContact = { 0, 0, 0 };
        float outSep = 0.0f;

        V3 closestP; bool generateContact = false, faceContact = false; uint32_t mask = 0;
        const float sqDist = distancePointTriangleSquared(centre, v0, v1, v2, adj,
                                                          closestP, generateContact,
                                                          faceContact, mask);
        const float sqInflated = inflatedRadius * inflatedRadius;

        bool switchTaken = false, boundarySkip = false;
        uint32_t armSlot = 0u;
        bool armKept = false;
        float cosTheta = 0.0f;
        if (dist >= 0.0f && sqInflated > sqDist && generateContact)
        {
            auto arm = [&](uint32_t adjIdx, V3 opposite) {
                if (adjIdx == HF_BOUNDARY) { boundarySkip = true; return; }
                switchTaken = true;
                V3 a0, a1, a2;
                getTriangleNoAdj(hf, adjIdx, hfScale, a0, a1, a2);
                const V3 nAdj = normalized(cross3(sub(a1, a0), sub(a2, a0)));
                generateContact = dot3(nAdj, sub(opposite, a0)) < 0.0f;
                armKept = generateContact;
            };
            if (mask == eE01)      { armSlot = 1u; arm(adj[0], v2); }
            else if (mask == eE12) { armSlot = 2u; arm(adj[1], v0); }
            else if (mask == eE02) { armSlot = 3u; arm(adj[2], v1); }

            if (generateContact)
            {
                patchNormal = triNormal;
                if (!faceContact)
                    patchNormal = normalized(sub(centre, closestP));

                cosTheta = dot3(patchNormal, triNormal);
                delayContacts = cosTheta <= 0.996f;
                if (delayContacts)
                    outMask = s.gti | mask;

                nb = 1;
                separation = std::sqrt(sqDist);
                outContact = V3{ 0, 0, 0 };
                outSep = separation;
            }
        }

        Expected& e = exp[i];
        e.nb = nb;
        e.outMask = outMask;
        e.pairOffset = pairOffset;
        e.pairOffsetPadded = pairOffsetPadded;
        e.contact = outContact;
        e.sep = outSep;
        e.maxDepth = separation;
        e.interm = s.tri;
        e.ordered = (nb != 0 && !delayContacts) ? (FACE_CONTACT_FLAG | s.tri) : s.tri;
        {
            const uint32_t delayMask = delayContacts ? DEFERRED_CONTACT_MASK : 0u;
            e.niNormal = quatRotateInv(sphereQ, quatRotate(meshQ, neg(patchNormal)));
            e.niIndex = delayMask + (nb << NB_CONTACTS_SHIFT) + s.tri;
        }
        e.mask = mask;
        e.faceContact = faceContact;
        e.delayContacts = delayContacts;
        e.accepted = nb != 0;
        e.switchTaken = switchTaken;
        e.boundarySkip = boundarySkip;
        e.dist = dist;
        e.cosTheta = cosTheta;
        e.armSlot = armSlot;
        e.armKept = armKept;
        e.sqDist = sqDist;
        e.sqInflated = sqInflated;
        e.contactDist = contactDist;
    }

    // ---- record arrays, seeded with sentinels ------------------------------
    uint32_t maxOffset = 0, maxPadded = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        maxOffset = std::max(maxOffset, exp[i].pairOffset);
        maxPadded = std::max(maxPadded, exp[i].pairOffsetPadded);
    }
    // Sized to the LARGER of the two index spaces, and deliberately: four of the
    // tail's five writes are indexed by `pairOffset` and the fifth by
    // `pairOffsetPadded`, and a port that swaps them must land on a checked sentinel
    // slot rather than off the end of a buffer -- an out-of-bounds write is a fault
    // or a corruption somewhere else, which is a worse signal than a wrong value in
    // a slot nobody claimed.
    const uint32_t slots = (maxOffset > maxPadded ? maxOffset : maxPadded) + 1u;
    const uint32_t niWords = 4u * slots;
    const uint32_t recCount = slots;
    const uint32_t ordCount = slots;

    const uint32_t U_SENTINEL = 0xdeadbeefu;
    const float F_SENTINEL = -98765.0f;
    std::vector<uint32_t> niInit(niWords, U_SENTINEL);
    std::vector<float> mdInit(recCount, F_SENTINEL);
    std::vector<uint32_t> imInit(recCount, U_SENTINEL);
    std::vector<uint32_t> ordInit(ordCount, U_SENTINEL);

    PxgBufferHandle bShapes = createBufferFrom(dev, shapes);
    PxgBufferHandle bXf = createBufferFrom(dev, xf);
    PxgBufferHandle bCd = createBufferFrom(dev, cdist);
    PxgBufferHandle bCm = createBufferFrom(dev, cms);
    PxgBufferHandle bPairBlob = createBufferFrom(dev, pairBlob);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairs);
    PxgBufferHandle bNi = createBufferFrom(dev, niInit);
    PxgBufferHandle bMd = createBufferFrom(dev, mdInit);
    PxgBufferHandle bIm = createBufferFrom(dev, imInit);
    PxgBufferHandle bOrd = createBufferFrom(dev, ordInit);
    std::vector<uint32_t> outInit(size_t(n) * 8, U_SENTINEL);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CorePush push{};
    push.cmInputs = dev->getBufferDeviceAddress(bCm);
    push.transformCache = dev->getBufferDeviceAddress(bXf);
    push.contactDistance = dev->getBufferDeviceAddress(bCd);
    push.gpuShapes = dev->getBufferDeviceAddress(bShapes);
    push.sphereTrimeshPair = dev->getBufferDeviceAddress(bPairBlob);
    push.ni = dev->getBufferDeviceAddress(bNi);
    push.maxDepth = dev->getBufferDeviceAddress(bMd);
    push.interm = dev->getBufferDeviceAddress(bIm);
    push.ordered = dev->getBufferDeviceAddress(bOrd);
    push.pairs = dev->getBufferDeviceAddress(bPairs);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numPairs = n;

    {
        Kernel k("sphereHfCoreBda.spv", "sphereHfCoreLaunch", {}, sizeof(CorePush));
        k.dispatch((n + 63u) / 64u, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto gotOut = downloadBuffer<uint32_t>(dev, bOut, outInit.size());
    const auto gotNi = downloadBuffer<uint32_t>(dev, bNi, niWords);
    const auto gotMd = downloadBuffer<uint32_t>(dev, bMd, recCount);
    const auto gotIm = downloadBuffer<uint32_t>(dev, bIm, recCount);
    const auto gotOrd = downloadBuffer<uint32_t>(dev, bOrd, ordCount);

    size_t bad = 0;
    auto bitsF = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    auto chkU = [&](uint32_t i, const char* w, uint32_t g, uint32_t e) {
        if (g != e) { ++bad; if (bad <= 24)
            std::printf("  [case %u %s] %s: got=0x%08x exp=0x%08x\n", i, specs[i].why, w, g, e); } };
    auto chkF = [&](uint32_t i, const char* w, uint32_t gBits, float e) {
        if (gBits != bitsF(e)) { float g; std::memcpy(&g, &gBits, 4); ++bad; if (bad <= 24)
            std::printf("  [case %u %s] %s: got=%.9g exp=%.9g\n", i, specs[i].why, w, g, e); } };

    size_t sawFace = 0, sawV = 0, sawE = 0, sawAccepted = 0, sawRejected = 0;
    size_t sawDelayed = 0, sawNotDelayed = 0, sawSwitchKept = 0, sawSwitchDropped = 0;
    size_t sawBoundarySkip = 0, sawWrongHanded = 0, sawDistNegative = 0, sawTooFar = 0;
    size_t sawRadiusGap = 0, sawSecondTri = 0, sawSharedDiagonal = 0, sawUnsharedDiagonal = 0;
    size_t sawNegativeHeight = 0;
    size_t sawE01Kept = 0, sawE12Kept = 0, sawE02Kept = 0;
    size_t sawE01BoundarySkip = 0, sawE12ZOnlyBoundary = 0, sawE02ZOnlyBoundary = 0;
    size_t sawCosMidBand = 0;

    for (uint32_t i = 0; i < n; ++i)
    {
        const Expected& e = exp[i];
        const uint32_t* r = gotOut.data() + size_t(i) * 8;

        chkU(i, "nbOutputContacts", r[0], e.nb);
        chkU(i, "outMask", r[1], e.outMask);
        chkU(i, "pairOffset", r[2], e.pairOffset);
        chkF(i, "contact.x", r[4], e.contact.x);
        chkF(i, "contact.y", r[5], e.contact.y);
        chkF(i, "contact.z", r[6], e.contact.z);
        chkF(i, "outSep", r[7], e.sep);

        chkF(i, "maxDepth", gotMd[e.pairOffset], e.maxDepth);
        chkU(i, "interm", gotIm[e.pairOffset], e.interm);
        chkU(i, "ordered", gotOrd[e.pairOffsetPadded], e.ordered);
        chkF(i, "ni.normal.x", gotNi[4 * e.pairOffset + 0], e.niNormal.x);
        chkF(i, "ni.normal.y", gotNi[4 * e.pairOffset + 1], e.niNormal.y);
        chkF(i, "ni.normal.z", gotNi[4 * e.pairOffset + 2], e.niNormal.z);
        chkU(i, "ni.index", gotNi[4 * e.pairOffset + 3], e.niIndex);

        // ---- coverage -------------------------------------------------------
        if (e.mask == 0 && e.faceContact) ++sawFace;
        if (e.mask == eV0 || e.mask == eV1 || e.mask == eV2) ++sawV;
        if (e.mask == eE01 || e.mask == eE02 || e.mask == eE12) ++sawE;
        if (e.accepted) ++sawAccepted; else ++sawRejected;
        if (e.accepted && e.delayContacts) ++sawDelayed;
        if (e.accepted && !e.delayContacts && !e.faceContact) ++sawNotDelayed;
        if (e.switchTaken && e.accepted) ++sawSwitchKept;
        if (e.switchTaken && !e.accepted) ++sawSwitchDropped;
        if (e.boundarySkip) ++sawBoundarySkip;
        if (e.switchTaken && e.armKept)
        {
            if (e.armSlot == 1u) ++sawE01Kept;
            else if (e.armSlot == 2u) ++sawE12Kept;
            else if (e.armSlot == 3u) ++sawE02Kept;
        }
        if (e.accepted && e.cosTheta > 0.5f && e.cosTheta <= 0.996f) ++sawCosMidBand;
        {
            // WHICH adjacency slot is the sentinel, not merely that one is.  Each arm
            // guards a DIFFERENT slot, so a mutant that guards the wrong one is only
            // visible where exactly one of them is BOUNDARY.
            const GpuShape& hfsx = shapes[specs[i].hfShape];
            const Hf& hfx = (hfsx.hullLo == uint32_t(addrHfA & 0xffffffffu)) ? hfA : hfB;
            V3 w0, w1, w2; uint32_t adjx[3];
            getTriangle(hfx, specs[i].tri, V3{ hfsx.sx, hfsx.sy, hfsx.sz }, w0, w1, w2, adjx);
            if (e.mask == eE01 && adjx[0] == HF_BOUNDARY) ++sawE01BoundarySkip;
            if (e.mask == eE12 && adjx[1] != HF_BOUNDARY && adjx[2] == HF_BOUNDARY)
                ++sawE12ZOnlyBoundary;
            if (e.mask == eE02 && adjx[0] != HF_BOUNDARY && adjx[2] == HF_BOUNDARY)
                ++sawE02ZOnlyBoundary;
        }
        if (e.dist < 0.0f) ++sawDistNegative;
        if (e.dist >= 0.0f && !(e.sqInflated > e.sqDist)) ++sawTooFar;
        {
            const GpuShape& hfs = shapes[specs[i].hfShape];
            if (hfs.sx < 0.0f || hfs.sz < 0.0f) ++sawWrongHanded;
            const GpuShape& sp = shapes[specs[i].sphereShape];
            // The gap between the two candidate radii: reading .y would accept
            // where .x rejects (or the other way round).
            if (sp.sx != sp.sy)
            {
                const float inflY = sp.sy + e.contactDist;
                const bool acceptX = e.dist >= 0.0f && e.sqInflated > e.sqDist;
                const bool acceptY = e.dist >= 0.0f && (inflY * inflY) > e.sqDist;
                if (acceptX != acceptY) ++sawRadiusGap;
            }
            const Hf& hf = (hfs.hullLo == uint32_t(addrHfA & 0xffffffffu)) ? hfA : hfB;
            if ((specs[i].tri & 1u) != 0u) ++sawSecondTri;
            if (zerothVertexShared(hf, specs[i].tri >> 1)) ++sawSharedDiagonal;
            else ++sawUnsharedDiagonal;
            uint32_t i0, i1, i2;
            triVertexIndices(hf, specs[i].tri, i0, i1, i2);
            if (hfHeight(hf, i0) < 0.0f || hfHeight(hf, i1) < 0.0f || hfHeight(hf, i2) < 0.0f)
                ++sawNegativeHeight;
        }
    }

    // ---- nothing may be written outside the claimed slots ------------------
    {
        std::vector<bool> claimed(recCount, false), claimedOrd(ordCount, false);
        for (uint32_t i = 0; i < n; ++i)
        {
            claimed[exp[i].pairOffset] = true;
            claimedOrd[exp[i].pairOffsetPadded] = true;
        }
        size_t stray = 0;
        for (uint32_t k = 0; k < recCount; ++k)
        {
            if (claimed[k]) continue;
            if (gotMd[k] != bitsF(F_SENTINEL)) ++stray;
            if (gotIm[k] != U_SENTINEL) ++stray;
            for (uint32_t w = 0; w < 4; ++w)
                if (gotNi[4 * k + w] != U_SENTINEL) ++stray;
        }
        for (uint32_t k = 0; k < ordCount; ++k)
            if (!claimedOrd[k] && gotOrd[k] != U_SENTINEL) ++stray;
        if (stray != 0)
        { bad += stray; std::printf("  %zu record slots written outside their pair offset\n", stray); }
    }

    dev->destroyBuffer(bHfA); dev->destroyBuffer(bHfB); dev->destroyBuffer(bShapes);
    dev->destroyBuffer(bXf); dev->destroyBuffer(bCd); dev->destroyBuffer(bCm);
    dev->destroyBuffer(bPairBlob); dev->destroyBuffer(bPairs);
    dev->destroyBuffer(bNi); dev->destroyBuffer(bMd); dev->destroyBuffer(bIm);
    dev->destroyBuffer(bOrd); dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(bad == 0, "sphereHfCore: %zu mismatches", bad);

    // Each of these guards a branch the mirror distinguishes.  Without them an
    // edit to the case table could quietly stop reaching a branch and the fixture
    // would still pass -- the failure mode that made the first P9.51b sweeps report
    // survivors caused by the fixture rather than by the shader.
    PXG_CHECK_MSG(sawFace > 0, "no case lands in the FACE region");
    PXG_CHECK_MSG(sawV > 0, "no case lands in a VERTEX region");
    PXG_CHECK_MSG(sawE > 0, "no case lands in an EDGE region");
    PXG_CHECK_MSG(sawAccepted > 0, "no case generates a contact");
    PXG_CHECK_MSG(sawRejected > 0, "no case is rejected");
    PXG_CHECK_MSG(sawDelayed > 0, "no case is DEFERRED (the 0.996 arm and outMask are untested)");
    PXG_CHECK_MSG(sawNotDelayed > 0,
                  "no NON-FACE case survives the 0.996 test -- the deferral would then be "
                  "indistinguishable from `not a face contact`");
    PXG_CHECK_MSG(sawE01Kept > 0,
                  "no E01 case has a CONVEX neighbour -- the arm's `keep` answer is untested, "
                  "and the vertex it tests (v2) is then indistinguishable from the two that "
                  "lie ON the neighbour's plane");
    PXG_CHECK_MSG(sawE12Kept > 0, "no E12 case has a CONVEX neighbour (see the E01 note)");
    PXG_CHECK_MSG(sawE02Kept > 0, "no E02 case has a CONVEX neighbour (see the E01 note)");
    PXG_CHECK_MSG(sawE01BoundarySkip > 0,
                  "no E01 case has adj.x == BOUNDARY -- the port's deliberate guard is untested");
    PXG_CHECK_MSG(sawE12ZOnlyBoundary > 0,
                  "no E12 case has a REAL adj.y and a BOUNDARY adj.z -- an arm that fetched the "
                  "wrong slot would then decode a valid triangle either way");
    PXG_CHECK_MSG(sawE02ZOnlyBoundary > 0,
                  "no E02 case has a REAL adj.x and a BOUNDARY adj.z -- a guard that tested the "
                  "wrong slot would then agree with the right one");
    PXG_CHECK_MSG(sawCosMidBand > 0,
                  "no accepted contact has cosTheta between 0.5 and 0.996 -- the tolerance itself "
                  "is then unpinned, only its two extremes are");
    PXG_CHECK_MSG(sawSwitchKept > 0,
                  "no edge case has a CONVEX neighbour -- the switch's `keep` answer is untested");
    PXG_CHECK_MSG(sawSwitchDropped > 0,
                  "no edge case has a COPLANAR or concave neighbour -- the switch's `drop` answer "
                  "is untested");
    PXG_CHECK_MSG(sawBoundarySkip > 0,
                  "no edge case sits on a BOUNDARY adjacency -- the port's deliberate divergence "
                  "from the .cu's unguarded getTriangle is then untested");
    PXG_CHECK_MSG(sawDistNegative > 0, "no case sits BEHIND the triangle plane (dist < 0 untested)");
    PXG_CHECK_MSG(sawTooFar > 0, "no case is outside the inflated radius");
    PXG_CHECK_MSG(sawRadiusGap > 0,
                  "no case sits in the gap between scale.x and scale.y -- which component the "
                  "radius comes from is then untested");
    PXG_CHECK_MSG(sawWrongHanded > 0, "no case uses a negative scale (wrongHanded untested)");
    PXG_CHECK_MSG(sawSecondTri > 0, "no case is a cell's SECOND triangle");
    PXG_CHECK_MSG(sawSharedDiagonal > 0, "no case sits on a tessellation-flagged cell");
    PXG_CHECK_MSG(sawUnsharedDiagonal > 0, "no case sits on an unflagged cell");
    PXG_CHECK_MSG(sawNegativeHeight > 0,
                  "no case reads a NEGATIVE height (the 16-bit sign extension is untested)");
}

} // namespace
