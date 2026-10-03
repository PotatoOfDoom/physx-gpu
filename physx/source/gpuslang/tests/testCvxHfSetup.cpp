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

// Verification of pxgSlangCvxHfSetup -- the prologue of
// convexHeightfieldNarrowphaseCore (convexHeightfield.cu:120-269), which is what
// `convexHfCoreBda` computes before it hands off to the SAT.
//
// The mirror below is written from the CUDA, not from the Slang: getTriangle
// (convexHeightfield.cu:83-116), getTriangleVertexIndices and
// getTriangleAdjacencyIndices (heightfieldUtil.cuh:120-250), hf2shapep, and the
// normal/vertex transforms at :136-155.
//
// ===========================================================================
// THE INPUT THIS FIXTURE EXISTS TO PRESENT
// ===========================================================================
//
// `chSetup` passes CT_NO_SWAP where `ctSetup` passes CT_GEOM_TRIANGLEMESH,
// because convexHeightfield.cu reads slot 0 as the convex unconditionally
// (:211, :214) while convexMesh.cu:135 swaps.  The obvious port defect is to copy
// the sibling and keep its swap -- and NO REALISTIC PAIR CAN SEE THAT, because a
// convex-vs-heightfield pair never carries a shape of type eTRIANGLEMESH in
// slot 0.  That is exactly why `H1-trimesh-swap-added` survives the P9.51a
// midphase sweep.
//
// THE SWAP PROBE below presents it anyway, with an input the engine does not
// produce: a
// shape whose `type` is eTRIANGLEMESH but whose blob is a VALID HEIGHTFIELD, in
// slot 0, against a second valid heightfield in slot 1.  The swap decision reads
// only `type`, so the decision is exercised exactly; and because both blobs are
// well-formed heightfields, BOTH branches produce readable output that differs in
// every field, instead of one branch reading a convex hull as a sample grid and
// reporting whatever garbage that yields.  A synthetic input is the honest way to
// test a branch upstream cannot reach -- the alternative is to leave it untested
// and call that coverage.

#include "pxgTestFramework.h"

#include "geometry/PxGeometry.h"
#include "foundation/PxQuat.h"

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
// PxVec3::getNormalized (PxVec3.h:286) -- one reciprocal, three multiplies.
static V3 normalized(V3 v)
{
    const float m = std::sqrt(dot3(v, v));
    if (!(m > 0.0f)) return V3{ 0, 0, 0 };
    const float r = 1.0f / m;
    return V3{ v.x * r, v.y * r, v.z * r };
}

// P9.57: FROM THE ENGINE ENUM, never a literal.  `PxGeometryType` grew
// eCONVEXCORE at position 4, which shifted eCONVEXMESH to 5, eTRIANGLEMESH to 8
// and eHEIGHTFIELD to 9; three fixtures and two shaders still carried the old
// numbers, and because a fixture and its shader carried the SAME wrong number
// the differential could not see it.  Taking the value from the header makes
// the compiler the oracle, exactly as this suite already does for the BV32
// stride.  tools/check_geom_type_constants.py keeps the Slang side honest.
static const uint32_t GEOM_TRIANGLEMESH = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
static const uint32_t GEOM_CONVEXMESH = uint32_t(physx::PxGeometryType::eCONVEXMESH);
static const uint32_t GEOM_HEIGHTFIELD = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
static const uint32_t HF_BOUNDARY = 0xFFFFFFFFu;

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

// ---------------------------------------------------------------------------
// The heightfield blob: [0] rows, [1] columns, then one 32-bit
// PxHeightFieldSample per vertex -- PxI16 height @0, materialIndex0 @2,
// materialIndex1 @3, with bit 23 (materialIndex0's high bit) the TESSELLATION
// flag that picks the quad's diagonal.
// ---------------------------------------------------------------------------
struct Hf
{
    uint32_t rows = 0, cols = 0;
    std::vector<uint32_t> samples;
    std::vector<uint32_t> blob;
};

static uint32_t sampleWord(int16_t height, uint8_t mat0, bool tessellationFlag, uint8_t mat1)
{
    return uint32_t(uint16_t(height))
         | (uint32_t(mat0 & 0x7f) << 16)
         | (tessellationFlag ? (1u << 23) : 0u)
         | (uint32_t(mat1 & 0x7f) << 24);
}

static void buildHf(Hf& h)
{
    h.blob.clear();
    h.blob.push_back(h.rows);
    h.blob.push_back(h.cols);
    for (uint32_t s : h.samples) h.blob.push_back(s);
}

static bool zerothVertexShared(const Hf& h, uint32_t cell)
{ return (h.samples[cell] & 0x800000u) != 0u; }

// The height is SIGNED 16-bit.  Reading the raw low half turns -300 into 65236.
static float hfHeight(const Hf& h, uint32_t vi)
{ return float(int16_t(uint16_t(h.samples[vi] & 0xffffu))); }

static V3 hfGetVertex(const Hf& h, uint32_t vi)
{ return V3{ float(vi / h.cols), hfHeight(h, vi), float(vi % h.cols) }; }

// getTriangleVertexIndices (heightfieldUtil.cuh:120-165)
static void triVertexIndices(const Hf& h, uint32_t t, uint32_t& i0, uint32_t& i1, uint32_t& i2)
{
    const uint32_t cell = t >> 1;
    const bool firstTri = (t & 1u) == 0u;
    const uint32_t c = h.cols;
    if (zerothVertexShared(h, cell))
    {
        if (firstTri) { i0 = cell + c; i1 = cell; i2 = cell + c + 1u; }
        else          { i0 = cell + 1u; i1 = cell + c + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + c; }
        else          { i0 = cell + c + 1u; i1 = cell + c; i2 = cell + 1u; }
    }
}

// getTriangleAdjacencyIndices (heightfieldUtil.cuh:167-250)
static void triAdjacency(const Hf& h, uint32_t t, uint32_t& a0, uint32_t& a1, uint32_t& a2)
{
    const uint32_t cell = t >> 1;
    const bool firstTri = (t & 1u) == 0u;
    const uint32_t c = h.cols, r = h.rows;
    a0 = HF_BOUNDARY;
    a2 = HF_BOUNDARY;
    if (zerothVertexShared(h, cell))
    {
        if (firstTri)
        {
            a1 = t + 1u;
            if (cell % c != 0u)        a0 = t - 1u;
            if (cell / c != r - 2u)    a2 = ((cell + c) * 2u) + 1u;
        }
        else
        {
            a1 = t - 1u;
            if (cell % c < c - 2u)     a0 = t + 1u;
            if (cell >= c - 1u)        a2 = (cell - c) * 2u;
        }
    }
    else
    {
        if (firstTri)
        {
            a1 = t + 1u;
            if (cell >= c - 1u)        a0 = ((cell - c) * 2u) + 1u;
            if (cell % c != 0u)        a2 = t - 1u;
        }
        else
        {
            a1 = t - 1u;
            if (cell / c != r - 2u)    a0 = (cell + c) * 2u;
            if (cell % c < c - 2u)     a2 = t + 1u;
        }
    }
}

// getTriangle (convexHeightfield.cu:83-116).  The handedness vector is dead; only
// the `wrongHanded` xor survives, and it permutes the vertices as [0, 1+wh, 2-wh]
// and the adjacency as [wh?2:0, 1, wh?0:2] -- two DIFFERENT permutations.
static void getTriangle(const Hf& h, uint32_t t, V3 scale,
                        V3& v0, V3& v1, V3& v2, uint32_t adj[3], bool& wrongHanded)
{
    wrongHanded = false;
    if (scale.z < 0.0f) wrongHanded = !wrongHanded;
    if (scale.x < 0.0f) wrongHanded = !wrongHanded;
    const uint32_t wh = wrongHanded ? 1u : 0u;

    uint32_t i0, i1, i2;
    triVertexIndices(h, t, i0, i1, i2);
    uint32_t tv[3];
    tv[0] = i0; tv[1u + wh] = i1; tv[2u - wh] = i2;

    uint32_t a0, a1, a2;
    triAdjacency(h, t, a0, a1, a2);
    adj[wh ? 2u : 0u] = a0; adj[1] = a1; adj[wh ? 0u : 2u] = a2;

    V3* out[3] = { &v0, &v1, &v2 };
    for (uint32_t vi = 0; vi < 3; ++vi)
    {
        const V3 g = hfGetVertex(h, tv[vi]);
        *out[vi] = V3{ g.x * scale.x, g.y * scale.y, g.z * scale.z };   // hf2shapep
    }
}

struct Pair { uint32_t cm, tri, off, pad; };

// convexTransform.transformInv(heightfieldTransform).p (convexHeightfield.cu:208) =
// PxTransform::transformInv(PxTransform) (PxTransform.h:162-169) LITERALLY:
// qinv = q0.getConjugate(); qinv.rotate(p1 - p0).  P9.94d.
static V3 toCvxPRef(V4 q0, V3 p0, V3 p1)
{
    const V3 d = sub(p1, p0);
    return gP94dToCvxPRotInv ? quatRotateInv(q0, d) : quatRotate(quatConj(q0), d);
}

// buildTriangleInformations (convexHeightfield.cu:120-157) over one triangle: the
// three vertices in convex space and the normal, as the stage stores them.
struct HfRef { V3 e0, e1, e2, en; uint32_t adj[3]; bool wrongHanded; };
static HfRef refHfTri(const Hf& hf, uint32_t tri, V3 hfScale, V4 q0, V3 p0, V4 q1, V3 p1)
{
    const V4 toCvxQ = quatMul(quatConj(q0), q1);
    const V3 toCvxP = toCvxPRef(q0, p0, p1);
    HfRef r;
    V3 w0, w1, w2;
    getTriangle(hf, tri, hfScale, w0, w1, w2, r.adj, r.wrongHanded);
    // The normal is the DENORMALIZED cross, rotated, THEN normalized (:136-137);
    // `rotate` carries no translation, the vertices get the full transform.
    r.en = normalized(quatRotate(toCvxQ, cross3(sub(w1, w0), sub(w2, w0))));
    r.e0 = add(quatRotate(toCvxQ, w0), toCvxP);
    r.e1 = add(quatRotate(toCvxQ, w1), toCvxP);
    r.e2 = add(quatRotate(toCvxQ, w2), toCvxP);
    return r;
}

// The twelve stored float words an HfRef predicts, and where they sit in the
// 128-byte output record (in floats).
static const uint32_t kRefWordAt[12] = { 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26 };
static void refWords(const HfRef& r, uint32_t w[12])
{
    const float f[12] = { r.e0.x, r.e0.y, r.e0.z, r.e1.x, r.e1.y, r.e1.z,
                          r.e2.x, r.e2.y, r.e2.z, r.en.x, r.en.y, r.en.z };
    std::memcpy(w, f, sizeof(f));
}

// Uploads one fixture, runs cvxHfSetupLaunch over `pairs` and returns the
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
        Kernel k("cvxHfSetupBda.spv", "cvxHfSetupLaunch", {}, sizeof(SetupPush));
        k.dispatch(n, {}, push, 1u, 1u);
        dev->flush();
    }

    std::vector<float> got = downloadBuffer<float>(dev, bOut, outInit.size());
    dev->destroyBuffer(bShapes); dev->destroyBuffer(bXf); dev->destroyBuffer(bCd);
    dev->destroyBuffer(bCm); dev->destroyBuffer(bPairBlob); dev->destroyBuffer(bPairs);
    dev->destroyBuffer(bOut);
    return got;
}

PXG_TEST(cvxHfSetupPrologue)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- two heightfields, different shapes so a mis-read of rows/columns
    //      cannot accidentally agree between them --------------------------
    Hf hfA, hfB;

    // 5 rows x 4 columns.  Heights mix positive, NEGATIVE (sign extension) and
    // zero; the tessellation flag is set on some cells and not others so both
    // diagonals are exercised.
    hfA.rows = 5; hfA.cols = 4;
    {
        const int16_t hts[20] = { 10, -300, 0, 45,
                                  -7, 120, 33, -1,
                                  5, 5, -32768, 900,
                                  60, -12, 7, 0,
                                  -100, 8, 21, 3 };
        for (uint32_t i = 0; i < 20; ++i)
        {
            const bool tess = (i % 3u) == 1u;
            hfA.samples.push_back(sampleWord(hts[i], uint8_t(1 + (i % 5)), tess,
                                             uint8_t(2 + (i % 7))));
        }
    }
    buildHf(hfA);

    // 4 rows x 5 columns -- rows and columns TRANSPOSED relative to hfA, so a port
    // that reads them the wrong way round produces different vertex indices here.
    hfB.rows = 4; hfB.cols = 5;
    {
        const int16_t hts[20] = { 2, 4, 6, 8, 10,
                                  -5, -10, 15, 0, 3,
                                  77, -77, 1, -1, 40,
                                  9, 18, -250, 6, 12 };
        for (uint32_t i = 0; i < 20; ++i)
        {
            const bool tess = (i % 2u) == 0u;
            hfB.samples.push_back(sampleWord(hts[i], uint8_t(3 + (i % 4)), tess,
                                             uint8_t(5 + (i % 3))));
        }
    }
    buildHf(hfB);

    // A stand-in hull blob; the setup stage only forwards its address.
    std::vector<uint32_t> hullBlob(64, 0x1234u);

    PxgBufferHandle bHfA = createBufferFrom(dev, hfA.blob);
    PxgBufferHandle bHfB = createBufferFrom(dev, hfB.blob);
    PxgBufferHandle bHull = createBufferFrom(dev, hullBlob);
    const uint64_t addrHfA = dev->getBufferDeviceAddress(bHfA);
    const uint64_t addrHfB = dev->getBufferDeviceAddress(bHfB);
    const uint64_t addrHull = dev->getBufferDeviceAddress(bHull);

    // ---- shapes ------------------------------------------------------------
    std::vector<GpuShape> shapes(7);
    auto setShape = [](GpuShape& s, V3 sc, V4 rot, uint64_t ptr, uint32_t type) {
        s.sx = sc.x; s.sy = sc.y; s.sz = sc.z;
        s.rx = rot.x; s.ry = rot.y; s.rz = rot.z; s.rw = rot.w;
        s.materialIndex = 0;
        s.hullLo = uint32_t(ptr & 0xffffffffu);
        s.hullHi = uint32_t(ptr >> 32);
        s.type = type;
        s.particleOrSoftbodyId = 0;
    };
    // 0: convex, non-identity scale ROTATION.  With the identity, vertex2Shape --
    //    which the hull half of the pipeline applies later -- degenerates to a
    //    componentwise multiply and both rotations inside it become no-ops.
    {
        const float a = 0.9f;
        setShape(shapes[0], { 1.3f, 0.8f, 1.1f },
                 { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, addrHull, GEOM_CONVEXMESH);
    }
    // 1: heightfield A.  THREE DIFFERENT SCALE COMPONENTS, so a hf2shapep that
    //    multiplies the wrong component into the wrong axis is visible; and a
    //    NON-IDENTITY SCALE ROTATION which the heightfield path must IGNORE (the
    //    .cu passes only scale.scale.x/y/z into getTriangle, never the rotation).
    {
        const float a = -0.6f;
        setShape(shapes[1], { 2.0f, 0.05f, 1.5f },
                 { 0.0f, std::sin(a * 0.5f), 0.0f, std::cos(a * 0.5f) }, addrHfA, GEOM_HEIGHTFIELD);
    }
    // 2: heightfield B with a NEGATIVE COLUMN SCALE -> wrongHanded, so both
    //    permutations fire.  Upstream's PxHeightFieldGeometry::isValid rejects
    //    this, which is why the port's comment calls the branch unreachable today;
    //    unreachable is not the same as absent, and this is where it is checked.
    setShape(shapes[2], { 1.25f, 0.5f, -0.8f }, { 0, 0, 0, 1 }, addrHfB, GEOM_HEIGHTFIELD);
    // 3: a second convex with a different scale rotation.
    {
        const float a = 0.4f;
        setShape(shapes[3], { 0.9f, 1.4f, 0.6f },
                 { 0.0f, 0.0f, std::sin(a * 0.5f), std::cos(a * 0.5f) }, addrHull, GEOM_CONVEXMESH);
    }
    // 4: heightfield B, positive scales.
    setShape(shapes[4], { 0.75f, 0.25f, 1.1f }, { 0, 0, 0, 1 }, addrHfB, GEOM_HEIGHTFIELD);
    // 6: TWO NEGATIVE SCALE COMPONENTS -- row AND column.  The handedness is an
    //    XOR, so two mirrors compose back to a right-handed frame and
    //    `wrongHanded` is FALSE here.  An OR agrees with the xor on every other
    //    input, which is why the first P9.51b sweep of this module reported
    //    `G1-handedness-is-an-or` as a SURVIVOR: no case carried both signs.  The
    //    trimesh fixture had the same gap and closed it the same way
    //    (testCvxTriSetup's `sawNegativePairNoFlip`), which is the reason to state
    //    the class rather than patch the instance -- a two-sign case belongs in
    //    every fixture over a sign-parity decision.
    setShape(shapes[6], { -1.4f, 0.4f, -0.9f }, { 0, 0, 0, 1 }, addrHfB, GEOM_HEIGHTFIELD);
    // 5: THE SYNTHETIC SWAP PROBE -- typed eTRIANGLEMESH, blob a valid heightfield.
    //    See the header: this is the only input that can tell CT_NO_SWAP apart from
    //    the trimesh sibling's CT_GEOM_TRIANGLEMESH.
    setShape(shapes[5], { 1.6f, 0.3f, 0.9f }, { 0, 0, 0, 1 }, addrHfA, GEOM_TRIANGLEMESH);

    // ---- transforms (PxsCachedTransform: quat @0, pos @16, 32 B) -----------
    std::vector<float> xf(6 * 8, 0.0f);
    auto setXf = [&](uint32_t i, V4 q, V3 p) {
        float* r = xf.data() + i * 8;
        r[0] = q.x; r[1] = q.y; r[2] = q.z; r[3] = q.w;
        r[4] = p.x; r[5] = p.y; r[6] = p.z;
    };
    // THE CONVEX BODY MUST ROTATE: the composition is conj(q0)*q1 with
    // conj(q0).rotate(p1 - p0), and with q0 the identity the conjugate is a no-op and
    // the translation needs no rotating -- three separate mistakes become invisible.
    { const float a = 0.55f;
      setXf(0, { 0.0f, 0.0f, std::sin(a * 0.5f), std::cos(a * 0.5f) }, { 0.5f, -0.25f, 1.0f }); }
    { const float a = 0.7f;
      setXf(1, { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, { -1.0f, 2.0f, 0.5f }); }
    { const float a = -1.1f;
      setXf(2, { 0.0f, 0.0f, std::sin(a * 0.5f), std::cos(a * 0.5f) }, { 3.0f, 0.0f, -1.5f }); }
    { const float a = -0.8f;
      setXf(3, { std::sin(a * 0.5f), 0.0f, 0.0f, std::cos(a * 0.5f) }, { 1.5f, -2.0f, 0.75f }); }
    { const float a = 1.3f;
      setXf(4, { 0.0f, std::sin(a * 0.5f), 0.0f, std::cos(a * 0.5f) }, { -0.5f, 1.25f, 2.0f }); }
    setXf(5, { 0, 0, 0, 1 }, { 2.25f, 0.5f, -1.0f });

    std::vector<float> cdist = { 0.02f, 0.031f, 0.047f, 0.013f, 0.005f, 0.061f };

    // ---- contact managers --------------------------------------------------
    // THE TRANSFORM REFS MUST NOT EQUAL THE SHAPE REFS.  While they match, the
    // contact distance -- indexed by the TRANSFORM refs -- reads the same slots
    // either way, so indexing it by the shape refs is invisible.
    std::vector<CmInput> cms = {
        { 0, 1, 4, 5 },   // convex vs heightfield A
        { 3, 2, 3, 1 },   // the rotated convex vs the NEGATIVE column scale
        { 0, 4, 0, 2 },   // convex vs heightfield B
        { 5, 4, 5, 0 },   // the swap probe: slot 0 is typed eTRIANGLEMESH
        { 0, 6, 4, 3 },   // TWO negative scales -> the xor says NOT wrongHanded
    };

    // ---- ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40 -------
    // The decoy values at @32 and @36 are what a port reading the declaration's
    // trailing RUNNING-END offsets as field offsets would pick up.
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
    setPair(4, 1300, 4, 99, 1400);

    // ---- the pairs to run --------------------------------------------------
    // hfA is 5x4, so its cells are the vertices with row < 4 and column < 3.
    //   cell 0  (r0,c0)  corner: both `cell % cols != 0` and `cell >= cols-1` fail
    //   cell 5  (r1,c1)  interior: every adjacency is a real triangle
    //   cell 10 (r2,c2)  c == cols-2, so `cell % cols < cols-2` fails
    //   cell 12 (r3,c0)  r == rows-2, so `cell / cols != rows-2` fails
    // and both triangles of a cell are driven, because the two arms of each
    // adjacency branch differ in WHICH guards they apply.
    std::vector<Pair> pairs = {
        { 0, 0, 0, 0 },     // cell 0, first triangle
        { 0, 1, 3, 0 },     // cell 0, second triangle
        { 0, 10, 1, 0 },    // cell 5, interior, first
        { 0, 11, 2, 0 },    // cell 5, interior, second
        { 0, 21, 4, 0 },    // cell 10, second
        { 0, 24, 5, 0 },    // cell 12, first
        { 0, 25, 6, 0 },    // cell 12, second
        // hfA's tessellation flag is set when the sample index % 3 == 1, so cells
        // 4 and 13 take the SHARED arm -- and they are the two positions where the
        // shared arm's own guards sit: cell 4 is in COLUMN 0 (`cell % columns != 0`
        // fails) and cell 13 is in ROW rows-2 (`cell / columns != rows - 2` fails).
        // Without them the first P9.51b sweep of this module reported
        // `J3-shared-first-column-guard-dropped` and
        // `J4-shared-first-row-guard-off-by-one` as survivors: the fixture had
        // shared cells, but none standing on a guard.
        { 0, 8, 8, 0 },     // cell 4, SHARED, first -- column 0
        { 0, 26, 9, 0 },    // cell 13, SHARED, first -- row rows-2
        { 1, 6, 2, 0 },     // hfB, NEGATIVE column scale -> wrongHanded
        { 1, 13, 1, 0 },    // hfB, wrongHanded, second triangle
        { 2, 8, 7, 0 },     // hfB, positive scales -- the same blob, not flipped
        { 3, 2, 4, 0 },     // THE SWAP PROBE
        { 4, 6, 3, 0 },     // two negative scales: no flip
        { 4, 15, 5, 0 },    // and its second triangle
    };

    const uint32_t n = uint32_t(pairs.size());
    const auto got = runSetupKernel(dev, shapes, xf, cdist, cms, pairBlob, pairs);

    size_t bad = 0;
    size_t sawWrongHanded = 0, sawSharedDiagonal = 0, sawUnsharedDiagonal = 0;
    size_t sawBoundaryAdj = 0, sawAllRealAdj = 0, sawSecondTri = 0;
    size_t sawNegativeHeight = 0, sawTrimeshTypedSlot0 = 0, sawIgnoredHfScaleRot = 0;
    size_t sawTwoNegativeNoFlip = 0;
    size_t sawSharedOnColumnGuard = 0, sawSharedOnRowGuard = 0, sawRodriguesDiffers = 0;
    auto sameF = [](float a, float b)
    { uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y; };

    for (uint32_t i = 0; i < n; ++i)
    {
        const Pair& pr = pairs[i];
        const CmInput& cm = cms[pr.cm];

        // NO SWAP.  Slot 0 is the convex and slot 1 the mesh, whatever their types
        // (convexHeightfield.cu:211, :214).
        const uint32_t sr0 = cm.shapeRef0, sr1 = cm.shapeRef1;
        const uint32_t tc0 = cm.tcRef0, tc1 = cm.tcRef1;
        if (shapes[sr0].type == GEOM_TRIANGLEMESH) ++sawTrimeshTypedSlot0;

        const GpuShape& cvx = shapes[sr0];
        const GpuShape& hfs = shapes[sr1];
        const Hf& hf = (hfs.hullLo == uint32_t(addrHfA & 0xffffffffu) &&
                        hfs.hullHi == uint32_t(addrHfA >> 32)) ? hfA : hfB;
        if ((hfs.rx != 0.0f) || (hfs.ry != 0.0f) || (hfs.rz != 0.0f)) ++sawIgnoredHfScaleRot;

        const V4 q0 = { xf[tc0 * 8 + 0], xf[tc0 * 8 + 1], xf[tc0 * 8 + 2], xf[tc0 * 8 + 3] };
        const V3 p0 = { xf[tc0 * 8 + 4], xf[tc0 * 8 + 5], xf[tc0 * 8 + 6] };
        const V4 q1 = { xf[tc1 * 8 + 0], xf[tc1 * 8 + 1], xf[tc1 * 8 + 2], xf[tc1 * 8 + 3] };
        const V3 p1 = { xf[tc1 * 8 + 4], xf[tc1 * 8 + 5], xf[tc1 * 8 + 6] };
        const HfRef ref = refHfTri(hf, pr.tri, V3{ hfs.sx, hfs.sy, hfs.sz }, q0, p0, q1, p1);
        const uint32_t* adj = ref.adj;
        const bool wrongHanded = ref.wrongHanded;
        // P9.94d vacuity: the same pair under the OLD Rodrigues rotations.  At least
        // one pair must store a different word, or this fixture never saw the
        // shader's switch to the literal PxQuat formula.
        {
            gP94dRodrigues = true;
            const HfRef twin = refHfTri(hf, pr.tri, V3{ hfs.sx, hfs.sy, hfs.sz }, q0, p0, q1, p1);
            gP94dRodrigues = false;
            uint32_t a[12], b[12];
            refWords(ref, a); refWords(twin, b);
            for (uint32_t k = 0; k < 12; ++k) if (a[k] != b[k]) { ++sawRodriguesDiffers; break; }
        }

        if (wrongHanded) ++sawWrongHanded;
        if (!wrongHanded && (hfs.sx < 0.0f || hfs.sz < 0.0f)) ++sawTwoNegativeNoFlip;
        if (zerothVertexShared(hf, pr.tri >> 1))
        {
            ++sawSharedDiagonal;
            const uint32_t cell = pr.tri >> 1;
            if ((pr.tri & 1u) == 0u && cell % hf.cols == 0u) ++sawSharedOnColumnGuard;
            if ((pr.tri & 1u) == 0u && cell / hf.cols == hf.rows - 2u) ++sawSharedOnRowGuard;
        }
        else ++sawUnsharedDiagonal;
        if ((pr.tri & 1u) != 0u) ++sawSecondTri;
        if (adj[0] == HF_BOUNDARY || adj[2] == HF_BOUNDARY) ++sawBoundaryAdj;
        else ++sawAllRealAdj;
        {
            uint32_t i0, i1, i2;
            triVertexIndices(hf, pr.tri, i0, i1, i2);
            if (hfHeight(hf, i0) < 0.0f || hfHeight(hf, i1) < 0.0f || hfHeight(hf, i2) < 0.0f)
                ++sawNegativeHeight;
        }

        const V3 en = ref.en, e0 = ref.e0, e1 = ref.e1, e2 = ref.e2;

        const float* r = got.data() + size_t(i) * (128 / 4);
        uint32_t u[4];
        std::memcpy(u, r, 16);
        const uint64_t gotHull = uint64_t(u[0]) | (uint64_t(u[1]) << 32);
        const uint64_t expHull = uint64_t(cvx.hullLo) | (uint64_t(cvx.hullHi) << 32);

        auto chkU = [&](const char* w, uint32_t g, uint32_t e)
        { if (g != e) { ++bad; if (bad <= 16) std::printf("  [pair %u] %s: got=%u exp=%u\n", i, w, g, e); } };
        auto chkF = [&](const char* w, float g, float e)
        { if (!sameF(g, e)) { ++bad; if (bad <= 16) std::printf("  [pair %u] %s: got=%.9g exp=%.9g\n", i, w, g, e); } };

        if (gotHull != expHull)
        { ++bad; std::printf("  [pair %u] hull address mismatch\n", i); }
        // A heightfield has no faceRemap: the triangle index is forwarded as-is.
        chkU("triangleIdx", u[2], pr.tri);
        {
            uint32_t startIndex, rounded;
            std::memcpy(&startIndex, pairBlob.data() + size_t(pr.cm) * 56 + 28, 4);
            std::memcpy(&rounded, pairBlob.data() + size_t(pr.cm) * 56 + 40, 4);
            chkU("pairOffsetPadded", u[3], rounded + pr.off);
            uint32_t adjs[4];
            std::memcpy(adjs, r + 28, 16);
            chkU("adj.x", adjs[0], adj[0]);
            chkU("adj.y", adjs[1], adj[1]);
            chkU("adj.z", adjs[2], adj[2]);
            chkU("pairOffset", adjs[3], startIndex + pr.off);
        }
        chkF("scale.x", r[4], cvx.sx);
        chkF("scale.y", r[5], cvx.sy);
        chkF("scale.z", r[6], cvx.sz);
        chkF("contactDist", r[7], cdist[tc0] + cdist[tc1]);
        chkF("scaleRot.x", r[8], cvx.rx);
        chkF("scaleRot.y", r[9], cvx.ry);
        chkF("scaleRot.z", r[10], cvx.rz);
        chkF("scaleRot.w", r[11], cvx.rw);
        chkF("v0.x", r[12], e0.x); chkF("v0.y", r[13], e0.y); chkF("v0.z", r[14], e0.z);
        chkF("v1.x", r[16], e1.x); chkF("v1.y", r[17], e1.y); chkF("v1.z", r[18], e1.z);
        chkF("v2.x", r[20], e2.x); chkF("v2.y", r[21], e2.y); chkF("v2.z", r[22], e2.z);
        chkF("n.x", r[24], en.x); chkF("n.y", r[25], en.y); chkF("n.z", r[26], en.z);
    }

    dev->destroyBuffer(bHfA); dev->destroyBuffer(bHfB); dev->destroyBuffer(bHull);

    PXG_CHECK_MSG(bad == 0, "cvxHfSetup: %zu mismatches", bad);
    // Every one of these guards a branch the mirror above distinguishes.  Without
    // them a later edit to the pair list could quietly stop reaching a branch and
    // the fixture would still pass, which is the failure mode that made the P9.51a
    // sweep report 18/30 with three of the survivors caused by the fixture rather
    // than by the shader.
    PXG_CHECK_MSG(sawTrimeshTypedSlot0 > 0,
                  "no pair has an eTRIANGLEMESH-typed shape in slot 0 (CT_NO_SWAP is untested)");
    PXG_CHECK_MSG(sawWrongHanded > 0, "no pair has a negative scale (wrongHanded is untested)");
    PXG_CHECK_MSG(sawTwoNegativeNoFlip > 0,
                  "no pair has BOTH the row and the column scale negative -- the only input on "
                  "which the handedness XOR differs from an OR");
    PXG_CHECK_MSG(sawSharedDiagonal > 0, "no pair sits on a tessellation-flagged cell");
    PXG_CHECK_MSG(sawUnsharedDiagonal > 0, "no pair sits on an unflagged cell");
    PXG_CHECK_MSG(sawSharedOnColumnGuard > 0,
                  "no pair is a SHARED cell's first triangle in COLUMN 0 -- the shared arm's "
                  "`cell %% columns != 0` guard is then untested");
    PXG_CHECK_MSG(sawSharedOnRowGuard > 0,
                  "no pair is a SHARED cell's first triangle in ROW rows-2 -- the shared arm's "
                  "`cell / columns != rows - 2` guard is then untested");
    PXG_CHECK_MSG(sawSecondTri > 0, "no pair is a cell's SECOND triangle");
    PXG_CHECK_MSG(sawBoundaryAdj > 0, "no pair has a BOUNDARY adjacency");
    PXG_CHECK_MSG(sawAllRealAdj > 0, "no pair has three real adjacencies (an interior cell)");
    PXG_CHECK_MSG(sawNegativeHeight > 0,
                  "no pair reads a NEGATIVE height (the 16-bit sign extension is untested)");
    PXG_CHECK_MSG(sawIgnoredHfScaleRot > 0,
                  "no heightfield carries a scale rotation the path must ignore");
    PXG_CHECK_MSG(sawRodriguesDiffers > 0,
                  "no pair stores a different word under the old Rodrigues rotation "
                  "(the P9.94d literal PxQuat formula is unobserved)");
}

// P9.94d: toCvxP = convexTransform.transformInv(heightfieldTransform).p
// (convexHeightfield.cu:208) is q0.getConjugate().rotate(p1 - p0) (PxTransform.h:162-169),
// NOT q0.rotateInv(p1 - p0).  The two agree on every nonzero component; this case makes
// them disagree on a zero.
//
// q0 = identity, p0 = +0, p1 = (-0,-0,-0): toCvxP is (+0,+0,+0) under the CUDA and
// (-0,-0,-0) under the old spelling (the arithmetic is spelled out at
// cvxTriSetup_p94dToCvxPZeroSign).  q1 = (0,1,0,0), 180 deg about y; toCvxQ =
// conj(q0)*q1 = (+0,1,+0,+0).  A heightfield vertex in COLUMN 0 has z = 0*colScale = +0,
// and with row > 0 (vx > 0) and a NEGATIVE height (dot2 = 2*h*heightScale < 0) its
// rotated z is vz*w2 + (x*vy - y*vx)*w + z*dot2 = (-0) + (-0) + (+0 * neg = -0) = -0, so
// the stored z is -0 + toCvxP.z: +0 (CUDA) against -0 (old).  2x2 heightfield, every
// height -5, no tessellation flag: triangle 0 is (r0,c0), (r0,c1), (r1,c0), and its third
// vertex is the (row 1, column 0) one.  No RNG.
PXG_TEST(cvxHfSetup_p94dToCvxPZeroSign)
{
    PxgSlangComputeDevice* dev = testDevice();
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    const float nz = -0.0f;

    Hf hf;
    hf.rows = 2; hf.cols = 2;
    for (uint32_t i = 0; i < 4; ++i) hf.samples.push_back(sampleWord(-5, 1, false, 2));
    buildHf(hf);
    std::vector<uint32_t> hullBlob(64, 0x1234u);
    PxgBufferHandle bHf = createBufferFrom(dev, hf.blob);
    PxgBufferHandle bHull = createBufferFrom(dev, hullBlob);
    const uint64_t addrHf = dev->getBufferDeviceAddress(bHf);
    const uint64_t addrHull = dev->getBufferDeviceAddress(bHull);

    std::vector<GpuShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    shapes[0].sx = 1.0f; shapes[0].sy = 1.0f; shapes[0].sz = 1.0f; shapes[0].rw = 1.0f;
    shapes[0].hullLo = uint32_t(addrHull & 0xffffffffu); shapes[0].hullHi = uint32_t(addrHull >> 32);
    shapes[0].type = GEOM_CONVEXMESH;
    shapes[1].sx = 1.0f; shapes[1].sy = 0.5f; shapes[1].sz = 1.0f; shapes[1].rw = 1.0f;
    shapes[1].hullLo = uint32_t(addrHf & 0xffffffffu); shapes[1].hullHi = uint32_t(addrHf >> 32);
    shapes[1].type = GEOM_HEIGHTFIELD;

    const V4 q0 = { 0, 0, 0, 1 }, q1 = { 0, 1, 0, 0 };
    const V3 p0 = { 0, 0, 0 }, p1 = { nz, nz, nz };
    std::vector<float> xf(2 * 8, 0.0f);
    xf[0] = q0.x; xf[1] = q0.y; xf[2] = q0.z; xf[3] = q0.w; xf[4] = p0.x; xf[5] = p0.y; xf[6] = p0.z;
    xf[8] = q1.x; xf[9] = q1.y; xf[10] = q1.z; xf[11] = q1.w; xf[12] = p1.x; xf[13] = p1.y; xf[14] = p1.z;
    std::vector<float> cdist = { 0.01f, 0.02f };
    std::vector<CmInput> cms = { { 0, 1, 0, 1 } };
    std::vector<uint8_t> pairBlob(56, 0);
    std::vector<Pair> pairs = { { 0, 0, 0, 0 } };

    const V3 hfScale = { shapes[1].sx, shapes[1].sy, shapes[1].sz };
    const HfRef lit = refHfTri(hf, 0, hfScale, q0, p0, q1, p1);
    gP94dToCvxPRotInv = true;
    const HfRef twin = refHfTri(hf, 0, hfScale, q0, p0, q1, p1);
    gP94dToCvxPRotInv = false;
    uint32_t a[12], b[12];
    refWords(lit, a); refWords(twin, b);
    PXG_CHECK_MSG(u32(lit.e2.z) == 0x00000000u && u32(twin.e2.z) == 0x80000000u,
                  "p94d: literal v2.z 0x%08x (want +0), twin 0x%08x (want -0)",
                  u32(lit.e2.z), u32(twin.e2.z));
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
    dev->destroyBuffer(bHf); dev->destroyBuffer(bHull);
}

// P9.94d: ctQuatRotate is PxQuat::rotate LITERALLY.  Random poses and heightfield
// scales over every triangle of a 4x4 heightfield: every stored vertex and normal
// word must equal the literal oracle BITWISE, and the old Rodrigues spelling must
// store a different word on some of them -- otherwise this test cannot tell the two
// formulas apart.
PXG_TEST(cvxHfSetup_p94dLiteralRotate)
{
    PxgSlangComputeDevice* dev = testDevice();
    auto u32 = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };

    uint32_t seed = 0x94d0002u;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float(seed >> 8) * (1.0f / 16777216.0f); };
    auto sym = [&](float m) { return (rnd() * 2.0f - 1.0f) * m; };
    auto unitQ = [&]() {
        V4 q = { sym(1), sym(1), sym(1), sym(1) };
        const float r = 1.0f / std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        return V4{ q.x * r, q.y * r, q.z * r, q.w * r };
    };

    Hf hf;
    hf.rows = 4; hf.cols = 4;
    for (uint32_t i = 0; i < 16; ++i)
        hf.samples.push_back(sampleWord(int16_t(int32_t(rnd() * 2000.0f) - 1000), 1, (i % 3u) == 0u, 2));
    buildHf(hf);
    std::vector<uint32_t> hullBlob(64, 0x1234u);
    PxgBufferHandle bHf = createBufferFrom(dev, hf.blob);
    PxgBufferHandle bHull = createBufferFrom(dev, hullBlob);
    const uint64_t addrHf = dev->getBufferDeviceAddress(bHf);
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
    std::vector<uint32_t> tris(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        GpuShape& h = shapes[1 + i];
        h.sx = 0.25f + rnd() * 2.0f; h.sy = 0.001f + rnd() * 0.01f; h.sz = 0.25f + rnd() * 2.0f;
        h.rw = 1.0f;
        h.hullLo = uint32_t(addrHf & 0xffffffffu); h.hullHi = uint32_t(addrHf >> 32);
        h.type = GEOM_HEIGHTFIELD;
        for (uint32_t t = 0; t < 2; ++t)
        {
            const V4 q = unitQ();
            float* x = xf.data() + size_t(2 * i + t) * 8;
            x[0] = q.x; x[1] = q.y; x[2] = q.z; x[3] = q.w;
            x[4] = sym(4); x[5] = sym(4); x[6] = sym(4);
        }
        cms[i] = { 0, 1 + i, 2 * i, 2 * i + 1 };
        // a cell in the first rows-1 rows and first cols-1 columns
        const uint32_t cell = (i % (hf.rows - 1)) * hf.cols + ((i / 3u) % (hf.cols - 1));
        tris[i] = cell * 2u + ((i / 9u) & 1u);
        pairs[i] = { i, tris[i], 0, 0 };
    }

    const auto got = runSetupKernel(dev, shapes, xf, cdist, cms, pairBlob, pairs);

    size_t bad = 0, rodriguesDiffers = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const GpuShape& h = shapes[1 + i];
        const float* x0 = xf.data() + size_t(2 * i) * 8;
        const float* x1 = x0 + 8;
        const V4 q0 = { x0[0], x0[1], x0[2], x0[3] }, q1 = { x1[0], x1[1], x1[2], x1[3] };
        const V3 p0 = { x0[4], x0[5], x0[6] }, p1 = { x1[4], x1[5], x1[6] };
        const V3 sc = { h.sx, h.sy, h.sz };
        const HfRef lit = refHfTri(hf, tris[i], sc, q0, p0, q1, p1);
        gP94dRodrigues = true;
        const HfRef twin = refHfTri(hf, tris[i], sc, q0, p0, q1, p1);
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
    std::printf("  cvxHfSetup_p94dLiteralRotate: %zu/%u pairs store a different word under the "
                "Rodrigues twin\n", rodriguesDiffers, N);
    PXG_CHECK_MSG(bad == 0, "p94d: %zu stored words differ from the literal PxQuat oracle", bad);
    PXG_CHECK_MSG(rodriguesDiffers > 0,
                  "p94d: the Rodrigues twin stores the same words on every pair -- the literal "
                  "formula is unobserved");
    dev->destroyBuffer(bHf); dev->destroyBuffer(bHull);
}

} // namespace
