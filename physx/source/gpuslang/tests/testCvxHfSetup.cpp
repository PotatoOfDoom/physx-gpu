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
// PxVec3::getNormalized (PxVec3.h:286) -- one reciprocal, three multiplies.
static V3 normalized(V3 v)
{
    const float m = std::sqrt(dot3(v, v));
    if (!(m > 0.0f)) return V3{ 0, 0, 0 };
    const float r = 1.0f / m;
    return V3{ v.x * r, v.y * r, v.z * r };
}

static const uint32_t GEOM_TRIANGLEMESH = 4;   // PxGeometryType::eTRIANGLEMESH
static const uint32_t GEOM_CONVEXMESH = 5;
static const uint32_t GEOM_HEIGHTFIELD = 6;
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
    // rotInv(q0, p1 - p0), and with q0 the identity the conjugate is a no-op and
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
    struct Pair { uint32_t cm, tri, off, pad; };
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

    const auto got = downloadBuffer<float>(dev, bOut, outInit.size());

    size_t bad = 0;
    size_t sawWrongHanded = 0, sawSharedDiagonal = 0, sawUnsharedDiagonal = 0;
    size_t sawBoundaryAdj = 0, sawAllRealAdj = 0, sawSecondTri = 0;
    size_t sawNegativeHeight = 0, sawTrimeshTypedSlot0 = 0, sawIgnoredHfScaleRot = 0;
    size_t sawTwoNegativeNoFlip = 0;
    size_t sawSharedOnColumnGuard = 0, sawSharedOnRowGuard = 0;
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
        const V4 toCvxQ = quatMul(quatConj(q0), q1);
        const V3 toCvxP = quatRotateInv(q0, sub(p1, p0));

        V3 w0, w1, w2;
        uint32_t adj[3] = { 0, 0, 0 };
        bool wrongHanded = false;
        getTriangle(hf, pr.tri, V3{ hfs.sx, hfs.sy, hfs.sz }, w0, w1, w2, adj, wrongHanded);

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

        // The normal is the DENORMALIZED cross, rotated, THEN normalized (:136-137);
        // `rotate` carries no translation, the vertices get the full transform.
        const V3 en = normalized(quatRotate(toCvxQ, cross3(sub(w1, w0), sub(w2, w0))));
        const V3 e0 = add(quatRotate(toCvxQ, w0), toCvxP);
        const V3 e1 = add(quatRotate(toCvxQ, w1), toCvxP);
        const V3 e2 = add(quatRotate(toCvxQ, w2), toCvxP);

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
    dev->destroyBuffer(bShapes); dev->destroyBuffer(bXf); dev->destroyBuffer(bCd);
    dev->destroyBuffer(bCm); dev->destroyBuffer(bPairBlob); dev->destroyBuffer(bPairs);
    dev->destroyBuffer(bOut);

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
                  "`cell % columns != 0` guard is then untested");
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
}

} // namespace
