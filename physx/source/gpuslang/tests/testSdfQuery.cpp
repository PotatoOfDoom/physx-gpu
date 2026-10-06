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

// testSdfQuery.cpp -- P9.95, the fixture for sdfQueryBda.slang
// (evaluatePointDistancesSDFBatch, convexMeshMidphase.cu:1233-1287).
//
// The kernel is a composition of two swept layers -- the mesh read and texture set-up
// (pxgSlangSdfTexture, P9.12c) and PxVolumeGrad / PxSdfDistance (pxgSlangSdfCollision,
// P9.12d) -- plus three things of its own: the batch/slice/guard plumbing, the
// normalizeSafe between them, and the store.  So it is held three ways, none with a
// tolerance:
//
//  (1) A CROSS-CHECK against npSdfCollision.spv's sdfCollDistance / sdfCollGrad over the
//      SAME blobs, textures and points: out.w must be that distance and out.xyz the
//      ENGINE's PxVec3::normalizeSafe of that gradient, bit for bit.  npSdfCollision is a
//      different .spv, and a mutation sweep with MUTATION_SWEEP_SPV_ONLY=1 rebuilds only
//      sdfQueryBda.spv, so under a module mutant this cross-check stays the clean
//      reference -- that is what lets the module sweeps below bite.
//  (2) An INDEPENDENT ANCHOR on two grid-aligned dense meshes (power-of-two spacing,
//      dyadic meshLower; power-of-two dimensions until P9.96 -- see buildAligned), where a point on
//      a texel centre has an exact integer texel coordinate: the distance must BE the stored texel, the gradient the per-axis
//      difference of the stored neighbours, normalised by the engine's PxVec3.  On the
//      sparse mesh the distance is bitwise against the host's PxSdfDistance over
//      sdfSampleRef (P9.96; until then it was held to sparseOracle inside P9.12a's
//      filter bound, which stays as a secondary bound at the nodes).  The node slots'
//      output bits were recorded before P9.96 replaced the sampler and are asserted
//      unchanged, and sdfQueryNonPow2 checks the same anchors on a 10 x 7 x 11 grid,
//      plus every slot's distance bitwise against the host's PxSdfDistance.
//  (3) PLUMBING facts: every slot past a shape's count keeps its poison, a trap slot past
//      the whole output keeps its poison, the inputs are bit-unchanged, a mesh without an
//      SDF writes nothing (D1), a scaled + rotated shape over the same mesh answers
//      exactly like the unscaled one (Q8), and a shape table with no fixed point makes
//      every index mixup visible.
//
// The launch goes through PxgCudaShim with the push size taken from the bridge's own
// launch-ABI row, and the blob is built from the six host-typed arguments with the
// bridge's natural-alignment rule, so a row of the wrong size refuses the launch here.
// Only PXG_SCENE=sdfquery drives the real engine path (PxDirectGPUAPI::evaluateSDFDistances).

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "PxgVulkanKernelAbi.h"
#include "PxgConvexConvexShape.h"     // PxgShape
#include "geometry/PxGeometry.h"
#include "foundation/PxVec4.h"
#include "pxgSdfSceneRef.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace physx { namespace gpuslang {
void pxgSetVulkanDriverShim(PxgCudaShim* shim);
} }

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgResult;
using namespace physx;
using namespace pxgsdfref;

// ---------------------------------------------------------------------------
// Layout pins.
// ---------------------------------------------------------------------------
// The push block = the host pack of PxgNarrowphaseCore.cpp:8912-8920.
struct SdfQueryBlob
{
    uint64_t gpuShapes, shapeRefsSDF, localPositions, pointCountPerShape;
    uint32_t maxPointCount, pad;
    uint64_t localGradAndSdf;
};
static_assert(sizeof(SdfQueryBlob) == 48u, "blob = pxgKernelAbiPushSize({8,8,8,8,4,8})");
static_assert(offsetof(SdfQueryBlob, localGradAndSdf) == 40u, "the PxU32 pads the last pointer to 40");
static_assert(sizeof(PxgShape) == 48u, "PxgShape stride (PxgShapeGpu in pxgSlangNpAbi)");
static_assert(offsetof(PxgShape, hullOrMeshPtr) == 32u, "PxgShape::hullOrMeshPtr @32");
static_assert(sizeof(PxShapeGPUIndex) == 4u, "PxShapeGPUIndex is a PxU32");
static_assert(sizeof(PxVec4) == 16u, "PxVec4 stride");

static const char* const SQ_SPV = "sdfQueryBda.spv";
static const char* const SQ_KERNEL = "evaluatePointDistancesSDFBatch";
static const uint32_t SQ_BLOCK = 1024u;   // PxgNarrowphaseCore.cpp:8921
static const uint32_t SQ_POISON = 0xEEEEEEEEu;

// npSdfCollision.slang's push block (testSdfCollision.cpp's P_Coll).
struct P_Coll
{
    uint64_t meshPtr, points, gradIn, dirSeed, outBuf;
    uint32_t numPoints, pad;
    uint64_t triVerts, triOut;
};
static const uint32_t STRIDE_DIST = 12u;
static const uint32_t STRIDE_GRAD = 32u;

// The bridge's blob rule (pxgKernelAbiPushSize): natural alignment by size, rounded to
// the largest alignment seen.
struct EngineBlob
{
    std::vector<uint8_t> bytes;
    size_t maxAlign = 1;
    template <typename T>
    void add(const T& v)
    {
        const size_t a = sizeof(T) >= 8 ? 8 : (sizeof(T) >= 4 ? 4 : (sizeof(T) >= 2 ? 2 : 1));
        if (a > maxAlign) maxAlign = a;
        bytes.resize((bytes.size() + (a - 1)) & ~(a - 1));
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), pb, pb + sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + (maxAlign - 1)) & ~(maxAlign - 1)); }
};

static bool bitEq4(const PxVec4& a, const PxVec4& b) { return std::memcmp(&a, &b, sizeof(PxVec4)) == 0; }
static bool isPoison(const PxVec4& v)
{
    uint32_t w[4];
    std::memcpy(w, &v, 16);
    return w[0] == SQ_POISON && w[1] == SQ_POISON && w[2] == SQ_POISON && w[3] == SQ_POISON;
}
static PxVec4 poison4() { PxVec4 v; const uint32_t w[4] = { SQ_POISON, SQ_POISON, SQ_POISON, SQ_POISON }; std::memcpy(&v, w, 16); return v; }

// ---------------------------------------------------------------------------
// The meshes.
// ---------------------------------------------------------------------------
static size_t dIdx(const SdfScene& s, uint32_t x, uint32_t y, uint32_t z)
{ return (size_t(z) * s.dy + y) * s.dx + x; }

// The patched texels of M_al (see buildAligned).
static const uint32_t FLAT_X = 4u, FLAT_Y = 3u, FLAT_Z = 5u;
static const uint32_t TINY_X = 7u, TINY_Y = 2u, TINY_Z = 3u;
static const float TINY_G = 1e-21f;   // below PX_NORMALIZATION_EPSILON (1e-20f)
static const uint32_t NEAR_X = 10u, NEAR_Y = 4u, NEAR_Z = 12u;
static const float NEAR_G = 5e-20f;   // 5x PX_NORMALIZATION_EPSILON: a loose upper bound only
// The TIGHT bracket of PX_NORMALIZATION_EPSILON (buildAligned (iv)): two ADJACENT floats.
static const uint32_t EPSLO_X = 12u, EPSLO_Y = 2u, EPSLO_Z = 20u;
static const float EPSLO_G = 1.00000078e-20f;   // 0x1e3ce512, |g| = 9.99997303e-21: the largest g left unchanged
static const uint32_t EPSHI_X = 5u, EPSHI_Y = 5u, EPSHI_Z = 25u;
static const float EPSHI_G = 1.00000086e-20f;   // 0x1e3ce513, |g| = 1.00000433e-20: the smallest g normalised

// A grid-aligned DENSE mesh, built like testSdfCollision's buildDenseAligned but with
// its own numbers: power-of-two spacing and dyadic meshLower, so the texel coordinate
// of `lower + k*spacing` is exactly k.  No texel is -0; the only zero texels are the
// four patched +0s below, and none is an anchor centre of the fixed points.
//
// AND POWER-OF-TWO DIMENSIONS, which were necessary until P9.96: pxgTex3D handed a
// Vulkan sampler NORMALISED coordinates, (k + 0.5) / dim (Vulkan allows unnormalised
// coordinates only on 1D/2D views), and the sampler scaled them back by dim.  For a
// dim that is not a power of two that round trip is not exact, the trilinear weights
// came out a few ulps off 0/1, and a "texel-centre" sample mixed in ~1e-7 of the
// neighbour.  MEASURED (P9.95e) with dims 10 x 7 x 11: the flat-gradient point read
// grad.z = -1.07e-06 instead of 0, and 85 of the 1024 points of sdfQueryGrid's first
// case missed the stored texels' gradient by an ulp or more.
// P9.96 removed the round trip: the filter now runs in texel space from texel loads,
// and a texel centre is exact for EVERY extent -- sdfQueryNonPow2 pins that on a
// 10 x 7 x 11 grid (140 of its 1024 slots differed on the old sampler, 0 now).  The
// meshes below keep their power-of-two dims so that their recorded bits stay
// comparable; nothing requires them any more.
static void buildAligned(SdfScene& s, uint32_t dx, uint32_t dy, uint32_t dz, const PxVec3& meshLower,
                         uint32_t seed, bool patch)
{
    s.dx = dx; s.dy = dy; s.dz = dz;
    s.sg = 0;
    s.nbx = s.nby = s.nbz = 0;
    s.blkx = s.blky = s.blkz = 0;
    s.smin = -0.31f; s.smax = 0.47f;          // unused on the dense path
    s.meshLower = meshLower;
    s.spacing = 0.25f;
    std::mt19937 rng(seed);
    s.coarse.resize(size_t(dx) * dy * dz);
    for (auto& v : s.coarse)
    {
        v = -1.75f + 3.5f * float(rng() % 1000000u) / 1000000.f;
        if (v == 0.0f) v = 0.015625f;
    }
    s.starts.clear();
    s.sub.clear();
    if (!patch)
        return;
    // (i) The six face neighbours of FLAT as three equal PAIRS: the gradient is exactly
    // (0,0,0), which normalizeSafe leaves alone (|g| = 0 < eps).  The centre texel is
    // left generic.
    s.coarse[dIdx(s, FLAT_X + 1, FLAT_Y, FLAT_Z)] = 0.4375f;
    s.coarse[dIdx(s, FLAT_X - 1, FLAT_Y, FLAT_Z)] = 0.4375f;
    s.coarse[dIdx(s, FLAT_X, FLAT_Y + 1, FLAT_Z)] = -0.8125f;
    s.coarse[dIdx(s, FLAT_X, FLAT_Y - 1, FLAT_Z)] = -0.8125f;
    s.coarse[dIdx(s, FLAT_X, FLAT_Y, FLAT_Z + 1)] = 1.1875f;
    s.coarse[dIdx(s, FLAT_X, FLAT_Y, FLAT_Z - 1)] = 1.1875f;
    // (ii) A gradient of exactly (1e-21, 0, 0): BELOW the epsilon but NOT zero, so
    // normalizeSafe must leave it unchanged -- where getNormalized would give (1,0,0),
    // a `return 0` variant (0,0,0), and normalize() (1,0,0) or NaN.
    s.coarse[dIdx(s, TINY_X + 1, TINY_Y, TINY_Z)] = TINY_G;
    s.coarse[dIdx(s, TINY_X - 1, TINY_Y, TINY_Z)] = 0.0f;
    s.coarse[dIdx(s, TINY_X, TINY_Y + 1, TINY_Z)] = 0.3125f;
    s.coarse[dIdx(s, TINY_X, TINY_Y - 1, TINY_Z)] = 0.3125f;
    s.coarse[dIdx(s, TINY_X, TINY_Y, TINY_Z + 1)] = -0.5625f;
    s.coarse[dIdx(s, TINY_X, TINY_Y, TINY_Z - 1)] = -0.5625f;
    // (iii) A gradient of exactly (5e-20, 0, 0): FIVE TIMES the epsilon, so normalizeSafe
    // must normalise it.  (ii) only bounds the epsilon from below -- (1e-21,0,0)
    // and (0,0,0) stay unchanged under any LARGER epsilon too -- so without this point an
    // epsilon of 1e-12f, 1e-6f or 1e-3f passed every check (P9.95e check, measured).
    // Together (ii) and (iii) bound it only loosely, to (1e-21, 5e-20]; (iv) is the tight bracket.
    // |g| = sqrt(2.5e-39): the square is SUBNORMAL, and the point also pins that it is
    // not flushed (with FTZ the magnitude would be 0 and the gradient stay unchanged).
    // The subnormal square keeps only 21 significant bits, so the result is NOT exactly
    // (1,0,0): the engine's PxVec3::normalizeSafe gives x = 0.999999881f (0x3f7ffffe),
    // and so does the kernel (MEASURED on llvmpipe) -- an FTZ-on-the-square variant or a
    // magnitude computed in another way would show here as well.
    s.coarse[dIdx(s, NEAR_X + 1, NEAR_Y, NEAR_Z)] = NEAR_G;
    s.coarse[dIdx(s, NEAR_X - 1, NEAR_Y, NEAR_Z)] = 0.0f;
    s.coarse[dIdx(s, NEAR_X, NEAR_Y + 1, NEAR_Z)] = -0.6875f;
    s.coarse[dIdx(s, NEAR_X, NEAR_Y - 1, NEAR_Z)] = -0.6875f;
    s.coarse[dIdx(s, NEAR_X, NEAR_Y, NEAR_Z + 1)] = 0.8125f;
    s.coarse[dIdx(s, NEAR_X, NEAR_Y, NEAR_Z - 1)] = 0.8125f;
    // (iv) The TIGHT bracket: gradients of exactly (EPSLO_G,0,0) and (EPSHI_G,0,0), the
    // same pattern (x+1 = g, x-1 = +0, so PxVolumeGrad's g - (+0) is exactly g; equal
    // y/z pairs, so those lanes are +0).  (ii) and (iii) left the epsilon free in
    // (1e-21, 5e-20]: 4e-20f and 2e-21f passed every check (P9.95e post-check, measured).
    // |g| = sqrtf(g*g) with g*g ~1e-40 SUBNORMAL (17 significant bits), so the reachable
    // magnitudes near 1e-20f are coarse.  EPSLO_G and EPSHI_G are ADJACENT floats whose
    // squares are the ADJACENT subnormals 0x000116c2 / 0x000116c3; their magnitudes,
    // 9.99997303e-21 and 1.00000433e-20, are the reachable magnitudes on either side of
    // 1e-20f (no float s has sqrtf(s) == 1e-20f, which is why `<` vs `<=` is equivalent).
    // So EPSLO must stay unchanged and EPSHI be normalised, and any epsilon outside
    // (9.99997303e-21, 1.00000433e-20] -- relative -2.7e-6 .. +4.3e-6 -- flips one of
    // them.  An epsilon inside that interval is indistinguishable from 1e-20f by ANY input.
    s.coarse[dIdx(s, EPSLO_X + 1, EPSLO_Y, EPSLO_Z)] = EPSLO_G;
    s.coarse[dIdx(s, EPSLO_X - 1, EPSLO_Y, EPSLO_Z)] = 0.0f;
    s.coarse[dIdx(s, EPSLO_X, EPSLO_Y + 1, EPSLO_Z)] = 0.1875f;
    s.coarse[dIdx(s, EPSLO_X, EPSLO_Y - 1, EPSLO_Z)] = 0.1875f;
    s.coarse[dIdx(s, EPSLO_X, EPSLO_Y, EPSLO_Z + 1)] = -0.9375f;
    s.coarse[dIdx(s, EPSLO_X, EPSLO_Y, EPSLO_Z - 1)] = -0.9375f;
    s.coarse[dIdx(s, EPSHI_X + 1, EPSHI_Y, EPSHI_Z)] = EPSHI_G;
    s.coarse[dIdx(s, EPSHI_X - 1, EPSHI_Y, EPSHI_Z)] = 0.0f;
    s.coarse[dIdx(s, EPSHI_X, EPSHI_Y + 1, EPSHI_Z)] = 1.0625f;
    s.coarse[dIdx(s, EPSHI_X, EPSHI_Y - 1, EPSHI_Z)] = 1.0625f;
    s.coarse[dIdx(s, EPSHI_X, EPSHI_Y, EPSHI_Z + 1)] = -0.4375f;
    s.coarse[dIdx(s, EPSHI_X, EPSHI_Y, EPSHI_Z - 1)] = -0.4375f;
}

// The host bite of (iv): the ENGINE's PxVec3::normalizeSafe leaves (EPSLO_G,0,0)
// unchanged and normalises (EPSHI_G,0,0), and the two are adjacent floats.  If either
// failed, the device checks of these points would not bracket the epsilon.
static void checkEpsBracketHost(const char* label)
{
    const PxVec3 lo(EPSLO_G, 0.0f, 0.0f), hi(EPSHI_G, 0.0f, 0.0f);
    PxVec3 nLo = lo, nHi = hi;
    nLo.normalizeSafe();                                             // ENGINE
    nHi.normalizeSafe();                                             // ENGINE
    PXG_CHECK_MSG(std::nextafter(EPSLO_G, 1.0f) == EPSHI_G, "%s: EPSLO_G and EPSHI_G are adjacent floats", label);
    PXG_CHECK_MSG(std::memcmp(&nLo, &lo, sizeof(PxVec3)) == 0,
                  "%s: host normalizeSafe(%.9g,0,0) leaves it unchanged (got (%.9g,%.9g,%.9g))", label, EPSLO_G,
                  nLo.x, nLo.y, nLo.z);
    PXG_CHECK_MSG(nHi.x > 0.5f && nHi.y == 0.0f && nHi.z == 0.0f,
                  "%s: host normalizeSafe(%.9g,0,0) normalises it (got (%.9g,%.9g,%.9g))", label, EPSHI_G,
                  nHi.x, nHi.y, nHi.z);
}

struct Mesh
{
    const char* name = "";
    SdfScene s;
    bool hasSdf = false;
    BlobLayout blob;
    PxgBufferHandle buf = 0;
    uint64_t addr = 0;
    HostSdfBox box;
};

// Lays out, uploads the texture(s) and the blob.  projectionFlag = 1 on EVERY blob,
// M_none included, so a count that let `.w` in (B3) would see a non-zero term.
static bool uploadMesh(PxgSlangComputeDevice* dev, Mesh& m, uint32_t nv, uint32_t nt, uint32_t nodes)
{
    m.blob = layOutBlobEx(m.hasSdf ? &m.s : nullptr, nv, nt, nodes, 1u, 0x0BADF00Du, nullptr, nullptr);
    if (m.hasSdf)
    {
        if (m.s.sg == 0u)
        {
            const uint64_t h = sdfUploadTex(CU_AD_FORMAT_FLOAT, m.s.dx, m.s.dy, m.s.dz, m.s.coarse.data(), sizeof(float));
            if (h == 0) return false;
            std::memcpy(&m.blob.bytes[m.blob.texObjOffset], &h, 8);
        }
        else
        {
            const uint64_t hc = sdfUploadTex(CU_AD_FORMAT_FLOAT, m.s.nbx + 1, m.s.nby + 1, m.s.nbz + 1,
                                             m.s.coarse.data(), sizeof(float));
            const uint64_t hs = sdfUploadTex(CU_AD_FORMAT_UNSIGNED_INT8, m.s.blkx * (m.s.sg + 1),
                                             m.s.blky * (m.s.sg + 1), m.s.blkz * (m.s.sg + 1), m.s.sub.data(), 1);
            if (hc == 0 || hs == 0) return false;
            std::memcpy(&m.blob.bytes[m.blob.texObjOffset], &hc, 8);
            std::memcpy(&m.blob.bytes[m.blob.texObjSparseOffset], &hs, 8);
        }
        m.box = hostSdfBox(m.s);
    }
    m.buf = createBufferFrom(dev, m.blob.bytes);
    m.addr = dev->getBufferDeviceAddress(m.buf);
    return m.addr != 0;
}

// The point at texel coordinate g, exact on an aligned mesh.
static PxVec3 at(const HostSdfBox& b, float gx, float gy, float gz)
{
    return b.lower + PxVec3(gx * b.dx, gy * b.dx, gz * b.dx);
}

// The coordinates c on `axis` whose texel coordinate is EXACTLY `target`:
// fl(fl(c - lower) * invDx) == target, sdfcTexCoord's arithmetic (C4) on the host.  On a
// mesh whose spacing is not a power of two the naive lower + target*dx often misses by
// an ulp, so the search walks outward from it and returns every hit it meets.
static std::vector<float> guardCoords(const HostSdfBox& b, int axis, float target)
{
    std::vector<float> hits;
    const float lo = (&b.lower.x)[axis];
    const float start = lo + target * b.dx;
    float up = start, down = std::nextafter(start, -1e30f);
    for (int i = 0; i < 2048; ++i)
    {
        if ((up - lo) * b.invDx == target) hits.push_back(up);
        if ((down - lo) * b.invDx == target) hits.push_back(down);
        up = std::nextafter(up, 1e30f);
        down = std::nextafter(down, -1e30f);
    }
    return hits;
}

// ---------------------------------------------------------------------------
// The cross-check: npSdfCollision's own entries over one blob.
// ---------------------------------------------------------------------------
struct Cross
{
    std::vector<float> d;        // PxSdfDistance
    std::vector<PxVec3> g;       // PxVolumeGrad, raw
};

static bool crossCheck(PxgSlangComputeDevice* dev, uint64_t blobAddr, const std::vector<PxVec3>& pts, Cross& c)
{
    const uint32_t N = uint32_t(pts.size());
    c.d.assign(N, 0.0f);
    c.g.assign(N, PxVec3(0.0f));
    if (N == 0) return true;
    std::vector<F4> pb(N);
    for (uint32_t i = 0; i < N; ++i) pb[i] = F4{ pts[i].x, pts[i].y, pts[i].z, 0.0f };
    std::vector<F4> zero4(N, F4{ 0.f, 0.f, 0.f, 0.f });
    std::vector<float> outInit(size_t(N) * STRIDE_GRAD, -12345.f);
    const PxgBufferHandle bP = createBufferFrom(dev, pb);
    const PxgBufferHandle bZ = createBufferFrom(dev, zero4);
    const PxgBufferHandle bD = createBufferFrom(dev, outInit);
    const PxgBufferHandle bG = createBufferFrom(dev, outInit);
    P_Coll p = {};
    p.meshPtr = blobAddr;
    p.points = dev->getBufferDeviceAddress(bP);
    p.gradIn = dev->getBufferDeviceAddress(bZ);
    p.dirSeed = dev->getBufferDeviceAddress(bZ);
    p.numPoints = N;
    bool ok = true;
    for (int pass = 0; pass < 2; ++pass)
    {
        P_Coll q = p;
        q.outBuf = dev->getBufferDeviceAddress(pass == 0 ? bD : bG);
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", pass == 0 ? "sdfCollDistance" : "sdfCollGrad", {},
                 sizeof(P_Coll));
        if (!k.isValid()) { ok = false; break; }
        k.dispatch((N + 63) / 64, {}, q);
    }
    dev->flush();
    if (ok)
    {
        const auto D = downloadBuffer<float>(dev, bD, size_t(N) * STRIDE_GRAD);
        const auto G = downloadBuffer<float>(dev, bG, size_t(N) * STRIDE_GRAD);
        for (uint32_t i = 0; i < N; ++i)
        {
            c.d[i] = D[size_t(i) * STRIDE_DIST + 8];
            c.g[i] = PxVec3(G[size_t(i) * STRIDE_GRAD + 1], G[size_t(i) * STRIDE_GRAD + 2], G[size_t(i) * STRIDE_GRAD + 3]);
        }
    }
    dev->destroyBuffer(bP); dev->destroyBuffer(bZ); dev->destroyBuffer(bD); dev->destroyBuffer(bG);
    return ok;
}

// What the kernel must store for one point, from the cross-check: (normalizeSafe(g), d)
// with the ENGINE's PxVec3::normalizeSafe (PxVec3.h:307-314).
static PxVec4 crossWant(const Cross& c, uint32_t i)
{
    PxVec3 g = c.g[i];
    g.normalizeSafe();
    return PxVec4(g, c.d[i]);
}

// The independent anchor at an integer INTERIOR node of an aligned dense mesh:
// d = the stored texel (+ a zero diffMag), grad = the stored neighbour differences.
static bool anchorWant(const SdfScene& s, uint32_t x, uint32_t y, uint32_t z, PxVec4& want)
{
    if (x < 1 || y < 1 || z < 1 || x > s.dx - 2 || y > s.dy - 2 || z > s.dz - 2)
        return false;
    const float centre = s.coarse[dIdx(s, x, y, z)];
    if (centre == 0.0f)
        return false;            // F15: never a zero texel in the independent anchor
    PxVec3 g(s.coarse[dIdx(s, x + 1, y, z)] - s.coarse[dIdx(s, x - 1, y, z)],
             s.coarse[dIdx(s, x, y + 1, z)] - s.coarse[dIdx(s, x, y - 1, z)],
             s.coarse[dIdx(s, x, y, z + 1)] - s.coarse[dIdx(s, x, y, z - 1)]);
    g.normalizeSafe();                                               // ENGINE
    want = PxVec4(g, centre + 0.0f);
    return true;
}

// ---------------------------------------------------------------------------
// One launch of the kernel through the shim, on the engine grid.
// ---------------------------------------------------------------------------
struct Launch
{
    std::vector<PxgShape> shapes;
    std::vector<uint32_t> shapeRefs, counts;
    uint32_t maxPointCount = 0;
    std::vector<PxVec4> in;      // nb * maxPointCount
    // results
    std::vector<PxVec4> out;     // nb * maxPointCount + 1 trap
    std::vector<PxgShape> shapesBack;
    std::vector<uint32_t> refsBack, countsBack;
    std::vector<PxVec4> inBack;
};

static bool runKernel(PxgSlangComputeDevice* dev, PxgCudaShim& shim, Launch& L)
{
    const uint32_t nb = uint32_t(L.shapeRefs.size());
    const size_t slots = size_t(nb) * L.maxPointCount;
    std::vector<PxVec4> outInit(slots + 1, poison4());
    const PxgBufferHandle bS = createBufferFrom(dev, L.shapes);
    const PxgBufferHandle bR = createBufferFrom(dev, L.shapeRefs);
    const PxgBufferHandle bC = createBufferFrom(dev, L.counts);
    const PxgBufferHandle bI = createBufferFrom(dev, L.in);
    const PxgBufferHandle bO = createBufferFrom(dev, outInit);

    EngineBlob blob;   // PX_CUDA_KERNEL_PARAM order of :8912-8920
    blob.add<uint64_t>(dev->getBufferDeviceAddress(bS));
    blob.add<uint64_t>(dev->getBufferDeviceAddress(bR));
    blob.add<uint64_t>(dev->getBufferDeviceAddress(bI));
    blob.add<uint64_t>(dev->getBufferDeviceAddress(bC));
    blob.add<uint32_t>(L.maxPointCount);
    blob.add<uint64_t>(dev->getBufferDeviceAddress(bO));
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 48u, "engine blob is 48 bytes (got %zu)", blob.bytes.size());

    const uint32_t gridX = (L.maxPointCount + (SQ_BLOCK - 1)) / SQ_BLOCK;   // :8922
    bool ok = true;
    if (gridX > 0 && nb > 0)
    {
        ok = shim.launchKernel(0, gridX, nb, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) == PxgResult::Success;
        shim.streamSynchronize();
    }
    L.out = downloadBuffer<PxVec4>(dev, bO, slots + 1);
    L.shapesBack = downloadBuffer<PxgShape>(dev, bS, L.shapes.size());
    L.refsBack = downloadBuffer<uint32_t>(dev, bR, L.shapeRefs.size());
    L.countsBack = downloadBuffer<uint32_t>(dev, bC, L.counts.size());
    L.inBack = downloadBuffer<PxVec4>(dev, bI, L.in.size());
    dev->destroyBuffer(bS); dev->destroyBuffer(bR); dev->destroyBuffer(bC); dev->destroyBuffer(bI); dev->destroyBuffer(bO);
    return ok;
}

// The untouched-state checks shared by every test.
static void checkUntouched(const char* label, const Launch& L)
{
    const uint32_t nb = uint32_t(L.shapeRefs.size());
    size_t tail = 0;
    for (uint32_t b = 0; b < nb; ++b)
        for (uint32_t i = L.counts[b]; i < L.maxPointCount; ++i)
            if (!isPoison(L.out[size_t(b) * L.maxPointCount + i])) ++tail;
    PXG_CHECK_MSG(tail == 0, "%s: %zu slot(s) in [count, maxPointCount) were written (Q5)", label, tail);
    PXG_CHECK_MSG(isPoison(L.out.back()), "%s: the trap slot past the whole output was written", label);
    PXG_CHECK_MSG(std::memcmp(L.shapesBack.data(), L.shapes.data(), L.shapes.size() * sizeof(PxgShape)) == 0 &&
                  L.refsBack == L.shapeRefs && L.countsBack == L.counts &&
                  std::memcmp(L.inBack.data(), L.in.data(), L.in.size() * sizeof(PxVec4)) == 0,
                  "%s: an input buffer changed", label);
}

static PxgShape meshShape(uint64_t blobAddr, const PxMeshScale& scale, uint32_t material)
{
    PxgShape s;
    std::memset(&s, 0, sizeof(s));
    s.scale = scale;
    s.materialIndex = material;
    s.hullOrMeshPtr = size_t(blobAddr);
    s.type = PxGeometryType::eTRIANGLEMESH;
    s.particleOrSoftbodyId = 0xFFFFFFFFu;
    return s;
}

// The device, the shim and the launch-ABI row, checked once per test.
struct Harness
{
    PxgSlangComputeDevice* dev = nullptr;
    PxgCudaShim* shim = nullptr;
    std::vector<Mesh*> meshes;
    bool ok = false;

    Harness()
    {
        dev = testDevice();
        if (!dev->supportsBindlessTextures())
        {
            std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
            return;
        }
        const physx::gpuslang::PxgKernelAbiRow* row = physx::gpuslang::pxgFindKernelAbi(SQ_KERNEL);
        const uint32_t rowPush = row ? physx::gpuslang::pxgKernelAbiPushSize(*row) : 0u;
        PXG_CHECK_MSG(row != nullptr && row->argCount == 6u && rowPush == 48u && row->hostPtrMask == 0u,
                      "launch-ABI row {8,8,8,8,4,8} -> 48-byte blob, hostPtrMask 0 (got %u args, %u bytes)",
                      row ? row->argCount : 0u, rowPush);
        static PxgKernelDef manifest[1];
        manifest[0] = PxgKernelDef{ SQ_SPV, SQ_KERNEL, rowPush, 0u, SQ_KERNEL };
        static auto loader = [](const char* f) { return loadShaderBlob(f); };
        shim = new PxgCudaShim(dev, manifest, 1, +loader);
        physx::gpuslang::pxgSetVulkanDriverShim(shim);
        ok = row != nullptr && rowPush == 48u;
    }
    ~Harness()
    {
        for (Mesh* m : meshes) { if (m->buf) dev->destroyBuffer(m->buf); delete m; }
        if (shim)
        {
            physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
            delete shim;
        }
    }
    Mesh* add(const char* name) { Mesh* m = new Mesh; m->name = name; meshes.push_back(m); return m; }
};

static void setW(PxVec4& v, uint32_t i)
{
    // .w is never read (Q6): a NaN with a payload, +inf, -inf, junk.
    static const uint32_t w[4] = { 0x7FC0BEEFu, 0x7F800000u, 0xFF800000u, 0x4640E6B7u };
    std::memcpy(&v.w, &w[i % 4u], 4);
}

// The grid-aligned M_al and its points.
static void buildMAl(Mesh& m) { buildAligned(m.s, 16, 8, 32, PxVec3(-1.0f, 0.5f, 2.25f), 0x95E1u, true); m.hasSdf = true; }
static void buildMAl2(Mesh& m) { buildAligned(m.s, 8, 16, 4, PxVec3(0.5f, -1.25f, 0.75f), 0x95E2u, false); m.hasSdf = true; }

struct NodePt { PxVec3 p; int gx, gy, gz; };   // g == -1: not a texel centre

static std::vector<NodePt> pointsMAl(const HostSdfBox& b, uint32_t n)
{
    std::vector<NodePt> v;
    auto node = [&](int x, int y, int z) { v.push_back({ at(b, float(x), float(y), float(z)), x, y, z }); };
    auto free = [&](const PxVec3& p) { v.push_back({ p, -1, -1, -1 }); };
    // dims 16 x 8 x 32: the guard is f in [1,14] x [1,6] x [1,30]; the box f in [0,15] x [0,7] x [0,31].
    node(int(FLAT_X), int(FLAT_Y), int(FLAT_Z));
    node(int(TINY_X), int(TINY_Y), int(TINY_Z));
    node(int(NEAR_X), int(NEAR_Y), int(NEAR_Z));
    node(int(EPSLO_X), int(EPSLO_Y), int(EPSLO_Z));
    node(int(EPSHI_X), int(EPSHI_Y), int(EPSHI_Z));
    node(1, 1, 1); node(14, 6, 30);                     // on the guard, inside it
    node(0, 3, 5); node(15, 3, 5); node(4, 0, 5); node(4, 7, 5); node(4, 3, 0); node(4, 3, 31);  // faces
    node(0, 0, 0); node(15, 7, 31);                     // corners
    node(2, 4, 7); node(6, 1, 2); node(3, 5, 8);        // interior
    free(at(b, 4.5f, 2.5f, 6.5f));                      // half-texel
    free(at(b, 2.3f, 1.7f, 4.1f));                      // off-node
    free(at(b, 18.0f, 3.0f, 5.0f)); free(at(b, -3.0f, 3.0f, 5.0f));   // outside each face
    free(at(b, 4.0f, 10.0f, 5.0f)); free(at(b, 4.0f, -2.5f, 5.0f));
    free(at(b, 4.0f, 3.0f, 34.0f)); free(at(b, 4.0f, 3.0f, -1.75f));
    free(at(b, -2.0f, -2.0f, -2.0f)); free(at(b, 17.0f, 9.0f, 33.5f)); free(at(b, -1.0f, 8.5f, 31.0f));   // corners
    // -0 lanes: x = -0 is INSIDE M_al's box (f.x = 3.5); y = -0 and z = -0 are outside.
    free(PxVec3(-0.0f, at(b, 0, 2, 0).y, at(b, 0, 0, 6).z));
    free(PxVec3(at(b, 5, 0, 0).x, -0.0f, at(b, 0, 0, 4).z));
    free(PxVec3(at(b, 3, 0, 0).x, at(b, 0, 4, 0).y, -0.0f));
    std::mt19937 rng(0x5DF95u);
    while (v.size() < n)
    {
        const uint32_t kind = rng() % 4u;
        if (kind < 2u)
            node(int(rng() % 16u), int(rng() % 8u), int(rng() % 32u));
        else if (kind == 2u)
            free(at(b, -2.0f + float(rng() % 2000u) / 100.0f, -2.0f + float(rng() % 1200u) / 100.0f,
                    -2.0f + float(rng() % 3600u) / 100.0f));
        else
            free(at(b, 0.5f + float(rng() % 15u), 0.5f + float(rng() % 7u), 0.5f + float(rng() % 31u)));
    }
    // The five patched points again past index 1024 (block 1), when there is room.
    if (n > 1050)
    {
        v[1030] = v[0];
        v[1035] = v[3];
        v[1041] = v[2];
        v[1047] = v[4];
        v[n - 1] = v[1];
    }
    return v;
}

static std::vector<PxVec3> justP(const std::vector<NodePt>& v)
{
    std::vector<PxVec3> p;
    for (const NodePt& n : v) p.push_back(n.p);
    return p;
}

// Checks one batch against the cross-check and, where it applies, the anchor.
// Returns the number of live slots that differ.
static size_t checkBatch(const char* label, const Launch& L, uint32_t b, const Cross& c,
                         const std::vector<NodePt>* nodes, const SdfScene* aligned, size_t* anchored)
{
    size_t bad = 0, poisoned = 0;
    for (uint32_t i = 0; i < L.counts[b]; ++i)
    {
        const PxVec4& got = L.out[size_t(b) * L.maxPointCount + i];
        if (isPoison(got)) { ++poisoned; continue; }
        const PxVec4 want = crossWant(c, i);
        if (!bitEq4(got, want))
        {
            if (bad < 6)
                std::printf("    %s batch %u slot %u: got (%.9g, %.9g, %.9g, %.9g) want (%.9g, %.9g, %.9g, %.9g)\n",
                            label, b, i, got.x, got.y, got.z, got.w, want.x, want.y, want.z, want.w);
            ++bad;
        }
        if (nodes && aligned && (*nodes)[i].gx >= 0)
        {
            PxVec4 a;
            if (anchorWant(*aligned, uint32_t((*nodes)[i].gx), uint32_t((*nodes)[i].gy), uint32_t((*nodes)[i].gz), a))
            {
                ++*anchored;
                if (!bitEq4(got, a))
                {
                    if (bad < 6)
                        std::printf("    %s batch %u slot %u: ANCHOR got (%.9g, %.9g, %.9g, %.9g) want (%.9g, %.9g, %.9g, %.9g)\n",
                                    label, b, i, got.x, got.y, got.z, got.w, a.x, a.y, a.z, a.w);
                    ++bad;
                }
            }
        }
    }
    PXG_CHECK_MSG(poisoned == 0, "%s batch %u: %zu live slot(s) still poisoned", label, b, poisoned);
    return bad;
}

// ===========================================================================
// sdfQuery: the full arena.
// ===========================================================================
PXG_TEST(sdfQuery)
{
    Harness H;
    if (!H.shim) return;
    PXG_CHECK_MSG(H.ok, "harness");
    if (!H.ok) return;
    PxgSlangComputeDevice* dev = H.dev;

    Mesh& mAl = *H.add("M_al");      buildMAl(mAl);
    Mesh& mSp = *H.add("M_sparse");  buildSparse(mSp.s); mSp.hasSdf = true;
    Mesh& mAl2 = *H.add("M_al2");    buildMAl2(mAl2);
    Mesh& mNone = *H.add("M_none");
    Mesh& mDec = *H.add("M_decoy");  buildDense(mDec.s); mDec.hasSdf = true;
    // Distinct geometry-half sizes, so a blob walk with a wrong stride lands elsewhere
    // in every one of them.
    const bool up = uploadMesh(dev, mAl, 5u, 9u, 1u) && uploadMesh(dev, mSp, 16u, 21u, 3u) &&
                    uploadMesh(dev, mAl2, 7u, 4u, 2u) && uploadMesh(dev, mNone, 6u, 8u, 1u) &&
                    uploadMesh(dev, mDec, 3u, 2u, 1u);
    PXG_CHECK_MSG(up, "mesh upload");
    if (!up) return;

    // The shape table: no batch's ref equals its batch id (no fixed point), every other
    // slot is a VALID decoy mesh (a kill is a wrong answer, not a crash).
    Launch L;
    L.shapes.resize(8);
    for (uint32_t k = 0; k < 8; ++k) L.shapes[k] = meshShape(mDec.addr, PxMeshScale(PxVec3(3.0f), PxQuat(PxIdentity)), 90u + k);
    L.shapes[5] = meshShape(mAl.addr, PxMeshScale(1.0f), 5u);
    L.shapes[2] = meshShape(mSp.addr, PxMeshScale(1.0f), 2u);
    L.shapes[7] = meshShape(mAl2.addr, PxMeshScale(1.0f), 7u);
    L.shapes[0] = meshShape(mNone.addr, PxMeshScale(1.0f), 0u);
    // M_al again, under a non-identity, NEGATIVE-component scale and a scale rotation:
    // the kernel ignores PxMeshScale (Q8), so batch 3 must equal batch 0 bit for bit.
    L.shapes[6] = meshShape(mAl.addr, PxMeshScale(PxVec3(-1.5f, 2.0f, 0.75f),
                                                  PxQuat(0.7f, PxVec3(0.6f, 0.0f, 0.8f))), 6u);
    L.shapeRefs = { 5u, 2u, 7u, 6u, 0u };
    L.counts = { 1100u, 37u, 5u, 1100u, 9u };
    L.maxPointCount = 1100u;
    const uint32_t NB = 5u, M = L.maxPointCount;

    const std::vector<NodePt> p0 = pointsMAl(mAl.box, 1100);
    std::vector<NodePt> p1;   // M_sparse: nodes, between texels, outside
    {
        const HostSdfBox& b = mSp.box;
        const int nodes[][3] = { {5,2,3}, {9,5,15}, {1,1,1}, {10,6,18}, {0,4,10}, {12,4,10}, {8,3,7}, {4,2,2},
                                 {6,5,13}, {3,1,17}, {11,7,2}, {2,5,9} };
        for (auto& n : nodes) p1.push_back({ at(b, float(n[0]), float(n[1]), float(n[2])), n[0], n[1], n[2] });
        std::mt19937 rng(0x5BA5u);
        while (p1.size() < 37)
            p1.push_back({ at(b, -1.5f + float(rng() % 1500u) / 100.0f, -1.5f + float(rng() % 1100u) / 100.0f,
                              -1.5f + float(rng() % 2300u) / 100.0f), -1, -1, -1 });
    }
    std::vector<NodePt> p2;   // M_al2: interior nodes plus one outside point
    {
        const HostSdfBox& b = mAl2.box;
        p2.push_back({ at(b, 3, 4, 2), 3, 4, 2 });
        p2.push_back({ at(b, 1, 7, 2), 1, 7, 2 });
        p2.push_back({ at(b, 6, 1, 1), 6, 1, 1 });
        p2.push_back({ at(b, 2.5f, 3.25f, 1.75f), -1, -1, -1 });
        p2.push_back({ at(b, 9.5f, -1.0f, 2.0f), -1, -1, -1 });
    }
    std::vector<NodePt> p4;   // M_none: anything
    for (uint32_t i = 0; i < 9; ++i) p4.push_back({ PxVec3(0.1f * float(i), -0.2f, 0.3f), -1, -1, -1 });

    // Inputs: live slots carry the points (batch 3 = batch 0), dead slots distinct junk.
    L.in.resize(size_t(NB) * M);
    for (size_t s = 0; s < L.in.size(); ++s)
        L.in[s] = PxVec4(1000.0f + float(s), -2000.0f - float(s), 3000.0f + float(s), 7.0f);
    const std::vector<NodePt>* lists[5] = { &p0, &p1, &p2, &p0, &p4 };
    for (uint32_t b = 0; b < NB; ++b)
        for (uint32_t i = 0; i < L.counts[b]; ++i)
        {
            PxVec4 v((*lists[b])[i].p, 0.0f);
            setW(v, i + b);
            L.in[size_t(b) * M + i] = v;
        }
    // Batches 1 and 2 must not share their first points with batch 0 (q-in-no-base).
    for (uint32_t i = 0; i < 5; ++i)
        PXG_CHECK_MSG(!(p1[i].p == p0[i].p) && !(p2[i].p == p0[i].p), "batch 1/2 point %u equals batch 0's", i);

    // The cross-checks, each over its own blob.
    Cross c0, c1, c2, c2onAl;
    bool cok = crossCheck(dev, mAl.addr, justP(p0), c0) && crossCheck(dev, mSp.addr, justP(p1), c1) &&
               crossCheck(dev, mAl2.addr, justP(p2), c2) && crossCheck(dev, mAl.addr, justP(p2), c2onAl);
    PXG_CHECK_MSG(cok, "the npSdfCollision cross-check pipelines");
    if (!cok) return;

    // Bite asserts, on the reference values, before the kernel runs.
    {
        size_t divDiffers = 0, generic = 0;
        for (const Cross* c : { &c0, &c1, &c2 })
            for (const PxVec3& g : c->g)
            {
                const float mag = g.magnitude();
                if (!(mag >= 1e-20f)) continue;
                ++generic;
                const float inv = 1.0f / mag;
                if (g.x / mag != g.x * inv || g.y / mag != g.y * inv || g.z / mag != g.z * inv) ++divDiffers;
            }
        PXG_CHECK_MSG(divDiffers >= 1, "no gradient where v/mag != v*(1/mag) (%zu of %zu): q-norm-divide would survive",
                      divDiffers, generic);
        PXG_CHECK_MSG(c0.g[0] == PxVec3(0.0f) && c0.g[1] == PxVec3(TINY_G, 0.0f, 0.0f) &&
                      c0.g[2] == PxVec3(NEAR_G, 0.0f, 0.0f),
                      "the patched points give gradients (0,0,0), (1e-21,0,0) and (5e-20,0,0) (got (%g,%g,%g), "
                      "(%g,%g,%g), (%g,%g,%g))", c0.g[0].x, c0.g[0].y, c0.g[0].z, c0.g[1].x, c0.g[1].y, c0.g[1].z,
                      c0.g[2].x, c0.g[2].y, c0.g[2].z);
        // The 5x-epsilon point bites only if the ENGINE normalises it.
        const PxVec4 wNear = crossWant(c0, 2);
        PXG_CHECK_MSG(wNear.x > 0.5f && wNear.y == 0.0f && wNear.z == 0.0f,
                      "the engine's normalizeSafe of (5e-20,0,0) is ~(1,0,0) (got (%.9g,%.9g,%.9g))",
                      wNear.x, wNear.y, wNear.z);
        // The tight bracket (buildAligned (iv)): the cross-check reads exactly (EPSLO_G,0,0)
        // and (EPSHI_G,0,0), and the ENGINE leaves the first alone and normalises the second.
        checkEpsBracketHost("sdfQuery");
        PXG_CHECK_MSG(c0.g[3] == PxVec3(EPSLO_G, 0.0f, 0.0f) && c0.g[4] == PxVec3(EPSHI_G, 0.0f, 0.0f),
                      "the bracket points give gradients (%.9g,0,0) and (%.9g,0,0) (got (%.9g,%g,%g), (%.9g,%g,%g))",
                      EPSLO_G, EPSHI_G, c0.g[3].x, c0.g[3].y, c0.g[3].z, c0.g[4].x, c0.g[4].y, c0.g[4].z);
        const PxVec4 wLo = crossWant(c0, 3), wHi = crossWant(c0, 4);
        PXG_CHECK_MSG(bitEq4(wLo, PxVec4(EPSLO_G, 0.0f, 0.0f, c0.d[3])),
                      "the engine's normalizeSafe leaves (%.9g,0,0) unchanged (got (%.9g,%.9g,%.9g))", EPSLO_G,
                      wLo.x, wLo.y, wLo.z);
        PXG_CHECK_MSG(wHi.x > 0.5f && wHi.y == 0.0f && wHi.z == 0.0f,
                      "the engine's normalizeSafe of (%.9g,0,0) is ~(1,0,0) (got (%.9g,%.9g,%.9g))", EPSHI_G,
                      wHi.x, wHi.y, wHi.z);
        size_t outside = 0;
        for (const NodePt& n : p0)
            if (!(n.p == n.p.maximum(mAl.box.lower).minimum(mAl.box.higher))) ++outside;
        PXG_CHECK_MSG(outside >= 20, "batch 0 has %zu outside points", outside);
    }

    const bool launched = runKernel(dev, *H.shim, L);
    PXG_CHECK_MSG(launched, "launch %s through the shim (grid (2,5,1))", SQ_KERNEL);
    if (!launched) return;

    size_t anchored = 0, anchoredAl2 = 0;
    size_t bad = checkBatch("sdfQuery", L, 0, c0, &p0, &mAl.s, &anchored);
    bad += checkBatch("sdfQuery", L, 1, c1, nullptr, nullptr, nullptr);
    bad += checkBatch("sdfQuery", L, 2, c2, &p2, &mAl2.s, &anchoredAl2);
    bad += checkBatch("sdfQuery", L, 3, c0, &p0, &mAl.s, &anchored);
    PXG_CHECK_MSG(bad == 0, "sdfQuery: %zu slot(s) differ bitwise from convexMeshMidphase.cu:1282-1286", bad);
    PXG_CHECK_MSG(anchored >= 2 * 200 && anchoredAl2 == 3, "sdfQuery: %zu / %zu independent anchors", anchored, anchoredAl2);

    // P9.96: the node slots' output BITS, recorded on the pre-P9.96 sampler and
    // asserted unchanged after the filter was replaced.  Every node of the aligned
    // meshes (borders included, anchorable or not) sits at an integer texel
    // coordinate, so every tap the kernel takes there has weight 0 or 1 under both
    // filters; this pins that the argument holds in fact, not only on paper.
    {
        uint64_t h = 1469598103934665603ull;   // FNV-1a 64
        size_t nodes = 0;
        for (uint32_t b : { 0u, 2u })
        {
            const std::vector<NodePt>& lst = b == 0u ? p0 : p2;
            for (uint32_t i = 0; i < L.counts[b]; ++i)
            {
                if (lst[i].gx < 0) continue;
                ++nodes;
                const uint8_t* q = reinterpret_cast<const uint8_t*>(&L.out[size_t(b) * M + i]);
                for (size_t k = 0; k < sizeof(PxVec4); ++k) { h ^= q[k]; h *= 1099511628211ull; }
            }
        }
        std::printf("    [ note ] node-slot bits: FNV-1a 0x%016llx over %zu slots\n", (unsigned long long)h, nodes);
        // Recorded on the pre-P9.96 tree (the Vulkan sampler, d0b07e4) with this exact
        // point set; the P9.96 filter must reproduce it.
        PXG_CHECK_MSG(h == 0x73976e65db8d123bull && nodes == 570u,
                      "sdfQuery: the node slots' bits changed (FNV-1a 0x%016llx over %zu slots, recorded "
                      "0x73976e65db8d123b over 570)", (unsigned long long)h, nodes);
    }

    // The patched points, exactly.
    {
        const float dFlat = mAl.s.coarse[dIdx(mAl.s, FLAT_X, FLAT_Y, FLAT_Z)];
        const float dTiny = mAl.s.coarse[dIdx(mAl.s, TINY_X, TINY_Y, TINY_Z)];
        const float dNear = mAl.s.coarse[dIdx(mAl.s, NEAR_X, NEAR_Y, NEAR_Z)];
        PxVec3 gNear(NEAR_G, 0.0f, 0.0f);
        gNear.normalizeSafe();                                           // ENGINE
        const float dLo = mAl.s.coarse[dIdx(mAl.s, EPSLO_X, EPSLO_Y, EPSLO_Z)];
        const float dHi = mAl.s.coarse[dIdx(mAl.s, EPSHI_X, EPSHI_Y, EPSHI_Z)];
        PxVec3 gHi(EPSHI_G, 0.0f, 0.0f);
        gHi.normalizeSafe();                                             // ENGINE
        const PxVec4 flat(0.0f, 0.0f, 0.0f, dFlat), tiny(TINY_G, 0.0f, 0.0f, dTiny), nearN(gNear, dNear);
        const PxVec4 lo(EPSLO_G, 0.0f, 0.0f, dLo), hi(gHi, dHi);
        for (uint32_t b : { 0u, 3u })
        {
            PXG_CHECK_MSG(bitEq4(L.out[size_t(b) * M + 0], flat) && bitEq4(L.out[size_t(b) * M + 1030], flat),
                          "batch %u: the flat gradient stays (0,0,0)", b);
            PXG_CHECK_MSG(bitEq4(L.out[size_t(b) * M + 1], tiny) && bitEq4(L.out[size_t(b) * M + 1099], tiny),
                          "batch %u: the sub-epsilon gradient stays (1e-21,0,0) -- normalizeSafe leaves it alone", b);
            PXG_CHECK_MSG(bitEq4(L.out[size_t(b) * M + 2], nearN) && bitEq4(L.out[size_t(b) * M + 1041], nearN),
                          "batch %u: the 5x-epsilon gradient (5e-20,0,0) is normalised (got x = %.9g)", b,
                          L.out[size_t(b) * M + 2].x);
            PXG_CHECK_MSG(bitEq4(L.out[size_t(b) * M + 3], lo) && bitEq4(L.out[size_t(b) * M + 1035], lo),
                          "batch %u: (%.9g,0,0), |g| = 9.99997303e-21 just below the epsilon, stays unchanged "
                          "(got x = %.9g)", b, EPSLO_G, L.out[size_t(b) * M + 3].x);
            PXG_CHECK_MSG(bitEq4(L.out[size_t(b) * M + 4], hi) && bitEq4(L.out[size_t(b) * M + 1047], hi),
                          "batch %u: (%.9g,0,0), |g| = 1.00000433e-20 just above the epsilon, is normalised "
                          "(got x = %.9g)", b, EPSHI_G, L.out[size_t(b) * M + 4].x);
        }
    }

    // M_sparse, every point, BIT FOR BIT against the host's PxSdfDistance: the box clamp
    // and the texel coordinate exactly as sdfcClampToBox / sdfcTexCoord compute them,
    // sdfSampleRef for Sample(), plus PxVec3::magnitude of the clamp offset (C3).  The
    // nodes keep sparseOracle (the engine's decodeSparse) as a secondary BOUND: the
    // spacing is 0.11, so a node's texel coordinate is an integer only to within an ulp,
    // and the coarse branch of the oracle is the engine's PxTriLerp (pxgSlangTex3d K2).
    {
        const SdfRefVols vols = sdfRefVols(mSp.s);
        const HostSdfBox& bx = mSp.box;
        const double bound = 0.5 * double(mSp.s.smax - mSp.s.smin) / 255.0 + 1e-4;
        size_t n = 0, off = 0, bad = 0;
        for (uint32_t i = 0; i < 37; ++i)
        {
            const PxVec3 p = p1[i].p;
            const PxVec3 hi(std::max(p.x, bx.lower.x), std::max(p.y, bx.lower.y), std::max(p.z, bx.lower.z));
            const PxVec3 cl(std::min(hi.x, bx.higher.x), std::min(hi.y, bx.higher.y), std::min(hi.z, bx.higher.z));
            const PxVec3 f = (cl - bx.lower) * bx.invDx;
            const float want = sdfSampleRef(mSp.s, vols, f) + (p - cl).magnitude();
            const float got = L.out[size_t(1) * M + i].w;
            if (f2u(got) != f2u(want))
            {
                if (bad < 5)
                    std::printf("    M_sparse point %u: %.9g vs host PxSdfDistance %.9g\n", i, got, want);
                ++bad;
            }
            if (p1[i].gx < 0) continue;
            ++n;
            const float o = sparseOracle(mSp.s, p1[i].gx, p1[i].gy, p1[i].gz);
            if (std::fabs(double(got) - double(o)) > bound)
            {
                std::printf("    M_sparse node (%d,%d,%d): %.9g vs sparseOracle %.9g\n", p1[i].gx, p1[i].gy, p1[i].gz,
                            got, o);
                ++off;
            }
        }
        PXG_CHECK_MSG(bad == 0, "M_sparse: %zu of 37 distances differ from the host PxSdfDistance", bad);
        PXG_CHECK_MSG(n == 12 && off == 0, "M_sparse: %zu of %zu node distances outside the decodeSparse bound", off, n);
    }

    // Q8: the scaled, rotated shape over the same mesh answers identically.
    PXG_CHECK_MSG(std::memcmp(&L.out[3 * size_t(M)], &L.out[0], size_t(M) * sizeof(PxVec4)) == 0,
                  "batch 3 (scaled, rotated M_al) differs from batch 0 -- the kernel must ignore PxMeshScale");
    // D1: a mesh without an SDF writes nothing.
    {
        size_t w = 0;
        for (uint32_t i = 0; i < M; ++i) if (!isPoison(L.out[4 * size_t(M) + i])) ++w;
        PXG_CHECK_MSG(w == 0, "batch 4 (no SDF): %zu slot(s) written (D1: the port writes nothing)", w);
    }
    // Batch 2's mesh is not M_al: at least one answer must differ from M_al's.
    {
        size_t differ = 0;
        for (uint32_t i = 0; i < 5; ++i)
            if (!bitEq4(crossWant(c2onAl, i), L.out[2 * size_t(M) + i])) ++differ;
        PXG_CHECK_MSG(differ >= 1, "batch 2 answers like M_al at every point -- a shape mixup would be invisible");
    }
    checkUntouched("sdfQuery", L);
}

// ===========================================================================
// sdfQueryNormalizeSafe: the five patched M_al points alone (plus one generic anchor).
// ===========================================================================
PXG_TEST(sdfQueryNormalizeSafe)
{
    Harness H;
    if (!H.shim) return;
    PXG_CHECK_MSG(H.ok, "harness");
    if (!H.ok) return;
    Mesh& mAl = *H.add("M_al");
    buildMAl(mAl);
    PXG_CHECK_MSG(uploadMesh(H.dev, mAl, 5u, 9u, 1u), "mesh upload");

    Launch L;
    L.shapes = { meshShape(mAl.addr, PxMeshScale(1.0f), 0u), meshShape(mAl.addr, PxMeshScale(1.0f), 1u) };
    L.shapeRefs = { 1u };
    L.counts = { 6u };
    L.maxPointCount = 7u;
    L.in.assign(7, PxVec4(9.0f, 9.0f, 9.0f, 9.0f));
    L.in[0] = PxVec4(at(mAl.box, float(FLAT_X), float(FLAT_Y), float(FLAT_Z)), 0.0f);
    L.in[1] = PxVec4(at(mAl.box, float(TINY_X), float(TINY_Y), float(TINY_Z)), 0.0f);
    L.in[2] = PxVec4(at(mAl.box, 2.0f, 4.0f, 7.0f), 0.0f);
    L.in[3] = PxVec4(at(mAl.box, float(NEAR_X), float(NEAR_Y), float(NEAR_Z)), 0.0f);
    L.in[4] = PxVec4(at(mAl.box, float(EPSLO_X), float(EPSLO_Y), float(EPSLO_Z)), 0.0f);
    L.in[5] = PxVec4(at(mAl.box, float(EPSHI_X), float(EPSHI_Y), float(EPSHI_Z)), 0.0f);
    for (uint32_t i = 0; i < 7; ++i) setW(L.in[i], i);
    PXG_CHECK_MSG(runKernel(H.dev, *H.shim, L), "launch");

    const float dFlat = mAl.s.coarse[dIdx(mAl.s, FLAT_X, FLAT_Y, FLAT_Z)];
    const float dTiny = mAl.s.coarse[dIdx(mAl.s, TINY_X, TINY_Y, TINY_Z)];
    PXG_CHECK_MSG(bitEq4(L.out[0], PxVec4(0.0f, 0.0f, 0.0f, dFlat)),
                  "flat: got (%g, %g, %g, %g)", L.out[0].x, L.out[0].y, L.out[0].z, L.out[0].w);
    PXG_CHECK_MSG(bitEq4(L.out[1], PxVec4(TINY_G, 0.0f, 0.0f, dTiny)),
                  "sub-epsilon: got (%g, %g, %g, %g)", L.out[1].x, L.out[1].y, L.out[1].z, L.out[1].w);
    PxVec4 a;
    PXG_CHECK_MSG(anchorWant(mAl.s, 2, 4, 7, a) && bitEq4(L.out[2], a),
                  "generic: got (%.9g, %.9g, %.9g, %.9g) want (%.9g, %.9g, %.9g, %.9g)",
                  L.out[2].x, L.out[2].y, L.out[2].z, L.out[2].w, a.x, a.y, a.z, a.w);
    // Five times the epsilon: the engine's PxVec3::normalizeSafe of (5e-20,0,0), ~(1,0,0).
    PxVec3 gNear(NEAR_G, 0.0f, 0.0f);
    gNear.normalizeSafe();
    const float dNear = mAl.s.coarse[dIdx(mAl.s, NEAR_X, NEAR_Y, NEAR_Z)];
    PXG_CHECK_MSG(gNear.x > 0.5f && gNear.y == 0.0f && gNear.z == 0.0f,
                  "host normalizeSafe(5e-20,0,0) = (%.9g,%.9g,%.9g)", gNear.x, gNear.y, gNear.z);
    PXG_CHECK_MSG(bitEq4(L.out[3], PxVec4(gNear, dNear)),
                  "5x epsilon: got (%.9g, %.9g, %.9g, %.9g)", L.out[3].x, L.out[3].y, L.out[3].z, L.out[3].w);
    // The tight bracket (buildAligned (iv)), each bit-equal to the ENGINE's normalizeSafe:
    // (EPSLO_G,0,0) unchanged, (EPSHI_G,0,0) normalised.
    checkEpsBracketHost("sdfQueryNormalizeSafe");
    PxVec3 gLo(EPSLO_G, 0.0f, 0.0f), gHi(EPSHI_G, 0.0f, 0.0f);
    gLo.normalizeSafe();                                             // ENGINE
    gHi.normalizeSafe();                                             // ENGINE
    const float dLo = mAl.s.coarse[dIdx(mAl.s, EPSLO_X, EPSLO_Y, EPSLO_Z)];
    const float dHi = mAl.s.coarse[dIdx(mAl.s, EPSHI_X, EPSHI_Y, EPSHI_Z)];
    PXG_CHECK_MSG(bitEq4(L.out[4], PxVec4(gLo, dLo)) && bitEq4(L.out[4], PxVec4(EPSLO_G, 0.0f, 0.0f, dLo)),
                  "just below epsilon (|g| = 9.99997303e-21), unchanged: got (%.9g, %.9g, %.9g, %.9g)",
                  L.out[4].x, L.out[4].y, L.out[4].z, L.out[4].w);
    PXG_CHECK_MSG(bitEq4(L.out[5], PxVec4(gHi, dHi)),
                  "just above epsilon (|g| = 1.00000433e-20), normalised: got (%.9g, %.9g, %.9g, %.9g) want x = %.9g",
                  L.out[5].x, L.out[5].y, L.out[5].z, L.out[5].w, gHi.x);
    checkUntouched("sdfQueryNormalizeSafe", L);
}

// ===========================================================================
// sdfQueryGrid: the block edges.  A zero grid is never dispatched (the host computes
// gridX = 0 only for maxPointCount 0, and the shim treats that as an EMPTY-GRID no-op).
// ===========================================================================
PXG_TEST(sdfQueryGrid)
{
    Harness H;
    if (!H.shim) return;
    PXG_CHECK_MSG(H.ok, "harness");
    if (!H.ok) return;
    Mesh& mAl = *H.add("M_al");
    buildMAl(mAl);
    PXG_CHECK_MSG(uploadMesh(H.dev, mAl, 5u, 9u, 1u), "mesh upload");

    struct Case { const char* name; uint32_t maxP, count; };
    const Case cases[3] = { { "max 1024: the last thread of block 0 is live", 1024u, 1024u },
                            { "max 1025: block 1 has exactly one live thread", 1025u, 1025u },
                            { "count 0: nothing is written", 1024u, 0u } };
    for (const Case& K : cases)
    {
        Launch L;
        L.shapes = { meshShape(mAl.addr, PxMeshScale(1.0f), 0u), meshShape(mAl.addr, PxMeshScale(1.0f), 1u) };
        L.shapeRefs = { 1u };
        L.counts = { K.count };
        L.maxPointCount = K.maxP;
        const std::vector<NodePt> pts = pointsMAl(mAl.box, K.maxP);
        L.in.resize(K.maxP);
        for (uint32_t i = 0; i < K.maxP; ++i) { L.in[i] = PxVec4(pts[i].p, 0.0f); setW(L.in[i], i); }
        std::vector<NodePt> live(pts.begin(), pts.begin() + K.count);
        Cross c;
        PXG_CHECK_MSG(crossCheck(H.dev, mAl.addr, justP(live), c), "%s: cross-check", K.name);
        PXG_CHECK_MSG(runKernel(H.dev, *H.shim, L), "%s: launch", K.name);
        size_t anchored = 0;
        const size_t bad = checkBatch(K.name, L, 0, c, &live, &mAl.s, &anchored);
        PXG_CHECK_MSG(bad == 0, "%s: %zu slot(s) differ", K.name, bad);
        if (K.count > 0)
            PXG_CHECK_MSG(!isPoison(L.out[K.count - 1]), "%s: the last live slot %u was written", K.name, K.count - 1);
        checkUntouched(K.name, L);
    }
}

// ===========================================================================
// sdfQueryGuard: PxVolumeGrad's inclusive guard (pxgSlangSdfCollision C7) as this
// kernel reaches it.  (Its own random source, so a change to the arena cannot move it.)
//
// On a grid-aligned mesh the two gradient paths agree EXACTLY on the guard -- the
// world-space taps p +- dx land on the texel-space taps f +- 1 -- so `>=` against `>`
// is invisible there (measured: both c7 mutants survived the first sweep).  Only a
// mesh with INEXACT texel coordinates tells them apart, at a point placed with f
// EXACTLY on a bound whose world taps nevertheless land a rounding error off the
// neighbouring texel centres.  Such points are rare: with spacing 0.07 (testSdfQuery's
// first try, the decoy mesh) NO float puts f exactly on four of the six bounds, and
// with a power-of-two spacing the taps are exact again.  The mesh below (dims 8 x 4 x
// 16, spacing 0.39656, meshLower (-0.5536, -1.00892, -0.62012)) came out of a host
// search for a configuration where all six bounds have such a point.  Each bound gets
// one, chosen so that the two paths' NORMALISED gradients differ bitwise -- checked
// here, before the kernel runs, from npSdfCollision's own distances at the six
// fall-back taps.
//
// P9.96 CHANGED WHAT "A ROUNDING ERROR OFF" BUYS.  pxgTex3D now rounds each filter
// weight to the nearest 1/256 (CUDA's 1.8 fixed point), so a tap an ulp off a texel
// coordinate reads EXACTLY what the exact tap reads unless that ulp moves a weight
// across a rounding boundary.  Measured: on the first 32 random base points four of the
// six bounds no longer had a telling point, and 64 rounds of random points (114,688
// candidates) found none either.  What still tells the paths apart is a coordinate that
// sits EXACTLY on a weight tie, f = j + (m + 0.5)/256, where the fall-back tap's ulp of
// error rounds the weight the other way.  So the first round keeps the pre-P9.96 random
// points and every later round places all three coordinates on such ties; two rounds
// suffice (printed).  The kernel check itself is unchanged.
// ===========================================================================
PXG_TEST(sdfQueryGuard)
{
    Harness H;
    if (!H.shim) return;
    PXG_CHECK_MSG(H.ok, "harness");
    if (!H.ok) return;
    Mesh& mD = *H.add("M_guard");
    buildAligned(mD.s, 8, 4, 16, PxVec3(-0.553599954f, -1.00891995f, -0.620120049f), 0x6A2Eu, false);
    mD.s.spacing = 0.396560013f;      // NOT a power of two: the point of this mesh
    mD.hasSdf = true;
    PXG_CHECK_MSG(uploadMesh(H.dev, mD, 3u, 2u, 1u), "mesh upload");
    const HostSdfBox& b = mD.box;

    // Candidates: every exact guard coordinate, over interior base points drawn in
    // rounds of 32 until every bound has a point that tells the two paths apart.  (The
    // first round is the pre-P9.96 set; see the P9.96 note above for why more rounds
    // are needed now.)
    std::mt19937 rng(0x6A2Du);
    std::vector<PxVec3> pts;
    bool have[6] = { false, false, false, false, false, false };
    int rounds = 0;
    size_t tried = 0;
    const float dx = b.dx;
    for (; rounds < 64 && !(have[0] && have[1] && have[2] && have[3] && have[4] && have[5]); ++rounds)
    {
        std::vector<PxVec3> cand;
        std::vector<int> candBound;
        for (int base = 0; base < 32; ++base)
        {
            PxVec3 q0 = at(b, 1.0f + float(rng() % 500u) / 100.0f, 1.0f + float(rng() % 100u) / 100.0f,
                           1.0f + float(rng() % 1300u) / 100.0f);
            if (rounds > 0)
            {
                // P9.96: every coordinate on an exact weight TIE, f = j + (m + 0.5)/256,
                // where an ulp of tap error decides which way the 1/256 weight rounds.
                for (int axis = 0; axis < 3; ++axis)
                {
                    const float target = 1.0f + float(rng() % (b.dims[axis] - 3u)) + (float(rng() % 256u) + 0.5f) / 256.0f;
                    const std::vector<float> hit = guardCoords(b, axis, target);
                    if (!hit.empty()) (&q0.x)[axis] = hit[0];
                }
            }
            for (int k = 0; k < 6; ++k)
            {
                const int axis = k / 2;
                const float target = (k % 2 == 0) ? 1.0f : float(b.dims[axis]) - 2.0f;
                for (float c : guardCoords(b, axis, target))
                {
                    PxVec3 q = q0;
                    (&q.x)[axis] = c;
                    cand.push_back(q);
                    candBound.push_back(k);
                }
            }
        }
        std::vector<PxVec3> taps;
        for (const PxVec3& q : cand)
        {
            taps.push_back(q + PxVec3(dx, 0.0f, 0.0f)); taps.push_back(q - PxVec3(dx, 0.0f, 0.0f));
            taps.push_back(q + PxVec3(0.0f, dx, 0.0f)); taps.push_back(q - PxVec3(0.0f, dx, 0.0f));
            taps.push_back(q + PxVec3(0.0f, 0.0f, dx)); taps.push_back(q - PxVec3(0.0f, 0.0f, dx));
        }
        Cross ci, ct;
        PXG_CHECK_MSG(crossCheck(H.dev, mD.addr, cand, ci) && crossCheck(H.dev, mD.addr, taps, ct), "cross-check");
        tried += cand.size();
        for (size_t j = 0; j < cand.size(); ++j)
        {
            const int k = candBound[j];
            if (have[k]) continue;
            PxVec3 fb(ct.d[6 * j + 0] - ct.d[6 * j + 1], ct.d[6 * j + 2] - ct.d[6 * j + 3], ct.d[6 * j + 4] - ct.d[6 * j + 5]);
            PxVec3 in = ci.g[j];
            fb.normalizeSafe();
            in.normalizeSafe();
            if (std::memcmp(&fb, &in, sizeof(PxVec3)) != 0)
            {
                have[k] = true;
                pts.push_back(cand[j]);
            }
        }
    }
    std::printf("    [ note ] sdfQueryGuard: %d round(s) of 32 base points, %zu guard candidates\n", rounds, tried);
    for (int k = 0; k < 6; ++k)
        PXG_CHECK_MSG(have[k], "bound %d (axis %d, %s): no point on it tells the two gradient paths apart", k, k / 2,
                      (k % 2 == 0) ? "f == 1" : "f == dims - 2");
    if (pts.size() != 6) return;

    Launch L;
    L.shapes = { meshShape(mD.addr, PxMeshScale(1.0f), 0u), meshShape(mD.addr, PxMeshScale(1.0f), 1u) };
    L.shapeRefs = { 1u };
    L.counts = { 6u };
    L.maxPointCount = 8u;
    L.in.assign(8, PxVec4(5.0f, 6.0f, 7.0f, 8.0f));
    for (uint32_t i = 0; i < 6; ++i) { L.in[i] = PxVec4(pts[i], 0.0f); setW(L.in[i], i); }
    Cross c;
    PXG_CHECK_MSG(crossCheck(H.dev, mD.addr, pts, c), "cross-check");
    PXG_CHECK_MSG(runKernel(H.dev, *H.shim, L), "launch");
    const size_t bad = checkBatch("sdfQueryGuard", L, 0, c, nullptr, nullptr, nullptr);
    PXG_CHECK_MSG(bad == 0, "sdfQueryGuard: %zu of 6 guard points differ", bad);
    checkUntouched("sdfQueryGuard", L);
}

// ===========================================================================
// sdfQueryNonPow2 (P9.96): the P9.95e finding as a regression test.  buildAligned with
// dims 10 x 7 x 11 -- none a power of two -- and 1024 points: nodes everywhere on the
// grid (borders included), half-texel points, and free points inside and outside the
// box.  Every slot must equal the npSdfCollision cross-check bit for bit, and every
// interior node the independent anchor (the stored texel and its neighbour
// differences).  Before P9.96 the sampler was handed (k + 0.5) / dim, a round trip that
// is inexact on these extents, and the anchors missed; since P9.96 the filter runs in
// texel space and a texel centre is exact for every extent.  The cross-check imports
// the same pxgSlangTex3d module as the kernel, so it cannot see a filter defect that
// keeps texel centres exact; every slot's DISTANCE is therefore also compared bit for
// bit with the host's PxSdfDistance (as M_sparse is in sdfQuery) -- the half-texel and
// free points included, which no anchor reaches.
// ===========================================================================
PXG_TEST(sdfQueryNonPow2)
{
    Harness H;
    if (!H.shim) return;
    PXG_CHECK_MSG(H.ok, "harness");
    if (!H.ok) return;
    Mesh& mN = *H.add("M_np2");
    buildAligned(mN.s, 10, 7, 11, PxVec3(-1.25f, 0.75f, -0.5f), 0x96A2u, false);
    mN.hasSdf = true;
    PXG_CHECK_MSG(uploadMesh(H.dev, mN, 4u, 6u, 1u), "mesh upload");

    const uint32_t NP = 1024u;
    std::vector<NodePt> pts;
    {
        const HostSdfBox& b = mN.box;
        auto node = [&](int x, int y, int z) { pts.push_back({ at(b, float(x), float(y), float(z)), x, y, z }); };
        // Every interior node once (8 x 5 x 9 = 360), then the eight corners.
        for (int z = 1; z <= 9; ++z)
            for (int y = 1; y <= 5; ++y)
                for (int x = 1; x <= 8; ++x)
                    node(x, y, z);
        for (int c = 0; c < 8; ++c) node((c & 1) ? 9 : 0, (c & 2) ? 6 : 0, (c & 4) ? 10 : 0);
        std::mt19937 rng(0x96A3u);
        while (pts.size() < NP)
        {
            const uint32_t kind = rng() % 3u;
            if (kind == 0u)
                node(int(rng() % 10u), int(rng() % 7u), int(rng() % 11u));
            else if (kind == 1u)
                pts.push_back({ at(b, 0.5f + float(rng() % 9u), 0.5f + float(rng() % 6u), 0.5f + float(rng() % 10u)),
                                -1, -1, -1 });
            else
                pts.push_back({ at(b, -2.0f + float(rng() % 1400u) / 100.0f, -2.0f + float(rng() % 1100u) / 100.0f,
                                   -2.0f + float(rng() % 1500u) / 100.0f), -1, -1, -1 });
        }
    }
    Launch L;
    L.shapes = { meshShape(mN.addr, PxMeshScale(1.0f), 0u), meshShape(mN.addr, PxMeshScale(1.0f), 1u) };
    L.shapeRefs = { 1u };
    L.counts = { NP };
    L.maxPointCount = NP;
    L.in.resize(NP);
    for (uint32_t i = 0; i < NP; ++i) { L.in[i] = PxVec4(pts[i].p, 0.0f); setW(L.in[i], i); }
    Cross c;
    PXG_CHECK_MSG(crossCheck(H.dev, mN.addr, justP(pts), c), "cross-check");
    PXG_CHECK_MSG(runKernel(H.dev, *H.shim, L), "launch");
    size_t anchored = 0;
    const size_t bad = checkBatch("sdfQueryNonPow2", L, 0, c, &pts, &mN.s, &anchored);
    PXG_CHECK_MSG(anchored >= 360, "sdfQueryNonPow2: only %zu interior nodes anchored", anchored);
    PXG_CHECK_MSG(bad == 0, "sdfQueryNonPow2: %zu of %u slot(s) differ (cross-check or texel anchor) on a "
                  "10 x 7 x 11 grid", bad, NP);
    // The host's PxSdfDistance for every slot: the box clamp and the texel coordinate
    // exactly as sdfcClampToBox / sdfcTexCoord compute them, sdfSampleRef for Sample()
    // (pxgtex::tex3D at f + 0.5 on this dense mesh), plus PxVec3::magnitude of the
    // clamp offset (C3).
    // Bite asserts: every non-equivalent filter variant must change at least one of
    // these host distances, so a module defect that keeps texel centres exact -- which
    // the cross-check (same module) and the anchors (centres only) would both miss --
    // still fails here.  Exempt: the out-of-domain family, for testSdfTexture's
    // reasons (the engine stays inside [0.5, N - 0.5]; no -0 texel; w > h here), and
    // the two tie rules -- no point here puts a weight on an exact 1/512 tie (the tie
    // passes of npTex3dBindless and sdfTexture do).
    size_t dBad = 0;
    {
        const SdfRefVols vols = sdfRefVols(mN.s);
        const HostSdfBox& bx = mN.box;
        std::vector<PxVec3> fs(NP), offs(NP);
        std::vector<float> wants(NP);
        for (uint32_t i = 0; i < NP; ++i)
        {
            const PxVec3 p = pts[i].p;
            const PxVec3 hi(std::max(p.x, bx.lower.x), std::max(p.y, bx.lower.y), std::max(p.z, bx.lower.z));
            const PxVec3 cl(std::min(hi.x, bx.higher.x), std::min(hi.y, bx.higher.y), std::min(hi.z, bx.higher.z));
            const PxVec3 f = (cl - bx.lower) * bx.invDx;
            fs[i] = f;
            offs[i] = p - cl;
            const float want = sdfSampleRef(mN.s, vols, f) + (p - cl).magnitude();
            wants[i] = want;
            const float got = L.out[i].w;
            if (f2u(got) != f2u(want))
            {
                if (dBad < 5)
                    std::printf("    sdfQueryNonPow2 slot %u: distance %.9g (0x%08x) vs host PxSdfDistance %.9g (0x%08x)\n",
                                i, got, f2u(got), want, f2u(want));
                ++dBad;
            }
        }
        auto outOfDomain = [](int v) {
            return v == pxgtex::V_CC_NONE || v == pxgtex::V_CC_LO_X || v == pxgtex::V_CC_HI_X ||
                   v == pxgtex::V_CC_HI_N_Y || v == pxgtex::V_CC_LO_0_Z || v == pxgtex::V_Y_DIMX ||
                   v == pxgtex::V_IDX_HI_X || v == pxgtex::V_TIES_DOWN || v == pxgtex::V_RNE;
        };
        std::printf("    [ note ] sdfQueryNonPow2 bite asserts over %u host distances:\n     ", NP);
        for (int var = 1; var < pxgtex::V_COUNT; ++var)
        {
            size_t differ = 0;
            for (uint32_t i = 0; i < NP; ++i)
                if (f2u(sdfSampleRef(mN.s, vols, fs[i], var) + offs[i].magnitude()) != f2u(wants[i])) ++differ;
            std::printf(" %s %zu%s", pxgtex::variantName(var), differ, var % 6 == 0 ? "\n     " : ";");
            if (pxgtex::variantIsEquivalent(var))
                PXG_CHECK_MSG(differ == 0, "%s is declared equivalent but changes %zu distances",
                              pxgtex::variantName(var), differ);
            else if (!outOfDomain(var))
                PXG_CHECK_MSG(differ >= 1, "%s changes no distance of sdfQueryNonPow2", pxgtex::variantName(var));
        }
        std::printf("\n");
    }
    PXG_CHECK_MSG(dBad == 0, "sdfQueryNonPow2: %zu of %u distances differ from the host PxSdfDistance", dBad, NP);
    std::printf("    [ note ] non-power-of-two 10 x 7 x 11: %u slots, %zu anchored, %zu differ; %u distances bitwise vs "
                "the host PxSdfDistance, %zu differ\n", NP, anchored, bad, NP, dBad);
    checkUntouched("sdfQueryNonPow2", L);
}

} // namespace

