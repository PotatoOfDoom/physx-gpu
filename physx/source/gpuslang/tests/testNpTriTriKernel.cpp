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

// testNpTriTriKernel.cpp -- P9.83b-8 (4/n): triangleTriangleCollision itself
// (triTriCollisionBda.slang), whole-buffer bitwise.
//
// ---------------------------------------------------------------------------
// THE ORACLE IS A REPLAY OF THE KERNEL ON THE HOST
// ---------------------------------------------------------------------------
// Every piece of the kernel already has a host mirror certified against its own
// Slang body: the gather (pxgTriTriFindRef.h), the contact body
// (pxgTriTriBodyRef.h), the geometry (pxgTriTriGeomRef.h), the patch machinery
// and the tail (pxgTriTriRef.h).  What cannot be mirrored is the SDF texture: a
// trilinear sample on llvmpipe is the driver's arithmetic.  So the replay takes
// the two texture-dependent values from CERTIFIED DEVICE ENTRIES, fed with inputs
// the HOST computed:
//
//   npSdfCollision:sdfCollDistance   the cull's sdfDistance at a centroid
//   npSdfCollision:sdfCollTriangle   the descent (doTriangleSDFCollision)
//
// Both are the same functions the kernel calls (sdfDistance wraps
// sdfDistanceEx; sdfDoTriangleCollision wraps sdfDoTriangleCollisionEx), both
// stamped NoContraction like the kernel, so equal input bits give equal output
// bits.  The replay is a pure host function over two caches keyed by those input
// bits.  A key it has not seen is recorded as MISSING and answered with "no
// contact"; the fixture then measures every missing key in one batch per SDF
// mesh and replays again, until a replay completes with nothing missing.  The
// set closes because every missing key belongs to a round the replay has
// reached, and each pass reaches further.
//
// WHAT THE COMPARISON SEES OF THE GLUE.  The probes are fed the centroids, the
// sub-triangle vertices and the tolerances the HOST derived -- from doFirst, the
// operand exchange, aToB and its inverse, the per-pass cullScale and the scale
// frames.  What the DESCENT consumes reaches the output bits directly (positions,
// separations, normals), so a kernel that feeds it differently fails wherever a
// contact survives.  The GATHER is weaker: its sample feeds only `d < r + tol`, so
// a glue difference there shows only on a triangle whose decision it flips AND that
// then yields a candidate.  For an SDF whose field is 1-Lipschitz a triangle on the
// gather boundary cannot yield one: the descent's sep >= d - r = cullScale, which
// fails the post's `sep < cullScale`.  So an ulp-level respelling of bToA is
// invisible here by construction -- it is pinned by review against
// PxTransform::getInverse -- the tolerance itself is made visible by the BAND
// counters (ReplayStats), and the gather's comparison is certified on its boundary
// by the gather fixture.
//
// WHAT THE CASES REACH, each gated by a host-counted discriminator (the scenes
// cannot reach half of it -- see RUNTIME_STATUS P9.83b-8 (4/n)):
//   * both arms, and pass 1 in BOTH arms -- including, per arm, a case whose
//     contacts come from pass 1 ALONE (a flat, far SDF on the other side empties
//     pass 0), so skipping or corrupting pass 1 changes the output;
//   * doFirst: every term and sub-term decides a case with contacts -- the host
//     evaluates five mutated spellings and requires each to disagree somewhere;
//   * the one-SDF (floor) branch in both arms, with refinement live, down to the
//     level cap, on the ratio boundary, and past the 256-entry refinement cap with a
//     child in the last index slot (the resume quirk);
//   * a gather over more than one round (the resume handshake);
//   * scaled and scale-rotated meshes in BOTH arms and on the floor branch's SDF
//     side, rotated poses;
//   * the cull bands and the 128-patch cap;
//   * generateContacts false, and a pair with the flag set and nothing found;
//   * a patch, a contact and a force overflow, each limit with its own slack;
//   * nine pairs of distinct content, transforms and contact distances at indices
//     apart from the shapes' (with decoys at the other index), the real pair at
//     overlap word 1 byte 1 with flag values whose bit 0 is clear, varied prior
//     status bytes, and a fully valid pair one past numWorkItems that must stay
//     untouched -- so no out-of-bounds read decides anything.
//
// ttbInitAssignState IS CHECKED STRUCTURALLY, as the Block fixture asked
// (pxgSlangTriTriBlock's init-removed note): no output can tell a missing init
// from Workgroup storage that happens to hold 0, so the test reads the kernel
// source and requires the call, and the barrier after it, before the outer loop.

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "pxgSdfSceneRef.h"
#include "pxgTriTriBodyRef.h"
#include "pxgTriTriFindRef.h"
#include "pxgTriTriRef.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace physx { namespace gpuslang {
void pxgSetVulkanDriverShim(PxgCudaShim* shim);
} }

namespace
{
using namespace pxgtest;
using physx::PxVec3;
using physx::PxQuat;
using physx::PxTransform;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
namespace tg = pxgtest::tritrigeomref;
namespace tf = pxgtest::tritrifindref;
namespace tb = pxgtest::tritribodyref;
namespace tr = pxgtest::tritriref;
namespace sd = pxgsdfref;

// ---- the engine records, byte for byte --------------------------------------
struct PushH                       // triTriCollisionBda.slang TriTriCollisionParams
{
    uint32_t numWorkItems, pad4;   // @0
    uint64_t cmInputs;             // @8
    uint64_t transformCache;       // @16
    uint64_t contactDistances;     // @24
    uint64_t shapes;               // @32
    uint64_t cmOutputs;            // @40
    uint64_t overlaps;             // @48
    uint64_t materials;            // @56
    uint64_t contactStream;        // @64
    uint64_t patchStream;          // @72
    uint64_t counters;             // @80
    uint64_t touchChangeFlags;     // @88
    uint64_t patchChangeFlags;     // @96
    uint64_t startContactPatches;  // @104
    uint64_t startContactPoints;   // @112
    uint64_t startContactForces;   // @120
    uint32_t patchBytesLimit;      // @128
    uint32_t contactBytesLimit;    // @132
    uint32_t forceBytesLimit;      // @136
    float clusterBias;             // @140
};
static_assert(sizeof(PushH) == 144, "the 144-byte engine blob");
static_assert(offsetof(PushH, clusterBias) == 140, "clusterBias at @140");

struct CmInputH { uint32_t shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; };
struct ShapeH
{
    float scale[3];
    float scaleRot[4];
    uint32_t materialIndex;   // @28
    uint64_t hullOrMeshPtr;   // @32
    uint32_t type;            // @40
    uint32_t particleOrSoftbodyId;
};
static_assert(sizeof(ShapeH) == 48, "PxgShape");
struct XformH { float q[4]; float p[3]; uint32_t flags; };
static_assert(sizeof(XformH) == 32, "PxsCachedTransform");

// npSdfCollision.slang SdfCollParams.
struct P_Coll
{
    uint64_t meshPtr, points, gradIn, dirSeed, outBuf;
    uint32_t numPoints, cpad;
    uint64_t triVerts, triOut;
};
static const uint32_t DIST_STRIDE = 12u, TRI_STRIDE = 176u;

const uint32_t POISON = 0xCDCDCDCDu;
const uint32_t FLAG_POISON = 0xA5A5A5A5u;
const uint64_t PTR_POISON = 0xD00DFEEDBAADF00Dull;
const uint32_t SENTINEL = 0xFFFFFFFFu;
const uint32_t PATCH_WORDS = 128u * 16u;
const uint32_t CONTACT_WORDS = 768u * 4u;
const uint32_t DECOY_WORDS = 3072u;
const uint32_t MAX_REFINEMENT_LEVEL = 8u;
const float REFINEMENT_RATIO_SQUARED = 256.0f;

uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// ---- meshes ------------------------------------------------------------------
struct MeshDef
{
    const char* name;
    std::vector<tg::F4> verts;
    std::vector<tg::U4> inds;
    bool sdf = false;
    uint32_t projFlag = 0;
    sd::SdfScene scene;
    sd::HostSdfBox box{};
    sd::BlobLayout blob;
    PxgBufferHandle hBlob = 0;
    uint64_t addr = 0;
};

void addGrid(MeshDef& m, const PxVec3& o, const PxVec3& u, const PxVec3& v, uint32_t nu, uint32_t nv)
{
    const uint32_t base = uint32_t(m.verts.size());
    for (uint32_t j = 0; j <= nv; ++j)
        for (uint32_t i = 0; i <= nu; ++i)
        {
            const PxVec3 p = o + u * (float(i) / float(nu)) + v * (float(j) / float(nv));
            m.verts.push_back(tg::F4{ p.x, p.y, p.z, 0.0f });
        }
    for (uint32_t j = 0; j < nv; ++j)
        for (uint32_t i = 0; i < nu; ++i)
        {
            const uint32_t a = base + j * (nu + 1u) + i, b = a + 1u, c = a + (nu + 1u), d = c + 1u;
            m.inds.push_back(tg::U4{ a, b, d, 0u });
            m.inds.push_back(tg::U4{ a, d, c, 0u });
        }
}

void addQuad(MeshDef& m, const PxVec3& o, const PxVec3& u, const PxVec3& v, uint32_t n) { addGrid(m, o, u, v, n, n); }

// The same box with its 8 corners SHARED: 8 vertices, 12 triangles.  Against the
// 24-vertex box it tells doFirst's numVerts (hdr.x) from numTris (hdr.y).
void buildBox8(MeshDef& m, float h)
{
    for (uint32_t c = 0; c < 8u; ++c)
        m.verts.push_back(tg::F4{ (c & 1u) ? h : -h, (c & 2u) ? h : -h, (c & 4u) ? h : -h, 0.0f });
    const uint32_t f[12][3] = { { 0, 2, 3 }, { 0, 3, 1 }, { 4, 5, 7 }, { 4, 7, 6 }, { 0, 4, 6 }, { 0, 6, 2 },
                                { 1, 3, 7 }, { 1, 7, 5 }, { 0, 1, 5 }, { 0, 5, 4 }, { 2, 6, 7 }, { 2, 7, 3 } };
    for (const auto& t : f) m.inds.push_back(tg::U4{ t[0], t[1], t[2], 0u });
}

// A closed box of half-extent h, each face an n x n grid, outward winding.
void buildBox(MeshDef& m, float h, uint32_t n)
{
    const PxVec3 X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);
    addQuad(m, PxVec3(-h, -h, h), X * (2 * h), Y * (2 * h), n);     // +z
    addQuad(m, PxVec3(h, -h, -h), X * (-2 * h), Y * (2 * h), n);    // -z
    addQuad(m, PxVec3(h, -h, h), Z * (-2 * h), Y * (2 * h), n);     // +x
    addQuad(m, PxVec3(-h, -h, -h), Z * (2 * h), Y * (2 * h), n);    // -x
    addQuad(m, PxVec3(-h, h, h), X * (2 * h), Z * (-2 * h), n);     // +y
    addQuad(m, PxVec3(-h, -h, -h), X * (2 * h), Z * (2 * h), n);    // -y
}

// A UV sphere of radius r, outward winding: every triangle has its own normal, which
// is what fills the patch buffer when the floor branch uses triangle normals.
void buildSphere(MeshDef& m, float r, uint32_t nLat, uint32_t nLon)
{
    const float PI = 3.14159265358979f;
    for (uint32_t i = 0; i <= nLat; ++i)
        for (uint32_t j = 0; j < nLon; ++j)
        {
            const float th = PI * float(i) / float(nLat), ph = 2.0f * PI * float(j) / float(nLon);
            m.verts.push_back(tg::F4{ r * std::sin(th) * std::cos(ph), r * std::cos(th), r * std::sin(th) * std::sin(ph), 0.0f });
        }
    for (uint32_t i = 0; i < nLat; ++i)
        for (uint32_t j = 0; j < nLon; ++j)
        {
            const uint32_t a = i * nLon + j, b = i * nLon + (j + 1u) % nLon, c = a + nLon, d = b + nLon;
            if (i != 0u) m.inds.push_back(tg::U4{ a, b, c, 0u });
            if (i + 1u != nLat) m.inds.push_back(tg::U4{ b, d, c, 0u });
        }
}

// A dense SDF of the box [-h, h]^3: texel (i, j, k) holds the analytic distance at
// meshLower + (i + 0.5, j + 0.5, k + 0.5) * spacing, which is where the dense
// sample at f + 0.5 lands.  Its physical accuracy only decides WHICH contacts the
// fixture reaches -- the replay takes every sample from the device.
void buildBoxSdf(sd::SdfScene& s, float h)
{
    s.dx = 19; s.dy = 17; s.dz = 21;   // no two equal
    s.sg = 0;
    s.nbx = s.nby = s.nbz = 0;
    s.blkx = s.blky = s.blkz = 0;
    s.smin = -0.5f; s.smax = 0.5f;
    s.spacing = 0.1f;
    s.meshLower = PxVec3(-0.95f, -0.85f, -1.05f);
    s.coarse.resize(size_t(s.dx) * s.dy * s.dz);
    for (uint32_t k = 0; k < s.dz; ++k)
        for (uint32_t j = 0; j < s.dy; ++j)
            for (uint32_t i = 0; i < s.dx; ++i)
            {
                const PxVec3 p = s.meshLower + PxVec3((float(i) + 0.5f) * s.spacing, (float(j) + 0.5f) * s.spacing,
                                                      (float(k) + 0.5f) * s.spacing);
                const PxVec3 q(std::fabs(p.x) - h, std::fabs(p.y) - h, std::fabs(p.z) - h);
                const PxVec3 o(std::max(q.x, 0.0f), std::max(q.y, 0.0f), std::max(q.z, 0.0f));
                const float inside = std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
                s.coarse[(size_t(k) * s.dy + j) * s.dx + i] = o.magnitude() + inside;
            }
    s.starts.clear();
    s.sub.clear();
}

// The box's SDF grid with every sample set to one negative value.  Its gradient is
// exactly zero everywhere (PxVolumeGrad differences equal samples) and a point inside
// the support has diff = 0, so PxSdfSampleWithGrad returns
// dir = (0 * |dist| + 0).getNormalized() = 0 -- a ZERO contact normal.  That is the
// only way to fill the patch buffer: a patch consumes every contact within
// dot > 0.85 (31.8 degrees) of its normal, so patches with unit normals are pairwise
// more than 31.8 degrees apart and at most ~44 fit on the sphere (Tammes).  A zero
// normal projects to 0 on every patch, so each such contact seeds its own.
void buildFlatSdf(sd::SdfScene& s, float value)
{
    buildBoxSdf(s, 0.5f);
    std::fill(s.coarse.begin(), s.coarse.end(), value);
}

bool uploadMesh(PxgSlangComputeDevice* dev, MeshDef& m, std::vector<PxgBufferHandle>& owned)
{
    const uint32_t nbNodes = 2u;   // the BV32 tree is skipped by stride; its content is junk
    m.blob = sd::layOutBlobEx(m.sdf ? &m.scene : nullptr, uint32_t(m.verts.size()), uint32_t(m.inds.size()),
                              nbNodes, m.projFlag, 0x0BADF00Du, m.verts.data(), m.inds.data());
    if (m.sdf)
    {
        const uint64_t tex = sd::sdfUploadTex(CU_AD_FORMAT_FLOAT, m.scene.dx, m.scene.dy, m.scene.dz,
                                              m.scene.coarse.data(), sizeof(float));
        if (tex == 0) return false;
        std::memcpy(&m.blob.bytes[m.blob.texObjOffset], &tex, 8);
        m.box = sd::hostSdfBox(m.scene);
    }
    m.hBlob = createBufferFrom(dev, m.blob.bytes);
    owned.push_back(m.hBlob);
    m.addr = dev->getBufferDeviceAddress(m.hBlob);
    return true;
}

// ---- the device-measured values, cached by input bits -------------------------
struct DistKey
{
    uint32_t mesh, x, y, z;
    bool operator<(const DistKey& o) const
    { return std::tie(mesh, x, y, z) < std::tie(o.mesh, o.x, o.y, o.z); }
};
struct TriKey
{
    uint32_t mesh, v[9], tol;
    bool operator<(const TriKey& o) const
    {
        if (mesh != o.mesh) return mesh < o.mesh;
        for (int k = 0; k < 9; ++k) if (v[k] != o.v[k]) return v[k] < o.v[k];
        return tol < o.tol;
    }
};
struct TriVal { float sep; PxVec3 point, dir; };

struct Oracle
{
    std::map<DistKey, float> dist;
    std::map<TriKey, TriVal> tri;
    std::map<DistKey, int> missD;
    std::map<TriKey, int> missT;
    size_t dispatches = 0;
};

DistKey distKey(uint32_t mesh, const PxVec3& p) { return DistKey{ mesh, bitsOf(p.x), bitsOf(p.y), bitsOf(p.z) }; }
TriKey triKey(uint32_t mesh, const PxVec3& a, const PxVec3& b, const PxVec3& c, float tol)
{
    TriKey k;
    k.mesh = mesh;
    const PxVec3* v[3] = { &a, &b, &c };
    for (int i = 0; i < 3; ++i) { k.v[3 * i] = bitsOf(v[i]->x); k.v[3 * i + 1] = bitsOf(v[i]->y); k.v[3 * i + 2] = bitsOf(v[i]->z); }
    k.tol = bitsOf(tol);
    return k;
}

float floatOf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// Measures every missing key, one batch per SDF mesh and kind.
bool fetchMissing(PxgSlangComputeDevice* dev, const std::vector<MeshDef>& meshes, Oracle& o)
{
    for (uint32_t mi = 0; mi < meshes.size(); ++mi)
    {
        if (!meshes[mi].sdf) continue;
        std::vector<DistKey> dk;
        for (const auto& e : o.missD) if (e.first.mesh == mi) dk.push_back(e.first);
        if (!dk.empty())
        {
            std::vector<tg::F4> pts(dk.size());
            for (size_t i = 0; i < dk.size(); ++i)
                pts[i] = tg::F4{ floatOf(dk[i].x), floatOf(dk[i].y), floatOf(dk[i].z), 0.0f };
            std::vector<float> outInit(dk.size() * DIST_STRIDE, -4242.f);
            PxgBufferHandle bP = createBufferFrom(dev, pts), bO = createBufferFrom(dev, outInit);
            P_Coll p{};
            p.meshPtr = meshes[mi].addr;
            p.points = dev->getBufferDeviceAddress(bP);
            p.outBuf = dev->getBufferDeviceAddress(bO);
            p.numPoints = uint32_t(dk.size());
            {
                Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollDistance", {}, sizeof(P_Coll));
                if (!k.isValid()) return false;
                k.dispatch((p.numPoints + 63u) / 64u, {}, p);
            }
            dev->flush();
            const auto out = downloadBuffer<float>(dev, bO, outInit.size());
            for (size_t i = 0; i < dk.size(); ++i)
            {
                // A probe that returned before writing (sdfReadMesh failing, say) must not
                // be cached as a measured distance.
                if (out[i * DIST_STRIDE + 8] == -4242.f)
                {
                    std::printf("  fetchMissing: sdfCollDistance left point %zu of %s unwritten\n", i, meshes[mi].name);
                    return false;
                }
                o.dist[dk[i]] = out[i * DIST_STRIDE + 8];
            }
            dev->destroyBuffer(bP); dev->destroyBuffer(bO);
            ++o.dispatches;
        }
        std::vector<TriKey> tk;
        for (const auto& e : o.missT) if (e.first.mesh == mi) tk.push_back(e.first);
        if (!tk.empty())
        {
            std::vector<tg::F4> tv(tk.size() * 3u), seed(tk.size(), tg::F4{ 0, 0, 0, 0 });
            for (size_t i = 0; i < tk.size(); ++i)
                for (uint32_t c = 0; c < 3u; ++c)
                    tv[i * 3 + c] = tg::F4{ floatOf(tk[i].v[3 * c]), floatOf(tk[i].v[3 * c + 1]), floatOf(tk[i].v[3 * c + 2]),
                                            c == 0u ? floatOf(tk[i].tol) : 0.0f };
            std::vector<float> outInit(tk.size() * TRI_STRIDE, -4242.f);
            PxgBufferHandle bT = createBufferFrom(dev, tv), bS = createBufferFrom(dev, seed),
                            bO = createBufferFrom(dev, outInit);
            P_Coll p{};
            p.meshPtr = meshes[mi].addr;
            p.dirSeed = dev->getBufferDeviceAddress(bS);
            p.triVerts = dev->getBufferDeviceAddress(bT);
            p.triOut = dev->getBufferDeviceAddress(bO);
            p.numPoints = uint32_t(tk.size());
            {
                Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTriangle", {}, sizeof(P_Coll));
                if (!k.isValid()) return false;
                k.dispatch((p.numPoints + 63u) / 64u, {}, p);
            }
            dev->flush();
            const auto out = downloadBuffer<float>(dev, bO, outInit.size());
            for (size_t i = 0; i < tk.size(); ++i)
            {
                const float* e = &out[i * TRI_STRIDE];
                for (uint32_t w = 20; w <= 26; ++w)
                    if (e[w] == -4242.f)
                    {
                        std::printf("  fetchMissing: sdfCollTriangle left word %u of triangle %zu of %s unwritten\n",
                                    w, i, meshes[mi].name);
                        return false;
                    }
                o.tri[tk[i]] = TriVal{ e[26], PxVec3(e[20], e[21], e[22]), PxVec3(e[23], e[24], e[25]) };
            }
            dev->destroyBuffer(bT); dev->destroyBuffer(bS); dev->destroyBuffer(bO);
            ++o.dispatches;
        }
    }
    o.missD.clear();
    o.missT.clear();
    return true;
}

// ---- one pair, as the kernel sees it ----------------------------------------------
struct PairSpec
{
    const char* label;
    uint32_t m0, m1;                 // mesh indices of shapeRef0 / shapeRef1
    PxVec3 s0, s1;                   // mesh scales
    PxQuat r0, r1;                   // mesh scale rotations
    uint32_t mat0, mat1;
    PxTransform t0, t1;
    float cd0, cd1;                  // contactDistance per transform-cache ref
    bool overlap;
    // the reservation
    uint32_t cP, cC, cF, err;
    uint32_t tight;                  // 0 none; 1 patch, 2 contact: one byte short; 3 force: between N and 2N
};

// The BAND counters exist because the first sweep of this fixture showed that the
// cull tolerance, its per-pass reset, the scale it divides by and the refinement
// ratio were all invisible: no case had a triangle whose decision they change.
// Each counter is a host-side discriminator for one of those mutants -- it counts
// the triangles on which the CUDA's spelling and the mutant's spelling DISAGREE,
// and the test requires it to be nonzero.
struct ReplayStats
{
    uint32_t passes = 0, rounds = 0, fullRounds = 0, found = 0, candidates = 0, refineEntries = 0;
    uint32_t cullBandPass1 = 0;     // raw sep in [cs1, cs1 / min(s1)), normalised sep < cd: a carried cullScale
    uint32_t cullBandPass0 = 0;     // raw sep between cd / min(s0) and cs0, normalised sep < cd: the wrong scale
    uint32_t candBandPass0 = 0;     // normalised sep in [cd, cs0), raw sep < cs0: cullScale as the candidate bound
    uint32_t ratioBand = 0;         // triRadius2 * 256 > sdfRadius2 >= triRadius2 * 255
    uint32_t level7Requests = 0;    // refinement requested by a level-7 sub-triangle (dropped by a cap of 7)
    uint32_t maxPatches = 0;
    uint32_t candPass[2] = { 0, 0 };
    bool doFirstByNoSdfAlone = false;
    tf::FindStats find;
};

// |vertex2ShapeNormalVector(dir)|^-1 * sep, the normalisation both posts apply.
float normalisedSep(float sep, const PxVec3& dir, const PxVec3& scale, const PxQuat& rot)
{
    const PxVec3 d = tg::refShape2Vertex(dir, scale, rot);
    const float m = d.magnitudeSquared();
    return m > 0.0f ? sep * (1.0f / physx::PxSqrt(m)) : sep;
}

struct ReplayOut
{
    bool doFirst = false;
    uint32_t N = 0;
    tr::RefPatch patches[128];
    PxTransform localT1;
    uint32_t localMat0 = 0, localMat1 = 0;
    ReplayStats st;
};

// convexMeshMidphase.cu:1314-1377 and :972-1231 over the certified mirrors.
// Returns false when a device value was missing (they are then in o.missD / o.missT).
bool replay(const PairSpec& ps, const std::vector<MeshDef>& meshes, float clusterBias, Oracle& o, ReplayOut& R)
{
    std::memset(R.patches, 0, sizeof(R.patches));
    R.N = 0;
    R.st = ReplayStats{};
    const MeshDef& E0 = meshes[ps.m0];
    const MeshDef& E1 = meshes[ps.m1];
    const float contactDistance = ps.cd0 + ps.cd1;                              // :1356

    // :1358-1360
    const bool doFirst = E0.sdf && (!E1.sdf || (E0.projFlag == 0u && E1.projFlag != 0u) ||
                                    (E1.projFlag == E0.projFlag && E1.verts.size() < E0.verts.size()));
    R.doFirst = doFirst;
    R.st.doFirstByNoSdfAlone = E0.sdf && !E1.sdf && !(E0.projFlag == 0u && E1.projFlag != 0u) &&
                               !(E1.projFlag == E0.projFlag && E1.verts.size() < E0.verts.size());
    const uint32_t lm0 = doFirst ? ps.m1 : ps.m0, lm1 = doFirst ? ps.m0 : ps.m1;
    const MeshDef& mesh0 = meshes[lm0];
    const MeshDef& mesh1 = meshes[lm1];
    const PxVec3 s0 = doFirst ? ps.s1 : ps.s0, s1 = doFirst ? ps.s0 : ps.s1;
    const PxQuat r0 = doFirst ? ps.r1 : ps.r0, r1 = doFirst ? ps.r0 : ps.r1;
    const PxTransform t0 = doFirst ? ps.t1 : ps.t0, t1 = doFirst ? ps.t0 : ps.t1;
    R.localT1 = t1;
    R.localMat0 = doFirst ? ps.mat1 : ps.mat0;
    R.localMat1 = doFirst ? ps.mat0 : ps.mat1;

    const PxTransform aToB = t1.transformInv(t0);                               // :1022
    const PxTransform bToA = aToB.getInverse();                                 // :1078
    bool complete = true;

    for (uint32_t outer = 0; outer < 2u && ps.overlap; ++outer)
    {
        float cullScale = contactDistance;
        uint32_t nb, sdfMesh;
        tf::GatherGeom g;
        if (outer == 0u)
        {
            nb = uint32_t(mesh0.inds.size());
            cullScale /= physx::PxMin(s1.x, physx::PxMin(s1.y, s1.z));
            sdfMesh = lm1;
            g = tf::GatherGeom{ &mesh0.inds, &mesh0.verts, s0, r0, s1, r1, aToB, 0.0f };
        }
        else
        {
            if (!mesh0.sdf) break;
            nb = uint32_t(mesh1.inds.size());
            cullScale /= physx::PxMin(s0.x, physx::PxMin(s0.y, s0.z));
            sdfMesh = lm0;
            g = tf::GatherGeom{ &mesh1.inds, &mesh1.verts, s1, r1, s0, r0, bToA, 0.0f };
        }
        g.tolerance = cullScale;
        const sd::HostSdfBox& box = meshes[sdfMesh].box;
        ++R.st.passes;

        auto dist = [&](uint32_t, uint32_t, const PxVec3& p) -> float {
            const DistKey k = distKey(sdfMesh, p);
            const auto it = o.dist.find(k);
            if (it != o.dist.end()) return it->second;
            o.missD[k] = 1; complete = false;
            return 1.0e30f;                          // "not found" until measured
        };
        auto descent = [&](const PxVec3& a, const PxVec3& b, const PxVec3& c) -> TriVal {
            const TriKey k = triKey(sdfMesh, a, b, c, cullScale);
            const auto it = o.tri.find(k);
            if (it != o.tri.end()) return it->second;
            o.missT[k] = 1; complete = false;
            return TriVal{ 1.0e30f, PxVec3(0.f), PxVec3(0.f) };
        };

        std::vector<tf::U2> subBuf(tf::TT_SUBCAP, tf::U2{ 0u, 0u });
        std::vector<tf::U2> idxBuf(tf::TT_TARGET, tf::U2{ 0u, 0u });
        uint32_t reg = 0, i = 0;
        uint32_t guard = 0;
        while (i < nb || reg > 0)
        {
            if (++guard > 4096u) return false;     // a replay that does not terminate is a failed replay
            const uint32_t nbFound = tf::refFind(g, nb, i, reg, subBuf.data(), idxBuf.data(), dist, R.st.find);
            reg = 0;
            ++R.st.rounds;
            if (nbFound == tf::TT_TARGET) ++R.st.fullRounds;
            R.st.found += nbFound;

            std::vector<tr::RefContact> c(1024);
            std::vector<uint8_t> want(1024, 0);
            std::vector<uint32_t> indv(1024), subv(1024);
            for (uint32_t t = 0; t < 1024u; ++t)
            {
                c[t] = tr::RefContact{ { 0, 0, 0 }, tb::PX_MAX_F32_, { 0, 0, 0 }, false };
                indv[t] = t < nbFound ? idxBuf[t].x : nb;
                subv[t] = t < nbFound ? idxBuf[t].y : 0u;
                if (indv[t] >= nb) continue;
                PxVec3 v0, v1, v2;
                tb::RefContact bc;
                bool needsRefinement = false;
                if (outer == 0u)
                {
                    tb::refPass0Vertices(mesh0.inds, mesh0.verts, s0, r0, aToB, indv[t], subv[t], s1, r1, v0, v1, v2);
                    const TriVal d = descent(v0, v1, v2);
                    bc.pos = d.point;
                    bool usedDir2;
                    tb::refPass0Post(v0, v1, v2, d.sep, d.dir, box.lower, box.higher, s1, r1, mesh0.sdf,
                                     cullScale, contactDistance, REFINEMENT_RATIO_SQUARED, bc, needsRefinement, usedDir2);
                    const float sepN = normalisedSep(d.sep, d.dir, s1, r1);
                    const float csWrong = contactDistance / physx::PxMin(s0.x, physx::PxMin(s0.y, s0.z));
                    const float lo = std::min(cullScale, csWrong), hi = std::max(cullScale, csWrong);
                    if (d.sep >= lo && d.sep < hi && sepN < contactDistance) ++R.st.cullBandPass0;
                    if (d.sep < cullScale && sepN >= contactDistance && sepN < cullScale) ++R.st.candBandPass0;
                    if (!mesh0.sdf)
                    {
                        const float tr2 = tg::refTriangleRadiusSquared(v0, v1, v2);
                        const float sr2 = tg::refSdfRadiusSquared(box.lower, box.higher);
                        if (tr2 * 256.0f > sr2 && !(tr2 * 255.0f > sr2)) ++R.st.ratioBand;
                    }
                }
                else
                {
                    tb::refPass1Vertices(mesh1.inds, mesh1.verts, s1, r1, aToB, indv[t], s0, r0, v0, v1, v2);
                    const TriVal d = descent(v0, v1, v2);
                    bc.pos = d.point;
                    tb::refPass1Post(d.sep, d.dir, s0, r0, aToB, cullScale, contactDistance, bc);
                    const float sepN = normalisedSep(d.sep, d.dir, s0, r0);
                    const float csCarried = cullScale / physx::PxMin(s1.x, physx::PxMin(s1.y, s1.z));
                    const float lo = std::min(cullScale, csCarried), hi = std::max(cullScale, csCarried);
                    if (d.sep >= lo && d.sep < hi && sepN < contactDistance) ++R.st.cullBandPass1;
                }
                want[t] = needsRefinement ? 1u : 0u;
                if (needsRefinement)
                {
                    uint32_t level, idx;
                    physx::Gu::decodeSubdivisionId(subv[t], level, idx);
                    if (level == 7u) ++R.st.level7Requests;
                }
                c[t] = tr::RefContact{ { bc.pos.x, bc.pos.y, bc.pos.z }, bc.separation,
                                       { bc.normal.x, bc.normal.y, bc.normal.z }, bc.candidateContact };
            }
            const uint32_t newCount = tf::refAddToRefinement(want, indv, subv, MAX_REFINEMENT_LEVEL, subBuf.data(), R.st.find);
            R.st.refineEntries += newCount;
            int gen = 0;
            for (uint32_t t = 0; t < 1024u; ++t) gen += c[t].candidate ? 1 : 0;
            R.st.candidates += uint32_t(gen);
            R.st.candPass[outer] += uint32_t(gen);
            reg = newCount;
            tr::refAssignContacts(c.data(), R.patches, clusterBias, gen, R.N);
            R.st.maxPatches = std::max(R.st.maxPatches, R.N);
        }
    }
    return complete;
}

// ---- the kernel source, for the structural checks --------------------------------
std::string kernelSource()
{
    std::string here(__FILE__);
    const size_t cut = here.find_last_of('/');
    const std::string path = here.substr(0, cut) + "/../shaders/triTriCollisionBda.slang";
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

// ---------------------------------------------------------------------------
PXG_TEST(npTriTriKernel_structure)
{
    const std::string src = kernelSource();
    PXG_CHECK_MSG(!src.empty(), "npTriTriKernel_structure: cannot read triTriCollisionBda.slang");
    const size_t entry = src.find("void triangleTriangleCollision(");
    const size_t init = src.find("ttbInitAssignState(tid);", entry);
    const size_t barrier = src.find("GroupMemoryBarrierWithGroupSync();", init);
    const size_t loop = src.find("for (uint outer = 0u;", entry);
    PXG_CHECK_MSG(entry != std::string::npos && init != std::string::npos && barrier != std::string::npos &&
                  loop != std::string::npos && init < barrier && barrier < loop,
                  "npTriTriKernel_structure: ttbInitAssignState(tid) and the barrier after it must precede the "
                  "outer loop (cu:1024-1027) -- no output can tell a missing init from storage holding 0");
    // The fixture hooks must stay undefined in the engine kernel.
    PXG_CHECK_MSG(src.find("#define TTB_PATCH_GUARD") == std::string::npos &&
                  src.find("#define TTF_SDF_DISTANCE") == std::string::npos,
                  "npTriTriKernel_structure: the kernel defines a fixture hook");
}

PXG_TEST(npTriTriKernel_pairs)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    static const PxgKernelDef manifest[] = {
        { "triTriCollisionBda.spv", "triangleTriangleCollision", 0u, 0u } };
    static auto loader = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim* shim = new PxgCudaShim(dev, manifest, 1, +loader);
    physx::gpuslang::pxgSetVulkanDriverShim(shim);

    std::vector<PxgBufferHandle> owned;

    // ---- the meshes ----
    std::vector<MeshDef> meshes(15);
    enum { BOX = 0, BOXP, BOXFINE, FLOOR, GRID, BOX2, FLOORBIG, BOXFINE2, SPHERE, FLATBOX,
           FARBOX, FARBOXFINE, BOXFINEP, BOX8, GRIDREF };
    meshes[BOX].name = "box";           buildBox(meshes[BOX], 0.5f, 1);  meshes[BOX].sdf = true;
    buildBoxSdf(meshes[BOX].scene, 0.5f);
    meshes[BOXP].name = "box, projection flag"; buildBox(meshes[BOXP], 0.5f, 1); meshes[BOXP].sdf = true;
    meshes[BOXP].projFlag = 1u; buildBoxSdf(meshes[BOXP].scene, 0.5f);
    meshes[BOXFINE].name = "fine box";  buildBox(meshes[BOXFINE], 0.5f, 3); meshes[BOXFINE].sdf = true;
    buildBoxSdf(meshes[BOXFINE].scene, 0.5f);
    meshes[FLOOR].name = "floor";
    addQuad(meshes[FLOOR], PxVec3(-3.f, 0.f, 3.f), PxVec3(6.f, 0.f, 0.f), PxVec3(0.f, 0.f, -6.f), 1);
    meshes[GRID].name = "grid";
    addQuad(meshes[GRID], PxVec3(-0.62f, 0.f, 0.62f), PxVec3(1.24f, 0.f, 0.f), PxVec3(0.f, 0.f, -1.24f), 40);
    meshes[BOX2].name = "box, second copy"; buildBox(meshes[BOX2], 0.5f, 1); meshes[BOX2].sdf = true;
    buildBoxSdf(meshes[BOX2].scene, 0.5f);
    meshes[BOXFINE2].name = "fine box, second copy"; buildBox(meshes[BOXFINE2], 0.5f, 3); meshes[BOXFINE2].sdf = true;
    buildBoxSdf(meshes[BOXFINE2].scene, 0.5f);
    meshes[SPHERE].name = "sphere";      buildSphere(meshes[SPHERE], 0.3f, 16, 24);
    meshes[FLATBOX].name = "box, flat SDF"; buildBox(meshes[FLATBOX], 0.5f, 1); meshes[FLATBOX].sdf = true;
    buildFlatSdf(meshes[FLATBOX].scene, -0.25f);
    // Boxes whose SDF is flat and FAR (+10 everywhere): nothing is ever gathered against
    // them, so a pair with one of these on the local mesh1 side has an empty pass 0 and
    // every contact comes from pass 1.
    meshes[FARBOX].name = "box, far SDF"; buildBox(meshes[FARBOX], 0.5f, 1); meshes[FARBOX].sdf = true;
    buildFlatSdf(meshes[FARBOX].scene, 10.0f);
    meshes[FARBOXFINE].name = "fine box, far SDF"; buildBox(meshes[FARBOXFINE], 0.5f, 3); meshes[FARBOXFINE].sdf = true;
    buildFlatSdf(meshes[FARBOXFINE].scene, 10.0f);
    meshes[BOXFINEP].name = "fine box, projection flag"; buildBox(meshes[BOXFINEP], 0.5f, 3);
    meshes[BOXFINEP].sdf = true; meshes[BOXFINEP].projFlag = 1u; buildBoxSdf(meshes[BOXFINEP].scene, 0.5f);
    meshes[BOX8].name = "box, 8 shared corners"; buildBox8(meshes[BOX8], 0.5f); meshes[BOX8].sdf = true;
    buildBoxSdf(meshes[BOX8].scene, 0.5f);
    // 11 x 13 cells of 0.15, inside the flat SDF's support ([-0.9, 0.9] x [-1, 1] in x, z):
    // 286 triangles whose longest edge (0.212) is refined against the support's diagonal
    // (sqrt(9.8 / 256) = 0.196) while their children's (0.106) is not -- so one round
    // requests 286 > 256 refinements, and the next gathers 1024 children.
    meshes[GRIDREF].name = "grid for the refinement cap";
    addGrid(meshes[GRIDREF], PxVec3(-0.825f, 0.f, 0.975f), PxVec3(1.65f, 0.f, 0.f), PxVec3(0.f, 0.f, -1.95f), 11, 13);
    meshes[FLOORBIG].name = "big floor";
    addQuad(meshes[FLOORBIG], PxVec3(-20.f, 0.f, 20.f), PxVec3(40.f, 0.f, 0.f), PxVec3(0.f, 0.f, -40.f), 1);
    for (MeshDef& m : meshes)
        if (!uploadMesh(dev, m, owned))
        {
            PXG_CHECK_MSG(false, "npTriTriKernel_pairs: the SDF texture upload failed for %s", m.name);
            physx::gpuslang::pxgSetVulkanDriverShim(nullptr); delete shim; return;
        }

    // ---- materials ----
    const uint32_t nbMaterials = 8;
    std::vector<tr::MaterialH> mats(nbMaterials);
    for (uint32_t i = 0; i < nbMaterials; ++i)
    {
        mats[i].dynamicFriction = 0.2f + 0.1f * float(i);
        mats[i].staticFriction = 0.35f + 0.07f * float(i);
        mats[i].restitution = (i % 3 == 0) ? -0.5f - 0.1f * float(i) : 0.1f * float(i);
        mats[i].damping = 0.5f * float(i);
        const uint32_t flags = (i % 4 == 0) ? tr::FLAG_COMPLIANT_ACC_SPRING : 0u;
        mats[i].flagsModes0 = flags | ((i % 4) << 16) | (((i + 1) % 4) << 24);
        mats[i].modes1 = (i + 2) % 4;
    }
    const PxgBufferHandle bMat = createBufferFrom(dev, mats);
    owned.push_back(bMat);
    const uint64_t aMat = dev->getBufferDeviceAddress(bMat);

    const float clusterBias = 1e-5f * 1.0f;   // PxgNarrowphaseCore.cpp:1371, toleranceLength 1
    const PxQuat I(physx::PxIdentity);
    const PxVec3 ONE(1.f, 1.f, 1.f);
    const PxQuat tilt = PxQuat(0.23f, PxVec3(0.3f, 1.0f, 0.2f).getNormalized());
    const PxQuat tilt2 = PxQuat(-0.41f, PxVec3(1.0f, 0.2f, -0.5f).getNormalized());
    const PxQuat srot = PxQuat(0.6f, PxVec3(0.2f, 0.4f, 1.0f).getNormalized());

    std::vector<PairSpec> cases;
    auto add = [&](PairSpec p) { cases.push_back(p); };
    // Two SDF boxes, same mesh, one resting into the other: unflipped, passes 0 and 1.
    add(PairSpec{ "two SDF boxes: unflipped, both passes", BOX, BOX2, ONE, ONE, I, I, 1, 5,
                  PxTransform(PxVec3(0.05f, 0.93f, -0.02f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u * 7u, 16u * 11u, 4u * 13u, 0u, 0u });
    // The shelf pair: SDF box first, SDF-less floor second -> flipped, pass 0 only, refinement.
    add(PairSpec{ "box on floor: flipped, floor branch, refinement", BOX, FLOOR, ONE, ONE, I, I, 2, 6,
                  PxTransform(PxVec3(0.3f, 0.47f, -0.2f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.02f, true, 64u * 3u, 16u * 5u, 4u * 9u, 0u, 0u });
    // The same pair in the other slot order -> unflipped arm with the floor branch.
    add(PairSpec{ "floor under box: unflipped floor branch", FLOOR, BOX, ONE, ONE, I, I, 6, 2,
                  PxTransform(PxVec3(0.f, 0.f, 0.f), I), PxTransform(PxVec3(-0.2f, 0.46f, 0.1f), tilt2),
                  0.02f, 0.02f, true, 0u, 0u, 0u, 8u, 0u });
    // doFirst's numVerts term: fine box (96 verts) first, plain box (24) second -> flipped, two SDFs,
    // so pass 1 runs IN THE FLIPPED ARM, which no scene reaches.
    add(PairSpec{ "fine box vs box: flipped by numVerts, both passes", BOXFINE, BOX, ONE, ONE, I, I, 3, 4,
                  PxTransform(PxVec3(0.02f, 0.94f, 0.03f), tilt2), PxTransform(PxVec3(0.f, 0.f, 0.f), tilt),
                  0.03f, 0.02f, true, 64u * 11u, 16u * 3u, 4u * 5u, 2u, 0u });
    add(PairSpec{ "box vs fine box: numVerts keeps the order", BOX, BOXFINE, ONE, ONE, I, I, 4, 3,
                  PxTransform(PxVec3(0.f, 0.f, 0.f), tilt), PxTransform(PxVec3(-0.03f, 0.95f, 0.01f), tilt2),
                  0.02f, 0.03f, true, 64u, 16u, 4u, 0u, 0u });
    // doFirst's projection-flag term.
    add(PairSpec{ "box vs box with projection flag: flipped by the flag", BOX, BOXP, ONE, ONE, I, I, 0, 7,
                  PxTransform(PxVec3(0.04f, 0.92f, 0.f), tilt2), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.02f, true, 0u, 0u, 0u, 0u, 0u });
    // Scaled and rotated meshes: cullScale divides by the smallest scale, and the frames matter.
    add(PairSpec{ "scaled boxes", BOX, BOX2, PxVec3(1.3f, 0.8f, 1.1f), PxVec3(0.9f, 1.2f, 1.05f), srot, tilt, 1, 2,
                  PxTransform(PxVec3(0.07f, 0.97f, -0.05f), tilt2), PxTransform(PxVec3(0.f, 0.f, 0.f), tilt),
                  0.02f, 0.02f, true, 64u * 5u, 16u * 9u, 4u * 2u, 0u, 0u });
    // More than one gather round: 3200 small grid triangles under a box.
    add(PairSpec{ "grid under box: several gather rounds", GRID, BOX, ONE, ONE, I, I, 5, 1,
                  PxTransform(PxVec3(0.f, 0.f, 0.f), I), PxTransform(PxVec3(0.f, 0.485f, 0.f), I),
                  0.03f, 0.03f, true, 64u * 2u, 16u * 2u, 4u * 2u, 0u, 0u });
    // The overlap flag clear: the loop never runs, the tail still writes.
    add(PairSpec{ "overlap flag clear", BOX, BOX2, ONE, ONE, I, I, 1, 5,
                  PxTransform(PxVec3(0.05f, 0.93f, -0.02f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, false, 64u, 16u, 4u, 0u, 0u });
    // Flag set, boxes apart: every pass runs and finds nothing.
    add(PairSpec{ "apart: nothing found", BOX, BOX2, ONE, ONE, I, I, 1, 5,
                  PxTransform(PxVec3(0.f, 3.f, 0.f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u, 16u, 4u, 0u, 0u });
    // An overflowing reservation.
    add(PairSpec{ "two SDF boxes, overflowing limits", BOX, BOX2, ONE, ONE, I, I, 1, 5,
                  PxTransform(PxVec3(0.05f, 0.93f, -0.02f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u * 7u, 16u * 11u, 4u * 13u, 1u, 1u });

    // ---- the cases the first sweep asked for -------------------------------------------
    // doFirst's FIRST term alone: an SDF-less mesh with MORE vertices in the second slot.
    // The floor case above is flipped by the numVerts term too (4 < 24), which is why
    // dropping the no-SDF term survived.
    add(PairSpec{ "box on grid, box first: doFirst by the missing SDF alone", BOX, GRID, ONE, ONE, I, I, 7, 3,
                  PxTransform(PxVec3(0.02f, 0.488f, -0.01f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.03f, 0.03f, true, 64u * 4u, 16u * 4u, 4u * 4u, 0u, 0u });
    // THE CULL BANDS.  A mutated cull tolerance is visible only when it SHRINKS: a
    // triangle whose raw separation is at or above cd / minScale has a normalised
    // separation of at least cd (|shape2Vertex(n)| <= 1 / minScale), so it can never
    // be a candidate, and a tolerance that only grows adds nothing.  So each case below
    // puts a scale ABOVE 1 on the side the mutant would wrongly divide by, and hovers a
    // fine box over another, tilted 0.05 rad, so the per-triangle gaps spread over the
    // band [0.8 cd, cd) that the mutant cuts off (cd = 0.1).
    const PxQuat hover(0.05f, PxVec3(1.0f, 0.0f, 0.0f));
    // Pass 0 must divide by the SDF side's (mesh1's) scale: mesh0 is scaled 1.25.
    add(PairSpec{ "hover, mesh0 scaled 1.25: pass 0 divides by mesh1's scale", BOXFINE, BOXFINE2,
                  PxVec3(1.25f, 1.25f, 1.25f), ONE, I, I, 2, 4,
                  PxTransform(PxVec3(0.f, 1.215f, 0.f), hover), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.05f, 0.05f, true, 64u * 6u, 16u * 6u, 4u * 6u, 0u, 0u });
    // Pass 1 must start again from contactDistance: mesh1 is scaled 1.25, so a cullScale
    // carried over from pass 0 would be divided by it a second time.
    add(PairSpec{ "hover, mesh1 scaled 1.25: pass 1 resets cullScale", BOXFINE, BOXFINE2,
                  ONE, PxVec3(1.25f, 1.25f, 1.25f), I, I, 3, 5,
                  PxTransform(PxVec3(0.f, 1.215f, 0.f), hover), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.05f, 0.05f, true, 0u, 0u, 0u, 0u, 0u });
    // The candidate bound is contactDistance, not cullScale: mesh1 scaled (0.8, 1, 0.8), so
    // cullScale = cd / 0.8 while the normals run along y, where the scale is 1.
    add(PairSpec{ "hover, mesh1 scaled (0.8, 1, 0.8): candidates bound by cd", BOXFINE, BOXFINE2,
                  ONE, PxVec3(0.8f, 1.0f, 0.8f), I, I, 1, 6,
                  PxTransform(PxVec3(0.f, 1.11f, 0.f), hover), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.05f, 0.05f, true, 64u, 16u, 4u, 0u, 0u });
    // A sphere inside the box, SDF-less: every triangle is a candidate and the SDF
    // gradients spread the normals, so the patch count is set by the 0.85 consumption
    // cone, not by the triangle count (29 patches of 720 contacts).
    add(PairSpec{ "sphere in box: many patches", SPHERE, BOX, ONE, ONE, I, I, 0, 7,
                  PxTransform(PxVec3(0.02f, -0.01f, 0.03f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.03f, 0.03f, true, 64u * 9u, 16u * 9u, 4u * 9u, 0u, 0u });
    // THE 128-PATCH CAP.  Unit normals cannot reach it (see buildFlatSdf); the flat SDF
    // gives all 720 contacts a zero normal, so each seeds its own patch until the cap.
    add(PairSpec{ "sphere in a flat SDF: the 128-patch cap", SPHERE, FLATBOX, ONE, ONE, I, I, 5, 0,
                  PxTransform(PxVec3(0.02f, -0.01f, 0.03f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.03f, 0.03f, true, 64u * 3u, 16u * 3u, 4u * 3u, 0u, 0u });
    // Refinement down to the level cap: a 40 x 40 floor under the box.
    add(PairSpec{ "big floor: refinement to the level cap", BOX, FLOORBIG, ONE, ONE, I, I, 2, 6,
                  PxTransform(PxVec3(0.1f, 0.47f, -0.15f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.02f, true, 64u, 16u, 4u, 0u, 0u });
    // The refinement RATIO on its boundary: the floor is scaled so that its level-4
    // sub-triangles have triRadius2 * 256 just above the SDF's radius -- refined with
    // 256, not with 255.  All level-4 children of the two congruent floor triangles are
    // congruent, so the whole level sits in the band.  k is solved on the host from the
    // level-0 radius in the frame the kernel measures it in.
    {
        const PxTransform tb0(PxVec3(0.04f, 0.47f, 0.03f), tilt);
        const PxTransform tf1(PxVec3(0.f, 0.f, 0.f), I);
        const PxTransform aToB = tb0.transformInv(tf1);   // local t1 = box, local t0 = floor (flipped)
        PxVec3 v0, v1, v2;
        tg::refGetTriangleVertices(meshes[FLOOR].inds, meshes[FLOOR].verts, ONE, I, aToB, 0u, 0u, v0, v1, v2);
        const float r0 = tg::refTriangleRadiusSquared(v0, v1, v2);
        const sd::HostSdfBox hb = sd::hostSdfBox(meshes[BOX].scene);
        const float sr2 = tg::refSdfRadiusSquared(hb.lower, hb.higher);
        const float k = std::sqrt(sr2 * 1.002f * 256.0f / (256.0f * r0));   // level 4: 4^4 / 256 = 1
        add(PairSpec{ "floor sized onto the refinement ratio", BOX, FLOOR, ONE, PxVec3(k, k, k), I, I, 3, 5,
                      tb0, tf1, 0.02f, 0.02f, true, 64u, 16u, 4u, 0u, 0u });
    }
    // ---- the cases the review asked for ----------------------------------------------
    // PASS 1 ALONE, both arms.  The far SDF sits on the local mesh1 side, so pass 0
    // gathers nothing and every contact is pass 1's.
    add(PairSpec{ "pass 1 alone: far SDF in slot 1", BOX, FARBOX, ONE, ONE, I, I, 4, 1,
                  PxTransform(PxVec3(0.05f, 0.93f, -0.02f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u * 2u, 16u * 3u, 4u * 5u, 0u, 0u });
    add(PairSpec{ "pass 1 alone, flipped: far fine box first", FARBOXFINE, BOX, ONE, ONE, I, I, 6, 3,
                  PxTransform(PxVec3(0.02f, 0.94f, 0.03f), tilt2), PxTransform(PxVec3(0.f, 0.f, 0.f), tilt),
                  0.03f, 0.02f, true, 64u * 3u, 16u * 2u, 4u * 7u, 0u, 0u });
    // The operand exchange of the scale frames: flipped (numVerts), both sides scaled
    // non-uniformly with their own scale rotations.
    add(PairSpec{ "flipped, both sides scaled and scale-rotated", BOXFINE, BOX,
                  PxVec3(1.2f, 0.9f, 1.1f), PxVec3(0.8f, 1.3f, 1.0f), srot, tilt, 2, 7,
                  PxTransform(PxVec3(0.03f, 1.1f, -0.02f), tilt2), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u * 4u, 16u * 4u, 4u * 4u, 0u, 0u });
    // The floor branch against a scaled, scale-rotated SDF side.
    add(PairSpec{ "floor under a scaled, scale-rotated box", FLOOR, BOX, ONE, PxVec3(0.9f, 1.2f, 1.1f), I, srot, 5, 4,
                  PxTransform(PxVec3(0.f, 0.f, 0.f), I), PxTransform(PxVec3(-0.1f, 0.58f, 0.05f), tilt2),
                  0.02f, 0.02f, true, 64u, 16u, 4u, 0u, 0u });
    // doFirst's sub-terms, one case each.
    add(PairSpec{ "both flagged: !mesh0 flag decides (no flip)", BOXP, BOXFINEP, ONE, ONE, I, I, 1, 2,
                  PxTransform(PxVec3(0.f, 0.f, 0.f), tilt), PxTransform(PxVec3(-0.03f, 0.95f, 0.01f), tilt2),
                  0.02f, 0.03f, true, 64u, 16u, 4u, 0u, 0u });
    add(PairSpec{ "flags differ, fewer verts second: flag equality decides", BOXFINEP, BOX, ONE, ONE, I, I, 3, 0,
                  PxTransform(PxVec3(0.02f, 0.94f, 0.03f), tilt2), PxTransform(PxVec3(0.f, 0.f, 0.f), tilt),
                  0.03f, 0.02f, true, 64u, 16u, 4u, 0u, 0u });
    add(PairSpec{ "8-corner box second: numVerts, not numTris", BOX, BOX8, ONE, ONE, I, I, 7, 6,
                  PxTransform(PxVec3(0.05f, 0.93f, -0.02f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u, 16u, 4u, 0u, 0u });
    // The refinement buffer's 256 cap, and a child in index slot 1023.
    add(PairSpec{ "grid in a flat SDF: refinement past 256, a child in the last slot", GRIDREF, FLATBOX, ONE, ONE, I, I, 2, 5,
                  PxTransform(PxVec3(0.f, 0.1f, 0.f), I), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.03f, 0.03f, true, 64u * 2u, 16u * 2u, 4u * 2u, 0u, 0u });
    // The contact and the force overflow, each limit with its own slack.
    add(PairSpec{ "two SDF boxes, contact limit one byte short", BOX, BOX2, ONE, ONE, I, I, 1, 5,
                  PxTransform(PxVec3(0.05f, 0.93f, -0.02f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u * 7u, 16u * 11u, 4u * 13u, 0u, 2u });
    add(PairSpec{ "two SDF boxes, force limit between N and 2N", BOX, BOX2, ONE, ONE, I, I, 1, 5,
                  PxTransform(PxVec3(0.05f, 0.93f, -0.02f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  0.02f, 0.03f, true, 64u * 7u, 16u * 11u, 4u * 13u, 4u, 3u });
    // FUNCTION-LEVEL: an infinite contact distance -- every triangle passes the cull and
    // the descent's tolerance is infinite too.  (It does NOT reach the unwritten `dir` of
    // Body T1: with an infinite tolerance the descent always writes it; only an infinite
    // SDF SAMPLE would leave it unwritten.)
    add(PairSpec{ "infinite contact distance (function-level)", BOXFINE, BOX, ONE, ONE, I, I, 1, 2,
                  PxTransform(PxVec3(0.f, 0.9f, 0.f), tilt), PxTransform(PxVec3(0.f, 0.f, 0.f), I),
                  std::numeric_limits<float>::infinity(), 0.02f, true, 64u, 16u, 4u, 0u, 0u });

    // ---- run each case ------------------------------------------------------------------
    // NP pairs are launched with numWorkItems = NP, in NP + 1 workgroups.  The real pair
    // sits at pairIdx 5 -- overlap WORD 1, byte 1 -- so a kernel that drops the word
    // index, or tests bit 0 of a flag whose value is 0x02 or 0x80, reads a clear flag.
    // Every other pair is a DECOY of its own content (a perturbed pose, scale, material
    // and contactDistance), so a kernel that reads another pair's cmInputs entry
    // computes something else.  Decoys below numWorkItems have a clear flag and reserve
    // nothing, so the counters stay deterministic; the pair AT numWorkItems is valid,
    // flagged and fully backed by data, and must stay untouched -- no out-of-bounds
    // read decides whether the early return is `>=`.
    //
    // Transforms and contact distances live at transformCacheRef = shapeRef + TOFS, and
    // each array holds a decoy at the OTHER ref, so a kernel that indexes one array by
    // the other ref reads a different pose, distance or shape.
    const uint32_t NP = 8u, REAL = 5u, NALL = NP + 1u, TOFS = 2u * NALL;
    const uint32_t realStatus[4] = { 0x02u, 0x41u, 0x00u, 0x83u };   // touch; no touch + bit 6; unknown; both
    Oracle oracle;
    size_t mismatches = 0, flippedCases = 0, pass1Cases = 0, multiRound = 0, refineCases = 0, multiPatch = 0;
    size_t overflowCases = 0, withContacts = 0, contactOvfCases = 0, forceOvfCases = 0;
    uint32_t cullBand1 = 0, cullBand0 = 0, candBand0 = 0, ratioBand = 0, level7 = 0, capDrops = 0;
    size_t fullPatchCases = 0, noSdfAloneCases = 0, refineOverflow = 0, childLastSlot = 0;
    size_t pass1AloneUnflipped = 0, pass1AloneFlipped = 0, flippedScaled = 0, scaledFloor = 0;
    size_t touchSeedCases = 0, noTouchSeedCases = 0;
    size_t doFirstKill[5] = { 0, 0, 0, 0, 0 };
    const char* doFirstMutant[5] = { "the SDF guard dropped", "the no-SDF term dropped", "`!mesh0 flag` dropped",
                                     "the flag equality dropped", "numTris for numVerts" };
    for (size_t ci = 0; ci < cases.size(); ++ci)
    {
        const PairSpec& ps = cases[ci];
        ReplayOut R;
        bool done = false;
        for (uint32_t it = 0; it < 64u && !done; ++it)
        {
            done = replay(ps, meshes, clusterBias, oracle, R);
            if (!done && !fetchMissing(dev, meshes, oracle)) break;
        }
        PXG_CHECK_MSG(done, "%s: the replay did not close", ps.label);
        if (!done) continue;

        // ---- inputs: the real pair, the decoys, the pair past numWorkItems ----
        std::vector<CmInputH> cm(NALL);
        std::vector<ShapeH> shapes(2u * TOFS);
        std::vector<XformH> xf(2u * TOFS);
        std::vector<float> cd(2u * TOFS);
        std::vector<uint32_t> ov((NALL + 3u) / 4u, 0u);
        auto putShape = [&](uint32_t idx, uint32_t mesh, const PxVec3& sc, const PxQuat& sr, uint32_t mat) {
            ShapeH& sh = shapes[idx];
            sh.scale[0] = sc.x; sh.scale[1] = sc.y; sh.scale[2] = sc.z;
            sh.scaleRot[0] = sr.x; sh.scaleRot[1] = sr.y; sh.scaleRot[2] = sr.z; sh.scaleRot[3] = sr.w;
            sh.materialIndex = mat;
            sh.hullOrMeshPtr = meshes[mesh].addr;
            sh.type = 5u; sh.particleOrSoftbodyId = 0xFFFFFFFFu;
        };
        auto putXf = [&](uint32_t idx, const PxTransform& tt) {
            XformH& x = xf[idx];
            x.q[0] = tt.q.x; x.q[1] = tt.q.y; x.q[2] = tt.q.z; x.q[3] = tt.q.w;
            x.p[0] = tt.p.x; x.p[1] = tt.p.y; x.p[2] = tt.p.z; x.flags = 0u;
        };
        for (uint32_t k = 0; k < NALL; ++k)
        {
            PairSpec d = ps;                                   // pair k's content
            if (k != REAL && k != NP)
            {
                const float f = float(k + 1u);
                d.t0.p += PxVec3(0.011f * f, -0.004f * f, -0.007f * f);
                d.s0 *= 1.0f + 0.02f * f;
                d.mat0 = (ps.mat0 + k + 1u) % nbMaterials;
                d.mat1 = (ps.mat1 + 2u * k + 1u) % nbMaterials;
                d.cd0 = ps.cd0 * (1.0f + 0.1f * f);
            }
            const uint32_t sref[2] = { 2u * k + 1u, 2u * k };          // deliberately not 0/1
            cm[k] = CmInputH{ sref[0], sref[1], sref[0] + TOFS, sref[1] + TOFS };
            const uint32_t mesh[2] = { d.m0, d.m1 };
            const PxVec3 sc[2] = { d.s0, d.s1 };
            const PxQuat sr[2] = { d.r0, d.r1 };
            const uint32_t mat[2] = { d.mat0, d.mat1 };
            const PxTransform tt[2] = { d.t0, d.t1 };
            const float cdd[2] = { d.cd0, d.cd1 };
            for (uint32_t s = 0; s < 2u; ++s)
            {
                putShape(sref[s], mesh[s], sc[s], sr[s], mat[s]);
                putXf(sref[s] + TOFS, tt[s]);
                cd[sref[s] + TOFS] = cdd[s];
                // the decoys at the other ref
                putShape(sref[s] + TOFS, mesh[s], sc[s] * 1.03f, sr[s], (mat[s] + 3u) % nbMaterials);
                putXf(sref[s], PxTransform(tt[s].p + PxVec3(0.f, 0.017f, 0.f), tt[s].q));
                cd[sref[s]] = cdd[s] * 0.7f;
            }
        }
        // The CUDA converts the PxU8 to bool (`!= 0`): 0x02 and 0x80 are true with bit 0 clear.
        const uint32_t realByte = (ci % 2u == 0u) ? 0x02u : 0x80u;
        if (ps.overlap) ov[REAL >> 2] |= realByte << ((REAL & 3u) * 8u);
        ov[NP >> 2] |= 0x01u << ((NP & 3u) * 8u);

        // ---- records, flags, counters, streams, decoys ----
        std::vector<uint32_t> records(NALL * 12u, POISON);
        const uint32_t seedStatus = realStatus[ci % 4u];
        for (uint32_t k = 0; k < NALL; ++k)
        {
            uint32_t* rec = &records[k * 12u];
            for (uint32_t w = 0; w < 8; w += 2) { rec[w] = uint32_t(PTR_POISON); rec[w + 1] = uint32_t(PTR_POISON >> 32); }
            // byte 0 nonzero (the CUDA writes it 0), byte 1 = the previous patch count
            // (equal to the new one in every third case), byte 2 the status, byte 3 junk.
            const uint32_t status = k == REAL ? seedStatus : ((2u ^ k) & 0xFFu);
            const uint32_t prev = (k == REAL && ci % 3u == 0u) ? (R.N & 0xFFu) : ((5u ^ k) & 0xFFu);
            rec[8] = (0x03u << 24) | (status << 16) | (prev << 8) | (0x5Au ^ k);
            rec[9] = 0xBEEF0000u | (0x77u + k);                        // nbContacts garbage, flags half kept
            rec[10] = 0x11111111u * (k + 1u); rec[11] = 0x22222222u * (k + 1u);
        }
        std::vector<uint32_t> touch(NALL, FLAG_POISON), patchF(NALL, FLAG_POISON);
        std::vector<uint32_t> counters{ ps.cP, ps.cC, ps.cF, ps.err };
        std::vector<uint32_t> pstream(PATCH_WORDS, POISON), cstream(CONTACT_WORDS, POISON);
        std::vector<uint32_t> decoy(3u * DECOY_WORDS, POISON);

        std::vector<PxgBufferHandle> local;
        auto mk = [&](auto& v) { PxgBufferHandle h = createBufferFrom(dev, v); local.push_back(h); return h; };
        const PxgBufferHandle bCm = mk(cm), bSh = mk(shapes), bXf = mk(xf), bCd = mk(cd), bOv = mk(ov);
        const PxgBufferHandle bRec = mk(records), bTouch = mk(touch), bPF = mk(patchF), bCnt = mk(counters);
        const PxgBufferHandle bPS = mk(pstream), bCS = mk(cstream), bDec = mk(decoy);
        const uint64_t aDec = dev->getBufferDeviceAddress(bDec);

        PushH push{};
        push.numWorkItems = NP;
        push.cmInputs = dev->getBufferDeviceAddress(bCm);
        push.transformCache = dev->getBufferDeviceAddress(bXf);
        push.contactDistances = dev->getBufferDeviceAddress(bCd);
        push.shapes = dev->getBufferDeviceAddress(bSh);
        push.cmOutputs = dev->getBufferDeviceAddress(bRec);
        push.overlaps = dev->getBufferDeviceAddress(bOv);
        push.materials = aMat;
        push.contactStream = dev->getBufferDeviceAddress(bCS) - uint64_t(ps.cC);
        push.patchStream = dev->getBufferDeviceAddress(bPS) - uint64_t(ps.cP);
        push.counters = dev->getBufferDeviceAddress(bCnt);
        push.touchChangeFlags = dev->getBufferDeviceAddress(bTouch);
        push.patchChangeFlags = dev->getBufferDeviceAddress(bPF);
        push.startContactPatches = aDec - uint64_t(ps.cP);
        push.startContactPoints = aDec + uint64_t(DECOY_WORDS) * 4u - uint64_t(ps.cC);
        push.startContactForces = aDec + uint64_t(DECOY_WORDS) * 8u - uint64_t(ps.cF);

        // ---- the oracle: the tail over the replay's patches ----
        uint32_t starts[128], total = 0;
        for (uint32_t pi = 0; pi < 128; ++pi) { starts[pi] = total; if (pi < R.N) total += R.patches[pi].pointCount; }
        const uint64_t needP = uint64_t(ps.cP) + 64ull * R.N, needC = uint64_t(ps.cC) + 16ull * total,
                       needF = uint64_t(ps.cF) + 8ull * total;
        // Each limit has its OWN slack, so a kernel that hands one limit to another's
        // check trips (or misses) a different overflow.  The force limit of mode 3 sits
        // between the N bytes the reservation adds and the 2N the check tests (:573).
        push.patchBytesLimit = ps.tight == 1u ? uint32_t(needP > 0 ? needP - 1u : 0u) : uint32_t(needP + 4096u);
        push.contactBytesLimit = ps.tight == 2u ? uint32_t(needC > 0 ? needC - 1u : 0u) : uint32_t(needC + 8192u);
        push.forceBytesLimit = ps.tight == 3u ? uint32_t(uint64_t(ps.cF) + 6ull * total) : uint32_t(needF + 12288u);
        push.clusterBias = clusterBias;

        std::vector<uint32_t> wRec = records, wTouch = touch, wPF = patchF, wCnt = counters, wPS = pstream, wCS = cstream;
        tr::RefRes realRes{};
        for (uint32_t k = 0; k < NP; ++k)                          // the pair at NP stays as seeded
        {
            const bool real = k == REAL;
            const uint32_t n = real ? R.N : 0u;
            const uint32_t tot = real ? total : 0u;
            uint32_t* rec = &wRec[k * 12u];
            if (tot != 0)
            {
                wCnt[0] = ps.cP + 64u * n; wCnt[1] = ps.cC + 16u * tot; wCnt[2] = ps.cF + 4u * tot;
            }
            const tr::RefRes r = tr::refReserve(rec[8], n, tot, ps.cP, ps.cC, ps.cF,
                                                push.patchBytesLimit, push.contactBytesLimit, push.forceBytesLimit);
            if (real) { wCnt[3] = ps.err | r.ovfBits; realRes = r; }
            wTouch[k] = r.touchChange ? 1u : 0u;
            wPF[k] = r.patchChange ? 1u : 0u;
            rec[8] = r.allFlags;
            rec[9] = (rec[9] & 0xFFFF0000u) | (r.nContacts & 0xFFFFu);
            uint64_t pP = 0, pC = 0, pF = 0;
            if (!r.overflow)
            {
                pF = push.startContactForces + uint64_t(r.forceOfs);
                pP = push.startContactPatches + uint64_t(r.patchOfs);
                pC = push.startContactPoints + uint64_t(r.contactOfs);
            }
            rec[0] = uint32_t(pP); rec[1] = uint32_t(pP >> 32);
            rec[2] = uint32_t(pC); rec[3] = uint32_t(pC >> 32);
            rec[4] = uint32_t(pF); rec[5] = uint32_t(pF >> 32);
            if (!real) continue;
            if (r.overflow) ++overflowCases;
            const tr::V4 q = { R.localT1.q.x, R.localT1.q.y, R.localT1.q.z, R.localT1.q.w };
            const tr::V3 tp = { R.localT1.p.x, R.localT1.p.y, R.localT1.p.z };
            for (uint32_t pi = 0; pi < R.N; ++pi)
            {
                const tr::RefPatch& p = R.patches[pi];
                if (r.patchOfs != SENTINEL)
                {
                    uint32_t word[14];
                    tr::refPatchHeader(mats.data(), R.localMat0, R.localMat1, R.doFirst, q, p.normal,
                                       p.pointCount, starts[pi], word);
                    tr::refStorePatchHeader(&wPS[pi * 16u], word);
                }
                if (r.contactOfs != SENTINEL)
                {
                    const uint32_t npts = std::min(p.pointCount, 6u);
                    for (uint32_t kk = 0; kk < npts; ++kk)
                    {
                        const tr::V4 w = tr::refPatchPoint(q, tp, p.keepPoints[kk]);
                        float* d = reinterpret_cast<float*>(&wCS[size_t(starts[pi] + kk) * 4u]);
                        d[0] = w.x; d[1] = w.y; d[2] = w.z; d[3] = w.w;
                    }
                }
            }
        }

        // ---- the kernel: one workgroup MORE than numWorkItems ----
        {
            Kernel k(Kernel::Bindless{}, "triTriCollisionBda.spv", "triangleTriangleCollision", {}, sizeof(PushH));
            PXG_CHECK_MSG(k.isValid(), "%s: the kernel pipeline must build", ps.label);
            if (k.isValid()) k.dispatch(NALL, {}, push);
        }
        dev->flush();

        const auto gRec = downloadBuffer<uint32_t>(dev, bRec, records.size());
        const auto gTouch = downloadBuffer<uint32_t>(dev, bTouch, touch.size());
        const auto gPF = downloadBuffer<uint32_t>(dev, bPF, patchF.size());
        const auto gCnt = downloadBuffer<uint32_t>(dev, bCnt, counters.size());
        const auto gPS = downloadBuffer<uint32_t>(dev, bPS, pstream.size());
        const auto gCS = downloadBuffer<uint32_t>(dev, bCS, cstream.size());
        const auto gDec = downloadBuffer<uint32_t>(dev, bDec, decoy.size());

        size_t mm = 0;
        auto cmp = [&](const char* what, const std::vector<uint32_t>& got, const std::vector<uint32_t>& want) {
            for (size_t w = 0; w < want.size(); ++w)
                if (got[w] != want[w])
                {
                    if (mm < 6)
                        std::printf("  %s: %s word %zu: got 0x%08x want 0x%08x\n", ps.label, what, w, got[w], want[w]);
                    ++mm;
                }
        };
        cmp("counters", gCnt, wCnt);
        cmp("records", gRec, wRec);
        cmp("touchChange", gTouch, wTouch);
        cmp("patchChange", gPF, wPF);
        cmp("patch stream", gPS, wPS);
        cmp("contact stream", gCS, wCS);
        cmp("decoy", gDec, decoy);
        mismatches += mm;

        // ---- the discriminators ----
        if (R.doFirst) ++flippedCases;
        if (R.st.passes >= 2u) ++pass1Cases;
        if (R.st.rounds > R.st.passes + 1u && R.st.fullRounds > 0u) ++multiRound;
        if (R.st.find.childEntries > 0u) ++refineCases;
        if (R.N >= 2u) ++multiPatch;
        if (R.N > 0u) ++withContacts;
        if (realRes.ovfBits & tr::OVF_CONTACT) ++contactOvfCases;
        if (realRes.ovfBits & tr::OVF_FORCE) ++forceOvfCases;
        cullBand1 += R.st.cullBandPass1; cullBand0 += R.st.cullBandPass0; candBand0 += R.st.candBandPass0;
        ratioBand += R.st.ratioBand; level7 += R.st.level7Requests; capDrops += uint32_t(R.st.find.levelCapDrops);
        refineOverflow += R.st.find.refineOverflow; childLastSlot += R.st.find.childLastSlot;
        if (R.st.maxPatches >= 128u) ++fullPatchCases;
        if (R.doFirst && R.st.doFirstByNoSdfAlone) ++noSdfAloneCases;
        const bool pass1Alone = R.N > 0u && R.st.candPass[0] == 0u && R.st.candPass[1] > 0u;
        if (pass1Alone) ++(R.doFirst ? pass1AloneFlipped : pass1AloneUnflipped);
        if (R.doFirst && R.N > 0u && ps.s0 != ONE && ps.s1 != ONE && !ps.r0.isIdentity() && !ps.r1.isIdentity()) ++flippedScaled;
        if (R.N > 0u && !meshes[R.doFirst ? ps.m1 : ps.m0].sdf && (R.doFirst ? ps.s0 : ps.s1) != ONE) ++scaledFloor;
        if (R.N > 0u && (seedStatus & tr::FLAG_HAS_TOUCH) != 0u && (seedStatus & tr::FLAG_HAS_NO_TOUCH) == 0u)
            ++touchSeedCases;
        if (R.N > 0u && (seedStatus & tr::FLAG_TOUCH_KNOWN) == tr::FLAG_HAS_NO_TOUCH) ++noTouchSeedCases;
        {
            // doFirst (:1358-1360) against five mutated spellings, in a case with contacts.
            const MeshDef& E0 = meshes[ps.m0];
            const MeshDef& E1 = meshes[ps.m1];
            const bool A = E0.sdf, B = !E1.sdf, C = E0.projFlag == 0u && E1.projFlag != 0u;
            const bool Eq = E1.projFlag == E0.projFlag, V = E1.verts.size() < E0.verts.size();
            const bool T = E1.inds.size() < E0.inds.size();
            const bool mut[5] = { B || C || (Eq && V),
                                  A && (C || (Eq && V)),
                                  A && (B || E1.projFlag != 0u || (Eq && V)),
                                  A && (B || C || V),
                                  A && (B || C || (Eq && T)) };
            for (uint32_t m = 0; m < 5u; ++m)
                if (R.N > 0u && mut[m] != R.doFirst) ++doFirstKill[m];
        }
        if (R.st.cullBandPass1 + R.st.cullBandPass0 + R.st.candBandPass0 + R.st.ratioBand + R.st.level7Requests +
            R.st.find.levelCapDrops + R.st.find.refineOverflow + R.st.find.childLastSlot > 0u)
            std::printf("      bands: cull pass1 %u, cull pass0 %u, candidate pass0 %u, ratio %u; level-7 requests %u, "
                        "cap drops %zu, refinement overflows %zu, child in the last slot %zu\n",
                        R.st.cullBandPass1, R.st.cullBandPass0, R.st.candBandPass0, R.st.ratioBand,
                        R.st.level7Requests, R.st.find.levelCapDrops, R.st.find.refineOverflow, R.st.find.childLastSlot);
        std::printf("  %-52s doFirst %d, %u pass(es), %u round(s) (%u full), %u found, %u+%u candidates, "
                    "%u refinement entries, %zu children, N = %u, %u points, ovf %u, %zu mismatches\n",
                    ps.label, int(R.doFirst), R.st.passes, R.st.rounds, R.st.fullRounds, R.st.found,
                    R.st.candPass[0], R.st.candPass[1], R.st.refineEntries, R.st.find.childEntries, R.N, total,
                    realRes.ovfBits, mm);

        for (PxgBufferHandle h : local) dev->destroyBuffer(h);
    }

    PXG_CHECK_MSG(mismatches == 0, "npTriTriKernel_pairs: %zu words differ from the replay", mismatches);
    PXG_CHECK_MSG(flippedCases >= 3 && pass1Cases >= 3 && multiRound >= 1 && refineCases >= 2 && multiPatch >= 2 &&
                  overflowCases >= 3 && withContacts >= 6,
                  "npTriTriKernel_pairs: coverage is thin -- %zu flipped, %zu with pass 1, %zu multi-round, "
                  "%zu refining, %zu with several patches, %zu overflowing, %zu with contacts",
                  flippedCases, pass1Cases, multiRound, refineCases, multiPatch, overflowCases, withContacts);
    PXG_CHECK_MSG(cullBand1 > 0 && cullBand0 > 0 && candBand0 > 0 && ratioBand > 0 && level7 > 0 && capDrops > 0 &&
                  fullPatchCases >= 1 && noSdfAloneCases >= 1,
                  "npTriTriKernel_pairs: a discriminator is empty -- cull band pass1 %u, pass0 %u, candidate band %u, "
                  "ratio band %u, level-7 requests %u, cap drops %u, 128-patch cases %zu, doFirst by no-SDF alone %zu",
                  cullBand1, cullBand0, candBand0, ratioBand, level7, capDrops, fullPatchCases, noSdfAloneCases);
    PXG_CHECK_MSG(refineOverflow > 0 && childLastSlot > 0 && contactOvfCases >= 1 && forceOvfCases >= 1 &&
                  pass1AloneUnflipped >= 1 && pass1AloneFlipped >= 1 && flippedScaled >= 1 && scaledFloor >= 1 &&
                  touchSeedCases >= 1 && noTouchSeedCases >= 1,
                  "npTriTriKernel_pairs: a discriminator is empty -- refinement overflows %zu, child in the last slot "
                  "%zu, contact overflows %zu, force overflows %zu, pass 1 alone %zu unflipped / %zu flipped, flipped "
                  "and scaled %zu, scaled floor %zu, prior touch %zu, prior no-touch %zu",
                  refineOverflow, childLastSlot, contactOvfCases, forceOvfCases, pass1AloneUnflipped, pass1AloneFlipped,
                  flippedScaled, scaledFloor, touchSeedCases, noTouchSeedCases);
    for (uint32_t m = 0; m < 5u; ++m)
        PXG_CHECK_MSG(doFirstKill[m] > 0, "npTriTriKernel_pairs: no case with contacts tells doFirst from its "
                      "mutant (%s)", doFirstMutant[m]);
    std::printf("  npTriTriKernel_pairs: %zu cases, %zu device probe dispatches for the replay; %zu flipped, "
                "%zu with pass 1 (%zu + %zu alone), %zu multi-round, %zu refining, %zu with several patches, "
                "%zu overflowing (%zu contact, %zu force); doFirst mutants told apart by %zu/%zu/%zu/%zu/%zu cases\n",
                cases.size(), oracle.dispatches, flippedCases, pass1Cases, pass1AloneUnflipped, pass1AloneFlipped,
                multiRound, refineCases, multiPatch, overflowCases, contactOvfCases, forceOvfCases,
                doFirstKill[0], doFirstKill[1], doFirstKill[2], doFirstKill[3], doFirstKill[4]);

    for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
    delete shim;
}
