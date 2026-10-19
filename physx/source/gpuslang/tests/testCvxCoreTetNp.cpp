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

// ===========================================================================
// convexCoreTetmeshNphase_Kernel32 on the SHIPPED engine .spv
// ===========================================================================
//
// The tetrahedron twin of cvxCoreClothNp (testFemClothSim.cpp).  Until this fixture
// cvxCoreTetNpBda.spv was loaded by the softbody convex-core scene only, and that
// scene has one softbody (id 0), one dynamic rigid, no filter pairs and one contact
// manager, so most of what PhysX 5.12 changed on this path -- the rigid-volume
// filter in front of generateContacts and the PxgRigidDbContactWriter's
// writeVolumeContact -- was unobservable there.
//
// THE ORACLE is convexCoreCollision.cu (5.12) :1019-1172, read for each check below
// and derived analytically where the geometry allows it; the references are encoded
// with the ENGINE's own inline helpers (PxgEncodeDeformableTetrahedron,
// PxgEncodeRigidFilterRef, PxgComparePairInfo), not transcriptions of them.
//
// THE GEOMETRY.
//
//   Two analytic convex-core BOXES of FULL extents (2, 2, 3) at the origin, so the top
//   face is y = +1.  The z extent differs from x/y on purpose: the BVH callback margin
//   is `bounds.getDimensions().minElement() * 0.0001f` (:560), and on a cube min and
//   max element coincide, so a port taking maxElement would be invisible.
//
//   Softbody SB_ID = 2 (not 0, so the PxgDeformableRef shift is observable), with
//   four tetrahedra whose vertices are world-space (identity transform cache):
//     tet 0  base triangle at y = 0.9, apex at 1.5  -> 0.1 inside the top face
//     tet 1  base triangle at y = 0.8, apex at 1.5  -> 0.2 inside
//     tet 2  far above the box                       -> visited, no contact
//     tet 3  like tet 0, but in a BV32 leaf whose box is culled
//   Two different depths make the tet index <-> vertices association observable.
//
//   The BV32 tree is one root with three LEAF children; the child boxes are chosen
//   by the fixture, not refitted, because they are what makes the traversal's cull
//   observable (the fattened convex bounds top out at y = 1 + contactDist = 1.02):
//     child 0  tet 0       box y >= -1000   always visited
//     child 1  tets 1, 2   box y >= 1.015   visited only with the FULL contactDist
//                                           (either shape's offset alone gives 1.014
//                                           or 1.006 and culls it)
//     child 2  tet 3       box y >= 1.025   culled: tet 3 penetrates, and any contact
//                                           of it means the cull is gone
//
//   Three contact managers, numTests = 2:
//     cm 0  core-first, dynamic rigid RIGID_A (both PxNodeIndex words non-zero)
//     cm 1  MESH-FIRST (checkTypes must swap), a STATIC rigid (PxNodeIndex()), so its
//           filter reference is PxgEncodeStaticShapeRef(cacheRef0) -- of the CORE's
//           cache ref, i.e. the post-swap one
//     cm 2  would collide too, but lies past numTests
//
// WHAT THE KERNEL MUST WRITE, per contact (:1158-1170):
//   point = points[i] + normal * dists[i] * 0.5f -- on the tet's face plane or the box
//   face, never on the midway plane; .w = restDistance[testIndex]
//   normalPen = (-normal, dists[i] * 0.4f when penetrating) -- the "fake ERP"
//   info = (rigidId.getInd(), PxgEncodeDeformableTetrahedron(softbodyId, tetIndex))
//   rigidMaterialIndex = PxU16(shape0.materialIndex), shape0 being the post-swap core
//   contactByRigid / tempContactByRigid / contactIndexSortedByRigid = the sort key
//   NO barycentric: sb_other_contact_remap_to_simLaunch computes it later.
// ===========================================================================

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers the engine
// descriptor headers decorate their inline members with; must precede PxgSoftBody.h.
#include "host_defines.h"
#include "PxgSoftBody.h"                     // real engine per-softbody descriptor
#include "PxNodeIndex.h"                     // PxNodeIndex (dynamic vs static rigid)
#include "PxgSimulationCoreDesc.h"           // PxgDeformablePairInfo / PxgEncode* (5.12)
#include "PxgDeformableContactWriters.h"     // PxgRigidDbContactWriter (5.12)
#include "PxgContactManager.h"               // PxgContactManagerInput
#include "PxgConvexConvexShape.h"            // PxgShape
#include "PxsTransformCache.h"               // PxsCachedTransform
#include "geometry/PxGeometry.h"             // PxGeometryType
#include "geometry/PxConvexCoreGeometry.h"   // PxConvexCore::Type::eBOX

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };

struct DevBuf
{
    PxgBufferHandle h = 0;
    uint64_t addr = 0;
};

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// Push blob, byte-identical to cvxCoreTetNpBda.slang's CvxCoreTetParams and to the
// kCvxCoreTet ABI row: numTests, seven pointers, the 80-byte PxgRigidDbContactWriter.
struct CvxCoreTetPush
{
    uint32_t numTests, pad4;
    uint64_t cmInputs, shapes, softbodies, shapeToRigidRemapTable, transformCache;
    uint64_t contactDistance, restDistance;
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo, w_outRigidMaterialIndex;
    uint64_t w_totalContactCount, w_contactByRigid, w_tempContactByRigid, w_contactIndexSortedByRigid;
    uint32_t w_maxNumContacts, pad140;
};
static_assert(sizeof(CvxCoreTetPush) == 144, "convex-core tet push blob must be 144 bytes");
static_assert(offsetof(CvxCoreTetPush, w_outPoint) == 64, "writer @64");
static_assert(offsetof(CvxCoreTetPush, w_maxNumContacts) == 136, "maxNumContacts @136");
static_assert(sizeof(PxgRigidDbContactWriter) == 80, "PxgRigidDbContactWriter sizeof");
static_assert(offsetof(CvxCoreTetPush, w_maxNumContacts) - offsetof(CvxCoreTetPush, w_outPoint) ==
              offsetof(PxgRigidDbContactWriter, maxNumContacts), "writer.maxNumContacts");

// The writer's output streams.  The barycentric starts at a sentinel: writeVolumeContact
// writes none, and a port that wrote one (writeSurfaceContact's tail) would show.
static const F4 kBarySentinel = { 7.0f, -7.0f, 7.0f, -7.0f };
struct WriterBufs
{
    DevBuf point, normalPen, barycentric, info, rigidMat, count, byRigid, tempByRigid, idxByRigid;
    uint32_t maxContacts;
};
static WriterBufs makeWriter(PxgSlangComputeDevice* dev, uint32_t maxContacts)
{
    WriterBufs w; w.maxContacts = maxContacts;
    std::vector<F4> zf4(maxContacts, F4{ 0, 0, 0, 0 });
    std::vector<F4> bary(maxContacts, kBarySentinel);
    std::vector<PxgDeformablePairInfo> zinfo(maxContacts);
    std::memset(zinfo.data(), 0, zinfo.size() * sizeof(PxgDeformablePairInfo));
    std::vector<uint32_t> zmat((maxContacts + 1u) / 2u, 0xffffffffu);   // PxU16 stream, two per word
    std::vector<uint32_t> zc = { 0u };
    std::vector<uint64_t> zu64(maxContacts, 0xffffffffffffffffull);
    std::vector<uint32_t> zu32(maxContacts, 0xffffffffu);
    w.point = makeDev(dev, zf4); w.normalPen = makeDev(dev, zf4); w.barycentric = makeDev(dev, bary);
    w.info = makeDev(dev, zinfo); w.rigidMat = makeDev(dev, zmat); w.count = makeDev(dev, zc);
    w.byRigid = makeDev(dev, zu64); w.tempByRigid = makeDev(dev, zu32); w.idxByRigid = makeDev(dev, zu32);
    return w;
}
static void destroyWriter(PxgSlangComputeDevice* dev, const WriterBufs& w)
{
    for (PxgBufferHandle h : { w.point.h, w.normalPen.h, w.barycentric.h, w.info.h, w.rigidMat.h,
                               w.count.h, w.byRigid.h, w.tempByRigid.h, w.idxByRigid.h })
        dev->destroyBuffer(h);
}
static void fillWriter(CvxCoreTetPush& p, const WriterBufs& w)
{
    p.w_outPoint = w.point.addr; p.w_outNormalPen = w.normalPen.addr; p.w_outBarycentric = w.barycentric.addr;
    p.w_outContactInfo = w.info.addr; p.w_outRigidMaterialIndex = w.rigidMat.addr;
    p.w_totalContactCount = w.count.addr; p.w_contactByRigid = w.byRigid.addr;
    p.w_tempContactByRigid = w.tempByRigid.addr; p.w_contactIndexSortedByRigid = w.idxByRigid.addr;
    p.w_maxNumContacts = w.maxContacts;
}

// Everything one dispatch wrote, the writer's count clamped to the buffers.
struct TetRun
{
    uint32_t count = 0;
    std::vector<F4> point, normalPen, bary;
    std::vector<PxgDeformablePairInfo> info;
    std::vector<uint32_t> mat, tempByRigid, idxByRigid;
    std::vector<uint64_t> byRigid;
};
static TetRun readRun(PxgSlangComputeDevice* dev, const WriterBufs& w)
{
    TetRun r;
    r.count = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    const uint32_t n = std::min(r.count, w.maxContacts);
    r.point = downloadBuffer<F4>(dev, w.point.h, n);
    r.normalPen = downloadBuffer<F4>(dev, w.normalPen.h, n);
    r.bary = downloadBuffer<F4>(dev, w.barycentric.h, n);
    r.info = downloadBuffer<PxgDeformablePairInfo>(dev, w.info.h, n);
    r.byRigid = downloadBuffer<uint64_t>(dev, w.byRigid.h, n);
    r.tempByRigid = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, n);
    r.idxByRigid = downloadBuffer<uint32_t>(dev, w.idxByRigid.h, n);
    const std::vector<uint32_t> words = downloadBuffer<uint32_t>(dev, w.rigidMat.h, (w.maxContacts + 1u) / 2u);
    for (uint32_t i = 0; i < n; ++i)
        r.mat.push_back((words[i / 2u] >> (16u * (i & 1u))) & 0xffffu);
    return r;
}

// Gu::BV32DataPacked, 292 words: mMin[32] float4 @0, mMax[32] float4 @128,
// mData[32] @256, mNbNodes @288, mDepth @289.
static const uint32_t BV32_WORDS = 292u;
static const uint32_t W_MMIN = 0u, W_MMAX = 128u, W_MDATA = 256u, W_NBNODES = 288u, W_DEPTH = 289u;
static uint32_t leafWord(uint32_t startIndex, uint32_t nbPrims) { return (startIndex << 7) | (nbPrims << 1) | 1u; }

static std::vector<PxgDeformablePairInfo> sortedFilters(std::vector<PxgDeformablePairInfo> f)
{
    // findRigidVolumeFilter binary-searches with PxgComparePairInfo, so the list must be
    // sorted by the same order -- the engine's own comparator, not a re-spelling of it.
    std::sort(f.begin(), f.end(), [](const PxgDeformablePairInfo& a, const PxgDeformablePairInfo& b) {
        return PxgComparePairInfo(a, b) < 0;
    });
    return f;
}

PXG_TEST(cvxCoreTetNp)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the values the SCENE cannot choose ------------------------------
    const uint32_t SB_ID = 2u;
    const PxNodeIndex RIGID_A(PxU64(0x00000007ABCDEF01ull));   // dynamic: low word valid, both words non-zero
    const PxNodeIndex RIGID_STATIC;                             // PxNodeIndex(): the static marker
    const PxNodeIndex RIGID_C(PxU64(0x0000000300000C0Cull));    // cm 2's, which must never appear
    const uint64_t SB_REMAP_POISON = 0x5555AAAA5555AAAAull;     // remap[softbody cache ref]
    PXG_CHECK(!RIGID_A.isStaticBody() && RIGID_STATIC.isStaticBody());
    const uint32_t MAT_A = 0x1234u, MAT_B = 0x0BEEu, MAT_C = 0x0CCCu;
    const float REST_0 = 0.05f, REST_1 = 0.07f, REST_POISON = 999.0f;
    // Asymmetric offsets: only their SUM (0.02) lifts the fattened top to 1.02 >= 1.015.
    const float OFF_CORE = 0.014f, OFF_SB = 0.006f;

    // cache refs: 0 box A, 1 softbody, 2 box B (static), 3 box C
    const uint32_t CR_A = 0u, CR_SB = 1u, CR_B = 2u, CR_C = 3u;

    // ---- the tetrahedra --------------------------------------------------
    const float BOX_TOP_Y = 1.0f;
    const float BASE_Y[4] = { 0.9f, 0.8f, 5.0f, 0.9f };
    std::vector<F4> verts;
    std::vector<uint32_t> tetInds;
    for (uint32_t t = 0; t < 4u; ++t)
    {
        const float by = BASE_Y[t];
        const float ay = (t == 2u) ? 6.0f : 1.5f;
        const float ox = (t == 1u) ? 0.1f : 0.0f;   // tet 1 is not a copy of tet 0
        const uint32_t base = uint32_t(verts.size());
        verts.push_back(F4{ ox - 0.5f, by, -0.5f, 1.0f });
        verts.push_back(F4{ ox + 0.5f, by, -0.5f, 1.0f });
        verts.push_back(F4{ ox + 0.0f, by,  0.5f, 1.0f });
        verts.push_back(F4{ ox + 0.0f, ay,  0.0f, 1.0f });
        for (uint32_t k = 0; k < 4u; ++k) tetInds.push_back(base + k);
    }
    DevBuf bVerts = makeDev(dev, verts);
    DevBuf bTets = makeDev(dev, tetInds);

    // ---- the BV32 tree: one root, three leaf children ----------------------
    std::vector<uint32_t> tree(4u + BV32_WORDS, 0u);
    tree[0] = uint32_t(verts.size()); tree[1] = 4u; tree[2] = 1u; tree[3] = 1u;   // header, skipped
    {
        const uint32_t nw = 4u;
        const float loY[3] = { -1000.0f, 1.015f, 1.025f };
        const uint32_t start[3] = { 0u, 1u, 3u }, nb[3] = { 1u, 2u, 1u };
        for (uint32_t k = 0; k < 3u; ++k)
        {
            tree[nw + W_MMIN + k * 4 + 0] = f2u(-1000.0f);
            tree[nw + W_MMIN + k * 4 + 1] = f2u(loY[k]);
            tree[nw + W_MMIN + k * 4 + 2] = f2u(-1000.0f);
            tree[nw + W_MMAX + k * 4 + 0] = f2u(1000.0f);
            tree[nw + W_MMAX + k * 4 + 1] = f2u(1000.0f);
            tree[nw + W_MMAX + k * 4 + 2] = f2u(1000.0f);
            tree[nw + W_MDATA + k] = leafWord(start[k], nb[k]);
        }
        tree[nw + W_NBNODES] = 3u;
        tree[nw + W_DEPTH] = 1u;
    }
    DevBuf bTree = makeDev(dev, tree);

    // ---- the filters (5.12): near misses only, none of which may suppress ----
    const uint64_t staticRefB = PxgEncodeRigidFilterRef(RIGID_STATIC, CR_B);
    const uint64_t refA = PxgEncodeRigidFilterRef(RIGID_A, CR_A);
    PXG_CHECK_MSG(staticRefB != RIGID_STATIC.getInd() && refA == RIGID_A.getInd(),
                  "the fixture assumes a static's filter ref is its shape ref and a body's its node");
    std::vector<PxgDeformablePairInfo> nearMisses = {
        // tet 0's first VERTEX (id 0, same feature id) under the right rigid: wrong feature type
        PxgDeformablePairInfo{ refA, PxgEncodeDeformableVertex(SB_ID, tetInds[0]) },
        // tet 0 of ANOTHER softbody under the right rigid
        PxgDeformablePairInfo{ refA, PxgEncodeDeformableTetrahedron(SB_ID + 1u, 0u) },
        // tet 0 under the SOFTBODY's static shape ref: what a key on the PRE-swap cacheRef0 hits
        PxgDeformablePairInfo{ PxgEncodeStaticShapeRef(CR_SB), PxgEncodeDeformableTetrahedron(SB_ID, 0u) },
        // tet 1 under the static's RAW node index: what a key without PxgEncodeRigidFilterRef hits
        PxgDeformablePairInfo{ RIGID_STATIC.getInd(), PxgEncodeDeformableTetrahedron(SB_ID, 1u) },
    };
    const std::vector<PxgDeformablePairInfo> filters = sortedFilters(nearMisses);
    DevBuf bFilters = makeDev(dev, filters);

    auto makeSoftbodies = [&](const DevBuf& f, uint32_t nbF) {
        std::vector<PxgSoftBody> arr(SB_ID + 1u);
        std::memset(arr.data(), 0, arr.size() * sizeof(PxgSoftBody));
        PxgSoftBody& sb = arr[SB_ID];
        sb.mTetMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
        sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
        sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
        sb.mNumTets = 4u;
        sb.mNumVerts = uint32_t(verts.size());
        sb.mRigidVolumeFilterPairs = reinterpret_cast<PxgDeformablePairInfo*>(size_t(f.addr));
        sb.mNumRigidVolumeFilterPairs = nbF;
        return makeDev(dev, arr);
    };
    DevBuf bSb = makeSoftbodies(bFilters, uint32_t(filters.size()));

    // ---- shapes ------------------------------------------------------------
    std::vector<PxgShape> shapes(4);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    auto makeCore = [](PxgShape& s, uint32_t mat) {
        // The core TYPE rides in hullOrMeshPtr, the core DATA in the PxMeshScale bytes
        // (convexCoreCollision.cu makeConvexShape); BoxCore holds FULL extents and the
        // scale rotation's w is the margin.
        s.type = PxU32(PxGeometryType::eCONVEXCORE);
        s.hullOrMeshPtr = size_t(PxConvexCore::Type::eBOX);
        s.scale.scale = PxVec3(2.0f, 2.0f, 3.0f);
        s.scale.rotation = PxQuat(0.0f, 0.0f, 0.0f, 0.0f);
        s.materialIndex = mat;
    };
    makeCore(shapes[0], MAT_A);
    shapes[1].type = PxU32(PxGeometryType::eTETRAHEDRONMESH);
    shapes[1].particleOrSoftbodyId = SB_ID;
    makeCore(shapes[2], MAT_B);
    makeCore(shapes[3], MAT_C);
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<PxsCachedTransform> tcache(4);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    for (PxsCachedTransform& t : tcache) t.transform = PxTransform(PxIdentity);
    DevBuf bTc = makeDev(dev, tcache);

    std::vector<PxgContactManagerInput> cms(3);
    std::memset(cms.data(), 0, cms.size() * sizeof(PxgContactManagerInput));
    cms[0].shapeRef0 = 0u; cms[0].shapeRef1 = 1u; cms[0].transformCacheRef0 = CR_A;  cms[0].transformCacheRef1 = CR_SB;
    cms[1].shapeRef0 = 1u; cms[1].shapeRef1 = 2u; cms[1].transformCacheRef0 = CR_SB; cms[1].transformCacheRef1 = CR_B;
    cms[2].shapeRef0 = 3u; cms[2].shapeRef1 = 1u; cms[2].transformCacheRef0 = CR_C;  cms[2].transformCacheRef1 = CR_SB;
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> contactDist = { OFF_CORE, OFF_SB, OFF_CORE, OFF_CORE };
    DevBuf bCd = makeDev(dev, contactDist);
    // restDistance is indexed by the TEST index (TestInput, :94), not a cache ref.
    std::vector<float> rest = { REST_0, REST_1, REST_POISON };
    DevBuf bRest = makeDev(dev, rest);
    std::vector<uint64_t> remap = { RIGID_A.getInd(), SB_REMAP_POISON, RIGID_STATIC.getInd(), RIGID_C.getInd() };
    DevBuf bRemap = makeDev(dev, remap);

    const uint32_t MAX_CONTACTS = 128u;
    CvxCoreTetPush push = {};
    push.numTests = 2u;
    push.cmInputs = bCms.addr;
    push.shapes = bShapes.addr;
    push.softbodies = bSb.addr;
    push.shapeToRigidRemapTable = bRemap.addr;
    push.transformCache = bTc.addr;
    push.contactDistance = bCd.addr;
    push.restDistance = bRest.addr;

    Kernel k("cvxCoreTetNpBda.spv", "convexCoreTetmeshNphase_Kernel32", {}, sizeof(CvxCoreTetPush));
    // Engine launch: grid ceil(numTests / 4), block (32, 4).
    auto run = [&](const CvxCoreTetPush& p, uint32_t maxContacts) {
        WriterBufs w = makeWriter(dev, maxContacts);
        CvxCoreTetPush q = p;
        fillWriter(q, w);
        k.dispatch((q.numTests + 3u) / 4u, {}, q, 1u, 1u);
        dev->flush();
        TetRun r = readRun(dev, w);
        destroyWriter(dev, w);
        return r;
    };

    // ---- the analytic expectations ----------------------------------------
    // convexBounds = convex.computeBounds() (support with margin 0: +-1, +-1, +-1.5),
    // fattenFast(contactDist), then the callback margin of THOSE bounds.
    const float cd = OFF_CORE + OFF_SB;
    const float dimX = (1.0f + cd) - (-1.0f - cd);
    const float dimZ = (1.5f + cd) - (-1.5f - cd);
    const float tetMargin = std::min(dimX, dimZ) * 0.0001f;
    auto expectedPen = [&](uint32_t tet) {
        const float dist = -((BOX_TOP_Y - BASE_Y[tet]) + tetMargin);
        return dist * 0.4f;   // "VR @@@ add fake ERP" (:1162)
    };
    const uint64_t tetRef[4] = {
        PxgEncodeDeformableTetrahedron(SB_ID, 0u), PxgEncodeDeformableTetrahedron(SB_ID, 1u),
        PxgEncodeDeformableTetrahedron(SB_ID, 2u), PxgEncodeDeformableTetrahedron(SB_ID, 3u) };

    // Checks every per-contact invariant; returns the per-(test, tet) contact counts.
    struct Tally { uint32_t n[2][4] = {}; uint32_t stray = 0; };
    auto check = [&](const TetRun& r, const char* tag) {
        Tally t;
        size_t mm = 0;
        const uint32_t n = std::min(r.count, MAX_CONTACTS);
        for (uint32_t i = 0; i < n; ++i)
        {
            // which contact manager: by the rigid reference, the only per-test field
            int test = -1;
            if (r.info[i].ref0 == RIGID_A.getInd()) test = 0;
            else if (r.info[i].ref0 == RIGID_STATIC.getInd()) test = 1;
            int tet = -1;
            for (uint32_t q = 0; q < 4u; ++q)
                if (r.info[i].ref1 == tetRef[q]) tet = int(q);
            if (test < 0 || tet < 0)
            {
                ++t.stray; ++mm;
                std::printf("  %s[%u] info (0x%016llx, 0x%016llx) is neither test 0/1 nor a tet of "
                            "softbody %u (a remap read at the PRE-swap cacheRef0 gives 0x%016llx, "
                            "cm 2 gives 0x%016llx)\n", tag, i, (unsigned long long)r.info[i].ref0,
                            (unsigned long long)r.info[i].ref1, SB_ID,
                            (unsigned long long)SB_REMAP_POISON, (unsigned long long)RIGID_C.getInd());
                continue;
            }
            ++t.n[test][tet];

            const float restExp = test == 0 ? REST_0 : REST_1;
            if (f2u(r.point[i].w) != f2u(restExp))
            { ++mm; std::printf("  %s[%u] outPoint.w = %g, expected restDistance[testIndex %d] = %g\n",
                                tag, i, r.point[i].w, test, restExp); }

            const uint32_t matExp = test == 0 ? MAT_A : MAT_B;
            if (r.mat[i] != matExp)
            { ++mm; std::printf("  %s[%u] rigidMaterialIndex 0x%04x, expected 0x%04x (shape0 AFTER the swap)\n",
                                tag, i, r.mat[i], matExp); }

            const uint64_t key = r.info[i].ref0;
            if (r.byRigid[i] != key || r.tempByRigid[i] != PxgGetContactSortKey(key) || r.idxByRigid[i] != i)
            { ++mm; std::printf("  %s[%u] sort key (0x%016llx, 0x%08x, %u), expected (0x%016llx, 0x%08x, %u)\n",
                                tag, i, (unsigned long long)r.byRigid[i], r.tempByRigid[i], r.idxByRigid[i],
                                (unsigned long long)key, PxgGetContactSortKey(key), i); }

            if (f2u(r.bary[i].x) != f2u(kBarySentinel.x) || f2u(r.bary[i].y) != f2u(kBarySentinel.y) ||
                f2u(r.bary[i].z) != f2u(kBarySentinel.z) || f2u(r.bary[i].w) != f2u(kBarySentinel.w))
            { ++mm; std::printf("  %s[%u] a barycentric was written (%g,%g,%g,%g); writeVolumeContact "
                                "writes none\n", tag, i, r.bary[i].x, r.bary[i].y, r.bary[i].z, r.bary[i].w); }

            // The tet sits inside the box below its top face, so the way out is +Y, and the
            // written normal is the NEGATED faceClip normal (:1164) -- the sign is the check.
            if (!(almostEqual(r.normalPen[i].x, 0.0f, 1e-4f, 1e-5f) && almostEqual(r.normalPen[i].z, 0.0f, 1e-4f, 1e-5f) &&
                  almostEqual(r.normalPen[i].y, 1.0f, 1e-4f, 1e-5f)))
            { ++mm; std::printf("  %s[%u] normal (%g,%g,%g), expected (0,1,0)\n", tag, i,
                                r.normalPen[i].x, r.normalPen[i].y, r.normalPen[i].z); }

            // depth + the minElement callback margin, scaled by the 0.4 ERP.  maxElement
            // would be 4e-5 off, no ERP 0.06 off, the other tet's depth 0.04 off.
            const float penExp = expectedPen(uint32_t(tet));
            if (!almostEqual(r.normalPen[i].w, penExp, 1e-4f, 2e-6f))
            { ++mm; std::printf("  %s[%u] pen = %.8g, expected %.8g (tet %d: depth %g + margin %g, x0.4)\n",
                                tag, i, r.normalPen[i].w, penExp, tet, BOX_TOP_Y - BASE_Y[tet], tetMargin); }

            // The half-separation shift lands the midpoint on one of the two face planes.
            const bool onTet = almostEqual(r.point[i].y, BASE_Y[tet], 1e-3f, 1e-3f);
            const bool onBox = almostEqual(r.point[i].y, BOX_TOP_Y, 1e-3f, 1e-3f);
            if (!onTet && !onBox)
            { ++mm; std::printf("  %s[%u] point.y = %g, expected the tet face %g or the box face %g "
                                "(%g is the un-shifted midpoint)\n", tag, i, r.point[i].y, BASE_Y[tet],
                                BOX_TOP_Y, 0.5f * (BASE_Y[tet] + BOX_TOP_Y)); }
        }
        PXG_CHECK_MSG(mm == 0, "%s: convexCoreTetmeshNphase_Kernel32 contact invariants (%zu mismatches)", tag, mm);
        return t;
    };

    // =======================================================================
    // CASE A -- the near-miss filter list, nothing suppressed.
    // =======================================================================
    push.numTests = 2u;
    const TetRun a = run(push, MAX_CONTACTS);
    PXG_CHECK_MSG(a.count > 0u && a.count <= MAX_CONTACTS, "case A emitted %u contacts", a.count);
    const Tally ta = check(a, "cvxCoreTet A");
    for (uint32_t test = 0; test < 2u; ++test)
    {
        PXG_CHECK_MSG(ta.n[test][0] > 0u && ta.n[test][1] > 0u,
                      "test %u: tet 0 gave %u contacts and tet 1 %u -- both penetrate the box and "
                      "no filter entry names them", test, ta.n[test][0], ta.n[test][1]);
        PXG_CHECK_MSG(ta.n[test][2] == 0u, "test %u: tet 2 is 4 above the box and gave %u contacts",
                      test, ta.n[test][2]);
        PXG_CHECK_MSG(ta.n[test][3] == 0u,
                      "test %u: tet 3 gave %u contacts, but its BV32 leaf (y >= 1.025) lies above the "
                      "fattened convex bounds (y <= 1.02) and must be culled", test, ta.n[test][3]);
    }
    // The swap arm must make the mesh-first pair identical to the core-first one.
    PXG_CHECK_MSG(ta.n[0][0] == ta.n[1][0] && ta.n[0][1] == ta.n[1][1],
                  "core-first (%u, %u) and mesh-first (%u, %u) contacts per tet disagree",
                  ta.n[0][0], ta.n[0][1], ta.n[1][0], ta.n[1][1]);
    std::printf("    [ info ] cvxCoreTet A: %u contacts; test 0 tets (%u, %u), test 1 tets (%u, %u); "
                "pen tet0 %.8g tet1 %.8g\n", a.count, ta.n[0][0], ta.n[0][1], ta.n[1][0], ta.n[1][1],
                expectedPen(0u), expectedPen(1u));

    // =======================================================================
    // CASE B -- numTests bounds the walk: with 3 the cm 2 pair (rigid C) appears,
    // which proves case A's absence of it was the bound and not the geometry.
    // =======================================================================
    {
        CvxCoreTetPush p = push;
        p.numTests = 3u;
        const TetRun b = run(p, MAX_CONTACTS);
        uint32_t ofC = 0u;
        for (uint32_t i = 0; i < std::min(b.count, MAX_CONTACTS); ++i)
            if (b.info[i].ref0 == RIGID_C.getInd()) ++ofC;
        PXG_CHECK_MSG(ofC > 0u && b.count == a.count + ofC,
                      "numTests 3: %u contacts of cm 2 and %u in all (case A had %u)", ofC, b.count, a.count);
    }

    // =======================================================================
    // CASE C -- THE FILTER BITES (5.12): (rigid A, tet 1) and (static shape ref of the
    // core B, tet 0) are added.  Test 0 keeps only tet 0 and test 1 only tet 1, each
    // with exactly the contacts it made in case A.
    // =======================================================================
    {
        std::vector<PxgDeformablePairInfo> fc = nearMisses;
        fc.push_back(PxgDeformablePairInfo{ refA, tetRef[1] });
        fc.push_back(PxgDeformablePairInfo{ staticRefB, tetRef[0] });
        fc = sortedFilters(fc);
        DevBuf bF = makeDev(dev, fc);
        DevBuf bSbC = makeSoftbodies(bF, uint32_t(fc.size()));
        CvxCoreTetPush p = push;
        p.softbodies = bSbC.addr;
        const TetRun c = run(p, MAX_CONTACTS);
        const Tally tc = check(c, "cvxCoreTet C");
        PXG_CHECK_MSG(tc.n[0][0] == ta.n[0][0] && tc.n[0][1] == 0u && tc.n[1][0] == 0u &&
                      tc.n[1][1] == ta.n[1][1] && c.count == ta.n[0][0] + ta.n[1][1],
                      "filtered: test 0 tets (%u, %u), test 1 tets (%u, %u), %u in all; expected "
                      "(%u, 0), (0, %u) -- the static rigid's filter key is "
                      "PxgEncodeStaticShapeRef(post-swap cacheRef0)", tc.n[0][0], tc.n[0][1],
                      tc.n[1][0], tc.n[1][1], c.count, ta.n[0][0], ta.n[1][1]);
        dev->destroyBuffer(bF.h); dev->destroyBuffer(bSbC.h);
    }

    // =======================================================================
    // CASE D -- the writer's maxNumContacts bound.  The count is reserved by the
    // atomicAdd regardless; only the stores past the bound are dropped.
    // =======================================================================
    {
        const uint32_t SMALL = 2u;
        WriterBufs w = makeWriter(dev, SMALL + 2u);
        CvxCoreTetPush p = push;
        fillWriter(p, w);
        p.w_maxNumContacts = SMALL;
        k.dispatch(1u, {}, p, 1u, 1u);
        dev->flush();
        const uint32_t cnt = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
        const std::vector<F4> pts = downloadBuffer<F4>(dev, w.point.h, SMALL + 2u);
        PXG_CHECK_MSG(cnt == a.count, "with maxNumContacts %u the count is %u, expected the unclamped %u",
                      SMALL, cnt, a.count);
        for (uint32_t i = SMALL; i < SMALL + 2u; ++i)
            PXG_CHECK_MSG(f2u(pts[i].w) == 0u && f2u(pts[i].y) == 0u,
                          "slot %u past maxNumContacts %u was written (%g, %g)", i, SMALL, pts[i].y, pts[i].w);
        destroyWriter(dev, w);
    }

    for (PxgBufferHandle h : { bVerts.h, bTets.h, bTree.h, bFilters.h, bSb.h, bShapes.h, bTc.h, bCms.h,
                               bCd.h, bRest.h, bRemap.h })
        dev->destroyBuffer(h);
}

} // namespace
