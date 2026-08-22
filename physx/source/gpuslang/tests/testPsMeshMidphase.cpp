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

// testPsMeshMidphase.cpp -- P9.10b, the fixture for psMeshMidphaseBda.slang
// (ps_meshCollisonLaunch on the engine BDA ABI).
//
// THE ORACLE IS A SECOND GPU PROGRAM, not a host transcription.  psMeshMid.slang
// is a Phase-1 port of the SAME traverser with its own fixture
// (psMeshMid_traverser, which does carry a full host reference) and its header
// lists exactly what it lacks -- the engine glue.  psMeshMidphaseBda lifts that
// body and adds only glue.  So this file is the P8.97d GPU-DIFFERENTIAL
// arrangement: seed ONE arena, upload it twice (once as the Phase-1 module's
// seventeen bound buffers, once as the BDA module's cooked blob and descriptor
// chain), run both, and compare.  A third reading of the CUDA would be a third
// chance to make the same mistake twice; this way the traverser's correctness
// rests on psMeshMid_traverser's oracle and this file measures only the lift.
//
// THERE IS NOT ONE TOLERANCE IN THIS FILE.  Both programs are compiled without
// FMA contraction, from arithmetic written the same way, on identical inputs, so
// they must agree BIT FOR BIT.  A near-miss here would be a real finding about
// the lift, not rounding, and it must not be absorbed by an epsilon.
//
// FOUR TESTS:
//   psMeshMidphaseLayout    -- every engine constant and offset the shader spells,
//                              checked against the real headers.
//   psMeshMidphase          -- the differential, three dispatch shapes: the static
//                              one-way path, the same buffers a SECOND time so the
//                              twelve-contact cap binds and the overflow reaches the
//                              writer, and a dynamic rigid so everything does.  Plus
//                              the LoadShapePair swap, run both ways round.
//   psMeshMidphaseFilter    -- the rigid filter and D2's sticky laneDone, which the
//                              Phase-1 module does not have and the differential
//                              therefore cannot see.
//   psMeshMidphaseWorkIndex -- the work loop's index arithmetic at
//                              nbIterationsPerWarps 1, 2 and 3.  At 1 the factor is
//                              an identity and pins nothing: that is the hole P9.9's
//                              review found in the boundary-density fixture, and it
//                              is closed here in front rather than after.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"     // PxgParticleSystem, PxgParticleSimBuffer, PxgParticleContactWriter
#include "PxgContactManager.h"     // PxgContactManagerInput
#include "PxgConvexConvexShape.h"  // PxgShape
#include "PxgSimulationCoreDesc.h" // PxgRigidFilterPair
#include "PxgNpKernelIndices.h"    // PS_MIDPHASE_COLLISION_WAPRS_PER_BLOCK
#include "PxgParticleSystemCoreKernelIndices.h"
#include "PxsTransformCache.h"     // PxsCachedTransform
#include "PxNodeIndex.h"
#include "geometry/PxGeometry.h"
#include "GuBV32.h"                // Gu::BV32DataPacked

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

// psMeshMid.slang's PmMesh / PmParams / Bv32Node / PsContact, byte for byte.
struct PmMesh { F4 meshP, meshQ, scale, scaleRot, misc; U4 rigid; };
struct PmParams { uint32_t numParticles, maxContacts, flags, particleSystemId; };
struct Bv32NodeHost
{
    F4 mMin[32];
    F4 mMax[32];
    uint32_t mData[32];
    uint32_t mNbNodes, mDepth, pad0, pad1;
};
struct PsC { F4 normal_pen; uint32_t rLo, rHi, pLo, pHi; };

static_assert(sizeof(Bv32NodeHost) == 1168, "Gu::BV32DataPacked stride");
static_assert(sizeof(PsC) == 32, "PxgParticlePrimitiveContact stride");

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
static DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// The 136-byte engine ABI blob: {1,4,8,8,8,8,8,8,8,72}.  Each argument at its
// natural alignment, the by-value writer inlined field by field at 64.
struct PsMeshMidBlob
{
    uint32_t isTGS;                            // @0   CUDA bool + 3 pad bytes
    float toleranceLength;                     // @4   present, never read (M2)
    uint64_t cmInputs;                         // @8
    uint64_t transformCache;                   // @16
    uint64_t contactDistance;                  // @24
    uint64_t restDistances;                    // @32
    uint64_t gpuShapes;                        // @40
    uint64_t particleSystems;                  // @48
    uint64_t shapeToRigidRemapTable;           // @56
    uint64_t w_particleContacts;               // @64
    uint64_t w_numTotalContacts;               // @72
    uint64_t w_contactSortedByParticle;        // @80
    uint64_t w_tempContactByParticle;          // @88
    uint64_t w_contactIndexSortedByParticle;   // @96
    uint64_t w_contactByRigid;                 // @104
    uint64_t w_tempContactByRigid;             // @112
    uint64_t w_contactIndexSortedByRigid;      // @120
    uint32_t w_maxContacts;                    // @128
    uint32_t _pad132;                          // @132 -> 136
};
static_assert(sizeof(PsMeshMidBlob) == 136, "{1,4,8,8,8,8,8,8,8,72} blob");
static_assert(offsetof(PsMeshMidBlob, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(PsMeshMidBlob, w_particleContacts) == 64, "the writer starts at 64");
static_assert(offsetof(PsMeshMidBlob, w_maxContacts) == 128, "maxContacts @128");

struct PsMeshWorkBlob { uint64_t coverage; uint32_t numParticles; uint32_t pad; };
static_assert(sizeof(PsMeshWorkBlob) == 16, "the coverage entry's push block");

static const uint32_t MAXMESH = 6u;    // PxgParticleContactInfo::MaxStaticContactsPerMesh
static const uint32_t MAXPART = 12u;   // ...::MaxStaticContactsPerParticle
static const uint32_t POISON_U32 = 0xABABABABu;

// The engine's own constants, restated here only so the layout test can compare
// them; nothing else in the file hard-codes them.
static const uint32_t MESH_COLLISION_GRID = 16384u;
static const uint32_t PS_MIDPHASE_WARPS = 1u;

static PxgParticleSystem makeZeroPs()
{
    PxgParticleSystem ps;
    std::memset(&ps, 0, sizeof(ps));
    return ps;
}

// ---------------------------------------------------------------------------
// Small host vector helpers, only for BUILDING the scene.  Nothing here is a
// reference implementation -- the reference is the other GPU program.
// ---------------------------------------------------------------------------
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static float l1n(V3 v) { return std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z); }

static V3 hQuatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 hQuatRotateInv(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 hVertex2Shape(V3 v, V3 sc, F4 rot)
{
    V3 t = hQuatRotate(rot, v);
    t = { t.x * sc.x, t.y * sc.y, t.z * sc.z };
    return hQuatRotateInv(rot, t);
}

// ---- the BV32 builder, as in testPsMeshMid / testNpTetTriMid ----
struct AABB { float mn[3], mx[3]; };
static AABB mergeAabb(const AABB& a, const AABB& b)
{
    AABB r;
    for (int k = 0; k < 3; ++k)
    {
        r.mn[k] = std::min(a.mn[k], b.mn[k]);
        r.mx[k] = std::max(a.mx[k], b.mx[k]);
    }
    return r;
}
static uint32_t buildNode(std::vector<Bv32NodeHost>& nodes, const std::vector<AABB>& prims,
                          uint32_t begin, uint32_t end, std::mt19937& rng, int depth)
{
    const uint32_t nodeIdx = uint32_t(nodes.size());
    nodes.push_back(Bv32NodeHost());
    std::memset(&nodes.back(), 0, sizeof(Bv32NodeHost));
    const uint32_t count = end - begin;
    const uint32_t nbChunks = std::min<uint32_t>(2 + rng() % 5, count);
    std::vector<uint32_t> cuts = { begin, end };
    while (cuts.size() < nbChunks + 1)
        cuts.push_back(begin + 1 + rng() % (count - 1));
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    uint32_t slot = 0;
    for (size_t cc = 0; cc + 1 < cuts.size(); ++cc, ++slot)
    {
        const uint32_t b = cuts[cc], e = cuts[cc + 1];
        AABB bounds = prims[b];
        for (uint32_t j = b + 1; j < e; ++j) bounds = mergeAabb(bounds, prims[j]);
        if (e - b <= 12 || (depth >= 4 && e - b <= 32))
            nodes[nodeIdx].mData[slot] = (b << 7) | ((e - b) << 1) | 1;
        else
            nodes[nodeIdx].mData[slot] = buildNode(nodes, prims, b, e, rng, depth + 1) << 11;
        nodes[nodeIdx].mMin[slot] = { bounds.mn[0], bounds.mn[1], bounds.mn[2], 0.0f };
        nodes[nodeIdx].mMax[slot] = { bounds.mx[0], bounds.mx[1], bounds.mx[2], 0.0f };
    }
    nodes[nodeIdx].mNbNodes = slot;
    return nodeIdx;
}

// ---------------------------------------------------------------------------
// The shared scene.  One arena, built once, uploaded two different ways.
// ---------------------------------------------------------------------------
struct Scene
{
    static const uint32_t NR = 250u;   // triangles
    static const uint32_t NP = 97u;    // particles -- NOT a multiple of 32
    static const uint32_t SYS = 5u;    // particle system id, distinct from every index
    static const uint32_t CAP = 4096u; // writer capacity

    std::vector<F4> verts;             // NR * 3, VERTEX space
    std::vector<U4> tris;              // NR
    std::vector<Bv32NodeHost> nodes;
    std::vector<F4> curPos, predPos;   // NPPAD, world
    PmMesh mesh{};
    float cdTri = 0.0f, cdPs = 0.0f;   // the two halves of contactDist
    uint32_t nppad = 0;
};

static void buildScene(Scene& S)
{
    std::mt19937 rng(92003);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    S.verts.resize(Scene::NR * 3);
    S.tris.resize(Scene::NR);
    std::vector<AABB> bounds(Scene::NR);
    for (uint32_t t = 0; t < Scene::NR; ++t)
    {
        const V3 c0 = { 1.2f * uv(rng), 1.2f * uv(rng), 1.2f * uv(rng) };
        for (;;)
        {
            for (int k = 0; k < 3; ++k)
                S.verts[t * 3 + k] = { c0.x + 0.12f * uv(rng), c0.y + 0.12f * uv(rng),
                                       c0.z + 0.12f * uv(rng), 0 };
            const V3 e0 = sub(v3(S.verts[t * 3 + 1]), v3(S.verts[t * 3]));
            const V3 e1 = sub(v3(S.verts[t * 3 + 2]), v3(S.verts[t * 3]));
            const V3 cn = cross3(e0, e1);
            if (std::sqrt(dot3(cn, cn)) > l1n(e0) * l1n(e1) / 3.0f) break;
        }
        S.tris[t] = { t * 3, t * 3 + 1, t * 3 + 2, 0 };
        AABB bb = { { 1e30f, 1e30f, 1e30f }, { -1e30f, -1e30f, -1e30f } };
        for (int k = 0; k < 3; ++k)
        {
            const float* p = &S.verts[t * 3 + k].x;
            for (int j = 0; j < 3; ++j)
            {
                bb.mn[j] = std::min(bb.mn[j], p[j]);
                bb.mx[j] = std::max(bb.mx[j], p[j]);
            }
        }
        bounds[t] = bb;
    }
    // Morton order, so the tree has real spatial locality and the box test culls.
    {
        auto expand = [](uint32_t v) {
            v &= 0x3ff; v = (v | (v << 16)) & 0x30000ff; v = (v | (v << 8)) & 0x300f00f;
            v = (v | (v << 4)) & 0x30c30c3; v = (v | (v << 2)) & 0x9249249; return v;
        };
        std::vector<uint32_t> order(Scene::NR);
        std::vector<uint64_t> code(Scene::NR);
        for (uint32_t t = 0; t < Scene::NR; ++t)
        {
            order[t] = t;
            uint32_t mm[3];
            for (int k = 0; k < 3; ++k)
            {
                const float c = 0.5f * (bounds[t].mn[k] + bounds[t].mx[k]);
                mm[k] = uint32_t(std::min(1023.0f, std::max(0.0f, (c + 1.5f) / 3.0f * 1023.0f)));
            }
            code[t] = (uint64_t(expand(mm[0])) << 2) | (uint64_t(expand(mm[1])) << 1) | expand(mm[2]);
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });
        std::vector<U4> t2(Scene::NR);
        std::vector<AABB> b2(Scene::NR);
        for (uint32_t t = 0; t < Scene::NR; ++t) { t2[t] = S.tris[order[t]]; b2[t] = bounds[order[t]]; }
        S.tris.swap(t2);
        bounds.swap(b2);
    }
    buildNode(S.nodes, bounds, 0, Scene::NR, rng, 0);

    // Mesh pose and scale.  The scale determinant is NEGATIVE, so MeshScaling's
    // flipNormal fires and the V1/V2 swap is under test in both programs.
    S.mesh.meshP = { 0.3f, -0.2f, 0.1f, 0 };
    {
        F4 q = { 0.2f, -0.1f, 0.3f, 1.0f };
        const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        S.mesh.meshQ = { q.x / m, q.y / m, q.z / m, q.w / m };
    }
    // The two contact distances are added ON THE HOST in float and the sum handed
    // to the Phase-1 module, so `cd[tri] + cd[ps]` and `mesh.scale.w` are the same
    // BITS rather than merely the same number.
    S.cdTri = 0.13f;
    S.cdPs = 0.09f;
    const float cd = S.cdTri + S.cdPs;
    S.mesh.scale = { 1.1f, -0.9f, 1.05f, cd };
    S.mesh.scaleRot = { 0, 0, 0, 1 };
    S.mesh.misc = { 0.03f, 0, 0, 0 };            // restDist
    S.mesh.rigid = { PX_INVALID_NODE, 0u, 0, 0 };  // STATIC by default

    // The densest spot, so some particles overflow the six-slot list and drive the
    // serialized worst-contact replacement (M6).
    uint32_t denseTri = 0;
    {
        size_t bestCnt = 0;
        for (uint32_t t = 0; t < Scene::NR; ++t)
        {
            V3 tc = { 0, 0, 0 };
            for (int k = 0; k < 3; ++k) tc = add(tc, v3(S.verts[S.tris[t].x + k]));
            tc = muls(tc, 1.0f / 3.0f);
            size_t cnt = 0;
            for (uint32_t u = 0; u < Scene::NR; ++u)
            {
                V3 uc = { 0, 0, 0 };
                for (int k = 0; k < 3; ++k) uc = add(uc, v3(S.verts[S.tris[u].x + k]));
                uc = muls(uc, 1.0f / 3.0f);
                const V3 d = sub(uc, tc);
                if (dot3(d, d) < 0.35f * 0.35f) ++cnt;
            }
            if (cnt > bestCnt) { bestCnt = cnt; denseTri = t; }
        }
    }

    S.nppad = ((Scene::NP + 31u) / 32u) * 32u;
    S.curPos.assign(S.nppad, { 100, 100, 100, 1 });
    S.predPos.assign(S.nppad, { 100, 100, 100, 1 });
    for (uint32_t p = 0; p < Scene::NP; ++p)
    {
        const uint32_t t = (p % 5 == 2) ? denseTri : (p * 29) % Scene::NR;
        V3 tc = { 0, 0, 0 };
        for (int k = 0; k < 3; ++k) tc = add(tc, v3(S.verts[S.tris[t].x + k]));
        tc = muls(tc, 1.0f / 3.0f);
        const V3 sp = hVertex2Shape(tc, v3(S.mesh.scale), S.mesh.scaleRot);
        const V3 wp = add(hQuatRotate(S.mesh.meshQ, sp), v3(S.mesh.meshP));
        V3 dir = { uv(rng), uv(rng), uv(rng) };
        const float m = std::sqrt(std::max(dot3(dir, dir), 1e-3f));
        dir = muls(dir, 1.0f / m);
        const float off = (p % 5 == 2) ? 0.02f + 0.13f * u01(rng)
                          : (p % 4 == 0) ? 0.9f * u01(rng) : 0.25f * u01(rng);
        const V3 c = add(wp, muls(dir, off));
        S.curPos[p] = { c.x, c.y, c.z, 1 };
        S.predPos[p] = { c.x + 0.15f * uv(rng), c.y + 0.15f * uv(rng), c.z + 0.15f * uv(rng), 1 };
    }
}

// ---------------------------------------------------------------------------
// The BDA side: one struct holding everything the descriptor chain needs, so a
// dispatch is a couple of lines and every buffer is freed in one place.
// ---------------------------------------------------------------------------
struct BdaRig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bBlob, bCin, bShapes, bTc, bCd, bRd, bRemap, bDesc;
    DevBuf bCur, bPred, bS2U, bRunsum, bSimBuf;
    DevBuf bInfos, bNodeIdx, bCounts;
    DevBuf bContacts, bCount, bSbP, bTbP, bIbP, bBr, bTbr, bIbr;
    PsMeshMidBlob blob{};

    template <typename T>
    DevBuf mk(PxgSlangComputeDevice* dev, const std::vector<T>& v)
    {
        DevBuf b = makeDev(dev, v);
        owned.push_back(b.h);
        return b;
    }
    void destroy(PxgSlangComputeDevice* dev)
    {
        for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
        owned.clear();
    }
};

// Distinct, non-zero, mutually different indices throughout: the transform-cache
// slots, the shape slots and the contact-manager slot must never coincide, so a
// kernel that reads the wrong one lands on a different number rather than on the
// right answer by luck.  This is the standing fixture rule from P8.73-P8.80.
static const uint32_t TRI_SHAPE = 2u, PS_SHAPE = 4u;
static const uint32_t TRI_CACHE = 3u, PS_CACHE = 6u;
static const uint32_t NB_SHAPES = 7u, NB_CACHE = 9u, NB_CM = 3u;
static const uint32_t CM_IDX = 0u;

struct BdaOptions
{
    bool triFirst = true;            // which half of the pair is the triangle mesh
    bool staticRigid = true;
    bool enableCcd = true;
    bool isTgs = true;
    uint32_t nbFilterPairs = 0u;
    std::vector<PxgRigidFilterPair> filterPairs;
};

static void buildBda(PxgSlangComputeDevice* dev, const Scene& S, const BdaOptions& opt, BdaRig& R)
{
    // ---- the cooked mesh blob: header, nodes, verts, indices ----
    // header = uint4{nbVerts, nbTris, nbBv32TreeNodes, nbPackedNodes}.  M4: only .w
    // sizes the node array, and .z is deliberately WRONG here so a port that uses it
    // walks off the nodes.
    const uint32_t nbVerts = uint32_t(S.verts.size());
    const uint32_t nbTris = uint32_t(S.tris.size());
    const uint32_t nbPacked = uint32_t(S.nodes.size());
    std::vector<uint8_t> blob;
    {
        const uint32_t header[4] = { nbVerts, nbTris, 7u /* the .z trap */, nbPacked };
        blob.resize(16);
        std::memcpy(blob.data(), header, 16);
        const size_t nodeBytes = size_t(nbPacked) * sizeof(Bv32NodeHost);
        blob.resize(16 + nodeBytes);
        std::memcpy(blob.data() + 16, S.nodes.data(), nodeBytes);
        const size_t vertBytes = size_t(nbVerts) * 16;
        blob.resize(blob.size() + vertBytes);
        std::memcpy(blob.data() + 16 + nodeBytes, S.verts.data(), vertBytes);
        const size_t idxBytes = size_t(nbTris) * 16;
        blob.resize(blob.size() + idxBytes);
        std::memcpy(blob.data() + 16 + nodeBytes + vertBytes, S.tris.data(), idxBytes);
    }
    R.bBlob = R.mk(dev, blob);

    // ---- particle system ----
    R.bCur = R.mk(dev, S.curPos);
    R.bPred = R.mk(dev, S.predPos);

    // mSortedToUnsortedMapping: a real permutation-like map, distinct from identity
    // so a filter query built from the wrong array is visible.
    std::vector<uint32_t> s2u(S.nppad);
    for (uint32_t i = 0; i < S.nppad; ++i) s2u[i] = (i * 37u + 11u) % Scene::NP;
    R.bS2U = R.mk(dev, s2u);

    std::vector<uint32_t> runsum = { 0u };
    R.bRunsum = R.mk(dev, runsum);

    DevBuf bPairs;
    if (!opt.filterPairs.empty()) bPairs = R.mk(dev, opt.filterPairs);
    std::vector<PxgParticleSimBuffer> simBuffers(1);
    std::memset(simBuffers.data(), 0, sizeof(PxgParticleSimBuffer));
    simBuffers[0].mNumFilterPairs = opt.nbFilterPairs;
    simBuffers[0].mFilterPairs =
        reinterpret_cast<PxParticleRigidFilterPair*>(size_t(bPairs.addr));
    R.bSimBuf = R.mk(dev, simBuffers);

    // The one-way outputs.  Poisoned everywhere, including the pad rows past the
    // live particle count, so an extra write fails as loudly as a wrong one.
    std::vector<F4> infos(size_t(S.nppad) * MAXPART, { -9, -9, -9, -9 });
    std::vector<uint32_t> nodeIdx(size_t(S.nppad) * MAXPART * 2u, 0xCCCCCCCCu);
    std::vector<uint32_t> counts(S.nppad, 0u);
    R.bInfos = R.mk(dev, infos);
    R.bNodeIdx = R.mk(dev, nodeIdx);
    R.bCounts = R.mk(dev, counts);

    // The particle-system pool has SYS+2 entries so index SYS is genuinely indexed
    // and the neighbours are poison.
    std::vector<PxgParticleSystem> descArr(Scene::SYS + 2u);
    std::memset(descArr.data(), 0xEE, descArr.size() * sizeof(PxgParticleSystem));
    PxgParticleSystem ps = makeZeroPs();
    ps.mCommonData.mNumParticles = Scene::NP;
    ps.mCommonData.mMaxParticles = S.nppad;
    ps.mCommonData.mNumParticleBuffers = 1u;
    ps.mCommonData.mMaxDiffuseParticles = 0u;   // M1: the diffuse pass returns early
    ps.mData.mFlags = opt.enableCcd ? PxU32(PxParticleFlag::eENABLE_SPECULATIVE_CCD) : 0u;
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(R.bCur.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bPred.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(R.bS2U.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(R.bRunsum.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(R.bSimBuf.addr));
    ps.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.bInfos.addr));
    ps.mOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.bNodeIdx.addr));
    ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.bCounts.addr));
    descArr[Scene::SYS] = ps;
    R.bDesc = R.mk(dev, descArr);

    // ---- narrowphase descriptors ----
    std::vector<PxgContactManagerInput> cmInputs(NB_CM);
    std::memset(cmInputs.data(), 0xEE, cmInputs.size() * sizeof(PxgContactManagerInput));
    if (opt.triFirst)
    {
        cmInputs[CM_IDX].shapeRef0 = TRI_SHAPE;
        cmInputs[CM_IDX].transformCacheRef0 = TRI_CACHE;
        cmInputs[CM_IDX].shapeRef1 = PS_SHAPE;
        cmInputs[CM_IDX].transformCacheRef1 = PS_CACHE;
    }
    else
    {
        cmInputs[CM_IDX].shapeRef0 = PS_SHAPE;
        cmInputs[CM_IDX].transformCacheRef0 = PS_CACHE;
        cmInputs[CM_IDX].shapeRef1 = TRI_SHAPE;
        cmInputs[CM_IDX].transformCacheRef1 = TRI_CACHE;
    }
    R.bCin = R.mk(dev, cmInputs);

    std::vector<PxgShape> shapes(NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    // The decoys are also triangle meshes, with a DIFFERENT scale and a null blob:
    // a kernel that picks the wrong shape slot gets a shape whose type check passes
    // and whose geometry pointer is zero, i.e. a fault rather than a near-miss.
    for (uint32_t i = 0; i < NB_SHAPES; ++i)
    {
        shapes[i].type = PxU32(PxGeometryType::eTRIANGLEMESH);
        shapes[i].scale = PxMeshScale(PxVec3(2.5f, 2.5f, 2.5f), PxQuat(PxIdentity));
    }
    shapes[TRI_SHAPE].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[TRI_SHAPE].scale = PxMeshScale(PxVec3(S.mesh.scale.x, S.mesh.scale.y, S.mesh.scale.z),
                                          PxQuat(S.mesh.scaleRot.x, S.mesh.scaleRot.y,
                                                 S.mesh.scaleRot.z, S.mesh.scaleRot.w));
    shapes[TRI_SHAPE].hullOrMeshPtr = size_t(R.bBlob.addr);
    shapes[PS_SHAPE].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[PS_SHAPE].particleOrSoftbodyId = Scene::SYS;
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> tcache(NB_CACHE);
    std::memset(tcache.data(), 0xEE, tcache.size() * sizeof(PxsCachedTransform));
    for (uint32_t i = 0; i < NB_CACHE; ++i)
        tcache[i].transform = PxTransform(PxVec3(-7.0f - float(i), 3.0f, 2.0f), PxQuat(PxIdentity));
    tcache[TRI_CACHE].transform = PxTransform(PxVec3(S.mesh.meshP.x, S.mesh.meshP.y, S.mesh.meshP.z),
                                              PxQuat(S.mesh.meshQ.x, S.mesh.meshQ.y,
                                                     S.mesh.meshQ.z, S.mesh.meshQ.w));
    R.bTc = R.mk(dev, tcache);

    // Every OTHER contact distance is distinct and none of them sums to the right
    // total with anything else, so reading the wrong cache slot changes the answer.
    std::vector<float> contactDist(NB_CACHE);
    for (uint32_t i = 0; i < NB_CACHE; ++i) contactDist[i] = 1.7f + 0.31f * float(i);
    contactDist[TRI_CACHE] = S.cdTri;
    contactDist[PS_CACHE] = S.cdPs;
    R.bCd = R.mk(dev, contactDist);

    std::vector<float> restDist(NB_CM);
    for (uint32_t i = 0; i < NB_CM; ++i) restDist[i] = -2.5f - 0.7f * float(i);
    restDist[CM_IDX] = S.mesh.misc.x;
    R.bRd = R.mk(dev, restDist);

    // shapeToRigidRemapTable is indexed by the TRIANGLE MESH's cache ref.
    std::vector<uint64_t> remap(NB_CACHE, 0ull);
    for (uint32_t i = 0; i < NB_CACHE; ++i)
        remap[i] = (uint64_t(0x5150u + i) << 32) | uint64_t(0x900u + i);
    {
        PxNodeIndex ni = opt.staticRigid ? PxNodeIndex(PX_INVALID_NODE, 0u)
                                         : PxNodeIndex(0x77u, 4u);
        remap[TRI_CACHE] = ni.getInd();
    }
    R.bRemap = R.mk(dev, remap);

    // ---- writer outputs ----
    std::vector<PsC> contacts(Scene::CAP);
    std::memset(contacts.data(), 0xAB, contacts.size() * sizeof(PsC));
    std::vector<uint32_t> count = { 0u };
    std::vector<uint64_t> sortedByParticle(Scene::CAP, 0xABABABABABABABABull);
    std::vector<uint32_t> tempByParticle(Scene::CAP, POISON_U32);
    std::vector<uint32_t> idxByParticle(Scene::CAP, POISON_U32);
    std::vector<uint64_t> byRigid(Scene::CAP, 0xABABABABABABABABull);
    std::vector<uint32_t> tempByRigid(Scene::CAP, POISON_U32);
    std::vector<uint32_t> idxByRigid(Scene::CAP, POISON_U32);
    R.bContacts = R.mk(dev, contacts);
    R.bCount = R.mk(dev, count);
    R.bSbP = R.mk(dev, sortedByParticle);
    R.bTbP = R.mk(dev, tempByParticle);
    R.bIbP = R.mk(dev, idxByParticle);
    R.bBr = R.mk(dev, byRigid);
    R.bTbr = R.mk(dev, tempByRigid);
    R.bIbr = R.mk(dev, idxByRigid);

    R.blob.isTGS = opt.isTgs ? 1u : 0u;
    // M2: never read.  A distinctive value, so a port that DOES read it is wrong
    // by a large margin rather than by a plausible one.
    R.blob.toleranceLength = -1234.5f;
    R.blob.cmInputs = R.bCin.addr;
    R.blob.transformCache = R.bTc.addr;
    R.blob.contactDistance = R.bCd.addr;
    R.blob.restDistances = R.bRd.addr;
    R.blob.gpuShapes = R.bShapes.addr;
    R.blob.particleSystems = R.bDesc.addr;
    R.blob.shapeToRigidRemapTable = R.bRemap.addr;
    R.blob.w_particleContacts = R.bContacts.addr;
    R.blob.w_numTotalContacts = R.bCount.addr;
    R.blob.w_contactSortedByParticle = R.bSbP.addr;
    R.blob.w_tempContactByParticle = R.bTbP.addr;
    R.blob.w_contactIndexSortedByParticle = R.bIbP.addr;
    R.blob.w_contactByRigid = R.bBr.addr;
    R.blob.w_tempContactByRigid = R.bTbr.addr;
    R.blob.w_contactIndexSortedByRigid = R.bIbr.addr;
    R.blob.w_maxContacts = Scene::CAP;
}

// ---------------------------------------------------------------------------
// The Phase-1 side.
// ---------------------------------------------------------------------------
struct RefRig
{
    std::vector<PxgBufferHandle> owned;
    PxgBufferHandle bM = 0, bN = 0, bT = 0, bV = 0, bCur = 0, bPred = 0;
    PxgBufferHandle bI = 0, bNi = 0, bC = 0;
    PxgBufferHandle bW = 0, bBR = 0, bTBR = 0, bIBR = 0, bSP = 0, bTBP = 0, bIBP = 0, bWc = 0;
    PmParams prm{};

    template <typename T>
    PxgBufferHandle mk(PxgSlangComputeDevice* dev, const std::vector<T>& v)
    {
        PxgBufferHandle h = createBufferFrom(dev, v);
        owned.push_back(h);
        return h;
    }
    void destroy(PxgSlangComputeDevice* dev)
    {
        for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
        owned.clear();
    }
    std::vector<PxgBufferHandle> bindings() const
    {
        return { bM, bN, bT, bV, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc };
    }
};

static void buildRef(PxgSlangComputeDevice* dev, const Scene& S, const BdaOptions& opt, RefRig& R)
{
    PmMesh mesh = S.mesh;
    if (!opt.staticRigid)
    {
        const PxNodeIndex ni(0x77u, 4u);
        mesh.rigid = { uint32_t(ni.getInd() & 0xFFFFFFFFull), uint32_t(ni.getInd() >> 32), 0, 0 };
    }
    else
    {
        const PxNodeIndex ni(PX_INVALID_NODE, 0u);
        mesh.rigid = { uint32_t(ni.getInd() & 0xFFFFFFFFull), uint32_t(ni.getInd() >> 32), 0, 0 };
    }
    std::vector<PmMesh> meshV = { mesh };

    R.bM = R.mk(dev, meshV);
    R.bN = R.mk(dev, S.nodes);
    R.bT = R.mk(dev, S.tris);
    R.bV = R.mk(dev, S.verts);
    R.bCur = R.mk(dev, S.curPos);
    R.bPred = R.mk(dev, S.predPos);

    std::vector<F4> infos(size_t(S.nppad) * MAXPART, { -9, -9, -9, -9 });
    std::vector<uint32_t> nodeIdx(size_t(S.nppad) * MAXPART * 2u, 0xCCCCCCCCu);
    std::vector<uint32_t> counts(S.nppad, 0u);
    R.bI = R.mk(dev, infos);
    R.bNi = R.mk(dev, nodeIdx);
    R.bC = R.mk(dev, counts);

    std::vector<PsC> contacts(Scene::CAP);
    std::memset(contacts.data(), 0xAB, contacts.size() * sizeof(PsC));
    std::vector<uint32_t> count = { 0u };
    std::vector<uint64_t> u64Poison(Scene::CAP, 0xABABABABABABABABull);
    std::vector<uint32_t> u32Poison(Scene::CAP, POISON_U32);
    R.bW = R.mk(dev, contacts);
    R.bBR = R.mk(dev, u64Poison);
    R.bTBR = R.mk(dev, u32Poison);
    R.bIBR = R.mk(dev, u32Poison);
    R.bSP = R.mk(dev, u64Poison);
    R.bTBP = R.mk(dev, u32Poison);
    R.bIBP = R.mk(dev, u32Poison);
    R.bWc = R.mk(dev, count);

    R.prm.numParticles = Scene::NP;
    R.prm.maxContacts = Scene::CAP;
    R.prm.flags = (opt.enableCcd ? 1u : 0u) | (opt.isTgs ? 2u : 0u) | (opt.staticRigid ? 8u : 0u);
    R.prm.particleSystemId = Scene::SYS;
}

static const std::vector<uint32_t>& refStrides()
{
    static const std::vector<uint32_t> s = { uint32_t(sizeof(PmMesh)), uint32_t(sizeof(Bv32NodeHost)),
                                             16, 16, 16, 16, 16, 8, 4, uint32_t(sizeof(PsC)),
                                             8, 4, 4, 8, 4, 4, 4 };
    return s;
}

// Group writer contacts by particle.  The global start indices race between
// groups, so the ARRAY order is not deterministic -- but a particle's own
// contacts are written sequentially by one lane from one base, so grouping by the
// particle index and sorting by array position recovers a canonical order that
// both programs must agree on exactly.
static std::vector<std::vector<PsC>> groupByParticle(const std::vector<PsC>& raw, uint32_t n)
{
    std::vector<std::pair<uint32_t, std::vector<PsC>>> tmp;
    std::vector<std::vector<PsC>> out(Scene::NP + 1u);
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t p = raw[i].pLo;
        out[p <= Scene::NP ? p : Scene::NP].push_back(raw[i]);
    }
    (void)tmp;
    return out;
}

static bool sameContact(const PsC& a, const PsC& b)
{
    return std::memcmp(&a, &b, sizeof(PsC)) == 0;
}

} // namespace

// ===========================================================================
PXG_TEST(psMeshMidphaseLayout)
{
    // The ABI blob.
    static_assert(sizeof(PxgParticleContactWriter) == 72, "the by-value writer is 72 bytes");
    PXG_CHECK_MSG(sizeof(PxgParticleContactWriter) == 72,
                  "PxgParticleContactWriter must be 72 bytes (got %zu)",
                  sizeof(PxgParticleContactWriter));
    PXG_CHECK(offsetof(PxgParticleContactWriter, particleContacts) == 0);
    PXG_CHECK(offsetof(PxgParticleContactWriter, numTotalContacts) == 8);
    PXG_CHECK(offsetof(PxgParticleContactWriter, contactSortedByParticle) == 16);
    PXG_CHECK(offsetof(PxgParticleContactWriter, tempContactByParticle) == 24);
    PXG_CHECK(offsetof(PxgParticleContactWriter, contactIndexSortedByParticle) == 32);
    PXG_CHECK(offsetof(PxgParticleContactWriter, contactByRigid) == 40);
    PXG_CHECK(offsetof(PxgParticleContactWriter, tempContactByRigid) == 48);
    PXG_CHECK(offsetof(PxgParticleContactWriter, contactIndexSortedByRigid) == 56);
    PXG_CHECK_MSG(offsetof(PxgParticleContactWriter, maxContacts) == 64,
                  "maxContacts must sit at 64 inside the writer (got %zu)",
                  offsetof(PxgParticleContactWriter, maxContacts));

    // The pair-loading surface.
    PXG_CHECK(sizeof(PxgContactManagerInput) == 16);
    PXG_CHECK(offsetof(PxgContactManagerInput, shapeRef0) == 0);
    PXG_CHECK(offsetof(PxgContactManagerInput, shapeRef1) == 4);
    PXG_CHECK(offsetof(PxgContactManagerInput, transformCacheRef0) == 8);
    PXG_CHECK(offsetof(PxgContactManagerInput, transformCacheRef1) == 12);
    PXG_CHECK_MSG(sizeof(PxgShape) == 48, "PxgShape stride (got %zu)", sizeof(PxgShape));
    PXG_CHECK(offsetof(PxgShape, hullOrMeshPtr) == 32);
    PXG_CHECK(offsetof(PxgShape, type) == 40);
    PXG_CHECK(offsetof(PxgShape, particleOrSoftbodyId) == 44);
    PXG_CHECK_MSG(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform stride (got %zu)",
                  sizeof(PxsCachedTransform));
    PXG_CHECK_MSG(offsetof(PxsCachedTransform, transform) == 0,
                  "the cached transform starts at 0 (quat first, then the position at 16)");

    // The two geometry types the shader keys on.
    PXG_CHECK_MSG(PxU32(PxGeometryType::eTRIANGLEMESH) == 8u,
                  "eTRIANGLEMESH == 8 (got %u)", unsigned(PxGeometryType::eTRIANGLEMESH));
    PXG_CHECK_MSG(PxU32(PxGeometryType::ePARTICLESYSTEM) == 6u,
                  "ePARTICLESYSTEM == 6 (got %u)", unsigned(PxGeometryType::ePARTICLESYSTEM));

    // The two contact caps (M6).
    PXG_CHECK_MSG(PxgParticleContactInfo::MaxStaticContactsPerMesh == MAXMESH,
                  "MaxStaticContactsPerMesh == 6 (got %u)",
                  unsigned(PxgParticleContactInfo::MaxStaticContactsPerMesh));
    PXG_CHECK_MSG(PxgParticleContactInfo::MaxStaticContactsPerParticle == MAXPART,
                  "MaxStaticContactsPerParticle == 12 (got %u)",
                  unsigned(PxgParticleContactInfo::MaxStaticContactsPerParticle));

    // The launch shape, which the shader spells as compile-time constants because
    // the grid is a fixed host constant rather than a dispatch parameter.
    PXG_CHECK_MSG(PxgParticleSystemKernelGridDim::PS_MESH_COLLISION == MESH_COLLISION_GRID,
                  "PS_MESH_COLLISION grid == 16384 (got %u)",
                  unsigned(PxgParticleSystemKernelGridDim::PS_MESH_COLLISION));
    PXG_CHECK_MSG(PS_MIDPHASE_COLLISION_WAPRS_PER_BLOCK == PS_MIDPHASE_WARPS,
                  "PS_MIDPHASE_COLLISION_WAPRS_PER_BLOCK == 1 (got %u)",
                  unsigned(PS_MIDPHASE_COLLISION_WAPRS_PER_BLOCK));

    // The BV32 node the traversal walks.
    PXG_CHECK_MSG(sizeof(Gu::BV32DataPacked) == sizeof(Bv32NodeHost),
                  "Gu::BV32DataPacked is %zu bytes, the fixture mirror is %zu",
                  sizeof(Gu::BV32DataPacked), sizeof(Bv32NodeHost));
    PXG_CHECK(offsetof(Gu::BV32DataPacked, mMin) == 0);
    PXG_CHECK(offsetof(Gu::BV32DataPacked, mMax) == 512);
    PXG_CHECK(offsetof(Gu::BV32DataPacked, mData) == 1024);
    PXG_CHECK(offsetof(Gu::BV32DataPacked, mNbNodes) == 1152);

    // The CCD flag the shader tests against mData.mFlags.
    // 1 << 3, and the neighbouring bit 1 << 2 is eFULL_DIFFUSE_ADVECTION -- a
    // plausible off-by-one that this file caught in the shader's first draft.
    PXG_CHECK_MSG(PxU32(PxParticleFlag::eENABLE_SPECULATIVE_CCD) == 8u,
                  "eENABLE_SPECULATIVE_CCD == 1<<3 (got %u)",
                  unsigned(PxParticleFlag::eENABLE_SPECULATIVE_CCD));
    PXG_CHECK_MSG(PxU32(PxParticleFlag::eFULL_DIFFUSE_ADVECTION) == 4u,
                  "1<<2 is eFULL_DIFFUSE_ADVECTION, which is what makes 1<<2 a trap");

    // The filter list the port walks (a third copy of find(), see the shader header).
    PXG_CHECK_MSG(sizeof(PxgRigidFilterPair) == 32, "PxgRigidFilterPair stride (got %zu)",
                  sizeof(PxgRigidFilterPair));
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mFilterPairs) == 40);
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mNumFilterPairs) == 64);
    PXG_CHECK_MSG(sizeof(PxgParticleSimBuffer) == 88,
                  "PxgParticleSimBuffer stride == 88 (got %zu)", sizeof(PxgParticleSimBuffer));

    // The descriptor fields the glue reads.
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedOriginPos_InvMass) == 392);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPositions_InvMass) == 24);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedToUnsortedMapping) == 56);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactInfos) == 200);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayNodeIndex) == 216);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactCount) == 232);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseOneWayContactInfos) == 592);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseOneWayNodeIndex) == 608);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseOneWayContactCount) == 616);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) == 552);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) == 560);
    PXG_CHECK(offsetof(PxgParticleSystem, mNumDiffuseParticles) == 488);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleSimBuffers) == 400);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mNumParticles) == 116);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mMaxDiffuseParticles) == 128);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mNumParticleBuffers) == 132);
    PXG_CHECK(offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mFlags) == 312);

    // PxNodeIndex's static test, which the shader spells as `n.x == 0xFFFFFFFF`.
    {
        const PxNodeIndex stat(PX_INVALID_NODE, 0u);
        const PxNodeIndex dyn(0x77u, 4u);
        PXG_CHECK(stat.isStaticBody());
        PXG_CHECK(!dyn.isStaticBody());
        PXG_CHECK_MSG(uint32_t(stat.getInd() & 0xFFFFFFFFull) == PX_INVALID_NODE,
                      "the LOW word is the one isStaticBody tests");
    }
}

// ===========================================================================
// The differential.
// ===========================================================================
PXG_TEST(psMeshMidphase)
{
    PxgSlangComputeDevice* dev = testDevice();

    Scene S;
    buildScene(S);

    const uint32_t nbWarps = (Scene::NP + 31u) / 32u;
    PXG_CHECK_MSG(Scene::NP % 32u != 0u, "the last warp must be partial");

    // ---------------------------------------------------------------
    // Shape 1: static rigid, one dispatch -> the one-way path only.
    // ---------------------------------------------------------------
    size_t withContacts = 0, overSix = 0;
    {
        BdaOptions opt;   // static, CCD on, TGS, no filters
        BdaRig A;
        RefRig B;
        buildBda(dev, S, opt, A);
        buildRef(dev, S, opt, B);

        {
            Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
            k.dispatch(nbWarps, {}, A.blob, 1u, 1u);
        }
        {
            Kernel k("psMeshMid.spv", "ctPsMeshMid", refStrides(), sizeof(PmParams));
            k.dispatch(nbWarps, B.bindings(), B.prm);
        }
        dev->flush();

        const auto aInfos = downloadBuffer<F4>(dev, A.bInfos.h, size_t(S.nppad) * MAXPART);
        const auto bInfos = downloadBuffer<F4>(dev, B.bI, size_t(S.nppad) * MAXPART);
        const auto aNi = downloadBuffer<uint32_t>(dev, A.bNodeIdx.h, size_t(S.nppad) * MAXPART * 2u);
        const auto bNi = downloadBuffer<uint32_t>(dev, B.bNi, size_t(S.nppad) * MAXPART * 2u);
        const auto aCnt = downloadBuffer<uint32_t>(dev, A.bCounts.h, S.nppad);
        const auto bCnt = downloadBuffer<uint32_t>(dev, B.bC, S.nppad);

        PXG_CHECK_MSG(std::memcmp(aCnt.data(), bCnt.data(), aCnt.size() * 4) == 0,
                      "static pass: the one-way COUNTS differ between the BDA kernel and psMeshMid");
        PXG_CHECK_MSG(std::memcmp(aInfos.data(), bInfos.data(), aInfos.size() * sizeof(F4)) == 0,
                      "static pass: the one-way CONTACT INFOS differ");
        PXG_CHECK_MSG(std::memcmp(aNi.data(), bNi.data(), aNi.size() * 4) == 0,
                      "static pass: the one-way NODE INDICES differ");

        // The scene must actually reach the traps, or the agreement is agreement
        // about nothing.  Counted rather than assumed.
        for (uint32_t p = 0; p < Scene::NP; ++p)
        {
            if (aCnt[p] != 0u) ++withContacts;
            if (aCnt[p] == MAXMESH) ++overSix;
        }
        PXG_CHECK_MSG(withContacts >= 20,
                      "only %zu particles found any contact -- the scene is too thin", withContacts);
        // SATURATION, not overflow: M6 makes the owning lane's count STOP at
        // MaxStaticContactsPerMesh, so a count above six is not reachable and asking
        // for one is asking for the impossible (this assertion did read `> MAXMESH`
        // in the first draft and failed for exactly that reason).  Saturated
        // particles are the necessary condition for the replacement pass; what
        // PROVES the pass is live is the sweep, where a mutant that deletes it must
        // die against psMeshMid, which keeps it -- and psMeshMid_traverser's own
        // host oracle counts the overflow rounds on this same scene construction.
        PXG_CHECK_MSG(overSix > 0,
                      "no particle saturated the six-slot list, so M6's replacement pass "
                      "cannot have run and the sweep cannot kill a mutant that drops it");

        // The pad rows past the live particles must still be poison in BOTH.
        size_t touchedPad = 0;
        for (uint32_t p = Scene::NP; p < S.nppad; ++p)
            if (aCnt[p] != 0u) ++touchedPad;
        PXG_CHECK_MSG(touchedPad == 0, "%zu pad particles were written", touchedPad);

        // Nothing should have reached the writer yet: with a fresh count array the
        // twelve-contact cap cannot bind on a first pass.
        const uint32_t aWc = downloadBuffer<uint32_t>(dev, A.bCount.h, 1)[0];
        const uint32_t bWc = downloadBuffer<uint32_t>(dev, B.bWc, 1)[0];
        PXG_CHECK_MSG(aWc == 0u && bWc == 0u,
                      "static first pass wrote %u/%u writer contacts; the cap cannot bind yet",
                      aWc, bWc);

        // -----------------------------------------------------------
        // Shape 2: the SAME buffers a second and third time.  Counts accumulate,
        // the twelve cap binds, and the overflow goes through the writer.
        // -----------------------------------------------------------
        {
            Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
            k.dispatch(nbWarps, {}, A.blob, 1u, 1u);
            k.dispatch(nbWarps, {}, A.blob, 1u, 1u);
        }
        {
            Kernel k("psMeshMid.spv", "ctPsMeshMid", refStrides(), sizeof(PmParams));
            k.dispatch(nbWarps, B.bindings(), B.prm);
            k.dispatch(nbWarps, B.bindings(), B.prm);
        }
        dev->flush();

        const auto aInfos2 = downloadBuffer<F4>(dev, A.bInfos.h, size_t(S.nppad) * MAXPART);
        const auto bInfos2 = downloadBuffer<F4>(dev, B.bI, size_t(S.nppad) * MAXPART);
        const auto aCnt2 = downloadBuffer<uint32_t>(dev, A.bCounts.h, S.nppad);
        const auto bCnt2 = downloadBuffer<uint32_t>(dev, B.bC, S.nppad);
        PXG_CHECK_MSG(std::memcmp(aCnt2.data(), bCnt2.data(), aCnt2.size() * 4) == 0,
                      "accumulation: the one-way COUNTS differ");
        PXG_CHECK_MSG(std::memcmp(aInfos2.data(), bInfos2.data(), aInfos2.size() * sizeof(F4)) == 0,
                      "accumulation: the one-way CONTACT INFOS differ");

        size_t capped = 0;
        for (uint32_t p = 0; p < Scene::NP; ++p)
            if (aCnt2[p] > MAXPART) ++capped;
        PXG_CHECK_MSG(capped > 0,
                      "three passes did not push any particle past MaxStaticContactsPerParticle, "
                      "so the twelve-contact cap and the writer overflow are untested");

        const uint32_t aWc2 = downloadBuffer<uint32_t>(dev, A.bCount.h, 1)[0];
        const uint32_t bWc2 = downloadBuffer<uint32_t>(dev, B.bWc, 1)[0];
        PXG_CHECK_MSG(aWc2 == bWc2, "overflow: writer counters differ (%u vs %u)", aWc2, bWc2);
        PXG_CHECK_MSG(aWc2 > 0u, "the overflow never reached the writer");
        PXG_CHECK_MSG(aWc2 <= Scene::CAP, "writer overflowed its capacity (%u)", aWc2);

        {
            const auto aCt = downloadBuffer<PsC>(dev, A.bContacts.h, Scene::CAP);
            const auto bCt = downloadBuffer<PsC>(dev, B.bW, Scene::CAP);
            const auto ga = groupByParticle(aCt, aWc2);
            const auto gb = groupByParticle(bCt, bWc2);
            size_t mism = 0;
            for (uint32_t p = 0; p <= Scene::NP; ++p)
            {
                if (ga[p].size() != gb[p].size()) { ++mism; continue; }
                for (size_t i = 0; i < ga[p].size(); ++i)
                    if (!sameContact(ga[p][i], gb[p][i])) { ++mism; break; }
            }
            PXG_CHECK_MSG(mism == 0, "overflow: %zu particles' writer contacts differ", mism);
            PXG_CHECK_MSG(ga[Scene::NP].empty(),
                          "the writer emitted a contact for an out-of-range particle index");
        }

        A.destroy(dev);
        B.destroy(dev);
    }

    // ---------------------------------------------------------------
    // Shape 3: a DYNAMIC rigid.  isStaticBody() is false, so the one-way arrays
    // must stay untouched and everything flows through the writer.
    // ---------------------------------------------------------------
    {
        BdaOptions opt;
        opt.staticRigid = false;
        BdaRig A;
        RefRig B;
        buildBda(dev, S, opt, A);
        buildRef(dev, S, opt, B);

        {
            Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
            k.dispatch(nbWarps, {}, A.blob, 1u, 1u);
        }
        {
            Kernel k("psMeshMid.spv", "ctPsMeshMid", refStrides(), sizeof(PmParams));
            k.dispatch(nbWarps, B.bindings(), B.prm);
        }
        dev->flush();

        const auto aCnt = downloadBuffer<uint32_t>(dev, A.bCounts.h, S.nppad);
        size_t touched = 0;
        for (uint32_t p = 0; p < S.nppad; ++p) if (aCnt[p] != 0u) ++touched;
        PXG_CHECK_MSG(touched == 0,
                      "dynamic rigid: %zu one-way counters were written; that path is static-only",
                      touched);

        const uint32_t aWc = downloadBuffer<uint32_t>(dev, A.bCount.h, 1)[0];
        const uint32_t bWc = downloadBuffer<uint32_t>(dev, B.bWc, 1)[0];
        PXG_CHECK_MSG(aWc == bWc, "dynamic rigid: writer counters differ (%u vs %u)", aWc, bWc);
        PXG_CHECK_MSG(aWc >= withContacts,
                      "dynamic rigid: only %u writer contacts for %zu contacting particles",
                      aWc, withContacts);

        const auto aCt = downloadBuffer<PsC>(dev, A.bContacts.h, Scene::CAP);
        const auto bCt = downloadBuffer<PsC>(dev, B.bW, Scene::CAP);
        const auto ga = groupByParticle(aCt, aWc);
        const auto gb = groupByParticle(bCt, bWc);
        size_t mism = 0;
        for (uint32_t p = 0; p <= Scene::NP; ++p)
        {
            if (ga[p].size() != gb[p].size()) { ++mism; continue; }
            for (size_t i = 0; i < ga[p].size(); ++i)
                if (!sameContact(ga[p][i], gb[p][i])) { ++mism; break; }
        }
        PXG_CHECK_MSG(mism == 0, "dynamic rigid: %zu particles' writer contacts differ", mism);

        // The three sibling arrays the writer fills beside the contact record.
        {
            const auto aIdx = downloadBuffer<uint32_t>(dev, A.bIbP.h, Scene::CAP);
            const auto aTbP = downloadBuffer<uint32_t>(dev, A.bTbP.h, Scene::CAP);
            const auto aBr = downloadBuffer<uint64_t>(dev, A.bBr.h, Scene::CAP);
            const auto aTbr = downloadBuffer<uint32_t>(dev, A.bTbr.h, Scene::CAP);
            const PxNodeIndex dyn(0x77u, 4u);
            size_t bad = 0;
            for (uint32_t i = 0; i < aWc; ++i)
            {
                if (aIdx[i] != i) ++bad;
                if (aTbP[i] != aCt[i].pLo) ++bad;
                if (aBr[i] != dyn.getInd()) ++bad;
                if (aTbr[i] != uint32_t(dyn.getInd() & 0xFFFFFFFFull)) ++bad;
            }
            PXG_CHECK_MSG(bad == 0, "%zu writer sibling-array entries are wrong", bad);
            // Past the live count everything must still be poison.
            size_t clobbered = 0;
            for (uint32_t i = aWc; i < Scene::CAP; ++i)
                if (aIdx[i] != POISON_U32) ++clobbered;
            PXG_CHECK_MSG(clobbered == 0, "%zu writer slots past the count were written", clobbered);
        }

        A.destroy(dev);
        B.destroy(dev);
    }

    // ---------------------------------------------------------------
    // Shape 4: LoadShapePair the other way round.  The trimesh moves to
    // shapeRef1, so the swap branch runs, and the result must be IDENTICAL.
    // ---------------------------------------------------------------
    {
        BdaOptions o0, o1;
        o1.triFirst = false;
        BdaRig A0, A1;
        buildBda(dev, S, o0, A0);
        buildBda(dev, S, o1, A1);
        {
            Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
            k.dispatch(nbWarps, {}, A0.blob, 1u, 1u);
            k.dispatch(nbWarps, {}, A1.blob, 1u, 1u);
        }
        dev->flush();
        const auto i0 = downloadBuffer<F4>(dev, A0.bInfos.h, size_t(S.nppad) * MAXPART);
        const auto i1 = downloadBuffer<F4>(dev, A1.bInfos.h, size_t(S.nppad) * MAXPART);
        const auto c0 = downloadBuffer<uint32_t>(dev, A0.bCounts.h, S.nppad);
        const auto c1 = downloadBuffer<uint32_t>(dev, A1.bCounts.h, S.nppad);
        PXG_CHECK_MSG(std::memcmp(c0.data(), c1.data(), c0.size() * 4) == 0,
                      "LoadShapePair: swapping the pair changed the one-way counts");
        PXG_CHECK_MSG(std::memcmp(i0.data(), i1.data(), i0.size() * sizeof(F4)) == 0,
                      "LoadShapePair: swapping the pair changed the one-way infos");
        size_t nonEmpty = 0;
        for (uint32_t p = 0; p < Scene::NP; ++p) if (c0[p] != 0u) ++nonEmpty;
        PXG_CHECK_MSG(nonEmpty >= 20, "the swap check ran on an empty result");
        A0.destroy(dev);
        A1.destroy(dev);
    }
}

// ===========================================================================
// The rigid filter and D2's sticky laneDone.  psMeshMid.slang does not have the
// filter at all -- its header lists it as unported glue -- so the differential
// cannot see this and it needs its own test.
// ===========================================================================
PXG_TEST(psMeshMidphaseFilter)
{
    PxgSlangComputeDevice* dev = testDevice();

    Scene S;
    buildScene(S);
    const uint32_t nbWarps = (Scene::NP + 31u) / 32u;

    // Baseline: no filters at all.
    BdaOptions base;
    BdaRig A;
    buildBda(dev, S, base, A);
    {
        Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
        k.dispatch(nbWarps, {}, A.blob, 1u, 1u);
    }
    dev->flush();
    const auto baseCnt = downloadBuffer<uint32_t>(dev, A.bCounts.h, S.nppad);
    const auto baseInfos = downloadBuffer<F4>(dev, A.bInfos.h, size_t(S.nppad) * MAXPART);
    A.destroy(dev);

    // Which particles the filter will name.  find() is keyed on the UNSORTED index
    // (mSortedToUnsortedMapping[i]), not on i, so a port that queries with the
    // sorted index filters the wrong particles.  The map here is not the identity
    // precisely so those two sets differ.
    std::vector<uint32_t> s2u(S.nppad);
    for (uint32_t i = 0; i < S.nppad; ++i) s2u[i] = (i * 37u + 11u) % Scene::NP;

    // Filter every third CONTACTING particle, so the filtered set is interesting
    // rather than a suffix, and so filtered and unfiltered lanes share warps.
    std::vector<uint32_t> filteredSorted;
    for (uint32_t i = 0; i < Scene::NP; ++i)
        if (baseCnt[i] != 0u && (i % 3u) == 1u) filteredSorted.push_back(i);
    PXG_CHECK_MSG(filteredSorted.size() >= 5,
                  "only %zu particles to filter -- too few to say anything",
                  filteredSorted.size());
    // ... and at least one warp must keep BOTH kinds, or the sticky-flag divergence
    // is never exercised.
    {
        std::vector<uint32_t> perWarpFiltered(nbWarps, 0), perWarpLive(nbWarps, 0);
        for (uint32_t i = 0; i < Scene::NP; ++i)
            if (baseCnt[i] != 0u) ++perWarpLive[i / 32u];
        for (uint32_t i : filteredSorted) ++perWarpFiltered[i / 32u];
        size_t mixed = 0;
        for (uint32_t w = 0; w < nbWarps; ++w)
            if (perWarpFiltered[w] > 0 && perWarpLive[w] > perWarpFiltered[w]) ++mixed;
        PXG_CHECK_MSG(mixed >= 2,
                      "only %zu warps mix filtered and unfiltered contacting lanes", mixed);
    }

    BdaOptions filt;
    const PxNodeIndex staticNode(PX_INVALID_NODE, 0u);
    for (uint32_t i : filteredSorted)
    {
        PxgRigidFilterPair pr;
        std::memset(&pr, 0, sizeof(pr));
        pr.index0 = staticNode.getInd();
        pr.index1 = s2u[i];      // the UNSORTED index, which is what find() compares
        pr.index2 = 0u;
        filt.filterPairs.push_back(pr);
    }
    // The binary search requires the list sorted by (index0, index1).
    std::sort(filt.filterPairs.begin(), filt.filterPairs.end(),
              [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
                  if (a.index0 != b.index0) return a.index0 < b.index0;
                  return a.index1 < b.index1;
              });
    filt.nbFilterPairs = uint32_t(filt.filterPairs.size());

    BdaRig Bf;
    buildBda(dev, S, filt, Bf);
    {
        Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
        k.dispatch(nbWarps, {}, Bf.blob, 1u, 1u);
    }
    dev->flush();
    const auto fCnt = downloadBuffer<uint32_t>(dev, Bf.bCounts.h, S.nppad);
    const auto fInfos = downloadBuffer<F4>(dev, Bf.bInfos.h, size_t(S.nppad) * MAXPART);

    std::vector<uint8_t> isFiltered(S.nppad, 0);
    for (uint32_t i : filteredSorted) isFiltered[i] = 1;

    size_t wrongFiltered = 0, wrongKept = 0;
    for (uint32_t p = 0; p < Scene::NP; ++p)
    {
        if (isFiltered[p])
        {
            if (fCnt[p] != 0u) ++wrongFiltered;
        }
        else
        {
            // Every unfiltered particle must be BIT-IDENTICAL to the baseline: the
            // filtered lanes stay in the barriers and the ballots, so their
            // neighbours' results cannot move.  That is the whole point of D2's
            // sticky flag over a divergent return.
            if (fCnt[p] != baseCnt[p]) { ++wrongKept; continue; }
            for (uint32_t k = 0; k < std::min(fCnt[p], MAXPART); ++k)
            {
                const size_t oi = size_t(p) + size_t(k) * Scene::NP;
                if (std::memcmp(&fInfos[oi], &baseInfos[oi], sizeof(F4)) != 0) { ++wrongKept; break; }
            }
        }
    }
    PXG_CHECK_MSG(wrongFiltered == 0,
                  "%zu filtered particles still produced contacts", wrongFiltered);
    PXG_CHECK_MSG(wrongKept == 0,
                  "%zu unfiltered particles changed when their warp-mates were filtered",
                  wrongKept);

    // And the filter must actually have removed something, or both checks above
    // are satisfied by a kernel that ignores the list entirely.
    size_t removed = 0;
    for (uint32_t i : filteredSorted) if (baseCnt[i] != 0u) ++removed;
    PXG_CHECK_MSG(removed >= 5, "the filter removed only %zu particles' contacts", removed);

    Bf.destroy(dev);
}

// ===========================================================================
// The work loop's index arithmetic.
// ===========================================================================
namespace
{
// A second, independently written enumeration of the same loop.  Deliberately NOT
// the closed form: P9.9's review killed a coverage assertion that restated the
// arithmetic in a specialised algebraic form which happened to be false at a
// different iteration count.
static std::vector<uint32_t> enumerateCoverage(uint32_t numParticles, uint32_t gridX)
{
    const uint32_t nbWarpsRequired = (numParticles + 31u) / 32u;
    const uint32_t totalNumWarps = MESH_COLLISION_GRID * PS_MIDPHASE_WARPS;
    const uint32_t nbIter = (nbWarpsRequired + totalNumWarps - 1u) / totalNumWarps;
    std::vector<uint32_t> cover(nbWarpsRequired, 0u);
    for (uint32_t b = 0; b < gridX; ++b)
        for (uint32_t it = 0; it < nbIter; ++it)
        {
            const uint32_t w = it + (0u + PS_MIDPHASE_WARPS * b) * nbIter;
            if (w < nbWarpsRequired) ++cover[w];
        }
    return cover;
}

static void runCoverageCase(PxgSlangComputeDevice* dev, uint32_t numParticles, uint32_t expectNbIter)
{
    const uint32_t nbWarpsRequired = (numParticles + 31u) / 32u;
    const uint32_t totalNumWarps = MESH_COLLISION_GRID * PS_MIDPHASE_WARPS;
    const uint32_t nbIter = (nbWarpsRequired + totalNumWarps - 1u) / totalNumWarps;
    PXG_CHECK_MSG(nbIter == expectNbIter,
                  "numParticles %u should give nbIterationsPerWarps %u, the fixture computes %u",
                  numParticles, expectNbIter, nbIter);

    const uint32_t PAD = 16u;
    std::vector<uint32_t> cover(nbWarpsRequired + PAD, 0u);
    for (uint32_t i = nbWarpsRequired; i < cover.size(); ++i) cover[i] = POISON_U32;
    DevBuf b = makeDev(dev, cover);

    PsMeshWorkBlob push = { b.addr, numParticles, 0u };
    {
        Kernel k("psMeshMidphaseBda.spv", "psMeshMidphaseWorkIndex", {}, sizeof(PsMeshWorkBlob));
        k.dispatch(MESH_COLLISION_GRID, {}, push, 1u, 1u);
    }
    dev->flush();
    const auto got = downloadBuffer<uint32_t>(dev, b.h, cover.size());
    dev->destroyBuffer(b.h);

    const auto want = enumerateCoverage(numParticles, MESH_COLLISION_GRID);
    PXG_CHECK_MSG(want.size() == nbWarpsRequired, "the enumeration disagrees on the extent");

    size_t uncovered = 0, over = 0, wrong = 0;
    for (uint32_t w = 0; w < nbWarpsRequired; ++w)
    {
        if (got[w] != want[w]) ++wrong;
        if (got[w] == 0u) ++uncovered;
        if (got[w] > 1u) ++over;
    }
    PXG_CHECK_MSG(wrong == 0,
                  "nbIter %u: %zu work indices have the wrong multiplicity (%zu never covered, "
                  "%zu covered more than once)", nbIter, wrong, uncovered, over);
    // This loop is CORRECT where the boundary-density one is not (candidate 16):
    // every work index is covered exactly once.  Asserted, so a port that acquires
    // the boundary kernel's double cover dies here.
    PXG_CHECK_MSG(uncovered == 0 && over == 0,
                  "nbIter %u: coverage is not exactly one per work index (%zu skipped, %zu doubled)",
                  nbIter, uncovered, over);
    size_t pastEnd = 0;
    for (uint32_t i = nbWarpsRequired; i < cover.size(); ++i)
        if (got[i] != POISON_U32) ++pastEnd;
    PXG_CHECK_MSG(pastEnd == 0, "nbIter %u: %zu slots past the last work index were written",
                  nbIter, pastEnd);
}
} // namespace

PXG_TEST(psMeshMidphaseWorkIndex)
{
    PxgSlangComputeDevice* dev = testDevice();

    // nbIterationsPerWarps == 1: every real scene's shape, and the size at which
    // the `* nbIterationsPerWarps` factor is an IDENTITY and pins nothing.
    runCoverageCase(dev, 32u * 1000u + 7u, 1u);
    // == 2 and == 3: the block stride is 2 and then 3 against a fixed iteration
    // stride of 1, so the factor is under test and an off-by-one in either the
    // warp count or the iteration count separates the covered SETS.
    runCoverageCase(dev, 32u * MESH_COLLISION_GRID + 1u, 2u);
    runCoverageCase(dev, 64u * MESH_COLLISION_GRID + 1u, 3u);
}
