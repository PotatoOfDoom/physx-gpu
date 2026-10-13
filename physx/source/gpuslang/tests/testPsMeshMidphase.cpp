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
#include "PxFiltering.h"          // PxParticleRigidFilterPair (16 B, the particle path)
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
// The root's shape.  ROOT_LEAF_CHUNK must stay at or below the leaf threshold in
// buildNode and ROOT_NODE_CHUNK comfortably above it, or the root stops producing
// one of the two kinds of child.
static const uint32_t ROOT_FANOUT = 26u;
static const uint32_t ROOT_LEAF_CHUNK = 8u;
static const uint32_t ROOT_NODE_CHUNK = 40u;

static uint32_t buildNode(std::vector<Bv32NodeHost>& nodes, const std::vector<AABB>& prims,
                          uint32_t begin, uint32_t end, std::mt19937& rng, int depth)
{
    const uint32_t nodeIdx = uint32_t(nodes.size());
    nodes.push_back(Bv32NodeHost());
    std::memset(&nodes.back(), 0, sizeof(Bv32NodeHost));
    const uint32_t count = end - begin;
    std::vector<uint32_t> cuts = { begin, end };
    if (depth == 0)
    {
        // The ROOT is deliberately WIDE, and deliberately UNEVEN, and both halves of
        // that matter -- the first draft got only the first half and traded away
        // three kills for it.
        //
        //  * WIDE, because a BV32 node holds up to 32 children and a leaf's offset
        //    is `(nodeIndex >> 1) & 31`.  With the 2..6 children a random split
        //    gives, slot indices never exceed 5 and a mutant that masks with 15
        //    instead of 31 is an identity.
        //  * UNEVEN, because a wide root of EVENLY sized chunks put every chunk
        //    under the leaf threshold, so every root child became a leaf, the tree
        //    became one level deep, and the child-offset shift and the node-index
        //    mask stopped being reachable at all.  t3 and t5 went from KILLED to
        //    SURVIVED on exactly that.
        //
        // So: alternating small chunks (leaves, at slots 0, 2, ... 24 -- including
        // slots past 15) and large ones (internal nodes, so the descent is real).
        uint32_t cursor = begin;
        for (uint32_t i = 0; i < ROOT_FANOUT - 1u && cursor < end; ++i)
        {
            const uint32_t sz = (i % 2u == 0u) ? ROOT_LEAF_CHUNK : ROOT_NODE_CHUNK;
            cursor = std::min(end, cursor + sz);
            cuts.push_back(cursor);
        }
    }
    else
    {
        const uint32_t nbChunks = std::min<uint32_t>(2 + rng() % 5, count);
        while (cuts.size() < nbChunks + 1)
            cuts.push_back(begin + 1 + rng() % (count - 1));
    }
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
    static const uint32_t NR = 624u;   // = 13 * ROOT_LEAF_CHUNK + 13 * ROOT_NODE_CHUNK
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
        // Every 37th triangle is an EXACT DUPLICATE of its predecessor.  Duplicates
        // give a particle two bit-identical contact depths, which is the only way a
        // fixture can reach the tie the worst-contact replacement's argmax breaks --
        // random float geometry never produces one.
        if (t > 0 && (t % 37u) == 5u)
            for (int k = 0; k < 3; ++k) S.verts[t * 3 + k] = S.verts[(t - 1) * 3 + k];
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
    // STRONGLY anisotropic, and in a ROTATED scale frame.  With a near-uniform
    // scale, shape space and vertex space are nearly the same place and a mutant
    // that swaps shape2Vertex for vertex2Shape in the box-culling setup culls the
    // same nodes; at 2.4 / 0.55 / 1.35 in a tilted frame it does not.
    S.mesh.scale = { 2.4f, -0.55f, 1.35f, cd };
    {
        F4 q = { 0.31f, 0.17f, -0.24f, 1.0f };
        const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        S.mesh.scaleRot = { q.x / m, q.y / m, q.z / m, q.w / m };
    }
    S.mesh.misc = { 0.03f, 0, 0, 0 };            // restDist
    S.mesh.rigid = { PX_INVALID_NODE, 0u, 0, 0 };  // STATIC by default

    // The densest spot, so some particles overflow the six-slot list and drive the
    // serialized worst-contact replacement (M6).
    //
    // Density is measured in SHAPE space and against the CONTACT DISTANCE, not in
    // vertex space against a fixed radius.  With a strongly anisotropic scale the
    // two are not the same question: a vertex-space cluster of radius 0.35 spreads
    // to 0.84 in shape space at scale 2.4, so triangles that look clustered are far
    // apart where the contact test actually happens -- and no particle saturates.
    // The first version of this fixture measured it in vertex space and stopped
    // producing overflow the moment the scale was made anisotropic.
    std::vector<V3> shapeCentroid(Scene::NR);
    for (uint32_t t = 0; t < Scene::NR; ++t)
    {
        V3 tc = { 0, 0, 0 };
        for (int k = 0; k < 3; ++k) tc = add(tc, v3(S.verts[S.tris[t].x + k]));
        shapeCentroid[t] = hVertex2Shape(muls(tc, 1.0f / 3.0f), v3(S.mesh.scale), S.mesh.scaleRot);
    }
    uint32_t denseTri = 0;
    {
        const float r = 1.5f * cd;
        size_t bestCnt = 0;
        for (uint32_t t = 0; t < Scene::NR; ++t)
        {
            size_t cnt = 0;
            for (uint32_t u = 0; u < Scene::NR; ++u)
            {
                const V3 d = sub(shapeCentroid[u], shapeCentroid[t]);
                if (dot3(d, d) < r * r) ++cnt;
            }
            if (cnt > bestCnt) { bestCnt = cnt; denseTri = t; }
        }
    }

    S.nppad = ((Scene::NP + 31u) / 32u) * 32u;
    S.curPos.assign(S.nppad, { 100, 100, 100, 1 });
    S.predPos.assign(S.nppad, { 100, 100, 100, 1 });
    for (uint32_t p = 0; p < Scene::NP; ++p)
    {
        // Every third particle sits in the densest cluster, so more of them
        // overflow the six-slot list and the replacement pass sees candidates at
        // widely different depths -- including candidates WORSE than all six
        // already stored, which is the only case where the swap's comparison
        // direction is observable.
        const uint32_t t = (p % 3 == 2) ? denseTri : (p * 29) % Scene::NR;
        const V3 sp = shapeCentroid[t];
        const V3 wp = add(hQuatRotate(S.mesh.meshQ, sp), v3(S.mesh.meshP));
        V3 dir = { uv(rng), uv(rng), uv(rng) };
        const float m = std::sqrt(std::max(dot3(dir, dir), 1e-3f));
        dir = muls(dir, 1.0f / m);
        // The dense particles sit INSIDE the contact band (offsets are world units
        // and the band is contactDist), so many of the clustered triangles qualify
        // at once and the six-slot list overflows; the rest are spread so that most
        // particles find a handful and some find none.
        const float off = (p % 3 == 2) ? cd * (0.05f + 0.85f * u01(rng))
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
    DevBuf bDiffInfos, bDiffNodeIdx, bDiffCounts, bNumDiffuse, bDiffCur, bDiffPred;
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
// The runsum boundary between the two particle buffers.  Filter pairs live in
// buffer 1, so a query that forgets to subtract the runsum looks in the wrong
// place -- see the note in buildBda.
static const uint32_t FILTER_BUFFER_BASE = 40u;

struct BdaOptions
{
    bool triFirst = true;            // which half of the pair is the triangle mesh
    bool staticRigid = true;
    bool enableCcd = true;
    bool isTgs = true;
    uint32_t nbFilterPairs = 0u;
    // PxParticleRigidFilterPair, NOT PxgRigidFilterPair: the particle path's
    // find() overload (deformableElementFilter.cuh:39) is over the 16-byte
    // PxFiltering.h struct.  See the note in pxgSlangParticleContact.slang --
    // this fixture used to mirror the 32-byte one and so agreed with a shader
    // that read the wrong stride.
    std::vector<PxParticleRigidFilterPair> filterPairs;
    // M1: when non-zero the descriptor gets a full diffuse side, so a gridDim.z of
    // 2 runs a real second pass instead of returning early.
    uint32_t maxDiffuse = 0u;
    // The diffuse pass's own positions and count.  They must NOT be the solid ones:
    // aliasing them makes "read the diffuse array" and "read the solid array"
    // indistinguishable, which is exactly how m1d and m1e survived the first sweep.
    const std::vector<F4>* diffuseCur = nullptr;
    const std::vector<F4>* diffusePred = nullptr;
    uint32_t numDiffuse = 0u;
    // A positive scale determinant, so MeshScaling::flipNormal is FALSE and the
    // V1/V2 swap is under test in both directions.
    bool positiveDet = false;
    // The writer's capacity.  Small enough and writeContact's own bound binds.
    uint32_t maxContacts = Scene::CAP;
    // Override the solid particle count and positions (used to build the PGS
    // reference the diffuse pass is compared against).
    const std::vector<F4>* solidCur = nullptr;
    const std::vector<F4>* solidPred = nullptr;
    uint32_t numSolid = Scene::NP;
};

// The mesh scale for a given variant.  ONE definition, used by both builders, so
// the two rigs cannot drift apart.
static F4 sceneScale(const Scene& S, bool positiveDet)
{
    F4 sc = S.mesh.scale;
    if (positiveDet) sc.y = -sc.y;   // flips the determinant, keeps the magnitudes
    return sc;
}

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
    R.bCur = R.mk(dev, opt.solidCur ? *opt.solidCur : S.curPos);
    R.bPred = R.mk(dev, opt.solidPred ? *opt.solidPred : S.predPos);

    // mSortedToUnsortedMapping: a real permutation-like map, distinct from identity
    // so a filter query built from the wrong array is visible.
    std::vector<uint32_t> s2u(S.nppad);
    for (uint32_t i = 0; i < S.nppad; ++i) s2u[i] = (i * 37u + 11u) % Scene::NP;
    R.bS2U = R.mk(dev, s2u);

    // TWO particle buffers with a NON-ZERO runsum.  find() resolves the buffer by
    // runsum and then queries with the LOCAL id (particleId - runsum[buffer]); with
    // one buffer at runsum 0 that subtraction is an identity and a port that drops
    // it cannot be caught -- which is how d2h survived the first sweep.  Every
    // filter pair lives in buffer ONE.
    std::vector<uint32_t> runsum = { 0u, FILTER_BUFFER_BASE };
    R.bRunsum = R.mk(dev, runsum);

    DevBuf bPairs;
    if (!opt.filterPairs.empty()) bPairs = R.mk(dev, opt.filterPairs);
    std::vector<PxgParticleSimBuffer> simBuffers(2);
    std::memset(simBuffers.data(), 0, simBuffers.size() * sizeof(PxgParticleSimBuffer));
    simBuffers[0].mNumFilterPairs = 0u;
    simBuffers[0].mFilterPairs = nullptr;
    simBuffers[1].mNumFilterPairs = opt.nbFilterPairs;
    simBuffers[1].mFilterPairs =
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
    ps.mCommonData.mNumParticles = opt.numSolid;
    ps.mCommonData.mMaxParticles = S.nppad;
    ps.mCommonData.mNumParticleBuffers = 2u;
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
    // M1's other half.  The diffuse pass reads its OWN position arrays, its own
    // particle count (through a pointer, not a field) and its own six one-way
    // pointers.  Here the diffuse positions ALIAS the solid ones and the count is
    // the same, so with CCD off the two passes must compute identical geometry and
    // the diffuse output can be compared against the solid output byte for byte --
    // which pins all eight selections at once without a second reference.
    if (opt.maxDiffuse != 0u)
    {
        std::vector<F4> dInfos(size_t(S.nppad) * MAXPART, { -9, -9, -9, -9 });
        std::vector<uint32_t> dNodeIdx(size_t(S.nppad) * MAXPART * 2u, 0xCCCCCCCCu);
        std::vector<uint32_t> dCounts(S.nppad, 0u);
        std::vector<uint32_t> nDiff = { opt.numDiffuse };
        R.bDiffInfos = R.mk(dev, dInfos);
        R.bDiffNodeIdx = R.mk(dev, dNodeIdx);
        R.bDiffCounts = R.mk(dev, dCounts);
        R.bNumDiffuse = R.mk(dev, nDiff);
        ps.mCommonData.mMaxDiffuseParticles = opt.maxDiffuse;
        // The engine declares this one as int*, not PxU32*; the shader reads the
        // same bits as a uint, which is correct for any non-negative count.
        ps.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bNumDiffuse.addr));
        R.bDiffCur = R.mk(dev, opt.diffuseCur ? *opt.diffuseCur : S.curPos);
        R.bDiffPred = R.mk(dev, opt.diffusePred ? *opt.diffusePred : S.predPos);
        ps.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDiffCur.addr));
        ps.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDiffPred.addr));
        ps.mDiffuseOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.bDiffInfos.addr));
        ps.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.bDiffNodeIdx.addr));
        ps.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.bDiffCounts.addr));
    }
    // P9.99b: slot 0 is a VALID decoy -- the same inputs, its OWN poisoned outputs.  A
    // kernel that resolved the system to index 0 instead of the shape's
    // particleOrSoftbodyId (the p2-meshmid-sys0 mutant) used to dereference the 0xEE
    // poison and fault; now it writes the decoy's outputs and leaves slot SYS's poisoned,
    // so it dies by value.  The other neighbours keep the 0xEE poison.
    {
        PxgParticleSystem decoy = ps;
        std::vector<F4> dInfos(size_t(S.nppad) * MAXPART, { -5, -5, -5, -5 });
        std::vector<uint32_t> dNodeIdx(size_t(S.nppad) * MAXPART * 2u, 0xDDDDDDDDu);
        std::vector<uint32_t> dCounts(S.nppad, 0u);
        decoy.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.mk(dev, dInfos).addr));
        decoy.mOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.mk(dev, dNodeIdx).addr));
        decoy.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.mk(dev, dCounts).addr));
        if (opt.maxDiffuse != 0u)
        {
            decoy.mDiffuseOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.mk(dev, dInfos).addr));
            decoy.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.mk(dev, dNodeIdx).addr));
            decoy.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.mk(dev, dCounts).addr));
        }
        descArr[0] = decoy;
    }
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
    {
        const F4 sc = sceneScale(S, opt.positiveDet);
        shapes[TRI_SHAPE].scale = PxMeshScale(PxVec3(sc.x, sc.y, sc.z),
                                              PxQuat(S.mesh.scaleRot.x, S.mesh.scaleRot.y,
                                                     S.mesh.scaleRot.z, S.mesh.scaleRot.w));
    }
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

    // The CUDA bool occupies ONE BYTE at offset 0; the three bytes behind it are
    // launch-blob padding holding whatever the launcher left.  Poisoning them is
    // what makes the masked read observable -- a port that tests the whole word
    // reads TRUE even when the byte says false.
    R.blob.isTGS = 0xABABAB00u | (opt.isTgs ? 1u : 0u);
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
    R.blob.w_maxContacts = opt.maxContacts;
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
    {
        const F4 sc = sceneScale(S, opt.positiveDet);
        mesh.scale.x = sc.x; mesh.scale.y = sc.y; mesh.scale.z = sc.z;
    }
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
    R.bCur = R.mk(dev, opt.solidCur ? *opt.solidCur : S.curPos);
    R.bPred = R.mk(dev, opt.solidPred ? *opt.solidPred : S.predPos);

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

    R.prm.numParticles = opt.numSolid;
    R.prm.maxContacts = opt.maxContacts;
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

    // The filter list the port walks.  16 bytes, two PxU64 -- the struct the
    // particle overload of find() actually indexes.
    PXG_CHECK_MSG(sizeof(PxParticleRigidFilterPair) == 16,
                  "PxParticleRigidFilterPair stride (got %zu)",
                  sizeof(PxParticleRigidFilterPair));
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

        // ALL SIX sibling arrays the writer fills beside the contact record.
        //
        // It used to be four.  P9.12g lifted writeContact into
        // pxgSlangParticleContact and swept it against THIS suite, and two
        // mutants survived -- one that offset contactIndexSortedByRigid and one
        // that wrote the rigid id into contactSortedByParticle.  Neither was
        // caused by the lift: those two arrays were bound here and then never
        // read back, so half the writer's stores had no assertion behind them at
        // all.  Giving the code a single home is what made the gap visible.
        {
            const auto aIdx = downloadBuffer<uint32_t>(dev, A.bIbP.h, Scene::CAP);
            const auto aTbP = downloadBuffer<uint32_t>(dev, A.bTbP.h, Scene::CAP);
            const auto aBr = downloadBuffer<uint64_t>(dev, A.bBr.h, Scene::CAP);
            const auto aTbr = downloadBuffer<uint32_t>(dev, A.bTbr.h, Scene::CAP);
            const auto aIbr = downloadBuffer<uint32_t>(dev, A.bIbr.h, Scene::CAP);
            const auto aSbP = downloadBuffer<uint64_t>(dev, A.bSbP.h, Scene::CAP);
            const PxNodeIndex dyn(0x77u, 4u);
            size_t bad = 0;
            for (uint32_t i = 0; i < aWc; ++i)
            {
                if (aIdx[i] != i) ++bad;
                if (aTbP[i] != aCt[i].pLo) ++bad;
                if (aBr[i] != dyn.getInd()) ++bad;
                if (aTbr[i] != uint32_t(dyn.getInd() & 0xFFFFFFFFull)) ++bad;
                // The two that were never checked.  Both are plain identities:
                // the rigid-sorted index is the slot itself, and the
                // particle-sorted key is the compressed particle index the
                // contact record carries.
                if (aIbr[i] != i) ++bad;
                if (aSbP[i] != (uint64_t(aCt[i].pHi) << 32) + uint64_t(aCt[i].pLo)) ++bad;
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
    // Shape 3b: CCD OFF.  The non-CCD branch sets cVolumeRadius = contactDist
    // DIRECTLY -- M3's point: no division by the smallest scale component, which
    // is what the softbody twin does and what copying it would introduce.  Every
    // other shape runs with CCD on, where that branch is dead.
    // Shape 3c: PGS.  isTGS reaches the shader as ONE BYTE with three poisoned
    // padding bytes behind it, so a full-word read sees TRUE where the byte says
    // false, and the TGS/PGS projection scales differ (0.5 vs 1.0).
    // Shape 3d: a POSITIVE scale determinant, so flipNormal is false and the
    // V1/V2 swap is tested in its other direction.
    // Shape 3e: a SMALL writer capacity, so writeContact's own bound binds.
    // ---------------------------------------------------------------
    {
        struct Variant { const char* name; BdaOptions opt; };
        std::vector<Variant> variants;
        { BdaOptions o; o.enableCcd = false;  variants.push_back({ "ccd-off", o }); }
        { BdaOptions o; o.isTgs = false;      variants.push_back({ "pgs", o }); }
        { BdaOptions o; o.positiveDet = true; variants.push_back({ "positive-det", o }); }
        { BdaOptions o; o.staticRigid = false; o.maxContacts = 24u;
          variants.push_back({ "small-writer-cap", o }); }

        for (const Variant& v : variants)
        {
            BdaRig A;
            RefRig B;
            buildBda(dev, S, v.opt, A);
            buildRef(dev, S, v.opt, B);
            {
                Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
                k.dispatch(nbWarps, {}, A.blob, 1u, 1u);
            }
            {
                Kernel k("psMeshMid.spv", "ctPsMeshMid", refStrides(), sizeof(PmParams));
                k.dispatch(nbWarps, B.bindings(), B.prm);
            }
            dev->flush();

            const auto aI = downloadBuffer<F4>(dev, A.bInfos.h, size_t(S.nppad) * MAXPART);
            const auto bI = downloadBuffer<F4>(dev, B.bI, size_t(S.nppad) * MAXPART);
            const auto aC = downloadBuffer<uint32_t>(dev, A.bCounts.h, S.nppad);
            const auto bC = downloadBuffer<uint32_t>(dev, B.bC, S.nppad);
            PXG_CHECK_MSG(std::memcmp(aC.data(), bC.data(), aC.size() * 4) == 0,
                          "%s: the one-way COUNTS differ", v.name);
            PXG_CHECK_MSG(std::memcmp(aI.data(), bI.data(), aI.size() * sizeof(F4)) == 0,
                          "%s: the one-way CONTACT INFOS differ", v.name);

            const uint32_t aW = downloadBuffer<uint32_t>(dev, A.bCount.h, 1)[0];
            const uint32_t bW = downloadBuffer<uint32_t>(dev, B.bWc, 1)[0];
            PXG_CHECK_MSG(aW == bW, "%s: writer counters differ (%u vs %u)", v.name, aW, bW);

            if (!v.opt.staticRigid)
            {
                // The small-cap variant: the counter runs past the capacity while
                // only the first maxContacts records may be written.  Everything
                // beyond must still be poison in BOTH programs.
                PXG_CHECK_MSG(aW > v.opt.maxContacts,
                              "%s: the writer counter (%u) did not pass the capacity (%u), so "
                              "writeContact's bound is not under test",
                              v.name, aW, v.opt.maxContacts);
                const auto aIdx = downloadBuffer<uint32_t>(dev, A.bIbP.h, Scene::CAP);
                size_t past = 0;
                for (uint32_t i = v.opt.maxContacts; i < Scene::CAP; ++i)
                    if (aIdx[i] != POISON_U32) ++past;
                PXG_CHECK_MSG(past == 0,
                              "%s: %zu writer slots beyond maxContacts were written",
                              v.name, past);
            }
            else
            {
                size_t live = 0;
                for (uint32_t p = 0; p < Scene::NP; ++p) if (aC[p] != 0u) ++live;
                PXG_CHECK_MSG(live >= 20, "%s: ran on an almost empty result (%zu)", v.name, live);
            }
            A.destroy(dev);
            B.destroy(dev);
        }
    }

    // ---------------------------------------------------------------
    // Shape 3f: an OVER-SIZED grid.  The real launch is a FIXED 16384 blocks
    // regardless of the particle count, so `workIndex < nbWarpsRequired` is crossed
    // on almost every block of almost every frame -- while every other dispatch in
    // this file sizes the grid exactly and never reaches the guard at all.  The
    // result must be identical to the exactly-sized run.
    // ---------------------------------------------------------------
    {
        BdaOptions o;
        BdaRig A0, A1;
        buildBda(dev, S, o, A0);
        buildBda(dev, S, o, A1);
        {
            Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
            k.dispatch(nbWarps, {}, A0.blob, 1u, 1u);
            k.dispatch(nbWarps + 37u, {}, A1.blob, 1u, 1u);
        }
        dev->flush();
        const auto c0 = downloadBuffer<uint32_t>(dev, A0.bCounts.h, S.nppad);
        const auto c1 = downloadBuffer<uint32_t>(dev, A1.bCounts.h, S.nppad);
        const auto i0 = downloadBuffer<F4>(dev, A0.bInfos.h, size_t(S.nppad) * MAXPART);
        const auto i1 = downloadBuffer<F4>(dev, A1.bInfos.h, size_t(S.nppad) * MAXPART);
        PXG_CHECK_MSG(std::memcmp(c0.data(), c1.data(), c0.size() * 4) == 0,
                      "an over-sized grid changed the one-way counts; blocks past "
                      "nbWarpsRequired must do nothing");
        PXG_CHECK_MSG(std::memcmp(i0.data(), i1.data(), i0.size() * sizeof(F4)) == 0,
                      "an over-sized grid changed the one-way infos");
        const uint32_t w0 = downloadBuffer<uint32_t>(dev, A0.bCount.h, 1)[0];
        const uint32_t w1 = downloadBuffer<uint32_t>(dev, A1.bCount.h, 1)[0];
        PXG_CHECK_MSG(w0 == w1, "an over-sized grid moved the writer counter (%u vs %u)", w0, w1);
        A0.destroy(dev);
        A1.destroy(dev);
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
// M1 -- the diffuse split.  gridDim.z == 2 is not a redundant dimension: z == 1 is
// a SECOND, differently-sourced pass.  psMeshMid.slang has no diffuse side at all,
// so the differential is blind to this and it needs its own test.
// ===========================================================================
PXG_TEST(psMeshMidphaseDiffuse)
{
    PxgSlangComputeDevice* dev = testDevice();

    Scene S;
    buildScene(S);
    const uint32_t nbWarps = (Scene::NP + 31u) / 32u;

    // CCD OFF for the whole test.  With CCD on the two passes would legitimately
    // differ: getParticleSpeculativeContactVolume divides by
    // PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE (1.0) instead of the TGS scale
    // (0.5), so the volume radius is not the same number.  With CCD off the radius
    // is contactDist in both and the passes are comparable BIT FOR BIT.
    BdaOptions opt;
    opt.enableCcd = false;

    // --- (a) mMaxDiffuseParticles == 0: the diffuse pass must return immediately,
    //         and the solid pass must be unaffected by z having two slices.
    BdaRig Z1, Z2;
    opt.maxDiffuse = 0u;
    buildBda(dev, S, opt, Z1);
    buildBda(dev, S, opt, Z2);
    {
        Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
        k.dispatch(nbWarps, {}, Z1.blob, 1u, 1u);   // solid only
        k.dispatch(nbWarps, {}, Z2.blob, 1u, 2u);   // solid + diffuse
    }
    dev->flush();
    {
        const auto c1 = downloadBuffer<uint32_t>(dev, Z1.bCounts.h, S.nppad);
        const auto c2 = downloadBuffer<uint32_t>(dev, Z2.bCounts.h, S.nppad);
        const auto i1 = downloadBuffer<F4>(dev, Z1.bInfos.h, size_t(S.nppad) * MAXPART);
        const auto i2 = downloadBuffer<F4>(dev, Z2.bInfos.h, size_t(S.nppad) * MAXPART);
        // This is the assertion that kills a port which ignores gridDim.z: it would
        // run the SOLID pass twice and every count would double.
        PXG_CHECK_MSG(std::memcmp(c1.data(), c2.data(), c1.size() * 4) == 0,
                      "gridDim.z == 2 changed the solid one-way counts; the second z slice "
                      "must be the diffuse pass, and with no diffuse particles it must "
                      "return before touching anything");
        PXG_CHECK_MSG(std::memcmp(i1.data(), i2.data(), i1.size() * sizeof(F4)) == 0,
                      "gridDim.z == 2 changed the solid one-way infos");
        size_t live = 0;
        for (uint32_t p = 0; p < Scene::NP; ++p) if (c1[p] != 0u) ++live;
        PXG_CHECK_MSG(live >= 20, "the diffuse test ran on an empty solid result");
    }
    Z1.destroy(dev);
    Z2.destroy(dev);

    // --- (b) a REAL diffuse side, with its OWN positions and its OWN count.
    //
    // Aliasing them onto the solid ones (which the first draft did) makes "read the
    // diffuse array" and "read the solid array" indistinguishable, and the sweep
    // said so: three mutants survived on it.  So the diffuse particles are a
    // DIFFERENT set here and the count is DIFFERENT from the solid one, and the
    // reference is a second BDA rig whose SOLID side carries exactly that set.
    //
    // That reference is dispatched as PGS, and the reason is M1's last unexercised
    // clause: getParticleSpeculativeContactVolume takes the DIFFUSE projection
    // scale for a diffuse thread whatever isTGS says, and the diffuse scale (1.0)
    // is numerically the PGS scale.  So "diffuse under TGS" and "solid under PGS"
    // must agree exactly -- and a port that hands the function `false` instead of
    // `isDiffuse` picks the TGS scale and disagrees.  CCD is ON here, which is what
    // makes any of that reachable.
    const uint32_t ND = Scene::NP - 13u;   // distinct from NP, and not a multiple of 32
    std::vector<F4> diffCur(S.nppad, { 100, 100, 100, 1 });
    std::vector<F4> diffPred(S.nppad, { 100, 100, 100, 1 });
    for (uint32_t i = 0; i < ND; ++i)
    {
        // A rotation of the solid set, so the two are genuinely different arrays
        // with genuinely different contents at every index.
        const uint32_t src = (i + 29u) % Scene::NP;
        diffCur[i] = S.curPos[src];
        diffPred[i] = S.predPos[src];
    }

    BdaOptions dOpt;                    // CCD ON, TGS, static
    dOpt.maxDiffuse = Scene::NP;
    dOpt.numDiffuse = ND;
    dOpt.diffuseCur = &diffCur;
    dOpt.diffusePred = &diffPred;

    BdaOptions rOpt;                    // the reference: solid, PGS, same particles
    rOpt.isTgs = false;
    rOpt.solidCur = &diffCur;
    rOpt.solidPred = &diffPred;
    rOpt.numSolid = ND;

    BdaRig D, Rf;
    buildBda(dev, S, dOpt, D);
    buildBda(dev, S, rOpt, Rf);
    {
        Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
        k.dispatch(nbWarps, {}, D.blob, 1u, 2u);
        k.dispatch((ND + 31u) / 32u, {}, Rf.blob, 1u, 1u);
    }
    dev->flush();
    {
        const auto dc = downloadBuffer<uint32_t>(dev, D.bDiffCounts.h, S.nppad);
        const auto rc = downloadBuffer<uint32_t>(dev, Rf.bCounts.h, S.nppad);
        const auto di = downloadBuffer<F4>(dev, D.bDiffInfos.h, size_t(S.nppad) * MAXPART);
        const auto ri = downloadBuffer<F4>(dev, Rf.bInfos.h, size_t(S.nppad) * MAXPART);
        const auto dn = downloadBuffer<uint32_t>(dev, D.bDiffNodeIdx.h, size_t(S.nppad) * MAXPART * 2u);
        const auto rn = downloadBuffer<uint32_t>(dev, Rf.bNodeIdx.h, size_t(S.nppad) * MAXPART * 2u);
        size_t live = 0;
        for (uint32_t p = 0; p < ND; ++p) if (dc[p] != 0u) ++live;
        PXG_CHECK_MSG(live >= 20,
                      "the diffuse pass produced almost nothing (%zu particles); it is not "
                      "reading the diffuse positions or the diffuse count", live);
        PXG_CHECK_MSG(std::memcmp(dc.data(), rc.data(), dc.size() * 4) == 0,
                      "the diffuse pass's one-way COUNTS differ from the PGS solid reference");
        PXG_CHECK_MSG(std::memcmp(di.data(), ri.data(), di.size() * sizeof(F4)) == 0,
                      "the diffuse pass's one-way INFOS differ from the PGS solid reference");
        PXG_CHECK_MSG(std::memcmp(dn.data(), rn.data(), dn.size() * 4) == 0,
                      "the diffuse pass's one-way NODE INDICES differ from the PGS solid reference");
        // The diffuse count really must be the one in use: everything from ND on
        // stays untouched.
        size_t pastEnd = 0;
        for (uint32_t p = ND; p < S.nppad; ++p) if (dc[p] != 0u) ++pastEnd;
        PXG_CHECK_MSG(pastEnd == 0,
                      "%zu diffuse particles past *mNumDiffuseParticles were processed", pastEnd);
        // M1's writer clause: both passes are static here, so nothing reaches it.
        const uint32_t wc = downloadBuffer<uint32_t>(dev, D.bCount.h, 1)[0];
        PXG_CHECK_MSG(wc == 0u, "the static diffuse run reached the writer (%u contacts)", wc);
    }
    D.destroy(dev);
    Rf.destroy(dev);

    // --- (c) a DYNAMIC rigid with a real diffuse side -- UPSTREAM CANDIDATE 19.
    //
    // M1 says the diffuse pass is suppressed AT THE WRITER, and it is: the
    // writeContact loop is behind `!isDiffuseParticlesThread`
    // (particleSystemMeshMidphase.cu:356).  But the RESERVATION above it is not:
    //     if (idxInWarp == 31) startIndex = atomicAdd(writer.numTotalContacts, count);
    // (:345) runs for the diffuse pass too.  So the diffuse pass advances the
    // global contact counter by everything it found and then writes none of it,
    // leaving HOLES in the primitive-contact array below a count that says they are
    // contacts.  Downstream reads numTotalContacts records.
    //
    // This test asserts the FAITHFUL behaviour, not the sensible one -- which is
    // what makes it a test rather than a fix.  Because the diffuse positions alias
    // the solid ones here, the reservation is exactly as large as the solid pass's,
    // so the counter must land on TWICE the solid-only figure while only the solid
    // half of the slots is written.  A port that suppressed the atomicAdd as well
    // dies on the count; a port that forgot to suppress writeContact dies on the
    // hole count.
    BdaRig Ds, Dd;
    BdaOptions dyn;
    dyn.enableCcd = false;
    dyn.staticRigid = false;
    dyn.maxDiffuse = 0u;
    dyn.numDiffuse = Scene::NP;
    buildBda(dev, S, dyn, Ds);          // solid only, no diffuse side
    dyn.maxDiffuse = Scene::NP;
    buildBda(dev, S, dyn, Dd);          // solid + a full diffuse side
    {
        Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
        k.dispatch(nbWarps, {}, Ds.blob, 1u, 1u);
        k.dispatch(nbWarps, {}, Dd.blob, 1u, 2u);
    }
    dev->flush();
    {
        const uint32_t wSolid = downloadBuffer<uint32_t>(dev, Ds.bCount.h, 1)[0];
        const uint32_t wBoth = downloadBuffer<uint32_t>(dev, Dd.bCount.h, 1)[0];
        PXG_CHECK_MSG(wSolid > 0u, "the dynamic solid pass wrote nothing");
        PXG_CHECK_MSG(wBoth == 2u * wSolid,
                      "candidate 19: the counter should have reached 2 * %u (the diffuse pass "
                      "reserves what it then declines to write), and reached %u",
                      wSolid, wBoth);

        // ...and exactly half the reserved slots must be untouched poison.
        const auto idx = downloadBuffer<uint32_t>(dev, Dd.bIbP.h, Scene::CAP);
        size_t holes = 0;
        for (uint32_t i = 0; i < wBoth && i < Scene::CAP; ++i)
            if (idx[i] == POISON_U32) ++holes;
        PXG_CHECK_MSG(holes == wSolid,
                      "candidate 19: %zu of the %u reserved writer slots are holes, expected %u",
                      holes, wBoth, wSolid);

        // The diffuse pass's own one-way arrays must stay empty under a dynamic
        // rigid, or "suppressed at the writer" would be indistinguishable from
        // "did not run at all".
        const auto dc = downloadBuffer<uint32_t>(dev, Dd.bDiffCounts.h, S.nppad);
        size_t live = 0;
        for (uint32_t p = 0; p < Scene::NP; ++p) if (dc[p] != 0u) ++live;
        PXG_CHECK_MSG(live == 0u,
                      "%zu diffuse one-way counters were written under a DYNAMIC rigid; "
                      "the one-way path is static-only for the diffuse pass too", live);
    }
    Ds.destroy(dev);
    Dd.destroy(dev);
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
    // Only particles whose UNSORTED index lands in buffer ONE can be filtered, and
    // that is deliberate: the filter list lives in buffer 1 at runsum
    // FILTER_BUFFER_BASE, so the query has to resolve the buffer AND subtract the
    // runsum to find them.
    std::vector<uint32_t> filteredSorted;
    for (uint32_t i = 0; i < Scene::NP; ++i)
        if (baseCnt[i] != 0u && (i % 3u) == 1u && s2u[i] >= FILTER_BUFFER_BASE)
            filteredSorted.push_back(i);
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
    auto isFilteredSorted = [](const std::vector<uint32_t>& v, uint32_t i) {
        return std::find(v.begin(), v.end(), i) != v.end();
    };
    for (uint32_t i : filteredSorted)
    {
        PxParticleRigidFilterPair pr;
        std::memset(&pr, 0, sizeof(pr));
        pr.mID0 = staticNode.getInd();
        // The LOCAL id inside buffer 1: the unsorted index minus that buffer's
        // runsum.  A port that queries with the unsorted index itself finds nothing.
        pr.mID1 = s2u[i] - FILTER_BUFFER_BASE;
        filt.filterPairs.push_back(pr);
    }
    // DECOYS FOR TWO OTHER RIGIDS.  Without them the list carries exactly one
    // mID0 value and every mID1 fits in 32 bits, and three distinct wrong
    // comparators are indistinguishable from the right one (P9.12h's module
    // sweep measured this: `f-eq-mid1-only`, `f-lt-mid1-only` and
    // `f-u64lt-words-swapped` all survived).
    //
    //   * each decoy's mID1 is the local id of a particle that must be KEPT, so
    //     a comparator that ignores mID0 filters it and the bit-identical
    //     baseline comparison below fails;
    //   * the two decoy node indices are chosen so the true 64-bit order and a
    //     low-word-first order DISAGREE.  PxNodeIndex packs mID in the low word
    //     and mLinkID in the high one, so
    //         real  = 0x00000000'FFFFFFFF
    //         decoy0= 0x00000001'FFFFFFF0
    //         decoy1= 0x00000002'00000010
    //     sorts real < decoy0 < decoy1 by value and decoy1 < decoy0 < real by
    //     low word.  A binary search with the words swapped walks the wrong way
    //     and stops finding the real entries.
    {
        std::vector<uint32_t> keptLocal;
        for (uint32_t i = 0; i < Scene::NP && keptLocal.size() < 6; ++i)
            if (baseCnt[i] != 0u && !isFilteredSorted(filteredSorted, i) &&
                s2u[i] >= FILTER_BUFFER_BASE)
                keptLocal.push_back(s2u[i] - FILTER_BUFFER_BASE);
        PXG_CHECK_MSG(keptLocal.size() >= 4, "only %zu kept particles to decoy with",
                      keptLocal.size());
        const PxNodeIndex decoy0(0xFFFFFFF0u, 1u), decoy1(0x00000010u, 2u);
        for (size_t k = 0; k < keptLocal.size(); ++k)
        {
            PxParticleRigidFilterPair pr;
            std::memset(&pr, 0, sizeof(pr));
            pr.mID0 = (k & 1u) ? decoy1.getInd() : decoy0.getInd();
            pr.mID1 = keptLocal[k];
            filt.filterPairs.push_back(pr);
        }
        PXG_CHECK_MSG(staticNode.getInd() < decoy0.getInd() &&
                      decoy0.getInd() < decoy1.getInd(),
                      "the decoy node indices must bracket the real one by VALUE");
        PXG_CHECK_MSG(uint32_t(decoy1.getInd()) < uint32_t(decoy0.getInd()) &&
                      uint32_t(decoy0.getInd()) < uint32_t(staticNode.getInd()),
                      "...and reverse it by LOW WORD, or the swapped comparator is invisible");
    }

    // The binary search requires the list sorted by (mID0, mID1).
    std::sort(filt.filterPairs.begin(), filt.filterPairs.end(),
              [](const PxParticleRigidFilterPair& a, const PxParticleRigidFilterPair& b) {
                  if (a.mID0 != b.mID0) return a.mID0 < b.mID0;
                  return a.mID1 < b.mID1;
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

    // A ONE-ENTRY LIST NAMING A DIFFERENT RIGID.  The binary search probes index
    // 0 and nothing else, so this is the only construction in which "the entry the
    // search lands on carries the query's mID1 but another rigid's mID0" is
    // guaranteed rather than hoped for -- and it is what a comparator that tests
    // only mID1 needs to be seen at all (`f-eq-mid1-only` survived the decoys
    // above because they all sort past the real block and are never probed).
    // Nothing may be filtered: this list is about a rigid this pair is not.
    {
        uint32_t keptLocal = 0u; bool have = false;
        for (uint32_t i = 0; i < Scene::NP && !have; ++i)
            if (baseCnt[i] != 0u && !isFilteredSorted(filteredSorted, i) &&
                s2u[i] >= FILTER_BUFFER_BASE)
            { keptLocal = s2u[i] - FILTER_BUFFER_BASE; have = true; }
        PXG_CHECK_MSG(have, "no kept particle in buffer 1 to build the one-entry list from");

        BdaOptions one;
        PxParticleRigidFilterPair pr;
        std::memset(&pr, 0, sizeof(pr));
        pr.mID0 = PxNodeIndex(0x00000010u, 2u).getInd();   // NOT this pair's rigid
        pr.mID1 = keptLocal;
        one.filterPairs.push_back(pr);
        one.nbFilterPairs = 1u;

        BdaRig O;
        buildBda(dev, S, one, O);
        {
            Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
            k.dispatch(nbWarps, {}, O.blob, 1u, 1u);
        }
        dev->flush();
        const auto oc = downloadBuffer<uint32_t>(dev, O.bCounts.h, S.nppad);
        const auto oi = downloadBuffer<F4>(dev, O.bInfos.h, size_t(S.nppad) * MAXPART);
        PXG_CHECK_MSG(std::memcmp(oc.data(), baseCnt.data(), oc.size() * 4) == 0,
                      "a filter list naming ANOTHER rigid changed the contact counts");
        PXG_CHECK_MSG(std::memcmp(oi.data(), baseInfos.data(), oi.size() * sizeof(F4)) == 0,
                      "a filter list naming ANOTHER rigid changed the one-way infos");
        O.destroy(dev);
    }

    Bf.destroy(dev);

    // The filter is guarded by `!isDiffuse` (particleSystemMeshMidphase.cu:717), so
    // the DIFFUSE pass ignores the list entirely.  Run the same filter list with a
    // real diffuse side aliased onto the solid particles and require the diffuse
    // one-way output to match the UNFILTERED baseline, not the filtered one.
    {
        BdaOptions df = filt;
        df.enableCcd = false;          // the diffuse projection scale is then irrelevant
        df.maxDiffuse = Scene::NP;
        df.numDiffuse = Scene::NP;     // aliased positions: the same particles, undiffused
        BdaRig Dg, Bg;
        BdaOptions bg;                 // the unfiltered, CCD-off solid baseline
        bg.enableCcd = false;
        buildBda(dev, S, df, Dg);
        buildBda(dev, S, bg, Bg);
        {
            Kernel k("psMeshMidphaseBda.spv", "ps_meshCollisonLaunch", {}, sizeof(PsMeshMidBlob));
            k.dispatch(nbWarps, {}, Dg.blob, 1u, 2u);
            k.dispatch(nbWarps, {}, Bg.blob, 1u, 1u);
        }
        dev->flush();
        const auto dgc = downloadBuffer<uint32_t>(dev, Dg.bDiffCounts.h, S.nppad);
        const auto bgc = downloadBuffer<uint32_t>(dev, Bg.bCounts.h, S.nppad);
        const auto dgi = downloadBuffer<F4>(dev, Dg.bDiffInfos.h, size_t(S.nppad) * MAXPART);
        const auto bgi = downloadBuffer<F4>(dev, Bg.bInfos.h, size_t(S.nppad) * MAXPART);
        PXG_CHECK_MSG(std::memcmp(dgc.data(), bgc.data(), dgc.size() * 4) == 0,
                      "the DIFFUSE pass honoured the rigid filter; the CUDA guards find() "
                      "with !isDiffuseParticlesThread");
        PXG_CHECK_MSG(std::memcmp(dgi.data(), bgi.data(), dgi.size() * sizeof(F4)) == 0,
                      "the DIFFUSE pass's one-way infos differ from the unfiltered baseline");
        // ...and the SOLID half of the very same dispatch must still be filtered,
        // or "ignored the list" would be indistinguishable from "there was no list".
        const auto dgs = downloadBuffer<uint32_t>(dev, Dg.bCounts.h, S.nppad);
        size_t stillFiltered = 0;
        for (uint32_t i : filteredSorted) if (dgs[i] == 0u) ++stillFiltered;
        PXG_CHECK_MSG(stillFiltered == filteredSorted.size(),
                      "only %zu of %zu particles are filtered on the SOLID half of the "
                      "diffuse dispatch", stillFiltered, filteredSorted.size());
        Dg.destroy(dev);
        Bg.destroy(dev);
    }
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
    // EXACTLY a multiple of 32.  `(n + 31) / 32` and `(n + 32) / 32` differ ONLY
    // here, so without this size an off-by-one in the warp count is an identity --
    // which is how w1 survived the first sweep.
    runCoverageCase(dev, 32u * 1000u, 1u);
    // == 2 and == 3: the block stride is 2 and then 3 against a fixed iteration
    // stride of 1, so the factor is under test and an off-by-one in either the
    // warp count or the iteration count separates the covered SETS.
    runCoverageCase(dev, 32u * MESH_COLLISION_GRID + 1u, 2u);
    runCoverageCase(dev, 64u * MESH_COLLISION_GRID + 1u, 3u);
}
