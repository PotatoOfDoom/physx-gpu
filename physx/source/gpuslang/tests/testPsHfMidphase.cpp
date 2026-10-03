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

// testPsHfMidphase.cpp -- P9.11, the fixture for psHfMidphaseBda.slang
// (ps_heightfieldCollisonLaunch on the engine BDA ABI).
//
// THE ORACLE IS A SECOND GPU PROGRAM, exactly as in P9.10.  psHfCollide.slang is
// a Phase-1 port of the same body carrying its own host reference
// (psHfCollide_core, testPsHfCollide.cpp), and psHfMidphaseBda lifts that body and
// adds only engine glue.  So this file seeds ONE arena, uploads it twice -- once
// as the Phase-1 module's fifteen bound buffers, once as the BDA module's
// heightfield blob and descriptor chain -- runs both, and byte-compares.
//
// NOT ONE TOLERANCE.  Both modules are compiled without FMA contraction (psHfCollide
// JOINED PXG_NOCONTRACT_SHADERS in this tranche -- see the note there; P9.10 shipped
// a differential that disagreed at the ULP because psMeshMid had not), from
// arithmetic written the same way, on identical inputs.  They must agree bit for
// bit; a near miss is a finding about the lift, not rounding.
//
// The heightfield lift is smaller than the mesh lift and the fixture is shaped
// accordingly: there is NO filter test, because this kernel has no rigid filter at
// all (upstream candidate 24), and no warp-emulation surface to probe.  What it
// gains instead is a much richer DISPATCH MATRIX, because the Phase-1 fixture
// already established one and the quirks live there:
//
//   psHfMidphaseLayout   -- every engine constant and offset the shader spells,
//                           including PxBitAndByte's masking conversion checked by
//                           calling the engine's own operator.
//   psHfMidphase         -- the differential across eight dispatch shapes: static
//                           TGS+CCD, dynamic TGS+CCD, TRIPLE static (which is where
//                           upstream candidate 20's plain store becomes visible),
//                           non-CCD, PGS+CCD, a small writer capacity, the
//                           LoadShapePair swap both ways round, and a grid-width
//                           sweep for the grid-stride loop.
//   psHfMidphaseDiffuse  -- gridDim.z == 2, which the Phase-1 module cannot see at
//                           all because it has no diffuse arrays.
//   psHfMidphaseClobber  -- upstream candidate 20 on its own terms: a pair that
//                           finds nothing must still zero another producer's count.
//                           The Phase-1 module cannot express this (it has no cmIdx),
//                           so it is asserted directly against the transcribed
//                           semantics -- once as the invariant a concurrent second
//                           pair must satisfy under every interleaving, and once
//                           SEQUENCED over a pre-seeded counter, where the outcome is
//                           exact.
//   psHfMidphaseAccum    -- candidate 20(d) and the twelve-contact cap, with the
//                           one-way counter PRE-SEEDED so the atomicAdd's returned
//                           base is not zero.  One pair, so unlike the clobber test
//                           there is no race and the outcome is exact.
//   psHfMidphaseBoundLiteral -- P9.94c: the directed axis-aligned poses of
//                           pxgPsHfBoundCases.h, where the local bound's spelling
//                           decides the contact COUNT; the differential plus the
//                           literal reference's counts.

#include "pxgTestFramework.h"
#include "pxgPsHfBoundCases.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"     // PxgParticleSystem, PxgParticleContactWriter
#include "PxgContactManager.h"     // PxgContactManagerInput
#include "PxgConvexConvexShape.h"  // PxgShape
#include "PxgNpKernelIndices.h"
#include "PxgParticleSystemCoreKernelIndices.h"
#include "PxsTransformCache.h"
#include "PxNodeIndex.h"
#include "geometry/PxGeometry.h"
#include "geometry/PxHeightFieldSample.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
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

// psHfCollide.slang's PhMesh / PhParams / PsContact, byte for byte.
struct PhMesh { F4 hfP, hfQ, scale, misc; U4 dims, rigid; };
struct PhParams { uint32_t numParticles, maxContacts, flags, particleSystemId; };
struct PsC { F4 normal_pen; uint32_t rLo, rHi, pLo, pHi; };
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

// The 136-byte engine ABI blob: {1,4,8,8,8,8,8,8,8,72}.  Byte for byte the same
// shape as ps_meshCollisonLaunch's -- the two kernels have identical signatures.
struct PsHfMidBlob
{
    uint32_t isTGS;                            // @0   CUDA bool + 3 pad bytes
    float toleranceLength;                     // @4   present, never read (H3)
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
static_assert(sizeof(PsHfMidBlob) == 136, "{1,4,8,8,8,8,8,8,8,72} blob");
static_assert(offsetof(PsHfMidBlob, w_particleContacts) == 64, "the writer starts at 64");
static_assert(offsetof(PsHfMidBlob, w_maxContacts) == 128, "maxContacts @128");

static const uint32_t MAXMESH = 6u;    // PxgParticleContactInfo::MaxStaticContactsPerMesh
static const uint32_t MAXPART = 12u;   // ...::MaxStaticContactsPerParticle
static const uint32_t HOLE = 127u;     // PxHeightFieldMaterial::eHOLE
static const uint32_t POISON_U32 = 0xABABABABu;
static const uint32_t HF_BLOCK = 64u;  // PxgParticleSystemKernelBlockDim
static const uint32_t HF_GRID = 4096u; // PxgParticleSystemKernelGridDim

static PxgParticleSystem makeZeroPs()
{
    PxgParticleSystem ps;
    std::memset(&ps, 0, sizeof(ps));
    return ps;
}

static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 hQuatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}

// ---------------------------------------------------------------------------
// The shared scene.  Built once, uploaded two different ways.
//
// It is a deliberate near-copy of psHfCollide_core's, because that scene was
// already designed to reach the traps: a RECTANGULAR grid so any rows/columns
// transposition is visible, holes, tessellation flags, materialIndex1 values with
// the reserved high bit SET (so the seven-bit mask is load-bearing -- 0xFF must
// read as eHOLE), and particles that straddle and overshoot every grid edge.
// ---------------------------------------------------------------------------
struct Scene
{
    // constexpr, not `static const`: these get ODR-used by std::vector::push_back
    // (which binds a const reference), and a `static const` member with no
    // out-of-line definition then fails to link.
    static constexpr uint32_t ROWS = 24u;
    static constexpr uint32_t COLS = 17u;   // != ROWS on purpose
    static constexpr uint32_t NP = 640u;
    static constexpr uint32_t SYS = 6u;
    static constexpr uint32_t CAP = 4096u;

    std::vector<uint32_t> samples;      // ROWS * COLS packed PxHeightFieldSample
    std::vector<F4> curPos, predPos;
    PhMesh mesh{};
    float cdPs = 0.0f, cdHf = 0.0f;     // the two halves of contactDist
    uint32_t nbHoles = 0, nbTess = 0, nbReserved = 0;
};

static float hfHeightOf(const Scene& S, uint32_t v)
{
    return float(int32_t(S.samples[v] << 16) >> 16);
}

static void buildScene(Scene& S)
{
    std::mt19937 rng(93011);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    S.samples.resize(Scene::ROWS * Scene::COLS);
    for (uint32_t r = 0; r < Scene::ROWS; ++r)
        for (uint32_t c = 0; c < Scene::COLS; ++c)
        {
            const int16_t h = int16_t(40.0f * std::sin(r * 0.7f) * std::cos(c * 0.5f) + 25.0f * uv(rng));
            uint32_t m0 = (rng() % 11 == 3) ? HOLE : (rng() % 5);
            // materialIndex1 is also a PxBitAndByte, whose high bit the engine's
            // conversion strips.  Storing 0xFF (hole PLUS the reserved bit) and
            // flagged non-holes makes the seven-bit mask load-bearing: a port that
            // reads the raw byte sees 255, not 127, and treats the hole as solid.
            uint32_t m1 = (rng() % 11 == 7) ? (HOLE | 0x80u)
                        : (rng() % 13 == 5) ? (0x80u | (rng() % 5))
                                            : (rng() % 5);
            const uint32_t tess = (rng() % 2) ? 0x80u : 0u;
            if (m0 == HOLE) ++S.nbHoles;
            if ((m1 & 0x7Fu) == HOLE) ++S.nbHoles;
            if (tess) ++S.nbTess;
            if (m1 & 0x80u) ++S.nbReserved;
            S.samples[r * Scene::COLS + c] =
                uint32_t(uint16_t(h)) | ((m0 | tess) << 16) | (m1 << 24);
        }

    S.mesh.hfP = { 0.4f, -0.3f, 0.2f, 0 };
    {
        F4 q = { 0.15f, -0.2f, 0.1f, 1.0f };
        const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        S.mesh.hfQ = { q.x / m, q.y / m, q.z / m, q.w / m };
    }
    // The two contact distances are summed ON THE HOST in float and the sum handed
    // to the Phase-1 module, so `cd[ps] + cd[hf]` and `mesh.scale.w` are the same
    // BITS and not merely the same number.
    S.cdPs = 0.21f;
    S.cdHf = 0.14f;
    const float cd = S.cdPs + S.cdHf;
    S.mesh.scale = { 0.5f, 0.02f, 0.55f, cd };
    S.mesh.misc = { 0.04f, 0, 0, 0 };
    S.mesh.dims = { Scene::ROWS, Scene::COLS, 0, 0 };
    S.mesh.rigid = { 0x99u, 4u, 0, 0 };

    // The edge and off-grid cases are SCATTERED through the array (every ninth
    // particle), not parked in a block at the end.  With them at the end they are
    // always the LAST iteration of the grid-stride loop, and a mutant that turns
    // the early-out's `continue` into a `return` -- abandoning every remaining
    // particle for that thread -- loses nothing and survives.  Scattered, it does
    // not.  (`e4-earlyout-return` survived the first sweep on exactly this.)
    S.curPos.resize(Scene::NP);
    S.predPos.resize(Scene::NP);
    for (uint32_t p = 0; p < Scene::NP; ++p)
    {
        float lr = 1.0f + (Scene::ROWS - 3.0f) * u01(rng);
        float lc2 = 1.0f + (Scene::COLS - 3.0f) * u01(rng);
        if (p % 9u == 4u)
        {
            switch ((p / 9u) % 8u)
            {
            case 0: lr = -0.4f + 0.3f * u01(rng); break;                          // straddle row 0
            case 1: lr = (Scene::ROWS - 1.0f) + 0.1f + 0.3f * u01(rng); break;    // straddle row rows-1
            case 2: lc2 = -0.4f + 0.3f * u01(rng); break;                         // straddle col 0
            case 3: lc2 = (Scene::COLS - 1.0f) + 0.1f + 0.3f * u01(rng); break;   // straddle col cols-1
            case 4: lr = Scene::ROWS + 2.0f; break;                               // out +row: early-out
            case 5: lc2 = -3.0f; break;                                           // out -col: early-out
            case 6: lr = -0.4f + 0.3f * u01(rng);
                    lc2 = (Scene::COLS - 1.0f) + 0.1f + 0.3f * u01(rng); break;   // corner straddle
            default: lr = Scene::ROWS + 3.0f; lc2 = Scene::COLS + 3.0f; break;    // out both
            }
        }
        const uint32_t vr = uint32_t(std::min(std::max(lr, 0.0f), Scene::ROWS - 1.0f));
        const uint32_t vc = uint32_t(std::min(std::max(lc2, 0.0f), Scene::COLS - 1.0f));
        const float lh = hfHeightOf(S, vr * Scene::COLS + vc);
        const V3 lp = { lr * S.mesh.scale.x, (lh + 60.0f * uv(rng)) * S.mesh.scale.y,
                        lc2 * S.mesh.scale.z };
        const V3 wp = add(hQuatRotate(S.mesh.hfQ, lp), v3(S.mesh.hfP));
        S.curPos[p] = { wp.x, wp.y, wp.z, 1 };
        S.predPos[p] = { wp.x + 0.15f * uv(rng), wp.y + 0.15f * uv(rng),
                         wp.z + 0.15f * uv(rng), 1 };
    }
}

// The P9.94c directed scene: one pose of pxgPsHfBoundCases.h, its two particles
// alternating over all NP slots (so every grid-stride lane sees both), predicted ==
// current.  The heightfield is the header's ramp: rows 0-2 at 30/20/10, the rest 0,
// materials 0 and no tessellation flag.
static void buildBoundScene(Scene& S, const pxgpshfbound::Pose& pose)
{
    using namespace pxgpshfbound;
    static_assert(kRows == Scene::ROWS && kCols == Scene::COLS, "the bound cases use the Scene grid");
    auto fb = [](uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
    S.samples.assign(Scene::ROWS * Scene::COLS, 0u);
    for (uint32_t r = 0; r < 3u; ++r)
        for (uint32_t c = 0; c < Scene::COLS; ++c)
            S.samples[r * Scene::COLS + c] = uint32_t(uint16_t(kRowHeight[r]));
    S.mesh.hfP = { fb(pose.p[0]), fb(pose.p[1]), fb(pose.p[2]), 0 };
    S.mesh.hfQ = { fb(pose.q[0]), fb(pose.q[1]), fb(pose.q[2]), fb(pose.q[3]) };
    S.cdPs = fb(pose.cdPs);
    S.cdHf = fb(pose.cdHf);
    S.mesh.scale = { 1.0f, 1.0f, 1.0f, S.cdPs + S.cdHf };
    S.mesh.misc = { kRestDist, 0, 0, 0 };
    S.mesh.dims = { Scene::ROWS, Scene::COLS, 0, 0 };
    S.mesh.rigid = { 0x99u, 4u, 0, 0 };
    S.curPos.resize(Scene::NP);
    for (uint32_t i = 0; i < Scene::NP; ++i)
    {
        const Part& part = pose.part[i % 2u];
        S.curPos[i] = { fb(part.cur[0]), fb(part.cur[1]), fb(part.cur[2]), 1.0f };
    }
    S.predPos = S.curPos;
}

// ---------------------------------------------------------------------------
// Distinct, mutually different indices throughout, so a kernel that reads the
// wrong slot lands on a different number rather than on the right answer by luck.
// ---------------------------------------------------------------------------
static const uint32_t PS_SHAPE = 3u, HF_SHAPE = 5u;
static const uint32_t PS_CACHE = 2u, HF_CACHE = 7u;
static const uint32_t NB_SHAPES = 8u, NB_CACHE = 10u, NB_CM = 4u;
static const uint32_t CM_IDX = 0u;

struct BdaOptions
{
    bool psFirst = true;             // which half of the pair is the particle system
    bool staticRigid = true;
    bool enableCcd = true;
    bool isTgs = true;
    uint32_t maxContacts = Scene::CAP;
    uint32_t maxDiffuse = 0u;
    uint32_t numDiffuse = 0u;
    const std::vector<F4>* diffuseCur = nullptr;
    const std::vector<F4>* diffusePred = nullptr;
    const std::vector<F4>* solidCur = nullptr;
    const std::vector<F4>* solidPred = nullptr;
    uint32_t numSolid = Scene::NP;
    // A SECOND contact-manager entry, used only by psHfMidphaseClobber: it points
    // at a heightfield whose samples are far below the particles, so the pair finds
    // nothing -- and upstream candidate 20 says its plain store still lands.
    bool secondEmptyPair = false;
    // Which contact-manager slot carries the LIVE pair.  Everything before it is
    // filled with the sunken heightfield, so the dispatch runs gridDim.y = liveCm+1
    // slices and only the last one finds anything.  Its purpose is narrow and
    // important: restDistances is indexed by cmIdx while contactDistance is indexed
    // by the two CACHE REFS, so with the live pair at slot 0 a port that hard-codes
    // 0 for cmIdx -- or reads restDistances with a cache ref -- can still be right.
    // Only usable with a DYNAMIC rigid, because a static one would let the empty
    // pairs' candidate-20 plain store erase the live pair's count.
    uint32_t liveCm = 0u;
    // How many COLUMNS the one-way arrays are allocated with.  Twelve is what the
    // engine allocates (MaxStaticContactsPerParticle); psHfMidphaseAccum allocates
    // more and poisons the surplus, so a port that drops the twelve-contact cap
    // writes into observable memory instead of past the end of the array.
    uint32_t infoColumns = MAXPART;
    // Point the pair's SECOND half at the sunken heightfield, so this rig's only pair
    // finds nothing at all.  Combined with seedCounts that is candidate 20(a) with no
    // race left in it: a pair that produces nothing still stores ZERO over a count
    // some other producer built.
    bool sunkenPairOnly = false;
    // A PRE-SEEDED one-way contact count per particle.  In the engine that array is
    // an accumulator shared with ps_meshCollisonLaunch and ps_sdfMeshCollisonLaunch
    // (candidate 20b), so a heightfield pair routinely starts against a non-zero
    // count; seeding it directly is that state, not a contrivance.
    const std::vector<uint32_t>* seedCounts = nullptr;
};

struct BdaRig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bBlob, bBlob2, bCin, bShapes, bTc, bCd, bRd, bRemap, bDesc;
    DevBuf bCur, bPred, bInfos, bNodeIdx, bCounts;
    DevBuf bDiffCur, bDiffPred, bDiffInfos, bDiffNodeIdx, bDiffCounts, bNumDiffuse;
    DevBuf bContacts, bCount, bSbP, bTbP, bIbP, bBr, bTbr, bIbr;
    PsHfMidBlob blob{};

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

static void buildBda(PxgSlangComputeDevice* dev, const Scene& S, const BdaOptions& opt, BdaRig& R)
{
    // The heightfield blob: nbRows at [0], nbCols at [1], the packed samples from
    // [2] on (particleSystemHFMidPhaseCG.cu:148-150).
    std::vector<uint32_t> blob;
    blob.push_back(Scene::ROWS);
    blob.push_back(Scene::COLS);
    blob.insert(blob.end(), S.samples.begin(), S.samples.end());
    R.bBlob = R.mk(dev, blob);

    // The second, DELIBERATELY EMPTY heightfield: same dimensions, every sample
    // pushed far below anything the particles can reach, so its pairs generate no
    // contacts at all.
    const bool needSunken = opt.secondEmptyPair || opt.liveCm != 0u || opt.sunkenPairOnly;
    if (needSunken)
    {
        std::vector<uint32_t> blob2;
        blob2.push_back(Scene::ROWS);
        blob2.push_back(Scene::COLS);
        for (uint32_t i = 0; i < S.samples.size(); ++i)
        {
            // height -30000 (well below), materials 0 (never a hole), no tess flag,
            // so this heightfield is a solid floor a very long way down.
            blob2.push_back(uint32_t(uint16_t(int16_t(-30000))));
        }
        R.bBlob2 = R.mk(dev, blob2);
    }

    R.bCur = R.mk(dev, opt.solidCur ? *opt.solidCur : S.curPos);
    R.bPred = R.mk(dev, opt.solidPred ? *opt.solidPred : S.predPos);

    const uint32_t infoCols = opt.infoColumns;
    std::vector<F4> infos(size_t(Scene::NP) * infoCols, { -9, -9, -9, -9 });
    std::vector<uint32_t> nodeIdx(size_t(Scene::NP) * infoCols * 2u, 0xCCCCCCCCu);
    std::vector<uint32_t> counts(Scene::NP, 0u);
    if (opt.seedCounts) counts = *opt.seedCounts;
    R.bInfos = R.mk(dev, infos);
    R.bNodeIdx = R.mk(dev, nodeIdx);
    R.bCounts = R.mk(dev, counts);

    std::vector<PxgParticleSystem> descArr(Scene::SYS + 2u);
    std::memset(descArr.data(), 0xEE, descArr.size() * sizeof(PxgParticleSystem));
    PxgParticleSystem ps = makeZeroPs();
    ps.mCommonData.mNumParticles = opt.numSolid;
    ps.mCommonData.mMaxParticles = Scene::NP;
    ps.mCommonData.mMaxDiffuseParticles = opt.maxDiffuse;
    ps.mData.mFlags = opt.enableCcd ? PxU32(PxParticleFlag::eENABLE_SPECULATIVE_CCD) : 0u;
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(R.bCur.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bPred.addr));
    ps.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.bInfos.addr));
    ps.mOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.bNodeIdx.addr));
    ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.bCounts.addr));

    if (opt.maxDiffuse != 0u)
    {
        std::vector<F4> dInfos(size_t(Scene::NP) * MAXPART, { -9, -9, -9, -9 });
        std::vector<uint32_t> dNodeIdx(size_t(Scene::NP) * MAXPART * 2u, 0xCCCCCCCCu);
        std::vector<uint32_t> dCounts(Scene::NP, 0u);
        std::vector<uint32_t> nDiff = { opt.numDiffuse };
        R.bDiffInfos = R.mk(dev, dInfos);
        R.bDiffNodeIdx = R.mk(dev, dNodeIdx);
        R.bDiffCounts = R.mk(dev, dCounts);
        R.bNumDiffuse = R.mk(dev, nDiff);
        R.bDiffCur = R.mk(dev, opt.diffuseCur ? *opt.diffuseCur : S.curPos);
        R.bDiffPred = R.mk(dev, opt.diffusePred ? *opt.diffusePred : S.predPos);
        ps.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bNumDiffuse.addr));
        ps.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDiffCur.addr));
        ps.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bDiffPred.addr));
        ps.mDiffuseOneWayContactInfos =
            reinterpret_cast<PxgParticleContactInfo*>(size_t(R.bDiffInfos.addr));
        ps.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.bDiffNodeIdx.addr));
        ps.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.bDiffCounts.addr));
    }
    descArr[Scene::SYS] = ps;
    R.bDesc = R.mk(dev, descArr);

    const uint32_t liveCm = opt.liveCm;
    std::vector<PxgContactManagerInput> cmInputs(NB_CM);
    std::memset(cmInputs.data(), 0xEE, cmInputs.size() * sizeof(PxgContactManagerInput));
    if (opt.psFirst)
    {
        cmInputs[liveCm].shapeRef0 = PS_SHAPE;
        cmInputs[liveCm].transformCacheRef0 = PS_CACHE;
        cmInputs[liveCm].shapeRef1 = HF_SHAPE;
        cmInputs[liveCm].transformCacheRef1 = HF_CACHE;
    }
    else
    {
        cmInputs[liveCm].shapeRef0 = HF_SHAPE;
        cmInputs[liveCm].transformCacheRef0 = HF_CACHE;
        cmInputs[liveCm].shapeRef1 = PS_SHAPE;
        cmInputs[liveCm].transformCacheRef1 = PS_CACHE;
    }
    if (opt.sunkenPairOnly)
    {
        // Still a real particle-system/heightfield pair -- it just meets a heightfield
        // 30000 units down, so it runs in full and produces nothing.
        if (opt.psFirst)
        {
            cmInputs[liveCm].shapeRef1 = HF_SHAPE + 1u;
            cmInputs[liveCm].transformCacheRef1 = HF_CACHE + 1u;
        }
        else
        {
            cmInputs[liveCm].shapeRef0 = HF_SHAPE + 1u;
            cmInputs[liveCm].transformCacheRef0 = HF_CACHE + 1u;
        }
    }
    // Every slot BEFORE the live one points at the sunken heightfield, so those
    // dispatch slices run in full and find nothing.
    for (uint32_t i = 0; i < liveCm; ++i)
    {
        cmInputs[i].shapeRef0 = PS_SHAPE;
        cmInputs[i].transformCacheRef0 = PS_CACHE;
        cmInputs[i].shapeRef1 = HF_SHAPE + 1u;
        cmInputs[i].transformCacheRef1 = HF_CACHE + 1u;
    }
    if (opt.secondEmptyPair)
    {
        // Pair 1: the same particle system against the sunken heightfield, on its
        // own shape and cache slots so nothing about pair 0 changes.
        cmInputs[1].shapeRef0 = PS_SHAPE;
        cmInputs[1].transformCacheRef0 = PS_CACHE;
        cmInputs[1].shapeRef1 = HF_SHAPE + 1u;
        cmInputs[1].transformCacheRef1 = HF_CACHE + 1u;
    }
    R.bCin = R.mk(dev, cmInputs);

    std::vector<PxgShape> shapes(NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    // Every decoy is also a heightfield, with a different scale and a null blob, so
    // a kernel that picks the wrong slot gets a shape whose type check passes and
    // whose geometry pointer is zero -- a fault rather than a near miss.
    for (uint32_t i = 0; i < NB_SHAPES; ++i)
    {
        shapes[i].type = PxU32(PxGeometryType::eHEIGHTFIELD);
        shapes[i].scale = PxMeshScale(PxVec3(3.0f, 3.0f, 3.0f), PxQuat(PxIdentity));
    }
    shapes[HF_SHAPE].type = PxU32(PxGeometryType::eHEIGHTFIELD);
    shapes[HF_SHAPE].scale = PxMeshScale(PxVec3(S.mesh.scale.x, S.mesh.scale.y, S.mesh.scale.z),
                                         PxQuat(PxIdentity));
    shapes[HF_SHAPE].hullOrMeshPtr = size_t(R.bBlob.addr);
    shapes[PS_SHAPE].type = PxU32(PxGeometryType::ePARTICLESYSTEM);
    shapes[PS_SHAPE].particleOrSoftbodyId = Scene::SYS;
    if (needSunken)
    {
        shapes[HF_SHAPE + 1u].type = PxU32(PxGeometryType::eHEIGHTFIELD);
        shapes[HF_SHAPE + 1u].scale = shapes[HF_SHAPE].scale;
        shapes[HF_SHAPE + 1u].hullOrMeshPtr = size_t(R.bBlob2.addr);
    }
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> tcache(NB_CACHE);
    std::memset(tcache.data(), 0xEE, tcache.size() * sizeof(PxsCachedTransform));
    for (uint32_t i = 0; i < NB_CACHE; ++i)
        tcache[i].transform = PxTransform(PxVec3(-9.0f - float(i), 4.0f, 3.0f), PxQuat(PxIdentity));
    tcache[HF_CACHE].transform = PxTransform(PxVec3(S.mesh.hfP.x, S.mesh.hfP.y, S.mesh.hfP.z),
                                             PxQuat(S.mesh.hfQ.x, S.mesh.hfQ.y,
                                                    S.mesh.hfQ.z, S.mesh.hfQ.w));
    if (needSunken)
        tcache[HF_CACHE + 1u].transform = tcache[HF_CACHE].transform;
    R.bTc = R.mk(dev, tcache);

    std::vector<float> contactDist(NB_CACHE);
    for (uint32_t i = 0; i < NB_CACHE; ++i) contactDist[i] = 2.3f + 0.41f * float(i);
    contactDist[PS_CACHE] = S.cdPs;
    contactDist[HF_CACHE] = S.cdHf;
    if (needSunken) contactDist[HF_CACHE + 1u] = S.cdHf;
    R.bCd = R.mk(dev, contactDist);

    // Every OTHER rest distance is distinct and wildly wrong, so reading the array
    // with the wrong index changes the penetration by a visible amount.
    std::vector<float> restDist(NB_CM);
    for (uint32_t i = 0; i < NB_CM; ++i) restDist[i] = -3.5f - 0.9f * float(i);
    restDist[liveCm] = S.mesh.misc.x;
    if (opt.secondEmptyPair) restDist[1] = S.mesh.misc.x;
    R.bRd = R.mk(dev, restDist);

    // Indexed by the HEIGHTFIELD's cache ref.
    std::vector<uint64_t> remap(NB_CACHE, 0ull);
    for (uint32_t i = 0; i < NB_CACHE; ++i)
        remap[i] = (uint64_t(0x6160u + i) << 32) | uint64_t(0xA00u + i);
    {
        const PxNodeIndex ni = opt.staticRigid ? PxNodeIndex(PX_INVALID_NODE, 0u)
                                               : PxNodeIndex(0x99u, 4u);
        remap[HF_CACHE] = ni.getInd();
        if (needSunken) remap[HF_CACHE + 1u] = ni.getInd();
    }
    R.bRemap = R.mk(dev, remap);

    std::vector<PsC> contacts(Scene::CAP);
    std::memset(contacts.data(), 0xAB, contacts.size() * sizeof(PsC));
    std::vector<uint32_t> count = { 0u };
    std::vector<uint64_t> u64Poison(Scene::CAP, 0xABABABABABABABABull);
    std::vector<uint32_t> u32Poison(Scene::CAP, POISON_U32);
    R.bContacts = R.mk(dev, contacts);
    R.bCount = R.mk(dev, count);
    R.bSbP = R.mk(dev, u64Poison);
    R.bTbP = R.mk(dev, u32Poison);
    R.bIbP = R.mk(dev, u32Poison);
    R.bBr = R.mk(dev, u64Poison);
    R.bTbr = R.mk(dev, u32Poison);
    R.bIbr = R.mk(dev, u32Poison);

    // The CUDA bool is ONE BYTE at offset 0 with three bytes of launch-blob padding
    // behind it.  Poisoning them is what makes the masked read observable.
    R.blob.isTGS = 0xABABAB00u | (opt.isTgs ? 1u : 0u);
    R.blob.toleranceLength = -4321.5f;   // H3: never read
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
    PxgBufferHandle bM = 0, bS = 0, bCur = 0, bPred = 0, bI = 0, bNi = 0, bC = 0;
    PxgBufferHandle bW = 0, bBR = 0, bTBR = 0, bIBR = 0, bSP = 0, bTBP = 0, bIBP = 0, bWc = 0;
    PhParams prm{};

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
        return { bM, bS, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc };
    }
};

static void buildRef(PxgSlangComputeDevice* dev, const Scene& S, const BdaOptions& opt, RefRig& R)
{
    PhMesh mesh = S.mesh;
    {
        const PxNodeIndex ni = opt.staticRigid ? PxNodeIndex(PX_INVALID_NODE, 0u)
                                               : PxNodeIndex(0x99u, 4u);
        mesh.rigid = { uint32_t(ni.getInd() & 0xFFFFFFFFull), uint32_t(ni.getInd() >> 32), 0, 0 };
    }
    std::vector<PhMesh> meshV = { mesh };

    R.bM = R.mk(dev, meshV);
    R.bS = R.mk(dev, S.samples);
    R.bCur = R.mk(dev, opt.solidCur ? *opt.solidCur : S.curPos);
    R.bPred = R.mk(dev, opt.solidPred ? *opt.solidPred : S.predPos);

    std::vector<F4> infos(size_t(Scene::NP) * MAXPART, { -9, -9, -9, -9 });
    std::vector<uint32_t> nodeIdx(size_t(Scene::NP) * MAXPART * 2u, 0xCCCCCCCCu);
    std::vector<uint32_t> counts(Scene::NP, 0u);
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
    static const std::vector<uint32_t> s = { uint32_t(sizeof(PhMesh)), 4, 16, 16, 16, 8, 4,
                                             uint32_t(sizeof(PsC)), 8, 4, 4, 8, 4, 4, 4 };
    return s;
}

// The writer's global start indices race between groups, so the ARRAY order is
// not deterministic -- but a particle's own contacts are written sequentially by
// one thread from one base, so grouping by particle index recovers a canonical
// order both programs must agree on exactly.
static std::vector<std::vector<PsC>> groupByParticle(const std::vector<PsC>& raw, uint32_t n)
{
    std::vector<std::vector<PsC>> out(Scene::NP + 1u);
    for (uint32_t i = 0; i < n && i < raw.size(); ++i)
    {
        const uint32_t p = raw[i].pLo;
        out[p <= Scene::NP ? p : Scene::NP].push_back(raw[i]);
    }
    return out;
}

static bool sameContact(const PsC& a, const PsC& b)
{
    return std::memcmp(&a, &b, sizeof(PsC)) == 0;
}

} // namespace

// ===========================================================================
PXG_TEST(psHfMidphaseLayout)
{
    // The ABI blob -- identical in shape to the mesh twin's, which is itself the
    // claim being pinned: the two kernels take the same ten arguments.
    PXG_CHECK_MSG(sizeof(PxgParticleContactWriter) == 72,
                  "PxgParticleContactWriter must be 72 bytes (got %zu)",
                  sizeof(PxgParticleContactWriter));
    PXG_CHECK(offsetof(PxgParticleContactWriter, particleContacts) == 0);
    PXG_CHECK(offsetof(PxgParticleContactWriter, numTotalContacts) == 8);
    PXG_CHECK(offsetof(PxgParticleContactWriter, maxContacts) == 64);

    PXG_CHECK(sizeof(PxgContactManagerInput) == 16);
    PXG_CHECK(offsetof(PxgContactManagerInput, shapeRef0) == 0);
    PXG_CHECK(offsetof(PxgContactManagerInput, transformCacheRef0) == 8);
    PXG_CHECK_MSG(sizeof(PxgShape) == 48, "PxgShape stride (got %zu)", sizeof(PxgShape));
    PXG_CHECK(offsetof(PxgShape, hullOrMeshPtr) == 32);
    PXG_CHECK(offsetof(PxgShape, type) == 40);
    PXG_CHECK(offsetof(PxgShape, particleOrSoftbodyId) == 44);
    PXG_CHECK(sizeof(PxsCachedTransform) == 32);

    // The two geometry types this kernel keys on.  eHEIGHTFIELD is 9; the draft of
    // the shader had 5, which is eCONVEXMESH -- an off-by-four in a 0-based enum is
    // exactly the sort of thing only a pinned check catches.
    PXG_CHECK_MSG(PxU32(PxGeometryType::ePARTICLESYSTEM) == 6u,
                  "ePARTICLESYSTEM == 6 (got %u)", unsigned(PxGeometryType::ePARTICLESYSTEM));
    PXG_CHECK_MSG(PxU32(PxGeometryType::eHEIGHTFIELD) == 9u,
                  "eHEIGHTFIELD == 9 (got %u)", unsigned(PxGeometryType::eHEIGHTFIELD));
    PXG_CHECK_MSG(PxU32(PxGeometryType::eCONVEXMESH) == 5u,
                  "eCONVEXMESH == 5, which is what makes 5 a trap for eHEIGHTFIELD");

    // ---- THE SAMPLE FORMAT (H7, H8), checked by calling the engine itself ----
    PXG_CHECK_MSG(sizeof(PxHeightFieldSample) == 4,
                  "PxHeightFieldSample must be 4 bytes (got %zu)", sizeof(PxHeightFieldSample));
    PXG_CHECK(offsetof(PxHeightFieldSample, height) == 0);
    PXG_CHECK(offsetof(PxHeightFieldSample, materialIndex0) == 2);
    PXG_CHECK(offsetof(PxHeightFieldSample, materialIndex1) == 3);
    PXG_CHECK_MSG(PxU32(PxHeightFieldMaterial::eHOLE) == HOLE,
                  "eHOLE == 127 (got %u)", unsigned(PxHeightFieldMaterial::eHOLE));
    {
        // PxBitAndByte's conversion operator STRIPS the high bit.  A hole that also
        // carries the tessellation flag must still READ as eHOLE -- this is the
        // single most dangerous trap in the kernel, because a raw-byte port sees
        // 255, fails the `!= eHOLE` test, and generates contacts through the hole.
        PxHeightFieldSample s;
        s.height = PxI16(-1234);
        s.materialIndex0 = PxBitAndByte(PxU8(PxHeightFieldMaterial::eHOLE), /*bit*/ true);
        s.materialIndex1 = PxBitAndByte(PxU8(PxHeightFieldMaterial::eHOLE), /*bit*/ true);
        PXG_CHECK_MSG(PxU32(s.materialIndex0) == HOLE,
                      "materialIndex0 with the tess bit set must read as eHOLE (got %u)",
                      unsigned(PxU32(s.materialIndex0)));
        PXG_CHECK_MSG(PxU32(s.materialIndex1) == HOLE,
                      "materialIndex1 with its reserved bit set must read as eHOLE (got %u)",
                      unsigned(PxU32(s.materialIndex1)));
        PXG_CHECK_MSG(s.tessFlag() == 1, "the tess flag is materialIndex0's high bit");

        // ...and the shader's packing of the same sample into one uint must decode
        // back to the same three fields.  This is the exact expression the shader
        // uses, checked against the engine's own struct through its raw bytes.
        uint32_t packed;
        std::memcpy(&packed, &s, 4);
        PXG_CHECK_MSG(int32_t(packed << 16) >> 16 == -1234,
                      "the shader's sign-extending height read must recover PxI16");
        PXG_CHECK_MSG(((packed >> 16) & 0x7Fu) == HOLE, "materialIndex0 is bits 16..22");
        PXG_CHECK_MSG(((packed >> 24) & 0x7Fu) == HOLE, "materialIndex1 is bits 24..30");
        PXG_CHECK_MSG((packed & 0x800000u) != 0u, "the tess flag is bit 23");
    }

    // The two contact caps.
    PXG_CHECK_MSG(PxgParticleContactInfo::MaxStaticContactsPerMesh == MAXMESH,
                  "MaxStaticContactsPerMesh == 6 (got %u)",
                  unsigned(PxgParticleContactInfo::MaxStaticContactsPerMesh));
    PXG_CHECK_MSG(PxgParticleContactInfo::MaxStaticContactsPerParticle == MAXPART,
                  "MaxStaticContactsPerParticle == 12 (got %u)",
                  unsigned(PxgParticleContactInfo::MaxStaticContactsPerParticle));
    // Candidate 21 is unreachable exactly because of this inequality; if it ever
    // stops holding, the writer loop's src = 0 restart starts duplicating.
    PXG_CHECK_MSG(PxgParticleContactInfo::MaxStaticContactsPerMesh
                      <= PxgParticleContactInfo::MaxStaticContactsPerParticle,
                  "upstream candidate 21 becomes REACHABLE if the per-mesh cap exceeds "
                  "the per-particle cap");

    // The launch shape, which the shader spells as compile-time constants.
    PXG_CHECK_MSG(PxgParticleSystemKernelGridDim::PS_HEIGHTFIELD_COLLISION == HF_GRID,
                  "PS_HEIGHTFIELD_COLLISION grid == 4096 (got %u)",
                  unsigned(PxgParticleSystemKernelGridDim::PS_HEIGHTFIELD_COLLISION));
    PXG_CHECK_MSG(PxgParticleSystemKernelBlockDim::PS_HEIGHTFIELD_COLLISION == HF_BLOCK,
                  "PS_HEIGHTFIELD_COLLISION block == 64 (got %u)",
                  unsigned(PxgParticleSystemKernelBlockDim::PS_HEIGHTFIELD_COLLISION));
    // It is NOT the mesh kernel's shape, and the shader must not inherit that one.
    PXG_CHECK_MSG(PxgParticleSystemKernelBlockDim::PS_HEIGHTFIELD_COLLISION
                      != PxgParticleSystemKernelBlockDim::PS_MESH_COLLISION,
                  "the heightfield and mesh kernels have different block widths");

    PXG_CHECK_MSG(PxU32(PxParticleFlag::eENABLE_SPECULATIVE_CCD) == 8u,
                  "eENABLE_SPECULATIVE_CCD == 1<<3 (got %u)",
                  unsigned(PxParticleFlag::eENABLE_SPECULATIVE_CCD));

    // The descriptor fields the glue reads.
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedOriginPos_InvMass) == 392);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPositions_InvMass) == 24);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactInfos) == 200);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayNodeIndex) == 216);
    PXG_CHECK(offsetof(PxgParticleSystem, mOneWayContactCount) == 232);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseOneWayContactInfos) == 592);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseOneWayNodeIndex) == 608);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseOneWayContactCount) == 616);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) == 552);
    PXG_CHECK(offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) == 560);
    PXG_CHECK(offsetof(PxgParticleSystem, mNumDiffuseParticles) == 488);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mNumParticles) == 116);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mMaxDiffuseParticles) == 128);
    PXG_CHECK(offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mFlags) == 312);

    {
        const PxNodeIndex stat(PX_INVALID_NODE, 0u);
        const PxNodeIndex dyn(0x99u, 4u);
        PXG_CHECK(stat.isStaticBody());
        PXG_CHECK(!dyn.isStaticBody());
    }
}

// ===========================================================================
// The differential.
// ===========================================================================
PXG_TEST(psHfMidphase)
{
    PxgSlangComputeDevice* dev = testDevice();

    Scene S;
    buildScene(S);
    PXG_CHECK_MSG(Scene::ROWS != Scene::COLS,
                  "the grid must be rectangular or a rows/columns transposition is invisible");
    PXG_CHECK_MSG(S.nbHoles > 20 && S.nbTess > 100 && S.nbReserved > 10,
                  "the scene is thin: %u holes, %u tess flags, %u reserved-bit materials",
                  S.nbHoles, S.nbTess, S.nbReserved);

    const uint32_t refGroups = (Scene::NP + 255u) / 256u;

    struct Variant { const char* name; BdaOptions opt; uint32_t reps; uint32_t bdaGrid; };
    std::vector<Variant> variants;
    {
        BdaOptions o;                       variants.push_back({ "static-tgs-ccd", o, 1u, 64u });
        BdaOptions d; d.staticRigid = false; variants.push_back({ "dynamic-tgs-ccd", d, 1u, 64u });
        // TRIPLE static: the same buffers three times.  This is where upstream
        // candidate 20's plain store shows -- the count does NOT reach 3x, it comes
        // back as the single-pass count, and both programs must agree on that.
        BdaOptions t;                       variants.push_back({ "triple-static", t, 3u, 64u });
        BdaOptions n; n.enableCcd = false;  variants.push_back({ "non-ccd", n, 1u, 64u });
        BdaOptions p; p.isTgs = false;      variants.push_back({ "pgs-ccd", p, 1u, 64u });
        BdaOptions c; c.staticRigid = false; c.maxContacts = 96u;
                                            variants.push_back({ "small-writer-cap", c, 1u, 64u });
        BdaOptions s; s.psFirst = false;    variants.push_back({ "pair-swapped", s, 1u, 64u });
        // Grid widths: the BDA kernel's stride is WorkgroupCount().x * 64, so every
        // width must cover every particle.  1 group is the extreme case (one block
        // walking all 640 particles); 4096 is the real launch.
        BdaOptions g1;                      variants.push_back({ "grid-1", g1, 1u, 1u });
        BdaOptions g2;                      variants.push_back({ "grid-3", g2, 1u, 3u });
        BdaOptions g3;                      variants.push_back({ "grid-real-4096", g3, 1u, HF_GRID });
        // cmIdx != 0: the live pair sits at slot 2 and slots 0/1 are the sunken
        // heightfield.  DYNAMIC, so the empty slices' candidate-20 plain store
        // cannot erase anything (there is none on the dynamic path).  This is the
        // only dispatch in the file that proves cmIdx is read at all -- and that
        // restDistances is indexed BY IT and not by a cache ref.
        BdaOptions m; m.staticRigid = false; m.liveCm = 2u;
                                            variants.push_back({ "cmidx-2-dynamic", m, 1u, 64u });
    }

    size_t anyContacts = 0, anyCapped = 0;
    // The uncapped dynamic run's contact multiset, kept as sorted 32-byte blobs.
    // The capped run below is checked as a SUBSET of it rather than against the
    // reference contact-for-contact -- see the note at the capacity check.
    std::vector<std::array<uint8_t, sizeof(PsC)>> universe;
    for (const Variant& v : variants)
    {
        BdaRig A;
        RefRig B;
        buildBda(dev, S, v.opt, A);
        buildRef(dev, S, v.opt, B);
        {
            Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
            for (uint32_t r = 0; r < v.reps; ++r)
                k.dispatch(v.bdaGrid, {}, A.blob, v.opt.liveCm + 1u, 1u);
        }
        {
            Kernel k("psHfCollide.spv", "ctPsHfCollide", refStrides(), sizeof(PhParams));
            for (uint32_t r = 0; r < v.reps; ++r)
                k.dispatch(refGroups, B.bindings(), B.prm);
        }
        dev->flush();

        const auto aI = downloadBuffer<F4>(dev, A.bInfos.h, size_t(Scene::NP) * MAXPART);
        const auto bI = downloadBuffer<F4>(dev, B.bI, size_t(Scene::NP) * MAXPART);
        const auto aC = downloadBuffer<uint32_t>(dev, A.bCounts.h, Scene::NP);
        const auto bC = downloadBuffer<uint32_t>(dev, B.bC, Scene::NP);
        const auto aN = downloadBuffer<uint32_t>(dev, A.bNodeIdx.h, size_t(Scene::NP) * MAXPART * 2u);
        const auto bN = downloadBuffer<uint32_t>(dev, B.bNi, size_t(Scene::NP) * MAXPART * 2u);

        PXG_CHECK_MSG(std::memcmp(aC.data(), bC.data(), aC.size() * 4) == 0,
                      "%s: the one-way COUNTS differ", v.name);
        PXG_CHECK_MSG(std::memcmp(aI.data(), bI.data(), aI.size() * sizeof(F4)) == 0,
                      "%s: the one-way CONTACT INFOS differ", v.name);
        PXG_CHECK_MSG(std::memcmp(aN.data(), bN.data(), aN.size() * 4) == 0,
                      "%s: the one-way NODE INDICES differ", v.name);

        const uint32_t aW = downloadBuffer<uint32_t>(dev, A.bCount.h, 1)[0];
        const uint32_t bW = downloadBuffer<uint32_t>(dev, B.bWc, 1)[0];
        PXG_CHECK_MSG(aW == bW, "%s: writer counters differ (%u vs %u)", v.name, aW, bW);

        const auto aCt = downloadBuffer<PsC>(dev, A.bContacts.h, Scene::CAP);
        const bool capBinds = v.opt.maxContacts < Scene::CAP;
        if (!capBinds)
        {
            const auto bCt = downloadBuffer<PsC>(dev, B.bW, Scene::CAP);
            const auto ga = groupByParticle(aCt, std::min(aW, Scene::CAP));
            const auto gb = groupByParticle(bCt, std::min(bW, Scene::CAP));
            size_t mism = 0;
            for (uint32_t p = 0; p <= Scene::NP; ++p)
            {
                if (ga[p].size() != gb[p].size()) { ++mism; continue; }
                for (size_t i = 0; i < ga[p].size(); ++i)
                    if (!sameContact(ga[p][i], gb[p][i])) { ++mism; break; }
            }
            PXG_CHECK_MSG(mism == 0, "%s: %zu particles' writer contacts differ", v.name, mism);

            // The five sibling arrays writeContact fills beside the contact record.
            // Their absolute ORDER races between groups, but each entry's relation
            // to its own slot does not, so these are order-free invariants -- and
            // without them a port that writes a constant, or the wrong half of a
            // packed id, is invisible (w11 and w12 survived the first sweep on it).
            if (aW > 0u && aW <= Scene::CAP)
            {
                const auto aIdxR = downloadBuffer<uint32_t>(dev, A.bIbr.h, Scene::CAP);
                const auto aIdxP = downloadBuffer<uint32_t>(dev, A.bIbP.h, Scene::CAP);
                const auto aTbP = downloadBuffer<uint32_t>(dev, A.bTbP.h, Scene::CAP);
                const auto aTbR = downloadBuffer<uint32_t>(dev, A.bTbr.h, Scene::CAP);
                const auto aBr = downloadBuffer<uint64_t>(dev, A.bBr.h, Scene::CAP);
                const auto aSbP = downloadBuffer<uint64_t>(dev, A.bSbP.h, Scene::CAP);
                const PxNodeIndex rid = v.opt.staticRigid ? PxNodeIndex(PX_INVALID_NODE, 0u)
                                                          : PxNodeIndex(0x99u, 4u);
                size_t bad = 0;
                const uint32_t n = std::min(aW, v.opt.maxContacts);
                for (uint32_t i = 0; i < n; ++i)
                {
                    if (aIdxR[i] != i) ++bad;
                    if (aIdxP[i] != i) ++bad;
                    if (aTbP[i] != aCt[i].pLo) ++bad;
                    if (aTbR[i] != uint32_t(rid.getInd() & 0xFFFFFFFFull)) ++bad;
                    if (aBr[i] != rid.getInd()) ++bad;
                    if (aSbP[i] != (uint64_t(aCt[i].pHi) << 32 | uint64_t(aCt[i].pLo))) ++bad;
                }
                PXG_CHECK_MSG(bad == 0,
                              "%s: %zu writer sibling-array entries disagree with their own "
                              "contact record", v.name, bad);
            }

            // Stash the uncapped DYNAMIC set as the universe the capped run must
            // draw from.  Every variant that reaches the writer has the same true
            // contact set; only how many of them fit differs.
            if (!v.opt.staticRigid && universe.empty())
            {
                for (uint32_t i = 0; i < std::min(aW, Scene::CAP); ++i)
                {
                    std::array<uint8_t, sizeof(PsC)> blob{};
                    std::memcpy(blob.data(), &aCt[i], sizeof(PsC));
                    universe.push_back(blob);
                }
                std::sort(universe.begin(), universe.end());
            }
        }
        else
        {
            // WHY THE CONTENTS ARE NOT COMPARED HERE, and it is not a weakening.
            // With a capacity below the total, both programs reserve from one shared
            // counter and write only the slots that land under it -- so WHICH
            // contacts survive depends on the order the atomicAdds resolve, which
            // depends on the launch geometry.  The two programs deliberately have
            // different geometry (64-wide grid-strided blocks against 256-wide
            // one-shot ones), so demanding the same survivors is demanding an
            // accident.  The COUNTER is still compared exactly, above.
            //
            // What is checked instead is stronger than nothing and is order-free:
            // every contact that did land must be one the uncapped run also
            // produced, and nothing past the capacity may be touched at all.
            PXG_CHECK_MSG(!universe.empty(),
                          "%s: the uncapped dynamic run must precede the capped one", v.name);
            size_t alien = 0;
            for (uint32_t i = 0; i < v.opt.maxContacts && i < Scene::CAP; ++i)
            {
                std::array<uint8_t, sizeof(PsC)> blob{};
                std::memcpy(blob.data(), &aCt[i], sizeof(PsC));
                if (!std::binary_search(universe.begin(), universe.end(), blob)) ++alien;
            }
            PXG_CHECK_MSG(alien == 0,
                          "%s: %zu of the %u written contacts are not in the uncapped run's set",
                          v.name, alien, v.opt.maxContacts);
        }

        // Per-variant sanity on the DESIGN, not on the port.
        size_t live = 0, capped = 0;
        for (uint32_t p = 0; p < Scene::NP; ++p)
        {
            if (aC[p] != 0u) ++live;
            if (aC[p] == MAXMESH) ++capped;
        }
        if (v.opt.staticRigid)
        {
            PXG_CHECK_MSG(live > 100, "%s: only %zu particles found contacts", v.name, live);
            anyContacts += live;
            anyCapped += capped;
            PXG_CHECK_MSG(aW == 0u,
                          "%s: the static path reached the writer (%u); H13 says remaining "
                          "is always zero for a static rigid", v.name, aW);
        }
        else
        {
            size_t touched = 0;
            for (uint32_t p = 0; p < Scene::NP; ++p) if (aC[p] != 0u) ++touched;
            PXG_CHECK_MSG(touched == 0,
                          "%s: %zu one-way counters were written under a DYNAMIC rigid",
                          v.name, touched);
            PXG_CHECK_MSG(aW > 0u, "%s: the dynamic path wrote nothing", v.name);
            if (v.opt.maxContacts < Scene::CAP)
            {
                PXG_CHECK_MSG(aW > v.opt.maxContacts,
                              "%s: the counter (%u) did not pass the capacity (%u), so "
                              "writeContact's bound is not under test",
                              v.name, aW, v.opt.maxContacts);
                const auto aIdx = downloadBuffer<uint32_t>(dev, A.bIbP.h, Scene::CAP);
                size_t past = 0;
                for (uint32_t i = v.opt.maxContacts; i < Scene::CAP; ++i)
                    if (aIdx[i] != POISON_U32) ++past;
                PXG_CHECK_MSG(past == 0,
                              "%s: %zu writer slots beyond maxContacts were written", v.name, past);
            }
        }

        // Candidate 20, seen from the differential: three identical static passes
        // must NOT leave a count of three times one pass.
        if (v.reps == 3u)
        {
            size_t over = 0;
            for (uint32_t p = 0; p < Scene::NP; ++p)
                if (aC[p] > MAXMESH) ++over;
            PXG_CHECK_MSG(over == 0,
                          "%s: %zu particles carry a count above MaxStaticContactsPerMesh; "
                          "upstream candidate 20's plain store means three passes leave the "
                          "SAME count as one, not three times it", v.name, over);
        }

        A.destroy(dev);
        B.destroy(dev);
    }

    PXG_CHECK_MSG(anyCapped >= 3,
                  "no particle saturated the six-slot list across the static variants "
                  "(%zu), so H11's drop-past-six is untested", anyCapped);
}

// ===========================================================================
// gridDim.z == 2 -- the diffuse split.  psHfCollide has diffuse FLAGS but no
// diffuse ARRAYS, so the differential is blind to the selection and this needs
// its own test.
// ===========================================================================
PXG_TEST(psHfMidphaseDiffuse)
{
    PxgSlangComputeDevice* dev = testDevice();

    Scene S;
    buildScene(S);
    const uint32_t GRID = 64u;

    // (a) maxDiffuse == 0: the diffuse pass must take H2's early-out and leave the
    //     solid result exactly as a z=1 dispatch does.  This is the assertion that
    //     kills a port ignoring gridDim.z -- it would run the solid pass twice.
    {
        BdaOptions o; o.enableCcd = false;
        BdaRig Z1, Z2;
        buildBda(dev, S, o, Z1);
        buildBda(dev, S, o, Z2);
        {
            Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
            k.dispatch(GRID, {}, Z1.blob, 1u, 1u);
            k.dispatch(GRID, {}, Z2.blob, 1u, 2u);
        }
        dev->flush();
        const auto c1 = downloadBuffer<uint32_t>(dev, Z1.bCounts.h, Scene::NP);
        const auto c2 = downloadBuffer<uint32_t>(dev, Z2.bCounts.h, Scene::NP);
        const auto i1 = downloadBuffer<F4>(dev, Z1.bInfos.h, size_t(Scene::NP) * MAXPART);
        const auto i2 = downloadBuffer<F4>(dev, Z2.bInfos.h, size_t(Scene::NP) * MAXPART);
        PXG_CHECK_MSG(std::memcmp(c1.data(), c2.data(), c1.size() * 4) == 0,
                      "gridDim.z == 2 changed the solid one-way counts; the second z slice "
                      "is the diffuse pass and with no diffuse particles it must return");
        PXG_CHECK_MSG(std::memcmp(i1.data(), i2.data(), i1.size() * sizeof(F4)) == 0,
                      "gridDim.z == 2 changed the solid one-way infos");
        size_t live = 0;
        for (uint32_t p = 0; p < Scene::NP; ++p) if (c1[p] != 0u) ++live;
        PXG_CHECK_MSG(live > 100, "the diffuse test ran on an empty solid result (%zu)", live);
        Z1.destroy(dev);
        Z2.destroy(dev);
    }

    // (b) a REAL diffuse side with its OWN positions and its OWN count.  The
    //     reference is a second BDA rig whose SOLID side carries that same set,
    //     dispatched as PGS -- because a diffuse thread takes the DIFFUSE projection
    //     scale whatever isTGS says, and the diffuse scale is numerically the PGS
    //     scale.  CCD is ON, which is what makes that reachable at all.
    const uint32_t ND = Scene::NP - 37u;
    std::vector<F4> diffCur(Scene::NP, { 1000, 1000, 1000, 1 });
    std::vector<F4> diffPred(Scene::NP, { 1000, 1000, 1000, 1 });
    for (uint32_t i = 0; i < ND; ++i)
    {
        const uint32_t src = (i + 53u) % Scene::NP;
        diffCur[i] = S.curPos[src];
        diffPred[i] = S.predPos[src];
    }

    BdaOptions dOpt;                 // CCD on, TGS, static
    dOpt.maxDiffuse = Scene::NP;
    dOpt.numDiffuse = ND;
    dOpt.diffuseCur = &diffCur;
    dOpt.diffusePred = &diffPred;

    BdaOptions rOpt;                 // the reference: solid, PGS, the same particles
    rOpt.isTgs = false;
    rOpt.solidCur = &diffCur;
    rOpt.solidPred = &diffPred;
    rOpt.numSolid = ND;

    BdaRig D, Rf;
    buildBda(dev, S, dOpt, D);
    buildBda(dev, S, rOpt, Rf);
    {
        Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
        k.dispatch(GRID, {}, D.blob, 1u, 2u);
        k.dispatch(GRID, {}, Rf.blob, 1u, 1u);
    }
    dev->flush();
    {
        const auto dc = downloadBuffer<uint32_t>(dev, D.bDiffCounts.h, Scene::NP);
        const auto rc = downloadBuffer<uint32_t>(dev, Rf.bCounts.h, Scene::NP);
        const auto di = downloadBuffer<F4>(dev, D.bDiffInfos.h, size_t(Scene::NP) * MAXPART);
        const auto ri = downloadBuffer<F4>(dev, Rf.bInfos.h, size_t(Scene::NP) * MAXPART);
        const auto dn = downloadBuffer<uint32_t>(dev, D.bDiffNodeIdx.h, size_t(Scene::NP) * MAXPART * 2u);
        const auto rn = downloadBuffer<uint32_t>(dev, Rf.bNodeIdx.h, size_t(Scene::NP) * MAXPART * 2u);
        size_t live = 0;
        for (uint32_t p = 0; p < ND; ++p) if (dc[p] != 0u) ++live;
        PXG_CHECK_MSG(live > 100,
                      "the diffuse pass produced almost nothing (%zu); it is not reading the "
                      "diffuse positions or the diffuse count", live);
        PXG_CHECK_MSG(std::memcmp(dc.data(), rc.data(), dc.size() * 4) == 0,
                      "the diffuse pass's one-way COUNTS differ from the PGS solid reference");
        PXG_CHECK_MSG(std::memcmp(di.data(), ri.data(), di.size() * sizeof(F4)) == 0,
                      "the diffuse pass's one-way INFOS differ from the PGS solid reference");
        PXG_CHECK_MSG(std::memcmp(dn.data(), rn.data(), dn.size() * 4) == 0,
                      "the diffuse pass's one-way NODE INDICES differ from the reference");
        // The count really must be the one in use.
        size_t pastEnd = 0;
        for (uint32_t p = ND; p < Scene::NP; ++p) if (dc[p] != 0u) ++pastEnd;
        PXG_CHECK_MSG(pastEnd == 0,
                      "%zu diffuse particles past *mNumDiffuseParticles were processed", pastEnd);
        // Both passes are static, so nothing reaches the writer.
        const uint32_t wc = downloadBuffer<uint32_t>(dev, D.bCount.h, 1)[0];
        PXG_CHECK_MSG(wc == 0u, "the static diffuse run reached the writer (%u)", wc);
    }
    D.destroy(dev);
    Rf.destroy(dev);

    // (c) a DYNAMIC rigid with a real diffuse side.  H14: unlike the mesh twin, the
    //     RESERVATION here is inside the !isDiffuse guard, so the diffuse pass must
    //     contribute NOTHING to the writer counter -- not even reserved holes.
    {
        BdaRig Ds, Dd;
        BdaOptions dyn;
        dyn.staticRigid = false;
        dyn.maxDiffuse = 0u;
        buildBda(dev, S, dyn, Ds);
        dyn.maxDiffuse = Scene::NP;
        dyn.numDiffuse = ND;
        dyn.diffuseCur = &diffCur;
        dyn.diffusePred = &diffPred;
        buildBda(dev, S, dyn, Dd);
        {
            Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
            k.dispatch(GRID, {}, Ds.blob, 1u, 1u);
            k.dispatch(GRID, {}, Dd.blob, 1u, 2u);
        }
        dev->flush();
        const uint32_t wSolid = downloadBuffer<uint32_t>(dev, Ds.bCount.h, 1)[0];
        const uint32_t wBoth = downloadBuffer<uint32_t>(dev, Dd.bCount.h, 1)[0];
        PXG_CHECK_MSG(wSolid > 0u, "the dynamic solid pass wrote nothing");
        PXG_CHECK_MSG(wBoth == wSolid,
                      "the diffuse pass moved the writer counter by %d; this kernel guards the "
                      "RESERVATION with !isDiffuse (H14), unlike the mesh twin (candidate 19)",
                      int(wBoth) - int(wSolid));
        Ds.destroy(dev);
        Dd.destroy(dev);
    }
}

// ===========================================================================
// UPSTREAM CANDIDATE 20 on its own terms.  The Phase-1 module has no cmIdx and so
// cannot express a second contact-manager pair; this test drives two pairs in one
// dispatch (gridDim.y = 2) where the SECOND finds nothing at all, and asserts the
// faithful, wrong outcome: the second pair's plain store of ZERO erases the first
// pair's count while leaving its contact slots written.
// ===========================================================================
PXG_TEST(psHfMidphaseClobber)
{
    PxgSlangComputeDevice* dev = testDevice();

    Scene S;
    buildScene(S);
    const uint32_t GRID = 64u;

    // One pair only -- the baseline.
    BdaOptions one; one.enableCcd = false;
    BdaRig A;
    buildBda(dev, S, one, A);

    // Two pairs, the second against a heightfield 30000 units down.
    BdaOptions two; two.enableCcd = false; two.secondEmptyPair = true;
    BdaRig B;
    buildBda(dev, S, two, B);

    {
        Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
        k.dispatch(GRID, {}, A.blob, 1u, 1u);
        k.dispatch(GRID, {}, B.blob, 2u, 1u);   // gridDim.y = 2 -> two contact managers
    }
    dev->flush();

    const auto aC = downloadBuffer<uint32_t>(dev, A.bCounts.h, Scene::NP);
    const auto bC = downloadBuffer<uint32_t>(dev, B.bCounts.h, Scene::NP);
    const auto aI = downloadBuffer<F4>(dev, A.bInfos.h, size_t(Scene::NP) * MAXPART);
    const auto bI = downloadBuffer<F4>(dev, B.bInfos.h, size_t(Scene::NP) * MAXPART);

    size_t live = 0;
    for (uint32_t p = 0; p < Scene::NP; ++p) if (aC[p] != 0u) ++live;
    PXG_CHECK_MSG(live > 100, "the clobber test ran on an empty baseline (%zu)", live);

    // The CONTACT SLOTS are the same in both runs: the second pair writes nothing,
    // so pair 0's contacts are still there byte for byte.
    PXG_CHECK_MSG(std::memcmp(aI.data(), bI.data(), aI.size() * sizeof(F4)) == 0,
                  "the empty second pair changed the one-way contact SLOTS, which it cannot "
                  "-- it finds no contacts");

    // The COUNTS are not.  Every particle pair 0 gave contacts to is at the mercy of
    // whichever of the two pairs stored last, and pair 1 stores ZERO.
    //
    // WHAT IS ASSERTED HERE IS ONLY THE INVARIANT, and that restriction was earned:
    // this block used to also require `zeroed > 0` -- that the empty pair actually
    // won the race at least once -- and that assertion is itself a coin toss.  It
    // came up tails on 6 runs out of 115 of the UNMUTATED shader (always the same
    // shape: 174 kept, 0 zeroed), which is not a finding about the port, and worse,
    // it made the mutation sweep unreliable: `x6-reclamp-removed` was reported KILLED
    // while `a5-drop-reclamp`, which is character-for-character the same program
    // change, was reported SURVIVED.  One of those two verdicts had to be noise.
    //
    // So the race gets the assertion that holds under EVERY interleaving -- the count
    // is one of exactly two admissible values -- and the tally is printed rather than
    // gated.  The claim the racy version was reaching for is made deterministically in
    // the sequenced case below.
    size_t zeroed = 0, kept = 0, impossible = 0;
    for (uint32_t p = 0; p < Scene::NP; ++p)
    {
        if (aC[p] == 0u) continue;
        if (bC[p] == 0u) ++zeroed;
        else if (bC[p] == aC[p]) ++kept;
        else ++impossible;
    }
    PXG_CHECK_MSG(impossible == 0,
                  "%zu particles carry a count that is neither the single-pair count nor zero; "
                  "candidate 20's race admits exactly those two outcomes", impossible);
    std::printf("    [ note ] candidate 20 race this run: %zu kept, %zu zeroed "
                "(both admissible)\n", kept, zeroed);

    // -----------------------------------------------------------------------
    // The same claim, SEQUENCED, so there is no race to lose.  A rig whose only
    // contact-manager pair meets the sunken heightfield -- it finds nothing -- run
    // over a counter PRE-SEEDED with the counts the live pair just produced.  That is
    // the engine's own arrangement from candidate 20(b): ps_meshCollisonLaunch (or an
    // earlier heightfield pair) accumulates into the array, and a heightfield pair
    // that finds nothing then plain-stores zero over the lot.
    //
    // Everything about the outcome is exact: every count zero, and not one contact
    // slot touched.  A count of zero over slots that other producers wrote, from a
    // kernel that wrote nothing itself.
    // -----------------------------------------------------------------------
    {
        std::vector<uint32_t> liveCounts(aC.begin(), aC.end());
        BdaOptions empty;
        empty.enableCcd = false;
        empty.sunkenPairOnly = true;
        empty.seedCounts = &liveCounts;
        BdaRig E;
        buildBda(dev, S, empty, E);
        {
            Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
            k.dispatch(GRID, {}, E.blob, 1u, 1u);
        }
        dev->flush();
        const auto eC = downloadBuffer<uint32_t>(dev, E.bCounts.h, Scene::NP);
        const auto eI = downloadBuffer<F4>(dev, E.bInfos.h, size_t(Scene::NP) * MAXPART);

        size_t survivingCount = 0, touchedSlot = 0;
        for (uint32_t p = 0; p < Scene::NP; ++p)
            if (eC[p] != 0u) ++survivingCount;
        for (size_t i = 0; i < eI.size(); ++i)
            if (!(eI[i].x == -9.0f && eI[i].y == -9.0f && eI[i].z == -9.0f && eI[i].w == -9.0f))
                ++touchedSlot;

        PXG_CHECK_MSG(survivingCount == 0,
                      "%zu particles kept a seeded one-way count after a pair that found "
                      "nothing; candidate 20's plain store sits OUTSIDE `if (contactCounts)` "
                      "and must zero every one of them", survivingCount);
        PXG_CHECK_MSG(touchedSlot == 0,
                      "the empty pair wrote %zu one-way contact slots; it finds no contacts "
                      "and must write none", touchedSlot);
        E.destroy(dev);
    }

    A.destroy(dev);
    B.destroy(dev);
}

// ===========================================================================
// UPSTREAM CANDIDATE 20(d), and the twelve-contact cap, on deterministic terms.
//
// The one-way contact count is an ACCUMULATOR shared with ps_meshCollisonLaunch and
// ps_sdfMeshCollisonLaunch, so a heightfield pair routinely starts against a count
// some other producer already raised -- that is candidate 20(b) seen from the other
// side.  Every other test in this file starts it at ZERO, and with a single static
// pair that pins the atomicAdd's returned base at 0 for every particle.  Two things
// are then unreachable: `endIndex`'s twelve-contact cap can never bind, and the base
// can never already be AT the cap.  `a9-drop-endindex-cap` survived the first sweep
// on exactly that, and it is not a small mutant -- it is the difference between
// writing inside the array and writing past the end of it.
//
// So this test PRE-SEEDS the counter per particle and dispatches ONE pair.  One pair
// means one writer per counter, so unlike psHfMidphaseClobber there is no race here
// at all and the outcome is exact:
//
//     base v, c contacts found  ->  slots [v, min(v + c, 12)) written,
//                                   count STORED as min(c, 12) = c
//
// For v >= 12 the loop writes NOTHING while the stored count still claims c
// contacts; for v + c > 12 it writes only the first 12 - v of them and the count
// still claims all c.  Both are candidate 20(d) stated exactly: the value stored is
// derived from contactCounts, never from what was actually written.
//
// The oracle for c, and for the contact values themselves, is a SECOND rig that is
// identical but for the seed -- so nothing about the collision maths is transcribed
// here a second time.  The one-way arrays carry TWELVE EXTRA COLUMNS, poisoned; no
// legal run touches them.
// ===========================================================================
PXG_TEST(psHfMidphaseAccum)
{
    PxgSlangComputeDevice* dev = testDevice();

    Scene S;
    buildScene(S);
    const uint32_t GRID = 64u;              // stride 4096 > NP: one thread per particle
    const uint32_t COLS = MAXPART + 12u;    // twelve live columns, twelve guard columns

    // Seven seeds, all distinct, bracketing the cap on both sides: below it, exactly
    // at it, and past it.  The period is 7, coprime with the scene's period-9
    // edge-case pattern, so no seed lands only on off-grid particles.
    //
    // EVERY NON-ZERO SEED IS GREATER THAN MAXMESH, and that is load-bearing.  A
    // particle whose AABB clamps to an empty row or column span takes the zero-span
    // early-out BEFORE the finalize block, so its counter is never stored and keeps
    // the seed.  With a seed of 7 or more that is unambiguous -- no legal count can
    // reach it -- so the two outcomes can be told apart from the host without
    // transcribing the row/column bound math a second time.  The seed of 0 is the
    // control: for those particles the two runs are the same run.
    //
    // The largest seed plus MaxStaticContactsPerMesh must stay inside COLS, so that a
    // port which drops the twelve-contact cap writes into the guard columns rather
    // than past the end of the allocation, where it would be a fault instead of a
    // finding.
    static const uint32_t SEEDS[7] = { 0u, 7u, 8u, 10u, 11u, 12u, 15u };
    static_assert(15u + MAXMESH <= MAXPART + 12u, "the guard region must contain an uncapped write");
    std::vector<uint32_t> seed(Scene::NP);
    for (uint32_t p = 0; p < Scene::NP; ++p) seed[p] = SEEDS[p % 7u];

    BdaOptions base;
    base.enableCcd = false;
    base.infoColumns = COLS;
    BdaOptions seeded = base;
    seeded.seedCounts = &seed;

    BdaRig A, B;
    buildBda(dev, S, base, A);
    buildBda(dev, S, seeded, B);
    {
        Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
        k.dispatch(GRID, {}, A.blob, 1u, 1u);
        k.dispatch(GRID, {}, B.blob, 1u, 1u);
    }
    dev->flush();

    const size_t NCOL = size_t(Scene::NP) * COLS;
    const auto aC = downloadBuffer<uint32_t>(dev, A.bCounts.h, Scene::NP);
    const auto bC = downloadBuffer<uint32_t>(dev, B.bCounts.h, Scene::NP);
    const auto aI = downloadBuffer<F4>(dev, A.bInfos.h, NCOL);
    const auto bI = downloadBuffer<F4>(dev, B.bInfos.h, NCOL);
    const auto aN = downloadBuffer<uint32_t>(dev, A.bNodeIdx.h, NCOL * 2u);
    const auto bN = downloadBuffer<uint32_t>(dev, B.bNodeIdx.h, NCOL * 2u);

    auto infoPoisoned = [](const F4& v)
    {
        return v.x == -9.0f && v.y == -9.0f && v.z == -9.0f && v.w == -9.0f;
    };

    // Counted rather than checked one by one: a port that gets the base wrong misses
    // on every particle at once, and 15360 identical failure lines say no more than
    // the first of them plus a total.
    size_t countMismatch = 0, valueMismatch = 0, nodeMismatch = 0;
    size_t strayLive = 0, strayGuardA = 0, strayGuardB = 0, overSix = 0;
    size_t live = 0, truncated = 0, suppressed = 0, emptyOverSeed = 0, earlyOut = 0;
    uint32_t firstBadP = 0xFFFFFFFFu, firstBadK = 0;

    for (uint32_t p = 0; p < Scene::NP; ++p)
    {
        const uint32_t v = seed[p];
        const uint32_t c = aC[p];           // the unseeded run's count IS the oracle
        if (c > MAXMESH) ++overSix;
        if (c != 0u) ++live;

        // A particle that took the zero-span early-out never reached the finalize, so
        // the seeded run left its counter alone.  That is distinguishable only
        // because every non-zero seed is past the largest legal count.
        const bool untouched = (v > MAXMESH) && (bC[p] == v) && (c == 0u);
        if (untouched) ++earlyOut;

        // Otherwise the stored value is min(contactCounts, 12), which depends on
        // nothing else -- so the seed cannot move it.  That is what makes the stored
        // count a LIE for v >= 12: it claims c contacts none of which were written.
        if (!untouched && bC[p] != c) ++countMismatch;

        const uint32_t endIndex = std::min(v + c, MAXPART);
        const uint32_t nWritten = (!untouched && c != 0u && endIndex > v) ? (endIndex - v) : 0u;

        if (!untouched && c != 0u)
        {
            if (v >= MAXPART) ++suppressed;                  // writes nothing, claims c
            else if (v + c > MAXPART) ++truncated;           // writes 12 - v, claims c
        }
        else if (!untouched && v != 0u) ++emptyOverSeed;      // stores 0 over a live count

        for (uint32_t k = 0; k < COLS; ++k)
        {
            const size_t idx = size_t(p) + size_t(k) * Scene::NP;
            const bool written = (k >= v) && (k < v + nWritten);
            if (written)
            {
                const size_t src = size_t(p) + size_t(k - v) * Scene::NP;
                if (std::memcmp(&bI[idx], &aI[src], sizeof(F4)) != 0)
                {
                    if (firstBadP == 0xFFFFFFFFu) { firstBadP = p; firstBadK = k; }
                    ++valueMismatch;
                }
                if (bN[2 * idx] != aN[2 * src] || bN[2 * idx + 1] != aN[2 * src + 1])
                    ++nodeMismatch;
            }
            else
            {
                if (!infoPoisoned(bI[idx]) || bN[2 * idx] != 0xCCCCCCCCu
                                           || bN[2 * idx + 1] != 0xCCCCCCCCu)
                {
                    if (firstBadP == 0xFFFFFFFFu) { firstBadP = p; firstBadK = k; }
                    if (k >= MAXPART) ++strayGuardB; else ++strayLive;
                }
            }
            // The baseline's own guard columns must be clean too -- it is the same
            // kernel and the same cap, only with every base at zero.
            if (k >= MAXPART && (!infoPoisoned(aI[idx]) || aN[2 * idx] != 0xCCCCCCCCu
                                                        || aN[2 * idx + 1] != 0xCCCCCCCCu))
                ++strayGuardA;
        }
    }

    PXG_CHECK_MSG(overSix == 0,
                  "%zu particles carry more than MaxStaticContactsPerMesh = 6 contacts", overSix);
    PXG_CHECK_MSG(live > 100, "the accumulation test ran on an empty baseline (%zu)", live);
    PXG_CHECK_MSG(countMismatch == 0,
                  "%zu particles that reached the finalize carry a stored count that is "
                  "neither the unseeded count nor the untouched seed; the plain store writes "
                  "min(contactCounts, 12), which does not depend on the atomicAdd's base",
                  countMismatch);
    PXG_CHECK_MSG(valueMismatch == 0,
                  "%zu one-way slots differ from the unseeded run shifted by the seed "
                  "(first at particle %u column %u); the write must start at the base the "
                  "atomicAdd returned", valueMismatch, firstBadP, firstBadK);
    PXG_CHECK_MSG(nodeMismatch == 0,
                  "%zu one-way NODE INDEX slots differ from the shifted unseeded run",
                  nodeMismatch);
    PXG_CHECK_MSG(strayLive == 0,
                  "%zu one-way slots inside the twelve live columns were written that the "
                  "base-plus-count range does not cover (first at particle %u column %u)",
                  strayLive, firstBadP, firstBadK);
    PXG_CHECK_MSG(strayGuardB == 0,
                  "%zu one-way slots PAST column 11 were written -- the twelve-contact cap in "
                  "`endIndex` is missing, and in the engine those writes land outside the "
                  "array entirely (%zu in the unseeded run)", strayGuardB, strayGuardA);
    PXG_CHECK_MSG(strayGuardA == 0,
                  "%zu one-way slots past column 11 were written with every base at zero",
                  strayGuardA);

    // Coverage.  Each of these is a shape the rest of the file cannot reach, and the
    // sweep survivor that motivated this test needs the second one.
    PXG_CHECK_MSG(truncated > 0,
                  "no particle had its contact list PARTIALLY truncated by the twelve-contact "
                  "cap (seed < 12 <= seed + count)");
    PXG_CHECK_MSG(suppressed > 0,
                  "no particle had a base at or past the twelve-contact cap with contacts to "
                  "write; candidate 20(d)'s purest case -- a count that claims contacts none "
                  "of which were written -- is untested");
    PXG_CHECK_MSG(emptyOverSeed > 0,
                  "no particle REACHED the finalize with a seeded count and found nothing; the "
                  "plain store sits OUTSIDE `if (contactCounts)` and that placement is untested");
    PXG_CHECK_MSG(earlyOut > 0,
                  "no particle took the zero-span early-out, so the fixture cannot tell "
                  "\"found nothing\" from \"never got there\" and the scene has lost its "
                  "off-grid particles");

    // H13: contactWritten == contactCounts for a static rigid, so `remaining` is zero
    // and the shared writer is never reached -- with or without a seed.
    const uint32_t wA = downloadBuffer<uint32_t>(dev, A.bCount.h, 1)[0];
    const uint32_t wB = downloadBuffer<uint32_t>(dev, B.bCount.h, 1)[0];
    PXG_CHECK_MSG(wA == 0u && wB == 0u,
                  "a static rigid reached the shared writer (%u unseeded, %u seeded); H13 says "
                  "`remaining` is always zero there", wA, wB);

    A.destroy(dev);
    B.destroy(dev);
}

// ===========================================================================
// P9.94c: the local bound, on inputs where its spelling decides the contact COUNT.
// The random scene's general rotation hides the difference between the CUDA's
// transformFast(pose.getInverse(), bound) and a rotateInv spelling (no contact
// moves), so a shader pair that both drifted would still pass psHfMidphase.  Here
// each pose of pxgPsHfBoundCases.h fixes the count to the literal reference's,
// which psHfCollide_boundLiteral derives and pins on the host.
// ===========================================================================
PXG_TEST(psHfMidphaseBoundLiteral)
{
    using namespace pxgpshfbound;
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t refGroups = (Scene::NP + 255u) / 256u;

    for (const Pose& pose : kPoses)
    {
        Scene S;
        buildBoundScene(S, pose);
        const uint32_t nLit[2] = { pose.part[0].nLit, pose.part[1].nLit };
        const uint32_t sumLit = (Scene::NP / 2u) * (nLit[0] + nLit[1]);

        for (const bool staticRigid : { true, false })
        {
            BdaOptions opt;
            opt.enableCcd = false;
            opt.staticRigid = staticRigid;
            const char* mode = staticRigid ? "static" : "dynamic";

            BdaRig A;
            RefRig B;
            buildBda(dev, S, opt, A);
            buildRef(dev, S, opt, B);
            {
                Kernel k("psHfMidphaseBda.spv", "ps_heightfieldCollisonLaunch", {}, sizeof(PsHfMidBlob));
                k.dispatch(64u, {}, A.blob, 1u, 1u);
            }
            {
                Kernel k("psHfCollide.spv", "ctPsHfCollide", refStrides(), sizeof(PhParams));
                k.dispatch(refGroups, B.bindings(), B.prm);
            }
            dev->flush();

            const auto aI = downloadBuffer<F4>(dev, A.bInfos.h, size_t(Scene::NP) * MAXPART);
            const auto bI = downloadBuffer<F4>(dev, B.bI, size_t(Scene::NP) * MAXPART);
            const auto aC = downloadBuffer<uint32_t>(dev, A.bCounts.h, Scene::NP);
            const auto bC = downloadBuffer<uint32_t>(dev, B.bC, Scene::NP);
            const auto aN = downloadBuffer<uint32_t>(dev, A.bNodeIdx.h, size_t(Scene::NP) * MAXPART * 2u);
            const auto bN = downloadBuffer<uint32_t>(dev, B.bNi, size_t(Scene::NP) * MAXPART * 2u);
            PXG_CHECK_MSG(std::memcmp(aC.data(), bC.data(), aC.size() * 4) == 0,
                          "%s %s: the one-way COUNTS differ", pose.name, mode);
            PXG_CHECK_MSG(std::memcmp(aI.data(), bI.data(), aI.size() * sizeof(F4)) == 0,
                          "%s %s: the one-way CONTACT INFOS differ", pose.name, mode);
            PXG_CHECK_MSG(std::memcmp(aN.data(), bN.data(), aN.size() * 4) == 0,
                          "%s %s: the one-way NODE INDICES differ", pose.name, mode);

            const uint32_t aW = downloadBuffer<uint32_t>(dev, A.bCount.h, 1)[0];
            const uint32_t bW = downloadBuffer<uint32_t>(dev, B.bWc, 1)[0];
            PXG_CHECK_MSG(aW == bW, "%s %s: writer counters differ (%u vs %u)", pose.name, mode, aW, bW);
            {
                const auto aCt = downloadBuffer<PsC>(dev, A.bContacts.h, Scene::CAP);
                const auto bCt = downloadBuffer<PsC>(dev, B.bW, Scene::CAP);
                const auto ga = groupByParticle(aCt, std::min(aW, Scene::CAP));
                const auto gb = groupByParticle(bCt, std::min(bW, Scene::CAP));
                size_t mism = 0;
                for (uint32_t p = 0; p <= Scene::NP; ++p)
                {
                    if (ga[p].size() != gb[p].size()) { ++mism; continue; }
                    for (size_t i = 0; i < ga[p].size(); ++i)
                        if (!sameContact(ga[p][i], gb[p][i])) { ++mism; break; }
                }
                PXG_CHECK_MSG(mism == 0, "%s %s: %zu particles' writer contacts differ", pose.name, mode, mism);
            }

            // The literal reference's count, not merely agreement between the twins.
            size_t wrong = 0;
            for (uint32_t i = 0; i < Scene::NP; ++i)
                if (aC[i] != (staticRigid ? nLit[i % 2u] : 0u)) ++wrong;
            PXG_CHECK_MSG(wrong == 0, "%s %s: %zu one-way counts differ from the literal reference (%u, %u)",
                          pose.name, mode, wrong, nLit[0], nLit[1]);
            const uint32_t wantW = staticRigid ? 0u : sumLit;
            PXG_CHECK_MSG(aW == wantW, "%s %s: writer counter %u, literal total %u", pose.name, mode, aW, wantW);

            A.destroy(dev);
            B.destroy(dev);
        }
    }
}
