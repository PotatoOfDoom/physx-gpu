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

// testPsSdfMidphase.cpp -- P9.12h, the fixture for ps_sdfMeshCollisonLaunch.
//
// ---------------------------------------------------------------------------
// THE ARRANGEMENT: A WRAPPER CHECKED AGAINST ITS OWN PARTS
// ---------------------------------------------------------------------------
// This kernel is composition.  The two halves that carry the work are already
// swept modules with their own bit-exact fixtures --
//
//   pxgSlangSdfFind      P9.12e  which particles are even considered
//   pxgSlangSdfParticle  P9.12f  what contact each of them produces
//
// -- so re-deriving either here would be a fourth transcription, which is what
// this port exists to avoid.  Instead the expectation is built from TWO
// INDEPENDENT DISPATCHES of those modules' own entry points:
//
//   npSdfFind.spv:sdfFindRounds            -> the exact gathered index list, per
//                                             resume round
//   npSdfParticle.spv:sdfParticleContactAt -> the per-particle contact decision
//
// and the host then predicts, from those two, every byte the kernel should have
// written: which particle lands in the one-way arrays and at which column-major
// slot, which reaches the contact stream and at which reserved index, and what
// the rotated normal is.
//
// That leaves the GLUE as the claim, which is what a wrapper tranche should
// test: the shape-pair resolve, the cull tolerance, the inverted mesh pose, the
// two DIFFERENT particle indices, the static/dynamic split, the capacity
// behaviour and the block-wide reservation.
//
// AND THE GLUE IS DISCRIMINATED, not merely exercised.  The reference dispatches
// are fed the cull tolerance and the inverse transform the HOST computed.  If
// the kernel derives either of them differently -- a wrong scale component, the
// quaternion not conjugated, the translation rotated forwards -- it gathers a
// different set or decides differently, and the comparison fails.  There is no
// path by which a wrong cullScale produces a right answer here.
//
//   psSdfMidphase        -- the solid pass: both shape-slot orders, a static and
//                           a dynamic rigid, filtered particles, the capacity
//                           cap and the two-round resume.
//   psSdfMidphaseDiffuse -- the diffuse pass: one-way contacts are produced, the
//                           contact stream is NOT touched, the filter is not
//                           consulted, and a system with no diffuse capacity
//                           returns before writing anything.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"     // PxgParticleSystem, PxgParticleContactWriter
#include "PxgContactManager.h"     // PxgContactManagerInput
#include "PxgConvexConvexShape.h"  // PxgShape
#include "PxsTransformCache.h"
#include "PxNodeIndex.h"
#include "geometry/PxGeometry.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "pxgSdfSceneRef.h"

#include <algorithm>
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
using namespace physx;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using namespace pxgsdfref;

struct PsC { F4 normal_pen; uint32_t rLo, rHi, pLo, pHi; };
static_assert(sizeof(PsC) == 32, "PxgParticlePrimitiveContact stride");

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

// psSdfMidphaseBda.slang's PsSdfMidPush, byte for byte.
struct PsSdfMidBlob
{
    float tolerenceLength;                     // @0   W1: never read
    uint32_t pad0;                             // @4
    uint64_t cmInputs;                         // @8
    uint64_t transformCache;                   // @16
    uint64_t contactDistance;                  // @24
    uint64_t restDistances;                    // @32
    uint64_t gpuShapes;                        // @40
    uint64_t particleSystems;                  // @48
    uint64_t shapeToRigidRemapTable;           // @56
    // PxgRigidParticleContactWriter (PhysX 5.12, 80 B) @64
    uint64_t w_particleContactInfos;           // @64
    uint64_t w_particleContactNormalPens;      // @72
    uint64_t w_numTotalContacts;               // @80
    uint64_t w_contactSortedByParticle;        // @88
    uint64_t w_tempContactByParticle;          // @96
    uint64_t w_contactIndexSortedByParticle;   // @104
    uint64_t w_contactByRigid;                 // @112
    uint64_t w_tempContactByRigid;             // @120
    uint64_t w_contactIndexSortedByRigid;      // @128
    uint32_t w_maxContacts;                    // @136
    uint32_t pad1;                             // -> 144
};
static_assert(sizeof(PsSdfMidBlob) == 144, "engine ABI blob size");
static_assert(offsetof(PsSdfMidBlob, w_particleContactInfos) == 64, "the writer lands at 64");

// The two reference entry points' push blocks (P9.12e / P9.12f).
// P_Find mirrors ALL of npSdfFind.slang's SdfFindParams (176 B since P9.34b/P9.35b).
// Until P9.98 it stopped at maxRounds (128 B), which the validation layer reported as
// VUID-07987 (declared block outside the layout's push range); sdfFindRounds never
// reads past @112, so the zeroed tail changes nothing it computes.
struct P_Find
{
    uint64_t meshPtr, positions, echoOut, roundsOut, bufOut, numRoundsOut;
    float aToBq[4];
    float scaleRot[4];
    float aToBp[3];  float tolerance;
    float scale[3];  uint32_t numVertices;
    uint32_t maxRounds, f0, f1, f2;
    uint64_t meshInd, touchedOut;   // 128, 136 -- the triangle / tet gathers' (0 here)
    float scale0Rot[4];             // 144
    float scale0[3]; uint32_t f3;   // 160 -> 176
};
static_assert(sizeof(P_Find) == 176, "npSdfFind push block (SdfFindParams)");
struct P_Part
{
    uint64_t meshPtr, positions, positionsPrev, outBuf;
    float aToBq[4];
    float scaleRot[4];
    float aToBp[3]; float cullScale;
    float scale[3];  float contactDist;
    float restDist; uint32_t numPairs, enableCCD, p0;
};
static const uint32_t FIND_TARGET = 1024u;
static const uint32_t FIND_MAXROUNDS = 16u;
static const uint32_t PART_STRIDE = 64u;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;

// ---------------------------------------------------------------------------
// The scene.
// ---------------------------------------------------------------------------
struct Scene
{
    // THE RESUME LOOP IS GATED BY THE QUALIFYING COUNT, NOT BY THE PARTICLE
    // COUNT.  findInterestingVertices scans until its 1024-slot buffer is FULL,
    // so a cloud of 1500 with 449 near the surface is one round, not two.  NP is
    // sized so that ~30% of it qualifies and the second round is real; runOne
    // requires >= 2 rounds rather than trusting this arithmetic.
    static const uint32_t NP = 4001u;
    // The diffuse pass has its OWN count and its OWN positions.  Both differ from
    // the solid ones on purpose: NPD < FIND_TARGET so the diffuse gather takes a
    // single round where the solid one takes two, and the cloud is a different
    // cloud so "read the diffuse array" and "read the solid array" are not the
    // same answer (W4).
    static const uint32_t NPD = 700u;
    static const uint32_t SYS = 2u;              // not 0 -- a dropped id is visible
    static const uint32_t NB_SYS = 4u;
    static const uint32_t TRI_SHAPE = 3u, PS_SHAPE = 1u, NB_SHAPES = 6u;
    static const uint32_t TRI_CACHE = 5u, PS_CACHE = 2u, NB_CACHE = 8u;
    static const uint32_t CAP = 4096u;
    static const uint32_t ONEWAY_COLS = 12u;     // MaxStaticContactsPerParticle

    SdfScene sdf;
    BlobLayout blob;
    PxVec3 meshP; PxQuat meshQ;
    PxVec3 scale; PxQuat scaleRot;
    float contactDistTri, contactDistPs, restDist;
    std::vector<F4> pos, posPrev;
    std::vector<F4> posD, posPrevD;              // the diffuse cloud
    std::vector<uint32_t> sortedToUnsorted;
    std::vector<uint32_t> filteredUnsorted;      // named by their UNSORTED index

    float contactDist() const { return contactDistTri + contactDistPs; }
    float cullScale() const
    {
        return contactDist() / PxMin(PxAbs(scale.x), PxMin(PxAbs(scale.y), PxAbs(scale.z)));
    }
    PxTransform meshPose() const { return PxTransform(meshP, meshQ); }
    PxTransform aToB() const { return meshPose().getInverse(); }
};

static void buildScene(Scene& S)
{
    buildSparse(S.sdf);
    S.blob = layOutBlob(S.sdf, 16u, 21u, 3u, 1u, 0u);

    // Nothing identity, nothing axis-aligned.  A kernel that forgets to invert
    // the pose, or conjugates without rotating the translation, lands elsewhere.
    S.meshQ = PxQuat(0.53f, PxVec3(0.36f, -0.8f, 0.48f).getNormalized());
    S.meshP = PxVec3(0.62f, -0.31f, 1.15f);
    // Three distinct, none 1, and one NEGATIVE: the CUDA divides the contact
    // distance by `scale.abs().minElement()`, and with an all-positive scale a
    // port that drops the abs() is right by accident.  |-0.74| is still the
    // smallest, so the two spellings differ only in sign -- which is the whole
    // point (W3).
    S.scale = PxVec3(1.17f, -0.74f, 1.86f);
    S.scaleRot = PxQuat(-0.44f, PxVec3(0.8f, 0.48f, 0.36f).getNormalized());
    // Two DIFFERENT contact distances, so a kernel that reads one cache slot
    // twice gets a different sum.
    S.contactDistTri = 0.052f;
    S.contactDistPs = 0.033f;
    S.restDist = 0.019f;

    const HostSdfBox box = hostSdfBox(S.sdf);
    const PxVec3 ext = box.higher - box.lower;
    const PxTransform bToA = S.aToB().getInverse();

    S.pos.resize(Scene::NP);
    S.posPrev.resize(Scene::NP);
    S.sortedToUnsorted.resize(Scene::NP);
    std::mt19937 rng(0x5DF12A7u);
    auto u01 = [&]() { return float(rng() % 1000000u) / 1000000.f; };
    auto sym = [&]() { return 2.0f * u01() - 1.0f; };
    auto toWorld = [&](const PxVec3& vtx) {
        PxVec3 t = S.scaleRot.rotate(vtx);
        t.x *= S.scale.x; t.y *= S.scale.y; t.z *= S.scale.z;
        return bToA.transform(S.scaleRot.rotateInv(t));
    };
    for (uint32_t i = 0; i < Scene::NP; ++i)
    {
        const PxVec3 vPrev(box.lower.x - 0.25f * ext.x + 1.5f * ext.x * u01(),
                           box.lower.y - 0.25f * ext.y + 1.5f * ext.y * u01(),
                           box.lower.z - 0.25f * ext.z + 1.5f * ext.z * u01());
        const float mag = (i % 3u == 0u) ? 0.05f * S.sdf.spacing
                        : (i % 3u == 1u) ? 0.8f * S.sdf.spacing
                        :                  0.4f * ext.magnitude();
        const PxVec3 v = vPrev + PxVec3(sym(), sym(), sym()) * mag;
        const PxVec3 w = toWorld(v), wPrev = toWorld(vPrev);
        S.pos[i] = F4{ w.x, w.y, w.z, 1.0f };
        S.posPrev[i] = F4{ wPrev.x, wPrev.y, wPrev.z, 1.0f };
        // A PERMUTATION, not the identity: the filter is queried with this index
        // and the contact is written with the sorted one, and if the two were
        // equal that swap would be invisible (W5).
        S.sortedToUnsorted[i] = (i * 7u + 13u) % Scene::NP;   // gcd(7, 4001) == 1
    }
    // The diffuse cloud: a different seed, a different displacement schedule and a
    // shifted box, so its candidate set is not a prefix of the solid one.
    S.posD.resize(Scene::NPD);
    S.posPrevD.resize(Scene::NPD);
    {
        std::mt19937 rngD(0xD1FF05Eu);
        auto d01 = [&]() { return float(rngD() % 1000000u) / 1000000.f; };
        auto dsym = [&]() { return 2.0f * d01() - 1.0f; };
        for (uint32_t i = 0; i < Scene::NPD; ++i)
        {
            const PxVec3 vPrev(box.lower.x - 0.4f * ext.x + 1.8f * ext.x * d01(),
                               box.lower.y - 0.4f * ext.y + 1.8f * ext.y * d01(),
                               box.lower.z - 0.4f * ext.z + 1.8f * ext.z * d01());
            const float mag = (i % 4u == 0u) ? 0.03f * S.sdf.spacing
                            : (i % 4u == 1u) ? 1.3f * S.sdf.spacing
                            : (i % 4u == 2u) ? 0.25f * ext.magnitude()
                            :                  0.6f * ext.magnitude();
            const PxVec3 v = vPrev + PxVec3(dsym(), dsym(), dsym()) * mag;
            const PxVec3 w = toWorld(v), wPrev = toWorld(vPrev);
            S.posD[i] = F4{ w.x, w.y, w.z, 1.0f };
            S.posPrevD[i] = F4{ wPrev.x, wPrev.y, wPrev.z, 1.0f };
        }
    }
    // Filtered particles, named by UNSORTED index and kept sorted for the binary
    // search the engine's filter performs.
    // Two groups.  The first is fixed and lands wherever it lands in the sorted
    // order.  The second is chosen BY SORTED INDEX below NPD, because the diffuse
    // pass only ever queries indices in [0, NPD) -- without them "the filter runs
    // for the diffuse pass too" is a mutant nothing can see (W6).  runOne
    // REQUIRES both groups to bite rather than assuming they do.
    S.filteredUnsorted = { 11u, 407u, 408u, 900u, 1499u };
    for (uint32_t i = 3u; i < Scene::NPD; i += 17u)
        S.filteredUnsorted.push_back(S.sortedToUnsorted[i]);
    std::sort(S.filteredUnsorted.begin(), S.filteredUnsorted.end());
    S.filteredUnsorted.erase(std::unique(S.filteredUnsorted.begin(), S.filteredUnsorted.end()),
                             S.filteredUnsorted.end());
}

// ---------------------------------------------------------------------------
// The device arena.
// ---------------------------------------------------------------------------
// THE GRID IS ALWAYS (numTests, 2, 1) -- PxgNarrowphaseCore.cpp:3904.  There is
// no such thing as a solid-only dispatch: block y=0 does the solid pass and
// block y=1 does the diffuse pass, in the same launch, sharing one contact
// stream.  So every configuration here runs BOTH, and `maxDiffuse == 0` is not
// "no diffuse test", it is the early-out that the y=1 block takes.
struct Options
{
    bool psFirst = false;        // which slot of the contact pair holds the mesh
    bool staticRigid = true;
    bool enableCcd = false;
    uint32_t maxDiffuse = 0u;    // 0 -> the y=1 block returns before any write
    uint32_t seedOneWay = 0u;    // pre-seed every one-way counter, both passes
};

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bBlob, bPos, bPrev, bPosD, bPrevD, bMap, bDesc, bCin, bShapes, bXf, bCd, bRest, bRemap;
    DevBuf bInfos, bNodeIdx, bCounts, bDiffInfos, bDiffNodeIdx, bDiffCounts, bNumDiffuse;
    DevBuf bRunsum, bSimBufs;
    DevBuf bContacts /* PxgDeformablePairInfo */, bNormalPens, bCount, bSbP, bTbP, bIbP, bBr, bTbr, bIbr;
    PsSdfMidBlob blob{};
    uint64_t texCoarse = 0, texSub = 0;
    PxNodeIndex rigidId;

    template <typename T>
    DevBuf mk(PxgSlangComputeDevice* dev, const std::vector<T>& d)
    {
        DevBuf b;
        b.h = createBufferFrom(dev, d);
        b.addr = dev->getBufferDeviceAddress(b.h);
        owned.push_back(b.h);
        return b;
    }
    void destroy(PxgSlangComputeDevice* dev)
    {
        for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
        owned.clear();
    }
};

static bool buildRig(PxgSlangComputeDevice* dev, Scene& S, const Options& opt, Rig& R)
{
    // The SDF textures, through the engine's own upload path.
    R.texCoarse = sdfUploadTex(CU_AD_FORMAT_FLOAT, S.sdf.nbx + 1, S.sdf.nby + 1, S.sdf.nbz + 1,
                               S.sdf.coarse.data(), sizeof(float));
    R.texSub = sdfUploadTex(CU_AD_FORMAT_UNSIGNED_INT8, S.sdf.blkx * (S.sdf.sg + 1),
                            S.sdf.blky * (S.sdf.sg + 1), S.sdf.blkz * (S.sdf.sg + 1),
                            S.sdf.sub.data(), 1);
    if (R.texCoarse == 0 || R.texSub == 0)
        return false;
    BlobLayout blob = S.blob;
    std::memcpy(&blob.bytes[blob.texObjOffset], &R.texCoarse, 8);
    std::memcpy(&blob.bytes[blob.texObjSparseOffset], &R.texSub, 8);
    R.bBlob = R.mk(dev, blob.bytes);

    R.bPos = R.mk(dev, S.pos);
    R.bPrev = R.mk(dev, S.posPrev);
    R.bPosD = R.mk(dev, S.posD);
    R.bPrevD = R.mk(dev, S.posPrevD);
    R.bMap = R.mk(dev, S.sortedToUnsorted);

    std::vector<F4> infos(size_t(Scene::NP) * Scene::ONEWAY_COLS, F4{ -9, -9, -9, -9 });
    std::vector<uint64_t> nodeIdx(size_t(Scene::NP) * Scene::ONEWAY_COLS, ~0ull);
    std::vector<uint32_t> counts(Scene::NP, opt.seedOneWay);
    R.bInfos = R.mk(dev, infos);
    R.bNodeIdx = R.mk(dev, nodeIdx);
    R.bCounts = R.mk(dev, counts);
    // The diffuse side is sized by NPD, not NP: the output index is
    // `ind + startIndex * numParticles` and numParticles is the DIFFUSE count.
    // Same-sized arrays would make a pass that used the wrong stride land inside
    // the allocation and go unnoticed.
    std::vector<F4> infosD(size_t(Scene::NPD) * Scene::ONEWAY_COLS, F4{ -9, -9, -9, -9 });
    std::vector<uint64_t> nodeIdxD(size_t(Scene::NPD) * Scene::ONEWAY_COLS, ~0ull);
    std::vector<uint32_t> countsD(Scene::NPD, opt.seedOneWay);
    R.bDiffInfos = R.mk(dev, infosD);
    R.bDiffNodeIdx = R.mk(dev, nodeIdxD);
    R.bDiffCounts = R.mk(dev, countsD);
    const std::vector<uint32_t> nDiff(1u, Scene::NPD);
    R.bNumDiffuse = R.mk(dev, nDiff);

    // One particle buffer, runsum {0}.  PhysX 5.12 removed the particle-rigid filter
    // list the 5.6.1 rig seeded here (S.filteredUnsorted): those particles are now
    // ordinary candidates, which predict() checks (W5 / W6).
    const std::vector<uint32_t> runsum(1u, 0u);
    R.bRunsum = R.mk(dev, runsum);
    R.rigidId = PxNodeIndex(opt.staticRigid ? 0xFFFFFFFFu : 0x51u, opt.staticRigid ? 0u : 3u);

    std::vector<PxgParticleSimBuffer> simBufs(1);
    std::memset(simBufs.data(), 0, sizeof(PxgParticleSimBuffer));
    R.bSimBufs = R.mk(dev, simBufs);

    std::vector<PxgParticleSystem> descArr(Scene::NB_SYS);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));
    PxgParticleSystem ps{};
    std::memset(&ps, 0, sizeof(ps));
    ps.mCommonData.mNumParticles = Scene::NP;
    ps.mCommonData.mMaxParticles = Scene::NP;
    ps.mCommonData.mMaxDiffuseParticles = opt.maxDiffuse;
    ps.mCommonData.mNumParticleBuffers = 1u;
    ps.mData.mFlags = opt.enableCcd ? PxU32(PxParticleFlag::eENABLE_SPECULATIVE_CCD) : 0u;
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(R.bPos.addr));
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(R.bPrev.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(R.bMap.addr));
    ps.mOneWayContactInfos = reinterpret_cast<PxgParticleContactInfo*>(size_t(R.bInfos.addr));
    ps.mOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.bNodeIdx.addr));
    ps.mOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.bCounts.addr));
    ps.mDiffuseSortedPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bPosD.addr));
    ps.mDiffuseSortedOriginPos_LifeTime = reinterpret_cast<float4*>(size_t(R.bPrevD.addr));
    ps.mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bNumDiffuse.addr));
    ps.mDiffuseOneWayContactInfos =
        reinterpret_cast<PxgParticleContactInfo*>(size_t(R.bDiffInfos.addr));
    ps.mDiffuseOneWayNodeIndex = reinterpret_cast<PxNodeIndex*>(size_t(R.bDiffNodeIdx.addr));
    ps.mDiffuseOneWayContactCount = reinterpret_cast<PxU32*>(size_t(R.bDiffCounts.addr));
    ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(R.bRunsum.addr));
    ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(R.bSimBufs.addr));
    descArr[Scene::SYS] = ps;
    R.bDesc = R.mk(dev, descArr);

    // ONE contact manager, so the block scan's global offset is predictable.
    std::vector<PxgContactManagerInput> cin(1);
    std::memset(cin.data(), 0xEE, sizeof(PxgContactManagerInput));
    if (opt.psFirst)
    {
        cin[0].shapeRef0 = Scene::PS_SHAPE;  cin[0].transformCacheRef0 = Scene::PS_CACHE;
        cin[0].shapeRef1 = Scene::TRI_SHAPE; cin[0].transformCacheRef1 = Scene::TRI_CACHE;
    }
    else
    {
        cin[0].shapeRef0 = Scene::TRI_SHAPE; cin[0].transformCacheRef0 = Scene::TRI_CACHE;
        cin[0].shapeRef1 = Scene::PS_SHAPE;  cin[0].transformCacheRef1 = Scene::PS_CACHE;
    }
    R.bCin = R.mk(dev, cin);

    std::vector<PxgShape> shapes(Scene::NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    // Every decoy is ALSO a triangle mesh with a null geometry pointer, so a
    // kernel that picks the wrong slot passes the type test and then faults --
    // a crash rather than a near miss.
    for (auto& sh : shapes)
    {
        sh.type = PxGeometryType::eTRIANGLEMESH;
        sh.scale = PxMeshScale(PxVec3(3.0f, 3.0f, 3.0f), PxQuat(PxIdentity));
        sh.hullOrMeshPtr = 0;
    }
    shapes[Scene::TRI_SHAPE].type = PxGeometryType::eTRIANGLEMESH;
    shapes[Scene::TRI_SHAPE].scale = PxMeshScale(S.scale, S.scaleRot);
    shapes[Scene::TRI_SHAPE].hullOrMeshPtr = size_t(R.bBlob.addr);
    shapes[Scene::PS_SHAPE].type = PxGeometryType::ePARTICLESYSTEM;
    shapes[Scene::PS_SHAPE].particleOrSoftbodyId = Scene::SYS;
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> xf(Scene::NB_CACHE);
    std::memset(xf.data(), 0, xf.size() * sizeof(PxsCachedTransform));
    for (auto& t : xf) t.transform = PxTransform(PxVec3(77.f, 77.f, 77.f), PxQuat(PxIdentity));
    xf[Scene::TRI_CACHE].transform = S.meshPose();
    R.bXf = R.mk(dev, xf);

    std::vector<float> cd(Scene::NB_CACHE, -5.f);
    cd[Scene::TRI_CACHE] = S.contactDistTri;
    cd[Scene::PS_CACHE] = S.contactDistPs;
    R.bCd = R.mk(dev, cd);
    const std::vector<float> rest(1u, S.restDist);
    R.bRest = R.mk(dev, rest);

    std::vector<uint64_t> remap(Scene::NB_CACHE, 0xDEADBEEFDEADBEEFull);
    remap[Scene::TRI_CACHE] = R.rigidId.getInd();
    R.bRemap = R.mk(dev, remap);

    // The writer arrays.
    // PhysX 5.12: two SoA streams, PxgDeformablePairInfo and float4 normalPen.
    const std::vector<PxgDeformablePairInfo> ctInit(Scene::CAP, PxgDeformablePairInfo{ ~0ull, ~0ull });
    const std::vector<F4> npInit(Scene::CAP, F4{ -7, -7, -7, -7 });
    const std::vector<uint32_t> u32Init(Scene::CAP, POISON_U32);
    const std::vector<uint64_t> u64Init(Scene::CAP, ~0ull);
    R.bContacts = R.mk(dev, ctInit);
    R.bNormalPens = R.mk(dev, npInit);
    R.bCount = R.mk(dev, std::vector<uint32_t>(1u, 0u));
    R.bSbP = R.mk(dev, u64Init);
    R.bTbP = R.mk(dev, u32Init);
    R.bIbP = R.mk(dev, u32Init);
    R.bBr = R.mk(dev, u64Init);
    R.bTbr = R.mk(dev, u32Init);
    R.bIbr = R.mk(dev, u32Init);

    R.blob = PsSdfMidBlob{};
    R.blob.tolerenceLength = 1234.5f;            // W1: must never matter
    R.blob.cmInputs = R.bCin.addr;
    R.blob.transformCache = R.bXf.addr;
    R.blob.contactDistance = R.bCd.addr;
    R.blob.restDistances = R.bRest.addr;
    R.blob.gpuShapes = R.bShapes.addr;
    R.blob.particleSystems = R.bDesc.addr;
    R.blob.shapeToRigidRemapTable = R.bRemap.addr;
    R.blob.w_particleContactInfos = R.bContacts.addr;
    R.blob.w_particleContactNormalPens = R.bNormalPens.addr;
    R.blob.w_numTotalContacts = R.bCount.addr;
    R.blob.w_contactSortedByParticle = R.bSbP.addr;
    R.blob.w_tempContactByParticle = R.bTbP.addr;
    R.blob.w_contactIndexSortedByParticle = R.bIbP.addr;
    R.blob.w_contactByRigid = R.bBr.addr;
    R.blob.w_tempContactByRigid = R.bTbr.addr;
    R.blob.w_contactIndexSortedByRigid = R.bIbr.addr;
    R.blob.w_maxContacts = Scene::CAP;
    return true;
}

// ---------------------------------------------------------------------------
// The two reference dispatches.
// ---------------------------------------------------------------------------
struct Reference
{
    std::vector<std::vector<uint32_t>> rounds;   // the gathered index list per round
    std::vector<uint8_t> candidate;              // per particle
    std::vector<F4> normalSep;                   // per particle, mesh-local
};

// `positions`/`positionsPrev`/`n` name the pass being predicted: the solid pass's
// own arrays, or the diffuse pass's.
static bool buildReference(PxgSlangComputeDevice* dev, const Scene& S, const Options& opt,
                           Rig& R, uint64_t positionsAddr, uint64_t positionsPrevAddr,
                           uint32_t n, Reference& ref)
{
    const PxTransform aToB = S.aToB();
    const float cull = S.cullScale();

    // ---- the gather ----
    {
        const std::vector<F4> echoInit(n, F4{ 0, 0, 0, 0 });
        const std::vector<uint32_t> rInit(FIND_MAXROUNDS * 4u, POISON_U32);
        const std::vector<uint32_t> bInit(size_t(FIND_MAXROUNDS) * FIND_TARGET, POISON_U32);
        const std::vector<uint32_t> nInit(1u, POISON_U32);
        DevBuf bE = R.mk(dev, echoInit), bR = R.mk(dev, rInit);
        DevBuf bB = R.mk(dev, bInit), bN = R.mk(dev, nInit);
        P_Find p{};
        p.meshPtr = R.bBlob.addr;
        p.positions = positionsAddr;             // W2: the PREDICTED buffer
        p.echoOut = bE.addr; p.roundsOut = bR.addr; p.bufOut = bB.addr; p.numRoundsOut = bN.addr;
        p.aToBq[0] = aToB.q.x; p.aToBq[1] = aToB.q.y; p.aToBq[2] = aToB.q.z; p.aToBq[3] = aToB.q.w;
        p.scaleRot[0] = S.scaleRot.x; p.scaleRot[1] = S.scaleRot.y;
        p.scaleRot[2] = S.scaleRot.z; p.scaleRot[3] = S.scaleRot.w;
        p.aToBp[0] = aToB.p.x; p.aToBp[1] = aToB.p.y; p.aToBp[2] = aToB.p.z;
        p.tolerance = cull;
        p.scale[0] = S.scale.x; p.scale[1] = S.scale.y; p.scale[2] = S.scale.z;
        p.numVertices = n;
        p.maxRounds = FIND_MAXROUNDS;
        {
            Kernel k(Kernel::Bindless{}, "npSdfFind.spv", "sdfFindRounds", {}, sizeof(P_Find));
            PXG_CHECK_MSG(k.isValid(), "the gather reference pipeline must build");
            if (!k.isValid()) return false;
            k.dispatch(1u, {}, p);
        }
        dev->flush();
        const uint32_t nr = downloadBuffer<uint32_t>(dev, bN.h, 1)[0];
        if (nr == POISON_U32 || nr > FIND_MAXROUNDS)
        {
            PXG_CHECK_MSG(false, "the gather reference reported %u rounds", nr);
            return false;
        }
        const auto hdr = downloadBuffer<uint32_t>(dev, bR.h, FIND_MAXROUNDS * 4u);
        const auto buf = downloadBuffer<uint32_t>(dev, bB.h, size_t(FIND_MAXROUNDS) * FIND_TARGET);
        ref.rounds.resize(nr);
        for (uint32_t r = 0; r < nr; ++r)
        {
            const uint32_t n = hdr[size_t(r) * 4];
            ref.rounds[r].assign(buf.begin() + size_t(r) * FIND_TARGET,
                                 buf.begin() + size_t(r) * FIND_TARGET + n);
        }
    }

    // ---- the per-particle decision ----
    {
        const std::vector<float> init(size_t(n) * PART_STRIDE, -4242.f);
        DevBuf bO = R.mk(dev, init);
        P_Part p{};
        p.meshPtr = R.bBlob.addr;
        p.positions = positionsAddr;
        p.positionsPrev = positionsPrevAddr;
        p.outBuf = bO.addr;
        p.aToBq[0] = aToB.q.x; p.aToBq[1] = aToB.q.y; p.aToBq[2] = aToB.q.z; p.aToBq[3] = aToB.q.w;
        p.scaleRot[0] = S.scaleRot.x; p.scaleRot[1] = S.scaleRot.y;
        p.scaleRot[2] = S.scaleRot.z; p.scaleRot[3] = S.scaleRot.w;
        p.aToBp[0] = aToB.p.x; p.aToBp[1] = aToB.p.y; p.aToBp[2] = aToB.p.z;
        p.cullScale = cull;
        p.scale[0] = S.scale.x; p.scale[1] = S.scale.y; p.scale[2] = S.scale.z;
        p.contactDist = S.contactDist();
        p.restDist = S.restDist;
        p.numPairs = n;
        p.enableCCD = opt.enableCcd ? 1u : 0u;
        {
            Kernel k(Kernel::Bindless{}, "npSdfParticle.spv", "sdfParticleContactAt", {},
                     sizeof(P_Part));
            PXG_CHECK_MSG(k.isValid(), "the contact reference pipeline must build");
            if (!k.isValid()) return false;
            k.dispatch((n + 63u) / 64u, {}, p);
        }
        dev->flush();
        const auto out = downloadBuffer<float>(dev, bO.h, size_t(n) * PART_STRIDE);
        ref.candidate.assign(n, 0u);
        ref.normalSep.assign(n, F4{ 0, 0, 0, 0 });
        for (uint32_t i = 0; i < n; ++i)
        {
            const float* o = &out[size_t(i) * PART_STRIDE];
            uint32_t cand; std::memcpy(&cand, &o[53], 4);
            ref.candidate[i] = cand != 0u ? 1u : 0u;
            ref.normalSep[i] = F4{ o[49], o[50], o[51], o[52] };
        }
    }
    return true;
}


// ---------------------------------------------------------------------------
// The expectation, built ENTIRELY from the two reference dispatches.
// ---------------------------------------------------------------------------
struct Expect
{
    std::vector<uint32_t> oneWayCount;     // final per-particle counter
    std::vector<int> oneWaySlotOf;         // -1, else the column a particle got
    std::vector<F4> oneWayInfo;            // by particle, when stored
    std::vector<uint32_t> writerParticle;  // the SORTED index, in writer order
    std::vector<F4> writerNormalPen;       // world-space normal
    uint32_t reserved = 0u;                // what numTotalContacts must end at
    uint32_t inFilterList = 0u;            // candidates the 5.6.1 filter list named
};

// One pass -- solid (isDiffuse false) or diffuse (true).  Everything that
// differs between them is a parameter, so the two are predicted by the SAME
// code and a mutant that mixes them up cannot be right for both.
static Expect predict(const Scene& S, const Options& opt, const Reference& ref,
                      uint32_t n, bool isDiffuse)
{
    Expect E;
    E.oneWayCount.assign(n, opt.seedOneWay);
    E.oneWaySlotOf.assign(n, -1);
    E.oneWayInfo.assign(n, F4{ 0, 0, 0, 0 });

    for (const std::vector<uint32_t>& round : ref.rounds)
    {
        // The block scan runs in thread order, and thread t holds round[t].
        std::vector<uint32_t> survivors;
        for (uint32_t tid = 0; tid < round.size(); ++tid)
        {
            const uint32_t ind = round[tid];
            bool cand = ref.candidate[ind] != 0u;

            // W5 / W6: PhysX 5.12 has no particle-rigid filter, so a candidate the
            // 5.6.1 list named (by its UNSORTED index) stays a candidate in both passes.
            if (cand)
            {
                const uint32_t unsorted = S.sortedToUnsorted[ind];
                const bool listed =
                    std::find(S.filteredUnsorted.begin(), S.filteredUnsorted.end(), unsorted)
                    != S.filteredUnsorted.end();
                if (listed) ++E.inFilterList;
            }

            if (opt.staticRigid && cand)
            {
                const uint32_t slot = E.oneWayCount[ind]++;    // W7: unconditional
                if (slot < Scene::ONEWAY_COLS)
                {
                    E.oneWaySlotOf[ind] = int(slot);
                    E.oneWayInfo[ind] = ref.normalSep[ind];    // W10: mesh-local
                    cand = false;                              // W9
                }
            }
            if (cand)
                survivors.push_back(ind);
        }
        // W11: the reservation counts every survivor, diffuse or not -- the
        // contact stream advances even when nothing is written into it.
        E.reserved += uint32_t(survivors.size());
        if (!isDiffuse)
            for (uint32_t ind : survivors)
            {
                const F4 nrm = ref.normalSep[ind];
                const PxVec3 world = S.meshQ.rotate(PxVec3(nrm.x, nrm.y, nrm.z));   // ENGINE
                E.writerParticle.push_back(ind);
                E.writerNormalPen.push_back(F4{ world.x, world.y, world.z, nrm.w });
            }
    }
    return E;
}

// The one-way side of a pass: counters, the column-major slot each particle got,
// its node index, and the requirement that every other slot is still poison.
static void checkOneWay(const char* label, const char* which, uint32_t n, const Expect& E,
                        const std::vector<uint32_t>& counts, const std::vector<F4>& infos,
                        const std::vector<uint64_t>& nodes, uint64_t rigidInd)
{
    size_t badCount = 0, badInfo = 0, badNode = 0, badClean = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (counts[i] != E.oneWayCount[i])
        {
            if (badCount < 4)
                std::printf("    %s/%s particle %u: one-way count %u, expected %u\n",
                            label, which, i, counts[i], E.oneWayCount[i]);
            ++badCount;
        }
        for (uint32_t c = 0; c < Scene::ONEWAY_COLS; ++c)
        {
            const size_t at = size_t(i) + size_t(c) * n;   // W8: column-major, THIS pass's n
            const bool live = (E.oneWaySlotOf[i] >= 0) && (uint32_t(E.oneWaySlotOf[i]) == c);
            if (live)
            {
                const F4 w = E.oneWayInfo[i];
                if (f2u(infos[at].x) != f2u(w.x) || f2u(infos[at].y) != f2u(w.y) ||
                    f2u(infos[at].z) != f2u(w.z) || f2u(infos[at].w) != f2u(w.w))
                {
                    if (badInfo < 4)
                        std::printf("    %s/%s particle %u slot %u: (%.9g,%.9g,%.9g,%.9g) expected "
                                    "(%.9g,%.9g,%.9g,%.9g)\n", label, which, i, c, infos[at].x,
                                    infos[at].y, infos[at].z, infos[at].w, w.x, w.y, w.z, w.w);
                    ++badInfo;
                }
                if (nodes[at] != rigidInd) ++badNode;
            }
            else if (infos[at].x != -9.f || nodes[at] != ~0ull)
            {
                ++badClean;
            }
        }
    }
    PXG_CHECK_MSG(badCount == 0, "%s/%s: %zu one-way counters wrong", label, which, badCount);
    PXG_CHECK_MSG(badInfo == 0, "%s/%s: %zu one-way contacts wrong", label, which, badInfo);
    PXG_CHECK_MSG(badNode == 0, "%s/%s: %zu one-way node indices wrong", label, which, badNode);
    PXG_CHECK_MSG(badClean == 0, "%s/%s: %zu one-way slots were written that should be poison",
                  label, which, badClean);
}

// ---------------------------------------------------------------------------
static void runOne(PxgSlangComputeDevice* dev, const char* label, const Options& opt)
{
    Scene S;
    buildScene(S);

    Rig R;
    if (!buildRig(dev, S, opt, R)) { R.destroy(dev); return; }

    const bool diffuseRuns = opt.maxDiffuse != 0u;

    Reference refS, refD;
    if (!buildReference(dev, S, opt, R, R.bPos.addr, R.bPrev.addr, Scene::NP, refS))
    { R.destroy(dev); return; }
    if (diffuseRuns &&
        !buildReference(dev, S, opt, R, R.bPosD.addr, R.bPrevD.addr, Scene::NPD, refD))
    { R.destroy(dev); return; }

    {
        Kernel k(Kernel::Bindless{}, "psSdfMidphaseBda.spv", "ps_sdfMeshCollisonLaunch", {},
                 sizeof(PsSdfMidBlob));
        PXG_CHECK_MSG(k.isValid(), "%s: the kernel pipeline must build", label);
        if (!k.isValid()) { R.destroy(dev); return; }
        // The engine's grid, verbatim: (numTests, 2, 1).  Both passes, always.
        k.dispatch(1u, {}, R.blob, 2u, 1u);
    }
    dev->flush();

    const Expect ES = predict(S, opt, refS, Scene::NP, false);
    const Expect ED = diffuseRuns ? predict(S, opt, refD, Scene::NPD, true) : Expect{};

    const auto counts = downloadBuffer<uint32_t>(dev, R.bCounts.h, Scene::NP);
    const auto infos = downloadBuffer<F4>(dev, R.bInfos.h, size_t(Scene::NP) * Scene::ONEWAY_COLS);
    const auto nodes = downloadBuffer<uint64_t>(dev, R.bNodeIdx.h,
                                                size_t(Scene::NP) * Scene::ONEWAY_COLS);
    const auto countsD = downloadBuffer<uint32_t>(dev, R.bDiffCounts.h, Scene::NPD);
    const auto infosD = downloadBuffer<F4>(dev, R.bDiffInfos.h,
                                           size_t(Scene::NPD) * Scene::ONEWAY_COLS);
    const auto nodesD = downloadBuffer<uint64_t>(dev, R.bDiffNodeIdx.h,
                                                 size_t(Scene::NPD) * Scene::ONEWAY_COLS);
    const uint32_t total = downloadBuffer<uint32_t>(dev, R.bCount.h, 1)[0];
    const auto ctInfo = downloadBuffer<PxgDeformablePairInfo>(dev, R.bContacts.h, Scene::CAP);
    const auto ctNp = downloadBuffer<F4>(dev, R.bNormalPens.h, Scene::CAP);
    std::vector<PsC> ct(Scene::CAP);
    for (uint32_t i = 0; i < Scene::CAP; ++i)
    {
        ct[i].normal_pen = ctNp[i];
        ct[i].rLo = uint32_t(ctInfo[i].ref0);
        ct[i].rHi = uint32_t(ctInfo[i].ref0 >> 32);
        ct[i].pLo = PxgGetParticleIndex(ctInfo[i].ref1);
        ct[i].pHi = PxgGetParticleSystemId(ctInfo[i].ref1);
    }
    const auto sbp = downloadBuffer<uint64_t>(dev, R.bSbP.h, Scene::CAP);
    const auto tbp = downloadBuffer<uint32_t>(dev, R.bTbP.h, Scene::CAP);
    const auto ibp = downloadBuffer<uint32_t>(dev, R.bIbP.h, Scene::CAP);
    const auto br = downloadBuffer<uint64_t>(dev, R.bBr.h, Scene::CAP);

    const uint64_t rigidInd = R.rigidId.getInd();

    // Coverage, DEMANDED rather than designed: the solid gather must resume at
    // least once, so the second round's `i` cursor and the groupshared buffer's
    // reuse are both exercised.
    PXG_CHECK_MSG(refS.rounds.size() >= 2, "%s: the solid gather took %zu round(s) -- the resume "
                  "path is untested", label, refS.rounds.size());
    // PhysX 5.12: the candidates the 5.6.1 filter list named must be in the solid
    // pass, so a port that still filtered them would lose contacts here.
    PXG_CHECK_MSG(ES.inFilterList > 0u, "%s: no solid candidate is one the 5.6.1 filter named, so "
                  "5.12's removal of the filter is unobservable", label);

    // ---- the solid pass's one-way arrays ----
    checkOneWay(label, "solid", Scene::NP, ES, counts, infos, nodes, rigidInd);

    // ---- the diffuse pass's ----
    if (diffuseRuns)
    {
        checkOneWay(label, "diffuse", Scene::NPD, ED, countsD, infosD, nodesD, rigidInd);
    }
    else
    {
        // mMaxDiffuseParticles == 0: the y=1 block returns before ANY write, so
        // the whole diffuse side must still be exactly as it was uploaded.
        size_t touched = 0;
        for (uint32_t i = 0; i < Scene::NPD; ++i)
            if (countsD[i] != opt.seedOneWay) ++touched;
        for (size_t j = 0; j < infosD.size(); ++j)
            if (infosD[j].x != -9.f || nodesD[j] != ~0ull) ++touched;
        PXG_CHECK_MSG(touched == 0, "%s: %zu diffuse words were written despite "
                      "mMaxDiffuseParticles == 0", label, touched);
    }

    // ---- the contact stream ----
    //
    // THE ONE THING THAT IS NOT DETERMINISTIC HERE, and is therefore not
    // asserted: WHICH indices the solid pass reserves.  Both blocks feed the
    // same numTotalContacts through one atomic, once per gather round, so the
    // interleaving is a race and the P9.11 rule applies -- never assert the
    // outcome of a race.  What IS determined:
    //   * the total is solid survivors + diffuse survivors (W11: the diffuse
    //     pass reserves indices it never writes);
    //   * the SET of written slots has exactly the solid survivors' size;
    //   * in increasing slot order the written contacts are the solid pass's,
    //     in its own round-then-thread order.
    PXG_CHECK_MSG(total == ES.reserved + ED.reserved,
                  "%s: numTotalContacts is %u, expected %u (solid %u + diffuse %u) -- the block "
                  "reservation counts every survivor, diffuse or not", label, total,
                  ES.reserved + ED.reserved, ES.reserved, ED.reserved);

    std::vector<uint32_t> written;
    for (uint32_t i = 0; i < Scene::CAP; ++i)
        if (ibp[i] != POISON_U32) written.push_back(i);

    size_t badCt = 0;
    PXG_CHECK_MSG(written.size() == ES.writerParticle.size(),
                  "%s: %zu contact slots written, expected %zu", label, written.size(),
                  ES.writerParticle.size());
    const size_t nCmp = PxMin(written.size(), ES.writerParticle.size());
    for (size_t k = 0; k < nCmp; ++k)
    {
        const uint32_t at = written[k];
        const F4 w = ES.writerNormalPen[k];
        const uint32_t ind = ES.writerParticle[k];
        const uint64_t compressed = PxgEncodeParticleRef(Scene::SYS, ind);
        if (f2u(ct[at].normal_pen.x) != f2u(w.x) || f2u(ct[at].normal_pen.y) != f2u(w.y) ||
            f2u(ct[at].normal_pen.z) != f2u(w.z) || f2u(ct[at].normal_pen.w) != f2u(w.w))
        {
            if (badCt < 4)
                std::printf("    %s contact %zu (slot %u): (%.9g,%.9g,%.9g,%.9g) expected "
                            "(%.9g,%.9g,%.9g,%.9g)\n", label, k, at, ct[at].normal_pen.x,
                            ct[at].normal_pen.y, ct[at].normal_pen.z, ct[at].normal_pen.w,
                            w.x, w.y, w.z, w.w);
            ++badCt;
        }
        // W5: the SORTED index reaches the writer, not the unsorted one.
        if (ct[at].pLo != ind || ct[at].pHi != Scene::SYS) ++badCt;
        if (sbp[at] != compressed) ++badCt;
        if (tbp[at] != ind) ++badCt;
        if (ibp[at] != at) ++badCt;
        if (br[at] != rigidInd) ++badCt;
        if (at >= total) ++badCt;                 // a written slot is a reserved slot
    }
    PXG_CHECK_MSG(badCt == 0, "%s: %zu contact-stream comparisons failed", label, badCt);

    size_t nOneWay = 0, nCapped = 0;
    for (uint32_t i = 0; i < Scene::NP; ++i)
    {
        if (ES.oneWaySlotOf[i] >= 0) ++nOneWay;
        if (ES.oneWayCount[i] > Scene::ONEWAY_COLS) ++nCapped;
    }
    if (opt.staticRigid && opt.seedOneWay >= Scene::ONEWAY_COLS)
        PXG_CHECK_MSG(nCapped > 0 && nOneWay == 0 && !ES.writerParticle.empty(),
                      "%s: the over-capacity configuration must drop every one-way write (%zu "
                      "stored) and push the candidates into the stream (%zu there)", label,
                      nOneWay, ES.writerParticle.size());
    size_t nOneWayD = 0;
    for (uint32_t i = 0; i < ED.oneWaySlotOf.size(); ++i)
        if (ED.oneWaySlotOf[i] >= 0) ++nOneWayD;
    std::printf("    [ note ] %s: solid %zu rounds / %zu one-way (%zu over the cap) / %zu stream; "
                "diffuse %zu rounds / %zu one-way / %u reserved; total %u\n", label,
                refS.rounds.size(), nOneWay, nCapped, ES.writerParticle.size(),
                refD.rounds.size(), nOneWayD, ED.reserved, total);
    R.destroy(dev);
}

static PxgCudaShim* gShim = nullptr;

static bool beginShim(PxgSlangComputeDevice* dev, PxgCudaShim*& shim)
{
    static const PxgKernelDef manifest[] = {
        { "psSdfMidphaseBda.spv", "ps_sdfMeshCollisonLaunch", 0u, 0u } };
    static auto loader = [](const char* f) { return loadShaderBlob(f); };
    shim = new PxgCudaShim(dev, manifest, 1, +loader);
    physx::gpuslang::pxgSetVulkanDriverShim(shim);
    return true;
}

static void endShim(PxgCudaShim*& shim)
{
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
    delete shim;
    shim = nullptr;
}

PXG_TEST(psSdfMidphase)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    PxgCudaShim* shim = nullptr;
    beginShim(dev, shim);

    // Both slot orders, so psLoadShapePair's swap is exercised in both
    // directions; a static and a dynamic rigid, so the one-way path and the
    // stream path each run alone; and a pre-seeded counter so the capacity cap
    // is reached without needing 12 contacts per particle.  maxDiffuse stays 0
    // throughout, so the y=1 block takes its early-out in every one of these --
    // which is checked, not assumed: the whole diffuse side must come back
    // untouched.
    { Options o; o.psFirst = false; o.staticRigid = true;  runOne(dev, "mesh-first/static", o); }
    { Options o; o.psFirst = true;  o.staticRigid = true;  runOne(dev, "ps-first/static", o); }
    { Options o; o.psFirst = false; o.staticRigid = false; runOne(dev, "mesh-first/dynamic", o); }
    // The last slot that still fits, and the first that does not.  The second is
    // the one that matters: W7 increments the counter unconditionally, W9 clears
    // the candidate ONLY when the contact was actually stored, so an over-full
    // particle keeps its candidacy and reaches the contact stream even though
    // the rigid is static.
    { Options o; o.psFirst = false; o.staticRigid = true; o.seedOneWay = Scene::ONEWAY_COLS - 1u;
      runOne(dev, "static/at-capacity", o); }
    { Options o; o.psFirst = false; o.staticRigid = true; o.seedOneWay = Scene::ONEWAY_COLS;
      runOne(dev, "static/over-capacity", o); }
    { Options o; o.psFirst = false; o.staticRigid = false; o.enableCcd = true;
      runOne(dev, "dynamic/ccd", o); }

    endShim(shim);
}

PXG_TEST(psSdfMidphaseDiffuse)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    PxgCudaShim* shim = nullptr;
    beginShim(dev, shim);

    // W6: the diffuse pass produces one-way contacts and never touches the
    // contact stream -- but W11 still advances the reservation, so the total is
    // the SUM of both passes' survivors while only the solid pass's are written.
    { Options o; o.maxDiffuse = Scene::NPD; o.staticRigid = true;
      runOne(dev, "diffuse/static", o); }
    { Options o; o.maxDiffuse = Scene::NPD; o.staticRigid = false;
      runOne(dev, "diffuse/dynamic", o); }
    { Options o; o.maxDiffuse = Scene::NPD; o.staticRigid = false; o.enableCcd = true;
      runOne(dev, "diffuse/ccd", o); }
    { Options o; o.maxDiffuse = Scene::NPD; o.psFirst = true; o.staticRigid = true;
      o.seedOneWay = Scene::ONEWAY_COLS; runOne(dev, "diffuse/over-capacity", o); }

    endShim(shim);
}

}  // namespace
