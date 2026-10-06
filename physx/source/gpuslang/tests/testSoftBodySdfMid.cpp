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

// testSoftBodySdfMid.cpp -- P9.35c, the fixture for
// sb_sdfMeshMidphaseGeneratePairsLaunch.
//
// The same arrangement as testFemClothSdfCg: a WRAPPER CHECKED AGAINST ITS OWN
// PARTS.  The expectation comes from two independent dispatches of the two
// modules' own entry points --
//
//   npSdfFind.spv:sdfFindTetRounds     -> the gathered tet list, per round
//   npSdfCollision.spv:sdfCollTetrahedron -> point / dir / separation per tet
//
// -- each fed the cull tolerance and the inverse pose that the HOST computed, so
// a kernel that derives either differently gathers a different set or descends
// from a different tet.  Everything else on the host is an ENGINE call
// (PxQuat::rotate/rotateInv, PxTransform::transform/rotate/getInverse) or four
// lines over one.
//
// WHAT THIS KERNEL PUTS UNDER TEST that the cloth one did not:
//   * cullScale takes the ABSOLUTE value, where the cloth sibling does not AND
//     where the gather this kernel itself calls does not.  One scene with a
//     negative component decides all three.
//   * the a-to-b transform is the BARE inverse of the mesh pose; the softbody's
//     own transform cache is never read, so the fixture poisons it.
//   * the filter runs LAST, on the TET index, inside the candidate test.
//   * the writer is writeContactNoBarycentric: no barycentric, no material.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgSoftBody.h"
#include "PxgFEMCore.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgSimulationCoreDesc.h"
#include "PxsTransformCache.h"
#include "PxNodeIndex.h"
#include "geometry/PxGeometry.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "PxgVulkanCudaContext.h"      // P9.97b: the engine's launch bridge
#include "PxgVulkanKernelAbi.h"        // P9.97b: the row the bridge lays the blob out from
#include "cudamanager/PxCudaContext.h" // PxCudaKernelParam, PX_CUDA_KERNEL_PARAM
#include "foundation/PxBounds3.h"
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

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

// softBodySdfMidphaseBda.slang's SbSdfMidParams, byte for byte -- and, since P9.97b,
// byte for byte the blob the BRIDGE builds from the engine's host pack too.  The bridge
// places each argument at paramAlign(size), and paramAlign(80) = 16, so the 80-byte
// writer starts at @96 behind 12 bytes of padding, not at the @88 its natural alignment
// of 8 would give.  Until P9.97b this struct said @88 / 168 bytes like the shader did,
// and every fixture here passed while the engine path read the writer one slot off;
// softBodySdfMidphaseBridgeSized below launches through the bridge to hold it.
struct SbSdfBlob
{
    float tolerenceLength;             // @0   dead
    uint32_t pad4;                     // @4
    uint64_t cmInputs;                 // @8
    uint64_t transformCache;           // @16
    uint64_t bounds;                   // @24  dead
    uint64_t contactDistance;          // @32
    uint64_t restDistance;             // @40
    uint64_t gpuShapes;                // @48
    uint64_t softbodies;               // @56
    uint64_t shapeToRigidRemap;        // @64
    uint64_t filterPairs;              // @72
    uint32_t nbFilterPairs;            // @80
    uint32_t pad84;                    // @84
    uint64_t pad88;                    // @88  paramAlign(80) = 16
    uint64_t w_outPoint;               // @96
    uint64_t w_outNormalPen;           // @104
    uint64_t w_outBarycentric;         // @112
    uint64_t w_outContactInfo;         // @120
    uint64_t w_totalContactCount;      // @128
    uint64_t w_contactByX;             // @136
    uint64_t w_tempContactByX;         // @144
    uint64_t w_contactIndexSortedByX;  // @152
    uint64_t w_contactSortedByX;       // @160
    uint32_t w_maxNumContacts;         // @168
    uint32_t pad172;                   // -> 176
};
static_assert(sizeof(SbSdfBlob) == 176, "engine ABI blob size (pxgKernelAbiPushSize of the row)");
static_assert(offsetof(SbSdfBlob, w_outPoint) == 96, "the writer lands at 96: paramAlign(80) = 16");
static_assert(offsetof(SbSdfBlob, w_maxNumContacts) == 168, "the writer's count lands at 96 + 72");

// npSdfFind.slang's SdfFindParams.
struct P_Find
{
    uint64_t meshPtr, positions, echoOut, roundsOut, bufOut, numRoundsOut;
    float aToBq[4];
    float scaleRot[4];
    float aToBp[3];  float tolerance;
    float scale[3];  uint32_t numVertices;
    uint32_t maxRounds, f0, f1, f2;
    uint64_t meshInd, touchedOut;
    float scale0Rot[4];
    float scale0[3]; uint32_t f3;
};
static_assert(sizeof(P_Find) == 176, "npSdfFind push block");

// npSdfCollision.slang's SdfCollParams.
struct P_Coll
{
    uint64_t meshPtr, points, gradIn, dirSeed, outBuf;
    uint32_t numPoints, cpad;
    uint64_t triVerts, triOut;
};

static const uint32_t FIND_TARGET = 1024u;
static const uint32_t FIND_MAXROUNDS = 24u;
static const uint32_t TET_TRACE_STRIDE = 256u;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;
static const uint32_t PXG_BITSHIFT_TET_ID_H = 20u;

static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }

static PxVec3 hVertex2Shape(const PxVec3& v, const PxVec3& scale, const PxQuat& rotation)
{
    PxVec3 t = rotation.rotate(v);
    t.x *= scale.x; t.y *= scale.y; t.z *= scale.z;
    return rotation.rotateInv(t);
}
static PxVec3 hShape2Vertex(const PxVec3& v, const PxVec3& scale, const PxQuat& rotation)
{
    PxVec3 t = rotation.rotate(v);
    t.x /= scale.x; t.y /= scale.y; t.z /= scale.z;
    return rotation.rotateInv(t);
}
static uint32_t encodeSb(uint32_t sbId, uint32_t tet)
{
    return (sbId << PXG_BITSHIFT_TET_ID_H) | tet;
}

// ---------------------------------------------------------------------------
// The scene: one SDF trimesh, one softbody, one or two contact managers.
// ---------------------------------------------------------------------------
struct Scene
{
    static const uint32_t TRI_SHAPE = 3u, SB_SHAPE = 1u, NB_SHAPES = 6u;
    static const uint32_t TRI_CACHE = 5u, SB_CACHE = 2u, NB_CACHE = 9u;
    static const uint32_t SB_ID = 2u, NB_BODIES = 5u;
    static const uint32_t NB_TETS = 4000u;
    static const uint32_t CAP = 8192u;

    SdfScene sdf;
    BlobLayout blob;
    PxVec3 meshP; PxQuat meshQ;
    PxVec3 scale; PxQuat scaleRot;
    float contactDistTri = 0.0f, contactDistSb = 0.0f;
    std::vector<float> restByPair;
    std::vector<F4> verts;
    std::vector<uint32_t> inds;
    std::vector<uint32_t> filteredTets;

    PxTransform meshPose() const { return PxTransform(meshP, meshQ); }
    float worldContactDist() const { return contactDistTri + contactDistSb; }
    // T2: the smallest MAGNITUDE.
    float cullScale() const
    {
        return worldContactDist() /
               PxMin(PxAbs(scale.x), PxMin(PxAbs(scale.y), PxAbs(scale.z)));
    }
    // T3: the BARE inverse -- the softbody's own pose is never involved.
    PxTransform aToB() const { return meshPose().getInverse(); }
};

static void buildScene(Scene& S, bool negScale)
{
    buildSparse(S.sdf);
    S.blob = layOutBlob(S.sdf, 16u, 21u, 3u, 1u, 0u);

    S.meshQ = PxQuat(0.53f, PxVec3(0.36f, -0.8f, 0.48f).getNormalized());
    S.meshP = PxVec3(0.62f, -0.31f, 1.15f);
    // The negative variant puts the negative component where it is NOT the
    // smallest in magnitude, so min(x,y,z) and min(|x|,|y|,|z|) differ in value
    // AND in sign -- which is what separates the kernel's cullScale (abs) from
    // the gather's radius divide (no abs) INSIDE ONE RUN.
    S.scale = negScale ? PxVec3(1.17f, -2.90f, 0.86f) : PxVec3(1.35f, 0.72f, 2.10f);
    S.scaleRot = PxQuat(-0.44f, PxVec3(0.8f, 0.48f, 0.36f).getNormalized());
    S.contactDistTri = 0.052f;
    S.contactDistSb = 0.31f;

    S.restByPair.assign(Scene::NB_CACHE, 0.0f);
    for (uint32_t i = 0; i < Scene::NB_CACHE; ++i) S.restByPair[i] = -3.0f - 0.125f * float(i);
    S.restByPair[0] = 0.019f;
    S.restByPair[1] = 0.041f;

    const HostSdfBox box = hostSdfBox(S.sdf);
    const PxVec3 ext = box.higher - box.lower;
    const PxTransform bToA = S.aToB().getInverse();

    S.verts.resize(size_t(Scene::NB_TETS) * 4u);
    S.inds.resize(size_t(Scene::NB_TETS) * 4u);
    std::mt19937 rng(0x5DF35Cu);
    auto u01 = [&]() { return float(rng() % 1000000u) / 1000000.f; };
    for (uint32_t t = 0; t < Scene::NB_TETS; ++t)
    {
        const float sz = (t % 7u == 0u) ? 0.55f * ext.x
                                        : 0.02f * ext.x * (1.0f + float(t % 5));
        PxVec3 c;
        if (t < Scene::NB_TETS - 200u)
            c = PxVec3(box.lower.x - 0.35f * ext.x + 1.7f * ext.x * u01(),
                       box.lower.y - 0.35f * ext.y + 1.7f * ext.y * u01(),
                       box.lower.z - 0.35f * ext.z + 1.7f * ext.z * u01());
        else
            c = PxVec3(box.lower.x - (8.0f + 3.0f * u01()) * ext.x,
                       box.lower.y - (8.0f + 3.0f * u01()) * ext.y,
                       box.lower.z - (8.0f + 3.0f * u01()) * ext.z);
        const PxVec3 corner[4] = { c + PxVec3(sz, 0.0f, 0.0f),
                                   c + PxVec3(-0.3f * sz, 0.9f * sz, 0.2f * sz),
                                   c + PxVec3(0.1f * sz, -0.4f * sz, -0.8f * sz),
                                   c + PxVec3(-0.2f * sz, -0.35f * sz, 0.75f * sz) };
        for (uint32_t k = 0; k < 4u; ++k)
        {
            // T4: laid out in the trimesh's SHAPE space and mapped back through the
            // inverse of the a-to-b transform ALONE -- no scale on the softbody side,
            // which is what makes the forward chain the kernel's.
            const PxVec3 world = bToA.transform(hShape2Vertex(corner[k], S.scale, S.scaleRot));
            S.verts[t * 4u + k] = F4{ world.x, world.y, world.z, 0.5f + 0.001f * float(k) };
        }
        const uint32_t r = t % 4u;
        for (uint32_t k = 0; k < 4u; ++k)
            S.inds[t * 4u + k] = t * 4u + (r + k) % 4u;
    }

    // T5: the filter is queried with the TET index, once.
    S.filteredTets.clear();
    for (uint32_t t = 5u; t < Scene::NB_TETS; t += 23u) S.filteredTets.push_back(t);
}

// ---------------------------------------------------------------------------
struct Options
{
    bool sbFirst = true;
    uint32_t numPairs = 1u;
    uint32_t capacity = Scene::CAP;
    uint32_t seedCount = 0u;
    bool negScale = false;
    bool exactGate = false;   // the manufactured cull-gate boundary
    // P9.97b: when set, the kernel is launched the way the ENGINE launches it -- the
    // host pack of PxgNarrowphaseCore.cpp:4888-4903 through PxCudaContext::launchKernel,
    // which lays the blob out from the argument sizes (paramAlign) -- instead of with
    // this file's own SbSdfBlob.
    physx::PxCudaContext* bridgeCtx = nullptr;
    CUfunction bridgeFn = nullptr;
};

// P9.97b -- a HOST mirror of PxgFEMContactWriter (PxgFEMCore.h:361-375), the 80-byte
// struct the engine passes BY VALUE as the last argument.  The engine type has no
// constructor this TU can call (it takes a PxgFEMCore*), so the mirror stands in for it,
// pinned to it member by member: same size, same alignment, same offsets.  What the
// bridge sees is only (pointer, sizeof) of this object, so pinning those is pinning
// everything the launch depends on.
struct HostFemWriter
{
    float4* outPoint;
    float4* outNormalPen;
    float4* outBarycentric;
    PxgFemOtherContactInfo* outContactInfo;
    PxU32* totalContactCount;
    PxU64* contactByX;
    PxU32* tempContactByX;
    PxU32* contactIndexSortedByX;
    PxU32* contactSortedByX;
    PxU32 maxNumContacts;
};
static_assert(sizeof(HostFemWriter) == sizeof(PxgFEMContactWriter) && sizeof(HostFemWriter) == 80,
              "the by-value writer is 80 bytes (9 pointers + PxU32 + 4 tail pad)");
static_assert(alignof(HostFemWriter) == alignof(PxgFEMContactWriter) && alignof(HostFemWriter) == 8,
              "alignof(writer) is 8 -- the bridge nevertheless places it at paramAlign(80) = 16");
#define PXG_SBSDF_PIN(f) static_assert(offsetof(HostFemWriter, f) == offsetof(PxgFEMContactWriter, f), #f)
PXG_SBSDF_PIN(outPoint); PXG_SBSDF_PIN(outNormalPen); PXG_SBSDF_PIN(outBarycentric);
PXG_SBSDF_PIN(outContactInfo); PXG_SBSDF_PIN(totalContactCount); PXG_SBSDF_PIN(contactByX);
PXG_SBSDF_PIN(tempContactByX); PXG_SBSDF_PIN(contactIndexSortedByX);
PXG_SBSDF_PIN(contactSortedByX); PXG_SBSDF_PIN(maxNumContacts);
#undef PXG_SBSDF_PIN

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bBlob, bCin, bShapes, bXf, bCd, bRest, bRemap, bPairs, bDesc, bBounds;
    DevBuf bVerts, bInds;
    DevBuf bPoint, bNormalPen, bBary, bInfo, bCount, bByX, bTempByX, bIdxByX, bSortedByX;
    SbSdfBlob blob{};
    uint64_t texCoarse = 0, texSub = 0;
    PxNodeIndex rigidId;
    std::vector<PxgRigidFilterPair> pairs;

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

    R.bVerts = R.mk(dev, S.verts);
    R.bInds = R.mk(dev, S.inds);

    std::vector<PxgSoftBody> descArr(Scene::NB_BODIES);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgSoftBody));
    PxgSoftBody sb; std::memset(&sb, 0, sizeof(sb));
    sb.mPosition_InvMass = reinterpret_cast<decltype(sb.mPosition_InvMass)>(size_t(R.bVerts.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(R.bInds.addr));
    sb.mNumTets = Scene::NB_TETS;
    descArr[Scene::SB_ID] = sb;
    R.bDesc = R.mk(dev, descArr);

    std::vector<PxgContactManagerInput> cin(opt.numPairs);
    std::memset(cin.data(), 0xEE, cin.size() * sizeof(PxgContactManagerInput));
    for (uint32_t p = 0; p < opt.numPairs; ++p)
    {
        if (opt.sbFirst)
        {
            cin[p].shapeRef0 = Scene::SB_SHAPE;  cin[p].transformCacheRef0 = Scene::SB_CACHE;
            cin[p].shapeRef1 = Scene::TRI_SHAPE; cin[p].transformCacheRef1 = Scene::TRI_CACHE;
        }
        else
        {
            cin[p].shapeRef0 = Scene::TRI_SHAPE; cin[p].transformCacheRef0 = Scene::TRI_CACHE;
            cin[p].shapeRef1 = Scene::SB_SHAPE;  cin[p].transformCacheRef1 = Scene::SB_CACHE;
        }
    }
    R.bCin = R.mk(dev, cin);

    std::vector<PxgShape> shapes(Scene::NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    // Every decoy is a TRIANGLE mesh with a null geometry pointer, so a kernel
    // that picks the wrong half of the pair both mis-sorts and then faults.
    for (auto& sh : shapes)
    {
        sh.type = PxU32(PxGeometryType::eTRIANGLEMESH);
        sh.scale = PxMeshScale(PxVec3(4.0f, 4.0f, 4.0f), PxQuat(PxIdentity));
        sh.hullOrMeshPtr = 0;
        sh.particleOrSoftbodyId = 0xffffffffu;
        sh.materialIndex = 0x77u;
    }
    shapes[Scene::TRI_SHAPE].scale = PxMeshScale(S.scale, S.scaleRot);
    shapes[Scene::TRI_SHAPE].hullOrMeshPtr = size_t(R.bBlob.addr);
    shapes[Scene::TRI_SHAPE].materialIndex = 0x2Bu;
    shapes[Scene::SB_SHAPE].type = PxU32(PxGeometryType::eTETRAHEDRONMESH);
    shapes[Scene::SB_SHAPE].particleOrSoftbodyId = Scene::SB_ID;
    // T4: the softbody's shape scale is never applied; loud on purpose.
    shapes[Scene::SB_SHAPE].scale = PxMeshScale(PxVec3(0.31f, 5.7f, 0.09f),
                                                PxQuat(0.6f, PxVec3(0.48f, -0.6f, 0.64031f)
                                                                 .getNormalized()));
    shapes[Scene::SB_SHAPE].materialIndex = 0x99u;
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> xf(Scene::NB_CACHE);
    std::memset(xf.data(), 0, xf.size() * sizeof(PxsCachedTransform));
    // T3: EVERY slot but the trimesh's is poison, INCLUDING the softbody's.  The
    // kernel never reads it; a port that composes both poses lands on garbage.
    for (auto& t : xf) t.transform = PxTransform(PxVec3(77.f, 77.f, 77.f), PxQuat(PxIdentity));
    xf[Scene::TRI_CACHE].transform = S.meshPose();
    R.bXf = R.mk(dev, xf);

    std::vector<float> cd(Scene::NB_CACHE, -5.f);
    cd[Scene::TRI_CACHE] = S.contactDistTri;
    cd[Scene::SB_CACHE] = S.contactDistSb;
    R.bCd = R.mk(dev, cd);
    R.bRest = R.mk(dev, S.restByPair);
    // T1: `bounds` is never read, so it points at something that would be wrong.
    R.bBounds = R.mk(dev, std::vector<float>(Scene::NB_CACHE * 6u, -12345.f));

    R.rigidId = PxNodeIndex(0x51u, 3u);
    std::vector<uint64_t> remap(Scene::NB_CACHE, 0xDEADBEEFDEADBEEFull);
    remap[Scene::TRI_CACHE] = R.rigidId.getInd();
    R.bRemap = R.mk(dev, remap);

    // The filter array carries FOUR distinct rigid ids, not just the one this
    // kernel queries with, and that is what makes the comparison itself testable.
    //
    // With a single id, index0 is constant across the array, so the search reduces
    // to index1 alone -- and index1's high word is always zero, because it is a
    // PxU32 element handle widened to PxU64.  Under those conditions comparing
    // index1 BEFORE index0, or comparing the two 32-bit halves low-word-first,
    // both give the same order and the same answer.  The first sweep left exactly
    // those two mutants alive.
    //
    // The decoys are chosen so neither shortcut survives: they share low words
    // across different high words (0x51 appears with link 0 and with link 3) and
    // share high words across different low words, so the ENGINE's ordering --
    // index0 first, as an unsigned 64-bit value, high word significant -- is the
    // only one under which the array is sorted and the binary search lands.
    R.pairs.clear();
    const PxNodeIndex decoys[3] = { PxNodeIndex(0x51u, 0u),      // same low word, link 0
                                    PxNodeIndex(0x99u, 1u),
                                    PxNodeIndex(0x10u, 7u) };    // sorts LAST only high-word-first
    for (uint32_t t : S.filteredTets)
    {
        PxgRigidFilterPair p;
        p.index0 = R.rigidId.getInd();
        p.index1 = encodeSb(Scene::SB_ID, t);
        p.index2 = 0u;
        R.pairs.push_back(p);
    }
    for (uint32_t d = 0; d < 3u; ++d)
        for (uint32_t t = 1u; t < Scene::NB_TETS; t += 17u)
        {
            PxgRigidFilterPair p;
            p.index0 = decoys[d].getInd();
            p.index1 = encodeSb(Scene::SB_ID, t);
            p.index2 = 0u;
            R.pairs.push_back(p);
        }
    std::sort(R.pairs.begin(), R.pairs.end(),
              [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
                  if (a.index0 != b.index0) return a.index0 < b.index0;
                  if (a.index1 != b.index1) return a.index1 < b.index1;
                  return a.index2 < b.index2;
              });
    R.bPairs = R.mk(dev, R.pairs);

    const std::vector<F4> f4Init(Scene::CAP, F4{ -7.f, -7.f, -7.f, -7.f });
    const std::vector<uint32_t> u32Init(Scene::CAP, POISON_U32);
    const std::vector<uint64_t> u64Init(Scene::CAP, ~0ull);
    const std::vector<uint32_t> infoInit(size_t(Scene::CAP) * 4u, POISON_U32);
    R.bPoint = R.mk(dev, f4Init);
    R.bNormalPen = R.mk(dev, f4Init);
    R.bBary = R.mk(dev, f4Init);
    R.bInfo = R.mk(dev, infoInit);
    R.bCount = R.mk(dev, std::vector<uint32_t>(1u, opt.seedCount));
    R.bByX = R.mk(dev, u64Init);
    R.bTempByX = R.mk(dev, u32Init);
    R.bIdxByX = R.mk(dev, u32Init);
    R.bSortedByX = R.mk(dev, u32Init);

    R.blob = SbSdfBlob{};
    R.blob.tolerenceLength = 1234.5f;             // T1: must never matter
    R.blob.cmInputs = R.bCin.addr;
    R.blob.transformCache = R.bXf.addr;
    R.blob.bounds = R.bBounds.addr;
    R.blob.contactDistance = R.bCd.addr;
    R.blob.restDistance = R.bRest.addr;
    R.blob.gpuShapes = R.bShapes.addr;
    R.blob.softbodies = R.bDesc.addr;
    R.blob.shapeToRigidRemap = R.bRemap.addr;
    R.blob.filterPairs = R.bPairs.addr;
    R.blob.nbFilterPairs = uint32_t(R.pairs.size());
    R.blob.w_outPoint = R.bPoint.addr;
    R.blob.w_outNormalPen = R.bNormalPen.addr;
    R.blob.w_outBarycentric = R.bBary.addr;
    R.blob.w_outContactInfo = R.bInfo.addr;
    R.blob.w_totalContactCount = R.bCount.addr;
    R.blob.w_contactByX = R.bByX.addr;
    R.blob.w_tempContactByX = R.bTempByX.addr;
    R.blob.w_contactIndexSortedByX = R.bIdxByX.addr;
    R.blob.w_contactSortedByX = R.bSortedByX.addr;
    R.blob.w_maxNumContacts = opt.capacity;
    return true;
}

// ---------------------------------------------------------------------------
// Reference dispatch 1: the gather (P9.35b's own entry point).
// ---------------------------------------------------------------------------
static bool refGather(PxgSlangComputeDevice* dev, const Scene& S, Rig& R,
                      std::vector<std::vector<uint32_t>>& rounds,
                      float toleranceOverride = 0.0f)
{
    const PxTransform aToB = S.aToB();
    const std::vector<F4> echoInit(Scene::NB_TETS, F4{ 0, 0, 0, 0 });
    const std::vector<uint32_t> touchInit(Scene::NB_TETS, 0u);
    const std::vector<uint32_t> rInit(FIND_MAXROUNDS * 4u, POISON_U32);
    const std::vector<uint32_t> bInit(size_t(FIND_MAXROUNDS) * FIND_TARGET, POISON_U32);
    const std::vector<uint32_t> nInit(1u, POISON_U32);
    DevBuf bE = R.mk(dev, echoInit), bT = R.mk(dev, touchInit), bR = R.mk(dev, rInit);
    DevBuf bB = R.mk(dev, bInit), bN = R.mk(dev, nInit);

    P_Find p{};
    p.meshPtr = R.bBlob.addr;
    p.positions = R.bVerts.addr;
    p.meshInd = R.bInds.addr;
    p.echoOut = bE.addr; p.touchedOut = bT.addr;
    p.roundsOut = bR.addr; p.bufOut = bB.addr; p.numRoundsOut = bN.addr;
    p.aToBq[0] = aToB.q.x; p.aToBq[1] = aToB.q.y; p.aToBq[2] = aToB.q.z; p.aToBq[3] = aToB.q.w;
    p.aToBp[0] = aToB.p.x; p.aToBp[1] = aToB.p.y; p.aToBp[2] = aToB.p.z;
    p.scaleRot[0] = S.scaleRot.x; p.scaleRot[1] = S.scaleRot.y;
    p.scaleRot[2] = S.scaleRot.z; p.scaleRot[3] = S.scaleRot.w;
    p.scale[0] = S.scale.x; p.scale[1] = S.scale.y; p.scale[2] = S.scale.z;
    p.tolerance = (toleranceOverride != 0.0f) ? toleranceOverride : S.cullScale();
    p.numVertices = Scene::NB_TETS;
    p.maxRounds = FIND_MAXROUNDS;
    {
        Kernel k(Kernel::Bindless{}, "npSdfFind.spv", "sdfFindTetRounds", {}, sizeof(P_Find));
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
    rounds.assign(nr, {});
    for (uint32_t r = 0; r < nr; ++r)
    {
        const uint32_t n = hdr[size_t(r) * 4];
        for (uint32_t t = 0; t < n; ++t)
            rounds[r].push_back(buf[size_t(r) * FIND_TARGET + t]);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Reference dispatch 2: the descent (P9.35a's own entry point).
// ---------------------------------------------------------------------------
struct Descent { F4 point, dir; float sep; };

static bool refDescent(PxgSlangComputeDevice* dev, const Scene& S, Rig& R,
                       const std::vector<uint32_t>& tets, std::vector<Descent>& out)
{
    const PxTransform aToB = S.aToB();
    const float cull = S.cullScale();
    const uint32_t n = uint32_t(tets.size());
    std::vector<F4> tv(size_t(n) * 4u);
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t t = tets[i];
        for (uint32_t k = 0; k < 4u; ++k)
        {
            const F4& w = S.verts[S.inds[t * 4u + k]];
            // T4: no scale on the softbody side.
            const PxVec3 v = hShape2Vertex(aToB.transform(PxVec3(w.x, w.y, w.z)),
                                           S.scale, S.scaleRot);
            tv[size_t(i) * 4u + k] = F4{ v.x, v.y, v.z, (k == 0u) ? cull : 0.0f };
        }
    }
    DevBuf bTv = R.mk(dev, tv);
    DevBuf bSeed = R.mk(dev, std::vector<F4>(n, F4{ 0, 0, 0, 0 }));
    DevBuf bOut = R.mk(dev, std::vector<float>(size_t(n) * TET_TRACE_STRIDE, -4242.f));

    P_Coll p{};
    p.meshPtr = R.bBlob.addr;
    p.dirSeed = bSeed.addr;
    p.triVerts = bTv.addr;
    p.triOut = bOut.addr;
    p.numPoints = n;
    {
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTetrahedron", {},
                 sizeof(P_Coll));
        PXG_CHECK_MSG(k.isValid(), "the descent reference pipeline must build");
        if (!k.isValid()) return false;
        k.dispatch((n + 63u) / 64u, {}, p);
    }
    dev->flush();
    const auto o = downloadBuffer<float>(dev, bOut.h, size_t(n) * TET_TRACE_STRIDE);
    out.resize(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const float* e = &o[size_t(i) * TET_TRACE_STRIDE];
        out[i].point = F4{ e[22], e[23], e[24], 0.0f };
        out[i].dir = F4{ e[25], e[26], e[27], 0.0f };
        out[i].sep = e[28];
    }
    return true;
}

// ---------------------------------------------------------------------------
struct Contact
{
    uint32_t tet;
    F4 point, normalPen;
    uint32_t pairInd1;
};
struct Expect
{
    std::vector<Contact> contacts;
    uint32_t gathered = 0, culled = 0, gated = 0, filtered = 0, zeroDir = 0;
    uint32_t exactGate = 0, exactGateWouldWrite = 0;
    size_t rounds = 0;
};

static Expect predictPair(const Scene& S, const Rig& R,
                          const std::vector<std::vector<uint32_t>>& rounds,
                          const std::vector<uint32_t>& tets,
                          const std::vector<Descent>& desc, float rest)
{
    Expect E;
    E.rounds = rounds.size();
    const float cull = S.cullScale();
    const float wcd = S.worldContactDist();
    const PxTransform meshPose = S.meshPose();

    auto listed = [&](uint32_t tetId) {
        PxgRigidFilterPair q;
        q.index0 = R.rigidId.getInd();
        q.index1 = encodeSb(Scene::SB_ID, tetId);
        q.index2 = 0u;
        return std::binary_search(R.pairs.begin(), R.pairs.end(), q,
                                  [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
                                      if (a.index0 != b.index0) return a.index0 < b.index0;
                                      return a.index1 < b.index1;
                                  });
    };

    for (const std::vector<uint32_t>& round : rounds)
    {
        for (uint32_t tid = 0; tid < round.size(); ++tid)
        {
            const uint32_t t = round[tid];
            ++E.gathered;
            const size_t at = size_t(std::lower_bound(tets.begin(), tets.end(), t) - tets.begin());

            float sep = desc[at].sep;

            // The one point where `<` and `<=` on the cull gate can part company,
            // counted twice: sitting on the gate, and would an `<=` spelling carry
            // the tet all the way to a written contact.
            if (f2u(sep) == f2u(cull))
            {
                ++E.exactGate;
                PxVec3 d2 = hShape2Vertex(PxVec3(desc[at].dir.x, desc[at].dir.y, desc[at].dir.z),
                                          S.scale, S.scaleRot);
                float m2 = d2.dot(d2);
                float s2 = sep;
                if (m2 > 0.0f) { m2 = 1.0f / std::sqrt(m2); s2 = s2 * m2; }
                if (s2 < wcd) ++E.exactGateWouldWrite;
            }

            if (!(sep < cull)) { ++E.culled; continue; }

            PxVec3 dir = hShape2Vertex(PxVec3(desc[at].dir.x, desc[at].dir.y, desc[at].dir.z),
                                       S.scale, S.scaleRot);
            float m = dir.dot(dir);
            if (m > 0.0f)
            {
                m = 1.0f / std::sqrt(m);
                sep = sep * m;
                dir = dir * m;
            }
            else
            {
                ++E.zeroDir;
            }
            if (!(sep < wcd)) { ++E.gated; continue; }

            // T7: to world space here, BEFORE the filter.
            const PxVec3 posShape = hVertex2Shape(
                PxVec3(desc[at].point.x, desc[at].point.y, desc[at].point.z), S.scale, S.scaleRot);
            const PxVec3 worldP = meshPose.transform(posShape);   // ENGINE
            const PxVec3 worldN = meshPose.rotate(dir);           // ENGINE

            // T5: LAST, one query, on the TET.
            if (listed(t)) { ++E.filtered; continue; }

            Contact k;
            k.tet = t;
            k.point = F4{ worldP.x, worldP.y, worldP.z, rest };
            k.normalPen = F4{ worldN.x, worldN.y, worldN.z, sep };
            k.pairInd1 = encodeSb(Scene::SB_ID, t);
            E.contacts.push_back(k);
        }
    }
    return E;
}

// ---------------------------------------------------------------------------
static void runOneScene(PxgSlangComputeDevice* dev, const char* label, const Options& opt,
                        Scene& S)
{
    Rig R;
    if (!buildRig(dev, S, opt, R))
    {
        PXG_CHECK_MSG(false, "%s: the SDF textures must upload", label);
        R.destroy(dev);
        return;
    }

    std::vector<std::vector<uint32_t>> rounds;
    if (!refGather(dev, S, R, rounds)) { R.destroy(dev); return; }
    std::vector<uint32_t> tets;
    for (const auto& r : rounds) tets.insert(tets.end(), r.begin(), r.end());
    std::sort(tets.begin(), tets.end());
    tets.erase(std::unique(tets.begin(), tets.end()), tets.end());
    if (tets.empty())
    {
        PXG_CHECK_MSG(false, "%s: the gather produced nothing -- the rig proves nothing", label);
        R.destroy(dev);
        return;
    }
    std::vector<Descent> desc;
    if (!refDescent(dev, S, R, tets, desc)) { R.destroy(dev); return; }

    // Every contact manager sees the SAME softbody here, so the per-pair difference
    // is `restDistance[cmIdx]` alone -- which is exactly the index this kernel is
    // meant to get right.
    std::vector<Expect> exp(opt.numPairs);
    for (uint32_t p = 0; p < opt.numPairs; ++p)
        exp[p] = predictPair(S, R, rounds, tets, desc, S.restByPair[p]);

    if (opt.bridgeCtx)
    {
        // P9.97b: THE ENGINE'S HOST PACK, PxgNarrowphaseCore.cpp:4870-4903, with the
        // engine's own argument types and order -- not this file's SbSdfBlob.  The bridge
        // (PxgVulkanCudaContext.cpp, sized launchKernel) places each argument at
        // paramAlign(size), so the 80-byte writer goes to @96, not to the @88 its natural
        // alignment of 8 would give.  A shader whose push block disagrees reads the wrong
        // bytes here and nowhere else in this file.
        const PxReal toleranceLength = R.blob.tolerenceLength;
        const PxgContactManagerInput* cmInputs =
            reinterpret_cast<const PxgContactManagerInput*>(size_t(R.bCin.addr));
        const PxsCachedTransform* transformCached =
            reinterpret_cast<const PxsCachedTransform*>(size_t(R.bXf.addr));
        const PxBounds3* boundsd = reinterpret_cast<const PxBounds3*>(size_t(R.bBounds.addr));
        const PxReal* contactDistanced = reinterpret_cast<const PxReal*>(size_t(R.bCd.addr));
        const PxReal* restDistanced = reinterpret_cast<const PxReal*>(size_t(R.bRest.addr));
        PxgShape* gpuShapes = reinterpret_cast<PxgShape*>(size_t(R.bShapes.addr));
        CUdeviceptr softBodiesd = CUdeviceptr(R.bDesc.addr);
        CUdeviceptr shapeToRigidRemapTabled = CUdeviceptr(R.bRemap.addr);
        CUdeviceptr filterPairs = CUdeviceptr(R.bPairs.addr);
        const PxU32 nbFilterPairs = PxU32(R.pairs.size());
        HostFemWriter writer;
        writer.outPoint = reinterpret_cast<float4*>(size_t(R.bPoint.addr));
        writer.outNormalPen = reinterpret_cast<float4*>(size_t(R.bNormalPen.addr));
        writer.outBarycentric = reinterpret_cast<float4*>(size_t(R.bBary.addr));
        writer.outContactInfo = reinterpret_cast<PxgFemOtherContactInfo*>(size_t(R.bInfo.addr));
        writer.totalContactCount = reinterpret_cast<PxU32*>(size_t(R.bCount.addr));
        writer.contactByX = reinterpret_cast<PxU64*>(size_t(R.bByX.addr));
        writer.tempContactByX = reinterpret_cast<PxU32*>(size_t(R.bTempByX.addr));
        writer.contactIndexSortedByX = reinterpret_cast<PxU32*>(size_t(R.bIdxByX.addr));
        writer.contactSortedByX = reinterpret_cast<PxU32*>(size_t(R.bSortedByX.addr));
        writer.maxNumContacts = opt.capacity;

        PxCudaKernelParam kernelParams[] =
        {
            PX_CUDA_KERNEL_PARAM(toleranceLength),
            PX_CUDA_KERNEL_PARAM(cmInputs),
            PX_CUDA_KERNEL_PARAM(transformCached),
            PX_CUDA_KERNEL_PARAM(boundsd),
            PX_CUDA_KERNEL_PARAM(contactDistanced),
            PX_CUDA_KERNEL_PARAM(restDistanced),
            PX_CUDA_KERNEL_PARAM(gpuShapes),
            PX_CUDA_KERNEL_PARAM(softBodiesd),
            PX_CUDA_KERNEL_PARAM(shapeToRigidRemapTabled),

            PX_CUDA_KERNEL_PARAM(filterPairs),
            PX_CUDA_KERNEL_PARAM(nbFilterPairs),

            PX_CUDA_KERNEL_PARAM(writer)
        };
        const PxCUresult lr = opt.bridgeCtx->launchKernel(opt.bridgeFn, opt.numPairs, 1, 1, 1024, 1, 1,
                                                          0, nullptr, kernelParams, sizeof(kernelParams),
                                                          nullptr, __FILE__, __LINE__);
        PXG_CHECK_MSG(lr == 0, "%s: the bridge must accept the engine's 12-argument host pack "
                               "(its blob is gated against the ABI row's push size)", label);
        if (lr != 0) { R.destroy(dev); return; }
        PXG_CHECK_MSG(opt.bridgeCtx->streamSynchronize(nullptr) == 0, "%s: streamSynchronize", label);
    }
    else
    {
        Kernel k(Kernel::Bindless{}, "softBodySdfMidphaseBda.spv",
                 "sb_sdfMeshMidphaseGeneratePairsLaunch", {}, sizeof(SbSdfBlob));
        PXG_CHECK_MSG(k.isValid(), "%s: the kernel pipeline must build", label);
        if (!k.isValid()) { R.destroy(dev); return; }
        k.dispatch(opt.numPairs, {}, R.blob);
    }
    dev->flush();

    const auto point = downloadBuffer<F4>(dev, R.bPoint.h, Scene::CAP);
    const auto npen = downloadBuffer<F4>(dev, R.bNormalPen.h, Scene::CAP);
    const auto bary = downloadBuffer<F4>(dev, R.bBary.h, Scene::CAP);
    const auto info = downloadBuffer<uint32_t>(dev, R.bInfo.h, size_t(Scene::CAP) * 4u);
    const auto byX = downloadBuffer<uint64_t>(dev, R.bByX.h, Scene::CAP);
    const auto tempByX = downloadBuffer<uint32_t>(dev, R.bTempByX.h, Scene::CAP);
    const auto idxByX = downloadBuffer<uint32_t>(dev, R.bIdxByX.h, Scene::CAP);
    const uint32_t total = downloadBuffer<uint32_t>(dev, R.bCount.h, 1)[0];

    uint32_t reserved = 0;
    for (uint32_t p = 0; p < opt.numPairs; ++p) reserved += uint32_t(exp[p].contacts.size());
    PXG_CHECK_MSG(total == opt.seedCount + reserved,
                  "%s: totalContactCount is %u, expected %u (seed %u + %u reserved)",
                  label, total, opt.seedCount + reserved, opt.seedCount, reserved);

    PXG_CHECK_MSG(!exp[0].contacts.empty(), "%s: no contact at all", label);
    PXG_CHECK_MSG(exp[0].culled > 0u && exp[0].gated > 0u,
                  "%s: %u tets failed the cull gate and %u the contact-distance gate (both must "
                  "be > 0, or the two gates are one)", label, exp[0].culled, exp[0].gated);
    PXG_CHECK_MSG(exp[0].filtered > 0u,
                  "%s: the filter removed nothing, so T5 is unobservable", label);
    if (opt.exactGate)
    {
        // THE PROOF, MADE EXECUTABLE -- and here it is STRONGER than the cloth
        // sibling's, because this kernel takes the absolute value.
        //
        // At the boundary  sep == cullScale == worldContactDist / min(|scale|).
        // `dir` comes back NORMALISED, so |dir| == 1; shape2Vertex divides each
        // rotated component by its scale and a rotation preserves length, so
        // |shape2Vertex(dir)| <= 1 / min(|scale|) and therefore m >= min(|scale|).
        // Then sep * m >= worldContactDist, and the SECOND gate is strict -- so an
        // `<=` here would admit exactly the tets the next line rejects.
        //
        // The cloth version of this argument needed "positive scale" as a
        // hypothesis, because its cullScale divides by the SIGNED minimum.  This
        // one does not: the bound |t_i / scale_i| <= |t_i| / min(|scale|) holds
        // whatever the signs are, which is T2 paying for itself.
        PXG_CHECK_MSG(exp[0].exactGate > 0u,
                      "%s: no tet sits exactly on the cull gate -- the search that was supposed "
                      "to manufacture one did not survive the rebuild", label);
        PXG_CHECK_MSG(exp[0].exactGateWouldWrite == 0u,
                      "%s: %u of %u tets on the cull gate would reach a write under an `<=` "
                      "spelling -- the min(|scale|) argument is WRONG and the two gates are "
                      "observably different; this needs a mutant, not a proof",
                      label, exp[0].exactGateWouldWrite, exp[0].exactGate);
    }
    else if (!opt.negScale)
    {
        PXG_CHECK_MSG(exp[0].rounds >= 2,
                      "%s: the gather took %zu round(s) -- the resume path is untested",
                      label, exp[0].rounds);
    }
    else
    {
        // T2's own contract, and it is about DISCRIMINATION.  The kernel's cullScale
        // takes the absolute value; the gather it calls does NOT.  With a negative
        // component that is not the smallest in magnitude the two spellings differ in
        // sign, so this run proves the kernel used the abs one -- by gathering with
        // the SIGNED tolerance as well and requiring the two sets to be far apart.
        // Without that, a port that dropped the abs would still pass everything
        // above, because the fixture would have derived the same wrong tolerance.
        const float cullAbs = S.cullScale();
        const float minSigned = PxMin(S.scale.x, PxMin(S.scale.y, S.scale.z));
        const float cullSigned = S.worldContactDist() / minSigned;
        PXG_CHECK_MSG(cullAbs > 0.0f && cullSigned < 0.0f,
                      "%s: the kernel's cullScale %.9g and the signed spelling %.9g must differ "
                      "in SIGN", label, cullAbs, cullSigned);
        std::vector<std::vector<uint32_t>> roundsSigned;
        if (refGather(dev, S, R, roundsSigned, cullSigned))
        {
            size_t nSigned = 0;
            for (const auto& r : roundsSigned) nSigned += r.size();
            const size_t diff = (nSigned > exp[0].gathered) ? (nSigned - exp[0].gathered)
                                                            : (exp[0].gathered - nSigned);
            PXG_CHECK_MSG(diff > 200u,
                          "%s: the signed spelling gathers %zu tets against the engine's %u -- "
                          "too close for the absolute value to be observable", label, nSigned,
                          exp[0].gathered);
            std::printf("    [ note ] %s: cullScale %.9g (abs) against %.9g (signed); %u tets "
                        "gathered against %zu\n", label, cullAbs, cullSigned, exp[0].gathered,
                        nSigned);
        }
    }

    size_t bad = 0;
    auto checkAt = [&](const Contact& e, uint32_t at) {
        if (at >= opt.capacity) return;
        if (!bitEq(point[at].x, e.point.x) || !bitEq(point[at].y, e.point.y) ||
            !bitEq(point[at].z, e.point.z) || !bitEq(point[at].w, e.point.w))
        {
            if (bad < 4)
                std::printf("    %s slot %u point (%.9g,%.9g,%.9g,%.9g) expected "
                            "(%.9g,%.9g,%.9g,%.9g)\n", label, at, point[at].x, point[at].y,
                            point[at].z, point[at].w, e.point.x, e.point.y, e.point.z, e.point.w);
            ++bad;
        }
        if (!bitEq(npen[at].x, e.normalPen.x) || !bitEq(npen[at].y, e.normalPen.y) ||
            !bitEq(npen[at].z, e.normalPen.z) || !bitEq(npen[at].w, e.normalPen.w))
        {
            if (bad < 4)
                std::printf("    %s slot %u normalPen (%.9g,%.9g,%.9g,%.9g) expected "
                            "(%.9g,%.9g,%.9g,%.9g)\n", label, at, npen[at].x, npen[at].y,
                            npen[at].z, npen[at].w, e.normalPen.x, e.normalPen.y,
                            e.normalPen.z, e.normalPen.w);
            ++bad;
        }
        const uint64_t rid = R.rigidId.getInd();
        if (info[size_t(at) * 4u + 0] != uint32_t(rid & 0xffffffffu)) ++bad;
        if (info[size_t(at) * 4u + 1] != uint32_t(rid >> 32)) ++bad;
        if (info[size_t(at) * 4u + 2] != e.pairInd1) ++bad;
        // T6: the fourth dword is NOT written, and the barycentric array is not
        // touched at all.  Both must still be exactly as uploaded.
        if (info[size_t(at) * 4u + 3] != POISON_U32) ++bad;
        if (!bitEq(bary[at].x, -7.f) || !bitEq(bary[at].w, -7.f)) ++bad;
        if (byX[at] != rid) ++bad;
        if (tempByX[at] != uint32_t(rid & 0xffffffffu)) ++bad;
        if (idxByX[at] != at) ++bad;
    };

    if (opt.numPairs == 1u)
    {
        for (size_t i = 0; i < exp[0].contacts.size(); ++i)
            checkAt(exp[0].contacts[i], uint32_t(opt.seedCount + i));
    }
    else
    {
        // Two blocks racing on one counter: the P9.11 rule applies, so the SET is
        // asserted and the interleaving is not.  Both pairs carry the same tets, so
        // the discriminator is the REST DISTANCE in point.w -- which is indexed by
        // cmIdx and is the thing this configuration exists to check.
        std::vector<const Contact*> want;
        for (uint32_t p = 0; p < opt.numPairs; ++p)
            for (const Contact& k : exp[p].contacts) want.push_back(&k);
        std::vector<uint8_t> used(want.size(), 0u);
        for (uint32_t at = opt.seedCount; at < PxMin(total, opt.capacity); ++at)
        {
            size_t hit = want.size();
            for (size_t j = 0; j < want.size(); ++j)
                if (!used[j] && want[j]->pairInd1 == info[size_t(at) * 4u + 2] &&
                    bitEq(want[j]->point.w, point[at].w)) { hit = j; break; }
            if (hit == want.size()) { ++bad; continue; }
            used[hit] = 1u;
            checkAt(*want[hit], at);
        }
    }
    PXG_CHECK_MSG(bad == 0, "%s: %zu contact-stream comparisons failed", label, bad);

    size_t spill = 0;
    for (uint32_t at = PxMin(total, opt.capacity); at < Scene::CAP; ++at)
        if (idxByX[at] != POISON_U32 || byX[at] != ~0ull || !bitEq(point[at].x, -7.f)) ++spill;
    PXG_CHECK_MSG(spill == 0, "%s: %zu slots beyond the reservation were written", label, spill);

    std::printf("    [ note ] %s: %zu round(s), %u gathered, %u culled (%u exactly on the gate, "
                "%u of those writable), %u gated, %u filtered, %u zero-dir -> %zu contacts; "
                "total %u (cap %u)\n",
                label, exp[0].rounds, exp[0].gathered, exp[0].culled, exp[0].exactGate,
                exp[0].exactGateWouldWrite, exp[0].gated, exp[0].filtered, exp[0].zeroDir,
                exp[0].contacts.size(), total, opt.capacity);
    R.destroy(dev);
}

static void runOne(PxgSlangComputeDevice* dev, const char* label, const Options& opt)
{
    Scene S;
    buildScene(S, opt.negScale);
    runOneScene(dev, label, opt, S);
}

static bool beginShim(PxgSlangComputeDevice* dev, PxgCudaShim*& shim)
{
    static const PxgKernelDef manifest[] = {
        { "softBodySdfMidphaseBda.spv", "sb_sdfMeshMidphaseGeneratePairsLaunch", 0u, 0u } };
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

PXG_TEST(softBodySdfMidphase)
{
    // P9.97b: the literal tie between this file's blob and the launch-ABI row.  The
    // static_asserts above pin SbSdfBlob to the shader's layout; this pins it to what
    // the ENGINE pushes.  One line, and it would have caught P9.35c's @88 writer.
    {
        const physx::gpuslang::PxgKernelAbiRow* row =
            physx::gpuslang::pxgFindKernelAbi("sb_sdfMeshMidphaseGeneratePairsLaunch");
        PXG_CHECK_MSG(row && sizeof(SbSdfBlob) == physx::gpuslang::pxgKernelAbiPushSize(*row),
                      "SbSdfBlob (%zu bytes) is the blob the bridge builds from the row (%u)",
                      sizeof(SbSdfBlob), row ? physx::gpuslang::pxgKernelAbiPushSize(*row) : 0u);
    }
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    PxgCudaShim* shim = nullptr;
    beginShim(dev, shim);

    { Options o; o.sbFirst = true;  runOne(dev, "sb-first", o); }
    { Options o; o.sbFirst = false; runOne(dev, "mesh-first", o); }
    { Options o; o.seedCount = 37u; runOne(dev, "seeded-counter", o); }
    { Options o; o.capacity = 24u;  runOne(dev, "capped", o); }
    // TWO contact managers: cmIdx reaches restDistances, and nothing else differs
    // between them, so point.w is the whole discriminator.
    { Options o; o.numPairs = 2u;   runOne(dev, "two-pairs", o); }

    endShim(shim);
}

// ---------------------------------------------------------------------------
// The exact-gate SEARCH, the same device P9.34c established.
//
// doTetrahedronSDFCollision's return is INDEPENDENT of the tolerance it is given
// except that the tolerance decides whether the value is returned at all -- its
// last line is PxSdfSampleWithGrad, and nothing inside the iteration reads the
// tolerance.  So a tet's separation is a constant V, and the job is to find a
// softbody contact distance whose derived cullScale IS that constant, bit for
// bit.  cullScale = fl(fl(cdTri + c) / min(|scale|)) is monotone in c, so the
// answer is within an ULP or two of V * min(|scale|) - cdTri.
//
// The equality is CONFIRMED by re-running the references, not assumed: moving c
// moves the tolerance, which moves both the gathered set and which tets clear the
// descent's own gate.
// ---------------------------------------------------------------------------
static bool probeSb(PxgSlangComputeDevice* dev, Scene& S, std::vector<uint32_t>& tets,
                    std::vector<Descent>& desc)
{
    Options o;
    Rig R;
    bool ok = false;
    if (buildRig(dev, S, o, R))
    {
        std::vector<std::vector<uint32_t>> rounds;
        if (refGather(dev, S, R, rounds))
        {
            tets.clear();
            for (const auto& r : rounds) tets.insert(tets.end(), r.begin(), r.end());
            std::sort(tets.begin(), tets.end());
            tets.erase(std::unique(tets.begin(), tets.end()), tets.end());
            ok = !tets.empty() && refDescent(dev, S, R, tets, desc);
        }
    }
    R.destroy(dev);
    return ok;
}

static bool searchExactGate(PxgSlangComputeDevice* dev, Scene& S, float& foundC, uint32_t& foundTet)
{
    const float minAbs = PxMin(PxAbs(S.scale.x), PxMin(PxAbs(S.scale.y), PxAbs(S.scale.z)));
    const HostSdfBox box = hostSdfBox(S.sdf);

    std::vector<uint32_t> tets;
    std::vector<Descent> desc;
    if (!probeSb(dev, S, tets, desc)) return false;

    // dir . diff must be strictly positive, or `dist < tolerance` fails once the
    // tolerance is pushed down onto V and the tet drops out of the descent.
    struct Cand { uint32_t tet; float V; float dot; };
    std::vector<Cand> cand;
    for (size_t i = 0; i < tets.size(); ++i)
    {
        const float V = desc[i].sep;
        if (!std::isfinite(V) || V > 1e30f) continue;
        const PxVec3 p(desc[i].point.x, desc[i].point.y, desc[i].point.z);
        const PxVec3 cl(PxMax(box.lower.x, PxMin(box.higher.x, p.x)),
                        PxMax(box.lower.y, PxMin(box.higher.y, p.y)),
                        PxMax(box.lower.z, PxMin(box.higher.z, p.z)));
        const PxVec3 d(desc[i].dir.x, desc[i].dir.y, desc[i].dir.z);
        const float dt = d.dot(p - cl);
        if (dt <= 0.0f) continue;
        cand.push_back(Cand{ tets[i], V, dt });
    }
    std::sort(cand.begin(), cand.end(),
              [](const Cand& a, const Cand& b) { return a.dot > b.dot; });

    const float saved = S.contactDistSb;
    uint32_t tried = 0;
    for (const Cand& k : cand)
    {
        if (tried >= 8u) break;
        const float ideal = k.V * minAbs - S.contactDistTri;
        float hit = 0.0f;
        bool haveHit = false;
        for (int step = -96; step <= 96 && !haveHit; ++step)
        {
            uint32_t u = f2u(ideal);
            u = uint32_t(int32_t(u) + step);
            float c; std::memcpy(&c, &u, 4);
            if (!(c > 0.0f)) continue;
            if (f2u((S.contactDistTri + c) / minAbs) == f2u(k.V)) { hit = c; haveHit = true; }
        }
        if (!haveHit) continue;

        ++tried;
        S.contactDistSb = hit;
        std::vector<uint32_t> t2;
        std::vector<Descent> d2;
        if (probeSb(dev, S, t2, d2))
        {
            const float cull = S.cullScale();
            for (size_t i = 0; i < t2.size(); ++i)
                if (t2[i] == k.tet && f2u(d2[i].sep) == f2u(cull))
                {
                    foundC = hit; foundTet = k.tet;
                    return true;
                }
        }
        S.contactDistSb = saved;
    }
    S.contactDistSb = saved;
    return false;
}

PXG_TEST(softBodySdfMidphaseExactGate)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    PxgCudaShim* shim = nullptr;
    beginShim(dev, shim);

    Scene S;
    buildScene(S, false);
    float c = 0.0f;
    uint32_t tet = 0u;
    if (!searchExactGate(dev, S, c, tet))
    {
        PXG_CHECK_MSG(false, "no contact distance puts a gathered tet's separation exactly on "
                             "the cull gate -- the `<` / `<=` boundary is untested");
    }
    else
    {
        std::printf("    [ note ] exact-gate: tet %u, softbody contact distance %.9g -> cullScale "
                    "%.9g, bit-identical to its separation\n", tet, c, S.cullScale());
        Options o; o.exactGate = true;
        runOneScene(dev, "exact-gate", o, S);
    }

    endShim(shim);
}

PXG_TEST(softBodySdfMidphaseNegScale)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    PxgCudaShim* shim = nullptr;
    beginShim(dev, shim);

    // T2 lives here.  With scale.y = -2.9 the kernel's cullScale (which takes the
    // absolute value) and the gather's radius divide (which does not) disagree in
    // sign INSIDE ONE RUN -- so this configuration checks both spellings at once,
    // and a port that makes them agree either way fails.
    {
        Scene probe;
        buildScene(probe, true);
        const float minSigned = PxMin(probe.scale.x, PxMin(probe.scale.y, probe.scale.z));
        const float minAbs = PxMin(PxAbs(probe.scale.x),
                                   PxMin(PxAbs(probe.scale.y), PxAbs(probe.scale.z)));
        PXG_CHECK_MSG(minSigned < 0.0f && minAbs > 0.0f,
                      "the negative-scale run must make min(x,y,z) = %.9g and min of the "
                      "magnitudes = %.9g differ in SIGN", minSigned, minAbs);
        PXG_CHECK_MSG(probe.cullScale() > 0.0f,
                      "the kernel's cullScale takes the absolute value, so it must be POSITIVE "
                      "here (%.9g) even though the gather's own divide is not", probe.cullScale());
    }
    { Options o; o.negScale = true; runOne(dev, "neg-scale/sb-first", o); }
    { Options o; o.negScale = true; o.sbFirst = false;
      runOne(dev, "neg-scale/mesh-first", o); }

    endShim(shim);
}

// ---------------------------------------------------------------------------
// P9.97b -- THE SAME KERNEL THROUGH THE ENGINE'S LAUNCH PATH.
//
// Every fixture above dispatches SbSdfBlob, this file's own transcription of the
// shader's push block -- so a block laid out differently from what the engine pushes
// passes all of them by construction: the fixture and the shader agree with EACH OTHER.
// That is exactly what happened in P9.35c.  The row is {4, 8 x 9, 4, 80} and the bridge
// places each argument at paramAlign(size) (PxgVulkanCudaContext.cpp, sized
// launchKernel), so the 80-byte writer starts at @96 and the blob is 176 bytes; the
// shader and SbSdfBlob put the writer at its natural alignment, @88, in a 168-byte
// block.  On the engine path the shader then read `w_outPoint` from the bridge's
// zero padding and every later writer field one slot off -- the femClothContactGenBda
// "segfault at 0" class (femClothContactGenBda.slang:283-289), latent only because no
// scene reached the eSoftbodySdfTrimesh bucket.
//
// This test closes the gap the way the engine closes it: the manifest's push size is
// the ROW's (PhysXGpuVulkan.cpp pushConstantSizeFor), the context is the bridge
// (PxgCreateVulkanCudaContext), and the arguments are the engine's host pack with its
// own types (runOneScene, opt.bridgeCtx).  The expectation is the same two-reference
// oracle, compared bitwise.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySdfMidphaseBridgeSized)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    const char* kName = "sb_sdfMeshMidphaseGeneratePairsLaunch";
    const physx::gpuslang::PxgKernelAbiRow* row = physx::gpuslang::pxgFindKernelAbi(kName);
    PXG_CHECK_MSG(row != nullptr, "the kernel has a launch-ABI row");
    if (!row)
        return;
    const uint32_t push = physx::gpuslang::pxgKernelAbiPushSize(*row);
    // The row itself, literally: the host pack PxgNarrowphaseCore.cpp:4888-4903 pushes.
    static const uint8_t kHostPack[12] = { 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 80 };
    bool rowIsHostPack = row->argCount == 12u;
    for (uint32_t i = 0; rowIsHostPack && i < 12u; ++i)
        rowIsHostPack = row->argSizes[i] == kHostPack[i];
    PXG_CHECK_MSG(rowIsHostPack, "the row is the host pack {4, 8 x 9, 4, 80}");
    PXG_CHECK_MSG(push == 176u, "the row's push size is %u; the writer at paramAlign(80) = 16 "
                                "makes it 176", push);
    // THE LITERAL TIE: this file's own blob must BE the bridge's blob, size first.
    PXG_CHECK_MSG(sizeof(SbSdfBlob) == push,
                  "SbSdfBlob is %zu bytes but the bridge pushes %u -- the fixture's blob is not the "
                  "engine's", sizeof(SbSdfBlob), push);

    const PxgKernelDef manifest[] = {
        { "softBodySdfMidphaseBda.spv", kName, push, 0u, kName } };
    static auto loader = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim* shim = new PxgCudaShim(dev, manifest, 1, +loader);
    physx::gpuslang::pxgSetVulkanDriverShim(shim);
    physx::PxCudaContext* ctx = physx::gpuslang::PxgCreateVulkanCudaContext(shim, manifest, 1);
    CUfunction fn = nullptr;
    const bool mapped = ctx && ctx->moduleGetFunction(&fn, nullptr, kName) == 0 && fn != nullptr;
    PXG_CHECK_MSG(mapped, "the bridge resolves %s", kName);
    if (mapped)
    {
        { Options o; o.bridgeCtx = ctx; o.bridgeFn = fn;                  runOne(dev, "bridge/sb-first", o); }
        { Options o; o.bridgeCtx = ctx; o.bridgeFn = fn; o.sbFirst = false; runOne(dev, "bridge/mesh-first", o); }
        { Options o; o.bridgeCtx = ctx; o.bridgeFn = fn; o.seedCount = 37u; runOne(dev, "bridge/seeded-counter", o); }
        // capacity is the writer's LAST field (maxNumContacts): a block that misplaces it
        // reads a pointer half as the cap, and only a capped run can tell.
        { Options o; o.bridgeCtx = ctx; o.bridgeFn = fn; o.capacity = 24u;  runOne(dev, "bridge/capped", o); }
        { Options o; o.bridgeCtx = ctx; o.bridgeFn = fn; o.numPairs = 2u;   runOne(dev, "bridge/two-pairs", o); }
    }
    if (ctx)
        ctx->release();
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
    delete shim;
}

}  // namespace
