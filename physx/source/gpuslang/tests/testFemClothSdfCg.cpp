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

// testFemClothSdfCg.cpp -- P9.34c, the fixture for cloth_sdfMeshContactGenLaunch.
//
// ---------------------------------------------------------------------------
// THE ARRANGEMENT: A WRAPPER CHECKED AGAINST ITS OWN PARTS
// ---------------------------------------------------------------------------
// Like psSdfMidphase, this kernel is COMPOSITION, and both halves that carry the
// work are already swept modules with their own bit-exact fixtures:
//
//   pxgSlangSdfFind       P9.34b  which cloth triangles are even considered
//   pxgSlangSdfCollision  P9.34a  where the descent lands on each of them
//
// So the expectation is built from TWO INDEPENDENT DISPATCHES of those modules'
// own entry points:
//
//   npSdfFind.spv:sdfFindTriangleRounds -> the gathered index list, per round
//   npSdfCollision.spv:sdfCollTriangle  -> point / dir / separation per triangle
//
// and the host predicts, from those two plus ENGINE transforms, every byte the
// kernel should have written.
//
// AND THE GLUE IS DISCRIMINATED, not merely exercised.  The reference dispatches
// are fed the cull tolerance, the inverse pose and the two scale frames that the
// HOST computed.  If the kernel derives any of them differently -- the wrong
// scale component, an abs() the CUDA does not have, the cloth scaled when it
// must not be, the quaternion not conjugated -- it gathers a different set or
// descends from a different triangle, and the comparison fails.
//
// WHAT THE HOST DOES ITSELF, and why that is safe: everything outside the two
// device-only pieces is either an ENGINE call or four lines of arithmetic over
// one.  vertex2Shape / shape2Vertex are rotate, three multiplies or divides,
// rotateInv -- and PxQuat::rotate / rotateInv are the engine's own and are
// CALLED, not transcribed.  PxTransform::transform / rotate likewise.  Only
// computeBarycentric (femClothPrimitives.cu:114) is a real transcription, and it
// is nine dot products and two divides with no branch.
//
//   femClothSdfContactGen        -- the solid pass: both shape-slot orders, the
//                                   filter, the resume loop, a pre-seeded
//                                   counter, the capacity cap, and two contact
//                                   managers in one launch.
//   femClothSdfContactGenNegScale -- S3: the cull scale divides by the SMALLEST
//                                   scale component and NOT by its magnitude, so
//                                   a negative component flips the tolerance's
//                                   sign.  The particle sibling takes abs() from
//                                   the same expression shape; this one does not.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgFEMCloth.h"
#include "PxgFEMCore.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgSimulationCoreDesc.h"
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

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

// femClothSdfMeshCgBda.slang's ClothSdfCgParams, byte for byte.
struct ClothSdfBlob
{
    uint32_t numWorkItems;             // @0
    uint32_t pad4;                     // @4
    uint64_t cmInputs;                 // @8
    uint64_t transformCache;           // @16
    uint64_t contactDistances;         // @24
    uint64_t restDistances;            // @32
    uint64_t gpuShapes;                // @40
    uint64_t femClothes;               // @48
    uint64_t shapeToRigidRemap;        // @56
    uint64_t filterPairs;              // @64
    uint32_t nbFilterPairs;            // @72
    uint32_t pad76;                    // @76
    uint64_t w_outPoint;               // @80
    uint64_t w_outNormalPen;           // @88
    uint64_t w_outBarycentric;         // @96
    uint64_t w_outContactInfo;         // @104
    uint64_t w_totalContactCount;      // @112
    uint64_t w_contactByX;             // @120
    uint64_t w_tempContactByX;         // @128
    uint64_t w_contactIndexSortedByX;  // @136
    uint64_t w_contactSortedByX;       // @144
    uint32_t w_maxNumContacts;         // @152
    uint32_t pad156;                   // -> 160
};
static_assert(sizeof(ClothSdfBlob) == 160, "engine ABI blob size");
static_assert(offsetof(ClothSdfBlob, w_outPoint) == 80, "the writer lands at 80");
static_assert(sizeof(PxgFEMContactWriter) == 80, "the by-value writer tail is 80 bytes");

// npSdfFind.slang's SdfFindParams (P9.34b extended it to 176 with explicit padding).
struct P_Find
{
    uint64_t meshPtr, positions, echoOut, roundsOut, bufOut, numRoundsOut;
    float aToBq[4];
    float scaleRot[4];
    float aToBp[3];  float tolerance;
    float scale[3];  uint32_t numVertices;
    uint32_t maxRounds, f0, f1, f2;
    uint64_t meshInd, fpad;
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
static const uint32_t TRI_TRACE_STRIDE = 176u;   // sdfCollTriangle's per-entry stride
static const uint32_t POISON_U32 = 0xEEEEEEEEu;

static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }

// cudaNpCommon.h:270 / :284 -- rotate, three multiplies or DIVIDES, rotateInv.
// PxQuat::rotate and rotateInv are the ENGINE's and are called.
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

// femClothPrimitives.cu:114.  The bool result is unused at this call site, and the
// eps window that produces it is therefore not transcribed.
static void hComputeBarycentric(const PxVec3& a, const PxVec3& b, const PxVec3& c,
                                const PxVec3& p, F4& bary)
{
    const PxVec3 v0 = b - a, v1 = c - a, v2 = p - a;
    const float d00 = v0.dot(v0);
    const float d01 = v0.dot(v1);
    const float d11 = v1.dot(v1);
    const float d20 = v2.dot(v0);
    const float d21 = v2.dot(v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    bary = F4{ 1.0f - v - w, v, w, 0.0f };
}

static const uint32_t PXG_BITSHIFT_ELEMENT_ID_H = 20u;
static uint32_t encodeCloth(uint32_t clothId, uint32_t elem)
{
    return (clothId << PXG_BITSHIFT_ELEMENT_ID_H) | elem;
}

// ---------------------------------------------------------------------------
// The scene: one SDF trimesh, TWO cloths, two contact managers.
// ---------------------------------------------------------------------------
struct ClothDef
{
    uint32_t nbTris = 0;
    uint32_t clothId = 0;
    uint32_t shapeSlot = 0;
    uint32_t cacheSlot = 0;
    float contactDist = 0.0f;
    PxVec3 posW; PxQuat rotW;
    std::vector<F4> verts;
    std::vector<uint32_t> inds;              // uint4 per triangle
    std::vector<uint32_t> filteredTris;      // triangle ids whose THREE verts are filtered
    std::vector<uint32_t> partialTris;       // triangle ids with only two verts filtered
};

struct Scene
{
    static const uint32_t TRI_SHAPE = 3u, NB_SHAPES = 7u;
    static const uint32_t TRI_CACHE = 5u, NB_CACHE = 9u;
    static const uint32_t NB_CLOTHES = 7u;
    static const uint32_t CAP = 8192u;

    SdfScene sdf;
    BlobLayout blob;
    PxVec3 meshP; PxQuat meshQ;
    PxVec3 scale; PxQuat scaleRot;
    float contactDistTri = 0.0f;
    uint32_t materialIndex = 0u;
    std::vector<float> restByPair;
    ClothDef cloth[2];

    PxTransform meshPose() const { return PxTransform(meshP, meshQ); }
};

// The engine's cullScale, S3: the SMALLEST component, sign included.
static float cullScaleOf(const Scene& S, const ClothDef& c)
{
    const float cd = S.contactDistTri + c.contactDist;
    return cd / PxMin(S.scale.x, PxMin(S.scale.y, S.scale.z));
}
static float contactDistOf(const Scene& S, const ClothDef& c)
{
    return S.contactDistTri + c.contactDist;
}
// PxTransform::transformInv(other) == getInverse() * other.
static PxTransform aToBOf(const Scene& S, const ClothDef& c)
{
    return S.meshPose().transformInv(PxTransform(c.posW, c.rotW));
}

static void buildCloth(const Scene& S, ClothDef& c, uint32_t seed, uint32_t tailFar,
                       float sizeScale)
{
    const HostSdfBox box = hostSdfBox(S.sdf);
    const PxVec3 ext = box.higher - box.lower;
    const PxTransform bToA = aToBOf(S, c).getInverse();

    c.verts.resize(size_t(c.nbTris) * 3u);
    c.inds.resize(size_t(c.nbTris) * 4u);
    std::mt19937 rng(seed);
    auto u01 = [&]() { return float(rng() % 1000000u) / 1000000.f; };
    for (uint32_t t = 0; t < c.nbTris; ++t)
    {
        // The triangles vary in SIZE as well as position: the gather's cull test
        // divides the bounding-sphere radius by the smallest scale component and
        // adds it to the tolerance, and the descent's step schedule is built from
        // the squared edge spread -- a rig of uniform triangles cannot tell either
        // term from a constant.
        const float sz = sizeScale * ((t % 7u == 0u) ? 0.55f * ext.x
                                                     : 0.02f * ext.x * (1.0f + float(t % 5)));
        PxVec3 ctr;
        if (t < c.nbTris - tailFar)
            ctr = PxVec3(box.lower.x - 0.35f * ext.x + 1.7f * ext.x * u01(),
                         box.lower.y - 0.35f * ext.y + 1.7f * ext.y * u01(),
                         box.lower.z - 0.35f * ext.z + 1.7f * ext.z * u01());
        else
            ctr = PxVec3(box.lower.x - (8.0f + 3.0f * u01()) * ext.x,
                         box.lower.y - (8.0f + 3.0f * u01()) * ext.y,
                         box.lower.z - (8.0f + 3.0f * u01()) * ext.z);
        const PxVec3 corner[3] = { ctr + PxVec3(sz, 0.0f, 0.0f),
                                   ctr + PxVec3(-0.3f * sz, 0.9f * sz, 0.2f * sz),
                                   ctr + PxVec3(0.1f * sz, -0.4f * sz, -0.8f * sz) };
        for (uint32_t k = 0; k < 3u; ++k)
        {
            // Laid out in the SDF's vertex space and mapped back through the whole
            // inverse chain, so the forward chain lands them where they are wanted.
            // The cloth's own scale is the IDENTITY on both ends (S4), which is what
            // makes this the inverse of what the kernel will do.
            const PxVec3 world = bToA.transform(hShape2Vertex(corner[k], S.scale, S.scaleRot));
            c.verts[t * 3u + k] = F4{ world.x, world.y, world.z, 0.5f + 0.001f * float(k) };
        }
        // The vertex indices of triangle t are NOT 3t, 3t+1, 3t+2 in order: the
        // three are rotated by t, so a port that reads .y where .x belongs picks a
        // different corner and the descent starts elsewhere.
        const uint32_t r = t % 3u;
        c.inds[t * 4u + 0] = t * 3u + r;
        c.inds[t * 4u + 1] = t * 3u + (r + 1u) % 3u;
        c.inds[t * 4u + 2] = t * 3u + (r + 2u) % 3u;
        c.inds[t * 4u + 3] = 0xCDCDCDCDu;    // .w is never read; poison says so
    }
}

static void buildScene(Scene& S, bool negScale)
{
    buildSparse(S.sdf);
    S.blob = layOutBlob(S.sdf, 16u, 21u, 3u, 1u, 0u);

    // Nothing identity, nothing axis-aligned.
    S.meshQ = PxQuat(0.53f, PxVec3(0.36f, -0.8f, 0.48f).getNormalized());
    S.meshP = PxVec3(0.62f, -0.31f, 1.15f);
    // S3.  The default is three distinct POSITIVE components; the negScale variant
    // puts the negative one where it is NOT the smallest in magnitude, so
    // min(x,y,z) and min(|x|,|y|,|z|) differ in BOTH value and sign.
    S.scale = negScale ? PxVec3(1.17f, -2.90f, 0.86f) : PxVec3(1.35f, 0.72f, 2.10f);
    S.scaleRot = PxQuat(-0.44f, PxVec3(0.8f, 0.48f, 0.36f).getNormalized());
    S.contactDistTri = 0.052f;
    S.materialIndex = 0x2Bu;

    S.cloth[0].nbTris = 4000u;   S.cloth[0].clothId = 2u;
    S.cloth[0].shapeSlot = 1u;   S.cloth[0].cacheSlot = 2u;
    S.cloth[0].contactDist = 0.31f;
    S.cloth[0].rotW = PxQuat(0.37f, PxVec3(0.4f, -0.8f, 0.44721f).getNormalized());
    S.cloth[0].posW = PxVec3(0.21f, -0.63f, 1.07f);

    S.cloth[1].nbTris = 900u;    S.cloth[1].clothId = 5u;
    S.cloth[1].shapeSlot = 4u;   S.cloth[1].cacheSlot = 6u;
    // A DIFFERENT contact distance, so the two pairs get different cull scales and
    // different gates -- a kernel that reads pair 0's distance for pair 1 is caught.
    S.cloth[1].contactDist = 0.11f;
    S.cloth[1].rotW = PxQuat(-0.21f, PxVec3(-0.36f, 0.48f, 0.79772f).getNormalized());
    S.cloth[1].posW = PxVec3(-0.14f, 0.28f, 0.91f);

    buildCloth(S, S.cloth[0], 0x71A9C3u, 200u, 1.0f);
    buildCloth(S, S.cloth[1], 0x3E10B5u, 60u, 0.7f);

    // The rest distances are indexed BY PAIR (S2) and the array is sized like the
    // transform cache, so every value a wrong index could reach is a different one.
    S.restByPair.assign(Scene::NB_CACHE, 0.0f);
    for (uint32_t i = 0; i < Scene::NB_CACHE; ++i)
        S.restByPair[i] = -3.0f - 0.125f * float(i);
    S.restByPair[0] = 0.019f;
    S.restByPair[1] = 0.041f;

    // S5: a triangle is skipped only when ALL THREE of its vertices are filtered.
    // Both groups are populated for both cloths: the fully-filtered ones must
    // actually be dropped, and the partial ones must survive, or the conjunction
    // is indistinguishable from a disjunction.
    for (uint32_t k = 0; k < 2u; ++k)
    {
        ClothDef& c = S.cloth[k];
        for (uint32_t t = 3u; t < c.nbTris; t += 29u) c.filteredTris.push_back(t);
        for (uint32_t t = 11u; t < c.nbTris; t += 31u)
            if (std::find(c.filteredTris.begin(), c.filteredTris.end(), t) == c.filteredTris.end())
                c.partialTris.push_back(t);
    }
}

// ---------------------------------------------------------------------------
// The device arena.
// ---------------------------------------------------------------------------
// What a configuration must PROVE about itself.  The two variants exist for
// different reasons and demanding the same coverage of both would be a coverage
// check that only means something for one of them.
enum Coverage
{
    COV_FULL,       // the resume loop, the three-way filter and both gates
    COV_NEGSCALE,   // the sign of cullScale, and that the sign changes the answer
    COV_EXACTGATE   // a triangle whose sep is BIT-IDENTICAL to cullScale
};

struct Options
{
    bool clothFirst = true;      // which slot of the contact pair holds the cloth
    uint32_t numPairs = 1u;      // 1 -> the write order is fully determined
    uint32_t capacity = Scene::CAP;
    uint32_t seedCount = 0u;     // pre-seed totalContactCount
    Coverage coverage = COV_FULL;
};

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bBlob, bCin, bShapes, bXf, bCd, bRest, bRemap, bPairs, bDesc;
    DevBuf bVerts[2], bInds[2];
    DevBuf bPoint, bNormalPen, bBary, bInfo, bCount, bByX, bTempByX, bIdxByX, bSortedByX;
    ClothSdfBlob blob{};
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

    std::vector<PxgFEMCloth> descArr(Scene::NB_CLOTHES);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgFEMCloth));
    for (uint32_t k = 0; k < 2u; ++k)
    {
        R.bVerts[k] = R.mk(dev, S.cloth[k].verts);
        R.bInds[k] = R.mk(dev, S.cloth[k].inds);
        PxgFEMCloth fc; std::memset(&fc, 0, sizeof(fc));
        fc.mPosition_InvMass = reinterpret_cast<float4*>(size_t(R.bVerts[k].addr));
        fc.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(R.bInds[k].addr));
        fc.mNbVerts = uint32_t(S.cloth[k].verts.size());
        fc.mNbTriangles = S.cloth[k].nbTris;
        descArr[S.cloth[k].clothId] = fc;
    }
    R.bDesc = R.mk(dev, descArr);

    std::vector<PxgContactManagerInput> cin(opt.numPairs);
    std::memset(cin.data(), 0xEE, cin.size() * sizeof(PxgContactManagerInput));
    for (uint32_t p = 0; p < opt.numPairs; ++p)
    {
        const ClothDef& c = S.cloth[p];
        if (opt.clothFirst)
        {
            cin[p].shapeRef0 = c.shapeSlot;      cin[p].transformCacheRef0 = c.cacheSlot;
            cin[p].shapeRef1 = Scene::TRI_SHAPE; cin[p].transformCacheRef1 = Scene::TRI_CACHE;
        }
        else
        {
            cin[p].shapeRef0 = Scene::TRI_SHAPE; cin[p].transformCacheRef0 = Scene::TRI_CACHE;
            cin[p].shapeRef1 = c.shapeSlot;      cin[p].transformCacheRef1 = c.cacheSlot;
        }
    }
    R.bCin = R.mk(dev, cin);

    std::vector<PxgShape> shapes(Scene::NB_SHAPES);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    // Every decoy is ALSO a triangle mesh with a NULL geometry pointer and NO
    // deformable id, so a kernel that picks the wrong slot both mis-sorts the pair
    // and then faults -- a crash rather than a near miss.
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
    shapes[Scene::TRI_SHAPE].materialIndex = S.materialIndex;
    for (uint32_t k = 0; k < 2u; ++k)
    {
        PxgShape& sh = shapes[S.cloth[k].shapeSlot];
        sh.particleOrSoftbodyId = S.cloth[k].clothId;
        // S4: the cloth's own shape scale is NEVER applied.  It is set to something
        // loud here precisely so that a port which does apply it is wrong.
        sh.scale = PxMeshScale(PxVec3(0.31f, 5.7f, 0.09f),
                               PxQuat(0.6f, PxVec3(0.48f, -0.6f, 0.64031f).getNormalized()));
        sh.materialIndex = 0x99u;
    }
    R.bShapes = R.mk(dev, shapes);

    std::vector<PxsCachedTransform> xf(Scene::NB_CACHE);
    std::memset(xf.data(), 0, xf.size() * sizeof(PxsCachedTransform));
    for (auto& t : xf) t.transform = PxTransform(PxVec3(77.f, 77.f, 77.f), PxQuat(PxIdentity));
    xf[Scene::TRI_CACHE].transform = S.meshPose();
    for (uint32_t k = 0; k < 2u; ++k)
        xf[S.cloth[k].cacheSlot].transform = PxTransform(S.cloth[k].posW, S.cloth[k].rotW);
    R.bXf = R.mk(dev, xf);

    std::vector<float> cd(Scene::NB_CACHE, -5.f);
    cd[Scene::TRI_CACHE] = S.contactDistTri;
    for (uint32_t k = 0; k < 2u; ++k)
        cd[S.cloth[k].cacheSlot] = S.cloth[k].contactDist;
    R.bCd = R.mk(dev, cd);
    R.bRest = R.mk(dev, S.restByPair);

    // A DYNAMIC rigid: getInd() is then a two-word value whose high half is not
    // zero, so the u64 comparison in the filter's binary search is load-bearing and
    // the 32-bit `tempContactByX` truncation is visible.
    R.rigidId = PxNodeIndex(0x51u, 3u);
    std::vector<uint64_t> remap(Scene::NB_CACHE, 0xDEADBEEFDEADBEEFull);
    remap[Scene::TRI_CACHE] = R.rigidId.getInd();
    R.bRemap = R.mk(dev, remap);

    // The filter list, sorted by (index0, index1, index2) exactly as the engine's
    // binary search requires.
    R.pairs.clear();
    for (uint32_t k = 0; k < 2u; ++k)
    {
        const ClothDef& c = S.cloth[k];
        auto addVert = [&](uint32_t v) {
            PxgRigidFilterPair p;
            p.index0 = R.rigidId.getInd();
            p.index1 = encodeCloth(c.clothId, v);
            p.index2 = 0u;
            R.pairs.push_back(p);
        };
        for (uint32_t t : c.filteredTris)
            for (uint32_t j = 0; j < 3u; ++j) addVert(c.inds[t * 4u + j]);
        // Two of three only.  WHICH two rotates with the triangle so that each of
        // the three conjuncts is the surviving one somewhere.
        for (size_t i = 0; i < c.partialTris.size(); ++i)
        {
            const uint32_t t = c.partialTris[i];
            const uint32_t skip = uint32_t(i % 3u);
            for (uint32_t j = 0; j < 3u; ++j)
                if (j != skip) addVert(c.inds[t * 4u + j]);
        }
    }
    std::sort(R.pairs.begin(), R.pairs.end(),
              [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
                  if (a.index0 != b.index0) return a.index0 < b.index0;
                  if (a.index1 != b.index1) return a.index1 < b.index1;
                  return a.index2 < b.index2;
              });
    R.pairs.erase(std::unique(R.pairs.begin(), R.pairs.end(),
                              [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
                                  return a.index0 == b.index0 && a.index1 == b.index1 &&
                                         a.index2 == b.index2;
                              }),
                  R.pairs.end());
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

    R.blob = ClothSdfBlob{};
    R.blob.numWorkItems = opt.numPairs;
    R.blob.cmInputs = R.bCin.addr;
    R.blob.transformCache = R.bXf.addr;
    R.blob.contactDistances = R.bCd.addr;
    R.blob.restDistances = R.bRest.addr;
    R.blob.gpuShapes = R.bShapes.addr;
    R.blob.femClothes = R.bDesc.addr;
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
// Reference dispatch 1: the gather (P9.34b's own entry point).
// ---------------------------------------------------------------------------
static bool refGather(PxgSlangComputeDevice* dev, const Scene& S, const ClothDef& c, Rig& R,
                      uint32_t clothSlot, std::vector<std::vector<uint32_t>>& rounds,
                      float toleranceOverride = 0.0f)
{
    const PxTransform aToB = aToBOf(S, c);
    const std::vector<F4> echoInit(c.nbTris, F4{ 0, 0, 0, 0 });
    const std::vector<uint32_t> rInit(FIND_MAXROUNDS * 4u, POISON_U32);
    const std::vector<uint32_t> bInit(size_t(FIND_MAXROUNDS) * FIND_TARGET * 2u, POISON_U32);
    const std::vector<uint32_t> nInit(1u, POISON_U32);
    DevBuf bE = R.mk(dev, echoInit), bR = R.mk(dev, rInit);
    DevBuf bB = R.mk(dev, bInit), bN = R.mk(dev, nInit);

    P_Find p{};
    p.meshPtr = R.bBlob.addr;
    p.positions = R.bVerts[clothSlot].addr;
    p.meshInd = R.bInds[clothSlot].addr;
    p.echoOut = bE.addr; p.roundsOut = bR.addr; p.bufOut = bB.addr; p.numRoundsOut = bN.addr;
    p.aToBq[0] = aToB.q.x; p.aToBq[1] = aToB.q.y; p.aToBq[2] = aToB.q.z; p.aToBq[3] = aToB.q.w;
    p.aToBp[0] = aToB.p.x; p.aToBp[1] = aToB.p.y; p.aToBp[2] = aToB.p.z;
    // scale1 is the TRIMESH's; scale0 is the cloth's, and S4 says it is the identity.
    p.scaleRot[0] = S.scaleRot.x; p.scaleRot[1] = S.scaleRot.y;
    p.scaleRot[2] = S.scaleRot.z; p.scaleRot[3] = S.scaleRot.w;
    p.scale[0] = S.scale.x; p.scale[1] = S.scale.y; p.scale[2] = S.scale.z;
    p.scale0Rot[0] = 0.0f; p.scale0Rot[1] = 0.0f; p.scale0Rot[2] = 0.0f; p.scale0Rot[3] = 1.0f;
    p.scale0[0] = 1.0f; p.scale0[1] = 1.0f; p.scale0[2] = 1.0f;
    p.tolerance = (toleranceOverride != 0.0f) ? toleranceOverride : cullScaleOf(S, c);
    p.numVertices = c.nbTris;
    p.maxRounds = FIND_MAXROUNDS;
    {
        Kernel k(Kernel::Bindless{}, "npSdfFind.spv", "sdfFindTriangleRounds", {}, sizeof(P_Find));
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
    const auto buf = downloadBuffer<uint32_t>(dev, bB.h,
                                              size_t(FIND_MAXROUNDS) * FIND_TARGET * 2u);
    rounds.assign(nr, {});
    for (uint32_t r = 0; r < nr; ++r)
    {
        const uint32_t n = hdr[size_t(r) * 4];
        for (uint32_t t = 0; t < n; ++t)
            rounds[r].push_back(buf[(size_t(r) * FIND_TARGET + t) * 2u]);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Reference dispatch 2: the descent (P9.34a's own entry point), over the union of
// the gathered indices.  The three vertices come from the HOST, through engine
// transforms -- that is what makes the kernel's own chain a claim rather than an
// assumption.
// ---------------------------------------------------------------------------
struct Descent { F4 point, dir; float sep; };

static bool refDescent(PxgSlangComputeDevice* dev, const Scene& S, const ClothDef& c, Rig& R,
                       const std::vector<uint32_t>& tris,
                       std::vector<Descent>& out, std::vector<PxVec3>& v0o,
                       std::vector<PxVec3>& v1o, std::vector<PxVec3>& v2o)
{
    const PxTransform aToB = aToBOf(S, c);
    const float cull = cullScaleOf(S, c);
    const PxVec3 unitScale(1.0f, 1.0f, 1.0f);
    const PxQuat unitRot(PxIdentity);

    const uint32_t n = uint32_t(tris.size());
    std::vector<F4> tv(size_t(n) * 3u);
    v0o.resize(n); v1o.resize(n); v2o.resize(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t t = tris[i];
        PxVec3 v[3];
        for (uint32_t k = 0; k < 3u; ++k)
        {
            const uint32_t vi = c.inds[t * 4u + k];
            const F4& w = c.verts[vi];
            // getTriangleVertices with the cloth's UNIT scale (S4), then into the
            // trimesh's VERTEX space, which is where the descent works.
            const PxVec3 shapeSpace = hVertex2Shape(PxVec3(w.x, w.y, w.z), unitScale, unitRot);
            v[k] = hShape2Vertex(aToB.transform(shapeSpace), S.scale, S.scaleRot);
        }
        v0o[i] = v[0]; v1o[i] = v[1]; v2o[i] = v[2];
        tv[size_t(i) * 3u + 0] = F4{ v[0].x, v[0].y, v[0].z, cull };   // .w carries tolerance
        tv[size_t(i) * 3u + 1] = F4{ v[1].x, v[1].y, v[1].z, 0.0f };
        tv[size_t(i) * 3u + 2] = F4{ v[2].x, v[2].y, v[2].z, 0.0f };
    }
    DevBuf bTv = R.mk(dev, tv);
    DevBuf bSeed = R.mk(dev, std::vector<F4>(n, F4{ 0, 0, 0, 0 }));
    DevBuf bOut = R.mk(dev, std::vector<float>(size_t(n) * TRI_TRACE_STRIDE, -4242.f));

    P_Coll p{};
    p.meshPtr = R.bBlob.addr;
    p.dirSeed = bSeed.addr;
    p.triVerts = bTv.addr;
    p.triOut = bOut.addr;
    p.numPoints = n;
    {
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTriangle", {}, sizeof(P_Coll));
        PXG_CHECK_MSG(k.isValid(), "the descent reference pipeline must build");
        if (!k.isValid()) return false;
        k.dispatch((n + 63u) / 64u, {}, p);
    }
    dev->flush();
    const auto o = downloadBuffer<float>(dev, bOut.h, size_t(n) * TRI_TRACE_STRIDE);
    out.resize(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const float* e = &o[size_t(i) * TRI_TRACE_STRIDE];
        out[i].point = F4{ e[20], e[21], e[22], 0.0f };
        out[i].dir = F4{ e[23], e[24], e[25], 0.0f };
        out[i].sep = e[26];
    }
    return true;
}

// ---------------------------------------------------------------------------
// The expectation for ONE contact manager, built entirely from those two.
// ---------------------------------------------------------------------------
struct Contact
{
    uint32_t tri;
    F4 point, normalPen, bary;
    uint32_t pairInd1;
};
struct Expect
{
    std::vector<Contact> contacts;      // in the kernel's own round-then-thread order
    uint32_t gathered = 0, skipped = 0, partialKept = 0;
    uint32_t culled = 0, gated = 0, zeroDir = 0;
    // Triangles sitting EXACTLY on the cull gate, and how many of those an `<=`
    // spelling would carry all the way to a written contact.
    uint32_t exactGate = 0, exactGateWouldWrite = 0;
    size_t rounds = 0;
};

static Expect predictPair(const Scene& S, const ClothDef& c, const Rig& R,
                          const std::vector<std::vector<uint32_t>>& rounds,
                          const std::vector<uint32_t>& tris,
                          const std::vector<Descent>& desc,
                          const std::vector<PxVec3>& v0a, const std::vector<PxVec3>& v1a,
                          const std::vector<PxVec3>& v2a, float rest)
{
    Expect E;
    E.rounds = rounds.size();
    const float cull = cullScaleOf(S, c);
    const float contactDistance = contactDistOf(S, c);
    const PxTransform meshPose = S.meshPose();

    auto listed = [&](uint32_t v) {
        PxgRigidFilterPair q;
        q.index0 = R.rigidId.getInd();
        q.index1 = encodeCloth(c.clothId, v);
        q.index2 = 0u;
        return std::binary_search(R.pairs.begin(), R.pairs.end(), q,
                                  [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
                                      if (a.index0 != b.index0) return a.index0 < b.index0;
                                      return a.index1 < b.index1;
                                  });
    };

    for (const std::vector<uint32_t>& round : rounds)
    {
        // The block scan runs in thread order and thread t holds round[t].
        for (uint32_t tid = 0; tid < round.size(); ++tid)
        {
            const uint32_t t = round[tid];
            ++E.gathered;
            const size_t at = size_t(std::lower_bound(tris.begin(), tris.end(), t) - tris.begin());

            // S5: ALL THREE.
            const bool f0 = listed(c.inds[t * 4u + 0]);
            const bool f1 = listed(c.inds[t * 4u + 1]);
            const bool f2 = listed(c.inds[t * 4u + 2]);
            if (f0 && f1 && f2) { ++E.skipped; continue; }
            if (f0 || f1 || f2) ++E.partialKept;

            const PxVec3 pointV(desc[at].point.x, desc[at].point.y, desc[at].point.z);
            float sep = desc[at].sep;

            F4 bary;
            hComputeBarycentric(v0a[at], v1a[at], v2a[at], pointV, bary);
            const PxVec3 posShape = hVertex2Shape(pointV, S.scale, S.scaleRot);

            // The one place `<` and `<=` can part company, counted so the fixture can
            // insist the rig reaches it -- and counted TWICE, because a triangle that
            // slips past a `<=` gate and is then stopped by the second one leaves no
            // trace either way.
            if (f2u(sep) == f2u(cull))
            {
                ++E.exactGate;
                PxVec3 d2 = hShape2Vertex(PxVec3(desc[at].dir.x, desc[at].dir.y, desc[at].dir.z),
                                          S.scale, S.scaleRot);
                float m2 = d2.dot(d2);
                float s2 = sep;
                if (m2 > 0.0f) { m2 = 1.0f / std::sqrt(m2); s2 = s2 * m2; }
                if (s2 < contactDistance) ++E.exactGateWouldWrite;
            }

            if (!(sep < cull)) { ++E.culled; continue; }

            // vertex2ShapeNormalVector IS shape2Vertex.
            PxVec3 dir = hShape2Vertex(PxVec3(desc[at].dir.x, desc[at].dir.y, desc[at].dir.z),
                                       S.scale, S.scaleRot);
            float m = dir.dot(dir);
            if (m > 0.0f)
            {
                m = 1.0f / std::sqrt(m);
                sep = sep * m;                 // S6: rescaled BEFORE the second gate
                dir = dir * m;
            }
            else
            {
                ++E.zeroDir;
            }
            if (!(sep < contactDistance)) { ++E.gated; continue; }

            const PxVec3 worldN = meshPose.rotate(dir);          // ENGINE
            const PxVec3 worldP = meshPose.transform(posShape);  // ENGINE
            Contact k;
            k.tri = t;
            k.point = F4{ worldP.x, worldP.y, worldP.z, rest };
            k.normalPen = F4{ worldN.x, worldN.y, worldN.z, sep };
            k.bary = bary;
            k.pairInd1 = encodeCloth(c.clothId, t);
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

    std::vector<Expect> exp(opt.numPairs);
    for (uint32_t p = 0; p < opt.numPairs; ++p)
    {
        std::vector<std::vector<uint32_t>> rounds;
        if (!refGather(dev, S, S.cloth[p], R, p, rounds)) { R.destroy(dev); return; }

        std::vector<uint32_t> tris;
        for (const auto& r : rounds) tris.insert(tris.end(), r.begin(), r.end());
        std::sort(tris.begin(), tris.end());
        tris.erase(std::unique(tris.begin(), tris.end()), tris.end());
        if (tris.empty())
        {
            PXG_CHECK_MSG(false, "%s: pair %u gathered nothing -- the rig proves nothing", label, p);
            R.destroy(dev);
            return;
        }

        std::vector<Descent> desc;
        std::vector<PxVec3> v0a, v1a, v2a;
        if (!refDescent(dev, S, S.cloth[p], R, tris, desc, v0a, v1a, v2a))
        { R.destroy(dev); return; }

        exp[p] = predictPair(S, S.cloth[p], R, rounds, tris, desc, v0a, v1a, v2a,
                             S.restByPair[p]);
    }

    {
        Kernel k(Kernel::Bindless{}, "femClothSdfMeshCgBda.spv", "cloth_sdfMeshContactGenLaunch",
                 {}, sizeof(ClothSdfBlob));
        PXG_CHECK_MSG(k.isValid(), "%s: the kernel pipeline must build", label);
        if (!k.isValid()) { R.destroy(dev); return; }
        k.dispatch(opt.numPairs, {}, R.blob);       // grid (numTests, 1, 1) -- S1
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

    // Coverage, DEMANDED rather than designed.
    PXG_CHECK_MSG(!exp[0].contacts.empty(), "%s: pair 0 produced no contact at all", label);
    PXG_CHECK_MSG(exp[0].skipped > 0u && exp[0].partialKept > 0u,
                  "%s: the filter dropped %u triangles and kept %u partially-filtered ones "
                  "(both must be > 0, or the three-way conjunction is unobservable)",
                  label, exp[0].skipped, exp[0].partialKept);
    if (opt.coverage == COV_FULL)
    {
        PXG_CHECK_MSG(exp[0].rounds >= 2,
                      "%s: pair 0's gather took %zu round(s) -- the resume path is untested",
                      label, exp[0].rounds);
        PXG_CHECK_MSG(exp[0].culled > 0u && exp[0].gated > 0u,
                      "%s: %u triangles failed the cull gate and %u the contact-distance gate "
                      "(both must be > 0, or the two gates are one)",
                      label, exp[0].culled, exp[0].gated);
    }
    else if (opt.coverage == COV_EXACTGATE)
    {
        // THIS IS A PROOF MADE EXECUTABLE, and it did not start out as one.
        //
        // `<` and `<=` on the cull gate part company at exactly one point, and the
        // search above manufactures it: a contact distance whose derived cullScale is
        // bit-identical to some gathered triangle's separation.  The first half of
        // the check is that the rig REACHES that point.
        //
        // The second half is the surprise.  Reaching it is not enough, because the
        // triangle an `<=` gate would let through is then stopped by the SECOND gate,
        // always, in exact arithmetic:
        //
        //   at the boundary   sep == cullScale == contactDistance / min(scale)
        //   `dir` comes back NORMALISED, so |dir| == 1
        //   shape2Vertex divides each rotated component by its scale, and a rotation
        //   preserves length, so |shape2Vertex(dir)| <= 1 / min(scale)
        //   the rescale is m = 1 / |shape2Vertex(dir)| >= min(scale)
        //   hence sep * m >= contactDistance, and `sep * m < contactDistance` is FALSE
        //
        // So `<` and `<=` write identical bytes for every positive-scale input, and
        // the mutant that swapped them is an EQUIVALENT rather than a gap.  The check
        // below is the claim, not a hope: if a triangle on the gate ever does reach a
        // write, the proof is wrong and this fires.
        PXG_CHECK_MSG(exp[0].exactGate > 0u,
                      "%s: no triangle sits exactly on the cull gate -- the search that was "
                      "supposed to manufacture one did not survive the rebuild", label);
        PXG_CHECK_MSG(exp[0].exactGateWouldWrite == 0u,
                      "%s: %u of %u triangles on the cull gate would reach a write under an "
                      "`<=` spelling -- the min(scale) argument above is WRONG and the two "
                      "gates are observably different; this needs a mutant, not a proof",
                      label, exp[0].exactGateWouldWrite, exp[0].exactGate);
    }
    else
    {
        // S3's own contract, and it is about DISCRIMINATION, not about exercising a
        // path.  Two things have to hold or the negative-scale run proves nothing:
        // the tolerance the kernel derives must be negative, and the abs() spelling
        // must gather a materially different set -- otherwise `min` and `min of the
        // magnitudes` would agree here and the mutant would be an equivalent.
        const float cull = cullScaleOf(S, S.cloth[0]);
        const float cullAbs = contactDistOf(S, S.cloth[0]) /
                              PxMin(PxAbs(S.scale.x), PxMin(PxAbs(S.scale.y), PxAbs(S.scale.z)));
        PXG_CHECK_MSG(cull < 0.0f && cullAbs > 0.0f,
                      "%s: cullScale %.9g / abs-spelling %.9g -- the two must differ in SIGN",
                      label, cull, cullAbs);
        std::vector<std::vector<uint32_t>> roundsAbs;
        if (refGather(dev, S, S.cloth[0], R, 0u, roundsAbs, cullAbs))
        {
            size_t nAbs = 0;
            for (const auto& r : roundsAbs) nAbs += r.size();
            PXG_CHECK_MSG(nAbs > size_t(exp[0].gathered) + 200u,
                          "%s: the abs() spelling gathers %zu triangles against the engine's %u "
                          "-- too close for the sign to be observable", label, nAbs,
                          exp[0].gathered);
        }
    }

    // ---- the contact stream ----
    //
    // With ONE contact manager there is one block, one atomic per gather round, and
    // the rounds run in order -- so the slot each contact lands in is DETERMINED and
    // is asserted.  With two, the two blocks race on the same counter, so the P9.11
    // rule applies: the SET is asserted, the interleaving is not.
    size_t bad = 0;
    auto checkAt = [&](const Contact& e, uint32_t at) {
        if (at >= opt.capacity) return;             // the writer returns before any store
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
        if (!bitEq(bary[at].x, e.bary.x) || !bitEq(bary[at].y, e.bary.y) ||
            !bitEq(bary[at].z, e.bary.z) || !bitEq(bary[at].w, e.bary.w))
        {
            if (bad < 4)
                std::printf("    %s slot %u bary (%.9g,%.9g,%.9g,%.9g) expected "
                            "(%.9g,%.9g,%.9g,%.9g)\n", label, at, bary[at].x, bary[at].y,
                            bary[at].z, bary[at].w, e.bary.x, e.bary.y, e.bary.z, e.bary.w);
            ++bad;
        }
        const uint64_t rid = R.rigidId.getInd();
        if (info[size_t(at) * 4u + 0] != uint32_t(rid & 0xffffffffu)) ++bad;
        if (info[size_t(at) * 4u + 1] != uint32_t(rid >> 32)) ++bad;
        if (info[size_t(at) * 4u + 2] != e.pairInd1) ++bad;
        // setRigidMaterialIndex(mat) then markInCollision(false): bit 31 cleared.
        if (info[size_t(at) * 4u + 3] != (S.materialIndex & 0x7fffffffu)) ++bad;
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
        // Slot-agnostic: pair every written slot with the expectation that carries
        // the same pairInd1, and require the pairing to be a bijection.
        std::vector<const Contact*> want;
        for (uint32_t p = 0; p < opt.numPairs; ++p)
            for (const Contact& k : exp[p].contacts) want.push_back(&k);
        std::vector<uint8_t> used(want.size(), 0u);
        uint32_t writtenSlots = 0;
        for (uint32_t at = opt.seedCount; at < PxMin(total, opt.capacity); ++at)
        {
            ++writtenSlots;
            size_t hit = want.size();
            for (size_t j = 0; j < want.size(); ++j)
                if (!used[j] && want[j]->pairInd1 == info[size_t(at) * 4u + 2]) { hit = j; break; }
            if (hit == want.size()) { ++bad; continue; }
            used[hit] = 1u;
            checkAt(*want[hit], at);
        }
        PXG_CHECK_MSG(writtenSlots == PxMin(total, opt.capacity) - opt.seedCount,
                      "%s: %u slots walked", label, writtenSlots);
    }
    PXG_CHECK_MSG(bad == 0, "%s: %zu contact-stream comparisons failed", label, bad);

    // Everything past the reservation, and everything past the capacity, must be
    // exactly as it was uploaded.
    size_t spill = 0;
    for (uint32_t at = PxMin(total, opt.capacity); at < Scene::CAP; ++at)
        if (idxByX[at] != POISON_U32 || byX[at] != ~0ull || !bitEq(point[at].x, -7.f)) ++spill;
    PXG_CHECK_MSG(spill == 0, "%s: %zu slots beyond the reservation were written", label, spill);

    std::printf("    [ note ] %s: %zu round(s), %u gathered, %u filtered out (%u partial kept), "
                "%u culled (%u exactly on the gate, %u of those writable), %u gated, %u zero-dir "
                "-> %zu contacts; total %u (cap %u)\n",
                label, exp[0].rounds, exp[0].gathered, exp[0].skipped, exp[0].partialKept,
                exp[0].culled, exp[0].exactGate, exp[0].exactGateWouldWrite, exp[0].gated,
                exp[0].zeroDir, exp[0].contacts.size(), total, opt.capacity);
    R.destroy(dev);
}

static void runOne(PxgSlangComputeDevice* dev, const char* label, const Options& opt,
                   bool negScale)
{
    Scene S;
    buildScene(S, negScale);
    runOneScene(dev, label, opt, S);
}

// ---------------------------------------------------------------------------
// The exact-gate SEARCH.
//
// `sep < cullScale` and `sep <= cullScale` differ at exactly one point, and no
// amount of random geometry lands on it.  So the boundary is MANUFACTURED, the
// same way P9.34b manufactured the gather's, and the search is possible because
// of one fact about the descent:
//
//   doTriangleSDFCollision's return is INDEPENDENT of the tolerance it is given,
//   except that the tolerance decides whether the value is returned at all.  The
//   last line is PxSdfSampleWithGrad, which returns `dist + dir.dot(diff)` when
//   `dist < tolerance` and PX_MAX_F32 otherwise; nothing inside the descent's
//   iteration reads the tolerance.
//
// So a triangle's separation is a constant V, and the job is to find a contact
// distance whose derived cullScale IS that constant, bit for bit:
//
//   cullScale = fl( fl(contactDistTri + c) / min(scale) )   ==   V
//
// which is monotone in `c`, so the answer -- if it exists -- is within an ULP or
// two of `V * min(scale) - contactDistTri`.  The equality is then CONFIRMED by
// re-running the references, not assumed: moving `c` moves the tolerance, which
// moves both the gathered set and which triangles clear the descent's own gate,
// and a candidate that drops out is simply the wrong candidate.
//
// One more condition, and leaving it out would have made the whole exercise
// decorative: `dist < tolerance` must still hold at V == tolerance, i.e.
// dir.dot(diff) must be strictly positive.  That is only true where the descent's
// point lies OUTSIDE the SDF box, so the candidates are ranked by that dot
// product, computed on the host from the trace's own `point` and `dir`.
// ---------------------------------------------------------------------------
static bool probePair0(PxgSlangComputeDevice* dev, Scene& S, std::vector<uint32_t>& tris,
                       std::vector<Descent>& desc)
{
    Options o;
    Rig R;
    bool ok = false;
    if (buildRig(dev, S, o, R))
    {
        std::vector<std::vector<uint32_t>> rounds;
        if (refGather(dev, S, S.cloth[0], R, 0u, rounds))
        {
            tris.clear();
            for (const auto& r : rounds) tris.insert(tris.end(), r.begin(), r.end());
            std::sort(tris.begin(), tris.end());
            tris.erase(std::unique(tris.begin(), tris.end()), tris.end());
            std::vector<PxVec3> a, b, c2;
            ok = !tris.empty() && refDescent(dev, S, S.cloth[0], R, tris, desc, a, b, c2);
        }
    }
    R.destroy(dev);
    return ok;
}

static bool searchExactGate(PxgSlangComputeDevice* dev, Scene& S, float& foundC, uint32_t& foundTri)
{
    const float minScale = PxMin(S.scale.x, PxMin(S.scale.y, S.scale.z));
    const HostSdfBox box = hostSdfBox(S.sdf);

    std::vector<uint32_t> tris;
    std::vector<Descent> desc;
    if (!probePair0(dev, S, tris, desc)) return false;

    // Rank the gathered triangles by dir.dot(diff), which is what keeps
    // `dist < tolerance` true once the tolerance is pushed down onto V.
    struct Cand { uint32_t tri; float V; float dot; };
    std::vector<Cand> cand;
    for (size_t i = 0; i < tris.size(); ++i)
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
        cand.push_back(Cand{ tris[i], V, dt });
    }
    std::sort(cand.begin(), cand.end(),
              [](const Cand& a, const Cand& b) { return a.dot > b.dot; });

    const float saved = S.cloth[0].contactDist;
    uint32_t tried = 0;
    for (const Cand& k : cand)
    {
        if (tried >= 8u) break;
        // The ULP walk around the ideal contact distance.
        const float ideal = k.V * minScale - S.contactDistTri;
        float hit = 0.0f;
        bool haveHit = false;
        for (int step = -96; step <= 96 && !haveHit; ++step)
        {
            uint32_t u = f2u(ideal);
            u = uint32_t(int32_t(u) + step);
            float c; std::memcpy(&c, &u, 4);
            if (!(c > 0.0f)) continue;
            const float cd = S.contactDistTri + c;
            if (f2u(cd / minScale) == f2u(k.V)) { hit = c; haveHit = true; }
        }
        if (!haveHit) continue;

        ++tried;
        S.cloth[0].contactDist = hit;
        std::vector<uint32_t> t2;
        std::vector<Descent> d2;
        if (probePair0(dev, S, t2, d2))
        {
            const float cull = cullScaleOf(S, S.cloth[0]);
            for (size_t i = 0; i < t2.size(); ++i)
                if (t2[i] == k.tri && f2u(d2[i].sep) == f2u(cull))
                {
                    foundC = hit; foundTri = k.tri;
                    return true;                    // S keeps the winning distance
                }
        }
        S.cloth[0].contactDist = saved;
    }
    S.cloth[0].contactDist = saved;
    return false;
}

static bool beginShim(PxgSlangComputeDevice* dev, PxgCudaShim*& shim)
{
    static const PxgKernelDef manifest[] = {
        { "femClothSdfMeshCgBda.spv", "cloth_sdfMeshContactGenLaunch", 0u, 0u } };
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

PXG_TEST(femClothSdfContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    PxgCudaShim* shim = nullptr;
    beginShim(dev, shim);

    // Both slot orders, so the cloth/trimesh sort is exercised in both directions.
    { Options o; o.clothFirst = true;  runOne(dev, "cloth-first", o, false); }
    { Options o; o.clothFirst = false; runOne(dev, "mesh-first", o, false); }
    // A pre-seeded counter: the block scan's base is the value it READ, not zero.
    { Options o; o.seedCount = 37u; runOne(dev, "seeded-counter", o, false); }
    // The capacity cap: the reservation still advances past maxNumContacts while
    // the writer stores nothing -- the count and the arrays disagree on purpose.
    { Options o; o.capacity = 24u; runOne(dev, "capped", o, false); }
    // TWO contact managers in one launch: pairIdx reaches cmInputs, restDistances
    // and a different cloth id, and each block gathers with its own cull scale.
    { Options o; o.numPairs = 2u; runOne(dev, "two-pairs", o, false); }

    endShim(shim);
}

PXG_TEST(femClothSdfContactGenExactGate)
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
    uint32_t tri = 0u;
    if (!searchExactGate(dev, S, c, tri))
    {
        // Not a soft failure: without the boundary this test asserts nothing, and
        // saying so is the point.  The search is deterministic, so this cannot be
        // a flake -- it would mean the scene changed under it.
        PXG_CHECK_MSG(false, "no contact distance puts a gathered triangle's separation "
                             "exactly on the cull gate -- the `<` / `<=` boundary is untested");
    }
    else
    {
        std::printf("    [ note ] exact-gate: triangle %u, contact distance %.9g -> cullScale "
                    "%.9g, bit-identical to its separation\n", tri, c, cullScaleOf(S, S.cloth[0]));
        Options o; o.coverage = COV_EXACTGATE;
        runOneScene(dev, "exact-gate", o, S);
    }

    endShim(shim);
}

PXG_TEST(femClothSdfContactGenNegScale)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    PxgCudaShim* shim = nullptr;
    beginShim(dev, shim);

    // S3.  cullScale divides by min(x, y, z) and NOT by min(|x|, |y|, |z|) -- the
    // particle sibling (psSdfMidphaseBda, W3) takes the absolute value from the same
    // expression shape and this one does not.  With scale.y = -2.9 the two spellings
    // differ in sign as well as magnitude, so a port that "fixes" this gathers a
    // completely different set.
    { Options o; o.coverage = COV_NEGSCALE; runOne(dev, "neg-scale/cloth-first", o, true); }
    { Options o; o.coverage = COV_NEGSCALE; o.clothFirst = false;
      runOne(dev, "neg-scale/mesh-first", o, true); }

    endShim(shim);
}

}  // namespace
