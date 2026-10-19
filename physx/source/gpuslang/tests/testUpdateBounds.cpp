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

// Verification of updateBounds.slang against a C++ transcription of
// updateTransformCacheAndBoundArrayLaunch (rigid + articulation paths) +
// updateCacheAndBound.cuh: world pose composition, transform cache refresh,
// frozen/unfrozen/updated/activate/deactivate bookkeeping and world-space
// AABB computation for sphere / capsule / box / convex geometry.
//
// updateTransformCacheAndBoundArrayLaunch is verified on the ENGINE launch
// ABI (P5 Step 4): the test fills the REAL engine PxgSimulationCoreDesc with
// live device addresses (including a raw PxgArticulation pool whose
// linkBody2Worlds / linkBody2Actors pointers are planted at the engine's
// field offsets) and launches through PxgCudaShim with the engine's grid
// geometry (GridDim/BlockDim::UPDATE_TRANSFORMCACHE_AND_BOUNDARRAY =
// 256 x 256), plus a 2-block variant that executes the grid-stride wrap.
// setRigidDynamicGlobalPose stays on the bound-buffer harness.
//
// mergeTransformCacheAndBoundArrayChanges (P9.95, the direct-GPU API's
// between-steps merge) is verified bitwise against a literal transcription of
// updateTransformAndBoundArray.cu:52-73 in the real PhysX types, through the
// framework (updateBounds_mergeTcBound) and through PxgCudaShim with pinned
// host arrays (updateBounds_mergeTcBound_pinned).

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the engine desc header
#include "PxgSimulationCoreDesc.h"
#include "PxgArticulation.h"
#include "PxgAABBManager.h"
#include "PxgBoundTransformUpdate.h" // P9.95: the merge kernel's record (5.12 renamed it from PxBoundTransformUpdate)
#include "PxsTransformCache.h"  // P9.95: PxsCachedTransform
#include "PxgVulkanKernelAbi.h" // P9.95: the merge kernel's launch-ABI row

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>

namespace
{

using namespace pxgtest;

struct UpdateBoundsParams
{
    uint32_t numShapes;      // (legacy slot, unused by the engine-ABI entry)
    uint32_t nbElements;     // setRigidDynamicGlobalPose
    uint32_t totalNumShapes; // setRigidDynamicGlobalPose
};

// Mirrors the bridge's marshalling: place each argument at the natural
// alignment inferred from its size, then round the blob to the largest
// alignment seen (a compiled struct's trailing padding).
struct EngineBlob
{
    std::vector<uint8_t> bytes;
    size_t maxAlign = 1;

    static size_t alignFor(size_t size)
    {
        if (size >= 16) return 16;
        if (size >= 8)  return 8;
        if (size >= 4)  return 4;
        if (size >= 2)  return 2;
        return 1;
    }
    template <typename T>
    void add(const T& v)
    {
        const size_t a = alignFor(sizeof(T));
        if (a > maxAlign) maxAlign = a;
        bytes.resize((bytes.size() + (a - 1)) & ~(a - 1));
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + (maxAlign - 1)) & ~(maxAlign - 1)); }
};

static std::vector<uint8_t> updateBoundsAbiLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// CPU reference: transcription of updateCacheAndBound.cuh
// ---------------------------------------------------------------------------

TransformV getAbsPose(const TransformV& body2World, const TransformV& shape2Actor, const TransformV& body2Actor)
{
    // P9.93: PxTransform::transformInv(PxTransform) LITERALLY (PxTransform.h:162-169):
    // the conjugate's rotate, not rotateInv -- they differ in the sign of exact zeros.
    // (TransformV::transformInv in pxgRefMath.h spells it the same way since P9.94a.)
    const QuatV qinv = body2Actor.q.getConjugate();
    const TransformV t0(qinv * shape2Actor.q, qinv.rotate(shape2Actor.p - body2Actor.p));
    return body2World.transform(t0);
}

Vec3 basisExtent(const Mat33& basis, const Vec3& extent)
{
    const Vec3 c0 = basis.col0 * extent.x;
    const Vec3 c1 = basis.col1 * extent.y;
    const Vec3 c2 = basis.col2 * extent.z;

    return Vec3(std::fabs(c0.x) + std::fabs(c1.x) + std::fabs(c2.x),
                std::fabs(c0.y) + std::fabs(c1.y) + std::fabs(c2.y),
                std::fabs(c0.z) + std::fabs(c1.z) + std::fabs(c2.z));
}

// P9.89: PxBounds3::transformFast(PxTransform) (PxBounds3.h:472-476, reached from
// updateCacheAndBound.cuh:148 and :156) = basisExtent(transform.transform(getCenter()),
// PxMat33(q), getExtents()).  The CENTER goes through PxTransform::transform = q.rotate(c) + p
// (PxTransform.h:128-132, PxQuat.h:286-295: vx*w2 + (y*vz - z*vy)*w + x*dot2); only the extents
// use the PxMat33(q) basis.  p89OldOrder = the pre-P9.89 port's twin, which centered with the
// PxMat33(q) product col0*c.x + col1*c.y + col2*c.z: mathematically equal, a different
// operation tree, different bits whenever the local center is non-zero.
void transformFastBounds(const TransformV& pose, const Vec3& localMin, const Vec3& localMax, Bounds3& out,
                         bool p89OldOrder = false)
{
    const Mat33 rot(pose.q);
    const Vec3 center = (localMax + localMin) * 0.5f;
    const Vec3 extents = (localMax - localMin) * 0.5f;
    const Vec3 newCenter = (p89OldOrder ? rot.transform(center) : pose.q.rotate(center)) + pose.p;
    const Vec3 newExtents = basisExtent(rot, extents);
    out.minimum = (newCenter - newExtents).toFloat3();
    out.maximum = (newCenter + newExtents).toFloat3();
}

// P9.89: PxMeshScale::toMat33 (PxMeshScale.h:120-128, reached from updateCacheAndBound.cuh:113-114):
// trans = rot.getTranspose(); trans.column_k *= scale[k]; return trans * rot.  Element (i,j) is
// ((R.col_i[0]*s0)*R.col_j[0] + (R.col_i[1]*s1)*R.col_j[1]) + (R.col_i[2]*s2)*R.col_j[2].
// p89OldOrder = the pre-P9.89 port's twin: rot^T * (diag(s) * rot), which scales R's ROWS and
// gives R.col_i[k]*(R.col_j[k]*s_k) -- the same sum order over k, a different product grouping
// (the diagonal is the same product; every off-diagonal element can differ).
Mat33 meshScaleToMat33(const Vec3& scale, const QuatV& rotation, bool p89OldOrder = false)
{
    const Mat33 rot(rotation);
    if (p89OldOrder)
    {
        Mat33 diagTimesRot(rot.col0.multiply(scale), rot.col1.multiply(scale), rot.col2.multiply(scale));
        return rot.getTranspose() * diagTimesRot;
    }
    Mat33 trans = rot.getTranspose();
    trans.col0 = trans.col0 * scale.x;
    trans.col1 = trans.col1 * scale.y;
    trans.col2 = trans.col2 * scale.z;
    return trans * rot;
}

// P9.86c: mutant twin of the convex-hull min/max loop.  The CUDA
// (updateCacheAndBound.cuh:134-135) folds the hull with PxVec3::minimum/maximum
// = componentwise PxMin/PxMax = selectMin/selectMax (a < b ? a : b / a > b ? a : b),
// so on a +-0 tie the NEW vertex's zero survives; that is what Vec3::minimum/
// maximum of the mirror do.  The pre-P9.86 port used GLSL FMin/FMax, which
// llvmpipe lowers under SignedZeroInfNanPreserve to bcsel(feq(a,b), imin/imax
// of the bit patterns, fmin/fmax): a +-0 tie gives -0 (min) / +0 (max) in
// either operand order.  ubLpFMin/ubLpFMax model that (valid for non-NaN data).
static float ubLpFMin(float a, float b)
{
    if (a == 0.f && b == 0.f)
        return (std::signbit(a) || std::signbit(b)) ? -0.0f : 0.0f;
    return a < b ? a : b;
}
static float ubLpFMax(float a, float b)
{
    if (a == 0.f && b == 0.f)
        return (std::signbit(a) && std::signbit(b)) ? -0.0f : 0.0f;
    return a > b ? a : b;
}

// What the reverted port (FMin/FMax fold) would store for one convex hull.
struct UbFMinMaxTwin
{
    bool valid = false; // the GPU-compatible convex hull path ran
    Float3 minimum{}, maximum{};
};

// blobDeviceBase: under the engine ABI Shape::hullOrMeshOffset carries the
// RAW DEVICE ADDRESS of the cooked hull (exactly what CUDA stores in
// PxgShape::hullOrMeshPtr); the reference rebases it onto the host copy of
// the blob.  The bound-buffer setRigidDynamicGlobalPose path keeps it as a
// plain byte offset (blobDeviceBase = 0).
// p89OldOrder: evaluate the mesh-scale matrix and the transformFast center with the
// pre-P9.89 port's operation trees (meshScaleToMat33 / transformFastBounds above).
void refUpdateBounds(const ShapeSim& shapeSim, const std::vector<Shape>& shapes,
                     const std::vector<uint32_t>& geometryBlob, const TransformV& pose, Bounds3& out,
                     uint64_t blobDeviceBase = 0, UbFMinMaxTwin* twin = nullptr, bool p89OldOrder = false)
{
    const Vec3 localMin(shapeSim.localBoundsMin);
    const Vec3 localMax(shapeSim.localBoundsMax);

    switch (shapeSim.shapeType)
    {
    case eSPHERE:
    {
        out.minimum = (pose.p + localMin).toFloat3();
        out.maximum = (pose.p + localMax).toFloat3();
        break;
    }
    case eCAPSULE:
    {
        const float radius = localMax.y;
        const float halfHeight = localMax.x - radius;
        const Vec3 d = pose.q.getBasisVector0();
        Vec3 extents;
        extents.x = std::fabs(d.x) * halfHeight + radius;
        extents.y = std::fabs(d.y) * halfHeight + radius;
        extents.z = std::fabs(d.z) * halfHeight + radius;
        out.minimum = (pose.p - extents).toFloat3();
        out.maximum = (pose.p + extents).toFloat3();
        break;
    }
    case eBOX:
    {
        const Vec3 halfExtents = localMax;
        const Vec3 extents = basisExtent(Mat33(pose.q), halfExtents);
        out.minimum = (pose.p - extents).toFloat3();
        out.maximum = (pose.p + extents).toFloat3();
        break;
    }
    case eCONVEXMESH:
    {
        const uint32_t hullIndex = shapeSim.hullDataIndex;
        if (hullIndex != 0xFFFFFFFFu)
        {
            const Shape& shape = shapes[hullIndex];
            Mat33 rot(pose.q);
            const Vec3 scale(shape.scale);
            const QuatV scaleRot(shape.scaleRotation);
            // P9.93: PxMeshScale::isIdentity() -- the scale only (PxMeshScale.h:104-107)
            const bool identity = scale.x == 1.f && scale.y == 1.f && scale.z == 1.f;
            if (!identity)
                rot = rot * meshScaleToMat33(scale, scaleRot, p89OldOrder);

            const uint32_t blobBase = uint32_t((shape.hullOrMeshOffset - blobDeviceBase) / 4);
            const uint32_t tmpX = geometryBlob[blobBase + 4];
            const uint32_t nbHullVertices = (tmpX >> 8) & 0xFFu;

            Vec3 minV(3.402823466e+38f);
            Vec3 maxV(-3.402823466e+38f);
            Vec3 minT(3.402823466e+38f);  // P9.86c twin (FMin/FMax fold)
            Vec3 maxT(-3.402823466e+38f);
            const uint32_t verticesBase = blobBase + 12;
            for (uint32_t i = 0; i < nbHullVertices; ++i)
            {
                float vx, vy, vz;
                std::memcpy(&vx, &geometryBlob[verticesBase + i * 4 + 0], 4);
                std::memcpy(&vy, &geometryBlob[verticesBase + i * 4 + 1], 4);
                std::memcpy(&vz, &geometryBlob[verticesBase + i * 4 + 2], 4);
                const Vec3 vertexV = rot.transform(Vec3(vx, vy, vz));
                minV = minV.minimum(vertexV);
                maxV = maxV.maximum(vertexV);
                minT = Vec3(ubLpFMin(minT.x, vertexV.x), ubLpFMin(minT.y, vertexV.y), ubLpFMin(minT.z, vertexV.z));
                maxT = Vec3(ubLpFMax(maxT.x, vertexV.x), ubLpFMax(maxT.y, vertexV.y), ubLpFMax(maxT.z, vertexV.z));
            }
            out.minimum = (minV + pose.p).toFloat3();
            out.maximum = (maxV + pose.p).toFloat3();
            if (twin)
            {
                twin->valid = true;
                twin->minimum = (minT + pose.p).toFloat3();
                twin->maximum = (maxT + pose.p).toFloat3();
            }
        }
        else
        {
            transformFastBounds(pose, localMin, localMax, out, p89OldOrder);
        }
        break;
    }
    default:
        transformFastBounds(pose, localMin, localMax, out, p89OldOrder);
        break;
    }
}

struct RefOutputs
{
    std::vector<CachedTransform> transformCache;
    std::vector<Bounds3> bounds;
    std::vector<uint32_t> frozen, unfrozen, updated, active, deactivate;
};

// Transcription of updateTransformCacheAndBoundArrayLaunch (both branches).
// artLinkB2W / artLinkB2A: per-articulation link transform arrays, indexed
// [artId * maxLinks + linkId].  twins (optional): per-shape FMin/FMax twin
// bounds of the convex hull path (P9.86c).
void referenceUpdate(uint32_t numShapes, const std::vector<ShapeSim>& shapeSims,
                     const std::vector<Shape>& shapes, const std::vector<uint32_t>& geometryBlob,
                     const std::vector<BodySim>& bodySims, const std::vector<uint32_t>& bodyDataIndices,
                     const std::vector<SolverBodySleepData>& sleepData,
                     const std::vector<SolverBodySleepData>& artiSleepData,
                     const std::vector<TransformV>& artLinkB2W, const std::vector<TransformV>& artLinkB2A,
                     uint32_t maxLinks, uint32_t maxNodes, uint64_t blobDeviceBase, RefOutputs& out,
                     std::vector<UbFMinMaxTwin>* twins = nullptr, bool p89OldOrder = false)
{
    if (twins)
        twins->assign(numShapes, UbFMinMaxTwin{});
    out.transformCache.assign(numShapes, CachedTransform{});
    out.bounds.assign(numShapes, Bounds3{});
    out.frozen.assign(numShapes, 0);
    out.unfrozen.assign(numShapes, 0);
    out.updated.assign(numShapes, 0);
    out.active.assign(maxNodes, 0);
    out.deactivate.assign(maxNodes, 0);

    for (uint32_t i = 0; i < numShapes; ++i)
    {
        const ShapeSim& shapeSim = shapeSims[i];
        const uint32_t bodySimIndex = uint32_t(shapeSim.bodySimIndex & 0xFFFFFFFFu);
        const bool isArticulation = (shapeSim.bodySimIndex >> 32) & 1;

        if (bodySimIndex == PX_INVALID_NODE)
            continue;

        const BodySim& bodySim = bodySims[bodySimIndex];
        const uint32_t shapeFlags = shapeSim.shapeFlags;
        const bool isBP = (shapeFlags & (eSIMULATION_SHAPE | eTRIGGER_SHAPE)) != 0;
        const bool isBPOrSq = (shapeFlags & (eSIMULATION_SHAPE | eTRIGGER_SHAPE | eSCENE_QUERY_SHAPE)) != 0;

        if (!isArticulation)
        {
            const uint32_t activeNodeIndex = bodyDataIndices[bodySimIndex];
            if (activeNodeIndex == 0xFFFFFFFFu)
                continue;

            const uint32_t internalFlags = sleepData[activeNodeIndex].internalFlags;
            const TransformV body2World(QuatV(bodySim.body2World.q),
                                        Vec3(bodySim.body2World.p.x, bodySim.body2World.p.y, bodySim.body2World.p.z));

            if ((internalFlags & eFREEZE_THIS_FRAME) && (internalFlags & eFROZEN))
            {
                out.frozen[i] = 1;
                out.transformCache[i].flags = 1; // PxsTransformFlag::eFROZEN
            }
            else if (internalFlags & eUNFREEZE_THIS_FRAME)
            {
                out.unfrozen[i] = 1;
            }

            if (!(internalFlags & eFROZEN) || (internalFlags & eFREEZE_THIS_FRAME))
            {
                if (isBP)
                    out.updated[i] = 1;

                const TransformV body2Actor(QuatV(bodySim.body2Actor_maxImpulseW.q),
                                            Vec3(bodySim.body2Actor_maxImpulseW.p.x, bodySim.body2Actor_maxImpulseW.p.y,
                                                 bodySim.body2Actor_maxImpulseW.p.z));
                const TransformV absPos = getAbsPose(body2World, TransformV(shapeSim.transform), body2Actor);

                out.transformCache[i].transform = absPos.toTransform28();
                out.transformCache[i].flags = 0;
                if (isBPOrSq)
                    refUpdateBounds(shapeSim, shapes, geometryBlob, absPos, out.bounds[i], blobDeviceBase,
                                    twins ? &(*twins)[i] : nullptr, p89OldOrder);
            }

            if (internalFlags & eACTIVATE_THIS_FRAME)
                out.active[bodySimIndex] = 1;
            else if (internalFlags & eDEACTIVATE_THIS_FRAME)
                out.deactivate[bodySimIndex] = 1;
        }
        else
        {
            // articulation branch: no sleep/freeze gating, link poses come
            // from the articulation's link transform arrays
            const uint32_t articulationId = bodySim.articulationRemapId;
            const uint32_t internalFlags = artiSleepData[articulationId].internalFlags;
            const uint32_t linkId = uint32_t(shapeSim.bodySimIndex >> 32) >> 1;

            const TransformV body2World = artLinkB2W[articulationId * maxLinks + linkId];

            if (isBP)
                out.updated[i] = 1;

            const TransformV body2Actor = artLinkB2A[articulationId * maxLinks + linkId];

            const TransformV absPos = getAbsPose(body2World, TransformV(shapeSim.transform), body2Actor);

            out.transformCache[i].transform = absPos.toTransform28();
            out.transformCache[i].flags = 0;
            if (isBPOrSq)
                refUpdateBounds(shapeSim, shapes, geometryBlob, absPos, out.bounds[i], blobDeviceBase,
                                twins ? &(*twins)[i] : nullptr, p89OldOrder);

            if (internalFlags & eACTIVATE_THIS_FRAME)
                out.active[bodySimIndex] = 1;
            else if (internalFlags & eDEACTIVATE_THIS_FRAME)
                out.deactivate[bodySimIndex] = 1;
        }
    }
}

// P9.89: which updateBounds() branch a shape takes (the paths the operation-order
// coverage below is counted over).
enum UbP89Path
{
    eUB_SPHERE, eUB_CAPSULE, eUB_BOX, eUB_HULL_IDENTITY, eUB_HULL_SCALED, eUB_FAST_CONVEX, eUB_FAST_OTHER,
    eUB_NB_PATHS
};
static const char* const kUbP89PathNames[eUB_NB_PATHS] = { "sphere", "capsule", "box", "hull(identity)",
                                                          "hull(scaled)", "fast(convex)", "fast(other)" };
static uint32_t ubP89Path(const ShapeSim& s, const std::vector<Shape>& shapes)
{
    switch (s.shapeType)
    {
    case eSPHERE: return eUB_SPHERE;
    case eCAPSULE: return eUB_CAPSULE;
    case eBOX: return eUB_BOX;
    case eCONVEXMESH:
    {
        if (s.hullDataIndex == 0xFFFFFFFFu)
            return eUB_FAST_CONVEX;
        const Shape& h = shapes[s.hullDataIndex];
        const bool identity = h.scale.x == 1.f && h.scale.y == 1.f && h.scale.z == 1.f;   // P9.93: scale only
        return identity ? eUB_HULL_IDENTITY : eUB_HULL_SCALED;
    }
    default: return eUB_FAST_OTHER;
    }
}

// ---------------------------------------------------------------------------
// test
// ---------------------------------------------------------------------------

QuatV randomQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    QuatV q(dist(rng), dist(rng), dist(rng), dist(rng));
    return q.getNormalized();
}

PXG_TEST(updateBounds_engineAbi)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-10.f, 10.f);

    const uint32_t numRandomShapes = 600;  // last 40 are articulation shapes
    const uint32_t numRigidShapes = numRandomShapes - 40;
    // P9.86c: two directed convex shapes appended after the random fleet
    // (600 = the hull whose x minimum is a +-0 tie, 601 = the x maximum tie)
    const uint32_t numDirShapes = 2;
    // P9.89: an order-sensitive fleet appended after the directed shapes (602.. =
    // scaled hull / fast-path convex / fast-path types 6..8, round-robin), drawn from
    // its own rng so the random fleet and the P9.86c shapes are unchanged
    const uint32_t numP89Shapes = 192;
    const uint32_t numP89Hulls = 2;
    const uint32_t numShapes = numRandomShapes + numDirShapes + numP89Shapes;
    const uint32_t maxNodes = 256;    // nodes [firstArtNode, firstArtNode+12) are articulation links
    const uint32_t numHulls = 8;      // random hulls (rollGeometry draws from these)
    const uint32_t numAllHulls = numHulls + numDirShapes + numP89Hulls;
    const uint32_t firstP89Hull = numHulls + numDirShapes;
    const uint32_t nbArts = 3;
    const uint32_t maxLinks = 4;
    const uint32_t firstArtNode = 240;
    const uint32_t dirNode = 252;     // unused by the fleet (rigid [0,240), links [240,252))
    const uint32_t p89Node = 253;     // P9.89 fleet body (also unused by the random fleet)
    std::mt19937 rng89(0x589u);

    // ---- convex geometry blob: numHulls hulls, header (48 bytes) + up to 64
    // float4 vertices; every hull base is 16-byte aligned (multiple-of-4 uints)
    std::vector<uint32_t> geometryBlob;
    std::vector<uint64_t> hullByteOffsets(numAllHulls);
    for (uint32_t h = 0; h < numHulls; ++h)
    {
        const uint32_t base = uint32_t(geometryBlob.size());
        hullByteOffsets[h] = uint64_t(base) * 4; // byte offset within the blob
        const uint32_t nbVerts = 4 + rng() % 60;

        geometryBlob.resize(base + 12 + nbVerts * 4, 0u);
        // uint4 at byte 16: x packs the vertex count in bits [8,16)
        geometryBlob[base + 4] = (nbVerts & 0xFFu) << 8;
        for (uint32_t v = 0; v < nbVerts; ++v)
        {
            const float vx = sym(rng), vy = sym(rng), vz = sym(rng);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 0], &vx, 4);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 1], &vy, 4);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 2], &vz, 4);
        }
    }

    // ---- P9.86c directed hulls (no rng: the random fleet is unchanged).
    // Two unit boxes with a face on the local z = 0 plane, every coordinate an
    // exact +-1 or +0.  Under the directed pose (dirNode below) rot.col0.x =
    // rot.col1.x = +0 and rot.col2.x = -1, so vertexV.x = ((+0)vx + (+0)vy) + (-1)vz:
    // on the z = 0 face it is -0 for (-1,-1,0) only and +0 for the other three.
    //  hull A (z in {-1,0}): the x MINIMUM is that zero; (-1,-1,0) comes FIRST, so
    //    the CUDA's selectMin keeps the later +0 ties (minX = +0) while an FMin
    //    fold gives -0.
    //  hull B (z in {0,1}): the x MAXIMUM is that zero; (-1,-1,0) comes LAST, so
    //    selectMax keeps -0 (maxX = -0) while an FMax fold gives +0.
    // pose.p.x = -0 carries the tie's sign into the stored bound (x + -0 = x).
    static const float dirHullVerts[numDirShapes][8][3] = {
        { { -1, -1, 0 }, { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 },
          { -1, -1, -1 }, { 1, -1, -1 }, { 1, 1, -1 }, { -1, 1, -1 } },
        { { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 }, { -1, -1, 0 },
          { -1, -1, 1 }, { 1, -1, 1 }, { 1, 1, 1 }, { -1, 1, 1 } },
    };
    for (uint32_t k = 0; k < numDirShapes; ++k)
    {
        const uint32_t base = uint32_t(geometryBlob.size()); // multiple of 4 uints (16-byte aligned)
        hullByteOffsets[numHulls + k] = uint64_t(base) * 4;
        geometryBlob.resize(base + 12 + 8 * 4, 0u);
        geometryBlob[base + 4] = 8u << 8;
        for (uint32_t v = 0; v < 8; ++v)
            std::memcpy(&geometryBlob[base + 12 + v * 4], dirHullVerts[k][v], 12);
    }

    // ---- P9.89 hulls: 64 distinct non-round vertices each (rng89), in +-10
    std::uniform_real_distribution<float> unit89(0.f, 1.f);
    std::uniform_real_distribution<float> sym89(-10.f, 10.f);
    for (uint32_t k = 0; k < numP89Hulls; ++k)
    {
        const uint32_t base = uint32_t(geometryBlob.size());
        hullByteOffsets[firstP89Hull + k] = uint64_t(base) * 4;
        const uint32_t nbVerts = 64;
        geometryBlob.resize(base + 12 + nbVerts * 4, 0u);
        geometryBlob[base + 4] = (nbVerts & 0xFFu) << 8;
        for (uint32_t v = 0; v < nbVerts; ++v)
        {
            const float vx = sym89(rng89), vy = sym89(rng89), vz = sym89(rng89);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 0], &vx, 4);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 1], &vy, 4);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 2], &vz, 4);
        }
    }

    // ---- shapes (PxgShape pool: convex hull entries; hullOrMeshOffset is
    // patched to a live device address once the blob is uploaded)
    std::vector<Shape> shapes(numAllHulls);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(Shape));
    for (uint32_t h = 0; h < numHulls; ++h)
    {
        Shape& s = shapes[h];
        if (unit(rng) < 0.5f)
        {
            s.scale = Float3{ 1.f, 1.f, 1.f };
            s.scaleRotation = Quat{ 0, 0, 0, 1 };
        }
        else
        {
            s.scale = Float3{ 0.5f + unit(rng), 0.5f + unit(rng), 0.5f + unit(rng) };
            const QuatV q = randomQuat(rng);
            s.scaleRotation = q.toQuat();
        }
        s.type = eCONVEXMESH;
    }
    for (uint32_t h = numHulls; h < firstP89Hull; ++h)
    {
        shapes[h].scale = Float3{ 1.f, 1.f, 1.f }; // identity mesh scale: no scale multiply
        shapes[h].scaleRotation = Quat{ 0, 0, 0, 1 };
        shapes[h].type = eCONVEXMESH;
    }
    // P9.89 hulls: strongly non-uniform, non-round mesh scales under random scale
    // rotations -- every off-diagonal element of PxMeshScale::toMat33 is live
    for (uint32_t h = firstP89Hull; h < numAllHulls; ++h)
    {
        shapes[h].scale = Float3{ 0.3f + 2.6f * unit89(rng89), 0.3f + 2.6f * unit89(rng89), 0.3f + 2.6f * unit89(rng89) };
        shapes[h].scaleRotation = randomQuat(rng89).toQuat();
        shapes[h].type = eCONVEXMESH;
    }

    // ---- bodies + sleep data
    std::vector<BodySim> bodySims(maxNodes);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(BodySim));
    std::vector<uint32_t> bodyDataIndices(maxNodes);
    std::vector<SolverBodySleepData> sleepData(maxNodes);
    for (uint32_t n = 0; n < maxNodes; ++n)
    {
        BodySim& b = bodySims[n];
        b.body2World.q = randomQuat(rng).toFloat4();
        b.body2World.p = Float4{ sym(rng), sym(rng), sym(rng), 0 };
        b.body2Actor_maxImpulseW.q = randomQuat(rng).toFloat4();
        b.body2Actor_maxImpulseW.p = Float4{ sym(rng) * 0.1f, sym(rng) * 0.1f, sym(rng) * 0.1f, 0 };

        // ~15% inactive nodes
        bodyDataIndices[n] = unit(rng) < 0.15f ? 0xFFFFFFFFu : n;

        uint32_t flags = 0;
        const float r = unit(rng);
        if (r < 0.15f)
            flags |= eFROZEN | eFREEZE_THIS_FRAME;
        else if (r < 0.3f)
            flags |= eFROZEN; // stays frozen -> no update
        else if (r < 0.45f)
            flags |= eUNFREEZE_THIS_FRAME;
        if (unit(rng) < 0.2f)
            flags |= eACTIVATE_THIS_FRAME;
        else if (unit(rng) < 0.2f)
            flags |= eDEACTIVATE_THIS_FRAME;
        sleepData[n].internalFlags = flags;
        sleepData[n].wakeCounter = unit(rng);
    }

    // articulation link nodes: node firstArtNode + artId*maxLinks + linkId
    for (uint32_t a = 0; a < nbArts; ++a)
        for (uint32_t l = 0; l < maxLinks; ++l)
            bodySims[firstArtNode + a * maxLinks + l].articulationRemapId = a;

    // per-articulation sleep data: activate / deactivate / freeze flags (the
    // articulation branch must honor only activate/deactivate)
    std::vector<SolverBodySleepData> artiSleepData(nbArts);
    artiSleepData[0] = SolverBodySleepData{ 0.5f, eFROZEN | eFREEZE_THIS_FRAME };
    artiSleepData[1] = SolverBodySleepData{ 0.25f, eACTIVATE_THIS_FRAME };
    artiSleepData[2] = SolverBodySleepData{ 0.125f, eDEACTIVATE_THIS_FRAME };

    // per-articulation link transforms
    // P9.86c directed body: an exact unit quaternion (120 deg axis permutation,
    // columns (+0,1,+0) / (+0,+0,-1) / (-1,+0,+0)), world position x = -0 (a
    // negated zero), identity COM frame; active, not frozen.  getAbsPose gives
    // pose.q = body2World.q and pose.p = rotate(+0 vec) + p = (-0,+0,+0) + (-0,5,5)
    // = (-0,5,5) in the CUDA, the port and the mirror alike.
    bodySims[dirNode].body2World.q = Float4{ -0.5f, -0.5f, 0.5f, 0.5f };
    bodySims[dirNode].body2World.p = Float4{ -0.0f, 5.f, 5.f, 0.f };
    bodySims[dirNode].body2Actor_maxImpulseW.q = Float4{ 0.f, 0.f, 0.f, 1.f };
    bodySims[dirNode].body2Actor_maxImpulseW.p = Float4{ 0.f, 0.f, 0.f, 0.f };
    bodyDataIndices[dirNode] = dirNode;
    sleepData[dirNode].internalFlags = 0;
    // P9.89 body: random rotations, small translations (so |pose.p| stays well below
    // the far-off local-bounds centers of the fast-path shapes and the center's
    // rotation bits survive the + p and +- extents roundings); active, not frozen.
    bodySims[p89Node].body2World.q = randomQuat(rng89).toFloat4();
    bodySims[p89Node].body2World.p = Float4{ 0.03f * sym89(rng89), 0.03f * sym89(rng89), 0.03f * sym89(rng89), 0.f };
    bodySims[p89Node].body2Actor_maxImpulseW.q = randomQuat(rng89).toFloat4();
    bodySims[p89Node].body2Actor_maxImpulseW.p =
        Float4{ 0.01f * sym89(rng89), 0.01f * sym89(rng89), 0.01f * sym89(rng89), 0.f };
    bodyDataIndices[p89Node] = p89Node;
    sleepData[p89Node].internalFlags = 0;

    std::vector<TransformV> artLinkB2W(nbArts * maxLinks);
    std::vector<TransformV> artLinkB2A(nbArts * maxLinks);
    for (uint32_t k = 0; k < nbArts * maxLinks; ++k)
    {
        artLinkB2W[k] = TransformV(randomQuat(rng), Vec3(sym(rng), sym(rng), sym(rng)));
        artLinkB2A[k] = TransformV(randomQuat(rng), Vec3(sym(rng) * 0.1f, sym(rng) * 0.1f, sym(rng) * 0.1f));
    }

    // ---- shape sims: geometry generator shared by rigid + articulation shapes
    auto rollGeometry = [&](ShapeSim& s) {
        const float typeRoll = unit(rng);
        if (typeRoll < 0.2f)
        {
            s.shapeType = eSPHERE;
            const float r = unit(rng) * 2.f;
            s.localBoundsMin = Float3{ -r, -r, -r };
            s.localBoundsMax = Float3{ r, r, r };
        }
        else if (typeRoll < 0.4f)
        {
            s.shapeType = eCAPSULE;
            const float r = 0.2f + unit(rng);
            const float hh = 0.2f + unit(rng) * 2.f;
            s.localBoundsMin = Float3{ -(hh + r), -r, -r };
            s.localBoundsMax = Float3{ hh + r, r, r };
        }
        else if (typeRoll < 0.6f)
        {
            s.shapeType = eBOX;
            const Float3 he{ 0.2f + unit(rng), 0.2f + unit(rng), 0.2f + unit(rng) };
            s.localBoundsMin = Float3{ -he.x, -he.y, -he.z };
            s.localBoundsMax = he;
        }
        else if (typeRoll < 0.85f)
        {
            s.shapeType = eCONVEXMESH;
            // ~20% GPU-incompatible hulls take the transformFast fallback
            s.hullDataIndex = unit(rng) < 0.2f ? 0xFFFFFFFFu : rng() % numHulls;
            s.localBoundsMin = Float3{ -1.f - unit(rng), -1.f - unit(rng), -1.f - unit(rng) };
            s.localBoundsMax = Float3{ 1.f + unit(rng), 1.f + unit(rng), 1.f + unit(rng) };
        }
        else
        {
            s.shapeType = uint16_t(6 + rng() % 3); // trimesh/heightfield-ish -> fallback path
            s.localBoundsMin = Float3{ -1.f - unit(rng), -2.f, -1.f };
            s.localBoundsMax = Float3{ 1.f + unit(rng), 2.f, 1.f };
        }
    };

    std::vector<ShapeSim> shapeSims(numShapes);
    std::memset(shapeSims.data(), 0, shapeSims.size() * sizeof(ShapeSim));
    for (uint32_t i = 0; i < numRandomShapes; ++i)
    {
        ShapeSim& s = shapeSims[i];
        const QuatV q = randomQuat(rng);
        s.transform.q = q.toQuat();
        s.transform.p = Float3{ sym(rng) * 0.2f, sym(rng) * 0.2f, sym(rng) * 0.2f };
        s.shapeFlags = uint16_t((unit(rng) < 0.8f ? eSIMULATION_SHAPE : 0) |
                                (unit(rng) < 0.5f ? eSCENE_QUERY_SHAPE : 0) |
                                (unit(rng) < 0.1f ? eTRIGGER_SHAPE : 0));
        if (i < numRigidShapes)
        {
            // 5% static (skipped by the kernel)
            s.bodySimIndex = unit(rng) < 0.05f ? makeNodeIndex(PX_INVALID_NODE)
                                               : makeNodeIndex(rng() % firstArtNode);
        }
        else
        {
            // articulation shapes across all arts/links
            const uint32_t a = rng() % nbArts;
            const uint32_t l = rng() % maxLinks;
            s.bodySimIndex = makeNodeIndex(firstArtNode + a * maxLinks + l, l, true);
        }
        rollGeometry(s);
    }
    // P9.86c directed shapes: identity shape2Actor on dirNode, hull A / hull B
    for (uint32_t k = 0; k < numDirShapes; ++k)
    {
        ShapeSim& s = shapeSims[numRandomShapes + k];
        s.transform.q = Quat{ 0.f, 0.f, 0.f, 1.f };
        s.transform.p = Float3{ 0.f, 0.f, 0.f };
        s.localBoundsMin = Float3{ -1.f, -1.f, -1.f };
        s.localBoundsMax = Float3{ 1.f, 1.f, 1.f };
        s.bodySimIndex = makeNodeIndex(dirNode);
        s.hullDataIndex = numHulls + k;
        s.shapeFlags = uint16_t(eSIMULATION_SHAPE);
        s.shapeType = eCONVEXMESH;
    }
    // P9.89 shapes on p89Node, round-robin over the three order-sensitive paths:
    //  k%3 == 0: a P9.89 scaled hull (PxMeshScale::toMat33 feeds every vertex);
    //  k%3 == 1: a GPU-incompatible convex (hullIndex invalid -> transformFast);
    //  k%3 == 2: a mesh/heightfield-type shape (6..8 -> transformFast).
    // The fast-path local bounds sit far from the shape origin (center components
    // in +-(4..20)) with small extents, so q.rotate(center) and the PxMat33(q)
    // product round apart in most components and the difference survives + p.
    for (uint32_t k = 0; k < numP89Shapes; ++k)
    {
        ShapeSim& s = shapeSims[numRandomShapes + numDirShapes + k];
        s.transform.q = randomQuat(rng89).toQuat();
        s.transform.p = Float3{ 0.02f * sym89(rng89), 0.02f * sym89(rng89), 0.02f * sym89(rng89) };
        auto farCoord = [&]() {
            const float m = 4.f + 16.f * unit89(rng89);
            return unit89(rng89) < 0.5f ? -m : m;
        };
        const Float3 c{ farCoord(), farCoord(), farCoord() };
        const Float3 e{ 0.05f + 0.45f * unit89(rng89), 0.05f + 0.45f * unit89(rng89), 0.05f + 0.45f * unit89(rng89) };
        s.localBoundsMin = Float3{ c.x - e.x, c.y - e.y, c.z - e.z };
        s.localBoundsMax = Float3{ c.x + e.x, c.y + e.y, c.z + e.z };
        s.bodySimIndex = makeNodeIndex(p89Node);
        s.shapeFlags = uint16_t(eSIMULATION_SHAPE);
        switch (k % 3)
        {
        case 0:
            s.shapeType = eCONVEXMESH;
            s.hullDataIndex = firstP89Hull + (k / 3) % numP89Hulls;
            break;
        case 1:
            s.shapeType = eCONVEXMESH;
            s.hullDataIndex = 0xFFFFFFFFu;
            break;
        default:
            s.shapeType = uint16_t(6 + (k / 3) % 3);
            s.hullDataIndex = 0xFFFFFFFFu;
            break;
        }
    }

    // ---- GPU (ENGINE launch ABI through PxgCudaShim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "updateBounds.spv", "updateTransformCacheAndBoundArrayLaunch", 8u, 0 }, // { u64 }
    };
    PxgCudaShim shim(dev, manifest, 1, &updateBoundsAbiLoader);

    PxgDevicePtr dShapeSims = 0, dShapes = 0, dBodySims = 0, dBodyIdx = 0, dSleep = 0, dArtiSleep = 0;
    PxgDevicePtr dCache = 0, dBounds = 0, dFrozen = 0, dUnfrozen = 0, dUpdated = 0, dActive = 0, dDeactivate = 0;
    PxgDevicePtr dBlob = 0, dArtPool = 0, dArtB2W = 0, dArtB2A = 0, dDesc = 0;
    shim.memAlloc(&dShapeSims, shapeSims.size() * sizeof(ShapeSim));
    shim.memAlloc(&dShapes, shapes.size() * sizeof(Shape));
    shim.memAlloc(&dBodySims, bodySims.size() * sizeof(BodySim));
    shim.memAlloc(&dBodyIdx, bodyDataIndices.size() * 4);
    shim.memAlloc(&dSleep, sleepData.size() * sizeof(SolverBodySleepData));
    shim.memAlloc(&dArtiSleep, artiSleepData.size() * sizeof(SolverBodySleepData));
    shim.memAlloc(&dCache, numShapes * sizeof(CachedTransform));
    shim.memAlloc(&dBounds, numShapes * sizeof(Bounds3));
    shim.memAlloc(&dFrozen, numShapes * 4);
    shim.memAlloc(&dUnfrozen, numShapes * 4);
    shim.memAlloc(&dUpdated, numShapes * 4);
    shim.memAlloc(&dActive, maxNodes * 4);
    shim.memAlloc(&dDeactivate, maxNodes * 4);
    shim.memAlloc(&dBlob, geometryBlob.size() * 4);
    shim.memAlloc(&dArtPool, nbArts * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dArtB2W, nbArts * maxLinks * sizeof(Transform28));
    shim.memAlloc(&dArtB2A, nbArts * maxLinks * sizeof(Transform28));
    shim.memAlloc(&dDesc, sizeof(physx::PxgSimulationCoreDesc));
    if (!dShapeSims || !dShapes || !dBodySims || !dBodyIdx || !dSleep || !dArtiSleep || !dCache ||
        !dBounds || !dFrozen || !dUnfrozen || !dUpdated || !dActive || !dDeactivate || !dBlob ||
        !dArtPool || !dArtB2W || !dArtB2A || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // hullOrMeshPtr now that the blob has a device address: LIVE device
    // pointer to the cooked hull (the CUDA meaning of PxgShape::hullOrMeshPtr)
    for (uint32_t h = 0; h < numAllHulls; ++h)
        shapes[h].hullOrMeshOffset = dBlob + hullByteOffsets[h];

    // raw PxgArticulation pool: 0xEE background, per-articulation link
    // transform array pointers planted at the engine's field offsets
    std::vector<uint8_t> artPool(nbArts * sizeof(physx::PxgArticulation), 0xEE);
    for (uint32_t a = 0; a < nbArts; ++a)
    {
        const uint64_t b2w = dArtB2W + uint64_t(a) * maxLinks * sizeof(Transform28);
        const uint64_t b2a = dArtB2A + uint64_t(a) * maxLinks * sizeof(Transform28);
        uint8_t* base = artPool.data() + a * sizeof(physx::PxgArticulation);
        std::memcpy(base + offsetof(physx::PxgArticulation, linkBody2Worlds), &b2w, 8);
        std::memcpy(base + offsetof(physx::PxgArticulation, linkBody2Actors), &b2a, 8);
    }

    std::vector<Transform28> artB2WUpload(nbArts * maxLinks), artB2AUpload(nbArts * maxLinks);
    for (uint32_t k = 0; k < nbArts * maxLinks; ++k)
    {
        artB2WUpload[k] = artLinkB2W[k].toTransform28();
        artB2AUpload[k] = artLinkB2A[k].toTransform28();
    }

    shim.memcpyHtoD(dShapeSims, shapeSims.data(), shapeSims.size() * sizeof(ShapeSim));
    shim.memcpyHtoD(dShapes, shapes.data(), shapes.size() * sizeof(Shape));
    shim.memcpyHtoD(dBodySims, bodySims.data(), bodySims.size() * sizeof(BodySim));
    shim.memcpyHtoD(dBodyIdx, bodyDataIndices.data(), bodyDataIndices.size() * 4);
    shim.memcpyHtoD(dSleep, sleepData.data(), sleepData.size() * sizeof(SolverBodySleepData));
    shim.memcpyHtoD(dArtiSleep, artiSleepData.data(), artiSleepData.size() * sizeof(SolverBodySleepData));
    shim.memcpyHtoD(dBlob, geometryBlob.data(), geometryBlob.size() * 4);
    shim.memcpyHtoD(dArtPool, artPool.data(), artPool.size());
    shim.memcpyHtoD(dArtB2W, artB2WUpload.data(), artB2WUpload.size() * sizeof(Transform28));
    shim.memcpyHtoD(dArtB2A, artB2AUpload.data(), artB2AUpload.size() * sizeof(Transform28));

    // the REAL engine desc with live device addresses
    physx::PxgSimulationCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.mFrozen = reinterpret_cast<physx::PxU32*>(dFrozen);
    desc.mUnfrozen = reinterpret_cast<physx::PxU32*>(dUnfrozen);
    desc.mUpdated = reinterpret_cast<physx::PxU32*>(dUpdated);
    desc.mActivate = reinterpret_cast<physx::PxU32*>(dActive);
    desc.mDeactivate = reinterpret_cast<physx::PxU32*>(dDeactivate);
    desc.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(dBodySims);
    desc.mShapeSimsBufferDeviceData = reinterpret_cast<const physx::PxgShapeSim*>(dShapeSims);
    desc.mArticulationPool = reinterpret_cast<physx::PxgArticulation*>(dArtPool);
    desc.mArticulationSleepDataPool = reinterpret_cast<physx::PxgSolverBodySleepData*>(dArtiSleep);
    desc.mTransformCache = reinterpret_cast<physx::PxsCachedTransform*>(dCache);
    desc.mBounds = reinterpret_cast<physx::PxBounds3*>(dBounds);
    desc.mBodyDataIndices = reinterpret_cast<physx::PxU32*>(dBodyIdx);
    desc.mSleepData = reinterpret_cast<physx::PxgSolverBodySleepData*>(dSleep);
    desc.mShapes = reinterpret_cast<physx::PxgShape*>(dShapes);
    desc.mNbTotalShapes = numShapes;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // the CUDA argument list, packed by the bridge's rule ({8})
    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 8, "engine blob is 8 bytes (got %zu)", blob.bytes.size());

    // ---- reference (grid-independent) ----
    RefOutputs ref;
    std::vector<UbFMinMaxTwin> twins;
    referenceUpdate(numShapes, shapeSims, shapes, geometryBlob, bodySims, bodyDataIndices, sleepData,
                    artiSleepData, artLinkB2W, artLinkB2A, maxLinks, maxNodes, dBlob, ref, &twins);

    // ---- P9.86c coverage: the directed hulls must make the CUDA's selectMin/
    // selectMax fold and the FMin/FMax twin store different bounds -- exactly the
    // x minimum of shape 600 and the x maximum of shape 601, nothing else.
    {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
        uint32_t minDiffers = 0, maxDiffers = 0, otherDiffers = 0;
        for (uint32_t k = 0; k < numDirShapes; ++k)
        {
            const uint32_t i = numRandomShapes + k;
            PXG_CHECK_MSG(twins[i].valid, "P9.86c coverage: directed shape %u must take the convex hull path", i);
            const float rw[6] = { ref.bounds[i].minimum.x, ref.bounds[i].minimum.y, ref.bounds[i].minimum.z,
                                  ref.bounds[i].maximum.x, ref.bounds[i].maximum.y, ref.bounds[i].maximum.z };
            const float tw[6] = { twins[i].minimum.x, twins[i].minimum.y, twins[i].minimum.z,
                                  twins[i].maximum.x, twins[i].maximum.y, twins[i].maximum.z };
            for (int c = 0; c < 6; ++c)
            {
                if (bits(rw[c]) == bits(tw[c]))
                    continue;
                if (k == 0 && c == 0)
                    ++minDiffers;
                else if (k == 1 && c == 3)
                    ++maxDiffers;
                else
                    ++otherDiffers;
            }
        }
        PXG_CHECK_MSG(minDiffers == 1 && maxDiffers == 1 && otherDiffers == 0,
                      "P9.86c coverage: directed hulls must make PxMin/PxMax differ from FMin/FMax in "
                      "bounds[600].minX and bounds[601].maxX only (min %u max %u other %u)",
                      minDiffers, maxDiffers, otherDiffers);
        // the CUDA image (selectMin/selectMax keep the LAST tied vertex's zero) ...
        const uint32_t i0 = numRandomShapes, i1 = numRandomShapes + 1;
        PXG_CHECK_MSG(bits(ref.transformCache[i0].transform.p.x) == 0x80000000u &&
                          bits(ref.transformCache[i1].transform.p.x) == 0x80000000u,
                      "P9.86c: directed pose.p.x must be -0 (%08x %08x)", bits(ref.transformCache[i0].transform.p.x),
                      bits(ref.transformCache[i1].transform.p.x));
        PXG_CHECK_MSG(bits(ref.bounds[i0].minimum.x) == 0x00000000u && bits(ref.bounds[i1].maximum.x) == 0x80000000u,
                      "P9.86c: CUDA image minX(600)=+0 maxX(601)=-0 (got %08x %08x)", bits(ref.bounds[i0].minimum.x),
                      bits(ref.bounds[i1].maximum.x));
        // ... and the reverted port's image (FMin -> -0, FMax -> +0)
        PXG_CHECK_MSG(bits(twins[i0].minimum.x) == 0x80000000u && bits(twins[i1].maximum.x) == 0x00000000u,
                      "P9.86c: FMin/FMax twin image minX(600)=-0 maxX(601)=+0 (got %08x %08x)",
                      bits(twins[i0].minimum.x), bits(twins[i1].maximum.x));
    }

    // ---- P9.89 coverage: the same fleet through the pre-P9.89 port's operation
    // trees (rot^T * (diag(s) * rot) for PxMeshScale::toMat33, the PxMat33(q) product
    // for the transformFast center) must store different bits than the CUDA's
    // ((rot^T column-scaled) * rot, q.rotate(center) + p) on the scaled-hull and on
    // both transformFast paths, and nowhere else.  The random fleet carries the data
    // that makes the association visible: 0.5 + unit(rng) mesh scales under random
    // scale rotations, and local bounds whose centers are non-zero (convex fast path:
    // all three axes; types 6..8: x).  The GPU is compared bitwise below, so a port
    // that evaluates either site in the old tree fails there.
    RefOutputs refOld;
    referenceUpdate(numShapes, shapeSims, shapes, geometryBlob, bodySims, bodyDataIndices, sleepData,
                    artiSleepData, artLinkB2W, artLinkB2A, maxLinks, maxNodes, dBlob, refOld, nullptr, true);
    {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
        uint32_t pathShapes[eUB_NB_PATHS] = {}, pathDiff[eUB_NB_PATHS] = {};
        uint32_t cacheDiff = 0;
        for (uint32_t i = 0; i < numShapes; ++i)
        {
            const uint32_t path = ubP89Path(shapeSims[i], shapes);
            const float rw[6] = { ref.bounds[i].minimum.x, ref.bounds[i].minimum.y, ref.bounds[i].minimum.z,
                                  ref.bounds[i].maximum.x, ref.bounds[i].maximum.y, ref.bounds[i].maximum.z };
            const float ow[6] = { refOld.bounds[i].minimum.x, refOld.bounds[i].minimum.y, refOld.bounds[i].minimum.z,
                                  refOld.bounds[i].maximum.x, refOld.bounds[i].maximum.y, refOld.bounds[i].maximum.z };
            ++pathShapes[path];
            for (int c = 0; c < 6; ++c)
                pathDiff[path] += bits(rw[c]) != bits(ow[c]) ? 1u : 0u;
            cacheDiff += std::memcmp(&ref.transformCache[i], &refOld.transformCache[i], sizeof(CachedTransform)) != 0;
        }
        for (uint32_t p = 0; p < eUB_NB_PATHS; ++p)
            std::printf("[ info ] P9.89 %-14s %3u shapes, %3u bound words differ CUDA vs old port order\n",
                        kUbP89PathNames[p], pathShapes[p], pathDiff[p]);
        // measured: 143 / 360 / 373 (66 / 8 / 15 of them from the random fleet alone)
        PXG_CHECK_MSG(pathDiff[eUB_HULL_SCALED] >= 100 && pathDiff[eUB_FAST_CONVEX] >= 200 &&
                          pathDiff[eUB_FAST_OTHER] >= 200,
                      "P9.89 coverage: the old port order must change many bound words on every affected path "
                      "(scaled hull %u want >= 100, fast convex %u want >= 200, fast other %u want >= 200)",
                      pathDiff[eUB_HULL_SCALED], pathDiff[eUB_FAST_CONVEX], pathDiff[eUB_FAST_OTHER]);
        PXG_CHECK_MSG(pathDiff[eUB_SPHERE] == 0 && pathDiff[eUB_CAPSULE] == 0 && pathDiff[eUB_BOX] == 0 &&
                          pathDiff[eUB_HULL_IDENTITY] == 0 && cacheDiff == 0,
                      "P9.89 twin: only the scaled-hull and transformFast bounds may differ (sphere %u capsule %u "
                      "box %u identity hull %u cache %u)",
                      pathDiff[eUB_SPHERE], pathDiff[eUB_CAPSULE], pathDiff[eUB_BOX], pathDiff[eUB_HULL_IDENTITY],
                      cacheDiff);
    }

    // engine grid (256 x 256 = one pass over 794 shapes) and a 2-block grid
    // (512 threads -> the CUDA grid-stride wrap executes; threads 88/89 pick
    // up the directed shapes 600/601, threads 90.. the P9.89 fleet)
    const std::vector<CachedTransform> cacheInit(numShapes, CachedTransform{});
    const std::vector<float> boundsInit(numShapes * 6, 0.f);
    const std::vector<uint32_t> zerosShapes(numShapes, 0), zerosNodes(maxNodes, 0);

    for (uint32_t gridDim : { 256u, 2u })
    {
        shim.memcpyHtoD(dCache, cacheInit.data(), cacheInit.size() * sizeof(CachedTransform));
        shim.memcpyHtoD(dBounds, boundsInit.data(), boundsInit.size() * 4);
        shim.memcpyHtoD(dFrozen, zerosShapes.data(), zerosShapes.size() * 4);
        shim.memcpyHtoD(dUnfrozen, zerosShapes.data(), zerosShapes.size() * 4);
        shim.memcpyHtoD(dUpdated, zerosShapes.data(), zerosShapes.size() * 4);
        shim.memcpyHtoD(dActive, zerosNodes.data(), zerosNodes.size() * 4);
        shim.memcpyHtoD(dDeactivate, zerosNodes.data(), zerosNodes.size() * 4);

        PXG_CHECK_MSG(shim.launchKernel(0, gridDim, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                          PxgResult::Success,
                      "launch updateTransformCacheAndBoundArrayLaunch (grid %u)", gridDim);
        shim.streamSynchronize();

        // ---- compare ----
        uint32_t dirChecked = 0;
        uint32_t cacheBad = 0, boundsBad[eUB_NB_PATHS] = {};
        std::vector<CachedTransform> gpuCache(numShapes);
        std::vector<float> gpuBounds(numShapes * 6);
        std::vector<uint32_t> gpuFrozen(numShapes), gpuUnfrozen(numShapes), gpuUpdated(numShapes);
        std::vector<uint32_t> gpuActive(maxNodes), gpuDeactivate(maxNodes);
        shim.memcpyDtoH(gpuCache.data(), dCache, numShapes * sizeof(CachedTransform));
        shim.memcpyDtoH(gpuBounds.data(), dBounds, numShapes * 6 * 4);
        shim.memcpyDtoH(gpuFrozen.data(), dFrozen, numShapes * 4);
        shim.memcpyDtoH(gpuUnfrozen.data(), dUnfrozen, numShapes * 4);
        shim.memcpyDtoH(gpuUpdated.data(), dUpdated, numShapes * 4);
        shim.memcpyDtoH(gpuActive.data(), dActive, maxNodes * 4);
        shim.memcpyDtoH(gpuDeactivate.data(), dDeactivate, maxNodes * 4);

        for (uint32_t i = 0; i < numShapes; ++i)
        {
            PXG_CHECK_MSG(gpuFrozen[i] == ref.frozen[i], "grid %u frozen[%u]: gpu=%u ref=%u", gridDim, i,
                          gpuFrozen[i], ref.frozen[i]);
            PXG_CHECK_MSG(gpuUnfrozen[i] == ref.unfrozen[i], "grid %u unfrozen[%u]", gridDim, i);
            PXG_CHECK_MSG(gpuUpdated[i] == ref.updated[i], "grid %u updated[%u]", gridDim, i);
            PXG_CHECK_MSG(gpuCache[i].flags == ref.transformCache[i].flags,
                          "grid %u cache.flags[%u]: gpu=%u ref=%u", gridDim, i, gpuCache[i].flags,
                          ref.transformCache[i].flags);

            // P9.89: the whole fleet is compared BITWISE (it was almostEqual 1e-4, which
            // could not see an operation-order divergence): the transform cache and all six
            // bound words of every shape.  Mismatches are counted per updateBounds() path.
            const Transform28& gt = gpuCache[i].transform;
            const Transform28& rt = ref.transformCache[i].transform;
            if (std::memcmp(&gt, &rt, sizeof(Transform28)) != 0)
            {
                if (cacheBad < 8)
                    std::printf("MM grid %u cache[%u]: gpu=(%a,%a,%a,%a | %a,%a,%a) ref=(%a,%a,%a,%a | %a,%a,%a)\n",
                                gridDim, i, gt.q.x, gt.q.y, gt.q.z, gt.q.w, gt.p.x, gt.p.y, gt.p.z, rt.q.x, rt.q.y,
                                rt.q.z, rt.q.w, rt.p.x, rt.p.y, rt.p.z);
                ++cacheBad;
            }

            const float* gb = &gpuBounds[i * 6];
            const float rb[6] = { ref.bounds[i].minimum.x, ref.bounds[i].minimum.y, ref.bounds[i].minimum.z,
                                  ref.bounds[i].maximum.x, ref.bounds[i].maximum.y, ref.bounds[i].maximum.z };
            const uint32_t path = ubP89Path(shapeSims[i], shapes);
            for (int c = 0; c < 6; ++c)
            {
                uint32_t g, r;
                std::memcpy(&g, &gb[c], 4);
                std::memcpy(&r, &rb[c], 4);
                if (g == r)
                    continue;
                if (boundsBad[path] < 4)
                    std::printf("MM grid %u bounds[%u][%d] (%s): gpu=%08x (%a) ref=%08x (%a)\n", gridDim, i, c,
                                kUbP89PathNames[path], g, gb[c], r, rb[c]);
                ++boundsBad[path];
            }

            // P9.86c: the directed shapes are compared BITWISE (almostEqual
            // treats +0 and -0 as equal): all six bound words and the cache.
            if (i >= numRandomShapes && i < numRandomShapes + numDirShapes)
            {
                uint32_t gw[6], rw[6];
                std::memcpy(gw, gb, sizeof(gw));
                std::memcpy(rw, rb, sizeof(rw));
                for (int c = 0; c < 6; ++c)
                    PXG_CHECK_MSG(gw[c] == rw[c], "grid %u directed bounds[%u][%d]: gpu=%08x ref=%08x", gridDim, i,
                                  c, gw[c], rw[c]);
                PXG_CHECK_MSG(std::memcmp(&gt, &rt, sizeof(Transform28)) == 0,
                              "grid %u directed cache[%u] differs bitwise", gridDim, i);
                ++dirChecked;
            }
        }
        PXG_CHECK_MSG(dirChecked == numDirShapes, "P9.86c coverage: grid %u compared %u directed shapes bitwise",
                      gridDim, dirChecked);
        PXG_CHECK_MSG(cacheBad == 0, "grid %u: %u transform cache entries differ bitwise", gridDim, cacheBad);
        for (uint32_t p = 0; p < eUB_NB_PATHS; ++p)
            PXG_CHECK_MSG(boundsBad[p] == 0, "grid %u: %u %s bound words differ bitwise from the CUDA order",
                          gridDim, boundsBad[p], kUbP89PathNames[p]);

        for (uint32_t n = 0; n < maxNodes; ++n)
        {
            PXG_CHECK_MSG(gpuActive[n] == ref.active[n], "grid %u active[%u]", gridDim, n);
            PXG_CHECK_MSG(gpuDeactivate[n] == ref.deactivate[n], "grid %u deactivate[%u]", gridDim, n);
        }
    }

    for (PxgDevicePtr h : { dShapeSims, dShapes, dBodySims, dBodyIdx, dSleep, dArtiSleep, dCache, dBounds,
                            dFrozen, dUnfrozen, dUpdated, dActive, dDeactivate, dBlob, dArtPool, dArtB2W,
                            dArtB2A, dDesc })
        shim.memFree(h);
}

// ---------------------------------------------------------------------------
// setRigidDynamicGlobalPose (direct GPU API entry in updateBounds.slang)
// ---------------------------------------------------------------------------
PXG_TEST(setRigidDynamicGlobalPose)
{
    std::mt19937 rng(778);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-10.f, 10.f);

    const uint32_t numBodies = 40;
    const uint32_t nbUpdates = 25;

    // bodies
    std::vector<BodySim> bodySims(numBodies);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(BodySim));
    for (BodySim& b : bodySims)
    {
        b.body2World.q = randomQuat(rng).toFloat4();
        b.body2World.p = Float4{ sym(rng), sym(rng), sym(rng), 0 };
        b.body2Actor_maxImpulseW.q = randomQuat(rng).toFloat4();
        b.body2Actor_maxImpulseW.p = Float4{ sym(rng) * 0.1f, sym(rng) * 0.1f, sym(rng) * 0.1f, 0 };
    }

    // shapes: 0-4 per body, sorted per-shape node index array
    std::vector<ShapeSim> shapeSims;
    std::vector<uint64_t> nodeIndices;
    std::vector<uint32_t> shapeIndices;
    for (uint32_t bodyIdx = 0; bodyIdx < numBodies; ++bodyIdx)
    {
        const uint32_t nbShapes = rng() % 5;
        for (uint32_t s = 0; s < nbShapes; ++s)
        {
            ShapeSim shape = {};
            shape.transform.q = randomQuat(rng).toQuat();
            shape.transform.p = Float3{ sym(rng) * 0.2f, sym(rng) * 0.2f, sym(rng) * 0.2f };
            shape.bodySimIndex = makeNodeIndex(bodyIdx);
            shape.shapeFlags = eSIMULATION_SHAPE;
            const float roll = unit(rng);
            if (roll < 0.4f)
            {
                shape.shapeType = eSPHERE;
                const float r = 0.5f + unit(rng);
                shape.localBoundsMin = Float3{ -r, -r, -r };
                shape.localBoundsMax = Float3{ r, r, r };
            }
            else if (roll < 0.8f)
            {
                shape.shapeType = eBOX;
                const Float3 he{ 0.2f + unit(rng), 0.2f + unit(rng), 0.2f + unit(rng) };
                shape.localBoundsMin = Float3{ -he.x, -he.y, -he.z };
                shape.localBoundsMax = he;
            }
            else
            {
                shape.shapeType = eCAPSULE;
                const float r = 0.2f + unit(rng);
                const float hh = 0.3f + unit(rng);
                shape.localBoundsMin = Float3{ -(hh + r), -r, -r };
                shape.localBoundsMax = Float3{ hh + r, r, r };
            }
            shape.hullDataIndex = 0xFFFFFFFFu;

            // ~10% deleted entries in the map
            const bool deleted = unit(rng) < 0.1f;
            nodeIndices.push_back(makeNodeIndex(bodyIdx));
            shapeIndices.push_back(deleted ? 0xFFFFFFFFu : uint32_t(shapeSims.size()));
            shapeSims.push_back(shape);
        }
    }
    const uint32_t totalNumShapes = uint32_t(nodeIndices.size());
    const uint32_t numShapes = uint32_t(shapeSims.size());

    // pose updates for a subset of bodies
    std::vector<uint32_t> updateIndices(numBodies);
    std::iota(updateIndices.begin(), updateIndices.end(), 0u);
    std::shuffle(updateIndices.begin(), updateIndices.end(), rng);
    updateIndices.resize(nbUpdates);

    std::vector<float> poseData(nbUpdates * 7);
    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        const QuatV q = randomQuat(rng);
        poseData[i * 7 + 0] = q.x;
        poseData[i * 7 + 1] = q.y;
        poseData[i * 7 + 2] = q.z;
        poseData[i * 7 + 3] = q.w;
        poseData[i * 7 + 4] = sym(rng);
        poseData[i * 7 + 5] = sym(rng);
        poseData[i * 7 + 6] = sym(rng);
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();

    Kernel kernel("updateBounds.spv", "setRigidDynamicGlobalPose",
                  { sizeof(ShapeSim), sizeof(Shape), sizeof(BodySim), 4, sizeof(SolverBodySleepData),
                    sizeof(CachedTransform), 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4 },
                  sizeof(UpdateBoundsParams));

    PxgBufferHandle shapeSimBuf = createBufferFrom(dev, shapeSims);
    PxgBufferHandle dummyShapes = dev->createBuffer(sizeof(Shape));
    PxgBufferHandle bodySimBuf = createBufferFrom(dev, bodySims);
    PxgBufferHandle dummy4a = dev->createBuffer(16);
    PxgBufferHandle dummySleep = dev->createBuffer(16);
    std::vector<CachedTransform> cacheInit(numShapes);
    std::memset(cacheInit.data(), 0, cacheInit.size() * sizeof(CachedTransform));
    PxgBufferHandle cacheBuf = createBufferFrom(dev, cacheInit);
    std::vector<float> boundsInit(numShapes * 6, 0.f);
    PxgBufferHandle boundsBuf = createBufferFrom(dev, boundsInit);
    std::vector<uint32_t> zerosShapes(numShapes, 0);
    PxgBufferHandle dummyFrozen = dev->createBuffer(16);
    PxgBufferHandle dummyUnfrozen = dev->createBuffer(16);
    PxgBufferHandle updatedBuf = createBufferFrom(dev, zerosShapes);
    PxgBufferHandle dummyActive = dev->createBuffer(16);
    PxgBufferHandle dummyDeactivate = dev->createBuffer(16);
    PxgBufferHandle dummyBlob = dev->createBuffer(16);
    PxgBufferHandle nodeIdxBuf = createBufferFrom(dev, nodeIndices);
    PxgBufferHandle shapeIdxBuf = createBufferFrom(dev, shapeIndices);
    PxgBufferHandle poseBuf = createBufferFrom(dev, poseData);
    PxgBufferHandle gpuIdxBuf = createBufferFrom(dev, updateIndices);

    const UpdateBoundsParams params = { 0, nbUpdates, totalNumShapes };
    kernel.dispatch((nbUpdates + 255) / 256,
                    { shapeSimBuf, dummyShapes, bodySimBuf, dummy4a, dummySleep, cacheBuf, boundsBuf,
                      dummyFrozen, dummyUnfrozen, updatedBuf, dummyActive, dummyDeactivate, dummyBlob,
                      nodeIdxBuf, shapeIdxBuf, poseBuf, gpuIdxBuf },
                    params);
    dev->flush();

    // ---- reference ----
    std::vector<BodySim> refBodySims = bodySims;
    std::vector<CachedTransform> refCache = cacheInit;
    std::vector<Bounds3> refBounds(numShapes, Bounds3{});
    std::vector<uint32_t> refUpdated(numShapes, 0);

    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        const uint32_t index = updateIndices[i];
        BodySim& bodySim = refBodySims[index];

        const QuatV inQ(poseData[i * 7 + 0], poseData[i * 7 + 1], poseData[i * 7 + 2], poseData[i * 7 + 3]);
        const Vec3 inP(poseData[i * 7 + 4], poseData[i * 7 + 5], poseData[i * 7 + 6]);

        const QuatV b2aQ(bodySim.body2Actor_maxImpulseW.q);
        const Vec3 b2aP(bodySim.body2Actor_maxImpulseW.p.x, bodySim.body2Actor_maxImpulseW.p.y,
                        bodySim.body2Actor_maxImpulseW.p.z);

        const QuatV q = inQ * b2aQ;
        const Vec3 p = inQ.rotate(b2aP) + inP;
        const TransformV body2World(q, p);

        bodySim.body2World.q = q.toFloat4();
        bodySim.body2World.p = Float4{ p.x, p.y, p.z, 0.f };

        const TransformV body2Actor(b2aQ, b2aP);

        // binarySearch: last pos with node <= value, then walk backward
        const uint64_t nodeIndex = makeNodeIndex(index);
        uint32_t left = 0, right = totalNumShapes;
        while (left < right)
        {
            const uint32_t pos = (left + right) / 2;
            if (nodeIndices[pos] <= nodeIndex)
                left = pos + 1;
            else
                right = pos;
        }
        uint32_t pos = left != 0 ? left - 1 : 0;

        while (pos != 0xFFFFFFFFu && nodeIndices[pos] == nodeIndex)
        {
            const uint32_t shapeIndex = shapeIndices[pos];
            if (shapeIndex != 0xFFFFFFFFu)
            {
                const ShapeSim& shapeSim = shapeSims[shapeIndex];
                const TransformV absPos = getAbsPose(body2World, TransformV(shapeSim.transform), body2Actor);
                refCache[shapeIndex].transform = absPos.toTransform28();
                refCache[shapeIndex].flags = 0;
                refUpdateBounds(shapeSim, {}, {}, absPos, refBounds[shapeIndex]);
                refUpdated[shapeIndex] = 1;
            }
            --pos;
        }
    }

    // ---- compare ----
    const std::vector<BodySim> gpuBodySims = downloadBuffer<BodySim>(dev, bodySimBuf, numBodies);
    const std::vector<CachedTransform> gpuCache = downloadBuffer<CachedTransform>(dev, cacheBuf, numShapes);
    const std::vector<float> gpuBounds = downloadBuffer<float>(dev, boundsBuf, numShapes * 6);
    const std::vector<uint32_t> gpuUpdated = downloadBuffer<uint32_t>(dev, updatedBuf, numShapes);

    for (uint32_t n = 0; n < numBodies; ++n)
    {
        const Float4& gq = gpuBodySims[n].body2World.q;
        const Float4& rq = refBodySims[n].body2World.q;
        const Float4& gp = gpuBodySims[n].body2World.p;
        const Float4& rp = refBodySims[n].body2World.p;
        PXG_CHECK_MSG(almostEqual(gq.x, rq.x) && almostEqual(gq.y, rq.y) && almostEqual(gq.z, rq.z) &&
                          almostEqual(gq.w, rq.w) && almostEqual(gp.x, rp.x, 1e-4f, 1e-4f) &&
                          almostEqual(gp.y, rp.y, 1e-4f, 1e-4f) && almostEqual(gp.z, rp.z, 1e-4f, 1e-4f),
                      "setGlobalPose body2World[%u]", n);
    }

    for (uint32_t s = 0; s < numShapes; ++s)
    {
        PXG_CHECK_MSG(gpuUpdated[s] == refUpdated[s], "setGlobalPose updated[%u]: gpu=%u ref=%u", s,
                      gpuUpdated[s], refUpdated[s]);
        if (!refUpdated[s])
            continue;

        const Transform28& gt = gpuCache[s].transform;
        const Transform28& rt = refCache[s].transform;
        PXG_CHECK_MSG(almostEqual(gt.q.x, rt.q.x, 1e-4f, 1e-5f) && almostEqual(gt.q.y, rt.q.y, 1e-4f, 1e-5f) &&
                          almostEqual(gt.q.z, rt.q.z, 1e-4f, 1e-5f) && almostEqual(gt.q.w, rt.q.w, 1e-4f, 1e-5f) &&
                          almostEqual(gt.p.x, rt.p.x, 1e-4f, 1e-4f) && almostEqual(gt.p.y, rt.p.y, 1e-4f, 1e-4f) &&
                          almostEqual(gt.p.z, rt.p.z, 1e-4f, 1e-4f),
                      "setGlobalPose cache[%u]", s);

        const float rb[6] = { refBounds[s].minimum.x, refBounds[s].minimum.y, refBounds[s].minimum.z,
                              refBounds[s].maximum.x, refBounds[s].maximum.y, refBounds[s].maximum.z };
        for (int c = 0; c < 6; ++c)
            PXG_CHECK_MSG(almostEqual(gpuBounds[s * 6 + c], rb[c], 1e-4f, 1e-3f),
                          "setGlobalPose bounds[%u][%d]: gpu=%g ref=%g", s, c, gpuBounds[s * 6 + c], rb[c]);
    }

    dev->destroyBuffer(shapeSimBuf);
    dev->destroyBuffer(dummyShapes);
    dev->destroyBuffer(bodySimBuf);
    dev->destroyBuffer(dummy4a);
    dev->destroyBuffer(dummySleep);
    dev->destroyBuffer(cacheBuf);
    dev->destroyBuffer(boundsBuf);
    dev->destroyBuffer(dummyFrozen);
    dev->destroyBuffer(dummyUnfrozen);
    dev->destroyBuffer(updatedBuf);
    dev->destroyBuffer(dummyActive);
    dev->destroyBuffer(dummyDeactivate);
    dev->destroyBuffer(dummyBlob);
    dev->destroyBuffer(nodeIdxBuf);
    dev->destroyBuffer(shapeIdxBuf);
    dev->destroyBuffer(poseBuf);
    dev->destroyBuffer(gpuIdxBuf);
}

// ---------------------------------------------------------------------------
// P9.86c: setRigidDynamicGlobalPose on CONVEX hulls with a +-0 tie at the x
// extreme (updateBounds.slang updateBounds(), the direct-API copy of
// updateCacheAndBound.cuh:126-143).  PxVec3::minimum/maximum are the PxMin/
// PxMax selects (a<b?a:b / a>b?a:b): on a +-0 tie the LAST tied vertex's zero
// survives.  GLSL FMin/FMax on llvmpipe give min(+0,-0) = -0 and max(+0,-0) =
// +0 in either order.  The pose: a user global pose rotated 180 degrees about
// z with p.x = -0 (a negated zero), body COM offset (0,0.5,-0.5), shape offset
// (0,1,-1) -> absPos = q(+0,+0,1,+0) p(-0,1,2); rot.col0 = (-1,+0,+0),
// rot.col1 = (+0,-1,+0), so a base vertex (+0,y,z) maps to x' = -0 iff y<0 and
// z<0, else +0.  pose.p.x = -0 carries the tie's sign into the stored bound.
//  hull 0 (apex -x -> x' = +1): the x MINIMUM is the zero tie; the -0 vertex
//    comes FIRST -> CUDA minX = +0, FMin fold -0.
//  hull 1 (apex +x -> x' = -1): the x MAXIMUM is the zero tie; the -0 vertex
//    comes LAST -> CUDA maxX = -0, FMax fold +0.
// All 12 bound words and the cache are compared BITWISE.
// ---------------------------------------------------------------------------
PXG_TEST(setRigidDynamicGlobalPose_convexZeroTie)
{
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };

    // P9.94a pin: pxgRefMath's TransformV::transformInv must be the CUDA's
    // getConjugate().rotate, not rotateInv.  identity.transformInv((-0,-0,1)):
    // the conjugate's x lane is -0, rotateInv's is +0.
    {
        const TransformV pin = TransformV(QuatV(0.f, 0.f, 0.f, 1.f), Vec3(0.f, 0.f, 0.f))
                                   .transformInv(TransformV(QuatV(0.f, 0.f, 0.f, 1.f), Vec3(-0.f, -0.f, 1.f)));
        PXG_CHECK_MSG(bits(pin.p.x) == 0x80000000u && bits(pin.p.y) == 0u && bits(pin.p.z) == 0x3f800000u,
                      "P9.94a: TransformV::transformInv must be conj().rotate (got %08x %08x %08x)",
                      bits(pin.p.x), bits(pin.p.y), bits(pin.p.z));
    }

    std::vector<BodySim> bodySims(1);
    std::memset(bodySims.data(), 0, sizeof(BodySim));
    bodySims[0].body2World.q = Float4{ 0.f, 0.f, 0.f, 1.f };
    bodySims[0].body2Actor_maxImpulseW.q = Float4{ 0.f, 0.f, 0.f, 1.f };
    bodySims[0].body2Actor_maxImpulseW.p = Float4{ 0.f, 0.5f, -0.5f, 0.f };

    const uint32_t numShapes = 2;
    std::vector<ShapeSim> shapeSims(numShapes);
    std::vector<Shape> shapes(numShapes);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(Shape));
    for (uint32_t s = 0; s < numShapes; ++s)
    {
        ShapeSim sh = {};
        sh.transform.q = Quat{ 0.f, 0.f, 0.f, 1.f };
        sh.transform.p = Float3{ 0.f, 1.f, -1.f };
        sh.localBoundsMin = Float3{ -1.f, -1.f, -1.f };
        sh.localBoundsMax = Float3{ 1.f, 1.f, 1.f };
        sh.bodySimIndex = makeNodeIndex(0);
        sh.hullDataIndex = s;
        sh.shapeFlags = eSIMULATION_SHAPE;
        sh.shapeType = eCONVEXMESH;
        shapeSims[s] = sh;

        shapes[s].scale = Float3{ 1.f, 1.f, 1.f };
        shapes[s].scaleRotation = Quat{ 0.f, 0.f, 0.f, 1.f };
        shapes[s].type = eCONVEXMESH;
        shapes[s].hullOrMeshOffset = uint64_t(s) * 128; // byte offset (bound-buffer path)
    }

    // two square pyramids with their base on the local x = 0 plane (all +0)
    static const float hullVerts[2][5][3] = {
        { { 0.f, -1.f, -1.f }, { 0.f, 1.f, 1.f }, { 0.f, 1.f, -1.f }, { 0.f, -1.f, 1.f }, { -1.f, 0.f, 0.f } },
        { { 0.f, 1.f, 1.f }, { 0.f, 1.f, -1.f }, { 0.f, -1.f, 1.f }, { 0.f, -1.f, -1.f }, { 1.f, 0.f, 0.f } },
    };
    std::vector<uint32_t> geometryBlob(64, 0u);
    for (uint32_t h = 0; h < 2; ++h)
    {
        const uint32_t base = h * 32;
        geometryBlob[base + 4] = 5u << 8;
        for (uint32_t v = 0; v < 5; ++v)
            std::memcpy(&geometryBlob[base + 12 + v * 4], hullVerts[h][v], 12);
    }

    const std::vector<uint64_t> nodeIndices = { makeNodeIndex(0), makeNodeIndex(0) };
    const std::vector<uint32_t> shapeIndices = { 0u, 1u };
    const std::vector<float> poseData = { 0.f, 0.f, 1.f, 0.f, -0.0f, 2.f, 3.f };
    const std::vector<uint32_t> updateIndices = { 0u };
    const uint32_t nbUpdates = 1, totalNumShapes = 2;

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    Kernel kernel("updateBounds.spv", "setRigidDynamicGlobalPose",
                  { sizeof(ShapeSim), sizeof(Shape), sizeof(BodySim), 4, sizeof(SolverBodySleepData),
                    sizeof(CachedTransform), 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4 },
                  sizeof(UpdateBoundsParams));

    PxgBufferHandle shapeSimBuf = createBufferFrom(dev, shapeSims);
    PxgBufferHandle shapesBuf = createBufferFrom(dev, shapes);
    PxgBufferHandle bodySimBuf = createBufferFrom(dev, bodySims);
    PxgBufferHandle dummy4a = dev->createBuffer(16);
    PxgBufferHandle dummySleep = dev->createBuffer(16);
    std::vector<CachedTransform> cacheInit(numShapes);
    std::memset(cacheInit.data(), 0, cacheInit.size() * sizeof(CachedTransform));
    PxgBufferHandle cacheBuf = createBufferFrom(dev, cacheInit);
    std::vector<float> boundsInit(numShapes * 6, 0.f);
    PxgBufferHandle boundsBuf = createBufferFrom(dev, boundsInit);
    std::vector<uint32_t> zerosShapes(numShapes, 0);
    PxgBufferHandle dummyFrozen = dev->createBuffer(16);
    PxgBufferHandle dummyUnfrozen = dev->createBuffer(16);
    PxgBufferHandle updatedBuf = createBufferFrom(dev, zerosShapes);
    PxgBufferHandle dummyActive = dev->createBuffer(16);
    PxgBufferHandle dummyDeactivate = dev->createBuffer(16);
    PxgBufferHandle blobBuf = createBufferFrom(dev, geometryBlob);
    PxgBufferHandle nodeIdxBuf = createBufferFrom(dev, nodeIndices);
    PxgBufferHandle shapeIdxBuf = createBufferFrom(dev, shapeIndices);
    PxgBufferHandle poseBuf = createBufferFrom(dev, poseData);
    PxgBufferHandle gpuIdxBuf = createBufferFrom(dev, updateIndices);

    const UpdateBoundsParams params = { 0, nbUpdates, totalNumShapes };
    kernel.dispatch(1,
                    { shapeSimBuf, shapesBuf, bodySimBuf, dummy4a, dummySleep, cacheBuf, boundsBuf, dummyFrozen,
                      dummyUnfrozen, updatedBuf, dummyActive, dummyDeactivate, blobBuf, nodeIdxBuf, shapeIdxBuf,
                      poseBuf, gpuIdxBuf },
                    params);
    dev->flush();

    // ---- reference (CUDA semantics) + FMin/FMax twin ----
    const QuatV inQ(poseData[0], poseData[1], poseData[2], poseData[3]);
    const Vec3 inP(poseData[4], poseData[5], poseData[6]);
    const QuatV b2aQ(bodySims[0].body2Actor_maxImpulseW.q);
    const Vec3 b2aP(bodySims[0].body2Actor_maxImpulseW.p.x, bodySims[0].body2Actor_maxImpulseW.p.y,
                    bodySims[0].body2Actor_maxImpulseW.p.z);
    const TransformV body2World(inQ * b2aQ, inQ.rotate(b2aP) + inP);
    const TransformV body2Actor(b2aQ, b2aP);

    std::vector<Bounds3> refB(numShapes, Bounds3{});
    std::vector<UbFMinMaxTwin> twins(numShapes);
    std::vector<Transform28> refCache(numShapes);
    uint32_t poseNegZeroSeen = 0, zeroTieMinSeen = 0, zeroTieMaxSeen = 0, otherDiffers = 0;
    for (uint32_t s = 0; s < numShapes; ++s)
    {
        const TransformV shape2Actor(shapeSims[s].transform);
        // rotateInv and the CUDA's getConjugate().rotate agree on THIS input (they
        // differ only in the sign of exact zeros, which these shapes never produce);
        // the pin below (P9.94a) is the input where they do not
        const Vec3 t0aP = b2aQ.rotateInv(shape2Actor.p - b2aP);
        const Vec3 t0bP = b2aQ.getConjugate().rotate(shape2Actor.p - b2aP);
        PXG_CHECK_MSG(bits(t0aP.x) == bits(t0bP.x) && bits(t0aP.y) == bits(t0bP.y) &&
                          bits(t0aP.z) == bits(t0bP.z),
                      "convexZeroTie: rotateInv and conj().rotate must agree on t0.p (shape %u)", s);

        const TransformV absPos = getAbsPose(body2World, shape2Actor, body2Actor);
        if (absPos.p.x == 0.f && std::signbit(absPos.p.x))
            ++poseNegZeroSeen;
        refCache[s] = absPos.toTransform28();
        refUpdateBounds(shapeSims[s], shapes, geometryBlob, absPos, refB[s], 0, &twins[s]);
        PXG_CHECK_MSG(twins[s].valid, "convexZeroTie: shape %u must take the convex hull path", s);

        const float rw[6] = { refB[s].minimum.x, refB[s].minimum.y, refB[s].minimum.z,
                              refB[s].maximum.x, refB[s].maximum.y, refB[s].maximum.z };
        const float tw[6] = { twins[s].minimum.x, twins[s].minimum.y, twins[s].minimum.z,
                              twins[s].maximum.x, twins[s].maximum.y, twins[s].maximum.z };
        for (int c = 0; c < 6; ++c)
        {
            if (bits(rw[c]) == bits(tw[c]))
                continue;
            if (s == 0 && c == 0)
                ++zeroTieMinSeen;
            else if (s == 1 && c == 3)
                ++zeroTieMaxSeen;
            else
                ++otherDiffers;
        }
    }
    PXG_CHECK_MSG(poseNegZeroSeen == 2 && zeroTieMinSeen == 1 && zeroTieMaxSeen == 1 && otherDiffers == 0,
                  "convexZeroTie coverage: poseNegZero %u (2) minTie %u (1) maxTie %u (1) other %u (0)",
                  poseNegZeroSeen, zeroTieMinSeen, zeroTieMaxSeen, otherDiffers);
    PXG_CHECK_MSG(bits(refB[0].minimum.x) == 0x00000000u && bits(refB[1].maximum.x) == 0x80000000u,
                  "convexZeroTie: CUDA image minX(0)=+0 maxX(1)=-0 (got %08x %08x)", bits(refB[0].minimum.x),
                  bits(refB[1].maximum.x));

    // ---- compare (bitwise) ----
    const std::vector<CachedTransform> gpuCache = downloadBuffer<CachedTransform>(dev, cacheBuf, numShapes);
    const std::vector<float> gpuBounds = downloadBuffer<float>(dev, boundsBuf, numShapes * 6);
    const std::vector<uint32_t> gpuUpdated = downloadBuffer<uint32_t>(dev, updatedBuf, numShapes);
    for (uint32_t s = 0; s < numShapes; ++s)
    {
        PXG_CHECK_MSG(gpuUpdated[s] == 1u, "convexZeroTie updated[%u] = %u", s, gpuUpdated[s]);
        PXG_CHECK_MSG(std::memcmp(&gpuCache[s].transform, &refCache[s], sizeof(Transform28)) == 0,
                      "convexZeroTie cache[%u] differs bitwise", s);
        const float rb[6] = { refB[s].minimum.x, refB[s].minimum.y, refB[s].minimum.z,
                              refB[s].maximum.x, refB[s].maximum.y, refB[s].maximum.z };
        for (int c = 0; c < 6; ++c)
            PXG_CHECK_MSG(bits(gpuBounds[s * 6 + c]) == bits(rb[c]), "convexZeroTie bounds[%u][%d]: gpu=%08x ref=%08x",
                          s, c, bits(gpuBounds[s * 6 + c]), bits(rb[c]));
    }

    for (PxgBufferHandle h : { shapeSimBuf, shapesBuf, bodySimBuf, dummy4a, dummySleep, cacheBuf, boundsBuf,
                               dummyFrozen, dummyUnfrozen, updatedBuf, dummyActive, dummyDeactivate, blobBuf,
                               nodeIdxBuf, shapeIdxBuf, poseBuf, gpuIdxBuf })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// P9.89: setRigidDynamicGlobalPose (RIGID_DYNAMIC_SET_GLOBAL_POSE) through the two
// order-sensitive updateBounds() branches it reaches in updateBounds.slang:177
// (PxMeshScale::toMat33, PxMeshScale.h:120-128: (rot^T column-scaled) * rot) and
// :214/:220 (PxBounds3::transformFast, PxBounds3.h:472-476: center = q.rotate(c) + p).
// The fleet: 8 bodies, 24 shapes each, round-robin scaled hull / GPU-incompatible
// convex / mesh-type shape; two 64-vertex hulls under non-uniform, non-round mesh
// scales and random scale rotations; fast-path local bounds far off the shape
// origin with small extents.  The CUDA-order mirror and the pre-P9.89 twin
// (rot^T * (diag(s) * rot), PxMat33(q) center product) must store different bits
// on every path, and the GPU is compared with the mirror BITWISE (cache, bounds,
// body2World, updated flags).
// ---------------------------------------------------------------------------
PXG_TEST(setRigidDynamicGlobalPose_p89MeshScaleCenterOrder)
{
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    std::mt19937 rng(0x5896u);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-10.f, 10.f);

    const uint32_t numBodies = 8;
    const uint32_t shapesPerBody = 24;
    const uint32_t numShapes = numBodies * shapesPerBody;
    const uint32_t numHulls = 3;   // P9.93: hull 2 is a UNIT scale with a rotated scale axis
    const uint32_t nbVerts = 64;

    // ---- bodies: random COM frames with small offsets
    std::vector<BodySim> bodySims(numBodies);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(BodySim));
    for (BodySim& b : bodySims)
    {
        b.body2World.q = randomQuat(rng).toFloat4();
        b.body2World.p = Float4{ sym(rng), sym(rng), sym(rng), 0 };
        b.body2Actor_maxImpulseW.q = randomQuat(rng).toFloat4();
        b.body2Actor_maxImpulseW.p = Float4{ 0.01f * sym(rng), 0.01f * sym(rng), 0.01f * sym(rng), 0 };
    }

    // ---- hulls (bound-buffer path: hullOrMeshOffset is a byte offset into the blob)
    std::vector<uint32_t> geometryBlob;
    std::vector<Shape> shapes(numHulls);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(Shape));
    for (uint32_t h = 0; h < numHulls; ++h)
    {
        const uint32_t base = uint32_t(geometryBlob.size());
        geometryBlob.resize(base + 12 + nbVerts * 4, 0u);
        geometryBlob[base + 4] = (nbVerts & 0xFFu) << 8;
        for (uint32_t v = 0; v < nbVerts; ++v)
        {
            const float vx = sym(rng), vy = sym(rng), vz = sym(rng);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 0], &vx, 4);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 1], &vy, 4);
            std::memcpy(&geometryBlob[base + 12 + v * 4 + 2], &vz, 4);
        }
        shapes[h].scale = Float3{ 0.3f + 2.6f * unit(rng), 0.3f + 2.6f * unit(rng), 0.3f + 2.6f * unit(rng) };
        shapes[h].scaleRotation = randomQuat(rng).toQuat();
        shapes[h].type = eCONVEXMESH;
        shapes[h].hullOrMeshOffset = uint64_t(base) * 4;
    }
    // P9.93: PxMeshScale::isIdentity() tests the scale only, so hull 2 -- scale
    // (1, 1, 1) under a rotated scale axis, a valid PxMeshScale -- takes the
    // IDENTITY path in the CUDA; the pre-P9.93 port multiplied by R^T R there
    shapes[2].scale = Float3{ 1.f, 1.f, 1.f };

    // ---- shapes, sorted by body (the per-shape node index array is ascending)
    std::vector<ShapeSim> shapeSims(numShapes);
    std::vector<uint64_t> nodeIndices(numShapes);
    std::vector<uint32_t> shapeIndices(numShapes);
    for (uint32_t i = 0; i < numShapes; ++i)
    {
        const uint32_t body = i / shapesPerBody, k = i % shapesPerBody;
        ShapeSim s = {};
        s.transform.q = randomQuat(rng).toQuat();
        s.transform.p = Float3{ 0.02f * sym(rng), 0.02f * sym(rng), 0.02f * sym(rng) };
        auto farCoord = [&]() {
            const float m = 4.f + 16.f * unit(rng);
            return unit(rng) < 0.5f ? -m : m;
        };
        const Float3 c{ farCoord(), farCoord(), farCoord() };
        const Float3 e{ 0.05f + 0.45f * unit(rng), 0.05f + 0.45f * unit(rng), 0.05f + 0.45f * unit(rng) };
        s.localBoundsMin = Float3{ c.x - e.x, c.y - e.y, c.z - e.z };
        s.localBoundsMax = Float3{ c.x + e.x, c.y + e.y, c.z + e.z };
        s.bodySimIndex = makeNodeIndex(body);
        s.shapeFlags = eSIMULATION_SHAPE;
        switch (k % 3)
        {
        case 0: s.shapeType = eCONVEXMESH; s.hullDataIndex = (k == 21u) ? 2u : (k / 3) % 2u; break;
        case 1: s.shapeType = eCONVEXMESH; s.hullDataIndex = 0xFFFFFFFFu; break;
        default: s.shapeType = uint16_t(6 + (k / 3) % 3); s.hullDataIndex = 0xFFFFFFFFu; break;
        }
        shapeSims[i] = s;
        nodeIndices[i] = makeNodeIndex(body);
        shapeIndices[i] = i;
    }

    // ---- one pose update per body (small translations: the far local centers
    // dominate |newCenter|, so the rotation's rounding survives + p)
    const uint32_t nbUpdates = numBodies;
    std::vector<uint32_t> updateIndices(nbUpdates);
    std::iota(updateIndices.begin(), updateIndices.end(), 0u);
    std::shuffle(updateIndices.begin(), updateIndices.end(), rng);
    std::vector<float> poseData(nbUpdates * 7);
    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        const QuatV q = randomQuat(rng);
        poseData[i * 7 + 0] = q.x;
        poseData[i * 7 + 1] = q.y;
        poseData[i * 7 + 2] = q.z;
        poseData[i * 7 + 3] = q.w;
        poseData[i * 7 + 4] = 0.03f * sym(rng);
        poseData[i * 7 + 5] = 0.03f * sym(rng);
        poseData[i * 7 + 6] = 0.03f * sym(rng);
    }

    // ---- reference (CUDA order) + the pre-P9.89 twin
    std::vector<BodySim> refBodySims = bodySims;
    std::vector<Transform28> refCache(numShapes);
    std::vector<Bounds3> refB(numShapes, Bounds3{}), oldB(numShapes, Bounds3{});
    std::vector<uint32_t> refUpdated(numShapes, 0);
    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        const uint32_t index = updateIndices[i];
        BodySim& bodySim = refBodySims[index];
        const QuatV inQ(poseData[i * 7 + 0], poseData[i * 7 + 1], poseData[i * 7 + 2], poseData[i * 7 + 3]);
        const Vec3 inP(poseData[i * 7 + 4], poseData[i * 7 + 5], poseData[i * 7 + 6]);
        const QuatV b2aQ(bodySim.body2Actor_maxImpulseW.q);
        const Vec3 b2aP(bodySim.body2Actor_maxImpulseW.p.x, bodySim.body2Actor_maxImpulseW.p.y,
                        bodySim.body2Actor_maxImpulseW.p.z);
        const TransformV body2World(inQ * b2aQ, inQ.rotate(b2aP) + inP);
        bodySim.body2World.q = body2World.q.toFloat4();
        bodySim.body2World.p = Float4{ body2World.p.x, body2World.p.y, body2World.p.z, 0.f };
        const TransformV body2Actor(b2aQ, b2aP);
        for (uint32_t s = index * shapesPerBody; s < (index + 1) * shapesPerBody; ++s)
        {
            const TransformV absPos = getAbsPose(body2World, TransformV(shapeSims[s].transform), body2Actor);
            refCache[s] = absPos.toTransform28();
            refUpdateBounds(shapeSims[s], shapes, geometryBlob, absPos, refB[s]);
            refUpdateBounds(shapeSims[s], shapes, geometryBlob, absPos, oldB[s], 0, nullptr, true);
            refUpdated[s] = 1;
        }
    }

    // ---- P9.89 coverage: the old port order must change the bits on every path
    {
        uint32_t pathDiff[eUB_NB_PATHS] = {};
        for (uint32_t s = 0; s < numShapes; ++s)
        {
            const float rw[6] = { refB[s].minimum.x, refB[s].minimum.y, refB[s].minimum.z,
                                  refB[s].maximum.x, refB[s].maximum.y, refB[s].maximum.z };
            const float ow[6] = { oldB[s].minimum.x, oldB[s].minimum.y, oldB[s].minimum.z,
                                  oldB[s].maximum.x, oldB[s].maximum.y, oldB[s].maximum.z };
            for (int c = 0; c < 6; ++c)
                pathDiff[ubP89Path(shapeSims[s], shapes)] += bits(rw[c]) != bits(ow[c]) ? 1u : 0u;
        }
        std::printf("[ info ] P9.89 setGlobalPose: %u / %u / %u of %u bound words each differ CUDA vs old port "
                    "order (scaled hull / fast convex / fast other)\n",
                    pathDiff[eUB_HULL_SCALED], pathDiff[eUB_FAST_CONVEX], pathDiff[eUB_FAST_OTHER],
                    numShapes / 3 * 6);
        // measured: 62 / 374 / 365 of 384
        PXG_CHECK_MSG(pathDiff[eUB_HULL_SCALED] >= 40 && pathDiff[eUB_FAST_CONVEX] >= 300 &&
                          pathDiff[eUB_FAST_OTHER] >= 300,
                      "P9.89 coverage: scaled hull %u (want >= 40), fast convex %u (>= 300), fast other %u (>= 300)",
                      pathDiff[eUB_HULL_SCALED], pathDiff[eUB_FAST_CONVEX], pathDiff[eUB_FAST_OTHER]);
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    Kernel kernel("updateBounds.spv", "setRigidDynamicGlobalPose",
                  { sizeof(ShapeSim), sizeof(Shape), sizeof(BodySim), 4, sizeof(SolverBodySleepData),
                    sizeof(CachedTransform), 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4 },
                  sizeof(UpdateBoundsParams));

    PxgBufferHandle shapeSimBuf = createBufferFrom(dev, shapeSims);
    PxgBufferHandle shapesBuf = createBufferFrom(dev, shapes);
    PxgBufferHandle bodySimBuf = createBufferFrom(dev, bodySims);
    PxgBufferHandle dummy4a = dev->createBuffer(16);
    PxgBufferHandle dummySleep = dev->createBuffer(16);
    std::vector<CachedTransform> cacheInit(numShapes);
    std::memset(cacheInit.data(), 0, cacheInit.size() * sizeof(CachedTransform));
    PxgBufferHandle cacheBuf = createBufferFrom(dev, cacheInit);
    std::vector<float> boundsInit(numShapes * 6, 0.f);
    PxgBufferHandle boundsBuf = createBufferFrom(dev, boundsInit);
    std::vector<uint32_t> zerosShapes(numShapes, 0);
    PxgBufferHandle dummyFrozen = dev->createBuffer(16);
    PxgBufferHandle dummyUnfrozen = dev->createBuffer(16);
    PxgBufferHandle updatedBuf = createBufferFrom(dev, zerosShapes);
    PxgBufferHandle dummyActive = dev->createBuffer(16);
    PxgBufferHandle dummyDeactivate = dev->createBuffer(16);
    PxgBufferHandle blobBuf = createBufferFrom(dev, geometryBlob);
    PxgBufferHandle nodeIdxBuf = createBufferFrom(dev, nodeIndices);
    PxgBufferHandle shapeIdxBuf = createBufferFrom(dev, shapeIndices);
    PxgBufferHandle poseBuf = createBufferFrom(dev, poseData);
    PxgBufferHandle gpuIdxBuf = createBufferFrom(dev, updateIndices);

    const UpdateBoundsParams params = { 0, nbUpdates, numShapes };
    kernel.dispatch(1,
                    { shapeSimBuf, shapesBuf, bodySimBuf, dummy4a, dummySleep, cacheBuf, boundsBuf, dummyFrozen,
                      dummyUnfrozen, updatedBuf, dummyActive, dummyDeactivate, blobBuf, nodeIdxBuf, shapeIdxBuf,
                      poseBuf, gpuIdxBuf },
                    params);
    dev->flush();

    // ---- compare (bitwise) ----
    const std::vector<BodySim> gpuBodySims = downloadBuffer<BodySim>(dev, bodySimBuf, numBodies);
    const std::vector<CachedTransform> gpuCache = downloadBuffer<CachedTransform>(dev, cacheBuf, numShapes);
    const std::vector<float> gpuBounds = downloadBuffer<float>(dev, boundsBuf, numShapes * 6);
    const std::vector<uint32_t> gpuUpdated = downloadBuffer<uint32_t>(dev, updatedBuf, numShapes);
    for (uint32_t n = 0; n < numBodies; ++n)
        PXG_CHECK_MSG(std::memcmp(&gpuBodySims[n].body2World, &refBodySims[n].body2World,
                                  sizeof(refBodySims[n].body2World)) == 0,
                      "p89 setGlobalPose body2World[%u] differs bitwise", n);
    uint32_t cacheBad = 0, boundsBad[eUB_NB_PATHS] = {};
    for (uint32_t s = 0; s < numShapes; ++s)
    {
        PXG_CHECK_MSG(gpuUpdated[s] == refUpdated[s], "p89 setGlobalPose updated[%u]: gpu=%u ref=%u", s,
                      gpuUpdated[s], refUpdated[s]);
        cacheBad += std::memcmp(&gpuCache[s].transform, &refCache[s], sizeof(Transform28)) != 0 ? 1u : 0u;
        const float rb[6] = { refB[s].minimum.x, refB[s].minimum.y, refB[s].minimum.z,
                              refB[s].maximum.x, refB[s].maximum.y, refB[s].maximum.z };
        const uint32_t path = ubP89Path(shapeSims[s], shapes);
        for (int c = 0; c < 6; ++c)
        {
            if (bits(gpuBounds[s * 6 + c]) == bits(rb[c]))
                continue;
            if (boundsBad[path] < 4)
                std::printf("MM p89 setGlobalPose bounds[%u][%d] (%s): gpu=%08x ref=%08x\n", s, c,
                            kUbP89PathNames[path], bits(gpuBounds[s * 6 + c]), bits(rb[c]));
            ++boundsBad[path];
        }
    }
    PXG_CHECK_MSG(cacheBad == 0, "p89 setGlobalPose: %u transform cache entries differ bitwise", cacheBad);
    for (uint32_t p = 0; p < eUB_NB_PATHS; ++p)
        PXG_CHECK_MSG(boundsBad[p] == 0, "p89 setGlobalPose: %u %s bound words differ bitwise from the CUDA order",
                      boundsBad[p], kUbP89PathNames[p]);

    for (PxgBufferHandle h : { shapeSimBuf, shapesBuf, bodySimBuf, dummy4a, dummySleep, cacheBuf, boundsBuf,
                               dummyFrozen, dummyUnfrozen, updatedBuf, dummyActive, dummyDeactivate, blobBuf,
                               nodeIdxBuf, shapeIdxBuf, poseBuf, gpuIdxBuf })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// P9.90: setRigidDynamicGlobalPose's pose chain in the CUDA operation order.
// body2World = inQ * b2aQ, inQ.rotate(b2aP) + inP (updateBodiesAndShapes.cu), then
// getAbsPose = body2World.transform(body2Actor.transformInv(shape2Actor))
// (updateCacheAndBound.cuh) -- PxQuat::rotate/rotateInv (PxQuat.h:286-310) whose
// dot2 = (x*vx + y*vy) + z*vz is the forward PxVec3 dot order, and the Hamilton
// product.  updateBounds is NoContraction-stamped; the chains are scalar mul+add
// with no OpDot.  1024 general orientations, non-round offsets, 4 shapes each
// (sphere / capsule / box / transformFast), everything compared BITWISE.  The
// twin evaluates every rotate/rotateInv with the re-associated dot2 x*vx + (y*vy +
// z*vz) (what llvmpipe does to an unstamped OpDot) and must change a large part of
// the cache and bounds.
// ---------------------------------------------------------------------------
namespace p90ub
{
static Vec3 rotate(const QuatV& q, const Vec3& v, bool oldOrder)
{
    if (!oldOrder)
        return q.rotate(v);
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + (q.y * vy + q.z * vz);
    return Vec3((vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2),
                (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
                (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}
static Vec3 rotateInv(const QuatV& q, const Vec3& v, bool oldOrder)
{
    if (!oldOrder)
        return q.rotateInv(v);
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + (q.y * vy + q.z * vz);
    return Vec3((vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2),
                (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
                (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}
static TransformV absPose(const TransformV& b2w, const TransformV& s2a, const TransformV& b2a, bool oldOrder)
{
    const TransformV t0(b2a.q.getConjugate() * s2a.q, rotateInv(b2a.q, s2a.p - b2a.p, oldOrder));
    return TransformV(b2w.q * t0.q, rotate(b2w.q, t0.p, oldOrder) + b2w.p);
}
} // namespace p90ub

PXG_TEST(setRigidDynamicGlobalPose_p90PoseOrder)
{
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    std::mt19937 rng(0x90u);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-1.f, 1.f);

    const uint32_t numBodies = 1024, shapesPerBody = 4, numShapes = numBodies * shapesPerBody;

    std::vector<BodySim> bodySims(numBodies);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(BodySim));
    for (BodySim& b : bodySims)
    {
        b.body2World.q = randomQuat(rng).toFloat4();
        b.body2World.p = Float4{ 3.7f * sym(rng), 3.7f * sym(rng), 3.7f * sym(rng), 0 };
        b.body2Actor_maxImpulseW.q = randomQuat(rng).toFloat4();
        b.body2Actor_maxImpulseW.p = Float4{ 0.73f * sym(rng), 0.73f * sym(rng), 0.73f * sym(rng), 0 };
    }

    std::vector<ShapeSim> shapeSims(numShapes);
    std::vector<uint64_t> nodeIndices(numShapes);
    std::vector<uint32_t> shapeIndices(numShapes);
    for (uint32_t i = 0; i < numShapes; ++i)
    {
        const uint32_t body = i / shapesPerBody, k = i % shapesPerBody;
        ShapeSim s = {};
        s.transform.q = randomQuat(rng).toQuat();
        s.transform.p = Float3{ 0.91f * sym(rng), 0.91f * sym(rng), 0.91f * sym(rng) };
        s.bodySimIndex = makeNodeIndex(body);
        s.shapeFlags = eSIMULATION_SHAPE;
        s.hullDataIndex = 0xFFFFFFFFu;
        switch (k)
        {
        case 0:
        {
            s.shapeType = eSPHERE;
            const float r = 0.13f + unit(rng);
            s.localBoundsMin = Float3{ -r, -r, -r };
            s.localBoundsMax = Float3{ r, r, r };
            break;
        }
        case 1:
        {
            s.shapeType = eCAPSULE;
            const float r = 0.11f + unit(rng), hh = 0.17f + unit(rng);
            s.localBoundsMin = Float3{ -(hh + r), -r, -r };
            s.localBoundsMax = Float3{ hh + r, r, r };
            break;
        }
        case 2:
        {
            s.shapeType = eBOX;
            const Float3 he{ 0.07f + unit(rng), 0.07f + unit(rng), 0.07f + unit(rng) };
            s.localBoundsMin = Float3{ -he.x, -he.y, -he.z };
            s.localBoundsMax = he;
            break;
        }
        default:
        {
            s.shapeType = uint16_t(6); // triangle mesh: transformFast
            const Float3 c{ 1.3f * sym(rng), 1.3f * sym(rng), 1.3f * sym(rng) };
            const Float3 e{ 0.05f + 0.4f * unit(rng), 0.05f + 0.4f * unit(rng), 0.05f + 0.4f * unit(rng) };
            s.localBoundsMin = Float3{ c.x - e.x, c.y - e.y, c.z - e.z };
            s.localBoundsMax = Float3{ c.x + e.x, c.y + e.y, c.z + e.z };
            break;
        }
        }
        shapeSims[i] = s;
        nodeIndices[i] = makeNodeIndex(body);
        shapeIndices[i] = i;
    }

    const uint32_t nbUpdates = numBodies;
    std::vector<uint32_t> updateIndices(nbUpdates);
    std::iota(updateIndices.begin(), updateIndices.end(), 0u);
    std::shuffle(updateIndices.begin(), updateIndices.end(), rng);
    std::vector<float> poseData(nbUpdates * 7);
    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        const QuatV q = randomQuat(rng);
        poseData[i * 7 + 0] = q.x;
        poseData[i * 7 + 1] = q.y;
        poseData[i * 7 + 2] = q.z;
        poseData[i * 7 + 3] = q.w;
        poseData[i * 7 + 4] = 2.3f * sym(rng);
        poseData[i * 7 + 5] = 2.3f * sym(rng);
        poseData[i * 7 + 6] = 2.3f * sym(rng);
    }

    // ---- reference (CUDA order) and the re-associated dot2 twin
    std::vector<BodySim> refBodySims = bodySims, oldBodySims = bodySims;
    std::vector<Transform28> refCache(numShapes), oldCache(numShapes);
    std::vector<Bounds3> refB(numShapes, Bounds3{}), oldB(numShapes, Bounds3{});
    for (int variant = 0; variant < 2; ++variant)
    {
        const bool oldOrder = variant == 1;
        std::vector<BodySim>& bs = oldOrder ? oldBodySims : refBodySims;
        std::vector<Transform28>& cache = oldOrder ? oldCache : refCache;
        std::vector<Bounds3>& bounds = oldOrder ? oldB : refB;
        for (uint32_t i = 0; i < nbUpdates; ++i)
        {
            const uint32_t index = updateIndices[i];
            BodySim& bodySim = bs[index];
            const QuatV inQ(poseData[i * 7 + 0], poseData[i * 7 + 1], poseData[i * 7 + 2], poseData[i * 7 + 3]);
            const Vec3 inP(poseData[i * 7 + 4], poseData[i * 7 + 5], poseData[i * 7 + 6]);
            const QuatV b2aQ(bodySim.body2Actor_maxImpulseW.q);
            const Vec3 b2aP(bodySim.body2Actor_maxImpulseW.p.x, bodySim.body2Actor_maxImpulseW.p.y,
                            bodySim.body2Actor_maxImpulseW.p.z);
            const TransformV body2World(inQ * b2aQ, p90ub::rotate(inQ, b2aP, oldOrder) + inP);
            bodySim.body2World.q = body2World.q.toFloat4();
            bodySim.body2World.p = Float4{ body2World.p.x, body2World.p.y, body2World.p.z, 0.f };
            const TransformV body2Actor(b2aQ, b2aP);
            for (uint32_t s = index * shapesPerBody; s < (index + 1) * shapesPerBody; ++s)
            {
                const TransformV absPos =
                    p90ub::absPose(body2World, TransformV(shapeSims[s].transform), body2Actor, oldOrder);
                cache[s] = absPos.toTransform28();
                refUpdateBounds(shapeSims[s], {}, {}, absPos, bounds[s]);
            }
        }
    }
    // the CUDA-order reference must be the file's own getAbsPose (sanity of the twin helper)
    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        const uint32_t index = updateIndices[i];
        const BodySim& b = refBodySims[index];
        const TransformV b2w(QuatV(b.body2World.q), Vec3(b.body2World.p.x, b.body2World.p.y, b.body2World.p.z));
        const TransformV b2a(QuatV(b.body2Actor_maxImpulseW.q),
                             Vec3(b.body2Actor_maxImpulseW.p.x, b.body2Actor_maxImpulseW.p.y,
                                  b.body2Actor_maxImpulseW.p.z));
        const Transform28 t = getAbsPose(b2w, TransformV(shapeSims[index * shapesPerBody].transform), b2a)
                                  .toTransform28();
        PXG_CHECK_MSG(std::memcmp(&t, &refCache[index * shapesPerBody], sizeof(Transform28)) == 0,
                      "p90 setGlobalPose: twin helper (CUDA order) disagrees with getAbsPose on body %u", index);
    }

    // ---- coverage guard: the re-associated dot2 must change the bits
    uint32_t twinB2W = 0, twinCache = 0, twinBounds[4] = {};
    for (uint32_t n = 0; n < numBodies; ++n)
        twinB2W += std::memcmp(&refBodySims[n].body2World, &oldBodySims[n].body2World,
                               sizeof(refBodySims[n].body2World)) != 0 ? 1u : 0u;
    for (uint32_t s = 0; s < numShapes; ++s)
    {
        twinCache += std::memcmp(&refCache[s], &oldCache[s], sizeof(Transform28)) != 0 ? 1u : 0u;
        const float rw[6] = { refB[s].minimum.x, refB[s].minimum.y, refB[s].minimum.z,
                              refB[s].maximum.x, refB[s].maximum.y, refB[s].maximum.z };
        const float ow[6] = { oldB[s].minimum.x, oldB[s].minimum.y, oldB[s].minimum.z,
                              oldB[s].maximum.x, oldB[s].maximum.y, oldB[s].maximum.z };
        for (int c = 0; c < 6; ++c)
            twinBounds[s % shapesPerBody] += bits(rw[c]) != bits(ow[c]) ? 1u : 0u;
    }
    std::printf("[ info ] P9.90 setGlobalPose: old dot2 order changes %u/%u body2World, %u/%u cache entries, "
                "bound words %u/%u/%u/%u of %u (sphere/capsule/box/fast)\n",
                twinB2W, numBodies, twinCache, numShapes, twinBounds[0], twinBounds[1], twinBounds[2],
                twinBounds[3], numBodies * 6);
    // measured: 182/1024 body2World, 1719/4096 cache, 1089/1015/1111/997 of 6144 bound words
    PXG_CHECK_MSG(twinB2W > numBodies / 8 && twinCache > numShapes / 3,
                  "P9.90 coverage: the x*vx + (y*vy + z*vz) twin changes only %u body2World / %u cache entries",
                  twinB2W, twinCache);
    for (int p = 0; p < 4; ++p)
        PXG_CHECK_MSG(twinBounds[p] > numBodies / 2, "P9.90 coverage: the twin changes only %u bound words of path %d",
                      twinBounds[p], p);

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    Kernel kernel("updateBounds.spv", "setRigidDynamicGlobalPose",
                  { sizeof(ShapeSim), sizeof(Shape), sizeof(BodySim), 4, sizeof(SolverBodySleepData),
                    sizeof(CachedTransform), 4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4 },
                  sizeof(UpdateBoundsParams));

    PxgBufferHandle shapeSimBuf = createBufferFrom(dev, shapeSims);
    PxgBufferHandle dummyShapes = dev->createBuffer(sizeof(Shape));
    PxgBufferHandle bodySimBuf = createBufferFrom(dev, bodySims);
    PxgBufferHandle dummy4a = dev->createBuffer(16);
    PxgBufferHandle dummySleep = dev->createBuffer(16);
    std::vector<CachedTransform> cacheInit(numShapes);
    std::memset(cacheInit.data(), 0, cacheInit.size() * sizeof(CachedTransform));
    PxgBufferHandle cacheBuf = createBufferFrom(dev, cacheInit);
    std::vector<float> boundsInit(numShapes * 6, 0.f);
    PxgBufferHandle boundsBuf = createBufferFrom(dev, boundsInit);
    std::vector<uint32_t> zerosShapes(numShapes, 0);
    PxgBufferHandle dummyFrozen = dev->createBuffer(16);
    PxgBufferHandle dummyUnfrozen = dev->createBuffer(16);
    PxgBufferHandle updatedBuf = createBufferFrom(dev, zerosShapes);
    PxgBufferHandle dummyActive = dev->createBuffer(16);
    PxgBufferHandle dummyDeactivate = dev->createBuffer(16);
    PxgBufferHandle dummyBlob = dev->createBuffer(16);
    PxgBufferHandle nodeIdxBuf = createBufferFrom(dev, nodeIndices);
    PxgBufferHandle shapeIdxBuf = createBufferFrom(dev, shapeIndices);
    PxgBufferHandle poseBuf = createBufferFrom(dev, poseData);
    PxgBufferHandle gpuIdxBuf = createBufferFrom(dev, updateIndices);

    const UpdateBoundsParams params = { 0, nbUpdates, numShapes };
    kernel.dispatch((nbUpdates + 255) / 256,
                    { shapeSimBuf, dummyShapes, bodySimBuf, dummy4a, dummySleep, cacheBuf, boundsBuf, dummyFrozen,
                      dummyUnfrozen, updatedBuf, dummyActive, dummyDeactivate, dummyBlob, nodeIdxBuf, shapeIdxBuf,
                      poseBuf, gpuIdxBuf },
                    params);
    dev->flush();

    const std::vector<BodySim> gpuBodySims = downloadBuffer<BodySim>(dev, bodySimBuf, numBodies);
    const std::vector<CachedTransform> gpuCache = downloadBuffer<CachedTransform>(dev, cacheBuf, numShapes);
    const std::vector<float> gpuBounds = downloadBuffer<float>(dev, boundsBuf, numShapes * 6);
    const std::vector<uint32_t> gpuUpdated = downloadBuffer<uint32_t>(dev, updatedBuf, numShapes);
    uint32_t b2wBad = 0, b2wAsTwin = 0, cacheBad = 0, cacheAsTwin = 0, boundsBad = 0, updBad = 0;
    for (uint32_t n = 0; n < numBodies; ++n)
    {
        if (std::memcmp(&gpuBodySims[n].body2World, &refBodySims[n].body2World,
                        sizeof(refBodySims[n].body2World)) == 0)
            continue;
        ++b2wBad;
        b2wAsTwin += std::memcmp(&gpuBodySims[n].body2World, &oldBodySims[n].body2World,
                                 sizeof(oldBodySims[n].body2World)) == 0 ? 1u : 0u;
    }
    for (uint32_t s = 0; s < numShapes; ++s)
    {
        updBad += gpuUpdated[s] != 1u ? 1u : 0u;
        if (std::memcmp(&gpuCache[s].transform, &refCache[s], sizeof(Transform28)) != 0)
        {
            ++cacheBad;
            cacheAsTwin += std::memcmp(&gpuCache[s].transform, &oldCache[s], sizeof(Transform28)) == 0 ? 1u : 0u;
        }
        const float rb[6] = { refB[s].minimum.x, refB[s].minimum.y, refB[s].minimum.z,
                              refB[s].maximum.x, refB[s].maximum.y, refB[s].maximum.z };
        for (int c = 0; c < 6; ++c)
        {
            if (bits(gpuBounds[s * 6 + c]) == bits(rb[c]))
                continue;
            if (boundsBad < 4)
                std::printf("MM p90 setGlobalPose bounds[%u][%d]: gpu=%08x ref=%08x\n", s, c,
                            bits(gpuBounds[s * 6 + c]), bits(rb[c]));
            ++boundsBad;
        }
    }
    PXG_CHECK_MSG(updBad == 0, "p90 setGlobalPose: %u updated flags not set", updBad);
    PXG_CHECK_MSG(b2wBad == 0, "p90 setGlobalPose: %u body2World differ bitwise (%u equal the old-order twin)",
                  b2wBad, b2wAsTwin);
    PXG_CHECK_MSG(cacheBad == 0, "p90 setGlobalPose: %u transform cache entries differ bitwise (%u equal the twin)",
                  cacheBad, cacheAsTwin);
    PXG_CHECK_MSG(boundsBad == 0, "p90 setGlobalPose: %u bound words differ bitwise from the CUDA order", boundsBad);

    for (PxgBufferHandle h : { shapeSimBuf, dummyShapes, bodySimBuf, dummy4a, dummySleep, cacheBuf, boundsBuf,
                               dummyFrozen, dummyUnfrozen, updatedBuf, dummyActive, dummyDeactivate, dummyBlob,
                               nodeIdxBuf, shapeIdxBuf, poseBuf, gpuIdxBuf })
        dev->destroyBuffer(h);
}

// updateChangedAABBMgrHandlesLaunch on the ENGINE ABI (P5 Step 4 root-cause
// fix): pack the per-shape "updated" flags into the AABB-manager changed-handle
// bitmap (bit j of word w = updated[w*32+j] != 0).  Without this the broadphase
// never learns which shapes moved and early-outs forever.  ABI = one desc
// pointer {8}; verified bit-exact against a CPU reference of the __ballot pack.
PXG_TEST(updateBounds_changedAABBMgrHandles)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(0xB17B17);
    const uint32_t numWords = 37;                 // 37 words -> 1184 shape bits
    const uint32_t numElements = numWords * 32;

    std::vector<uint32_t> updated(numElements);
    for (uint32_t i = 0; i < numElements; ++i)
        updated[i] = (rng() & 3u) == 0u ? (rng() | 0u) : 0u;   // ~25% nonzero, arbitrary values

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "updateBounds.spv", "updateChangedAABBMgrHandlesLaunch", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &updateBoundsAbiLoader);

    PxgDevicePtr dUpdated = 0, dChanged = 0, dDesc = 0;
    shim.memAlloc(&dUpdated, numElements * 4);
    shim.memAlloc(&dChanged, numWords * 4);
    shim.memAlloc(&dDesc, sizeof(physx::PxgSimulationCoreDesc));
    if (!dUpdated || !dChanged || !dDesc) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    std::vector<uint32_t> changedInit(numWords, 0xDEADBEEFu);
    shim.memcpyHtoD(dUpdated, updated.data(), numElements * 4);
    shim.memcpyHtoD(dChanged, changedInit.data(), numWords * 4);

    physx::PxgSimulationCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.mUpdated = reinterpret_cast<physx::PxU32*>(dUpdated);
    desc.mChangedAABBMgrHandles = reinterpret_cast<physx::PxU32*>(dChanged);
    desc.mBitMapWordCounts = numWords;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 8, "engine blob is 8 bytes (got %zu)", blob.bytes.size());

    // engine grid: UPDATE_AABBMGR_HANDLES is a fixed grid; the kernel is
    // grid-strided over words, so any group count covers all words.
    PXG_CHECK_MSG(shim.launchKernel(0, 8, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success, "launch updateChangedAABBMgrHandlesLaunch");
    shim.streamSynchronize();

    // CPU reference: bit j of word w set iff updated[w*32+j] != 0
    std::vector<uint32_t> ref(numWords, 0);
    for (uint32_t w = 0; w < numWords; ++w)
        for (uint32_t j = 0; j < 32; ++j)
            if (updated[w * 32 + j] != 0u)
                ref[w] |= (1u << j);

    std::vector<uint32_t> got(numWords);
    shim.memcpyDtoH(got.data(), dChanged, numWords * 4);

    size_t bad = 0;
    for (uint32_t w = 0; w < numWords; ++w)
        if (got[w] != ref[w]) ++bad;
    PXG_CHECK_MSG(bad == 0, "changed-handle bitmap mismatch (%zu / %u words)", bad, numWords);

    shim.memFree(dUpdated);
    shim.memFree(dChanged);
    shim.memFree(dDesc);
}

// ---------------------------------------------------------------------------
// mergeTransformCacheAndBoundArrayChanges (P9.95): the direct-GPU API's
// between-steps merge of new / re-inserted shapes, on the ENGINE launch ABI.
// ---------------------------------------------------------------------------

static_assert(sizeof(physx::PxBounds3) == 24u, "PxBounds3 stride");
static_assert(sizeof(physx::PxsCachedTransform) == 32u, "PxsCachedTransform stride");
static_assert(offsetof(physx::PxsCachedTransform, flags) == 28u, "PxsCachedTransform::flags @28");
static_assert(sizeof(physx::PxgBoundTransformUpdate) == 8u, "PxgBoundTransformUpdate stride");
static_assert(offsetof(physx::PxgBoundTransformUpdate, indexFrom) == 4u, "PxgBoundTransformUpdate::indexFrom @4");

// The push block = the host pack of PxgSimulationController::mergeBoundsAndTransformsChanges.
// 5.12 inserted the PxgUpdateActorDataDesc* as argument #2.
struct MergeTcBoundPush
{
    uint64_t deviceBounds, deviceTransforms, updateActorDesc, boundsArray, transformsArray, changes;
    uint32_t numChanges, pad;
};
static_assert(sizeof(MergeTcBoundPush) == 56u, "blob = pxgKernelAbiPushSize({8,8,8,8,8,8,4})");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mUpdated) == 48u, "PxgUpdateActorDataDesc::mUpdated @48");

static const char* const MTC_SPV = "updateBounds.spv";
static const char* const MTC_KERNEL = "mergeTransformCacheAndBoundArrayChanges";
static const uint32_t MTC_BLOCK = 256u;   // PxgSimulationCoreKernelIndices.h:44
static const uint32_t MTC_SLOTS = 1024u;  // live slots per array
// A poisoned tail of as many slots again, inside the compared images: a mutant with a
// wrong stride then writes into a COMPARED region instead of faulting or corrupting
// the harness.
static const uint32_t MTC_ALL = 2u * MTC_SLOTS;
static const uint32_t MTC_NEW = 0x80000000u;   // indexFrom's MSB: copy from the CPU arrays
// The tail trap: record [numChanges], one past the last real record.  Slot 1000 is
// never an indexTo and never a source, so a lane that reads the trap shows up there.
static const uint32_t MTC_TRAP_TO = 1000u;
static const uint32_t MTC_TRAP_FROM = 600u;

// The engine grid (PxgSimulationController.cpp:756-757): ceil(numChanges / 256).
static uint32_t mtcGrid(uint32_t n) { return (n + MTC_BLOCK - 1u) / MTC_BLOCK; }

// updateTransformAndBoundArray.cu:29-57 (5.12), LITERALLY, in the real PhysX types,
// run over gridDim x 256 threads.  Sequential execution equals every parallel order
// here because no record set below has a write conflict: every indexTo is unique, no
// indexTo is a device-to-device source, and every device-to-device indexFrom is
// unique (the 5.12 mUpdated move zeroes it; see mtcRecords).
static void refMergeTcBound(physx::PxBounds3* PX_RESTRICT deviceBounds,
                            physx::PxsCachedTransform* PX_RESTRICT deviceTransforms,
                            physx::PxU32* PX_RESTRICT updated,   // updateActorDesc->mUpdated
                            const physx::PxBounds3* PX_RESTRICT boundsArray,
                            const physx::PxsCachedTransform* PX_RESTRICT transformsArray,
                            const physx::PxgBoundTransformUpdate* PX_RESTRICT changes,
                            const physx::PxU32 numChanges, uint32_t gridDim)
{
    for (uint32_t blockIdx = 0; blockIdx < gridDim; ++blockIdx)
        for (uint32_t threadIdx = 0; threadIdx < MTC_BLOCK; ++threadIdx)
        {
            int idx = int(blockIdx * MTC_BLOCK + threadIdx);
            if (idx < numChanges) {
                physx::PxU32 indexTo = changes[idx].indexTo;
                physx::PxU32 indexFrom = changes[idx].indexFrom & 0x7FFFFFFF;
                bool isNew = (changes[idx].indexFrom & (1U << 31)) != 0;
                if (isNew) {
                    deviceBounds[indexTo] = boundsArray[indexFrom];
                    deviceTransforms[indexTo] = transformsArray[indexFrom];
                } else {
                    deviceBounds[indexTo] = deviceBounds[indexFrom];
                    deviceTransforms[indexTo] = deviceTransforms[indexFrom];
                    updated[indexTo] = updated[indexFrom];
                    updated[indexFrom] = 0;
                }
            }
        }
}

// The wrong spellings the discrimination twins run (host only, never the oracle).
enum class MtcTwin { AlwaysCpu, AlwaysDevice, SwapToFrom, NoFlags, LeNoClamp, NoUpdatedMove };

static void twinMergeTcBound(MtcTwin t, physx::PxBounds3* deviceBounds, physx::PxsCachedTransform* deviceTransforms,
                             physx::PxU32* updated, const physx::PxBounds3* boundsArray, const physx::PxsCachedTransform* transformsArray,
                             const physx::PxgBoundTransformUpdate* changes, physx::PxU32 numChanges, uint32_t gridDim)
{
    for (uint32_t i = 0; i < gridDim * MTC_BLOCK; ++i)
    {
        // LeNoClamp: a `<=` guard with no clamp reads the record one past the end.
        if (!(t == MtcTwin::LeNoClamp ? i <= numChanges : i < numChanges))
            continue;
        physx::PxU32 indexTo = changes[i].indexTo;
        physx::PxU32 indexFrom = changes[i].indexFrom & 0x7FFFFFFFu;
        bool isNew = (changes[i].indexFrom & MTC_NEW) != 0u;
        if (t == MtcTwin::SwapToFrom) std::swap(indexTo, indexFrom);
        if (t == MtcTwin::AlwaysCpu) isNew = true;
        if (t == MtcTwin::AlwaysDevice) isNew = false;
        const physx::PxU32 oldFlags = deviceTransforms[indexTo].flags;
        deviceBounds[indexTo] = isNew ? boundsArray[indexFrom] : deviceBounds[indexFrom];
        deviceTransforms[indexTo] = isNew ? transformsArray[indexFrom] : deviceTransforms[indexFrom];
        if (t == MtcTwin::NoFlags) deviceTransforms[indexTo].flags = oldFlags;
        if (!isNew && t != MtcTwin::NoUpdatedMove)
        {
            updated[indexTo] = updated[indexFrom];
            updated[indexFrom] = 0;
        }
    }
}

// Source payloads, placed in EVERY word position across the population: a qNaN with a
// payload, a SIGNALLING NaN (a float move may quiet it), -0 (a float add of +0 turns it
// into +0), the smallest denormal (a flushing float path zeroes it), and both
// infinities.  The flags word @28 gets eFROZEN, all-ones and a pattern instead.
static const uint32_t MTC_PAYLOAD[6] = { 0x7FC12345u, 0x7F800001u, 0x80000000u, 0x00000001u, 0x7F800000u, 0xFF800000u };
static const uint32_t MTC_FLAGS[3] = { 1u, 0xFFFFFFFFu, 0xA5A5A5A5u };

// Every other word is a unique poison, tag 0xC (CPU) / 0xD (device), so an untouched
// or misplaced write is recognisable: bounds words carry w = 0..5, transform words
// w = 8..15.
static uint32_t mtcWord(uint32_t tag, uint32_t slot, uint32_t w, bool transform)
{
    if ((slot + w) % 3u == 0u)
        return (transform && w == 7u) ? MTC_FLAGS[(slot / 3u) % 3u] : MTC_PAYLOAD[(slot / 3u + w) % 6u];
    return (tag << 28) | (slot << 4) | (transform ? 8u + w : w);
}

struct MtcWorld
{
    std::vector<physx::PxBounds3> devBounds, cpuBounds;
    std::vector<physx::PxsCachedTransform> devXf, cpuXf;
    std::vector<physx::PxU32> updated;                     // updateActorDesc->mUpdated, per shape slot
    std::vector<physx::PxgBoundTransformUpdate> changes;   // numChanges real records + the tail trap
    uint32_t numChanges = 0;
};

static void mtcFill(MtcWorld& w)
{
    w.devBounds.resize(MTC_ALL); w.cpuBounds.resize(MTC_ALL);
    w.devXf.resize(MTC_ALL); w.cpuXf.resize(MTC_ALL);
    // mUpdated: a unique non-zero pattern per slot (the engine stores 0 / 1; any
    // word must move as is), so a lost move, a lost clear or a wrong slot shows
    w.updated.resize(MTC_ALL);
    for (uint32_t slot = 0; slot < MTC_ALL; ++slot) w.updated[slot] = 0xE0000000u | (slot << 4) | 7u;
    for (uint32_t slot = 0; slot < MTC_ALL; ++slot)
    {
        uint32_t b[6], x[8];
        for (uint32_t k = 0; k < 6u; ++k) b[k] = mtcWord(0xDu, slot, k, false);
        for (uint32_t k = 0; k < 8u; ++k) x[k] = mtcWord(0xDu, slot, k, true);
        std::memcpy(&w.devBounds[slot], b, 24); std::memcpy(&w.devXf[slot], x, 32);
        for (uint32_t k = 0; k < 6u; ++k) b[k] = mtcWord(0xCu, slot, k, false);
        for (uint32_t k = 0; k < 8u; ++k) x[k] = mtcWord(0xCu, slot, k, true);
        std::memcpy(&w.cpuBounds[slot], b, 24); std::memcpy(&w.cpuXf[slot], x, 32);
    }
}

// nSame isNew records with from == to (PxgBoundsArray::updateBounds / setBounds for a
// new shape), nDiff isNew with from != to (legal for the kernel; it separates the two
// source arrays), nDev device-to-device (reinsertBroadPhase's {new id, old id}).
//   indexTo: unique, from [0,512) U {1023}; 1000 never.
//   device-to-device indexFrom: unique, [512,1023) \ {1000} -- disjoint from every indexTo.
//   isNew indexFrom: anything in [0,1024) (a CPU slot is never written).
// Required, when the class exists: an isNew record to 0 / from 0 (raw word 0x80000000,
// the one a mask that reads it as "from 0, not new" gets wrong), and an isNew record
// to 1023 whose from is 0 as well.
static std::vector<physx::PxgBoundTransformUpdate> mtcRecords(std::mt19937& rng, uint32_t nSame, uint32_t nDiff,
                                                             uint32_t nDev)
{
    const uint32_t n = nSame + nDiff + nDev;
    std::vector<uint32_t> pool;
    for (uint32_t t = 1; t < 512u; ++t) pool.push_back(t);
    std::shuffle(pool.begin(), pool.end(), rng);
    size_t next = 0;
    auto takeTo = [&]() { return pool[next++]; };
    std::uniform_int_distribution<uint32_t> anySlot(0u, MTC_SLOTS - 1u);
    // device-to-device sources: unique (5.12 zeroes mUpdated[indexFrom], so a shared
    // source would make the result depend on the lane order)
    std::vector<uint32_t> devFromPool;
    for (uint32_t f = 512u; f < 1023u; ++f)
        if (f != MTC_TRAP_TO) devFromPool.push_back(f);
    std::shuffle(devFromPool.begin(), devFromPool.end(), rng);

    std::vector<physx::PxgBoundTransformUpdate> r;
    for (uint32_t i = 0; i < nSame; ++i)
    {
        const uint32_t to = (i == 0u) ? 0u : takeTo();
        r.push_back({ to, to | MTC_NEW });
    }
    for (uint32_t i = 0; i < nDiff; ++i)
    {
        const uint32_t to = (i == 0u) ? 1023u : takeTo();
        uint32_t from = (i == 0u) ? 0u : anySlot(rng);
        while (from == to) from = anySlot(rng);
        r.push_back({ to, from | MTC_NEW });
    }
    for (uint32_t i = 0; i < nDev; ++i)
        r.push_back({ takeTo(), devFromPool[i] });
    std::shuffle(r.begin(), r.end(), rng);
    PXG_CHECK_MSG(r.size() == n, "record count");
    r.push_back({ MTC_TRAP_TO, MTC_TRAP_FROM | MTC_NEW });   // the tail trap, at [numChanges]
    return r;
}

template <typename T>
static std::vector<uint32_t> mtcWords(const std::vector<T>& v)
{
    std::vector<uint32_t> w(v.size() * sizeof(T) / 4u);
    std::memcpy(w.data(), v.data(), w.size() * 4u);
    return w;
}

// Bitwise compare of two word images; prints the first 8 mismatches.
static uint32_t mtcCompare(const char* what, const char* array, const std::vector<uint32_t>& got,
                           const std::vector<uint32_t>& want, uint32_t wordsPerSlot)
{
    uint32_t bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (got[i] != want[i] && bad++ < 8u)
            std::printf("    %s %s: slot %zu word %zu: got 0x%08x, expected 0x%08x\n", what, array,
                        i / wordsPerSlot, i % wordsPerSlot, got[i], want[i]);
    return bad;
}

// Runs the oracle over a copy of `w` and returns the expected images.
static MtcWorld mtcExpected(const MtcWorld& w)
{
    MtcWorld e = w;
    refMergeTcBound(e.devBounds.data(), e.devXf.data(), e.updated.data(), e.cpuBounds.data(), e.cpuXf.data(), e.changes.data(),
                    w.numChanges, mtcGrid(w.numChanges));
    return e;
}

// The oracle must itself be a bit-exact move: every record's destination equals its
// source word for word (this is what proves the host copy keeps the sNaN payload,
// -0 and the denormal), and the copied words include each payload at least once.
static void mtcCheckOracle(const char* what, const MtcWorld& w, const MtcWorld& e)
{
    uint32_t moved[6] = { 0, 0, 0, 0, 0, 0 }, bad = 0;
    for (uint32_t i = 0; i < w.numChanges; ++i)
    {
        const uint32_t to = w.changes[i].indexTo, from = w.changes[i].indexFrom & 0x7FFFFFFFu;
        const bool isNew = (w.changes[i].indexFrom & MTC_NEW) != 0u;
        uint32_t src[14], dst[14];
        std::memcpy(src, isNew ? &w.cpuBounds[from] : &w.devBounds[from], 24);
        std::memcpy(src + 6, isNew ? &w.cpuXf[from] : &w.devXf[from], 32);
        std::memcpy(dst, &e.devBounds[to], 24);
        std::memcpy(dst + 6, &e.devXf[to], 32);
        for (uint32_t k = 0; k < 14u; ++k)
        {
            if (src[k] != dst[k]) ++bad;
            for (uint32_t p = 0; p < 6u; ++p)
                if (k != 13u && src[k] == MTC_PAYLOAD[p]) ++moved[p];
        }
    }
    PXG_CHECK_MSG(bad == 0, "%s: the host oracle changed %u copied words (it must be a bit-exact move)", what, bad);
    if (w.numChanges >= 100u)
        for (uint32_t p = 0; p < 6u; ++p)
            PXG_CHECK_MSG(moved[p] > 0u, "%s: payload 0x%08x is never copied", what, MTC_PAYLOAD[p]);
}

static uint32_t mtcCompareWorld(const char* what, const MtcWorld& got, const MtcWorld& want)
{
    uint32_t bad = 0;
    bad += mtcCompare(what, "device bounds", mtcWords(got.devBounds), mtcWords(want.devBounds), 6u);
    bad += mtcCompare(what, "device transforms", mtcWords(got.devXf), mtcWords(want.devXf), 8u);
    bad += mtcCompare(what, "CPU bounds", mtcWords(got.cpuBounds), mtcWords(want.cpuBounds), 6u);
    bad += mtcCompare(what, "CPU transforms", mtcWords(got.cpuXf), mtcWords(want.cpuXf), 8u);
    bad += mtcCompare(what, "changes", mtcWords(got.changes), mtcWords(want.changes), 2u);
    bad += mtcCompare(what, "updated", got.updated, want.updated, 1u);
    return bad;
}

// mergeTransformCacheAndBoundArrayChanges through the framework: one 56-byte push
// block packed in the order of PxgSimulationController.cpp:752-754, every array a
// device buffer.  Three launches at the engine's grid ceil(n / 256):
//   L1  n = 300  grid 2, the second workgroup partly filled (lanes 300..511 idle);
//   L2  n = 256  grid 1, exactly full;
//   L3  n = 1    grid 1, the scene's lane-D shape (one device-to-device record).
// Each compares every word of all five arrays (device arrays INCLUDING their poisoned
// tails, the CPU arrays and the records, which must be unchanged) against the literal
// oracle above.  Before the GPU runs, five wrong spellings are run on the host over
// L1's data and each must change at least one word: proof that the data can see them.
PXG_TEST(updateBounds_mergeTcBound)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::mt19937 rng(9501u);

    struct Launch { const char* name; uint32_t nSame, nDiff, nDev; };
    const Launch launches[3] = { { "L1 n=300", 100u, 60u, 140u },
                                 { "L2 n=256", 80u, 50u, 126u },
                                 { "L3 n=1", 0u, 0u, 1u } };

    Kernel k(MTC_SPV, MTC_KERNEL, {}, uint32_t(sizeof(MergeTcBoundPush)));
    for (const Launch& L : launches)
    {
        MtcWorld w;
        mtcFill(w);
        w.changes = mtcRecords(rng, L.nSame, L.nDiff, L.nDev);
        w.numChanges = L.nSame + L.nDiff + L.nDev;
        const MtcWorld e = mtcExpected(w);
        mtcCheckOracle(L.name, w, e);

        if (&L == &launches[0])
        {
            const MtcTwin twins[6] = { MtcTwin::AlwaysCpu, MtcTwin::AlwaysDevice, MtcTwin::SwapToFrom,
                                       MtcTwin::NoFlags, MtcTwin::LeNoClamp, MtcTwin::NoUpdatedMove };
            const char* twinNames[6] = { "always-CPU", "always-device", "swapped to/from", "flags not copied",
                                         "<= without clamp (reads the tail trap)", "5.6.1: mUpdated not moved" };
            for (uint32_t t = 0; t < 6u; ++t)
            {
                MtcWorld m = w;
                twinMergeTcBound(twins[t], m.devBounds.data(), m.devXf.data(), m.updated.data(), m.cpuBounds.data(),
                                 m.cpuXf.data(), m.changes.data(), w.numChanges, mtcGrid(w.numChanges));
                const bool differs = mtcWords(m.devBounds) != mtcWords(e.devBounds) ||
                                     mtcWords(m.devXf) != mtcWords(e.devXf) || m.updated != e.updated;
                PXG_CHECK_MSG(differs, "discrimination twin '%s' leaves L1's image unchanged", twinNames[t]);
            }
        }

        std::vector<PxgBufferHandle> all;
        auto mk = [&](auto const& v) { PxgBufferHandle h = createBufferFrom(dev, v); all.push_back(h); return h; };
        const PxgBufferHandle bDevB = mk(w.devBounds), bDevX = mk(w.devXf), bCpuB = mk(w.cpuBounds),
                              bCpuX = mk(w.cpuXf), bChg = mk(w.changes), bUpd = mk(w.updated);
        if (!dev->getBufferDeviceAddress(bDevB))
        {
            std::printf("  NOTE: no BDA -> skip\n");
            for (PxgBufferHandle h : all) dev->destroyBuffer(h);
            return;
        }
        // the engine's device copy of PxgUpdateActorDataDesc; only mUpdated is read
        physx::PxgUpdateActorDataDesc desc;
        std::memset(&desc, 0, sizeof(desc));
        desc.mUpdated = reinterpret_cast<physx::PxU32*>(dev->getBufferDeviceAddress(bUpd));
        std::vector<uint8_t> descBytes(sizeof(desc));
        std::memcpy(descBytes.data(), &desc, sizeof(desc));
        const PxgBufferHandle bDesc = mk(descBytes);
        MergeTcBoundPush p = {};
        p.deviceBounds = dev->getBufferDeviceAddress(bDevB);
        p.deviceTransforms = dev->getBufferDeviceAddress(bDevX);
        p.updateActorDesc = dev->getBufferDeviceAddress(bDesc);
        p.boundsArray = dev->getBufferDeviceAddress(bCpuB);
        p.transformsArray = dev->getBufferDeviceAddress(bCpuX);
        p.changes = dev->getBufferDeviceAddress(bChg);
        p.numChanges = w.numChanges;
        k.dispatch(mtcGrid(w.numChanges), {}, p);
        dev->flush();

        MtcWorld g;
        g.devBounds = downloadBuffer<physx::PxBounds3>(dev, bDevB, MTC_ALL);
        g.devXf = downloadBuffer<physx::PxsCachedTransform>(dev, bDevX, MTC_ALL);
        g.cpuBounds = downloadBuffer<physx::PxBounds3>(dev, bCpuB, MTC_ALL);
        g.cpuXf = downloadBuffer<physx::PxsCachedTransform>(dev, bCpuX, MTC_ALL);
        g.changes = downloadBuffer<physx::PxgBoundTransformUpdate>(dev, bChg, w.changes.size());
        g.updated = downloadBuffer<physx::PxU32>(dev, bUpd, MTC_ALL);
        for (PxgBufferHandle h : all) dev->destroyBuffer(h);

        const uint32_t bad = mtcCompareWorld(L.name, g, e);
        PXG_CHECK_MSG(bad == 0, "%s (grid %u): %u words differ bitwise from updateTransformAndBoundArray.cu:29-57",
                      L.name, mtcGrid(w.numChanges), bad);
    }
}

// The engine path with PINNED host arrays: exactly what mergeBoundsAndTransformsChanges
// hands the kernel.  The CPU bounds, CPU transforms and the records come from
// memHostAlloc, are filled through their CPU pointers and are passed as
// hostGetDevicePointer(...) -- what getMappedDevicePtr does (PxgCudaUtils.h ->
// memHostGetDevicePointer).  The device arrays come from memAlloc.  The launch goes
// through PxgCudaShim with the push size taken from the bridge's launch-ABI row, so
// a row of the wrong size refuses the launch here.
//
// What this proves: the kernel reads pinned memory through the translated address and
// never writes it.  What it cannot prove: anything about the row's hostPtrMask.  The
// harness bypasses the bridge's translateHostPtrs, and on llvmpipe a pinned
// allocation's device address equals its CPU address, so a mask would be inert anyway.
PXG_TEST(updateBounds_mergeTcBound_pinned)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    const physx::gpuslang::PxgKernelAbiRow* row = physx::gpuslang::pxgFindKernelAbi(MTC_KERNEL);
    const uint32_t rowPush = row ? physx::gpuslang::pxgKernelAbiPushSize(*row) : 0u;
    PXG_CHECK_MSG(row != nullptr && row->argCount == 7u && rowPush == 56u,
                  "launch-ABI row {8,8,8,8,8,8,4} -> 56-byte blob (got %u args, %u bytes)",
                  row ? row->argCount : 0u, rowPush);
    PXG_CHECK_MSG(row == nullptr || row->hostPtrMask == 0u,
                  "hostPtrMask must be 0: the host arrays arrive already translated by getMappedDevicePtr");

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { MTC_SPV, MTC_KERNEL, rowPush, 0u, MTC_KERNEL } };
    PxgCudaShim shim(dev, manifest, 1, &updateBoundsAbiLoader);
    std::mt19937 rng(9502u);

    struct Launch { const char* name; uint32_t nSame, nDiff, nDev; };
    const Launch launches[2] = { { "pinned n=5", 2u, 1u, 2u }, { "pinned n=257", 90u, 50u, 117u } };
    for (const Launch& L : launches)
    {
        MtcWorld w;
        mtcFill(w);
        w.changes = mtcRecords(rng, L.nSame, L.nDiff, L.nDev);
        w.numChanges = L.nSame + L.nDiff + L.nDev;
        const MtcWorld e = mtcExpected(w);
        mtcCheckOracle(L.name, w, e);

        const size_t bBytes = w.cpuBounds.size() * 24u, xBytes = w.cpuXf.size() * 32u, cBytes = w.changes.size() * 8u;
        const size_t uBytes = w.updated.size() * 4u;
        PxgDevicePtr dDevB = 0, dDevX = 0, dUpd = 0, dDesc = 0;
        void* hCpuB = nullptr;
        void* hCpuX = nullptr;
        void* hChg = nullptr;
        shim.memAlloc(&dDevB, bBytes);
        shim.memAlloc(&dDevX, xBytes);
        shim.memAlloc(&dUpd, uBytes);
        shim.memAlloc(&dDesc, sizeof(physx::PxgUpdateActorDataDesc));
        shim.memHostAlloc(&hCpuB, bBytes);
        shim.memHostAlloc(&hCpuX, xBytes);
        shim.memHostAlloc(&hChg, cBytes);
        const PxgDevicePtr dCpuB = hCpuB ? shim.hostGetDevicePointer(hCpuB) : 0;
        const PxgDevicePtr dCpuX = hCpuX ? shim.hostGetDevicePointer(hCpuX) : 0;
        const PxgDevicePtr dChg = hChg ? shim.hostGetDevicePointer(hChg) : 0;
        if (!dDevB || !dDevX || !dUpd || !dDesc || !dCpuB || !dCpuX || !dChg)
        {
            std::printf("  NOTE: no BDA / pinned memory -> skip\n");
            if (dDevB) shim.memFree(dDevB);
            if (dDevX) shim.memFree(dDevX);
            if (dUpd) shim.memFree(dUpd);
            if (dDesc) shim.memFree(dDesc);
            if (hCpuB) shim.memFreeHost(hCpuB);
            if (hCpuX) shim.memFreeHost(hCpuX);
            if (hChg) shim.memFreeHost(hChg);
            return;
        }
        std::memcpy(hCpuB, w.cpuBounds.data(), bBytes);
        std::memcpy(hCpuX, w.cpuXf.data(), xBytes);
        std::memcpy(hChg, w.changes.data(), cBytes);
        shim.memcpyHtoD(dDevB, w.devBounds.data(), bBytes);
        shim.memcpyHtoD(dDevX, w.devXf.data(), xBytes);
        shim.memcpyHtoD(dUpd, w.updated.data(), uBytes);
        physx::PxgUpdateActorDataDesc desc;   // the device copy; only mUpdated is read
        std::memset(&desc, 0, sizeof(desc));
        desc.mUpdated = reinterpret_cast<physx::PxU32*>(dUpd);
        shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

        EngineBlob blob;   // PX_CUDA_KERNEL_PARAM order of mergeBoundsAndTransformsChanges
        blob.add<uint64_t>(dDevB);
        blob.add<uint64_t>(dDevX);
        blob.add<uint64_t>(dDesc);
        blob.add<uint64_t>(dCpuB);
        blob.add<uint64_t>(dCpuX);
        blob.add<uint64_t>(dChg);
        blob.add<uint32_t>(w.numChanges);
        blob.finish();
        PXG_CHECK_MSG(blob.bytes.size() == 56u, "engine blob is 56 bytes (got %zu)", blob.bytes.size());
        PXG_CHECK_MSG(shim.launchKernel(0, mtcGrid(w.numChanges), 1, 1, blob.bytes.data(),
                                        uint32_t(blob.bytes.size())) == PxgResult::Success,
                      "%s: launch %s", L.name, MTC_KERNEL);
        shim.streamSynchronize();

        MtcWorld g = w;
        shim.memcpyDtoH(g.devBounds.data(), dDevB, bBytes);
        shim.memcpyDtoH(g.devXf.data(), dDevX, xBytes);
        shim.memcpyDtoH(g.updated.data(), dUpd, uBytes);
        std::memcpy(g.cpuBounds.data(), hCpuB, bBytes);   // re-read through the CPU pointers
        std::memcpy(g.cpuXf.data(), hCpuX, xBytes);
        std::memcpy(g.changes.data(), hChg, cBytes);
        shim.memFree(dDevB);
        shim.memFree(dDevX);
        shim.memFree(dUpd);
        shim.memFree(dDesc);
        shim.memFreeHost(hCpuB);
        shim.memFreeHost(hCpuX);
        shim.memFreeHost(hChg);

        const uint32_t bad = mtcCompareWorld(L.name, g, e);
        PXG_CHECK_MSG(bad == 0, "%s (grid %u): %u words differ bitwise from updateTransformAndBoundArray.cu:29-57",
                      L.name, mtcGrid(w.numChanges), bad);
    }
}

} // namespace
