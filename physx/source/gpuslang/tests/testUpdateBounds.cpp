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

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the engine desc header
#include "PxgSimulationCoreDesc.h"
#include "PxgArticulation.h"

#include <numeric>
#include <random>

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
    const TransformV t0 = body2Actor.transformInv(shape2Actor);
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

void transformFastBounds(const TransformV& pose, const Vec3& localMin, const Vec3& localMax, Bounds3& out)
{
    const Mat33 rot(pose.q);
    const Vec3 center = (localMax + localMin) * 0.5f;
    const Vec3 extents = (localMax - localMin) * 0.5f;
    const Vec3 newCenter = rot.transform(center) + pose.p;
    const Vec3 newExtents = basisExtent(rot, extents);
    out.minimum = (newCenter - newExtents).toFloat3();
    out.maximum = (newCenter + newExtents).toFloat3();
}

Mat33 meshScaleToMat33(const Vec3& scale, const QuatV& rotation)
{
    const Mat33 rot(rotation);
    Mat33 diagTimesRot(rot.col0.multiply(scale), rot.col1.multiply(scale), rot.col2.multiply(scale));
    return rot.getTranspose() * diagTimesRot;
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
void refUpdateBounds(const ShapeSim& shapeSim, const std::vector<Shape>& shapes,
                     const std::vector<uint32_t>& geometryBlob, const TransformV& pose, Bounds3& out,
                     uint64_t blobDeviceBase = 0, UbFMinMaxTwin* twin = nullptr)
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
            const bool identity = scale.x == 1.f && scale.y == 1.f && scale.z == 1.f && scaleRot.x == 0.f &&
                                  scaleRot.y == 0.f && scaleRot.z == 0.f;
            if (!identity)
                rot = rot * meshScaleToMat33(scale, scaleRot);

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
            transformFastBounds(pose, localMin, localMax, out);
        }
        break;
    }
    default:
        transformFastBounds(pose, localMin, localMax, out);
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
                     std::vector<UbFMinMaxTwin>* twins = nullptr)
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
                                    twins ? &(*twins)[i] : nullptr);
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
                                twins ? &(*twins)[i] : nullptr);

            if (internalFlags & eACTIVATE_THIS_FRAME)
                out.active[bodySimIndex] = 1;
            else if (internalFlags & eDEACTIVATE_THIS_FRAME)
                out.deactivate[bodySimIndex] = 1;
        }
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
    const uint32_t numShapes = numRandomShapes + numDirShapes;
    const uint32_t maxNodes = 256;    // nodes [firstArtNode, firstArtNode+12) are articulation links
    const uint32_t numHulls = 8;      // random hulls (rollGeometry draws from these)
    const uint32_t numAllHulls = numHulls + numDirShapes;
    const uint32_t nbArts = 3;
    const uint32_t maxLinks = 4;
    const uint32_t firstArtNode = 240;
    const uint32_t dirNode = 252;     // unused by the fleet (rigid [0,240), links [240,252))

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
    for (uint32_t h = numHulls; h < numAllHulls; ++h)
    {
        shapes[h].scale = Float3{ 1.f, 1.f, 1.f }; // identity mesh scale: no scale multiply
        shapes[h].scaleRotation = Quat{ 0, 0, 0, 1 };
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

    // engine grid (256 x 256 = one pass over 602 shapes) and a 2-block grid
    // (512 threads -> the CUDA grid-stride wrap executes; threads 88/89 pick
    // up the directed shapes 600/601)
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

            const Transform28& gt = gpuCache[i].transform;
            const Transform28& rt = ref.transformCache[i].transform;
            PXG_CHECK_MSG(almostEqual(gt.q.x, rt.q.x, 1e-4f, 1e-5f) && almostEqual(gt.q.y, rt.q.y, 1e-4f, 1e-5f) &&
                              almostEqual(gt.q.z, rt.q.z, 1e-4f, 1e-5f) && almostEqual(gt.q.w, rt.q.w, 1e-4f, 1e-5f),
                          "grid %u cache.q[%u]: gpu=(%g,%g,%g,%g) ref=(%g,%g,%g,%g)", gridDim, i, gt.q.x,
                          gt.q.y, gt.q.z, gt.q.w, rt.q.x, rt.q.y, rt.q.z, rt.q.w);
            PXG_CHECK_MSG(almostEqual(gt.p.x, rt.p.x, 1e-4f, 1e-4f) && almostEqual(gt.p.y, rt.p.y, 1e-4f, 1e-4f) &&
                              almostEqual(gt.p.z, rt.p.z, 1e-4f, 1e-4f),
                          "grid %u cache.p[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", gridDim, i, gt.p.x, gt.p.y,
                          gt.p.z, rt.p.x, rt.p.y, rt.p.z);

            const float* gb = &gpuBounds[i * 6];
            const float rb[6] = { ref.bounds[i].minimum.x, ref.bounds[i].minimum.y, ref.bounds[i].minimum.z,
                                  ref.bounds[i].maximum.x, ref.bounds[i].maximum.y, ref.bounds[i].maximum.z };
            for (int c = 0; c < 6; ++c)
                PXG_CHECK_MSG(almostEqual(gb[c], rb[c], 1e-4f, 1e-3f), "grid %u bounds[%u][%d]: gpu=%g ref=%g",
                              gridDim, i, c, gb[c], rb[c]);

            // P9.86c: the directed shapes are compared BITWISE (almostEqual
            // treats +0 and -0 as equal): all six bound words and the cache.
            if (i >= numRandomShapes)
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
        // the mirror's transformInv uses rotateInv; the CUDA's PxTransform::
        // transformInv is getConjugate().rotate -- pin that both agree on this input
        const TransformV t0a = body2Actor.transformInv(shape2Actor);
        const Vec3 t0bP = b2aQ.getConjugate().rotate(shape2Actor.p - b2aP);
        PXG_CHECK_MSG(bits(t0a.p.x) == bits(t0bP.x) && bits(t0a.p.y) == bits(t0bP.y) &&
                          bits(t0a.p.z) == bits(t0bP.z),
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

} // namespace
