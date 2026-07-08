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
// updateTransformCacheAndBoundArrayLaunch (rigid-body path) +
// updateCacheAndBound.cuh: world pose composition, transform cache refresh,
// frozen/unfrozen/updated/activate/deactivate bookkeeping and world-space
// AABB computation for sphere / capsule / box / convex geometry.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"

#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

struct UpdateBoundsParams
{
    uint32_t numShapes;
};

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

void refUpdateBounds(const ShapeSim& shapeSim, const std::vector<Shape>& shapes,
                     const std::vector<uint32_t>& geometryBlob, const TransformV& pose, Bounds3& out)
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

            const uint32_t blobBase = uint32_t(shape.hullOrMeshOffset / 4);
            const uint32_t tmpX = geometryBlob[blobBase + 4];
            const uint32_t nbHullVertices = (tmpX >> 8) & 0xFFu;

            Vec3 minV(3.402823466e+38f);
            Vec3 maxV(-3.402823466e+38f);
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
            }
            out.minimum = (minV + pose.p).toFloat3();
            out.maximum = (maxV + pose.p).toFloat3();
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

void referenceUpdate(uint32_t numShapes, const std::vector<ShapeSim>& shapeSims,
                     const std::vector<Shape>& shapes, const std::vector<uint32_t>& geometryBlob,
                     const std::vector<BodySim>& bodySims, const std::vector<uint32_t>& bodyDataIndices,
                     const std::vector<SolverBodySleepData>& sleepData, uint32_t maxNodes, RefOutputs& out)
{
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

        if (isArticulation)
            continue; // articulation branch not ported yet

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
                refUpdateBounds(shapeSim, shapes, geometryBlob, absPos, out.bounds[i]);
        }

        if (internalFlags & eACTIVATE_THIS_FRAME)
            out.active[bodySimIndex] = 1;
        else if (internalFlags & eDEACTIVATE_THIS_FRAME)
            out.deactivate[bodySimIndex] = 1;
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

PXG_TEST(updateBounds_allGeometryTypes)
{
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-10.f, 10.f);

    const uint32_t numShapes = 600;
    const uint32_t maxNodes = 256;
    const uint32_t numHulls = 8;

    // ---- convex geometry blob: numHulls hulls, header (48 bytes) + up to 64 float4 vertices
    std::vector<uint32_t> geometryBlob;
    std::vector<uint64_t> hullOffsets(numHulls);
    std::vector<uint32_t> hullVertexCounts(numHulls);
    for (uint32_t h = 0; h < numHulls; ++h)
    {
        const uint32_t base = uint32_t(geometryBlob.size());
        hullOffsets[h] = uint64_t(base) * 4; // byte offset
        const uint32_t nbVerts = 4 + rng() % 60;
        hullVertexCounts[h] = nbVerts;

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

    // ---- shapes (PxgShape pool: convex hull entries)
    std::vector<Shape> shapes(numHulls);
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
        s.hullOrMeshOffset = hullOffsets[h];
        s.type = eCONVEXMESH;
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

    // ---- shape sims
    std::vector<ShapeSim> shapeSims(numShapes);
    std::memset(shapeSims.data(), 0, shapeSims.size() * sizeof(ShapeSim));
    for (uint32_t i = 0; i < numShapes; ++i)
    {
        ShapeSim& s = shapeSims[i];
        const QuatV q = randomQuat(rng);
        s.transform.q = q.toQuat();
        s.transform.p = Float3{ sym(rng) * 0.2f, sym(rng) * 0.2f, sym(rng) * 0.2f };

        // 5% static
        s.bodySimIndex = unit(rng) < 0.05f ? makeNodeIndex(PX_INVALID_NODE) : makeNodeIndex(rng() % maxNodes);
        s.shapeFlags = uint16_t((unit(rng) < 0.8f ? eSIMULATION_SHAPE : 0) |
                                (unit(rng) < 0.5f ? eSCENE_QUERY_SHAPE : 0) |
                                (unit(rng) < 0.1f ? eTRIGGER_SHAPE : 0));

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
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();

    Kernel kernel("updateBounds.spv", "updateTransformCacheAndBoundArrayLaunch",
                  { sizeof(ShapeSim), sizeof(Shape), sizeof(BodySim), 4, sizeof(SolverBodySleepData),
                    sizeof(CachedTransform), 4, 4, 4, 4, 4, 4, 4 },
                  sizeof(UpdateBoundsParams));

    PxgBufferHandle shapeSimBuf = createBufferFrom(dev, shapeSims);
    PxgBufferHandle shapesBuf = createBufferFrom(dev, shapes);
    PxgBufferHandle bodySimBuf = createBufferFrom(dev, bodySims);
    PxgBufferHandle bodyIdxBuf = createBufferFrom(dev, bodyDataIndices);
    PxgBufferHandle sleepBuf = createBufferFrom(dev, sleepData);
    std::vector<CachedTransform> cacheInit(numShapes);
    std::memset(cacheInit.data(), 0, cacheInit.size() * sizeof(CachedTransform));
    PxgBufferHandle cacheBuf = createBufferFrom(dev, cacheInit);
    std::vector<float> boundsInit(numShapes * 6, 0.f);
    PxgBufferHandle boundsBuf = createBufferFrom(dev, boundsInit);
    std::vector<uint32_t> zerosShapes(numShapes, 0), zerosNodes(maxNodes, 0);
    PxgBufferHandle frozenBuf = createBufferFrom(dev, zerosShapes);
    PxgBufferHandle unfrozenBuf = createBufferFrom(dev, zerosShapes);
    PxgBufferHandle updatedBuf = createBufferFrom(dev, zerosShapes);
    PxgBufferHandle activeBuf = createBufferFrom(dev, zerosNodes);
    PxgBufferHandle deactivateBuf = createBufferFrom(dev, zerosNodes);
    PxgBufferHandle blobBuf = createBufferFrom(dev, geometryBlob);

    const UpdateBoundsParams params = { numShapes };
    const uint32_t blockSize = 256;
    kernel.dispatch((numShapes + blockSize - 1) / blockSize,
                    { shapeSimBuf, shapesBuf, bodySimBuf, bodyIdxBuf, sleepBuf, cacheBuf, boundsBuf,
                      frozenBuf, unfrozenBuf, updatedBuf, activeBuf, deactivateBuf, blobBuf },
                    params);
    dev->flush();

    // ---- reference ----
    RefOutputs ref;
    referenceUpdate(numShapes, shapeSims, shapes, geometryBlob, bodySims, bodyDataIndices, sleepData,
                    maxNodes, ref);

    // ---- compare ----
    const std::vector<CachedTransform> gpuCache = downloadBuffer<CachedTransform>(dev, cacheBuf, numShapes);
    const std::vector<float> gpuBounds = downloadBuffer<float>(dev, boundsBuf, numShapes * 6);
    const std::vector<uint32_t> gpuFrozen = downloadBuffer<uint32_t>(dev, frozenBuf, numShapes);
    const std::vector<uint32_t> gpuUnfrozen = downloadBuffer<uint32_t>(dev, unfrozenBuf, numShapes);
    const std::vector<uint32_t> gpuUpdated = downloadBuffer<uint32_t>(dev, updatedBuf, numShapes);
    const std::vector<uint32_t> gpuActive = downloadBuffer<uint32_t>(dev, activeBuf, maxNodes);
    const std::vector<uint32_t> gpuDeactivate = downloadBuffer<uint32_t>(dev, deactivateBuf, maxNodes);

    for (uint32_t i = 0; i < numShapes; ++i)
    {
        PXG_CHECK_MSG(gpuFrozen[i] == ref.frozen[i], "frozen[%u]: gpu=%u ref=%u", i, gpuFrozen[i], ref.frozen[i]);
        PXG_CHECK_MSG(gpuUnfrozen[i] == ref.unfrozen[i], "unfrozen[%u]", i);
        PXG_CHECK_MSG(gpuUpdated[i] == ref.updated[i], "updated[%u]", i);
        PXG_CHECK_MSG(gpuCache[i].flags == ref.transformCache[i].flags, "cache.flags[%u]: gpu=%u ref=%u",
                      i, gpuCache[i].flags, ref.transformCache[i].flags);

        const Transform28& gt = gpuCache[i].transform;
        const Transform28& rt = ref.transformCache[i].transform;
        PXG_CHECK_MSG(almostEqual(gt.q.x, rt.q.x, 1e-4f, 1e-5f) && almostEqual(gt.q.y, rt.q.y, 1e-4f, 1e-5f) &&
                          almostEqual(gt.q.z, rt.q.z, 1e-4f, 1e-5f) && almostEqual(gt.q.w, rt.q.w, 1e-4f, 1e-5f),
                      "cache.q[%u]: gpu=(%g,%g,%g,%g) ref=(%g,%g,%g,%g)", i, gt.q.x, gt.q.y, gt.q.z, gt.q.w,
                      rt.q.x, rt.q.y, rt.q.z, rt.q.w);
        PXG_CHECK_MSG(almostEqual(gt.p.x, rt.p.x, 1e-4f, 1e-4f) && almostEqual(gt.p.y, rt.p.y, 1e-4f, 1e-4f) &&
                          almostEqual(gt.p.z, rt.p.z, 1e-4f, 1e-4f),
                      "cache.p[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", i, gt.p.x, gt.p.y, gt.p.z, rt.p.x,
                      rt.p.y, rt.p.z);

        const float* gb = &gpuBounds[i * 6];
        const float rb[6] = { ref.bounds[i].minimum.x, ref.bounds[i].minimum.y, ref.bounds[i].minimum.z,
                              ref.bounds[i].maximum.x, ref.bounds[i].maximum.y, ref.bounds[i].maximum.z };
        for (int c = 0; c < 6; ++c)
            PXG_CHECK_MSG(almostEqual(gb[c], rb[c], 1e-4f, 1e-3f), "bounds[%u][%d]: gpu=%g ref=%g", i, c,
                          gb[c], rb[c]);
    }

    for (uint32_t n = 0; n < maxNodes; ++n)
    {
        PXG_CHECK_MSG(gpuActive[n] == ref.active[n], "active[%u]", n);
        PXG_CHECK_MSG(gpuDeactivate[n] == ref.deactivate[n], "deactivate[%u]", n);
    }

    dev->destroyBuffer(shapeSimBuf);
    dev->destroyBuffer(shapesBuf);
    dev->destroyBuffer(bodySimBuf);
    dev->destroyBuffer(bodyIdxBuf);
    dev->destroyBuffer(sleepBuf);
    dev->destroyBuffer(cacheBuf);
    dev->destroyBuffer(boundsBuf);
    dev->destroyBuffer(frozenBuf);
    dev->destroyBuffer(unfrozenBuf);
    dev->destroyBuffer(updatedBuf);
    dev->destroyBuffer(activeBuf);
    dev->destroyBuffer(deactivateBuf);
    dev->destroyBuffer(blobBuf);
}

} // namespace
