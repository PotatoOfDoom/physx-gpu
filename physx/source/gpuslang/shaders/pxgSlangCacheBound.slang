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

// pxgSlangCacheBound.slang
//
// P8.117a: the transform-cache / bounds machinery of updateCacheAndBound.cuh,
// LIFTED verbatim out of updateBounds.slang into an importable module so
// artiUpdateKinematic (articulationDirectGpuApi.cu) can reuse it instead of
// transcribing it a second time.  updateBounds.slang now imports this module
// and its entries drive the exact same code, so the lift is verified by the
// existing updateTransformCacheAndBoundArrayLaunch / setRigidDynamicGlobalPose
// coverage the moment it lands (the P8.90 "lift, don't rewrite" doctrine).
//
// Contents, each a line-for-line mirror of updateCacheAndBound.cuh:
//   getAbsPose()             -- body2World * body2Actor^-1 * shape2Actor
//   basisExtent()            -- rotated-extent AABB half-widths
//   transformFastBounds()    -- PxBounds3::transformFast()
//   meshScaleToMat33()       -- PxMeshScale::toMat33(): (rot^T column-scaled) * rot
//   updateBoundsBda()        -- the geometry switch (sphere / capsule / box /
//                               convex hull walk / local-bounds fast path)
//   updateCacheAndBoundBda() -- setTransformCache() + optional bounds refresh
//   binarySearchNodeIndexBda() -- binarySearch<PxNodeIndex> (reduction.cuh):
//                               index of the last element <= value, over a
//                               BDA uint2 array (64-bit compare, y-major)

import pxgSlangCommon;
import pxgSlangSimCoreDesc;

// mirrors getAbsPose() in updateCacheAndBound.cuh
public PxgTransform28 getAbsPose(PxgTransform28 body2World, PxgTransform28 shape2Actor,
                                 PxgTransform28 body2Actor)
{
    const PxgTransform28 t0 = transformInvTransform(body2Actor, shape2Actor);
    return transformTransform(body2World, t0);
}

// mirrors basisExtent() in updateCacheAndBound.cuh
public float3 basisExtent(PxgMat33 basis, float3 extent)
{
    const float3 c0 = basis.col0 * extent.x;
    const float3 c1 = basis.col1 * extent.y;
    const float3 c2 = basis.col2 * extent.z;

    return float3(
        abs(c0.x) + abs(c1.x) + abs(c2.x),
        abs(c0.y) + abs(c1.y) + abs(c2.y),
        abs(c0.z) + abs(c1.z) + abs(c2.z));
}

// mirrors PxBounds3::transformFast(const PxTransform&, const PxBounds3&)
public void transformFastBounds(PxgTransform28 pose, float3 localMin, float3 localMax,
                                out float3 outMin, out float3 outMax)
{
    const PxgMat33 rot = mat33FromQuat(pose.q());
    const float3 center = (localMax + localMin) * 0.5f;
    const float3 extents = (localMax - localMin) * 0.5f;
    // P9.89: updateCacheAndBound.cuh:148/:156 -> PxBounds3.h:472-476 centers with
    // transform.transform(getCenter()) = q.rotate(c) + p (PxTransform.h:128-132,
    // PxQuat.h:286-295), NOT the PxMat33(q) product col0*c.x + col1*c.y + col2*c.z:
    // equal in exact arithmetic, a different operation tree in float.  PxMat33(q) is
    // only the basisExtent basis.
    const float3 newCenter = quatRotate(pose.q(), center) + pose.p();
    const float3 newExtents = basisExtent(rot, extents);
    outMin = newCenter - newExtents;
    outMax = newCenter + newExtents;
}

// mirrors PxMeshScale::toMat33(): rotation^-1 * diag(scale) * rotation
public PxgMat33 meshScaleToMat33(float3 scale, float4 rotation)
{
    const PxgMat33 rot = mat33FromQuat(rotation);
    // P9.89: updateCacheAndBound.cuh:113-114 -> PxMeshScale.h:122-127: trans =
    // rot.getTranspose(); trans.column_k *= scale[k]; return trans * rot.  Element
    // (i,j) = sum_k (rot.col_i[k] * s_k) * rot.col_j[k].  Scaling rot's ROWS instead
    // (rot^T * (diag(s) * rot)) groups the same product as rot.col_i[k] * (rot.col_j[k]
    // * s_k) and rounds the off-diagonal elements differently.
    PxgMat33 trans; // transpose(rot), column k scaled by scale[k]
    trans.col0 = float3(rot.col0.x, rot.col1.x, rot.col2.x) * scale.x;
    trans.col1 = float3(rot.col0.y, rot.col1.y, rot.col2.y) * scale.y;
    trans.col2 = float3(rot.col0.z, rot.col1.z, rot.col2.z) * scale.z;
    return mat33MulMat33(trans, rot);
}

// BDA variant of updateBounds() in updateCacheAndBound.cuh: identical branch
// structure, with the PxgShape pool and the world-space bounds array reached
// through device pointers, and hullOrMeshPtr dereferenced as the raw device
// address of the cooked hull blob (exactly what the CUDA does).
public void updateBoundsBda(PxgShapeSim shapeSim, Ptr<PxgShapeGpu> convexShapes,
                            Ptr<PxBounds3Gpu> boundsArray, PxgTransform28 pose, uint index)
{
    const float3 localMin = shapeSim.localBoundsMin();
    const float3 localMax = shapeSim.localBoundsMax();
    const uint shapeType = shapeSim.mShapeFlags_mShapeType >> 16;

    float3 outMin;
    float3 outMax;

    if (shapeType == PXG_GEOM_SPHERE)
    {
        outMin = pose.p() + localMin;
        outMax = pose.p() + localMax;
    }
    else if (shapeType == PXG_GEOM_CAPSULE)
    {
        const float radius = localMax.y;
        const float halfHeight = localMax.x - radius;
        const float3 d = quatBasisVector0(pose.q());
        const float3 extents = abs(d) * halfHeight + radius;
        outMin = pose.p() - extents;
        outMax = pose.p() + extents;
    }
    else if (shapeType == PXG_GEOM_BOX)
    {
        const float3 halfExtents = localMax;
        const float3 extents = basisExtent(mat33FromQuat(pose.q()), halfExtents);
        outMin = pose.p() - extents;
        outMax = pose.p() + extents;
    }
    else if (shapeType == PXG_GEOM_CONVEXMESH)
    {
        const uint hullIndex = shapeSim.mHullDataIndex;

        if (hullIndex != 0xFFFFFFFFu)
        {
            const PxgShapeGpu shape = convexShapes[hullIndex];
            PxgMat33 rot = mat33FromQuat(pose.q());
            const bool scaleIsIdentity =
                shape.scaleX == 1.0f && shape.scaleY == 1.0f && shape.scaleZ == 1.0f &&
                shape.scaleRotX == 0.0f && shape.scaleRotY == 0.0f && shape.scaleRotZ == 0.0f;
            if (!scaleIsIdentity)
                rot = mat33MulMat33(rot, meshScaleToMat33(
                    float3(shape.scaleX, shape.scaleY, shape.scaleZ),
                    float4(shape.scaleRotX, shape.scaleRotY, shape.scaleRotZ, shape.scaleRotW)));

            // convex blob layout (convexFormat.h): the uint4 at byte 16 packs
            // the hull vertex count in bits [8, 16) of its first word
            // (u16High(u32Low(tmp.x))); the float4 vertex array starts at
            // byte 48 (sizeof(uint4) + 2 * sizeof(float4)).
            const uint64_t convexPtr = shape.hullOrMeshPtr;
            const uint4 tmp = (Ptr<uint4>(convexPtr))[1];
            Ptr<float4> pVertices = Ptr<float4>(convexPtr + 48);

            const uint nbHullVertices = (tmp.x >> 8) & 0xFFu;

            float3 minV = float3(PXG_MAX_F32);
            float3 maxV = float3(-PXG_MAX_F32);

            for (uint i = 0; i < nbHullVertices; ++i)
            {
                const float4 vf = pVertices[i];
                const float3 vertexV = mat33Transform(rot, vf.xyz);
                // P9.86: updateCacheAndBound.cuh:134-135 PxVec3::minimum/maximum = componentwise
                // PxMin/PxMax = selectMin/selectMax (a < b ? a : b / a > b ? a : b): keep the NEW
                // vertex on a +-0 tie (and on NaN). GLSL FMin/FMax give an implementation-defined zero sign.
                minV = float3(minV.x < vertexV.x ? minV.x : vertexV.x,
                              minV.y < vertexV.y ? minV.y : vertexV.y,
                              minV.z < vertexV.z ? minV.z : vertexV.z);
                maxV = float3(maxV.x > vertexV.x ? maxV.x : vertexV.x,
                              maxV.y > vertexV.y ? maxV.y : vertexV.y,
                              maxV.z > vertexV.z ? maxV.z : vertexV.z);
            }

            outMin = minV + pose.p();
            outMax = maxV + pose.p();
        }
        else
        {
            // GPU-incompatible hull (more than 64 verts / 31 verts per face)
            transformFastBounds(pose, localMin, localMax, outMin, outMax);
        }
    }
    else
    {
        // Updates any dynamic meshes or HFs that may be attached to simulation shapes
        transformFastBounds(pose, localMin, localMax, outMin, outMax);
    }

    PxBounds3Gpu updatedBound;
    updatedBound.minX = outMin.x;
    updatedBound.minY = outMin.y;
    updatedBound.minZ = outMin.z;
    updatedBound.maxX = outMax.x;
    updatedBound.maxY = outMax.y;
    updatedBound.maxZ = outMax.z;
    boundsArray[index] = updatedBound;
}

// mirrors updateCacheAndBound() in updateCacheAndBound.cuh (BDA form)
public void updateCacheAndBoundBda(PxgTransform28 absPos, PxgShapeSim shapeSim, uint index,
                                   Ptr<PxsCachedTransform> cacheArray, Ptr<PxBounds3Gpu> boundsArray,
                                   Ptr<PxgShapeGpu> shapes, bool isBP)
{
    // setTransformCache(): writes the full 32-byte PxsCachedTransform
    PxsCachedTransform ct;
    ct.transform = absPos;
    ct.flags = 0;
    cacheArray[index] = ct;
    if (isBP)
        updateBoundsBda(shapeSim, shapes, boundsArray, absPos, index);
}

// mirrors binarySearch<PxNodeIndex>() in reduction.cuh over a BDA uint2 array:
// index of the last element <= value (0 when the array is empty or all
// elements are greater; the caller's backward walk handles both).
public uint binarySearchNodeIndexBda(Ptr<uint2> nodeIndices, uint totalCount, uint2 value)
{
    uint left = 0;
    uint right = totalCount;

    while (left < right)
    {
        const uint pos = (left + right) / 2;
        const uint2 element = nodeIndices[pos];
        // 64-bit compare: element <= value
        const bool lessEqual = element.y < value.y || (element.y == value.y && element.x <= value.x);
        if (lessEqual)
            left = pos + 1;
        else
            right = pos;
    }

    return left != 0 ? left - 1 : 0;
}
