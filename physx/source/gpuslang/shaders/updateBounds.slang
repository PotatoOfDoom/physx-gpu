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

// updateBounds.slang
//
// Slang port of the transform-cache / bounds update kernel
// (gpusimulationcontroller/src/CUDA/updateTransformAndBoundArray.cu +
//  gpucommon/src/CUDA/updateCacheAndBound.cuh):
//   - updateTransformCacheAndBoundArrayLaunch (rigid-body path)
//
// Per simulation shape this kernel composes the world pose
// (body2World * body2Actor^-1 * shape2Actor), refreshes the transform cache
// and recomputes the world-space AABB for sphere / capsule / box / convex
// mesh geometry (other types use the transformed local-bounds fast path,
// like the CUDA original).
//
// Differences from CUDA:
//   - PxgShape::hullOrMeshPtr embedded a raw device pointer to the cooked
//     convex data; the port stores a byte offset into the bound geometry
//     blob buffer (gGeometryBlob) instead.  The blob layout itself is
//     unchanged (vertices start at byte 48, vertex count packed in the
//     uint4 at byte 16 — see convexFormat.h).
//   - The articulation branch is not yet ported: it walks per-articulation
//     link arrays through embedded device pointers (PxgArticulation).
//     Articulated shapes are skipped; see the migration notes.

import pxgSlangCommon;

static const uint BLOCK_SIZE = 256;

struct UpdateBoundsParams
{
    uint numShapes;
};

[[vk::push_constant]]
ConstantBuffer<UpdateBoundsParams> gParams;

RWStructuredBuffer<PxgShapeSim> gShapeSims;             // binding 0
RWStructuredBuffer<PxgShape> gShapes;                   // binding 1
RWStructuredBuffer<PxgBodySim> gBodySims;               // binding 2
RWStructuredBuffer<uint> gBodyDataIndices;              // binding 3
RWStructuredBuffer<PxgSolverBodySleepData> gSleepData;  // binding 4
RWStructuredBuffer<PxsCachedTransform> gTransformCache; // binding 5
RWStructuredBuffer<float> gBounds;                      // binding 6 (PxBounds3: 6 tightly packed floats per entry)
RWStructuredBuffer<uint> gFrozen;                       // binding 7
RWStructuredBuffer<uint> gUnfrozen;                     // binding 8
RWStructuredBuffer<uint> gUpdated;                      // binding 9
RWStructuredBuffer<uint> gActive;                       // binding 10
RWStructuredBuffer<uint> gDeactivate;                   // binding 11
RWStructuredBuffer<uint> gGeometryBlob;                 // binding 12 (cooked convex hulls, uint-addressed)

// PxBounds3 is 24 bytes (min, max as PxVec3).  To keep the C++-compatible
// tight packing independent of vector-alignment rules the bounds array is
// bound as a raw float buffer: entry i occupies floats [6*i, 6*i+6).
void storeBounds(uint index, float3 minimum, float3 maximum)
{
    const uint base = index * 6;
    gBounds[base + 0] = minimum.x;
    gBounds[base + 1] = minimum.y;
    gBounds[base + 2] = minimum.z;
    gBounds[base + 3] = maximum.x;
    gBounds[base + 4] = maximum.y;
    gBounds[base + 5] = maximum.z;
}

// mirrors getAbsPose() in updateCacheAndBound.cuh
PxgTransform28 getAbsPose(PxgTransform28 body2World, PxgTransform28 shape2Actor, PxgTransform28 body2Actor)
{
    const PxgTransform28 t0 = transformInvTransform(body2Actor, shape2Actor);
    return transformTransform(body2World, t0);
}

// mirrors basisExtent() in updateCacheAndBound.cuh
float3 basisExtent(PxgMat33 basis, float3 extent)
{
    const float3 c0 = basis.col0 * extent.x;
    const float3 c1 = basis.col1 * extent.y;
    const float3 c2 = basis.col2 * extent.z;

    return float3(
        abs(c0.x) + abs(c1.x) + abs(c2.x),
        abs(c0.y) + abs(c1.y) + abs(c2.y),
        abs(c0.z) + abs(c1.z) + abs(c2.z));
}

// mirrors PxBounds3::transformFast()
void transformFastBounds(PxgTransform28 pose, float3 localMin, float3 localMax,
                         out float3 outMin, out float3 outMax)
{
    const PxgMat33 rot = mat33FromQuat(pose.q());
    const float3 center = (localMax + localMin) * 0.5f;
    const float3 extents = (localMax - localMin) * 0.5f;
    const float3 newCenter = mat33Transform(rot, center) + pose.p();
    const float3 newExtents = basisExtent(rot, extents);
    outMin = newCenter - newExtents;
    outMax = newCenter + newExtents;
}

// mirrors PxMeshScale::toMat33(): rotation^-1 * diag(scale) * rotation
PxgMat33 meshScaleToMat33(float3 scale, float4 rotation)
{
    const PxgMat33 rot = mat33FromQuat(rotation);
    PxgMat33 trans; // transpose(rot)
    trans.col0 = float3(rot.col0.x, rot.col1.x, rot.col2.x);
    trans.col1 = float3(rot.col0.y, rot.col1.y, rot.col2.y);
    trans.col2 = float3(rot.col0.z, rot.col1.z, rot.col2.z);

    PxgMat33 diagTimesRot;
    diagTimesRot.col0 = rot.col0 * scale;
    diagTimesRot.col1 = rot.col1 * scale;
    diagTimesRot.col2 = rot.col2 * scale;
    // transpose(rot) * (diag(scale) * rot):
    return mat33MulMat33(trans, diagTimesRot);
}

// mirrors updateBounds() in updateCacheAndBound.cuh
void updateBounds(PxgShapeSim shapeSim, PxgTransform28 pose, uint index)
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
            const PxgShape shape = gShapes[hullIndex];
            PxgMat33 rot = mat33FromQuat(pose.q());
            const bool scaleIsIdentity =
                shape.scaleX == 1.0f && shape.scaleY == 1.0f && shape.scaleZ == 1.0f &&
                shape.scaleRotX == 0.0f && shape.scaleRotY == 0.0f && shape.scaleRotZ == 0.0f;
            if (!scaleIsIdentity)
                rot = mat33MulMat33(rot, meshScaleToMat33(shape.scale(), shape.scaleRotation()));

            // convex blob layout (convexFormat.h): the uint4 at byte 16 packs
            // the hull vertex count in bits [8, 16) of its first word; the
            // float4 vertex array starts at byte 48.
            const uint blobBase = shape.hullOrMeshOffset.x / 4; // uint index
            const uint tmpX = gGeometryBlob[blobBase + 4];
            const uint nbHullVertices = (tmpX >> 8) & 0xFFu;

            float3 minV = float3(PXG_MAX_F32);
            float3 maxV = float3(-PXG_MAX_F32);

            const uint verticesBase = blobBase + 12; // byte 48
            for (uint i = 0; i < nbHullVertices; ++i)
            {
                const float3 v = float3(
                    asfloat(gGeometryBlob[verticesBase + i * 4 + 0]),
                    asfloat(gGeometryBlob[verticesBase + i * 4 + 1]),
                    asfloat(gGeometryBlob[verticesBase + i * 4 + 2]));
                const float3 vertexV = mat33Transform(rot, v);
                minV = min(minV, vertexV);
                maxV = max(maxV, vertexV);
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

    storeBounds(index, outMin, outMax);
}

// mirrors updateTransformCacheAndBoundArrayLaunch() in
// updateTransformAndBoundArray.cu (rigid-body branch)
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateTransformCacheAndBoundArrayLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    // The CUDA kernel uses a grid-stride loop; here the host dispatches
    // ceil(numShapes / BLOCK_SIZE) groups so a single shot covers the range.
    const uint i = dispatchThreadId.x;
    const uint numShapes = gParams.numShapes;

    {
        if (i >= numShapes)
            return;

        const PxgShapeSim shapeSim = gShapeSims[i];

        const uint2 bodySimNodeIndex = shapeSim.mBodySimIndex;

        // not static body or deleted shape
        if (nodeIndexIsStaticBody(bodySimNodeIndex))
            return;

        const uint elementIndex = i; // transform cache and bound array index
        const uint bodySimIndex = nodeIndexIndex(bodySimNodeIndex);

        const PxgBodySim bodySim = gBodySims[bodySimIndex];

        const uint shapeFlags = shapeSim.mShapeFlags_mShapeType & 0xFFFFu;
        const bool isBP = (shapeFlags & (PXG_SHAPE_SIMULATION | PXG_SHAPE_TRIGGER)) != 0;
        const bool isBPOrSq = (shapeFlags & (PXG_SHAPE_SIMULATION | PXG_SHAPE_TRIGGER | PXG_SHAPE_SCENE_QUERY)) != 0;

        if (!nodeIndexIsArticulation(bodySimNodeIndex))
        {
            const uint activeNodeIndex = gBodyDataIndices[bodySimIndex];

            // if activeNodeIndex is valid, which means this node is active
            if (activeNodeIndex != 0xFFFFFFFFu)
            {
                const uint internalFlags = gSleepData[activeNodeIndex].internalFlags;
                const PxgTransform28 body2World = makeTransform28(bodySim.body2World.q, bodySim.body2World.p.xyz);

                if ((internalFlags & PXG_BODY_FREEZE_THIS_FRAME) != 0 && (internalFlags & PXG_BODY_FROZEN) != 0)
                {
                    gFrozen[i] = 1;
                    gTransformCache[elementIndex].flags = PXG_TRANSFORM_CACHE_FROZEN;
                }
                else if ((internalFlags & PXG_BODY_UNFREEZE_THIS_FRAME) != 0)
                {
                    gUnfrozen[i] = 1;
                }

                if ((internalFlags & PXG_BODY_FROZEN) == 0 || (internalFlags & PXG_BODY_FREEZE_THIS_FRAME) != 0)
                {
                    if (isBP)
                        gUpdated[elementIndex] = 1;

                    const PxgTransform28 body2Actor = makeTransform28(
                        bodySim.body2Actor_maxImpulseW.q, bodySim.body2Actor_maxImpulseW.p.xyz);
                    const PxgTransform28 absPos = getAbsPose(body2World, shapeSim.mTransform, body2Actor);

                    // updateCacheAndBound()
                    gTransformCache[elementIndex].transform = absPos;
                    gTransformCache[elementIndex].flags = 0;
                    if (isBPOrSq)
                        updateBounds(shapeSim, absPos, elementIndex);
                }

                if ((internalFlags & PXG_BODY_ACTIVATE_THIS_FRAME) != 0)
                    gActive[bodySimIndex] = 1;
                else if ((internalFlags & PXG_BODY_DEACTIVATE_THIS_FRAME) != 0)
                    gDeactivate[bodySimIndex] = 1;
            }
        }
        // else: articulation branch not yet ported (see file header)
    }
}
