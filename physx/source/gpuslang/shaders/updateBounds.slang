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
//   - updateTransformCacheAndBoundArrayLaunch (rigid + articulation paths)
//
// Per simulation shape this kernel composes the world pose
// (body2World * body2Actor^-1 * shape2Actor), refreshes the transform cache
// and recomputes the world-space AABB for sphere / capsule / box / convex
// mesh geometry (other types use the transformed local-bounds fast path,
// like the CUDA original).
//
// updateTransformCacheAndBoundArrayLaunch runs on the ENGINE launch ABI
// (P5 Step 4): the push blob is the CUDA parameter list — one device pointer
// to the engine's PxgSimulationCoreDesc — and every array (shape sims, body
// sims, shapes, sleep data, transform cache, bounds, frozen/unfrozen/updated/
// activate/deactivate marks, articulation pool) is walked through the
// byte-exact pxgSlangSimCoreDesc mirrors via BDA.  PxgShape::hullOrMeshPtr is
// dereferenced as the raw device pointer it is under CUDA (vertices start at
// byte 48, vertex count packed in the uint4 at byte 16 — see convexFormat.h).
// The articulation branch is transcribed faithfully: it indexes
// mArticulationPool by the 544-byte PxgArticulation stride and reads the
// linkBody2Worlds / linkBody2Actors PxTransform arrays through the embedded
// device pointers (stride/offset constants proven by testSimCoreDescLayout).
//
// setRigidDynamicGlobalPose below remains in bound-buffer (StructuredBuffer)
// form; its PxgShape::hullOrMeshPtr is repurposed as a byte offset into the
// bound geometry blob buffer (gGeometryBlob).

import pxgSlangCommon;
import pxgSlangSimCoreDesc;

static const uint BLOCK_SIZE = 256;

struct UpdateBoundsParams
{
    uint numShapes;      // (legacy bound-buffer slot; unused by the BDA entry)
    uint nbElements;     // setRigidDynamicGlobalPose: number of pose updates
    uint totalNumShapes; // setRigidDynamicGlobalPose: size of the node/shape index maps
};

// ENGINE launch ABI of updateTransformCacheAndBoundArrayLaunch: exactly the
// CUDA parameter list — one device pointer ({8} blob).
struct SimCoreAbiParams
{
    uint64_t scDesc; // const PxgSimulationCoreDesc*
};

[[vk::push_constant]]
ConstantBuffer<UpdateBoundsParams> gParams;

[[vk::push_constant]]
ConstantBuffer<SimCoreAbiParams> gSimCoreAbi;

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

// additional bindings for setRigidDynamicGlobalPose (direct GPU API)
RWStructuredBuffer<uint2> gRigidNodeIndices;            // binding 13: sorted PxNodeIndex per shape
RWStructuredBuffer<uint> gShapeIndices;                 // binding 14: shape index per sorted entry
RWStructuredBuffer<float> gPoseData;                    // binding 15: user PxTransform array (7 floats each)
RWStructuredBuffer<uint> gGpuIndices;                   // binding 16: PxRigidDynamicGPUIndex array

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

// ---------------------------------------------------------------------------
// updateTransformCacheAndBoundArrayLaunch — ENGINE launch ABI (BDA)
// ---------------------------------------------------------------------------

// BDA variant of updateBounds() in updateCacheAndBound.cuh: identical branch
// structure, with the PxgShape pool and the world-space bounds array reached
// through device pointers, and hullOrMeshPtr dereferenced as the raw device
// address of the cooked hull blob (exactly what the CUDA does).
void updateBoundsBda(PxgShapeSim shapeSim, Ptr<PxgShapeGpu> convexShapes, Ptr<PxBounds3Gpu> boundsArray,
                     PxgTransform28 pose, uint index)
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
void updateCacheAndBoundBda(PxgTransform28 absPos, PxgShapeSim shapeSim, uint index,
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

// mirrors updateTransformCacheAndBoundArrayLaunch() in
// updateTransformAndBoundArray.cu — ENGINE launch ABI: one push argument
// (the PxgSimulationCoreDesc device pointer), grid-stride loop over shapes,
// rigid + articulation branches, launched with the engine's grid geometry
// (PxgSimulationCoreKernelGridDim/BlockDim::UPDATE_TRANSFORMCACHE_AND_BOUNDARRAY
// = 256 x 256).
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateTransformCacheAndBoundArrayLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const PxgSimulationCoreDescGpu scDesc = (Ptr<PxgSimulationCoreDescGpu>(gSimCoreAbi.scDesc))[0];

    Ptr<PxgSolverBodySleepData> gSleepDataP = Ptr<PxgSolverBodySleepData>(scDesc.mSleepData);

    Ptr<uint> gBodyDataIndicesP = Ptr<uint>(scDesc.mBodyDataIndices);
    Ptr<PxgShapeGpu> gShapesP = Ptr<PxgShapeGpu>(scDesc.mShapes);

    Ptr<PxgBodySim> gBodySimPool = Ptr<PxgBodySim>(scDesc.mBodySimBufferDeviceData);

    const uint gNumShapes = scDesc.mNbTotalShapes;
    Ptr<PxgShapeSim> gShapeSimPool = Ptr<PxgShapeSim>(scDesc.mShapeSimsBufferDeviceData);

    const uint64_t gArticulations = scDesc.mArticulationPool;
    Ptr<PxgSolverBodySleepData> gArticulationSleepData =
        Ptr<PxgSolverBodySleepData>(scDesc.mArticulationSleepDataPool);

    Ptr<PxsCachedTransform> gTransformCacheP = Ptr<PxsCachedTransform>(scDesc.mTransformCache);
    Ptr<PxBounds3Gpu> gBoundsP = Ptr<PxBounds3Gpu>(scDesc.mBounds);

    // Each shape has a corresponding unfrozen element
    Ptr<uint> frozen = Ptr<uint>(scDesc.mFrozen);
    Ptr<uint> unfrozen = Ptr<uint>(scDesc.mUnfrozen);
    // Each shape has a updated element corresponding to the elementIndex
    Ptr<uint> updated = Ptr<uint>(scDesc.mUpdated);

    // Each body has a corresponding active and deactive element
    Ptr<uint> active = Ptr<uint>(scDesc.mActivate);
    Ptr<uint> deactivate = Ptr<uint>(scDesc.mDeactivate);

    const uint idx = dispatchThreadId.x;

    for (uint i = idx; i < gNumShapes; i += BLOCK_SIZE * WorkgroupCount().x)
    {
        const PxgShapeSim shapeSim = gShapeSimPool[i];

        const uint2 bodySimNodeIndex = shapeSim.mBodySimIndex; // bodySimIndex is the same as nodeIndex in the IG

        // not static body or deleted shape
        if (!nodeIndexIsStaticBody(bodySimNodeIndex))
        {
            const uint elementIndex = i; // this is the transform cache and bound array index
            const uint bodySimIndex = nodeIndexIndex(bodySimNodeIndex);

            const PxgBodySim bodySim = gBodySimPool[bodySimIndex];

            const uint shapeFlags = shapeSim.mShapeFlags_mShapeType & 0xFFFFu;
            const bool isBP = (shapeFlags & (PXG_SHAPE_SIMULATION | PXG_SHAPE_TRIGGER)) != 0;

            const bool isBPOrSq =
                (shapeFlags & (PXG_SHAPE_SIMULATION | PXG_SHAPE_TRIGGER | PXG_SHAPE_SCENE_QUERY)) != 0;

            if (!nodeIndexIsArticulation(bodySimNodeIndex))
            {
                const uint activeNodeIndex = gBodyDataIndicesP[bodySimIndex];

                // if activeNodeIndex is valid, which means this node is active
                if (activeNodeIndex != 0xFFFFFFFFu)
                {
                    const uint internalFlags = gSleepDataP[activeNodeIndex].internalFlags;
                    const PxgTransform28 body2World =
                        makeTransform28(bodySim.body2World.q, bodySim.body2World.p.xyz);

                    if ((internalFlags & PXG_BODY_FREEZE_THIS_FRAME) != 0 && (internalFlags & PXG_BODY_FROZEN) != 0)
                    {
                        frozen[i] = 1;
                        gTransformCacheP[elementIndex].flags = PXG_TRANSFORM_CACHE_FROZEN;
                    }
                    else if ((internalFlags & PXG_BODY_UNFREEZE_THIS_FRAME) != 0)
                    {
                        unfrozen[i] = 1;
                    }

                    if ((internalFlags & PXG_BODY_FROZEN) == 0 || (internalFlags & PXG_BODY_FREEZE_THIS_FRAME) != 0)
                    {
                        if (isBP)
                            updated[elementIndex] = 1;

                        const PxgTransform28 body2Actor = makeTransform28(
                            bodySim.body2Actor_maxImpulseW.q, bodySim.body2Actor_maxImpulseW.p.xyz);
                        const PxgTransform28 absPos = getAbsPose(body2World, shapeSim.mTransform, body2Actor);

                        updateCacheAndBoundBda(absPos, shapeSim, elementIndex, gTransformCacheP, gBoundsP,
                                               gShapesP, isBPOrSq);
                    }

                    if ((internalFlags & PXG_BODY_ACTIVATE_THIS_FRAME) != 0)
                        active[bodySimIndex] = 1;
                    else if ((internalFlags & PXG_BODY_DEACTIVATE_THIS_FRAME) != 0)
                        deactivate[bodySimIndex] = 1;
                }
            }
            else
            {
                // This is articulation
                const uint articulationId = bodySim.articulationRemapId;
                const PxgSolverBodySleepData artiSleepData = gArticulationSleepData[articulationId];
                const uint internalFlags = artiSleepData.internalFlags;

                const uint linkId = nodeIndexArticulationLinkId(bodySimNodeIndex);

                // articulation.linkBody2Worlds[linkId] (PxTransform, 28-byte stride)
                Ptr<PxgTransform28> linkBody2Worlds = Ptr<PxgTransform28>(
                    artiLinkTransformArray(gArticulations, articulationId, PXG_ARTI_LINK_BODY2WORLDS_OFFSET));
                const PxgTransform28 body2World = linkBody2Worlds[linkId];

                if (isBP)
                    updated[elementIndex] = 1;

                // articulation.linkBody2Actors[linkId]
                Ptr<PxgTransform28> linkBody2Actors = Ptr<PxgTransform28>(
                    artiLinkTransformArray(gArticulations, articulationId, PXG_ARTI_LINK_BODY2ACTORS_OFFSET));
                const PxgTransform28 body2Actor = linkBody2Actors[linkId];

                const PxgTransform28 absPos = getAbsPose(body2World, shapeSim.mTransform, body2Actor);

                updateCacheAndBoundBda(absPos, shapeSim, elementIndex, gTransformCacheP, gBoundsP,
                                       gShapesP, isBPOrSq);

                if ((internalFlags & PXG_BODY_ACTIVATE_THIS_FRAME) != 0)
                    active[bodySimIndex] = 1;
                else if ((internalFlags & PXG_BODY_DEACTIVATE_THIS_FRAME) != 0)
                    deactivate[bodySimIndex] = 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// setRigidDynamicGlobalPose (direct GPU API, updateBodiesAndShapes.cu)
// ---------------------------------------------------------------------------

// mirrors binarySearch<PxNodeIndex>() in reduction.cuh: index of the last
// element <= value (0 when the array is empty or all elements are greater;
// the caller's backward walk handles both).
uint binarySearchNodeIndex(uint totalCount, uint2 value)
{
    uint left = 0;
    uint right = totalCount;

    while (left < right)
    {
        const uint pos = (left + right) / 2;
        const uint2 element = gRigidNodeIndices[pos];
        // 64-bit compare: element <= value
        const bool lessEqual = element.y < value.y || (element.y == value.y && element.x <= value.x);
        if (lessEqual)
            left = pos + 1;
        else
            right = pos;
    }

    return left != 0 ? left - 1 : 0;
}

// mirrors setRigidDynamicGlobalPose() in updateBodiesAndShapes.cu: writes the
// new actor pose into the body sim (as body2World = pose * body2Actor) and
// refreshes transform cache + bounds for every shape of the rigid body.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void setRigidDynamicGlobalPose(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint index = gGpuIndices[i];

    // input PxTransform (7 tightly packed floats: q, p)
    const float4 inQ = float4(gPoseData[i * 7 + 0], gPoseData[i * 7 + 1], gPoseData[i * 7 + 2],
                              gPoseData[i * 7 + 3]);
    const float3 inP = float3(gPoseData[i * 7 + 4], gPoseData[i * 7 + 5], gPoseData[i * 7 + 6]);

    PxgBodySim bodySim = gBodySims[index];

    // body2World = trans.transform(body2Actor) (PxAlignedTransform math)
    const float4 b2aQ = bodySim.body2Actor_maxImpulseW.q;
    const float4 b2aP = bodySim.body2Actor_maxImpulseW.p;
    const float4 q = quatMul(inQ, b2aQ);
    const float3 p = quatRotate(inQ, b2aP.xyz) + inP;
    const PxgTransform28 body2World = makeTransform28(q, p);

    gBodySims[index].body2World.p = float4(p, 0.0f);
    gBodySims[index].body2World.q = q;

    const PxgTransform28 body2Actor = makeTransform28(b2aQ, b2aP.xyz);

    if (gParams.totalNumShapes == 0)
    {
        // guard against no actors having shapes in the scene
        return;
    }

    // we know this is not an articulation link, so all fine
    const uint2 nodeIndex = uint2(index, 0);

    // search for the last position matching the node index, then walk
    // backward over the sorted per-shape node index array
    uint pos = binarySearchNodeIndex(gParams.totalNumShapes, nodeIndex);

    while (pos != 0xFFFFFFFFu && all(gRigidNodeIndices[pos] == nodeIndex))
    {
        const uint shapeIndex = gShapeIndices[pos];
        if (shapeIndex != 0xFFFFFFFFu)
        {
            const PxgShapeSim shapeSim = gShapeSims[shapeIndex];

            const PxgTransform28 absPos = getAbsPose(body2World, shapeSim.mTransform, body2Actor);

            // updateCacheAndBound(..., isBP = true)
            gTransformCache[shapeIndex].transform = absPos;
            gTransformCache[shapeIndex].flags = 0;
            updateBounds(shapeSim, absPos, shapeIndex);

            gUpdated[shapeIndex] = 1;
        }
        pos--;
    }
}
