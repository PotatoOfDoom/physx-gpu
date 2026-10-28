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
//   - mergeTransformCacheAndBoundArrayChanges (P9.95, at the end of the file:
//     the direct-GPU API's between-steps merge of new / re-inserted shapes)
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
// mArticulationPool by the 496-byte PxgArticulation stride and reads the
// linkBody2Worlds / linkBody2Actors PxTransform arrays through the embedded
// device pointers (stride/offset constants proven by testSimCoreDescLayout).
//
// setRigidDynamicGlobalPose below remains in bound-buffer (StructuredBuffer)
// form; its PxgShape::hullOrMeshPtr is repurposed as a byte offset into the
// bound geometry blob buffer (gGeometryBlob).

import pxgSlangCommon;
import pxgSlangSimCoreDesc;
import pxgSlangCacheBound; // P8.117a: getAbsPose / updateCacheAndBoundBda / basisExtent etc.

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

// ENGINE launch ABI of mergeChangedAABBMgrHandlesLaunch (P9.79): also exactly
// one device pointer ({8} blob), but to a DIFFERENT descriptor —
// PxgUpdateActorDataDesc, the direct-GPU API's smaller twin.  Kept as its own
// push block rather than folded into SimCoreAbiParams: the two blobs happen to
// be the same size, and a shared name would make that coincidence look like a
// shared layout.
struct UpdateActorAbiParams
{
    uint64_t updateActorDesc; // const PxgUpdateActorDataDesc*
};

[[vk::push_constant]]
ConstantBuffer<UpdateBoundsParams> gParams;

[[vk::push_constant]]
ConstantBuffer<SimCoreAbiParams> gSimCoreAbi;

[[vk::push_constant]]
ConstantBuffer<UpdateActorAbiParams> gUpdateActorAbi;

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

// (getAbsPose / basisExtent / transformFastBounds / meshScaleToMat33 moved to
//  pxgSlangCacheBound.slang -- P8.117a lift)

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
            // P9.93: PxMeshScale::isIdentity() tests the scale only (see
            // pxgSlangCacheBound.slang's updateBoundsBda, the engine twin).
            const bool scaleIsIdentity =
                shape.scaleX == 1.0f && shape.scaleY == 1.0f && shape.scaleZ == 1.0f;
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
                // P9.86: updateCacheAndBound.cuh:134-135 PxVec3::minimum/maximum are
                // PxMin/PxMax selects (a<b?a:b / a>b?a:b); GLSL FMin/FMax differ on
                // +0/-0 ties and NaN.
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

    storeBounds(index, outMin, outMax);
}

// ---------------------------------------------------------------------------
// updateTransformCacheAndBoundArrayLaunch — ENGINE launch ABI (BDA)
// ---------------------------------------------------------------------------

// (updateBoundsBda / updateCacheAndBoundBda moved to pxgSlangCacheBound.slang
//  -- P8.117a lift)

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

// mirrors updateChangedAABBMgrHandlesLaunch() in updateTransformAndBoundArray.cu:
// pack the per-shape "updated" flags into the AABB-manager changed-handle
// bitmap (one bit per shape, 32 shapes per word).  This is the kernel that
// tells the broadphase which shapes moved this frame; without it the BP
// early-outs and never re-sweeps (P5 Step 4 root-cause fix).  Engine ABI:
// one device pointer to PxgSimulationCoreDesc (gSimCoreAbi.scDesc), the same
// mirror updateTransformCacheAndBoundArrayLaunch uses.
//
// CUDA uses __ballot_sync(FULL_MASK, updated[i]) so lane j of a 32-lane warp
// contributes bit j; lane 0 writes the packed word.  The llvmpipe subgroup
// width is not a fixed 32, so reproduce the ballot with a groupshared OR per
// warp.  The group is the CUDA's block, UPDATE_AABBMGR_HANDLES = 256 threads =
// 8 warps, one output word per warp per pass, grid-strided over words: until
// 256 threads it was one 32-thread group, i.e. one word per group per pass, so
// a grid of 64 packed 64 words per pass where the CUDA packs 512.  Each word is
// still built from its own 32 flags and written by one thread, so the bitmap is
// the same bits.
static const uint AABB_HANDLES_WARPS = 8u;   // PxgSimulationCoreKernelBlockDim::UPDATE_AABBMGR_HANDLES / 32
groupshared uint gAabbMgrWord[AABB_HANDLES_WARPS];

[shader("compute")]
[numthreads(256, 1, 1)]   // PxgSimulationCoreKernelBlockDim::UPDATE_AABBMGR_HANDLES
void updateChangedAABBMgrHandlesLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                       uint3 groupId : SV_GroupID)
{
    const PxgSimulationCoreDescGpu scDesc = (Ptr<PxgSimulationCoreDescGpu>(gSimCoreAbi.scDesc))[0];

    const uint numWords = scDesc.mBitMapWordCounts;
    const uint numElements = numWords * 32u;

    Ptr<uint> updated = Ptr<uint>(scDesc.mUpdated);
    Ptr<uint> changed = Ptr<uint>(scDesc.mChangedAABBMgrHandles);

    const uint lane = groupThreadId.x & 31u;
    const uint warpInGroup = groupThreadId.x >> 5;

    // group-uniform trip count: a pass is the group's 8 words, base .. base + 7
    for (uint base = groupId.x * AABB_HANDLES_WARPS; base < numWords;
         base += WorkgroupCount().x * AABB_HANDLES_WARPS)
    {
        if (lane == 0)
            gAabbMgrWord[warpInGroup] = 0u;
        GroupMemoryBarrierWithGroupSync();

        const uint w = base + warpInGroup;
        const uint i = w * 32u + lane;
        // The clamp bounds the address the masked-off SIMD lanes form past the last word
        // (P9.22); a live lane's index is unchanged.
        if (w < numWords && updated[min(i, numElements - 1u)] != 0u)
            InterlockedOr(gAabbMgrWord[warpInGroup], 1u << lane);
        GroupMemoryBarrierWithGroupSync();

        if (lane == 0 && w < numWords)
            changed[w] = gAabbMgrWord[warpInGroup];
        GroupMemoryBarrierWithGroupSync();
    }
}

// mirrors mergeChangedAABBMgrHandlesLaunch() in updateTransformAndBoundArray.cu:217
// -- the DIRECT-GPU API's twin of the kernel directly above.
//
// *** WHAT IS DIFFERENT, AND WHY BOTH EXIST. ***
//
// Same ballot, same one-word-per-32-shapes packing, and exactly two differences:
//
//   1. the descriptor.  The sibling reads PxgSimulationCoreDesc; this one reads
//      PxgUpdateActorDataDesc, which is uploaded on a different schedule (see
//      PxgSimulationCore::gpuDmaUpdateData) and whose mUpdated array is the
//      DIRECT-API changed-handle bitmap rather than the pipeline's own.
//   2. `|=` instead of `=`.  That single character is the whole point of the
//      kernel: the CPU-side API and the direct-GPU API each maintain their own
//      idea of which shapes moved this frame, and the broadphase needs the
//      UNION.  An `=` here would silently drop every handle the CPU path marked,
//      and the broadphase would never re-sweep those shapes.
//
// Launched from ScPipeline.cpp:355, once per step, under
// `if (isDirectGPUAPIInitialized())` -- which is why nothing in this harness
// reached it before PXG_SCENE=directapi existed (P9.79).  Engine launch is
// grid 64 / block 256 (PxgSimulationCoreKernelGridDim/BlockDim::UPDATE_AABBMGR_HANDLES).
//
// The groupshared-OR stands in for __ballot_sync for the same reason as in the
// sibling: the llvmpipe subgroup width is not a fixed 32.  The read-modify-write
// is done by each warp's lane 0 alone, exactly as the CUDA does it -- one thread
// per word, so no atomic is needed and none is used.  Like the sibling, the group
// is the CUDA's 256-thread block, one word per warp per pass.
groupshared uint gAabbMgrMergeWord[AABB_HANDLES_WARPS];

[shader("compute")]
[numthreads(256, 1, 1)]   // PxgSimulationCoreKernelBlockDim::UPDATE_AABBMGR_HANDLES
void mergeChangedAABBMgrHandlesLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                      uint3 groupId : SV_GroupID)
{
    const PxgUpdateActorDataDescGpu desc =
        (Ptr<PxgUpdateActorDataDescGpu>(gUpdateActorAbi.updateActorDesc))[0];

    const uint numWords = desc.mBitMapWordCounts;
    const uint numElements = numWords * 32u;

    Ptr<uint> updated = Ptr<uint>(desc.mUpdated);
    Ptr<uint> changed = Ptr<uint>(desc.mChangedAABBMgrHandles);

    const uint lane = groupThreadId.x & 31u;
    const uint warpInGroup = groupThreadId.x >> 5;

    for (uint base = groupId.x * AABB_HANDLES_WARPS; base < numWords;
         base += WorkgroupCount().x * AABB_HANDLES_WARPS)
    {
        if (lane == 0)
            gAabbMgrMergeWord[warpInGroup] = 0u;
        GroupMemoryBarrierWithGroupSync();

        const uint w = base + warpInGroup;
        const uint shape = w * 32u + lane;
        // The clamp bounds the address the masked-off SIMD lanes form past the last word
        // (P9.22); a live lane's index is unchanged.
        if (w < numWords && updated[min(shape, numElements - 1u)] != 0u)
            InterlockedOr(gAabbMgrMergeWord[warpInGroup], 1u << lane);
        GroupMemoryBarrierWithGroupSync();

        if (lane == 0 && w < numWords)
            changed[w] = changed[w] | gAabbMgrMergeWord[warpInGroup];   // CUDA: |=, not =
        GroupMemoryBarrierWithGroupSync();
    }
}

// ENGINE launch ABI of mergeTransformCacheAndBoundArrayChanges (P9.95):
// exactly the CUDA parameter list (updateTransformAndBoundArray.cu:29-37),
// packed by PxgSimulationController::mergeBoundsAndTransformsChanges ->
// {8,8,8,8,8,8,4} = 52 -> 56 B.  5.12 inserted the PxgUpdateActorDataDesc*
// as argument #2 (5.6.1: {8,8,8,8,8,4} = 48 B).
// The three "CPU" arrays are pinned host allocations (the scene's host
// allocator is gpucommon PxgMemoryManager's pinned callback, i.e. memHostAlloc
// on the Vulkan context) that reach the
// blob ALREADY translated to device addresses by getMappedDevicePtr
// (:730-737); the ABI row's hostPtrMask is therefore 0, as for every other
// argument here.
struct MergeTcBoundParams
{
    uint64_t deviceBounds;     // @0  PxBounds3*                     (GPU-BP mBoxFpBoundsBuf)
    uint64_t deviceTransforms; // @8  PxsCachedTransform*            (NP transform cache)
    uint64_t updateActorDesc;  // @16 PxgUpdateActorDataDesc*        (device copy, mUpdated @48)
    uint64_t boundsArray;      // @24 const PxBounds3*               (mapped PxgBoundsArray::mBounds)
    uint64_t transformsArray;  // @32 const PxsCachedTransform*      (mapped PxsTransformCache)
    uint64_t changes;          // @40 const PxgBoundTransformUpdate* (mapped staging, 8 B per record)
    uint numChanges;           // @48 (+4 pad -> 56)
};

[[vk::push_constant]]
ConstantBuffer<MergeTcBoundParams> gMergeTcBound;

// The CUDA indexes with 64-bit pointer arithmetic (ptr[PxU32] = base + u64(i) *
// sizeof), and so does this port.  A 32-bit WORD index (slot * 6u, slot * 8u)
// would wrap mod 2^32 and hide a missing 0x7FFFFFFF mask: (2^31 + f) * 6 ==
// 6f (mod 2^32), so the masked and the unmasked index would address the same
// word.  The byte strides are kept apart from the word counts so a mutant of
// one cannot silently be a mutant of the other.
static const uint64_t MERGE_BOUNDS_BYTES = 24ull;    // sizeof(PxBounds3)
static const uint64_t MERGE_TRANSFORM_BYTES = 32ull; // sizeof(PxsCachedTransform), flags @28
static const uint MERGE_BOUNDS_WORDS = 6u;
static const uint MERGE_TRANSFORM_WORDS = 8u;

// dstBase[dstSlot] = srcBase[srcSlot] -- the PxBounds3 struct assignment, 24 B.
// Moved as uint words, not floats (the P9.93 convention for pure moves): a
// float load/store pair need not preserve sNaN payloads or denormals, a CUDA
// struct copy does.  All words are loaded before any is stored, as a struct
// assignment reads its source before it writes (only observable for
// dst == src, which no producer emits).
void mergeCopyBounds(uint64_t dstBase, uint dstSlot, uint64_t srcBase, uint srcSlot)
{
    Ptr<uint> dst = Ptr<uint>(dstBase + uint64_t(dstSlot) * MERGE_BOUNDS_BYTES);
    Ptr<uint> src = Ptr<uint>(srcBase + uint64_t(srcSlot) * MERGE_BOUNDS_BYTES);
    uint w[MERGE_BOUNDS_WORDS];
    for (uint k = 0u; k < MERGE_BOUNDS_WORDS; ++k)
        w[k] = src[k];
    // load mergeCopyBounds
    for (uint k = 0u; k < MERGE_BOUNDS_WORDS; ++k)
        dst[k] = w[k];
    // end mergeCopyBounds
}

// dstBase[dstSlot] = srcBase[srcSlot] -- the PxsCachedTransform struct
// assignment, 32 B including the flags word @28 (eFROZEN), which the CUDA
// struct copy moves as well.
void mergeCopyTransform(uint64_t dstBase, uint dstSlot, uint64_t srcBase, uint srcSlot)
{
    Ptr<uint> dst = Ptr<uint>(dstBase + uint64_t(dstSlot) * MERGE_TRANSFORM_BYTES);
    Ptr<uint> src = Ptr<uint>(srcBase + uint64_t(srcSlot) * MERGE_TRANSFORM_BYTES);
    uint w[MERGE_TRANSFORM_WORDS];
    for (uint k = 0u; k < MERGE_TRANSFORM_WORDS; ++k)
        w[k] = src[k];
    // load mergeCopyTransform
    for (uint k = 0u; k < MERGE_TRANSFORM_WORDS; ++k)
        dst[k] = w[k];
    // end mergeCopyTransform
}

// mirrors mergeTransformCacheAndBoundArrayChanges() in updateTransformAndBoundArray.cu:52
// -- the DIRECT-GPU API's between-steps merge.  With eENABLE_DIRECT_GPU_API the
// bounds and the transform cache are uploaded whole only on the first step
// (PxgSimulationController.cpp:707-710); every shape added or re-inserted
// after that is recorded by PxgBoundsArray::updateChanges
// (PxgAABBManager.h:308-321) and merged here, one thread per record:
//   indexFrom's MSB set   (isNew: a new shape, or setBounds) -> copy the slot
//                         from the CPU arrays;
//   indexFrom's MSB clear (a re-insertion under a new element id,
//                         ScShapeSimBase.cpp:106-157) -> move the slot within
//                         the DEVICE arrays, whose contents are the live GPU
//                         state the CPU arrays do not have.
//
// One thread per change and NO grid-stride loop: the host sizes the grid to
// ceil(numChanges / 256) (PxgSimulationController.cpp:756-760) and never
// launches with numChanges == 0 (:712).  numthreads == the CUDA block (256,
// PxgSimulationCoreKernelIndices.h:44) because the bridge forwards only the grid.
//
// Speculation (P9.22/P9.23): on llvmpipe the per-lane guard becomes a mask and
// the loads under it are still issued for the excluded lanes, so the record
// index is clamped to the last real record.  With that, every lane holds a
// real record, and the copy addresses stay in range for both branches.  The
// device BOUNDS are grown to the CPU bounds array's size before the launch
// (:741-747); the device TRANSFORM CACHE is grown only to mUsedSize
// (PxsTransformCache::getTotalSize, PxsTransformCache.h:109-111), which is
// <= the CPU array's power-of-two capacity.  Every recorded index went through
// PxsTransformCache::initEntry (mUsedSize > index), so it is in range for both.
// Unverified upstream candidate (read, not run; documented, not reported):
// PxgAABBManager::createAggregate (PxgAABBManager.cpp:292) records a new
// bounds entry via setBounds, but ScScene.cpp:2855-2859 runs no transform-cache
// initEntry for the aggregate, so a PxAggregate added alone between steps could
// make the merge copy transformsArray[aggIdx] past mUsedSize -- the CUDA and
// this port behave the same.
// check_speculated_loads.py sees the clamp through `rec` (measured: the same
// entry spelled `changes[idx * 2u + ...]` without the clamp adds two LOW-launch
// findings), but it does NOT analyse the memory-sourced copy indices, and
// removing the clamp is invisible to it and to every fixture (the guard masks
// all stores).  The clamp is protected by review, not by a gate.
//
// Not reproduced, because it cannot be observed: the CUDA reads
// changes[idx].indexFrom twice; the array is const and restrict and nothing
// writes it during the launch, so one read is the same value.
//
// Upstream candidates, NOT verified at runtime and NOT changed here (the port
// reproduces the CUDA):
//   - Two records can target one slot in one launch, and then the result
//     depends on scheduling: resetFiltering twice on one shape in one window
//     records {j, i} and then {j, j|new} (the second call finds the shape
//     pending insert and keeps the id, ScShapeSimBase.cpp:108-147).  The isNew
//     writer would also copy a stale host pose for a dynamic in direct mode.
//   - isNew copies the CPU slot, which for a DYNAMIC body in direct mode holds
//     the pose the host last knew (writeback is off): attachShape on a moved
//     dynamic after init merges its new shape at the spawn pose for one step
//     (ScShapeSimBase.cpp:184-191).
//   - The merge runs only when a shape instance changed
//     (PxgSimulationController.cpp:799); a change recorded without one would be
//     dropped at resetChanges (PxgSimulationCore.cpp:2256-2261).  No public
//     trigger is known.
//   - Port-only: dispatches are deferred to the next flush, so a host write to
//     the mapped CPU arrays between this launch and the flush would be seen by
//     the kernel.  Every window in PXG_SCENE=directapimerge opens after
//     fetchResults(true); re-examine if a scene ever edits between simulate and
//     fetchResults.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void mergeTransformCacheAndBoundArrayChanges(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    // CUDA: int idx = blockIdx.x * blockDim.x + threadIdx.x, compared against the
    // PxU32 numChanges -- an UNSIGNED compare under the usual conversions; the grid
    // is far below 2^31 threads, so a uint is exact.  Spelled as a plain uint so the
    // speculated-load checker can see the guard.
    const uint idx = dispatchThreadId.x;
    const uint numChanges = gMergeTcBound.numChanges;
    if (idx < numChanges)
    {
        const uint rec = min(idx, numChanges - 1u);   // speculation clamp (P9.22); the guard decides
        Ptr<uint> changes = Ptr<uint>(gMergeTcBound.changes);
        const uint indexTo = changes[rec * 2u + 0u];
        const uint rawFrom = changes[rec * 2u + 1u];
        const uint indexFrom = rawFrom & 0x7FFFFFFFu;
        const bool isNew = (rawFrom & (1u << 31u)) != 0u;
        if (isNew)
        {
            mergeCopyBounds(gMergeTcBound.deviceBounds, indexTo, gMergeTcBound.boundsArray, indexFrom);
            mergeCopyTransform(gMergeTcBound.deviceTransforms, indexTo, gMergeTcBound.transformsArray, indexFrom);
        }
        else
        {
            mergeCopyBounds(gMergeTcBound.deviceBounds, indexTo, gMergeTcBound.deviceBounds, indexFrom);
            mergeCopyTransform(gMergeTcBound.deviceTransforms, indexTo, gMergeTcBound.deviceTransforms, indexFrom);
            // 5.12: a re-inserted bound (e.g. a kinematic/dynamic switch) carries its
            // direct-GPU-API mUpdated flag from indexFrom to indexTo, and the source
            // flag is reset.
            Ptr<uint> updated = Ptr<uint>((Ptr<PxgUpdateActorDataDescGpu>(gMergeTcBound.updateActorDesc))[0].mUpdated);
            updated[indexTo] = updated[indexFrom];
            updated[indexFrom] = 0u;
        }
    }
}
