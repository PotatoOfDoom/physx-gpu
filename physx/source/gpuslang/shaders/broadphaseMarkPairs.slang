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

// broadphaseMarkPairs.slang
//
// Slang port of the broad-phase incremental delete/update marking kernels
// (gpubroadphase/src/CUDA/broadphase.cu):
//   - markRemovedPairsLaunch: for each removed handle, mark its start/end sort
//     handles as deleted (bit 2) in the previous frame's handle list and empty
//     its old integer bounds.
//   - markRemovedPairsProjectionsLaunch: invalidate the removed handles'
//     projections and SAP-box endpoints so the next radix sort shuffles them to
//     the end of the list.
//   - markUpdatedPairsLaunch: for each changed (but not added/removed) handle in
//     the AABB-manager bitmap, refresh its projections from the new bounds.
//   - markUpdatedPairsLaunch2: the same refresh driven by an explicit updated-
//     handle array (SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU).
//
// All four are pure per-thread grid-stride scatter kernels — no groupshared,
// scans, atomics or barriers.
//
// P5 Step 4 (BDA, engine launch path): all four take ONE PxgBroadPhaseDesc
// device pointer (bpDescd) and walk the byte-exact PxgBroadPhaseDescGpu mirror
// (pxgSlangBpDesc.slang, proven by testBpDescLayout), exactly as the engine
// launches them (PxgCudaBroadPhaseSap markRemovedPairs*/markUpdatedPairsKernel,
// single bpDescd param).  In the .cu each per-axis array is a distinct device
// pointer (boxSapBox1D[axis], boxHandles[1][axis] = the previous frame's, and
// boxProjections[axis]); the bound-buffer form packed all three axes into one
// flat buffer with an axis*nbProj / axis*numHandles stride — that stride
// collapses to per-pointer indexing here (sapBox[handle*2+i] directly, no
// numHandles multiplier).  Grid is grid-strided; gridDim comes from
// WorkgroupCount().x, not a push field.

import pxgSlangBpDesc;

static const uint AXIS_COUNT = 3;
static const uint PXG_INVALID_BP_PROJECTION = 0xffffffffu;
static const uint PXG_INVALID_BP_SAP_BOX = 0xffffffffu;

// engine ABI: the PxgBroadPhaseDesc device pointer only
struct BpDescParams
{
    uint64_t bpDesc;
}

[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

uint markDeleted(uint handle) { return handle | (1u << 2); }

// ---------------------------------------------------------------------------
// markRemovedPairsLaunch (BP_MARK_DELETEDPAIRS)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markRemovedPairsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint num = desc[0].numRemovedHandles;
    const uint stride = WorkgroupCount().x * 256u;

    Ptr<uint> removedHandles = Ptr<uint>(desc[0].updateData_removedHandles);
    Ptr<uint> oldBounds = Ptr<uint>(desc[0].oldIntegerBounds);

    for (uint i = dtid.x; i < num; i += stride)
    {
        const uint handle = removedHandles[i];
        [unroll]
        for (uint axis = 0; axis < AXIS_COUNT; ++axis)
        {
            Ptr<uint> sapBox = Ptr<uint>(desc[0].boxSapBox1D[axis]);
            Ptr<uint> boxHandle1 = Ptr<uint>(desc[0].boxHandles[3u + axis]); // boxHandles[1][axis] (previous)
            const uint minIndex = sapBox[handle * 2u + 0u];
            const uint maxIndex = sapBox[handle * 2u + 1u];
            boxHandle1[minIndex] = markDeleted(boxHandle1[minIndex]);
            boxHandle1[maxIndex] = markDeleted(boxHandle1[maxIndex]);
        }
        // setEmpty on the old integer bounds
        const uint b = handle * 6u;
        oldBounds[b + 0u] = 0xff7fffffu;
        oldBounds[b + 1u] = 0xff7fffffu;
        oldBounds[b + 2u] = 0xff7fffffu;
        oldBounds[b + 3u] = 0x00800000u;
        oldBounds[b + 4u] = 0x00800000u;
        oldBounds[b + 5u] = 0x00800000u;
    }
}

// ---------------------------------------------------------------------------
// markRemovedPairsProjectionsLaunch (BP_UPDATE_DELETEDPAIRS)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markRemovedPairsProjectionsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint num = desc[0].numRemovedHandles;
    const uint stride = WorkgroupCount().x * 256u;

    Ptr<uint> removedHandles = Ptr<uint>(desc[0].updateData_removedHandles);

    for (uint i = dtid.x; i < num; i += stride)
    {
        const uint handle = removedHandles[i];
        [unroll]
        for (uint axis = 0; axis < AXIS_COUNT; ++axis)
        {
            Ptr<uint> sapBox = Ptr<uint>(desc[0].boxSapBox1D[axis]);
            Ptr<uint> boxProjection = Ptr<uint>(desc[0].boxProjections[axis]);
            const uint minIndex = sapBox[handle * 2u + 0u];
            const uint maxIndex = sapBox[handle * 2u + 1u];
            boxProjection[minIndex] = PXG_INVALID_BP_PROJECTION;
            boxProjection[maxIndex] = PXG_INVALID_BP_PROJECTION;
            sapBox[handle * 2u + 0u] = PXG_INVALID_BP_SAP_BOX;
            sapBox[handle * 2u + 1u] = PXG_INVALID_BP_SAP_BOX;
        }
    }
}

// write the box's new min/max projections at its per-axis SAP-box positions
void writeUpdatedProjections(Ptr<PxgBroadPhaseDescGpu> desc, uint handle)
{
    Ptr<uint> newBounds = Ptr<uint>(desc[0].newIntegerBounds);
    const uint nb = handle * 6u;
    [unroll]
    for (uint axis = 0; axis < AXIS_COUNT; ++axis)
    {
        Ptr<uint> sapBox = Ptr<uint>(desc[0].boxSapBox1D[axis]);
        Ptr<uint> boxProjection = Ptr<uint>(desc[0].boxProjections[axis]);
        const uint minV = newBounds[nb + axis];
        const uint maxV = newBounds[nb + axis + 3u];
        const uint minIndex = sapBox[handle * 2u + 0u];
        const uint maxIndex = sapBox[handle * 2u + 1u];
        boxProjection[minIndex] = minV;
        boxProjection[maxIndex] = maxV;
    }
}

// ---------------------------------------------------------------------------
// markUpdatedPairsLaunch (BP_UPDATE_UPDATEDPAIRS) — bitmap driven
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markUpdatedPairsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint numElements = desc[0].aabbMngr_changedHandleBitMapWordCounts * 32u;
    const uint stride = WorkgroupCount().x * 256u;

    Ptr<uint> changedMap = Ptr<uint>(desc[0].aabbMngr_changedHandleMap);
    Ptr<uint> addedMap = Ptr<uint>(desc[0].aabbMngr_addedHandleMap);
    Ptr<uint> removedMap = Ptr<uint>(desc[0].aabbMngr_removedHandleMap);

    for (uint handle = dtid.x; handle < numElements; handle += stride)
    {
        const uint word = handle / 32u;
        const uint mask = changedMap[word];
        const uint addedOrRemoved = addedMap[word] | removedMap[word];
        // .cu tests bit (threadIdx.x & 31); blockDim = 256 is a multiple of 32,
        // so (handle & 31) is identical and grid-agnostic.
        if (((mask & (~addedOrRemoved)) & (1u << (handle & 31u))) != 0u)
            writeUpdatedProjections(desc, handle);
    }
}

// ---------------------------------------------------------------------------
// markUpdatedPairsLaunch2 (BP_UPDATE_UPDATEDPAIRS2) — updated-handle array driven
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markUpdatedPairsLaunch2(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint num = desc[0].numUpdatedHandles;
    const uint stride = WorkgroupCount().x * 256u;

    Ptr<uint> updatedHandles = Ptr<uint>(desc[0].updateData_updatedHandles);

    for (uint i = dtid.x; i < num; i += stride)
        writeUpdatedProjections(desc, updatedHandles[i]);
}
