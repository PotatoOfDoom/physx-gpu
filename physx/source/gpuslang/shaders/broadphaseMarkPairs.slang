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
// scans, atomics or barriers.  Per-axis arrays are packed into flat buffers.

static const uint AXIS_COUNT = 3;
static const uint PXG_INVALID_BP_PROJECTION = 0xffffffffu;
static const uint PXG_INVALID_BP_SAP_BOX = 0xffffffffu;

struct MarkParams
{
    uint numHandles;         // nbProjections = numHandles * 2
    uint numRemovedHandles;
    uint numUpdatedHandles;
    uint changedWordCounts;  // aabbMngr_changedHandleBitMapWordCounts
    uint gridDimX;           // groups launched (grid-stride)
}

[[vk::push_constant]]
ConstantBuffer<MarkParams> gParams;

RWStructuredBuffer<uint> gRemovedHandles;  // binding 0  updateData_removedHandles
RWStructuredBuffer<uint> gUpdatedHandles;  // binding 1  updateData_updatedHandles
RWStructuredBuffer<uint> gSapBox;          // binding 2  boxSapBox1D[axis] flat [3][numHandles][2]
RWStructuredBuffer<uint> gBoxHandles1;     // binding 3  boxHandles[1][axis] flat [3][nbProj] (previous)
RWStructuredBuffer<uint> gBoxProjections;  // binding 4  boxProjections[axis] flat [3][nbProj]
RWStructuredBuffer<uint> gOldBounds;       // binding 5  oldIntegerBounds [numHandles][6]
RWStructuredBuffer<uint> gNewBounds;       // binding 6  newIntegerBounds [numHandles][6]
RWStructuredBuffer<uint> gChangedMap;      // binding 7  aabbMngr_changedHandleMap [words]
RWStructuredBuffer<uint> gAddedMap;        // binding 8  aabbMngr_addedHandleMap [words]
RWStructuredBuffer<uint> gRemovedMap;      // binding 9  aabbMngr_removedHandleMap [words]

uint markDeleted(uint handle) { return handle | (1u << 2); }

// sapBox[axis][handle].mMinMax[i]
uint sapPos(uint axis, uint handle, uint numHandles, uint i)
{
    return gSapBox[(axis * numHandles + handle) * 2u + i];
}

// ---------------------------------------------------------------------------
// markRemovedPairsLaunch (BP_MARK_DELETEDPAIRS)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markRemovedPairsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint numHandles = gParams.numHandles;
    const uint nbProj = numHandles * 2u;
    const uint num = gParams.numRemovedHandles;
    const uint stride = gParams.gridDimX * 256u;

    for (uint i = dtid.x; i < num; i += stride)
    {
        const uint handle = gRemovedHandles[i];
        [unroll]
        for (uint axis = 0; axis < AXIS_COUNT; ++axis)
        {
            const uint minIndex = sapPos(axis, handle, numHandles, 0u);
            const uint maxIndex = sapPos(axis, handle, numHandles, 1u);
            const uint base = axis * nbProj;
            gBoxHandles1[base + minIndex] = markDeleted(gBoxHandles1[base + minIndex]);
            gBoxHandles1[base + maxIndex] = markDeleted(gBoxHandles1[base + maxIndex]);
        }
        // setEmpty on the old integer bounds
        const uint b = handle * 6u;
        gOldBounds[b + 0u] = 0xff7fffffu;
        gOldBounds[b + 1u] = 0xff7fffffu;
        gOldBounds[b + 2u] = 0xff7fffffu;
        gOldBounds[b + 3u] = 0x00800000u;
        gOldBounds[b + 4u] = 0x00800000u;
        gOldBounds[b + 5u] = 0x00800000u;
    }
}

// ---------------------------------------------------------------------------
// markRemovedPairsProjectionsLaunch (BP_UPDATE_DELETEDPAIRS)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markRemovedPairsProjectionsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint numHandles = gParams.numHandles;
    const uint nbProj = numHandles * 2u;
    const uint num = gParams.numRemovedHandles;
    const uint stride = gParams.gridDimX * 256u;

    for (uint i = dtid.x; i < num; i += stride)
    {
        const uint handle = gRemovedHandles[i];
        [unroll]
        for (uint axis = 0; axis < AXIS_COUNT; ++axis)
        {
            const uint sb = (axis * numHandles + handle) * 2u;
            const uint minIndex = gSapBox[sb + 0u];
            const uint maxIndex = gSapBox[sb + 1u];
            const uint base = axis * nbProj;
            gBoxProjections[base + minIndex] = PXG_INVALID_BP_PROJECTION;
            gBoxProjections[base + maxIndex] = PXG_INVALID_BP_PROJECTION;
            gSapBox[sb + 0u] = PXG_INVALID_BP_SAP_BOX;
            gSapBox[sb + 1u] = PXG_INVALID_BP_SAP_BOX;
        }
    }
}

// write the box's new min/max projections at its per-axis SAP-box positions
void writeUpdatedProjections(uint handle, uint numHandles, uint nbProj)
{
    const uint nb = handle * 6u;
    [unroll]
    for (uint axis = 0; axis < AXIS_COUNT; ++axis)
    {
        const uint minV = gNewBounds[nb + axis];
        const uint maxV = gNewBounds[nb + axis + 3u];
        const uint minIndex = sapPos(axis, handle, numHandles, 0u);
        const uint maxIndex = sapPos(axis, handle, numHandles, 1u);
        const uint base = axis * nbProj;
        gBoxProjections[base + minIndex] = minV;
        gBoxProjections[base + maxIndex] = maxV;
    }
}

// ---------------------------------------------------------------------------
// markUpdatedPairsLaunch (BP_UPDATE_UPDATEDPAIRS) — bitmap driven
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markUpdatedPairsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint numHandles = gParams.numHandles;
    const uint nbProj = numHandles * 2u;
    const uint numElements = gParams.changedWordCounts * 32u;
    const uint stride = gParams.gridDimX * 256u;

    for (uint handle = dtid.x; handle < numElements; handle += stride)
    {
        const uint word = handle / 32u;
        const uint mask = gChangedMap[word];
        const uint addedOrRemoved = gAddedMap[word] | gRemovedMap[word];
        if (((mask & (~addedOrRemoved)) & (1u << (handle & 31u))) != 0u)
            writeUpdatedProjections(handle, numHandles, nbProj);
    }
}

// ---------------------------------------------------------------------------
// markUpdatedPairsLaunch2 (BP_UPDATE_UPDATEDPAIRS2) — updated-handle array driven
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markUpdatedPairsLaunch2(uint3 dtid : SV_DispatchThreadID)
{
    const uint numHandles = gParams.numHandles;
    const uint nbProj = numHandles * 2u;
    const uint num = gParams.numUpdatedHandles;
    const uint stride = gParams.gridDimX * 256u;

    for (uint i = dtid.x; i < num; i += stride)
        writeUpdatedProjections(gUpdatedHandles[i], numHandles, nbProj);
}
