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

// broadphaseClearNewFlag.slang
//
// Slang port of the two remaining broad-phase glue kernels
// (gpubroadphase/src/CUDA/broadphase.cu):
//   - clearNewFlagLaunch: after the first frame that inserted a batch of newly
//     created boxes, rewrite the current frame's start/end sort handles for
//     those boxes with the "new" flag cleared (isNew = false), so they are no
//     longer treated as freshly inserted on the next incremental pass.
//   - bpSignalComplete: a single-thread store of 1 into the completion-signal
//     word (the CPU polls this to know the broad-phase dispatch chain is done).
//
// Both are pure per-thread scatter kernels — no groupshared, scans, atomics or
// barriers.  The per-axis arrays are packed into flat buffers.

static const uint AXIS_COUNT = 3;

struct ClearParams
{
    uint numHandles;         // nbProjections = numHandles * 2
    uint numCreatedHandles;  // grid-stride loop bound
    uint gridDimX;           // groups launched (grid-stride)
}

[[vk::push_constant]]
ConstantBuffer<ClearParams> gParams;

RWStructuredBuffer<uint> gCreatedHandles;  // binding 0  updateData_createdHandles
RWStructuredBuffer<uint> gSapBox;          // binding 1  boxSapBox1D[axis] flat [3][numHandles][2]
RWStructuredBuffer<uint> gBoxHandles0;     // binding 2  boxHandles[0][axis] flat [3][nbProj] (current)
RWStructuredBuffer<uint> gSignal;          // binding 3  completion-signal word

// createHandle(handle, isStart, isNew) — PxgBroadPhaseDesc.h
uint createHandle(uint handle, bool isStart, bool isNew)
{
    return (handle << 3) | (isStart ? 1u : 0u) | (isNew ? 2u : 0u);
}

// sapBox[axis][handle].mMinMax[i]
uint sapPos(uint axis, uint handle, uint numHandles, uint i)
{
    return gSapBox[(axis * numHandles + handle) * 2u + i];
}

// ---------------------------------------------------------------------------
// clearNewFlagLaunch (BP_CLEAR_NEWFLAG)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void clearNewFlagLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint numHandles = gParams.numHandles;
    const uint nbProj = numHandles * 2u;
    const uint num = gParams.numCreatedHandles;
    const uint stride = gParams.gridDimX * 256u;

    for (uint i = dtid.x; i < num; i += stride)
    {
        const uint objectHandle = gCreatedHandles[i];
        const uint startHandle = createHandle(objectHandle, true, false);
        const uint endHandle = createHandle(objectHandle, false, false);
        [unroll]
        for (uint axis = 0; axis < AXIS_COUNT; ++axis)
        {
            const uint minIndex = sapPos(axis, objectHandle, numHandles, 0u);
            const uint maxIndex = sapPos(axis, objectHandle, numHandles, 1u);
            const uint base = axis * nbProj;
            gBoxHandles0[base + minIndex] = startHandle;
            gBoxHandles0[base + maxIndex] = endHandle;
        }
    }
}

// ---------------------------------------------------------------------------
// bpSignalComplete (BP_SIGNAL_COMPLETE)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void bpSignalComplete(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x == 0u)
        gSignal[0] = 1u;
}
