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
// P5 Step 4 (BDA, engine launch path): the two broad-phase glue kernels,
// driven exactly as the engine launches them (void** form, one bpDesc pointer /
// one raw signal pointer).  clearNewFlagLaunch takes the PxgBroadPhaseDesc
// aggregate BY DEVICE POINTER and walks its nested pointers through the
// byte-exact PxgBroadPhaseDescGpu mirror (layout proven by testBpDescLayout);
// bpSignalComplete stores 1 through the raw signal pointer.  Grid-stride loop,
// thread mapping and handle encoding mirror the .cu verbatim.

import pxgSlangBpDesc;

struct ClearNewFlagParams
{
    uint64_t bpDesc; // device address of the PxgBroadPhaseDesc aggregate
}
[[vk::push_constant]]
ConstantBuffer<ClearNewFlagParams> gClear;

// mirrors clearNewFlagLaunch(const PxgBroadPhaseDesc* bpDesc)  // BP_CLEAR_NEWFLAG
// engine grid: PxgBPKernelGridDim::BP_CLEAR_NEWFLAG blocks x 256 threads,
// grid-stride over numCreatedHandles.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearNewFlagLaunch(uint3 dtid : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
                        uint3 gtid : SV_GroupThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gClear.bpDesc);
    const uint numCreatedHandleSize = desc[0].numCreatedHandles;

    Ptr<uint> createdHandles = Ptr<uint>(desc[0].updateData_createdHandles);
    Ptr<uint> boxSapBox1DX = Ptr<uint>(desc[0].boxSapBox1D[0]); // PxgSapBox1D = 2 uints
    Ptr<uint> boxSapBox1DY = Ptr<uint>(desc[0].boxSapBox1D[1]);
    Ptr<uint> boxSapBox1DZ = Ptr<uint>(desc[0].boxSapBox1D[2]);
    Ptr<uint> boxHandlesX = Ptr<uint>(desc[0].boxHandles[0]); // boxHandles[0][X]
    Ptr<uint> boxHandlesY = Ptr<uint>(desc[0].boxHandles[1]); // boxHandles[0][Y]
    Ptr<uint> boxHandlesZ = Ptr<uint>(desc[0].boxHandles[2]); // boxHandles[0][Z]

    // CUDA grid-stride loop: stride = blockDim.x * gridDim.x, with gridDim
    // available as the NumWorkgroups builtin (WorkgroupCount()).
    const uint stride = WorkgroupCount().x * 256u;
    for (uint i = dtid.x; i < numCreatedHandleSize; i += stride)
    {
        const uint objectHandle = createdHandles[i];
        const uint startHandle = bpCreateHandle(objectHandle, true, false);
        const uint endHandle = bpCreateHandle(objectHandle, false, false);

        boxHandlesX[boxSapBox1DX[objectHandle * 2u + 0u]] = startHandle;
        boxHandlesX[boxSapBox1DX[objectHandle * 2u + 1u]] = endHandle;
        boxHandlesY[boxSapBox1DY[objectHandle * 2u + 0u]] = startHandle;
        boxHandlesY[boxSapBox1DY[objectHandle * 2u + 1u]] = endHandle;
        boxHandlesZ[boxSapBox1DZ[objectHandle * 2u + 0u]] = startHandle;
        boxHandlesZ[boxSapBox1DZ[objectHandle * 2u + 1u]] = endHandle;
    }
}

struct SignalParams
{
    uint64_t signal; // device address of the completion-signal word
}
[[vk::push_constant]]
ConstantBuffer<SignalParams> gSignalP;

// mirrors bpSignalComplete(PxU32* signal)  // BP_SIGNAL_COMPLETE (grid 1x1)
[shader("compute")]
[numthreads(1, 1, 1)]
void bpSignalComplete(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x == 0u)
        (Ptr<uint>(gSignalP.signal))[0] = 1u;
}
