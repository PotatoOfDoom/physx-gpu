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

// solverDmaBack.slang — the solver DMA-back glue kernel from
// gpusolver/src/CUDA/solverMultiBlock.cu on the BDA ENGINE ABI (P5 Step 5):
//   - dmaBackChangedElems: word-wise grid-stride copy of the force-change
//     threshold-element array (produced by createForceChangeThresholdElements)
//     into the host-mapped output.  The CUDA copies
//     sizeof(Dy::ThresholdStreamElement)*nbElems / sizeof(PxU32) = 10*nbElems
//     32-bit words (the element is 40 bytes).  The source pointer and the
//     element count are read from the PxgSolverCoreDesc on-device
//     (forceChangeThresholdElements / nbForceChangeElements), exactly like
//     the .cu; the destination is the host-mapped pointer the engine passes.
//   (Its 5.6.1 sibling dmaConstraintResidual went with the solver residual
//   reporting PhysX removed in 5.8; 5.12 has no such kernel, so neither does
//   this module.)
//
// Engine launch packs (PxgCudaSolverCore.cpp, sized PxCudaKernelParam form;
// argSizes transcribed in bridge/PxgVulkanKernelAbi.cpp):
//   dmaBackChangedElems(const PxgSolverCoreDesc*, Dy::ThresholdStreamElement*)
//     -> {8,8} = 16-byte blob; grid = PxgKernelGridDim::DMA_CHANGED_ELEMS,
//     block = PxgKernelBlockDim::DMA_CHANGED_ELEMS = 512; the CUDA
//     blockDim.x * gridDim.x stride becomes 512 * WorkgroupCount().x.

import pxgSlangCommon;
import pxgSlangSolverDesc;

// Dy::ThresholdStreamElement is 40 bytes = 10 uint words
static const uint ELEMENT_WORDS = 10u;

// ---------------------------------------------------------------------------
// dmaBackChangedElems (BDA engine ABI)
// ---------------------------------------------------------------------------
struct DmaChangedElemsAbiParams
{
    uint64_t solverCoreDesc;   // @0  const PxgSolverCoreDesc*
    uint64_t hostChangedElems; // @8  Dy::ThresholdStreamElement* (host-mapped)
}                              // = 16
[[vk::push_constant]]
ConstantBuffer<DmaChangedElemsAbiParams> gDmaChangedAbi;

[shader("compute")]
[numthreads(512, 1, 1)]
void dmaBackChangedElems(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gDmaChangedAbi.solverCoreDesc))[0];

    // Null-guard (llvmpipe): a scene without force thresholds leaves
    // forceChangeThresholdElements unallocated (the engine then never launches
    // this kernel, but the guard keeps the zero-trip loop hoist-safe).
    if (desc.forceChangeThresholdElements == 0 || gDmaChangedAbi.hostChangedElems == 0)
        return;

    Ptr<uint> src = Ptr<uint>(desc.forceChangeThresholdElements);
    Ptr<uint> dst = Ptr<uint>(gDmaChangedAbi.hostChangedElems);

    const uint nbWords = ELEMENT_WORDS * desc.nbForceChangeElements;
    const uint stride = 512u * WorkgroupCount().x; // blockDim.x * gridDim.x
    const uint globalThreadIdx = groupThreadId.x + groupId.x * 512u;

    for (uint i = globalThreadIdx; i < nbWords; i += stride)
        dst[i] = src[i];
}
