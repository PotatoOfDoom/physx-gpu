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

// solverDmaBack.slang
//
// Slang port of the two solver DMA-back glue kernels from
// gpusolver/src/CUDA/solverMultiBlock.cu:
//   - dmaBackChangedElems: word-wise grid-stride copy of the force-change
//     threshold-element array (produced by createForceChangeThresholdElements)
//     into the host-mapped output.  The CUDA copies
//     sizeof(Dy::ThresholdStreamElement)*nbElems / sizeof(PxU32) = 10*nbElems
//     32-bit words (the element is 40 bytes).
//   - dmaConstraintResidual: gather each constraint's residual — the .w of
//     PxgConstraintWriteback::angularImpulse_residual (the second float4) — into
//     a flat residuals array, one thread per constraint.

struct DmaParams
{
    uint nbElems;     // dmaBackChangedElems: number of ThresholdStreamElements
    uint count;       // dmaConstraintResidual: number of writebacks
    uint gridStride;  // blockDim.x * gridDim.x
}

// Dy::ThresholdStreamElement is 40 bytes = 10 uint words
static const uint ELEMENT_WORDS = 10u;

struct ConstraintWriteback
{
    float4 linearImpulse_broken;
    float4 angularImpulse_residual;
}

[[vk::push_constant]]
ConstantBuffer<DmaParams> gParams;

RWStructuredBuffer<uint> gChangedSrc;              // binding 0  forceChangeThresholdElements (word view)
RWStructuredBuffer<uint> gChangedDst;             // binding 1  host-mapped output (word view)
RWStructuredBuffer<ConstraintWriteback> gWritebacks; // binding 2
RWStructuredBuffer<float> gResiduals;             // binding 3

// ---------------------------------------------------------------------------
// dmaBackChangedElems
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void dmaBackChangedElems(uint3 dtid : SV_DispatchThreadID)
{
    const uint nbWords = ELEMENT_WORDS * gParams.nbElems;
    const uint stride = gParams.gridStride;
    for (uint i = dtid.x; i < nbWords; i += stride)
        gChangedDst[i] = gChangedSrc[i];
}

// ---------------------------------------------------------------------------
// dmaConstraintResidual
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void dmaConstraintResidual(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x < gParams.count)
        gResiduals[dtid.x] = gWritebacks[dtid.x].angularImpulse_residual.w;
}
