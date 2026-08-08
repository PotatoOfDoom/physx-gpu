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

// dmaArtiResidualBda.slang
//
// dmaArticulationResidual (gpuarticulation/src/CUDA/forwardDynamic2.cu:3766) on
// the ENGINE launch ABI.
//
// One thread per articulation.  It reads the articulation's two residual
// accumulators out of the pool and writes them into a PINNED HOST array the
// solver reads back on the CPU to report per-articulation convergence error:
// slot [index] takes the internal (position-iteration) residual, slot
// [index + nbArticulations] the contact residual.
//
// Small, but it is the kernel that forces the `PxgArticulationCoreDesc` address
// map, which every remaining articulation tranche needs — the solver block
// threads that one descriptor through all of it.
//
// ===========================================================================
// THE INDEX CHAIN, AND WHY IT IS THREE HOPS AND NOT ONE
// ===========================================================================
//
//   islandNodeIndices[threadIdx + articulationOffset].index()  -> nodeIndex
//   bodySimPool[nodeIndex].articulationRemapId                 -> index
//   articulations[index]                                       -> the record
//
// The dispatch index is a position in the island's articulation range, NOT an
// articulation id: `articulationOffset` is where the articulations begin inside
// the island's node-index array.  The node index then names a body-sim slot,
// and only that slot carries the articulation's pool index.  All three hops are
// distinguishable in the fixture, which uses a non-zero offset and a remap that
// is neither the identity nor the node index.
//
// Note the asymmetry the CUDA deliberately keeps: the OUTPUT is indexed by
// `index` (the pool index), not by the dispatch index.  Two articulations at
// different island positions therefore write to slots determined by the remap,
// which is what makes a remap bug visible on the host side.
//
// ===========================================================================
// THE PINNED-HOST ARGUMENT
// ===========================================================================
//
// `errorPinnedHost` is `PxPinnedArray<Dy::ErrorAccumulator>::begin()` cast
// straight to CUdeviceptr (PxgTGSCudaSolverCore.cpp:1601) — under CUDA's
// unified addressing a pinned host pointer IS a device pointer, so PhysX hands
// it over untranslated.  Here the CPU mapping and the buffer device address are
// unrelated numbers, so the launch-ABI row marks argument 1 in `hostPtrMask`
// and the bridge substitutes the device address on the way in.  Without that
// bit the shader would write to a wild address.

import pxgSlangCommon;
import pxgSlangArticulationDesc;

// PxgArticulationCoreKernelBlockDim has no entry for this one: the solver core
// launches it with a literal 256-thread block and
// ceil(nbArticulations / 256) blocks (PxgTGSCudaSolverCore.cpp:1608).
static const uint DAR_BLOCK_SIZE = 256;

// ENGINE launch ABI: exactly the CUDA parameter list — two device pointers
// ({8,8} blob), the second of which arrives as a pinned-host address.
struct DmaArtiResidualParams
{
    uint64_t scDesc;         // PxgArticulationCoreDesc*
    uint64_t errorPinnedHost; // Dy::ErrorAccumulator*
};

[[vk::push_constant]]
ConstantBuffer<DmaArtiResidualParams> gDmaArtiResidualAbi;

// Dy::ErrorAccumulator is 12 bytes of plain scalars; the CUDA's whole-struct
// assignment is a 12-byte copy, reproduced as three u32 writes.
void darCopyAccumulator(uint64_t dst, uint64_t src)
{
    Ptr<uint> d = Ptr<uint>(dst);
    Ptr<uint> s = Ptr<uint>(src);
    for (uint i = 0; i < (PXG_ARTI_ERROR_ACCUMULATOR_SIZE / 4u); ++i)
        d[i] = s[i];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void dmaArticulationResidual(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint64_t scDesc = gDmaArtiResidualAbi.scDesc;

    const uint nbArticulations = artiU32Field(scDesc, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint globalThreadIdx = dispatchThreadId.x;

    if (globalThreadIdx >= nbArticulations)
        return;

    const uint articulationOffset = artiU32Field(scDesc, PXG_ARTI_CORE_ARTICULATION_OFFSET);
    const uint64_t islandNodeIndices = artiPtrField(scDesc, PXG_ARTI_CORE_ISLAND_NODE_INDICES);
    const uint64_t bodySimPool = artiPtrField(scDesc, PXG_ARTI_CORE_BODY_SIM_POOL);
    const uint64_t articulations = artiPtrField(scDesc, PXG_ARTI_CORE_ARTICULATIONS);

    const uint2 node = (Ptr<uint2>(islandNodeIndices))[globalThreadIdx + articulationOffset];
    const uint nodeIndex = nodeIndexIndex(node);

    const uint index = artiU32Field(bodySimPool + uint64_t(nodeIndex) * PXG_BODY_SIM_SIZE,
                                   PXG_BODY_SIM_ARTICULATION_REMAP_ID);

    const uint64_t articulation = articulations + uint64_t(index) * PXG_ARTI_STRIDE;
    const uint64_t out = gDmaArtiResidualAbi.errorPinnedHost;

    // the position-iteration residual: this kernel runs after the position
    // iterations and before the velocity iterations
    darCopyAccumulator(out + uint64_t(index) * PXG_ARTI_ERROR_ACCUMULATOR_SIZE,
                       articulation + PXG_ARTI_INTERNAL_RESIDUAL);

    darCopyAccumulator(out + uint64_t(index + nbArticulations) * PXG_ARTI_ERROR_ACCUMULATOR_SIZE,
                       articulation + PXG_ARTI_CONTACT_RESIDUAL);
}
