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


// npArtiSetupICKernel.slang
//
// P8.97d: `setupInternalConstraintLaunch1T` (internalConstraints2.cu:1200-1266)
// — the assembled setup kernel for the articulation solve chain, composing the
// three swept setup bodies in the engine's order:
//
//   1. siSetupInternalConstraints   (P8.97a, npArtiSetupInternal)
//   2. stSetupSpatialTendons        (P8.97b, npArtiSetupTendons)
//   3. stSetupFixedTendons          (P8.97b, npArtiSetupTendons)
//   4. mjSetupMimicJoints           (P8.97c, npArtiSetupMimic)
//
// The wrapper's own surface is the index chain plus the argument pack:
//
//  * launch shape: block (32, 1, 1), grid ceil(nbArticulations / 32) — the
//    host passes numWarpsPerBlock = 1 (PxgArticulationCore.cpp:492-517), the
//    P8.88 one-warp shape.  SIC_WARPS_PER_BLOCK is per-kernel on purpose; a
//    port that reuses P8.85's 2 skips every odd warp's articulations.
//  * per-warp slices: artiBlock advances by ONE block (stride PXG_AB_BLOCK_SIZE,
//    no maxLinks factor); the two tendon-constraint slices carry BOTH factors
//    (maxTendons * maxElements * PXG_AB_ITC_SIZE); the bitfield slice is
//    maxLinks * wordSize rows of PXG_AB_BITFIELD_STACK_SIZE with
//    wordSize = (maxLinks + 63) / 64 — the wrapper computes it, the descriptor
//    does not carry it.
//  * the argument pack is {desc*, stepDt, dt, invDt, isTGSSolver} — CUDA packs
//    the bool as ONE byte at offset 20, so the shader reads the word and masks
//    (the updateBodiesLaunch_Part2 arrangement).
//  * the spatial and fixed tendon constraint arrays are two DIFFERENT
//    descriptor pointers (200 / 224, mapped in P8.97b's groundwork) writing
//    interleaved-compatible ITC blocks; swapping them compiles and runs.
//
// Verified by the GPU-differential fixture testArtiSetupICLaunch.cpp: the real
// kernel over a device-resident PxgArticulationCoreDesc versus the four
// already-swept per-family test entries dispatched slice-by-slice on an
// identically seeded second arena, all buffers compared byte for byte.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangArticulationDesc;
import npArtiSetupInternal;
import npArtiSetupTendons;
import npArtiSetupMimic;

struct SicParams
{
    uint64_t desc;    // @0  PxgArticulationCoreDesc*
    float stepDt;     // @8
    float dt;         // @12
    float invDt;      // @16
    uint isTGSWord;   // @20: the CUDA bool in byte 0
};

[[vk::push_constant]]
ConstantBuffer<SicParams> gSic;

// ONE warp per block (numWarpsPerBlock = 1 on the host) -- deliberately not
// shared with the (32, 2, 1) kernels of the unconstrained chain
static const uint SIC_WARPS_PER_BLOCK = 1u;

[shader("compute")]
[numthreads(32, 1, 1)]
void setupInternalConstraintLaunch1T(uint3 groupThreadId : SV_GroupThreadID,
                                     uint3 groupId : SV_GroupID)
{
    const uint64_t d = gSic.desc;

    const uint lane = groupThreadId.x & 31u;
    const uint globalWarpIndex = groupId.x * SIC_WARPS_PER_BLOCK + groupThreadId.y;
    const uint globalThreadIndex = globalWarpIndex * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint64_t w = uint64_t(globalWarpIndex);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint64_t artiLinks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                             + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t artiDofs = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                            + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const float stepDt = gSic.stepDt;
    const float dt = gSic.dt;
    const float invDt = gSic.invDt;
    const bool isTGSSolver = (gSic.isTGSWord & 0xFFu) != 0u;

    siSetupInternalConstraints(artiBlock, artiLinks, artiDofs,
                               stepDt, dt, invDt, isTGSSolver, lane);

    {
        const uint maxSpatialTendons = artiU32Field(d, PXG_ARTI_CORE_MAX_SPATIAL_TENDONS);
        const uint maxAttachments = artiU32Field(d, PXG_ARTI_CORE_MAX_ATTACHMENTS);

        const uint64_t tendons = artiPtrField(d, PXG_ARTI_CORE_SPATIAL_TENDON_BLOCKS)
                               + w * uint64_t(maxSpatialTendons) * PXG_AB_ST_SIZE;
        const uint64_t constraints =
            artiPtrField(d, PXG_ARTI_CORE_SPATIAL_TENDON_CONSTRAINT_BLOCKS)
            + w * uint64_t(maxSpatialTendons) * uint64_t(maxAttachments) * PXG_AB_ITC_SIZE;
        const uint64_t attachments = artiPtrField(d, PXG_ARTI_CORE_ATTACHMENT_BLOCKS)
            + w * uint64_t(maxSpatialTendons) * uint64_t(maxAttachments) * PXG_AB_AT_SIZE;

        stSetupSpatialTendons(artiBlock, artiLinks, artiDofs, tendons, constraints,
                              attachments, maxAttachments,
                              stepDt, dt, invDt, isTGSSolver, lane);
    }

    {
        const uint maxFixedTendons = artiU32Field(d, PXG_ARTI_CORE_MAX_FIXED_TENDONS);
        const uint maxTendonJoints = artiU32Field(d, PXG_ARTI_CORE_MAX_TENDON_JOINTS);

        const uint64_t tendons = artiPtrField(d, PXG_ARTI_CORE_FIXED_TENDON_BLOCKS)
                               + w * uint64_t(maxFixedTendons) * PXG_AB_FT_SIZE;
        const uint64_t constraints =
            artiPtrField(d, PXG_ARTI_CORE_FIXED_TENDON_CONSTRAINT_BLOCKS)
            + w * uint64_t(maxFixedTendons) * uint64_t(maxTendonJoints) * PXG_AB_ITC_SIZE;
        const uint64_t tendonJoints = artiPtrField(d, PXG_ARTI_CORE_TENDON_JOINT_BLOCKS)
            + w * uint64_t(maxFixedTendons) * uint64_t(maxTendonJoints) * PXG_AB_TJ_SIZE;

        stSetupFixedTendons(artiBlock, artiLinks, artiDofs, tendons, constraints,
                            tendonJoints, maxTendonJoints,
                            stepDt, dt, invDt, isTGSSolver, lane);
    }

    {
        const uint wordSize = (maxLinks + 63u) / 64u;
        const uint64_t bitFields = artiPtrField(d, PXG_ARTI_CORE_PATH_TO_ROOT_BITFIELD_BLOCKS)
            + w * uint64_t(maxLinks) * uint64_t(wordSize) * PXG_AB_BITFIELD_STACK_SIZE;

        const uint maxMimicJoints = artiU32Field(d, PXG_ARTI_CORE_MAX_MIMIC_JOINTS);
        const uint64_t mimicJoints = artiPtrField(d, PXG_ARTI_CORE_MIMIC_JOINT_BLOCKS)
                                   + w * uint64_t(maxMimicJoints) * PXG_AB_MJ_SIZE;

        mjSetupMimicJoints(artiBlock, artiLinks, artiDofs, bitFields, wordSize,
                           mimicJoints, lane);
    }
}
