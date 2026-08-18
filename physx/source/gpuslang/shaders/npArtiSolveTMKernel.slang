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


// npArtiSolveTMKernel.slang
//
// P8.98c: `artiSolveInternalTendonAndMimicJointConstraints1T`
// (internalConstraints2.cu:2775-2815) — the per-iteration tendon/mimic solve
// kernel, composing:
//
//   1. aiAverageAndPropagate  (P8.94c, npArtiAvgImpulse) — only when the
//      block's mStateDirty carries eHAS_IMPULSES (2)
//   2. svSolveSpatialConstraints  (P8.98b, npArtiSolveTendons)
//   3. svSolveFixedConstraints    (P8.98b)
//   4. svSolveMimicConstraints    (P8.98b)
//
// The wrapper's own surface:
//
//  * launch shape: block (32, 1, 1), grid ceil(nbArticulations / 32)
//    (SOLVE_INTERNAL_CONSTRAINTS = 32, numWarpsPerBlock = 1,
//    PxgArticulationCore.cpp:634-637) — the per-warp slice index is
//    blockIdx.x directly.
//  * the OUTER GUARD: the whole body runs only when the lane has spatial
//    tendons, fixed tendons OR mimic joints — a lane with none leaves
//    EVERYTHING untouched, including its mStateDirty byte (which shares a
//    word with three neighbour lanes: the sub-word store must not clobber
//    them).
//  * mStateDirty: read as a byte; when eHAS_IMPULSES (2) is set the
//    averaging preamble runs; then the byte is set to eVEL_DIRTY (1)
//    UNCONDITIONALLY (inside the guard) via the atomic sub-word store.
//  * the argument pack {desc*, biasCoefficient, dt, recipDt,
//    velocityIteration, isTGS} packs the TWO bools as adjacent bytes at
//    20 and 21 — one word read, velocityIteration in byte 0, isTGS in
//    byte 1 (the P8.83 two-bool arrangement).
//
// Verified by testArtiSolveTMLaunch.cpp: the real kernel over a
// device-resident PxgArticulationCoreDesc against a FULL host mirror
// (aiMirror + the hoisted refSolve* bodies), byte for byte.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangArticulationDesc;
import npArtiSolveTendons;
import npArtiAvgImpulse;

struct StmParams
{
    uint64_t desc;         // @0  PxgArticulationCoreDesc*
    float biasCoefficient; // @8
    float dt;              // @12
    float recipDt;         // @16
    uint boolsWord;        // @20: velocityIteration byte 0, isTGS byte 1
};

[[vk::push_constant]]
ConstantBuffer<StmParams> gStm;

static const uint STM_VEL_DIRTY = 1u;      // PxgArtiStateDirtyFlag::eVEL_DIRTY
static const uint STM_HAS_IMPULSES = 2u;   // PxgArtiStateDirtyFlag::eHAS_IMPULSES

[shader("compute")]
[numthreads(32, 1, 1)]
void artiSolveInternalTendonAndMimicJointConstraints1T(uint3 groupThreadId : SV_GroupThreadID,
                                                       uint3 groupId : SV_GroupID)
{
    const uint64_t d = gStm.desc;

    const uint lane = groupThreadId.x;
    const uint globalThreadIndex = groupId.x * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint64_t w = uint64_t(groupId.x);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;

    // the OUTER GUARD
    if (abU32(artiBlock, PXG_AB_mNumSpatialTendons, lane) != 0u
        || abU32(artiBlock, PXG_AB_mNumFixedTendons, lane) != 0u
        || abU32(artiBlock, PXG_AB_mNumMimicJoints, lane) != 0u)
    {
        const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
        const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

        const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                                + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
        const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                               + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

        const uint dirtyFlag = abU8(artiBlock, PXG_AB_mStateDirty, lane);
        if ((dirtyFlag & STM_HAS_IMPULSES) != 0u)
        {
            const uint64_t slabHasChanges = artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES);
            const uint64_t impulses = artiPtrField(d, PXG_ARTI_CORE_IMPULSES);
            const uint nbSlabs = artiU32Field(d, PXG_ARTI_CORE_NB_SLABS);

            aiAverageAndPropagate(slabHasChanges, impulses, artiBlock, linkBase, dofBase,
                                  globalThreadIndex, maxLinks, nbArticulations, nbSlabs,
                                  lane);
        }

        abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, STM_VEL_DIRTY);

        const uint wordSize = (maxLinks + 63u) / 64u;
        const uint64_t bitFields = artiPtrField(d, PXG_ARTI_CORE_PATH_TO_ROOT_BITFIELD_BLOCKS)
            + w * uint64_t(maxLinks) * uint64_t(wordSize) * PXG_AB_BITFIELD_STACK_SIZE;

        const bool isVelocityIteration = (gStm.boolsWord & 0xFFu) != 0u;
        const bool isTGS = ((gStm.boolsWord >> 8u) & 0xFFu) != 0u;

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

            svSolveSpatialConstraints(artiBlock, linkBase, dofBase, bitFields, wordSize,
                                      tendons, constraints, attachments, maxAttachments,
                                      isTGS, lane);
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

            svSolveFixedConstraints(artiBlock, linkBase, dofBase, bitFields, wordSize,
                                    tendons, constraints, tendonJoints, maxTendonJoints,
                                    lane);
        }

        {
            const uint maxMimicJoints = artiU32Field(d, PXG_ARTI_CORE_MAX_MIMIC_JOINTS);
            const uint64_t mimicJoints = artiPtrField(d, PXG_ARTI_CORE_MIMIC_JOINT_BLOCKS)
                                       + w * uint64_t(maxMimicJoints) * PXG_AB_MJ_SIZE;

            svSolveMimicConstraints(artiBlock, linkBase, dofBase, bitFields, wordSize,
                                    mimicJoints, gStm.biasCoefficient, gStm.dt, gStm.recipDt,
                                    isVelocityIteration, lane);
        }
    }
}
