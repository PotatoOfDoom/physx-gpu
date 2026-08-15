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


// npArtiSumSelfBatches.slang
//
// P8.96c: `artiSumSelfContactAndJointBatches`
// (internalConstraints2.cu:3441-3749) -- the SELF-batch scatter, third and
// last kernel of the articulation batch-prep chain.  Batches1 already
// computed the self temp sums (the four TEMP_SELF_* blocks); this kernel
// turns them into batch headers and interleaved unique-id arrays placed
// AFTER the static batches, plus the per-articulation-block self bookkeeping.
//
// SERIAL RECOMPUTE, third use of the P8.96a strategy: every warp op is over
// values deterministically computable from the inputs, so one thread per
// block recomputes histogram, warp scans and scatters as plain loops.
// ZERO new ABI: the four TEMP_SELF_* constants are P8.96a's, the prep/solver
// descriptor fields are the P5 layer, the two per-BLOCK scalars
// (mTotalSelfConstraintCount / mSelfConstraintOffset, 7072/7076) are P8.68's.
//
// Differences from the static twin (npArtiSumBatches2), each earned by the
// reading rather than assumed:
//  1. NO PartitionNodeData bookkeeping at all -- the per-link static
//     count/start fields are the static kernel's job.  Instead each warp's
//     lane 0 writes TWO PER-BLOCK SCALARS into PxgArticulationBlockData:
//     mTotalSelfConstraintCount = maxContact + maxJoint (the WARP maxes) and
//     mSelfConstraintOffset = the warp's INITIAL blockOffset -- written
//     BEFORE the two scatter loops advance the cursor, and written even when
//     both maxes are zero (guard is only workIndex < nbArticulations for the
//     warp's lane-0 workIndex).
//  2. Every derived offset gains the STATIC term: articulationBatchOffset
//     adds numArtiStaticBatches, both unique-index offsets add the
//     numTotalStaticArti* term (four addends, not three), the header batch
//     index uses artiStatic*BatchOffset + numArtiStatic*Batches, and the
//     batch-indices arrays add numArtiStatic*Batches on top of the dynamic
//     counts.  All six sums have a distinct-scalar fixture case behind them.
//  3. The single blockOffset cursor still runs contacts first, joints
//     continuing -- unchanged (same trap as the twin, same mutant).
//  4. Block 0's totals go to numArtiSelfBatches / numArtiSelfContactBatches /
//     numArtiSelf1dConstraintBatches and the island's mSelfArtiBatchCount
//     (@56 -- the field AFTER the static kernel's @52).

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangPrepCommon;
import pxgSlangPrepDesc;
import pxgSlangSolverDesc;

static const uint ASSB_BLOCK_DIM = 512u;
static const uint ASSB_WARPS = 16u;
static const uint ASSB_GRID = 32u;

// CLAMP THE INDEX, NOT JUST THE VALUE (the P8.96c finding, dmesg-verified in
// the batches2 twin): llvmpipe compiles this guarded uniform load to an
// UNCONDITIONAL scalar load + select, so the unclamped form reads tens of KB
// past the counts array and segfaults on an unlucky allocation layout.  This
// kernel never drew the unlucky layout in testing, but it carries the same
// pattern; fixed together with both twins.
uint assbCountAt(Ptr<uint> counts, uint workIndex, uint nb)
{
    const uint idx = workIndex < nb ? workIndex : 0u;
    const uint v = counts[idx];
    return workIndex < nb ? v : 0u;
}

struct AssbParams
{
    uint64_t scDesc;             // @0
    uint64_t solverCoreDesc;     // @8
    uint64_t prePrepDesc;        // @16
    uint64_t constraintPrepDesc; // @24
    uint nbArticulations;        // @32
    uint pad0;                   // @36 -> 40
};

[[vk::push_constant]]
ConstantBuffer<AssbParams> gAssb;

[shader("compute")]
[numthreads(512, 1, 1)]
void artiSumSelfContactAndJointBatches(uint3 groupId : SV_GroupID,
                                       uint3 groupThreadId : SV_GroupThreadID)
{
    if (groupThreadId.x != 0u)
        return;
    const uint b = groupId.x;

    const uint64_t d = gAssb.scDesc;
    Ptr<PxgPrePrepDescGpu> pp = Ptr<PxgPrePrepDescGpu>(gAssb.prePrepDesc);
    Ptr<PxgConstraintPrepareDescGpu> cp =
        Ptr<PxgConstraintPrepareDescGpu>(gAssb.constraintPrepDesc);
    Ptr<PxgSolverCoreDescGpu> sc = Ptr<PxgSolverCoreDescGpu>(gAssb.solverCoreDesc);
    const uint nb = gAssb.nbArticulations;

    // the wrapper's derived offsets (internalConstraints2.cu:3461-3469) --
    // difference 2: the static terms join every sum
    const uint numRigidContacts = pp[0].numTotalContacts;
    const uint numRigidJoints = pp[0].numTotalConstraints;
    const uint numRigidStaticContacts = pp[0].numTotalStaticContacts;
    const uint numRigidStaticJoints = pp[0].numTotalStaticConstraints;
    const uint articulationBatchOffset =
        pp[0].numBatches + pp[0].numArtiBatches + pp[0].numArtiStaticBatches;
    const uint contactUniqueIndexOffset =
        numRigidContacts + pp[0].numTotalStaticContacts + pp[0].numTotalArtiContacts
        + pp[0].numTotalStaticArtiContacts;
    const uint jointUniqueIndexOffset =
        numRigidJoints + pp[0].numTotalStaticConstraints + pp[0].numTotalArtiConstraints
        + pp[0].numTotalStaticArtiConstraints;
    const uint numRigidBatches = cp[0].numBatches;
    const uint numArtiContactBatches = cp[0].numArtiContactBatches;
    const uint numArtiJointBatches = cp[0].numArti1dConstraintBatches;
    // :3553-3556 -- read AFTER the static kernel wrote them (stream order)
    const uint numArtiStaticContacts = cp[0].numArtiStaticContactBatches;
    const uint numArtiStaticJoints = cp[0].numArtiStatic1dConstraintBatches;
    const uint artiSelfContactBatchOffset =
        pp[0].artiStaticContactBatchOffset + numArtiStaticContacts;
    const uint artiSelfJointBatchOffset =
        pp[0].artiStaticConstraintBatchOffset + numArtiStaticJoints;

    Ptr<uint> selfContactCount = Ptr<uint>(pp[0].mArtiSelfContactCounts);
    Ptr<uint> selfJointCount = Ptr<uint>(pp[0].mArtiSelfConstraintCounts);
    Ptr<uint> contactSelfUniqueIds = Ptr<uint>(pp[0].mArtiSelfContactIndices);
    Ptr<uint> jointSelfUniqueIds = Ptr<uint>(pp[0].mArtiSelfConstraintIndices);
    Ptr<PxgConstraintBatchHeaderGpu> batchHeaders =
        Ptr<PxgConstraintBatchHeaderGpu>(pp[0].mBatchHeaders);
    Ptr<uint> outContactUniqueIds = Ptr<uint>(pp[0].mContactUniqueIndices);
    Ptr<uint> outJointUniqueIds = Ptr<uint>(pp[0].mConstraintUniqueIndices);
    Ptr<uint> artiContactBatchIdx = Ptr<uint>(cp[0].artiContactConstraintBatchIndices);
    Ptr<uint> artiJointBatchIdx = Ptr<uint>(cp[0].artiJointConstraintBatchIndices);

    Ptr<uint> tmpCU = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONTACT_UNIQUE));
    Ptr<uint> tmpJU = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONSTRAINT_UNIQUE));
    Ptr<uint> tmpCH = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONTACT_HEADER));
    Ptr<uint> tmpJH = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONSTRAINT_HEADER));

    // the 32-entry block histogram: EXCLUSIVE prefix at [b], plus the
    // inclusive totals for block 0's four writes (:3512-3540)
    uint histCU = 0u, histJU = 0u, histCH = 0u, histJH = 0u;
    uint totCH = 0u, totJH = 0u;
    for (uint i = 0u; i < ASSB_GRID; ++i)
    {
        if (i < b)
        {
            histCU += tmpCU[i];
            histJU += tmpJU[i];
            histCH += tmpCH[i];
            histJH += tmpJH[i];
        }
        totCH += tmpCH[i];
        totJH += tmpJH[i];
    }
    if (b == 0u)
    {
        pp[0].numArtiSelfBatches = totCH + totJH;
        cp[0].numArtiSelfContactBatches = totCH;
        cp[0].numArtiSelf1dConstraintBatches = totJH;
        // islandContextPool[0].mSelfArtiBatchCount (@56 in PxgIslandContext,
        // pinned by the solver-desc mirror)
        Ptr<uint> island = Ptr<uint>(sc[0].islandContextPool + 56u);
        island[0] = totCH + totJH;
    }

    const uint totalBlockRequired = (nb + (ASSB_BLOCK_DIM - 1u)) / ASSB_BLOCK_DIM;
    const uint numIter = (totalBlockRequired + 31u) / 32u;

    const uint64_t artiBlocks = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS);

    // the four Accum values, once per block, carried across iterations
    uint accCU = 0u, accJU = 0u, accCH = 0u, accJH = 0u;

    for (uint a = 0u; a < numIter; ++a)
    {
        // the 16 warp sums/maxes, then their exclusive scans
        uint wSumC[16]; uint wSumJ[16]; uint wMaxC[16]; uint wMaxJ[16];
        for (uint w = 0u; w < ASSB_WARPS; ++w)
        {
            uint sC = 0u, sJ = 0u, mC = 0u, mJ = 0u;
            for (uint l = 0u; l < 32u; ++l)
            {
                const uint wi = a * ASSB_BLOCK_DIM + (w * 32u + l) + numIter * b * ASSB_BLOCK_DIM;
                const uint c = assbCountAt(selfContactCount, wi, nb);
                const uint j = assbCountAt(selfJointCount, wi, nb);
                sC += c; sJ += j;
                mC = c > mC ? c : mC;
                mJ = j > mJ ? j : mJ;
            }
            wSumC[w] = sC; wSumJ[w] = sJ; wMaxC[w] = mC; wMaxJ[w] = mJ;
        }

        uint iterCU = 0u, iterJU = 0u, iterCH = 0u, iterJH = 0u;  // this iteration's totals
        for (uint w = 0u; w < ASSB_WARPS; ++w)
        {
            iterCU += wSumC[w]; iterJU += wSumJ[w]; iterCH += wMaxC[w]; iterJH += wMaxJ[w];
        }

        uint excCU = 0u, excJU = 0u, excCH = 0u, excJH = 0u;      // running exclusive scan
        for (uint w = 0u; w < ASSB_WARPS; ++w)
        {
            uint contactOffset = contactUniqueIndexOffset + excCU + histCU + accCU;
            uint jointOffset = jointUniqueIndexOffset + excJU + histJU + accJU;
            uint contactBlockOffset = excCH + histCH + accCH;
            uint jointBlockOffset = excJH + histJH + accJH;
            uint blockOffset = contactBlockOffset + jointBlockOffset + articulationBatchOffset;

            // difference 1: the two per-BLOCK scalars, written by the warp's
            // lane 0 BEFORE the scatter loops advance blockOffset, whenever
            // the warp's first articulation exists -- zero maxes included
            const uint warpBase = a * ASSB_BLOCK_DIM + w * 32u + numIter * b * ASSB_BLOCK_DIM;
            if (warpBase < nb)
            {
                const uint64_t artiBlock = artiBlocks
                                         + uint64_t(warpBase / 32u) * PXG_AB_BLOCK_SIZE;
                abStoreBlockScalar(artiBlock, PXG_AB_mTotalSelfConstraintCount,
                                   wMaxC[w] + wMaxJ[w]);
                abStoreBlockScalar(artiBlock, PXG_AB_mSelfConstraintOffset, blockOffset);
            }

            // ---- contacts ----
            for (uint i = 0u; i < wMaxC[w]; ++i)
            {
                uint mask = 0u, stride = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = warpBase + l;
                    if (assbCountAt(selfContactCount, wi, nb) > i)
                    {
                        mask |= (1u << l);
                        ++stride;
                    }
                }
                uint excl = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = warpBase + l;
                    if (assbCountAt(selfContactCount, wi, nb) > i)
                    {
                        const uint uid = contactSelfUniqueIds[wi + nb * i];  // column-major
                        outContactUniqueIds[contactOffset + excl] = uid;
                        ++excl;
                    }
                }
                PxgConstraintBatchHeaderGpu h;
                h.mDescStride_constraintType = stride | (3u << 16);   // eARTICULATION_CONTACT
                h.mConstraintBatchIndex = contactBlockOffset + i + artiSelfContactBatchOffset;
                h.mStartPartitionIndex = contactOffset - numRigidContacts - numRigidStaticContacts;
                h.mask = mask;
                batchHeaders[blockOffset] = h;
                artiContactBatchIdx[contactBlockOffset + numArtiContactBatches + i
                                    + numArtiStaticContacts] = blockOffset - numRigidBatches;
                contactOffset += stride;
                ++blockOffset;   // difference 3: still ONE cursor
            }

            // ---- joints, the SAME cursor ----
            for (uint i = 0u; i < wMaxJ[w]; ++i)
            {
                uint mask = 0u, stride = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = warpBase + l;
                    if (assbCountAt(selfJointCount, wi, nb) > i)
                    {
                        mask |= (1u << l);
                        ++stride;
                    }
                }
                uint excl = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = warpBase + l;
                    if (assbCountAt(selfJointCount, wi, nb) > i)
                    {
                        const uint uid = jointSelfUniqueIds[wi + nb * i];
                        outJointUniqueIds[jointOffset + excl] = uid;
                        ++excl;
                    }
                }
                PxgConstraintBatchHeaderGpu h;
                h.mDescStride_constraintType = stride | (2u << 16);   // eARTICULATION_CONSTRAINT_1D
                h.mConstraintBatchIndex = jointBlockOffset + i + artiSelfJointBatchOffset;
                h.mStartPartitionIndex = jointOffset - numRigidJoints - numRigidStaticJoints;
                h.mask = mask;
                batchHeaders[blockOffset] = h;
                artiJointBatchIdx[jointBlockOffset + numArtiJointBatches + i
                                  + numArtiStaticJoints] = blockOffset - numRigidBatches;
                jointOffset += stride;
                ++blockOffset;
            }

            excCU += wSumC[w]; excJU += wSumJ[w]; excCH += wMaxC[w]; excJH += wMaxJ[w];
        }

        accCU += iterCU; accJU += iterJU; accCH += iterCH; accJH += iterJH;
    }
}
