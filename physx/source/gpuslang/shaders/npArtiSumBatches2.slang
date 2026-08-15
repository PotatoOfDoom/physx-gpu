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


// npArtiSumBatches2.slang
//
// P8.96b: `artiSumInternalContactAndJointBatches2Launch`
// (internalConstraints2.cu:3390, body :3031) -- the scatter half of the prep
// pair: batches1's per-block sums become batch headers, interleaved unique-id
// arrays, and per-link static contact/joint bookkeeping.
//
// SERIAL RECOMPUTE, the P8.96a strategy: every warp op in the CUDA (two scan
// layers, ballots, warpScanExclusive) is over values deterministically
// computable from the inputs, so ONE thread per block recomputes the whole
// block -- the 32-entry block histogram, the per-iteration 16-warp scans, and
// the per-warp interleaved scatters -- as plain loops.  Zero new ABI: every
// struct and offset below is the P5 prep/solver-desc layer or P8.68's blocks.
//
// TRAPS (from the reading, plan file P8.96b):
//  1. the link-block base is the WARP's slice (a*16 + w + numIter*b*16), the
//     lane within it is the CUDA lane l;
//  2. the four Accum values carry ACROSS iterations;
//  3. blockOffset is ONE cursor: contacts first, joints CONTINUE;
//  4. mStartPartitionIndex subtracts rigid+rigidStatic contacts vs joints;
//  5. the batch-indices arrays offset by the DYNAMIC arti batch counts and
//     store blockOffset - numRigidBatches;
//  6. BOTH node indices can be articulations -- both branches run;
//  7. a link's start index is written only on the FIRST hit (== 0 before ++);
//  8. the uid matrices are column-major with stride nbArticulations.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangPrepCommon;
import pxgSlangPrepDesc;
import pxgSlangSolverDesc;

static const uint ASB2_BLOCK_DIM = 512u;
static const uint ASB2_WARPS = 16u;
static const uint ASB2_GRID = 32u;

uint asb2CountAt(Ptr<uint> counts, uint workIndex, uint nb)
{
    return workIndex < nb ? counts[workIndex] : 0u;
}

// the per-link bookkeeping both scatter loops share (traps 1, 6, 7)
void asb2Bookkeep(uint64_t linkSlice, uint lane, uint2 node, uint blockOffset, uint cntField,
                  uint startField)
{
    if (nodeIndexIsArticulation(node))
    {
        const uint64_t link = linkSlice
                            + uint64_t(nodeIndexArticulationLinkId(node)) * PXG_AB_LINK_SIZE;
        if (abU32(link, cntField, lane) == 0u)
            abStoreU32(link, startField, lane, blockOffset);
        abStoreU32(link, cntField, lane, abU32(link, cntField, lane) + 1u);
    }
}

struct Asb2Params
{
    uint64_t scDesc;             // @0
    uint64_t solverCoreDesc;     // @8
    uint64_t prePrepDesc;        // @16
    uint64_t constraintPrepDesc; // @24
    uint nbArticulations;        // @32
    uint pad0;                   // @36 -> 40
};

[[vk::push_constant]]
ConstantBuffer<Asb2Params> gAsb2;

[shader("compute")]
[numthreads(512, 1, 1)]
void artiSumInternalContactAndJointBatches2Launch(uint3 groupId : SV_GroupID,
                                                  uint3 groupThreadId : SV_GroupThreadID)
{
    if (groupThreadId.x != 0u)
        return;
    const uint b = groupId.x;

    const uint64_t d = gAsb2.scDesc;
    Ptr<PxgPrePrepDescGpu> pp = Ptr<PxgPrePrepDescGpu>(gAsb2.prePrepDesc);
    Ptr<PxgConstraintPrepareDescGpu> cp =
        Ptr<PxgConstraintPrepareDescGpu>(gAsb2.constraintPrepDesc);
    Ptr<PxgSolverCoreDescGpu> sc = Ptr<PxgSolverCoreDescGpu>(gAsb2.solverCoreDesc);
    const uint nb = gAsb2.nbArticulations;

    // the wrapper's derived offsets (internalConstraints2.cu:3400-3416)
    const uint numRigidContacts = pp[0].numTotalContacts;
    const uint numRigidJoints = pp[0].numTotalConstraints;
    const uint articulationBatchOffset = pp[0].numBatches + pp[0].numArtiBatches;
    const uint contactUniqueIndexOffset =
        numRigidContacts + pp[0].numTotalStaticContacts + pp[0].numTotalArtiContacts;
    const uint jointUniqueIndexOffset =
        numRigidJoints + pp[0].numTotalStaticConstraints + pp[0].numTotalArtiConstraints;
    const uint artiStaticContactBatchOffset = pp[0].artiStaticContactBatchOffset;
    const uint artiStaticJointBatchOffset = pp[0].artiStaticConstraintBatchOffset;
    const uint numRigidBatches = cp[0].numBatches;
    const uint numArtiContactBatches = cp[0].numArtiContactBatches;
    const uint numArtiJointBatches = cp[0].numArti1dConstraintBatches;
    const uint numRigidStaticContacts = pp[0].numTotalStaticContacts;
    const uint numRigidStaticJoints = pp[0].numTotalStaticConstraints;

    Ptr<uint> staticContactCount = Ptr<uint>(pp[0].mArtiStaticContactCounts);
    Ptr<uint> staticJointCount = Ptr<uint>(pp[0].mArtiStaticConstraintCounts);
    Ptr<uint> contactStaticUniqueIds = Ptr<uint>(pp[0].mArtiStaticContactIndices);
    Ptr<uint> jointStaticUniqueIndices = Ptr<uint>(pp[0].mArtiStaticConstraintIndices);
    Ptr<PartitionNodeDataGpu> pNodeData = Ptr<PartitionNodeDataGpu>(pp[0].mPartitionNodeData);
    Ptr<PxgConstraintBatchHeaderGpu> batchHeaders =
        Ptr<PxgConstraintBatchHeaderGpu>(pp[0].mBatchHeaders);
    Ptr<uint> outContactUniqueIds = Ptr<uint>(pp[0].mContactUniqueIndices);
    Ptr<uint> outJointUniqueIndices = Ptr<uint>(pp[0].mConstraintUniqueIndices);
    Ptr<uint> artiContactBatchIdx = Ptr<uint>(cp[0].artiContactConstraintBatchIndices);
    Ptr<uint> artiJointBatchIdx = Ptr<uint>(cp[0].artiJointConstraintBatchIndices);

    Ptr<uint> tmpCU = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONTACT_UNIQUE));
    Ptr<uint> tmpJU = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONSTRAINT_UNIQUE));
    Ptr<uint> tmpCH = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONTACT_HEADER));
    Ptr<uint> tmpJH = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONSTRAINT_HEADER));

    // the 32-entry block histogram: EXCLUSIVE prefix at [b], plus the inclusive
    // totals for block 0's four writes
    uint histCU = 0u, histJU = 0u, histCH = 0u, histJH = 0u;
    uint totCH = 0u, totJH = 0u;
    for (uint i = 0u; i < ASB2_GRID; ++i)
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
        pp[0].numArtiStaticBatches = totCH + totJH;
        cp[0].numArtiStaticContactBatches = totCH;
        cp[0].numArtiStatic1dConstraintBatches = totJH;
        // islandContextPool[0].mStaticArtiBatchCount (@52 in PxgIslandContext,
        // pinned by the solver-desc mirror)
        Ptr<uint> island = Ptr<uint>(sc[0].islandContextPool + 52u);
        island[0] = totCH + totJH;
    }

    const uint totalBlockRequired = (nb + (ASB2_BLOCK_DIM - 1u)) / ASB2_BLOCK_DIM;
    const uint numIter = (totalBlockRequired + 31u) / 32u;

    const uint64_t linkBlocks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS);
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);

    // trap 2: the four Accum values, once per block, carried across iterations
    uint accCU = 0u, accJU = 0u, accCH = 0u, accJH = 0u;

    for (uint a = 0u; a < numIter; ++a)
    {
        // the 16 warp sums/maxes, then their exclusive scans
        uint wSumC[16]; uint wSumJ[16]; uint wMaxC[16]; uint wMaxJ[16];
        for (uint w = 0u; w < ASB2_WARPS; ++w)
        {
            uint sC = 0u, sJ = 0u, mC = 0u, mJ = 0u;
            for (uint l = 0u; l < 32u; ++l)
            {
                const uint wi = a * ASB2_BLOCK_DIM + (w * 32u + l) + numIter * b * ASB2_BLOCK_DIM;
                const uint c = asb2CountAt(staticContactCount, wi, nb);
                const uint j = asb2CountAt(staticJointCount, wi, nb);
                sC += c; sJ += j;
                mC = c > mC ? c : mC;
                mJ = j > mJ ? j : mJ;
            }
            wSumC[w] = sC; wSumJ[w] = sJ; wMaxC[w] = mC; wMaxJ[w] = mJ;
        }

        uint iterCU = 0u, iterJU = 0u, iterCH = 0u, iterJH = 0u;  // this iteration's totals
        for (uint w = 0u; w < ASB2_WARPS; ++w)
        {
            iterCU += wSumC[w]; iterJU += wSumJ[w]; iterCH += wMaxC[w]; iterJH += wMaxJ[w];
        }

        uint excCU = 0u, excJU = 0u, excCH = 0u, excJH = 0u;      // running exclusive scan
        for (uint w = 0u; w < ASB2_WARPS; ++w)
        {
            uint contactOffset = contactUniqueIndexOffset + excCU + histCU + accCU;
            uint jointOffset = jointUniqueIndexOffset + excJU + histJU + accJU;
            uint contactBlockOffset = excCH + histCH + accCH;
            uint jointBlockOffset = excJH + histJH + accJH;
            uint blockOffset = contactBlockOffset + jointBlockOffset + articulationBatchOffset;

            // trap 1: this warp's link-block slice
            const uint64_t linkSlice = linkBlocks
                                     + uint64_t(a * 16u + w + numIter * b * 16u)
                                       * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;

            // ---- contacts ----
            for (uint i = 0u; i < wMaxC[w]; ++i)
            {
                uint mask = 0u, stride = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = a * ASB2_BLOCK_DIM + (w * 32u + l)
                                  + numIter * b * ASB2_BLOCK_DIM;
                    if (asb2CountAt(staticContactCount, wi, nb) > i)
                    {
                        mask |= (1u << l);
                        ++stride;
                    }
                }
                uint excl = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = a * ASB2_BLOCK_DIM + (w * 32u + l)
                                  + numIter * b * ASB2_BLOCK_DIM;
                    if (asb2CountAt(staticContactCount, wi, nb) > i)
                    {
                        const uint uid = contactStaticUniqueIds[wi + nb * i];   // trap 8
                        outContactUniqueIds[contactOffset + excl] = uid;
                        const PartitionNodeDataGpu nd = pNodeData[uid];
                        asb2Bookkeep(linkSlice, l, nd.mNodeIndex0, blockOffset,
                                     PXG_AB_LINK_mNbStaticContacts,
                                     PXG_AB_LINK_mStaticContactStartIndex);
                        asb2Bookkeep(linkSlice, l, nd.mNodeIndex1, blockOffset,
                                     PXG_AB_LINK_mNbStaticContacts,
                                     PXG_AB_LINK_mStaticContactStartIndex);
                        ++excl;
                    }
                }
                PxgConstraintBatchHeaderGpu h;
                h.mDescStride_constraintType = stride | (3u << 16);   // eARTICULATION_CONTACT
                h.mConstraintBatchIndex = contactBlockOffset + i + artiStaticContactBatchOffset;
                h.mStartPartitionIndex = contactOffset - numRigidContacts - numRigidStaticContacts;
                h.mask = mask;
                batchHeaders[blockOffset] = h;
                artiContactBatchIdx[contactBlockOffset + numArtiContactBatches + i] =
                    blockOffset - numRigidBatches;
                contactOffset += stride;
                ++blockOffset;   // trap 3
            }

            // ---- joints, the SAME cursor ----
            for (uint i = 0u; i < wMaxJ[w]; ++i)
            {
                uint mask = 0u, stride = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = a * ASB2_BLOCK_DIM + (w * 32u + l)
                                  + numIter * b * ASB2_BLOCK_DIM;
                    if (asb2CountAt(staticJointCount, wi, nb) > i)
                    {
                        mask |= (1u << l);
                        ++stride;
                    }
                }
                uint excl = 0u;
                for (uint l = 0u; l < 32u; ++l)
                {
                    const uint wi = a * ASB2_BLOCK_DIM + (w * 32u + l)
                                  + numIter * b * ASB2_BLOCK_DIM;
                    if (asb2CountAt(staticJointCount, wi, nb) > i)
                    {
                        const uint uid = jointStaticUniqueIndices[wi + nb * i];
                        outJointUniqueIndices[jointOffset + excl] = uid;
                        const PartitionNodeDataGpu nd = pNodeData[uid];
                        asb2Bookkeep(linkSlice, l, nd.mNodeIndex0, blockOffset,
                                     PXG_AB_LINK_mNbStaticJoints,
                                     PXG_AB_LINK_mStaticJointStartIndex);
                        asb2Bookkeep(linkSlice, l, nd.mNodeIndex1, blockOffset,
                                     PXG_AB_LINK_mNbStaticJoints,
                                     PXG_AB_LINK_mStaticJointStartIndex);
                        ++excl;
                    }
                }
                PxgConstraintBatchHeaderGpu h;
                h.mDescStride_constraintType = stride | (2u << 16);   // eARTICULATION_CONSTRAINT_1D
                h.mConstraintBatchIndex = jointBlockOffset + i + artiStaticJointBatchOffset;
                h.mStartPartitionIndex = jointOffset - numRigidJoints - numRigidStaticJoints;
                h.mask = mask;
                batchHeaders[blockOffset] = h;
                artiJointBatchIdx[jointBlockOffset + numArtiJointBatches + i] =
                    blockOffset - numRigidBatches;
                jointOffset += stride;
                ++blockOffset;
            }

            excCU += wSumC[w]; excJU += wSumJ[w]; excCH += wMaxC[w]; excJH += wMaxJ[w];
        }

        accCU += iterCU; accJU += iterJU; accCH += iterCH; accJH += iterJH;
    }
}
