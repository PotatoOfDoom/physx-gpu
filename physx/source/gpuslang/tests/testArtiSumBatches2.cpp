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


// P8.96b: npArtiSumBatches2.slang -- artiSumInternalContactAndJointBatches2Launch.
//
// Pure integer scatter, so the oracle is trivially bit-exact: the mirror
// replicates the CUDA's block/iteration/warp structure as host loops and every
// output word -- both uid arrays, the batch-header array (poison past the live
// batches included), both batch-index arrays, the four scalar outputs, the
// island context, and the link blocks' bookkeeping fields -- compares ==.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//   * two sizes: 700 (numIter 1) and 17000 (numIter 2) articulations, the
//     second with counts only in workIndex < 1152 so blocks 0 and 1 both own
//     TWO consecutive 512-spans with live data -- the block-slice stride and
//     numIter mutations change the answer (the P8.96a lesson applied from the
//     start rather than learned again);
//   * a PartitionNodeData table with all four pairings (arti+rigid,
//     rigid+arti, arti+arti, rigid+rigid), so both bookkeeping branches and
//     the neither case run;
//   * counts 0..3 per articulation with contact and joint counts DIFFERENT,
//     so the interleaving (contacts first, joints continuing the same
//     blockOffset cursor) is observable;
//   * every derived-offset scalar nonzero and distinct, so a dropped term in
//     any of the six offset sums is visible;
//   * link blocks poisoned except the bookkeeping fields of LIVE (slice,
//     lane) pairs, which the engine zeroes upstream and the fixture zeroes
//     the same way -- and a first-hit start-index that must overwrite poison
//     exactly once.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxgArticulationCoreDesc.h"
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"
#include "PxgPartitionNode.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxU32;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> asb2Loader(const char* file) { return loadShaderBlob(file); }

const uint32_t ASB2_BLOCKS = 32, ASB2_DIM = 512, ASB2_WARPS = 16;
const uint32_t ASB2_MAX_LINKS = 4;

// counts: distinct patterns, contact != joint, zero on some
static uint32_t c2C(uint32_t i) { return (i * 7u + 2u) % 4u; }
static uint32_t c2J(uint32_t i) { return (i * 5u + 1u) % 3u; }

struct Asb2Case
{
    uint32_t nb;         // nbArticulations
    uint32_t liveMax;    // counts forced zero at or above this workIndex
};

static uint32_t liveCount(const Asb2Case& C, uint32_t i, bool joint)
{
    if (i >= C.liveMax) return 0;
    return joint ? c2J(i) : c2C(i);
}

// scalars, every one nonzero and distinct
const uint32_t S_numTotalContacts = 100, S_numTotalConstraints = 37;
const uint32_t S_numBatches = 11, S_numArtiBatches = 5;
const uint32_t S_numTotalStaticContacts = 13, S_numTotalStaticConstraints = 3;
const uint32_t S_numTotalArtiContacts = 7, S_numTotalArtiConstraints = 9;
const uint32_t S_artiStaticContactBatchOffset = 21, S_artiStaticJointBatchOffset = 33;
const uint32_t S_cpNumBatches = 17, S_numArtiContactBatches = 2, S_numArtiJointBatches = 4;

static size_t asb2Run(PxgCudaShim& shim, const Asb2Case& C)
{
    const uint32_t nb = C.nb;
    const uint32_t nbSlices = (C.liveMax + 31) / 32;
    const uint32_t maxCnt = 4;

    // ---- inputs ----
    std::vector<uint32_t> cntC(nb), cntJ(nb);
    for (uint32_t i = 0; i < nb; ++i)
    {
        cntC[i] = liveCount(C, i, false);
        cntJ[i] = liveCount(C, i, true);
    }
    // uid matrices, column-major stride nb; uids index the node table
    const uint32_t NB_UIDS = 64;
    std::vector<uint32_t> uidC(size_t(nb) * maxCnt, 0xABABABABu);
    std::vector<uint32_t> uidJ(size_t(nb) * maxCnt, 0xABABABABu);
    for (uint32_t i = 0; i < nb; ++i)
    {
        for (uint32_t k = 0; k < cntC[i]; ++k) uidC[i + size_t(nb) * k] = (i * 3u + k) % NB_UIDS;
        for (uint32_t k = 0; k < cntJ[i]; ++k) uidJ[i + size_t(nb) * k] = (i * 5u + k + 1u) % NB_UIDS;
    }
    // the node table: all four pairings, link ids < ASB2_MAX_LINKS
    std::vector<physx::PartitionNodeData> nodes(NB_UIDS);
    std::memset(nodes.data(), 0xAB, nodes.size() * sizeof(physx::PartitionNodeData));
    for (uint32_t u = 0; u < NB_UIDS; ++u)
    {
        const uint32_t mode = u % 4u;   // 0: A arti, 1: B arti, 2: both, 3: neither
        const uint32_t linkA = u % ASB2_MAX_LINKS, linkB = (u + 1) % ASB2_MAX_LINKS;
        nodes[u].mNodeIndex0 = (mode == 0 || mode == 2)
                                   ? physx::PxNodeIndex(u + 1, linkA)
                                   : physx::PxNodeIndex(u + 1);
        nodes[u].mNodeIndex1 = (mode == 1 || mode == 2)
                                   ? physx::PxNodeIndex(u + 100, linkB)
                                   : physx::PxNodeIndex(u + 100);
    }

    // batches1's temp blocks, computed with the batches1 reduction (the real
    // producer) on the host
    const uint32_t totalBlockRequired = (nb + ASB2_DIM - 1) / ASB2_DIM;
    const uint32_t numIter = (totalBlockRequired + 31) / 32;
    std::vector<uint32_t> tCU(ASB2_BLOCKS), tJU(ASB2_BLOCKS), tCH(ASB2_BLOCKS), tJH(ASB2_BLOCKS);
    for (uint32_t b = 0; b < ASB2_BLOCKS; ++b)
    {
        uint32_t su = 0, sj = 0, hc = 0, hj = 0;
        for (uint32_t a = 0; a < numIter; ++a)
            for (uint32_t w = 0; w < ASB2_WARPS; ++w)
            {
                uint32_t mc = 0, mj = 0;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    const uint32_t wi = a * ASB2_DIM + w * 32 + l + numIter * b * ASB2_DIM;
                    const uint32_t c = wi < nb ? cntC[wi] : 0, j = wi < nb ? cntJ[wi] : 0;
                    su += c; sj += j;
                    mc = c > mc ? c : mc;
                    mj = j > mj ? j : mj;
                }
                hc += mc; hj += mj;
            }
        tCU[b] = su; tJU[b] = sj; tCH[b] = hc; tJH[b] = hj;
    }
    uint32_t totCH = 0, totJH = 0, totCU = 0, totJU = 0;
    for (uint32_t b = 0; b < ASB2_BLOCKS; ++b)
    { totCH += tCH[b]; totJH += tJH[b]; totCU += tCU[b]; totJU += tJU[b]; }

    // ---- outputs, poisoned, with canaries past the live regions ----
    const uint32_t contactUidBase = S_numTotalContacts + S_numTotalStaticContacts
                                  + S_numTotalArtiContacts;
    const uint32_t jointUidBase = S_numTotalConstraints + S_numTotalStaticConstraints
                                + S_numTotalArtiConstraints;
    const uint32_t batchBase = S_numBatches + S_numArtiBatches;
    std::vector<uint32_t> outCU(contactUidBase + totCU + 8, 0xABABABABu);
    std::vector<uint32_t> outJU(jointUidBase + totJU + 8, 0xABABABABu);
    std::vector<uint32_t> headers((batchBase + totCH + totJH + 4) * 4, 0xABABABABu);
    std::vector<uint32_t> idxC(S_numArtiContactBatches + totCH + 8, 0xABABABABu);
    std::vector<uint32_t> idxJ(S_numArtiJointBatches + totJH + 8, 0xABABABABu);

    // link blocks: poison, bookkeeping fields of live (slice, lane) pairs zeroed
    std::vector<physx::PxgArticulationBlockLinkData> lk(size_t(nbSlices) * ASB2_MAX_LINKS);
    std::memset(lk.data(), 0xAB, lk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    for (uint32_t s = 0; s < nbSlices; ++s)
        for (uint32_t li = 0; li < ASB2_MAX_LINKS; ++li)
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                lk[size_t(s) * ASB2_MAX_LINKS + li].mNbStaticContacts[lane] = 0;
                lk[size_t(s) * ASB2_MAX_LINKS + li].mNbStaticJoints[lane] = 0;
            }

    // ---- the mirror ----
    std::vector<uint32_t> wCU = outCU, wJU = outJU, wHdr = headers, wIC = idxC, wIJ = idxJ;
    std::vector<physx::PxgArticulationBlockLinkData> wLk = lk;
    uint32_t wNumArtiStaticBatches = 0xABABABABu, wCpC = 0xABABABABu, wCpJ = 0xABABABABu,
             wIsland = 0xABABABABu;
    {
        auto book = [&](const physx::PxNodeIndex& n, uint32_t sliceIdx, uint32_t lane,
                        uint32_t blockOffset, bool joint) {
            if (!n.isArticulation()) return;
            physx::PxgArticulationBlockLinkData& L =
                wLk[size_t(sliceIdx) * ASB2_MAX_LINKS + n.articulationLinkId()];
            PxU32* cnt = joint ? L.mNbStaticJoints : L.mNbStaticContacts;
            PxU32* start = joint ? L.mStaticJointStartIndex : L.mStaticContactStartIndex;
            if (cnt[lane] == 0) start[lane] = blockOffset;
            cnt[lane]++;
        };
        for (uint32_t b = 0; b < ASB2_BLOCKS; ++b)
        {
            uint32_t histCU = 0, histJU = 0, histCH = 0, histJH = 0;
            for (uint32_t i = 0; i < b; ++i)
            { histCU += tCU[i]; histJU += tJU[i]; histCH += tCH[i]; histJH += tJH[i]; }
            if (b == 0)
            {
                wNumArtiStaticBatches = totCH + totJH;
                wCpC = totCH;
                wCpJ = totJH;
                wIsland = totCH + totJH;
            }
            uint32_t accCU = 0, accJU = 0, accCH = 0, accJH = 0;
            for (uint32_t a = 0; a < numIter; ++a)
            {
                uint32_t wS[16][4];
                uint32_t itr[4] = { 0, 0, 0, 0 };
                for (uint32_t w = 0; w < ASB2_WARPS; ++w)
                {
                    uint32_t sC = 0, sJ = 0, mC = 0, mJ = 0;
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        const uint32_t wi = a * ASB2_DIM + w * 32 + l + numIter * b * ASB2_DIM;
                        const uint32_t c = wi < nb ? cntC[wi] : 0, j = wi < nb ? cntJ[wi] : 0;
                        sC += c; sJ += j;
                        mC = c > mC ? c : mC;
                        mJ = j > mJ ? j : mJ;
                    }
                    wS[w][0] = sC; wS[w][1] = sJ; wS[w][2] = mC; wS[w][3] = mJ;
                    itr[0] += sC; itr[1] += sJ; itr[2] += mC; itr[3] += mJ;
                }
                uint32_t exc[4] = { 0, 0, 0, 0 };
                for (uint32_t w = 0; w < ASB2_WARPS; ++w)
                {
                    uint32_t contactOffset = contactUidBase + exc[0] + histCU + accCU;
                    uint32_t jointOffset = jointUidBase + exc[1] + histJU + accJU;
                    uint32_t cbo = exc[2] + histCH + accCH;
                    uint32_t jbo = exc[3] + histJH + accJH;
                    uint32_t blockOffset = cbo + jbo + batchBase;
                    const uint32_t sliceIdx = a * 16 + w + numIter * b * 16;
                    for (uint32_t i = 0; i < wS[w][2]; ++i)
                    {
                        uint32_t mask = 0, stride = 0, excl = 0;
                        for (uint32_t l = 0; l < 32; ++l)
                        {
                            const uint32_t wi = a * ASB2_DIM + w * 32 + l
                                              + numIter * b * ASB2_DIM;
                            if ((wi < nb ? cntC[wi] : 0u) > i) { mask |= 1u << l; ++stride; }
                        }
                        for (uint32_t l = 0; l < 32; ++l)
                        {
                            const uint32_t wi = a * ASB2_DIM + w * 32 + l
                                              + numIter * b * ASB2_DIM;
                            if ((wi < nb ? cntC[wi] : 0u) > i)
                            {
                                const uint32_t uid = uidC[wi + size_t(nb) * i];
                                wCU[contactOffset + excl] = uid;
                                book(nodes[uid].mNodeIndex0, sliceIdx, l, blockOffset, false);
                                book(nodes[uid].mNodeIndex1, sliceIdx, l, blockOffset, false);
                                ++excl;
                            }
                        }
                        wHdr[blockOffset * 4 + 0] = stride | (3u << 16);
                        wHdr[blockOffset * 4 + 1] = cbo + i + S_artiStaticContactBatchOffset;
                        wHdr[blockOffset * 4 + 2] = contactOffset - S_numTotalContacts
                                                  - S_numTotalStaticContacts;
                        wHdr[blockOffset * 4 + 3] = mask;
                        wIC[cbo + S_numArtiContactBatches + i] = blockOffset - S_cpNumBatches;
                        contactOffset += stride;
                        ++blockOffset;
                    }
                    for (uint32_t i = 0; i < wS[w][3]; ++i)
                    {
                        uint32_t mask = 0, stride = 0, excl = 0;
                        for (uint32_t l = 0; l < 32; ++l)
                        {
                            const uint32_t wi = a * ASB2_DIM + w * 32 + l
                                              + numIter * b * ASB2_DIM;
                            if ((wi < nb ? cntJ[wi] : 0u) > i) { mask |= 1u << l; ++stride; }
                        }
                        for (uint32_t l = 0; l < 32; ++l)
                        {
                            const uint32_t wi = a * ASB2_DIM + w * 32 + l
                                              + numIter * b * ASB2_DIM;
                            if ((wi < nb ? cntJ[wi] : 0u) > i)
                            {
                                const uint32_t uid = uidJ[wi + size_t(nb) * i];
                                wJU[jointOffset + excl] = uid;
                                book(nodes[uid].mNodeIndex0, sliceIdx, l, blockOffset, true);
                                book(nodes[uid].mNodeIndex1, sliceIdx, l, blockOffset, true);
                                ++excl;
                            }
                        }
                        wHdr[blockOffset * 4 + 0] = stride | (2u << 16);
                        wHdr[blockOffset * 4 + 1] = jbo + i + S_artiStaticJointBatchOffset;
                        wHdr[blockOffset * 4 + 2] = jointOffset - S_numTotalConstraints
                                                  - S_numTotalStaticConstraints;
                        wHdr[blockOffset * 4 + 3] = mask;
                        wIJ[jbo + S_numArtiJointBatches + i] = blockOffset - S_cpNumBatches;
                        jointOffset += stride;
                        ++blockOffset;
                    }
                    exc[0] += wS[w][0]; exc[1] += wS[w][1]; exc[2] += wS[w][2]; exc[3] += wS[w][3];
                }
                accCU += itr[0]; accJU += itr[1]; accCH += itr[2]; accJH += itr[3];
            }
        }
    }

    // ---- device ----
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dCntC = up(cntC.data(), cntC.size() * 4), dCntJ = up(cntJ.data(), cntJ.size() * 4);
    PxgDevicePtr dUidC = up(uidC.data(), uidC.size() * 4), dUidJ = up(uidJ.data(), uidJ.size() * 4);
    PxgDevicePtr dNodes = up(nodes.data(), nodes.size() * sizeof(physx::PartitionNodeData));
    PxgDevicePtr dTCU = up(tCU.data(), 128), dTJU = up(tJU.data(), 128);
    PxgDevicePtr dTCH = up(tCH.data(), 128), dTJH = up(tJH.data(), 128);
    PxgDevicePtr dOutCU = up(outCU.data(), outCU.size() * 4);
    PxgDevicePtr dOutJU = up(outJU.data(), outJU.size() * 4);
    PxgDevicePtr dHdr = up(headers.data(), headers.size() * 4);
    PxgDevicePtr dIC = up(idxC.data(), idxC.size() * 4), dIJ = up(idxJ.data(), idxJ.size() * 4);
    PxgDevicePtr dLk = up(lk.data(), lk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    physx::PxgIslandContext island;
    std::memset(&island, 0xAB, sizeof(island));
    PxgDevicePtr dIsland = up(&island, sizeof(island));
    PxgDevicePtr dDesc = 0, dPre = 0, dCp = 0, dSc = 0;
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    shim.memAlloc(&dPre, sizeof(physx::PxgPrePrepDesc));
    shim.memAlloc(&dCp, sizeof(physx::PxgConstraintPrepareDesc));
    shim.memAlloc(&dSc, sizeof(physx::PxgSolverCoreDesc));
    if (!dCntC || !dLk || !dDesc || !dPre || !dCp || !dSc || !dIsland)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.mArticulationLinkBlocks = reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.mMaxLinksPerArticulation = ASB2_MAX_LINKS;
    desc.mTempContactUniqueIndicesBlock = reinterpret_cast<PxU32*>(dTCU);
    desc.mTempConstraintUniqueIndicesBlock = reinterpret_cast<PxU32*>(dTJU);
    desc.mTempContactHeaderBlock = reinterpret_cast<PxU32*>(dTCH);
    desc.mTempConstraintHeaderBlock = reinterpret_cast<PxU32*>(dTJH);
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    physx::PxgPrePrepDesc pre;
    std::memset(&pre, 0xAB, sizeof(pre));
    pre.numTotalContacts = S_numTotalContacts;
    pre.numTotalConstraints = S_numTotalConstraints;
    pre.numBatches = S_numBatches;
    pre.numArtiBatches = S_numArtiBatches;
    pre.numTotalStaticContacts = S_numTotalStaticContacts;
    pre.numTotalStaticConstraints = S_numTotalStaticConstraints;
    pre.numTotalArtiContacts = S_numTotalArtiContacts;
    pre.numTotalArtiConstraints = S_numTotalArtiConstraints;
    pre.artiStaticContactBatchOffset = S_artiStaticContactBatchOffset;
    pre.artiStaticConstraintBatchOffset = S_artiStaticJointBatchOffset;
    pre.mArtiStaticContactCounts = reinterpret_cast<PxU32*>(dCntC);
    pre.mArtiStaticConstraintCounts = reinterpret_cast<PxU32*>(dCntJ);
    pre.mArtiStaticContactIndices = reinterpret_cast<PxU32*>(dUidC);
    pre.mArtiStaticConstraintIndices = reinterpret_cast<PxU32*>(dUidJ);
    pre.mPartitionNodeData = reinterpret_cast<physx::PartitionNodeData*>(dNodes);
    pre.mBatchHeaders = reinterpret_cast<physx::PxgConstraintBatchHeader*>(dHdr);
    pre.mContactUniqueIndices = reinterpret_cast<PxU32*>(dOutCU);
    pre.mConstraintUniqueIndices = reinterpret_cast<PxU32*>(dOutJU);
    pre.numArtiStaticBatches = 0xABABABABu;
    shim.memcpyHtoD(dPre, &pre, sizeof(pre));

    physx::PxgConstraintPrepareDesc cpd;
    std::memset(&cpd, 0xAB, sizeof(cpd));
    cpd.numBatches = S_cpNumBatches;
    cpd.numArtiContactBatches = S_numArtiContactBatches;
    cpd.numArti1dConstraintBatches = S_numArtiJointBatches;
    cpd.artiContactConstraintBatchIndices = reinterpret_cast<PxU32*>(dIC);
    cpd.artiJointConstraintBatchIndices = reinterpret_cast<PxU32*>(dIJ);
    shim.memcpyHtoD(dCp, &cpd, sizeof(cpd));

    physx::PxgSolverCoreDesc scd;
    std::memset(&scd, 0xAB, sizeof(scd));
    scd.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsland);
    shim.memcpyHtoD(dSc, &scd, sizeof(scd));

    struct Params { uint64_t sc, sol, pre, cp; uint32_t nb, pad; } prm = {
        uint64_t(dDesc), uint64_t(dSc), uint64_t(dPre), uint64_t(dCp), nb, 0 };
    PXG_CHECK_MSG(shim.launchKernel(0, ASB2_BLOCKS, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiSumInternalContactAndJointBatches2Launch");
    shim.streamSynchronize();

    // ---- compare ----
    size_t bad = 0;
    auto cmp = [&](const char* what, PxgDevicePtr dp, const void* wp, size_t bytes) {
        std::vector<uint8_t> g(bytes);
        shim.memcpyDtoH(g.data(), dp, bytes);
        const uint32_t* gw = reinterpret_cast<const uint32_t*>(g.data());
        const uint32_t* ww = reinterpret_cast<const uint32_t*>(wp);
        for (size_t k = 0; k < bytes / 4; ++k)
            if (gw[k] != ww[k])
            {
                if (bad < 10)
                    std::printf("MM asb2 nb=%u %s word %zu  ref %08x  gpu %08x\n", nb, what, k,
                                ww[k], gw[k]);
                ++bad;
            }
    };
    cmp("outContactUids", dOutCU, wCU.data(), wCU.size() * 4);
    cmp("outJointUids", dOutJU, wJU.data(), wJU.size() * 4);
    cmp("headers", dHdr, wHdr.data(), wHdr.size() * 4);
    cmp("idxContact", dIC, wIC.data(), wIC.size() * 4);
    cmp("idxJoint", dIJ, wIJ.data(), wIJ.size() * 4);
    cmp("linkBlocks", dLk, wLk.data(), wLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    // the scalar outputs
    physx::PxgPrePrepDesc gotPre;
    shim.memcpyDtoH(&gotPre, dPre, sizeof(gotPre));
    physx::PxgConstraintPrepareDesc gotCp;
    shim.memcpyDtoH(&gotCp, dCp, sizeof(gotCp));
    physx::PxgIslandContext gotIsl;
    shim.memcpyDtoH(&gotIsl, dIsland, sizeof(gotIsl));
    if (gotPre.numArtiStaticBatches != wNumArtiStaticBatches) { ++bad; std::printf("MM asb2 numArtiStaticBatches\n"); }
    if (gotCp.numArtiStaticContactBatches != wCpC) { ++bad; std::printf("MM asb2 cp.contact\n"); }
    if (gotCp.numArtiStatic1dConstraintBatches != wCpJ) { ++bad; std::printf("MM asb2 cp.joint\n"); }
    if (gotIsl.mStaticArtiBatchCount != wIsland) { ++bad; std::printf("MM asb2 island\n"); }

    for (PxgDevicePtr dp : bufs) if (dp) shim.memFree(dp);
    shim.memFree(dDesc); shim.memFree(dPre); shim.memFree(dCp); shim.memFree(dSc);
    return bad;
}

PXG_TEST(artiSumBatches2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSumBatches2.spv", "artiSumInternalContactAndJointBatches2Launch", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &asb2Loader);

    const Asb2Case cases[2] = { { 700, 700 }, { 17000, 1152 } };
    size_t bad = 0;
    for (const Asb2Case& C : cases)
    {
        const uint32_t iters = (((C.nb + 511) / 512) + 31) / 32;
        std::printf("[ info ] asb2: nb=%u liveMax=%u -> numIter %u\n", C.nb, C.liveMax, iters);
        bad += asb2Run(shim, C);
    }
    PXG_CHECK_MSG(bad == 0, "the batch scatter does not match the engine");
}

}  // namespace
