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


// P8.96c: npArtiSumSelfBatches.slang -- artiSumSelfContactAndJointBatches.
//
// Pure integer scatter again, so the oracle is bit-exact: the mirror
// replicates the CUDA's block/iteration/warp structure as host loops and every
// output word compares ==.
//
// WHAT THE FIXTURE IS SHAPED TO SEE, beyond the batches2 template:
//   * the STATIC input terms: every derived offset in this kernel gains a
//     static addend its twin does not have (numArtiStaticBatches in the batch
//     base, numTotalStaticArti* in the uid bases, numArtiStatic*Batches in the
//     header batch index and the batch-index arrays).  All five static scalars
//     are nonzero, distinct from each other and from every P8.96b scalar, so a
//     port that reuses the twin's three-term sums lands on wrong words;
//   * the two PER-BLOCK scalars (mTotalSelfConstraintCount /
//     mSelfConstraintOffset): written by every warp whose lane-0 workIndex
//     exists -- INCLUDING warps whose self counts are all zero (count 0,
//     offset = the running cursor), which the liveMax < nb tail provides; the
//     whole poisoned PxgArticulationBlockData array is compared byte for byte,
//     so a write through a lane-indexed accessor (into the tail padding) or a
//     missing zero-warp write fails loudly;
//   * the same two sizes as the twin (700 at numIter 1, 17000 at numIter 2 --
//     the P8.96a scale lesson), counts 0..3 with contact != joint so the
//     shared contacts-then-joints cursor is observable;
//   * the static count/offset INPUT scalars must come back unchanged -- this
//     kernel reads its predecessor's outputs and owns only the Self fields.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxgArticulationCoreDesc.h"
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"

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

static std::vector<uint8_t> assbLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t ASSB_BLOCKS = 32, ASSB_DIM = 512, ASSB_WARPS = 16;

// counts: distinct patterns, contact != joint, zero on some -- deliberately
// DIFFERENT from the batches2 fixture's patterns so a shared .spv cannot pass
// by accident
static uint32_t cSC(uint32_t i) { return (i * 11u + 3u) % 4u; }
static uint32_t cSJ(uint32_t i) { return (i * 13u + 2u) % 3u; }

struct AssbCase
{
    uint32_t nb;         // nbArticulations
    uint32_t liveMax;    // counts forced zero at or above this workIndex
};

static uint32_t liveCount(const AssbCase& C, uint32_t i, bool joint)
{
    if (i >= C.liveMax) return 0;
    return joint ? cSJ(i) : cSC(i);
}

// scalars, every one nonzero and distinct (including vs the batches2 set)
const uint32_t S_numTotalContacts = 100, S_numTotalConstraints = 37;
const uint32_t S_numBatches = 11, S_numArtiBatches = 5;
const uint32_t S_numTotalStaticContacts = 13, S_numTotalStaticConstraints = 3;
const uint32_t S_numTotalArtiContacts = 7, S_numTotalArtiConstraints = 9;
const uint32_t S_numTotalStaticArtiContacts = 19, S_numTotalStaticArtiConstraints = 23;
const uint32_t S_preNumArtiStaticBatches = 29;
const uint32_t S_artiStaticContactBatchOffset = 21, S_artiStaticJointBatchOffset = 33;
const uint32_t S_cpNumBatches = 17, S_numArtiContactBatches = 2, S_numArtiJointBatches = 4;
const uint32_t S_cpStaticContactBatches = 6, S_cpStaticJointBatches = 8;

static size_t assbRun(PxgCudaShim& shim, const AssbCase& C)
{
    const uint32_t nb = C.nb;
    const uint32_t nbBlocks = (nb + 31) / 32;
    const uint32_t maxCnt = 4;

    // ---- inputs ----
    std::vector<uint32_t> cntC(nb), cntJ(nb);
    for (uint32_t i = 0; i < nb; ++i)
    {
        cntC[i] = liveCount(C, i, false);
        cntJ[i] = liveCount(C, i, true);
    }
    // uid matrices, column-major stride nb; values are opaque payloads here
    // (no PartitionNodeData table -- this kernel does no per-link bookkeeping)
    std::vector<uint32_t> uidC(size_t(nb) * maxCnt, 0xABABABABu);
    std::vector<uint32_t> uidJ(size_t(nb) * maxCnt, 0xABABABABu);
    for (uint32_t i = 0; i < nb; ++i)
    {
        for (uint32_t k = 0; k < cntC[i]; ++k) uidC[i + size_t(nb) * k] = i * 17u + k * 5u + 1u;
        for (uint32_t k = 0; k < cntJ[i]; ++k) uidJ[i + size_t(nb) * k] = i * 23u + k * 7u + 2u;
    }

    // batches1's self temp blocks, computed with the batches1 reduction (the
    // real producer) on the host
    const uint32_t totalBlockRequired = (nb + ASSB_DIM - 1) / ASSB_DIM;
    const uint32_t numIter = (totalBlockRequired + 31) / 32;
    std::vector<uint32_t> tCU(ASSB_BLOCKS), tJU(ASSB_BLOCKS), tCH(ASSB_BLOCKS), tJH(ASSB_BLOCKS);
    for (uint32_t b = 0; b < ASSB_BLOCKS; ++b)
    {
        uint32_t su = 0, sj = 0, hc = 0, hj = 0;
        for (uint32_t a = 0; a < numIter; ++a)
            for (uint32_t w = 0; w < ASSB_WARPS; ++w)
            {
                uint32_t mc = 0, mj = 0;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    const uint32_t wi = a * ASSB_DIM + w * 32 + l + numIter * b * ASSB_DIM;
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
    for (uint32_t b = 0; b < ASSB_BLOCKS; ++b)
    { totCH += tCH[b]; totJH += tJH[b]; totCU += tCU[b]; totJU += tJU[b]; }

    // ---- outputs, poisoned, with canaries past the live regions ----
    // difference 2 vs the twin: the static terms join every base
    const uint32_t contactUidBase = S_numTotalContacts + S_numTotalStaticContacts
                                  + S_numTotalArtiContacts + S_numTotalStaticArtiContacts;
    const uint32_t jointUidBase = S_numTotalConstraints + S_numTotalStaticConstraints
                                + S_numTotalArtiConstraints + S_numTotalStaticArtiConstraints;
    const uint32_t batchBase = S_numBatches + S_numArtiBatches + S_preNumArtiStaticBatches;
    const uint32_t selfContactBatchOffset = S_artiStaticContactBatchOffset
                                          + S_cpStaticContactBatches;
    const uint32_t selfJointBatchOffset = S_artiStaticJointBatchOffset
                                        + S_cpStaticJointBatches;
    std::vector<uint32_t> outCU(contactUidBase + totCU + 8, 0xABABABABu);
    std::vector<uint32_t> outJU(jointUidBase + totJU + 8, 0xABABABABu);
    std::vector<uint32_t> headers((batchBase + totCH + totJH + 4) * 4, 0xABABABABu);
    std::vector<uint32_t> idxC(S_numArtiContactBatches + S_cpStaticContactBatches + totCH + 8,
                               0xABABABABu);
    std::vector<uint32_t> idxJ(S_numArtiJointBatches + S_cpStaticJointBatches + totJH + 8,
                               0xABABABABu);

    // the articulation blocks: whole-struct poison; the kernel owns exactly
    // two per-BLOCK scalar words per live warp
    std::vector<physx::PxgArticulationBlockData> ab(nbBlocks);
    std::memset(ab.data(), 0xAB, ab.size() * sizeof(physx::PxgArticulationBlockData));

    // ---- the mirror ----
    std::vector<uint32_t> wCU = outCU, wJU = outJU, wHdr = headers, wIC = idxC, wIJ = idxJ;
    std::vector<physx::PxgArticulationBlockData> wAb = ab;
    uint32_t wNumSelfBatches = 0xABABABABu, wCpC = 0xABABABABu, wCpJ = 0xABABABABu,
             wIsland = 0xABABABABu;
    {
        for (uint32_t b = 0; b < ASSB_BLOCKS; ++b)
        {
            uint32_t histCU = 0, histJU = 0, histCH = 0, histJH = 0;
            for (uint32_t i = 0; i < b; ++i)
            { histCU += tCU[i]; histJU += tJU[i]; histCH += tCH[i]; histJH += tJH[i]; }
            if (b == 0)
            {
                wNumSelfBatches = totCH + totJH;
                wCpC = totCH;
                wCpJ = totJH;
                wIsland = totCH + totJH;
            }
            uint32_t accCU = 0, accJU = 0, accCH = 0, accJH = 0;
            for (uint32_t a = 0; a < numIter; ++a)
            {
                uint32_t wS[16][4];
                uint32_t itr[4] = { 0, 0, 0, 0 };
                for (uint32_t w = 0; w < ASSB_WARPS; ++w)
                {
                    uint32_t sC = 0, sJ = 0, mC = 0, mJ = 0;
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        const uint32_t wi = a * ASSB_DIM + w * 32 + l + numIter * b * ASSB_DIM;
                        const uint32_t c = wi < nb ? cntC[wi] : 0, j = wi < nb ? cntJ[wi] : 0;
                        sC += c; sJ += j;
                        mC = c > mC ? c : mC;
                        mJ = j > mJ ? j : mJ;
                    }
                    wS[w][0] = sC; wS[w][1] = sJ; wS[w][2] = mC; wS[w][3] = mJ;
                    itr[0] += sC; itr[1] += sJ; itr[2] += mC; itr[3] += mJ;
                }
                uint32_t exc[4] = { 0, 0, 0, 0 };
                for (uint32_t w = 0; w < ASSB_WARPS; ++w)
                {
                    uint32_t contactOffset = contactUidBase + exc[0] + histCU + accCU;
                    uint32_t jointOffset = jointUidBase + exc[1] + histJU + accJU;
                    uint32_t cbo = exc[2] + histCH + accCH;
                    uint32_t jbo = exc[3] + histJH + accJH;
                    uint32_t blockOffset = cbo + jbo + batchBase;
                    const uint32_t warpBase = a * ASSB_DIM + w * 32 + numIter * b * ASSB_DIM;
                    // the per-block scalars: BEFORE the loops, zero maxes included
                    if (warpBase < nb)
                    {
                        wAb[warpBase / 32].mTotalSelfConstraintCount = wS[w][2] + wS[w][3];
                        wAb[warpBase / 32].mSelfConstraintOffset = blockOffset;
                    }
                    for (uint32_t i = 0; i < wS[w][2]; ++i)
                    {
                        uint32_t mask = 0, stride = 0, excl = 0;
                        for (uint32_t l = 0; l < 32; ++l)
                            if ((warpBase + l < nb ? cntC[warpBase + l] : 0u) > i)
                            { mask |= 1u << l; ++stride; }
                        for (uint32_t l = 0; l < 32; ++l)
                            if ((warpBase + l < nb ? cntC[warpBase + l] : 0u) > i)
                            {
                                wCU[contactOffset + excl] = uidC[warpBase + l + size_t(nb) * i];
                                ++excl;
                            }
                        wHdr[blockOffset * 4 + 0] = stride | (3u << 16);
                        wHdr[blockOffset * 4 + 1] = cbo + i + selfContactBatchOffset;
                        wHdr[blockOffset * 4 + 2] = contactOffset - S_numTotalContacts
                                                  - S_numTotalStaticContacts;
                        wHdr[blockOffset * 4 + 3] = mask;
                        wIC[cbo + S_numArtiContactBatches + i + S_cpStaticContactBatches] =
                            blockOffset - S_cpNumBatches;
                        contactOffset += stride;
                        ++blockOffset;
                    }
                    for (uint32_t i = 0; i < wS[w][3]; ++i)
                    {
                        uint32_t mask = 0, stride = 0, excl = 0;
                        for (uint32_t l = 0; l < 32; ++l)
                            if ((warpBase + l < nb ? cntJ[warpBase + l] : 0u) > i)
                            { mask |= 1u << l; ++stride; }
                        for (uint32_t l = 0; l < 32; ++l)
                            if ((warpBase + l < nb ? cntJ[warpBase + l] : 0u) > i)
                            {
                                wJU[jointOffset + excl] = uidJ[warpBase + l + size_t(nb) * i];
                                ++excl;
                            }
                        wHdr[blockOffset * 4 + 0] = stride | (2u << 16);
                        wHdr[blockOffset * 4 + 1] = jbo + i + selfJointBatchOffset;
                        wHdr[blockOffset * 4 + 2] = jointOffset - S_numTotalConstraints
                                                  - S_numTotalStaticConstraints;
                        wHdr[blockOffset * 4 + 3] = mask;
                        wIJ[jbo + S_numArtiJointBatches + i + S_cpStaticJointBatches] =
                            blockOffset - S_cpNumBatches;
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
    PxgDevicePtr dTCU = up(tCU.data(), 128), dTJU = up(tJU.data(), 128);
    PxgDevicePtr dTCH = up(tCH.data(), 128), dTJH = up(tJH.data(), 128);
    PxgDevicePtr dOutCU = up(outCU.data(), outCU.size() * 4);
    PxgDevicePtr dOutJU = up(outJU.data(), outJU.size() * 4);
    PxgDevicePtr dHdr = up(headers.data(), headers.size() * 4);
    PxgDevicePtr dIC = up(idxC.data(), idxC.size() * 4), dIJ = up(idxJ.data(), idxJ.size() * 4);
    PxgDevicePtr dAb = up(ab.data(), ab.size() * sizeof(physx::PxgArticulationBlockData));
    physx::PxgIslandContext island;
    std::memset(&island, 0xAB, sizeof(island));
    PxgDevicePtr dIsland = up(&island, sizeof(island));
    PxgDevicePtr dDesc = 0, dPre = 0, dCp = 0, dSc = 0;
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    shim.memAlloc(&dPre, sizeof(physx::PxgPrePrepDesc));
    shim.memAlloc(&dCp, sizeof(physx::PxgConstraintPrepareDesc));
    shim.memAlloc(&dSc, sizeof(physx::PxgSolverCoreDesc));
    if (!dCntC || !dAb || !dDesc || !dPre || !dCp || !dSc || !dIsland)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mTempSelfContactUniqueIndicesBlock = reinterpret_cast<PxU32*>(dTCU);
    desc.mTempSelfConstraintUniqueIndicesBlock = reinterpret_cast<PxU32*>(dTJU);
    desc.mTempSelfContactHeaderBlock = reinterpret_cast<PxU32*>(dTCH);
    desc.mTempSelfConstraintHeaderBlock = reinterpret_cast<PxU32*>(dTJH);
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    physx::PxgPrePrepDesc pre;
    std::memset(&pre, 0xAB, sizeof(pre));
    pre.numTotalContacts = S_numTotalContacts;
    pre.numTotalConstraints = S_numTotalConstraints;
    pre.numBatches = S_numBatches;
    pre.numArtiBatches = S_numArtiBatches;
    pre.numArtiStaticBatches = S_preNumArtiStaticBatches;
    pre.numTotalStaticContacts = S_numTotalStaticContacts;
    pre.numTotalStaticConstraints = S_numTotalStaticConstraints;
    pre.numTotalArtiContacts = S_numTotalArtiContacts;
    pre.numTotalArtiConstraints = S_numTotalArtiConstraints;
    pre.numTotalStaticArtiContacts = S_numTotalStaticArtiContacts;
    pre.numTotalStaticArtiConstraints = S_numTotalStaticArtiConstraints;
    pre.artiStaticContactBatchOffset = S_artiStaticContactBatchOffset;
    pre.artiStaticConstraintBatchOffset = S_artiStaticJointBatchOffset;
    pre.mArtiSelfContactCounts = reinterpret_cast<PxU32*>(dCntC);
    pre.mArtiSelfConstraintCounts = reinterpret_cast<PxU32*>(dCntJ);
    pre.mArtiSelfContactIndices = reinterpret_cast<PxU32*>(dUidC);
    pre.mArtiSelfConstraintIndices = reinterpret_cast<PxU32*>(dUidJ);
    pre.mBatchHeaders = reinterpret_cast<physx::PxgConstraintBatchHeader*>(dHdr);
    pre.mContactUniqueIndices = reinterpret_cast<PxU32*>(dOutCU);
    pre.mConstraintUniqueIndices = reinterpret_cast<PxU32*>(dOutJU);
    pre.numArtiSelfBatches = 0xABABABABu;
    shim.memcpyHtoD(dPre, &pre, sizeof(pre));

    physx::PxgConstraintPrepareDesc cpd;
    std::memset(&cpd, 0xAB, sizeof(cpd));
    cpd.numBatches = S_cpNumBatches;
    cpd.numArtiContactBatches = S_numArtiContactBatches;
    cpd.numArti1dConstraintBatches = S_numArtiJointBatches;
    cpd.numArtiStaticContactBatches = S_cpStaticContactBatches;
    cpd.numArtiStatic1dConstraintBatches = S_cpStaticJointBatches;
    cpd.artiContactConstraintBatchIndices = reinterpret_cast<PxU32*>(dIC);
    cpd.artiJointConstraintBatchIndices = reinterpret_cast<PxU32*>(dIJ);
    cpd.numArtiSelfContactBatches = 0xABABABABu;
    cpd.numArtiSelf1dConstraintBatches = 0xABABABABu;
    shim.memcpyHtoD(dCp, &cpd, sizeof(cpd));

    physx::PxgSolverCoreDesc scd;
    std::memset(&scd, 0xAB, sizeof(scd));
    scd.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsland);
    shim.memcpyHtoD(dSc, &scd, sizeof(scd));

    struct Params { uint64_t sc, sol, pre, cp; uint32_t nb, pad; } prm = {
        uint64_t(dDesc), uint64_t(dSc), uint64_t(dPre), uint64_t(dCp), nb, 0 };
    PXG_CHECK_MSG(shim.launchKernel(0, ASSB_BLOCKS, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiSumSelfContactAndJointBatches");
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
                    std::printf("MM assb nb=%u %s word %zu  ref %08x  gpu %08x\n", nb, what, k,
                                ww[k], gw[k]);
                ++bad;
            }
    };
    cmp("outContactUids", dOutCU, wCU.data(), wCU.size() * 4);
    cmp("outJointUids", dOutJU, wJU.data(), wJU.size() * 4);
    cmp("headers", dHdr, wHdr.data(), wHdr.size() * 4);
    cmp("idxContact", dIC, wIC.data(), wIC.size() * 4);
    cmp("idxJoint", dIJ, wIJ.data(), wIJ.size() * 4);
    cmp("artiBlocks", dAb, wAb.data(), wAb.size() * sizeof(physx::PxgArticulationBlockData));
    // the scalar outputs -- and the static INPUTS, which must be untouched
    physx::PxgPrePrepDesc gotPre;
    shim.memcpyDtoH(&gotPre, dPre, sizeof(gotPre));
    physx::PxgConstraintPrepareDesc gotCp;
    shim.memcpyDtoH(&gotCp, dCp, sizeof(gotCp));
    if (gotPre.numArtiSelfBatches != wNumSelfBatches) { ++bad; std::printf("MM assb numArtiSelfBatches\n"); }
    if (gotPre.numArtiStaticBatches != S_preNumArtiStaticBatches) { ++bad; std::printf("MM assb pre.static stomped\n"); }
    if (gotCp.numArtiSelfContactBatches != wCpC) { ++bad; std::printf("MM assb cp.selfContact\n"); }
    if (gotCp.numArtiSelf1dConstraintBatches != wCpJ) { ++bad; std::printf("MM assb cp.selfJoint\n"); }
    if (gotCp.numArtiStaticContactBatches != S_cpStaticContactBatches) { ++bad; std::printf("MM assb cp.staticContact stomped\n"); }
    if (gotCp.numArtiStatic1dConstraintBatches != S_cpStaticJointBatches) { ++bad; std::printf("MM assb cp.staticJoint stomped\n"); }
    // the island: ONE field owned, everything else must come back poisoned
    physx::PxgIslandContext refIsl;
    std::memset(&refIsl, 0xAB, sizeof(refIsl));
    refIsl.mSelfArtiBatchCount = wIsland;
    physx::PxgIslandContext gotIsl;
    shim.memcpyDtoH(&gotIsl, dIsland, sizeof(gotIsl));
    if (std::memcmp(&gotIsl, &refIsl, sizeof(refIsl)) != 0)
    { ++bad; std::printf("MM assb island (mSelfArtiBatchCount or a stray write)\n"); }

    for (PxgDevicePtr dp : bufs) if (dp) shim.memFree(dp);
    shim.memFree(dDesc); shim.memFree(dPre); shim.memFree(dCp); shim.memFree(dSc);
    return bad;
}

PXG_TEST(artiSumSelfBatches)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSumSelfBatches.spv", "artiSumSelfContactAndJointBatches", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &assbLoader);

    const AssbCase cases[2] = { { 700, 700 }, { 17000, 1152 } };
    size_t bad = 0;
    for (const AssbCase& C : cases)
    {
        const uint32_t iters = (((C.nb + 511) / 512) + 31) / 32;
        std::printf("[ info ] assb: nb=%u liveMax=%u -> numIter %u\n", C.nb, C.liveMax, iters);
        bad += assbRun(shim, C);
    }
    PXG_CHECK_MSG(bad == 0, "the self-batch scatter does not match the engine");
}

}  // namespace
