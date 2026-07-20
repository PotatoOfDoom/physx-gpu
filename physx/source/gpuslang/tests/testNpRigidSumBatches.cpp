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

// Verification of npRigidSumBatches.slang on the ENGINE launch ABI: the
// rigidSumInternalContactAndJointBatches1/2 pair (constraintBlockPrePrep.cu
// L1472-1604 / L1607-1956) driven exactly as
// PxgSolverCore::constraintPrePrepParallel launches them — REAL engine
// PxgPrePrepDesc / PxgSolverCoreDesc / PxgConstraintPrepareDesc structs
// filled with live device addresses, uploaded, and passed by device pointer
// through PxgCudaShim with the engine grid (32 blocks x 512 threads) and the
// engine parameter blobs ({desc*, nbBodies} / {preDesc*, coreDesc*,
// prepDesc*, nbBodies}).  The CPU reference is an exact serial transcription
// of the CUDA control flow (warp reductions, exclusive scans, shared
// accumulators, ballot scatter).  Every quantity is uint add/max, so all
// collectives are associative reductions over fixed domains -- ALL outputs
// bitwise, with sentinel-filled output buffers proving the exact write set,
// including the desc-field outputs (prePrepDesc->numStaticBatches,
// prepDesc->numStatic*Batches, islandContextPool[0].mStaticRigidBatchCount).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

const uint32_t NB_BLOCKS = 32, BLOCK_DIM = 512, WARPS = 16;
const uint32_t SENTINEL = 0xAAAAAAAAu;

static std::vector<uint8_t> rsLoader(const char* file) { return loadShaderBlob(file); }

// Mirrors the bridge's marshalling (natural alignment, tail rounded).
struct EngineBlob
{
    std::vector<uint8_t> bytes;
    size_t maxAlign = 1;
    static size_t alignFor(size_t size)
    {
        if (size >= 16) return 16;
        if (size >= 8)  return 8;
        if (size >= 4)  return 4;
        if (size >= 2)  return 2;
        return 1;
    }
    template <typename T>
    void add(const T& v)
    {
        const size_t a = alignFor(sizeof(T));
        if (a > maxAlign) maxAlign = a;
        bytes.resize((bytes.size() + (a - 1)) & ~(a - 1));
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), pb, pb + sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + (maxAlign - 1)) & ~(maxAlign - 1)); }
};

struct Cfg
{
    uint32_t contactUniqueIndexOffset, jointUniqueIndexOffset, batchOffset,
             staticContactBatchOffset, staticJointBatchOffset, numContactBatches,
             numConstraintBatches;
};

static uint32_t iters(uint32_t nbBodies)
{
    const uint32_t blocks = (nbBodies + BLOCK_DIM - 1) / BLOCK_DIM;
    return (blocks + NB_BLOCKS - 1) / NB_BLOCKS;
}

struct RefOut
{
    std::vector<uint32_t> temp;            // 128
    std::vector<uint32_t> startIndices;    // 2 * nbBodies (contact ++ joint)
    std::vector<uint32_t> outContactIds;
    std::vector<uint32_t> outJointIds;
    std::vector<uint32_t> headers;         // PxgConstraintBatchHeader image flattened (4 words each)
    std::vector<uint32_t> contactBatchIdx;
    std::vector<uint32_t> jointBatchIdx;
    std::vector<uint32_t> totals;          // 5
};

// serial transcription of both CUDA kernels
static void refRun(uint32_t nbBodies, const std::vector<uint32_t>& cc,
                   const std::vector<uint32_t>& jc, const std::vector<uint32_t>& cids,
                   const std::vector<uint32_t>& jids, const Cfg& cfg, RefOut& o)
{
    const uint32_t numIter = iters(nbBodies);
    auto cnt = [&](bool joint, uint32_t wi) -> uint32_t {
        if (wi >= nbBodies) return 0;
        return joint ? jc[wi] : cc[wi];
    };

    // kernel 1: sums, and SUMS OF PER-WARP MAXES for the header arrays (the
    // per-warp Max reductions at cu L1544 feed warp 0's Add reduction at
    // L1574 -- not a block max)
    o.temp.assign(128, SENTINEL);
    for (uint32_t b = 0; b < NB_BLOCKS; ++b)
    {
        uint32_t sumC = 0, sumJ = 0, sumMaxC = 0, sumMaxJ = 0;
        for (uint32_t i = 0; i < numIter; ++i)
            for (uint32_t w = 0; w < WARPS; ++w)
            {
                uint32_t mC = 0, mJ = 0;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    const uint32_t wi = i * BLOCK_DIM + (w * 32 + l) + numIter * b * BLOCK_DIM;
                    const uint32_t c = cnt(false, wi), j = cnt(true, wi);
                    sumC += c; sumJ += j;
                    mC = c > mC ? c : mC;
                    mJ = j > mJ ? j : mJ;
                }
                sumMaxC += mC;
                sumMaxJ += mJ;
            }
        o.temp[b] = sumC; o.temp[32 + b] = sumJ;
        o.temp[64 + b] = sumMaxC; o.temp[96 + b] = sumMaxJ;
    }

    // kernel 2
    uint32_t chTot = 0, jhTot = 0;
    for (uint32_t p = 0; p < 32; ++p) { chTot += o.temp[64 + p]; jhTot += o.temp[96 + p]; }
    o.totals.assign(5, SENTINEL);
    o.totals[0] = chTot + jhTot; o.totals[1] = chTot; o.totals[2] = jhTot;
    o.totals[3] = chTot + jhTot; o.totals[4] = chTot + jhTot;

    for (uint32_t b = 0; b < NB_BLOCKS; ++b)
    {
        uint32_t hist[4];
        for (int a = 0; a < 4; ++a)
        {
            hist[a] = 0;
            for (uint32_t p = 0; p < b; ++p) hist[a] += o.temp[a * 32 + p];
        }
        uint32_t accum[4] = { 0, 0, 0, 0 };  // cSum, jSum, cMax, jMax accumulators

        for (uint32_t i = 0; i < numIter; ++i)
        {
            uint32_t warpSum[4][WARPS];
            for (uint32_t w = 0; w < WARPS; ++w)
            {
                uint32_t sC = 0, sJ = 0, mC = 0, mJ = 0;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    const uint32_t wi = i * BLOCK_DIM + (w * 32 + l) + numIter * b * BLOCK_DIM;
                    const uint32_t c = cnt(false, wi), j = cnt(true, wi);
                    sC += c; sJ += j;
                    mC = c > mC ? c : mC;
                    mJ = j > mJ ? j : mJ;
                }
                warpSum[0][w] = sC; warpSum[1][w] = sJ; warpSum[2][w] = mC; warpSum[3][w] = mJ;
            }

            for (uint32_t w = 0; w < WARPS; ++w)
            {
                uint32_t pre[4] = { 0, 0, 0, 0 };
                for (uint32_t q = 0; q < w; ++q)
                    for (int a = 0; a < 4; ++a) pre[a] += warpSum[a][q];

                uint32_t contactOffset = cfg.contactUniqueIndexOffset + pre[0] + hist[0] + accum[0];
                uint32_t jointOffset = cfg.jointUniqueIndexOffset + pre[1] + hist[1] + accum[1];
                const uint32_t contactBlockOffset = pre[2] + hist[2] + accum[2];
                const uint32_t jointBlockOffset = pre[3] + hist[3] + accum[3];
                uint32_t blockOffset = contactBlockOffset + jointBlockOffset + cfg.batchOffset;
                const uint32_t maxContact = warpSum[2][w], maxJoint = warpSum[3][w];

                for (uint32_t l = 0; l < 32; ++l)
                {
                    const uint32_t wi = i * BLOCK_DIM + (w * 32 + l) + numIter * b * BLOCK_DIM;
                    if (wi < nbBodies)
                    {
                        o.startIndices[wi] = blockOffset;
                        o.startIndices[nbBodies + wi] = blockOffset + maxContact;
                    }
                }

                for (uint32_t r = 0; r < maxContact; ++r)
                {
                    uint32_t mask = 0;
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        const uint32_t wi = i * BLOCK_DIM + (w * 32 + l) + numIter * b * BLOCK_DIM;
                        if (cnt(false, wi) > r) mask |= (1u << l);
                    }
                    const uint32_t stride = uint32_t(__builtin_popcount(mask));
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        const uint32_t wi = i * BLOCK_DIM + (w * 32 + l) + numIter * b * BLOCK_DIM;
                        if (cnt(false, wi) > r)
                        {
                            const uint32_t off = uint32_t(__builtin_popcount(mask & ((1u << l) - 1u)));
                            o.outContactIds[contactOffset + off] = cids[wi + nbBodies * r];
                        }
                    }
                    const uint32_t batchIndex = contactBlockOffset + r + cfg.staticContactBatchOffset;
                    o.headers[blockOffset * 4 + 0] = stride | (1u << 16);  // eCONTACT
                    o.headers[blockOffset * 4 + 1] = batchIndex;
                    o.headers[blockOffset * 4 + 2] = contactOffset;
                    o.headers[blockOffset * 4 + 3] = mask;
                    o.contactBatchIdx[contactBlockOffset + cfg.numContactBatches + r] = blockOffset;
                    contactOffset += stride;
                    ++blockOffset;
                }

                for (uint32_t r = 0; r < maxJoint; ++r)
                {
                    uint32_t mask = 0;
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        const uint32_t wi = i * BLOCK_DIM + (w * 32 + l) + numIter * b * BLOCK_DIM;
                        if (cnt(true, wi) > r) mask |= (1u << l);
                    }
                    const uint32_t stride = uint32_t(__builtin_popcount(mask));
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        const uint32_t wi = i * BLOCK_DIM + (w * 32 + l) + numIter * b * BLOCK_DIM;
                        if (cnt(true, wi) > r)
                        {
                            const uint32_t off = uint32_t(__builtin_popcount(mask & ((1u << l) - 1u)));
                            o.outJointIds[jointOffset + off] = jids[wi + nbBodies * r];
                        }
                    }
                    const uint32_t batchIndex = jointBlockOffset + r + cfg.staticJointBatchOffset;
                    o.headers[blockOffset * 4 + 0] = stride | (0u << 16);  // eCONSTRAINT_1D
                    o.headers[blockOffset * 4 + 1] = batchIndex;
                    o.headers[blockOffset * 4 + 2] = jointOffset;
                    o.headers[blockOffset * 4 + 3] = mask;
                    o.jointBatchIdx[jointBlockOffset + r + cfg.numConstraintBatches] = blockOffset;
                    jointOffset += stride;
                    ++blockOffset;
                }
            }

            for (int a = 0; a < 4; ++a)
                for (uint32_t w = 0; w < WARPS; ++w)
                    accum[a] += warpSum[a][w];
        }
    }
}

static void runCase(uint32_t nbBodies, uint32_t seed, size_t& mmTotal)
{
    using namespace physx;

    std::mt19937 rng(seed);
    const uint32_t MAXC = 5, MAXJ = 4;

    std::vector<uint32_t> cc(nbBodies), jc(nbBodies);
    std::vector<uint32_t> cids(nbBodies * MAXC), jids(nbBodies * MAXJ);
    uint64_t totC = 0, totJ = 0;
    for (uint32_t i = 0; i < nbBodies; ++i)
    {
        // clustered counts incl. long zero stretches (idle warps/blocks)
        cc[i] = (i % 13 < 6) ? rng() % MAXC : 0;
        jc[i] = (i % 17 < 5) ? rng() % MAXJ : 0;
        totC += cc[i]; totJ += jc[i];
    }
    for (auto& v : cids) v = rng();
    for (auto& v : jids) v = rng();

    // The kernel derives the seven config values from desc fields; split each
    // across its contributing engine fields to exercise the summing.
    Cfg cfg;
    cfg.contactUniqueIndexOffset = 64 + rng() % 64;
    cfg.jointUniqueIndexOffset = 32 + rng() % 32;
    cfg.batchOffset = 100 + rng() % 100;
    cfg.staticContactBatchOffset = rng() % 50;
    cfg.staticJointBatchOffset = rng() % 50;
    cfg.numContactBatches = rng() % 40;
    cfg.numConstraintBatches = rng() % 40;

    const uint32_t headerCap = cfg.batchOffset + 16384;
    const uint32_t cIdCap = cfg.contactUniqueIndexOffset + uint32_t(totC) + 64;
    const uint32_t jIdCap = cfg.jointUniqueIndexOffset + uint32_t(totJ) + 64;
    const uint32_t cbCap = cfg.numContactBatches + 8192;
    const uint32_t jbCap = cfg.numConstraintBatches + 8192;

    RefOut ref;
    ref.startIndices.assign(2 * nbBodies, SENTINEL);
    ref.outContactIds.assign(cIdCap, SENTINEL);
    ref.outJointIds.assign(jIdCap, SENTINEL);
    ref.headers.assign(headerCap * 4, SENTINEL);
    ref.contactBatchIdx.assign(cbCap, SENTINEL);
    ref.jointBatchIdx.assign(jbCap, SENTINEL);
    refRun(nbBodies, cc, jc, cids, jids, cfg, ref);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npRigidSumBatches.spv", "rigidSumInternalContactAndJointBatches1", 16u, 0 },
        { "npRigidSumBatches.spv", "rigidSumInternalContactAndJointBatches2", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &rsLoader);

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr pd = 0;
        shim.memAlloc(&pd, bytes);
        if (pd && data)
            shim.memcpyHtoD(pd, data, bytes);
        return pd;
    };

    const std::vector<uint32_t> tempInit(128, SENTINEL);
    const std::vector<uint32_t> startInit(nbBodies, SENTINEL);
    const std::vector<uint32_t> ocInit(cIdCap, SENTINEL), ojInit(jIdCap, SENTINEL);
    const std::vector<uint32_t> hInit(headerCap * 4, SENTINEL);
    const std::vector<uint32_t> cbInit(cbCap, SENTINEL), jbInit(jbCap, SENTINEL);

    const PxgDevicePtr dCC = alloc(cc.data(), cc.size() * 4);
    const PxgDevicePtr dJC = alloc(jc.data(), jc.size() * 4);
    const PxgDevicePtr dCids = alloc(cids.data(), cids.size() * 4);
    const PxgDevicePtr dJids = alloc(jids.data(), jids.size() * 4);
    const PxgDevicePtr dTempCU = alloc(tempInit.data(), 32 * 4);
    const PxgDevicePtr dTempJU = alloc(tempInit.data(), 32 * 4);
    const PxgDevicePtr dTempCH = alloc(tempInit.data(), 32 * 4);
    const PxgDevicePtr dTempJH = alloc(tempInit.data(), 32 * 4);
    const PxgDevicePtr dStartC = alloc(startInit.data(), nbBodies * 4);
    const PxgDevicePtr dStartJ = alloc(startInit.data(), nbBodies * 4);
    const PxgDevicePtr dOC = alloc(ocInit.data(), cIdCap * 4);
    const PxgDevicePtr dOJ = alloc(ojInit.data(), jIdCap * 4);
    const PxgDevicePtr dH = alloc(hInit.data(), headerCap * 16);
    const PxgDevicePtr dCB = alloc(cbInit.data(), cbCap * 4);
    const PxgDevicePtr dJB = alloc(jbInit.data(), jbCap * 4);
    if (!dCC) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the REAL engine descriptors, filled as the engine fills them ----
    PxgPrePrepDesc pre;
    std::memset(&pre, 0xCD, sizeof(pre)); // background pattern: untouched fields must not matter
    pre.numTotalContacts = cfg.contactUniqueIndexOffset;
    pre.numTotalConstraints = cfg.jointUniqueIndexOffset;
    pre.numBatches = cfg.batchOffset / 2;
    pre.numArtiBatches = (cfg.batchOffset - cfg.batchOffset / 2) / 2;
    pre.numArtiSelfBatches = (cfg.batchOffset - cfg.batchOffset / 2) / 3;
    pre.numArtiStaticBatches = cfg.batchOffset - pre.numBatches - pre.numArtiBatches - pre.numArtiSelfBatches;
    pre.artiStaticContactBatchOffset = cfg.staticContactBatchOffset / 2;
    pre.artiStaticConstraintBatchOffset = cfg.staticJointBatchOffset / 2;
    pre.numStaticBatches = SENTINEL; // output field
    pre.mRigidStaticContactCounts = reinterpret_cast<PxU32*>(dCC);
    pre.mRigidStaticConstraintCounts = reinterpret_cast<PxU32*>(dJC);
    pre.mRigidStaticContactIndices = reinterpret_cast<PxU32*>(dCids);
    pre.mRigidStaticConstraintIndices = reinterpret_cast<PxU32*>(dJids);
    pre.mRigidStaticContactStartIndices = reinterpret_cast<PxU32*>(dStartC);
    pre.mRigidStaticConstraintStartIndices = reinterpret_cast<PxU32*>(dStartJ);
    pre.mTempContactUniqueIndices = reinterpret_cast<PxU32*>(dTempCU);
    pre.mTempConstraintUniqueIndices = reinterpret_cast<PxU32*>(dTempJU);
    pre.mTempContactBlockHeader = reinterpret_cast<PxU32*>(dTempCH);
    pre.mTempConstraintBlockHeader = reinterpret_cast<PxU32*>(dTempJH);
    pre.mBatchHeaders = reinterpret_cast<PxgConstraintBatchHeader*>(dH);
    pre.mContactUniqueIndices = reinterpret_cast<PxU32*>(dOC);
    pre.mConstraintUniqueIndices = reinterpret_cast<PxU32*>(dOJ);

    PxgConstraintPrepareDesc prep;
    std::memset(&prep, 0xCD, sizeof(prep));
    prep.jointConstraintBatchIndices = reinterpret_cast<PxU32*>(dJB);
    prep.contactConstraintBatchIndices = reinterpret_cast<PxU32*>(dCB);
    prep.numContactBatches = cfg.numContactBatches;
    prep.num1dConstraintBatches = cfg.numConstraintBatches;
    prep.numArtiStaticContactBatches = (cfg.staticContactBatchOffset - pre.artiStaticContactBatchOffset) / 2;
    prep.numArtiSelfContactBatches = cfg.staticContactBatchOffset - pre.artiStaticContactBatchOffset -
                                     prep.numArtiStaticContactBatches;
    prep.numArtiStatic1dConstraintBatches =
        (cfg.staticJointBatchOffset - pre.artiStaticConstraintBatchOffset) / 2;
    prep.numArtiSelf1dConstraintBatches = cfg.staticJointBatchOffset - pre.artiStaticConstraintBatchOffset -
                                          prep.numArtiStatic1dConstraintBatches;
    prep.numStaticContactBatches = SENTINEL;       // output field
    prep.numStatic1dConstraintBatches = SENTINEL;  // output field
    prep.numStaticBatches = SENTINEL;              // output field

    PxgIslandContext isl[1];
    std::memset(isl, 0xCD, sizeof(isl));
    isl[0].mStaticRigidBatchCount = SENTINEL;      // output field
    const PxgDevicePtr dIsl = alloc(isl, sizeof(isl));

    PxgSolverCoreDesc score;
    std::memset(&score, 0xCD, sizeof(score));
    score.islandContextPool = reinterpret_cast<PxgIslandContext*>(dIsl);

    const PxgDevicePtr dPre = alloc(&pre, sizeof(pre));
    const PxgDevicePtr dPrep = alloc(&prep, sizeof(prep));
    const PxgDevicePtr dScore = alloc(&score, sizeof(score));

    // ---- engine launches: 32 blocks x 512, CUDA-shaped parameter blobs ----
    EngineBlob b1;
    b1.add<uint64_t>(dPre);
    b1.add<uint32_t>(nbBodies);
    b1.finish();
    PXG_CHECK_MSG(b1.bytes.size() == 16, "rs1 blob is 16 bytes (got %zu)", b1.bytes.size());
    PXG_CHECK_MSG(shim.launchKernel(0, NB_BLOCKS, 1, 1, b1.bytes.data(), uint32_t(b1.bytes.size())) ==
                      PxgResult::Success, "launch rigidSum1");

    EngineBlob b2;
    b2.add<uint64_t>(dPre);
    b2.add<uint64_t>(dScore);
    b2.add<uint64_t>(dPrep);
    b2.add<uint32_t>(nbBodies);
    b2.finish();
    PXG_CHECK_MSG(b2.bytes.size() == 32, "rs2 blob is 32 bytes (got %zu)", b2.bytes.size());
    PXG_CHECK_MSG(shim.launchKernel(1, NB_BLOCKS, 1, 1, b2.bytes.data(), uint32_t(b2.bytes.size())) ==
                      PxgResult::Success, "launch rigidSum2");
    shim.streamSynchronize();

    // ---- bitwise compare (sentinels prove the exact write set) ----
    size_t mm = 0;
    auto cmp = [&](const char* name, const uint32_t* r, size_t n, PxgDevicePtr h) {
        std::vector<uint32_t> g(n);
        shim.memcpyDtoH(g.data(), h, n * 4);
        size_t bad = 0, first = ~size_t(0);
        for (size_t x = 0; x < n; ++x)
            if (r[x] != g[x]) { if (!bad) first = x; ++bad; }
        if (bad)
        {
            std::printf("MM %s: %zu diffs, first at %zu (R %08x G %08x)\n", name, bad, first,
                        r[first], g[first]);
            mm += bad;
        }
    };
    cmp("tempCU", ref.temp.data() + 0, 32, dTempCU);
    cmp("tempJU", ref.temp.data() + 32, 32, dTempJU);
    cmp("tempCH", ref.temp.data() + 64, 32, dTempCH);
    cmp("tempJH", ref.temp.data() + 96, 32, dTempJH);
    cmp("startC", ref.startIndices.data(), nbBodies, dStartC);
    cmp("startJ", ref.startIndices.data() + nbBodies, nbBodies, dStartJ);
    cmp("outContact", ref.outContactIds.data(), cIdCap, dOC);
    cmp("outJoint", ref.outJointIds.data(), jIdCap, dOJ);
    cmp("headers", ref.headers.data(), headerCap * 4, dH);
    cmp("cbIdx", ref.contactBatchIdx.data(), cbCap, dCB);
    cmp("jbIdx", ref.jointBatchIdx.data(), jbCap, dJB);

    // ---- desc-field outputs (read the descs back) ----
    PxgPrePrepDesc preOut;
    PxgConstraintPrepareDesc prepOut;
    PxgIslandContext islOut;
    shim.memcpyDtoH(&preOut, dPre, sizeof(preOut));
    shim.memcpyDtoH(&prepOut, dPrep, sizeof(prepOut));
    shim.memcpyDtoH(&islOut, dIsl, sizeof(islOut));
    if (preOut.numStaticBatches != ref.totals[0]) { std::printf("MM pre.numStaticBatches\n"); ++mm; }
    if (prepOut.numStaticContactBatches != ref.totals[1]) { std::printf("MM prep.numStaticContactBatches\n"); ++mm; }
    if (prepOut.numStatic1dConstraintBatches != ref.totals[2]) { std::printf("MM prep.numStatic1dConstraintBatches\n"); ++mm; }
    if (prepOut.numStaticBatches != ref.totals[3]) { std::printf("MM prep.numStaticBatches\n"); ++mm; }
    if (islOut.mStaticRigidBatchCount != ref.totals[4]) { std::printf("MM island.mStaticRigidBatchCount\n"); ++mm; }
    // input desc fields must be untouched (the kernels only WRITE the fields above)
    if (preOut.numBatches != pre.numBatches || preOut.numTotalContacts != pre.numTotalContacts) { std::printf("MM pre input fields clobbered\n"); ++mm; }
    if (prepOut.numContactBatches != prep.numContactBatches) { std::printf("MM prep input fields clobbered\n"); ++mm; }
    mmTotal += mm;

    std::printf("[ info ] rs nbBodies %u iters %u totC %llu totJ %llu batches %u batchOff %u cfg01 %u %u\n", nbBodies,
                iters(nbBodies), (unsigned long long)totC, (unsigned long long)totJ,
                ref.totals[0], cfg.batchOffset, cfg.contactUniqueIndexOffset, cfg.jointUniqueIndexOffset);

    for (PxgDevicePtr h : { dCC, dJC, dCids, dJids, dTempCU, dTempJU, dTempCH, dTempJH, dStartC, dStartJ,
                            dOC, dOJ, dH, dCB, dJB, dIsl, dPre, dPrep, dScore })
        shim.memFree(h);
}

PXG_TEST(npRigidSumBatches_scan)
{
    size_t mm = 0;
    runCase(700, 178001, mm);    // 1 iteration, most blocks idle
    runCase(20000, 178002, mm);  // 2 iterations per block, accumulators live
    PXG_CHECK_MSG(mm == 0, "rigidSumBatches: %zu mismatched words", mm);
}

} // namespace
