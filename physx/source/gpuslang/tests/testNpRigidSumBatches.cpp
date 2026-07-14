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

// Verification of npRigidSumBatches.slang: the
// rigidSumInternalContactAndJointBatches1/2 pair (constraintBlockPrePrep.cu
// L1472-1604 / L1607-1956) against an exact serial CPU transcription of the
// CUDA control flow (warp reductions, exclusive scans, shared accumulators,
// ballot scatter).  Every quantity is uint add/max, so all collectives are
// associative reductions over fixed domains -- ALL outputs bitwise, with
// sentinel-filled output buffers proving the exact write set.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t NB_BLOCKS = 32, BLOCK_DIM = 512, WARPS = 16;
const uint32_t SENTINEL = 0xAAAAAAAAu;

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
    std::vector<uint32_t> startIndices;    // 2 * nbBodies
    std::vector<uint32_t> outContactIds;
    std::vector<uint32_t> outJointIds;
    std::vector<uint32_t> headers;         // uint4 image flattened
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
    std::vector<uint32_t> tempInit(128, SENTINEL);
    std::vector<uint32_t> cfgVec = { cfg.contactUniqueIndexOffset, cfg.jointUniqueIndexOffset,
                                     cfg.batchOffset, cfg.staticContactBatchOffset,
                                     cfg.staticJointBatchOffset, cfg.numContactBatches,
                                     cfg.numConstraintBatches };
    std::vector<uint32_t> startInit(2 * nbBodies, SENTINEL);
    std::vector<uint32_t> ocInit(cIdCap, SENTINEL), ojInit(jIdCap, SENTINEL);
    std::vector<uint32_t> hInit(headerCap * 4, SENTINEL);
    std::vector<uint32_t> cbInit(cbCap, SENTINEL), jbInit(jbCap, SENTINEL);
    std::vector<uint32_t> totInit(5, SENTINEL);

    PxgBufferHandle bCC = createBufferFrom(dev, cc);
    PxgBufferHandle bJC = createBufferFrom(dev, jc);
    PxgBufferHandle bTemp = createBufferFrom(dev, tempInit);
    PxgBufferHandle bCfg = createBufferFrom(dev, cfgVec);
    PxgBufferHandle bCids = createBufferFrom(dev, cids);
    PxgBufferHandle bJids = createBufferFrom(dev, jids);
    PxgBufferHandle bStart = createBufferFrom(dev, startInit);
    PxgBufferHandle bOC = createBufferFrom(dev, ocInit);
    PxgBufferHandle bOJ = createBufferFrom(dev, ojInit);
    PxgBufferHandle bH = createBufferFrom(dev, hInit);
    PxgBufferHandle bCB = createBufferFrom(dev, cbInit);
    PxgBufferHandle bJB = createBufferFrom(dev, jbInit);
    PxgBufferHandle bTot = createBufferFrom(dev, totInit);

    const std::vector<PxgBufferHandle> bufs = { bCC, bJC, bTemp, bCfg, bCids, bJids, bStart,
                                                bOC, bOJ, bH, bCB, bJB, bTot };
    const std::vector<uint32_t> strides = { 4, 4, 4, 4, 4, 4, 4, 4, 4, 16, 4, 4, 4 };

    struct P { uint32_t nbBodies; } prm = { nbBodies };
    Kernel k1("npRigidSumBatches.spv", "rsBatches1", strides, sizeof(P));
    k1.dispatch(1, bufs, prm);
    dev->flush();
    Kernel k2("npRigidSumBatches.spv", "rsBatches2", strides, sizeof(P));
    k2.dispatch((NB_BLOCKS * BLOCK_DIM + 63) / 64, bufs, prm);
    dev->flush();

    size_t mm = 0;
    auto cmp = [&](const char* name, const std::vector<uint32_t>& r, PxgBufferHandle h) {
        const auto g = downloadBuffer<uint32_t>(dev, h, uint32_t(r.size()));
        size_t bad = 0, first = ~size_t(0);
        for (size_t x = 0; x < r.size(); ++x)
            if (r[x] != g[x]) { if (!bad) first = x; ++bad; }
        if (bad)
        {
            std::printf("MM %s: %zu diffs, first at %zu (R %08x G %08x)\n", name, bad, first,
                        r[first], g[first]);
            size_t shown = 0;
            for (size_t x = 0; x < r.size() && shown < 8; ++x)
                if (r[x] != g[x])
                { std::printf("   [%zu] R %08x G %08x\n", x, r[x], g[x]); ++shown; }
            mm += bad;
        }
    };
    cmp("temp", ref.temp, bTemp);
    cmp("start", ref.startIndices, bStart);
    cmp("outContact", ref.outContactIds, bOC);
    cmp("outJoint", ref.outJointIds, bOJ);
    cmp("headers", ref.headers, bH);
    cmp("cbIdx", ref.contactBatchIdx, bCB);
    cmp("jbIdx", ref.jointBatchIdx, bJB);
    cmp("totals", ref.totals, bTot);
    mmTotal += mm;

    std::printf("[ info ] rs nbBodies %u iters %u totC %llu totJ %llu batches %u batchOff %u cfg01 %u %u\n", nbBodies,
                iters(nbBodies), (unsigned long long)totC, (unsigned long long)totJ,
                ref.totals[0], cfg.batchOffset, cfg.contactUniqueIndexOffset, cfg.jointUniqueIndexOffset);

    for (PxgBufferHandle h : bufs) dev->destroyBuffer(h);
}

PXG_TEST(npRigidSumBatches_scan)
{
    size_t mm = 0;
    runCase(700, 178001, mm);    // 1 iteration, most blocks idle
    runCase(20000, 178002, mm);  // 2 iterations per block, accumulators live
    PXG_CHECK_MSG(mm == 0, "rigidSumBatches: %zu mismatched words", mm);
}

} // namespace
