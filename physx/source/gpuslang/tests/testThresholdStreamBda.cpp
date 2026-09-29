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

// P9.91: thresholdStreamBda.slang -- the force-threshold reporting pipeline of
// accumulateThresholdStream.cu on the ENGINE launch ABI.
//
// The fixture drives the fourteen kernels exactly as PxgCudaSolverCore::
// accumulatedForceThresholdStream() does (push blobs {desc}, {desc, rsDesc},
// {desc, sharedDesc}, {desc, rsDesc, startBit}; the engine's grids 32 and 64; the
// thresholdStream -> tmpThresholdStream copy; the two ping-pong PxgRadixSortDescs of
// PxgRadixSortBuffers::constructRadixSortDesc; the nbPasses rule of :1705-1710; the
// memsetD32(writeIndex, 1, ...) before the mask kernels; setThresholdElementsMask only
// when nbPrev > 0), on a PxgSolverCoreDesc byte image whose offsets are the ones
// testSolverDescLayout pins against the real struct.
//
// The oracle is a literal C++ transcription of the CUDA kernels -- the warp and
// block scans in the CUDA's association, sanitizeKeys' padding lanes, the `pos > 1`
// of setThresholdElementsMask -- run on the same buffers, and EVERY buffer the
// pipeline touches is compared bitwise afterwards, including the bytes the CUDA
// leaves alone (poisoned beforehand on both sides): the radix key/rank ping-pong
// buffers and radixBlockCounts, the sorted stream with its accumulatedForce and
// untouched pad, the accumulated forces and their between-block totals, the reused
// write-index/mask buffer, the bool bytes, the per-pair forces, the exceeded and
// force-change elements, and the two counters the kernels write into the desc.
//
// Three tests:
//   thresholdStreamBda_keyStages    the sort-key extraction and the B sort, compared after
//                                   each stage (the padding keys are dead by frame end)
//   thresholdStreamBda_frameLarge   one frame, nbPrev = 0, 40001 elements: every
//                                   block-partitioned kernel iterates (the radix sort
//                                   twice per block, the accumulation five times), every
//                                   grid-stride kernel loops, and the mask scan spans
//                                   many blocks (with no previous set there is no
//                                   persistent region, so no cross-block hand-off).
//   thresholdStreamBda_twoFrames    frame 2 diffs against frame 1's exceeded set:
//                                   persistent, lost, found, a changed shape pair, the
//                                   binary search's walk back over same-node elements,
//                                   and the CUDA's `pos > 1` stopping one short of slot 0
//                                   -- each with a coverage counter.  The three mask
//                                   regions stay inside block 0's chunk: that is the
//                                   domain where the CUDA's in-launch persistent-mask
//                                   hand-off is ordered at all (see the shader).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> tsbLoader(const char* file) { return loadShaderBlob(file); }

// Dy::ThresholdStreamElement
struct TsElem
{
    uint64_t shapeInteraction;
    float normalForce;
    float threshold;
    uint64_t nodeIndexA;   // PxNodeIndex::mInd: low word index(), high word linkId
    uint64_t nodeIndexB;
    float accumulatedForce;
    uint32_t pad;
};
static_assert(sizeof(TsElem) == 40, "Dy::ThresholdStreamElement is 40 bytes");

// PxgRadixSortDesc
struct RsDesc
{
    uint64_t inputKeys, inputRanks, outputKeys, outputRanks, radixBlockCounts;
    uint32_t count, pad;
};
static_assert(sizeof(RsDesc) == 48, "PxgRadixSortDesc is 48 bytes");

// PxgSolverCoreDesc byte offsets (pxgSlangSolverDesc.slang, pinned by testSolverDescLayout)
enum : uint32_t
{
    kCoreDescSize = 320,
    kOffThresholdStream = 80, kOffTmpThresholdStream = 88, kOffExceeded = 96, kOffPrevExceeded = 104,
    kOffForceChange = 112, kOffAccForce = 120, kOffAccForceBB = 128, kOffWriteIndex = 136,
    kOffWriteIndexBB = 144, kOffWriteable = 152, kOffObjPairs = 160,
    kOffSharedThresholdStreamIndex = 280, kOffNbExceeded = 288, kOffNbPrevExceeded = 292,
    kOffNbForceChange = 296,
    kSharedDescSize = 224, kOffDt = 76
};

static const uint32_t GRID = 32, GRID_WIDE = 64, BLOCK = 256;
static const uint32_t POISON = 0xA5A5A5A5u;

uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
uint32_t idxOf(uint64_t node) { return uint32_t(node); }

enum : uint32_t
{
    K_RS_SINGLE, K_RS_BLOCKS, K_INIT_B, K_INIT_A, K_REORG, K_ACC, K_OUT_ACC, K_WRITEOUT,
    K_EXC, K_OUT_EXC, K_SET_MASK, K_MASK, K_OUT_MASK, K_CREATE, K_COUNT
};

// ===========================================================================
// the oracle: every buffer of one frame, and the CUDA kernels on them
// ===========================================================================
struct Frame
{
    // device-visible state (sizes fixed by the caller; everything poisoned first)
    std::vector<TsElem> stream, tmp, exceeded, prev, forceChange;
    std::vector<float> acc, accBB, objPairs;
    std::vector<uint32_t> writeIdx, writeIdxBB;
    std::vector<uint8_t> writeable;
    std::vector<uint32_t> keys0, ranks0, keys1, ranks1, radixCounts;
    uint32_t n = 0, nbPrev = 0, nbExceeded = POISON, nbForceChange = POISON;
    float dt = 0.5f;
    uint32_t maxNodes = 0;       // the host's nbPasses input
    uint32_t memsetWords = 0;    // memsetD32(writeIndex, 1, memsetWords)
    // coverage of setThresholdElementsMask
    uint32_t cvPersistent = 0, cvWalkBack = 0, cvSicStop = 0, cvNoMatch = 0;
};

uint32_t nbPassesFor(uint32_t maxNodes)
{
    uint32_t hb = 0;
    for (uint32_t v = maxNodes; v > 1; v >>= 1) ++hb;   // PxHighestSetBit
    uint32_t passes = (hb + 1 + 3) / 4;
    if (passes & 1) ++passes;
    return passes;
}

// radixSortSingleBlock + radixSortCalculateRanks at 256 threads, grid 32
void oRadixPass(Frame& f, std::vector<uint32_t>& inK, std::vector<uint32_t>& inR,
                std::vector<uint32_t>& outK, std::vector<uint32_t>& outR, uint32_t startBit)
{
    const uint32_t n = f.n;
    const uint32_t rows = (n + 3) / 4;
    const uint32_t tbr = (rows + BLOCK - 1) / BLOCK;
    const uint32_t nipb = (tbr + GRID - 1) / GRID;
    for (uint32_t b = 0; b < GRID; ++b)
    {
        uint32_t hist[16] = {};
        const uint32_t s = std::min(nipb * b * BLOCK, rows), e = std::min(s + nipb * BLOCK, rows);
        for (uint32_t u = s; u < e; ++u)
            for (uint32_t c = 0; c < 4; ++c)
            {
                const uint32_t k = u * 4 + c;
                ++hist[k < n ? ((inK[k] >> startBit) & 0xF) : 0xF];   // sanitizeKeys
            }
        for (uint32_t d = 0; d < 16; ++d)
            f.radixCounts[d * GRID + b] = hist[d];
    }
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b2) {
        return ((inK[a] >> startBit) & 0xF) < ((inK[b2] >> startBit) & 0xF); });
    for (uint32_t i = 0; i < n; ++i)
    {
        outK[i] = inK[order[i]];
        outR[i] = inR[order[i]];
    }
}

void oRadixSort(Frame& f)
{
    const uint32_t passes = nbPassesFor(f.maxNodes);
    for (uint32_t p = 0; p < passes; ++p)
    {
        if (p & 1) oRadixPass(f, f.keys1, f.ranks1, f.keys0, f.ranks0, p * 4);
        else       oRadixPass(f, f.keys0, f.ranks0, f.keys1, f.ranks1, p * 4);
    }
}

// Hillis-Steele inclusive over `len` lanes: x[l] = x[l] + x[l - r]
void hsInclusive(float* x, uint32_t len)
{
    for (uint32_t r = 1; r < len; r <<= 1)
    {
        float y[32];
        for (uint32_t l = 0; l < len; ++l) y[l] = (l >= r) ? x[l] + x[l - r] : x[l];
        std::memcpy(x, y, len * sizeof(float));
    }
}

bool nodePairEq(const TsElem& a, const TsElem& b) { return a.nodeIndexA == b.nodeIndexA && a.nodeIndexB == b.nodeIndexB; }

// one block-partitioned pass: callback(workIndex, blockIndex) in the CUDA's order
template <class Fn>
void forBlockChunks(uint32_t total, Fn fn)
{
    const uint32_t tbr = (total + BLOCK - 1) / BLOCK;
    const uint32_t nipb = (tbr + GRID - 1) / GRID;
    for (uint32_t b = 0; b < GRID; ++b)
        for (uint32_t it = 0; it < nipb; ++it)
            for (uint32_t t = 0; t < BLOCK; ++t)
            {
                const uint32_t wi = t + it * BLOCK + BLOCK * b * nipb;
                if (wi < total) fn(wi, b);
            }
}

// integer block-partitioned exclusive count: per-block running count in chunk order,
// the block total into bb[b]
template <class FlagFn, class OutFn>
void oBlockCount(uint32_t total, std::vector<uint32_t>& bb, FlagFn flag, OutFn out)
{
    std::vector<uint32_t> run(GRID, 0);
    forBlockChunks(total, [&](uint32_t wi, uint32_t b) {
        const uint32_t fl = flag(wi);
        out(wi, run[b], fl);
        run[b] += fl;
    });
    for (uint32_t b = 0; b < GRID; ++b) bb[b] = run[b];
}

void oComputeAccumulate(Frame& f)
{
    const uint32_t n = f.n;
    const uint32_t tbr = (n + BLOCK - 1) / BLOCK;
    const uint32_t nipb = (tbr + GRID - 1) / GRID;
    for (uint32_t b = 0; b < GRID; ++b)
    {
        float sBlockF = 0.f;
        uint32_t sBlockP = 0;
        for (uint32_t it = 0; it < nipb; ++it)
        {
            float val[BLOCK], accumVal[BLOCK], warpF[8];
            uint32_t isNew[BLOCK], accumPairs[BLOCK], warpP[8];
            for (uint32_t t = 0; t < BLOCK; ++t)
            {
                const uint32_t wi = t + it * BLOCK + BLOCK * b * nipb;
                val[t] = 0.f;
                isNew[t] = 0;
                if (wi < n)
                {
                    val[t] = f.stream[wi].normalForce;
                    isNew[t] = (wi + 1 < n) ? (nodePairEq(f.stream[wi], f.stream[wi + 1]) ? 0u : 1u) : 1u;
                }
            }
            for (uint32_t w = 0; w < 8; ++w)
            {
                float x[32];
                for (uint32_t l = 0; l < 32; ++l) x[l] = val[w * 32 + l];
                hsInclusive(x, 32);
                uint32_t c = 0;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    accumVal[w * 32 + l] = x[l] - val[w * 32 + l];
                    accumPairs[w * 32 + l] = c;
                    c += isNew[w * 32 + l];
                }
                warpF[w] = accumVal[w * 32 + 31] + val[w * 32 + 31];
                warpP[w] = accumPairs[w * 32 + 31] + isNew[w * 32 + 31];
            }
            const float prevF = sBlockF;
            const uint32_t prevP = sBlockP;
            float x8[8];
            std::memcpy(x8, warpF, sizeof(x8));
            hsInclusive(x8, 8);
            float acc8[8];
            uint32_t accP8[8], c = 0;
            for (uint32_t w = 0; w < 8; ++w) { acc8[w] = x8[w] - warpF[w]; accP8[w] = c; c += warpP[w]; }
            sBlockF += (acc8[7] + warpF[7]);
            sBlockP += (accP8[7] + warpP[7]);
            for (uint32_t t = 0; t < BLOCK; ++t)
            {
                const uint32_t wi = t + it * BLOCK + BLOCK * b * nipb;
                if (wi < n)
                {
                    f.acc[wi] = val[t] + accumVal[t] + prevF + acc8[t / 32];
                    f.writeIdx[wi] = accumPairs[t] + prevP + accP8[t / 32];
                    f.writeable[wi] = uint8_t(isNew[t]);
                }
            }
        }
        f.accBB[b] = sBlockF;
        f.writeIdxBB[b] = sBlockP;
    }
}

void oOutputAccumulate(Frame& f)
{
    float x[32];
    std::memcpy(x, f.accBB.data(), sizeof(x));
    hsInclusive(x, 32);
    float res[32];
    uint32_t pres[32], c = 0;
    for (uint32_t i = 0; i < 32; ++i) { res[i] = x[i] - f.accBB[i]; pres[i] = c; c += f.writeIdxBB[i]; }
    forBlockChunks(f.n, [&](uint32_t wi, uint32_t b) {
        f.writeIdx[wi] = f.writeIdx[wi] + pres[b];
        f.acc[wi] = f.acc[wi] + res[b];
    });
}

void oWriteout(Frame& f)
{
    for (uint32_t i = 0; i < f.n; ++i)
        if (f.writeable[i]) f.objPairs[f.writeIdx[i]] = f.acc[i];
}

void oComputeExceeded(Frame& f)
{
    std::vector<uint8_t> exc(f.n, 0);
    for (uint32_t i = 0; i < f.n; ++i)
    {
        const uint32_t w = f.writeIdx[i];
        float a = f.objPairs[w];
        if (w > 0) a -= f.objPairs[w - 1];
        f.stream[i].accumulatedForce = a;
        exc[i] = a > (f.stream[i].threshold * f.dt) ? 1 : 0;
    }
    oBlockCount(f.n, f.writeIdxBB, [&](uint32_t wi) { return uint32_t(exc[wi]); },
                [&](uint32_t wi, uint32_t ex, uint32_t fl) { f.writeIdx[wi] = ex; f.writeable[wi] = uint8_t(fl); });
}

void storeFields(TsElem& d, const TsElem& s, float accumulatedForce)
{
    d.shapeInteraction = s.shapeInteraction;
    d.nodeIndexA = s.nodeIndexA;
    d.nodeIndexB = s.nodeIndexB;
    d.normalForce = s.normalForce;
    d.accumulatedForce = accumulatedForce;
    d.threshold = s.threshold;   // pad untouched
}

void oOutputExceeded(Frame& f)
{
    uint32_t pre[32], c = 0;
    for (uint32_t i = 0; i < 32; ++i) { pre[i] = c; c += f.writeIdxBB[i]; }
    forBlockChunks(f.n, [&](uint32_t wi, uint32_t b) {
        const uint32_t w = f.writeIdx[wi] + pre[b];
        const bool ex = f.writeable[wi] != 0;
        if (ex) storeFields(f.exceeded[w], f.stream[wi], f.stream[wi].accumulatedForce);
        if (wi == f.n - 1) f.nbExceeded = ex ? w + 1 : w;
    });
}

bool elemLe(const TsElem& a, const TsElem& b)
{
    return a.nodeIndexA < b.nodeIndexA || (a.nodeIndexA == b.nodeIndexA && a.nodeIndexB <= b.nodeIndexB);
}

void oSetMask(Frame& f)
{
    std::vector<uint32_t>& mask = f.writeIdx;
    for (uint32_t wi = 0; wi < f.nbExceeded; ++wi)
    {
        const TsElem& el = f.exceeded[wi];
        uint32_t left = 0, right = f.nbPrev;
        while (left < right)
        {
            const uint32_t pos = (left + right) / 2;
            if (elemLe(f.prev[pos], el)) left = pos + 1; else right = pos;
        }
        uint32_t pos = left ? left - 1 : 0;
        bool done = false, matched = false;
        while (!done)
        {
            done = true;
            const TsElem& pp = f.prev[pos];
            if (nodePairEq(pp, el))
            {
                if (pp.shapeInteraction == el.shapeInteraction)
                {
                    mask[pos] = 0;
                    mask[f.nbPrev + wi] = 0;
                    matched = true;
                    ++f.cvPersistent;
                }
                else if (pos > 1)
                {
                    --pos;
                    done = false;
                    ++f.cvWalkBack;
                }
                else if (pos == 1 && nodePairEq(f.prev[0], el) && f.prev[0].shapeInteraction == el.shapeInteraction)
                    ++f.cvSicStop;   // slot 0 holds the match; the CUDA's `pos > 1` never looks
            }
        }
        if (!matched) ++f.cvNoMatch;
    }
}

void oComputeMask(Frame& f)
{
    std::vector<uint32_t>& mask = f.writeIdx;
    const uint32_t total = f.nbExceeded + f.nbPrev * 2;
    const uint32_t P = f.nbPrev + f.nbExceeded;
    for (uint32_t k = 0; k < f.nbPrev; ++k)
        mask[P + k] = mask[k] ? 0u : 1u;
    std::vector<uint32_t> raw(mask.begin(), mask.begin() + total);
    oBlockCount(total, f.writeIdxBB, [&](uint32_t wi) { return raw[wi] != 0 ? 1u : 0u; },
                [&](uint32_t wi, uint32_t ex, uint32_t fl) { mask[wi] = ex; f.writeable[wi] = uint8_t(fl); });
}

void oOutputMask(Frame& f)
{
    const uint32_t total = f.nbExceeded + f.nbPrev * 2;
    uint32_t pre[32], c = 0;
    for (uint32_t i = 0; i < 32; ++i) { pre[i] = c; c += f.writeIdxBB[i]; }
    forBlockChunks(total, [&](uint32_t wi, uint32_t b) { f.writeIdx[wi] += pre[b]; });
}

void oCreate(Frame& f)
{
    const uint32_t total = f.nbExceeded + f.nbPrev * 2;
    const uint32_t P = f.nbPrev + f.nbExceeded;
    for (uint32_t wi = 0; wi < total; ++wi)
    {
        const bool has = f.writeable[wi] != 0;
        const uint32_t w = f.writeIdx[wi];
        if (has)
        {
            const bool lost = wi < f.nbPrev;
            const bool found = wi < P && !lost;
            const TsElem& pr = lost ? f.prev[wi] : found ? f.exceeded[wi - f.nbPrev] : f.prev[wi - P];
            storeFields(f.forceChange[w], pr, lost ? 0.f : pr.accumulatedForce);
        }
        if (wi == total - 1) f.nbForceChange = has ? w + 1 : w;
    }
}

void oInitB(Frame& f)
{
    for (uint32_t i = 0; i < f.n; ++i) { f.keys0[i] = idxOf(f.stream[i].nodeIndexB); f.ranks0[i] = i; }
    for (uint32_t i = 0; i < ((4 - (f.n & 3)) & 3); ++i) { f.keys0[f.n + i] = 0xffffffffu; f.ranks0[f.n + i] = f.n + i; }
}

void oInitA(Frame& f)
{
    for (uint32_t i = 0; i < f.n; ++i) f.keys0[i] = idxOf(f.stream[f.ranks0[i]].nodeIndexA);
    for (uint32_t i = 0; i < ((4 - (f.n & 3)) & 3); ++i) f.keys0[f.n + i] = 0xffffffffu;
}

void oFrame(Frame& f)
{
    f.tmp = f.stream;   // memcpyDtoD(tmpThresholdStream, thresholdStream)
    oInitB(f);
    oRadixSort(f);
    oInitA(f);
    oRadixSort(f);
    for (uint32_t i = 0; i < f.n; ++i)
        std::memcpy(&f.stream[i], &f.tmp[f.ranks0[i]], 32);   // words 0..7
    oComputeAccumulate(f);
    oOutputAccumulate(f);
    oWriteout(f);
    oComputeExceeded(f);
    oOutputExceeded(f);
    for (uint32_t i = 0; i < f.memsetWords; ++i) f.writeIdx[i] = 1u;
    if (f.nbPrev > 0) oSetMask(f);
    oComputeMask(f);
    oOutputMask(f);
    oCreate(f);
}

// ===========================================================================
// the device side: the same buffers, the engine's launch sequence
// ===========================================================================
struct DevFrame
{
    PxgDevicePtr stream = 0, tmp = 0, exceeded = 0, prev = 0, forceChange = 0, acc = 0, accBB = 0, objPairs = 0,
                 writeIdx = 0, writeIdxBB = 0, writeable = 0, keys0 = 0, ranks0 = 0, keys1 = 0, ranks1 = 0,
                 radixCounts = 0, coreDesc = 0, sharedDesc = 0, rsDescs = 0;
    uint8_t core[kCoreDescSize];
};

template <class T>
PxgDevicePtr upload(PxgCudaShim& shim, const std::vector<T>& v)
{
    PxgDevicePtr p = 0;
    shim.memAlloc(&p, std::max<size_t>(v.size() * sizeof(T), 16));
    if (p && !v.empty()) shim.memcpyHtoD(p, v.data(), v.size() * sizeof(T));
    return p;
}

template <class T>
std::vector<T> download(PxgCudaShim& shim, PxgDevicePtr p, size_t count)
{
    std::vector<T> v(count);
    if (count) shim.memcpyDtoH(v.data(), p, count * sizeof(T));
    return v;
}

void putU64(uint8_t* b, uint32_t off, uint64_t v) { std::memcpy(b + off, &v, 8); }
void putU32(uint8_t* b, uint32_t off, uint32_t v) { std::memcpy(b + off, &v, 4); }
uint32_t getU32(const uint8_t* b, uint32_t off) { uint32_t v; std::memcpy(&v, b + off, 4); return v; }

bool devUpload(PxgCudaShim& shim, const Frame& f, DevFrame& d, PxgDevicePtr prevOverride)
{
    d.stream = upload(shim, f.stream);
    d.tmp = upload(shim, f.tmp);
    d.exceeded = upload(shim, f.exceeded);
    d.prev = prevOverride ? prevOverride : upload(shim, f.prev);
    d.forceChange = upload(shim, f.forceChange);
    d.acc = upload(shim, f.acc);
    d.accBB = upload(shim, f.accBB);
    d.objPairs = upload(shim, f.objPairs);
    d.writeIdx = upload(shim, f.writeIdx);
    d.writeIdxBB = upload(shim, f.writeIdxBB);
    d.writeable = upload(shim, f.writeable);
    d.keys0 = upload(shim, f.keys0);
    d.ranks0 = upload(shim, f.ranks0);
    d.keys1 = upload(shim, f.keys1);
    d.ranks1 = upload(shim, f.ranks1);
    d.radixCounts = upload(shim, f.radixCounts);
    if (!d.stream) return false;

    std::memset(d.core, 0xEE, sizeof(d.core));
    putU64(d.core, kOffThresholdStream, d.stream);
    putU64(d.core, kOffTmpThresholdStream, d.tmp);
    putU64(d.core, kOffExceeded, d.exceeded);
    putU64(d.core, kOffPrevExceeded, d.prev);
    putU64(d.core, kOffForceChange, d.forceChange);
    putU64(d.core, kOffAccForce, d.acc);
    putU64(d.core, kOffAccForceBB, d.accBB);
    putU64(d.core, kOffWriteIndex, d.writeIdx);
    putU64(d.core, kOffWriteIndexBB, d.writeIdxBB);
    putU64(d.core, kOffWriteable, d.writeable);
    putU64(d.core, kOffObjPairs, d.objPairs);
    putU32(d.core, kOffSharedThresholdStreamIndex, f.n);
    putU32(d.core, kOffNbExceeded, POISON);
    putU32(d.core, kOffNbPrevExceeded, f.nbPrev);
    putU32(d.core, kOffNbForceChange, POISON);
    shim.memAlloc(&d.coreDesc, kCoreDescSize);
    shim.memcpyHtoD(d.coreDesc, d.core, kCoreDescSize);

    uint8_t shared[kSharedDescSize];
    std::memset(shared, 0xEE, sizeof(shared));
    std::memcpy(shared + kOffDt, &f.dt, 4);
    shim.memAlloc(&d.sharedDesc, kSharedDescSize);
    shim.memcpyHtoD(d.sharedDesc, shared, kSharedDescSize);

    // PxgRadixSortBuffers::constructRadixSortDesc; `count` is never set by the host
    RsDesc rs[2];
    std::memset(rs, 0xEE, sizeof(rs));
    rs[0].inputKeys = d.keys0; rs[0].inputRanks = d.ranks0; rs[0].outputKeys = d.keys1; rs[0].outputRanks = d.ranks1;
    rs[1].inputKeys = d.keys1; rs[1].inputRanks = d.ranks1; rs[1].outputKeys = d.keys0; rs[1].outputRanks = d.ranks0;
    rs[0].radixBlockCounts = rs[1].radixBlockCounts = d.radixCounts;
    shim.memAlloc(&d.rsDescs, sizeof(rs));
    shim.memcpyHtoD(d.rsDescs, rs, sizeof(rs));
    return true;
}

bool devFrame(PxgCudaShim& shim, const Frame& f, DevFrame& d)
{
    struct P1 { uint64_t desc; } p1 = { d.coreDesc };
    struct P2 { uint64_t desc, rs; } p2rs0 = { d.coreDesc, d.rsDescs };
    struct P2S { uint64_t desc, shared; } p2s = { d.coreDesc, d.sharedDesc };
    struct P3 { uint64_t desc, rs; uint32_t startBit, pad; };
    bool ok = true;
    auto L = [&](uint32_t id, uint32_t grid, const void* blob, uint32_t size) {
        ok = ok && shim.launchKernel(id, grid, 1, 1, blob, size) == PxgResult::Success;
    };
    auto radix = [&]() {
        const uint32_t passes = nbPassesFor(f.maxNodes);
        for (uint32_t p = 0; p < passes; ++p)
        {
            P3 p3 = { d.coreDesc, d.rsDescs + (p & 1) * sizeof(RsDesc), p * 4, 0 };
            L(K_RS_SINGLE, GRID, &p3, sizeof(p3));
            L(K_RS_BLOCKS, GRID, &p3, sizeof(p3));
        }
    };
    ok = ok && shim.memcpyDtoD(d.tmp, d.stream, f.n * sizeof(TsElem)) == PxgResult::Success;
    L(K_INIT_B, GRID_WIDE, &p2rs0, sizeof(p2rs0));
    radix();
    L(K_INIT_A, GRID_WIDE, &p2rs0, sizeof(p2rs0));
    radix();
    L(K_REORG, GRID_WIDE, &p2rs0, sizeof(p2rs0));
    L(K_ACC, GRID, &p1, sizeof(p1));
    L(K_OUT_ACC, GRID, &p1, sizeof(p1));
    L(K_WRITEOUT, GRID_WIDE, &p1, sizeof(p1));
    L(K_EXC, GRID, &p2s, sizeof(p2s));
    L(K_OUT_EXC, GRID, &p1, sizeof(p1));
    ok = ok && shim.memsetD32(d.writeIdx, 1u, f.memsetWords) == PxgResult::Success;
    if (f.nbPrev > 0) L(K_SET_MASK, GRID_WIDE, &p1, sizeof(p1));
    L(K_MASK, GRID, &p1, sizeof(p1));
    L(K_OUT_MASK, GRID, &p1, sizeof(p1));
    L(K_CREATE, GRID_WIDE, &p1, sizeof(p1));
    shim.streamSynchronize();
    return ok;
}

template <class T>
size_t cmpBits(const char* what, const std::vector<T>& gpu, const std::vector<T>& ref)
{
    size_t bad = 0;
    const size_t bytes = gpu.size() * sizeof(T);
    const uint8_t* g = reinterpret_cast<const uint8_t*>(gpu.data());
    const uint8_t* r = reinterpret_cast<const uint8_t*>(ref.data());
    for (size_t i = 0; i < bytes; i += 4)
    {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, g + i, std::min<size_t>(4, bytes - i));
        std::memcpy(&b, r + i, std::min<size_t>(4, bytes - i));
        if (a != b)
        {
            if (bad < 3) std::printf("  MM %s word %zu: gpu %08x ref %08x\n", what, i / 4, a, b);
            ++bad;
        }
    }
    return bad;
}

size_t devCompare(PxgCudaShim& shim, const Frame& f, const DevFrame& d)
{
    size_t bad = 0;
    bad += cmpBits("stream", download<TsElem>(shim, d.stream, f.stream.size()), f.stream);
    bad += cmpBits("tmpStream", download<TsElem>(shim, d.tmp, f.tmp.size()), f.tmp);
    bad += cmpBits("exceeded", download<TsElem>(shim, d.exceeded, f.exceeded.size()), f.exceeded);
    bad += cmpBits("forceChange", download<TsElem>(shim, d.forceChange, f.forceChange.size()), f.forceChange);
    bad += cmpBits("accForce", download<float>(shim, d.acc, f.acc.size()), f.acc);
    bad += cmpBits("accForceBetweenBlocks", download<float>(shim, d.accBB, f.accBB.size()), f.accBB);
    bad += cmpBits("objectPairs", download<float>(shim, d.objPairs, f.objPairs.size()), f.objPairs);
    bad += cmpBits("writeIndex/mask", download<uint32_t>(shim, d.writeIdx, f.writeIdx.size()), f.writeIdx);
    bad += cmpBits("writeIndexBetweenBlocks", download<uint32_t>(shim, d.writeIdxBB, f.writeIdxBB.size()), f.writeIdxBB);
    bad += cmpBits("writeable", download<uint8_t>(shim, d.writeable, f.writeable.size()), f.writeable);
    bad += cmpBits("keys0", download<uint32_t>(shim, d.keys0, f.keys0.size()), f.keys0);
    bad += cmpBits("ranks0", download<uint32_t>(shim, d.ranks0, f.ranks0.size()), f.ranks0);
    bad += cmpBits("keys1", download<uint32_t>(shim, d.keys1, f.keys1.size()), f.keys1);
    bad += cmpBits("ranks1", download<uint32_t>(shim, d.ranks1, f.ranks1.size()), f.ranks1);
    bad += cmpBits("radixBlockCounts", download<uint32_t>(shim, d.radixCounts, f.radixCounts.size()), f.radixCounts);
    // the desc: only nbExceededThresholdElements and nbForceChangeElements are written
    std::vector<uint8_t> core = download<uint8_t>(shim, d.coreDesc, kCoreDescSize);
    std::vector<uint8_t> refCore(d.core, d.core + kCoreDescSize);
    putU32(refCore.data(), kOffNbExceeded, f.nbExceeded);
    putU32(refCore.data(), kOffNbForceChange, f.nbForceChange);
    bad += cmpBits("solverCoreDesc", core, refCore);
    return bad;
}

void poisonFrame(Frame& f, uint32_t n, uint32_t capExceeded, uint32_t capForceChange, uint32_t memsetWords)
{
    TsElem pe;
    std::memset(&pe, 0xA5, sizeof(pe));
    const uint32_t keyCap = (n + 3) & ~3u;
    f.n = n;
    f.tmp.assign(n, pe);
    f.exceeded.assign(capExceeded, pe);
    f.forceChange.assign(capForceChange, pe);
    const float pf = [] { float x; uint32_t u = POISON; std::memcpy(&x, &u, 4); return x; }();
    f.acc.assign(n, pf);
    f.accBB.assign(GRID, pf);
    f.objPairs.assign(n + 1, pf);
    f.writeIdx.assign(std::max(n, memsetWords) + 8, POISON);
    f.writeIdxBB.assign(GRID, POISON);
    f.writeable.assign(std::max(n, memsetWords) + 8, 0xA5);
    f.keys0.assign(keyCap, POISON);
    f.ranks0.assign(keyCap, POISON);
    f.keys1.assign(keyCap, POISON);
    f.ranks1.assign(keyCap, POISON);
    f.radixCounts.assign(16 * GRID, POISON);
    f.memsetWords = memsetWords;
}

PxgCudaShim* makeShim(PxgSlangComputeDevice* dev)
{
    static const PxgKernelDef manifest[K_COUNT] = {
        { "thresholdStreamBda.spv", "bodyInputAndRanksSingleBlockLaunch", 24u, 0 },
        { "thresholdStreamBda.spv", "bodyInputAndRanksBlocksLaunch", 24u, 0 },
        { "thresholdStreamBda.spv", "initialRanksAndBodyIndexB", 16u, 0 },
        { "thresholdStreamBda.spv", "initialRanksAndBodyIndexA", 16u, 0 },
        { "thresholdStreamBda.spv", "reorganizeThresholdElements", 16u, 0 },
        { "thresholdStreamBda.spv", "computeAccumulateThresholdStream", 8u, 0 },
        { "thresholdStreamBda.spv", "outputAccumulateThresholdStream", 8u, 0 },
        { "thresholdStreamBda.spv", "writeoutAccumulatedForcePerObject", 8u, 0 },
        { "thresholdStreamBda.spv", "computeExceededForceThresholdElementIndice", 16u, 0 },
        { "thresholdStreamBda.spv", "outputExceededForceThresholdElementIndice", 8u, 0 },
        { "thresholdStreamBda.spv", "setThresholdElementsMask", 8u, 0 },
        { "thresholdStreamBda.spv", "computeThresholdElementMaskIndices", 8u, 0 },
        { "thresholdStreamBda.spv", "outputThresholdPairsMaskIndices", 8u, 0 },
        { "thresholdStreamBda.spv", "createForceChangeThresholdElements", 8u, 0 },
    };
    return new PxgCudaShim(dev, manifest, K_COUNT, &tsbLoader);
}

// A shuffled stream of body pairs, 1..maxShapes elements (shape pairs) per body pair,
// node indices below maxNodes; a few pairs against the static body (index
// PX_INVALID_NODE = 0xffffffff, sorted by the radix passes on its low bits only) and a
// few articulation-link nodes (a non-zero high word).  `exceedFrac` of the pairs get a
// tiny threshold, the rest a huge one.
std::vector<TsElem> makeStream(std::mt19937& rng, uint32_t n, uint32_t maxNodes, uint32_t maxShapes, float exceedFrac)
{
    std::vector<TsElem> s;
    std::uniform_real_distribution<float> uf(0.1f, 100.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    uint64_t shape = 0x10000;
    while (s.size() < n)
    {
        uint64_t a = rng() % (maxNodes - 1), b = rng() % (maxNodes - 1);
        if (rng() % 16 == 0) b = 0xffffffffull;                        // static body
        if (rng() % 32 == 0) a |= uint64_t(((rng() % 7) << 1) | 1) << 32; // articulation link
        if (a == b) continue;
        if (b < a) std::swap(a, b);                                      // PxOrder
        const uint32_t shapes = 1 + rng() % maxShapes;
        const float thr = u01(rng) < exceedFrac ? 1.0e-3f : 1.0e9f;
        for (uint32_t k = 0; k < shapes && s.size() < n; ++k)
        {
            TsElem e;
            std::memset(&e, 0, sizeof(e));
            e.shapeInteraction = (shape++) << 4;
            e.normalForce = uf(rng);
            e.threshold = thr;
            e.nodeIndexA = a;
            e.nodeIndexB = b;
            e.accumulatedForce = 1234.5f;   // stale values the pipeline must overwrite
            e.pad = 0x5A5A5A5Au;            // ... or leave alone
            s.push_back(e);
        }
    }
    std::shuffle(s.begin(), s.end(), rng);
    return s;
}

// ---------------------------------------------------------------------------
// The sort-key stages one at a time.  initialRanksAndBodyIndexB's padding keys are read by
// nothing downstream -- the radix passes treat every lane past the key count as 0xffffffff
// whatever memory holds (sanitizeKeys), and initialRanksAndBodyIndexA writes the same
// slots again -- so the end-of-frame compare cannot see them.  The CUDA still writes those
// bytes, so the port must; this test looks right after each stage.
PXG_TEST(thresholdStreamBda_keyStages)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim* shim = makeShim(dev);
    std::mt19937 rng(0x9193);
    const uint32_t N = 1001;    // three padding lanes

    Frame f;
    poisonFrame(f, N, N, N, N + 64);
    f.maxNodes = 900;
    f.stream = makeStream(rng, N, f.maxNodes, 3, 0.5f);
    DevFrame d;
    if (!devUpload(*shim, f, d, 0)) { std::printf("  NOTE: no BDA -> skip\n"); delete shim; return; }

    struct P2 { uint64_t desc, rs; } p2 = { d.coreDesc, d.rsDescs };
    struct P3 { uint64_t desc, rs; uint32_t startBit, pad; };
    PXG_CHECK(shim->launchKernel(K_INIT_B, GRID_WIDE, 1, 1, &p2, sizeof(p2)) == PxgResult::Success);
    shim->streamSynchronize();
    oInitB(f);
    size_t bad = cmpBits("keys0 after initB", download<uint32_t>(*shim, d.keys0, f.keys0.size()), f.keys0);
    bad += cmpBits("ranks0 after initB", download<uint32_t>(*shim, d.ranks0, f.ranks0.size()), f.ranks0);

    const uint32_t passes = nbPassesFor(f.maxNodes);
    for (uint32_t p = 0; p < passes; ++p)
    {
        P3 p3 = { d.coreDesc, d.rsDescs + (p & 1) * sizeof(RsDesc), p * 4, 0 };
        PXG_CHECK(shim->launchKernel(K_RS_SINGLE, GRID, 1, 1, &p3, sizeof(p3)) == PxgResult::Success);
        PXG_CHECK(shim->launchKernel(K_RS_BLOCKS, GRID, 1, 1, &p3, sizeof(p3)) == PxgResult::Success);
    }
    shim->streamSynchronize();
    oRadixSort(f);
    bad += cmpBits("keys0 after B sort", download<uint32_t>(*shim, d.keys0, f.keys0.size()), f.keys0);
    bad += cmpBits("ranks0 after B sort", download<uint32_t>(*shim, d.ranks0, f.ranks0.size()), f.ranks0);
    bad += cmpBits("keys1 after B sort", download<uint32_t>(*shim, d.keys1, f.keys1.size()), f.keys1);
    bad += cmpBits("radixBlockCounts after B sort", download<uint32_t>(*shim, d.radixCounts, f.radixCounts.size()), f.radixCounts);

    PXG_CHECK(shim->launchKernel(K_INIT_A, GRID_WIDE, 1, 1, &p2, sizeof(p2)) == PxgResult::Success);
    shim->streamSynchronize();
    oInitA(f);
    bad += cmpBits("keys0 after initA", download<uint32_t>(*shim, d.keys0, f.keys0.size()), f.keys0);
    bad += cmpBits("ranks0 after initA", download<uint32_t>(*shim, d.ranks0, f.ranks0.size()), f.ranks0);
    PXG_CHECK_MSG(bad == 0, "%zu words differ from the CUDA transcription", bad);
    delete shim;
}

// ---------------------------------------------------------------------------
PXG_TEST(thresholdStreamBda_frameLarge)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim* shim = makeShim(dev);
    std::mt19937 rng(0x9191);
    const uint32_t N = 40001;   // N % 4 == 1: three sanitizeKeys padding lanes

    Frame f;
    poisonFrame(f, N, N, N, N + 64);
    f.maxNodes = 5000;          // PxHighestSetBit(5000) + 1 = 13 bits -> 4 passes
    f.dt = 0.5f;
    f.stream = makeStream(rng, N, f.maxNodes, 4, 0.6f);

    // Pin the `>` of :440 with ties: run the oracle once, then give a few pairs the
    // threshold t = force / dt exactly (dt = 0.5 makes t * dt == force), so the port's
    // compare decides them the CUDA's way or the counts diverge.
    {
        Frame probe = f;
        oFrame(probe);
        uint32_t ties = 0;
        for (uint32_t i = 0; i < N && ties < 40; i += 997)
        {
            const TsElem& e = probe.stream[i];
            const float t = e.accumulatedForce / f.dt;
            if (t * f.dt != e.accumulatedForce) continue;
            for (TsElem& x : f.stream)
                if (x.nodeIndexA == e.nodeIndexA && x.nodeIndexB == e.nodeIndexB) x.threshold = t;
            ++ties;
        }
        PXG_CHECK_MSG(ties >= 10, "only %u tie pairs", ties);
    }

    DevFrame d;
    if (!devUpload(*shim, f, d, 0)) { std::printf("  NOTE: no BDA -> skip\n"); delete shim; return; }
    PXG_CHECK_MSG(devFrame(*shim, f, d), "a launch was refused");
    oFrame(f);

    uint32_t tiesOff = 0;
    for (const TsElem& e : f.stream)
        if (e.threshold != 1.0e-3f && e.threshold != 1.0e9f && e.accumulatedForce == e.threshold * f.dt) ++tiesOff;
    std::printf("[ info ] thresholdStreamBda large: n=%u nbExceeded=%u nbForceChange=%u, %u tie elements "
                "(force == threshold*dt, not exceeded)\n", N, f.nbExceeded, f.nbForceChange, tiesOff);
    PXG_CHECK_MSG(f.nbExceeded > 16384 && f.nbExceeded < N, "nbExceeded %u: the grid-stride create loop must iterate", f.nbExceeded);
    PXG_CHECK_MSG(f.nbForceChange == f.nbExceeded, "with no previous set every exceeded element is FOUND");
    PXG_CHECK_MSG(tiesOff >= 10, "tie elements %u", tiesOff);
    const size_t bad = devCompare(*shim, f, d);
    PXG_CHECK_MSG(bad == 0, "%zu words differ from the CUDA transcription", bad);
    delete shim;
}

// ---------------------------------------------------------------------------
PXG_TEST(thresholdStreamBda_twoFrames)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim* shim = makeShim(dev);
    std::mt19937 rng(0x9192);
    const uint32_t N = 3001;

    // frame 1: a handful of exceeding pairs among many that do not
    Frame f1;
    poisonFrame(f1, N, N, N, N + 64);
    f1.maxNodes = 700;          // 10 bits -> 3 passes, rounded up to 4
    f1.dt = 0.5f;
    f1.stream = makeStream(rng, N, f1.maxNodes, 3, 0.0f);

    // Directed exceeding pairs.  Node values are chosen so that (1, 2) sorts FIRST in
    // the exceeded set and carries two shape pairs, which is what sets up the CUDA's
    // `pos > 1` stop in frame 2; (300, 301) carries three, for the walk back; (660, 661)
    // gets a shapeInteraction in frame 2 that differs from frame 1's in the HIGH word
    // only (a Sc::ShapeInteraction* is a 64-bit pointer), so the match compares both.
    struct Want { uint64_t a, b; uint32_t shapes; };
    const Want wants[] = { { 1, 2, 2 }, { 5, 9, 1 }, { 40, 41, 1 }, { 300, 301, 3 }, { 400, 0xffffffffull, 1 },
                           { 500, 520, 2 }, { 600, 610, 1 }, { 650, 651, 1 }, { 660, 661, 1 } };
    uint64_t shape = 0x7000000;
    std::vector<uint32_t> slots(N);
    std::iota(slots.begin(), slots.end(), 0u);
    std::shuffle(slots.begin(), slots.end(), rng);   // distinct slots, so no directed element is lost
    uint32_t nextSlot = 0;
    for (const Want& w : wants)
        for (uint32_t k = 0; k < w.shapes; ++k)
        {
            TsElem e;
            std::memset(&e, 0, sizeof(e));
            e.shapeInteraction = (shape++) << 4;
            e.normalForce = 10.f + float(k);
            e.threshold = 1.0f;
            e.nodeIndexA = w.a;
            e.nodeIndexB = w.b;
            e.accumulatedForce = 1234.5f;
            e.pad = 0x5A5A5A5Au;
            f1.stream[slots[nextSlot++]] = e;   // overwrite random slots; the order stays shuffled
        }
    // no accidental duplicates of the directed pairs among the filler
    for (TsElem& e : f1.stream)
        if (e.threshold != 1.0f)
            for (const Want& w : wants)
                if (e.nodeIndexA == w.a && e.nodeIndexB == w.b) e.nodeIndexB = w.b + 7;

    DevFrame d1;
    if (!devUpload(*shim, f1, d1, 0)) { std::printf("  NOTE: no BDA -> skip\n"); delete shim; return; }
    PXG_CHECK_MSG(devFrame(*shim, f1, d1), "frame 1: a launch was refused");
    oFrame(f1);
    PXG_CHECK_MSG(devCompare(*shim, f1, d1) == 0, "frame 1 differs from the CUDA transcription");
    PXG_CHECK_MSG(f1.nbExceeded >= 10 && f1.nbExceeded <= 20, "frame 1 nbExceeded %u", f1.nbExceeded);

    // frame 2: prev = frame 1's exceeded set (its device buffer, as the host swaps it)
    Frame f2;
    poisonFrame(f2, N, N, N, N + 64);
    f2.maxNodes = f1.maxNodes;
    f2.dt = f1.dt;
    f2.prev = f1.exceeded;
    f2.prev.resize(f1.nbExceeded);
    f2.nbPrev = f1.nbExceeded;
    f2.stream = f1.stream;
    std::shuffle(f2.stream.begin(), f2.stream.end(), rng);
    uint64_t newShape = 0x9000000;
    for (TsElem& e : f2.stream)
    {
        e.accumulatedForce = 777.f;
        if (e.threshold != 1.0f) continue;
        if (e.nodeIndexA == 5)   e.threshold = 1.0e9f;                    // LOST
        if (e.nodeIndexA == 40)  e.shapeInteraction = (newShape++) << 4;   // shape pair replaced: lost + found
        if (e.nodeIndexA == 600) e.threshold = 1.0e9f;                    // LOST
        if (e.nodeIndexA == 660) e.shapeInteraction += 1ull << 32;        // same low word, another shape pair
    }
    // FOUND: two filler pairs start exceeding
    uint32_t found = 0;
    for (TsElem& e : f2.stream)
        if (e.threshold == 1.0e9f && e.nodeIndexA > 100 && e.nodeIndexA < 200 && found < 2)
        {
            const uint64_t a = e.nodeIndexA, b = e.nodeIndexB;
            for (TsElem& x : f2.stream)
                if (x.nodeIndexA == a && x.nodeIndexB == b) x.threshold = 1.0f;
            ++found;
        }
    PXG_CHECK_MSG(found == 2, "found pairs %u", found);

    DevFrame d2;
    // the previous set is frame 1's exceeded buffer; the oracle's copy is f2.prev
    PXG_CHECK(devUpload(*shim, f2, d2, d1.exceeded));
    PXG_CHECK_MSG(devFrame(*shim, f2, d2), "frame 2: a launch was refused");
    oFrame(f2);
    const uint32_t total = f2.nbExceeded + 2 * f2.nbPrev;
    std::printf("[ info ] thresholdStreamBda two frames: nbPrev=%u nbExceeded=%u total=%u nbForceChange=%u; "
                "persistent %u, walk-backs %u, `pos > 1` stops %u, unmatched %u\n",
                f2.nbPrev, f2.nbExceeded, total, f2.nbForceChange, f2.cvPersistent, f2.cvWalkBack, f2.cvSicStop,
                f2.cvNoMatch);
    PXG_CHECK_MSG(total <= BLOCK, "the three mask regions must stay in block 0's chunk (total %u)", total);
    PXG_CHECK_MSG(f2.cvPersistent >= 3, "persistent matches %u", f2.cvPersistent);
    PXG_CHECK_MSG(f2.cvWalkBack >= 1, "the binary search's walk back never ran");
    PXG_CHECK_MSG(f2.cvSicStop >= 1, "the CUDA's `pos > 1` stop was never reached");
    PXG_CHECK_MSG(f2.cvNoMatch >= 3, "unmatched current elements %u", f2.cvNoMatch);
    // the previous buffer the kernels read is frame 1's exceeded buffer: compare it too
    // (setThresholdElementsMask and createForceChange only read it)
    size_t bad = devCompare(*shim, f2, d2);
    bad += cmpBits("prevExceeded", download<TsElem>(*shim, d1.exceeded, f2.prev.size()), f2.prev);
    PXG_CHECK_MSG(bad == 0, "frame 2: %zu words differ from the CUDA transcription", bad);
    delete shim;
}

} // namespace
