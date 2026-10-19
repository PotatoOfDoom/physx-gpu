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


// P8.111: the `artiComputeDependencies` engine entry
// (forwardDynamic2.cu:3315) -- the per-partition dependency precompute
// feeding artiPropagateImpulses2/artiPropagateVelocity: common node, slab
// average scale, and the per-partition propagate-up path.  Pure per-thread
// bitfield algebra over desc-level pools (no articulation blocks at all),
// mirrored here word for word.
//
// TWO ROUNDS, chosen RACE-FREE around the engine's own aliasing (the
// seventh upstream-bug candidate, see the module header): the zero loop
// indexes mPathToRootsPerPartition at w*wordSize but the per-partition
// write at w + p*B*wordSize, which overlaps adjacent warps at wordSize >= 2.
//   * round A: 40 articulations (TWO warps), maxLinks 7 -> wordSize 1 --
//     the multi-warp addressing, no overlap possible;
//   * round B: 20 articulations (ONE warp), maxLinks 70 -> wordSize 2 --
//     the word loops carry real work, no second warp to race with.
//
// The fixture seeds: link path bitfields with mixed zero/non-zero words
// (the highest-word-first scan must skip zero words), slab dirty masks with
// x-only / z-only / both / none patterns plus junk in y/w (unread here),
// and lanes whose dirty links appear only in LATER partitions (commonNode
// stays 0xFFFFFFFF through the early ones -- the no-reload path).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiCdLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t CD_D_NB_ARTICULATIONS = 44;
const uint32_t CD_D_NB_SLABS = 80;
const uint32_t CD_D_SLAB_DIRTY_MASKS = 96;
const uint32_t CD_D_PATHS_PER_PARTITION = 104;
const uint32_t CD_D_PATH_TO_ROOT_BITFIELDS = 176;
const uint32_t CD_D_IMPULSE_HOLDING_LINK = 112;
const uint32_t CD_D_PARTITION_AVG_SCALE = 120;
const uint32_t CD_D_TEMP_PATH = 152;
const uint32_t CD_D_TEMP_SHARED = 160;
const uint32_t CD_D_TEMP_ROOT = 168;
const uint32_t CD_D_MAX_LINKS = 248;
const uint32_t CD_DESC_SIZE = 344;
const uint32_t CD_NONE = 0xFFFFFFFFu;
const uint32_t CD_STACK = 256;   // PxgArticulationBitFieldStackData: PxU64[32]

static uint64_t cdWord(uint32_t a, uint32_t b)
{
    uint64_t x = 0x9E3779B97F4A7C15uLL * (a * 131u + b * 17u + 7u);
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9uLL;
    x ^= x >> 32;
    return x;
}

struct CdArena
{
    uint32_t nbArti, nbWarps, maxLinks, wordSize, nbSlabs, nbPartitions, batch;
    std::vector<uint64_t> linkBits;       // [warp][link][word][lane]
    std::vector<uint32_t> masks;          // uint4 per (partition, slab, arti)
    std::vector<uint64_t> paths;          // mPathToRootsPerPartition slots
    std::vector<uint64_t> tmpPath, tmpShared, tmpRoot;
    std::vector<uint32_t> holding;        // nbArti * nbPartitions
    std::vector<float> pscale;
};

static void cdSeed(CdArena& A, uint32_t nbArti, uint32_t maxLinks, uint32_t nbSlabs,
                   uint32_t nbPartitions, uint32_t salt)
{
    A.nbArti = nbArti;
    A.nbWarps = (nbArti + 31) / 32;
    A.maxLinks = maxLinks;
    A.wordSize = (maxLinks + 63) / 64;
    A.nbSlabs = nbSlabs;
    A.nbPartitions = nbPartitions;
    A.batch = (nbArti + 31) / 32;

    A.linkBits.assign(size_t(A.nbWarps) * maxLinks * A.wordSize * 32, 0);
    for (uint32_t w = 0; w < A.nbWarps; ++w)
        for (uint32_t l = 0; l < maxLinks; ++l)
            for (uint32_t j = 0; j < A.wordSize; ++j)
                for (uint32_t lane = 0; lane < 32; ++lane)
                {
                    // a mix: some links all-zero, some with a zero HIGH word
                    // (the scan must skip it), the rest dense
                    const uint32_t mode = (l + lane + j + salt) % 5u;
                    uint64_t v = cdWord(w * 997u + l * 31u + lane + salt, j);
                    // ENGINE INVARIANT: a path-to-root only marks links that
                    // exist -- bits >= maxLinks would make commonNode an
                    // impossible link id and the reload read out of bounds
                    const uint32_t bitBase = j * 64u;
                    const uint32_t nbits =
                        maxLinks > bitBase ? (maxLinks - bitBase < 64u ? maxLinks - bitBase : 64u)
                                           : 0u;
                    const uint64_t mask =
                        nbits >= 64u ? ~0uLL : ((nbits == 0u) ? 0uLL : ((1uLL << nbits) - 1uLL));
                    v &= mask;
                    if (mode == 0u) v = 0;
                    if (mode == 1u && j + 1u == A.wordSize && A.wordSize > 1u) v = 0;
                    A.linkBits[((size_t(w) * maxLinks + l) * A.wordSize + j) * 32 + lane] = v;
                }

    A.masks.assign(size_t(nbPartitions) * nbSlabs * nbArti * 4, CD_NONE);
    for (uint32_t p = 0; p < nbPartitions; ++p)
        for (uint32_t s = 0; s < nbSlabs; ++s)
            for (uint32_t a = 0; a < nbArti; ++a)
            {
                const size_t mi = (size_t(p) * nbSlabs * nbArti + size_t(s) * nbArti + a) * 4;
                // lanes a%7==5 see their first dirty link only from partition 2
                // (the no-reload path); a%7==6 never
                const uint32_t firstP = (a % 7u == 5u) ? 2u : 0u;
                if (a % 7u == 6u || p < firstP)
                {
                    A.masks[mi + 1] = 0xDEADBEEFu;   // junk in y/w: never read here
                    A.masks[mi + 3] = 0xBEEFCAFEu;
                    continue;
                }
                const uint32_t mode = (a + s + p + salt) % 4u;
                const uint32_t la = (a * 3u + s * 5u + p * 11u + salt) % A.maxLinks;
                const uint32_t lb = (a * 7u + s * 13u + p * 3u + 1u + salt) % A.maxLinks;
                if (mode == 0u || mode == 2u) A.masks[mi + 0] = la;
                if (mode == 1u || mode == 2u) A.masks[mi + 2] = lb;
                A.masks[mi + 1] = 0xDEADBEEFu;
                A.masks[mi + 3] = 0xBEEFCAFEu;
            }

    // slot pool: big enough for the LAST partition's base + wordSize
    const size_t maxSlot = size_t(A.nbWarps - 1)
                         + size_t(nbPartitions - 1) * A.batch * A.wordSize + A.wordSize;
    A.paths.assign((maxSlot + 2) * 32, 0xABABABABABABABABuLL);
    A.tmpPath.assign(size_t(A.nbWarps) * A.wordSize * 32, 0xABABABABABABABABuLL);
    A.tmpShared.assign(size_t(A.nbWarps) * A.wordSize * 32, 0xABABABABABABABABuLL);
    A.tmpRoot.assign(size_t(A.nbWarps) * A.wordSize * 32, 0xABABABABABABABABuLL);
    A.holding.assign(size_t(nbArti) * nbPartitions, 0xCDCDCDCDu);
    A.pscale.assign(size_t(nbArti) * nbPartitions, -777.0f);
}

static uint32_t cdHighestBit(uint64_t v)
{
    uint32_t r = 0;
    while (v >>= 1) ++r;
    return r;
}

// the mirror: the kernel, word for word, on the host
static void cdMirror(CdArena& W)
{
    for (uint32_t a = 0; a < W.nbArti; ++a)
    {
        const uint32_t w = a / 32u, lane = a % 32u;
        const uint32_t WS = W.wordSize;
        uint64_t* tmpPath = &W.tmpPath[size_t(w) * WS * 32];
        uint64_t* tmpShared = &W.tmpShared[size_t(w) * WS * 32];
        uint64_t* tmpRoot = &W.tmpRoot[size_t(w) * WS * 32];
        const uint64_t* linkBits = &W.linkBits[size_t(w) * W.maxLinks * WS * 32];

        for (uint32_t j = 0; j < WS; ++j)
        {
            tmpPath[j * 32 + lane] = 0;
            tmpRoot[j * 32 + lane] = 0;
            tmpShared[j * 32 + lane] = 0;
            W.paths[(size_t(w) * WS + j) * 32 + lane] = 0;
        }

        uint32_t commonNode = CD_NONE, oldCommonNode = CD_NONE;
        bool sharedPathEmpty = true;

        for (uint32_t p = 0; p < W.nbPartitions; ++p)
        {
            const uint32_t partitionOffset = a + W.nbArti * p;
            float count = 0.f;

            for (uint32_t s = 0; s < W.nbSlabs; ++s)
            {
                const size_t mi =
                    (size_t(p) * W.nbSlabs * W.nbArti + size_t(s) * W.nbArti + a) * 4;
                const uint32_t dx = W.masks[mi + 0], dz = W.masks[mi + 2];
                if (dx != CD_NONE || dz != CD_NONE)
                    count += 1.f;
                for (uint32_t i = 0; i < 2; ++i)
                {
                    const uint32_t dirtyIndex = i ? dz : dx;
                    if (dirtyIndex == CD_NONE)
                        continue;
                    const uint64_t* lp = &linkBits[size_t(dirtyIndex) * WS * 32];
                    uint32_t commonLink = 0;
                    if (sharedPathEmpty)
                    {
                        for (int j = int(WS) - 1; j >= 0; --j)
                        {
                            const uint64_t word = lp[size_t(j) * 32 + lane];
                            if (word) { commonLink = cdHighestBit(word) + uint32_t(j) * 64u; break; }
                        }
                    }
                    else
                    {
                        for (int j = int(WS) - 1; j >= 0; --j)
                        {
                            const uint64_t word =
                                tmpShared[size_t(j) * 32 + lane] & lp[size_t(j) * 32 + lane];
                            if (word) { commonLink = cdHighestBit(word) + uint32_t(j) * 64u; break; }
                        }
                    }
                    commonNode = commonLink < commonNode ? commonLink : commonNode;
                    for (uint32_t j = 0; j < WS; ++j)
                    {
                        const uint64_t word = lp[size_t(j) * 32 + lane];
                        tmpPath[j * 32 + lane] |= word;
                        tmpShared[j * 32 + lane] |= word;
                        sharedPathEmpty = sharedPathEmpty && (word == 0);
                    }
                }
            }

            if (commonNode != oldCommonNode)
            {
                const uint64_t* lp = &linkBits[size_t(commonNode) * WS * 32];
                for (uint32_t j = 0; j < WS; ++j)
                    tmpRoot[j * 32 + lane] = lp[size_t(j) * 32 + lane];
                oldCommonNode = commonNode;
            }

            const uint32_t nextPartition = p + 1;
            if (nextPartition < W.nbPartitions)
            {
                const size_t base = size_t(w) + size_t(nextPartition) * W.batch * WS;
                for (uint32_t j = 0; j < WS; ++j)
                    W.paths[(base + j) * 32 + lane] = tmpPath[j * 32 + lane];
            }

            for (uint32_t j = 0; j < WS; ++j)
            {
                const uint64_t word = tmpRoot[j * 32 + lane];
                tmpShared[j * 32 + lane] = word;
                sharedPathEmpty = sharedPathEmpty && (word == 0);
                tmpPath[j * 32 + lane] = 0;
            }

            W.holding[partitionOffset] = commonNode;
            W.pscale[partitionOffset] = count < 1.f ? 0.f : 1.f / count;
        }
    }
}

// ---------------------------------------------------------------------------

static size_t cdRun(PxgCudaShim& shim, uint32_t nbArti, uint32_t maxLinks, uint32_t nbSlabs,
                    uint32_t nbPartitions, uint32_t salt)
{
    CdArena A;
    cdSeed(A, nbArti, maxLinks, nbSlabs, nbPartitions, salt);

    CdArena W = A;
    cdMirror(W);

    // coverage over the mirror's own outputs
    uint32_t nZeroScale = 0, nNone = 0, nReal = 0;
    for (size_t i = 0; i < W.holding.size(); ++i)
    {
        if (W.pscale[i] == 0.f) ++nZeroScale;
        if (W.holding[i] == CD_NONE) ++nNone; else ++nReal;
    }
    std::printf("[ info ] cd(nb=%u,ml=%u,ws=%u,p=%u): %u zero-scale, %u none, %u real nodes\n",
                nbArti, maxLinks, A.wordSize, nbPartitions, nZeroScale, nNone, nReal);
    PXG_CHECK_MSG(nZeroScale > 3 && nNone > 3 && nReal > 3, "coverage hole in the outputs");

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dBits = up(A.linkBits.data(), A.linkBits.size() * 8);
    PxgDevicePtr dMasks = up(A.masks.data(), A.masks.size() * 4);
    PxgDevicePtr dPaths = up(A.paths.data(), A.paths.size() * 8);
    PxgDevicePtr dTp = up(A.tmpPath.data(), A.tmpPath.size() * 8);
    PxgDevicePtr dTs = up(A.tmpShared.data(), A.tmpShared.size() * 8);
    PxgDevicePtr dTr = up(A.tmpRoot.data(), A.tmpRoot.size() * 8);
    PxgDevicePtr dHold = up(A.holding.data(), A.holding.size() * 4);
    PxgDevicePtr dPs = up(A.pscale.data(), A.pscale.size() * 4);

    std::vector<uint8_t> descImg(CD_DESC_SIZE, 0xAB);
    auto putU32 = [&](uint32_t off, uint32_t v) { std::memcpy(descImg.data() + off, &v, 4); };
    auto putU64 = [&](uint32_t off, uint64_t v) { std::memcpy(descImg.data() + off, &v, 8); };
    putU32(CD_D_NB_ARTICULATIONS, nbArti);
    putU32(CD_D_NB_SLABS, nbSlabs);
    putU32(CD_D_MAX_LINKS, maxLinks);
    putU64(CD_D_SLAB_DIRTY_MASKS, uint64_t(dMasks));
    putU64(CD_D_PATHS_PER_PARTITION, uint64_t(dPaths));
    putU64(CD_D_PATH_TO_ROOT_BITFIELDS, uint64_t(dBits));
    putU64(CD_D_IMPULSE_HOLDING_LINK, uint64_t(dHold));
    putU64(CD_D_PARTITION_AVG_SCALE, uint64_t(dPs));
    putU64(CD_D_TEMP_PATH, uint64_t(dTp));
    putU64(CD_D_TEMP_SHARED, uint64_t(dTs));
    putU64(CD_D_TEMP_ROOT, uint64_t(dTr));
    PxgDevicePtr dDesc = up(descImg.data(), descImg.size());

    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    struct Params
    {
        uint64_t desc;
        uint32_t nbPartitions, pad;
    } prm = { uint64_t(dDesc), nbPartitions, 0u };
    const uint32_t blocks = (nbArti + 63) / 64;
    PXG_CHECK_MSG(shim.launchKernel(0, blocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiComputeDependencies");
    shim.streamSynchronize();

    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const void* host, size_t n, const char* name) {
        std::vector<uint8_t> g(n);
        shim.memcpyDtoH(g.data(), dp, n);
        if (std::memcmp(g.data(), host, n) != 0)
        {
            size_t mm = 0, first = size_t(-1);
            const uint8_t* h = reinterpret_cast<const uint8_t*>(host);
            for (size_t i = 0; i < n; ++i)
                if (g[i] != h[i]) { ++mm; if (first == size_t(-1)) first = i; }
            std::printf("MM cd/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            // DEBUG: first differing 4-byte words with values
            const uint32_t* hw = reinterpret_cast<const uint32_t*>(host);
            const uint32_t* gw = reinterpret_cast<const uint32_t*>(g.data());
            int shown = 0;
            for (size_t i = 0; i * 4 < n && shown < 6; ++i)
                if (hw[i] != gw[i])
                {
                    std::printf("    word %zu ref %08x gpu %08x\n", i, hw[i], gw[i]);
                    ++shown;
                }
            bad += mm;
        }
    };
    cmp(dBits, W.linkBits.data(), W.linkBits.size() * 8, "linkBits(RO)");
    cmp(dMasks, W.masks.data(), W.masks.size() * 4, "masks(RO)");
    cmp(dPaths, W.paths.data(), W.paths.size() * 8, "pathsPerPartition");
    cmp(dTp, W.tmpPath.data(), W.tmpPath.size() * 8, "tmpPath");
    cmp(dTs, W.tmpShared.data(), W.tmpShared.size() * 8, "tmpShared");
    cmp(dTr, W.tmpRoot.data(), W.tmpRoot.size() * 8, "tmpRoot");
    cmp(dHold, W.holding.data(), W.holding.size() * 4, "impulseHoldingLink");
    cmp(dPs, W.pscale.data(), W.pscale.size() * 4, "partitionAverageScale");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

}  // namespace

PXG_TEST(artiComputeDeps)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiPropagate2.spv", "artiComputeDependencies", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiCdLoader);

    size_t bad = 0;
    // round A: two warps, wordSize 1 -- the multi-warp addressing
    bad += cdRun(shim, 40u, 7u, 3u, 4u, 19u);
    // round B: one warp, maxLinks 70 -> wordSize 2 -- the word loops
    bad += cdRun(shim, 20u, 70u, 2u, 3u, 23u);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
