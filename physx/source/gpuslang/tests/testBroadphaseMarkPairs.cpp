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

// Verification of broadphaseMarkPairs.slang against C++ transcriptions of the
// four incremental delete/update marking scatter kernels: marking removed
// handles as deleted, invalidating removed projections/SAP-boxes, and
// refreshing updated-box projections (bitmap- and array-driven).  All outputs
// are compared bit-exactly over a consistent synthetic SAP arrangement.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t INVALID = 0xffffffffu;
const uint32_t GRID_X = 32;

struct MarkParams
{
    uint32_t numHandles;
    uint32_t numRemovedHandles;
    uint32_t numUpdatedHandles;
    uint32_t changedWordCounts;
    uint32_t gridDimX;
};

// per-axis SAP arrangement: sapBox[axis][b]={minPos,maxPos} partitioning [0,2N)
void genArrangement(std::mt19937& rng, uint32_t N, std::vector<uint32_t>& sapBox)
{
    const uint32_t nbProj = N * 2;
    sapBox.assign(3 * N * 2, 0);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> perm(nbProj);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t b = 0; b < N; ++b)
        {
            sapBox[(ax * N + b) * 2 + 0] = std::min(perm[2 * b], perm[2 * b + 1]);
            sapBox[(ax * N + b) * 2 + 1] = std::max(perm[2 * b], perm[2 * b + 1]);
        }
    }
}

std::vector<uint32_t> genBounds(std::mt19937& rng, uint32_t N)
{
    std::vector<uint32_t> b(N * 6);
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t lo = rng() % 100000, ext = 1 + rng() % 50000;
            b[i * 6 + ax] = lo;
            b[i * 6 + ax + 3] = lo + ext;
        }
    return b;
}

PXG_TEST(broadphaseMarkPairs_removed)
{
    std::mt19937 rng(22010);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox;
    genArrangement(rng, N, sapBox);

    // previous-frame handles: encode (handle<<3)|isStart at each box's positions
    std::vector<uint32_t> handles1(3 * nbProj, 0);
    for (uint32_t ax = 0; ax < 3; ++ax)
        for (uint32_t b = 0; b < N; ++b)
        {
            uint32_t mn = sapBox[(ax * N + b) * 2 + 0], mx = sapBox[(ax * N + b) * 2 + 1];
            handles1[ax * nbProj + mn] = (b << 3) | 1u;
            handles1[ax * nbProj + mx] = (b << 3) | 0u;
        }
    std::vector<uint32_t> oldBounds = genBounds(rng, N);

    // removed handles: ~30% distinct
    std::vector<uint32_t> removed;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 30) removed.push_back(b);
    const uint32_t numRem = uint32_t(removed.size());

    // reference
    std::vector<uint32_t> rHandles = handles1, rOld = oldBounds;
    for (uint32_t h : removed)
    {
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[(ax * N + h) * 2 + 0], mx = sapBox[(ax * N + h) * 2 + 1];
            rHandles[ax * nbProj + mn] |= (1u << 2);
            rHandles[ax * nbProj + mx] |= (1u << 2);
        }
        rOld[h * 6 + 0] = rOld[h * 6 + 1] = rOld[h * 6 + 2] = 0xff7fffffu;
        rOld[h * 6 + 3] = rOld[h * 6 + 4] = rOld[h * 6 + 5] = 0x00800000u;
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0), newB = genBounds(rng, N), proj(3 * nbProj, 0);
    PxgBufferHandle bRem = createBufferFrom(dev, removed);
    PxgBufferHandle bSap = createBufferFrom(dev, sapBox);
    PxgBufferHandle bH1 = createBufferFrom(dev, handles1);
    PxgBufferHandle bOld = createBufferFrom(dev, oldBounds);
    PxgBufferHandle bProj = createBufferFrom(dev, proj);
    PxgBufferHandle bNew = createBufferFrom(dev, newB);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    MarkParams p = { N, numRem, 0, 0, GRID_X };
    const std::vector<uint32_t> strides(10, 4);
    Kernel k("broadphaseMarkPairs.spv", "markRemovedPairsLaunch", strides, sizeof(MarkParams));
    k.dispatch(GRID_X, { bRem, bDummy, bSap, bH1, bProj, bOld, bNew, bDummy, bDummy, bDummy }, p);
    dev->flush();

    const auto gH = downloadBuffer<uint32_t>(dev, bH1, 3 * nbProj);
    const auto gOld = downloadBuffer<uint32_t>(dev, bOld, N * 6);
    size_t hM = 0, oM = 0;
    for (uint32_t i = 0; i < 3 * nbProj; ++i) if (gH[i] != rHandles[i]) ++hM;
    for (uint32_t i = 0; i < N * 6; ++i) if (gOld[i] != rOld[i]) ++oM;
    PXG_CHECK_MSG(hM == 0, "%zu handle-delete mismatches (numRem=%u)", hM, numRem);
    PXG_CHECK_MSG(oM == 0, "%zu old-bounds mismatches", oM);
    for (PxgBufferHandle h : { bRem, bSap, bH1, bOld, bProj, bNew, bDummy }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseMarkPairs_removedProjections)
{
    std::mt19937 rng(22020);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox;
    genArrangement(rng, N, sapBox);
    std::vector<uint32_t> proj(3 * nbProj);
    for (uint32_t i = 0; i < proj.size(); ++i) proj[i] = rng();

    std::vector<uint32_t> removed;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 30) removed.push_back(b);
    const uint32_t numRem = uint32_t(removed.size());

    std::vector<uint32_t> rProj = proj, rSap = sapBox;
    for (uint32_t h : removed)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[(ax * N + h) * 2 + 0], mx = sapBox[(ax * N + h) * 2 + 1];
            rProj[ax * nbProj + mn] = INVALID;
            rProj[ax * nbProj + mx] = INVALID;
            rSap[(ax * N + h) * 2 + 0] = INVALID;
            rSap[(ax * N + h) * 2 + 1] = INVALID;
        }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0), newB = genBounds(rng, N), h1(3 * nbProj, 0), old(N * 6, 0);
    PxgBufferHandle bRem = createBufferFrom(dev, removed);
    PxgBufferHandle bSap = createBufferFrom(dev, sapBox);
    PxgBufferHandle bProj = createBufferFrom(dev, proj);
    PxgBufferHandle bH1 = createBufferFrom(dev, h1);
    PxgBufferHandle bOld = createBufferFrom(dev, old);
    PxgBufferHandle bNew = createBufferFrom(dev, newB);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    MarkParams p = { N, numRem, 0, 0, GRID_X };
    const std::vector<uint32_t> strides(10, 4);
    Kernel k("broadphaseMarkPairs.spv", "markRemovedPairsProjectionsLaunch", strides, sizeof(MarkParams));
    k.dispatch(GRID_X, { bRem, bDummy, bSap, bH1, bProj, bOld, bNew, bDummy, bDummy, bDummy }, p);
    dev->flush();

    const auto gProj = downloadBuffer<uint32_t>(dev, bProj, 3 * nbProj);
    const auto gSap = downloadBuffer<uint32_t>(dev, bSap, 3 * N * 2);
    size_t pM = 0, sM = 0;
    for (uint32_t i = 0; i < 3 * nbProj; ++i) if (gProj[i] != rProj[i]) ++pM;
    for (uint32_t i = 0; i < 3 * N * 2; ++i) if (gSap[i] != rSap[i]) ++sM;
    PXG_CHECK_MSG(pM == 0, "%zu projection-invalidate mismatches (numRem=%u)", pM, numRem);
    PXG_CHECK_MSG(sM == 0, "%zu sapbox-invalidate mismatches", sM);
    for (PxgBufferHandle h : { bRem, bSap, bProj, bH1, bOld, bNew, bDummy }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseMarkPairs_updated)
{
    std::mt19937 rng(22030);
    const uint32_t words = 40, N = words * 32, nbProj = N * 2; // numElements == numHandles
    std::vector<uint32_t> sapBox;
    genArrangement(rng, N, sapBox);
    std::vector<uint32_t> newB = genBounds(rng, N);
    std::vector<uint32_t> changed(words), added(words), removedM(words);
    for (uint32_t w = 0; w < words; ++w) { changed[w] = rng(); added[w] = rng(); removedM[w] = rng(); }

    const uint32_t SENT = 0xdeadbeefu;
    std::vector<uint32_t> proj(3 * nbProj, SENT), rProj(3 * nbProj, SENT);
    for (uint32_t handle = 0; handle < N; ++handle)
    {
        uint32_t w = handle / 32;
        if (((changed[w] & ~(added[w] | removedM[w])) & (1u << (handle & 31))) != 0u)
            for (uint32_t ax = 0; ax < 3; ++ax)
            {
                uint32_t mn = sapBox[(ax * N + handle) * 2 + 0], mx = sapBox[(ax * N + handle) * 2 + 1];
                rProj[ax * nbProj + mn] = newB[handle * 6 + ax];
                rProj[ax * nbProj + mx] = newB[handle * 6 + ax + 3];
            }
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0), h1(3 * nbProj, 0), old(N * 6, 0);
    PxgBufferHandle bSap = createBufferFrom(dev, sapBox);
    PxgBufferHandle bProj = createBufferFrom(dev, proj);
    PxgBufferHandle bNew = createBufferFrom(dev, newB);
    PxgBufferHandle bCh = createBufferFrom(dev, changed);
    PxgBufferHandle bAdd = createBufferFrom(dev, added);
    PxgBufferHandle bRm = createBufferFrom(dev, removedM);
    PxgBufferHandle bH1 = createBufferFrom(dev, h1);
    PxgBufferHandle bOld = createBufferFrom(dev, old);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    MarkParams p = { N, 0, 0, words, GRID_X };
    const std::vector<uint32_t> strides(10, 4);
    Kernel k("broadphaseMarkPairs.spv", "markUpdatedPairsLaunch", strides, sizeof(MarkParams));
    k.dispatch(GRID_X, { bDummy, bDummy, bSap, bH1, bProj, bOld, bNew, bCh, bAdd, bRm }, p);
    dev->flush();

    const auto gProj = downloadBuffer<uint32_t>(dev, bProj, 3 * nbProj);
    size_t pM = 0;
    for (uint32_t i = 0; i < 3 * nbProj; ++i) if (gProj[i] != rProj[i]) ++pM;
    PXG_CHECK_MSG(pM == 0, "%zu updated-projection mismatches", pM);
    for (PxgBufferHandle h : { bSap, bProj, bNew, bCh, bAdd, bRm, bH1, bOld, bDummy }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseMarkPairs_updated2)
{
    std::mt19937 rng(22040);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox;
    genArrangement(rng, N, sapBox);
    std::vector<uint32_t> newB = genBounds(rng, N);

    std::vector<uint32_t> updated;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 35) updated.push_back(b);
    const uint32_t numUpd = uint32_t(updated.size());

    const uint32_t SENT = 0xdeadbeefu;
    std::vector<uint32_t> proj(3 * nbProj, SENT), rProj(3 * nbProj, SENT);
    for (uint32_t h : updated)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[(ax * N + h) * 2 + 0], mx = sapBox[(ax * N + h) * 2 + 1];
            rProj[ax * nbProj + mn] = newB[h * 6 + ax];
            rProj[ax * nbProj + mx] = newB[h * 6 + ax + 3];
        }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0), h1(3 * nbProj, 0), old(N * 6, 0);
    PxgBufferHandle bUpd = createBufferFrom(dev, updated);
    PxgBufferHandle bSap = createBufferFrom(dev, sapBox);
    PxgBufferHandle bProj = createBufferFrom(dev, proj);
    PxgBufferHandle bNew = createBufferFrom(dev, newB);
    PxgBufferHandle bH1 = createBufferFrom(dev, h1);
    PxgBufferHandle bOld = createBufferFrom(dev, old);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    MarkParams p = { N, 0, numUpd, 0, GRID_X };
    const std::vector<uint32_t> strides(10, 4);
    Kernel k("broadphaseMarkPairs.spv", "markUpdatedPairsLaunch2", strides, sizeof(MarkParams));
    k.dispatch(GRID_X, { bDummy, bUpd, bSap, bH1, bProj, bOld, bNew, bDummy, bDummy, bDummy }, p);
    dev->flush();

    const auto gProj = downloadBuffer<uint32_t>(dev, bProj, 3 * nbProj);
    size_t pM = 0;
    for (uint32_t i = 0; i < 3 * nbProj; ++i) if (gProj[i] != rProj[i]) ++pM;
    PXG_CHECK_MSG(pM == 0, "%zu updated2-projection mismatches (numUpd=%u)", pM, numUpd);
    for (PxgBufferHandle h : { bUpd, bSap, bProj, bNew, bH1, bOld, bDummy }) dev->destroyBuffer(h);
}

} // namespace
