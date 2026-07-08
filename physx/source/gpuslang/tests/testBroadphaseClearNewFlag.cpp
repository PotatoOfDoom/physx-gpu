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

// Verification of broadphaseClearNewFlag.slang against C++ transcriptions of
// the two remaining broad-phase glue kernels: clearNewFlagLaunch (rewrite the
// current frame's start/end sort handles for newly created boxes with the
// "new" flag cleared) and bpSignalComplete (single-thread completion-signal
// store).  Outputs are compared bit-exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t GRID_X = 32;

struct ClearParams
{
    uint32_t numHandles;
    uint32_t numCreatedHandles;
    uint32_t gridDimX;
};

// createHandle(handle, isStart, isNew) — PxgBroadPhaseDesc.h
uint32_t createHandle(uint32_t h, bool isStart, bool isNew)
{
    return (h << 3) | (isStart ? 1u : 0u) | (isNew ? 2u : 0u);
}

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

PXG_TEST(broadphaseClearNewFlag_clear)
{
    std::mt19937 rng(23010);
    const uint32_t N = 2000, nbProj = N * 2;
    std::vector<uint32_t> sapBox;
    genArrangement(rng, N, sapBox);

    // current-frame handles pre-filled with the "new" flag SET at every box's
    // start/end positions, so the kernel must clear it for created handles.
    std::vector<uint32_t> handles0(3 * nbProj, 0);
    for (uint32_t ax = 0; ax < 3; ++ax)
        for (uint32_t b = 0; b < N; ++b)
        {
            uint32_t mn = sapBox[(ax * N + b) * 2 + 0], mx = sapBox[(ax * N + b) * 2 + 1];
            handles0[ax * nbProj + mn] = createHandle(b, true, true);
            handles0[ax * nbProj + mx] = createHandle(b, false, true);
        }

    // created handles: ~40% distinct
    std::vector<uint32_t> created;
    for (uint32_t b = 0; b < N; ++b) if (rng() % 100 < 40) created.push_back(b);
    const uint32_t numCreated = uint32_t(created.size());

    // reference: clear the "new" flag (isNew=false) for created handles
    std::vector<uint32_t> rHandles = handles0;
    for (uint32_t h : created)
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            uint32_t mn = sapBox[(ax * N + h) * 2 + 0], mx = sapBox[(ax * N + h) * 2 + 1];
            rHandles[ax * nbProj + mn] = createHandle(h, true, false);
            rHandles[ax * nbProj + mx] = createHandle(h, false, false);
        }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle bCreated = createBufferFrom(dev, created);
    PxgBufferHandle bSap = createBufferFrom(dev, sapBox);
    PxgBufferHandle bH0 = createBufferFrom(dev, handles0);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    ClearParams p = { N, numCreated, GRID_X };
    const std::vector<uint32_t> strides(4, 4);
    Kernel k("broadphaseClearNewFlag.spv", "clearNewFlagLaunch", strides, sizeof(ClearParams));
    k.dispatch(GRID_X, { bCreated, bSap, bH0, bDummy }, p);
    dev->flush();

    const auto gH = downloadBuffer<uint32_t>(dev, bH0, 3 * nbProj);
    size_t hM = 0;
    for (uint32_t i = 0; i < 3 * nbProj; ++i) if (gH[i] != rHandles[i]) ++hM;
    PXG_CHECK_MSG(hM == 0, "%zu clear-new-flag mismatches (numCreated=%u)", hM, numCreated);
    for (PxgBufferHandle h : { bCreated, bSap, bH0, bDummy }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseClearNewFlag_signal)
{
    PxgSlangComputeDevice* dev = testDevice();
    // signal[0] must become 1; the neighbouring words must stay untouched.
    std::vector<uint32_t> signal = { 0u, 0xdeadbeefu, 0xcafef00du, 0x12345678u };
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle bSignal = createBufferFrom(dev, signal);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    ClearParams p = { 0, 0, 1 };
    const std::vector<uint32_t> strides(4, 4);
    Kernel k("broadphaseClearNewFlag.spv", "bpSignalComplete", strides, sizeof(ClearParams));
    k.dispatch(1, { bDummy, bDummy, bDummy, bSignal }, p);
    dev->flush();

    const auto gS = downloadBuffer<uint32_t>(dev, bSignal, 4);
    PXG_CHECK_MSG(gS[0] == 1u, "signal[0]=%u expected 1", gS[0]);
    PXG_CHECK_MSG(gS[1] == 0xdeadbeefu && gS[2] == 0xcafef00du && gS[3] == 0x12345678u,
                  "signal spilled past word 0: %u %u %u", gS[1], gS[2], gS[3]);
    for (PxgBufferHandle h : { bSignal, bDummy }) dev->destroyBuffer(h);
}

} // namespace
