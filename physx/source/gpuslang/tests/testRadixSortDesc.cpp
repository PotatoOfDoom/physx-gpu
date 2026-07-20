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

// P5 Step 4: the descriptor-based multi-launch radix sort on the ENGINE ABI —
// reproduces PxgCudaBroadPhaseSap::runRadixSort exactly: THREE contiguous
// PxgRadixSortDesc entries (one per axis) with ping-pong buffers, 8 passes of
// (histogram, ranks) with startBit += 4 and desc base offset (pass&1)*3, grid
// (32, 3, 1) with blockIdx.y selecting the axis.  After 8 passes each axis's
// key/rank arrays are fully sorted; compared bitwise against a CPU stable sort.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <algorithm>
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

static std::vector<uint8_t> rsdLoader(const char* file) { return loadShaderBlob(file); }

// mirrors PxgRadixSortDesc (5 pointers + count; sizeof = 48)
struct RadixSortDescMirror
{
    uint64_t inputKeys;
    uint64_t inputRanks;
    uint64_t outputKeys;
    uint64_t outputRanks;
    uint64_t radixBlockCounts;
    uint32_t count;
    uint32_t pad;
};
static_assert(sizeof(RadixSortDescMirror) == 48, "desc mirror must be 48 bytes");

PXG_TEST(radixSortDesc_enginePasses)
{
    std::mt19937 rng(0x5EED);
    const uint32_t N = 6000; // odd-ish size across many blocks
    const uint32_t RADIX_BLOCKS = 32, RADIX_SIZE = 16;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "radixSortDesc.spv", "radixSortMultiBlockLaunchWithCount", 16u, 0 },
        { "radixSortDesc.spv", "radixSortMultiCalculateRanksLaunchWithCount", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &rsdLoader);

    // per-axis keys (32-bit) + identity ranks
    std::vector<std::vector<uint32_t>> keys(3), ranks(3);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        keys[ax].resize(N);
        ranks[ax].resize(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            keys[ax][i] = rng();
            ranks[ax][i] = i;
        }
    }

    // ping-pong device buffers per axis + per-axis histograms
    PxgDevicePtr dKeys[3][2], dRanks[3][2], dCounts[3];
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        for (uint32_t s = 0; s < 2; ++s)
        {
            shim.memAlloc(&dKeys[ax][s], N * 4);
            shim.memAlloc(&dRanks[ax][s], N * 4);
        }
        shim.memAlloc(&dCounts[ax], RADIX_SIZE * RADIX_BLOCKS * 4);
        if (!dKeys[ax][0]) { std::printf("  NOTE: no BDA -> skip\n"); return; }
        shim.memcpyHtoD(dKeys[ax][0], keys[ax].data(), N * 4);
        shim.memcpyHtoD(dRanks[ax][0], ranks[ax].data(), N * 4);
    }

    // SIX descs: [0..2] = pass-even (in=slot0, out=slot1), [3..5] = pass-odd
    // (in=slot1, out=slot0) — the engine's radixSortDescBuf ping-pong layout.
    RadixSortDescMirror descs[6];
    std::memset(descs, 0, sizeof(descs));
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        descs[ax] = { dKeys[ax][0], dRanks[ax][0], dKeys[ax][1], dRanks[ax][1], dCounts[ax], N, 0 };
        descs[3 + ax] = { dKeys[ax][1], dRanks[ax][1], dKeys[ax][0], dRanks[ax][0], dCounts[ax], N, 0 };
    }
    PxgDevicePtr dDescs = 0;
    shim.memAlloc(&dDescs, sizeof(descs));
    shim.memcpyHtoD(dDescs, descs, sizeof(descs));

    // 8 passes exactly like runRadixSort: desc base += (pass&1)*3, startBit += 4.
    for (uint32_t pass = 0; pass < 8; ++pass)
    {
        struct { uint64_t desc; uint32_t numKeys, startBit; } prm =
            { dDescs + uint64_t(pass & 1) * 3 * sizeof(RadixSortDescMirror), N, pass * 4 };
        PXG_CHECK_MSG(shim.launchKernel(0, RADIX_BLOCKS, 3, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "histogram pass %u", pass);
        PXG_CHECK_MSG(shim.launchKernel(1, RADIX_BLOCKS, 3, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "ranks pass %u", pass);
    }
    shim.streamSynchronize();

    // after 8 passes the result is back in slot 0
    size_t bad = 0;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        // CPU reference: stable sort of (key, originalIndex)
        std::vector<uint32_t> order(N);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(),
                         [&](uint32_t a, uint32_t b) { return keys[ax][a] < keys[ax][b]; });

        std::vector<uint32_t> gk(N), gr(N);
        shim.memcpyDtoH(gk.data(), dKeys[ax][0], N * 4);
        shim.memcpyDtoH(gr.data(), dRanks[ax][0], N * 4);
        for (uint32_t i = 0; i < N; ++i)
        {
            if (gk[i] != keys[ax][order[i]]) ++bad;
            if (gr[i] != order[i]) ++bad; // identity ranks => rank == original index (stability!)
        }
    }
    PXG_CHECK_MSG(bad == 0, "engine 8-pass 3-axis radix sort bitwise-correct (%zu bad)", bad);

    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        for (uint32_t s = 0; s < 2; ++s) { shim.memFree(dKeys[ax][s]); shim.memFree(dRanks[ax][s]); }
        shim.memFree(dCounts[ax]);
    }
    shim.memFree(dDescs);
}

} // namespace
