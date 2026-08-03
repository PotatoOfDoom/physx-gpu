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

// P8.35 — initializeManifolds (initManifoldsBda.spv) on the engine ABI.
//
// Suite (ctest filter "initManifolds"):
//   initManifoldsReplicate — the replicated record, bit-exact, over several
//                            record sizes and counts including the ragged tail.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

struct InitPush { uint64_t destination, source; uint32_t dataSize, nbTimesToReplicate; };
static_assert(sizeof(InitPush) == 24, "init blob must be 24 bytes");

PXG_TEST(initManifoldsReplicate)
{
    PxgSlangComputeDevice* dev = testDevice();

    // (dataSize in bytes, nbTimesToReplicate).  256 and 1216 are the two real
    // manifold sizes (PxgPersistentContactManifold / ...MultiManifold); the others
    // probe the block packing.  With the engine's 512-thread block, 128 B gives
    // nbThreadsPerElement = 8 and nbPerBlock = 64, so one workgroup covers 64
    // records and the counts below straddle that boundary.
    struct Case { uint32_t dataSize, count; };
    const Case cases[] = {
        { 1216u, 1u }, { 1216u, 7u }, { 1216u, 64u },
        { 256u, 5u },       // PxgPersistentContactManifold
        { 128u, 1u }, { 128u, 63u }, { 128u, 64u }, { 128u, 65u }, { 128u, 100u },
        { 64u, 33u },       // nbThreadsPerElement = 4 -> nbPerBlock = 128
        { 2048u, 3u },      // nbThreadsPerElement = 128 -> nbPerBlock = 4, ragged
        { 4096u, 3u },      // MaxStructureSize: nbThreadsPerElement = 256, the
                            // largest record the groupshared array can hold, and
                            // the one a 128-thread block would divide by zero on
    };

    size_t mm = 0;
    for (const Case& c : cases)
    {
        const uint32_t f4PerRecord = c.dataSize / 16u;
        std::vector<float> src(f4PerRecord * 4u);
        for (size_t i = 0; i < src.size(); ++i) src[i] = 1.0f + float(i) * 0.25f;

        // A sentinel tail: one extra record's worth past the replicated range,
        // which must come back untouched.  The kernel's `idx < nbTimesToReplicate`
        // guard is the only thing stopping it, and a port that dropped the guard
        // writes exactly here.
        const uint32_t totalF4 = f4PerRecord * (c.count + 1u);
        std::vector<float> dst(size_t(totalF4) * 4u, -777.0f);

        PxgBufferHandle bS = createBufferFrom(dev, src);
        PxgBufferHandle bD = createBufferFrom(dev, dst);

        InitPush push{};
        push.destination = dev->getBufferDeviceAddress(bD);
        push.source = dev->getBufferDeviceAddress(bS);
        push.dataSize = c.dataSize;
        push.nbTimesToReplicate = c.count;

        // grid 128 x block 512 mirrors PxgNarrowPhaseGridDims/BlockDims
        // ::INITIALIZE_MANIFOLDS (the block size is declared in the shader).
        Kernel k("initManifoldsBda.spv", "initializeManifolds", {}, sizeof(InitPush));
        k.dispatch(128u, {}, push, 1u, 1u);
        dev->flush();

        const auto got = downloadBuffer<float>(dev, bD, uint32_t(dst.size()));

        size_t bad = 0;
        for (uint32_t r = 0; r < c.count; ++r)
            for (uint32_t f = 0; f < f4PerRecord * 4u; ++f)
            {
                const float e = src[f];
                const float g = got[size_t(r) * f4PerRecord * 4u + f];
                if (g != e) { ++bad; if (bad <= 4)
                    std::printf("  size=%u count=%u rec %u float %u: got=%g exp=%g\n",
                                c.dataSize, c.count, r, f, g, e); }
            }
        // the sentinel record must be untouched
        for (uint32_t f = 0; f < f4PerRecord * 4u; ++f)
        {
            const float g = got[size_t(c.count) * f4PerRecord * 4u + f];
            if (g != -777.0f)
            { ++bad; if (bad <= 6) std::printf("  size=%u count=%u: wrote past the range at float %u (%g)\n",
                                               c.dataSize, c.count, f, g); }
        }
        mm += bad;

        dev->destroyBuffer(bS);
        dev->destroyBuffer(bD);
    }

    PXG_CHECK_MSG(mm == 0, "initializeManifolds replicates bit-exact (%zu mismatches)", mm);
}

} // namespace
