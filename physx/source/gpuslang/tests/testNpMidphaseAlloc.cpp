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

// Verification of npMidphaseAlloc.slang: the convex-mesh midphase
// collision-stack allocator (midphaseAllocate.cuh — mem requirement,
// worst-case padding, maxPairs incl. its deliberate unsigned wrap, and the
// pointer-bump layout as byte offsets) plus the ConvexTriNormalAndIndex /
// ConvexTriIntermediateData bit-packing accessors (convexNpCommon.h).
// Everything is pure integer arithmetic, compared bit-exactly, with the
// struct sizes cross-checked against the CUDA layout constants.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U2 { uint32_t x, y; };
struct U4 { uint32_t x, y, z, w; };
struct MaIn { uint32_t unpadded, padded, stackBytes, numCMs; };
struct MaOut { U4 offsetsA, offsetsB, sizes; };
struct MaParams { uint32_t numTests, pad0, pad1, pad2; };

// CUDA struct sizes (convexNpCommon.h): uint4 16, ConvexTriNormalAndIndex
// (PxVec3+int) 16, ConvexTriContacts (PxU32) 4, PxReal 4,
// ConvexTriIntermediateData (PxU32) 4, PxU32 4.
static const uint32_t SZ_U4 = 16, SZ_NI = 16, SZ_CT = 4, SZ_IM = 4, SZ_R = 4, SZ_U = 4;

static uint32_t refMemReq() { return SZ_U4 + SZ_NI + SZ_CT + SZ_R + SZ_IM + SZ_U + SZ_U * 2; }
static uint32_t refPadding(uint32_t numCMs)
{
    return 3 * (refMemReq() - 2 * SZ_U) + numCMs * 6 * SZ_U;
}
static uint32_t refMaxPairs(uint32_t stackBytes, uint32_t numCMs)
{
    return (uint32_t)(stackBytes - refPadding(numCMs)) / refMemReq(); // wraps like CUDA
}

static MaOut refAlloc(const MaIn& e)
{
    const uint32_t n4 = (e.unpadded + 3) & ~3u;
    MaOut o;
    uint32_t p = 0;
    o.offsetsA.x = p; p += n4 * SZ_U4;
    o.offsetsA.y = p; p += n4 * SZ_NI;
    o.offsetsA.z = p; p += n4 * SZ_CT;
    o.offsetsA.w = p; p += e.padded * SZ_U;
    o.offsetsB.x = p; p += n4 * SZ_R;
    o.offsetsB.y = p; p += n4 * SZ_IM;
    o.offsetsB.z = p;
    o.offsetsB.w = refMemReq();
    o.sizes = { refPadding(e.numCMs), refMaxPairs(e.stackBytes, e.numCMs), n4, 0 };
    return o;
}

PXG_TEST(npMidphaseAlloc_alloc)
{
    std::mt19937 rng(79013);
    const uint32_t N = 8192;
    std::vector<MaIn> in(N);
    std::vector<MaOut> ref(N);
    size_t wrapped = 0, exact4 = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        MaIn e;
        e.unpadded = rng() % 100000;
        if (i % 64 == 0) e.unpadded = 0;                       // n4 == 0
        if (i % 64 == 32) e.unpadded = 0xfffffffdu + i % 3;    // (unpadded+3) wrap
        // padded count: per-CM 4-padding + doubled temp buffer, as the callers build it
        e.padded = ((e.unpadded + 3) & ~3u) * 2 + (rng() % 64) * 4;
        e.numCMs = rng() % 5000;
        if (i % 16 == 15)
        {
            e.stackBytes = rng() % refPadding(e.numCMs); // padding exceeds the stack -> wrap
            ++wrapped;
        }
        else
            e.stackBytes = 1000000 + rng() % 100000000;
        if (e.unpadded % 4 == 0) ++exact4;
        in[i] = e;
        ref[i] = refAlloc(e);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<MaOut> outInit(N);
    std::vector<U2> accD(N);
    std::vector<U4> accOutD(N);
    PxgBufferHandle bI = createBufferFrom(dev, in);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);
    PxgBufferHandle bA = createBufferFrom(dev, accD);
    PxgBufferHandle bAo = createBufferFrom(dev, accOutD);

    MaParams prm = { N, 0, 0, 0 };
    Kernel k("npMidphaseAlloc.spv", "ctMidphaseAlloc",
             { (uint32_t)sizeof(MaIn), (uint32_t)sizeof(MaOut), 8, 16 }, sizeof(MaParams));
    k.dispatch(GROUPS, { bI, bO, bA, bAo }, prm);
    dev->flush();

    const auto g = downloadBuffer<MaOut>(dev, bO, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const MaOut& r = ref[i]; const MaOut& v = g[i];
        if (v.offsetsA.x != r.offsetsA.x || v.offsetsA.y != r.offsetsA.y || v.offsetsA.z != r.offsetsA.z || v.offsetsA.w != r.offsetsA.w ||
            v.offsetsB.x != r.offsetsB.x || v.offsetsB.y != r.offsetsB.y || v.offsetsB.z != r.offsetsB.z || v.offsetsB.w != r.offsetsB.w ||
            v.sizes.x != r.sizes.x || v.sizes.y != r.sizes.y || v.sizes.z != r.sizes.z) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "ctMidphaseAlloc: %zu mismatches", mm);
    PXG_CHECK_MSG(wrapped > 100 && exact4 > 1000, "ctMidphaseAlloc: coverage thin (wrapped=%zu exact4=%zu)", wrapped, exact4);

    dev->destroyBuffer(bI); dev->destroyBuffer(bO); dev->destroyBuffer(bA); dev->destroyBuffer(bAo);
}

PXG_TEST(npMidphaseAlloc_accessors)
{
    std::mt19937 rng(79031);
    const uint32_t N = 8192;
    std::vector<U2> in(N);
    std::vector<U4> ref(N);
    size_t deferred = 0, edge = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t idx = rng();
        const uint32_t msk = rng();
        in[i] = { idx, msk };
        const uint32_t tri = idx & ((1u << 27) - 1);
        const uint32_t nb = (idx & ~0x80000000u) >> 27;
        const bool def = (idx & 0x80000000u) != 0;
        const bool edg = (msk & 0x80000000u) != 0;
        ref[i] = { tri, nb, (def ? 1u : 0u) | (edg ? 2u : 0u), 0 };
        if (def) ++deferred;
        if (edg) ++edge;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<MaIn> inD(N);
    std::vector<MaOut> outD(N);
    std::vector<U4> accOutInit(N);
    PxgBufferHandle bI = createBufferFrom(dev, inD);
    PxgBufferHandle bO = createBufferFrom(dev, outD);
    PxgBufferHandle bA = createBufferFrom(dev, in);
    PxgBufferHandle bAo = createBufferFrom(dev, accOutInit);

    MaParams prm = { N, 0, 0, 0 };
    Kernel k("npMidphaseAlloc.spv", "ctCvxTriAccessors",
             { (uint32_t)sizeof(MaIn), (uint32_t)sizeof(MaOut), 8, 16 }, sizeof(MaParams));
    k.dispatch(GROUPS, { bI, bO, bA, bAo }, prm);
    dev->flush();

    const auto g = downloadBuffer<U4>(dev, bAo, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (g[i].x != ref[i].x || g[i].y != ref[i].y || g[i].z != ref[i].z) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctCvxTriAccessors: %zu mismatches", mm);
    PXG_CHECK_MSG(deferred > 2000 && edge > 2000, "ctCvxTriAccessors: coverage thin (def=%zu edge=%zu)", deferred, edge);

    dev->destroyBuffer(bI); dev->destroyBuffer(bO); dev->destroyBuffer(bA); dev->destroyBuffer(bAo);
}

} // namespace
