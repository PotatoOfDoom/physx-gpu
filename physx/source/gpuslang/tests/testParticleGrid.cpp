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

// P6.2 particle grid build + reorder test: the four grid-stage kernels on the
// engine ABI (BDA).
//   (1,2) radixSortMultiBlockLaunchWithoutCount /
//         radixSortMultiCalculateRanksLaunchWithoutCount (radixSortDesc.spv) —
//         numKeys sourced from desc.count.  Run the same 8-pass 3-axis engine
//         sort as testRadixSortDesc (but with count in the descriptor and no
//         numKeys push param) and compare bitwise against a CPU stable sort.
//   (3)   ps_calculateHashLaunch (particleGridBda.spv) — push sentinel
//         particles through a REAL PxgParticleSystem and verify the computed
//         grid-cell hash + identity mapping match a CPU transcription of
//         gridCal.cuh's calcGridPos/calcGridHash.
//   (4)   ps_reorderDataAndFindCellStartLaunch (particleGridBda.spv) — a small
//         permutation must produce the right sorted arrays + cellStart/cellEnd.
//
// A static-offset proof (particleGridDescLayout) pins the pxgSlangParticleDesc
// mirror fields these two grid kernels dereference against the real engine
// PxgParticleSystem (the 3-way validation P6.0 used: SPIR-V ArrayStride is
// implied by the 656-byte stride check, offsetof here, device round-trip in the
// kernel tests).  No mirror extension was needed for P6.2 — every field both
// kernels touch was already present from P6.0.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

// no-CUDA-toolkit stand-in expanding the __device__/__align__ qualifiers the
// engine descriptor header decorates its inline members with; must precede
// PxgParticleSystem.h so its device-qualified ctors parse.
#include "host_defines.h"
#include "PxgParticleSystem.h"   // real engine per-system descriptor

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };

// A device buffer + its buffer-device-address (BDA).
struct DevBuf
{
    PxgBufferHandle h = 0;
    uint64_t addr = 0;
};

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// Zeroes and returns a real PxgParticleSystem (no vtable — memset-safe).
static PxgParticleSystem makeZeroDesc()
{
    PxgParticleSystem ps;
    std::memset(&ps, 0, sizeof(ps));
    return ps;
}

// CPU transcription of gridCal.cuh calcGridPos + calcGridHash.
static void cpuGridPos(const F4& p, float cellWidth, int32_t& gx, int32_t& gy, int32_t& gz)
{
    gx = int32_t(std::floor(p.x / cellWidth));
    gy = int32_t(std::floor(p.y / cellWidth));
    gz = int32_t(std::floor(p.z / cellWidth));
}
static uint32_t cpuGridHash(int32_t gx, int32_t gy, int32_t gz, uint32_t sx, uint32_t sy, uint32_t sz)
{
    const uint32_t hx = uint32_t(gx) & (sx - 1u);
    const uint32_t hy = uint32_t(gy) & (sy - 1u);
    const uint32_t hz = uint32_t(gz) & (sz - 1u);
    return ((hz * sy) * sx) + (hy * sx) + hx;
}

static std::vector<uint8_t> pgLoader(const char* file) { return loadShaderBlob(file); }

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

// ---------------------------------------------------------------------------
// Descriptor mirror layout proof — pins the fields the grid kernels dereference.
// ---------------------------------------------------------------------------
PXG_TEST(particleGridDescLayout)
{
    // stride (SPIR-V ArrayStride) — the kernels index particleSystems[id].
    PXG_CHECK_MSG(sizeof(PxgParticleSystem) == 656, "PxgParticleSystem sizeof == 656 (got %zu)",
                  sizeof(PxgParticleSystem));

    // ps_calculateHashLaunch fields
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedPositions_InvMass) == 0);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedToUnsortedMapping) == 56);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) == 88);
    PXG_CHECK(offsetof(PxGpuParticleData, mGridCellWidth) == 0);   // -> struct off 88
    PXG_CHECK(offsetof(PxGpuParticleData, mGridSizeX) == 4);       // -> struct off 92
    PXG_CHECK(offsetof(PxGpuParticleData, mGridSizeY) == 8);       // -> struct off 96
    PXG_CHECK(offsetof(PxGpuParticleData, mGridSizeZ) == 12);      // -> struct off 100
    PXG_CHECK(offsetof(PxGpuParticleData, mNumParticles) == 28);   // -> struct off 116
    PXG_CHECK(offsetof(PxgParticleSystem, mGridParticleHash) == 144);

    // ps_reorderDataAndFindCellStartLaunch fields (regular-particle path)
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedVelocities) == 8);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedPhaseArray) == 16);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPositions_InvMass) == 24);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedVelocities) == 32);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPhaseArray) == 40);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedToSortedMapping) == 48);
    PXG_CHECK(offsetof(PxGpuParticleData, mMaxDiffuseParticles) == 40); // -> struct off 128
    PXG_CHECK(offsetof(PxgParticleSystem, mOriginPos_InvMass) == 136);
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedDeltaP) == 160);
    PXG_CHECK(offsetof(PxgParticleSystem, mCellStart) == 168);
    PXG_CHECK(offsetof(PxgParticleSystem, mCellEnd) == 176);
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedOriginPos_InvMass) == 392);
    PXG_CHECK(offsetof(PxgParticleSystem, mNumDiffuseParticles) == 488);
}

// ---------------------------------------------------------------------------
// (1,2) radix WithoutCount pair — 8-pass 3-axis engine sort, numKeys from count.
// ---------------------------------------------------------------------------
PXG_TEST(particleGridRadixWithoutCount)
{
    std::mt19937 rng(0x6C0DE);
    const uint32_t N = 6000; // odd-ish size across many blocks
    const uint32_t RADIX_BLOCKS = 32, RADIX_SIZE = 16;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "radixSortDesc.spv", "radixSortMultiBlockLaunchWithoutCount", 16u, 0 },
        { "radixSortDesc.spv", "radixSortMultiCalculateRanksLaunchWithoutCount", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &pgLoader);

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

    // SIX descs with count == N: [0..2] pass-even, [3..5] pass-odd ping-pong.
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

    // 8 passes: desc base += (pass&1)*3, startBit += 4.  NO numKeys push param —
    // the kernel reads it from desc.count.
    for (uint32_t pass = 0; pass < 8; ++pass)
    {
        struct { uint64_t desc; uint32_t startBit; } prm =
            { dDescs + uint64_t(pass & 1) * 3 * sizeof(RadixSortDescMirror), pass * 4 };
        PXG_CHECK_MSG(shim.launchKernel(0, RADIX_BLOCKS, 3, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "histogram pass %u", pass);
        PXG_CHECK_MSG(shim.launchKernel(1, RADIX_BLOCKS, 3, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "ranks pass %u", pass);
    }
    shim.streamSynchronize();

    size_t bad = 0;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
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
    PXG_CHECK_MSG(bad == 0, "WithoutCount 8-pass 3-axis radix sort bitwise-correct (%zu bad)", bad);

    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        for (uint32_t s = 0; s < 2; ++s) { shim.memFree(dKeys[ax][s]); shim.memFree(dRanks[ax][s]); }
        shim.memFree(dCounts[ax]);
    }
    shim.memFree(dDescs);
}

// ---------------------------------------------------------------------------
// (3) ps_calculateHashLaunch — grid-cell hash + identity mapping.
// ---------------------------------------------------------------------------
PXG_TEST(particleGridCalculateHash)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float cellWidth = 0.5f;
    const uint32_t gsx = 32, gsy = 32, gsz = 32; // power-of-two grid

    // Sentinel positions — include negatives to exercise the two's-complement wrap.
    std::vector<F4> positions(N);
    positions[0] = { 0.10f, 0.10f, 0.10f, 1.0f };
    positions[1] = { 0.60f, 0.10f, 0.10f, 1.0f };
    positions[2] = { 1.30f, 2.70f, 0.40f, 1.0f };
    positions[3] = { -0.25f, 0.10f, 0.10f, 1.0f };   // negative x -> wrap
    positions[4] = { -1.75f, -3.10f, -0.10f, 1.0f }; // all negative -> wrap
    positions[5] = { 5.50f, 6.25f, 7.75f, 1.0f };
    positions[6] = { 15.90f, 0.05f, 0.05f, 0.0f };   // w==0 still hashed (kernel ignores w)
    positions[7] = { 0.00f, 0.00f, 0.00f, 1.0f };

    std::vector<uint32_t> hashArray(N, 0xdeadbeefu);   // sentinel
    std::vector<uint32_t> mapping(N, 0xffffffffu);     // sentinel

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bHash = makeDev(dev, hashArray);
    DevBuf bMap = makeDev(dev, mapping);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    ps.mGridParticleHash = reinterpret_cast<PxU32*>(size_t(bHash.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bMap.addr));
    ps.mCommonData.mNumParticles = N;
    ps.mCommonData.mGridCellWidth = cellWidth;
    ps.mCommonData.mGridSizeX = gsx;
    ps.mCommonData.mGridSizeY = gsy;
    ps.mCommonData.mGridSizeZ = gsz;

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    struct HashPush { uint64_t particleSystems, activeParticleSystems; };
    HashPush push = { bDesc.addr, bActive.addr };

    Kernel k("particleGridBda.spv", "ps_calculateHashLaunch", {}, sizeof(HashPush));
    // grid = (numBlocks, numActiveParticleSystems, 1) = (1, 1, 1)
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference.
    std::vector<uint32_t> refHash(N), refMap(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        int32_t gx, gy, gz;
        cpuGridPos(positions[i], cellWidth, gx, gy, gz);
        refHash[i] = cpuGridHash(gx, gy, gz, gsx, gsy, gsz);
        refMap[i] = i;
    }

    const auto gHash = downloadBuffer<uint32_t>(dev, bHash.h, N);
    const auto gMap = downloadBuffer<uint32_t>(dev, bMap.h, N);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (gHash[i] != refHash[i]) { ++mm; std::printf("  hash MM %u: ref %u got %u\n", i, refHash[i], gHash[i]); }
        if (gMap[i] != refMap[i]) { ++mm; std::printf("  map MM %u: ref %u got %u\n", i, refMap[i], gMap[i]); }
    }
    PXG_CHECK_MSG(mm == 0, "ps_calculateHashLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bHash.h); dev->destroyBuffer(bMap.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// (4) ps_reorderDataAndFindCellStartLaunch — gather + cell start/end.
// ---------------------------------------------------------------------------
PXG_TEST(particleGridReorder)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const uint32_t numCells = 8;
    const uint32_t EMPTY = 0xffffffffu;

    // Hashes are the ALREADY-SORTED cell ids (as the radix sort leaves them),
    // with duplicates to exercise cell boundaries: [0,0,2,2,2,5,7,7].
    std::vector<uint32_t> sortedHash = { 0u, 0u, 2u, 2u, 2u, 5u, 7u, 7u };
    // gridParticleIndex (mSortedToUnsortedMapping) — a permutation of [0,N).
    std::vector<uint32_t> mapping = { 3u, 1u, 0u, 5u, 2u, 7u, 4u, 6u };

    std::vector<F4> unsortedPos(N), unsortedVel(N), originPos(N);
    std::vector<uint32_t> phases(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        unsortedPos[i] = { float(i) + 0.25f, 10.f + float(i), -float(i) * 2.f, 1.0f + 0.1f * float(i) };
        unsortedVel[i] = { 0.5f * float(i), -0.3f * float(i), 0.2f, 0.0f };
        originPos[i] = { float(i) * 0.5f, float(i) + 1.f, 2.f - float(i), 1.0f };
        phases[i] = 0x40000000u | (i * 7u);
    }

    // outputs pre-seeded with sentinels.
    std::vector<F4> sortedPos(N, { -1.f, -1.f, -1.f, -1.f });
    std::vector<F4> sortedVel(N, { -2.f, -2.f, -2.f, -2.f });
    std::vector<F4> sortedOrigin(N, { -3.f, -3.f, -3.f, -3.f });
    std::vector<F4> sortedDeltaP(N, { -4.f, -4.f, -4.f, -4.f });
    std::vector<uint32_t> sortedPhases(N, 0xa5a5a5a5u);
    std::vector<uint32_t> reverseLookup(N, 0xbbbbbbbbu);
    std::vector<uint32_t> cellStart(numCells, EMPTY);       // launch site memsets to EMPTY_CELL
    std::vector<uint32_t> cellEnd(numCells, 0xccccccccu);    // sentinel

    DevBuf bHash = makeDev(dev, sortedHash);
    DevBuf bMap = makeDev(dev, mapping);
    DevBuf bUPos = makeDev(dev, unsortedPos);
    DevBuf bUVel = makeDev(dev, unsortedVel);
    DevBuf bOrigin = makeDev(dev, originPos);
    DevBuf bPhases = makeDev(dev, phases);
    DevBuf bSPos = makeDev(dev, sortedPos);
    DevBuf bSVel = makeDev(dev, sortedVel);
    DevBuf bSOrigin = makeDev(dev, sortedOrigin);
    DevBuf bSDeltaP = makeDev(dev, sortedDeltaP);
    DevBuf bSPhases = makeDev(dev, sortedPhases);
    DevBuf bReverse = makeDev(dev, reverseLookup);
    DevBuf bCellStart = makeDev(dev, cellStart);
    DevBuf bCellEnd = makeDev(dev, cellEnd);

    PxgParticleSystem ps = makeZeroDesc();
    ps.mGridParticleHash = reinterpret_cast<PxU32*>(size_t(bHash.addr));
    ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(bMap.addr));
    ps.mUnsortedToSortedMapping = reinterpret_cast<PxU32*>(size_t(bReverse.addr));
    ps.mCellStart = reinterpret_cast<PxU32*>(size_t(bCellStart.addr));
    ps.mCellEnd = reinterpret_cast<PxU32*>(size_t(bCellEnd.addr));
    ps.mUnsortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bUPos.addr));
    ps.mUnsortedVelocities = reinterpret_cast<float4*>(size_t(bUVel.addr));
    ps.mUnsortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bPhases.addr));
    ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bSPos.addr));
    ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(bSVel.addr));
    ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bSPhases.addr));
    ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(bSDeltaP.addr));
    ps.mOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bOrigin.addr));
    ps.mSortedOriginPos_InvMass = reinterpret_cast<float4*>(size_t(bSOrigin.addr));
    ps.mCommonData.mNumParticles = N;
    ps.mCommonData.mMaxDiffuseParticles = 0; // diffuse slab (blockIdx.z==1) returns early

    std::vector<PxgParticleSystem> descArr = { ps };
    DevBuf bDesc = makeDev(dev, descArr);

    struct ReorderPush { uint64_t particleSystems; uint32_t id; };
    ReorderPush push = { bDesc.addr, 0u };

    Kernel k("particleGridBda.spv", "ps_reorderDataAndFindCellStartLaunch", {}, sizeof(ReorderPush));
    // grid = (numBlocks, 1, 2) — the diffuse slab (z==1) early-returns (maxDiffuse==0).
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 2u);
    dev->flush();

    // CPU reference.
    std::vector<F4> refSPos(N), refSVel(N), refSOrigin(N), refSDeltaP(N);
    std::vector<uint32_t> refSPhases(N), refReverse = reverseLookup;
    std::vector<uint32_t> refCellStart(numCells, EMPTY), refCellEnd = cellEnd;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t si = mapping[i];
        const F4 p = unsortedPos[si];
        const F4 v = unsortedVel[si];
        const uint32_t hash = sortedHash[i];
        const uint32_t otherHash = (i > 0) ? sortedHash[i - 1] : 0u;

        if (i == 0 || hash != otherHash)
        {
            refCellStart[hash] = i;
            if (i > 0) refCellEnd[otherHash] = i;
        }
        if (i == N - 1) refCellEnd[hash] = i + 1;

        refSPos[i] = p;
        refSVel[i] = v;
        refReverse[si] = i;
        const F4 orig = originPos[si];
        refSOrigin[i] = orig;
        refSDeltaP[i] = { p.x - orig.x, p.y - orig.y, p.z - orig.z, p.w };
        refSPhases[i] = phases[si];
    }

    const auto gSPos = downloadBuffer<F4>(dev, bSPos.h, N);
    const auto gSVel = downloadBuffer<F4>(dev, bSVel.h, N);
    const auto gSOrigin = downloadBuffer<F4>(dev, bSOrigin.h, N);
    const auto gSDeltaP = downloadBuffer<F4>(dev, bSDeltaP.h, N);
    const auto gSPhases = downloadBuffer<uint32_t>(dev, bSPhases.h, N);
    const auto gReverse = downloadBuffer<uint32_t>(dev, bReverse.h, N);
    const auto gCellStart = downloadBuffer<uint32_t>(dev, bCellStart.h, numCells);
    const auto gCellEnd = downloadBuffer<uint32_t>(dev, bCellEnd.h, numCells);

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8)
                std::printf("  %s MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", what, i,
                            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp4(refSPos[i], gSPos[i], "sortedPos", i);
        cmp4(refSVel[i], gSVel[i], "sortedVel", i);
        cmp4(refSOrigin[i], gSOrigin[i], "sortedOrigin", i);
        cmp4(refSDeltaP[i], gSDeltaP[i], "sortedDeltaP", i);
        if (gSPhases[i] != refSPhases[i]) { ++mm; std::printf("  phase MM %u\n", i); }
        if (gReverse[i] != refReverse[i]) { ++mm; std::printf("  reverse MM %u: ref %u got %u\n", i, refReverse[i], gReverse[i]); }
    }
    for (uint32_t c = 0; c < numCells; ++c)
    {
        if (gCellStart[c] != refCellStart[c]) { ++mm; std::printf("  cellStart MM %u: ref %u got %u\n", c, refCellStart[c], gCellStart[c]); }
        if (gCellEnd[c] != refCellEnd[c]) { ++mm; std::printf("  cellEnd MM %u: ref %u got %u\n", c, refCellEnd[c], gCellEnd[c]); }
    }
    PXG_CHECK_MSG(mm == 0, "ps_reorderDataAndFindCellStartLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bHash.h); dev->destroyBuffer(bMap.h); dev->destroyBuffer(bUPos.h);
    dev->destroyBuffer(bUVel.h); dev->destroyBuffer(bOrigin.h); dev->destroyBuffer(bPhases.h);
    dev->destroyBuffer(bSPos.h); dev->destroyBuffer(bSVel.h); dev->destroyBuffer(bSOrigin.h);
    dev->destroyBuffer(bSDeltaP.h); dev->destroyBuffer(bSPhases.h); dev->destroyBuffer(bReverse.h);
    dev->destroyBuffer(bCellStart.h); dev->destroyBuffer(bCellEnd.h); dev->destroyBuffer(bDesc.h);
}

}  // namespace
