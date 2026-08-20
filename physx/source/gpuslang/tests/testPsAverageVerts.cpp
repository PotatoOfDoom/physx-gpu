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

// testPsAverageVerts.cpp -- P9.3c fixture for ps_averageVertsLaunch, ported as
// the fourth entry point of npPsSprings.slang.
//
// ORACLE: a host transcription of particlesystem.cu:4331-4425, compared
// bit-exactly over the whole mAccumDeltaP region plus every read-only array.
// The record layouts it walks are pinned by psAverageVertsLayout below and by
// psSpringsLayout in testPsSprings.cpp.
//
// THE KERNEL IS A READ-MODIFY-WRITE ACCUMULATE, so the single most likely way to
// ship a green test that proves nothing is to leave mAccumDeltaP zeroed: then
// `deltaP + averageDeltaP` and `averageDeltaP` are the same number.  It is
// therefore PRE-SEEDED non-zero in all four lanes at every slot.  Likewise the
// .w lane: in the live pipeline ps_solveSpringsLaunch writes each endpoint's
// inverse mass into its copy, so diff.w cancels to exactly 0 and a float3 port
// is accidentally correct -- this fixture deliberately breaks that invariant on
// some copies so the .w arithmetic is observable.
//
// THREE INDEX SPACES, and the fixture exists to keep them apart:
//   groupThreadIdx  buffer-local, indexes mAccumulatedCopiesPerParticles
//   index           = groupThreadIdx + runsum[particleBufferIndex], the search key
//   sortedInd       = mUnsortedToSortedMapping[index], indexes every float4 array
// The probe cloth buffers sit at offset > 0, the mapping is a permutation with
// no fixed points and is not an involution, and no two of the three ever
// coincide for a probe.
//
// POISON ON BOTH SIDES OF EVERY INDEX, with two POLARITY rules the house
// 0xABABABAB would break:
//   * mAccumulatedCopiesPerParticles gets a lead-in word and a tail, and both
//     must be TERMINATING: numCopies = endInd - startInd is unsigned, so a
//     0xABABABAB lead-in gives ~4e9 copies and the loop hangs the GPU instead of
//     failing the test.  Lead-in = 2, tail = the total (a mutant reading it gets
//     start == end and skips the store, which is a visible mismatch).
//   * mSortedClothStartIndices' tail pad must be SMALL, not 0xABABABAB:
//     binarySearch shrinks `right` whenever data[pos] > value, so a huge pad
//     makes a numElements+1 mutant behave exactly like the correct search.
//
// GEOMETRY -- the standing pool, with the grid deliberately NOT square:
//   pool[0] = A      : 3 live cloth buffers over 3 particle buffers, NON-identity
//                      cloth-buffer -> particle-buffer mapping
//   pool[1] = DECOY  : never active, 4 live cloth buffers -- all outputs must
//                      come back exactly as seeded
//   pool[2] = B      : 1 live cloth buffer with mNumCloths == 0 over a real
//                      zero-length array followed by a poison record that WOULD
//                      pass the range test if it were read
//   active = { 2, 0 }  -- not a permutation of the pool, so indexing the pool by
//                         blockIdx.z directly hits the decoy and dies
// gridY = 4 (the decoy's cloth count) vs gridZ = 2, so a Y/Z swap is not an
// identity -- the lesson P9.3b's sweep paid for.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxParticleBuffer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// { particleSystems*, activeParticleSystems*, PxReal invDt } -> 20 bytes,
// rounded to the 8-alignment -> 24.
struct AvPush
{
    uint64_t particleSystems;
    uint64_t activeParticleSystems;
    float invDt;
    uint32_t pad20;
};
static_assert(sizeof(AvPush) == 24, "the ps_averageVertsLaunch blob is 24 bytes");

// ---------------------------------------------------------------------------
// Layout proof -- every constant the kernel hard-codes, against the real headers.
// ---------------------------------------------------------------------------
PXG_TEST(psAverageVertsLayout)
{
    // A14: seven 4-byte members, align 4, NO trailing pad -- the array stride is
    // 28 and not rounded up to 32.  This is the kernel's only non-8-aligned walk.
    PXG_CHECK_MSG(sizeof(PxParticleCloth) == 28, "sizeof(PxParticleCloth) == 28 (got %zu)",
                  sizeof(PxParticleCloth));
    PXG_CHECK(alignof(PxParticleCloth) == 4);
    PXG_CHECK(offsetof(PxParticleCloth, startVertexIndex) == 0);
    PXG_CHECK(offsetof(PxParticleCloth, numVertices) == 4);
    PXG_CHECK(offsetof(PxParticleCloth, clothBlendScale) == 8);

    PXG_CHECK_MSG(sizeof(PxgParticleSimBuffer) == 88, "sizeof(PxgParticleSimBuffer) == 88 (got %zu)",
                  sizeof(PxgParticleSimBuffer));
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mNumActiveParticles) == 56);

    // the cloth-record fields this kernel reads (the rest are pinned by
    // psSpringsLayout in testPsSprings.cpp)
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mAccumulatedCopiesPerParticles) == 8);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mSortedClothStartIndices) == 40);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mCloths) == 48);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mRemapPositions) == 56);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mParticleBufferIndex) == 88);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumSprings) == 92);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumCloths) == 100);

    // the descriptor fields it walks
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPositions_InvMass) == 24);
    // A6: the two mappings are ADJACENT 8-byte slots -- the wrong one is a
    // one-line slip, which is why the fixture populates both.
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedToSortedMapping) == 48);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedToUnsortedMapping) == 56);
    PXG_CHECK(offsetof(PxgParticleSystem, mAccumDeltaP) == 152);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleSimBuffers) == 400);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408);
    PXG_CHECK(offsetof(PxgParticleSystem, mClothSimBuffers) == 432);
    PXG_CHECK(offsetof(PxgParticleSystem, mNumClothBuffers) == 464);
}

// ---------------------------------------------------------------------------
// Host mirror of gpucommon/src/CUDA/reduction.cuh:726-752, transcribed.
// ---------------------------------------------------------------------------
static uint32_t refBinarySearch(const uint32_t* data, uint32_t numElements, uint32_t value)
{
    uint32_t left = 0u, right = numElements;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2u;
        const uint32_t element = data[pos];
        if (element <= value) left = pos + 1u;
        else                  right = pos;
    }
    return left ? left - 1u : 0u;
}

static const uint32_t POISON_U32 = 0xABABABABu;
static const uint32_t BLOCK = 1024u;   // PxgParticleSystemKernelBlockDim::UPDATEBOUND
static const uint32_t CLOTH_ARRAY_SLOTS = 4u;
static const uint32_t COPIES_LEAD_IN = 1u;
static const uint32_t MAPPING_TAIL = 2200u;
// Large enough that a dropped `groupThreadIdx < nbParticles` guard -- which lets
// gti run all the way to gridX*1024 -- still lands inside the allocation.  The
// tail's SHAPE matters as much as its size: a flat tail gives start == end, the
// numCopies guard skips, and the dropped-bound mutant is an IDENTITY.  It
// therefore RISES for eight entries (so the first eight out-of-bound threads
// each find a one-element run pointing at the eight poison slots past the copy
// region, and write where the reference does not) and then goes flat (so every
// thread beyond that is safely inert).
static const uint32_t COPIES_TAIL = 2100u;

static uint32_t hashu(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

struct ClothRec
{
    uint32_t particleBufferIndex = 0;
    uint32_t numSprings = 0;
    uint32_t numCloths = 0;              // what the record reports (B.cb0 says 0)
    bool live = false;                   // active system AND inside mNumClothBuffers

    std::vector<uint32_t> startIndices;  // numCloths live + 1 SMALL pad (see header)
    std::vector<PxParticleCloth> cloths; // numCloths live + 2 pad records
    std::vector<uint32_t> copies;        // lead-in + nbParticles + tail
    std::vector<F4> remap;               // 2*nbSprings + totalCopies + 8 poison
    uint32_t nbParticles = 0;            // from the PARTICLE buffer it points at
    uint32_t totalCopies = 0;

    DevBuf bStart, bCloths, bCopies, bRemap;
};

struct SysRec
{
    uint32_t numClothBuffers = 0;
    uint32_t numParticles = 0;                 // the sorted-array extent
    std::vector<uint32_t> runsum;              // one entry per particle buffer + total
    std::vector<uint32_t> nbActivePerBuffer;
    std::vector<ClothRec> cloths;
    std::vector<F4> pos, accum0;               // seeds
    std::vector<uint32_t> u2s, s2u;
    DevBuf bPos, bAccum, bU2S, bS2U, bRunsum, bSimBufs, bClothArr;
    bool active = false;
};

struct Coverage
{
    uint32_t stores = 0, skippedFixed = 0, zeroCopyLive = 0, sentinelZero = 0;
    uint32_t offsetNonZeroProbes = 0, blendBranch = 0, fallbackBranch = 0;
    uint32_t exactKeyHit = 0, belowFirstStart = 0, pastLastCloth = 0;
    uint32_t clothIndexGE1 = 0, lastClothSelected = 0;
    uint32_t singleCopy = 0, multiCopy = 0, wLaneDiffers = 0, numClothsZeroLive = 0;
};

// ---------------------------------------------------------------------------
// The mirror: particlesystem.cu:4331-4425, transcribed.
// ---------------------------------------------------------------------------
static void refAverageVerts(SysRec& sys, uint32_t gridX, std::vector<F4>& out,
                            std::set<uint32_t>& written, Coverage& cov)
{
    for (uint32_t b = 0; b < uint32_t(sys.cloths.size()); ++b)
    {
        if (b >= sys.numClothBuffers) continue;          // the cloth-buffer guard
        ClothRec& c = sys.cloths[b];

        const uint32_t offset = sys.runsum[c.particleBufferIndex];
        const uint32_t nbParticles = sys.nbActivePerBuffer[c.particleBufferIndex];
        const uint32_t nbSprings = c.numSprings;
        const uint32_t springOffset = nbSprings * 2u;

        for (uint32_t gti = 0; gti < gridX * BLOCK; ++gti)
        {
            if (!(gti < nbParticles && nbSprings > 0u)) continue;

            const uint32_t index = gti + offset;
            if (offset != 0u) ++cov.offsetNonZeroProbes;
            const uint32_t sortedInd = sys.u2s[index];
            const F4 pos = sys.pos[sortedInd];

            if (!(pos.w > 0.0f)) { ++cov.skippedFixed; continue; }

            const F4 deltaP = out[sortedInd];

            const uint32_t startInd = (gti == 0u) ? 0u : c.copies[COPIES_LEAD_IN + gti - 1u];
            const uint32_t endInd = c.copies[COPIES_LEAD_IN + gti];
            const uint32_t numCopies = endInd - startInd;

            if (numCopies == 0u) { ++cov.zeroCopyLive; continue; }
            if (gti == 0u && offset != 0u) ++cov.sentinelZero;
            if (numCopies == 1u) ++cov.singleCopy;
            if (numCopies >= 3u) ++cov.multiCopy;

            const F4 f0 = c.remap[springOffset + startInd];
            float d[4] = { f0.x - pos.x, f0.y - pos.y, f0.z - pos.z, f0.w - pos.w };
            for (uint32_t j = startInd + 1u; j < endInd; ++j)
            {
                const F4 fj = c.remap[springOffset + j];
                d[0] += fj.x - pos.x; d[1] += fj.y - pos.y;
                d[2] += fj.z - pos.z; d[3] += fj.w - pos.w;
            }
            if (d[3] != 0.0f) ++cov.wLaneDiffers;

            const uint32_t numCloths = c.numCloths;
            if (numCloths == 0u) ++cov.numClothsZeroLive;
            const uint32_t clothIndex = refBinarySearch(c.startIndices.data(), numCloths, index);

            bool useBlend = false;
            if (clothIndex < numCloths)
            {
                const uint32_t sv = c.cloths[clothIndex].startVertexIndex;
                const uint32_t nv = c.cloths[clothIndex].numVertices;
                useBlend = ((sv + offset) + nv) > index;
                if (clothIndex >= 1u) ++cov.clothIndexGE1;
                if (clothIndex + 1u == numCloths) ++cov.lastClothSelected;
                if (index < sv + offset) ++cov.belowFirstStart;
                if (index == (sv + offset) + nv) ++cov.pastLastCloth;
                for (uint32_t q = 0; q < numCloths; ++q)
                    if (c.startIndices[q] == index && q >= 1u) ++cov.exactKeyHit;
            }

            float scale;
            if (useBlend) { scale = c.cloths[clothIndex].clothBlendScale; ++cov.blendBranch; }
            else          { scale = 1.0f / float(numCopies);              ++cov.fallbackBranch; }

            out[sortedInd] = { deltaP.x + d[0] * scale, deltaP.y + d[1] * scale,
                               deltaP.z + d[2] * scale, deltaP.w + d[3] * scale };
            ++cov.stores;
            written.insert(sortedInd);
        }
    }
}

PXG_TEST(psAverageVerts)
{
    PxgSlangComputeDevice* dev = testDevice();

    SysRec sysA, sysD, sysB;
    SysRec* pool[3] = { &sysA, &sysD, &sysB };

    // ---- the copy-count vectors: non-uniform, with interior plateaus --------
    auto copyCounts = [](uint32_t n, uint32_t salt)
    {
        std::vector<uint32_t> v(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            const uint32_t h = hashu(i, salt) % 5u;          // 0..4
            v[i] = h;
        }
        if (n > 0) v[0] = 3u;                                // A1: gti 0 must have copies
        if (n > 6) { v[3] = 0u; v[5] = 0u; v[6] = 0u; }       // A11: interior plateaus
        if (n > 12) { v[9] = 1u; v[11] = 4u; v[12] = 2u; }    // A9: 1 and >=3 both present
        if (n > 25) { v[24] = 3u; v[25] = 1u; }               // A10(a): live past the last cloth
        return v;
    };

    auto makeCloth = [&](uint32_t particleBufferIndex, uint32_t numSprings,
                         const std::vector<uint32_t>& startIdx,
                         const std::vector<PxParticleCloth>& clothRecs,
                         uint32_t reportedNumCloths, uint32_t nbParticles, uint32_t salt)
    {
        ClothRec c;
        c.particleBufferIndex = particleBufferIndex;
        c.numSprings = numSprings;
        c.numCloths = reportedNumCloths;
        c.nbParticles = nbParticles;

        // startIndices: live entries then ONE SMALL pad (0) -- see the header.
        c.startIndices = startIdx;
        c.startIndices.push_back(0u);

        // cloths: live records, then a POISON record that would PASS the range
        // test if it were read (A13), then a filler so a 32-stride read of the
        // last live index stays mapped (A14).
        c.cloths = clothRecs;
        PxParticleCloth poison; std::memset(&poison, 0, sizeof(poison));
        poison.startVertexIndex = 0u; poison.numVertices = 0xFFFFu; poison.clothBlendScale = 7.5f;
        c.cloths.push_back(poison);
        PxParticleCloth filler; std::memset(&filler, 0, sizeof(filler));
        filler.startVertexIndex = 999u; filler.numVertices = 1u; filler.clothBlendScale = 9.25f;
        c.cloths.push_back(filler);

        // accumulatedCopies: lead-in (small, TERMINATING) + inclusive runsum + tail
        const std::vector<uint32_t> counts = copyCounts(nbParticles, salt);
        c.copies.assign(COPIES_LEAD_IN, 2u);
        uint32_t total = 0;
        for (uint32_t i = 0; i < nbParticles; ++i) { total += counts[i]; c.copies.push_back(total); }
        c.totalCopies = total;
        for (uint32_t i = 0; i < COPIES_TAIL; ++i)
            c.copies.push_back(total + PxMin(i + 1u, 8u));

        // remap: the two endpoint planes (a disjoint numeric range, A5), then the
        // copy region, then 8 poison slots (A9b).
        const uint32_t springOffset = numSprings * 2u;
        c.remap.resize(size_t(springOffset) + total + 8u);
        for (uint32_t j = 0; j < springOffset; ++j)
            c.remap[j] = { -1000.0f - float(j), -2000.0f - float(j),
                           -3000.0f - float(j), -4000.0f - float(j) };
        for (uint32_t k = 0; k < total; ++k)
            c.remap[springOffset + k] = { 100.0f + float(k), 200.0f - 0.5f * float(k),
                                          300.0f + 0.25f * float(k), 0.0f };  // .w filled later
        float pv; const uint32_t pu = POISON_U32; std::memcpy(&pv, &pu, 4);
        for (uint32_t k = 0; k < 8u; ++k)
            c.remap[size_t(springOffset) + total + k] = { pv, pv, pv, pv };
        return c;
    };

    auto cloth = [](uint32_t startVertexIndex, uint32_t numVertices, float blendScale)
    {
        PxParticleCloth c; std::memset(&c, 0, sizeof(c));
        c.startVertexIndex = startVertexIndex; c.numVertices = numVertices;
        c.clothBlendScale = blendScale;
        return c;
    };

    // ===================== system A ========================================
    sysA.active = true;
    sysA.numParticles = 1200u;
    sysA.nbActivePerBuffer = { 24u, 40u, 1100u };
    sysA.runsum = { 0u, 24u, 64u, 1164u };
    sysA.numClothBuffers = 4u;
    // A.cb0 -> particle buffer 1 (offset 24).  Three cloths whose LOCAL starts
    // are 0/8/16 while the search key is global: the A3 index-space defect makes
    // every particle here resolve to the LAST cloth.  Particles 24..39 fall past
    // every cloth span and take the 1/numCopies branch.
    sysA.cloths.push_back(makeCloth(1u, 7u, { 0u, 8u, 16u },
        { cloth(0u, 8u, 0.25f), cloth(8u, 8u, 0.5f), cloth(16u, 8u, 0.125f) }, 3u, 40u, 11u));
    // A.cb1 -> particle buffer 0 (offset 0).  startIndices[0] > 0, so keys 0..3
    // exercise binarySearch's clamp-to-0; the spans OVERLAP so a lower_bound
    // mutant still passes the range test and dies on the SCALE instead.
    sysA.cloths.push_back(makeCloth(0u, 5u, { 4u, 12u },
        { cloth(4u, 12u, 0.2f), cloth(12u, 8u, 0.4f) }, 2u, 24u, 22u));
    // A.cb2 -> particle buffer 2 (offset 64).  1100 particles => gridX = 2.
    sysA.cloths.push_back(makeCloth(2u, 1100u, { 0u },
        { cloth(0u, 900u, 0.0625f) }, 1u, 1100u, 33u));
    // A.cb3 reports mNumSprings == 0 over REAL backing sized for four springs,
    // so `nbSprings > 0` has something to stop rather than being tested by an
    // empty array by accident.  A port that drops it reads the endpoint plane as
    // if it were the copy region (springOffset becomes 0) and writes.
    sysA.cloths.push_back(makeCloth(0u, 4u, { 0u },
        { cloth(0u, 24u, 0.75f) }, 1u, 24u, 44u));
    sysA.cloths.back().numSprings = 0u;   // the RECORD lies; the backing is real

    // ===================== the decoy =======================================
    sysD.active = false;
    sysD.numParticles = 200u;
    sysD.nbActivePerBuffer = { 30u, 20u, 25u };
    sysD.runsum = { 0u, 30u, 50u, 75u };
    sysD.numClothBuffers = 4u;                 // gridY = 4, and gridZ = 2
    for (uint32_t b = 0; b < 4u; ++b)
        sysD.cloths.push_back(makeCloth(b % 3u, 5u + b, { 0u, 4u },
            { cloth(0u, 4u, 0.3f), cloth(4u, 6u, 0.6f) }, 2u,
            sysD.nbActivePerBuffer[b % 3u], 100u + b));

    // ===================== system B ========================================
    sysB.active = true;
    sysB.numParticles = 200u;
    sysB.nbActivePerBuffer = { 30u, 20u, 25u };
    sysB.runsum = { 0u, 30u, 50u, 75u };
    sysB.numClothBuffers = 1u;
    // B.cb0 reports mNumCloths == 0 over a ZERO-length live array, but the
    // poison record behind it would pass the range test -- so hoisting the cloth
    // load above the bound (the idiomatic BDA spelling) is caught (A13).
    sysB.cloths.push_back(makeCloth(0u, 3u, {}, {}, 0u, 30u, 55u));
    for (uint32_t b = 1; b < 4u; ++b)          // padding past mNumClothBuffers = 1
        sysB.cloths.push_back(makeCloth(b % 3u, 6u + b, { 0u, 5u },
            { cloth(0u, 5u, 0.35f), cloth(5u, 7u, 0.7f) }, 2u,
            sysB.nbActivePerBuffer[b % 3u], 200u + b));

    // ---- sorted arrays: a real permutation with no fixed point, not an
    // involution; both mappings populated so a @56 mutant reads plausible data.
    for (SysRec* s : pool)
    {
        const uint32_t N = s->numParticles;
        // [0, N) is a real permutation; the tail past N exists only so a dropped
        // bound resolves to a LIVE slot (0) and is observable rather than fatal.
        s->u2s.assign(size_t(N) + MAPPING_TAIL, 0u);
        s->s2u.assign(size_t(N) + MAPPING_TAIL, 0u);
        for (uint32_t i = 0; i < N; ++i) s->u2s[i] = (i * 37u + 11u) % N;
        for (uint32_t i = 0; i < N; ++i) s->s2u[s->u2s[i]] = i;
        s->pos.resize(N); s->accum0.resize(N);
        const float salt = s->active ? 3.0f : 91.0f;
        for (uint32_t i = 0; i < N; ++i)
        {
            s->pos[i] = { 0.25f * float(i) + salt, -0.5f * float(i), 0.125f * float(i),
                          1.0f + 0.01f * float(i) };
            s->accum0[i] = { 3.0f + 0.5f * float(i), -7.0f - 0.25f * float(i),
                             11.0f + 0.125f * float(i), 0.75f + 0.001f * float(i) };
        }
    }
    // FIXED particles (w == 0), chosen by (cloth buffer, gti) so at least one is
    // the first particle of a cloth's range and at least one has a live run.
    struct FixedAt { SysRec* s; uint32_t b, gti; };
    const FixedAt fixedAt[] = { { &sysA, 0u, 2u }, { &sysA, 0u, 8u }, { &sysA, 1u, 0u },
                                { &sysA, 2u, 17u }, { &sysB, 0u, 4u }, { &sysB, 0u, 11u } };
    for (const FixedAt& f : fixedAt)
    {
        const ClothRec& c = f.s->cloths[f.b];
        const uint32_t idx = f.gti + f.s->runsum[c.particleBufferIndex];
        f.s->pos[f.s->u2s[idx]].w = 0.0f;
    }

    // ---- the copy .w lane: EQUAL to the owner's pos.w for most copies (the
    // live pipeline's invariant, under which diff.w cancels to 0 and a float3
    // port is accidentally right), deliberately BROKEN for a minority.
    uint32_t brokenW = 0;
    for (SysRec* s : pool)
        for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
        {
            ClothRec& c = s->cloths[b];
            const uint32_t offset = s->runsum[c.particleBufferIndex];
            const uint32_t nb = s->nbActivePerBuffer[c.particleBufferIndex];
            const uint32_t springOffset = c.numSprings * 2u;
            for (uint32_t gti = 0; gti < nb; ++gti)
            {
                const uint32_t st = (gti == 0u) ? 0u : c.copies[COPIES_LEAD_IN + gti - 1u];
                const uint32_t en = c.copies[COPIES_LEAD_IN + gti];
                const uint32_t idx = gti + offset;
                if (idx >= s->numParticles) continue;
                const float w = s->pos[s->u2s[idx]].w;
                for (uint32_t j = st; j < en; ++j)
                {
                    const bool breakIt = ((gti % 7u) == 4u);
                    c.remap[springOffset + j].w = breakIt ? (w + 0.5f + 0.25f * float(j)) : w;
                    if (breakIt) ++brokenW;
                }
            }
        }
    PXG_CHECK_MSG(brokenW >= 8u, "the .w invariant must be broken on some copies (%u)", brokenW);

    // ---- grid ------------------------------------------------------------
    uint32_t maxNbParticles = 0, maxClothBuffers = 0;
    for (SysRec* s : pool)
    {
        maxClothBuffers = PxMax(maxClothBuffers, uint32_t(s->cloths.size()));
        for (uint32_t n : s->nbActivePerBuffer) maxNbParticles = PxMax(maxNbParticles, n);
    }
    const uint32_t gridX = (maxNbParticles + BLOCK - 1u) / BLOCK;
    const uint32_t ACTIVE_COUNT = 2u;
    // padded past the dispatched extent so a Y/Z swap reads a live decoy index
    // rather than running off the allocation
    const std::vector<uint32_t> active = { 2u, 0u, 1u, 1u, 1u, 1u, 1u, 1u };

    PXG_CHECK_MSG(maxClothBuffers == CLOTH_ARRAY_SLOTS, "the Y extent must be %u (got %u)",
                  CLOTH_ARRAY_SLOTS, maxClothBuffers);
    PXG_CHECK_MSG(gridX > 1u, "the X grid must span more than one block (got %u)", gridX);
    PXG_CHECK_MSG(maxClothBuffers != ACTIVE_COUNT, "the grid must NOT be square (%u vs %u)",
                  maxClothBuffers, ACTIVE_COUNT);
    PXG_CHECK_MSG(active.size() >= size_t(maxClothBuffers),
                  "the active list must survive being indexed by the Y extent");
    for (SysRec* s : pool)
        for (uint32_t i = 0; i < s->numParticles; ++i)
        {
            PXG_CHECK_MSG(s->u2s[i] != i, "the mapping must have no fixed point (at %u)", i);
            PXG_CHECK_MSG(s->u2s[s->u2s[i]] != i, "the mapping must not be an involution (at %u)", i);
        }

    for (uint32_t i = 0; i < 3u; ++i)
        pool[i]->active = (i == 2u || i == 0u);
    for (SysRec* s : pool)
        for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
            s->cloths[b].live = s->active && (b < s->numClothBuffers);

    // ---- one round = upload, dispatch, compare ---------------------------
    auto runRound = [&](float invDt, std::vector<std::vector<F4>>& outImages, Coverage& cov) -> size_t
    {
        for (SysRec* s : pool)
        {
            s->bPos = makeDev(dev, s->pos);
            s->bAccum = makeDev(dev, s->accum0);
            s->bU2S = makeDev(dev, s->u2s);
            s->bS2U = makeDev(dev, s->s2u);
            s->bRunsum = makeDev(dev, s->runsum);

            std::vector<PxgParticleSimBuffer> simBufs(s->nbActivePerBuffer.size());
            for (uint32_t i = 0; i < uint32_t(simBufs.size()); ++i)
            {
                std::memset(&simBufs[i], 0xEE, sizeof(PxgParticleSimBuffer));
                simBufs[i].mNumActiveParticles = s->nbActivePerBuffer[i];
            }
            s->bSimBufs = makeDev(dev, simBufs);

            std::vector<PxgParticleClothSimBuffer> arr(s->cloths.size());
            for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
            {
                ClothRec& c = s->cloths[b];
                c.bStart = makeDev(dev, c.startIndices);
                c.bCloths = makeDev(dev, c.cloths);
                c.bCopies = makeDev(dev, c.copies);
                c.bRemap = makeDev(dev, c.remap);

                PxgParticleClothSimBuffer cb;
                std::memset(&cb, 0xEE, sizeof(cb));
                cb.mSortedClothStartIndices = reinterpret_cast<PxU32*>(size_t(c.bStart.addr));
                cb.mCloths = reinterpret_cast<PxParticleCloth*>(size_t(c.bCloths.addr));
                // the OFFSET pointer: the lead-in word sits BEFORE it
                cb.mAccumulatedCopiesPerParticles = reinterpret_cast<PxU32*>(
                    size_t(c.bCopies.addr) + size_t(COPIES_LEAD_IN) * sizeof(PxU32));
                cb.mRemapPositions = reinterpret_cast<float4*>(size_t(c.bRemap.addr));
                cb.mParticleBufferIndex = c.particleBufferIndex;
                cb.mNumSprings = c.numSprings;
                cb.mNumCloths = c.numCloths;
                arr[b] = cb;
            }
            s->bClothArr = makeDev(dev, arr);
        }

        DevBuf bActive = makeDev(dev, active);

        std::vector<PxgParticleSystem> descArr;
        for (SysRec* s : pool)
        {
            PxgParticleSystem ps;
            std::memset(&ps, 0xEE, sizeof(ps));
            ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s->bPos.addr));
            ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(s->bAccum.addr));
            ps.mUnsortedToSortedMapping = reinterpret_cast<PxU32*>(size_t(s->bU2S.addr));
            ps.mSortedToUnsortedMapping = reinterpret_cast<PxU32*>(size_t(s->bS2U.addr));
            ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(s->bRunsum.addr));
            ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(s->bSimBufs.addr));
            ps.mClothSimBuffers = reinterpret_cast<PxgParticleClothSimBuffer*>(size_t(s->bClothArr.addr));
            ps.mNumClothBuffers = s->numClothBuffers;
            descArr.push_back(ps);
        }
        DevBuf bDesc = makeDev(dev, descArr);

        {
            Kernel k("npPsSprings.spv", "ps_averageVertsLaunch", {}, sizeof(AvPush));
            AvPush push; std::memset(&push, 0, sizeof(push));
            push.particleSystems = bDesc.addr;
            push.activeParticleSystems = bActive.addr;
            push.invDt = invDt;
            k.dispatch(gridX, {}, push, maxClothBuffers, ACTIVE_COUNT);
            dev->flush();
        }

        // the reference
        std::vector<std::vector<F4>> ref;
        std::set<uint32_t> writtenA, writtenB;
        for (uint32_t i = 0; i < 3u; ++i)
        {
            SysRec* s = pool[i];
            std::vector<F4> img = s->accum0;
            if (s->active)
            {
                std::set<uint32_t>& w = (s == &sysA) ? writtenA : writtenB;
                const size_t before = w.size();
                size_t storesBefore = cov.stores;
                refAverageVerts(*s, gridX, img, w, cov);
                // the store is non-atomic: no sorted slot may be written twice
                PXG_CHECK_MSG(w.size() - before == cov.stores - storesBefore,
                              "every store must land on a distinct sorted slot (%zu vs %zu)",
                              w.size() - before, cov.stores - storesBefore);
            }
            ref.push_back(img);
        }

        size_t mm = 0;
        outImages.clear();
        for (uint32_t i = 0; i < 3u; ++i)
        {
            SysRec* s = pool[i];
            const auto got = downloadBuffer<F4>(dev, s->bAccum.h, s->numParticles);
            outImages.push_back(got);
            for (uint32_t j = 0; j < s->numParticles; ++j)
            {
                const float* r = &ref[i][j].x; const float* g = &got[j].x;
                for (int k2 = 0; k2 < 4; ++k2)
                    if (f2u(r[k2]) != f2u(g[k2]))
                    {
                        if (mm < 12)
                            std::printf("  sys %u slot %u.%d: ref 0x%08x (%g) got 0x%08x (%g)\n",
                                        i, j, k2, f2u(r[k2]), double(r[k2]), f2u(g[k2]), double(g[k2]));
                        ++mm;
                    }
            }
            // nothing else may be touched: every read-only array back verbatim
            const auto gp = downloadBuffer<F4>(dev, s->bPos.h, s->numParticles);
            for (uint32_t j = 0; j < s->numParticles; ++j)
                if (f2u(gp[j].x) != f2u(s->pos[j].x) || f2u(gp[j].w) != f2u(s->pos[j].w)) ++mm;
            for (ClothRec& c : s->cloths)
            {
                const auto gr = downloadBuffer<F4>(dev, c.bRemap.h, c.remap.size());
                for (size_t j = 0; j < c.remap.size(); ++j)
                    if (f2u(gr[j].x) != f2u(c.remap[j].x) || f2u(gr[j].w) != f2u(c.remap[j].w)) ++mm;
                const auto gc = downloadBuffer<uint32_t>(dev, c.bCopies.h, c.copies.size());
                for (size_t j = 0; j < c.copies.size(); ++j) if (gc[j] != c.copies[j]) ++mm;
            }
        }

        for (SysRec* s : pool)
        {
            dev->destroyBuffer(s->bPos.h); dev->destroyBuffer(s->bAccum.h);
            dev->destroyBuffer(s->bU2S.h); dev->destroyBuffer(s->bS2U.h);
            dev->destroyBuffer(s->bRunsum.h); dev->destroyBuffer(s->bSimBufs.h);
            dev->destroyBuffer(s->bClothArr.h);
            for (ClothRec& c : s->cloths)
            {
                dev->destroyBuffer(c.bStart.h); dev->destroyBuffer(c.bCloths.h);
                dev->destroyBuffer(c.bCopies.h); dev->destroyBuffer(c.bRemap.h);
            }
        }
        dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
        return mm;
    };

    Coverage cov1, cov2;
    std::vector<std::vector<F4>> img1, img2;
    const size_t mm1 = runRound(60.0f, img1, cov1);
    // A7: invDt is a DEAD argument.  Two wildly different values must produce
    // bit-identical output; any use of it at all dies here.
    const size_t mm2 = runRound(-1.0e30f, img2, cov2);

    PXG_CHECK_MSG(mm1 == 0, "the PBD cloth vertex averaging is bit-exact (%zu mismatches)", mm1);
    PXG_CHECK_MSG(mm2 == 0, "the second round is bit-exact too (%zu mismatches)", mm2);

    size_t invMm = 0;
    for (size_t i = 0; i < img1.size(); ++i)
        for (size_t j = 0; j < img1[i].size(); ++j)
        {
            const float* a = &img1[i][j].x; const float* b = &img2[i][j].x;
            for (int k2 = 0; k2 < 4; ++k2) if (f2u(a[k2]) != f2u(b[k2])) ++invMm;
        }
    PXG_CHECK_MSG(invMm == 0, "invDt is a DEAD argument: the two rounds must agree (%zu)", invMm);

    // Every decision reached, counted rather than assumed.
    PXG_CHECK_MSG(cov1.stores > 800u, "the kernel must do real work (%u)", cov1.stores);
    PXG_CHECK_MSG(cov1.skippedFixed >= 3u, "fixed particles must be skipped (%u)", cov1.skippedFixed);
    PXG_CHECK_MSG(cov1.zeroCopyLive >= 3u, "empty copy runs must occur (%u)", cov1.zeroCopyLive);
    PXG_CHECK_MSG(cov1.sentinelZero >= 2u,
                  "gti == 0 in an offset > 0 buffer must occur (%u)", cov1.sentinelZero);
    PXG_CHECK_MSG(cov1.offsetNonZeroProbes >= 200u, "most probes must sit at offset > 0 (%u)",
                  cov1.offsetNonZeroProbes);
    PXG_CHECK_MSG(cov1.blendBranch >= 6u, "the blendScale branch must fire (%u)", cov1.blendBranch);
    PXG_CHECK_MSG(cov1.fallbackBranch >= 6u, "the 1/numCopies branch must fire (%u)",
                  cov1.fallbackBranch);
    PXG_CHECK_MSG(cov1.exactKeyHit >= 1u, "a key must land exactly on a startIndices entry (%u)",
                  cov1.exactKeyHit);
    PXG_CHECK_MSG(cov1.belowFirstStart >= 1u,
                  "a key must fall BELOW its cloth's start and still blend (%u)", cov1.belowFirstStart);
    PXG_CHECK_MSG(cov1.pastLastCloth >= 1u,
                  "a key must sit exactly on the strict > boundary (%u)", cov1.pastLastCloth);
    PXG_CHECK_MSG(cov1.clothIndexGE1 >= 4u, "cloth index >= 1 must occur (stride!) (%u)",
                  cov1.clothIndexGE1);
    PXG_CHECK_MSG(cov1.lastClothSelected >= 2u, "the last cloth must be selected (%u)",
                  cov1.lastClothSelected);
    PXG_CHECK_MSG(cov1.singleCopy >= 2u, "single-copy runs must occur (%u)", cov1.singleCopy);
    PXG_CHECK_MSG(cov1.multiCopy >= 4u, "multi-copy runs must occur (%u)", cov1.multiCopy);
    PXG_CHECK_MSG(cov1.wLaneDiffers >= 4u, "the .w lane must be observable (%u)", cov1.wLaneDiffers);
    PXG_CHECK_MSG(cov1.numClothsZeroLive >= 3u, "mNumCloths == 0 must be exercised (%u)",
                  cov1.numClothsZeroLive);
}

} // namespace
