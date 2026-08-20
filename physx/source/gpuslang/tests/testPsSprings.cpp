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

// testPsSprings.cpp -- P9.3a fixture for ps_initializeSpringsLaunch and
// ps_updateRemapVertsLaunch, ported as npPsSprings.slang.
//
// ORACLE: host transcription, carried by offsets GENERATED from the engine
// headers and pinned by the layout test below, whole-region 0xAB poison compare,
// and the sweep.  Both kernels are pure data movement, so the layout test is the
// load-bearing part -- there is no arithmetic for a numerical oracle to check.
//
// THE NEW THING HERE IS THE GRID.  Every particle kernel this port has mapped so
// far puts the particle system on gridDim.y.  The cloth chain is 3D: the system
// is on gridDim.Z and the CLOTH BUFFER is on gridDim.Y.  The fixture therefore
// dispatches with a genuine 3D grid and gives the two systems DIFFERENT cloth
// buffer counts, so a port that keeps the 2D convention reads the wrong system
// and touches only buffer 0 -- and dies.
//
// GEOMETRY:
//   pool[0] = A: 2 cloth buffers (7 and 5 springs), particle-buffer offsets 0 and 40
//   pool[1] = DECOY: never active, 3 cloth buffers, all outputs must stay poisoned
//   pool[2] = B: 1 cloth buffer (3 springs)
//   active = { 2, 0 }  -- not a permutation, so indexing the pool by blockIdx.z
//                         directly hits the decoy and dies
//   gridDim = (ceil(maxSprings*2 / 1024), 3, 2)
// The grid's Y extent is the ACROSS-system maximum (3, the decoy's count), so B's
// buffers 1 and 2 and A's buffer 2 are past their own mNumClothBuffers -- which is
// exactly what trap K6 guards, and their records must come back poisoned.
//
// Spring indices are deliberately NOT the identity and NOT sorted, the two
// endpoints of a spring differ, and mUnsortedToSortedMapping is a real
// permutation, so the two indirections of trap K3 cannot collapse into one.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxParticleBuffer.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
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

struct SpPush { uint64_t particleSystems, activeParticleSystems; };
static_assert(sizeof(SpPush) == 16, "kTwoPtr blob is 16 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;
static const uint32_t BLOCK = 1024u;

// ---------------------------------------------------------------------------
// Layout proof -- every constant npPsSprings.slang and pxgSlangParticleDesc
// hard-code for the cloth records, pinned against the real engine headers.
// ---------------------------------------------------------------------------
PXG_TEST(psSpringsLayout)
{
    PXG_CHECK_MSG(sizeof(PxgParticleClothSimBuffer) == 112,
                  "sizeof(PxgParticleClothSimBuffer) == 112 (got %zu)",
                  sizeof(PxgParticleClothSimBuffer));
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mAccumulatedSpringsPerPartitions) == 0);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mAccumulatedCopiesPerParticles) == 8);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mRemapOutput) == 16);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mOrderedSprings) == 24);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mTriangles) == 32);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mSortedClothStartIndices) == 40);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mCloths) == 48);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mRemapPositions) == 56);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mRemapVelocities) == 64);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mSpringLambda) == 72);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mInflatableLambda) == 80);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mParticleBufferIndex) == 88);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumSprings) == 92);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumPartitions) == 96);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumCloths) == 100);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumTriangles) == 104);

    PXG_CHECK_MSG(sizeof(PxParticleSpring) == 24, "sizeof(PxParticleSpring) == 24 (got %zu)",
                  sizeof(PxParticleSpring));
    PXG_CHECK(offsetof(PxParticleSpring, ind0) == 0);
    PXG_CHECK(offsetof(PxParticleSpring, ind1) == 4);
    PXG_CHECK(offsetof(PxParticleSpring, length) == 8);
    PXG_CHECK(offsetof(PxParticleSpring, stiffness) == 12);
    PXG_CHECK(offsetof(PxParticleSpring, damping) == 16);

    // Seven 4-byte members and align 4, so the array stride is 28 and NOT
    // rounded up to 32 -- the one place a plausible assumption is wrong.
    PXG_CHECK_MSG(sizeof(PxParticleCloth) == 28, "sizeof(PxParticleCloth) == 28 (got %zu)",
                  sizeof(PxParticleCloth));
    PXG_CHECK(offsetof(PxParticleCloth, startVertexIndex) == 0);
    PXG_CHECK(offsetof(PxParticleCloth, clothBlendScale) == 8);
    PXG_CHECK(offsetof(PxParticleCloth, numTriangles) == 24);

    // the descriptor fields the two kernels walk
    PXG_CHECK(offsetof(PxgParticleSystem, mClothSimBuffers) == 432);
    PXG_CHECK(offsetof(PxgParticleSystem, mNumClothBuffers) == 464);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedToSortedMapping) == 48);
}

struct ClothBuf
{
    uint32_t numSprings = 0;
    uint32_t particleBufferIndex = 0;
    std::vector<PxParticleSpring> springs;
    DevBuf bSprings, bLambda, bRemapPos, bRemapVel;
};

struct Sys
{
    uint32_t numClothBuffers = 0;
    std::vector<ClothBuf> cloths;
    std::vector<F4> pos, vel;
    std::vector<uint32_t> unsortedToSorted, runsum;
    DevBuf bPos, bVel, bU2S, bRunsum, bClothArr;
};

static uint32_t hashu(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

PXG_TEST(psSprings)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t NUM_PARTICLES = 96u;
    // Per-buffer output sizes, each with a poisoned tail past the live range.
    auto lambdaSlots = [](uint32_t n) { return n + 8u; };
    auto remapSlots = [](uint32_t n) { return n * 2u + 8u; };

    // Every system's cloth ARRAY is allocated at the across-system maximum, but
    // mNumClothBuffers reports only the live count.  The padding entries are REAL
    // records with real springs and their own poisoned outputs, so a mutant that
    // widens the per-system bound writes into them and dies.  The first sweep
    // found this: without the padding the mutant read past a 1-element array,
    // landed on whatever followed, and happened to early-out -- it SURVIVED.
    const uint32_t CLOTH_ARRAY_SLOTS = 3u;

    auto buildSys = [&](std::vector<std::pair<uint32_t, uint32_t>> springCountsAndBufIdx,
                        uint32_t salt)
    {
        Sys s;
        s.numClothBuffers = uint32_t(springCountsAndBufIdx.size());
        // pad the ARRAY (not the count) out to the allocated size
        while (springCountsAndBufIdx.size() < CLOTH_ARRAY_SLOTS)
            springCountsAndBufIdx.push_back({ 6u + uint32_t(springCountsAndBufIdx.size()), 0u });
        s.pos.resize(NUM_PARTICLES);
        s.vel.resize(NUM_PARTICLES);
        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
        {
            s.pos[i] = { float(salt) + float(i) * 0.25f, float(i) * -0.5f, float(i) * 0.125f,
                         1.0f + float(i) * 0.01f };
            s.vel[i] = { float(i) * 0.3f, float(salt) - float(i) * 0.2f, float(i) * 0.05f, 0.0f };
        }
        // a REAL permutation, so the sorted lookup cannot collapse to identity
        s.unsortedToSorted.resize(NUM_PARTICLES);
        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
            s.unsortedToSorted[i] = (i * 37u + 11u) % NUM_PARTICLES;
        // two particle buffers, at offsets 0 and 40
        s.runsum = { 0u, 40u, 80u };

        for (auto& pr : springCountsAndBufIdx)
        {
            ClothBuf c;
            c.numSprings = pr.first;
            c.particleBufferIndex = pr.second;
            c.springs.resize(c.numSprings);
            for (uint32_t k = 0; k < c.numSprings; ++k)
            {
                PxParticleSpring sp;
                std::memset(&sp, 0, sizeof(sp));
                // local indices, deliberately not the identity and not sorted,
                // and the two endpoints of a spring always differ
                // local indices: buffer offsets are 0 or 40 and there are 96
                // particles, so keep both endpoints inside [0, 39] and distinct.
                sp.ind0 = hashu(k, salt + 1u) % 20u;
                sp.ind1 = 20u + (hashu(k, salt + 2u) % 19u);
                sp.length = 0.5f + float(k) * 0.03f;
                sp.stiffness = 100.0f + float(k);
                sp.damping = 0.1f + float(k) * 0.01f;
                c.springs[k] = sp;
            }
            s.cloths.push_back(c);
        }
        return s;
    };

    // A: two cloth buffers with DIFFERENT spring counts and DIFFERENT particle
    // buffers.  B: one.  DECOY: three, so the grid's Y extent (3) exceeds both
    // active systems' counts and trap K6 has something to guard.
    // The third buffer is deliberately BIG: with only small buffers the grid's X
    // extent is 1 block and every grid-decomposition mutant (dropping the
    // blockIdx.x term, halving the block dim) is an identity.  1100 springs give
    // ceil(2200/1024) = 3 blocks for the remap and 2 for the init.
    Sys sysA = buildSys({ { 7u, 0u }, { 5u, 1u }, { 1100u, 0u } }, 100u);
    Sys sysD = buildSys({ { 4u, 0u }, { 4u, 0u }, { 4u, 0u } }, 200u);
    Sys sysB = buildSys({ { 3u, 1u } }, 300u);

    Sys* order[3] = { &sysA, &sysD, &sysB };
    uint32_t maxSprings = 0, maxClothBuffers = 0;
    for (Sys* s : order)
    {
        maxClothBuffers = PxMax(maxClothBuffers, uint32_t(s->cloths.size()));
        for (auto& c : s->cloths) maxSprings = PxMax(maxSprings, c.numSprings);
    }
    PXG_CHECK_MSG(maxClothBuffers == CLOTH_ARRAY_SLOTS,
                  "the Y extent must be %u (got %u)", CLOTH_ARRAY_SLOTS, maxClothBuffers);
    PXG_CHECK_MSG(sysB.numClothBuffers < CLOTH_ARRAY_SLOTS,
                  "B must have real cloth records past its own bound");
    PXG_CHECK_MSG((maxSprings * 2u + BLOCK - 1u) / BLOCK > 1u,
                  "the remap grid must span more than one block (springs=%u)", maxSprings);

    for (Sys* s : order)
    {
        s->bPos = makeDev(dev, s->pos);
        s->bVel = makeDev(dev, s->vel);
        s->bU2S = makeDev(dev, s->unsortedToSorted);
        s->bRunsum = makeDev(dev, s->runsum);

        std::vector<PxgParticleClothSimBuffer> arr(s->cloths.size());
        for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
        {
            ClothBuf& c = s->cloths[b];
            c.bSprings = makeDev(dev, c.springs);
            std::vector<uint32_t> lam(lambdaSlots(c.numSprings), POISON_U32);
            std::vector<uint32_t> rp(size_t(remapSlots(c.numSprings)) * 4u, POISON_U32);
            std::vector<uint32_t> rv(size_t(remapSlots(c.numSprings)) * 4u, POISON_U32);
            c.bLambda = makeDev(dev, lam);
            c.bRemapPos = makeDev(dev, rp);
            c.bRemapVel = makeDev(dev, rv);

            PxgParticleClothSimBuffer cb;
            std::memset(&cb, 0xEE, sizeof(cb));
            cb.mOrderedSprings = reinterpret_cast<PxParticleSpring*>(size_t(c.bSprings.addr));
            cb.mSpringLambda = reinterpret_cast<PxReal*>(size_t(c.bLambda.addr));
            cb.mRemapPositions = reinterpret_cast<float4*>(size_t(c.bRemapPos.addr));
            cb.mRemapVelocities = reinterpret_cast<float4*>(size_t(c.bRemapVel.addr));
            cb.mParticleBufferIndex = c.particleBufferIndex;
            cb.mNumSprings = c.numSprings;
            arr[b] = cb;
        }
        s->bClothArr = makeDev(dev, arr);
    }

    std::vector<uint32_t> active = { 2u, 0u };
    DevBuf bActive = makeDev(dev, active);

    std::vector<PxgParticleSystem> descArr;
    for (Sys* s : order)
    {
        PxgParticleSystem ps;
        std::memset(&ps, 0xEE, sizeof(ps));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s->bPos.addr));
        ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(s->bVel.addr));
        ps.mUnsortedToSortedMapping = reinterpret_cast<PxU32*>(size_t(s->bU2S.addr));
        ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(s->bRunsum.addr));
        ps.mClothSimBuffers = reinterpret_cast<PxgParticleClothSimBuffer*>(size_t(s->bClothArr.addr));
        ps.mNumClothBuffers = s->numClothBuffers;
        descArr.push_back(ps);
    }
    DevBuf bDesc = makeDev(dev, descArr);

    SpPush push = { bDesc.addr, bActive.addr };

    // The real grids, both 3D: X over the work, Y over the cloth buffers,
    // Z over the active systems.
    const uint32_t gridInit = (maxSprings + BLOCK - 1u) / BLOCK;
    const uint32_t gridRemap = (maxSprings * 2u + BLOCK - 1u) / BLOCK;
    {
        Kernel k("npPsSprings.spv", "ps_initializeSpringsLaunch", {}, sizeof(SpPush));
        k.dispatch(gridInit, {}, push, maxClothBuffers, uint32_t(active.size()));
        dev->flush();
    }
    {
        Kernel k("npPsSprings.spv", "ps_updateRemapVertsLaunch", {}, sizeof(SpPush));
        k.dispatch(gridRemap, {}, push, maxClothBuffers, uint32_t(active.size()));
        dev->flush();
    }

    size_t mm = 0;
    for (Sys* s : order)
    {
        const bool isActive = (s != &sysD);
        for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
        {
            ClothBuf& c = s->cloths[b];
            // a record past this system's own mNumClothBuffers must stay poisoned
            const bool live = isActive && (b < s->numClothBuffers);

            std::vector<uint32_t> refLam(lambdaSlots(c.numSprings), POISON_U32);
            std::vector<uint32_t> refRp(size_t(remapSlots(c.numSprings)) * 4u, POISON_U32);
            std::vector<uint32_t> refRv(size_t(remapSlots(c.numSprings)) * 4u, POISON_U32);

            if (live)
            {
                for (uint32_t k = 0; k < c.numSprings; ++k)
                    refLam[k] = f2u(0.0f);

                const uint32_t offset = s->runsum[c.particleBufferIndex];
                for (uint32_t gti = 0; gti < c.numSprings * 2u; ++gti)
                {
                    const uint32_t workIndex = gti / 2u;
                    const uint32_t index = gti & 1u;
                    const PxParticleSpring& sp = c.springs[workIndex];
                    const uint32_t thisInd = (index == 0u) ? (sp.ind0 + offset) : (sp.ind1 + offset);
                    const uint32_t sortedInd = s->unsortedToSorted[thisInd];
                    const uint32_t dst = workIndex + c.numSprings * index;
                    const F4 p = s->pos[sortedInd];
                    const F4 v = s->vel[sortedInd];
                    refRp[dst * 4u + 0u] = f2u(p.x); refRp[dst * 4u + 1u] = f2u(p.y);
                    refRp[dst * 4u + 2u] = f2u(p.z); refRp[dst * 4u + 3u] = f2u(p.w);
                    refRv[dst * 4u + 0u] = f2u(v.x); refRv[dst * 4u + 1u] = f2u(v.y);
                    refRv[dst * 4u + 2u] = f2u(v.z); refRv[dst * 4u + 3u] = f2u(v.w);
                }
            }

            const auto gLam = downloadBuffer<uint32_t>(dev, c.bLambda.h, lambdaSlots(c.numSprings));
            const auto gRp = downloadBuffer<uint32_t>(dev, c.bRemapPos.h, remapSlots(c.numSprings) * 4u);
            const auto gRv = downloadBuffer<uint32_t>(dev, c.bRemapVel.h, remapSlots(c.numSprings) * 4u);

            auto cmp = [&](const std::vector<uint32_t>& r, const std::vector<uint32_t>& g,
                           const char* what)
            {
                for (size_t w = 0; w < r.size(); ++w)
                    if (r[w] != g[w])
                    {
                        if (mm < 12)
                            std::printf("  sys %p buf %u %s MM %zu: ref 0x%08x got 0x%08x\n",
                                        (void*)s, b, what, w, r[w], g[w]);
                        ++mm;
                    }
            };
            cmp(refLam, gLam, "lambda");
            cmp(refRp, gRp, "remapPos");
            cmp(refRv, gRv, "remapVel");
        }
    }

    PXG_CHECK_MSG(mm == 0, "the PBD spring gather pair is bit-exact (%zu mismatches)", mm);

    for (Sys* s : order)
    {
        dev->destroyBuffer(s->bPos.h); dev->destroyBuffer(s->bVel.h);
        dev->destroyBuffer(s->bU2S.h); dev->destroyBuffer(s->bRunsum.h);
        dev->destroyBuffer(s->bClothArr.h);
        for (auto& c : s->cloths)
        {
            dev->destroyBuffer(c.bSprings.h); dev->destroyBuffer(c.bLambda.h);
            dev->destroyBuffer(c.bRemapPos.h); dev->destroyBuffer(c.bRemapVel.h);
        }
    }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

} // namespace
