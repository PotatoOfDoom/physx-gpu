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

// testPsDiffuseBuffers.cpp -- P9.13c, the fixture for the three per-buffer
// bookkeeping kernels of the diffuse chain.
//
// ---------------------------------------------------------------------------
// THE ARRANGEMENT
// ---------------------------------------------------------------------------
// All three kernels are pure data movement over a small, fully host-visible
// structure: no float arithmetic, no engine primitive to call.  So the oracle is
// a host mirror of the CUDA's control flow, and what carries it is (1) every
// offset generated from the engine headers and pinned by static_assert below --
// so the P8.49a failure mode is closed by construction -- and (2) whole-region
// poison comparison, which makes an EXTRA write fail as loudly as a wrong one.
//
// The three run in the engine's own order within a frame:
//
//     UpdateUnsorted   gather each buffer's particles into the flat arrays
//     ...(create/compact, not this tranche, stage into slot [1])...
//     Copy             publish slot [1] into slot [0], clear [1]
//     Sum              add the [0]s into the system's single count
//
// and the fixture drives them in that order against one arena, so Copy's output
// is genuinely Sum's input rather than two independent checks.
//
// ---------------------------------------------------------------------------
// WHAT THE GEOMETRY IS BUILT TO DISCRIMINATE
// ---------------------------------------------------------------------------
//   * FIVE buffers with FIVE DIFFERENT counts and FIVE DIFFERENT capacities,
//     none equal to another and none equal to a buffer index, so a wrong index
//     or a swapped count/capacity lands on a different number;
//   * TWO particle systems with DIFFERENT buffer counts, driven through an
//     activeParticleSystems indirection that is NOT the identity -- so reading
//     the grid axis directly instead of through the active list is visible;
//   * a grid y extent larger than either system's buffer count, which is what
//     the engine does (mMaxDiffuseBuffersPerSystem is a maximum), so the U2
//     guard is exercised rather than assumed;
//   * one buffer whose staged count EXCEEDS its capacity, which is the only
//     case in which Copy's clamp does anything;
//   * one buffer with a PUBLISHED count of zero, which is the only case in which
//     UpdateUnsorted's start-index write is skipped (U4);
//   * one buffer with a NEGATIVE staged count, which is the only input that
//     separates the CUDA's signed clamp from an unsigned one (see A_SPECS).

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"     // PxgParticleSystem, PxgParticleDiffuseSimBuffer
#include "PxParticleBuffer.h"      // PxDiffuseParticleParams

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

// ---------------------------------------------------------------------------
// The ABI, generated from the engine headers and pinned here.  The shader's
// PxgParticleDiffuseSimBufferGpu carries the same numbers in its comments; if
// the engine struct ever moves, THIS is what fails, not a silent mis-read.
// ---------------------------------------------------------------------------
static_assert(sizeof(PxDiffuseParticleParams) == 40, "PxDiffuseParticleParams is 40 bytes");
static_assert(sizeof(PxgParticleDiffuseSimBuffer) == 88, "the diffuse sim-buffer stride");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mParams) == 0, "mParams is EMBEDDED at 0");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mDiffusePositions_LifeTime) == 40, "@40");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mDiffuseVelocities) == 48, "@48");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mMaxNumParticles) == 56, "@56");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mNumDiffuseParticles) == 64, "@64");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mNumActiveDiffuseParticles) == 72, "@72");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mStartIndex) == 80, "@80");
static_assert(offsetof(PxgParticleDiffuseSimBuffer, mFlags) == 84, "@84 -> 88");
// The system-side fields these three walk.
static_assert(offsetof(PxgParticleSystem, mDiffuseSimBuffers) == 432, "@432");
static_assert(offsetof(PxgParticleSystem, mNumDiffuseBuffers) == 440, "@440");
static_assert(offsetof(PxgParticleSystem, mNumDiffuseParticles) == 448, "@448");
static_assert(offsetof(PxgParticleSystem, mDiffusePosition_LifeTime) == 456, "@456");
static_assert(offsetof(PxgParticleSystem, mDiffuseVelocity) == 464, "@464");

struct F4 { float x, y, z, w; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

// psDiffuseBuffers.slang's PsDiffBufPush, byte for byte.
struct PsDiffBufBlob
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    uint32_t count;                 // @16
    uint32_t pad0;                  // -> 24
};
static_assert(sizeof(PsDiffBufBlob) == 24, "engine ABI blob size");

// ---------------------------------------------------------------------------
// The scene.
// ---------------------------------------------------------------------------
static const uint32_t NB_SYS_SLOTS = 5u;   // the descriptor array; only two live
static const uint32_t SYS_A = 3u;          // not 0 and not 1: a dropped indirection shows
static const uint32_t SYS_B = 1u;
static const uint32_t GRID_Y = 6u;         // > either system's buffer count (U2)
static const uint32_t FLAT_CAP = 4096u;
static const uint32_t POISON_U32 = 0xEEEEEEEEu;
static const float POISON_F = -9.0f;

struct BufSpec
{
    uint32_t capacity;   // mMaxNumParticles
    int32_t  staged;     // mNumDiffuseParticles[1] before Copy
    uint32_t published;  // mNumDiffuseParticles[0] before Copy -- what UpdateUnsorted reads
    uint32_t startSeed;  // mStartIndex before the run
};

// FIVE buffers for system A, two for system B.  Every capacity, staged count,
// published count and seed distinct from every other and from every index.
static const BufSpec A_SPECS[5] = {
    { 300u,  17,  40u, 7777u },   // ordinary
    { 250u,  33,   0u, 8888u },   // published 0 -> U4: the start index must NOT move
    { 180u, 400,  25u, 9999u },   // staged 400 > capacity 180 -> U5's clamp bites
    { 512u,   0,  61u, 6666u },   // staged 0 -> Copy publishes 0
    // A NEGATIVE staged count.  Not reachable from the engine -- the staging slot
    // is only ever atomicAdd'ed up from zero and reset by Copy -- but it is the
    // ONLY input that separates the CUDA's signed `PxMin(PxI32(cap), staged)`
    // from an unsigned min, which otherwise returns the capacity.  Seeded
    // directly here because the fixture owns the arena, and kept because a port
    // that is right only for inputs the engine happens to produce is right by
    // luck.  The prediction below follows the signed spelling, so a negative
    // count propagates into the Sum's unsigned total exactly as it would on the
    // device.
    { 96u,   -5,  52u, 3333u },
};
static const BufSpec B_SPECS[2] = {
    { 128u,  11,  13u, 5555u },
    { 640u,  90,  29u, 4444u },
};

struct Sys
{
    uint32_t nbBufs;
    const BufSpec* specs;
};

struct Rig
{
    std::vector<PxgBufferHandle> owned;
    DevBuf bDesc, bActive, bBufsA, bBufsB, bCountA, bCountB;
    DevBuf bSrcPosA, bSrcVelA, bSrcPosB, bSrcVelB;
    DevBuf bFlatPosA, bFlatVelA, bFlatPosB, bFlatVelB;
    DevBuf bNumA, bNumB;          // the systems' single counts
    DevBuf bPinnedA, bPinnedB;    // the per-buffer pinned mirrors
    DevBuf bTwoA, bTwoB;          // the per-buffer two-slot counts
    PsDiffBufBlob blob{};

    template <typename T>
    DevBuf mk(PxgSlangComputeDevice* dev, const std::vector<T>& d)
    {
        DevBuf b;
        b.h = createBufferFrom(dev, d);
        b.addr = dev->getBufferDeviceAddress(b.h);
        owned.push_back(b.h);
        return b;
    }
    void destroy(PxgSlangComputeDevice* dev)
    {
        for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
        owned.clear();
    }
};

// The per-buffer source arrays carry a value derived from (system, buffer,
// index) so a gather from the wrong buffer or the wrong slot is visible in the
// number itself rather than only in its position.
static float srcVal(uint32_t sys, uint32_t buf, uint32_t i, int comp)
{
    return float(1000000 * (sys + 1) + 10000 * (buf + 1) + 10 * int(i) + comp);
}

static void buildSide(PxgSlangComputeDevice* dev, Rig& R, const Sys& sys, uint32_t sysTag,
                      DevBuf& bBufs, DevBuf& bSrcPos, DevBuf& bSrcVel,
                      DevBuf& bFlatPos, DevBuf& bFlatVel, DevBuf& bNum,
                      DevBuf& bTwo, DevBuf& bPinned)
{
    const uint32_t nb = sys.nbBufs;

    // The two-slot counts, laid out [published, staged] per buffer.
    std::vector<int32_t> two(size_t(nb) * 2u);
    for (uint32_t b = 0; b < nb; ++b)
    {
        two[size_t(b) * 2 + 0] = int32_t(sys.specs[b].published);
        two[size_t(b) * 2 + 1] = sys.specs[b].staged;
    }
    bTwo = R.mk(dev, two);

    // The pinned host mirrors, poisoned so an unwritten one is visible.
    bPinned = R.mk(dev, std::vector<int32_t>(nb, int32_t(POISON_U32)));

    // The per-buffer source arrays, one contiguous block per side with each
    // buffer at a DIFFERENT base so a wrong base is visible.
    const uint32_t stride = 1024u;
    std::vector<F4> srcPos(size_t(nb) * stride), srcVel(size_t(nb) * stride);
    for (uint32_t b = 0; b < nb; ++b)
        for (uint32_t i = 0; i < stride; ++i)
        {
            const size_t at = size_t(b) * stride + i;
            srcPos[at] = F4{ srcVal(sysTag, b, i, 0), srcVal(sysTag, b, i, 1),
                             srcVal(sysTag, b, i, 2), srcVal(sysTag, b, i, 3) };
            srcVel[at] = F4{ srcVal(sysTag, b, i, 4), srcVal(sysTag, b, i, 5),
                             srcVal(sysTag, b, i, 6), srcVal(sysTag, b, i, 7) };
        }
    bSrcPos = R.mk(dev, srcPos);
    bSrcVel = R.mk(dev, srcVel);

    bFlatPos = R.mk(dev, std::vector<F4>(FLAT_CAP, F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    bFlatVel = R.mk(dev, std::vector<F4>(FLAT_CAP, F4{ POISON_F, POISON_F, POISON_F, POISON_F }));
    bNum = R.mk(dev, std::vector<uint32_t>(1u, POISON_U32));

    // The buffer records themselves.
    std::vector<PxgParticleDiffuseSimBuffer> bufs(nb);
    std::memset(bufs.data(), 0, bufs.size() * sizeof(PxgParticleDiffuseSimBuffer));
    for (uint32_t b = 0; b < nb; ++b)
    {
        PxDiffuseParticleParams pr;
        pr.threshold = 1.0f + float(b);   // never read by these three; a canary
        bufs[b].mParams = pr;
        bufs[b].mDiffusePositions_LifeTime =
            reinterpret_cast<float4*>(size_t(bSrcPos.addr + uint64_t(b) * stride * sizeof(F4)));
        bufs[b].mDiffuseVelocities =
            reinterpret_cast<float4*>(size_t(bSrcVel.addr + uint64_t(b) * stride * sizeof(F4)));
        bufs[b].mMaxNumParticles = sys.specs[b].capacity;
        bufs[b].mNumDiffuseParticles =
            reinterpret_cast<int*>(size_t(bTwo.addr + uint64_t(b) * 2u * sizeof(int32_t)));
        bufs[b].mNumActiveDiffuseParticles =
            reinterpret_cast<int*>(size_t(bPinned.addr + uint64_t(b) * sizeof(int32_t)));
        bufs[b].mStartIndex = sys.specs[b].startSeed;
        bufs[b].mFlags = 0xABCD0000u | b;
    }
    bBufs = R.mk(dev, bufs);
}

static bool buildRig(PxgSlangComputeDevice* dev, Rig& R)
{
    const Sys sysA{ 5u, A_SPECS };
    const Sys sysB{ 2u, B_SPECS };

    buildSide(dev, R, sysA, 0u, R.bBufsA, R.bSrcPosA, R.bSrcVelA,
              R.bFlatPosA, R.bFlatVelA, R.bNumA, R.bTwoA, R.bPinnedA);
    buildSide(dev, R, sysB, 1u, R.bBufsB, R.bSrcPosB, R.bSrcVelB,
              R.bFlatPosB, R.bFlatVelB, R.bNumB, R.bTwoB, R.bPinnedB);

    std::vector<PxgParticleSystem> descArr(NB_SYS_SLOTS);
    std::memset(descArr.data(), 0, descArr.size() * sizeof(PxgParticleSystem));
    descArr[SYS_A].mDiffuseSimBuffers =
        reinterpret_cast<PxgParticleDiffuseSimBuffer*>(size_t(R.bBufsA.addr));
    descArr[SYS_A].mNumDiffuseBuffers = 5u;
    descArr[SYS_A].mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bNumA.addr));
    descArr[SYS_A].mDiffusePosition_LifeTime = reinterpret_cast<float4*>(size_t(R.bFlatPosA.addr));
    descArr[SYS_A].mDiffuseVelocity = reinterpret_cast<float4*>(size_t(R.bFlatVelA.addr));

    descArr[SYS_B].mDiffuseSimBuffers =
        reinterpret_cast<PxgParticleDiffuseSimBuffer*>(size_t(R.bBufsB.addr));
    descArr[SYS_B].mNumDiffuseBuffers = 2u;
    descArr[SYS_B].mNumDiffuseParticles = reinterpret_cast<int*>(size_t(R.bNumB.addr));
    descArr[SYS_B].mDiffusePosition_LifeTime = reinterpret_cast<float4*>(size_t(R.bFlatPosB.addr));
    descArr[SYS_B].mDiffuseVelocity = reinterpret_cast<float4*>(size_t(R.bFlatVelB.addr));
    R.bDesc = R.mk(dev, descArr);

    // NOT the identity: grid slot 0 -> system 3, grid slot 1 -> system 1.
    R.bActive = R.mk(dev, std::vector<uint32_t>{ SYS_A, SYS_B });

    R.blob.particleSystems = R.bDesc.addr;
    R.blob.activeParticleSystems = R.bActive.addr;
    R.blob.count = 0x5A5A5A5Au;   // DEAD in both consumers; a canary for a reader
    return true;
}

// ---------------------------------------------------------------------------
// The host mirror: the CUDA's control flow, transcribed once.
// ---------------------------------------------------------------------------
struct Expect
{
    std::vector<F4> flatPos, flatVel;
    std::vector<uint32_t> startIndex;
    std::vector<int32_t> published, staged, pinned;
    uint32_t systemCount = 0u;
};

static Expect predict(const Sys& sys, uint32_t sysTag)
{
    const uint32_t nb = sys.nbBufs;
    Expect E;
    E.flatPos.assign(FLAT_CAP, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    E.flatVel.assign(FLAT_CAP, F4{ POISON_F, POISON_F, POISON_F, POISON_F });
    E.startIndex.resize(nb);
    E.published.resize(nb);
    E.staged.resize(nb);
    E.pinned.resize(nb);

    // ---- ps_updateUnsortedDiffuseArrayLaunch (reads slot [0], pre-Copy) ----
    for (uint32_t b = 0; b < nb; ++b)
    {
        uint32_t offset = 0;
        for (uint32_t j = 0; j < b; ++j) offset += sys.specs[j].published;   // U3
        const uint32_t n = sys.specs[b].published;
        // U4: with n == 0 the kernel returns before writing the start index.
        E.startIndex[b] = (n == 0u) ? sys.specs[b].startSeed : offset;
        for (uint32_t i = 0; i < n; ++i)
        {
            E.flatPos[offset + i] = F4{ srcVal(sysTag, b, i, 0), srcVal(sysTag, b, i, 1),
                                        srcVal(sysTag, b, i, 2), srcVal(sysTag, b, i, 3) };
            E.flatVel[offset + i] = F4{ srcVal(sysTag, b, i, 4), srcVal(sysTag, b, i, 5),
                                        srcVal(sysTag, b, i, 6), srcVal(sysTag, b, i, 7) };
        }
    }

    // ---- ps_diffuseParticleCopy ----
    for (uint32_t b = 0; b < nb; ++b)
    {
        const int32_t cap = int32_t(sys.specs[b].capacity);
        const int32_t staged = sys.specs[b].staged;
        const int32_t n = (cap < staged) ? cap : staged;   // U5: SIGNED min
        E.published[b] = n;
        E.pinned[b] = n;
        E.staged[b] = 0;
    }

    // ---- ps_diffuseParticleSum (reads the counts Copy just published) ----
    uint32_t total = 0;
    for (uint32_t b = 0; b < nb; ++b) total += uint32_t(E.published[b]);
    E.systemCount = total;
    return E;
}

// ---------------------------------------------------------------------------
static void checkSide(PxgSlangComputeDevice* dev, const char* label, const Rig& R,
                      const Sys& sys, uint32_t sysTag,
                      const DevBuf& bFlatPos, const DevBuf& bFlatVel, const DevBuf& bNum,
                      const DevBuf& bTwo, const DevBuf& bPinned, const DevBuf& bBufs)
{
    const uint32_t nb = sys.nbBufs;
    const Expect E = predict(sys, sysTag);

    const auto flatPos = downloadBuffer<F4>(dev, bFlatPos.h, FLAT_CAP);
    const auto flatVel = downloadBuffer<F4>(dev, bFlatVel.h, FLAT_CAP);
    const auto two = downloadBuffer<int32_t>(dev, bTwo.h, size_t(nb) * 2u);
    const auto pinned = downloadBuffer<int32_t>(dev, bPinned.h, nb);
    const auto num = downloadBuffer<uint32_t>(dev, bNum.h, 1);
    const auto bufs = downloadBuffer<PxgParticleDiffuseSimBuffer>(dev, bBufs.h, nb);

    size_t badFlat = 0;
    for (uint32_t i = 0; i < FLAT_CAP; ++i)
    {
        const F4 p = flatPos[i], q = E.flatPos[i];
        const F4 v = flatVel[i], w = E.flatVel[i];
        if (std::memcmp(&p, &q, sizeof(F4)) != 0 || std::memcmp(&v, &w, sizeof(F4)) != 0)
        {
            if (badFlat < 4)
                std::printf("    %s flat slot %u: pos (%g,%g,%g,%g) expected (%g,%g,%g,%g)\n",
                            label, i, p.x, p.y, p.z, p.w, q.x, q.y, q.z, q.w);
            ++badFlat;
        }
    }
    PXG_CHECK_MSG(badFlat == 0, "%s: %zu flat-array slots wrong (poison outside the gather "
                  "is checked too)", label, badFlat);

    size_t badCount = 0, badStart = 0;
    for (uint32_t b = 0; b < nb; ++b)
    {
        if (two[size_t(b) * 2 + 0] != E.published[b])
        {
            std::printf("    %s buffer %u: published %d, expected %d\n",
                        label, b, two[size_t(b) * 2 + 0], E.published[b]);
            ++badCount;
        }
        if (two[size_t(b) * 2 + 1] != E.staged[b]) ++badCount;
        if (pinned[b] != E.pinned[b])
        {
            std::printf("    %s buffer %u: pinned mirror %d, expected %d\n",
                        label, b, pinned[b], E.pinned[b]);
            ++badCount;
        }
        if (bufs[b].mStartIndex != E.startIndex[b])
        {
            std::printf("    %s buffer %u: mStartIndex %u, expected %u\n",
                        label, b, bufs[b].mStartIndex, E.startIndex[b]);
            ++badStart;
        }
        // Nothing in this tranche may touch the capacity, the flags or the params.
        if (bufs[b].mMaxNumParticles != sys.specs[b].capacity) ++badCount;
        if (bufs[b].mFlags != (0xABCD0000u | b)) ++badCount;
        if (bufs[b].mParams.threshold != 1.0f + float(b)) ++badCount;
    }
    PXG_CHECK_MSG(badCount == 0, "%s: %zu per-buffer count/immutable-field mismatches",
                  label, badCount);
    PXG_CHECK_MSG(badStart == 0, "%s: %zu start indices wrong", label, badStart);
    PXG_CHECK_MSG(num[0] == E.systemCount, "%s: the system count is %u, expected %u",
                  label, num[0], E.systemCount);
}

static void runAll(PxgSlangComputeDevice* dev)
{
    Rig R;
    if (!buildRig(dev, R)) { R.destroy(dev); return; }

    // The engine's own order within a frame, over ONE arena.
    {
        Kernel k("psDiffuseBuffers.spv", "ps_updateUnsortedDiffuseArrayLaunch", {},
                 sizeof(PsDiffBufBlob));
        PXG_CHECK_MSG(k.isValid(), "the UpdateUnsorted pipeline must build");
        if (!k.isValid()) { R.destroy(dev); return; }
        // Grid (ceil(maxPerBuffer/1024), GRID_Y, nbActiveSystems).  GRID_Y is
        // deliberately larger than either system's buffer count (U2).
        k.dispatch(1u, {}, R.blob, GRID_Y, 2u);
    }
    dev->flush();
    {
        Kernel k("psDiffuseBuffers.spv", "ps_diffuseParticleCopy", {}, sizeof(PsDiffBufBlob));
        PXG_CHECK_MSG(k.isValid(), "the Copy pipeline must build");
        if (!k.isValid()) { R.destroy(dev); return; }
        k.dispatch(1u, {}, R.blob, GRID_Y, 2u);
    }
    dev->flush();
    {
        Kernel k("psDiffuseBuffers.spv", "ps_diffuseParticleSum", {}, sizeof(PsDiffBufBlob));
        PXG_CHECK_MSG(k.isValid(), "the Sum pipeline must build");
        if (!k.isValid()) { R.destroy(dev); return; }
        k.dispatch(2u, {}, R.blob, 1u, 1u);   // grid (nbActiveSystems, 1, 1)
    }
    dev->flush();

    const Sys sysA{ 5u, A_SPECS };
    const Sys sysB{ 2u, B_SPECS };
    checkSide(dev, "sysA", R, sysA, 0u, R.bFlatPosA, R.bFlatVelA, R.bNumA,
              R.bTwoA, R.bPinnedA, R.bBufsA);
    checkSide(dev, "sysB", R, sysB, 1u, R.bFlatPosB, R.bFlatVelB, R.bNumB,
              R.bTwoB, R.bPinnedB, R.bBufsB);

    // Coverage, demanded rather than designed.
    {
        const Expect EA = predict(sysA, 0u);
        bool clamped = false, zeroCount = false;
        for (uint32_t b = 0; b < 5u; ++b)
        {
            if (A_SPECS[b].staged > int32_t(A_SPECS[b].capacity)) clamped = true;
            if (A_SPECS[b].published == 0u &&
                EA.startIndex[b] == A_SPECS[b].startSeed) zeroCount = true;
        }
        bool negative = false;
        for (uint32_t b = 0; b < 5u; ++b) if (A_SPECS[b].staged < 0) negative = true;
        PXG_CHECK_MSG(clamped, "no buffer over-subscribed its capacity -- U5's clamp is dead");
        PXG_CHECK_MSG(zeroCount, "no buffer had zero published particles -- U4 is untested");
        PXG_CHECK_MSG(negative, "no buffer staged a negative count -- signed and unsigned min "
                      "are indistinguishable");
        std::printf("    [ note ] sysA total %u over 5 buffers, sysB total %u over 2; "
                    "grid y = %u against buffer counts 4 and 2\n",
                    EA.systemCount, predict(sysB, 1u).systemCount, GRID_Y);
    }
    R.destroy(dev);
}

} // namespace

PXG_TEST(psDiffuseBuffers)
{
    PxgSlangComputeDevice* dev = testDevice();
    runAll(dev);
}
