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

// Verification of npInsertPatch.slang: insertIntoPatchStream.  THE WHOLE STREAM
// IS COMPARED BITWISE (P9.87) against an image that starts as the upload and
// overlays exactly the bytes contactPatchUtils.cuh:161-178 assigns: words 0-12
// whole, word 13 its LOW half only (:171, a PxU16 store), words 14-15 never.  So
// pad[0..4] of every written slot, and the four slack slots, must come back as
// uploaded.  The upload differs per word position, so a store that copies another
// poisoned word into the pad is visible too.  (Before P9.87 the fixture overlaid
// named fields only, and the shader's whole-word store of word 13 -- which zeroed
// pad[0] under a comment saying the CUDA leaves it -- passed.)  Skipped
// (0xffffffff) work items have no slot at all.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct MaterialData { float dynamicFriction, staticFriction, restitution, damping; uint32_t flags, fricCombineMode, restCombineMode, dampingCombineMode; };
struct PatchWork { uint32_t patchIndex, nbContacts, materialIndex0, materialIndex1; float nx, ny, nz, nw; };

// byte layout pinned to physx PxContactPatch (PX_ALIGN(16), 64 bytes)
struct alignas(16) TestContactPatch
{
    float mm[4];                       // 0
    alignas(16) float normal[3];       // 16
    float restitution, dynamicFriction, staticFriction, damping; // 28
    uint16_t startContactIndex;        // 44
    uint8_t nbContacts;                // 46
    uint8_t materialFlags;             // 47
    uint16_t internalFlags;            // 48
    uint16_t materialIndex0;           // 50
    uint16_t materialIndex1;           // 52
    uint16_t pad[5];                   // 54
};
static_assert(sizeof(TestContactPatch) == 64, "patch size");
static_assert(offsetof(TestContactPatch, normal) == 16, "normal offset");
static_assert(offsetof(TestContactPatch, restitution) == 28, "restitution offset");
static_assert(offsetof(TestContactPatch, startContactIndex) == 44, "startContactIndex offset");
static_assert(offsetof(TestContactPatch, internalFlags) == 48, "internalFlags offset");
static_assert(offsetof(TestContactPatch, materialIndex1) == 52, "materialIndex1 offset");
static_assert(offsetof(TestContactPatch, pad) == 54, "pad offset");

uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

const uint32_t MODE_MIN = 1, MODE_MULTIPLY = 2, MODE_MAX = 3;
const uint32_t FLAG_DISABLE_FRICTION = 1, FLAG_DISABLE_STRONG_FRICTION = 2, FLAG_COMPLIANT_ACC_SPRING = 16;

float combineScalars(float a, float b, uint32_t mode)
{
    switch (mode) { case 0: return 0.5f * (a + b); case 1: return std::min(a, b); case 2: return a * b; case 3: return std::max(a, b); }
    return 0.f;
}

void combine(const MaterialData& m0, const MaterialData& m1, float& sta, float& dyn, float& rest, float& damping, uint32_t& flags)
{
    const float r0 = m0.restitution, r1 = m1.restitution;
    const bool c0 = r0 < 0.f, c1 = r1 < 0.f, one = c0 != c1, both = c0 && c1;
    const bool a0 = (m0.flags & FLAG_COMPLIANT_ACC_SPRING) != 0, a1 = (m1.flags & FLAG_COMPLIANT_ACC_SPRING) != 0, oneAcc = a0 != a1;
    if (both && oneAcc) rest = a0 ? r0 : r1;
    else { uint32_t m = one ? MODE_MIN : std::max(m0.restCombineMode, m1.restCombineMode); float f = (both && m == MODE_MULTIPLY) ? -1.f : 1.f; rest = f * combineScalars(r0, r1, m); }
    if (both && oneAcc) damping = a0 ? m0.damping : m1.damping;
    else { uint32_t m = one ? MODE_MAX : std::max(m0.dampingCombineMode, m1.dampingCombineMode); damping = combineScalars(m0.damping, m1.damping, m); }
    const uint32_t cf = m0.flags | m1.flags;
    if ((cf & FLAG_DISABLE_FRICTION) == 0) { uint32_t fm = std::max(m0.fricCombineMode, m1.fricCombineMode); float d = combineScalars(m0.dynamicFriction, m1.dynamicFriction, fm), s = combineScalars(m0.staticFriction, m1.staticFriction, fm); float fd = std::max(d, 0.f); dyn = fd; sta = (s - fd) >= 0.f ? s : fd; flags = cf; }
    else { flags = cf | FLAG_DISABLE_STRONG_FRICTION; dyn = 0.f; sta = 0.f; }
}

PXG_TEST(npInsertPatch_write)
{
    std::mt19937 rng(45010);
    const uint32_t nbMaterials = 300;
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    // P9.88: the table is 0x10000 + nbMaterials long and the second bank [0x10000 + k] holds
    // DIFFERENT materials, so an index k | 0x10000 tells a lookup that narrows to PxU16 first
    // (materialCombiner.cuh:35) from one that does not -- and both stay in bounds.
    std::vector<MaterialData> mats(0x10000u + nbMaterials);
    for (uint32_t mIdx = 0; mIdx < 0x10000u + nbMaterials; mIdx = (mIdx + 1 == nbMaterials) ? 0x10000u : mIdx + 1)
    {
        MaterialData& m = mats[mIdx];
        m.dynamicFriction = rf(0.f, 1.5f); m.staticFriction = rf(0.f, 1.5f);
        m.restitution = (rng() % 4 == 0) ? rf(-1.f, 0.f) : rf(0.f, 1.f);
        m.damping = rf(0.f, 40.f);
        uint32_t f = 0; if (rng() % 8 == 0) f |= FLAG_DISABLE_FRICTION; if (rng() % 3 == 0) f |= FLAG_COMPLIANT_ACC_SPRING;
        m.flags = f; m.fricCombineMode = rng() % 4; m.restCombineMode = rng() % 4; m.dampingCombineMode = rng() % 4;
    }

    // one 64-byte patch slot per active work item
    std::vector<PatchWork> work(N);
    uint32_t nextSlot = 0, nbActive = 0, nbSkip = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        PatchWork w{};
        w.materialIndex0 = rng() % nbMaterials; w.materialIndex1 = rng() % nbMaterials;
        if (rng() % 4 == 0) w.materialIndex0 |= 0x10000u;
        if (rng() % 4 == 0) w.materialIndex1 |= 0x10000u;
        // One item in sixteen past 255: nbContacts is a PxU8 field (:163), so the
        // CUDA keeps the low byte -- the `& 0xff` of word 11 needs an input above it.
        w.nbContacts = (rng() % 16 == 0) ? 0x100u + rng() % 0x300u : 1 + rng() % 6;
        w.nx = rf(-1, 1); w.ny = rf(-1, 1); w.nz = rf(-1, 1); w.nw = 0;
        if (rng() % 10 == 0) { w.patchIndex = 0xffffffffu; ++nbSkip; }
        else { w.patchIndex = nextSlot * 64u; ++nextSlot; ++nbActive; }
        work[i] = w;
    }
    const uint32_t patchStreamWords = (nextSlot + 4) * 16; // + slack

    // patch stream pre-filled with a per-word sentinel so pad and slack stay detectable,
    // and so pad words 13, 14 and 15 of one slot all differ
    std::vector<uint32_t> streamInit(patchStreamWords);
    for (uint32_t k = 0; k < patchStreamWords; ++k) streamInit[k] = 0xabcddcbau ^ (k * 0x00010001u);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    PxgBufferHandle bWork = createBufferFrom(dev, work);
    PxgBufferHandle bStream = createBufferFrom(dev, streamInit);

    struct InsertParams { uint32_t numTests; } p = { N };
    Kernel k("npInsertPatch.spv", "insertIntoPatchStream", { 32, 32, 4 }, sizeof(InsertParams));
    k.dispatch(GROUPS, { bMat, bWork, bStream }, p);
    dev->flush();

    const auto gStream = downloadBuffer<uint32_t>(dev, bStream, patchStreamWords);

    // The expected image: the upload, with exactly the CUDA's assignments overlaid.
    std::vector<uint32_t> want(streamInit);
    size_t wideCounts = 0, hiBit = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (work[i].patchIndex == 0xffffffffu) continue;
        uint32_t* d = &want[work[i].patchIndex / 4];
        float sta, dyn, rest, damping; uint32_t flags;
        combine(mats[work[i].materialIndex0 & 0xffffu], mats[work[i].materialIndex1 & 0xffffu], sta, dyn, rest, damping, flags);
        if ((work[i].materialIndex0 | work[i].materialIndex1) & 0x10000u) ++hiBit;
        const uint32_t one = bitsOf(1.f);
        d[0] = one; d[1] = one; d[2] = one; d[3] = one;
        d[4] = bitsOf(work[i].nx); d[5] = bitsOf(work[i].ny); d[6] = bitsOf(work[i].nz);
        d[7] = bitsOf(rest); d[8] = bitsOf(dyn); d[9] = bitsOf(sta); d[10] = bitsOf(damping);
        d[11] = ((work[i].nbContacts & 0xffu) << 16) | ((flags & 0xffu) << 24);   // startContactIndex = 0
        d[12] = (work[i].materialIndex0 & 0xffffu) << 16;                        // internalFlags = 0
        d[13] = (d[13] & 0xffff0000u) | (work[i].materialIndex1 & 0xffffu);      // pad[0] kept (:171)
        // d[14], d[15]: pad[1..4], never written
        if (work[i].nbContacts > 0xffu) ++wideCounts;
    }
    size_t mm = 0, padBad = 0;
    for (uint32_t w = 0; w < patchStreamWords; ++w)
        if (gStream[w] != want[w])
        {
            if (mm < 6) std::printf("  stream word %u (slot %u word %u): got %08x want %08x\n", w, w / 16u, w % 16u,
                                    gStream[w], want[w]);
            ++mm;
            if ((w % 16u) >= 13u) ++padBad;
        }
    PXG_CHECK_MSG(mm == 0, "insertIntoPatchStream: %zu / %u stream words differ (%zu in words 13-15 of a slot)",
                  mm, patchStreamWords, padBad);
    PXG_CHECK_MSG(wideCounts > 50, "only %zu items with nbContacts past 255 -- the PxU8 truncation is unexercised",
                  wideCounts);
    PXG_CHECK_MSG(hiBit > N / 8, "only %zu written patches with a material index past 0xFFFF", hiBit);
    PXG_CHECK_MSG(nbActive > N / 2 && nbSkip > N / 50, "activity mix under-exercised (active=%u skip=%u)", nbActive, nbSkip);

    for (PxgBufferHandle h : { bMat, bWork, bStream }) dev->destroyBuffer(h);
}

} // namespace
