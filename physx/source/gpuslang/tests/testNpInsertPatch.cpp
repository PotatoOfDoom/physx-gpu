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

// Verification of npInsertPatch.slang: insertIntoPatchStream.  The written
// PxContactPatch headers are overlaid on the downloaded patch-stream bytes using
// a struct whose byte layout is pinned (static_assert) to the real PxContactPatch,
// and every field is compared against a C++ reference combining the materials.
// Skipped (0xffffffff) work items must leave their slots untouched.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

    std::vector<MaterialData> mats(nbMaterials);
    for (auto& m : mats)
    {
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
        w.nbContacts = 1 + rng() % 6;
        w.nx = rf(-1, 1); w.ny = rf(-1, 1); w.nz = rf(-1, 1); w.nw = 0;
        if (rng() % 10 == 0) { w.patchIndex = 0xffffffffu; ++nbSkip; }
        else { w.patchIndex = nextSlot * 64u; ++nextSlot; ++nbActive; }
        work[i] = w;
    }
    const uint32_t patchStreamWords = (nextSlot + 4) * 16; // + slack

    // patch stream pre-filled with a sentinel so skipped slots / pad stay detectable
    std::vector<uint32_t> streamInit(patchStreamWords, 0xabcddcbau);

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

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (work[i].patchIndex == 0xffffffffu) continue;
        const uint32_t base = work[i].patchIndex / 4;
        TestContactPatch patch;
        std::memcpy(&patch, &gStream[base], 64);

        float sta, dyn, rest, damping; uint32_t flags;
        combine(mats[work[i].materialIndex0], mats[work[i].materialIndex1], sta, dyn, rest, damping, flags);

        bool ok = patch.mm[0] == 1.f && patch.mm[1] == 1.f && patch.mm[2] == 1.f && patch.mm[3] == 1.f &&
                  patch.normal[0] == work[i].nx && patch.normal[1] == work[i].ny && patch.normal[2] == work[i].nz &&
                  patch.restitution == rest && patch.dynamicFriction == dyn && patch.staticFriction == sta &&
                  patch.damping == damping && patch.startContactIndex == 0 &&
                  patch.nbContacts == uint8_t(work[i].nbContacts) && patch.materialFlags == uint8_t(flags) &&
                  patch.internalFlags == 0 && patch.materialIndex0 == uint16_t(work[i].materialIndex0) &&
                  patch.materialIndex1 == uint16_t(work[i].materialIndex1);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "insertIntoPatchStream: %zu / %u patch mismatches", mm, nbActive);
    PXG_CHECK_MSG(nbActive > N / 2 && nbSkip > N / 50, "activity mix under-exercised (active=%u skip=%u)", nbActive, nbSkip);

    for (PxgBufferHandle h : { bMat, bWork, bStream }) dev->destroyBuffer(h);
}

} // namespace
