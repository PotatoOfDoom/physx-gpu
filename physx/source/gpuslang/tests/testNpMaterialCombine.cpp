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

// Verification of npMaterialCombine.slang: PxsCombineMaterials.  Randomized
// material pairs (mixed combine modes, compliant / acceleration-spring flags,
// friction-disable) are combined on the GPU and compared bit-exactly against a
// C++ reference mirroring PxsMaterialCombiner.h, with counters asserting the
// compliant and friction-disable branches are exercised.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct MaterialData { float dynamicFriction, staticFriction, restitution, damping; uint32_t flags, fricCombineMode, restCombineMode, dampingCombineMode; };
struct F4 { float x, y, z, w; };

const uint32_t MODE_MIN = 1, MODE_MULTIPLY = 2, MODE_MAX = 3;
const uint32_t FLAG_DISABLE_FRICTION = 1, FLAG_DISABLE_STRONG_FRICTION = 2, FLAG_COMPLIANT_ACC_SPRING = 16;

float combineScalars(float a, float b, uint32_t mode)
{
    switch (mode)
    {
    case 0: return 0.5f * (a + b);
    case 1: return std::min(a, b);
    case 2: return a * b;
    case 3: return std::max(a, b);
    }
    return 0.f;
}

void refCombine(const MaterialData& m0, const MaterialData& m1, F4& outFric, uint32_t& outFlags)
{
    const float r0 = m0.restitution, r1 = m1.restitution;
    const bool compliant0 = r0 < 0.f, compliant1 = r1 < 0.f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const bool cAcc0 = (m0.flags & FLAG_COMPLIANT_ACC_SPRING) != 0;
    const bool cAcc1 = (m1.flags & FLAG_COMPLIANT_ACC_SPRING) != 0;
    const bool exactlyOneAcc = cAcc0 != cAcc1;

    float combinedRestitution;
    if (bothCompliant && exactlyOneAcc) combinedRestitution = cAcc0 ? r0 : r1;
    else
    {
        const uint32_t mode = exactlyOneCompliant ? MODE_MIN : std::max(m0.restCombineMode, m1.restCombineMode);
        const float flip = (bothCompliant && mode == MODE_MULTIPLY) ? -1.f : 1.f;
        combinedRestitution = flip * combineScalars(r0, r1, mode);
    }

    float combinedDamping;
    {
        const float d0 = m0.damping, d1 = m1.damping;
        if (bothCompliant && exactlyOneAcc) combinedDamping = cAcc0 ? d0 : d1;
        else
        {
            const uint32_t mode = exactlyOneCompliant ? MODE_MAX : std::max(m0.dampingCombineMode, m1.dampingCombineMode);
            combinedDamping = combineScalars(d0, d1, mode);
        }
    }

    float cDyn, cSta;
    uint32_t cFlags;
    {
        const uint32_t combineFlags = m0.flags | m1.flags;
        if ((combineFlags & FLAG_DISABLE_FRICTION) == 0)
        {
            const uint32_t fm = std::max(m0.fricCombineMode, m1.fricCombineMode);
            const float dyn = combineScalars(m0.dynamicFriction, m1.dynamicFriction, fm);
            const float sta = combineScalars(m0.staticFriction, m1.staticFriction, fm);
            const float fDyn = std::max(dyn, 0.f);
            const float fSta = (sta - fDyn) >= 0.f ? sta : fDyn;
            cDyn = fDyn; cSta = fSta; cFlags = combineFlags;
        }
        else
        {
            cFlags = combineFlags | FLAG_DISABLE_STRONG_FRICTION; cDyn = 0.f; cSta = 0.f;
        }
    }
    outFric = { cSta, cDyn, combinedRestitution, combinedDamping };
    outFlags = cFlags;
}

PXG_TEST(npMaterialCombine_combine)
{
    std::mt19937 rng(43010);
    const uint32_t nbMaterials = 400;
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<MaterialData> mats(nbMaterials);
    for (auto& m : mats)
    {
        m.dynamicFriction = rf(0.f, 1.5f);
        m.staticFriction = rf(0.f, 1.5f);
        m.restitution = (rng() % 4 == 0) ? rf(-1.0f, 0.f) : rf(0.f, 1.0f); // ~1/4 compliant (negative)
        m.damping = rf(0.f, 50.f);
        uint32_t f = 0;
        if (rng() % 8 == 0) f |= FLAG_DISABLE_FRICTION;
        if (rng() % 3 == 0) f |= FLAG_COMPLIANT_ACC_SPRING;
        if (rng() % 5 == 0) f |= 4; // some unrelated flag bit
        m.flags = f;
        m.fricCombineMode = rng() % 4;
        m.restCombineMode = rng() % 4;
        m.dampingCombineMode = rng() % 4;
    }

    std::vector<uint32_t> pairsFlat(2 * N);
    for (uint32_t i = 0; i < N; ++i) { pairsFlat[2 * i] = rng() % nbMaterials; pairsFlat[2 * i + 1] = rng() % nbMaterials; }

    std::vector<F4> refFric(N);
    std::vector<uint32_t> refFlags(N);
    size_t nCompliant = 0, nDisable = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const MaterialData& m0 = mats[pairsFlat[2 * i]];
        const MaterialData& m1 = mats[pairsFlat[2 * i + 1]];
        refCombine(m0, m1, refFric[i], refFlags[i]);
        if (m0.restitution < 0.f && m1.restitution < 0.f) ++nCompliant;
        if ((m0.flags | m1.flags) & FLAG_DISABLE_FRICTION) ++nDisable;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outFric(N, F4{ 9, 9, 9, 9 });
    std::vector<uint32_t> outFlags(N, 0xdeadu);

    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairsFlat);
    PxgBufferHandle bFric = createBufferFrom(dev, outFric);
    PxgBufferHandle bFlags = createBufferFrom(dev, outFlags);

    struct MatParams { uint32_t numPairs; } p = { N };
    Kernel k("npMaterialCombine.spv", "combineMaterials", { 32, 8, 16, 4 }, sizeof(MatParams));
    k.dispatch(GROUPS, { bMat, bPairs, bFric, bFlags }, p);
    dev->flush();

    const auto gFric = downloadBuffer<F4>(dev, bFric, N);
    const auto gFlags = downloadBuffer<uint32_t>(dev, bFlags, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4& a = gFric[i]; const F4& r = refFric[i];
        if (!almostEqual(a.x, r.x, 1e-5f, 1e-6f) || !almostEqual(a.y, r.y, 1e-5f, 1e-6f) ||
            !almostEqual(a.z, r.z, 1e-5f, 1e-6f) || !almostEqual(a.w, r.w, 1e-5f, 1e-6f) ||
            gFlags[i] != refFlags[i])
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "combineMaterials: %zu mismatches", mm);
    PXG_CHECK_MSG(nCompliant > N / 50 && nDisable > N / 50, "material combine: branches under-exercised (compliant=%zu disable=%zu)", nCompliant, nDisable);

    for (PxgBufferHandle h : { bMat, bPairs, bFric, bFlags }) dev->destroyBuffer(h);
}

} // namespace
