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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct MaterialData { float dynamicFriction, staticFriction, restitution, damping; uint32_t flags, fricCombineMode, restCombineMode, dampingCombineMode; };
struct F4 { float x, y, z, w; };

const uint32_t MODE_MIN = 1, MODE_MULTIPLY = 2, MODE_MAX = 3;
const uint32_t FLAG_DISABLE_FRICTION = 1, FLAG_DISABLE_STRONG_FRICTION = 2, FLAG_COMPLIANT_ACC_SPRING = 16;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// P9.86e: eMIN/eMAX are PxMin/PxMax<float> = intrinsics::selectMin/selectMax (PxsMaterialCombiner.h:43/47/60/64
// -> PxMath.h:81-101 -> PxUnixMathIntrinsics.h:116-125): on a +-0 tie the SECOND operand survives.  std::min /
// std::max keep the FIRST operand on a tie, so they are not PxMin/PxMax; the oracle spells the selects.
// gCombineVariant is a mutant twin replaying a reverted port: 1 = eMIN as GLSL FMin, 2 = eMAX as FMax, with
// llvmpipe's zero order under the SZINP stamp (FMin(+-0,-+0) = -0, FMax = +0, either operand order).
static int gCombineVariant = 0;
static float fminLp(float a, float b) { return (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); }
static float fmaxLp(float a, float b) { return (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); }
float combineScalars(float a, float b, uint32_t mode)
{
    switch (mode)
    {
    case 0: return 0.5f * (a + b);
    case 1: return (gCombineVariant == 1) ? fminLp(a, b) : ((a < b) ? a : b);   // PxMin = selectMin
    case 2: return a * b;
    case 3: return (gCombineVariant == 2) ? fmaxLp(a, b) : ((a > b) ? a : b);   // PxMax = selectMax
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
            const float fDyn = (dyn > 0.f) ? dyn : 0.f;   // PxMax(dynFriction, 0.0f) = selectMax (PxsMaterialCombiner.h:161)
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

    // P9.88: the table is 0x10000 + nbMaterials long and the second bank [0x10000 + k] holds
    // DIFFERENT materials, so an index k | 0x10000 tells a lookup that narrows to PxU16 first
    // (materialCombiner.cuh:35) from one that does not -- and both stay in bounds.
    std::vector<MaterialData> mats(0x10000u + nbMaterials);
    for (uint32_t mIdx = 0; mIdx < 0x10000u + nbMaterials; mIdx = (mIdx + 1 == nbMaterials) ? 0x10000u : mIdx + 1)
    {
        MaterialData& m = mats[mIdx];
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
    size_t hiBit = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        pairsFlat[2 * i] = rng() % nbMaterials; pairsFlat[2 * i + 1] = rng() % nbMaterials;
        if (rng() % 4 == 0) pairsFlat[2 * i] |= 0x10000u;
        if (rng() % 4 == 0) pairsFlat[2 * i + 1] |= 0x10000u;
        if ((pairsFlat[2 * i] | pairsFlat[2 * i + 1]) & 0x10000u) ++hiBit;
    }

    std::vector<F4> refFric(N);
    std::vector<uint32_t> refFlags(N);
    size_t nCompliant = 0, nDisable = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const MaterialData& m0 = mats[pairsFlat[2 * i] & 0xffffu];
        const MaterialData& m1 = mats[pairsFlat[2 * i + 1] & 0xffffu];
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
    // P9.86e: every output is a pure combine result (no rounding the GPU may do differently), so the
    // four floats compare bitwise, sign of zero included.
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4& a = gFric[i]; const F4& r = refFric[i];
        if (f2u(a.x) != f2u(r.x) || f2u(a.y) != f2u(r.y) || f2u(a.z) != f2u(r.z) || f2u(a.w) != f2u(r.w) ||
            gFlags[i] != refFlags[i])
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "combineMaterials: %zu mismatches", mm);
    PXG_CHECK_MSG(hiBit > N / 4, "combineMaterials: only %zu pairs with bit 16 set", hiBit);
    PXG_CHECK_MSG(nCompliant > N / 50 && nDisable > N / 50, "material combine: branches under-exercised (compliant=%zu disable=%zu)", nCompliant, nDisable);

    for (PxgBufferHandle h : { bMat, bPairs, bFric, bFlags }) dev->destroyBuffer(h);
}

// P9.86e DIRECTED SIGNED-ZERO MATERIALS.  eMIN/eMAX are PxMin/PxMax<float> selects: a +-0 tie returns the
// SECOND operand, so PxMin(-0,+0) = +0 and PxMax(+0,-0) = -0, where GLSL FMin/FMax on llvmpipe give -0 / +0
// in either order.  -0 is a valid material value (NpMaterial checks finite and >= 0.f), and a restitution
// of -0 is rigid (-0 < 0 is false).  The random bank never produces -0.
PXG_TEST(npMaterialCombine_materialZeroSign)
{
    const float PZ = 0.f, NZ = u2f(0x80000000u);
    // {dyn, sta, rest, damp, flags, fricMode, restMode, dampMode}; modes 1 = eMIN, 3 = eMAX.
    const std::vector<MaterialData> mats = {
        { PZ, NZ, NZ, NZ, 0, 1, 1, 1 },          // D0  -0 under eMIN
        { PZ, PZ, PZ, PZ, 0, 1, 1, 1 },          // D1  +0 under eMIN
        { PZ, PZ, PZ, PZ, 0, 3, 3, 3 },          // D2  +0 under eMAX
        { PZ, NZ, NZ, NZ, 0, 3, 3, 3 },          // D3  -0 under eMAX
        { PZ, PZ, -0.5f, PZ, 0, 0, 0, 0 },       // D4  compliant (rest < 0), damping +0
        { PZ, PZ, NZ, NZ, 0, 0, 0, 0 },          // D5  rigid (rest -0), damping -0
        { NZ, PZ, PZ, PZ, 0, 0, 0, 0 },          // D6  dynamic friction -0 under eAVERAGE
    };
    // (ma, mb) and (mb, ma) are the two shape orders.  rest/dynF/staF/damp are the CUDA's bits computed by hand
    // from PxsCombineMaterials; killsVariant names the host twin whose output must differ.
    struct DirCase { uint32_t ma, mb, rest, dyn, sta, damp; int killsVariant; };
    const uint32_t NDIR = 6;
    const DirCase dc[NDIR] = {
        { 0, 1, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 1 },   // K0 eMIN(-0,+0): PxMin +0, FMin -0
        { 2, 3, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 2 },   // K1 eMAX(+0,-0): PxMax -0, FMax +0
        { 4, 5, 0xBF000000u, 0u, 0x00000000u, 0x80000000u, 2 },   // K2 one compliant: damping forced to eMAX(+0,-0)
        { 1, 0, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 0 },   // K3 control: eMIN(+0,-0) = -0 in both
        { 3, 2, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K4 control: eMAX(-0,+0) = +0 in both
        { 6, 6, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K5 d = 0.5*(-0 + -0) = -0: PxMax(d, 0) keeps dynF +0
    };
    PXG_CHECK_MSG(std::signbit(0.5f * (mats[6].dynamicFriction + mats[6].dynamicFriction)),
                  "combineMaterials directed: K5 must reach the dynF clamp with -0");

    std::vector<uint32_t> pairsFlat(2 * NDIR);
    std::vector<F4> refFric(NDIR);
    std::vector<uint32_t> refFlags(NDIR);
    size_t cMinTie = 0, cMaxTie = 0;
    for (uint32_t k = 0; k < NDIR; ++k)
    {
        pairsFlat[2 * k] = dc[k].ma; pairsFlat[2 * k + 1] = dc[k].mb;
        refCombine(mats[dc[k].ma], mats[dc[k].mb], refFric[k], refFlags[k]);
        const F4& r = refFric[k];
        PXG_CHECK_MSG(f2u(r.z) == dc[k].rest && f2u(r.y) == dc[k].dyn && f2u(r.x) == dc[k].sta && f2u(r.w) == dc[k].damp,
                      "combineMaterials directed %u: oracle rest/dynF/staF/damp %08x %08x %08x %08x, CUDA %08x %08x %08x %08x",
                      k, f2u(r.z), f2u(r.y), f2u(r.x), f2u(r.w), dc[k].rest, dc[k].dyn, dc[k].sta, dc[k].damp);
        for (int v = 1; v <= 2; ++v)
        {
            F4 tw; uint32_t twFlags;
            gCombineVariant = v;
            refCombine(mats[dc[k].ma], mats[dc[k].mb], tw, twFlags);
            gCombineVariant = 0;
            const bool differs = f2u(tw.x) != f2u(r.x) || f2u(tw.y) != f2u(r.y) || f2u(tw.z) != f2u(r.z) ||
                                 f2u(tw.w) != f2u(r.w) || twFlags != refFlags[k];
            PXG_CHECK_MSG(differs == (dc[k].killsVariant == v), "combineMaterials directed %u: twin %d %s the output",
                          k, v, differs ? "changes" : "does not change");
            if (differs) ++(v == 1 ? cMinTie : cMaxTie);
        }
    }
    PXG_CHECK_MSG(cMinTie == 1 && cMaxTie == 2, "combineMaterials: signed-zero combine coverage thin (minTie %zu maxTie %zu)",
                  cMinTie, cMaxTie);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> outFric(NDIR, F4{ 9, 9, 9, 9 });
    std::vector<uint32_t> outFlags(NDIR, 0xdeadu);
    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairsFlat);
    PxgBufferHandle bFric = createBufferFrom(dev, outFric);
    PxgBufferHandle bFlags = createBufferFrom(dev, outFlags);
    struct MatParams { uint32_t numPairs; } p = { NDIR };
    Kernel kr("npMaterialCombine.spv", "combineMaterials", { 32, 8, 16, 4 }, sizeof(MatParams));
    kr.dispatch(1, { bMat, bPairs, bFric, bFlags }, p);
    dev->flush();
    const auto gFric = downloadBuffer<F4>(dev, bFric, NDIR);
    const auto gFlags = downloadBuffer<uint32_t>(dev, bFlags, NDIR);
    for (uint32_t k = 0; k < NDIR; ++k)
    {
        const F4& a = gFric[k]; const F4& r = refFric[k];
        PXG_CHECK_MSG(f2u(a.x) == f2u(r.x) && f2u(a.y) == f2u(r.y) && f2u(a.z) == f2u(r.z) && f2u(a.w) == f2u(r.w) &&
                      gFlags[k] == refFlags[k],
                      "combineMaterials directed %u: GPU rest/dynF/staF/damp %08x %08x %08x %08x, oracle %08x %08x %08x %08x",
                      k, f2u(a.z), f2u(a.y), f2u(a.x), f2u(a.w), f2u(r.z), f2u(r.y), f2u(r.x), f2u(r.w));
    }

    for (PxgBufferHandle h : { bMat, bPairs, bFric, bFlags }) dev->destroyBuffer(h);
}

} // namespace
