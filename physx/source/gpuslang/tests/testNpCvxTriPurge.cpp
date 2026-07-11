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

// Verification of npCvxTriPurge.slang: ctCvxTriPurge — purgeEmptyTriangles
// (convexMeshCorrelate.cu :58-87), the warp-SIMD in-place stream compaction that
// keeps only triangles with contacts.  One thread simulates the whole 32-lane
// warp, so the compacted order (lane order per 32-block) is deterministic; the
// three in-place/out arrays (depthBuffer, nicBuffer, tempCounts) AND the returned
// nbTriangles are compared position-for-position bit-exactly against an exact CPU
// transcription (identical block loop / ballot / exclusive-popc dest / in-place
// stores).  Only float touched is the FLT_MAX sentinel test (exact); everything
// else is integer, so the arbitrary depth/normal payloads compare bit-for-bit.
// Fixtures cover both invalidity paths (nbContacts==0 and depth==FLT_MAX), the
// deferred bit (masked out by getNbContacts), full and partial tail blocks, the
// in-place overlap (little compaction), and count 0/1.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct PgParams { uint32_t numTests, stride, pad0, pad1; };

static const uint32_t STRIDE = 128u; // per-test block length (count <= STRIDE)
static const float FLT_MAX_R = 3.402823466e38f;
static const uint32_t DEFERRED = 0x80000000u;
static const uint32_t NB_SHIFT = 27u;

static uint32_t getNbContactsRef(uint32_t index) { return (index & ~DEFERRED) >> NB_SHIFT; }
static uint32_t popc(uint32_t x) { uint32_t c = 0; while (x) { c += x & 1u; x >>= 1; } return c; }
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

struct PurgeRef { std::vector<float> depth; std::vector<F4> nic; std::vector<uint32_t> temp; uint32_t nbTriangles; };

static PurgeRef purgeRef(std::vector<float> depth, std::vector<F4> nic, uint32_t count, uint32_t tempInit)
{
    PurgeRef r;
    r.depth = std::move(depth);  // length STRIDE, in-place
    r.nic = std::move(nic);      // length STRIDE, in-place
    r.temp.assign(STRIDE, tempInit);
    uint32_t nbTriangles = 0u;
    for (int loaded = 0; loaded < (int)count; loaded += 32)
    {
        const int nbToLoad = std::min((int)count - loaded, 32);
        float dT[32]; uint32_t nbC[32]; F4 nicT[32]; uint32_t validBits = 0u;
        for (uint32_t tI = 0; tI < 32; ++tI)
        {
            dT[tI] = ((int)tI < nbToLoad) ? r.depth[loaded + tI] : FLT_MAX_R;
            nicT[tI] = r.nic[loaded + tI];
            nbC[tI] = (nbToLoad != 0) ? getNbContactsRef(asU32(nicT[tI].w)) : 0u;
            const bool valid = (dT[tI] != FLT_MAX_R) && (nbC[tI] > 0u);
            if (valid) validBits |= (1u << tI);
        }
        for (uint32_t tI = 0; tI < 32; ++tI)
        {
            if ((validBits & (1u << tI)) == 0u) continue;
            const uint32_t dest = popc(validBits & ((1u << tI) - 1u)) + nbTriangles;
            r.depth[dest] = dT[tI];
            r.temp[dest] = ((uint32_t(loaded) + tI) << 4) | nbC[tI];
            r.nic[dest] = nicT[tI];
        }
        nbTriangles += popc(validBits);
    }
    r.nbTriangles = nbTriangles;
    return r;
}

PXG_TEST(npCvxTriPurge_purge)
{
    std::mt19937 rng(1140114);
    const uint32_t TEMP_INIT = 0xEEEEEEEEu;

    std::vector<float> depthFlat;
    std::vector<F4> nicFlat;
    std::vector<uint32_t> countIn;
    std::vector<PurgeRef> ref;

    size_t nValid = 0, nInvCount = 0, nInvDepth = 0, nPartial = 0, nFull = 0, nEmpty = 0, nDeferred = 0;

    std::uniform_real_distribution<float> uf(-8.0f, 8.0f);

    auto emit = [&](uint32_t count, const std::vector<float>& depth, const std::vector<F4>& nic)
    {
        // pad both arrays to STRIDE (slots [count, STRIDE) are read by the tail
        // quirk but never written; they must match, so seed them deterministically)
        std::vector<float> d(STRIDE); std::vector<F4> n(STRIDE);
        for (uint32_t i = 0; i < STRIDE; ++i)
        {
            d[i] = (i < depth.size()) ? depth[i] : (float(i) + 0.5f); // padding depth finite
            // padding nic carries nbContacts>0 so the depth-gate boundary (tI ==
            // nbToLoad reading slot `count`) is only excluded by the FLT_MAX gate,
            // not incidentally by a zero contact count -> pins `tI < nbToLoad`
            const uint32_t padIndex = ((1u + (i % 15u)) << NB_SHIFT) | (i & 0xFFFFu) | ((i & 1u) ? DEFERRED : 0u);
            n[i] = (i < nic.size()) ? nic[i] : F4{ float(i), -float(i), 1.0f, asF32(padIndex) };
        }
        for (uint32_t i = 0; i < STRIDE; ++i) { depthFlat.push_back(d[i]); nicFlat.push_back(n[i]); }
        countIn.push_back(count);
        ref.push_back(purgeRef(d, n, count, TEMP_INIT));
    };

    auto genEntry = [&](float& depthOut, F4& nicOut)
    {
        const uint32_t roll = rng() % 10;
        uint32_t nb;
        float dep;
        if (roll < 4) { nb = 1u + rng() % 15u; dep = uf(rng); ++nValid; }              // valid (finite depth, nb>0)
        else if (roll < 7) { nb = 0u; dep = uf(rng); ++nInvCount; }                     // invalid: nbContacts==0
        else { nb = rng() % 16u; dep = FLT_MAX_R; ++nInvDepth; }                         // invalid: depth==FLT_MAX sentinel
        // pack index: low 27 bits = triangle index, bits 27-30 = nbContacts, bit31 deferred (random)
        const uint32_t tri = rng() % (1u << 20);
        uint32_t index = (nb << NB_SHIFT) | tri;
        if (rng() % 2) { index |= DEFERRED; ++nDeferred; } // deferred bit must be masked out
        depthOut = dep;
        nicOut = F4{ uf(rng), uf(rng), uf(rng), asF32(index) };
    };

    auto runFamily = [&](uint32_t count)
    {
        std::vector<float> depth(count);
        std::vector<F4> nic(count);
        for (uint32_t i = 0; i < count; ++i) genEntry(depth[i], nic[i]);
        if (count == 0) ++nEmpty;
        else if (count % 32 == 0) ++nFull;
        else ++nPartial;
        emit(count, depth, nic);
    };

    // varied counts: 0, 1, partial/full blocks, up to STRIDE
    for (uint32_t iter = 0; iter < 1400; ++iter)
    {
        uint32_t count;
        const uint32_t r = rng() % 20;
        if (r == 0) count = 0;
        else if (r == 1) count = 1;
        else if (r < 6) count = 32u * (1u + rng() % 4u);        // exact multiples of 32
        else count = 1u + rng() % STRIDE;                        // arbitrary (partial tails)
        runFamily(count);
    }

    // a batch of ALL-valid configs (little compaction -> in-place writes overlap
    // the read window heavily, stressing the read-then-write ordering)
    for (uint32_t iter = 0; iter < 200; ++iter)
    {
        const uint32_t count = 33u + rng() % (STRIDE - 33u);
        std::vector<float> depth(count);
        std::vector<F4> nic(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t nb = 1u + rng() % 15u;
            const uint32_t index = (nb << NB_SHIFT) | (rng() % (1u << 20)) | ((rng() % 2) ? DEFERRED : 0u);
            depth[i] = uf(rng);
            nic[i] = F4{ uf(rng), uf(rng), uf(rng), asF32(index) };
            ++nValid;
        }
        if (count % 32 == 0) ++nFull; else ++nPartial;
        emit(count, depth, nic);
    }

    const uint32_t N = (uint32_t)countIn.size();
    PxgSlangComputeDevice* dev = testDevice();

    PxgBufferHandle bD = createBufferFrom(dev, depthFlat);
    PxgBufferHandle bN = createBufferFrom(dev, nicFlat);
    std::vector<uint32_t> tempInit(N * STRIDE, TEMP_INIT);
    PxgBufferHandle bT = createBufferFrom(dev, tempInit);
    PxgBufferHandle bC = createBufferFrom(dev, countIn);

    PgParams prm = { N, STRIDE, 0, 0 };
    Kernel k("npCvxTriPurge.spv", "ctCvxTriPurge",
             { 4, (uint32_t)sizeof(F4), 4, 4 }, sizeof(PgParams));
    k.dispatch((N + 63) / 64, { bD, bN, bT, bC }, prm);
    dev->flush();

    const auto gD = downloadBuffer<float>(dev, bD, N * STRIDE);
    const auto gN = downloadBuffer<F4>(dev, bN, N * STRIDE);
    const auto gT = downloadBuffer<uint32_t>(dev, bT, N * STRIDE);
    const auto gC = downloadBuffer<uint32_t>(dev, bC, N);

    size_t mm = 0;
    for (uint32_t p = 0; p < N; ++p)
    {
        const PurgeRef& r = ref[p];
        if (gC[p] != r.nbTriangles) { ++mm; continue; }
        for (uint32_t i = 0; i < STRIDE; ++i)
        {
            if (asU32(gD[p * STRIDE + i]) != asU32(r.depth[i])) ++mm;
            if (gT[p * STRIDE + i] != r.temp[i]) ++mm;
            const F4 g = gN[p * STRIDE + i]; const F4 e = r.nic[i];
            if (asU32(g.x) != asU32(e.x) || asU32(g.y) != asU32(e.y) ||
                asU32(g.z) != asU32(e.z) || asU32(g.w) != asU32(e.w)) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "cvxTriPurge: %zu mismatches", mm);
    PXG_CHECK_MSG(nValid > 500 && nInvCount > 200 && nInvDepth > 200 && nPartial > 200 &&
                  nFull > 50 && nEmpty > 10 && nDeferred > 200,
                  "cvxTriPurge coverage thin (valid=%zu invC=%zu invD=%zu partial=%zu full=%zu empty=%zu deferred=%zu)",
                  nValid, nInvCount, nInvDepth, nPartial, nFull, nEmpty, nDeferred);

    for (PxgBufferHandle h : { bD, bN, bT, bC }) dev->destroyBuffer(h);
}

} // namespace
