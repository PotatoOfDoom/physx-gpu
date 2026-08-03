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

// P8.32 — delayGenerateContact (deferContactBda.spv / pxgSlangDeferContact).
//
// Suite (ctest filter "deferContact"):
//   deferContactPredicate — the two sentinel feature indices, the negation, the
//                           strict `<`, and the threshold, over a swept angle.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;

struct DeferPush { uint64_t minNormal, triNormal, feature, outFlags; uint32_t count, pad36; };
static_assert(sizeof(DeferPush) == 40, "defer blob must be 40 bytes");

static const uint32_t TRI_FEATURE_IDX  = 0x7FFFFFFFu;
static const uint32_t EDGE_FEATURE_IDX = 0x80000000u;
static const float THRESH = 0.707106781f;

// convexTriangle.cuh:1402, verbatim.
static bool cpuDefer(float mn[3], float tn[3], uint32_t feature)
{
    if (feature != EDGE_FEATURE_IDX && feature != TRI_FEATURE_IDX)
    {
        const float t[3] = { -tn[0], -tn[1], -tn[2] };
        const float cosTheta = mn[0]*t[0] + mn[1]*t[1] + mn[2]*t[2];
        return cosTheta < THRESH;
    }
    return false;
}

PXG_TEST(deferContactPredicate)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<float> mns, tns;
    std::vector<uint32_t> feats;

    // The triangle normal is fixed at +y; minNormal sweeps the angle against its
    // NEGATION, which is what the predicate actually measures.  Sweeping the full
    // circle covers both sides of the 45-degree bar many times over.
    const float tn[3] = { 0.0f, 1.0f, 0.0f };
    // Feature indices exercised at every angle: a plain hull-face index (defers
    // when the angle says so) and BOTH sentinels (never defer, whatever the
    // angle).  The two sentinels are 0x7FFFFFFF and 0x80000000 -- adjacent as
    // unsigned values but differing in the top bit, so a port that tested only
    // one, or that used a range check, fails on the other.
    const uint32_t features[] = { 0u, 7u, 12345u, TRI_FEATURE_IDX, EDGE_FEATURE_IDX, 0x7FFFFFFEu, 0x80000001u };

    for (int deg = 0; deg < 360; ++deg)
    {
        const float r = float(deg) * 3.14159265358979f / 180.0f;
        // In the x/y plane, so the dot with (0,-1,0) is -sin(r) and every angle
        // is reachable.
        const float mn[3] = { std::cos(r), std::sin(r), 0.0f };
        for (uint32_t f : features)
        {
            mns.push_back(mn[0]); mns.push_back(mn[1]); mns.push_back(mn[2]); mns.push_back(0.0f);
            tns.push_back(tn[0]); tns.push_back(tn[1]); tns.push_back(tn[2]); tns.push_back(0.0f);
            feats.push_back(f);
        }
    }
    // The exact boundary: cosTheta == THRESH must NOT defer (`<`, not `<=`).
    // minNormal = (sin, -cos) against -tn = (0,-1,0) gives cosTheta = cos exactly.
    {
        const float c = THRESH;
        const float sMag = std::sqrt(1.0f - c * c);
        const float mn[3] = { sMag, -c, 0.0f };
        mns.push_back(mn[0]); mns.push_back(mn[1]); mns.push_back(mn[2]); mns.push_back(0.0f);
        tns.push_back(tn[0]); tns.push_back(tn[1]); tns.push_back(tn[2]); tns.push_back(0.0f);
        feats.push_back(3u);
        // ... and a hair below it, which must defer.
        const float c2 = std::nextafterf(THRESH, 0.0f);
        const float s2 = std::sqrt(1.0f - c2 * c2);
        mns.push_back(s2); mns.push_back(-c2); mns.push_back(0.0f); mns.push_back(0.0f);
        tns.push_back(tn[0]); tns.push_back(tn[1]); tns.push_back(tn[2]); tns.push_back(0.0f);
        feats.push_back(3u);
    }

    const uint32_t N = uint32_t(feats.size());
    std::vector<uint32_t> outInit(N, 0xFFFFFFFFu);

    PxgBufferHandle bM = createBufferFrom(dev, mns);
    PxgBufferHandle bT = createBufferFrom(dev, tns);
    PxgBufferHandle bF = createBufferFrom(dev, feats);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);

    DeferPush push{};
    push.minNormal = dev->getBufferDeviceAddress(bM);
    push.triNormal = dev->getBufferDeviceAddress(bT);
    push.feature   = dev->getBufferDeviceAddress(bF);
    push.outFlags  = dev->getBufferDeviceAddress(bO);
    push.count = N;

    Kernel k("deferContactBda.spv", "deferContactLaunch", {}, sizeof(DeferPush));
    k.dispatch((N + 63u) / 64u, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<uint32_t>(dev, bO, N);

    size_t mm = 0, nDefer = 0, nKeep = 0, nSentinel = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float mn[3] = { mns[4*i], mns[4*i+1], mns[4*i+2] };
        float tn2[3] = { tns[4*i], tns[4*i+1], tns[4*i+2] };
        const uint32_t exp = cpuDefer(mn, tn2, feats[i]) ? 1u : 0u;
        if (got[i] != exp)
        {
            ++mm;
            if (mm <= 8)
                std::printf("  case %u (feature=0x%08X): got=%u exp=%u\n", i, feats[i], got[i], exp);
        }
        if (feats[i] == TRI_FEATURE_IDX || feats[i] == EDGE_FEATURE_IDX) ++nSentinel;
        else if (exp) ++nDefer; else ++nKeep;
    }

    // The sweep must actually reach both outcomes on non-sentinel features, and
    // must actually exercise the sentinels -- otherwise it proves less than it
    // appears to.
    size_t vacuous = 0;
    if (nDefer == 0) { ++vacuous; std::printf("  no case defers -- the sweep is vacuous\n"); }
    if (nKeep == 0)  { ++vacuous; std::printf("  no case keeps -- the sweep is vacuous\n"); }
    if (nSentinel == 0) { ++vacuous; std::printf("  the sentinels are never exercised\n"); }

    PXG_CHECK_MSG(mm == 0, "delayGenerateContact matches CPU over %u cases (%zu mismatches)", N, mm);
    PXG_CHECK_MSG(vacuous == 0, "the sweep reaches every outcome (%zu gaps)", vacuous);

    for (PxgBufferHandle h : { bM, bT, bF, bO }) dev->destroyBuffer(h);
}

} // namespace
