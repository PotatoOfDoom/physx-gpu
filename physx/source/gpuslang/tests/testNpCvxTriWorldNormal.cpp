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

// Verification of npCvxTriWorldNormal.slang: ctCvxTriWorldNormal — getWorldNormal
// (convexMeshOutput.cu :101-112).  One thread per independent evaluation; the
// world normal is compared bit-exactly against an exact CPU transcription of
// quatRotate (PxQuat::rotate) + getNormalized (m>0 ? v*(1/sqrt(m)) : 0).  Exact-
// integer rotations (identity + 180-degree axis quaternions, integer-exact
// regardless of FMA contraction) plus general unit quaternions (matching the
// shared quatRotate helper the transform tranches already verify) keep the only
// rounding inside getNormalized, which is identical on both sides.  The
// inactive-lane path (threadIndexInManifold >= numContacts -> normal 0 ->
// rotate 0 -> m 0) must return the zero vector, not a NaN.

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

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct WnIn { F4 q, normal0; U4 cfg; };
struct WnParams { uint32_t numTests, pad0, pad1, pad2; };

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// mirrors quatRotate() in pxgSlangCommon.slang (PxQuat::rotate), same op order
static V3 quatRotateRef(F4 q, V3 v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return {
        (vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2)
    };
}

static V3 getNormalizedRef(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f) { const float r = 1.0f / std::sqrt(m); return { v.x * r, v.y * r, v.z * r }; }
    return { 0, 0, 0 };
}

static V3 worldNormalRef(F4 q, V3 normal0, uint32_t numContacts, uint32_t tIdxInManifold)
{
    const V3 normal = (tIdxInManifold >= numContacts) ? V3{ 0, 0, 0 } : normal0;
    return getNormalizedRef(quatRotateRef(q, normal));
}

PXG_TEST(npCvxTriWorldNormal_worldnormal)
{
    std::mt19937 rng(1150115);
    std::vector<WnIn> in;
    std::vector<V3> ref;

    size_t nActive = 0, nInactive = 0, nExactRot = 0, nGeneralRot = 0;

    std::uniform_real_distribution<float> uf(-4.0f, 4.0f);
    std::normal_distribution<float> gn(0, 1);

    // the exact-integer quaternions: identity + 180 deg about x/y/z
    const F4 exactQ[4] = { { 0, 0, 0, 1 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };

    auto emit = [&](F4 q, V3 normal0, uint32_t numContacts, uint32_t tIdx)
    {
        WnIn e; e.q = q; e.normal0 = { normal0.x, normal0.y, normal0.z, 0 };
        e.cfg = { numContacts, tIdx, 0, 0 };
        in.push_back(e);
        ref.push_back(worldNormalRef(q, normal0, numContacts, tIdx));
        if (tIdx >= numContacts) ++nInactive; else ++nActive;
    };

    // family A: exact-integer rotations (identity + 180 deg axis quats)
    for (uint32_t iter = 0; iter < 700; ++iter)
    {
        const F4 q = exactQ[rng() % 4];
        // integer normal so the rotated vector is integer-exact; getNormalized
        // is the sole rounding step and is identical on both sides
        V3 n = { float(int(rng() % 9) - 4), float(int(rng() % 9) - 4), float(int(rng() % 9) - 4) };
        const uint32_t numContacts = 4u + rng() % 3u;          // 4..6 (more actives)
        const uint32_t tIdx = rng() % 8u;                      // tIdx>=numContacts => inactive
        emit(q, n, numContacts, tIdx);
        ++nExactRot;
    }

    // family B: general unit quaternions (rely on the shared quatRotate helper
    // being bit-exact against its transcription, as the transform tranches show)
    for (uint32_t iter = 0; iter < 500; ++iter)
    {
        float qx = gn(rng), qy = gn(rng), qz = gn(rng), qw = gn(rng);
        const float ql = 1.0f / std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        const F4 q = { qx * ql, qy * ql, qz * ql, qw * ql };
        V3 n = { uf(rng), uf(rng), uf(rng) };
        const uint32_t numContacts = 1u + rng() % 6u;
        const uint32_t tIdx = rng() % 8u;
        emit(q, n, numContacts, tIdx);
        ++nGeneralRot;
    }

    // family C: force the zero-vector normalize branch (inactive lane) explicitly
    for (uint32_t iter = 0; iter < 150; ++iter)
    {
        const F4 q = (rng() % 2) ? exactQ[rng() % 4]
                                 : F4{ gn(rng), gn(rng), gn(rng), gn(rng) };
        V3 n = { uf(rng), uf(rng), uf(rng) };
        emit(q, n, 1u + rng() % 3u, 6u + rng() % 3u); // tIdx (6..8) >= numContacts (1..3) => inactive
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> outInit(N, { -777, -777, -777, -777 });

    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    WnParams prm = { N, 0, 0, 0 };
    Kernel k("npCvxTriWorldNormal.spv", "ctCvxTriWorldNormal",
             { (uint32_t)sizeof(WnIn), (uint32_t)sizeof(F4) }, sizeof(WnParams));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<F4>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const V3 e = ref[t]; const F4 g = gOut[t];
        if (asU32(g.x) != asU32(e.x) || asU32(g.y) != asU32(e.y) || asU32(g.z) != asU32(e.z)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cvxTriWorldNormal: %zu mismatches", mm);
    PXG_CHECK_MSG(nActive > 600 && nInactive > 200 && nExactRot > 300 && nGeneralRot > 300,
                  "cvxTriWorldNormal coverage thin (active=%zu inactive=%zu exact=%zu general=%zu)",
                  nActive, nInactive, nExactRot, nGeneralRot);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
