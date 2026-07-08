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

// Verification of solverInitVelocitiesTGS.slang: initializeSolverVelocitiesTGS.
// A synthetic set of pre-solve body velocities + per-(body,slab) remap
// references (with ~15% invalid) is seeded; the kernel's scatter of the
// repacked vel0/vel1/vel2 triples into the remapped slab slots and the
// accumulation-offset copy are compared bit-exactly against a C++ reference.
// Remap targets are spaced 96 apart so the [r, r+32, r+64] triples never alias,
// and the output-offset region sits above them, so a full float4 comparison of
// the whole body-velocity buffer (touched slots vs preserved sentinel) is exact.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct InitVelParams { uint32_t numSlabs, allBodiesCount, bodyOffset, outputOffset, numThreadsPerBody; };
struct F4 { float x, y, z, w; };

const uint32_t INVALID = 0xffffffffu;
const float SENT = -9.0f;

uint32_t nextPow2(uint32_t v) { uint32_t p = 1; while (p < v) p <<= 1; return p; }

PXG_TEST(solverInitVelocitiesTGS_scatter)
{
    std::mt19937 rng(36010);
    const uint32_t numSlabs = 4;
    const uint32_t allBodiesCount = 200;
    const uint32_t bodyOffset = 40;
    const uint32_t numThreadsPerBody = std::min(nextPow2(numSlabs), 32u); // 4
    const uint32_t numBodiesPerWarp = 32u / numThreadsPerBody;            // 8
    const uint32_t outputOffset = 70000;
    const uint32_t bodyVelElems = 72000;

    // input velocities: lin at [bodyId], ang at [bodyId + allBodiesCount]
    std::vector<F4> outVel(2 * allBodiesCount);
    auto rf = [&]() { return float(int(rng() % 20000) - 10000) * 0.01f; };
    for (auto& v : outVel) v = { rf(), rf(), rf(), rf() };

    // references: one uint per (body,slab) for dynamic bodies [bodyOffset, allBodiesCount)
    const uint32_t nPairs = numSlabs * (allBodiesCount - bodyOffset);
    std::vector<uint32_t> refs(nPairs);
    for (uint32_t i = 0; i < nPairs; ++i)
        refs[i] = (rng() % 100 < 15) ? INVALID : (96u * i); // spaced so triples never alias

    std::vector<F4> bodyVelInit(bodyVelElems, F4{ SENT, SENT, SENT, SENT });

    // ---- C++ reference ----
    std::vector<F4> refBody = bodyVelInit;
    auto vpack = [&](uint32_t bodyId, F4& v0, F4& v1, F4& v2) {
        const F4 lin = outVel[bodyId];
        const F4 ang = outVel[bodyId + allBodiesCount];
        v0 = { lin.x, lin.y, lin.z, ang.x };
        v1 = { ang.y, ang.z, 0.f, 0.f };
        v2 = { 0.f, 0.f, 0.f, 0.f };
    };
    for (uint32_t bodyId = 0; bodyId < allBodiesCount; ++bodyId)
    {
        F4 v0, v1, v2; vpack(bodyId, v0, v1, v2);
        if (bodyId >= bodyOffset)
        {
            for (uint32_t b = 0; b < numSlabs; ++b)
            {
                const uint32_t r = refs[numSlabs * (bodyId - bodyOffset) + b];
                if (r != INVALID) { refBody[r] = v0; refBody[r + 32] = v1; refBody[r + 64] = v2; }
            }
        }
        refBody[outputOffset + bodyId] = v0;
        refBody[outputOffset + bodyId + allBodiesCount] = v1;
        refBody[outputOffset + bodyId + 2 * allBodiesCount] = v2;
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t numWarps = (allBodiesCount + numBodiesPerWarp - 1) / numBodiesPerWarp; // 25

    PxgBufferHandle bRefs = createBufferFrom(dev, refs);
    PxgBufferHandle bBody = createBufferFrom(dev, bodyVelInit);
    PxgBufferHandle bOut = createBufferFrom(dev, outVel);

    Kernel k("solverInitVelocitiesTGS.spv", "initializeSolverVelocitiesTGS", { 4, 16, 16 }, sizeof(InitVelParams));
    InitVelParams p = { numSlabs, allBodiesCount, bodyOffset, outputOffset, numThreadsPerBody };
    k.dispatch(numWarps, { bRefs, bBody, bOut }, p);
    dev->flush();

    const auto gBody = downloadBuffer<F4>(dev, bBody, bodyVelElems);
    size_t mm = 0;
    for (uint32_t e = 0; e < bodyVelElems; ++e)
    {
        const F4& a = gBody[e];
        const F4& r = refBody[e];
        if (a.x != r.x || a.y != r.y || a.z != r.z || a.w != r.w) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "initializeSolverVelocitiesTGS: %zu mismatches", mm);

    dev->destroyBuffer(bRefs);
    dev->destroyBuffer(bBody);
    dev->destroyBuffer(bOut);
}

} // namespace
