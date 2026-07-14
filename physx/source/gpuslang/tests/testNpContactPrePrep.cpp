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

// Verification of npContactPrePrep.slang: the per-lane contact-patch
// packing core of createSolverContactConstraintDescsFromPatchWithSharedMem
// (constraintBlockPrePrep.cu L167-253) + the contact emission (L304-319)
// + mWriteback (L1044), against an exact CPU transcription.  Pure
// copies/flag logic -- ALL outputs bitwise (memcmp).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

const uint32_t MAX_PTS = 20;

struct CpIn
{
    U4 meta;   // hasPatch, wuFlags, internalFlags, materialFlags
    F4 mass;
    F4 nr;
    F4 mat;    // dynamicFriction, staticFriction, damping, restDistance
    U4 cnt;    // startContactIndex, nbContacts, prevFrictionPatchCount, forceIndex
    F4 pts[MAX_PTS];
};

struct CpOut
{
    F4 invMassOut;
    F4 nrOut;
    F4 dampDataOut;
    F4 dataOut;
    U4 counts;
    F4 pointsOut[12];
};

static float asFl(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// exact transcription (constraintBlockPrePrep.cu L167-253 + L304-319 + L1044)
static void refPack(const CpIn& s, CpOut& o)
{
    std::memset(&o, 0, sizeof(o));

    uint32_t numContacts = 0, stride = 1, contactStartIndex = 0, ptOffset = 0;
    float dynamicFriction = 0.f, staticFriction = 0.f, damping = 0.f;
    uint32_t solverFlags = 0;
    bool perPointFriction = false;

    if (s.meta.x)
    {
        perPointFriction = (s.meta.z & 16u) != 0 || (s.meta.y & (1u << 2)) != 0;
        bool disableFriction = false;
        bool isAccelerationSpring = false;

        if ((s.meta.z & 4u) == 0)  // eFORCE_NO_RESPONSE
        {
            const bool isModifiable = (s.meta.z & 128u) != 0;
            ptOffset = s.cnt.x;
            if (isModifiable)
            {
                ptOffset += s.cnt.x;
                stride = 2;
            }
            numContacts = s.cnt.y;
            contactStartIndex = s.cnt.x;

            F4 m = s.mass;
            if (s.meta.y & (1u << 14)) { m.x = 0.f; m.y = 0.f; }
            if (s.meta.y & (1u << 15)) { m.z = 0.f; m.w = 0.f; }
            o.invMassOut = m;
            o.nrOut = s.nr;
            damping = s.mat.z;
            dynamicFriction = s.mat.x;
            staticFriction = s.mat.y;
            disableFriction = (s.meta.w & 1u) != 0;
            isAccelerationSpring = (s.meta.w & 16u) != 0;
        }

        if (perPointFriction) solverFlags |= 2u;
        if (s.meta.y & (1u << 9)) solverFlags |= 1u;
        if (disableFriction) solverFlags |= 4u;
        if (isAccelerationSpring) solverFlags |= 8u;
    }

    const uint32_t packed = (numContacts & 0xFFu) | (solverFlags << 8) | ((s.cnt.z & 0xFFu) << 16);
    o.dataOut = F4{ s.mat.w, staticFriction, dynamicFriction, asFl(packed) };
    o.dampDataOut = F4{ damping, asFl(s.cnt.w + contactStartIndex), 0.f, 0.f };
    o.counts = U4{ numContacts, contactStartIndex, stride, perPointFriction ? 1u : 0u };

    uint32_t emit = 0;
    for (uint32_t a = 0; a < numContacts * stride; a += stride)
    {
        o.pointsOut[emit * 2 + 0] = s.pts[ptOffset + a];
        if (stride != 1)
            o.pointsOut[emit * 2 + 1] = s.pts[ptOffset + a + 1];
        else
            o.pointsOut[emit * 2 + 1] = F4{ 0.f, 0.f, 0.f, 3.402823466e+38f };
        ++emit;
    }
}

PXG_TEST(npContactPrePrep_pack)
{
    std::mt19937 rng(177001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CpIn> in(N);
    std::vector<CpOut> ref(N);
    size_t cNoPatch = 0, cNoResp = 0, cMod = 0, cPlain = 0, cDom0 = 0, cDom1 = 0, cPPF = 0,
           cThresh = 0, cDisFric = 0, cAccel = 0, cNoForce = 0, cEmit = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CpIn s{};
        const bool hasPatch = (i % 9 != 4);

        // work-unit flags: salt irrelevant bits + the four load-bearing ones
        uint32_t wu = (rng() % 2 ? (1u << 0) : 0u) | (rng() % 2 ? (1u << 5) : 0u) |
                      (rng() % 2 ? (1u << 8) : 0u) | (rng() % 2 ? (1u << 10) : 0u);
        if (rng() % 3 == 0) wu |= (1u << 2);    // eDISABLE_STRONG_FRICTION
        if (rng() % 2) wu |= (1u << 9);         // eFORCE_THRESHOLD
        if (rng() % 3 == 0) wu |= (1u << 14);   // eDOMINANCE_0
        if (rng() % 3 == 0) wu |= (1u << 15);   // eDOMINANCE_1

        // patch internal flags: salt irrelevant bits (1<<0, 1<<3, 1<<5, 1<<6)
        // + gates (review-177: neighbor bits of every tested mask salted)
        uint32_t internalFlags = (rng() % 2 ? 1u : 0u) | (rng() % 2 ? 8u : 0u) |
                                 (rng() % 2 ? 32u : 0u) | (rng() % 2 ? 64u : 0u);
        if (i % 7 == 3) internalFlags |= 4u;    // eFORCE_NO_RESPONSE
        if (rng() % 4 == 0) internalFlags |= 16u;   // eHAS_TARGET_VELOCITY
        if (i % 3 == 1) internalFlags |= 128u;  // eCOMPRESSED_MODIFIED_CONTACT

        uint32_t materialFlags = (rng() % 2 ? 2u : 0u) | (rng() % 2 ? 4u : 0u) |
                                 (rng() % 2 ? 8u : 0u) | (rng() % 2 ? 32u : 0u);  // salt bits
        if (rng() % 3 == 0) materialFlags |= 1u;    // eDISABLE_FRICTION
        if (rng() % 3 == 0) materialFlags |= 16u;   // eCOMPLIANT_ACCELERATION_SPRING

        s.meta = U4{ hasPatch ? 1u : 0u, wu, internalFlags, materialFlags };
        s.mass = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f) };
        s.nr = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(0.f, 1.f) };
        s.mat = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 100.f), rf(-0.1f, 0.5f) };

        const bool modifiable = (internalFlags & 128u) != 0;
        const uint32_t maxStart = modifiable ? 4u : 8u;   // keep pts in range
        const uint32_t startIdx = rng() % (maxStart + 1u);
        const uint32_t nb = rng() % 7u;                   // 0..6 contacts
        const uint32_t forceIdx = (i % 5 == 2) ? 0xFFFFFFFFu : rng() % 100000u;
        // prevFriction up to 399: pins the & 0xFF truncation (CUDA's PxU8
        // cast; review-177 x4 showed the mask was dead under < 256 fixtures)
        s.cnt = U4{ startIdx, nb, rng() % 400u, forceIdx };

        for (uint32_t p = 0; p < MAX_PTS; ++p)
            s.pts[p] = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-1, 1) };

        in[i] = s;
        refPack(s, ref[i]);

        if (!hasPatch) ++cNoPatch;
        else if (internalFlags & 4u) ++cNoResp;
        else if (modifiable) ++cMod;
        else ++cPlain;
        if (ref[i].counts.w) ++cPPF;
        if (hasPatch && (wu & (1u << 14))) ++cDom0;
        if (hasPatch && (wu & (1u << 15))) ++cDom1;
        if (wu & (1u << 9)) ++cThresh;
        if (hasPatch && !(internalFlags & 4u) && (materialFlags & 1u)) ++cDisFric;
        if (hasPatch && !(internalFlags & 4u) && (materialFlags & 16u)) ++cAccel;
        if (forceIdx == 0xFFFFFFFFu) ++cNoForce;
        cEmit += ref[i].counts.x;
    }

    std::printf("[ info ] cpp cov: nopatch %zu noresp %zu mod %zu plain %zu dom0 %zu dom1 %zu "
                "ppf %zu thresh %zu disfric %zu accel %zu noforce %zu emitted %zu\n",
                cNoPatch, cNoResp, cMod, cPlain, cDom0, cDom1, cPPF, cThresh, cDisFric, cAccel,
                cNoForce, cEmit);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npContactPrePrep.spv", "cppPack",
             { (uint32_t)sizeof(CpIn), (uint32_t)sizeof(CpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (std::memcmp(&ref[i], &g[i], sizeof(CpOut)) != 0)
        {
            if (mm < 5)
                std::printf("MM %u nR %u nG %u dataR (%g %g %g %08x) dataG (%g %g %g %08x)\n", i,
                            ref[i].counts.x, g[i].counts.x, ref[i].dataOut.x, ref[i].dataOut.y,
                            ref[i].dataOut.z, *(const uint32_t*)&ref[i].dataOut.w, g[i].dataOut.x,
                            g[i].dataOut.y, g[i].dataOut.z, *(const uint32_t*)&g[i].dataOut.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "cppPack: %zu mismatches", mm);

    bool cov = cNoPatch > 500 && cNoResp > 400 && cMod > 1000 && cPlain > 2000 && cDom0 > 800 &&
               cDom1 > 800 && cPPF > 1000 && cThresh > 2000 && cDisFric > 700 && cAccel > 700 &&
               cNoForce > 800 && cEmit > 8000;
    PXG_CHECK_MSG(cov, "cppPack: coverage thin (nopatch %zu noresp %zu mod %zu emit %zu)",
                  cNoPatch, cNoResp, cMod, cEmit);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
