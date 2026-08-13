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

// P8.90: pxgSlangArticulationDynamic.slang — the impulse-response pair.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// `Dy::FeatherstoneArticulation::translateSpatialVector` is PX_CUDA_CALLABLE
// (DyFeatherstoneArticulation.h:828), and so are `innerProduct`, `operator*`,
// `operator+=` and `cross`.  The expectation below is therefore the engine's own
// arithmetic wrapped in the CUDA's six-line structure — the same standard the
// rest of this chain holds itself to, and a STRONGER reference than the Phase-1
// fixtures these bodies were lifted from, which compare against a host
// transcription.
//
// ===========================================================================
// WHAT THE CASES ARE FOR
// ===========================================================================
//
// Every trap the module header names gets inputs that can see it:
//
//   * dof counts 0, 1, 2 and 3 — 0 makes both functions pure translations, and
//     the `jDelta` three-lane dot must still run over zeroed lanes;
//   * a `qstZ` seed that is NOT zero, so "accumulates" is distinguishable from
//     "assigns" (the CUDA's own caller seeds zeros, which cannot tell them apart);
//   * `QMinusSTZ` lanes at and beyond `dofCount` set to LOUD values, so a port
//     that bounds the `jDelta` dot by `dofCount` — right today, wrong later —
//     produces a different number here;
//   * a zero offset stripe, where `translateSpatialVector` is the identity and
//     the sign of the negation cannot be seen, beside a majority where it can;
//   * motion matrices whose top and bottom halves differ in magnitude, so the
//     dual `innerProduct` and the Euclidean `dot` cannot coincide.

#include "pxgTestFramework.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulation.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxReal;
using physx::PxU32;
using physx::PxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::Dy::FeatherstoneArticulation;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }

enum Op : uint32_t { AD_OP_IMPULSE = 0, AD_OP_ACCEL = 1, AD_NUM_OPS = 2 };

struct AdCase
{
    F4 vTop, vBot;
    F4 off;
    F4 mmTop[3], mmBot[3];
    F4 aTop[3], aBot[3];
    F4 invS[3];
    F4 qIn;
    U4 cfg;
};

struct AdOut
{
    F4 rTop, rBot;
    F4 qOut;
};

static PxVec3 v3(const F4& f) { return PxVec3(f.x, f.y, f.z); }
static void put3(F4& d, const PxVec3& v) { d.x = v.x; d.y = v.y; d.z = v.z; }

// ---------------------------------------------------------------------------
// The two mirrors: the CUDA's structure, the engine's arithmetic
// ---------------------------------------------------------------------------
static UnAlignedSpatialVector refPropagateImpulseW(const UnAlignedSpatialVector* isInvD,
                                                   const PxVec3& childToParent,
                                                   const UnAlignedSpatialVector* motionMatrix,
                                                   const UnAlignedSpatialVector& Z, PxU32 dofCount,
                                                   PxReal* qstZ)
{
    UnAlignedSpatialVector temp = Z;
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxReal stZ = -motionMatrix[ind].innerProduct(Z);
            qstZ[ind] += stZ;
            temp += isInvD[ind] * stZ;
        }
    }
    return FeatherstoneArticulation::translateSpatialVector(childToParent, temp);
}

static UnAlignedSpatialVector refPropagateAccelerationW(
    const PxVec3& parentToChild, const PxVec3* invStISW,
    const UnAlignedSpatialVector* motionMatrix,
    const UnAlignedSpatialVector& parentLinkAccelerationW, PxU32 dofCount,
    const UnAlignedSpatialVector* IsW, const PxReal* QMinusSTZ)
{
    UnAlignedSpatialVector pDeltaV =
        FeatherstoneArticulation::translateSpatialVector(-parentToChild, parentLinkAccelerationW);

    PxReal tJointDelta[3] = { 0.f, 0.f, 0.f };
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxReal temp = IsW[ind].innerProduct(pDeltaV);
            tJointDelta[ind] = (QMinusSTZ[ind] - temp);
        }
    }

    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxVec3 iStIsTi = invStISW[ind];
            const PxReal jDelta = iStIsTi.x * tJointDelta[0] + iStIsTi.y * tJointDelta[1]
                                + iStIsTi.z * tJointDelta[2];
            pDeltaV += motionMatrix[ind] * jDelta;
        }
    }
    return pDeltaV;
}

PXG_TEST(artiDynamic)
{
    const uint32_t CASES_PER_OP = 257;   // prime, coprime with every stripe below
    const uint32_t N = AD_NUM_OPS * CASES_PER_OP;

    // non-round and distinct per case and per slot
    auto val = [](uint32_t i, uint32_t k) {
        return (float((i * 37u + k * 5u) % 1009u) + 1.0f) * 0.013671875f - 6.5f;
    };

    std::vector<AdCase> in(N);
    std::vector<AdOut> poison(N);

    uint32_t seenDof[4] = { 0, 0, 0, 0 };
    size_t zeroOff = 0, loudTail = 0, nonZeroSeed = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        AdCase c{};
        c.cfg.y = i % AD_NUM_OPS;
        // dof counts 0..3, every one reached for both opcodes
        c.cfg.x = (i / AD_NUM_OPS) % 4u;
        seenDof[c.cfg.x] += 1u;

        c.vTop = F4{ val(i, 0), val(i, 1), val(i, 2), 0.f };
        c.vBot = F4{ val(i, 3) * 4.0f, val(i, 4) * 4.0f, val(i, 5) * 4.0f, 0.f };
        c.off = F4{ val(i, 6), val(i, 7), val(i, 8), 0.f };

        for (uint32_t k = 0; k < 3; ++k)
        {
            // the two halves differ in magnitude, so the dual product and the
            // Euclidean one cannot coincide
            c.mmTop[k] = F4{ val(i, 10 + k * 3 + 0), val(i, 10 + k * 3 + 1), val(i, 10 + k * 3 + 2), 0.f };
            c.mmBot[k] = F4{ val(i, 20 + k * 3 + 0) * 7.0f, val(i, 20 + k * 3 + 1) * 7.0f,
                             val(i, 20 + k * 3 + 2) * 7.0f, 0.f };
            c.aTop[k] = F4{ val(i, 30 + k * 3 + 0), val(i, 30 + k * 3 + 1), val(i, 30 + k * 3 + 2), 0.f };
            c.aBot[k] = F4{ val(i, 40 + k * 3 + 0) * 3.0f, val(i, 40 + k * 3 + 1) * 3.0f,
                            val(i, 40 + k * 3 + 2) * 3.0f, 0.f };
            c.invS[k] = F4{ val(i, 50 + k * 3 + 0), val(i, 50 + k * 3 + 1), val(i, 50 + k * 3 + 2), 0.f };
        }

        // a NON-ZERO seed: the CUDA's own caller passes {0,0,0}, which cannot tell
        // "accumulates" from "assigns".  The tail lanes are LOUD so a jDelta dot
        // bounded by dofCount reads differently.
        c.qIn = F4{ val(i, 60), val(i, 61), val(i, 62), 0.f };
        ++nonZeroSeed;
        if (c.cfg.x < 3)
        {
            (&c.qIn.x)[2] = 1000.0f + float(i % 17);
            ++loudTail;
        }

        if (i % 13 == 4)
        {
            // the offset is exactly zero: translateSpatialVector is the identity
            // and the negation's sign is invisible.  The other 12/13 can see it.
            c.off = F4{ 0.f, 0.f, 0.f, 0.f };
            ++zeroOff;
        }

        in[i] = c;

        AdOut p{};
        uint32_t* pw = reinterpret_cast<uint32_t*>(&p);
        for (uint32_t k = 0; k < sizeof(AdOut) / 4; ++k)
            pw[k] = 0xAB000000u | ((i & 0xFFFFu) << 8) | (k & 0xFFu);
        poison[i] = p;
    }

    std::printf("[ info ] ad dofs: %u/%u/%u/%u  zeroOff %zu loudTail %zu seeds %zu\n", seenDof[0],
                seenDof[1], seenDof[2], seenDof[3], zeroOff, loudTail, nonZeroSeed);
    for (uint32_t d = 0; d < 4; ++d)
        PXG_CHECK_MSG(seenDof[d] > 100, "dof count %u occurs only %u times", d, seenDof[d]);
    PXG_CHECK(zeroOff > 20);
    PXG_CHECK(loudTail > 300);

    // ---- the expectation ----
    std::vector<AdOut> ref = poison;
    for (uint32_t i = 0; i < N; ++i)
    {
        const AdCase& c = in[i];
        AdOut& o = ref[i];

        UnAlignedSpatialVector mm[3], a[3];
        PxVec3 invS[3];
        for (uint32_t k = 0; k < 3; ++k)
        {
            mm[k] = UnAlignedSpatialVector(v3(c.mmTop[k]), v3(c.mmBot[k]));
            a[k] = UnAlignedSpatialVector(v3(c.aTop[k]), v3(c.aBot[k]));
            invS[k] = v3(c.invS[k]);
        }
        const UnAlignedSpatialVector v(v3(c.vTop), v3(c.vBot));
        PxReal q[3] = { c.qIn.x, c.qIn.y, c.qIn.z };

        if (c.cfg.y == AD_OP_IMPULSE)
        {
            const UnAlignedSpatialVector r =
                refPropagateImpulseW(a, v3(c.off), mm, v, c.cfg.x, q);
            put3(o.rTop, r.top);
            put3(o.rBot, r.bottom);
            put3(o.qOut, PxVec3(q[0], q[1], q[2]));
        }
        else
        {
            const UnAlignedSpatialVector r =
                refPropagateAccelerationW(v3(c.off), invS, mm, v, c.cfg.x, a, q);
            put3(o.rTop, r.top);
            put3(o.rBot, r.bottom);
            // qOut stays POISON: the acceleration direction produces none
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, poison);

    struct P { uint32_t numCases; } prm = { N };
    Kernel k("npArtiDynamic.spv", "artiDynamicOps",
             { (uint32_t)sizeof(AdCase), (uint32_t)sizeof(AdOut) }, sizeof(P));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto got = downloadBuffer<AdOut>(dev, bOut, N);

    const size_t WORDS = sizeof(AdOut) / 4;
    size_t bad = 0, signedZeros = 0, badImpulse = 0, badAccel = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* r = reinterpret_cast<const float*>(&ref[i]);
        const float* g = reinterpret_cast<const float*>(&got[i]);
        bool caseBad = false;
        for (size_t x = 0; x < WORDS; ++x)
        {
            if (bitEq(r[x], g[x]))
                continue;
            if (r[x] == 0.f && g[x] == 0.f)
            {
                ++signedZeros;
                continue;
            }
            if (bad < 12)
                std::printf("MM case %u op %u dof %u word %zu  ref %.9g (%08x)  gpu %.9g (%08x)\n",
                            i, in[i].cfg.y, in[i].cfg.x, x, r[x], f2u(r[x]), g[x], f2u(g[x]));
            caseBad = true;
            ++bad;
        }
        if (caseBad)
            (in[i].cfg.y == AD_OP_IMPULSE ? badImpulse : badAccel) += 1;
    }

    if (bad)
        std::printf("[ info ] impulse cases bad %zu, accel cases bad %zu\n", badImpulse, badAccel);
    std::printf("[ info ] ad %u cases x %zu words, +-0 tolerated %zu\n", N, WORDS, signedZeros);
    PXG_CHECK_MSG(bad == 0, "the impulse-response pair matches the engine bit for bit");
}

}  // namespace
