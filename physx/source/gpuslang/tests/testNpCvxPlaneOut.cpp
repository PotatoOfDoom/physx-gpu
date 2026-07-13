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

// Verification of npCvxPlaneOut.slang: the convexPlaneNphase_kernel_core
// output tail (cudaGJKEPA.cu L3072-3181).  The CPU reference below is an
// exact transcription of the CUDA source, including its quirks:
//   - contact/force reservations add 16*nb / 4*nb but the overflow tests add
//     only ONE element ((offset+16) > limit / (offset+4) > limit), and the two
//     tests are INDEPENDENT ifs (both flags can set in the same pass);
//   - on a valid contact offset the wrapped force offset (0xFFFFFFFF after a
//     force-only overflow) is written to the output verbatim; on contact
//     overflow BOTH output pointers become NULL (0xFFFFFFFE sentinel here,
//     distinct from the legitimate wrapped 0xFFFFFFFF);
//   - the patch rollback lives INSIDE the lane-0 reservation gate (no ballot):
//     HAS_NO_TOUCH rewrite with numPatches = 0, nbContacts = 0, touchChange =
//     previouslyHadTouch, patchChange = prevPatches != 0;
//   - insertIntoPatchStream is called unconditionally on lane 0 (even with
//     nbContacts = 0 and patchIndex = 0xFFFFFFFF) with the world normal
//     planeTransform.rotate((1,0,0)) -- localNormal is never reassigned in
//     the plane kernel (L2825).
// One pair per test.  Integer outputs exact; the two rotated float outputs
// (stream points, patch normal) to tight tolerance.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct F4 { float x, y, z, w; };
struct PlaneOutIn { F4 planeTq, planeTp; F4 p[32]; U4 misc, cfg, counters; };
struct PlaneOutOut { F4 stream[4]; U4 counters, outputs, flags; F4 patchArgs; };

static const uint32_t NULL_SENTINEL = 0xFFFFFFFEu;

static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }

// PxQuat::rotate, exact form
static void quatRotate(const F4& q, float vx, float vy, float vz, float* o)
{
    const float x = 2.f * vx, y = 2.f * vy, z = 2.f * vz;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * x + q.y * y + q.z * z;
    o[0] = x * w2 + (q.y * z - q.z * y) * q.w + q.x * d2;
    o[1] = y * w2 + (q.z * x - q.x * z) * q.w + q.y * d2;
    o[2] = z * w2 + (q.x * y - q.y * x) * q.w + q.z * d2;
}

// exact transcription of cudaGJKEPA.cu L3072-3181
static void refPlaneOut(const PlaneOutIn& s, PlaneOutOut& o)
{
    const uint32_t mask = s.misc.x;
    const uint32_t nbContacts = s.misc.y;
    const uint32_t allflags = s.misc.z;

    uint32_t contactsBytes = s.counters.x, forceBytes = s.counters.y, patchesBytes = s.counters.z;
    uint32_t ovf = 0;
    for (int j = 0; j < 4; ++j) o.stream[j] = F4{ 0, 0, 0, 0 };

    // L3072-3091: reservation, single-element overflow tests, independent ifs
    uint32_t contactByteOffset = 0xFFFFFFFFu;
    uint32_t forceByteOffset = 0xFFFFFFFFu;
    if (nbContacts)
    {
        contactByteOffset = contactsBytes;
        contactsBytes += 16u * nbContacts;      // sizeof(PxContact) * nb
        forceByteOffset = forceBytes;
        forceBytes += 4u * nbContacts;          // sizeof(PxU32) * nb
        if ((contactByteOffset + 16u) > s.cfg.x) { ovf |= 1u; contactByteOffset = 0xFFFFFFFFu; }
        if ((forceByteOffset + 4u) > s.cfg.y) { ovf |= 2u; forceByteOffset = 0xFFFFFFFFu; }
    }

    // L3099-3117: world-space stream write / output pointers
    uint32_t outForces, outPoints;
    if (contactByteOffset != 0xFFFFFFFFu)
    {
        for (uint32_t L = 0; L < 32; ++L)
            if (mask & (1u << L))
            {
                const uint32_t index = popc(mask & ((1u << L) - 1u)); // warpScanExclusive
                float w[3];
                quatRotate(s.planeTq, s.p[L].x, s.p[L].y, s.p[L].z, w);
                o.stream[(contactByteOffset - s.counters.x) / 16u + index] =
                    F4{ w[0] + s.planeTp.x, w[1] + s.planeTp.y, w[2] + s.planeTp.z, s.p[L].w };
            }
        outForces = forceByteOffset;   // may be the wrapped 0xFFFFFFFF -- ported exactly
        outPoints = contactByteOffset;
    }
    else
    {
        outForces = NULL_SENTINEL; // NULL
        outPoints = NULL_SENTINEL;
    }

    // L3119-3148: status flags / touch / patch change / allflags rewrite
    const uint32_t oldStatusFlags = (allflags >> 16) & 0xFF;
    uint32_t status = oldStatusFlags & ~3u;                   // clear eTOUCH_KNOWN
    status |= nbContacts ? 2u : 1u;                           // eHAS_TOUCH : eHAS_NO_TOUCH

    const uint32_t prevPatches = (allflags & 0xFFFF) >> 8;
    const bool previouslyHadTouch = (oldStatusFlags & 2u) != 0;
    const bool prevTouchKnown = (oldStatusFlags & 3u) != 0;
    const uint32_t numPatches = (nbContacts > 0) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0;

    uint32_t touchChange = ((previouslyHadTouch != currentlyHasTouch) || !prevTouchKnown) ? 1u : 0u;
    uint32_t patchChange = (prevPatches != numPatches) ? 1u : 0u;
    uint32_t allflagsOut = (((prevPatches << 8) | status) << 16) | (numPatches << 8);
    uint32_t nbOut = nbContacts;

    // L3151-3177: patch reservation with in-gate rollback
    uint32_t patchIndex = 0xFFFFFFFFu;
    uint32_t cpOut = NULL_SENTINEL;
    if (nbContacts)
    {
        patchIndex = patchesBytes;
        patchesBytes += 64u; // sizeof(PxContactPatch)
        if ((patchIndex + 64u) > s.cfg.z)
        {
            ovf |= 4u;
            patchIndex = 0xFFFFFFFFu;
            status = (status & ~3u) | 1u;
            allflagsOut = ((prevPatches << 8) | status) << 16; // numPatches = 0
            nbOut = 0;
            touchChange = previouslyHadTouch ? 1u : 0u;
            patchChange = (prevPatches != 0) ? 1u : 0u;
        }
        else
            cpOut = patchIndex;
    }

    // L3179-3180: insertIntoPatchStream args (world normal, nbContacts)
    float wn[3];
    quatRotate(s.planeTq, 1.f, 0.f, 0.f, wn);

    o.counters = U4{ contactsBytes, forceBytes, patchesBytes, ovf };
    o.outputs = U4{ outForces, outPoints, cpOut, nbOut };
    o.flags = U4{ touchChange, patchChange, allflagsOut, patchIndex };
    o.patchArgs = F4{ wn[0], wn[1], wn[2], float(nbContacts) };
}

static bool eqF(float a, float b) { return almostEqual(a, b, 1e-6f, 1e-6f); }

PXG_TEST(npCvxPlaneOut_outputTail)
{
    std::mt19937 rng(155401);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<PlaneOutIn> in(N);
    std::vector<PlaneOutOut> ref(N);
    size_t cOvfC = 0, cOvfF = 0, cOvfP = 0, cBoth = 0, cWrapForce = 0, cEmpty = 0, cExactC = 0, cExactP = 0, cSparse = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PlaneOutIn s{};

        // random unit quaternion + position
        float q[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { q[k] = rf(-1, 1); m2 += q[k] * q[k]; }
        const float inv = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        s.planeTq = F4{ q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv };
        s.planeTp = F4{ rf(-10, 10), rf(-10, 10), rf(-10, 10), 0 };

        for (int L = 0; L < 32; ++L)
            s.p[L] = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-1, 1) };

        // nbContacts 0..4; mask contiguous (no-run path) or sparse (reduce path)
        const uint32_t nb = (i % 11 == 0) ? 0u : (1u + rng() % 4);
        uint32_t mask;
        if (nb == 0) mask = 0;
        else if (rng() & 1u) mask = (1u << nb) - 1u; // contiguous
        else
        {
            mask = 0; ++cSparse;
            while (popc(mask) < nb) mask |= 1u << (rng() % 32); // sparse
        }
        const uint32_t mats = (rng() & 0xFFFFu) | ((rng() & 0xFFFFu) << 16);
        s.misc = U4{ mask, nb, uint32_t(rng()), mats }; // fully random allflags word

        // initial counters and limits per buffer: ample / tight / exact-single-element
        const uint32_t initC = 16 * (rng() % 64), initF = 4 * (rng() % 64), initP = 64 * (rng() % 32);
        uint32_t limC = initC + 16 * 40, limF = initF + 4 * 40, limP = initP + 64 * 8;
        const uint32_t vC = rng() % 5, vF = rng() % 5, vP = rng() % 5;
        if (vC == 1) limC = initC + rng() % 16;            // (initC + 16) > lim -> overflow
        else if (vC == 2) { limC = initC + 16; ++cExactC; }  // exact single-element fit: NO overflow even for nb > 1 (quirk)
        if (vF == 1) limF = initF + rng() % 4;
        else if (vF == 2) limF = initF + 4;
        if (vP == 1) limP = initP + rng() % 64;            // (initP + 64) > lim -> rollback
        else if (vP == 2) { limP = initP + 64; ++cExactP; }
        s.cfg = U4{ limC, limF, limP, 0 };
        s.counters = U4{ initC, initF, initP, 0 };

        in[i] = s;
        refPlaneOut(s, ref[i]);
        const uint32_t f = ref[i].counters.w;
        if (f & 1u) ++cOvfC;
        if (f & 2u) ++cOvfF;
        if (f & 4u) ++cOvfP;
        if ((f & 3u) == 3u) ++cBoth;
        if ((f & 3u) == 2u) ++cWrapForce; // force-only: wrapped 0xFFFFFFFF written to outputs.x
        if (nb == 0) ++cEmpty;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PlaneOutOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct PlaneOutParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxPlaneOut.spv", "cvxPlaneOut",
             { (uint32_t)sizeof(PlaneOutIn), (uint32_t)sizeof(PlaneOutOut) }, sizeof(PlaneOutParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PlaneOutOut>(dev, bOut, N);
    size_t mmS = 0, mmC = 0, mmO = 0, mmFl = 0, mmA = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            const F4& rs = ref[i].stream[j]; const F4& gs = g[i].stream[j];
            if (!eqF(rs.x, gs.x) || !eqF(rs.y, gs.y) || !eqF(rs.z, gs.z) || !eqF(rs.w, gs.w)) { ++mmS; break; }
        }
        const U4& rc = ref[i].counters; const U4& gc = g[i].counters;
        if (rc.x != gc.x || rc.y != gc.y || rc.z != gc.z || rc.w != gc.w) ++mmC;
        const U4& ro = ref[i].outputs; const U4& go = g[i].outputs;
        if (ro.x != go.x || ro.y != go.y || ro.z != go.z || ro.w != go.w) ++mmO;
        const U4& rl = ref[i].flags; const U4& gl = g[i].flags;
        if (rl.x != gl.x || rl.y != gl.y || rl.z != gl.z || rl.w != gl.w) ++mmFl;
        const F4& ra = ref[i].patchArgs; const F4& ga = g[i].patchArgs;
        if (!eqF(ra.x, ga.x) || !eqF(ra.y, ga.y) || !eqF(ra.z, ga.z) || ra.w != ga.w) ++mmA;
    }
    PXG_CHECK_MSG(mmS == 0, "planeOut: %zu contact-stream mismatches", mmS);
    PXG_CHECK_MSG(mmC == 0, "planeOut: %zu counter mismatches", mmC);
    PXG_CHECK_MSG(mmO == 0, "planeOut: %zu output-pointer mismatches", mmO);
    PXG_CHECK_MSG(mmFl == 0, "planeOut: %zu flag mismatches", mmFl);
    PXG_CHECK_MSG(mmA == 0, "planeOut: %zu patch-arg mismatches", mmA);
    PXG_CHECK_MSG(cOvfC > N / 20 && cOvfF > N / 20 && cOvfP > N / 20 && cBoth > N / 100 &&
                  cWrapForce > N / 100 && cEmpty > N / 20 && cExactC > N / 12 && cExactP > N / 12 && cSparse > N / 8,
                  "planeOut: coverage thin (ovfC %zu ovfF %zu ovfP %zu both %zu wrapF %zu empty %zu exactC %zu exactP %zu sparse %zu)",
                  cOvfC, cOvfF, cOvfP, cBoth, cWrapForce, cEmpty, cExactC, cExactP, cSparse);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
