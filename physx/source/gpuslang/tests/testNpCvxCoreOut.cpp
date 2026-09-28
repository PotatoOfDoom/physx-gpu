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

// Verification of npCvxCoreOut.slang: the two TestOutput::writeOutput
// overloads of convexCoreCollision.cu (L177-445).  Exact CPU transcription of
// both: the single-element 64-bit-promoted overflow tests, the wrapped-vs-NULL
// output pointer quirks, the allflags/touch/patch-change rewrites, the patch
// rollbacks, the patch-major sequential contact stream of the Gu::Contact
// variant with its numPoints-vs-numPatches gate split, and the inline
// PxContactPatch array write over the tranche-43 combineMaterials core.
// ALL float outputs compare bitwise (pure copies plus the combineScalars
// arithmetic executed identically on both sides).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct CoreOutIn
{
    F4 contacts[4];
    F4 normal;
    U4 misc;   // x = nbContacts (array), y = allflags, z = mNumPatches, w = mNumPoints
    U4 cfg;    // limits
    U4 counters;
    U4 mat;    // materialIndex0/1
    F4 mat0F, mat1F; // dynFriction, staFriction, restitution, damping
    U4 mat0U, mat1U; // flags, fricMode, restMode, dampMode
    U4 patchCounts[4];
    F4 patchN[16];
    F4 points[64];
};
struct CoreOutOut
{
    F4 stream[64];
    U4 counters;
    U4 outputs;
    U4 flags;
    F4 patchArgs;
    U4 patchImg[64];
};

const uint32_t HAS_NO_TOUCH = 1, HAS_TOUCH = 2, TOUCH_KNOWN = 3;
const uint32_t OVF_CONTACT = 1, OVF_FORCE = 2, OVF_PATCH = 4;
const uint32_t NULL_SENTINEL = 0xFFFFFFFEu;
const uint32_t MODE_MIN = 1, MODE_MULTIPLY = 2, MODE_MAX = 3;
const uint32_t FLAG_DISABLE_FRICTION = 1, FLAG_DISABLE_STRONG_FRICTION = 2, FLAG_COMPLIANT_ACC = 16;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// P9.86e: eMIN/eMAX are PxMin/PxMax<float> = intrinsics::selectMin/selectMax (PxsMaterialCombiner.h:43/47/60/64
// -> PxMath.h:81-101 -> PxUnixMathIntrinsics.h:116-125): on a +-0 tie the SECOND operand survives.  The oracle
// spells the selects (std::fmin/std::fmax are not PxMin/PxMax: their zero-tie sign is libm dependent).
// gCombineVariant is a mutant twin replaying a reverted port: 1 = eMIN as GLSL FMin, 2 = eMAX as FMax, with
// llvmpipe's zero order under the SZINP stamp (FMin(+-0,-+0) = -0, FMax = +0, either operand order).
static int gCombineVariant = 0;
static float fminLp(float a, float b) { return (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); }
static float fmaxLp(float a, float b) { return (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); }
static float combineScalars(float a, float b, uint32_t mode)
{
    if (mode == 0) return 0.5f * (a + b);
    if (mode == MODE_MIN) return (gCombineVariant == 1) ? fminLp(a, b) : ((a < b) ? a : b);   // PxMin = selectMin
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return (gCombineVariant == 2) ? fmaxLp(a, b) : ((a > b) ? a : b);   // PxMax = selectMax
    return 0.0f;
}
static void combineMats(const CoreOutIn& s, float& sta, float& dyn, float& rest, float& damp, uint32_t& flags)
{
    const float r0 = s.mat0F.z, r1 = s.mat1F.z;
    const bool c0 = r0 < 0.f, c1 = r1 < 0.f;
    const bool oneC = c0 != c1, bothC = c0 && c1;
    const bool a0 = (s.mat0U.x & FLAG_COMPLIANT_ACC) != 0, a1 = (s.mat1U.x & FLAG_COMPLIANT_ACC) != 0;
    const bool oneA = a0 != a1;

    if (bothC && oneA) rest = a0 ? r0 : r1;
    else
    {
        const uint32_t mode = oneC ? MODE_MIN : std::max(s.mat0U.z, s.mat1U.z);
        const float flip = (bothC && mode == MODE_MULTIPLY) ? -1.f : 1.f;
        rest = flip * combineScalars(r0, r1, mode);
    }
    {
        const float d0 = s.mat0F.w, d1 = s.mat1F.w;
        if (bothC && oneA) damp = a0 ? d0 : d1;
        else damp = combineScalars(d0, d1, oneC ? MODE_MAX : std::max(s.mat0U.w, s.mat1U.w));
    }
    {
        const uint32_t cf = s.mat0U.x | s.mat1U.x;
        if ((cf & FLAG_DISABLE_FRICTION) == 0)
        {
            const uint32_t mode = std::max(s.mat0U.y, s.mat1U.y);
            const float d = combineScalars(s.mat0F.x, s.mat1F.x, mode);
            const float st = combineScalars(s.mat0F.y, s.mat1F.y, mode);
            const float fd = (d > 0.f) ? d : 0.f;   // PxMax(dynFriction, 0.0f) = selectMax (PxsMaterialCombiner.h:161)
            dyn = fd;
            sta = (st - fd) >= 0.f ? st : fd;
            flags = cf;
        }
        else { flags = cf | FLAG_DISABLE_STRONG_FRICTION; dyn = 0.f; sta = 0.f; }
    }
}

// shared allflags/patch skeleton; entry-specific gates passed in
struct Tail
{
    uint32_t touchChange, patchChange, allflagsOut, nbOut, patchIndex, cpOut, ovf;
    uint32_t contactsBytes, forceBytes, patchesBytes;
    uint32_t outForces, outPoints;
};

static void refWriteContacts(const CoreOutIn& s, CoreOutOut& o)
{
    const uint32_t nb = s.misc.x, allflags = s.misc.y;
    uint32_t cBytes = s.counters.x, fBytes = s.counters.y, pBytes = s.counters.z, ovf = 0;

    std::memset(&o, 0, sizeof(o));

    uint32_t cOff = 0xFFFFFFFFu, fOff = 0xFFFFFFFFu;
    if (nb)
    {
        cOff = cBytes; cBytes += 16 * nb;
        fOff = fBytes; fBytes += 4 * nb;
        if ((uint64_t(cOff) + 16ull) > uint64_t(s.cfg.x)) { ovf |= OVF_CONTACT; cOff = 0xFFFFFFFFu; }
        if ((uint64_t(fOff) + 4ull) > uint64_t(s.cfg.y)) { ovf |= OVF_FORCE; fOff = 0xFFFFFFFFu; }
    }

    uint32_t outF, outP;
    if (cOff != 0xFFFFFFFFu)
    {
        const uint32_t base = (cOff - s.counters.x) / 16;
        for (uint32_t i = 0; i < nb; ++i) o.stream[base + i] = s.contacts[i];
        outF = fOff; outP = cOff;
    }
    else { outF = NULL_SENTINEL; outP = NULL_SENTINEL; }

    const uint32_t oldStatus = (allflags >> 16) & 0xFF;
    uint32_t status = oldStatus & ~TOUCH_KNOWN;
    status |= nb ? HAS_TOUCH : HAS_NO_TOUCH;
    const uint32_t prevPatches = (allflags & 0xFFFF) >> 8;
    const bool prevTouch = (oldStatus & HAS_TOUCH) != 0;
    const bool prevKnown = (oldStatus & TOUCH_KNOWN) != 0;
    const uint32_t numPatches = nb > 0 ? 1 : 0;
    const bool curTouch = nb > 0;

    uint32_t touchChange = ((prevTouch != curTouch) || !prevKnown) ? 1 : 0;
    uint32_t patchChange = (prevPatches != numPatches) ? 1 : 0;
    uint32_t allflagsOut = (((prevPatches << 8) | status) << 16) | (numPatches << 8);
    uint32_t nbOut = nb;

    uint32_t patchIndex = 0xFFFFFFFFu, cpOut = NULL_SENTINEL;
    if (nb)
    {
        patchIndex = pBytes; pBytes += 64;
        if ((uint64_t(patchIndex) + 64ull) > uint64_t(s.cfg.z))
        {
            ovf |= OVF_PATCH; patchIndex = 0xFFFFFFFFu;
            status &= ~TOUCH_KNOWN; status |= HAS_NO_TOUCH;
            allflagsOut = ((prevPatches << 8) | status) << 16;
            nbOut = 0;
            touchChange = prevTouch ? 1 : 0;
            patchChange = (prevPatches != 0) ? 1 : 0;
        }
        else cpOut = patchIndex;
    }

    o.counters = U4{ cBytes, fBytes, pBytes, ovf };
    o.outputs = U4{ outF, outP, cpOut, nbOut };
    o.flags = U4{ touchChange, patchChange, allflagsOut, patchIndex };
    o.patchArgs = F4{ s.normal.x, s.normal.y, s.normal.z, float(nb) };
}

static void refWriteContact(const CoreOutIn& s, CoreOutOut& o)
{
    const uint32_t numPatches = s.misc.z & 0xFF;   // PxU8
    const uint32_t numPoints = s.misc.w & 0xFFFF;  // PxU16
    const uint32_t allflags = s.misc.y;
    uint32_t cBytes = s.counters.x, fBytes = s.counters.y, pBytes = s.counters.z, ovf = 0;

    std::memset(&o, 0, sizeof(o));

    uint32_t cOff = 0xFFFFFFFFu, fOff = 0xFFFFFFFFu;
    if (numPoints)
    {
        cOff = cBytes; cBytes += 16 * numPoints;
        fOff = fBytes; fBytes += 4 * numPoints;
        if ((uint64_t(cOff) + 16ull) > uint64_t(s.cfg.x)) { ovf |= OVF_CONTACT; cOff = 0xFFFFFFFFu; }
        if ((uint64_t(fOff) + 4ull) > uint64_t(s.cfg.y)) { ovf |= OVF_FORCE; fOff = 0xFFFFFFFFu; }
    }

    uint32_t outF, outP;
    if (cOff != 0xFFFFFFFFu)
    {
        uint32_t slot = (cOff - s.counters.x) / 16;
        for (uint32_t i = 0; i < numPatches; ++i)
            for (uint32_t j = 0; j < (&s.patchCounts[i / 4].x)[i & 3]; ++j)
                o.stream[slot++] = s.points[i * 4 + j];
        outF = fOff; outP = cOff;
    }
    else { outF = NULL_SENTINEL; outP = NULL_SENTINEL; }

    const uint32_t oldStatus = (allflags >> 16) & 0xFF;
    uint32_t status = oldStatus & ~TOUCH_KNOWN;
    status |= numPoints ? HAS_TOUCH : HAS_NO_TOUCH;   // gate: numPoints
    const uint32_t prevPatches = (allflags & 0xFFFF) >> 8;
    const bool prevTouch = (oldStatus & HAS_TOUCH) != 0;
    const bool prevKnown = (oldStatus & TOUCH_KNOWN) != 0;
    const bool curTouch = numPatches > 0;             // gate: numPatches (cu L374)

    uint32_t touchChange = ((prevTouch != curTouch) || !prevKnown) ? 1 : 0;
    uint32_t patchChange = (prevPatches != numPatches) ? 1 : 0;
    uint32_t allflagsOut = (((prevPatches << 8) | status) << 16) | (numPatches << 8);
    uint32_t nbOut = numPoints;

    uint32_t patchIndex = 0xFFFFFFFFu, cpOut = NULL_SENTINEL;
    if (numPatches)
    {
        patchIndex = pBytes; pBytes += numPatches * 64;
        if ((uint64_t(patchIndex) + uint64_t(numPatches) * 64ull) > uint64_t(s.cfg.z))
        {
            ovf |= OVF_PATCH; patchIndex = 0xFFFFFFFFu;
            status &= ~TOUCH_KNOWN; status |= HAS_NO_TOUCH;
            allflagsOut = ((prevPatches << 8) | status) << 16;
            nbOut = 0;
            touchChange = prevTouch ? 1 : 0;
            patchChange = (prevPatches != 0) ? 1 : 0;
        }
        else cpOut = patchIndex;
    }

    if (patchIndex != 0xFFFFFFFFu)
    {
        float sta, dyn, rest, damp; uint32_t matFlags;
        combineMats(s, sta, dyn, rest, damp, matFlags);
        uint32_t startIdx = 0;
        for (uint32_t i = 0; i < numPatches; ++i)
        {
            const uint32_t nbp = (&s.patchCounts[i / 4].x)[i & 3];
            const F4& n = s.patchN[i];
            o.patchImg[i * 4 + 0] = U4{ f2u(1.f), f2u(1.f), f2u(1.f), f2u(1.f) };
            o.patchImg[i * 4 + 1] = U4{ f2u(n.x), f2u(n.y), f2u(n.z), f2u(rest) };
            o.patchImg[i * 4 + 2] = U4{ f2u(dyn), f2u(sta), f2u(damp),
                                        (startIdx & 0xFFFF) | ((nbp & 0xFF) << 16) | ((matFlags & 0xFF) << 24) };
            o.patchImg[i * 4 + 3] = U4{ (s.mat.x & 0xFFFF) << 16, s.mat.y & 0xFFFF, 0, 0 };
            startIdx += nbp;
        }
    }

    o.counters = U4{ cBytes, fBytes, pBytes, ovf };
    o.outputs = U4{ outF, outP, cpOut, nbOut };
    o.flags = U4{ touchChange, patchChange, allflagsOut, patchIndex };
    o.patchArgs = F4{ 0, 0, 0, 0 };
}

static bool eqOut(const CoreOutOut& r, const CoreOutOut& q)
{
    for (int j = 0; j < 64; ++j)
    {
        const F4& a = r.stream[j]; const F4& b = q.stream[j];
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w)) return false;
        const U4& c = r.patchImg[j]; const U4& d = q.patchImg[j];
        if (c.x != d.x || c.y != d.y || c.z != d.z || c.w != d.w) return false;
    }
    return r.counters.x == q.counters.x && r.counters.y == q.counters.y &&
           r.counters.z == q.counters.z && r.counters.w == q.counters.w &&
           r.outputs.x == q.outputs.x && r.outputs.y == q.outputs.y &&
           r.outputs.z == q.outputs.z && r.outputs.w == q.outputs.w &&
           r.flags.x == q.flags.x && r.flags.y == q.flags.y &&
           r.flags.z == q.flags.z && r.flags.w == q.flags.w &&
           f2u(r.patchArgs.x) == f2u(q.patchArgs.x) && f2u(r.patchArgs.y) == f2u(q.patchArgs.y) &&
           f2u(r.patchArgs.z) == f2u(q.patchArgs.z) && f2u(r.patchArgs.w) == f2u(q.patchArgs.w);
}

static void fillCommon(CoreOutIn& s, std::mt19937& rng, uint32_t i,
                       size_t& cOvfC, size_t& cOvfF, size_t& cOvfP, size_t& cHigh,
                       size_t& cHighF, size_t& cHighP)
{
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    // allflags: random prev patches (0..3), random status bits, random garbage elsewhere
    const uint32_t prevPatches = rng() % 4, prevStatus = rng() % 8;
    s.misc.y = ((prevPatches << 8 | prevStatus) << 16) | ((rng() % 4) << 8);

    // counters/limits: mostly roomy, with targeted tight and high-region cases
    uint32_t initC = 16 * (rng() % 256), initF = 4 * (rng() % 256), initP = 64 * (rng() % 64);
    uint32_t limC = 1u << 20, limF = 1u << 18, limP = 1u << 18;
    if (i % 5 == 1) { limC = initC + 16 * (rng() % 3); ++cOvfC; }         // contact tight
    if (i % 7 == 2) { limF = initF + 4 * (rng() % 3); ++cOvfF; }          // force tight
    if (i % 6 == 3) { limP = initP + 64 * (rng() % 3); ++cOvfP; }         // patch tight
    if (i % 11 == 4) { initC = 0xFFFFFFF0u; limC = 0xFFFFFFFFu; ++cHigh; } // 64-bit promote region
    if (i % 11 == 7) { initF = 0xFFFFFFFCu; limF = 0xFFFFFFFFu; ++cHighF; } // force counter high region (review 162)
    if (i % 11 == 9) { initP = 0xFFFFFFC0u; limP = 0xFFFFFFFFu; ++cHighP; } // patch counter high region (review 162)
    s.counters = U4{ initC, initF, initP, 0 };
    s.cfg = U4{ limC, limF, limP, 0 };

    s.mat = U4{ uint32_t(rng() % 0x10000), uint32_t(rng() % 0x10000), 0, 0 };
    s.mat0F = F4{ rf(0, 1.5f), rf(0, 1.5f), (rng() % 3 == 0) ? -rf(0.1f, 2) : rf(0, 1), rf(0, 2) };
    s.mat1F = F4{ rf(0, 1.5f), rf(0, 1.5f), (rng() % 3 == 0) ? -rf(0.1f, 2) : rf(0, 1), rf(0, 2) };
    const uint32_t fl0 = ((rng() % 4 == 0) ? FLAG_DISABLE_FRICTION : 0) | ((rng() % 3 == 0) ? FLAG_COMPLIANT_ACC : 0);
    const uint32_t fl1 = ((rng() % 4 == 0) ? FLAG_DISABLE_FRICTION : 0) | ((rng() % 3 == 0) ? FLAG_COMPLIANT_ACC : 0);
    s.mat0U = U4{ fl0, uint32_t(rng() % 4), uint32_t(rng() % 4), uint32_t(rng() % 4) };
    s.mat1U = U4{ fl1, uint32_t(rng() % 4), uint32_t(rng() % 4), uint32_t(rng() % 4) };
}

PXG_TEST(npCvxCoreOut_writeContacts)
{
    std::mt19937 rng(162001);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CoreOutIn> in(N);
    std::vector<CoreOutOut> ref(N);
    size_t cOvfC = 0, cOvfF = 0, cOvfP = 0, cHigh = 0, cHighF = 0, cHighP = 0, cZero = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CoreOutIn s{};
        fillCommon(s, rng, i, cOvfC, cOvfF, cOvfP, cHigh, cHighF, cHighP);
        const uint32_t nb = (i % 9 == 0) ? 0 : 1 + rng() % 4; // MAX_CONVEX_CONTACTS = 4
        if (!nb) ++cZero;
        s.misc.x = nb;
        for (uint32_t c = 0; c < 4; ++c)
            s.contacts[c] = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-0.2f, 0.2f) };
        s.normal = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        in[i] = s;
        refWriteContacts(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CoreOutOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreOut.spv", "ccoWriteContacts",
             { (uint32_t)sizeof(CoreOutIn), (uint32_t)sizeof(CoreOutOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CoreOutOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!eqOut(ref[i], g[i])) ++mm;
    PXG_CHECK_MSG(mm == 0, "ccoWriteContacts: %zu mismatches", mm);
    PXG_CHECK_MSG(cZero > N / 16 && cOvfC > N / 8 && cOvfF > N / 10 && cOvfP > N / 8 &&
                  cHigh > N / 16 && cHighF > N / 16 && cHighP > N / 16,
                  "ccoWriteContacts: coverage thin (zero %zu ovfC %zu ovfF %zu ovfP %zu high %zu/%zu/%zu)",
                  cZero, cOvfC, cOvfF, cOvfP, cHigh, cHighF, cHighP);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

PXG_TEST(npCvxCoreOut_writeContact)
{
    std::mt19937 rng(162002);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CoreOutIn> in(N);
    std::vector<CoreOutOut> ref(N);
    size_t cOvfC = 0, cOvfF = 0, cOvfP = 0, cHigh = 0, cHighF = 0, cHighP = 0, cZero = 0, cMany = 0, cSplit = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CoreOutIn s{};
        fillCommon(s, rng, i, cOvfC, cOvfF, cOvfP, cHigh, cHighF, cHighP);
        const uint32_t nPatches = (i % 9 == 0) ? 0 : 1 + rng() % 16; // MAX_PATCHES = 16
        if (!nPatches) ++cZero;
        if (nPatches > 8) ++cMany;
        uint32_t nPoints = 0;
        for (uint32_t p = 0; p < nPatches; ++p)
        {
            const uint32_t np = 1 + rng() % 4; // MAX_PATCH_POINTS = 4
            (&s.patchCounts[p / 4].x)[p & 3] = np;
            nPoints += np;
            s.patchN[p] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
            for (uint32_t j = 0; j < np; ++j)
                s.points[p * 4 + j] = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-0.2f, 0.2f) };
        }
        s.misc.z = nPatches;
        s.misc.w = nPoints;
        if (i % 23 == 5) { s.misc.w = 0; ++cSplit; }               // patches without points
        if (i % 23 == 16) { s.misc.z = 0; s.misc.w = 3; ++cSplit; } // points without patches
        in[i] = s;
        refWriteContact(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CoreOutOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreOut.spv", "ccoWriteContact",
             { (uint32_t)sizeof(CoreOutIn), (uint32_t)sizeof(CoreOutOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CoreOutOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!eqOut(ref[i], g[i])) ++mm;
    PXG_CHECK_MSG(mm == 0, "ccoWriteContact: %zu mismatches", mm);
    PXG_CHECK_MSG(cZero > N / 16 && cMany > N / 4 && cOvfC > N / 8 && cOvfF > N / 10 && cOvfP > N / 8 &&
                  cHigh > N / 16 && cHighF > N / 16 && cHighP > N / 16 && cSplit > N / 16,
                  "ccoWriteContact: coverage thin (zero %zu many %zu ovfC %zu ovfF %zu ovfP %zu high %zu/%zu/%zu split %zu)",
                  cZero, cMany, cOvfC, cOvfF, cOvfP, cHigh, cHighF, cHighP, cSplit);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// P9.86e DIRECTED SIGNED-ZERO MATERIALS through ccoWriteContact's patch image.  eMIN/eMAX are PxMin/PxMax<float>
// selects: a +-0 tie returns the SECOND operand, so PxMin(-0,+0) = +0 and PxMax(+0,-0) = -0, where GLSL
// FMin/FMax on llvmpipe give -0 / +0 in either order.  -0 is a valid material value (NpMaterial checks finite
// and >= 0.f), and a restitution of -0 is rigid (-0 < 0 is false).  fillCommon never produces -0.
PXG_TEST(npCvxCoreOut_materialZeroSign)
{
    const float PZ = 0.f, NZ = u2f(0x80000000u);
    // {dyn, sta, rest, damp} + {flags, fricMode, restMode, dampMode}; modes 1 = eMIN, 3 = eMAX.
    struct DirMat { F4 f; U4 u; };
    const DirMat dm[7] = {
        { { PZ, NZ, NZ, NZ }, { 0, 1, 1, 1 } },      // D0  -0 under eMIN
        { { PZ, PZ, PZ, PZ }, { 0, 1, 1, 1 } },      // D1  +0 under eMIN
        { { PZ, PZ, PZ, PZ }, { 0, 3, 3, 3 } },      // D2  +0 under eMAX
        { { PZ, NZ, NZ, NZ }, { 0, 3, 3, 3 } },      // D3  -0 under eMAX
        { { PZ, PZ, -0.5f, PZ }, { 0, 0, 0, 0 } },   // D4  compliant (rest < 0), damping +0
        { { PZ, PZ, NZ, NZ }, { 0, 0, 0, 0 } },      // D5  rigid (rest -0), damping -0
        { { NZ, PZ, PZ, PZ }, { 0, 0, 0, 0 } },      // D6  dynamic friction -0 under eAVERAGE
    };
    // w7..w10 = restitution / dynF / staF / damping, the CUDA's bits computed by hand from PxsCombineMaterials;
    // (ma, mb) and (mb, ma) are the two shape orders.  killsVariant names the host twin whose image must differ.
    struct DirCase { uint32_t ma, mb, w7, w8, w9, w10; int killsVariant; };
    const uint32_t NDIR = 6;
    const DirCase dc[NDIR] = {
        { 0, 1, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 1 },   // K0 eMIN(-0,+0): PxMin +0, FMin -0
        { 2, 3, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 2 },   // K1 eMAX(+0,-0): PxMax -0, FMax +0
        { 4, 5, 0xBF000000u, 0u, 0x00000000u, 0x80000000u, 2 },   // K2 one compliant: damping forced to eMAX(+0,-0)
        { 1, 0, 0x80000000u, 0u, 0x80000000u, 0x80000000u, 0 },   // K3 control: eMIN(+0,-0) = -0 in both
        { 3, 2, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K4 control: eMAX(-0,+0) = +0 in both
        { 6, 6, 0x00000000u, 0u, 0x00000000u, 0x00000000u, 0 },   // K5 d = 0.5*(-0 + -0) = -0: PxMax(d, 0) keeps dynF +0
    };
    PXG_CHECK_MSG(std::signbit(0.5f * (dm[6].f.x + dm[6].f.x)), "ccoWriteContact directed: K5 must reach the dynF clamp with -0");

    std::vector<CoreOutIn> in(NDIR);
    std::vector<CoreOutOut> ref(NDIR);
    size_t cMinTie = 0, cMaxTie = 0;
    for (uint32_t k = 0; k < NDIR; ++k)
    {
        CoreOutIn s{};
        s.misc = U4{ 0, 0, 1, 1 };                         // one patch with one point, allflags 0
        s.patchCounts[0] = U4{ 1, 0, 0, 0 };
        s.patchN[0] = F4{ 0, 1, 0, 0 };
        s.points[0] = F4{ 0.25f, -0.5f, 1, -0.125f };
        s.counters = U4{ 0, 0, 0, 0 };
        s.cfg = U4{ 1u << 20, 1u << 18, 1u << 18, 0 };
        s.mat = U4{ 7 + k, 11 + k, 0, 0 };
        s.mat0F = dm[dc[k].ma].f; s.mat0U = dm[dc[k].ma].u;
        s.mat1F = dm[dc[k].mb].f; s.mat1U = dm[dc[k].mb].u;
        in[k] = s;
        refWriteContact(s, ref[k]);
        const U4 w1 = ref[k].patchImg[1], w2 = ref[k].patchImg[2];
        PXG_CHECK_MSG(ref[k].flags.w == 0u && w1.w == dc[k].w7 && w2.x == dc[k].w8 && w2.y == dc[k].w9 && w2.z == dc[k].w10,
                      "ccoWriteContact directed %u: oracle rest/dynF/staF/damp %08x %08x %08x %08x, CUDA %08x %08x %08x %08x",
                      k, w1.w, w2.x, w2.y, w2.z, dc[k].w7, dc[k].w8, dc[k].w9, dc[k].w10);
        for (int v = 1; v <= 2; ++v)
        {
            CoreOutOut tw;
            gCombineVariant = v;
            refWriteContact(s, tw);
            gCombineVariant = 0;
            const bool differs = !eqOut(tw, ref[k]);
            PXG_CHECK_MSG(differs == (dc[k].killsVariant == v), "ccoWriteContact directed %u: twin %d %s the image",
                          k, v, differs ? "changes" : "does not change");
            if (differs) ++(v == 1 ? cMinTie : cMaxTie);
        }
    }
    PXG_CHECK_MSG(cMinTie == 1 && cMaxTie == 2, "ccoWriteContact: signed-zero combine coverage thin (minTie %zu maxTie %zu)",
                  cMinTie, cMaxTie);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<CoreOutOut> outInit(NDIR);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct P { uint32_t numTests; } prm = { NDIR };
    Kernel kr("npCvxCoreOut.spv", "ccoWriteContact",
              { (uint32_t)sizeof(CoreOutIn), (uint32_t)sizeof(CoreOutOut) }, sizeof(P));
    kr.dispatch(1, { bIn, bOut }, prm);
    dev->flush();
    const auto g = downloadBuffer<CoreOutOut>(dev, bOut, NDIR);
    for (uint32_t k = 0; k < NDIR; ++k)
        PXG_CHECK_MSG(eqOut(ref[k], g[k]), "ccoWriteContact directed %u: GPU rest/dynF/staF/damp %08x %08x %08x %08x, oracle %08x %08x %08x %08x",
                      k, g[k].patchImg[1].w, g[k].patchImg[2].x, g[k].patchImg[2].y, g[k].patchImg[2].z,
                      ref[k].patchImg[1].w, ref[k].patchImg[2].x, ref[k].patchImg[2].y, ref[k].patchImg[2].z);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
