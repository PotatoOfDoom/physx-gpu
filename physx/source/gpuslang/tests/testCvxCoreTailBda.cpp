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

// P9.87 -- the two convexCore emission tails, run THEMSELVES on raw device streams
// (cvxCoreTailBda.slang), against a mirror transcribed from the CUDA:
//
//   cvxCoreTailBda_single  cvxCoreWriteOutput   <- TestOutput::writeOutput, the PxContact
//                          overload (convexCoreCollision.cu:177-290) and the
//                          insertIntoPatchStream it ends with (contactPatchUtils.cuh:139-180)
//   cvxCoreTailBda_multi   cvxCoreWriteOutput32 <- the Gu::Contact overload (:292-445)
//
// EVERY BUFFER STARTS AS PER-WORD POISON AND IS COMPARED WHOLE: the patch stream,
// the contact stream, cmOutputs, both change-flag arrays and the per-case counters.
// A word the mirror says nothing writes must come back as uploaded.  That is the
// P9.87 property -- PxContactPatch is written field by field, so bytes 54-63 (word
// 13's high half, words 14-15) are never stored -- and it also catches a tail that
// writes more than it should (a loop to MAX_PATCHES, a header behind a dead
// sentinel).  The poison varies per word and per buffer, so a read-modify-write
// that reads the WRONG word is as visible as one that zeroes.
//
// WHAT THE CASES HAVE TO CONSTRUCT, over and above random content:
//   * each overflow on its own (patch, contact, force) and patch+force together;
//     force-only overflow is the case where contactForces is stored as
//     startContactForces + 0xFFFFFFFF, upstream's sharp edge (cu:230, :349);
//   * a contact limit of exactly offset + 16 with several points, which the CUDA
//     accepts because it tests ONE PxContact (:201, :321) -- a port that tested the
//     whole reservation would reject it;
//   * material indices with non-zero bits 8-15 (words 12/13 store the full PxU16)
//     and with bit 16 set (the lookup narrows to PxU16 first, cuh:149-150 / cu:415-416):
//     materials[k] and materials[0x10000 + k] differ, so an unnarrowed lookup reads a
//     different, in-bounds material;
//   * prior allflags that are poison AND prior allflags that are realistic, so
//     eTOUCH_KNOWN and eHAS_TOUCH are seen both ways;
//   * (multi) junk patch counts ABOVE numPatches in half the cases: the CUDA reads
//     numPatchPoints(i) only for i < numPatches (:340-347, :421-443), so a tail that
//     walked all MAX_PATCHES would write extra points and headers.
//
// numPoints IS ALWAYS THE SUM OF THE KEPT COUNTS, each 1..4, because that is the only
// state Contact32 produces (pxgSlangCvxCoreOut32.slang, difference 1).  One mutant is
// therefore equivalent here by construction: gating the touch change on numPoints
// instead of numPatches (out32-touch-gates-points).

#include "pxgTestFramework.h"
#include "pxgTriTriRef.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgtest::tritriref;   // MaterialH, combine, bitsOfF

// The case record, 1376 bytes, matching cvxCoreTailBda.slang's explicit offsets.
struct CctCaseH
{
    uint32_t hdr[4];            // @0    nbContacts|numPatches, numPoints, mi0, mi1
    uint32_t lim[4];            // @16   patch, contact, force byte limits
    uint32_t patchCount[16];    // @32
    float patchNormal[16][4];   // @96
    float points[64][4];        // @352
};
static_assert(sizeof(CctCaseH) == 1376, "case stride");

struct CctPushH
{
    uint64_t cases, cmOutputs, materials, contactStream, patchStream, counters, touchFlags,
             patchFlags, startPatches, startPoints, startForces;
    uint32_t numCases, pad;
};
static_assert(sizeof(CctPushH) == 96, "push block");

const uint32_t INVALID = 0xFFFFFFFFu;
const uint32_t N = 512;
const uint32_t PATCH_WORDS = 256;     // per case: 16 patches x 64 B
const uint32_t CONTACT_WORDS = 256;   // per case: 64 points x 16 B
const uint32_t OUT_WORDS = 12;        // PxsContactManagerOutput, 48 B
const uint32_t OVF_SEED = 8u;         // not one of the three overflow bits: must survive the ORs
const uint32_t NMAT = 0x10000u + 8u;

// Fake start* bases.  They are never dereferenced -- the tails only add offsets to
// them -- and they differ from the stream addresses, so a tail that built an
// output pointer from the stream instead of from start* is visible.
const uint64_t START_PATCHES = 0x0000100000000000ull;
const uint64_t START_POINTS = 0x0000200000000040ull;
const uint64_t START_FORCES = 0x0000300000000080ull;

enum { B_PATCH = 0, B_CONTACT, B_OUT, B_TOUCH, B_PCHG };
uint32_t poison(uint32_t buf, uint32_t word)
{
    static const uint32_t salt[5] = { 0x00000000u, 0x5A5A0000u, 0x00005A5Au, 0x3C3C3C3Cu, 0x81818181u };
    return (0xCDCDCDCDu ^ salt[buf]) ^ (word * 0x00010001u);
}

std::vector<MaterialH> makeMaterials(std::mt19937& rng)
{
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    std::vector<MaterialH> mats(NMAT);
    for (auto& m : mats)
    {
        // Strictly non-zero magnitudes: PxMin/PxMax and the shader's min/max may
        // differ on +0 vs -0 (P9.86), which is not what this fixture is about.
        m.dynamicFriction = rf(0.05f, 1.5f);
        m.staticFriction = rf(0.05f, 1.5f);
        m.restitution = (rng() % 4 == 0) ? rf(-2.0f, -0.1f) : rf(0.05f, 1.0f);
        m.damping = rf(0.05f, 1.0f);
        uint32_t flags = 0;
        if (rng() % 8 == 0) flags |= 1u;    // eDISABLE_FRICTION
        if (rng() % 4 == 0) flags |= 2u;    // eDISABLE_STRONG_FRICTION
        if (rng() % 3 == 0) flags |= 16u;   // eCOMPLIANT_ACCELERATION_SPRING
        if (rng() % 4 == 0) flags |= 4u;
        m.flagsModes0 = flags | ((rng() % 4) << 16) | ((rng() % 4) << 24);
        // dampingCombineMode is the low byte; the three pad bytes above it carry junk.
        m.modes1 = (rng() % 4) | (uint32_t(rng()) & 0xFFFFFF00u);
    }
    return mats;
}

uint32_t pickMaterial(std::mt19937& rng)
{
    switch (rng() % 4)
    {
    case 0: return rng() % 8;
    case 1: return rng() & 0xFFFFu;
    case 2: return 0x10000u | (rng() % 8);        // bit 16: the lookup must narrow first
    default: return 0x0100u | (rng() & 0xFF00u) | (rng() % 256);   // bits 8-15 set
    }
}

struct Expected
{
    std::vector<uint32_t> patch, contact, out, touch, pchg, ctr;
};

// The per-case prior content.
void initImages(Expected& e, std::mt19937& rng)
{
    e.patch.resize(size_t(N) * PATCH_WORDS);
    e.contact.resize(size_t(N) * CONTACT_WORDS);
    e.out.resize(size_t(N) * OUT_WORDS);
    e.touch.resize(N);
    e.pchg.resize(N);
    e.ctr.resize(size_t(N) * 4);
    for (size_t w = 0; w < e.patch.size(); ++w) e.patch[w] = poison(B_PATCH, uint32_t(w));
    for (size_t w = 0; w < e.contact.size(); ++w) e.contact[w] = poison(B_CONTACT, uint32_t(w));
    for (size_t w = 0; w < e.out.size(); ++w) e.out[w] = poison(B_OUT, uint32_t(w));
    for (uint32_t t = 0; t < N; ++t) { e.touch[t] = poison(B_TOUCH, t); e.pchg[t] = poison(B_PCHG, t); }
    for (uint32_t t = 0; t < N; ++t)
    {
        e.ctr[t * 4 + 0] = t * PATCH_WORDS * 4;     // patchesBytes
        e.ctr[t * 4 + 1] = t * CONTACT_WORDS * 4;   // contactsBytes
        e.ctr[t * 4 + 2] = t * 256u;                // forceAndIndiceBytes
        e.ctr[t * 4 + 3] = OVF_SEED;
        if (t % 2 == 1)
        {
            // A realistic prior: any allflagsStart byte, 0..16 patches, any status
            // byte (the eTOUCH_KNOWN / eHAS_TOUCH bits both ways, and the higher
            // status bits the tails must carry through), any prevPatches byte.
            e.out[t * OUT_WORDS + 8] = (uint32_t(rng()) & 0xFFu) | ((rng() % 17) << 8) |
                                       ((uint32_t(rng()) & 0xFFu) << 16) | ((uint32_t(rng()) & 0xFFu) << 24);
            e.out[t * OUT_WORDS + 9] = uint32_t(rng());
        }
    }
}

void put64(std::vector<uint32_t>& v, size_t w, uint64_t x) { v[w] = uint32_t(x); v[w + 1] = uint32_t(x >> 32); }

struct Coverage
{
    size_t written = 0, zero = 0, ovfPatch = 0, ovfContact = 0, ovfForceOnly = 0, ovfPatchForce = 0,
           contactQuirk = 0, exact = 0, highBitWritten = 0, midByteWritten = 0, multiPatch = 0,
           junkCounts = 0, prevKnown = 0, prevUnknown = 0, prevTouch = 0, prevNoTouch = 0;
};

// The limit classes, by case index.
enum LimitClass { L_FREE = 0, L_PATCH_OVF, L_CONTACT_OVF, L_FORCE_OVF, L_EXACT, L_PATCH_FORCE_OVF, L_QUIRK };
LimitClass limitClass(uint32_t t)
{
    switch (t % 10)
    {
    case 3: return L_PATCH_OVF;
    case 4: return L_CONTACT_OVF;
    case 5: return L_FORCE_OVF;
    case 6: return L_EXACT;
    case 7: return L_PATCH_FORCE_OVF;
    case 8: return L_QUIRK;
    default: return L_FREE;
    }
}

void setLimits(CctCaseH& c, uint32_t t, uint32_t patchBytes, uint32_t contactBytes, uint32_t forceBytes)
{
    const uint32_t p0 = t * PATCH_WORDS * 4, c0 = t * CONTACT_WORDS * 4, f0 = t * 256u;
    const uint32_t FREE = 0xF0000000u;
    c.lim[0] = FREE; c.lim[1] = FREE; c.lim[2] = FREE; c.lim[3] = 0;
    switch (limitClass(t))
    {
    case L_PATCH_OVF:       c.lim[0] = p0 + patchBytes - 1u; break;
    case L_CONTACT_OVF:     c.lim[1] = c0 + 15u; break;
    case L_FORCE_OVF:       c.lim[2] = f0 + 3u; break;
    case L_EXACT:           c.lim[0] = p0 + patchBytes; c.lim[1] = c0 + contactBytes; c.lim[2] = f0 + forceBytes; break;
    case L_PATCH_FORCE_OVF: c.lim[0] = p0 + patchBytes - 1u; c.lim[2] = f0 + 3u; break;
    case L_QUIRK:           c.lim[1] = c0 + 16u; c.lim[2] = f0 + 4u; break;
    default: break;
    }
}

// The output-and-flags half shared by both overloads (cu:197-262 / :317-404): the
// overloads differ only in what gates what, which the caller passes in.
struct FlagsIn
{
    uint32_t numPoints;       // what the reservations and eHAS_TOUCH gate on
    uint32_t numPatches;      // written into allflags, compared for patchChange
    bool currentlyHasTouch;   // single: nbContacts > 0; multi: numPatches > 0 (:374)
    uint32_t patchBytes;      // 64 or numPatches * 64
};

struct FlagsOut { uint32_t contactOff, patchIndex; };

FlagsOut refOutputAndFlags(uint32_t t, const CctCaseH& c, const FlagsIn& f, Expected& e, Coverage& cov)
{
    uint32_t* ctr = &e.ctr[t * 4];
    uint32_t* out = &e.out[t * OUT_WORDS];

    uint32_t contactOff = INVALID, forceOff = INVALID;
    if (f.numPoints)
    {
        contactOff = ctr[1]; ctr[1] += 16u * f.numPoints;
        forceOff = ctr[2]; ctr[2] += 4u * f.numPoints;
        // sizeof() is a size_t: the sums are 64-bit (:201, :207).
        if (uint64_t(contactOff) + 16u > c.lim[1]) { ctr[3] |= 1u; contactOff = INVALID; }
        if (uint64_t(forceOff) + 4u > c.lim[2]) { ctr[3] |= 2u; forceOff = INVALID; }
    }
    if (contactOff != INVALID)
    {
        put64(e.out, t * OUT_WORDS + 4, START_FORCES + uint64_t(forceOff));
        put64(e.out, t * OUT_WORDS + 2, START_POINTS + uint64_t(contactOff));
    }
    else
    {
        put64(e.out, t * OUT_WORDS + 4, 0);
        put64(e.out, t * OUT_WORDS + 2, 0);
    }

    const uint32_t allflags = out[8];
    const uint32_t oldStatus = (allflags >> 16) & 0xFFu;   // u16Low(u32High(allflags))
    uint32_t status = oldStatus & ~3u;
    status |= f.numPoints ? 2u : 1u;
    const uint32_t prevPatches = (allflags >> 8) & 0xFFu;  // u16High(u32Low(allflags))
    const bool prevHadTouch = (oldStatus & 2u) != 0;
    const bool prevKnown = (oldStatus & 3u) != 0;
    (prevKnown ? cov.prevKnown : cov.prevUnknown)++;
    (prevHadTouch ? cov.prevTouch : cov.prevNoTouch)++;

    e.touch[t] = ((prevHadTouch != f.currentlyHasTouch) || !prevKnown) ? 1u : 0u;
    e.pchg[t] = (prevPatches != f.numPatches) ? 1u : 0u;
    out[8] = (prevPatches << 24) | (status << 16) | (f.numPatches << 8);
    out[9] = (out[9] & 0xFFFF0000u) | (f.numPoints & 0xFFFFu);

    uint32_t patchIndex = INVALID;
    if (f.numPatches)
    {
        patchIndex = ctr[0]; ctr[0] += f.patchBytes;
        if (uint64_t(patchIndex) + f.patchBytes > c.lim[0])
        {
            ctr[3] |= 4u;
            patchIndex = INVALID;
            status = (status & ~3u) | 1u;
            out[8] = (prevPatches << 24) | (status << 16);
            out[9] = out[9] & 0xFFFF0000u;
            e.touch[t] = prevHadTouch ? 1u : 0u;
            e.pchg[t] = (prevPatches != 0) ? 1u : 0u;
        }
        else
            put64(e.out, t * OUT_WORDS + 0, START_PATCHES + uint64_t(patchIndex));
    }

    if (f.numPoints)
    {
        const bool cOvf = contactOff == INVALID, fOvf = forceOff == INVALID, pOvf = f.numPatches && patchIndex == INVALID;
        if (pOvf && !fOvf && !cOvf) ++cov.ovfPatch;
        if (cOvf && !pOvf) ++cov.ovfContact;
        if (fOvf && !cOvf && !pOvf) ++cov.ovfForceOnly;
        if (pOvf && fOvf) ++cov.ovfPatchForce;
        if (!cOvf && f.numPoints >= 2 && c.lim[1] == t * CONTACT_WORDS * 4 + 16u) ++cov.contactQuirk;
        if (!cOvf && !fOvf && !pOvf && limitClass(t) == L_EXACT) ++cov.exact;
    }
    return { contactOff, patchIndex };
}

// One PxContactPatch header (cuh:161-178 / cu:425-441), field by field: bytes 54-63
// are never stored, so word 13's high half and words 14-15 keep the stream's content.
void refHeader(Expected& e, uint32_t patchByte, const float n[4], uint32_t startContactIndex,
               uint32_t nbContacts, const CombinedH& cm, uint32_t mi0, uint32_t mi1)
{
    uint32_t* w = &e.patch[patchByte / 4];
    const uint32_t one = bitsOfF(1.0f);
    w[0] = one; w[1] = one; w[2] = one; w[3] = one;
    w[4] = bitsOfF(n[0]); w[5] = bitsOfF(n[1]); w[6] = bitsOfF(n[2]);
    w[7] = bitsOfF(cm.rest);
    w[8] = bitsOfF(cm.dyn);
    w[9] = bitsOfF(cm.sta);
    w[10] = bitsOfF(cm.damping);
    w[11] = (startContactIndex & 0xFFFFu) | ((nbContacts & 0xFFu) << 16) | ((cm.flags & 0xFFu) << 24);
    w[12] = (mi0 & 0xFFFFu) << 16;                      // internalFlags = 0
    w[13] = (w[13] & 0xFFFF0000u) | (mi1 & 0xFFFFu);    // PxU16 store: pad[0] untouched
    // w[14], w[15]: pad[1..4], never stored
}

size_t compareBuf(const char* name, const std::vector<uint32_t>& got, const std::vector<uint32_t>& want,
                  uint32_t wordsPerCase, size_t& stray, uint32_t buf)
{
    size_t mm = 0;
    for (size_t w = 0; w < want.size(); ++w)
        if (got[w] != want[w])
        {
            if (buf < 5 && want[w] == poison(buf, uint32_t(w))) ++stray;
            if (mm < 6)
                std::printf("  %s word %zu (case %zu, word %zu): got 0x%08x want 0x%08x\n", name, w,
                            w / wordsPerCase, w % wordsPerCase, got[w], want[w]);
            ++mm;
        }
    return mm;
}

struct DeviceRun
{
    std::vector<uint32_t> patch, contact, out, touch, pchg, ctr;
};

DeviceRun runDevice(const char* entry, const std::vector<CctCaseH>& cases, const std::vector<MaterialH>& mats,
                    const Expected& init)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bCases = createBufferFrom(dev, cases);
    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    PxgBufferHandle bPatch = createBufferFrom(dev, init.patch);
    PxgBufferHandle bContact = createBufferFrom(dev, init.contact);
    PxgBufferHandle bOut = createBufferFrom(dev, init.out);
    PxgBufferHandle bTouch = createBufferFrom(dev, init.touch);
    PxgBufferHandle bPchg = createBufferFrom(dev, init.pchg);
    PxgBufferHandle bCtr = createBufferFrom(dev, init.ctr);

    CctPushH push{};
    push.cases = dev->getBufferDeviceAddress(bCases);
    push.cmOutputs = dev->getBufferDeviceAddress(bOut);
    push.materials = dev->getBufferDeviceAddress(bMat);
    push.contactStream = dev->getBufferDeviceAddress(bContact);
    push.patchStream = dev->getBufferDeviceAddress(bPatch);
    push.counters = dev->getBufferDeviceAddress(bCtr);
    push.touchFlags = dev->getBufferDeviceAddress(bTouch);
    push.patchFlags = dev->getBufferDeviceAddress(bPchg);
    push.startPatches = START_PATCHES;
    push.startPoints = START_POINTS;
    push.startForces = START_FORCES;
    push.numCases = N;

    Kernel k("cvxCoreTailBda.spv", entry, {}, sizeof(CctPushH));
    k.dispatch((N + 63) / 64, {}, push, 1u, 1u);
    dev->flush();

    DeviceRun r;
    r.patch = downloadBuffer<uint32_t>(dev, bPatch, init.patch.size());
    r.contact = downloadBuffer<uint32_t>(dev, bContact, init.contact.size());
    r.out = downloadBuffer<uint32_t>(dev, bOut, init.out.size());
    r.touch = downloadBuffer<uint32_t>(dev, bTouch, init.touch.size());
    r.pchg = downloadBuffer<uint32_t>(dev, bPchg, init.pchg.size());
    r.ctr = downloadBuffer<uint32_t>(dev, bCtr, init.ctr.size());
    for (PxgBufferHandle h : { bCases, bMat, bPatch, bContact, bOut, bTouch, bPchg, bCtr })
        dev->destroyBuffer(h);
    return r;
}

void compareAll(const char* test, const DeviceRun& g, const Expected& e)
{
    size_t stray = 0;
    const size_t mmPatch = compareBuf("patch", g.patch, e.patch, PATCH_WORDS, stray, B_PATCH);
    const size_t mmContact = compareBuf("contact", g.contact, e.contact, CONTACT_WORDS, stray, B_CONTACT);
    const size_t mmOut = compareBuf("cmOutput", g.out, e.out, OUT_WORDS, stray, B_OUT);
    const size_t mmTouch = compareBuf("touchChange", g.touch, e.touch, 1, stray, B_TOUCH);
    const size_t mmPchg = compareBuf("patchChange", g.pchg, e.pchg, 1, stray, B_PCHG);
    size_t strayNone = 0;
    const size_t mmCtr = compareBuf("counters", g.ctr, e.ctr, 4, strayNone, 99);
    PXG_CHECK_MSG(mmPatch == 0, "%s: %zu patch-stream mismatches", test, mmPatch);
    PXG_CHECK_MSG(mmContact == 0, "%s: %zu contact-stream mismatches", test, mmContact);
    PXG_CHECK_MSG(mmOut == 0, "%s: %zu cmOutputs mismatches", test, mmOut);
    PXG_CHECK_MSG(mmTouch == 0 && mmPchg == 0, "%s: %zu touch / %zu patch change-flag mismatches", test, mmTouch, mmPchg);
    PXG_CHECK_MSG(mmCtr == 0, "%s: %zu counter mismatches", test, mmCtr);
    PXG_CHECK_MSG(stray == 0, "%s: %zu writes where nothing may be written", test, stray);
}

void checkCoverage(const char* test, const Coverage& c, bool multi)
{
    std::printf("  [%s] written %zu zero %zu ovf patch %zu contact %zu forceOnly %zu patch+force %zu "
                "quirk %zu exact %zu hiBit %zu midByte %zu multiPatch %zu junk %zu known %zu/%zu touch %zu/%zu\n",
                test, c.written, c.zero, c.ovfPatch, c.ovfContact, c.ovfForceOnly, c.ovfPatchForce,
                c.contactQuirk, c.exact, c.highBitWritten, c.midByteWritten, c.multiPatch, c.junkCounts,
                c.prevKnown, c.prevUnknown, c.prevTouch, c.prevNoTouch);
    PXG_CHECK_MSG(c.written > N / 3 && c.zero > 10 && c.ovfPatch > 10 && c.ovfContact > 10 &&
                  c.ovfForceOnly > 10 && c.ovfPatchForce > 10 && c.contactQuirk > 10 && c.exact > 10 &&
                  c.highBitWritten > 10 && c.midByteWritten > 10 && c.prevKnown > 50 && c.prevUnknown > 50 &&
                  c.prevTouch > 50 && c.prevNoTouch > 50,
                  "%s: coverage thin", test);
    if (multi)
        PXG_CHECK_MSG(c.multiPatch > N / 4 && c.junkCounts > N / 4, "%s: multi-patch coverage thin", test);
}

} // namespace

// ---------------------------------------------------------------------------
// cvxCoreWriteOutput: cu:177-290 + insertIntoPatchStream (cuh:139-180).
// ---------------------------------------------------------------------------
PXG_TEST(cvxCoreTailBda_single)
{
    std::mt19937 rng(8701);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    const std::vector<MaterialH> mats = makeMaterials(rng);

    Expected e;
    initImages(e, rng);
    const Expected init = e;
    Coverage cov;

    std::vector<CctCaseH> cases(N);
    std::memset(cases.data(), 0, cases.size() * sizeof(CctCaseH));
    for (uint32_t t = 0; t < N; ++t)
    {
        CctCaseH& c = cases[t];
        // L_QUIRK needs several points; otherwise 0..4, with zero one case in five.
        const uint32_t nb = (limitClass(t) == L_QUIRK) ? 2u + rng() % 3u : ((rng() % 5 == 0) ? 0u : 1u + rng() % 4u);
        c.hdr[0] = nb;
        c.hdr[2] = pickMaterial(rng);
        c.hdr[3] = pickMaterial(rng);
        for (uint32_t k = 0; k < 3; ++k) c.patchNormal[0][k] = rf(-1.f, 1.f);
        c.patchNormal[0][3] = rf(-1.f, 1.f);   // ignored
        for (uint32_t i = 0; i < 4; ++i)       // all four filled: only i < nb may be written
            for (uint32_t k = 0; k < 4; ++k) c.points[i][k] = rf(-5.f, 5.f);
        setLimits(c, t, 64u, 16u * nb, 4u * nb);

        // ---- the mirror ----
        FlagsIn f{ nb, nb ? 1u : 0u, nb > 0, 64u };
        const FlagsOut o = refOutputAndFlags(t, c, f, e, cov);
        if (o.contactOff != INVALID)
            for (uint32_t i = 0; i < nb; ++i)
                for (uint32_t k = 0; k < 4; ++k)
                    e.contact[o.contactOff / 4 + i * 4 + k] = bitsOfF(c.points[i][k]);
        if (nb == 0) ++cov.zero;
        if (o.patchIndex != INVALID)
        {
            const uint32_t mi0 = c.hdr[2] & 0xFFFFu, mi1 = c.hdr[3] & 0xFFFFu;   // cuh:149-150
            const CombinedH cm = combine(mats[mi0], mats[mi1]);
            refHeader(e, o.patchIndex, c.patchNormal[0], 0u, nb, cm, mi0, mi1);
            ++cov.written;
            if ((c.hdr[2] | c.hdr[3]) & 0x10000u) ++cov.highBitWritten;
            if ((mi0 | mi1) & 0xFF00u) ++cov.midByteWritten;
        }
    }

    const DeviceRun g = runDevice("cctSingle", cases, mats, init);
    compareAll("cvxCoreTailBda_single", g, e);
    checkCoverage("single", cov, false);
}

// ---------------------------------------------------------------------------
// cvxCoreWriteOutput32: cu:292-445.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCoreTailBda_multi)
{
    std::mt19937 rng(8702);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    const std::vector<MaterialH> mats = makeMaterials(rng);

    Expected e;
    initImages(e, rng);
    const Expected init = e;
    Coverage cov;

    std::vector<CctCaseH> cases(N);
    std::memset(cases.data(), 0, cases.size() * sizeof(CctCaseH));
    for (uint32_t t = 0; t < N; ++t)
    {
        CctCaseH& c = cases[t];
        // Zero one case in eight; L_QUIRK needs points.
        const uint32_t numPatches = (rng() % 8 == 0 && limitClass(t) != L_QUIRK) ? 0u : 1u + rng() % 16u;
        uint32_t numPoints = 0;
        const bool junk = (t % 4) >= 2;
        for (uint32_t i = 0; i < 16; ++i)
        {
            if (i < numPatches) { c.patchCount[i] = 1u + rng() % 4u; numPoints += c.patchCount[i]; }
            else c.patchCount[i] = junk ? 1u + rng() % 4u : 0u;
            for (uint32_t k = 0; k < 4; ++k) c.patchNormal[i][k] = rf(-1.f, 1.f);
        }
        if (limitClass(t) == L_QUIRK && numPoints < 2)   // one patch of one point: give it three
        {
            c.patchCount[0] = 3u;
            numPoints = 3u;
        }
        for (uint32_t i = 0; i < 64; ++i)   // all 64 filled: only the counted ones may be written
            for (uint32_t k = 0; k < 4; ++k) c.points[i][k] = rf(-5.f, 5.f);
        c.hdr[0] = numPatches;
        c.hdr[1] = numPoints;   // the accumulator's invariant: the sum of the kept counts
        c.hdr[2] = pickMaterial(rng);
        c.hdr[3] = pickMaterial(rng);
        setLimits(c, t, 64u * numPatches, 16u * numPoints, 4u * numPoints);
        if (junk && numPatches < 16) ++cov.junkCounts;

        // ---- the mirror ----
        FlagsIn f{ numPoints & 0xFFFFu, numPatches & 0xFFu, numPatches > 0, 64u * numPatches };
        const FlagsOut o = refOutputAndFlags(t, c, f, e, cov);
        if (o.contactOff != INVALID)
        {
            // (:340-347) ONE cursor, patch-major, i < numPatches only.
            uint32_t slot = 0;
            for (uint32_t i = 0; i < numPatches; ++i)
                for (uint32_t j = 0; j < c.patchCount[i]; ++j, ++slot)
                    for (uint32_t k = 0; k < 4; ++k)
                        e.contact[o.contactOff / 4 + slot * 4 + k] = bitsOfF(c.points[i * 4 + j][k]);
        }
        if (numPatches == 0) ++cov.zero;
        if (o.patchIndex != INVALID)
        {
            const uint32_t mi0 = c.hdr[2] & 0xFFFFu, mi1 = c.hdr[3] & 0xFFFFu;   // cu:415-416
            const CombinedH cm = combine(mats[mi0], mats[mi1]);
            uint32_t start = 0;   // a PxU32 running sum, stored as PxU16 (:421, :428, :442)
            for (uint32_t i = 0; i < numPatches; ++i)
            {
                refHeader(e, o.patchIndex + i * 64u, c.patchNormal[i], start, c.patchCount[i], cm, mi0, mi1);
                start += c.patchCount[i];
            }
            ++cov.written;
            if (numPatches >= 2) ++cov.multiPatch;
            if ((c.hdr[2] | c.hdr[3]) & 0x10000u) ++cov.highBitWritten;
            if ((mi0 | mi1) & 0xFF00u) ++cov.midByteWritten;
        }
    }

    const DeviceRun g = runDevice("cctMulti", cases, mats, init);
    compareAll("cvxCoreTailBda_multi", g, e);
    checkCoverage("multi", cov, true);
}
