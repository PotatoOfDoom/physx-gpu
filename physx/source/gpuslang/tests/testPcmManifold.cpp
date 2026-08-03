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

// P8.23 — the PCM manifold gate (pcmManifoldBda.spv / pxgSlangPcmManifold).
//
// Suites (substring-matched by the "pcmManifold" ctest filter):
//   pcmManifoldInvalidate   — invalidateManifold's two thresholds, incl. both boundaries
//   pcmManifoldRefreshFrame — refreshManifolds, ONE dispatch, vs a CPU transcription
//   pcmManifoldRefreshMulti — refreshManifolds across SIX dispatches on evolving state
//
// The third fixture is the one that matters and the reason this module was
// extracted before the traversal half.  refreshManifolds is the only stateful
// part of midphaseGeneratePairs: it writes penetration back into a persistent
// record and destructively zeroes mNbManifolds.  A one-shot fixture cannot tell
// a correct port from one that always reports "regenerate everything" — both
// produce the right contacts on frame one.  pcmManifoldRefreshMulti asserts on
// the RECORD between dispatches, so a port that over-invalidates is caught by
// the record that must survive all six frames untouched.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;

struct DevBuf
{
    PxgBufferHandle h = 0;
    uint64_t addr = 0;
};

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// --- the record, mirrored byte for byte -------------------------------------
// PxgPersistentContactManifold.h:68 / :80.  The static_asserts are the point:
// if any of these drifts, the shader's hard-coded offsets are wrong and every
// fixture below would still "pass" against a matching-but-wrong CPU model.
struct Vec3 { float x, y, z; };
struct PcmContact
{
    Vec3 pointA;      // @0
    Vec3 pointB;      // @12
    Vec3 normal;      // @24
    float penetration;// @36
    uint32_t triIndex;// @40
    uint32_t pad;     // @44
};
static_assert(sizeof(PcmContact) == 48, "PxgContact is 48 bytes");

static const uint32_t MAX_MANIFOLDS = 4;
static const uint32_t MAX_CONTACTS = 6;
static const uint32_t MAX_SLOTS = MAX_MANIFOLDS * MAX_CONTACTS;   // 24

struct PcmRecord
{
    PcmContact contacts[MAX_SLOTS];      // @0    .. 1152
    uint32_t nbContacts[MAX_MANIFOLDS];  // @1152 .. 1168
    float relQ[4];                       // @1168 .. 1184
    float relP[4];                       // @1184 .. 1200
    uint32_t nbManifolds;                // @1200
    uint32_t pad[3];                     // -> 1216
};
static_assert(sizeof(PcmRecord) == 1216, "PxgPersistentContactMultiManifold is 1216 bytes");
static_assert(offsetof(PcmRecord, nbContacts) == 1152, "mNbContacts @1152");
static_assert(offsetof(PcmRecord, relQ) == 1168, "mRelativeTransform.q @1168");
static_assert(offsetof(PcmRecord, relP) == 1184, "mRelativeTransform.p @1184");
static_assert(offsetof(PcmRecord, nbManifolds) == 1200, "mNbManifolds @1200");

// --- CPU transcriptions of the two .cu functions ----------------------------
Vec3 sub(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross3(Vec3 a, Vec3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

// PxQuat::rotate: v + 2w(q x v) + 2(q x (q x v)), q = (x,y,z,w)
Vec3 quatRotate(const float q[4], Vec3 v)
{
    const Vec3 qv = { q[0], q[1], q[2] };
    Vec3 t = cross3(qv, v); t = { 2.0f * t.x, 2.0f * t.y, 2.0f * t.z };
    const Vec3 c = cross3(qv, t);
    return { v.x + q[3] * t.x + c.x, v.y + q[3] * t.y + c.y, v.z + q[3] * t.z + c.z };
}

// manifold.cuh:104
bool cpuInvalidate(const PcmRecord& r, const float q[4], Vec3 p, float minMargin, float ratio)
{
    const float thresholdP = minMargin * ratio;
    const float thresholdQ = 0.9998f;
    const Vec3 d = { p.x - r.relP[0], p.y - r.relP[1], p.z - r.relP[2] };
    const float deltaP = std::fmax(std::fabs(d.x), std::fmax(std::fabs(d.y), std::fabs(d.z)));
    const float deltaQ = q[0] * r.relQ[0] + q[1] * r.relQ[1] + q[2] * r.relQ[2] + q[3] * r.relQ[3];
    return (deltaP > thresholdP) || (deltaQ < thresholdQ);
}

// manifold.cuh:40 — mutates the record exactly as the kernel does.
bool cpuRefresh(PcmRecord& r, const float q[4], Vec3 p, float projectBreakingThreshold)
{
    const float sqT = projectBreakingThreshold * projectBreakingThreshold;
    const uint32_t numManifolds = r.nbManifolds;
    bool any = false;
    for (uint32_t lane = 0; lane < MAX_SLOTS; ++lane)
    {
        const uint32_t m = lane / MAX_CONTACTS;
        if (m >= numManifolds) continue;
        const uint32_t c = lane % MAX_CONTACTS;
        if (c >= r.nbContacts[m]) continue;

        PcmContact& ct = r.contacts[lane];
        const Vec3 localAInB = [&] {
            const Vec3 rot = quatRotate(q, ct.pointA);
            return Vec3{ rot.x + p.x, rot.y + p.y, rot.z + p.z };
        }();
        const Vec3 v = sub(localAInB, ct.pointB);
        const float penNew = dot3(v, ct.normal);
        const float penOld = ct.penetration;
        const Vec3 projectedPoint = { localAInB.x - ct.normal.x * penOld,
                                      localAInB.y - ct.normal.y * penOld,
                                      localAInB.z - ct.normal.z * penOld };
        const Vec3 projectedDifference = sub(ct.pointB, projectedPoint);
        const float distance2d = dot3(projectedDifference, projectedDifference);
        ct.penetration = penNew;                 // write-back, unconditional in-slot
        if (distance2d > sqT) any = true;
    }
    if (any) r.nbManifolds = 0;                  // manifold.cuh:97
    return any;
}

// --- a deterministic record builder ----------------------------------------
// Two manifolds, 6 and 3 contacts, with contacts placed so that pointB is
// EXACTLY where the identity transform puts pointA (zero drift) unless the test
// perturbs it.  That makes "does not invalidate" the natural state and forces a
// port that over-reports to fail.
// driftSlot >= 0 restricts the drift to that ONE slot, leaving every other
// contact exactly on its normal.  A record built that way is what separates a
// real group-wide OR from a lane-0-only test: with the drift on slot 7, lane 0
// sees nothing and only the group reduction can report the loss.
PcmRecord makeRecord(uint32_t nbManifolds, const uint32_t nbPerManifold[MAX_MANIFOLDS],
                     float drift, int driftSlot = -1)
{
    PcmRecord r;
    std::memset(&r, 0, sizeof(r));
    r.nbManifolds = nbManifolds;
    for (uint32_t m = 0; m < MAX_MANIFOLDS; ++m) r.nbContacts[m] = nbPerManifold[m];
    r.relQ[0] = 0.0f; r.relQ[1] = 0.0f; r.relQ[2] = 0.0f; r.relQ[3] = 1.0f;   // identity
    r.relP[0] = 0.0f; r.relP[1] = 0.0f; r.relP[2] = 0.0f; r.relP[3] = 0.0f;

    for (uint32_t lane = 0; lane < MAX_SLOTS; ++lane)
    {
        const float f = float(lane);
        PcmContact& c = r.contacts[lane];
        c.pointA = { 0.1f * f, 0.2f + 0.05f * f, -0.3f + 0.02f * f };
        // pointB = pointA displaced along the normal, plus an in-plane drift that
        // the projection test is exactly designed to measure.
        c.normal = { 0.0f, 1.0f, 0.0f };
        c.penetration = -0.01f * (1.0f + 0.1f * f);
        const float d = (driftSlot < 0 || int(lane) == driftSlot) ? drift : 0.0f;
        // pointB is placed so the PROJECTED difference the kernel measures is
        // exactly `d`, and nothing else.  refreshManifolds computes
        //   projectedPoint = localAInB - normal*penOld
        //   projectedDifference = pointB - projectedPoint
        // and with an identity transform localAInB == pointA, so pointB must be
        // pointA - normal*pen + (d,0,0).  Writing `+ penetration` here instead --
        // as this builder first did -- leaves a residual of 2*penetration in the
        // difference, which alone exceeded the threshold and made EVERY slot
        // "lost" regardless of d.  A mutant that reduced the group-wide OR to
        // lane 0 then survived, because lane 0 was always lost too.
        c.pointB = { c.pointA.x + d, c.pointA.y - c.penetration, c.pointA.z };
        c.triIndex = lane;
    }
    return r;
}

std::vector<uint8_t> toBytes(const std::vector<PcmRecord>& recs)
{
    std::vector<uint8_t> b(recs.size() * sizeof(PcmRecord));
    std::memcpy(b.data(), recs.data(), b.size());
    return b;
}

struct InvalidatePush
{
    uint64_t manifolds, outFlags;
    uint32_t count; float minMargin, ratio, pad28;
    float q[4]; float p[4];
};
struct RefreshPush
{
    uint64_t manifolds, outLost;
    uint32_t count; float projectBreakingThreshold; float pad24, pad28;
    float q[4]; float p[4];
};
static_assert(sizeof(InvalidatePush) == 64, "invalidate blob is 64 bytes");
static_assert(sizeof(RefreshPush) == 64, "refresh blob is 64 bytes");

// ---------------------------------------------------------------------------
// invalidateManifold — the two thresholds, and both of their boundaries.
// ---------------------------------------------------------------------------
PXG_TEST(pcmManifoldInvalidate)
{
    PxgSlangComputeDevice* dev = testDevice();

    const float minMargin = 0.25f, ratio = 0.2f;
    const float thresholdP = minMargin * ratio;              // 0.05

    // Each record gets a different STORED relative transform; the dispatch feeds
    // one aToB and every record measures its own delta against it.
    struct Case { float relP[3]; float relQ[4]; const char* what; };
    const Case cases[] = {
        { { 0.0f, 0.0f, 0.0f }, { 0, 0, 0, 1 }, "no motion at all" },
        { { -thresholdP, 0.0f, 0.0f }, { 0, 0, 0, 1 }, "delta EXACTLY thresholdP (must stay valid: > not >=)" },
        { { -0.0501f, 0.0f, 0.0f }, { 0, 0, 0, 1 }, "delta just over thresholdP" },
        { { 0.0f, -0.03f, 0.04f }, { 0, 0, 0, 1 }, "under on every axis but the MAX is what counts" },
        { { 0.0f, 0.0f, 0.0f }, { 0, 0, 0.0099998f, 0.99995f }, "small rotation, dot above 0.9998" },
        { { 0.0f, 0.0f, 0.0f }, { 0, 0.03f, 0, 0.99955f }, "rotation past about one degree" },
        { { 0.0f, 0.0f, 0.0f }, { 0, 0, 0, -1 }, "opposite-sign quaternion: dot is -1, invalidates" },
        // The stored transform is AHEAD of aToB, so the delta is NEGATIVE and only
        // its magnitude crosses the threshold.  Without the abs() this reads as
        // -0.06, which is not > 0.05, and the record wrongly stays valid.
        { { 0.06f, 0.0f, 0.0f }, { 0, 0, 0, 1 }, "NEGATIVE delta whose magnitude is over threshold" },
        { { 0.0f, 0.049f, 0.0f }, { 0, 0, 0, 1 }, "negative delta just under threshold" },
        // aToB.q is identity, so dot(aToB.q, prevQ) is exactly prevQ.w.  Setting it
        // to the threshold itself pins the `<` vs `<=` boundary: 0.9998 is NOT less
        // than 0.9998, so this record must stay VALID.
        { { 0.0f, 0.0f, 0.0f }, { 0, 0, 0.02f, 0.9998f }, "deltaQ EXACTLY thresholdQ (must stay valid: < not <=)" },
    };
    const uint32_t N = uint32_t(sizeof(cases) / sizeof(cases[0]));
    const uint32_t nbPer[MAX_MANIFOLDS] = { 6, 3, 0, 0 };

    std::vector<PcmRecord> recs;
    for (uint32_t i = 0; i < N; ++i)
    {
        PcmRecord r = makeRecord(2, nbPer, 0.0f);
        for (int k = 0; k < 3; ++k) r.relP[k] = cases[i].relP[k];
        for (int k = 0; k < 4; ++k) r.relQ[k] = cases[i].relQ[k];
        recs.push_back(r);
    }

    const float q[4] = { 0.0f, 0.0f, 0.0f, 1.0f };   // aToB: identity rotation
    const Vec3 p = { 0.0f, 0.0f, 0.0f };

    std::vector<uint8_t> bytes = toBytes(recs);
    std::vector<uint32_t> outInit(N, 0xFFFFFFFFu);
    DevBuf bRec = makeDev(dev, bytes);
    DevBuf bOut = makeDev(dev, outInit);

    InvalidatePush push{};
    push.manifolds = bRec.addr; push.outFlags = bOut.addr;
    push.count = N; push.minMargin = minMargin; push.ratio = ratio;
    for (int k = 0; k < 4; ++k) push.q[k] = q[k];
    push.p[0] = p.x; push.p[1] = p.y; push.p[2] = p.z; push.p[3] = 0.0f;

    Kernel k("pcmManifoldBda.spv", "pcmInvalidateLaunch", {}, sizeof(InvalidatePush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<uint32_t>(dev, bOut.h, N);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t exp = cpuInvalidate(recs[i], q, p, minMargin, ratio) ? 1u : 0u;
        if (got[i] != exp)
        {
            ++mm;
            std::printf("  invalidate MM case %u (%s): got=%u exp=%u\n", i, cases[i].what, got[i], exp);
        }
    }
    PXG_CHECK_MSG(mm == 0, "invalidateManifold matches CPU on all %u cases (%zu mismatches)", N, mm);

    dev->destroyBuffer(bRec.h); dev->destroyBuffer(bOut.h);
}

// ---------------------------------------------------------------------------
// refreshManifolds, a SINGLE dispatch: penetration write-back, the per-slot
// guards, the drift test, and the mNbManifolds zeroing.
// ---------------------------------------------------------------------------
PXG_TEST(pcmManifoldRefreshFrame)
{
    PxgSlangComputeDevice* dev = testDevice();

    const float threshold = 0.02f;
    // drift 0.0 stays inside the threshold; drift 0.05 breaks it.  The third
    // record has ONE manifold so slots 6..23 must be skipped; the fourth has
    // zero manifolds so nothing may be touched at all.
    const uint32_t nbA[MAX_MANIFOLDS] = { 6, 3, 0, 0 };
    const uint32_t nbB[MAX_MANIFOLDS] = { 6, 3, 0, 0 };
    const uint32_t nbC[MAX_MANIFOLDS] = { 4, 0, 0, 0 };
    const uint32_t nbD[MAX_MANIFOLDS] = { 6, 6, 6, 6 };

    std::vector<PcmRecord> recs = {
        makeRecord(2, nbA, 0.0f),    // no drift -> survives
        makeRecord(2, nbB, 0.05f),   // drift past threshold -> zeroed
        makeRecord(1, nbC, 0.0f),    // one manifold, 4 contacts
        makeRecord(0, nbD, 0.05f),   // zero manifolds: counts are set but unreachable
        makeRecord(2, nbA, 0.05f, 7), // ONLY slot 7 drifts -> lane 0 sees nothing
    };
    const uint32_t N = uint32_t(recs.size());

    const float q[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const Vec3 p = { 0.0f, 0.0f, 0.0f };

    std::vector<PcmRecord> expect = recs;
    std::vector<uint32_t> expLost(N);
    for (uint32_t i = 0; i < N; ++i) expLost[i] = cpuRefresh(expect[i], q, p, threshold) ? 1u : 0u;

    std::vector<uint8_t> bytes = toBytes(recs);
    std::vector<uint32_t> outInit(N, 0xFFFFFFFFu);
    DevBuf bRec = makeDev(dev, bytes);
    DevBuf bOut = makeDev(dev, outInit);

    RefreshPush push{};
    push.manifolds = bRec.addr; push.outLost = bOut.addr;
    push.count = N; push.projectBreakingThreshold = threshold;
    for (int k = 0; k < 4; ++k) push.q[k] = q[k];
    push.p[0] = p.x; push.p[1] = p.y; push.p[2] = p.z; push.p[3] = 0.0f;

    Kernel k("pcmManifoldBda.spv", "pcmRefreshLaunch", {}, sizeof(RefreshPush));
    k.dispatch(N, {}, push, 1u, 1u);
    dev->flush();

    const auto gotLost = downloadBuffer<uint32_t>(dev, bOut.h, N);
    const auto gotBytes = downloadBuffer<uint8_t>(dev, bRec.h, uint32_t(bytes.size()));
    std::vector<PcmRecord> gotRecs(N);
    std::memcpy(gotRecs.data(), gotBytes.data(), bytes.size());

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (gotLost[i] != expLost[i])
        { ++mm; std::printf("  refresh MM rec %u: lost got=%u exp=%u\n", i, gotLost[i], expLost[i]); }
        if (gotRecs[i].nbManifolds != expect[i].nbManifolds)
        { ++mm; std::printf("  refresh MM rec %u: nbManifolds got=%u exp=%u\n",
                            i, gotRecs[i].nbManifolds, expect[i].nbManifolds); }
        for (uint32_t s = 0; s < MAX_SLOTS; ++s)
        {
            const float g = gotRecs[i].contacts[s].penetration;
            const float e = expect[i].contacts[s].penetration;
            if (g != e)
            { ++mm; std::printf("  refresh MM rec %u slot %u: pen got=%.9g exp=%.9g\n", i, s, g, e); }
        }
    }
    PXG_CHECK_MSG(mm == 0, "refreshManifolds bit-exact vs CPU over %u records (%zu mismatches)", N, mm);

    dev->destroyBuffer(bRec.h); dev->destroyBuffer(bOut.h);

    // ---- second dispatch: a translation ALONG THE CONTACT NORMAL -------------
    // The first dispatch uses an identity transform, which makes penNew equal
    // penOld exactly -- and under that equality `projectedPoint = localAInB -
    // normal*penOld` and the same line with penNew are literally the same
    // expression, so a mutant that swaps them is EQUIVALENT and survives.  The
    // two are only distinguishable when the transform moves the point along its
    // own normal, which is what 0.03 in y does here: with penOld the projected
    // difference carries that 0.03 and breaks the 0.02 threshold, with penNew it
    // cancels and does not.  Different verdict, so the mutant dies.
    //
    // It is a SECOND dispatch rather than a change to the first because the
    // first has a job that this transform would destroy: record 4 drifts on slot
    // 7 alone, which is what forces a real group-wide OR, and a normal-direction
    // translation makes every slot lose contact and lets a lane-0-only
    // reduction pass again.
    {
        const float q2[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        const Vec3 p2 = { 0.0f, 0.03f, 0.0f };

        std::vector<PcmRecord> recs2 = recs;          // fresh, unrefreshed copies
        std::vector<PcmRecord> expect2 = recs2;
        std::vector<uint32_t> expLost2(N);
        for (uint32_t i = 0; i < N; ++i)
            expLost2[i] = cpuRefresh(expect2[i], q2, p2, threshold) ? 1u : 0u;

        std::vector<uint8_t> bytes2 = toBytes(recs2);
        std::vector<uint32_t> outInit2(N, 0xFFFFFFFFu);
        DevBuf bRec2 = makeDev(dev, bytes2);
        DevBuf bOut2 = makeDev(dev, outInit2);

        RefreshPush push2{};
        push2.manifolds = bRec2.addr; push2.outLost = bOut2.addr;
        push2.count = N; push2.projectBreakingThreshold = threshold;
        for (int t = 0; t < 4; ++t) push2.q[t] = q2[t];
        push2.p[0] = p2.x; push2.p[1] = p2.y; push2.p[2] = p2.z; push2.p[3] = 0.0f;

        Kernel k2("pcmManifoldBda.spv", "pcmRefreshLaunch", {}, sizeof(RefreshPush));
        k2.dispatch(N, {}, push2, 1u, 1u);
        dev->flush();

        const auto gotLost2 = downloadBuffer<uint32_t>(dev, bOut2.h, N);
        const auto gotBytes2 = downloadBuffer<uint8_t>(dev, bRec2.h, uint32_t(bytes2.size()));
        std::vector<PcmRecord> gotRecs2(N);
        std::memcpy(gotRecs2.data(), gotBytes2.data(), bytes2.size());

        size_t mm2 = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            if (gotLost2[i] != expLost2[i])
            { ++mm2; std::printf("  refresh(normal) MM rec %u: lost got=%u exp=%u\n",
                                 i, gotLost2[i], expLost2[i]); }
            if (gotRecs2[i].nbManifolds != expect2[i].nbManifolds)
            { ++mm2; std::printf("  refresh(normal) MM rec %u: nbManifolds got=%u exp=%u\n",
                                 i, gotRecs2[i].nbManifolds, expect2[i].nbManifolds); }
            for (uint32_t sl = 0; sl < MAX_SLOTS; ++sl)
            {
                const float g = gotRecs2[i].contacts[sl].penetration;
                const float e = expect2[i].contacts[sl].penetration;
                if (g != e)
                { ++mm2; std::printf("  refresh(normal) MM rec %u slot %u: pen got=%.9g exp=%.9g\n",
                                     i, sl, g, e); }
            }
        }
        PXG_CHECK_MSG(mm2 == 0,
                      "refreshManifolds bit-exact under a normal-direction move (%zu mismatches)", mm2);

        dev->destroyBuffer(bRec2.h); dev->destroyBuffer(bOut2.h);
    }
}

// ---------------------------------------------------------------------------
// refreshManifolds across SIX dispatches on evolving state.
//
// THIS is the fixture the module exists for.  The record is re-dispatched with
// a transform that creeps, and the CPU model is carried forward in lockstep, so
// every frame's input is the PREVIOUS frame's output.  Two properties a
// one-shot fixture cannot check:
//   - the penetration write-back COMPOUNDS: frame n's projection uses the
//     penetration frame n-1 stored, so a port that wrote penNew before taking
//     the projection diverges on frame 2, not frame 1.
//   - record 0 must survive all six frames with mNbManifolds INTACT.  A port
//     that over-reports "lost" — the failure mode that looks perfect in every
//     single-dispatch test, because regenerating everything still yields correct
//     contacts — zeroes it and is caught here.
// ---------------------------------------------------------------------------
PXG_TEST(pcmManifoldRefreshMulti)
{
    PxgSlangComputeDevice* dev = testDevice();

    const float threshold = 0.05f;
    const uint32_t nbA[MAX_MANIFOLDS] = { 6, 6, 0, 0 };
    const uint32_t nbB[MAX_MANIFOLDS] = { 5, 2, 1, 0 };

    std::vector<PcmRecord> recs = {
        makeRecord(2, nbA, 0.0f),     // stays inside the threshold for all six frames
        makeRecord(3, nbB, 0.0f),     // ragged contact counts, also survives
    };
    const uint32_t N = uint32_t(recs.size());

    std::vector<PcmRecord> model = recs;
    std::vector<uint8_t> bytes = toBytes(recs);
    std::vector<uint32_t> outInit(N, 0xFFFFFFFFu);
    DevBuf bRec = makeDev(dev, bytes);
    DevBuf bOut = makeDev(dev, outInit);

    Kernel k("pcmManifoldBda.spv", "pcmRefreshLaunch", {}, sizeof(RefreshPush));

    size_t mm = 0;
    const int FRAMES = 6;
    for (int f = 0; f < FRAMES; ++f)
    {
        // A translation that creeps along y by a hair each frame.  It moves the
        // contact along its own normal, which changes penetration every frame
        // while leaving the IN-PLANE projection (the thing the threshold tests)
        // essentially still — so a faithful port never loses a contact, and one
        // that mixes penOld/penNew drifts out of the band and does.
        const float dy = 0.002f * float(f + 1);
        // A small but REAL rotation about z.  With an identity quaternion the
        // vector part is zero, so quatRotate's `q.w * t` term vanishes and a port
        // that dropped it would look correct; a tenth of a degree is enough to
        // make the term load-bearing while keeping every contact inside the
        // threshold, so the records still have to survive all six frames.
        const float halfAngle = 0.00087266f;   // ~0.1 degree, halved
        const float q[4] = { 0.0f, 0.0f, std::sin(halfAngle), std::cos(halfAngle) };
        const Vec3 p = { 0.0f, dy, 0.0f };

        RefreshPush push{};
        push.manifolds = bRec.addr; push.outLost = bOut.addr;
        push.count = N; push.projectBreakingThreshold = threshold;
        for (int t = 0; t < 4; ++t) push.q[t] = q[t];
        push.p[0] = p.x; push.p[1] = p.y; push.p[2] = p.z; push.p[3] = 0.0f;

        k.dispatch(N, {}, push, 1u, 1u);
        dev->flush();

        std::vector<uint32_t> expLost(N);
        for (uint32_t i = 0; i < N; ++i) expLost[i] = cpuRefresh(model[i], q, p, threshold) ? 1u : 0u;

        const auto gotLost = downloadBuffer<uint32_t>(dev, bOut.h, N);
        const auto gotBytes = downloadBuffer<uint8_t>(dev, bRec.h, uint32_t(bytes.size()));
        std::vector<PcmRecord> gotRecs(N);
        std::memcpy(gotRecs.data(), gotBytes.data(), bytes.size());

        for (uint32_t i = 0; i < N; ++i)
        {
            if (gotLost[i] != expLost[i])
            { ++mm; std::printf("  multi MM f%d rec %u: lost got=%u exp=%u\n", f, i, gotLost[i], expLost[i]); }
            if (gotRecs[i].nbManifolds != model[i].nbManifolds)
            { ++mm; std::printf("  multi MM f%d rec %u: nbManifolds got=%u exp=%u\n",
                                f, i, gotRecs[i].nbManifolds, model[i].nbManifolds); }
            // The whole point: nothing may have been invalidated.
            if (gotRecs[i].nbManifolds == 0u)
            { ++mm; std::printf("  multi MM f%d rec %u: record was INVALIDATED but must survive\n", f, i); }
            for (uint32_t s = 0; s < MAX_SLOTS; ++s)
            {
                const float g = gotRecs[i].contacts[s].penetration;
                const float e = model[i].contacts[s].penetration;
                if (g != e)
                { ++mm; std::printf("  multi MM f%d rec %u slot %u: pen got=%.9g exp=%.9g\n", f, i, s, g, e); }
            }
        }
        if (mm) break;   // one bad frame is enough; later frames would just echo it
    }
    PXG_CHECK_MSG(mm == 0, "refreshManifolds bit-exact across %d dispatches (%zu mismatches)", FRAMES, mm);

    dev->destroyBuffer(bRec.h); dev->destroyBuffer(bOut.h);
}

} // namespace
