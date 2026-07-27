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

// P8.0 SOFTBODY (FEM-volume) bring-up test: the softbody free-fall
// predict/integrate kernel (softBodySimBda.spv) on the engine ABI (BDA), driven by
// the REAL engine descriptor.  A genuine PxgSoftBody (gpusimulationcontroller
// header) is filled with device buffer addresses + scalars, DMA'd to the device,
// and read back through the Slang pxgSlangSoftBodyDesc mirror — the first-ever
// PxgSoftBody BDA round-trip.  static_asserts / offsetof checks pin the mirror's
// byte offsets against the real engine struct (the highest-risk item, sizeof ==
// 432), and the kernel's output is compared BIT-EXACT against a CPU transcription
// of the softBodyGM.cu math.
//
// Suites (substring-matched by the "softBodySim" ctest filter):
//   softBodySimDescLayout        — byte-offset proof of the descriptor mirror
//   softBodySimPreIntegratePGS   — sb_gm_preIntegrateLaunch PGS predict+integrate
//   softBodySimPreIntegrateTGS   — sb_gm_preIntegrateLaunch TGS branch (vel + zero posDelta)

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers the engine
// descriptor header decorates its inline members with; must precede PxgSoftBody.h.
#include "host_defines.h"
#include "PxgSoftBody.h"                     // real engine per-softbody descriptor
#include "PxActor.h"                         // PxActorFlag::eDISABLE_GRAVITY
#include "PxDeformableBodyFlag.h"            // PxDeformableBodyFlag::eKINEMATIC / eENABLE_SPECULATIVE_CCD
#include "PxDeformableVolumeFlag.h"          // PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };

// A device buffer + its buffer-device-address (BDA).
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

// Same literal the shader uses for PX_MAX_REAL; keeping both sides identical makes
// the max-velocity clamp `maxVel != PX_MAX_REAL` short-circuit false (clamp dead),
// so the 1/sqrt transcendental never feeds the output (no bit-exactness seam).
static const float PX_MAX_REAL_F = 3.40282347e+38f;

// The push-constant blob, byte-identical to softBodySimBda.slang's
// [[vk::push_constant]] PreIntegrateParams (which the shim reproduces from the CUDA
// arg pack).  ABI row {8,8,12,4,1,8,1} -> 56 bytes.
struct PrePush
{
    uint64_t softbodies, activeId;
    float gx, gy, gz, dt;
    uint32_t isTGSWord, pad36;
    uint64_t speculativeCCD;
    uint32_t extForcesWord, pad52;
};
static_assert(sizeof(PrePush) == 56, "PreIntegrate push-constant blob must be 56 bytes");

// Zeroed real PxgSoftBody (no vtable — memset-safe, like PxgFEMCloth / PxgParticleSystem).
static PxgSoftBody makeZeroDesc()
{
    PxgSoftBody c;
    std::memset(&c, 0, sizeof(c));
    return c;
}

// ---------------------------------------------------------------------------
// Descriptor mirror layout proof — pins pxgSlangSoftBodyDesc against the real
// engine PxgSoftBody (offsets the kernel dereferences).  sizeof == 432.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimDescLayout)
{
    // sizeof / array stride: the kernel indexes softbodies[id], so the STRIDE must
    // match (verified in SPIR-V as ArrayStride 432).
    PXG_CHECK_MSG(sizeof(PxgSoftBody) == 432, "PxgSoftBody sizeof == 432 (got %zu)", sizeof(PxgSoftBody));

    // device-pointer members the pre-integrate driver reads/writes
    PXG_CHECK(offsetof(PxgSoftBody, mSimVelocity_InvMass) == 88);
    PXG_CHECK(offsetof(PxgSoftBody, mSimPosition_InvMass) == 96);
    PXG_CHECK(offsetof(PxgSoftBody, mSimKinematicTarget) == 104);
    PXG_CHECK(offsetof(PxgSoftBody, mVertsAreDeformed) == 112);
    PXG_CHECK(offsetof(PxgSoftBody, mVertsCanNotDeform) == 120);
    PXG_CHECK(offsetof(PxgSoftBody, mSimDeltaPos) == 128);

    // scalar tail
    PXG_CHECK(offsetof(PxgSoftBody, mLinearDamping) == 316);
    PXG_CHECK(offsetof(PxgSoftBody, mMaxLinearVelocity) == 320);
    PXG_CHECK(offsetof(PxgSoftBody, mPenBiasClamp) == 324);
    PXG_CHECK(offsetof(PxgSoftBody, mNumVertsGM) == 372);
    PXG_CHECK(offsetof(PxgSoftBody, mElementIndex) == 384);
    PXG_CHECK(offsetof(PxgSoftBody, mGpuRemapIndex) == 388);

    // the byte-packed flag triple (@392): byte0 == mActorFlags, byte1 == mBodyFlags,
    // bytes2..3 == mVolumeFlags -> the shader's mFlagsWord fold.
    PXG_CHECK(offsetof(PxgSoftBody, mActorFlags) == 392);
    PXG_CHECK(offsetof(PxgSoftBody, mBodyFlags) == 393);
    PXG_CHECK(offsetof(PxgSoftBody, mVolumeFlags) == 394);
    PXG_CHECK(offsetof(PxgSoftBody, mSimPullIndices) == 400); // realigned 8-byte ptr after the flag word
    PXG_CHECK(offsetof(PxgSoftBody, mAwake) == 408);
    PXG_CHECK(offsetof(PxgSoftBody, mRestDistance) == 420);

    // flag bit values the mirror hard-codes (pxgSlangSoftBodyDesc constants).
    PXG_CHECK(PxActorFlag::eDISABLE_GRAVITY == (1 << 1));                     // 2
    PXG_CHECK(PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD == (1 << 1));     // 2
    PXG_CHECK(PxDeformableBodyFlag::eKINEMATIC == (1 << 2));                  // 4
    PXG_CHECK(PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC == (1 << 1));      // 2
}

// ---------------------------------------------------------------------------
// Shared preIntegrate driver: dispatch + bit-exact CPU compare for a given
// isTGS mode.  invMass encoded in .w; vertex 3 has w==0 (fixed -> no delta),
// the rest are free (invMass != 0 -> fall).  Non-kinematic softbody (mBodyFlags
// == 0, mVolumeFlags == 0), gravity enabled (mActorFlags == 0).
// ---------------------------------------------------------------------------
static void runPreIntegrate(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const float3 g = { 0.0f, -9.81f, 0.0f };
    const float linearDamping = 0.1f;
    const float maxLinVel = PX_MAX_REAL_F;   // default -> clamp dead, no sqrt seam
    const uint32_t actorFlags = 0u;          // gravity enabled

    std::vector<F4> positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 3) ? 0.0f : 1.0f; // vertex 3 is fixed (invMass == 0)
        positions[i] = { float(i) * 1.0f, 5.0f + float(i) * 0.25f, -float(i) * 0.5f, w };
        vels[i] = { 0.1f * float(i), -0.2f * float(i), 0.05f * float(i), 0.0f };
    }
    std::vector<F4> posDelta(N, { 9.f, 9.f, 9.f, 9.f });
    std::vector<float> speculativeCCD = { 7.0f };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, posDelta);
    DevBuf bCcd = makeDev(dev, speculativeCCD);

    PxgSoftBody c = makeZeroDesc();
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mSimKinematicTarget = nullptr;         // no kinematic target on the free-fall scene
    c.mLinearDamping = linearDamping;
    c.mMaxLinearVelocity = maxLinVel;
    c.mNumVertsGM = N;
    c.mActorFlags = PxU8(actorFlags);
    c.mBodyFlags = 0u;                        // not eKINEMATIC / not eENABLE_SPECULATIVE_CCD
    c.mVolumeFlags = 0u;                      // not ePARTIALLY_KINEMATIC
    c.mAwake = 1u;                            // gThreadIdx==0 must reset this to 0

    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    PrePush push = {};
    push.softbodies = bDesc.addr;
    push.activeId = bActive.addr;
    push.gx = g.x; push.gy = g.y; push.gz = g.z; push.dt = dt;
    push.isTGSWord = isTGS ? 1u : 0u;
    push.speculativeCCD = bCcd.addr;
    push.extForcesWord = 0u;                  // externalForcesEveryTgsIterationEnabled = false

    Kernel k("softBodySimBda.spv", "sb_gm_preIntegrateLaunch", {}, sizeof(PrePush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference (bit-exact expression order, matching softBodyGM.cu:72 else-branch:
    // non-kinematic softbody, no partially-kinematic target).
    std::vector<F4> refPos = positions, refVel = vels, refDelta = posDelta;
    for (uint32_t i = 0; i < N; ++i)
    {
        F4 tPos = positions[i];
        float vx = 0.f, vy = 0.f, vz = 0.f;
        const float ox = tPos.x, oy = tPos.y, oz = tPos.z;

        // kinematic == false (mVolumeFlags has no ePARTIALLY_KINEMATIC)
        if (tPos.w != 0.0f)
        {
            vx = vels[i].x; vy = vels[i].y; vz = vels[i].z;
            // !externalForcesEveryTgsIterationEnabled
            // actorFlags has no eDISABLE_GRAVITY -> apply gravity
            vx += g.x * dt; vy += g.y * dt; vz += g.z * dt;
            const float damping = 1.0f - std::fmin(dt * linearDamping, 1.0f);
            vx *= damping; vy *= damping; vz *= damping;
        }

        // Max velocity clamping (dead: maxVel == PX_MAX_REAL -> ternary picks vel).
        const float maxVel = maxLinVel;
        const float velMagSq = vx * vx + vy * vy + vz * vz;
        if (maxVel != PX_MAX_REAL_F && maxVel * maxVel < velMagSq)
        {
            const float s = maxVel / std::sqrt(velMagSq);
            vx *= s; vy *= s; vz *= s;
        }

        float dx = 0.f, dy = 0.f, dz = 0.f;
        if (!isTGS) { dx += vx * dt; dy += vy * dt; dz += vz * dt; }

        refPos[i] = { ox + dx, oy + dy, oz + dz, tPos.w };
        refVel[i] = { vx, vy, vz, tPos.w };
        refDelta[i] = { dx, dy, dz, tPos.w };
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);
    const auto gDesc = downloadBuffer<PxgSoftBody>(dev, bDesc.h, 1);

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8)
                std::printf("  %s MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", what, i,
                            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp4(refPos[i], gPos[i], "pos", i);
        cmp4(refVel[i], gVel[i], "vel", i);
        cmp4(refDelta[i], gDelta[i], "delta", i);
    }
    // globalThreadIndex==0 resets mAwake to 0 (PxU32 store).
    if (gDesc[0].mAwake != 0u) { ++mm; std::printf("  mAwake MM: %u (expected 0)\n", gDesc[0].mAwake); }

    PXG_CHECK_MSG(mm == 0, "sb_gm_preIntegrateLaunch (%s) bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm);

    // Sanity beacons the bit-exact compare already implies, kept human-readable:
    //  - free vertex 0 (invMass != 0) falls: PGS y-position strictly below its start.
    if (!isTGS)
        PXG_CHECK_MSG(gPos[0].y < positions[0].y,
                      "free vertex 0 should fall (y %g -> %g)", positions[0].y, gPos[0].y);
    //  - fixed vertex 3 (invMass == 0) never moves and gets a zero delta.
    PXG_CHECK_MSG(f2u(gPos[3].x) == f2u(positions[3].x) && f2u(gPos[3].y) == f2u(positions[3].y) &&
                  f2u(gPos[3].z) == f2u(positions[3].z),
                  "fixed vertex 3 must not move");
    PXG_CHECK_MSG(gDelta[3].x == 0.f && gDelta[3].y == 0.f && gDelta[3].z == 0.f,
                  "fixed vertex 3 delta must be zero");
    //  - TGS: delta stays zero everywhere but velocity still received gravity.
    if (isTGS)
    {
        PXG_CHECK_MSG(gDelta[0].x == 0.f && gDelta[0].y == 0.f && gDelta[0].z == 0.f,
                      "TGS free vertex 0 delta must be zero");
        PXG_CHECK_MSG(gVel[0].y < vels[0].y, "TGS free vertex 0 velocity must receive gravity");
    }

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bCcd.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimPreIntegratePGS) { runPreIntegrate(/*isTGS*/ false); }
PXG_TEST(softBodySimPreIntegrateTGS) { runPreIntegrate(/*isTGS*/ true); }


// ---------------------------------------------------------------------------
// P8.0 tranche 2 — sb_refitBoundLaunch (softBody.cu:99, softBodyRefitBda.spv).
//
// The softbody BVH bounds refit: walk the cooked Gu::BV32DataPacked TETRAHEDRON tree
// bottom-up, rewrite every node's 32 per-child AABBs from the CURRENT
// mPosition_InvMass, aggregate each node's own bound into mPackedNodeBounds, and publish
// the root bound into the broad-phase boundArray slot (verbatim — the broad phase adds
// the contact distance itself).  Without it a moving softbody stays where it was cooked
// as far as the broad phase and every midphase query are concerned.
//
// The test tree is TWO levels deep so BOTH aggregation paths run:
//   node 0 (root, level 0): 8 INTERNAL children -> nodes 1..8
//   nodes 1..8 (level 1): ONE leaf child each, holding exactly ONE tet
// One tet per NODE (not merely per leaf child) is what gives the case its teeth: the
// kernel's leaf walk aggregates across ALL of a node's leaf children and then writes that
// single aggregate into every child slot, so several extremal vertices under one node mask
// each other.  With one tet per node, mPackedNodeBounds[i+1] IS tet i's own bound.
// Level 1 exercises the leaf tetrahedron min/max ballot walk; level 0 exercises the
// `!isLeaf -> read mPackedNodeBounds[childOffset]` path, which is only correct if level 1
// really ran first.  The speculative-CCD branch is covered BOTH ways (flag off = the
// contact-distance slots must stay untouched; flag on = both slots get
// speculativeCCDContactOffset + mOriginalContactOffset).
// ---------------------------------------------------------------------------
struct SbRefitPush
{
    uint64_t softBodies, activeSoftBodies;
    uint32_t nbActive, pad20;
    uint64_t bpContactDistance, npContactDistance, speculativeCCD, boundArray, elemIndex;
};
static_assert(sizeof(SbRefitPush) == 64, "sb refitBound push blob must be 64 bytes");
static_assert(offsetof(SbRefitPush, bpContactDistance) == 24, "bpContactDistance @24");
static_assert(offsetof(SbRefitPush, boundArray) == 48, "boundArray @48");
static_assert(offsetof(SbRefitPush, elemIndex) == 56, "elemIndex @56");

namespace p80refit
{
static const uint32_t BV32_WORDS = 292u;   // Gu::BV32DataPacked = 1168 B
static const uint32_t W_MMIN  = 0u;
static const uint32_t W_MMAX  = 128u;
static const uint32_t W_MDATA = 256u;
static const uint32_t W_NBNODES = 288u;
static const uint32_t W_DEPTH = 289u;

static uint32_t leafWord(uint32_t startIndex, uint32_t nbPrims) { return (startIndex << 7) | (nbPrims << 1) | 1u; }
static uint32_t internalWord(uint32_t childOffset) { return childOffset << 11; }

struct V3 { float x, y, z; };
static float fmn(float a, float b) { return a < b ? a : b; }
static float fmx(float a, float b) { return a > b ? a : b; }
static V3 vmin(V3 a, V3 b) { return V3{ fmn(a.x, b.x), fmn(a.y, b.y), fmn(a.z, b.z) }; }
static V3 vmax(V3 a, V3 b) { return V3{ fmx(a.x, b.x), fmx(a.y, b.y), fmx(a.z, b.z) }; }

struct Bounds { V3 mn, mx; };
struct TetIdx4 { uint32_t x, y, z, w; };
struct TreeBlob
{
    std::vector<uint32_t> w;
    uint32_t nodeWord(uint32_t node) const { return 4u + node * BV32_WORDS; }
};

// The CUDA kernel's 32-lane semantics, transcribed exactly (-ffp-contract=off): the leaf
// walk broadcasts a warp-wide min/max into EVERY lane, non-leaf lanes then OVERWRITE
// their own lane value with the child's stored bound, and the node aggregate reduces over
// all 32 lanes (lanes >= nbChildren still carry the leaf aggregate or the +-MAX identity,
// exactly what CUDA reduces).
static void refitNodeRef(uint32_t nodeIndex, TreeBlob& tree, std::vector<Bounds>& packed,
                         const std::vector<F4>& pos, const std::vector<TetIdx4>& tets)
{
    const uint32_t nw = tree.nodeWord(nodeIndex);
    const uint32_t nbChildren = tree.w[nw + W_NBNODES];
    const float MAXF = 3.4028234663852886e+38f;

    std::vector<V3> laneMin(32, V3{ MAXF, MAXF, MAXF });
    std::vector<V3> laneMax(32, V3{ -MAXF, -MAXF, -MAXF });

    uint32_t resultWarp = 0u;
    for (uint32_t k = 0; k < 32u; ++k)
        if (k < nbChildren && (tree.w[nw + W_MDATA + k] & 1u)) resultWarp |= (1u << k);

    for (uint32_t iw = 0; iw < 32u; ++iw)
    {
        if (((resultWarp >> iw) & 1u) == 0u) continue;
        const uint32_t d = tree.w[nw + W_MDATA + iw];
        const uint32_t nbTets = (d >> 1) & 63u;
        const uint32_t startIndex = d >> 7;

        std::vector<V3> tMin = laneMin, tMax = laneMax;
        for (uint32_t lane = 0; lane < 32u; ++lane)
        {
            if (lane >= nbTets) continue;
            const TetIdx4& t = tets[startIndex + lane];
            const F4& a = pos[t.x]; const F4& b = pos[t.y]; const F4& c = pos[t.z]; const F4& e = pos[t.w];
            // pairwise fold, exactly as softBody.cu does it
            float tx = fmn(a.x, b.x),  ty = fmn(a.y, b.y),  tz = fmn(a.z, b.z);
            float tx2 = fmn(c.x, e.x), ty2 = fmn(c.y, e.y), tz2 = fmn(c.z, e.z);
            tx = fmn(tx, tx2); ty = fmn(ty, ty2); tz = fmn(tz, tz2);
            tMin[lane] = vmin(tMin[lane], V3{ tx, ty, tz });
            tx = fmx(a.x, b.x);  ty = fmx(a.y, b.y);  tz = fmx(a.z, b.z);
            tx2 = fmx(c.x, e.x); ty2 = fmx(c.y, e.y); tz2 = fmx(c.z, e.z);
            tx = fmx(tx, tx2); ty = fmx(ty, ty2); tz = fmx(tz, tz2);
            tMax[lane] = vmax(tMax[lane], V3{ tx, ty, tz });
        }
        V3 rn = tMin[0], rx = tMax[0];
        for (uint32_t lane = 1; lane < 32u; ++lane) { rn = vmin(rn, tMin[lane]); rx = vmax(rx, tMax[lane]); }
        for (uint32_t lane = 0; lane < 32u; ++lane) { laneMin[lane] = rn; laneMax[lane] = rx; }
    }

    for (uint32_t lane = 0; lane < 32u; ++lane)
        if (lane < nbChildren && !(tree.w[nw + W_MDATA + lane] & 1u))
        {
            const uint32_t childOffset = tree.w[nw + W_MDATA + lane] >> 11;
            laneMin[lane] = packed[childOffset].mn;
            laneMax[lane] = packed[childOffset].mx;
        }

    for (uint32_t lane = 0; lane < nbChildren; ++lane)
    {
        std::memcpy(&tree.w[nw + W_MMIN + lane * 4 + 0], &laneMin[lane].x, 4);
        std::memcpy(&tree.w[nw + W_MMIN + lane * 4 + 1], &laneMin[lane].y, 4);
        std::memcpy(&tree.w[nw + W_MMIN + lane * 4 + 2], &laneMin[lane].z, 4);
        tree.w[nw + W_MMIN + lane * 4 + 3] = f2u(0.0f);
        std::memcpy(&tree.w[nw + W_MMAX + lane * 4 + 0], &laneMax[lane].x, 4);
        std::memcpy(&tree.w[nw + W_MMAX + lane * 4 + 1], &laneMax[lane].y, 4);
        std::memcpy(&tree.w[nw + W_MMAX + lane * 4 + 2], &laneMax[lane].z, 4);
        tree.w[nw + W_MMAX + lane * 4 + 3] = f2u(0.0f);
    }

    V3 an = laneMin[0], ax = laneMax[0];
    for (uint32_t lane = 1; lane < 32u; ++lane) { an = vmin(an, laneMin[lane]); ax = vmax(ax, laneMax[lane]); }
    packed[nodeIndex].mn = an;
    packed[nodeIndex].mx = ax;
}
} // namespace p80refit

static void runSbRefitBound(bool speculativeCCD)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    // Geometry built for MUTATION TEETH, not realism.  A plain shared-vertex mesh cannot
    // observe the four tet slots independently: mutants that fold worldV2 or worldV3 out
    // of the min/max still reproduced the exact reference bounds, because those vertices
    // also appeared in other slots of other tets in the SAME leaf, which re-supplied the
    // extreme.  So each tet is its OWN leaf child, and carries in ONE chosen slot a vertex
    // that is uniquely extremal (in all three axes) within that leaf — v8/v9 for the x
    // slot, v10/v11 for y, v12/v13 for z, v14/v15 for w, min and max respectively.
    // Each tet is isolated in its own NODE (see the header), so dropping any single slot
    // in any axis, min or max, moves that node's bound and the test catches it
    // (verified: 8/8 slot mutants caught, all 8 pass unmutated).
    std::vector<F4> pos = {
        { -0.31718f,  0.90211f, -0.14477f, 0.5f },   // v0..v7: the ordinary interior
        {  1.17093f, -0.42388f,  0.63512f, 0.5f },
        {  0.55401f,  1.30877f,  0.21939f, 0.5f },
        { -1.02664f,  0.13205f,  1.44170f, 0.5f },
        {  2.31882f,  0.77519f, -0.90613f, 0.5f },
        {  0.04471f, -1.15938f,  0.38204f, 0.5f },
        { -0.68350f,  2.02117f,  1.09926f, 0.5f },
        {  1.83007f,  0.29644f, -1.27735f, 0.5f },
        { -3.11482f, -3.23617f, -3.17094f, 0.5f },   // v8  x-slot MIN
        {  4.13275f,  4.27618f,  4.19043f, 0.5f },   // v9  x-slot MAX
        { -3.53914f, -3.61228f, -3.47651f, 0.5f },   // v10 y-slot MIN
        {  4.61307f,  4.73862f,  4.67195f, 0.5f },   // v11 y-slot MAX
        { -4.07226f, -4.19533f, -4.13478f, 0.5f },   // v12 z-slot MIN
        {  5.09841f,  5.21374f,  5.17662f, 0.5f },   // v13 z-slot MAX
        { -4.61395f, -4.73208f, -4.67517f, 0.5f },   // v14 w-slot MIN
        {  5.63118f,  5.71429f,  5.69236f, 0.5f },   // v15 w-slot MAX
    };
    std::vector<TetIdx4> tets = {
        {  8u,  0u,  1u,  2u },   // leaf 0 — x slot min
        {  9u,  1u,  2u,  3u },   // leaf 1 — x slot max
        {  0u, 10u,  2u,  3u },   // leaf 2 — y slot min
        {  1u, 11u,  3u,  4u },   // leaf 3 — y slot max
        {  2u,  3u, 12u,  5u },   // leaf 4 — z slot min
        {  3u,  4u, 13u,  6u },   // leaf 5 — z slot max
        {  4u,  5u,  6u, 14u },   // leaf 6 — w slot min
        {  5u,  6u,  7u, 15u },   // leaf 7 — w slot max
    };

    const uint32_t nbNodes = 9u, maxDepth = 2u;   // root + one node per tet
    const float POISON = -7777.0f;

    TreeBlob tree;
    // + 4 trailing words: llvmpipe widens the final remapPackedNodeIndex load past the
    // last element, so a blob ending exactly on it can fault (the femClothSimRefitBound
    // precedent — a load-width artifact, not a kernel off-by-one).
    tree.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes + 4u, 0u);
    tree.w[0] = uint32_t(pos.size());
    tree.w[1] = uint32_t(tets.size());
    tree.w[2] = maxDepth;
    tree.w[3] = nbNodes;

    for (uint32_t n = 0; n < nbNodes; ++n)
        for (uint32_t k = 0; k < 32u * 4u; ++k)
        {
            tree.w[tree.nodeWord(n) + W_MMIN + k] = f2u(POISON);
            tree.w[tree.nodeWord(n) + W_MMAX + k] = f2u(POISON);
        }

    // root: 8 INTERNAL children -> nodes 1..8
    for (uint32_t k2 = 0; k2 < 8u; ++k2)
        tree.w[tree.nodeWord(0) + W_MDATA + k2] = internalWord(1u + k2);
    tree.w[tree.nodeWord(0) + W_NBNODES] = 8u;
    tree.w[tree.nodeWord(0) + W_DEPTH] = 0u;
    // nodes 1..8: ONE leaf child each, holding tet (n-1) alone
    for (uint32_t n = 1; n <= 8u; ++n)
    {
        tree.w[tree.nodeWord(n) + W_MDATA + 0] = leafWord(n - 1u, 1u);
        tree.w[tree.nodeWord(n) + W_NBNODES] = 1u;
        tree.w[tree.nodeWord(n) + W_DEPTH] = 1u;
    }

    const uint32_t wDepthInfo = 4u + nbNodes * BV32_WORDS;
    tree.w[wDepthInfo + 0] = 0u; tree.w[wDepthInfo + 1] = 1u;   // level 0: node 0
    tree.w[wDepthInfo + 2] = 1u; tree.w[wDepthInfo + 3] = 8u;   // level 1: nodes 1..8
    const uint32_t wRemap = wDepthInfo + maxDepth * 2u;
    for (uint32_t n = 0; n < nbNodes; ++n) tree.w[wRemap + n] = n;   // identity

    const uint32_t sbId = 0u, elemIdx = 2u;
    const float origContactOffset = 0.037f, specCCD = 0.011f;

    std::vector<float> packedBoundsInit(nbNodes * 6u, POISON);
    std::vector<float> boundArrayInit((elemIdx + 2u) * 6u, POISON);
    std::vector<float> bpDist(elemIdx + 2u, POISON), npDist(elemIdx + 2u, POISON);

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bTet = makeDev(dev, tets);
    DevBuf bTree = makeDev(dev, tree.w);
    DevBuf bPacked = makeDev(dev, packedBoundsInit);
    DevBuf bBounds = makeDev(dev, boundArrayInit);
    DevBuf bBp = makeDev(dev, bpDist);
    DevBuf bNp = makeDev(dev, npDist);
    std::vector<uint32_t> active = { sbId };
    DevBuf bActive = makeDev(dev, active);
    std::vector<uint32_t> elemIndices = { elemIdx };
    DevBuf bElem = makeDev(dev, elemIndices);
    std::vector<float> spec = { specCCD };
    DevBuf bSpec = makeDev(dev, spec);

    PxgSoftBody c = makeZeroDesc();
    c.mTetMeshData = reinterpret_cast<decltype(c.mTetMeshData)>(size_t(bTree.addr));
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mTetIndices = reinterpret_cast<decltype(c.mTetIndices)>(size_t(bTet.addr));
    c.mPackedNodeBounds = reinterpret_cast<decltype(c.mPackedNodeBounds)>(size_t(bPacked.addr));
    c.mOriginalContactOffset = origContactOffset;
    if (speculativeCCD)
        c.mBodyFlags = PxDeformableBodyFlags(PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD);
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    SbRefitPush rp{};
    rp.softBodies = bDesc.addr;
    rp.activeSoftBodies = bActive.addr;
    rp.nbActive = 1u;
    rp.bpContactDistance = bBp.addr;
    rp.npContactDistance = bNp.addr;
    rp.speculativeCCD = bSpec.addr;
    rp.boundArray = bBounds.addr;
    rp.elemIndex = bElem.addr;

    Kernel k("softBodyRefitBda.spv", "sb_refitBoundLaunch", {}, sizeof(SbRefitPush));
    k.dispatch(1u, {}, rp, 1u, 1u);
    dev->flush();

    // --- CPU reference (same bottom-up order the kernel walks) ------------------
    TreeBlob refTree = tree;
    std::vector<Bounds> refPacked(nbNodes);
    for (uint32_t n = 0; n < nbNodes; ++n)
    { refPacked[n].mn = V3{ POISON, POISON, POISON }; refPacked[n].mx = refPacked[n].mn; }
    for (uint32_t i = maxDepth; i > 0u; --i)
    {
        const uint32_t off = refTree.w[wDepthInfo + (i - 1u) * 2u + 0];
        const uint32_t cnt = refTree.w[wDepthInfo + (i - 1u) * 2u + 1];
        for (uint32_t j = 0; j < cnt; ++j)
            refitNodeRef(refTree.w[wRemap + j + off], refTree, refPacked, pos, tets);
    }

    const auto gTree = downloadBuffer<uint32_t>(dev, bTree.h, tree.w.size());
    const auto gPacked = downloadBuffer<float>(dev, bPacked.h, packedBoundsInit.size());
    const auto gBounds = downloadBuffer<float>(dev, bBounds.h, boundArrayInit.size());
    const auto gBp = downloadBuffer<float>(dev, bBp.h, bpDist.size());
    const auto gNp = downloadBuffer<float>(dev, bNp.h, npDist.size());

    size_t mm = 0;
    for (uint32_t n = 0; n < nbNodes; ++n)
    {
        const uint32_t nw = refTree.nodeWord(n);
        for (uint32_t k2 = 0; k2 < 32u * 4u; ++k2)
        {
            if (gTree[nw + W_MMIN + k2] != refTree.w[nw + W_MMIN + k2])
            { ++mm; std::printf("  node %u mMin word %u MM: ref %08x got %08x\n", n, k2,
                                refTree.w[nw + W_MMIN + k2], gTree[nw + W_MMIN + k2]); }
            if (gTree[nw + W_MMAX + k2] != refTree.w[nw + W_MMAX + k2])
            { ++mm; std::printf("  node %u mMax word %u MM: ref %08x got %08x\n", n, k2,
                                refTree.w[nw + W_MMAX + k2], gTree[nw + W_MMAX + k2]); }
        }
    }
    for (uint32_t n = 0; n < nbNodes; ++n)
    {
        const float r[6] = { refPacked[n].mn.x, refPacked[n].mn.y, refPacked[n].mn.z,
                             refPacked[n].mx.x, refPacked[n].mx.y, refPacked[n].mx.z };
        for (uint32_t e = 0; e < 6u; ++e)
            if (f2u(gPacked[n * 6 + e]) != f2u(r[e]))
            { ++mm; std::printf("  packedNodeBounds[%u][%u] MM: ref %g got %g\n", n, e, r[e], gPacked[n * 6 + e]); }
    }

    // Self-check: node 0's aggregate IS the whole-mesh AABB.
    V3 wholeMin{ 3.4028234663852886e+38f, 3.4028234663852886e+38f, 3.4028234663852886e+38f };
    V3 wholeMax{ -wholeMin.x, -wholeMin.y, -wholeMin.z };
    for (const TetIdx4& t : tets)
        for (uint32_t vi : { t.x, t.y, t.z, t.w })
        {
            wholeMin = vmin(wholeMin, V3{ pos[vi].x, pos[vi].y, pos[vi].z });
            wholeMax = vmax(wholeMax, V3{ pos[vi].x, pos[vi].y, pos[vi].z });
        }
    PXG_CHECK_MSG(f2u(refPacked[0].mn.x) == f2u(wholeMin.x) && f2u(refPacked[0].mx.z) == f2u(wholeMax.z),
                  "reference root bound must equal the whole-mesh AABB (self-check)");
    PXG_CHECK_MSG(f2u(gPacked[0]) != f2u(POISON), "root packedNodeBounds must be overwritten (refit did not run)");

    // Broad-phase slot: the root bound VERBATIM (the broad phase adds contact dist itself).
    const float refBound[6] = { refPacked[0].mn.x, refPacked[0].mn.y, refPacked[0].mn.z,
                                refPacked[0].mx.x, refPacked[0].mx.y, refPacked[0].mx.z };
    for (uint32_t e = 0; e < 6u; ++e)
        if (f2u(gBounds[elemIdx * 6 + e]) != f2u(refBound[e]))
        { ++mm; std::printf("  boundArray[%u][%u] MM: ref %g got %g\n", elemIdx, e, refBound[e],
                            gBounds[elemIdx * 6 + e]); }
    for (uint32_t e = 0; e < 6u; ++e)
    {
        if (f2u(gBounds[0 * 6 + e]) != f2u(POISON)) { ++mm; std::printf("  boundArray[0][%u] clobbered\n", e); }
        if (f2u(gBounds[(elemIdx + 1u) * 6 + e]) != f2u(POISON))
        { ++mm; std::printf("  boundArray[%u][%u] clobbered\n", elemIdx + 1u, e); }
    }

    // The speculative-CCD branch, both ways.
    const float refDist = specCCD + origContactOffset;
    for (uint32_t e = 0; e < uint32_t(bpDist.size()); ++e)
    {
        const float want = (speculativeCCD && e == elemIdx) ? refDist : POISON;
        if (f2u(gBp[e]) != f2u(want))
        { ++mm; std::printf("  bpContactDistance[%u] MM: ref %g got %g\n", e, want, gBp[e]); }
        if (f2u(gNp[e]) != f2u(want))
        { ++mm; std::printf("  npContactDistance[%u] MM: ref %g got %g\n", e, want, gNp[e]); }
    }
    PXG_CHECK_MSG(mm == 0, "sb_refitBoundLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bTet.h); dev->destroyBuffer(bTree.h);
    dev->destroyBuffer(bPacked.h); dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bBp.h);
    dev->destroyBuffer(bNp.h); dev->destroyBuffer(bActive.h); dev->destroyBuffer(bElem.h);
    dev->destroyBuffer(bSpec.h); dev->destroyBuffer(bDesc.h);
}

PXG_TEST(softBodySimRefitBound) { runSbRefitBound(/*speculativeCCD*/ false); }
PXG_TEST(softBodySimRefitBoundCCD) { runSbRefitBound(/*speculativeCCD*/ true); }

} // namespace
