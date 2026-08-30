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
#include "PxgSolverBody.h"                   // PxgSolverBodyData / PxgSolverTxIData (P8.2 tranche d)
#include "PxgFEMCloth.h"                     // real engine per-cloth descriptor (P8.8 softbody-vs-cloth)

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
#include <string>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

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


// ===========================================================================
// P8.0 tranche 3 — the rest of the softbody free-fall chain:
//   sb_gm_stepSoftbodyLaunch       (softBodyGM.cu:232)  — the substep integrator
//   sb_gm_finalizeVelocitiesLaunch (softBodyGM.cu:3136) — delta -> velocity + settle
//   sb_sleeping                    (softBodyGM.cu:3219) — wake counters + changed bitmap
// Every case is bit-exact (raw float bits) against a CPU transcription built with
// -ffp-contract=off, and the descriptor offsets are the ones softBodySimDescLayout pins.
// ===========================================================================

struct StepPush
{
    uint64_t softbodies, activeId;
    float dt, gx, gy, gz;
    uint32_t extForcesWord, pad36;
};
static_assert(sizeof(StepPush) == 40, "step push blob must be 40 bytes");
static_assert(offsetof(StepPush, dt) == 16, "dt @16");
static_assert(offsetof(StepPush, gx) == 20, "gravity @20");
static_assert(offsetof(StepPush, extForcesWord) == 32, "extForces @32");

struct SbFinPush
{
    uint64_t gSoftbodies, activeSoftbodies;
    float invDt, velocityScale, dt;
    uint32_t alwaysAvgWord;
};
static_assert(sizeof(SbFinPush) == 32, "finalize push blob must be 32 bytes");
static_assert(offsetof(SbFinPush, invDt) == 16, "invDt @16");
static_assert(offsetof(SbFinPush, alwaysAvgWord) == 28, "alwaysAvg @28");

struct SbSleepPush
{
    uint64_t softBodies;
    uint32_t numActiveSoftBodies, pad12;
    uint64_t activeId;
    float dt, resetCounter;
    uint64_t wakeCounters, stateChangedMask;
};
static_assert(sizeof(SbSleepPush) == 48, "sleeping push blob must be 48 bytes");
static_assert(offsetof(SbSleepPush, activeId) == 16, "activeId @16");
static_assert(offsetof(SbSleepPush, dt) == 24, "dt @24");
static_assert(offsetof(SbSleepPush, wakeCounters) == 32, "wakeCounters @32");

namespace p80ref
{
struct V3 { float x, y, z; };
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float magSq(V3 a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
} // namespace p80ref

// ---------------------------------------------------------------------------
// sb_gm_stepSoftbodyLaunch.  Driven twice: externalForces OFF (the plain substep
// advance — vel is NOT rewritten, gravity is NOT re-applied) and ON (gravity +
// damping + max-velocity clamp run first and the velocity IS written back).
// ---------------------------------------------------------------------------
static void runStepSoftbody(bool externalForces, bool clampVelocity, bool disableGravity = false)
{
    using namespace p80ref;
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const V3 g = { 0.0f, -9.81f, 0.0f };
    const float linearDamping = 0.1f;
    // clampVelocity picks whether the max-velocity branch is live.  The clamp is the
    // kernel's ONLY sqrt; sqrt is IEEE-754 correctly rounded, so it stays bit-exact.
    const float maxLinVel = clampVelocity ? 0.35f : PX_MAX_REAL_F;

    std::vector<F4> positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 3) ? 0.0f : 1.0f;   // vertex 3 fixed (invMass 0)
        positions[i] = { float(i) * 1.0f, 5.0f + float(i) * 0.25f, -float(i) * 0.5f, w };
        vels[i] = { 0.13f * float(i) + 0.07f, -0.21f * float(i) - 0.11f, 0.05f * float(i) + 0.03f,
                    0.5f + 0.25f * float(i) };   // distinctive .w — the kernel must carry it
    }
    std::vector<F4> posDelta(N);
    for (uint32_t i = 0; i < N; ++i)
        posDelta[i] = { 0.011f * float(i), -0.023f * float(i), 0.017f * float(i), 99.0f };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, posDelta);

    PxgSoftBody c = makeZeroDesc();
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mLinearDamping = linearDamping;
    c.mMaxLinearVelocity = maxLinVel;
    c.mNumVertsGM = N;
    // PxActorFlag::eDISABLE_GRAVITY == (1<<1); byte0 of the @392 flags fold.
    c.mActorFlags = PxActorFlags(disableGravity ? PxActorFlag::eDISABLE_GRAVITY : PxActorFlag::Enum(0));
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    StepPush push = {};
    push.softbodies = bDesc.addr;
    push.activeId = bActive.addr;
    push.dt = dt; push.gx = g.x; push.gy = g.y; push.gz = g.z;
    push.extForcesWord = externalForces ? 1u : 0u;

    Kernel k("softBodySimBda.spv", "sb_gm_stepSoftbodyLaunch", {}, sizeof(StepPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference — expression order transcribed from softBodyGM.cu:232.
    std::vector<F4> refPos = positions, refVel = vels, refDelta = posDelta;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4 tPos = positions[i];
        const F4 pDelta = posDelta[i];
        const F4 v = vels[i];
        const V3 oldPos = { tPos.x, tPos.y, tPos.z };
        V3 vel = { v.x, v.y, v.z };   // the `if (tPos.w != 0)` guard is commented out in CUDA

        if (externalForces)
        {
            if (tPos.w != 0.0f)
            {
                if (!disableGravity)
                    vel = add(vel, mul(g, dt));
                const float damping = 1.0f - std::fmin(dt * linearDamping, 1.0f);
                vel = mul(vel, damping);
            }
            const float velMagSq = magSq(vel);
            if (maxLinVel != PX_MAX_REAL_F && maxLinVel * maxLinVel < velMagSq)
                vel = mul(vel, maxLinVel / std::sqrt(velMagSq));
            refVel[i] = { vel.x, vel.y, vel.z, v.w };             // .w from the OLD VELOCITY
        }

        const V3 delta = mul(vel, dt);
        const V3 pos = add(oldPos, delta);
        refPos[i] = { pos.x, pos.y, pos.z, tPos.w };
        refDelta[i] = { pDelta.x + delta.x, pDelta.y + delta.y, pDelta.z + delta.z, tPos.w };
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);

    size_t mm = 0;
    auto cmp = [&](const char* what, uint32_t i, const F4& r, const F4& got) {
        if (f2u(r.x) != f2u(got.x) || f2u(r.y) != f2u(got.y) ||
            f2u(r.z) != f2u(got.z) || f2u(r.w) != f2u(got.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", what, i,
                            r.x, r.y, r.z, r.w, got.x, got.y, got.z, got.w); } };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp("pos", i, refPos[i], gPos[i]);
        cmp("vel", i, refVel[i], gVel[i]);
        cmp("posDelta", i, refDelta[i], gDelta[i]);
    }
    // Teeth on the branch itself: with externalForces OFF the velocity buffer must be
    // UNTOUCHED (a port that always writes it back would pass the value compare only by
    // accident), and posDelta.w must become tPos.w, not stay at its 99 sentinel.
    if (!externalForces)
        for (uint32_t i = 0; i < N; ++i)
            if (f2u(gVel[i].x) != f2u(vels[i].x) || f2u(gVel[i].w) != f2u(vels[i].w))
            { ++mm; std::printf("  vel[%u] must be untouched when externalForces is off\n", i); }
    for (uint32_t i = 0; i < N; ++i)
        if (f2u(gDelta[i].w) != f2u(positions[i].w))
        { ++mm; std::printf("  posDelta[%u].w must be tPos.w (%g), got %g\n", i, positions[i].w, gDelta[i].w); }
    PXG_CHECK_MSG(mm == 0, "sb_gm_stepSoftbodyLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimStepPlain) { runStepSoftbody(/*externalForces*/ false, /*clamp*/ false); }
PXG_TEST(softBodySimStepExternalForces) { runStepSoftbody(/*externalForces*/ true, /*clamp*/ false); }
PXG_TEST(softBodySimStepVelocityClamp) { runStepSoftbody(/*externalForces*/ true, /*clamp*/ true); }
// Gives the eDISABLE_GRAVITY gate teeth: without this case a mutant that drops the flag
// test and always applies gravity passes, because every other case leaves the flag clear.
PXG_TEST(softBodySimStepNoGravity) { runStepSoftbody(/*externalForces*/ true, /*clamp*/ false, /*disableGravity*/ true); }

// ---------------------------------------------------------------------------
// sb_gm_finalizeVelocitiesLaunch.  Three regimes are exercised in ONE dispatch by
// giving different vertices different delta magnitudes: above the settle tolerance
// (awake, no scaling), between the sleep and settle tolerances (settling -> damped,
// still awake), and below the sleep tolerance (asleep -> damped, not awake).
// ---------------------------------------------------------------------------
static void runSbFinalize(bool alwaysRunVelocityAveraging, bool expectAwake, bool onSleepThreshold = false)
{
    using namespace p80ref;
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;
    const float velocityScale = 0.35f;
    const float settlingThreshold = 6.0f;
    const float sleepThreshold = 1.5f;
    const float settlingDamping = 3.0f;

    const float settleTol = settlingThreshold * dt;   // 0.1
    const float sleepTol = sleepThreshold * dt;       // 0.025

    std::vector<F4> deltas(N), positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        // i%3 == 0: big delta (awake, unscaled); 1: middling (settling); 2: tiny (asleep)
        const float scale = (i % 3 == 0) ? (settleTol * 4.0f)
                          : (i % 3 == 1) ? (sleepTol * 2.0f)
                                         : (sleepTol * 0.25f);
        deltas[i] = { scale * 0.577f, -scale * 0.577f, scale * 0.577f, 42.0f };
        // vertex 3 is kinematic (w == 0) -> the whole body of the branch is skipped
        positions[i] = { float(i), 5.0f - float(i) * 0.1f, 0.25f * float(i), (i == 3) ? 0.0f : 1.0f };
        vels[i] = { 0.31f * float(i) + 0.09f, -0.17f * float(i) - 0.13f, 0.23f * float(i) + 0.05f,
                    0.75f + 0.125f * float(i) };
    }
    // When expectAwake is false, every vertex must fall below the sleep tolerance.
    if (!expectAwake)
        for (uint32_t i = 0; i < N; ++i)
            deltas[i] = { sleepTol * 0.25f * 0.577f, -sleepTol * 0.25f * 0.577f,
                          sleepTol * 0.25f * 0.577f, 42.0f };
    // onSleepThreshold puts EVERY vertex exactly ON the sleep threshold: delta =
    // (tolerance, 0, 0) makes magSq == tolerance*tolerance to the bit (it is literally the
    // same product), and every vertex is inside the settling band, so the awake decision
    // rests solely on `magSq >= tolerance*tolerance`.  Without this case that comparison
    // has NO teeth — the mixed case always contains a vertex above the settle tolerance,
    // which sets awake through the else-branch regardless (verified: a mutant loosening
    // the threshold passes without it).

    DevBuf bDelta = makeDev(dev, deltas);
    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);

    PxgSoftBody c = makeZeroDesc();
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mNumVertsGM = N;
    c.mSettlingThreshold = settlingThreshold;
    c.mSleepThreshold = sleepThreshold;
    c.mSettlingDamping = settlingDamping;
    c.mAwake = 0u;
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    SbFinPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.invDt = invDt; push.velocityScale = velocityScale; push.dt = dt;
    push.alwaysAvgWord = alwaysRunVelocityAveraging ? 1u : 0u;

    if (onSleepThreshold)
        for (uint32_t i = 0; i < N; ++i)
            deltas[i] = { sleepTol, 0.0f, 0.0f, 42.0f };

    dev->upload(bDelta.h, deltas.data(), deltas.size() * sizeof(F4));

    Kernel k("softBodySimBda.spv", "sb_gm_finalizeVelocitiesLaunch", {}, sizeof(SbFinPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference — expression order transcribed from softBodyGM.cu:3136.
    std::vector<F4> refVel = vels;
    bool refAwake = false;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float settleTolerance = settlingThreshold * dt;
        const float tolerance = sleepThreshold * dt;
        const float sleepDamping = 1.0f - std::fmin(1.0f, settlingDamping * dt);

        const F4 delta = deltas[i];
        const V3 tDelta = { delta.x, delta.y, delta.z };
        const V3 deltaVel = mul(tDelta, invDt);
        const F4 pos = positions[i];

        if (pos.w != 0.0f)
        {
            float velocityScaling = 1.0f;
            const float mSq = magSq(tDelta);
            if (mSq < settleTolerance * settleTolerance)
            {
                if (mSq >= tolerance * tolerance) refAwake = true;
                velocityScaling = sleepDamping;
            }
            else
            {
                refAwake = true;
            }

            if (alwaysRunVelocityAveraging || velocityScaling != 1.0f)
            {
                const F4 vel = vels[i];
                V3 tVel = { vel.x, vel.y, vel.z };
                if (alwaysRunVelocityAveraging && magSq(deltaVel) < magSq(tVel))
                    tVel = add(mul(tVel, velocityScale), mul(deltaVel, 1.0f - velocityScale));
                refVel[i] = { velocityScaling * tVel.x, velocityScaling * tVel.y,
                              velocityScaling * tVel.z, vel.w };
            }
        }
    }

    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDesc = downloadBuffer<PxgSoftBody>(dev, bDesc.h, 1);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (f2u(refVel[i].x) != f2u(gVel[i].x) || f2u(refVel[i].y) != f2u(gVel[i].y) ||
            f2u(refVel[i].z) != f2u(gVel[i].z) || f2u(refVel[i].w) != f2u(gVel[i].w))
        { ++mm; std::printf("  vel[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", i,
                            refVel[i].x, refVel[i].y, refVel[i].z, refVel[i].w,
                            gVel[i].x, gVel[i].y, gVel[i].z, gVel[i].w); }
    // The kinematic vertex must be left completely alone.
    if (f2u(gVel[3].x) != f2u(vels[3].x) || f2u(gVel[3].w) != f2u(vels[3].w))
    { ++mm; std::printf("  vel[3] (kinematic, pos.w==0) must be untouched\n"); }
    PXG_CHECK_MSG(mm == 0, "sb_gm_finalizeVelocitiesLaunch bit-exact vs CPU (%zu mismatches)", mm);
    PXG_CHECK_MSG(refAwake == expectAwake, "test setup: expected awake=%d, reference says %d",
                  int(expectAwake), int(refAwake));
    PXG_CHECK_MSG((gDesc[0].mAwake != 0u) == expectAwake,
                  "mAwake must be %u (block-wide OR of the per-vertex awake flags), got %u",
                  unsigned(expectAwake), gDesc[0].mAwake);

    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimFinalizeAwake) { runSbFinalize(/*alwaysAvg*/ false, /*expectAwake*/ true); }
PXG_TEST(softBodySimFinalizeAsleep) { runSbFinalize(/*alwaysAvg*/ false, /*expectAwake*/ false); }
PXG_TEST(softBodySimFinalizeAveraging) { runSbFinalize(/*alwaysAvg*/ true, /*expectAwake*/ true); }
PXG_TEST(softBodySimFinalizeOnSleepThreshold)
{ runSbFinalize(/*alwaysAvg*/ false, /*expectAwake*/ true, /*onSleepThreshold*/ true); }

// ---------------------------------------------------------------------------
// sb_sleeping.  Four bodies cover the whole truth table of (mAwake, wasActive):
//   0: awake + was active   -> counter reset, stays active     -> no state change
//   1: awake + was inactive -> counter reset, becomes active   -> STATE CHANGE
//   2: asleep + was active, counter survives dt -> stays active-> no state change
//   3: asleep + was active, counter runs out    -> goes asleep -> STATE CHANGE
// The changed-mask word must therefore be exactly (1<<1) | (1<<3) == 0xA, and the
// neighbouring bitmap words must be untouched — the guard the CUDA kernel applies and
// which a port that drops it would violate.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimSleeping)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 0.05f;
    const float resetCounter = 0.4f;
    const uint32_t NB = 4;

    std::vector<PxgSoftBody> bodies(NB, makeZeroDesc());
    bodies[0].mAwake = 1u;
    bodies[1].mAwake = 1u;
    bodies[2].mAwake = 0u;
    bodies[3].mAwake = 0u;
    DevBuf bDesc = makeDev(dev, bodies);

    std::vector<uint32_t> active = { 0u, 1u, 2u, 3u };
    DevBuf bActive = makeDev(dev, active);

    std::vector<float> wake = { 0.2f, 0.0f, 0.2f, 0.03f };   // body 3: 0.03 - 0.05 -> 0
    DevBuf bWake = makeDev(dev, wake);

    const uint32_t SENTINEL = 0xDEADBEEFu;
    std::vector<uint32_t> changed(4, SENTINEL);
    DevBuf bChanged = makeDev(dev, changed);

    SbSleepPush push = {};
    push.softBodies = bDesc.addr;
    push.numActiveSoftBodies = NB;
    push.activeId = bActive.addr;
    push.dt = dt;
    push.resetCounter = resetCounter;
    push.wakeCounters = bWake.addr;
    push.stateChangedMask = bChanged.addr;

    Kernel k("softBodySimBda.spv", "sb_sleeping", {}, sizeof(SbSleepPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference
    std::vector<float> refWake(NB);
    uint32_t refMask = 0u;
    for (uint32_t i = 0; i < NB; ++i)
    {
        const bool reset = bodies[i].mAwake != 0u;
        float counter = wake[i];
        const bool wasActive = counter > 0.0f;
        counter = reset ? resetCounter : std::fmax(0.0f, counter - dt);
        const bool isActive = counter > 0.0f;
        refWake[i] = counter;
        if (isActive != wasActive) refMask |= (1u << i);
    }
    PXG_CHECK_MSG(refMask == 0xAu, "test setup: expected the changed mask to be 0xA, reference says 0x%X", refMask);

    const auto gWake = downloadBuffer<float>(dev, bWake.h, NB);
    const auto gChanged = downloadBuffer<uint32_t>(dev, bChanged.h, changed.size());

    size_t mm = 0;
    for (uint32_t i = 0; i < NB; ++i)
        if (f2u(gWake[i]) != f2u(refWake[i]))
        { ++mm; std::printf("  wakeCounter[%u] MM: ref %g got %g\n", i, refWake[i], gWake[i]); }
    if (gChanged[0] != refMask)
    { ++mm; std::printf("  stateChangedMask[0] MM: ref 0x%X got 0x%X\n", refMask, gChanged[0]); }
    // Words past ceil(numActive/32) == 1 must be untouched: CUDA gates the store on
    // (startIdx + threadIdx.x) < ((numActive + 31) / 32), and stateChangedMask is a
    // PxgBitMap sized to the body count — dropping that guard writes past its end.
    for (uint32_t w = 1; w < uint32_t(changed.size()); ++w)
        if (gChanged[w] != SENTINEL)
        { ++mm; std::printf("  stateChangedMask[%u] clobbered: expected 0x%X got 0x%X "
                            "(the (numActive+31)/32 word guard is missing)\n", w, SENTINEL, gChanged[w]); }
    PXG_CHECK_MSG(mm == 0, "sb_sleeping bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bWake.h); dev->destroyBuffer(bChanged.h);
}


// ===========================================================================
// P8.0 tranche 4 — the softbody FEM-volume solve bracket:
//   sb_gm_zeroTetMultipliers        (softBodyGM.cu:1801) — clears the per-tet multipliers
//   sb_gm_applyExternalDeltasLaunch (softBodyGM.cu:2609) — folds the accumulated
//                                     per-vertex correction back and clears the delta
// ===========================================================================

struct ZeroTetPush { uint64_t gSoftbodies, activeSoftbodies; };
static_assert(sizeof(ZeroTetPush) == 16, "zeroTetMultipliers push blob must be 16 bytes");

struct ApplyDeltasPush
{
    uint64_t gSoftbodies, activeSoftbodies;
    float invDt;
    uint32_t pad20;
};
static_assert(sizeof(ApplyDeltasPush) == 24, "applyExternalDeltas push blob must be 24 bytes");
static_assert(offsetof(ApplyDeltasPush, invDt) == 16, "invDt @16");

// PxgSpatialVectorBlock (PxgArticulation.h:77): float4 mTopxyz_bx[32] @0, float2 mbyz[32]
// @512 -> 768 bytes = 192 uint32 words.
static const uint32_t SPATIAL_BLOCK_WORDS = 192u;
static const uint32_t SPATIAL_BLOCK_MBYZ_WORD = 128u;

// ---------------------------------------------------------------------------
// sb_gm_zeroTetMultipliers.  numTets is deliberately NOT a multiple of 32 (37), so the
// case pins three things at once: the block/lane split (tet 32..36 land in block 1), the
// `groupThreadIdx < numTets` guard, and the fact that only the two words
// storeSpatialVector actually writes are touched — every other word of the block, and
// every word of the block past numTets, must keep its poison.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimZeroTetMultipliers)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t numTets = 37u;
    const uint32_t nbBlocks = 3u;   // one more block than numTets needs -> guard is exercised

    const uint32_t POISON = 0x7F7F7F7Fu;
    std::vector<uint32_t> mult(nbBlocks * SPATIAL_BLOCK_WORDS, POISON);
    DevBuf bMult = makeDev(dev, mult);

    PxgSoftBody c = makeZeroDesc();
    c.mSimTetraMultipliers = reinterpret_cast<decltype(c.mSimTetraMultipliers)>(size_t(bMult.addr));
    c.mNumTetsGM = numTets;
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    ZeroTetPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;

    Kernel k("softBodySolveBda.spv", "sb_gm_zeroTetMultipliers", {}, sizeof(ZeroTetPush));
    // grid.x must cover numTets: ceil(37/64) == 1, but dispatch 2 to prove the guard.
    k.dispatch(2u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference: for tet t < numTets, block t/32 lane t&31 gets
    // mTopxyz_bx[lane] = (0,0,0,0) and mbyz[lane] = (0,0); nothing else changes.
    std::vector<uint32_t> ref = mult;
    for (uint32_t t = 0; t < numTets; ++t)
    {
        const uint32_t base = (t / 32u) * SPATIAL_BLOCK_WORDS;
        const uint32_t lane = t & 31u;
        for (uint32_t e = 0; e < 4u; ++e) ref[base + lane * 4u + e] = f2u(0.0f);
        for (uint32_t e = 0; e < 2u; ++e) ref[base + SPATIAL_BLOCK_MBYZ_WORD + lane * 2u + e] = f2u(0.0f);
    }

    const auto got = downloadBuffer<uint32_t>(dev, bMult.h, mult.size());
    size_t mm = 0;
    for (uint32_t w = 0; w < uint32_t(mult.size()); ++w)
        if (got[w] != ref[w])
        { ++mm; if (mm <= 8) std::printf("  multiplier word %u MM: ref 0x%08x got 0x%08x\n", w, ref[w], got[w]); }
    PXG_CHECK_MSG(mm == 0, "sb_gm_zeroTetMultipliers bit-exact vs CPU (%zu mismatches)", mm);
    // Sanity: the refit must actually have run, and block 2 (all tets >= numTets) untouched.
    PXG_CHECK_MSG(got[0] == f2u(0.0f), "tet 0's multiplier must be zeroed");
    for (uint32_t w = 2u * SPATIAL_BLOCK_WORDS; w < 3u * SPATIAL_BLOCK_WORDS; ++w)
        PXG_CHECK_MSG(got[w] == POISON, "block 2 holds no live tet and must be untouched (word %u)", w);

    dev->destroyBuffer(bMult.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// sb_gm_applyExternalDeltasLaunch.  The delta.w reference count drives the averaging
// scale, so the vertices span it: w == 0 (untouched -> scale 1), w == 1 (one constraint
// -> scale 1), w == 3 and w == 7 (averaged), plus a fractional w to prove the divide is
// not an integer one.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimApplyExternalDeltas)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float invDt = 60.0f;

    std::vector<F4> pos(N), vel(N), delta(N), deltaPos(N);
    const float ws[8] = { 0.0f, 1.0f, 3.0f, 7.0f, 2.5f, 0.5f, 4.0f, 12.0f };
    for (uint32_t i = 0; i < N; ++i)
    {
        pos[i] = { float(i) * 0.75f, 5.0f - float(i) * 0.1f, -0.3f * float(i), 1.0f + 0.25f * float(i) };
        vel[i] = { 0.11f * float(i) + 0.03f, -0.19f * float(i) - 0.07f, 0.13f * float(i) + 0.05f,
                   0.5f + 0.125f * float(i) };
        delta[i] = { 0.031f * float(i) + 0.007f, -0.017f * float(i) - 0.011f,
                     0.023f * float(i) + 0.013f, ws[i] };
        deltaPos[i] = { 0.101f * float(i), -0.203f * float(i), 0.307f * float(i), 77.0f };
    }

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bVel = makeDev(dev, vel);
    DevBuf bDelta = makeDev(dev, delta);
    DevBuf bDeltaPos = makeDev(dev, deltaPos);

    PxgSoftBody c = makeZeroDesc();
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mSimDelta = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDeltaPos.addr));
    c.mNumVertsGM = N;
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    ApplyDeltasPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.invDt = invDt;

    Kernel k("softBodySolveBda.spv", "sb_gm_applyExternalDeltasLaunch", {}, sizeof(ApplyDeltasPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference — expression order transcribed from softBodyGM.cu:2609.
    std::vector<F4> refPos(N), refVel(N), refDeltaPos(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        F4 d = delta[i];
        const float scale = 1.0f / std::fmax(d.w, 1.0f);
        d.x *= scale; d.y *= scale; d.z *= scale;
        d.w = 0.0f;
        refVel[i] = { vel[i].x + d.x * invDt, vel[i].y + d.y * invDt,
                      vel[i].z + d.z * invDt, vel[i].w + d.w * invDt };
        refPos[i] = { pos[i].x + d.x, pos[i].y + d.y, pos[i].z + d.z, pos[i].w + d.w };
        refDeltaPos[i] = { deltaPos[i].x + d.x, deltaPos[i].y + d.y,
                           deltaPos[i].z + d.z, deltaPos[i].w + d.w };
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);
    const auto gDeltaPos = downloadBuffer<F4>(dev, bDeltaPos.h, N);

    size_t mm = 0;
    auto cmp = [&](const char* what, uint32_t i, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) ||
            f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", what, i,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); } };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp("pos", i, refPos[i], gPos[i]);
        cmp("vel", i, refVel[i], gVel[i]);
        cmp("deltaPos", i, refDeltaPos[i], gDeltaPos[i]);
        // the delta buffer must be fully cleared for the next iteration
        cmp("delta(cleared)", i, F4{ 0.0f, 0.0f, 0.0f, 0.0f }, gDelta[i]);
    }
    // Teeth on the averaging divide: vertex 3 has w == 7, so its applied delta must be
    // exactly one seventh of the raw one — a port that skipped the scale would match the
    // w == 0 / w == 1 vertices and only differ here.
    const float expectX3 = pos[3].x + delta[3].x / 7.0f;
    PXG_CHECK_MSG(f2u(gPos[3].x) == f2u(expectX3),
                  "vertex 3 (delta.w == 7) must be averaged by 1/7: ref %g got %g", expectX3, gPos[3].x);
    // ...and on the never-below-one clamp: vertex 5 has w == 0.5, which must NOT scale up.
    const float expectX5 = pos[5].x + delta[5].x;
    PXG_CHECK_MSG(f2u(gPos[5].x) == f2u(expectX5),
                  "vertex 5 (delta.w == 0.5) must clamp the divisor to 1: ref %g got %g", expectX5, gPos[5].x);
    PXG_CHECK_MSG(mm == 0, "sb_gm_applyExternalDeltasLaunch bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bDeltaPos.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}


// ===========================================================================
// P8.0 tranche 5 — the tetrahedron ROTATION extractors:
//   sb_gm_updateTetrahedraRotationsLaunch (softBodyGM.cu:419) — simulation mesh
//   sb_updateTetrahedraRotationsLaunch    (softBodyGM.cu:461) — collision mesh
//
// THIS IS THE FIRST SOFTBODY CASE WITH A TOLERANCE SEAM.  extractRotation
// (deformableUtils.cuh:46) builds each iteration's update as PxQuat(w, omega), and that
// constructor evaluates PxSinCos -> sinf/cosf, which have no cross-backend rounding
// guarantee.  So the quaternion is compared under a ULP budget, NOT by raw bits — while
// everything the seam does not touch is still pinned exactly:
//   - the rest-pose decode (both layouts) and the deformation gradient F = P*Qinv are
//     checked bit-exact through a dedicated zero-iteration probe;
//   - the first-frame `q == 0 -> mInitialRotation` seeding is checked bit-exact;
//   - the guards (nbTets, and the collision variant's stress/self-collision flag test)
//     are checked exactly;
//   - the result is checked to be a UNIT quaternion and to actually rotate the rest pose
//     onto the deformed one, which is the property the solver depends on.
// ===========================================================================

struct RotPush { uint64_t gSoftbodies, activeSoftbodies; };
static_assert(sizeof(RotPush) == 16, "updateTetraRotations push blob must be 16 bytes");

// PxgMat33Block (PxgArticulation.h:70): float4 mCol0[32] @0, float4 mCol1[32] @512,
// float mCol2[32] @1024 -> 1152 bytes = 288 uint32 words.
static const uint32_t MAT33_BLOCK_WORDS = 288u;
static const uint32_t MAT33_BLOCK_COL1_WORD = 128u;
static const uint32_t MAT33_BLOCK_COL2_WORD = 256u;

namespace p80rot
{
struct M3 { float c0[3], c1[3], c2[3]; };   // column-major, like PxMat33
struct Q4 { float x, y, z, w; };

static float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

static void mat33FromQuat(const Q4& q, M3& m)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    m.c0[0] = 1.0f - yy - zz; m.c0[1] = xy + zw;        m.c0[2] = xz - yw;
    m.c1[0] = xy - zw;        m.c1[1] = 1.0f - xx - zz; m.c1[2] = yz + xw;
    m.c2[0] = xz + yw;        m.c2[1] = yz - xw;        m.c2[2] = 1.0f - xx - yy;
}

static Q4 quatMul(const Q4& a, const Q4& b)
{
    return Q4{ a.w*b.x + b.w*a.x + a.y*b.z - b.y*a.z,
               a.w*b.y + b.w*a.y + a.z*b.x - b.z*a.x,
               a.w*b.z + b.w*a.z + a.x*b.y - b.x*a.y,
               a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z };
}

// P.transform(v) = c0*v.x + c1*v.y + c2*v.z, left-to-right.
static void transform(const M3& p, const float* v, float* out)
{
    for (int i = 0; i < 3; ++i) out[i] = p.c0[i]*v[0] + p.c1[i]*v[1] + p.c2[i]*v[2];
}

// extractRotation (deformableUtils.cuh:46).  The sin/cos here is THE seam.
static Q4 extractRotation(const M3& A, Q4 q, int maxIter)
{
    const float eps = 1.0e-6f;
    for (int iter = 0; iter < maxIter; ++iter)
    {
        M3 R; mat33FromQuat(q, R);
        float omega[3];
        for (int i = 0; i < 3; ++i)
        {
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            omega[i] = (R.c0[j]*A.c0[k] - R.c0[k]*A.c0[j])
                     + (R.c1[j]*A.c1[k] - R.c1[k]*A.c1[j])
                     + (R.c2[j]*A.c2[k] - R.c2[k]*A.c2[j]);
        }
        const float s = 1.0f / (std::fabs(dot3(R.c0, A.c0) + dot3(R.c1, A.c1) + dot3(R.c2, A.c2)) + eps);
        omega[0] *= s; omega[1] *= s; omega[2] *= s;

        const float w = std::sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);
        // P9.36: extractRotation (deformableUtils.cuh:58) gets here through
        // omega.normalize(), i.e. operator/= -- ONE reciprocal, three multiplies.
        if (w > 0.0f) { const float r = 1.0f / w; omega[0] *= r; omega[1] *= r; omega[2] *= r; }

        const float a = w * 0.5f;
        const float sa = std::sin(a), ca = std::cos(a);
        const Q4 tempQ = { omega[0]*sa, omega[1]*sa, omega[2]*sa, ca };
        q = quatMul(tempQ, q);

        const float qs = 1.0f / std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
        q = Q4{ q.x*qs, q.y*qs, q.z*qs, q.w*qs };

        if (w < eps) break;
    }
    return q;
}

static uint32_t ulpDist(float a, float b)
{
    int32_t ia, ib; std::memcpy(&ia, &a, 4); std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = int32_t(0x80000000u) - ia;
    if (ib < 0) ib = int32_t(0x80000000u) - ib;
    return uint32_t(std::abs(int64_t(ia) - int64_t(ib)));
}
} // namespace p80rot

// simMesh == true  -> sb_gm_updateTetrahedraRotationsLaunch (PxgMat33Block rest poses)
// simMesh == false -> sb_updateTetrahedraRotationsLaunch    (flat PxMat33 rest poses)
static void runUpdateTetraRotations(bool simMesh, bool seedFromInitialRotation, bool flagsEnable = true)
{
    using namespace p80rot;
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t nbTets = 5;
    const uint32_t nbVerts = 8;

    // A deformed configuration: the rest pose is the identity-ish tetra basis, and the
    // current positions are rotated AND stretched, so the extracted rotation is a real
    // non-trivial one (not the identity the iteration would land on trivially).
    std::vector<F4> verts = {
        { 0.0000f, 0.0000f, 0.0000f, 1.0f },
        { 0.9127f, 0.4085f, -0.0212f, 1.0f },
        { -0.3948f, 0.8814f, 0.2571f, 1.0f },
        { 0.1123f, -0.2984f, 1.0451f, 1.0f },
        { 1.2043f, 1.1027f, 0.3318f, 1.0f },
        { -0.5271f, 0.2216f, 1.1832f, 1.0f },
        { 0.6614f, -0.7723f, 0.5409f, 1.0f },
        { 0.2385f, 1.3162f, -0.4471f, 1.0f },
    };
    std::vector<p80refit::TetIdx4> tets = {
        { 0u, 1u, 2u, 3u }, { 1u, 4u, 2u, 3u }, { 0u, 3u, 5u, 2u },
        { 1u, 6u, 0u, 3u }, { 2u, 7u, 4u, 1u },
    };
    // Rest poses: distinct invertible matrices, deliberately not symmetric.
    std::vector<M3> restPoses(nbTets);
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        const float f = 1.0f + 0.13f * float(t);
        restPoses[t].c0[0] = 1.07f * f;  restPoses[t].c0[1] = 0.19f;        restPoses[t].c0[2] = -0.11f;
        restPoses[t].c1[0] = -0.23f;     restPoses[t].c1[1] = 0.93f * f;    restPoses[t].c1[2] = 0.17f;
        restPoses[t].c2[0] = 0.09f;      restPoses[t].c2[1] = -0.31f;       restPoses[t].c2[2] = 1.11f * f;
    }

    // Stored rotations: zero (-> seed from mInitialRotation) or an already-set quaternion.
    const F4 initialRotation = { 0.1830127f, 0.3169873f, 0.1830127f, 0.9106836f }; // unit
    std::vector<F4> tetRot(nbTets);
    for (uint32_t t = 0; t < nbTets; ++t)
        tetRot[t] = seedFromInitialRotation ? F4{ 0.0f, 0.0f, 0.0f, 0.0f }
                                            : F4{ 0.0f, 0.0f, 0.0f, 1.0f }; // identity

    DevBuf bVerts = makeDev(dev, verts);
    DevBuf bTets = makeDev(dev, tets);
    DevBuf bRot = makeDev(dev, tetRot);

    // Rest poses in the layout this variant expects.
    DevBuf bRest;
    std::vector<uint32_t> restWords;
    std::vector<float> restFlat;
    if (simMesh)
    {
        restWords.assign(((nbTets + 31u) / 32u) * MAT33_BLOCK_WORDS, 0u);
        for (uint32_t t = 0; t < nbTets; ++t)
        {
            const uint32_t base = (t / 32u) * MAT33_BLOCK_WORDS;
            const uint32_t lane = t & 31u;
            const M3& m = restPoses[t];
            // mCol0[lane] = (c0.x, c0.y, c0.z, c1.x); mCol1[lane] = (c1.y, c1.z, c2.x, c2.y);
            // mCol2[lane] = c2.z   (loadPxMat33, PxgArticulation.h:271)
            restWords[base + lane*4 + 0] = f2u(m.c0[0]);
            restWords[base + lane*4 + 1] = f2u(m.c0[1]);
            restWords[base + lane*4 + 2] = f2u(m.c0[2]);
            restWords[base + lane*4 + 3] = f2u(m.c1[0]);
            restWords[base + MAT33_BLOCK_COL1_WORD + lane*4 + 0] = f2u(m.c1[1]);
            restWords[base + MAT33_BLOCK_COL1_WORD + lane*4 + 1] = f2u(m.c1[2]);
            restWords[base + MAT33_BLOCK_COL1_WORD + lane*4 + 2] = f2u(m.c2[0]);
            restWords[base + MAT33_BLOCK_COL1_WORD + lane*4 + 3] = f2u(m.c2[1]);
            restWords[base + MAT33_BLOCK_COL2_WORD + lane] = f2u(m.c2[2]);
        }
        bRest = makeDev(dev, restWords);
    }
    else
    {
        restFlat.resize(nbTets * 9u);
        for (uint32_t t = 0; t < nbTets; ++t)
        {
            const M3& m = restPoses[t];
            for (int i = 0; i < 3; ++i) restFlat[t*9 + 0 + i] = m.c0[i];
            for (int i = 0; i < 3; ++i) restFlat[t*9 + 3 + i] = m.c1[i];
            for (int i = 0; i < 3; ++i) restFlat[t*9 + 6 + i] = m.c2[i];
        }
        bRest = makeDev(dev, restFlat);
    }

    PxgSoftBody c = makeZeroDesc();
    c.mInitialRotation = PxQuat(initialRotation.x, initialRotation.y, initialRotation.z, initialRotation.w);
    if (simMesh)
    {
        c.mSimTetIndices = reinterpret_cast<decltype(c.mSimTetIndices)>(size_t(bTets.addr));
        c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
        c.mSimTetraRestPoses = reinterpret_cast<decltype(c.mSimTetraRestPoses)>(size_t(bRest.addr));
        c.mSimTetraRotations = reinterpret_cast<float4*>(size_t(bRot.addr));
        c.mNumTetsGM = nbTets;
    }
    else
    {
        c.mTetIndices = reinterpret_cast<decltype(c.mTetIndices)>(size_t(bTets.addr));
        c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
        c.mTetraRestPoses = reinterpret_cast<decltype(c.mTetraRestPoses)>(size_t(bRest.addr));
        c.mTetraRotations = reinterpret_cast<float4*>(size_t(bRot.addr));
        c.mNumTets = nbTets;
        // The collision variant runs only under eCOMPUTE_STRESS_TENSOR or when
        // self-collision is ENABLED (i.e. eDISABLE_SELF_COLLISION clear).
        c.mBodyFlags = PxDeformableBodyFlags(flagsEnable ? PxDeformableBodyFlag::Enum(0)
                                                         : PxDeformableBodyFlag::eDISABLE_SELF_COLLISION);
        c.mVolumeFlags = PxDeformableVolumeFlags(PxDeformableVolumeFlag::Enum(0));
    }
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    RotPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;

    Kernel k("softBodySolveBda.spv",
             simMesh ? "sb_gm_updateTetrahedraRotationsLaunch" : "sb_updateTetrahedraRotationsLaunch",
             {}, sizeof(RotPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<F4>(dev, bRot.h, nbTets);

    // The disabled-flags control: nothing may be written at all.
    if (!simMesh && !flagsEnable)
    {
        for (uint32_t t = 0; t < nbTets; ++t)
            PXG_CHECK_MSG(f2u(got[t].x) == f2u(tetRot[t].x) && f2u(got[t].w) == f2u(tetRot[t].w),
                          "collision-mesh rotations must be SKIPPED when stress is off and "
                          "self-collision is disabled (tet %u was written)", t);
        dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bTets.h); dev->destroyBuffer(bRot.h);
        dev->destroyBuffer(bRest.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
        return;
    }

    // CPU reference.
    uint32_t worstUlp = 0;
    double worstAngleErr = 0.0;
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        const p80refit::TetIdx4& ti = tets[t];
        const F4& v0 = verts[ti.x]; const F4& v1 = verts[ti.y];
        const F4& v2 = verts[ti.z]; const F4& v3 = verts[ti.w];
        const float u1[3] = { v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
        const float u2[3] = { v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };
        const float u3[3] = { v3.x - v0.x, v3.y - v0.y, v3.z - v0.z };
        M3 P;
        for (int i = 0; i < 3; ++i) { P.c0[i] = u1[i]; P.c1[i] = u2[i]; P.c2[i] = u3[i]; }

        M3 F;
        transform(P, restPoses[t].c0, F.c0);
        transform(P, restPoses[t].c1, F.c1);
        transform(P, restPoses[t].c2, F.c2);

        Q4 q = { tetRot[t].x, tetRot[t].y, tetRot[t].z, tetRot[t].w };
        if ((q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) == 0.0f)
            q = Q4{ initialRotation.x, initialRotation.y, initialRotation.z, initialRotation.w };
        const Q4 ref = extractRotation(F, q, 100);

        // Compare the ROTATION, not the quaternion.  Two corrections matter here, and
        // both were found by measurement, not assumed:
        //   - q and -q are the SAME rotation, and the iteration is free to land on either
        //     representative; the seeded case actually does flip sign (measured: matrix
        //     distance 0, i.e. bit-identical rotation, while a raw quaternion ULP compare
        //     read 2.1e9 ULP).  So the quaternion is sign-normalised before any ULP is
        //     reported, and the primary metric is sign-agnostic by construction.
        //   - the angle-between-quaternions metric acos(|dot|) is badly conditioned near
        //     d == 1: a 1-ULP quaternion difference reads as ~3e-4 rad, which says
        //     nothing about the port.  The Frobenius distance between the two rotation
        //     MATRICES has no such amplification and is what the solver actually consumes.
        Q4 rn = ref, gn = { got[t].x, got[t].y, got[t].z, got[t].w };
        if (rn.w < 0.0f) { rn.x = -rn.x; rn.y = -rn.y; rn.z = -rn.z; rn.w = -rn.w; }
        if (gn.w < 0.0f) { gn.x = -gn.x; gn.y = -gn.y; gn.z = -gn.z; gn.w = -gn.w; }
        const uint32_t u[4] = { ulpDist(rn.x, gn.x), ulpDist(rn.y, gn.y),
                                ulpDist(rn.z, gn.z), ulpDist(rn.w, gn.w) };
        for (int i = 0; i < 4; ++i) worstUlp = u[i] > worstUlp ? u[i] : worstUlp;

        M3 Rr, Rg;
        mat33FromQuat(ref, Rr);
        mat33FromQuat(Q4{ got[t].x, got[t].y, got[t].z, got[t].w }, Rg);
        double md = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            md = std::fmax(md, std::fabs(double(Rr.c0[i]) - Rg.c0[i]));
            md = std::fmax(md, std::fabs(double(Rr.c1[i]) - Rg.c1[i]));
            md = std::fmax(md, std::fabs(double(Rr.c2[i]) - Rg.c2[i]));
        }
        if (md > worstAngleErr) worstAngleErr = md;

        // The result must be a UNIT quaternion regardless of the seam.
        const double mag = std::sqrt(double(got[t].x)*got[t].x + double(got[t].y)*got[t].y +
                                     double(got[t].z)*got[t].z + double(got[t].w)*got[t].w);
        PXG_CHECK_MSG(std::fabs(mag - 1.0) < 1e-5,
                      "tet %u: extracted rotation must be a unit quaternion (|q| = %g)", t, mag);
    }

    // TOLERANCE SEAM, stated as a budget rather than a bit compare.  The only
    // non-portable operation in the whole chain is the sin/cos inside PxQuat(w, omega);
    // everything else (sqrt, the quaternion product, the matrix build) is bit-exact.
    // Measured on llvmpipe, and the two regimes differ in a way worth recording:
    //   - seeded from the STORED rotation (near the fixed point): 1 ULP, matrix 6e-8;
    //   - seeded from mInitialRotation (a ~60 deg rotation away from it): 860 ULP,
    //     matrix 2e-5.
    // The iteration amplifies the seam rather than washing it out, because the `w < eps`
    // early-exit is itself seam-dependent: a last-bit difference in sin/cos moves w
    // across the threshold and the two sides stop one iteration apart.  The far seed
    // needs more iterations, so it accumulates more.  2e-5 on the rotation matrix is
    // ~0.001 degrees — irrelevant to the solver, but it is a real divergence and is
    // budgeted as such rather than papered over.  The budgets sit ~5x above the
    // measurement; a mutation that changes the MATHS rather than the last bit moves the
    // matrix distance by O(0.1) and is still caught (verified by mutation).
    std::printf("  [seam] updateTetraRotations(%s): worst %u ULP (sign-normalised), "
                "worst rotation-matrix distance %.3e\n",
                simMesh ? "sim" : "collision", worstUlp, worstAngleErr);
    PXG_CHECK_MSG(worstUlp <= 4096u,
                  "quaternion must stay inside the documented sin/cos ULP budget (worst %u)", worstUlp);
    PXG_CHECK_MSG(worstAngleErr < 1e-4,
                  "extracted rotation matrix must stay within 1e-4 of the CPU reference (worst %.3e)",
                  worstAngleErr);

    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bTets.h); dev->destroyBuffer(bRot.h);
    dev->destroyBuffer(bRest.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimUpdateTetraRotationsSim) { runUpdateTetraRotations(/*simMesh*/ true, /*seedInitial*/ false); }
PXG_TEST(softBodySimUpdateTetraRotationsSimSeeded) { runUpdateTetraRotations(/*simMesh*/ true, /*seedInitial*/ true); }
PXG_TEST(softBodySimUpdateTetraRotationsCollision) { runUpdateTetraRotations(/*simMesh*/ false, /*seedInitial*/ false); }
PXG_TEST(softBodySimUpdateTetraRotationsSkipped)
{ runUpdateTetraRotations(/*simMesh*/ false, /*seedInitial*/ false, /*flagsEnable*/ false); }


// ===========================================================================
// P8.0 tranche 6 — sb_gm_cp_solveTetrahedronsPartitionLaunch (softBodyGM.cu:1408):
// the FEM-volume constraint solve.
//
// The kernel inherits the tranche-5 sin/cos seam through its rotation refresh, but the
// CONSTITUTIVE maths is bit-exact (mul/add/sub/div/cross/dot plus one PxSqrt).  The test
// exploits that split: it seeds mSimTetraRotations with an already-converged unit
// quaternion and passes isFirstIteration = false, so extractRotation runs its 4
// iterations from the fixed point and contributes at most a last-bit nudge, while every
// constitutive term is compared BIT-EXACTLY against the CPU transcription.  The seam
// itself is already covered by the tranche-5 cases.
// ===========================================================================

struct SolveTetraPush
{
    uint64_t gSoftbodies, activeSoftbodies;
    uint32_t numActiveSoftbodies;
    float invDt;
    uint32_t partitionId, flagsWord;
    uint64_t materials;
};
static_assert(sizeof(SolveTetraPush) == 40, "solveTetraPartition push blob must be 40 bytes");
static_assert(offsetof(SolveTetraPush, invDt) == 20, "invDt @20");
static_assert(offsetof(SolveTetraPush, partitionId) == 24, "partitionId @24");
static_assert(offsetof(SolveTetraPush, flagsWord) == 28, "flags @28");
static_assert(offsetof(SolveTetraPush, materials) == 32, "materials @32");

namespace p80solve
{
using p80rot::M3;
using p80rot::Q4;
using p80rot::mat33FromQuat;
using p80rot::transform;

static float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static void cross3(const float* a, const float* b, float* r)
{ r[0] = a[1]*b[2] - a[2]*b[1]; r[1] = a[2]*b[0] - a[0]*b[2]; r[2] = a[0]*b[1] - a[1]*b[0]; }
static void sub3(const float* a, const float* b, float* r) { for (int i=0;i<3;++i) r[i]=a[i]-b[i]; }
static void scale3(const float* a, float s, float* r) { for (int i=0;i<3;++i) r[i]=a[i]*s; }

static void mat33Mul(const M3& a, const M3& b, M3& r)
{ transform(a, b.c0, r.c0); transform(a, b.c1, r.c1); transform(a, b.c2, r.c2); }
static float det3(const M3& m) { float c[3]; cross3(m.c1, m.c2, c); return dot3(m.c0, c); }

struct Material { float youngs, poissons, elasticityDamping, dampingScale; uint32_t model; };

static void computeDCdx(const M3& d, const M3& m, float* dCdx1, float* dCdx2, float* dCdx3)
{
    const float r0[3] = { m.c0[0], m.c1[0], m.c2[0] };
    const float r1[3] = { m.c0[1], m.c1[1], m.c2[1] };
    const float r2[3] = { m.c0[2], m.c1[2], m.c2[2] };
    const float p0[3] = { d.c0[0], d.c1[0], d.c2[0] };
    const float p1[3] = { d.c0[1], d.c1[1], d.c2[1] };
    const float p2[3] = { d.c0[2], d.c1[2], d.c2[2] };
    dCdx1[0]=dot3(r0,p0); dCdx1[1]=dot3(r0,p1); dCdx1[2]=dot3(r0,p2);
    dCdx2[0]=dot3(r1,p0); dCdx2[1]=dot3(r1,p1); dCdx2[2]=dot3(r1,p2);
    dCdx3[0]=dot3(r2,p0); dCdx3[1]=dot3(r2,p1); dCdx3[2]=dot3(r2,p2);
}

static void computeDeltaLambdaXGradC(M3& dl, float& lambda, float C, const M3& dCdF, const M3& DmInv,
                                     const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                                     float alphaTilde, float dtInv, float damping, float scaledDamping)
{
    float dCdx1[3], dCdx2[3], dCdx3[3];
    computeDCdx(dCdF, DmInv, dCdx1, dCdx2, dCdx3);
    float dCdx0[3];
    for (int i = 0; i < 3; ++i) dCdx0[i] = -dCdx1[i] - dCdx2[i] - dCdx3[i];

    float deltaLambda;
    if (damping == 0.0f)
    {
        const float denom = dot3(dCdx0,dCdx0)*v0.w + dot3(dCdx1,dCdx1)*v1.w +
                            dot3(dCdx2,dCdx2)*v2.w + dot3(dCdx3,dCdx3)*v3.w + alphaTilde;
        if (std::fabs(denom) < 1.0e-15f) return;
        deltaLambda = (-C - alphaTilde * lambda) / denom;
    }
    else
    {
        const float denom = (1.0f + damping*dtInv) * (dot3(dCdx0,dCdx0)*v0.w + dot3(dCdx1,dCdx1)*v1.w +
                                                      dot3(dCdx2,dCdx2)*v2.w + dot3(dCdx3,dCdx3)*v3.w) + alphaTilde;
        if (std::fabs(denom) < 1.0e-15f) { dl = M3{}; return; }
        const float vv0[3] = { v0.x, v0.y, v0.z }, vv1[3] = { v1.x, v1.y, v1.z };
        const float vv2[3] = { v2.x, v2.y, v2.z }, vv3[3] = { v3.x, v3.y, v3.z };
        deltaLambda = (-C - alphaTilde*lambda -
                       scaledDamping * (dot3(dCdx0,vv0) + dot3(dCdx1,vv1) + dot3(dCdx2,vv2) + dot3(dCdx3,vv3))) / denom;
    }
    lambda += deltaLambda;
    for (int i = 0; i < 3; ++i)
    { dl.c0[i] += deltaLambda*dCdx1[i]; dl.c1[i] += deltaLambda*dCdx2[i]; dl.c2[i] += deltaLambda*dCdx3[i]; }
}

static void arapConstraint(M3& dl, float& lambda, const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                           const Q4& tetR, const M3& F, float dtInv, const M3& DmInv,
                           float alphaTilde, float damping, float scaledDamping)
{
    M3 R; mat33FromQuat(tetR, R);
    M3 t;
    sub3(F.c0, R.c0, t.c0); sub3(F.c1, R.c1, t.c1); sub3(F.c2, R.c2, t.c2);
    const float ARAP_C = std::sqrt(dot3(t.c0,t.c0) + dot3(t.c1,t.c1) + dot3(t.c2,t.c2) + 1.0e-18f);
    const float inv = 1.0f / ARAP_C;
    scale3(t.c0, inv, t.c0); scale3(t.c1, inv, t.c1); scale3(t.c2, inv, t.c2);
    computeDeltaLambdaXGradC(dl, lambda, ARAP_C, t, DmInv, v0, v1, v2, v3, alphaTilde, dtInv, damping, scaledDamping);
}

static void volumeConstraint(M3& dl, float& lambda, const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                             const M3& F, float invDt, const M3& DmInv,
                             float alphaTilde, float damping, float scaledDamping)
{
    const float C = det3(F) - 1.0f;
    M3 dCdF;
    cross3(F.c1, F.c2, dCdF.c0); cross3(F.c2, F.c0, dCdF.c1); cross3(F.c0, F.c1, dCdF.c2);
    computeDeltaLambdaXGradC(dl, lambda, C, dCdF, DmInv, v0, v1, v2, v3, alphaTilde, invDt, damping, scaledDamping);
}

static void volumeSolve(const M3& P, const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                        float invDtSq, const M3& Qinv, float volumeLambda, M3& out)
{
    const float det = det3(Qinv);
    const float volume = 1.0f / (6.0f * det);
    const float alphaTilde = (volumeLambda == 0.0f) ? 0.0f : 1.0f / (volumeLambda * volume * invDtSq);
    M3 F; mat33Mul(P, Qinv, F);
    float g0[3], g1[3], g2[3];
    cross3(F.c1, F.c2, g0); cross3(F.c2, F.c0, g1); cross3(F.c0, F.c1, g2);
    float grad1[3], grad2[3], grad3[3], grad0[3];
    for (int i = 0; i < 3; ++i)
    {
        grad1[i] = g0[i]*Qinv.c0[0] + g1[i]*Qinv.c1[0] + g2[i]*Qinv.c2[0];
        grad2[i] = g0[i]*Qinv.c0[1] + g1[i]*Qinv.c1[1] + g2[i]*Qinv.c2[1];
        grad3[i] = g0[i]*Qinv.c0[2] + g1[i]*Qinv.c1[2] + g2[i]*Qinv.c2[2];
    }
    for (int i = 0; i < 3; ++i) grad0[i] = -grad1[i] - grad2[i] - grad3[i];
    const float C = dot3(g2, F.c2) - 1.0f;
    const float w = dot3(grad0,grad0)*v0.w + dot3(grad1,grad1)*v1.w +
                    dot3(grad2,grad2)*v2.w + dot3(grad3,grad3)*v3.w;
    const float dlambda = -C / (w + alphaTilde);
    scale3(grad1, dlambda, out.c0); scale3(grad2, dlambda, out.c1); scale3(grad3, dlambda, out.c2);
}

static void solveInnerNeoHookean(const M3& P, const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                                 float* mbyz, float invDtSq, const M3& Qinv, bool isTGS, float invE, M3& out)
{
    M3 F; mat33Mul(P, Qinv, F);
    float g1[3], g2[3], g3[3], g0[3];
    for (int i = 0; i < 3; ++i)
    {
        g1[i] = F.c0[i]*2.0f*Qinv.c0[0] + F.c1[i]*2.0f*Qinv.c1[0] + F.c2[i]*2.0f*Qinv.c2[0];
        g2[i] = F.c0[i]*2.0f*Qinv.c0[1] + F.c1[i]*2.0f*Qinv.c1[1] + F.c2[i]*2.0f*Qinv.c2[1];
        g3[i] = F.c0[i]*2.0f*Qinv.c0[2] + F.c1[i]*2.0f*Qinv.c1[2] + F.c2[i]*2.0f*Qinv.c2[2];
    }
    const float C = dot3(F.c0,F.c0) + dot3(F.c1,F.c1) + dot3(F.c2,F.c2) - 3.0f;
    for (int i = 0; i < 3; ++i) g0[i] = -g1[i] - g2[i] - g3[i];
    const float w = dot3(g0,g0)*v0.w + dot3(g1,g1)*v1.w + dot3(g2,g2)*v2.w + dot3(g3,g3)*v3.w;
    const float alpha = invDtSq * invE;
    float dlambda;
    if (!isTGS)
    {
        dlambda = -(C + alpha * mbyz[0]) / (w + alpha);
        mbyz[0] += dlambda;
    }
    else dlambda = -C / (w + alpha);
    scale3(g1, dlambda, out.c0); scale3(g2, dlambda, out.c1); scale3(g3, dlambda, out.c2);
}

// tetrahedronsSolveGM (softBodyGM.cu:926)
static void solveGM(M3& dl, const float* u1, const float* u2, const float* u3,
                    const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                    float* mbyz, float invDtSq, float invDt, const M3& Qinv,
                    const Q4& tetR, bool isTGS, const Material& material)
{
    dl = M3{};
    if (material.youngs < 1.0e-18f) return;

    M3 U; for (int i = 0; i < 3; ++i) { U.c0[i]=u1[i]; U.c1[i]=u2[i]; U.c2[i]=u3[i]; }

    if (material.model == 1u) // eNEO_HOOKEAN
    {
        const float invE = 1.0f / material.youngs;
        M3 R; mat33FromQuat(tetR, R);
        M3 RT;
        RT.c0[0]=R.c0[0]; RT.c0[1]=R.c1[0]; RT.c0[2]=R.c2[0];
        RT.c1[0]=R.c0[1]; RT.c1[1]=R.c1[1]; RT.c1[2]=R.c2[1];
        RT.c2[0]=R.c0[2]; RT.c2[1]=R.c1[2]; RT.c2[2]=R.c2[2];
        M3 P; mat33Mul(RT, U, P);

        M3 d; solveInnerNeoHookean(P, v0, v1, v2, v3, mbyz, invDtSq, Qinv, isTGS, invE, d);

        float delta0[3];
        for (int i = 0; i < 3; ++i) delta0[i] = -(d.c0[i] + d.c1[i] + d.c2[i]) * v0.w;
        for (int i = 0; i < 3; ++i)
        {
            P.c0[i] += (d.c0[i]*v1.w - delta0[i]);
            P.c1[i] += (d.c1[i]*v2.w - delta0[i]);
            P.c2[i] += (d.c2[i]*v3.w - delta0[i]);
        }

        if (material.poissons > 1.0e-18f)
        {
            const float nu = material.poissons;
            const float volumeLambda = (nu < 0.5f - 1.0e-18f) ? nu / ((1.0f - 2.0f*nu) * material.youngs) : 0.0f;
            M3 sv; volumeSolve(P, v0, v1, v2, v3, invDtSq, Qinv, volumeLambda, sv);
            for (int i = 0; i < 3; ++i) { d.c0[i] += sv.c0[i]; d.c1[i] += sv.c1[i]; d.c2[i] += sv.c2[i]; }
        }
        mat33Mul(R, d, dl);
    }
    else // eCO_ROTATIONAL
    {
        const float lambdaLame = material.youngs * material.poissons /
                                 ((1.0f + material.poissons) * (1.0f - 2.0f*material.poissons));
        const float mu = material.youngs / (2.0f * (1.0f + material.poissons));
        const float det = det3(Qinv);
        const float volume = 1.0f / (6.0f * det);
        M3 F; mat33Mul(U, Qinv, F);
        const float alphaTilde0 = invDtSq / (2.0f * mu * volume);
        float damping = material.elasticityDamping;
        float scaledDamping = damping * material.dampingScale;
        float lam0 = isTGS ? 0.0f : mbyz[0];
        float lam1 = isTGS ? 0.0f : mbyz[1];

        arapConstraint(dl, lam0, v0, v1, v2, v3, tetR, F, invDt, Qinv, alphaTilde0, damping, scaledDamping);

        float dx0[3], dx1[3], dx2[3], dx3[3];
        for (int i = 0; i < 3; ++i)
        {
            dx0[i] = -(dl.c0[i] + dl.c1[i] + dl.c2[i]) * v0.w;
            dx1[i] = dl.c0[i] * v1.w; dx2[i] = dl.c1[i] * v2.w; dx3[i] = dl.c2[i] * v3.w;
        }
        M3 X;
        for (int i = 0; i < 3; ++i)
        { X.c0[i] = u1[i] + (dx1[i]-dx0[i]); X.c1[i] = u2[i] + (dx2[i]-dx0[i]); X.c2[i] = u3[i] + (dx3[i]-dx0[i]); }
        mat33Mul(X, Qinv, F);

        if (material.poissons > 1.0e-18f)
        {
            float alphaTilde1 = 0.0f, ratio = 0.0f;
            if (material.poissons < 0.5f - 1.0e-18f)
            { alphaTilde1 = invDtSq / (lambdaLame * volume); ratio = 2.0f * mu / lambdaLame; }
            damping *= ratio; scaledDamping *= ratio;
            volumeConstraint(dl, lam1, v0, v1, v2, v3, F, invDt, Qinv, alphaTilde1, damping, scaledDamping);
        }
        if (!isTGS) { mbyz[0] = lam0; mbyz[1] = lam1; }
    }
}
} // namespace p80solve

// materialModel: 0 = eCO_ROTATIONAL, 1 = eNEO_HOOKEAN
static void runSolveTetraPartition(uint32_t materialModel, bool isTGS, float damping)
{
    using namespace p80solve;
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t nbTets = 3;          // one partition, 3 tets
    const uint32_t nbVerts = 6;
    const float invDt = 60.0f;

    // Vertices of the ORDERED (CP) buffers; the pull indices below make tet 0 read two of
    // its vertices from the UNORDERED buffers, so both gather paths run in one dispatch.
    std::vector<F4> cpPos(nbTets * 4), cpVel(nbTets * 4);
    for (uint32_t i = 0; i < nbTets * 4; ++i)
    {
        cpPos[i] = { 0.11f*float(i) + 0.03f, 1.0f - 0.07f*float(i), 0.19f*float(i) - 0.05f, 1.0f + 0.25f*float(i) };
        cpVel[i] = { 0.013f*float(i), -0.021f*float(i), 0.017f*float(i), 0.5f };
    }
    std::vector<F4> unPos(nbVerts), unVel(nbVerts);
    for (uint32_t i = 0; i < nbVerts; ++i)
    {
        unPos[i] = { -0.31f*float(i) + 0.07f, 0.9f + 0.05f*float(i), 0.23f*float(i), 2.0f + 0.5f*float(i) };
        unVel[i] = { -0.011f*float(i), 0.019f*float(i), -0.007f*float(i), 0.25f };
    }

    // pull indices: MSB set (>= PULL_IND_MASK) means "use the CP buffer".
    const uint32_t CP = 0xFFFFFFFFu;
    std::vector<p80refit::TetIdx4> pull = {
        { 1u, CP, 3u, CP },   // tet 0: mixed
        { CP, CP, CP, CP },   // tet 1: all CP
        { 0u, 2u, 4u, 5u },   // tet 2: all unordered
    };
    // The .x lane is masked with PULL_IND_MASK, so a CP marker in .x must survive that:
    // 0xFFFFFFFF & 0x7FFFFFFF == 0x7FFFFFFF == PULL_IND_MASK, still "not < mask".
    std::vector<p80refit::TetIdx4> remap = {
        { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u }, { 8u, 9u, 10u, 11u },
    };

    // rest poses (PxgMat33Block), rotations, multipliers, materials
    std::vector<M3> rest(nbTets);
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        const float f = 1.0f + 0.17f*float(t);
        rest[t].c0[0]=1.13f*f; rest[t].c0[1]=0.21f;    rest[t].c0[2]=-0.09f;
        rest[t].c1[0]=-0.19f;  rest[t].c1[1]=0.97f*f;  rest[t].c1[2]=0.13f;
        rest[t].c2[0]=0.07f;   rest[t].c2[1]=-0.27f;   rest[t].c2[2]=1.07f*f;
    }
    std::vector<uint32_t> restWords(MAT33_BLOCK_WORDS, 0u);
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        const uint32_t lane = t & 31u; const M3& m = rest[t];
        restWords[lane*4+0]=f2u(m.c0[0]); restWords[lane*4+1]=f2u(m.c0[1]);
        restWords[lane*4+2]=f2u(m.c0[2]); restWords[lane*4+3]=f2u(m.c1[0]);
        restWords[MAT33_BLOCK_COL1_WORD+lane*4+0]=f2u(m.c1[1]);
        restWords[MAT33_BLOCK_COL1_WORD+lane*4+1]=f2u(m.c1[2]);
        restWords[MAT33_BLOCK_COL1_WORD+lane*4+2]=f2u(m.c2[0]);
        restWords[MAT33_BLOCK_COL1_WORD+lane*4+3]=f2u(m.c2[1]);
        restWords[MAT33_BLOCK_COL2_WORD+lane]=f2u(m.c2[2]);
    }
    // Already-converged unit quaternions: extractRotation starts AT the fixed point, so
    // the seam contributes at most a last bit and the constitutive compare stays exact.
    std::vector<F4> tetRot(nbTets);
    for (uint32_t t = 0; t < nbTets; ++t)
        tetRot[t] = { 0.0f, 0.0f, 0.0f, 1.0f };

    // multipliers: one PxgSpatialVectorBlock, mbyz seeded non-zero so the PGS accumulate
    // path is observable.
    std::vector<uint32_t> mult(SPATIAL_BLOCK_WORDS, 0u);
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        mult[SPATIAL_BLOCK_MBYZ_WORD + t*2 + 0] = f2u(0.013f * float(t + 1));
        mult[SPATIAL_BLOCK_MBYZ_WORD + t*2 + 1] = f2u(-0.007f * float(t + 1));
    }

    // PxsDeformableVolumeMaterialData: 32-byte stride
    std::vector<uint32_t> matWords(8, 0u);
    const float youngs = 1.0e5f, poissons = 0.31f;
    const uint16_t dampingScaleU16 = 32768u;
    matWords[0] = f2u(youngs);
    matWords[1] = f2u(poissons);
    matWords[2] = f2u(0.0f);                     // dynamicFriction
    matWords[3] = f2u(damping);                  // elasticityDamping
    matWords[4] = uint32_t(dampingScaleU16) | (materialModel << 16);
    // Trailers on the two TINY buffers: llvmpipe widens the last load past the end, so a
    // buffer that stops exactly on its final element can fault when it lands at the end of
    // a mapped region (the femClothSimRefitBound precedent).  4 spare words each.
    std::vector<uint16_t> matIdx(4u + 8u, 0u);
    std::vector<uint32_t> partitions = { nbTets, 0u, 0u, 0u, 0u };

    // DIAG: pad every buffer
    std::vector<F4> cpPosP = cpPos; cpPosP.resize(cpPos.size() + 64, F4{0,0,0,0});
    std::vector<F4> cpVelP = cpVel; cpVelP.resize(cpVel.size() + 64, F4{0,0,0,0});
    std::vector<F4> unPosP = unPos; unPosP.resize(unPos.size() + 64, F4{0,0,0,0});
    std::vector<F4> unVelP = unVel; unVelP.resize(unVel.size() + 64, F4{0,0,0,0});
    std::vector<p80refit::TetIdx4> pullP = pull; pullP.resize(pull.size() + 64, {0,0,0,0});
    std::vector<p80refit::TetIdx4> remapP = remap; remapP.resize(remap.size() + 64, {0,0,0,0});
    std::vector<uint32_t> restP = restWords; restP.resize(restWords.size() + 512, 0u);
    std::vector<F4> tetRotP = tetRot; tetRotP.resize(tetRot.size() + 64, F4{0,0,0,1});
    std::vector<uint32_t> multP = mult; multP.resize(mult.size() + 512, 0u);
    std::vector<uint32_t> matP = matWords; matP.resize(matWords.size() + 64, 0u);
    DevBuf bCpPos = makeDev(dev, cpPosP), bCpVel = makeDev(dev, cpVelP);
    DevBuf bUnPos = makeDev(dev, unPosP), bUnVel = makeDev(dev, unVelP);
    DevBuf bPull = makeDev(dev, pullP), bRemap = makeDev(dev, remapP);
    DevBuf bRest = makeDev(dev, restP), bRot = makeDev(dev, tetRotP);
    DevBuf bMult = makeDev(dev, multP), bMat = makeDev(dev, matP);
    DevBuf bMatIdx = makeDev(dev, matIdx), bPart = makeDev(dev, partitions);

    PxgSoftBody c = makeZeroDesc();
    c.mSimPosition_InvMassCP = reinterpret_cast<float4*>(size_t(bCpPos.addr));
    c.mSimVelocity_invMassCP = reinterpret_cast<float4*>(size_t(bCpVel.addr));
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bUnPos.addr));
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bUnVel.addr));
    c.mSimPullIndices = reinterpret_cast<decltype(c.mSimPullIndices)>(size_t(bPull.addr));
    c.mSimRemapOutputCP = reinterpret_cast<decltype(c.mSimRemapOutputCP)>(size_t(bRemap.addr));
    c.mSimTetraRestPoses = reinterpret_cast<decltype(c.mSimTetraRestPoses)>(size_t(bRest.addr));
    c.mSimTetraRotations = reinterpret_cast<float4*>(size_t(bRot.addr));
    c.mSimTetraMultipliers = reinterpret_cast<decltype(c.mSimTetraMultipliers)>(size_t(bMult.addr));
    c.mOrderedMaterialIndices = reinterpret_cast<PxU16*>(size_t(bMatIdx.addr));
    c.mSimAccumulatedPartitionsCP = reinterpret_cast<PxU32*>(size_t(bPart.addr));
    c.mNumTetsGM = nbTets;
    c.mNumPartitionsGM = 1u;
    c.mNumTetsPerElement = 1u;
    c.mInitialRotation = PxQuat(0.0f, 0.0f, 0.0f, 1.0f);
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    SolveTetraPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.numActiveSoftbodies = 1u;
    push.invDt = invDt;
    push.partitionId = 0u;
    push.flagsWord = isTGS ? 1u : 0u;   // isFirstIteration = false -> 4 rotation iterations
    push.materials = bMat.addr;

    Kernel k("softBodySolveBda.spv", "sb_gm_cp_solveTetrahedronsPartitionLaunch", {}, sizeof(SolveTetraPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // ---- CPU reference -----------------------------------------------------
    std::vector<F4> refCpPos = cpPos, refCpVel = cpVel;
    std::vector<float> refMbyz(nbTets * 2);
    for (uint32_t t = 0; t < nbTets; ++t)
    { refMbyz[t*2+0] = 0.013f*float(t+1); refMbyz[t*2+1] = -0.007f*float(t+1); }

    const float invDtSq = invDt * invDt;
    Material mat = { youngs, poissons, damping, float(dampingScaleU16) * (1.0f/65535.0f), materialModel };

    for (uint32_t t = 0; t < nbTets; ++t)
    {
        F4 p[4], v[4];
        const uint32_t pi[4] = { pull[t].x & 0x7FFFFFFFu, pull[t].y, pull[t].z, pull[t].w };
        for (uint32_t i = 0; i < 4; ++i)
        {
            if (pi[i] < 0x7FFFFFFFu) { p[i] = unPos[pi[i]]; v[i] = unVel[pi[i]]; }
            else { p[i] = cpPos[t + i*nbTets]; v[i] = cpVel[t + i*nbTets]; }
        }
        const float u1[3] = { p[1].x-p[0].x, p[1].y-p[0].y, p[1].z-p[0].z };
        const float u2[3] = { p[2].x-p[0].x, p[2].y-p[0].y, p[2].z-p[0].z };
        const float u3[3] = { p[3].x-p[0].x, p[3].y-p[0].y, p[3].z-p[0].z };

        // The rotation refresh: seeded at the fixed point, 4 iterations.
        M3 U; for (int i=0;i<3;++i){ U.c0[i]=u1[i]; U.c1[i]=u2[i]; U.c2[i]=u3[i]; }
        M3 Fm; mat33Mul(U, rest[t], Fm);
        Q4 q = { tetRot[t].x, tetRot[t].y, tetRot[t].z, tetRot[t].w };
        if ((q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) == 0.0f) q = Q4{0,0,0,1};
        q = p80rot::extractRotation(Fm, q, 4);

        M3 dl;
        solveGM(dl, u1, u2, u3, v[0], v[1], v[2], v[3], &refMbyz[t*2], invDtSq, invDt,
                rest[t], q, isTGS, mat);

        const uint32_t wi[4] = { remap[t].x, remap[t].y, remap[t].z, remap[t].w };
        for (uint32_t i = 0; i < 4; ++i)
        {
            float dp[3];
            if (i == 0) for (int j=0;j<3;++j) dp[j] = -(dl.c0[j] + dl.c1[j] + dl.c2[j]) * p[i].w;
            else { const float* cc = (i==1)?dl.c0:((i==2)?dl.c1:dl.c2); for (int j=0;j<3;++j) dp[j] = cc[j]*p[i].w; }
            refCpPos[wi[i]] = { p[i].x+dp[0], p[i].y+dp[1], p[i].z+dp[2], p[i].w };
            refCpVel[wi[i]] = { v[i].x+dp[0]*invDt, v[i].y+dp[1]*invDt, v[i].z+dp[2]*invDt, v[i].w };
        }
    }

    const auto gPos = downloadBuffer<F4>(dev, bCpPos.h, cpPos.size());
    const auto gVel = downloadBuffer<F4>(dev, bCpVel.h, cpVel.size());
    const auto gMult = downloadBuffer<uint32_t>(dev, bMult.h, mult.size());

    // TOLERANCE SEAM, inherited — not from the constitutive maths, which is pure
    // mul/add/sub/div/cross/dot plus one PxSqrt and would be bit-exact on its own, but
    // from the rotation refresh: computeGMTetrahedronRotations -> extractRotation ->
    // PxQuat(w, omega) -> sin/cos (the tranche-5 seam).  Every solve term is a function of
    // that quaternion, so a last-bit difference in it shows up in the emitted positions and
    // velocities.  The case seeds the rotation at an already-converged unit quaternion and
    // uses isFirstIteration = false (4 iterations) to keep the amplification small; what
    // remains is budgeted from measurement, exactly as in tranche 5.
    size_t mm = 0;
    uint32_t worstUlp = 0;
    double worstRel = 0.0;
    auto cmpSoft = [&](const char* what, uint32_t i, const F4& r, const F4& g) {
        const float rc[4] = { r.x, r.y, r.z, r.w };
        const float gc[4] = { g.x, g.y, g.z, g.w };
        for (int e = 0; e < 4; ++e)
        {
            const uint32_t u = p80rot::ulpDist(rc[e], gc[e]);
            if (u > worstUlp) worstUlp = u;
            const double denom = std::fmax(1.0e-6, std::fabs(double(rc[e])));
            const double rel = std::fabs(double(rc[e]) - gc[e]) / denom;
            if (rel > worstRel) worstRel = rel;
            // Budget from measurement across the five cases: 10-248 ULP / up to 2.24e-5
            // relative, the damped co-rotational case being the widest (its denominator
            // carries `1 + damping*dtInv` and a scaledDamping*velocity term, both of which
            // amplify a last-bit rotation difference).  4096 ULP with a 1e-3 relative cap
            // sits ~15x above that and ~100x below what a real maths error produces.
            if (u > 4096u && rel > 1.0e-3)
            { ++mm; std::printf("  %s[%u].%d beyond budget: ref %.9g got %.9g (%u ULP, rel %.3e)\n",
                                what, i, e, rc[e], gc[e], u, rel); }
        }
    };
    for (uint32_t i = 0; i < uint32_t(cpPos.size()); ++i)
    {
        cmpSoft("cpPos", i, refCpPos[i], gPos[i]);
        cmpSoft("cpVel", i, refCpVel[i], gVel[i]);
    }
    std::printf("  [seam] solveTetraPartition(model %u, %s): worst %u ULP, worst relative %.3e\n",
                materialModel, isTGS ? "TGS" : "PGS", worstUlp, worstRel);
    // The PGS multiplier accumulation must be written back (and must NOT be under TGS).
    for (uint32_t t = 0; t < nbTets; ++t)
        for (uint32_t e = 0; e < 2u; ++e)
        {
            const float want = isTGS ? ((e == 0) ? 0.013f*float(t+1) : -0.007f*float(t+1)) : refMbyz[t*2+e];
            const uint32_t gotW = gMult[SPATIAL_BLOCK_MBYZ_WORD + t*2 + e];
            const uint32_t u = p80rot::ulpDist(want, u2f(gotW));
            if (u > worstUlp) worstUlp = u;
            const double mrel = std::fabs(double(want) - u2f(gotW)) / std::fmax(1.0e-6, std::fabs(double(want)));
            if (u > 4096u && mrel > 1.0e-3)
            { ++mm; std::printf("  multiplier[%u][%u] beyond budget: ref %.9g got %.9g (%u ULP)\n",
                                t, e, want, u2f(gotW), u); }
        }
    PXG_CHECK_MSG(mm == 0, "sb_gm_cp_solveTetrahedronsPartitionLaunch within the documented sin/cos budget vs CPU (%zu outliers)", mm);

    // The solve must have DONE something: at least one position must have moved.
    size_t moved = 0;
    for (uint32_t i = 0; i < uint32_t(cpPos.size()); ++i)
        if (f2u(gPos[i].x) != f2u(cpPos[i].x) || f2u(gPos[i].y) != f2u(cpPos[i].y)) ++moved;
    PXG_CHECK_MSG(moved > 0, "the constraint solve must move at least one vertex (it moved none)");

    dev->destroyBuffer(bCpPos.h); dev->destroyBuffer(bCpVel.h); dev->destroyBuffer(bUnPos.h);
    dev->destroyBuffer(bUnVel.h); dev->destroyBuffer(bPull.h); dev->destroyBuffer(bRemap.h);
    dev->destroyBuffer(bRest.h); dev->destroyBuffer(bRot.h); dev->destroyBuffer(bMult.h);
    dev->destroyBuffer(bMat.h); dev->destroyBuffer(bMatIdx.h); dev->destroyBuffer(bPart.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimSolveCorotationalPGS) { runSolveTetraPartition(0u, /*isTGS*/ false, /*damping*/ 0.0f); }
PXG_TEST(softBodySimSolveCorotationalDamped) { runSolveTetraPartition(0u, /*isTGS*/ false, /*damping*/ 0.35f); }
PXG_TEST(softBodySimSolveCorotationalTGS) { runSolveTetraPartition(0u, /*isTGS*/ true, /*damping*/ 0.0f); }
PXG_TEST(softBodySimSolveNeoHookeanPGS) { runSolveTetraPartition(1u, /*isTGS*/ false, /*damping*/ 0.0f); }
PXG_TEST(softBodySimSolveNeoHookeanTGS) { runSolveTetraPartition(1u, /*isTGS*/ true, /*damping*/ 0.0f); }

// MINIMAL REPRO, NOT REGISTERED.  Dispatch only — no CPU reference, no comparison, no
// downloads, dummy rest poses.  It still SIGSEGVs, which is what proves the tranche-6
// fault is GPU-side and not a defect in the driver code above.  This is the starting
// point for a standalone Mesa repro: strip the PhysX descriptor and feed the same push
// blob from a bare Vulkan harness.  Rename to PXG_TEST to re-enable.
static void softBodySimSolveProbe_DISABLED()
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t nbTets = 3;
    std::vector<F4> cpPos(nbTets*4, F4{0.1f,0.2f,0.3f,1.0f}), cpVel(nbTets*4, F4{0,0,0,0.5f});
    std::vector<F4> unPos(64, F4{0.4f,0.5f,0.6f,2.0f}), unVel(64, F4{0,0,0,0.25f});
    std::vector<p80refit::TetIdx4> pull(64, {0u,1u,2u,3u}), remap(64, {0u,1u,2u,3u});
    std::vector<uint32_t> restW(MAT33_BLOCK_WORDS + 512, 0u);
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        restW[t*4+0]=f2u(1.1f); restW[t*4+3]=f2u(0.0f);
        restW[MAT33_BLOCK_COL1_WORD+t*4+0]=f2u(1.2f);
        restW[MAT33_BLOCK_COL2_WORD+t]=f2u(1.3f);
    }
    std::vector<F4> rot(64, F4{0,0,0,1});
    std::vector<uint32_t> mult(SPATIAL_BLOCK_WORDS + 512, 0u);
    std::vector<uint32_t> matW(64, 0u);
    matW[0]=f2u(1.0e5f); matW[1]=f2u(0.31f); matW[4]=32768u;
    std::vector<uint16_t> matIdx(64, 0u);
    std::vector<uint32_t> parts(64, nbTets);

    DevBuf a=makeDev(dev,cpPos), b=makeDev(dev,cpVel), c2=makeDev(dev,unPos), d=makeDev(dev,unVel);
    DevBuf e=makeDev(dev,pull), f=makeDev(dev,remap), g=makeDev(dev,restW), h=makeDev(dev,rot);
    DevBuf i2=makeDev(dev,mult), j=makeDev(dev,matW), k2=makeDev(dev,matIdx), l=makeDev(dev,parts);

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimPosition_InvMassCP = reinterpret_cast<float4*>(size_t(a.addr));
    sb.mSimVelocity_invMassCP = reinterpret_cast<float4*>(size_t(b.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(c2.addr));
    sb.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(d.addr));
    sb.mSimPullIndices = reinterpret_cast<decltype(sb.mSimPullIndices)>(size_t(e.addr));
    sb.mSimRemapOutputCP = reinterpret_cast<decltype(sb.mSimRemapOutputCP)>(size_t(f.addr));
    sb.mSimTetraRestPoses = reinterpret_cast<decltype(sb.mSimTetraRestPoses)>(size_t(g.addr));
    sb.mSimTetraRotations = reinterpret_cast<float4*>(size_t(h.addr));
    sb.mSimTetraMultipliers = reinterpret_cast<decltype(sb.mSimTetraMultipliers)>(size_t(i2.addr));
    sb.mOrderedMaterialIndices = reinterpret_cast<PxU16*>(size_t(k2.addr));
    sb.mSimAccumulatedPartitionsCP = reinterpret_cast<PxU32*>(size_t(l.addr));
    sb.mNumTetsGM = nbTets; sb.mNumPartitionsGM = 1u; sb.mNumTetsPerElement = 1u;
    sb.mInitialRotation = PxQuat(0,0,0,1);
    std::vector<PxgSoftBody> da = { sb };
    DevBuf m=makeDev(dev,da);
    std::vector<uint32_t> act = {0u};
    DevBuf n=makeDev(dev,act);

    SolveTetraPush push = {};
    push.gSoftbodies=m.addr; push.activeSoftbodies=n.addr; push.numActiveSoftbodies=1u;
    push.invDt=60.0f; push.partitionId=0u; push.flagsWord=0u; push.materials=j.addr;

    Kernel kk("softBodySolveBda.spv", "sb_gm_cp_solveTetrahedronsPartitionLaunch", {}, sizeof(SolveTetraPush));
    kk.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();
    std::printf("  [probe] dispatch + flush survived\n");
}
static void (*sbKeepSolveProbeReferenced)() = &softBodySimSolveProbe_DISABLED;


// ===========================================================================
// P8.2 tranche a — sb_midphaseGeneratePairsLaunch (softbodyMidPhase.cu:199).
//
// One warp per contact-manager pair walks the softbody's own BV32 tetrahedron BVH against
// the rigid's fattened world AABB and appends uint4(cmIdx, tetIdx, 0, 0) to the shared
// collision stack for every SURFACE tetrahedron whose bounds overlap.
//
// The fixture drives TWO contact-manager pairs, which is deliberate: the engine packs
// MIDPHASE_WARPS_PER_BLOCK = 2 warps per block and launches ceil(numTests/2) blocks, and
// the Vulkan bridge forwards only gridDim.  Dispatching ONE group for two pairs is exactly
// what the engine does, so a port that indexed by gid.x alone would process only pair 0 and
// this test would see half the expected output.  That is the mismatch that shipped
// undetected in femClothSelfCollisionBda; here it is a hard assertion.
//
// The two pairs also exercise both branches of the shape swap: pair 0 has the tetrahedron
// mesh as shape0, pair 1 has it as shape1.
//
// Four tetrahedra pin the three per-primitive gates independently:
//   tet 0  surface, valid, inside the box   -> EMITTED
//   tet 1  INTERIOR (hint 0), inside        -> dropped by the surface-hint gate
//   tet 2  surface, valid, far outside      -> dropped by the tet-bounds test
//   tet 3  surface, DEGENERATE, inside      -> dropped by isValidTet
//   tet 4  surface, valid, in the FATTENING band only -> EMITTED only because the box is
//          grown by contactDistance[sb] + contactDistance[rigid].  Its min x is 2.05:
//          min x is 2.09: outside the raw box (2.0), outside it grown by EITHER contact
//          distance alone (2.08 / 2.02), and inside only when BOTH are summed (2.10).
//          Without this tet a
//          port that dropped the fattening passes the test unnoticed -- it did, until the
//          mutation sweep caught it.
// Every child bound is deliberately generous so the node-level cull cannot mask a
// primitive-level gate: each tet must be rejected by its OWN test, not by the tree walk.
// ===========================================================================
struct SbMidphasePush
{
    uint32_t numTests; float toleranceLength;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, softbodies;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(SbMidphasePush) == 80, "sb midphase push blob must be 80 bytes");
static_assert(offsetof(SbMidphasePush, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(SbMidphasePush, stackSizeBytes) == 56, "stackSizeBytes @56");
static_assert(offsetof(SbMidphasePush, stackPtr) == 64, "stackPtr @64");

struct CmInputHost { uint32_t shapeRef0, shapeRef1, cacheRef0, cacheRef1; };
struct PxgShapeHost
{
    float sx, sy, sz, srx, sry, srz, srw;
    uint32_t materialIndex; uint32_t hullLo, hullHi; uint32_t type, particleOrSoftbodyId;
};
static_assert(sizeof(PxgShapeHost) == 48, "PxgShape mirror must be 48 bytes");
static const uint32_t GEOM_TETRAHEDRONMESH_H = 7u;   // PxGeometryType::eTETRAHEDRONMESH

PXG_TEST(softBodySimMidphaseGeneratePairs)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the tetrahedron mesh ------------------------------------------------
    // 16 vertices = 4 tets x 4 corners, so each tet owns its own vertices and can be
    // placed or degenerated without disturbing the others.
    const float FAR = 100.0f;
    std::vector<F4> verts;
    auto pushTet = [&](float ox, float oy, float oz, bool degenerate) {
        verts.push_back(F4{ ox,        oy,        oz,        1.0f });
        verts.push_back(F4{ ox + 1.0f, oy,        oz,        1.0f });
        verts.push_back(F4{ ox,        oy + 1.0f, oz,        1.0f });
        // a degenerate tet puts the 4th corner in the plane of the other three
        verts.push_back(degenerate ? F4{ ox, oy, oz, 1.0f } : F4{ ox, oy, oz + 1.0f, 1.0f });
    };
    pushTet(0.0f, 0.0f, 0.0f, false);      // tet 0 — inside the box
    pushTet(0.0f, 0.0f, 0.0f, false);      // tet 1 — inside, but INTERIOR
    pushTet(FAR,  FAR,  FAR,  false);      // tet 2 — far outside
    pushTet(0.0f, 0.0f, 0.0f, true);       // tet 3 — inside, DEGENERATE
    pushTet(2.09f, 0.0f, 0.0f, false);     // tet 4 — only the FULLY fattened box reaches it

    const uint32_t NB_TETS = 5u;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < NB_TETS; ++t)
        tets.push_back(TetIdx4{ 4u * t + 0u, 4u * t + 1u, 4u * t + 2u, 4u * t + 3u });

    // mTetMeshSurfaceHint is a PxU8 array: tets 0/2/3 are surface, tet 1 is interior.
    // Packed into words so the shader's byte extraction is exercised at four distinct
    // byte offsets within one word.
    std::vector<uint8_t> hintBytes = { 1u, 0u, 1u, 1u, 1u };
    std::vector<uint32_t> hints(2, 0u);
    for (uint32_t i = 0; i < NB_TETS; ++i) hints[i / 4u] |= uint32_t(hintBytes[i]) << (8u * (i % 4u));

    // ---- the BV32 tree: one root node with four LEAF children, one tet each ----
    const uint32_t nbNodes = 1u, maxDepth = 1u;
    TreeBlob tree;
    tree.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes + 4u, 0u);
    tree.w[0] = uint32_t(verts.size());   // nbVerts
    tree.w[1] = NB_TETS;                  // nbTets
    tree.w[2] = maxDepth;
    tree.w[3] = nbNodes;                  // nbBv32TreeNodes
    const uint32_t nw = tree.nodeWord(0);
    tree.w[nw + W_NBNODES] = NB_TETS;
    for (uint32_t k = 0; k < NB_TETS; ++k)
    {
        // GENEROUS child bounds — every child overlaps the rigid box, so the node-level
        // cull can never stand in for a primitive-level gate.
        const float lo = -1000.0f, hi = 1000.0f;
        tree.w[nw + W_MMIN + k * 4 + 0] = f2u(lo);
        tree.w[nw + W_MMIN + k * 4 + 1] = f2u(lo);
        tree.w[nw + W_MMIN + k * 4 + 2] = f2u(lo);
        tree.w[nw + W_MMAX + k * 4 + 0] = f2u(hi);
        tree.w[nw + W_MMAX + k * 4 + 1] = f2u(hi);
        tree.w[nw + W_MMAX + k * 4 + 2] = f2u(hi);
        tree.w[nw + W_MDATA + k] = leafWord(/*startIndex*/ k, /*nbPrims*/ 1u);
    }

    DevBuf bTree = makeDev(dev, tree.w);
    DevBuf bVerts = makeDev(dev, verts);
    DevBuf bTets = makeDev(dev, tets);
    DevBuf bHints = makeDev(dev, hints);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
    sb.mNumTets = NB_TETS;
    sb.mNumVerts = uint32_t(verts.size());
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the narrowphase inputs ---------------------------------------------
    // shape 0 = the tetrahedron mesh (softbody 0); shape 1 = a rigid box.
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TETRAHEDRONMESH_H; shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = 3u /* eBOX */;          shapes[1].particleOrSoftbodyId = 0xFFFFFFFFu;
    DevBuf bShapes = makeDev(dev, shapes);

    // Pair 0: tetmesh as shape0.  Pair 1: tetmesh as shape1 -> exercises the swap.
    std::vector<CmInputHost> cms = {
        { 0u, 1u, /*cacheRef0 sb*/ 0u, /*cacheRef1 rigid*/ 1u },
        { 1u, 0u, /*cacheRef0 rigid*/ 1u, /*cacheRef1 sb*/ 0u },
    };
    DevBuf bCms = makeDev(dev, cms);

    // bounds[]: PxBounds3, 24 B each.  Slot 0 is the softbody's (unused by this kernel),
    // slot 1 is the rigid's -- a unit box at the origin, which tets 0/1/3 sit inside and
    // tet 2 (at 100,100,100) is far outside.
    std::vector<float> bounds = {
        -50.0f, -50.0f, -50.0f,  50.0f,  50.0f,  50.0f,     // slot 0 (softbody, unused)
        -2.0f,  -2.0f,  -2.0f,   2.0f,   2.0f,   2.0f,      // slot 1 (rigid)
    };
    DevBuf bBounds = makeDev(dev, bounds);
    // DELIBERATELY ASYMMETRIC: 0.08 + 0.02 = 0.10, and tet 4 sits at 2.09.  Either half
    // alone falls short (2.08 and 2.02), so only the SUM reaches it.  With equal halves --
    // and with the tet at 2.05 -- a port that used just one of them still passed; the
    // mutation sweep tried exactly that twice before this fixture had teeth on it.
    std::vector<float> contactDist = { 0.08f, 0.02f };      // total 0.1 -> box grows to 2.1
    DevBuf bCd = makeDev(dev, contactDist);

    const uint32_t STACK_ENTRIES = 32u;
    const uint32_t POISON = 0xCDCDCDCDu;
    std::vector<uint32_t> stack(STACK_ENTRIES * 4u, POISON);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairCount = { 0u };
    DevBuf bCount = makeDev(dev, pairCount);

    if (!bTree.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    SbMidphasePush push = {};
    push.numTests = 2u;
    push.toleranceLength = 1.0f;      // ABI parity only
    push.cmInputs = bCms.addr;
    push.transformCache = 0ull;       // ABI parity only — never read
    push.bounds = bBounds.addr;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.stackSizeBytes = STACK_ENTRIES * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bCount.addr;

    Kernel k("softBodyContactGenBda.spv", "sb_midphaseGeneratePairsLaunch", {}, sizeof(SbMidphasePush));
    // THE ENGINE'S GRID: ceil(numTests / MIDPHASE_WARPS_PER_BLOCK) = ceil(2/2) = 1.
    // One group must cover BOTH pairs.
    k.dispatch(/*gridX*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, stack.size());

    // ---- expected: exactly (0, tet0) and (1, tet0) ---------------------------
    // tets 0 and 4 for each of the two contact managers.
    PXG_CHECK_MSG(gCount == 4u,
                  "midphase must emit exactly 4 pairs (tets 0 and 4 for each of the two "
                  "contact managers), got %u -- a count of 2 means the sub-warp loop is "
                  "missing and only pair 0 was processed", gCount);

    bool sawPair0 = false, sawPair1 = false;
    bool sawFattenTet[2] = { false, false };
    for (uint32_t i = 0; i < gCount && i < STACK_ENTRIES; ++i)
    {
        const uint32_t cm = gStack[i * 4u + 0u], prim = gStack[i * 4u + 1u];
        const uint32_t z = gStack[i * 4u + 2u], w = gStack[i * 4u + 3u];
        std::printf("  pair %u: cmIdx=%u tet=%u (z=%u w=%u)\n", i, cm, prim, z, w);
        PXG_CHECK_MSG(prim == 0u || prim == 4u,
                      "only tets 0 and 4 may be emitted, got tet %u for cmIdx %u", prim, cm);
        PXG_CHECK_MSG(z == 0u && w == 0u, "the uint4 payload's z/w must be zero (got %u,%u)", z, w);
        if (cm == 0u) sawPair0 = true;
        if (cm == 1u) sawPair1 = true;
        if (prim == 4u && cm < 2u) sawFattenTet[cm] = true;
    }
    PXG_CHECK_MSG(sawPair0, "contact-manager pair 0 produced no entry");
    // THE FATTENING TEETH: tet 4 sits at x >= 2.05, outside the raw rigid box (2.0) and
    // inside it only once grown by contactDistance[sb] + contactDistance[rigid] = 0.1.
    PXG_CHECK_MSG(sawFattenTet[0] && sawFattenTet[1],
                  "tet 4 is reachable ONLY by the fattened box -- a port that dropped "
                  "aabb.fattenFast(contactDist) loses it (cm0 %d, cm1 %d)",
                  int(sawFattenTet[0]), int(sawFattenTet[1]));
    // THE GRID TEETH: pair 1 only exists if the block's SECOND warp was walked.
    PXG_CHECK_MSG(sawPair1,
                  "contact-manager pair 1 produced no entry -- the engine packs 2 warps per "
                  "block and the bridge forwards only gridDim, so the shader must walk both");
    // ...and pair 1 also proves the shape swap: its tetmesh is shape1, so a port that always
    // took shape0 would have read the rigid box's particleOrSoftbodyId (0xFFFFFFFF) as the
    // softbody index and faulted or produced nothing.

    // Entries past the emitted pairs must keep their poison.
    for (uint32_t i = gCount; i < STACK_ENTRIES; ++i)
        for (uint32_t c = 0; c < 4u; ++c)
            PXG_CHECK_MSG(gStack[i * 4u + c] == POISON,
                          "stack entry %u word %u was written past the pair count", i, c);

    // ---- overflow: the counter keeps counting, the writes are DROPPED --------
    // pushOntoStackFullWarp reserves its slot with the atomic FIRST and only then checks
    // `stackIndex < stackSize` (bv32Traversal.cuh:197-206), so an undersized stack must
    // still report the true pair count while writing at most stackSize entries.  The engine
    // relies on exactly that to size mStackSizeNeededOnDevice for the next frame.
    {
        const uint32_t SMALL = 2u;
        const uint32_t PAD = 8u;   // trailing slots that must NEVER be written
        std::vector<uint32_t> smallStack((SMALL + PAD) * 4u, POISON);
        DevBuf bSmall = makeDev(dev, smallStack);
        std::vector<uint32_t> c2 = { 0u };
        DevBuf bCount2 = makeDev(dev, c2);
        SbMidphasePush p2 = push;
        p2.stackSizeBytes = SMALL * 16u;
        p2.stackPtr = bSmall.addr;
        p2.midphasePairsNum = bCount2.addr;
        Kernel k2("softBodyContactGenBda.spv", "sb_midphaseGeneratePairsLaunch", {}, sizeof(SbMidphasePush));
        k2.dispatch(1u, {}, p2, 1u, 1u);
        dev->flush();
        const uint32_t n2 = downloadBuffer<uint32_t>(dev, bCount2.h, 1)[0];
        const auto s2 = downloadBuffer<uint32_t>(dev, bSmall.h, smallStack.size());
        PXG_CHECK_MSG(n2 == 4u, "the pair COUNT must be the true total even when the stack "
                                "overflows (got %u, expected 4)", n2);
        uint32_t written = 0u;
        for (uint32_t i = 0; i < SMALL; ++i)
            if (s2[i * 4u + 0u] != POISON) ++written;
        PXG_CHECK_MSG(written == SMALL,
                      "the first %u slots should still have been filled (got %u)", SMALL, written);
        // THE OVERFLOW TEETH: the surplus must be DROPPED, not written past the end.  Only
        // this padding can see that -- counting the filled slots cannot, because a port that
        // skipped the bounds check writes into memory the allocation happens to have.
        for (uint32_t i = SMALL; i < SMALL + PAD; ++i)
            for (uint32_t c = 0; c < 4u; ++c)
                PXG_CHECK_MSG(s2[i * 4u + c] == POISON,
                              "slot %u word %u was written past stackSize -- the "
                              "`stackIndex < stackSize` drop is missing", i, c);
        dev->destroyBuffer(bSmall.h); dev->destroyBuffer(bCount2.h);
    }

    dev->destroyBuffer(bTree.h); dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bTets.h);
    dev->destroyBuffer(bHints.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bShapes.h);
    dev->destroyBuffer(bCms.h); dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bStack.h); dev->destroyBuffer(bCount.h);
}

// ===========================================================================
// P8.2 tranche b — sb_primitiveContactGenLaunch, PLANE path (softbodyPrimitives.cu:573
// -> tetPlaneCollide :247).  Four lanes test the tet's four corners against the plane and
// emit one contact per penetrating corner through PxgFEMContactWriter.
//
// The writer is passed BY VALUE in CUDA, so it is inlined into the push blob at @112 and
// the blob is 192 bytes.  The static_asserts below are the guard on that layout -- get one
// field offset wrong and every contact lands in the wrong buffer.
//
// The fixture places ONE tet straddling the y = 0 plane so the four corners split 2/2:
// two below (penetrating, contacts expected) and two above by more than the contact
// distance (no contact).  That split is what gives the per-corner test teeth -- a port
// that emitted per-TET rather than per-CORNER, or that used the wrong plane axis, lands on
// a different count.
// ===========================================================================
struct FemWriterHost
{
    uint64_t outPoint, outNormalPen, outBarycentric, outContactInfo, totalContactCount;
    uint64_t contactByX, tempContactByX, contactIndexSortedByX, contactSortedByX;
    uint32_t maxNumContacts, pad;
};
struct SbContactGenPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes,
             softbodies, shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad84;
    uint64_t softBodyRigidPairs;
    uint32_t numSoftBodyRigidPairs, pad100;
    uint64_t stackSizeNeededOnDevice;
    FemWriterHost w;
};
static_assert(sizeof(SbContactGenPush) == 192, "sb contact-gen push blob must be 192 bytes");
static_assert(offsetof(SbContactGenPush, stackPtr) == 64, "stackPtr @64");
static_assert(offsetof(SbContactGenPush, stackSizeBytes) == 80, "stackSizeBytes @80");
static_assert(offsetof(SbContactGenPush, stackSizeNeededOnDevice) == 104, "stackSizeNeeded @104");
static_assert(offsetof(SbContactGenPush, w) == 112, "the writer is inlined at @112");
static_assert(offsetof(SbContactGenPush, w) + offsetof(FemWriterHost, maxNumContacts) == 184,
              "writer.maxNumContacts @184");

// PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd u32 @12.
struct FemOtherContactInfoHost { uint64_t pairInd0; uint32_t pairInd1, rigidMatInd; };
static_assert(sizeof(FemOtherContactInfoHost) == 16, "PxgFemOtherContactInfo is 16 bytes");

// ===========================================================================
// P8.2 tranche c — the collision-mesh -> simulation-mesh contact REMAP
// (softbodyPrimitives.cu:2134) and the sorted-order GATHER (:1695).
//
// Both are lifts of npSbRemap.slang's unit-tested descriptor-set ports onto the BDA ABI,
// so the arithmetic is already covered there.  What these fixtures pin is what the LIFT
// can get wrong: the push-blob layout, the optional prevContactCount pointer, the grid
// stride, and the three tie-breaks that are easy to lose in translation.
//
// The remap fixture uses THREE sim tets behind one collision tet:
//   candidate 0  does NOT contain the point, nearest distance LARGE
//   candidate 1  does NOT contain the point, nearest distance SMALL   <- min fallback winner
//   candidate 2  CONTAINS the point                                   <- containment winner
// Contact A's point is inside candidate 2, so containment must win and pairInd1 must name
// tet 2.  Contact B's point is outside all three, so the min fallback must name tet 1 --
// the nearest, not the first and not the last.
// ===========================================================================
struct SbRemapPush
{
    uint64_t softbodies; uint32_t maxNumContacts, pad12;
    uint64_t point, outBarycentric, outContactInfo, totalContactCount, prevContactCount;
};
static_assert(sizeof(SbRemapPush) == 56, "sb remap push blob must be 56 bytes");
static_assert(offsetof(SbRemapPush, point) == 16, "point @16");
static_assert(offsetof(SbRemapPush, prevContactCount) == 48, "prevContactCount @48");

struct SbReorderPush
{
    uint64_t contacts, normalPens, barycentrics, contactInfos, numContacts, remapByRigid,
             sortedContacts, sortedNormalPens, sortedBarycentrics, sortedContactInfos;
};
static_assert(sizeof(SbReorderPush) == 80, "sb reorder push blob must be 80 bytes");

PXG_TEST(softBodySimContactRemapToSim)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    // Three sim tets.  Tet 2 is the unit tet at the origin and CONTAINS (0.2, 0.2, 0.2).
    // Tets 0 and 1 sit away from it at different distances.
    std::vector<F4> simVerts = {
        { 10.0f, 0.0f, 0.0f, 1.0f }, { 11.0f, 0.0f, 0.0f, 1.0f },
        { 10.0f, 1.0f, 0.0f, 1.0f }, { 10.0f, 0.0f, 1.0f, 1.0f },   // tet 0, far
        {  2.0f, 0.0f, 0.0f, 1.0f }, {  3.0f, 0.0f, 0.0f, 1.0f },
        {  2.0f, 1.0f, 0.0f, 1.0f }, {  2.0f, 0.0f, 1.0f, 1.0f },   // tet 1, nearer
        {  0.0f, 0.0f, 0.0f, 1.0f }, {  1.0f, 0.0f, 0.0f, 1.0f },
        {  0.0f, 1.0f, 0.0f, 1.0f }, {  0.0f, 0.0f, 1.0f, 1.0f },   // tet 2, contains A
    };
    for (int t = 0; t < 4; ++t) simVerts.push_back(F4{ 0, 0, 0, 0 });   // trailer
    std::vector<TetIdx4> simTets = {
        { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u }, { 8u, 9u, 10u, 11u }, { 0u, 0u, 0u, 0u },
    };
    // collision tet 0 maps to sim candidates {0, 1, 2}, in that order.
    std::vector<uint32_t> remapColToSim = { 0u, 1u, 2u, 0u, 0u, 0u, 0u, 0u };
    std::vector<uint32_t> accumRemap = { 3u, 3u, 3u, 3u };   // inclusive prefix sum

    DevBuf bSv = makeDev(dev, simVerts), bSt = makeDev(dev, simTets);
    DevBuf bRc = makeDev(dev, remapColToSim), bAr = makeDev(dev, accumRemap);

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bSt.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSv.addr));
    sb.mTetsRemapColToSim = reinterpret_cast<PxU32*>(size_t(bRc.addr));
    sb.mTetsAccumulatedRemapColToSim = reinterpret_cast<PxU32*>(size_t(bAr.addr));
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // contact A is inside tet 2; contact B is outside all three but nearest to tet 1.
    std::vector<F4> points = { { 0.2f, 0.2f, 0.2f, 0.0f }, { 1.6f, 0.2f, 0.2f, 0.0f },
                               { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
    DevBuf bPt = makeDev(dev, points);
    // both contacts start out naming collision tet 0 of softbody 0
    std::vector<FemOtherContactInfoHost> infos(4);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemOtherContactInfoHost));
    infos[0].pairInd0 = 0x1111ull; infos[0].pairInd1 = 0u; infos[0].rigidMatInd = 5u;
    infos[1].pairInd0 = 0x2222ull; infos[1].pairInd1 = 0u; infos[1].rigidMatInd = 6u;
    DevBuf bInfo = makeDev(dev, infos);
    std::vector<F4> bary(4, F4{ -1, -1, -1, -1 });
    DevBuf bBary = makeDev(dev, bary);
    std::vector<uint32_t> count = { 2u, 0u, 0u, 0u };
    DevBuf bCount = makeDev(dev, count);

    if (!bSv.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    SbRemapPush push = {};
    push.softbodies = bDesc.addr;
    push.maxNumContacts = 64u;
    push.point = bPt.addr;
    push.outBarycentric = bBary.addr;
    push.outContactInfo = bInfo.addr;
    push.totalContactCount = bCount.addr;
    push.prevContactCount = 0ull;    // OPTIONAL in CUDA -> must not be dereferenced

    Kernel k("softBodyContactRemapBda.spv", "sb_other_contact_remap_to_simLaunch", {}, sizeof(SbRemapPush));
    k.dispatch(/*gridX*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, bInfo.h, 4);
    const auto gBary = downloadBuffer<F4>(dev, bBary.h, 4);
    for (uint32_t i = 0; i < 2u; ++i)
        std::printf("  contact %u: pairInd1 -> tet %u, bary(%.4f, %.4f, %.4f, %.4f)\n",
                    i, gInfo[i].pairInd1 & 0xfffffu, gBary[i].x, gBary[i].y, gBary[i].z, gBary[i].w);

    // CONTAINMENT wins for contact A: the point is inside candidate 2.
    PXG_CHECK_MSG((gInfo[0].pairInd1 & 0xfffffu) == 2u,
                  "contact A is inside sim tet 2, so containment must win (got tet %u)",
                  gInfo[0].pairInd1 & 0xfffffu);
    // ...and its barycentric must reconstruct the point: 0.2/0.2/0.2 inside the unit tet.
    PXG_CHECK_MSG(std::fabs(gBary[0].y - 0.2f) < 1e-4f && std::fabs(gBary[0].z - 0.2f) < 1e-4f &&
                  std::fabs(gBary[0].w - 0.2f) < 1e-4f && std::fabs(gBary[0].x - 0.4f) < 1e-4f,
                  "contact A's barycentric must be (0.4, 0.2, 0.2, 0.2), got (%.4f, %.4f, %.4f, %.4f)",
                  gBary[0].x, gBary[0].y, gBary[0].z, gBary[0].w);
    // The MIN FALLBACK wins for contact B: nearest is candidate 1, not 0 and not 2.
    PXG_CHECK_MSG((gInfo[1].pairInd1 & 0xfffffu) == 1u,
                  "contact B is outside every candidate, so the NEAREST (tet 1) must win "
                  "(got tet %u)", gInfo[1].pairInd1 & 0xfffffu);
    // the fallback clamps every component into [0,1]
    for (uint32_t c = 0; c < 4u; ++c)
    {
        const float v = (c == 0) ? gBary[1].x : (c == 1) ? gBary[1].y : (c == 2) ? gBary[1].z : gBary[1].w;
        PXG_CHECK_MSG(v >= 0.0f && v <= 1.0f, "the fallback barycentric must be clamped (comp %u = %g)", c, v);
    }
    // the softbody id survives the rewrite, and the rest of the info word is untouched
    PXG_CHECK_MSG((gInfo[0].pairInd1 >> 20u) == 0u, "the softbody id must survive the rewrite");
    PXG_CHECK_MSG(gInfo[0].pairInd0 == 0x1111ull && gInfo[0].rigidMatInd == 5u,
                  "the remap must touch ONLY pairInd1");
    PXG_CHECK_MSG(gInfo[1].pairInd0 == 0x2222ull && gInfo[1].rigidMatInd == 6u,
                  "the remap must touch ONLY pairInd1");
    // contacts past the count must be untouched
    PXG_CHECK_MSG(gBary[2].x == -1.0f, "contact 2 is past the count and must be untouched");

    // ---- the clamp write-back ------------------------------------------------
    {
        std::vector<uint32_t> c2 = { 9u, 0u, 0u, 0u };   // more contacts than maxNumContacts
        DevBuf bC2 = makeDev(dev, c2);
        SbRemapPush p2 = push;
        p2.maxNumContacts = 2u;
        p2.totalContactCount = bC2.addr;
        Kernel("softBodyContactRemapBda.spv", "sb_other_contact_remap_to_simLaunch", {}, sizeof(SbRemapPush))
            .dispatch(1u, {}, p2, 1u, 1u);
        dev->flush();
        const uint32_t back = downloadBuffer<uint32_t>(dev, bC2.h, 1)[0];
        PXG_CHECK_MSG(back == 2u, "an over-large contact count must be CLAMPED and written back "
                                  "(9 -> maxNumContacts 2), got %u", back);
        dev->destroyBuffer(bC2.h);
    }

    // ---- prevContactCount, the pointer that must NOT be dereferenced when null ----
    // Both halves of the pointer-select in the shader are exercised here: the dispatches
    // above pass null (start at 0), this one passes a real pointer holding 1 so contact 0
    // is left alone and only contact 1 is remapped.  Without both, a regression that
    // simply ignored prevContactCount would pass -- and the null half is what faulted on
    // llvmpipe, so the two cases together fence the fix from either side.
    {
        std::vector<F4> b3(4, F4{ -1, -1, -1, -1 });
        DevBuf bB3 = makeDev(dev, b3);
        // a FRESH info buffer: the dispatches above already rewrote pairInd1 in bInfo, and
        // reusing it would hand this dispatch collision tet 1 (range 0) instead of tet 0,
        // making the "contact 1 was processed" check pass on an empty candidate set.
        std::vector<FemOtherContactInfoHost> i3 = infos;
        DevBuf bI3 = makeDev(dev, i3);
        std::vector<uint32_t> prev = { 1u, 0u, 0u, 0u };
        DevBuf bPrev = makeDev(dev, prev);
        SbRemapPush p3 = push;
        p3.outBarycentric = bB3.addr;
        p3.outContactInfo = bI3.addr;
        p3.prevContactCount = bPrev.addr;
        Kernel("softBodyContactRemapBda.spv", "sb_other_contact_remap_to_simLaunch", {}, sizeof(SbRemapPush))
            .dispatch(1u, {}, p3, 1u, 1u);
        dev->flush();
        const auto g3 = downloadBuffer<F4>(dev, bB3.h, 4);
        const auto gi3 = downloadBuffer<FemOtherContactInfoHost>(dev, bI3.h, 4);
        PXG_CHECK_MSG(g3[0].x == -1.0f,
                      "prevContactCount = 1 must SKIP contact 0, but its barycentric was "
                      "written (%.4f)", g3[0].x);
        PXG_CHECK_MSG(gi3[0].pairInd1 == 0u,
                      "prevContactCount = 1 must leave contact 0's pairInd1 alone (got 0x%08x)",
                      gi3[0].pairInd1);
        // contact 1 still runs the full nearest-candidate search, so it must land on tet 1.
        PXG_CHECK_MSG((gi3[1].pairInd1 & 0xfffffu) == 1u,
                      "prevContactCount = 1 must still remap contact 1 to the nearest tet 1 "
                      "(got tet %u)", gi3[1].pairInd1 & 0xfffffu);
        PXG_CHECK_MSG(g3[1].x != -1.0f, "contact 1's barycentric must have been written");
        dev->destroyBuffer(bB3.h); dev->destroyBuffer(bI3.h); dev->destroyBuffer(bPrev.h);
    }

    dev->destroyBuffer(bSv.h); dev->destroyBuffer(bSt.h); dev->destroyBuffer(bRc.h);
    dev->destroyBuffer(bAr.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bPt.h);
    dev->destroyBuffer(bInfo.h); dev->destroyBuffer(bBary.h); dev->destroyBuffer(bCount.h);
}

// ===========================================================================
// The three tie-breaks, and the accumulated-remap START index.
//
// softBodySimContactRemapToSim above pins the two OUTCOMES (containment wins; otherwise
// the nearest candidate wins) but it cannot pin HOW a tie is broken: with three candidates
// in one batch and no two of them equidistant, every plausible variant of the reduction
// agrees.  Mutation-tested, four independent defects survived that fixture --
// `<` weakened to `<=`, the winner scan's `break` dropped, the ballot's
// "first hit wins" guard dropped, and startInd forced to 0 -- so each gets geometry here
// that makes it observable:
//
//   contact 0 -> collision tet 0, candidates [0, 40): a SAME-LANE tie.  Candidates 5 and
//     37 are byte-identical tets, 32 apart, so they land in the same lane in different
//     batches and compare exactly equal.  `<` keeps the first, `<=` takes the second.
//   contact 1 -> collision tet 1, candidates [40, 50): a CROSS-LANE tie.  Candidates 3
//     and 6 of that range are byte-identical, so two lanes hold the equal minimum and only
//     the scan's `break` makes the lower one win.  This contact also has a nonzero
//     collision tet index, so a startInd forced to 0 searches the wrong candidate range
//     and lands in contact 0's tets instead.
//   contact 2 -> collision tet 2, candidates [50, 54): TWO CONTAINING candidates, a unit
//     tet and a double-size tet sharing the origin corner.  Both contain the point, so
//     only the ballot's first-hit guard decides, and the two give different barycentrics.
//
// The ties are exact because the tied candidates are the SAME tet duplicated at different
// sim indices -- identical coordinates run identical arithmetic, so the comparison is
// bit-equal without relying on any float coincidence.
// ===========================================================================
PXG_TEST(softBodySimContactRemapTieBreaks)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    // A unit tet whose closest point to the ORIGIN is its corner (dx, 0, 0): squared
    // distance dx*dx, and identical dx gives a bit-identical tet.
    std::vector<F4> simVerts;
    std::vector<TetIdx4> simTets;
    auto pushTet = [&](float ox, float oy, float oz, float s) {
        const uint32_t base = uint32_t(simVerts.size());
        simVerts.push_back(F4{ ox, oy, oz, 1.0f });
        simVerts.push_back(F4{ ox + s, oy, oz, 1.0f });
        simVerts.push_back(F4{ ox, oy + s, oz, 1.0f });
        simVerts.push_back(F4{ ox, oy, oz + s, 1.0f });
        simTets.push_back(TetIdx4{ base, base + 1u, base + 2u, base + 3u });
    };

    const uint32_t kNear = 5u, kNearTwin = 37u;        // same lane (32 apart), batches 0 and 1
    for (uint32_t t = 0; t < 40u; ++t)
        pushTet((t == kNear || t == kNearTwin) ? 2.0f : (6.0f + float(t)), 0.0f, 0.0f, 1.0f);
    const uint32_t kCross = 43u, kCrossTwin = 46u;     // workIndex 3 and 6 -> different lanes
    for (uint32_t t = 40u; t < 50u; ++t)
        pushTet((t == kCross || t == kCrossTwin) ? 2.0f : (6.0f + float(t)), 0.0f, 0.0f, 1.0f);
    pushTet(0.0f, 0.0f, 0.0f, 1.0f);                   // tet 50: unit tet at the origin
    pushTet(0.0f, 0.0f, 0.0f, 2.0f);                   // tet 51: double-size, also contains
    pushTet(20.0f, 0.0f, 0.0f, 1.0f);                  // tet 52: far
    pushTet(30.0f, 0.0f, 0.0f, 1.0f);                  // tet 53: far
    PXG_CHECK_MSG(simTets.size() == 54u, "the tie-break fixture needs 54 sim tets, built %zu",
                  simTets.size());

    // collision tet 0 -> [0,40), collision tet 1 -> [40,50), collision tet 2 -> [50,54).
    std::vector<uint32_t> remapColToSim(54);
    for (uint32_t i = 0; i < 54u; ++i) remapColToSim[i] = i;
    std::vector<uint32_t> accumRemap = { 40u, 50u, 54u, 54u };   // inclusive prefix sum

    DevBuf bSv = makeDev(dev, simVerts), bSt = makeDev(dev, simTets);
    DevBuf bRc = makeDev(dev, remapColToSim), bAr = makeDev(dev, accumRemap);
    if (!bSv.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bSt.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSv.addr));
    sb.mTetsRemapColToSim = reinterpret_cast<PxU32*>(size_t(bRc.addr));
    sb.mTetsAccumulatedRemapColToSim = reinterpret_cast<PxU32*>(size_t(bAr.addr));
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<F4> points = { { 0.0f, 0.0f, 0.0f, 0.0f },      // outside every candidate
                               { 0.0f, 0.0f, 0.0f, 0.0f },
                               { 0.1f, 0.1f, 0.1f, 0.0f },      // inside tets 50 AND 51
                               { 0, 0, 0, 0 } };
    DevBuf bPt = makeDev(dev, points);
    std::vector<FemOtherContactInfoHost> infos(4);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemOtherContactInfoHost));
    infos[0].pairInd1 = 0u;   // softbody 0, collision tet 0
    infos[1].pairInd1 = 1u;   // softbody 0, collision tet 1
    infos[2].pairInd1 = 2u;   // softbody 0, collision tet 2
    DevBuf bInfo = makeDev(dev, infos);
    std::vector<F4> bary(4, F4{ -1, -1, -1, -1 });
    DevBuf bBary = makeDev(dev, bary);
    std::vector<uint32_t> count = { 3u, 0u, 0u, 0u };
    DevBuf bCount = makeDev(dev, count);

    SbRemapPush push = {};
    push.softbodies = bDesc.addr;
    push.maxNumContacts = 64u;
    push.point = bPt.addr;
    push.outBarycentric = bBary.addr;
    push.outContactInfo = bInfo.addr;
    push.totalContactCount = bCount.addr;
    push.prevContactCount = 0ull;

    Kernel("softBodyContactRemapBda.spv", "sb_other_contact_remap_to_simLaunch", {}, sizeof(SbRemapPush))
        .dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, bInfo.h, 4);
    const auto gBary = downloadBuffer<F4>(dev, bBary.h, 4);
    for (uint32_t i = 0; i < 3u; ++i)
        std::printf("  contact %u -> sim tet %u, bary(%.4f, %.4f, %.4f, %.4f)\n",
                    i, gInfo[i].pairInd1 & 0xfffffu, gBary[i].x, gBary[i].y, gBary[i].z, gBary[i].w);

    PXG_CHECK_MSG((gInfo[0].pairInd1 & 0xfffffu) == kNear,
                  "a SAME-LANE tie must keep the FIRST candidate (tet %u), not its twin 32 "
                  "later (tet %u) -- got tet %u", kNear, kNearTwin, gInfo[0].pairInd1 & 0xfffffu);
    PXG_CHECK_MSG((gInfo[1].pairInd1 & 0xfffffu) == kCross,
                  "a CROSS-LANE tie must win on the LOWEST lane (tet %u), not the highest "
                  "(tet %u) -- got tet %u; a startInd of 0 would land near tet %u instead",
                  kCross, kCrossTwin, gInfo[1].pairInd1 & 0xfffffu, kNear);
    PXG_CHECK_MSG((gInfo[2].pairInd1 & 0xfffffu) == 50u,
                  "with two CONTAINING candidates the FIRST one (tet 50) must win, not the "
                  "last (tet 51) -- got tet %u", gInfo[2].pairInd1 & 0xfffffu);
    // tet 50 is the unit tet, so (0.1, 0.1, 0.1) has barycentric (0.7, 0.1, 0.1, 0.1);
    // tet 51 is twice the size and would give (0.85, 0.05, 0.05, 0.05).  Checking the
    // weights and not just the index keeps the containing-tet choice honest.
    PXG_CHECK_MSG(std::fabs(gBary[2].x - 0.7f) < 1e-4f && std::fabs(gBary[2].y - 0.1f) < 1e-4f &&
                  std::fabs(gBary[2].z - 0.1f) < 1e-4f && std::fabs(gBary[2].w - 0.1f) < 1e-4f,
                  "the containing tet's barycentric must be (0.7, 0.1, 0.1, 0.1), got "
                  "(%.4f, %.4f, %.4f, %.4f)", gBary[2].x, gBary[2].y, gBary[2].z, gBary[2].w);

    // ---- the count clamp must actually BOUND the loop, not just write itself back ----
    // The clamp write-back alone is checked in the fixture above.  What is checked here is
    // that the clamped count is the value the loop USES: with 12 contacts declared and
    // maxNumContacts 3, contacts 3..11 must be left untouched.  Deleting the clamp of the
    // local `numContacts` -- while keeping the write-back -- is otherwise invisible.
    {
        const uint32_t kBig = 12u;
        std::vector<F4> pB(kBig, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
        DevBuf bPB = makeDev(dev, pB);
        std::vector<FemOtherContactInfoHost> iB(kBig);
        std::memset(iB.data(), 0, iB.size() * sizeof(FemOtherContactInfoHost));
        DevBuf bIB = makeDev(dev, iB);
        std::vector<F4> bB(kBig, F4{ -1, -1, -1, -1 });
        DevBuf bBB = makeDev(dev, bB);
        std::vector<uint32_t> cB = { kBig, 0u, 0u, 0u };
        DevBuf bCB = makeDev(dev, cB);

        SbRemapPush pC = push;
        pC.maxNumContacts = 3u;
        pC.point = bPB.addr;
        pC.outContactInfo = bIB.addr;
        pC.outBarycentric = bBB.addr;
        pC.totalContactCount = bCB.addr;
        Kernel("softBodyContactRemapBda.spv", "sb_other_contact_remap_to_simLaunch", {}, sizeof(SbRemapPush))
            .dispatch(1u, {}, pC, 1u, 1u);
        dev->flush();

        const uint32_t back = downloadBuffer<uint32_t>(dev, bCB.h, 1)[0];
        PXG_CHECK_MSG(back == 3u, "the over-large count must be clamped and written back "
                                  "(12 -> 3), got %u", back);
        const auto gB = downloadBuffer<F4>(dev, bBB.h, kBig);
        for (uint32_t i = 0; i < 3u; ++i)
            PXG_CHECK_MSG(gB[i].x != -1.0f, "contact %u is within the clamp and must be "
                                            "remapped", i);
        for (uint32_t i = 3u; i < kBig; ++i)
            PXG_CHECK_MSG(gB[i].x == -1.0f,
                          "contact %u is past the CLAMPED count and must be untouched, but its "
                          "barycentric was written (%.4f) -- the clamp was written back without "
                          "bounding the loop", i, gB[i].x);
        dev->destroyBuffer(bPB.h); dev->destroyBuffer(bIB.h);
        dev->destroyBuffer(bBB.h); dev->destroyBuffer(bCB.h);
    }

    dev->destroyBuffer(bSv.h); dev->destroyBuffer(bSt.h); dev->destroyBuffer(bRc.h);
    dev->destroyBuffer(bAr.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bPt.h);
    dev->destroyBuffer(bInfo.h); dev->destroyBuffer(bBary.h); dev->destroyBuffer(bCount.h);
}

PXG_TEST(softBodySimReorderPSContacts)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 5u;
    // remapByRigid is a PERMUTATION with no fixed points, so an identity gather fails.
    std::vector<uint32_t> remap = { 4u, 2u, 0u, 3u, 1u,  0u, 0u, 0u };
    std::vector<F4> pts(N), nps(N), bary(N);
    std::vector<FemOtherContactInfoHost> infos(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        pts[i]  = F4{ float(i) + 0.1f, 0, 0, 0 };
        nps[i]  = F4{ 0, float(i) + 0.2f, 0, 0 };
        bary[i] = F4{ 0, 0, float(i) + 0.3f, 0 };
        infos[i].pairInd0 = 0x1000ull + i; infos[i].pairInd1 = 100u + i; infos[i].rigidMatInd = i;
    }
    for (int t = 0; t < 2; ++t) { pts.push_back(F4{0,0,0,0}); nps.push_back(F4{0,0,0,0}); bary.push_back(F4{0,0,0,0}); }
    infos.resize(N + 2);
    const float POISONF = -777.0f;
    std::vector<F4> sPts(N + 2, F4{ POISONF, POISONF, POISONF, POISONF });
    std::vector<F4> sNps = sPts, sBary = sPts;
    std::vector<FemOtherContactInfoHost> sInfos(N + 2);
    std::memset(sInfos.data(), 0xDD, sInfos.size() * sizeof(FemOtherContactInfoHost));

    DevBuf bR = makeDev(dev, remap), bP = makeDev(dev, pts), bNp = makeDev(dev, nps), bB = makeDev(dev, bary);
    DevBuf bI = makeDev(dev, infos);
    DevBuf bSP = makeDev(dev, sPts), bSNp = makeDev(dev, sNps), bSB = makeDev(dev, sBary), bSI = makeDev(dev, sInfos);
    std::vector<uint32_t> cnt = { N, 0u, 0u, 0u };
    DevBuf bC = makeDev(dev, cnt);

    if (!bR.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    SbReorderPush push = {};
    push.contacts = bP.addr; push.normalPens = bNp.addr; push.barycentrics = bB.addr;
    push.contactInfos = bI.addr; push.numContacts = bC.addr; push.remapByRigid = bR.addr;
    push.sortedContacts = bSP.addr; push.sortedNormalPens = bSNp.addr;
    push.sortedBarycentrics = bSB.addr; push.sortedContactInfos = bSI.addr;

    Kernel k("softBodyContactRemapBda.spv", "sb_reorderPSContactsLaunch", {}, sizeof(SbReorderPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const auto gP = downloadBuffer<F4>(dev, bSP.h, N + 2);
    const auto gNpO = downloadBuffer<F4>(dev, bSNp.h, N + 2);
    const auto gB = downloadBuffer<F4>(dev, bSB.h, N + 2);
    const auto gI = downloadBuffer<FemOtherContactInfoHost>(dev, bSI.h, N + 2);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t r = remap[i];
        if (std::fabs(gP[i].x - pts[r].x) > 1e-6f) { ++mm; std::printf("  point[%u] MM: got %g want %g\n", i, gP[i].x, pts[r].x); }
        if (std::fabs(gNpO[i].y - nps[r].y) > 1e-6f) { ++mm; std::printf("  normalPen[%u] MM\n", i); }
        if (std::fabs(gB[i].z - bary[r].z) > 1e-6f) { ++mm; std::printf("  bary[%u] MM\n", i); }
        if (gI[i].pairInd0 != infos[r].pairInd0 || gI[i].pairInd1 != infos[r].pairInd1 ||
            gI[i].rigidMatInd != infos[r].rigidMatInd) { ++mm; std::printf("  info[%u] MM\n", i); }
    }
    PXG_CHECK_MSG(mm == 0, "the reorder must gather through remapByRigid (%zu mismatches)", mm);
    // an identity gather would be indistinguishable only if the permutation had fixed points
    PXG_CHECK_MSG(std::fabs(gP[0].x - pts[0].x) > 1e-6f,
                  "test is toothless: remap[0] must not be 0");
    // entries past the count keep their poison
    for (uint32_t i = N; i < N + 2u; ++i)
        PXG_CHECK_MSG(gP[i].x == POISONF && gB[i].x == POISONF,
                      "entry %u is past the contact count and must be untouched", i);

    dev->destroyBuffer(bR.h); dev->destroyBuffer(bP.h); dev->destroyBuffer(bNp.h);
    dev->destroyBuffer(bB.h); dev->destroyBuffer(bI.h); dev->destroyBuffer(bSP.h);
    dev->destroyBuffer(bSNp.h); dev->destroyBuffer(bSB.h); dev->destroyBuffer(bSI.h);
    dev->destroyBuffer(bC.h);
}

PXG_TEST(softBodySimPrimitiveContactGenPlane)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    const float cd = 0.05f;             // per-shape contact distance -> total 0.10
    const float totalCd = 2.0f * cd;
    const float restD = 0.02f;

    // One tet straddling y = 0: corners 0 and 1 are BELOW (penetrating), corner 2 is just
    // above but within the contact distance, corner 3 is far above and must be rejected.
    std::vector<F4> verts = {
        { 0.0f, -0.30f, 0.0f, 1.0f },   // corner 0 — deep
        { 1.0f, -0.10f, 0.0f, 1.0f },   // corner 1 — shallow
        { 0.0f,  0.05f, 1.0f, 1.0f },   // corner 2 — above but INSIDE the 0.10 band
        { 0.0f,  5.00f, 0.0f, 1.0f },   // corner 3 — far above, rejected
    };
    std::vector<TetIdx4> tets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };   // + trailer
    for (int t = 0; t < 2; ++t) verts.push_back(F4{ 0, 0, 0, 0 });             // + trailer
    DevBuf bVerts = makeDev(dev, verts);
    DevBuf bTets = makeDev(dev, tets);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
    sb.mNumTets = 1u;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // shape 0 = tetmesh (softbody 0), shape 1 = the PLANE, shape 2 a BOX.
    //
    // THE BOX CONTROL CHANGED WHEN THE HULL BRANCH LANDED, DELIBERATELY.  It used to be a
    // box with no hull at all, sitting on the plane's own transform: while the kernel ignored
    // every non-plane type that was a valid way to pin the boundary of a partial port, and
    // the comment here said the GJK tranche would have to change it consciously.  It does --
    // a bare box now walks its hull pointer and a null one faults the device, which is how
    // this fixture reported the change (SIGSEGV, not a wrong number).  The control keeps its
    // job -- pair 1 must contribute nothing -- but now earns it geometrically: a real cooked
    // cube, parked 100 units away on its own transform slot.  restDistance[0] is still the
    // value that must never be read.
    std::vector<F4> boxBlob;
    {
        const uint32_t hdr = (12u << 16) | (8u << 8) | 6u;   // nbEdges | nbVerts | nbPolygons
        auto asF = [](uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; };
        boxBlob.push_back(F4{ 0, 0, 0, 0 });                          // the skipped float4
        boxBlob.push_back(F4{ asF(hdr), asF(0u), asF(0u), asF(0u) });  // uint4 header
        boxBlob.push_back(F4{ 0.0f, 0.0f, 0.0f, 1.0f });               // extents; .w = inner sphere
        for (uint32_t m = 0; m < 8u; ++m)
            boxBlob.push_back(F4{ (m & 1u) ? 1.0f : -1.0f, (m & 2u) ? 1.0f : -1.0f,
                                  (m & 4u) ? 1.0f : -1.0f, 0.0f });
    }
    DevBuf bBoxBlob = makeDev(dev, boxBlob);

    std::vector<PxgShapeHost> shapes(3);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TETRAHEDRONMESH_H; shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = 1u /* ePLANE */;        shapes[1].materialIndex = 7u;
    shapes[2].type = 3u /* eBOX */;          shapes[2].materialIndex = 9u;
    shapes[2].sx = shapes[2].sy = shapes[2].sz = 0.25f;
    shapes[2].srw = 1.0f;
    shapes[2].hullLo = uint32_t(bBoxBlob.addr & 0xFFFFFFFFull);
    shapes[2].hullHi = uint32_t(bBoxBlob.addr >> 32);
    DevBuf bShapes = makeDev(dev, shapes);
    // The PLANE pair is cmIdx 1 ON PURPOSE: restDistance is indexed BY cmIdx, so putting
    // the observable pair at a nonzero index is the only way to tell restDistance[cmIdx]
    // from restDistance[0].  The mutation sweep read [0] and passed until this swap.
    // The box pair takes transform-cache slot 2, which is where the 100-unit offset lives.
    std::vector<CmInputHost> cms = { { 0u, 2u, 0u, 2u }, { 0u, 1u, 0u, 1u } };
    DevBuf bCms = makeDev(dev, cms);

    // A PxPlane's surface is x == 0 in plane space, so a WORLD y = 0 ground plane whose
    // normal is +y needs the quaternion that maps +x onto +y: a 90-degree rotation about z.
    const float s45 = 0.70710678f;
    std::vector<float> xforms = {
        0,0,0,1,  0,0,0,  0,                      // slot 0 (softbody, unused)
        0.0f, 0.0f, s45, s45,  0.0f, 0.0f, 0.0f,  0.0f,   // slot 1: plane, +x -> +y
        0.0f, 0.0f, 0.0f, 1.0f,  100.0f, 100.0f, 100.0f, 0.0f,  // slot 2: the box, far away
    };
    // TRAILER.  llvmpipe widens the final load of a tightly-sized buffer past its end and
    // faults; the softbody refit test hit the same thing.  Sixteen slack bytes on every
    // buffer whose last element is read is the standing workaround.
    for (int t = 0; t < 4; ++t) xforms.push_back(0.0f);
    DevBuf bXf = makeDev(dev, xforms);
    std::vector<float> cds = { cd, cd, cd, 0.0f };              // + trailer
    DevBuf bCd = makeDev(dev, cds);
    // DISTINCT per cmIdx: slot 0 belongs to the box pair and must never be read.
    std::vector<float> restDs = { 0.99f, restD, 0.0f, 0.0f };
    DevBuf bRest = makeDev(dev, restDs);
    // PxNodeIndex per transform-cache slot; slot 1 is the rigid's, slot 2 the far box's.
    std::vector<uint64_t> remap = { 0ull, 0x00000000ABCDEF01ull, 0x00000000BADF00Dull, 0ull };
    DevBuf bRemap = makeDev(dev, remap);

    // the midphase's output: one pair (cmIdx 0, tet 0)
    // two pairs: (cmIdx 0, tet 0) against the PLANE and (cmIdx 1, tet 0) against the BOX
    // (cmIdx 0, tet 0) against the BOX and (cmIdx 1, tet 0) against the PLANE
    std::vector<uint32_t> stack = { 0u, 0u, 0u, 0u,  1u, 0u, 0u, 0u,  0u, 0u, 0u, 0u };
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 2u };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> stackNeeded = { 0u };
    DevBuf bNeeded = makeDev(dev, stackNeeded);

    const uint32_t MAXC = 32u;
    const uint32_t POISON = 0xEEEEEEEEu;
    std::vector<F4> outPoint(MAXC, F4{ -1, -1, -1, -1 }), outNormal(MAXC, F4{ -1, -1, -1, -1 });
    std::vector<F4> outBary(MAXC, F4{ 0, 0, 0, 0 });
    std::vector<FemOtherContactInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(FemOtherContactInfoHost));
    std::vector<uint64_t> byX(MAXC, 0ull);
    std::vector<uint32_t> tempByX(MAXC, POISON), idxSorted(MAXC, POISON), sortedByX(MAXC, POISON);
    std::vector<uint32_t> total = { 0u };
    DevBuf bPt = makeDev(dev, outPoint), bNp = makeDev(dev, outNormal), bBary = makeDev(dev, outBary);
    DevBuf bInfo = makeDev(dev, outInfo), bByX = makeDev(dev, byX), bTmp = makeDev(dev, tempByX);
    DevBuf bIdx = makeDev(dev, idxSorted), bSorted = makeDev(dev, sortedByX), bTotal = makeDev(dev, total);

    if (!bVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    SbContactGenPush push = {};
    push.tolerenceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.restDistance = bRest.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.shapeToRigidRemapTable = bRemap.addr;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.stackSizeBytes = 32u;               // two uint4 slots
    push.softBodyRigidPairs = 0ull;
    push.numSoftBodyRigidPairs = 0u;         // no attachments -> the filter never matches
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.w.outPoint = bPt.addr; push.w.outNormalPen = bNp.addr; push.w.outBarycentric = bBary.addr;
    push.w.outContactInfo = bInfo.addr; push.w.totalContactCount = bTotal.addr;
    push.w.contactByX = bByX.addr; push.w.tempContactByX = bTmp.addr;
    push.w.contactIndexSortedByX = bIdx.addr; push.w.contactSortedByX = bSorted.addr;
    push.w.maxNumContacts = MAXC;

    Kernel k("softBodyContactGenBda.spv", "sb_primitiveContactGenLaunch", {}, sizeof(SbContactGenPush));
    k.dispatch(/*gridX*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    const auto gPt = downloadBuffer<F4>(dev, bPt.h, MAXC);
    const auto gNp = downloadBuffer<F4>(dev, bNp.h, MAXC);
    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, bInfo.h, MAXC);
    const auto gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];

    // Corners 0, 1 and 2 are all within contactDist (0.10) of the plane; corner 3 is not.
    // Three from the PLANE pair; the BOX pair must contribute NOTHING in this tranche.
    PXG_CHECK_MSG(n == 3u, "the plane path must emit ONE contact PER PENETRATING CORNER "
                           "(3 of the tet's 4 are within 0.10 of the plane) and the far BOX pair "
                           "must contribute nothing, got %u", n);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        PXG_CHECK_MSG(gInfo[i].rigidMatInd != 9u,
                      "contact %u came from the BOX rigid, which sits 100 units away -- the "
                      "hull branch must still respect the contact distance", i);

    bool seen[4] = { false, false, false, false };
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
    {
        std::printf("  contact %u: point(%.4f, %.4f, %.4f) rest=%.4f normal(%.3f, %.3f, %.3f) pen=%.4f\n",
                    i, gPt[i].x, gPt[i].y, gPt[i].z, gPt[i].w, gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w);
        // the contact point IS the tet corner, and the penetration is its signed height
        for (uint32_t c = 0; c < 4u; ++c)
            if (std::fabs(gPt[i].x - verts[c].x) < 1e-5f && std::fabs(gPt[i].y - verts[c].y) < 1e-5f &&
                std::fabs(gPt[i].z - verts[c].z) < 1e-5f)
            { seen[c] = true; PXG_CHECK_MSG(std::fabs(gNp[i].w - verts[c].y) < 1e-5f,
                  "penetration must be the corner's signed plane distance: corner %u y=%g, got pen %g",
                  c, verts[c].y, gNp[i].w); }
        // the plane normal is +y in world space
        PXG_CHECK_MSG(std::fabs(gNp[i].x) < 1e-5f && std::fabs(gNp[i].y - 1.0f) < 1e-5f &&
                      std::fabs(gNp[i].z) < 1e-5f,
                      "the world normal must be the plane's +y (got %.4f, %.4f, %.4f)",
                      gNp[i].x, gNp[i].y, gNp[i].z);
        PXG_CHECK_MSG(std::fabs(gPt[i].w - restD) < 1e-6f,
                      "point.w must carry the rest distance %g (got %g)", restD, gPt[i].w);
        // PxEncodeSoftBodyIndex(0, 0) == 0; the material index rides the low 31 bits.
        PXG_CHECK_MSG(gInfo[i].pairInd1 == 0u, "pairInd1 must be encode(softbody 0, tet 0) = 0, got %u",
                      gInfo[i].pairInd1);
        PXG_CHECK_MSG(gInfo[i].rigidMatInd == 7u,
                      "the rigid material index must be stored with bit31 cleared (got 0x%08x)",
                      gInfo[i].rigidMatInd);
        PXG_CHECK_MSG(gInfo[i].pairInd0 == 0x00000000ABCDEF01ull,
                      "pairInd0 must be the rigid's PxNodeIndex from the remap table");
    }
    PXG_CHECK_MSG(seen[0] && seen[1] && seen[2],
                  "corners 0, 1 and 2 must each produce a contact (%d %d %d)",
                  int(seen[0]), int(seen[1]), int(seen[2]));
    PXG_CHECK_MSG(!seen[3], "corner 3 is 5.0 above the plane and must NOT produce a contact");
    // the sort side-buffers must be primed for the radix sort that follows
    const auto gTmp = downloadBuffer<uint32_t>(dev, bTmp.h, MAXC);
    const auto gIdx = downloadBuffer<uint32_t>(dev, bIdx.h, MAXC);
    for (uint32_t i = 0; i < n; ++i)
    {
        PXG_CHECK_MSG(gTmp[i] == 0xABCDEF01u, "tempContactByX must be the low half of the rigid id");
        PXG_CHECK_MSG(gIdx[i] == i, "contactIndexSortedByX must be seeded with the identity");
    }
    // stackSizeNeededOnDevice reports what the midphase actually found, in BYTES.
    PXG_CHECK_MSG(gNeeded == 32u, "stackSizeNeeded must be numFound * sizeof(uint4) = 32, got %u", gNeeded);

    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bTets.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bBoxBlob.h);
    dev->destroyBuffer(bCms.h); dev->destroyBuffer(bXf.h);
    dev->destroyBuffer(bCd.h); dev->destroyBuffer(bRest.h); dev->destroyBuffer(bRemap.h);
    dev->destroyBuffer(bStack.h); dev->destroyBuffer(bPairNum.h); dev->destroyBuffer(bNeeded.h);
    dev->destroyBuffer(bPt.h); dev->destroyBuffer(bNp.h); dev->destroyBuffer(bBary.h);
    dev->destroyBuffer(bInfo.h); dev->destroyBuffer(bByX.h); dev->destroyBuffer(bTmp.h);
    dev->destroyBuffer(bIdx.h); dev->destroyBuffer(bSorted.h); dev->destroyBuffer(bTotal.h);
}


// ===========================================================================
// P8.0 tranche 7 — the three kernels that CLOSE the softbody step:
//   sb_gm_cp_averageVertsLaunch        (softBodyGM.cu:1509) — merge the partition copies
//   sb_gm_applyDeformationDeltasLaunch (softBodyGM.cu:2665) — the Jacobi delta apply
//   sb_gm_updateTetModelVertsLaunch    (softBodyGM.cu:1589) — skin sim -> collision mesh
//
// NO tolerance seam here: not one transcendental between the three.  Everything is
// compared through raw float bits (f2u), and each case carries teeth aimed at the one
// thing a plausible-looking port would get wrong.
// ===========================================================================

// { PxgSoftBody*, const PxU32*, PxReal } -> 24-byte blob; identical to ApplyDeltasPush,
// but declared separately so the static_asserts document each kernel's own pack.
struct AvgVertsPush { uint64_t gSoftbodies, activeSoftbodies; float invDt; uint32_t _pad20; };
static_assert(sizeof(AvgVertsPush) == 24, "averageVerts push blob must be 24 bytes");
static_assert(offsetof(AvgVertsPush, invDt) == 16, "invDt @16");

// { PxgSoftBody*, const PxU32* } -> 16-byte blob.
struct SkinPush { uint64_t gSoftbodies, activeSoftbodies; };
static_assert(sizeof(SkinPush) == 16, "updateTetModelVerts push blob must be 16 bytes");

// ---------------------------------------------------------------------------
// sb_gm_cp_averageVertsLaunch.
//
// The partition solve writes each vertex once PER COPY into mSimPosition_InvMassCP,
// past a leading block of numElements * numVertsPerElement entries; this kernel folds
// them back by averaging the displacements.  The case pins, in order:
//   - the OFFSET into the copy buffer (the leading element-corner block is poisoned
//     with values that would produce a wildly different average if read by mistake);
//   - the inclusive-prefix-sum walk of mSimAccumulatedCopiesCP, with a per-vertex copy
//     count that varies 1/2/3/1/2/1 so vertex 0's implicit start-at-0 and the general
//     accum[i-1] case are both live;
//   - the 1/numCopies average (a port that summed without dividing matches only the
//     single-copy vertices);
//   - the pos_invMass.w > 0 gate (a pinned vertex must come back untouched);
//   - the w channel of the velocity write, which carries INVERSE MASS and not a
//     velocity component.
// `kinematic` re-runs the whole thing with PxDeformableBodyFlag::eKINEMATIC set, where
// every buffer must survive completely untouched.
// ---------------------------------------------------------------------------
static void runAverageVerts(bool kinematic)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t numVerts = 6u;
    const uint32_t numTets = 4u;          // numTetsPerElement == 1 -> numElements == 4
    const uint32_t numVertsPerElement = 4u;
    const uint32_t offset = numTets * numVertsPerElement;   // 16
    const float invDt = 60.0f;

    // copies per vertex: 1, 2, 3, 1, 2, 1  -> inclusive prefix sum
    const uint32_t copies[6] = { 1u, 2u, 3u, 1u, 2u, 1u };
    std::vector<uint32_t> accum(numVerts);
    { uint32_t run = 0u; for (uint32_t i = 0; i < numVerts; ++i) { run += copies[i]; accum[i] = run; } }
    const uint32_t totalCopies = accum[numVerts - 1u];   // 10

    std::vector<F4> pos(numVerts), vel(numVerts), deltaPos(numVerts);
    for (uint32_t i = 0; i < numVerts; ++i)
    {
        // vertex 4 is PINNED (invMass == 0) and must be skipped entirely
        const float invMass = (i == 4u) ? 0.0f : (0.5f + 0.25f * float(i));
        pos[i] = { 0.75f * float(i), 3.0f - 0.1f * float(i), -0.3f * float(i), invMass };
        // the velocity's w channel is deliberately NEVER equal to pos.w — the kernel must
        // restore it from pos_invMass, and the difference is what gives that check teeth
        vel[i] = { 0.11f * float(i) + 0.03f, -0.19f * float(i) - 0.07f, 0.13f * float(i) + 0.05f,
                   0.625f + 0.125f * float(i) };
        deltaPos[i] = { 0.101f * float(i), -0.203f * float(i), 0.307f * float(i), 77.0f };
    }

    // The copy buffer: the leading `offset` entries belong to the element corners and
    // must NEVER be read by this kernel — poison them far away from the real copies.
    std::vector<F4> cp(offset + totalCopies);
    for (uint32_t j = 0; j < offset; ++j)
        cp[j] = { -1000.0f - float(j), 1000.0f + float(j), -2000.0f, 999.0f };
    for (uint32_t i = 0; i < numVerts; ++i)
    {
        const uint32_t start = (i == 0u) ? 0u : accum[i - 1u];
        for (uint32_t c = 0; c < copies[i]; ++c)
        {
            // each copy sits at pos + a distinct displacement, so the average is a
            // different number for every copy count
            const float s = 0.01f * float(c + 1u) + 0.003f * float(i);
            cp[offset + start + c] = { pos[i].x + s, pos[i].y - 2.0f * s, pos[i].z + 3.0f * s,
                                       pos[i].w + 0.5f };
        }
    }

    DevBuf bPos = makeDev(dev, pos), bVel = makeDev(dev, vel), bDeltaPos = makeDev(dev, deltaPos);
    DevBuf bCp = makeDev(dev, cp), bAccum = makeDev(dev, accum);

    PxgSoftBody c = makeZeroDesc();
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDeltaPos.addr));
    c.mSimPosition_InvMassCP = reinterpret_cast<float4*>(size_t(bCp.addr));
    c.mSimAccumulatedCopiesCP = reinterpret_cast<PxU32*>(size_t(bAccum.addr));
    c.mNumVertsGM = numVerts;
    c.mNumTetsGM = numTets;
    c.mNumTetsPerElement = 1u;
    if (kinematic)
        c.mBodyFlags = PxU8(PxDeformableBodyFlag::eKINEMATIC);
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    AvgVertsPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.invDt = invDt;

    Kernel k("softBodySolveBda.spv", "sb_gm_cp_averageVertsLaunch", {}, sizeof(AvgVertsPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference — expression order transcribed from softBodyGM.cu:1509.
    std::vector<F4> refPos = pos, refVel = vel, refDeltaPos = deltaPos;
    if (!kinematic)
    {
        for (uint32_t i = 0; i < numVerts; ++i)
        {
            if (!(pos[i].w > 0.0f)) continue;
            const uint32_t start = (i == 0u) ? 0u : accum[i - 1u];
            const uint32_t end = accum[i];
            const F4 p = pos[i];
            F4 diff = { cp[offset + start].x - p.x, cp[offset + start].y - p.y,
                        cp[offset + start].z - p.z, cp[offset + start].w - p.w };
            for (uint32_t j = start + 1u; j < end; ++j)
            {
                diff.x += cp[offset + j].x - p.x; diff.y += cp[offset + j].y - p.y;
                diff.z += cp[offset + j].z - p.z; diff.w += cp[offset + j].w - p.w;
            }
            const float scale = 1.0f / float(end - start);
            diff.w = 0.0f;
            const F4 avgDelta = { diff.x * scale, diff.y * scale, diff.z * scale, diff.w * scale };
            refPos[i] = { p.x + avgDelta.x, p.y + avgDelta.y, p.z + avgDelta.z, p.w + avgDelta.w };
            // NOTE: w comes from pos_invMass, not from the averaged velocity
            refVel[i] = { vel[i].x + avgDelta.x * invDt, vel[i].y + avgDelta.y * invDt,
                          vel[i].z + avgDelta.z * invDt, p.w };
            refDeltaPos[i] = { deltaPos[i].x + avgDelta.x, deltaPos[i].y + avgDelta.y,
                               deltaPos[i].z + avgDelta.z, deltaPos[i].w + avgDelta.w };
        }
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, numVerts);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, numVerts);
    const auto gDeltaPos = downloadBuffer<F4>(dev, bDeltaPos.h, numVerts);
    const auto gCp = downloadBuffer<F4>(dev, bCp.h, cp.size());

    size_t mm = 0;
    auto cmp = [&](const char* what, uint32_t i, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) ||
            f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", what, i,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); } };
    for (uint32_t i = 0; i < numVerts; ++i)
    {
        cmp("pos", i, refPos[i], gPos[i]);
        cmp("vel", i, refVel[i], gVel[i]);
        cmp("deltaPos", i, refDeltaPos[i], gDeltaPos[i]);
    }
    // The copy buffer is read-only for this kernel — including the poisoned prefix.
    for (uint32_t j = 0; j < uint32_t(cp.size()); ++j)
        cmp("copyBuffer(readonly)", j, cp[j], gCp[j]);

    PXG_CHECK_MSG(mm == 0, "sb_gm_cp_averageVertsLaunch bit-exact vs CPU (%zu mismatches)", mm);

    if (kinematic)
    {
        // Teeth on the eKINEMATIC early-out: NOTHING may move.  Vertex 2 has three copies
        // all displaced in the same direction, so any port that ran the body would show it.
        PXG_CHECK_MSG(f2u(gPos[2].x) == f2u(pos[2].x),
                      "eKINEMATIC must skip the average entirely: pos[2].x %g -> %g",
                      pos[2].x, gPos[2].x);
    }
    else
    {
        // Teeth 1 — the DIVIDE.  Vertex 2 has three copies displaced by 0.016/0.026/0.036
        // in x, so the average is 0.026; a port that summed without dividing lands on
        // 0.078 and a port that used the wrong copy count lands elsewhere again.
        const float s0 = 0.01f * 1.0f + 0.003f * 2.0f, s1 = 0.01f * 2.0f + 0.003f * 2.0f,
                    s2 = 0.01f * 3.0f + 0.003f * 2.0f;
        const float expect2x = pos[2].x + ((s0 + s1 + s2) * (1.0f / 3.0f));
        PXG_CHECK_MSG(std::fabs(gPos[2].x - expect2x) < 1e-6f,
                      "vertex 2 (3 copies) must be averaged by 1/3: ref %g got %g",
                      expect2x, gPos[2].x);
        // Teeth 2 — the pos_invMass.w > 0 gate: vertex 4 is pinned and owns two copies
        // that would move it if the gate were dropped.
        PXG_CHECK_MSG(f2u(gPos[4].x) == f2u(pos[4].x) && f2u(gVel[4].x) == f2u(vel[4].x),
                      "pinned vertex 4 (invMass == 0) must be untouched: pos %g vel %g",
                      gPos[4].x, gVel[4].x);
        // Teeth 3 — the velocity w channel is the INVERSE MASS, restored from
        // pos_invMass and NOT the averaged vel.w (which differs for every vertex here).
        PXG_CHECK_MSG(f2u(gVel[0].w) == f2u(pos[0].w),
                      "velocity.w must carry pos_invMass.w (%g), got %g", pos[0].w, gVel[0].w);
        PXG_CHECK_MSG(f2u(gVel[0].w) != f2u(vel[0].w),
                      "test is toothless: vel[0].w must differ from pos[0].w");
        // Teeth 4 — the OFFSET.  Vertex 0's single copy sits at cp[16]; reading cp[0]
        // (the poisoned element-corner block) instead would drag it to about -1000.
        PXG_CHECK_MSG(gPos[0].x > 0.0f - 1.0f,
                      "vertex 0 must read the copy at cp[offset], not cp[0] (got %g)", gPos[0].x);
    }

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDeltaPos.h);
    dev->destroyBuffer(bCp.h); dev->destroyBuffer(bAccum.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimAverageVerts) { runAverageVerts(/*kinematic*/ false); }
PXG_TEST(softBodySimAverageVertsKinematic) { runAverageVerts(/*kinematic*/ true); }

// ---------------------------------------------------------------------------
// sb_gm_applyDeformationDeltasLaunch.
//
// The Jacobi twin of sb_gm_applyExternalDeltasLaunch, differing in exactly two things,
// and the case is built so a port that got either wrong FAILS:
//   - the vertex INDIRECTION through mSimJacobiVertIndices (the indices are a permuted,
//     non-contiguous subset, and every vertex NOT in the list must survive untouched —
//     a port that used the lane index directly would move the wrong four);
//   - mJacobiScale (@428) instead of 1/max(delta.w, 1).  delta.w is set to 4 on one of
//     the vertices, so the two rules give visibly different answers (0.375 vs 0.25).
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimApplyDeformationDeltas)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8u;             // sim vertices
    const uint32_t nbJacobi = 4u;      // the Jacobi subset
    const float invDt = 60.0f;
    const float jacobiScale = 0.375f;

    // Deliberately permuted and non-contiguous: lane i does NOT work on vertex i.
    const std::vector<uint32_t> jacobiIdx = { 5u, 1u, 7u, 2u };

    std::vector<F4> pos(N), vel(N), delta(N), deltaPos(N);
    const float ws[8] = { 0.0f, 1.0f, 4.0f, 7.0f, 2.5f, 0.5f, 3.0f, 12.0f };
    for (uint32_t i = 0; i < N; ++i)
    {
        pos[i] = { float(i) * 0.75f, 5.0f - float(i) * 0.1f, -0.3f * float(i), 1.0f + 0.25f * float(i) };
        vel[i] = { 0.11f * float(i) + 0.03f, -0.19f * float(i) - 0.07f, 0.13f * float(i) + 0.05f,
                   0.5f + 0.125f * float(i) };
        delta[i] = { 0.031f * float(i) + 0.007f, -0.017f * float(i) - 0.011f,
                     0.023f * float(i) + 0.013f, ws[i] };
        deltaPos[i] = { 0.101f * float(i), -0.203f * float(i), 0.307f * float(i), 77.0f };
    }

    DevBuf bPos = makeDev(dev, pos), bVel = makeDev(dev, vel);
    DevBuf bDelta = makeDev(dev, delta), bDeltaPos = makeDev(dev, deltaPos);
    DevBuf bIdx = makeDev(dev, jacobiIdx);

    PxgSoftBody c = makeZeroDesc();
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mSimDelta = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDeltaPos.addr));
    c.mSimJacobiVertIndices = reinterpret_cast<PxU32*>(size_t(bIdx.addr));
    c.mNumJacobiVertices = nbJacobi;
    c.mNumVertsGM = N;
    c.mJacobiScale = jacobiScale;
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    AvgVertsPush push = {};   // same {8,8,4} pack
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.invDt = invDt;

    Kernel k("softBodySolveBda.spv", "sb_gm_applyDeformationDeltasLaunch", {}, sizeof(AvgVertsPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference — expression order transcribed from softBodyGM.cu:2665.
    std::vector<F4> refPos = pos, refVel = vel, refDelta = delta, refDeltaPos = deltaPos;
    for (uint32_t lane = 0; lane < nbJacobi; ++lane)
    {
        const uint32_t v = jacobiIdx[lane];
        F4 d = delta[v];
        d.x *= jacobiScale; d.y *= jacobiScale; d.z *= jacobiScale;
        d.w = 0.0f;
        refVel[v] = { vel[v].x + d.x * invDt, vel[v].y + d.y * invDt,
                      vel[v].z + d.z * invDt, vel[v].w + d.w * invDt };
        refPos[v] = { pos[v].x + d.x, pos[v].y + d.y, pos[v].z + d.z, pos[v].w + d.w };
        refDeltaPos[v] = { deltaPos[v].x + d.x, deltaPos[v].y + d.y,
                           deltaPos[v].z + d.z, deltaPos[v].w + d.w };
        refDelta[v] = { 0.0f, 0.0f, 0.0f, 0.0f };
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);
    const auto gDeltaPos = downloadBuffer<F4>(dev, bDeltaPos.h, N);

    size_t mm = 0;
    auto cmp = [&](const char* what, uint32_t i, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) ||
            f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", what, i,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); } };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp("pos", i, refPos[i], gPos[i]);
        cmp("vel", i, refVel[i], gVel[i]);
        cmp("delta", i, refDelta[i], gDelta[i]);
        cmp("deltaPos", i, refDeltaPos[i], gDeltaPos[i]);
    }
    PXG_CHECK_MSG(mm == 0, "sb_gm_applyDeformationDeltasLaunch bit-exact vs CPU (%zu mismatches)", mm);

    // Teeth 1 — mJacobiScale, not 1/max(delta.w,1).  Vertex 2 carries delta.w == 4, so
    // the external-delta rule would scale by 0.25 while this kernel must scale by 0.375.
    const float expectJacobi = pos[2].x + delta[2].x * jacobiScale;
    const float expectExternal = pos[2].x + delta[2].x * 0.25f;
    PXG_CHECK_MSG(f2u(gPos[2].x) == f2u(expectJacobi),
                  "vertex 2 must use mJacobiScale (%g), got %g", expectJacobi, gPos[2].x);
    PXG_CHECK_MSG(f2u(expectJacobi) != f2u(expectExternal),
                  "test is toothless: the two scale rules must disagree on vertex 2");
    // Teeth 2 — the INDIRECTION.  Vertices 0, 3, 4 and 6 are not in the Jacobi subset and
    // must be bit-identical to their inputs, deltas included (NOT cleared).
    const uint32_t untouched[4] = { 0u, 3u, 4u, 6u };
    for (uint32_t t = 0; t < 4u; ++t)
    {
        const uint32_t v = untouched[t];
        PXG_CHECK_MSG(f2u(gPos[v].x) == f2u(pos[v].x),
                      "vertex %u is outside the Jacobi subset and must not move (%g -> %g)",
                      v, pos[v].x, gPos[v].x);
        PXG_CHECK_MSG(f2u(gDelta[v].x) == f2u(delta[v].x),
                      "vertex %u is outside the Jacobi subset and its delta must NOT be cleared", v);
    }
    // ...and the four that ARE in it did move and were cleared.
    for (uint32_t lane = 0; lane < nbJacobi; ++lane)
    {
        const uint32_t v = jacobiIdx[lane];
        PXG_CHECK_MSG(f2u(gPos[v].x) != f2u(pos[v].x),
                      "test is toothless: Jacobi vertex %u must actually move", v);
        PXG_CHECK_MSG(f2u(gDelta[v].x) == f2u(0.0f), "Jacobi vertex %u's delta must be cleared", v);
    }

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bDeltaPos.h); dev->destroyBuffer(bIdx.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// sb_gm_updateTetModelVertsLaunch.
//
// The skinning step: each COLLISION vertex is the barycentric combination of its host
// SIMULATION tetrahedron's four corners.  The case pins:
//   - mNumVerts (@364, the COLLISION count) as the guard, deliberately DIFFERENT from
//     mNumVertsGM (@372) — a port that used the sim count would over- or under-run;
//   - the two-level indirection remapTable -> mSimTetIndices -> mSimPosition_InvMass,
//     with the four tet corner indices permuted so x/y/z/w cannot be swapped silently;
//   - the four-term blend p0*b.x + p1*b.y + p2*b.z + p3*b.w including the w channel;
//   - that only the first mNumVerts entries of mPosition_InvMass are written.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimUpdateTetModelVerts)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t numCollVerts = 5u;
    const uint32_t numSimVerts = 9u;
    const uint32_t numSimTets = 3u;

    // Sim tetrahedra with PERMUTED corner indices — x/y/z/w are all distinct vertices and
    // no tet uses them in ascending order, so a transposed blend shows up immediately.
    std::vector<p80refit::TetIdx4> simTets = {
        { 3u, 0u, 7u, 1u },
        { 5u, 8u, 2u, 4u },
        { 6u, 1u, 8u, 0u },
    };
    // Which sim tet hosts each collision vertex (two share tet 2 -> the map is not 1:1).
    std::vector<uint32_t> remap = { 1u, 0u, 2u, 2u, 0u };
    std::vector<F4> bary(numCollVerts);
    for (uint32_t i = 0; i < numCollVerts; ++i)
    {
        // deliberately NOT a partition of unity on every vertex: the kernel does not
        // normalise, so an implementation that did would diverge here
        bary[i] = { 0.5f - 0.07f * float(i), 0.2f + 0.05f * float(i),
                    0.2f - 0.03f * float(i), 0.1f + 0.11f * float(i) };
    }

    std::vector<F4> simPos(numSimVerts);
    for (uint32_t i = 0; i < numSimVerts; ++i)
        simPos[i] = { 1.5f * float(i) - 2.0f, 0.25f * float(i) + 0.5f,
                      -0.6f * float(i) + 1.0f, 0.125f * float(i) + 0.75f };

    // One extra entry past mNumVerts, poisoned: it must survive.
    const F4 POISON = { -12345.0f, -12345.0f, -12345.0f, -12345.0f };
    std::vector<F4> collPos(numCollVerts + 1u, POISON);

    DevBuf bTets = makeDev(dev, simTets), bRemap = makeDev(dev, remap), bBary = makeDev(dev, bary);
    DevBuf bSimPos = makeDev(dev, simPos), bCollPos = makeDev(dev, collPos);

    PxgSoftBody c = makeZeroDesc();
    c.mSimTetIndices = reinterpret_cast<decltype(c.mSimTetIndices)>(size_t(bTets.addr));
    c.mVertsRemapInGridModel = reinterpret_cast<PxU32*>(size_t(bRemap.addr));
    c.mVertsBarycentricInGridModel = reinterpret_cast<float4*>(size_t(bBary.addr));
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSimPos.addr));
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bCollPos.addr));
    c.mNumVerts = numCollVerts;      // the guard
    c.mNumVertsGM = numSimVerts;     // deliberately different
    c.mNumTetsGM = numSimTets;
    c.mNumTetsPerElement = 1u;
    std::vector<PxgSoftBody> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    SkinPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;

    Kernel k("softBodySolveBda.spv", "sb_gm_updateTetModelVertsLaunch", {}, sizeof(SkinPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference — expression order transcribed from softBodyGM.cu:1589.
    std::vector<F4> ref = collPos;
    for (uint32_t i = 0; i < numCollVerts; ++i)
    {
        const p80refit::TetIdx4 t = simTets[remap[i]];
        const F4 p0 = simPos[t.x], p1 = simPos[t.y], p2 = simPos[t.z], p3 = simPos[t.w];
        const F4 b = bary[i];
        ref[i] = { p0.x * b.x + p1.x * b.y + p2.x * b.z + p3.x * b.w,
                   p0.y * b.x + p1.y * b.y + p2.y * b.z + p3.y * b.w,
                   p0.z * b.x + p1.z * b.y + p2.z * b.z + p3.z * b.w,
                   p0.w * b.x + p1.w * b.y + p2.w * b.z + p3.w * b.w };
    }

    const auto got = downloadBuffer<F4>(dev, bCollPos.h, collPos.size());
    const auto gSimPos = downloadBuffer<F4>(dev, bSimPos.h, numSimVerts);

    size_t mm = 0;
    for (uint32_t i = 0; i < uint32_t(collPos.size()); ++i)
        if (f2u(ref[i].x) != f2u(got[i].x) || f2u(ref[i].y) != f2u(got[i].y) ||
            f2u(ref[i].z) != f2u(got[i].z) || f2u(ref[i].w) != f2u(got[i].w))
        { ++mm; std::printf("  collPos[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", i,
                            ref[i].x, ref[i].y, ref[i].z, ref[i].w,
                            got[i].x, got[i].y, got[i].z, got[i].w); }
    PXG_CHECK_MSG(mm == 0, "sb_gm_updateTetModelVertsLaunch bit-exact vs CPU (%zu mismatches)", mm);

    // Teeth 1 — the guard is mNumVerts, not mNumVertsGM: entry 5 must keep its poison.
    PXG_CHECK_MSG(f2u(got[numCollVerts].x) == f2u(POISON.x),
                  "entry past mNumVerts must be untouched (got %g)", got[numCollVerts].x);
    // Teeth 2 — the sim mesh is READ-ONLY here.
    for (uint32_t i = 0; i < numSimVerts; ++i)
        PXG_CHECK_MSG(f2u(gSimPos[i].x) == f2u(simPos[i].x),
                      "the simulation mesh must not be written (vertex %u)", i);
    // Teeth 3 — the CORNER ORDER.  Collision vertices 0 and 4 share barycentrics-shaped
    // input but different tets; more to the point, blending tet 1 with its corners sorted
    // ascending (2,4,5,8) instead of as stored (5,8,2,4) gives a different answer.
    {
        const F4 b = bary[0];
        const F4 q0 = simPos[2], q1 = simPos[4], q2 = simPos[5], q3 = simPos[8];
        const float sorted = q0.x * b.x + q1.x * b.y + q2.x * b.z + q3.x * b.w;
        PXG_CHECK_MSG(f2u(got[0].x) != f2u(sorted),
                      "test is toothless: sorted-corner blend must differ from the stored order");
    }
    // Teeth 4 — every collision vertex must actually have been written.
    for (uint32_t i = 0; i < numCollVerts; ++i)
        PXG_CHECK_MSG(f2u(got[i].x) != f2u(POISON.x), "collision vertex %u was never written", i);

    dev->destroyBuffer(bTets.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bBary.h);
    dev->destroyBuffer(bSimPos.h); dev->destroyBuffer(bCollPos.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ===========================================================================
// P8.2 tranche d — sb_rigidContactPrepareLaunch (softBody.cu:773), chain kernel 5.
//
// Turns a generated softbody-vs-rigid contact into a solver constraint: the response terms
// scaled by the rigid body's sqrt inverse inertia, the error, and the penetration-bias
// clamp, written into the block-of-32 PxgFemRigidConstraintBlock that kernels 6 and 7 read.
//
// The CPU reference below is a second, independent transcription of prepareFEMContacts
// (deformableUtils.cuh:132) -- testFemClothSim.cpp carries one too, for the cloth kernel.
// That duplication is deliberate here and is a CROSS-CHECK rather than a risk: since
// tranche d the shader-side function is SHARED (pxgSlangFemRigidContact), so the two
// fixtures verify one implementation from two independently written references. A mistake
// in either reference shows up as a failure, not as two tests agreeing on a wrong answer.
// (The references are not shared because each test file carries its own F4 / f2u
// vocabulary; unifying that is a separate cleanup with no bearing on shipped code.)
// ===========================================================================
struct SbRigidConstraintHost
{
    F4 raXn_resp[32];          // @0
    F4 raXnF0_resp[32];        // @512
    F4 raXnF1_resp[32];        // @1024
    F4 normal_errorW[32];      // @1536
    F4 fricTan0_invMass0[32];  // @2048
    F4 barycentric[32];        // @2560
    float maxPenBias[32];      // @3072 -> 3200
};
static_assert(sizeof(SbRigidConstraintHost) == 3200, "PxgFemRigidConstraintBlock sizeof == 3200");
static_assert(offsetof(SbRigidConstraintHost, normal_errorW) == 1536, "normal_errorW @1536");
static_assert(offsetof(SbRigidConstraintHost, maxPenBias) == 3072, "maxPenBias @3072");

struct SbPrepPush
{
    uint64_t softbodies, contacts, normalPens, barycentrics, contactInfos, numContacts;
    uint64_t constraints, prePrepDesc, prepDesc, rigidAppliedForces;
    float invDt; uint32_t pad84;
    uint64_t sharedDesc;
    uint32_t isTGS, pad100;
};
static_assert(sizeof(SbPrepPush) == 104, "sb prepare push blob must be 104 bytes");
static_assert(offsetof(SbPrepPush, invDt) == 80, "invDt @80");
static_assert(offsetof(SbPrepPush, sharedDesc) == 88, "sharedDesc @88");
static_assert(offsetof(SbPrepPush, isTGS) == 96, "isTGS @96");

namespace p82dref
{
struct V3 { float x, y, z; };
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 mat33Mul(const float m[9], V3 v)
{
    return V3{ m[0] * v.x + m[3] * v.y + m[6] * v.z,
               m[1] * v.x + m[4] * v.y + m[7] * v.z,
               m[2] * v.x + m[5] * v.y + m[8] * v.z };
}
// PxComputeBasisVectors (PxMathUtils.h:297).
static void computeBasis(V3 dir, V3& right, V3& up)
{
    if (std::fabs(dir.y) <= 0.9999f)
    {
        right = V3{ dir.z, 0.0f, -dir.x };
        const float m = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (m > 0.0f) right = V3{ right.x / m, right.y / m, right.z / m };
        up = V3{ dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x };
    }
    else
    {
        right = V3{ 1.0f, 0.0f, 0.0f };
        up = V3{ 0.0f, dir.z, -dir.y };
        const float m = std::sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
        if (m > 0.0f) up = V3{ up.x / m, up.y / m, up.z / m };
    }
}
// prepareFEMContacts (deformableUtils.cuh:132) — rigid / static (non-articulation) path.
static void refPrepare(SbRigidConstraintHost& c, int slot, V3 normal, V3 p, float pen, V3 delta,
                       F4 barycentric, V3 bodyFrameP, float invMass0, float penBiasClampRigid,
                       const float sqrtInvInertia[9], float penBiasClampFEM, bool isStatic)
{
    V3 t0, t1;
    computeBasis(normal, t0, t1);
    const bool isKinematic = (invMass0 == 0.0f) && (!isStatic);

    const V3 ra = V3{ p.x - bodyFrameP.x, p.y - bodyFrameP.y, p.z - bodyFrameP.z };
    const V3 raXn = cross3(ra, normal);
    const V3 raXF0 = cross3(ra, t0);
    const V3 raXF1 = cross3(ra, t1);

    const V3 raXnSI = mat33Mul(sqrtInvInertia, raXn);
    const float resp0 = dot3(raXnSI, raXnSI) + invMass0;
    const V3 raXF0SI = mat33Mul(sqrtInvInertia, raXF0);
    const V3 raXF1SI = mat33Mul(sqrtInvInertia, raXF1);
    const float respF0 = dot3(raXF0SI, raXF0SI) + invMass0;
    const float respF1 = dot3(raXF1SI, raXF1SI) + invMass0;

    if (isKinematic)
    {
        c.raXn_resp[slot] = F4{ raXn.x, raXn.y, raXn.z, resp0 };
        c.raXnF0_resp[slot] = F4{ raXF0.x, raXF0.y, raXF0.z, respF0 };
        c.raXnF1_resp[slot] = F4{ raXF1.x, raXF1.y, raXF1.z, respF1 };
    }
    else
    {
        c.raXn_resp[slot] = F4{ raXnSI.x, raXnSI.y, raXnSI.z, resp0 };
        c.raXnF0_resp[slot] = F4{ raXF0SI.x, raXF0SI.y, raXF0SI.z, respF0 };
        c.raXnF1_resp[slot] = F4{ raXF1SI.x, raXF1SI.y, raXF1SI.z, respF1 };
    }
    c.normal_errorW[slot] = F4{ normal.x, normal.y, normal.z, pen + dot3(delta, normal) };
    c.fricTan0_invMass0[slot] = F4{ t0.x, t0.y, t0.z, invMass0 };
    c.maxPenBias[slot] = (penBiasClampRigid > penBiasClampFEM) ? penBiasClampRigid : penBiasClampFEM;
    c.barycentric[slot] = barycentric;
}
// FEMCollision::initialize + computeFEMChange (deformableUtils.cuh:280 / :488), PGS path,
// STATIC rigid at rest, null-material friction.  Returns the per-contact position delta.
// Written so the absolute magnitude of the solve is checkable, not just its shape: a
// ratio-only assertion cannot see a dropped mass-splitting factor, because that scales
// every vertex identically.
struct SolveRef { float deltaLambdaN; V3 deltaPos; };
static SolveRef refSolvePGS(const SbRigidConstraintHost& c, int slot, float appliedForceRef,
                            float dt, V3 deformableLinDelta, float massSplit,
                            float rigidRefCount)
{
    const float invDt = 1.0f / dt;
    const F4 raXn_resp = c.raXn_resp[slot];
    const F4 normal_biasW = c.normal_errorW[slot];
    const V3 normal = V3{ normal_biasW.x, normal_biasW.y, normal_biasW.z };
    const float initPen = normal_biasW.w;
    const float maxPenBias = c.maxPenBias[slot];

    // static rigid at rest: linVel = angVel = 0, so normalVel = 0 and angDelta = 0.
    const V3 relLinDelta = V3{ -deformableLinDelta.x, -deformableLinDelta.y, -deformableLinDelta.z };
    const float error = (initPen + dot3(relLinDelta, normal)) * invDt;
    const float errorBiased = (maxPenBias > error) ? maxPenBias : error;
    const float CN = errorBiased;

    SolveRef r;
    r.deltaLambdaN = 0.0f;
    r.deltaPos = V3{ 0, 0, 0 };
    if (!(CN < 0.0f)) return r;   // inactive unless wasActive, which the caller handles

    const float unitResponse = rigidRefCount * raXn_resp.w + massSplit;
    const float invDenom = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;
    float dLN = -CN * invDenom;
    if (dLN < -appliedForceRef) dLN = -appliedForceRef;
    r.deltaLambdaN = dLN;
    // deltaLambdaT collapses to 0 on the null-material path (combinedFriction == 0).
    r.deltaPos = V3{ -(dLN * normal.x) * dt, -(dLN * normal.y) * dt, -(dLN * normal.z) * dt };
    return r;
}


// PxQuat::rotate (PxQuat.h:286) — needed by the attachment prepare reference.
static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
static V3 vecNormalized(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f) { const float rs = 1.0f / std::sqrt(m); return V3{ v.x * rs, v.y * rs, v.z * rs }; }
    return V3{ 0, 0, 0 };
}
// getSoftBodyInvMass (softBody.cuh:38) — the TET attachment's inverse mass.
static float softBodyInvMass(float baryMass, F4 bary)
{
    return baryMass * std::sqrt(bary.x * bary.x + bary.y * bary.y + bary.z * bary.z + bary.w * bary.w);
}

// The block-of-32 PxgFEMRigidAttachmentConstraint (PxgFEMCore.h:191).
struct AttachConstraintHost
{
    F4 baryOrType[32];        // @0
    F4 raXn0_biasW[32];       // @512
    F4 raXn1_biasW[32];       // @1024
    F4 raXn2_biasW[32];       // @1536
    F4 velMult_invMass[32];   // @2048
    F4 low_high_limits[32];   // @2560
    F4 axis_angle[32];        // @3072
    uint32_t elemId[32];      // @3584
    uint64_t rigidId[32];     // @3712 -> 3968
};

// sb_rigidAttachmentPrepareLaunch (softBodyGM.cu:1822), rigid / static path.
static void refAttachPrepare(AttachConstraintHost& c, int slot, F4 bodyQ, V3 bodyP, V3 ra4,
                             V3 point, float invMass1, const float sqrtInvInertia[9],
                             float invMass0, bool isStatic, F4 axisAngle)
{
    const V3 normal0{ 1, 0, 0 }, normal1{ 0, 1, 0 }, normal2{ 0, 0, 1 };
    float inertia[9];
    float inertiaScale = 1.0f;
    if (invMass0 == 0.0f && !isStatic)
    {
        for (int k = 0; k < 9; ++k) inertia[k] = (k % 4 == 0) ? 1.0f : 0.0f;   // identity
        inertiaScale = 0.0f;
    }
    else
        for (int k = 0; k < 9; ++k) inertia[k] = sqrtInvInertia[k];

    const V3 ra = quatRotate(bodyQ, ra4);
    const V3 error{ ra.x + bodyP.x - point.x, ra.y + bodyP.y - point.y, ra.z + bodyP.z - point.z };
    const V3 worldAxis = vecNormalized(quatRotate(bodyQ, V3{ axisAngle.x, axisAngle.y, axisAngle.z }));

    const V3 raXn0 = cross3(ra, normal0), raXn1 = cross3(ra, normal1), raXn2 = cross3(ra, normal2);
    const V3 s0 = mat33Mul(inertia, raXn0), s1 = mat33Mul(inertia, raXn1), s2 = mat33Mul(inertia, raXn2);
    const float resp0 = dot3(s0, s0) * inertiaScale + invMass0 + invMass1;
    const float resp1 = dot3(s1, s1) * inertiaScale + invMass0 + invMass1;
    const float resp2 = dot3(s2, s2) * inertiaScale + invMass0 + invMass1;

    c.raXn0_biasW[slot] = F4{ s0.x, s0.y, s0.z, dot3(error, normal0) };
    c.raXn1_biasW[slot] = F4{ s1.x, s1.y, s1.z, dot3(error, normal1) };
    c.raXn2_biasW[slot] = F4{ s2.x, s2.y, s2.z, dot3(error, normal2) };
    c.velMult_invMass[slot] = F4{ (resp0 > 0.f) ? 1.f / resp0 : 0.f,
                                  (resp1 > 0.f) ? 1.f / resp1 : 0.f,
                                  (resp2 > 0.f) ? 1.f / resp2 : 0.f, invMass0 };
    c.axis_angle[slot] = F4{ worldAxis.x, worldAxis.y, worldAxis.z, axisAngle.w };
}

// sb_solveRigidSoftAttachmentLaunch (softBodyGM.cu:2085), PGS, STATIC anchor at rest.
// Returns the position delta the solve applies BEFORE the per-vertex inverse-mass and
// barycentric scaling.  Needed for an ABSOLUTE check: signs and ratios alone cannot see a
// dropped negation (it flips every run alike) nor a changed bias factor (it scales
// everything alike).
static V3 refAttachSolvePGS(const AttachConstraintHost& c, int slot, V3 linVel1, float dt)
{
    const float invDt = 1.0f / dt;
    const F4 raXn0_biasW = c.raXn0_biasW[slot];
    const F4 velMult = c.velMult_invMass[slot];
    // A static anchor at rest contributes no rigid velocity.
    // NOTE the upstream quirk: all THREE components use raXn0_biasW.w, because CUDA passes
    // constraint.raXn0_biasW three times -- the Y and Z position errors are never read.
    const float positionError = raXn0_biasW.w;
    const float biasFactor = 0.5f;
    const V3 velError{ linVel1.x - positionError * biasFactor * invDt,
                       linVel1.y - positionError * biasFactor * invDt,
                       linVel1.z - positionError * biasFactor * invDt };
    const V3 impulse{ velError.x * velMult.x, velError.y * velMult.y, velError.z * velMult.z };
    return V3{ -impulse.x * dt, -impulse.y * dt, -impulse.z * dt };
}

} // namespace p82dref

struct V3ish { float x, y, z; };
static void sbPokePtr(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }

PXG_TEST(softBodySimRigidContactPrepare)
{
    using namespace p80refit;   // F4 / TetIdx4 / makeDev / makeZeroDesc vocabulary
    PxgSlangComputeDevice* dev = testDevice();

    // --- one sim tet, with a DISTINCT delta per vertex ---------------------
    // The four deltas differ in every component so the FOUR-weight barycentric projection
    // is observable: dropping the w term, or reusing the triangle projection, lands on a
    // different `delta` and therefore a different error in normal_errorW.w.
    std::vector<F4> simDeltaPos = {
        { 0.0100f, -0.0200f,  0.0300f, 0.5f },
        { 0.0700f,  0.0110f, -0.0250f, 0.5f },
        { -0.0400f, 0.0620f,  0.0170f, 0.5f },
        { 0.0230f, -0.0510f,  0.0880f, 0.5f },
        { 9.0f, 9.0f, 9.0f, 9.0f },            // trailer: must never be read
    };
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };
    DevBuf bDelta = makeDev(dev, simDeltaPos), bTets = makeDev(dev, simTets);
    if (!bDelta.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // The two clamps must DIFFER, or max() and min() agree and the choice is untestable.
    const float penBiasClampFEM = -1.0e10f;
    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bTets.addr));
    sb.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    sb.mPenBiasClamp = penBiasClampFEM;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // --- two contacts: one real, one naming the INVALID tet ---------------
    // The second exists to pin that the appliedForces slot is cleared for EVERY in-range
    // contact BEFORE the tet guard, and that the guard is `continue` (the thread keeps
    // striding) rather than `return`.
    float rawN[3] = { 0.2f, 0.9f, 0.3f };
    const float nmag = std::sqrt(rawN[0] * rawN[0] + rawN[1] * rawN[1] + rawN[2] * rawN[2]);
    const p82dref::V3 planeNormal = { rawN[0] / nmag, rawN[1] / nmag, rawN[2] / nmag };
    const float restW = 0.0125f;    // NON-zero: pen = normal_pen.w - rest must be observable
    const float penW = -0.0200f;
    const F4 bary = { 0.4f, 0.2f, 0.3f, 0.1f };   // all four weights nonzero and distinct

    // A THIRD contact at index 33, so the block/offset split has teeth: 33 lands in block 1
    // slot 1.  With everything at index 0 both `workIndex / 32` and `workIndex & 31` are 0,
    // and flattening either one is invisible.  Contacts 1..32 are skipped (invalid tet), so
    // only 0 and 33 write.
    const uint32_t kSecond = 33u;
    const uint32_t kN = kSecond + 1u;
    const F4 bary2 = { 0.1f, 0.5f, 0.15f, 0.25f };
    const float restW2 = 0.0075f, penW2 = -0.031f;
    float rawN2[3] = { -0.4f, 0.8f, 0.45f };
    const float nmag2 = std::sqrt(rawN2[0] * rawN2[0] + rawN2[1] * rawN2[1] + rawN2[2] * rawN2[2]);
    const p82dref::V3 planeNormal2 = { rawN2[0] / nmag2, rawN2[1] / nmag2, rawN2[2] / nmag2 };

    std::vector<F4> contacts(kN, F4{ 1.0f, 1.0f, 1.0f, 0.0f });
    std::vector<F4> normalPens(kN, F4{ 0.0f, 1.0f, 0.0f, -0.1f });
    std::vector<F4> barycentrics(kN, F4{ 1.0f, 0.0f, 0.0f, 0.0f });
    contacts[0] = F4{ 0.5f, -0.02f, 0.7f, restW };
    normalPens[0] = F4{ planeNormal.x, planeNormal.y, planeNormal.z, penW };
    barycentrics[0] = bary;
    contacts[kSecond] = F4{ -0.3f, 0.11f, 0.45f, restW2 };
    normalPens[kSecond] = F4{ planeNormal2.x, planeNormal2.y, planeNormal2.z, penW2 };
    barycentrics[kSecond] = bary2;

    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    std::vector<FemOtherContactInfoHost> infos(kN);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemOtherContactInfoHost));
    for (uint32_t i = 0; i < kN; ++i)
    {
        infos[i].pairInd0 = staticNodeInd;
        infos[i].pairInd1 = 0x000fffffu;   // softbody 0, tet 0xfffff -> SKIPPED
    }
    infos[0].pairInd1 = 0u;            // softbody 0, sim tet 0
    infos[kSecond].pairInd1 = 0u;      // softbody 0, sim tet 0

    DevBuf bContacts = makeDev(dev, contacts), bNormalPens = makeDev(dev, normalPens);
    DevBuf bBary = makeDev(dev, barycentrics), bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { kN };
    DevBuf bNum = makeDev(dev, numContacts);

    // TWO blocks: contact 33 writes block 1, so a single block would be an out-of-bounds
    // write and the block-index check below would be testing nothing.
    std::vector<uint8_t> constraintBytes(2u * sizeof(SbRigidConstraintHost), 0xCD);
    DevBuf bConstraint = makeDev(dev, constraintBytes);
    // seeded NON-zero so the unconditional clear is observable for every in-range contact
    std::vector<float> appliedForces(kN + 1u, 0.0f);
    for (uint32_t i = 0; i < kN; ++i) appliedForces[i] = 100.0f + float(i);
    appliedForces[kN] = 789.0f;   // past the count: must survive
    DevBuf bForces = makeDev(dev, appliedForces);

    // --- static-body frame / mass / inertia (solver body 0) ---------------
    std::vector<F4> body2World = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
    DevBuf bBody2World = makeDev(dev, body2World);
    const float penBiasClampRigid = -5.0f;   // > penBiasClampFEM, so max() picks THIS one
    std::vector<uint8_t> solverBodyData(96, 0);
    std::memcpy(solverBodyData.data() + 12, &penBiasClampRigid, 4);   // penBiasClamp @12
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    // a NON-zero, non-symmetric-looking sqrtInvInertia so the response rows have teeth.
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);    // sqrtInvInertia @28
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);
    std::vector<uint32_t> solverBodyIndices = { 0u };   // never dereferenced for a static body
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);

    std::vector<uint8_t> prepDesc(304, 0);
    sbPokePtr(prepDesc, 64, bSolverBodyData.addr);      // solverBodyDataPool @64
    sbPokePtr(prepDesc, 72, bSolverTxIData.addr);       // solverBodyTxIDataPool @72
    sbPokePtr(prepDesc, 168, bBody2World.addr);         // body2WorldPool @168
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    sbPokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);   // solverBodyIndices @272
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);

    SbPrepPush pp = {};
    pp.softbodies = bDesc.addr; pp.contacts = bContacts.addr; pp.normalPens = bNormalPens.addr;
    pp.barycentrics = bBary.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.constraints = bConstraint.addr; pp.prePrepDesc = bPrePrepDesc.addr;
    pp.prepDesc = bPrepDesc.addr; pp.rigidAppliedForces = bForces.addr;
    pp.invDt = 60.0f;            // DEFERRED: unread, as in CUDA
    pp.sharedDesc = 0ull;        // DEFERRED: articulation-only, must not be dereferenced
    pp.isTGS = 0u;

    Kernel("softBodyRigidContactBda.spv", "sb_rigidContactPrepareLaunch", {}, sizeof(SbPrepPush))
        .dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    // ---- CPU reference -------------------------------------------------
    // the FOUR-weight projection of mSimDeltaPos through the tet's vertices
    const p82dref::V3 delta = {
        simDeltaPos[0].x * bary.x + simDeltaPos[1].x * bary.y + simDeltaPos[2].x * bary.z + simDeltaPos[3].x * bary.w,
        simDeltaPos[0].y * bary.x + simDeltaPos[1].y * bary.y + simDeltaPos[2].y * bary.z + simDeltaPos[3].y * bary.w,
        simDeltaPos[0].z * bary.x + simDeltaPos[1].z * bary.y + simDeltaPos[2].z * bary.z + simDeltaPos[3].z * bary.w,
    };
    SbRigidConstraintHost refC;
    std::memset(&refC, 0xCD, sizeof(refC));
    const p82dref::V3 solverNormal = { -planeNormal.x, -planeNormal.y, -planeNormal.z };
    const float pen = penW - restW;
    p82dref::refPrepare(refC, 0, solverNormal, p82dref::V3{ contacts[0].x, contacts[0].y, contacts[0].z },
                        pen, delta, bary, p82dref::V3{ 0, 0, 0 }, /*invMass0*/ 0.0f,
                        penBiasClampRigid, sqrtInvInertia, penBiasClampFEM, /*isStatic*/ true);

    // the second contact's reference, in block 1 slot 1
    const p82dref::V3 delta2 = {
        simDeltaPos[0].x * bary2.x + simDeltaPos[1].x * bary2.y + simDeltaPos[2].x * bary2.z + simDeltaPos[3].x * bary2.w,
        simDeltaPos[0].y * bary2.x + simDeltaPos[1].y * bary2.y + simDeltaPos[2].y * bary2.z + simDeltaPos[3].y * bary2.w,
        simDeltaPos[0].z * bary2.x + simDeltaPos[1].z * bary2.y + simDeltaPos[2].z * bary2.z + simDeltaPos[3].z * bary2.w,
    };
    SbRigidConstraintHost refC2;
    std::memset(&refC2, 0xCD, sizeof(refC2));
    p82dref::refPrepare(refC2, int(kSecond & 31u),
                        p82dref::V3{ -planeNormal2.x, -planeNormal2.y, -planeNormal2.z },
                        p82dref::V3{ contacts[kSecond].x, contacts[kSecond].y, contacts[kSecond].z },
                        penW2 - restW2, delta2, bary2, p82dref::V3{ 0, 0, 0 }, 0.0f,
                        penBiasClampRigid, sqrtInvInertia, penBiasClampFEM, true);

    const auto gBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, 2u * sizeof(SbRigidConstraintHost));
    SbRigidConstraintHost gotC, gotC1;
    std::memcpy(&gotC, gBytes.data(), sizeof(gotC));
    std::memcpy(&gotC1, gBytes.data() + sizeof(gotC), sizeof(gotC1));
    const auto gForces = downloadBuffer<float>(dev, bForces.h, kN + 1u);

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  prep %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmp4("raXn_resp", refC.raXn_resp[0], gotC.raXn_resp[0]);
    cmp4("raXnF0_resp", refC.raXnF0_resp[0], gotC.raXnF0_resp[0]);
    cmp4("raXnF1_resp", refC.raXnF1_resp[0], gotC.raXnF1_resp[0]);
    cmp4("normal_errorW", refC.normal_errorW[0], gotC.normal_errorW[0]);
    cmp4("fricTan0_invMass0", refC.fricTan0_invMass0[0], gotC.fricTan0_invMass0[0]);
    cmp4("barycentric", refC.barycentric[0], gotC.barycentric[0]);
    if (f2u(refC.maxPenBias[0]) != f2u(gotC.maxPenBias[0]))
    { ++mm; std::printf("  prep maxPenBias MM: ref %g got %g\n", refC.maxPenBias[0], gotC.maxPenBias[0]); }
    PXG_CHECK_MSG(mm == 0, "sb_rigidContactPrepareLaunch bit-exact vs CPU (%zu mismatches)", mm);

    // teeth: the sqrtInvInertia matrix must actually be exercised
    PXG_CHECK_MSG(f2u(gotC.raXn_resp[0].x) != 0u || f2u(gotC.raXn_resp[0].y) != 0u ||
                  f2u(gotC.raXn_resp[0].z) != 0u,
                  "the prepared raXn response must be non-zero (sqrtInvInertia teeth)");
    // the NEGATION: the stored normal is the negated contact normal, not the raw one.
    PXG_CHECK_MSG(f2u(gotC.normal_errorW[0].x) == f2u(-planeNormal.x),
                  "the constraint normal must be the NEGATED contact normal (got %g, raw %g)",
                  gotC.normal_errorW[0].x, planeNormal.x);
    // pen - rest: with restW nonzero, forgetting the subtraction shifts the error by restW.
    PXG_CHECK_MSG(f2u(gotC.normal_errorW[0].w) != f2u(penW + p82dref::dot3(delta, solverNormal)),
                  "the error must use pen - restDistance, not the raw penetration");

    // the pen-bias clamp is the MAX of the two, and they differ, so min() lands elsewhere
    PXG_CHECK_MSG(f2u(gotC.maxPenBias[0]) == f2u(penBiasClampRigid),
                  "maxPenBias must be max(rigid %g, FEM %g) = %g, got %g",
                  penBiasClampRigid, penBiasClampFEM, penBiasClampRigid, gotC.maxPenBias[0]);

    // ---- the block / offset split: contact 33 belongs in BLOCK 1, SLOT 1 ----------
    size_t mm2 = 0;
    auto cmp4b = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm2; std::printf("  prep[33] %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm,
                             r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    const int sl = int(kSecond & 31u);
    cmp4b("raXn_resp", refC2.raXn_resp[sl], gotC1.raXn_resp[sl]);
    cmp4b("normal_errorW", refC2.normal_errorW[sl], gotC1.normal_errorW[sl]);
    cmp4b("fricTan0_invMass0", refC2.fricTan0_invMass0[sl], gotC1.fricTan0_invMass0[sl]);
    cmp4b("barycentric", refC2.barycentric[sl], gotC1.barycentric[sl]);
    PXG_CHECK_MSG(mm2 == 0, "contact 33 must land in constraint BLOCK 1 SLOT 1, bit-exact "
                            "(%zu mismatches) -- a flattened block index or a zeroed offset "
                            "writes somewhere else", mm2);
    // and slot 0 of block 1 must still hold the sentinel: nothing wrote there
    PXG_CHECK_MSG(f2u(gotC1.barycentric[0].x) == f2u(gotC1.raXn_resp[0].x),
                  "block 1 slot 0 must be untouched (only contact 33 writes block 1)");

    // contacts 1..32 named the invalid tet: SKIPPED, so block 0 slot 1 keeps the sentinel...
    PXG_CHECK_MSG(f2u(gotC.normal_errorW[1].x) == f2u(gotC.barycentric[1].x),
                  "the skipped contact must not have written its constraint slot");
    // ...but every in-range applied-force slot was still cleared, because the clear precedes
    // the guard.
    for (uint32_t i = 0; i < kN; ++i)
        PXG_CHECK_MSG(f2u(gForces[i]) == 0u,
                      "applied forces must be cleared for EVERY in-range contact, including "
                      "the ones the tet guard skips (slot %u = %g)", i, gForces[i]);
    PXG_CHECK_MSG(f2u(gForces[kN]) == f2u(789.0f),
                  "the slot past the contact count must be untouched (got %g)", gForces[kN]);

    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bTets.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bNormalPens.h); dev->destroyBuffer(bBary.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bConstraint.h);
    dev->destroyBuffer(bForces.h); dev->destroyBuffer(bBody2World.h);
    dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bSolverBodyIndices.h); dev->destroyBuffer(bPrepDesc.h);
    dev->destroyBuffer(bPrePrepDesc.h);
}

// ---------------------------------------------------------------------------
// The invalid-tet guard is `continue`, not `return` -- and why that needs its own fixture.
//
// In softBodySimRigidContactPrepare above, every thread handles exactly ONE contact, so
// `continue` and `return` are indistinguishable: there is nothing after the skip for the
// thread to do.  Mutation-tested, replacing continue with return survives that fixture
// untouched.  The difference only appears once a thread handles TWO contacts, i.e. once
//   nbIterationsPerBlock = ceil(ceil(N / 256) / 1024) > 1,
// which needs N > 262144.  A real scene reaches that easily (maxDeformableVolumeContacts
// defaults to 1M); a unit fixture cannot allocate for it.
//
// It does not have to.  Only the COUNT has to be large -- the buffers only have to cover the
// indices actually touched.  With gridX = 1 the workIndex is `i * 256 + gtid`, so the single
// dispatched group touches 0..511 and nothing else, whatever N says.  So: declare 262145
// contacts, allocate 512, put the INVALID tet at index 0 and a VALID contact at index 256.
// Thread 0 meets the guard on its first iteration and the real contact on its second.
// With `continue` the constraint at 256 is written; with `return` it is silently dropped,
// which in a large scene means contacts vanish behind an invalid one.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimRigidContactPrepareStride)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<F4> simDeltaPos = {
        { 0.0100f, -0.0200f,  0.0300f, 0.5f }, {  0.0700f, 0.0110f, -0.0250f, 0.5f },
        { -0.0400f, 0.0620f,  0.0170f, 0.5f }, {  0.0230f, -0.0510f, 0.0880f, 0.5f },
    };
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };
    DevBuf bDelta = makeDev(dev, simDeltaPos), bTets = makeDev(dev, simTets);
    if (!bDelta.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const float penBiasClampFEM = -1.0e10f;
    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bTets.addr));
    sb.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    sb.mPenBiasClamp = penBiasClampFEM;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    const uint32_t kTouched = 512u;      // what gridX = 1 actually reaches
    const uint32_t kDeclared = 262145u;  // > 256 * 1024 -> nbIterationsPerBlock == 2
    const uint32_t kHit = 256u;          // thread 0's SECOND iteration

    float rawN[3] = { 0.2f, 0.9f, 0.3f };
    const float nmag = std::sqrt(rawN[0] * rawN[0] + rawN[1] * rawN[1] + rawN[2] * rawN[2]);
    const p82dref::V3 planeNormal = { rawN[0] / nmag, rawN[1] / nmag, rawN[2] / nmag };
    const float restW = 0.0125f, penW = -0.0200f;
    const F4 bary = { 0.4f, 0.2f, 0.3f, 0.1f };

    std::vector<F4> contacts(kTouched, F4{ 1.0f, 1.0f, 1.0f, 0.0f });
    std::vector<F4> normalPens(kTouched, F4{ 0.0f, 1.0f, 0.0f, -0.1f });
    std::vector<F4> barycentrics(kTouched, F4{ 1.0f, 0.0f, 0.0f, 0.0f });
    contacts[kHit] = F4{ 0.5f, -0.02f, 0.7f, restW };
    normalPens[kHit] = F4{ planeNormal.x, planeNormal.y, planeNormal.z, penW };
    barycentrics[kHit] = bary;

    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    std::vector<FemOtherContactInfoHost> infos(kTouched);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemOtherContactInfoHost));
    for (uint32_t i = 0; i < kTouched; ++i)
    {
        infos[i].pairInd0 = staticNodeInd;
        infos[i].pairInd1 = 0x000fffffu;   // invalid tet -> skipped
    }
    infos[kHit].pairInd1 = 0u;             // the ONE real contact, on iteration 2 of thread 0

    DevBuf bContacts = makeDev(dev, contacts), bNormalPens = makeDev(dev, normalPens);
    DevBuf bBary = makeDev(dev, barycentrics), bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { kDeclared };
    DevBuf bNum = makeDev(dev, numContacts);

    const uint32_t kBlocks = kTouched / 32u;
    std::vector<uint8_t> constraintBytes(kBlocks * sizeof(SbRigidConstraintHost), 0xCD);
    DevBuf bConstraint = makeDev(dev, constraintBytes);
    std::vector<float> appliedForces(kTouched, 55.0f);
    DevBuf bForces = makeDev(dev, appliedForces);

    std::vector<F4> body2World = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
    DevBuf bBody2World = makeDev(dev, body2World);
    const float penBiasClampRigid = -5.0f;
    std::vector<uint8_t> solverBodyData(96, 0);
    std::memcpy(solverBodyData.data() + 12, &penBiasClampRigid, 4);
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);
    std::vector<uint32_t> solverBodyIndices = { 0u };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);

    std::vector<uint8_t> prepDesc(304, 0);
    sbPokePtr(prepDesc, 64, bSolverBodyData.addr);
    sbPokePtr(prepDesc, 72, bSolverTxIData.addr);
    sbPokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    sbPokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);

    SbPrepPush pp = {};
    pp.softbodies = bDesc.addr; pp.contacts = bContacts.addr; pp.normalPens = bNormalPens.addr;
    pp.barycentrics = bBary.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.constraints = bConstraint.addr; pp.prePrepDesc = bPrePrepDesc.addr;
    pp.prepDesc = bPrepDesc.addr; pp.rigidAppliedForces = bForces.addr;
    pp.invDt = 60.0f; pp.sharedDesc = 0ull; pp.isTGS = 0u;

    Kernel("softBodyRigidContactBda.spv", "sb_rigidContactPrepareLaunch", {}, sizeof(SbPrepPush))
        .dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    const p82dref::V3 delta = {
        simDeltaPos[0].x * bary.x + simDeltaPos[1].x * bary.y + simDeltaPos[2].x * bary.z + simDeltaPos[3].x * bary.w,
        simDeltaPos[0].y * bary.x + simDeltaPos[1].y * bary.y + simDeltaPos[2].y * bary.z + simDeltaPos[3].y * bary.w,
        simDeltaPos[0].z * bary.x + simDeltaPos[1].z * bary.y + simDeltaPos[2].z * bary.z + simDeltaPos[3].z * bary.w,
    };
    SbRigidConstraintHost refC;
    std::memset(&refC, 0xCD, sizeof(refC));
    p82dref::refPrepare(refC, int(kHit & 31u),
                        p82dref::V3{ -planeNormal.x, -planeNormal.y, -planeNormal.z },
                        p82dref::V3{ contacts[kHit].x, contacts[kHit].y, contacts[kHit].z },
                        penW - restW, delta, bary, p82dref::V3{ 0, 0, 0 }, 0.0f,
                        penBiasClampRigid, sqrtInvInertia, penBiasClampFEM, true);

    const auto gBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, kBlocks * sizeof(SbRigidConstraintHost));
    SbRigidConstraintHost gotBlk;
    std::memcpy(&gotBlk, gBytes.data() + (kHit / 32u) * sizeof(SbRigidConstraintHost), sizeof(gotBlk));
    const int sl = int(kHit & 31u);

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  stride %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmp4("raXn_resp", refC.raXn_resp[sl], gotBlk.raXn_resp[sl]);
    cmp4("normal_errorW", refC.normal_errorW[sl], gotBlk.normal_errorW[sl]);
    cmp4("barycentric", refC.barycentric[sl], gotBlk.barycentric[sl]);
    PXG_CHECK_MSG(mm == 0,
                  "the contact at index %u is thread 0's SECOND iteration, reached only "
                  "because the invalid-tet guard is `continue` and not `return` -- it must "
                  "be prepared bit-exactly (%zu mismatches)", kHit, mm);

    const auto gForces = downloadBuffer<float>(dev, bForces.h, kTouched);
    PXG_CHECK_MSG(f2u(gForces[kHit]) == 0u,
                  "the second iteration's applied-force slot must be cleared (got %g)", gForces[kHit]);
    PXG_CHECK_MSG(f2u(gForces[0]) == 0u, "the first iteration's slot must be cleared too");

    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bTets.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bNormalPens.h); dev->destroyBuffer(bBary.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bConstraint.h);
    dev->destroyBuffer(bForces.h); dev->destroyBuffer(bBody2World.h);
    dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bSolverBodyIndices.h); dev->destroyBuffer(bPrepDesc.h);
    dev->destroyBuffer(bPrePrepDesc.h);
}

// ===========================================================================
// P8.2 tranche e — the query (reference-count) and solve kernels, at UNIT level.
//
// The PXG_SCENE=softbody settle proves the chain runs end to end, but it is a coarse
// instrument: it only sees the aggregate.  Mutation-tested against it, three real defects
// survive, and each is blind for a structural reason the scene cannot fix:
//
//   - dropping the mass-splitting multiply.  With one softbody and one contact per vertex
//     every reference count is 1, and multiplying by 1 is a no-op.  Needs a vertex shared
//     by TWO contacts, which is what this fixture builds.
//   - dropping the applied-force write-back.  The accumulated lambda only bites through
//     the `max(-CN*invDenom, -accumulated)` clamp on a LATER solver iteration, so a
//     single-dispatch scene never reads it back.
//   - ignoring wasActive.  Contacts in the scene stay geometrically active, so the
//     conservative "once active, stay active" path is never load-bearing there.
//
// The fixture drives query -> solve directly on synthetic data, with TWO contacts on the
// SAME tetrahedron so every vertex is referenced twice.
// ===========================================================================
struct SbQueryPush
{
    uint64_t softbodies, contactInfos, constraints, numContacts, prePrepDesc, solverCoreDesc;
    uint64_t artiCoreDesc, sharedDesc;
    float dt; uint32_t pad68;
    uint64_t appliedForces, rigidBodyReferenceCounts;
};
static_assert(sizeof(SbQueryPush) == 88, "sb query push blob must be 88 bytes");
static_assert(offsetof(SbQueryPush, dt) == 64, "dt @64");
static_assert(offsetof(SbQueryPush, rigidBodyReferenceCounts) == 80, "refCounts @80");

struct SbSolvePush
{
    uint64_t softbodies, contactInfos, constraints, numContacts, prePrepDesc, solverCoreDesc;
    uint64_t artiCoreDesc, sharedDesc, rigidDeltaVel, appliedForces, rigidBodyReferenceCounts;
    float dt; uint32_t pad92;
    uint64_t materials, rigidBodyMaterials;
};
static_assert(sizeof(SbSolvePush) == 112, "sb solve push blob must be 112 bytes");
static_assert(offsetof(SbSolvePush, rigidDeltaVel) == 64, "rigidDeltaVel @64");
static_assert(offsetof(SbSolvePush, dt) == 88, "dt @88");
static_assert(offsetof(SbSolvePush, materials) == 96, "materials @96");

PXG_TEST(softBodySimRigidContactQuerySolve)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;

    // ONE tet, four movable vertices.  mSimDeltaPos.w carries the inverse mass; mSimDelta
    // is the accumulator (xyz = position delta, w = reference count).
    const float invMass = 0.5f;
    std::vector<F4> simDeltaPos = {
        { 0.0010f, -0.0020f, 0.0015f, invMass }, { 0.0007f, 0.0011f, -0.0025f, invMass },
        { -0.0004f, 0.0062f, 0.0017f, invMass }, { 0.0023f, -0.0051f, 0.0088f, invMass },
    };
    std::vector<F4> simDelta(4, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };
    DevBuf bSimDeltaPos = makeDev(dev, simDeltaPos), bSimDelta = makeDev(dev, simDelta);
    DevBuf bTets = makeDev(dev, simTets);
    if (!bSimDeltaPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const float penBiasClampFEM = -1.0e10f;
    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bTets.addr));
    sb.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bSimDeltaPos.addr));
    sb.mSimDelta = reinterpret_cast<float4*>(size_t(bSimDelta.addr));
    sb.mPenBiasClamp = penBiasClampFEM;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // TWO contacts, both on tet 0 -> every vertex is referenced twice, so the
    // mass-splitting multiply by the reference count is observable.
    const uint32_t kN = 2u;
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    std::vector<FemOtherContactInfoHost> infos(kN);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemOtherContactInfoHost));
    for (uint32_t i = 0; i < kN; ++i) { infos[i].pairInd0 = staticNodeInd; infos[i].pairInd1 = 0u; }
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { kN };
    DevBuf bNum = makeDev(dev, numContacts);

    // A prepared constraint block: a penetrating contact whose normal points +Y.
    SbRigidConstraintHost blk;
    std::memset(&blk, 0, sizeof(blk));
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    const float penBiasClampRigid = -5.0f;
    const F4 bary = { 0.4f, 0.2f, 0.3f, 0.1f };
    for (uint32_t i = 0; i < kN; ++i)
        // The SOLVER normal, which prepare produces by NEGATING the contact normal.  A
        // floor whose contact normal is +Y therefore gives -Y here, and
        // deltaPos = -(lambda * normal) * dt then pushes the vertices UP.  Passing +Y
        // would model a ceiling and the deltas would legitimately point down.
        p82dref::refPrepare(blk, int(i), p82dref::V3{ 0.0f, -1.0f, 0.0f },
                            p82dref::V3{ 0.5f, -0.02f, 0.7f }, -0.02f,
                            p82dref::V3{ 0.001f, -0.002f, 0.0015f }, bary,
                            p82dref::V3{ 0, 0, 0 }, 0.0f, penBiasClampRigid,
                            sqrtInvInertia, penBiasClampFEM, true);
    std::vector<uint8_t> constraintBytes(sizeof(SbRigidConstraintHost));
    std::memcpy(constraintBytes.data(), &blk, sizeof(blk));
    DevBuf bConstraint = makeDev(dev, constraintBytes);

    std::vector<float> appliedForces(kN, 0.0f);
    DevBuf bForces = makeDev(dev, appliedForces);
    std::vector<uint32_t> refCounts = { 0u, 0u };
    DevBuf bRefCounts = makeDev(dev, refCounts);
    std::vector<F4> rigidDeltaVel(2u * kN, F4{ 9.f, 9.f, 9.f, 9.f });
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);

    // solver-body scaffolding: a single static body at rest.
    std::vector<F4> body2World = { { 0, 0, 0, 1 }, { 0, 0, 0, 0 } };
    DevBuf bBody2World = makeDev(dev, body2World);
    std::vector<uint8_t> solverBodyData(96, 0);
    std::memcpy(solverBodyData.data() + 12, &penBiasClampRigid, 4);
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);
    std::vector<uint32_t> solverBodyIndices = { 0u };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);
    std::vector<F4> outSolverVelocity(4, F4{ 0, 0, 0, 0 });
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);
    std::vector<F4> solverBodyVelPool(4, F4{ 0, 0, 0, 0 });
    DevBuf bVelPool = makeDev(dev, solverBodyVelPool);

    std::vector<uint8_t> prepDesc(304, 0);
    sbPokePtr(prepDesc, 64, bSolverBodyData.addr);
    sbPokePtr(prepDesc, 72, bSolverTxIData.addr);
    sbPokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    sbPokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);
    std::vector<uint8_t> solverCoreDesc(320, 0);
    sbPokePtr(solverCoreDesc, 0, bOutSolverVel.addr);          // outSolverVelocity @0
    { const uint32_t one = 1u; std::memcpy(solverCoreDesc.data() + 268, &one, 4); }
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);
    std::vector<uint8_t> sharedDesc(224, 0);
    sbPokePtr(sharedDesc, 160, bVelPool.addr);                 // iter_solverBodyVelPool @160
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    // ======================= QUERY =======================
    SbQueryPush qp = {};
    qp.softbodies = bDesc.addr; qp.contactInfos = bInfos.addr; qp.constraints = bConstraint.addr;
    qp.numContacts = bNum.addr; qp.prePrepDesc = bPrePrepDesc.addr;
    qp.solverCoreDesc = bSolverCoreDesc.addr; qp.artiCoreDesc = 0ull;
    qp.sharedDesc = bSharedDesc.addr; qp.dt = dt;
    qp.appliedForces = bForces.addr; qp.rigidBodyReferenceCounts = bRefCounts.addr;
    Kernel("softBodyRigidContactBda.spv", "sb_queryRigidSoftContactReferenceCountLaunch", {}, sizeof(SbQueryPush))
        .dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    const auto gDeltaAfterQuery = downloadBuffer<F4>(dev, bSimDelta.h, 4);
    const auto gInfoAfterQuery = downloadBuffer<FemOtherContactInfoHost>(dev, bInfos.h, kN);

    // Every barycentric weight is > 1e-3 and every inverse mass non-zero, so BOTH contacts
    // bump ALL FOUR vertices: each count must be exactly 2.  That is what makes the
    // mass-splitting multiply below observable at all.
    for (uint32_t v = 0; v < 4u; ++v)
        PXG_CHECK_MSG(f2u(gDeltaAfterQuery[v].w) == f2u(2.0f),
                      "vertex %u must be referenced by BOTH contacts (count %g, expected 2)",
                      v, gDeltaAfterQuery[v].w);
    for (uint32_t i = 0; i < kN; ++i)
        PXG_CHECK_MSG((gInfoAfterQuery[i].rigidMatInd & 0x80000000u) != 0u,
                      "contact %u penetrates, so the query must mark it in-collision", i);

    // ======================= SOLVE =======================
    SbSolvePush sp = {};
    sp.softbodies = bDesc.addr; sp.contactInfos = bInfos.addr; sp.constraints = bConstraint.addr;
    sp.numContacts = bNum.addr; sp.prePrepDesc = bPrePrepDesc.addr;
    sp.solverCoreDesc = bSolverCoreDesc.addr; sp.artiCoreDesc = 0ull;
    sp.sharedDesc = bSharedDesc.addr; sp.rigidDeltaVel = bRigidDeltaVel.addr;
    sp.appliedForces = bForces.addr; sp.rigidBodyReferenceCounts = bRefCounts.addr;
    sp.dt = dt; sp.materials = 0ull; sp.rigidBodyMaterials = 0ull;
    Kernel("softBodyRigidContactBda.spv", "sb_solveRigidSoftCollisionLaunch", {}, sizeof(SbSolvePush))
        .dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    const auto gDelta = downloadBuffer<F4>(dev, bSimDelta.h, 4);
    const auto gForces = downloadBuffer<float>(dev, bForces.h, kN);
    const auto gRigidDV = downloadBuffer<F4>(dev, bRigidDeltaVel.h, 2u * kN);
    std::printf("  after solve: dy = (%.6g, %.6g, %.6g, %.6g), forces = (%.6g, %.6g)\n",
                gDelta[0].y, gDelta[1].y, gDelta[2].y, gDelta[3].y, gForces[0], gForces[1]);

    // The solve must PUSH THE VERTICES OUT, i.e. along +Y (the constraint normal).
    for (uint32_t v = 0; v < 4u; ++v)
        PXG_CHECK_MSG(gDelta[v].y > 0.0f,
                      "vertex %u must be pushed OUT of the floor (+Y), got dy = %g",
                      v, gDelta[v].y);
    // the reference counts must SURVIVE the solve (it reads them, it must not clear them)
    for (uint32_t v = 0; v < 4u; ++v)
        PXG_CHECK_MSG(f2u(gDelta[v].w) == f2u(2.0f),
                      "vertex %u's reference count must survive the solve (%g)", v, gDelta[v].w);

    // THE MASS-SPLIT CHECK, in ABSOLUTE terms.  A ratio between two vertices cannot see a
    // dropped reference-count multiply, because that scales every vertex identically -- and
    // mutation-testing confirmed exactly that: the ratio check alone let `no-masssplit`
    // through.  So the reference computes the whole solve.
    //
    // invMass_split = invMass * referenceCount = 0.5 * 2 = 1.0, and BOTH contacts write to
    // every vertex, so each vertex accumulates twice the single-contact contribution.
    const p82dref::V3 linDelta = {
        simDeltaPos[0].x * bary.x + simDeltaPos[1].x * bary.y + simDeltaPos[2].x * bary.z + simDeltaPos[3].x * bary.w,
        simDeltaPos[0].y * bary.x + simDeltaPos[1].y * bary.y + simDeltaPos[2].y * bary.z + simDeltaPos[3].y * bary.w,
        simDeltaPos[0].z * bary.x + simDeltaPos[1].z * bary.y + simDeltaPos[2].z * bary.z + simDeltaPos[3].z * bary.w,
    };
    const float invMassSplit = invMass * 2.0f;   // the reference count the query wrote
    const float massSplit = bary.x * bary.x * invMassSplit + bary.y * bary.y * invMassSplit
                          + bary.z * bary.z * invMassSplit + bary.w * bary.w * invMassSplit;
    const p82dref::SolveRef sr =
        p82dref::refSolvePGS(blk, 0, /*appliedForceRef*/ 0.0f, dt, linDelta, massSplit,
                             /*rigidRefCount*/ 1.0f);
    const float bcArr[4] = { bary.x, bary.y, bary.z, bary.w };
    size_t mm = 0;
    for (uint32_t v = 0; v < 4u; ++v)
    {
        const float expect = 2.0f * sr.deltaPos.y * (invMassSplit * bcArr[v]);
        if (!(std::fabs(gDelta[v].y - expect) < 1e-6f * PxMax(1.0f, std::fabs(expect))))
        { ++mm; std::printf("  vertex %u delta.y MM: ref %.8g got %.8g\n", v, expect, gDelta[v].y); }
    }
    PXG_CHECK_MSG(mm == 0, "the solved per-vertex deltas must match the CPU reference "
                           "including the mass-splitting factor (%zu mismatches)", mm);
    PXG_CHECK_MSG(sr.deltaLambdaN > 0.0f,
                  "the reference must produce a real impulse or the check has no teeth "
                  "(deltaLambdaN = %g)", sr.deltaLambdaN);

    // the applied force must be accumulated and written back (it is the clamp reference
    // the NEXT solver iteration reads)
    for (uint32_t i = 0; i < kN; ++i)
        PXG_CHECK_MSG(gForces[i] != 0.0f,
                      "contact %u's accumulated lambda must be written back (got %g)", i, gForces[i]);

    // a STATIC rigid gets a zero delta with count 0 (computeRigidChange is deferred)
    for (uint32_t i = 0; i < kN; ++i)
        PXG_CHECK_MSG(f2u(gRigidDV[i].w) == f2u(0.0f),
                      "a static rigid contact must write count 0 (got %g)", gRigidDV[i].w);

    dev->destroyBuffer(bSimDeltaPos.h); dev->destroyBuffer(bSimDelta.h); dev->destroyBuffer(bTets.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h);
    dev->destroyBuffer(bConstraint.h); dev->destroyBuffer(bForces.h); dev->destroyBuffer(bRefCounts.h);
    dev->destroyBuffer(bRigidDeltaVel.h); dev->destroyBuffer(bBody2World.h);
    dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bSolverBodyIndices.h); dev->destroyBuffer(bOutSolverVel.h);
    dev->destroyBuffer(bVelPool.h); dev->destroyBuffer(bPrepDesc.h);
    dev->destroyBuffer(bPrePrepDesc.h); dev->destroyBuffer(bSolverCoreDesc.h);
    dev->destroyBuffer(bSharedDesc.h);
}


// ---------------------------------------------------------------------------
// "Once active, stay active" (softBody.cu:1888) — and why it needs its own fixture.
//
// The query does NOT re-evaluate a contact that is already marked in-collision; it keeps it
// active unconditionally, to hold the reference count CONSERVATIVE across solver iterations.
// Neither the scene nor softBodySimRigidContactQuerySolve can see that: in both, every
// marked contact is still geometrically active, so `wasActive || (CN < 0)` gives the same
// answer with or without the first term.  Mutation-tested, replacing the wasActive branch
// with `if (false)` survives both.
//
// This fixture makes the two disagree.  It pre-marks the contact in-collision and gives it
// a SEPARATING geometry (positive penetration), so CN >= 0 and the contact is no longer
// active on its own.  With the wasActive path it still counts; without it, the reference
// counts stay zero and the next solve divides by the wrong denominator.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimRigidContactStaysActive)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;

    const float invMass = 0.5f;
    std::vector<F4> simDeltaPos(4, F4{ 0.0f, 0.0f, 0.0f, invMass });
    std::vector<F4> simDelta(4, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };
    DevBuf bSimDeltaPos = makeDev(dev, simDeltaPos), bSimDelta = makeDev(dev, simDelta);
    DevBuf bTets = makeDev(dev, simTets);
    if (!bSimDeltaPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const float penBiasClampFEM = -1.0e10f;
    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bTets.addr));
    sb.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bSimDeltaPos.addr));
    sb.mSimDelta = reinterpret_cast<float4*>(size_t(bSimDelta.addr));
    sb.mPenBiasClamp = penBiasClampFEM;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // TWO contacts on tet 0.  Contact 0 is pre-marked in-collision; contact 1 is not.  Both
    // are SEPARATING, so neither would activate on its own.
    const uint32_t kN = 2u;
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    std::vector<FemOtherContactInfoHost> infos(kN);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemOtherContactInfoHost));
    for (uint32_t i = 0; i < kN; ++i) { infos[i].pairInd0 = staticNodeInd; infos[i].pairInd1 = 0u; }
    infos[0].rigidMatInd = 0x80000000u;   // markInCollision(true) from a previous iteration
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { kN };
    DevBuf bNum = makeDev(dev, numContacts);

    // POSITIVE penetration -> error > 0 -> CN >= 0 -> not active on geometry alone.
    SbRigidConstraintHost blk;
    std::memset(&blk, 0, sizeof(blk));
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    const float penBiasClampRigid = -5.0f;
    // Vertex 3's weight is BELOW the 1e-3 gate (softBody.cu:1911), so it must not be
    // counted: a vertex a contact barely touches has no claim on the mass split.  The other
    // three are well above it.  Without this, every weight in every fixture is > 1e-3 and
    // deleting the gate is invisible.
    const F4 bary = { 0.0005f, 0.5f, 0.4990f, 0.0005f };
    for (uint32_t i = 0; i < kN; ++i)
        p82dref::refPrepare(blk, int(i), p82dref::V3{ 0.0f, -1.0f, 0.0f },
                            p82dref::V3{ 0.5f, 0.30f, 0.7f }, /*pen*/ +0.05f,
                            p82dref::V3{ 0, 0, 0 }, bary, p82dref::V3{ 0, 0, 0 }, 0.0f,
                            penBiasClampRigid, sqrtInvInertia, penBiasClampFEM, true);
    std::vector<uint8_t> constraintBytes(sizeof(SbRigidConstraintHost));
    std::memcpy(constraintBytes.data(), &blk, sizeof(blk));
    DevBuf bConstraint = makeDev(dev, constraintBytes);

    std::vector<float> appliedForces(kN, 0.0f);
    DevBuf bForces = makeDev(dev, appliedForces);
    std::vector<uint32_t> refCounts = { 0u, 0u };
    DevBuf bRefCounts = makeDev(dev, refCounts);

    std::vector<F4> body2World = { { 0, 0, 0, 1 }, { 0, 0, 0, 0 } };
    DevBuf bBody2World = makeDev(dev, body2World);
    std::vector<uint8_t> solverBodyData(96, 0);
    std::memcpy(solverBodyData.data() + 12, &penBiasClampRigid, 4);
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);
    std::vector<uint32_t> solverBodyIndices = { 0u };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);
    std::vector<F4> outSolverVelocity(4, F4{ 0, 0, 0, 0 });
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);
    std::vector<F4> solverBodyVelPool(4, F4{ 0, 0, 0, 0 });
    DevBuf bVelPool = makeDev(dev, solverBodyVelPool);

    std::vector<uint8_t> prepDesc(304, 0);
    sbPokePtr(prepDesc, 64, bSolverBodyData.addr);
    sbPokePtr(prepDesc, 72, bSolverTxIData.addr);
    sbPokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    sbPokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);
    std::vector<uint8_t> solverCoreDesc(320, 0);
    sbPokePtr(solverCoreDesc, 0, bOutSolverVel.addr);
    { const uint32_t one = 1u; std::memcpy(solverCoreDesc.data() + 268, &one, 4); }
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);
    std::vector<uint8_t> sharedDesc(224, 0);
    sbPokePtr(sharedDesc, 160, bVelPool.addr);
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    SbQueryPush qp = {};
    qp.softbodies = bDesc.addr; qp.contactInfos = bInfos.addr; qp.constraints = bConstraint.addr;
    qp.numContacts = bNum.addr; qp.prePrepDesc = bPrePrepDesc.addr;
    qp.solverCoreDesc = bSolverCoreDesc.addr; qp.artiCoreDesc = 0ull;
    qp.sharedDesc = bSharedDesc.addr; qp.dt = dt;
    qp.appliedForces = bForces.addr; qp.rigidBodyReferenceCounts = bRefCounts.addr;
    Kernel("softBodyRigidContactBda.spv", "sb_queryRigidSoftContactReferenceCountLaunch", {}, sizeof(SbQueryPush))
        .dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    const auto gDelta = downloadBuffer<F4>(dev, bSimDelta.h, 4);
    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, bInfos.h, kN);
    std::printf("  counts = (%g, %g, %g, %g); contact 1 marked = %d\n",
                gDelta[0].w, gDelta[1].w, gDelta[2].w, gDelta[3].w,
                (gInfo[1].rigidMatInd & 0x80000000u) != 0u ? 1 : 0);

    // Contact 1 is separating and was NOT marked, so it must stay inactive: it contributes
    // nothing.  Contact 0 is separating too but WAS marked, so the wasActive path keeps it
    // active and it contributes exactly one count per vertex.
    PXG_CHECK_MSG((gInfo[1].rigidMatInd & 0x80000000u) == 0u,
                  "a separating, unmarked contact must NOT be marked in-collision -- if it "
                  "is, the fixture's geometry is not actually separating and it tests nothing");
    for (uint32_t v = 1; v < 3u; ++v)
        PXG_CHECK_MSG(f2u(gDelta[v].w) == f2u(1.0f),
                      "vertex %u must be counted exactly once: only the ALREADY-MARKED "
                      "contact stays active (count %g, expected 1)", v, gDelta[v].w);
    // THE 1e-3 GATE, on TWO different components.  Vertices 0 and 3 carry weights of
    // 0.0005, below the threshold, so the contact must not claim them however active it is.
    // Two components rather than one because the four gates are separate lines of code:
    // testing only one leaves the others free to be deleted.
    PXG_CHECK_MSG(f2u(gDelta[0].w) == f2u(0.0f),
                  "vertex 0's weight is %g, below the 1e-3 gate, so it must NOT be "
                  "reference-counted (got %g)", bary.x, gDelta[0].w);
    PXG_CHECK_MSG(f2u(gDelta[3].w) == f2u(0.0f),
                  "vertex 3's weight is %g, below the 1e-3 gate, so it must NOT be "
                  "reference-counted (got %g)", bary.w, gDelta[3].w);

    dev->destroyBuffer(bSimDeltaPos.h); dev->destroyBuffer(bSimDelta.h); dev->destroyBuffer(bTets.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h);
    dev->destroyBuffer(bConstraint.h); dev->destroyBuffer(bForces.h); dev->destroyBuffer(bRefCounts.h);
    dev->destroyBuffer(bBody2World.h); dev->destroyBuffer(bSolverBodyData.h);
    dev->destroyBuffer(bSolverTxIData.h); dev->destroyBuffer(bSolverBodyIndices.h);
    dev->destroyBuffer(bOutSolverVel.h); dev->destroyBuffer(bVelPool.h);
    dev->destroyBuffer(bPrepDesc.h); dev->destroyBuffer(bPrePrepDesc.h);
    dev->destroyBuffer(bSolverCoreDesc.h); dev->destroyBuffer(bSharedDesc.h);
}


// ---------------------------------------------------------------------------
// An INACTIVE contact must contribute nothing — the solve's isActive guard.
//
// softBodySimRigidContactQuerySolve has both contacts active, so forcing the solve's
// `if (isActive)` to `if (true)` changes nothing there and the mutant survives.  This
// fixture is the complement: a single contact that is separating and unmarked, so the
// query leaves it inactive and the solve must leave every buffer alone.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimRigidContactInactive)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;

    const float invMass = 0.5f;
    std::vector<F4> simDeltaPos(4, F4{ 0.0f, 0.0f, 0.0f, invMass });
    std::vector<F4> simDelta(4, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };
    DevBuf bSimDeltaPos = makeDev(dev, simDeltaPos), bSimDelta = makeDev(dev, simDelta);
    DevBuf bTets = makeDev(dev, simTets);
    if (!bSimDeltaPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const float penBiasClampFEM = -1.0e10f;
    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bTets.addr));
    sb.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bSimDeltaPos.addr));
    sb.mSimDelta = reinterpret_cast<float4*>(size_t(bSimDelta.addr));
    sb.mPenBiasClamp = penBiasClampFEM;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ONE contact: separating (positive penetration) and NOT marked in-collision.
    const uint32_t kN = 1u;
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    std::vector<FemOtherContactInfoHost> infos(kN);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemOtherContactInfoHost));
    infos[0].pairInd0 = staticNodeInd; infos[0].pairInd1 = 0u;
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { kN };
    DevBuf bNum = makeDev(dev, numContacts);

    SbRigidConstraintHost blk;
    std::memset(&blk, 0, sizeof(blk));
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    const float penBiasClampRigid = -5.0f;
    const F4 bary = { 0.25f, 0.25f, 0.25f, 0.25f };
    p82dref::refPrepare(blk, 0, p82dref::V3{ 0.0f, -1.0f, 0.0f },
                        p82dref::V3{ 0.5f, 0.30f, 0.7f }, /*pen*/ +0.05f,
                        p82dref::V3{ 0, 0, 0 }, bary, p82dref::V3{ 0, 0, 0 }, 0.0f,
                        penBiasClampRigid, sqrtInvInertia, penBiasClampFEM, true);
    std::vector<uint8_t> constraintBytes(sizeof(SbRigidConstraintHost));
    std::memcpy(constraintBytes.data(), &blk, sizeof(blk));
    DevBuf bConstraint = makeDev(dev, constraintBytes);

    std::vector<float> appliedForces = { 0.0f };
    DevBuf bForces = makeDev(dev, appliedForces);
    std::vector<uint32_t> refCounts = { 0u, 0u };
    DevBuf bRefCounts = makeDev(dev, refCounts);
    std::vector<F4> rigidDeltaVel(2u * kN, F4{ 9.f, 9.f, 9.f, 9.f });
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);

    std::vector<F4> body2World = { { 0, 0, 0, 1 }, { 0, 0, 0, 0 } };
    DevBuf bBody2World = makeDev(dev, body2World);
    std::vector<uint8_t> solverBodyData(96, 0);
    std::memcpy(solverBodyData.data() + 12, &penBiasClampRigid, 4);
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);
    std::vector<uint32_t> solverBodyIndices = { 0u };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);
    std::vector<F4> outSolverVelocity(4, F4{ 0, 0, 0, 0 });
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);
    std::vector<F4> solverBodyVelPool(4, F4{ 0, 0, 0, 0 });
    DevBuf bVelPool = makeDev(dev, solverBodyVelPool);

    std::vector<uint8_t> prepDesc(304, 0);
    sbPokePtr(prepDesc, 64, bSolverBodyData.addr);
    sbPokePtr(prepDesc, 72, bSolverTxIData.addr);
    sbPokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    sbPokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);
    std::vector<uint8_t> solverCoreDesc(320, 0);
    sbPokePtr(solverCoreDesc, 0, bOutSolverVel.addr);
    { const uint32_t one = 1u; std::memcpy(solverCoreDesc.data() + 268, &one, 4); }
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);
    std::vector<uint8_t> sharedDesc(224, 0);
    sbPokePtr(sharedDesc, 160, bVelPool.addr);
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    SbQueryPush qp = {};
    qp.softbodies = bDesc.addr; qp.contactInfos = bInfos.addr; qp.constraints = bConstraint.addr;
    qp.numContacts = bNum.addr; qp.prePrepDesc = bPrePrepDesc.addr;
    qp.solverCoreDesc = bSolverCoreDesc.addr; qp.artiCoreDesc = 0ull;
    qp.sharedDesc = bSharedDesc.addr; qp.dt = dt;
    qp.appliedForces = bForces.addr; qp.rigidBodyReferenceCounts = bRefCounts.addr;
    Kernel("softBodyRigidContactBda.spv", "sb_queryRigidSoftContactReferenceCountLaunch", {}, sizeof(SbQueryPush))
        .dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    SbSolvePush sp = {};
    sp.softbodies = bDesc.addr; sp.contactInfos = bInfos.addr; sp.constraints = bConstraint.addr;
    sp.numContacts = bNum.addr; sp.prePrepDesc = bPrePrepDesc.addr;
    sp.solverCoreDesc = bSolverCoreDesc.addr; sp.artiCoreDesc = 0ull;
    sp.sharedDesc = bSharedDesc.addr; sp.rigidDeltaVel = bRigidDeltaVel.addr;
    sp.appliedForces = bForces.addr; sp.rigidBodyReferenceCounts = bRefCounts.addr;
    sp.dt = dt; sp.materials = 0ull; sp.rigidBodyMaterials = 0ull;
    Kernel("softBodyRigidContactBda.spv", "sb_solveRigidSoftCollisionLaunch", {}, sizeof(SbSolvePush))
        .dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    const auto gDelta = downloadBuffer<F4>(dev, bSimDelta.h, 4);
    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, bInfos.h, kN);
    const auto gForces = downloadBuffer<float>(dev, bForces.h, kN);

    PXG_CHECK_MSG((gInfo[0].rigidMatInd & 0x80000000u) == 0u,
                  "a separating contact must not be marked in-collision -- if it is, this "
                  "fixture's geometry is not separating and it tests nothing");
    for (uint32_t v = 0; v < 4u; ++v)
    {
        PXG_CHECK_MSG(f2u(gDelta[v].w) == f2u(0.0f),
                      "an inactive contact must not reference-count vertex %u (got %g)",
                      v, gDelta[v].w);
        PXG_CHECK_MSG(f2u(gDelta[v].x) == f2u(0.0f) && f2u(gDelta[v].y) == f2u(0.0f) &&
                      f2u(gDelta[v].z) == f2u(0.0f),
                      "an inactive contact must not move vertex %u (got %g, %g, %g)",
                      v, gDelta[v].x, gDelta[v].y, gDelta[v].z);
    }
    PXG_CHECK_MSG(f2u(gForces[0]) == f2u(0.0f),
                  "an inactive contact must not accumulate an applied force (got %g)", gForces[0]);

    dev->destroyBuffer(bSimDeltaPos.h); dev->destroyBuffer(bSimDelta.h); dev->destroyBuffer(bTets.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h);
    dev->destroyBuffer(bConstraint.h); dev->destroyBuffer(bForces.h); dev->destroyBuffer(bRefCounts.h);
    dev->destroyBuffer(bRigidDeltaVel.h); dev->destroyBuffer(bBody2World.h);
    dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bSolverBodyIndices.h); dev->destroyBuffer(bOutSolverVel.h);
    dev->destroyBuffer(bVelPool.h); dev->destroyBuffer(bPrepDesc.h);
    dev->destroyBuffer(bPrePrepDesc.h); dev->destroyBuffer(bSolverCoreDesc.h);
    dev->destroyBuffer(bSharedDesc.h);
}


// ===========================================================================
// P8.3 — the softbody-vs-rigid ATTACHMENT prepare, at unit level.
//
// TWO attachments in one dispatch, deliberately of DIFFERENT kinds:
//   slot 0  a VERTEX attachment (baryOrType all zero) -> invMass1 is the vertex's own .w;
//   slot 1  a TET attachment with four distinct weights -> invMass1 is
//           getSoftBodyInvMass(blended .w, bary) = blended.w * |bary|.
//
// The second is the point of the fixture.  |bary| for (0.4, 0.2, 0.3, 0.1) is
// sqrt(0.30) = 0.5477, so a port that forgot the square root and used the blended `.w`
// directly -- which is exactly what the cloth path does -- lands on a velocity multiplier
// that is wrong by that factor, and every response term with it.
//
// The rigid anchor has a NON-IDENTITY rotation and a translated frame, so quatRotateVec and
// the ra / error chain have teeth; with an identity quaternion the rotate is invisible.
// ===========================================================================
struct SbPrepAttachPush
{
    uint64_t softbodies, rigidAttachments, activeRigidAttachments, rigidAttachmentIds;
    uint32_t numRigidAttachments, pad36;
    uint64_t constraints, prePrepDesc, prepDesc, sharedDesc, rigidDeltaVel;
};
static_assert(sizeof(SbPrepAttachPush) == 80, "sb attach prepare blob must be 80 bytes");
static_assert(offsetof(SbPrepAttachPush, numRigidAttachments) == 32, "count @32");
static_assert(offsetof(SbPrepAttachPush, constraints) == 40, "constraints @40");
static_assert(offsetof(SbPrepAttachPush, rigidDeltaVel) == 72, "rigidDeltaVel @72");

struct SbSolveAttachPush
{
    uint64_t softbodies, attachments;
    uint32_t numAttachments, pad20;
    uint64_t prePrepDesc, solverCoreDesc, artiCoreDesc, sharedDesc;
    float dt; uint32_t pad60;
    uint64_t rigidDeltaVel;
};
static_assert(sizeof(SbSolveAttachPush) == 72, "sb attach solve blob must be 72 bytes");
static_assert(offsetof(SbSolveAttachPush, numAttachments) == 16, "count @16");
static_assert(offsetof(SbSolveAttachPush, dt) == 56, "dt @56");
static_assert(offsetof(SbSolveAttachPush, rigidDeltaVel) == 64, "rigidDeltaVel @64");

PXG_TEST(softBodySimRigidAttachPrepare)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    // four sim vertices with DISTINCT inverse masses, so the blend is observable
    std::vector<F4> simPos = {
        { 0.10f, 0.20f, 0.30f, 0.50f }, { 1.10f, 0.25f, 0.35f, 0.25f },
        { 0.15f, 1.20f, 0.40f, 0.75f }, { 0.20f, 0.30f, 1.30f, 0.60f },
    };
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };
    DevBuf bPos = makeDev(dev, simPos), bTets = makeDev(dev, simTets);
    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bTets.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // --- two attachments, raw bytes at the mirror offsets ------------------
    // A THIRD attachment on a KINEMATIC anchor.  The prepare swaps in an IDENTITY inertia
    // and zeroes the inertia scale when `invMass0 == 0 && !isStatic` -- i.e. exactly a
    // kinematic rigid.  With only static anchors that branch is never taken, `inertiaScale`
    // stays 1, and multiplying by it is invisible; mutation-tested, deleting the multiply
    // survives a static-only fixture.  Node index 0 is non-static and the solver body data
    // is zeroed, so slot 2 is kinematic.
    const uint32_t kN = 3u;
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    const uint64_t kinematicNodeInd = 0x0000000000000000ull;
    const F4 tetBary = { 0.4f, 0.2f, 0.3f, 0.1f };
    const F4 axisAngle = { 0.3f, -0.5f, 0.8f, 0.25f };
    const F4 lowHigh = { -1.0f, 1.0f, -2.0f, 2.0f };
    const V3ish ra4v = { 0.07f, -0.11f, 0.05f };
    std::vector<uint8_t> attBytes(kN * 96u, 0);
    auto pokeAtt = [&](uint32_t i, uint64_t idx0, uint64_t idx1, F4 localPose, F4 bary) {
        uint8_t* b = attBytes.data() + size_t(i) * 96u;
        std::memcpy(b + 0, &idx0, 8);
        std::memcpy(b + 8, &idx1, 8);
        std::memcpy(b + 32, &localPose, 16);
        std::memcpy(b + 48, &lowHigh, 16);
        std::memcpy(b + 64, &axisAngle, 16);
        std::memcpy(b + 80, &bary, 16);
    };
    const F4 localPose0 = { ra4v.x, ra4v.y, ra4v.z, 0.0f };
    pokeAtt(0, staticNodeInd, 0ull, localPose0, F4{ 0, 0, 0, 0 });   // VERTEX, elemIdx 0
    pokeAtt(1, staticNodeInd, 0ull, localPose0, tetBary);            // TET, elemIdx 0
    pokeAtt(2, kinematicNodeInd, 0ull, localPose0, F4{ 0, 0, 0, 0 }); // VERTEX on a KINEMATIC anchor
    DevBuf bAtt = makeDev(dev, attBytes);

    std::vector<uint32_t> activeAtt = { 0u, 1u, 2u };
    DevBuf bActive = makeDev(dev, activeAtt);
    std::vector<uint64_t> attIds(kN, 0xdeadbeefdeadbeefull);
    DevBuf bAttIds = makeDev(dev, attIds);
    std::vector<uint8_t> constraintBytes(sizeof(p82dref::AttachConstraintHost), 0xCD);
    DevBuf bConstraint = makeDev(dev, constraintBytes);
    std::vector<F4> rigidDeltaVel(2u * kN, F4{ 9.f, 9.f, 9.f, 9.f });
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);

    // --- the rigid anchor: NON-identity rotation, translated frame ---------
    // a quaternion for ~37 degrees about a tilted axis, normalised on the host.
    float qraw[4] = { 0.18f, 0.26f, -0.11f, 0.94f };
    const float qm = std::sqrt(qraw[0]*qraw[0] + qraw[1]*qraw[1] + qraw[2]*qraw[2] + qraw[3]*qraw[3]);
    const F4 bodyQ = { qraw[0]/qm, qraw[1]/qm, qraw[2]/qm, qraw[3]/qm };
    const F4 bodyP = { 0.33f, -0.21f, 0.44f, 0.0f };
    std::vector<F4> body2World = { bodyQ, bodyP };
    DevBuf bBody2World = makeDev(dev, body2World);

    const float invMass0 = 0.0f;   // static anchor
    std::vector<uint8_t> solverBodyData(96, 0);
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);
    std::vector<uint32_t> solverBodyIndices = { 0u };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);

    std::vector<uint8_t> prepDesc(304, 0);
    sbPokePtr(prepDesc, 64, bSolverBodyData.addr);
    sbPokePtr(prepDesc, 72, bSolverTxIData.addr);
    sbPokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    sbPokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);

    SbPrepAttachPush pp = {};
    pp.softbodies = bDesc.addr; pp.rigidAttachments = bAtt.addr;
    pp.activeRigidAttachments = bActive.addr; pp.rigidAttachmentIds = bAttIds.addr;
    pp.numRigidAttachments = kN; pp.constraints = bConstraint.addr;
    pp.prePrepDesc = bPrePrepDesc.addr; pp.prepDesc = bPrepDesc.addr;
    pp.sharedDesc = 0ull;   // DEFERRED: articulation-only, must not be dereferenced
    pp.rigidDeltaVel = bRigidDeltaVel.addr;

    Kernel("softBodyRigidAttachBda.spv", "sb_rigidAttachmentPrepareLaunch", {}, sizeof(SbPrepAttachPush))
        .dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    // ---- CPU reference ---------------------------------------------------
    p82dref::AttachConstraintHost ref;
    std::memset(&ref, 0xCD, sizeof(ref));
    const p82dref::V3 ra4{ ra4v.x, ra4v.y, ra4v.z };
    const p82dref::V3 bodyPv{ bodyP.x, bodyP.y, bodyP.z };

    // slot 0: VERTEX -> the vertex's own position and inverse mass
    const p82dref::V3 pt0{ simPos[0].x, simPos[0].y, simPos[0].z };
    p82dref::refAttachPrepare(ref, 0, bodyQ, bodyPv, ra4, pt0, simPos[0].w,
                              sqrtInvInertia, invMass0, true, axisAngle);
    // slot 1: TET -> the four-weight blend, and the |bary| scale on the inverse mass
    F4 blended = { 0, 0, 0, 0 };
    const float bw[4] = { tetBary.x, tetBary.y, tetBary.z, tetBary.w };
    for (int v = 0; v < 4; ++v)
    { blended.x += simPos[v].x * bw[v]; blended.y += simPos[v].y * bw[v];
      blended.z += simPos[v].z * bw[v]; blended.w += simPos[v].w * bw[v]; }
    const p82dref::V3 pt1{ blended.x, blended.y, blended.z };
    const float invMass1Tet = p82dref::softBodyInvMass(blended.w, tetBary);
    p82dref::refAttachPrepare(ref, 1, bodyQ, bodyPv, ra4, pt1, invMass1Tet,
                              sqrtInvInertia, invMass0, true, axisAngle);
    // slot 2: the KINEMATIC anchor -> identity inertia, inertiaScale 0
    p82dref::refAttachPrepare(ref, 2, bodyQ, bodyPv, ra4, pt0, simPos[0].w,
                              sqrtInvInertia, invMass0, /*isStatic*/ false, axisAngle);

    const auto gBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, sizeof(p82dref::AttachConstraintHost));
    p82dref::AttachConstraintHost got;
    std::memcpy(&got, gBytes.data(), sizeof(got));
    const auto gIds = downloadBuffer<uint64_t>(dev, bAttIds.h, kN);
    const auto gRDV = downloadBuffer<F4>(dev, bRigidDeltaVel.h, 2u * kN);

    // Bit-exact, with ONE narrow exception: -0 and +0 compare equal.
    //
    // The identity-inertia row of the kinematic slot produces a component that is
    // mathematically exactly zero (raXn2 crossed with +Z has a zero z, and the identity
    // matrix passes it through).  GPU and CPU agree on the value and disagree on the SIGN
    // BIT of that zero, which carries no numerical information -- IEEE says -0 == +0, and
    // every consumer of this field adds or multiplies it.  The exception is scoped to
    // exactly that: `v == 0.0f` is true only for the two zeros, so every NON-zero component
    // is still compared by raw bits.  Widening this to a tolerance would give up the
    // bit-exactness the NoContraction stamp buys.
    auto f2uz = [](float v) { return f2u(v == 0.0f ? 0.0f : v); };
    size_t mm = 0;
    auto cmp4 = [&](const char* nm, int slot, const F4& r, const F4& g) {
        if (f2uz(r.x) != f2uz(g.x) || f2uz(r.y) != f2uz(g.y) || f2uz(r.z) != f2uz(g.z) || f2uz(r.w) != f2uz(g.w))
        { ++mm; std::printf("  slot %d %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", slot, nm,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    for (int sl = 0; sl < 3; ++sl)
    {
        cmp4("raXn0_biasW", sl, ref.raXn0_biasW[sl], got.raXn0_biasW[sl]);
        cmp4("raXn1_biasW", sl, ref.raXn1_biasW[sl], got.raXn1_biasW[sl]);
        cmp4("raXn2_biasW", sl, ref.raXn2_biasW[sl], got.raXn2_biasW[sl]);
        cmp4("velMult_invMass", sl, ref.velMult_invMass[sl], got.velMult_invMass[sl]);
        cmp4("axis_angle", sl, ref.axis_angle[sl], got.axis_angle[sl]);
    }
    PXG_CHECK_MSG(mm == 0, "sb_rigidAttachmentPrepareLaunch bit-exact vs CPU (%zu mismatches)", mm);

    // THE |bary| SCALE: the two slots must NOT agree on the velocity multiplier, or the
    // tet path is indistinguishable from the vertex path and the sqrt has no teeth.
    PXG_CHECK_MSG(f2u(got.velMult_invMass[0].x) != f2u(got.velMult_invMass[1].x),
                  "the vertex and tet attachments must produce DIFFERENT velocity "
                  "multipliers, or this fixture cannot see getSoftBodyInvMass (both %g)",
                  got.velMult_invMass[0].x);
    // and the tet inverse mass must be the blended .w times |bary|, not the blended .w
    const float naive = blended.w;
    PXG_CHECK_MSG(std::fabs(invMass1Tet - naive) > 1e-3f,
                  "the fixture's barycentric must make |bary| differ from 1 (%.5f vs %.5f)",
                  invMass1Tet, naive);

    // the passthrough fields, and the ids the prepare records
    // The KINEMATIC slot must differ from the STATIC slot 0, which is otherwise identical
    // to it -- same point, same inverse mass, same anchor pose.  Only the inertia branch
    // differs, so if they agree the identity path was not taken and inertiaScale is
    // untested.
    PXG_CHECK_MSG(f2u(got.velMult_invMass[0].x) != f2u(got.velMult_invMass[2].x),
                  "the static and kinematic anchors must produce DIFFERENT velocity "
                  "multipliers -- the kinematic one takes the identity-inertia branch "
                  "(both %g)", got.velMult_invMass[0].x);

    for (int sl = 0; sl < 3; ++sl)
    {
        PXG_CHECK_MSG(f2u(got.low_high_limits[sl].x) == f2u(lowHigh.x),
                      "slot %d low_high_limits must pass through unchanged", sl);
        PXG_CHECK_MSG(got.elemId[sl] == 0u, "slot %d elemId must be recorded", sl);
        const uint64_t wantId = (sl == 2) ? kinematicNodeInd : staticNodeInd;
        PXG_CHECK_MSG(got.rigidId[sl] == wantId, "slot %d rigidId must be recorded", sl);
        PXG_CHECK_MSG(gIds[sl] == wantId,
                      "rigidAttachmentIds[%d] must be written by the prepare", sl);
    }
    // baryOrType passes through: zero for the vertex slot, the weights for the tet slot
    PXG_CHECK_MSG(f2u(got.baryOrType[0].x) == f2u(0.0f) && f2u(got.baryOrType[1].x) == f2u(tetBary.x),
                  "baryOrType must pass through per slot (%g, %g)",
                  got.baryOrType[0].x, got.baryOrType[1].x);
    // rigidDeltaVel is zeroed for every in-range attachment, both halves
    for (uint32_t i = 0; i < 2u * kN; ++i)
        PXG_CHECK_MSG(f2u(gRDV[i].x) == f2u(0.0f),
                      "rigidDeltaVel[%u] must be zeroed by the prepare (got %g)", i, gRDV[i].x);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bTets.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bAtt.h); dev->destroyBuffer(bActive.h); dev->destroyBuffer(bAttIds.h);
    dev->destroyBuffer(bConstraint.h); dev->destroyBuffer(bRigidDeltaVel.h);
    dev->destroyBuffer(bBody2World.h); dev->destroyBuffer(bSolverBodyData.h);
    dev->destroyBuffer(bSolverTxIData.h); dev->destroyBuffer(bSolverBodyIndices.h);
    dev->destroyBuffer(bPrepDesc.h); dev->destroyBuffer(bPrePrepDesc.h);
}


// ---------------------------------------------------------------------------
// P8.3 — the attachment SOLVE.  Prepare then solve, on a constraint the prepare itself
// produced, so the two halves are checked against each other rather than against a
// hand-written block that could be wrong in a way both agree on.
//
// The anchor is displaced from the attached point, so there is a real position error to
// correct, and the solve must pull the softbody vertex TOWARD the anchor.  Both the vertex
// and the tet attachment are driven, and the tet's four vertices must each receive a share
// scaled by ITS barycentric weight and ITS inverse mass -- the scatter that mirrors CUDA's
// AtomicAdd loop.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimRigidAttachSolve)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;

    const float invMass = 0.5f;
    std::vector<F4> simPos = {
        { 0.10f, 0.20f, 0.30f, invMass }, { 1.10f, 0.25f, 0.35f, invMass },
        { 0.15f, 1.20f, 0.40f, invMass }, { 0.20f, 0.30f, 1.30f, invMass },
    };
    // a NON-ZERO velocity, so the solve has something other than the position error to act on
    std::vector<F4> simVel(4, F4{ 0.05f, -0.12f, 0.07f, invMass });
    std::vector<F4> simDelta(4, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 0u, 0u, 0u } };
    DevBuf bPos = makeDev(dev, simPos), bVel = makeDev(dev, simVel);
    DevBuf bDelta = makeDev(dev, simDelta), bTets = makeDev(dev, simTets);
    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bTets.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    sb.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    sb.mSimDelta = reinterpret_cast<float4*>(size_t(bDelta.addr));
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    const uint32_t kN = 2u;
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    const F4 tetBary = { 0.4f, 0.2f, 0.3f, 0.1f };
    const F4 axisAngle = { 0.0f, 1.0f, 0.0f, 0.25f };
    const F4 lowHigh = { -1.0f, 1.0f, -2.0f, 2.0f };
    // the anchor sits WELL AWAY from the attached point -> a real position error
    const F4 localPose0 = { 0.8f, 0.9f, -0.6f, 0.0f };
    std::vector<uint8_t> attBytes(kN * 96u, 0);
    auto pokeAtt = [&](uint32_t i, F4 bary) {
        uint8_t* b = attBytes.data() + size_t(i) * 96u;
        const uint64_t i0 = staticNodeInd, i1 = 0ull;
        std::memcpy(b + 0, &i0, 8); std::memcpy(b + 8, &i1, 8);
        std::memcpy(b + 32, &localPose0, 16);
        std::memcpy(b + 48, &lowHigh, 16);
        std::memcpy(b + 64, &axisAngle, 16);
        std::memcpy(b + 80, &bary, 16);
    };
    pokeAtt(0, F4{ 0, 0, 0, 0 });   // VERTEX
    pokeAtt(1, tetBary);            // TET
    DevBuf bAtt = makeDev(dev, attBytes);

    std::vector<uint32_t> activeAtt = { 0u, 1u };
    DevBuf bActive = makeDev(dev, activeAtt);
    std::vector<uint64_t> attIds(kN, 0ull);
    DevBuf bAttIds = makeDev(dev, attIds);
    std::vector<uint8_t> constraintBytes(sizeof(p82dref::AttachConstraintHost), 0);
    DevBuf bConstraint = makeDev(dev, constraintBytes);
    std::vector<F4> rigidDeltaVel(2u * kN, F4{ 9.f, 9.f, 9.f, 9.f });
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);

    std::vector<F4> body2World = { { 0, 0, 0, 1 }, { 0, 0, 0, 0 } };
    DevBuf bBody2World = makeDev(dev, body2World);
    std::vector<uint8_t> solverBodyData(96, 0);
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);
    std::vector<uint32_t> solverBodyIndices = { 0u };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);
    std::vector<F4> outSolverVelocity(4, F4{ 0, 0, 0, 0 });
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);
    std::vector<F4> velPool(4, F4{ 0, 0, 0, 0 });
    DevBuf bVelPool = makeDev(dev, velPool);

    std::vector<uint8_t> prepDesc(304, 0);
    sbPokePtr(prepDesc, 64, bSolverBodyData.addr);
    sbPokePtr(prepDesc, 72, bSolverTxIData.addr);
    sbPokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    sbPokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);
    std::vector<uint8_t> solverCoreDesc(320, 0);
    sbPokePtr(solverCoreDesc, 0, bOutSolverVel.addr);
    { const uint32_t one = 1u; std::memcpy(solverCoreDesc.data() + 268, &one, 4); }
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);
    std::vector<uint8_t> sharedDesc(224, 0);
    sbPokePtr(sharedDesc, 160, bVelPool.addr);
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    // ---- PREPARE, so the solve reads a constraint this port produced ----
    SbPrepAttachPush pp = {};
    pp.softbodies = bDesc.addr; pp.rigidAttachments = bAtt.addr;
    pp.activeRigidAttachments = bActive.addr; pp.rigidAttachmentIds = bAttIds.addr;
    pp.numRigidAttachments = kN; pp.constraints = bConstraint.addr;
    pp.prePrepDesc = bPrePrepDesc.addr; pp.prepDesc = bPrepDesc.addr;
    pp.sharedDesc = 0ull; pp.rigidDeltaVel = bRigidDeltaVel.addr;
    Kernel("softBodyRigidAttachBda.spv", "sb_rigidAttachmentPrepareLaunch", {}, sizeof(SbPrepAttachPush))
        .dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    // ---- SOLVE ----
    SbSolveAttachPush sp = {};
    sp.softbodies = bDesc.addr; sp.attachments = bConstraint.addr; sp.numAttachments = kN;
    sp.prePrepDesc = bPrePrepDesc.addr; sp.solverCoreDesc = bSolverCoreDesc.addr;
    sp.artiCoreDesc = 0ull; sp.sharedDesc = bSharedDesc.addr; sp.dt = dt;
    sp.rigidDeltaVel = bRigidDeltaVel.addr;
    Kernel("softBodyRigidAttachBda.spv", "sb_solveRigidSoftAttachmentLaunch", {}, sizeof(SbSolveAttachPush))
        .dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, 4);
    std::printf("  delta = (%.6g,%.6g,%.6g w=%.3g) (%.6g,%.6g,%.6g w=%.3g)\n",
                gDelta[0].x, gDelta[0].y, gDelta[0].z, gDelta[0].w,
                gDelta[1].x, gDelta[1].y, gDelta[1].z, gDelta[1].w);

    // Every vertex must have moved: vertex 0 gets the vertex attachment AND its tet share,
    // vertices 1..3 get only their tet share.
    for (uint32_t v = 0; v < 4u; ++v)
        PXG_CHECK_MSG(gDelta[v].x != 0.0f || gDelta[v].y != 0.0f || gDelta[v].z != 0.0f,
                      "vertex %u must be moved by the attachment solve (got %g,%g,%g)",
                      v, gDelta[v].x, gDelta[v].y, gDelta[v].z);

    // Reference counts: vertex 0 is touched TWICE (the vertex attachment and the tet's
    // first corner), vertices 1..3 once each.  That is the AtomicAdd(..., 1.f) half.
    PXG_CHECK_MSG(f2u(gDelta[0].w) == f2u(2.0f),
                  "vertex 0 is touched by BOTH attachments, so its reference count must be 2 "
                  "(got %g)", gDelta[0].w);
    for (uint32_t v = 1; v < 4u; ++v)
        PXG_CHECK_MSG(f2u(gDelta[v].w) == f2u(1.0f),
                      "vertex %u is touched by the tet attachment only, so its reference "
                      "count must be 1 (got %g)", v, gDelta[v].w);

    // THE PER-VERTEX SCALE: the tet share of vertices 1..3 is deltaPos * (invMass * bary),
    // and every inverse mass here is equal, so the ratio between two of them is exactly the
    // ratio of their barycentric weights.  A scatter that dropped the weight would give
    // three identical deltas.
    const float r21 = gDelta[2].y / gDelta[1].y;
    const float e21 = tetBary.z / tetBary.y;   // 0.3 / 0.2
    PXG_CHECK_MSG(std::fabs(r21 - e21) < 1e-3f,
                  "the tet scatter must scale each vertex by ITS barycentric weight: "
                  "ratio %.5f vs expected %.5f", r21, e21);
    const float r31 = gDelta[3].y / gDelta[1].y;
    const float e31 = tetBary.w / tetBary.y;   // 0.1 / 0.2
    PXG_CHECK_MSG(std::fabs(r31 - e31) < 1e-3f,
                  "the tet scatter must scale vertex 3 by its weight: ratio %.5f vs %.5f",
                  r31, e31);

    // ---- ABSOLUTE magnitude, against the CPU reference -------------------------
    // The constraint block the solve reads was produced by the prepare, which is itself
    // bit-exact against refAttachPrepare, so recomputing the solve on the host closes the
    // loop end to end.  This is the check that sees a dropped negation on deltaPos and a
    // changed bias factor; the sign and ratio checks above see neither.
    {
        p82dref::AttachConstraintHost blk;
        const auto cBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, sizeof(blk));
        std::memcpy(&blk, cBytes.data(), sizeof(blk));
        const p82dref::V3 lv{ simVel[0].x, simVel[0].y, simVel[0].z };
        // vertex 1 is touched by the TET attachment (slot 1) only, so its delta is
        // deltaPos * (invMass * bary.y) with no second contribution to disentangle.
        const p82dref::V3 dp = p82dref::refAttachSolvePGS(blk, 1, lv, dt);
        const float scale = invMass * tetBary.y;
        const float expectX = dp.x * scale, expectY = dp.y * scale;
        PXG_CHECK_MSG(std::fabs(gDelta[1].x - expectX) < 1e-6f * PxMax(1.0f, std::fabs(expectX)) &&
                      std::fabs(gDelta[1].y - expectY) < 1e-6f * PxMax(1.0f, std::fabs(expectY)),
                      "vertex 1's delta must match the CPU solve reference: ref(%.8g, %.8g) "
                      "got(%.8g, %.8g)", expectX, expectY, gDelta[1].x, gDelta[1].y);
        PXG_CHECK_MSG(std::fabs(expectX) > 1e-4f,
                      "the reference must produce a real displacement or the check has no "
                      "teeth (%g)", expectX);
    }

    // ---- DIRECTION: mirror the anchor and the delta must flip sign -------------
    // Movement and ratios alone cannot see the sign of `deltaPos = -impulse * dt`;
    // mutation-tested, dropping the negation preserves both.  Running the same solve with
    // the anchor mirrored through the origin makes the position error change sign, so every
    // component of every vertex delta must change sign with it.
    //
    // All THREE components track the error's X component, not their own: CUDA's
    // calculateAttachmentDeltaImpulsePGS passes constraint.raXn0_biasW three times, so the
    // Y and Z position errors are never read.  That upstream quirk is reproduced here, and
    // this check is where it becomes visible.
    {
        std::vector<F4> delta2(4, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
        DevBuf bDelta2 = makeDev(dev, delta2);
        PxgSoftBody sb2 = sb;
        sb2.mSimDelta = reinterpret_cast<float4*>(size_t(bDelta2.addr));
        std::vector<PxgSoftBody> descArr2 = { sb2 };
        DevBuf bDesc2 = makeDev(dev, descArr2);

        const F4 mirrored = { -localPose0.x, -localPose0.y, -localPose0.z, 0.0f };
        std::vector<uint8_t> att2 = attBytes;
        for (uint32_t i = 0; i < kN; ++i)
            std::memcpy(att2.data() + size_t(i) * 96u + 32, &mirrored, 16);
        DevBuf bAtt2 = makeDev(dev, att2);
        std::vector<uint8_t> constraint2(sizeof(p82dref::AttachConstraintHost), 0);
        DevBuf bConstraint2 = makeDev(dev, constraint2);

        SbPrepAttachPush pp2 = pp;
        pp2.softbodies = bDesc2.addr; pp2.rigidAttachments = bAtt2.addr;
        pp2.constraints = bConstraint2.addr;
        Kernel("softBodyRigidAttachBda.spv", "sb_rigidAttachmentPrepareLaunch", {}, sizeof(SbPrepAttachPush))
            .dispatch(1u, {}, pp2, 1u, 1u);
        dev->flush();
        SbSolveAttachPush sp2 = sp;
        sp2.softbodies = bDesc2.addr; sp2.attachments = bConstraint2.addr;
        Kernel("softBodyRigidAttachBda.spv", "sb_solveRigidSoftAttachmentLaunch", {}, sizeof(SbSolveAttachPush))
            .dispatch(1u, {}, sp2, 1u, 1u);
        dev->flush();

        const auto gD2 = downloadBuffer<F4>(dev, bDelta2.h, 4);
        std::printf("  mirrored delta[1] = (%.6g, %.6g, %.6g)\n", gD2[1].x, gD2[1].y, gD2[1].z);
        for (uint32_t v = 0; v < 4u; ++v)
        {
            PXG_CHECK_MSG(gDelta[v].x * gD2[v].x < 0.0f,
                          "mirroring the anchor must FLIP the sign of vertex %u's delta.x "
                          "(%g then %g) -- a solve that dropped the negation on deltaPos "
                          "would keep both signs", v, gDelta[v].x, gD2[v].x);
            PXG_CHECK_MSG(gDelta[v].y * gD2[v].y < 0.0f,
                          "mirroring must flip vertex %u's delta.y (%g then %g)",
                          v, gDelta[v].y, gD2[v].y);
        }
        dev->destroyBuffer(bDelta2.h); dev->destroyBuffer(bDesc2.h);
        dev->destroyBuffer(bAtt2.h); dev->destroyBuffer(bConstraint2.h);
    }

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bTets.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bAtt.h);
    dev->destroyBuffer(bActive.h); dev->destroyBuffer(bAttIds.h); dev->destroyBuffer(bConstraint.h);
    dev->destroyBuffer(bRigidDeltaVel.h); dev->destroyBuffer(bBody2World.h);
    dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bSolverBodyIndices.h); dev->destroyBuffer(bOutSolverVel.h);
    dev->destroyBuffer(bVelPool.h); dev->destroyBuffer(bPrepDesc.h);
    dev->destroyBuffer(bPrePrepDesc.h); dev->destroyBuffer(bSolverCoreDesc.h);
    dev->destroyBuffer(bSharedDesc.h);
}

// ===========================================================================
// P8.4 tranche a -- sb_selfCollisionMidphaseGeneratePairsLaunch.
//
// WHAT THIS FIXTURE IS FOR.  The traverser ARITHMETIC (closestPtTetrahedron,
// isPointOutsideOfPlane4, the inside/outside contact decision) is already covered
// bit-exactly by the npSbSelf / ctSbSelfQuery fixtures, which is where it was ported.
// This one covers everything the BDA lift ADDS: the 2D grid and the activeSoftbodies
// indirection, the hard-coded grid stride, the two body-flag early-outs, every gate the
// driver applies per query vertex, the cross-lane argmin, and the inlined
// PxgSoftBodyContactWriter.  Geometry is therefore compared to 1e-6; the counts, the
// encoded pair ids and the drop/keep decisions are compared EXACTLY, and they are what
// the mutation sweep aims at.
// ===========================================================================
struct SbSelfPush
{
    uint64_t activeSoftbodies, contactDistance, softbodies;
    uint64_t outPoint, outNormalPen, outBary0, outBary1, outContactInfo, totalContactCount;
    uint32_t maxContacts, pad76;
};
static_assert(sizeof(SbSelfPush) == 80, "self-collision midphase blob must be 80 bytes");

PXG_TEST(softBodySimSelfCollisionMidphase)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the tetrahedron mesh ------------------------------------------------
    // Ten unit tets, each owning its own four corners, spread 10 units apart so a query
    // vertex can only ever reach its OWN target: every drop below is attributable to the
    // gate it is aimed at, never to a neighbour stealing the contact.  A unit tet at
    // origin o has corners o, o+x, o+y, o+z; a query at o + (0.12, 0.15, 0.05) is inside
    // it and closest to the z-face, so closestP = o + (0.12, 0.15, 0) and the penetration
    // is -0.05.
    std::vector<F4> verts;
    std::vector<TetIdx4> tets;
    std::vector<uint8_t> tetHint, vertHint;
    std::vector<float> stress;

    auto addTet = [&](float ox, float oy, float oz, bool degenerate, uint8_t hint, float st) {
        const uint32_t b = uint32_t(verts.size());
        verts.push_back(F4{ ox,        oy,        oz,        1.0f });
        verts.push_back(F4{ ox + 1.0f, oy,        oz,        1.0f });
        verts.push_back(F4{ ox,        oy + 1.0f, oz,        1.0f });
        verts.push_back(degenerate ? F4{ ox, oy, oz, 1.0f } : F4{ ox, oy, oz + 1.0f, 1.0f });
        for (int i = 0; i < 4; ++i) vertHint.push_back(0u);
        tets.push_back(TetIdx4{ b, b + 1u, b + 2u, b + 3u });
        tetHint.push_back(hint);
        stress.push_back(st);
        return uint32_t(tets.size() - 1u);
    };

    // ORDER MATTERS: the shallow tet is primitive 0 and therefore LANE 0 of the shared
    // leaf, the deep one is lane 1.  Put the deep one first and "lane 0 always wins" --
    // i.e. no argmin at all -- produces the right answer by accident.
    const uint32_t T_SHALLOW = addTet( 0.0f, 0.0f, 0.02f, false, 0xfu, 0.0f);  // ...not here
    const uint32_t T_DEEP    = addTet( 0.0f, 0.0f, 0.00f, false, 0xfu, 0.0f);  // q1 wins here
    const uint32_t T_DEGEN   = addTet( 2.0f, 0.0f, 0.00f, true,  0xfu, 0.0f);  // isValidTet
    const uint32_t T_STRESS  = addTet(10.0f, 0.0f, 0.00f, false, 0xfu, 5.0f);  // stress gate
    const uint32_t T_SHARED  = addTet(20.0f, 0.0f, 0.00f, false, 0xfu, 0.0f);  // shared vertex
    const uint32_t T_FILTPR  = addTet(30.0f, 0.0f, 0.00f, false, 0xfu, 0.0f);  // pair filter
    const uint32_t T_FILTWC  = addTet(40.0f, 0.0f, 0.00f, false, 0xfu, 0.0f);  // wildcard filter
    const uint32_t T_REST    = addTet(50.0f, 0.0f, 0.00f, false, 0xfu, 0.0f);  // rest-pose gate
    const uint32_t T_NOSURF  = addTet(60.0f, 0.0f, 0.00f, false, 0x0u, 0.0f);  // tet hint == 0
    const uint32_t T_GOOD    = addTet(70.0f, 0.0f, 0.00f, false, 0xfu, 0.0f);  // q9 + q10
    const uint32_t NB_TETS = uint32_t(tets.size());
    (void)T_SHALLOW; (void)T_STRESS; (void)T_FILTWC; (void)T_REST; (void)T_NOSURF;

    // The corner of T_SHARED that is ALSO a query vertex.
    const uint32_t qShared = tets[T_SHARED].x;
    vertHint[qShared] = 1u;

    auto addQuery = [&](float x, float y, float z, uint8_t hint) {
        const uint32_t i = uint32_t(verts.size());
        verts.push_back(F4{ x, y, z, 1.0f });
        vertHint.push_back(hint);
        return i;
    };
    const uint32_t q1  = addQuery( 0.12f, 0.15f, 0.05f, 1u);   // KEPT   (T_DEEP beats T_SHALLOW)
    const uint32_t q2  = addQuery( 2.12f, 0.15f, 0.05f, 1u);   // dropped: T_DEGEN is invalid
    const uint32_t q3  = addQuery( 0.15f, 0.12f, 0.06f, 0u);   // dropped: not a surface VERTEX
    const uint32_t q4  = addQuery(30.12f, 0.15f, 0.05f, 1u);   // dropped: per-pair filter
    const uint32_t q5  = addQuery(40.12f, 0.15f, 0.05f, 1u);   // dropped: wildcard filter
    const uint32_t q6  = addQuery(50.12f, 0.15f, 0.05f, 1u);   // dropped: penetrating at REST
    const uint32_t q7  = addQuery(10.12f, 0.15f, 0.05f, 1u);   // dropped: tet stress too high
    const uint32_t q8  = addQuery(60.12f, 0.15f, 0.05f, 1u);   // dropped: tet hint == 0
    const uint32_t q9  = addQuery(70.12f, 0.15f, 0.05f, 1u);   // KEPT   (inside T_GOOD)
    // OUTSIDE T_GOOD's x-face by 0.07 -- inside the DOUBLED contact distance (0.1) and
    // outside the raw one (0.05).  This is the only tooth on `contactDistance * 2`.
    const uint32_t q10 = addQuery(69.93f, 0.30f, 0.30f, 1u);   // KEPT   (outside branch)
    // Just outside T_GOOD's a-b EDGE.  The closest point is ON the edge, so the winning
    // FACE normal (0,0,-1) and the normalised separation (0,-0.217,-0.976) genuinely
    // differ -- which is the only way to see the `dot(normal, v) > 0.707 -> normal = v`
    // replacement at all.  q10's separation happens to be parallel to its face normal, so
    // it cannot see it; that gap let the mutant through until this vertex was added.
    const uint32_t q11 = addQuery(70.50f, -0.02f, -0.09f, 1u);  // KEPT (edge, normal swap)
    (void)q2; (void)q3; (void)q7; (void)q8;

    // Rest pose: corners keep their world position; every QUERY vertex is moved 5 units
    // clear of its target so the rest gate passes -- except q6, whose rest position IS its
    // world position, i.e. it was already penetrating at rest and must be dropped.
    std::vector<F4> rest = verts;
    const uint32_t movedAtRest[] = { q1, q2, q3, q4, q5, q7, q8, q9, q10, q11, qShared };
    for (uint32_t v : movedAtRest) rest[v].y -= 5.0f;

    // T_DEGEN needs a NON-degenerate REST pose, and that is not cosmetic.  A flat tet makes
    // every one of isPointOutsideOfPlane4's four plane products identically zero, so the
    // predicate returns true for ANY point -- including in the rest pose, where it then
    // reads as "was already penetrating at rest" and drops the contact.  With the rest tet
    // flat too, the rest gate silently does isValidTet's job and a port that dropped
    // isValidTet still passes.  isValidTet is checked on the WORLD tet only, so giving the
    // REST tet real volume separates the two gates and leaves the mutant nowhere to hide.
    rest[tets[T_DEGEN].w] = F4{ 2.0f, 0.0f, 1.0f, 1.0f };

    // mSurfaceVertsHint / mTetMeshSurfaceHint are PxU8 arrays; pack them into words so the
    // shader's byte extraction is exercised at all four offsets.
    auto packBytes = [](const std::vector<uint8_t>& b) {
        std::vector<uint32_t> w((b.size() + 3u) / 4u + 1u, 0u);
        for (size_t i = 0; i < b.size(); ++i) w[i / 4u] |= uint32_t(b[i]) << (8u * (i % 4u));
        return w;
    };
    const std::vector<uint32_t> vertHintWords = packBytes(vertHint);
    const std::vector<uint32_t> tetHintWords = packBytes(tetHint);

    // vertToSurfaceTetRemap: a distinctive value per vertex so pairInd0 cannot pass by
    // accidentally matching the vertex index or the tet index.
    std::vector<uint32_t> vertToSurfaceTet(verts.size());
    for (size_t i = 0; i < vertToSurfaceTet.size(); ++i) vertToSurfaceTet[i] = 100u + uint32_t(i);

    // ---- the BV32 tree -------------------------------------------------------
    // Child 0 is a leaf holding BOTH T_DEEP and T_SHALLOW, so those two land in DIFFERENT
    // lanes of the same warp: the deeper one can only win through the cross-lane argmin.
    // The remaining tets get one leaf each.
    const uint32_t nbChildren = 1u + (NB_TETS - 2u);
    const uint32_t nbNodes = 1u, maxDepth = 1u;
    TreeBlob tree;
    tree.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes + 4u, 0u);
    tree.w[0] = uint32_t(verts.size());
    tree.w[1] = NB_TETS;
    tree.w[2] = maxDepth;
    tree.w[3] = nbNodes;
    const uint32_t nw = tree.nodeWord(0);
    tree.w[nw + W_NBNODES] = nbChildren;
    for (uint32_t k = 0; k < nbChildren; ++k)
    {
        // GENEROUS child bounds: every child is visited, so the node-level cull can never
        // stand in for a per-primitive gate.
        const float lo = -1000.0f, hi = 1000.0f;
        for (uint32_t c = 0; c < 3u; ++c)
        {
            tree.w[nw + W_MMIN + k * 4 + c] = f2u(lo);
            tree.w[nw + W_MMAX + k * 4 + c] = f2u(hi);
        }
        tree.w[nw + W_MDATA + k] = (k == 0u) ? leafWord(/*start*/ 0u, /*nbPrims*/ 2u)
                                             : leafWord(/*start*/ k + 1u, /*nbPrims*/ 1u);
    }

    // ---- filter pairs (sorted by index0, then index1) ------------------------
    const uint32_t SB_ID = 3u;   // NOT zero: the encode shift has to actually shift
    auto enc = [](uint32_t id, uint32_t elem) { return (id << 20) | elem; };
    struct PairNRHost { uint64_t index0; uint32_t index1, index2; };
    std::vector<PairNRHost> filters = {
        { enc(SB_ID, q4), enc(SB_ID, T_FILTPR), 0u },      // q4 vs its own target only
        { enc(SB_ID, q5), enc(SB_ID, 0xfffffu), 0u },      // q5 against the whole body
    };
    PXG_CHECK(filters[0].index0 < filters[1].index0);      // the binary search needs order

    DevBuf bTree = makeDev(dev, tree.w);
    DevBuf bVerts = makeDev(dev, verts);
    DevBuf bRest = makeDev(dev, rest);
    DevBuf bTets = makeDev(dev, tets);
    DevBuf bVertHint = makeDev(dev, vertHintWords);
    DevBuf bTetHint = makeDev(dev, tetHintWords);
    DevBuf bStress = makeDev(dev, stress);
    DevBuf bRemap = makeDev(dev, vertToSurfaceTet);
    DevBuf bFilters = makeDev(dev, filters);
    if (!bTree.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- four descriptors; only index 3 may produce contacts -----------------
    PxgSoftBody sb = makeZeroDesc();
    sb.mTetMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bTetHint.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
    sb.mRestPosition = reinterpret_cast<float4*>(size_t(bRest.addr));
    sb.mTetraStressCoefficient = reinterpret_cast<PxReal*>(size_t(bStress.addr));
    sb.mSurfaceVertsHint = reinterpret_cast<PxU8*>(size_t(bVertHint.addr));
    sb.mSurfaceVertToTetRemap = reinterpret_cast<PxU32*>(size_t(bRemap.addr));
    sb.mFilteringPairs = reinterpret_cast<decltype(sb.mFilteringPairs)>(size_t(bFilters.addr));
    sb.mNumFilterPairs = uint32_t(filters.size());
    sb.mNumTets = NB_TETS;
    sb.mNumVerts = uint32_t(verts.size());
    sb.mElementIndex = 1u;                       // contactDistance is indexed by THIS
    sb.mSelfCollisionFilterDistance = 0.2f;      // squared -> 0.04, the rest-pose floor
    sb.mSelfCollisionStressTolerance = 1.0f;     // T_STRESS carries 5.0
    sb.mBodyFlags = 0u;

    std::vector<PxgSoftBody> descArr(4u, makeZeroDesc());
    descArr[3] = sb;                                                    // the live one
    descArr[1] = sb; descArr[1].mBodyFlags = PxDeformableBodyFlags(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION);
    descArr[2] = sb; descArr[2].mBodyFlags = PxDeformableBodyFlags(PxDeformableBodyFlag::eKINEMATIC);
    DevBuf bDesc = makeDev(dev, descArr);

    // activeSoftbodies is DELIBERATELY not the identity: cmIdx 1 -> softbody 3.  A shader
    // that used gid.x, or skipped the indirection, would land on a disabled body.
    std::vector<uint32_t> active = { 2u, SB_ID, 1u };
    DevBuf bActive = makeDev(dev, active);

    // contactDistance[mElementIndex] = 0.05, DOUBLED by the kernel to 0.1.
    std::vector<float> contactDist = { 999.0f, 0.05f };
    DevBuf bCd = makeDev(dev, contactDist);

    const uint32_t MAXC = 16u;
    const uint32_t POISON = 0xCDCDCDCDu;
    std::vector<F4> zeroF4(MAXC, F4{ 0, 0, 0, 0 });
    std::vector<uint32_t> poison(MAXC * 4u, POISON);
    DevBuf bPoint = makeDev(dev, zeroF4);
    DevBuf bNormalPen = makeDev(dev, zeroF4);
    DevBuf bBary0 = makeDev(dev, zeroF4);
    DevBuf bBary1 = makeDev(dev, zeroF4);
    DevBuf bInfo = makeDev(dev, poison);
    std::vector<uint32_t> zeroCount = { 0u };
    DevBuf bCount = makeDev(dev, zeroCount);

    SbSelfPush push = {};
    push.activeSoftbodies = bActive.addr;
    push.contactDistance = bCd.addr;
    push.softbodies = bDesc.addr;
    push.outPoint = bPoint.addr;
    push.outNormalPen = bNormalPen.addr;
    push.outBary0 = bBary0.addr;
    push.outBary1 = bBary1.addr;
    push.outContactInfo = bInfo.addr;
    push.totalContactCount = bCount.addr;
    push.maxContacts = MAXC;

    Kernel k("softBodySelfCollisionBda.spv", "sb_selfCollisionMidphaseGeneratePairsLaunch",
             {}, sizeof(SbSelfPush));
    // THE ENGINE'S GRID, reproduced exactly: x = SB_SBMIDPHASE = 1024 (fixed, independent
    // of the vertex count), y = the number of active self-collision softbodies.  With the
    // CUDA stride (blockDim.y * gridDim.x = 2048) every vertex is visited EXACTLY ONCE; a
    // stride derived from the dispatch size instead would visit most of them 1024 times
    // and the contact count below would explode.
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ uint32_t(active.size()), 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    const auto gPoint = downloadBuffer<F4>(dev, bPoint.h, MAXC);
    const auto gNormalPen = downloadBuffer<F4>(dev, bNormalPen.h, MAXC);
    const auto gBary0 = downloadBuffer<F4>(dev, bBary0.h, MAXC);
    const auto gBary1 = downloadBuffer<F4>(dev, bBary1.h, MAXC);
    const auto gInfo = downloadBuffer<uint32_t>(dev, bInfo.h, MAXC * 4u);

    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
        std::printf("  contact %u: p=(%.5f %.5f %.5f) n=(%.5f %.5f %.5f) pen=%.6f "
                    "bary0=(%.5f %.5f %.5f) ind0=%u ind1=%u\n",
                    i, gPoint[i].x, gPoint[i].y, gPoint[i].z, gNormalPen[i].x, gNormalPen[i].y,
                    gNormalPen[i].z, gNormalPen[i].w, gBary0[i].x, gBary0[i].y, gBary0[i].z,
                    gInfo[i * 4u + 0u], gInfo[i * 4u + 2u]);

    PXG_CHECK_MSG(gCount == 4u,
                  "exactly four query vertices survive every gate (q1, q9, q10, q11) -- got "
                  "%u.  0 means the body-flag early-out or the activeSoftbodies indirection "
                  "is wrong; more than 4 means a gate was dropped or a vertex was visited "
                  "more than once", gCount);

    // ---- match each emitted contact to the query it must have come from ------
    auto near6 = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };
    bool sawQ1 = false, sawQ9 = false, sawQ10 = false, sawQ11 = false;
    for (uint32_t i = 0; i < gCount; ++i)
    {
        const uint32_t ind0 = gInfo[i * 4u + 0u];
        const uint32_t ind0hi = gInfo[i * 4u + 1u];
        const uint32_t ind1 = gInfo[i * 4u + 2u];
        PXG_CHECK_MSG(ind0hi == 0u,
                      "pairInd0 is a PxU64 field taking a PxU32 value: the high word must be "
                      "ZEROED, not left poisoned (contact %u has 0x%08x)", i, ind0hi);

        if (ind0 == enc(SB_ID, vertToSurfaceTet[q1]))
        {
            sawQ1 = true;
            PXG_CHECK_MSG(ind1 == enc(SB_ID, T_DEEP),
                          "q1 penetrates T_DEEP by 0.05 and T_SHALLOW by 0.03; the cross-lane "
                          "argmin must pick the DEEPER one (got tet %u, want %u)",
                          ind1 & 0xfffffu, T_DEEP);
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
            PXG_CHECK(near6(gNormalPen[i].x, 0.0f) && near6(gNormalPen[i].y, 0.0f) &&
                      near6(gNormalPen[i].z, -1.0f));
            PXG_CHECK(near6(gPoint[i].x, 0.12f) && near6(gPoint[i].y, 0.15f) && near6(gPoint[i].z, 0.0f));
            // barycentric0 carries the QUERY POINT and barycentric1 the closest point --
            // that is what the CUDA writer stores, and the sim-mesh remap depends on it.
            PXG_CHECK_MSG(near6(gBary0[i].x, 0.12f) && near6(gBary0[i].y, 0.15f) &&
                          near6(gBary0[i].z, 0.05f),
                          "barycentric0 must be the query POINT, got (%.5f %.5f %.5f)",
                          gBary0[i].x, gBary0[i].y, gBary0[i].z);
            PXG_CHECK(near6(gBary1[i].x, gPoint[i].x) && near6(gBary1[i].z, gPoint[i].z));
        }
        else if (ind0 == enc(SB_ID, vertToSurfaceTet[q9]))
        {
            sawQ9 = true;
            PXG_CHECK(ind1 == enc(SB_ID, T_GOOD));
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
        }
        else if (ind0 == enc(SB_ID, vertToSurfaceTet[q10]))
        {
            sawQ10 = true;
            PXG_CHECK(ind1 == enc(SB_ID, T_GOOD));
            // The OUTSIDE branch: a positive separation, and the normal replaced by the
            // normalised separation vector because their dot exceeds 0.707.
            PXG_CHECK_MSG(near6(gNormalPen[i].w, 0.07f),
                          "q10 sits 0.07 OUTSIDE T_GOOD's x-face; the outside branch reports "
                          "a positive distance (got %.6f)", gNormalPen[i].w);
            PXG_CHECK(near6(gNormalPen[i].x, -1.0f));
        }
        else if (ind0 == enc(SB_ID, vertToSurfaceTet[q11]))
        {
            sawQ11 = true;
            PXG_CHECK(ind1 == enc(SB_ID, T_GOOD));
            PXG_CHECK_MSG(near6(gNormalPen[i].w, 0.09219544f),
                          "q11's separation is |(0, -0.02, -0.09)| (got %.8f)", gNormalPen[i].w);
            PXG_CHECK_MSG(near6(gNormalPen[i].y, -0.21693045f) &&
                          near6(gNormalPen[i].z, -0.97618705f),
                          "the winning FACE normal is (0,0,-1); because it agrees with the "
                          "normalised separation to better than 0.707 the contact normal must "
                          "become the SEPARATION, got (%.6f %.6f %.6f)",
                          gNormalPen[i].x, gNormalPen[i].y, gNormalPen[i].z);
        }
        else
        {
            PXG_CHECK_MSG(false, "contact %u came from an unexpected vertex (pairInd0 = 0x%08x, "
                                 "surface tet %u)", i, ind0, ind0 & 0xfffffu);
        }
    }
    PXG_CHECK_MSG(sawQ1, "q1 must produce a contact -- it is inside T_DEEP");
    PXG_CHECK_MSG(sawQ9, "q9 must produce a contact -- it is inside T_GOOD");
    PXG_CHECK_MSG(sawQ10,
                  "q10 must produce a contact: it is 0.07 outside T_GOOD, which is inside "
                  "the DOUBLED contact distance (0.05 * 2) and outside the raw one.  This is "
                  "the only check that sees the `* 2.f`");
    PXG_CHECK_MSG(sawQ11, "q11 must produce a contact -- it is 0.092 outside T_GOOD's edge");

    // Slots past the count keep their poison -- nothing writes past the reported total.
    for (uint32_t i = gCount; i < MAXC; ++i)
        PXG_CHECK_MSG(gInfo[i * 4u + 0u] == POISON,
                      "contact info slot %u was written past the contact count", i);

    // ---- maxContacts: the counter keeps counting, the writes are DROPPED -----
    // PxgSoftBodyContactWriter::writeContact reserves the index with the atomic FIRST and
    // only then checks `index >= maxContacts`, so an undersized buffer must still report
    // the true total.  checkBufferOverflows() reads exactly that to size the next frame.
    {
        std::vector<uint32_t> info2(MAXC * 4u, POISON);
        DevBuf bInfo2 = makeDev(dev, info2);
        std::vector<uint32_t> c2 = { 0u };
        DevBuf bCount2 = makeDev(dev, c2);
        SbSelfPush p2 = push;
        p2.outContactInfo = bInfo2.addr;
        p2.totalContactCount = bCount2.addr;
        p2.maxContacts = 1u;
        Kernel k2("softBodySelfCollisionBda.spv", "sb_selfCollisionMidphaseGeneratePairsLaunch",
                  {}, sizeof(SbSelfPush));
        k2.dispatch(1024u, {}, p2, uint32_t(active.size()), 1u);
        dev->flush();
        const uint32_t n2 = downloadBuffer<uint32_t>(dev, bCount2.h, 1)[0];
        const auto i2 = downloadBuffer<uint32_t>(dev, bInfo2.h, MAXC * 4u);
        PXG_CHECK_MSG(n2 == 4u, "the contact COUNT must be the true total even when the "
                                "buffer overflows (got %u, expected 4)", n2);
        uint32_t written = 0u;
        for (uint32_t i = 0; i < MAXC; ++i)
            if (i2[i * 4u + 0u] != POISON) ++written;
        PXG_CHECK_MSG(written == 1u,
                      "with maxContacts = 1 exactly one contact may be STORED (got %u) -- a "
                      "port that dropped the bound writes into memory it does not own",
                      written);
        dev->destroyBuffer(bInfo2.h); dev->destroyBuffer(bCount2.h);
    }

    // ---- mFilteringPairs == NULL, which is the COMMON case -------------------
    // PxgSimulationCore only assigns mFilteringPairs inside its dirty-filter branch
    // (PxgSimulationCore.cpp:1446), so a body that never registered a filter reaches this
    // kernel with a NULL pointer and nbPairs == 0.  That is what the first live
    // self-collision scene hit, and it took the whole engine down: llvmpipe speculates the
    // zero-trip binary search's first load, faults at address 0, and substitutes garbage
    // that DIFFERS BETWEEN LANES -- so findFilter stops being uniform, the caller's
    // `continue` stops being group-uniform, and the workgroup barrier after it is reached
    // by only part of the group.  The symptom was a SIGSEGV that moved with the groupshared
    // layout and vanished under PXG_SYNC_EACH, which is why it read as anything but a null
    // pointer for a long time.
    //
    // The original fixture always passed a valid pairs buffer, so it could not see this.
    // It does now: the same dispatch with mFilteringPairs zeroed must produce the same
    // three contacts as above, since an empty filter list filters nothing.
    {
        std::vector<PxgSoftBody> noFilterDesc = descArr;
        for (PxgSoftBody& d : noFilterDesc)
        {
            d.mFilteringPairs = nullptr;
            d.mNumFilterPairs = 0u;
        }
        DevBuf bNoFilt = makeDev(dev, noFilterDesc);
        std::vector<uint32_t> info4(MAXC * 4u, POISON);
        DevBuf bInfo4 = makeDev(dev, info4);
        std::vector<uint32_t> c4 = { 0u };
        DevBuf bCount4 = makeDev(dev, c4);
        SbSelfPush p4 = push;
        p4.softbodies = bNoFilt.addr;
        p4.outContactInfo = bInfo4.addr;
        p4.totalContactCount = bCount4.addr;
        Kernel("softBodySelfCollisionBda.spv", "sb_selfCollisionMidphaseGeneratePairsLaunch",
               {}, sizeof(SbSelfPush))
            .dispatch(1024u, {}, p4, uint32_t(active.size()), 1u);
        dev->flush();
        const uint32_t n4 = downloadBuffer<uint32_t>(dev, bCount4.h, 1)[0];

        // q4 and q5 were dropped ONLY by the filters, so with no filter list they survive:
        // the four contacts above become six.  An empty list must filter nothing -- not
        // everything, and not something lane-dependent.
        PXG_CHECK_MSG(n4 == 6u,
                      "with mFilteringPairs = NULL the two filtered vertices (q4, q5) must come "
                      "back, giving 6 contacts, got %u.  A null pairs pointer must be READ AS "
                      "an empty list, not faulted on and not silently made lane-dependent", n4);
        dev->destroyBuffer(bNoFilt.h); dev->destroyBuffer(bInfo4.h); dev->destroyBuffer(bCount4.h);
    }

    // ---- every active body disabled -> no contacts at all -------------------
    {
        std::vector<uint32_t> onlyDisabled = { 1u, 2u };
        DevBuf bActive2 = makeDev(dev, onlyDisabled);
        std::vector<uint32_t> c3 = { 0u };
        DevBuf bCount3 = makeDev(dev, c3);
        SbSelfPush p3 = push;
        p3.activeSoftbodies = bActive2.addr;
        p3.totalContactCount = bCount3.addr;
        Kernel k3("softBodySelfCollisionBda.spv", "sb_selfCollisionMidphaseGeneratePairsLaunch",
                  {}, sizeof(SbSelfPush));
        k3.dispatch(1024u, {}, p3, 2u, 1u);
        dev->flush();
        const uint32_t n3 = downloadBuffer<uint32_t>(dev, bCount3.h, 1)[0];
        PXG_CHECK_MSG(n3 == 0u,
                      "eDISABLE_SELF_COLLISION and eKINEMATIC each suppress the whole body "
                      "(got %u contacts from two bodies that carry one flag each)", n3);
        dev->destroyBuffer(bActive2.h); dev->destroyBuffer(bCount3.h);
    }

    dev->destroyBuffer(bTree.h); dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bRest.h);
    dev->destroyBuffer(bTets.h); dev->destroyBuffer(bVertHint.h); dev->destroyBuffer(bTetHint.h);
    dev->destroyBuffer(bStress.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bPoint.h); dev->destroyBuffer(bNormalPen.h); dev->destroyBuffer(bBary0.h);
    dev->destroyBuffer(bBary1.h); dev->destroyBuffer(bInfo.h); dev->destroyBuffer(bCount.h);
}

// ===========================================================================
// P8.4 tranche b — sb_fem_contact_remap_to_simLaunch, the DEFORMABLE-vs-DEFORMABLE remap.
//
// The arithmetic is the sibling's and is covered by the two fixtures above.  What is new,
// and what this pins, is everything the second grid brings: which HALF of the pair each
// y-grid remaps, which barycentric array it reads the contact point OUT of and writes the
// result back INTO, the `hasTet` neutralisation branch that the other remap does not have,
// and the fact that pairInd0 is a PxU64 field taking a PxU32 value.
// ===========================================================================
struct FemFemContactInfoHost { uint64_t pairInd0; uint32_t pairInd1, auxInd; };
static_assert(sizeof(FemFemContactInfoHost) == 16, "PxgFemFemContactInfo is 16 bytes");

struct SbFemRemapPush
{
    uint64_t softbodies, outBarycentric0, outBarycentric1, outContactInfo,
             totalContactCount, prevContactCount;
    uint32_t maxNbContacts, pad52;
};
static_assert(sizeof(SbFemRemapPush) == 56, "fem remap push blob must be 56 bytes");
static_assert(offsetof(SbFemRemapPush, maxNbContacts) == 48, "maxNbContacts @48");

PXG_TEST(softBodySimFemContactRemap)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    // Five sim tets.  0 is far, 1 is nearer, 2 is the unit tet at the origin and CONTAINS
    // (0.2, 0.2, 0.2); 3 and 4 are DEGENERATE (fourth corner repeats the first), which is
    // what drives the hasTet branch this kernel has and its sibling does not.
    std::vector<F4> simVerts = {
        { 10.0f, 0.0f, 0.0f, 1.0f }, { 11.0f, 0.0f, 0.0f, 1.0f },
        { 10.0f, 1.0f, 0.0f, 1.0f }, { 10.0f, 0.0f, 1.0f, 1.0f },   // tet 0, far
        {  2.0f, 0.0f, 0.0f, 1.0f }, {  3.0f, 0.0f, 0.0f, 1.0f },
        {  2.0f, 1.0f, 0.0f, 1.0f }, {  2.0f, 0.0f, 1.0f, 1.0f },   // tet 1, nearer
        {  0.0f, 0.0f, 0.0f, 1.0f }, {  1.0f, 0.0f, 0.0f, 1.0f },
        {  0.0f, 1.0f, 0.0f, 1.0f }, {  0.0f, 0.0f, 1.0f, 1.0f },   // tet 2, contains
        { 20.0f, 0.0f, 0.0f, 1.0f }, { 21.0f, 0.0f, 0.0f, 1.0f },
        { 20.0f, 1.0f, 0.0f, 1.0f }, { 20.0f, 0.0f, 0.0f, 1.0f },   // tet 3, DEGENERATE
        { 30.0f, 0.0f, 0.0f, 1.0f }, { 31.0f, 0.0f, 0.0f, 1.0f },
        { 30.0f, 1.0f, 0.0f, 1.0f }, { 30.0f, 0.0f, 0.0f, 1.0f },   // tet 4, DEGENERATE
    };
    std::vector<TetIdx4> simTets = {
        { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u }, { 8u, 9u, 10u, 11u },
        { 12u, 13u, 14u, 15u }, { 16u, 17u, 18u, 19u },
    };
    // collision tet 0 -> sim candidates {0,1,2}; collision tet 1 -> {3,4}, both degenerate.
    std::vector<uint32_t> remapColToSim = { 0u, 1u, 2u, 3u, 4u, 0u, 0u, 0u };
    std::vector<uint32_t> accumRemap = { 3u, 5u, 5u, 5u };   // inclusive prefix sum

    DevBuf bSv = makeDev(dev, simVerts), bSt = makeDev(dev, simTets);
    DevBuf bRc = makeDev(dev, remapColToSim), bAr = makeDev(dev, accumRemap);
    if (!bSv.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bSt.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSv.addr));
    sb.mTetsRemapColToSim = reinterpret_cast<PxU32*>(size_t(bRc.addr));
    sb.mTetsAccumulatedRemapColToSim = reinterpret_cast<PxU32*>(size_t(bAr.addr));
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    auto enc = [](uint32_t id, uint32_t elem) { return (id << 20) | elem; };

    // Two contacts, and the two halves of each pair are DELIBERATELY DIFFERENT so that a
    // port which remapped the wrong half, or read the wrong barycentric array, cannot
    // produce these numbers:
    //   contact 0: pairInd1 -> collision tet 0 (three real candidates)
    //              pairInd0 -> collision tet 1 (two DEGENERATE candidates)
    //   contact 1: both halves -> collision tet 0, but from different query points.
    std::vector<FemFemContactInfoHost> infos(4);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemFemContactInfoHost));
    // The HIGH word of pairInd0 starts dirty: the shader must zero it, not leave it.
    infos[0].pairInd0 = (uint64_t(0xDEADBEEFu) << 32) | enc(0u, 1u);
    infos[0].pairInd1 = enc(0u, 0u);
    infos[0].auxInd = 0xABCD1234u;
    infos[1].pairInd0 = (uint64_t(0xDEADBEEFu) << 32) | enc(0u, 0u);
    infos[1].pairInd1 = enc(0u, 0u);
    infos[1].auxInd = 0x5A5A5A5Au;
    DevBuf bInfo = makeDev(dev, infos);

    // The barycentric arrays carry the contact POINT on the way in.
    const F4 SENT = { -7.0f, -7.0f, -7.0f, -7.0f };
    std::vector<F4> bary0 = { { 5.0f, 5.0f, 5.0f, 0.0f },      // contact 0 (degenerate side)
                              { 0.2f, 0.2f, 0.2f, 0.0f },      // contact 1 -> CONTAINED
                              SENT, SENT };
    std::vector<F4> bary1 = { { 0.2f, 0.2f, 0.2f, 0.0f },      // contact 0 -> CONTAINED
                              { 1.6f, 0.2f, 0.2f, 0.0f },      // contact 1 -> nearest = tet 1
                              SENT, SENT };
    DevBuf bB0 = makeDev(dev, bary0), bB1 = makeDev(dev, bary1);
    std::vector<uint32_t> count = { 2u, 0u, 0u, 0u };
    DevBuf bCount = makeDev(dev, count);

    SbFemRemapPush push = {};
    push.softbodies = bDesc.addr;
    push.outBarycentric0 = bB0.addr;
    push.outBarycentric1 = bB1.addr;
    push.outContactInfo = bInfo.addr;
    push.totalContactCount = bCount.addr;
    push.prevContactCount = 0ull;       // OPTIONAL in CUDA -> must not be dereferenced
    push.maxNbContacts = 64u;

    // THE ENGINE'S GRID: gridDim.y == 2 (PxgNarrowphaseCore.cpp:4149).  x is deliberately
    // OVER-SIZED relative to what two contacts need, so an inverted or off-by-one
    // `gid.x >= SB_REMAP_GRID` early-out shows up as either nothing happening or work
    // happening twice.
    Kernel k("softBodyContactRemapBda.spv", "sb_fem_contact_remap_to_simLaunch", {}, sizeof(SbFemRemapPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 2u, 1u);
    dev->flush();

    const auto gInfo = downloadBuffer<FemFemContactInfoHost>(dev, bInfo.h, 4);
    const auto g0 = downloadBuffer<F4>(dev, bB0.h, 4);
    const auto g1 = downloadBuffer<F4>(dev, bB1.h, 4);
    for (uint32_t i = 0; i < 2u; ++i)
        std::printf("  contact %u: pairInd0 -> tet %u (hi 0x%08x), pairInd1 -> tet %u | "
                    "bary0(%.4f %.4f %.4f %.4f) bary1(%.4f %.4f %.4f %.4f)\n",
                    i, uint32_t(gInfo[i].pairInd0) & 0xfffffu, uint32_t(gInfo[i].pairInd0 >> 32),
                    gInfo[i].pairInd1 & 0xfffffu,
                    g0[i].x, g0[i].y, g0[i].z, g0[i].w, g1[i].x, g1[i].y, g1[i].z, g1[i].w);

    auto near4 = [](float a, float b) { return std::fabs(a - b) < 1e-4f; };

    // ---- y-grid 0 remaps pairInd1 through outBarycentric1 --------------------
    PXG_CHECK_MSG((gInfo[0].pairInd1 & 0xfffffu) == 2u,
                  "contact 0's pairInd1 side queries (0.2,0.2,0.2) from outBarycentric1 and is "
                  "CONTAINED by sim tet 2 (got tet %u)", gInfo[0].pairInd1 & 0xfffffu);
    PXG_CHECK_MSG(near4(g1[0].x, 0.4f) && near4(g1[0].y, 0.2f) && near4(g1[0].z, 0.2f) &&
                  near4(g1[0].w, 0.2f),
                  "and its barycentric must be (0.4,0.2,0.2,0.2), got (%.4f %.4f %.4f %.4f)",
                  g1[0].x, g1[0].y, g1[0].z, g1[0].w);
    PXG_CHECK_MSG((gInfo[1].pairInd1 & 0xfffffu) == 1u,
                  "contact 1's pairInd1 side is outside every candidate, so the NEAREST (sim "
                  "tet 1) must win (got tet %u)", gInfo[1].pairInd1 & 0xfffffu);

    // ---- y-grid 1 remaps pairInd0 through outBarycentric0 --------------------
    // If a port had the two grids the wrong way round, contact 1 would land on sim tet 1
    // here and sim tet 2 above -- so these two together, not either alone, pin the mapping.
    PXG_CHECK_MSG((uint32_t(gInfo[1].pairInd0) & 0xfffffu) == 2u,
                  "contact 1's pairInd0 side queries (0.2,0.2,0.2) from outBarycentric0 and is "
                  "CONTAINED by sim tet 2 (got tet %u)", uint32_t(gInfo[1].pairInd0) & 0xfffffu);
    PXG_CHECK_MSG(near4(g0[1].x, 0.4f) && near4(g0[1].y, 0.2f) && near4(g0[1].z, 0.2f) &&
                  near4(g0[1].w, 0.2f),
                  "and its barycentric must be (0.4,0.2,0.2,0.2), got (%.4f %.4f %.4f %.4f)",
                  g0[1].x, g0[1].y, g0[1].z, g0[1].w);

    // ---- the hasTet branch ---------------------------------------------------
    PXG_CHECK_MSG((uint32_t(gInfo[0].pairInd0) & 0xfffffu) == 0u,
                  "contact 0's pairInd0 side names collision tet 1, whose only two candidates "
                  "are DEGENERATE -- with no valid tet anywhere the contact must be remapped "
                  "to sim tet 0 (got tet %u)", uint32_t(gInfo[0].pairInd0) & 0xfffffu);
    PXG_CHECK_MSG(g0[0].x == 0.0f && g0[0].y == 0.0f && g0[0].z == 0.0f && g0[0].w == 0.0f,
                  "...and its barycentric must be ZEROED, which is what neutralises it -- got "
                  "(%.4f %.4f %.4f %.4f)", g0[0].x, g0[0].y, g0[0].z, g0[0].w);

    // ---- pairInd0 is a PxU64 field taking a PxU32 value ----------------------
    for (uint32_t i = 0; i < 2u; ++i)
        PXG_CHECK_MSG(uint32_t(gInfo[i].pairInd0 >> 32) == 0u,
                      "contact %u's pairInd0 high word must be ZEROED, not left at 0x%08x",
                      i, uint32_t(gInfo[i].pairInd0 >> 32));

    // ---- what must NOT be touched -------------------------------------------
    PXG_CHECK_MSG(gInfo[0].auxInd == 0xABCD1234u && gInfo[1].auxInd == 0x5A5A5A5Au,
                  "auxInd carries the validity/pair-type bits and the remap must not write it "
                  "(got 0x%08x, 0x%08x)", gInfo[0].auxInd, gInfo[1].auxInd);
    PXG_CHECK_MSG((gInfo[0].pairInd1 >> 20u) == 0u && (uint32_t(gInfo[0].pairInd0) >> 20u) == 0u,
                  "the softbody id must survive both rewrites");
    PXG_CHECK_MSG(g0[2].x == SENT.x && g1[2].x == SENT.x,
                  "contact 2 is past the count and must be untouched");

    // ---- the same result at the engine's OTHER x-grid ------------------------
    // The remap is NOT idempotent: a second visit would read an already-remapped SIM index
    // as a collision index and search the wrong candidate range.  The softbody-vs-softbody
    // call site dispatches 8192 blocks where self-collision dispatches 1024, so the shader
    // pins one and refuses the rest; running both shapes must give byte-identical output.
    {
        std::vector<FemFemContactInfoHost> i2 = infos;
        std::vector<F4> a0 = bary0, a1 = bary1;
        DevBuf bI2 = makeDev(dev, i2), bA0 = makeDev(dev, a0), bA1 = makeDev(dev, a1);
        std::vector<uint32_t> c2 = { 2u, 0u, 0u, 0u };
        DevBuf bC2 = makeDev(dev, c2);
        SbFemRemapPush p2 = push;
        p2.outContactInfo = bI2.addr; p2.outBarycentric0 = bA0.addr;
        p2.outBarycentric1 = bA1.addr; p2.totalContactCount = bC2.addr;
        Kernel("softBodyContactRemapBda.spv", "sb_fem_contact_remap_to_simLaunch", {}, sizeof(SbFemRemapPush))
            .dispatch(8192u, {}, p2, 2u, 1u);
        dev->flush();
        const auto j2 = downloadBuffer<FemFemContactInfoHost>(dev, bI2.h, 4);
        const auto d0 = downloadBuffer<F4>(dev, bA0.h, 4);
        const auto d1 = downloadBuffer<F4>(dev, bA1.h, 4);
        for (uint32_t i = 0; i < 2u; ++i)
        {
            PXG_CHECK_MSG(j2[i].pairInd0 == gInfo[i].pairInd0 && j2[i].pairInd1 == gInfo[i].pairInd1,
                          "contact %u must remap identically at gridX 8192 and 1024 "
                          "(pairInd0 0x%llx vs 0x%llx, pairInd1 0x%08x vs 0x%08x)", i,
                          (unsigned long long)j2[i].pairInd0, (unsigned long long)gInfo[i].pairInd0,
                          j2[i].pairInd1, gInfo[i].pairInd1);
            PXG_CHECK(d0[i].x == g0[i].x && d1[i].x == g1[i].x);
        }
        dev->destroyBuffer(bI2.h); dev->destroyBuffer(bA0.h);
        dev->destroyBuffer(bA1.h); dev->destroyBuffer(bC2.h);
    }

    // ---- the clamp write-back, and prevContactCount as a REAL pointer --------
    {
        std::vector<FemFemContactInfoHost> i3 = infos;
        std::vector<F4> a0 = bary0, a1 = bary1;
        DevBuf bI3 = makeDev(dev, i3), bA0 = makeDev(dev, a0), bA1 = makeDev(dev, a1);
        std::vector<uint32_t> c3 = { 9u, 0u, 0u, 0u };   // more than maxNbContacts
        DevBuf bC3 = makeDev(dev, c3);
        std::vector<uint32_t> prev = { 1u, 0u, 0u, 0u };
        DevBuf bPrev = makeDev(dev, prev);
        SbFemRemapPush p3 = push;
        p3.outContactInfo = bI3.addr; p3.outBarycentric0 = bA0.addr;
        p3.outBarycentric1 = bA1.addr; p3.totalContactCount = bC3.addr;
        p3.prevContactCount = bPrev.addr; p3.maxNbContacts = 2u;
        Kernel("softBodyContactRemapBda.spv", "sb_fem_contact_remap_to_simLaunch", {}, sizeof(SbFemRemapPush))
            .dispatch(1024u, {}, p3, 2u, 1u);
        dev->flush();
        const uint32_t back = downloadBuffer<uint32_t>(dev, bC3.h, 1)[0];
        const auto j3 = downloadBuffer<FemFemContactInfoHost>(dev, bI3.h, 4);
        const auto e1 = downloadBuffer<F4>(dev, bA1.h, 4);
        PXG_CHECK_MSG(back == 2u, "an over-large contact count must be CLAMPED and written back "
                                  "(9 -> maxNbContacts 2), got %u", back);
        PXG_CHECK_MSG(j3[0].pairInd1 == infos[0].pairInd1,
                      "prevContactCount = 1 must SKIP contact 0 (pairInd1 became 0x%08x)",
                      j3[0].pairInd1);
        PXG_CHECK_MSG(e1[0].x == bary1[0].x,
                      "prevContactCount = 1 must leave contact 0's barycentric alone (got %.4f)",
                      e1[0].x);
        PXG_CHECK_MSG((j3[1].pairInd1 & 0xfffffu) == 1u,
                      "...and must still remap contact 1 (got tet %u)", j3[1].pairInd1 & 0xfffffu);
        dev->destroyBuffer(bI3.h); dev->destroyBuffer(bA0.h); dev->destroyBuffer(bA1.h);
        dev->destroyBuffer(bC3.h); dev->destroyBuffer(bPrev.h);
    }

    dev->destroyBuffer(bSv.h); dev->destroyBuffer(bSt.h); dev->destroyBuffer(bRc.h);
    dev->destroyBuffer(bAr.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bInfo.h);
    dev->destroyBuffer(bB0.h); dev->destroyBuffer(bB1.h); dev->destroyBuffer(bCount.h);
}

// ===========================================================================
// P8.4 tranche b — the SOFTBODY-vs-SOFTBODY constraint prepare and solve.
//
// CPU reference.  The two kernels are short but every term is a place a symmetric port can
// go wrong in a way that still looks plausible: a sign, a barycentric scale, which of the
// two sides an inverse mass came from.  Signs and ratios cannot catch those -- P8.3 learned
// that the hard way, where mirroring the input flipped both runs alike and the product
// stayed negative -- so both fixtures compare against an ABSOLUTE reference computed here.
// The CPU side is built with -ffp-contract=off (tests/CMakeLists.txt:1204) and the shaders
// are NoContraction-stamped, so the comparisons below are bit-exact apart from the one
// tangent normalise.
// ===========================================================================
namespace p84ref
{
using p80refit::V3;
using p80refit::TetIdx4;

static V3 v3(float x, float y, float z) { return V3{ x, y, z }; }
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 mul3(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// getSoftBodyInvMass (softBody.cuh:38) — the blended inverse mass scaled by the LENGTH of
// the barycentric vector.  The same factor P8.3 found in the attachment prepare.
static float softBodyInvMassScaled(float baryMass, F4 b)
{
    const float scale = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w);
    return baryMass * scale;
}

// computeInvMass (softBody.cu:608).
static float computeInvMass(TetIdx4 tet, const std::vector<F4>& pos, F4 b)
{
    const float blended = pos[tet.x].w * b.x + pos[tet.y].w * b.y +
                          pos[tet.z].w * b.z + pos[tet.w].w * b.w;
    return softBodyInvMassScaled(blended, b);
}

// computeTetraContact (softBody.cuh:61).
static F4 computeTetraContact(const std::vector<F4>& vals, TetIdx4 tet, F4 b, F4& invMass)
{
    const F4 v0 = vals[tet.x], v1 = vals[tet.y], v2 = vals[tet.z], v3 = vals[tet.w];
    invMass = F4{ v0.w, v1.w, v2.w, v3.w };
    return F4{ v0.x * b.x + v1.x * b.y + v2.x * b.z + v3.x * b.w,
               v0.y * b.x + v1.y * b.y + v2.y * b.z + v3.y * b.w,
               v0.z * b.x + v1.z * b.y + v2.z * b.z + v3.z * b.w,
               v0.w * b.x + v1.w * b.y + v2.w * b.z + v3.w * b.w };
}

// PxVec3::normalize — returns the LENGTH and leaves the vector unit.
static float normalizeInPlace(V3& v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f) { const float len = std::sqrt(m); v = mul3(v, 1.0f / len); return len; }
    return 0.0f;
}

// updateTetraPosDelta (softBody.cuh:77) — accumulated into a host mirror of mSimDelta.
static void updateTetraPosDelta(F4 invMasses, F4 b, TetIdx4 tet, V3 deltaPos,
                                std::vector<F4>& outDeltas)
{
    const uint32_t idx[4] = { tet.x, tet.y, tet.z, tet.w };
    const float im[4] = { invMasses.x, invMasses.y, invMasses.z, invMasses.w };
    const float bc[4] = { b.x, b.y, b.z, b.w };
    for (int k = 0; k < 4; ++k)
    {
        if (!(im[k] > 0.0f && std::fabs(bc[k]) > 1e-6f)) continue;
        const V3 dP = mul3(deltaPos, im[k] * bc[k]);
        outDeltas[idx[k]].x += dP.x;
        outDeltas[idx[k]].y += dP.y;
        outDeltas[idx[k]].z += dP.z;
        outDeltas[idx[k]].w += 1.0f;   // the averaging weight
    }
}

// One softbody's inputs, as the two kernels read them.
struct SbSide
{
    std::vector<F4> simPos;      // mSimPosition_InvMass
    std::vector<F4> simVel;      // mSimVelocity_InvMass
    std::vector<F4> simDeltaPos; // mSimDeltaPos  (TGS reads this instead)
    std::vector<TetIdx4> tets;
    std::vector<uint16_t> matIdx;
    float restDistance = 0.0f;
};

// sb_solveOutputSSDeltaVLaunch{,TGS} (softBody.cu:1424 / :1691), one contact.
static void refSolveSS(bool isTGS, const SbSide& s0, const SbSide& s1,
                       uint32_t tetId0, uint32_t tetId1, F4 b0, F4 b1,
                       F4 normal_pen, float velMultiplier,
                       const std::vector<float>& matFriction,
                       float dt, float biasCoefficient,
                       float& afx, float& afy,
                       std::vector<F4>& delta0, std::vector<F4>& delta1)
{
    const TetIdx4 t0 = s0.tets[tetId0], t1 = s1.tets[tetId1];

    F4 invMasses0, invMasses1;
    const F4 pos0 = computeTetraContact(isTGS ? s0.simDeltaPos : s0.simPos, t0, b0, invMasses0);
    const F4 pos1 = computeTetraContact(isTGS ? s1.simDeltaPos : s1.simPos, t1, b1, invMasses1);

    V3 relVel = v3(0, 0, 0);
    if (!isTGS)
    {
        F4 vm0, vm1;
        const F4 vel0 = computeTetraContact(s0.simVel, t0, b0, vm0);
        const F4 vel1 = computeTetraContact(s1.simVel, t1, b1, vm1);
        invMasses0 = vm0; invMasses1 = vm1;
        relVel = sub3(v3(vel1.x, vel1.y, vel1.z), v3(vel0.x, vel0.y, vel0.z));
    }

    const float frictionCoefficient =
        (matFriction[s0.matIdx[tetId0]] + matFriction[s1.matIdx[tetId1]]) * 0.5f;

    const V3 normal = v3(normal_pen.x, normal_pen.y, normal_pen.z);
    const float rest = normal_pen.w;
    const V3 relPos = v3(pos1.x - pos0.x, pos1.y - pos0.y, pos1.z - pos0.z);
    const float projectedErr = dot3(relPos, normal) + rest;

    V3 tanDir; float tanMagnitude, biasedErr, normalVel;
    if (isTGS)
    {
        const float normalPos = dot3(relPos, normal);
        tanDir = sub3(relPos, mul3(normal, normalPos));
        tanMagnitude = normalizeInPlace(tanDir);
        biasedErr = -projectedErr;
        normalVel = 0.0f;
    }
    else
    {
        normalVel = dot3(relVel, normal);
        tanDir = sub3(relVel, mul3(normal, normalVel));
        tanMagnitude = normalizeInPlace(tanDir);
        biasedErr = -projectedErr * biasCoefficient;
    }

    const float deltaF = std::fmax(-afx, (biasedErr - normalVel) * velMultiplier);
    afx += deltaF;
    const float friction = afx * frictionCoefficient;
    const float requiredForce = tanMagnitude * velMultiplier;
    const float deltaFr = std::fmin(requiredForce + afy, friction) - afy;
    afy += deltaFr;

    if (deltaF != 0.0f || deltaFr != 0.0f)
    {
        V3 deltaPos = mul3(add3(mul3(normal, -deltaF), mul3(tanDir, deltaFr)), dt);
        if (isTGS)
            deltaPos = mul3(mul3(add3(mul3(normal, -deltaF), mul3(tanDir, deltaFr)), biasCoefficient), dt);
        updateTetraPosDelta(invMasses0, b0, t0, deltaPos, delta0);
        updateTetraPosDelta(invMasses1, b1, t1, mul3(deltaPos, -1.0f), delta1);
    }
}
} // namespace p84ref

struct SbSsPrepPush
{
    uint64_t softbodies, contacts, normalPens, barycentrics0, barycentrics1,
             contactInfos, numContacts, constraints, appliedForces;
    uint32_t maxContacts, pad76;
};
static_assert(sizeof(SbSsPrepPush) == 80, "ss prepare push blob must be 80 bytes");
static_assert(offsetof(SbSsPrepPush, maxContacts) == 72, "maxContacts @72");

struct SbSsSolvePush
{
    uint64_t softbodies, contactInfos, constraints, numContacts;
    float dt, biasCoefficient;
    uint64_t appliedForces, materials;
};
static_assert(sizeof(SbSsSolvePush) == 56, "ss solve push blob must be 56 bytes");
static_assert(offsetof(SbSsSolvePush, dt) == 32, "dt @32");
static_assert(offsetof(SbSsSolvePush, appliedForces) == 40, "appliedForces @40");

// PxgSoftBodySoftBodyConstraintBlock (PxgSoftBodyCore.h:51), block-of-32 SoA, 1792 bytes.
struct SsConstraintBlockHost
{
    F4 barycentric0[32];
    F4 barycentric1[32];
    F4 normal_pen[32];
    float velMultiplier[32];
    float friction[32];
};
static_assert(sizeof(SsConstraintBlockHost) == 1792, "SS constraint block is 1792 bytes");

// ===========================================================================
// sb_softbodyContactPrepareLaunch.
//
// What this pins: the CONSTRAINT BLOCK's block-of-32 SoA addressing (a contact's slot is
// decided by its INDEX, not by the thread that happens to run it), the negated normal and
// the folded rest distance, `getSoftBodyInvMass`'s barycentric-length scale on BOTH sides,
// and the 1e-16 guard that keeps an all-kinematic contact from dividing by zero.
// ===========================================================================
PXG_TEST(softBodySimSelfContactPrepare)
{
    using namespace p80refit;
    using namespace p84ref;
    PxgSlangComputeDevice* dev = testDevice();

    // TWO softbodies with DIFFERENT rest distances and different inverse masses, so a port
    // that read both sides from one descriptor cannot reproduce these numbers.
    // Softbody 0: tet 0 has invMasses (1,2,3,4); tet 1 is all-KINEMATIC (invMass 0).
    std::vector<F4> pos0 = {
        { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 2.0f },
        { 0.0f, 1.0f, 0.0f, 3.0f }, { 0.0f, 0.0f, 1.0f, 4.0f },   // tet 0
        { 5.0f, 0.0f, 0.0f, 0.0f }, { 6.0f, 0.0f, 0.0f, 0.0f },
        { 5.0f, 1.0f, 0.0f, 0.0f }, { 5.0f, 0.0f, 1.0f, 0.0f },   // tet 1, kinematic
    };
    std::vector<TetIdx4> tets0 = { { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u } };
    // Softbody 1: tet 0 has invMasses all 0.5; tet 1 is all-KINEMATIC too.
    std::vector<F4> pos1 = {
        { 0.0f, 0.0f, 2.0f, 0.5f }, { 1.0f, 0.0f, 2.0f, 0.5f },
        { 0.0f, 1.0f, 2.0f, 0.5f }, { 0.0f, 0.0f, 3.0f, 0.5f },   // tet 0
        { 9.0f, 0.0f, 0.0f, 0.0f }, { 10.0f, 0.0f, 0.0f, 0.0f },
        { 9.0f, 1.0f, 0.0f, 0.0f }, { 9.0f, 0.0f, 1.0f, 0.0f },   // tet 1, kinematic
    };
    std::vector<TetIdx4> tets1 = { { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u } };

    DevBuf bP0 = makeDev(dev, pos0), bT0 = makeDev(dev, tets0);
    DevBuf bP1 = makeDev(dev, pos1), bT1 = makeDev(dev, tets1);
    if (!bP0.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody a = makeZeroDesc();
    a.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bP0.addr));
    a.mSimTetIndices = reinterpret_cast<decltype(a.mSimTetIndices)>(size_t(bT0.addr));
    a.mRestDistance = 0.05f;
    PxgSoftBody b = makeZeroDesc();
    b.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bP1.addr));
    b.mSimTetIndices = reinterpret_cast<decltype(b.mSimTetIndices)>(size_t(bT1.addr));
    b.mRestDistance = 0.03f;
    std::vector<PxgSoftBody> descArr = { a, b };
    DevBuf bDesc = makeDev(dev, descArr);

    auto enc = [](uint32_t id, uint32_t elem) { return (id << 20) | elem; };

    // 34 contacts, so the constraint array spans TWO blocks of 32 and the last one lands at
    // block 1 lane 1.  Contact 1 uses the two KINEMATIC tets and is the only one whose
    // inverse masses sum to zero.
    const uint32_t NBC = 34u;
    const F4 BARY0 = { 0.4f, 0.2f, 0.2f, 0.2f };
    const F4 BARY1 = { 0.25f, 0.25f, 0.25f, 0.25f };
    std::vector<FemFemContactInfoHost> infos(NBC);
    std::vector<F4> normalPens(NBC), bary0(NBC, BARY0), bary1(NBC, BARY1);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemFemContactInfoHost));
    for (uint32_t i = 0; i < NBC; ++i)
    {
        const uint32_t kin = (i == 1u) ? 1u : 0u;
        infos[i].pairInd0 = enc(0u, kin);
        infos[i].pairInd1 = enc(1u, kin);
        infos[i].auxInd = 0x8000u | i;
        // A distinct penetration per contact, so a slot that ends up in the wrong block or
        // lane is identifiable rather than merely "wrong".
        normalPens[i] = F4{ 0.0f, 1.0f, 0.0f, 0.5f - 0.001f * float(i) };
    }
    DevBuf bInfo = makeDev(dev, infos), bNp = makeDev(dev, normalPens);
    DevBuf bB0 = makeDev(dev, bary0), bB1 = makeDev(dev, bary1);
    std::vector<uint32_t> count = { NBC, 0u, 0u, 0u };
    DevBuf bCount = makeDev(dev, count);

    std::vector<SsConstraintBlockHost> blocks(2);
    std::memset(blocks.data(), 0xCD, blocks.size() * sizeof(SsConstraintBlockHost));
    DevBuf bBlocks = makeDev(dev, blocks);
    std::vector<float> applied(NBC * 2u, -3.0f);   // float2 per contact, poisoned
    DevBuf bApplied = makeDev(dev, applied);

    SbSsPrepPush push = {};
    push.softbodies = bDesc.addr;
    push.contacts = 0ull;                 // ABI parity only — never read
    push.normalPens = bNp.addr;
    push.barycentrics0 = bB0.addr;
    push.barycentrics1 = bB1.addr;
    push.contactInfos = bInfo.addr;
    push.numContacts = bCount.addr;
    push.constraints = bBlocks.addr;
    push.appliedForces = bApplied.addr;
    push.maxContacts = 1024u;

    Kernel k("softBodySelfSolveBda.spv", "sb_softbodyContactPrepareLaunch", {}, sizeof(SbSsPrepPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);   // PxgSoftBodyKernelGridDim::SB_UPDATEROTATION
    dev->flush();

    const auto gBlocks = downloadBuffer<SsConstraintBlockHost>(dev, bBlocks.h, 2);
    const auto gApplied = downloadBuffer<float>(dev, bApplied.h, NBC * 2u);

    // ---- the reference, contact by contact -----------------------------------
    const float REST = 0.05f + 0.03f;
    for (uint32_t i = 0; i < NBC; ++i)
    {
        const uint32_t blk = i / 32u, lane = i % 32u;
        const uint32_t kin = (i == 1u) ? 1u : 0u;
        const float invMass0 = computeInvMass(tets0[kin], pos0, BARY0);
        const float invMass1 = computeInvMass(tets1[kin], pos1, BARY1);
        const float wantVm = 1.0f / std::fmax(1e-16f, invMass0 + invMass1);
        const F4 np = normalPens[i];

        const F4 gN = gBlocks[blk].normal_pen[lane];
        PXG_CHECK_MSG(f2u(gN.x) == f2u(-np.x) && f2u(gN.y) == f2u(-np.y) && f2u(gN.z) == f2u(-np.z),
                      "contact %u (block %u lane %u): the prepare must NEGATE the contact "
                      "normal, got (%g %g %g) for input (%g %g %g)",
                      i, blk, lane, gN.x, gN.y, gN.z, np.x, np.y, np.z);
        PXG_CHECK_MSG(f2u(gN.w) == f2u(-REST + np.w),
                      "contact %u: normal_pen.w must be -rest + pen = %.9g, got %.9g (a slot in "
                      "the wrong block or lane shows up here first)",
                      i, -REST + np.w, gN.w);
        PXG_CHECK_MSG(f2u(gBlocks[blk].velMultiplier[lane]) == f2u(wantVm),
                      "contact %u: velMultiplier must be 1/max(1e-16, %.9g + %.9g) = %.9g, "
                      "got %.9g", i, invMass0, invMass1, wantVm, gBlocks[blk].velMultiplier[lane]);
        PXG_CHECK(f2u(gBlocks[blk].barycentric0[lane].x) == f2u(BARY0.x));
        PXG_CHECK(f2u(gBlocks[blk].barycentric1[lane].w) == f2u(BARY1.w));
        PXG_CHECK_MSG(gApplied[i * 2u] == 0.0f && gApplied[i * 2u + 1u] == 0.0f,
                      "contact %u's applied force must be ZEROED by the prepare (got %g, %g)",
                      i, gApplied[i * 2u], gApplied[i * 2u + 1u]);
    }

    // ---- the barycentric-length scale, stated as a number ---------------------
    // |BARY0| = sqrt(0.28) = 0.529150, so dropping the sqrt in getSoftBodyInvMass changes
    // invMass0 from 2.2 * 0.529150 to 2.2 -- a factor of ~1.9 on the velMultiplier.  This is
    // the same term P8.3 found in the attachment prepare and it has no cloth counterpart.
    {
        const float blended0 = pos0[0].w * BARY0.x + pos0[1].w * BARY0.y +
                               pos0[2].w * BARY0.z + pos0[3].w * BARY0.w;
        const float scaled0 = softBodyInvMassScaled(blended0, BARY0);
        std::printf("  invMass0: blended %.6f -> scaled %.6f (|bary| = %.6f); velMultiplier[0] = %.6f\n",
                    blended0, scaled0, scaled0 / blended0, gBlocks[0].velMultiplier[0]);
        PXG_CHECK_MSG(std::fabs(scaled0 / blended0 - 0.5291503f) < 1e-5f,
                      "the fixture's barycentric must actually exercise the scale");
    }

    // ---- the 1e-16 guard ------------------------------------------------------
    PXG_CHECK_MSG(gBlocks[0].velMultiplier[1] == 1e16f,
                  "contact 1 joins two all-kinematic tets, so both inverse masses are 0 and "
                  "the guard must give 1/1e-16 = 1e16 (got %g) -- the contact generation DOES "
                  "produce such contacts, so this is reachable, not defensive",
                  gBlocks[0].velMultiplier[1]);

    // ---- the clamp write-back -------------------------------------------------
    {
        std::vector<uint32_t> c2 = { 99u, 0u, 0u, 0u };
        DevBuf bC2 = makeDev(dev, c2);
        SbSsPrepPush p2 = push;
        p2.numContacts = bC2.addr;
        p2.maxContacts = 5u;
        Kernel("softBodySelfSolveBda.spv", "sb_softbodyContactPrepareLaunch", {}, sizeof(SbSsPrepPush))
            .dispatch(1024u, {}, p2, 1u, 1u);
        dev->flush();
        const uint32_t back = downloadBuffer<uint32_t>(dev, bC2.h, 1)[0];
        PXG_CHECK_MSG(back == 5u,
                      "an over-large contact count must be clamped to maxContacts and written "
                      "back (99 -> 5), got %u", back);
        dev->destroyBuffer(bC2.h);
    }

    dev->destroyBuffer(bP0.h); dev->destroyBuffer(bT0.h); dev->destroyBuffer(bP1.h);
    dev->destroyBuffer(bT1.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bInfo.h);
    dev->destroyBuffer(bNp.h); dev->destroyBuffer(bB0.h); dev->destroyBuffer(bB1.h);
    dev->destroyBuffer(bCount.h); dev->destroyBuffer(bBlocks.h); dev->destroyBuffer(bApplied.h);
}

// ===========================================================================
// sb_solveOutputSSDeltaVLaunch{,TGS}.
//
// The symmetric twin of sb_solveRigidSoftCollisionLaunch: both sides are softbody tets, so
// the SAME impulse goes into both delta buffers with opposite signs and each side scales it
// by its OWN inverse masses and barycentric.  Everything is compared to p84ref's absolute
// reference rather than to a sign or a ratio -- a mirrored fixture flips both runs alike and
// would let a dropped negation through, which is exactly how P8.3's solve mutant survived
// its first fixture.
//
// The two entry points are checked against SEPARATE reference runs and against each other:
// PGS and TGS differ in four places (which buffer supplies the positions, whether the error
// is bias-scaled, whether the normal impulse sees the relative velocity, and whether the
// tangential magnitude is a velocity or a position), so a TGS entry point that is merely a
// copy of PGS has to disagree with the TGS reference.
// ===========================================================================
struct DeformableMaterialHost   // PxsDeformableVolumeMaterialData, 32 B
{
    float youngs, poissons, dynamicFriction, elasticityDamping;
    uint16_t dampingScale, materialModel;
    float deformThreshold, deformLowLimitRatio, deformHighLimitRatio;
};
static_assert(sizeof(DeformableMaterialHost) == 32, "PxsDeformableVolumeMaterialData is 32 bytes");
static_assert(offsetof(DeformableMaterialHost, dynamicFriction) == 8, "dynamicFriction @8");

PXG_TEST(softBodySimSelfContactSolve)
{
    using namespace p80refit;
    using namespace p84ref;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- two softbodies, two tets each ---------------------------------------
    SbSide s0, s1;
    s0.simPos = {
        { 0.00f, 0.00f, 0.00f, 1.0f }, { 1.00f, 0.00f, 0.00f, 2.0f },
        { 0.00f, 1.00f, 0.00f, 0.0f }, { 0.00f, 0.00f, 1.00f, 4.0f },   // tet 0 (one KINEMATIC corner)
        { 4.00f, 0.00f, 0.00f, 1.5f }, { 5.00f, 0.00f, 0.00f, 0.5f },
        { 4.00f, 1.00f, 0.00f, 2.5f }, { 4.00f, 0.00f, 1.00f, 3.5f },   // tet 1
    };
    // The .w column DELIBERATELY DIFFERS from mSimPosition_InvMass's.  In the real engine
    // the two hold the same inverse mass, so the difference is unobservable there -- but
    // CUDA's PGS path takes its inverse masses from the VELOCITY read (softBody.cu:1471-6,
    // where the second computeTetraContact overwrites the out-parameter), and a fixture
    // whose two columns agree cannot tell a port that read the position column instead.
    // The kinematic corner keeps invMass 0 in BOTH so it stays kinematic either way.
    s0.simVel = {
        { 0.10f, -0.20f, 0.05f, 1.25f }, { 0.20f, -0.10f, 0.00f, 2.50f },
        { 0.00f, -0.30f, 0.10f, 0.00f }, { 0.15f, -0.25f, 0.20f, 3.50f },
        { 0.05f,  0.10f, 0.00f, 1.75f }, { 0.00f,  0.20f, 0.10f, 0.25f },
        { 0.10f,  0.00f, 0.30f, 2.75f }, { 0.20f,  0.15f, 0.05f, 3.25f },
    };
    // TGS reads mSimDeltaPos for BOTH the position and the inverse masses, so its .w column
    // is deliberately DIFFERENT from mSimPosition_InvMass's: a TGS port that read the
    // position buffer instead would silently pick up the other column.
    s0.simDeltaPos = {
        { 0.010f, -0.020f, 0.005f, 0.7f }, { 0.020f, -0.010f, 0.000f, 1.3f },
        { 0.000f, -0.030f, 0.010f, 0.0f }, { 0.015f, -0.025f, 0.020f, 2.1f },
        { 0.005f,  0.010f, 0.000f, 0.9f }, { 0.000f,  0.020f, 0.010f, 0.4f },
        { 0.010f,  0.000f, 0.030f, 1.7f }, { 0.020f,  0.015f, 0.005f, 2.3f },
    };
    s0.tets = { { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u } };
    s0.matIdx = { 0u, 3u };            // tet 1 is an ODD index -> the PxU16 high half
    s0.restDistance = 0.05f;

    s1.simPos = {
        { 0.00f, 0.00f, 2.00f, 0.5f }, { 1.00f, 0.00f, 2.00f, 1.0f },
        { 0.00f, 1.00f, 2.00f, 1.5f }, { 0.00f, 0.00f, 3.00f, 0.0f },   // tet 0
        { 8.00f, 0.00f, 0.00f, 2.0f }, { 9.00f, 0.00f, 0.00f, 1.0f },
        { 8.00f, 1.00f, 0.00f, 0.5f }, { 8.00f, 0.00f, 1.00f, 1.0f },   // tet 1
    };
    s1.simVel = {
        { -0.05f, 0.30f, 0.00f, 0.75f }, {  0.00f, 0.25f, 0.10f, 1.25f },
        {  0.10f, 0.40f, 0.05f, 1.75f }, { -0.10f, 0.35f, 0.15f, 0.00f },
        {  0.20f, 0.05f, 0.10f, 2.25f }, {  0.10f, 0.00f, 0.20f, 1.25f },
        {  0.00f, 0.15f, 0.05f, 0.75f }, {  0.05f, 0.10f, 0.00f, 1.25f },
    };
    s1.simDeltaPos = {
        { -0.005f, 0.030f, 0.000f, 1.1f }, {  0.000f, 0.025f, 0.010f, 0.6f },
        {  0.010f, 0.040f, 0.005f, 2.2f }, { -0.010f, 0.035f, 0.015f, 0.0f },
        {  0.020f, 0.005f, 0.010f, 1.4f }, {  0.010f, 0.000f, 0.020f, 0.8f },
        {  0.000f, 0.015f, 0.005f, 1.9f }, {  0.005f, 0.010f, 0.000f, 1.2f },
    };
    s1.tets = { { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u } };
    s1.matIdx = { 1u, 2u };
    s1.restDistance = 0.03f;

    // Four materials with distinct friction, so the (f0 + f1) * 0.5 mean is identifiable and
    // a port that used just one side's material, or the wrong PxU16 half, cannot match.
    std::vector<DeformableMaterialHost> materials(4);
    std::memset(materials.data(), 0, materials.size() * sizeof(DeformableMaterialHost));
    const std::vector<float> matFriction = { 0.30f, 0.70f, 0.11f, 0.90f };
    for (size_t i = 0; i < materials.size(); ++i) materials[i].dynamicFriction = matFriction[i];

    auto packU16 = [](const std::vector<uint16_t>& h) {
        std::vector<uint32_t> w((h.size() + 1u) / 2u + 1u, 0u);
        for (size_t i = 0; i < h.size(); ++i) w[i / 2u] |= uint32_t(h[i]) << (16u * (i % 2u));
        return w;
    };
    const std::vector<uint32_t> mat0Words = packU16(s0.matIdx);
    const std::vector<uint32_t> mat1Words = packU16(s1.matIdx);

    DevBuf bP0 = makeDev(dev, s0.simPos), bV0 = makeDev(dev, s0.simVel), bD0 = makeDev(dev, s0.simDeltaPos);
    DevBuf bT0 = makeDev(dev, s0.tets), bM0 = makeDev(dev, mat0Words);
    DevBuf bP1 = makeDev(dev, s1.simPos), bV1 = makeDev(dev, s1.simVel), bD1 = makeDev(dev, s1.simDeltaPos);
    DevBuf bT1 = makeDev(dev, s1.tets), bM1 = makeDev(dev, mat1Words);
    DevBuf bMat = makeDev(dev, materials);
    if (!bP0.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    std::vector<F4> zeroDelta(8, F4{ 0, 0, 0, 0 });
    DevBuf bDl0 = makeDev(dev, zeroDelta), bDl1 = makeDev(dev, zeroDelta);

    PxgSoftBody a = makeZeroDesc();
    a.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bP0.addr));
    a.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bV0.addr));
    a.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bD0.addr));
    a.mSimDelta = reinterpret_cast<float4*>(size_t(bDl0.addr));
    a.mSimTetIndices = reinterpret_cast<decltype(a.mSimTetIndices)>(size_t(bT0.addr));
    a.mMaterialIndices = reinterpret_cast<PxU16*>(size_t(bM0.addr));
    a.mRestDistance = s0.restDistance;
    PxgSoftBody b = makeZeroDesc();
    b.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bP1.addr));
    b.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(bV1.addr));
    b.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bD1.addr));
    b.mSimDelta = reinterpret_cast<float4*>(size_t(bDl1.addr));
    b.mSimTetIndices = reinterpret_cast<decltype(b.mSimTetIndices)>(size_t(bT1.addr));
    b.mMaterialIndices = reinterpret_cast<PxU16*>(size_t(bM1.addr));
    b.mRestDistance = s1.restDistance;
    std::vector<PxgSoftBody> descArr = { a, b };
    DevBuf bDesc = makeDev(dev, descArr);

    auto enc = [](uint32_t id, uint32_t elem) { return (id << 20) | elem; };

    // Two contacts, deliberately CROSSED over the tets so both PxU16 halves of both
    // material tables are read and neither side can stand in for the other.
    const uint32_t NBC = 2u;
    const uint32_t tetId0[NBC] = { 0u, 1u };
    const uint32_t tetId1[NBC] = { 1u, 0u };
    // Contact 1's side-0 barycentric has a ZERO component: a contact sitting on a tet FACE
    // weighs the opposite corner at 0, and updateTetraPosDelta's `|bary| > 1e-6` gate is
    // what keeps that corner out of the averaging count.  Without such a component the gate
    // is unexercised and deleting it passes.
    const F4 bary0[NBC] = { { 0.4f, 0.2f, 0.2f, 0.2f }, { 0.1f, 0.0f, 0.7f, 0.2f } };
    const F4 bary1[NBC] = { { 0.25f, 0.25f, 0.25f, 0.25f }, { 0.5f, 0.2f, 0.2f, 0.1f } };
    // Contact 1's normal points so that the two tets are strongly SEPARATING.  That makes
    // the raw normal impulse large and negative, which is the only way to reach
    // `deltaF = max(-appliedForce.x, ...)`: with a starting force of 0.02 the clamp releases
    // exactly that much and no more.  Without a separating contact the clamp never bites and
    // deleting it passes.
    const F4 normalPen[NBC] = { { 0.0f, -1.0f, 0.0f, -0.02f }, { 0.0f, 0.0f, 1.0f, -0.05f } };
    const float velMult[NBC] = { 0.65f, 1.30f };
    const float AF_X0 = 0.02f, AF_Y0 = 0.01f;   // a NON-zero starting force: the solve
                                                // accumulates, it does not assign

    std::vector<FemFemContactInfoHost> infos(NBC);
    std::memset(infos.data(), 0, infos.size() * sizeof(FemFemContactInfoHost));
    std::vector<SsConstraintBlockHost> blocks(1);
    std::memset(blocks.data(), 0, sizeof(SsConstraintBlockHost));
    std::vector<float> applied(NBC * 2u);
    for (uint32_t i = 0; i < NBC; ++i)
    {
        infos[i].pairInd0 = enc(0u, tetId0[i]);
        infos[i].pairInd1 = enc(1u, tetId1[i]);
        infos[i].auxInd = 0x8000u | i;
        blocks[0].barycentric0[i] = bary0[i];
        blocks[0].barycentric1[i] = bary1[i];
        blocks[0].normal_pen[i] = normalPen[i];
        blocks[0].velMultiplier[i] = velMult[i];
        applied[i * 2u] = AF_X0;
        applied[i * 2u + 1u] = AF_Y0;
    }

    const float DT = 0.0166667f, BIAS = 0.6f;

    auto runOne = [&](bool isTGS, std::vector<F4>& outD0, std::vector<F4>& outD1,
                      std::vector<float>& outAf)
    {
        std::vector<FemFemContactInfoHost> ii = infos;
        std::vector<SsConstraintBlockHost> bb = blocks;
        std::vector<float> af = applied;
        std::vector<F4> d0 = zeroDelta, d1 = zeroDelta;
        DevBuf bI = makeDev(dev, ii), bB = makeDev(dev, bb), bA = makeDev(dev, af);
        DevBuf bX0 = makeDev(dev, d0), bX1 = makeDev(dev, d1);
        std::vector<uint32_t> cnt = { NBC, 0u, 0u, 0u };
        DevBuf bC = makeDev(dev, cnt);

        // Re-point the descriptors' delta buffers at this run's own copies.
        std::vector<PxgSoftBody> dd = descArr;
        dd[0].mSimDelta = reinterpret_cast<float4*>(size_t(bX0.addr));
        dd[1].mSimDelta = reinterpret_cast<float4*>(size_t(bX1.addr));
        DevBuf bDd = makeDev(dev, dd);

        SbSsSolvePush p = {};
        p.softbodies = bDd.addr;
        p.contactInfos = bI.addr;
        p.constraints = bB.addr;
        p.numContacts = bC.addr;
        p.dt = DT;
        p.biasCoefficient = BIAS;
        p.appliedForces = bA.addr;
        p.materials = bMat.addr;

        Kernel(isTGS ? "softBodySelfSolveBda.spv" : "softBodySelfSolveBda.spv",
               isTGS ? "sb_solveOutputSSDeltaVLaunchTGS" : "sb_solveOutputSSDeltaVLaunch",
               {}, sizeof(SbSsSolvePush))
            .dispatch(/*gridX*/ 1024u, {}, p, 1u, 1u);
        dev->flush();

        outD0 = downloadBuffer<F4>(dev, bX0.h, 8);
        outD1 = downloadBuffer<F4>(dev, bX1.h, 8);
        outAf = downloadBuffer<float>(dev, bA.h, NBC * 2u);
        dev->destroyBuffer(bI.h); dev->destroyBuffer(bB.h); dev->destroyBuffer(bA.h);
        dev->destroyBuffer(bX0.h); dev->destroyBuffer(bX1.h); dev->destroyBuffer(bC.h);
        dev->destroyBuffer(bDd.h);
    };

    auto reference = [&](bool isTGS, std::vector<F4>& refD0, std::vector<F4>& refD1,
                         std::vector<float>& refAf)
    {
        refD0.assign(8, F4{ 0, 0, 0, 0 });
        refD1.assign(8, F4{ 0, 0, 0, 0 });
        refAf.assign(NBC * 2u, 0.0f);
        for (uint32_t i = 0; i < NBC; ++i)
        {
            float afx = AF_X0, afy = AF_Y0;
            refSolveSS(isTGS, s0, s1, tetId0[i], tetId1[i], bary0[i], bary1[i],
                       normalPen[i], velMult[i], matFriction, DT, BIAS, afx, afy, refD0, refD1);
            refAf[i * 2u] = afx;
            refAf[i * 2u + 1u] = afy;
        }
    };

    auto near6 = [](float a, float b) { return std::fabs(a - b) <= 1e-6f * std::fmax(1.0f, std::fabs(b)); };

    std::vector<F4> pgsD0, pgsD1, tgsD0, tgsD1;
    std::vector<float> pgsAf, tgsAf;
    runOne(/*isTGS*/ false, pgsD0, pgsD1, pgsAf);
    runOne(/*isTGS*/ true, tgsD0, tgsD1, tgsAf);

    for (int pass = 0; pass < 2; ++pass)
    {
        const bool isTGS = (pass == 1);
        const char* tag = isTGS ? "TGS" : "PGS";
        std::vector<F4> rD0, rD1; std::vector<float> rAf;
        reference(isTGS, rD0, rD1, rAf);
        const std::vector<F4>& gD0 = isTGS ? tgsD0 : pgsD0;
        const std::vector<F4>& gD1 = isTGS ? tgsD1 : pgsD1;
        const std::vector<float>& gAf = isTGS ? tgsAf : pgsAf;

        std::printf("  %s: delta0[0]=(%.7f %.7f %.7f w=%.1f) delta1[0]=(%.7f %.7f %.7f w=%.1f) "
                    "af0=(%.7f %.7f)\n", tag,
                    gD0[0].x, gD0[0].y, gD0[0].z, gD0[0].w,
                    gD1[0].x, gD1[0].y, gD1[0].z, gD1[0].w, gAf[0], gAf[1]);

        for (uint32_t v = 0; v < 8u; ++v)
        {
            PXG_CHECK_MSG(near6(gD0[v].x, rD0[v].x) && near6(gD0[v].y, rD0[v].y) &&
                          near6(gD0[v].z, rD0[v].z) && gD0[v].w == rD0[v].w,
                          "%s side 0 vertex %u: got (%.8f %.8f %.8f w=%.1f), want "
                          "(%.8f %.8f %.8f w=%.1f)", tag, v,
                          gD0[v].x, gD0[v].y, gD0[v].z, gD0[v].w,
                          rD0[v].x, rD0[v].y, rD0[v].z, rD0[v].w);
            PXG_CHECK_MSG(near6(gD1[v].x, rD1[v].x) && near6(gD1[v].y, rD1[v].y) &&
                          near6(gD1[v].z, rD1[v].z) && gD1[v].w == rD1[v].w,
                          "%s side 1 vertex %u: got (%.8f %.8f %.8f w=%.1f), want "
                          "(%.8f %.8f %.8f w=%.1f)", tag, v,
                          gD1[v].x, gD1[v].y, gD1[v].z, gD1[v].w,
                          rD1[v].x, rD1[v].y, rD1[v].z, rD1[v].w);
        }
        for (uint32_t i = 0; i < NBC * 2u; ++i)
            PXG_CHECK_MSG(near6(gAf[i], rAf[i]),
                          "%s applied force %u: got %.8f, want %.8f (the solve ACCUMULATES onto "
                          "the incoming force, it does not assign)", tag, i, gAf[i], rAf[i]);
    }

    // ---- the .w averaging weight ---------------------------------------------
    // AtomicAdd(float4&, PxVec3, 1.f) adds a FOURTH component that the later averaging pass
    // divides by.  Dropping it leaves the deltas looking right here and halves every motion
    // downstream, so it is checked explicitly rather than only through the reference.
    {
        uint32_t touched = 0u;
        for (uint32_t v = 0; v < 8u; ++v) if (pgsD0[v].w != 0.0f) ++touched;
        PXG_CHECK_MSG(touched > 0u, "at least one side-0 vertex must have received a delta");
        for (uint32_t v = 0; v < 8u; ++v)
            PXG_CHECK_MSG(pgsD0[v].w == 0.0f || pgsD0[v].w == 1.0f,
                          "each touched vertex gets exactly ONE unit of averaging weight per "
                          "contact (vertex %u has %g)", v, pgsD0[v].w);
    }

    // ---- the normal force may not go negative --------------------------------
    PXG_CHECK_MSG(pgsAf[2] == 0.0f,
                  "contact 1 is separating hard, so the normal impulse is clamped at exactly "
                  "-appliedForce.x and the accumulated force lands on 0 (got %g).  A normal "
                  "force is push-only; without the clamp it would go negative and PULL the "
                  "two tets together", pgsAf[2]);

    // ---- the ZERO-weight corner is skipped -----------------------------------
    // Contact 1 weighs side-0 tet 1's second corner (vertex 5) at exactly 0.  That vertex
    // has a POSITIVE inverse mass, so only the barycentric gate can keep it out -- and it
    // must, or the averaging pass divides a zero delta by a nonzero count.
    PXG_CHECK_MSG(pgsD0[5].w == 0.0f && pgsD0[5].x == 0.0f,
                  "side 0 vertex 5 is weighted 0 by its only contact and must be left alone "
                  "(got %g %g %g w=%g)", pgsD0[5].x, pgsD0[5].y, pgsD0[5].z, pgsD0[5].w);

    // ---- the KINEMATIC corner is skipped -------------------------------------
    // s0 tet 0 corner 2 and s1 tet 0 corner 3 both have invMass 0, and updateTetraPosDelta
    // gates on `invMasses > 0`.  A port that dropped that gate writes into a vertex the
    // solver is not allowed to move.
    PXG_CHECK_MSG(pgsD0[2].x == 0.0f && pgsD0[2].y == 0.0f && pgsD0[2].z == 0.0f && pgsD0[2].w == 0.0f,
                  "side 0 vertex 2 has invMass 0 and must NOT be moved (got %g %g %g w=%g)",
                  pgsD0[2].x, pgsD0[2].y, pgsD0[2].z, pgsD0[2].w);

    // ---- PGS and TGS must actually differ ------------------------------------
    bool differs = false;
    for (uint32_t v = 0; v < 8u && !differs; ++v)
        differs = (pgsD0[v].x != tgsD0[v].x) || (pgsD0[v].y != tgsD0[v].y) || (pgsD0[v].z != tgsD0[v].z);
    PXG_CHECK_MSG(differs,
                  "PGS and TGS differ in four places -- the position source, the bias scale, "
                  "the normal-velocity term and the tangential magnitude -- so identical "
                  "output means one entry point is a copy of the other");

    // ---- the symmetry, stated directly ---------------------------------------
    // Side 1 receives the NEGATED impulse.  Contact 0's side-1 tet is tet 1 (vertices 4..7),
    // all with positive inverse mass and positive barycentric, so every one of them must
    // have moved OPPOSITE to the side-0 impulse along the contact normal (0,-1,0).
    {
        float sum0 = 0.0f, sum1 = 0.0f;
        for (uint32_t v = 0; v < 4u; ++v) sum0 += pgsD0[v].y;      // side 0, tet 0
        for (uint32_t v = 4u; v < 8u; ++v) sum1 += pgsD1[v].y;     // side 1, tet 1
        std::printf("  symmetry: side0 tet0 sum dy = %.8f, side1 tet1 sum dy = %.8f\n", sum0, sum1);
        PXG_CHECK_MSG(sum0 * sum1 < 0.0f,
                      "the two sides of a contact must move in OPPOSITE directions "
                      "(%.8f and %.8f)", sum0, sum1);
    }

    dev->destroyBuffer(bP0.h); dev->destroyBuffer(bV0.h); dev->destroyBuffer(bD0.h);
    dev->destroyBuffer(bT0.h); dev->destroyBuffer(bM0.h); dev->destroyBuffer(bP1.h);
    dev->destroyBuffer(bV1.h); dev->destroyBuffer(bD1.h); dev->destroyBuffer(bT1.h);
    dev->destroyBuffer(bM1.h); dev->destroyBuffer(bMat.h); dev->destroyBuffer(bDl0.h);
    dev->destroyBuffer(bDl1.h); dev->destroyBuffer(bDesc.h);
}

// ===========================================================================
// P8.5 — sb_calculateStressLaunch, the co-rotated Cauchy stress field.
//
// Compared against an absolute CPU reference, bit-exact apart from the final sqrt: the
// kernel is two 3x3 products and a symmetrised difference, all contraction-free under the
// NoContraction stamp and -ffp-contract=off.  What the fixture aims at is the parts a
// plausible-looking port gets wrong silently: the co-rotation (transposing the WRONG
// matrix, or skipping it), the Lame reparameterisation order, the diagonal the scalar
// coefficient is built from, and the launch gate that decides whether any of it runs.
// ===========================================================================
namespace p85ref
{
struct M33 { float c[9]; };   // column-major: c[0..2] = column0, c[3..5] = column1, ...

static M33 m33FromCols(const float* c0, const float* c1, const float* c2)
{
    M33 m;
    for (int i = 0; i < 3; ++i) { m.c[i] = c0[i]; m.c[3 + i] = c1[i]; m.c[6 + i] = c2[i]; }
    return m;
}
// PxMat33::transform: column0 * v.x + column1 * v.y + column2 * v.z.
static void m33Transform(const M33& m, const float* v, float* out)
{
    for (int i = 0; i < 3; ++i) out[i] = m.c[i] * v[0] + m.c[3 + i] * v[1] + m.c[6 + i] * v[2];
}
static M33 m33Mul(const M33& a, const M33& b)
{
    M33 r;
    m33Transform(a, &b.c[0], &r.c[0]);
    m33Transform(a, &b.c[3], &r.c[3]);
    m33Transform(a, &b.c[6], &r.c[6]);
    return r;
}
static M33 m33Transpose(const M33& m)
{
    M33 r;
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            r.c[col * 3 + row] = m.c[row * 3 + col];
    return r;
}
// PxMat33(PxQuat) (PxMat33.h:138), operation for operation.
static M33 m33FromQuat(F4 q)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    M33 m;
    m.c[0] = 1.0f - yy - zz; m.c[1] = xy + zw;        m.c[2] = xz - yw;
    m.c[3] = xy - zw;        m.c[4] = 1.0f - xx - zz; m.c[5] = yz + xw;
    m.c[6] = xz + yw;        m.c[7] = yz - xw;        m.c[8] = 1.0f - xx - yy;
    return m;
}

// sb_calculateStressLaunch (softBodyGM.cu:2716), one tetrahedron.
static void refStress(const std::vector<F4>& pos, p80refit::TetIdx4 tet, const M33& qinv,
                      F4 tetraRot, float youngs, float poissons, M33& sigma, float& stress)
{
    const F4 p0 = pos[tet.x], p1 = pos[tet.y], p2 = pos[tet.z], p3 = pos[tet.w];
    const float u1[3] = { p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
    const float u2[3] = { p2.x - p0.x, p2.y - p0.y, p2.z - p0.z };
    const float u3[3] = { p3.x - p0.x, p3.y - p0.y, p3.z - p0.z };

    const M33 P = m33FromCols(u1, u2, u3);
    M33 F = m33Mul(P, qinv);
    F = m33Mul(m33Transpose(m33FromQuat(tetraRot)), F);

    float lambda = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
    float mu = 0.5f * youngs / (1.0f + poissons);
    lambda += mu * (5.0f / 6.0f);
    mu = mu * (4.0f / 3.0f);

    const M33 FT = m33Transpose(F);
    M33 eps;
    for (int i = 0; i < 9; ++i) eps.c[i] = (FT.c[i] + F.c[i]) * 0.5f;
    eps.c[0] -= 1.0f; eps.c[4] -= 1.0f; eps.c[8] -= 1.0f;

    const float trace = eps.c[0] + eps.c[4] + eps.c[8];
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
        {
            const float ident = (col == row) ? 1.0f : 0.0f;
            sigma.c[col * 3 + row] = eps.c[col * 3 + row] * 2.0f * mu + ident * lambda * trace;
        }

    const float q1 = eps.c[0], q2 = eps.c[4], q3 = eps.c[8];
    const float q12 = q1 - q2, q23 = q2 - q3, q31 = q3 - q1;
    stress = std::sqrt(q12 * q12 + q23 * q23 + q31 * q31) * 0.70710678118654752440084436210485f;
}
} // namespace p85ref

struct SbStressPush { uint64_t softbodies, activeSoftbodies, materials; };
static_assert(sizeof(SbStressPush) == 24, "stress push blob must be 24 bytes");

PXG_TEST(softBodySimCalculateStress)
{
    using namespace p80refit;
    using namespace p85ref;
    PxgSlangComputeDevice* dev = testDevice();

    // Three tets sharing one vertex pool.  Tet 0 is UNDEFORMED (its edge matrix is exactly
    // the inverse of its rest pose, so the strain is zero and the coefficient must be too);
    // tet 1 is stretched along x; tet 2 is stretched UNIFORMLY, which strains the material
    // but leaves the three diagonal DIFFERENCES equal -- so its coefficient is zero while
    // its sigma is not.  A port that built the coefficient from the strain magnitude instead
    // of from the diagonal differences passes tet 0 and fails tet 2.
    std::vector<F4> pos = {
        { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f },   // tet 0, unit
        { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.4f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f },   // tet 1, stretched in x
        { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.3f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.3f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.3f, 1.0f },   // tet 2, uniform stretch
    };
    std::vector<TetIdx4> tets = { { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u }, { 8u, 9u, 10u, 11u } };
    const uint32_t NB_TETS = uint32_t(tets.size());

    // Rest poses: the INVERSE of the undeformed edge matrix, which for a unit tet is the
    // identity.  Stored FLAT at a 36-byte stride, not as a PxgMat33Block.
    std::vector<float> restPoses(NB_TETS * 9u, 0.0f);
    for (uint32_t t = 0; t < NB_TETS; ++t)
    { restPoses[t * 9u + 0u] = 1.0f; restPoses[t * 9u + 4u] = 1.0f; restPoses[t * 9u + 8u] = 1.0f; }
    // Tet 1's rest pose is deliberately NON-SYMMETRIC.  With every rest pose an identity,
    // `P * Qinv` and `Qinv * P` agree and a port that swapped the deformation gradient's
    // operands passes -- that mutant survived until this shear was added.  Column-major:
    // column1 gains an x term, column2 a y term.
    restPoses[1 * 9u + 3u] = 0.3f;   // column1.x
    restPoses[1 * 9u + 7u] = 0.2f;   // column2.y

    // A NON-IDENTITY rotation on tet 1: the co-rotation must actually remove it, and it must
    // be the TRANSPOSE that is applied.  q = 30 degrees about z.
    const float half = 0.2617993878f;   // 15 degrees in radians
    std::vector<F4> tetraRot = {
        { 0.0f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 0.0f, std::sin(half), std::cos(half) },
        { 0.0f, 0.0f, 0.0f, 1.0f },
    };

    // Two materials, so the PxU16 index array is read at both halves of a word and a port
    // that used material 0 for everything cannot match.
    std::vector<DeformableMaterialHost> materials(2);
    std::memset(materials.data(), 0, materials.size() * sizeof(DeformableMaterialHost));
    materials[0].youngs = 1.0e5f; materials[0].poissons = 0.30f;
    materials[1].youngs = 4.0e5f; materials[1].poissons = 0.45f;
    const std::vector<uint16_t> matIdx = { 0u, 1u, 1u };
    std::vector<uint32_t> matWords((matIdx.size() + 1u) / 2u + 1u, 0u);
    for (size_t i = 0; i < matIdx.size(); ++i) matWords[i / 2u] |= uint32_t(matIdx[i]) << (16u * (i % 2u));

    DevBuf bPos = makeDev(dev, pos), bTets = makeDev(dev, tets);
    DevBuf bRest = makeDev(dev, restPoses), bRot = makeDev(dev, tetraRot);
    DevBuf bMatIdx = makeDev(dev, matWords), bMat = makeDev(dev, materials);
    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const float SENT = -777.0f;
    std::vector<float> stresses(NB_TETS * 9u + 9u, SENT);   // flat PxMat33 array + a trailer
    std::vector<float> coeffs(NB_TETS + 4u, SENT);
    DevBuf bStresses = makeDev(dev, stresses), bCoeffs = makeDev(dev, coeffs);

    PxgSoftBody sb = makeZeroDesc();
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mTetraRestPoses = reinterpret_cast<PxMat33*>(size_t(bRest.addr));
    sb.mTetraRotations = reinterpret_cast<float4*>(size_t(bRot.addr));
    sb.mMaterialIndices = reinterpret_cast<PxU16*>(size_t(bMatIdx.addr));
    sb.mTetraStresses = reinterpret_cast<PxMat33*>(size_t(bStresses.addr));
    sb.mTetraStressCoefficient = reinterpret_cast<PxReal*>(size_t(bCoeffs.addr));
    sb.mNumTets = NB_TETS;
    // Self-collision ON (the flag is DISABLE), stress tensor NOT requested: the kernel must
    // still run, because the self-collision gate is what consumes the coefficient.
    sb.mBodyFlags = 0u;
    sb.mVolumeFlags = PxDeformableVolumeFlags(0);

    // TWO active bodies sharing the inputs but writing to SEPARATE outputs, and the active
    // list is not the identity.  With a single body and a 1x1 grid, gid.x and gid.y are both
    // 0 and the two are indistinguishable -- `softbodyId from gid.x` and `tetIdx from gid.y`
    // both survived until this second body existed.
    std::vector<float> stressesB(stresses.size(), SENT), coeffsB(coeffs.size(), SENT);
    DevBuf bStressesB = makeDev(dev, stressesB), bCoeffsB = makeDev(dev, coeffsB);
    PxgSoftBody sbB = sb;
    sbB.mTetraStresses = reinterpret_cast<PxMat33*>(size_t(bStressesB.addr));
    sbB.mTetraStressCoefficient = reinterpret_cast<PxReal*>(size_t(bCoeffsB.addr));

    std::vector<PxgSoftBody> descArr(3u, makeZeroDesc());
    descArr[1] = sb;
    descArr[2] = sbB;
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 1u, 2u };
    DevBuf bActive = makeDev(dev, active);

    SbStressPush push = {};
    push.softbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.materials = bMat.addr;

    Kernel k("softBodySolveBda.spv", "sb_calculateStressLaunch", {}, sizeof(SbStressPush));
    // The engine's grid: ceil(maxTetrahedrons / 256) in x, the active softbodies in y.
    k.dispatch(/*gridX*/ 1u, {}, push, /*gridY*/ uint32_t(active.size()), 1u);
    dev->flush();

    const auto gStresses = downloadBuffer<float>(dev, bStresses.h, stresses.size());
    const auto gCoeffs = downloadBuffer<float>(dev, bCoeffs.h, coeffs.size());

    for (uint32_t t = 0; t < NB_TETS; ++t)
    {
        M33 qinv;
        for (int i = 0; i < 9; ++i) qinv.c[i] = restPoses[t * 9u + uint32_t(i)];
        M33 wantSigma; float wantStress = 0.0f;
        refStress(pos, tets[t], qinv, tetraRot[t],
                  materials[matIdx[t]].youngs, materials[matIdx[t]].poissons, wantSigma, wantStress);

        std::printf("  tet %u: coeff %.7f (want %.7f), sigma[0][0] %.5f (want %.5f)\n",
                    t, gCoeffs[t], wantStress, gStresses[t * 9u], wantSigma.c[0]);

        for (int i = 0; i < 9; ++i)
            PXG_CHECK_MSG(f2u(gStresses[t * 9u + uint32_t(i)]) == f2u(wantSigma.c[i]),
                          "tet %u sigma element %d: got %.9g, want %.9g", t, i,
                          gStresses[t * 9u + uint32_t(i)], wantSigma.c[i]);
        PXG_CHECK_MSG(std::fabs(gCoeffs[t] - wantStress) <= 1e-6f,
                      "tet %u stress coefficient: got %.9g, want %.9g", t, gCoeffs[t], wantStress);
    }

    // The three shapes, stated as facts rather than left to the reference.
    PXG_CHECK_MSG(gCoeffs[0] == 0.0f,
                  "tet 0 is undeformed, so its strain is identically zero and so is its "
                  "coefficient (got %.9g)", gCoeffs[0]);
    PXG_CHECK_MSG(gCoeffs[1] > 0.1f,
                  "tet 1 is stretched 1.4x along x under a 30-degree rotation; the co-rotation "
                  "must remove the rotation and leave a real strain (got %.9g)", gCoeffs[1]);
    PXG_CHECK_MSG(gCoeffs[2] == 0.0f && gStresses[2 * 9u] != 0.0f,
                  "tet 2 is stretched UNIFORMLY: its diagonal differences cancel so the "
                  "coefficient is zero (got %.9g) while sigma is NOT (got %.9g).  A "
                  "coefficient built from the strain magnitude would be nonzero here",
                  gCoeffs[2], gStresses[2 * 9u]);
    PXG_CHECK_MSG(gStresses[NB_TETS * 9u] == SENT && gCoeffs[NB_TETS] == SENT,
                  "nothing may be written past mNumTets");

    // The SECOND active body must have been written too, and identically: it shares every
    // input and differs only in where its results go.  This is what pins gid.y as the body
    // selector and gid.x as the tetrahedron block.
    {
        const auto gStressesB = downloadBuffer<float>(dev, bStressesB.h, stressesB.size());
        const auto gCoeffsB = downloadBuffer<float>(dev, bCoeffsB.h, coeffsB.size());
        for (uint32_t t = 0; t < NB_TETS; ++t)
        {
            PXG_CHECK_MSG(f2u(gCoeffsB[t]) == f2u(gCoeffs[t]),
                          "the second active body's tet %u coefficient must match the first's "
                          "(%.9g vs %.9g) -- a y-grid that does not select the body leaves it "
                          "untouched at %g", t, gCoeffsB[t], gCoeffs[t], SENT);
            for (int i = 0; i < 9; ++i)
                PXG_CHECK(f2u(gStressesB[t * 9u + uint32_t(i)]) == f2u(gStresses[t * 9u + uint32_t(i)]));
        }
    }

    // ---- the launch gate ------------------------------------------------------
    // The kernel runs when eCOMPUTE_STRESS_TENSOR is set OR self-collision is enabled.  With
    // self-collision DISABLED and the tensor not requested, it must write nothing at all;
    // with the tensor requested it must run again even though self-collision stays off.
    for (int pass = 0; pass < 2; ++pass)
    {
        const bool requestTensor = (pass == 1);
        std::vector<PxgSoftBody> d2 = descArr;
        d2[1].mBodyFlags = PxDeformableBodyFlags(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION);
        d2[1].mVolumeFlags = requestTensor
            ? PxDeformableVolumeFlags(PxDeformableVolumeFlag::eCOMPUTE_STRESS_TENSOR)
            : PxDeformableVolumeFlags(0);
        std::vector<float> s2(stresses.size(), SENT), c2(coeffs.size(), SENT);
        DevBuf bS2 = makeDev(dev, s2), bC2 = makeDev(dev, c2);
        d2[1].mTetraStresses = reinterpret_cast<PxMat33*>(size_t(bS2.addr));
        d2[1].mTetraStressCoefficient = reinterpret_cast<PxReal*>(size_t(bC2.addr));
        d2[2] = d2[1];   // both active slots point at the same outputs for this sub-test
        DevBuf bD2 = makeDev(dev, d2);
        SbStressPush p2 = push;
        p2.softbodies = bD2.addr;
        Kernel("softBodySolveBda.spv", "sb_calculateStressLaunch", {}, sizeof(SbStressPush))
            .dispatch(1u, {}, p2, uint32_t(active.size()), 1u);
        dev->flush();
        const auto g2 = downloadBuffer<float>(dev, bC2.h, coeffs.size());
        if (requestTensor)
            PXG_CHECK_MSG(g2[1] != SENT,
                          "eCOMPUTE_STRESS_TENSOR alone must run the kernel even with "
                          "self-collision disabled (coefficient 1 is still %g)", g2[1]);
        else
            PXG_CHECK_MSG(g2[0] == SENT && g2[1] == SENT && g2[2] == SENT,
                          "with self-collision DISABLED and no tensor requested the kernel must "
                          "write nothing (got %g, %g, %g)", g2[0], g2[1], g2[2]);
        dev->destroyBuffer(bS2.h); dev->destroyBuffer(bC2.h); dev->destroyBuffer(bD2.h);
    }

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bTets.h); dev->destroyBuffer(bRest.h);
    dev->destroyBuffer(bRot.h); dev->destroyBuffer(bMatIdx.h); dev->destroyBuffer(bMat.h);
    dev->destroyBuffer(bStresses.h); dev->destroyBuffer(bCoeffs.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bActive.h); dev->destroyBuffer(bStressesB.h); dev->destroyBuffer(bCoeffsB.h);
}

// ===========================================================================
// P8.6 — the SPHERE and CAPSULE contact-gen path, through the shared GJK/EPA driver.
//
// The geometry is chosen so every expected number is exact BEFORE the run: the tetrahedron
// is the unit tet at the origin, and each rigid is placed so its nearest approach to the tet
// is a clean distance along -x from the x = 0 face, whose nearest point is (0, 0.25, 0.25).
// (That the numbers must be predictable in advance is not a style preference here -- the
// step-3 driver fixture was calibrated against broken output and "passed" on a value I could
// explain but had not predicted.)
//
//   pair 0  sphere, centre (-0.3, 0.25, 0.25), radius 0.4  -> gap 0.3, pen 0.3 - 0.4 = -0.1
//   pair 1  sphere, centre (-9.0, 0.25, 0.25), radius 0.4  -> far outside, NO contact
//   pair 2  capsule, centre (-0.6, 0.25, 0.25), half-height 0.5, radius 0.2
//                    -> nearest endpoint (-0.1, ...), gap 0.1, pen 0.1 - 0.2 = -0.1
//
// The two -0.1 penetrations come from different shapes and different radii, which is the
// point: the radius is folded into GJK's contact distance and subtracted back out as
// penOffset, so a port that dropped either half is off by exactly one radius and the two
// cases disagree about which.
// ===========================================================================
PXG_TEST(softBodySimPrimitiveContactGenGjk)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<F4> verts = {
        { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f },
    };
    std::vector<TetIdx4> tets = { { 0u, 1u, 2u, 3u } };
    DevBuf bVerts = makeDev(dev, verts), bTets = makeDev(dev, tets);
    if (!bVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
    sb.mNumTets = 1u;
    sb.mNumVerts = 4u;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // shape 0 = the tet mesh; shapes 1..3 = the rigids.
    const uint32_t GEOM_SPHERE_H = 0u, GEOM_CAPSULE_H = 2u;
    std::vector<PxgShapeHost> shapes(4);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TETRAHEDRONMESH_H; shapes[0].particleOrSoftbodyId = 0u;
    // PxMeshScale: for a SPHERE the engine stores the radius in both x and y; the port reads
    // x for the contact-distance inflation and y for penOffset, as upstream does.
    shapes[1].type = GEOM_SPHERE_H;  shapes[1].sx = 0.4f; shapes[1].sy = 0.4f; shapes[1].sz = 0.4f;
    shapes[1].srw = 1.0f; shapes[1].materialIndex = 11u; shapes[1].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[2] = shapes[1]; shapes[2].materialIndex = 12u;
    // For a CAPSULE x is the HALF-HEIGHT and y is the radius -- deliberately different values
    // so a port that read the wrong one cannot match.
    shapes[3].type = GEOM_CAPSULE_H; shapes[3].sx = 0.5f; shapes[3].sy = 0.2f; shapes[3].sz = 0.2f;
    shapes[3].srw = 1.0f; shapes[3].materialIndex = 13u; shapes[3].particleOrSoftbodyId = 0xFFFFFFFFu;
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<CmInputHost> cms = {
        { 0u, 1u, 0u, 1u },
        { 0u, 2u, 0u, 2u },
        { 0u, 3u, 0u, 3u },
    };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform, 32 B: PxQuat q @0 (identity), PxVec3 p @16.
    std::vector<float> xf(4 * 8, 0.0f);
    auto setXf = [&](uint32_t slot, float x, float y, float z) {
        xf[slot * 8 + 3] = 1.0f;                              // q.w
        xf[slot * 8 + 4] = x; xf[slot * 8 + 5] = y; xf[slot * 8 + 6] = z;
    };
    setXf(0, 0.0f, 0.0f, 0.0f);          // the softbody's slot, unused
    setXf(1, -0.3f, 0.25f, 0.25f);       // sphere, touching
    setXf(2, -9.0f, 0.25f, 0.25f);       // sphere, far
    setXf(3, -0.6f, 0.25f, 0.25f);       // capsule centre
    DevBuf bXf = makeDev(dev, xf);

    // cDistance = contactDistance[sb] + contactDistance[rigid] = 0, so the ONLY inflation is
    // the radius the branch folds in.  That isolates the radius handling.
    std::vector<float> cd(4, 0.0f);
    DevBuf bCd = makeDev(dev, cd);
    std::vector<float> rest = { 0.03f, 0.04f, 0.05f };
    DevBuf bRest = makeDev(dev, rest);
    std::vector<uint64_t> remap = { 0ull, 0x101ull, 0x202ull, 0x303ull };
    DevBuf bRemap = makeDev(dev, remap);

    std::vector<uint32_t> stack = { 0u, 0u, 0u, 0u,  1u, 0u, 0u, 0u,  2u, 0u, 0u, 0u };
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 3u };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 16u;
    std::vector<F4> outPoint(MAXC, F4{ -1, -1, -1, -1 }), outNormal(MAXC, F4{ -1, -1, -1, -1 });
    std::vector<F4> outBary(MAXC, F4{ 0, 0, 0, 0 });
    std::vector<FemOtherContactInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(FemOtherContactInfoHost));
    std::vector<uint64_t> byX(MAXC, 0ull);
    std::vector<uint32_t> tempByX(MAXC, 0u), idxSorted(MAXC, 0u), sortedByX(MAXC, 0u), total = { 0u };
    DevBuf bPt = makeDev(dev, outPoint), bNp = makeDev(dev, outNormal), bBary = makeDev(dev, outBary);
    DevBuf bInfo = makeDev(dev, outInfo), bByX = makeDev(dev, byX), bTmp = makeDev(dev, tempByX);
    DevBuf bIdx = makeDev(dev, idxSorted), bSorted = makeDev(dev, sortedByX), bTotal = makeDev(dev, total);

    SbContactGenPush push = {};
    push.tolerenceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.restDistance = bRest.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.shapeToRigidRemapTable = bRemap.addr;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.stackSizeBytes = 48u;               // three uint4 slots
    push.softBodyRigidPairs = 0ull;
    push.numSoftBodyRigidPairs = 0u;
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.w.outPoint = bPt.addr; push.w.outNormalPen = bNp.addr; push.w.outBarycentric = bBary.addr;
    push.w.outContactInfo = bInfo.addr; push.w.totalContactCount = bTotal.addr;
    push.w.contactByX = bByX.addr; push.w.tempContactByX = bTmp.addr;
    push.w.contactIndexSortedByX = bIdx.addr; push.w.contactSortedByX = bSorted.addr;
    push.w.maxNumContacts = MAXC;

    Kernel k("softBodyContactGenBda.spv", "sb_primitiveContactGenLaunch", {}, sizeof(SbContactGenPush));
    // THE ENGINE'S GRID (PxgNarrowphaseCore.cpp:4300): SB_CG_GRID = 512 blocks.  The kernel
    // strides by SB_CG_GRID * MIDPHASE_WARPS_PER_BLOCK = 1024, so a one-group dispatch covers
    // only warps 0 and 1 -- i.e. the first TWO pairs.  The plane fixture has exactly two and
    // never noticed; this one has three, and its third pair vanished until the grid was
    // right.  Dispatch what the engine dispatches.
    k.dispatch(/*gridX*/ 512u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    const auto gPt = downloadBuffer<F4>(dev, bPt.h, MAXC);
    const auto gNp = downloadBuffer<F4>(dev, bNp.h, MAXC);
    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, bInfo.h, MAXC);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        std::printf("  contact %u: point=(%.5f %.5f %.5f rest=%.3f) normal=(%.5f %.5f %.5f) "
                    "pen=%.6f mat=%u\n", i, gPt[i].x, gPt[i].y, gPt[i].z, gPt[i].w,
                    gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w, gInfo[i].rigidMatInd & 0x7fffffffu);

    PXG_CHECK_MSG(n == 2u,
                  "the touching sphere and the capsule must each emit ONE contact and the far "
                  "sphere none, got %u", n);

    bool sawSphere = false, sawCapsule = false;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t mat = gInfo[i].rigidMatInd & 0x7fffffffu;
        PXG_CHECK_MSG(mat != 12u, "the far sphere (material 12) must not produce a contact");
        // The witness on the TET is the x = 0 face point nearest the rigid.
        PXG_CHECK_MSG(std::fabs(gPt[i].x) < 1e-5f && std::fabs(gPt[i].y - 0.25f) < 1e-5f &&
                      std::fabs(gPt[i].z - 0.25f) < 1e-5f,
                      "contact %u's point is the witness on the TET, (0, 0.25, 0.25), got "
                      "(%.6f %.6f %.6f)", i, gPt[i].x, gPt[i].y, gPt[i].z);
        PXG_CHECK_MSG(std::fabs(gNp[i].x - 1.0f) < 1e-4f && std::fabs(gNp[i].y) < 1e-4f &&
                      std::fabs(gNp[i].z) < 1e-4f,
                      "and the normal is +x, got (%.5f %.5f %.5f)", gNp[i].x, gNp[i].y, gNp[i].z);
        // BOTH shapes penetrate by exactly 0.1, from different gaps and different radii.
        PXG_CHECK_MSG(std::fabs(gNp[i].w + 0.1f) < 1e-5f,
                      "contact %u's penetration is gap - radius = -0.1 (got %.7f).  Dropping "
                      "the radius from the contact distance, or from penOffset, changes this "
                      "by exactly one radius -- and the two cases have DIFFERENT radii, so no "
                      "single wrong constant can satisfy both", i, gNp[i].w);
        if (mat == 11u) { sawSphere = true;  PXG_CHECK(std::fabs(gPt[i].w - 0.03f) < 1e-6f); }
        if (mat == 13u) { sawCapsule = true; PXG_CHECK(std::fabs(gPt[i].w - 0.05f) < 1e-6f); }
    }
    PXG_CHECK_MSG(sawSphere, "the touching SPHERE produced no contact");
    PXG_CHECK_MSG(sawCapsule, "the CAPSULE produced no contact -- its cloud is the two segment "
                              "endpoints, so a port that treated it as a single point would "
                              "measure from the centre and miss by the half-height");

    for (PxgBufferHandle h : { bVerts.h, bTets.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h,
                               bRest.h, bRemap.h, bStack.h, bPairNum.h, bNeeded.h, bPt.h,
                               bNp.h, bBary.h, bInfo.h, bByX.h, bTmp.h, bIdx.h, bSorted.h, bTotal.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.7 — the BOX and CONVEXMESH contact-gen path (the convex-hull branch).
//
// Sphere and capsule build their point cloud from the transform alone.  A hull instead has
// to be READ: the cooked blob is walked (skip one float4, uint4 header, float4 extents, then
// the vertices), the vertex count is unpacked out of a bit-field, and every vertex goes
// through vertex2Shape and then the rigid transform.  Four things can be got wrong, and the
// fixture is built so each one changes an observable number:
//
//   * the blob offsets      -> garbage vertices, nothing lands anywhere near the tet;
//   * the vertex count      -> a partial hull, whose support in +x is smaller;
//   * PxMeshScale           -> the wrong box size, hence the wrong gap;
//   * the radius inflation  -> a hull must NOT inflate the contact distance the way a sphere
//                             does, and must NOT subtract a radius from the penetration.
//
// THE EXPECTED VALUES ARE COMPUTED, NOT CALIBRATED.  For a separated convex pair whose
// nearest feature is one hull CORNER against the interior of the tet's x = 0 face, the answer
// is the support function and nothing more: take the world corner with the largest x, the
// witness on the tet is that corner projected onto x = 0, the normal is +x and the separation
// is -corner.x.  The reference below is that support computation; the tet is the doubled unit
// tet so the face triangle (y >= 0, z >= 0, y + z <= 2) comfortably contains both projections.
// A generic rotation is used precisely so ONE corner wins -- with an axis-aligned box the
// nearest-point pair is a whole rectangle and the witness would not be predictable.
//
//   pair 0  BOX,        scale (0.3, 0.2, 0.1), identity mesh-scale rotation, generic rigid
//                       rotation, centre (-0.45, 0.6, 0.6)  -> gap 0.154364, winner vertex 0
//   pair 1  BOX,        same hull and scale, identity rotation, centre (-0.70, 0.6, 0.6)
//                       -> gap 0.4 exactly, and cDistance is 0.3 -> NO contact.  This is the
//                       inflation trap: a port that added scale.y (0.2) or scale.x (0.3) to
//                       the contact distance the way the sphere branch does would report a
//                       contact here.
//   pair 2  CONVEXMESH, scale (0.4, 0.1, 0.15) under a NON-IDENTITY mesh-scale rotation,
//                       centre (-0.53, 0.6, 0.6)            -> gap 0.196020, winner vertex 7.
//                       Ignoring the scale rotation -- treating vertex2Shape as a componentwise
//                       multiply -- gives 0.13 instead, and the fixture asserts the two differ
//                       before it compares anything, so the discrimination cannot evaporate.
//
// THE HULL IS IRREGULAR AND ITS VERTEX ORDER IS LOAD-BEARING.  The first draft used a unit
// cube, and a mutant that shifted the vertex-array base by one float4 SURVIVED: a cube is
// symmetric enough that dropping one corner (and reading one past the end) leaves the support
// in +x untouched, and reading the extents float4 as a vertex just re-adds a corner the cube
// already had.  The eight vertices below are irregular and deliberately ordered so that
// pair 0's supporting vertex is index 0 and pair 2's is index 7 -- the two ends of the array.
// A base that is one float4 too high loses index 0 and moves pair 0; one too low, or a vertex
// count read out of the wrong bits, loses index 7 and moves pair 2.
//
// ONE INPUT IS NOT PINNED, AND SAYING SO IS BETTER THAN PRETENDING.  `inSphereRadius0`
// -- extents.w times the smallest axis scale for a box, times 1 for a convex mesh -- reaches
// nothing but EPA's convergence epsilon, `0.5 * (r0 + r1) * 2.25e-4`, which is order 1e-4.
// Removing the box's scale factor entirely, or taking the minimum over the wrong two axes,
// leaves every number this fixture can observe unchanged (verified: both mutants survive,
// including with pair 3 driving EPA).  It is transcribed from softbodyPrimitives.cu:513-519
// and carried on trust, not on evidence; a test that could see it would have to be an EPA
// convergence test, not a contact test.
// ===========================================================================
namespace p87hull
{
struct HV3 { float x, y, z; };
struct HQ4 { float x, y, z, w; };

// PxQuat::rotate / rotateInv (PxQuat.h:286, :324), transcribed so the reference and the
// shader agree term for term rather than merely mathematically.
inline HV3 qrot(HQ4 q, HV3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
inline HV3 qrotInv(HQ4 q, HV3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
// vertex2Shape (cudaNpCommon.h:270): rotate into the scale frame, scale, rotate back.
inline HV3 vertex2Shape(HV3 v, HV3 s, HQ4 rot)
{
    HV3 t = qrot(rot, v);
    t.x *= s.x; t.y *= s.y; t.z *= s.z;
    return qrotInv(rot, t);
}
inline HQ4 qnorm(float x, float y, float z, float w)
{
    const float m = std::sqrt(x * x + y * y + z * z + w * w);
    return { x / m, y / m, z / m, w / m };
}
// The eight hull vertices in VERTEX space, in the order they sit in the cooked blob.
// Irregular, and ordered so the two supporting vertices are index 0 and index 7 (see header).
static const HV3 HULL[8] = {
    {  0.90f, -0.70f,  0.80f },   // 0 — pair 0's support
    { -0.80f,  0.50f, -0.60f },
    { -0.90f, -0.40f,  0.70f },
    {  0.20f,  0.90f, -0.80f },
    { -0.30f, -0.90f,  0.20f },
    {  0.40f,  0.30f, -0.95f },
    { -0.60f,  0.80f,  0.90f },
    {  1.00f,  0.60f,  0.35f },   // 7 — pair 2's support
};

// The support point in +x over the hull, the index that won, and the margin to the runner-up.
// A small margin would mean the winning vertex is not actually unique and the witness point
// is not predictable, so the fixture checks it.
inline HV3 supportMaxX(HV3 scale, HQ4 scaleRot, HQ4 rigidQ, HV3 rigidP, bool useScaleRot,
                       float& margin, int& winner)
{
    HV3 best = { -1e30f, 0.0f, 0.0f };
    float second = -1e30f;
    winner = -1;
    for (int m = 0; m < 8; ++m)
    {
        const HV3 c = HULL[m];
        HV3 sv = useScaleRot ? vertex2Shape(c, scale, scaleRot)
                             : HV3{ c.x * scale.x, c.y * scale.y, c.z * scale.z };
        const HV3 r = qrot(rigidQ, sv);
        const HV3 w = { r.x + rigidP.x, r.y + rigidP.y, r.z + rigidP.z };
        if (w.x > best.x) { second = best.x; best = w; winner = m; }
        else if (w.x > second) { second = w.x; }
    }
    margin = best.x - second;
    return best;
}
} // namespace p87hull

PXG_TEST(softBodySimPrimitiveContactGenHull)
{
    using namespace p80refit;
    using namespace p87hull;
    PxgSlangComputeDevice* dev = testDevice();

    // The DOUBLED unit tet: the x = 0 face is the triangle y >= 0, z >= 0, y + z <= 2, which
    // both witness projections fall well inside (see the header).
    std::vector<F4> verts = {
        { 0.0f, 0.0f, 0.0f, 1.0f }, { 2.0f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 2.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 2.0f, 1.0f },
    };
    std::vector<TetIdx4> tets = { { 0u, 1u, 2u, 3u } };
    DevBuf bVerts = makeDev(dev, verts), bTets = makeDev(dev, tets);
    if (!bVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
    sb.mNumTets = 1u;
    sb.mNumVerts = 4u;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the cooked hull blob ------------------------------------------------
    // sbPrimitiveCollision skips one float4 (`convexPtr = hullPtr + sizeof(float4)`), then
    // reads a uint4 header, then a float4 of extents, then the vertices as float4.  The
    // vertex count lives in the header's first word as u16High(u32Low), i.e. bits 8..15.
    const uint32_t NB_HULL_VERTS = 8u;
    const uint32_t hdr0 = (12u << 16) | (NB_HULL_VERTS << 8) | 6u;   // nbEdges | nbVerts | nbPolys
    auto asF = [](uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; };
    std::vector<F4> blob;
    blob.push_back(F4{ 9.0f, 9.0f, 9.0f, 9.0f });                     // the skipped float4
    blob.push_back(F4{ asF(hdr0), asF(0u), asF(0u), asF(0u) });        // uint4 header
    // extents.xyz is never read by this path; .w is the stored inner-sphere radius.  The xyz
    // are set to a point that could never be a support vertex, so a base offset that read the
    // extents AS a vertex is not accidentally harmless.
    blob.push_back(F4{ 0.0f, 0.0f, 0.0f, 1.0f });
    for (uint32_t m = 0; m < NB_HULL_VERTS; ++m)
        blob.push_back(F4{ HULL[m].x, HULL[m].y, HULL[m].z, 0.0f });
    DevBuf bBlob = makeDev(dev, blob);

    // A SECOND blob, the unit cube, for the overlapping pair 3 -- a real box, so its
    // penetration depth is decidable in advance (see below).  Two blobs also mean the hull
    // pointer is genuinely per-shape rather than a single global the fixture cannot tell apart.
    std::vector<F4> blobBox;
    blobBox.push_back(F4{ 9.0f, 9.0f, 9.0f, 9.0f });
    blobBox.push_back(F4{ asF(hdr0), asF(0u), asF(0u), asF(0u) });
    blobBox.push_back(F4{ 0.0f, 0.0f, 0.0f, 1.0f });
    for (uint32_t m = 0; m < NB_HULL_VERTS; ++m)
        blobBox.push_back(F4{ (m & 1u) ? 1.0f : -1.0f, (m & 2u) ? 1.0f : -1.0f,
                              (m & 4u) ? 1.0f : -1.0f, 0.0f });
    DevBuf bBlobBox = makeDev(dev, blobBox);

    // ---- the four rigids -----------------------------------------------------
    const uint32_t GEOM_BOX_H = 3u, GEOM_CONVEXMESH_H = 4u;
    const HQ4 identQ = { 0.0f, 0.0f, 0.0f, 1.0f };
    const HQ4 rigidQ0 = qnorm(0.13f, 0.27f, 0.19f, 0.94f);  // generic: one vertex wins
    const HQ4 meshRot2 = qnorm(0.8f, 0.5f, 0.1f, 0.9f);     // generic mesh-scale rotation
    const HV3 scale0 = { 0.3f, 0.2f, 0.1f };
    const HV3 scale2 = { 0.4f, 0.1f, 0.15f };
    const HV3 p0 = { -0.45f, 0.6f, 0.6f }, p1 = { -0.70f, 0.6f, 0.6f }, p2 = { -0.53f, 0.6f, 0.6f };
    // Pair 3, the OVERLAP: a genuine axis-aligned box of half extents (0.3, 0.2, 0.1) centred
    // at (-0.25, 0.6, 0.6), so it spans x in [-0.55, 0.05] and pokes 0.05 through the tet's
    // x = 0 face.  Every other escape is far longer -- 0.4 to the y = 0 and z = 0 faces, 0.26
    // to the slant face x + y + z = 2 -- so the minimum translation is 0.05 along -x and the
    // penetration is decidable without running anything.  The WITNESS POINT is not: the
    // overlap is a slab and any (y, z) in it is a valid pair, so only the normal and the
    // depth are asserted, and deliberately so.
    const HV3 p3 = { -0.25f, 0.6f, 0.6f };
    const float WANT_PEN3 = -0.05f;

    std::vector<PxgShapeHost> shapes(5);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TETRAHEDRONMESH_H; shapes[0].particleOrSoftbodyId = 0u;
    auto setHull = [&](uint32_t i, uint32_t type, HV3 s, HQ4 r, uint32_t mat, uint64_t hull) {
        shapes[i].type = type;
        shapes[i].sx = s.x; shapes[i].sy = s.y; shapes[i].sz = s.z;
        shapes[i].srx = r.x; shapes[i].sry = r.y; shapes[i].srz = r.z; shapes[i].srw = r.w;
        shapes[i].materialIndex = mat;
        shapes[i].particleOrSoftbodyId = 0xFFFFFFFFu;
        shapes[i].hullLo = uint32_t(hull & 0xFFFFFFFFull);
        shapes[i].hullHi = uint32_t(hull >> 32);
    };
    setHull(1u, GEOM_BOX_H,        scale0, identQ,   21u, bBlob.addr);
    setHull(2u, GEOM_BOX_H,        scale0, identQ,   22u, bBlob.addr);
    setHull(3u, GEOM_CONVEXMESH_H, scale2, meshRot2, 23u, bBlob.addr);
    setHull(4u, GEOM_BOX_H,        scale0, identQ,   24u, bBlobBox.addr);
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u }, { 0u, 2u, 0u, 2u },
                                     { 0u, 3u, 0u, 3u }, { 0u, 4u, 0u, 4u } };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.
    std::vector<float> xf(5 * 8, 0.0f);
    auto setXf = [&](uint32_t slot, HQ4 q, HV3 p) {
        xf[slot * 8 + 0] = q.x; xf[slot * 8 + 1] = q.y; xf[slot * 8 + 2] = q.z; xf[slot * 8 + 3] = q.w;
        xf[slot * 8 + 4] = p.x; xf[slot * 8 + 5] = p.y; xf[slot * 8 + 6] = p.z;
    };
    setXf(0, identQ, HV3{ 0.0f, 0.0f, 0.0f });
    setXf(1, rigidQ0, p0);
    setXf(2, identQ,  p1);
    setXf(3, identQ,  p2);
    setXf(4, identQ,  p3);
    DevBuf bXf = makeDev(dev, xf);

    // cDistance = contactDistance[sbCacheRef] + contactDistance[rigidCacheRef] = 0.3.
    std::vector<float> cd = { 0.0f, 0.3f, 0.3f, 0.3f, 0.3f };
    DevBuf bCd = makeDev(dev, cd);
    std::vector<float> rest = { 0.03f, 0.04f, 0.05f, 0.06f };
    DevBuf bRest = makeDev(dev, rest);
    std::vector<uint64_t> remap = { 0ull, 0x101ull, 0x202ull, 0x303ull, 0x404ull };
    DevBuf bRemap = makeDev(dev, remap);

    std::vector<uint32_t> stack = { 0u, 0u, 0u, 0u,  1u, 0u, 0u, 0u,
                                    2u, 0u, 0u, 0u,  3u, 0u, 0u, 0u };
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 4u };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 16u;
    std::vector<F4> outPoint(MAXC, F4{ -1, -1, -1, -1 }), outNormal(MAXC, F4{ -1, -1, -1, -1 });
    std::vector<F4> outBary(MAXC, F4{ 0, 0, 0, 0 });
    std::vector<FemOtherContactInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(FemOtherContactInfoHost));
    std::vector<uint64_t> byX(MAXC, 0ull);
    std::vector<uint32_t> tempByX(MAXC, 0u), idxSorted(MAXC, 0u), sortedByX(MAXC, 0u), total = { 0u };
    DevBuf bPt = makeDev(dev, outPoint), bNp = makeDev(dev, outNormal), bBary = makeDev(dev, outBary);
    DevBuf bInfo = makeDev(dev, outInfo), bByX = makeDev(dev, byX), bTmp = makeDev(dev, tempByX);
    DevBuf bIdx = makeDev(dev, idxSorted), bSorted = makeDev(dev, sortedByX), bTotal = makeDev(dev, total);

    SbContactGenPush push = {};
    push.tolerenceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.restDistance = bRest.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.shapeToRigidRemapTable = bRemap.addr;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.stackSizeBytes = 64u;               // four uint4 slots
    push.softBodyRigidPairs = 0ull;
    push.numSoftBodyRigidPairs = 0u;
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.w.outPoint = bPt.addr; push.w.outNormalPen = bNp.addr; push.w.outBarycentric = bBary.addr;
    push.w.outContactInfo = bInfo.addr; push.w.totalContactCount = bTotal.addr;
    push.w.contactByX = bByX.addr; push.w.tempContactByX = bTmp.addr;
    push.w.contactIndexSortedByX = bIdx.addr; push.w.contactSortedByX = bSorted.addr;
    push.w.maxNumContacts = MAXC;

    // ---- what must come out, worked out BEFORE the dispatch -------------------
    float margin0 = 0.0f, margin1 = 0.0f, margin2 = 0.0f, marginN = 0.0f;
    int win0 = -1, win1 = -1, win2 = -1, winN = -1;
    const HV3 want0 = supportMaxX(scale0, identQ,   rigidQ0, p0, true,  margin0, win0);
    const HV3 want1 = supportMaxX(scale0, identQ,   identQ,  p1, true,  margin1, win1);
    const HV3 want2 = supportMaxX(scale2, meshRot2, identQ,  p2, true,  margin2, win2);
    const HV3 naive2 = supportMaxX(scale2, meshRot2, identQ, p2, false, marginN, winN);
    std::printf("  predicted: pair0 vertex %d (%.6f %.6f %.6f) gap %.6f margin %.4f\n",
                win0, want0.x, want0.y, want0.z, -want0.x, margin0);
    std::printf("  predicted: pair1 gap %.6f -- outside the 0.3 contact distance, NO contact\n",
                -want1.x);
    std::printf("  predicted: pair2 vertex %d (%.6f %.6f %.6f) gap %.6f margin %.4f "
                "(scale-rotation-blind would give %.6f)\n",
                win2, want2.x, want2.y, want2.z, -want2.x, margin2, -naive2.x);

    // The premises the prediction rests on, asserted rather than assumed.
    PXG_CHECK_MSG(margin0 > 0.02f && margin2 > 0.02f,
                  "the winning vertex must be unique or the witness point is not predictable "
                  "(margins %.5f, %.5f)", margin0, margin2);
    PXG_CHECK_MSG(win0 == 0 && win2 == 7,
                  "the supporting vertices must sit at the two ENDS of the blob's vertex array "
                  "(got %d and %d) -- that is what makes a one-float4 shift of the array base, "
                  "or a short vertex count, observable", win0, win2);
    PXG_CHECK_MSG(-want0.x > 0.0f && -want0.x < 0.3f && -want2.x > 0.0f && -want2.x < 0.3f,
                  "both hulls must be SEPARATED and inside the 0.3 contact distance "
                  "(gaps %.5f, %.5f)", -want0.x, -want2.x);
    PXG_CHECK_MSG(-want1.x > 0.3f && -want1.x < 0.3f + scale0.y,
                  "the far box must sit in the band between the true contact distance (0.3) "
                  "and the wrongly-inflated one (0.3 + scale.y = %.2f), got gap %.5f -- outside "
                  "that band it stops testing the inflation rule",
                  0.3f + scale0.y, -want1.x);
    PXG_CHECK_MSG(want0.y > 0.0f && want0.z > 0.0f && (want0.y + want0.z) < 2.0f &&
                  want2.y > 0.0f && want2.z > 0.0f && (want2.y + want2.z) < 2.0f,
                  "both corners must project INSIDE the tet's x = 0 face triangle, or the "
                  "nearest feature is an edge and the witness moves");
    PXG_CHECK_MSG(std::fabs(want2.x - naive2.x) > 0.01f,
                  "pair 2 exists to catch a vertex2Shape that ignores the mesh-scale rotation; "
                  "if the faithful and the blind answer agree (%.6f vs %.6f) it catches nothing",
                  want2.x, naive2.x);

    Kernel k("softBodyContactGenBda.spv", "sb_primitiveContactGenLaunch", {}, sizeof(SbContactGenPush));
    k.dispatch(/*gridX*/ 512u, {}, push, 1u, 1u);     // the engine's grid, SB_CG_GRID
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    const auto gPt = downloadBuffer<F4>(dev, bPt.h, MAXC);
    const auto gNp = downloadBuffer<F4>(dev, bNp.h, MAXC);
    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, bInfo.h, MAXC);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        std::printf("  contact %u: point=(%.6f %.6f %.6f rest=%.3f) normal=(%.5f %.5f %.5f) "
                    "pen=%.6f mat=%u\n", i, gPt[i].x, gPt[i].y, gPt[i].z, gPt[i].w,
                    gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w, gInfo[i].rigidMatInd & 0x7fffffffu);

    PXG_CHECK_MSG(n == 3u,
                  "the near BOX, the CONVEXMESH and the overlapping BOX must each emit one "
                  "contact and the far box none, got %u", n);

    bool sawBox = false, sawConvex = false, sawOverlap = false;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t mat = gInfo[i].rigidMatInd & 0x7fffffffu;
        PXG_CHECK_MSG(mat != 22u,
                      "the far box is 0.4 away and the contact distance is 0.3, so it must NOT "
                      "produce a contact.  It does if the hull branch inflates the contact "
                      "distance by a radius the way the sphere branch does");
        PXG_CHECK_MSG(std::fabs(gNp[i].x - 1.0f) < 1e-3f && std::fabs(gNp[i].y) < 1e-3f &&
                      std::fabs(gNp[i].z) < 1e-3f,
                      "the normal is +x, got (%.5f %.5f %.5f)", gNp[i].x, gNp[i].y, gNp[i].z);

        if (mat == 24u)
        {
            // The OVERLAP: EPA's answer, not GJK's.  Only the depth is decidable -- the
            // witness pair is any point of the slab (see the geometry note above).
            sawOverlap = true;
            PXG_CHECK_MSG(std::fabs(gNp[i].w - WANT_PEN3) < 2e-3f,
                          "the overlapping box pokes 0.05 through the x = 0 face and every "
                          "other escape is at least 0.26, so the depth is %.5f, got %.6f",
                          WANT_PEN3, gNp[i].w);
            PXG_CHECK(std::fabs(gPt[i].w - 0.06f) < 1e-6f);
            continue;
        }

        const HV3& want = (mat == 21u) ? want0 : want2;
        PXG_CHECK_MSG(std::fabs(gPt[i].x) < 1e-3f &&
                      std::fabs(gPt[i].y - want.y) < 1e-3f && std::fabs(gPt[i].z - want.z) < 1e-3f,
                      "contact %u (material %u): the witness on the TET is the winning hull "
                      "corner projected onto x = 0, (0, %.6f, %.6f), got (%.6f %.6f %.6f)",
                      i, mat, want.y, want.z, gPt[i].x, gPt[i].y, gPt[i].z);
        PXG_CHECK_MSG(std::fabs(gNp[i].w + want.x) < 1e-3f,
                      "contact %u (material %u): the separation is -corner.x = %.6f, got %.6f.  "
                      "A hull folds NO radius into the contact distance and subtracts none "
                      "from the penetration", i, mat, -want.x, gNp[i].w);
        if (mat == 21u) { sawBox = true;    PXG_CHECK(std::fabs(gPt[i].w - 0.03f) < 1e-6f); }
        if (mat == 23u) { sawConvex = true; PXG_CHECK(std::fabs(gPt[i].w - 0.05f) < 1e-6f); }
    }
    PXG_CHECK_MSG(sawBox, "the near BOX produced no contact");
    PXG_CHECK_MSG(sawConvex, "the CONVEXMESH produced no contact");
    PXG_CHECK_MSG(sawOverlap, "the OVERLAPPING box produced no contact -- this is the only pair "
                              "that drives a hull cloud through EPA rather than GJK alone");

    for (PxgBufferHandle h : { bVerts.h, bTets.h, bDesc.h, bBlob.h, bBlobBox.h, bShapes.h,
                               bCms.h, bXf.h, bCd.h, bRest.h, bRemap.h, bStack.h, bPairNum.h,
                               bNeeded.h, bPt.h, bNp.h, bBary.h, bInfo.h, bByX.h, bTmp.h,
                               bIdx.h, bSorted.h, bTotal.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.8 — sb_clothMidphaseGeneratePairsLaunch, the softbody-vs-cloth midphase.
//
// One warp per query TETRAHEDRON walks the CLOTH's BV32 triangle tree and keeps the four
// closest triangles.  The traverser core (TriangleLeafBoundMinMaxTraverser) already has a
// thorough randomized fixture as `gpuslang.npTetTriMid`; what is NEW here and what this
// fixture is aimed at is the DRIVER around it:
//
//   * LoadShapePairWarp<eTETRAHEDRONMESH, eTRIANGLEMESH> -- which half of the pair is which;
//   * the cloth's tree address (mTriMeshData + sizeof(uint4)) and its vertex/index arrays;
//   * the tetrahedron SURFACE-HINT gate;
//   * the world -> CLOTH-SPACE transform of the query tet;
//   * contactDistance[sbCacheRef] + contactDistance[clothCacheRef];
//   * the top-4 cap and the uint4(cmIdx, tetIdx, primIndex, 0) write format.
//
// THE GEOMETRY IS CHOSEN SO THE SEPARATION IS THE VERTICAL GAP, EXACTLY.  satIntersect's
// first term is the triangle's own plane measured against the triangle's own vertices, which
// is identically 0; the answer is then the largest of the four tet-plane terms.  With a tet
// whose bottom face is horizontal at height g above a triangle lying in y = 0 DIRECTLY BELOW
// it, the bottom-face plane contributes exactly g and every side plane contributes a negative
// number (each has at least one triangle vertex on its inner side).  So separation == g, and
// no calibration is involved.
//
// The cloth sits at a translation of (10, 20, 30).  The triangle vertices are read in cloth
// LOCAL space while the tet vertices are world, so a port that dropped transformInv puts the
// tet 37 units away and emits nothing at all.
//
// KILLED BY THIS FIXTURE: the world -> cloth transformInv; reading only one of the two
// contactDistance entries; the surface-hint gate; loading the shape pair by POSITION instead
// of by type; the tree's uint4 header skip; swapping the tet and primitive fields of the
// emitted uint4; the front-face gate; the grid stride; the MaxContacts = 4 cap; the argmin
// direction; and the leaf's primStartIndex shift.
//
// NOT KILLED HERE, WITH REASONS RATHER THAN SILENCE:
//   * the EVOLVING tolerance (min(bestDist, contactDist) rather than contactDist) is provably
//     equivalent on the outputs -- satIntersect returns early only once the partial value
//     already exceeds the tolerance, and the caller's only uses are `< bestDist` and
//     `< contactDist`, both of which the partial value already fails.  It is an optimisation,
//     not a decision.
//   * intersectBoxFullWarp's per-axis displacement is traverser CORE, and the randomized
//     `gpuslang.npTetTriMid` fixture kills a mutation on it (verified, not assumed).
//   * `nbPrimitives = (data >> 1) & 63` cannot be told from `& 31` by EITHER fixture: the two
//     differ only for a leaf holding 32 or more primitives, and a leaf is walked by one warp,
//     so neither test harness builds one.
// ===========================================================================
namespace p88cloth
{
// Gu::BV32DataPacked, 1168 bytes.
struct Bv32NodeHost
{
    float mMin[32][4];
    float mMax[32][4];
    uint32_t mData[32];
    uint32_t mNbNodes;
    uint32_t mDepth;
    uint32_t pad0, pad1;
};
static_assert(sizeof(Bv32NodeHost) == 1168, "Gu::BV32DataPacked is 1168 bytes");

// { PxReal toleranceLength, cmInputs*, transformCache*, bounds*, contactDistance*,
//   gpuShapes*, softbodies*, clothes*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum* }
struct SbClothMidPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, softbodies, clothes;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(SbClothMidPush) == 88, "sb_clothMidphase push blob must be 88 bytes");
} // namespace p88cloth

PXG_TEST(softBodySimClothMidphase)
{
    using namespace p80refit;
    using namespace p88cloth;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the cloth: six triangles in the plane y = 0, wound so the normal is +y ----------
    // (v0, v1, v2) with cross(v1 - v0, v2 - v0) pointing at +y, so the traverser's front-face
    // gate `dot(n, centroid) + d >= 0` admits a tet sitting ABOVE the sheet.
    const uint32_t NTRI = 6u;
    std::vector<F4> triVerts;
    std::vector<TetIdx4> triIdx;
    // Primitive 0 is a DUMMY, parked 1000 units away and never reachable, so the leaf's
    // primStartIndex is 1 rather than 0.  With a zero start the field is invisible: `data >> 7`
    // and `data >> 8` both yield 0 and a mutant on the shift survives.
    const uint32_t TRI_BASE = 1u;
    {
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ 1000.0f, 0.0f, 0.0f, 0.0f });
        triVerts.push_back(F4{ 1000.0f, 0.0f, 1.0f, 0.0f });
        triVerts.push_back(F4{ 1001.0f, 0.0f, 0.0f, 0.0f });
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    for (uint32_t t = 0; t < NTRI; ++t)
    {
        // Six unit right triangles side by side along x, each in its own 2-unit slot -- laid
        // out in REVERSE, so the triangle in slot k has primitive index NTRI-1-k.  The tets
        // below run forwards, which makes tri == 5 - tet and lets the fixture tell the two
        // fields of the emitted uint4 apart.  With a matching layout a mutant that swapped
        // them survived.
        const float ox = float(NTRI - 1u - t) * 2.0f;
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ ox,        0.0f, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox,        0.0f, 1.0f, 0.0f });
        triVerts.push_back(F4{ ox + 1.0f, 0.0f, 0.0f, 0.0f });
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    // Triangle 6: a BACK-FACING one, in tet 0's slot and CLOSER to it (y = 0.05 against a
    // gap of 0.10) than the front-facing triangle underneath.  Its normal is -y, so the
    // traverser's front-face gate must reject it; without the gate it qualifies at a
    // separation of 0.05 and tet 0 reports two triangles instead of one.
    const uint32_t BACKFACING_TRI = TRI_BASE + NTRI;
    {
        const float ox = 0.0f;                       // tet 0's slot
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ ox,        0.05f, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox + 1.0f, 0.05f, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox,        0.05f, 1.0f, 0.0f });   // reversed winding -> n = -y
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    // A STACK of five more front-facing triangles under tet 1, at y = -0.05 .. -0.01.
    // Tet 1's bottom face is at y = 0.06, so together with the triangle of its own slot it
    // sees SIX qualifying triangles at separations 0.06, 0.07, 0.08, 0.09, 0.10, 0.11 -- and
    // the traverser keeps only the FOUR closest.  Without this stack the MaxContacts = 4 cap
    // never binds and a mutant that raises it survives; with it, the cap and the argmin
    // ordering are both observable.
    //
    // THE STACK RUNS FROM FAR TO NEAR ON PURPOSE, so the LAST primitive in the leaf is one of
    // the four that survive the cap.  Ordered the other way the last primitive is the one the
    // cap discards anyway, and a mutant on primStartIndex -- which shifts exactly which
    // primitive falls off the end -- changes nothing observable.
    const uint32_t STACK_TRI0 = TRI_BASE + NTRI + 1u;
    const uint32_t NB_STACK = 5u;
    for (uint32_t k = 0; k < NB_STACK; ++k)
    {
        const float ox = 2.0f;                       // tet 1's slot
        const float y = -0.01f * float(NB_STACK - k);
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ ox,        y, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox,        y, 1.0f, 0.0f });
        triVerts.push_back(F4{ ox + 1.0f, y, 0.0f, 0.0f });
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    const uint32_t NPRIM = uint32_t(triIdx.size());
    DevBuf bTriVerts = makeDev(dev, triVerts), bTriIdx = makeDev(dev, triIdx);
    if (!bTriVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the cloth's BV32 tree: one root node holding one leaf with all six triangles ----
    // mData encoding (Gu::BV32Data): bit 0 = isLeaf, bits 1..6 = nbPrimitives, bits 7.. =
    // primStartIndex.  The root's child 0 IS that leaf, and its box covers the whole sheet.
    std::vector<uint32_t> treeBlob;
    {
        Bv32NodeHost node;
        std::memset(&node, 0, sizeof(node));
        node.mNbNodes = 1u;
        node.mMin[0][0] = -1.0f; node.mMin[0][1] = -1.0f; node.mMin[0][2] = -1.0f;
        node.mMax[0][0] = 12.0f; node.mMax[0][1] =  1.0f; node.mMax[0][2] =  2.0f;
        // isLeaf | nbPrimitives<<1 | primStartIndex<<7.  The leaf covers the real
        // triangles only, starting at TRI_BASE.
        node.mData[0] = 1u | ((NPRIM - TRI_BASE) << 1) | (TRI_BASE << 7);
        // mTriMeshData starts with uint4(nbVerts, nbPrimitives, maxDepth, nbBv32TreeNodes);
        // the node array begins right after it.
        treeBlob.resize(4 + sizeof(Bv32NodeHost) / 4, 0u);
        treeBlob[0] = uint32_t(triVerts.size()); treeBlob[1] = NPRIM; treeBlob[2] = 1u; treeBlob[3] = 1u;
        std::memcpy(treeBlob.data() + 4, &node, sizeof(node));
    }
    DevBuf bTree = makeDev(dev, treeBlob);

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mTriMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTriVerts.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bTriIdx.addr));
    cloth.mNbVerts = uint32_t(triVerts.size());
    cloth.mNbTriangles = NPRIM;
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    // ---- the softbody: one tet per triangle slot, at a chosen gap -------------------------
    // Bottom face horizontal at y = gap, footprint inside the triangle below it (see header).
    // Tet 0 .. 3 are SURFACE tets at increasing gaps; tet 4 is a SURFACE tet too far away;
    // tet 5 sits at the closest gap of all but is INTERIOR (hint 0) and must emit nothing.
    const float gaps[6] = { 0.10f, 0.06f, 0.14f, 0.02f, 0.90f, 0.01f };
    const uint8_t hints[6] = { 1u, 1u, 1u, 1u, 1u, 0u };
    const float CLOTH_X = 10.0f, CLOTH_Y = 20.0f, CLOTH_Z = 30.0f;
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < 6u; ++t)
    {
        const float ox = float(t) * 2.0f, g = gaps[t];
        const uint32_t b = uint32_t(tetVerts.size());
        // world = cloth-local + the cloth's translation
        auto W = [&](float x, float y, float z) {
            return F4{ x + CLOTH_X, y + CLOTH_Y, z + CLOTH_Z, 1.0f };
        };
        // The winding matters: isValidTet demands dot(v1-v0, cross(v2-v0, v3-v0)) > 1e-9, so
        // the bottom face is listed counter-clockwise seen from BELOW.  The obvious ordering
        // gives -0.16 and the whole tet is silently skipped.
        tetVerts.push_back(W(ox + 0.2f, g,        0.2f));
        tetVerts.push_back(W(ox + 0.2f, g,        0.6f));
        tetVerts.push_back(W(ox + 0.6f, g,        0.2f));
        tetVerts.push_back(W(ox + 0.2f, g + 1.0f, 0.2f));
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
    }
    std::vector<uint32_t> hintWords((6u + 3u) / 4u, 0u);
    for (uint32_t t = 0; t < 6u; ++t)
        reinterpret_cast<uint8_t*>(hintWords.data())[t] = hints[t];
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets), bHints = makeDev(dev, hintWords);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = 6u;
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  The CLOTH is shapeRef0 and the SOFTBODY shapeRef1, i.e. the SWAPPED
    // order, so LoadShapePairWarp has to find the tet mesh by TYPE rather than by position.
    const uint32_t GEOM_TRIANGLEMESH_H = 8u;
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;      shapes[0].particleOrSoftbodyId = 0u; shapes[0].srw = 1.0f;
    shapes[1].type = GEOM_TETRAHEDRONMESH_H;   shapes[1].particleOrSoftbodyId = 0u; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*cloth cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.  Slot 0 = the softbody (unused),
    // slot 1 = the cloth, translated so the transformInv is observable.
    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f;                                    // slot 0 identity
    xf[8 + 3] = 1.0f;                                 // slot 1 identity rotation
    xf[8 + 4] = CLOTH_X; xf[8 + 5] = CLOTH_Y; xf[8 + 6] = CLOTH_Z;
    DevBuf bXf = makeDev(dev, xf);

    // contactDistance is summed over BOTH cache refs: 0.05 + 0.15 = 0.20.  Reading only one
    // of them gives 0.05 or 0.15, and the gap ladder above straddles both.
    std::vector<float> cd = { 0.05f, 0.15f, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);

    const uint32_t CAP = 64u;
    std::vector<uint32_t> stack(CAP * 4u, 0xFFFFFFFFu);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 0u };
    DevBuf bPairNum = makeDev(dev, pairNum);

    SbClothMidPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.bounds = 0ull;                 // ABI parity only — the kernel never reads it
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.clothes = bCloth.addr;
    push.stackSizeBytes = CAP * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;

    Kernel k("softBodyClothMidphaseBda.spv", "sb_clothMidphaseGeneratePairsLaunch",
             {}, sizeof(SbClothMidPush));
    // The engine's grid (PxgNarrowphaseCore.cpp:4617): x = SB_SBMIDPHASE = 1024, y = numTests.
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 1u, 1u);
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
    std::printf("  %u pair(s)\n", n);
    for (uint32_t i = 0; i < n && i < CAP; ++i)
        std::printf("    pair %u: cmIdx=%u tet=%u tri=%u (w=%u)\n", i,
                    gStack[i * 4 + 0], gStack[i * 4 + 1], gStack[i * 4 + 2], gStack[i * 4 + 3]);

    // What must come out, decided by the geometry alone:
    //   tet 0 -> its own slot triangle (primitive 5) and NOT the back-facing one;
    //   tet 1 -> the FOUR closest of the six it sees: its slot triangle (4) and the first
    //            three of the stack (7, 8, 9) -- 10 and 11 are further and are dropped;
    //   tet 2 -> primitive 3; tet 3 -> primitive 2;
    //   tet 4 is 0.90 away and tet 5 is interior, so neither appears.
    PXG_CHECK_MSG(n == 7u,
                  "three single-triangle tets plus tet 1's capped four = 7 pairs, got %u", n);

    std::vector<std::set<uint32_t>> got(6);
    uint32_t nBackFacing = 0u, bogus = 0u;
    for (uint32_t i = 0; i < n && i < CAP; ++i)
    {
        const uint32_t cm = gStack[i * 4 + 0], tet = gStack[i * 4 + 1], tri = gStack[i * 4 + 2];
        PXG_CHECK_MSG(cm == 0u, "pair %u must carry cmIdx 0 (blockIdx.y), got %u", i, cm);
        PXG_CHECK_MSG(gStack[i * 4 + 3] == 0u, "the fourth word of the pair is always 0");
        if (tri == BACKFACING_TRI) { ++nBackFacing; continue; }
        if (tet < 6u) got[tet].insert(tri); else ++bogus;
    }
    PXG_CHECK_MSG(bogus == 0u, "%u pair(s) named a tetrahedron outside the mesh", bogus);
    PXG_CHECK_MSG(nBackFacing == 0u,
                  "triangle %u is BACK-FACING and 0.05 from tet 0 -- closer than the "
                  "front-facing one it competes with -- so the front-face gate has to reject "
                  "it (%u pair(s) named it)", BACKFACING_TRI, nBackFacing);

    auto want = [&](uint32_t tet, std::set<uint32_t> expect) {
        std::string g, e;
        for (uint32_t v : got[tet]) g += " " + std::to_string(v);
        for (uint32_t v : expect)  e += " " + std::to_string(v);
        PXG_CHECK_MSG(got[tet] == expect, "tet %u must pair with{%s }, got{%s }",
                      tet, e.c_str(), g.c_str());
    };
    want(0u, { TRI_BASE + NTRI - 1u - 0u });
    want(1u, { TRI_BASE + NTRI - 1u - 1u, STACK_TRI0 + 2u, STACK_TRI0 + 3u, STACK_TRI0 + 4u });
    want(2u, { TRI_BASE + NTRI - 1u - 2u });
    want(3u, { TRI_BASE + NTRI - 1u - 3u });
    want(4u, {});
    want(5u, {});
    PXG_CHECK_MSG(got[1].size() == 4u,
                  "tet 1 sees SIX qualifying triangles and the traverser keeps four; a raised "
                  "MaxContacts shows up here (got %zu)", got[1].size());
    PXG_CHECK_MSG(got[4].empty(),
                  "tet 4 is 0.90 above its triangle and the contact distance is 0.20 -- a port "
                  "that read only ONE of the two contactDistance entries changes this");
    PXG_CHECK_MSG(got[5].empty(),
                  "tet 5 is the CLOSEST of all (0.01) but its surface hint is 0, so it is an "
                  "interior tetrahedron and must generate nothing");

    for (PxgBufferHandle h : { bTriVerts.h, bTriIdx.h, bTree.h, bCloth.h, bTetVerts.h, bTets.h,
                               bHints.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h, bStack.h,
                               bPairNum.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.8 step 2 — sb_clothContactGenLaunch, the tet-vs-triangle contact generator.
//
// Each midphase pair becomes a GJK/EPA query between a TETRAHEDRON and a TRIANGLE.  The
// driver itself (pxgSlangGjk / pxgSlangEpa / tetPrimitivesCollide2) is the one P8.6 extracted
// and already carries its own fixtures; what this one is aimed at is the framing:
//
//   * the point clouds -- 4 tet vertices against 3 triangle vertices taken to WORLD;
//   * BOTH barycentric sets, and both gating the contact;
//   * the surface-FACE rejection through the 4-bit mTetMeshSurfaceHint mask;
//   * the two-level filter, whose vertex level is an AND over all three triangle vertices;
//   * the write format: point = the CLOTH witness, normal stored NEGATED, and
//     pairInd0 = cloth / pairInd1 = softbody, which is the OPPOSITE order to the rigid path.
//
// THE NUMBERS ARE PREDICTED FROM THE GEOMETRY.  A tetrahedron sitting a clean distance g
// above a horizontal triangle that its bottom face projects inside has exactly one nearest
// pair: some point of the bottom face against the point below it.  The separation is g and
// the direction is vertical, so pen = +g and the stored normal is (0, -1, 0) negated to
// (0, +1, 0)... which is to say the sign convention itself is what gets asserted, not a
// fitted constant.
//
// KILLED BY THIS FIXTURE: the normal's negation; storing the softbody witness instead of the
// cloth one; the pairInd0/pairInd1 order; the surface-face rejection; BOTH levels of the
// filter, including the vertex level's AND (a partial filter is the only configuration that
// separates it from an OR -- with all three vertices filtered the two agree); and reading
// either single contactDistance entry instead of their sum.
//
// NOT KILLED, FOR STRUCTURAL REASONS RATHER THAN OVERSIGHT:
//   * the triangle's inner-sphere radius, upstream's literal 1e-6f, reaches nothing but EPA's
//     convergence epsilon and no separated pair runs EPA at all.  Same family as the box
//     inner sphere in P8.7, and recorded for the same reason.
//   * BOTH barycentric gates are defensive and cannot fail here.  A GJK witness lies ON the
//     hull it came from by construction, so its barycentric coordinates are inside the
//     simplex, and the [-1e-5, 1+1e-5] window only catches a numerically degenerate solve.
//     Deleting either gate changes nothing this fixture -- or any well-conditioned one -- can
//     see.  They are transcribed because upstream has them, not because a test drives them.
// ===========================================================================
namespace p88cg
{
// The writer is PxgSoftBodyContactWriter (PxgSoftBodyCore.h:242): six pointers + PxU32.
struct SbClothCgPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, softbodies, clothes;
    uint64_t stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad76;
    uint64_t filterPairs;
    uint32_t nbFilterPairs, pad92;
    uint64_t stackSizeNeededOnDevice;
    uint64_t outPoint, outNormalPen, outBarycentric0, outBarycentric1, outContactInfo,
             totalContactCount;
    uint32_t maxContacts, pad156;
};
static_assert(sizeof(SbClothCgPush) == 160, "sb_clothContactGen push blob must be 160 bytes");

// PxgFemFemContactInfo (PxgFEMCore.h:51), 16 B.
struct FemFemInfoHost { uint32_t pairInd0Lo, pairInd0Hi, pairInd1, auxInd; };
static_assert(sizeof(FemFemInfoHost) == 16, "PxgFemFemContactInfo is 16 bytes");

// PxgNonRigidFilterPair (PxgSimulationCoreDesc.h:235), 16 B.
struct NonRigidPairHost { uint32_t index0Lo, index0Hi, index1, index2; };
static_assert(sizeof(NonRigidPairHost) == 16, "PxgNonRigidFilterPair is 16 bytes");

static const uint32_t PXG_SHIFT = 20u;
static inline uint32_t encodeId(uint32_t id, uint32_t elem) { return (id << PXG_SHIFT) | elem; }
static const uint32_t MAX_TET = 0x000fffffu, MAX_VTX = 0x000fffffu;
} // namespace p88cg

PXG_TEST(softBodySimClothContactGen)
{
    using namespace p80refit;
    using namespace p88cg;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the cloth: four triangles in y = 0, wound so the normal is +y --------------------
    const uint32_t NTRI = 5u;
    std::vector<F4> triVerts;
    std::vector<TetIdx4> triIdxArr;
    // Laid out in REVERSE, so the triangle in tet t's slot is primitive NTRI-1-t.  With a
    // matching layout every encoded index is 0 and a mutant that swapped pairInd0 with
    // pairInd1 survived -- the two encodings were literally the same number.
    for (uint32_t t = 0; t < NTRI; ++t)
    {
        const float ox = float(NTRI - 1u - t) * 4.0f;
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ ox - 1.0f, 0.0f, -1.0f, 0.0f });
        triVerts.push_back(F4{ ox - 1.0f, 0.0f,  2.0f, 0.0f });
        triVerts.push_back(F4{ ox + 2.0f, 0.0f, -1.0f, 0.0f });
        triIdxArr.push_back({ b, b + 1u, b + 2u, 0u });
    }
    DevBuf bTriVerts = makeDev(dev, triVerts), bTriIdx = makeDev(dev, triIdxArr);
    if (!bTriVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTriVerts.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bTriIdx.addr));
    cloth.mNbVerts = uint32_t(triVerts.size());
    cloth.mNbTriangles = NTRI;
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    // ---- the softbody: four tets, one per triangle ----------------------------------------
    // Bottom face horizontal at y = gap, wound so isValidTet holds, footprint well inside the
    // triangle below.  The face masks differ on purpose:
    //   tet 0: mask 1 -- face T0 (corners 0,1,2) is the BOTTOM face and is on the surface;
    //   tet 1: mask 1, larger gap, still inside the contact distance;
    //   tet 2: mask 0b1110 -- every face EXCEPT the bottom one.  The contact direction is
    //          vertical, so no surface face opposes it and the pair must be rejected;
    //   tet 3: mask 1, but filtered out per-vertex on all three triangle vertices;
    //   tet 4: mask 1, but filtered out at the TET level -- (this tet) vs (the whole cloth).
    const float GAP0 = 0.20f, GAP1 = 0.35f;
    const float gaps[5] = { GAP0, GAP1, 0.20f, 0.20f, 0.20f };
    const uint8_t masks[5] = { 0x1u, 0x1u, 0xEu, 0x1u, 0x1u };
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < 5u; ++t)
    {
        const float ox = float(t) * 4.0f, g = gaps[t];
        const uint32_t b = uint32_t(tetVerts.size());
        // corners 0,1,2 form the bottom face; corner 3 is the apex above it.
        tetVerts.push_back(F4{ ox,        g,        0.0f, 1.0f });
        tetVerts.push_back(F4{ ox,        g,        1.0f, 1.0f });
        tetVerts.push_back(F4{ ox + 1.0f, g,        0.0f, 1.0f });
        tetVerts.push_back(F4{ ox,        g + 1.0f, 0.0f, 1.0f });
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
    }
    std::vector<uint32_t> hintWords(3u, 0u);
    for (uint32_t t = 0; t < 5u; ++t)
        reinterpret_cast<uint8_t*>(hintWords.data())[t] = masks[t];
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets), bHints = makeDev(dev, hintWords);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = 5u;
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // Cloth first again, so the pair is found by TYPE.
    const uint32_t GEOM_TRIANGLEMESH_H = 8u;
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;    shapes[0].particleOrSoftbodyId = 0u; shapes[0].srw = 1.0f;
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = 0u; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*cloth cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    // Identity cloth transform: this fixture is about the contact, not the transform (the
    // midphase fixture pins that), so the triangle world positions are the stored ones.
    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f; xf[8 + 3] = 1.0f;
    DevBuf bXf = makeDev(dev, xf);

    // 0.16 + 0.14 = 0.30: gaps of 0.20 are inside the SUM and 0.35 is outside.  Neither term
    // alone reaches 0.20, so a port that reads only one of the two cache refs -- whichever one
    // -- loses tet 0's contact.  Values that let one side carry the test on its own let a
    // single-sided read survive.
    std::vector<float> cd = { 0.16f, 0.14f, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);

    // The vertex-level filter for tet 3: all THREE of its triangle's vertices, against the
    // softbody-wide wildcard.  PxgNonRigidFilterPair is searched with index0 as a PxU64 and
    // index1 as a PxU32, so the array must be sorted by (index0, index1).
    std::vector<NonRigidPairHost> filters;
    // Tet 3 pairs with triangle NTRI-1-3: ALL THREE of its vertices are filtered, which is
    // what it takes to skip a triangle.
    for (uint32_t k = 0; k < 3u; ++k)
        filters.push_back({ encodeId(0u, MAX_TET), 0u, encodeId(0u, triIdxArr[NTRI - 1u - 3u].x + k), 0u });
    // Tet 4 is killed one level up, by the TET-vs-WHOLE-CLOTH entry.  Without it that first
    // filter is never exercised and deleting it changes nothing.
    filters.push_back({ encodeId(0u, 4u), 0u, encodeId(0u, MAX_VTX), 0u });
    // ...and ONE vertex of tet 0's triangle (index NTRI-1) is filtered too.  One is not
    // enough -- the vertex-level test is an AND over all three -- so tet 0 must STILL produce
    // its contact.  This is the only configuration that separates the AND from an OR: with
    // all three filtered the two agree, and a mutant swapping them survived until this pair
    // existed.
    filters.push_back({ encodeId(0u, MAX_TET), 0u, encodeId(0u, triIdxArr[NTRI - 1u].x), 0u });
    std::sort(filters.begin(), filters.end(), [](const NonRigidPairHost& a, const NonRigidPairHost& b) {
        return (a.index0Lo != b.index0Lo) ? (a.index0Lo < b.index0Lo) : (a.index1 < b.index1);
    });
    DevBuf bFilters = makeDev(dev, filters);

    // The midphase output: one pair per tet, each against the triangle of its own slot.
    std::vector<uint32_t> stack;
    for (uint32_t t = 0; t < 5u; ++t)
    { stack.push_back(0u); stack.push_back(t); stack.push_back(NTRI - 1u - t); stack.push_back(0u); }
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 5u };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 16u;
    std::vector<F4> outPoint(MAXC, F4{ -9, -9, -9, -9 }), outNp(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<F4> outB0(MAXC, F4{ -9, -9, -9, -9 }), outB1(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<FemFemInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(FemFemInfoHost));
    std::vector<uint32_t> total = { 0u };
    DevBuf bPt = makeDev(dev, outPoint), bNp = makeDev(dev, outNp);
    DevBuf bB0 = makeDev(dev, outB0), bB1 = makeDev(dev, outB1);
    DevBuf bInfo = makeDev(dev, outInfo), bTotal = makeDev(dev, total);

    SbClothCgPush push = {};
    push.tolerenceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.clothes = bCloth.addr;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.stackSizeBytes = 5u * 16u;
    push.filterPairs = bFilters.addr;
    push.nbFilterPairs = uint32_t(filters.size());
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.outPoint = bPt.addr; push.outNormalPen = bNp.addr;
    push.outBarycentric0 = bB0.addr; push.outBarycentric1 = bB1.addr;
    push.outContactInfo = bInfo.addr; push.totalContactCount = bTotal.addr;
    push.maxContacts = MAXC;

    Kernel k("softBodyClothContactGenBda.spv", "sb_clothContactGenLaunch", {}, sizeof(SbClothCgPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);   // the engine's SB_MESHCG grid
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    const auto gPt = downloadBuffer<F4>(dev, bPt.h, MAXC);
    const auto gNp = downloadBuffer<F4>(dev, bNp.h, MAXC);
    const auto gB0 = downloadBuffer<F4>(dev, bB0.h, MAXC);
    const auto gB1 = downloadBuffer<F4>(dev, bB1.h, MAXC);
    const auto gInfo = downloadBuffer<FemFemInfoHost>(dev, bInfo.h, MAXC);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        std::printf("  contact %u: point=(%.5f %.5f %.5f) normal=(%.5f %.5f %.5f) pen=%.6f\n"
                    "              bary0=(%.4f %.4f %.4f %.4f) bary1=(%.4f %.4f %.4f %.4f) "
                    "cloth=0x%05x soft=0x%05x\n",
                    i, gPt[i].x, gPt[i].y, gPt[i].z, gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w,
                    gB0[i].x, gB0[i].y, gB0[i].z, gB0[i].w, gB1[i].x, gB1[i].y, gB1[i].z, gB1[i].w,
                    gInfo[i].pairInd0Lo, gInfo[i].pairInd1);

    PXG_CHECK_MSG(n == 1u,
                  "only tet 0 may produce a contact: tet 1 is 0.35 away against a contact "
                  "distance of 0.30, tet 2's bottom face is not a SURFACE face, tet 3 is "
                  "filtered on all three triangle vertices and tet 4 is filtered at the TET "
                  "level.  Tet 0 has ONE of its triangle's vertices filtered, which must NOT "
                  "be enough.  Got %u", n);
    if (n == 0u || n > MAXC) { /* the checks below would read poison */ }
    else
    {
        // The pair identity, and its ORDER: pairInd0 is the cloth, pairInd1 the softbody.
        // Tet 0 pairs with triangle 3, so the two encodings are DIFFERENT numbers and the
        // order of the two fields is observable.
        PXG_CHECK_MSG(gInfo[0].pairInd0Lo == encodeId(0u, NTRI - 1u) &&
                      gInfo[0].pairInd1 == encodeId(0u, 0u),
                      "pairInd0 must be the CLOTH index (triangle %u) and pairInd1 the "
                      "SOFTBODY index (tet 0); got 0x%08x / 0x%08x",
                      NTRI - 1u, gInfo[0].pairInd0Lo, gInfo[0].pairInd1);
        // The witness on the cloth is on the triangle's plane, y = 0.
        PXG_CHECK_MSG(std::fabs(gPt[0].y) < 1e-4f,
                      "the stored point is the witness on the CLOTH, so it lies in y = 0 "
                      "(got y = %.6f).  Storing the softbody witness instead gives y = %.2f",
                      gPt[0].y, GAP0);
        // pen is the separation along the un-negated direction: exactly the gap.
        PXG_CHECK_MSG(std::fabs(gNp[0].w - GAP0) < 2e-3f,
                      "the penetration is the witness separation, i.e. the gap %.2f (got %.6f)",
                      GAP0, gNp[0].w);
        // THE SIGN, derived rather than read off.  tetPrimitivesCollide2's direction points
        // from cloud A toward cloud B -- here from the SOFTBODY toward the CLOTH, i.e. -y --
        // which is what makes pen = dot(wB - wA, direction) come out POSITIVE for a separated
        // pair.  The writer stores -direction, so the stored normal points from the cloth up
        // at the softbody: (0, +1, 0).  (The same convention as the rigid path, where the
        // sphere is cloud A and the normal came out pointing at the tet.)
        PXG_CHECK_MSG(std::fabs(gNp[0].x) < 1e-3f && std::fabs(gNp[0].z) < 1e-3f &&
                      std::fabs(gNp[0].y - 1.0f) < 1e-3f,
                      "the stored normal is -direction and points from the cloth at the "
                      "softbody, so a tet directly above the sheet gives (0, +1, 0); got "
                      "(%.5f %.5f %.5f)", gNp[0].x, gNp[0].y, gNp[0].z);
        // ...and the two are locked together: pen must be the separation measured along the
        // UN-negated direction.  This is the assertion that catches a missing negation, since
        // it fails if the normal is stored un-negated while pen keeps its sign.
        PXG_CHECK_MSG(std::fabs(gNp[0].w - (-(gPt[0].y - GAP0) * gNp[0].y)) < 2e-3f,
                      "pen (%.6f) must equal dot(wB - wA, -storedNormal)", gNp[0].w);
        // Both barycentric sets are convex combinations; the triangle one has w == 0 and the
        // tet one puts no weight on the apex (corner 3), which is a whole unit above.
        const float s0 = gB0[0].x + gB0[0].y + gB0[0].z;
        const float s1 = gB1[0].x + gB1[0].y + gB1[0].z + gB1[0].w;
        PXG_CHECK_MSG(std::fabs(s0 - 1.0f) < 1e-3f && gB0[0].w == 0.0f,
                      "the TRIANGLE barycentric sums to 1 with a zero fourth component "
                      "(got %.5f, w = %.5f)", s0, gB0[0].w);
        PXG_CHECK_MSG(std::fabs(s1 - 1.0f) < 1e-3f,
                      "the TET barycentric sums to 1 (got %.5f)", s1);
        PXG_CHECK_MSG(gB1[0].w < 1e-3f,
                      "the tet witness is on the BOTTOM face, so the apex (component w) "
                      "carries no weight; got %.5f", gB1[0].w);
    }
    PXG_CHECK_MSG(gPt[1].x == -9.0f,
                  "nothing may be written past the contact count");

    for (PxgBufferHandle h : { bTriVerts.h, bTriIdx.h, bCloth.h, bTetVerts.h, bTets.h, bHints.h,
                               bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h, bFilters.h, bStack.h,
                               bPairNum.h, bNeeded.h, bPt.h, bNp.h, bB0.h, bB1.h, bInfo.h, bTotal.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.8 step 3 — sb_clothVertMidphaseGeneratePairsLaunch, the vertex-side midphase.
//
// The opposite direction from step 1: a cloth VERTEX queries the SOFTBODY's tetrahedron tree,
// AABB against AABB.  Every surviving primitive is written -- there is no top-N cap here --
// through pushOntoStackFullWarp: one atomicAdd of the population count, each qualifying lane
// storing at its scan offset.
//
// The geometry is all boxes, so every expected pair follows from arithmetic:
//   * the query volume is a CUBE of half-width cDistance around the vertex, not a sphere;
//   * a leaf tet passes when its own world AABB touches that cube -- PxBounds3::intersects
//     counts touching, so the comparisons are not strict;
//   * a degenerate tet fails isValidTet and is dropped whatever its box says.
//
// It also pins the BROADPHASE EARLY-OUT, whose sign is easy to invert: with
// nbCollisionPairUpdatesPerTimestep > 1 the cloth's bounds are SHRUNK by
// (n-1) * contactDistance and the pair is dropped if the shrunken box misses the softbody's.
// Reading that as an inflation turns a skip into a no-op.
//
// KILLED BY THIS FIXTURE: the early-out's sign; the cube-vs-sphere query volume; the
// isValidTet gate; PxBounds3::intersects counting touching; the vert/tet field order; the
// summed contact distance; inflating only one side of the query cube; the stack-capacity
// guard; the scan prefix in pushOntoStackFullWarp; and the atomicAdd taking the population
// count rather than 1.
//
// Six of those needed the geometry sharpened first, and each took a fixture change rather
// than a shader change: a degenerate tet whose flat box the query still MEETS (placed
// elsewhere the AABB test rejects first and the validity gate is untestable); a query cube
// touching a box boundary EXACTLY, which needs contact distances that are exact binary
// fractions (0.0625 + 0.1875) so the sum lands on the boundary rather than a hair past it; a
// vertex reaching in the NEGATIVE direction; a scrambled vertex-to-tet mapping; two ADJACENT
// tets so one ballot has two lanes set; and a deliberately undersized stack.
// ===========================================================================
namespace p88vm
{
struct SbClothVertMidPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, softbodies, clothes;
    uint32_t stackSizeBytes; float nbCollisionPairUpdatesPerTimestep;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(SbClothVertMidPush) == 88, "sb_clothVertMidphase push blob must be 88 bytes");
static_assert(offsetof(SbClothVertMidPush, nbCollisionPairUpdatesPerTimestep) == 68,
              "the PxReal sits at @68, where step 1's pack has dead padding");
} // namespace p88vm

PXG_TEST(softBodySimClothVertMidphase)
{
    using namespace p80refit;
    using namespace p88cloth;   // Bv32NodeHost
    using namespace p88vm;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the softbody: five tets, each a corner tet of a unit box in its own 4-unit slot ---
    // Tet t's world AABB is exactly x in [4t, 4t+1], y in [0, 1], z in [0, 1].  Tet 4 is
    // DEGENERATE (all four corners coplanar), so its AABB is FLAT in y.
    // Tets 5 and 6 are ADJACENT (x in [20, 21] and [21, 22]) rather than four units apart, so
    // a single query cube can reach both and TWO lanes of one ballot qualify at once.  With
    // every ballot holding a single lane the scan prefix is always zero and a mutant that
    // dropped it survived.
    const uint32_t NTET = 7u;
    const float TET_OX[7] = { 0.0f, 4.0f, 8.0f, 12.0f, 16.0f, 20.0f, 21.0f };
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < NTET; ++t)
    {
        const float ox = TET_OX[t];
        const uint32_t b = uint32_t(tetVerts.size());
        const float ay = (t == 4u) ? 0.0f : 1.0f;      // tet 4: apex in the base plane
        tetVerts.push_back(F4{ ox,        0.0f, 0.0f, 1.0f });
        tetVerts.push_back(F4{ ox,        0.0f, 1.0f, 1.0f });
        tetVerts.push_back(F4{ ox + 1.0f, 0.0f, 0.0f, 1.0f });
        tetVerts.push_back(F4{ ox,        ay,   0.0f, 1.0f });
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
    }
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets);
    if (!bTetVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // The softbody's BV32 tree: one root node, one leaf holding all five tets.
    std::vector<uint32_t> treeBlob;
    {
        Bv32NodeHost node;
        std::memset(&node, 0, sizeof(node));
        node.mNbNodes = 1u;
        node.mMin[0][0] = -1.0f; node.mMin[0][1] = -1.0f; node.mMin[0][2] = -1.0f;
        node.mMax[0][0] = 25.0f; node.mMax[0][1] =  2.0f; node.mMax[0][2] =  2.0f;
        node.mData[0] = 1u | (NTET << 1) | (0u << 7);
        treeBlob.resize(4 + sizeof(Bv32NodeHost) / 4, 0u);
        treeBlob[0] = uint32_t(tetVerts.size()); treeBlob[1] = NTET; treeBlob[2] = 1u; treeBlob[3] = 1u;
        std::memcpy(treeBlob.data() + 4, &node, sizeof(node));
    }
    DevBuf bTree = makeDev(dev, treeBlob);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mNumTets = NTET;
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the cloth: five query vertices ---------------------------------------------------
    // cDistance = 0.0625 + 0.1875 = 0.25 EXACTLY -- both terms are binary fractions, so the
    // sum carries no rounding error and the touching case below lands ON the boundary rather
    // than a hair past it.  Neither term alone reaches 0.20, so a single-sided read loses
    // pairs.
    //
    // THE VERTEX ORDER IS SCRAMBLED relative to the tets on purpose: with vertex k querying
    // tet k, a mutant that swapped the two fields of the emitted uint4 survives.
    //   v0: (7.80, -0.20, 0.5)  -- 0.20 outside TET 2's box in BOTH x and y.  The Euclidean
    //       distance is 0.283 > 0.25, so a SPHERE query would miss it; the CUBE hits, and
    //       that is what separates the two.
    //   v1: (3.60, 0.5, 0.5)    -- 0.40 outside tet 1's box   -> miss
    //   v2: (-0.10, 0.5, 0.5)   -- 0.10 outside TET 0's box   -> hit
    //   v3: (16.50, 0.10, 0.5)  -- tet 4's flat AABB spans y = [0, 0]; the query cube reaches
    //       y = [-0.15, 0.35] and DOES overlap it, so the box test passes and only isValidTet
    //       can reject.  At y = 0.5 the box test rejects first and the validity gate becomes
    //       untestable -- which is how a mutant deleting it survived.
    //   v4: (11.75, 0.5, 0.5)   -- its cube reaches x = 12.0 EXACTLY, where TET 3's box
    //       begins.  PxBounds3::intersects counts touching, so this is a hit; making the
    //       comparison strict loses precisely this pair and nothing else.
    //   v5: (5.10, 0.5, 0.5)    -- 0.10 BEYOND the far side of TET 1's box, so its cube has to
    //       reach in the NEGATIVE x direction to touch.  Every other hit above reaches
    //       positively, and a mutant that inflated only the max side survived until this
    //       vertex existed.
    //   v6: (21.00, 0.5, 0.5)   -- reaches BOTH tet 5 and tet 6, which are adjacent.  Two
    //       lanes of the same ballot qualify, which is the only way the scan prefix in
    //       pushOntoStackFullWarp becomes observable.
    const float CD_SB = 0.0625f, CD_CLOTH = 0.1875f;
    std::vector<F4> clothVerts = {
        {  7.80f, -0.20f, 0.5f, 1.0f },
        {  3.60f,  0.5f,  0.5f, 1.0f },
        { -0.10f,  0.5f,  0.5f, 1.0f },
        { 16.50f,  0.10f, 0.5f, 1.0f },
        { 11.75f,  0.5f,  0.5f, 1.0f },
        {  5.10f,  0.5f,  0.5f, 1.0f },
        { 21.00f,  0.5f,  0.5f, 1.0f },
    };
    DevBuf bClothVerts = makeDev(dev, clothVerts);

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bClothVerts.addr));
    cloth.mNbVerts = uint32_t(clothVerts.size());
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    // Cloth first, so the pair is resolved by TYPE.
    const uint32_t GEOM_TRIANGLEMESH_H = 8u;
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;    shapes[0].particleOrSoftbodyId = 0u; shapes[0].srw = 1.0f;
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = 0u; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*cloth cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> xf(2 * 8 + 4, 0.0f); xf[3] = 1.0f; xf[8 + 3] = 1.0f;
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> cd = { CD_SB, CD_CLOTH, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);

    // PxBounds3 (24 B each): slot 0 = the softbody, slot 1 = the cloth.  Overlapping, so the
    // early-out does not fire in the main run.
    std::vector<float> bounds = { -1.0f, -1.0f, -1.0f,  25.0f, 2.0f, 2.0f,
                                  -1.0f, -1.0f, -1.0f,  25.0f, 2.0f, 2.0f,
                                   0.0f,  0.0f,  0.0f,   0.0f, 0.0f, 0.0f };
    DevBuf bBounds = makeDev(dev, bounds);

    const uint32_t CAP = 64u;
    std::vector<uint32_t> stack(CAP * 4u, 0xFFFFFFFFu);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 0u };
    DevBuf bPairNum = makeDev(dev, pairNum);

    SbClothVertMidPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.bounds = bBounds.addr;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.clothes = bCloth.addr;
    push.stackSizeBytes = CAP * 16u;
    push.nbCollisionPairUpdatesPerTimestep = 1.0f;   // the early-out is off in the main run
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;

    Kernel k("softBodyClothVertMidphaseBda.spv", "sb_clothVertMidphaseGeneratePairsLaunch",
             {}, sizeof(SbClothVertMidPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 1u, 1u);
    dev->flush();

    const uint32_t nVm = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gVmStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
    std::printf("  %u pair(s)\n", nVm);
    std::set<std::pair<uint32_t, uint32_t>> vmGot;
    for (uint32_t i = 0; i < nVm && i < CAP; ++i)
    {
        std::printf("    pair %u: cmIdx=%u vert=%u tet=%u (w=%u)\n", i,
                    gVmStack[i * 4 + 0], gVmStack[i * 4 + 1], gVmStack[i * 4 + 2], gVmStack[i * 4 + 3]);
        PXG_CHECK_MSG(gVmStack[i * 4 + 0] == 0u, "pair %u must carry cmIdx 0", i);
        PXG_CHECK_MSG(gVmStack[i * 4 + 3] == 0u, "the fourth word is always 0");
        vmGot.insert({ gVmStack[i * 4 + 1], gVmStack[i * 4 + 2] });
    }

    PXG_CHECK_MSG(nVm == 6u,
                  "v0 is 0.20 outside tet 2's box in two axes, v2 is 0.10 outside tet 0's, "
                  "v4's cube touches tet 3's exactly and v5 reaches tet 1's from BEYOND it; "
                  "v6 reaches BOTH adjacent tets 5 and 6; v1 is 0.40 away and v3 lands on a "
                  "DEGENERATE tet.  Six pairs, got %u", nVm);
    PXG_CHECK_MSG(vmGot.count({ 2u, 0u }) == 1u,
                  "cloth vertex 2 must pair with tet 0 -- note the indices DIFFER, which is "
                  "what makes the two fields of the emitted pair distinguishable");
    PXG_CHECK_MSG(vmGot.count({ 0u, 2u }) == 1u,
                  "cloth vertex 0 must pair with tet 2.  Its Euclidean distance to the box is "
                  "0.283, MORE than the 0.25 contact distance -- the query volume is a CUBE, "
                  "and a port that used a sphere loses exactly this pair");
    PXG_CHECK_MSG(vmGot.count({ 4u, 3u }) == 1u,
                  "cloth vertex 4's query cube reaches x = 12.0 exactly, where tet 3's box "
                  "begins.  PxBounds3::intersects counts TOUCHING, so this is a hit; a strict "
                  "comparison loses it");
    PXG_CHECK_MSG(vmGot.count({ 5u, 1u }) == 1u,
                  "cloth vertex 5 sits 0.10 PAST tet 1's box, so its query cube has to reach "
                  "in the negative direction; inflating only the max side loses this one");
    PXG_CHECK_MSG(vmGot.count({ 6u, 5u }) == 1u && vmGot.count({ 6u, 6u }) == 1u,
                  "cloth vertex 6 reaches BOTH adjacent tets, so one ballot has two lanes set "
                  "and they must land at DIFFERENT stack slots.  A scan offset that ignores "
                  "the prefix writes both to the same slot and one pair vanishes from the "
                  "stack while the counter still says six");
    PXG_CHECK_MSG(vmGot.count({ 1u, 1u }) == 0u,
                  "cloth vertex 1 is 0.40 from tet 1's box against a contact distance of 0.25");
    PXG_CHECK_MSG(vmGot.count({ 3u, 4u }) == 0u,
                  "cloth vertex 3's query cube DOES overlap tet 4's flat box, so the AABB test "
                  "passes; tet 4 is degenerate -- its four corners are coplanar -- and only "
                  "isValidTet can drop it");

    // ---- the broadphase early-out ---------------------------------------------------------
    // With n > 1 the cloth's bounds shrink by (n-1) * contactDistance[cloth].  A shrink of
    // 4 * 0.1875 = 0.75 per side pulls the cloth box in to [-0.25, 24.25] x [-0.25, 1.25],
    // which still meets the softbody's -- so this sub-run must behave EXACTLY as the main one.
    // A port that INFLATED instead would also still overlap, so that alone proves nothing;
    // the second sub-run is the one that decides.
    {
        std::vector<uint32_t> pn2 = { 0u };
        DevBuf bPn2 = makeDev(dev, pn2);
        SbClothVertMidPush p2 = push;
        p2.nbCollisionPairUpdatesPerTimestep = 5.0f;
        p2.midphasePairsNum = bPn2.addr;
        Kernel("softBodyClothVertMidphaseBda.spv", "sb_clothVertMidphaseGeneratePairsLaunch",
               {}, sizeof(SbClothVertMidPush)).dispatch(1024u, {}, p2, 1u, 1u);
        dev->flush();
        const uint32_t n2 = downloadBuffer<uint32_t>(dev, bPn2.h, 1)[0];
        PXG_CHECK_MSG(n2 == 6u,
                      "a shrink of 4 * 0.1875 = 0.75 per side still leaves the two boxes "
                      "overlapping, so the pair survives and the same six come out (got %u)", n2);
        dev->destroyBuffer(bPn2.h);
    }
    {
        // A cloth box spanning x in [-1.5, -0.6] overlaps the softbody's, which starts at
        // -1.0.  Its width is 0.9; shrinking by 0.75 per side turns it INSIDE-OUT
        // ([-0.75, -1.35]) and it can no longer meet anything, so the whole pair is skipped.
        // Inflating by 0.75 keeps them overlapping and produces the usual three -- which is
        // what makes this run, and not the one above, the one that pins the sign.
        std::vector<float> b2 = bounds;
        b2[6] = -1.5f; b2[7] = -1.5f; b2[8] = -1.5f;      // cloth min
        b2[9] = -0.6f; b2[10] = 1.5f; b2[11] = 1.5f;      // cloth max
        DevBuf bB2 = makeDev(dev, b2);
        std::vector<uint32_t> pn3 = { 0u };
        DevBuf bPn3 = makeDev(dev, pn3);
        SbClothVertMidPush p3 = push;
        p3.nbCollisionPairUpdatesPerTimestep = 5.0f;
        p3.bounds = bB2.addr;
        p3.midphasePairsNum = bPn3.addr;
        Kernel("softBodyClothVertMidphaseBda.spv", "sb_clothVertMidphaseGeneratePairsLaunch",
               {}, sizeof(SbClothVertMidPush)).dispatch(1024u, {}, p3, 1u, 1u);
        dev->flush();
        const uint32_t n3 = downloadBuffer<uint32_t>(dev, bPn3.h, 1)[0];
        PXG_CHECK_MSG(n3 == 0u,
                      "the shrunken cloth box is empty and cannot meet the softbody, so the "
                      "pair must be skipped entirely (got %u).  This is the run that pins the "
                      "SIGN of the broadphase early-out", n3);
        dev->destroyBuffer(bB2.h); dev->destroyBuffer(bPn3.h);
    }

    // ---- the stack cap --------------------------------------------------------------------
    // A full stack drops the WRITE but not the COUNT: the engine reads the counter back to
    // size the next allocation, so a kernel that stopped counting would under-allocate and
    // lose pairs on the following frame.  With room for two slots the counter must still
    // report all six, and nothing may be written past the second slot.
    {
        const uint32_t SMALL = 2u;
        std::vector<uint32_t> smallStack(CAP * 4u, 0xFFFFFFFFu);
        DevBuf bSmall = makeDev(dev, smallStack);
        std::vector<uint32_t> pn4 = { 0u };
        DevBuf bPn4 = makeDev(dev, pn4);
        SbClothVertMidPush p4 = push;
        p4.stackSizeBytes = SMALL * 16u;
        p4.stackPtr = bSmall.addr;
        p4.midphasePairsNum = bPn4.addr;
        Kernel("softBodyClothVertMidphaseBda.spv", "sb_clothVertMidphaseGeneratePairsLaunch",
               {}, sizeof(SbClothVertMidPush)).dispatch(1024u, {}, p4, 1u, 1u);
        dev->flush();
        const uint32_t n4 = downloadBuffer<uint32_t>(dev, bPn4.h, 1)[0];
        const auto gSmall = downloadBuffer<uint32_t>(dev, bSmall.h, CAP * 4u);
        PXG_CHECK_MSG(n4 == 6u,
                      "a full stack must still COUNT every pair -- the engine sizes its next "
                      "allocation from this number (got %u)", n4);
        uint32_t written = 0u;
        for (uint32_t i = 0; i < CAP; ++i)
            if (gSmall[i * 4 + 0] != 0xFFFFFFFFu) ++written;
        PXG_CHECK_MSG(written <= SMALL,
                      "only %u slots exist, so at most that many pairs may be STORED; %u were "
                      "written", SMALL, written);
        dev->destroyBuffer(bSmall.h); dev->destroyBuffer(bPn4.h);
    }

    for (PxgBufferHandle h : { bTetVerts.h, bTets.h, bTree.h, bDesc.h, bClothVerts.h, bCloth.h,
                               bShapes.h, bCms.h, bXf.h, bCd.h, bBounds.h, bStack.h, bPairNum.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.8 step 4 — sb_clothVertContactGenLaunch, the vertex contact generator.
//
// One thread per (cloth vertex, tetrahedron) pair, no warp cooperation, no GJK: a
// point-vs-tetrahedron closest-point query answers it outright.  The two branches are
// DIFFERENT CONTACTS and the fixture drives both:
//
//   INSIDE  (all four PointOutsideOfPlane4 products >= 0): normal = the winning face's
//           outward normal, pen = -distance, and NO contact-distance test at all;
//   OUTSIDE: rejected when sqDist > cDistance^2, then pen = +distance and the normal is the
//           closest-point direction when it agrees with the face normal past 0.707,
//           otherwise the face normal.
//
// THE GEOMETRY IS A HALF-UNIT TET WITH ONE HORIZONTAL SURFACE FACE, so every expected number
// is a distance to that face and nothing is calibrated:
//   tet corners (0,0,0), (0,0,1), (1,0,0), (0,1,0) -- the y = 0 face is face bit 1.
//   A vertex at (0.2, h, 0.2) is |h| from that face; POSITIVE h is INSIDE the tet, since
//   the tetrahedron sits above its y = 0 face.
// ===========================================================================
namespace p88vc
{
struct SbClothVertCgPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, softbodies, clothes;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum, filterPairs;
    uint32_t nbFilterPairs, pad92;
    uint64_t stackSizeNeededOnDevice;
    uint64_t outPoint, outNormalPen, outBarycentric0, outBarycentric1, outContactInfo,
             totalContactCount;
    uint32_t maxContacts, pad156;
};
static_assert(sizeof(SbClothVertCgPush) == 160, "sb_clothVertContactGen push blob must be 160 bytes");
static_assert(offsetof(SbClothVertCgPush, stackPtr) == 64,
              "stackSizeBytes sits BEFORE the stack pointer here, unlike step 2's pack");
} // namespace p88vc

PXG_TEST(softBodySimClothVertContactGen)
{
    using namespace p80refit;
    using namespace p88cg;    // FemFemInfoHost / NonRigidPairHost / encodeId / MAX_TET / MAX_VTX
    using namespace p88vc;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the softbody: five copies of the same tet, one per test case ---------------------
    // Corners (ox,0,0), (ox,0,1), (ox+1,0,0), (ox,1,0): the y = 0 face is spanned by corners
    // 0, 1, 2 and is FACE BIT 1 of the surface mask.
    const uint32_t NTET = 6u;
    const uint8_t masks[6] = { 0x1u, 0x1u, 0x1u, 0x1u, 0x0u, 0x1u };  // tet 4: no surface face
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < NTET; ++t)
    {
        const float ox = float(t) * 4.0f;
        const uint32_t b = uint32_t(tetVerts.size());
        tetVerts.push_back(F4{ ox,        0.0f, 0.0f, 1.0f });
        tetVerts.push_back(F4{ ox,        0.0f, 1.0f, 1.0f });
        tetVerts.push_back(F4{ ox + 1.0f, 0.0f, 0.0f, 1.0f });
        tetVerts.push_back(F4{ ox,        1.0f, 0.0f, 1.0f });
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
    }
    std::vector<uint32_t> hintWords(2u, 0u);
    for (uint32_t t = 0; t < NTET; ++t)
        reinterpret_cast<uint8_t*>(hintWords.data())[t] = masks[t];
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets), bHints = makeDev(dev, hintWords);
    if (!bTetVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = NTET;
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the cloth: one query vertex per tet ----------------------------------------------
    // cDistance = 0.0625 + 0.1875 = 0.25.
    // The tet's interior is x-ox >= 0, y >= 0, z >= 0 and (x-ox) + y + z <= 1, so a vertex at
    // POSITIVE y inside that footprint is INSIDE the tetrahedron -- the y = 0 face has the tet
    // ABOVE it.  Getting that backwards is how the first draft of this fixture mislabelled two
    // of its own cases; the shader was consistent throughout.
    //   v0 -> tet 0: (0.20, -0.30, 0.20) OUTSIDE, 0.30 below the face -> beyond 0.25, none
    //   v1 -> tet 1: (4.20, -0.10, 0.20) OUTSIDE, 0.10 below          -> pen +0.10
    //   v2 -> tet 2: (8.20,  0.40, 0.20) INSIDE, and 0.40 from the face -- FARTHER than the
    //                0.25 contact distance.  It must still produce a contact, with pen -0.40:
    //                the inside branch has no distance test at all, and this is the case that
    //                says so.
    //   v3 -> tet 3: (12.20, 0.10, 0.20) INSIDE and shallow          -> pen -0.10
    //   v4 -> tet 4: (16.20, 0.10, 0.20) would be inside too, but tet 4 has NO surface face.
    //   v5 -> tet 5: (21.05, -0.20, 0.0) OUTSIDE, and its projection onto the face plane falls
    //                OUTSIDE the face triangle -- so the closest point is the triangle CORNER
    //                (21, 0, 0) and the direction to it, (-0.05, 0.20, 0) normalised, is NOT
    //                the face normal.  It agrees with the face normal to 0.970, past the 0.707
    //                threshold, so the contact normal is that DIRECTION and not the face's.
    //                Every other case here has the witness in the face interior, where the two
    //                coincide and the threshold is invisible.
    //
    // CLOTH VERTEX 0 IS A DUMMY, parked far away and never paired, so cloth vertex t+1 queries
    // tet t.  With the two indices equal both encodings are the same number and a mutant that
    // swapped pairInd0 with pairInd1 survives.
    const float CD_SB = 0.0625f, CD_CLOTH = 0.1875f;
    std::vector<F4> clothVerts = {
        { 500.0f, 500.0f, 500.0f, 1.0f },
        {  0.20f, -0.30f, 0.20f, 1.0f },
        {  4.20f, -0.10f, 0.20f, 1.0f },
        {  8.20f,  0.40f, 0.20f, 1.0f },
        { 12.20f,  0.10f, 0.20f, 1.0f },
        { 16.20f,  0.10f, 0.20f, 1.0f },
        { 21.05f, -0.20f, 0.00f, 1.0f },
    };
    DevBuf bClothVerts = makeDev(dev, clothVerts);

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bClothVerts.addr));
    cloth.mNbVerts = uint32_t(clothVerts.size());
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    const uint32_t GEOM_TRIANGLEMESH_H = 8u;
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;    shapes[0].particleOrSoftbodyId = 0u; shapes[0].srw = 1.0f;
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = 0u; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*cloth cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> xf(2 * 8 + 4, 0.0f); xf[3] = 1.0f; xf[8 + 3] = 1.0f;
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> cd = { CD_SB, CD_CLOTH, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);

    // No filter entries in the main run; a sub-run below turns each of the three OR terms on.
    std::vector<NonRigidPairHost> noFilters = { { 0u, 0u, 0u, 0u } };
    DevBuf bNoFilters = makeDev(dev, noFilters);

    std::vector<uint32_t> stack;
    for (uint32_t t = 0; t < NTET; ++t)
    { stack.push_back(0u); stack.push_back(t + 1u); stack.push_back(t); stack.push_back(0u); }
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { NTET };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 16u;
    std::vector<F4> outPoint(MAXC, F4{ -9, -9, -9, -9 }), outNp(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<F4> outB0(MAXC, F4{ -9, -9, -9, -9 }), outB1(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<FemFemInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(FemFemInfoHost));
    std::vector<uint32_t> total = { 0u };
    DevBuf bPt = makeDev(dev, outPoint), bNp = makeDev(dev, outNp);
    DevBuf bB0 = makeDev(dev, outB0), bB1 = makeDev(dev, outB1);
    DevBuf bInfo = makeDev(dev, outInfo), bTotal = makeDev(dev, total);

    SbClothVertCgPush push = {};
    push.tolerenceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.clothes = bCloth.addr;
    push.stackSizeBytes = NTET * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.filterPairs = bNoFilters.addr;
    push.nbFilterPairs = 0u;
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.outPoint = bPt.addr; push.outNormalPen = bNp.addr;
    push.outBarycentric0 = bB0.addr; push.outBarycentric1 = bB1.addr;
    push.outContactInfo = bInfo.addr; push.totalContactCount = bTotal.addr;
    push.maxContacts = MAXC;

    Kernel k("softBodyClothVertContactGenBda.spv", "sb_clothVertContactGenLaunch",
             {}, sizeof(SbClothVertCgPush));
    k.dispatch(/*gridX*/ 512u, {}, push, 1u, 1u);   // the engine's grid
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    const auto gPt = downloadBuffer<F4>(dev, bPt.h, MAXC);
    const auto gNp = downloadBuffer<F4>(dev, bNp.h, MAXC);
    const auto gB0 = downloadBuffer<F4>(dev, bB0.h, MAXC);
    const auto gB1 = downloadBuffer<F4>(dev, bB1.h, MAXC);
    const auto gInfo = downloadBuffer<FemFemInfoHost>(dev, bInfo.h, MAXC);
    const uint32_t gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];

    // Index the results by the cloth vertex they name.
    int slot[7] = { -1, -1, -1, -1, -1, -1, -1 };
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
    {
        const uint32_t vtx = gInfo[i].pairInd0Lo & 0x000fffffu;
        std::printf("  contact %u: vtx=%u point=(%.5f %.5f %.5f) normal=(%.5f %.5f %.5f) "
                    "pen=%.6f b0=(%.2f %.2f %.2f %.2f) b1=(%.3f %.3f %.3f %.3f)\n",
                    i, vtx, gPt[i].x, gPt[i].y, gPt[i].z, gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w,
                    gB0[i].x, gB0[i].y, gB0[i].z, gB0[i].w, gB1[i].x, gB1[i].y, gB1[i].z, gB1[i].w);
        if (vtx < 7u) slot[vtx] = int(i);
    }

    PXG_CHECK_MSG(n == 4u,
                  "v1 is 0.10 outside, v2 and v3 are inside, v5 reaches a face CORNER; v0 is "
                  "0.30 outside against a contact distance of 0.25 and tet 4 has no surface "
                  "face.  Four, got %u", n);
    PXG_CHECK_MSG(slot[1] < 0,
                  "v0 is OUTSIDE and 0.30 from the face, beyond the 0.25 contact distance");
    PXG_CHECK_MSG(slot[5] < 0, "tet 4's surface mask is 0, so it has no face to be close to");
    PXG_CHECK_MSG(slot[2] >= 0 && slot[3] >= 0 && slot[4] >= 0,
                  "v1, v2 and v3 must all produce a contact (%d %d %d)",
                  slot[2], slot[3], slot[4]);

    // v1: OUTSIDE.  Witness is its projection onto y = 0, penetration POSITIVE.
    {
        const F4 pt = gPt[slot[2]], np = gNp[slot[2]];
        PXG_CHECK_MSG(std::fabs(pt.y) < 1e-5f && std::fabs(pt.x - 4.20f) < 1e-5f &&
                      std::fabs(pt.z - 0.20f) < 1e-5f,
                      "v1's witness is its projection onto the y = 0 face, (4.20, 0, 0.20); "
                      "got (%.6f %.6f %.6f)", pt.x, pt.y, pt.z);
        PXG_CHECK_MSG(std::fabs(np.w - 0.10f) < 1e-5f,
                      "v1 is OUTSIDE, so its penetration is +0.10 (got %.6f)", np.w);
    }
    // v2: INSIDE and FARTHER from the face than the contact distance.  This is the case that
    // proves the inside branch skips the distance test -- applying it here loses the contact.
    {
        const F4 np = gNp[slot[3]];
        PXG_CHECK_MSG(std::fabs(np.w + 0.40f) < 1e-5f,
                      "v2 is INSIDE and 0.40 from the surface face, well beyond the 0.25 "
                      "contact distance, and must still report pen = -0.40 (got %.6f).  A port "
                      "that applied the distance test to the inside branch drops it entirely",
                      np.w);
    }
    // v3: INSIDE and shallow.  Negative penetration, witness still on the surface face.
    {
        const F4 pt = gPt[slot[4]], np = gNp[slot[4]];
        PXG_CHECK_MSG(std::fabs(np.w + 0.10f) < 1e-5f,
                      "v3 is INSIDE, so its penetration is NEGATIVE, -0.10 (got %.6f).  Note "
                      "that v1 sits the same 0.10 from the same face on the OTHER side and "
                      "reports +0.10 -- the sign carries inside/outside, nothing else", np.w);
        PXG_CHECK_MSG(std::fabs(pt.y) < 1e-5f,
                      "even for an inside vertex the stored point is the witness on the "
                      "SURFACE face, y = 0 (got %.6f)", pt.y);
    }
    // v5: the witness is a face CORNER, so the closest-point direction differs from the face
    // normal -- and agrees with it past 0.707, which selects the DIRECTION.
    {
        const F4 pt = gPt[slot[6]], np = gNp[slot[6]];
        PXG_CHECK_MSG(std::fabs(pt.x - 21.0f) < 1e-5f && std::fabs(pt.y) < 1e-5f &&
                      std::fabs(pt.z) < 1e-5f,
                      "v5's projection falls outside the face triangle, so the witness is its "
                      "CORNER (21, 0, 0); got (%.6f %.6f %.6f)", pt.x, pt.y, pt.z);
        const float wantPen = std::sqrt(0.05f * 0.05f + 0.20f * 0.20f);   // 0.206155
        PXG_CHECK_MSG(std::fabs(np.w - wantPen) < 1e-4f,
                      "v5 is OUTSIDE, so pen is the distance to that corner, %.6f (got %.6f)",
                      wantPen, np.w);
        const float nx = -0.05f / wantPen, ny = 0.20f / wantPen;
        PXG_CHECK_MSG(std::fabs(np.x - nx) < 1e-4f && std::fabs(np.y - ny) < 1e-4f &&
                      std::fabs(np.z) < 1e-4f,
                      "the direction to the corner agrees with the face normal to %.3f, past "
                      "the 0.707 threshold, so the contact normal is that DIRECTION "
                      "(%.5f %.5f 0) and not the face's (0 1 0); got (%.5f %.5f %.5f)",
                      ny, nx, ny, np.x, np.y, np.z);
    }
    // barycentric0 is the constant marker; barycentric1 is a real convex combination.
    for (uint32_t v = 2; v <= 4u; ++v)
    {
        const F4 b0 = gB0[slot[v]], b1 = gB1[slot[v]];
        PXG_CHECK_MSG(b0.x == 0.0f && b0.y == 0.0f && b0.z == 0.0f && b0.w == 1.0f,
                      "the cloth side of this contact is ONE vertex, so barycentric0 is the "
                      "degenerate marker (0,0,0,1); got (%g %g %g %g)", b0.x, b0.y, b0.z, b0.w);
        const float s = b1.x + b1.y + b1.z + b1.w;
        PXG_CHECK_MSG(std::fabs(s - 1.0f) < 1e-4f,
                      "barycentric1 is the tet barycentric of the witness and sums to 1 "
                      "(got %.6f)", s);
        PXG_CHECK_MSG(b1.w < 1e-5f,
                      "the witness lies on the y = 0 face, so the apex (component w) carries "
                      "no weight; got %.6f", b1.w);
    }
    // pairInd0 = cloth, pairInd1 = softbody, and the two differ because the encodings do.
    PXG_CHECK_MSG(gInfo[slot[2]].pairInd0Lo == encodeId(0u, 2u) &&
                  gInfo[slot[2]].pairInd1 == encodeId(0u, 1u),
                  "pairInd0 is the CLOTH vertex (2) and pairInd1 the TETRAHEDRON (1) -- the "
                  "two differ, which is what makes their order observable (got 0x%08x / "
                  "0x%08x)", gInfo[slot[2]].pairInd0Lo, gInfo[slot[2]].pairInd1);
    PXG_CHECK_MSG(gNeeded == NTET * 16u,
                  "stackSizeNeededOnDevice reports what the midphase FOUND in bytes, %u; "
                  "got %u", NTET * 16u, gNeeded);

    // ---- the overflow report ---------------------------------------------------------------
    // With the stack sized for two pairs but SIX found, the kernel processes two and must
    // still report six * sizeof(uint4).  Reporting what FITTED under-allocates the next frame,
    // and with numFound == numPairs -- as in the main run -- the two are indistinguishable.
    {
        std::vector<uint32_t> nd2 = { 0u };
        DevBuf bNd2 = makeDev(dev, nd2);
        std::vector<uint32_t> t3 = { 0u };
        DevBuf bT3 = makeDev(dev, t3);
        SbClothVertCgPush p3 = push;
        p3.stackSizeBytes = 2u * 16u;
        p3.stackSizeNeededOnDevice = bNd2.addr;
        p3.totalContactCount = bT3.addr;
        Kernel("softBodyClothVertContactGenBda.spv", "sb_clothVertContactGenLaunch",
               {}, sizeof(SbClothVertCgPush)).dispatch(512u, {}, p3, 1u, 1u);
        dev->flush();
        const uint32_t nd = downloadBuffer<uint32_t>(dev, bNd2.h, 1)[0];
        PXG_CHECK_MSG(nd == NTET * 16u,
                      "the stack held two pairs but six were FOUND, and the report is what was "
                      "found: %u bytes, got %u", NTET * 16u, nd);
        dev->destroyBuffer(bNd2.h); dev->destroyBuffer(bT3.h);
    }

    // ---- the three-way filter --------------------------------------------------------------
    // Each of the OR's three terms, one at a time, must suppress v1's contact on its own.
    // An AND-chain would need all three and lets each single-term run through.
    for (int term = 0; term < 3; ++term)
    {
        std::vector<NonRigidPairHost> f;
        // the pair under test is (cloth vertex 2, tetrahedron 1)
        if (term == 0) f.push_back({ encodeId(0u, MAX_TET), 0u, encodeId(0u, 2u), 0u });
        if (term == 1) f.push_back({ encodeId(0u, 1u), 0u, encodeId(0u, MAX_VTX), 0u });
        if (term == 2) f.push_back({ encodeId(0u, 1u), 0u, encodeId(0u, 2u), 0u });
        DevBuf bF = makeDev(dev, f);
        std::vector<uint32_t> t2 = { 0u };
        DevBuf bT2 = makeDev(dev, t2);
        std::vector<FemFemInfoHost> i2(MAXC);
        std::memset(i2.data(), 0xEE, i2.size() * sizeof(FemFemInfoHost));
        DevBuf bI2 = makeDev(dev, i2);
        SbClothVertCgPush p2 = push;
        p2.filterPairs = bF.addr;
        p2.nbFilterPairs = 1u;
        p2.totalContactCount = bT2.addr;
        p2.outContactInfo = bI2.addr;
        Kernel("softBodyClothVertContactGenBda.spv", "sb_clothVertContactGenLaunch",
               {}, sizeof(SbClothVertCgPush)).dispatch(512u, {}, p2, 1u, 1u);
        dev->flush();
        const uint32_t n2 = downloadBuffer<uint32_t>(dev, bT2.h, 1)[0];
        PXG_CHECK_MSG(n2 == 3u,
                      "filter term %d alone must suppress the (vertex 2, tet 1) contact -- the three tests are "
                      "an OR, so any one of them is enough.  Expected 3 of the 4 contacts, "
                      "got %u", term, n2);
        dev->destroyBuffer(bF.h); dev->destroyBuffer(bT2.h); dev->destroyBuffer(bI2.h);
    }

    for (PxgBufferHandle h : { bTetVerts.h, bTets.h, bHints.h, bDesc.h, bClothVerts.h, bCloth.h,
                               bShapes.h, bCms.h, bXf.h, bCd.h, bNoFilters.h, bStack.h,
                               bPairNum.h, bNeeded.h, bPt.h, bNp.h, bB0.h, bB1.h, bInfo.h, bTotal.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.8 step 5 — sb_clothContactPrepareLaunch, the constraint prepare.
//
// It consumes BOTH contact shapes out of one stream, and clothBC.w is the discriminator:
// w == 0 means the cloth element is a TRIANGLE (step 2), w != 0 means a VERTEX (step 4),
// which is exactly why step 4 writes the degenerate marker (0, 0, 0, 1).  The two paths take
// their friction and their effective mass from different places, so the fixture runs one
// contact of each and computes both by hand:
//
//   velMultiplier = 1 / (clothDenom + softBodyDenom), where clothDenom is the
//     barycentric-squared-weighted sum of the three triangle vertices' inverse masses in the
//     TRIANGLE case and the single vertex's inverse mass in the VERTEX case;
//   friction      = 0.5 * (clothFriction + softbodyFriction), the cloth half coming from the
//     material table via mMaterialIndices in one case and from mDynamicFrictions in the other;
//   normal        = -stored normal (the SECOND negation);
//   pen           = storedPen - (clothRest + softbodyRest) - dot(clothDelta - sbDelta, normal).
//
// Every input is a small exact binary fraction so the reference and the shader agree bit for
// bit rather than within a tolerance.
// ===========================================================================
namespace p88pr
{
struct SbClothPrepPush
{
    uint64_t softbodies, clothes, normalPens, barycentrics0, barycentrics1, contactInfos,
             numContacts, constraints, lambdaNs;
    uint32_t maxContacts, pad76;
    uint64_t softbodyMaterials, clothMaterials;
};
static_assert(sizeof(SbClothPrepPush) == 96, "sb_clothContactPrepare push blob must be 96 bytes");

// PxgSoftBodySoftBodyConstraintBlock — block-of-32 SoA, 1792 B.
struct SsConstraintBlock
{
    float bary0[32][4];
    float bary1[32][4];
    float normalPen[32][4];
    float velMultiplier[32];
    float friction[32];
};
static_assert(sizeof(SsConstraintBlock) == 1792, "the constraint block is 1792 bytes");

// PxsDeformableVolumeMaterialData / PxsDeformableSurfaceMaterialData: 32 B, friction @8.
struct MaterialHost { float youngs, poissons, dynamicFriction, pad[5]; };
static_assert(sizeof(MaterialHost) == 32, "material data is 32 bytes");
} // namespace p88pr

PXG_TEST(softBodySimClothContactPrepare)
{
    using namespace p80refit;
    using namespace p88cg;    // FemFemInfoHost / encodeId
    using namespace p88pr;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the cloth ------------------------------------------------------------------------
    // mAccumulatedDeltaPos carries the accumulated motion in xyz and the INVERSE MASS in w.
    // Four vertices: triangle 0 uses 0,1,2; vertex 3 is the one the vertex-contact names.
    std::vector<F4> clothDelta = {
        { 0.125f, 0.0f,   0.0f,   0.5f  },   // vertex 0: invMass 0.5
        { 0.0f,   0.25f,  0.0f,   0.25f },   // vertex 1: invMass 0.25
        { 0.0f,   0.0f,   0.5f,   2.0f  },   // vertex 2: invMass 2.0
        { 0.5f,   0.25f,  0.125f, 4.0f  },   // vertex 3: invMass 4.0
    };
    std::vector<TetIdx4> clothTris = { { 0u, 1u, 2u, 0u } };
    std::vector<uint16_t> clothMatIdx = { 1u, 0u, 0u, 0u };   // triangle 0 -> material 1
    std::vector<float> clothVertFriction = { 0.0f, 0.0f, 0.0f, 0.75f };  // vertex 3 -> 0.75
    DevBuf bClothDelta = makeDev(dev, clothDelta), bClothTris = makeDev(dev, clothTris);
    DevBuf bClothMatIdx = makeDev(dev, clothMatIdx), bClothFric = makeDev(dev, clothVertFriction);
    if (!bClothDelta.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const float CLOTH_REST = 0.0625f, SB_REST = 0.03125f;
    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bClothDelta.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bClothTris.addr));
    cloth.mMaterialIndices = reinterpret_cast<PxU16*>(size_t(bClothMatIdx.addr));
    cloth.mDynamicFrictions = reinterpret_cast<PxReal*>(size_t(bClothFric.addr));
    cloth.mNbVerts = 4u; cloth.mNbTriangles = 1u;
    cloth.mRestDistance = CLOTH_REST;
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    // ---- the softbody ---------------------------------------------------------------------
    std::vector<F4> sbDelta = {
        { 0.25f,  0.0f,   0.0f,  1.0f  },   // invMass 1.0
        { 0.0f,   0.125f, 0.0f,  0.5f  },
        { 0.0f,   0.0f,   0.25f, 0.25f },
        { 0.125f, 0.125f, 0.0f,  8.0f  },
    };
    // TWO tets, and the contact names tet 1 -- an ODD index, so mMaterialIndices' PxU16 entry
    // sits in the HIGH half of its word.  With an even index a PxU32 read of the same address
    // happens to yield the same number and a mutant on the element width survives.
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u }, { 0u, 1u, 2u, 3u } };
    std::vector<uint16_t> sbMatIdx = { 0u, 1u, 0u, 0u };    // tet 1 -> material 1
    DevBuf bSbDelta = makeDev(dev, sbDelta), bSimTets = makeDev(dev, simTets);
    DevBuf bSbMatIdx = makeDev(dev, sbMatIdx);

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bSbDelta.addr));
    sb.mSimTetIndices = reinterpret_cast<uint4*>(size_t(bSimTets.addr));
    sb.mMaterialIndices = reinterpret_cast<PxU16*>(size_t(bSbMatIdx.addr));
    sb.mRestDistance = SB_REST;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // Material 1 is the live one on both sides; material 0 is a decoy, so an index read that
    // ignored mMaterialIndices and took slot 0 would pick up 0.0 instead.
    std::vector<MaterialHost> sbMaterials(2), clothMaterials(2);
    std::memset(sbMaterials.data(), 0, sbMaterials.size() * sizeof(MaterialHost));
    std::memset(clothMaterials.data(), 0, clothMaterials.size() * sizeof(MaterialHost));
    sbMaterials[1].dynamicFriction = 0.25f;
    clothMaterials[1].dynamicFriction = 0.5f;
    DevBuf bSbMat = makeDev(dev, sbMaterials), bClothMat = makeDev(dev, clothMaterials);

    // ---- two contacts ----------------------------------------------------------------------
    // contact 0: cloth TRIANGLE 0 (clothBC.w == 0) against tet 0
    // contact 1: cloth VERTEX 3   (clothBC.w != 0) against tet 0
    const F4 BC0_TRI = { 0.25f, 0.5f, 0.25f, 0.0f };
    const F4 BC0_VTX = { 0.0f,  0.0f, 0.0f,  1.0f };
    const F4 BC1     = { 0.125f, 0.25f, 0.5f, 0.125f };
    const F4 NP0 = { 0.5f, -0.5f, 0.0f, 0.25f };     // stored normal, stored pen
    const F4 NP1 = { 0.0f,  1.0f, 0.0f, 0.125f };
    std::vector<F4> normalPens = { NP0, NP1 };
    std::vector<F4> bary0 = { BC0_TRI, BC0_VTX };
    std::vector<F4> bary1 = { BC1, BC1 };
    std::vector<FemFemInfoHost> infos = {
        { encodeId(0u, 0u), 0u, encodeId(0u, 1u), 0u },   // cloth triangle 0, tet 1
        { encodeId(0u, 3u), 0u, encodeId(0u, 1u), 0u },   // cloth vertex 3,   tet 1
    };
    DevBuf bNp = makeDev(dev, normalPens), bB0 = makeDev(dev, bary0), bB1 = makeDev(dev, bary1);
    DevBuf bInfo = makeDev(dev, infos);

    // maxContacts is LARGER than the count, so the clamp is a no-op here; a sub-run below
    // makes it smaller.
    std::vector<uint32_t> numContacts = { 2u };
    DevBuf bNum = makeDev(dev, numContacts);

    const float SENT = -7.0f;
    std::vector<SsConstraintBlock> blocks(2);
    std::memset(blocks.data(), 0, blocks.size() * sizeof(SsConstraintBlock));
    for (uint32_t b = 0; b < 2u; ++b)
        for (uint32_t k = 0; k < 32u; ++k)
        { blocks[b].velMultiplier[k] = SENT; blocks[b].friction[k] = SENT;
          blocks[b].normalPen[k][3] = SENT; }
    DevBuf bCons = makeDev(dev, blocks);
    std::vector<float> lambdas(8, SENT);
    DevBuf bLambda = makeDev(dev, lambdas);

    SbClothPrepPush push = {};
    push.softbodies = bDesc.addr;
    push.clothes = bCloth.addr;
    push.normalPens = bNp.addr;
    push.barycentrics0 = bB0.addr;
    push.barycentrics1 = bB1.addr;
    push.contactInfos = bInfo.addr;
    push.numContacts = bNum.addr;
    push.constraints = bCons.addr;
    push.lambdaNs = bLambda.addr;
    push.maxContacts = 8u;
    push.softbodyMaterials = bSbMat.addr;
    push.clothMaterials = bClothMat.addr;

    Kernel k("softBodyClothPrepareBda.spv", "sb_clothContactPrepareLaunch", {}, sizeof(SbClothPrepPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);   // the engine's SB_UPDATEROTATION grid
    dev->flush();

    const auto gBlocks = downloadBuffer<SsConstraintBlock>(dev, bCons.h, 2);
    const auto gLambda = downloadBuffer<float>(dev, bLambda.h, 8);
    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bNum.h, 1)[0];

    // ---- the reference, worked out here rather than read off ------------------------------
    auto sbDenomOf = [&](F4 bc) {
        return bc.x * bc.x * sbDelta[0].w + bc.y * bc.y * sbDelta[1].w +
               bc.z * bc.z * sbDelta[2].w + bc.w * bc.w * sbDelta[3].w;
    };
    auto sbProject = [&](F4 bc) {
        return F4{ bc.x * sbDelta[0].x + bc.y * sbDelta[1].x + bc.z * sbDelta[2].x + bc.w * sbDelta[3].x,
                   bc.x * sbDelta[0].y + bc.y * sbDelta[1].y + bc.z * sbDelta[2].y + bc.w * sbDelta[3].y,
                   bc.x * sbDelta[0].z + bc.y * sbDelta[1].z + bc.z * sbDelta[2].z + bc.w * sbDelta[3].z,
                   0.0f };
    };
    struct Want { float nx, ny, nz, pen, velMul, fric; };
    auto reference = [&](uint32_t idx, F4 bc0, F4 np, uint32_t clothElem, bool triangle) {
        const F4 cd = clothDelta[clothElem];
        const F4 sd = sbProject(BC1);
        const float nx = -np.x, ny = -np.y, nz = -np.z;
        const float pen = np.w - (CLOTH_REST + SB_REST) -
                          ((cd.x - sd.x) * nx + (cd.y - sd.y) * ny + (cd.z - sd.z) * nz);
        float clothDenom, clothFric;
        if (triangle)
        {
            clothDenom = bc0.x * bc0.x * clothDelta[0].w + bc0.y * bc0.y * clothDelta[1].w +
                         bc0.z * bc0.z * clothDelta[2].w;
            clothFric = clothMaterials[clothMatIdx[clothElem]].dynamicFriction;
        }
        else
        {
            clothDenom = clothDelta[clothElem].w;
            clothFric = clothVertFriction[clothElem];
        }
        const float denom = clothDenom + sbDenomOf(BC1);
        const float sbFric = sbMaterials[sbMatIdx[1]].dynamicFriction;
        Want w;
        w.nx = nx; w.ny = ny; w.nz = nz; w.pen = pen;
        w.velMul = (denom == 0.0f) ? 0.0f : 1.0f / denom;
        w.fric = 0.5f * (clothFric + sbFric);
        PX_UNUSED(idx);
        return w;
    };
    const Want w0 = reference(0u, BC0_TRI, NP0, 0u, true);    // triangle 0's vertices are 0,1,2
    const Want w1 = reference(1u, BC0_VTX, NP1, 3u, false);   // cloth vertex 3

    for (uint32_t c = 0; c < 2u; ++c)
    {
        const Want& w = (c == 0u) ? w0 : w1;
        const auto& B = gBlocks[0];
        std::printf("  constraint %u: normal=(%.6f %.6f %.6f) pen=%.6f velMul=%.6f fric=%.6f\n",
                    c, B.normalPen[c][0], B.normalPen[c][1], B.normalPen[c][2],
                    B.normalPen[c][3], B.velMultiplier[c], B.friction[c]);
        PXG_CHECK_MSG(f2u(B.normalPen[c][0]) == f2u(w.nx) && f2u(B.normalPen[c][1]) == f2u(w.ny) &&
                      f2u(B.normalPen[c][2]) == f2u(w.nz),
                      "constraint %u's normal is the stored one NEGATED, (%.6f %.6f %.6f); got "
                      "(%.6f %.6f %.6f)", c, w.nx, w.ny, w.nz,
                      B.normalPen[c][0], B.normalPen[c][1], B.normalPen[c][2]);
        PXG_CHECK_MSG(f2u(B.normalPen[c][3]) == f2u(w.pen),
                      "constraint %u's penetration is storedPen - thickness - "
                      "dot(clothDelta - sbDelta, normal) = %.9g; got %.9g", c, w.pen,
                      B.normalPen[c][3]);
        PXG_CHECK_MSG(f2u(B.velMultiplier[c]) == f2u(w.velMul),
                      "constraint %u's velMultiplier is 1 / (clothDenom + sbDenom) = %.9g; "
                      "got %.9g", c, w.velMul, B.velMultiplier[c]);
        PXG_CHECK_MSG(f2u(B.friction[c]) == f2u(w.fric),
                      "constraint %u's friction is the mean of the two sides, %.9g; got %.9g",
                      c, w.fric, B.friction[c]);
        PXG_CHECK_MSG(gLambda[c] == 0.0f, "lambda %u must be initialised to zero", c);
    }

    // The two paths must actually DIFFER, or the discriminator is untested.
    PXG_CHECK_MSG(f2u(w0.velMul) != f2u(w1.velMul) && f2u(w0.fric) != f2u(w1.fric),
                  "the triangle and vertex paths must produce different numbers, or clothBC.w "
                  "is not being tested at all (velMul %.9g vs %.9g, friction %.9g vs %.9g)",
                  w0.velMul, w1.velMul, w0.fric, w1.fric);
    // The barycentrics are copied through verbatim.
    PXG_CHECK_MSG(gBlocks[0].bary0[0][0] == BC0_TRI.x && gBlocks[0].bary0[1][3] == BC0_VTX.w &&
                  gBlocks[0].bary1[0][3] == BC1.w,
                  "both barycentric sets are copied into the block unchanged");
    PXG_CHECK_MSG(gBlocks[0].velMultiplier[2] == SENT && gLambda[2] == SENT,
                  "nothing may be written past the contact count");
    PXG_CHECK_MSG(gNum == 2u, "the count is unchanged when maxContacts exceeds it (got %u)", gNum);

    // ---- the block-of-32 addressing ----------------------------------------------------------
    // With only two contacts, workIndex & 31 == workIndex and workIndex / 32 == 0, so neither
    // half of the SoA addressing is exercised.  Thirty-four copies of contact 0 put the last
    // two in BLOCK 1, slots 0 and 1.
    {
        const uint32_t NC = 34u;
        std::vector<F4> np2(NC, NP0), b02(NC, BC0_TRI), b12(NC, BC1);
        std::vector<FemFemInfoHost> in2(NC, infos[0]);
        std::vector<uint32_t> n3 = { NC };
        std::vector<SsConstraintBlock> b3(3);
        std::memset(b3.data(), 0, b3.size() * sizeof(SsConstraintBlock));
        for (uint32_t b = 0; b < 3u; ++b)
            for (uint32_t j = 0; j < 32u; ++j) b3[b].velMultiplier[j] = SENT;
        std::vector<float> l3(64, SENT);
        DevBuf bNp2 = makeDev(dev, np2), bB02 = makeDev(dev, b02), bB12 = makeDev(dev, b12);
        DevBuf bIn2 = makeDev(dev, in2), bN3 = makeDev(dev, n3);
        DevBuf bC3 = makeDev(dev, b3), bL3 = makeDev(dev, l3);
        SbClothPrepPush p3 = push;
        p3.normalPens = bNp2.addr; p3.barycentrics0 = bB02.addr; p3.barycentrics1 = bB12.addr;
        p3.contactInfos = bIn2.addr; p3.numContacts = bN3.addr; p3.constraints = bC3.addr;
        p3.lambdaNs = bL3.addr; p3.maxContacts = 64u;
        Kernel("softBodyClothPrepareBda.spv", "sb_clothContactPrepareLaunch", {}, sizeof(SbClothPrepPush))
            .dispatch(1024u, {}, p3, 1u, 1u);
        dev->flush();
        const auto gB3 = downloadBuffer<SsConstraintBlock>(dev, bC3.h, 3);
        for (uint32_t j = 0; j < 32u; ++j)
            PXG_CHECK_MSG(f2u(gB3[0].velMultiplier[j]) == f2u(w0.velMul),
                          "block 0 slot %u must be built", j);
        PXG_CHECK_MSG(f2u(gB3[1].velMultiplier[0]) == f2u(w0.velMul) &&
                      f2u(gB3[1].velMultiplier[1]) == f2u(w0.velMul),
                      "contacts 32 and 33 belong to BLOCK 1, slots 0 and 1 -- a slot index "
                      "that forgot the & 31, or a block index that forgot the / 32, puts them "
                      "somewhere else entirely");
        PXG_CHECK_MSG(gB3[1].velMultiplier[2] == SENT && gB3[2].velMultiplier[0] == SENT,
                      "nothing past contact 33 may be written");
        for (PxgBufferHandle h : { bNp2.h, bB02.h, bB12.h, bIn2.h, bN3.h, bC3.h, bL3.h })
            dev->destroyBuffer(h);
    }

    // ---- the in-place clamp ----------------------------------------------------------------
    // With maxContacts BELOW the count, the kernel must both stop at the limit AND write the
    // clamped value back -- everything downstream reads it, so a kernel that clamped only its
    // own loop would leave the solver walking past the constraints it prepared.
    {
        std::vector<uint32_t> n2 = { 2u };
        DevBuf bN2 = makeDev(dev, n2);
        std::vector<SsConstraintBlock> b2 = blocks;
        DevBuf bC2 = makeDev(dev, b2);
        std::vector<float> l2(8, SENT);
        DevBuf bL2 = makeDev(dev, l2);
        SbClothPrepPush p2 = push;
        p2.maxContacts = 1u;
        p2.numContacts = bN2.addr;
        p2.constraints = bC2.addr;
        p2.lambdaNs = bL2.addr;
        Kernel("softBodyClothPrepareBda.spv", "sb_clothContactPrepareLaunch", {}, sizeof(SbClothPrepPush))
            .dispatch(1024u, {}, p2, 1u, 1u);
        dev->flush();
        const uint32_t gN2 = downloadBuffer<uint32_t>(dev, bN2.h, 1)[0];
        const auto gB2 = downloadBuffer<SsConstraintBlock>(dev, bC2.h, 2);
        PXG_CHECK_MSG(gN2 == 1u,
                      "maxContacts = 1 below a count of 2 must be written BACK to the counter, "
                      "not just applied locally (got %u)", gN2);
        PXG_CHECK_MSG(gB2[0].velMultiplier[1] == SENT,
                      "the second contact is past the clamped count and must be untouched");
        PXG_CHECK_MSG(gB2[0].velMultiplier[0] != SENT, "the first contact must still be built");
        dev->destroyBuffer(bN2.h); dev->destroyBuffer(bC2.h); dev->destroyBuffer(bL2.h);
    }

    for (PxgBufferHandle h : { bClothDelta.h, bClothTris.h, bClothMatIdx.h, bClothFric.h,
                               bCloth.h, bSbDelta.h, bSimTets.h, bSbMatIdx.h, bDesc.h,
                               bSbMat.h, bClothMat.h, bNp.h, bB0.h, bB1.h, bInfo.h, bNum.h,
                               bCons.h, bLambda.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.8 step 6 — sb_solveOutputSCDeltaVLaunch, the last kernel of the chain.
//
// It solves the constraints step 5 prepared and scatters opposite-signed corrections onto both
// bodies.  The fixture runs one contact of each shape and reproduces the solver by hand:
//
//   relDelta  = (cloth side, barycentrically weighted) - (softbody side, ditto)
//   CN        = pen + dot(relDelta, normal)
//   dLambdaN  = max(-CN * denomInv, -lambdaN)     <- the clamp that stops a contact PULLING
//   commonVec = dLambdaN * normal, plus the friction term when the tangential motion exceeds
//               1e-14 and the friction coefficient is non-zero
//
// then `+w * bc * commonVec` on the cloth and `-w * bc * commonVec` on the softbody.
//
// FOUR THINGS THIS FIXTURE IS BUILT TO SEE, none of which a single well-behaved contact shows:
//   * the OPPOSITE SIGNS -- the cloth and softbody corrections must point against each other;
//   * the VERTEX path's two asymmetries: no barycentric factor on the cloth correction, and an
//     atomic guarded by the inverse mass ALONE where the triangle path also demands bc > 1e-6;
//   * the .w CONTRIBUTION COUNT, which the averaging pass downstream divides by;
//   * the two guards -- a zero inverse mass and a barycentric at or below 1e-6 must both
//     suppress their scatter, and the fixture gives contact 0 one of each.
// ===========================================================================
namespace p88sv
{
struct SbClothSolvePush
{
    uint64_t softbodies, clothes, contactInfos, constraints, numContacts, lambdaNs;
};
static_assert(sizeof(SbClothSolvePush) == 48, "sb_solveOutputSCDeltaV push blob must be 48 bytes");
} // namespace p88sv

PXG_TEST(softBodySimClothContactSolve)
{
    using namespace p80refit;
    using namespace p88cg;    // FemFemInfoHost / encodeId
    using namespace p88pr;    // SsConstraintBlock
    using namespace p88sv;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the cloth ------------------------------------------------------------------------
    // mAccumulatedDeltaPos: accumulated motion in xyz, INVERSE MASS in w.
    // Vertex 2 is PINNED (invMass 0), so contact 0's third cloth scatter must be suppressed.
    std::vector<F4> clothAccum = {
        { 0.25f,  0.0f,   0.0f,  0.5f  },
        { 0.0f,   0.125f, 0.0f,  0.25f },
        { 0.0f,   0.0f,   0.25f, 0.0f  },   // PINNED
        { 0.125f, 0.25f,  0.0f,  4.0f  },   // the vertex contact's element
    };
    std::vector<TetIdx4> clothTris = { { 0u, 1u, 2u, 0u } };
    const float SENT = -7.0f;
    std::vector<F4> clothOut(4, F4{ SENT, SENT, SENT, SENT });
    DevBuf bClothAccum = makeDev(dev, clothAccum), bClothTris = makeDev(dev, clothTris);
    DevBuf bClothOut = makeDev(dev, clothOut);
    if (!bClothAccum.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bClothAccum.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bClothTris.addr));
    cloth.mDeltaPos = reinterpret_cast<float4*>(size_t(bClothOut.addr));
    cloth.mNbVerts = 4u; cloth.mNbTriangles = 1u;
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    // ---- the softbody ---------------------------------------------------------------------
    std::vector<F4> sbAccum = {
        { 0.125f, 0.0f,   0.0f,  1.0f },
        { 0.0f,   0.25f,  0.0f,  0.5f },
        { 0.0f,   0.0f,   0.125f, 2.0f },
        { 0.25f,  0.125f, 0.0f,  8.0f },
    };
    std::vector<TetIdx4> simTets = { { 0u, 1u, 2u, 3u } };
    std::vector<F4> sbOut(4, F4{ SENT, SENT, SENT, SENT });
    DevBuf bSbAccum = makeDev(dev, sbAccum), bSimTets = makeDev(dev, simTets);
    DevBuf bSbOut = makeDev(dev, sbOut);

    PxgSoftBody sb = makeZeroDesc();
    sb.mSimDeltaPos = reinterpret_cast<float4*>(size_t(bSbAccum.addr));
    sb.mSimTetIndices = reinterpret_cast<uint4*>(size_t(bSimTets.addr));
    sb.mSimDelta = reinterpret_cast<float4*>(size_t(bSbOut.addr));
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- two constraints ------------------------------------------------------------------
    // contact 0: cloth TRIANGLE.  clothBC.z is 0 exactly, so its third scatter is suppressed by
    //            the barycentric guard AND vertex 2 is pinned -- two independent reasons, and
    //            the shader must honour either.  softBodyBC.w is 1e-7, BELOW the 1e-6 guard.
    // contact 1: cloth VERTEX 3, with friction on so the tangential term runs.
    const F4 BC0_TRI = { 0.5f, 0.5f, 0.0f, 0.0f };
    const F4 BC0_VTX = { 0.0f, 0.0f, 0.0f, 1.0f };
    const F4 BC1_A   = { 0.5f, 0.25f, 0.25f, 1e-7f };
    const F4 BC1_B   = { 0.25f, 0.25f, 0.25f, 0.25f };
    const F4 NP_A = { 0.0f, 1.0f, 0.0f, -0.25f };   // normal +y, penetration -0.25
    const F4 NP_B = { 1.0f, 0.0f, 0.0f, -0.5f };
    // contact 4 is SEPARATED: a POSITIVE penetration makes -CN * denomInv negative, so the
    // clamp against -lambdaN (zero on the first iteration) must hold the impulse at zero.
    // Without the clamp the contact PULLS the bodies together, which no penetrating contact in
    // this fixture can show.
    const F4 NP_C = { 1.0f, 0.0f, 0.0f, 0.5f };
    const float DENOM_A = 2.0f, DENOM_B = 4.0f;
    const float FRIC_A = 0.0f, FRIC_B = 0.5f;       // contact 1 exercises the friction branch

    std::vector<SsConstraintBlock> blocks(1);
    std::memset(blocks.data(), 0, sizeof(SsConstraintBlock));
    auto setC = [&](uint32_t k, F4 b0, F4 b1, F4 np, float denom, float fric) {
        blocks[0].bary0[k][0] = b0.x; blocks[0].bary0[k][1] = b0.y;
        blocks[0].bary0[k][2] = b0.z; blocks[0].bary0[k][3] = b0.w;
        blocks[0].bary1[k][0] = b1.x; blocks[0].bary1[k][1] = b1.y;
        blocks[0].bary1[k][2] = b1.z; blocks[0].bary1[k][3] = b1.w;
        blocks[0].normalPen[k][0] = np.x; blocks[0].normalPen[k][1] = np.y;
        blocks[0].normalPen[k][2] = np.z; blocks[0].normalPen[k][3] = np.w;
        blocks[0].velMultiplier[k] = denom;
        blocks[0].friction[k] = fric;
    };
    setC(0u, BC0_TRI, BC1_A, NP_A, DENOM_A, FRIC_A);
    setC(1u, BC0_VTX, BC1_B, NP_B, DENOM_B, FRIC_B);
    // contact 2 has velMultiplier 0 -- a degenerate constraint that must be SKIPPED, and
    // skipping it must not stop contact 3 from being solved.
    setC(2u, BC0_VTX, BC1_B, NP_B, 0.0f, 0.0f);
    setC(3u, BC0_VTX, BC1_B, NP_B, DENOM_B, 0.0f);
    setC(4u, BC0_VTX, BC1_B, NP_C, DENOM_B, 0.0f);
    DevBuf bCons = makeDev(dev, blocks);

    std::vector<FemFemInfoHost> infos = {
        { encodeId(0u, 0u), 0u, encodeId(0u, 0u), 0u },   // cloth triangle 0, tet 0
        { encodeId(0u, 3u), 0u, encodeId(0u, 0u), 0u },   // cloth vertex 3,   tet 0
        { encodeId(0u, 3u), 0u, encodeId(0u, 0u), 0u },
        { encodeId(0u, 3u), 0u, encodeId(0u, 0u), 0u },
        { encodeId(0u, 3u), 0u, encodeId(0u, 0u), 0u },
    };
    DevBuf bInfo = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { 5u };
    DevBuf bNum = makeDev(dev, numContacts);
    std::vector<float> lambdas = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    DevBuf bLambda = makeDev(dev, lambdas);

    SbClothSolvePush push = {};
    push.softbodies = bDesc.addr;
    push.clothes = bCloth.addr;
    push.contactInfos = bInfo.addr;
    push.constraints = bCons.addr;
    push.numContacts = bNum.addr;
    push.lambdaNs = bLambda.addr;

    Kernel k("softBodyClothSolveBda.spv", "sb_solveOutputSCDeltaVLaunch", {}, sizeof(SbClothSolvePush));
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);
    dev->flush();

    const auto gClothOut = downloadBuffer<F4>(dev, bClothOut.h, 4);
    const auto gSbOut = downloadBuffer<F4>(dev, bSbOut.h, 4);
    const auto gLambda = downloadBuffer<float>(dev, bLambda.h, 5);

    // ---- the reference ---------------------------------------------------------------------
    struct V3 { float x, y, z; };
    auto solve = [&](V3 relDelta, F4 np, float denomInv, float fric, float& lambdaN) {
        const V3 n = { np.x, np.y, np.z };
        const float CN = np.w + relDelta.x * n.x + relDelta.y * n.y + relDelta.z * n.z;
        const float dLN = std::max(-CN * denomInv, -lambdaN);
        lambdaN += dLN;
        V3 cv = { dLN * n.x, dLN * n.y, dLN * n.z };
        const float rn = relDelta.x * n.x + relDelta.y * n.y + relDelta.z * n.z;
        V3 t = { relDelta.x - rn * n.x, relDelta.y - rn * n.y, relDelta.z - rn * n.z };
        const float tsq = t.x * t.x + t.y * t.y + t.z * t.z;
        if (fric != 0.0f && tsq > 1.0e-14f)
        {
            // P9.36: FEMClothExternalSolve.cu:268-269 is `PxRecipSqrt(tanMagSq)` then
            // `tanDir *= invTanMag` -- one reciprocal, three multiplies.  Three of the
            // four ports of this block in this corpus already spelled it that way
            // (femClothClothContactBda:430, npClothCollisionConstraint:182,
            // npSbCollisionConstraint); this pair was the fourth.
            const float tm = std::sqrt(tsq);
            const float invTm = 1.0f / tm;
            t.x *= invTm; t.y *= invTm; t.z *= invTm;
            const float CT = relDelta.x * t.x + relDelta.y * t.y + relDelta.z * t.z;
            float dLT = std::max(0.0f, CT * denomInv);
            dLT = -std::min(dLT, fric * std::fabs(dLN));
            cv.x += 0.5f * dLT * t.x; cv.y += 0.5f * dLT * t.y; cv.z += 0.5f * dLT * t.z;
        }
        return cv;
    };
    auto sbWeighted = [&](F4 bc) {
        return V3{ bc.x * sbAccum[0].x + bc.y * sbAccum[1].x + bc.z * sbAccum[2].x + bc.w * sbAccum[3].x,
                   bc.x * sbAccum[0].y + bc.y * sbAccum[1].y + bc.z * sbAccum[2].y + bc.w * sbAccum[3].y,
                   bc.x * sbAccum[0].z + bc.y * sbAccum[1].z + bc.z * sbAccum[2].z + bc.w * sbAccum[3].z };
    };

    // contact 0, the TRIANGLE path
    float lam0 = 0.0f;
    const V3 sbw0 = sbWeighted(BC1_A);
    const V3 rel0 = { BC0_TRI.x * clothAccum[0].x + BC0_TRI.y * clothAccum[1].x + BC0_TRI.z * clothAccum[2].x - sbw0.x,
                      BC0_TRI.x * clothAccum[0].y + BC0_TRI.y * clothAccum[1].y + BC0_TRI.z * clothAccum[2].y - sbw0.y,
                      BC0_TRI.x * clothAccum[0].z + BC0_TRI.y * clothAccum[1].z + BC0_TRI.z * clothAccum[2].z - sbw0.z };
    const V3 cv0 = solve(rel0, NP_A, DENOM_A, FRIC_A, lam0);

    // contact 1, the VERTEX path
    float lam1 = 0.0f;
    const V3 sbw1 = sbWeighted(BC1_B);
    const V3 rel1 = { clothAccum[3].x - sbw1.x, clothAccum[3].y - sbw1.y, clothAccum[3].z - sbw1.z };
    const V3 cv1 = solve(rel1, NP_B, DENOM_B, FRIC_B, lam1);
    // contact 3 repeats contact 1 without friction and adds into the SAME vertices.
    float lam3 = 0.0f;
    const V3 cv3 = solve(rel1, NP_B, DENOM_B, 0.0f, lam3);
    // contact 4: separated, so the clamp holds it at zero.
    float lam4 = 0.0f;
    const V3 cv4 = solve(rel1, NP_C, DENOM_B, 0.0f, lam4);

    std::printf("  cloth out: v0=(%.6f %.6f %.6f w=%.1f) v1=(%.6f %.6f %.6f w=%.1f)\n"
                "             v2=(%.6f %.6f %.6f w=%.1f) v3=(%.6f %.6f %.6f w=%.1f)\n",
                gClothOut[0].x, gClothOut[0].y, gClothOut[0].z, gClothOut[0].w,
                gClothOut[1].x, gClothOut[1].y, gClothOut[1].z, gClothOut[1].w,
                gClothOut[2].x, gClothOut[2].y, gClothOut[2].z, gClothOut[2].w,
                gClothOut[3].x, gClothOut[3].y, gClothOut[3].z, gClothOut[3].w);
    std::printf("  sb out:    v0=(%.6f %.6f %.6f w=%.1f) v3=(%.6f %.6f %.6f w=%.1f)\n",
                gSbOut[0].x, gSbOut[0].y, gSbOut[0].z, gSbOut[0].w,
                gSbOut[3].x, gSbOut[3].y, gSbOut[3].z, gSbOut[3].w);
    std::printf("  lambdas:   %.6f %.6f %.6f %.6f %.6f\n",
                gLambda[0], gLambda[1], gLambda[2], gLambda[3], gLambda[4]);

    auto near3 = [](F4 got, float wx, float wy, float wz, float ww) {
        return std::fabs(got.x - wx) < 1e-5f && std::fabs(got.y - wy) < 1e-5f &&
               std::fabs(got.z - wz) < 1e-5f && std::fabs(got.w - ww) < 1e-5f;
    };

    // The atomics start from the sentinel, so every expectation is SENT + the contributions.
    PXG_CHECK_MSG(near3(gClothOut[0], SENT + clothAccum[0].w * BC0_TRI.x * cv0.x,
                                      SENT + clothAccum[0].w * BC0_TRI.x * cv0.y,
                                      SENT + clothAccum[0].w * BC0_TRI.x * cv0.z, SENT + 1.0f),
                  "cloth vertex 0 takes w * bc.x * commonVec and ONE contribution count");
    PXG_CHECK_MSG(near3(gClothOut[1], SENT + clothAccum[1].w * BC0_TRI.y * cv0.x,
                                      SENT + clothAccum[1].w * BC0_TRI.y * cv0.y,
                                      SENT + clothAccum[1].w * BC0_TRI.y * cv0.z, SENT + 1.0f),
                  "cloth vertex 1 likewise, with its own inverse mass and barycentric");
    PXG_CHECK_MSG(gClothOut[2].x == SENT && gClothOut[2].w == SENT,
                  "cloth vertex 2 is PINNED (inverse mass 0) and its barycentric is 0, so its "
                  "scatter must be suppressed -- got (%.6f, w %.6f)", gClothOut[2].x, gClothOut[2].w);
    // Vertex 3 receives from contacts 1 AND 3 (contact 2 is degenerate and skipped), so its
    // count is exactly TWO -- which is also what says contact 2 was skipped without taking the
    // rest of the loop with it.
    PXG_CHECK_MSG(near3(gClothOut[3], SENT + clothAccum[3].w * (cv1.x + cv3.x + cv4.x),
                                      SENT + clothAccum[3].w * (cv1.y + cv3.y + cv4.y),
                                      SENT + clothAccum[3].w * (cv1.z + cv3.z + cv4.z), SENT + 3.0f),
                  "cloth vertex 3 takes w * commonVec with NO barycentric factor, summed over "
                  "contacts 1, 3 and 4; its count must be 3, not 4 -- contact 2's velMultiplier "
                  "is zero and it is skipped, but skipping it must not abandon the rest");

    // The softbody side, opposite sign.  Vertex 0 gets all three live contacts.
    PXG_CHECK_MSG(near3(gSbOut[0],
                        SENT - sbAccum[0].w * (BC1_A.x * cv0.x + BC1_B.x * (cv1.x + cv3.x + cv4.x)),
                        SENT - sbAccum[0].w * (BC1_A.x * cv0.y + BC1_B.x * (cv1.y + cv3.y + cv4.y)),
                        SENT - sbAccum[0].w * (BC1_A.x * cv0.z + BC1_B.x * (cv1.z + cv3.z + cv4.z)),
                        SENT + 4.0f),
                  "the softbody correction has the OPPOSITE SIGN to the cloth's, and vertex 0 "
                  "receives all four live contacts");
    // Tet vertex 3's barycentric in contact 0 is 1e-7, BELOW the 1e-6 guard, so only contacts
    // 1 and 3 reach it.
    PXG_CHECK_MSG(std::fabs(gSbOut[3].w - (SENT + 3.0f)) < 1e-5f,
                  "contact 0's fourth barycentric is 1e-7, below the 1e-6 guard, so tet vertex "
                  "3 must receive THREE contributions, not four (got count %.6f)",
                  gSbOut[3].w - SENT);

    PXG_CHECK_MSG(std::fabs(gLambda[0] - lam0) < 1e-6f && std::fabs(gLambda[1] - lam1) < 1e-6f &&
                  std::fabs(gLambda[3] - lam3) < 1e-6f,
                  "the accumulated lambdas are written back (%.6f %.6f %.6f vs %.6f %.6f %.6f)",
                  gLambda[0], gLambda[1], gLambda[3], lam0, lam1, lam3);
    PXG_CHECK_MSG(gLambda[2] == 0.0f,
                  "contact 2 is degenerate and must not touch its lambda (got %.6f)", gLambda[2]);
    PXG_CHECK_MSG(gLambda[4] == 0.0f && std::fabs(cv4.x) < 1e-9f,
                  "contact 4 is SEPARATED, so -CN * denomInv is negative and the clamp against "
                  "-lambdaN holds the impulse at exactly zero.  Without the clamp the contact "
                  "pulls the bodies together and the lambda goes negative (got %.6f)",
                  gLambda[4]);

    // The two sides must actually oppose, or the sign is untested.
    PXG_CHECK_MSG((cv0.y > 0.0f) == (gClothOut[0].y - SENT > 0.0f) &&
                  (gClothOut[0].y - SENT) * (gSbOut[0].y - SENT) < 0.0f,
                  "the cloth and softbody corrections must point AGAINST each other; both "
                  "positive or both negative means one of the two signs is wrong");
    // And friction must have done something, or the tangential branch is dead code here.
    PXG_CHECK_MSG(f2u(cv1.y) != f2u(cv3.y) || f2u(cv1.z) != f2u(cv3.z),
                  "contacts 1 and 3 differ ONLY in their friction coefficient, so their "
                  "corrections must differ -- otherwise the tangential branch never ran");

    // ---- continue vs return: NOT TESTABLE HERE, and the arithmetic says why ---------------
    // A degenerate constraint CONTINUES rather than returning, so it skips only its own
    // contact.  That differs from a return only when a thread owns MORE THAN ONE contact, i.e.
    // when nbIterationsPerBlock > 1.  With the engine's fixed 1024 x 256 grid -- which this
    // port hard-codes, as every other kernel in the chain does -- that needs
    // nbBlocksRequired > 1024, i.e. more than 1024 * 256 = 262144 contacts.
    //
    // Dispatching fewer blocks does NOT help: SC_GRID is a constant, so a one-block dispatch
    // still computes nbIterationsPerBlock = ceil(3 / 1024) = 1 and simply leaves contacts
    // 256 and up unvisited (measured: indices 0..255 solved, 256 onwards untouched).
    //
    // So `continue` is the faithful transcription and is kept, but a mutant swapping it for
    // `return` survives every fixture that fits in memory.  Stated rather than papered over.

    for (PxgBufferHandle h : { bClothAccum.h, bClothTris.h, bClothOut.h, bCloth.h, bSbAccum.h,
                               bSimTets.h, bSbOut.h, bDesc.h, bCons.h, bInfo.h, bNum.h, bLambda.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.9 -- sb_sbMidphaseGeneratePairsLaunch, the WHOLE softbody-vs-softbody narrowphase.
//
// WHAT THIS FIXTURE IS FOR.  The traverser ARITHMETIC (closestPtTetrahedron,
// isPointOutsideOfPlane4, the inside/outside decision) is already covered bit-exactly by the
// npSbSelf / ctSbSelfQuery fixtures and again by softBodySimSelfCollisionMidphase.  This one
// covers what the SS lift ADDS on top of the self-collision kernel:
//
//   * blockIdx.z -- the BIDIRECTIONAL pass.  Upstream launches gridDim.z = 2 and flips which
//     shape supplies the query vertices; dropping it halves the contact set.  The fixture
//     puts each direction's geometry in its OWN region of space so a missing flip shows up as
//     a missing contact, not as a changed number;
//   * TWO descriptors rather than one -- query verts come from softbody0, the tree and its
//     surface-tet hints from softbody1, the filter list from softbody0;
//   * the contact distance is the SUM of the two shapes' entries, not one doubled.  Two
//     query vertices straddle that sum from both sides, so every single-value or
//     doubled-value reading of it is killed;
//   * the cmInputs / gpuShapes indirection and the both-must-be-tetmesh early-out;
//   * the 112-byte blob with PxgSoftBodyContactWriter inlined at @56.
//
// Geometry is compared to 1e-6; counts, pair ids and drop/keep decisions are compared EXACTLY.
// ===========================================================================
struct SbSbPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, softbodies;
    uint64_t outPoint, outNormalPen, outBary0, outBary1, outContactInfo, totalContactCount;
    uint32_t maxContacts, pad108;
};
static_assert(sizeof(SbSbPush) == 112, "sb-sb midphase blob must be 112 bytes");
static_assert(offsetof(SbSbPush, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(SbSbPush, softbodies) == 48, "softbodies @48");
static_assert(offsetof(SbSbPush, outPoint) == 56, "the inlined writer starts @56");
static_assert(offsetof(SbSbPush, maxContacts) == 104, "maxContacts @104");

PXG_TEST(softBodySimSoftBodySoftBodyMidphase)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t A_ID = 3u;    // the body that queries on flip = 0
    const uint32_t B_ID = 5u;    // the body that queries on flip = 1
    const uint32_t ANY_TET = 0x000fffffu;   // PX_MAX_NB_DEFORMABLE_VOLUME_TET
    auto enc = [](uint32_t id, uint32_t elem) { return (id << 20) | elem; };

    // ---- geometry ------------------------------------------------------------
    // REGION X (x ~ 0..71) holds B's tetrahedra and A's query vertices  -> the flip = 0 pass.
    // REGION Y (x ~ 100)   holds A's tetrahedron and B's query vertex   -> the flip = 1 pass.
    // Keeping the two directions in disjoint regions is what makes a missing z dimension show
    // up as a MISSING CONTACT rather than as a changed count somewhere ambiguous.
    //
    // A unit tet at origin o has corners o, o+x, o+y, o+z; a query at o + (0.12, 0.15, 0.05)
    // is inside it and closest to the z-face, so closestP = o + (0.12, 0.15, 0) and pen = -0.05.
    std::vector<F4> aVerts, bVerts;
    std::vector<TetIdx4> aTets, bTets;
    std::vector<uint8_t> aVertHint, bVertHint, aTetHint, bTetHint;

    auto addTet = [](std::vector<F4>& v, std::vector<TetIdx4>& t, std::vector<uint8_t>& vh,
                     std::vector<uint8_t>& th, float ox, float oy, float oz, bool degenerate,
                     uint8_t hint) {
        const uint32_t b = uint32_t(v.size());
        v.push_back(F4{ ox,        oy,        oz,        1.0f });
        v.push_back(F4{ ox + 1.0f, oy,        oz,        1.0f });
        v.push_back(F4{ ox,        oy + 1.0f, oz,        1.0f });
        v.push_back(degenerate ? F4{ ox, oy, oz, 1.0f } : F4{ ox, oy, oz + 1.0f, 1.0f });
        for (int i = 0; i < 4; ++i) vh.push_back(0u);   // corners are never queries here
        t.push_back(TetIdx4{ b, b + 1u, b + 2u, b + 3u });
        th.push_back(hint);
        return uint32_t(t.size() - 1u);
    };
    auto addQuery = [](std::vector<F4>& v, std::vector<uint8_t>& vh,
                       float x, float y, float z, uint8_t hint) {
        const uint32_t i = uint32_t(v.size());
        v.push_back(F4{ x, y, z, 1.0f });
        vh.push_back(hint);
        return i;
    };

    // B's tets, region X.  ORDER MATTERS: the shallow one is primitive 0 and therefore LANE 0
    // of the shared leaf, the deep one lane 1.  Put the deep one first and "lane 0 always
    // wins" -- i.e. no cross-lane argmin at all -- produces the right answer by accident.
    const uint32_t B_SHALLOW = addTet(bVerts, bTets, bVertHint, bTetHint,  0.0f, 0.0f, 0.02f, false, 0xfu);
    const uint32_t B_DEEP    = addTet(bVerts, bTets, bVertHint, bTetHint,  0.0f, 0.0f, 0.00f, false, 0xfu);
    const uint32_t B_DEGEN   = addTet(bVerts, bTets, bVertHint, bTetHint,  2.0f, 0.0f, 0.00f, true,  0xfu);
    const uint32_t B_FILTPR  = addTet(bVerts, bTets, bVertHint, bTetHint, 30.0f, 0.0f, 0.00f, false, 0xfu);
    const uint32_t B_FILTWC  = addTet(bVerts, bTets, bVertHint, bTetHint, 40.0f, 0.0f, 0.00f, false, 0xfu);
    const uint32_t B_NOSURF  = addTet(bVerts, bTets, bVertHint, bTetHint, 60.0f, 0.0f, 0.00f, false, 0x0u);
    const uint32_t B_GOOD    = addTet(bVerts, bTets, bVertHint, bTetHint, 70.0f, 0.0f, 0.00f, false, 0xfu);
    const uint32_t NB_B_TETS = uint32_t(bTets.size());
    (void)B_SHALLOW; (void)B_DEGEN; (void)B_FILTWC; (void)B_NOSURF;

    // A's single tet, region Y -- the target of the flip = 1 pass.
    const uint32_t A_GOOD = addTet(aVerts, aTets, aVertHint, aTetHint, 100.0f, 0.0f, 0.0f, false, 0xfu);

    // A's query vertices, region X.
    const uint32_t a1 = addQuery(aVerts, aVertHint,  0.12f, 0.15f, 0.05f, 1u);  // KEPT (B_DEEP)
    const uint32_t a2 = addQuery(aVerts, aVertHint,  2.12f, 0.15f, 0.05f, 1u);  // drop: degenerate
    const uint32_t a3 = addQuery(aVerts, aVertHint,  0.15f, 0.12f, 0.06f, 0u);  // drop: vert hint 0
    const uint32_t a4 = addQuery(aVerts, aVertHint, 30.12f, 0.15f, 0.05f, 1u);  // drop: per-pair filter
    const uint32_t a5 = addQuery(aVerts, aVertHint, 40.12f, 0.15f, 0.05f, 1u);  // drop: wildcard filter
    const uint32_t a6 = addQuery(aVerts, aVertHint, 60.12f, 0.15f, 0.05f, 1u);  // drop: tet hint 0
    const uint32_t a7 = addQuery(aVerts, aVertHint, 70.12f, 0.15f, 0.05f, 1u);  // KEPT (inside B_GOOD)
    // THE CONTACT-DISTANCE PAIR.  contactDistance[cacheRef0] = 0.03 and [cacheRef1] = 0.02, so
    // the sum is 0.05.  a8 sits 0.040 outside B_GOOD's x-face and a9 sits 0.055 outside it:
    //   sum  0.05  -> a8 KEPT, a9 dropped          <- the only reading that gives 1 of 2
    //   ref0 alone 0.03 -> both dropped
    //   ref1 alone 0.02 -> both dropped
    //   2 * ref0   0.06 -> BOTH kept
    //   2 * ref1   0.04 -> both dropped (the test is strict `<`)
    const uint32_t a8 = addQuery(aVerts, aVertHint, 69.960f, 0.30f, 0.30f, 1u); // KEPT (outside)
    const uint32_t a9 = addQuery(aVerts, aVertHint, 69.945f, 0.30f, 0.30f, 1u); // drop: too far
    (void)a2; (void)a3; (void)a6; (void)a9;

    // B's query vertex, region Y -- the ONLY contact the flip = 1 pass can produce.
    const uint32_t b1 = addQuery(bVerts, bVertHint, 100.12f, 0.15f, 0.05f, 1u); // KEPT (A_GOOD)

    // mSurfaceVertsHint / mTetMeshSurfaceHint are PxU8 arrays; pack them into words so the
    // shader's byte extraction is exercised at all four offsets.
    auto packBytes = [](const std::vector<uint8_t>& b) {
        std::vector<uint32_t> w((b.size() + 3u) / 4u + 1u, 0u);
        for (size_t i = 0; i < b.size(); ++i) w[i / 4u] |= uint32_t(b[i]) << (8u * (i % 4u));
        return w;
    };
    const std::vector<uint32_t> aVertHintWords = packBytes(aVertHint);
    const std::vector<uint32_t> bVertHintWords = packBytes(bVertHint);
    const std::vector<uint32_t> aTetHintWords = packBytes(aTetHint);
    const std::vector<uint32_t> bTetHintWords = packBytes(bTetHint);

    // vertToSurfaceTetRemap: DIFFERENT bases per body, and neither is the identity, so pairInd0
    // cannot pass by accidentally matching a vertex index, a tet index or the other body's remap.
    std::vector<uint32_t> aRemap(aVerts.size()), bRemap(bVerts.size());
    for (size_t i = 0; i < aRemap.size(); ++i) aRemap[i] = 100u + uint32_t(i);
    for (size_t i = 0; i < bRemap.size(); ++i) bRemap[i] = 700u + uint32_t(i);

    // ---- the two BV32 trees --------------------------------------------------
    // B's tree: child 0 is a leaf holding BOTH B_SHALLOW and B_DEEP, so those land in DIFFERENT
    // lanes of the same warp and the deeper one can only win through the cross-lane argmin.
    // The remaining tets get one leaf each.  Child bounds are deliberately GENEROUS so every
    // child is visited and the node-level cull can never stand in for a per-primitive gate.
    auto buildTree = [](uint32_t nbVerts, uint32_t nbTets, bool pairFirstLeaf) {
        const uint32_t nbChildren = pairFirstLeaf ? (1u + (nbTets - 2u)) : nbTets;
        const uint32_t nbNodes = 1u, maxDepth = 1u;
        TreeBlob t;
        t.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes + 4u, 0u);
        t.w[0] = nbVerts; t.w[1] = nbTets; t.w[2] = maxDepth; t.w[3] = nbNodes;
        const uint32_t nw = t.nodeWord(0);
        t.w[nw + W_NBNODES] = nbChildren;
        for (uint32_t k = 0; k < nbChildren; ++k)
        {
            for (uint32_t c = 0; c < 3u; ++c)
            {
                t.w[nw + W_MMIN + k * 4 + c] = f2u(-1000.0f);
                t.w[nw + W_MMAX + k * 4 + c] = f2u(1000.0f);
            }
            if (pairFirstLeaf)
                t.w[nw + W_MDATA + k] = (k == 0u) ? leafWord(0u, 2u) : leafWord(k + 1u, 1u);
            else
                t.w[nw + W_MDATA + k] = leafWord(k, 1u);
        }
        return t;
    };
    TreeBlob bTree = buildTree(uint32_t(bVerts.size()), NB_B_TETS, /*pairFirstLeaf*/ true);
    TreeBlob aTree = buildTree(uint32_t(aVerts.size()), uint32_t(aTets.size()), false);

    // ---- filter pairs, sorted by index0 then index1 --------------------------
    // They hang off body A (softbody0 on the flip = 0 pass).  B carries NONE, which also
    // exercises the null-mFilteringPairs path the self-collision tranche had to harden.
    struct PairNRHost { uint64_t index0; uint32_t index1, index2; };
    std::vector<PairNRHost> aFilters = {
        { enc(A_ID, a4), enc(B_ID, B_FILTPR), 0u },   // a4 against ONE tet of B
        { enc(A_ID, a5), enc(B_ID, ANY_TET),  0u },   // a5 against the whole of B
    };
    PXG_CHECK(aFilters[0].index0 < aFilters[1].index0);   // the binary search needs order

    DevBuf dbATree = makeDev(dev, aTree.w);
    DevBuf dbBTree = makeDev(dev, bTree.w);
    DevBuf dbAVerts = makeDev(dev, aVerts);
    DevBuf dbBVerts = makeDev(dev, bVerts);
    DevBuf dbATets = makeDev(dev, aTets);
    DevBuf dbBTets = makeDev(dev, bTets);
    DevBuf dbAVertHint = makeDev(dev, aVertHintWords);
    DevBuf dbBVertHint = makeDev(dev, bVertHintWords);
    DevBuf dbATetHint = makeDev(dev, aTetHintWords);
    DevBuf dbBTetHint = makeDev(dev, bTetHintWords);
    DevBuf dbARemap = makeDev(dev, aRemap);
    DevBuf dbBRemap = makeDev(dev, bRemap);
    DevBuf dbAFilters = makeDev(dev, aFilters);
    if (!dbATree.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sbA = makeZeroDesc();
    sbA.mTetMeshData = reinterpret_cast<void*>(size_t(dbATree.addr));
    sbA.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(dbATetHint.addr));
    sbA.mTetIndices = reinterpret_cast<decltype(sbA.mTetIndices)>(size_t(dbATets.addr));
    sbA.mPosition_InvMass = reinterpret_cast<float4*>(size_t(dbAVerts.addr));
    sbA.mSurfaceVertsHint = reinterpret_cast<PxU8*>(size_t(dbAVertHint.addr));
    sbA.mSurfaceVertToTetRemap = reinterpret_cast<PxU32*>(size_t(dbARemap.addr));
    sbA.mFilteringPairs = reinterpret_cast<decltype(sbA.mFilteringPairs)>(size_t(dbAFilters.addr));
    sbA.mNumFilterPairs = uint32_t(aFilters.size());
    sbA.mNumTets = uint32_t(aTets.size());
    sbA.mNumVerts = uint32_t(aVerts.size());

    PxgSoftBody sbB = makeZeroDesc();
    sbB.mTetMeshData = reinterpret_cast<void*>(size_t(dbBTree.addr));
    sbB.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(dbBTetHint.addr));
    sbB.mTetIndices = reinterpret_cast<decltype(sbB.mTetIndices)>(size_t(dbBTets.addr));
    sbB.mPosition_InvMass = reinterpret_cast<float4*>(size_t(dbBVerts.addr));
    sbB.mSurfaceVertsHint = reinterpret_cast<PxU8*>(size_t(dbBVertHint.addr));
    sbB.mSurfaceVertToTetRemap = reinterpret_cast<PxU32*>(size_t(dbBRemap.addr));
    sbB.mFilteringPairs = nullptr;      // the null path
    sbB.mNumFilterPairs = 0u;
    sbB.mNumTets = NB_B_TETS;
    sbB.mNumVerts = uint32_t(bVerts.size());

    std::vector<PxgSoftBody> descArr(8u, makeZeroDesc());
    descArr[A_ID] = sbA;
    descArr[B_ID] = sbB;
    DevBuf dbDesc = makeDev(dev, descArr);

    // gpuShapes: the two live tetmesh shapes sit at NON-adjacent, NON-zero slots, and the
    // decoy pair's shapes are boxes.  A port that ignored shapeRef and used cmIdx, or that
    // dropped the both-must-be-tetmesh early-out, lands on a zeroed descriptor and faults.
    std::vector<PxgShapeHost> shapes(7);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    for (auto& s : shapes) { s.srw = 1.0f; s.particleOrSoftbodyId = 0xFFFFFFFFu; s.type = 3u /*eBOX*/; }
    shapes[2].type = GEOM_TETRAHEDRONMESH_H; shapes[2].particleOrSoftbodyId = A_ID;
    shapes[5].type = GEOM_TETRAHEDRONMESH_H; shapes[5].particleOrSoftbodyId = B_ID;
    DevBuf dbShapes = makeDev(dev, shapes);

    // cmInputs[0] is a DECOY box-vs-box pair; cmInputs[1] is the real one.  Dispatching
    // gridY = 2 therefore also proves cmIdx comes from gid.y and that the type gate fires.
    std::vector<CmInputHost> cms = {
        { 0u, 1u, 0u, 0u },              // decoy: two boxes
        { 2u, 5u, /*cacheRef0*/ 1u, /*cacheRef1*/ 3u },
    };
    DevBuf dbCms = makeDev(dev, cms);

    // contactDistance is indexed by the transform-cache refs; the slots between them are
    // poison so an off-by-one index is not survivable.
    std::vector<float> contactDist = { 999.0f, 0.03f, 999.0f, 0.02f, 999.0f };
    DevBuf dbCd = makeDev(dev, contactDist);

    const uint32_t MAXC = 16u;
    const uint32_t POISON = 0xCDCDCDCDu;
    std::vector<F4> zeroF4(MAXC, F4{ 0, 0, 0, 0 });
    std::vector<uint32_t> poison(MAXC * 4u, POISON);
    DevBuf dbPoint = makeDev(dev, zeroF4);
    DevBuf dbNormalPen = makeDev(dev, zeroF4);
    DevBuf dbBary0 = makeDev(dev, zeroF4);
    DevBuf dbBary1 = makeDev(dev, zeroF4);
    DevBuf dbInfo = makeDev(dev, poison);
    std::vector<uint32_t> zeroCount = { 0u };
    DevBuf dbCount = makeDev(dev, zeroCount);

    SbSbPush push = {};
    push.toleranceLength = 1.0f;      // read by nothing -- it is a blob placeholder
    push.cmInputs = dbCms.addr;
    push.transformCache = dbCms.addr; // ditto: never dereferenced, but must occupy @16
    push.bounds = dbCms.addr;         // ditto @24
    push.contactDistance = dbCd.addr;
    push.gpuShapes = dbShapes.addr;
    push.softbodies = dbDesc.addr;
    push.outPoint = dbPoint.addr;
    push.outNormalPen = dbNormalPen.addr;
    push.outBary0 = dbBary0.addr;
    push.outBary1 = dbBary1.addr;
    push.outContactInfo = dbInfo.addr;
    push.totalContactCount = dbCount.addr;
    push.maxContacts = MAXC;

    Kernel k("softBodySoftBodyMidphaseBda.spv", "sb_sbMidphaseGeneratePairsLaunch",
             {}, sizeof(SbSbPush));
    // THE ENGINE'S GRID, reproduced exactly: x = SB_SBMIDPHASE = 1024 (fixed, independent of
    // the vertex count), y = numTests, z = 2 (the bidirectional flip).  With the CUDA stride
    // (blockDim.y * gridDim.x = 2048) every vertex is visited EXACTLY ONCE per (y, z).
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ uint32_t(cms.size()), /*gridZ*/ 2u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, dbCount.h, 1)[0];
    const auto gPoint = downloadBuffer<F4>(dev, dbPoint.h, MAXC);
    const auto gNormalPen = downloadBuffer<F4>(dev, dbNormalPen.h, MAXC);
    const auto gBary0 = downloadBuffer<F4>(dev, dbBary0.h, MAXC);
    const auto gBary1 = downloadBuffer<F4>(dev, dbBary1.h, MAXC);
    const auto gInfo = downloadBuffer<uint32_t>(dev, dbInfo.h, MAXC * 4u);

    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
        std::printf("  contact %u: p=(%.5f %.5f %.5f) n=(%.5f %.5f %.5f) pen=%.6f "
                    "bary0=(%.5f %.5f %.5f) ind0=(body %u elem %u) ind1=(body %u elem %u)\n",
                    i, gPoint[i].x, gPoint[i].y, gPoint[i].z, gNormalPen[i].x, gNormalPen[i].y,
                    gNormalPen[i].z, gNormalPen[i].w, gBary0[i].x, gBary0[i].y, gBary0[i].z,
                    gInfo[i * 4u + 0u] >> 20, gInfo[i * 4u + 0u] & 0xfffffu,
                    gInfo[i * 4u + 2u] >> 20, gInfo[i * 4u + 2u] & 0xfffffu);

    PXG_CHECK_MSG(gCount == 4u,
                  "exactly four queries survive every gate -- a1, a7 and a8 on the flip=0 pass "
                  "and b1 on the flip=1 pass -- got %u.  3 usually means the z dimension was "
                  "dropped (b1 is the only contact that pass can make); more than 4 means a "
                  "gate went missing or a vertex was visited more than once", gCount);

    auto near6 = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };
    bool sawA1 = false, sawA7 = false, sawA8 = false, sawB1 = false;
    for (uint32_t i = 0; i < gCount; ++i)
    {
        const uint32_t ind0 = gInfo[i * 4u + 0u];
        const uint32_t ind0hi = gInfo[i * 4u + 1u];
        const uint32_t ind1 = gInfo[i * 4u + 2u];
        PXG_CHECK_MSG(ind0hi == 0u,
                      "pairInd0 is a PxU64 field taking a PxU32 value: the high word must be "
                      "ZEROED, not left poisoned (contact %u has 0x%08x)", i, ind0hi);

        if (ind0 == enc(A_ID, aRemap[a1]))
        {
            sawA1 = true;
            PXG_CHECK_MSG(ind1 == enc(B_ID, B_DEEP),
                          "a1 penetrates B_DEEP by 0.05 and B_SHALLOW by 0.03; the cross-lane "
                          "argmin must pick the DEEPER one (got body %u tet %u, want body %u "
                          "tet %u)", ind1 >> 20, ind1 & 0xfffffu, B_ID, B_DEEP);
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
            PXG_CHECK(near6(gNormalPen[i].x, 0.0f) && near6(gNormalPen[i].y, 0.0f) &&
                      near6(gNormalPen[i].z, -1.0f));
            PXG_CHECK(near6(gPoint[i].x, 0.12f) && near6(gPoint[i].y, 0.15f) &&
                      near6(gPoint[i].z, 0.0f));
            // barycentric0 carries the QUERY POINT and barycentric1 the closest point -- that
            // is what the CUDA writer stores, and the sim-mesh remap depends on it.
            PXG_CHECK(near6(gBary0[i].x, 0.12f) && near6(gBary0[i].y, 0.15f) &&
                      near6(gBary0[i].z, 0.05f));
            PXG_CHECK(near6(gBary1[i].x, gPoint[i].x) && near6(gBary1[i].z, gPoint[i].z));
        }
        else if (ind0 == enc(A_ID, aRemap[a7]))
        {
            sawA7 = true;
            PXG_CHECK(ind1 == enc(B_ID, B_GOOD));
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
        }
        else if (ind0 == enc(A_ID, aRemap[a8]))
        {
            sawA8 = true;
            PXG_CHECK(ind1 == enc(B_ID, B_GOOD));
            // The OUTSIDE branch: pen is the POSITIVE separation and the normal is the
            // normalised point-to-surface direction, here parallel to the x-face normal.
            PXG_CHECK_MSG(std::fabs(gNormalPen[i].w - 0.04f) <= 2e-6f,
                          "a8 sits 0.040 outside B_GOOD's x-face; got pen %.7f", gNormalPen[i].w);
            PXG_CHECK(near6(gNormalPen[i].y, 0.0f) && near6(gNormalPen[i].z, 0.0f));
            PXG_CHECK_MSG(gNormalPen[i].x < -0.99f,
                          "the outside normal must point from the tet towards the query vertex "
                          "(-x here); got %.5f", gNormalPen[i].x);
        }
        else if (ind0 == enc(B_ID, bRemap[b1]))
        {
            // THE FLIP.  Note the pair ids are the other way round: body B supplies the vertex
            // and body A the tetrahedron, which can only happen on blockIdx.z == 1.
            sawB1 = true;
            PXG_CHECK_MSG(ind1 == enc(A_ID, A_GOOD),
                          "on the flipped pass the TET must come from body A (got body %u)",
                          ind1 >> 20);
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
            PXG_CHECK(near6(gPoint[i].x, 100.12f) && near6(gPoint[i].z, 0.0f));
        }
        else
            PXG_CHECK_MSG(false, "contact %u came from an unexpected query: ind0 = body %u "
                                 "elem %u", i, ind0 >> 20, ind0 & 0xfffffu);
    }

    PXG_CHECK_MSG(sawA1 && sawA7 && sawA8,
                  "the flip=0 pass must emit a1, a7 and a8 (%d %d %d)",
                  int(sawA1), int(sawA7), int(sawA8));
    PXG_CHECK_MSG(sawB1,
                  "the flip=1 pass must emit b1 -- if this is the only failure, blockIdx.z is "
                  "being ignored and half the contact set is missing");

    for (PxgBufferHandle h : { dbATree.h, dbBTree.h, dbAVerts.h, dbBVerts.h, dbATets.h,
                               dbBTets.h, dbAVertHint.h, dbBVertHint.h, dbATetHint.h,
                               dbBTetHint.h, dbARemap.h, dbBRemap.h, dbAFilters.h, dbDesc.h,
                               dbShapes.h, dbCms.h, dbCd.h, dbPoint.h, dbNormalPen.h,
                               dbBary0.h, dbBary1.h, dbInfo.h, dbCount.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.10 step 1 -- sb_meshMidphaseGeneratePairsLaunch, the softbody-vs-TRIANGLE-MESH midphase.
//
// WHAT THIS FIXTURE IS FOR.  The traverser is the one softBodySimClothMidphase already covers
// (same struct, same SAT, same top-4 extraction; the .cu's <true>/<false> template argument
// only switches a cull the caller already performs).  This fixture aims at what the TRIMESH
// lift adds and nothing else:
//
//   * the COOKED BLOB WALK -- header, BV32 node array, vertex pool, index pool -- including
//     that the node array is sized by header.w and NOT header.z, which are different counts;
//   * the two-step transform into the mesh's VERTEX space: world -> mesh local via
//     transformInv (a real 180-degree rotation here, not identity, so dropping the rotation is
//     visible) and local -> vertex via shape2Vertex, which DIVIDES by a non-uniform scale;
//   * the contact distance divided by the smallest |scale| component.  The scale is
//     (2, 1, 0.5), so the correct band is 0.20 / 0.5 = 0.40 and the tet ladder straddles both
//     0.40 and the 0.20 an undivided reading would give;
//   * the 80-byte blob.
//
// The mesh is laid out in VERTEX space exactly as the cloth fixture lays out its sheet, and the
// tetrahedra are placed in WORLD space through the inverse of the full transform, so the gap
// ladder below is a ladder in vertex space and can be read off directly.
// ===========================================================================
namespace p810mesh
{
// { PxReal toleranceLength, cmInputs*, transformCache*, bounds*, contactDistance*, gpuShapes*,
//   softbodies*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum* }
struct SbMeshMidPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, softbodies;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(SbMeshMidPush) == 80, "sb mesh midphase blob must be 80 bytes");
static_assert(offsetof(SbMeshMidPush, softbodies) == 48, "softbodies @48");
static_assert(offsetof(SbMeshMidPush, stackSizeBytes) == 56, "stackSizeBytes @56");
static_assert(offsetof(SbMeshMidPush, stackPtr) == 64, "stackPtr @64");
} // namespace p810mesh

PXG_TEST(softBodySimMeshMidphase)
{
    using namespace p80refit;
    using namespace p88cloth;      // Bv32NodeHost
    using namespace p810mesh;
    PxgSlangComputeDevice* dev = testDevice();

    // The mesh's placement.  scaleRot stays identity so shape2Vertex is a plain componentwise
    // divide and the expected geometry can be written down; the SCALE is non-uniform and the
    // rigid rotation is a real 180 degrees about y, which is exact in floating point:
    //     world = R * (vertex * scale) + T,   R(x, y, z) = (-x, y, -z)
    const float SX = 2.0f, SY = 1.0f, SZ = 0.5f;   // min |scale| = 0.5
    const float TX = 10.0f, TY = 20.0f, TZ = 30.0f;
    auto toWorld = [&](float vx, float vy, float vz) {
        return F4{ -(vx * SX) + TX, (vy * SY) + TY, -(vz * SZ) + TZ, 1.0f };
    };

    // ---- the triangle mesh, in VERTEX space, in the plane y = 0 with normal +y -------------
    const uint32_t NTRI = 6u;
    std::vector<F4> triVerts;
    std::vector<TetIdx4> triIdx;
    // Primitive 0 is a DUMMY parked 1000 units away, so the leaf's primStartIndex is 1 rather
    // than 0.  With a zero start the field is invisible: `data >> 7` and `data >> 8` both yield
    // 0 and a mutant on the shift survives.
    const uint32_t TRI_BASE = 1u;
    {
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ 1000.0f, 0.0f, 0.0f, 0.0f });
        triVerts.push_back(F4{ 1000.0f, 0.0f, 1.0f, 0.0f });
        triVerts.push_back(F4{ 1001.0f, 0.0f, 0.0f, 0.0f });
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    for (uint32_t t = 0; t < NTRI; ++t)
    {
        // Six unit right triangles side by side along x, laid out in REVERSE so the triangle
        // in slot k has primitive index NTRI-1-k.  The tets run forwards, so tri == 5 - tet
        // and the two fields of the emitted uint4 cannot be confused for each other.
        const float ox = float(NTRI - 1u - t) * 2.0f;
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ ox,        0.0f, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox,        0.0f, 1.0f, 0.0f });
        triVerts.push_back(F4{ ox + 1.0f, 0.0f, 0.0f, 0.0f });
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    // A BACK-FACING triangle in tet 0's slot and CLOSER to it (0.05 against a gap of 0.20).
    // Its normal is -y, so the front-face gate must reject it; without the gate tet 0 reports
    // two triangles instead of one.
    {
        const float ox = 0.0f;
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ ox,        0.15f, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox + 1.0f, 0.15f, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox,        0.15f, 1.0f, 0.0f });   // reversed winding -> n = -y
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    // A STACK of five more front-facing triangles under tet 1, so that tet sees SIX qualifying
    // triangles and the MaxContacts = 4 cap actually binds.  FAR TO NEAR on purpose, so the
    // LAST primitive in the leaf is one of the four that survive: ordered the other way the
    // last primitive is the one the cap discards anyway and a primStartIndex mutant -- which
    // shifts exactly which primitive falls off the end -- changes nothing observable.
    const uint32_t STACK_TRI0 = TRI_BASE + NTRI + 1u;
    const uint32_t NB_STACK = 5u;
    for (uint32_t k = 0; k < NB_STACK; ++k)
    {
        const float ox = 2.0f;                       // tet 1's slot
        const float y = -0.02f * float(NB_STACK - k);
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ ox,        y, 0.0f, 0.0f });
        triVerts.push_back(F4{ ox,        y, 1.0f, 0.0f });
        triVerts.push_back(F4{ ox + 1.0f, y, 0.0f, 0.0f });
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    const uint32_t NPRIM = uint32_t(triIdx.size());
    const uint32_t NVERT = uint32_t(triVerts.size());

    // ---- the COOKED BLOB: header | BV32 nodes | vertex pool | index pool -------------------
    // This is the shape readTriangleMesh walks (triangleMesh.cuh:120), packed into one buffer
    // so a wrong stride anywhere lands on the wrong pool rather than on unmapped memory.
    std::vector<uint32_t> blob;
    {
        Bv32NodeHost node;
        std::memset(&node, 0, sizeof(node));
        node.mNbNodes = 1u;
        node.mMin[0][0] = -1.0f; node.mMin[0][1] = -1.0f; node.mMin[0][2] = -1.0f;
        node.mMax[0][0] = 12.0f; node.mMax[0][1] =  1.0f; node.mMax[0][2] =  2.0f;
        // isLeaf | nbPrimitives<<1 | primStartIndex<<7; the leaf covers the real triangles.
        node.mData[0] = 1u | ((NPRIM - TRI_BASE) << 1) | (TRI_BASE << 7);

        blob.resize(4u, 0u);
        blob[0] = NVERT;
        blob[1] = NPRIM;
        // header.z is nbBv32TreeNodes and header.w is nbPackedNodes.  THEY ARE DIFFERENT
        // COUNTS and only .w sizes the node array -- so .z is set to a deliberately wrong 7
        // here.  A port that sizes with .z walks the vertex pointer seven nodes too far and
        // reads garbage triangles.
        blob[2] = 7u;
        blob[3] = 1u;                       // nbPackedNodes
        const size_t nodeWords = sizeof(Bv32NodeHost) / 4u;
        blob.resize(4u + nodeWords, 0u);
        std::memcpy(blob.data() + 4, &node, sizeof(node));
        const size_t vBase = blob.size();
        blob.resize(vBase + NVERT * 4u, 0u);
        std::memcpy(blob.data() + vBase, triVerts.data(), NVERT * sizeof(F4));
        const size_t iBase = blob.size();
        blob.resize(iBase + NPRIM * 4u, 0u);
        std::memcpy(blob.data() + iBase, triIdx.data(), NPRIM * sizeof(TetIdx4));
    }
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the softbody: one tet per triangle slot, at a chosen VERTEX-SPACE gap -------------
    // The correct contact band is (0.05 + 0.15) / min|scale| = 0.20 / 0.5 = 0.40.  The ladder
    // straddles it AND the undivided 0.20:
    //   gap 0.20 -> in either reading            gap 0.28 -> ONLY with the division
    //   gap 0.12 -> in either reading            gap 0.45 -> in NEITHER
    const float gaps[6] = { 0.20f, 0.12f, 0.28f, 0.04f, 0.45f, 0.02f };
    const uint8_t hints[6] = { 1u, 1u, 1u, 1u, 1u, 0u };
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < 6u; ++t)
    {
        const float ox = float(t) * 2.0f, g = gaps[t];
        const uint32_t b = uint32_t(tetVerts.size());
        // The winding matters: isValidTet demands dot(v1-v0, cross(v2-v0, v3-v0)) > 1e-9, so
        // the bottom face is listed counter-clockwise seen from BELOW.  The corners are given
        // in VERTEX space and mapped to world through the inverse transform, so the handedness
        // the kernel sees is the one written here -- the mesh's mirroring x/z rotation does
        // NOT flip it, because the kernel undoes that rotation before the test.  (Reordering
        // the first two corners "to compensate" gives -0.16 and every tet is silently skipped;
        // measured.)
        tetVerts.push_back(toWorld(ox + 0.2f, g,        0.2f));
        tetVerts.push_back(toWorld(ox + 0.2f, g,        0.6f));
        tetVerts.push_back(toWorld(ox + 0.6f, g,        0.2f));
        tetVerts.push_back(toWorld(ox + 0.2f, g + 1.0f, 0.2f));
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
    }
    std::vector<uint32_t> hintWords((6u + 3u) / 4u, 0u);
    for (uint32_t t = 0; t < 6u; ++t)
        reinterpret_cast<uint8_t*>(hintWords.data())[t] = hints[t];
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets), bHints = makeDev(dev, hintWords);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = 6u;
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  The TRIMESH is shapeRef0 and the SOFTBODY shapeRef1, i.e. the SWAPPED
    // order, so LoadShapePairWarp has to find the tet mesh by TYPE rather than by position.
    const uint32_t GEOM_TRIANGLEMESH_H = 8u;
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;    shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = SX; shapes[0].sy = SY; shapes[0].sz = SZ; shapes[0].srw = 1.0f;  // identity scaleRot
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = 0u;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*trimesh cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.  Slot 0 = the softbody (unused),
    // slot 1 = the trimesh, rotated 180 degrees about y and translated.
    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f;                                     // slot 0 identity
    xf[8 + 0] = 0.0f; xf[8 + 1] = 1.0f; xf[8 + 2] = 0.0f; xf[8 + 3] = 0.0f;   // 180 deg about y
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);

    // contactDistance is summed over BOTH cache refs: 0.05 + 0.15 = 0.20, then divided.
    std::vector<float> cd = { 0.05f, 0.15f, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);

    const uint32_t CAP = 64u;
    std::vector<uint32_t> stack(CAP * 4u, 0xFFFFFFFFu);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 0u };
    DevBuf bPairNum = makeDev(dev, pairNum);

    SbMeshMidPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.bounds = 0ull;                 // ABI parity only -- the kernel never reads it
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.stackSizeBytes = CAP * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;

    Kernel k("softBodyMeshMidphaseBda.spv", "sb_meshMidphaseGeneratePairsLaunch",
             {}, sizeof(SbMeshMidPush));
    // The engine's grid (PxgNarrowphaseCore.cpp:5055): x = SB_SBMIDPHASE = 1024, y = numTests.
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 1u, 1u);
    dev->flush();

    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);

    std::vector<std::pair<uint32_t, uint32_t>> emitted;   // (tet, tri)
    for (uint32_t i = 0; i < gNum && i < CAP; ++i)
    {
        PXG_CHECK_MSG(gStack[i * 4u + 0u] == 0u,
                      "pair %u carries cmIdx %u; blockIdx.y is the ONLY contact-manager index "
                      "here", i, gStack[i * 4u + 0u]);
        emitted.push_back({ gStack[i * 4u + 1u], gStack[i * 4u + 2u] });
        std::printf("  pair %u: tet %u  tri %u\n", i, gStack[i * 4u + 1u], gStack[i * 4u + 2u]);
    }
    std::sort(emitted.begin(), emitted.end());

    // tet 0, 2, 3 keep ONE triangle each (their own slot); tet 1 keeps FOUR (the cap);
    // tet 4 is beyond the band and tet 5 is interior.
    PXG_CHECK_MSG(gNum == 7u,
                  "expected 7 pairs -- one each for tets 0, 2 and 3, four for tet 1 (the "
                  "MaxContacts cap), none for tet 4 (beyond 0.40) and none for tet 5 "
                  "(interior) -- got %u.  0 usually means the blob walk or the vertex-space "
                  "transform is wrong; 6 means the scale division is missing (tet 2 at 0.28 "
                  "falls outside an undivided 0.20 band)", gNum);

    auto has = [&](uint32_t tet, uint32_t tri) {
        return std::find(emitted.begin(), emitted.end(), std::make_pair(tet, tri)) != emitted.end();
    };
    auto countFor = [&](uint32_t tet) {
        uint32_t n = 0; for (auto& e : emitted) if (e.first == tet) ++n; return n;
    };

    // tri == 5 - tet for the slot triangles (the mesh is laid out in reverse).
    PXG_CHECK_MSG(has(0u, TRI_BASE + (NTRI - 1u - 0u)),
                  "tet 0 must match the triangle in ITS OWN slot");
    PXG_CHECK_MSG(has(2u, TRI_BASE + (NTRI - 1u - 2u)),
                  "tet 2 sits 0.28 above its triangle: inside the DIVIDED band 0.40 and "
                  "outside the undivided 0.20.  Missing means the scale division is gone");
    PXG_CHECK_MSG(has(3u, TRI_BASE + (NTRI - 1u - 3u)), "tet 3 must match its own slot");
    PXG_CHECK_MSG(countFor(0u) == 1u,
                  "tet 0 must keep exactly ONE triangle: the back-facing one at 0.15 is "
                  "closer, and only the front-face gate keeps it out (got %u)", countFor(0u));
    PXG_CHECK_MSG(countFor(4u) == 0u, "tet 4 sits 0.45 out, beyond the 0.40 band");
    PXG_CHECK_MSG(countFor(5u) == 0u, "tet 5 is INTERIOR (surface hint 0) and must be skipped");

    // Tet 1 sees six qualifying triangles and keeps the four CLOSEST: its own slot at 0.12 and
    // the three nearest of the stack (0.02, 0.04, 0.06 below the plane -> 0.14, 0.16, 0.18).
    PXG_CHECK_MSG(countFor(1u) == 4u,
                  "tet 1 sees SIX qualifying triangles; MaxContacts caps it at four (got %u)",
                  countFor(1u));
    PXG_CHECK_MSG(has(1u, TRI_BASE + (NTRI - 1u - 1u)),
                  "tet 1's own slot triangle is the CLOSEST of the six and must survive the cap");
    PXG_CHECK_MSG(has(1u, STACK_TRI0 + NB_STACK - 1u),
                  "the nearest stack triangle must survive the cap");
    PXG_CHECK_MSG(!has(1u, STACK_TRI0),
                  "the FARTHEST stack triangle must be the one the cap discards");

    for (PxgBufferHandle h : { bBlob.h, bTetVerts.h, bTets.h, bHints.h, bDesc.h, bShapes.h,
                               bCms.h, bXf.h, bCd.h, bStack.h, bPairNum.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.10 step 2 -- sb_meshContactGenLaunch, the softbody-vs-TRIANGLE-MESH contact generator.
//
// WHAT THIS FIXTURE IS FOR.  The GJK/EPA core (tetPrimitivesCollide2) is covered bit-exactly by
// the npTet* battery and again by every softbody contact-gen fixture before this one, and the
// push blob is BYTE-IDENTICAL to sb_primitiveContactGenLaunch's, which its own fixture already
// covers.  What is new here is the geometry feed and the gate:
//
//   * the triangle is read out of the COOKED BLOB and taken to WORLD through vertex2Shape,
//     which MULTIPLIES by the mesh scale -- the mirror of step 1's shape2Vertex, which divides.
//     The mesh here carries scale (2, 1, 0.5) and a real 180-degree rotation, so a port that
//     used the wrong one puts the triangle somewhere else entirely and emits nothing;
//   * the front-face gate is `dot(gjkDirection, n) >= 0` with
//     `n = normalize(cross(v2 - v0, v1 - v0))` -- the NEGATIVE of the familiar winding normal.
//     Two triangles with OPPOSITE windings sit under two identical tetrahedra, so the gate is
//     two-sided: writing the familiar cross product swaps exactly which of them survives;
//   * the RIGID-shaped output: pairInd0 is the PxNodeIndex from
//     shapeToRigidRemapTable[TRIMESH cache ref], the material index comes from the TRIMESH
//     shape and is folded with & 0x7fffffff, and the three sort side-arrays are written;
//   * the stack clamp and the unclamped stackSizeNeededOnDevice tail.
// ===========================================================================
namespace p810cg
{
struct SbMeshCgPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, softbodies,
             shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad84;
    uint64_t filterPairs;
    uint32_t nbFilterPairs, pad100;
    uint64_t stackSizeNeededOnDevice;
    uint64_t wPoint, wNormalPen, wBarycentric, wContactInfo, wTotalCount, wContactByX,
             wTempContactByX, wContactIndexSortedByX, wContactSortedByX;
    uint32_t wMaxContacts, pad188;
};
static_assert(sizeof(SbMeshCgPush) == 192, "sb mesh CG blob must be 192 bytes");
static_assert(offsetof(SbMeshCgPush, stackSizeBytes) == 80, "stackSizeBytes @80");
static_assert(offsetof(SbMeshCgPush, filterPairs) == 88, "filterPairs @88");
static_assert(offsetof(SbMeshCgPush, stackSizeNeededOnDevice) == 104, "@104");
static_assert(offsetof(SbMeshCgPush, wPoint) == 112, "the inlined writer starts @112");
static_assert(offsetof(SbMeshCgPush, wMaxContacts) == 184, "maxNumContacts @184");

// PxgRigidFilterPair, 32 B.
struct RigidPairHost { uint64_t index0, index1; uint32_t index2, p0, p1, p2; };
static_assert(sizeof(RigidPairHost) == 32, "PxgRigidFilterPair is 32 bytes");
} // namespace p810cg

PXG_TEST(softBodySimMeshContactGen)
{
    using namespace p80refit;
    using namespace p88cloth;      // Bv32NodeHost
    using namespace p810cg;
    PxgSlangComputeDevice* dev = testDevice();

    // The mesh's placement, the same map step 1's fixture uses:
    //     world = R * (vertex * scale) + T,   R(x, y, z) = (-x, y, -z)
    const float SX = 2.0f, SY = 1.0f, SZ = 0.5f;
    const float TX = 10.0f, TY = 20.0f, TZ = 30.0f;
    auto toWorld = [&](float vx, float vy, float vz) {
        return F4{ -(vx * SX) + TX, (vy * SY) + TY, -(vz * SZ) + TZ, 1.0f };
    };

    // ---- the triangle mesh, in VERTEX space, two unit triangles in the plane y = 0 ---------
    // TRI_UP is wound so the FAMILIAR normal cross(v1-v0, v2-v0) is +y, which makes the
    // kernel's n = cross(v2-v0, v1-v0) point DOWN -- the same way GJK's direction points for a
    // tetrahedron sitting above it, so the gate ADMITS it.
    // TRI_DOWN is wound the other way, so the kernel's n points UP and the gate REJECTS it.
    // A port using the familiar cross product swaps exactly these two verdicts.
    //
    // *** THE TRIANGLES SIT AWAY FROM THE ORIGIN AND ARE SMALL, AND THAT IS THE ONLY REASON
    // THE MULTIPLY-VS-DIVIDE MUTANT DIES. ***  A triangle spanning vertex x in [0, 1] still
    // covers the tetrahedron under EITHER reading of the scale -- the scale is 1 in y, so the
    // vertical gap never moves, and any horizontal overlap at all yields the same contact.
    // Measured: that version let the mutant through.  At vertex x in [1.0, 1.4] the two
    // readings put the triangle at shape x in [2.0, 2.8] and [0.5, 0.7] respectively, 1.4
    // apart -- far outside the 0.05 contact distance -- so the wrong one emits nothing.
    std::vector<F4> triVerts;
    std::vector<TetIdx4> triIdx;
    const uint32_t TRI_UP = 0u, TRI_DOWN = 1u;
    {
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ 1.0f, 0.0f, 0.0f, 0.0f });
        triVerts.push_back(F4{ 1.0f, 0.0f, 0.4f, 0.0f });
        triVerts.push_back(F4{ 1.4f, 0.0f, 0.0f, 0.0f });
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });          // familiar normal +y
    }
    {
        const uint32_t b = uint32_t(triVerts.size());
        triVerts.push_back(F4{ 3.0f, 0.0f, 0.0f, 0.0f });
        triVerts.push_back(F4{ 3.4f, 0.0f, 0.0f, 0.0f });
        triVerts.push_back(F4{ 3.0f, 0.0f, 0.4f, 0.0f });     // familiar normal -y
        triIdx.push_back({ b, b + 1u, b + 2u, 0u });
    }
    const uint32_t NPRIM = uint32_t(triIdx.size());
    const uint32_t NVERT = uint32_t(triVerts.size());

    // ---- the COOKED BLOB: header | BV32 nodes | vertex pool | index pool -------------------
    // The tree itself is never walked by this kernel, but the node array still has to be
    // SKIPPED at the right stride to reach the vertex pool, so a real node is present.
    std::vector<uint32_t> blob;
    {
        Bv32NodeHost node;
        std::memset(&node, 0, sizeof(node));
        node.mNbNodes = 1u;
        node.mData[0] = 1u | (NPRIM << 1);
        blob.resize(4u, 0u);
        blob[0] = NVERT; blob[1] = NPRIM; blob[2] = 7u /* header.z, a decoy */; blob[3] = 1u;
        const size_t nodeWords = sizeof(Bv32NodeHost) / 4u;
        blob.resize(4u + nodeWords, 0u);
        std::memcpy(blob.data() + 4, &node, sizeof(node));
        const size_t vBase = blob.size();
        blob.resize(vBase + NVERT * 4u, 0u);
        std::memcpy(blob.data() + vBase, triVerts.data(), NVERT * sizeof(F4));
        const size_t iBase = blob.size();
        blob.resize(iBase + NPRIM * 4u, 0u);
        std::memcpy(blob.data() + iBase, triIdx.data(), NPRIM * sizeof(TetIdx4));
    }
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the softbody: three identical tets, each 0.02 above a triangle in vertex space ----
    // GAP is inside the contact distance 0.05, so GJK reports CLOSE rather than OVERLAP and the
    // emitted penetration is the POSITIVE separation.
    const float GAP = 0.02f;
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    // The footprint is a 0.1 right triangle well inside its 0.4 triangle, and the corners are
    // wound so isValidTet's signed volume is positive.
    auto addTetOver = [&](float ox) {
        const uint32_t b = uint32_t(tetVerts.size());
        tetVerts.push_back(toWorld(ox + 0.05f, GAP,        0.05f));
        tetVerts.push_back(toWorld(ox + 0.05f, GAP,        0.15f));
        tetVerts.push_back(toWorld(ox + 0.15f, GAP,        0.05f));
        tetVerts.push_back(toWorld(ox + 0.05f, GAP + 1.0f, 0.05f));
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
        return uint32_t(tets.size() - 1u);
    };
    const uint32_t TET_HIT  = addTetOver(1.0f);   // over TRI_UP    -> a contact
    const uint32_t TET_BACK = addTetOver(3.0f);   // over TRI_DOWN  -> gate rejects
    const uint32_t TET_FILT = addTetOver(1.0f);   // over TRI_UP    -> filtered out
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets);

    const uint32_t SB_ID = 3u;                    // NOT zero: the encode shift must shift
    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mNumTets = uint32_t(tets.size());
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr(SB_ID + 1u, makeZeroDesc());
    descArr[SB_ID] = sb;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  TRIMESH is shapeRef0 and SOFTBODY shapeRef1, i.e. SWAPPED, so the shape
    // load has to find the tet mesh by TYPE.  materialIndex carries bit 31 so the
    // `& 0x7fffffff` fold in the writer is observable.
    const uint32_t GEOM_TRIANGLEMESH_H = 8u;
    const uint32_t TRI_MATERIAL = 0x8000002Au;    // -> 0x2A = 42 after the fold
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;    shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = SX; shapes[0].sy = SY; shapes[0].sz = SZ; shapes[0].srw = 1.0f;
    shapes[0].materialIndex = TRI_MATERIAL;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = SB_ID;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    shapes[1].materialIndex = 999u;               // must NOT be the one that lands in the info
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*trimesh cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform, 32 B: slot 0 = the softbody (identity), slot 1 = the trimesh.
    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f;
    xf[8 + 1] = 1.0f;                                          // 180 deg about y
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);

    // contactDistance is summed over BOTH cache refs: 0.02 + 0.03 = 0.05.
    std::vector<float> cd = { 0.02f, 0.03f, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);
    // restDistance is indexed by cmIdx, and slot 0 is a decoy so an off-by-one shows.
    std::vector<float> rd = { 0.125f, 777.0f };
    DevBuf bRd = makeDev(dev, rd);

    // shapeToRigidRemapTable is indexed by the TRIMESH cache ref (1), not the softbody's (0).
    // Slot 0 is a decoy that must never appear in the output.
    std::vector<uint64_t> remap = { 0xDEAD0000DEAD0000ull, 0x0000000700000005ull };
    DevBuf bRemap = makeDev(dev, remap);
    const uint64_t RIGID_ID = remap[1];

    auto enc = [](uint32_t id, uint32_t elem) { return (id << 20) | elem; };
    std::vector<RigidPairHost> filters = { { RIGID_ID, enc(SB_ID, TET_FILT), 0u, 0u, 0u, 0u } };
    DevBuf bFilters = makeDev(dev, filters);

    // ---- the midphase stack.  FOUR entries but the kernel is told the stack holds only
    // THREE, so the last is CLAMPED away -- while stackSizeNeededOnDevice must still report
    // the UNCLAMPED demand.
    std::vector<uint32_t> stack;
    auto pushPair = [&](uint32_t tet, uint32_t tri) {
        stack.push_back(0u); stack.push_back(tet); stack.push_back(tri); stack.push_back(0u);
    };
    pushPair(TET_HIT,  TRI_UP);      // KEPT
    pushPair(TET_BACK, TRI_DOWN);    // dropped by the front-face gate
    pushPair(TET_FILT, TRI_UP);      // dropped by the filter
    pushPair(TET_HIT,  TRI_UP);      // CLAMPED AWAY (beyond the declared stack)
    const uint32_t NB_STACK_ENTRIES = uint32_t(stack.size() / 4u);
    const uint32_t DECLARED = NB_STACK_ENTRIES - 1u;
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { NB_STACK_ENTRIES };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 16u;
    const uint32_t POISON = 0xCDCDCDCDu;
    std::vector<F4> zeroF4(MAXC, F4{ 0, 0, 0, 0 });
    std::vector<uint32_t> poison(MAXC * 4u, POISON);
    std::vector<uint32_t> poison2(MAXC * 2u, POISON);
    DevBuf bPoint = makeDev(dev, zeroF4), bNormalPen = makeDev(dev, zeroF4),
           bBary = makeDev(dev, zeroF4), bInfo = makeDev(dev, poison),
           bByX = makeDev(dev, poison2), bTempByX = makeDev(dev, poison),
           bIdxSorted = makeDev(dev, poison), bSorted = makeDev(dev, poison);
    std::vector<uint32_t> zeroCount = { 0u };
    DevBuf bCount = makeDev(dev, zeroCount);

    SbMeshCgPush push = {};
    push.tolerenceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.shapeToRigidRemapTable = bRemap.addr;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.stackSizeBytes = DECLARED * 16u;
    push.filterPairs = bFilters.addr;
    push.nbFilterPairs = uint32_t(filters.size());
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.wPoint = bPoint.addr;
    push.wNormalPen = bNormalPen.addr;
    push.wBarycentric = bBary.addr;
    push.wContactInfo = bInfo.addr;
    push.wTotalCount = bCount.addr;
    push.wContactByX = bByX.addr;
    push.wTempContactByX = bTempByX.addr;
    push.wContactIndexSortedByX = bIdxSorted.addr;
    push.wContactSortedByX = bSorted.addr;
    push.wMaxContacts = MAXC;

    Kernel k("softBodyMeshContactGenBda.spv", "sb_meshContactGenLaunch", {}, sizeof(SbMeshCgPush));
    // The engine's grid (PxgNarrowphaseCore.cpp:5152): x = SB_MESHCG = 1024, y = 1.
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    const auto gPoint = downloadBuffer<F4>(dev, bPoint.h, MAXC);
    const auto gNormalPen = downloadBuffer<F4>(dev, bNormalPen.h, MAXC);
    const auto gInfo = downloadBuffer<uint32_t>(dev, bInfo.h, MAXC * 4u);
    const auto gByX = downloadBuffer<uint32_t>(dev, bByX.h, MAXC * 2u);
    const auto gTempByX = downloadBuffer<uint32_t>(dev, bTempByX.h, MAXC * 4u);
    const auto gIdxSorted = downloadBuffer<uint32_t>(dev, bIdxSorted.h, MAXC * 4u);
    const auto gSorted = downloadBuffer<uint32_t>(dev, bSorted.h, MAXC * 4u);
    const uint32_t gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];

    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
        std::printf("  contact %u: p=(%.5f %.5f %.5f) rest=%.5f n=(%.5f %.5f %.5f) pen=%.6f "
                    "ind0=0x%08x%08x ind1=(body %u tet %u) mat=%u\n",
                    i, gPoint[i].x, gPoint[i].y, gPoint[i].z, gPoint[i].w,
                    gNormalPen[i].x, gNormalPen[i].y, gNormalPen[i].z, gNormalPen[i].w,
                    gInfo[i * 4u + 1u], gInfo[i * 4u + 0u],
                    gInfo[i * 4u + 2u] >> 20, gInfo[i * 4u + 2u] & 0xfffffu,
                    gInfo[i * 4u + 3u]);

    PXG_CHECK_MSG(gCount == 1u,
                  "exactly ONE of the four stack entries survives: the back-facing pair is "
                  "rejected by the gate, the filtered pair by the filter list, and the fourth "
                  "is clamped away -- got %u.  2 usually means the front-face cross product is "
                  "the familiar way round; 0 means the triangle is not where vertex2Shape "
                  "should have put it", gCount);

    // ---- the surviving contact ----------------------------------------------
    PXG_CHECK_MSG(gInfo[0] == uint32_t(RIGID_ID & 0xFFFFFFFFull) &&
                  gInfo[1] == uint32_t(RIGID_ID >> 32),
                  "pairInd0 must be the PxNodeIndex from shapeToRigidRemapTable[TRIMESH cache "
                  "ref] = 0x%08x%08x, got 0x%08x%08x (0xDEAD.. means the SOFTBODY's ref was "
                  "used)", uint32_t(RIGID_ID >> 32), uint32_t(RIGID_ID & 0xFFFFFFFFull),
                  gInfo[1], gInfo[0]);
    PXG_CHECK_MSG(gInfo[2] == enc(SB_ID, TET_HIT),
                  "pairInd1 must encode (softbody %u, tet %u); got body %u tet %u",
                  SB_ID, TET_HIT, gInfo[2] >> 20, gInfo[2] & 0xfffffu);
    PXG_CHECK_MSG(gInfo[3] == (TRI_MATERIAL & 0x7FFFFFFFu),
                  "the material index comes from the TRIMESH shape and is folded with "
                  "& 0x7fffffff: expected %u, got %u", TRI_MATERIAL & 0x7FFFFFFFu, gInfo[3]);

    auto near = [](float a, float b, float t) { return std::fabs(a - b) <= t; };
    PXG_CHECK_MSG(near(gPoint[0].w, 0.125f, 1e-6f),
                  "the point's w is restDistance[cmIdx=0] = 0.125, got %.6f (777 means cmIdx "
                  "was ignored)", gPoint[0].w);
    // The tet sits above the triangle, GJK's direction runs cloud A (tet) -> cloud B
    // (triangle), so the normal points DOWN and the separation is POSITIVE.
    // GJK's direction runs cloud A (the tet, above) -> cloud B (the triangle, below), so it
    // points DOWN -- and this call site passes flipOutputNormal = true, so what is STORED is
    // its negation, pointing UP out of the floor.  That is the solver's convention, and it is
    // the difference between the floor holding the softbody and the floor swallowing it.
    PXG_CHECK_MSG(near(gNormalPen[0].y, 1.0f, 1e-4f) && near(gNormalPen[0].x, 0.0f, 1e-4f) &&
                  near(gNormalPen[0].z, 0.0f, 1e-4f),
                  "the STORED normal is the negation of GJK's direction (flipOutputNormal), so "
                  "it points UP out of the triangle; got (%.5f %.5f %.5f)",
                  gNormalPen[0].x, gNormalPen[0].y, gNormalPen[0].z);
    PXG_CHECK_MSG(near(gNormalPen[0].w, GAP, 2e-4f),
                  "the tet's bottom face is %.3f above the triangle and the two are separated, "
                  "so the stored penetration is that POSITIVE gap; got %.6f", GAP,
                  gNormalPen[0].w);
    // The stored point is the witness on the SECOND cloud, i.e. on the TRIANGLE (y = TY in
    // world, since the triangle's vertex-space plane y = 0 maps to world y = TY).
    PXG_CHECK_MSG(near(gPoint[0].y, TY, 1e-3f),
                  "the stored point is the witness on the TRIANGLE, at world y = %.3f; got "
                  "%.5f (a value %.3f higher is the witness on the TET instead)", TY,
                  gPoint[0].y, GAP);

    // ---- the writer's sort side-arrays ---------------------------------------
    // THE ALTERNATIVE WRITE MODE.  This call site passes alternativeWriteMode = true, so the
    // rigid id goes to contactSortedByX (low word only) and contactByX / tempContactByX are
    // NOT touched.  The hull path does the opposite.  Asserting the hull path's pair here is
    // exactly the mistake this fixture made first time round.
    PXG_CHECK_MSG(gSorted[0] == uint32_t(RIGID_ID & 0xFFFFFFFFull),
                  "contactSortedByX carries the rigid id's LOW word; got 0x%08x", gSorted[0]);
    PXG_CHECK_MSG(gByX[0] == POISON && gByX[1] == POISON,
                  "contactByX must be LEFT ALONE in the alternative write mode");
    PXG_CHECK_MSG(gTempByX[0] == POISON,
                  "tempContactByX must be LEFT ALONE in the alternative write mode");
    PXG_CHECK_MSG(gIdxSorted[0] == 0u,
                  "contactIndexSortedByX is seeded with the identity in BOTH modes");
    PXG_CHECK_MSG(gPoint[1].x == 0.0f && gNormalPen[1].x == 0.0f && gInfo[4] == POISON,
                  "nothing beyond the single contact may be touched");

    // ---- the clamp and the unclamped demand ----------------------------------
    PXG_CHECK_MSG(gNeeded == NB_STACK_ENTRIES * 16u,
                  "stackSizeNeededOnDevice must report the UNCLAMPED demand %u bytes so the "
                  "host can grow the stack; got %u (%u is the clamped count and would pin the "
                  "stack at its current size forever)",
                  NB_STACK_ENTRIES * 16u, gNeeded, DECLARED * 16u);

    for (PxgBufferHandle h : { bBlob.h, bTetVerts.h, bTets.h, bDesc.h, bShapes.h, bCms.h, bXf.h,
                               bCd.h, bRd.h, bRemap.h, bFilters.h, bStack.h, bPairNum.h,
                               bNeeded.h, bPoint.h, bNormalPen.h, bBary.h, bInfo.h, bByX.h,
                               bTempByX.h, bIdxSorted.h, bSorted.h, bCount.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.11 step 1 -- sb_heightfieldMidphaseGeneratePairsLaunch, the softbody-vs-HEIGHTFIELD
// midphase.  There is no BVH on this path: the tetrahedron's world AABB is pushed into the
// heightfield's unscaled row/column space and the integer rectangle it lands on names the
// candidate quads directly.  What this fixture aims at:
//
//   * the flat cooked blob -- nbRows @0, nbCols @4, 4-byte samples @8 -- and the SIGNED 16-bit
//     height, which a raw mask would read as a huge positive number;
//   * the material bytes' bit 7 (the tessFlag): a material of 127 WITH the flag set is still a
//     hole, and one of 255 is a hole too.  Comparing the raw byte gets both backwards;
//   * the row/column clamps, including that getMinRow clamps to rows-2 and getMaxRow to rows-1;
//   * the y-range reject: a tetrahedron entirely above (or entirely below) all four corner
//     heights of a quad contributes nothing;
//   * the emitted triangle index (vertexIndex << 1) | isMaterial1;
//   * the transform: a real rotation and a NEGATIVE row scale, which mirrors the axis and
//     needs the min/max swap;
//   * the 72-byte blob and the 64-warps-per-contact-manager work partition.
// ===========================================================================
namespace p811hf
{
struct SbHfMidPush
{
    uint32_t numNPWorkItems; float toleranceLength;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, softbodies;
    uint32_t stackSizeBytes, pad52;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(SbHfMidPush) == 72, "sb heightfield midphase blob must be 72 bytes");
static_assert(offsetof(SbHfMidPush, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(SbHfMidPush, stackSizeBytes) == 48, "stackSizeBytes @48");
static_assert(offsetof(SbHfMidPush, stackPtr) == 56, "stackPtr @56");

// PxHeightFieldSample, 4 B: PxI16 height, PxBitAndByte materialIndex0, materialIndex1.
static uint32_t hfSample(int16_t height, uint8_t mat0, uint8_t mat1)
{
    return (uint32_t(uint16_t(height))) | (uint32_t(mat0) << 16) | (uint32_t(mat1) << 24);
}
} // namespace p811hf

PXG_TEST(softBodySimHeightfieldMidphase)
{
    using namespace p80refit;
    using namespace p811hf;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the heightfield: 4 rows x 4 columns, flat at height 0 except one raised corner ----
    // Row index runs along x and column along z in heightfield space.  Sample (r, c) lives at
    // vertexIndex = r * nbCols + c.
    const uint32_t NB_ROWS = 4u, NB_COLS = 4u;
    const uint8_t MAT_SOLID = 3u;
    const uint8_t MAT_HOLE = 127u;                 // PxHeightFieldMaterial::eHOLE
    const uint8_t MAT_HOLE_TESS = 127u | 0x80u;    // eHOLE with the tessellation flag set
    const uint8_t MAT_SOLID_TESS = 3u | 0x80u;     // a SOLID material with the flag set

    std::vector<uint32_t> blob(2u + NB_ROWS * NB_COLS, 0u);
    blob[0] = NB_ROWS;
    blob[1] = NB_COLS;
    auto setSample = [&](uint32_t r, uint32_t c, int16_t h, uint8_t m0, uint8_t m1) {
        blob[2u + r * NB_COLS + c] = hfSample(h, m0, m1);
    };
    for (uint32_t r = 0; r < NB_ROWS; ++r)
        for (uint32_t c = 0; c < NB_COLS; ++c)
            setSample(r, c, 0, MAT_SOLID, MAT_SOLID);
    // Quad (0,0): its FIRST triangle is a hole, its second solid.  Only one pair may come out.
    setSample(0u, 0u, 0, MAT_HOLE, MAT_SOLID);
    // Quad (0,1): a hole whose material byte ALSO carries the tessellation flag.  A port that
    // compares the raw byte sees 255 != 127 and lets both triangles through.
    setSample(0u, 1u, 0, MAT_HOLE_TESS, MAT_HOLE_TESS);
    // Quad (1,0): SOLID but with the flag set, i.e. the mirror mistake -- masking wrongly the
    // other way would turn 131 into something that is not 3 and drop a real triangle.
    setSample(1u, 0u, 0, MAT_SOLID_TESS, MAT_SOLID_TESS);
    // Rows 2 AND 3 are RAISED to height 50, so every corner of a quad in row 2 is above the
    // tetrahedra and the y-range test rejects it.  BOTH rows have to move: with only row 2
    // raised the quad straddles the tetrahedron -- two corners at +50 and two at 0 or below --
    // and it is CORRECTLY not rejected.  Measured, on the first version of this fixture.
    for (uint32_t c = 0; c < NB_COLS; ++c) { setSample(2u, c, 50, MAT_SOLID, MAT_SOLID); }
    for (uint32_t c = 0; c < NB_COLS; ++c) { setSample(3u, c, 50, MAT_SOLID, MAT_SOLID); }
    // *** THE ONE PLACE THE HEIGHT'S SIGN IS OBSERVABLE. ***  The heights are PxI16, and reading
    // them as unsigned turns -1 into 65535.  For a quad whose corners are ALL negative that
    // changes nothing -- the signed read rejects it through con1 (minY >= 0 > every corner) and
    // the unsigned read through con0 (maxY < 65535), so both reject and the mutant survives.
    // It only shows on a MIXED quad: corners at -1 and +50 with a tetrahedron whose maxY is
    // below both.  Signed: con0 fails on -1 and con1 fails on +50, so the quad is kept.
    // Unsigned: -1 has become 65535 and now maxY is below ALL FOUR, so con0 fires and the quad
    // is dropped.  Quad (1,1) is built that way and T_SIGN sits under it.
    setSample(1u, 1u, -1, MAT_SOLID, MAT_SOLID);
    setSample(1u, 2u, -1, MAT_SOLID, MAT_SOLID);
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the heightfield's placement -------------------------------------------------------
    // Non-unit row, height and column scales, so all three divisions matter, plus a rigid
    // rotation of 180 degrees about y, which is exact in floating point.
    //     world = R * (hf * scale) + T,   R(x, y, z) = (-x, y, -z)
    //
    // *** THE SCALES ARE POSITIVE, AND THE NEGATIVE-SCALE SWAP IS THEREFORE NOT COVERED. ***
    // Upstream un-scales the local bound by 1/|scale| and then swaps min and max when the
    // scale is negative (softbodyHFMidPhase.cu:215-229).  That is NOT the same as dividing by
    // the SIGNED scale: for rowScale = -1 and a tetrahedron at row 0.2..0.6 the shape-space
    // bound is -0.6..-0.2, the division by |scale| leaves it there, and the swap produces
    // min = -0.2, max = -0.6 -- an inverted interval whose floor/ceil clamps collapse to
    // maxRow == minRow, so the quad count is zero and NOTHING is emitted.  Measured: a first
    // version of this fixture used rowScale = -1 and got 0 pairs for exactly that reason.
    // The port transcribes the branch as upstream has it; the fixture does not pretend to
    // exercise it, and a mutant deleting the swap survives.  Stated rather than papered over.
    const float ROW_SCALE = 2.0f, HEIGHT_SCALE = 0.01f, COL_SCALE = 0.5f;
    const float TX = 5.0f, TY = 7.0f, TZ = 9.0f;
    auto toWorld = [&](float hr, float hh, float hc) {
        return F4{ -(hr * ROW_SCALE) + TX, (hh * HEIGHT_SCALE) + TY, -(hc * COL_SCALE) + TZ, 1.0f };
    };

    // ---- the softbody: small tetrahedra hovering over chosen quads -------------------------
    // Each sits just above height 0 in heightfield units, inside ONE quad's footprint.
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    std::vector<uint8_t> hints;
    auto addTet = [&](float r, float c, float h, uint8_t hint) {
        const uint32_t b = uint32_t(tetVerts.size());
        tetVerts.push_back(toWorld(r + 0.2f, h,        c + 0.2f));
        tetVerts.push_back(toWorld(r + 0.2f, h,        c + 0.6f));
        tetVerts.push_back(toWorld(r + 0.6f, h,        c + 0.2f));
        tetVerts.push_back(toWorld(r + 0.2f, h + 5.0f, c + 0.2f));
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
        hints.push_back(hint);
        return uint32_t(tets.size() - 1u);
    };
    const uint32_t T_HOLE0   = addTet(0.0f, 0.0f, 1.0f, 1u);  // quad (0,0): 1 of 2 is a hole
    const uint32_t T_HOLETES = addTet(0.0f, 1.0f, 1.0f, 1u);  // quad (0,1): BOTH are holes
    const uint32_t T_SOLTES  = addTet(1.0f, 0.0f, 1.0f, 1u);  // quad (1,0): both solid
    const uint32_t T_BELOW   = addTet(2.0f, 1.0f, 1.0f, 1u);  // under a quad raised to +50
    const uint32_t T_ABOVE   = addTet(0.0f, 2.0f, 200.0f, 1u); // far ABOVE a quad at height 0
    const uint32_t T_SIGN    = addTet(1.0f, 1.0f, 1.0f, 1u);  // over the MIXED-sign quad (1,1)
    const uint32_t T_INNER   = addTet(1.0f, 1.0f, 1.0f, 0u);  // interior tet, hint 0
    // *** THE ONLY TETRAHEDRON WHOSE LOCAL BOUND GOES NEGATIVE IN Y. ***
    //
    // The y range reaches `heightfieldComputePairs` as PxU32 (heightfieldUtil.cuh:334), so the
    // caller performs an implicit float->unsigned conversion, and CUDA compiles that to
    // cvt.rzi.u32.f32, which SATURATES: a negative becomes 0, not a huge number.  Every other
    // tetrahedron here sits at h = 1.0, and the world-space fattening of 0.002 is 0.2 in
    // height units, so their lMin.y is +0.8 and the negative branch is never reached -- which
    // is why this fixture passed identically with and without the saturation (measured).
    //
    // At h = 0.1 the same arithmetic gives lMin.y = -0.1.  Under the CORRECT reading minY is
    // 0, exactly as it is for the h = 1.0 tetrahedra (truncation), and quad (0,2) -- corners
    // 0, 0, -1, 0 -- survives con1 because `0 > 0` is false.  Under the C++ HOST reading, where
    // a negative float becomes a huge unsigned, minY exceeds every corner, con1 fires and the
    // quad is dropped.  Two pairs against zero: the two readings cannot both pass.
    const uint32_t T_NEGY    = addTet(0.0f, 2.0f, 0.1f, 1u);
    (void)T_BELOW; (void)T_ABOVE; (void)T_INNER;
    std::vector<uint32_t> hintWords((hints.size() + 3u) / 4u + 1u, 0u);
    for (size_t i = 0; i < hints.size(); ++i)
        reinterpret_cast<uint8_t*>(hintWords.data())[i] = hints[i];
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets),
           bHints = makeDev(dev, hintWords);

    // numTets comes out of the TETMESH BLOB HEADER (.y), not mNumTets -- so the header carries
    // the real count and mNumTets is set to a decoy that must never be used.
    std::vector<uint32_t> tetMeshHeader = { 999u, uint32_t(tets.size()), 1u, 1u };
    DevBuf bTetMesh = makeDev(dev, tetMeshHeader);

    const uint32_t SB_ID = 2u;
    PxgSoftBody sb = makeZeroDesc();
    sb.mTetMeshData = reinterpret_cast<void*>(size_t(bTetMesh.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = 1u;                      // the DECOY: using it would visit one tet only
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr(SB_ID + 1u, makeZeroDesc());
    descArr[SB_ID] = sb;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  HEIGHTFIELD is shapeRef0, SOFTBODY shapeRef1 -- the SWAPPED order.
    const uint32_t GEOM_HEIGHTFIELD_H = 9u;
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_HEIGHTFIELD_H;     shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = ROW_SCALE; shapes[0].sy = HEIGHT_SCALE; shapes[0].sz = COL_SCALE;
    shapes[0].srw = 1.0f;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = SB_ID;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*hf cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f;
    xf[8 + 1] = 1.0f;                                       // 180 deg about y
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);

    // A SMALL contact distance: the bound is fattened by it in WORLD units before the transform,
    // and anything large would spill each tetrahedron into its neighbours' quads.
    std::vector<float> cd = { 0.001f, 0.001f, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);

    const uint32_t CAP = 64u;
    std::vector<uint32_t> stack(CAP * 4u, 0xFFFFFFFFu);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 0u };
    DevBuf bPairNum = makeDev(dev, pairNum);

    SbHfMidPush push = {};
    push.numNPWorkItems = 1u;
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bDesc.addr;
    push.stackSizeBytes = CAP * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;

    Kernel k("softBodyHfMidphaseBda.spv", "sb_heightfieldMidphaseGeneratePairsLaunch",
             {}, sizeof(SbHfMidPush));
    // The engine's grid (PxgNarrowphaseCore.cpp:5360): x = SB_SBMIDPHASE = 1024, y = 1.
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);

    std::set<std::pair<uint32_t, uint32_t>> emitted;   // (tet, triangleIdx)
    for (uint32_t i = 0; i < gNum && i < CAP; ++i)
    {
        PXG_CHECK_MSG(gStack[i * 4u + 0u] == 0u, "pair %u carries cmIdx %u, not 0", i,
                      gStack[i * 4u + 0u]);
        emitted.insert({ gStack[i * 4u + 1u], gStack[i * 4u + 2u] });
        std::printf("  pair %u: tet %u  tri %u (vertexIndex %u, material slot %u)\n",
                    i, gStack[i * 4u + 1u], gStack[i * 4u + 2u],
                    gStack[i * 4u + 2u] >> 1, gStack[i * 4u + 2u] & 1u);
    }

    auto countFor = [&](uint32_t tet) {
        uint32_t n = 0; for (auto& e : emitted) if (e.first == tet) ++n; return n;
    };
    auto has = [&](uint32_t tet, uint32_t vtx, uint32_t slot) {
        return emitted.count({ tet, (vtx << 1) | slot }) != 0;
    };

    PXG_CHECK_MSG(countFor(T_HOLE0) == 1u,
                  "quad (0,0) has ONE hole material and one solid, so exactly one triangle "
                  "survives; got %u", countFor(T_HOLE0));
    PXG_CHECK_MSG(has(T_HOLE0, 0u, 1u),
                  "the surviving triangle of quad (0,0) is the SECOND material slot, so its "
                  "index is (vertexIndex << 1) | 1");
    PXG_CHECK_MSG(countFor(T_HOLETES) == 0u,
                  "quad (0,1) is a hole in BOTH slots, but its material byte carries the "
                  "tessellation flag as well.  Getting %u instead of 0 means the flag was not "
                  "masked off before comparing against eHOLE", countFor(T_HOLETES));
    PXG_CHECK_MSG(countFor(T_SOLTES) == 2u,
                  "quad (1,0) is SOLID in both slots with the tessellation flag set, so both "
                  "triangles survive; got %u", countFor(T_SOLTES));
    PXG_CHECK_MSG(countFor(T_INNER) == 0u,
                  "the interior tetrahedron (surface hint 0) must be skipped entirely");
    // The two halves of the y-range test, one each.
    PXG_CHECK_MSG(countFor(T_BELOW) == 0u,
                  "this tetrahedron sits entirely BELOW all four corner heights of its quad "
                  "(+50), so con0 rejects it; got %u", countFor(T_BELOW));
    PXG_CHECK_MSG(countFor(T_ABOVE) == 0u,
                  "this tetrahedron sits entirely ABOVE all four corner heights of its quad "
                  "(0), so con1 rejects it; got %u", countFor(T_ABOVE));

    PXG_CHECK_MSG(countFor(T_SIGN) == 2u,
                  "quad (1,1) has corners at -1 and +50, both slots solid, so BOTH triangles "
                  "survive under a SIGNED height read; got %u.  0 means the PxI16 height was "
                  "read as unsigned and -1 became 65535", countFor(T_SIGN));

    // ---- TWO MUTANTS SURVIVE THIS FIXTURE, AND THE ARITHMETIC SAYS WHY ------------------
    //
    // getMinRow clamps to rows-2 and getMaxRow to rows-1, and swapping the first for the
    // second changes nothing here -- NOT because the fixture is thin, but because upstream's
    // own rectangle guard gets there first.  The guard is `!(lMin.x > nbRows - 1) || ...`, so
    // any tetrahedron whose minimum row would exceed rows-2 is either past nbRows-1 and thrown
    // out by the guard, or below it and clamped identically by both versions.  A tetrahedron
    // placed at row 3.4 to try to catch it is simply rejected earlier (measured: 0 pairs).
    //
    // Dropping `fattenFast(contactDist)` also survives.  The contact distance here is
    // 0.001 + 0.001 = 0.002 in WORLD units, i.e. 0.001 of a row at rowScale 2 -- three orders
    // of magnitude below the quad size, so the fattened and unfattened bounds floor and ceil
    // to the same integers.  Raising it enough to cross a boundary would spill every other
    // tetrahedron into its neighbours' quads and dissolve the rest of the fixture.  Recorded
    // rather than papered over; step 2's fixture is where the contact distance has teeth.

    PXG_CHECK_MSG(countFor(T_NEGY) == 2u,
                  "this tetrahedron's local bound reaches y = -0.1, and the y range is passed "
                  "to heightfieldComputePairs as PxU32.  CUDA's cvt.rzi.u32.f32 SATURATES a "
                  "negative to 0, so minY is 0, `0 > 0` is false for quad (0,2)'s corner at "
                  "height 0, and both its triangles survive.  Got %u -- 0 means the conversion "
                  "was transcribed with C++ host semantics, where a negative float becomes a "
                  "huge unsigned and con1 rejects every quad", countFor(T_NEGY));

    PXG_CHECK_MSG(gNum == 7u,
                  "seven pairs in total -- one from quad (0,0), two from quad (1,0), two from "
                  "quad (1,1) and two from quad (0,2) via the negative-y tetrahedron; got "
                  "%u.  0 usually means the blob walk or the inverse transform is wrong; a "
                  "large number means the row/column rectangle is not being clamped and the "
                  "whole field is being scanned", gNum);

    for (PxgBufferHandle h : { bBlob.h, bTetVerts.h, bTets.h, bHints.h, bTetMesh.h, bDesc.h,
                               bShapes.h, bCms.h, bXf.h, bCd.h, bStack.h, bPairNum.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.19 -- the FEM-vs-FEM ATTACHMENT pair.  Two fixtures, deliberately INDEPENDENT: the prepare
// is checked against an explicit expectation and the solve is fed a constraint block written by
// the HOST rather than by the prepare, so a scatter bug and a solve bug cannot cancel.
//
// What they pin:
//   * both push blobs (32 and 24 bytes) and the two struct layouts -- the warp-strided
//     PxgFEMFEMAttachmentConstraint (stride 2688, every field an array of 32 indexed by
//     workIndex & 31) and PxgFEMFEMAttachment (stride 80);
//   * the activeFemAttachments INDIRECTION (a non-identity permutation) and the block/offset
//     split across a 32-attachment boundary;
//   * that low_high_angles takes (x, y, z, 0) -- the source .w is DROPPED -- and that elemId1,
//     a PxU32 widened into a PxU64 slot, gets its upper word CLEARED;
//   * the barycentric anchor interpolation, the inverse-mass split, the `wTot > 0` gate and the
//     per-corner `invMassBary > 0` gate;
//   * the cone-limit branch, which no scene reaches (an attachment without a
//     PxConeLimitedConstraint carries -1 in all three parameters), including the
//     constraintOffset term that only a non-zero offset can expose.
// ===========================================================================
namespace p819att
{
struct FemAttachPrepPush
{
    uint64_t femAttachments, activeFemAttachments;
    uint32_t numFemAttachments, pad20;
    uint64_t attachmentConstraints;
};
static_assert(sizeof(FemAttachPrepPush) == 32, "fem attach prepare blob must be 32 bytes");
static_assert(offsetof(FemAttachPrepPush, numFemAttachments) == 16, "numFemAttachments @16");
static_assert(offsetof(FemAttachPrepPush, attachmentConstraints) == 24, "constraints @24");

struct FemAttachSolvePush
{
    uint64_t softbodies, attachments;
    uint32_t numAttachments, pad20;
};
static_assert(sizeof(FemAttachSolvePush) == 24, "fem attach solve blob must be 24 bytes");
static_assert(offsetof(FemAttachSolvePush, numAttachments) == 16, "numAttachments @16");

// PxgFEMFEMAttachment (PxgSimulationCoreDesc.h:265), sizeof 80.
struct FemAttachmentHost
{
    F4 lowHighAngle;      // @0   coneLimitParams.low_high_angle
    F4 barycentric;       // @16  coneLimitParams.barycentric
    F4 bary0;             // @32
    F4 bary1;             // @48
    uint64_t index0;      // @64
    uint32_t index1;      // @72
    float constraintOffset; // @76 -> 80
};
static_assert(sizeof(FemAttachmentHost) == 80, "PxgFEMFEMAttachment mirror must be 80 bytes");
static_assert(offsetof(FemAttachmentHost, index0) == 64, "index0 @64");
static_assert(offsetof(FemAttachmentHost, constraintOffset) == 76, "constraintOffset @76");

// PxgFEMFEMAttachmentConstraint (PxgFEMCore.h:204) -- the warp-strided block.
static const size_t FAC_STRIDE          = 2688;
static const size_t FAC_LOW_HIGH_ANGLES = 0;
static const size_t FAC_ATT_BARYCENTRIC = 512;
static const size_t FAC_BARYCENTRIC0    = 1024;
static const size_t FAC_BARYCENTRIC1    = 1536;
static const size_t FAC_ELEMID0         = 2048;
static const size_t FAC_ELEMID1         = 2304;
static const size_t FAC_CONSTR_OFFSET   = 2560;

static const uint32_t SB_ID_SHIFT = 20u;
static uint32_t encodeSb(uint32_t id, uint32_t elem) { return (id << SB_ID_SHIFT) | elem; }
} // namespace p819att

PXG_TEST(softBodySimFemAttachmentPrepare)
{
    using namespace p819att;
    PxgSlangComputeDevice* dev = testDevice();

    // 34 attachments: enough to straddle the 32-wide block, so blockIdx 0 and 1 and offsets 0
    // and 1 are all exercised.  The ACTIVE list is a non-identity permutation -- attachment
    // data must follow activeFemAttachments[workIndex], not workIndex.
    const uint32_t N = 34u;
    std::vector<FemAttachmentHost> atts(N);
    std::vector<uint32_t> active(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        FemAttachmentHost& a = atts[i];
        // Distinct, recognisable values; the .w of lowHighAngle is POISON and must be dropped.
        a.lowHighAngle = F4{ 1.0f + float(i), 2.0f + float(i), 3.0f + float(i), 99.0f };
        a.barycentric  = F4{ 0.1f * float(i), 0.2f, 0.3f, 0.4f };
        a.bary0        = F4{ float(i), 1.0f, 2.0f, 3.0f };
        a.bary1        = F4{ 4.0f, float(i), 5.0f, 6.0f };
        // index0's UPPER word is non-zero on purpose: it is a PxU64 copied WHOLE.
        a.index0 = (uint64_t(0xABCD0000u + i) << 32) | uint64_t(encodeSb(1u, 7u + i));
        a.index1 = encodeSb(2u, 11u + i);
        a.constraintOffset = 0.5f * float(i);
        active[i] = (i + 7u) % N;    // the permutation
    }
    DevBuf bAtt = makeDev(dev, atts);
    DevBuf bActive = makeDev(dev, active);
    if (!bAtt.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // Two constraint blocks, poisoned so an untouched field is visible.
    const size_t NB_BLOCKS = 2;
    std::vector<uint32_t> con(NB_BLOCKS * FAC_STRIDE / 4, 0xCDCDCDCDu);
    DevBuf bCon = makeDev(dev, con);

    FemAttachPrepPush push = {};
    push.femAttachments = bAtt.addr;
    push.activeFemAttachments = bActive.addr;
    push.numFemAttachments = N;
    push.attachmentConstraints = bCon.addr;

    Kernel k("softBodyFemAttachBda.spv", "femAttachmentPrepareLaunch", {}, sizeof(FemAttachPrepPush));
    // The engine's grid (PxgSoftBodyCore.cpp:2113): SB_UPDATEROTATION = 1024 blocks of 256.
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<uint32_t>(dev, bCon.h, uint32_t(con.size()));
    auto f4At = [&](size_t byteOff) {
        const size_t w = byteOff / 4;
        return F4{ u2f(got[w]), u2f(got[w + 1]), u2f(got[w + 2]), u2f(got[w + 3]) };
    };
    auto u64At = [&](size_t byteOff) {
        const size_t w = byteOff / 4;
        return uint64_t(got[w]) | (uint64_t(got[w + 1]) << 32);
    };
    auto fAt = [&](size_t byteOff) { return u2f(got[byteOff / 4]); };
    auto nearF = [](float a, float b) { return std::fabs(a - b) < 1e-6f; };

    size_t mm = 0;
    for (uint32_t w = 0; w < N; ++w)
    {
        const size_t blk = w / 32, off = w & 31;
        const size_t base = blk * FAC_STRIDE;
        const FemAttachmentHost& a = atts[active[w]];

        const F4 lha = f4At(base + FAC_LOW_HIGH_ANGLES + off * 16);
        if (!nearF(lha.x, a.lowHighAngle.x) || !nearF(lha.y, a.lowHighAngle.y) ||
            !nearF(lha.z, a.lowHighAngle.z) || !nearF(lha.w, 0.0f))
        { ++mm; std::printf("  w=%u low_high_angles MM: got (%.3f %.3f %.3f %.3f) ref (%.3f %.3f %.3f 0)\n",
                            w, lha.x, lha.y, lha.z, lha.w,
                            a.lowHighAngle.x, a.lowHighAngle.y, a.lowHighAngle.z); }

        const F4 ab = f4At(base + FAC_ATT_BARYCENTRIC + off * 16);
        if (!nearF(ab.x, a.barycentric.x) || !nearF(ab.w, a.barycentric.w))
        { ++mm; std::printf("  w=%u attachmentBarycentric MM\n", w); }

        const F4 b0 = f4At(base + FAC_BARYCENTRIC0 + off * 16);
        const F4 b1 = f4At(base + FAC_BARYCENTRIC1 + off * 16);
        if (!nearF(b0.x, a.bary0.x) || !nearF(b0.w, a.bary0.w))
        { ++mm; std::printf("  w=%u barycentric0 MM: got %.3f ref %.3f\n", w, b0.x, a.bary0.x); }
        if (!nearF(b1.y, a.bary1.y) || !nearF(b1.w, a.bary1.w))
        { ++mm; std::printf("  w=%u barycentric1 MM: got %.3f ref %.3f\n", w, b1.y, a.bary1.y); }

        const uint64_t e0 = u64At(base + FAC_ELEMID0 + off * 8);
        if (e0 != a.index0)
        { ++mm; std::printf("  w=%u elemId0 MM: got 0x%llx ref 0x%llx (a PxU64 copied WHOLE)\n",
                            w, (unsigned long long)e0, (unsigned long long)a.index0); }
        const uint64_t e1 = u64At(base + FAC_ELEMID1 + off * 8);
        if (e1 != uint64_t(a.index1))
        { ++mm; std::printf("  w=%u elemId1 MM: got 0x%llx ref 0x%x (a PxU32 zero-EXTENDED)\n",
                            w, (unsigned long long)e1, a.index1); }

        if (!nearF(fAt(base + FAC_CONSTR_OFFSET + off * 4), a.constraintOffset))
        { ++mm; std::printf("  w=%u constraintOffset MM\n", w); }
    }
    PXG_CHECK_MSG(mm == 0, "femAttachmentPrepare scatter (%zu mismatches)", mm);

    // Slots 34..63 of block 1 were never written and must still be poison -- the strided loop
    // must not run past numFemAttachments.
    size_t spill = 0;
    for (uint32_t w = N; w < 64u; ++w)
    {
        const size_t base = (w / 32) * FAC_STRIDE, off = w & 31;
        if (got[(base + FAC_ELEMID0 + off * 8) / 4] != 0xCDCDCDCDu) ++spill;
    }
    PXG_CHECK_MSG(spill == 0, "the prepare wrote %zu slots beyond numFemAttachments", spill);

    for (PxgBufferHandle h : { bAtt.h, bActive.h, bCon.h }) dev->destroyBuffer(h);
}

PXG_TEST(softBodySimFemAttachmentSolve)
{
    using namespace p819att;
    using p80refit::TetIdx4;
    PxgSlangComputeDevice* dev = testDevice();

    // Two softbodies, three tetrahedra each with DISJOINT vertices, so the three attachments
    // below never share a corner and each expected delta can be read on its own.
    //
    // The inverse masses and the barycentrics are deliberately NON-UNIFORM and DIFFERENT between
    // the two sides.  With (1,0,0,0) on both -- the first version of this fixture -- four mutants
    // survived: swapping barycentric0/1 is invisible when they are equal, and with only corner 0
    // ever weighted, dropping the per-corner `invMassBary > 0` gate and dropping the barycentric
    // factor from b0w both change nothing.  Two corners now carry weight ZERO for two DIFFERENT
    // reasons -- one has invMass 0, the other has barycentric 0 -- and each kills one of those.
    //
    // SIX tetrahedra since P9.33b, one per cone state -- see CONE_STATE below.  Tet 5 is the
    // fully pinned pair (wTot == 0); it used to be tet 2, which collided with the one state
    // that reaches the gate's `else` arm and would have left that arm unobserved.
    const uint32_t NB_TETS = 6u;
    const float invM0[6][4] = { { 1.0f, 2.0f, 0.0f, 4.0f }, { 1.0f, 2.0f, 0.0f, 4.0f },
                                { 1.0f, 2.0f, 0.0f, 4.0f }, { 1.0f, 2.0f, 0.0f, 4.0f },
                                { 1.0f, 2.0f, 0.0f, 4.0f }, { 0, 0, 0, 0 } };
    const float invM1[6][4] = { { 0.5f, 0.0f, 1.5f, 2.0f }, { 0.5f, 0.0f, 1.5f, 2.0f },
                                { 0.5f, 0.0f, 1.5f, 2.0f }, { 0.5f, 0.0f, 1.5f, 2.0f },
                                { 0.5f, 0.0f, 1.5f, 2.0f }, { 0, 0, 0, 0 } };

    // Every cone state the engine can present, one per attachment.  Until P9.33b this rig had
    // two, and NEITHER was the one the engine actually writes:
    //   0  THE ENGINE DEFAULT (0, 0, -1, 0), set for every attachment created without a
    //      PxConeLimitedConstraint (PxgSimulationController.cpp:3696, and :3859 for the
    //      softbody-cloth pair).  `minDist >= 0` makes isConeLimitedEnabled TRUE, so the
    //      branch runs on EVERY attachment in EVERY scene -- the opposite of what this
    //      shader's header claimed until P9.33b.  len clamps into [0,0], so the error is the
    //      caller's relPos, which already carries `- worldAxis * constraintOffset`.
    //   1  enabled through maxDist alone; the angle arm stays on its maxAngle < 0 path.
    //   2  DISABLED.  The engine never writes this, but it is the only state that reaches the
    //      gate's `else` arm, so it stays as the negative control.
    //   3  A REAL CONE with the relative position FAR outside it -> cosAngle < cosMaxAngle and
    //      the sin/cos arm runs.  Nothing here reached that arm before, though the header
    //      said the fixture drove it directly.
    //   4  A real cone with the relative position INSIDE it -> the deviation is skipped.
    //      Without this one `if (cosAngle < cosMaxAngle)` is only ever true, and a port that
    //      dropped the test entirely would pass.
    //   5  the fully pinned pair, on the engine default, so wTot == 0 and nothing is written.
    static const F4 CONE_STATE[6] = {
        F4{ 0.0f, 0.0f, -1.0f, 0.0f },     // engine default: enabled via minDist
        F4{ -1.0f, 0.5f, -1.0f, 0.0f },    // enabled via maxDist
        F4{ -1.0f, -1.0f, -1.0f, 0.0f },   // disabled
        F4{ -1.0f, -1.0f, 0.30f, 0.0f },   // real cone, relPos outside
        F4{ -1.0f, -1.0f, 3.00f, 0.0f },   // real cone, relPos inside
        F4{ 0.0f, 0.0f, -1.0f, 0.0f },     // engine default, fully pinned
    };
    static const float PX_MAX_REAL_HOST = 3.4028235e38f;

    auto makeVerts = [&](float oy, const float invM[6][4]) {
        std::vector<F4> v;
        for (uint32_t t = 0; t < NB_TETS; ++t)
        {
            const float ox = 10.0f * float(t);
            v.push_back(F4{ ox,        oy,        0.0f, invM[t][0] });
            v.push_back(F4{ ox + 1.0f, oy,        0.0f, invM[t][1] });
            v.push_back(F4{ ox,        oy + 1.0f, 0.0f, invM[t][2] });
            v.push_back(F4{ ox,        oy,        1.0f, invM[t][3] });
        }
        return v;
    };
    std::vector<F4> v0 = makeVerts(0.0f, invM0);
    std::vector<F4> v1 = makeVerts(2.0f, invM1);

    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < NB_TETS; ++t)
        tets.push_back(TetIdx4{ t * 4u, t * 4u + 1u, t * 4u + 2u, t * 4u + 3u });

    std::vector<F4> delta0(v0.size(), F4{ 0, 0, 0, 0 }), delta1(v1.size(), F4{ 0, 0, 0, 0 });

    DevBuf bV0 = makeDev(dev, v0), bV1 = makeDev(dev, v1), bTets = makeDev(dev, tets),
           bD0 = makeDev(dev, delta0), bD1 = makeDev(dev, delta1);
    if (!bV0.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb0 = makeZeroDesc(), sb1 = makeZeroDesc();
    sb0.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bV0.addr));
    sb0.mSimTetIndices = reinterpret_cast<uint4*>(size_t(bTets.addr));
    sb0.mSimDelta = reinterpret_cast<float4*>(size_t(bD0.addr));
    sb1.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bV1.addr));
    sb1.mSimTetIndices = reinterpret_cast<uint4*>(size_t(bTets.addr));
    sb1.mSimDelta = reinterpret_cast<float4*>(size_t(bD1.addr));
    std::vector<PxgSoftBody> descArr = { sb0, sb1 };
    DevBuf bDesc = makeDev(dev, descArr);

    // The constraint block, written by the HOST -- see the section header on why.
    const uint32_t NB_ATT = NB_TETS;              // one attachment per cone state
    const F4 BARY0{ 0.5f, 0.25f, 0.25f, 0.0f };   // corner 3 carries NO weight despite invMass 4
    const F4 BARY1{ 0.1f, 0.2f, 0.3f, 0.4f };     // corner 1 carries none because invMass is 0
    const float CONSTR_OFFSET = 0.4f;
    std::vector<uint32_t> con(((NB_ATT + 31u) / 32u) * FAC_STRIDE / 4, 0u);
    auto putF4 = [&](size_t byteOff, F4 v) {
        const size_t w = byteOff / 4;
        con[w] = f2u(v.x); con[w + 1] = f2u(v.y); con[w + 2] = f2u(v.z); con[w + 3] = f2u(v.w);
    };
    auto putU64 = [&](size_t byteOff, uint64_t v) {
        const size_t w = byteOff / 4;
        con[w] = uint32_t(v & 0xFFFFFFFFull); con[w + 1] = uint32_t(v >> 32);
    };
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        putU64(FAC_ELEMID0 + a * 8, encodeSb(0u, a));   // softbody 0, tet a
        putU64(FAC_ELEMID1 + a * 8, encodeSb(1u, a));   // softbody 1, tet a
        putF4(FAC_BARYCENTRIC0 + a * 16, BARY0);
        putF4(FAC_BARYCENTRIC1 + a * 16, BARY1);
        putF4(FAC_LOW_HIGH_ANGLES + a * 16, CONE_STATE[a]);
        // The cone axis comes from projectP - pos0 with this barycentric.
        putF4(FAC_ATT_BARYCENTRIC + a * 16, F4{ 0.0f, 1.0f, 0.0f, 0.0f });
        con[(FAC_CONSTR_OFFSET + a * 4) / 4] = f2u(CONSTR_OFFSET);
    }
    DevBuf bCon = makeDev(dev, con);

    FemAttachSolvePush push = {};
    push.softbodies = bDesc.addr;
    push.attachments = bCon.addr;
    push.numAttachments = NB_ATT;

    Kernel k("softBodyFemAttachBda.spv", "sb_solveOutputSoftBodyAttachmentDeltaVLaunch",
             {}, sizeof(FemAttachSolvePush));
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);
    dev->flush();

    // ---- the CPU reference, transcribed from the .cu -----------------------------------------
    std::vector<F4> ref0(v0.size(), F4{ 0, 0, 0, 0 }), ref1(v1.size(), F4{ 0, 0, 0, 0 });
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const uint32_t base = a * 4u;
        auto interp = [&](const std::vector<F4>& v, F4 b) {
            return F4{ v[base].x * b.x + v[base+1].x * b.y + v[base+2].x * b.z + v[base+3].x * b.w,
                       v[base].y * b.x + v[base+1].y * b.y + v[base+2].y * b.z + v[base+3].y * b.w,
                       v[base].z * b.x + v[base+1].z * b.y + v[base+2].z * b.z + v[base+3].z * b.w,
                       0.0f };
        };
        const F4 pos0 = interp(v0, BARY0), pos1 = interp(v1, BARY1);
        float ex = pos1.x - pos0.x, ey = pos1.y - pos0.y, ez = pos1.z - pos0.z;

        // The cone gate, transcribed rather than special-cased: an OR of three `>= 0`
        // tests (attachments.cuh:36), evaluated per attachment from its own state.
        const F4 lha = CONE_STATE[a];
        const float minDist = lha.x, maxDist = lha.y, maxAngle = lha.z;
        if (maxAngle >= 0.0f || minDist >= 0.0f || maxDist >= 0.0f)
        {
            const F4 proj = interp(v0, F4{ 0.0f, 1.0f, 0.0f, 0.0f });
            float ax = proj.x - pos0.x, ay = proj.y - pos0.y, az = proj.z - pos0.z;
            // PxVec3::getNormalized is a RECIPROCAL and three multiplies, not three
            // divides: `1/sqrt(m)` rounds once and each multiply rounds again, while
            // `ax/am` rounds once per component, and the two disagree in the last bit.
            // This mirror divided until P9.33a, which is part of why the comparison
            // below carried a 2e-5 tolerance and could not have caught the difference.
            const float am2 = ax * ax + ay * ay + az * az;
            if (am2 > 0.0f)
            { const float rs = 1.0f / std::sqrt(am2); ax *= rs; ay *= rs; az *= rs; }
            float rx = ex - ax * CONSTR_OFFSET, ry = ey - ay * CONSTR_OFFSET, rz = ez - az * CONSTR_OFFSET;
            float len = std::sqrt(rx * rx + ry * ry + rz * rz);

            float dirx, diry, dirz;
            if (maxAngle == 0.0f)
            {
                dirx = ax; diry = ay; dirz = az;
            }
            else if (maxAngle > 0.0f)
            {
                dirx = (len > 1e-6f) ? rx / len : ax;
                diry = (len > 1e-6f) ? ry / len : ay;
                dirz = (len > 1e-6f) ? rz / len : az;
                const float cosAngle = dirx * ax + diry * ay + dirz * az;
                const float sinMax = std::sin(maxAngle), cosMax = std::cos(maxAngle);
                if (cosAngle < cosMax)
                {
                    const float t1x = diry * az - dirz * ay;
                    const float t1y = dirz * ax - dirx * az;
                    const float t1z = dirx * ay - diry * ax;
                    float b1x = ay * t1z - az * t1y;
                    float b1y = az * t1x - ax * t1z;
                    float b1z = ax * t1y - ay * t1x;
                    const float bm2 = b1x * b1x + b1y * b1y + b1z * b1z;
                    if (bm2 > 0.0f)
                    { const float rs = 1.0f / std::sqrt(bm2); b1x *= rs; b1y *= rs; b1z *= rs; }
                    else { b1x = b1y = b1z = 0.0f; }
                    dirx = cosMax * ax + sinMax * b1x;
                    diry = cosMax * ay + sinMax * b1y;
                    dirz = cosMax * az + sinMax * b1z;
                }
            }
            else
            {
                dirx = (len > 1e-6f) ? rx / len : ax;
                diry = (len > 1e-6f) ? ry / len : ay;
                dirz = (len > 1e-6f) ? rz / len : az;
            }

            len = std::fmin(std::fmax(len, minDist), maxDist >= 0.0f ? maxDist : PX_MAX_REAL_HOST);
            ex = rx - len * dirx; ey = ry - len * diry; ez = rz - len * dirz;
        }

        const float b0w[4] = { BARY0.x * v0[base].w, BARY0.y * v0[base+1].w,
                               BARY0.z * v0[base+2].w, BARY0.w * v0[base+3].w };
        const float b1w[4] = { BARY1.x * v1[base].w, BARY1.y * v1[base+1].w,
                               BARY1.z * v1[base+2].w, BARY1.w * v1[base+3].w };
        const float bb0[4] = { BARY0.x, BARY0.y, BARY0.z, BARY0.w };
        const float bb1[4] = { BARY1.x, BARY1.y, BARY1.z, BARY1.w };
        // TWO SUMS, THEN ONE ADD -- not one interleaved accumulation.  The CUDA
        // (softBodyGM.cu:2408-2410) computes wTot0 over body 0 and wTot1 over body 1 and
        // adds them once; accumulating `b0w[c]*bb0[c] + b1w[c]*bb1[c]` into a running total
        // visits the same eight products in a different order and lands on a different last
        // bit.  This mirror did that until P9.33a, and the 2e-5 tolerance above hid it.
        float wTot0 = 0.0f, wTot1 = 0.0f;
        for (int c = 0; c < 4; ++c) wTot0 += b0w[c] * bb0[c];
        for (int c = 0; c < 4; ++c) wTot1 += b1w[c] * bb1[c];
        const float wTot = wTot0 + wTot1;
        if (!(wTot > 0.0f)) continue;
        // `error * (1.0f/wTot)` (softBodyGM.cu:2414): a reciprocal and three multiplies,
        // which is not `ex / wTot` three times.
        const float inv = 1.0f / wTot;
        const float dx = ex * inv, dy = ey * inv, dz = ez * inv;
        for (int c = 0; c < 4; ++c)
        {
            if (b1w[c] > 0.0f)
            {
                ref1[base + c].x += -dx * b1w[c]; ref1[base + c].y += -dy * b1w[c];
                ref1[base + c].z += -dz * b1w[c]; ref1[base + c].w += 1.0f;
            }
            if (b0w[c] > 0.0f)
            {
                ref0[base + c].x += dx * b0w[c]; ref0[base + c].y += dy * b0w[c];
                ref0[base + c].z += dz * b0w[c]; ref0[base + c].w += 1.0f;
            }
        }
    }

    const auto g0 = downloadBuffer<F4>(dev, bD0.h, uint32_t(delta0.size()));
    const auto g1 = downloadBuffer<F4>(dev, bD1.h, uint32_t(delta1.size()));
    // BITWISE since P9.33a.  This comparison carried a 2e-5 tolerance, which is what
    // let the shader's `rsqrt` and this mirror's component-wise divide sit side by side
    // for five tranches without anyone learning whether they agreed.  Both are now the
    // reciprocal form the engine computes, so the tolerance has nothing left to hide and
    // is gone; a divergence in the cone branch is a failure again.
    auto nearV = [](F4 g, F4 r) {
        return f2u(g.x) == f2u(r.x) && f2u(g.y) == f2u(r.y) &&
               f2u(g.z) == f2u(r.z) && f2u(g.w) == f2u(r.w);
    };
    size_t mm = 0;
    for (uint32_t c = 0; c < uint32_t(v0.size()); ++c)
    {
        if (!nearV(g0[c], ref0[c]))
        { ++mm; std::printf("  sb0 vert %u MM: got (%.6f %.6f %.6f | %.1f) ref (%.6f %.6f %.6f | %.1f)\n",
                            c, g0[c].x, g0[c].y, g0[c].z, g0[c].w, ref0[c].x, ref0[c].y, ref0[c].z, ref0[c].w); }
        if (!nearV(g1[c], ref1[c]))
        { ++mm; std::printf("  sb1 vert %u MM: got (%.6f %.6f %.6f | %.1f) ref (%.6f %.6f %.6f | %.1f)\n",
                            c, g1[c].x, g1[c].y, g1[c].z, g1[c].w, ref1[c].x, ref1[c].y, ref1[c].z, ref1[c].w); }
    }
    PXG_CHECK_MSG(mm == 0, "sb_solveOutputSoftBodyAttachmentDeltaV (%zu mismatches)", mm);

    // Pin the structural facts the reference alone would not: the two zero-weight corners are
    // untouched for two DIFFERENT reasons, and the fully pinned attachment writes nothing.
    auto isZero = [&](F4 g) { return g.x == 0.0f && g.y == 0.0f && g.z == 0.0f && g.w == 0.0f; };
    PXG_CHECK_MSG(isZero(g0[2]), "sb0 corner 2 has invMass 0 and must be untouched");
    PXG_CHECK_MSG(isZero(g0[3]), "sb0 corner 3 has barycentric 0 and must be untouched even though "
                                 "its invMass is 4 -- only the PRODUCT gates the write");
    PXG_CHECK_MSG(isZero(g1[1]), "sb1 corner 1 has invMass 0 and must be untouched");
    const uint32_t PINNED_ATT = NB_ATT - 1u;      // the last cone state carries the pinned pair
    for (uint32_t c = PINNED_ATT * 4u; c < PINNED_ATT * 4u + 4u; ++c)
        PXG_CHECK_MSG(isZero(g0[c]) && isZero(g1[c]),
                      "attachment %u's tetrahedra are fully pinned (wTot == 0) and must be "
                      "untouched", PINNED_ATT);
    PXG_CHECK_MSG(!isZero(g0[0]) && !isZero(g0[1]) && !isZero(g1[0]) && !isZero(g1[2]) && !isZero(g1[3]),
                  "the weighted corners must all have been written");

    // Every cone state must actually have moved something, or a state is in the table without
    // being exercised -- which is how this rig came to claim it drove the sin/cos arm while
    // reaching only the maxAngle < 0 one.  The four unpinned corners of each of the first
    // NB_ATT-1 attachments are checked as a group, since which corners carry weight is fixed
    // by BARY0/BARY1 above.
    for (uint32_t a = 0; a + 1u < NB_ATT; ++a)
    {
        const uint32_t b = a * 4u;
        PXG_CHECK_MSG(!isZero(g0[b]) || !isZero(g0[b + 1u]),
                      "cone state %u left softbody 0 untouched; it is in the table without "
                      "being observed", a);
        PXG_CHECK_MSG(!isZero(g1[b]) || !isZero(g1[b + 2u]),
                      "cone state %u left softbody 1 untouched", a);
    }
    // States 3 and 4 are the same real cone from opposite sides of `cosAngle < cosMaxAngle`.
    // If they produced the SAME correction the rig is not straddling the test, and a port
    // that deleted the deviation entirely would pass.
    PXG_CHECK_MSG(f2u(g0[12].x) != f2u(g0[16].x) || f2u(g0[12].y) != f2u(g0[16].y) ||
                  f2u(g0[12].z) != f2u(g0[16].z),
                  "cone states 3 (relPos outside the cone) and 4 (inside) produced identical "
                  "corrections, so `cosAngle < cosMaxAngle` is not being straddled");

    std::printf("  att0 sb0=(%.5f %.5f %.5f|%.1f) sb1=(%.5f %.5f %.5f|%.1f)\n",
                g0[0].x, g0[0].y, g0[0].z, g0[0].w, g1[0].x, g1[0].y, g1[0].z, g1[0].w);

    for (PxgBufferHandle h : { bV0.h, bV1.h, bTets.h, bD0.h, bD1.h, bDesc.h, bCon.h })
        dev->destroyBuffer(h);
}


// ---------------------------------------------------------------------------
// P9.33c: sb_solveOutputClothAttachmentDeltaVLaunch (softBodyGM.cu:2424, launch
// PxgSoftBodyCore.cpp:2005) -- a softbody TETRAHEDRON pinned to a cloth TRIANGLE, the third
// and last consumer of PxgFEMFEMAttachmentConstraint.
// ---------------------------------------------------------------------------
namespace p933att
{
struct ClothSbAttachSolvePush
{
    uint64_t softbodies, clothes, attachments;
    uint32_t numAttachments, pad28;
};
static_assert(sizeof(ClothSbAttachSolvePush) == 32, "cloth-sb attach solve blob must be 32 bytes");
static_assert(offsetof(ClothSbAttachSolvePush, clothes) == 8, "clothes @8");
static_assert(offsetof(ClothSbAttachSolvePush, attachments) == 16, "attachments @16");
static_assert(offsetof(ClothSbAttachSolvePush, numAttachments) == 24, "numAttachments @24");
// The sibling's blob is 24 bytes with attachments @8.  Stated as a compile-time fact so the
// two cannot quietly become interchangeable.
static_assert(sizeof(ClothSbAttachSolvePush) != sizeof(p819att::FemAttachSolvePush),
              "the three- and four-argument attachment solves must not share a blob");

static const uint32_t CLOTH_ID_SHIFT = 20u;
static const uint32_t CLOTH_ELEM_MASK = 0x000fffffu;
static uint32_t encodeCloth(uint32_t clothId, uint32_t elem)
{
    return (clothId << CLOTH_ID_SHIFT) | elem;
}
} // namespace p933att

PXG_TEST(softBodySimClothAttachmentSolve)
{
    using namespace p819att;
    using namespace p933att;
    using p80refit::TetIdx4;
    PxgSlangComputeDevice* dev = testDevice();

    // Softbody id 2 and cloth id 3 -- never 0 on either side, so a port that forgets a shift
    // lands on a zeroed descriptor rather than on a plausible one.
    const uint32_t SB_ID = 2u, CLOTH_ID = 3u;

    // NINE attachments, each with its own tetrahedron and its own triangle, all vertices
    // disjoint, so every expected delta can be read on its own.  The table below is what the
    // first sweep forced: with six attachments and one fixed barycentric, four mutants were
    // no-ops against the rig rather than against the code.
    //
    //   idx  cone state        cloth weights
    //    0   engine default    corner 0 has inverse mass 0
    //    1   maxDist-enabled   corner 1 has inverse mass 0
    //    2   DISABLED          corner 2 has inverse mass 0
    //    3   real cone, out    corner 0 has barycentric 0
    //    4   real cone, in     corner 1 has barycentric 0
    //    5   engine default    corner 2 has barycentric 0
    //    6   real cone         DEGENERATE: the cloth anchor coincides with the softbody one
    //                          and the constraint offset is 0, so relPos is exactly zero and
    //                          `(len > 1e-6f) ? relPos/len : coneAxis` takes its GUARDED arm.
    //                          Dropping the guard is a 0/0 that only shows up here.
    //    7   engine default    DEGENERATE again -- the guard is written TWICE, once in the
    //                          maxAngle > 0 arm and once in the else arm, and the first sweep
    //                          reached only one of them.  One degenerate attachment per arm.
    //    8   engine default    all three corners weighted (the positive control)
    //    9   engine default    fully pinned on both sides -> wTot == 0
    static const F4 CONE_STATE[10] = {
        F4{ 0.0f, 0.0f, -1.0f, 0.0f },     // engine default: enabled via minDist
        F4{ -1.0f, 0.5f, -1.0f, 0.0f },    // enabled via maxDist
        F4{ -1.0f, -1.0f, -1.0f, 0.0f },   // disabled -- the only state reaching the else arm
        F4{ -1.0f, -1.0f, 0.30f, 0.0f },   // real cone, relPos outside -> the sin/cos arm
        F4{ -1.0f, -1.0f, 3.00f, 0.0f },   // real cone, relPos inside  -> deviation skipped
        F4{ 0.0f, 0.0f, -1.0f, 0.0f },     // 5: zero-barycentric corner 2
        F4{ -1.0f, -1.0f, 0.30f, 0.0f },   // 6: real cone AND degenerate -> the angle arm's guard
        F4{ 0.0f, 0.0f, -1.0f, 0.0f },     // 7: engine default AND degenerate -> the else arm's
        F4{ 0.0f, 0.0f, -1.0f, 0.0f },     // 8: the all-weighted positive control
        F4{ 0.0f, 0.0f, -1.0f, 0.0f },     // 9: fully pinned
    };
    static const float PX_MAX_REAL_HOST = 3.4028235e38f;
    const uint32_t NB_ATT = 10u;
    const uint32_t PINNED_ATT = 9u;
    auto isDegenerate = [](uint32_t a) { return a == 6u || a == 7u; };
    auto zeroMassCorner = [](uint32_t a) -> int { return (a < 3u) ? int(a) : -1; };
    auto zeroBaryCorner = [](uint32_t a) -> int { return (a >= 3u && a < 6u) ? int(a - 3u) : -1; };

    // The softbody barycentric is fixed; tet corner 1 always carries inverse mass 0.
    const F4 BARY_SB{ 0.1f, 0.2f, 0.3f, 0.4f };
    // `barycentric0.w` is a JUNK NON-ZERO value on every attachment.  The engine always writes
    // 0 there (PxgSimulationController.cpp:3840, copied whole by the prepare at
    // softBodyGM.cu:2073), and the CUDA is immune to it anyway -- b0w.w is a written literal
    // 0.0f, wTot0 sums three terms and updateTriPositionDelta never reads .w.  The rig writes
    // junk precisely BECAUSE the code is supposed to be immune: a port that restores the
    // fourth term is not, and with a zero there nothing could tell them apart.
    const float BARY0_W_JUNK = 0.37f;

    std::vector<F4> sbVerts, clothVerts;
    std::vector<TetIdx4> tets, tris;
    std::vector<F4> baryCloth(NB_ATT);
    std::vector<float> constrOffset(NB_ATT, 0.4f);
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const bool pinned = (a == PINNED_ATT);
        const bool degenerate = isDegenerate(a);
        const float ox = 10.0f * float(a);
        const uint32_t sb = uint32_t(sbVerts.size());
        const float sbM[4] = { pinned ? 0.0f : 0.5f, 0.0f,
                               pinned ? 0.0f : 1.5f, pinned ? 0.0f : 2.0f };
        sbVerts.push_back(F4{ ox,        2.0f, 0.0f, sbM[0] });
        sbVerts.push_back(F4{ ox + 1.0f, 2.0f, 0.0f, sbM[1] });
        sbVerts.push_back(F4{ ox,        3.0f, 0.0f, sbM[2] });
        sbVerts.push_back(F4{ ox,        2.0f, 1.0f, sbM[3] });
        tets.push_back(TetIdx4{ sb, sb + 1u, sb + 2u, sb + 3u });

        // The cloth barycentric: one corner zeroed on attachments 3..5, and the degenerate
        // one puts all the weight on corner 0 so its anchor can be placed EXACTLY.
        F4 bc{ 0.55f, 0.45f, 0.0f, BARY0_W_JUNK };
        const int zb = zeroBaryCorner(a);
        if (zb == 0) bc = F4{ 0.0f, 0.6f, 0.4f, BARY0_W_JUNK };
        else if (zb == 1) bc = F4{ 0.6f, 0.0f, 0.4f, BARY0_W_JUNK };
        else if (zb == 2) bc = F4{ 0.55f, 0.45f, 0.0f, BARY0_W_JUNK };
        else if (a >= 6u) bc = F4{ 0.3f, 0.45f, 0.25f, BARY0_W_JUNK };   // all three weighted
        // (the degenerate pair overrides this two lines down)
        if (degenerate) { bc = F4{ 1.0f, 0.0f, 0.0f, BARY0_W_JUNK }; constrOffset[a] = 0.0f; }
        baryCloth[a] = bc;

        const uint32_t cv = uint32_t(clothVerts.size());
        float cM[3] = { pinned ? 0.0f : 1.0f, pinned ? 0.0f : 3.0f, pinned ? 0.0f : 4.0f };
        const int zm = zeroMassCorner(a);
        if (zm >= 0) cM[zm] = 0.0f;
        if (degenerate)
        {
            // pos0 = 1.0 * c0 exactly, so placing c0 on the softbody anchor makes relPos1
            // exactly zero.  pos1 = sum sbVerts[sb+k] * BARY_SB[k], computed here.
            const float px = ox * (BARY_SB.x + BARY_SB.z + BARY_SB.w) + (ox + 1.0f) * BARY_SB.y;
            const float py = 2.0f * BARY_SB.x + 2.0f * BARY_SB.y + 3.0f * BARY_SB.z + 2.0f * BARY_SB.w;
            const float pz = 1.0f * BARY_SB.w;
            clothVerts.push_back(F4{ px, py, pz, cM[0] });
        }
        else
        {
            clothVerts.push_back(F4{ ox, 0.0f, 0.0f, cM[0] });
        }
        clothVerts.push_back(F4{ ox + 1.3f, 0.2f, 0.0f, cM[1] });
        clothVerts.push_back(F4{ ox + 0.4f, 1.1f, 0.0f, cM[2] });
        tris.push_back(TetIdx4{ cv, cv + 1u, cv + 2u, 0u });   // .w patched below
    }
    // A WITNESS vertex, weighted and in bounds, sitting in every triangle's unused .w slot.
    // The engine puts 0 there (PxgFEMCloth.cpp:152); a port that kept the tetrahedron's
    // four-corner scatter would write this vertex, and the check below says it must not.
    const uint32_t WITNESS = uint32_t(clothVerts.size());
    clothVerts.push_back(F4{ -100.0f, -100.0f, -100.0f, 9.0f });
    for (uint32_t a = 0; a < NB_ATT; ++a)
        tris[a].w = WITNESS;

    DevBuf bSbV = makeDev(dev, sbVerts);
    if (!bSbV.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bSbV.h); return; }
    DevBuf bClV = makeDev(dev, clothVerts), bTets = makeDev(dev, tets), bTris = makeDev(dev, tris);
    std::vector<F4> sbDelta(sbVerts.size(), F4{ 0, 0, 0, 0 });
    std::vector<F4> clDelta(clothVerts.size(), F4{ 0, 0, 0, 0 });
    // The slot AFTER mDeltaPos in the cloth descriptor, seeded with a sentinel: if the port
    // writes mAccumulatedDeltaPos instead, this buffer moves and mDeltaPos stays at zero.
    std::vector<F4> clAccum(clothVerts.size(), F4{ -7.0f, -7.0f, -7.0f, -7.0f });
    DevBuf bSbD = makeDev(dev, sbDelta), bClD = makeDev(dev, clDelta), bClA = makeDev(dev, clAccum);

    std::vector<PxgSoftBody> sbArr(SB_ID + 1u, makeZeroDesc());
    sbArr[SB_ID].mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSbV.addr));
    sbArr[SB_ID].mSimTetIndices = reinterpret_cast<uint4*>(size_t(bTets.addr));
    sbArr[SB_ID].mSimDelta = reinterpret_cast<float4*>(size_t(bSbD.addr));
    DevBuf bSbDesc = makeDev(dev, sbArr);

    std::vector<PxgFEMCloth> clArr(CLOTH_ID + 1u);
    std::memset(clArr.data(), 0, clArr.size() * sizeof(PxgFEMCloth));
    clArr[CLOTH_ID].mPosition_InvMass = reinterpret_cast<float4*>(size_t(bClV.addr));
    clArr[CLOTH_ID].mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bTris.addr));
    clArr[CLOTH_ID].mDeltaPos = reinterpret_cast<float4*>(size_t(bClD.addr));
    clArr[CLOTH_ID].mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bClA.addr));
    clArr[CLOTH_ID].mNbVerts = uint32_t(clothVerts.size());
    clArr[CLOTH_ID].mNbTriangles = NB_ATT;
    DevBuf bClDesc = makeDev(dev, clArr);

    // ---- the constraint block ---------------------------------------------------------
    std::vector<uint32_t> con(((NB_ATT + 31u) / 32u) * FAC_STRIDE / 4, 0u);
    auto putF4 = [&](size_t byteOff, F4 v) {
        const size_t w = byteOff / 4;
        con[w] = f2u(v.x); con[w + 1] = f2u(v.y); con[w + 2] = f2u(v.z); con[w + 3] = f2u(v.w);
    };
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        // Both slots are PxU64 read as PxU32.  The upper words carry a pattern so a uint2
        // read produces an id far outside either descriptor array.
        const size_t w0 = (FAC_ELEMID0 + a * 8) / 4;
        con[w0] = encodeCloth(CLOTH_ID, a);          // elemId0 is the CLOTH
        con[w0 + 1] = 0xC1C10000u | a;
        const size_t w1 = (FAC_ELEMID1 + a * 8) / 4;
        con[w1] = encodeSb(SB_ID, a);                // elemId1 is the SOFTBODY
        con[w1 + 1] = 0x5B5B0000u | a;
        putF4(FAC_BARYCENTRIC0 + a * 16, baryCloth[a]);
        putF4(FAC_BARYCENTRIC1 + a * 16, BARY_SB);
        putF4(FAC_LOW_HIGH_ANGLES + a * 16, CONE_STATE[a]);
        // The cone axis comes from `pos1 - projectP` over the SOFTBODY's four corners.
        putF4(FAC_ATT_BARYCENTRIC + a * 16, F4{ 0.0f, 1.0f, 0.0f, 0.0f });
        con[(FAC_CONSTR_OFFSET + a * 4) / 4] = f2u(constrOffset[a]);
    }
    DevBuf bCon = makeDev(dev, con);

    ClothSbAttachSolvePush push = {};
    push.softbodies = bSbDesc.addr;
    push.clothes = bClDesc.addr;
    push.attachments = bCon.addr;
    push.numAttachments = NB_ATT;
    Kernel k("softBodyClothAttachBda.spv", "sb_solveOutputClothAttachmentDeltaVLaunch",
             {}, sizeof(ClothSbAttachSolvePush));
    k.dispatch(1024u, {}, push, 1u, 1u);
    dev->flush();

    // ---- the mirror, transcribed from softBodyGM.cu:2424 --------------------------------
    std::vector<F4> refSb(sbVerts.size(), F4{ 0, 0, 0, 0 });
    std::vector<F4> refCl(clothVerts.size(), F4{ 0, 0, 0, 0 });
    uint32_t skippedByMass = 0u, coneBranchTaken = 0u, sincosTaken = 0u, epsGuardTaken = 0u;
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const uint32_t sb = a * 4u, cv = a * 3u;
        const F4 BARY_CLOTH = baryCloth[a];
        const float OFFSET = constrOffset[a];
        // Four terms on the softbody side, THREE on the cloth side.
        auto blend4 = [&](const std::vector<F4>& v, uint32_t b, F4 w) {
            return F4{ v[b].x * w.x + v[b+1].x * w.y + v[b+2].x * w.z + v[b+3].x * w.w,
                       v[b].y * w.x + v[b+1].y * w.y + v[b+2].y * w.z + v[b+3].y * w.w,
                       v[b].z * w.x + v[b+1].z * w.y + v[b+2].z * w.z + v[b+3].z * w.w, 0.0f };
        };
        auto blend3 = [&](const std::vector<F4>& v, uint32_t b, F4 w) {
            return F4{ v[b].x * w.x + v[b+1].x * w.y + v[b+2].x * w.z,
                       v[b].y * w.x + v[b+1].y * w.y + v[b+2].y * w.z,
                       v[b].z * w.x + v[b+1].z * w.y + v[b+2].z * w.z, 0.0f };
        };
        const F4 pos1 = blend4(sbVerts, sb, BARY_SB);
        const F4 pos0 = blend3(clothVerts, cv, BARY_CLOTH);
        float ex = pos1.x - pos0.x, ey = pos1.y - pos0.y, ez = pos1.z - pos0.z;

        const F4 lha = CONE_STATE[a];
        const float minDist = lha.x, maxDist = lha.y, maxAngle = lha.z;
        if (maxAngle >= 0.0f || minDist >= 0.0f || maxDist >= 0.0f)
        {
            ++coneBranchTaken;
            // D4: projectP over the SOFTBODY's corners, and `pos1 - projectP` -- the sibling
            // is `projectP - pos0` over the other body's corners, i.e. the negation.
            const F4 proj = blend4(sbVerts, sb, F4{ 0.0f, 1.0f, 0.0f, 0.0f });
            float ax = pos1.x - proj.x, ay = pos1.y - proj.y, az = pos1.z - proj.z;
            const float am2 = ax * ax + ay * ay + az * az;
            if (am2 > 0.0f)
            { const float rs = 1.0f / std::sqrt(am2); ax *= rs; ay *= rs; az *= rs; }
            else { ax = ay = az = 0.0f; }
            const float rx = ex - ax * OFFSET, ry = ey - ay * OFFSET, rz = ez - az * OFFSET;
            float len = std::sqrt(rx * rx + ry * ry + rz * rz);
            if (!(len > 1e-6f)) ++epsGuardTaken;

            float dirx, diry, dirz;
            if (maxAngle == 0.0f) { dirx = ax; diry = ay; dirz = az; }
            else if (maxAngle > 0.0f)
            {
                dirx = (len > 1e-6f) ? rx / len : ax;
                diry = (len > 1e-6f) ? ry / len : ay;
                dirz = (len > 1e-6f) ? rz / len : az;
                const float cosAngle = dirx * ax + diry * ay + dirz * az;
                const float sinMax = std::sin(maxAngle), cosMax = std::cos(maxAngle);
                if (cosAngle < cosMax)
                {
                    ++sincosTaken;
                    const float t1x = diry * az - dirz * ay;
                    const float t1y = dirz * ax - dirx * az;
                    const float t1z = dirx * ay - diry * ax;
                    float b1x = ay * t1z - az * t1y;
                    float b1y = az * t1x - ax * t1z;
                    float b1z = ax * t1y - ay * t1x;
                    const float bm2 = b1x * b1x + b1y * b1y + b1z * b1z;
                    if (bm2 > 0.0f)
                    { const float rs = 1.0f / std::sqrt(bm2); b1x *= rs; b1y *= rs; b1z *= rs; }
                    else { b1x = b1y = b1z = 0.0f; }
                    dirx = cosMax * ax + sinMax * b1x;
                    diry = cosMax * ay + sinMax * b1y;
                    dirz = cosMax * az + sinMax * b1z;
                }
            }
            else
            {
                dirx = (len > 1e-6f) ? rx / len : ax;
                diry = (len > 1e-6f) ? ry / len : ay;
                dirz = (len > 1e-6f) ? rz / len : az;
            }
            len = std::fmin(std::fmax(len, minDist), maxDist >= 0.0f ? maxDist : PX_MAX_REAL_HOST);
            ex = rx - len * dirx; ey = ry - len * diry; ez = rz - len * dirz;
        }

        // D2: b0w.w is a literal zero and wTot0 has THREE terms.
        const float b0w[4] = { BARY_CLOTH.x * clothVerts[cv].w, BARY_CLOTH.y * clothVerts[cv+1].w,
                               BARY_CLOTH.z * clothVerts[cv+2].w, 0.0f };
        const float b1w[4] = { BARY_SB.x * sbVerts[sb].w, BARY_SB.y * sbVerts[sb+1].w,
                               BARY_SB.z * sbVerts[sb+2].w, BARY_SB.w * sbVerts[sb+3].w };
        const float bb0[4] = { BARY_CLOTH.x, BARY_CLOTH.y, BARY_CLOTH.z, BARY_CLOTH.w };
        const float bb1[4] = { BARY_SB.x, BARY_SB.y, BARY_SB.z, BARY_SB.w };
        float wTot0 = 0.0f, wTot1 = 0.0f;
        for (int c = 0; c < 3; ++c) wTot0 += b0w[c] * bb0[c];     // THREE
        for (int c = 0; c < 4; ++c) wTot1 += b1w[c] * bb1[c];     // four
        const float wTot = wTot0 + wTot1;
        if (!(wTot > 0.0f)) { ++skippedByMass; continue; }

        const float inv = 1.0f / wTot;
        const float dx = ex * inv, dy = ey * inv, dz = ez * inv;
        for (int c = 0; c < 4; ++c)
            if (b1w[c] > 0.0f)
            {
                refSb[sb + c].x += -dx * b1w[c]; refSb[sb + c].y += -dy * b1w[c];
                refSb[sb + c].z += -dz * b1w[c]; refSb[sb + c].w += 1.0f;
            }
        for (int c = 0; c < 3; ++c)               // D3: three corners, each skippable
            if (b0w[c] > 0.0f)
            {
                refCl[cv + c].x += dx * b0w[c]; refCl[cv + c].y += dy * b0w[c];
                refCl[cv + c].z += dz * b0w[c]; refCl[cv + c].w += 1.0f;
            }
    }

    const auto gSb = downloadBuffer<F4>(dev, bSbD.h, uint32_t(sbVerts.size()));
    const auto gCl = downloadBuffer<F4>(dev, bClD.h, uint32_t(clothVerts.size()));
    const auto gAcc = downloadBuffer<F4>(dev, bClA.h, uint32_t(clothVerts.size()));

    // BITWISE.  Every vertex has exactly one writer, and P9.33b removed the tolerance that
    // hid three association defects in the sibling's mirror.
    uint32_t mm = 0u;
    auto cmp = [&](const std::vector<F4>& got, const std::vector<F4>& ref, const char* tag) {
        for (uint32_t v = 0; v < uint32_t(ref.size()); ++v)
            if (f2u(got[v].x) != f2u(ref[v].x) || f2u(got[v].y) != f2u(ref[v].y) ||
                f2u(got[v].z) != f2u(ref[v].z) || f2u(got[v].w) != f2u(ref[v].w))
            {
                if (mm == 0u)
                    std::printf("    first %s difference at vertex %u: gpu (%.9g %.9g %.9g | %.9g)"
                                " vs (%.9g %.9g %.9g | %.9g)\n", tag, v,
                                got[v].x, got[v].y, got[v].z, got[v].w,
                                ref[v].x, ref[v].y, ref[v].z, ref[v].w);
                ++mm;
            }
    };
    cmp(gSb, refSb, "softbody");
    cmp(gCl, refCl, "cloth");
    PXG_CHECK_MSG(mm == 0u, "%u vertex deltas differ from the mirror", mm);

    // D1: mDeltaPos, not the slot next to it.
    uint32_t accTouched = 0u;
    for (uint32_t v = 0; v < uint32_t(clothVerts.size()); ++v)
        if (gAcc[v].x != -7.0f || gAcc[v].y != -7.0f || gAcc[v].z != -7.0f || gAcc[v].w != -7.0f)
            ++accTouched;
    PXG_CHECK_MSG(accTouched == 0u,
                  "%u words of mAccumulatedDeltaPos moved; the cloth output is mDeltaPos, which "
                  "sits one pointer slot earlier in the descriptor", accTouched);

    auto isZero = [&](F4 g) { return g.x == 0.0f && g.y == 0.0f && g.z == 0.0f && g.w == 0.0f; };
    // D2/D3: the triangle has THREE corners, and the fourth index slot is not one of them.
    PXG_CHECK_MSG(isZero(gCl[WITNESS]),
                  "the witness vertex sits in every triangle's unused .w index slot and must "
                  "never be written; a port that kept the tetrahedron's four-corner scatter "
                  "would have moved it (got %g %g %g | %g)",
                  gCl[WITNESS].x, gCl[WITNESS].y, gCl[WITNESS].z, gCl[WITNESS].w);

    // The structural facts the mirror shares the port's reading of, so it cannot pin them.
    PXG_CHECK_MSG(skippedByMass == 1u,
                  "exactly one attachment must be skipped for wTot == 0, not %u", skippedByMass);
    PXG_CHECK_MSG(coneBranchTaken == NB_ATT - 1u,
                  "the cone gate must fire on every state but the disabled one (%u of %u)",
                  coneBranchTaken, NB_ATT);
    PXG_CHECK_MSG(sincosTaken == 1u,
                  "exactly one cone state must reach the sin/cos arm; %u did.  Two real cones "
                  "are seeded on purpose, one outside and one inside, so that "
                  "`cosAngle < cosMaxAngle` is straddled rather than only ever true",
                  sincosTaken);
    PXG_CHECK_MSG(epsGuardTaken == 2u,
                  "exactly two attachments must reach the `len > 1e-6f` guard's FALSE arm -- one "
                  "in the maxAngle > 0 arm and one in the else arm, because the guard is written "
                  "twice; %u did.  Without both, dropping one of them is a division that never "
                  "divides by zero and the mutant survives", epsGuardTaken);

    // D3's two skip reasons, in EACH of the three cloth corner slots.  The first sweep put
    // its zeros in corners 1 and 2 only, so the mutant that drops corner 0's guard was a
    // no-op against the rig rather than against the code.
    uint32_t zeroWeightChecked = 0u;
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        if (a == PINNED_ATT) continue;
        const int slots[2] = { zeroMassCorner(a), zeroBaryCorner(a) };
        for (int which = 0; which < 2; ++which)
        {
            const int c = slots[which];
            if (c < 0) continue;
            PXG_CHECK_MSG(isZero(gCl[a * 3u + uint32_t(c)]),
                          "cloth corner %d of attachment %u carries zero weight (%s) and must "
                          "take no write at all -- not even the .w counter", c, a,
                          which == 0 ? "inverse mass" : "barycentric");
            ++zeroWeightChecked;
        }
    }
    PXG_CHECK_MSG(zeroWeightChecked == 6u,
                  "the rig must place a zero cloth weight in each of the three corner slots for "
                  "both reasons; %u were found", zeroWeightChecked);
    // The positive control: attachment 8 weights all three corners, so all three must move.
    for (uint32_t c = 0; c < 3u; ++c)
        PXG_CHECK_MSG(!isZero(gCl[8u * 3u + c]),
                      "attachment 8 weights all three cloth corners; corner %u was not written", c);
    for (uint32_t a = 0; a + 1u < NB_ATT; ++a)
        PXG_CHECK_MSG(isZero(gSb[a * 4u + 1u]),
                      "softbody corner 1 of attachment %u has inverse mass 0", a);
    for (uint32_t c = PINNED_ATT * 4u; c < PINNED_ATT * 4u + 4u; ++c)
        PXG_CHECK_MSG(isZero(gSb[c]), "the fully pinned attachment must leave the softbody alone");
    for (uint32_t c = PINNED_ATT * 3u; c < PINNED_ATT * 3u + 3u; ++c)
        PXG_CHECK_MSG(isZero(gCl[c]), "the fully pinned attachment must leave the cloth alone");

    // States 3 and 4 straddle `cosAngle < cosMaxAngle`; identical corrections would mean the
    // rig is on one side of the test and a port that dropped it would pass.
    PXG_CHECK_MSG(f2u(gCl[3u * 3u + 1u].x) != f2u(gCl[4u * 3u + 0u].x) ||
                  f2u(gCl[3u * 3u + 1u].y) != f2u(gCl[4u * 3u + 0u].y),
                  "cone states 3 (outside) and 4 (inside) produced identical corrections");

    std::printf("    [ note ] %u attachments (%u through the cone gate, %u reaching sin/cos, "
                "%u on the near-zero guard, %u skipped for wTot == 0); %u zero-weight cloth "
                "corners checked, %zu vertex deltas bit-exact, witness and "
                "mAccumulatedDeltaPos untouched\n",
                NB_ATT, coneBranchTaken, sincosTaken, epsGuardTaken, skippedByMass,
                zeroWeightChecked, sbVerts.size() + clothVerts.size());

    for (PxgBufferHandle h : { bSbV.h, bClV.h, bTets.h, bTris.h, bSbD.h, bClD.h, bClA.h,
                               bSbDesc.h, bClDesc.h, bCon.h })
        dev->destroyBuffer(h);
}


// ---------------------------------------------------------------------------
// P9.33c coverage rig.  Dropping the `nbIterationsPerBlock *` factor from the work index
// makes the per-block ranges OVERLAP, so the tail is never visited and everything else is
// visited twice -- but only past gridDim*blockDim = 1024*256 = 262144 attachments, below
// which nbIterationsPerBlock is 1 and the formula collapses to a flat index.
//
// The small fixture above cannot see that, so this one runs 262400 attachments.  It does NOT
// give each its own vertices: 262400 tetrahedra and triangles would be ~90 MB of buffers to
// answer a question about visit counts.  Instead the attachments cycle over 1024 elements, and
// the `.w` lane of each vertex is an exact integer count of how many attachments reached it --
// 257 for the first 256 elements and 256 for the rest, which is a number the rig derives
// rather than assumes.  A dropped factor changes both the totals and their distribution.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimClothAttachmentCoverage)
{
    using namespace p819att;
    using namespace p933att;
    using p80refit::TetIdx4;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t SB_ID = 1u, CLOTH_ID = 1u;
    const uint32_t GRID = 1024u, BLOCK = 256u;
    const uint32_t NB_ATT = 262400u;          // 1025 blocks required over a 1024-wide grid
    const uint32_t NB_ELEM = 1024u;           // the element ring the attachments cycle over
    // The ring sits ABOVE 65536 so the ELEMENT MASK's width is observable.  The engine's mask
    // is 0x000fffff (20 bits); a port that narrowed it to 0x0000ffff would fold index
    // ELEM_BASE+e down to ELEM_BASE+e-65536, which is a different, still in-bounds element --
    // so the counts land on the wrong tetrahedra rather than faulting, and only a rig that
    // crosses the boundary can tell.  Indices below it were the last gap the first sweep left.
    const uint32_t ELEM_BASE = 66000u;
    const uint32_t NB_SLOTS = ELEM_BASE + NB_ELEM;   // index arrays only; vertices stay small

    const uint32_t nbBlocksRequired = (NB_ATT + BLOCK - 1u) / BLOCK;
    const uint32_t nbIterationsPerBlock = (nbBlocksRequired + GRID - 1u) / GRID;
    PXG_CHECK_MSG(nbIterationsPerBlock == 2u,
                  "the rig must reach two iterations per thread, got %u", nbIterationsPerBlock);

    // The index arrays span the whole slot range; only the last NB_ELEM entries are ever
    // referenced, and the rest point at vertex 0 so a folded index is in bounds and WRONG
    // rather than out of bounds and fatal.
    std::vector<F4> sbVerts, clothVerts;
    std::vector<TetIdx4> tets(NB_SLOTS, TetIdx4{ 0u, 0u, 0u, 0u });
    std::vector<TetIdx4> tris(NB_SLOTS, TetIdx4{ 0u, 0u, 0u, 0u });
    for (uint32_t e = 0; e < NB_ELEM; ++e)
    {
        const float ox = 4.0f * float(e);
        const uint32_t sb = uint32_t(sbVerts.size());
        sbVerts.push_back(F4{ ox,        2.0f, 0.0f, 0.5f });
        sbVerts.push_back(F4{ ox + 1.0f, 2.0f, 0.0f, 0.5f });
        sbVerts.push_back(F4{ ox,        3.0f, 0.0f, 0.5f });
        sbVerts.push_back(F4{ ox,        2.0f, 1.0f, 0.5f });
        tets[ELEM_BASE + e] = TetIdx4{ sb, sb + 1u, sb + 2u, sb + 3u };
        const uint32_t cv = uint32_t(clothVerts.size());
        clothVerts.push_back(F4{ ox,        0.0f, 0.0f, 1.0f });
        clothVerts.push_back(F4{ ox + 1.0f, 0.0f, 0.0f, 1.0f });
        clothVerts.push_back(F4{ ox,        1.0f, 0.0f, 1.0f });
        tris[ELEM_BASE + e] = TetIdx4{ cv, cv + 1u, cv + 2u, 0u };
    }

    DevBuf bSbV = makeDev(dev, sbVerts);
    if (!bSbV.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bSbV.h); return; }
    DevBuf bClV = makeDev(dev, clothVerts), bTets = makeDev(dev, tets), bTris = makeDev(dev, tris);
    std::vector<F4> sbDelta(sbVerts.size(), F4{ 0, 0, 0, 0 });
    std::vector<F4> clDelta(clothVerts.size(), F4{ 0, 0, 0, 0 });
    DevBuf bSbD = makeDev(dev, sbDelta), bClD = makeDev(dev, clDelta);

    std::vector<PxgSoftBody> sbArr(SB_ID + 1u, makeZeroDesc());
    sbArr[SB_ID].mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSbV.addr));
    sbArr[SB_ID].mSimTetIndices = reinterpret_cast<uint4*>(size_t(bTets.addr));
    sbArr[SB_ID].mSimDelta = reinterpret_cast<float4*>(size_t(bSbD.addr));
    DevBuf bSbDesc = makeDev(dev, sbArr);

    std::vector<PxgFEMCloth> clArr(CLOTH_ID + 1u);
    std::memset(clArr.data(), 0, clArr.size() * sizeof(PxgFEMCloth));
    clArr[CLOTH_ID].mPosition_InvMass = reinterpret_cast<float4*>(size_t(bClV.addr));
    clArr[CLOTH_ID].mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bTris.addr));
    clArr[CLOTH_ID].mDeltaPos = reinterpret_cast<float4*>(size_t(bClD.addr));
    clArr[CLOTH_ID].mNbVerts = uint32_t(clothVerts.size());
    clArr[CLOTH_ID].mNbTriangles = NB_SLOTS;
    DevBuf bClDesc = makeDev(dev, clArr);

    const uint32_t NB_BLOCKS = (NB_ATT + 31u) / 32u;
    std::vector<uint32_t> con(size_t(NB_BLOCKS) * FAC_STRIDE / 4, 0u);
    const F4 BARY_SB{ 0.25f, 0.25f, 0.25f, 0.25f };
    const F4 BARY_CLOTH{ 0.25f, 0.35f, 0.40f, 0.0f };
    std::vector<uint32_t> expect(NB_ELEM, 0u);
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const uint32_t e = a % NB_ELEM;
        ++expect[e];
        const size_t base = size_t(a / 32u) * FAC_STRIDE, off = a & 31u;
        con[(base + FAC_ELEMID0 + off * 8) / 4] = encodeCloth(CLOTH_ID, ELEM_BASE + e);
        con[(base + FAC_ELEMID1 + off * 8) / 4] = encodeSb(SB_ID, ELEM_BASE + e);
        const size_t b0 = (base + FAC_BARYCENTRIC0 + off * 16) / 4;
        con[b0] = f2u(BARY_CLOTH.x); con[b0+1] = f2u(BARY_CLOTH.y); con[b0+2] = f2u(BARY_CLOTH.z);
        const size_t b1 = (base + FAC_BARYCENTRIC1 + off * 16) / 4;
        con[b1] = f2u(BARY_SB.x); con[b1+1] = f2u(BARY_SB.y);
        con[b1+2] = f2u(BARY_SB.z); con[b1+3] = f2u(BARY_SB.w);
        // The DISABLED cone state, so this rig measures the work index and nothing else.
        const size_t lh = (base + FAC_LOW_HIGH_ANGLES + off * 16) / 4;
        con[lh] = f2u(-1.0f); con[lh+1] = f2u(-1.0f); con[lh+2] = f2u(-1.0f);
    }
    DevBuf bCon = makeDev(dev, con);

    ClothSbAttachSolvePush push = {};
    push.softbodies = bSbDesc.addr;
    push.clothes = bClDesc.addr;
    push.attachments = bCon.addr;
    push.numAttachments = NB_ATT;
    Kernel k("softBodyClothAttachBda.spv", "sb_solveOutputClothAttachmentDeltaVLaunch",
             {}, sizeof(ClothSbAttachSolvePush));
    k.dispatch(GRID, {}, push, 1u, 1u);
    dev->flush();

    const auto gSb = downloadBuffer<F4>(dev, bSbD.h, uint32_t(sbVerts.size()));
    const auto gCl = downloadBuffer<F4>(dev, bClD.h, uint32_t(clothVerts.size()));

    // Every corner of element e must carry exactly `expect[e]` writers -- all four tet corners
    // and all three triangle corners, since every weight here is strictly positive.
    uint32_t wrong = 0u, unvisited = 0u, firstWrong = NB_ELEM;
    for (uint32_t e = 0; e < NB_ELEM; ++e)
    {
        const float want = float(expect[e]);
        for (uint32_t c = 0; c < 4u; ++c)
            if (gSb[e * 4u + c].w != want)
            {
                if (gSb[e * 4u + c].w == 0.0f) ++unvisited; else ++wrong;
                if (firstWrong == NB_ELEM) firstWrong = e;
            }
        for (uint32_t c = 0; c < 3u; ++c)
            if (gCl[e * 3u + c].w != want)
            {
                if (gCl[e * 3u + c].w == 0.0f) ++unvisited; else ++wrong;
                if (firstWrong == NB_ELEM) firstWrong = e;
            }
    }
    PXG_CHECK_MSG(unvisited == 0u,
                  "%u corners were never written (first at element %u); with the strided formula "
                  "every one of the %u attachments is visited exactly once",
                  unvisited, firstWrong, NB_ATT);
    PXG_CHECK_MSG(wrong == 0u,
                  "%u corners carry a writer count other than the derived one (first at element "
                  "%u, expected %u, got %g); a count above it means a work index was visited "
                  "more than once", wrong, firstWrong,
                  firstWrong < NB_ELEM ? expect[firstWrong] : 0u,
                  firstWrong < NB_ELEM ? gSb[firstWrong * 4u].w : 0.0f);

    std::printf("    [ note ] %u attachments over %u elements at indices %u..%u (past the "
                "16-bit boundary on purpose), %u iterations per thread; writer counts exactly "
                "%u on the first %u elements and %u on the rest\n",
                NB_ATT, NB_ELEM, ELEM_BASE, ELEM_BASE + NB_ELEM - 1u, nbIterationsPerBlock,
                expect[0], NB_ATT % NB_ELEM, expect[NB_ELEM - 1u]);

    for (PxgBufferHandle h : { bSbV.h, bClV.h, bTets.h, bTris.h, bSbD.h, bClD.h,
                               bSbDesc.h, bClDesc.h, bCon.h })
        dev->destroyBuffer(h);
}

} // namespace
