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
//   softBodySimHex*              — P9.95f: the hexahedral (voxel) solve -- the Jacobi
//                                  partition entry and the GS entry's hex branch

#include "pxgTestFramework.h"
// P9.81: the heightfield decoders (Hf / buildHf / getTriangleNoAdj), so the step-2
// contact-gen fixture below mirrors the SAME triangle the kernel decodes rather than
// a second transcription of it.
#include "pxgHeightfieldRef.h"

// P9.57: FROM THE ENGINE ENUM, never a literal -- `PxGeometryType` grew
// eCONVEXCORE at position 4 and shifted everything above eBOX by one.  The copy
// this replaces said eCONVEXMESH was 4, the shader said the same, and the pair
// agreed on a value the engine never produces: the softbody-vs-convex-mesh
// contact was dropped in silence and both sides called it correct.
#include "geometry/PxGeometry.h"

// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers the engine
// descriptor header decorates its inline members with; must precede PxgSoftBody.h.
#include "host_defines.h"
#include "PxgSoftBody.h"                     // real engine per-softbody descriptor
#include "PxActor.h"                         // PxActorFlag::eDISABLE_GRAVITY
#include "PxDeformableBodyFlag.h"            // PxDeformableBodyFlag::eKINEMATIC / eENABLE_SPECULATIVE_CCD
#include "PxDeformableVolumeFlag.h"          // PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC
#include "PxgSolverBody.h"                   // PxgSolverBodyData / PxgSolverTxIData (P8.2 tranche d)
#include "CmSpatialVector.h"                 // Cm::UnAlignedSpatialVector (tranche A: articulation prepare oracle)
#include "PxSpatialMatrix.h"                 // PxSpatialMatrix::operator* -- the engine's own product
#include "PxgArticulation.h"                 // PxgArticulation record (tranche A)
#include "PxgArticulationLink.h"             // PxgArticulationLink (penBiasClamp)
#include "PxgArticulationCoreDesc.h"         // mMaxLinksPerArticulation / nbArticulations
#include "PxgBodySim.h"                      // articulationRemapId
#include "PxgFEMCloth.h"                     // real engine per-cloth descriptor (P8.8 softbody-vs-cloth)
#include "PxgParticleSystem.h"               // real engine per-particle-system descriptor (P9.39a)
#include "pxgHfLocalBoundRef.h"              // P9.89: transformFast(getInverse()) oracle + twins
#include "PxgSimulationCoreDesc.h"           // 5.12: PxgDeformablePairInfo, PxgEncodeDeformable*
#include "PxgDeformableContactWriters.h"     // 5.12: PxgRigidDbContactWriter / PxgDbDbContactWriter
#include "PxsDeformableVolumeMaterialCore.h" // 5.12: PxsDeformableVolumeMaterialData


#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <set>
#include <algorithm>
#include <string>

namespace
{

// P9.64 -- the FEM constraint-prepare and cloth-cloth entries are launched by the
// engine with a FIXED grid of 1024 (PxgKernelGridDim), and their shaders partition
// the work by that constant: each block takes
// `nbIterationsPerBlock * groupId.x * RC_BLOCK` as its offset, with the grid baked
// into nbIterationsPerBlock.  A fixture dispatching ONE group therefore reproduces
// neither the engine's launch nor the partitioning, and -- with 1023 groups absent
// -- nothing checks that those groups write nothing.
//
// Honest limit: at these fixtures' single contact the loop still runs one iteration
// whatever the grid, so the engine's grid alone does not make the partitioning
// CONSTANT observable; that needs more contacts than one block's share.  The grid
// is the floor of the property, not the whole of it.  tools/check_fixture_grid.py
// checks the floor, which is what it can see.
static const uint32_t FEM_PREP_GRID = 1024;

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
// engine PxgSoftBody (offsets the kernels dereference).  PhysX 5.12: sizeof == 464 (the
// three typed element filter arrays + counts replace mFilteringPairs / mNumFilterPairs and
// shift the scalar tail by 24 B; mRestDistance became mRestOffset).
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimDescLayout)
{
    // sizeof / array stride: the kernels index softbodies[id] (sbAt), so the STRIDE must
    // match PXG_SOFTBODY_STRIDE.
    PXG_CHECK_MSG(sizeof(PxgSoftBody) == 464, "PxgSoftBody sizeof == 464 (got %zu)", sizeof(PxgSoftBody));

    // device-pointer members the pre-integrate driver reads/writes
    PXG_CHECK(offsetof(PxgSoftBody, mSimVelocity_InvMass) == 88);
    PXG_CHECK(offsetof(PxgSoftBody, mSimPosition_InvMass) == 96);
    PXG_CHECK(offsetof(PxgSoftBody, mSimKinematicTarget) == 104);
    PXG_CHECK(offsetof(PxgSoftBody, mVertsAreDeformed) == 112);
    PXG_CHECK(offsetof(PxgSoftBody, mVertsCanNotDeform) == 120);
    PXG_CHECK(offsetof(PxgSoftBody, mSimDeltaPos) == 128);
    PXG_CHECK(offsetof(PxgSoftBody, mSimDelta) == 272);
    PXG_CHECK(offsetof(PxgSoftBody, mMaterialIndices) == 296);

    // the 5.12 element filter arrays
    PXG_CHECK(offsetof(PxgSoftBody, mRigidVolumeFilterPairs) == 304);
    PXG_CHECK(offsetof(PxgSoftBody, mVolumeVolumeFilterPairs) == 312);
    PXG_CHECK(offsetof(PxgSoftBody, mVolumeSurfaceFilterPairs) == 320);
    PXG_CHECK(offsetof(PxgSoftBody, mNumRigidVolumeFilterPairs) == 328);
    PXG_CHECK(offsetof(PxgSoftBody, mNumVolumeVolumeFilterPairs) == 332);
    PXG_CHECK(offsetof(PxgSoftBody, mNumVolumeSurfaceFilterPairs) == 336);

    // scalar tail
    PXG_CHECK(offsetof(PxgSoftBody, mLinearDamping) == 340);
    PXG_CHECK(offsetof(PxgSoftBody, mMaxLinearVelocity) == 344);
    PXG_CHECK(offsetof(PxgSoftBody, mPenBiasClamp) == 348);
    PXG_CHECK(offsetof(PxgSoftBody, mSettlingThreshold) == 352);
    PXG_CHECK(offsetof(PxgSoftBody, mNumVertsGM) == 396);
    PXG_CHECK(offsetof(PxgSoftBody, mElementIndex) == 408);
    PXG_CHECK(offsetof(PxgSoftBody, mGpuRemapIndex) == 412);

    // the byte-packed flag triple (@416): byte0 == mActorFlags, byte1 == mBodyFlags,
    // bytes2..3 == mVolumeFlags -> the shader's mFlagsWord fold.
    PXG_CHECK(offsetof(PxgSoftBody, mActorFlags) == 416);
    PXG_CHECK(offsetof(PxgSoftBody, mBodyFlags) == 417);
    PXG_CHECK(offsetof(PxgSoftBody, mVolumeFlags) == 418);
    PXG_CHECK(offsetof(PxgSoftBody, mSimPullIndices) == 424); // realigned 8-byte ptr after the flag word
    PXG_CHECK(offsetof(PxgSoftBody, mAwake) == 432);
    PXG_CHECK(offsetof(PxgSoftBody, mRestOffset) == 444);
    PXG_CHECK(offsetof(PxgSoftBody, mJacobiScale) == 452);

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
    // P9.86: gravity's x and z are -0 and vertex 0's velocity is -0 in x and z, so
    // vel * dt is -0 there; the CUDA's zero-seeded `delta(0); delta += vel * dt`
    // (softBodyGM.cu:182-185) makes that +0, a folded seed would keep -0.
    const float3 g = { -0.0f, -9.81f, -0.0f };
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
    vels[0].x = -0.0f; vels[0].z = -0.0f;
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
    uint32_t zeroSeedSeen = 0;
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
        if (!isTGS && f2u(vx * dt) == 0x80000000u && f2u(dx) == 0u) ++zeroSeedSeen;
        if (!isTGS && f2u(vz * dt) == 0x80000000u && f2u(dz) == 0u) ++zeroSeedSeen;

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
    if (!isTGS)
        PXG_CHECK_MSG(zeroSeedSeen >= 2, "vel*dt == -0 must meet the +0 delta seed (%u)", zeroSeedSeen);

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

// P9.86: the two warp reductions (softBody.cu:259-260 leaf walk, :292-293 node aggregate)
// are warpShuffleMin/Max (gpucommon/src/CUDA/shuffle.cuh:49-72), a __shfl_xor butterfly
// combining with fminf/fmaxf -- NOT the PxMin/PxMax select.  NVIDIA's rule: a NaN operand
// is dropped (NaN,NaN -> canonical NaN) and a zero tie orders -0 < +0 whatever the operand
// order, so the reduction is a total order and a serial scan is exact.  The per-tet
// pairwise fold (softBody.cu:224-255) stays PxMin/PxMax = fmn/fmx.
static float cuFminf(float a, float b)
{
    if (a != a) return (b != b) ? u2f(0x7FFFFFFFu) : b;
    if (b != b) return a;
    if (a == b) return u2f(f2u(a) | f2u(b));   // +-0 tie -> -0
    return a < b ? a : b;
}
static float cuFmaxf(float a, float b)
{
    if (a != a) return (b != b) ? u2f(0x7FFFFFFFu) : b;
    if (b != b) return a;
    if (a == b) return u2f(f2u(a) & f2u(b));   // +-0 tie -> +0
    return a > b ? a : b;
}
static V3 cuFmin3(V3 a, V3 b) { return V3{ cuFminf(a.x, b.x), cuFminf(a.y, b.y), cuFminf(a.z, b.z) }; }
static V3 cuFmax3(V3 a, V3 b) { return V3{ cuFmaxf(a.x, b.x), cuFmaxf(a.y, b.y), cuFmaxf(a.z, b.z) }; }

// Mutant twin: 0 = the CUDA (fminf/fmaxf); 1 = the pre-P9.86 port's PxMin/PxMax select
// scan (a zero tie keeps the LATER lane's sign).  Tie counters count reductions where the
// two spellings differ in any bit (incremented only for variant 0).
static int gRefitReduceVariant = 0;
static uint32_t gTieLeafMin = 0, gTieLeafMax = 0, gTieAggMin = 0, gTieAggMax = 0;

static bool v3BitsDiffer(V3 a, V3 b)
{ return f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z); }

static void warpMinMaxRef(const std::vector<V3>& lmin, const std::vector<V3>& lmax, V3& outMin, V3& outMax,
                          uint32_t& tieMin, uint32_t& tieMax)
{
    V3 cn = lmin[0], cx = lmax[0], sn = lmin[0], sx = lmax[0];
    for (uint32_t lane = 1; lane < 32u; ++lane)
    {
        cn = cuFmin3(cn, lmin[lane]); cx = cuFmax3(cx, lmax[lane]);
        sn = vmin(sn, lmin[lane]);    sx = vmax(sx, lmax[lane]);
    }
    if (gRefitReduceVariant == 0)
    {
        if (v3BitsDiffer(cn, sn)) ++tieMin;
        if (v3BitsDiffer(cx, sx)) ++tieMax;
    }
    outMin = (gRefitReduceVariant == 1) ? sn : cn;
    outMax = (gRefitReduceVariant == 1) ? sx : cx;
}

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
        V3 rn, rx;
        warpMinMaxRef(tMin, tMax, rn, rx, gTieLeafMin, gTieLeafMax);   // softBody.cu:259-260
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

    V3 an, ax;
    warpMinMaxRef(laneMin, laneMax, an, ax, gTieAggMin, gTieAggMax);   // softBody.cu:292-293
    packed[nodeIndex].mn = an;
    packed[nodeIndex].mx = ax;
}
} // namespace p80refit

static void runSbRefitBound(bool speculativeCCD, bool zeroSign = false)
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
    // P9.86 zero-sign variant: bound faces at exactly x=0 / y=0 / z=0 with MIXED zero signs
    // across lanes of one warp reduction (+-0 vertex coordinates: mirrored meshes, IEEE
    // products like 0 * negative).  Nodes 3 and 4 hold TWO tets in one leaf child (leaf
    // walk ties, softBody.cu:259-260); nodes 1/2 feed the root aggregate tie
    // (softBody.cu:292-293 -> mPackedNodeBounds[0] -> boundArray).  Every tet has at most
    // one zero per channel, so the per-tet PxMin/PxMax select fold is sign-unambiguous.
    //   (A) root min.x : lane0 (node1) -0, lane1 (node2) +0  -> CUDA fminf -0, select +0
    //   (B) root max.z : lane0 +0, lane1 -0                  -> CUDA fmaxf +0, select -0
    //   (C) node3 leaf min.y : tet2 -0, tet3 +0              -> CUDA -0, select +0
    //   (D) node4 leaf max.y : tet4 +0, tet5 -0              -> CUDA +0, select -0
    if (zeroSign)
    {
        pos = {
            {  1.0f,    2.0f,   -1.5f,   0.5f },   // 0 g0
            {  2.0f,    0.5f,   -2.5f,   0.5f },   // 1 g1
            {  1.5f,    3.0f,   -0.25f,  0.5f },   // 2 g2
            {  2.5f,    1.0f,   -3.0f,   0.5f },   // 3 g3
            {  1.0f,   -2.0f,   -1.5f,   0.5f },   // 4 n0
            {  2.0f,   -0.5f,   -2.5f,   0.5f },   // 5 n1
            {  1.5f,   -3.0f,   -0.25f,  0.5f },   // 6 n2
            { -0.0f,    1.25f,   0.0f,   0.5f },   // 7 s0: x -0, z +0
            {  0.0f,    1.5f,   -0.0f,   0.5f },   // 8 s1: x +0, z -0
            {  0.75f,  -0.0f,   -1.0f,   0.5f },   // 9 s2: y -0
            {  0.875f,  0.0f,   -1.25f,  0.5f },   // 10 s3: y +0
            {  0.625f,  0.0f,   -0.5f,   0.5f },   // 11 s4: y +0
            {  0.5f,   -0.0f,   -0.75f,  0.5f },   // 12 s5: y -0
        };
        tets = {
            {  7u, 0u, 1u, 2u },   // node 1
            {  8u, 0u, 1u, 2u },   // node 2
            {  9u, 0u, 1u, 2u },   // node 3, lane 0
            { 10u, 0u, 1u, 2u },   // node 3, lane 1
            { 11u, 4u, 5u, 6u },   // node 4, lane 0
            { 12u, 4u, 5u, 6u },   // node 4, lane 1
            // nodes 5..8: the non-zero bulk bound only (permutations of one tet)
            {  0u, 1u, 2u, 3u }, { 1u, 2u, 3u, 0u }, { 2u, 3u, 0u, 1u }, { 3u, 0u, 1u, 2u },
        };
        PXG_CHECK_MSG(f2u(pos[7].x) == 0x80000000u && f2u(pos[8].z) == 0x80000000u &&
                      f2u(pos[9].y) == 0x80000000u && f2u(pos[12].y) == 0x80000000u,
                      "host build must keep the -0 vertex coordinates");
    }
    static const uint32_t kStart[2][8] = { { 0, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 4, 6, 7, 8, 9 } };
    static const uint32_t kCount[2][8] = { { 1, 1, 1, 1, 1, 1, 1, 1 }, { 1, 1, 2, 2, 1, 1, 1, 1 } };

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
    // nodes 1..8: ONE leaf child each, holding tet (n-1) alone (zero-sign variant: nodes
    // 3 and 4 hold two tets each, see kStart/kCount)
    for (uint32_t n = 1; n <= 8u; ++n)
    {
        tree.w[tree.nodeWord(n) + W_MDATA + 0] = leafWord(kStart[zeroSign][n - 1u], kCount[zeroSign][n - 1u]);
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
    gRefitReduceVariant = 0;
    gTieLeafMin = gTieLeafMax = gTieAggMin = gTieAggMax = 0;
    for (uint32_t i = maxDepth; i > 0u; --i)
    {
        const uint32_t off = refTree.w[wDepthInfo + (i - 1u) * 2u + 0];
        const uint32_t cnt = refTree.w[wDepthInfo + (i - 1u) * 2u + 1];
        for (uint32_t j = 0; j < cnt; ++j)
            refitNodeRef(refTree.w[wRemap + j + off], refTree, refPacked, pos, tets);
    }
    if (zeroSign)
    {
        const uint32_t tlm = gTieLeafMin, tlx = gTieLeafMax, tam = gTieAggMin, tax = gTieAggMax;
        // Twin walk with the pre-P9.86 select scan: it must flip all four zero signs.
        TreeBlob twinTree = tree;
        std::vector<Bounds> twinPacked(nbNodes);
        for (uint32_t n = 0; n < nbNodes; ++n)
        { twinPacked[n].mn = V3{ POISON, POISON, POISON }; twinPacked[n].mx = twinPacked[n].mn; }
        gRefitReduceVariant = 1;
        for (uint32_t i = maxDepth; i > 0u; --i)
        {
            const uint32_t off = twinTree.w[wDepthInfo + (i - 1u) * 2u + 0];
            const uint32_t cnt = twinTree.w[wDepthInfo + (i - 1u) * 2u + 1];
            for (uint32_t j = 0; j < cnt; ++j)
                refitNodeRef(twinTree.w[wRemap + j + off], twinTree, twinPacked, pos, tets);
        }
        gRefitReduceVariant = 0;
        PXG_CHECK_MSG(tlm == 1u && tlx == 1u && tam == 1u && tax == 1u,
                      "zero-sign coverage: leaf %u/%u agg %u/%u fminf-vs-select ties (want 1 each)",
                      tlm, tlx, tam, tax);
        PXG_CHECK_MSG(f2u(refPacked[0].mn.x) == 0x80000000u && f2u(refPacked[0].mx.z) == 0u &&
                      f2u(refPacked[3].mn.y) == 0x80000000u && f2u(refPacked[4].mx.y) == 0u,
                      "CUDA fminf/fmaxf zero rule: root mn.x -0, mx.z +0, node3 mn.y -0, node4 mx.y +0");
        PXG_CHECK_MSG(f2u(twinPacked[0].mn.x) == 0u && f2u(twinPacked[0].mx.z) == 0x80000000u &&
                      f2u(twinPacked[3].mn.y) == 0u && f2u(twinPacked[4].mx.y) == 0x80000000u,
                      "select-scan twin must flip all four zero signs -- the case no longer observes the revert");
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
            wholeMin = cuFmin3(wholeMin, V3{ pos[vi].x, pos[vi].y, pos[vi].z });   // warpShuffle fminf order
            wholeMax = cuFmax3(wholeMax, V3{ pos[vi].x, pos[vi].y, pos[vi].z });
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
PXG_TEST(softBodySimRefitBoundZeroSign) { runSbRefitBound(/*speculativeCCD*/ false, /*zeroSign*/ true); }


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

// PhysX 5.12: (gSoftbodies, activeSoftbodies, invDt, dt, alwaysRunVelocityAveraging,
// nbPosIters) -- velocityScale became nbPosIters (LinearBlendVelocityAveraging).
struct SbFinPush
{
    uint64_t gSoftbodies, activeSoftbodies;
    float invDt, dt;
    uint32_t alwaysAvgWord;
    uint32_t nbPosIters;
};
static_assert(sizeof(SbFinPush) == 32, "finalize push blob must be 32 bytes");
static_assert(offsetof(SbFinPush, invDt) == 16, "invDt @16");
static_assert(offsetof(SbFinPush, alwaysAvgWord) == 24, "alwaysAvg @24");
static_assert(offsetof(SbFinPush, nbPosIters) == 28, "nbPosIters @28");

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
    // LinearBlendVelocityAveraging(nbPosIters): scale = PxMin(0.9f, 2 * PxSqrt(1 / nbPosIters)).
    // 7 position iterations give 2 * sqrt(1/7) ~ 0.756, below the 0.9 cap, so the sqrt and
    // the divide both reach the result.
    const uint32_t nbPosIters = 7u;
    const float blendScale = [&] { const float t = 2.0f * std::sqrt(1.0f / float(nbPosIters));
                                   return (0.9f < t) ? 0.9f : t; }();
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
    push.invDt = invDt; push.dt = dt;
    push.alwaysAvgWord = alwaysRunVelocityAveraging ? 1u : 0u;
    push.nbPosIters = nbPosIters;

    if (onSleepThreshold)
        for (uint32_t i = 0; i < N; ++i)
            deltas[i] = { sleepTol, 0.0f, 0.0f, 42.0f };

    dev->upload(bDelta.h, deltas.data(), deltas.size() * sizeof(F4));

    Kernel k("softBodySimBda.spv", "sb_gm_finalizeVelocitiesLaunch", {}, sizeof(SbFinPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference -- 5.12 finalizeVertexVelocity + LinearBlendVelocityAveraging
    // (deformableUtils.cuh), expression order transcribed from the CUDA.
    std::vector<F4> refVel = vels;
    bool refAwake = false;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float settleTolerance = settlingThreshold * dt;
        const float tolerance = sleepThreshold * dt;
        const float sleepDamping = 1.0f - ((1.0f < settlingDamping * dt) ? 1.0f : settlingDamping * dt);

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
                    tVel = add(mul(tVel, blendScale), mul(deltaVel, 1.0f - blendScale));
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
//   sb_gm_applyExternalDeltasLaunch (softBodyGM.cu:2222, 5.12) — folds the accumulated
//                                     per-vertex correction back and clears the delta
// ===========================================================================

struct ZeroTetPush { uint64_t gSoftbodies, activeSoftbodies; };
static_assert(sizeof(ZeroTetPush) == 16, "zeroTetMultipliers push blob must be 16 bytes");

// 5.12: (PxgSoftBody*, const PxU32*, PxReal invDt, bool isVelocityIteration) -> 21 bytes,
// rounded up to 24.
struct ApplyDeltasPush
{
    uint64_t gSoftbodies, activeSoftbodies;
    float invDt;
    uint32_t isVelocityIteration;   // bool byte0
};
static_assert(sizeof(ApplyDeltasPush) == 24, "applyExternalDeltas push blob must be 24 bytes");
static_assert(offsetof(ApplyDeltasPush, invDt) == 16, "invDt @16");
static_assert(offsetof(ApplyDeltasPush, isVelocityIteration) == 20, "isVelocityIteration @20");

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
// not an integer one.  5.12: a velocity iteration writes only the velocity correction and
// leaves position and mSimDeltaPos alone (softBodyGM.cu:2259); the delta is cleared either way.
// ---------------------------------------------------------------------------
static void runApplyExternalDeltas(bool isVelocityIteration)
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
    // P9.86: vertex 6 is PINNED with a -0 inverse mass (legal: every consumer tests
    // w != 0 / w > 0, and the user writes the sim position/invMass buffer directly).  The
    // CUDA's w lanes are IEEE `pos.w + 0.f` / `deltaPos.w + 0.f` (softBodyGM.cu:2652/2656)
    // -> +0; a folded literal-zero w lane would keep -0.  vel.w = -0 pins the computed-zero
    // velocity lane (`vel.w + 0*invDt`, +0 on both sides).
    pos[6].w = -0.0f; deltaPos[6].w = -0.0f; vel[6].w = -0.0f;

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
    push.isVelocityIteration = isVelocityIteration ? 0xFFFFFF01u : 0xFFFFFF00u;   // only byte 0 is the bool

    Kernel k("softBodySolveBda.spv", "sb_gm_applyExternalDeltasLaunch", {}, sizeof(ApplyDeltasPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // CPU reference — expression order transcribed from softBodyGM.cu:2222 (5.12).
    std::vector<F4> refPos(N), refVel(N), refDeltaPos(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        F4 d = delta[i];
        const float scale = 1.0f / ((d.w > 1.0f) ? d.w : 1.0f);   // PxMax<float> = selectMax (softBodyGM.cu:2644)
        d.x *= scale; d.y *= scale; d.z *= scale;
        d.w = 0.0f;
        refVel[i] = { vel[i].x + d.x * invDt, vel[i].y + d.y * invDt,
                      vel[i].z + d.z * invDt, vel[i].w + d.w * invDt };
        refPos[i] = { pos[i].x + d.x, pos[i].y + d.y, pos[i].z + d.z, pos[i].w + d.w };
        refDeltaPos[i] = { deltaPos[i].x + d.x, deltaPos[i].y + d.y,
                           deltaPos[i].z + d.z, deltaPos[i].w + d.w };
        if (isVelocityIteration)
        {
            refPos[i] = pos[i];
            refDeltaPos[i] = deltaPos[i];
        }
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
    const float expectX3 = vel[3].x + (delta[3].x * (1.0f / 7.0f)) * invDt;
    PXG_CHECK_MSG(f2u(gVel[3].x) == f2u(expectX3),
                  "vertex 3 (delta.w == 7) must be averaged by 1/7: ref %g got %g", expectX3, gVel[3].x);
    // ...and on the never-below-one clamp: vertex 5 has w == 0.5, which must NOT scale up.
    const float expectX5 = vel[5].x + (delta[5].x * 1.0f) * invDt;
    PXG_CHECK_MSG(f2u(gVel[5].x) == f2u(expectX5),
                  "vertex 5 (delta.w == 0.5) must clamp the divisor to 1: ref %g got %g", expectX5, gVel[5].x);
    PXG_CHECK_MSG(mm == 0, "sb_gm_applyExternalDeltasLaunch bit-exact vs CPU (%zu mismatches)", mm);
    // P9.86 coverage: exactly one -0 inverse mass meets the literal +0 w lane, and the
    // IEEE reference turns it into +0.
    uint32_t negZeroInvMass = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (f2u(pos[i].w) == 0x80000000u && f2u(refPos[i].w) == 0u &&
            f2u(deltaPos[i].w) == 0x80000000u && f2u(refDeltaPos[i].w) == 0u) ++negZeroInvMass;
    PXG_CHECK_MSG(negZeroInvMass == (isVelocityIteration ? 0u : 1u),
                  "a -0 inverse mass must meet the literal +0 w lane (%u)", negZeroInvMass);
    if (isVelocityIteration)
        PXG_CHECK_MSG(f2u(gPos[6].w) == 0x80000000u && f2u(gDeltaPos[6].w) == 0x80000000u && f2u(gVel[6].w) == 0u,
                      "velocity iteration: position / deltaPos untouched, vel.w + 0 -> +0 (pos %08x deltaPos %08x vel %08x)",
                      f2u(gPos[6].w), f2u(gDeltaPos[6].w), f2u(gVel[6].w));
    else
        PXG_CHECK_MSG(f2u(gPos[6].w) == 0u && f2u(gDeltaPos[6].w) == 0u && f2u(gVel[6].w) == 0u,
                      "vertex 6's -0 inverse mass must come back +0 (pos %08x deltaPos %08x vel %08x)",
                      f2u(gPos[6].w), f2u(gDeltaPos[6].w), f2u(gVel[6].w));

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bDeltaPos.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(softBodySimApplyExternalDeltas) { runApplyExternalDeltas(false); }
PXG_TEST(softBodySimApplyExternalDeltasVel) { runApplyExternalDeltas(true); }


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

// PhysX 5.12: PxsDeformableVolumeMaterialData lost dampingScale; the GM solve damps with
// elasticityDamping alone (5.6.1's scaledDamping = damping * toUniformReal(dampingScale)).
struct Material { float youngs, poissons, elasticityDamping; uint32_t model; };

// P9.86 mutant twin (0 = the CUDA).  1 replays a port whose ARAP-call accumulate lost the
// IEEE `+0 + x`: the PxMat33(PxZero) seed of tetrahedronsSolveGM (softBodyGM.cu:1122) is
// forwarded into `dl += deltaLambda*dCdx` (:601-603) and llvmpipe folds the literal-zero
// add, so a -0 product survives where the CUDA stores +0.
static int gDlSeedFoldVariant = 0;

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
                                     float alphaTilde, float dtInv, float damping,
                                     bool seedFold = false)
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
                       damping * (dot3(dCdx0,vv0) + dot3(dCdx1,vv1) + dot3(dCdx2,vv2) + dot3(dCdx3,vv3))) / denom;
    }
    lambda += deltaLambda;
    for (int i = 0; i < 3; ++i)
    {
        if (seedFold) { dl.c0[i] = deltaLambda*dCdx1[i]; dl.c1[i] = deltaLambda*dCdx2[i]; dl.c2[i] = deltaLambda*dCdx3[i]; }
        else { dl.c0[i] += deltaLambda*dCdx1[i]; dl.c1[i] += deltaLambda*dCdx2[i]; dl.c2[i] += deltaLambda*dCdx3[i]; }
    }
}

static void arapConstraint(M3& dl, float& lambda, const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                           const Q4& tetR, const M3& F, float dtInv, const M3& DmInv,
                           float alphaTilde, float damping)
{
    M3 R; mat33FromQuat(tetR, R);
    M3 t;
    sub3(F.c0, R.c0, t.c0); sub3(F.c1, R.c1, t.c1); sub3(F.c2, R.c2, t.c2);
    const float ARAP_C = std::sqrt(dot3(t.c0,t.c0) + dot3(t.c1,t.c1) + dot3(t.c2,t.c2) + 1.0e-18f);
    const float inv = 1.0f / ARAP_C;
    scale3(t.c0, inv, t.c0); scale3(t.c1, inv, t.c1); scale3(t.c2, inv, t.c2);
    computeDeltaLambdaXGradC(dl, lambda, ARAP_C, t, DmInv, v0, v1, v2, v3, alphaTilde, dtInv, damping,
                             gDlSeedFoldVariant == 1);
}

static void volumeConstraint(M3& dl, float& lambda, const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                             const M3& F, float invDt, const M3& DmInv,
                             float alphaTilde, float damping)
{
    const float C = det3(F) - 1.0f;
    M3 dCdF;
    cross3(F.c1, F.c2, dCdF.c0); cross3(F.c2, F.c0, dCdF.c1); cross3(F.c0, F.c1, dCdF.c2);
    computeDeltaLambdaXGradC(dl, lambda, C, dCdF, DmInv, v0, v1, v2, v3, alphaTilde, invDt, damping);
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
        float lam0 = isTGS ? 0.0f : mbyz[0];
        float lam1 = isTGS ? 0.0f : mbyz[1];

        arapConstraint(dl, lam0, v0, v1, v2, v3, tetR, F, invDt, Qinv, alphaTilde0, damping);

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
            damping *= ratio;
            volumeConstraint(dl, lam1, v0, v1, v2, v3, F, invDt, Qinv, alphaTilde1, damping);
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
    matWords[0] = f2u(youngs);
    matWords[1] = f2u(poissons);
    matWords[2] = f2u(0.0f);                     // dynamicFriction
    matWords[3] = f2u(damping);                  // elasticityDamping
    // 5.12: materialModel (PxU16) @16, _pad (PxU16) @18 -- the pad is poisoned so a port
    // that still read the 5.6.1 dampingScale/materialModel pair would see it.
    matWords[4] = materialModel | (0xBEEFu << 16);
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
    Material mat = { youngs, poissons, damping, materialModel };

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
// 5.12: with dampingScale gone the velocity term is damped by elasticityDamping itself (5.6.1
// halved it through dampingScale = 32768/65535).  At 0.35 one lane of this fixed scene then lands
// on a cancellation that magnifies the rotation seam to ~1e4 ULP (0.3 -> 42 ULP, 0.5 -> 268 ULP,
// measured), so the damped case runs at 0.5; the budget is unchanged.
PXG_TEST(softBodySimSolveCorotationalDamped) { runSolveTetraPartition(0u, /*isTGS*/ false, /*damping*/ 0.5f); }
PXG_TEST(softBodySimSolveCorotationalTGS) { runSolveTetraPartition(0u, /*isTGS*/ true, /*damping*/ 0.0f); }
PXG_TEST(softBodySimSolveNeoHookeanPGS) { runSolveTetraPartition(1u, /*isTGS*/ false, /*damping*/ 0.0f); }
PXG_TEST(softBodySimSolveNeoHookeanTGS) { runSolveTetraPartition(1u, /*isTGS*/ true, /*damping*/ 0.0f); }

// ---------------------------------------------------------------------------
// P9.86: sb_gm_cp_solveTetrahedronsPartitionLaunch on RESTING unit tets, compared
// BITWISE.  Co-rotational, TGS, undamped, poissons = 0 (the volume pass is skipped in
// both the CUDA, softBodyGM.cu:1004, and the port), Qinv = I, tetRot = identity, so
// F = R = I exactly: every ARAP dCdx lane is an exact +0, C = sqrt(1e-18) and
// deltaLambda = -C/alphaTilde0 < 0, so every product deltaLambda*dCdx is -0.  The CUDA
// accumulates it onto the PxMat33(PxZero) seed (softBodyGM.cu:1122, :601-603) and holds
// +0; a port that folds the literal-zero seed add keeps -0.  The sign reaches the scatter
// (softBodyGM.cu:1136-1145) through a zero operand of the other sign:
//   tet 0: +0 coordinates, gathered velocities (-0,-0,-0) (a body damped to rest)
//          -> the 12 x/y/z velocity lanes of slots 0-3;
//   tet 1: -0 zero coordinates, velocities +0 -> 9 position lanes of slots 4-7.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimSolveCorotationalRestZeroSign)
{
    using namespace p80solve;
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t nbTets = 2;
    const float invDt = 60.0f;
    const float W[4] = { 1.0f, 0.5f, 2.0f, 0.25f };

    std::vector<F4> unPos(8), unVel(8);
    const float nz = -0.0f;
    // tet 0: verts 0-3, +0 coordinates, -0 velocities
    unPos[0] = { 0.0f, 0.0f, 0.0f, W[0] };
    unPos[1] = { 1.0f, 0.0f, 0.0f, W[1] };
    unPos[2] = { 0.0f, 1.0f, 0.0f, W[2] };
    unPos[3] = { 0.0f, 0.0f, 1.0f, W[3] };
    for (uint32_t i = 0; i < 4; ++i) unVel[i] = { nz, nz, nz, W[i] };
    // tet 1: verts 4-7, -0 zero coordinates, +0 velocities
    unPos[4] = { nz, nz, nz, W[0] };
    unPos[5] = { 1.0f, nz, nz, W[1] };
    unPos[6] = { nz, 1.0f, nz, W[2] };
    unPos[7] = { nz, nz, 1.0f, W[3] };
    for (uint32_t i = 0; i < 4; ++i) unVel[4 + i] = { 0.0f, 0.0f, 0.0f, W[i] };

    std::vector<F4> cpPos(8, F4{ 777.0f, 777.0f, 777.0f, 777.0f }), cpVel = cpPos;
    std::vector<p80refit::TetIdx4> pull = { { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u } };
    std::vector<p80refit::TetIdx4> remap = pull;

    M3 rest[2];
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        rest[t] = M3{};
        rest[t].c0[0] = 1.0f; rest[t].c1[1] = 1.0f; rest[t].c2[2] = 1.0f;
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
    std::vector<F4> tetRot(nbTets, F4{ 0.0f, 0.0f, 0.0f, 1.0f });
    std::vector<uint32_t> mult(SPATIAL_BLOCK_WORDS, 0u);

    std::vector<uint32_t> matWords(8, 0u);
    const float youngs = 1.0e5f, poissons = 0.0f;
    matWords[0] = f2u(youngs);
    matWords[1] = f2u(poissons);
    matWords[2] = f2u(0.0f);
    matWords[3] = f2u(0.0f);                     // elasticityDamping
    matWords[4] = 0u;                            // materialModel eCO_ROTATIONAL @16, _pad @18
    std::vector<uint16_t> matIdx(4u + 8u, 0u);
    std::vector<uint32_t> partitions = { nbTets, 0u, 0u, 0u, 0u };

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
    push.flagsWord = 1u;   // TGS, isFirstIteration = false
    push.materials = bMat.addr;

    Kernel k("softBodySolveBda.spv", "sb_gm_cp_solveTetrahedronsPartitionLaunch", {}, sizeof(SolveTetraPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // ---- CPU reference (variant 0 = the CUDA, 1 = the seed-fold twin) ------
    const float invDtSq = invDt * invDt;
    const Material mat = { youngs, poissons, 0.0f, 0u };
    auto ref = [&](int variant, std::vector<F4>& rp, std::vector<F4>& rv, Q4 rq[2]) {
        rp = cpPos; rv = cpVel;
        for (uint32_t t = 0; t < nbTets; ++t)
        {
            const uint32_t pi[4] = { pull[t].x, pull[t].y, pull[t].z, pull[t].w };
            F4 p[4], v[4];
            for (uint32_t i = 0; i < 4; ++i) { p[i] = unPos[pi[i]]; v[i] = unVel[pi[i]]; }
            const float u1[3] = { p[1].x-p[0].x, p[1].y-p[0].y, p[1].z-p[0].z };
            const float u2[3] = { p[2].x-p[0].x, p[2].y-p[0].y, p[2].z-p[0].z };
            const float u3[3] = { p[3].x-p[0].x, p[3].y-p[0].y, p[3].z-p[0].z };
            M3 U; for (int i=0;i<3;++i){ U.c0[i]=u1[i]; U.c1[i]=u2[i]; U.c2[i]=u3[i]; }
            M3 Fm; mat33Mul(U, rest[t], Fm);
            Q4 q = { tetRot[t].x, tetRot[t].y, tetRot[t].z, tetRot[t].w };
            q = p80rot::extractRotation(Fm, q, 4);
            rq[t] = q;

            float mbyz[2] = { 0.0f, 0.0f };   // read only under PGS
            gDlSeedFoldVariant = variant;
            M3 dl;
            solveGM(dl, u1, u2, u3, v[0], v[1], v[2], v[3], mbyz, invDtSq, invDt, rest[t], q, /*isTGS*/ true, mat);
            gDlSeedFoldVariant = 0;

            const uint32_t wi[4] = { remap[t].x, remap[t].y, remap[t].z, remap[t].w };
            for (uint32_t i = 0; i < 4; ++i)
            {
                float dp[3];
                if (i == 0) for (int j=0;j<3;++j) dp[j] = -(dl.c0[j] + dl.c1[j] + dl.c2[j]) * p[i].w;
                else { const float* cc = (i==1)?dl.c0:((i==2)?dl.c1:dl.c2); for (int j=0;j<3;++j) dp[j] = cc[j]*p[i].w; }
                rp[wi[i]] = { p[i].x+dp[0], p[i].y+dp[1], p[i].z+dp[2], p[i].w };
                rv[wi[i]] = { v[i].x+dp[0]*invDt, v[i].y+dp[1]*invDt, v[i].z+dp[2]*invDt, v[i].w };
            }
        }
    };
    std::vector<F4> wantPos, wantVel, twinPos, twinVel;
    Q4 wantQ[2], twinQ[2];
    ref(0, wantPos, wantVel, wantQ);
    ref(1, twinPos, twinVel, twinQ);

    // Coverage: the twin must flip exactly the 12 velocity lanes of tet 0 and the 9
    // position lanes of tet 1 (everything else identical), and the -0 inputs must be there.
    uint32_t velFlip = 0, posFlip = 0, otherFlip = 0, negZeroVelIn = 0;
    for (uint32_t s = 0; s < 8; ++s)
    {
        const float wp[3] = { wantPos[s].x, wantPos[s].y, wantPos[s].z };
        const float tp[3] = { twinPos[s].x, twinPos[s].y, twinPos[s].z };
        const float wv[3] = { wantVel[s].x, wantVel[s].y, wantVel[s].z };
        const float tv[3] = { twinVel[s].x, twinVel[s].y, twinVel[s].z };
        for (int e = 0; e < 3; ++e)
        {
            const bool pf = f2u(wp[e]) != f2u(tp[e]), vf = f2u(wv[e]) != f2u(tv[e]);
            if (s < 4) { velFlip += vf ? 1u : 0u; otherFlip += pf ? 1u : 0u; }
            else { posFlip += pf ? 1u : 0u; otherFlip += vf ? 1u : 0u; }
        }
    }
    for (uint32_t i = 0; i < 4; ++i)
        negZeroVelIn += (f2u(unVel[i].x) == 0x80000000u) + (f2u(unVel[i].y) == 0x80000000u) + (f2u(unVel[i].z) == 0x80000000u);
    PXG_CHECK_MSG(velFlip == 12u && posFlip == 9u && otherFlip == 0u && negZeroVelIn == 12u,
                  "the seed-fold twin must flip 12 velocity + 9 position lanes (vel %u pos %u other %u, -0 vel in %u)",
                  velFlip, posFlip, otherFlip, negZeroVelIn);

    const auto gPos = downloadBuffer<F4>(dev, bCpPos.h, cpPos.size());
    const auto gVel = downloadBuffer<F4>(dev, bCpVel.h, cpVel.size());
    const auto gRot = downloadBuffer<F4>(dev, bRot.h, nbTets);

    // Precondition: the rotation refresh is exact at the identity (the sin/cos seam of the
    // tolerance-based cases contributes nothing here).
    for (uint32_t t = 0; t < nbTets; ++t)
        PXG_CHECK_MSG(f2u(gRot[t].x) == 0u && f2u(gRot[t].y) == 0u && f2u(gRot[t].z) == 0u &&
                      f2u(gRot[t].w) == 0x3f800000u && f2u(wantQ[t].x) == 0u && f2u(wantQ[t].y) == 0u &&
                      f2u(wantQ[t].z) == 0u && f2u(wantQ[t].w) == 0x3f800000u,
                      "tet %u rotation must stay the exact identity (got %08x %08x %08x %08x)", t,
                      f2u(gRot[t].x), f2u(gRot[t].y), f2u(gRot[t].z), f2u(gRot[t].w));

    size_t mm = 0;
    auto cmp = [&](const char* what, uint32_t i, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) ||
            f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%08x,%08x,%08x,%08x) got(%08x,%08x,%08x,%08x)\n", what, i,
                            f2u(r.x), f2u(r.y), f2u(r.z), f2u(r.w), f2u(g.x), f2u(g.y), f2u(g.z), f2u(g.w)); } };
    for (uint32_t s = 0; s < 8; ++s)
    {
        cmp("cpPos", s, wantPos[s], gPos[s]);
        cmp("cpVel", s, wantVel[s], gVel[s]);
    }
    PXG_CHECK_MSG(mm == 0, "sb_gm_cp_solveTetrahedronsPartitionLaunch rest tets bit-exact vs CUDA zero rule (%zu)", mm);
    PXG_CHECK_MSG(f2u(gVel[0].x) == 0x80000000u && f2u(gVel[1].x) == 0u &&
                  f2u(gPos[4].x) == 0x80000000u && f2u(gPos[5].y) == 0u,
                  "rest-tet zero signs: vel0.x %08x (want -0) vel1.x %08x (want +0) pos4.x %08x (want -0) pos5.y %08x (want +0)",
                  f2u(gVel[0].x), f2u(gVel[1].x), f2u(gPos[4].x), f2u(gPos[5].y));

    dev->destroyBuffer(bCpPos.h); dev->destroyBuffer(bCpVel.h); dev->destroyBuffer(bUnPos.h);
    dev->destroyBuffer(bUnVel.h); dev->destroyBuffer(bPull.h); dev->destroyBuffer(bRemap.h);
    dev->destroyBuffer(bRest.h); dev->destroyBuffer(bRot.h); dev->destroyBuffer(bMult.h);
    dev->destroyBuffer(bMat.h); dev->destroyBuffer(bMatIdx.h); dev->destroyBuffer(bPart.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ===========================================================================
// P9.95f — the HEXAHEDRAL (voxel) solve: sb_gm_cp_solveHexahedronsPartition_parallelGS
// (softBodyGM.cu:1313-1404) through its two callers,
//   sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch (softBodyGM.cu:1463), the 9th
//     ("Jacobi") partition: no position/velocity writes, AtomicAdd(mSimDelta, (p-pos, 1));
//   the hex branch of sb_gm_cp_solveTetrahedronsPartitionLaunch (softBodyGM.cu:1450-1455):
//     in-place p/v and mSimDeltaPos += p - pos.
//
// The oracle (namespace p95hex) is the .cu spelled literally on the host -- F4 p[8], v[8]
// and the two flat tables are fine here -- reusing the tranche-5/6 transcriptions
// p80rot::extractRotation and p80solve::solveGM (tetrahedronsSolveGM), under the TU's
// -ffp-contract=off.
//
// BIT-EXACT despite the sin/cos seam, by construction rather than by tolerance:
//   - rest corners are the unit voxel (x = bit2, y = bit1, z = bit0) and current positions
//     are diag(s)*rest + t with dyadic s, t, so every sub-tet's F = U*Qinv is EXACTLY
//     diagonal: extractRotation sees omega == 0, sin(0) == 0, cos(0) == 1, and an identity q
//     is a bit-exact fixed point for 4 or 100 iterations;
//   - the sub-tets of a voxel run in sequence on the gathered copies, so a moved corner
//     would hand the next sub-tet a non-affine F.  Non-zero inverse mass therefore goes only
//     to corners that belong to EXACTLY ONE sub-tet: {1,7,4,2} non-flipped, {0,6,5,3}
//     flipped (the central row owns none);
//   - Jacobi voxels read the unwritten positions, and a vertex shared by two Jacobi voxels
//     is single-owner in both or massless; no vertex gets more than two Jacobi
//     contributions, so the atomic order cannot change a bit (0 + a + b == 0 + b + a), and
//     a non-zero mSimDelta seed sits only on one-contribution vertices;
//   - the 6-tet table has no single-owner corner (every row holds corners 1 and 6), so 6-tet
//     voxels get a massless ROUTING tier (bit-exact) and a ULP-BUDGET tier.
// ===========================================================================

struct SolveTetraJacobiPush
{
    uint64_t gSoftbodies, activeSoftbodies;
    uint32_t numActiveSoftbodies;
    float invDt;
    uint32_t flagsWord, pad28;
    uint64_t materials;
};
static_assert(sizeof(SolveTetraJacobiPush) == 40, "solveTetraJacobi push blob must be 40 bytes");
static_assert(offsetof(SolveTetraJacobiPush, invDt) == 20, "invDt @20");
static_assert(offsetof(SolveTetraJacobiPush, flagsWord) == 24, "isTGS @24");
static_assert(offsetof(SolveTetraJacobiPush, pad28) == 28, "pad @28");
static_assert(offsetof(SolveTetraJacobiPush, materials) == 32, "materials @32");

namespace p95hex
{
using p80rot::M3;
using p80rot::Q4;
using p80refit::TetIdx4;

// softBodyGM.cu:1209 and :1211-1213, verbatim.
static const uint32_t tets6PerVoxel[24] = { 0,1,6,2,  0,1,4,6,  1,4,6,5,  1,2,3,6,  1,3,7,6,  1,5,6,7 };
static const uint32_t tets5PerVoxel[40] = {
                     0, 6, 3, 5,   0, 1, 5, 3,   6, 7, 3, 5,   4, 5, 6, 0,   2, 3, 0, 6,
                     1, 7, 4, 2,   1, 0, 2, 4,   7, 6, 4, 2,   5, 4, 1, 7,   3, 2, 7, 1 };
static const uint32_t PULL_IND_MASK = 0x7fffffffu;
static const uint32_t FLIPPED_MASK = 0x80000000u;
// SB_PARTITION_LIMIT (8) comes from PxgSoftBody.h itself.
static const uint32_t POISON = 0x7F7F7F7Fu;
static const float SENTINEL = 777.0f;
static const uint32_t PAD = 64u;

// One softbody's buffers, host side.  Every array is padded, and every padding / untouched
// slot carries poison, so a stray read or write past the live range shows.
struct HexBody
{
    uint32_t tpe = 5, nbElements = 0, numPartitions = 9;
    uint8_t bodyFlags = 0;
    Q4 initialRotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    uint32_t nbVerts = 0;                // live vertices; [nbVerts, nbVerts+PAD) are sentinels
    std::vector<uint32_t> acc;           // mSimAccumulatedPartitionsCP
    std::vector<TetIdx4> pull;           // mSimPullIndices, 2*nbElements live
    std::vector<F4> pos, vel, delta, deltaPos, cpPos, cpVel;
    std::vector<TetIdx4> remap;          // mSimRemapOutputCP (the hex path must not touch it)
    std::vector<uint32_t> rest;          // mSimTetraRestPoses, PxgMat33Block words
    std::vector<F4> rot;                 // mSimTetraRotations
    std::vector<uint32_t> mult;          // mSimTetraMultipliers, PxgSpatialVectorBlock words
    std::vector<uint16_t> matIdx;        // mOrderedMaterialIndices, preceded by matLead halves
    uint32_t matLead = 0;                // 1 -> the descriptor points 2 bytes into the buffer
    uint32_t nbTets() const { return nbElements * tpe; }
};

static void initBody(HexBody& b, uint32_t tpe, uint32_t nbElements, uint32_t nbVerts)
{
    b.tpe = tpe; b.nbElements = nbElements; b.nbVerts = nbVerts;
    const uint32_t nbTets = tpe * nbElements;
    const uint32_t nbBlocks = (nbTets + 31u) / 32u + 2u;
    b.acc.assign(16u, POISON);
    // padding pull entries point at a sentinel vertex: a mutant that reads past the live
    // range gathers garbage, not an address fault
    b.pull.assign(2u * nbElements + PAD, TetIdx4{ nbVerts, nbVerts, nbVerts, nbVerts });
    const F4 sent = { SENTINEL, SENTINEL, SENTINEL, SENTINEL };
    b.pos.assign(nbVerts + PAD, sent);
    b.vel.assign(nbVerts + PAD, sent);
    b.delta.assign(nbVerts + PAD, F4{ 1234.5f, -1234.5f, 99.0f, 7.0f });
    b.deltaPos.assign(nbVerts + PAD, F4{ 555.0f, -555.0f, 55.0f, 5.0f });
    b.cpPos.assign(nbTets * 4u + PAD, sent);
    b.cpVel.assign(nbTets * 4u + PAD, sent);
    b.remap.assign(nbTets + PAD, TetIdx4{ POISON, POISON, POISON, POISON });
    b.rest.assign(nbBlocks * MAT33_BLOCK_WORDS, POISON);
    b.rot.assign(nbTets + PAD, F4{ u2f(POISON), u2f(POISON), u2f(POISON), u2f(POISON) });
    b.mult.assign(nbBlocks * SPATIAL_BLOCK_WORDS, POISON);
    b.matIdx.assign(nbElements + PAD, uint16_t(0u));
}

static void setRest(HexBody& b, uint32_t slot, const M3& m)
{
    const uint32_t base = (slot / 32u) * MAT33_BLOCK_WORDS, lane = slot & 31u;
    b.rest[base + lane*4+0] = f2u(m.c0[0]); b.rest[base + lane*4+1] = f2u(m.c0[1]);
    b.rest[base + lane*4+2] = f2u(m.c0[2]); b.rest[base + lane*4+3] = f2u(m.c1[0]);
    b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+0] = f2u(m.c1[1]);
    b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+1] = f2u(m.c1[2]);
    b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+2] = f2u(m.c2[0]);
    b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+3] = f2u(m.c2[1]);
    b.rest[base + MAT33_BLOCK_COL2_WORD + lane] = f2u(m.c2[2]);
}

// loadPxMat33(mSimTetraRestPoses[slot/32], slot&31)
static M3 getRest(const HexBody& b, uint32_t slot)
{
    const uint32_t base = (slot / 32u) * MAT33_BLOCK_WORDS, lane = slot & 31u;
    M3 m;
    m.c0[0] = u2f(b.rest[base + lane*4+0]); m.c0[1] = u2f(b.rest[base + lane*4+1]);
    m.c0[2] = u2f(b.rest[base + lane*4+2]); m.c1[0] = u2f(b.rest[base + lane*4+3]);
    m.c1[1] = u2f(b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+0]);
    m.c1[2] = u2f(b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+1]);
    m.c2[0] = u2f(b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+2]);
    m.c2[1] = u2f(b.rest[base + MAT33_BLOCK_COL1_WORD + lane*4+3]);
    m.c2[2] = u2f(b.rest[base + MAT33_BLOCK_COL2_WORD + lane]);
    return m;
}

static uint32_t mbyzWord(uint32_t slot) { return (slot / 32u) * SPATIAL_BLOCK_WORDS + SPATIAL_BLOCK_MBYZ_WORD + (slot & 31u) * 2u; }

// The rest-pose inverse of a sub-tet with corners a..d of the unit voxel (x = bit2,
// y = bit1, z = bit0).  Exact: the entries are multiples of 1/2, +0 never -0.
static M3 unitRestInverse(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    auto cx = [](uint32_t j) { return double((j >> 2) & 1u); };
    auto cy = [](uint32_t j) { return double((j >> 1) & 1u); };
    auto cz = [](uint32_t j) { return double(j & 1u); };
    const double D[3][3] = { { cx(b)-cx(a), cx(c)-cx(a), cx(d)-cx(a) },
                             { cy(b)-cy(a), cy(c)-cy(a), cy(d)-cy(a) },
                             { cz(b)-cz(a), cz(c)-cz(a), cz(d)-cz(a) } };   // columns = edges
    const double det = D[0][0]*(D[1][1]*D[2][2]-D[1][2]*D[2][1]) - D[0][1]*(D[1][0]*D[2][2]-D[1][2]*D[2][0])
                     + D[0][2]*(D[1][0]*D[2][1]-D[1][1]*D[2][0]);
    double inv[3][3];
    inv[0][0] =  (D[1][1]*D[2][2]-D[1][2]*D[2][1]) / det; inv[0][1] = -(D[0][1]*D[2][2]-D[0][2]*D[2][1]) / det;
    inv[0][2] =  (D[0][1]*D[1][2]-D[0][2]*D[1][1]) / det; inv[1][0] = -(D[1][0]*D[2][2]-D[1][2]*D[2][0]) / det;
    inv[1][1] =  (D[0][0]*D[2][2]-D[0][2]*D[2][0]) / det; inv[1][2] = -(D[0][0]*D[1][2]-D[0][2]*D[1][0]) / det;
    inv[2][0] =  (D[1][0]*D[2][1]-D[1][1]*D[2][0]) / det; inv[2][1] = -(D[0][0]*D[2][1]-D[0][1]*D[2][0]) / det;
    inv[2][2] =  (D[0][0]*D[1][1]-D[0][1]*D[1][0]) / det;
    M3 m;   // column-major: m.cK[r] = inv[r][K]
    for (int r = 0; r < 3; ++r)
    {
        m.c0[r] = float(inv[r][0]) + 0.0f; m.c1[r] = float(inv[r][1]) + 0.0f; m.c2[r] = float(inv[r][2]) + 0.0f;
    }
    return m;
}

// The table row of sub-tet i, as the .cu addresses it: ptr[4*i + k + offset].
static void subTetCorners(uint32_t tpe, bool flipped, uint32_t i, uint32_t c[4])
{
    const uint32_t offset = (tpe == 5u && flipped) ? 20u : 0u;
    const uint32_t* ptr = tpe == 5u ? tets5PerVoxel : tets6PerVoxel;
    for (uint32_t k = 0; k < 4; ++k) c[k] = ptr[4u * i + k + offset];
}

// One voxel at element w: pull indices (flipped bit in the MSB of pull[w].x), the rest pose of
// each sub-tet slot w + nbElements*i from the unit voxel, identity rotations, PGS multiplier
// seeds.  `verts[j]` is the vertex of corner j.
static void setVoxel(HexBody& b, uint32_t w, const uint32_t verts[8], bool flipped, bool markFlippedBit,
                     uint32_t matIndex)
{
    b.pull[w] = TetIdx4{ verts[0] | (markFlippedBit ? FLIPPED_MASK : 0u), verts[1], verts[2], verts[3] };
    b.pull[w + b.nbElements] = TetIdx4{ verts[4], verts[5], verts[6], verts[7] };
    for (uint32_t i = 0; i < b.tpe; ++i)
    {
        uint32_t c[4];
        subTetCorners(b.tpe, flipped, i, c);
        const uint32_t slot = w + b.nbElements * i;
        setRest(b, slot, unitRestInverse(c[0], c[1], c[2], c[3]));
        b.rot[slot] = F4{ 0.0f, 0.0f, 0.0f, 1.0f };
        b.mult[mbyzWord(slot) + 0] = f2u(0.013f * float(slot + 1));
        b.mult[mbyzWord(slot) + 1] = f2u(-0.007f * float(slot + 1));
    }
    b.matIdx[b.matLead + w] = uint16_t(matIndex);
}

static const float AFF_S[3] = { 1.5f, 0.75f, 1.25f };   // diag(s): dyadic
static const float AFF_T[3] = { 0.25f, 3.0f, -0.5f };

// Position of lattice point (i,j,k) under the affine seed.
static F4 affinePos(uint32_t i, uint32_t j, uint32_t k, float invMass)
{
    return F4{ AFF_S[0] * float(i) + AFF_T[0], AFF_S[1] * float(j) + AFF_T[1], AFF_S[2] * float(k) + AFF_T[2], invMass };
}

static bool singleOwner(uint32_t corner, bool flipped)
{
    static const bool nf[8] = { false, true, true, false, true, false, false, true };   // {1,7,4,2}
    static const bool f[8]  = { true, false, false, true, false, true, true, false };   // {0,6,5,3}
    return flipped ? f[corner] : nf[corner];
}

// A voxel with 8 PRIVATE vertices [base, base+8) at lattice cell (ox,oy,oz); single-owner
// corners get invMass from `masses` (cycled), the others 0.
static void privateVoxel(HexBody& b, uint32_t w, uint32_t base, uint32_t ox, uint32_t oy, uint32_t oz,
                         bool flipped, bool markFlippedBit, uint32_t matIndex, bool massless)
{
    static const float masses[4] = { 1.0f, 0.5f, 2.0f, 0.25f };
    uint32_t verts[8];
    uint32_t m = 0;
    for (uint32_t j = 0; j < 8; ++j)
    {
        verts[j] = base + j;
        const float im = (!massless && b.tpe == 5u && singleOwner(j, flipped)) ? masses[(m++) & 3u] : 0.0f;
        b.pos[base + j] = affinePos(ox + ((j >> 2) & 1u), oy + ((j >> 1) & 1u), oz + (j & 1u), im);
        b.vel[base + j] = F4{ 0.011f * float(base + j) - 0.05f, -0.013f * float(j) + 0.02f, 0.007f * float(j), im };
    }
    setVoxel(b, w, verts, flipped, markFlippedBit, matIndex);
}

// Non-affine seed for the budget tiers: every corner massive, positions rotated by `th` rad
// and jittered, so the rotation refresh (seeded at the identity) is far from converged.
//
// THE ANGLE IS NOT FREE for a 4-iteration solve.  At 1.9 rad the dot sum that extractRotation
// divides by (|sum R_i . A_i| + eps) is small for some sub-tets, the first omega is several
// radians, and the 4-iteration result is ILL-CONDITIONED: measured on the host oracle alone, a
// 1-ULP nudge of one input position moved Jacobi outputs by up to 127% (2.1e9 ULP), while the
// same nudge moved the 100-iteration result by 160 ULP.  The llvmpipe sin/cos seam is such a
// nudge, so a 4-iteration case at 1.9 rad cannot hold any budget.  The 100-iteration GS case
// keeps 1.9 rad; the 4-iteration Jacobi case uses 0.5 rad, where the same 1-ULP nudge moves
// at most 512 ULP / 4.6e-5 relative (inside the budget) and 4 vs 100 iterations still differ
// by far more than 10x the budget (scanned 0.3..1.3 rad, P9.95f).
static void nonAffineVoxel(HexBody& b, uint32_t w, uint32_t base, uint32_t ox, bool flipped, bool markFlippedBit,
                           double th = 1.9)
{
    static const float masses[8] = { 1.0f, 0.5f, 2.0f, 0.25f, 0.75f, 1.5f, 1.25f, 0.625f };
    // rotation by `th` rad about (1,2,3)/|(1,2,3)|, written out as a matrix (row-major)
    const double ax = 1.0 / std::sqrt(14.0), ay = 2.0 / std::sqrt(14.0), az = 3.0 / std::sqrt(14.0);
    const double cs = std::cos(th), sn = std::sin(th), oc = 1.0 - cs;
    const double R[3][3] = { { cs + ax*ax*oc, ax*ay*oc - az*sn, ax*az*oc + ay*sn },
                             { ay*ax*oc + az*sn, cs + ay*ay*oc, ay*az*oc - ax*sn },
                             { az*ax*oc - ay*sn, az*ay*oc + ax*sn, cs + az*az*oc } };
    uint32_t verts[8];
    for (uint32_t j = 0; j < 8; ++j)
    {
        verts[j] = base + j;
        const double r[3] = { double(ox + ((j >> 2) & 1u)) * 1.3, double((j >> 1) & 1u) * 0.8, double(j & 1u) * 1.1 };
        const double jit[3] = { 0.031 * double((j * 7u) % 5u), -0.027 * double((j * 3u) % 4u), 0.019 * double((j * 5u) % 3u) };
        float p[3];
        for (int a = 0; a < 3; ++a)
            p[a] = float(R[a][0] * r[0] + R[a][1] * r[1] + R[a][2] * r[2] + jit[a]);
        b.pos[base + j] = F4{ p[0], p[1] + 2.0f, p[2], masses[j] };
        b.vel[base + j] = F4{ 0.05f * float(j) - 0.1f, 0.03f, -0.02f * float(j), masses[j] };
    }
    setVoxel(b, w, verts, flipped, markFlippedBit, 0u);
}

struct Stats { uint32_t assocWitness = 0; uint32_t movedCorners = 0; };

// sb_gm_cp_solveSubTetrahedron (softBodyGM.cu:1149-1205)
static void solveSubTetrahedron(HexBody& b, uint32_t workIndex, F4& p0, F4& p1, F4& p2, F4& p3,
                                F4& v0, F4& v1, F4& v2, F4& v3, float invDtSq, float invDt, bool isTGS,
                                bool isFirstIteration, const p80solve::Material& material, Stats& st)
{
    const M3 Qinv = getRest(b, workIndex);
    const F4 q4 = b.rot[workIndex];
    Q4 q = { q4.x, q4.y, q4.z, q4.w };

    const float u1[3] = { p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
    const float u2[3] = { p2.x - p0.x, p2.y - p0.y, p2.z - p0.z };
    const float u3[3] = { p3.x - p0.x, p3.y - p0.y, p3.z - p0.z };
    {
        if ((q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) == 0.0f)
            q = b.initialRotation;
        M3 P; for (int i = 0; i < 3; ++i) { P.c0[i] = u1[i]; P.c1[i] = u2[i]; P.c2[i] = u3[i]; }
        M3 F; p80solve::mat33Mul(P, Qinv, F);
        q = p80rot::extractRotation(F, q, isFirstIteration ? 100 : 4);
        b.rot[workIndex] = F4{ q.x, q.y, q.z, q.w };
    }

    bool zero = true;
    for (int i = 0; i < 3; ++i)
        zero = zero && Qinv.c0[i] == 0.0f && Qinv.c1[i] == 0.0f && Qinv.c2[i] == 0.0f;
    if (zero)
        return;

    float mb[2] = { u2f(b.mult[mbyzWord(workIndex)]), u2f(b.mult[mbyzWord(workIndex) + 1]) };
    M3 dl;
    p80solve::solveGM(dl, u1, u2, u3, v0, v1, v2, v3, mb, invDtSq, invDt, Qinv, q, isTGS, material);
    b.mult[mbyzWord(workIndex)] = f2u(mb[0]);
    b.mult[mbyzWord(workIndex) + 1] = f2u(mb[1]);

    float sum[3];
    for (int i = 0; i < 3; ++i)
    {
        sum[i] = dl.c0[i] + dl.c1[i] + dl.c2[i];   // left-associated, as PxVec3 evaluates it
        if (p0.w != 0.0f && f2u(sum[i]) != f2u(dl.c0[i] + (dl.c1[i] + dl.c2[i]))) ++st.assocWitness;
    }
    float dp[3];
    for (int i = 0; i < 3; ++i) dp[i] = -sum[i] * p0.w;
    p0.x += dp[0]; p0.y += dp[1]; p0.z += dp[2]; p0.w += 0.0f;
    v0.x += dp[0] * invDt; v0.y += dp[1] * invDt; v0.z += dp[2] * invDt; v0.w += 0.0f;
    const float* cols[3] = { dl.c0, dl.c1, dl.c2 };
    F4* ps[3] = { &p1, &p2, &p3 };
    F4* vs[3] = { &v1, &v2, &v3 };
    for (int k = 0; k < 3; ++k)
    {
        F4& p = *ps[k]; F4& v = *vs[k];
        float d[3];
        for (int i = 0; i < 3; ++i) d[i] = cols[k][i] * p.w;
        p.x += d[0]; p.y += d[1]; p.z += d[2]; p.w += 0.0f;
        v.x += d[0] * invDt; v.y += d[1] * invDt; v.z += d[2] * invDt; v.w += 0.0f;
    }
}

// sb_gm_cp_solveHexahedronsPartition_parallelGS<updatePosVel> (softBodyGM.cu:1313-1404)
static void solveHexahedron(HexBody& b, uint32_t workIndex, bool updatePosVel, float invDt, bool isTGS,
                            bool isFirstIteration, const std::vector<p80solve::Material>& materials, Stats& st)
{
    const float invDtSq = invDt * invDt;
    const uint32_t nbElements = b.nbTets() / b.tpe;
    const uint16_t globalMaterialIndex = b.matIdx[b.matLead + workIndex];
    const p80solve::Material& material = materials[globalMaterialIndex];

    TetIdx4 pullInd[2] = { b.pull[workIndex], b.pull[workIndex + nbElements] };
    const bool flipped = (pullInd[0].x & FLIPPED_MASK) != 0u;
    pullInd[0].x &= PULL_IND_MASK;
    const uint32_t idx[8] = { pullInd[0].x, pullInd[0].y, pullInd[0].z, pullInd[0].w,
                              pullInd[1].x, pullInd[1].y, pullInd[1].z, pullInd[1].w };
    F4 p[8], v[8];
    for (uint32_t i = 0; i < 8; ++i) { p[i] = b.pos[idx[i]]; v[i] = b.vel[idx[i]]; }

    uint32_t offset = 0;
    if (b.tpe == 5u && flipped)
        offset = 20;
    const uint32_t* ptr = b.tpe == 5u ? tets5PerVoxel : tets6PerVoxel;
    for (uint32_t i = 0; i < b.tpe; ++i)
    {
        const uint32_t a = ptr[4*i + 0 + offset], c1 = ptr[4*i + 1 + offset];
        const uint32_t c2 = ptr[4*i + 2 + offset], d = ptr[4*i + 3 + offset];
        solveSubTetrahedron(b, workIndex + nbElements * i, p[a], p[c1], p[c2], p[d], v[a], v[c1], v[c2], v[d],
                            invDtSq, invDt, isTGS, isFirstIteration, material, st);
    }

    if (updatePosVel)
    {
        for (uint32_t i = 0; i < 8; ++i)
        {
            const uint32_t index = idx[i];
            F4& dp = b.deltaPos[index];
            const F4 old = b.pos[index];
            dp.x += p[i].x - old.x; dp.y += p[i].y - old.y; dp.z += p[i].z - old.z; dp.w += p[i].w - old.w;
            b.pos[index] = p[i];
            b.vel[index] = v[i];
        }
    }
    else
    {
        for (uint32_t i = 0; i < 8; ++i)
        {
            const uint32_t index = idx[i];
            const F4 old = b.pos[index];
            const F4 delta4 = { p[i].x - old.x, p[i].y - old.y, p[i].z - old.z, p[i].w - old.w };
            if (f2u(delta4.x) != 0u || f2u(delta4.y) != 0u || f2u(delta4.z) != 0u) ++st.movedCorners;
            F4& d = b.delta[index];
            d.x += delta4.x; d.y += delta4.y; d.z += delta4.z; d.w += 1.0f;
        }
    }
}

// The Jacobi kernel (softBodyGM.cu:1463-1507) over one body.
static void jacobiKernel(HexBody& b, float invDt, bool isTGS, const std::vector<p80solve::Material>& mats,
                         Stats& st, bool isFirstIteration = false)
{
    if (b.numPartitions <= SB_PARTITION_LIMIT || b.tpe == 1u) return;
    if (b.bodyFlags & PxDeformableBodyFlag::eKINEMATIC) return;
    for (uint32_t w = b.acc[SB_PARTITION_LIMIT - 1]; w < b.acc[SB_PARTITION_LIMIT]; ++w)
        solveHexahedron(b, w, /*updatePosVel*/ false, invDt, isTGS, isFirstIteration, mats, st);
}

// The GS kernel's hex branch (softBodyGM.cu:1416-1457) over one body and one partition.
static void gsKernel(HexBody& b, uint32_t partitionId, float invDt, bool isTGS, bool isFirstIteration,
                     const std::vector<p80solve::Material>& mats, Stats& st)
{
    if (b.bodyFlags & PxDeformableBodyFlag::eKINEMATIC) return;
    if (!(b.nbTets() > 0u && partitionId < b.numPartitions)) return;
    const uint32_t start = partitionId > 0u ? b.acc[partitionId - 1] : 0u, end = b.acc[partitionId];
    for (uint32_t w = start; w < end; ++w)
        solveHexahedron(b, w, /*updatePosVel*/ true, invDt, isTGS, isFirstIteration, mats, st);
}

// ---- device side ----------------------------------------------------------
struct DevHex { DevBuf pos, vel, delta, deltaPos, cpPos, cpVel, remap, pull, rest, rot, mult, matIdx, acc; };

static PxgSoftBody uploadBody(PxgSlangComputeDevice* dev, const HexBody& b, DevHex& d)
{
    d.pos = makeDev(dev, b.pos); d.vel = makeDev(dev, b.vel);
    d.delta = makeDev(dev, b.delta); d.deltaPos = makeDev(dev, b.deltaPos);
    d.cpPos = makeDev(dev, b.cpPos); d.cpVel = makeDev(dev, b.cpVel);
    d.remap = makeDev(dev, b.remap); d.pull = makeDev(dev, b.pull);
    d.rest = makeDev(dev, b.rest); d.rot = makeDev(dev, b.rot); d.mult = makeDev(dev, b.mult);
    d.matIdx = makeDev(dev, b.matIdx); d.acc = makeDev(dev, b.acc);
    PxgSoftBody c = makeZeroDesc();
    c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(d.pos.addr));
    c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(d.vel.addr));
    c.mSimDelta = reinterpret_cast<float4*>(size_t(d.delta.addr));
    c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(d.deltaPos.addr));
    c.mSimPosition_InvMassCP = reinterpret_cast<float4*>(size_t(d.cpPos.addr));
    c.mSimVelocity_invMassCP = reinterpret_cast<float4*>(size_t(d.cpVel.addr));
    c.mSimRemapOutputCP = reinterpret_cast<decltype(c.mSimRemapOutputCP)>(size_t(d.remap.addr));
    c.mSimPullIndices = reinterpret_cast<decltype(c.mSimPullIndices)>(size_t(d.pull.addr));
    c.mSimTetraRestPoses = reinterpret_cast<decltype(c.mSimTetraRestPoses)>(size_t(d.rest.addr));
    c.mSimTetraRotations = reinterpret_cast<float4*>(size_t(d.rot.addr));
    c.mSimTetraMultipliers = reinterpret_cast<decltype(c.mSimTetraMultipliers)>(size_t(d.mult.addr));
    c.mOrderedMaterialIndices = reinterpret_cast<PxU16*>(size_t(d.matIdx.addr + 2u * b.matLead));
    c.mSimAccumulatedPartitionsCP = reinterpret_cast<PxU32*>(size_t(d.acc.addr));
    c.mNumTetsGM = b.nbTets();
    c.mNumPartitionsGM = b.numPartitions;
    c.mNumTetsPerElement = b.tpe;
    c.mNumVertsGM = b.nbVerts;
    c.mBodyFlags = PxU8(b.bodyFlags);
    c.mInitialRotation = PxQuat(b.initialRotation.x, b.initialRotation.y, b.initialRotation.z, b.initialRotation.w);
    return c;
}

static HexBody downloadBody(PxgSlangComputeDevice* dev, const HexBody& shape, const DevHex& d)
{
    HexBody g = shape;
    g.pos = downloadBuffer<F4>(dev, d.pos.h, shape.pos.size());
    g.vel = downloadBuffer<F4>(dev, d.vel.h, shape.vel.size());
    g.delta = downloadBuffer<F4>(dev, d.delta.h, shape.delta.size());
    g.deltaPos = downloadBuffer<F4>(dev, d.deltaPos.h, shape.deltaPos.size());
    g.cpPos = downloadBuffer<F4>(dev, d.cpPos.h, shape.cpPos.size());
    g.cpVel = downloadBuffer<F4>(dev, d.cpVel.h, shape.cpVel.size());
    g.remap = downloadBuffer<TetIdx4>(dev, d.remap.h, shape.remap.size());
    g.pull = downloadBuffer<TetIdx4>(dev, d.pull.h, shape.pull.size());
    g.rest = downloadBuffer<uint32_t>(dev, d.rest.h, shape.rest.size());
    g.rot = downloadBuffer<F4>(dev, d.rot.h, shape.rot.size());
    g.mult = downloadBuffer<uint32_t>(dev, d.mult.h, shape.mult.size());
    g.matIdx = downloadBuffer<uint16_t>(dev, d.matIdx.h, shape.matIdx.size());
    g.acc = downloadBuffer<uint32_t>(dev, d.acc.h, shape.acc.size());
    return g;
}

static void destroyBody(PxgSlangComputeDevice* dev, DevHex& d)
{
    const DevBuf* all[] = { &d.pos, &d.vel, &d.delta, &d.deltaPos, &d.cpPos, &d.cpVel, &d.remap, &d.pull,
                            &d.rest, &d.rot, &d.mult, &d.matIdx, &d.acc };
    for (const DevBuf* x : all) dev->destroyBuffer(x->h);
}

// Every buffer of `got` against `want`.  budgetUlp == 0: bitwise.  Otherwise a float lane
// mismatches only beyond BOTH budgetUlp ULP and budgetRel relative (the inherited sin/cos
// seam), and the integer buffers stay exact.  Returns the mismatch count, records the worst.
struct CmpResult { size_t mm = 0; uint32_t worstUlp = 0; double worstRel = 0.0; };
static CmpResult compareBody(const char* tag, const HexBody& want, const HexBody& got,
                             uint32_t budgetUlp = 0u, double budgetRel = 0.0, bool quiet = false)
{
    CmpResult r;
    auto lane = [&](const char* what, size_t i, int e, float a, float g) {
        if (f2u(a) == f2u(g)) return;
        bool bad = true;
        if (budgetUlp != 0u)
        {
            const uint32_t u = p80rot::ulpDist(a, g);
            const double rel = std::fabs(double(a) - double(g)) / std::fmax(1.0e-6, std::fabs(double(a)));
            if (std::isfinite(a) && std::isfinite(g))
            {
                if (u > r.worstUlp) r.worstUlp = u;
                if (rel > r.worstRel) r.worstRel = rel;
                bad = u > budgetUlp && rel > budgetRel;
            }
        }
        if (bad)
        {
            ++r.mm;
            if (r.mm <= 12 && !quiet)
                std::printf("  [%s] %s[%zu].%d MM: want %08x (%.9g) got %08x (%.9g)\n", tag, what, i, e,
                            f2u(a), a, f2u(g), g);
        }
    };
    auto f4s = [&](const char* what, const std::vector<F4>& a, const std::vector<F4>& g) {
        for (size_t i = 0; i < a.size(); ++i)
        { lane(what, i, 0, a[i].x, g[i].x); lane(what, i, 1, a[i].y, g[i].y);
          lane(what, i, 2, a[i].z, g[i].z); lane(what, i, 3, a[i].w, g[i].w); } };
    auto words = [&](const char* what, const std::vector<uint32_t>& a, const std::vector<uint32_t>& g, bool asFloat) {
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (asFloat) lane(what, i, 0, u2f(a[i]), u2f(g[i]));
            else if (a[i] != g[i]) { ++r.mm; if (r.mm <= 12) std::printf("  [%s] %s[%zu] MM: want %08x got %08x\n", tag, what, i, a[i], g[i]); }
        } };
    auto idx4 = [&](const char* what, const std::vector<TetIdx4>& a, const std::vector<TetIdx4>& g) {
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].x != g[i].x || a[i].y != g[i].y || a[i].z != g[i].z || a[i].w != g[i].w)
            { ++r.mm; if (r.mm <= 12) std::printf("  [%s] %s[%zu] MM\n", tag, what, i); } };
    f4s("pos", want.pos, got.pos);
    f4s("vel", want.vel, got.vel);
    f4s("delta", want.delta, got.delta);
    f4s("deltaPos", want.deltaPos, got.deltaPos);
    f4s("cpPos", want.cpPos, got.cpPos);
    f4s("cpVel", want.cpVel, got.cpVel);
    f4s("rot", want.rot, got.rot);
    words("mult", want.mult, got.mult, /*asFloat*/ true);
    words("rest", want.rest, got.rest, false);
    words("acc", want.acc, got.acc, false);
    idx4("pull", want.pull, got.pull);
    idx4("remap", want.remap, got.remap);
    for (size_t i = 0; i < want.matIdx.size(); ++i)
        if (want.matIdx[i] != got.matIdx[i]) { ++r.mm; std::printf("  [%s] matIdx[%zu] MM\n", tag, i); }
    return r;
}

// PxsDeformableVolumeMaterialData words (32-byte stride) for a list of materials.
static std::vector<uint32_t> materialWords(const std::vector<p80solve::Material>& mats)
{
    std::vector<uint32_t> w(8u * mats.size() + 64u, 0u);
    for (size_t m = 0; m < mats.size(); ++m)
    {
        w[8*m + 0] = f2u(mats[m].youngs);
        w[8*m + 1] = f2u(mats[m].poissons);
        w[8*m + 2] = f2u(0.0f);
        w[8*m + 3] = f2u(mats[m].elasticityDamping);
        w[8*m + 4] = mats[m].model | (0xBEEFu << 16);   // 5.12: materialModel @16, poisoned _pad @18
    }
    return w;
}

static p80solve::Material coRot(float damping = 0.0f)
{ return p80solve::Material{ 1.0e5f, 0.31f, damping, 0u }; }
static p80solve::Material neoHookean()
{ return p80solve::Material{ 2.0e5f, 0.27f, 0.0f, 1u }; }

// Runs the Jacobi entry over `bodies` (descriptor index = position in the vector) in the
// order `active`, with the host grid (ceil(maxJacobi/32), nbActive, 1).
static void dispatchJacobi(PxgSlangComputeDevice* dev, std::vector<HexBody>& bodies, std::vector<DevHex>& devs,
                           const std::vector<uint32_t>& active, uint32_t maxJacobi, bool isTGS, float invDt,
                           const std::vector<p80solve::Material>& mats)
{
    std::vector<PxgSoftBody> descs;
    devs.resize(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) descs.push_back(uploadBody(dev, bodies[i], devs[i]));
    DevBuf bDesc = makeDev(dev, descs);
    DevBuf bActive = makeDev(dev, active);
    DevBuf bMat = makeDev(dev, materialWords(mats));
    SolveTetraJacobiPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.numActiveSoftbodies = 0xDEAD0000u;      // never read (dead upstream too)
    push.invDt = invDt;
    push.flagsWord = (isTGS ? 1u : 0u) | 0xA5C3E100u;   // pad bytes 25..27 poisoned (trap H8)
    push.pad28 = 0xDEADBEEFu;
    push.materials = bMat.addr;
    Kernel k("softBodySolveBda.spv", "sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch", {}, sizeof(SolveTetraJacobiPush));
    k.dispatch((maxJacobi + 31u) / 32u, {}, push, uint32_t(active.size()), 1u);
    dev->flush();
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h); dev->destroyBuffer(bMat.h);
}

// Runs the GS entry for one partition, host grid (ceil(maxPerPartition/32), nbActive, 1).
static void dispatchGS(PxgSlangComputeDevice* dev, std::vector<HexBody>& bodies, std::vector<DevHex>& devs,
                       const std::vector<uint32_t>& active, uint32_t maxPerPartition, uint32_t partitionId,
                       bool isTGS, bool isFirstIteration, float invDt, const std::vector<p80solve::Material>& mats,
                       bool upload = true)
{
    std::vector<PxgSoftBody> descs;
    if (upload) devs.resize(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i)
    {
        if (upload) descs.push_back(uploadBody(dev, bodies[i], devs[i]));
        else
        {
            // re-describe the already uploaded buffers (a second dispatch over the same state)
            DevHex tmp;
            PxgSoftBody c = uploadBody(dev, bodies[i], tmp);
            destroyBody(dev, tmp);
            const DevHex& d = devs[i];
            c.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(d.pos.addr));
            c.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(d.vel.addr));
            c.mSimDelta = reinterpret_cast<float4*>(size_t(d.delta.addr));
            c.mSimDeltaPos = reinterpret_cast<float4*>(size_t(d.deltaPos.addr));
            c.mSimPosition_InvMassCP = reinterpret_cast<float4*>(size_t(d.cpPos.addr));
            c.mSimVelocity_invMassCP = reinterpret_cast<float4*>(size_t(d.cpVel.addr));
            c.mSimRemapOutputCP = reinterpret_cast<decltype(c.mSimRemapOutputCP)>(size_t(d.remap.addr));
            c.mSimPullIndices = reinterpret_cast<decltype(c.mSimPullIndices)>(size_t(d.pull.addr));
            c.mSimTetraRestPoses = reinterpret_cast<decltype(c.mSimTetraRestPoses)>(size_t(d.rest.addr));
            c.mSimTetraRotations = reinterpret_cast<float4*>(size_t(d.rot.addr));
            c.mSimTetraMultipliers = reinterpret_cast<decltype(c.mSimTetraMultipliers)>(size_t(d.mult.addr));
            c.mOrderedMaterialIndices = reinterpret_cast<PxU16*>(size_t(d.matIdx.addr + 2u * bodies[i].matLead));
            c.mSimAccumulatedPartitionsCP = reinterpret_cast<PxU32*>(size_t(d.acc.addr));
            descs.push_back(c);
        }
    }
    DevBuf bDesc = makeDev(dev, descs);
    DevBuf bActive = makeDev(dev, active);
    DevBuf bMat = makeDev(dev, materialWords(mats));
    SolveTetraPush push = {};
    push.gSoftbodies = bDesc.addr;
    push.activeSoftbodies = bActive.addr;
    push.numActiveSoftbodies = uint32_t(active.size());
    push.invDt = invDt;
    push.partitionId = partitionId;
    push.flagsWord = (isTGS ? 1u : 0u) | (isFirstIteration ? 0x100u : 0u);
    push.materials = bMat.addr;
    Kernel k("softBodySolveBda.spv", "sb_gm_cp_solveTetrahedronsPartitionLaunch", {}, sizeof(SolveTetraPush));
    k.dispatch((maxPerPartition + 31u) / 32u, {}, push, uint32_t(active.size()), 1u);
    dev->flush();
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h); dev->destroyBuffer(bMat.h);
}

// A body descriptor that the Jacobi guard rejects (0 partitions): occupies descriptor slot 0
// so the live body is never at groupId.y.
static HexBody idleBody()
{
    HexBody b;
    initBody(b, 5u, 1u, 8u);
    b.numPartitions = 0u;
    b.acc.assign(16u, 0u);
    return b;
}

static bool sameBits(const std::vector<F4>& a, const std::vector<F4>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (f2u(a[i].x) != f2u(b[i].x) || f2u(a[i].y) != f2u(b[i].y) || f2u(a[i].z) != f2u(b[i].z) ||
            f2u(a[i].w) != f2u(b[i].w)) return false;
    return true;
}

static bool isIdentityBits(const F4& q)
{ return f2u(q.x) == 0u && f2u(q.y) == 0u && f2u(q.z) == 0u && f2u(q.w) == 0x3f800000u; }
} // namespace p95hex

// ---------------------------------------------------------------------------
// The two-voxel Jacobi body: elements 0,1 are GS elements (partitions 0 and 7), elements 2
// (A, non-flipped) and 3 (B, flipped, FLIPPED bit in pull[3].x) are the Jacobi partition
// [acc[7], acc[8]) = [2, 4).  B sits on A's +x face: A corner j+4 == B corner j, on a 3x2x2
// lattice whose vertex (i,j,k) is (i*2 + j)*2 + k, so A = verts 0..7, B = verts 4..11.
// acc = {1,1,1,1,1,1,1,2,4}: acc[6] == 1 != acc[7], so a range that starts one partition
// early reaches element 1, whose slots are poison (review D1).
// ---------------------------------------------------------------------------
static p95hex::HexBody makeJacobi5Body()
{
    using namespace p95hex;
    HexBody b;
    initBody(b, 5u, 4u, 12u);
    b.acc = { 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 4u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 9u;
    // inverse masses: single-owner corners only; A4 == B0 and A7 == B3 are single-owner in both
    const float im[12] = { 0.0f, 1.0f, 0.5f, 0.0f, 2.0f, 0.0f, 0.0f, 0.25f, 0.0f, 1.5f, 0.75f, 0.0f };
    for (uint32_t i = 0; i < 3; ++i)
        for (uint32_t j = 0; j < 2; ++j)
            for (uint32_t k = 0; k < 2; ++k)
            {
                const uint32_t v = (i * 2u + j) * 2u + k;
                b.pos[v] = affinePos(i, j, k, im[v]);
                b.vel[v] = F4{ 0.021f * float(v) - 0.1f, 0.017f * float(v), -0.009f * float(v) + 0.03f, im[v] };
            }
    const uint32_t A[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const uint32_t B[8] = { 4, 5, 6, 7, 8, 9, 10, 11 };
    setVoxel(b, 2u, A, /*flipped*/ false, false, 0u);
    setVoxel(b, 3u, B, /*flipped*/ true, /*bit*/ true, 0u);
    // mSimDelta: 0 on the two-contribution verts (4..7); seeds on one-contribution verts:
    // 0.25-family (store-instead-of-add) and -0 (the -0 + +0 = +0 probe).
    for (uint32_t v = 0; v < 12; ++v) b.delta[v] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    b.delta[1] = F4{ 0.25f, -0.125f, 0.0625f, 3.0f };
    b.delta[2] = F4{ -0.0f, -0.0f, -0.0f, -0.0f };
    b.delta[0] = F4{ -0.0f, 0.5f, -0.0f, 1.0f };
    b.delta[9] = F4{ 0.25f, 0.25f, 0.25f, 0.0f };
    b.delta[11] = F4{ -0.0f, -0.0f, -0.0f, 2.0f };
    // the GS elements 0 and 1 gather the sentinel vertex; their slots stay poison
    return b;
}

static void runHexJacobi5(bool isTGS)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot() };
    std::vector<HexBody> bodies = { idleBody(), makeJacobi5Body() };
    const HexBody upload = bodies[1];
    std::vector<DevHex> devs;
    dispatchJacobi(dev, bodies, devs, { 1u }, /*maxJacobi*/ 2u, isTGS, invDt, mats);

    HexBody want = upload;
    Stats st;
    jacobiKernel(want, invDt, isTGS, mats, st);
    const HexBody got = downloadBody(dev, upload, devs[1]);

    // Preconditions of the bitwise construction, checked on the ORACLE: it must have moved
    // corners (a non-trivial solve), produced an identity rotation everywhere it wrote, and
    // -- review D5 -- contain at least one mobile corner-0 lane where the association of
    // c0 + c1 + c2 matters, or the sum-order mutant would be invisible.
    PXG_CHECK_MSG(st.movedCorners >= 6u, "the Jacobi oracle must move the mobile corners (%u)", st.movedCorners);
    PXG_CHECK_MSG(st.assocWitness > 0u, "no mobile corner-0 lane separates (c0+c1)+c2 from c0+(c1+c2)");
    std::printf("  [hexJacobi5 %s] oracle: %u moved corner deltas, %u association-witness lanes\n",
                isTGS ? "TGS" : "PGS", st.movedCorners, st.assocWitness);
    for (uint32_t w = 2; w < 4; ++w)
        for (uint32_t i = 0; i < 5; ++i)
            PXG_CHECK_MSG(isIdentityBits(want.rot[w + 4u * i]), "oracle rotation slot %u not the exact identity", w + 4u * i);
    // The w lane counts contributions: 2 on the shared face, 1 elsewhere (plus the seed).
    PXG_CHECK_MSG(want.delta[4].w == 2.0f && want.delta[1].w == 4.0f && want.delta[11].w == 3.0f,
                  "oracle w-counts (%g %g %g)", want.delta[4].w, want.delta[1].w, want.delta[11].w);

    const CmpResult r = compareBody(isTGS ? "hexJacobi5TGS" : "hexJacobi5", want, got);
    PXG_CHECK_MSG(r.mm == 0, "sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch bit-exact vs the CUDA transcription (%zu)", r.mm);
    // Explicit teeth, in case the oracle and the shader agreed on a vacuous state:
    PXG_CHECK_MSG(f2u(got.rot[0].x) == POISON && f2u(got.rot[1].x) == POISON,
                  "the GS elements' rotation slots (0, 1) must keep their poison");
    PXG_CHECK_MSG(f2u(got.rot[4u * 4u + 4u].x) == POISON, "the slot past the last element must keep its poison");
    PXG_CHECK_MSG(f2u(got.pos[4].x) == f2u(upload.pos[4].x) && f2u(got.deltaPos[4].x) == f2u(upload.deltaPos[4].x),
                  "the Jacobi kernel must not write positions or mSimDeltaPos");
    if (isTGS)
        PXG_CHECK_MSG(got.mult == upload.mult, "TGS must not store multipliers");
    else
        PXG_CHECK_MSG(got.mult[mbyzWord(2)] != upload.mult[mbyzWord(2)], "PGS must store the multipliers");
    for (auto& d : devs) destroyBody(dev, d);
}

// 70 Jacobi elements on one body (3 workgroups of 32): massless private voxels, so the
// case is exact and every vertex must count exactly one contribution -- the per-group
// stride (groupId.x * 32) is the thing under test.
static void runHexJacobiWide()
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot() };
    const uint32_t nbE = 70u;
    HexBody b;
    initBody(b, 5u, nbE, nbE * 8u);
    b.acc = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, nbE };
    b.acc.resize(16u, POISON);
    b.numPartitions = 9u;
    for (uint32_t w = 0; w < nbE; ++w)
    {
        privateVoxel(b, w, w * 8u, (w % 7u) * 2u, (w / 7u) % 5u, w / 35u, (w & 1u) != 0u, (w & 1u) != 0u, 0u,
                     /*massless*/ true);
        for (uint32_t j = 0; j < 8; ++j) b.delta[w * 8u + j] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    std::vector<HexBody> bodies = { idleBody(), b };
    const HexBody upload = b;
    std::vector<DevHex> devs;
    dispatchJacobi(dev, bodies, devs, { 1u }, nbE, /*isTGS*/ false, invDt, mats);
    HexBody want = upload;
    Stats st;
    jacobiKernel(want, invDt, false, mats, st);
    const HexBody got = downloadBody(dev, upload, devs[1]);
    uint32_t badCount = 0;
    for (uint32_t v = 0; v < nbE * 8u; ++v)
        if (got.delta[v].w != 1.0f) ++badCount;
    PXG_CHECK_MSG(badCount == 0u, "70-element Jacobi: %u vertices do not count exactly one contribution", badCount);
    const CmpResult r = compareBody("hexJacobiWide", want, got);
    PXG_CHECK_MSG(r.mm == 0, "70-element Jacobi dispatch (3 groups) bit-exact (%zu)", r.mm);
    for (auto& d : devs) destroyBody(dev, d);
}

PXG_TEST(softBodySimHexJacobi5) { runHexJacobi5(/*isTGS*/ false); runHexJacobiWide(); }
PXG_TEST(softBodySimHexJacobi5TGS) { runHexJacobi5(/*isTGS*/ true); }

// ---------------------------------------------------------------------------
// The two early-outs (softBodyGM.cu:1482, :1485) in ONE dispatch of four bodies, listed
// through a permuted activeSoftbodies: (i) hex with 8 partitions, (ii) tet (tpe 1) with 9,
// (iii) hex with 9 but eKINEMATIC, (iv) live.  (i)-(iii) carry valid, non-empty acc[7..8]
// ranges and full buffers, so a dropped guard does real (and visible) work.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimHexJacobiEarlyOuts)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot() };
    std::vector<HexBody> bodies(4, makeJacobi5Body());
    bodies[0].numPartitions = 8u;                                  // (i)
    bodies[1].tpe = 1u; bodies[1].nbElements = 20u;                // (ii) same 20 tets, one per element
    bodies[2].bodyFlags = uint8_t(PxDeformableBodyFlag::eKINEMATIC);   // (iii)
    // (iv) = bodies[3], live
    const std::vector<HexBody> upload = bodies;
    std::vector<DevHex> devs;
    dispatchJacobi(dev, bodies, devs, { 2u, 0u, 3u, 1u }, /*maxJacobi*/ 2u, /*isTGS*/ false, invDt, mats);
    size_t mm = 0;
    for (uint32_t i = 0; i < 4; ++i)
    {
        HexBody want = upload[i];
        Stats st;
        jacobiKernel(want, invDt, false, mats, st);
        PXG_CHECK_MSG((i == 3u) == (st.movedCorners > 0u), "oracle: only body (iv) may do work (body %u)", i);
        const HexBody got = downloadBody(dev, upload[i], devs[i]);
        static const char* tags[4] = { "earlyOut(i) 8 partitions", "earlyOut(ii) tet", "earlyOut(iii) kinematic",
                                       "earlyOut(iv) live" };
        mm += compareBody(tags[i], want, got).mm;
    }
    PXG_CHECK_MSG(mm == 0, "Jacobi early-outs: (i)-(iii) untouched, (iv) solved (%zu)", mm);
    for (auto& d : devs) destroyBody(dev, d);
}

// ---------------------------------------------------------------------------
// The degenerate rest pose (Qinv == 0, softBodyGM.cu:1176-1177): the rotation is STILL
// stored, then the sub-tet returns before p/v are touched.  Review D3: in a Jacobi tail the
// difference between "returned" and "added a zero delta" is invisible, so the case has two
// halves --
//   (a) Jacobi entry, PGS, NEO-HOOKEAN: the degenerate slot's multiplier must keep its seed
//       (the neo-Hookean multiplier moves even at Qinv == 0, the co-rotational one does not)
//       and its rotation, seeded with the non-unit (0,0,0,2), must come back normalised to
//       the identity (stored before the return).  A second slot starts at q == (0,0,0,0)
//       and must be seeded from mInitialRotation, set to the non-identity (1,0,0,0) in
//       both halves.
//   (b) GS entry, co-rotational: the degenerate sub-tet's single-owner corner 1 carries
//       p.w == v.w == -0; returned early it is written back verbatim (-0), while a sub-tet
//       that ran would have turned both into +0.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimHexJacobiDegenerate)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    for (int half = 0; half < 2; ++half)
    {
        const bool jacobi = half == 0;
        const std::vector<p80solve::Material> mats = { jacobi ? neoHookean() : coRot() };
        HexBody b;
        initBody(b, 5u, 1u, 8u);
        b.acc = jacobi ? std::vector<uint32_t>{ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u }
                       : std::vector<uint32_t>{ 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u };
        b.acc.resize(16u, POISON);
        b.numPartitions = jacobi ? 9u : 8u;
        privateVoxel(b, 0u, 0u, 0u, 0u, 0u, false, false, 0u, false);
        // A NON-identity mInitialRotation: every other hex fixture uses the identity, so a
        // component-order slip in the float4 sbHexDescOf builds from the four scalars @348..@360
        // would otherwise go unseen (an X/Y swap survived all 12 fixtures).  (1,0,0,0), 180 deg
        // about X, is an exact extractRotation fixed point under the affine (diagonal F) seed:
        // omega == 0 on the first iteration, so slot 3 comes back as exactly (1,0,0,0).
        b.initialRotation = Q4{ 1.0f, 0.0f, 0.0f, 0.0f };
        for (uint32_t j = 0; j < 8; ++j) b.delta[j] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
        setRest(b, 1u, M3{});                         // sub-tet row 1 (corners 0,1,5,3): Qinv = +0
        b.rot[1u] = F4{ 0.0f, 0.0f, 0.0f, 2.0f };     // non-unit: the store normalises it
        b.rot[3u] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };     // seeded from mInitialRotation
        if (!jacobi)
        {
            b.pos[1].w = -0.0f; b.vel[1].w = -0.0f;   // corner 1 is in row 1 only
            for (uint32_t j = 0; j < 8; ++j) b.deltaPos[j] = F4{ 0.125f, -0.25f, 0.5f, 0.0f };
        }
        const HexBody upload = b;
        std::vector<HexBody> bodies = { idleBody(), b };
        std::vector<DevHex> devs;
        if (jacobi) dispatchJacobi(dev, bodies, devs, { 1u }, 1u, /*isTGS*/ false, invDt, mats);
        else dispatchGS(dev, bodies, devs, { 1u }, 1u, /*partition*/ 0u, false, /*first*/ true, invDt, mats);
        HexBody want = upload;
        Stats st;
        if (jacobi) jacobiKernel(want, invDt, false, mats, st);
        else gsKernel(want, 0u, invDt, false, true, mats, st);
        const HexBody got = downloadBody(dev, upload, devs[1]);
        const CmpResult r = compareBody(jacobi ? "hexDegenerate(a) Jacobi" : "hexDegenerate(b) GS", want, got);
        PXG_CHECK_MSG(r.mm == 0, "degenerate rest pose (%s) bit-exact (%zu)", jacobi ? "Jacobi" : "GS", r.mm);
        PXG_CHECK_MSG(isIdentityBits(want.rot[1]) && isIdentityBits(got.rot[1]),
                      "the degenerate slot's rotation is stored (normalised to identity) before the return");
        PXG_CHECK_MSG(f2u(got.rot[3].x) == 0x3f800000u && f2u(got.rot[3].y) == 0u &&
                      f2u(got.rot[3].z) == 0u && f2u(got.rot[3].w) == 0u,
                      "q == 0 must be seeded from mInitialRotation (1,0,0,0) (got %08x %08x %08x %08x)",
                      f2u(got.rot[3].x), f2u(got.rot[3].y), f2u(got.rot[3].z), f2u(got.rot[3].w));
        if (jacobi)
            PXG_CHECK_MSG(got.mult[mbyzWord(1)] == upload.mult[mbyzWord(1)] &&
                          got.mult[mbyzWord(2)] != upload.mult[mbyzWord(2)],
                          "the degenerate slot's multiplier keeps its seed, a live slot's moves");
        else
            PXG_CHECK_MSG(f2u(got.pos[1].w) == 0x80000000u && f2u(got.vel[1].w) == 0x80000000u &&
                          f2u(got.pos[0].w) == 0u,
                          "corner 1 (only in the degenerate sub-tet) keeps -0 in w (pos %08x vel %08x); "
                          "corner 0 is +0", f2u(got.pos[1].w), f2u(got.vel[1].w));
        for (auto& d : devs) destroyBody(dev, d);
    }
}

// ---------------------------------------------------------------------------
// mOrderedMaterialIndices[w] is a PxU16 read through its containing word, with the half
// chosen by the ADDRESS (P9.88).  Two Jacobi voxels, material 0 (co-rotational, damped) and
// 1 (neo-Hookean), with the array based at an address == 2 (mod 4): element 0 reads the HIGH
// half, element 1 the LOW half of the next word; the neighbouring halves hold the OTHER
// index, so either half rule mistaken picks the wrong material.
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimHexJacobiMaterialHalf)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot(0.35f), neoHookean() };
    HexBody b;
    initBody(b, 5u, 2u, 16u);
    b.acc = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 2u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 9u;
    b.matLead = 1u;
    privateVoxel(b, 0u, 0u, 0u, 0u, 0u, false, false, 0u, false);
    privateVoxel(b, 1u, 8u, 3u, 0u, 0u, true, true, 1u, false);
    b.matIdx[0] = 1u;       // the low half of element 0's word
    b.matIdx[3] = 0u;       // the high half of element 1's word
    for (uint32_t v = 0; v < 16; ++v) b.delta[v] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    const HexBody upload = b;
    std::vector<HexBody> bodies = { idleBody(), b };
    std::vector<DevHex> devs;
    dispatchJacobi(dev, bodies, devs, { 1u }, 2u, false, invDt, mats);
    HexBody want = upload;
    Stats st;
    jacobiKernel(want, invDt, false, mats, st);
    // the two materials must give different answers, or the half rule has no teeth
    HexBody swapped = upload;
    swapped.matIdx[1] = 1u; swapped.matIdx[2] = 0u;
    Stats st2;
    jacobiKernel(swapped, invDt, false, mats, st2);
    PXG_CHECK_MSG(!sameBits(swapped.delta, want.delta) || swapped.mult != want.mult, "swapping the materials must change the oracle");
    const HexBody got = downloadBody(dev, upload, devs[1]);
    const CmpResult r = compareBody("hexMaterialHalf", want, got);
    PXG_CHECK_MSG(r.mm == 0, "Jacobi material lookup through the PxU16 half rule bit-exact (%zu)", r.mm);
    for (auto& d : devs) destroyBody(dev, d);
}

// ---------------------------------------------------------------------------
// 6 tets per voxel, massless (the bit-exact ROUTING tier): rotations, multipliers, the +0
// deltas and the w-counts, with nbElements = 5 so that the slot rules w + 5i, w*6 + i and
// w + 6i are pairwise different (review D2), and the FLIPPED bit set on one voxel (it must
// not select an offset for the 6-tet table).
// ---------------------------------------------------------------------------
PXG_TEST(softBodySimHexJacobi6Routing)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot() };
    HexBody b;
    initBody(b, 6u, 5u, 40u);
    b.acc = { 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 5u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 9u;
    for (uint32_t w = 0; w < 5; ++w)
    {
        privateVoxel(b, w, 8u * w, 2u * w, 0u, 0u, false, /*bit*/ w == 3u, 0u, /*massless*/ true);
        for (uint32_t j = 0; j < 8; ++j) b.delta[8u * w + j] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    // elements 0 and 1 are GS elements: their slots must stay poison
    for (uint32_t w = 0; w < 2; ++w)
        for (uint32_t i = 0; i < 6; ++i)
        {
            const uint32_t s = w + 5u * i;
            for (uint32_t e = 0; e < 4; ++e) (&b.rot[s].x)[e] = u2f(POISON);
            b.mult[mbyzWord(s)] = POISON; b.mult[mbyzWord(s) + 1] = POISON;
        }
    const HexBody upload = b;
    std::vector<HexBody> bodies = { idleBody(), b };
    std::vector<DevHex> devs;
    dispatchJacobi(dev, bodies, devs, { 1u }, 3u, false, invDt, mats);
    HexBody want = upload;
    Stats st;
    jacobiKernel(want, invDt, false, mats, st);
    for (uint32_t w = 2; w < 5; ++w)
        for (uint32_t i = 0; i < 6; ++i)
            PXG_CHECK_MSG(isIdentityBits(want.rot[w + 5u * i]), "oracle 6-tet slot %u not identity", w + 5u * i);
    const HexBody got = downloadBody(dev, upload, devs[1]);
    const CmpResult r = compareBody("hex6Routing", want, got);
    PXG_CHECK_MSG(r.mm == 0, "6-tet Jacobi routing bit-exact (%zu)", r.mm);
    for (auto& d : devs) destroyBody(dev, d);
}

// The ULP budget of the budget tiers: runSolveTetraPartition's (4096 ULP AND 1e-3
// relative), the inherited sin/cos seam of extractRotation.  Each budget case also proves
// that 100 vs 4 rotation iterations differ by >= 10x that budget from its seed, so the
// isFirstIteration wiring is observable through the budget (review D4).
static const uint32_t HEX_BUDGET_ULP = 4096u;
static const double HEX_BUDGET_REL = 1.0e-3;

static void checkIterationGap(const char* tag, const p95hex::HexBody& it4, const p95hex::HexBody& it100)
{
    const p95hex::CmpResult g = p95hex::compareBody(tag, it4, it100, HEX_BUDGET_ULP * 10u, HEX_BUDGET_REL * 10.0, /*quiet*/ true);
    std::printf("  [%s] 4 vs 100 rotation iterations: %zu lanes beyond 10x the budget (worst %u ULP, rel %.3e)\n",
                tag, g.mm, g.worstUlp, g.worstRel);
    PXG_CHECK_MSG(g.mm > 0u, "%s: 4 and 100 rotation iterations must differ by >= 10x the budget", tag);
}

// 6 tets per voxel with masses on every corner and a rotated (0.5 rad, see nonAffineVoxel),
// jittered seed: the general Jacobi path under the budget.  isFirstIteration is hard-wired
// false (:1503), so this is a 4-iteration solve and the seed must keep it well-conditioned.
PXG_TEST(softBodySimHexJacobi6Budget)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot() };
    HexBody b;
    initBody(b, 6u, 5u, 16u);
    b.acc = { 1u, 1u, 1u, 1u, 1u, 1u, 1u, 3u, 5u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 9u;
    nonAffineVoxel(b, 3u, 0u, 0u, false, false, /*th*/ 0.5);
    nonAffineVoxel(b, 4u, 8u, 2u, false, true, /*th*/ 0.5);
    for (uint32_t v = 0; v < 16; ++v) b.delta[v] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    const HexBody upload = b;
    std::vector<HexBody> bodies = { idleBody(), b };
    std::vector<DevHex> devs;
    dispatchJacobi(dev, bodies, devs, { 1u }, 2u, false, invDt, mats);
    HexBody want = upload, want100 = upload;
    Stats st, st100;
    jacobiKernel(want, invDt, false, mats, st, /*isFirstIteration*/ false);
    jacobiKernel(want100, invDt, false, mats, st100, /*isFirstIteration*/ true);
    checkIterationGap("hex6Budget", want, want100);
    const HexBody got = downloadBody(dev, upload, devs[1]);
    const CmpResult r = compareBody("hex6Budget", want, got, HEX_BUDGET_ULP, HEX_BUDGET_REL);
    std::printf("  [seam] hex 6-tet Jacobi: worst %u ULP, worst relative %.3e\n", r.worstUlp, r.worstRel);
    PXG_CHECK_MSG(r.mm == 0, "6-tet Jacobi within the documented sin/cos budget (%zu)", r.mm);
    PXG_CHECK_MSG(st.movedCorners >= 8u, "the budget tier must move corners (%u)", st.movedCorners);
    for (auto& d : devs) destroyBody(dev, d);
}

// ---------------------------------------------------------------------------
// The GS entry's hex branch (softBodyGM.cu:1450-1455): two private voxels, element 0
// FLIPPED in partition 0 and element 1 non-flipped in partition 1, dispatched one partition
// after the other with isFirstIteration = 1 (100 rotation iterations, still the fixed
// point).  In-place p/v on the UNORDERED buffers, mSimDeltaPos += p - pos onto a non-zero
// seed whose w is -0 (H2), an invMass -0 corner whose w must come back +0, velocities
// changed on the mobile corners, and nothing in mSimDelta, the CP buffers or the remap.
// Run under PGS and TGS: the hoisted isTGS must reach the hex branch (TGS stores no
// multipliers, PGS does).
// ---------------------------------------------------------------------------
static void runHexGS5(bool isTGS)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot(0.35f) };
    HexBody b;
    initBody(b, 5u, 2u, 16u);
    b.acc = { 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 8u;
    privateVoxel(b, 0u, 0u, 0u, 0u, 0u, /*flipped*/ true, true, 0u, false);
    privateVoxel(b, 1u, 8u, 3u, 0u, 0u, false, false, 0u, false);
    for (uint32_t v = 0; v < 16; ++v)
        b.deltaPos[v] = F4{ 0.125f + 0.0625f * float(v), -0.25f, 0.5f, -0.0f };
    b.pos[8].w = -0.0f; b.vel[8].w = -0.0f;     // element 1 corner 0: massless, solved -> +0
    const HexBody upload = b;
    std::vector<HexBody> bodies = { idleBody(), b };
    std::vector<DevHex> devs;
    dispatchGS(dev, bodies, devs, { 1u }, 1u, /*partition*/ 0u, isTGS, /*first*/ true, invDt, mats);
    dispatchGS(dev, bodies, devs, { 1u }, 1u, /*partition*/ 1u, isTGS, /*first*/ true, invDt, mats, /*upload*/ false);
    HexBody want = upload;
    Stats st;
    gsKernel(want, 0u, invDt, isTGS, true, mats, st);
    gsKernel(want, 1u, invDt, isTGS, true, mats, st);
    const HexBody got = downloadBody(dev, upload, devs[1]);
    const CmpResult r = compareBody(isTGS ? "hexGS5TGS" : "hexGS5", want, got);
    PXG_CHECK_MSG(r.mm == 0, "GS hex branch bit-exact vs the CUDA transcription (%zu)", r.mm);
    uint32_t velMoved = 0;
    for (uint32_t v = 0; v < 16; ++v)
        if (f2u(got.vel[v].x) != f2u(upload.vel[v].x)) ++velMoved;
    PXG_CHECK_MSG(velMoved >= 8u, "the GS tail must write the mobile corners' velocities (%u)", velMoved);
    PXG_CHECK_MSG(f2u(got.pos[8].w) == 0u && f2u(got.vel[8].w) == 0u && f2u(got.deltaPos[3].w) == 0u,
                  "w lanes: -0 invMass -> +0 (pos %08x vel %08x), deltaPos -0 + (+0) -> +0 (%08x)",
                  f2u(got.pos[8].w), f2u(got.vel[8].w), f2u(got.deltaPos[3].w));
    PXG_CHECK_MSG(sameBits(got.delta, upload.delta), "the GS tail must not write mSimDelta");
    if (isTGS)
        PXG_CHECK_MSG(got.mult == upload.mult, "TGS must not store multipliers");
    else
        PXG_CHECK_MSG(got.mult != upload.mult, "PGS must store the multipliers");
    for (auto& d : devs) destroyBody(dev, d);
}

PXG_TEST(softBodySimHexGS5) { runHexGS5(/*isTGS*/ false); }
PXG_TEST(softBodySimHexGS5TGS) { runHexGS5(/*isTGS*/ true); }

// 6 tets per voxel through the GS entry, massless: routing, slots, in-place tail.
PXG_TEST(softBodySimHexGS6Routing)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot() };
    HexBody b;
    initBody(b, 6u, 5u, 40u);
    b.acc = { 2u, 3u, 5u, 5u, 5u, 5u, 5u, 5u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 8u;
    for (uint32_t w = 0; w < 5; ++w)
        privateVoxel(b, w, 8u * w, 2u * w, 0u, 0u, false, /*bit*/ w == 4u, 0u, /*massless*/ true);
    const HexBody upload = b;
    std::vector<HexBody> bodies = { idleBody(), b };
    std::vector<DevHex> devs;
    dispatchGS(dev, bodies, devs, { 1u }, 2u, /*partition*/ 2u, false, /*first*/ false, invDt, mats);
    HexBody want = upload;
    Stats st;
    gsKernel(want, 2u, invDt, false, false, mats, st);
    const HexBody got = downloadBody(dev, upload, devs[1]);
    const CmpResult r = compareBody("hexGS6Routing", want, got);
    PXG_CHECK_MSG(r.mm == 0, "6-tet GS routing bit-exact (%zu)", r.mm);
    // Massless and affine, so positions, velocities, mSimDeltaPos and the identity rotations
    // all come back with their upload bits; the PGS multipliers are what moves (C != 0 for the
    // stretched voxel, and with no mass dlambda = (-C - alpha*lambda)/alpha).
    PXG_CHECK_MSG(want.mult != upload.mult && got.mult != upload.mult,
                  "the GS hex branch must do work (the PGS multipliers of partition 2's voxels move)");
    for (auto& d : devs) destroyBody(dev, d);
}

// 6 tets per voxel through the GS entry with isFirstIteration = 1 from a large-rotation seed:
// the budget tier, and the proof that the forwarded flag (100 iterations) is observable.
PXG_TEST(softBodySimHexGS6Budget)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const float invDt = 60.0f;
    const std::vector<p80solve::Material> mats = { coRot() };
    HexBody b;
    initBody(b, 6u, 2u, 16u);
    b.acc = { 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 8u;
    nonAffineVoxel(b, 0u, 0u, 0u, false, false);
    nonAffineVoxel(b, 1u, 8u, 2u, false, false);
    const HexBody upload = b;
    std::vector<HexBody> bodies = { idleBody(), b };
    std::vector<DevHex> devs;
    dispatchGS(dev, bodies, devs, { 1u }, 2u, /*partition*/ 0u, false, /*first*/ true, invDt, mats);
    HexBody want = upload, want4 = upload;
    Stats st, st4;
    gsKernel(want, 0u, invDt, false, /*first*/ true, mats, st);
    gsKernel(want4, 0u, invDt, false, /*first*/ false, mats, st4);
    checkIterationGap("hexGS6Budget", want4, want);
    const HexBody got = downloadBody(dev, upload, devs[1]);
    const CmpResult r = compareBody("hexGS6Budget", want, got, HEX_BUDGET_ULP, HEX_BUDGET_REL);
    std::printf("  [seam] hex 6-tet GS: worst %u ULP, worst relative %.3e\n", r.worstUlp, r.worstRel);
    PXG_CHECK_MSG(r.mm == 0, "6-tet GS within the documented sin/cos budget (%zu)", r.mm);
    for (auto& d : devs) destroyBody(dev, d);
}

// The GS early-out on eKINEMATIC is the existing tet-path code; this pins that the hex
// branch sits behind it (regression only).
PXG_TEST(softBodySimHexGSKinematic)
{
    using namespace p95hex;
    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<p80solve::Material> mats = { coRot() };
    HexBody b;
    initBody(b, 5u, 1u, 8u);
    b.acc = { 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u };
    b.acc.resize(16u, POISON);
    b.numPartitions = 8u;
    b.bodyFlags = uint8_t(PxDeformableBodyFlag::eKINEMATIC);
    privateVoxel(b, 0u, 0u, 0u, 0u, 0u, false, false, 0u, false);
    const HexBody upload = b;
    std::vector<HexBody> bodies = { idleBody(), b };
    std::vector<DevHex> devs;
    dispatchGS(dev, bodies, devs, { 1u }, 1u, 0u, false, true, 60.0f, mats);
    const HexBody got = downloadBody(dev, upload, devs[1]);
    PXG_CHECK_MSG(compareBody("hexGSKinematic", upload, got).mm == 0, "a kinematic voxel body must be untouched");
    for (auto& d : devs) destroyBody(dev, d);
}

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
static const uint32_t GEOM_TETRAHEDRONMESH_H = uint32_t(physx::PxGeometryType::eTETRAHEDRONMESH);   // PxGeometryType::eTETRAHEDRONMESH

// ===========================================================================
// P9.39a -- sb_psMidphaseGeneratePairsLaunch, the softbody-vs-PARTICLE midphase.
//
// The TET twin of cloth_psMidphaseGeneratePairsLaunch (P9.30) and the first of the
// six kernels of the last unmapped softbody chain.  It shares this file rather than
// starting a new one because the tetrahedron mesh, the BV32 TreeBlob and the shape
// plumbing are all already here -- duplicating them is the duplication this port
// exists to avoid.
//
// WHAT THIS RIG HAS THAT THE CLOTH SIBLING'S DOES NOT: a DEGENERATE tetrahedron
// sitting exactly where a particle would otherwise hit it.  softbodyMidPhase.cu:287
// gates the bound test on `isValidTet` (signed volume > 1e-9), and the cloth
// traverser has no such gate, so a port that clones the cloth one emits a pair there.
// The scene is built so that tet is INSIDE the query box: only the gate can exclude
// it, and the expected set says it is excluded.
// ---------------------------------------------------------------------------
struct SbPsMidphasePush
{
    uint32_t numWorkItems; float toleranceLength;
    uint64_t cmInputs, gpuShapes, transformCache, bounds, contactDistance, softbodies,
             particleSystems;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(SbPsMidphasePush) == 88, "sb ps midphase blob must be 88 bytes");
static_assert(offsetof(SbPsMidphasePush, gpuShapes) == 16, "gpuShapes @16 -- THIRD");
static_assert(offsetof(SbPsMidphasePush, softbodies) == 48, "softbodies @48");
static_assert(offsetof(SbPsMidphasePush, particleSystems) == 56, "particleSystems @56");
static_assert(offsetof(SbPsMidphasePush, stackSizeBytes) == 64, "stackSizeBytes @64");
static_assert(offsetof(SbPsMidphasePush, stackPtr) == 72, "stackPtr @72");
static_assert(offsetof(SbPsMidphasePush, midphasePairsNum) == 80, "midphasePairsNum @80");

static const uint32_t GEOM_PARTICLESYSTEM_H2 = uint32_t(physx::PxGeometryType::ePARTICLESYSTEM);   // PxGeometryType::ePARTICLESYSTEM

PXG_TEST(softBodySimPsMidphase)
{
    using namespace p80refit;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t SB_ID = 1u, PS_ID = 2u;
    // ASYMMETRIC on purpose, as the sibling rig learned to be: 0.24 + 0.06 = 0.30, and
    // one tet is placed so that NEITHER half alone reaches it.  With equal halves a
    // port that used only one of the two contact distances still passed.
    // DYADIC on purpose: 0.25 + 0.0625 = 0.3125 exactly, so a particle can be placed
    // where cx + R lands EXACTLY on a tet's tMin and the strict/loose boundary of the
    // overlap test becomes reachable.  Still asymmetric, so reading one instead of the
    // sum is still separable.
    const float CD_PARTICLE = 0.25f, CD_SOFTBODY = 0.0625f;
    const float R = CD_PARTICLE + CD_SOFTBODY;

    // ---- tets.  Each owns its four vertices so it can be moved or degenerated
    // without disturbing its neighbours.
    std::vector<F4> verts;
    // side 0.5 -- dyadic, so every vertex coordinate is exact
    auto pushTet = [&](float ox, float oy, float oz, bool degenerate) {
        verts.push_back(F4{ ox,        oy,        oz,        1.0f });
        verts.push_back(F4{ ox + 0.5f, oy,        oz,        1.0f });
        verts.push_back(F4{ ox,        oy + 0.5f, oz,        1.0f });
        // a degenerate tet puts the 4th corner ON the plane z == oz of the other
        // three, so its BOUND still overlaps but its signed volume is exactly 0
        verts.push_back(degenerate ? F4{ ox + 0.25f, oy + 0.125f, oz, 1.0f }
                                   : F4{ ox,         oy,          oz + 0.5f, 1.0f });
    };
    pushTet(0.0f,  0.0f, 0.0f, false);   // tet 0 -- a particle sits right on it
    pushTet(1.0f,  0.0f, 0.0f, false);   // tet 1 -- reached together with tet 2
    pushTet(2.0f,  0.0f, 0.0f, false);   // tet 2 -- ditto, one particle reaches BOTH
    pushTet(3.0f,  0.0f, 0.0f, true);    // tet 3 -- DEGENERATE, inside the box (T2)
    pushTet(5.0f,  0.0f, 0.0f, false);   // tet 4 -- only the FULL sum of CDs reaches it
    pushTet(8.0f,  0.0f, 0.0f, false);   // tet 5 -- the EXACT-TOUCH boundary
    pushTet(10.0f, 0.0f, 0.0f, false);   // tet 6 -- reached across the box's LOW side
    pushTet(80.0f, 80.0f, 80.0f, false); // tet 7 -- far outside everything

    const uint32_t NB_TETS = 8u;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < NB_TETS; ++t)
        tets.push_back(TetIdx4{ 4u * t + 0u, 4u * t + 1u, 4u * t + 2u, 4u * t + 3u });

    DevBuf bVerts = makeDev(dev, verts);
    if (!bVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bVerts.h); return; }
    DevBuf bTets = makeDev(dev, tets);

    // One root node whose children are all LEAVES, one tet each, with GENEROUS child
    // bounds: every child overlaps every query, so the node-level cull can never stand
    // in for the primitive-level predicate the mirror computes.
    const uint32_t nbNodes = 1u, maxDepth = 1u;
    TreeBlob tree;
    tree.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes + 4u, 0u);
    tree.w[0] = uint32_t(verts.size());
    tree.w[1] = NB_TETS;
    tree.w[2] = maxDepth;
    tree.w[3] = nbNodes;
    const uint32_t nw = tree.nodeWord(0);
    tree.w[nw + W_NBNODES] = NB_TETS;
    for (uint32_t k = 0; k < NB_TETS; ++k)
    {
        const float lo = -1000.0f, hi = 1000.0f;
        tree.w[nw + W_MMIN + k * 4 + 0] = f2u(lo);
        tree.w[nw + W_MMIN + k * 4 + 1] = f2u(lo);
        tree.w[nw + W_MMIN + k * 4 + 2] = f2u(lo);
        tree.w[nw + W_MMAX + k * 4 + 0] = f2u(hi);
        tree.w[nw + W_MMAX + k * 4 + 1] = f2u(hi);
        tree.w[nw + W_MMAX + k * 4 + 2] = f2u(hi);
        // leaf holding exactly tet k -- the file's own encoder, not a hand-rolled one
        tree.w[nw + W_MDATA + k] = leafWord(k, 1u);
    }
    DevBuf bTree = makeDev(dev, tree.w);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
    sb.mNumTets = NB_TETS;
    sb.mNumVerts = uint32_t(verts.size());
    std::vector<PxgSoftBody> descArr(SB_ID + 1u, makeZeroDesc());
    descArr[SB_ID] = sb;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- particles ------------------------------------------------------------
    std::vector<F4> parts = {
        { 0.125f,  0.125f,   0.125f, 1.0f },  // inside tet 0's bound
        { 1.75f,   0.0f,     0.0f,   1.0f },  // between tets 1 and 2: reaches BOTH
        { 3.25f,   0.0625f,  0.0f,   1.0f },  // inside the DEGENERATE tet 3's bound (T2)
        { 5.0f,   -0.28125f, 0.0f,   1.0f },  // below tet 4: 0.28125 needs the SUM
                                              // (0.25 alone falls short, 0.3125 reaches)
        { 7.6875f, 0.0f,     0.0f,   1.0f },  // EXACT TOUCH: cx + R == 7.6875 + 0.3125
                                              // == 8.0 == tet 5's tMin.x, so `tMin.x >
                                              // qMax.x` compares EQUAL values and the
                                              // strict/loose choice decides the pair
        { 10.25f,  0.75f,    0.0f,   1.0f },  // ABOVE tet 6: the box's LOW side is what
                                              // reaches it, so shrinking qMin alone is
                                              // observable
        { 40.0f,   40.0f,    40.0f,  1.0f },  // nowhere near anything
    };
    const uint32_t NB_PARTICLES = uint32_t(parts.size());
    DevBuf bParts = makeDev(dev, parts);

    std::vector<PxgParticleSystem> psArr(PS_ID + 1u);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    psArr[PS_ID].mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bParts.addr));
    psArr[PS_ID].mCommonData.mNumParticles = NB_PARTICLES;
    DevBuf bPs = makeDev(dev, psArr);

    // ---- shapes.  TWO managers with the particle system in OPPOSITE slots, so the
    // routing by TYPE (T4) is separable from slot order.
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_PARTICLESYSTEM_H2;  shapes[0].particleOrSoftbodyId = PS_ID;
    shapes[0].sx = shapes[0].sy = shapes[0].sz = 1.0f; shapes[0].srw = 1.0f;
    shapes[1].type = GEOM_TETRAHEDRONMESH_H;  shapes[1].particleOrSoftbodyId = SB_ID;
    shapes[1].sx = shapes[1].sy = shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);

    // cache ref 0 = particle system (CD 0.24), 1 = softbody (CD 0.06).
    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u }, { 1u, 0u, 1u, 0u } };
    DevBuf bCms = makeDev(dev, cms);
    std::vector<float> cd = { CD_PARTICLE, CD_SOFTBODY };
    DevBuf bCd = makeDev(dev, cd);
    std::vector<float> xf(2 * 8, 0.0f); xf[3] = 1.0f; xf[8 + 3] = 1.0f;   // ABI parity only
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> dummyBounds(2 * 6, 0.0f);                          // ABI parity only
    DevBuf bBounds = makeDev(dev, dummyBounds);

    // The stack holds FEWER pairs than the query finds, so "drop the pair, keep
    // counting" is measured rather than assumed.
    const uint32_t NB_ALLOC = 128u;
    const uint32_t STACK_PAIRS = 5u;
    const uint32_t POISON2 = 0xEEEEEEEEu;
    std::vector<uint32_t> stack(NB_ALLOC * 4u, POISON2);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> zero = { 0u };
    DevBuf bNum = makeDev(dev, zero);

    SbPsMidphasePush p = {};
    p.numWorkItems = uint32_t(cms.size());
    p.toleranceLength = 1.0f;
    p.cmInputs = bCms.addr; p.gpuShapes = bShapes.addr; p.transformCache = bXf.addr;
    p.bounds = bBounds.addr; p.contactDistance = bCd.addr;
    p.softbodies = bDesc.addr; p.particleSystems = bPs.addr;
    p.stackSizeBytes = STACK_PAIRS * 16u;
    p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;

    Kernel k("softBodyPsMidphaseBda.spv", "sb_psMidphaseGeneratePairsLaunch",
             {}, sizeof(SbPsMidphasePush));
    k.dispatch(1024u, {}, p, uint32_t(cms.size()), 1u);   // grid (1024, numWorkItems, 1)
    dev->flush();

    // ---- the expected set -----------------------------------------------------
    struct Pair { uint32_t cm, particle, tet; };
    std::vector<Pair> expected;
    uint32_t degenerateSuppressed = 0u, sumOnly = 0u;
    for (uint32_t cm = 0; cm < uint32_t(cms.size()); ++cm)
        for (uint32_t pi = 0; pi < NB_PARTICLES; ++pi)
        {
            const float cx = parts[pi].x, cy = parts[pi].y, cz = parts[pi].z;
            for (uint32_t t = 0; t < NB_TETS; ++t)
            {
                const F4& a = verts[t * 4 + 0]; const F4& b = verts[t * 4 + 1];
                const F4& c = verts[t * 4 + 2]; const F4& d = verts[t * 4 + 3];
                // isValidTet: dot(b - a, cross(c - a, d - a)) > 1e-9  (T2)
                const float e1x = b.x - a.x, e1y = b.y - a.y, e1z = b.z - a.z;
                const float e2x = c.x - a.x, e2y = c.y - a.y, e2z = c.z - a.z;
                const float e3x = d.x - a.x, e3y = d.y - a.y, e3z = d.z - a.z;
                const float crx = e2y * e3z - e2z * e3y;
                const float cry = e2z * e3x - e2x * e3z;
                const float crz = e2x * e3y - e2y * e3x;
                const bool valid = (e1x * crx + e1y * cry + e1z * crz) > 1e-9f;

                const float tMinX = std::min(std::min(a.x, b.x), std::min(c.x, d.x));
                const float tMinY = std::min(std::min(a.y, b.y), std::min(c.y, d.y));
                const float tMinZ = std::min(std::min(a.z, b.z), std::min(c.z, d.z));
                const float tMaxX = std::max(std::max(a.x, b.x), std::max(c.x, d.x));
                const float tMaxY = std::max(std::max(a.y, b.y), std::max(c.y, d.y));
                const float tMaxZ = std::max(std::max(a.z, b.z), std::max(c.z, d.z));
                const bool overlap = !(tMinX > cx + R || cx - R > tMaxX ||
                                       tMinY > cy + R || cy - R > tMaxY ||
                                       tMinZ > cz + R || cz - R > tMaxZ);

                if (overlap && !valid) ++degenerateSuppressed;
                // would EITHER half alone have reached it?
                if (overlap && valid)
                {
                    const float half = std::max(CD_PARTICLE, CD_SOFTBODY);
                    const bool halfHit = !(tMinX > cx + half || cx - half > tMaxX ||
                                           tMinY > cy + half || cy - half > tMaxY ||
                                           tMinZ > cz + half || cz - half > tMaxZ);
                    if (!halfHit) ++sumOnly;
                }
                if (overlap && valid) expected.push_back(Pair{ cm, pi, t });
            }
        }

    // The rig only means something if it actually exercises what it claims to.
    PXG_CHECK_MSG(degenerateSuppressed > 0u,
                  "no particle's box overlaps the DEGENERATE tet, so the isValidTet gate "
                  "(T2) is never the thing that excludes a pair and a port without it passes");
    PXG_CHECK_MSG(sumOnly > 0u,
                  "every hit is reachable with only the LARGER of the two contact distances, "
                  "so a port that reads one instead of their sum still passes");
    PXG_CHECK_MSG(expected.size() > STACK_PAIRS,
                  "the rig found %zu pairs against a %u-pair stack; it must OVERFLOW or the "
                  "drop-but-keep-counting behaviour is not measured", expected.size(), STACK_PAIRS);
    // The exact-touch particle must really touch: cx + R == tet 5's tMin.x to the bit.
    PXG_CHECK_MSG(parts[4].x + R == verts[5 * 4 + 0].x,
                  "the boundary case is not on the boundary (%.9g + %.9g != %.9g); the "
                  "strict/loose choice in the overlap test is then unreachable",
                  parts[4].x, R, verts[5 * 4 + 0].x);

    const uint32_t found = downloadBuffer<uint32_t>(dev, bNum.h, 1)[0];
    PXG_CHECK_MSG(found == uint32_t(expected.size()),
                  "the midphase COUNTER reads %u, the mirror says %zu.  The counter takes every "
                  "hit even when the stack cannot hold it", found, expected.size());

    const auto st = downloadBuffer<uint32_t>(dev, bStack.h, NB_ALLOC * 4u);
    uint32_t written = 0u, bad = 0u;
    for (uint32_t i = 0; i < NB_ALLOC; ++i)
    {
        if (st[i * 4] == POISON2) continue;
        ++written;
        if (i >= STACK_PAIRS) { ++bad; continue; }
        const uint32_t cm = st[i * 4 + 0], pi = st[i * 4 + 1], tet = st[i * 4 + 2];
        const uint32_t w = st[i * 4 + 3];
        if (w != 0u)
        {
            ++bad;
            std::printf("  slot %u has .w = %u; the CUDA writes make_uint4(cm, query, prim, 0)\n", i, w);
            continue;
        }
        bool member = false;
        for (const Pair& e : expected)
            if (e.cm == cm && e.particle == pi && e.tet == tet) { member = true; break; }
        if (!member)
        {
            ++bad;
            std::printf("  slot %u holds (cm %u, particle %u, tet %u), not an expected pair -- "
                        "T1 puts the PARTICLE in .y and the TET in .z\n", i, cm, pi, tet);
        }
    }
    PXG_CHECK_MSG(bad == 0u, "%u written pairs are wrong or past the stack bound", bad);
    PXG_CHECK_MSG(written == STACK_PAIRS,
                  "%u slots written, expected exactly the %u the stack holds", written, STACK_PAIRS);

    // ---- pass 2: a stack that holds EVERYTHING, compared as a SET ----------------
    // The pass above can only ask whether each written pair is a MEMBER of the expected
    // set, and membership is too weak: with two contact managers producing the same
    // (particle, tet) pairs, a port that wrote a constant 0 for the manager index still
    // wrote members.  The sweep found exactly that.  Here the whole set is compared,
    // multiplicity and all, so the manager index has to be right.
    {
        std::vector<uint32_t> bigStack(NB_ALLOC * 4u, POISON2);
        DevBuf bBig = makeDev(dev, bigStack);
        std::vector<uint32_t> z2 = { 0u };
        DevBuf bBigNum = makeDev(dev, z2);
        SbPsMidphasePush q = p;
        q.stackSizeBytes = NB_ALLOC * 16u;
        q.stackPtr = bBig.addr; q.midphasePairsNum = bBigNum.addr;
        Kernel k2("softBodyPsMidphaseBda.spv", "sb_psMidphaseGeneratePairsLaunch",
                  {}, sizeof(SbPsMidphasePush));
        k2.dispatch(1024u, {}, q, uint32_t(cms.size()), 1u);
        dev->flush();

        const uint32_t n2 = downloadBuffer<uint32_t>(dev, bBigNum.h, 1)[0];
        PXG_CHECK_MSG(n2 == uint32_t(expected.size()),
                      "with room for every pair the counter reads %u, mirror %zu", n2,
                      expected.size());
        const auto b2 = downloadBuffer<uint32_t>(dev, bBig.h, NB_ALLOC * 4u);
        std::vector<Pair> got;
        for (uint32_t i = 0; i < NB_ALLOC; ++i)
            if (b2[i * 4] != POISON2)
                got.push_back(Pair{ b2[i * 4 + 0], b2[i * 4 + 1], b2[i * 4 + 2] });
        auto key = [](const Pair& x) { return (uint64_t(x.cm) << 40) | (uint64_t(x.particle) << 20) | x.tet; };
        std::vector<uint64_t> ka, kb;
        for (const Pair& x : got) ka.push_back(key(x));
        for (const Pair& x : expected) kb.push_back(key(x));
        std::sort(ka.begin(), ka.end()); std::sort(kb.begin(), kb.end());
        PXG_CHECK_MSG(ka == kb,
                      "the written pair SET differs from the mirror's (%zu vs %zu entries); "
                      "membership alone would not have caught a constant manager index",
                      ka.size(), kb.size());
        uint32_t cm0 = 0u, cm1 = 0u;
        for (const Pair& x : got) { if (x.cm == 0u) ++cm0; else if (x.cm == 1u) ++cm1; }
        PXG_CHECK_MSG(cm0 > 0u && cm1 > 0u,
                      "both managers must appear in the output (cm0 %u, cm1 %u), or the "
                      "manager index is not load-bearing", cm0, cm1);
    }

    // ---- the numWorkItems GUARD ---------------------------------------------------
    // `if (cmIdx >= numWorkItems) return;` never fires when the dispatch has exactly
    // numWorkItems y-groups, which is what the engine launches -- so a port that
    // dropped the guard passed, and the sweep said so.  Here the dispatch is
    // deliberately WIDER than the work: the extra y-groups must contribute nothing.
    {
        std::vector<uint32_t> wideStack(NB_ALLOC * 4u, POISON2);
        DevBuf bWide = makeDev(dev, wideStack);
        std::vector<uint32_t> z3 = { 0u };
        DevBuf bWideNum = makeDev(dev, z3);
        SbPsMidphasePush q = p;
        q.stackSizeBytes = NB_ALLOC * 16u;
        q.stackPtr = bWide.addr; q.midphasePairsNum = bWideNum.addr;
        Kernel k3("softBodyPsMidphaseBda.spv", "sb_psMidphaseGeneratePairsLaunch",
                  {}, sizeof(SbPsMidphasePush));
        k3.dispatch(1024u, {}, q, uint32_t(cms.size()) + 3u, 1u);   // THREE extra y-groups
        dev->flush();
        const uint32_t nWide = downloadBuffer<uint32_t>(dev, bWideNum.h, 1)[0];
        PXG_CHECK_MSG(nWide == uint32_t(expected.size()),
                      "a dispatch %u y-groups wider than numWorkItems produced %u pairs, "
                      "expected %zu -- the guard is what makes the extra groups inert",
                      3u, nWide, expected.size());
    }

    std::printf("    [ note ] %zu pairs over %u particles x %u tets x %zu managers; %u suppressed "
                "by isValidTet, %u reachable only with the SUM of both contact distances; "
                "%u written into a %u-pair stack, counter %u\n",
                expected.size(), NB_PARTICLES, NB_TETS, cms.size(), degenerateSuppressed,
                sumOnly, written, STACK_PAIRS, found);
}

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
// PxgRigidDbContactWriter (PxgDeformableContactWriters.h, 5.12; PxgFEMContactWriter in 5.6.1).
struct FemWriterHost
{
    uint64_t outPoint, outNormalPen, outBarycentric, outContactInfo, outRigidMaterialIndex, totalContactCount;
    uint64_t contactByRigid, tempContactByRigid, contactIndexSortedByRigid;
    uint32_t maxNumContacts, pad;
};
static_assert(sizeof(FemWriterHost) == sizeof(PxgRigidDbContactWriter), "PxgRigidDbContactWriter mirror");
static_assert(offsetof(FemWriterHost, outRigidMaterialIndex) == offsetof(PxgRigidDbContactWriter, outRigidMaterialIndex), "outRigidMaterialIndex");
static_assert(offsetof(FemWriterHost, contactIndexSortedByRigid) == offsetof(PxgRigidDbContactWriter, contactIndexSortedByRigid), "contactIndexSortedByRigid");
static_assert(offsetof(FemWriterHost, maxNumContacts) == offsetof(PxgRigidDbContactWriter, maxNumContacts), "maxNumContacts");
struct SbContactGenPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes,
             softbodies, shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad84;
    uint64_t stackSizeNeededOnDevice;
    FemWriterHost w;
};
// 5.12: the softBodyRigidPairs filter arguments are gone (the filter is per softbody,
// mRigidVolumeFilterPairs) and the writer is PxgRigidDbContactWriter -> 176 bytes.
static_assert(sizeof(SbContactGenPush) == 176, "sb contact-gen push blob must be 176 bytes");
static_assert(offsetof(SbContactGenPush, stackPtr) == 64, "stackPtr @64");
static_assert(offsetof(SbContactGenPush, stackSizeBytes) == 80, "stackSizeBytes @80");
static_assert(offsetof(SbContactGenPush, stackSizeNeededOnDevice) == 88, "stackSizeNeeded @88");
static_assert(offsetof(SbContactGenPush, w) == 96, "the writer is inlined at @96");
static_assert(offsetof(SbContactGenPush, w) + offsetof(FemWriterHost, maxNumContacts) == 168,
              "writer.maxNumContacts @184");

// 5.12: contacts carry a PxgDeformablePairInfo (16 B: ref0 u64 @0, ref1 u64 @8), the 5.6.1
// PxgFemOtherContactInfo's packed ids replaced by PxgDeformableRef encodings.
static_assert(sizeof(PxgDeformablePairInfo) == 16, "PxgDeformablePairInfo is 16 bytes");
static uint64_t sbTetRef(uint32_t softbodyId, uint32_t tet) { return PxgEncodeDeformableTetrahedron(softbodyId, tet); }

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
// Contact A's point is inside candidate 2, so containment must win and ref1 must name
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
    std::vector<PxgDeformablePairInfo> infos(4);
    std::memset(infos.data(), 0, infos.size() * sizeof(PxgDeformablePairInfo));
    infos[0].ref0 = 0x1111ull; infos[0].ref1 = sbTetRef(0u, 0u);
    infos[1].ref0 = 0x2222ull; infos[1].ref1 = sbTetRef(0u, 0u);
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

    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, 4);
    const auto gBary = downloadBuffer<F4>(dev, bBary.h, 4);
    for (uint32_t i = 0; i < 2u; ++i)
        std::printf("  contact %u: ref1 -> tet %u, bary(%.4f, %.4f, %.4f, %.4f)\n",
                    i, PxgGetDeformableFeatureId(gInfo[i].ref1), gBary[i].x, gBary[i].y, gBary[i].z, gBary[i].w);

    // CONTAINMENT wins for contact A: the point is inside candidate 2.
    PXG_CHECK_MSG(gInfo[0].ref1 == sbTetRef(0u, 2u),
                  "contact A is inside sim tet 2, so containment must win (got ref1 0x%llx)",
                  (unsigned long long)gInfo[0].ref1);
    // ...and its barycentric must reconstruct the point: 0.2/0.2/0.2 inside the unit tet.
    PXG_CHECK_MSG(std::fabs(gBary[0].y - 0.2f) < 1e-4f && std::fabs(gBary[0].z - 0.2f) < 1e-4f &&
                  std::fabs(gBary[0].w - 0.2f) < 1e-4f && std::fabs(gBary[0].x - 0.4f) < 1e-4f,
                  "contact A's barycentric must be (0.4, 0.2, 0.2, 0.2), got (%.4f, %.4f, %.4f, %.4f)",
                  gBary[0].x, gBary[0].y, gBary[0].z, gBary[0].w);
    // The MIN FALLBACK wins for contact B: nearest is candidate 1, not 0 and not 2.
    PXG_CHECK_MSG(gInfo[1].ref1 == sbTetRef(0u, 1u),
                  "contact B is outside every candidate, so the NEAREST (tet 1) must win "
                  "(got ref1 0x%llx)", (unsigned long long)gInfo[1].ref1);
    // the fallback clamps every component into [0,1]
    for (uint32_t c = 0; c < 4u; ++c)
    {
        const float v = (c == 0) ? gBary[1].x : (c == 1) ? gBary[1].y : (c == 2) ? gBary[1].z : gBary[1].w;
        PXG_CHECK_MSG(v >= 0.0f && v <= 1.0f, "the fallback barycentric must be clamped (comp %u = %g)", c, v);
    }
    // the softbody id survives the rewrite (checked through the whole ref1 above), and ref0
    // is untouched
    PXG_CHECK_MSG(gInfo[0].ref0 == 0x1111ull, "the remap must touch ONLY ref1");
    PXG_CHECK_MSG(gInfo[1].ref0 == 0x2222ull, "the remap must touch ONLY ref1");
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
        // a FRESH info buffer: the dispatches above already rewrote ref1 in bInfo, and
        // reusing it would hand this dispatch collision tet 1 (range 0) instead of tet 0,
        // making the "contact 1 was processed" check pass on an empty candidate set.
        std::vector<PxgDeformablePairInfo> i3 = infos;
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
        const auto gi3 = downloadBuffer<PxgDeformablePairInfo>(dev, bI3.h, 4);
        PXG_CHECK_MSG(g3[0].x == -1.0f,
                      "prevContactCount = 1 must SKIP contact 0, but its barycentric was "
                      "written (%.4f)", g3[0].x);
        PXG_CHECK_MSG(gi3[0].ref1 == sbTetRef(0u, 0u),
                      "prevContactCount = 1 must leave contact 0's ref1 alone (got 0x%llx)",
                      (unsigned long long)gi3[0].ref1);
        // contact 1 still runs the full nearest-candidate search, so it must land on tet 1.
        PXG_CHECK_MSG(gi3[1].ref1 == sbTetRef(0u, 1u),
                      "prevContactCount = 1 must still remap contact 1 to the nearest tet 1 "
                      "(got tet %u)", PxgGetDeformableFeatureId(gi3[1].ref1));
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
    std::vector<PxgDeformablePairInfo> infos(4);
    std::memset(infos.data(), 0, infos.size() * sizeof(PxgDeformablePairInfo));
    infos[0].ref1 = sbTetRef(0u, 0u);   // softbody 0, collision tet 0
    infos[1].ref1 = sbTetRef(0u, 1u);   // softbody 0, collision tet 1
    infos[2].ref1 = sbTetRef(0u, 2u);   // softbody 0, collision tet 2
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

    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, 4);
    const auto gBary = downloadBuffer<F4>(dev, bBary.h, 4);
    for (uint32_t i = 0; i < 3u; ++i)
        std::printf("  contact %u -> sim tet %u, bary(%.4f, %.4f, %.4f, %.4f)\n",
                    i, PxgGetDeformableFeatureId(gInfo[i].ref1), gBary[i].x, gBary[i].y, gBary[i].z, gBary[i].w);

    PXG_CHECK_MSG(gInfo[0].ref1 == sbTetRef(0u, kNear),
                  "a SAME-LANE tie must keep the FIRST candidate (tet %u), not its twin 32 "
                  "later (tet %u) -- got tet %u", kNear, kNearTwin, PxgGetDeformableFeatureId(gInfo[0].ref1));
    PXG_CHECK_MSG(gInfo[1].ref1 == sbTetRef(0u, kCross),
                  "a CROSS-LANE tie must win on the LOWEST lane (tet %u), not the highest "
                  "(tet %u) -- got tet %u; a startInd of 0 would land near tet %u instead",
                  kCross, kCrossTwin, PxgGetDeformableFeatureId(gInfo[1].ref1), kNear);
    PXG_CHECK_MSG(gInfo[2].ref1 == sbTetRef(0u, 50u),
                  "with two CONTAINING candidates the FIRST one (tet 50) must win, not the "
                  "last (tet 51) -- got tet %u", PxgGetDeformableFeatureId(gInfo[2].ref1));
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
        std::vector<PxgDeformablePairInfo> iB(kBig);
        std::memset(iB.data(), 0, iB.size() * sizeof(PxgDeformablePairInfo));
        for (PxgDeformablePairInfo& pi : iB) pi.ref1 = sbTetRef(0u, 0u);
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
    std::vector<PxgDeformablePairInfo> infos(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        pts[i]  = F4{ float(i) + 0.1f, 0, 0, 0 };
        nps[i]  = F4{ 0, float(i) + 0.2f, 0, 0 };
        bary[i] = F4{ 0, 0, float(i) + 0.3f, 0 };
        infos[i].ref0 = 0x1000ull + i; infos[i].ref1 = sbTetRef(i, 100u + i) | (uint64_t(i & 1u) << 33);
    }
    for (int t = 0; t < 2; ++t) { pts.push_back(F4{0,0,0,0}); nps.push_back(F4{0,0,0,0}); bary.push_back(F4{0,0,0,0}); }
    infos.resize(N + 2);
    const float POISONF = -777.0f;
    std::vector<F4> sPts(N + 2, F4{ POISONF, POISONF, POISONF, POISONF });
    std::vector<F4> sNps = sPts, sBary = sPts;
    std::vector<PxgDeformablePairInfo> sInfos(N + 2);
    std::memset(sInfos.data(), 0xDD, sInfos.size() * sizeof(PxgDeformablePairInfo));

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
    k.dispatch(FEM_PREP_GRID, {}, push, 1u, 1u);
    dev->flush();

    const auto gP = downloadBuffer<F4>(dev, bSP.h, N + 2);
    const auto gNpO = downloadBuffer<F4>(dev, bSNp.h, N + 2);
    const auto gB = downloadBuffer<F4>(dev, bSB.h, N + 2);
    const auto gI = downloadBuffer<PxgDeformablePairInfo>(dev, bSI.h, N + 2);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t r = remap[i];
        if (std::fabs(gP[i].x - pts[r].x) > 1e-6f) { ++mm; std::printf("  point[%u] MM: got %g want %g\n", i, gP[i].x, pts[r].x); }
        if (std::fabs(gNpO[i].y - nps[r].y) > 1e-6f) { ++mm; std::printf("  normalPen[%u] MM\n", i); }
        if (std::fabs(gB[i].z - bary[r].z) > 1e-6f) { ++mm; std::printf("  bary[%u] MM\n", i); }
        if (gI[i].ref0 != infos[r].ref0 || gI[i].ref1 != infos[r].ref1) { ++mm; std::printf("  info[%u] MM\n", i); }
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
    std::vector<PxgDeformablePairInfo> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(PxgDeformablePairInfo));
    std::vector<uint16_t> outMat(MAXC + 8u, 0xEEEEu);   // PxU16 rigid material per contact (5.12)
    DevBuf bOutMat = makeDev(dev, outMat);
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
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.w.outPoint = bPt.addr; push.w.outNormalPen = bNp.addr; push.w.outBarycentric = bBary.addr;
    push.w.outContactInfo = bInfo.addr; push.w.totalContactCount = bTotal.addr;
    push.w.outRigidMaterialIndex = bOutMat.addr;
    push.w.contactByRigid = bByX.addr; push.w.tempContactByRigid = bTmp.addr;
    push.w.contactIndexSortedByRigid = bIdx.addr;
    push.w.maxNumContacts = MAXC;

    Kernel k("softBodyContactGenBda.spv", "sb_primitiveContactGenLaunch", {}, sizeof(SbContactGenPush));
    k.dispatch(/*gridX*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    const auto gPt = downloadBuffer<F4>(dev, bPt.h, MAXC);
    const auto gNp = downloadBuffer<F4>(dev, bNp.h, MAXC);
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);
    const auto gMat = downloadBuffer<uint16_t>(dev, bOutMat.h, MAXC);
    const auto gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];

    // Corners 0, 1 and 2 are all within contactDist (0.10) of the plane; corner 3 is not.
    // Three from the PLANE pair; the BOX pair must contribute NOTHING in this tranche.
    PXG_CHECK_MSG(n == 3u, "the plane path must emit ONE contact PER PENETRATING CORNER "
                           "(3 of the tet's 4 are within 0.10 of the plane) and the far BOX pair "
                           "must contribute nothing, got %u", n);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        PXG_CHECK_MSG(gMat[i] != 9u,
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
        // 5.12: ref1 = PxgEncodeDeformableTetrahedron(0, 0); the material is a PxU16 stream.
        PXG_CHECK_MSG(gInfo[i].ref1 == sbTetRef(0u, 0u), "ref1 must be the tetrahedron ref of (softbody 0, tet 0), got 0x%llx",
                      (unsigned long long)gInfo[i].ref1);
        PXG_CHECK_MSG(gMat[i] == 7u, "the rigid material index must be stored as a PxU16 (got 0x%04x)", gMat[i]);
        PXG_CHECK_MSG(gInfo[i].ref0 == 0x00000000ABCDEF01ull,
                      "ref0 must be the rigid's PxNodeIndex from the remap table");
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
        PXG_CHECK_MSG(gTmp[i] == 0xABCDEF01u, "tempContactByRigid must be the low half of the rigid id");
        PXG_CHECK_MSG(gIdx[i] == i, "contactIndexSortedByRigid must be seeded with the identity");
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
    // P9.86: Jacobi vertex 7 (lane 2) is PINNED with a -0 inverse mass; the CUDA's IEEE
    // `pos.w + 0.f` / `deltaPos.w + 0.f` (softBodyGM.cu:2707/2710) gives +0.  Vertex 6 is
    // outside the subset and must keep its -0 bits.
    pos[7].w = -0.0f; deltaPos[7].w = -0.0f; vel[7].w = -0.0f;
    pos[6].w = -0.0f; deltaPos[6].w = -0.0f;

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
    // P9.86 coverage: exactly one Jacobi vertex carries a -0 inverse mass into the literal
    // +0 w lane, and the IEEE reference turns it into +0.
    uint32_t negZeroJacobi = 0;
    for (uint32_t lane = 0; lane < nbJacobi; ++lane)
    {
        const uint32_t v = jacobiIdx[lane];
        if (f2u(pos[v].w) == 0x80000000u && f2u(refPos[v].w) == 0u &&
            f2u(deltaPos[v].w) == 0x80000000u && f2u(refDeltaPos[v].w) == 0u) ++negZeroJacobi;
    }
    PXG_CHECK_MSG(negZeroJacobi == 1u, "a -0 Jacobi inverse mass must meet the literal +0 w lane (%u)", negZeroJacobi);
    PXG_CHECK_MSG(f2u(gPos[7].w) == 0u && f2u(gDeltaPos[7].w) == 0u && f2u(gVel[7].w) == 0u,
                  "Jacobi vertex 7's -0 inverse mass must come back +0 (%08x %08x %08x)",
                  f2u(gPos[7].w), f2u(gDeltaPos[7].w), f2u(gVel[7].w));
    PXG_CHECK_MSG(f2u(gPos[6].w) == 0x80000000u && f2u(gDeltaPos[6].w) == 0x80000000u,
                  "vertex 6 is outside the Jacobi subset and must keep its -0 inverse mass");

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
// PhysX 5.12 replaced the softbody CONSUMER chain these fixtures pinned: the 5.6.1
// softbody-softbody / softbody-cloth / softbody-rigid constraint blocks, the
// PxgFEMRigidAttachmentConstraint / PxgFEMFEMAttachmentConstraint blocks, the packed
// PxgFemOtherContactInfo / PxgFemFemContactInfo ids, the separate TGS kernels and the
// solve-time reference counting are gone (PxgRigidDbContactBlock / PxgDbDbContactBlock /
// PxgRigidDbAttachmentBlock / PxgDbDbAttachmentBlock, PxgDeformablePairInfo, the
// sb_query*ReferenceCountLaunch pre-counts and the isTGS / isVelocityIteration arguments).
// The softBodySimRigidContact*, softBodySimRigidAttach*, softBodySimSelfContact*,
// softBodySimClothContactPrepare/Solve, softBodySimFemAttachment* and
// softBodySimClothAttachment* fixtures were written against those layouts and were deleted
// with them; testSoftBodySolve512.cpp covers the 5.12 kernels (prepares, pre-counts, solves,
// attachments, applyExternalDeltas) against a transcription of the 5.12 CUDA.
// ===========================================================================

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
//
// PhysX 5.12: the contact is written through PxgDbDbContactWriter::writeVolumeVolumeContact
// with ref0 = PxgEncodeDeformableVertex(body, query vertex) and ref1 =
// PxgEncodeDeformableTetrahedron(body, tet), and NO barycentric (5.6.1 wrote the vertex's
// surface tet and parked the two points in the barycentric slots).  The element filter is
// mVolumeVolumeFilterPairs, a sorted PxgDeformablePairInfo list searched for (vertex, ANY)
// per query vertex and for (vertex, tet) and (ANY, tet) per candidate.
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
    const uint32_t T_FILTANY = addTet(80.0f, 0.0f, 0.00f, false, 0xfu, 0.0f);  // (ANY vertex, tet) filter (5.12)
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
    const uint32_t q12 = addQuery(80.12f, 0.15f, 0.05f, 1u);   // dropped: (ANY, T_FILTANY) filter
    (void)q2; (void)q3; (void)q7; (void)q8;

    // Rest pose: corners keep their world position; every QUERY vertex is moved 5 units
    // clear of its target so the rest gate passes -- except q6, whose rest position IS its
    // world position, i.e. it was already penetrating at rest and must be dropped.
    std::vector<F4> rest = verts;
    const uint32_t movedAtRest[] = { q1, q2, q3, q4, q5, q7, q8, q9, q10, q11, q12, qShared };
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

    // vertToSurfaceTetRemap: a distinctive value per vertex.  5.12 no longer reads it (ref0
    // names the query VERTEX itself), so a port that still did would name tet 100 + v.
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

    // ---- filter pairs (PxgDeformablePairInfo, sorted by ref0, then ref1) -----
    const uint32_t SB_ID = 3u;   // NOT zero: the encode shift has to actually shift
    const uint32_t ANY = PxgDeformableRef::ANY_FEATURE;
    std::vector<PxgDeformablePairInfo> filters(3);
    filters[0].ref0 = PxgEncodeDeformableVertex(SB_ID, q4);  filters[0].ref1 = PxgEncodeDeformableTetrahedron(SB_ID, T_FILTPR);
    filters[1].ref0 = PxgEncodeDeformableVertex(SB_ID, q5);  filters[1].ref1 = PxgEncodeDeformableTetrahedron(SB_ID, ANY);
    filters[2].ref0 = PxgEncodeDeformableVertex(SB_ID, ANY); filters[2].ref1 = PxgEncodeDeformableTetrahedron(SB_ID, T_FILTANY);
    for (size_t f = 1; f < filters.size(); ++f)              // the binary search needs order
        PXG_CHECK(PxgComparePairInfo(filters[f - 1], filters[f]) < 0);

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
    sb.mVolumeVolumeFilterPairs = reinterpret_cast<decltype(sb.mVolumeVolumeFilterPairs)>(size_t(bFilters.addr));
    sb.mNumVolumeVolumeFilterPairs = uint32_t(filters.size());
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
    // 5.12 writes no barycentric for a volume-volume contact: both slots keep their poison.
    DevBuf bBary0 = makeDev(dev, poison);
    DevBuf bBary1 = makeDev(dev, poison);
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
    const auto gBary0 = downloadBuffer<uint32_t>(dev, bBary0.h, MAXC * 4u);
    const auto gBary1 = downloadBuffer<uint32_t>(dev, bBary1.h, MAXC * 4u);
    const auto gInfoW = downloadBuffer<uint32_t>(dev, bInfo.h, MAXC * 4u);
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);

    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
        std::printf("  contact %u: p=(%.5f %.5f %.5f) n=(%.5f %.5f %.5f) pen=%.6f ref0=0x%llx ref1=0x%llx\n",
                    i, gPoint[i].x, gPoint[i].y, gPoint[i].z, gNormalPen[i].x, gNormalPen[i].y,
                    gNormalPen[i].z, gNormalPen[i].w, (unsigned long long)gInfo[i].ref0,
                    (unsigned long long)gInfo[i].ref1);
    for (uint32_t w = 0; w < MAXC * 4u; ++w)
        PXG_CHECK_MSG(gBary0[w] == POISON && gBary1[w] == POISON,
                      "a volume-volume contact writes no barycentric (word %u: 0x%08x 0x%08x)", w, gBary0[w], gBary1[w]);

    PXG_CHECK_MSG(gCount == 4u,
                  "exactly four query vertices survive every gate (q1, q9, q10, q11) -- got "
                  "%u.  0 means the body-flag early-out or the activeSoftbodies indirection "
                  "is wrong; more than 4 means a gate was dropped or a vertex was visited "
                  "more than once", gCount);

    // ---- match each emitted contact to the query it must have come from ------
    auto near6 = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };
    bool sawQ1 = false, sawQ9 = false, sawQ10 = false, sawQ11 = false;
    auto vref = [&](uint32_t v) { return PxgEncodeDeformableVertex(SB_ID, v); };
    auto tref = [&](uint32_t t) { return PxgEncodeDeformableTetrahedron(SB_ID, t); };
    for (uint32_t i = 0; i < gCount; ++i)
    {
        const uint64_t ind0 = gInfo[i].ref0;
        const uint64_t ind1 = gInfo[i].ref1;

        if (ind0 == vref(q1))
        {
            sawQ1 = true;
            PXG_CHECK_MSG(ind1 == tref(T_DEEP),
                          "q1 penetrates T_DEEP by 0.05 and T_SHALLOW by 0.03; the cross-lane "
                          "argmin must pick the DEEPER one (got tet %u, want %u)",
                          PxgGetDeformableFeatureId(ind1), T_DEEP);
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
            PXG_CHECK(near6(gNormalPen[i].x, 0.0f) && near6(gNormalPen[i].y, 0.0f) &&
                      near6(gNormalPen[i].z, -1.0f));
            PXG_CHECK(near6(gPoint[i].x, 0.12f) && near6(gPoint[i].y, 0.15f) && near6(gPoint[i].z, 0.0f));
            PXG_CHECK_MSG(gPoint[i].w == 0.0f, "point.w is 0 for a self contact (got %g)", gPoint[i].w);
        }
        else if (ind0 == vref(q9))
        {
            sawQ9 = true;
            PXG_CHECK(ind1 == tref(T_GOOD));
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
        }
        else if (ind0 == vref(q10))
        {
            sawQ10 = true;
            PXG_CHECK(ind1 == tref(T_GOOD));
            // The OUTSIDE branch: a positive separation, and the normal replaced by the
            // normalised separation vector because their dot exceeds 0.707.
            PXG_CHECK_MSG(near6(gNormalPen[i].w, 0.07f),
                          "q10 sits 0.07 OUTSIDE T_GOOD's x-face; the outside branch reports "
                          "a positive distance (got %.6f)", gNormalPen[i].w);
            PXG_CHECK(near6(gNormalPen[i].x, -1.0f));
        }
        else if (ind0 == vref(q11))
        {
            sawQ11 = true;
            PXG_CHECK(ind1 == tref(T_GOOD));
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
            PXG_CHECK_MSG(false, "contact %u came from an unexpected vertex (ref0 = 0x%llx, "
                                 "feature %u)", i, (unsigned long long)ind0, PxgGetDeformableFeatureId(ind0));
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
        PXG_CHECK_MSG(gInfoW[i * 4u + 0u] == POISON,
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
            d.mVolumeVolumeFilterPairs = nullptr;
            d.mNumVolumeVolumeFilterPairs = 0u;
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

        // q4, q5 and q12 were dropped ONLY by the filters, so with no filter list they
        // survive: the four contacts above become seven.  An empty list must filter nothing
        // -- not everything, and not something lane-dependent.
        PXG_CHECK_MSG(n4 == 7u,
                      "with mVolumeVolumeFilterPairs = NULL the three filtered vertices (q4, q5, q12) "
                      "must come back, giving 7 contacts, got %u.  A null pairs pointer must be READ AS "
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
// P8.4 tranche b — sb_fem_contact_remap_to_simLaunch, the DEFORMABLE-vs-DEFORMABLE remap
// (softbodyPrimitives.cu, PhysX 5.12).
//
// The search arithmetic is the sibling's and is covered by the fixtures above.  What this
// pins is the 5.12 shape of the two y-grid rows:
//   row 0 (side 1): ref1 names a COLLISION tetrahedron met at the contact; the witness comes
//          from the `point` stream; the winner (containment, else nearest) is written to ref1
//          and outBarycentric1; with no geometrically valid candidate ref1 becomes the
//          INVALID reference and the barycentric is zeroed;
//   row 1 (side 0): ref0 names a collision-mesh VERTEX; a lookup, not a search: ref0 becomes
//          the simulation tetrahedron mVertsRemapInGridModel[v] and outBarycentric0 receives
//          mVertsBarycentricInGridModel[v].
// Each row touches only its own side.  The count clamp is written back once, and
// prevContactCount skips the contacts already remapped.
// ===========================================================================
// sb_fem_contact_remap_to_simLaunch (5.12): { softbodies*, point*, outBarycentric0*,
// outBarycentric1*, outContactInfo*, totalContactCount*, prevContactCount*, PxU32 maxNbContacts }
// -> 64-byte blob.
struct SbFemRemapPush
{
    uint64_t softbodies, point, outBarycentric0, outBarycentric1, outContactInfo,
             totalContactCount, prevContactCount;
    uint32_t maxNbContacts, pad60;
};
static_assert(sizeof(SbFemRemapPush) == 64, "fem remap push blob must be 64 bytes");
static_assert(offsetof(SbFemRemapPush, maxNbContacts) == 56, "maxNbContacts @56");

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
    // The collision-vertex embedding the side-0 row looks up: distinct per vertex.
    std::vector<uint32_t> vertRemap = { 4u, 2u, 0u, 3u, 1u, 0u, 0u, 0u };
    std::vector<F4> vertBary = { { 0.1f, 0.2f, 0.3f, 0.4f }, { 0.7f, 0.1f, 0.1f, 0.1f },
                                 { 0.25f, 0.25f, 0.25f, 0.25f }, { -0.0f, 0.5f, 0.5f, 0.0f },
                                 { 1.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0, 0 } };

    DevBuf bSv = makeDev(dev, simVerts), bSt = makeDev(dev, simTets);
    DevBuf bRc = makeDev(dev, remapColToSim), bAr = makeDev(dev, accumRemap);
    DevBuf bVr = makeDev(dev, vertRemap), bVb = makeDev(dev, vertBary);
    if (!bSv.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // Softbody 1 is the real one; softbody 0 is a zero decoy, so a remap that drops the id
    // reads null pointers.
    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bSt.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSv.addr));
    sb.mTetsRemapColToSim = reinterpret_cast<PxU32*>(size_t(bRc.addr));
    sb.mTetsAccumulatedRemapColToSim = reinterpret_cast<PxU32*>(size_t(bAr.addr));
    sb.mVertsRemapInGridModel = reinterpret_cast<PxU32*>(size_t(bVr.addr));
    sb.mVertsBarycentricInGridModel = reinterpret_cast<float4*>(size_t(bVb.addr));
    std::vector<PxgSoftBody> descArr = { makeZeroDesc(), sb };
    DevBuf bDesc = makeDev(dev, descArr);

    const uint32_t SB = 1u;
    // Three contacts.  Side 0 names a collision VERTEX, side 1 a collision TETRAHEDRON:
    //   contact 0: vertex 3, tet 0, point (0.2,0.2,0.2) -> CONTAINED by sim tet 2
    //   contact 1: vertex 1, tet 0, point (1.6,0.2,0.2) -> outside all, NEAREST sim tet 1
    //   contact 2: vertex 4, tet 1 (only degenerate candidates) -> INVALID ref, zero barycentric
    const uint32_t vOf[3] = { 3u, 1u, 4u }, tOf[3] = { 0u, 0u, 1u };
    std::vector<PxgDeformablePairInfo> infos(4);
    std::memset(infos.data(), 0, infos.size() * sizeof(PxgDeformablePairInfo));
    for (uint32_t i = 0; i < 3u; ++i)
    {
        infos[i].ref0 = PxgEncodeDeformableVertex(SB, vOf[i]);
        infos[i].ref1 = PxgEncodeDeformableTetrahedron(SB, tOf[i]);
    }
    DevBuf bInfo = makeDev(dev, infos);
    std::vector<F4> points = { { 0.2f, 0.2f, 0.2f, 0.0f }, { 1.6f, 0.2f, 0.2f, 0.0f },
                               { 5.0f, 5.0f, 5.0f, 0.0f }, { 0, 0, 0, 0 } };
    DevBuf bPt = makeDev(dev, points);
    const F4 SENT = { -7.0f, -7.0f, -7.0f, -7.0f };
    std::vector<F4> bary0(4, SENT), bary1(4, SENT);
    DevBuf bB0 = makeDev(dev, bary0), bB1 = makeDev(dev, bary1);
    std::vector<uint32_t> count = { 3u, 0u, 0u, 0u }, prevZero = { 0u, 0u, 0u, 0u };
    DevBuf bCount = makeDev(dev, count), bPrev0 = makeDev(dev, prevZero);

    SbFemRemapPush push = {};
    push.softbodies = bDesc.addr;
    push.point = bPt.addr;
    push.outBarycentric0 = bB0.addr;
    push.outBarycentric1 = bB1.addr;
    push.outContactInfo = bInfo.addr;
    push.totalContactCount = bCount.addr;
    push.prevContactCount = bPrev0.addr;   // the CUDA dereferences it unconditionally
    push.maxNbContacts = 64u;

    // THE ENGINE'S GRID: gridDim.y == 2.  x is deliberately OVER-SIZED relative to what three
    // contacts need, so an inverted or off-by-one `gid.x >= SB_REMAP_GRID` early-out shows up as
    // either nothing happening or work happening twice.
    Kernel k("softBodyContactRemapBda.spv", "sb_fem_contact_remap_to_simLaunch", {}, sizeof(SbFemRemapPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 2u, 1u);
    dev->flush();

    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, 4);
    const auto g0 = downloadBuffer<F4>(dev, bB0.h, 4);
    const auto g1 = downloadBuffer<F4>(dev, bB1.h, 4);
    for (uint32_t i = 0; i < 3u; ++i)
        std::printf("  contact %u: ref0 0x%llx ref1 0x%llx | bary0(%.4f %.4f %.4f %.4f) bary1(%.4f %.4f %.4f %.4f)\n",
                    i, (unsigned long long)gInfo[i].ref0, (unsigned long long)gInfo[i].ref1,
                    g0[i].x, g0[i].y, g0[i].z, g0[i].w, g1[i].x, g1[i].y, g1[i].z, g1[i].w);

    auto near4 = [](float a, float b) { return std::fabs(a - b) < 1e-4f; };
    auto same4 = [](F4 a, F4 b) { return f2u(a.x) == f2u(b.x) && f2u(a.y) == f2u(b.y) && f2u(a.z) == f2u(b.z) && f2u(a.w) == f2u(b.w); };

    // ---- row 0: side 1, the search ----------------------------------------------------
    PXG_CHECK_MSG(gInfo[0].ref1 == PxgEncodeDeformableTetrahedron(SB, 2u),
                  "contact 0's witness is CONTAINED by sim tet 2 (ref1 0x%llx)", (unsigned long long)gInfo[0].ref1);
    PXG_CHECK_MSG(near4(g1[0].x, 0.4f) && near4(g1[0].y, 0.2f) && near4(g1[0].z, 0.2f) && near4(g1[0].w, 0.2f),
                  "and its barycentric must be (0.4,0.2,0.2,0.2), got (%.4f %.4f %.4f %.4f)",
                  g1[0].x, g1[0].y, g1[0].z, g1[0].w);
    PXG_CHECK_MSG(gInfo[1].ref1 == PxgEncodeDeformableTetrahedron(SB, 1u),
                  "contact 1's witness is outside every candidate, so the NEAREST (sim tet 1) must win "
                  "(ref1 0x%llx)", (unsigned long long)gInfo[1].ref1);
    // the hasTet branch: no valid candidate -> the invalid reference, zero barycentric
    PXG_CHECK_MSG(gInfo[2].ref1 == PxgEncodeInvalidDeformableRef(SB),
                  "contact 2's candidates are all DEGENERATE, so ref1 must be the invalid reference "
                  "(got 0x%llx)", (unsigned long long)gInfo[2].ref1);
    PXG_CHECK_MSG(same4(g1[2], F4{ 0.0f, 0.0f, 0.0f, 0.0f }),
                  "...and its barycentric must be ZEROED, got (%.4f %.4f %.4f %.4f)", g1[2].x, g1[2].y, g1[2].z, g1[2].w);

    // ---- row 1: side 0, the vertex lookup ---------------------------------------------
    for (uint32_t i = 0; i < 3u; ++i)
    {
        PXG_CHECK_MSG(gInfo[i].ref0 == PxgEncodeDeformableTetrahedron(SB, vertRemap[vOf[i]]),
                      "contact %u's ref0 must become the sim tet mVertsRemapInGridModel[%u] = %u (got 0x%llx)",
                      i, vOf[i], vertRemap[vOf[i]], (unsigned long long)gInfo[i].ref0);
        PXG_CHECK_MSG(same4(g0[i], vertBary[vOf[i]]),
                      "contact %u's outBarycentric0 must be mVertsBarycentricInGridModel[%u] bit for bit", i, vOf[i]);
    }
    PXG_CHECK_MSG(same4(g0[3], SENT) && same4(g1[3], SENT), "contact 3 is past the count and must be untouched");

    // ---- each row touches only its own side -------------------------------------------
    {
        std::vector<PxgDeformablePairInfo> i2 = infos;
        std::vector<F4> a0 = bary0, a1 = bary1;
        DevBuf bI2 = makeDev(dev, i2), bA0 = makeDev(dev, a0), bA1 = makeDev(dev, a1);
        std::vector<uint32_t> c2 = { 3u, 0u, 0u, 0u };
        DevBuf bC2 = makeDev(dev, c2);
        SbFemRemapPush p2 = push;
        p2.outContactInfo = bI2.addr; p2.outBarycentric0 = bA0.addr;
        p2.outBarycentric1 = bA1.addr; p2.totalContactCount = bC2.addr;
        // gridY = 1: row 0 only
        Kernel("softBodyContactRemapBda.spv", "sb_fem_contact_remap_to_simLaunch", {}, sizeof(SbFemRemapPush))
            .dispatch(8192u, {}, p2, 1u, 1u);
        dev->flush();
        const auto j2 = downloadBuffer<PxgDeformablePairInfo>(dev, bI2.h, 4);
        const auto d0 = downloadBuffer<F4>(dev, bA0.h, 4);
        const auto d1 = downloadBuffer<F4>(dev, bA1.h, 4);
        for (uint32_t i = 0; i < 3u; ++i)
        {
            PXG_CHECK_MSG(j2[i].ref0 == infos[i].ref0 && same4(d0[i], SENT),
                          "row 0 must not touch side 0 (contact %u)", i);
            // The same result at the engine's OTHER x-grid (8192 for softbody-softbody).
            PXG_CHECK_MSG(j2[i].ref1 == gInfo[i].ref1 && same4(d1[i], g1[i]),
                          "contact %u must remap identically at gridX 8192 and 1024", i);
        }
        dev->destroyBuffer(bI2.h); dev->destroyBuffer(bA0.h);
        dev->destroyBuffer(bA1.h); dev->destroyBuffer(bC2.h);
    }

    // ---- the clamp write-back, and prevContactCount ------------------------------------
    {
        std::vector<PxgDeformablePairInfo> i3 = infos;
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
        const auto j3 = downloadBuffer<PxgDeformablePairInfo>(dev, bI3.h, 4);
        const auto e0 = downloadBuffer<F4>(dev, bA0.h, 4);
        const auto e1 = downloadBuffer<F4>(dev, bA1.h, 4);
        PXG_CHECK_MSG(back == 2u, "an over-large contact count must be CLAMPED and written back "
                                  "(9 -> maxNbContacts 2), got %u", back);
        PXG_CHECK_MSG(j3[0].ref0 == infos[0].ref0 && j3[0].ref1 == infos[0].ref1 && same4(e0[0], SENT) &&
                      same4(e1[0], SENT), "prevContactCount = 1 must SKIP contact 0 on both rows");
        PXG_CHECK_MSG(j3[1].ref1 == gInfo[1].ref1 && j3[1].ref0 == gInfo[1].ref0,
                      "...and must still remap contact 1 on both rows");
        PXG_CHECK_MSG(j3[2].ref0 == infos[2].ref0 && j3[2].ref1 == infos[2].ref1 && same4(e0[2], SENT),
                      "contact 2 is past the CLAMPED count and must be untouched");
        dev->destroyBuffer(bI3.h); dev->destroyBuffer(bA0.h); dev->destroyBuffer(bA1.h);
        dev->destroyBuffer(bC3.h); dev->destroyBuffer(bPrev.h);
    }

    for (DevBuf* b : { &bSv, &bSt, &bRc, &bAr, &bVr, &bVb, &bDesc, &bInfo, &bPt, &bB0, &bB1, &bCount, &bPrev0 })
        dev->destroyBuffer(b->h);
}

// ===========================================================================
// P9.86c -- the SIGN OF ZERO (and NaN) through the remap's PxClamp.
// computeTetBarycentric (deformableCollision.cuh:133-136) and the sb_other fallback
// (softbodyPrimitives.cu:2261-2264) clamp with PxClamp = selectMin(1, selectMax(0, t)),
// which keeps -0 and NaN; Slang clamp() is GLSL FClamp, which gives +0 / 0 on llvmpipe.
// The fixtures above only produce +0 and compare with tolerances.  Here a positively
// oriented T0 and an inverted T1 put the contact (or its closest point) exactly on a
// vertex, so the ratios are exact signed zeros (-0 in .y, and in .z/.w for T1); the
// degenerate T2 gives 0/0.  Every output is compared bitwise against a host mirror.
// ===========================================================================
namespace sbRemapZref
{
using p80refit::TetIdx4;
struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }             // PxVec3::dot
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return V3{ f.x, f.y, f.z }; }

// 0 = the CUDA (PxClamp = selectMin(1, selectMax(0, t)) at deformableCollision.cuh:133-136 and
// softbodyPrimitives.cu:2261-2264); 1 = FClamp twin inside computeTetBarycentric (revert r1);
// 2 = FClamp twin in the other-remap fallback clamp only (revert r2).
static int gClampVariant = 0;
static float pxClamp01(float t) { const float m = (0.0f > t) ? 0.0f : t; return (1.0f < m) ? 1.0f : m; }
// llvmpipe GLSL FClamp(t, 0, 1) = fmin(fmax(t, +0), 1) with ordered zeros (P9.86 F4): -0 -> +0.
static float fclampTwin(float t) { if (t != t) return 0.0f; const float m = (t > 0.0f) ? t : 0.0f; return (m < 1.0f) ? m : 1.0f; }

// Gu::closestPtPointTriangle2 (GuDistancePointTriangle.h:44), CUDA association kept.
static V3 closestPtPointTriangle2(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return add(a, mul(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return add(a, mul(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, mul(sub(c, b), w));
    }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    return add(add(a, mul(ab, v)), mul(ac, w));                                            // a + ab*v + ac*w
}
// Gu::closestPtPointTetrahedron (GuDistancePointTetrahedron.h:77), strict < keeps the first face.
static V3 closestPtPointTetrahedron(V3 p, V3 a, V3 b, V3 c, V3 d)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ad = sub(d, a), bc = sub(c, b), bd = sub(d, b);
    V3 best = closestPtPointTriangle2(p, a, b, c, ab, ac);
    V3 diff = sub(best, p);
    float bestSq = dot3(diff, diff);
    const V3 f[3] = { closestPtPointTriangle2(p, a, c, d, ac, ad), closestPtPointTriangle2(p, a, d, b, ad, ab),
                      closestPtPointTriangle2(p, b, d, c, bd, bc) };
    for (int i = 0; i < 3; ++i)
    {
        diff = sub(f[i], p);
        const float sq = dot3(diff, diff);
        if (sq < bestSq) { best = f[i]; bestSq = sq; }
    }
    return best;
}
// computeTetBarycentric (deformableCollision.cuh:96).
static bool computeTetBarycentric(V3 a, V3 b, V3 c, V3 d, V3 point, F4& bary)
{
    const float eps = 1e-5f, minEps = -eps, maxEps = 1.0f + eps;
    const V3 ba = sub(b, a), ca = sub(c, a), da = sub(d, a), pa = sub(point, a);
    const float detBcd = dot3(ba, cross3(ca, da));
    const float v = dot3(pa, cross3(ca, da)) / detBcd;
    const float w = dot3(ba, cross3(pa, da)) / detBcd;
    const float x = dot3(ba, cross3(ca, pa)) / detBcd;
    const float u = 1.0f - v - w - x;
    float (*cl)(float) = (gClampVariant == 1) ? fclampTwin : pxClamp01;
    bary = F4{ cl(u), cl(v), cl(w), cl(x) };
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}
static bool isValidTet(V3 a, V3 b, V3 c, V3 d) { return dot3(sub(b, a), cross3(sub(c, a), sub(d, a))) > 1e-9f; }

struct Scene { std::vector<F4> verts; std::vector<TetIdx4> tets; std::vector<uint32_t> remap, accum; };
struct Out { uint32_t simTet; F4 bary; };

// One warp of sb_other_contact_remap_to_simLaunch (fem == false, softbodyPrimitives.cu:2134) or
// sb_fem_contact_remap_to_simLaunch (fem == true, :2277) for one contact: 32 lanes in order,
// ballot + lowestSetIndex, warpReduction<MinOpFloat> with the lowest lane winning ties.
static Out remapRef(const Scene& sc, V3 contact, uint32_t colTet, bool fem)
{
    const uint32_t endInd = sc.accum[colTet];
    const uint32_t startInd = (colTet != 0u) ? sc.accum[colTet - 1u] : 0u;
    const uint32_t range = endInd - startInd;
    uint32_t laneTet[32]; F4 laneBary[32]; float laneSq[32];
    for (int l = 0; l < 32; ++l) { laneTet[l] = 0xffffffffu; laneBary[l] = F4{ 0, 0, 0, 0 }; laneSq[l] = 3.402823466e38f; }
    bool hasTet = false;
    for (uint32_t j = 0; j < (range + 31u) / 32u; ++j)
    {
        for (uint32_t l = 0; l < 32u; ++l)
        {
            const uint32_t wi = l + j * 32u;
            if (wi >= range) continue;
            const uint32_t st = sc.remap[startInd + wi];
            const TetIdx4 t = sc.tets[st];
            const V3 a = v3(sc.verts[t.x]), b = v3(sc.verts[t.y]), c = v3(sc.verts[t.z]), d = v3(sc.verts[t.w]);
            if (fem && !isValidTet(a, b, c, d)) continue;
            hasTet = true;
            F4 tb;
            if (computeTetBarycentric(a, b, c, d, contact, tb)) return Out{ st, tb };   // lowest hitting lane
            const V3 cp = closestPtPointTetrahedron(contact, a, b, c, d);
            const V3 dv = sub(contact, cp);
            const float sq = dot3(dv, dv);
            if (sq < laneSq[l]) { laneSq[l] = sq; laneTet[l] = st; computeTetBarycentric(a, b, c, d, cp, laneBary[l]); }
        }
    }
    // 5.12: no geometrically valid candidate -> the invalid reference simTetRef started as.
    if (fem && !hasTet) return Out{ 0xffffffffu, F4{ 0.0f, 0.0f, 0.0f, 0.0f } };
    float best = 3.402823466e38f;
    for (int l = 0; l < 32; ++l) best = (laneSq[l] < best) ? laneSq[l] : best;
    int w = 0;
    for (int l = 0; l < 32; ++l) if (laneSq[l] == best) { w = l; break; }
    F4 bb = laneBary[w];
    if (!fem)   // softbodyPrimitives.cu:2261-2264; the FEM twin (:2427-2430) is commented out
    {
        float (*cl)(float) = (gClampVariant == 2) ? fclampTwin : pxClamp01;
        bb = F4{ cl(bb.x), cl(bb.y), cl(bb.z), cl(bb.w) };
    }
    return Out{ laneTet[w], bb };
}
// The ref1 the remap writes (5.12): the winning simulation tetrahedron, or the invalid
// reference (PxgEncodeInvalidDeformableRef) the search starts from.
static uint64_t outRef(const Out& o, uint32_t softbodyId)
{
    return (o.simTet == 0xffffffffu) ? PxgEncodeInvalidDeformableRef(softbodyId)
                                     : PxgEncodeDeformableTetrahedron(softbodyId, o.simTet);
}
// Bitwise, except that any NaN matches any NaN: the CUDA's 0/0 is 0x7fffffff, x86 gives 0xffc00000.
static bool sameLane(float a, float b) { return (a != a && b != b) || f2u(a) == f2u(b); }
static bool sameBits(F4 a, F4 b) { return sameLane(a.x, b.x) && sameLane(a.y, b.y) && sameLane(a.z, b.z) && sameLane(a.w, b.w); }
} // namespace sbRemapZref

PXG_TEST(softBodySimContactRemapZeroSign)
{
    namespace Z = sbRemapZref;
    PxgSlangComputeDevice* dev = testDevice();

    Z::Scene sc;
    sc.verts = { {  0.0f,  0.0f,  0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, 1.0f },
                 {  0.0f,  1.0f, -1.0f, 1.0f }, {  1.0f, -1.0f,  0.0f, 1.0f },    // T0, det +3
                 {  2.0f,  0.0f,  0.0f, 1.0f }, {  3.0f,  0.0f,  0.0f, 1.0f },
                 {  2.0f,  0.0f,  1.0f, 1.0f }, {  2.0f,  1.0f,  0.0f, 1.0f },    // T1, det -1
                 { 20.0f,  0.0f,  0.0f, 1.0f }, { 21.0f,  0.0f,  0.0f, 1.0f },
                 { 20.0f,  1.0f,  0.0f, 1.0f } };                                 // T2 = {a,b,c,a}, det 0
    sc.tets = { { 0u, 1u, 2u, 3u }, { 4u, 5u, 6u, 7u }, { 8u, 9u, 10u, 8u } };
    sc.remap = { 0u, 1u, 2u, 0u };
    sc.accum = { 1u, 2u, 3u, 3u };   // collision tet c -> {Tc}

    std::vector<F4> vtx = sc.verts;
    for (int t = 0; t < 4; ++t) vtx.push_back(F4{ 0, 0, 0, 0 });   // trailer, as the sibling fixture
    DevBuf bSv = makeDev(dev, vtx), bSt = makeDev(dev, sc.tets);
    DevBuf bRc = makeDev(dev, sc.remap), bAr = makeDev(dev, sc.accum);
    if (!bSv.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    PxgSoftBody sb = makeZeroDesc();
    sb.mSimTetIndices = reinterpret_cast<decltype(sb.mSimTetIndices)>(size_t(bSt.addr));
    sb.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(bSv.addr));
    sb.mTetsRemapColToSim = reinterpret_cast<PxU32*>(size_t(bRc.addr));
    sb.mTetsAccumulatedRemapColToSim = reinterpret_cast<PxU32*>(size_t(bAr.addr));
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- sb_other_contact_remap_to_simLaunch: r1 via all five, r2 via the three fallbacks ----
    // Contact 4 is the NaN hedge: the degenerate T2 (no isValidTet gate in this kernel) gives
    // 0/0 in every lane, which PxClamp keeps and FClamp turns into 0 -- a kill that does not
    // rest on FClamp's zero sign.
    const Z::V3 P[5] = { { 0.1f,  0.2f,  0.3f },    // col 0: outside T0, vertex region of a -> FALLBACK
                         { 0.0f,  0.0f,  0.0f },    // col 0: AT a -> CONTAINED, v = -0
                         { 1.5f, -0.5f, -0.5f },    // col 1: outside T1, vertex region of a -> FALLBACK
                         { 2.0f,  0.0f,  0.0f },    // col 1: AT a -> CONTAINED, v = w = x = -0
                         { 20.2f, 0.2f,  0.5f } };  // col 2: degenerate T2 -> FALLBACK, closest z = 0, all NaN
    const uint32_t colOf[5] = { 0u, 0u, 1u, 1u, 2u };
    std::vector<F4> points;
    for (int i = 0; i < 5; ++i) points.push_back(F4{ P[i].x, P[i].y, P[i].z, 0.0f });
    std::vector<PxgDeformablePairInfo> infos(5);
    std::memset(infos.data(), 0, infos.size() * sizeof(PxgDeformablePairInfo));
    for (uint32_t i = 0; i < 5u; ++i)
    {
        infos[i].ref0 = 0x100ull + i; infos[i].ref1 = PxgEncodeDeformableTetrahedron(0u, colOf[i]);
    }
    std::vector<F4> bary(5, F4{ -1, -1, -1, -1 });
    std::vector<uint32_t> count = { 5u, 0u, 0u, 0u };
    DevBuf bPt = makeDev(dev, points), bInfo = makeDev(dev, infos);
    DevBuf bBary = makeDev(dev, bary), bCount = makeDev(dev, count);

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
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, 5);
    const auto gBary = downloadBuffer<F4>(dev, bBary.h, 5);

    uint32_t negZeroY = 0, nanSeen = 0, r1TwinDiffers = 0, r2TwinDiffers = 0;
    for (uint32_t i = 0; i < 5u; ++i)
    {
        Z::gClampVariant = 1; const Z::Out t1 = Z::remapRef(sc, P[i], colOf[i], /*fem*/ false);
        Z::gClampVariant = 2; const Z::Out t2 = Z::remapRef(sc, P[i], colOf[i], /*fem*/ false);
        Z::gClampVariant = 0; const Z::Out e  = Z::remapRef(sc, P[i], colOf[i], /*fem*/ false);
        if (f2u(e.bary.y) == 0x80000000u) ++negZeroY;
        for (const float c : { e.bary.x, e.bary.y, e.bary.z, e.bary.w }) if (c != c) ++nanSeen;
        if (!Z::sameBits(e.bary, t1.bary)) ++r1TwinDiffers;
        if (!Z::sameBits(e.bary, t2.bary)) ++r2TwinDiffers;
        PXG_CHECK_MSG(gInfo[i].ref1 == Z::outRef(e, 0u),
                      "other remap contact %u -> ref1 0x%llx, CUDA 0x%llx", i, (unsigned long long)gInfo[i].ref1,
                      (unsigned long long)Z::outRef(e, 0u));
        PXG_CHECK_MSG(Z::sameBits(gBary[i], e.bary),
                      "other remap contact %u: barycentric bits %08x %08x %08x %08x, CUDA %08x %08x %08x %08x "
                      "(PxClamp keeps -0, FClamp gives +0)", i,
                      f2u(gBary[i].x), f2u(gBary[i].y), f2u(gBary[i].z), f2u(gBary[i].w),
                      f2u(e.bary.x), f2u(e.bary.y), f2u(e.bary.z), f2u(e.bary.w));
    }
    PXG_CHECK_MSG(negZeroY == 4u, "contacts 0-3 must each carry a -0 barycentric (%u/4)", negZeroY);
    PXG_CHECK_MSG(nanSeen == 4u, "the degenerate contact 4 must carry four NaN lanes (%u/4)", nanSeen);
    PXG_CHECK_MSG(r1TwinDiffers == 5u,
                  "the FClamp-in-computeTetBarycentric twin must change all five contacts (%u/5)", r1TwinDiffers);
    PXG_CHECK_MSG(r2TwinDiffers == 3u,
                  "the FClamp-in-fallback twin must change the three FALLBACK contacts (%u/3)", r2TwinDiffers);

    // ---- sb_fem_contact_remap_to_simLaunch: the helper clamp only ----
    // 5.12: only side 1 (the tetrahedron met at the contact point, the `point` stream) is
    // searched; side 0 is a collision-mesh vertex looked up in mVertsRemapInGridModel, which
    // has no barycentric arithmetic and is pinned by softBodySimFemContactRemap.
    const Z::V3 Q1[3] = { P[0], P[1], P[0] };   // ref1 side, the witness in `point`
    const uint32_t fcol[3] = { 0u, 0u, 1u };    // contact 2: only the inverted T1 -> hasTet false control
    std::vector<PxgDeformablePairInfo> finfo(4);
    std::memset(finfo.data(), 0, finfo.size() * sizeof(PxgDeformablePairInfo));
    std::vector<F4> fpt(4, F4{ 0, 0, 0, 0 }), fb0(4, F4{ -7, -7, -7, -7 }), fb1(4, F4{ -7, -7, -7, -7 });
    for (uint32_t k = 0; k < 3u; ++k)
    {
        finfo[k].ref0 = 0x5A5A0000ull + k;   // side 0 is not remapped by the y-grid 0 dispatch below
        finfo[k].ref1 = PxgEncodeDeformableTetrahedron(0u, fcol[k]);
        fpt[k] = F4{ Q1[k].x, Q1[k].y, Q1[k].z, 0.0f };
    }
    std::vector<uint32_t> fcount = { 3u, 0u, 0u, 0u }, fprev = { 0u, 0u, 0u, 0u };
    DevBuf bFi = makeDev(dev, finfo), bF0 = makeDev(dev, fb0), bF1 = makeDev(dev, fb1);
    DevBuf bFc = makeDev(dev, fcount), bFp = makeDev(dev, fprev), bFpt = makeDev(dev, fpt);
    SbFemRemapPush pf = {};
    pf.softbodies = bDesc.addr;
    pf.point = bFpt.addr;
    pf.outBarycentric0 = bF0.addr;
    pf.outBarycentric1 = bF1.addr;
    pf.outContactInfo = bFi.addr;
    pf.totalContactCount = bFc.addr;
    pf.prevContactCount = bFp.addr;   // the CUDA dereferences it unconditionally
    pf.maxNbContacts = 64u;
    // gridY = 1: the side-1 row alone (what softbodyFemContactApplyCollisionToSimMeshMapping
    // launches for a pair whose side 0 is not a collision vertex).
    Kernel("softBodyContactRemapBda.spv", "sb_fem_contact_remap_to_simLaunch", {}, sizeof(SbFemRemapPush))
        .dispatch(1024u, {}, pf, 1u, 1u);
    dev->flush();
    const auto gFi = downloadBuffer<PxgDeformablePairInfo>(dev, bFi.h, 4);
    const auto g0 = downloadBuffer<F4>(dev, bF0.h, 4);
    const auto g1 = downloadBuffer<F4>(dev, bF1.h, 4);

    uint32_t femNegZeroY = 0, femR1TwinDiffers = 0, femInvalid = 0;
    for (uint32_t k = 0; k < 3u; ++k)
    {
        Z::gClampVariant = 1; const Z::Out t1 = Z::remapRef(sc, Q1[k], fcol[k], /*fem*/ true);
        Z::gClampVariant = 0; const Z::Out e  = Z::remapRef(sc, Q1[k], fcol[k], /*fem*/ true);
        if (e.simTet == 0xffffffffu) ++femInvalid;
        else
        {
            if (f2u(e.bary.y) == 0x80000000u) ++femNegZeroY;
            if (!Z::sameBits(e.bary, t1.bary)) ++femR1TwinDiffers;
        }
        PXG_CHECK_MSG(gFi[k].ref1 == Z::outRef(e, 0u), "fem remap contact %u -> ref1 0x%llx, CUDA 0x%llx",
                      k, (unsigned long long)gFi[k].ref1, (unsigned long long)Z::outRef(e, 0u));
        PXG_CHECK_MSG(gFi[k].ref0 == finfo[k].ref0, "fem remap contact %u: the side-1 row must not touch ref0", k);
        PXG_CHECK_MSG(Z::sameBits(g1[k], e.bary),
                      "fem remap contact %u: barycentric bits %08x %08x %08x %08x, CUDA "
                      "%08x %08x %08x %08x", k, f2u(g1[k].x), f2u(g1[k].y), f2u(g1[k].z), f2u(g1[k].w),
                      f2u(e.bary.x), f2u(e.bary.y), f2u(e.bary.z), f2u(e.bary.w));
        PXG_CHECK_MSG(f2u(g0[k].x) == f2u(-7.0f), "fem remap contact %u: the side-1 row must not write outBarycentric0", k);
    }
    PXG_CHECK_MSG(femNegZeroY == 2u, "the two valid-tet FEM writes must carry a -0 barycentric (%u/2)", femNegZeroY);
    PXG_CHECK_MSG(femR1TwinDiffers == 2u,
                  "the FClamp-in-computeTetBarycentric twin must change the two FEM writes (%u/2)", femR1TwinDiffers);
    PXG_CHECK_MSG(femInvalid == 1u, "contact 2 (only the inverted T1) must be the invalid-ref control (%u/1)", femInvalid);

    for (DevBuf* b : { &bSv, &bSt, &bRc, &bAr, &bDesc, &bPt, &bInfo, &bBary, &bCount,
                       &bFi, &bF0, &bF1, &bFc, &bFp, &bFpt })
        dev->destroyBuffer(b->h);
}

// PxsDeformableVolumeMaterialData (PxsDeformableVolumeMaterialCore.h, 5.12), 32 B: the 5.6.1
// dampingScale is gone and materialModel is followed by a PxU16 pad.
struct DeformableMaterialHost
{
    float youngs, poissons, dynamicFriction, elasticityDamping;
    uint16_t materialModel, pad;
    float deformThreshold, deformLowLimitRatio, deformHighLimitRatio;
};
static_assert(sizeof(DeformableMaterialHost) == sizeof(PxsDeformableVolumeMaterialData), "PxsDeformableVolumeMaterialData mirror");
static_assert(offsetof(DeformableMaterialHost, dynamicFriction) == offsetof(PxsDeformableVolumeMaterialData, dynamicFriction), "dynamicFriction");
static_assert(offsetof(DeformableMaterialHost, materialModel) == offsetof(PxsDeformableVolumeMaterialData, materialModel), "materialModel");
static_assert(offsetof(DeformableMaterialHost, deformThreshold) == offsetof(PxsDeformableVolumeMaterialData, deformThreshold), "deformThreshold");

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
    const uint32_t GEOM_SPHERE_H = uint32_t(physx::PxGeometryType::eSPHERE),
                   GEOM_CAPSULE_H = uint32_t(physx::PxGeometryType::eCAPSULE);
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
    std::vector<PxgDeformablePairInfo> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(PxgDeformablePairInfo));
    std::vector<uint16_t> outMat(MAXC + 8u, 0xEEEEu);   // PxU16 rigid material per contact (5.12)
    DevBuf bOutMat = makeDev(dev, outMat);
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
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.w.outPoint = bPt.addr; push.w.outNormalPen = bNp.addr; push.w.outBarycentric = bBary.addr;
    push.w.outContactInfo = bInfo.addr; push.w.totalContactCount = bTotal.addr;
    push.w.outRigidMaterialIndex = bOutMat.addr;
    push.w.contactByRigid = bByX.addr; push.w.tempContactByRigid = bTmp.addr;
    push.w.contactIndexSortedByRigid = bIdx.addr;
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
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);
    const auto gMat = downloadBuffer<uint16_t>(dev, bOutMat.h, MAXC);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        std::printf("  contact %u: point=(%.5f %.5f %.5f rest=%.3f) normal=(%.5f %.5f %.5f) "
                    "pen=%.6f mat=%u\n", i, gPt[i].x, gPt[i].y, gPt[i].z, gPt[i].w,
                    gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w, uint32_t(gMat[i]));

    PXG_CHECK_MSG(n == 2u,
                  "the touching sphere and the capsule must each emit ONE contact and the far "
                  "sphere none, got %u", n);

    bool sawSphere = false, sawCapsule = false;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t mat = uint32_t(gMat[i]);
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
    const uint32_t GEOM_BOX_H = uint32_t(physx::PxGeometryType::eBOX),
                   GEOM_CONVEXMESH_H = uint32_t(physx::PxGeometryType::eCONVEXMESH);
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
    std::vector<PxgDeformablePairInfo> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(PxgDeformablePairInfo));
    std::vector<uint16_t> outMat(MAXC + 8u, 0xEEEEu);   // PxU16 rigid material per contact (5.12)
    DevBuf bOutMat = makeDev(dev, outMat);
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
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.w.outPoint = bPt.addr; push.w.outNormalPen = bNp.addr; push.w.outBarycentric = bBary.addr;
    push.w.outContactInfo = bInfo.addr; push.w.totalContactCount = bTotal.addr;
    push.w.outRigidMaterialIndex = bOutMat.addr;
    push.w.contactByRigid = bByX.addr; push.w.tempContactByRigid = bTmp.addr;
    push.w.contactIndexSortedByRigid = bIdx.addr;
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
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);
    const auto gMat = downloadBuffer<uint16_t>(dev, bOutMat.h, MAXC);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        std::printf("  contact %u: point=(%.6f %.6f %.6f rest=%.3f) normal=(%.5f %.5f %.5f) "
                    "pen=%.6f mat=%u\n", i, gPt[i].x, gPt[i].y, gPt[i].z, gPt[i].w,
                    gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w, uint32_t(gMat[i]));

    PXG_CHECK_MSG(n == 3u,
                  "the near BOX, the CONVEXMESH and the overlapping BOX must each emit one "
                  "contact and the far box none, got %u", n);

    bool sawBox = false, sawConvex = false, sawOverlap = false;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t mat = uint32_t(gMat[i]);
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
//
// P9.89: the SAT's and tetBoundingBox's PxMin/PxMax selects are fixtured in testNpTetTriMid.cpp
// (softBodySimClothMidphase_p89NaNSelect through this kernel, softBodySimClothMidphase_p89SatProbe
// through the test-only sb_clothMidphaseSatProbe), next to the p89tt model they share with
// npTetTriMid.
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
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(bTriIdx.addr));
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
    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
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
//   * the filter (5.12): the softbody's mVolumeSurfaceFilterPairs searched for (tet, ANY
//     triangle), (tet, triangle) and (ANY tet, triangle);
//   * the write format (5.12): point = the SOFTBODY witness (the FEM remap turns it into a
//     simulation tetrahedron later), normal stored NEGATED, barycentric0 = the triangle's and
//     NO barycentric1, ref0 = the cloth TRIANGLE / ref1 = the softbody TETRAHEDRON.
//
// THE NUMBERS ARE PREDICTED FROM THE GEOMETRY.  A tetrahedron sitting a clean distance g
// above a horizontal triangle that its bottom face projects inside has exactly one nearest
// pair: some point of the bottom face against the point below it.  The separation is g and
// the direction is vertical, so pen = +g and the stored normal is (0, -1, 0) negated to
// (0, +1, 0)... which is to say the sign convention itself is what gets asserted, not a
// fitted constant.
//
// KILLED BY THIS FIXTURE: the normal's negation; storing the cloth witness instead of the
// softbody one; the ref0/ref1 order; the surface-face rejection; each of the three filter
// searches (a filter on a DIFFERENT triangle of the same tet must not kill it); and reading
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
// 5.12: the filterPairs / nbFilterPairs arguments are gone (the filter is the softbody's
// mVolumeSurfaceFilterPairs) and the writer is PxgDbDbContactWriter (six pointers + PxU32,
// 56 B) -> 144-byte blob.
struct SbClothCgPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, softbodies, clothes;
    uint64_t stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad76;
    uint64_t stackSizeNeededOnDevice;
    uint64_t outPoint, outNormalPen, outBarycentric0, outBarycentric1, outContactInfo,
             totalContactCount;
    uint32_t maxContacts, pad140;
};
static_assert(sizeof(SbClothCgPush) == 144, "sb_clothContactGen push blob must be 144 bytes");
static_assert(offsetof(SbClothCgPush, outPoint) == 88, "the writer is inlined at @88");
static_assert(sizeof(PxgDbDbContactWriter) == 56, "PxgDbDbContactWriter is 56 bytes");

static const uint32_t ANY = PxgDeformableRef::ANY_FEATURE;
} // namespace p88cg

PXG_TEST(softBodySimClothContactGen)
{
    using namespace p80refit;
    using namespace p88cg;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the cloth: six triangles in y = 0, wound so the normal is +y ---------------------
    const uint32_t NTRI = 6u;
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
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(bTriIdx.addr));
    cloth.mNbVerts = uint32_t(triVerts.size());
    cloth.mNbTriangles = NTRI;
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    // ---- the softbody: six tets, one per triangle -----------------------------------------
    // Bottom face horizontal at y = gap, wound so isValidTet holds, footprint well inside the
    // triangle below.  The face masks differ on purpose:
    //   tet 0: mask 1 -- face T0 (corners 0,1,2) is the BOTTOM face and is on the surface;
    //   tet 1: mask 1, larger gap, OUTSIDE the contact distance;
    //   tet 2: mask 0b1110 -- every face EXCEPT the bottom one.  The contact direction is
    //          vertical, so no surface face opposes it and the pair must be rejected;
    //   tet 3: mask 1, but filtered by (tet 3, its triangle);
    //   tet 4: mask 1, but filtered at the TET level -- (tet 4, ANY triangle of the cloth);
    //   tet 5: mask 1, but filtered at the TRIANGLE level -- (ANY tet, tet 5's triangle).
    const uint32_t NT = 6u;
    const float GAP0 = 0.20f, GAP1 = 0.35f;
    const float gaps[NT] = { GAP0, GAP1, 0.20f, 0.20f, 0.20f, 0.20f };
    const uint8_t masks[NT] = { 0x1u, 0x1u, 0xEu, 0x1u, 0x1u, 0x1u };
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    for (uint32_t t = 0; t < NT; ++t)
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
    for (uint32_t t = 0; t < NT; ++t)
        reinterpret_cast<uint8_t*>(hintWords.data())[t] = masks[t];
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets), bHints = makeDev(dev, hintWords);

    // The element filters (PxgDeformablePairInfo, sorted by (ref0, ref1)).  Tet 0 carries a
    // filter against a DIFFERENT triangle (tet 1's), which must not suppress its own pair.
    auto triOf = [&](uint32_t t) { return NTRI - 1u - t; };
    std::vector<PxgDeformablePairInfo> filters(4);
    filters[0] = { PxgEncodeDeformableTetrahedron(0u, 0u), PxgEncodeDeformableTriangle(0u, triOf(1u)) };
    filters[1] = { PxgEncodeDeformableTetrahedron(0u, 3u), PxgEncodeDeformableTriangle(0u, triOf(3u)) };
    filters[2] = { PxgEncodeDeformableTetrahedron(0u, 4u), PxgEncodeDeformableTriangle(0u, ANY) };
    filters[3] = { PxgEncodeDeformableTetrahedron(0u, ANY), PxgEncodeDeformableTriangle(0u, triOf(5u)) };
    std::sort(filters.begin(), filters.end(), [](const PxgDeformablePairInfo& a, const PxgDeformablePairInfo& b) {
        return PxgComparePairInfo(a, b) < 0;
    });
    DevBuf bFilters = makeDev(dev, filters);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mVolumeSurfaceFilterPairs = reinterpret_cast<decltype(sb.mVolumeSurfaceFilterPairs)>(size_t(bFilters.addr));
    sb.mNumVolumeSurfaceFilterPairs = uint32_t(filters.size());
    sb.mNumTets = NT;
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    // Cloth first again, so the pair is found by TYPE.
    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
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


    // The midphase output: one pair per tet, each against the triangle of its own slot.
    std::vector<uint32_t> stack;
    for (uint32_t t = 0; t < NT; ++t)
    { stack.push_back(0u); stack.push_back(t); stack.push_back(NTRI - 1u - t); stack.push_back(0u); }
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { NT };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 16u;
    std::vector<F4> outPoint(MAXC, F4{ -9, -9, -9, -9 }), outNp(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<F4> outB0(MAXC, F4{ -9, -9, -9, -9 }), outB1(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<PxgDeformablePairInfo> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(PxgDeformablePairInfo));
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
    push.stackSizeBytes = NT * 16u;
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
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
        std::printf("  contact %u: point=(%.5f %.5f %.5f) normal=(%.5f %.5f %.5f) pen=%.6f\n"
                    "              bary0=(%.4f %.4f %.4f %.4f) bary1=(%.4f %.4f %.4f %.4f) "
                    "ref0=0x%llx ref1=0x%llx\n",
                    i, gPt[i].x, gPt[i].y, gPt[i].z, gNp[i].x, gNp[i].y, gNp[i].z, gNp[i].w,
                    gB0[i].x, gB0[i].y, gB0[i].z, gB0[i].w, gB1[i].x, gB1[i].y, gB1[i].z, gB1[i].w,
                    (unsigned long long)gInfo[i].ref0, (unsigned long long)gInfo[i].ref1);

    PXG_CHECK_MSG(n == 1u,
                  "only tet 0 may produce a contact: tet 1 is 0.35 away against a contact "
                  "distance of 0.30, tet 2's bottom face is not a SURFACE face, and tets 3, 4 "
                  "and 5 are filtered by (tet, triangle), (tet, ANY) and (ANY, triangle).  Tet 0 "
                  "is filtered against a DIFFERENT triangle, which must NOT suppress it.  Got %u", n);
    if (n == 0u || n > MAXC) { /* the checks below would read poison */ }
    else
    {
        // The pair identity, and its ORDER: ref0 is the cloth TRIANGLE, ref1 the softbody
        // TETRAHEDRON.  Tet 0 pairs with triangle NTRI-1, so the two feature ids differ.
        PXG_CHECK_MSG(gInfo[0].ref0 == PxgEncodeDeformableTriangle(0u, NTRI - 1u) &&
                      gInfo[0].ref1 == PxgEncodeDeformableTetrahedron(0u, 0u),
                      "ref0 must be the CLOTH triangle %u and ref1 the SOFTBODY tet 0; got 0x%llx / 0x%llx",
                      NTRI - 1u, (unsigned long long)gInfo[0].ref0, (unsigned long long)gInfo[0].ref1);
        // 5.12: the witness on the SOFTBODY is stored, on the tet's bottom face, y = GAP0.
        PXG_CHECK_MSG(std::fabs(gPt[0].y - GAP0) < 1e-4f,
                      "the stored point is the witness on the SOFTBODY, so it lies in y = %.2f "
                      "(got y = %.6f).  Storing the cloth witness instead gives y = 0",
                      GAP0, gPt[0].y);
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
        // wA is the stored point and wB lies in y = 0, so pen = dot(wB - wA, -storedNormal).
        PXG_CHECK_MSG(std::fabs(gNp[0].w - (gPt[0].y * gNp[0].y)) < 2e-3f,
                      "pen (%.6f) must equal dot(wB - wA, -storedNormal)", gNp[0].w);
        // The triangle barycentric is a convex combination with w == 0; the tet one is no
        // longer written (5.12) -- its slot keeps the poison.
        const float s0 = gB0[0].x + gB0[0].y + gB0[0].z;
        PXG_CHECK_MSG(std::fabs(s0 - 1.0f) < 1e-3f && gB0[0].w == 0.0f,
                      "the TRIANGLE barycentric sums to 1 with a zero fourth component "
                      "(got %.5f, w = %.5f)", s0, gB0[0].w);
        PXG_CHECK_MSG(gB1[0].x == -9.0f && gB1[0].w == -9.0f,
                      "5.12 writes no tet barycentric (the FEM remap computes it); got (%.5f ... %.5f)",
                      gB1[0].x, gB1[0].w);
    }
    PXG_CHECK_MSG(gPt[1].x == -9.0f,
                  "nothing may be written past the contact count");

    for (PxgBufferHandle h : { bTriVerts.h, bTriIdx.h, bCloth.h, bTetVerts.h, bTets.h, bHints.h,
                               bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h, bFilters.h, bStack.h,
                               bPairNum.h, bNeeded.h, bPt.h, bNp.h, bB0.h, bB1.h, bInfo.h, bTotal.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P9.86c (softBodyClothContactGenBda-r1) -- the -0 lane of the tet barycentric.
// computeTetBarycentric (deformableCollision.cuh:96-138) clamps every lane with
// PxClamp(t, 0, 1) = PxMin(1, PxMax(0, t)) (PxMath.h:142-146) = selectMin(1, selectMax(0, t))
// (PxUnixMathIntrinsics.h:116-125).  For t = -0 the selects keep -0 (0 > -0 is false); a
// GLSL FClamp returns +0 on llvmpipe (F4).  The step-2 tet above gives x = +0, and its
// tolerance checks are sign-blind, so that revert survived.
// Here the step-2 tet is mirrored in x and z so its bottom face is the y = 0 plane:
// a = (0,0,0), b = (0,0,-1), c = (-1,0,0), d = (0,1,0), detBcd = 1.  The tet witness is a
// convex combination of bottom corners (pxgSlangGjk.slang:532), so cpA.y = +0 exactly and
// pa = (px, +0, pz) with px, pz < 0.  Then ca x pa = (-0, pz, +0) and
// x = ba . (ca x pa) = (+0)(-0) + (+0)pz + (-1)(+0) = -0, with or without FMA.  5.6.1 stored
// barycentric1.w = 0x80000000 (the reverted FClamp stored +0).
// PhysX 5.12 no longer STORES the tet barycentric here: it only gates the contact (a -0 and
// a +0 pass the window alike), and the point stored is the SOFTBODY witness, which
// sb_fem_contact_remap_to_simLaunch turns into the tet barycentric -- through the same
// computeTetBarycentric, whose -0 lane softBodySimContactRemapZeroSign pins bitwise.  What
// stays observable here is the gate (the contact exists), that barycentric1 is untouched,
// and the stored witness: cpA.y = u*(+0) + v*(+0) + w*(+0) = +0 exactly.
// The stored point is closestPointB = cpA + upperBound * o.closestPointDir
// (pxgSlangGjk.slang:548).  o.closestPointDir = -normalize(cp) is not the stored normal
// (-o.direction): here the final simplex is a face of the Minkowski difference whose
// support points all have y = 0 - (-GAP) exactly, so closestPointTriW's face normal has
// x = z = +-0 and cp.x, cp.z are +-0 exactly -- gPt.x/z are cpA.x/z bit for bit.  The
// vertical-normal CHECK is a proxy for that; if it were ever wrong, the lane 0-2 compares
// fail at HEAD, never under a mutant only.
// ===========================================================================
namespace p86cgz
{
struct V3z { float x, y, z; };
static V3z zSub(V3z a, V3z b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float zDot(V3z a, V3z b) { return a.x * b.x + a.y * b.y + a.z * b.z; }   // PxVec3::dot
static V3z zCross(V3z a, V3z b)                                                    // PxVec3::cross
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
// false = the CUDA: PxClamp = selectMin(1, selectMax(0, t)).
// true  = the reverted port's GLSL FClamp on llvmpipe (fmax(+-0, 0) = +0, F4).
static bool gTetBaryFClampTwin = false;
static float zClamp01(float t)
{
    if (gTetBaryFClampTwin) { const float m = (t > 0.0f) ? t : 0.0f; return (m < 1.0f) ? m : 1.0f; }
    const float m = (0.0f > t) ? 0.0f : t;
    return (1.0f < m) ? 1.0f : m;
}
// computeTetBarycentric, deformableCollision.cuh:96-138.
static bool zTetBary(V3z a, V3z b, V3z c, V3z d, V3z p, F4& bary)
{
    const float minEps = -1e-5f, maxEps = 1.0f + 1e-5f;
    const V3z ba = zSub(b, a), ca = zSub(c, a), da = zSub(d, a), pa = zSub(p, a);
    const float detBcd = zDot(ba, zCross(ca, da));
    const float v = zDot(pa, zCross(ca, da)) / detBcd;
    const float w = zDot(ba, zCross(pa, da)) / detBcd;
    const float x = zDot(ba, zCross(ca, pa)) / detBcd;
    const float u = 1.0f - v - w - x;
    bary = F4{ zClamp01(u), zClamp01(v), zClamp01(w), zClamp01(x) };
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}
} // namespace p86cgz

PXG_TEST(softBodySimClothContactGenBaryNegZero)
{
    using namespace p80refit;
    using namespace p88cg;
    using namespace p86cgz;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- one cloth triangle 0.2 below the tet, strictly inside the tet's bottom face ------
    const float GAP = 0.20f;
    std::vector<F4> triVerts = { F4{ -0.25f, -GAP, -0.25f, 0.0f }, F4{ -0.25f, -GAP, -0.5f, 0.0f },
                                 F4{ -0.5f, -GAP, -0.25f, 0.0f } };
    std::vector<TetIdx4> triIdxArr = { { 0u, 1u, 2u, 0u } };
    DevBuf bTriVerts = makeDev(dev, triVerts), bTriIdx = makeDev(dev, triIdxArr);
    if (!bTriVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTriVerts.addr));
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(bTriIdx.addr));
    cloth.mNbVerts = 3u;
    cloth.mNbTriangles = 1u;
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    // ---- one resting, axis-aligned tet; corner 0 at the origin, bottom face in y = 0 -------
    const V3z A[4] = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } };
    std::vector<F4> tetVerts;
    for (int i = 0; i < 4; ++i) tetVerts.push_back(F4{ A[i].x, A[i].y, A[i].z, 1.0f });
    std::vector<TetIdx4> tets = { { 0u, 1u, 2u, 3u } };
    std::vector<uint32_t> hintWords = { 0x1u };   // face T0 (corners 0,1,2) is a surface face
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets), bHints = makeDev(dev, hintWords);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = 1u;
    sb.mNumVerts = 4u;
    std::vector<PxgSoftBody> descArr = { sb };
    DevBuf bDesc = makeDev(dev, descArr);

    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;    shapes[0].particleOrSoftbodyId = 0u; shapes[0].srw = 1.0f;
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = 0u; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*cloth cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);
    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f; xf[8 + 3] = 1.0f;
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> cd = { 0.16f, 0.14f, 0.0f, 0.0f };   // 0.30 > GAP
    DevBuf bCd = makeDev(dev, cd);

    std::vector<uint32_t> stack = { 0u, 0u, 0u, 0u };   // (cm 0, tet 0, tri 0)
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 1u };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 4u;
    std::vector<F4> outPoint(MAXC, F4{ -9, -9, -9, -9 }), outNp(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<F4> outB0(MAXC, F4{ -9, -9, -9, -9 }), outB1(MAXC, F4{ -9, -9, -9, -9 });
    std::vector<PxgDeformablePairInfo> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(PxgDeformablePairInfo));
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
    push.stackSizeBytes = 16u;     // no filter pairs: the softbody's list is null
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.outPoint = bPt.addr; push.outNormalPen = bNp.addr;
    push.outBarycentric0 = bB0.addr; push.outBarycentric1 = bB1.addr;
    push.outContactInfo = bInfo.addr; push.totalContactCount = bTotal.addr;
    push.maxContacts = MAXC;

    Kernel k("softBodyClothContactGenBda.spv", "sb_clothContactGenLaunch", {}, sizeof(SbClothCgPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bTotal.h, 1)[0];
    const auto gPt = downloadBuffer<F4>(dev, bPt.h, MAXC);
    const auto gNp = downloadBuffer<F4>(dev, bNp.h, MAXC);
    const auto gB1 = downloadBuffer<F4>(dev, bB1.h, MAXC);
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);

    PXG_CHECK_MSG(n == 1u, "the resting tet 0.2 above a triangle inside its bottom face must "
                           "produce its one contact, got %u", n);
    if (n == 1u)
    {
        std::printf("  point=(%a %a %a) normal=(%a %a %a) bary1=(0x%08x 0x%08x 0x%08x 0x%08x)\n",
                    gPt[0].x, gPt[0].y, gPt[0].z, gNp[0].x, gNp[0].y, gNp[0].z,
                    f2u(gB1[0].x), f2u(gB1[0].y), f2u(gB1[0].z), f2u(gB1[0].w));
        PXG_CHECK_MSG(gNp[0].x == 0.0f && gNp[0].z == 0.0f && gNp[0].y == 1.0f,
                      "the contact must be exactly vertical; got (%a %a %a)",
                      gNp[0].x, gNp[0].y, gNp[0].z);
        PXG_CHECK_MSG(gPt[0].x <= -0.25f && gPt[0].z <= -0.25f && gPt[0].x + gPt[0].z >= -0.75f,
                      "the witness must lie over the triangle, strictly inside the tet face "
                      "(px < 0, pz < 0); got x=%a z=%a", gPt[0].x, gPt[0].z);

        // cpA.y = u*(+0) + v*(+0) + w*(+0) = +0 (pxgSlangGjk.slang:532), stored as the point.
        PXG_CHECK_MSG(f2u(gPt[0].y) == 0u, "the stored softbody witness has y = +0 exactly; got 0x%08x", f2u(gPt[0].y));
        // The directed tet still gives computeTetBarycentric a -0 lane at this witness; it only
        // gates the contact now (5.12), and a -0 passes the window like a +0.
        const V3z wit = { gPt[0].x, gPt[0].y, gPt[0].z };
        F4 want;
        gTetBaryFClampTwin = false;
        const bool inWin = zTetBary(A[0], A[1], A[2], A[3], wit, want);
        PXG_CHECK_MSG(inWin, "the witness must pass the eps window");
        PXG_CHECK_MSG(f2u(want.w) == 0x80000000u, "coverage: the CUDA mirror's x lane is -0 (0x%08x)", f2u(want.w));
        for (uint32_t i = 0; i < MAXC; ++i)
            PXG_CHECK_MSG(gB1[i].x == -9.0f && gB1[i].y == -9.0f && gB1[i].z == -9.0f && gB1[i].w == -9.0f,
                          "5.12 writes no tet barycentric (slot %u)", i);
        PXG_CHECK_MSG(gInfo[0].ref0 == PxgEncodeDeformableTriangle(0u, 0u) &&
                      gInfo[0].ref1 == PxgEncodeDeformableTetrahedron(0u, 0u),
                      "ref0 = the triangle, ref1 = the tetrahedron (0x%llx 0x%llx)",
                      (unsigned long long)gInfo[0].ref0, (unsigned long long)gInfo[0].ref1);
    }

    for (PxgBufferHandle h : { bTriVerts.h, bTriIdx.h, bCloth.h, bTetVerts.h, bTets.h, bHints.h,
                               bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h, bStack.h,
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
    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
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
//
// PhysX 5.12: the contact goes through the vertex overload of
// PxgDbDbContactWriter::writeSurfaceVolumeContact -- ref0 = the CLOTH vertex, ref1 = the
// softbody TETRAHEDRON, and NO barycentric (the tet's is computed by the FEM remap from the
// stored point).  The filter is the softbody's mVolumeSurfaceFilterPairs, searched for
// (ANY tet, vertex), (tet, ANY vertex) and (tet, vertex).
// ===========================================================================
namespace p88vc
{
struct SbClothVertCgPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, softbodies, clothes;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum;
    uint64_t stackSizeNeededOnDevice;
    uint64_t outPoint, outNormalPen, outBarycentric0, outBarycentric1, outContactInfo,
             totalContactCount;
    uint32_t maxContacts, pad140;
};
static_assert(sizeof(SbClothVertCgPush) == 144, "sb_clothVertContactGen push blob must be 144 bytes");
static_assert(offsetof(SbClothVertCgPush, outPoint) == 88, "the writer is inlined at @88");
static_assert(offsetof(SbClothVertCgPush, stackPtr) == 64,
              "stackSizeBytes sits BEFORE the stack pointer here, unlike step 2's pack");
} // namespace p88vc

PXG_TEST(softBodySimClothVertContactGen)
{
    using namespace p80refit;
    using namespace p88cg;    // ANY
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
    // P9.86c: tet 6 (TET_ZS) is tet 0's shape at ox = 24 with its corners REORDERED so that
    // a = (25,0,0), b = (24,0,0), c = (24,0,1), d = (24,1,0): detBcd is still +1, but
    // cross(c-a, d-a) = (-1,-1,-1).  Only the directed sub-run at the end pairs it; the main
    // stack never names it, so every count above and below is unchanged.
    const uint32_t TET_ZS = NTET, VTX_ZS = 7u;
    {
        const uint32_t b = uint32_t(tetVerts.size());   // 24
        tetVerts.push_back(F4{ 24.0f, 0.0f, 0.0f, 1.0f });
        tetVerts.push_back(F4{ 24.0f, 0.0f, 1.0f, 1.0f });
        tetVerts.push_back(F4{ 25.0f, 0.0f, 0.0f, 1.0f });
        tetVerts.push_back(F4{ 24.0f, 1.0f, 0.0f, 1.0f });
        tets.push_back({ b + 2u, b, b + 1u, b + 3u });
    }
    std::vector<uint32_t> hintWords(2u, 0u);
    for (uint32_t t = 0; t < NTET; ++t)
        reinterpret_cast<uint8_t*>(hintWords.data())[t] = masks[t];
    reinterpret_cast<uint8_t*>(hintWords.data())[TET_ZS] = 0x1u;   // face (a,b,c), y = 0
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets), bHints = makeDev(dev, hintWords);
    if (!bTetVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = NTET + 1u;
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
    // tet t.  With the two indices equal both feature ids are the same number and a mutant
    // that swapped ref0's and ref1's features would survive.
    const float CD_SB = 0.0625f, CD_CLOTH = 0.1875f;
    std::vector<F4> clothVerts = {
        { 500.0f, 500.0f, 500.0f, 1.0f },
        {  0.20f, -0.30f, 0.20f, 1.0f },
        {  4.20f, -0.10f, 0.20f, 1.0f },
        {  8.20f,  0.40f, 0.20f, 1.0f },
        { 12.20f,  0.10f, 0.20f, 1.0f },
        { 16.20f,  0.10f, 0.20f, 1.0f },
        { 21.05f, -0.20f, 0.00f, 1.0f },
        { 25.0625f, -0.125f, 0.0f, 1.0f },   // v7 -> tet 6, the directed sub-run only
    };
    DevBuf bClothVerts = makeDev(dev, clothVerts);

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bClothVerts.addr));
    cloth.mNbVerts = uint32_t(clothVerts.size());
    std::vector<PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
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

    // No filter entries in the main run (a null mVolumeSurfaceFilterPairs); a sub-run below
    // turns each of the three OR terms on.

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
    std::vector<PxgDeformablePairInfo> outInfo(MAXC);
    std::memset(outInfo.data(), 0xEE, outInfo.size() * sizeof(PxgDeformablePairInfo));
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
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);
    const uint32_t gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];

    // Index the results by the cloth vertex they name.
    int slot[7] = { -1, -1, -1, -1, -1, -1, -1 };
    for (uint32_t i = 0; i < n && i < MAXC; ++i)
    {
        const uint32_t vtx = PxgGetDeformableFeatureId(gInfo[i].ref0);
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
    // 5.12: the vertex overload writes NO barycentric -- both slots keep their poison.
    for (uint32_t i = 0; i < MAXC; ++i)
        PXG_CHECK_MSG(gB0[i].x == -9.0f && gB0[i].w == -9.0f && gB1[i].x == -9.0f && gB1[i].w == -9.0f,
                      "a vertex-vs-tet contact writes no barycentric (slot %u)", i);
    // ref0 = the cloth VERTEX, ref1 = the softbody TETRAHEDRON; the two feature ids differ.
    for (uint32_t v = 2; v <= 6u; ++v)
    {
        if (slot[v] < 0) continue;
        PXG_CHECK_MSG(gInfo[slot[v]].ref0 == PxgEncodeDeformableVertex(0u, v) &&
                      gInfo[slot[v]].ref1 == PxgEncodeDeformableTetrahedron(0u, v - 1u),
                      "ref0 is the CLOTH vertex (%u) and ref1 the TETRAHEDRON (%u); got 0x%llx / 0x%llx",
                      v, v - 1u, (unsigned long long)gInfo[slot[v]].ref0, (unsigned long long)gInfo[slot[v]].ref1);
    }
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
        // the pair under test is (cloth vertex 2, tetrahedron 1); (tet, vertex) order
        std::vector<PxgDeformablePairInfo> f(1);
        if (term == 0) f[0] = { PxgEncodeDeformableTetrahedron(0u, ANY), PxgEncodeDeformableVertex(0u, 2u) };
        if (term == 1) f[0] = { PxgEncodeDeformableTetrahedron(0u, 1u), PxgEncodeDeformableVertex(0u, ANY) };
        if (term == 2) f[0] = { PxgEncodeDeformableTetrahedron(0u, 1u), PxgEncodeDeformableVertex(0u, 2u) };
        DevBuf bF = makeDev(dev, f);
        std::vector<PxgSoftBody> fDesc = descArr;
        fDesc[0].mVolumeSurfaceFilterPairs = reinterpret_cast<decltype(fDesc[0].mVolumeSurfaceFilterPairs)>(size_t(bF.addr));
        fDesc[0].mNumVolumeSurfaceFilterPairs = 1u;
        DevBuf bFDesc = makeDev(dev, fDesc);
        std::vector<uint32_t> t2 = { 0u };
        DevBuf bT2 = makeDev(dev, t2);
        std::vector<PxgDeformablePairInfo> i2(MAXC);
        std::memset(i2.data(), 0xEE, i2.size() * sizeof(PxgDeformablePairInfo));
        DevBuf bI2 = makeDev(dev, i2);
        SbClothVertCgPush p2 = push;
        p2.softbodies = bFDesc.addr;
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
        dev->destroyBuffer(bF.h); dev->destroyBuffer(bFDesc.h); dev->destroyBuffer(bT2.h); dev->destroyBuffer(bI2.h);
    }

    // ---- P9.86c (softBodyClothVertContactGenBda-r1): the -0 barycentric lane ---------------
    // computeTetBarycentric (deformableCollision.cuh:96-138) clamps each lane with PxClamp =
    // selectMin(1, selectMax(0, t)) (PxMath.h:142, PxUnixMathIntrinsics.h:116-125), which keeps
    // a -0; GLSL FClamp gives +0 on llvmpipe (F4).  v7 = a + (0.0625, -0.125, 0) lies in a's
    // vertex region of the y = 0 face, so closestPtPointTriangle2 returns a itself and
    // pa = (+0,+0,+0).  Then v = pa . (ca x da) / detBcd = ((+0)(-1) + (+0)(-1) + (+0)(-1)) / 1
    // = -0, while w = x = +0 and u = 1: every value is exact, so neither operation order nor
    // FMA contraction can move the sign.  5.6.1 stored barycentric1 = (1, -0, +0, +0); PhysX
    // 5.12 stores NO barycentric here (the FEM remap computes it from the stored point through
    // the same computeTetBarycentric, whose -0 lane softBodySimContactRemapZeroSign pins), so
    // what stays observable is the witness bit pattern and that both slots are untouched.
    {
        std::vector<uint32_t> sz = { 0u, VTX_ZS, TET_ZS, 0u };
        std::vector<uint32_t> pnz = { 1u }, ndz = { 0u }, tz = { 0u };
        std::vector<F4> poison(4u, F4{ -9, -9, -9, -9 });
        std::vector<PxgDeformablePairInfo> iz(4u);
        std::memset(iz.data(), 0xEE, iz.size() * sizeof(PxgDeformablePairInfo));
        DevBuf bSz = makeDev(dev, sz), bPnz = makeDev(dev, pnz), bNdz = makeDev(dev, ndz);
        DevBuf bTz = makeDev(dev, tz), bPtz = makeDev(dev, poison), bNpz = makeDev(dev, poison);
        DevBuf bB0z = makeDev(dev, poison), bB1z = makeDev(dev, poison), bIz = makeDev(dev, iz);
        SbClothVertCgPush pz = push;
        pz.stackSizeBytes = 16u;
        pz.stackPtr = bSz.addr;
        pz.midphasePairsNum = bPnz.addr;
        pz.stackSizeNeededOnDevice = bNdz.addr;
        pz.outPoint = bPtz.addr; pz.outNormalPen = bNpz.addr;
        pz.outBarycentric0 = bB0z.addr; pz.outBarycentric1 = bB1z.addr;
        pz.outContactInfo = bIz.addr; pz.totalContactCount = bTz.addr;
        pz.maxContacts = 4u;
        Kernel("softBodyClothVertContactGenBda.spv", "sb_clothVertContactGenLaunch",
               {}, sizeof(SbClothVertCgPush)).dispatch(512u, {}, pz, 1u, 1u);
        dev->flush();
        const uint32_t nz = downloadBuffer<uint32_t>(dev, bTz.h, 1)[0];
        const F4 gp = downloadBuffer<F4>(dev, bPtz.h, 1)[0];
        const F4 gnp = downloadBuffer<F4>(dev, bNpz.h, 1)[0];
        const F4 gb = downloadBuffer<F4>(dev, bB1z.h, 1)[0];
        PXG_CHECK_MSG(nz == 1u, "v7 is 0.14 outside tet 6, inside the 0.25 contact distance: one "
                      "contact, got %u", nz);
        PXG_CHECK_MSG(f2u(gp.x) == f2u(25.0f) && f2u(gp.y) == 0u && f2u(gp.z) == 0u,
                      "precondition: the witness is corner a = (25,+0,+0) bit for bit, else pa is "
                      "not (+0,+0,+0); got %08x %08x %08x", f2u(gp.x), f2u(gp.y), f2u(gp.z));
        const TetIdx4 ti = tets[TET_ZS];
        auto v3 = [&](uint32_t i) { return p86cgz::V3z{ tetVerts[i].x, tetVerts[i].y, tetVerts[i].z }; };
        const p86cgz::V3z wp = { gp.x, gp.y, gp.z };
        F4 ref, twin;
        p86cgz::gTetBaryFClampTwin = false;
        p86cgz::zTetBary(v3(ti.x), v3(ti.y), v3(ti.z), v3(ti.w), wp, ref);
        p86cgz::gTetBaryFClampTwin = true;
        p86cgz::zTetBary(v3(ti.x), v3(ti.y), v3(ti.z), v3(ti.w), wp, twin);
        p86cgz::gTetBaryFClampTwin = false;
        const uint32_t baryNegZeroSeen =
            uint32_t(f2u(ref.x) == 0x80000000u && f2u(twin.x) == 0u) +
            uint32_t(f2u(ref.y) == 0x80000000u && f2u(twin.y) == 0u) +
            uint32_t(f2u(ref.z) == 0x80000000u && f2u(twin.z) == 0u) +
            uint32_t(f2u(ref.w) == 0x80000000u && f2u(twin.w) == 0u);
        PXG_CHECK_MSG(baryNegZeroSeen == 1u,
                      "coverage: the directed tet must give one -0 barycentric lane that FClamp "
                      "turns into +0 (%u)", baryNegZeroSeen);
        PXG_CHECK_MSG(f2u(ref.x) == 0x3f800000u && f2u(ref.y) == 0x80000000u &&
                      f2u(ref.z) == 0u && f2u(ref.w) == 0u,
                      "mirror image (1, -0, +0, +0); got %08x %08x %08x %08x",
                      f2u(ref.x), f2u(ref.y), f2u(ref.z), f2u(ref.w));
        PXG_CHECK_MSG(gb.x == -9.0f && gb.y == -9.0f && gb.z == -9.0f && gb.w == -9.0f,
                      "5.12 writes no tet barycentric; got %08x %08x %08x %08x",
                      f2u(gb.x), f2u(gb.y), f2u(gb.z), f2u(gb.w));
        const float wantPen = std::sqrt(0.01953125f);
        PXG_CHECK_MSG(std::fabs(gnp.w - wantPen) < 1e-5f &&
                      std::fabs(gnp.x + 0.0625f / wantPen) < 1e-4f &&
                      std::fabs(gnp.y - 0.125f / wantPen) < 1e-4f && std::fabs(gnp.z) < 1e-4f,
                      "v7 is OUTSIDE at distance %.6f from corner a, normal toward the vertex; "
                      "got (%.5f %.5f %.5f) pen %.6f", wantPen, gnp.x, gnp.y, gnp.z, gnp.w);
        for (PxgBufferHandle h : { bSz.h, bPnz.h, bNdz.h, bTz.h, bPtz.h, bNpz.h, bB0z.h, bB1z.h, bIz.h })
            dev->destroyBuffer(h);
    }

    for (PxgBufferHandle h : { bTetVerts.h, bTets.h, bHints.h, bDesc.h, bClothVerts.h, bCloth.h,
                               bShapes.h, bCms.h, bXf.h, bCd.h, bStack.h,
                               bPairNum.h, bNeeded.h, bPt.h, bNp.h, bB0.h, bB1.h, bInfo.h, bTotal.h })
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
//   * the 112-byte blob with the contact writer inlined at @56.
//
// PhysX 5.12: PxgDbDbContactWriter::writeVolumeVolumeContact, ref0 = the query VERTEX of
// softbody0 and ref1 = the TETRAHEDRON of softbody1, no barycentric; the element filter is
// softbody0's mVolumeVolumeFilterPairs (PxgDeformablePairInfo), searched for (vertex, ANY) per
// query vertex and for (vertex, tet) and (ANY, tet) per candidate.
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
    const uint32_t ANY = PxgDeformableRef::ANY_FEATURE;
    auto vref = [](uint32_t id, uint32_t v) { return PxgEncodeDeformableVertex(id, v); };
    auto tref = [](uint32_t id, uint32_t t) { return PxgEncodeDeformableTetrahedron(id, t); };

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
    const uint32_t B_FILTANY = addTet(bVerts, bTets, bVertHint, bTetHint, 50.0f, 0.0f, 0.00f, false, 0xfu);  // 5.12
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
    const uint32_t a10 = addQuery(aVerts, aVertHint, 50.12f, 0.15f, 0.05f, 1u); // drop: (ANY, B_FILTANY) filter
    (void)a2; (void)a3; (void)a6; (void)a9; (void)a10;

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

    // vertToSurfaceTetRemap: DIFFERENT bases per body, and neither is the identity.  5.12 no
    // longer reads it (ref0 names the query VERTEX), so a port that still did would be caught.
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

    // ---- filter pairs (PxgDeformablePairInfo), sorted by ref0 then ref1 -------
    // They hang off body A (softbody0 on the flip = 0 pass).  B carries NONE, which also
    // exercises the null filter-list path the self-collision tranche had to harden.
    std::vector<PxgDeformablePairInfo> aFilters(3);
    aFilters[0].ref0 = vref(A_ID, a4);  aFilters[0].ref1 = tref(B_ID, B_FILTPR);    // a4 against ONE tet of B
    aFilters[1].ref0 = vref(A_ID, a5);  aFilters[1].ref1 = tref(B_ID, ANY);         // a5 against the whole of B
    aFilters[2].ref0 = vref(A_ID, ANY); aFilters[2].ref1 = tref(B_ID, B_FILTANY);   // every A vertex vs one tet
    for (size_t f = 1; f < aFilters.size(); ++f)                                     // the binary search needs order
        PXG_CHECK(PxgComparePairInfo(aFilters[f - 1], aFilters[f]) < 0);

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
    sbA.mVolumeVolumeFilterPairs = reinterpret_cast<decltype(sbA.mVolumeVolumeFilterPairs)>(size_t(dbAFilters.addr));
    sbA.mNumVolumeVolumeFilterPairs = uint32_t(aFilters.size());
    sbA.mNumTets = uint32_t(aTets.size());
    sbA.mNumVerts = uint32_t(aVerts.size());

    PxgSoftBody sbB = makeZeroDesc();
    sbB.mTetMeshData = reinterpret_cast<void*>(size_t(dbBTree.addr));
    sbB.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(dbBTetHint.addr));
    sbB.mTetIndices = reinterpret_cast<decltype(sbB.mTetIndices)>(size_t(dbBTets.addr));
    sbB.mPosition_InvMass = reinterpret_cast<float4*>(size_t(dbBVerts.addr));
    sbB.mSurfaceVertsHint = reinterpret_cast<PxU8*>(size_t(dbBVertHint.addr));
    sbB.mSurfaceVertToTetRemap = reinterpret_cast<PxU32*>(size_t(dbBRemap.addr));
    sbB.mVolumeVolumeFilterPairs = nullptr;      // the null path
    sbB.mNumVolumeVolumeFilterPairs = 0u;
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
    // 5.12 writes no barycentric for a volume-volume contact: both slots keep their poison.
    DevBuf dbBary0 = makeDev(dev, poison);
    DevBuf dbBary1 = makeDev(dev, poison);
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
    const auto gBary0 = downloadBuffer<uint32_t>(dev, dbBary0.h, MAXC * 4u);
    const auto gBary1 = downloadBuffer<uint32_t>(dev, dbBary1.h, MAXC * 4u);
    const auto gInfo = downloadBuffer<PxgDeformablePairInfo>(dev, dbInfo.h, MAXC);

    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
        std::printf("  contact %u: p=(%.5f %.5f %.5f) n=(%.5f %.5f %.5f) pen=%.6f "
                    "ref0=(body %u type %u elem %u) ref1=(body %u type %u elem %u)\n",
                    i, gPoint[i].x, gPoint[i].y, gPoint[i].z, gNormalPen[i].x, gNormalPen[i].y,
                    gNormalPen[i].z, gNormalPen[i].w,
                    PxgGetDeformableId(gInfo[i].ref0), uint32_t(PxgGetDeformableFeatureType(gInfo[i].ref0)),
                    PxgGetDeformableFeatureId(gInfo[i].ref0), PxgGetDeformableId(gInfo[i].ref1),
                    uint32_t(PxgGetDeformableFeatureType(gInfo[i].ref1)), PxgGetDeformableFeatureId(gInfo[i].ref1));
    for (uint32_t w = 0; w < MAXC * 4u; ++w)
        PXG_CHECK_MSG(gBary0[w] == POISON && gBary1[w] == POISON,
                      "a volume-volume contact writes no barycentric (word %u: 0x%08x 0x%08x)", w, gBary0[w], gBary1[w]);

    PXG_CHECK_MSG(gCount == 4u,
                  "exactly four queries survive every gate -- a1, a7 and a8 on the flip=0 pass "
                  "and b1 on the flip=1 pass -- got %u.  3 usually means the z dimension was "
                  "dropped (b1 is the only contact that pass can make); more than 4 means a "
                  "gate went missing or a vertex was visited more than once", gCount);

    auto near6 = [](float a, float b) { return std::fabs(a - b) <= 1e-6f; };
    bool sawA1 = false, sawA7 = false, sawA8 = false, sawB1 = false;
    for (uint32_t i = 0; i < gCount; ++i)
    {
        const uint64_t ind0 = gInfo[i].ref0;
        const uint64_t ind1 = gInfo[i].ref1;

        if (ind0 == vref(A_ID, a1))
        {
            sawA1 = true;
            PXG_CHECK_MSG(ind1 == tref(B_ID, B_DEEP),
                          "a1 penetrates B_DEEP by 0.05 and B_SHALLOW by 0.03; the cross-lane "
                          "argmin must pick the DEEPER one (got body %u tet %u, want body %u "
                          "tet %u)", PxgGetDeformableId(ind1), PxgGetDeformableFeatureId(ind1), B_ID, B_DEEP);
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
            PXG_CHECK(near6(gNormalPen[i].x, 0.0f) && near6(gNormalPen[i].y, 0.0f) &&
                      near6(gNormalPen[i].z, -1.0f));
            PXG_CHECK(near6(gPoint[i].x, 0.12f) && near6(gPoint[i].y, 0.15f) &&
                      near6(gPoint[i].z, 0.0f));
            PXG_CHECK_MSG(gPoint[i].w == 0.0f, "point.w is 0 for a volume-volume contact (got %g)", gPoint[i].w);
        }
        else if (ind0 == vref(A_ID, a7))
        {
            sawA7 = true;
            PXG_CHECK(ind1 == tref(B_ID, B_GOOD));
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
        }
        else if (ind0 == vref(A_ID, a8))
        {
            sawA8 = true;
            PXG_CHECK(ind1 == tref(B_ID, B_GOOD));
            // The OUTSIDE branch: pen is the POSITIVE separation and the normal is the
            // normalised point-to-surface direction, here parallel to the x-face normal.
            PXG_CHECK_MSG(std::fabs(gNormalPen[i].w - 0.04f) <= 2e-6f,
                          "a8 sits 0.040 outside B_GOOD's x-face; got pen %.7f", gNormalPen[i].w);
            PXG_CHECK(near6(gNormalPen[i].y, 0.0f) && near6(gNormalPen[i].z, 0.0f));
            PXG_CHECK_MSG(gNormalPen[i].x < -0.99f,
                          "the outside normal must point from the tet towards the query vertex "
                          "(-x here); got %.5f", gNormalPen[i].x);
        }
        else if (ind0 == vref(B_ID, b1))
        {
            // THE FLIP.  Note the pair ids are the other way round: body B supplies the vertex
            // and body A the tetrahedron, which can only happen on blockIdx.z == 1.
            sawB1 = true;
            PXG_CHECK_MSG(ind1 == tref(A_ID, A_GOOD),
                          "on the flipped pass the TET must come from body A (got body %u)",
                          PxgGetDeformableId(ind1));
            PXG_CHECK(near6(gNormalPen[i].w, -0.05f));
            PXG_CHECK(near6(gPoint[i].x, 100.12f) && near6(gPoint[i].z, 0.0f));
        }
        else
            PXG_CHECK_MSG(false, "contact %u came from an unexpected query: ref0 = body %u "
                                 "elem %u", i, PxgGetDeformableId(ind0), PxgGetDeformableFeatureId(ind0));
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
    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
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
//   * the RIGID-shaped output (5.12, PxgRigidDbContactWriter::writeVolumeContact): ref0 is
//     the PxNodeIndex from shapeToRigidRemapTable[TRIMESH cache ref], ref1 the tetrahedron
//     ref, the PxU16 material comes from the TRIMESH shape, and the three sort side-arrays
//     (contactByRigid, tempContactByRigid, contactIndexSortedByRigid) are written;
//   * the element filter (5.12: the softbody's mRigidVolumeFilterPairs, keyed on
//     PxgEncodeRigidFilterRef(rigid, trimesh cache ref));
//   * the stack clamp and the unclamped stackSizeNeededOnDevice tail.
// ===========================================================================
namespace p810cg
{
// 5.12: the filterPairs / nbFilterPairs arguments are gone (the filter is the softbody's
// mRigidVolumeFilterPairs) and the writer is PxgRigidDbContactWriter -> 176 bytes, the same
// layout as sb_primitiveContactGenLaunch and sb_heightfieldContactGenLaunch.
struct SbMeshCgPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, softbodies,
             shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad84;
    uint64_t stackSizeNeededOnDevice;
    uint64_t wPoint, wNormalPen, wBarycentric, wContactInfo, wRigidMaterialIndex, wTotalCount,
             wContactByRigid, wTempContactByRigid, wContactIndexSortedByRigid;
    uint32_t wMaxContacts, pad172;
};
static_assert(sizeof(SbMeshCgPush) == 176, "sb mesh CG blob must be 176 bytes");
static_assert(offsetof(SbMeshCgPush, stackSizeBytes) == 80, "stackSizeBytes @80");
static_assert(offsetof(SbMeshCgPush, stackSizeNeededOnDevice) == 88, "@88");
static_assert(offsetof(SbMeshCgPush, wPoint) == 96, "the inlined writer starts @96");
static_assert(offsetof(SbMeshCgPush, wMaxContacts) == 168, "maxNumContacts @168");

// PxgEncodeRigidFilterRef of a PxNodeIndex held as its 64-bit word.
static uint64_t rigidFilterRef(uint64_t node, uint32_t cacheRef)
{
    PxNodeIndex n;
    std::memcpy(&n, &node, sizeof(n));
    return PxgEncodeRigidFilterRef(n, cacheRef);
}
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

    // shapeToRigidRemapTable is indexed by the TRIMESH cache ref (1), not the softbody's (0).
    // Slot 0 is a decoy that must never appear in the output.
    std::vector<uint64_t> remap = { 0xDEAD0000DEAD0000ull, 0x0000000700000005ull };
    DevBuf bRemap = makeDev(dev, remap);
    const uint64_t RIGID_ID = remap[1];

    // The element filter (5.12): (rigid filter ref of the trimesh, TET_FILT).
    std::vector<PxgDeformablePairInfo> filters = { { rigidFilterRef(RIGID_ID, 1u),
                                                     PxgEncodeDeformableTetrahedron(SB_ID, TET_FILT) } };
    DevBuf bFilters = makeDev(dev, filters);

    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mRigidVolumeFilterPairs = reinterpret_cast<decltype(sb.mRigidVolumeFilterPairs)>(size_t(bFilters.addr));
    sb.mNumRigidVolumeFilterPairs = uint32_t(filters.size());
    sb.mNumTets = uint32_t(tets.size());
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr(SB_ID + 1u, makeZeroDesc());
    descArr[SB_ID] = sb;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  TRIMESH is shapeRef0 and SOFTBODY shapeRef1, i.e. SWAPPED, so the shape
    // load has to find the tet mesh by TYPE.  materialIndex carries bit 31 so the PxU16
    // narrowing in the writer is observable.
    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
    const uint32_t TRI_MATERIAL = 0x8001002Au;    // -> 0x002A = 42 after PxU16(...)
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
           bIdxSorted = makeDev(dev, poison), bSorted = makeDev(dev, poison);   // bSorted: the PxU16 materials
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
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.wPoint = bPoint.addr;
    push.wNormalPen = bNormalPen.addr;
    push.wBarycentric = bBary.addr;
    push.wContactInfo = bInfo.addr;
    push.wRigidMaterialIndex = bSorted.addr;
    push.wTotalCount = bCount.addr;
    push.wContactByRigid = bByX.addr;
    push.wTempContactByRigid = bTempByX.addr;
    push.wContactIndexSortedByRigid = bIdxSorted.addr;
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
    const auto gMat = downloadBuffer<uint16_t>(dev, bSorted.h, MAXC * 8u);
    const uint32_t gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    const auto gPair = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);

    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
        std::printf("  contact %u: p=(%.5f %.5f %.5f) rest=%.5f n=(%.5f %.5f %.5f) pen=%.6f "
                    "ref0=0x%llx ref1=0x%llx mat=%u\n",
                    i, gPoint[i].x, gPoint[i].y, gPoint[i].z, gPoint[i].w,
                    gNormalPen[i].x, gNormalPen[i].y, gNormalPen[i].z, gNormalPen[i].w,
                    (unsigned long long)gPair[i].ref0, (unsigned long long)gPair[i].ref1, uint32_t(gMat[i]));

    PXG_CHECK_MSG(gCount == 1u,
                  "exactly ONE of the four stack entries survives: the back-facing pair is "
                  "rejected by the gate, the filtered pair by the filter list, and the fourth "
                  "is clamped away -- got %u.  2 usually means the front-face cross product is "
                  "the familiar way round; 0 means the triangle is not where vertex2Shape "
                  "should have put it", gCount);

    // ---- the surviving contact ----------------------------------------------
    PXG_CHECK_MSG(gPair[0].ref0 == RIGID_ID,
                  "ref0 must be the PxNodeIndex from shapeToRigidRemapTable[TRIMESH cache "
                  "ref] = 0x%llx, got 0x%llx (0xDEAD.. means the SOFTBODY's ref was used)",
                  (unsigned long long)RIGID_ID, (unsigned long long)gPair[0].ref0);
    PXG_CHECK_MSG(gPair[0].ref1 == PxgEncodeDeformableTetrahedron(SB_ID, TET_HIT),
                  "ref1 must be the tetrahedron ref (softbody %u, tet %u); got 0x%llx",
                  SB_ID, TET_HIT, (unsigned long long)gPair[0].ref1);
    PXG_CHECK_MSG(gMat[0] == uint16_t(TRI_MATERIAL),
                  "the material index comes from the TRIMESH shape, stored as a PxU16: "
                  "expected %u, got %u", uint32_t(uint16_t(TRI_MATERIAL)), uint32_t(gMat[0]));

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
    // 5.12: the alternative write mode is gone; writeVolumeContact writes the sort key
    // (writeSortKey): contactByRigid = ref0, tempContactByRigid = PxgGetContactSortKey(ref0)
    // (its low word) and contactIndexSortedByRigid = the index.
    PXG_CHECK_MSG(gByX[0] == uint32_t(RIGID_ID & 0xFFFFFFFFull) && gByX[1] == uint32_t(RIGID_ID >> 32),
                  "contactByRigid carries the whole rigid id; got 0x%08x%08x", gByX[1], gByX[0]);
    PXG_CHECK_MSG(gTempByX[0] == uint32_t(RIGID_ID & 0xFFFFFFFFull),
                  "tempContactByRigid carries the rigid id's LOW word; got 0x%08x", gTempByX[0]);
    PXG_CHECK_MSG(gIdxSorted[0] == 0u, "contactIndexSortedByRigid is seeded with the identity");
    PXG_CHECK_MSG(gPoint[1].x == 0.0f && gNormalPen[1].x == 0.0f && gInfo[4] == POISON &&
                  gByX[2] == POISON && gTempByX[1] == POISON && gIdxSorted[1] == POISON,
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
// P9.89 -- sb_meshContactGenLaunch's PENETRATION, bit for bit.
//
// writeContactNoBarycentricFullWarp (softbodyPrimitives.cu:297) spreads the projection over
// the warp: lane i < 3 forms p_i = (witnessB_i - witnessA_i) * normal_i, lanes >= 3 form
// (0 - 0) * 0, and then EVERY lane evaluates (:339)
//
//     pen = ((shfl(pen, 0) + shfl(pen, 1)) + shfl(pen, 2)) + penOffset
//
// The mesh call site passes penOffset = a literal 0 and normal = the normalised GJK/EPA
// direction (:888-889).  Two things about that expression are pinned here:
//
//   * THE `+ penOffset` LEAF.  x + 0.0f is x for every x except -0, which it turns into +0.
//     The port spelled the sum as `dot(wB - wA, nrm)` and dropped the leaf, so a -0 sum
//     stayed -0.  The HF sibling (same helper, penOffset 0 at :1135-1136) got the P9.86
//     select at softBodyHfContactGenBda.slang:398-402; the mesh path never did.  A -0 sum
//     is reachable: a tetrahedron that TOUCHES the triangle exactly is not separated for
//     GJK, so EPA runs and ends with lowerBound = +-0; closestPoint = bestDir * (+-0) has +-0
//     components and closestPointB = closestPointA - closestPoint (epa.cuh:508, :521) IS
//     closestPointA.  Every lane's witnessB - witnessA is then +0, and when the normalised
//     direction has a set sign bit in all three components -- (-0, -1, -0) off a horizontal
//     mesh -- every product is +0 * (<= -0) = -0 and so is the sum.  The CUDA stores +0,
//     the old port -0.  LEVEL below is 24 such touches; the signs of the two zero components
//     come out of EPA's face cross products, and measured, 11 of the 24 hit (-0, -1, -0)
//     (the old port stored -0 on exactly those 11) while the other 13 carry a +0 component,
//     where both spellings agree.
//
//   * THE ASSOCIATION.  ((p0 + p1) + p2), not p0 + (p1 + p2): the shader is
//     NoContraction-stamped, which is what makes llvmpipe evaluate OpDot left to right.
//     SEPARATED below is 72 vertex-vertex contacts with non-round offsets, each converging
//     on GJK's size-1 simplex {apex - triangle vertex}, so closestPointA IS the apex
//     (gjk.cuh:412, closestPointA = vA[a0]) and the host knows every operand: the stored
//     point is witnessB (:414) and the stored normal is -normal, an exact negation.
//     Measured: the same shader built WITHOUT the stamp fails 16 of the 72, each with the
//     p0 + (p1 + p2) value.  The fma twin guards against a driver that fuses; llvmpipe
//     evaluates even an explicit GLSL Fma unfused, so an fma spelling is equivalent here.
//
// The host mirror below transcribes :339 literally; the twins are the old port's spelling
// (no penOffset), the right-to-left sum and the fma chain, and each must differ from the
// mirror on several contacts or this fixture no longer tells them apart.
// ===========================================================================
PXG_TEST(softBodySimMeshContactGen_p89PenOffset)
{
    using namespace p80refit;
    using namespace p88cloth;      // Bv32NodeHost
    using namespace p810cg;
    PxgSlangComputeDevice* dev = testDevice();

    // The mesh's placement, softBodySimMeshContactGen's: world = R * (vertex * scale) + T,
    // R(x, y, z) = (-x, y, -z).  Exact for these inputs, so the host knows the world
    // triangle bit for bit and the mesh plane is exactly y = TY.
    const float SX = 2.0f, SY = 1.0f, SZ = 0.5f;
    const float TX = 10.0f, TY = 20.0f, TZ = 30.0f;
    auto toWorld = [&](float vx, float vy, float vz) {
        return F4{ -(vx * SX) + TX, (vy * SY) + TY, -(vz * SZ) + TZ, 1.0f };
    };

    // Two triangles in the vertex-space plane y = 0, both wound so the kernel's reversed
    // cross product points DOWN and the front-face gate admits a tet from above: the
    // right-angled one of the base fixture and a scalene one.
    std::vector<F4> triVerts;
    std::vector<TetIdx4> triIdx;
    auto addTri = [&](F4 a, F4 b, F4 c) {
        const uint32_t base = uint32_t(triVerts.size());
        triVerts.push_back(a); triVerts.push_back(b); triVerts.push_back(c);
        triIdx.push_back({ base, base + 1u, base + 2u, 0u });
        return uint32_t(triIdx.size() - 1u);
    };
    addTri(F4{ 1.0f, 0.0f, 0.0f, 0.0f }, F4{ 1.0f, 0.0f, 0.4f, 0.0f }, F4{ 1.4f, 0.0f, 0.0f, 0.0f });
    addTri(F4{ 2.1f, 0.0f, 0.1f, 0.0f }, F4{ 2.4f, 0.0f, 0.7f, 0.0f }, F4{ 2.9f, 0.0f, 0.2f, 0.0f });
    const uint32_t NPRIM = uint32_t(triIdx.size());
    const uint32_t NVERT = uint32_t(triVerts.size());
    std::vector<F4> W(NVERT);
    for (uint32_t i = 0; i < NVERT; ++i) W[i] = toWorld(triVerts[i].x, triVerts[i].y, triVerts[i].z);
    PXG_CHECK_MSG(W[0].y == TY && W[4].y == TY, "the mesh plane must be exactly y = TY");

    std::vector<uint32_t> blob;
    {
        Bv32NodeHost node;
        std::memset(&node, 0, sizeof(node));
        node.mNbNodes = 1u;
        node.mData[0] = 1u | (NPRIM << 1);
        blob.resize(4u, 0u);
        blob[0] = NVERT; blob[1] = NPRIM; blob[2] = 0u; blob[3] = 1u;
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

    // ---- the tetrahedra; vertex 0 of each is the one nearest the mesh -------------------
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    std::vector<uint32_t> tetTri;
    auto addTet = [&](uint32_t tri, F4 p0, F4 p1, F4 p2, F4 p3) {
        const uint32_t b = uint32_t(tetVerts.size());
        tetVerts.push_back(p0); tetVerts.push_back(p1); tetVerts.push_back(p2); tetVerts.push_back(p3);
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
        tetTri.push_back(tri);
        return uint32_t(tets.size() - 1u);
    };
    auto P = [](float x, float y, float z) { return F4{ x, y, z, 1.0f }; };

    // LEVEL: apex-down tets whose apex lies EXACTLY on the plane y = TY, strictly inside a
    // triangle, base 0.2-0.4 above.  Deterministic pseudo-random placement.
    const uint32_t NB_LEVEL = 24u;
    {
        uint32_t lcg = 12345u;
        auto rnd = [&]() { lcg = lcg * 1664525u + 1013904223u; return float(lcg >> 8) / 16777216.0f; };
        for (uint32_t tri = 0; tri < 2u; ++tri)
            for (uint32_t k = 0; k < NB_LEVEL / 2u; ++k)
            {
                const F4 a = W[tri * 3u], b = W[tri * 3u + 1u], c = W[tri * 3u + 2u];
                float u = 0.1f + 0.6f * rnd(), v = 0.1f + 0.6f * rnd();
                if (u + v > 0.85f) { u *= 0.5f; v *= 0.5f; }
                const float x = a.x + u * (b.x - a.x) + v * (c.x - a.x);
                const float z = a.z + u * (b.z - a.z) + v * (c.z - a.z);
                const float h = 0.2f + 0.2f * rnd();
                const float r = 0.03f + 0.03f * rnd();
                const float ph = 6.2831853f * rnd();
                F4 q[3];
                for (int j = 0; j < 3; ++j)
                {
                    const float ang = ph + 2.0943951f * float(j);
                    q[j] = P(x + r * std::cos(ang), TY + h + 0.02f * rnd(), z + r * std::sin(ang));
                }
                addTet(tri, P(x, TY, z), q[0], q[1], q[2]);
            }
    }

    // SEPARATED: for every triangle vertex Wv, twelve apexes in Wv's Voronoi region -- off the
    // triangle along the outward bisector (turned by up to +-0.3 rad) and 0.012-0.036
    // above it, at most 0.046 from Wv, inside the 0.05 contact distance.  Every edge from
    // the apex runs AWAY from Wv (base = apex + L*u + r*(perpendicular), u = the unit
    // Wv -> apex), so (apex, Wv) is the unique closest pair and GJK ends on the size-1
    // simplex: witnessA IS the apex.
    const uint32_t NB_SEP_PER_VERT = 12u;
    {
        double offs[NB_SEP_PER_VERT][3];            // { lateral, height, turn }
        uint32_t lcg = 777u;
        auto rnd = [&]() { lcg = lcg * 1664525u + 1013904223u; return double(lcg >> 8) / 16777216.0; };
        for (uint32_t k = 0; k < NB_SEP_PER_VERT; ++k)
        {
            offs[k][0] = 0.004 + 0.024 * rnd();
            offs[k][1] = 0.012 + 0.024 * rnd();
            offs[k][2] = -0.29 + 0.58 * rnd();
        }
        for (uint32_t v = 0; v < NVERT; ++v)
        {
            const uint32_t tri = v / 3u, i0 = tri * 3u;
            const F4 w = W[v], n1 = W[i0 + (v - i0 + 1u) % 3u], n2 = W[i0 + (v - i0 + 2u) % 3u];
            double e1x = n1.x - w.x, e1z = n1.z - w.z, e2x = n2.x - w.x, e2z = n2.z - w.z;
            const double l1 = std::sqrt(e1x * e1x + e1z * e1z), l2 = std::sqrt(e2x * e2x + e2z * e2z);
            double ox = -(e1x / l1 + e2x / l2), oz = -(e1z / l1 + e2z / l2);
            const double lo = std::sqrt(ox * ox + oz * oz);
            ox /= lo; oz /= lo;
            for (uint32_t k = 0; k < NB_SEP_PER_VERT; ++k)
            {
                const double cs = std::cos(offs[k][2]), sn = std::sin(offs[k][2]);
                const double dx = (ox * cs - oz * sn) * offs[k][0], dz = (ox * sn + oz * cs) * offs[k][0];
                const double dy = offs[k][1];
                const F4 apex = P(float(w.x + dx), float(w.y + dy), float(w.z + dz));
                const double ln = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double ux = dx / ln, uy = dy / ln, uz = dz / ln;
                // p1 = normalize(cross(u, +y)) = normalize((-uz, 0, ux)), p2 = cross(u, p1)
                const double lp = std::sqrt(ux * ux + uz * uz);
                const double p1x = -uz / lp, p1z = ux / lp;
                const double p2x = uy * p1z, p2y = uz * p1x - ux * p1z, p2z = -uy * p1x;
                F4 q[3];
                for (int j = 0; j < 3; ++j)
                {
                    const double L = 0.25 + 0.037 * double(j + int(k)), rr = 0.09 + 0.011 * double(j);
                    const double ang = 0.37 * double(k) + 2.0943951 * double(j);
                    const double c = std::cos(ang) * rr, s = std::sin(ang) * rr;
                    q[j] = P(float(apex.x + ux * L + p1x * c + p2x * s),
                             float(apex.y + uy * L + p2y * s),
                             float(apex.z + uz * L + p1z * c + p2z * s));
                }
                addTet(tri, apex, q[0], q[1], q[2]);
            }
        }
    }
    const uint32_t NB = uint32_t(tets.size());
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets);

    const uint32_t SB_ID = 1u;
    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mNumTets = NB;
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr(SB_ID + 1u, makeZeroDesc());
    descArr[SB_ID] = sb;
    DevBuf bDesc = makeDev(dev, descArr);

    const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H;    shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = SX; shapes[0].sy = SY; shapes[0].sz = SZ; shapes[0].srw = 1.0f;
    shapes[0].materialIndex = 5u;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = SB_ID;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*trimesh cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);
    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f;
    xf[8 + 1] = 1.0f;                                          // 180 deg about y
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> cd = { 0.02f, 0.03f, 0.0f, 0.0f };      // contact distance 0.05
    DevBuf bCd = makeDev(dev, cd);
    std::vector<float> rd = { 0.125f, 0.0f };
    DevBuf bRd = makeDev(dev, rd);
    std::vector<uint64_t> remap = { 0ull, 0x0000000900000004ull };
    DevBuf bRemap = makeDev(dev, remap);

    // One stack entry per tet, so the pairs land on NB different (block, warp) slots.
    std::vector<uint32_t> stack;
    for (uint32_t t = 0; t < NB; ++t)
    {
        stack.push_back(0u); stack.push_back(t); stack.push_back(tetTri[t]); stack.push_back(0u);
    }
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { NB };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    const uint32_t MAXC = 128u;
    PXG_CHECK_MSG(NB <= MAXC, "fixture: %u pairs exceed the %u-contact buffer", NB, MAXC);
    std::vector<F4> zeroF4(MAXC, F4{ 0, 0, 0, 0 });
    std::vector<uint32_t> zeroU(MAXC * 4u, 0u);
    DevBuf bPoint = makeDev(dev, zeroF4), bNormalPen = makeDev(dev, zeroF4),
           bBary = makeDev(dev, zeroF4), bInfo = makeDev(dev, zeroU),
           bByX = makeDev(dev, zeroU), bTempByX = makeDev(dev, zeroU),
           bIdxSorted = makeDev(dev, zeroU), bSorted = makeDev(dev, zeroU);
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
    push.stackSizeBytes = NB * 16u;
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.wPoint = bPoint.addr;
    push.wNormalPen = bNormalPen.addr;
    push.wBarycentric = bBary.addr;
    push.wContactInfo = bInfo.addr;
    push.wRigidMaterialIndex = bSorted.addr;
    push.wTotalCount = bCount.addr;
    push.wContactByRigid = bByX.addr;
    push.wTempContactByRigid = bTempByX.addr;
    push.wContactIndexSortedByRigid = bIdxSorted.addr;
    push.wMaxContacts = MAXC;

    Kernel k("softBodyMeshContactGenBda.spv", "sb_meshContactGenLaunch", {}, sizeof(SbMeshCgPush));
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    const auto gPoint = downloadBuffer<F4>(dev, bPoint.h, MAXC);
    const auto gNormalPen = downloadBuffer<F4>(dev, bNormalPen.h, MAXC);
    const auto gInfo = downloadBuffer<uint32_t>(dev, bInfo.h, MAXC * 4u);

    PXG_CHECK_MSG(gCount == NB, "every one of the %u pairs is a contact (%u LEVEL touches, %u "
                  "separated within the contact distance); got %u", NB, NB_LEVEL,
                  NB - NB_LEVEL, gCount);

    // The contact order is the atomics'; map back by the encoded tet (ref1, 5.12).
    std::vector<int> slotOf(NB, -1);
    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
    {
        uint64_t ref1;
        std::memcpy(&ref1, &gInfo[i * 4u + 2u], sizeof(ref1));
        const uint32_t t = PxgGetDeformableFeatureId(ref1);
        PXG_CHECK_MSG(PxgGetDeformableId(ref1) == SB_ID && t < NB && slotOf[t] < 0 &&
                      ref1 == PxgEncodeDeformableTetrahedron(SB_ID, t),
                      "contact %u: ref1 0x%llx is not a fresh tet of softbody %u", i,
                      (unsigned long long)ref1, SB_ID);
        if (t < NB && slotOf[t] < 0) slotOf[t] = int(i);
    }

    uint32_t levelNegTwin = 0u, levelPosTwin = 0u, sepAssocDiffer = 0u, sepFmaDiffer = 0u;
    for (uint32_t t = 0; t < NB; ++t)
    {
        const int i = slotOf[t];
        PXG_CHECK_MSG(i >= 0, "tet %u produced no contact", t);
        if (i < 0) continue;
        const F4 apex = tetVerts[tets[t].x];
        const F4 wB = gPoint[i];
        // the stored xyz is -normal (flipOutputNormal = true); negation is exact
        const float nx = -gNormalPen[i].x, ny = -gNormalPen[i].y, nz = -gNormalPen[i].z;
        const float pen = gNormalPen[i].w;

        if (t < NB_LEVEL)
        {
            // A touch: the witness is the apex, on the plane, and the normal is straight down
            // with two zero components whose signs EPA's cross products decide.
            PXG_CHECK_MSG(f2u(wB.y) == f2u(TY) && std::fabs(wB.x - apex.x) <= 1e-5f &&
                          std::fabs(wB.z - apex.z) <= 1e-5f,
                          "level tet %u: the witness is the touching apex (%a %a %a); got "
                          "(%a %a %a)", t, apex.x, apex.y, apex.z, wB.x, wB.y, wB.z);
            PXG_CHECK_MSG(nx == 0.0f && nz == 0.0f && f2u(ny) == 0xbf800000u,
                          "level tet %u: the normal of a touch on a horizontal mesh is "
                          "(+-0, -1, +-0); got (0x%08x 0x%08x 0x%08x)", t, f2u(nx), f2u(ny),
                          f2u(nz));
            // witnessB - witnessA is +0 in every lane (see the header), so lane i holds
            // +0 * normal_i.  CUDA :339, transcribed: ((p0 + p1) + p2) + penOffset(0).
            const float d = 0.0f;
            const float p0 = d * nx, p1 = d * ny, p2 = d * nz;
            const float twin = (p0 + p1) + p2;            // the old port: OpDot, no penOffset
            const float cudaPen = twin + 0.0f;             // softbodyPrimitives.cu:339, :888
            if (f2u(twin) == 0x80000000u) ++levelNegTwin; else ++levelPosTwin;
            PXG_CHECK_MSG(f2u(cudaPen) == 0u && f2u(pen) == f2u(cudaPen),
                          "level tet %u: a touching contact's pen is the CUDA's sum + penOffset "
                          "0 = +0 (0x00000000); got 0x%08x -- 0x80000000 is the bare dot "
                          "without the `+ penOffset` leaf (normal 0x%08x 0x%08x 0x%08x)",
                          t, f2u(pen), f2u(nx), f2u(ny), f2u(nz));
        }
        else
        {
            // Vertex-vertex: witnessA is the apex exactly (GJK size-1 simplex).
            const float d0 = wB.x - apex.x, d1 = wB.y - apex.y, d2 = wB.z - apex.z;
            const float p0 = d0 * nx, p1 = d1 * ny, p2 = d2 * nz;
            const float cudaPen = ((p0 + p1) + p2) + 0.0f; // softbodyPrimitives.cu:339
            const float twinRL = p0 + (p1 + p2);           // undecorated OpDot on llvmpipe
            const float twinFma = std::fmaf(d2, nz, std::fmaf(d1, ny, p0));
            if (f2u(twinRL) != f2u(cudaPen)) ++sepAssocDiffer;
            if (f2u(twinFma) != f2u(cudaPen)) ++sepFmaDiffer;
            PXG_CHECK_MSG(pen > 0.0f && ny < 0.0f,
                          "separated tet %u: the apex is above its triangle vertex, so pen > 0 "
                          "and the normal points down; got pen %a, n.y %a", t, pen, ny);
            PXG_CHECK_MSG(f2u(pen) == f2u(cudaPen),
                          "separated tet %u: pen must be ((p0 + p1) + p2) + 0 of the CUDA "
                          "(softbodyPrimitives.cu:339) = 0x%08x; got 0x%08x (p0 + (p1 + p2) "
                          "= 0x%08x, fma chain = 0x%08x)", t, f2u(cudaPen), f2u(pen),
                          f2u(twinRL), f2u(twinFma));
        }
    }
    std::printf("  level: %u touches with a -0 bare dot, %u with +0; separated: %u of %u "
                "re-associate, %u of %u fuse differently\n", levelNegTwin, levelPosTwin,
                sepAssocDiffer, NB - NB_LEVEL, sepFmaDiffer, NB - NB_LEVEL);
    PXG_CHECK_MSG(levelNegTwin >= 6u && levelPosTwin >= 6u,
                  "coverage: the LEVEL touches must include both zero-sign patterns -- normal "
                  "(-0, -1, -0), where the bare dot is -0 and only `+ penOffset` makes it +0, "
                  "and a +0 component, where both spellings agree; got %u and %u",
                  levelNegTwin, levelPosTwin);
    PXG_CHECK_MSG(sepAssocDiffer >= 8u && sepFmaDiffer >= 4u,
                  "coverage: the separated contacts must tell ((p0 + p1) + p2) from p0 + (p1 + "
                  "p2) and from the fma chain; they differ on %u and %u contacts",
                  sepAssocDiffer, sepFmaDiffer);

    for (PxgBufferHandle h : { bBlob.h, bTetVerts.h, bTets.h, bDesc.h, bShapes.h, bCms.h, bXf.h,
                               bCd.h, bRd.h, bRemap.h, bStack.h, bPairNum.h, bNeeded.h,
                               bPoint.h, bNormalPen.h, bBary.h, bInfo.h, bByX.h, bTempByX.h,
                               bIdxSorted.h, bSorted.h, bCount.h })
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
    const uint32_t GEOM_HEIGHTFIELD_H = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
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
// P9.89 -- sb_heightfieldMidphaseGeneratePairsLaunch under a GENERIC rotation, pair set
// against the foundation's own PxBounds3::transformFast(hfTransform.getInverse(), bound)
// (softbodyHFMidPhase.cu:205).
//
// The fixture above turns the field by 180 degrees about y, which is exact, so every
// spelling of the inverse transform agrees there -- including the port's pre-P9.89
// rotateInv(q, centre - p) and rotateInv(q, e_i) basis, which differ from the CUDA's
// conj(q).rotate(centre) + rotateInv(q, -p) and PxMat33(conj q) by an ULP or two for any
// rotation with inexact entries.  An ULP of the local bound is a floor/ceil/truncation on
// the other side of an integer, i.e. a different pair SET.
//
// Same oracle, twins and knife-edge search as femClothSimHeightfieldMidphaseRotated
// (pxgHfLocalBoundRef.h), driven through the SOFTBODY kernel: each tetrahedron names one
// vertex four times, so tetBoundingBox + fattenFast(r) (softbodyHFMidPhase.cu:394-396) is
// (p - r, p + r), the sphere bound the search places.  2300 surface tetrahedra: past
// 2048, so the element loop runs twice.  Per twin, the fixture REQUIRES points that
// separate it from the CUDA, so it cannot pass on the pre-P9.89 port.
// ===========================================================================
PXG_TEST(softBodySimHeightfieldMidphaseRotated)
{
    using namespace p80refit;
    using namespace p811hf;
    PxgSlangComputeDevice* dev = testDevice();

    const hflb::Placement hf = hflb::rotatedPlacement();
    const std::vector<uint32_t> blob = hflb::rotatedBlob();
    const hflb::Field field{ blob.data() };
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const float CD_SB = 0.07f, CD_HF = 0.045f;
    const float radius = CD_SB + CD_HF;

    const uint32_t NB = 2300u;
    std::vector<physx::PxVec3> pts;
    std::vector<uint8_t> isKnife;
    hflb::rotatedPoints(hf, field, radius, NB, pts, isKnife);
    uint32_t nbKnives = 0u;
    for (uint8_t k : isKnife) nbKnives += k;

    // ---- the oracle, its composition check and the twins ----------------------------
    std::vector<std::pair<uint32_t, uint32_t>> ref;           // (tet, triangleIdx)
    std::vector<std::vector<uint32_t>> refPer(NB);
    uint32_t composedMismatch = 0u;
    for (uint32_t i = 0; i < NB; ++i)
    {
        const physx::PxBounds3 b = hflb::sphereBound(pts[i], radius);
        physx::PxVec3 mn, mx, cmn, cmx;
        hflb::localBound(hf, b, hflb::CUDA, mn, mx);
        hflb::localBound(hf, b, hflb::COMPOSED, cmn, cmx);
        for (uint32_t a = 0; a < 3u; ++a)
            if (f2u(mn[a]) != f2u(cmn[a]) || f2u(mx[a]) != f2u(cmx[a])) ++composedMismatch;
        hflb::pairs(field, mn, mx, refPer[i]);
        for (uint32_t t : refPer[i]) ref.push_back({ i, t });
    }
    std::sort(ref.begin(), ref.end());
    PXG_CHECK_MSG(composedMismatch == 0u,
                  "the hand-composed chain with nothing substituted differs from the "
                  "foundation's transformFast in %u bound components", composedMismatch);

    uint32_t sepCount[6] = {}, sepKnife[6] = {};
    for (uint32_t t = 0; t < 6u; ++t)
    {
        std::vector<std::pair<uint32_t, uint32_t>> twin;
        for (uint32_t i = 0; i < NB; ++i)
        {
            std::vector<uint32_t> got;
            hflb::pairsOfPoint(hf, field, pts[i], radius, hflb::ROT_TWINS[t], got);
            if (got != refPer[i]) { ++sepCount[t]; sepKnife[t] += isKnife[i]; }
            for (uint32_t g : got) twin.push_back({ i, g });
        }
        std::sort(twin.begin(), twin.end());
        PXG_CHECK_MSG(twin != ref && sepCount[t] >= 3u,
                      "twin '%s': only %u of %u tetrahedra emit a different pair set than the "
                      "CUDA (need >= 3) -- the fixture would pass on the pre-P9.89 port",
                      hflb::ROT_TWIN_NAMES[t], sepCount[t], NB);
    }

    // ---- the softbody: vertex i, tetrahedron i = (i, i, i, i), every hint set ----------
    std::vector<F4> tetVerts(NB);
    for (uint32_t i = 0; i < NB; ++i) tetVerts[i] = F4{ pts[i].x, pts[i].y, pts[i].z, 1.0f };
    std::vector<TetIdx4> tets(NB);
    for (uint32_t i = 0; i < NB; ++i) tets[i] = TetIdx4{ i, i, i, i };
    std::vector<uint32_t> hintWords((NB + 3u) / 4u + 1u, 0x01010101u);
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets),
           bHints = makeDev(dev, hintWords);
    std::vector<uint32_t> tetMeshHeader = { 999u, NB, 1u, 1u };     // numTets from .y
    DevBuf bTetMesh = makeDev(dev, tetMeshHeader);

    const uint32_t SB_ID = 2u;
    PxgSoftBody sb = makeZeroDesc();
    sb.mTetMeshData = reinterpret_cast<void*>(size_t(bTetMesh.addr));
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = NB;
    sb.mNumVerts = NB;
    std::vector<PxgSoftBody> descArr(SB_ID + 1u, makeZeroDesc());
    descArr[SB_ID] = sb;
    DevBuf bDesc = makeDev(dev, descArr);

    const uint32_t GEOM_HEIGHTFIELD_H = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_HEIGHTFIELD_H;     shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = hf.scale.x; shapes[0].sy = hf.scale.y; shapes[0].sz = hf.scale.z;
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
    xf[8 + 0] = hf.q.x; xf[8 + 1] = hf.q.y; xf[8 + 2] = hf.q.z; xf[8 + 3] = hf.q.w;
    xf[8 + 4] = hf.p.x; xf[8 + 5] = hf.p.y; xf[8 + 6] = hf.p.z;
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> cd = { CD_SB, CD_HF, 0.0f, 0.0f };
    DevBuf bCd = makeDev(dev, cd);

    const uint32_t CAP = 16384u;
    PXG_CHECK_MSG(ref.size() < CAP, "the reference has %zu pairs; raise CAP", ref.size());
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
    k.dispatch(/*gridX*/ 1024u, {}, push, 1u, 1u);          // PxgNarrowphaseCore.cpp:5360
    dev->flush();

    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
    std::vector<std::pair<uint32_t, uint32_t>> got;
    std::vector<std::vector<uint32_t>> gotPer(NB);
    for (uint32_t i = 0; i < gNum && i < CAP; ++i)
    {
        got.push_back({ gStack[i * 4u + 1u], gStack[i * 4u + 2u] });
        if (gStack[i * 4u + 1u] < NB) gotPer[gStack[i * 4u + 1u]].push_back(gStack[i * 4u + 2u]);
    }
    std::sort(got.begin(), got.end());
    uint32_t badElems = 0u, badKnives = 0u, firstBad = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < NB; ++i)
    {
        std::sort(gotPer[i].begin(), gotPer[i].end());
        std::vector<uint32_t> want = refPer[i];
        std::sort(want.begin(), want.end());
        if (gotPer[i] != want)
        {
            ++badElems; badKnives += isKnife[i];
            if (firstBad == 0xFFFFFFFFu) firstBad = i;
        }
    }
    PXG_CHECK_MSG(gNum == uint32_t(ref.size()) && got == ref && badElems == 0u,
                  "%u pairs against the foundation oracle's %zu; %u tetrahedra (%u of them "
                  "knife-edge points) emit a different set, first tet %u.  A knife-edge "
                  "mismatch is the association order of transformFast(getInverse()) -- "
                  "softbodyHFMidPhase.cu:205", gNum, ref.size(), badElems, badKnives, firstBad);

    std::printf("  %u tets (%u knife-edge), %zu pairs; tets separating each twin from the "
                "CUDA (knife-edge): all %u (%u), centre %u (%u), basis %u (%u), col0 %u (%u), "
                "col1 %u (%u), col2 %u (%u)\n",
                NB, nbKnives, ref.size(), sepCount[0], sepKnife[0], sepCount[1], sepKnife[1],
                sepCount[2], sepKnife[2], sepCount[3], sepKnife[3], sepCount[4], sepKnife[4],
                sepCount[5], sepKnife[5]);

    for (PxgBufferHandle h : { bBlob.h, bTetVerts.h, bTets.h, bHints.h, bTetMesh.h, bDesc.h,
                               bShapes.h, bCms.h, bXf.h, bCd.h, bStack.h, bPairNum.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P9.81 -- sb_heightfieldContactGenLaunch, the softbody-vs-HEIGHTFIELD contact generator and
// step 2 of the pair whose step 1 sits directly above.
//
// WHAT THIS FIXTURE IS FOR.  The GJK/EPA core is covered bit-exactly by the npTet* battery and
// again by every softbody contact-gen fixture before this one, and the heightfield DECODER is
// covered by step 1.  What is new is that this kernel is the mesh sibling's twin in shape and
// differs from it in exactly three places, and every one of the three is the kind of difference
// a scene-level settle number cannot see:
//
//   (1) NO FILTER ARM -- two fewer arguments, so the blob is 176 bytes and not 192.  Pinned by
//       the static_asserts below, which is where a shared row would be caught.
//   (2) NO FRONT-FACE GATE.  The mesh path keeps a contact only when
//       `dot(gjkDirection, n) >= 0`; this one keeps it either way.  So TWO tetrahedra, one
//       ABOVE the triangle and one BELOW it, must BOTH produce a contact.  A port that copies
//       the sibling's gate emits one of the two, never both.
//   (3) THE NORMAL WRITTEN IS THE TRIANGLE NORMAL, NOT THE GJK DIRECTION -- and therefore the
//       penetration is the witness separation projected on the TRIANGLE normal.  On a flat
//       field the triangle normal is exactly vertical, so a third tetrahedron is placed
//       OVERHANGING AN EDGE: its closest feature is the triangle's boundary, GJK's direction is
//       steeply tilted, and the two readings separate three ways at once --
//           stored normal   : exactly vertical    vs  tilted by ~0.95 horizontally
//           stored |pen|    : the VERTICAL gap    vs  the full slant distance
//           and the edge contact's pen equals the flat-above contact's, which it cannot under
//           the other reading.
//       One of those would do; three make the diagnosis unambiguous when it fires.
//
// Plus the parts it shares with its sibling and which must not drift: the rigid-shaped output
// (pairInd0 from shapeToRigidRemapTable[HEIGHTFIELD cache ref], the material from the
// HEIGHTFIELD shape folded with & 0x7fffffff), alternativeWriteMode's side arrays, the stack
// clamp and the UNCLAMPED stackSizeNeededOnDevice tail.
//
// THREE MORE TETRAHEDRA, EACH ADDED BECAUSE A MUTANT SURVIVED WITHOUT IT:
//   * an APEX-DOWN one, whose FOURTH VERTEX is the closest feature.  Every other tet here
//     points its base at the triangle and its apex away, so replacing v3 with v0 changes
//     nothing -- the same blind spot P9.73 found in the softbody floor scene, and a mutant
//     doing exactly that survived the first sweep.
//   * a PENETRATING one, the only pair that reaches GJK's OVERLAP arm and therefore EPA at
//     all.  It was added to test the one surviving mutant (a reversed search seed) on the
//     branch where a seed is most likely to matter.  It survived that too, which is what
//     turns "the seed is an initial direction, not an input" from a claim into a measurement.
//   * a LEVEL one (P9.86), whose apex lies ON the triangle's plane: the only ZERO penetration.
//     The bare dot(wB - wA, n) is -0 there; the CUDA's `+ penOffset` (0) makes it +0, and a
//     revert of the port's select survived until normalPen.w was compared bitwise.
//
// The triangle is not transcribed a second time: the expectation comes from hfref's
// getTriangleNoAdj, the same decoder step 1's fixture checks, taken to world through the same
// 180-degree rotation.
// ===========================================================================
namespace p981hfcg
{
struct SbHfCgPush
{
    float tolerenceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, softbodies,
             shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad84;
    uint64_t stackSizeNeededOnDevice;
    uint64_t wPoint, wNormalPen, wBarycentric, wContactInfo, wRigidMaterialIndex, wTotalCount,
             wContactByRigid, wTempContactByRigid, wContactIndexSortedByRigid;
    uint32_t wMaxContacts, pad172;
};
// 5.6.1 had the mesh sibling at 192 bytes (filterPairs / nbFilterPairs); in 5.12 both carry
// PxgRigidDbContactWriter at @96 and are 176 bytes.
static_assert(sizeof(SbHfCgPush) == 176, "sb heightfield CG blob must be 176 bytes");
static_assert(offsetof(SbHfCgPush, stackSizeBytes) == 80, "stackSizeBytes @80");
static_assert(offsetof(SbHfCgPush, stackSizeNeededOnDevice) == 88, "@88 -- @104 is the sibling");
static_assert(offsetof(SbHfCgPush, wPoint) == 96, "the inlined writer starts @96");
static_assert(offsetof(SbHfCgPush, wMaxContacts) == 168, "maxNumContacts @168");
} // namespace p981hfcg

PXG_TEST(softBodySimHeightfieldContactGen)
{
    using namespace p80refit;
    using namespace p981hfcg;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the heightfield: 4 rows x 5 columns, flat at height 0, every cell solid -----------
    // *** NOT SQUARE, AND THAT IS DELIBERATE. ***  Every vertex index in the decoder is
    // `row * nbCols + column`, so on a SQUARE field reading nbRows where nbCols belongs is
    // invisible -- the first draft of this fixture was 4 x 4 and a mutant swapping the two
    // survived.  With 4 x 5 the two readings land on different vertices.
    const uint32_t NB_ROWS = 4u, NB_COLS = 5u;
    const uint8_t MAT_SOLID = 3u;
    hfref::Hf hf;
    hf.rows = NB_ROWS; hf.cols = NB_COLS;
    hf.samples.assign(size_t(NB_ROWS) * NB_COLS, hfref::sampleWord(0, MAT_SOLID, false, MAT_SOLID));
    // *** THE FIELD IS NOT FLAT, ONLY THE TARGET CELL IS. ***  Every other sample is raised by
    // a distinct amount.  On a wholly flat field every triangle lies in the plane y = TY, so
    // ANY error that picks the wrong vertices -- a shifted sample base, a dropped rotation, a
    // swapped index -- still yields a horizontal triangle with a vertical normal and the
    // assertions below all pass.  Measured: the first draft was flat and `hf-samples-offset-4`
    // and `hf-drop-rotation` both survived for exactly that reason (Class D, P9.79).  With the
    // surroundings raised, picking the wrong vertices TILTS the triangle and the normal moves.
    for (uint32_t r = 0; r < NB_ROWS; ++r)
        for (uint32_t c = 0; c < NB_COLS; ++c)
            hf.samples[r * NB_COLS + c] =
                hfref::sampleWord(int16_t(37 + 13 * int(r) + 29 * int(c)), MAT_SOLID, false,
                                  MAT_SOLID);
    // the four corners of the target cell, back to height 0 so the triangle stays horizontal
    // and the penetration arithmetic below stays a one-liner
    for (uint32_t r = 1u; r <= 2u; ++r)
        for (uint32_t c = 1u; c <= 2u; ++c)
            hf.samples[r * NB_COLS + c] = hfref::sampleWord(0, MAT_SOLID, false, MAT_SOLID);
    hfref::buildHf(hf);
    DevBuf bBlob = makeDev(dev, hf.blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- its placement: the SAME map step 1 uses ------------------------------------------
    //     world = R * (hf * scale) + T,   R(x, y, z) = (-x, y, -z)
    // A 180-degree rotation about y is exact in floating point, so it pins the quaternion
    // rotate without spending tolerance on it.  All three scales are positive: the negative
    // scale swap is step 1's documented gap and this fixture does not pretend to cover it.
    const float ROW_SCALE = 2.0f, HEIGHT_SCALE = 0.01f, COL_SCALE = 0.5f;
    const float TX = 5.0f, TY = 7.0f, TZ = 9.0f;
    auto toWorld = [&](hfref::V3 v) {
        return hfref::V3{ -v.x + TX, v.y + TY, -v.z + TZ };
    };

    // ---- the triangle this fixture aims at, from the SAME decoder step 1 checks ------------
    // Cell (row 1, column 1) is vertexIndex 1 * nbCols + 1; its FIRST material slot is
    // triangle (vertexIndex << 1) | 0.  Derived rather than written out, so the non-square
    // field above cannot silently stop being the thing that is indexed.
    const uint32_t CELL_R = 1u, CELL_C = 1u;
    const uint32_t TRI = ((CELL_R * NB_COLS + CELL_C) << 1) | 0u;
    hfref::V3 s0, s1, s2;
    hfref::getTriangleNoAdj(hf, TRI, hfref::V3{ ROW_SCALE, HEIGHT_SCALE, COL_SCALE }, s0, s1, s2);
    const hfref::V3 W0 = toWorld(s0), W1 = toWorld(s1), W2 = toWorld(s2);
    // *** THE CROSS PRODUCT IS THE OTHER WAY ROUND *** -- v20 x v21, not the familiar one.
    const hfref::V3 v20{ W2.x - W0.x, W2.y - W0.y, W2.z - W0.z };
    const hfref::V3 v21{ W1.x - W0.x, W1.y - W0.y, W1.z - W0.z };
    hfref::V3 nExp{ v20.y * v21.z - v20.z * v21.y,
                    v20.z * v21.x - v20.x * v21.z,
                    v20.x * v21.y - v20.y * v21.x };
    {
        const float m = std::sqrt(nExp.x * nExp.x + nExp.y * nExp.y + nExp.z * nExp.z);
        nExp = hfref::V3{ nExp.x / m, nExp.y / m, nExp.z / m };
    }
    const hfref::V3 C{ (W0.x + W1.x + W2.x) / 3.0f, (W0.y + W1.y + W2.y) / 3.0f,
                       (W0.z + W1.z + W2.z) / 3.0f };
    std::printf("  triangle %u: W0=(%.4f %.4f %.4f) W1=(%.4f %.4f %.4f) W2=(%.4f %.4f %.4f)\n"
                "  centroid=(%.4f %.4f %.4f)  n=(%.5f %.5f %.5f)\n",
                TRI, W0.x, W0.y, W0.z, W1.x, W1.y, W1.z, W2.x, W2.y, W2.z,
                C.x, C.y, C.z, nExp.x, nExp.y, nExp.z);
    // A flat field: every triangle lies in the plane y = TY, so its normal is exactly vertical.
    // If that ever stops holding the geometry below stops meaning what it says, so it is checked
    // rather than assumed.
    PXG_CHECK_MSG(std::fabs(nExp.x) < 1e-6f && std::fabs(nExp.z) < 1e-6f &&
                  std::fabs(std::fabs(nExp.y) - 1.0f) < 1e-6f,
                  "the field is flat, so the triangle normal must be exactly vertical; got "
                  "(%.6f %.6f %.6f)", nExp.x, nExp.y, nExp.z);

    // ---- the tetrahedra, built directly in WORLD space -------------------------------------
    // The kernel reads mPosition_InvMass as world, so there is no transform to invert here.
    // Each is a small triangular prism collapsed to a tet: a horizontal base face of side ~0.12
    // and an apex 0.5 away from it, so the closest feature to the triangle is the BASE.
    const float GAP = 0.02f;         // the vertical separation, well inside cDistance
    const float EDGE_OUT = 0.06f;    // how far the edge tet overhangs, three times the gap
    std::vector<F4> tetVerts;
    std::vector<TetIdx4> tets;
    auto addTet = [&](float cx, float cz, float baseY, float apexDir) {
        const uint32_t b = uint32_t(tetVerts.size());
        tetVerts.push_back(F4{ cx - 0.06f, baseY, cz - 0.06f, 1.0f });
        tetVerts.push_back(F4{ cx + 0.06f, baseY, cz - 0.06f, 1.0f });
        tetVerts.push_back(F4{ cx,         baseY, cz + 0.06f, 1.0f });
        tetVerts.push_back(F4{ cx,         baseY + apexDir * 0.5f, cz, 1.0f });
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
        return uint32_t(tets.size() - 1u);
    };
    // ABOVE: base GAP above the plane, apex pointing up and away.
    const uint32_t T_ABOVE = addTet(C.x, C.z, C.y + GAP, +1.0f);
    // BELOW: base GAP below the plane, apex pointing down and away.  Difference (2): the mesh
    // sibling's gate admits one of these two and rejects the other; this kernel keeps both.
    const uint32_t T_BELOW = addTet(C.x, C.z, C.y - GAP, -1.0f);
    // EDGE: the same height as ABOVE, but pushed past the triangle's W0 corner along the
    // outward direction, so the closest point on the triangle is its BOUNDARY and GJK's
    // direction is tilted ~72 degrees off vertical.  Difference (3).
    const float outx = (W0.x - C.x), outz = (W0.z - C.z);
    const float outm = std::sqrt(outx * outx + outz * outz);
    const uint32_t T_EDGE = addTet(W0.x + (outx / outm) * EDGE_OUT,
                                   W0.z + (outz / outm) * EDGE_OUT, C.y + GAP, +1.0f);
    // FAR: ten units up, far outside any contact distance.
    const uint32_t T_FAR = addTet(C.x, C.z, C.y + 10.0f, +1.0f);
    // APEX: the only tetrahedron whose FOURTH VERTEX is the closest feature.  Every tet above
    // has its base toward the triangle and its apex pointing away, so replacing v3 with v0
    // changes nothing and a mutant that does exactly that survives -- measured, and the same
    // blind spot P9.73 found in the softbody floor scene.  This one is inverted: its BASE is
    // 0.5 above the plane and its apex reaches down to GAP above it.
    // DEEP: the ONLY penetrating pair here, and the only one that reaches GJK's OVERLAP arm and
    // therefore EPA.  Every other tetrahedron is separated, where GJK converges to the unique
    // closest-point pair whatever direction it was seeded with -- which is why a mutant
    // reversing the search direction survived a fixture made only of separated pairs.
    const float DEEP_D = 0.01f;   // how far its base sits BELOW the triangle's plane
    const uint32_t T_DEEP = addTet(C.x + 0.25f, C.z, C.y - DEEP_D, +1.0f);
    const uint32_t T_APEX = [&]() {
        const uint32_t b = uint32_t(tetVerts.size());
        const float baseY = C.y + GAP + 0.5f;
        tetVerts.push_back(F4{ C.x - 0.3f, baseY, C.z - 0.06f, 1.0f });
        tetVerts.push_back(F4{ C.x - 0.3f, baseY, C.z + 0.06f, 1.0f });
        tetVerts.push_back(F4{ C.x - 0.18f, baseY, C.z, 1.0f });
        tetVerts.push_back(F4{ C.x - 0.24f, C.y + GAP, C.z, 1.0f });   // the apex, reaching DOWN
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
        return uint32_t(tets.size() - 1u);
    }();
    // LEVEL (P9.86): the only pair whose penetration is exactly ZERO.  Its apex P sits AT the
    // triangle's height, 0.001/0.01 short of W1 in x/z and inside W1's Voronoi region, and all
    // three edges from P run away from W1, so GJK converges on the size-1 simplex {P - W1}:
    // closestPointA = P, cp.y = 7 - 7 = +0, closestPointDir.y = -0, wB.y = 7 + ub*(-0) = 7.
    // With this triangle's n = (-0, -1, -0) the three lanes of dot(wB - wA, n) are dx*(-0),
    // (+0)*(-1), dz*(-0) -- all -0 -- so the bare dot is -0.  The CUDA adds penOffset, a
    // literal 0 (softbodyPrimitives.cu:339, passed at :1136), which makes it +0; the port's
    // select at softBodyHfContactGenBda.slang:402 does the same.  Checked BITWISE below.
    // (addTet with apexDir -1 would NOT do: its v2 at cz + 0.06 leans toward W1 and the closest
    // point would slide up an edge, making dy nonzero.)
    const float LVL_DX = 0.001f, LVL_DZ = 0.01f;
    const F4 P_LEVEL{ W1.x - LVL_DX, W1.y, W1.z - LVL_DZ, 1.0f };
    const uint32_t T_LEVEL = [&]() {
        const uint32_t b = uint32_t(tetVerts.size());
        const float baseY = W1.y + 0.5f;
        tetVerts.push_back(F4{ P_LEVEL.x - 0.06f, baseY, P_LEVEL.z - 0.06f, 1.0f });
        tetVerts.push_back(F4{ P_LEVEL.x + 0.06f, baseY, P_LEVEL.z - 0.06f, 1.0f });
        tetVerts.push_back(F4{ P_LEVEL.x,         baseY, P_LEVEL.z - 0.12f, 1.0f });
        tetVerts.push_back(P_LEVEL);                                   // the apex, ON the plane
        tets.push_back({ b, b + 1u, b + 2u, b + 3u });
        return uint32_t(tets.size() - 1u);
    }();
    PXG_CHECK_MSG(W1.y == TY && std::signbit(nExp.x) && nExp.x == 0.0f &&
                  std::signbit(nExp.z) && nExp.z == 0.0f && nExp.y < 0.0f,
                  "LEVEL needs the triangle normal n = (-0, <0, -0) and W1 on y = TY; got "
                  "(%a %a %a), W1.y = %a", nExp.x, nExp.y, nExp.z, W1.y);
    DevBuf bTetVerts = makeDev(dev, tetVerts), bTets = makeDev(dev, tets);

    const uint32_t SB_ID = 2u;
    PxgSoftBody sb = makeZeroDesc();
    sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bTetVerts.addr));
    sb.mNumVerts = uint32_t(tetVerts.size());
    std::vector<PxgSoftBody> descArr(SB_ID + 1u, makeZeroDesc());
    descArr[SB_ID] = sb;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  HEIGHTFIELD is shapeRef0, SOFTBODY shapeRef1 -- the SWAPPED order, so a
    // port that trusts position rather than geometry type reads the wrong half.
    const uint32_t GEOM_HEIGHTFIELD_H = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
    const uint32_t HF_MATERIAL = 0x80000005u;   // the HIGH BIT is set: the fold must drop it
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_HEIGHTFIELD_H;     shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = ROW_SCALE; shapes[0].sy = HEIGHT_SCALE; shapes[0].sz = COL_SCALE;
    shapes[0].srw = 1.0f;
    shapes[0].materialIndex = HF_MATERIAL;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TETRAHEDRONMESH_H; shapes[1].particleOrSoftbodyId = SB_ID;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    shapes[1].materialIndex = 0x0BADBADu;    // must never be the one that reaches the contact
    DevBuf bShapes = makeDev(dev, shapes);
    // cache refs: HEIGHTFIELD 1, SOFTBODY 0 -- distinct, so indexing the remap table or the
    // contact distances with the wrong one shows.
    std::vector<CmInputHost> cms = { { 0u, 1u, /*hf cache*/ 1u, /*sb cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    // transform cache: slot 0 (the softbody) identity, slot 1 (the heightfield) R and T.
    std::vector<float> xf(2 * 8 + 4, 0.0f);
    xf[3] = 1.0f;
    xf[8 + 1] = 1.0f;                                       // 180 deg about y
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);

    // cDistance = contactDistance[sbCacheRef] + contactDistance[hfCacheRef] = 0.012 + 0.011.
    // *** NEITHER HALF ALONE REACHES THE GAP. ***  0.023 clears the 0.02 separation; 0.012 and
    // 0.011 do not.  With 0.03 + 0.02 -- the first draft -- a mutant that dropped one of the
    // two addends still had 0.02 and still emitted every contact, and it survived.
    std::vector<float> cd = { 0.012f, 0.011f, 999.0f, 999.0f };
    DevBuf bCd = makeDev(dev, cd);
    std::vector<float> rd = { 0.125f, 777.0f };
    DevBuf bRd = makeDev(dev, rd);

    // shapeToRigidRemapTable is indexed by the HEIGHTFIELD's cache ref (1).  Slot 0 carries a
    // poison so using the softbody's ref is visible rather than merely wrong.
    const uint64_t RIGID_ID = 0x00000007ABCD1234ull;
    std::vector<uint64_t> remap = { 0xDEADDEADDEADDEADull, RIGID_ID };
    DevBuf bRemap = makeDev(dev, remap);

    // ---- the pair stack, and the clamp -----------------------------------------------------
    // EIGHT entries are declared in midphasePairsNum but stackSizeBytes only covers SEVEN, so
    // the eighth -- which WOULD emit, being a copy of the first -- must be clamped away, while
    // the tail still reports the unclamped demand.
    const uint32_t NB_STACK_ENTRIES = 8u, DECLARED = 7u;
    std::vector<uint32_t> stack;
    auto push = [&](uint32_t tet) {
        stack.push_back(0u); stack.push_back(tet); stack.push_back(TRI); stack.push_back(0u);
    };
    push(T_ABOVE); push(T_BELOW); push(T_EDGE); push(T_APEX); push(T_DEEP); push(T_LEVEL);
    push(T_FAR);
    push(T_ABOVE);                       // beyond the clamp
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { NB_STACK_ENTRIES };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    // ---- the contact writer's buffers ------------------------------------------------------
    const uint32_t MAXC = 8u, POISON = 0xFEEDFACEu;
    std::vector<F4> zeroF4(MAXC, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<uint32_t> poison(MAXC * 4u, POISON);
    DevBuf bPoint = makeDev(dev, zeroF4), bNormalPen = makeDev(dev, zeroF4),
           bBary = makeDev(dev, zeroF4);
    DevBuf bInfo = makeDev(dev, poison), bByX = makeDev(dev, poison),
           bTempByX = makeDev(dev, poison), bIdxSorted = makeDev(dev, poison),
           bSorted = makeDev(dev, poison);
    std::vector<uint32_t> count = { 0u };
    DevBuf bCount = makeDev(dev, count);

    SbHfCgPush p = {};
    p.tolerenceLength = 1.0f;
    p.cmInputs = bCms.addr;
    p.transformCache = bXf.addr;
    p.contactDistance = bCd.addr;
    p.restDistance = bRd.addr;
    p.gpuShapes = bShapes.addr;
    p.softbodies = bDesc.addr;
    p.shapeToRigidRemapTable = bRemap.addr;
    p.stackPtr = bStack.addr;
    p.midphasePairsNum = bPairNum.addr;
    p.stackSizeBytes = DECLARED * 16u;
    p.stackSizeNeededOnDevice = bNeeded.addr;
    p.wPoint = bPoint.addr; p.wNormalPen = bNormalPen.addr; p.wBarycentric = bBary.addr;
    p.wContactInfo = bInfo.addr; p.wRigidMaterialIndex = bSorted.addr; p.wTotalCount = bCount.addr;
    p.wContactByRigid = bByX.addr; p.wTempContactByRigid = bTempByX.addr;
    p.wContactIndexSortedByRigid = bIdxSorted.addr; p.wMaxContacts = MAXC;

    {
        Kernel k("softBodyHfContactGenBda.spv", "sb_heightfieldContactGenLaunch",
                 {}, sizeof(SbHfCgPush));
        // The engine's grid (PxgNarrowphaseCore.cpp:5462): x = SB_HFCG = 1024, y = z = 1,
        // block (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK).
        k.dispatch(/*gridX*/ 1024u, {}, p, 1u, 1u);
    }
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    const auto gPoint = downloadBuffer<F4>(dev, bPoint.h, MAXC);
    const auto gNormalPen = downloadBuffer<F4>(dev, bNormalPen.h, MAXC);
    const auto gInfo = downloadBuffer<uint32_t>(dev, bInfo.h, MAXC * 4u);
    const auto gByX = downloadBuffer<uint32_t>(dev, bByX.h, MAXC * 2u);
    const auto gTempByX = downloadBuffer<uint32_t>(dev, bTempByX.h, MAXC * 4u);
    const auto gIdxSorted = downloadBuffer<uint32_t>(dev, bIdxSorted.h, MAXC * 4u);
    const auto gMat = downloadBuffer<uint16_t>(dev, bSorted.h, MAXC * 8u);   // PxU16 materials (5.12)
    const uint32_t gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    const auto gPair = downloadBuffer<PxgDeformablePairInfo>(dev, bInfo.h, MAXC);

    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
        std::printf("  contact %u: p=(%.5f %.5f %.5f) rest=%.5f n=(%.6f %.6f %.6f) pen=%.6f "
                    "ref0=0x%llx tet %u mat=%u\n",
                    i, gPoint[i].x, gPoint[i].y, gPoint[i].z, gPoint[i].w,
                    gNormalPen[i].x, gNormalPen[i].y, gNormalPen[i].z, gNormalPen[i].w,
                    (unsigned long long)gPair[i].ref0, PxgGetDeformableFeatureId(gPair[i].ref1), uint32_t(gMat[i]));

    // ---- difference (2): BOTH sides of the triangle survive ---------------------------------
    PXG_CHECK_MSG(gCount == 6u,
                  "six of the eight stack entries must emit: the tet ABOVE the triangle, the "
                  "one BELOW it, the one overhanging its EDGE, the APEX-down one, the DEEP "
                  "penetrating one and the LEVEL one.  The far tet is outside the contact "
                  "distance and the eighth entry is beyond the stack clamp.  Got %u -- 4 means the mesh sibling's "
                  "front-face gate was copied in, and this kernel does not have one "
                  "(softbodyPrimitives.cu:1130 vs :878-881)", gCount);

    // index the contacts by their tetrahedron, since the emission order is an atomic race
    int idxAbove = -1, idxBelow = -1, idxEdge = -1, idxApex = -1, idxDeep = -1,
        idxLevel = -1;
    for (uint32_t i = 0; i < gCount && i < MAXC; ++i)
    {
        const uint32_t tet = PxgGetDeformableFeatureId(gPair[i].ref1);
        if (tet == T_ABOVE) idxAbove = int(i);
        else if (tet == T_BELOW) idxBelow = int(i);
        else if (tet == T_EDGE) idxEdge = int(i);
        else if (tet == T_APEX) idxApex = int(i);
        else if (tet == T_DEEP) idxDeep = int(i);
        else if (tet == T_LEVEL) idxLevel = int(i);
        PXG_CHECK_MSG(tet != T_FAR, "the tet ten units above the triangle must not emit: "
                                    "cDistance is 0.023");
    }
    // *** RETURNING HERE IS NOT TIDINESS. ***  Every check below indexes gNormalPen with one of
    // these, and a missing contact leaves the index at -1.  The first version read on and
    // SEGFAULTED, which the sweep duly scored as a kill -- but a fixture that crashes instead
    // of printing its diagnosis has thrown away the only thing that makes the kill useful.
    if (idxAbove < 0 || idxBelow < 0 || idxEdge < 0 || idxApex < 0 || idxDeep < 0 ||
        idxLevel < 0)
    {
        PXG_CHECK_MSG(false,
                      "the six contacts must be one per tetrahedron (above %d, below %d, edge "
                      "%d, apex %d, deep %d, level %d); a -1 is a tetrahedron that produced "
                      "nothing and a repeat means the clamped eighth entry was processed",
                      idxAbove, idxBelow, idxEdge, idxApex, idxDeep, idxLevel);
        for (PxgBufferHandle h : { bBlob.h, bTetVerts.h, bTets.h, bDesc.h, bShapes.h, bCms.h,
                                   bXf.h, bCd.h, bRd.h, bRemap.h, bStack.h, bPairNum.h,
                                   bNeeded.h, bPoint.h, bNormalPen.h, bBary.h, bInfo.h, bByX.h,
                                   bTempByX.h, bIdxSorted.h, bSorted.h, bCount.h })
            dev->destroyBuffer(h);
        return;
    }

    auto near = [](float a, float b, float t) { return std::fabs(a - b) <= t; };

    // ---- difference (3): the stored normal is the TRIANGLE normal, negated -----------------
    const int idxOf[6] = { idxAbove, idxBelow, idxEdge, idxApex, idxDeep, idxLevel };
    const char* nameOf[6] = { "above", "below", "edge", "apex", "deep", "level" };
    for (int which = 0; which < 6; ++which)
    {
        const int i = idxOf[which];
        const char* nm = nameOf[which];
        PXG_CHECK_MSG(near(gNormalPen[i].x, -nExp.x, 1e-4f) &&
                      near(gNormalPen[i].y, -nExp.y, 1e-4f) &&
                      near(gNormalPen[i].z, -nExp.z, 1e-4f),
                      "%s: the stored normal is the TRIANGLE normal negated "
                      "(flipOutputNormal = true), i.e. (%.5f %.5f %.5f); got (%.6f %.6f %.6f).  "
                      "A tilted normal on the EDGE contact means GJK's direction was stored "
                      "instead -- that is the mesh sibling's spelling, not this one's",
                      nm, -nExp.x, -nExp.y, -nExp.z,
                      gNormalPen[i].x, gNormalPen[i].y, gNormalPen[i].z);
    }

    // ---- and therefore the penetration is the VERTICAL gap, on all three -------------------
    // pen = dot(witnessB - witnessA, n).  For the ABOVE tet the witnesses differ by (0,-GAP,0)
    // and for the BELOW tet by (0,+GAP,0); the EDGE tet's base sits at the same height as the
    // ABOVE tet's, so its projection on a VERTICAL normal is the same number -- while the slant
    // distance its GJK direction would measure is sqrt(EDGE_OUT^2 + GAP^2) = 0.0632, three
    // times larger.  That is the sharpest of the three reads.
    const float penAbove = -GAP * nExp.y;
    PXG_CHECK_MSG(near(gNormalPen[idxAbove].w, penAbove, 2e-4f),
                  "above: pen is the witness separation projected on the triangle normal, "
                  "%.6f; got %.6f", penAbove, gNormalPen[idxAbove].w);
    PXG_CHECK_MSG(near(gNormalPen[idxBelow].w, -penAbove, 2e-4f),
                  "below: the same gap on the other side, so pen flips sign to %.6f; got %.6f",
                  -penAbove, gNormalPen[idxBelow].w);
    PXG_CHECK_MSG(near(gNormalPen[idxEdge].w, penAbove, 2e-4f),
                  "edge: its base is at the SAME height as the above tet's, so projected on the "
                  "triangle normal its pen is the SAME %.6f.  Got %.6f -- about %.4f is the "
                  "slant distance, which is what GJK's direction would have measured",
                  penAbove, gNormalPen[idxEdge].w,
                  std::sqrt(EDGE_OUT * EDGE_OUT + GAP * GAP));
    PXG_CHECK_MSG(near(gNormalPen[idxApex].w, penAbove, 2e-4f),
                  "apex: its FOURTH VERTEX reaches down to the same height, so its pen is the "
                  "same %.6f; got %.6f (a much larger gap means the apex was not the closest "
                  "feature, i.e. the fourth vertex never reached the cloud)",
                  penAbove, gNormalPen[idxApex].w);

    // The DEEP tet OVERLAPS: its base is DEEP_D below the plane, so the witness separation runs
    // the other way along n and the projection changes sign against the separated contacts.
    // This is the one pair that reaches GJK's OVERLAP arm and therefore EPA at all.
    PXG_CHECK_MSG(near(gNormalPen[idxDeep].w, DEEP_D * nExp.y, 2e-4f),
                  "deep: it penetrates by %.3f, so projected on the triangle normal its pen is "
                  "%.6f -- the OPPOSITE SIGN to the separated contacts' %.6f; got %.6f",
                  DEEP_D, DEEP_D * nExp.y, penAbove, gNormalPen[idxDeep].w);

    // ---- WHERE the edge contact's witness landed -------------------------------------------
    // The edge tet overhangs the W0 corner, so the closest point on the triangle is ON ITS
    // BOUNDARY at that corner's x.  This is the assertion that pins the triangle's POSITION --
    // every check above is about its normal and its height, and on a horizontal triangle those
    // survive any in-plane displacement.  Measured: with only the normal and the height checked,
    // `hf-drop-rotation` moved the whole triangle two units and SURVIVED.
    PXG_CHECK_MSG(near(gPoint[idxEdge].x, W0.x, 1e-3f),
                  "the edge witness lies on the triangle's boundary through W0, at x = %.4f; got "
                  "%.5f -- a different x means the decoded triangle is not where the heightfield "
                  "puts it", W0.x, gPoint[idxEdge].x);

    // ---- LEVEL: a ZERO penetration is +0, never -0 (P9.86) ------------------------------------
    // BITWISE: near() is fabs(a - b) and cannot see the sign of zero.
    {
        const int i = idxLevel;
        PXG_CHECK_MSG(f2u(gNormalPen[i].x) == 0u && f2u(gNormalPen[i].z) == 0u &&
                      gNormalPen[i].y > 0.0f,
                      "level: the stored -n must be (+0, >0, +0), i.e. n = (-0, <0, -0); got "
                      "(0x%08x 0x%08x 0x%08x)", f2u(gNormalPen[i].x), f2u(gNormalPen[i].y),
                      f2u(gNormalPen[i].z));
        PXG_CHECK_MSG(f2u(gPoint[i].y) == f2u(P_LEVEL.y),
                      "level: the triangle witness must be at the apex's height EXACTLY (dy = +0); "
                      "got y = %a vs %a", gPoint[i].y, P_LEVEL.y);
        PXG_CHECK_MSG(gPoint[i].x > P_LEVEL.x && gPoint[i].z > P_LEVEL.z,
                      "level: the triangle witness lies toward W1, +x/+z of the apex; got "
                      "(%.6f %.6f) vs apex (%.6f %.6f)", gPoint[i].x, gPoint[i].z, P_LEVEL.x,
                      P_LEVEL.z);
        // wA = P: the simplex is size 1.  The twin is the UNFIXED spelling dot(wB - wA, n).
        const float nx = -gNormalPen[i].x, ny = -gNormalPen[i].y, nz = -gNormalPen[i].z;
        const float twinRevert = (gPoint[i].x - P_LEVEL.x) * nx + (gPoint[i].y - P_LEVEL.y) * ny +
                                 (gPoint[i].z - P_LEVEL.z) * nz;
        const uint32_t penNegZeroSeen =
            (twinRevert == 0.0f && std::signbit(twinRevert)) ? 1u : 0u;
        PXG_CHECK_MSG(penNegZeroSeen == 1u,
                      "coverage: the unfixed spelling dot(wB - wA, n) must be -0 on the LEVEL "
                      "contact, or this case no longer exercises the P9.86 select; got %a",
                      twinRevert);
        const float cudaPen = twinRevert + 0.0f;   // softbodyPrimitives.cu:339, penOffset 0 (:1136)
        PXG_CHECK_MSG(f2u(cudaPen) == 0u && f2u(gNormalPen[i].w) == f2u(cudaPen),
                      "level: the CUDA adds penOffset 0, so a zero pen is +0 (0x00000000); got "
                      "0x%08x -- 0x80000000 means the select at softBodyHfContactGenBda.slang:402 "
                      "was dropped", f2u(gNormalPen[i].w));
        uint32_t zeroPenSeen = 0u;
        for (uint32_t j = 0; j < gCount && j < MAXC; ++j)
            if (gNormalPen[j].w == 0.0f)
            {
                ++zeroPenSeen;
                PXG_CHECK_MSG(f2u(gNormalPen[j].w) == 0u,
                              "contact %u: a zero pen must be +0; got 0x%08x", j,
                              f2u(gNormalPen[j].w));
            }
        PXG_CHECK_MSG(zeroPenSeen == 1u, "coverage: exactly one zero-pen contact (LEVEL); got %u",
                      zeroPenSeen);
    }

    // ---- the rigid-shaped output, shared with the sibling and not allowed to drift ----------
    for (int which = 0; which < 6; ++which)
    {
        const int i = idxOf[which];
        PXG_CHECK_MSG(gPair[i].ref0 == RIGID_ID,
                      "ref0 is the PxNodeIndex from shapeToRigidRemapTable[HEIGHTFIELD "
                      "cache ref] = 0x%llx; got 0x%llx (0xDEAD.. means the SOFTBODY's "
                      "ref was used)", (unsigned long long)RIGID_ID, (unsigned long long)gPair[i].ref0);
        PXG_CHECK_MSG(gPair[i].ref1 == PxgEncodeDeformableTetrahedron(SB_ID, PxgGetDeformableFeatureId(gPair[i].ref1)) &&
                      PxgGetDeformableId(gPair[i].ref1) == SB_ID,
                      "ref1 is a tetrahedron ref of softbody %u; got 0x%llx",
                      SB_ID, (unsigned long long)gPair[i].ref1);
        PXG_CHECK_MSG(gMat[i] == uint16_t(HF_MATERIAL),
                      "the material comes from the HEIGHTFIELD shape, stored as a PxU16: "
                      "expected %u, got %u (%u is the softbody's decoy)",
                      uint32_t(uint16_t(HF_MATERIAL)), uint32_t(gMat[i]), 0x0BADBADu);
        PXG_CHECK_MSG(near(gPoint[i].w, 0.125f, 1e-6f),
                      "the point's w is restDistance[cmIdx = 0] = 0.125; got %.6f (777 means "
                      "cmIdx was ignored)", gPoint[i].w);
        // the stored point is the witness on the SECOND cloud, i.e. on the TRIANGLE, whose
        // plane is y = TY exactly
        PXG_CHECK_MSG(near(gPoint[i].y, TY, 2e-3f),
                      "the stored point is the witness on the TRIANGLE, at world y = %.3f; got "
                      "%.5f (a value %.3f away is the witness on the TET instead)",
                      TY, gPoint[i].y, GAP);
    }

    // ---- alternativeWriteMode's side arrays ------------------------------------------------
    // 5.12: no alternative write mode -- writeSortKey fills all three side arrays.
    for (uint32_t i = 0; i < gCount; ++i)
    {
        PXG_CHECK_MSG(gByX[2u * i] == uint32_t(RIGID_ID & 0xFFFFFFFFull) && gByX[2u * i + 1u] == uint32_t(RIGID_ID >> 32),
                      "contactByRigid[%u] carries the whole rigid id; got 0x%08x%08x", i, gByX[2u * i + 1u], gByX[2u * i]);
        PXG_CHECK_MSG(gTempByX[i] == uint32_t(RIGID_ID & 0xFFFFFFFFull),
                      "tempContactByRigid[%u] carries the rigid id's LOW word; got 0x%08x", i, gTempByX[i]);
        PXG_CHECK_MSG(gIdxSorted[i] == i,
                      "contactIndexSortedByRigid is seeded with the identity; slot %u holds %u", i, gIdxSorted[i]);
    }
    PXG_CHECK_MSG(gPoint[6].x == 0.0f && gNormalPen[6].x == 0.0f && gInfo[6 * 4u] == POISON,
                  "nothing beyond the six contacts may be touched");

    // ---- the clamp and the unclamped demand ------------------------------------------------
    PXG_CHECK_MSG(gNeeded == NB_STACK_ENTRIES * 16u,
                  "stackSizeNeededOnDevice must report the UNCLAMPED demand %u bytes so the "
                  "host can grow the stack; got %u (%u is the clamped count and would pin the "
                  "stack at its current size forever)",
                  NB_STACK_ENTRIES * 16u, gNeeded, DECLARED * 16u);

    // ---- 5.12: the element filter -------------------------------------------------------
    // sbHeightfieldCollision now searches the softbody's mRigidVolumeFilterPairs for
    // (PxgEncodeRigidFilterRef(rigid, heightfield cache ref), tet).  Filtering T_ABOVE alone
    // must drop exactly its contact; a filter keyed on the SOFTBODY's cache ref must not.
    {
        std::vector<PxgDeformablePairInfo> filters = {
            { p810cg::rigidFilterRef(RIGID_ID, 1u), PxgEncodeDeformableTetrahedron(SB_ID, T_ABOVE) },
            { p810cg::rigidFilterRef(RIGID_ID, 1u) + 1ull, PxgEncodeDeformableTetrahedron(SB_ID, T_BELOW) },
        };
        DevBuf bF = makeDev(dev, filters);
        std::vector<PxgSoftBody> fDesc = descArr;
        fDesc[SB_ID].mRigidVolumeFilterPairs = reinterpret_cast<decltype(fDesc[SB_ID].mRigidVolumeFilterPairs)>(size_t(bF.addr));
        fDesc[SB_ID].mNumRigidVolumeFilterPairs = uint32_t(filters.size());
        DevBuf bFDesc = makeDev(dev, fDesc);
        std::vector<uint32_t> c2 = { 0u };
        DevBuf bC2 = makeDev(dev, c2);
        std::vector<uint32_t> poisonInfo(MAXC * 4u, POISON);
        DevBuf bI2 = makeDev(dev, poisonInfo);
        SbHfCgPush p2 = p;
        p2.softbodies = bFDesc.addr;
        p2.wTotalCount = bC2.addr;
        p2.wContactInfo = bI2.addr;
        Kernel("softBodyHfContactGenBda.spv", "sb_heightfieldContactGenLaunch", {}, sizeof(SbHfCgPush))
            .dispatch(1024u, {}, p2, 1u, 1u);
        dev->flush();
        const uint32_t n2 = downloadBuffer<uint32_t>(dev, bC2.h, 1)[0];
        const auto i2 = downloadBuffer<PxgDeformablePairInfo>(dev, bI2.h, MAXC);
        PXG_CHECK_MSG(n2 == 5u, "the (heightfield, T_ABOVE) filter must drop exactly one of the six contacts; got %u", n2);
        for (uint32_t i = 0; i < n2 && i < MAXC; ++i)
            PXG_CHECK_MSG(PxgGetDeformableFeatureId(i2[i].ref1) != T_ABOVE, "T_ABOVE is filtered and must not emit");
        dev->destroyBuffer(bF.h); dev->destroyBuffer(bFDesc.h); dev->destroyBuffer(bC2.h); dev->destroyBuffer(bI2.h);
    }

    for (PxgBufferHandle h : { bBlob.h, bTetVerts.h, bTets.h, bDesc.h, bShapes.h, bCms.h, bXf.h,
                               bCd.h, bRd.h, bRemap.h, bStack.h, bPairNum.h, bNeeded.h,
                               bPoint.h, bNormalPen.h, bBary.h, bInfo.h, bByX.h, bTempByX.h,
                               bIdxSorted.h, bSorted.h, bCount.h })
        dev->destroyBuffer(h);
}


// (PhysX 5.12 removed sb_solveOutputParticleAttachmentDeltaVLaunch -- softbody-particle
// attachments -- and sb_copyOrApplySoftBodyDataDEPRECATED; the softBodySimParticleAttachment*
// and softBodySimCopy* fixtures that pinned them were deleted with the kernels.)

} // namespace
