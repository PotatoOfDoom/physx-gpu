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

// testPsMaterials.cpp -- P9.1a fixture for ps_updateMaterials
// (particlesystem.cu:5780-5820), ported as npPsMaterials.slang.
//
// ORACLE, stated honestly.  ps_updateMaterials is an `extern "C" __global__`
// kernel and the four coefficients it derives have no PX_CUDA_CALLABLE original
// to call, so this is the P8.73-P8.80 arrangement, not an engine oracle: the
// expectation is a HOST TRANSCRIPTION of the five arithmetic lines.  What carries
// it is (a) every struct offset generated from the real engine headers and pinned
// by static_assert below, so the P8.49a failure mode -- shader and fixture sharing
// one wrong number -- is closed by construction; (b) whole-region 0xAB poison
// comparison, so an EXTRA write fails as loudly as a wrong one; and (c) the
// mutation sweep.  The tranche note says this rather than presenting the sweep as
// an oracle.
//
// GEOMETRY, chosen so every trap in the module header is observable.  Two of
// these choices were forced by the sweep -- the first draft used 300 groups and
// a two-system pool, and both of those made a mutant an IDENTITY rather than a
// wrong answer; they are called out below so a later edit does not undo them.
//
//   - THREE systems in the descriptor pool, only TWO active, and the active list
//     is { 2, 0 } -- deliberately NOT a permutation of { 0 .. gridY-1 }.  The
//     first draft used { 1, 0 }, which does NOT kill the mutant that indexes the
//     pool by blockIdx.y directly: nothing else in this kernel depends on
//     blockIdx.y, so swapping which y-slice serves which system computes exactly
//     the same thing.  With a DECOY at pool index 1, that mutant instead writes
//     the decoy's records (asserted to stay entirely poisoned) and never writes
//     B's (asserted to be filled).
//   - The two active systems have DIFFERENT phase-group counts (A: 500, B: 130)
//     against a grid sized by the ACROSS-SYSTEM max (2 blocks of 256).  Block 1
//     covers i = 256..511, so it is live for A and entirely past the bound for B;
//     B's slots 130..519 must come back poisoned.  That is trap T2, and it kills
//     a mutant that uses the grid's bound instead of the per-system one.
//   - A is 500 groups, not 300, because of the block-stride mutant: with
//     numthreads(256) a halved stride of 128 reaches [0, 128*(GRID_X-1)+255] =
//     [0, 383], which COVERS 300 groups (the duplicate writes being idempotent)
//     and so survives.  At 500 groups it leaves 384..499 poisoned and dies.  The
//     invariant is asserted below rather than left as a comment.
//   - Two blocks are required at all: with one block a mutant that drops the
//     `PS_UPDATE_MATERIALS_BLOCK * groupId.x` term is an identity.
//   - FIVE materials, and phaseToMat maps group -> material through a
//     non-identity permutation, so i and mi differ for all but a few groups
//     (trap T1).  Several groups deliberately SHARE a material, which is legal
//     and must produce two identical derived slots.
//   - Every per-system scalar (mFluidRestOffset, mParticleContactDistance,
//     mInvRestDensity, mFluidSurfaceConstraintScale) differs across all three
//     systems, so a mutant that reads the wrong system's descriptor dies.
//   - All the counts are mutually distinct per the standing rule: 500 groups,
//     130 groups, 200 decoy groups, 520 slots, 5 materials, 3 systems, 2 active,
//     256 threads, 2 blocks.
//   - mFluidRestOffset and mParticleContactDistance are chosen so `rest` is not
//     1.0 and not 0.5 -- a `rest` of 1 makes cohesion1 == -2 and cohesion2 == 3
//     regardless of the sqr() spelling, which would mask traps T6/T7.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"     // real engine per-system descriptor
#include "PxsPBDMaterialCore.h"    // real engine PxsPBDMaterialData

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

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

// The push-constant blob the bridge builds for kPsUpdateMaterials { 8, 8, 8, 4 }.
struct UmPush
{
    uint64_t particleSystems;
    uint64_t activeParticleSystems;
    uint64_t pbdMaterials;
    float invTotalDt;   // dead in the kernel body; present so the blob lines up
    uint32_t _pad28;
};
static_assert(sizeof(UmPush) == 32, "kPsUpdateMaterials blob is 32 bytes");

// The engine's PxgPBDParticleMaterialDerived, as four floats.
struct Derived4 { float st, coh, c1, c2; };
static_assert(sizeof(Derived4) == 16, "derived record is 16 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;

// ---------------------------------------------------------------------------
// Layout proof -- every constant npPsMaterials.slang hard-codes, pinned against
// the real engine headers.
// ---------------------------------------------------------------------------
PXG_TEST(psMaterialsLayout)
{
    // PXG_PBD_MATERIAL_STRIDE
    PXG_CHECK_MSG(sizeof(PxsPBDMaterialData) == 60,
                  "sizeof(PxsPBDMaterialData) == 60 (got %zu)", sizeof(PxsPBDMaterialData));
    // PXG_PBD_DERIVED_STRIDE
    PXG_CHECK_MSG(sizeof(PxgPBDParticleMaterialDerived) == 16,
                  "sizeof(PxgPBDParticleMaterialDerived) == 16 (got %zu)",
                  sizeof(PxgPBDParticleMaterialDerived));

    // PXG_PBD_MAT_F_SURFACE_TENSION / _COHESION are FLOAT indices 8 and 9.
    PXG_CHECK(offsetof(PxsPBDMaterialData, surfaceTension) == 32);
    PXG_CHECK(offsetof(PxsPBDMaterialData, cohesion) == 36);

    // The derived record's field order (trap T8).
    PXG_CHECK(offsetof(PxgPBDParticleMaterialDerived, surfaceTensionDerived) == 0);
    PXG_CHECK(offsetof(PxgPBDParticleMaterialDerived, cohesion) == 4);
    PXG_CHECK(offsetof(PxgPBDParticleMaterialDerived, cohesion1) == 8);
    PXG_CHECK(offsetof(PxgPBDParticleMaterialDerived, cohesion2) == 12);

    // The mirror fields the kernel dereferences, as absolute byte offsets.
    PXG_CHECK(offsetof(PxgParticleSystem, mDerivedPBDMaterialData) == 584);
    PXG_CHECK(offsetof(PxgParticleSystem, mPhaseGroupToMaterialHandle) == 600);
    PXG_CHECK(offsetof(PxgParticleSystem, mData) == 248);
    PXG_CHECK(offsetof(PxgParticleSystemData, mFluidSurfaceConstraintScale) == 44);  // abs 292
    PXG_CHECK(offsetof(PxgParticleSystemData, mFluidRestOffset) == 56);              // abs 304
    PXG_CHECK(offsetof(PxgParticleSystemData, mInvRestDensity) == 60);               // abs 308
    PXG_CHECK(offsetof(PxgParticleSystemData, mNumPhaseToMaterials) == 80);          // abs 324
    PXG_CHECK(offsetof(PxGpuParticleSystem, mCommonData) == 88);
    PXG_CHECK(offsetof(PxGpuParticleData, mParticleContactDistance) == 16);          // abs 104
}

// The host transcription of particlesystem.cu:5803-5817.  Deliberately spelled
// term for term, including the multiply-then-divide order and the unary minus.
static Derived4 refDerived(float invRestDensity, float matSurfaceTension,
                           float fluidSurfaceConstraintScale, float matCohesion,
                           float fluidRestOffset, float particleContactDistance)
{
    const float fluidRestDistance = 2.f * fluidRestOffset;

    Derived4 d;
    d.st = invRestDensity * matSurfaceTension / fluidSurfaceConstraintScale;
    d.coh = matCohesion * particleContactDistance;

    const float rest = fluidRestDistance / particleContactDistance;
    const float sqrRest = rest * rest;
    d.c1 = -(1.0f + rest) / sqrRest;
    d.c2 = (sqrRest + rest + 1.0f) / sqrRest;
    return d;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------
PXG_TEST(psMaterials)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t NUM_MATERIALS = 5u;
    const uint32_t GROUPS_A = 500u;   // spans two blocks of 256, and 500 > 384
    const uint32_t GROUPS_B = 130u;   // entirely inside block 0
    const uint32_t GROUPS_DECOY = 200u;  // the never-active system's own count
    const uint32_t SLOTS = 520u;      // allocated per system; > GROUPS_A so a tail exists
    const uint32_t BLOCK = 256u;
    const uint32_t GRID_X = (GROUPS_A + BLOCK - 1u) / BLOCK;   // == 2
    PXG_CHECK_MSG(GRID_X == 2u, "the fixture needs two blocks (got %u)", GRID_X);

    // ---- materials: five, every field distinct and non-round ----
    std::vector<float> materials(NUM_MATERIALS * 15u, 0.0f);   // 15 floats == 60 bytes
    for (uint32_t m = 0; m < NUM_MATERIALS; ++m)
    {
        float* mat = materials.data() + m * 15u;
        // poison every field the kernel must NOT read, with values that would be
        // very visible if picked up by a wrong float index.
        for (uint32_t f = 0; f < 15u; ++f)
            mat[f] = -1000.0f - float(m) * 100.0f - float(f);
        mat[8] = 0.375f + 0.125f * float(m);   // surfaceTension @32
        mat[9] = 1.5f + 0.25f * float(m);      // cohesion      @36
    }

    // ---- phase-group -> material maps: non-identity, with deliberate sharing ----
    // A: mi = (i * 7 + 3) % 5  -- i and mi coincide only rarely.
    // B: mi = (i * 3 + 1) % 5  -- a different permutation, so a mutant that reads
    //    system A's handle array while writing B's record dies.
    // ONE definition, used for both the upload and the expectation: a fixture
    // that spells its map twice can disagree with itself.
    struct MF
    {
        static uint32_t a(uint32_t i) { return (i * 7u + 3u) % 5u; }
        static uint32_t b(uint32_t i) { return (i * 3u + 1u) % 5u; }
    };
    PXG_CHECK(NUM_MATERIALS == 5u);   // the modulus MF hard-codes

    std::vector<uint16_t> phaseToMatA(SLOTS, uint16_t(0xEEEE));
    std::vector<uint16_t> phaseToMatB(SLOTS, uint16_t(0xEEEE));
    std::vector<uint16_t> phaseToMatD(SLOTS, uint16_t(0));   // decoy: valid, never read
    for (uint32_t i = 0; i < GROUPS_A; ++i) phaseToMatA[i] = uint16_t(MF::a(i));
    for (uint32_t i = 0; i < GROUPS_B; ++i) phaseToMatB[i] = uint16_t(MF::b(i));

    // ---- derived records, poisoned ----
    std::vector<uint32_t> derivedA(SLOTS * 4u, POISON_U32);
    std::vector<uint32_t> derivedB(SLOTS * 4u, POISON_U32);
    std::vector<uint32_t> derivedD(SLOTS * 4u, POISON_U32);   // must stay ENTIRELY poisoned

    DevBuf bMat = makeDev(dev, materials);
    DevBuf bP2mA = makeDev(dev, phaseToMatA);
    DevBuf bP2mB = makeDev(dev, phaseToMatB);
    DevBuf bP2mD = makeDev(dev, phaseToMatD);
    DevBuf bDerA = makeDev(dev, derivedA);
    DevBuf bDerB = makeDev(dev, derivedB);
    DevBuf bDerD = makeDev(dev, derivedD);

    // ---- the two system descriptors ----
    // Every scalar differs between A and B, and `rest` is neither 1 nor 0.5.
    const float frOffA = 0.13f, pcdA = 0.44f, invRhoA = 0.031f, fscsA = 1.7f;
    const float frOffB = 0.29f, pcdB = 0.91f, invRhoB = 0.0175f, fscsB = 0.65f;
    const float frOffD = 0.07f, pcdD = 0.23f, invRhoD = 0.0092f, fscsD = 2.3f;

    auto mkSystem = [&](uint32_t numGroups, uint64_t p2m, uint64_t der,
                        float frOff, float pcd, float invRho, float fscs)
    {
        PxgParticleSystem ps;
        // 0xEE fill so any field the kernel must not read is a screaming value
        // rather than a plausible zero.
        std::memset(&ps, 0xEE, sizeof(ps));
        ps.mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(p2m));
        ps.mDerivedPBDMaterialData = reinterpret_cast<PxgPBDParticleMaterialDerived*>(size_t(der));
        ps.mData.mNumPhaseToMaterials = numGroups;
        ps.mData.mFluidRestOffset = frOff;
        ps.mData.mInvRestDensity = invRho;
        ps.mData.mFluidSurfaceConstraintScale = fscs;
        ps.mCommonData.mParticleContactDistance = pcd;
        return ps;
    };

    // The pool holds THREE systems and only TWO are active, with the active list
    // deliberately NOT a permutation of {0..gridY-1}: active = { 2, 0 }.  A merely
    // permuted list (e.g. {1,0}) would NOT kill the mutant that indexes the pool
    // by blockIdx.y directly, because nothing else in the kernel depends on
    // blockIdx.y -- swapping which y-slice serves which system is an identity.
    // With a decoy at pool index 1 the mutant instead writes the DECOY's records
    // (which must stay poisoned) and never writes B's (which must be filled).
    std::vector<PxgParticleSystem> descArr;
    descArr.push_back(mkSystem(GROUPS_A, bP2mA.addr, bDerA.addr, frOffA, pcdA, invRhoA, fscsA));
    descArr.push_back(mkSystem(GROUPS_DECOY, bP2mD.addr, bDerD.addr, frOffD, pcdD, invRhoD, fscsD));
    descArr.push_back(mkSystem(GROUPS_B, bP2mB.addr, bDerB.addr, frOffB, pcdB, invRhoB, fscsB));
    std::vector<uint32_t> active = { 2u, 0u };

    DevBuf bDesc = makeDev(dev, descArr);
    DevBuf bActive = makeDev(dev, active);

    UmPush push = {};
    push.particleSystems = bDesc.addr;
    push.activeParticleSystems = bActive.addr;
    push.pbdMaterials = bMat.addr;
    push.invTotalDt = 12345.0f;   // dead: any read of it would be very visible
    push._pad28 = 0u;

    Kernel k("npPsMaterials.spv", "ps_updateMaterials", {}, sizeof(UmPush));
    k.dispatch(GRID_X, {}, push, /*gridY*/ uint32_t(active.size()), /*gridZ*/ 1u);
    dev->flush();

    // ---- expectation ----
    std::vector<uint32_t> refA(SLOTS * 4u, POISON_U32);
    std::vector<uint32_t> refB(SLOTS * 4u, POISON_U32);
    auto fill = [&](std::vector<uint32_t>& ref, uint32_t numGroups,
                    uint32_t (*matFor)(uint32_t), float frOff, float pcd,
                    float invRho, float fscs)
    {
        for (uint32_t i = 0; i < numGroups; ++i)
        {
            const uint32_t mi = matFor(i);
            const float* mat = materials.data() + mi * 15u;
            const Derived4 d = refDerived(invRho, mat[8], fscs, mat[9], frOff, pcd);
            ref[i * 4u + 0u] = f2u(d.st);
            ref[i * 4u + 1u] = f2u(d.coh);
            ref[i * 4u + 2u] = f2u(d.c1);
            ref[i * 4u + 3u] = f2u(d.c2);
        }
    };
    fill(refA, GROUPS_A, &MF::a, frOffA, pcdA, invRhoA, fscsA);
    fill(refB, GROUPS_B, &MF::b, frOffB, pcdB, invRhoB, fscsB);

    const auto gotA = downloadBuffer<uint32_t>(dev, bDerA.h, SLOTS * 4u);
    const auto gotB = downloadBuffer<uint32_t>(dev, bDerB.h, SLOTS * 4u);

    size_t mm = 0;
    auto cmp = [&](const std::vector<uint32_t>& ref, const std::vector<uint32_t>& got,
                   const char* what)
    {
        for (uint32_t w = 0; w < SLOTS * 4u; ++w)
        {
            if (ref[w] != got[w])
            {
                if (mm < 12)
                {
                    float rf, gf;
                    std::memcpy(&rf, &ref[w], 4); std::memcpy(&gf, &got[w], 4);
                    std::printf("  %s MM group %u field %u: ref 0x%08x (%g) got 0x%08x (%g)\n",
                                what, w / 4u, w % 4u, ref[w], rf, got[w], gf);
                }
                ++mm;
            }
        }
    };
    const auto gotD = downloadBuffer<uint32_t>(dev, bDerD.h, SLOTS * 4u);
    cmp(refA, gotA, "A");
    cmp(refB, gotB, "B");
    cmp(derivedD, gotD, "decoy");   // the never-active system: all poison

    PXG_CHECK_MSG(mm == 0, "ps_updateMaterials bit-exact over both systems (%zu mismatches)", mm);

    // Explicit coverage assertions, so a fixture that silently stopped exercising
    // a trap fails rather than passing vacuously.
    PXG_CHECK_MSG(gotB[GROUPS_B * 4u] == POISON_U32,
                  "B's first out-of-bound slot must stay poisoned (trap T2)");
    PXG_CHECK_MSG(gotA[(GROUPS_A - 1u) * 4u] != POISON_U32,
                  "A's last in-bound slot must be written (block 1 is live)");
    PXG_CHECK_MSG(gotA[GROUPS_A * 4u] == POISON_U32,
                  "A's first out-of-bound slot must stay poisoned");
    // A block-stride mutant of 128 with numthreads(256) covers [0, 128*(G-1)+255].
    // For this to leave a HOLE rather than be an idempotent identity, the live
    // group count must exceed that reach.
    PXG_CHECK_MSG(GROUPS_A > 128u * (GRID_X - 1u) + 255u,
                  "GROUPS_A (%u) must exceed a halved block stride's reach (%u)",
                  GROUPS_A, 128u * (GRID_X - 1u) + 255u);
    {
        // trap T1: at least one group whose index differs from its material, and
        // at least one pair of distinct groups sharing a material.
        bool sawDiffer = false, sawShare = false;
        for (uint32_t i = 0; i < GROUPS_A && !(sawDiffer && sawShare); ++i)
        {
            if (MF::a(i) != i) sawDiffer = true;
            for (uint32_t j = i + 1u; j < GROUPS_A; ++j)
                if (MF::a(i) == MF::a(j)) { sawShare = true; break; }
        }
        PXG_CHECK_MSG(sawDiffer && sawShare,
                      "the group->material map must be non-identity AND share (trap T1)");
    }
    {
        // traps T6/T7 need a `rest` that is neither 1 nor 0.5.
        const float restA = (2.f * frOffA) / pcdA;
        const float restB = (2.f * frOffB) / pcdB;
        PXG_CHECK_MSG(restA != 1.0f && restA != 0.5f && restB != 1.0f && restB != 0.5f,
                      "rest must not be 1 or 0.5 (A %g, B %g)", restA, restB);
    }

    dev->destroyBuffer(bMat.h); dev->destroyBuffer(bP2mA.h); dev->destroyBuffer(bP2mB.h);
    dev->destroyBuffer(bP2mD.h); dev->destroyBuffer(bDerA.h); dev->destroyBuffer(bDerB.h);
    dev->destroyBuffer(bDerD.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

} // namespace
