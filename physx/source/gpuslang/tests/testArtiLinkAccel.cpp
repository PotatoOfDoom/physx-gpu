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

// P8.92a: npArtiLinkAccel.slang — the bias-force sweep and the parallel-axis
// theorem.
//
// ===========================================================================
// THE ORACLE, AND IT IS A STRONG ONE
// ===========================================================================
//
// Everything both bodies are built from is PX_CUDA_CALLABLE — `PxQuat::rotate`
// and `rotateInv`, `PxVec3::multiply`, `PxMat33::getTranspose` and its two
// `operator*` forms, `FeatherstoneArticulation::translateSpatialVector`, and the
// engine's own `loadSpatialVector` / `storeSpatialVector`.  The expectation below
// is therefore the engine's arithmetic on the host, compared bit for bit; only
// the loop's STRUCTURE is transcribed, which is where all seven traps live.
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
//   * **three different topologies in the same warp** — a chain, a binary tree
//     and a star.  The tree is the one that matters: a node with TWO children
//     receives two accumulations, so a port that assigns where it should add is
//     wrong at an interior link rather than only at the root;
//   * **link counts distinct per lane and distinct from the lane index**, the
//     standing rule;
//   * a link with `invMass == 0` and another with one zero inertia COMPONENT, so
//     both zero guards fire on their own;
//   * a lane whose quaternion is the IDENTITY beside lanes with a general
//     rotation, so the body-frame round trip is visible where it matters and its
//     absence is visible where it does not;
//   * `mMotionAcceleration` and `mMotionAccelerationInternal` both non-zero and
//     DIFFERENT, so their sum cannot be mistaken for either;
//   * `mBiasForce` pre-seeded NON-ZERO, so `+=` is distinguishable from `=` and
//     the root's final zeroing is a real change rather than a no-op;
//   * the whole link-block region compared byte for byte against a 0xAB poison
//     image, so an extra write fails as loudly as a wrong one.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxMat33.h"
#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "pxgArtiRefHelpers.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxMat33;
using physx::PxQuat;
using physx::PxReal;
using physx::PxU32;
using physx::PxVec3;
using physx::loadPxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::Dy::FeatherstoneArticulation;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiLinkAccelLoader(const char* file) { return loadShaderBlob(file); }

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// ===========================================================================
// asmTranslateInertiaMass
// ===========================================================================

PXG_TEST(artiTranslateInertiaMass)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiLinkAccel.spv", "artiTranslateInertiaMass", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiLinkAccelLoader);

    const uint32_t N = 257;   // prime, coprime with every stripe below
    auto val = [](uint32_t i, uint32_t k) {
        return (float((i * 41u + k * 7u) % 733u) + 1.0f) * 0.017578125f - 6.5f;
    };

    std::vector<float> in(size_t(N) * 13);
    size_t zeroMass = 0, zeroOffset = 0, symmetric = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float* c = in.data() + size_t(i) * 13;
        for (uint32_t k = 0; k < 9; ++k)
            c[k] = val(i, k);

        // a SYMMETRIC inertia every eleventh case beside the general one: the
        // result is symmetric either way, so a transposition in `s^T * s` is only
        // visible against a non-symmetric `inertia`
        if (i % 11 == 3)
        {
            c[3] = c[1];
            c[6] = c[2];
            c[7] = c[5];
            ++symmetric;
        }

        c[9] = val(i, 10) * 2.0f + 5.0f;   // mass, never zero by construction
        c[10] = val(i, 11);
        c[11] = val(i, 12);
        c[12] = val(i, 13);

        if (i % 17 == 5)
        {
            // zero mass: the whole shift term vanishes and the result must be the
            // inertia unchanged, bit for bit
            c[9] = 0.0f;
            ++zeroMass;
        }
        if (i % 19 == 7)
        {
            // zero offset: `s` is the zero matrix, same conclusion by a different
            // route
            c[10] = c[11] = c[12] = 0.0f;
            ++zeroOffset;
        }
    }
    std::printf("[ info ] tim: %zu symmetric, %zu zero-mass, %zu zero-offset of %u\n", symmetric,
                zeroMass, zeroOffset, N);
    PXG_CHECK(symmetric > 15 && zeroMass > 10 && zeroOffset > 10);

    std::vector<uint32_t> poison(size_t(N) * 9);
    for (size_t k = 0; k < poison.size(); ++k)
        poison[k] = 0xAB000000u | uint32_t(k & 0xFFFFFFu);

    PxgDevicePtr dIn = 0, dOut = 0;
    shim.memAlloc(&dIn, in.size() * sizeof(float));
    shim.memAlloc(&dOut, poison.size() * 4);
    if (!dIn || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dIn, in.data(), in.size() * sizeof(float));
    shim.memcpyHtoD(dOut, poison.data(), poison.size() * 4);

    struct Params
    {
        uint64_t inFloats, outFloats;
        uint32_t nbCases, pad0, pad1, pad2;
    } prm = { uint64_t(dIn), uint64_t(dOut), N, 0, 0, 0 };
    PXG_CHECK_MSG(sizeof(prm) == 32, "push-constant blob is %zu bytes", sizeof(prm));

    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiTranslateInertiaMass");
    shim.streamSynchronize();

    std::vector<float> got(poison.size());
    shim.memcpyDtoH(got.data(), dOut, got.size() * 4);

    size_t bad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* c = in.data() + size_t(i) * 13;
        const PxMat33 inertia(PxVec3(c[0], c[1], c[2]), PxVec3(c[3], c[4], c[5]),
                              PxVec3(c[6], c[7], c[8]));
        const PxMat33 r = refTranslateInertiaMass(inertia, c[9], PxVec3(c[10], c[11], c[12]));

        for (uint32_t k = 0; k < 9; ++k)
        {
            const float w = (&r.column0.x)[k];
            const float g = got[size_t(i) * 9 + k];
            if (f2u(w) == f2u(g))
                continue;
            if (bad < 12)
                std::printf("MM tim case %u slot %u  ref %.9g (%08x)  gpu %.9g (%08x)\n", i, k, w,
                            f2u(w), g, f2u(g));
            ++bad;
        }
    }
    PXG_CHECK_MSG(bad == 0, "the parallel-axis theorem is bit-identical to the engine's");

    shim.memFree(dIn);
    shim.memFree(dOut);
}

// ===========================================================================
// computeJointTransmittedFrictionForce
// ===========================================================================

const uint32_t FR_LANES = 23;        // live lanes; nine more left poisoned
const uint32_t FR_MAX_LINKS = 8;     // > every count, and not a lane index in use

// counts 2..7, distinct from the lane index for every live lane (checked below)
static uint32_t frNumLinks(uint32_t k) { return 2u + ((k * 5u + 1u) % 6u); }

// three topologies in one warp -- see the header
static uint32_t frParent(uint32_t k, uint32_t i)
{
    if (i == 0)
        return 0;
    switch (k % 3u)
    {
    case 0: return i - 1u;              // chain
    case 1: return (i - 1u) / 2u;       // binary tree: link 1 and 2 share parent 0
    default: return 0u;                 // star
    }
}

static void frFill(physx::PxgArticulationBlockLinkData* lk, uint32_t k, uint32_t numLinks,
                   uint32_t lane)
{
    auto v = [&](uint32_t i, uint32_t f) {
        return (float((k * 19u + i * 11u + f) % 331u) + 1.0f) * 0.0234375f - 3.75f;
    };

    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& b = lk[i];
        b.mParents[lane] = frParent(k, i);
        b.mRw_x[lane] = v(i, 0);
        b.mRw_y[lane] = v(i, 1);
        b.mRw_z[lane] = v(i, 2);

        // the inverse inertia / inverse mass, with the two zero cases placed on
        // DIFFERENT links so each guard fires on its own
        float invIx = v(i, 3) * 0.25f + 2.0f;
        float invIy = v(i, 4) * 0.25f + 3.0f;
        float invIz = v(i, 5) * 0.25f + 4.0f;
        float invM = v(i, 6) * 0.125f + 1.5f;
        if (i == 1 && (k % 5u) == 2u)
            invM = 0.0f;                 // a static link
        if (i == 2 && (k % 7u) == 3u)
            invIy = 0.0f;                // one zero inertia component
        b.mInvInertiaXYZ_invMassW[lane] = make_float4(invIx, invIy, invIz, invM);

        // a general rotation, except one lane where it is the identity
        PxQuat q(v(i, 7), v(i, 8), v(i, 9), v(i, 10) + 2.0f);
        if ((k % 9u) == 4u)
            q = PxQuat(physx::PxIdentity);
        else
            q = q.getNormalized();
        b.mAccumulatedPose.q[lane] = make_float4(q.x, q.y, q.z, q.w);

        // the two acceleration halves, non-zero and DIFFERENT
        storeSpatialVector(b.mMotionAcceleration,
                           UnAlignedSpatialVector(PxVec3(v(i, 11), v(i, 12), v(i, 13)),
                                                  PxVec3(v(i, 14) * 3.0f, v(i, 15) * 3.0f,
                                                         v(i, 16) * 3.0f)),
                           lane);
        storeSpatialVector(b.mMotionAccelerationInternal,
                           UnAlignedSpatialVector(PxVec3(v(i, 17) * 0.5f, v(i, 18) * 0.5f,
                                                         v(i, 19) * 0.5f),
                                                  PxVec3(v(i, 20), v(i, 21), v(i, 22))),
                           lane);

        // seeded NON-ZERO, so "+=" is distinguishable from "=" and the root's
        // final zeroing is a real change
        storeSpatialVector(b.mBiasForce,
                           UnAlignedSpatialVector(PxVec3(v(i, 23) * 7.0f, v(i, 24) * 7.0f,
                                                         v(i, 25) * 7.0f),
                                                  PxVec3(v(i, 26) * 11.0f, v(i, 27) * 11.0f,
                                                         v(i, 28) * 11.0f)),
                           lane);
    }
}

// The mirror: the CUDA's structure, the engine's arithmetic.
static void frMirror(physx::PxgArticulationBlockLinkData* lk, uint32_t numLinks, uint32_t lane)
{
    for (uint32_t linkID = numLinks - 1; linkID > 0; --linkID)
    {
        physx::PxgArticulationBlockLinkData& link = lk[linkID];
        const PxU32 parent = link.mParents[lane];
        const UnAlignedSpatialVector biasForce = loadSpatialVector(lk[parent].mBiasForce, lane);

        const float rwx = link.mRw_x[lane];
        const float rwy = link.mRw_y[lane];
        const float rwz = link.mRw_z[lane];

        const float4 iim = link.mInvInertiaXYZ_invMassW[lane];
        const float4 q = link.mAccumulatedPose.q[lane];
        const PxQuat rot(q.x, q.y, q.z, q.w);
        const PxReal m = iim.w == 0.f ? 0.f : 1.f / iim.w;
        const PxVec3 inertia(iim.x == 0.f ? 0.f : 1.f / iim.x, iim.y == 0.f ? 0.f : 1.f / iim.y,
                             iim.z == 0.f ? 0.f : 1.f / iim.z);

        const UnAlignedSpatialVector motionAcceleration =
            loadSpatialVector(link.mMotionAcceleration, lane)
            + loadSpatialVector(link.mMotionAccelerationInternal, lane);
        UnAlignedSpatialVector Ia = loadSpatialVector(link.mBiasForce, lane);
        Ia.bottom += rot.rotate(rot.rotateInv(motionAcceleration.top).multiply(inertia));
        Ia.top += motionAcceleration.bottom * m;

        storeSpatialVector(link.mBiasForce, Ia, lane);

        storeSpatialVector(lk[parent].mBiasForce,
                           biasForce + FeatherstoneArticulation::translateSpatialVector(
                                           PxVec3(rwx, rwy, rwz), Ia),
                           lane);
    }

    storeSpatialVector(lk[0].mBiasForce, UnAlignedSpatialVector::Zero(), lane);
}

PXG_TEST(artiJointFriction)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiLinkAccel.spv", "artiJointFriction", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiLinkAccelLoader);

    std::vector<physx::PxgArticulationBlockLinkData> lkImg(FR_MAX_LINKS);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));

    std::vector<uint32_t> numLinks(32);
    uint32_t topo[3] = { 0, 0, 0 }, staticLinks = 0, zeroInertia = 0, identityQuat = 0;
    for (uint32_t lane = 0; lane < 32; ++lane)
        numLinks[lane] = (lane < FR_LANES) ? frNumLinks(lane) : 0u;
    for (uint32_t lane = 0; lane < FR_LANES; ++lane)
    {
        PXG_CHECK_MSG(numLinks[lane] <= FR_MAX_LINKS, "lane %u has %u links", lane,
                      numLinks[lane]);
        PXG_CHECK_MSG(numLinks[lane] != lane, "lane %u's link count equals its index", lane);
        frFill(lkImg.data(), lane, numLinks[lane], lane);
        topo[lane % 3u] += 1u;
        if ((lane % 5u) == 2u && numLinks[lane] > 1) ++staticLinks;
        if ((lane % 7u) == 3u && numLinks[lane] > 2) ++zeroInertia;
        if ((lane % 9u) == 4u) ++identityQuat;
    }
    std::printf("[ info ] fr: topo chain/tree/star %u/%u/%u, %u static, %u zero-inertia, "
                "%u identity-quat\n",
                topo[0], topo[1], topo[2], staticLinks, zeroInertia, identityQuat);
    PXG_CHECK(topo[0] > 4 && topo[1] > 4 && topo[2] > 4);
    PXG_CHECK(staticLinks > 1 && zeroInertia > 1 && identityQuat > 1);
    // some tree lane must actually have a node with two children: with the
    // binary-tree parent map that needs three links, so links 1 and 2 both hang
    // off link 0 and it receives TWO accumulations
    uint32_t forkedTrees = 0;
    for (uint32_t lane = 0; lane < FR_LANES; ++lane)
        if ((lane % 3u) == 1u && numLinks[lane] >= 3)
            ++forkedTrees;
    PXG_CHECK_MSG(forkedTrees > 2, "only %u tree lanes have a forked node", forkedTrees);

    PxgDevicePtr dLk = 0, dCounts = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dCounts, numLinks.size() * sizeof(uint32_t));
    if (!dLk || !dCounts)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dCounts, numLinks.data(), numLinks.size() * sizeof(uint32_t));

    struct Params
    {
        uint64_t linkBlocks, numLinksPer;
        uint32_t nbLanes, pad0, pad1, pad2;
    } prm = { uint64_t(dLk), uint64_t(dCounts), FR_LANES, 0, 0, 0 };
    PXG_CHECK_MSG(sizeof(prm) == 32, "push-constant blob is %zu bytes", sizeof(prm));

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiJointFriction");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockLinkData> want = lkImg;
    for (uint32_t lane = 0; lane < FR_LANES; ++lane)
        frMirror(want.data(), numLinks[lane], lane);

    std::vector<physx::PxgArticulationBlockLinkData> got(lkImg.size());
    shim.memcpyDtoH(got.data(), dLk, got.size() * sizeof(physx::PxgArticulationBlockLinkData));

    const uint32_t* w = reinterpret_cast<const uint32_t*>(want.data());
    const uint32_t* g = reinterpret_cast<const uint32_t*>(got.data());
    const size_t words = want.size() * sizeof(physx::PxgArticulationBlockLinkData) / 4;
    size_t bad = 0, signedZeros = 0;
    for (size_t k = 0; k < words; ++k)
    {
        if (w[k] == g[k])
            continue;
        if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
        {
            ++signedZeros;
            continue;
        }
        if (bad < 12)
            std::printf("MM fr word %zu  ref %08x  gpu %08x\n", k, w[k], g[k]);
        ++bad;
    }
    std::printf("[ info ] fr: signed zeros forgiven %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "the bias-force sweep matches the engine-built mirror");

    shim.memFree(dLk);
    shim.memFree(dCounts);
}


// ===========================================================================
// P8.92b: computeLinkAcceleration
// ===========================================================================
//
// The mirror below follows the CUDA's structure — TWO separate down-sweep loops,
// written out — deliberately, because the shader shares one body between them
// behind an `internal` flag.  A mirror that copied the shader's refactor would
// agree with it whichever of the six differences went wrong.
//
// Everything inside is the engine's: `Dy::SpatialMatrix::operator*`,
// `PxMat33::operator*` on a vector, `PxVec3::cross` and `magnitude`,
// `translateSpatialVector`, `PxMat33::getInverse`, `refTranslateInertiaMass`, and
// the engine's own load/store helpers.
//
// The fixture's cases, beyond the standing poison discipline:
//
//   * fixed-base and free-base articulations in the same warp — for a fixed base
//     the whole momentum tail is skipped and three root fields are zeroed
//     instead, so the two paths write DIFFERENT sets of fields;
//   * link counts 1..6, distinct from the lane index.  A count of 1 is the
//     degenerate articulation where both down-sweeps have no iterations at all;
//   * a lane with zero mass and zero isolated inertia, so `postMomentum.bottom`
//     is exactly zero and the `angRatio` guard fires — and the compound inertia
//     is singular, so `getInverse` returns the identity;
//   * `dt` is not 1, so every `x * dt` is observable;
//   * a per-lane slice of the flat `motionVelocities` array, each with a poisoned
//     tail, so a write past a lane's own link count lands where it can be seen.

const uint32_t LA_LANES = 19;
const uint32_t LA_MAX_LINKS = 6;
const uint32_t LA_MAX_DOFS = 11;      // distinct from MAX_LINKS and from every count
const uint32_t LA_VEL_SLOTS = 9;      // > LA_MAX_LINKS: an overrun has room to show
const float LA_DT = 0.03125f;         // not 1

static uint32_t laNumLinks(uint32_t k) { return 1u + ((k * 5u + 2u) % 6u); }
static bool laFixBase(uint32_t k) { return (k % 4u) == 1u; }
static bool laMassless(uint32_t k) { return (k % 11u) == 6u; }
static uint8_t laDofs(uint32_t k, uint32_t i) { return uint8_t(1u + ((k + i) % 3u)); }

static uint32_t laTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        t += laDofs(k, i);
    return t;
}

static void laFill(physx::PxgArticulationBlockLinkData* lk, physx::PxgArticulationBlockDofData* df,
                   physx::PxgArticulationBlockData& ab, uint32_t k, uint32_t numLinks, uint32_t lane)
{
    auto v = [&](uint32_t i, uint32_t f) {
        return (float((k * 23u + i * 13u + f) % 397u) + 1.0f) * 0.01953125f - 3.5f;
    };
    const bool massless = laMassless(k);

    ab.mNumLinks[lane] = numLinks;
    ab.mFlags[lane] = laFixBase(k) ? PxU32(physx::PxArticulationFlag::eFIX_BASE
                                           | physx::PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES)
                                   : PxU32(physx::PxArticulationFlag::eDISABLE_SELF_COLLISION);
    ab.mCOM_TotalInvMassW[lane] = make_float4(v(0, 90), v(0, 91), v(0, 92), v(0, 93) * 0.1f + 0.4f);

    physx::Dy::SpatialMatrix invInertia(
        PxMat33(PxVec3(v(0, 1), v(0, 2), v(0, 3)), PxVec3(v(0, 4), v(0, 5), v(0, 6)),
                PxVec3(v(0, 7), v(0, 8), v(0, 9))),
        PxMat33(PxVec3(v(0, 10), v(0, 11), v(0, 12)), PxVec3(v(0, 13), v(0, 14), v(0, 15)),
                PxVec3(v(0, 16), v(0, 17), v(0, 18))),
        PxMat33(PxVec3(v(0, 19), v(0, 20), v(0, 21)), PxVec3(v(0, 22), v(0, 23), v(0, 24)),
                PxVec3(v(0, 25), v(0, 26), v(0, 27))));
    storeSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, invInertia);

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& b = lk[i];
        const uint8_t dofs = (i == 0) ? uint8_t(0) : laDofs(k, i);
        b.mParents[lane] = (i == 0) ? 0u : (i - 1u);
        b.mDofs[lane] = dofs;
        b.mJointOffset[lane] = cursor;
        b.mRw_x[lane] = v(i, 30);
        b.mRw_y[lane] = v(i, 31);
        b.mRw_z[lane] = v(i, 32);
        b.mMass[lane] = massless ? 0.f : (v(i, 33) * 0.5f + 3.0f);

        PxMat33 iso(PxVec3(v(i, 34), v(i, 35), v(i, 36)), PxVec3(v(i, 37), v(i, 38), v(i, 39)),
                    PxVec3(v(i, 40), v(i, 41), v(i, 42)));
        if (massless)
            iso = PxMat33(physx::PxZero);
        storePxMat33(b.mIsolatedInertia, lane, iso);

        b.mAccumulatedPose.p[lane] = make_float4(v(i, 43), v(i, 44), v(i, 45), 0.f);
        b.mAccumulatedPose.q[lane] = make_float4(0.f, 0.f, 0.f, 1.f);

        storeSpatialVector(b.mMotionVelocity,
                           UnAlignedSpatialVector(PxVec3(v(i, 46), v(i, 47), v(i, 48)),
                                                  PxVec3(v(i, 49) * 2.f, v(i, 50) * 2.f,
                                                         v(i, 51) * 2.f)),
                           lane);
        storeSpatialVector(b.mCoriolis,
                           UnAlignedSpatialVector(PxVec3(v(i, 52), v(i, 53), v(i, 54)),
                                                  PxVec3(v(i, 55) * 3.f, v(i, 56) * 3.f,
                                                         v(i, 57) * 3.f)),
                           lane);
        storeSpatialVector(b.mZAVector,
                           UnAlignedSpatialVector(PxVec3(v(i, 58), v(i, 59), v(i, 60)),
                                                  PxVec3(v(i, 61), v(i, 62), v(i, 63))),
                           lane);
        storeSpatialVector(b.mZAIntVector,
                           UnAlignedSpatialVector(PxVec3(v(i, 64) * 0.25f, v(i, 65) * 0.25f,
                                                         v(i, 66) * 0.25f),
                                                  PxVec3(v(i, 67), v(i, 68), v(i, 69))),
                           lane);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& d = df[cursor + dd];
            const uint32_t o = 70 + dd * 20;
            storeSpatialVector(d.mIsW,
                               UnAlignedSpatialVector(PxVec3(v(i, o + 0), v(i, o + 1), v(i, o + 2)),
                                                      PxVec3(v(i, o + 3) * 5.f, v(i, o + 4) * 5.f,
                                                             v(i, o + 5) * 5.f)),
                               lane);
            storeSpatialVector(d.mWorldMotionMatrix,
                               UnAlignedSpatialVector(PxVec3(v(i, o + 6), v(i, o + 7), v(i, o + 8)),
                                                      PxVec3(v(i, o + 9) * 7.f, v(i, o + 10) * 7.f,
                                                             v(i, o + 11) * 7.f)),
                               lane);
            // the two per-dof terms, DIFFERENT so neither can stand in for the other
            d.mQstZ[lane] = v(i, o + 12);
            d.mQstZIcInternal[lane] = v(i, o + 13) * 4.0f + 1.0f;
            d.mJointVelocities[lane] = v(i, o + 14);
            // seeded NON-ZERO: pass one ASSIGNS this, pass two ADDS to it
            d.mJointAccel[lane] = v(i, o + 15) * 9.0f;
            d.mJointUnconstrainedVelocities[lane] = v(i, o + 16) * 13.0f;
            d.mInvStIsT_x[lane] = v(i, o + 17);
            d.mInvStIsT_y[lane] = v(i, o + 18);
            d.mInvStIsT_z[lane] = v(i, o + 19);
        }
        cursor += dofs;
    }
}

bool gLaCompoundFoldTwin = false;

// The CUDA's structure, the engine's arithmetic.  `flatVel` is this lane's slice
// of the flat motion-velocity array.
static void laMirror(physx::PxgArticulationBlockLinkData* lk,
                     physx::PxgArticulationBlockDofData* df, physx::PxgArticulationBlockData& ab,
                     UnAlignedSpatialVector* flatVel, uint32_t numLinks, float dt, uint32_t lane)
{
    const bool fixBase = (ab.mFlags[lane] & PxU32(physx::PxArticulationFlag::eFIX_BASE)) != 0;
    const float4 COM_invMassW = ab.mCOM_TotalInvMassW[lane];
    const PxVec3 COM(COM_invMassW.x, COM_invMassW.y, COM_invMassW.z);

    UnAlignedSpatialVector preMomentum(PxVec3(0.f), PxVec3(0.f));
    UnAlignedSpatialVector postMomentum(PxVec3(0.f), PxVec3(0.f));
    PxMat33 compoundInertia(physx::PxZero);

    if (!fixBase)
    {
        physx::Dy::SpatialMatrix invInertia;
        loadSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, invInertia);
        const UnAlignedSpatialVector za = loadSpatialVector(lk[0].mZAVector, lane);
        const UnAlignedSpatialVector accel = -(invInertia * za);
        storeSpatialVector(lk[0].mMotionAcceleration, accel, lane);
        const UnAlignedSpatialVector motionVel =
            loadSpatialVector(lk[0].mMotionVelocity, lane) + accel * dt;
        const PxReal mass = lk[0].mMass[lane];
        preMomentum.top = motionVel.bottom * mass;
        storeSpatialVector(lk[0].mMotionVelocity, motionVel, lane);
    }
    else
    {
        storeSpatialVector(lk[0].mMotionAcceleration, UnAlignedSpatialVector::Zero(), lane);
        storeSpatialVector(lk[0].mMotionAccelerationInternal, UnAlignedSpatialVector::Zero(), lane);
        storeSpatialVector(lk[0].mMotionVelocity, UnAlignedSpatialVector::Zero(), lane);
        flatVel[0] = UnAlignedSpatialVector::Zero();
    }

    for (uint32_t linkID = 1; linkID < numLinks; ++linkID)
    {
        physx::PxgArticulationBlockLinkData& tLink = lk[linkID];
        physx::PxgArticulationBlockDofData* tDofs = df + tLink.mJointOffset[lane];
        const PxU32 parent = tLink.mParents[lane];
        const PxVec3 rw(tLink.mRw_x[lane], tLink.mRw_y[lane], tLink.mRw_z[lane]);

        const UnAlignedSpatialVector linkMotionVel = loadSpatialVector(tLink.mMotionVelocity, lane);
        const UnAlignedSpatialVector pMotionAcceleration =
            FeatherstoneArticulation::translateSpatialVector(
                -rw, loadSpatialVector(lk[parent].mMotionAcceleration, lane));

        UnAlignedSpatialVector motionAcceleration = pMotionAcceleration;
        const PxU32 dofs = tLink.mDofs[lane];

        PxReal tJAccel[3] = { 0.f, 0.f, 0.f };
        for (PxU32 ind = 0; ind < 3; ++ind)
            if (ind < dofs)
                tJAccel[ind] = tDofs[ind].mQstZ[lane]
                             - loadSpatialVector(tDofs[ind].mIsW, lane)
                                   .innerProduct(pMotionAcceleration);

        for (PxU32 ind = 0; ind < 3; ++ind)
        {
            if (ind < dofs)
            {
                const PxReal jAccel = tDofs[ind].mInvStIsT_x[lane] * tJAccel[0]
                                    + tDofs[ind].mInvStIsT_y[lane] * tJAccel[1]
                                    + tDofs[ind].mInvStIsT_z[lane] * tJAccel[2];
                tDofs[ind].mJointAccel[lane] = jAccel;
                const PxReal jVel = tDofs[ind].mJointVelocities[lane] + jAccel * dt;
                tDofs[ind].mJointVelocities[lane] = jVel;
                tDofs[ind].mJointUnconstrainedVelocities[lane] = jVel;
                motionAcceleration +=
                    loadSpatialVector(tDofs[ind].mWorldMotionMatrix, lane) * jAccel;
            }
        }

        storeSpatialVector(tLink.mMotionAcceleration, motionAcceleration, lane);
        const UnAlignedSpatialVector motionVel = linkMotionVel + motionAcceleration * dt;
        const PxReal mass = tLink.mMass[lane];
        preMomentum.top += motionVel.bottom * mass;
        flatVel[linkID] = motionVel;
        storeSpatialVector(tLink.mMotionVelocity, motionVel, lane);
    }

    const PxVec3 preRootVel = preMomentum.top * COM_invMassW.w;

    if (!fixBase)
    {
        physx::Dy::SpatialMatrix invInertia;
        loadSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, invInertia);
        const UnAlignedSpatialVector zaInt = loadSpatialVector(lk[0].mZAIntVector, lane);
        const UnAlignedSpatialVector accel = -(invInertia * zaInt);
        storeSpatialVector(lk[0].mMotionAccelerationInternal, accel, lane);
        addSpatialVector(lk[0].mMotionAcceleration, accel, lane);
        UnAlignedSpatialVector motionVel = loadSpatialVector(lk[0].mMotionVelocity, lane);

        PxMat33 inertia;
        loadPxMat33(lk[0].mIsolatedInertia, lane, inertia);
        const PxReal mass = lk[0].mMass[lane];
        const PxVec3 pos = loadPxVec3(lk[0].mAccumulatedPose.p, lane);
        const PxVec3 offset = (pos - COM);

        preMomentum.bottom = inertia * motionVel.top
                           + offset.cross(motionVel.bottom - preRootVel) * mass;
        if (gLaCompoundFoldTwin)   // P9.86f twin: the port's folded `0 + T` (root term taken as T)
            compoundInertia = refTranslateInertiaMass(inertia, mass, offset);
        else
            compoundInertia += refTranslateInertiaMass(inertia, mass, offset);
        motionVel += accel * dt;
        postMomentum.top = motionVel.bottom * mass;
        storeSpatialVector(lk[0].mMotionVelocity, motionVel, lane);
        flatVel[0] = motionVel;
    }

    for (uint32_t linkID = 1; linkID < numLinks; ++linkID)
    {
        physx::PxgArticulationBlockLinkData& tLink = lk[linkID];
        physx::PxgArticulationBlockDofData* tDofs = df + tLink.mJointOffset[lane];
        const PxU32 parent = tLink.mParents[lane];
        const PxVec3 rw(tLink.mRw_x[lane], tLink.mRw_y[lane], tLink.mRw_z[lane]);

        const UnAlignedSpatialVector pMotionAcceleration =
            FeatherstoneArticulation::translateSpatialVector(
                -rw, loadSpatialVector(lk[parent].mMotionAccelerationInternal, lane));

        UnAlignedSpatialVector motionAcceleration =
            pMotionAcceleration + loadSpatialVector(tLink.mCoriolis, lane);
        const PxU32 dofs = tLink.mDofs[lane];

        PxReal tJAccel[3] = { 0.f, 0.f, 0.f };
        for (PxU32 ind = 0; ind < 3; ++ind)
            if (ind < dofs)
                tJAccel[ind] = tDofs[ind].mQstZIcInternal[lane]
                             - loadSpatialVector(tDofs[ind].mIsW, lane)
                                   .innerProduct(pMotionAcceleration);

        for (PxU32 ind = 0; ind < 3; ++ind)
        {
            if (ind < dofs)
            {
                const PxReal jAccel = tDofs[ind].mInvStIsT_x[lane] * tJAccel[0]
                                    + tDofs[ind].mInvStIsT_y[lane] * tJAccel[1]
                                    + tDofs[ind].mInvStIsT_z[lane] * tJAccel[2];
                tDofs[ind].mJointAccel[lane] += jAccel;
                const PxReal jVel = tDofs[ind].mJointVelocities[lane] + jAccel * dt;
                tDofs[ind].mJointVelocities[lane] = jVel;
                tDofs[ind].mJointUnconstrainedVelocities[lane] = jVel;
                motionAcceleration +=
                    loadSpatialVector(tDofs[ind].mWorldMotionMatrix, lane) * jAccel;
            }
        }

        storeSpatialVector(tLink.mMotionAccelerationInternal, motionAcceleration, lane);

        const PxVec3 pos = loadPxVec3(tLink.mAccumulatedPose.p, lane);
        UnAlignedSpatialVector motionVel = loadSpatialVector(tLink.mMotionVelocity, lane);

        PxMat33 inertia;
        loadPxMat33(tLink.mIsolatedInertia, lane, inertia);
        const PxReal mass = tLink.mMass[lane];
        const PxVec3 offset = (pos - COM);

        compoundInertia += refTranslateInertiaMass(inertia, mass, offset);
        preMomentum.bottom += inertia * motionVel.top
                            + offset.cross(motionVel.bottom - preRootVel) * mass;

        motionVel += motionAcceleration * dt;
        postMomentum.top += motionVel.bottom * mass;
        flatVel[linkID] = motionVel;
        storeSpatialVector(tLink.mMotionVelocity, motionVel, lane);
    }

    if (!fixBase)
    {
        const PxVec3 postRootVel = postMomentum.top * COM_invMassW.w;

        for (uint32_t linkID = 0; linkID < numLinks; ++linkID)
        {
            const physx::PxgArticulationBlockLinkData& tLink = lk[linkID];
            const UnAlignedSpatialVector motionVel = loadSpatialVector(tLink.mMotionVelocity, lane);
            PxMat33 inertia;
            loadPxMat33(tLink.mIsolatedInertia, lane, inertia);
            const PxReal mass = tLink.mMass[lane];
            const PxVec3 pos = loadPxVec3(tLink.mAccumulatedPose.p, lane);
            const PxVec3 offset = (pos - COM);
            postMomentum.bottom += inertia * motionVel.top
                                 + offset.cross(motionVel.bottom - postRootVel) * mass;
        }

        const PxMat33 invCompoundInertia = compoundInertia.getInverse();
        const PxReal denom0 = postMomentum.bottom.magnitude();
        const PxReal num = preMomentum.bottom.magnitude();
        const PxReal angRatio = denom0 == 0.f ? 1.f : num / denom0;
        const PxVec3 deltaAngMom = postMomentum.bottom * (angRatio - 1.f);
        const PxVec3 deltaAng = invCompoundInertia * deltaAngMom;

        for (uint32_t linkID = 0; linkID < numLinks; ++linkID)
        {
            const PxVec3 offset = (loadPxVec3(lk[linkID].mAccumulatedPose.p, lane) - COM);
            const UnAlignedSpatialVector velChange(deltaAng, offset.cross(deltaAng));
            addSpatialVector(lk[linkID].mMotionVelocity, velChange, lane);
            postMomentum.top += velChange.bottom * lk[linkID].mMass[lane];
        }

        const PxVec3 deltaLinMom = preMomentum.top - postMomentum.top;
        const PxVec3 deltaLin = deltaLinMom * COM_invMassW.w;

        for (uint32_t linkID = 0; linkID < numLinks; ++linkID)
        {
            UnAlignedSpatialVector vel = loadSpatialVector(lk[linkID].mMotionVelocity, lane);
            vel.bottom += deltaLin;
            storeSpatialVector(lk[linkID].mMotionVelocity, vel, lane);
            flatVel[linkID] = vel;
        }
    }
}

PXG_TEST(artiLinkAcceleration)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiLinkAccel.spv", "artiLinkAcceleration", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiLinkAccelLoader);

    std::vector<physx::PxgArticulationBlockLinkData> lkImg(LA_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(LA_MAX_DOFS);
    std::vector<physx::PxgArticulationBlockData> abImg(1);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));

    std::vector<uint32_t> numLinks(32, 0u);
    uint32_t fixed = 0, freeBase = 0, massless = 0, single = 0;
    for (uint32_t lane = 0; lane < LA_LANES; ++lane)
    {
        numLinks[lane] = laNumLinks(lane);
        PXG_CHECK_MSG(numLinks[lane] <= LA_MAX_LINKS, "lane %u has %u links", lane, numLinks[lane]);
        PXG_CHECK_MSG(numLinks[lane] != lane, "lane %u's link count equals its index", lane);
        PXG_CHECK_MSG(laTotalDofs(lane, numLinks[lane]) <= LA_MAX_DOFS,
                      "lane %u needs %u dofs", lane, laTotalDofs(lane, numLinks[lane]));
        laFill(lkImg.data(), dfImg.data(), abImg[0], lane, numLinks[lane], lane);
        laFixBase(lane) ? ++fixed : ++freeBase;
        if (laMassless(lane)) ++massless;
        if (numLinks[lane] == 1) ++single;
    }
    std::printf("[ info ] la: %u fixed-base, %u free-base, %u massless, %u single-link\n", fixed,
                freeBase, massless, single);
    PXG_CHECK(fixed > 3 && freeBase > 8 && massless > 0 && single > 1);

    // per-lane slices of the flat array, each with a poisoned tail
    std::vector<uint32_t> velPoison(size_t(32) * LA_VEL_SLOTS * 6);
    for (size_t j = 0; j < velPoison.size(); ++j)
        velPoison[j] = 0xAC000000u | uint32_t(j & 0xFFFFFFu);

    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0, dCounts = 0, dVel = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dCounts, numLinks.size() * sizeof(uint32_t));
    shim.memAlloc(&dVel, velPoison.size() * 4);
    if (!dLk || !dDf || !dAb || !dCounts || !dVel)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // the per-lane pointer the block carries -- a stride-7 permutation over the
    // 32 slices, so no lane writes its own index's slice
    uint32_t sliceOf[32];
    for (uint32_t lane = 0; lane < 32; ++lane)
        sliceOf[lane] = (lane * 7u + 5u) % 32u;
    for (uint32_t lane = 0; lane < 32; ++lane)
    {
        PXG_CHECK_MSG(sliceOf[lane] != lane, "lane %u writes its own slice", lane);
        abImg[0].mMotionVelocitiesPtr[lane] = reinterpret_cast<UnAlignedSpatialVector*>(
            dVel + PxgDevicePtr(size_t(sliceOf[lane]) * LA_VEL_SLOTS * 6 * sizeof(float)));
    }

    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dCounts, numLinks.data(), numLinks.size() * sizeof(uint32_t));
    shim.memcpyHtoD(dVel, velPoison.data(), velPoison.size() * 4);

    struct Params
    {
        uint64_t artiBlock, linkBlocks, dofBlocks, numLinksPer;
        float dt;
        uint32_t nbLanes, pad0, pad1;
    } prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dCounts), LA_DT, LA_LANES, 0,
              0 };
    PXG_CHECK_MSG(sizeof(prm) == 48, "push-constant blob is %zu bytes", sizeof(prm));

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiLinkAcceleration");
    shim.streamSynchronize();

    // ---- the expectation ----
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<uint32_t> wantVel = velPoison;
    for (uint32_t lane = 0; lane < LA_LANES; ++lane)
        laMirror(wantLk.data(), wantDf.data(), wantAb[0],
                 reinterpret_cast<UnAlignedSpatialVector*>(wantVel.data())
                     + size_t(sliceOf[lane]) * LA_VEL_SLOTS,
                 numLinks[lane], LA_DT, lane);

    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<uint32_t> gotVel(velPoison.size());
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotVel.data(), dVel, gotVel.size() * 4);

    size_t bad = 0, signedZeros = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == g[k])
                continue;
            if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
            {
                ++signedZeros;
                continue;
            }
            if (bad < 12)
                std::printf("MM la %s word %zu  ref %08x  gpu %08x\n", what, k, w[k], g[k]);
            ++bad;
        }
    };
    cmp("link", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    cmp("dof", wantDf.data(), gotDf.data(),
        wantDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    // the articulation block is an INPUT here: nothing may touch it
    cmp("arti", wantAb.data(), gotAb.data(),
        wantAb.size() * sizeof(physx::PxgArticulationBlockData));
    cmp("flat vel", wantVel.data(), gotVel.data(), wantVel.size() * 4);

    std::printf("[ info ] la: signed zeros forgiven %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "the acceleration sweep matches the engine-built mirror");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
    shim.memFree(dCounts);
    shim.memFree(dVel);
}


// ---------------------------------------------------------------------------
// P9.86f: the compound inertia's zero seed (forwardDynamic2.cu:1946/:2111)
// ---------------------------------------------------------------------------
// PxMat33(PxZero) += translateInertia(...): the root's term is IEEE `+0 + T`, which turns a -0 element
// of T into +0.  T has a -0 element only when the isolated inertia has one AND the (s^T s) * mass term
// is -0 there (mass -0, or mass +0 against a negative (s^T s) element); the sign then reaches the
// inverse's zero cofactors and, when angRatio is exactly 1, deltaAng's zero lanes -- visible in
// mMotionVelocity.top where that lane is -0.  Two directed single-link free-base lanes (found by a
// host search over this mirror): lane 0 with mass -0, lane 1 with mass +0.  invInertia = 0, so both
// root accelerations are -0 lanes.  gLaCompoundFoldTwin replays the folded port (root term = T); it
// must change each lane's velocity, and the GPU must match the CUDA mirror BITWISE there (the body
// test above forgives signed zeros).
PXG_TEST(artiLinkAcceleration_compoundZeroSign)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiLinkAccel.spv", "artiLinkAcceleration", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiLinkAccelLoader);

    struct Lane { PxVec3 c0, c1, c2; float mass; PxVec3 com, pos, top, bottom; };
    const Lane lanes[] = {
        { PxVec3(-1.f, 1.f, -1.f), PxVec3(-0.f, 1.f, 0.f), PxVec3(-0.f, 1.f, 1.f), -0.f,
          PxVec3(1.f, -1.f, 2.f), PxVec3(-1.f, 0.f, -0.f), PxVec3(-0.f, 1.f, -1.f), PxVec3(-1.f, -1.f, -1.f) },
        { PxVec3(0.f, 2.f, 0.f), PxVec3(2.f, -0.f, -0.f), PxVec3(0.f, -1.f, -1.f), 0.f,
          PxVec3(-0.f, 1.f, 0.5f), PxVec3(1.f, 0.f, -0.f), PxVec3(-0.f, -1.f, -0.f), PxVec3(-0.f, 2.f, 0.f) },
    };
    const uint32_t NL = uint32_t(sizeof(lanes) / sizeof(lanes[0]));

    std::vector<physx::PxgArticulationBlockLinkData> lkImg(1);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(1);
    std::vector<physx::PxgArticulationBlockData> abImg(1);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));

    std::vector<uint32_t> numLinks(32, 0u);
    for (uint32_t lane = 0; lane < NL; ++lane)
    {
        numLinks[lane] = 1u;
        laFill(lkImg.data(), dfImg.data(), abImg[0], /*k*/ 0u, 1u, lane);
        PXG_CHECK(!laFixBase(0u) && !laMassless(0u));
        const Lane& L = lanes[lane];
        abImg[0].mCOM_TotalInvMassW[lane] = make_float4(L.com.x, L.com.y, L.com.z, 0.5f);
        storeSpatialMatrix(abImg[0].mInvSpatialArticulatedInertia, lane,
                           physx::Dy::SpatialMatrix(PxMat33(physx::PxZero), PxMat33(physx::PxZero), PxMat33(physx::PxZero)));
        lkImg[0].mMass[lane] = L.mass;
        storePxMat33(lkImg[0].mIsolatedInertia, lane, PxMat33(L.c0, L.c1, L.c2));
        lkImg[0].mAccumulatedPose.p[lane] = make_float4(L.pos.x, L.pos.y, L.pos.z, 0.f);
        storeSpatialVector(lkImg[0].mMotionVelocity, UnAlignedSpatialVector(L.top, L.bottom), lane);
        storeSpatialVector(lkImg[0].mZAVector, UnAlignedSpatialVector(PxVec3(1.f), PxVec3(1.f)), lane);
        storeSpatialVector(lkImg[0].mZAIntVector, UnAlignedSpatialVector(PxVec3(1.f), PxVec3(1.f)), lane);
    }

    std::vector<uint32_t> velPoison(size_t(32) * LA_VEL_SLOTS * 6);
    for (size_t j = 0; j < velPoison.size(); ++j)
        velPoison[j] = 0xAC000000u | uint32_t(j & 0xFFFFFFu);

    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0, dCounts = 0, dVel = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dCounts, numLinks.size() * sizeof(uint32_t));
    shim.memAlloc(&dVel, velPoison.size() * 4);
    if (!dLk || !dDf || !dAb || !dCounts || !dVel)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    for (uint32_t lane = 0; lane < 32; ++lane)
        abImg[0].mMotionVelocitiesPtr[lane] = reinterpret_cast<UnAlignedSpatialVector*>(
            dVel + PxgDevicePtr(size_t(lane) * LA_VEL_SLOTS * 6 * sizeof(float)));

    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dCounts, numLinks.data(), numLinks.size() * sizeof(uint32_t));
    shim.memcpyHtoD(dVel, velPoison.data(), velPoison.size() * 4);

    struct Params
    {
        uint64_t artiBlock, linkBlocks, dofBlocks, numLinksPer;
        float dt;
        uint32_t nbLanes, pad0, pad1;
    } prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dCounts), LA_DT, NL, 0, 0 };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiLinkAcceleration");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<uint32_t> gotVel(velPoison.size());
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotVel.data(), dVel, gotVel.size() * 4);

    for (uint32_t lane = 0; lane < NL; ++lane)
    {
        UnAlignedSpatialVector want[2], flat[2][LA_VEL_SLOTS];
        for (int twin = 0; twin < 2; ++twin)
        {
            std::vector<physx::PxgArticulationBlockLinkData> wLk = lkImg;
            std::vector<physx::PxgArticulationBlockDofData> wDf = dfImg;
            std::vector<physx::PxgArticulationBlockData> wAb = abImg;
            gLaCompoundFoldTwin = (twin == 1);
            laMirror(wLk.data(), wDf.data(), wAb[0], flat[twin], 1u, LA_DT, lane);
            gLaCompoundFoldTwin = false;
            want[twin] = loadSpatialVector(wLk[0].mMotionVelocity, lane);
        }
        PXG_CHECK_MSG(std::memcmp(&want[0], &want[1], sizeof(want[0])) != 0,
                      "lane %u: the folded-seed twin does not change mMotionVelocity", lane);
        const UnAlignedSpatialVector got = loadSpatialVector(gotLk[0].mMotionVelocity, lane);
        const float* w = &want[0].top.x;
        const float* g = &got.top.x;
        for (int j = 0; j < 6; ++j)
            PXG_CHECK_MSG(f2u(w[j]) == f2u(g[j]), "lane %u mMotionVelocity[%d]: gpu %08x cuda %08x (twin %08x)", lane, j,
                          f2u(g[j]), f2u(w[j]), f2u((&want[1].top.x)[j]));
        const uint32_t* gv = gotVel.data() + size_t(lane) * LA_VEL_SLOTS * 6;
        const float* fw = &flat[0][0].top.x;
        for (int j = 0; j < 6; ++j)
            PXG_CHECK_MSG(f2u(fw[j]) == gv[j], "lane %u flat vel[%d]: gpu %08x cuda %08x", lane, j, gv[j], f2u(fw[j]));
    }

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
    shim.memFree(dCounts);
    shim.memFree(dVel);
}

// ---------------------------------------------------------------------------
// P8.92c: the kernel
// ---------------------------------------------------------------------------
//
// 130 articulations, grid `num1TBlocks` = 3, block (32, `2`, 1) — the shape of
// the PARTIAL kernel, not of the two (32, 1, 1) kernels between them.  Five live
// warp slices and a sixth allocated and left entirely poisoned.  The mirror is
// the same `laMirror` the body test uses, followed by `frMirror`, in that order:
// the friction sweep reads both acceleration fields the acceleration sweep wrote.

const uint32_t LAK_NB_ARTI = 130;
const uint32_t LAK_WARPS_ALLOC = 6;

PXG_TEST(artiAccelerationLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // engine ABI { 8 } -- the descriptor pointer alone
        { "npArtiLinkAccel.spv", "computeUnconstrainedAccelerationsLaunch1T", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiLinkAccelLoader);

    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(LAK_WARPS_ALLOC) * LA_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(LAK_WARPS_ALLOC) * LA_MAX_DOFS);
    std::vector<physx::PxgArticulationBlockData> abImg(LAK_WARPS_ALLOC);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));

    std::vector<uint32_t> numLinks(LAK_NB_ARTI);
    for (uint32_t k = 0; k < LAK_NB_ARTI; ++k)
    {
        numLinks[k] = laNumLinks(k);
        const uint32_t warp = k / 32u, lane = k % 32u;
        PXG_CHECK_MSG(laTotalDofs(k, numLinks[k]) <= LA_MAX_DOFS, "articulation %u needs %u dofs",
                      k, laTotalDofs(k, numLinks[k]));
        laFill(lkImg.data() + size_t(warp) * LA_MAX_LINKS, dfImg.data() + size_t(warp) * LA_MAX_DOFS,
               abImg[warp], k, numLinks[k], lane);
    }

    std::vector<uint32_t> velPoison(size_t(LAK_NB_ARTI + 8) * LA_VEL_SLOTS * 6);
    for (size_t j = 0; j < velPoison.size(); ++j)
        velPoison[j] = 0xAC000000u | uint32_t(j & 0xFFFFFFu);

    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0, dVel = 0, dDesc = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dVel, velPoison.size() * 4);
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    if (!dLk || !dDf || !dAb || !dVel || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // a stride-31 permutation over the slices, so no articulation writes the slice
    // its own index would name
    std::vector<uint32_t> sliceOf(LAK_NB_ARTI);
    for (uint32_t k = 0; k < LAK_NB_ARTI; ++k)
    {
        sliceOf[k] = (k * 31u + 17u) % (LAK_NB_ARTI + 8u);
        const uint32_t warp = k / 32u, lane = k % 32u;
        abImg[warp].mMotionVelocitiesPtr[lane] = reinterpret_cast<UnAlignedSpatialVector*>(
            dVel + PxgDevicePtr(size_t(sliceOf[k]) * LA_VEL_SLOTS * 6 * sizeof(float)));
    }

    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dVel, velPoison.data(), velPoison.size() * 4);

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0x5A, sizeof(desc));
    desc.nbArticulations = LAK_NB_ARTI;
    desc.dt = LA_DT;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks = reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.mArticulationDofBlocks = reinterpret_cast<physx::PxgArticulationBlockDofData*>(dDf);
    desc.mMaxLinksPerArticulation = LA_MAX_LINKS;
    desc.mMaxDofsPerArticulation = LA_MAX_DOFS;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    struct Params { uint64_t desc; } prm = { uint64_t(dDesc) };
    const uint32_t num1TBlocks = (LAK_NB_ARTI + 63u) / 64u;
    PXG_CHECK_MSG(num1TBlocks == 3u, "fixture assumes three 1T blocks, got %u", num1TBlocks);
    // grid num1TBlocks and TWO warps per block -- not the (32, 1, 1) x 2 shape
    PXG_CHECK_MSG(shim.launchKernel(0, num1TBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch computeUnconstrainedAccelerationsLaunch1T");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<uint32_t> wantVel = velPoison;
    for (uint32_t k = 0; k < LAK_NB_ARTI; ++k)
    {
        const uint32_t warp = k / 32u, lane = k % 32u;
        physx::PxgArticulationBlockLinkData* lk = wantLk.data() + size_t(warp) * LA_MAX_LINKS;
        laMirror(lk, wantDf.data() + size_t(warp) * LA_MAX_DOFS, wantAb[warp],
                 reinterpret_cast<UnAlignedSpatialVector*>(wantVel.data())
                     + size_t(sliceOf[k]) * LA_VEL_SLOTS,
                 numLinks[k], LA_DT, lane);
        // AFTER the sweep, and reading what it wrote
        frMirror(lk, numLinks[k], lane);
    }

    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<uint32_t> gotVel(velPoison.size());
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotVel.data(), dVel, gotVel.size() * 4);

    size_t bad = 0, signedZeros = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == g[k])
                continue;
            if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
            {
                ++signedZeros;
                continue;
            }
            if (bad < 12)
                std::printf("MM lak %s word %zu  ref %08x  gpu %08x\n", what, k, w[k], g[k]);
            ++bad;
        }
    };
    cmp("link", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    cmp("dof", wantDf.data(), gotDf.data(),
        wantDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    cmp("arti", wantAb.data(), gotAb.data(),
        wantAb.size() * sizeof(physx::PxgArticulationBlockData));
    cmp("flat vel", wantVel.data(), gotVel.data(), wantVel.size() * 4);

    std::printf("[ info ] lak: signed zeros forgiven %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "the kernel's whole block image matches the mirror");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
    shim.memFree(dVel);
    shim.memFree(dDesc);
}

}  // namespace
