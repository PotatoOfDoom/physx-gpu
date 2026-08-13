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

}  // namespace
