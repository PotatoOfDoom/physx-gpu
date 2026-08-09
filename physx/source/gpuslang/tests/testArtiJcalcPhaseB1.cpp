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

// P8.73: jcalc phase B part 1 -- the root link block fill and the path-to-root
// bitfields.
//
// WHAT THE ORACLE IS, HONESTLY.  `jcalc` is __device__-only, so unlike phase A
// (which could call the engine's own PX_CUDA_CALLABLE setJointFrame) there is no
// engine function to compare against.  Three things carry the verification
// instead, and none of them is "the fixture replays the shader":
//
//   1. Every offset the shader uses is generated from the engine headers and
//      pinned by static_assert in testArtiBlockLayout.cpp, so a shared wrong
//      constant -- the P8.49a failure mode -- cannot happen.
//   2. The expected bitfields are derived a SECOND WAY.  The input the shader
//      reads is the engine's flat `pathToRoot` array, whose layout is the one
//      FeatherstoneArticulation::initPathToRoot builds (root gets count 0, every
//      other link gets its ancestors EXCLUDING the root -- which is why the
//      kernel ORs bit 0 in separately).  The expectation here is built by
//      walking the parent array instead, so the two only agree if both the array
//      construction and the kernel's reading of it are right.
//   3. The whole destination is compared, not just the written fields: both
//      regions are poisoned with 0xAB and every byte outside the expected writes
//      must still be 0xAB.
//
// THE SHAPE IS CHOSEN TO BREAK TWO SPECIFIC THINGS.
//
// * Five articulations at lanes 0-4.  mDofs/mDisableGravity/mRetainsAcceleration
//   are PxU8[32] fields, so lanes 0-3 share one 32-bit word and lane 4 starts the
//   next.  A non-atomic byte store loses lanes, a wrong (lane&3)*8 shift writes
//   the wrong byte, and lanes 5-31 must be left at poison.  With one articulation
//   none of that would be visible.
// * Two sub-cases, maxNumLinks 4 (wordSize 1) and maxNumLinks 70 (wordSize 2),
//   the second with a 70-link chain so path indices run past 63.  A kernel that
//   hard-codes word 0, or shifts by `index` instead of `index % 64`, passes the
//   first case and fails the second.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> jcalcB1Loader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;        // articulations in the warp
static const uint32_t GUARD = 128;

struct Case
{
    const char* name;
    uint32_t maxNumLinks;
    uint32_t numLinks[LANES];
};

// the parent of link i, for the tree this fixture builds.  Lanes 0-3 are chains;
// lane 4 branches at link 1 so a "parent == linkID - 1" shortcut would show up.
static uint32_t parentOf(uint32_t lane, uint32_t linkID)
{
    if (linkID == 0)
        return 0;
    if (lane == 4)
        return (linkID <= 2) ? (linkID - 1) : 1u;
    return linkID - 1;
}

PXG_TEST(artiJcalcPhaseB1)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64, u64, u64, u32, u32 }
        { "npArtiJointFrame.spv", "artiJcalcPhaseB1", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcB1Loader);

    const Case cases[] = {
        { "maxNumLinks 4 (wordSize 1)",  4u,  { 1u, 2u, 3u, 4u, 4u } },
        { "maxNumLinks 70 (wordSize 2)", 70u, { 70u, 68u, 65u, 66u, 2u } },
    };

    const size_t LINK_BLOCK = sizeof(PxgArticulationBlockLinkData);
    const size_t BITFIELD = sizeof(PxgArticulationBitFieldStackData);

    size_t bad = 0;

    for (const Case& c : cases)
    {
        const uint32_t maxNumLinks = c.maxNumLinks;
        const uint32_t wordSize = (maxNumLinks + 63u) / 64u;

        // ---- host-side records ----
        std::vector<PxgArticulationLink> links(size_t(LANES) * maxNumLinks);
        // POISONED, not zeroed, and the first sweep proved why: numChildren is a
        // PxU16 at offset 58 and offsetSlop sits at 60, so a word read at 58
        // returns numChildren in its low half plus offsetSlop's low half above it.
        // With offsetSlop zero the wrong-width read gives the right answer and the
        // mutant survives -- the same trap nbDof set in P8.72.  Every field the
        // kernel is supposed to read is assigned below; everything else stays 0xEE.
        std::memset(links.data(), 0xEE, links.size() * sizeof(PxgArticulationLink));
        std::vector<PxTransform> body2Worlds(size_t(LANES) * maxNumLinks);
        std::vector<PxU32> pathToRoot;              // all lanes, concatenated
        std::vector<uint32_t> pathBase(LANES, 0);   // where each lane's array starts

        for (uint32_t L = 0; L < LANES; ++L)
        {
            const uint32_t nl = c.numLinks[L];
            for (uint32_t i = 0; i < maxNumLinks; ++i)
            {
                PxgArticulationLink& lk = links[size_t(L) * maxNumLinks + i];
                lk.initialAngVel = PxVec3(1.0f + L, 2.0f + i, -3.0f - L);
                lk.penBiasClamp = -100.0f - L;      // must NOT reach the block
                lk.initialLinVel = PxVec3(-4.0f - L, 5.0f + i, 6.0f + L);
                lk.invMass = 42.0f + L;             // must NOT reach the block
                lk.maxAngularVelocitySq = 700.0f + L;
                lk.maxLinearVelocitySq = 800.0f + L;
                lk.linearDamping = 0.25f * (L + 1);
                lk.angularDamping = 0.5f * (L + 1);
                lk.childrenOffset = 1000u + 10u * L + i;
                lk.numChildren = PxU16(3u + L);
                lk.disableGravity = ((L + i) & 1) != 0;
                lk.retainsAccelerations = ((L + i) & 2) != 0;
                body2Worlds[size_t(L) * maxNumLinks + i] =
                    PxTransform(PxVec3(0.1f * (L + 1), 0.2f * (i + 1), -0.3f * (L + 1)),
                                PxQuat(0.2f, -0.4f, 0.5f, 0.7f).getNormalized());
            }

            // the engine's own path-to-root layout (initPathToRoot): element 0 is
            // the root's, every other link's run holds its ancestors EXCLUDING the
            // root, deepest last.
            pathBase[L] = uint32_t(pathToRoot.size());
            pathToRoot.push_back(0);
            links[size_t(L) * maxNumLinks + 0].pathToRootOffset = pathBase[L];
            links[size_t(L) * maxNumLinks + 0].numPathToRoot = 0;
            for (uint32_t linkID = 1; linkID < nl; ++linkID)
            {
                std::vector<PxU32> chain;
                for (uint32_t p = linkID; p != 0; p = parentOf(L, p))
                    chain.push_back(p);
                // deepest last, as the engine writes it
                const uint32_t start = uint32_t(pathToRoot.size());
                for (size_t k = chain.size(); k-- > 0;)
                    pathToRoot.push_back(chain[k]);
                PxgArticulationLink& lk = links[size_t(L) * maxNumLinks + linkID];
                lk.pathToRootOffset = start;
                lk.numPathToRoot = PxU16(chain.size());
            }
        }

        // ---- device buffers ----
        // only element 0 of the link block array is written by this phase; two
        // elements are allocated so a wrong struct stride lands inside the checked
        // region rather than off the end of the allocation.
        const size_t LINK_REGION = 2 * LINK_BLOCK + GUARD;
        const size_t BF_REGION = size_t(maxNumLinks) * wordSize * BITFIELD + GUARD;

        PxgDevicePtr dLinks = 0, dB2W = 0, dPath = 0, dArtis = 0, dLinkBlocks = 0, dBitFields = 0;
        shim.memAlloc(&dLinks, links.size() * sizeof(PxgArticulationLink));
        shim.memAlloc(&dB2W, body2Worlds.size() * sizeof(PxTransform));
        shim.memAlloc(&dPath, pathToRoot.size() * sizeof(PxU32));
        shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
        shim.memAlloc(&dLinkBlocks, LINK_REGION);
        shim.memAlloc(&dBitFields, BF_REGION);
        if (!dLinks || !dB2W || !dPath || !dArtis || !dLinkBlocks || !dBitFields)
        {
            std::printf("  NOTE: no BDA -> skip\n");
            return;
        }

        std::vector<PxgArticulation> recs(LANES);
        std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
        for (uint32_t L = 0; L < LANES; ++L)
        {
            recs[L].data.numLinks = PxU16(c.numLinks[L]);
            recs[L].links = reinterpret_cast<PxgArticulationLink*>(
                dLinks + PxgDevicePtr(size_t(L) * maxNumLinks * sizeof(PxgArticulationLink)));
            recs[L].linkBody2Worlds = reinterpret_cast<PxTransform*>(
                dB2W + PxgDevicePtr(size_t(L) * maxNumLinks * sizeof(PxTransform)));
            recs[L].pathToRoot = reinterpret_cast<PxU32*>(dPath);
        }

        shim.memcpyHtoD(dLinks, links.data(), links.size() * sizeof(PxgArticulationLink));
        shim.memcpyHtoD(dB2W, body2Worlds.data(), body2Worlds.size() * sizeof(PxTransform));
        shim.memcpyHtoD(dPath, pathToRoot.data(), pathToRoot.size() * sizeof(PxU32));
        shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));

        std::vector<uint8_t> poisonLink(LINK_REGION, 0xAB);
        std::vector<uint8_t> poisonBF(BF_REGION, 0xAB);
        shim.memcpyHtoD(dLinkBlocks, poisonLink.data(), poisonLink.size());
        shim.memcpyHtoD(dBitFields, poisonBF.data(), poisonBF.size());

        struct Params
        {
            uint64_t artis;
            uint64_t linkBlocks;
            uint64_t bitFields;
            uint32_t maxNumLinks;
            uint32_t nbArticulations;
        } prm = { uint64_t(dArtis), uint64_t(dLinkBlocks), uint64_t(dBitFields), maxNumLinks, LANES };

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiJcalcPhaseB1");
        shim.streamSynchronize();

        // ---- the expectation, built over the poison image ----
        std::vector<uint8_t> wantLink = poisonLink;
        std::vector<uint8_t> wantBF = poisonBF;
        auto putU8 = [&](std::vector<uint8_t>& img, size_t off, uint8_t v) { img[off] = v; };
        auto putU32 = [&](std::vector<uint8_t>& img, size_t off, uint32_t v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putF = [&](std::vector<uint8_t>& img, size_t off, float v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putU64 = [&](std::vector<uint8_t>& img, size_t off, uint64_t v) {
            std::memcpy(img.data() + off, &v, 8);
        };

        for (uint32_t L = 0; L < LANES; ++L)
        {
            const PxgArticulationLink& root = links[size_t(L) * maxNumLinks];

            putU8(wantLink, offsetof(PxgArticulationBlockLinkData, mDisableGravity) + L,
                  root.disableGravity ? 1u : 0u);
            putU8(wantLink, offsetof(PxgArticulationBlockLinkData, mRetainsAcceleration) + L,
                  root.retainsAccelerations ? 1u : 0u);
            putU8(wantLink, offsetof(PxgArticulationBlockLinkData, mDofs) + L, 0u);

            const size_t damp =
                offsetof(PxgArticulationBlockLinkData,
                         mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW) + size_t(L) * 16;
            putF(wantLink, damp + 0, root.linearDamping);
            putF(wantLink, damp + 4, root.angularDamping);
            putF(wantLink, damp + 8, root.maxLinearVelocitySq);
            putF(wantLink, damp + 12, root.maxAngularVelocitySq);

            // motion velocity: top = angular, bottom = linear; the .w halves of the
            // two record float4s (penBiasClamp, invMass) must not appear anywhere.
            const size_t mv = offsetof(PxgArticulationBlockLinkData, mMotionVelocity);
            putF(wantLink, mv + 0 + size_t(L) * 16 + 0, root.initialAngVel.x);
            putF(wantLink, mv + 0 + size_t(L) * 16 + 4, root.initialAngVel.y);
            putF(wantLink, mv + 0 + size_t(L) * 16 + 8, root.initialAngVel.z);
            putF(wantLink, mv + 0 + size_t(L) * 16 + 12, root.initialLinVel.x);
            putF(wantLink, mv + 512 + size_t(L) * 8 + 0, root.initialLinVel.y);
            putF(wantLink, mv + 512 + size_t(L) * 8 + 4, root.initialLinVel.z);

            const PxTransform& pose = body2Worlds[size_t(L) * maxNumLinks];
            const size_t ap = offsetof(PxgArticulationBlockLinkData, mAccumulatedPose);
            putF(wantLink, ap + 0 + size_t(L) * 16 + 0, pose.q.x);
            putF(wantLink, ap + 0 + size_t(L) * 16 + 4, pose.q.y);
            putF(wantLink, ap + 0 + size_t(L) * 16 + 8, pose.q.z);
            putF(wantLink, ap + 0 + size_t(L) * 16 + 12, pose.q.w);
            putF(wantLink, ap + 512 + size_t(L) * 16 + 0, pose.p.x);
            putF(wantLink, ap + 512 + size_t(L) * 16 + 4, pose.p.y);
            putF(wantLink, ap + 512 + size_t(L) * 16 + 8, pose.p.z);
            putF(wantLink, ap + 512 + size_t(L) * 16 + 12, 0.0f);

            putU32(wantLink, offsetof(PxgArticulationBlockLinkData, mChildrenOffset) + size_t(L) * 4,
                   root.childrenOffset);
            putU32(wantLink, offsetof(PxgArticulationBlockLinkData, mNumChildren) + size_t(L) * 4,
                   root.numChildren);

            // ---- the bitfields, derived by walking parents ----
            for (uint32_t linkID = 0; linkID < maxNumLinks; ++linkID)
            {
                std::vector<uint64_t> row(wordSize, 0);
                if (linkID == 0)
                {
                    row[0] = 1ull;
                }
                else if (linkID < c.numLinks[L])
                {
                    for (uint32_t p = linkID;; p = parentOf(L, p))
                    {
                        row[p / 64] |= 1ull << (p % 64);
                        if (p == 0)
                            break;
                    }
                }
                for (uint32_t w = 0; w < wordSize; ++w)
                    putU64(wantBF, (size_t(linkID) * wordSize + w) * BITFIELD + size_t(L) * 8,
                           row[w]);
            }
        }
        // Rows past a lane's last link are CLEARED (the clear loop covers all
        // maxNumLinks * wordSize rows), which the zero rows above already encode.
        // Lanes 5-31 are never written at all and stay at poison -- that is the
        // check that the byte stores do not spill across lanes.

        // ---- compare ----
        auto region = [&](const char* what, PxgDevicePtr p, const std::vector<uint8_t>& want) {
            std::vector<uint8_t> got(want.size(), 0);
            shim.memcpyDtoH(got.data(), p, got.size());
            size_t local = 0;
            for (size_t i = 0; i < want.size(); ++i)
            {
                if (got[i] != want[i])
                {
                    if (bad < 12)
                        std::printf("  [%s] %s +%zu: got 0x%02X want 0x%02X\n",
                                    c.name, what, i, got[i], want[i]);
                    ++bad;
                    if (++local >= 4)
                        return;
                }
            }
        };
        region("linkBlocks", dLinkBlocks, wantLink);
        region("bitFields", dBitFields, wantBF);

        shim.memFree(dLinks);
        shim.memFree(dB2W);
        shim.memFree(dPath);
        shim.memFree(dArtis);
        shim.memFree(dLinkBlocks);
        shim.memFree(dBitFields);
    }

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  jcalc phase B1: 5 articulations x {wordSize 1, wordSize 2 with a 70-link "
                "chain} -- root block fill and path-to-root bitfields byte-exact, poison "
                "intact outside\n");
}

} // namespace
