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

// P8.74: jcalc phase B2 -- the per-link block fields.
//
// Same oracle situation as P8.73 and stated the same way: jcalc is __device__-only,
// so the expectation is built by the fixture from its own inputs.  What carries it
// is that every offset comes from the engine headers via static_assert, that the
// whole destination is compared against a 0xAB poison image so a stray write fails,
// and a mutation sweep.  This is data movement, not maths -- there is no numeric
// result to be subtly wrong, only addresses.
//
// The shape targets the three specific ways an address can be wrong here:
//
//  * FIVE articulations at lanes 0-4, because mDofs and mJointType are PxU8[32]
//    and lanes 0-3 share a 32-bit word.  Lanes 5-31 must stay at poison.
//  * numLinks differing per lane (1..5) over maxNumLinks 5, so the loop bound is
//    load-bearing: link blocks past a lane's last link must stay at poison, and
//    link block 0 must be untouched -- the root is P8.73's job, not this one.
//  * Every value distinct per (lane, link), so a field written at the wrong link
//    stride, the wrong field offset or the wrong lane index all differ from the
//    expectation rather than coinciding.
//
// The records are poisoned with 0xEE rather than zeroed.  That is not caution, it
// is the third tranche running where a zeroed neighbour hid a wrong-width read:
// nbDof is a PxU8 at offset 4 of ArticulationJointCoreData and numChildren is a
// PxU16 at 58 with offsetSlop at 60.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "DyArticulationJointCore.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> jcalcB2Loader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;
static const uint32_t MAX_LINKS = 5;
static const uint32_t GUARD = 128;

PXG_TEST(artiJcalcPhaseB2)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiJointFrame.spv", "artiJcalcPhaseB2", 24u, 0 }, // { u64, u64, u32, u32 }
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcB2Loader);

    const uint32_t numLinks[LANES] = { 1u, 2u, 3u, 4u, 5u };

    // ---- host-side records, all poisoned first ----
    std::vector<PxgArticulationLink> links(size_t(LANES) * MAX_LINKS);
    std::memset(links.data(), 0xEE, links.size() * sizeof(PxgArticulationLink));
    std::vector<ArticulationJointCoreData> jdata(size_t(LANES) * MAX_LINKS);
    std::memset(jdata.data(), 0xEE, jdata.size() * sizeof(ArticulationJointCoreData));
    std::vector<PxTransform> body2Worlds(size_t(LANES) * MAX_LINKS);
    std::vector<PxQuat> relativeQuats(size_t(LANES) * MAX_LINKS);
    std::vector<PxU32> parents(size_t(LANES) * MAX_LINKS);
    std::vector<ArticulationJointCore> joints;
    joints.reserve(size_t(LANES) * MAX_LINKS);

    const PxArticulationJointType::Enum types[5] = {
        PxArticulationJointType::eFIX, PxArticulationJointType::ePRISMATIC,
        PxArticulationJointType::eREVOLUTE, PxArticulationJointType::eREVOLUTE_UNWRAPPED,
        PxArticulationJointType::eSPHERICAL
    };

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t i = 0; i < MAX_LINKS; ++i)
        {
            const size_t k = size_t(L) * MAX_LINKS + i;
            PxgArticulationLink& lk = links[k];
            lk.linearDamping = 0.125f * (L + 1) + 0.001f * i;
            lk.angularDamping = 0.25f * (L + 1) + 0.002f * i;
            lk.maxLinearVelocitySq = 100.0f * (L + 1) + i;
            lk.maxAngularVelocitySq = 200.0f * (L + 1) + i;
            lk.childrenOffset = 7000u + 100u * L + i;
            lk.numChildren = PxU16(1u + L + i);
            lk.disableGravity = ((L * 3 + i) & 1) != 0;
            lk.retainsAccelerations = ((L + i * 2) & 1) != 0;

            jdata[k].nbDof = PxU8((i == 0) ? 0u : (1u + ((L + i) % 3u)));
            jdata[k].jointOffset = 10u * L + i;

            body2Worlds[k] = PxTransform(PxVec3(1.0f * L + 0.25f * i, -2.0f * L, 3.0f + i),
                                         PxQuat(0.2f, -0.4f, 0.5f, 0.7f).getNormalized());
            relativeQuats[k] = PxQuat(0.1f * (L + 1), 0.2f * (i + 1), -0.3f, 0.9f);
            parents[k] = (i == 0) ? 0u : (i - 1u);

            ArticulationJointCore j(
                PxTransform(PxVec3(-1.0f * i, 0.5f * L, 2.0f), PxQuat(0.35f, 0.15f, -0.55f, 0.74f)),
                PxTransform(PxVec3(0.75f * L, -0.5f * i, -1.25f), PxQuat(-0.1f, 0.6f, 0.3f, 0.73f)));
            j.jointType = PxU8(types[(L + i) % 5]);
            joints.push_back(j);
        }
    }

    const size_t LINK_BLOCK = sizeof(PxgArticulationBlockLinkData);
    const size_t LINK_REGION = size_t(MAX_LINKS) * LINK_BLOCK + GUARD;

    PxgDevicePtr dLinks = 0, dJData = 0, dJoints = 0, dB2W = 0, dRQ = 0, dParents = 0;
    PxgDevicePtr dArtis = 0, dLinkBlocks = 0;
    shim.memAlloc(&dLinks, links.size() * sizeof(PxgArticulationLink));
    shim.memAlloc(&dJData, jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memAlloc(&dJoints, joints.size() * sizeof(ArticulationJointCore));
    shim.memAlloc(&dB2W, body2Worlds.size() * sizeof(PxTransform));
    shim.memAlloc(&dRQ, relativeQuats.size() * sizeof(PxQuat));
    shim.memAlloc(&dParents, parents.size() * sizeof(PxU32));
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dLinkBlocks, LINK_REGION);
    if (!dLinks || !dJData || !dJoints || !dB2W || !dRQ || !dParents || !dArtis || !dLinkBlocks)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<PxgArticulation> recs(LANES);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t L = 0; L < LANES; ++L)
    {
        const size_t off = size_t(L) * MAX_LINKS;
        recs[L].data.numLinks = PxU16(numLinks[L]);
        recs[L].links = reinterpret_cast<PxgArticulationLink*>(
            dLinks + PxgDevicePtr(off * sizeof(PxgArticulationLink)));
        recs[L].jointData = reinterpret_cast<ArticulationJointCoreData*>(
            dJData + PxgDevicePtr(off * sizeof(ArticulationJointCoreData)));
        recs[L].joints = reinterpret_cast<ArticulationJointCore*>(
            dJoints + PxgDevicePtr(off * sizeof(ArticulationJointCore)));
        recs[L].linkBody2Worlds = reinterpret_cast<PxTransform*>(
            dB2W + PxgDevicePtr(off * sizeof(PxTransform)));
        recs[L].relativeQuat = reinterpret_cast<PxQuat*>(
            dRQ + PxgDevicePtr(off * sizeof(PxQuat)));
        recs[L].parents = reinterpret_cast<PxU32*>(
            dParents + PxgDevicePtr(off * sizeof(PxU32)));
    }

    shim.memcpyHtoD(dLinks, links.data(), links.size() * sizeof(PxgArticulationLink));
    shim.memcpyHtoD(dJData, jdata.data(), jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memcpyHtoD(dJoints, joints.data(), joints.size() * sizeof(ArticulationJointCore));
    shim.memcpyHtoD(dB2W, body2Worlds.data(), body2Worlds.size() * sizeof(PxTransform));
    shim.memcpyHtoD(dRQ, relativeQuats.data(), relativeQuats.size() * sizeof(PxQuat));
    shim.memcpyHtoD(dParents, parents.data(), parents.size() * sizeof(PxU32));
    shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));

    std::vector<uint8_t> poison(LINK_REGION, 0xAB);
    shim.memcpyHtoD(dLinkBlocks, poison.data(), poison.size());

    struct Params
    {
        uint64_t artis;
        uint64_t linkBlocks;
        uint32_t nbArticulations;
        uint32_t pad0;
    } prm = { uint64_t(dArtis), uint64_t(dLinkBlocks), LANES, 0u };

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiJcalcPhaseB2");
    shim.streamSynchronize();

    // ---- the expectation, over the poison image ----
    std::vector<uint8_t> want = poison;
    auto putU8 = [&](size_t off, uint8_t v) { want[off] = v; };
    auto putU32 = [&](size_t off, uint32_t v) { std::memcpy(want.data() + off, &v, 4); };
    auto putF = [&](size_t off, float v) { std::memcpy(want.data() + off, &v, 4); };
    auto putPose = [&](size_t blk, size_t field, uint32_t L, const PxTransform& t) {
        putF(blk + field + 0 + size_t(L) * 16 + 0, t.q.x);
        putF(blk + field + 0 + size_t(L) * 16 + 4, t.q.y);
        putF(blk + field + 0 + size_t(L) * 16 + 8, t.q.z);
        putF(blk + field + 0 + size_t(L) * 16 + 12, t.q.w);
        putF(blk + field + 512 + size_t(L) * 16 + 0, t.p.x);
        putF(blk + field + 512 + size_t(L) * 16 + 4, t.p.y);
        putF(blk + field + 512 + size_t(L) * 16 + 8, t.p.z);
        putF(blk + field + 512 + size_t(L) * 16 + 12, 0.0f);
    };

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t linkID = 1; linkID < numLinks[L]; ++linkID)
        {
            const size_t k = size_t(L) * MAX_LINKS + linkID;
            const size_t blk = size_t(linkID) * LINK_BLOCK;
            const PxgArticulationLink& lk = links[k];

            const size_t rq = blk + offsetof(PxgArticulationBlockLinkData, mRelativeQuat) +
                              size_t(L) * 16;
            putF(rq + 0, relativeQuats[k].x);
            putF(rq + 4, relativeQuats[k].y);
            putF(rq + 8, relativeQuats[k].z);
            putF(rq + 12, relativeQuats[k].w);

            putU32(blk + offsetof(PxgArticulationBlockLinkData, mJointOffset) + size_t(L) * 4,
                   jdata[k].jointOffset);
            putU8(blk + offsetof(PxgArticulationBlockLinkData, mDofs) + L, jdata[k].nbDof);
            putU8(blk + offsetof(PxgArticulationBlockLinkData, mJointType) + L,
                  joints[k].jointType);

            const size_t damp =
                blk + offsetof(PxgArticulationBlockLinkData,
                               mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW) +
                size_t(L) * 16;
            putF(damp + 0, lk.linearDamping);
            putF(damp + 4, lk.angularDamping);
            putF(damp + 8, lk.maxLinearVelocitySq);
            putF(damp + 12, lk.maxAngularVelocitySq);

            putU32(blk + offsetof(PxgArticulationBlockLinkData, mParents) + size_t(L) * 4,
                   parents[k]);
            putU32(blk + offsetof(PxgArticulationBlockLinkData, mChildrenOffset) + size_t(L) * 4,
                   lk.childrenOffset);
            putU32(blk + offsetof(PxgArticulationBlockLinkData, mNumChildren) + size_t(L) * 4,
                   lk.numChildren);
            putU8(blk + offsetof(PxgArticulationBlockLinkData, mDisableGravity) + L,
                  lk.disableGravity ? 1u : 0u);
            putU8(blk + offsetof(PxgArticulationBlockLinkData, mRetainsAcceleration) + L,
                  lk.retainsAccelerations ? 1u : 0u);

            putPose(blk, offsetof(PxgArticulationBlockLinkData, mChildPose), L,
                    joints[k].childPose);
            putPose(blk, offsetof(PxgArticulationBlockLinkData, mParentPose), L,
                    joints[k].parentPose);
            putPose(blk, offsetof(PxgArticulationBlockLinkData, mAccumulatedPose), L,
                    body2Worlds[k]);
        }
    }

    // ---- compare, byte for byte, including the untouched link blocks ----
    std::vector<uint8_t> got(LINK_REGION, 0);
    shim.memcpyDtoH(got.data(), dLinkBlocks, got.size());
    size_t bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
    {
        if (got[i] != want[i])
        {
            if (bad < 12)
                std::printf("  linkBlocks +%zu (block %zu, +%zu): got 0x%02X want 0x%02X\n",
                            i, i / LINK_BLOCK, i % LINK_BLOCK, got[i], want[i]);
            ++bad;
        }
    }

    shim.memFree(dLinks);
    shim.memFree(dJData);
    shim.memFree(dJoints);
    shim.memFree(dB2W);
    shim.memFree(dRQ);
    shim.memFree(dParents);
    shim.memFree(dArtis);
    shim.memFree(dLinkBlocks);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  jcalc phase B2: 5 articulations x 1..5 links -- 13 per-link block fields "
                "byte-exact, block 0 and the unused blocks still poison\n");
}

} // namespace
