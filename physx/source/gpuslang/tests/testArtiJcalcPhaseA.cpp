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

// P8.72: jcalc phase A -- the record-side recompute.
//
// The oracle splits, and the split is the point.  The only non-trivial maths in
// this phase is `computeJointAxis` + `setJointFrame`, and the fixture gets that
// from THE ENGINE ITSELF by calling the real `PX_CUDA_CALLABLE` functions on the
// host -- the same oracle P8.69 used, now driven through a four-link loop.  What
// is left over (the dof accumulation, `jointOffsets`, and two cfm multiplies) is
// arithmetic computed directly from the fixture's own inputs.  Nothing about the
// joint maths is transcribed here.
//
// The articulation is shaped so every index in the loop is load-bearing:
//
//   link 1  revolute   1 dof   jointOffset 0
//   link 2  spherical  2 dof   jointOffset 1   <- jointOffset != linkID from here
//   link 3  prismatic  1 dof   jointOffset 3
//
// (5.12 keeps the joint axes in a local [3] array -- no record jointAxis, no
// totalDofs accumulation -- so only the motion matrices / relative quats carry
// them out.)  Link 3 also
// has `jCalcUpdateFrames` CLEAR, which the engine treats as "frames already
// current": its relative quat and motion matrix must stay untouched while its
// joint offset and cfm are still written.
//
// Four dispatches cover the guard matrix -- (confiDirty, dataDirty) over
// {1,1},{1,0},{0,1},{0,0} -- with the record re-uploaded each time, and the
// fixed-base flag toggled so both sides of the root's `? 0 :` run.  The (0,0)
// case must write NOTHING: it is the early-out, and the poisoned destinations
// prove it.

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

static std::vector<uint8_t> jcalcALoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t NB_LINKS = 4;
static const uint32_t TOTAL_DOFS = 4;   // 1 + 2 + 1

PXG_TEST(artiJcalcPhaseA)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiJointFrame.spv", "artiJcalcPhaseA", 16u, 0 }, // { u64, f32, u32 }
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcALoader);

    struct LinkSpec
    {
        PxArticulationJointType::Enum type;
        uint32_t dof;
        uint8_t dofIds[3];
        uint32_t jointOffset;
        bool updateFrames;
    };
    const LinkSpec spec[NB_LINKS] = {
        { PxArticulationJointType::eFIX, 0, { 0, 0, 0 }, 0, false },   // root, unused
        { PxArticulationJointType::eREVOLUTE,  1, { 1, 0, 0 }, 0, true  },
        { PxArticulationJointType::eSPHERICAL, 2, { 0, 2, 0 }, 1, true  },
        { PxArticulationJointType::ePRISMATIC, 1, { 4, 0, 0 }, 3, false },
    };
    const PxQuat pq[NB_LINKS] = { PxQuat(PxIdentity), PxQuat(0.2f, -0.4f, 0.5f, 0.7f),
                                  PxQuat(0.35f, 0.15f, -0.55f, 0.74f), PxQuat(-0.1f, 0.6f, 0.3f, 0.73f) };
    const PxQuat cq[NB_LINKS] = { PxQuat(PxIdentity), PxQuat(0.35f, 0.15f, -0.55f, 0.74f),
                                  PxQuat(-0.1f, 0.6f, 0.3f, 0.73f), PxQuat(0.2f, -0.4f, 0.5f, 0.7f) };
    const PxVec3 cp[NB_LINKS] = { PxVec3(0.0f), PxVec3(0.3f, -0.7f, 1.1f),
                                  PxVec3(-0.6f, 0.8f, 0.1f), PxVec3(1.5f, -0.25f, 0.4f) };
    const float linkCfmScale[NB_LINKS] = { 0.125f, 0.5f, -0.75f, 2.25f };
    const float linkInvMass[NB_LINKS]  = { 4.0f, 0.5f, 8.0f, 0.25f };
    const float INV_LENGTH_SCALE = 0.375f;

    // ---- device-side records ----
    std::vector<PxgArticulationLink> links(NB_LINKS);
    std::memset(links.data(), 0, links.size() * sizeof(PxgArticulationLink));
    std::vector<PxgArticulationLinkProp> props(NB_LINKS);
    std::memset(props.data(), 0, props.size() * sizeof(PxgArticulationLinkProp));
    std::vector<ArticulationJointCoreData> jdata(NB_LINKS);
    // POISONED, not zeroed.  nbDof is a PxU8 at offset 4 of an 8-byte struct, so
    // a word read there returns nbDof plus the three bytes after it -- and with
    // those bytes zero the wrong read gives the right answer.  The first sweep
    // had exactly that mutant survive.
    std::memset(jdata.data(), 0xEE, jdata.size() * sizeof(ArticulationJointCoreData));
    for (uint32_t i = 0; i < NB_LINKS; ++i)
    {
        links[i].cfmScale = linkCfmScale[i];
        props[i].invMass = linkInvMass[i];
        jdata[i].nbDof = PxU8(spec[i].dof);
        jdata[i].jointOffset = spec[i].jointOffset;
    }

    std::vector<ArticulationJointCore> joints;
    joints.reserve(NB_LINKS);
    for (uint32_t i = 0; i < NB_LINKS; ++i)
    {
        ArticulationJointCore j(PxTransform(PxVec3(0.0f), pq[i]), PxTransform(cp[i], cq[i]));
        j.jointType = PxU8(spec[i].type);
        j.jCalcUpdateFrames = spec[i].updateFrames;
        for (uint32_t k = 0; k < PxArticulationAxis::eCOUNT; ++k)
            j.dofIds[k] = 0xFF;
        for (uint32_t k = 0; k < 3; ++k)
            j.dofIds[k] = spec[i].dofIds[k];
        joints.push_back(j);
    }

    const size_t MM_BYTES = NB_LINKS * sizeof(SpatialSubspaceMatrix);
    const size_t GUARD = 64;

    PxgDevicePtr dLinks = 0, dProps = 0, dJoints = 0, dJData = 0;
    PxgDevicePtr dMM = 0, dRQ = 0, dOff = 0, dScale = 0, dCfms = 0, dArti = 0;
    shim.memAlloc(&dLinks, links.size() * sizeof(PxgArticulationLink));
    shim.memAlloc(&dProps, props.size() * sizeof(PxgArticulationLinkProp));
    shim.memAlloc(&dJoints, NB_LINKS * sizeof(ArticulationJointCore));
    shim.memAlloc(&dJData, jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memAlloc(&dMM, MM_BYTES + GUARD);
    shim.memAlloc(&dRQ, NB_LINKS * sizeof(PxQuat) + GUARD);
    shim.memAlloc(&dOff, NB_LINKS * 4 + GUARD);
    shim.memAlloc(&dScale, NB_LINKS * 4 + GUARD);
    shim.memAlloc(&dCfms, NB_LINKS * 4 + GUARD);
    shim.memAlloc(&dArti, sizeof(PxgArticulation));
    if (!dLinks || !dProps || !dJoints || !dJData || !dMM || !dRQ || !dOff ||
        !dScale || !dCfms || !dArti)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    PxgArticulation rec;
    std::memset(&rec, 0, sizeof(rec));
    rec.data.numLinks = PxU16(NB_LINKS);
    rec.data.numJointDofs = PxU16(TOTAL_DOFS);
    rec.links = reinterpret_cast<PxgArticulationLink*>(dLinks);
    rec.linkProps = reinterpret_cast<PxgArticulationLinkProp*>(dProps);
    rec.joints = reinterpret_cast<ArticulationJointCore*>(dJoints);
    rec.jointData = reinterpret_cast<ArticulationJointCoreData*>(dJData);
    rec.motionMatrix = reinterpret_cast<SpatialSubspaceMatrix*>(dMM);
    rec.relativeQuat = reinterpret_cast<PxQuat*>(dRQ);
    rec.jointOffsets = reinterpret_cast<PxU32*>(dOff);
    rec.cfmScale = reinterpret_cast<PxReal*>(dScale);
    rec.cfms = reinterpret_cast<PxReal*>(dCfms);

    shim.memcpyHtoD(dLinks, links.data(), links.size() * sizeof(PxgArticulationLink));
    shim.memcpyHtoD(dProps, props.data(), props.size() * sizeof(PxgArticulationLinkProp));
    shim.memcpyHtoD(dJData, jdata.data(), jdata.size() * sizeof(ArticulationJointCoreData));

    size_t bad = 0;
    auto fail = [&](const char* what, uint32_t idx, uint32_t got, uint32_t want) {
        if (bad < 12)
            std::printf("  %s[%u]: got 0x%08X want 0x%08X\n", what, idx, got, want);
        ++bad;
    };
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };

    struct Sub { const char* name; bool confiDirty; bool dataDirty; bool fixBase; };
    const Sub subs[] = {
        { "confi+data, free base", true,  true,  false },
        { "confi only, fixed base", true,  false, true  },
        { "data only, free base",   false, true,  false },
        { "neither (early out)",    false, false, false },
    };

    for (const Sub& sub : subs)
    {
        // ---- fresh poisoned destinations and a fresh joint array every run ----
        std::vector<uint8_t> poisonMM(MM_BYTES + GUARD, 0xAB);
        std::vector<uint8_t> poisonRQ(NB_LINKS * sizeof(PxQuat) + GUARD, 0xAB);
        std::vector<uint8_t> poison4(NB_LINKS * 4 + GUARD, 0xAB);
        shim.memcpyHtoD(dMM, poisonMM.data(), poisonMM.size());
        shim.memcpyHtoD(dRQ, poisonRQ.data(), poisonRQ.size());
        shim.memcpyHtoD(dOff, poison4.data(), poison4.size());
        shim.memcpyHtoD(dScale, poison4.data(), poison4.size());
        shim.memcpyHtoD(dCfms, poison4.data(), poison4.size());
        shim.memcpyHtoD(dJoints, joints.data(), NB_LINKS * sizeof(ArticulationJointCore));

        rec.data.flags = PxU8(sub.fixBase ? PxArticulationFlag::eFIX_BASE : 0);
        rec.data.confiDirty = sub.confiDirty;
        rec.data.updateDirty = sub.dataDirty ? 1u : 0u;
        shim.memcpyHtoD(dArti, &rec, sizeof(rec));

        struct Params { uint64_t arti; float invLengthScale; uint32_t pad; } prm = {
            uint64_t(dArti), INV_LENGTH_SCALE, 0
        };
        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiJcalcPhaseA");
        shim.streamSynchronize();

        // ---- the ORACLE: the engine's own functions, on the host ----
        std::vector<uint8_t> wantMM(MM_BYTES + GUARD, 0xAB);
        std::vector<uint8_t> wantRQ(NB_LINKS * sizeof(PxQuat) + GUARD, 0xAB);
        std::vector<uint8_t> wantOff(NB_LINKS * 4 + GUARD, 0xAB);
        std::vector<uint8_t> wantScale(NB_LINKS * 4 + GUARD, 0xAB);
        std::vector<uint8_t> wantCfms(NB_LINKS * 4 + GUARD, 0xAB);
        std::vector<uint8_t> wantFrames(NB_LINKS, 0);
        for (uint32_t i = 0; i < NB_LINKS; ++i)
            wantFrames[i] = spec[i].updateFrames ? 1 : 0;

        if (sub.confiDirty || sub.dataDirty)
        {
            float scale[NB_LINKS];
            std::memcpy(scale, wantScale.data(), NB_LINKS * 4);   // starts as poison
            if (sub.confiDirty)
            {
                scale[0] = sub.fixBase ? 0.0f : linkCfmScale[0] * INV_LENGTH_SCALE;
                std::memcpy(wantScale.data(), &scale[0], 4);
            }
            if (sub.dataDirty)
            {
                const float c0 = sub.fixBase ? 0.0f : linkInvMass[0] * scale[0];
                std::memcpy(wantCfms.data(), &c0, 4);
            }

            for (uint32_t linkID = 1; linkID < NB_LINKS; ++linkID)
            {
                const uint32_t dof = spec[linkID].dof;

                ArticulationJointCore j = joints[linkID];   // a copy: setJointFrame mutates
                Cm::UnAlignedSpatialVector axis[3];
                for (uint32_t k = 0; k < 3; ++k)
                {
                    axis[k] = Cm::UnAlignedSpatialVector::Zero();
                    if (k < dof)
                    {
                        const PxU32 ind = j.dofIds[k];
                        if (ind < 6)
                            axis[k][ind] = 1.0f;
                    }
                }

                if (spec[linkID].updateFrames)
                {
                    Cm::UnAlignedSpatialVector mm[3];
                    for (uint32_t k = 0; k < 3; ++k)
                        mm[k] = Cm::UnAlignedSpatialVector::Zero();
                    PxQuat rq(PxIdentity);
                    j.setJointFrame(mm, axis, rq, dof);
                    std::memcpy(wantRQ.data() + linkID * sizeof(PxQuat), &rq, sizeof(PxQuat));
                    for (uint32_t k = 0; k < 3; ++k)
                        std::memcpy(wantMM.data() + linkID * sizeof(SpatialSubspaceMatrix) +
                                        k * sizeof(Cm::UnAlignedSpatialVector),
                                    &mm[k], sizeof(Cm::UnAlignedSpatialVector));
                    wantFrames[linkID] = 0;   // setJointFrame clears it
                }

                const uint32_t off = spec[linkID].jointOffset;
                std::memcpy(wantOff.data() + linkID * 4, &off, 4);
                if (sub.confiDirty)
                {
                    scale[linkID] = linkCfmScale[linkID] * INV_LENGTH_SCALE;
                    std::memcpy(wantScale.data() + linkID * 4, &scale[linkID], 4);
                }
                if (sub.dataDirty)
                {
                    const float c = linkInvMass[linkID] * scale[linkID];
                    std::memcpy(wantCfms.data() + linkID * 4, &c, 4);
                }
            }
        }

        // ---- compare ----
        auto region = [&](const char* what, PxgDevicePtr p, const std::vector<uint8_t>& want) {
            std::vector<uint8_t> got(want.size(), 0);
            shim.memcpyDtoH(got.data(), p, got.size());
            for (size_t i = 0; i < want.size(); i += 4)
            {
                uint32_t g, w;
                std::memcpy(&g, got.data() + i, 4);
                std::memcpy(&w, want.data() + i, 4);
                if (g != w)
                {
                    if (bad < 12)
                        std::printf("  [%s] %s +%zu: got 0x%08X want 0x%08X\n",
                                    sub.name, what, i, g, w);
                    ++bad;
                    return;
                }
            }
        };
        region("motionMatrix", dMM, wantMM);
        region("relativeQuat", dRQ, wantRQ);
        region("jointOffsets", dOff, wantOff);
        region("cfmScale", dScale, wantScale);
        region("cfms", dCfms, wantCfms);

        // the jCalcUpdateFrames flags, and confiDirty on the record itself
        // read back as raw bytes: ArticulationJointCore has no default ctor, and
        // only one byte per joint is being checked
        std::vector<uint8_t> gotJointBytes(NB_LINKS * sizeof(ArticulationJointCore), 0);
        shim.memcpyDtoH(gotJointBytes.data(), dJoints, gotJointBytes.size());
        for (uint32_t i = 0; i < NB_LINKS; ++i)
        {
            const uint8_t f = gotJointBytes[i * sizeof(ArticulationJointCore) +
                                            offsetof(ArticulationJointCore, jCalcUpdateFrames)];
            if (uint32_t(f ? 1 : 0) != wantFrames[i])
                fail("jCalcUpdateFrames", i, f ? 1u : 0u, wantFrames[i]);
        }

        PxgArticulation gotRec;
        std::memset(&gotRec, 0, sizeof(gotRec));
        shim.memcpyDtoH(&gotRec, dArti, sizeof(gotRec));
        const uint32_t wantConfi = 0;   // always cleared when the phase runs...
        const uint32_t wantConfiFinal = (sub.confiDirty || sub.dataDirty) ? wantConfi
                                                                          : (sub.confiDirty ? 1u : 0u);
        if (uint32_t(gotRec.data.confiDirty ? 1 : 0) != wantConfiFinal)
            fail("confiDirty", 0, gotRec.data.confiDirty ? 1u : 0u, wantConfiFinal);
        // updateDirty must be left ALONE -- the CUDA holds it for phase C
        if (gotRec.data.updateDirty != (sub.dataDirty ? 1u : 0u))
            fail("updateDirty", 0, gotRec.data.updateDirty, sub.dataDirty ? 1u : 0u);
        if (gotRec.data.flags != PxU8(sub.fixBase ? PxArticulationFlag::eFIX_BASE : 0))
            fail("flags (clobbered by the confiDirty byte write)", 0, gotRec.data.flags,
                 sub.fixBase ? 1u : 0u);
        PX_UNUSED(bits);
    }

    PXG_CHECK_MSG(bad == 0, "%zu region(s)/flag(s) wrong", bad);
    std::printf("  jcalc phase A: 4 links (revolute/spherical-2dof/prismatic), jointOffset "
                "0/1/3, 4 dirty-flag combinations -- joint frames bit-exact vs the engine\n");
}

} // namespace
