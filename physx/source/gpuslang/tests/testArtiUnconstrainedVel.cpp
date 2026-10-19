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

// P8.83: the assembled computeUnconstrainedVelocities1TLaunch, on the engine
// launch ABI and the engine launch geometry.
//
// ===========================================================================
// WHAT THIS TEST IS FOR, AND WHAT IT DELIBERATELY DOES NOT RE-CHECK
// ===========================================================================
//
// Eleven tranches each brought one piece of this kernel up with its own fixture
// and its own mutation sweep.  Re-deriving all of their field-level
// expectations here would be a second transcription of the same CUDA -- exactly
// what building the phases as FUNCTIONS was meant to avoid.  So this fixture
// checks what only the WRAPPER can get wrong, plus one witness per composed
// piece to prove that piece ran at all:
//
//   1. the three-hop index chain -- island slot (+ articulationOffset) -> node
//      index -> body sim -> articulationRemapId -> pool record.  None of the
//      hops is the identity and none equals another;
//   2. `solverBodyIndices[nodeIndex] = globalThreadIndex`, the kernel's only
//      write outside the block format, checked against a poisoned array so a
//      write to the wrong slot is a failure and not merely a wrong value;
//   3. THE PER-WARP BASES for all nine block arrays.  Two warps run and a THIRD
//      warp's slice is allocated and left poisoned: every byte of it must come
//      back 0xAB.  That is what catches a base multiplied by the wrong pitch,
//      because the four pitches here are all different (maxLinks 3, maxDofs 2,
//      maxAttachments 3, maxTendonJoints 3 ... and the two product bases,
//      maxSpatialTendons*maxAttachments and maxFixedTendons*maxTendonJoints);
//   4. THE GUARD ORDER, which is the one thing no phase fixture could test --
//      see the four cases below;
//   5. the launch shape: block (32, 2, 1) with 34 articulations, so warp 0
//      takes lanes 0..31 and warp 1 takes lanes 0..1 of the NEXT warp slice.
//      A kernel that ignored threadIdx.y would process 32 articulations and
//      leave two lanes untouched; one that used globalThreadIndex as the lane
//      would index lane 32 and 33 of a 32-lane block array.
//
// ===========================================================================
// THE FOUR GUARD CASES
// ===========================================================================
//
// `refillBlockData` is `artiBlock.mArticulationIndex[lane] != articulationIndex`
// OR forceRecomputeBlockFormat, and `confiDirty` is read ONCE before phase A
// clears it.  The four reachable combinations are given their own lanes:
//
//   lane 0  FULL     confiDirty=1, updateDirty=0,  block index stale
//   lane 1  REFILL   confiDirty=0, updateDirty!=0, block index stale
//   lane 2  REFRESH  confiDirty=0, updateDirty!=0, block index PRE-SEEDED
//   lane 3  CLEAN    confiDirty=0, updateDirty=0,  block index PRE-SEEDED
//
// Lanes 2 and 3 have their block header pre-seeded with mNumLinks = 1 while the
// RECORD says 2.  That is the discriminator for the else-branch: the wrapper
// must take numLinks from the BLOCK, so its link-velocity loop must run over 1
// link and not 2.  A wrapper that always reads the record cannot tell the two
// apart, and neither could any per-phase fixture.
//
// Lane 2 vs lane 3 is separated without transcribing the refresh path: the
// refresh overwrites the root's mAccumulatedPose from the record's
// linkBody2Worlds[0], and the fused velocity loop then copies mAccumulatedPose
// into mPreTransform.  So lane 2's pre-transform must equal the RECORD's root
// pose and lane 3's must equal the PRE-SEEDED one.  Two different values, one
// comparison, no re-derivation.
//
// ===========================================================================
// THE ORACLE SITUATION, STATED PLAINLY
// ===========================================================================
//
// There is none for the composition, in the sense P8.69 and P8.81 had one:
// `computeUnconstrainedVelocitiesInternal1T` is __device__-only.  What carries
// this tranche is (a) the eleven pieces, each swept green on its own, (b) this
// fixture's wrapper-level invariants, (c) the untouched third warp, and (d) the
// live scene -- `PXG_GPU_STEP=1 PXG_SCENE=articulation` drops
// computeUnconstrainedVelocities1TLaunch from the engine's
// "Could not find registered CUDA function" list, 231 names to 230, with the
// 240 steps still completing and no fault under PXG_SYNC_EACH=1.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "PxgArticulationLink.h"
#include "PxgArticulationTendon.h"
#include "PxgBodySim.h"
#include "DyArticulationJointCore.h"
#include "DyArticulationMimicJointCore.h"
#include "DyCpuGpuArticulation.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiUncVelLoader(const char* file) { return loadShaderBlob(file); }

// 130 articulations: THREE blocks of (32, 2, 1), so five warp slices carry work
// (0..4) and the sixth the grid reaches is empty.  The block count is chosen and
// not merely inherited.  With one block `blockIdx.x * blockDim.y` is zero
// whatever the multiplier is, so a wrong warps-per-block is invisible.  With TWO
// blocks a multiplier of 1 maps (block 1, warp 0) onto warp slice 1 -- which
// block 0 already covered with the same data -- so the duplicated work is
// idempotent and the mutant still survives.  Only at three blocks does the wrong
// multiplier leave a HOLE: warp indices 0,1,1,2,2,3 cover threads 0..127 and
// articulations 128 and 129 are never processed at all.
static const uint32_t NB_ARTI = 130;
static const uint32_t WARPS_ALLOC = 7;   // five carry work; warps 5 and 6 stay poison
static const uint32_t WARPS_USED = 5;

// EVERY pitch different, and every one different from every element count.  Four
// of the seven were equal in the first draft of this fixture and three mutants
// survived on the coincidence -- the mimic base built from maxSpatialTendons, the
// attachment pitch built from maxSpatialTendons, and the bitfield base built
// without wordSize (which is 1 unless maxLinks exceeds 64).
static const uint32_t MAX_LINKS = 65;    // > 64, so wordSize is 2 and not 1
static const uint32_t MAX_DOFS = 2;
static const uint32_t MAX_SPATIAL_TENDONS = 4;
static const uint32_t MAX_ATTACHMENTS = 5;
static const uint32_t MAX_FIXED_TENDONS = 6;
static const uint32_t MAX_TENDON_JOINTS = 7;
static const uint32_t MAX_MIMIC_JOINTS = 8;

// what each articulation actually has (all below its pitch, so the unused tail
// of every per-warp slice is a second poison check).  TWO tendons of each kind
// with TWO elements each, because a single tendon leaves the element pitch
// unmultiplied and therefore untested.
static const uint32_t NB_LINKS = 2;      // root + one revolute child
static const uint32_t NB_DOFS = 1;
static const uint32_t NB_SPATIAL = 2;
static const uint32_t NB_ATTACH = 2;
static const uint32_t NB_FIXED = 2;
static const uint32_t NB_TJ = 2;
static const uint32_t NB_MIMIC = 2;

static const uint32_t WORD_SIZE = (MAX_LINKS + 63u) / 64u;   // 2

// CONFI_SEEDED is the case the first draft was missing: confiDirty set while the
// block index MATCHES, so refillBlockData is false and the phase B branch is
// carried by confiDirty alone.  Without it a wrapper that re-reads confiDirty
// after phase A has cleared it behaves identically, because every other
// confiDirty lane also has refillBlockData true.
enum Case { CASE_FULL = 0, CASE_REFILL, CASE_REFRESH, CASE_CLEAN, CASE_CONFI_SEEDED };
static Case caseOf(uint32_t k)
{
    return (k == 1u) ? CASE_REFILL : (k == 2u) ? CASE_REFRESH
         : (k == 3u) ? CASE_CLEAN : (k == 4u) ? CASE_CONFI_SEEDED : CASE_FULL;
}
static bool seededCase(Case c)
{
    return c == CASE_REFRESH || c == CASE_CLEAN || c == CASE_CONFI_SEEDED;
}
// the branch that copies the record into the block format, in initialize mode
static bool fillsBlocks(Case c)
{
    return c == CASE_FULL || c == CASE_REFILL || c == CASE_CONFI_SEEDED;
}

// the pre-seeded block header for the two cases that must NOT be refilled
static const uint32_t SEEDED_NUM_LINKS = 1;
static const uint32_t SEEDED_TOTAL_DOFS = 0;

PXG_TEST(artiUnconstrainedVel)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // engine ABI { 8, 1, 1 } -> 16-byte blob
        { "npArtiJointFrame.spv", "computeUnconstrainedVelocities1TLaunch", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiUncVelLoader);

    auto fval = [](uint32_t k, uint32_t f) {
        return float(1000 * (k + 1)) + 0.03125f * float(f + 1);
    };
    // 5.12: the joint anchors computeLinkRw reads (small, per-component distinct)
    // Every fourth articulation anchors its joints at the body origins: both
    // offsets exactly +0, so computeLinkRw's `PxVec3(0.f) - childOffset` is the
    // IEEE 0 - (+0) = +0 (a literal-zero fold gives -0) and the parent term
    // rotates a zero vector.
    auto zeroAnchor = [](uint32_t k) { return (k % 4u) == 3u; };
    auto jointParentOffset = [zeroAnchor](uint32_t k, uint32_t i) {
        if (zeroAnchor(k))
            return PxVec3(0.0f);
        return PxVec3(0.25f + 0.125f * float(k % 5), -0.5f + 0.0625f * float(i), 0.375f);
    };
    auto jointChildOffset = [zeroAnchor](uint32_t k, uint32_t i) {
        if (zeroAnchor(k))
            return PxVec3(0.0f);
        return PxVec3(-0.125f, 0.75f - 0.0625f * float(k % 3), 0.5f + 0.25f * float(i));
    };

    // ---- the index chain: three hops, none of them the identity ----
    const uint32_t ARTI_OFFSET = 5;
    const uint32_t ISLAND_SLOTS = ARTI_OFFSET + NB_ARTI + 3;
    const uint32_t BODY_SIM_SLOTS = 192;   // > NB_ARTI and coprime with the 7 below,
                                           // so every articulation gets a distinct node
    std::vector<uint32_t> NODE(NB_ARTI), POOL(NB_ARTI);
    for (uint32_t k = 0; k < NB_ARTI; ++k)
    {
        NODE[k] = (k * 7u + 11u) % BODY_SIM_SLOTS;      // every node distinct, never == k
        POOL[k] = (k + 13u) % NB_ARTI;                  // a permutation, never the identity
    }

    std::vector<PxNodeIndex> island(ISLAND_SLOTS);
    for (uint32_t i = 0; i < ISLAND_SLOTS; ++i)
        island[i] = PxNodeIndex(0x0BAD00u + i);         // decoys outside the range
    for (uint32_t k = 0; k < NB_ARTI; ++k)
        island[ARTI_OFFSET + k] = PxNodeIndex(NODE[k]);

    std::vector<PxgBodySim> bodySims(BODY_SIM_SLOTS);
    std::memset(bodySims.data(), 0xEE, bodySims.size() * sizeof(PxgBodySim));
    for (uint32_t i = 0; i < BODY_SIM_SLOTS; ++i)
        bodySims[i].articulationRemapId = 0xDEAD0000u + i;
    for (uint32_t k = 0; k < NB_ARTI; ++k)
    {
        bodySims[NODE[k]].articulationRemapId = POOL[k];
        // .z is the sleep threshold; .x/.y are plausible decoys for a wrong offset
        bodySims[NODE[k]].freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex =
            make_float4(fval(k, 90), fval(k, 91), fval(k, 92), fval(k, 93));
    }

    // ---- per-articulation records, indexed by POOL[k] ----
    // the RECORD arrays only need the links an articulation actually has; the
    // pitches above are BLOCK pitches and would waste 65 slots per articulation
    const size_t L_STRIDE = NB_LINKS;
    std::vector<PxgArticulationLink> links(size_t(NB_ARTI) * L_STRIDE);
    std::memset(links.data(), 0xEE, links.size() * sizeof(PxgArticulationLink));
    std::vector<ArticulationJointCoreData> jdata(size_t(NB_ARTI) * L_STRIDE);
    std::memset(jdata.data(), 0xEE, jdata.size() * sizeof(ArticulationJointCoreData));
    std::vector<PxTransform> body2Worlds(size_t(NB_ARTI) * L_STRIDE);
    std::vector<PxQuat> relativeQuats(size_t(NB_ARTI) * L_STRIDE);
    std::vector<PxU32> parents(size_t(NB_ARTI) * L_STRIDE, 0u);
    std::vector<PxU32> jointOffsets(size_t(NB_ARTI) * L_STRIDE, 0u);
    std::vector<PxReal> cfmScale(size_t(NB_ARTI) * L_STRIDE, 0.0f);
    std::vector<PxReal> cfms(size_t(NB_ARTI) * L_STRIDE, 0.0f);
    std::vector<PxgArticulationLinkProp> props(size_t(NB_ARTI) * L_STRIDE);
    std::memset(props.data(), 0, props.size() * sizeof(PxgArticulationLinkProp));
    std::vector<PxU32> pathToRoot(size_t(NB_ARTI) * L_STRIDE, 0u);
    std::vector<ArticulationJointCore> joints;
    joints.reserve(size_t(NB_ARTI) * L_STRIDE);

    const size_t D_STRIDE = NB_DOFS;
    std::vector<PxReal> jointPositions(size_t(NB_ARTI) * D_STRIDE, 0.0f);
    std::vector<PxReal> jointVelocities(size_t(NB_ARTI) * D_STRIDE, 0.0f);
    std::vector<PxReal> jointTargetPos(size_t(NB_ARTI) * D_STRIDE, 0.0f);
    std::vector<PxReal> jointTargetVel(size_t(NB_ARTI) * D_STRIDE, 0.0f);
    std::vector<Cm::UnAlignedSpatialVector> jointAxis(size_t(NB_ARTI) * D_STRIDE);
    std::memset(jointAxis.data(), 0, jointAxis.size() * sizeof(Cm::UnAlignedSpatialVector));
    std::vector<SpatialSubspaceMatrix> motionMatrix(size_t(NB_ARTI) * L_STRIDE);
    std::memset(motionMatrix.data(), 0, motionMatrix.size() * sizeof(SpatialSubspaceMatrix));
    std::vector<Cm::UnAlignedSpatialVector> motionVelocities(size_t(NB_ARTI) * L_STRIDE);
    std::memset(motionVelocities.data(), 0, motionVelocities.size() * sizeof(Cm::UnAlignedSpatialVector));
    std::vector<Cm::UnAlignedSpatialVector> rootPreMotion(NB_ARTI);
    std::memset(rootPreMotion.data(), 0, rootPreMotion.size() * sizeof(Cm::UnAlignedSpatialVector));

    const PxTransform idT(PxIdentity);
    for (uint32_t k = 0; k < NB_ARTI; ++k)
    {
        for (uint32_t i = 0; i < L_STRIDE; ++i)
        {
            const size_t x = size_t(k) * L_STRIDE + i;
            PxgArticulationLink& lk = links[x];
            lk.linearDamping = fval(k, 10 + i);
            lk.angularDamping = fval(k, 20 + i);
            lk.maxLinearVelocitySq = fval(k, 30 + i);
            lk.maxAngularVelocitySq = fval(k, 40 + i);
            lk.childrenOffset = 4000u + 10u * k + i;
            lk.numChildren = PxU16((i == 0) ? 1u : 0u);
            lk.disableGravity = ((k + i) & 1) != 0;
            lk.retainsAccelerations = ((k + i) & 2) != 0;
            lk.cfmScale = fval(k, 50 + i);
            lk.initialAngVel = PxVec3(fval(k, 60 + i), fval(k, 61 + i), fval(k, 62 + i));
            lk.initialLinVel = PxVec3(fval(k, 63 + i), fval(k, 64 + i), fval(k, 65 + i));
            lk.penBiasClamp = fval(k, 66 + i);
            lk.invMass = fval(k, 67 + i);
            lk.offsetSlop = fval(k, 68 + i);
            lk.pathToRootOffset = uint32_t(x);
            lk.numPathToRoot = PxU16(i);           // root has none, link 1 has itself
            pathToRoot[x] = i;

            jdata[x].nbDof = PxU8((i == 0) ? 0u : NB_DOFS);
            jdata[x].jointOffset = (i == 0) ? 0u : 0u;   // the single dof lives at slot 0

            // the poses differ per link with a quadratic term, so rw's three
            // components are never equal to one another (P8.81's lesson)
            body2Worlds[x] = PxTransform(
                PxVec3(fval(k, 70) * 3.0f + 0.75f * float(i * i),
                       fval(k, 71) * 11.0f - 0.4f * float(i),
                       fval(k, 72) * 7.0f + 1.5f * float(i)),
                PxQuat(0.2f, -0.4f, 0.5f, 0.7f).getNormalized());
            relativeQuats[x] = PxQuat(0.1f, 0.2f, -0.3f, 0.9f).getNormalized();
            parents[x] = (i == 0) ? 0u : (i - 1u);
            props[x].invInertia = PxVec3(fval(k, 80), fval(k, 81), fval(k, 82));
            props[x].invMass = fval(k, 83);

            // 5.12: Rw = Dy::computeLinkRw over the joint's link-local anchors, so
            // the frames carry distinct per-component translations (rotations
            // stay identity: the motion-matrix witnesses are unchanged)
            const PxTransform parentFrame(jointParentOffset(k, i), PxQuat(PxIdentity));
            const PxTransform childFrame(jointChildOffset(k, i), PxQuat(PxIdentity));
            ArticulationJointCore j(parentFrame, childFrame);
            j.jointType = PxU8(PxArticulationJointType::eREVOLUTE);
            j.dofIds[0] = PxU8(PxArticulationAxis::eTWIST);
            j.jCalcUpdateFrames = true;
            j.maxJointVelocity[0] = fval(k, 84);
            joints.push_back(j);
        }
        jointPositions[size_t(k) * D_STRIDE] = fval(k, 100);
        jointVelocities[size_t(k) * D_STRIDE] = fval(k, 101);
        jointTargetPos[size_t(k) * D_STRIDE] = fval(k, 102);
        jointTargetVel[size_t(k) * D_STRIDE] = fval(k, 103);
    }

    // ---- tendons and mimic joints: one of each, so every per-warp base is
    // exercised by a real write rather than by an empty loop ----
    std::vector<PxGpuSpatialTendonData> sParams(size_t(NB_ARTI) * NB_SPATIAL);
    std::vector<PxGpuFixedTendonData> fParams(size_t(NB_ARTI) * NB_FIXED);
    std::vector<PxgArticulationTendonElementFixedData> sFixed(size_t(NB_ARTI) * NB_SPATIAL * NB_ATTACH);
    std::vector<PxGpuTendonAttachmentData> sMod(size_t(NB_ARTI) * NB_SPATIAL * NB_ATTACH);
    std::vector<PxgArticulationTendonElementFixedData> fFixed(size_t(NB_ARTI) * NB_FIXED * NB_TJ);
    std::vector<PxGpuTendonJointCoefficientData> fMod(size_t(NB_ARTI) * NB_FIXED * NB_TJ);
    std::vector<ArticulationMimicJointCore> mimic(size_t(NB_ARTI) * NB_MIMIC);
    std::memset(sParams.data(), 0, sParams.size() * sizeof(PxGpuSpatialTendonData));
    std::memset(fParams.data(), 0, fParams.size() * sizeof(PxGpuFixedTendonData));
    std::memset(sFixed.data(), 0, sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    std::memset(sMod.data(), 0, sMod.size() * sizeof(PxGpuTendonAttachmentData));
    std::memset(fFixed.data(), 0, fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    std::memset(fMod.data(), 0, fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
    std::memset(mimic.data(), 0, mimic.size() * sizeof(ArticulationMimicJointCore));
    // the element index inside a tendon's slice is `t * maxElements + j`, so the
    // second tendon is the one that makes the pitch observable -- with a single
    // tendon everything lands at slot 0 whatever the pitch is
    auto sIdx = [](uint32_t k, uint32_t t) { return size_t(k) * NB_SPATIAL + t; };
    auto sElem = [](uint32_t k, uint32_t t, uint32_t j) {
        return (size_t(k) * NB_SPATIAL + t) * NB_ATTACH + j;
    };
    auto fIdx = [](uint32_t k, uint32_t t) { return size_t(k) * NB_FIXED + t; };
    auto fElem = [](uint32_t k, uint32_t t, uint32_t j) {
        return (size_t(k) * NB_FIXED + t) * NB_TJ + j;
    };
    for (uint32_t k = 0; k < NB_ARTI; ++k)
    {
        for (uint32_t t = 0; t < NB_SPATIAL; ++t)
        {
            PxGpuSpatialTendonData& p = sParams[sIdx(k, t)];
            p.stiffness = fval(k, 110 + t * 10);
            p.damping = fval(k, 111 + t * 10);
            p.limitStiffness = fval(k, 112 + t * 10);
            p.offset = fval(k, 113 + t * 10);
            for (uint32_t j = 0; j < NB_ATTACH; ++j)
            {
                PxGpuTendonAttachmentData& m = sMod[sElem(k, t, j)];
                m.relativeOffset = PxVec3(fval(k, 150 + t * 20 + j),
                                          fval(k, 151 + t * 20 + j), fval(k, 152 + t * 20 + j));
                m.restLength = fval(k, 153 + t * 20 + j);
                m.coefficient = fval(k, 154 + t * 20 + j);
                m.lowLimit = fval(k, 155 + t * 20 + j);
                m.highLimit = fval(k, 156 + t * 20 + j);
                sFixed[sElem(k, t, j)].linkInd = PxU16(1u + t * 2u + j);
            }
        }
        for (uint32_t t = 0; t < NB_FIXED; ++t)
        {
            PxGpuFixedTendonData& p = fParams[fIdx(k, t)];
            p.stiffness = fval(k, 130 + t * 10);
            p.damping = fval(k, 131 + t * 10);
            p.limitStiffness = fval(k, 132 + t * 10);
            p.offset = fval(k, 133 + t * 10);
            p.lowLimit = fval(k, 134 + t * 10);
            p.highLimit = fval(k, 135 + t * 10);
            p.restLength = fval(k, 136 + t * 10);
            for (uint32_t j = 0; j < NB_TJ; ++j)
            {
                PxGpuTendonJointCoefficientData& m = fMod[fElem(k, t, j)];
                m.coefficient = fval(k, 200 + t * 20 + j);
                m.recipCoefficient = fval(k, 201 + t * 20 + j);
                m.axis = 1u;
                fFixed[fElem(k, t, j)].linkInd = PxU16(1u);
            }
        }
        for (uint32_t i = 0; i < NB_MIMIC; ++i)
        {
            ArticulationMimicJointCore& m = mimic[size_t(k) * NB_MIMIC + i];
            m.linkA = 0u;
            m.linkB = 1u;
            m.axisA = PxArticulationAxis::eTWIST;
            m.axisB = PxArticulationAxis::eTWIST;
            m.gearRatio = fval(k, 140 + i * 4);
            m.offset = fval(k, 141 + i * 4);
            m.naturalFrequency = fval(k, 142 + i * 4);
            m.dampingRatio = fval(k, 143 + i * 4);
        }
    }

    // ---- device side ----
    const size_t AB_SZ = sizeof(PxgArticulationBlockData);
    const size_t LK_SZ = sizeof(PxgArticulationBlockLinkData);
    const size_t DF_SZ = sizeof(PxgArticulationBlockDofData);
    const size_t ST_SZ = sizeof(PxgArticulationBlockSpatialTendonData);
    const size_t AT_SZ = sizeof(PxgArticulationBlockAttachmentData);
    const size_t FT_SZ = sizeof(PxgArticulationBlockFixedTendonData);
    const size_t TJ_SZ = sizeof(PxgArticulationBlockTendonJointData);
    const size_t MJ_SZ = sizeof(PxgArticulationBlockMimicJointData);
    const size_t BF_SZ = sizeof(PxgArticulationBitFieldData);

    const size_t AB_WARP = AB_SZ;
    const size_t LK_WARP = size_t(MAX_LINKS) * LK_SZ;
    const size_t DF_WARP = size_t(MAX_DOFS) * DF_SZ;
    const size_t ST_WARP = size_t(MAX_SPATIAL_TENDONS) * ST_SZ;
    const size_t AT_WARP = size_t(MAX_SPATIAL_TENDONS) * MAX_ATTACHMENTS * AT_SZ;
    const size_t FT_WARP = size_t(MAX_FIXED_TENDONS) * FT_SZ;
    const size_t TJ_WARP = size_t(MAX_FIXED_TENDONS) * MAX_TENDON_JOINTS * TJ_SZ;
    const size_t MJ_WARP = size_t(MAX_MIMIC_JOINTS) * MJ_SZ;
    const size_t BF_WARP = size_t(MAX_LINKS) * WORD_SIZE * BF_SZ;

    PxgDevicePtr dIsland = 0, dBodySims = 0, dSolverBodyIdx = 0, dDesc = 0, dArtis = 0;
    PxgDevicePtr dLinks = 0, dJData = 0, dJoints = 0, dB2W = 0, dRQ = 0, dParents = 0;
    PxgDevicePtr dJointOffsets = 0, dCfmScale = 0, dCfms = 0, dProps = 0, dPathToRoot = 0;
    PxgDevicePtr dJPos = 0, dJVel = 0, dJTPos = 0, dJTVel = 0, dJAxis = 0, dMM = 0;
    PxgDevicePtr dMotionVel = 0, dRootPre = 0;
    PxgDevicePtr dSParams = 0, dFParams = 0, dSFixed = 0, dSMod = 0, dFFixed = 0, dFMod = 0;
    PxgDevicePtr dMimic = 0, dSTend = 0, dFTend = 0;
    PxgDevicePtr dAB = 0, dLK = 0, dDF = 0, dST = 0, dAT = 0, dFT = 0, dTJ = 0, dMJ = 0, dBF = 0;

    shim.memAlloc(&dIsland, island.size() * sizeof(PxNodeIndex));
    shim.memAlloc(&dBodySims, bodySims.size() * sizeof(PxgBodySim));
    shim.memAlloc(&dSolverBodyIdx, size_t(BODY_SIM_SLOTS) * sizeof(PxU32));
    shim.memAlloc(&dDesc, sizeof(PxgArticulationCoreDesc));
    shim.memAlloc(&dArtis, size_t(NB_ARTI) * sizeof(PxgArticulation));
    shim.memAlloc(&dLinks, links.size() * sizeof(PxgArticulationLink));
    shim.memAlloc(&dJData, jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memAlloc(&dJoints, joints.size() * sizeof(ArticulationJointCore));
    shim.memAlloc(&dB2W, body2Worlds.size() * sizeof(PxTransform));
    shim.memAlloc(&dRQ, relativeQuats.size() * sizeof(PxQuat));
    shim.memAlloc(&dParents, parents.size() * sizeof(PxU32));
    shim.memAlloc(&dJointOffsets, jointOffsets.size() * sizeof(PxU32));
    shim.memAlloc(&dCfmScale, cfmScale.size() * sizeof(PxReal));
    shim.memAlloc(&dCfms, cfms.size() * sizeof(PxReal));
    shim.memAlloc(&dProps, props.size() * sizeof(PxgArticulationLinkProp));
    shim.memAlloc(&dPathToRoot, pathToRoot.size() * sizeof(PxU32));
    shim.memAlloc(&dJPos, jointPositions.size() * sizeof(PxReal));
    shim.memAlloc(&dJVel, jointVelocities.size() * sizeof(PxReal));
    shim.memAlloc(&dJTPos, jointTargetPos.size() * sizeof(PxReal));
    shim.memAlloc(&dJTVel, jointTargetVel.size() * sizeof(PxReal));
    shim.memAlloc(&dJAxis, jointAxis.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dMM, motionMatrix.size() * sizeof(SpatialSubspaceMatrix));
    shim.memAlloc(&dMotionVel, motionVelocities.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dRootPre, rootPreMotion.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dSParams, sParams.size() * sizeof(PxGpuSpatialTendonData));
    shim.memAlloc(&dFParams, fParams.size() * sizeof(PxGpuFixedTendonData));
    shim.memAlloc(&dSFixed, sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memAlloc(&dSMod, sMod.size() * sizeof(PxGpuTendonAttachmentData));
    shim.memAlloc(&dFFixed, fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memAlloc(&dFMod, fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
    shim.memAlloc(&dMimic, mimic.size() * sizeof(ArticulationMimicJointCore));
    shim.memAlloc(&dSTend, size_t(NB_ARTI) * NB_SPATIAL * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dFTend, size_t(NB_ARTI) * NB_FIXED * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dAB, WARPS_ALLOC * AB_WARP);
    shim.memAlloc(&dLK, WARPS_ALLOC * LK_WARP);
    shim.memAlloc(&dDF, WARPS_ALLOC * DF_WARP);
    shim.memAlloc(&dST, WARPS_ALLOC * ST_WARP);
    shim.memAlloc(&dAT, WARPS_ALLOC * AT_WARP);
    shim.memAlloc(&dFT, WARPS_ALLOC * FT_WARP);
    shim.memAlloc(&dTJ, WARPS_ALLOC * TJ_WARP);
    shim.memAlloc(&dMJ, WARPS_ALLOC * MJ_WARP);
    shim.memAlloc(&dBF, WARPS_ALLOC * BF_WARP);

    if (!dIsland || !dBodySims || !dSolverBodyIdx || !dDesc || !dArtis || !dLinks || !dJData ||
        !dJoints || !dB2W || !dRQ || !dParents || !dJointOffsets || !dCfmScale || !dCfms ||
        !dProps || !dPathToRoot || !dJPos || !dJVel || !dJTPos || !dJTVel || !dJAxis || !dMM ||
        !dMotionVel || !dRootPre || !dSParams || !dFParams || !dSFixed || !dSMod || !dFFixed ||
        !dFMod || !dMimic || !dSTend || !dFTend || !dAB || !dLK || !dDF || !dST || !dAT ||
        !dFT || !dTJ || !dMJ || !dBF)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // tendon records carry DEVICE pointers into the element arrays
    std::vector<PxgArticulationTendon> sTend(size_t(NB_ARTI) * NB_SPATIAL);
    std::vector<PxgArticulationTendon> fTend(size_t(NB_ARTI) * NB_FIXED);
    for (uint32_t k = 0; k < NB_ARTI; ++k)
    {
        for (uint32_t t = 0; t < NB_SPATIAL; ++t)
        {
            PxgArticulationTendon& tn = sTend[sIdx(k, t)];
            tn.mFixedElements = reinterpret_cast<void*>(
                dSFixed + PxgDevicePtr(sElem(k, t, 0)
                                       * sizeof(PxgArticulationTendonElementFixedData)));
            tn.mModElements = reinterpret_cast<void*>(
                dSMod + PxgDevicePtr(sElem(k, t, 0) * sizeof(PxGpuTendonAttachmentData)));
            tn.mNbElements = NB_ATTACH;
        }
        for (uint32_t t = 0; t < NB_FIXED; ++t)
        {
            PxgArticulationTendon& tn = fTend[fIdx(k, t)];
            tn.mFixedElements = reinterpret_cast<void*>(
                dFFixed + PxgDevicePtr(fElem(k, t, 0)
                                       * sizeof(PxgArticulationTendonElementFixedData)));
            tn.mModElements = reinterpret_cast<void*>(
                dFMod + PxgDevicePtr(fElem(k, t, 0) * sizeof(PxGpuTendonJointCoefficientData)));
            tn.mNbElements = NB_TJ;
        }
    }

    // the pool: articulation k lives at slot POOL[k]
    std::vector<PxgArticulation> recs(NB_ARTI);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t k = 0; k < NB_ARTI; ++k)
    {
        PxgArticulation& r = recs[POOL[k]];
        const size_t lo = size_t(k) * L_STRIDE;
        const size_t dofo = size_t(k) * D_STRIDE;

        r.data.index = POOL[k];
        r.data.numLinks = PxU16(NB_LINKS);
        r.data.numJointDofs = PxU16(NB_DOFS);
        r.data.numSpatialTendons = PxU16(NB_SPATIAL);
        r.data.numFixedTendons = PxU16(NB_FIXED);
        r.data.numMimicJoints = PxU16(NB_MIMIC);
        r.data.numPathToRoots = NB_LINKS;
        r.data.flags = PxU8(0x20u | (k & 1u));   // a distinct bit plus a per-lane one
        const Case c = caseOf(k);
        r.data.confiDirty = (c == CASE_FULL || c == CASE_CONFI_SEEDED);
        // the REFRESH lane carries the tendon and mimic bits as well, because the
        // three update copies are guarded by them AFTER jcalcRefreshBlocks has
        // zeroed updateDirty -- a wrapper that re-reads the field there sees 0
        // and silently skips all three
        r.data.updateDirty =
            (c == CASE_REFILL)
                ? PxU32(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS)
                : (c == CASE_REFRESH)
                      ? PxU32(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS
                              | Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON
                              | Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON
                              | Dy::ArticulationDirtyFlag::eDIRTY_MIMIC_JOINT)
                      : 0u;
        r.data.gpuDirtyFlag = 0u;

        r.links = reinterpret_cast<PxgArticulationLink*>(
            dLinks + PxgDevicePtr(lo * sizeof(PxgArticulationLink)));
        r.jointData = reinterpret_cast<ArticulationJointCoreData*>(
            dJData + PxgDevicePtr(lo * sizeof(ArticulationJointCoreData)));
        r.joints = reinterpret_cast<ArticulationJointCore*>(
            dJoints + PxgDevicePtr(lo * sizeof(ArticulationJointCore)));
        r.linkBody2Worlds = reinterpret_cast<PxTransform*>(
            dB2W + PxgDevicePtr(lo * sizeof(PxTransform)));
        r.relativeQuat = reinterpret_cast<PxQuat*>(dRQ + PxgDevicePtr(lo * sizeof(PxQuat)));
        r.parents = reinterpret_cast<PxU32*>(dParents + PxgDevicePtr(lo * sizeof(PxU32)));
        r.jointOffsets = reinterpret_cast<PxU32*>(dJointOffsets + PxgDevicePtr(lo * sizeof(PxU32)));
        r.cfmScale = reinterpret_cast<PxReal*>(dCfmScale + PxgDevicePtr(lo * sizeof(PxReal)));
        r.cfms = reinterpret_cast<PxReal*>(dCfms + PxgDevicePtr(lo * sizeof(PxReal)));
        r.linkProps = reinterpret_cast<PxgArticulationLinkProp*>(
            dProps + PxgDevicePtr(lo * sizeof(PxgArticulationLinkProp)));
        r.pathToRoot = reinterpret_cast<PxU32*>(dPathToRoot);
        r.jointPositions = reinterpret_cast<PxReal*>(dJPos + PxgDevicePtr(dofo * sizeof(PxReal)));
        r.jointVelocities = reinterpret_cast<PxReal*>(dJVel + PxgDevicePtr(dofo * sizeof(PxReal)));
        r.jointTargetPositions =
            reinterpret_cast<PxReal*>(dJTPos + PxgDevicePtr(dofo * sizeof(PxReal)));
        r.jointTargetVelocities =
            reinterpret_cast<PxReal*>(dJTVel + PxgDevicePtr(dofo * sizeof(PxReal)));
        // (5.12: no record jointAxis / rootPreMotionVelocity; the two pools
        // below stay allocated and must come back as uploaded)
        r.motionMatrix = reinterpret_cast<SpatialSubspaceMatrix*>(
            dMM + PxgDevicePtr(lo * sizeof(SpatialSubspaceMatrix)));
        r.motionVelocities = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dMotionVel + PxgDevicePtr(lo * sizeof(Cm::UnAlignedSpatialVector)));
        r.spatialTendonParams = reinterpret_cast<PxGpuSpatialTendonData*>(
            dSParams + PxgDevicePtr(sIdx(k, 0) * sizeof(PxGpuSpatialTendonData)));
        r.spatialTendons = reinterpret_cast<PxgArticulationTendon*>(
            dSTend + PxgDevicePtr(sIdx(k, 0) * sizeof(PxgArticulationTendon)));
        r.fixedTendonParams = reinterpret_cast<PxGpuFixedTendonData*>(
            dFParams + PxgDevicePtr(fIdx(k, 0) * sizeof(PxGpuFixedTendonData)));
        r.fixedTendons = reinterpret_cast<PxgArticulationTendon*>(
            dFTend + PxgDevicePtr(fIdx(k, 0) * sizeof(PxgArticulationTendon)));
        r.mimicJointCores = reinterpret_cast<ArticulationMimicJointCore*>(
            dMimic + PxgDevicePtr(size_t(k) * NB_MIMIC * sizeof(ArticulationMimicJointCore)));
    }

    // The kernel MUTATES the record side -- phase A clears confiDirty and the
    // joints' jCalcUpdateFrames, the refresh clears updateDirty, phase A writes
    // the motion matrices and quats -- so every pass re-uploads the whole
    // record-side image rather than running on what the previous pass left.
    auto uploadRecords = [&]() {
        shim.memcpyHtoD(dIsland, island.data(), island.size() * sizeof(PxNodeIndex));
        shim.memcpyHtoD(dBodySims, bodySims.data(), bodySims.size() * sizeof(PxgBodySim));
        shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));
        shim.memcpyHtoD(dLinks, links.data(), links.size() * sizeof(PxgArticulationLink));
        shim.memcpyHtoD(dJData, jdata.data(), jdata.size() * sizeof(ArticulationJointCoreData));
        shim.memcpyHtoD(dJoints, joints.data(), joints.size() * sizeof(ArticulationJointCore));
        shim.memcpyHtoD(dB2W, body2Worlds.data(), body2Worlds.size() * sizeof(PxTransform));
        shim.memcpyHtoD(dRQ, relativeQuats.data(), relativeQuats.size() * sizeof(PxQuat));
        shim.memcpyHtoD(dParents, parents.data(), parents.size() * sizeof(PxU32));
        shim.memcpyHtoD(dJointOffsets, jointOffsets.data(), jointOffsets.size() * sizeof(PxU32));
        shim.memcpyHtoD(dCfmScale, cfmScale.data(), cfmScale.size() * sizeof(PxReal));
        shim.memcpyHtoD(dCfms, cfms.data(), cfms.size() * sizeof(PxReal));
        shim.memcpyHtoD(dProps, props.data(), props.size() * sizeof(PxgArticulationLinkProp));
        shim.memcpyHtoD(dPathToRoot, pathToRoot.data(), pathToRoot.size() * sizeof(PxU32));
        shim.memcpyHtoD(dJPos, jointPositions.data(), jointPositions.size() * sizeof(PxReal));
        shim.memcpyHtoD(dJVel, jointVelocities.data(), jointVelocities.size() * sizeof(PxReal));
        shim.memcpyHtoD(dJTPos, jointTargetPos.data(), jointTargetPos.size() * sizeof(PxReal));
        shim.memcpyHtoD(dJTVel, jointTargetVel.data(), jointTargetVel.size() * sizeof(PxReal));
        shim.memcpyHtoD(dJAxis, jointAxis.data(),
                        jointAxis.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyHtoD(dMM, motionMatrix.data(),
                        motionMatrix.size() * sizeof(SpatialSubspaceMatrix));
        shim.memcpyHtoD(dMotionVel, motionVelocities.data(),
                        motionVelocities.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyHtoD(dRootPre, rootPreMotion.data(),
                        rootPreMotion.size() * sizeof(Cm::UnAlignedSpatialVector));
        shim.memcpyHtoD(dSParams, sParams.data(), sParams.size() * sizeof(PxGpuSpatialTendonData));
        shim.memcpyHtoD(dFParams, fParams.data(), fParams.size() * sizeof(PxGpuFixedTendonData));
        shim.memcpyHtoD(dSFixed, sFixed.data(),
                        sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
        shim.memcpyHtoD(dSMod, sMod.data(), sMod.size() * sizeof(PxGpuTendonAttachmentData));
        shim.memcpyHtoD(dFFixed, fFixed.data(),
                        fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
        shim.memcpyHtoD(dFMod, fMod.data(), fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
        shim.memcpyHtoD(dMimic, mimic.data(), mimic.size() * sizeof(ArticulationMimicJointCore));
        shim.memcpyHtoD(dSTend, sTend.data(), sTend.size() * sizeof(PxgArticulationTendon));
        shim.memcpyHtoD(dFTend, fTend.data(), fTend.size() * sizeof(PxgArticulationTendon));
    };

    // ---- the block arrays: poison, with the three pre-seeded lanes on top ----
    std::vector<uint8_t> abImg(size_t(WARPS_ALLOC) * AB_WARP, 0xAB);
    std::vector<uint8_t> lkImg(size_t(WARPS_ALLOC) * LK_WARP, 0xAB);
    auto seedU32 = [](std::vector<uint8_t>& img, size_t off, uint32_t v) {
        std::memcpy(img.data() + off, &v, 4);
    };
    // lanes 2, 3 and 4 of warp 0 get a MATCHING articulation index (so
    // refillBlockData is false) and a SMALLER link count than the record's.
    // That second half is the else-branch discriminator: the wrapper must take
    // numLinks from the BLOCK when it does not refill.
    for (uint32_t k = 2; k <= 4; ++k)
    {
        seedU32(abImg, offsetof(PxgArticulationBlockData, mArticulationIndex) + size_t(k) * 4,
                POOL[k]);
        seedU32(abImg, offsetof(PxgArticulationBlockData, mNumLinks) + size_t(k) * 4,
                SEEDED_NUM_LINKS);
        seedU32(abImg, offsetof(PxgArticulationBlockData, mTotalDofs) + size_t(k) * 4,
                SEEDED_TOTAL_DOFS);
        // a root pose the refresh path must overwrite and the clean path must not
        const size_t pose = offsetof(PxgArticulationBlockLinkData, mAccumulatedPose);
        for (uint32_t c = 0; c < 4; ++c)
            seedU32(lkImg, pose + size_t(k) * 16 + size_t(c) * 4,
                    0x3F000000u + k * 0x00010000u + c);            // q
        for (uint32_t c = 0; c < 4; ++c)
            seedU32(lkImg, pose + 512 + size_t(k) * 16 + size_t(c) * 4,
                    0x3E800000u + k * 0x00010000u + c);            // p (incl. p.w)
    }

    const std::vector<uint8_t> dfImg(size_t(WARPS_ALLOC) * DF_WARP, 0xAB);
    const std::vector<uint8_t> stImg(size_t(WARPS_ALLOC) * ST_WARP, 0xAB);
    const std::vector<uint8_t> atImg(size_t(WARPS_ALLOC) * AT_WARP, 0xAB);
    const std::vector<uint8_t> ftImg(size_t(WARPS_ALLOC) * FT_WARP, 0xAB);
    const std::vector<uint8_t> tjImg(size_t(WARPS_ALLOC) * TJ_WARP, 0xAB);
    const std::vector<uint8_t> mjImg(size_t(WARPS_ALLOC) * MJ_WARP, 0xAB);
    const std::vector<uint8_t> bfImg(size_t(WARPS_ALLOC) * BF_WARP, 0xAB);
    const std::vector<uint8_t> sbiImg(size_t(BODY_SIM_SLOTS) * sizeof(PxU32), 0xAB);

    auto uploadBlocks = [&]() {
        shim.memcpyHtoD(dAB, abImg.data(), abImg.size());
        shim.memcpyHtoD(dLK, lkImg.data(), lkImg.size());
        shim.memcpyHtoD(dDF, dfImg.data(), dfImg.size());
        shim.memcpyHtoD(dST, stImg.data(), stImg.size());
        shim.memcpyHtoD(dAT, atImg.data(), atImg.size());
        shim.memcpyHtoD(dFT, ftImg.data(), ftImg.size());
        shim.memcpyHtoD(dTJ, tjImg.data(), tjImg.size());
        shim.memcpyHtoD(dMJ, mjImg.data(), mjImg.size());
        shim.memcpyHtoD(dBF, bfImg.data(), bfImg.size());
        shim.memcpyHtoD(dSolverBodyIdx, sbiImg.data(), sbiImg.size());
    };

    // ---- the descriptor ----
    PxgArticulationCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dBodySims);
    desc.islandNodeIndices = reinterpret_cast<PxNodeIndex*>(dIsland);
    desc.solverBodyIndices = reinterpret_cast<PxU32*>(dSolverBodyIdx);
    desc.articulations = reinterpret_cast<PxgArticulation*>(dArtis);
    desc.articulationOffset = ARTI_OFFSET;
    desc.nbArticulations = NB_ARTI;
    desc.invLengthScale = 0.5f;
    desc.mArticulationBlocks = reinterpret_cast<PxgArticulationBlockData*>(dAB);
    desc.mArticulationLinkBlocks = reinterpret_cast<PxgArticulationBlockLinkData*>(dLK);
    desc.mArticulationDofBlocks = reinterpret_cast<PxgArticulationBlockDofData*>(dDF);
    desc.mArticulationSpatialTendonBlocks =
        reinterpret_cast<PxgArticulationBlockSpatialTendonData*>(dST);
    desc.mArticulationAttachmentBlocks = reinterpret_cast<PxgArticulationBlockAttachmentData*>(dAT);
    desc.mArticulationFixedTendonBlocks =
        reinterpret_cast<PxgArticulationBlockFixedTendonData*>(dFT);
    desc.mArticulationTendonJointBlocks =
        reinterpret_cast<PxgArticulationBlockTendonJointData*>(dTJ);
    desc.mArticulationMimicJointBlocks = reinterpret_cast<PxgArticulationBlockMimicJointData*>(dMJ);
    desc.mPathToRootBitFieldBlocks = reinterpret_cast<PxgArticulationBitFieldData*>(dBF);
    desc.mMaxLinksPerArticulation = MAX_LINKS;
    desc.mMaxDofsPerArticulation = MAX_DOFS;
    desc.mMaxSpatialTendonsPerArticulation = MAX_SPATIAL_TENDONS;
    desc.mMaxAttachmentPerArticulation = MAX_ATTACHMENTS;
    desc.mMaxFixedTendonsPerArticulation = MAX_FIXED_TENDONS;
    desc.mMaxTendonJointPerArticulation = MAX_TENDON_JOINTS;
    desc.mMaxMimicJointsPerArticulation = MAX_MIMIC_JOINTS;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    size_t bad = 0;
    auto getU32 = [](const std::vector<uint8_t>& v, size_t off) {
        uint32_t x; std::memcpy(&x, v.data() + off, 4); return x;
    };
    auto getU64 = [](const std::vector<uint8_t>& v, size_t off) {
        uint64_t x; std::memcpy(&x, v.data() + off, 8); return x;
    };
    auto bits = [](float f) { uint32_t x; std::memcpy(&x, &f, 4); return x; };

    // =====================================================================
    // THREE PASSES, because the two CUDA bools are the only part of the ABI
    // a single launch cannot check.  Byte 8 is directAPI and byte 9 is
    // recomputeBlockFormat; a decoder that reads both from byte 0, or reads
    // directAPI from the whole word, agrees with the correct one on a zero
    // blob and only on a zero blob.
    // =====================================================================
    struct Pass { uint32_t bools; const char* name; };
    const Pass passes[] = {
        { 0x00000000u, "directAPI=0 recompute=0" },
        { 0x00000001u, "directAPI=1 recompute=0" },
        { 0x00000100u, "directAPI=0 recompute=1" },
    };

    for (uint32_t pi = 0; pi < 3; ++pi)
    {
        const uint32_t bools = passes[pi].bools;
        const bool directAPI = (bools & 0xffu) != 0u;
        const bool forceRecompute = ((bools >> 8) & 0xffu) != 0u;

        uploadRecords();
        uploadBlocks();

        struct Params { uint64_t desc; uint32_t bools; uint32_t pad; }
            prm = { uint64_t(dDesc), bools, 0u };
        const uint32_t numBlocks = (NB_ARTI + 63u) / 64u;
        PXG_CHECK_MSG(numBlocks == 3u, "fixture assumes three blocks");
        PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch computeUnconstrainedVelocities1TLaunch");
        shim.streamSynchronize();

        std::vector<uint8_t> ab(abImg.size()), lk(lkImg.size()), df(dfImg.size());
        std::vector<uint8_t> st(stImg.size()), at(atImg.size()), ft(ftImg.size());
        std::vector<uint8_t> tj(tjImg.size()), mj(mjImg.size()), bf(bfImg.size());
        std::vector<PxU32> sbi(BODY_SIM_SLOTS);
        std::vector<PxgArticulation> outRecs(NB_ARTI);
        shim.memcpyDtoH(ab.data(), dAB, ab.size());
        shim.memcpyDtoH(lk.data(), dLK, lk.size());
        shim.memcpyDtoH(df.data(), dDF, df.size());
        shim.memcpyDtoH(st.data(), dST, st.size());
        shim.memcpyDtoH(at.data(), dAT, at.size());
        shim.memcpyDtoH(ft.data(), dFT, ft.size());
        shim.memcpyDtoH(tj.data(), dTJ, tj.size());
        shim.memcpyDtoH(mj.data(), dMJ, mj.size());
        shim.memcpyDtoH(bf.data(), dBF, bf.size());
        shim.memcpyDtoH(sbi.data(), dSolverBodyIdx, sbi.size() * sizeof(PxU32));
        shim.memcpyDtoH(outRecs.data(), dArtis, outRecs.size() * sizeof(PxgArticulation));

        auto fail = [&](const char* what, uint32_t k, uint64_t got, uint64_t want) {
            if (bad < 16)
                std::printf("  [%s] arti %u: %s got 0x%016llX want 0x%016llX\n", passes[pi].name,
                            k, what, (unsigned long long)got, (unsigned long long)want);
            ++bad;
        };

        // ---- 1. solverBodyIndices: exactly the named slots, the rest poison ----
        std::vector<bool> touched(BODY_SIM_SLOTS, false);
        for (uint32_t k = 0; k < NB_ARTI; ++k)
        {
            if (sbi[NODE[k]] != k) fail("solverBodyIndices[node]", k, sbi[NODE[k]], k);
            touched[NODE[k]] = true;
        }
        for (uint32_t i = 0; i < BODY_SIM_SLOTS; ++i)
            if (!touched[i] && sbi[i] != 0xABABABABu)
                fail("solverBodyIndices untouched slot", i, sbi[i], 0xABABABABu);

        // ---- 2. per articulation ----
        for (uint32_t k = 0; k < NB_ARTI; ++k)
        {
            const uint32_t warp = k / 32u;
            const uint32_t lane = k % 32u;
            const Case c = caseOf(k);
            // forceRecomputeBlockFormat makes refillBlockData true for EVERY
            // lane, so the seeded cases stop being seeded
            const bool refill = forceRecompute || !seededCase(c);
            const bool filled = refill || (c == CASE_CONFI_SEEDED);
            const bool refreshed = !filled && (c == CASE_REFRESH);

            const size_t abB = size_t(warp) * AB_WARP;
            const size_t lkB = size_t(warp) * LK_WARP;
            const size_t dfB = size_t(warp) * DF_WARP;

            // -- the header fill and its guard --
            const uint32_t wantLinks = filled ? NB_LINKS : SEEDED_NUM_LINKS;
            const uint32_t wantDofs = filled ? NB_DOFS : SEEDED_TOTAL_DOFS;
            const uint32_t gotLinks =
                getU32(ab, abB + offsetof(PxgArticulationBlockData, mNumLinks) + lane * 4);
            const uint32_t gotDofs =
                getU32(ab, abB + offsetof(PxgArticulationBlockData, mTotalDofs) + lane * 4);
            if (gotLinks != wantLinks) fail("mNumLinks", k, gotLinks, wantLinks);
            if (gotDofs != wantDofs) fail("mTotalDofs", k, gotDofs, wantDofs);

            const uint32_t gotIdx =
                getU32(ab, abB + offsetof(PxgArticulationBlockData, mArticulationIndex) + lane * 4);
            if (gotIdx != POOL[k]) fail("mArticulationIndex", k, gotIdx, POOL[k]);

            if (filled)
            {
                const uint32_t gotSleep =
                    getU32(ab, abB + offsetof(PxgArticulationBlockData, mSleepThreshold) + lane * 4);
                const uint32_t wantSleep = bits(fval(k, 92));
                if (gotSleep != wantSleep) fail("mSleepThreshold", k, gotSleep, wantSleep);

                const uint8_t gotState =
                    ab[abB + offsetof(PxgArticulationBlockData, mStateDirty) + lane];
                if (gotState != 0u) fail("mStateDirty", k, gotState, 0u);

                const uint64_t gotMv = getU64(ab, abB
                    + offsetof(PxgArticulationBlockData, mMotionVelocitiesPtr) + lane * 8);
                const uint64_t wantMv = uint64_t(dMotionVel)
                    + uint64_t(size_t(k) * L_STRIDE * sizeof(Cm::UnAlignedSpatialVector));
                if (gotMv != wantMv) fail("mMotionVelocitiesPtr", k, gotMv, wantMv);

                const uint32_t gotFlags =
                    getU32(ab, abB + offsetof(PxgArticulationBlockData, mFlags) + lane * 4);
                const uint32_t wantFlags = 0x20u | (k & 1u);
                if (gotFlags != wantFlags) fail("mFlags", k, gotFlags, wantFlags);
            }

            // -- the record side: phase A clears confiDirty, and ONLY the refresh
            //    branch clears updateDirty.  This is what separates "the refresh
            //    ran" from "the fill ran", and it is the one witness phase A has
            //    that survives into a wrapper-level check.
            {
                const PxgArticulation& o = outRecs[POOL[k]];
                const bool phaseARan = (recs[POOL[k]].data.confiDirty != 0)
                                    || (recs[POOL[k]].data.updateDirty != 0u);
                const uint32_t wantConfi = 0u;   // set or not, phase A leaves it 0
                if (uint32_t(o.data.confiDirty ? 1u : 0u) != wantConfi && phaseARan)
                    fail("confiDirty not cleared by phase A", k, o.data.confiDirty ? 1u : 0u, 0u);
                const uint32_t wantUpd = refreshed ? 0u : recs[POOL[k]].data.updateDirty;
                if (o.data.updateDirty != wantUpd)
                    fail("updateDirty", k, o.data.updateDirty, wantUpd);
            }

            // -- initialize(): mDeltaQ is the IDENTITY quaternion, not zero --
            {
                const size_t dq = lkB + offsetof(PxgArticulationBlockLinkData, mDeltaQ) + lane * 16;
                const uint32_t q[4] = { getU32(lk, dq), getU32(lk, dq + 4), getU32(lk, dq + 8),
                                        getU32(lk, dq + 12) };
                if (q[0] != 0u || q[1] != 0u || q[2] != 0u || q[3] != bits(1.0f))
                    fail("mDeltaQ[0] not identity", k, q[3], bits(1.0f));
                const size_t dm =
                    lkB + offsetof(PxgArticulationBlockLinkData, mDeltaMotion) + lane * 16;
                if (getU32(lk, dm) != 0u) fail("mDeltaMotion[0] not zeroed", k, getU32(lk, dm), 0u);

                // and it is bounded by the RESOLVED link count, not the record's:
                // where the block says one link, link block 1 must be untouched
                if (!filled)
                {
                    const size_t dq1 = lkB + LK_SZ
                        + offsetof(PxgArticulationBlockLinkData, mDeltaQ) + lane * 16;
                    if (getU32(lk, dq1 + 12) != 0xABABABABu)
                        fail("link1 mDeltaQ initialised past the block's link count", k,
                             getU32(lk, dq1 + 12), 0xABABABABu);
                }
            }

            // -- the fused velocity loop: the ROOT's pre-transform is a RAW copy
            //    of mAccumulatedPose, p.w included --
            {
                const size_t pre = lkB + offsetof(PxgArticulationBlockLinkData, mPreTransform);
                const size_t acc = lkB + offsetof(PxgArticulationBlockLinkData, mAccumulatedPose);
                for (uint32_t comp = 0; comp < 4; ++comp)
                {
                    const uint32_t gq = getU32(lk, pre + lane * 16 + comp * 4);
                    const uint32_t wq = getU32(lk, acc + lane * 16 + comp * 4);
                    if (gq != wq) fail("root mPreTransform.q != mAccumulatedPose.q", k, gq, wq);
                    const uint32_t gp = getU32(lk, pre + 512 + lane * 16 + comp * 4);
                    const uint32_t wp = getU32(lk, acc + 512 + lane * 16 + comp * 4);
                    if (gp != wp) fail("root mPreTransform.p != mAccumulatedPose.p", k, gp, wp);
                }
                // CLEAN alone must have left the seeded pose in place
                const uint32_t gotPx = getU32(lk, acc + 512 + lane * 16);
                if (!filled && !refreshed)
                {
                    const uint32_t wantPx = 0x3E800000u + k * 0x00010000u;
                    if (gotPx != wantPx) fail("CLEAN root pose overwritten", k, gotPx, wantPx);
                }
                else
                {
                    const uint32_t wantPx = bits(body2Worlds[size_t(k) * L_STRIDE].p.x);
                    if (gotPx != wantPx) fail("root pose not from the record", k, gotPx, wantPx);
                }
                // directAPI suppresses the refresh path's root motion velocity
                if (refreshed)
                {
                    // PxgSpatialVectorBlock is float4 mTopxyz_bx[32] + float2
                    // mbyz[32], so the angular x of a lane is at lane * 16 --
                    // NOT lane * 4.  A first draft read lane*4 and picked up
                    // lane 0's angular z, which is exactly the kind of
                    // wrong-stride read the block map exists to prevent.
                    const uint32_t gotVel = getU32(lk,
                        lkB + offsetof(PxgArticulationBlockLinkData, mMotionVelocity) + lane * 16);
                    const uint32_t wantVel =
                        directAPI ? 0xABABABABu : bits(links[size_t(k) * L_STRIDE].initialAngVel.x);
                    if (gotVel != wantVel) fail("refresh root mMotionVelocity", k, gotVel, wantVel);
                }
            }

            // -- everything below only runs where the block copy filled --
            if (!filled)
                continue;

            const size_t blk1 = lkB + LK_SZ;
            const uint8_t dofs = lk[blk1 + offsetof(PxgArticulationBlockLinkData, mDofs) + lane];
            if (dofs != NB_DOFS) fail("link1 mDofs", k, dofs, NB_DOFS);
            const uint32_t par =
                getU32(lk, blk1 + offsetof(PxgArticulationBlockLinkData, mParents) + lane * 4);
            if (par != 0u) fail("link1 mParents", k, par, 0u);

            // 5.12: Dy::computeLinkRw (the engine's own) from the two world
            // orientations and the joint anchors; link 1 is revolute (no slide)
            const PxVec3 rw = Dy::computeLinkRw(body2Worlds[size_t(k) * L_STRIDE + 0].q,
                                                body2Worlds[size_t(k) * L_STRIDE + 1].q,
                                                jointParentOffset(k, 1), jointChildOffset(k, 1),
                                                PxVec3(0.0f));
            const uint32_t rwx =
                getU32(lk, blk1 + offsetof(PxgArticulationBlockLinkData, mRw_x) + lane * 4);
            const uint32_t rwy =
                getU32(lk, blk1 + offsetof(PxgArticulationBlockLinkData, mRw_y) + lane * 4);
            const uint32_t rwz =
                getU32(lk, blk1 + offsetof(PxgArticulationBlockLinkData, mRw_z) + lane * 4);
            if (rwx != bits(rw.x)) fail("link1 mRw_x", k, rwx, bits(rw.x));
            if (rwy != bits(rw.y)) fail("link1 mRw_y", k, rwy, bits(rw.y));
            if (rwz != bits(rw.z)) fail("link1 mRw_z", k, rwz, bits(rw.z));

            // link 1 goes through storeSpatialTransform, which writes p.w = 0 --
            // the asymmetry with the root's raw copy
            const uint32_t pw = getU32(lk, blk1
                + offsetof(PxgArticulationBlockLinkData, mPreTransform) + 512 + lane * 16 + 12);
            if (pw != 0u) fail("link1 mPreTransform.p.w not zeroed", k, pw, 0u);

            const uint32_t mjv =
                getU32(df, dfB + offsetof(PxgArticulationBlockDofData, mConstraintData)
                              + offsetof(PxgArticulationInternalConstraintData, mMaxJointVelocity)
                              + lane * 4);
            if (mjv != bits(fval(k, 84))) fail("dof0 mMaxJointVelocity", k, mjv, bits(fval(k, 84)));

            const float jv = fval(k, 101);
            const float mx = fval(k, 84);
            // forwardDynamic2.cu:995 verbatim: PxMin<float> = selectMin (b on a tie / NaN)
            const float ratio = (jv != 0.0f) ? PxMin(1.0f, mx / PxAbs(jv)) : 1.0f;
            const uint32_t gotJv =
                getU32(df, dfB + offsetof(PxgArticulationBlockDofData, mJointVelocities) + lane * 4);
            if (gotJv != bits(jv * ratio)) fail("dof0 mJointVelocities", k, gotJv, bits(jv * ratio));

            // the bitfields: link 1 carries its own index and the root, and the
            // row stride is wordSize (2 here, because maxLinks exceeds 64)
            const size_t bfw = size_t(warp) * BF_WARP + size_t(1) * WORD_SIZE * BF_SZ;
            const uint64_t gotBf = getU64(bf, bfw + lane * 8);
            const uint64_t wantBf = (uint64_t(1) << 1) | uint64_t(1);
            if (gotBf != wantBf) fail("link1 bitField", k, gotBf, wantBf);
            // the second word of link 1's row exists only because wordSize is 2
            const uint64_t gotBf1 = getU64(bf, bfw + BF_SZ + lane * 8);
            if (gotBf1 != 0ull) fail("link1 bitField word 1", k, gotBf1, 0ull);

            // the tendon topology is written on the INITIALIZE path only
            const size_t atB = size_t(warp) * AT_WARP;
            const uint16_t li = uint16_t(getU32(at, atB
                + offsetof(PxgArticulationBlockAttachmentData, mLinkIndex) + (lane / 2) * 4)
                >> ((lane & 1u) * 16u));
            if (li != 1u) fail("attachment mLinkIndex (topology)", k, li, 1u);

            const size_t stB = size_t(warp) * ST_WARP;
            const size_t ftB = size_t(warp) * FT_WARP;
            const size_t tjB = size_t(warp) * TJ_WARP;
            const size_t mjB = size_t(warp) * MJ_WARP;
            for (uint32_t t = 0; t < NB_SPATIAL; ++t)
            {
                const uint32_t stS = getU32(st, stB + size_t(t) * ST_SZ
                    + offsetof(PxgArticulationBlockSpatialTendonData, mStiffness) + lane * 4);
                if (stS != bits(fval(k, 110 + t * 10)))
                    fail("spatialTendon mStiffness", k, stS, bits(fval(k, 110 + t * 10)));
            }
            for (uint32_t t = 0; t < NB_FIXED; ++t)
            {
                const uint32_t ftS = getU32(ft, ftB + size_t(t) * FT_SZ
                    + offsetof(PxgArticulationBlockFixedTendonData, mStiffness) + lane * 4);
                if (ftS != bits(fval(k, 130 + t * 10)))
                    fail("fixedTendon mStiffness", k, ftS, bits(fval(k, 130 + t * 10)));
            }
            for (uint32_t i = 0; i < NB_MIMIC; ++i)
            {
                const uint32_t mjG = getU32(mj, mjB + size_t(i) * MJ_SZ
                    + offsetof(PxgArticulationBlockMimicJointData, mGearRatio) + lane * 4);
                if (mjG != bits(fval(k, 140 + i * 4)))
                    fail("mimicJoint mGearRatio", k, mjG, bits(fval(k, 140 + i * 4)));
            }
            (void)tjB;
        }

        // ---- 3. the ELEMENT pitches, over the whole attachment and tendon-joint
        //         regions.  `t * maxElements + j` is the only place the two
        //         product pitches appear, so one field checked at EVERY slot of
        //         every lane pins them: a wrong pitch writes a slot that must be
        //         poison and leaves a slot that must be written.
        for (uint32_t warp = 0; warp < WARPS_USED; ++warp)
        {
            for (uint32_t slot = 0; slot < MAX_SPATIAL_TENDONS * MAX_ATTACHMENTS; ++slot)
            {
                const uint32_t t = slot / MAX_ATTACHMENTS, j = slot % MAX_ATTACHMENTS;
                const bool live = (t < NB_SPATIAL) && (j < NB_ATTACH);
                for (uint32_t lane = 0; lane < 32u; ++lane)
                {
                    const uint32_t k = warp * 32u + lane;
                    if (k >= NB_ARTI) break;
                    const Case c = caseOf(k);
                    const bool wrote = forceRecompute || (c != CASE_CLEAN);
                    const uint32_t got = getU32(at, size_t(warp) * AT_WARP + size_t(slot) * AT_SZ
                        + offsetof(PxgArticulationBlockAttachmentData, mRestDistance) + lane * 4);
                    const uint32_t want = (live && wrote)
                        ? bits(fval(k, 153 + t * 20 + j)) : 0xABABABABu;
                    if (got != want) fail("attachment mRestDistance", k, got, want);
                }
            }
            for (uint32_t slot = 0; slot < MAX_FIXED_TENDONS * MAX_TENDON_JOINTS; ++slot)
            {
                const uint32_t t = slot / MAX_TENDON_JOINTS, j = slot % MAX_TENDON_JOINTS;
                const bool live = (t < NB_FIXED) && (j < NB_TJ);
                for (uint32_t lane = 0; lane < 32u; ++lane)
                {
                    const uint32_t k = warp * 32u + lane;
                    if (k >= NB_ARTI) break;
                    const Case c = caseOf(k);
                    const bool wrote = forceRecompute || (c != CASE_CLEAN);
                    const uint32_t got = getU32(tj, size_t(warp) * TJ_WARP + size_t(slot) * TJ_SZ
                        + offsetof(PxgArticulationBlockTendonJointData, mCoefficient) + lane * 4);
                    const uint32_t want = (live && wrote)
                        ? bits(fval(k, 200 + t * 20 + j)) : 0xABABABABu;
                    if (got != want) fail("tendonJoint mCoefficient", k, got, want);
                }
            }
        }

        // ---- 4. the warp slices no articulation reaches: every byte poison ----
        auto warpUntouched = [&](const char* name, const std::vector<uint8_t>& v, size_t warpBytes) {
            for (uint32_t w = WARPS_USED; w < WARPS_ALLOC; ++w)
            {
                const size_t base = size_t(w) * warpBytes;
                for (size_t i = 0; i < warpBytes; ++i)
                    if (v[base + i] != 0xAB)
                    {
                        if (bad < 16)
                            std::printf("  [%s] warp %u %s +%zu: got 0x%02X want 0xAB\n",
                                        passes[pi].name, w, name, i, v[base + i]);
                        ++bad;
                        return;
                    }
            }
        };
        warpUntouched("articulationBlocks", ab, AB_WARP);
        warpUntouched("linkBlocks", lk, LK_WARP);
        warpUntouched("dofBlocks", df, DF_WARP);
        warpUntouched("spatialTendonBlocks", st, ST_WARP);
        warpUntouched("attachmentBlocks", at, AT_WARP);
        warpUntouched("fixedTendonBlocks", ft, FT_WARP);
        warpUntouched("tendonJointBlocks", tj, TJ_WARP);
        warpUntouched("mimicJointBlocks", mj, MJ_WARP);
        warpUntouched("bitFields", bf, BF_WARP);
    }

    PXG_CHECK_MSG(bad == 0, "%zu check(s) wrong", bad);
    std::printf("  computeUnconstrainedVelocities1TLaunch: %u articulations over 5 warp slices "
                "in 3 blocks (32x2), 5 guard cases x 3 ABI passes, 9 per-warp bases, "
                "2 unused warp slices still poison\n", NB_ARTI);
}

} // namespace
