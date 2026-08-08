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

// P8.65: engine-ABI fixture for newArticulationsLaunch (newArticulationsBda.slang).
//
// The kernel takes a real PxgUpdateArticulationDesc and fills the persistent
// mArticulationPool plus every per-articulation array the staged records point
// at.  This fixture drives it with TWO articulations that differ in every way
// that could hide a bug:
//
//   * their pool slots are 3 and 1 — neither equals its warp index, so reading
//     `data.index` and using the warp index instead are distinguishable, and
//     slots 0/2/4 must stay poisoned;
//   * their source slices start at linkStartIndex 1 and 5, so a shader that
//     ignores startIndex reads the wrong links;
//   * A has one spatial tendon (3 attachments), one fixed tendon (2 tendon
//     joints) and 2 mimic joints; B has NONE of those, so both the populated
//     and the zero-count tendon paths run in the same dispatch;
//   * their dirty flags are DISJOINT — A is POSITIONS|FORCES|EXT_ACCEL, B is
//     VELOCITIES|JOINT_TARGET_POS|JOINT_TARGET_VEL — so for every one of the
//     five dof arrays, one articulation copies from dofData and the other
//     zero-fills.  Because `offset` only advances on the branches that are
//     taken, A's forces come from dofData[+5] while B's target positions come
//     from dofData[+2]: a shader that advanced the offset unconditionally, or
//     never, reads the wrong slice for both.
//
// Every constant the shader addresses with is pinned below to host-side
// sizeof/offsetof of the REAL engine structs by static_assert.  That is the
// lesson of P8.49a: a fixture that defines its own copy of a layout constant
// cannot detect that the constant is wrong, because shader and oracle move
// together.  Here the engine header is the oracle.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgSimulationCoreDesc.h"
#include "PxgSimulationCoreKernelIndices.h"
#include "PxgSolverBody.h"
#include "DyArticulationCore.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> newArtiLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// Compile-time pins for every constant in pxgSlangArticulationDesc.slang.
// A header change breaks this build instead of corrupting device memory.
// ---------------------------------------------------------------------------
namespace pin
{
using namespace physx;

static_assert(sizeof(PxgArticulation) == 544, "PXG_ARTI_STRIDE");
static_assert(offsetof(PxgArticulationData, index) == 0, "PXG_ARTI_DATA_INDEX");
static_assert(offsetof(PxgArticulationData, numLinks) == 16, "PXG_ARTI_DATA_NUM_LINKS");
static_assert(offsetof(PxgArticulationData, numJointDofs) == 18, "PXG_ARTI_DATA_NUM_JOINT_DOFS");
static_assert(offsetof(PxgArticulationData, numSpatialTendons) == 20, "..NUM_SPATIAL_TENDONS");
static_assert(offsetof(PxgArticulationData, numFixedTendons) == 22, "..NUM_FIXED_TENDONS");
static_assert(offsetof(PxgArticulationData, numMimicJoints) == 24, "..NUM_MIMIC_JOINTS");
static_assert(offsetof(PxgArticulationData, numPathToRoots) == 28, "..NUM_PATH_TO_ROOTS");
// the 16-bit counts must really be 16-bit: the shader extracts them as
// halfwords out of a 32-bit load, which is only correct at that width
static_assert(sizeof(PxgArticulationData::numLinks) == 2, "numLinks halfword");
static_assert(sizeof(PxgArticulationData::numJointDofs) == 2, "numJointDofs halfword");
static_assert(sizeof(PxgArticulationData::numSpatialTendons) == 2, "numSpatialTendons halfword");
static_assert(sizeof(PxgArticulationData::numFixedTendons) == 2, "numFixedTendons halfword");
static_assert(sizeof(PxgArticulationData::numMimicJoints) == 2, "numMimicJoints halfword");
static_assert(sizeof(PxgArticulationData::numPathToRoots) == 4, "numPathToRoots word");

static_assert(offsetof(PxgArticulation, links) == 32, "PXG_ARTI_LINKS");
static_assert(offsetof(PxgArticulation, joints) == 40, "PXG_ARTI_JOINTS");
static_assert(offsetof(PxgArticulation, jointData) == 48, "PXG_ARTI_JOINT_DATA");
static_assert(offsetof(PxgArticulation, motionVelocities) == 56, "PXG_ARTI_MOTION_VELOCITIES");
static_assert(offsetof(PxgArticulation, motionAccelerations) == 64, "PXG_ARTI_MOTION_ACCELERATIONS");
static_assert(offsetof(PxgArticulation, corioliseVectors) == 80, "PXG_ARTI_CORIOLISE_VECTORS");
static_assert(offsetof(PxgArticulation, externalAccelerations) == 96, "PXG_ARTI_EXTERNAL_ACCELERATIONS");
static_assert(offsetof(PxgArticulation, jointPositions) == 112, "PXG_ARTI_JOINT_POSITIONS");
static_assert(offsetof(PxgArticulation, jointVelocities) == 120, "PXG_ARTI_JOINT_VELOCITIES");
static_assert(offsetof(PxgArticulation, jointForce) == 136, "PXG_ARTI_JOINT_FORCE");
static_assert(offsetof(PxgArticulation, jointTargetPositions) == 144, "PXG_ARTI_JOINT_TARGET_POSITIONS");
static_assert(offsetof(PxgArticulation, jointTargetVelocities) == 152, "PXG_ARTI_JOINT_TARGET_VELOCITIES");
static_assert(offsetof(PxgArticulation, linkBody2Worlds) == 184, "PXG_ARTI_LINK_BODY2WORLDS");
static_assert(offsetof(PxgArticulation, linkBody2Actors) == 200, "PXG_ARTI_LINK_BODY2ACTORS");
static_assert(offsetof(PxgArticulation, parents) == 208, "PXG_ARTI_PARENTS");
static_assert(offsetof(PxgArticulation, linkWakeCounters) == 240, "PXG_ARTI_LINK_WAKE_COUNTERS");
static_assert(offsetof(PxgArticulation, linkSleepData) == 248, "PXG_ARTI_LINK_SLEEP_DATA");
static_assert(offsetof(PxgArticulation, linkProps) == 256, "PXG_ARTI_LINK_PROPS");
static_assert(offsetof(PxgArticulation, children) == 264, "PXG_ARTI_CHILDREN");
static_assert(offsetof(PxgArticulation, pathToRoot) == 272, "PXG_ARTI_PATH_TO_ROOT");
static_assert(offsetof(PxgArticulation, spatialTendonParams) == 304, "PXG_ARTI_SPATIAL_TENDON_PARAMS");
static_assert(offsetof(PxgArticulation, spatialTendons) == 312, "PXG_ARTI_SPATIAL_TENDONS");
static_assert(offsetof(PxgArticulation, fixedTendonParams) == 320, "PXG_ARTI_FIXED_TENDON_PARAMS");
static_assert(offsetof(PxgArticulation, fixedTendons) == 328, "PXG_ARTI_FIXED_TENDONS");
static_assert(offsetof(PxgArticulation, mimicJointCores) == 352, "PXG_ARTI_MIMIC_JOINT_CORES");

static_assert(sizeof(PxgArticulationLink) == 80, "PXG_ARTI_LINK_SIZE");
static_assert(sizeof(PxgArticulationLinkProp) == 16, "PXG_ARTI_LINK_PROP_SIZE");
static_assert(sizeof(PxgArticulationLinkSleepData) == 32, "PXG_ARTI_LINK_SLEEP_DATA_SIZE");
static_assert(sizeof(Dy::ArticulationBitField) == 8, "PXG_ARTI_BITFIELD_SIZE");
static_assert(sizeof(PxTransform) == 28, "PXG_ARTI_TRANSFORM_SIZE");
static_assert(sizeof(Dy::ArticulationJointCore) == 544, "PXG_ARTI_JOINT_CORE_SIZE");
static_assert(sizeof(Dy::ArticulationJointCoreData) == 8, "PXG_ARTI_JOINT_CORE_DATA_SIZE");
static_assert(sizeof(Cm::UnAlignedSpatialVector) == 24, "PXG_ARTI_SPATIAL_VECTOR_SIZE");
static_assert(sizeof(Dy::ArticulationMimicJointCore) == 32, "PXG_ARTI_MIMIC_JOINT_CORE_SIZE");
static_assert(sizeof(PxgSolverBodySleepData) == 8, "PXG_ARTI_SLEEP_DATA_SIZE");

static_assert(sizeof(PxgArticulationTendon) == 24, "PXG_ARTI_TENDON_SIZE");
static_assert(sizeof(PxgArticulationTendonElementFixedData) == 16, "PXG_ARTI_TENDON_FIXED_ELEM_SIZE");
static_assert(sizeof(PxGpuTendonAttachmentData) == 32, "PXG_ARTI_ATTACHMENT_MOD_SIZE");
static_assert(sizeof(PxGpuTendonJointCoefficientData) == 16, "PXG_ARTI_TENDON_JOINT_COEFF_SIZE");
static_assert(sizeof(PxGpuSpatialTendonData) == 16, "PXG_ARTI_SPATIAL_TENDON_DATA_SIZE");
static_assert(sizeof(PxGpuFixedTendonData) == 32, "PXG_ARTI_FIXED_TENDON_DATA_SIZE");
static_assert(offsetof(PxgArticulationTendon, mFixedElements) == 0, "PXG_TENDON_FIXED_ELEMENTS");
static_assert(offsetof(PxgArticulationTendon, mModElements) == 8, "PXG_TENDON_MOD_ELEMENTS");
static_assert(offsetof(PxgArticulationTendon, mNbElements) == 16, "PXG_TENDON_NB_ELEMENTS");

static_assert(sizeof(PxgArticulationSimUpdate) == 48, "PxgArticulationSimUpdateGpu stride");
static_assert(offsetof(PxgArticulationSimUpdate, dirtyFlags) == 0, "dirtyFlags");
static_assert(offsetof(PxgArticulationSimUpdate, articulationIndex) == 4, "articulationIndex");
static_assert(offsetof(PxgArticulationSimUpdate, linkStartIndex) == 8, "linkStartIndex");
static_assert(offsetof(PxgArticulationSimUpdate, dofDataStartIndex) == 12, "dofDataStartIndex");
static_assert(offsetof(PxgArticulationSimUpdate, spatialTendonStartIndex) == 16, "spatialTendonStartIndex");
static_assert(offsetof(PxgArticulationSimUpdate, fixedTendonStartIndex) == 24, "fixedTendonStartIndex");
static_assert(offsetof(PxgArticulationSimUpdate, mimicJointStartIndex) == 32, "mimicJointStartIndex");
static_assert(offsetof(PxgArticulationSimUpdate, pathToRootIndex) == 36, "pathToRootIndex");

static_assert(sizeof(PxgUpdateArticulationDesc) == 216, "PxgUpdateArticulationDescGpu size");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewArticulations) == 0, "mNewArticulations");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinkWakeCounters) == 8, "mNewLinkWakeCounters");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinks) == 16, "mNewLinks");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinkProps) == 24, "mNewLinkProps");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinkParents) == 32, "mNewLinkParents");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinkChildren) == 40, "mNewLinkChildren");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinkBody2Worlds) == 48, "mNewLinkBody2Worlds");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinkBody2Actors) == 56, "mNewLinkBody2Actors");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewLinkExtAccels) == 64, "mNewLinkExtAccels");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewJointCores) == 72, "mNewJointCores");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewJointData) == 80, "mNewJointData");
static_assert(offsetof(PxgUpdateArticulationDesc, mIndicesOffset) == 88, "mIndicesOffset");
static_assert(offsetof(PxgUpdateArticulationDesc, mArticulationPool) == 96, "mArticulationPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mArticulationSleepDataPool) == 104, "mArticulationSleepDataPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewSpatialTendonParamsPool) == 112, "mNewSpatialTendonParamsPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewSpatialTendonPool) == 120, "mNewSpatialTendonPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewAttachmentFixedPool) == 128, "mNewAttachmentFixedPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewAttachmentModPool) == 136, "mNewAttachmentModPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewTendonAttachmentRemapPool) == 144, "..RemapPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewFixedTendonParamsPool) == 152, "mNewFixedTendonParamsPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewFixedTendonPool) == 160, "mNewFixedTendonPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewTendonJointFixedPool) == 168, "mNewTendonJointFixedPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewTendonJointCoefficientPool) == 176, "..CoefficientPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewTendonTendonJointRemapPool) == 184, "..TendonJointRemapPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewArticulationMimicJointPool) == 192, "..MimicJointPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNewPathToRootPool) == 200, "mNewPathToRootPool");
static_assert(offsetof(PxgUpdateArticulationDesc, mNbNewArticulations) == 208, "mNbNewArticulations");

// the launch geometry the shader hard-codes as NA_WARPS_PER_BLOCK
static_assert(physx::PxgSimulationCoreKernelBlockDim::NEW_ARTICULATION == 256, "NEW_ARTICULATION blockDim");

// the dirty-flag bits
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS) == 2, "DIRTY_POSITIONS");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_VELOCITIES) == 4, "DIRTY_VELOCITIES");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FORCES) == 8, "DIRTY_FORCES");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_EXT_ACCEL) == 512, "DIRTY_EXT_ACCEL");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_VEL) == 4096, "DIRTY_JOINT_TARGET_VEL");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_POS) == 8192, "DIRTY_JOINT_TARGET_POS");
} // namespace pin

// ---------------------------------------------------------------------------
// A byte arena: one device allocation whose interior addresses become the
// staged records' pointer fields.  The shim resolves interior pointers, so a
// single alloc stands in for the ~25 per-articulation buffers the engine makes.
// ---------------------------------------------------------------------------
struct Arena
{
    std::vector<uint8_t> host;
    size_t cursor = 0;

    // 0xAB poison everywhere: any region the kernel must NOT touch is checked
    // to still be poison, and any region it must fill is checked byte-exactly.
    explicit Arena(size_t bytes) : host(bytes, 0xAB) {}

    size_t take(size_t bytes)
    {
        const size_t off = (cursor + 15u) & ~size_t(15);
        cursor = off + bytes;
        return off;
    }
};

// deterministic source content: distinct per pool and per byte
static uint8_t pat(uint32_t poolId, size_t i) { return uint8_t(poolId * 37u + uint32_t(i) * 11u + 3u); }
static void patternFill(std::vector<uint8_t>& v, uint32_t poolId)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = pat(poolId, i);
}

// one articulation's shape, and the arena offsets of everything it owns
struct ArtiSpec
{
    uint32_t poolIndex;
    uint32_t nbLinks, numDofs, nbSpatial, nbFixed, numMimic, numPathToRoots;
    uint32_t linkStart, dofStart, spatialStart, fixedStart, mimicStart, pathStart;
    uint32_t dirtyFlags;

    size_t oLinks, oWake, oProps, oParents, oChildren, oB2W, oB2A, oJoints, oJointData;
    size_t oSpatialTendons, oSpatialParams, oFixedTendons, oFixedParams;
    size_t oMimic, oPathToRoot;
    size_t oMotionVel, oMotionAccel, oCoriolise, oLinkSleep;
    size_t oJointPos, oJointVel, oJointForce, oJointTgtPos, oJointTgtVel, oExtAccel;
};

PXG_TEST(newArticulations)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "newArticulationsBda.spv", "newArticulationsLaunch", 16u, 0 }, // { u64 desc, u64 dofData }
    };
    PxgCudaShim shim(dev, manifest, 1, &newArtiLoader);

    // ---- the articulation table ----
    //
    // TEN articulations, so the dispatch is TWO blocks of 8 virtual warps and the
    // block base in `groupId.x * NA_WARPS_PER_BLOCK` is load-bearing.  With a
    // single block that term is always zero and a wrong warps-per-block constant
    // is invisible; the two interesting articulations therefore sit at warp
    // indices 8 and 9, i.e. in the SECOND block.
    //
    // Warps 0-7 are minimal fillers (one link, one dof, no tendons, NO dirty
    // flags) which also cover the all-clean dof path: every one of the five dof
    // arrays takes its else-branch.
    const uint32_t NB_ARTI = 10;
    std::vector<ArtiSpec> specs(NB_ARTI);
    for (uint32_t i = 0; i < 8; ++i)
    {
        ArtiSpec& f = specs[i];
        f.poolIndex = 4 + i;      // slots 4..11; A and B take 3 and 1
        f.nbLinks = 1; f.numDofs = 1;
        f.nbSpatial = 0; f.nbFixed = 0; f.numMimic = 0; f.numPathToRoots = 1;
        f.linkStart = i; f.dofStart = 21; f.spatialStart = 3; f.fixedStart = 3;
        f.mimicStart = 6; f.pathStart = 8 + i % 4;
        f.dirtyFlags = 0;
    }
    {
        ArtiSpec& a = specs[8];
        // SIX links, not three, for one specific reason: the largest zero-fill
        // region is linkSleepData at 32 bytes per link, and a fill region of 32
        // u32 words or fewer is covered by lanes 0..31 whatever the loop stride
        // is.  Six links makes that region 48 words, so the 32-lane stride in
        // naFill becomes load-bearing and a wrong stride loses words 32..47.
        a.poolIndex = 3; a.nbLinks = 6; a.numDofs = 5;
        a.nbSpatial = 1; a.nbFixed = 1; a.numMimic = 2; a.numPathToRoots = 4;
        a.linkStart = 1; a.dofStart = 0; a.spatialStart = 1; a.fixedStart = 2;
        a.mimicStart = 2; a.pathStart = 1;
        // POSITIONS, FORCES and TARGET_POS but NOT TARGET_VEL.  Three
        // consequences the sweep needs: the offset advance after FORCES feeds a
        // LATER dirty array (TARGET_POS reads dofData[+10], so dropping that
        // advance is visible); TARGET_POS is set where TARGET_VEL is clear (so
        // the two bits are distinguishable); and EXT_ACCEL is set on A and clear
        // on B, covering both sides of that branch.
        a.dirtyFlags = uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS) |
                       uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FORCES) |
                       uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_POS) |
                       uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_EXT_ACCEL);

        ArtiSpec& b = specs[9];
        b.poolIndex = 1; b.nbLinks = 2; b.numDofs = 2;
        b.nbSpatial = 0; b.nbFixed = 0; b.numMimic = 0; b.numPathToRoots = 3;
        b.linkStart = 5; b.dofStart = 15; b.spatialStart = 2; b.fixedStart = 3;
        b.mimicStart = 4; b.pathStart = 5;
        b.dirtyFlags = uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_VELOCITIES) |
                       uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_POS) |
                       uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_VEL);
    }

    // A's tendon element counts
    const uint32_t A_NB_ATTACH = 3;   // spatial-tendon attachments
    const uint32_t A_NB_TJOINT = 2;   // fixed-tendon tendon joints
    const uint32_t A_ATTACH_START = 5;  // mNewTendonAttachmentRemapPool[spatialStart]
    const uint32_t A_TJOINT_START = 2;  // mNewTendonTendonJointRemapPool[fixedStart]

    // ---- carve the destination arena ----
    Arena ar(256 * 1024);
    auto carve = [&ar](ArtiSpec& s) {
        // zero-count arrays still get a poisoned region, so "wrote nothing" is
        // checkable rather than merely untested
        const size_t guard = 64;
        s.oLinks = ar.take(80u * s.nbLinks + guard);
        s.oWake = ar.take(4u * s.nbLinks + guard);
        s.oProps = ar.take(16u * s.nbLinks + guard);
        s.oParents = ar.take(4u * s.nbLinks + guard);
        s.oChildren = ar.take(8u * s.nbLinks + guard);
        s.oB2W = ar.take(28u * s.nbLinks + guard);
        s.oB2A = ar.take(28u * s.nbLinks + guard);
        s.oJoints = ar.take(544u * s.nbLinks + guard);
        s.oJointData = ar.take(8u * s.nbLinks + guard);
        s.oSpatialTendons = ar.take(24u * s.nbSpatial + guard);
        s.oSpatialParams = ar.take(16u * s.nbSpatial + guard);
        s.oFixedTendons = ar.take(24u * s.nbFixed + guard);
        s.oFixedParams = ar.take(32u * s.nbFixed + guard);
        s.oMimic = ar.take(32u * s.numMimic + guard);
        s.oPathToRoot = ar.take(4u * s.numPathToRoots + guard);
        s.oMotionVel = ar.take(24u * s.nbLinks + guard);
        s.oMotionAccel = ar.take(24u * s.nbLinks + guard);
        s.oCoriolise = ar.take(24u * s.nbLinks + guard);
        s.oLinkSleep = ar.take(32u * s.nbLinks + guard);
        s.oJointPos = ar.take(4u * s.numDofs + guard);
        s.oJointVel = ar.take(4u * s.numDofs + guard);
        s.oJointForce = ar.take(4u * s.numDofs + guard);
        s.oJointTgtPos = ar.take(4u * s.numDofs + guard);
        s.oJointTgtVel = ar.take(4u * s.numDofs + guard);
        s.oExtAccel = ar.take(24u * s.nbLinks + guard);
    };
    for (uint32_t i = 0; i < NB_ARTI; ++i)
        carve(specs[i]);
    const ArtiSpec& A = specs[8];
    const ArtiSpec& B = specs[9];
    // A's tendon element destinations, reached through the tendon records
    const size_t oAttachFixedDst = ar.take(16u * A_NB_ATTACH + 64);
    const size_t oAttachModDst = ar.take(32u * A_NB_ATTACH + 64);
    const size_t oTJointFixedDst = ar.take(16u * A_NB_TJOINT + 64);
    const size_t oTJointCoeffDst = ar.take(16u * A_NB_TJOINT + 64);

    // ---- the flattened source pools ----
    const uint32_t LINK_POOL = 8;
    std::vector<uint8_t> pLinks(80u * LINK_POOL), pWake(4u * LINK_POOL), pProps(16u * LINK_POOL);
    std::vector<uint8_t> pParents(4u * LINK_POOL), pChildren(8u * LINK_POOL);
    std::vector<uint8_t> pB2W(28u * LINK_POOL), pB2A(28u * LINK_POOL);
    std::vector<uint8_t> pJoints(544u * LINK_POOL), pJointData(8u * LINK_POOL);
    std::vector<uint8_t> pExtAccel(24u * LINK_POOL);
    patternFill(pLinks, 1); patternFill(pWake, 2); patternFill(pProps, 3);
    patternFill(pParents, 4); patternFill(pChildren, 5);
    patternFill(pB2W, 6); patternFill(pB2A, 7);
    patternFill(pJoints, 8); patternFill(pJointData, 9); patternFill(pExtAccel, 10);

    std::vector<uint8_t> pSpatialParams(16u * 4), pFixedParams(32u * 4);
    std::vector<uint8_t> pAttachFixed(16u * 16), pAttachMod(32u * 16);
    std::vector<uint8_t> pTJointFixed(16u * 16), pTJointCoeff(16u * 16);
    std::vector<uint8_t> pMimic(32u * 8), pPathToRoot(4u * 12);
    patternFill(pSpatialParams, 11); patternFill(pFixedParams, 12);
    patternFill(pAttachFixed, 13); patternFill(pAttachMod, 14);
    patternFill(pTJointFixed, 15); patternFill(pTJointCoeff, 16);
    patternFill(pMimic, 17); patternFill(pPathToRoot, 18);

    // A consumes dofData[0..14] (3 dirty arrays x 5 dofs), B dofData[15..20]
    // (3 x 2); the fillers are entirely clean and read none.
    std::vector<float> dofData(24);
    for (size_t i = 0; i < dofData.size(); ++i)
        dofData[i] = 100.0f + float(i) * 0.5f;

    std::vector<uint32_t> pAttachRemap(4, 0), pTJointRemap(4, 0);
    pAttachRemap[A.spatialStart] = A_ATTACH_START;
    pTJointRemap[A.fixedStart] = A_TJOINT_START;

    // ---- device allocations ----
    PxgDevicePtr dAr = 0, dPool = 0, dSleep = 0, dNewArtis = 0, dUpdates = 0, dDesc = 0, dDof = 0;
    PxgDevicePtr dLinks = 0, dWake = 0, dProps = 0, dParents = 0, dChildren = 0, dB2W = 0, dB2A = 0;
    PxgDevicePtr dJoints = 0, dJointData = 0, dExtAccel = 0;
    PxgDevicePtr dSpatialTendons = 0, dSpatialParams = 0, dFixedTendons = 0, dFixedParams = 0;
    PxgDevicePtr dAttachFixed = 0, dAttachMod = 0, dAttachRemap = 0;
    PxgDevicePtr dTJointFixed = 0, dTJointCoeff = 0, dTJointRemap = 0;
    PxgDevicePtr dMimic = 0, dPathToRoot = 0;

    const uint32_t POOL_SLOTS = 13;  // slots 0, 2 and 12 stay poisoned
    shim.memAlloc(&dAr, ar.host.size());
    shim.memAlloc(&dPool, POOL_SLOTS * sizeof(PxgArticulation));
    shim.memAlloc(&dSleep, POOL_SLOTS * sizeof(PxgSolverBodySleepData));
    shim.memAlloc(&dNewArtis, NB_ARTI * sizeof(PxgArticulation));
    shim.memAlloc(&dUpdates, NB_ARTI * sizeof(PxgArticulationSimUpdate));
    shim.memAlloc(&dDesc, sizeof(PxgUpdateArticulationDesc));
    shim.memAlloc(&dDof, dofData.size() * 4);
    shim.memAlloc(&dLinks, pLinks.size());
    shim.memAlloc(&dWake, pWake.size());
    shim.memAlloc(&dProps, pProps.size());
    shim.memAlloc(&dParents, pParents.size());
    shim.memAlloc(&dChildren, pChildren.size());
    shim.memAlloc(&dB2W, pB2W.size());
    shim.memAlloc(&dB2A, pB2A.size());
    shim.memAlloc(&dJoints, pJoints.size());
    shim.memAlloc(&dJointData, pJointData.size());
    shim.memAlloc(&dExtAccel, pExtAccel.size());
    shim.memAlloc(&dSpatialTendons, 4 * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dSpatialParams, pSpatialParams.size());
    shim.memAlloc(&dFixedTendons, 4 * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dFixedParams, pFixedParams.size());
    shim.memAlloc(&dAttachFixed, pAttachFixed.size());
    shim.memAlloc(&dAttachMod, pAttachMod.size());
    shim.memAlloc(&dAttachRemap, pAttachRemap.size() * 4);
    shim.memAlloc(&dTJointFixed, pTJointFixed.size());
    shim.memAlloc(&dTJointCoeff, pTJointCoeff.size());
    shim.memAlloc(&dTJointRemap, pTJointRemap.size() * 4);
    shim.memAlloc(&dMimic, pMimic.size());
    shim.memAlloc(&dPathToRoot, pPathToRoot.size());

    if (!dAr || !dPool || !dSleep || !dNewArtis || !dUpdates || !dDesc || !dDof || !dLinks ||
        !dSpatialTendons || !dFixedTendons || !dAttachRemap || !dTJointRemap || !dMimic || !dPathToRoot)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // ---- the staged PxgArticulation records: REAL engine structs ----
    auto stage = [&](const ArtiSpec& s, PxgArticulation& rec) {
        std::memset(&rec, 0xEE, sizeof(rec));
        rec.data.index = s.poolIndex;
        rec.data.numLinks = PxU16(s.nbLinks);
        rec.data.numJointDofs = PxU16(s.numDofs);
        rec.data.numSpatialTendons = PxU16(s.nbSpatial);
        rec.data.numFixedTendons = PxU16(s.nbFixed);
        rec.data.numMimicJoints = PxU16(s.numMimic);
        rec.data.numPathToRoots = s.numPathToRoots;
        rec.links = reinterpret_cast<PxgArticulationLink*>(dAr + s.oLinks);
        rec.linkWakeCounters = reinterpret_cast<PxReal*>(dAr + s.oWake);
        rec.linkProps = reinterpret_cast<PxgArticulationLinkProp*>(dAr + s.oProps);
        rec.parents = reinterpret_cast<PxU32*>(dAr + s.oParents);
        rec.children = reinterpret_cast<Dy::ArticulationBitField*>(dAr + s.oChildren);
        rec.linkBody2Worlds = reinterpret_cast<PxTransform*>(dAr + s.oB2W);
        rec.linkBody2Actors = reinterpret_cast<PxTransform*>(dAr + s.oB2A);
        rec.joints = reinterpret_cast<Dy::ArticulationJointCore*>(dAr + s.oJoints);
        rec.jointData = reinterpret_cast<Dy::ArticulationJointCoreData*>(dAr + s.oJointData);
        rec.spatialTendons = reinterpret_cast<PxgArticulationTendon*>(dAr + s.oSpatialTendons);
        rec.spatialTendonParams = reinterpret_cast<PxGpuSpatialTendonData*>(dAr + s.oSpatialParams);
        rec.fixedTendons = reinterpret_cast<PxgArticulationTendon*>(dAr + s.oFixedTendons);
        rec.fixedTendonParams = reinterpret_cast<PxGpuFixedTendonData*>(dAr + s.oFixedParams);
        rec.mimicJointCores = reinterpret_cast<Dy::ArticulationMimicJointCore*>(dAr + s.oMimic);
        rec.pathToRoot = reinterpret_cast<PxU32*>(dAr + s.oPathToRoot);
        rec.motionVelocities = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dAr + s.oMotionVel);
        rec.motionAccelerations = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dAr + s.oMotionAccel);
        rec.corioliseVectors = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dAr + s.oCoriolise);
        rec.linkSleepData = reinterpret_cast<PxgArticulationLinkSleepData*>(dAr + s.oLinkSleep);
        rec.jointPositions = reinterpret_cast<PxReal*>(dAr + s.oJointPos);
        rec.jointVelocities = reinterpret_cast<PxReal*>(dAr + s.oJointVel);
        rec.jointForce = reinterpret_cast<PxReal*>(dAr + s.oJointForce);
        rec.jointTargetPositions = reinterpret_cast<PxReal*>(dAr + s.oJointTgtPos);
        rec.jointTargetVelocities = reinterpret_cast<PxReal*>(dAr + s.oJointTgtVel);
        rec.externalAccelerations = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dAr + s.oExtAccel);
    };
    std::vector<PxgArticulation> recs(NB_ARTI);
    for (uint32_t i = 0; i < NB_ARTI; ++i)
        stage(specs[i], recs[i]);

    // ---- the tendon records A copies through ----
    PxgArticulationTendon spatialTendons[4];
    std::memset(spatialTendons, 0xEE, sizeof(spatialTendons));
    spatialTendons[A.spatialStart].mFixedElements =
        reinterpret_cast<PxgArticulationTendonElementFixedData*>(dAr + oAttachFixedDst);
    spatialTendons[A.spatialStart].mModElements = reinterpret_cast<void*>(dAr + oAttachModDst);
    spatialTendons[A.spatialStart].mNbElements = A_NB_ATTACH;

    PxgArticulationTendon fixedTendons[4];
    std::memset(fixedTendons, 0xEE, sizeof(fixedTendons));
    fixedTendons[A.fixedStart].mFixedElements =
        reinterpret_cast<PxgArticulationTendonElementFixedData*>(dAr + oTJointFixedDst);
    fixedTendons[A.fixedStart].mModElements = reinterpret_cast<void*>(dAr + oTJointCoeffDst);
    fixedTendons[A.fixedStart].mNbElements = A_NB_TJOINT;

    // ---- the sim-update records ----
    std::vector<PxgArticulationSimUpdate> updates(NB_ARTI);
    std::memset(updates.data(), 0xEE, updates.size() * sizeof(PxgArticulationSimUpdate));
    auto fillUpdate = [](const ArtiSpec& s, PxgArticulationSimUpdate& u) {
        u.dirtyFlags = s.dirtyFlags;
        u.articulationIndex = s.poolIndex;
        u.linkStartIndex = s.linkStart;
        u.dofDataStartIndex = s.dofStart;
        u.spatialTendonStartIndex = s.spatialStart;
        u.fixedTendonStartIndex = s.fixedStart;
        u.mimicJointStartIndex = s.mimicStart;
        u.pathToRootIndex = s.pathStart;
    };
    for (uint32_t i = 0; i < NB_ARTI; ++i)
        fillUpdate(specs[i], updates[i]);

    // ---- the descriptor ----
    PxgUpdateArticulationDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.mNewArticulations = reinterpret_cast<PxgArticulation*>(dNewArtis);
    desc.mNewLinkWakeCounters = reinterpret_cast<PxReal*>(dWake);
    desc.mNewLinks = reinterpret_cast<PxgArticulationLink*>(dLinks);
    desc.mNewLinkProps = reinterpret_cast<PxgArticulationLinkProp*>(dProps);
    desc.mNewLinkParents = reinterpret_cast<PxU32*>(dParents);
    desc.mNewLinkChildren = reinterpret_cast<Dy::ArticulationBitField*>(dChildren);
    desc.mNewLinkBody2Worlds = reinterpret_cast<PxTransform*>(dB2W);
    desc.mNewLinkBody2Actors = reinterpret_cast<PxTransform*>(dB2A);
    desc.mNewLinkExtAccels = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dExtAccel);
    desc.mNewJointCores = reinterpret_cast<Dy::ArticulationJointCore*>(dJoints);
    desc.mNewJointData = reinterpret_cast<Dy::ArticulationJointCoreData*>(dJointData);
    desc.mIndicesOffset = reinterpret_cast<PxgArticulationSimUpdate*>(dUpdates);
    desc.mArticulationPool = reinterpret_cast<PxgArticulation*>(dPool);
    desc.mArticulationSleepDataPool = reinterpret_cast<PxgSolverBodySleepData*>(dSleep);
    desc.mNewSpatialTendonParamsPool = reinterpret_cast<PxGpuSpatialTendonData*>(dSpatialParams);
    desc.mNewSpatialTendonPool = reinterpret_cast<PxgArticulationTendon*>(dSpatialTendons);
    desc.mNewAttachmentFixedPool =
        reinterpret_cast<PxgArticulationTendonElementFixedData*>(dAttachFixed);
    desc.mNewAttachmentModPool = reinterpret_cast<PxGpuTendonAttachmentData*>(dAttachMod);
    desc.mNewTendonAttachmentRemapPool = reinterpret_cast<PxU32*>(dAttachRemap);
    desc.mNewFixedTendonParamsPool = reinterpret_cast<PxGpuFixedTendonData*>(dFixedParams);
    desc.mNewFixedTendonPool = reinterpret_cast<PxgArticulationTendon*>(dFixedTendons);
    desc.mNewTendonJointFixedPool =
        reinterpret_cast<PxgArticulationTendonElementFixedData*>(dTJointFixed);
    desc.mNewTendonJointCoefficientPool =
        reinterpret_cast<PxGpuTendonJointCoefficientData*>(dTJointCoeff);
    desc.mNewTendonTendonJointRemapPool = reinterpret_cast<PxU32*>(dTJointRemap);
    desc.mNewArticulationMimicJointPool = reinterpret_cast<Dy::ArticulationMimicJointCore*>(dMimic);
    desc.mNewPathToRootPool = reinterpret_cast<PxU32*>(dPathToRoot);
    desc.mNbNewArticulations = NB_ARTI;

    // ---- upload ----
    std::vector<uint8_t> poolPoison(POOL_SLOTS * sizeof(PxgArticulation), 0xAB);
    std::vector<uint8_t> sleepPoison(POOL_SLOTS * sizeof(PxgSolverBodySleepData), 0xAB);
    shim.memcpyHtoD(dAr, ar.host.data(), ar.host.size());
    shim.memcpyHtoD(dPool, poolPoison.data(), poolPoison.size());
    shim.memcpyHtoD(dSleep, sleepPoison.data(), sleepPoison.size());
    shim.memcpyHtoD(dNewArtis, recs.data(), recs.size() * sizeof(PxgArticulation));
    shim.memcpyHtoD(dUpdates, updates.data(), updates.size() * sizeof(PxgArticulationSimUpdate));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memcpyHtoD(dDof, dofData.data(), dofData.size() * 4);
    shim.memcpyHtoD(dLinks, pLinks.data(), pLinks.size());
    shim.memcpyHtoD(dWake, pWake.data(), pWake.size());
    shim.memcpyHtoD(dProps, pProps.data(), pProps.size());
    shim.memcpyHtoD(dParents, pParents.data(), pParents.size());
    shim.memcpyHtoD(dChildren, pChildren.data(), pChildren.size());
    shim.memcpyHtoD(dB2W, pB2W.data(), pB2W.size());
    shim.memcpyHtoD(dB2A, pB2A.data(), pB2A.size());
    shim.memcpyHtoD(dJoints, pJoints.data(), pJoints.size());
    shim.memcpyHtoD(dJointData, pJointData.data(), pJointData.size());
    shim.memcpyHtoD(dExtAccel, pExtAccel.data(), pExtAccel.size());
    shim.memcpyHtoD(dSpatialTendons, spatialTendons, sizeof(spatialTendons));
    shim.memcpyHtoD(dSpatialParams, pSpatialParams.data(), pSpatialParams.size());
    shim.memcpyHtoD(dFixedTendons, fixedTendons, sizeof(fixedTendons));
    shim.memcpyHtoD(dFixedParams, pFixedParams.data(), pFixedParams.size());
    shim.memcpyHtoD(dAttachFixed, pAttachFixed.data(), pAttachFixed.size());
    shim.memcpyHtoD(dAttachMod, pAttachMod.data(), pAttachMod.size());
    shim.memcpyHtoD(dAttachRemap, pAttachRemap.data(), pAttachRemap.size() * 4);
    shim.memcpyHtoD(dTJointFixed, pTJointFixed.data(), pTJointFixed.size());
    shim.memcpyHtoD(dTJointCoeff, pTJointCoeff.data(), pTJointCoeff.size());
    shim.memcpyHtoD(dTJointRemap, pTJointRemap.data(), pTJointRemap.size() * 4);
    shim.memcpyHtoD(dMimic, pMimic.data(), pMimic.size());
    shim.memcpyHtoD(dPathToRoot, pPathToRoot.data(), pPathToRoot.size());

    // ---- dispatch: the engine's grid, ceil(nbArticulations / warpsPerBlock)
    //      (PxgSimulationCore.cpp:2388) -> 2 blocks of (32, 8, 1) ----
    const uint32_t warpsPerBlock = physx::PxgSimulationCoreKernelBlockDim::NEW_ARTICULATION / 32u;
    const uint32_t numBlocks = (NB_ARTI + warpsPerBlock - 1u) / warpsPerBlock;
    PXG_CHECK_MSG(numBlocks == 2, "the fixture must span 2 blocks (got %u)", numBlocks);
    struct Params { uint64_t desc, dofData; } prm = { uint64_t(dDesc), uint64_t(dDof) };
    PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch newArticulationsLaunch");
    shim.streamSynchronize();

    // ---- read back ----
    std::vector<uint8_t> gotAr(ar.host.size(), 0);
    std::vector<uint8_t> gotPool(poolPoison.size(), 0);
    std::vector<uint8_t> gotSleep(sleepPoison.size(), 0);
    shim.memcpyDtoH(gotAr.data(), dAr, gotAr.size());
    shim.memcpyDtoH(gotPool.data(), dPool, gotPool.size());
    shim.memcpyDtoH(gotSleep.data(), dSleep, gotSleep.size());

    size_t bad = 0;
    auto region = [&](const char* what, size_t off, const uint8_t* expect, size_t n) {
        for (size_t i = 0; i < n; ++i)
            if (gotAr[off + i] != expect[i])
            {
                if (bad < 12)
                    std::printf("  %s byte %zu: got 0x%02X want 0x%02X\n", what, i,
                                gotAr[off + i], expect[i]);
                ++bad;
                return;
            }
    };
    auto copied = [&](const char* what, size_t off, const std::vector<uint8_t>& pool,
                      size_t srcByte, size_t n) {
        region(what, off, pool.data() + srcByte, n);
    };
    auto zeroed = [&](const char* what, size_t off, size_t n) {
        std::vector<uint8_t> z(n, 0);
        region(what, off, z.data(), n);
    };
    // the guard bytes past every region must still be poison
    auto guardIntact = [&](const char* what, size_t off, size_t n) {
        for (size_t i = n; i < n + 64; ++i)
            if (gotAr[off + i] != 0xAB)
            {
                if (bad < 12)
                    std::printf("  %s overran by %zu bytes (0x%02X)\n", what, i - n + 1,
                                gotAr[off + i]);
                ++bad;
                return;
            }
    };


    // The tendon-record copies need the raw staged bytes, not a pattern pool,
    // so they are checked explicitly rather than through the generic helper.
    {
        // A: spatialTendons[1] (24 B) -> A.oSpatialTendons
        region("A.spatialTendons", A.oSpatialTendons,
               reinterpret_cast<const uint8_t*>(&spatialTendons[A.spatialStart]), 24);
        guardIntact("A.spatialTendons", A.oSpatialTendons, 24);
        region("A.fixedTendons", A.oFixedTendons,
               reinterpret_cast<const uint8_t*>(&fixedTendons[A.fixedStart]), 24);
        guardIntact("A.fixedTendons", A.oFixedTendons, 24);
        // B has none: its regions must be untouched poison
        guardIntact("B.spatialTendons(empty)", B.oSpatialTendons, 0);
        guardIntact("B.fixedTendons(empty)", B.oFixedTendons, 0);
        guardIntact("B.spatialParams(empty)", B.oSpatialParams, 0);
        guardIntact("B.fixedParams(empty)", B.oFixedParams, 0);
        guardIntact("B.mimic(empty)", B.oMimic, 0);
    }

    auto checkLinkArrays = [&](const char* tag, const ArtiSpec& s) {
        char nm[96];
        auto n = [&](const char* f) { std::snprintf(nm, sizeof(nm), "%s.%s", tag, f); return nm; };
        copied(n("links"), s.oLinks, pLinks, 80u * s.linkStart, 80u * s.nbLinks);
        guardIntact(n("links"), s.oLinks, 80u * s.nbLinks);
        copied(n("wakeCounters"), s.oWake, pWake, 4u * s.linkStart, 4u * s.nbLinks);
        guardIntact(n("wakeCounters"), s.oWake, 4u * s.nbLinks);
        copied(n("linkProps"), s.oProps, pProps, 16u * s.linkStart, 16u * s.nbLinks);
        guardIntact(n("linkProps"), s.oProps, 16u * s.nbLinks);
        copied(n("parents"), s.oParents, pParents, 4u * s.linkStart, 4u * s.nbLinks);
        guardIntact(n("parents"), s.oParents, 4u * s.nbLinks);
        copied(n("children"), s.oChildren, pChildren, 8u * s.linkStart, 8u * s.nbLinks);
        guardIntact(n("children"), s.oChildren, 8u * s.nbLinks);
        copied(n("body2Worlds"), s.oB2W, pB2W, 28u * s.linkStart, 28u * s.nbLinks);
        guardIntact(n("body2Worlds"), s.oB2W, 28u * s.nbLinks);
        copied(n("body2Actors"), s.oB2A, pB2A, 28u * s.linkStart, 28u * s.nbLinks);
        guardIntact(n("body2Actors"), s.oB2A, 28u * s.nbLinks);
        copied(n("joints"), s.oJoints, pJoints, 544u * s.linkStart, 544u * s.nbLinks);
        guardIntact(n("joints"), s.oJoints, 544u * s.nbLinks);
        copied(n("jointData"), s.oJointData, pJointData, 8u * s.linkStart, 8u * s.nbLinks);
        guardIntact(n("jointData"), s.oJointData, 8u * s.nbLinks);
        copied(n("pathToRoot"), s.oPathToRoot, pPathToRoot, 4u * s.pathStart, 4u * s.numPathToRoots);
        guardIntact(n("pathToRoot"), s.oPathToRoot, 4u * s.numPathToRoots);
        zeroed(n("motionVelocities"), s.oMotionVel, 24u * s.nbLinks);
        guardIntact(n("motionVelocities"), s.oMotionVel, 24u * s.nbLinks);
        zeroed(n("motionAccelerations"), s.oMotionAccel, 24u * s.nbLinks);
        guardIntact(n("motionAccelerations"), s.oMotionAccel, 24u * s.nbLinks);
        zeroed(n("corioliseVectors"), s.oCoriolise, 24u * s.nbLinks);
        guardIntact(n("corioliseVectors"), s.oCoriolise, 24u * s.nbLinks);
        zeroed(n("linkSleepData"), s.oLinkSleep, 32u * s.nbLinks);
        guardIntact(n("linkSleepData"), s.oLinkSleep, 32u * s.nbLinks);

        // the five dof arrays: copied from dofData in the dirty order, zeroed otherwise
        const uint32_t bits[5] = {
            uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS),
            uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_VELOCITIES),
            uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FORCES),
            uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_POS),
            uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_VEL),
        };
        const size_t offs[5] = { s.oJointPos, s.oJointVel, s.oJointForce,
                                 s.oJointTgtPos, s.oJointTgtVel };
        const char* names[5] = { "jointPositions", "jointVelocities", "jointForce",
                                 "jointTargetPositions", "jointTargetVelocities" };
        uint32_t cursor = s.dofStart;
        for (int k = 0; k < 5; ++k)
        {
            if (s.dirtyFlags & bits[k])
            {
                std::vector<uint8_t> want(4u * s.numDofs);
                std::memcpy(want.data(), dofData.data() + cursor, want.size());
                region(n(names[k]), offs[k], want.data(), want.size());
                cursor += s.numDofs;
            }
            else
                zeroed(n(names[k]), offs[k], 4u * s.numDofs);
            guardIntact(n(names[k]), offs[k], 4u * s.numDofs);
        }

        if (s.dirtyFlags & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_EXT_ACCEL))
            copied(n("externalAccelerations"), s.oExtAccel, pExtAccel,
                   24u * s.linkStart, 24u * s.nbLinks);
        else
            zeroed(n("externalAccelerations"), s.oExtAccel, 24u * s.nbLinks);
        guardIntact(n("externalAccelerations"), s.oExtAccel, 24u * s.nbLinks);
    };
    char tag[16];
    for (uint32_t i = 0; i < NB_ARTI; ++i)
    {
        std::snprintf(tag, sizeof(tag), "arti%u", i);
        checkLinkArrays(tag, specs[i]);
    }

    // A's tendon params, and the per-tendon element copies through the records
    copied("A.spatialParams", A.oSpatialParams, pSpatialParams, 16u * A.spatialStart, 16u * A.nbSpatial);
    guardIntact("A.spatialParams", A.oSpatialParams, 16u * A.nbSpatial);
    copied("A.fixedParams", A.oFixedParams, pFixedParams, 32u * A.fixedStart, 32u * A.nbFixed);
    guardIntact("A.fixedParams", A.oFixedParams, 32u * A.nbFixed);
    copied("A.mimicJointCores", A.oMimic, pMimic, 32u * A.mimicStart, 32u * A.numMimic);
    guardIntact("A.mimicJointCores", A.oMimic, 32u * A.numMimic);
    copied("A.attachFixed", oAttachFixedDst, pAttachFixed, 16u * A_ATTACH_START, 16u * A_NB_ATTACH);
    guardIntact("A.attachFixed", oAttachFixedDst, 16u * A_NB_ATTACH);
    copied("A.attachMod", oAttachModDst, pAttachMod, 32u * A_ATTACH_START, 32u * A_NB_ATTACH);
    guardIntact("A.attachMod", oAttachModDst, 32u * A_NB_ATTACH);
    copied("A.tendonJointFixed", oTJointFixedDst, pTJointFixed, 16u * A_TJOINT_START, 16u * A_NB_TJOINT);
    guardIntact("A.tendonJointFixed", oTJointFixedDst, 16u * A_NB_TJOINT);
    copied("A.tendonJointCoeff", oTJointCoeffDst, pTJointCoeff, 16u * A_TJOINT_START, 16u * A_NB_TJOINT);
    guardIntact("A.tendonJointCoeff", oTJointCoeffDst, 16u * A_NB_TJOINT);

    // ---- the pool itself: the record lands in its OWN slot, nothing else moves ----
    std::vector<int> ownerOfSlot(POOL_SLOTS, -1);
    for (uint32_t i = 0; i < NB_ARTI; ++i)
    {
        PXG_CHECK_MSG(ownerOfSlot[specs[i].poolIndex] < 0, "pool slot %u claimed twice",
                      specs[i].poolIndex);
        ownerOfSlot[specs[i].poolIndex] = int(i);
    }
    for (uint32_t slot = 0; slot < POOL_SLOTS; ++slot)
    {
        std::vector<uint8_t> poison(sizeof(PxgArticulation), 0xAB);
        const uint8_t* want = ownerOfSlot[slot] >= 0
            ? reinterpret_cast<const uint8_t*>(&recs[size_t(ownerOfSlot[slot])])
            : poison.data();
        for (size_t i = 0; i < sizeof(PxgArticulation); ++i)
            if (gotPool[slot * sizeof(PxgArticulation) + i] != want[i])
            {
                if (bad < 12)
                    std::printf("  pool slot %u byte %zu: got 0x%02X want 0x%02X\n", slot, i,
                                gotPool[slot * sizeof(PxgArticulation) + i], want[i]);
                ++bad;
                break;
            }
    }

    // ---- the sleep-data pool: zeroed at both slots, poison elsewhere ----
    for (uint32_t slot = 0; slot < POOL_SLOTS; ++slot)
    {
        const uint8_t w = ownerOfSlot[slot] >= 0 ? 0x00 : 0xAB;
        for (size_t i = 0; i < sizeof(PxgSolverBodySleepData); ++i)
            if (gotSleep[slot * sizeof(PxgSolverBodySleepData) + i] != w)
            {
                if (bad < 12)
                    std::printf("  sleepData slot %u byte %zu: got 0x%02X want 0x%02X\n", slot, i,
                                gotSleep[slot * sizeof(PxgSolverBodySleepData) + i], w);
                ++bad;
                break;
            }
    }

    PXG_CHECK_MSG(bad == 0, "%zu region(s) wrong", bad);
    std::printf("  newArticulationsLaunch: %u articulations over %u blocks, pool slots "
                "3 and 1 in block 1, 1 spatial + 1 fixed tendon, disjoint dirty flags "
                "-- all regions exact\n", NB_ARTI, numBlocks);
}

} // namespace
