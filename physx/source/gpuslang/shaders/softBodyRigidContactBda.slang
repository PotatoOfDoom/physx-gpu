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

// softBodyRigidContactBda.slang
//
// P8.2 tranche d — chain kernel 5 of 7: sb_rigidContactPrepareLaunch (softBody.cu:773,
// launch PxgSoftBodyCore.cpp:2239 via prepRigidContactConstraint, SB_RS_CONTACTPREPARE).
//
// WHERE THIS SITS.  Tranches a-c got a softbody-vs-rigid contact GENERATED, re-addressed
// from the collision mesh to the simulation mesh, and sorted by rigid body.  Those contacts
// are still just geometry: a point, a normal, a penetration and a barycentric.  This kernel
// turns each one into a SOLVER CONSTRAINT -- the response terms (raXn and the two friction
// tangents, each scaled by the rigid body's sqrt inverse inertia), the error, and the
// penetration-bias clamp -- and writes them into the block-of-32 PxgFemRigidConstraintBlock
// that kernels 6 and 7 read back.  Nothing moves yet; this is the last step before the solve.
//
// The probe recorded in RUNTIME_STATUS confirmed the input actually exists: with a
// PxPlaneGeometry floor the mapped chain produces 128 contacts, onset timed to the step where
// the cube's bottom face crosses the plane.
//
// THIS IS A NEAR-CLONE of cloth_rigidContactPrepareLaunch (femClothRigidContactBda.slang),
// and deliberately so: the CUDA argument packs are position-for-position and type-for-type
// identical apart from argument 0 (PxgSoftBody* vs PxgFEMCloth*), so the ABI row is the same
// {8,8,8,8,8,8,8,8,8,8,4,8,1} = 104-byte blob that is already proven in the bridge.  The
// heavy part -- prepareFEMContacts, 78 lines of constraint-block writing -- is not copied:
// it was EXTRACTED into pxgSlangFemRigidContact and is now shared byte for byte by both
// shaders.  That extraction was verified to be a pure refactor: all five cloth entry points
// compile to byte-identical SPIR-V before and after it.
//
// WHAT ACTUALLY DIFFERS from the cloth kernel, in full:
//   - PxgSoftBodyGpu for PxgFEMClothGpu, and the softbody id/element packing
//     (PXG_BITSHIFT_TET_ID = 20, PX_MAX_NB_DEFORMABLE_VOLUME_TET) for the cloth one;
//   - mSimDeltaPos (@128) for mAccumulatedDeltaPos, and mSimTetIndices (@80) for
//     mTriangleVertexIndices;
//   - the barycentric projection is UNCONDITIONAL and over FOUR weights
//     (barycentricProject, softBody.cu:640).  The cloth version branches on
//     `barycentric.w == 0` between a three-weight triangle blend and a direct vertex read,
//     because a cloth contact can name either a triangle or a vertex.  A softbody contact
//     always names a tetrahedron, so there is no branch to reproduce -- and the fourth
//     weight is the tet's fourth vertex, not a discriminator.
//
// SEMANTICS THAT ARE EASY TO LOSE, and are preserved deliberately:
//   - the normal handed to prepareFEMContacts is NEGATED (softBody.cu:835): the contact
//     normal points from the softbody toward the rigid, and the constraint wants the
//     opposite.  Dropping the negation still produces plausible-looking constraints that
//     push the wrong way;
//   - the penetration is `normal_pen.w - rest` (:836), i.e. the REST DISTANCE is subtracted
//     from the raw penetration.  contacts_restW.w carries that rest distance, which is why
//     the contact buffer is a float4 and not a float3;
//   - `rigidAppliedForces[workIndex] = 0.0f` is written for EVERY contact in range, before
//     the tetInd guard (:803).  It is the accumulator kernels 6 and 7 read, so a contact
//     skipped by the guard must still have had its slot cleared;
//   - the guard is `continue`, not `return` (:819).  A contact naming tet 0xfffff is
//     skipped, but the thread keeps striding -- unlike the out-of-range check above it,
//     which is a real `return`.
//
// DEVIATIONS from CUDA:
//   - invDt and isTGS are in the push blob (the ABI row mirrors the CUDA arg pack) and are
//     NOT READ.  That is not an omission: CUDA takes both and never reads them either --
//     brace-matched over the whole 122-line prepareFEMContacts body
//     (deformableUtils.cuh:132-253) they appear on the signature line only;
//   - sharedDesc is in the blob and not dereferenced, because the only thing CUDA reads
//     from it is mBodySimBufferDeviceData at deformableUtils.cuh:162, inside
//     `if (rigidId.isArticulation())`.  The articulation branch is DEFERRED, as in the cloth
//     port; a static plane floor never takes it;
//   - the grid is reproduced as constants.  The engine launches grid/block
//     PxgSoftBodyKernelGridDim::SB_UPDATEROTATION x SB_UPDATEROTATION and the bridge
//     forwards only gridDim, so RC_BLOCK must equal this kernel's [numthreads] -- the same
//     arrangement the cloth kernel uses, and the grid-mismatch class that bit
//     femClothSelfCollisionBda.
//
// THE TWO llvmpipe LANDMINES from tranche c apply here and are avoided by construction:
// there is no optionally-null pointer in this kernel (every push-blob pointer is always
// valid), and no block redefines the strided loop's bound, so neither the speculated-load
// nor the dropped-loop-mask shape is present.  Any future edit that introduces one needs
// the pointer SELECTED rather than the load guarded, and a `min()` plus a flat conditional
// store rather than a clamp-and-store block.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the response terms are dot-of-cross
// chains through mat33MulVec that would otherwise fuse to FMAs, and the constraint block is
// compared bit-exactly against a CPU transcription.  No transcendental except the sqrt in
// computeBasisVectors, which is correctly rounded -- fully bit-exact, no tolerance seam.

import pxgSlangCommon;            // PxgSolverBodyData, PxgSolverTxIData
import pxgSlangPrepDesc;          // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;        // PxgSolverCoreDescGpu, PxgSolverSharedDesc{PGS,TGS}Gpu
import pxgSlangSoftBodyDesc;
import pxgSlangSoftBodySolve;   // P9.39d: barycentricProject, one copy for the corpus
import pxgSlangAtomicFloat;       // pxgAtomicAddF4W / pxgAtomicAddF4XYZ (P8.1)
import pxgSlangFemRigidContact;   // prepareFEMContacts, contactInitialize, readRigidVelocity

// PxgSoftBodyKernelGridDim/BlockDim::SB_UPDATEROTATION — the engine's fixed launch
// (PxgSoftBodyCore.cpp:2259).  RC_BLOCK MUST equal [numthreads] below: the bridge forwards
// only gridDim, so the stride is reconstructed from these constants, not from blockDim.
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;

// PX_MAX_NB_DEFORMABLE_VOLUME_TET (PxDeformableVolume.h) == the element mask AND the
// invalid marker; PXG_BITSHIFT_TET_ID (PxgSimulationCoreDesc.h:40) — softbody id / tet
// index packing (PxGetSoftBodyId / PxGetSoftBodyElementIndex).
// PX_MAX_NB_DEFORMABLE_VOLUME_TET and PXG_BITSHIFT_TET_ID now come from
// pxgSlangSoftBodySolve (P9.39d) -- they were declared here as well and the
// import made the duplicate an ambiguity error rather than a silent second copy.

// PxgFemOtherContactInfo (PxgFEMCore.h:128, PX_ALIGN(16), 16 bytes):
//   PxU64 pairInd0 @0 (rigid id), PxU32 pairInd1 @8 (softbody compressed id),
//   PxU32 rigidMatInd_isInCollision @12 (bit31 = isInCollision).
static const uint CI_pairInd0 = 0u;
static const uint CI_pairInd1 = 8u;
static const uint CI_rigidMatInd = 12u;
static const uint CI_STRIDE   = 16u;
static const uint CI_COLLISION_FLAG = 0x80000000u;

// ---------------------------------------------------------------------------
// sb_rigidContactPrepareLaunch push blob.
// CUDA arg pack (PxgSoftBodyCore.cpp:2242): { PxgSoftBody* softbodies,
// float4* contacts_restW, float4* normalPens, float4* barycentrics,
// PxgFemOtherContactInfo* contactInfos, PxU32* numContacts,
// PxgFemRigidConstraintBlock* primitiveConstraints, PxgPrePrepDesc* preDesc,
// PxgConstraintPrepareDesc* prepareDesc, PxReal* rigidAppliedForces, PxReal invDt,
// PxgSolverSharedDescBase* sharedDesc, bool isTGS }.  Ten device pointers, then PxReal
// @80, then a pointer realigned to @88, then bool @96 -> 104-byte blob.
// (ABI row {8,8,8,8,8,8,8,8,8,8,4,8,1} — byte-identical to cloth_rigidContactPrepareLaunch.)
// ---------------------------------------------------------------------------
struct SbPrepContactParams
{
    uint64_t softbodies;          // @0
    uint64_t contacts;            // @8   contacts_restW: xyz = point, w = rest distance
    uint64_t normalPens;          // @16  xyz = normal, w = penetration
    uint64_t barycentrics;        // @24
    uint64_t contactInfos;        // @32
    uint64_t numContacts;         // @40
    uint64_t constraints;         // @48  output
    uint64_t prePrepDesc;         // @56
    uint64_t prepDesc;            // @64
    uint64_t rigidAppliedForces;  // @72  output (zeroed here)
    float invDt;                  // @80  (DEFERRED: unread, as in CUDA)
    uint _pad84;                  // @84
    uint64_t sharedDesc;          // @88  (DEFERRED: articulation/bodySim never dereferenced)
    uint isTGS;                   // @96  (bool, byte0; DEFERRED: unread, as in CUDA)
    uint _pad100;                 // @100 -> 104
}
[[vk::push_constant]]
ConstantBuffer<SbPrepContactParams> gSbPrep;


[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK; PxgSoftBodyKernelBlockDim::SB_UPDATEROTATION
void sb_rigidContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSbPrep.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSbPrep.softbodies);
    Ptr<float> rigidAppliedForces = Ptr<float>(gSbPrep.rigidAppliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;

        // P9.23: THE GUARD ABOVE IS PER-LANE, so this platform turns it into a
        // mask -- every read below is still ISSUED for the lanes it excludes,
        // with an address built from the unclamped index.  That is the defect
        // P9.22 measured as a crash (see countAt in npRigidSumBatches.slang).
        // Clamping once here keeps those speculated addresses inside the
        // allocation; for any lane the guard admits, workIndex == rawWorkIndex,
        // so nothing computed changes.  tNumContacts is at least 1 wherever this
        // line runs -- nbIterationsPerBlock is 0 otherwise, and that condition is
        // uniform, so the loop is not entered at all.
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);

        // Cleared for EVERY in-range contact, before the tet guard below -- see the header.
        rigidAppliedForces[workIndex] = 0.0f;

        const uint64_t ciBase = gSbPrep.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        // First actor: rigid body.  Second actor: soft body.
        const uint64_t rigidId = Ptr<uint64_t>(ciBase + CI_pairInd0)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + CI_pairInd1)[0];

        const uint softBodyId = pairInd1 >> PXG_BITSHIFT_TET_ID;
        const uint tetInd = pairInd1 & PX_MAX_NB_DEFORMABLE_VOLUME_TET;

        // `continue`, NOT `return`: the thread keeps striding (softBody.cu:819).
        if (tetInd == PX_MAX_NB_DEFORMABLE_VOLUME_TET)
            continue;

        const uint4 tetrahedronIdx =
            Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[tetInd];
        Ptr<float4> delta_invMass = Ptr<float4>(softbodies[softBodyId].mSimDeltaPos);

        const float4 barycentric = Ptr<float4>(gSbPrep.barycentrics)[workIndex];
        const float4 contact_restW = Ptr<float4>(gSbPrep.contacts)[workIndex];
        const float4 normal_pen = Ptr<float4>(gSbPrep.normalPens)[workIndex];

        const float3 p = float3(contact_restW.x, contact_restW.y, contact_restW.z);
        const float rest = contact_restW.w;

        const float4 delta_invMass1 = barycentricProject(delta_invMass, tetrahedronIdx, barycentric);
        const float3 delta = float3(delta_invMass1.x, delta_invMass1.y, delta_invMass1.z);

        // NEGATED normal, and the rest distance subtracted from the penetration.
        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);
        const float pen = normal_pen.w - rest;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gSbPrep.constraints + uint64_t(index) * FRC_STRIDE;

        prepareFEMContacts(cBase, offset, normal, p, pen, delta, rigidId, barycentric,
                           gSbPrep.prepDesc, gSbPrep.prePrepDesc,
                           softbodies[softBodyId].mPenBiasClamp);
    }
}

// ===========================================================================
// P8.2 tranche e — chain kernels 6 and 7: the softbody-vs-rigid COLLISION SOLVE.
//
//   sb_queryRigidSoftContactReferenceCountLaunch (softBody.cu:1944, helper :1824, launch
//   PxgSoftBodyCore.cpp:1756) — a checkOnlyActivity pass.  It decides which contacts are
//   ACTIVE, marks them in-collision, and bumps a reference count on every tet vertex the
//   contact touches (and on the rigid body).  Those counts are the MASS-SPLITTING
//   denominators the solve below divides by, so this pass is not bookkeeping -- get the
//   counts wrong and every contact applies the wrong share of its impulse.
//
//   sb_solveRigidSoftCollisionLaunch (softBody.cu:2062, helper :1963, launch :1040) — the
//   solve proper.  For each active contact it evaluates the constraint, accumulates the
//   applied force, and scatters the position delta onto the four tet vertices.  This is the
//   kernel that finally MOVES the softbody in response to a rigid contact.
//
// Both are near-clones of cloth_queryRigidClothContactReferenceCountLaunch and
// cloth_solveRigidClothCollisionLaunch, with the same argument packs (positions and types
// identical apart from argument 0 and the material type), hence the same proven ABI rows.
// The shared machinery -- readRigidVelocity (PxgVelocityReader), contactInitialize
// (FEMCollision::initialize) and getGlobalRigidBodyId -- lives in pxgSlangFemRigidContact.
//
// WHAT DIFFERS from the cloth pair:
//   - FOUR tet vertices instead of a triangle-or-vertex branch.  CUDA templates
//     FEMCollision on PxVec4 here and PxVec3 there; the only place the width reaches the
//     shared code is the mass-splitting dot, which is why contactInitialize takes it as a
//     scalar (see the module);
//   - the reference counts live in mSimDelta.w and the deltas/inverse masses in
//     mSimDeltaPos -- TWO different buffers, as in cloth (mDeltaPos / mAccumulatedDeltaPos);
//   - readSoftBody multiplies the inverse masses BY the reference counts
//     (deformableUtils.cuh:614).  By, not divided by -- that is upstream's mass-splitting
//     convention and reproducing it is the whole point of the query pass.
//
// DEFERRED, exactly as in the cloth port: the ARTICULATION branch.
//
// P9.46: FRICTION is no longer deferred.  `materials` and `rigidBodyMaterials` are both
// read in the solve now -- the softbody half through the per-tet material index
// (deformableUtils.cuh:604), the rigid half through the contact info's material index
// (deformableUtils.cuh:469) -- and PxsCombinePxReal joins them under the RIGID material's
// fricCombineMode.  See pxgSlangFemRigidContact.slang for the readers.
//
// P9.45: the DYNAMIC-rigid REACTION is no longer deferred.  computeRigidChange
// (deformableUtils.cuh:418, called at softBody.cu:2046-2051) runs under the .cu's own
// guard and writes count 1 plus the two delta rows into rigidDeltaVel, which P9.44's
// accumulateDeltaVRigidFirst/SecondLaunch reduce onto the solver body.  A static or
// kinematic contact still writes zeros with count 0.  An ARTICULATION-link contact also
// still writes zeros, and that is a consequence of the articulation deferral above rather
// than a new one: without the articulation anchor rows from prepareFEMContacts and without
// the link velocity in readRigidVelocity, a reaction would be wrong, not merely missing.
// ===========================================================================

// PxgSolverCoreDesc::numSolverBodies @268 (pxgSlangSolverDesc mirror).
// FEMCollision::readSoftBody (deformableUtils.cuh:580) — the FOUR-vertex tet form.
// Returns the (optionally mass-split) per-vertex inverse masses and fills the blended
// linear delta.  countReferenceOnly mirrors CUDA's checkOnlyActivity: on that path the
// reference-count multiply is SKIPPED, because the counts are still being accumulated.
float4 readSoftBodyInvMasses(Ptr<PxgSoftBodyGpu> softbodies, uint softBodyId, uint tetId,
                             float4 bc, bool countReferenceOnly, out float3 deformableLinDelta,
                             uint64_t materials, out float deformableFriction)
{
    Ptr<float4> posDeltas = Ptr<float4>(softbodies[softBodyId].mSimDeltaPos);
    const uint4 tetId4 = Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[tetId];

    // The CUDA's FEMCollision member default (deformableUtils.cuh:270).
    deformableFriction = 0.0f;

    const float4 pd0 = posDeltas[tetId4.x];
    const float4 pd1 = posDeltas[tetId4.y];
    const float4 pd2 = posDeltas[tetId4.z];
    const float4 pd3 = posDeltas[tetId4.w];

    const float4 softBodyDelta = pd0 * bc.x + pd1 * bc.y + pd2 * bc.z + pd3 * bc.w;
    deformableLinDelta = float3(softBodyDelta.x, softBodyDelta.y, softBodyDelta.z);

    float4 invM = float4(pd0.w, pd1.w, pd2.w, pd3.w);
    if (!countReferenceOnly)
    {
        // P9.46: unlike the cloth there is ONE friction source -- the per-tet material index
        // into the volume material table (deformableUtils.cuh:604).  The volume and surface
        // records both put dynamicFriction at @8 with stride 32, so the same loader serves.
        deformableFriction = femMaterialFrictionAt(
            materials, softbodies[softBodyId].mMaterialIndices, tetId);

        // Mass-splitting by the per-vertex reference counts (mSimDelta.w).
        Ptr<float4> simDelta = Ptr<float4>(softbodies[softBodyId].mSimDelta);
        invM = float4(invM.x * simDelta[tetId4.x].w, invM.y * simDelta[tetId4.y].w,
                      invM.z * simDelta[tetId4.z].w, invM.w * simDelta[tetId4.w].w);
    }
    return invM;
}

// the 4-wide mass-splitting term, bc.multiply(bc).dot(invM) -- the tetrahedron form of what
// the cloth callers compute over three components.
float tetMassSplit(float4 bc, float4 invM)
{
    return bc.x * bc.x * invM.x + bc.y * bc.y * invM.y
         + bc.z * bc.z * invM.z + bc.w * bc.w * invM.w;
}

// ---------------------------------------------------------------------------
// sb_queryRigidSoftContactReferenceCountLaunch push blob.
// CUDA arg pack (PxgSoftBodyCore.cpp:1766): { PxgSoftBody* softbodies,
// PxgFemOtherContactInfo* contactInfos, PxgFemRigidConstraintBlock* constraints,
// PxU32* numContacts, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
// PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc, PxReal dt,
// PxReal* appliedForces, PxU32* rigidBodyReferenceCounts } -> 88-byte blob.
// (ABI row {8,8,8,8,8,8,8,8,4,8,8} — identical to the cloth query row.)
// ---------------------------------------------------------------------------
struct SbQueryContactParams
{
    uint64_t softbodies;                // @0
    uint64_t contactInfos;              // @8
    uint64_t constraints;               // @16
    uint64_t numContacts;               // @24
    uint64_t prePrepDesc;               // @32
    uint64_t solverCoreDesc;            // @40
    uint64_t artiCoreDesc;              // @48 (DEFERRED: articulation never dereferenced)
    uint64_t sharedDesc;                // @56
    float dt;                           // @64
    uint _pad68;                        // @68
    uint64_t appliedForces;             // @72
    uint64_t rigidBodyReferenceCounts;  // @80 -> 88
}
[[vk::push_constant]]
ConstantBuffer<SbQueryContactParams> gSbQuery;

// ---------------------------------------------------------------------------
// sb_solveRigidSoftCollisionLaunch push blob.
// CUDA arg pack (PxgSoftBodyCore.cpp:1050): the query pack with float4* rigidDeltaVel
// inserted at @64 and { PxReal dt, PxsDeformableVolumeMaterialData* materials,
// const PxsMaterialData* rigidBodyMaterials } appended -> 112-byte blob.
// (ABI row {8,8,8,8,8,8,8,8,8,8,8,4,8,8} — identical to the cloth solve row.)
// ---------------------------------------------------------------------------
struct SbSolveContactParams
{
    uint64_t softbodies;                // @0
    uint64_t contactInfos;              // @8
    uint64_t constraints;               // @16
    uint64_t numContacts;               // @24
    uint64_t prePrepDesc;               // @32
    uint64_t solverCoreDesc;            // @40
    uint64_t artiCoreDesc;              // @48 (DEFERRED)
    uint64_t sharedDesc;                // @56
    uint64_t rigidDeltaVel;             // @64  output
    uint64_t appliedForces;             // @72  in/out
    uint64_t rigidBodyReferenceCounts;  // @80
    float dt;                           // @88
    uint _pad92;                        // @92
    uint64_t materials;                 // @96  PxsDeformableVolumeMaterialData*
    uint64_t rigidBodyMaterials;        // @104 PxsMaterialData* -> 112
}
[[vk::push_constant]]
ConstantBuffer<SbSolveContactParams> gSbSolve;

void queryRigidSoftContactReferenceCount(bool isTGS, uint idx, uint blockX)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSbQuery.solverCoreDesc)[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gSbQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSbQuery.softbodies);
    Ptr<float> appliedForces = Ptr<float>(gSbQuery.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * blockX * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;

        // P9.23: THE GUARD ABOVE IS PER-LANE, so this platform turns it into a
        // mask -- every read below is still ISSUED for the lanes it excludes,
        // with an address built from the unclamped index.  That is the defect
        // P9.22 measured as a crash (see countAt in npRigidSumBatches.slang).
        // Clamping once here keeps those speculated addresses inside the
        // allocation; for any lane the guard admits, workIndex == rawWorkIndex,
        // so nothing computed changes.  tNumContacts is at least 1 wherever this
        // line runs -- nbIterationsPerBlock is 0 otherwise, and that condition is
        // uniform, so the loop is not entered at all.
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);

        const uint64_t ciBase = gSbQuery.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint rigidMatInd = Ptr<uint>(ciBase + CI_rigidMatInd)[0];
        const bool wasActive = (rigidMatInd & CI_COLLISION_FLAG) != 0u;
        const uint64_t rigidId = Ptr<uint64_t>(ciBase + CI_pairInd0)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + CI_pairInd1)[0];

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gSbQuery.constraints + uint64_t(index) * FRC_STRIDE;

        const float4 fricTan0_invMass0 = Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset];
        const float4 bc = Ptr<float4>(cBase + FRC_barycentric)[offset];

        const uint softBodyId = pairInd1 >> PXG_BITSHIFT_TET_ID;
        const uint tetId = pairInd1 & PX_MAX_NB_DEFORMABLE_VOLUME_TET;

        if (tetId < PX_MAX_NB_DEFORMABLE_VOLUME_TET)
        {
            const int globalRigidBodyId = getGlobalRigidBodyId(rigidId, gSbQuery.prePrepDesc, numSolverBodies);

            float3 deformableLinDelta;
            float queryFriction;   // unused: the query never reads a material table
            const float4 deformableInvMasses =
                readSoftBodyInvMasses(softbodies, softBodyId, tetId, bc,
                                      /*countReferenceOnly*/ true, deformableLinDelta,
                                      /*materials*/ 0UL, queryFriction);

            bool isActive;
            if (wasActive)
            {
                // Once activated, keep the pair active to maintain a CONSERVATIVE reference
                // count (softBody.cu:1888).  Dropping this makes the counts oscillate.
                isActive = true;
            }
            else
            {
                const float3 bcVec = float3(bc.x, bc.y, bc.z);
                const ContactResult cr =
                    contactInitialize(cBase, offset, rigidId, isTGS, appliedForces[workIndex],
                                      gSbQuery.dt, bcVec, deformableLinDelta,
                                      tetMassSplit(bc, deformableInvMasses),
                                      /*rigidRefCount*/ 1u, wasActive, /*checkOnly*/ true,
                                      gSbQuery.solverCoreDesc, gSbQuery.sharedDesc, gSbQuery.prePrepDesc,
                                      // NULL material tables on the query path (softBody.cu:1895),
                                      // and checkOnlyActivity returns before the tangent block.
                                      /*combinedFriction*/ 0.0f);
                isActive = cr.isActive;
                if (isActive)
                    Ptr<uint>(ciBase + CI_rigidMatInd)[0] = rigidMatInd | CI_COLLISION_FLAG;
            }

            if (isActive)
            {
                // Increment the reference count of the four tet vertices.  The bc > 1e-3
                // gate is upstream's (softBody.cu:1911): a vertex a contact barely touches
                // must not claim a share of the mass split.
                const uint4 tetId4 = Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[tetId];
                const uint64_t simDelta = softbodies[softBodyId].mSimDelta;
                if (bc.x > 1e-3f && deformableInvMasses.x != 0.0f) pxgAtomicAddF4W(simDelta, tetId4.x, 1.0f);
                if (bc.y > 1e-3f && deformableInvMasses.y != 0.0f) pxgAtomicAddF4W(simDelta, tetId4.y, 1.0f);
                if (bc.z > 1e-3f && deformableInvMasses.z != 0.0f) pxgAtomicAddF4W(simDelta, tetId4.z, 1.0f);
                if (bc.w > 1e-3f && deformableInvMasses.w != 0.0f) pxgAtomicAddF4W(simDelta, tetId4.w, 1.0f);

                if (globalRigidBodyId != -1 && fricTan0_invMass0.w != 0.0f)
                {
                    uint prev;
                    InterlockedAdd((Ptr<uint>(gSbQuery.rigidBodyReferenceCounts))[globalRigidBodyId], 1u, prev);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)] // == RC_BLOCK; SB_UPDATEROTATION
void sb_queryRigidSoftContactReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    queryRigidSoftContactReferenceCount(false, groupThreadId.x, groupId.x);
}

[shader("compute")]
[numthreads(256, 1, 1)] // == RC_BLOCK; SB_UPDATEROTATION
void sb_queryRigidSoftContactReferenceCountLaunchTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    queryRigidSoftContactReferenceCount(true, groupThreadId.x, groupId.x);
}

void solveRigidSoftContact(bool isTGS, uint idx, uint blockX)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSbSolve.solverCoreDesc)[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gSbSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSbSolve.softbodies);
    Ptr<float> appliedForces = Ptr<float>(gSbSolve.appliedForces);
    Ptr<float4> rigidDeltaVel = Ptr<float4>(gSbSolve.rigidDeltaVel);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * blockX * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;

        // P9.23: THE GUARD ABOVE IS PER-LANE, so this platform turns it into a
        // mask -- every read below is still ISSUED for the lanes it excludes,
        // with an address built from the unclamped index.  That is the defect
        // P9.22 measured as a crash (see countAt in npRigidSumBatches.slang).
        // Clamping once here keeps those speculated addresses inside the
        // allocation; for any lane the guard admits, workIndex == rawWorkIndex,
        // so nothing computed changes.  tNumContacts is at least 1 wherever this
        // line runs -- nbIterationsPerBlock is 0 otherwise, and that condition is
        // uniform, so the loop is not entered at all.
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);

        const uint64_t ciBase = gSbSolve.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint rigidMatInd = Ptr<uint>(ciBase + CI_rigidMatInd)[0];
        const bool isActive = (rigidMatInd & CI_COLLISION_FLAG) != 0u;
        const uint64_t rigidId = Ptr<uint64_t>(ciBase + CI_pairInd0)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + CI_pairInd1)[0];

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gSbSolve.constraints + uint64_t(index) * FRC_STRIDE;

        const float4 fricTan0_invMass0 = Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset];
        const float4 bc = Ptr<float4>(cBase + FRC_barycentric)[offset];

        const uint softBodyId = pairInd1 >> PXG_BITSHIFT_TET_ID;
        const uint tetId = pairInd1 & PX_MAX_NB_DEFORMABLE_VOLUME_TET;

        if (tetId < PX_MAX_NB_DEFORMABLE_VOLUME_TET)
        {
            float3 deltaLinVel0 = float3(0.0f, 0.0f, 0.0f);
            float3 deltaAngVel0 = float3(0.0f, 0.0f, 0.0f);
            float count = 0.0f;

            if (isActive)
            {
                // readRigidBody: the reference count is 1 for a static body
                // (globalRigidBodyId == -1); the friction half follows below.
                const int globalRigidBodyId = getGlobalRigidBodyId(rigidId, gSbSolve.prePrepDesc, numSolverBodies);
                uint rigidBodyReferenceCount = 1u;
                if (globalRigidBodyId != -1 && fricTan0_invMass0.w != 0.0f)
                    rigidBodyReferenceCount = Ptr<uint>(gSbSolve.rigidBodyReferenceCounts)[globalRigidBodyId];

                float3 deformableLinDelta;
                float softBodyFriction;
                const float4 deformableInvMasses =
                    readSoftBodyInvMasses(softbodies, softBodyId, tetId, bc,
                                          /*countReferenceOnly*/ false, deformableLinDelta,
                                          gSbSolve.materials, softBodyFriction);

                // P9.46: the same combine as the cloth side, under the RIGID material's mode.
                // The rigid material index is the low 31 bits of the contact info word the
                // query ORs its in-collision flag into (PxgFEMCore.h:160).
                const RigidFriction rf = readRigidFriction(gSbSolve.rigidBodyMaterials,
                                                           rigidMatInd & ~CI_COLLISION_FLAG);
                const float combinedFriction = combineFriction(rf.friction, softBodyFriction,
                                                               rf.combineMode);

                const float3 bcVec = float3(bc.x, bc.y, bc.z);
                const ContactResult cr =
                    contactInitialize(cBase, offset, rigidId, isTGS, appliedForces[workIndex],
                                      gSbSolve.dt, bcVec, deformableLinDelta,
                                      tetMassSplit(bc, deformableInvMasses),
                                      rigidBodyReferenceCount, isActive, /*checkOnly*/ false,
                                      gSbSolve.solverCoreDesc, gSbSolve.sharedDesc, gSbSolve.prePrepDesc,
                                      combinedFriction);

                // computeFEMChange: deltaPos = -(deltaLambdaN*normal + deltaLambdaT*tangent)*dt.
                const float3 deltaPos = float3(
                    -(cr.deltaLambdaN * cr.normal.x + cr.deltaLambdaT * cr.tangent.x) * gSbSolve.dt,
                    -(cr.deltaLambdaN * cr.normal.y + cr.deltaLambdaT * cr.tangent.y) * gSbSolve.dt,
                    -(cr.deltaLambdaN * cr.normal.z + cr.deltaLambdaT * cr.tangent.z) * gSbSolve.dt);
                appliedForces[workIndex] = cr.accumulatedDeltaLambdaN;

                // writeSoftBody (deformableUtils.cuh:620): scatter onto the four vertices,
                // each scaled by ITS OWN mass-split inverse mass times ITS barycentric.
                const uint4 tetId4 = Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[tetId];
                const uint64_t simDelta = softbodies[softBodyId].mSimDelta;
                if (deformableInvMasses.x > 0.0f)
                    pxgAtomicAddF4XYZ(simDelta, tetId4.x, deltaPos * (deformableInvMasses.x * bc.x));
                if (deformableInvMasses.y > 0.0f)
                    pxgAtomicAddF4XYZ(simDelta, tetId4.y, deltaPos * (deformableInvMasses.y * bc.y));
                if (deformableInvMasses.z > 0.0f)
                    pxgAtomicAddF4XYZ(simDelta, tetId4.z, deltaPos * (deformableInvMasses.z * bc.z));
                if (deformableInvMasses.w > 0.0f)
                    pxgAtomicAddF4XYZ(simDelta, tetId4.w, deltaPos * (deformableInvMasses.w * bc.w));

                // computeRigidChange (softBody.cu:2046-2051, deformableUtils.cuh:418).  P9.45.
                // A STATIC body still leaves count and the deltas at 0 -- rigidId.isStaticBody()
                // is the .cu's own guard, and fricTan0_invMass0.w is 0 for a kinematic one.
                // The ARTICULATION case stays out: prepareFEMContacts defers the articulation
                // anchor build (pxgSlangFemRigidContact.slang:143), so raXn_resp would carry
                // the rigid response rows for a link, and readRigidVelocity defers the link
                // velocity read -- a reaction computed from either would be wrong, not just
                // absent.  Writing zeros for a link is what this module already did.
                if (fricTan0_invMass0.w != 0.0f && !nodeIsStaticBody(rigidId)
                    && !nodeIsArticulation(rigidId))
                {
                    count = 1.0f;
                    const RigidChange rc = computeRigidChange(cr, /*isArticulation*/ false,
                                                              fricTan0_invMass0.w,
                                                              rigidBodyReferenceCount);
                    deltaLinVel0 = rc.deltaLinVel0;
                    deltaAngVel0 = rc.deltaAngVel0;
                }
            }

            // writeRigidBody: for a static contact CUDA writes zeros with count 0.
            rigidDeltaVel[workIndex] = float4(deltaLinVel0.x, deltaLinVel0.y, deltaLinVel0.z, count);
            rigidDeltaVel[workIndex + tNumContacts] = float4(deltaAngVel0.x, deltaAngVel0.y, deltaAngVel0.z, 0.0f);
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)] // == RC_BLOCK; SB_UPDATEROTATION
void sb_solveRigidSoftCollisionLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    solveRigidSoftContact(false, groupThreadId.x, groupId.x);
}

[shader("compute")]
[numthreads(256, 1, 1)] // == RC_BLOCK; SB_UPDATEROTATION
void sb_solveRigidSoftCollisionLaunchTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    solveRigidSoftContact(true, groupThreadId.x, groupId.x);
}
