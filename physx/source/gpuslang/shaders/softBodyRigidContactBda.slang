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
import pxgSlangSoftBodyDesc;      // PxgSoftBodyGpu (mSimTetIndices@80, mSimDeltaPos@128, ...)
import pxgSlangFemRigidContact;   // prepareFEMContacts + the FRC_* block layout

// PxgSoftBodyKernelGridDim/BlockDim::SB_UPDATEROTATION — the engine's fixed launch
// (PxgSoftBodyCore.cpp:2259).  RC_BLOCK MUST equal [numthreads] below: the bridge forwards
// only gridDim, so the stride is reconstructed from these constants, not from blockDim.
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;

// PX_MAX_NB_DEFORMABLE_VOLUME_TET (PxDeformableVolume.h) == the element mask AND the
// invalid marker; PXG_BITSHIFT_TET_ID (PxgSimulationCoreDesc.h:40) — softbody id / tet
// index packing (PxGetSoftBodyId / PxGetSoftBodyElementIndex).
static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffffu;
static const uint PXG_BITSHIFT_TET_ID = 20u;

// PxgFemOtherContactInfo (PxgFEMCore.h:128, PX_ALIGN(16), 16 bytes):
//   PxU64 pairInd0 @0 (rigid id), PxU32 pairInd1 @8 (softbody compressed id),
//   PxU32 rigidMatInd_isInCollision @12 (bit31 = isInCollision).
static const uint CI_pairInd0 = 0u;
static const uint CI_pairInd1 = 8u;
static const uint CI_STRIDE   = 16u;

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

// barycentricProject (softBody.cu:640) — the FOUR-weight tetrahedron blend.  Note this is
// the tet form, not barycentricProjectTri (:631) which drops the w term.
float4 barycentricProject(uint4 tetrahedronIdx, Ptr<float4> position_invmass, float4 barycentric)
{
    const float4 a = position_invmass[tetrahedronIdx.x];
    const float4 b = position_invmass[tetrahedronIdx.y];
    const float4 c = position_invmass[tetrahedronIdx.z];
    const float4 d = position_invmass[tetrahedronIdx.w];

    return a * barycentric.x + b * barycentric.y + c * barycentric.z + d * barycentric.w;
}

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
        const uint workIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

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

        const float4 delta_invMass1 = barycentricProject(tetrahedronIdx, delta_invMass, barycentric);
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
