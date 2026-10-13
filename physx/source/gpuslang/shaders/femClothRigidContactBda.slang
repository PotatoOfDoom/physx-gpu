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

// femClothRigidContactBda — P7.1 tranche 4b-2/3: the FEM-cloth RIGID-CONTACT
// prepare + solve kernels on the Vulkan engine ABI (BDA).  The machinery that pushes
// a cloth vertex OUT of a rigid primitive (toward a cloth draping on a static floor /
// sphere), ported line-for-line from
//   gpusimulationcontroller/src/CUDA/FEMClothConstraintPrep.cu (the prepare, :70)
//   gpusimulationcontroller/src/CUDA/FEMClothExternalSolve.cu   (the solves, :288 / :433)
// with the prepareFEMContacts / FEMCollision leaves from src/CUDA/deformableUtils.cuh:
//   - cloth_rigidContactPrepareLaunch (FEMClothConstraintPrep.cu:70, launch
//     PxgFEMClothCore.cpp:326): builds the block-of-32 PxgFemRigidConstraintBlock from
//     each SORTED cloth-vs-rigid contact (contact point, flipped normal, penetration,
//     accumulated-delta, barycentric) via prepareFEMContacts (deformableUtils.cuh:132)
//     — the rigid response rows (raXn x sqrtInvInertia, the friction tangents t0/t1 via
//     PxComputeBasisVectors) and the position error.
//   - cloth_queryRigidClothContactReferenceCount{,TGS}Launch (FEMClothExternalSolve.cu:
//     414 / :558, launch PxgFEMClothCore.cpp:1733): a checkOnlyActivity pass that marks
//     each penetrating contact in-collision and increments the referenced cloth
//     vertex's mDeltaPos.w reference count (so applyExternalDeltas can mass-average).
//   - cloth_solveRigidClothCollision{,TGS}Launch (FEMClothExternalSolve.cu:535 / :578,
//     launch PxgFEMClothCore.cpp:1778): solves the contact constraint (FEMCollision::
//     initialize, deformableUtils.cuh:280) and scatters the cloth-side position
//     correction into mDeltaPos (folded next frame by the already-mapped
//     cloth_applyExternalDeltasLaunch).
//
// SCOPE — cloth vs a STATIC floor / sphere (the drape target).  For a static body the
// rigid side has zero velocity + infinite mass + zero inverse inertia, so the reaction
// path collapses: the same precedent as the P7.1 tranche 3a attachment port, we DEFER
// the dynamic-rigid reaction (see the "DEVIATIONS" list).  The cloth-side push is exact
// (bit-exact vs the CPU transcription in testFemClothSim.cpp), PGS and TGS.
//
// DEVIATIONS from the CUDA original (recipe policy; each preserves the constraint
// semantics for a cloth-vs-STATIC-rigid contact and is exercised by testFemClothSim.cpp):
//   - the descriptors arrive as device pointers read through Ptr<> (BDA), reusing the
//     byte-exact pxgSlang* mirrors (FEMClothDesc / PrepDesc / SolverDesc / Common).  The
//     block-of-32 PxgFemRigidConstraintBlock and the 16-byte PxgFemOtherContactInfo are
//     addressed by explicit byte offset through Ptr<> (no descriptor block copy), exactly
//     as the other BDA kernels read their SoA blocks;
//   - the ARTICULATION branch is DEFERRED (a static floor / sphere is never an
//     articulation: rigidId.isArticulation()==false).  Both prepareFEMContacts's
//     articulation constraint build (spatialResponseMatrixW path) and the solve /
//     query velocity-reader's articulation link-velocity read are omitted — same
//     precedent as psContactSolveTgsBda / femClothRigidAttachBda, which sink the
//     outArtiVelocity read behind the isArticulation guard so a rigid / static contact
//     never dereferences it;
//   - the DYNAMIC-rigid REACTION is PORTED as of P9.45.  FEMCollision::computeRigidChange
//     (deformableUtils.cuh:418) runs under the .cu's own guard
//     (fricTan0_invMass0.w != 0 && !rigidId.isStaticBody()), writing count 1 and the two
//     delta rows into rigidDeltaVel; the query pass's rigid reference-count increment
//     (:404-408) is ported with it, so the mass-splitting divisor the solve reads back is
//     the real per-body contact count instead of the default 1.  P9.44's
//     accumulateDeltaVRigidFirst/SecondLaunch reduce those rows onto the solver body, which
//     is what makes a cloth able to HOLD a dynamic rigid (scene `clothnet`).  A static or
//     kinematic contact still writes (0,0,0,0) with count 0, exactly as before.  The
//     ARTICULATION-link case is the one that stays out, and it is an extension of the
//     articulation deferral above, not a separate one: prepareFEMContacts never builds the
//     articulation anchor rows and readRigidVelocity never reads the link velocity, so a
//     reaction derived from either would be WRONG rather than merely absent.  A link
//     contact therefore keeps writing zeros, as this module always did;
//   - FRICTION is the null-material path.  prepareFEMContacts's t0/t1 (PxComputeBasis-
//     Vectors) and the friction response rows raXnF0/raXnF1 are built verbatim, and the
//     solve computes the full tangent kinematics (tanVel / CT0 / CT1 / relTanDelta /
//     tanMagSq and the tangent impulse), but getCombinedFriction() == 0 — the readCloth /
//     readRigidBody NULL-material branch the CUDA's OWN query pass takes
//     (FEMClothExternalSolve.cu:362 passes NULL,NULL).  With zero combined friction
//     deltaLambdaT == -min(mag, 0) == -0, so the cloth-side push is bit-exact with a
//     frictionless (zero dynamicFriction) rigid+cloth material — the drape target.  The
//     material-strided friction read (PxsMaterialData::dynamicFriction / cloth
//     mDynamicFrictions) is a follow-on tranche;
//   - cloth.mDeltaPos accumulation uses REAL atomics (pxgSlangAtomicFloat), matching the
//     CUDA atomicAdd(&mDeltaPos[v].w, 1) (query) and atomicAdd(&mDeltaPos[v].x/.y/.z, dP)
//     (solve writeCloth).  A cloth TRIANGLE contact bumps all three of its vertices and
//     adjacent triangles share vertices by construction, so contention here is not an
//     edge case -- it is the normal case.  Previously plain
//     load-add-store.  For the single-contact unit test (and any contact set with no two
//     contacts sharing a cloth vertex within one pass) there is no cross-thread
//     contention, so the read-modify-write is bit-identical to the atomic — same
//     precedent as femClothRigidAttachBda::applyClothDelta / the particle one-way solve;
//   - PxComputeBasisVectors's right.normalize() lowers to `right / sqrt(magSq)` (a true
//     divide, PxVec3::normalize(), PxVec3.h:295) — the CPU reference uses the identical
//     expression, so t0/t1 are bit-exact.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the raXn cross /
// sqrtInvInertia dot responses feed the compared constraint block, and the error / CN /
// deltaLambdaN bias math feeds the compared mDeltaPos — either would fuse to FMAs
// otherwise, breaking bitwise parity with the -ffp-contract=off reference (same reasoning
// as psContactSolveTgsBda).

import pxgSlangAtomicFloat;   // atomicAdd(float*) parity — see that module
import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangPrepDesc;       // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;     // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu, PxgSolverSharedDescTGSGpu
import pxgSlangCommon;         // PxgSolverBodyData, PxgSolverTxIData, PxgAlignedTransform
import pxgSlangFemRigidContact; // prepareFEMContacts + the FRC_* block layout and its helpers
                               // (extracted so sb_rigidContactPrepareLaunch shares it byte
                               //  for byte instead of carrying a second copy)

// PxgSoftBodyKernelBlockDim/GridDim::SB_UPDATEROTATION (PxgSoftBodyCoreKernelIndices.h).
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;

// PX_MAX_NB_DEFORMABLE_SURFACE_TRI (PxDeformableSurface.h:56) == the element mask /
// invalid marker; PXG_BITSHIFT_ELEMENT_ID (PxgSimulationCoreDesc.h:41) — cloth id / elem
// index packing (PxGetClothId / PxGetClothElementIndex, PxgSimulationCoreDesc.h:135).
static const uint PX_MAX_NB_DEFORMABLE_SURFACE_TRI = 0x000fffffu;
static const uint PXG_CLOTH_ID_SHIFT = 20u;

// PxgFemOtherContactInfo (PxgFEMCore.h:128, PX_ALIGN(16), 16 bytes):
//   PxU64 pairInd0 @0 (rigid id), PxU32 pairInd1 @8 (cloth compressed id),
//   PxU32 rigidMatInd_isInCollision @12 (bit31 = isInCollision).
static const uint CI_pairInd0    = 0u;
static const uint CI_pairInd1    = 8u;
static const uint CI_rigidMatInd = 12u;
static const uint CI_STRIDE      = 16u;
static const uint CI_COLLISION_FLAG = 0x80000000u;

// ===========================================================================
// Push-constant blobs (byte-identical to the CUDA arg packs / ABI rows).
// ===========================================================================

// cloth_rigidContactPrepareLaunch (FEMClothConstraintPrep.cu:70).  CUDA arg pack
// (PxgFEMClothCore.cpp:328): { PxgFEMCloth* femClothes, float4* contacts_restW,
// float4* normalPens, float4* barycentrics, const PxgFemOtherContactInfo* contactInfos,
// PxU32* numContacts, PxgFemRigidConstraintBlock* constraints, PxgPrePrepDesc* preDesc,
// PxgConstraintPrepareDesc* prepareDesc, PxReal* rigidLambdaNs, PxReal invDt,
// PxgSolverSharedDescBase* sharedDesc, bool isTGS } -> natural-alignment blob = 104 bytes.
// (ABI row {8,8,8,8,8,8,8,8,8,8,4,8,1}.)
struct PrepContactParams
{
    uint64_t femClothes;     // @0
    uint64_t contacts;       // @8
    uint64_t normalPens;     // @16
    uint64_t barycentrics;   // @24
    uint64_t contactInfos;   // @32
    uint64_t numContacts;    // @40
    uint64_t constraints;    // @48
    uint64_t prePrepDesc;    // @56
    uint64_t prepareDesc;    // @64
    uint64_t rigidLambdaNs;  // @72
    float invDt;             // @80
    uint _pad84;             // @84
    uint64_t sharedDesc;     // @88  (DEFERRED: articulation/bodySim never dereferenced)
    uint isTGS;              // @96  (bool, byte0)
    uint _pad100;            // @100 -> 104
}

// cloth_queryRigidClothContactReferenceCount{,TGS}Launch (FEMClothExternalSolve.cu:414 /
// :558).  CUDA arg pack (PxgFEMClothCore.cpp:1741): { PxgFEMCloth* femClothes,
// PxgFemOtherContactInfo* contactInfos, PxgFemRigidConstraintBlock* constraints,
// PxU32* numContacts, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
// PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc, PxReal dt,
// PxReal* appliedForces, PxU32* rigidBodyReferenceCounts } -> blob = 88 bytes.
// (ABI row {8,8,8,8,8,8,8,8,4,8,8}.)
struct QueryContactParams
{
    uint64_t femClothes;                // @0
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

// cloth_solveRigidClothCollision{,TGS}Launch (FEMClothExternalSolve.cu:535 / :578).
// CUDA arg pack (PxgFEMClothCore.cpp:1797): { PxgFEMCloth* femClothes,
// PxgFemOtherContactInfo* contactInfos, PxgFemRigidConstraintBlock* constraints,
// PxU32* numContacts, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
// PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc,
// float4* rigidDeltaVel, PxReal* appliedForces, PxU32* rigidBodyReferenceCounts,
// PxReal dt, PxsDeformableSurfaceMaterialData* materials,
// const PxsMaterialData* rigidBodyMaterials } -> blob = 112 bytes.
// (ABI row {8,8,8,8,8,8,8,8,8,8,8,4,8,8}.)
struct SolveContactParams
{
    uint64_t femClothes;                // @0
    uint64_t contactInfos;              // @8
    uint64_t constraints;               // @16
    uint64_t numContacts;               // @24
    uint64_t prePrepDesc;               // @32
    uint64_t solverCoreDesc;            // @40
    uint64_t artiCoreDesc;              // @48 (DEFERRED: articulation never dereferenced)
    uint64_t sharedDesc;                // @56
    uint64_t rigidDeltaVel;             // @64
    uint64_t appliedForces;             // @72
    uint64_t rigidBodyReferenceCounts;  // @80
    float dt;                           // @88
    uint _pad92;                        // @92
    uint64_t materials;                 // @96  PxsDeformableSurfaceMaterialData*
    uint64_t rigidBodyMaterials;        // @104 PxsMaterialData* -> 112
}

[[vk::push_constant]] ConstantBuffer<PrepContactParams> gPrep;
[[vk::push_constant]] ConstantBuffer<QueryContactParams> gQuery;
[[vk::push_constant]] ConstantBuffer<SolveContactParams> gSolve;


// ===========================================================================
// cloth_rigidContactPrepareLaunch (FEMClothConstraintPrep.cu:70)
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_rigidContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gPrep.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gPrep.femClothes);
    Ptr<float> rigidLambdaNs = Ptr<float>(gPrep.rigidLambdaNs);

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

        rigidLambdaNs[workIndex] = 0.0f;

        const uint64_t ciBase = gPrep.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint64_t pairInd0 = Ptr<uint64_t>(ciBase + CI_pairInd0)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + CI_pairInd1)[0];

        // First actor: rigid body.  Second actor: cloth.
        const uint64_t rigidId = pairInd0;
        const uint clothId = pairInd1 >> PXG_CLOTH_ID_SHIFT;
        const uint elementId = pairInd1 & PX_MAX_NB_DEFORMABLE_SURFACE_TRI;

        if (elementId == PX_MAX_NB_DEFORMABLE_SURFACE_TRI)
            continue;

        Ptr<float4> accumDelta_invMass = Ptr<float4>(femClothes[clothId].mAccumulatedDeltaPos);

        const float4 contact_restW = Ptr<float4>(gPrep.contacts)[workIndex];
        const float4 normal_pen = Ptr<float4>(gPrep.normalPens)[workIndex];
        const float4 barycentric = Ptr<float4>(gPrep.barycentrics)[workIndex];

        const float3 p = float3(contact_restW.x, contact_restW.y, contact_restW.z);

        float4 deltaP;
        if (barycentric.w == 0.0f)
        {
            const uint4 vertexIndices = Ptr<uint4>(femClothes[clothId].mTriangleVertexIndices)[elementId];
            const float4 a = accumDelta_invMass[vertexIndices.x];
            const float4 b = accumDelta_invMass[vertexIndices.y];
            const float4 cc = accumDelta_invMass[vertexIndices.z];
            deltaP = a * barycentric.x + b * barycentric.y + cc * barycentric.z;
        }
        else
        {
            deltaP = accumDelta_invMass[elementId];
        }

        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);
        const float pen = normal_pen.w - contact_restW.w;
        const float3 delta = float3(deltaP.x, deltaP.y, deltaP.z);

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gPrep.constraints + uint64_t(index) * FRC_STRIDE;

        prepareFEMContacts(cBase, offset, normal, p, pen, delta, rigidId, barycentric,
                           gPrep.prepareDesc, gPrep.prePrepDesc, femClothes[clothId].mPenBiasClamp);
    }
}

// ---------------------------------------------------------------------------
// FEMCollision::readCloth (deformableUtils.cuh:488) — cloth VERTEX / TRIANGLE contact.
// Returns the mass-split inverse masses; also fills deformableLinDelta (out).  P9.46: it
// also fills the deformable friction, which comes from TWO different sources -- the
// per-triangle material table for a triangle contact (:513-514), the per-vertex
// mDynamicFrictions array for a vertex contact (:533).  Both are gated on `materials`.
// ---------------------------------------------------------------------------
float3 readClothVertexInvMasses(Ptr<PxgFEMClothGpu> femClothes, uint clothId, uint elementId, float4 bc,
                                bool countReferenceOnly, out float3 deformableLinDelta,
                                uint64_t materials, out float deformableFriction)
{
    Ptr<float4> clothPosDeltas = Ptr<float4>(femClothes[clothId].mAccumulatedDeltaPos);
    Ptr<float4> clothDeltaPos = Ptr<float4>(femClothes[clothId].mDeltaPos);

    // The CUDA's FEMCollision member default (deformableUtils.cuh:270); the
    // countReferenceOnly path never assigns it.
    deformableFriction = 0.0f;

    float3 invM;
    float4 clothDelta;
    if (bc.w == 0.0f)  // Cloth triangle
    {
        const uint4 triVertId = Ptr<uint4>(femClothes[clothId].mTriangleVertexIndices)[elementId];
        const float4 pd0 = clothPosDeltas[triVertId.x];
        const float4 pd1 = clothPosDeltas[triVertId.y];
        const float4 pd2 = clothPosDeltas[triVertId.z];
        clothDelta = pd0 * bc.x + pd1 * bc.y + pd2 * bc.z;
        invM = float3(pd0.w, pd1.w, pd2.w);
        if (!countReferenceOnly)
        {
            // P9.46: a TRIANGLE contact takes its friction from the material table through
            // the per-triangle index (:513-514); a VERTEX contact takes it from the
            // per-vertex mDynamicFrictions array instead (:533).  Two different sources for
            // the same quantity, and only the .cu says which is which.
            deformableFriction = femMaterialFrictionAt(
                materials, femClothes[clothId].mMaterialIndices, elementId);

            // Mass-splitting by the per-vertex reference counts (mDeltaPos.w).
            const float rc0 = clothDeltaPos[triVertId.x].w;
            const float rc1 = clothDeltaPos[triVertId.y].w;
            const float rc2 = clothDeltaPos[triVertId.z].w;
            invM = float3(invM.x * rc0, invM.y * rc1, invM.z * rc2);
        }
    }
    else  // Cloth vertex
    {
        clothDelta = clothPosDeltas[elementId];
        invM = float3(clothDelta.w, 0.0f, 0.0f);
        if (!countReferenceOnly)
        {
            if (materials != 0UL)
                deformableFriction = Ptr<float>(femClothes[clothId].mDynamicFrictions)[elementId];

            const float rc = clothDeltaPos[elementId].w;
            invM = float3(invM.x * rc, 0.0f, 0.0f);
        }
    }
    deformableLinDelta = float3(clothDelta.x, clothDelta.y, clothDelta.z);
    return invM;
}

// FEMCollision::getGlobalRigidBodyId (deformableUtils.cuh:442): -1 for a static body.
// ===========================================================================
// cloth_queryRigidClothContactReferenceCount{,TGS}Launch (FEMClothExternalSolve.cu:414 /
// :558).  A checkOnlyActivity pass: mark each penetrating contact in-collision and bump
// the referenced cloth vertex's mDeltaPos.w reference count.
// ===========================================================================
void queryRigidClothContactReferenceCount(bool isTGS, uint idx, uint blockX)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gQuery.solverCoreDesc);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gQuery.numContacts)[0];
    const float dt = gQuery.dt;

    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gQuery.femClothes);
    Ptr<float> appliedForces = Ptr<float>(gQuery.appliedForces);

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

        const uint64_t ciBase = gQuery.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint rigidMatInd = Ptr<uint>(ciBase + CI_rigidMatInd)[0];
        const bool wasActive = (rigidMatInd & CI_COLLISION_FLAG) != 0u;
        const uint64_t rigidId = Ptr<uint64_t>(ciBase + CI_pairInd0)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + CI_pairInd1)[0];

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gQuery.constraints + uint64_t(index) * FRC_STRIDE;

        const float4 fricTan0_invMass0 = Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset];
        const float4 bc = Ptr<float4>(cBase + FRC_barycentric)[offset];

        const uint clothId = pairInd1 >> PXG_CLOTH_ID_SHIFT;
        const uint elementId = pairInd1 & PX_MAX_NB_DEFORMABLE_SURFACE_TRI;

        if (elementId < PX_MAX_NB_DEFORMABLE_SURFACE_TRI)
        {
            // :349, before the wasActive branch — a static body yields -1.
            const int globalRigidBodyId = getGlobalRigidBodyId(rigidId, gQuery.prePrepDesc, numSolverBodies);
            const float3 bcVec = (bc.w == 0.0f) ? float3(bc.x, bc.y, bc.z) : float3(1.0f, 0.0f, 0.0f);

            float3 deformableLinDelta;
            float queryFriction;   // unused: the query never reads a material table
            const float3 deformableInvMasses = readClothVertexInvMasses(femClothes, clothId, elementId, bc,
                                                                        /*countReferenceOnly*/ true, deformableLinDelta,
                                                                        /*materials*/ 0UL, queryFriction);

            bool isActive = false;
            if (wasActive)
            {
                isActive = true;
            }
            else
            {
                // readRigidBody(NULL,NULL): rigidBodyReferenceCount == 1, friction 0.
                // the 3-wide mass-splitting term: bc.multiply(bc).dot(invM).  Computed at
                // the call site because a tetrahedron needs the 4-wide form (see the module).
                const float3 bcSq = float3(bcVec.x * bcVec.x, bcVec.y * bcVec.y, bcVec.z * bcVec.z);
                const float massSplit = dotVec(bcSq, deformableInvMasses);
                const ContactResult cr = contactInitialize(cBase, offset, rigidId, isTGS,
                                                           appliedForces[workIndex], dt, bcVec,
                                                           deformableLinDelta, massSplit,
                                                           /*rigidRefCount*/ 1u, wasActive, /*checkOnly*/ true,
                                                           gQuery.solverCoreDesc, gQuery.sharedDesc, gQuery.prePrepDesc,
                                                           // The QUERY pass passes NULL for BOTH material tables
                                                           // (FEMClothExternalSolve.cu:362 readRigidBody(...,NULL,NULL)
                                                           // and readCloth(...,NULL,checkOnly)), so the combined
                                                           // friction is combineFriction(0, 0, eMAX) == 0.  It could
                                                           // not matter anyway: checkOnlyActivity returns before the
                                                           // tangent block.
                                                           /*combinedFriction*/ 0.0f);
                isActive = cr.isActive;
                if (isActive)
                {
                    // markInCollision(true): set bit31.
                    Ptr<uint>(ciBase + CI_rigidMatInd)[0] = rigidMatInd | CI_COLLISION_FLAG;
                }
            }

            if (isActive)
            {
                Ptr<float4> mDeltaPos = Ptr<float4>(femClothes[clothId].mDeltaPos);
                if (bc.w == 0.0f)  // Cloth triangle: bump the three vertices' reference counts.
                {
                    const uint4 triVertId = Ptr<uint4>(femClothes[clothId].mTriangleVertexIndices)[elementId];
                    if (bc.x > 1e-3f && deformableInvMasses.x != 0.0f)
                    {
                        pxgAtomicAddF4W(femClothes[clothId].mDeltaPos, triVertId.x, 1.0f);
                    }
                    if (bc.y > 1e-3f && deformableInvMasses.y != 0.0f)
                    {
                        pxgAtomicAddF4W(femClothes[clothId].mDeltaPos, triVertId.y, 1.0f);
                    }
                    if (bc.z > 1e-3f && deformableInvMasses.z != 0.0f)
                    {
                        pxgAtomicAddF4W(femClothes[clothId].mDeltaPos, triVertId.z, 1.0f);
                    }
                }
                else  // Cloth vertex: bump the single vertex's reference count.
                {
                    if (deformableInvMasses.x != 0.0f)
                    {
                        pxgAtomicAddF4W(femClothes[clothId].mDeltaPos, elementId, 1.0f);
                    }
                }

                // Increment the reference count of the rigid body (:404-408).  P9.45.
                // Guarded exactly as the .cu: a static body has globalRigidBodyId == -1 and a
                // kinematic one has invMass 0, and both keep the solve's count at its
                // default of 1.  The softbody query already carried this line; the cloth one
                // did not, and the two modules now match.
                if (globalRigidBodyId != -1 && fricTan0_invMass0.w != 0.0f)
                {
                    uint prev;
                    InterlockedAdd((Ptr<uint>(gQuery.rigidBodyReferenceCounts))[globalRigidBodyId], 1u, prev);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_queryRigidClothContactReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    queryRigidClothContactReferenceCount(false, groupThreadId.x, groupId.x);
}

[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_queryRigidClothContactReferenceCountTGSLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    queryRigidClothContactReferenceCount(true, groupThreadId.x, groupId.x);
}

// ===========================================================================
// cloth_solveRigidClothCollision{,TGS}Launch (FEMClothExternalSolve.cu:535 / :578).
// Solve the contact constraint; scatter the cloth-side correction into mDeltaPos and
// write the (zero, static) rigid reaction into rigidDeltaVel.
// ===========================================================================
void solveRigidClothContact(bool isTGS, uint idx, uint blockX)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gSolve.solverCoreDesc);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gSolve.numContacts)[0];
    const float dt = gSolve.dt;

    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gSolve.femClothes);
    Ptr<float> appliedForces = Ptr<float>(gSolve.appliedForces);
    Ptr<float4> rigidDeltaVel = Ptr<float4>(gSolve.rigidDeltaVel);

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

        const uint64_t ciBase = gSolve.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint rigidMatInd = Ptr<uint>(ciBase + CI_rigidMatInd)[0];
        const bool isActive = (rigidMatInd & CI_COLLISION_FLAG) != 0u;
        const uint64_t rigidId = Ptr<uint64_t>(ciBase + CI_pairInd0)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + CI_pairInd1)[0];

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gSolve.constraints + uint64_t(index) * FRC_STRIDE;

        const float4 fricTan0_invMass0 = Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset];
        const float4 bc = Ptr<float4>(cBase + FRC_barycentric)[offset];

        const uint clothId = pairInd1 >> PXG_CLOTH_ID_SHIFT;
        const uint elementId = pairInd1 & PX_MAX_NB_DEFORMABLE_SURFACE_TRI;

        if (elementId < PX_MAX_NB_DEFORMABLE_SURFACE_TRI)
        {
            const int globalRigidBodyId = getGlobalRigidBodyId(rigidId, gSolve.prePrepDesc, numSolverBodies);
            float3 deltaLinVel0 = float3(0.0f, 0.0f, 0.0f);
            float3 deltaAngVel0 = float3(0.0f, 0.0f, 0.0f);
            float count = 0.0f;

            if (isActive)
            {
                const float3 bcVec = (bc.w == 0.0f) ? float3(bc.x, bc.y, bc.z) : float3(1.0f, 0.0f, 0.0f);

                // readRigidBody: rigidBodyReferenceCount == 1 for a static body
                // (globalRigidBodyId == -1); the friction half follows below.
                uint rigidBodyReferenceCount = 1u;
                if (globalRigidBodyId != -1 && fricTan0_invMass0.w != 0.0f)
                    rigidBodyReferenceCount = Ptr<uint>(gSolve.rigidBodyReferenceCounts)[globalRigidBodyId];

                float3 deformableLinDelta;
                float clothFriction;
                const float3 deformableInvMasses = readClothVertexInvMasses(femClothes, clothId, elementId, bc,
                                                                            /*countReferenceOnly*/ false, deformableLinDelta,
                                                                            gSolve.materials, clothFriction);

                // P9.46: getCombinedFriction() = PxsCombinePxReal(rigidFriction, clothFriction,
                // the RIGID material's fricCombineMode).  The rigid material index is the low
                // 31 bits of the contact info word the query ORs its in-collision flag into
                // (PxgFEMCore.h:160), so it must be masked, not read raw.
                const RigidFriction rf = readRigidFriction(gSolve.rigidBodyMaterials,
                                                           rigidMatInd & ~CI_COLLISION_FLAG);
                const float combinedFriction = combineFriction(rf.friction, clothFriction, rf.combineMode);

                const float3 bcSq = float3(bcVec.x * bcVec.x, bcVec.y * bcVec.y, bcVec.z * bcVec.z);
                const float massSplit = dotVec(bcSq, deformableInvMasses);
                const ContactResult cr = contactInitialize(cBase, offset, rigidId, isTGS,
                                                           appliedForces[workIndex], dt, bcVec,
                                                           deformableLinDelta, massSplit,
                                                           rigidBodyReferenceCount, isActive, /*checkOnly*/ false,
                                                           gSolve.solverCoreDesc, gSolve.sharedDesc, gSolve.prePrepDesc,
                                                           combinedFriction);

                // computeFEMChange: deltaPos = -(deltaLambdaN*normal + deltaLambdaT*tangent)*dt.
                const float3 deltaPos = float3(
                    -(cr.deltaLambdaN * cr.normal.x + cr.deltaLambdaT * cr.tangent.x) * dt,
                    -(cr.deltaLambdaN * cr.normal.y + cr.deltaLambdaT * cr.tangent.y) * dt,
                    -(cr.deltaLambdaN * cr.normal.z + cr.deltaLambdaT * cr.tangent.z) * dt);
                appliedForces[workIndex] = cr.accumulatedDeltaLambdaN;

                // writeCloth: scatter the cloth-side correction (plain RMW, see header).
                Ptr<float4> mDeltaPos = Ptr<float4>(femClothes[clothId].mDeltaPos);
                if (bc.w == 0.0f)  // Cloth triangle
                {
                    const uint4 triVertInds = Ptr<uint4>(femClothes[clothId].mTriangleVertexIndices)[elementId];
                    const float bcArr[3] = { bc.x, bc.y, bc.z };
                    const uint triArr[3] = { triVertInds.x, triVertInds.y, triVertInds.z };
                    const float invMArr[3] = { deformableInvMasses.x, deformableInvMasses.y, deformableInvMasses.z };
                    for (uint it = 0; it < 3u; ++it)
                    {
                        if (invMArr[it] > 0.0f)
                        {
                            const float s = invMArr[it] * bcArr[it];
                            pxgAtomicAddF4XYZ(femClothes[clothId].mDeltaPos, triArr[it], deltaPos * s);
                        }
                    }
                }
                else  // Cloth vertex
                {
                    if (deformableInvMasses.x > 0.0f)
                    {
                        const float s = deformableInvMasses.x;
                        pxgAtomicAddF4XYZ(femClothes[clothId].mDeltaPos, elementId, deltaPos * s);
                    }
                }

                // computeRigidChange (FEMClothExternalSolve.cu:509-514, deformableUtils.cuh:418).
                // P9.45.  A STATIC body still leaves count and the deltas at 0 -- that is the
                // .cu's own guard -- and so does a kinematic one (fricTan0_invMass0.w == 0).
                // The ARTICULATION case stays out: prepareFEMContacts defers the articulation
                // anchor build (pxgSlangFemRigidContact.slang:143) and readRigidVelocity defers
                // the link velocity read, so a reaction for a link would be wrong rather than
                // merely absent.  Writing zeros for a link is what this module already did.
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

            // writeRigidBody: for a static contact the CUDA writes zeros with count 0.
            rigidDeltaVel[workIndex] = float4(deltaLinVel0.x, deltaLinVel0.y, deltaLinVel0.z, count);
            rigidDeltaVel[workIndex + tNumContacts] = float4(deltaAngVel0.x, deltaAngVel0.y, deltaAngVel0.z, 0.0f);
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_solveRigidClothCollisionLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    solveRigidClothContact(false, groupThreadId.x, groupId.x);
}

[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_solveRigidClothCollisionTGSLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    solveRigidClothContact(true, groupThreadId.x, groupId.x);
}
