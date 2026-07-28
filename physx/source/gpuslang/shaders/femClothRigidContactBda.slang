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
//   - the DYNAMIC-rigid REACTION is DEFERRED.  For a static body FEMCollision::
//     computeRigidChange is guarded behind !rigidId.isStaticBody() (never taken) and the
//     rigid reference-count increment behind globalRigidBodyId != -1 (never taken, static
//     -> -1), so nothing is lost; the downstream accumulateRigidDeltas fold is a follow-on
//     tranche.  The solve's writeRigidBody still writes rigidDeltaVel[workIndex] and
//     [workIndex + numContacts] = (0,0,0,0) with count 0 — exactly what the CUDA writes
//     for a static contact (deltaLinVel0 == deltaAngVel0 == 0);
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

// PxgSoftBodyKernelBlockDim/GridDim::SB_UPDATEROTATION (PxgSoftBodyCoreKernelIndices.h).
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;

// PX_MAX_NB_DEFORMABLE_SURFACE_TRI (PxDeformableSurface.h:56) == the element mask /
// invalid marker; PXG_BITSHIFT_ELEMENT_ID (PxgSimulationCoreDesc.h:41) — cloth id / elem
// index packing (PxGetClothId / PxGetClothElementIndex, PxgSimulationCoreDesc.h:135).
static const uint PX_MAX_NB_DEFORMABLE_SURFACE_TRI = 0x000fffffu;
static const uint PXG_CLOTH_ID_SHIFT = 20u;

// FEMCollision::initialize friction threshold (deformableUtils.cuh:292).
static const float FEMC_THRESHOLD = 1.0e-14f;

// ---------------------------------------------------------------------------
// Byte layout of the block-of-32 PxgFemRigidConstraintBlock (PxgFEMCore.h:167).
// 32 constraints per block (SoA warp-packed), indexed [offset]; stride 3200.
// ---------------------------------------------------------------------------
static const uint FRC_raXn_resp        = 0u;      // float4[32]
static const uint FRC_raXnF0_resp      = 512u;    // float4[32]
static const uint FRC_raXnF1_resp      = 1024u;   // float4[32]
static const uint FRC_normal_errorW    = 1536u;   // float4[32]
static const uint FRC_fricTan0_invMass = 2048u;   // float4[32]
static const uint FRC_barycentric      = 2560u;   // float4[32]
static const uint FRC_maxPenBias       = 3072u;   // float[32]
static const uint FRC_STRIDE           = 3200u;

// PxgFemOtherContactInfo (PxgFEMCore.h:128, PX_ALIGN(16), 16 bytes):
//   PxU64 pairInd0 @0 (rigid id), PxU32 pairInd1 @8 (cloth compressed id),
//   PxU32 rigidMatInd_isInCollision @12 (bit31 = isInCollision).
static const uint CI_pairInd0    = 0u;
static const uint CI_pairInd1    = 8u;
static const uint CI_rigidMatInd = 12u;
static const uint CI_STRIDE      = 16u;
static const uint CI_COLLISION_FLAG = 0x80000000u;

// ---------------------------------------------------------------------------
// Small vector helpers (byte-for-byte the CUDA foundation math).
// ---------------------------------------------------------------------------
float3 crossVec(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float dotVec(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxMat33 (column-major sqrtInvInertia[9]) * PxVec3.
float3 mat33MulVec(float m[9], float3 v)
{
    return float3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
                  m[1] * v.x + m[4] * v.y + m[7] * v.z,
                  m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// PxComputeBasisVectors (PxMathUtils.h:297): two orthonormal tangents to `dir`.
// right.normalize() is a true divide by the magnitude (PxVec3::normalize(), PxVec3.h:295).
void computeBasisVectors(float3 dir, out float3 right, out float3 up)
{
    if (abs(dir.y) <= 0.9999f)
    {
        right = float3(dir.z, 0.0f, -dir.x);
        const float m = sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (m > 0.0f)
            right = float3(right.x / m, right.y / m, right.z / m);
        up = float3(dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x);
    }
    else
    {
        right = float3(1.0f, 0.0f, 0.0f);
        up = float3(0.0f, dir.z, -dir.y);
        const float m = sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
        if (m > 0.0f)
            up = float3(up.x / m, up.y / m, up.z / m);
    }
}

// PxNodeIndex (PxNodeIndex.h:49) packed in the low/high 32 bits of a PxU64.
bool nodeIsStaticBody(uint64_t ind) { return uint(ind & 0xffffffffUL) == 0xffffffffu; }
uint nodeIndex(uint64_t ind) { return uint(ind & 0xffffffffUL); }
bool nodeIsArticulation(uint64_t ind) { return (uint(ind >> 32) & 1u) != 0u; }

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
    uint64_t prepDesc;       // @64
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
    uint64_t materials;                 // @96  (DEFERRED: null-material path)
    uint64_t rigidBodyMaterials;        // @104 (DEFERRED) -> 112
}

[[vk::push_constant]] ConstantBuffer<PrepContactParams> gPrep;
[[vk::push_constant]] ConstantBuffer<QueryContactParams> gQuery;
[[vk::push_constant]] ConstantBuffer<SolveContactParams> gSolve;

// ===========================================================================
// prepareFEMContacts (deformableUtils.cuh:132) — the rigid / static (non-articulation)
// path.  Writes the block-of-32 PxgFemRigidConstraintBlock slot [offset] at cBase.
// ===========================================================================
void prepareFEMContacts(uint64_t cBase, uint offset, float3 normal,
                        float3 p, float pen, float3 delta, uint64_t rigidId, float4 barycentric,
                        uint64_t prepDescPtr, uint64_t prePrepDescPtr, float penBiasClampFEM)
{
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(prepDescPtr);
    const uint64_t bodyFramesBase = prepDesc[0].body2WorldPool;
    const uint64_t solverBodyDataBase = prepDesc[0].solverBodyDataPool;
    const uint64_t solverTxIDataBase = prepDesc[0].solverBodyTxIDataPool;

    // Select two tangent vectors to the normal (PxComputeBasisVectors).
    float3 t0, t1;
    computeBasisVectors(normal, t0, t1);

    // ARTICULATION anchor branch DEFERRED (see header): a static floor / sphere is never
    // an articulation, so the else (rigid / static) path is taken.  solverBodyIdx == 0 for
    // a static body (solverBodyIndices untouched).
    const bool isStatic = nodeIsStaticBody(rigidId);
    uint idx = 0u;
    if (!isStatic)
        idx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDescPtr)[0].solverBodyIndices)[nodeIndex(rigidId)];

    PxgSolverTxIData txI = Ptr<PxgSolverTxIData>(solverTxIDataBase)[idx];
    float invSqrtInertia[9];
    for (uint k = 0; k < 9u; ++k)
        invSqrtInertia[k] = txI.sqrtInvInertia[k];

    const PxgSolverBodyData sbd = Ptr<PxgSolverBodyData>(solverBodyDataBase)[idx];
    const float4 linVel_invMass0 = sbd.initialLinVelXYZ_invMassW;
    const float penBiasClampRigid = sbd.initialAngVelXYZ_penBiasClamp.w;
    const float invMass0 = linVel_invMass0.w;

    // both static and kinematic objects have invMass == 0; a static body is never kinematic.
    const bool isKinematic = (invMass0 == 0.0f) && (!isStatic);

    const float4 bfP = Ptr<float4>(bodyFramesBase + uint64_t(idx) * 32u + 16u)[0];
    const float3 bodyFrame0p = float3(bfP.x, bfP.y, bfP.z);

    const float3 ra = float3(p.x - bodyFrame0p.x, p.y - bodyFrame0p.y, p.z - bodyFrame0p.z);
    const float3 raXn = crossVec(ra, normal);
    const float3 raXF0 = crossVec(ra, t0);
    const float3 raXF1 = crossVec(ra, t1);

    const float3 raXnSqrtInertia = mat33MulVec(invSqrtInertia, raXn);
    const float resp0 = dotVec(raXnSqrtInertia, raXnSqrtInertia) + invMass0;

    const float3 raXF0SqrtInertia = mat33MulVec(invSqrtInertia, raXF0);
    const float3 raXF1SqrtInertia = mat33MulVec(invSqrtInertia, raXF1);

    const float respF0 = dotVec(raXF0SqrtInertia, raXF0SqrtInertia) + invMass0;
    const float respF1 = dotVec(raXF1SqrtInertia, raXF1SqrtInertia) + invMass0;

    float4 raXn_resp;
    float4 raXnF0_resp;
    float4 raXnF1_resp;
    if (isKinematic)
    {
        raXn_resp = float4(raXn.x, raXn.y, raXn.z, resp0);
        raXnF0_resp = float4(raXF0.x, raXF0.y, raXF0.z, respF0);
        raXnF1_resp = float4(raXF1.x, raXF1.y, raXF1.z, respF1);
    }
    else
    {
        raXn_resp = float4(raXnSqrtInertia.x, raXnSqrtInertia.y, raXnSqrtInertia.z, resp0);
        raXnF0_resp = float4(raXF0SqrtInertia.x, raXF0SqrtInertia.y, raXF0SqrtInertia.z, respF0);
        raXnF1_resp = float4(raXF1SqrtInertia.x, raXF1SqrtInertia.y, raXF1SqrtInertia.z, respF1);
    }

    const float maxPenBias = max(penBiasClampRigid, penBiasClampFEM);
    const float error = pen + dotVec(delta, normal);

    Ptr<float4>(cBase + FRC_normal_errorW)[offset] = float4(normal.x, normal.y, normal.z, error);
    Ptr<float4>(cBase + FRC_raXn_resp)[offset] = raXn_resp;
    Ptr<float4>(cBase + FRC_raXnF0_resp)[offset] = raXnF0_resp;
    Ptr<float4>(cBase + FRC_raXnF1_resp)[offset] = raXnF1_resp;
    Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset] = float4(t0.x, t0.y, t0.z, invMass0);
    Ptr<float>(cBase + FRC_maxPenBias)[offset] = maxPenBias;
    Ptr<float4>(cBase + FRC_barycentric)[offset] = barycentric;
}

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
        const uint workIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

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
                           gPrep.prepDesc, gPrep.prePrepDesc, femClothes[clothId].mPenBiasClamp);
    }
}

// ---------------------------------------------------------------------------
// PxgVelocityReader (dataReadWriteHelper.cuh:70) — non-articulation rigid path.
// ARTICULATION branch DEFERRED (see header).  Returns lin/ang velocity (+ lin/ang delta
// for TGS) of the rigid body; a static body reads solverBodyIdx 0 (zero for a floor).
// ---------------------------------------------------------------------------
struct RigidState
{
    float3 linVel;
    float3 angVel;
    float3 linDelta;
    float3 angDelta;
}

RigidState readRigidVelocity(uint64_t rigidId, bool isTGS, uint64_t solverCoreDescPtr,
                             uint64_t sharedDescPtr, uint64_t prePrepDescPtr)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(solverCoreDescPtr);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;

    const bool isStatic = nodeIsStaticBody(rigidId);
    uint solverBodyIdx = 0u;
    if (!isStatic)
        solverBodyIdx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDescPtr)[0].solverBodyIndices)[nodeIndex(rigidId)];

    RigidState s;
    s.linVel = float3(0.0f, 0.0f, 0.0f);
    s.angVel = float3(0.0f, 0.0f, 0.0f);
    s.linDelta = float3(0.0f, 0.0f, 0.0f);
    s.angDelta = float3(0.0f, 0.0f, 0.0f);

    if (isTGS)
    {
        Ptr<float4> pool = Ptr<float4>(Ptr<PxgSolverSharedDescTGSGpu>(sharedDescPtr)[0].iter_solverBodyVelPool);
        const float4 v0 = pool[accOffset + solverBodyIdx];
        const float4 v1 = pool[accOffset + solverBodyIdx + numSolverBodies];
        const float4 v2 = pool[accOffset + solverBodyIdx + numSolverBodies + numSolverBodies];
        s.linVel = float3(v0.x, v0.y, v0.z);
        s.angVel = float3(v0.w, v1.x, v1.y);
        s.linDelta = float3(v1.z, v1.w, v2.x);
        s.angDelta = float3(v2.y, v2.z, v2.w);
    }
    else
    {
        Ptr<float4> initialVel = Ptr<float4>(solverCoreDesc[0].outSolverVelocity);
        Ptr<float4> pool = Ptr<float4>(Ptr<PxgSolverSharedDescPGSGpu>(sharedDescPtr)[0].iter_solverBodyVelPool);
        const float4 iv0 = initialVel[solverBodyIdx];
        const float4 iv1 = initialVel[solverBodyIdx + numSolverBodies];
        const float4 dv0 = pool[accOffset + solverBodyIdx];
        const float4 dv1 = pool[accOffset + solverBodyIdx + numSolverBodies];
        s.linVel = float3(iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z);
        s.angVel = float3(iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z);
    }
    return s;
}

// ---------------------------------------------------------------------------
// FEMCollision::initialize (deformableUtils.cuh:280) — cloth vs rigid, static /
// null-material path.  Returns isActive; on the full path also fills deltaLambdaN,
// deltaLambdaT, tangent (deltaLambdaT == 0 with the deferred null-material friction).
// ---------------------------------------------------------------------------
struct ContactResult
{
    bool isActive;
    float deltaLambdaN;
    float deltaLambdaT;
    float3 normal;
    float3 tangent;
    float accumulatedDeltaLambdaN;
}

ContactResult contactInitialize(uint64_t cBase, uint offset, uint64_t rigidId, bool isTGS,
                                float appliedForceRef, float dt, float3 bcVec,
                                float3 deformableLinDelta, float3 deformableVertexInvMasses,
                                uint rigidBodyReferenceCount, bool wasActive, bool checkOnlyActivity,
                                uint64_t solverCoreDescPtr, uint64_t sharedDescPtr, uint64_t prePrepDescPtr)
{
    ContactResult r;
    r.isActive = false;
    r.deltaLambdaN = 0.0f;
    r.deltaLambdaT = 0.0f;
    r.tangent = float3(0.0f, 0.0f, 0.0f);
    r.accumulatedDeltaLambdaN = appliedForceRef;

    const float invDt = 1.0f / dt;

    const float4 raXn_resp = Ptr<float4>(cBase + FRC_raXn_resp)[offset];
    const float4 normal_biasW = Ptr<float4>(cBase + FRC_normal_errorW)[offset];
    const float4 raXnF0_resp = Ptr<float4>(cBase + FRC_raXnF0_resp)[offset];
    const float4 raXnF1_resp = Ptr<float4>(cBase + FRC_raXnF1_resp)[offset];
    const float maxPenBias = Ptr<float>(cBase + FRC_maxPenBias)[offset];
    const float4 fricTan0_invMass0 = Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset];

    const float3 normal = float3(normal_biasW.x, normal_biasW.y, normal_biasW.z);
    r.normal = normal;
    const float3 fric0 = float3(fricTan0_invMass0.x, fricTan0_invMass0.y, fricTan0_invMass0.z);
    const float3 fric1 = crossVec(normal, fric0);
    const float initPen = normal_biasW.w;
    const float3 raXn = float3(raXn_resp.x, raXn_resp.y, raXn_resp.z);

    const RigidState st = readRigidVelocity(rigidId, isTGS, solverCoreDescPtr, sharedDescPtr, prePrepDescPtr);
    const float3 linVel = st.linVel;
    const float3 angVel = st.angVel;

    float CN;
    float3 relLinDelta;
    float3 angDelta;
    if (isTGS)
    {
        const float normalVel = dotVec(linVel, normal) + dotVec(angVel, raXn);
        relLinDelta = float3(st.linDelta.x - deformableLinDelta.x, st.linDelta.y - deformableLinDelta.y,
                             st.linDelta.z - deformableLinDelta.z);
        angDelta = st.angDelta;
        const float error = (initPen + dotVec(relLinDelta, normal) + dotVec(angDelta, raXn)) * invDt;
        const float errorBiased = max(maxPenBias, error);
        CN = errorBiased + normalVel;
    }
    else
    {
        const float normalVel = dotVec(linVel, normal) + dotVec(angVel, raXn);
        relLinDelta = float3(-deformableLinDelta.x, -deformableLinDelta.y, -deformableLinDelta.z);
        angDelta = float3(0.0f, 0.0f, 0.0f);
        const float error = (initPen + dotVec(relLinDelta, normal)) * invDt;
        const float errorBiased = max(maxPenBias, error);
        CN = errorBiased + normalVel;
    }

    r.isActive = wasActive || (CN < 0.0f);

    if (checkOnlyActivity)
        return r;

    if (!r.isActive)
        return r;

    // Deformable body term in the denominator (mass-splitting).  bc.multiply(bc).dot(invM).
    const float3 bcSq = float3(bcVec.x * bcVec.x, bcVec.y * bcVec.y, bcVec.z * bcVec.z);
    const float deformableInvMass_massSplitting = dotVec(bcSq, deformableVertexInvMasses);

    const float rigidRefCount = float(rigidBodyReferenceCount);
    const float unitResponse = rigidRefCount * raXn_resp.w + deformableInvMass_massSplitting;
    const float invDenom = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

    r.deltaLambdaN = max(-CN * invDenom, -r.accumulatedDeltaLambdaN);
    r.accumulatedDeltaLambdaN += r.deltaLambdaN;

    // Friction constraint in the tangent direction (null-material path -> combinedFriction 0).
    const float3 raXnF0 = float3(raXnF0_resp.x, raXnF0_resp.y, raXnF0_resp.z);
    const float3 raXnF1 = float3(raXnF1_resp.x, raXnF1_resp.y, raXnF1_resp.z);

    const float tanVel0 = dotVec(linVel, fric0) + dotVec(angVel, raXnF0);
    const float tanVel1 = dotVec(linVel, fric1) + dotVec(angVel, raXnF1);

    const float CT0 = (dotVec(fric0, relLinDelta) + dotVec(angDelta, raXnF0)) * invDt + tanVel0;
    const float CT1 = (dotVec(fric1, relLinDelta) + dotVec(angDelta, raXnF1)) * invDt + tanVel1;
    const float3 relTanDelta = float3(CT0 * fric0.x + CT1 * fric1.x, CT0 * fric0.y + CT1 * fric1.y,
                                      CT0 * fric0.z + CT1 * fric1.z);
    const float tanMagSq = dotVec(relTanDelta, relTanDelta);

    if (tanMagSq > FEMC_THRESHOLD)
    {
        const float CT = sqrt(tanMagSq);
        const float invTanMag = 1.0f / CT;
        r.tangent = float3(relTanDelta.x * invTanMag, relTanDelta.y * invTanMag, relTanDelta.z * invTanMag);

        const float unitResponseT0 = rigidRefCount * raXnF0_resp.w + deformableInvMass_massSplitting;
        const float invTanDenom0 = (unitResponseT0 > 0.0f) ? (1.0f / unitResponseT0) : 0.0f;
        const float unitResponseT1 = rigidRefCount * raXnF1_resp.w + deformableInvMass_massSplitting;
        const float invTanDenom1 = (unitResponseT1 > 0.0f) ? (1.0f / unitResponseT1) : 0.0f;

        const float deltaLambdaT0 = CT0 * invTanDenom0;
        const float deltaLambdaT1 = CT1 * invTanDenom1;

        const float dLT = sqrt(deltaLambdaT0 * deltaLambdaT0 + deltaLambdaT1 * deltaLambdaT1);
        // getCombinedFriction() == 0 (null-material path, DEFERRED): deltaLambdaT == -0.
        const float combinedFriction = 0.0f;
        r.deltaLambdaT = -min(dLT, combinedFriction * abs(r.deltaLambdaN));
    }

    return r;
}

// ---------------------------------------------------------------------------
// FEMCollision::readCloth (deformableUtils.cuh:488) — cloth VERTEX / TRIANGLE contact.
// Returns the mass-split inverse masses; also fills deformableLinDelta (out).  Friction
// (mDynamicFrictions / materials) is the DEFERRED null-material path.
// ---------------------------------------------------------------------------
float3 readClothVertexInvMasses(Ptr<PxgFEMClothGpu> femClothes, uint clothId, uint elementId, float4 bc,
                                bool countReferenceOnly, out float3 deformableLinDelta)
{
    Ptr<float4> clothPosDeltas = Ptr<float4>(femClothes[clothId].mAccumulatedDeltaPos);
    Ptr<float4> clothDeltaPos = Ptr<float4>(femClothes[clothId].mDeltaPos);

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
            const float rc = clothDeltaPos[elementId].w;
            invM = float3(invM.x * rc, 0.0f, 0.0f);
        }
    }
    deformableLinDelta = float3(clothDelta.x, clothDelta.y, clothDelta.z);
    return invM;
}

// FEMCollision::getGlobalRigidBodyId (deformableUtils.cuh:442): -1 for a static body.
int getGlobalRigidBodyId(uint64_t rigidId, uint64_t prePrepDescPtr, uint numSolverBodies)
{
    if (nodeIsStaticBody(rigidId))
        return -1;
    const uint solverBodyIdx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDescPtr)[0].solverBodyIndices)[nodeIndex(rigidId)];
    return nodeIsArticulation(rigidId) ? int(numSolverBodies + solverBodyIdx) : int(solverBodyIdx);
}

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
        const uint workIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * blockX * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t ciBase = gQuery.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint rigidMatInd = Ptr<uint>(ciBase + CI_rigidMatInd)[0];
        const bool wasActive = (rigidMatInd & CI_COLLISION_FLAG) != 0u;
        const uint64_t rigidId = Ptr<uint64_t>(ciBase + CI_pairInd0)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + CI_pairInd1)[0];

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gQuery.constraints + uint64_t(index) * FRC_STRIDE;

        const float4 bc = Ptr<float4>(cBase + FRC_barycentric)[offset];

        const uint clothId = pairInd1 >> PXG_CLOTH_ID_SHIFT;
        const uint elementId = pairInd1 & PX_MAX_NB_DEFORMABLE_SURFACE_TRI;

        if (elementId < PX_MAX_NB_DEFORMABLE_SURFACE_TRI)
        {
            const float3 bcVec = (bc.w == 0.0f) ? float3(bc.x, bc.y, bc.z) : float3(1.0f, 0.0f, 0.0f);

            float3 deformableLinDelta;
            const float3 deformableInvMasses = readClothVertexInvMasses(femClothes, clothId, elementId, bc,
                                                                        /*countReferenceOnly*/ true, deformableLinDelta);

            bool isActive = false;
            if (wasActive)
            {
                isActive = true;
            }
            else
            {
                // readRigidBody(NULL,NULL): rigidBodyReferenceCount == 1, friction 0.
                const ContactResult cr = contactInitialize(cBase, offset, rigidId, isTGS,
                                                           appliedForces[workIndex], dt, bcVec,
                                                           deformableLinDelta, deformableInvMasses,
                                                           /*rigidRefCount*/ 1u, wasActive, /*checkOnly*/ true,
                                                           gQuery.solverCoreDesc, gQuery.sharedDesc, gQuery.prePrepDesc);
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

                // Rigid reference-count increment DEFERRED: globalRigidBodyId == -1 for a
                // static body, so the CUDA skips it.
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
        const uint workIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * blockX * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

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
                // (globalRigidBodyId == -1); friction is the DEFERRED null-material path.
                uint rigidBodyReferenceCount = 1u;
                if (globalRigidBodyId != -1 && fricTan0_invMass0.w != 0.0f)
                    rigidBodyReferenceCount = Ptr<uint>(gSolve.rigidBodyReferenceCounts)[globalRigidBodyId];

                float3 deformableLinDelta;
                const float3 deformableInvMasses = readClothVertexInvMasses(femClothes, clothId, elementId, bc,
                                                                            /*countReferenceOnly*/ false, deformableLinDelta);

                const ContactResult cr = contactInitialize(cBase, offset, rigidId, isTGS,
                                                           appliedForces[workIndex], dt, bcVec,
                                                           deformableLinDelta, deformableInvMasses,
                                                           rigidBodyReferenceCount, isActive, /*checkOnly*/ false,
                                                           gSolve.solverCoreDesc, gSolve.sharedDesc, gSolve.prePrepDesc);

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

                // computeRigidChange DEFERRED: static body -> count stays 0, deltas stay 0.
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
