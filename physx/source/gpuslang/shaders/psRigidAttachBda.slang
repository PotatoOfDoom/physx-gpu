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

// psRigidAttachBda.slang
//
// P9.76 — the PARTICLE-vs-rigid ATTACHMENT trio, the third and last of the three
// FEM/particle attachment families in this port (cloth: femClothRigidAttachBda,
// softbody: softBodyRigidAttachBda).
//
//   ps_rigidAttachmentPrepareLaunch (particlesystem.cu:5437, launch
//   PxgParticleSystemCore.cpp:2366, PS_PREP_RIGID_ATTACHMENTS) — turns each
//   PxParticleRigidAttachment the application handed to
//   PxParticleBuffer::setRigidAttachments() into a three-axis constraint block.
//
//   ps_solveRigidAttachmentsLaunch      (particlesystem.cu:5567, PGS)
//   ps_solveRigidAttachmentsTGSLaunch   (particlesystem.cu:5648, TGS)
//   (launch PxgParticleSystemCore.cpp:2152, PS_SOLVE_RIGID_ATTACHMENTS{,_TGS})
//
// ===========================================================================
// THE ABI FINDING: ONE HOST PACK, TWO KERNEL SIGNATURES
// ===========================================================================
//
// The two solve kernels have DIFFERENT CUDA SIGNATURES:
//
//     ps_solveRigidAttachmentsLaunch      8 parameters, ends at `dt`   (64 bytes)
//     ps_solveRigidAttachmentsTGSLaunch  10 parameters, + biasCoefficient,
//                                                       + isVelocityIteration (72)
//
// and the HOST builds ONE 10-argument pack for both, choosing only the
// CUfunction with a ternary (PxgParticleSystemCore.cpp:2152-2167).  Under
// cuLaunchKernel's void** form that is harmless: the driver reads as many
// entries as the kernel's signature declares and ignores the rest.
//
// So the ABI row must be the HOST's, for BOTH kernels: {8,8,4,8,8,8,8,4,4,1},
// 72 bytes.  A row built from the PGS kernel's own signature would be 64, and
// the bridge -- which lays the blob out from the host's PxCudaKernelParam array
// (PxgVulkanCudaContext.cpp:443-462) and gates on
// `blobSize == pushConstantSizeFor(name)` -- would then refuse every PGS launch
// and, worse, silently truncate the TGS one that shares the row.  That is
// P9.76b's class with the two halves swapped, and the checker
// tools/check_launch_abi_pack.py is what holds it: it derives the blob from the
// host's PX_CUDA_KERNEL_PARAM list and compares it against the row.
//
// WHAT THE SHIM DOES *NOT* CHECK, measured rather than assumed.  The first draft
// of this header claimed a PGS entry point declaring a SHORT push block would be
// refused.  That is FALSE, and the experiment that says so is two lines of trace:
// this struct was temporarily shrunk to 56 bytes, rebuilt, and the engine ran
//     [pxg-launch] ps_solveRigidAttachmentsLaunch  grid=(1024,1,1) blob=72
//     [pxg-launch] sized ps_solveRigidAttachmentsLaunch OK grid(1024,1,1)
// -- accepted, and the particleattach scene still passed.  pushConstantSizeFor()
// reads the ABI ROW and never the SPIR-V (PhysXGpuVulkan.cpp:178-182), so a
// shader whose push struct is SHORTER than the row is not refused: it is blind
// to the tail, silently.  (The scene passed only because the experiment had
// stubbed `dt` with the right constant; nothing reported the truncation.)
//
// That is the more dangerous half of the pair, because refusal at least leaves a
// trace line.  It is why BOTH entry points below declare the FULL 72 bytes even
// though the PGS arm reads only the first 60 of them: the struct must match the
// row's layout, and a struct that merely happens to be long enough for what this
// port reads today would go wrong the moment the arm grows a field.
//
// (Checked, not assumed: PxgParticleSystemCore.cpp:2152 is the ONLY launch site
// of either kernel id in the engine.)
//
// ===========================================================================
// WHAT DIFFERS FROM THE SOFTBODY AND CLOTH TWINS
// ===========================================================================
//
//  1. NO identity-inertia special case.  The softbody and cloth prepares give
//     `invMass0 == 0 && !isStatic` the IDENTITY inertia and an inertiaScale of 0
//     (softBodyGM.cu:1973-1979).  The particle prepare has NO such branch: it
//     reads sqrtInvInertia unconditionally and forms
//     `resp = dot(s,s) + invMass0 + invMass1` with no scale (:5533-5539).
//     Transcribed as written.  It looks like the sibling and is not.
//
//  2. A BINARY SEARCH, not an index array.  The attachment a thread owns is found
//     by searching `mAttachmentRunsum` for the owning particle BUFFER
//     (:5474-5480) -- the attachments of all buffers of one system are one flat
//     run -- and the particle id is then buffer-local and needs
//     `mParticleBufferRunsum` added (:5487).
//
//  3. The particle is addressed SORTED.  `mUnsortedToSortedMapping` maps the
//     user's id to the simulation's, and the constraint stores
//     PxEncodeParticleIndex(systemId, SORTED id) (:5556).
//
//  4. The constraint block is PxgParticleRigidConstraint (PxgSimulationCoreDesc.h
//     :222), NOT PxgFEMRigidAttachmentConstraint: no baryOrType and no elemId,
//     and the particle id is a PxU64, so the offsets below are its own.
//
//  5. The PGS solve scales the particle delta by dt, the TGS solve does NOT
//     (:5638 `(-deltaImpulse) * invMass1 * dt` vs :5718 `(-deltaImpulse) *
//     invMass1`).  That is the SAME asymmetry the softbody pair has upside down
//     (softBodyGM.cu scales in both arms while FEMClothExternalSolve.cu:1668
//     scales in neither).  Each port follows its own CUDA; documented, not
//     reconciled.
//
//  6. The rigid reaction is NOT deferred here.  Unlike the softbody port, the
//     particle solves are the producer half of the P9.44 chain that is already
//     mapped (accumulateDeltaVRigidFirst/SecondLaunchMulti*), so `deltaVel` is
//     written exactly as CUDA writes it -- and note it is written ONLY inside
//     the `!deltaImpulse.isZero()` guard, so a lane that produces no impulse
//     leaves BOTH of its deltaVel rows untouched rather than zeroing them.
//
// DEFERRED, exactly as in the cloth and softbody ports and for the same reason:
//   - the CONE-LIMIT branch of both solves (attachments.cuh:113 / :176).  A plain
//     PxConeLimitedConstraint is default-constructed to
//     (mAngle, mLowLimit, mHighLimit) = (-1,-1,-1), for which
//     isConeLimitedEnabled() is false.  The fixture asserts that its data takes
//     the false path, so the deferral is a measured claim and not a hope.
//   - the ARTICULATION anchor.  Both solves read solverBodyIndices for a non-
//     static rigid and the CUDA itself carries the TODO ("need to figure out how
//     to make this work for articulation links!", :5616).
//
// TWO DEAD PARAMETERS, transcribed because the pack carries them: the prepare's
// `dt` (@40) and `isTGS` (@44) are read by nothing in particlesystem.cu:5437's
// body.  They are in the block because the HOST puts them there, and the block
// must match the host pack byte for byte or the launch is refused.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the prepare is a dot-of-cross
// chain through mat33MulVec plus a quaternion rotate, and its constraint block is
// compared bit-exactly against a CPU transcription.

import pxgSlangCommon;            // PxgSolverBodyData, PxgSolverTxIData
import pxgSlangPrepDesc;          // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;        // PxgSolverCoreDescGpu, PxgSolverSharedDesc{PGS,TGS}Gpu
import pxgSlangParticleDesc;      // PxgParticleSystemGpu, PxgParticleSimBufferGpu
import pxgSlangAtomicFloat;       // pxgAtomicAddF4XYZ / pxgAtomicAddF4W
import pxgSlangFemRigidAttach;    // quatRotateVec, vecGetNormalized
import pxgSlangFemRigidContact;   // crossVec, dotVec, mat33MulVec, nodeIsStaticBody, nodeIndex

// PxgParticleSystemKernelBlockDim/GridDim::PS_COLLISION.  PA_BLOCK MUST equal
// [numthreads] below: the bridge forwards only gridDim, so the stride is
// reconstructed from these constants and not from blockDim.
static const uint PA_BLOCK = 256u;   // PS_COLLISION block
static const uint PA_GRID = 1024u;   // PS_COLLISION grid

// ---------------------------------------------------------------------------
// PxgParticleRigidConstraint (PxgSimulationCoreDesc.h:222) -- SIX float4[32]
// followed by TWO PxU64[32].  3584 bytes, 16-aligned.  Not the FEM block: there
// is no baryOrType and no elemId, and particleId is 8 bytes wide.
// ---------------------------------------------------------------------------
static const uint PRC_raXn0_biasW = 0u;        // float4[32]
static const uint PRC_raXn1_biasW = 512u;
static const uint PRC_raXn2_biasW = 1024u;
static const uint PRC_velMult_invMass = 1536u;
static const uint PRC_low_high_limits = 2048u;
static const uint PRC_axis_angle = 2560u;
static const uint PRC_particleId = 3072u;      // PxU64[32]
static const uint PRC_rigidId = 3328u;         // PxU64[32]
static const uint PRC_STRIDE = 3584u;

// PxParticleRigidAttachment (PxDeformableAttachment.h:49) over
// PxParticleRigidFilterPair (PxFiltering.h:740).  mID0/mID1 (16) + the
// 16-aligned PxVec4 mLocalPose0 (16) + PxConeLimitParams (32) = 64.
static const uint PRA_mID0 = 0u;        // PxU64, the rigid node index
static const uint PRA_mID1 = 8u;        // PxU64, the BUFFER-LOCAL particle id
static const uint PRA_localPose0 = 16u; // PxVec4
static const uint PRA_lowHigh = 32u;    // PxVec4, mConeLimitParams.lowHighLimits
static const uint PRA_axisAngle = 48u;  // PxVec4, mConeLimitParams.axisAngle
static const uint PRA_STRIDE = 64u;

// PxAlignedTransform: quat @0, p @16, stride 32.
static const uint PXG_ALIGNED_TRANSFORM_STRIDE = 32u;

// ---------------------------------------------------------------------------
// ps_rigidAttachmentPrepareLaunch push blob.
// CUDA arg pack (PxgParticleSystemCore.cpp:2369): { PxgParticleSystem*
// particleSystems, PxU64* rigidAttachmentIds, PxgParticleRigidConstraint*
// rigidConstraints, PxgPrePrepDesc* preDesc, PxgConstraintPrepareDesc*
// prepareDesc, PxReal dt, bool isTGS, PxU32* activeParticlesystems }
// -> ABI row {8,8,8,8,8,4,1,8}, 56-byte blob (isTGS's byte plus 3 tail pad
// realigns the @48 pointer).
// ---------------------------------------------------------------------------
struct PsPrepAttachParams
{
    uint64_t particleSystems;         // @0
    uint64_t rigidAttachmentIds;      // @8   output
    uint64_t rigidConstraints;        // @16  output
    uint64_t preDesc;                 // @24
    uint64_t prepareDesc;             // @32
    float dt;                         // @40  DEAD (see header)
    uint isTGS;                       // @44  DEAD, bool + 3 tail pad
    uint64_t activeParticleSystems;   // @48 -> 56
}
[[vk::push_constant]]
ConstantBuffer<PsPrepAttachParams> gPsPrepAtt;

// ---------------------------------------------------------------------------
// ps_solveRigidAttachments{,TGS}Launch push blob -- ONE block for BOTH, because
// the host builds one pack for both (see the ABI finding in the header).  The
// PGS arm reads only the first 60 bytes of it; the block is full-length because
// it must mirror the ROW, not the arm.
// CUDA arg pack (PxgParticleSystemCore.cpp:2155): { PxgParticleSystem*
// particleSystems, PxgParticleRigidConstraint* attachments, PxU32
// numAttachments, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc*
// solverCoreDesc, PxgSolverSharedDesc<IterativeSolveData>* sharedDesc, float4*
// deltaVel, PxReal dt, PxReal biasCoefficient, bool isVelocityIteration }
// -> ABI row {8,8,4,8,8,8,8,4,4,1}, 72-byte blob.
// ---------------------------------------------------------------------------
struct PsSolveAttachParams
{
    uint64_t particleSystems;    // @0
    uint64_t attachments;        // @8
    uint numAttachments;         // @16
    uint _pad20;                 // @20
    uint64_t prePrepDesc;        // @24
    uint64_t solverCoreDesc;     // @32
    uint64_t sharedDesc;         // @40
    uint64_t deltaVel;           // @48  output
    float dt;                    // @56
    float biasCoefficient;       // @60  TGS only
    uint isVelocityIteration;    // @64  TGS only, bool + 3 tail pad -> 72
    uint _pad68;                 // @68
}
[[vk::push_constant]]
ConstantBuffer<PsSolveAttachParams> gPsSolveAtt;

// binarySearch (gpucommon/src/CUDA/reduction.cuh:729): the largest index with
// data[index] <= value, or 0 when data[0] > value.  Transcribed including the
// `left ? left - 1 : 0` tail -- the two spellings differ exactly when no element
// is <= value, which is the single-buffer case this port runs first.
uint psAttachBinarySearch(uint64_t data, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        const uint element = Ptr<uint>(data)[pos];
        if (element <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return (left != 0u) ? (left - 1u) : 0u;
}

// ===========================================================================
// ps_rigidAttachmentPrepareLaunch (particlesystem.cu:5437)
//
// Grid (ceil(maxRigidAttachmentsPerSystem / PS_COLLISION), nbActiveSystems, 1),
// block (PS_COLLISION, 1, 1).  NO grid-stride loop: the launch sizes the grid to
// the work, and groupId.y selects the system.  (gpuslang.gridStride's rule is
// about a CUDA loop that the port dropped; there is no loop here to drop.)
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // == PA_BLOCK; PS_COLLISION
void ps_rigidAttachmentPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gPsPrepAtt.particleSystems);
    Ptr<PxgConstraintPrepareDescGpu> prepareDesc = Ptr<PxgConstraintPrepareDescGpu>(gPsPrepAtt.prepareDesc);
    Ptr<PxgPrePrepDescGpu> preDesc = Ptr<PxgPrePrepDescGpu>(gPsPrepAtt.preDesc);

    const uint64_t bodyFramesBase = prepareDesc[0].body2WorldPool;
    const uint64_t solverBodyDataBase = prepareDesc[0].solverBodyDataPool;
    const uint64_t solverTxIDataBase = prepareDesc[0].solverBodyTxIDataPool;
    const uint64_t solverBodyIndices = preDesc[0].solverBodyIndices;

    const uint particleSystemId = Ptr<uint>(gPsPrepAtt.activeParticleSystems)[groupId.y];

    const uint attachmentIdx = groupThreadId.x + groupId.x * PA_BLOCK;
    if (attachmentIdx >= particleSystems[particleSystemId].mNumRigidAttachments)
        return;

    const uint constraintIdx = attachmentIdx + particleSystems[particleSystemId].mRigidAttachmentOffset;
    const uint index = constraintIdx / 32u;
    const uint offset = constraintIdx & 31u;
    const uint64_t cBase = gPsPrepAtt.rigidConstraints + uint64_t(index) * PRC_STRIDE;

    // Which particle BUFFER of this system owns attachment `attachmentIdx`
    // (:5474).  The attachments of all buffers form one flat run per system.
    const uint64_t attachmentRunsum = particleSystems[particleSystemId].mAttachmentRunsum;
    const uint bufferIndex = psAttachBinarySearch(attachmentRunsum,
                                                  particleSystems[particleSystemId].mCommonData_NumParticleBuffers,
                                                  attachmentIdx);
    const uint attachmentOffset = (bufferIndex == 0u) ? 0u : Ptr<uint>(attachmentRunsum)[bufferIndex];

    const uint64_t simBuffers = particleSystems[particleSystemId].mParticleSimBuffers;
    const uint64_t attBase = Ptr<PxgParticleSimBufferGpu>(simBuffers)[bufferIndex].mRigidAttachments
                             + uint64_t(attachmentIdx - attachmentOffset) * PRA_STRIDE;

    const uint64_t attID0 = Ptr<uint64_t>(attBase + PRA_mID0)[0];
    const uint64_t attID1 = Ptr<uint64_t>(attBase + PRA_mID1)[0];
    const float4 localPose0 = Ptr<float4>(attBase + PRA_localPose0)[0];
    const float4 low_high_limits = Ptr<float4>(attBase + PRA_lowHigh)[0];
    const float4 axis_angle = Ptr<float4>(attBase + PRA_axisAngle)[0];

    // mID1 is the BUFFER-LOCAL particle id; the run-sum makes it system-global,
    // and the reverse lookup makes it a SORTED index (:5487-5491).  CUDA narrows
    // the PxU64 sum to PxU32 on assignment, which `uint(attID1)` reproduces.
    const uint particleId = uint(attID1)
                          + Ptr<uint>(particleSystems[particleSystemId].mParticleBufferRunsum)[bufferIndex];
    const uint sortedParticleId = Ptr<uint>(particleSystems[particleSystemId].mUnsortedToSortedMapping)[particleId];

    Ptr<float4> position_invmass = Ptr<float4>(particleSystems[particleSystemId].mSortedPositions_InvMass);
    const float4 pointInvM = position_invmass[sortedParticleId];
    const float invMass1 = pointInvM.w;
    const float3 point = float3(pointInvM.x, pointInvM.y, pointInvM.z);
    const float3 axis = float3(axis_angle.x, axis_angle.y, axis_angle.z);
    const float3 ra4 = float3(localPose0.x, localPose0.y, localPose0.z);

    const uint64_t rigidId = attID0;
    uint idx = 0u;
    if (!nodeIsStaticBody(rigidId))
        idx = Ptr<uint>(solverBodyIndices)[nodeIndex(rigidId)];

    Ptr<uint64_t>(gPsPrepAtt.rigidAttachmentIds)[constraintIdx] = rigidId;

    // NO identity-inertia special case here -- see header note 1.
    float invSqrtInertia[9];
    {
        PxgSolverTxIData txI = Ptr<PxgSolverTxIData>(solverTxIDataBase)[idx];
        for (uint k = 0; k < 9u; ++k)
            invSqrtInertia[k] = txI.sqrtInvInertia[k];
    }
    const float4 linVel_invMass0 =
        Ptr<PxgSolverBodyData>(solverBodyDataBase)[idx].initialLinVelXYZ_invMassW;
    const float invMass0 = linVel_invMass0.w;

    const float4 bfQ = Ptr<float4>(bodyFramesBase + uint64_t(idx) * PXG_ALIGNED_TRANSFORM_STRIDE + 0u)[0];
    const float4 bfP = Ptr<float4>(bodyFramesBase + uint64_t(idx) * PXG_ALIGNED_TRANSFORM_STRIDE + 16u)[0];
    const float3 bodyFrame0p = float3(bfP.x, bfP.y, bfP.z);

    const float3 worldAxis = vecGetNormalized(quatRotateVec(bfQ, axis));
    const float3 ra = quatRotateVec(bfQ, ra4);
    const float3 error = float3(ra.x + bodyFrame0p.x - point.x,
                                ra.y + bodyFrame0p.y - point.y,
                                ra.z + bodyFrame0p.z - point.z);

    const float3 normal0 = float3(1.0f, 0.0f, 0.0f);
    const float3 normal1 = float3(0.0f, 1.0f, 0.0f);
    const float3 normal2 = float3(0.0f, 0.0f, 1.0f);

    const float3 raXn0 = crossVec(ra, normal0);
    const float3 raXn1 = crossVec(ra, normal1);
    const float3 raXn2 = crossVec(ra, normal2);

    const float3 raXnSqrtInertia0 = mat33MulVec(invSqrtInertia, raXn0);
    const float3 raXnSqrtInertia1 = mat33MulVec(invSqrtInertia, raXn1);
    const float3 raXnSqrtInertia2 = mat33MulVec(invSqrtInertia, raXn2);

    const float resp0 = dotVec(raXnSqrtInertia0, raXnSqrtInertia0) + invMass0 + invMass1;
    const float resp1 = dotVec(raXnSqrtInertia1, raXnSqrtInertia1) + invMass0 + invMass1;
    const float resp2 = dotVec(raXnSqrtInertia2, raXnSqrtInertia2) + invMass0 + invMass1;

    const float velMultiplier0 = (resp0 > 0.0f) ? (1.0f / resp0) : 0.0f;
    const float velMultiplier1 = (resp1 > 0.0f) ? (1.0f / resp1) : 0.0f;
    const float velMultiplier2 = (resp2 > 0.0f) ? (1.0f / resp2) : 0.0f;

    const float biasedErr0 = dotVec(error, normal0);
    const float biasedErr1 = dotVec(error, normal1);
    const float biasedErr2 = dotVec(error, normal2);

    Ptr<float4>(cBase + PRC_raXn0_biasW)[offset] = float4(raXnSqrtInertia0.x, raXnSqrtInertia0.y, raXnSqrtInertia0.z, biasedErr0);
    Ptr<float4>(cBase + PRC_raXn1_biasW)[offset] = float4(raXnSqrtInertia1.x, raXnSqrtInertia1.y, raXnSqrtInertia1.z, biasedErr1);
    Ptr<float4>(cBase + PRC_raXn2_biasW)[offset] = float4(raXnSqrtInertia2.x, raXnSqrtInertia2.y, raXnSqrtInertia2.z, biasedErr2);
    Ptr<float4>(cBase + PRC_velMult_invMass)[offset] = float4(velMultiplier0, velMultiplier1, velMultiplier2, invMass0);
    // PxEncodeParticleIndex(systemId, SORTED id) = (u64(systemId) << 32) | id.
    Ptr<uint64_t>(cBase + PRC_particleId)[offset] =
        (uint64_t(particleSystemId) << 32) | uint64_t(sortedParticleId);
    Ptr<uint64_t>(cBase + PRC_rigidId)[offset] = attID0;
    Ptr<float4>(cBase + PRC_low_high_limits)[offset] = low_high_limits;
    // The NEGATED spelling is commented out upstream (:5560); the positive one is live.
    Ptr<float4>(cBase + PRC_axis_angle)[offset] = float4(worldAxis.x, worldAxis.y, worldAxis.z, axis_angle.w);
}

// The pieces both solves read out of a constraint lane before they diverge.
struct PsAttachLane
{
    float4 raXn0_biasW;
    float4 raXn1_biasW;
    float4 raXn2_biasW;
    float4 velMult;
    float3 raXn0;
    float3 raXn1;
    float3 raXn2;
}
PsAttachLane psAttachLane(uint64_t cBase, uint offset)
{
    PsAttachLane l;
    l.raXn0_biasW = Ptr<float4>(cBase + PRC_raXn0_biasW)[offset];
    l.raXn1_biasW = Ptr<float4>(cBase + PRC_raXn1_biasW)[offset];
    l.raXn2_biasW = Ptr<float4>(cBase + PRC_raXn2_biasW)[offset];
    l.velMult = Ptr<float4>(cBase + PRC_velMult_invMass)[offset];
    l.raXn0 = float3(l.raXn0_biasW.x, l.raXn0_biasW.y, l.raXn0_biasW.z);
    l.raXn1 = float3(l.raXn1_biasW.x, l.raXn1_biasW.y, l.raXn1_biasW.z);
    l.raXn2 = float3(l.raXn2_biasW.x, l.raXn2_biasW.y, l.raXn2_biasW.z);
    return l;
}

// The rigid reaction both solves write (:5641 / :5721).  Written ONLY when the
// impulse is non-zero: a lane with a zero impulse leaves both rows alone.
void psAttachWriteRigidDelta(uint workIndex, uint numAttachments, float3 deltaLinVel, float3 deltaAngVel)
{
    Ptr<float4> deltaVel = Ptr<float4>(gPsSolveAtt.deltaVel);
    deltaVel[workIndex] = float4(deltaLinVel.x, deltaLinVel.y, deltaLinVel.z, 1.0f);
    deltaVel[workIndex + numAttachments] = float4(deltaAngVel.x, deltaAngVel.y, deltaAngVel.z, 0.0f);
}

// ===========================================================================
// ps_solveRigidAttachmentsLaunch (particlesystem.cu:5567) — PGS.
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // == PA_BLOCK; PS_COLLISION
void ps_solveRigidAttachmentsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gPsSolveAtt.particleSystems);
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gPsSolveAtt.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gPsSolveAtt.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gPsSolveAtt.prePrepDesc);

    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyDeltaVel = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);
    Ptr<float4> initialVel = Ptr<float4>(solverCoreDesc[0].outSolverVelocity);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;

    const uint numAttachments = gPsSolveAtt.numAttachments;
    const float dt = gPsSolveAtt.dt;
    const float invDt = 1.0f / dt;
    const uint nbBlocksRequired = (numAttachments + PA_BLOCK - 1u) / PA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PA_GRID - 1u) / PA_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PA_BLOCK + idx + nbIterationsPerBlock * groupId.x * PA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gPsSolveAtt.attachments + uint64_t(index) * PRC_STRIDE;

        const uint64_t compressedId = Ptr<uint64_t>(cBase + PRC_particleId)[offset];
        const uint particleSystemId = uint(compressedId >> 32);   // PxGetParticleSystemId
        const uint particleId = uint(compressedId);               // PxGetParticleIndex

        Ptr<float4> sortedPos = Ptr<float4>(particleSystems[particleSystemId].mSortedPositions_InvMass);
        Ptr<float4> sortedVel = Ptr<float4>(particleSystems[particleSystemId].mSortedVelocities);

        const float4 vel = sortedVel[particleId];
        const float4 pos_invmass = sortedPos[particleId];
        const float3 linVel1 = float3(vel.x, vel.y, vel.z);
        const float invMass1 = pos_invmass.w;

        const uint64_t rigidId = Ptr<uint64_t>(cBase + PRC_rigidId)[offset];
        const bool isStatic = nodeIsStaticBody(rigidId);
        uint solverBodyIndex = 0u;
        if (!isStatic)
            solverBodyIndex = Ptr<uint>(prePrepDesc[0].solverBodyIndices)[nodeIndex(rigidId)];
        else if (invMass1 == 0.0f)
            continue;   // an infinite-mass particle pinned to a static body: nothing to solve

        const float4 iv0 = initialVel[solverBodyIndex];
        const float4 iv1 = initialVel[solverBodyIndex + numSolverBodies];
        const float4 dv0 = solverBodyDeltaVel[accOffset + solverBodyIndex];
        const float4 dv1 = solverBodyDeltaVel[accOffset + solverBodyIndex + numSolverBodies];
        const float3 rigidLinVel = float3(iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z);
        const float3 rigidAngVel = float3(iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z);

        // calculateAttachmentDeltaImpulsePGS (attachments.cuh:83), the TEMPLATE
        // overload (:138) -- so raXn0/1/2 are three DISTINCT rows and each axis
        // reads its OWN position error.  P9.76a is the record of what assuming
        // otherwise costs.
        const PsAttachLane l = psAttachLane(cBase, offset);
        const float velOfRigidX = rigidLinVel.x + dotVec(rigidAngVel, l.raXn0);
        const float velOfRigidY = rigidLinVel.y + dotVec(rigidAngVel, l.raXn1);
        const float velOfRigidZ = rigidLinVel.z + dotVec(rigidAngVel, l.raXn2);

        const float biasFactor = 0.5f;
        // Cone-LIMITED branch DEFERRED (see header).
        const float3 velError = float3(linVel1.x - l.raXn0_biasW.w * biasFactor * invDt,
                                       linVel1.y - l.raXn1_biasW.w * biasFactor * invDt,
                                       linVel1.z - l.raXn2_biasW.w * biasFactor * invDt);

        const float deltaF0 = (velError.x - velOfRigidX) * l.velMult.x;
        const float deltaF1 = (velError.y - velOfRigidY) * l.velMult.y;
        const float deltaF2 = (velError.z - velOfRigidZ) * l.velMult.z;
        const float3 deltaImpulse = float3(deltaF0, deltaF1, deltaF2);

        if (deltaImpulse.x != 0.0f || deltaImpulse.y != 0.0f || deltaImpulse.z != 0.0f)
        {
            // PGS scales by dt here; the TGS twin does NOT (see header note 5).
            const float3 deltaParticle = float3((-deltaImpulse.x) * invMass1 * dt,
                                                (-deltaImpulse.y) * invMass1 * dt,
                                                (-deltaImpulse.z) * invMass1 * dt);
            const uint64_t accumDeltaP = particleSystems[particleSystemId].mAccumDeltaP;
            pxgAtomicAddF4XYZ(accumDeltaP, particleId, deltaParticle);
            // AtomicAdd(..., 0.f): CUDA really does an atomic add of ZERO to .w
            // (atomic.cuh:67-73), so it is transcribed rather than dropped.
            pxgAtomicAddF4W(accumDeltaP, particleId, 0.0f);

            const float3 deltaLinVel = float3(deltaImpulse.x * l.velMult.w,
                                              deltaImpulse.y * l.velMult.w,
                                              deltaImpulse.z * l.velMult.w);
            const float3 deltaAngVel = float3(l.raXn0.x * deltaF0 + l.raXn1.x * deltaF1 + l.raXn2.x * deltaF2,
                                              l.raXn0.y * deltaF0 + l.raXn1.y * deltaF1 + l.raXn2.y * deltaF2,
                                              l.raXn0.z * deltaF0 + l.raXn1.z * deltaF1 + l.raXn2.z * deltaF2);
            psAttachWriteRigidDelta(workIndex, numAttachments, deltaLinVel, deltaAngVel);
        }
    }
}

// ===========================================================================
// ps_solveRigidAttachmentsTGSLaunch (particlesystem.cu:5648) — TGS.
//
// Reads mSortedDeltaP (a POSITION delta) where the PGS arm reads
// mSortedVelocities, unpacks FOUR vectors from THREE pool rows, and forms a
// position error scaled by biasCoefficient.  Written as its own function rather
// than an isTGS flag through the PGS body: the two share their loop and their
// lane read and nothing else, and P9.76c is the record of what a shared body
// hides when the shared half is the smaller half.
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // == PA_BLOCK; PS_COLLISION
void ps_solveRigidAttachmentsTGSLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gPsSolveAtt.particleSystems);
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gPsSolveAtt.solverCoreDesc);
    Ptr<PxgSolverSharedDescTGSGpu> sharedDesc = Ptr<PxgSolverSharedDescTGSGpu>(gPsSolveAtt.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gPsSolveAtt.prePrepDesc);

    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyVel = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;

    const uint numAttachments = gPsSolveAtt.numAttachments;
    const float dt = gPsSolveAtt.dt;
    const float biasCoefficient = gPsSolveAtt.biasCoefficient;
    const bool isVelocityIteration = gPsSolveAtt.isVelocityIteration != 0u;
    const uint nbBlocksRequired = (numAttachments + PA_BLOCK - 1u) / PA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PA_GRID - 1u) / PA_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PA_BLOCK + idx + nbIterationsPerBlock * groupId.x * PA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gPsSolveAtt.attachments + uint64_t(index) * PRC_STRIDE;

        const uint64_t compressedId = Ptr<uint64_t>(cBase + PRC_particleId)[offset];
        const uint particleSystemId = uint(compressedId >> 32);
        const uint particleId = uint(compressedId);

        Ptr<float4> sortedDeltaPInvMassW = Ptr<float4>(particleSystems[particleSystemId].mSortedDeltaP);
        const float4 deltaP_invMassW = sortedDeltaPInvMassW[particleId];
        const float invMass1 = deltaP_invMassW.w;
        const float3 deltaP = float3(deltaP_invMassW.x, deltaP_invMassW.y, deltaP_invMassW.z);

        const uint64_t rigidId = Ptr<uint64_t>(cBase + PRC_rigidId)[offset];
        const bool isStatic = nodeIsStaticBody(rigidId);
        uint solverBodyIndex = 0u;
        if (!isStatic)
            solverBodyIndex = Ptr<uint>(prePrepDesc[0].solverBodyIndices)[nodeIndex(rigidId)];
        else if (invMass1 == 0.0f)
            continue;

        // readVelocitiesTGS's packing (dataReadWriteHelper.cuh:144-151), open-coded
        // here exactly as particlesystem.cu:5706-5714 open-codes it.  THREE rows.
        const float4 v0 = solverBodyVel[accOffset + solverBodyIndex];
        const float4 v1 = solverBodyVel[accOffset + solverBodyIndex + numSolverBodies];
        const float4 v2 = solverBodyVel[accOffset + solverBodyIndex + numSolverBodies + numSolverBodies];
        const float3 rigidLinVel = float3(v0.x, v0.y, v0.z);
        const float3 rigidAngVel = float3(v0.w, v1.x, v1.y);
        const float3 rigidLinDelta = float3(v1.z, v1.w, v2.x);
        const float3 rigidAngDelta = float3(v2.y, v2.z, v2.w);

        // calculateAttachmentDeltaImpulseTGS (attachments.cuh:147).
        const PsAttachLane l = psAttachLane(cBase, offset);
        const float velOfRigid0 = rigidLinVel.x + dotVec(rigidAngVel, l.raXn0);
        const float velOfRigid1 = rigidLinVel.y + dotVec(rigidAngVel, l.raXn1);
        const float velOfRigid2 = rigidLinVel.z + dotVec(rigidAngVel, l.raXn2);

        const float3 linDelta = float3(deltaP.x - rigidLinDelta.x,
                                       deltaP.y - rigidLinDelta.y,
                                       deltaP.z - rigidLinDelta.z);

        // Cone-LIMITED branch DEFERRED (see header).
        const float3 tgsError = float3(linDelta.x - l.raXn0_biasW.w - dotVec(rigidAngDelta, l.raXn0),
                                       linDelta.y - l.raXn1_biasW.w - dotVec(rigidAngDelta, l.raXn1),
                                       linDelta.z - l.raXn2_biasW.w - dotVec(rigidAngDelta, l.raXn2));

        const float velDt = isVelocityIteration ? 0.0f : dt;
        const float deltaF0 = (tgsError.x - velOfRigid0 * velDt) * l.velMult.x * biasCoefficient;
        const float deltaF1 = (tgsError.y - velOfRigid1 * velDt) * l.velMult.y * biasCoefficient;
        const float deltaF2 = (tgsError.z - velOfRigid2 * velDt) * l.velMult.z * biasCoefficient;
        const float3 deltaImpulse = float3(deltaF0, deltaF1, deltaF2);

        if (deltaImpulse.x != 0.0f || deltaImpulse.y != 0.0f || deltaImpulse.z != 0.0f)
        {
            // NO dt here -- see header note 5.
            const float3 deltaParticle = float3((-deltaImpulse.x) * invMass1,
                                                (-deltaImpulse.y) * invMass1,
                                                (-deltaImpulse.z) * invMass1);
            const uint64_t accumDeltaP = particleSystems[particleSystemId].mAccumDeltaP;
            pxgAtomicAddF4XYZ(accumDeltaP, particleId, deltaParticle);
            pxgAtomicAddF4W(accumDeltaP, particleId, 0.0f);

            const float3 deltaLinVel = float3(deltaImpulse.x * l.velMult.w,
                                              deltaImpulse.y * l.velMult.w,
                                              deltaImpulse.z * l.velMult.w);
            const float3 deltaAngVel = float3(l.raXn0.x * deltaF0 + l.raXn1.x * deltaF1 + l.raXn2.x * deltaF2,
                                              l.raXn0.y * deltaF0 + l.raXn1.y * deltaF1 + l.raXn2.y * deltaF2,
                                              l.raXn0.z * deltaF0 + l.raXn1.z * deltaF1 + l.raXn2.z * deltaF2);
            psAttachWriteRigidDelta(workIndex, numAttachments, deltaLinVel, deltaAngVel);
        }
    }
}
