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

// pxgSlangDeformableUtils -- the PhysX 5.12 deformableUtils.cuh pieces the deformable
// consumers (softBody.cu, softBodyGM.cu, FEMCloth.cu) call, transcribed from
// gpusimulationcontroller/src/CUDA/deformableUtils.cuh.  The solve-side half
// (deformableAndParticleUtils.cuh) is pxgSlangDeformableAndParticleUtils.
//
//   barycentricProjectTet / barycentricProjectTri   the per-vertex float4 blends.
//   prepareRigidDbContact     writes one PxgRigidDbContactBlock lane (both arms: the
//                             articulation link and the rigid / static body).  Replaces
//                             the 5.6.1 prepareFEMContacts (pxgSlangFemRigidContact): no
//                             barycentric slot, maxPenBiasClamp instead of maxPenBias.
//   prepareRigidDbAttachment  writes one PxgRigidDbAttachmentBlock lane (both arms).  The
//                             5.6.1 cone limits are gone; the deformable ref, the rigid
//                             node and the rigid attachment reference count are stored.
//   finalizeVertexVelocity    with the two averaging operators,
//                             LinearBlendVelocityAveraging (softbody) and
//                             MagnitudeClampVelocityAveraging (cloth), selected by
//                             PXG_VEL_AVG_* + the blend scale.
//
// FLOAT SPELLING.  PxMax / PxMin on floats are intrinsics::selectMax / selectMin (the dp*
// helpers of pxgSlangDeformableAndParticleUtils), PxVec3 dot is (x + y) + z, PxMat33 * v is
// (c0 * v.x + c1 * v.y) + c2 * v.z.  Every importer must be in PXG_NOCONTRACT_SHADERS.
//
// The block lane is the kernel's (threadIdx.x & 31), which equals (workIndex & 31) because
// the strided workIndex differs from threadIdx.x by a multiple of the 256-thread block;
// callers pass it in.
module pxgSlangDeformableUtils;

import pxgSlangPrepDesc;            // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangCommon;              // PxgSolverBodyData, PxgSolverTxIData, PxgBodySim
import pxgSlangSolverDesc;          // PxgSolverSharedDescPGSGpu (mBodySimBufferDeviceData, articulations)
import pxgSlangArticulationDesc;    // PXG_ARTI_*, artiPtrField
import pxgSlangArticulationBlocks;  // AbMatrix6x6
import pxgSlangSpatialMatrix;       // AbSpatialVector, asmSVec, asmM6MulVec, asmLoad6x6Record
import pxgSlangDeformableRef;       // PXG_RDBC_*, PXG_RDBA_*, PxgNode*, PxgGetDeformableFeatureType
import pxgSlangDeformableAndParticleUtils; // dpSelectMax / dpSelectMin, dpDot3, dpCross3, dpComputeBasisVectors

// PxMat33 (column-major m[9]) * PxVec3: (column0 * v.x + column1 * v.y) + column2 * v.z.
public float3 duMat33MulVec(float m[9], float3 v)
{
    return float3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
                  m[1] * v.x + m[4] * v.y + m[7] * v.z,
                  m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// PxQuat::rotate (PxQuat.h), also PxTransform::rotate / PxAlignedTransform::rotate.
public float3 duQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// ---------------------------------------------------------------------------
// barycentricProjectTet / barycentricProjectTri: ((a * b.x + b * b.y) + c * b.z) [+ d * b.w].
// ---------------------------------------------------------------------------
public float4 barycentricProjectTet(uint4 vertIdx, uint64_t buffer, float4 barycentric)
{
    Ptr<float4> buf = Ptr<float4>(buffer);
    const float4 a = buf[vertIdx.x];
    const float4 b = buf[vertIdx.y];
    const float4 c = buf[vertIdx.z];
    const float4 d = buf[vertIdx.w];
    return a * barycentric.x + b * barycentric.y + c * barycentric.z + d * barycentric.w;
}

public float4 barycentricProjectTri(uint4 vertIdx, uint64_t buffer, float4 barycentric)
{
    Ptr<float4> buf = Ptr<float4>(buffer);
    const float4 a = buf[vertIdx.x];
    const float4 b = buf[vertIdx.y];
    const float4 c = buf[vertIdx.z];
    return a * barycentric.x + b * barycentric.y + c * barycentric.z;
}

// The articulation of a link node: articulations[bodySims[node].articulationRemapId].
uint64_t duArticulationOf(uint64_t sharedDescPtr, uint64_t rigidId)
{
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(sharedDescPtr);
    Ptr<PxgBodySim> bodySims = Ptr<PxgBodySim>(sharedDesc[0].mBodySimBufferDeviceData);
    const uint artiId = bodySims[PxgNodeIndexOf(rigidId)].articulationRemapId;
    return sharedDesc[0].articulations + uint64_t(artiId) * PXG_ARTI_STRIDE;
}

// ===========================================================================
// prepareRigidDbAttachment (deformableUtils.cuh).
//   block      the PxgRigidDbAttachmentBlock base, lane its slot.
//   point / deformableInvMass   the deformable attach point and its (raw bc^2) inverse mass.
//   rigidLocalPose              the anchor in the rigid's frame.
//   elemId     the PxgDeformableRef; a VERTEX ref leaves the barycentric slot alone.
//   rigidDeltaVel == 0 is the CUDA NULL (the rigid arm zeroes the two slots otherwise).
// The articulation arm writes no rigidDeltaVel slot (the CUDA has none there either).
// ===========================================================================
public void prepareRigidDbAttachment(uint64_t block, uint lane, float3 point, float deformableInvMass,
                                     float3 rigidLocalPose, uint64_t rigidId, uint64_t elemId,
                                     float4 barycentric, uint rigidBodyRefCount,
                                     uint64_t prepareDescPtr, uint64_t solverBodyIndices,
                                     uint64_t sharedDescPtr, uint64_t rigidDeltaVel, uint workIndex,
                                     uint numRigidAttachments)
{
    const bool elemIsVertex = PxgGetDeformableFeatureType(elemId) == PXG_DREF_eVERTEX;
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(prepareDescPtr);

    uint idx = 0u;
    if (!PxgNodeIsStaticBody(rigidId))
        idx = Ptr<uint>(solverBodyIndices)[PxgNodeIndexOf(rigidId)];

    const float3 normal0 = float3(1.0f, 0.0f, 0.0f);
    const float3 normal1 = float3(0.0f, 1.0f, 0.0f);
    const float3 normal2 = float3(0.0f, 0.0f, 1.0f);

    if (PxgNodeIsArticulation(rigidId))
    {
        const uint64_t articulation = duArticulationOf(sharedDescPtr, rigidId);
        const uint linkID = PxgNodeArticulationLinkId(rigidId);
        // linkBody2Worlds[linkID]: PxTransform (28 B), q @0, p @16.
        Ptr<float> body2World = Ptr<float>(artiPtrField(articulation, PXG_ARTI_LINK_BODY2WORLDS)
                                           + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
        const float4 b2wQ = float4(body2World[0], body2World[1], body2World[2], body2World[3]);
        const float3 bodyFrame0p = float3(body2World[4], body2World[5], body2World[6]);

        const float3 ra = duQuatRotate(b2wQ, rigidLocalPose);
        // error = ra + bodyFrame0p - point, left to right.
        const float3 error = float3(ra.x + bodyFrame0p.x - point.x, ra.y + bodyFrame0p.y - point.y,
                                    ra.z + bodyFrame0p.z - point.z);

        const float3 raXn0 = dpCross3(ra, normal0);
        const float3 raXn1 = dpCross3(ra, normal1);
        const float3 raXn2 = dpCross3(ra, normal2);

        const AbMatrix6x6 spatialResponse =
            asmLoad6x6Record(artiPtrField(articulation, PXG_ARTI_SPATIAL_RESPONSE_MATRIX_W), linkID);
        const AbSpatialVector deltaV0 = asmM6MulVec(spatialResponse, asmSVec(normal0, raXn0));
        const AbSpatialVector deltaV1 = asmM6MulVec(spatialResponse, asmSVec(normal1, raXn1));
        const AbSpatialVector deltaV2 = asmM6MulVec(spatialResponse, asmSVec(normal2, raXn2));

        const float resp0 = dpDot3(deltaV0.top, raXn0) + dpDot3(deltaV0.bottom, normal0) + deformableInvMass;
        const float resp1 = dpDot3(deltaV1.top, raXn1) + dpDot3(deltaV1.bottom, normal1) + deformableInvMass;
        const float resp2 = dpDot3(deltaV2.top, raXn2) + dpDot3(deltaV2.bottom, normal2) + deformableInvMass;

        const float velMultiplier0 = (resp0 > 0.0f) ? (1.0f / resp0) : 0.0f;
        const float velMultiplier1 = (resp1 > 0.0f) ? (1.0f / resp1) : 0.0f;
        const float velMultiplier2 = (resp2 > 0.0f) ? (1.0f / resp2) : 0.0f;

        const float biasedErr0 = dpDot3(error, normal0);
        const float biasedErr1 = dpDot3(error, normal1);
        const float biasedErr2 = dpDot3(error, normal2);

        Ptr<float4>(block + PXG_RDBA_raXn0_biasW)[lane] = float4(raXn0.x, raXn0.y, raXn0.z, biasedErr0);
        Ptr<float4>(block + PXG_RDBA_raXn1_biasW)[lane] = float4(raXn1.x, raXn1.y, raXn1.z, biasedErr1);
        Ptr<float4>(block + PXG_RDBA_raXn2_biasW)[lane] = float4(raXn2.x, raXn2.y, raXn2.z, biasedErr2);
        // Articulations don't use invMass0; it is 1 so the linear impulse stays an impulse.
        Ptr<float4>(block + PXG_RDBA_velMultiplierXYZ_invMassW)[lane] =
            float4(velMultiplier0, velMultiplier1, velMultiplier2, 1.0f);
        Ptr<uint64_t>(block + PXG_RDBA_deformableRef)[lane] = elemId;
        Ptr<uint64_t>(block + PXG_RDBA_rigidId)[lane] = rigidId;
        if (!elemIsVertex)
            Ptr<float4>(block + PXG_RDBA_barycentric)[lane] = barycentric;
        Ptr<uint>(block + PXG_RDBA_rigidBodyRefCount)[lane] = rigidBodyRefCount;
    }
    else
    {
        const PxgSolverBodyData sbd = Ptr<PxgSolverBodyData>(prepDesc[0].solverBodyDataPool)[idx];
        const float4 linVel_invMass0 = sbd.initialLinVelXYZ_invMassW;
        const float invMass0 = linVel_invMass0.w;

        float invSqrtInertia0[9];
        float inertiaScale = 1.0f;
        if (invMass0 == 0.0f && !PxgNodeIsStaticBody(rigidId))
        {
            // PxMat33(PxIdentity), column-major.
            invSqrtInertia0[0] = 1.0f; invSqrtInertia0[1] = 0.0f; invSqrtInertia0[2] = 0.0f;
            invSqrtInertia0[3] = 0.0f; invSqrtInertia0[4] = 1.0f; invSqrtInertia0[5] = 0.0f;
            invSqrtInertia0[6] = 0.0f; invSqrtInertia0[7] = 0.0f; invSqrtInertia0[8] = 1.0f;
            inertiaScale = 0.0f;
        }
        else
        {
            const PxgSolverTxIData txI = Ptr<PxgSolverTxIData>(prepDesc[0].solverBodyTxIDataPool)[idx];
            for (uint k = 0; k < 9u; ++k)
                invSqrtInertia0[k] = txI.sqrtInvInertia[k];
        }

        // body2WorldPool[idx]: PxAlignedTransform (32 B), q @0, p @16.
        const uint64_t bodyFrame = prepDesc[0].body2WorldPool + uint64_t(idx) * 32UL;
        const float4 bfQ = Ptr<float4>(bodyFrame)[0];
        const float4 bfP = Ptr<float4>(bodyFrame + 16UL)[0];
        const float3 bodyFrame0p = float3(bfP.x, bfP.y, bfP.z);

        const float3 ra = duQuatRotate(bfQ, rigidLocalPose);
        const float3 error = float3(ra.x + bodyFrame0p.x - point.x, ra.y + bodyFrame0p.y - point.y,
                                    ra.z + bodyFrame0p.z - point.z);

        const float3 raXn0 = dpCross3(ra, normal0);
        const float3 raXn1 = dpCross3(ra, normal1);
        const float3 raXn2 = dpCross3(ra, normal2);

        const float3 raXnSqrtInertia0 = duMat33MulVec(invSqrtInertia0, raXn0);
        const float3 raXnSqrtInertia1 = duMat33MulVec(invSqrtInertia0, raXn1);
        const float3 raXnSqrtInertia2 = duMat33MulVec(invSqrtInertia0, raXn2);
        const float resp0 = (dpDot3(raXnSqrtInertia0, raXnSqrtInertia0)) * inertiaScale + invMass0 + deformableInvMass;
        const float resp1 = (dpDot3(raXnSqrtInertia1, raXnSqrtInertia1)) * inertiaScale + invMass0 + deformableInvMass;
        const float resp2 = (dpDot3(raXnSqrtInertia2, raXnSqrtInertia2)) * inertiaScale + invMass0 + deformableInvMass;

        const float velMultiplier0 = (resp0 > 0.0f) ? (1.0f / resp0) : 0.0f;
        const float velMultiplier1 = (resp1 > 0.0f) ? (1.0f / resp1) : 0.0f;
        const float velMultiplier2 = (resp2 > 0.0f) ? (1.0f / resp2) : 0.0f;

        const float biasedErr0 = dpDot3(error, normal0);
        const float biasedErr1 = dpDot3(error, normal1);
        const float biasedErr2 = dpDot3(error, normal2);

        Ptr<float4>(block + PXG_RDBA_raXn0_biasW)[lane] =
            float4(raXnSqrtInertia0.x, raXnSqrtInertia0.y, raXnSqrtInertia0.z, biasedErr0);
        Ptr<float4>(block + PXG_RDBA_raXn1_biasW)[lane] =
            float4(raXnSqrtInertia1.x, raXnSqrtInertia1.y, raXnSqrtInertia1.z, biasedErr1);
        Ptr<float4>(block + PXG_RDBA_raXn2_biasW)[lane] =
            float4(raXnSqrtInertia2.x, raXnSqrtInertia2.y, raXnSqrtInertia2.z, biasedErr2);
        Ptr<float4>(block + PXG_RDBA_velMultiplierXYZ_invMassW)[lane] =
            float4(velMultiplier0, velMultiplier1, velMultiplier2, invMass0);
        Ptr<uint64_t>(block + PXG_RDBA_deformableRef)[lane] = elemId;
        Ptr<uint64_t>(block + PXG_RDBA_rigidId)[lane] = rigidId;
        if (!elemIsVertex)
            Ptr<float4>(block + PXG_RDBA_barycentric)[lane] = barycentric;
        Ptr<uint>(block + PXG_RDBA_rigidBodyRefCount)[lane] = rigidBodyRefCount;

        // The pointer is compared, never loaded through, so there is nothing to speculate.
        if (rigidDeltaVel != 0UL)
        {
            Ptr<float4>(rigidDeltaVel)[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            Ptr<float4>(rigidDeltaVel)[workIndex + numRigidAttachments] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ===========================================================================
// prepareRigidDbContact (deformableUtils.cuh).
//   block / lane           the PxgRigidDbContactBlock base and slot.
//   normal, p, pen, delta  the contact; error = pen + delta.dot(normal) re-references the
//                          predicted-position penetration to the start of the step.
//   solverBodyIndices      preDesc->solverBodyIndices.
//   penBiasClampFEM        the deformable's mPenBiasClamp; the slot keeps the pair max.
// invDt and isTGS are taken by the CUDA function and never read; they are not parameters.
// ===========================================================================
public void prepareRigidDbContact(uint64_t block, uint lane, float3 normal, uint64_t sharedDescPtr,
                                  float3 p, float pen, float3 delta, uint64_t rigidId,
                                  uint64_t prepareDescPtr, uint64_t solverBodyIndices,
                                  float penBiasClampFEM)
{
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(prepareDescPtr);

    float3 t0, t1;
    dpComputeBasisVectors(normal, t0, t1);

    float penBiasClampRigid;
    float4 raXn_resp;
    float4 raXnF0_resp;
    float4 raXnF1_resp;
    float invMass0;

    if (PxgNodeIsArticulation(rigidId))
    {
        const uint64_t articulation = duArticulationOf(sharedDescPtr, rigidId);
        const uint linkID = PxgNodeArticulationLinkId(rigidId);
        Ptr<float> body2World = Ptr<float>(artiPtrField(articulation, PXG_ARTI_LINK_BODY2WORLDS)
                                           + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
        // links[linkID].initialAngVelXYZ_penBiasClamp.w
        penBiasClampRigid = Ptr<float>(artiPtrField(articulation, PXG_ARTI_LINKS)
                                       + uint64_t(linkID) * PXG_ARTI_LINK_SIZE
                                       + PXG_ARTI_LINKREC_INITIAL_ANGVEL_PENBIAS)[3];

        const float3 bodyFrame0p = float3(body2World[4], body2World[5], body2World[6]);

        const float3 ra = float3(p.x - bodyFrame0p.x, p.y - bodyFrame0p.y, p.z - bodyFrame0p.z);
        const float3 raXn = dpCross3(ra, normal);
        const float3 raXF0 = dpCross3(ra, t0);
        const float3 raXF1 = dpCross3(ra, t1);

        const AbMatrix6x6 spatialResponse =
            asmLoad6x6Record(artiPtrField(articulation, PXG_ARTI_SPATIAL_RESPONSE_MATRIX_W), linkID);

        const AbSpatialVector deltaV0 = asmM6MulVec(spatialResponse, asmSVec(normal, raXn));
        const float resp0 = dpDot3(deltaV0.top, raXn) + dpDot3(deltaV0.bottom, normal);

        const AbSpatialVector deltaFV0 = asmM6MulVec(spatialResponse, asmSVec(t0, raXF0));
        const AbSpatialVector deltaFV1 = asmM6MulVec(spatialResponse, asmSVec(t1, raXF1));

        const float respF0 = dpDot3(deltaFV0.top, raXF0) + dpDot3(deltaFV0.bottom, t0);
        const float respF1 = dpDot3(deltaFV1.top, raXF1) + dpDot3(deltaFV1.bottom, t1);

        raXn_resp = float4(raXn.x, raXn.y, raXn.z, resp0);
        raXnF0_resp = float4(raXF0.x, raXF0.y, raXF0.z, respF0);
        raXnF1_resp = float4(raXF1.x, raXF1.y, raXF1.z, respF1);

        // Articulations don't use invMass0; 1 gives the linear impulse rather than a velocity change.
        invMass0 = 1.0f;
    }
    else
    {
        uint idx = 0u;
        if (!PxgNodeIsStaticBody(rigidId))
            idx = Ptr<uint>(solverBodyIndices)[PxgNodeIndexOf(rigidId)];

        const PxgSolverTxIData txI = Ptr<PxgSolverTxIData>(prepDesc[0].solverBodyTxIDataPool)[idx];
        float invSqrtInertia0[9];
        for (uint k = 0; k < 9u; ++k)
            invSqrtInertia0[k] = txI.sqrtInvInertia[k];

        const PxgSolverBodyData sbd = Ptr<PxgSolverBodyData>(prepDesc[0].solverBodyDataPool)[idx];
        const float4 linVel_invMass0 = sbd.initialLinVelXYZ_invMassW;
        penBiasClampRigid = sbd.initialAngVelXYZ_penBiasClamp.w;
        invMass0 = linVel_invMass0.w;

        // both static and kinematic object have invMass = 0.f
        const bool isKinematic = (invMass0 == 0.0f) && (!PxgNodeIsStaticBody(rigidId));

        const float4 bfP = Ptr<float4>(prepDesc[0].body2WorldPool + uint64_t(idx) * 32UL + 16UL)[0];
        const float3 bodyFrame0p = float3(bfP.x, bfP.y, bfP.z);

        const float3 ra = float3(p.x - bodyFrame0p.x, p.y - bodyFrame0p.y, p.z - bodyFrame0p.z);
        const float3 raXn = dpCross3(ra, normal);
        const float3 raXF0 = dpCross3(ra, t0);
        const float3 raXF1 = dpCross3(ra, t1);

        const float3 raXnSqrtInertia = duMat33MulVec(invSqrtInertia0, raXn);
        const float resp0 = (dpDot3(raXnSqrtInertia, raXnSqrtInertia)) + invMass0;

        const float3 raXF0SqrtInertia = duMat33MulVec(invSqrtInertia0, raXF0);
        const float3 raXF1SqrtInertia = duMat33MulVec(invSqrtInertia0, raXF1);

        const float respF0 = (dpDot3(raXF0SqrtInertia, raXF0SqrtInertia)) + invMass0;
        const float respF1 = (dpDot3(raXF1SqrtInertia, raXF1SqrtInertia)) + invMass0;

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
    }

    const float error = pen + dpDot3(delta, normal);

    Ptr<float4>(block + PXG_RDBC_normal_errorW)[lane] = float4(normal.x, normal.y, normal.z, error);
    Ptr<float4>(block + PXG_RDBC_raXn_resp)[lane] = raXn_resp;
    Ptr<float4>(block + PXG_RDBC_raXnF0_resp)[lane] = raXnF0_resp;
    Ptr<float4>(block + PXG_RDBC_raXnF1_resp)[lane] = raXnF1_resp;
    Ptr<float4>(block + PXG_RDBC_fricTan0_invMass0)[lane] = float4(t0.x, t0.y, t0.z, invMass0);
    Ptr<float>(block + PXG_RDBC_maxPenBiasClamp)[lane] = dpSelectMax(penBiasClampRigid, penBiasClampFEM);
}

// ===========================================================================
// finalizeVertexVelocity<AverageOp> (deformableUtils.cuh).
// The averaging operator is a runtime selector here:
//   PXG_VEL_AVG_LINEAR_BLEND    LinearBlendVelocityAveraging(nbPosIters), scale from
//                               linearBlendVelocityAveragingScale (softbody).
//   PXG_VEL_AVG_MAGNITUDE_CLAMP MagnitudeClampVelocityAveraging (cloth); scale unused.
// Returns whether the vertex is awake; `vel` is updated in place (left as it was when the
// averaging and the settle damping both skip it, and for a kinematic vertex).
// ===========================================================================
public static const uint PXG_VEL_AVG_LINEAR_BLEND = 0u;
public static const uint PXG_VEL_AVG_MAGNITUDE_CLAMP = 1u;

// LinearBlendVelocityAveraging::scale = PxMin(0.9f, 2.0f * PxSqrt(1.0f / PxReal(nbPosIters))).
public float linearBlendVelocityAveragingScale(uint nbPosIters)
{
    return dpSelectMin(0.9f, 2.0f * sqrt(1.0f / float(nbPosIters)));
}

public bool finalizeVertexVelocity(float4 pos, inout float4 vel, float4 posDelta, float settlingThreshold,
                                   float sleepThreshold, float settlingDamping, float dt, float invDt,
                                   bool alwaysRunVelocityAveraging, uint averageOp, float averageScale)
{
    if (pos.w == 0.0f)
        return false;

    const float settleTolerance = settlingThreshold * dt;
    const float tolerance = sleepThreshold * dt;
    const float sleepDamping = 1.0f - dpSelectMin(1.0f, settlingDamping * dt);

    const float3 tDelta = float3(posDelta.x, posDelta.y, posDelta.z);
    const float magSq = dpDot3(tDelta, tDelta);

    bool awake;
    float velocityScaling = 1.0f;
    if (magSq < settleTolerance * settleTolerance)
    {
        awake = magSq >= tolerance * tolerance;
        velocityScaling = sleepDamping;
    }
    else
    {
        awake = true;
    }

    if (alwaysRunVelocityAveraging || velocityScaling != 1.0f)
    {
        float3 tVel = float3(vel.x, vel.y, vel.z);
        if (alwaysRunVelocityAveraging)
        {
            const float3 deltaVel = float3(tDelta.x * invDt, tDelta.y * invDt, tDelta.z * invDt);
            if (averageOp == PXG_VEL_AVG_LINEAR_BLEND)
            {
                // LinearBlendVelocityAveraging: tVel * scale + deltaVel * (1 - scale).
                if (dpDot3(deltaVel, deltaVel) < dpDot3(tVel, tVel))
                {
                    const float oneMinus = 1.0f - averageScale;
                    tVel = float3(tVel.x * averageScale + deltaVel.x * oneMinus,
                                  tVel.y * averageScale + deltaVel.y * oneMinus,
                                  tVel.z * averageScale + deltaVel.z * oneMinus);
                }
            }
            else
            {
                // MagnitudeClampVelocityAveraging: tVel * sqrt(|deltaVel|^2 / |tVel|^2).
                const float deltaVelMagSqr = dpDot3(deltaVel, deltaVel);
                const float velMagSqr = dpDot3(tVel, tVel);
                if (deltaVelMagSqr < velMagSqr)
                {
                    const float s = sqrt(deltaVelMagSqr / velMagSqr);
                    tVel = float3(tVel.x * s, tVel.y * s, tVel.z * s);
                }
            }
        }
        vel = float4(velocityScaling * tVel.x, velocityScaling * tVel.y, velocityScaling * tVel.z, vel.w);
    }
    return awake;
}
