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

// npArtiJointPrep.slang
//
// P8.103: artiTGSJointConstraintBlockPrepareParallelLaunch
// (artiConstraintPrep2.cu:1964 -> artiJointConstraintBlockPrepare :1861 ->
//  setupArtiSolverConstraintBlockGPU TGS :1771 -> preprocessRowsBlock :1396
//  + setUpArti1DConstraintBlock TGS :1594 + intializeTGSBlock1D :1545).
//
// The row MATH was Phase-1-ported and swept as npArti1DPrep.slang's a1dTGS
// battery (abstracted flat inputs); this module is the lift onto the engine
// block layouts over the P8.102a body layer (AcpBody / acp*), plus the
// header fill and the desc-walking wrapper.  The scalar family
// (computeJointSpeedTGS, computeMaxBiasVelocityTGS,
// queryReduced1dConstraintSolverConstantsTGS (hoisted in P8.103),
// computeMinMaxImpulseOrForceAsImpulse, raiseInternalFlagsTGS) is shared
// through pxgSlangPrepCommon.
//
// ===========================================================================
// TRAPS (each carries a sweep mutant unless marked)
// ===========================================================================
//  1. **The batch guard sums numArti1d + static + SELF batches**, and the
//     batch index is offset by prepDesc->numBatches.
//  2. **remappedBodyA/BIndex only for articulation node indices** (as in the
//     contact prep).
//  3. **The row pools index differently**: blockJointPrepPool by
//     descIndexBatch, blockJointPrepPool0/1 by descIndexBatch *
//     MAX_CONSTRAINT_ROWS (20).
//  4. **writeBackOffset comes from solverConstantData[
//     artiConstraintUniqueIndices[mStartPartitionIndex + lane]]
//     .mConstraintWriteBackIndex** -- a per-LANE two-hop indirection.
//  5. **params.dt is sharedDesc->stepDt in TGS** (fillJointBlockParams
//     :1853), so BOTH break impulses scale by the STEP dt, not the sim dt.
//  6. **The header sign asymmetry**: rAWorld_invMass0D0.w = -ims.x,
//     rBWorld_invMass1D1.w = +ims.z, invInertiaScale0 = -ims.y,
//     invInertiaScale1 = +ims.w.
//  7. **rowCounts is the SETUP RETURN** (degenerate rows skipped), packed as
//     uchar4(rowCounts, breakable, 0, 0); breakable tests both break forces
//     against PX_MAX_F32.
//  8. **erp = 0.5f * biasCoefficient**; a degenerate row (unitResponse <=
//     1e-12) writes NOTHING in the TGS variant (the PGS twin writes
//     ang0Writeback first).
//  9. **computeJointSpeedTGS runs UNGATED and keys on isKinematic** (the PGS
//     twin gates on need and keys on rigid-dynamic).
// 10. **computeMaxBiasVelocityTGS gets isExtendedConstraint = TRUE** (the
//     rigid joint path passes false).
// 11. **angularErrorScale = eANGULAR_CONSTRAINT ? 1 : 0 WITHOUT zeroing the
//     angular Jacobian lanes** (the rigid TGS joint prep zeroes them).
// 12. min/max impulses via computeMinMaxImpulseOrForceAsImpulse with the
//     SIM dt.
// 13. **The insertion sort by solveHint is stable** (strict <), per lane,
//     over the UNSORTED parameter rows.
// 14. One PxgArticulationBlockResponse appended per NON-degenerate row from
//     batch.mArticulationResponseIndex.
// ===========================================================================

import pxgSlangCommon;
import pxgSlangSpatialMatrix;   // AbSpatialVector / asmSVec (imports are not transitive)
import pxgSlangPrepCommon;
import pxgSlangSolverBlocks;
import pxgSlangPrepDesc;
import pxgSlangSolverDesc;
import npArtiConstraintPrep;   // AcpBody + the acp* body layer

static const float AJP_DY_ARTI_MIN_RESPONSE = 1e-12f;   // DY_ARTICULATION_MIN_RESPONSE

struct AjpParams
{
    uint64_t prepDesc;    // PxgConstraintPrepareDesc*
    uint64_t sharedDesc;  // PxgSolverSharedDesc<IterativeSolveDataTGS>*
}

[[vk::push_constant]]
ConstantBuffer<AjpParams> gAjp;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiTGSJointConstraintBlockPrepareParallelLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                                      uint3 groupId : SV_GroupID)
{
    const uint blockStride = 64u / 32u;
    const uint warpIndex = groupId.x * blockStride + groupThreadId.x / 32u;
    const uint lane = groupThreadId.x & 31u;

    Ptr<PxgConstraintPrepareDescGpu> prepDesc =
        Ptr<PxgConstraintPrepareDescGpu>(gAjp.prepDesc);
    Ptr<PxgSolverSharedDescTGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescTGSGpu>(gAjp.sharedDesc);

    // trap 1
    const uint totalArti1dBatches = prepDesc[0].numArti1dConstraintBatches
        + prepDesc[0].numArtiStatic1dConstraintBatches
        + prepDesc[0].numArtiSelf1dConstraintBatches;
    const uint i = warpIndex;
    if (i >= totalArti1dBatches)
        return;

    Ptr<uint> batchIndices = Ptr<uint>(prepDesc[0].artiJointConstraintBatchIndices);
    const uint batchIndex = batchIndices[i] + prepDesc[0].numBatches;

    Ptr<PxgBlockConstraintBatchGpu> batches =
        Ptr<PxgBlockConstraintBatchGpu>(sharedDesc[0].iter_blockConstraintBatch);
    const uint descIndexBatch = batches[batchIndex].mConstraintBatchIndex;
    const uint descStride = batches[batchIndex].descStride_constraintType & 0xFFFFu;

    if (lane >= descStride)
        return;

    const uint2 nodeIndexA = batches[batchIndex].bodyANodeIndex[lane];
    const uint2 nodeIndexB = batches[batchIndex].bodyBNodeIndex[lane];

    // trap 2
    const uint bodyAIndex = nodeIndexIsArticulation(nodeIndexA)
        ? batches[batchIndex].remappedBodyAIndex[lane] : batches[batchIndex].bodyAIndex[lane];
    const uint bodyBIndex = nodeIndexIsArticulation(nodeIndexB)
        ? batches[batchIndex].remappedBodyBIndex[lane] : batches[batchIndex].bodyBIndex[lane];

    AcpBody b0, b1;
    acpCreateSolverExtBody(nodeIndexA, bodyAIndex, sharedDesc[0].articulations,
                           prepDesc[0].solverBodyDataPool, prepDesc[0].solverBodyTxIDataPool,
                           prepDesc[0].body2WorldPool, b0);
    acpCreateSolverExtBody(nodeIndexB, bodyBIndex, sharedDesc[0].articulations,
                           prepDesc[0].solverBodyDataPool, prepDesc[0].solverBodyTxIDataPool,
                           prepDesc[0].body2WorldPool, b1);

    // trap 3: the three prep pools
    Ptr<PxgBlockConstraint1DData> cdata =
        Ptr<PxgBlockConstraint1DData>(prepDesc[0].blockJointPrepPool);
    Ptr<PxgBlockConstraint1DVelocities> rvel =
        Ptr<PxgBlockConstraint1DVelocities>(prepDesc[0].blockJointPrepPool0);
    Ptr<PxgBlockConstraint1DParameters> rpar =
        Ptr<PxgBlockConstraint1DParameters>(prepDesc[0].blockJointPrepPool1);
    const uint cdi = descIndexBatch;
    const uint rowBase = descIndexBatch * MAX_CONSTRAINT_ROWS;

    Ptr<PxgTGSBlockSolverConstraint1DHeader> headers =
        Ptr<PxgTGSBlockSolverConstraint1DHeader>(sharedDesc[0].iter_blockJointConstraintHeaders);
    Ptr<PxgTGSBlockSolverConstraint1DCon> cons =
        Ptr<PxgTGSBlockSolverConstraint1DCon>(sharedDesc[0].iter_blockJointConstraintRowsCon);
    Ptr<PxgArticulationBlockResponse> responses =
        Ptr<PxgArticulationBlockResponse>(sharedDesc[0].iter_artiResponse);
    const uint hdi = descIndexBatch;
    const uint conBase = batches[batchIndex].startConstraintIndex;
    const uint respBase = batches[batchIndex].mArticulationResponseIndex;

    // trap 4: the two-hop writeback indirection
    Ptr<uint> uniqueIndices = Ptr<uint>(prepDesc[0].artiConstraintUniqueIndices);
    const uint uniqueIndex = uniqueIndices[batches[batchIndex].mStartPartitionIndex + lane];
    Ptr<PxgSolverConstraintManagerConstantsGpu> constants =
        Ptr<PxgSolverConstraintManagerConstantsGpu>(prepDesc[0].solverConstantData);

    // ---- setupArtiSolverConstraintBlockGPU, TGS overload (:1771) ----
    const float stepDt = sharedDesc[0].stepDt;       // params.dt (trap 5)
    const float simDt = sharedDesc[0].dt;            // params.totalDt
    const float recipStepDt = sharedDesc[0].stepInvDtF32;
    const float recipSimDt = sharedDesc[0].invDtF32;
    const float lengthScale = sharedDesc[0].lengthScale;
    const float biasCoefficient = prepDesc[0].biasCoefficient;

    headers[hdi].writeBackOffset[lane] = constants[uniqueIndex].mConstraintWriteBackIndex;

    const float4 ims = cdata[cdi].mInvMassScale[lane];
    const float4 raWorld_linBreakForce = cdata[cdi].mRAWorld_linBreakForce[lane];
    const float4 rbWorld_angBreakForce = cdata[cdi].mRBWorld_AngBreakForce[lane];
    const float linBreakImpulse = raWorld_linBreakForce.w * stepDt;   // trap 5
    const float angBreakForce = rbWorld_angBreakForce.w;
    const float angBreakImpulse = angBreakForce * stepDt;

    // trap 6
    headers[hdi].rAWorld_invMass0D0[lane] =
        float4(raWorld_linBreakForce.xyz, -ims.x);
    headers[hdi].rBWorld_invMass1D1[lane] =
        float4(rbWorld_angBreakForce.xyz, ims.z);
    headers[hdi].angBreakImpulse[lane] = angBreakImpulse;
    headers[hdi].invInertiaScale0[lane] = -ims.y;
    headers[hdi].invInertiaScale1[lane] = ims.w;
    headers[hdi].linBreakImpulse[lane] = linBreakImpulse;

    const uint numRows = cdata[cdi].mNumRows[lane];
    const uint breakable = ((raWorld_linBreakForce.w != PREP_PX_MAX_F32)
                            || (angBreakForce != PREP_PX_MAX_F32)) ? 1u : 0u;

    // ---- preprocessRowsBlock (:1396): stable insertion sort by solveHint ----
    uint sortedRows[MAX_CONSTRAINT_ROWS];
    for (uint r = 0u; r < numRows; ++r)
    {
        const uint hintR = rpar[rowBase + r].solveHint[lane];
        uint j = r;
        for (; j > 0u && hintR < rpar[rowBase + sortedRows[j - 1u]].solveHint[lane]; --j)
            sortedRows[j] = sortedRows[j - 1u];
        sortedRows[j] = r;
    }

    // ---- setUpArti1DConstraintBlock, TGS (:1594) ----
    const float erp = 0.5f * biasCoefficient;   // trap 8
    const float cfm = max(b0.cfm, b1.cfm);
    const float4 imsRow = cdata[cdi].mInvMassScale[lane];

    uint outCount = 0u;
    for (uint r = 0u; r < numRows; ++r)
    {
        const uint index = sortedRows[r];
        const uint rpFlags = rpar[rowBase + index].flags[lane];
        const uint rpSolveHint = rpar[rowBase + index].solveHint[lane];

        const float4 lin0XYZ_geomError = rvel[rowBase + index].linear0XYZ_geometricErrorW[lane];
        const float4 lin1XYZ_minImpulse = rvel[rowBase + index].linear1XYZ_minImpulseW[lane];
        const float4 ang0XYZ_velTarget = rvel[rowBase + index].angular0XYZ_velocityTargetW[lane];
        const float4 ang1XYZ_maxImpulse = rvel[rowBase + index].angular1XYZ_maxImpulseW[lane];

        const float3 clin0 = lin0XYZ_geomError.xyz;
        const float3 clin1 = lin1XYZ_minImpulse.xyz;
        const float3 cang0 = ang0XYZ_velTarget.xyz;
        const float3 cang1 = ang1XYZ_maxImpulse.xyz;

        const AbSpatialVector response0 = acpCreateImpulseResponseVector(b0, asmSVec(cang0, clin0));
        const AbSpatialVector response1 =
            acpCreateImpulseResponseVector(b1, asmSVec(-cang1, -clin1));

        const AbSpatialVector deltaV0 =
            acpGetImpulseResponse(b0, asmSVec(clin0 * imsRow.x, cang0 * imsRow.y));
        const AbSpatialVector deltaV1 =
            acpGetImpulseResponse(b1, asmSVec(-clin1 * imsRow.z, -cang1 * imsRow.w));

        const float resp0 = dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
        const float resp1 = dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);
        const float unitResponse = resp0 + resp1;

        if (unitResponse <= AJP_DY_ARTI_MIN_RESPONSE)
            continue;   // trap 8: degenerate -- NOTHING written in TGS

        const uint c = conBase + outCount;
        cons[c].resp0[lane] = resp0;
        cons[c].resp1[lane] = resp1;

        const uint rr = respBase + outCount;   // trap 14
        responses[rr].deltaRALin_x[lane] = deltaV0.bottom.x;
        responses[rr].deltaRALin_y[lane] = deltaV0.bottom.y;
        responses[rr].deltaRALin_z[lane] = deltaV0.bottom.z;
        responses[rr].deltaRAAng_x[lane] = deltaV0.top.x;
        responses[rr].deltaRAAng_y[lane] = deltaV0.top.y;
        responses[rr].deltaRAAng_z[lane] = deltaV0.top.z;
        responses[rr].deltaRBLin_x[lane] = deltaV1.bottom.x;
        responses[rr].deltaRBLin_y[lane] = deltaV1.bottom.y;
        responses[rr].deltaRBLin_z[lane] = deltaV1.bottom.z;
        responses[rr].deltaRBAng_x[lane] = deltaV1.top.x;
        responses[rr].deltaRBAng_y[lane] = deltaV1.top.y;
        responses[rr].deltaRBAng_z[lane] = deltaV1.top.z;

        // trap 9: ungated, keyed on isKinematic
        float jointSpeedForRestitutionBounce;
        float initJointSpeed;
        {
            const float vel0 = acpProjectVelocity(b0, asmSVec(cang0, clin0));
            const float vel1 = acpProjectVelocity(b1, asmSVec(cang1, clin1));
            computeJointSpeedTGS(vel0, b0.isKinematic, vel1, b1.isKinematic,
                                 jointSpeedForRestitutionBounce, initJointSpeed);
        }

        // ---- intializeTGSBlock1D (:1545) ----
        const float geometricError = lin0XYZ_geomError.w;
        const float velocityTarget = ang0XYZ_velTarget.w;
        const float stiffness = rpar[rowBase + index].springStiffness_restitution[lane];
        const float damping = rpar[rowBase + index].springDamping_velocityThreshold[lane];
        const float restitution = stiffness;             // the union's other face
        const float bounceVelocityThreshold = damping;

        // trap 10: isExtendedConstraint = true
        const float maxBiasVelocity = computeMaxBiasVelocityTGS(
            rpFlags, jointSpeedForRestitutionBounce, bounceVelocityThreshold, restitution,
            geometricError, true, lengthScale, recipSimDt);

        float coeff0, coeff1, coeff2, coeff3;
        queryReduced1dConstraintSolverConstantsTGS(
            rpFlags, stiffness, damping, restitution, bounceVelocityThreshold, geometricError,
            velocityTarget, jointSpeedForRestitutionBounce, initJointSpeed, erp, stepDt,
            recipStepDt, coeff0, coeff1, coeff2, coeff3);

        cons[c].lin0XYZ_initBiasOrCoeff0[lane] = float4(response0.bottom, coeff0);
        cons[c].lin1XYZ_biasScaleOrCoeff1[lane] = float4(-response1.bottom, coeff1);
        cons[c].ang0XYZ_velMultiplierOrCoeff2[lane] = float4(response0.top, coeff2);
        cons[c].ang1XYZ_velTargetOrCoeff3[lane] = float4(-response1.top, coeff3);
        cons[c].geometricError[lane] = geometricError;
        cons[c].maxBias[lane] = maxBiasVelocity;
        cons[c].appliedForce[lane] = 0.0f;

        // trap 11: the Jacobian lanes stay -- only the scale toggles
        cons[c].angularErrorScale[lane] =
            (rpFlags & PX1D_FLAG_ANGULAR_CONSTRAINT) != 0u ? 1.0f : 0.0f;

        // trap 12: SIM dt
        const bool hasDriveLimit = (rpFlags & PX1D_FLAG_HAS_DRIVE_LIMIT) != 0u;
        const bool driveLimitsAreForces =
            (cdata[cdi].mFlags[lane] & PX_CONSTRAINT_FLAG_DRIVE_LIMITS_ARE_FORCES) != 0u;
        float minImpulse, maxImpulse;
        computeMinMaxImpulseOrForceAsImpulse(lin1XYZ_minImpulse.w, ang1XYZ_maxImpulse.w,
                                             hasDriveLimit, driveLimitsAreForces, simDt,
                                             minImpulse, maxImpulse);
        cons[c].minImpulse[lane] = minImpulse;
        cons[c].maxImpulse[lane] = maxImpulse;

        uint rowFlags = 0u;
        raiseInternalFlagsTGS(rpFlags, rpSolveHint, rowFlags);
        cons[c].flags[lane] = rowFlags;

        outCount++;
    }

    // trap 7: the SETUP RETURN, packed as uchar4(rowCounts, breakable, 0, 0)
    headers[hdi].rowCounts_breakable_orthoAxisCount[lane] = outCount | (breakable << 8);
    headers[hdi].cfm[lane] = cfm;
}
