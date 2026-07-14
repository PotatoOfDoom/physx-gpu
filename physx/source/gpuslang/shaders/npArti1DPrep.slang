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

// npArti1DPrep.slang
//
// Slang port of the articulation 1D (joint) constraint prep of
// gpusolver/src/CUDA/artiConstraintPrep2.cu:
//   - preprocessRowsBlock (L1396-1421): per-thread stable insertion sort of
//     row indices on rowParameters[].solveHint (strict <; equal hints keep
//     the original row order);
//   - setUpArti1DConstraintBlock, PGS variant (L1424-1540), with
//     intializeBlock1D (jointConstraintBlockPrep.cuh L244-298) and
//     Dy::computeJointSpeedPGS (DyCpuGpu1dConstraint.h L90-130).
// The spatial-response body helpers (ExtBodyQ image, createImpulseResponse-
// Vector, getImpulseResponse = 6x6 column product, projectVelocity) are the
// same forms established by npArtiExtBody.slang (tranches 179-181).
// queryReduced1dConstraintSolverConstantsPGS and the Px1D/DY_SC flag
// constants come from pxgSlangPrepCommon (ported for the rigid joint prep).
//
// Exact-order notes replicated from the CUDA:
//   - cmod.ang0Writeback is written BEFORE the degenerate-row skip
//     (unitResponse <= DY_ARTICULATION_MIN_RESPONSE -> continue), so a
//     skipped row leaves its cang0 in output slot outCount until/unless a
//     later successful row overwrites that slot.
//   - the caller's pre-intializeBlock1D SPRING/ACCELERATION_SPRING flag
//     raises (cu L1524-1528) are dead: intializeBlock1D resets the flags
//     word to 0 and re-raises the same two bits; eOUTPUT_FORCE is raised
//     by the caller AFTER the call.  The port writes the final value.
//   - the cfm = PxMax(b0.cfm, b1.cfm) computed by the PGS variant is passed
//     into intializeBlock1D's cfm parameter, which is UNUSED in its body --
//     no observable output depends on it in the PGS path.
//   - driveScale = fminf(dt, 1) gated on eHAS_DRIVE_LIMIT &&
//     eDRIVE_LIMITS_ARE_FORCES, applied to min/maxImpulse (contrast the
//     rigid path, which uses computeMinMaxImpulseOrForceAsImpulse without
//     the fminf clamp).
//   - b0IsRigidDynamic = (linkIndex == PxSolverConstraintDesc::RIGID_BODY):
//     TRUE for every rigid-branch body (dynamic, kinematic or static),
//     i.e. exactly !isArticulation.
// All outputs bitwise (float arith NoContraction-stamped); uninitialized-
// in-CUDA output lanes are zero-pinned by fixture zero-init on both sides.

import pxgSlangCommon;
import pxgSlangPrepCommon;

static const uint MAX_ROWS_Q = 20u;  // Dy::MAX_CONSTRAINT_ROWS
static const float DY_ARTI_MIN_RESPONSE_Q = 1e-12f;

struct J1In
{
    uint4 bMeta;        // b0 isArticulation, b0 isKinematic, b1 isArticulation, b1 isKinematic
    float4 b0resp[9];
    float4 b0velTop;    // w = penBiasClamp (unused here)
    float4 b0velBot;    // w = maxImpulse (unused here)
    float4 b0q;         // unused by the 1D path; kept for the shared body image
    float4 b0p;         // w = cfm
    float4 b1resp[9];
    float4 b1velTop;
    float4 b1velBot;
    float4 b1q;
    float4 b1p;
    uint4 cfg;          // numRows, constraintData->mFlags, 0, 0
    float4 invMassScale;  // lin0X_ang0Y_lin1Z_ang1W
    float4 scal;        // dt, recipdt, 0, 0
    float4 rvel[80];    // 4 per row: linear0XYZ_geometricErrorW, linear1XYZ_minImpulseW,
                        //   angular0XYZ_velocityTargetW, angular1XYZ_maxImpulseW
    uint4 rparU[20];    // x = flags, y = solveHint
    float4 rparF[20];   // springStiffness, springDamping, restitution, velocityThreshold
}

struct J1Out
{
    uint4 hdr;          // x = outCount (returned row count)
    uint4 sorted[5];    // sortedRowIndices, 20 lanes
    uint4 modFlags[5];  // cmod.flags per output slot, 20 lanes
    float4 conOut[100]; // 5 per output slot: lin0XYZ_minImpulse, lin1XYZ_maxImpulse,
                        //   ang0XYZ_resp0, ang1XYZ_resp1, (initJointSpeed, 0, 0, 0)
    float4 modOut[40];  // 2 per output slot: (ang0Writeback, coeff0),
                        //   (coeff1, appliedForce, residual, 0)
    float4 artiR[60];   // 3 per successful row: (dRALin, dRAAng.x),
                        //   (dRAAng.yz, dRBLin.xy), (dRBLin.z, dRBAng)
}

struct J1Params { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<J1Params> gParams;

RWStructuredBuffer<J1In> gIn;    // binding 0
RWStructuredBuffer<J1Out> gOut;  // binding 1

float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

float pxMaxQ(float a, float b) { return (a < b) ? b : a; }

// column[j][i] flattened at j*6+i over resp[9] float4 lanes
float respAt(float4 m[9], uint j, uint i)
{
    const uint f = j * 6u + i;
    return m[f / 4u][f & 3u];
}

// PxSpatialMatrix::multiplyInertia (PxSpatialMatrix.h L278-284)
float3 mulInertiaQ(float4 resp[9], float3 v)
{
    return float3(
        respAt(resp, 3u, 0u) * v.x + respAt(resp, 3u, 1u) * v.y + respAt(resp, 3u, 2u) * v.z,
        respAt(resp, 4u, 0u) * v.x + respAt(resp, 4u, 1u) * v.y + respAt(resp, 4u, 2u) * v.z,
        respAt(resp, 5u, 0u) * v.x + respAt(resp, 5u, 1u) * v.y + respAt(resp, 5u, 2u) * v.z);
}

// PxSpatialMatrix::operator* (PxSpatialMatrix.h L57-77): result[i] =
// sum over j (ascending) of column[j][i] * st[j]
void spatialMulQ(float4 resp[9], float3 vTop, float3 vBot, out float3 oTop, out float3 oBot)
{
    float st[6];
    st[0] = vTop.x; st[1] = vTop.y; st[2] = vTop.z;
    st[3] = vBot.x; st[4] = vBot.y; st[5] = vBot.z;
    float result[6];
    for (uint i = 0u; i < 6u; ++i)
    {
        result[i] = 0.0f;
        for (uint j = 0u; j < 6u; ++j)
            result[i] += respAt(resp, j, i) * st[j];
    }
    oTop = float3(result[0], result[1], result[2]);
    oBot = float3(result[3], result[4], result[5]);
}

struct ExtBodyQ
{
    float4 resp[9];
    float3 velTop; float3 velBot;
    float cfm;
    uint isArticulation, isKinematic;
};

ExtBodyQ loadBody(uint isArt, uint isKin, float4 resp[9], float4 velTop, float4 velBot, float4 p)
{
    ExtBodyQ b;
    for (uint f = 0u; f < 9u; ++f)
        b.resp[f] = resp[f];
    b.velTop = velTop.xyz;
    b.velBot = velBot.xyz;
    b.cfm = p.w;
    b.isArticulation = isArt;
    b.isKinematic = isKin;
    return b;
}

// createImpulseResponseVector (cu L136-150)
void cirvQ(ExtBodyQ b, float3 top, float3 bot, out float3 oTop, out float3 oBot)
{
    if (b.isArticulation != 0u || b.isKinematic != 0u)
    {
        oTop = top;
        oBot = bot;
    }
    else
    {
        oTop = mulInertiaQ(b.resp, top);
        oBot = bot;
    }
}

// projectVelocity (cu L152-155): velocity.dot(v) = top.top + bottom.bottom
float projVelQ(ExtBodyQ b, float3 vTop, float3 vBot)
{
    return dot3Q(b.velTop, vTop) + dot3Q(b.velBot, vBot);
}

void packArtiQ(inout float4 arr[60], uint slot, float3 dALin, float3 dAAng, float3 dBLin,
               float3 dBAng)
{
    arr[slot * 3u + 0u] = float4(dALin, dAAng.x);
    arr[slot * 3u + 1u] = float4(dAAng.y, dAAng.z, dBLin.x, dBLin.y);
    arr[slot * 3u + 2u] = float4(dBLin.z, dBAng.x, dBAng.y, dBAng.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void a1dPGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const J1In s = gIn[tid];

    J1Out o;
    o.hdr = uint4(0, 0, 0, 0);
    for (uint z = 0u; z < 5u; ++z) { o.sorted[z] = uint4(0, 0, 0, 0); o.modFlags[z] = uint4(0, 0, 0, 0); }
    for (uint z = 0u; z < 100u; ++z) o.conOut[z] = float4(0, 0, 0, 0);
    for (uint z = 0u; z < 40u; ++z) o.modOut[z] = float4(0, 0, 0, 0);
    for (uint z = 0u; z < 60u; ++z) o.artiR[z] = float4(0, 0, 0, 0);

    const ExtBodyQ b0 = loadBody(s.bMeta.x, s.bMeta.y, s.b0resp, s.b0velTop, s.b0velBot, s.b0p);
    const ExtBodyQ b1 = loadBody(s.bMeta.z, s.bMeta.w, s.b1resp, s.b1velTop, s.b1velBot, s.b1p);

    const uint numRows = min(s.cfg.x, MAX_ROWS_Q);
    const uint cdFlags = s.cfg.y;
    const float dt = s.scal.x;
    const float recipdt = s.scal.y;
    const float4 ims = s.invMassScale;

    // ---- preprocessRowsBlock (cu L1396-1421): stable insertion sort on solveHint ----
    uint sortedRows[MAX_ROWS_Q];
    for (uint i = 0u; i < numRows; ++i)
    {
        const uint hintI = s.rparU[i].y;
        uint j = i;
        for (; j > 0u && hintI < s.rparU[sortedRows[j - 1u]].y; --j)
            sortedRows[j] = sortedRows[j - 1u];
        sortedRows[j] = i;
    }
    for (uint i = 0u; i < numRows; ++i)
        o.sorted[i / 4u][i & 3u] = sortedRows[i];

    // ---- setUpArti1DConstraintBlock, PGS (cu L1424-1540) ----
    const float erp = 1.0f;
    uint outCount = 0u;

    // cfm = PxMax(b0.cfm, b1.cfm) is computed by the CUDA (L1449) and passed
    // into intializeBlock1D, whose body never reads it -- no output effect.
    const float cfmDead = pxMaxQ(b0.cfm, b1.cfm);

    for (uint i = 0u; i < numRows; ++i)
    {
        const uint index = sortedRows[i];
        const uint rpFlags = s.rparU[index].x;

        const float driveScale =
            ((rpFlags & PX1D_FLAG_HAS_DRIVE_LIMIT) != 0u &&
             (cdFlags & PX_CONSTRAINT_FLAG_DRIVE_LIMITS_ARE_FORCES) != 0u)
                ? min(dt, 1.0f)  // fminf
                : 1.0f;

        const float4 lin0XYZ_geomError = s.rvel[index * 4u + 0u];
        const float4 lin1XYZ_minImpulse = s.rvel[index * 4u + 1u];
        const float4 ang0XYZ_velTarget = s.rvel[index * 4u + 2u];
        const float4 ang1XYZ_maxImpulse = s.rvel[index * 4u + 3u];

        const float3 clin0 = lin0XYZ_geomError.xyz;
        const float3 clin1 = lin1XYZ_minImpulse.xyz;
        const float3 cang0 = ang0XYZ_velTarget.xyz;
        const float3 cang1 = ang1XYZ_maxImpulse.xyz;

        const float minImpulse = lin1XYZ_minImpulse.w * driveScale;
        const float maxImpulse = ang1XYZ_maxImpulse.w * driveScale;

        // response0 = cirv(b0, (cang0, clin0)); response1 = cirv(b1, (-cang1, -clin1))
        float3 r0Top, r0Bot, r1Top, r1Bot;
        cirvQ(b0, cang0, clin0, r0Top, r0Bot);
        cirvQ(b1, -cang1, -clin1, r1Top, r1Bot);

        // deltaV = getImpulseResponse(b, (clin, cang).scale(lin, ang))
        float3 dV0Top, dV0Bot, dV1Top, dV1Bot;
        spatialMulQ(b0.resp, clin0 * ims.x, cang0 * ims.y, dV0Top, dV0Bot);
        spatialMulQ(b1.resp, -clin1 * ims.z, -cang1 * ims.w, dV1Top, dV1Bot);

        const float resp0 = dot3Q(dV0Top, r0Top) + dot3Q(dV0Bot, r0Bot);
        const float resp1 = dot3Q(dV1Top, r1Top) + dot3Q(dV1Bot, r1Bot);
        const float unitResponse = resp0 + resp1;

        // ang0Writeback lands in slot outCount BEFORE the degenerate skip
        o.modOut[outCount * 2u + 0u] =
            float4(cang0, o.modOut[outCount * 2u + 0u].w);

        if (unitResponse <= DY_ARTI_MIN_RESPONSE_Q)
            continue;  // degenerate constraint

        packArtiQ(o.artiR, outCount, dV0Bot, dV0Top, dV1Bot, dV1Top);

        const bool needNormalVel = (rpFlags & PX1D_FLAG_RESTITUTION) != 0u ||
                                   ((rpFlags & PX1D_FLAG_SPRING) != 0u &&
                                    (rpFlags & PX1D_FLAG_ACCELERATION_SPRING) != 0u);

        // linkIndex == RIGID_BODY (0xffff) for every rigid-branch body
        const bool isRigid0 = b0.isArticulation == 0u;
        const bool isRigid1 = b1.isArticulation == 0u;

        float jointSpeedForRestitutionBounce = 0.0f;
        float initJointSpeed = 0.0f;
        if (needNormalVel || isRigid0 || isRigid1)
        {
            const float vel0 = projVelQ(b0, cang0, clin0);
            const float vel1 = projVelQ(b1, cang1, clin1);
            // Dy::computeJointSpeedPGS (DyCpuGpu1dConstraint.h L90-130)
            initJointSpeed = 0.0f;
            if (isRigid0)
                initJointSpeed = vel0;
            else if (isRigid1)
                initJointSpeed = -vel1;
            jointSpeedForRestitutionBounce = vel0 - vel1;
        }

        // ---- intializeBlock1D (jointConstraintBlockPrep.cuh L244-298) ----
        const float geomError = lin0XYZ_geomError.w;
        const float velTarget = ang0XYZ_velTarget.w;
        float coeff0, coeff1;
        queryReduced1dConstraintSolverConstantsPGS(rpFlags, s.rparF[index].x, s.rparF[index].y,
                                                   s.rparF[index].z, s.rparF[index].w, geomError,
                                                   velTarget, jointSpeedForRestitutionBounce, erp,
                                                   dt, recipdt, coeff0, coeff1);

        // _linear0 = response0.bottom, _linear1 = -response1.bottom,
        // _angular0 = response0.top, _angular1 = -response1.top
        o.conOut[outCount * 5u + 0u] = float4(r0Bot, minImpulse);
        o.conOut[outCount * 5u + 1u] = float4(-r1Bot, maxImpulse);
        o.conOut[outCount * 5u + 2u] = float4(r0Top, resp0);
        o.conOut[outCount * 5u + 3u] = float4(-r1Top, resp1);
        o.conOut[outCount * 5u + 4u] = float4(initJointSpeed, 0.0f, 0.0f, 0.0f);

        o.modOut[outCount * 2u + 0u] = float4(cang0, coeff0);
        o.modOut[outCount * 2u + 1u] = float4(coeff1, 0.0f, 0.0f, 0.0f);  // appliedForce/residual = 0

        // final flags: intializeBlock1D resets to 0 and raises SPRING/ACCEL;
        // the caller raises OUTPUT_FORCE afterwards (pre-call raises are dead)
        uint rowFlags = 0u;
        if ((rpFlags & PX1D_FLAG_SPRING) != 0u)
            rowFlags |= DY_SC_FLAG_SPRING;
        if ((rpFlags & PX1D_FLAG_ACCELERATION_SPRING) != 0u)
            rowFlags |= DY_SC_FLAG_ACCELERATION_SPRING;
        if ((rpFlags & PX1D_FLAG_OUTPUT_FORCE) != 0u)
            rowFlags |= DY_SC_FLAG_OUTPUT_FORCE;
        o.modFlags[outCount / 4u][outCount & 3u] = rowFlags;

        outCount += 1u;
    }

    o.hdr.x = outCount;
    gOut[tid] = o;
}
