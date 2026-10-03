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


// npArtiConstraintPrep.slang
//
// P8.102a: the PxgSolverExtBody2 layer of the articulation constraint prep
// (artiConstraintPrep2.cu:55-165) — the body abstraction every ext contact
// and 1D row of artiTGSContactConstraintBlockPrepareLaunch is built on.
//
//   acpCreateSolverExtBody   createPxgSolverExtBody (:55) — BOTH arms
//   acpCreateImpulseResponseVector                  (:136)
//   acpProjectVelocity / acpProjectAngular          (:152/:157)
//   acpGetImpulseResponse                           (:162)
//
// ===========================================================================
// TRAPS (each carries a mutant in the sweep unless marked)
// ===========================================================================
//
//  1. **The articulation arm indexes the RECORD pool by solverBodyIndex,
//     the link arrays by nodeIndex.articulationLinkId()** (the HIGH word
//     >> 1) — two different indices from one PxNodeIndex.
//  2. **linkBody2Worlds is a PxTransform array — 28-byte stride** (q at 0,
//     p at 16); spatialResponseMatrixW is a 144-byte PxSpatialMatrix array
//     (36 floats, column[i][j] at i*6+j); motionVelocities is a 24-byte
//     UnAlignedSpatialVector array (top=angular first); cfms is a float
//     array; links is the 80-byte PxgArticulationLink array with
//     penBiasClamp at +12 and offsetSlop at +60.
//  3. **The articulation arm's maxImpulse is PX_MAX_F32** (the engine's own
//     TODO), NOT the record's; isKinematic is false; cfm comes from the
//     record's cfms[linkID].
//  4. **A STATIC rigid body** (nodeIndex.isStaticBody(): mID == ~0) skips
//     the PxgSolverBodyData velocity/penBias/invMass loads entirely —
//     velocity 0, penBiasClamp -PX_MAX_F32, invMass 0, offsetSlop 0 — but
//     STILL loads bodyFrames[solverBodyIndex] for body2World and STILL
//     reads solverBody.flags and solverBody.maxImpulse.
//  5. **Static OR kinematic zeroes the whole 6x6 response**; only a live
//     dynamic builds it via PxSpatialMatrix::initialize(sqrtInvInertia,
//     invMass) — mass on the top-right diagonal, the sqrt inertia in the
//     bottom-left 3x3, everything else zero (the engine's own layout).
//  6. **The rigid body2World drops the w components** of the
//     PxAlignedTransform's float4 p; the quat carries over.
//  7. **createImpulseResponseVector**: articulation OR kinematic returns
//     the impulse UNCHANGED; a dynamic rigid maps the TOP through
//     multiplyInertia (the bottom-left 3x3 of the response — the
//     sqrtInertia trick) and keeps the bottom.
//  8. projectVelocity is the PLAIN dot (top.top + bottom.bottom), NOT the
//     dual product of the articulation solver.
// ===========================================================================

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDesc;
import pxgSlangPrepCommon;   // PX_RIGID_BODY_FLAG_KINEMATIC + the patch helpers
import pxgSlangSolverBlocks; // PxgTGSBlockSolver* + PxgArticulationBlockResponse
import pxgSlangPrepDesc;     // PxgConstraintPrepareDescGpu + PxgBlockConstraintBatchGpu
import pxgSlangSolverDesc;   // PxgSolverSharedDescTGSGpu

public static const float ACP_MAX_F32 = 3.402823466e+38f;

// PxgSolverExtBody2, register form
public struct AcpBody
{
    public AbMatrix6x6 response;        // mSpatialResponse
    public AbSpatialVector velocity;    // top = angular, bottom = linear
    public float3 b2wP;                 // body2World
    public float4 b2wQ;
    public float penBiasClamp;
    public float maxImpulse;
    public uint linkIndex;              // PxSolverConstraintDesc::RIGID_BODY for rigids
    public bool isKinematic;
    public uint bodyIndex;
    public uint2 islandNodeIndex;
    public float cfm;
    public float offsetSlop;
}

public static const uint ACP_RIGID_BODY_LINK = 0xffffu;   // PxSolverConstraintDesc::RIGID_BODY

// PxSpatialMatrix from a RECORD array (144-byte stride, 36 floats)
public AbMatrix6x6 acpLoad6x6Record(uint64_t base, uint index)
{
    Ptr<float> f = Ptr<float>(base + uint64_t(index) * 144u);
    AbMatrix6x6 m;
    for (uint k = 0u; k < 36u; ++k)
        m.e[k] = f[k];
    return m;
}

// createPxgSolverExtBody (artiConstraintPrep2.cu:55)
public void acpCreateSolverExtBody(uint2 nodeIndex, uint solverBodyIndex,
                                   uint64_t articulations, uint64_t solverBodyData,
                                   uint64_t txIDatas, uint64_t bodyFrames,
                                   out AcpBody b)
{
    if (nodeIndexIsArticulation(nodeIndex))
    {
        const uint linkID = nodeIndexArticulationLinkId(nodeIndex);
        const uint64_t a = articulations + uint64_t(solverBodyIndex) * PXG_ARTI_STRIDE;

        b.bodyIndex = solverBodyIndex;
        b.linkIndex = linkID;

        // linkBody2Worlds: PxTransform (28 B) -- q at 0, p at 16
        {
            Ptr<float> t = Ptr<float>(artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS)
                                      + uint64_t(linkID) * 28u);
            b.b2wQ = float4(t[0], t[1], t[2], t[3]);
            b.b2wP = float3(t[4], t[5], t[6]);
        }

        b.response = acpLoad6x6Record(artiPtrField(a, PXG_ARTI_SPATIAL_RESPONSE_MATRIX_W), linkID);
        b.isKinematic = false;
        b.islandNodeIndex = nodeIndex;

        // links[linkID].initialAngVelXYZ_penBiasClamp.w / .offsetSlop
        {
            const uint64_t link = artiPtrField(a, PXG_ARTI_LINKS)
                                + uint64_t(linkID) * PXG_ARTI_LINK_SIZE;
            b.penBiasClamp = (Ptr<float>(link + PXG_ARTI_LINKREC_INITIAL_ANGVEL_PENBIAS))[3];
            b.offsetSlop = (Ptr<float>(link))[15];   // offset 60
        }

        b.maxImpulse = ACP_MAX_F32;   // the engine's own TODO

        // motionVelocities[linkID]: 24-byte UnAlignedSpatialVector, top first
        {
            Ptr<float> v = Ptr<float>(artiPtrField(a, PXG_ARTI_MOTION_VELOCITIES)
                                      + uint64_t(linkID) * 24u);
            b.velocity = asmSVec(float3(v[0], v[1], v[2]), float3(v[3], v[4], v[5]));
        }

        b.cfm = (Ptr<float>(artiPtrField(a, PXG_ARTI_CFMS)))[linkID];
    }
    else
    {
        const bool isStatic = nodeIndexIsStaticBody(nodeIndex);
        Ptr<PxgSolverBodyData> sbd = Ptr<PxgSolverBodyData>(solverBodyData);
        const PxgSolverBodyData solverBody = sbd[solverBodyIndex];
        const bool isKinematic = (solverBody.flags & PX_RIGID_BODY_FLAG_KINEMATIC) != 0u;

        AbSpatialVector velocity = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        float penBiasClamp = -ACP_MAX_F32;
        float invMass = 0.0f;
        Ptr<PxgAlignedTransform> frames = Ptr<PxgAlignedTransform>(bodyFrames);
        const PxgAlignedTransform body2World = frames[solverBodyIndex];
        float offsetSlop = 0.0f;
        if (!isStatic)
        {
            const float4 initialAngVelXYZ_penBiasClamp = solverBody.initialAngVelXYZ_penBiasClamp;
            const float4 initialLinVelXYZ_invMassW = solverBody.initialLinVelXYZ_invMassW;
            penBiasClamp = initialAngVelXYZ_penBiasClamp.w;
            invMass = initialLinVelXYZ_invMassW.w;
            velocity = asmSVec(initialAngVelXYZ_penBiasClamp.xyz, initialLinVelXYZ_invMassW.xyz);
            offsetSlop = solverBody.offsetSlop;
        }

        if (isStatic || isKinematic)
        {
            b.response = abMatrix6x6Zero();
        }
        else
        {
            // PxSpatialMatrix::initialize(sqrtInvInertia, invMass):
            // top-left zero, mass on the top-right diagonal, the sqrt
            // inertia in the bottom-left 3x3, bottom-right zero
            AbMatrix6x6 m = abMatrix6x6Zero();
            m.e[0u * 6u + 3u] = invMass;
            m.e[1u * 6u + 4u] = invMass;
            m.e[2u * 6u + 5u] = invMass;
            const PxgSolverTxIData txi = (Ptr<PxgSolverTxIData>(txIDatas))[solverBodyIndex];
            for (uint c = 0u; c < 3u; ++c)
                for (uint r = 0u; r < 3u; ++r)
                    m.e[(3u + c) * 6u + r] = txi.sqrtInvInertia[c * 3u + r];
            b.response = m;
        }

        b.isKinematic = isKinematic;
        b.linkIndex = ACP_RIGID_BODY_LINK;
        b.bodyIndex = solverBodyIndex;
        b.islandNodeIndex = nodeIndex;
        b.penBiasClamp = penBiasClamp;
        b.b2wP = body2World.p.xyz;   // the w component is dropped
        b.b2wQ = body2World.q;
        b.maxImpulse = solverBody.maxImpulse;
        b.velocity = velocity;
        b.cfm = 0.0f;
        b.offsetSlop = offsetSlop;
    }
}

// createImpulseResponseVector (:136): articulation or kinematic keeps the
// impulse; a dynamic rigid maps the top through multiplyInertia
public AbSpatialVector acpCreateImpulseResponseVector(AcpBody b, AbSpatialVector impulse)
{
    if (nodeIndexIsArticulation(b.islandNodeIndex) || b.isKinematic)
        return impulse;

    // PxSpatialMatrix::multiplyInertia: the bottom-left 3x3, rows dotted
    const float3 t = impulse.top;
    const float3 mt = float3(
        b.response.e[3u * 6u + 0u] * t.x + b.response.e[3u * 6u + 1u] * t.y + b.response.e[3u * 6u + 2u] * t.z,
        b.response.e[4u * 6u + 0u] * t.x + b.response.e[4u * 6u + 1u] * t.y + b.response.e[4u * 6u + 2u] * t.z,
        b.response.e[5u * 6u + 0u] * t.x + b.response.e[5u * 6u + 1u] * t.y + b.response.e[5u * 6u + 2u] * t.z);
    return asmSVec(mt, impulse.bottom);
}

// projectVelocity (:152) -- the PLAIN dot, not the dual product
public float acpProjectVelocity(AcpBody b, AbSpatialVector responseVector)
{
    return dot(b.velocity.top, responseVector.top) + dot(b.velocity.bottom, responseVector.bottom);
}

// projectAngular (:157)
public float acpProjectAngular(AcpBody b, float3 responseVector)
{
    return dot(b.velocity.top, responseVector);
}

// getImpulseResponse (:162)
public AbSpatialVector acpGetImpulseResponse(AcpBody b, AbSpatialVector vector)
{
    return asmM6MulVec(b.response, vector);
}

// ---------------------------------------------------------------------------
// test entry: builds one body per lane from a mixed node table, dumps the
// body fields + the three response products for a fixed probe vector
// ---------------------------------------------------------------------------

struct AcpParams
{
    uint64_t nodes;           // @0  uint2 per lane (PxNodeIndex)
    uint64_t bodyIndices;     // @8  uint per lane (solverBodyIndex)
    uint64_t articulations;   // @16
    uint64_t solverBodyData;  // @24
    uint64_t txIDatas;        // @32
    uint64_t bodyFrames;      // @40
    uint64_t io;              // @48 out: 72-float stride per lane (70 written)
    uint nbLanes;             // @56
    uint pad0;                // @60 -> 64
}

[[vk::push_constant]]
ConstantBuffer<AcpParams> gAcp;

[shader("compute")]
[numthreads(32, 1, 1)]
void acpBodyTest(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint lane = dispatchThreadId.x;
    if (lane >= gAcp.nbLanes)
        return;

    const uint2 node = (Ptr<uint2>(gAcp.nodes))[lane];
    const uint bodyIndex = (Ptr<uint>(gAcp.bodyIndices))[lane];

    AcpBody b;
    acpCreateSolverExtBody(node, bodyIndex, gAcp.articulations, gAcp.solverBodyData,
                           gAcp.txIDatas, gAcp.bodyFrames, b);

    // a per-lane probe vector
    const AbSpatialVector probe = asmSVec(
        float3(0.3f + 0.01f * float(lane), -0.2f, 0.5f),
        float3(0.7f, 0.11f * float(lane), -0.4f));

    const AbSpatialVector respVec = acpCreateImpulseResponseVector(b, probe);
    const AbSpatialVector impResp = acpGetImpulseResponse(b, probe);
    const float projV = acpProjectVelocity(b, probe);
    const float projA = acpProjectAngular(b, probe.top);

    Ptr<float> o = Ptr<float>(gAcp.io + uint64_t(lane) * 72u * 4u);
    uint k = 0u;
    for (uint i = 0u; i < 36u; ++i) o[k++] = b.response.e[i];
    o[k++] = b.velocity.top.x; o[k++] = b.velocity.top.y; o[k++] = b.velocity.top.z;
    o[k++] = b.velocity.bottom.x; o[k++] = b.velocity.bottom.y; o[k++] = b.velocity.bottom.z;
    o[k++] = b.b2wP.x; o[k++] = b.b2wP.y; o[k++] = b.b2wP.z;
    o[k++] = b.b2wQ.x; o[k++] = b.b2wQ.y; o[k++] = b.b2wQ.z; o[k++] = b.b2wQ.w;
    o[k++] = b.penBiasClamp;
    o[k++] = b.maxImpulse;
    o[k++] = asfloat(b.linkIndex);
    o[k++] = b.isKinematic ? 1.0f : 0.0f;
    o[k++] = asfloat(b.bodyIndex);
    o[k++] = b.cfm;
    o[k++] = b.offsetSlop;
    o[k++] = respVec.top.x; o[k++] = respVec.top.y; o[k++] = respVec.top.z;
    o[k++] = respVec.bottom.x; o[k++] = respVec.bottom.y; o[k++] = respVec.bottom.z;
    o[k++] = impResp.top.x; o[k++] = impResp.top.y; o[k++] = impResp.top.z;
    o[k++] = impResp.bottom.x; o[k++] = impResp.bottom.y; o[k++] = impResp.bottom.z;
    o[k++] = projV;
    o[k++] = projA;
}

// ===========================================================================
// P8.102b: setupFinalizeExtSolverConstraintsBlock, the TGS overload
// (artiConstraintPrep2.cu:617-1112) — one lane of the ext contact-row prep.
//
// TRAPS (each a sweep mutant unless marked):
//  9.  The contactCount == 0 early-out writes EXACTLY four header fields
//      (numFrictionConstr 0, numNormalConstr 0, forceWritebackOffset ~0,
//      invMass0_1_angDom0_1 zero) and returns.
// 10.  invMass0_1_angDom0_1 stores the invMassScale terms SWIZZLED:
//      (lin0X, lin1Z, ang0Y, ang1W) — x, z, y, w of the source.
// 11.  The friction header's biasCoefficient is invDt (not p8, not invDtp8).
// 12.  The per-contact slop scales by max(angNorVel/linNorVel, 1) with a
//      linNorVel==0 guard that substitutes 1 — component-wise zeroing of
//      raXn/rbXn against THAT scaled slop; the friction rows use the RAW
//      solverOffsetSlop.
// 13.  **The engine's own precedence quirk, transcribed as-is (FIFTH
//      upstream-bug candidate)**: `cTargetVel + isGreater2 ? -vrel*rest : 0`
//      parses as `(cTargetVel + (float)isGreater2) != 0 ? ... : 0` — the
//      bounce term REPLACES the target velocity whenever the sum is nonzero,
//      and a nonzero cTargetVel with no bounce still selects the bounce arm.
//      (The rigid twin at constraintPrepTGS has the parenthesized form; the
//      arti path does not.)  Also isGreater2 tests restitution > 1e-5, not
//      > 0 as the rigid twin does.
// 14.  minNormalForce = (sum of deltaF / contactCount) * 0.25 with deltaF =
//      max((targetVelocity - penetrationInvDt - vrel) * velMultiplier, 0).
// 15.  Kinematic bodies subtract/add their PROJECTED velocity from the
//      target (vrel1/vrel2 for normals, projectVelocity for frictions).
// 16.  The friction anchor count doubles (t0+t1 per anchor), +1 for
//      torsional when exactly ONE anchor and either torsional term nonzero;
//      frictionScale halves static/dynamic friction at aCount == 2.
// 17.  Friction anchors rotate through the body quats and the error is
//      (ra - rb) + (frame0p - frame1p), dotted per tangent; perPointFriction
//      reads contactID[j] for the target-velocity index, else contact 0.
// 18.  The torsional row writes response vectors built from (normal, 0) —
//      no lever arms — and frictionError/targetVel 0; torsionalFrictionScale
//      = max(torsionalData.y, sqrt(max(0, -minPen) * torsionalData.x)).
// 19.  Every row appends one PxgArticulationBlockResponse (deltaV0/deltaV1,
//      linear parts in the Lin rows, angular in the Ang rows) and the
//      cursor advances across normals -> t0/t1 pairs -> torsional.
// ===========================================================================

public uint acpSetupFinalizeExtTGS(
    Ptr<PxgBlockContactData> contactData, uint cdi,
    Ptr<PxgBlockContactPoint> contacts, uint contactBase,
    uint contactCount,
    Ptr<PxgBlockFrictionPatch> frictionPatch, uint fpi,
    Ptr<PxgBlockFrictionAnchorPatch> fAnchor, uint fai,
    AcpBody b0, AcpBody b1,
    float invDt, float stepDt, float invTotalDt,
    float bounceThresholdF32, float biasCoefficient,
    uint threadIndex, uint forceWritebackBufferOffset, bool perPointFriction,
    Ptr<PxgTGSBlockSolverContactHeader> contactHeader, uint chi,
    Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeader, uint fhi,
    Ptr<PxgTGSBlockSolverContactPoint> contactPoints, uint cpBase,
    Ptr<PxgTGSBlockSolverContactFriction> frictions, uint frBase,
    Ptr<PxgArticulationBlockResponse> responses, uint respBase,
    float solverOffsetSlop, float2 torsionalFrictionData, float totalDt)
{
    // PxgMaterialContactData, unpacked from the packed float4
    const float4 cd4 = contactData[cdi].contactData[threadIndex];
    const uint packedW = asuint(cd4.w);
    const uint flags = (packedW >> 8) & 0xFFu;   // mSolverFlags (byte 1 of the packed word)
    const float restDistance = cd4.x;
    const float staticFrictionIn = cd4.y;
    const float dynamicFrictionIn = cd4.z;

    const float3 bodyFrame0p = b0.b2wP;
    const float3 bodyFrame1p = b1.b2wP;

    const float cfm = (b0.cfm > b1.cfm) ? b0.cfm : b1.cfm;   // P9.86c: PxMax = selectMax a>b?a:b (artiConstraintPrep2.cu:660); FMax gives +0 for (+0,-0)
    const float maxPenBias = max(b0.penBiasClamp, b1.penBiasClamp);
    const float p8 = min(0.8f, biasCoefficient);
    const float bounceThreshold = bounceThresholdF32;
    const float invDtp8 = invDt * p8;

    const float4 lin0X_ang0Y_lin1Z_ang1W = contactData[cdi].mInvMassScale[threadIndex];

    const float3 linVel0 = b0.velocity.bottom;
    const float3 linVel1 = b1.velocity.bottom;
    const float3 angVel0 = b0.velocity.top;
    const float3 angVel1 = b1.velocity.top;

    uint responsesWritten = 0u;

    const uint aCount = frictionPatch[fpi].anchorCount[threadIndex];
    uint anchorCount = aCount * 2u;
    const bool hasTorsional =
        (torsionalFrictionData.x != 0.0f || torsionalFrictionData.y != 0.0f) && aCount == 1u;
    if (hasTorsional)
        anchorCount++;

    if (contactCount == 0u)
    {
        // trap 9: exactly these four, then out
        frictionHeader[fhi].numFrictionConstr[threadIndex] = 0u;
        contactHeader[chi].numNormalConstr[threadIndex] = 0u;
        contactHeader[chi].forceWritebackOffset[threadIndex] = 0xffffffffu;
        contactHeader[chi].invMass0_1_angDom0_1[threadIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return 0u;
    }

    const float maxImpulse = (b0.maxImpulse < b1.maxImpulse) ? b0.maxImpulse : b1.maxImpulse;   // P9.86: PxMin select (artiConstraintPrep2.cu:700, NaN parity)

    contactHeader[chi].forceWritebackOffset[threadIndex] = forceWritebackBufferOffset;
    contactHeader[chi].flags[threadIndex] = flags;
    contactHeader[chi].maxPenBias[threadIndex] = maxPenBias;
    // trap 10: the swizzle
    contactHeader[chi].invMass0_1_angDom0_1[threadIndex] =
        float4(lin0X_ang0Y_lin1Z_ang1W.x, lin0X_ang0Y_lin1Z_ang1W.z,
               lin0X_ang0Y_lin1Z_ang1W.y, lin0X_ang0Y_lin1Z_ang1W.w);

    const float4 normal4_rW = contactData[cdi].normal_restitutionW[threadIndex];
    const float3 normal = normal4_rW.xyz;
    const float restitution = normal4_rW.w;
    const float norVel0 = dot(normal, linVel0);
    const float norVel1 = dot(normal, linVel1);
    const float linNorVel = norVel0 - norVel1;
    const float damping = contactData[cdi].damping[threadIndex];

    frictionHeader[fhi].biasCoefficient[threadIndex] = invDt;   // trap 11
    contactHeader[chi].restitutionXdt[threadIndex] = restitution * stepDt;
    contactHeader[chi].cfm[threadIndex] = cfm;
    contactHeader[chi].p8[threadIndex] = p8;

    float normalForce = 0.0f;
    float minPen = 0.0f;

    for (uint j = 0u; j < contactCount; ++j)
    {
        const float4 point4_separationW = contacts[contactBase + j].point_separationW[threadIndex];
        const float3 point = point4_separationW.xyz;
        const float separation = point4_separationW.w;

        const float4 targetVel4_maxImpulseW =
            contacts[contactBase + j].targetVel_maxImpulseW[threadIndex];
        const float3 targetVel = targetVel4_maxImpulseW.xyz;
        const float cTargetVel = dot(normal, targetVel);

        const float3 ra = point - bodyFrame0p;
        const float3 rb = point - bodyFrame1p;

        float3 raXn = cross(ra, normal);
        float3 rbXn = cross(rb, normal);

        const float angNorVel = dot(raXn, angVel0) - dot(rbXn, angVel1);

        // trap 12: the scaled slop
        const float slop = solverOffsetSlop
                         * max(linNorVel == 0.0f ? 1.0f : angNorVel / linNorVel, 1.0f);
        raXn.x = abs(raXn.x) < slop ? 0.0f : raXn.x;
        raXn.y = abs(raXn.y) < slop ? 0.0f : raXn.y;
        raXn.z = abs(raXn.z) < slop ? 0.0f : raXn.z;
        rbXn.x = abs(rbXn.x) < slop ? 0.0f : rbXn.x;
        rbXn.y = abs(rbXn.y) < slop ? 0.0f : rbXn.y;
        rbXn.z = abs(rbXn.z) < slop ? 0.0f : rbXn.z;

        const AbSpatialVector response0 = acpCreateImpulseResponseVector(b0, asmSVec(raXn, normal));
        const AbSpatialVector response1 =
            acpCreateImpulseResponseVector(b1, asmSVec(-rbXn, -normal));

        const AbSpatialVector deltaV0 = acpGetImpulseResponse(
            b0, asmSVec(normal * lin0X_ang0Y_lin1Z_ang1W.x, raXn * lin0X_ang0Y_lin1Z_ang1W.y));
        const AbSpatialVector deltaV1 = acpGetImpulseResponse(
            b1, asmSVec(-normal * lin0X_ang0Y_lin1Z_ang1W.z, -rbXn * lin0X_ang0Y_lin1Z_ang1W.w));

        const float resp0 = dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
        const float resp1 = dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);
        const float unitResponse = resp0 + resp1;

        const float vrel1 = norVel0 + acpProjectAngular(b0, raXn);
        const float vrel2 = norVel1 + acpProjectAngular(b1, rbXn);
        const float vrel = vrel1 - vrel2;

        const float penetration = separation - restDistance;
        const bool isSeparated = penetration > 0.0f;
        const float penetrationInvDt = penetration * invTotalDt;

        float scaledBias;
        float velMultiplier;
        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / (unitResponse + cfm)) : 0.0f;

        float compliantContactCoeff = 0.0f;
        if (restitution < 0.0f)
        {
            const bool collidingWithVrel = ((-vrel) * totalDt) > penetration;
            compliantContactCoeff = computeCompliantContactCoefficientsTGSFull(
                stepDt, flags, restitution, damping, unitResponse, recipResponse,
                isSeparated, collidingWithVrel, velMultiplier, scaledBias);
        }
        else
        {
            velMultiplier = recipResponse;
            scaledBias = isSeparated ? -invDt : -invDtp8;
        }

        contactPoints[cpBase + j].resp0[threadIndex] = resp0;
        contactPoints[cpBase + j].resp1[threadIndex] = resp1;

        // P9.86: PxMin = selectMin a < b ? a : b, tie keeps the +0 seed (artiConstraintPrep2.cu:827);
        // stamped FMin returns -0 for (-0, +0).
        minPen = (penetration < minPen) ? penetration : minPen;

        // trap 13: the precedence quirk and the 1e-5 threshold, as found
        const bool isGreater2 = ((restitution > 1e-5f) && (bounceThreshold > vrel))
                             && ((-vrel) > penetrationInvDt);

        float totalError = penetration;
        const float tvCond = cTargetVel + (isGreater2 ? 1.0f : 0.0f);
        float targetVelocity = (tvCond != 0.0f) ? (-vrel) * restitution : 0.0f;

        const float deltaF =
            max((targetVelocity + (-penetrationInvDt - vrel)) * velMultiplier, 0.0f);
        normalForce += deltaF;

        if (b0.isKinematic)
            targetVelocity -= vrel1;
        if (b1.isKinematic)
            targetVelocity += vrel2;

        if (isGreater2)
        {
            const float ratio = totalDt + penetration / vrel;
            totalError += ratio * targetVelocity;
        }

        contactPoints[cpBase + j].raXn_extraCoeff[threadIndex] =
            float4(response0.top, compliantContactCoeff);
        contactPoints[cpBase + j].rbXn_targetVelW[threadIndex] =
            float4(-response1.top, targetVelocity);
        contactPoints[cpBase + j].appliedForce[threadIndex] = 0.0f;
        // P9.86: PxMin is the select a < b ? a : b (keeps b on a +0/-0 tie, artiConstraintPrep2.cu:856);
        // llvmpipe's FMin returns -0 for either order, so PxMin(-0 body, +0 ignore()) would differ.
        contactPoints[cpBase + j].maxImpulse[threadIndex] =
            (maxImpulse < targetVel4_maxImpulseW.w) ? maxImpulse : targetVel4_maxImpulseW.w;
        contactPoints[cpBase + j].separation[threadIndex] = totalError;
        contactPoints[cpBase + j].biasCoefficient[threadIndex] = scaledBias;

        const uint r = respBase + responsesWritten;
        responses[r].deltaRALin_x[threadIndex] = deltaV0.bottom.x;
        responses[r].deltaRALin_y[threadIndex] = deltaV0.bottom.y;
        responses[r].deltaRALin_z[threadIndex] = deltaV0.bottom.z;
        responses[r].deltaRAAng_x[threadIndex] = deltaV0.top.x;
        responses[r].deltaRAAng_y[threadIndex] = deltaV0.top.y;
        responses[r].deltaRAAng_z[threadIndex] = deltaV0.top.z;
        responses[r].deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
        responses[r].deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
        responses[r].deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
        responses[r].deltaRBAng_x[threadIndex] = deltaV1.top.x;
        responses[r].deltaRBAng_y[threadIndex] = deltaV1.top.y;
        responses[r].deltaRBAng_z[threadIndex] = deltaV1.top.z;
        responsesWritten++;
    }

    contactHeader[chi].minNormalForce[threadIndex] = (normalForce / float(contactCount)) * 0.25f;

    const float frictionScale = (aCount == 2u) ? 0.5f : 1.0f;
    const float staticFriction = staticFrictionIn * frictionScale;
    const float dynamicFriction = dynamicFrictionIn * frictionScale;

    const bool haveFriction = anchorCount != 0u;
    contactHeader[chi].numNormalConstr[threadIndex] = contactCount;
    frictionHeader[fhi].numFrictionConstr[threadIndex] = anchorCount;

    contactHeader[chi].normal_staticFriction[threadIndex] = float4(normal, staticFriction);
    frictionHeader[fhi].dynamicFriction[threadIndex] = dynamicFriction;

    if (haveFriction)
    {
        const float3 vrel3 = linVel0 - linVel1;

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        const float3 t0Fallback1 = float3(0.0f, -normal.z, normal.y);
        const float3 t0Fallback2 = float3(-normal.y, normal.x, 0.0f);
        const float3 t0Fallback = orthoThreshold > abs(normal.x) ? t0Fallback1 : t0Fallback2;

        float3 t0 = vrel3 - normal * dot(normal, vrel3);
        t0 = dot(t0, t0) > p1 ? t0 : t0Fallback;
        // spelled out, NOT the normalize() intrinsic (1 ULP off on llvmpipe),
        // and recip-MULTIPLY rather than divide: PxVec3's operator/ computes
        // 1/f and scales, so this is the engine's own rounding.  The fallback
        // guarantees m > 0.
        t0 = t0 * (1.0f / sqrt(dot(t0, t0)));

        const float3 t1 = cross(normal, t0);

        frictionHeader[fhi].broken[threadIndex] = 0u;
        frictionHeader[fhi].frictionNormals0[threadIndex] = float4(t0, 0.0f);
        frictionHeader[fhi].frictionNormals1[threadIndex] = float4(t1, 0.0f);

        const float3 relTr = bodyFrame0p - bodyFrame1p;

        for (uint j = 0u; j < aCount; ++j)
        {
            const uint f0i = frBase + 2u * j;
            const uint f1i = frBase + 2u * j + 1u;

            const float4 body0Anchor4 = (j == 0u) ? fAnchor[fai].body0Anchors0[threadIndex]
                                                  : fAnchor[fai].body0Anchors1[threadIndex];
            const float4 body1Anchor4 = (j == 0u) ? fAnchor[fai].body1Anchors0[threadIndex]
                                                  : fAnchor[fai].body1Anchors1[threadIndex];

            const float3 ra = quatRotate(b0.b2wQ, body0Anchor4.xyz);
            const float3 rb = quatRotate(b1.b2wQ, body1Anchor4.xyz);

            const uint index = perPointFriction
                ? ((j == 0u) ? frictionPatch[fpi].contactID0[threadIndex]
                             : frictionPatch[fpi].contactID1[threadIndex])
                : 0u;
            const float4 targetVel4 = contacts[contactBase + index].targetVel_maxImpulseW[threadIndex];
            const float3 tvel = targetVel4.xyz;

            const float3 error = (ra - rb) + relTr;

            // t0 row
            {
                float3 raXn = cross(ra, t0);
                float3 rbXn = cross(rb, t0);
                raXn.x = abs(raXn.x) < solverOffsetSlop ? 0.0f : raXn.x;
                raXn.y = abs(raXn.y) < solverOffsetSlop ? 0.0f : raXn.y;
                raXn.z = abs(raXn.z) < solverOffsetSlop ? 0.0f : raXn.z;
                rbXn.x = abs(rbXn.x) < solverOffsetSlop ? 0.0f : rbXn.x;
                rbXn.y = abs(rbXn.y) < solverOffsetSlop ? 0.0f : rbXn.y;
                rbXn.z = abs(rbXn.z) < solverOffsetSlop ? 0.0f : rbXn.z;

                const AbSpatialVector response0 =
                    acpCreateImpulseResponseVector(b0, asmSVec(raXn, t0));
                const AbSpatialVector response1 =
                    acpCreateImpulseResponseVector(b1, asmSVec(-rbXn, -t0));
                const AbSpatialVector deltaV0 = acpGetImpulseResponse(
                    b0, asmSVec(t0 * lin0X_ang0Y_lin1Z_ang1W.x, raXn * lin0X_ang0Y_lin1Z_ang1W.y));
                const AbSpatialVector deltaV1 = acpGetImpulseResponse(
                    b1, asmSVec(-t0 * lin0X_ang0Y_lin1Z_ang1W.z, -rbXn * lin0X_ang0Y_lin1Z_ang1W.w));

                const float resp0 =
                    dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
                const float resp1 =
                    dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);

                float targetVelF = dot(tvel, t0);
                if (b0.isKinematic)
                    targetVelF -= acpProjectVelocity(b0, asmSVec(raXn, t0));
                if (b1.isKinematic)
                    targetVelF += acpProjectVelocity(b1, asmSVec(rbXn, t0));

                const float frictionError = dot(error, t0);

                frictions[f0i].resp0[threadIndex] = resp0;
                frictions[f0i].resp1[threadIndex] = resp1;
                frictions[f0i].raXn_error[threadIndex] = float4(response0.top, frictionError);
                frictions[f0i].rbXn_targetVelW[threadIndex] = float4(-response1.top, targetVelF);
                frictions[f0i].appliedForce[threadIndex] = 0.0f;

                const uint r = respBase + responsesWritten;
                responses[r].deltaRALin_x[threadIndex] = deltaV0.bottom.x;
                responses[r].deltaRALin_y[threadIndex] = deltaV0.bottom.y;
                responses[r].deltaRALin_z[threadIndex] = deltaV0.bottom.z;
                responses[r].deltaRAAng_x[threadIndex] = deltaV0.top.x;
                responses[r].deltaRAAng_y[threadIndex] = deltaV0.top.y;
                responses[r].deltaRAAng_z[threadIndex] = deltaV0.top.z;
                responses[r].deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
                responses[r].deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
                responses[r].deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
                responses[r].deltaRBAng_x[threadIndex] = deltaV1.top.x;
                responses[r].deltaRBAng_y[threadIndex] = deltaV1.top.y;
                responses[r].deltaRBAng_z[threadIndex] = deltaV1.top.z;
                responsesWritten++;
            }

            // t1 row
            {
                float3 raXn = cross(ra, t1);
                float3 rbXn = cross(rb, t1);
                raXn.x = abs(raXn.x) < solverOffsetSlop ? 0.0f : raXn.x;
                raXn.y = abs(raXn.y) < solverOffsetSlop ? 0.0f : raXn.y;
                raXn.z = abs(raXn.z) < solverOffsetSlop ? 0.0f : raXn.z;
                rbXn.x = abs(rbXn.x) < solverOffsetSlop ? 0.0f : rbXn.x;
                rbXn.y = abs(rbXn.y) < solverOffsetSlop ? 0.0f : rbXn.y;
                rbXn.z = abs(rbXn.z) < solverOffsetSlop ? 0.0f : rbXn.z;

                const AbSpatialVector response0 =
                    acpCreateImpulseResponseVector(b0, asmSVec(raXn, t1));
                const AbSpatialVector response1 =
                    acpCreateImpulseResponseVector(b1, asmSVec(-rbXn, -t1));
                const AbSpatialVector deltaV0 = acpGetImpulseResponse(
                    b0, asmSVec(t1 * lin0X_ang0Y_lin1Z_ang1W.x, raXn * lin0X_ang0Y_lin1Z_ang1W.y));
                const AbSpatialVector deltaV1 = acpGetImpulseResponse(
                    b1, asmSVec(-t1 * lin0X_ang0Y_lin1Z_ang1W.z, -rbXn * lin0X_ang0Y_lin1Z_ang1W.w));

                const float resp0 =
                    dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
                const float resp1 =
                    dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);

                float targetVelF = dot(tvel, t1);
                if (b0.isKinematic)
                    targetVelF -= acpProjectVelocity(b0, asmSVec(raXn, t1));
                if (b1.isKinematic)
                    targetVelF += acpProjectVelocity(b1, asmSVec(rbXn, t1));

                const float frictionError = dot(error, t1);

                frictions[f1i].resp0[threadIndex] = resp0;
                frictions[f1i].resp1[threadIndex] = resp1;
                frictions[f1i].raXn_error[threadIndex] = float4(response0.top, frictionError);
                frictions[f1i].rbXn_targetVelW[threadIndex] = float4(-response1.top, targetVelF);
                frictions[f1i].appliedForce[threadIndex] = 0.0f;

                const uint r = respBase + responsesWritten;
                responses[r].deltaRALin_x[threadIndex] = deltaV0.bottom.x;
                responses[r].deltaRALin_y[threadIndex] = deltaV0.bottom.y;
                responses[r].deltaRALin_z[threadIndex] = deltaV0.bottom.z;
                responses[r].deltaRAAng_x[threadIndex] = deltaV0.top.x;
                responses[r].deltaRAAng_y[threadIndex] = deltaV0.top.y;
                responses[r].deltaRAAng_z[threadIndex] = deltaV0.top.z;
                responses[r].deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
                responses[r].deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
                responses[r].deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
                responses[r].deltaRBAng_x[threadIndex] = deltaV1.top.x;
                responses[r].deltaRBAng_y[threadIndex] = deltaV1.top.y;
                responses[r].deltaRBAng_z[threadIndex] = deltaV1.top.z;
                responsesWritten++;
            }
        }

        if (hasTorsional)
        {
            const uint f0i = frBase + 2u;

            // P9.86: PxMax(float) = selectMax (a > b ? a : b), tie -> b (artiConstraintPrep2.cu:1069):
            // with no penetrating contact minPen stays +0, so PxMax(0, -0) = -0, sqrt(-0 * x) = -0
            // (x >= 0), and the CUDA stores -0 whenever torsionalFrictionData.y is +/-0.
            const float negMinPen = -minPen;
            const float torsionalPen = (0.0f > negMinPen) ? 0.0f : negMinPen;
            const float torsionalSqrt = sqrt(torsionalPen * torsionalFrictionData.x);
            const float torsionalScale = (torsionalFrictionData.y > torsionalSqrt)
                                       ? torsionalFrictionData.y : torsionalSqrt;
            frictionHeader[fhi].torsionalFrictionScale[threadIndex] = torsionalScale;

            const AbSpatialVector response0 =
                acpCreateImpulseResponseVector(b0, asmSVec(normal, float3(0.0f, 0.0f, 0.0f)));
            const AbSpatialVector response1 =
                acpCreateImpulseResponseVector(b1, asmSVec(-normal, float3(0.0f, 0.0f, 0.0f)));
            const AbSpatialVector deltaV0 = acpGetImpulseResponse(
                b0, asmSVec(float3(0.0f, 0.0f, 0.0f) * lin0X_ang0Y_lin1Z_ang1W.x,
                            normal * lin0X_ang0Y_lin1Z_ang1W.y));
            const AbSpatialVector deltaV1 = acpGetImpulseResponse(
                b1, asmSVec(float3(0.0f, 0.0f, 0.0f) * lin0X_ang0Y_lin1Z_ang1W.z,
                            (-normal) * lin0X_ang0Y_lin1Z_ang1W.w));

            const float resp0 =
                dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
            const float resp1 =
                dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);

            frictions[f0i].resp0[threadIndex] = resp0;
            frictions[f0i].resp1[threadIndex] = resp1;
            frictions[f0i].raXn_error[threadIndex] = float4(response0.top, 0.0f);
            frictions[f0i].rbXn_targetVelW[threadIndex] = float4(-response1.top, 0.0f);
            frictions[f0i].appliedForce[threadIndex] = 0.0f;

            const uint r = respBase + responsesWritten;
            responses[r].deltaRALin_x[threadIndex] = deltaV0.bottom.x;
            responses[r].deltaRALin_y[threadIndex] = deltaV0.bottom.y;
            responses[r].deltaRALin_z[threadIndex] = deltaV0.bottom.z;
            responses[r].deltaRAAng_x[threadIndex] = deltaV0.top.x;
            responses[r].deltaRAAng_y[threadIndex] = deltaV0.top.y;
            responses[r].deltaRAAng_z[threadIndex] = deltaV0.top.z;
            responses[r].deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
            responses[r].deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
            responses[r].deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
            responses[r].deltaRBAng_x[threadIndex] = deltaV1.top.x;
            responses[r].deltaRBAng_y[threadIndex] = deltaV1.top.y;
            responses[r].deltaRBAng_z[threadIndex] = deltaV1.top.z;
            responsesWritten++;
        }
    }

    return responsesWritten;
}

// ---------------------------------------------------------------------------
// test entry for the TGS ext contact setup: builds the two bodies from the
// node table (the P8.102a machinery), then runs one batch slot per lane with
// per-lane counts/flags from the io block
// ---------------------------------------------------------------------------

struct AcsParams
{
    uint64_t contactData;     // @0   one PxgBlockContactData
    uint64_t contacts;        // @8   PxgBlockContactPoint[maxContacts]
    uint64_t frictionPatch;   // @16  one PxgBlockFrictionPatch
    uint64_t fAnchor;         // @24  one PxgBlockFrictionAnchorPatch
    uint64_t nodes;           // @32  uint2 pairs per lane (body0, body1)
    uint64_t bodyIndices;     // @40  uint pairs per lane
    uint64_t articulations;   // @48
    uint64_t solverBodyData;  // @56
    uint64_t txIDatas;        // @64
    uint64_t bodyFrames;      // @72
    uint64_t contactHeader;   // @80  one PxgTGSBlockSolverContactHeader
    uint64_t frictionHeader;  // @88  one PxgTGSBlockSolverFrictionHeader
    uint64_t contactPoints;   // @96  rows
    uint64_t frictions;       // @104 rows
    uint64_t responses;       // @112 rows
    uint64_t io;              // @120 per-lane uint4: {contactCount, writebackOffset,
                              //      perPointFriction, respCountOut}
    float invDt;              // @128
    float stepDt;             // @132
    float invTotalDt;         // @136
    float bounceThreshold;    // @140
    float biasCoefficient;    // @144
    float solverOffsetSlop;   // @148
    float torsionalX;         // @152
    float torsionalY;         // @156
    float totalDt;            // @160
    uint nbLanes;             // @164 -> 168
}

[[vk::push_constant]]
ConstantBuffer<AcsParams> gAcs;

[shader("compute")]
[numthreads(32, 1, 1)]
void acpSetupTest(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint lane = dispatchThreadId.x;
    if (lane >= gAcs.nbLanes)
        return;

    const uint2 node0 = (Ptr<uint2>(gAcs.nodes))[lane * 2u + 0u];
    const uint2 node1 = (Ptr<uint2>(gAcs.nodes))[lane * 2u + 1u];
    const uint bi0 = (Ptr<uint>(gAcs.bodyIndices))[lane * 2u + 0u];
    const uint bi1 = (Ptr<uint>(gAcs.bodyIndices))[lane * 2u + 1u];

    AcpBody b0, b1;
    acpCreateSolverExtBody(node0, bi0, gAcs.articulations, gAcs.solverBodyData,
                           gAcs.txIDatas, gAcs.bodyFrames, b0);
    acpCreateSolverExtBody(node1, bi1, gAcs.articulations, gAcs.solverBodyData,
                           gAcs.txIDatas, gAcs.bodyFrames, b1);

    Ptr<uint4> pio = Ptr<uint4>(gAcs.io);
    const uint4 io = pio[lane];
    const uint contactCount = io.x;
    const uint writebackOffset = io.y;
    const bool perPointFriction = io.z != 0u;

    const uint written = acpSetupFinalizeExtTGS(
        Ptr<PxgBlockContactData>(gAcs.contactData), 0u,
        Ptr<PxgBlockContactPoint>(gAcs.contacts), 0u,
        contactCount,
        Ptr<PxgBlockFrictionPatch>(gAcs.frictionPatch), 0u,
        Ptr<PxgBlockFrictionAnchorPatch>(gAcs.fAnchor), 0u,
        b0, b1,
        gAcs.invDt, gAcs.stepDt, gAcs.invTotalDt,
        gAcs.bounceThreshold, gAcs.biasCoefficient,
        lane, writebackOffset, perPointFriction,
        Ptr<PxgTGSBlockSolverContactHeader>(gAcs.contactHeader), 0u,
        Ptr<PxgTGSBlockSolverFrictionHeader>(gAcs.frictionHeader), 0u,
        Ptr<PxgTGSBlockSolverContactPoint>(gAcs.contactPoints), 0u,
        Ptr<PxgTGSBlockSolverContactFriction>(gAcs.frictions), 0u,
        Ptr<PxgArticulationBlockResponse>(gAcs.responses), 0u,
        gAcs.solverOffsetSlop, float2(gAcs.torsionalX, gAcs.torsionalY), gAcs.totalDt);

    pio[lane] = uint4(io.x, io.y, io.z, written);
}

// ---------------------------------------------------------------------------
// P9.94b: helper-level probe of alignedTransformInvTransform (pxgSlangPrepCommon)
// -- PxAlignedTransform::transformInv(PxAlignedTransform), the body1ToBody0 of
// getFrictionPatches.  Its p differs from the old quatRotateInv spelling only in
// the sign of an exact zero, which no kernel output can see (pointsAreClose
// erases it), so the revert mutant is killed here, one 32-byte transform per lane.
// Test-only entry: no manifest row.
// ---------------------------------------------------------------------------

struct AtiParams
{
    uint64_t a;    // @0  PxAlignedTransform per lane (this)
    uint64_t b;    // @8  PxAlignedTransform per lane (src)
    uint64_t out;  // @16 PxAlignedTransform per lane
    uint n;        // @24
    uint pad;      // @28 -> 32
}

[[vk::push_constant]]
ConstantBuffer<AtiParams> gAti;

[shader("compute")]
[numthreads(32, 1, 1)]
void acpTransformInvProbe(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint lane = dispatchThreadId.x;
    if (lane >= gAti.n)
        return;
    (Ptr<PxgAlignedTransform>(gAti.out))[lane] =
        alignedTransformInvTransform((Ptr<PxgAlignedTransform>(gAti.a))[lane],
                                     (Ptr<PxgAlignedTransform>(gAti.b))[lane]);
}

// ===========================================================================
// P8.102c: the ENGINE entry -- artiTGSContactConstraintBlockPrepareLaunch
// (artiConstraintPrep2.cu:1388 -> artiContactConstraintBlockPrepare :1201
//  -> artiCreateFinalizeSolverContactsBlockGPU :1114).
//
// TRAPS (continuing the module's numbering; each carries a sweep mutant):
// 20. **The batch guard is nbArtiContactBatches + nbArtiStaticContactBatches
//     + nbArtiSelfContactBatches**, and the batch index is offset by
//     prepDesc->numBatches (the NON-articulation batch count).
// 21. **The body index selects remappedBodyA/BIndex only for articulation
//     node indices**; rigids read bodyA/BIndex.
// 22. **nbContacts comes from the MATERIAL byte** (packed word & 0xFF), not
//     from a separate count -- the P8.102b setup received it as a parameter,
//     here it is byte 0 of contactData.w.
// 23. **solverOffsetSlop is max(b0.offsetSlop, b1.offsetSlop)** -- the
//     commented-out old call in the CUDA used prepDesc->solverOffsetSlop and
//     the live one does NOT.
// 24. **The friction-index scatter** writes createPatchIndex(descIndexBatch,
//     lane) at edgeIndex + totalCurrentEdges * mPatchIndex[lane] -- the
//     PREVIOUS-edge total feeds getFrictionPatches instead.
// 25. **getFrictionPatches is gated on !(perPointFriction || disableFriction),
//     growPatches on !disableFriction alone**; frictionPatch.broken[lane]=0
//     always.  (The CUDA's __syncwarp() after correlate is dropped exactly as
//     the rigid twin's port established: every lane touches only its own
//     [lane] column, there is no cross-lane traffic to order.)
// 26. **The post-setup writes**: patchIndex[lane] = mFrictionPatchIndex[lane],
//     then anchorPoints[0/1] = b0.body2World.transform(body0Anchors[0/1])
//     gated on anchorCount >= 1 / == 2 -- through BODY 0's transform, w = 0.
// 27. **The TGS dt mapping** (:1392): invDt=stepInvDtF32, stepDt=stepDt,
//     invTotalDt=invDtF32, totalDt=dt.
// ===========================================================================

struct AcwParams
{
    uint64_t prepDesc;    // PxgConstraintPrepareDesc*
    uint64_t sharedDesc;  // PxgSolverSharedDesc<IterativeSolveDataTGS>*
}

[[vk::push_constant]]
ConstantBuffer<AcwParams> gAcw;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiTGSContactConstraintBlockPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                                uint3 groupId : SV_GroupID)
{
    const uint blockStride = 64u / 32u;
    const uint warpIndex = groupId.x * blockStride + groupThreadId.x / 32u;
    const uint lane = groupThreadId.x & 31u;

    Ptr<PxgConstraintPrepareDescGpu> prepDesc =
        Ptr<PxgConstraintPrepareDescGpu>(gAcw.prepDesc);
    Ptr<PxgSolverSharedDescTGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescTGSGpu>(gAcw.sharedDesc);

    // trap 20
    const uint nbContactBatchesWithSelf = prepDesc[0].numArtiContactBatches
        + prepDesc[0].numArtiStaticContactBatches + prepDesc[0].numArtiSelfContactBatches;
    const uint i = warpIndex;
    if (i >= nbContactBatchesWithSelf)
        return;

    Ptr<uint> batchIndices = Ptr<uint>(prepDesc[0].artiContactConstraintBatchIndices);
    const uint batchIndex = batchIndices[i] + prepDesc[0].numBatches;

    Ptr<PxgBlockConstraintBatchGpu> batches =
        Ptr<PxgBlockConstraintBatchGpu>(sharedDesc[0].iter_blockConstraintBatch);
    const uint descIndexBatch = batches[batchIndex].mConstraintBatchIndex;
    const uint responseIndex = batches[batchIndex].mArticulationResponseIndex;
    const uint descStride = batches[batchIndex].descStride_constraintType & 0xFFFFu;

    if (lane >= descStride)
        return;

    const uint2 nodeIndexA = batches[batchIndex].bodyANodeIndex[lane];
    const uint2 nodeIndexB = batches[batchIndex].bodyBNodeIndex[lane];

    // trap 21
    const uint bodyAIndex = nodeIndexIsArticulation(nodeIndexA)
        ? batches[batchIndex].remappedBodyAIndex[lane] : batches[batchIndex].bodyAIndex[lane];
    const uint bodyBIndex = nodeIndexIsArticulation(nodeIndexB)
        ? batches[batchIndex].remappedBodyBIndex[lane] : batches[batchIndex].bodyBIndex[lane];

    Ptr<PxgBlockWorkUnit> workUnits = Ptr<PxgBlockWorkUnit>(prepDesc[0].blockWorkUnit);
    Ptr<uint2> currFrictionIndices = Ptr<uint2>(prepDesc[0].blockCurrentFrictionIndices);
    Ptr<uint2> prevFrictionIndices = Ptr<uint2>(prepDesc[0].blockPreviousFrictionIndices);

    // trap 24: PxgBlockFrictionIndex::createPatchIndex(descIndexBatch, lane)
    const uint edgeIndex = workUnits[descIndexBatch].mEdgeIndex[lane];
    const uint frictionIndex =
        edgeIndex + prepDesc[0].totalCurrentEdges * workUnits[descIndexBatch].mPatchIndex[lane];
    currFrictionIndices[frictionIndex] =
        uint2((descIndexBatch << 5) | lane, descIndexBatch >> 27);

    AcpBody b0, b1;
    acpCreateSolverExtBody(nodeIndexA, bodyAIndex, sharedDesc[0].articulations,
                           prepDesc[0].solverBodyDataPool, prepDesc[0].solverBodyTxIDataPool,
                           prepDesc[0].body2WorldPool, b0);
    acpCreateSolverExtBody(nodeIndexB, bodyBIndex, sharedDesc[0].articulations,
                           prepDesc[0].solverBodyDataPool, prepDesc[0].solverBodyTxIDataPool,
                           prepDesc[0].body2WorldPool, b1);

    // trap 23
    const float offsetSlop = max(b0.offsetSlop, b1.offsetSlop);

    const uint forceWritebackOffset = workUnits[descIndexBatch].mWriteback[lane];
    const float2 torsionalData = workUnits[descIndexBatch].mTorsionalFrictionData[lane];

    // ---- artiCreateFinalizeSolverContactsBlockGPU (:1114) ----

    Ptr<PxgBlockContactData> contactPrepPool =
        Ptr<PxgBlockContactData>(prepDesc[0].blockContactCurrentPrepPool);

    // trap 22
    const float4 cd4 = contactPrepPool[descIndexBatch].contactData[lane];
    const uint materialWord = asuint(cd4.w);
    const uint nbContacts = materialWord & 0xFFu;
    const uint flags = (materialWord >> 8) & 0xFFu;
    const uint prevFrictionPatchCount = (materialWord >> 16) & 0xFFu;

    const float4 normal4 = contactPrepPool[descIndexBatch].normal_restitutionW[lane];

    // PxAlignedTransform(PxTransform): p.w = 0
    PxgAlignedTransform bodyFrame0;
    bodyFrame0.q = b0.b2wQ;
    bodyFrame0.p = float4(b0.b2wP, 0.0f);
    PxgAlignedTransform bodyFrame1;
    bodyFrame1.q = b1.b2wQ;
    bodyFrame1.p = float4(b1.b2wP, 0.0f);

    Ptr<PxgBlockFrictionPatch> currFrictionPatches =
        Ptr<PxgBlockFrictionPatch>(sharedDesc[0].blockCurrentFrictionPatches);
    Ptr<PxgBlockFrictionPatch> prevFrictionPatches =
        Ptr<PxgBlockFrictionPatch>(sharedDesc[0].blockPreviousFrictionPatches);
    Ptr<PxgBlockFrictionAnchorPatch> currAnchors =
        Ptr<PxgBlockFrictionAnchorPatch>(prepDesc[0].blockCurrentAnchorPatches);
    Ptr<PxgBlockFrictionAnchorPatch> prevAnchors =
        Ptr<PxgBlockFrictionAnchorPatch>(prepDesc[0].blockPreviousAnchorPatches);
    Ptr<PxgBlockContactPoint> contactPrepPoints =
        Ptr<PxgBlockContactPoint>(prepDesc[0].blockContactPoints);
    const uint contactBase = batches[batchIndex].blockContactIndex;

    correlatePatchesPtr(currFrictionPatches, descIndexBatch, nbContacts, normal4.xyz,
                        bodyFrame0, bodyFrame1, lane);

    const bool perPointFriction = (flags & PXG_SC_FLAG_PER_POINT_FRICTION) != 0;
    const bool disableFriction = (flags & PXG_SC_FLAG_DISABLE_FRICTION) != 0;

    currFrictionPatches[descIndexBatch].broken[lane] = 0;
    float patchExtents = 0.0f;

    // trap 25
    if (!(perPointFriction || disableFriction))
    {
        getFrictionPatchesPtr(currFrictionPatches, descIndexBatch, currAnchors, descIndexBatch,
                              prevFrictionIndices, prepDesc[0].blockPreviousFrictionIndices != 0,
                              edgeIndex, prevFrictionPatches, prevAnchors, prevFrictionPatchCount,
                              bodyFrame0, bodyFrame1, prepDesc[0].correlationDistance, lane,
                              prepDesc[0].totalPreviousEdges, patchExtents, nbContacts);
    }

    if (!disableFriction)
        growPatchesPtr(currFrictionPatches, descIndexBatch, currAnchors, descIndexBatch,
                       contactPrepPoints, contactBase, nbContacts, bodyFrame0, bodyFrame1,
                       prepDesc[0].frictionOffsetThreshold + cd4.x, lane, patchExtents);

    Ptr<PxgTGSBlockSolverContactHeader> contactHeaders =
        Ptr<PxgTGSBlockSolverContactHeader>(sharedDesc[0].iter_blockContactHeaders);
    Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders =
        Ptr<PxgTGSBlockSolverFrictionHeader>(sharedDesc[0].iter_blockFrictionHeaders);
    Ptr<PxgTGSBlockSolverContactPoint> outContactPoints =
        Ptr<PxgTGSBlockSolverContactPoint>(sharedDesc[0].iter_blockContactPoints);
    Ptr<PxgTGSBlockSolverContactFriction> outFrictions =
        Ptr<PxgTGSBlockSolverContactFriction>(sharedDesc[0].iter_blockFrictions);
    Ptr<PxgArticulationBlockResponse> responses =
        Ptr<PxgArticulationBlockResponse>(sharedDesc[0].iter_artiResponse);

    // trap 27: (stepInvDtF32, stepDt, invDtF32, dt)
    acpSetupFinalizeExtTGS(
        contactPrepPool, descIndexBatch,
        contactPrepPoints, contactBase, nbContacts,
        currFrictionPatches, descIndexBatch,
        currAnchors, descIndexBatch,
        b0, b1,
        sharedDesc[0].stepInvDtF32, sharedDesc[0].stepDt, sharedDesc[0].invDtF32,
        prepDesc[0].bounceThresholdF32, prepDesc[0].biasCoefficient,
        lane, forceWritebackOffset, perPointFriction,
        contactHeaders, descIndexBatch,
        frictionHeaders, descIndexBatch,
        outContactPoints, batches[batchIndex].startConstraintIndex,
        outFrictions, batches[batchIndex].startFrictionIndex,
        responses, responseIndex,
        offsetSlop, torsionalData, sharedDesc[0].dt);

    // trap 26
    currFrictionPatches[descIndexBatch].patchIndex[lane] =
        workUnits[descIndexBatch].mFrictionPatchIndex[lane];

    if (currFrictionPatches[descIndexBatch].anchorCount[lane] >= 1u)
    {
        const float3 a0 = currAnchors[descIndexBatch].body0Anchors0[lane].xyz;
        currFrictionPatches[descIndexBatch].anchorPoints0[lane] =
            float4(quatRotate(b0.b2wQ, a0) + b0.b2wP, 0.0f);
    }
    if (currFrictionPatches[descIndexBatch].anchorCount[lane] == 2u)
    {
        const float3 a1 = currAnchors[descIndexBatch].body0Anchors1[lane].xyz;
        currFrictionPatches[descIndexBatch].anchorPoints1[lane] =
            float4(quatRotate(b0.b2wQ, a1) + b0.b2wP, 0.0f);
    }
}

// ===========================================================================
// P8.121: the PGS twin -- setupFinalizeExtSolverConstraintsBlock, PGS
// overload (artiConstraintPrep2.cu:169-613), and the engine entry
// artiContactConstraintBlockPrepareLaunch (:1380 -> the shared template
// body :1201 with the PGS dt mapping).  The output block types are the PGS
// PxgBlockSolver* family (PxgConstraintBlock.h), mirrored file-locally as
// AcpPgs* exactly as P8.120 mirrored the 1D family into npArtiJointPrep.
//
// TRAPS, continuing the module numbering (each a sweep mutant unless the
// note says the fixture pins it structurally):
// 28. **The contactCount == 0 early-out writes exactly THREE fields**
//     (numFrictionConstr 0, numNormalConstr 0, forceWritebackOffset ~0) --
//     the TGS twin (trap 9) also zeroes invMass0_1_angDom0_1; PGS leaves it
//     POISONED.
// 29. **p8 is FIXED at 0.8** -- the biasCoefficient parameter is DEAD in the
//     PGS overload (the TGS twin clamps min(0.8, biasCoefficient)).
// 30. **The header stores restitution and cfm** (the TGS twin's
//     restitutionXdt/p8/maxPenBias slots do not exist in the PGS header);
//     accumNormalForce is never written by prep.
// 31. **isSeparated is penetration >= 0** (the TGS twin tests > 0).
// 32. **penetrationInvDt scales by the SAME invDt** (the TGS twin uses the
//     total-dt inverse), and the non-separated bias arm clamps from BELOW at
//     maxPenBias: fmaxf(maxPenBias, penetration * invDtp8).
// 33. **isGreater2 tests restitution > 0** (the TGS twin: > 1e-5).
// 34. **targetVelocity is the PARENTHESIZED form**
//     cTargetVel + (isGreater2 ? (-vrel)*restitution : 0) -- the arti TGS
//     twin's :837 precedence quirk (fifth upstream candidate) is ABSENT in
//     the PGS overload.
// 35. **The velocity adjust is keyed on !islandNodeIndex.isArticulation()**
//     -- ANY rigid (dynamic, static, kinematic), not just kinematic as in
//     the TGS twin; same for the friction rows via projectVelocity.
// 36. **The ccd gate is the engine's own dead code**: isCCD is hard-coded
//     false (:213), so ccdMaxSeparation == PX_MAX_F32 and
//     (ccdSeparationCondition && isGreater2) reduces to isGreater2 -- every
//     bounce zeroes scaledBias.  Transcribed with the constant-true compare.
// 37. **The compliant branch returns coeff0 = a from the FULL
//     computeCompliantContactCoefficients and coeff1 = -(dt*restitution) *
//     penetration**; the rigid-PGS prep uses the REDUCED query form, but the
//     arti path needs velMultiplier for the deltaF below.  The non-compliant
//     branch stores coeff0 = biasedErr, coeff1 = unbiasedErr, with
//     unbiasedErr -= max(0, scaledBias) only when !isGreater2.
// 38. **deltaF uses tvel saved BEFORE the rigid-velocity adjust** and the
//     spelling (tvel - (vrel + max(penetrationInvDt, 0))) * velMultiplier
//     (the TGS twin: (targetVelocity + (-penetrationInvDt - vrel))).
// 39. The PGS contact row carries raXn_targetVelocity / rbXn_maxImpulse
//     (min'd against min(b0,b1).maxImpulse) / resp0 / resp1 / coeff0 /
//     coeff1 / appliedForce.
// 40. **The friction anchor error is (ra + frame0p) - (rb + frame1p)** --
//     NOT the TGS twin's (ra - rb) + relTr spelling; the rounding differs.
// 41. **Each friction row stores bias = t.dot(error) * invDt in
//     raXn_bias.w** (the TGS twin stores the raw frictionError; its invDt
//     lives in the friction header's biasCoefficient, which the PGS header
//     does not have).
// 42. Friction rows use the RAW solverOffsetSlop (normal rows the scaled
//     one) -- shared with the TGS twin, swept once more here.
// 43. perPointFriction reads frictionPatch.contactID[j] for the
//     target-velocity contact index, else contact 0.
// 44. **NO torsional friction in PGS**: anchorCount = aCount * 2 exactly,
//     no torsional row, and the work unit's mTorsionalFrictionData is never
//     read by the PGS entry.  (No minPen tracking either -- it only fed the
//     TGS torsional scale.)
// 45. The engine dt mapping (:1384) is (invDtF32, dt, invDtF32, dt), and
//     the PGS setup body reads only the first two -- invTotalDt, totalDt,
//     biasCoefficient, ccdMaxSeparationThreshold and torsionalFrictionData
//     are dead parameters, dropped here with this note.
// ===========================================================================

// mirrors PxgBlockSolverContactHeader (1920 bytes, PxgConstraintBlock.h:45)
struct AcpPgsContactHeader
{
    float4 invMass0_1_angDom0_1[32];  // 0
    float4 normal_staticFriction[32]; // 512
    float accumNormalForce[32];       // 1024 (solver-side; prep never writes)
    float minNormalForce[32];         // 1152
    uint flags[32];                   // 1280
    uint numNormalConstr[32];         // 1408
    uint forceWritebackOffset[32];    // 1536
    float restitution[32];            // 1664
    float cfm[32];                    // 1792
}                                     // = 1920

// mirrors PxgBlockSolverFrictionHeader (1408 bytes)
struct AcpPgsFrictionHeader
{
    float4 frictionNormals0[32];  // 0
    float4 frictionNormals1[32];  // 512
    uint numFrictionConstr[32];   // 1024
    float dynamicFriction[32];    // 1152
    uint broken[32];              // 1280
}                                 // = 1408

// mirrors PxgBlockSolverContactPoint (1664 bytes)
struct AcpPgsContactPoint
{
    float4 raXn_targetVelocity[32]; // 0
    float4 rbXn_maxImpulse[32];     // 512
    float appliedForce[32];         // 1024
    float resp0[32];                // 1152
    float resp1[32];                // 1280
    float coeff0[32];               // 1408
    float coeff1[32];               // 1536
}                                   // = 1664

// mirrors PxgBlockSolverContactFriction (1408 bytes)
struct AcpPgsContactFriction
{
    float4 raXn_bias[32];       // 0
    float4 rbXn_targetVelW[32]; // 512
    float appliedForce[32];     // 1024
    float resp0[32];            // 1152
    float resp1[32];            // 1280
}                               // = 1408

// mirrors computeCompliantContactCoefficients (constraintPrepShared.cuh:274)
// -- the FULL PGS form.  The rigid-PGS prep gets away with the reduced
// queryReducedCompliantContactCoefficients; the arti path cannot, because
// velMultiplier feeds the minNormalForce deltaF (trap 38).
float acpComputeCompliantContactCoefficientsPGS(
    float dt, uint flags, float restitution, float damping, float unitResponse,
    float recipResponse, float penetration, float targetVelocity, bool isSeparated,
    bool collidingWithVrel, inout float velMultiplier, out float impulseMultiplier,
    out float unbiasedErr, out float biasedErr)
{
    const bool accelSpring = (flags & PXG_SC_FLAG_COMPLIANT_ACCELERATION_SPRING) != 0u;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;

    const float nrdt = dt * restitution;
    const float dampingIfEnabled = computeCompliantDamping(isSeparated, collidingWithVrel, damping);
    const float a = dt * (dampingIfEnabled - nrdt);
    const float b = -(nrdt * penetration * massIfAccelElseOne);
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
    const float scaledBias = x * b;

    velMultiplier = x * a * massIfAccelElseOne;
    impulseMultiplier = 1.0f - x;
    biasedErr = targetVelocity * velMultiplier - scaledBias;
    unbiasedErr = biasedErr;

    return a;
}

uint acpSetupFinalizeExtPGS(
    Ptr<PxgBlockContactData> contactData, uint cdi,
    Ptr<PxgBlockContactPoint> contacts, uint contactBase,
    uint contactCount,
    Ptr<PxgBlockFrictionPatch> frictionPatch, uint fpi,
    Ptr<PxgBlockFrictionAnchorPatch> fAnchor, uint fai,
    AcpBody b0, AcpBody b1,
    float invDt, float dt, float bounceThresholdF32,
    uint threadIndex, uint forceWritebackBufferOffset, bool perPointFriction,
    Ptr<AcpPgsContactHeader> contactHeader, uint chi,
    Ptr<AcpPgsFrictionHeader> frictionHeader, uint fhi,
    Ptr<AcpPgsContactPoint> contactPoints, uint cpBase,
    Ptr<AcpPgsContactFriction> frictions, uint frBase,
    Ptr<PxgArticulationBlockResponse> responses, uint respBase,
    float solverOffsetSlop)
{
    const float4 cd4 = contactData[cdi].contactData[threadIndex];
    const uint packedW = asuint(cd4.w);
    const uint flags = (packedW >> 8) & 0xFFu;   // mSolverFlags
    const float restDistance = cd4.x;
    const float staticFrictionIn = cd4.y;
    const float dynamicFrictionIn = cd4.z;

    const float3 bodyFrame0p = b0.b2wP;
    const float3 bodyFrame1p = b1.b2wP;

    const float maxPenBias = max(b0.penBiasClamp, b1.penBiasClamp);
    const float p8 = 0.8f;   // trap 29: FIXED; biasCoefficient is dead here
    const float bounceThreshold = bounceThresholdF32;
    const float invDtp8 = invDt * p8;

    const float4 lin0X_ang0Y_lin1Z_ang1W = contactData[cdi].mInvMassScale[threadIndex];

    const float3 linVel0 = b0.velocity.bottom;
    const float3 linVel1 = b1.velocity.bottom;
    const float3 angVel0 = b0.velocity.top;
    const float3 angVel1 = b1.velocity.top;

    const float cfm = (b0.cfm > b1.cfm) ? b0.cfm : b1.cfm;   // P9.86c: PxMax = selectMax a>b?a:b (artiConstraintPrep2.cu:232); FMax gives +0 for (+0,-0)

    uint responsesWritten = 0u;

    const uint aCount = frictionPatch[fpi].anchorCount[threadIndex];
    const uint anchorCount = aCount * 2u;   // trap 44: no torsional row

    if (contactCount == 0u)
    {
        // trap 28: exactly these three, then out
        frictionHeader[fhi].numFrictionConstr[threadIndex] = 0u;
        contactHeader[chi].numNormalConstr[threadIndex] = 0u;
        contactHeader[chi].forceWritebackOffset[threadIndex] = 0xffffffffu;
        return 0u;
    }

    const float maxImpulse = (b0.maxImpulse < b1.maxImpulse) ? b0.maxImpulse : b1.maxImpulse;   // P9.86: PxMin select (artiConstraintPrep2.cu:249, NaN parity)

    contactHeader[chi].forceWritebackOffset[threadIndex] = forceWritebackBufferOffset;
    contactHeader[chi].flags[threadIndex] = flags;
    // the invMassScale swizzle, shared with the TGS twin (trap 10)
    contactHeader[chi].invMass0_1_angDom0_1[threadIndex] =
        float4(lin0X_ang0Y_lin1Z_ang1W.x, lin0X_ang0Y_lin1Z_ang1W.z,
               lin0X_ang0Y_lin1Z_ang1W.y, lin0X_ang0Y_lin1Z_ang1W.w);

    const float4 normal4_rW = contactData[cdi].normal_restitutionW[threadIndex];
    const float3 normal = normal4_rW.xyz;
    const float restitution = normal4_rW.w;
    const float damping = contactData[cdi].damping[threadIndex];
    const float norVel0 = dot(normal, linVel0);
    const float norVel1 = dot(normal, linVel1);
    const float linNorVel = norVel0 - norVel1;

    // trap 30
    contactHeader[chi].restitution[threadIndex] = restitution;
    contactHeader[chi].cfm[threadIndex] = cfm;

    float normalForce = 0.0f;

    for (uint j = 0u; j < contactCount; ++j)
    {
        const float4 point4_separationW = contacts[contactBase + j].point_separationW[threadIndex];
        const float3 point = point4_separationW.xyz;
        const float separation = point4_separationW.w;

        const float4 targetVel4_maxImpulseW =
            contacts[contactBase + j].targetVel_maxImpulseW[threadIndex];
        const float3 targetVel = targetVel4_maxImpulseW.xyz;
        const float cTargetVel = dot(normal, targetVel);

        const float3 ra = point - bodyFrame0p;
        const float3 rb = point - bodyFrame1p;

        float3 raXn = cross(ra, normal);
        float3 rbXn = cross(rb, normal);

        const float angNorVel = dot(raXn, angVel0) - dot(rbXn, angVel1);

        // the scaled slop (shared with the TGS twin, trap 12)
        const float slop = solverOffsetSlop
                         * max(linNorVel == 0.0f ? 1.0f : angNorVel / linNorVel, 1.0f);
        raXn.x = abs(raXn.x) < slop ? 0.0f : raXn.x;
        raXn.y = abs(raXn.y) < slop ? 0.0f : raXn.y;
        raXn.z = abs(raXn.z) < slop ? 0.0f : raXn.z;
        rbXn.x = abs(rbXn.x) < slop ? 0.0f : rbXn.x;
        rbXn.y = abs(rbXn.y) < slop ? 0.0f : rbXn.y;
        rbXn.z = abs(rbXn.z) < slop ? 0.0f : rbXn.z;

        const AbSpatialVector response0 = acpCreateImpulseResponseVector(b0, asmSVec(raXn, normal));
        const AbSpatialVector response1 =
            acpCreateImpulseResponseVector(b1, asmSVec(-rbXn, -normal));

        const AbSpatialVector deltaV0 = acpGetImpulseResponse(
            b0, asmSVec(normal * lin0X_ang0Y_lin1Z_ang1W.x, raXn * lin0X_ang0Y_lin1Z_ang1W.y));
        const AbSpatialVector deltaV1 = acpGetImpulseResponse(
            b1, asmSVec(-normal * lin0X_ang0Y_lin1Z_ang1W.z, -rbXn * lin0X_ang0Y_lin1Z_ang1W.w));

        const float resp0 = dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
        const float resp1 = dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);
        const float unitResponse = resp0 + resp1;

        const float vrel1 = norVel0 + acpProjectAngular(b0, raXn);
        const float vrel2 = norVel1 + acpProjectAngular(b1, rbXn);
        const float vrel = vrel1 - vrel2;

        const float penetration = separation - restDistance;
        const bool isSeparated = penetration >= 0.0f;   // trap 31: >=
        // trap 32: the SAME invDt, and the p8 arm clamps at maxPenBias
        const float penetrationInvDt = penetration * invDt;
        const float penetrationInvDtPt8 = max(maxPenBias, penetration * invDtp8);

        const bool collidingWithVrel = (-vrel) > penetrationInvDt;
        // trap 33: > 0, not > 1e-5
        const bool isGreater2 = (restitution > 0.0f) && (bounceThreshold > vrel)
                             && collidingWithVrel;

        // trap 36: isCCD is the engine's own hard-coded false, so this
        // compare is constant-true and the gate reduces to isGreater2
        const bool ccdSeparationCondition = ACP_MAX_F32 >= penetration;

        // trap 34: the PARENTHESIZED form -- no precedence quirk in PGS
        // P9.86f: equivalent as spelled -- the literal 0.0f reaches the add only as a select arm on a
        // runtime condition, which llvmpipe does not fold (p86SignedZero rows 23-25): IEEE `cTargetVel + 0`, as the CUDA (artiConstraintPrep2.cu:352).
        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.0f);

        // trap 38: saved BEFORE the adjust
        const float tvel = targetVelocity;

        // trap 35: ANY rigid, not just kinematic
        if (!nodeIndexIsArticulation(b0.islandNodeIndex))
            targetVelocity -= vrel1;
        if (!nodeIndexIsArticulation(b1.islandNodeIndex))
            targetVelocity += vrel2;

        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / (unitResponse + cfm)) : 0.0f;
        float velMultiplier = recipResponse;

        float coeff0;
        float coeff1;
        if (restitution < 0.0f)   // compliant contact
        {
            float impulseMultiplier;
            float unbiasedErr;
            float biasedErr;
            // trap 37: coeff0 = a (the return), velMultiplier updated
            coeff0 = acpComputeCompliantContactCoefficientsPGS(
                dt, flags, restitution, damping, unitResponse, recipResponse,
                penetration, targetVelocity, isSeparated, collidingWithVrel,
                velMultiplier, impulseMultiplier, unbiasedErr, biasedErr);

            const float nrdt = dt * restitution;
            coeff1 = -nrdt * penetration;
        }
        else
        {
            float scaledBias = isSeparated ? penetrationInvDt : penetrationInvDtPt8;
            if (ccdSeparationCondition && isGreater2)
                scaledBias = 0.0f;

            float unbiasedErr = targetVelocity;
            const float biasedErr = unbiasedErr - scaledBias;

            if (!isGreater2)
                unbiasedErr -= max(0.0f, scaledBias);

            coeff0 = biasedErr;
            coeff1 = unbiasedErr;
        }

        // trap 39: the PGS row layout
        contactPoints[cpBase + j].raXn_targetVelocity[threadIndex] =
            float4(response0.top, targetVelocity);
        // P9.86: PxMin as its select (artiConstraintPrep2.cu:397; see the TGS twin)
        contactPoints[cpBase + j].rbXn_maxImpulse[threadIndex] =
            float4(-response1.top, (maxImpulse < targetVel4_maxImpulseW.w) ? maxImpulse : targetVel4_maxImpulseW.w);
        contactPoints[cpBase + j].resp0[threadIndex] = resp0;
        contactPoints[cpBase + j].resp1[threadIndex] = resp1;
        contactPoints[cpBase + j].coeff0[threadIndex] = coeff0;
        contactPoints[cpBase + j].coeff1[threadIndex] = coeff1;
        contactPoints[cpBase + j].appliedForce[threadIndex] = 0.0f;

        // trap 38: the PRE-adjust tvel and the max(penInvDt, 0) spelling
        const float deltaF =
            max((tvel - (vrel + max(penetrationInvDt, 0.0f))) * velMultiplier, 0.0f);
        normalForce += deltaF;

        const uint r = respBase + responsesWritten;
        responses[r].deltaRALin_x[threadIndex] = deltaV0.bottom.x;
        responses[r].deltaRALin_y[threadIndex] = deltaV0.bottom.y;
        responses[r].deltaRALin_z[threadIndex] = deltaV0.bottom.z;
        responses[r].deltaRAAng_x[threadIndex] = deltaV0.top.x;
        responses[r].deltaRAAng_y[threadIndex] = deltaV0.top.y;
        responses[r].deltaRAAng_z[threadIndex] = deltaV0.top.z;
        responses[r].deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
        responses[r].deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
        responses[r].deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
        responses[r].deltaRBAng_x[threadIndex] = deltaV1.top.x;
        responses[r].deltaRBAng_y[threadIndex] = deltaV1.top.y;
        responses[r].deltaRBAng_z[threadIndex] = deltaV1.top.z;
        responsesWritten++;
    }

    contactHeader[chi].minNormalForce[threadIndex] = (normalForce / float(contactCount)) * 0.25f;

    const float frictionScale = (aCount == 2u) ? 0.5f : 1.0f;
    const float staticFriction = staticFrictionIn * frictionScale;
    const float dynamicFriction = dynamicFrictionIn * frictionScale;

    const bool haveFriction = anchorCount != 0u;
    contactHeader[chi].numNormalConstr[threadIndex] = contactCount;
    frictionHeader[fhi].numFrictionConstr[threadIndex] = anchorCount;

    contactHeader[chi].normal_staticFriction[threadIndex] = float4(normal, staticFriction);
    frictionHeader[fhi].dynamicFriction[threadIndex] = dynamicFriction;

    if (haveFriction)
    {
        const float3 vrel3 = linVel0 - linVel1;

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        const float3 t0Fallback1 = float3(0.0f, -normal.z, normal.y);
        const float3 t0Fallback2 = float3(-normal.y, normal.x, 0.0f);
        const float3 t0Fallback = orthoThreshold > abs(normal.x) ? t0Fallback1 : t0Fallback2;

        float3 t0 = vrel3 - normal * dot(normal, vrel3);
        t0 = dot(t0, t0) > p1 ? t0 : t0Fallback;
        // the explicit recip-multiply spelling (see the TGS twin's note)
        t0 = t0 * (1.0f / sqrt(dot(t0, t0)));

        const float3 t1 = cross(normal, t0);

        frictionHeader[fhi].broken[threadIndex] = 0u;
        frictionHeader[fhi].frictionNormals0[threadIndex] = float4(t0, 0.0f);
        frictionHeader[fhi].frictionNormals1[threadIndex] = float4(t1, 0.0f);

        for (uint j = 0u; j < aCount; ++j)
        {
            const uint f0i = frBase + 2u * j;
            const uint f1i = frBase + 2u * j + 1u;

            const float4 body0Anchor4 = (j == 0u) ? fAnchor[fai].body0Anchors0[threadIndex]
                                                  : fAnchor[fai].body0Anchors1[threadIndex];
            const float4 body1Anchor4 = (j == 0u) ? fAnchor[fai].body1Anchors0[threadIndex]
                                                  : fAnchor[fai].body1Anchors1[threadIndex];

            const float3 ra = quatRotate(b0.b2wQ, body0Anchor4.xyz);
            const float3 rb = quatRotate(b1.b2wQ, body1Anchor4.xyz);

            // trap 40: (ra + p0) - (rb + p1), NOT (ra - rb) + relTr
            const float3 error = (ra + bodyFrame0p) - (rb + bodyFrame1p);

            // trap 43
            const uint index = perPointFriction
                ? ((j == 0u) ? frictionPatch[fpi].contactID0[threadIndex]
                             : frictionPatch[fpi].contactID1[threadIndex])
                : 0u;
            const float4 targetVel4 = contacts[contactBase + index].targetVel_maxImpulseW[threadIndex];
            const float3 tvel = targetVel4.xyz;

            // t0 row
            {
                float3 raXn = cross(ra, t0);
                float3 rbXn = cross(rb, t0);
                raXn.x = abs(raXn.x) < solverOffsetSlop ? 0.0f : raXn.x;
                raXn.y = abs(raXn.y) < solverOffsetSlop ? 0.0f : raXn.y;
                raXn.z = abs(raXn.z) < solverOffsetSlop ? 0.0f : raXn.z;
                rbXn.x = abs(rbXn.x) < solverOffsetSlop ? 0.0f : rbXn.x;
                rbXn.y = abs(rbXn.y) < solverOffsetSlop ? 0.0f : rbXn.y;
                rbXn.z = abs(rbXn.z) < solverOffsetSlop ? 0.0f : rbXn.z;

                const AbSpatialVector response0 =
                    acpCreateImpulseResponseVector(b0, asmSVec(raXn, t0));
                const AbSpatialVector response1 =
                    acpCreateImpulseResponseVector(b1, asmSVec(-rbXn, -t0));
                const AbSpatialVector deltaV0 = acpGetImpulseResponse(
                    b0, asmSVec(t0 * lin0X_ang0Y_lin1Z_ang1W.x, raXn * lin0X_ang0Y_lin1Z_ang1W.y));
                const AbSpatialVector deltaV1 = acpGetImpulseResponse(
                    b1, asmSVec(-t0 * lin0X_ang0Y_lin1Z_ang1W.z, -rbXn * lin0X_ang0Y_lin1Z_ang1W.w));

                const float resp0 =
                    dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
                const float resp1 =
                    dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);

                float targetVelF = dot(tvel, t0);
                // trap 35/42: !isArticulation-keyed, via projectVelocity
                if (!nodeIndexIsArticulation(b0.islandNodeIndex))
                    targetVelF -= acpProjectVelocity(b0, asmSVec(raXn, t0));
                if (!nodeIndexIsArticulation(b1.islandNodeIndex))
                    targetVelF += acpProjectVelocity(b1, asmSVec(rbXn, t0));

                // trap 41
                const float bias = dot(t0, error) * invDt;
                frictions[f0i].raXn_bias[threadIndex] = float4(response0.top, bias);
                frictions[f0i].rbXn_targetVelW[threadIndex] = float4(-response1.top, targetVelF);
                frictions[f0i].appliedForce[threadIndex] = 0.0f;
                frictions[f0i].resp0[threadIndex] = resp0;
                frictions[f0i].resp1[threadIndex] = resp1;

                const uint r = respBase + responsesWritten;
                responses[r].deltaRALin_x[threadIndex] = deltaV0.bottom.x;
                responses[r].deltaRALin_y[threadIndex] = deltaV0.bottom.y;
                responses[r].deltaRALin_z[threadIndex] = deltaV0.bottom.z;
                responses[r].deltaRAAng_x[threadIndex] = deltaV0.top.x;
                responses[r].deltaRAAng_y[threadIndex] = deltaV0.top.y;
                responses[r].deltaRAAng_z[threadIndex] = deltaV0.top.z;
                responses[r].deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
                responses[r].deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
                responses[r].deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
                responses[r].deltaRBAng_x[threadIndex] = deltaV1.top.x;
                responses[r].deltaRBAng_y[threadIndex] = deltaV1.top.y;
                responses[r].deltaRBAng_z[threadIndex] = deltaV1.top.z;
                responsesWritten++;
            }

            // t1 row
            {
                float3 raXn = cross(ra, t1);
                float3 rbXn = cross(rb, t1);
                raXn.x = abs(raXn.x) < solverOffsetSlop ? 0.0f : raXn.x;
                raXn.y = abs(raXn.y) < solverOffsetSlop ? 0.0f : raXn.y;
                raXn.z = abs(raXn.z) < solverOffsetSlop ? 0.0f : raXn.z;
                rbXn.x = abs(rbXn.x) < solverOffsetSlop ? 0.0f : rbXn.x;
                rbXn.y = abs(rbXn.y) < solverOffsetSlop ? 0.0f : rbXn.y;
                rbXn.z = abs(rbXn.z) < solverOffsetSlop ? 0.0f : rbXn.z;

                const AbSpatialVector response0 =
                    acpCreateImpulseResponseVector(b0, asmSVec(raXn, t1));
                const AbSpatialVector response1 =
                    acpCreateImpulseResponseVector(b1, asmSVec(-rbXn, -t1));
                const AbSpatialVector deltaV0 = acpGetImpulseResponse(
                    b0, asmSVec(t1 * lin0X_ang0Y_lin1Z_ang1W.x, raXn * lin0X_ang0Y_lin1Z_ang1W.y));
                const AbSpatialVector deltaV1 = acpGetImpulseResponse(
                    b1, asmSVec(-t1 * lin0X_ang0Y_lin1Z_ang1W.z, -rbXn * lin0X_ang0Y_lin1Z_ang1W.w));

                const float resp0 =
                    dot(deltaV0.top, response0.top) + dot(deltaV0.bottom, response0.bottom);
                const float resp1 =
                    dot(deltaV1.top, response1.top) + dot(deltaV1.bottom, response1.bottom);

                float targetVelF = dot(tvel, t1);
                if (!nodeIndexIsArticulation(b0.islandNodeIndex))
                    targetVelF -= acpProjectVelocity(b0, asmSVec(raXn, t1));
                if (!nodeIndexIsArticulation(b1.islandNodeIndex))
                    targetVelF += acpProjectVelocity(b1, asmSVec(rbXn, t1));

                const float bias = dot(t1, error) * invDt;
                frictions[f1i].raXn_bias[threadIndex] = float4(response0.top, bias);
                frictions[f1i].rbXn_targetVelW[threadIndex] = float4(-response1.top, targetVelF);
                frictions[f1i].appliedForce[threadIndex] = 0.0f;
                frictions[f1i].resp0[threadIndex] = resp0;
                frictions[f1i].resp1[threadIndex] = resp1;

                const uint r = respBase + responsesWritten;
                responses[r].deltaRALin_x[threadIndex] = deltaV0.bottom.x;
                responses[r].deltaRALin_y[threadIndex] = deltaV0.bottom.y;
                responses[r].deltaRALin_z[threadIndex] = deltaV0.bottom.z;
                responses[r].deltaRAAng_x[threadIndex] = deltaV0.top.x;
                responses[r].deltaRAAng_y[threadIndex] = deltaV0.top.y;
                responses[r].deltaRAAng_z[threadIndex] = deltaV0.top.z;
                responses[r].deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
                responses[r].deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
                responses[r].deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
                responses[r].deltaRBAng_x[threadIndex] = deltaV1.top.x;
                responses[r].deltaRBAng_y[threadIndex] = deltaV1.top.y;
                responses[r].deltaRBAng_z[threadIndex] = deltaV1.top.z;
                responsesWritten++;
            }
        }
    }

    return responsesWritten;
}

// ---------------------------------------------------------------------------
// the ENGINE entry -- artiContactConstraintBlockPrepareLaunch (:1380), the
// PGS instantiation of the shared template body (:1201).  Identical to the
// TGS entry above through the body creation and friction machinery; the
// differences are the PGS shared desc, the PGS output block types, and the
// dt mapping (trap 45).  gAcw is reused: the CUDA parameter pack is the
// same two pointers.
// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void artiContactConstraintBlockPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                             uint3 groupId : SV_GroupID)
{
    const uint blockStride = 64u / 32u;
    const uint warpIndex = groupId.x * blockStride + groupThreadId.x / 32u;
    const uint lane = groupThreadId.x & 31u;

    Ptr<PxgConstraintPrepareDescGpu> prepDesc =
        Ptr<PxgConstraintPrepareDescGpu>(gAcw.prepDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescPGSGpu>(gAcw.sharedDesc);

    // trap 20 (shared with the TGS twin)
    const uint nbContactBatchesWithSelf = prepDesc[0].numArtiContactBatches
        + prepDesc[0].numArtiStaticContactBatches + prepDesc[0].numArtiSelfContactBatches;
    const uint i = warpIndex;
    if (i >= nbContactBatchesWithSelf)
        return;

    Ptr<uint> batchIndices = Ptr<uint>(prepDesc[0].artiContactConstraintBatchIndices);
    const uint batchIndex = batchIndices[i] + prepDesc[0].numBatches;

    Ptr<PxgBlockConstraintBatchGpu> batches =
        Ptr<PxgBlockConstraintBatchGpu>(sharedDesc[0].iter_blockConstraintBatch);
    const uint descIndexBatch = batches[batchIndex].mConstraintBatchIndex;
    const uint responseIndex = batches[batchIndex].mArticulationResponseIndex;
    const uint descStride = batches[batchIndex].descStride_constraintType & 0xFFFFu;

    if (lane >= descStride)
        return;

    const uint2 nodeIndexA = batches[batchIndex].bodyANodeIndex[lane];
    const uint2 nodeIndexB = batches[batchIndex].bodyBNodeIndex[lane];

    // trap 21
    const uint bodyAIndex = nodeIndexIsArticulation(nodeIndexA)
        ? batches[batchIndex].remappedBodyAIndex[lane] : batches[batchIndex].bodyAIndex[lane];
    const uint bodyBIndex = nodeIndexIsArticulation(nodeIndexB)
        ? batches[batchIndex].remappedBodyBIndex[lane] : batches[batchIndex].bodyBIndex[lane];

    Ptr<PxgBlockWorkUnit> workUnits = Ptr<PxgBlockWorkUnit>(prepDesc[0].blockWorkUnit);
    Ptr<uint2> currFrictionIndices = Ptr<uint2>(prepDesc[0].blockCurrentFrictionIndices);
    Ptr<uint2> prevFrictionIndices = Ptr<uint2>(prepDesc[0].blockPreviousFrictionIndices);

    // trap 24
    const uint edgeIndex = workUnits[descIndexBatch].mEdgeIndex[lane];
    const uint frictionIndex =
        edgeIndex + prepDesc[0].totalCurrentEdges * workUnits[descIndexBatch].mPatchIndex[lane];
    currFrictionIndices[frictionIndex] =
        uint2((descIndexBatch << 5) | lane, descIndexBatch >> 27);

    AcpBody b0, b1;
    acpCreateSolverExtBody(nodeIndexA, bodyAIndex, sharedDesc[0].articulations,
                           prepDesc[0].solverBodyDataPool, prepDesc[0].solverBodyTxIDataPool,
                           prepDesc[0].body2WorldPool, b0);
    acpCreateSolverExtBody(nodeIndexB, bodyBIndex, sharedDesc[0].articulations,
                           prepDesc[0].solverBodyDataPool, prepDesc[0].solverBodyTxIDataPool,
                           prepDesc[0].body2WorldPool, b1);

    // trap 23
    const float offsetSlop = max(b0.offsetSlop, b1.offsetSlop);

    const uint forceWritebackOffset = workUnits[descIndexBatch].mWriteback[lane];
    // trap 44: mTorsionalFrictionData is NOT read by the PGS entry

    // ---- artiCreateFinalizeSolverContactsBlockGPU (:1114), PGS ----

    Ptr<PxgBlockContactData> contactPrepPool =
        Ptr<PxgBlockContactData>(prepDesc[0].blockContactCurrentPrepPool);

    // trap 22
    const float4 cd4 = contactPrepPool[descIndexBatch].contactData[lane];
    const uint materialWord = asuint(cd4.w);
    const uint nbContacts = materialWord & 0xFFu;
    const uint flags = (materialWord >> 8) & 0xFFu;
    const uint prevFrictionPatchCount = (materialWord >> 16) & 0xFFu;

    const float4 normal4 = contactPrepPool[descIndexBatch].normal_restitutionW[lane];

    // PxAlignedTransform(PxTransform): p.w = 0
    PxgAlignedTransform bodyFrame0;
    bodyFrame0.q = b0.b2wQ;
    bodyFrame0.p = float4(b0.b2wP, 0.0f);
    PxgAlignedTransform bodyFrame1;
    bodyFrame1.q = b1.b2wQ;
    bodyFrame1.p = float4(b1.b2wP, 0.0f);

    Ptr<PxgBlockFrictionPatch> currFrictionPatches =
        Ptr<PxgBlockFrictionPatch>(sharedDesc[0].blockCurrentFrictionPatches);
    Ptr<PxgBlockFrictionPatch> prevFrictionPatches =
        Ptr<PxgBlockFrictionPatch>(sharedDesc[0].blockPreviousFrictionPatches);
    Ptr<PxgBlockFrictionAnchorPatch> currAnchors =
        Ptr<PxgBlockFrictionAnchorPatch>(prepDesc[0].blockCurrentAnchorPatches);
    Ptr<PxgBlockFrictionAnchorPatch> prevAnchors =
        Ptr<PxgBlockFrictionAnchorPatch>(prepDesc[0].blockPreviousAnchorPatches);
    Ptr<PxgBlockContactPoint> contactPrepPoints =
        Ptr<PxgBlockContactPoint>(prepDesc[0].blockContactPoints);
    const uint contactBase = batches[batchIndex].blockContactIndex;

    correlatePatchesPtr(currFrictionPatches, descIndexBatch, nbContacts, normal4.xyz,
                        bodyFrame0, bodyFrame1, lane);

    const bool perPointFriction = (flags & PXG_SC_FLAG_PER_POINT_FRICTION) != 0;
    const bool disableFriction = (flags & PXG_SC_FLAG_DISABLE_FRICTION) != 0;

    currFrictionPatches[descIndexBatch].broken[lane] = 0;
    float patchExtents = 0.0f;

    // trap 25
    if (!(perPointFriction || disableFriction))
    {
        getFrictionPatchesPtr(currFrictionPatches, descIndexBatch, currAnchors, descIndexBatch,
                              prevFrictionIndices, prepDesc[0].blockPreviousFrictionIndices != 0,
                              edgeIndex, prevFrictionPatches, prevAnchors, prevFrictionPatchCount,
                              bodyFrame0, bodyFrame1, prepDesc[0].correlationDistance, lane,
                              prepDesc[0].totalPreviousEdges, patchExtents, nbContacts);
    }

    if (!disableFriction)
        growPatchesPtr(currFrictionPatches, descIndexBatch, currAnchors, descIndexBatch,
                       contactPrepPoints, contactBase, nbContacts, bodyFrame0, bodyFrame1,
                       prepDesc[0].frictionOffsetThreshold + cd4.x, lane, patchExtents);

    Ptr<AcpPgsContactHeader> contactHeaders =
        Ptr<AcpPgsContactHeader>(sharedDesc[0].iter_blockContactHeaders);
    Ptr<AcpPgsFrictionHeader> frictionHeaders =
        Ptr<AcpPgsFrictionHeader>(sharedDesc[0].iter_blockFrictionHeaders);
    Ptr<AcpPgsContactPoint> outContactPoints =
        Ptr<AcpPgsContactPoint>(sharedDesc[0].iter_blockContactPoints);
    Ptr<AcpPgsContactFriction> outFrictions =
        Ptr<AcpPgsContactFriction>(sharedDesc[0].iter_blockFrictions);
    Ptr<PxgArticulationBlockResponse> responses =
        Ptr<PxgArticulationBlockResponse>(sharedDesc[0].iter_artiResponse);

    // trap 45: (invDtF32, dt) -- the PGS setup reads no other dt
    acpSetupFinalizeExtPGS(
        contactPrepPool, descIndexBatch,
        contactPrepPoints, contactBase, nbContacts,
        currFrictionPatches, descIndexBatch,
        currAnchors, descIndexBatch,
        b0, b1,
        sharedDesc[0].invDtF32, sharedDesc[0].dt,
        prepDesc[0].bounceThresholdF32,
        lane, forceWritebackOffset, perPointFriction,
        contactHeaders, descIndexBatch,
        frictionHeaders, descIndexBatch,
        outContactPoints, batches[batchIndex].startConstraintIndex,
        outFrictions, batches[batchIndex].startFrictionIndex,
        responses, responseIndex,
        offsetSlop);

    // trap 26
    currFrictionPatches[descIndexBatch].patchIndex[lane] =
        workUnits[descIndexBatch].mFrictionPatchIndex[lane];

    if (currFrictionPatches[descIndexBatch].anchorCount[lane] >= 1u)
    {
        const float3 a0 = currAnchors[descIndexBatch].body0Anchors0[lane].xyz;
        currFrictionPatches[descIndexBatch].anchorPoints0[lane] =
            float4(quatRotate(b0.b2wQ, a0) + b0.b2wP, 0.0f);
    }
    if (currFrictionPatches[descIndexBatch].anchorCount[lane] == 2u)
    {
        const float3 a1 = currAnchors[descIndexBatch].body0Anchors1[lane].xyz;
        currFrictionPatches[descIndexBatch].anchorPoints1[lane] =
            float4(quatRotate(b0.b2wQ, a1) + b0.b2wP, 0.0f);
    }
}
