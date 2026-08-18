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
import pxgSlangPrepCommon;   // PX_RIGID_BODY_FLAG_KINEMATIC
import pxgSlangSolverBlocks; // PxgTGSBlockSolver* + PxgArticulationBlockResponse

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

    const float cfm = max(b0.cfm, b1.cfm);
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

    const float maxImpulse = min(b0.maxImpulse, b1.maxImpulse);

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

        minPen = min(penetration, minPen);

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
        contactPoints[cpBase + j].maxImpulse[threadIndex] =
            min(maxImpulse, targetVel4_maxImpulseW.w);
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

            const float torsionalScale = max(torsionalFrictionData.y,
                                             sqrt(max(0.0f, -minPen) * torsionalFrictionData.x));
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
