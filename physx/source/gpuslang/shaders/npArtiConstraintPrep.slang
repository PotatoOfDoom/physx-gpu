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
