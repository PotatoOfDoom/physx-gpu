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

// npArtiExtBody.slang
//
// Slang port of the articulation-constraint-prep body foundation of
// gpusolver/src/CUDA/artiConstraintPrep2.cu (L55-165):
//   - createPxgSolverExtBody: the PxgSolverExtBody2 packing -- articulation
//     branch (link pose/velocity/spatial response/penBiasClamp/cfm/
//     offsetSlop, maxImpulse = PX_MAX_F32, linkIndex = link id) vs rigid
//     branch (static defaults: zero velocity, penBiasClamp = -PX_MAX_F32,
//     invMass 0; kinematic gate = PxRigidBodyFlag::eKINEMATIC(1); the
//     spatial response ZEROED for static/kinematic bodies, otherwise
//     PxSpatialMatrix::initialize(sqrtInvInertia, invMass)
//     (PxSpatialMatrix.h L101-127: mass diagonal in columns 0-2 rows 3-5,
//     sqrtInvInertia columns in columns 3-5 rows 0-2), linkIndex =
//     PxSolverConstraintDesc::RIGID_BODY = 0xffff (PxSolverDefs.h L113),
//     cfm = 0);
//   - createImpulseResponseVector: passthrough for articulations/kinematics,
//     else (multiplyInertia(top), bottom) with multiplyInertia reading
//     columns 3-5's first three entries (PxSpatialMatrix.h L278-284);
//   - projectVelocity = velocity.dot(v) (spatial dot = top.top + bottom.bottom,
//     CmSpatialVector.h L100-103);
//   - projectAngular = velocity.top.dot(v);
//   - getImpulseResponse = mSpatialResponse * v (the 6x6 column product,
//     result[i] = sum over j of column[j][i] * st[j] accumulated in j order,
//     PxSpatialMatrix.h L57-77).
// The node-index decode (isArticulation/linkId/isStaticBody) and the
// solver-body/articulation memory fetches are plumbing supplied by the
// fixtures.  ALL outputs bitwise (float arith is NoContraction-stamped).

static const float FLT_MAXQ = 3.402823466e+38f;

struct EbIn
{
    uint4 meta;        // x = isArticulation, y = linkID (arti) / isStatic (rigid),
                       // z = solver-body flags word, w = bodyIndex
    float4 artQ;       // articulation link body2World quat
    float4 artP;       // link body2World pos, w = penBiasClamp (initialAngVelXYZ_penBiasClamp.w)
    float4 artVelTop;  // link motion velocity top, w = cfm
    float4 artVelBot;  // link motion velocity bottom, w = offsetSlop
    float4 artResp[9]; // spatialResponseMatrixW: column[j][i] at flat j*6+i
    float4 rbLin;      // solverBody initialLinVelXYZ_invMassW
    float4 rbAng;      // solverBody initialAngVelXYZ_penBiasClamp
    float4 rbFrameQ;   // bodyFrames[solverBodyIndex] quat
    float4 rbFrameP;   // bodyFrames pos
    float4 rbInertia[3];  // txIData sqrtInvInertia columns
    float4 rbMisc;     // x = solverBody maxImpulse, y = solverBody offsetSlop
    float4 impTop;     // impulse for createImpulseResponseVector / projectVelocity
    float4 impBot;
    float4 vecTop;     // vector for getImpulseResponse
    float4 vecBot;
    float4 pav;        // vector for projectAngular
}

struct EbOut
{
    float4 resp[9];    // packed mSpatialResponse
    float4 velTop;     // packed velocity top, w = penBiasClamp
    float4 velBot;     // packed velocity bottom, w = maxImpulse
    float4 b2wQ;       // packed body2World quat
    float4 b2wP;       // packed body2World pos, w = cfm
    uint4 meta2;       // x = linkIndex, y = isKinematic, z = bodyIndex, w = 0
    float4 misc;       // x = offsetSlop, y = projectVelocity, z = projectAngular, w = 0
    float4 respVecTop; // createImpulseResponseVector(imp)
    float4 respVecBot;
    float4 impRespTop; // getImpulseResponse(vec)
    float4 impRespBot;
}

struct EbParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<EbParams> gParams;

RWStructuredBuffer<EbIn> gIn;    // binding 0
RWStructuredBuffer<EbOut> gOut;  // binding 1

float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// column[j][i] flattened at j*6+i over resp[9] float4 lanes
float respAt(float4 m[9], uint j, uint i)
{
    const uint f = j * 6u + i;
    return m[f / 4u][f & 3u];
}

void respSet(inout float4 m[9], uint j, uint i, float v)
{
    const uint f = j * 6u + i;
    m[f / 4u][f & 3u] = v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aebBody(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const EbIn s = gIn[tid];

    const bool isArticulation = s.meta.x != 0u;

    // ---- createPxgSolverExtBody (cu L55-134) ----
    float4 resp[9];
    float3 velTop, velBot, b2wP;
    float4 b2wQ;
    float penBiasClamp, maxImpulse, cfm, offsetSlop;
    uint linkIndex;
    bool isKinematic;

    if (isArticulation)
    {
        linkIndex = s.meta.y;
        b2wQ = s.artQ;
        b2wP = s.artP.xyz;
        for (uint f = 0u; f < 9u; ++f)
            resp[f] = s.artResp[f];
        isKinematic = false;
        penBiasClamp = s.artP.w;
        maxImpulse = FLT_MAXQ;  // PX_MAX_F32 (KS - TODO - hook up!)
        velTop = s.artVelTop.xyz;
        velBot = s.artVelBot.xyz;
        cfm = s.artVelTop.w;
        offsetSlop = s.artVelBot.w;
    }
    else
    {
        const bool isStatic = s.meta.y != 0u;
        isKinematic = (s.meta.z & 1u) != 0u;  // PxRigidBodyFlag::eKINEMATIC

        velTop = float3(0, 0, 0);
        velBot = float3(0, 0, 0);
        penBiasClamp = -FLT_MAXQ;
        float invMass = 0.0f;
        offsetSlop = 0.0f;
        if (!isStatic)
        {
            penBiasClamp = s.rbAng.w;
            invMass = s.rbLin.w;
            velTop = s.rbAng.xyz;
            velBot = s.rbLin.xyz;
            offsetSlop = s.rbMisc.y;
        }

        if (isStatic || isKinematic)
        {
            for (uint f = 0u; f < 9u; ++f)
                resp[f] = float4(0, 0, 0, 0);
        }
        else
        {
            // PxSpatialMatrix::initialize(sqrtInvInertia, invMass)
            for (uint f = 0u; f < 9u; ++f)
                resp[f] = float4(0, 0, 0, 0);
            respSet(resp, 0u, 3u, invMass);
            respSet(resp, 1u, 4u, invMass);
            respSet(resp, 2u, 5u, invMass);
            for (uint c = 0u; c < 3u; ++c)
            {
                const float3 col = s.rbInertia[c].xyz;
                respSet(resp, 3u + c, 0u, col.x);
                respSet(resp, 3u + c, 1u, col.y);
                respSet(resp, 3u + c, 2u, col.z);
            }
        }

        linkIndex = 0xFFFFu;  // PxSolverConstraintDesc::RIGID_BODY
        b2wQ = s.rbFrameQ;
        b2wP = s.rbFrameP.xyz;
        maxImpulse = s.rbMisc.x;
        cfm = 0.0f;
    }

    // ---- createImpulseResponseVector (cu L136-150) ----
    float3 rvTop, rvBot;
    if (isArticulation || isKinematic)
    {
        rvTop = s.impTop.xyz;
        rvBot = s.impBot.xyz;
    }
    else
    {
        // multiplyInertia: rows from columns 3-5's first three entries
        const float3 v = s.impTop.xyz;
        rvTop = float3(
            respAt(resp, 3u, 0u) * v.x + respAt(resp, 3u, 1u) * v.y + respAt(resp, 3u, 2u) * v.z,
            respAt(resp, 4u, 0u) * v.x + respAt(resp, 4u, 1u) * v.y + respAt(resp, 4u, 2u) * v.z,
            respAt(resp, 5u, 0u) * v.x + respAt(resp, 5u, 1u) * v.y + respAt(resp, 5u, 2u) * v.z);
        rvBot = s.impBot.xyz;
    }

    // ---- projectVelocity (cu L152-155): spatial dot top+bottom ----
    const float projVel = dot3Q(velTop, s.impTop.xyz) + dot3Q(velBot, s.impBot.xyz);

    // ---- projectAngular (cu L157-160) ----
    const float projAng = dot3Q(velTop, s.pav.xyz);

    // ---- getImpulseResponse (cu L162-165): 6x6 column product ----
    float st[6];
    st[0] = s.vecTop.x; st[1] = s.vecTop.y; st[2] = s.vecTop.z;
    st[3] = s.vecBot.x; st[4] = s.vecBot.y; st[5] = s.vecBot.z;
    float result[6];
    for (uint i = 0u; i < 6u; ++i)
    {
        result[i] = 0.0f;
        for (uint j = 0u; j < 6u; ++j)
            result[i] += respAt(resp, j, i) * st[j];
    }

    EbOut o;
    for (uint f = 0u; f < 9u; ++f)
        o.resp[f] = resp[f];
    o.velTop = float4(velTop, penBiasClamp);
    o.velBot = float4(velBot, maxImpulse);
    o.b2wQ = b2wQ;
    o.b2wP = float4(b2wP, cfm);
    o.meta2 = uint4(linkIndex, isKinematic ? 1u : 0u, s.meta.w, 0u);
    o.misc = float4(offsetSlop, projVel, projAng, 0.0f);
    o.respVecTop = float4(rvTop, 0.0f);
    o.respVecBot = float4(rvBot, 0.0f);
    o.impRespTop = float4(result[0], result[1], result[2], 0.0f);
    o.impRespBot = float4(result[3], result[4], result[5], 0.0f);
    gOut[tid] = o;
}
