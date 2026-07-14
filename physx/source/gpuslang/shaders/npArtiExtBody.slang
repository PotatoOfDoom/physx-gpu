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

float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// PxQuat::rotate (PxQuat.h L286-295)
float3 quatRotQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

float pxMaxQ(float a, float b) { return (a < b) ? b : a; }
float pxMinQ(float a, float b) { return (a < b) ? a : b; }

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
        rvTop = mulInertiaQ(resp, s.impTop.xyz);
        rvBot = s.impBot.xyz;
    }

    // ---- projectVelocity (cu L152-155): spatial dot top+bottom ----
    const float projVel = dot3Q(velTop, s.impTop.xyz) + dot3Q(velBot, s.impBot.xyz);

    // ---- projectAngular (cu L157-160) ----
    const float projAng = dot3Q(velTop, s.pav.xyz);

    // ---- getImpulseResponse (cu L162-165): 6x6 column product ----
    float3 irTop, irBot;
    spatialMulQ(resp, s.vecTop.xyz, s.vecBot.xyz, irTop, irBot);

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
    o.impRespTop = float4(irTop, 0.0f);
    o.impRespBot = float4(irBot, 0.0f);
    gOut[tid] = o;
}

// ============================================================================
// entry 2: setupFinalizeExtSolverConstraintsBlock, PGS variant (tranche 180)
// (artiConstraintPrep2.cu L169-613) -- the arti contact finalize core.
// The per-lane block-array loads/stores are SoA plumbing; the ported compute
// is everything between them: maxPenBias/cfm/invDtp8, the early-out, the
// invMass lane reordering (x,z,y,w), the per-contact loop (dynamic offset
// slop, response/deltaV pairs through the tranche-179 helpers, unitResponse,
// vrel decomposition, penetration/bounce/CCD gates, the compliant-contact
// coefficients of constraintPrepShared.cuh L273-296 with
// computeCompliantDamping L178-182, the non-articulated targetVelocity
// bias, deltaF/minNormalForce) and the friction block (anchorCount*2 rows,
// frictionScale 0.5 at 2 anchors, the t0 orthobasis with the 0.70710678
// fallback split and PxVec3::normalize's guarded reciprocal-multiply,
// t1 = normal x t0, per-anchor world anchors/error, constant offset slop,
// per-row response/deltaV/resp pairs, target velocities with the
// projectVelocity bias, bias = t.dot(error) * invDt).  isCCD is
// hard-false upstream (cu L213), so ccdMaxSeparation = PX_MAX_F32.
// CUDA leaves all block outputs UNWRITTEN on the contactCount == 0
// early-out except the three header stores -- the port zero-fills and sets
// a writeMask instead (documented deviation; consumers gate on the counts).
// appliedForce is a constant 0 store on both sides (covered by zero-fill).
// ============================================================================

static const uint MAX_CTS = 6u;
static const uint MAX_ANCH = 2u;

struct ExtBodyQ
{
    float4 resp[9];
    float3 velTop; float3 velBot;
    float4 q; float3 p;
    float penBiasClamp, maxImpulse, cfm;
    uint isArticulation, isKinematic;
};

// createImpulseResponseVector over the packed body (cu L136-150)
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

float projVelQ(ExtBodyQ b, float3 vTop, float3 vBot)
{
    return dot3Q(b.velTop, vTop) + dot3Q(b.velBot, vBot);
}

float projAngQ(ExtBodyQ b, float3 v) { return dot3Q(b.velTop, v); }

// PxVec3::normalize: guarded reciprocal-multiply (PxVec3.h)
float3 normalizeQ(float3 v)
{
    const float m = sqrt(dot3Q(v, v));
    if (m > 0.0f)
    {
        const float inv = 1.0f / m;
        return v * inv;
    }
    return v;
}

// computeCompliantDamping (constraintPrepShared.cuh L178-182)
float compliantDampingQ(bool isSeparated, bool collidingWithVrel, float damping)
{
    return (isSeparated && !collidingWithVrel) ? 0.0f : damping;
}

// computeCompliantContactCoefficients (constraintPrepShared.cuh L273-296);
// returns a; velMultiplier out (impulseMultiplier/errs unused downstream here)
float compliantCoeffsQ(float dt, uint flags, float restitution, float damping, float unitResponse,
                       float recipResponse, float penetration, float targetVelocity,
                       bool isSeparated, bool collidingWithVrel, out float velMultiplier)
{
    const bool accelSpring = (flags & 8u) != 0u;  // eCOMPLIANT_ACCELERATION_SPRING
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;

    const float nrdt = dt * restitution;
    const float dampingIfEnabled = compliantDampingQ(isSeparated, collidingWithVrel, damping);
    const float a = dt * (dampingIfEnabled - nrdt);
    const float b = -(nrdt * penetration * massIfAccelElseOne);
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
    velMultiplier = x * a * massIfAccelElseOne;
    return a;
}

struct FzIn
{
    uint4 bMeta;        // b0 isArt, b0 isKin, b1 isArt, b1 isKin
    float4 b0resp[9];
    float4 b0velTop;    // w = penBiasClamp
    float4 b0velBot;    // w = maxImpulse
    float4 b0q;
    float4 b0p;         // w = cfm
    float4 b1resp[9];
    float4 b1velTop;
    float4 b1velBot;
    float4 b1q;
    float4 b1p;
    uint4 cfgU;         // solverFlags, contactCount, aCount, perPointFriction
    float4 invMassScale;  // lin0X_ang0Y_lin1Z_ang1W
    float4 normalRest;  // normal.xyz, restitution
    float4 mat;         // restDistance, damping, staticFriction, dynamicFriction
    float4 scal;        // invDt, dt, bounceThreshold, solverOffsetSlop
    float4 cPts[12];    // per contact: point_separationW, targetVel_maxImpulseW
    float4 anchors0[2];
    float4 anchors1[2];
    uint4 contactIDs;   // contactID[0], contactID[1]
}

struct FzOut
{
    uint4 hdrU;         // forceWritebackOffset, flags, numNormalConstr, numFrictionConstr
    uint4 writeMask;    // x: 1 = early-out, 2 = main block, 4 = friction block
    float4 invMassOut;  // invMass0_1_angDom0_1 (x, z, y, w reorder)
    float4 normalSF;    // normal_staticFriction
    float4 hdrF;        // restitution, cfm, minNormalForce, dynamicFriction
    float4 fricN[2];    // frictionNormals t0, t1
    float4 cOut[18];    // per contact: raXn_targetVelocity, rbXn_maxImpulse,
                        //   (resp0, resp1, coeff0, coeff1)
    float4 artiC[18];   // per contact: (dRALin, dRAAng.x), (dRAAng.yz, dRBLin.xy),
                        //   (dRBLin.z, dRBAng) -- 12 floats packed as 3 float4
    float4 fOut[12];    // per friction row (a0t0, a0t1, a1t0, a1t1):
                        //   raXn_bias, rbXn_targetVelW, (resp0, resp1, 0, 0)
    float4 artiF[12];   // per friction row: 12 floats packed as 3 float4
}

RWStructuredBuffer<FzIn> gIn2;    // binding 2
RWStructuredBuffer<FzOut> gOut2;  // binding 3

ExtBodyQ loadBody(uint isArt, uint isKin, float4 resp[9], float4 velTop, float4 velBot, float4 q,
                  float4 p)
{
    ExtBodyQ b;
    for (uint f = 0u; f < 9u; ++f)
        b.resp[f] = resp[f];
    b.velTop = velTop.xyz;
    b.velBot = velBot.xyz;
    b.q = q;
    b.p = p.xyz;
    b.penBiasClamp = velTop.w;
    b.maxImpulse = velBot.w;
    b.cfm = p.w;
    b.isArticulation = isArt;
    b.isKinematic = isKin;
    return b;
}

void packArti(inout float4 arr[18], uint slot, float3 dALin, float3 dAAng, float3 dBLin,
              float3 dBAng)
{
    arr[slot * 3u + 0u] = float4(dALin, dAAng.x);
    arr[slot * 3u + 1u] = float4(dAAng.y, dAAng.z, dBLin.x, dBLin.y);
    arr[slot * 3u + 2u] = float4(dBLin.z, dBAng.x, dBAng.y, dBAng.z);
}

void packArtiF(inout float4 arr[12], uint slot, float3 dALin, float3 dAAng, float3 dBLin,
               float3 dBAng)
{
    arr[slot * 3u + 0u] = float4(dALin, dAAng.x);
    arr[slot * 3u + 1u] = float4(dAAng.y, dAAng.z, dBLin.x, dBLin.y);
    arr[slot * 3u + 2u] = float4(dBLin.z, dBAng.x, dBAng.y, dBAng.z);
}

float3 slopZero(float3 v, float slop)
{
    return float3(abs(v.x) < slop ? 0.0f : v.x, abs(v.y) < slop ? 0.0f : v.y,
                  abs(v.z) < slop ? 0.0f : v.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aebFinalizePGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const FzIn s = gIn2[tid];

    FzOut o;
    o.hdrU = uint4(0, 0, 0, 0);
    o.writeMask = uint4(0, 0, 0, 0);
    o.invMassOut = float4(0, 0, 0, 0);
    o.normalSF = float4(0, 0, 0, 0);
    o.hdrF = float4(0, 0, 0, 0);
    o.fricN[0] = float4(0, 0, 0, 0);
    o.fricN[1] = float4(0, 0, 0, 0);
    for (uint z = 0u; z < 18u; ++z) { o.cOut[z] = float4(0, 0, 0, 0); o.artiC[z] = float4(0, 0, 0, 0); }
    for (uint z = 0u; z < 12u; ++z) { o.fOut[z] = float4(0, 0, 0, 0); o.artiF[z] = float4(0, 0, 0, 0); }

    const ExtBodyQ b0 = loadBody(s.bMeta.x, s.bMeta.y, s.b0resp, s.b0velTop, s.b0velBot, s.b0q, s.b0p);
    const ExtBodyQ b1 = loadBody(s.bMeta.z, s.bMeta.w, s.b1resp, s.b1velTop, s.b1velBot, s.b1q, s.b1p);

    const uint flags = s.cfgU.x;          // PxU8 mSolverFlags
    const uint contactCount = s.cfgU.y;
    const uint aCount = s.cfgU.z;
    const bool perPointFriction = s.cfgU.w != 0u;
    const float restDistance = s.mat.x;
    const float invDt = s.scal.x;
    const float dt = s.scal.y;
    const float bounceThreshold = s.scal.z;
    const float solverOffsetSlop = s.scal.w;

    // isCCD is hard-false (cu L213)
    const float ccdMaxSeparation = FLT_MAXQ;

    const float maxPenBias = max(b0.penBiasClamp, b1.penBiasClamp);  // fmaxf
    const float invDtp8 = invDt * 0.8f;
    const float4 lin0X_ang0Y_lin1Z_ang1W = s.invMassScale;
    const float cfm = pxMaxQ(b0.cfm, b1.cfm);  // PxMax

    const uint anchorCount = aCount * 2u;

    if (contactCount == 0u)
    {
        // early-out: only the three header stores happen
        o.hdrU = uint4(0xFFFFFFFFu, 0u, 0u, 0u);
        o.writeMask.x = 1u;
        gOut2[tid] = o;
        return;
    }

    const float maxImpulse = pxMinQ(b0.maxImpulse, b1.maxImpulse);  // PxMin

    o.invMassOut = float4(lin0X_ang0Y_lin1Z_ang1W.x, lin0X_ang0Y_lin1Z_ang1W.z,
                          lin0X_ang0Y_lin1Z_ang1W.y, lin0X_ang0Y_lin1Z_ang1W.w);

    const float3 normal = s.normalRest.xyz;
    const float restitution = s.normalRest.w;
    const float damping = s.mat.y;
    const float3 linVel0 = b0.velBot, linVel1 = b1.velBot;
    const float norVel0 = dot3Q(normal, linVel0);
    const float norVel1 = dot3Q(normal, linVel1);
    const float linNorVel = norVel0 - norVel1;

    float normalForce = 0.0f;

    for (uint j = 0u; j < contactCount && j < MAX_CTS; ++j)
    {
        const float4 point4 = s.cPts[j * 2u + 0u];
        const float3 point = point4.xyz;
        const float separation = point4.w;
        const float4 tv4 = s.cPts[j * 2u + 1u];
        const float3 targetVel3 = tv4.xyz;

        const float cTargetVel = dot3Q(normal, targetVel3);

        const float3 ra = point - b0.p;
        const float3 rb = point - b1.p;
        float3 raXn = crossQ(ra, normal);
        float3 rbXn = crossQ(rb, normal);

        const float angNorVel = dot3Q(raXn, b0.velTop) - dot3Q(rbXn, b1.velTop);
        const float slop =
            solverOffsetSlop * pxMaxQ(linNorVel == 0.0f ? 1.0f : angNorVel / linNorVel, 1.0f);

        raXn = slopZero(raXn, slop);
        rbXn = slopZero(rbXn, slop);

        float3 r0Top, r0Bot, r1Top, r1Bot;
        cirvQ(b0, raXn, normal, r0Top, r0Bot);
        cirvQ(b1, -rbXn, -normal, r1Top, r1Bot);

        // (normal, raXn).scale(lin, ang) = (normal*lin, raXn*ang)
        float3 dV0Top, dV0Bot, dV1Top, dV1Bot;
        spatialMulQ(b0.resp, normal * lin0X_ang0Y_lin1Z_ang1W.x,
                    raXn * lin0X_ang0Y_lin1Z_ang1W.y, dV0Top, dV0Bot);
        spatialMulQ(b1.resp, -normal * lin0X_ang0Y_lin1Z_ang1W.z,
                    -rbXn * lin0X_ang0Y_lin1Z_ang1W.w, dV1Top, dV1Bot);

        const float resp0 = dot3Q(dV0Top, r0Top) + dot3Q(dV0Bot, r0Bot);
        const float resp1 = dot3Q(dV1Top, r1Top) + dot3Q(dV1Bot, r1Bot);
        const float unitResponse = resp0 + resp1;

        const float vrel1 = norVel0 + projAngQ(b0, raXn);
        const float vrel2 = norVel1 + projAngQ(b1, rbXn);
        const float vrel = vrel1 - vrel2;

        const float penetration = separation - restDistance;
        const bool isSeparated = penetration >= 0.0f;
        const float penetrationInvDt = penetration * invDt;
        const float penetrationInvDtPt8 = max(maxPenBias, penetration * invDtp8);  // fmaxf

        const bool collidingWithVrel = (-vrel) > penetrationInvDt;
        const bool isGreater2 = (restitution > 0.0f) && (bounceThreshold > vrel) && collidingWithVrel;
        const bool ccdSeparationCondition = ccdMaxSeparation >= penetration;

        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.0f);
        const float tvel = targetVelocity;

        if (b0.isArticulation == 0u)
            targetVelocity -= vrel1;
        if (b1.isArticulation == 0u)
            targetVelocity += vrel2;

        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / (unitResponse + cfm)) : 0.0f;
        float velMultiplier = recipResponse;

        float coeff0, coeff1;
        if (restitution < 0.0f)  // compliant contact
        {
            coeff0 = compliantCoeffsQ(dt, flags, restitution, damping, unitResponse, recipResponse,
                                      penetration, targetVelocity, isSeparated, collidingWithVrel,
                                      velMultiplier);
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
                unbiasedErr -= pxMaxQ(0.0f, scaledBias);

            coeff0 = biasedErr;
            coeff1 = unbiasedErr;
        }

        o.cOut[j * 3u + 0u] = float4(r0Top, targetVelocity);
        o.cOut[j * 3u + 1u] = float4(-r1Top, pxMinQ(maxImpulse, tv4.w));
        o.cOut[j * 3u + 2u] = float4(resp0, resp1, coeff0, coeff1);

        const float deltaF =
            max((tvel - (vrel + max(penetrationInvDt, 0.0f))) * velMultiplier, 0.0f);  // fmaxf
        normalForce += deltaF;

        packArti(o.artiC, j, dV0Bot, dV0Top, dV1Bot, dV1Top);
    }

    o.hdrF.z = (normalForce / float(contactCount)) * 0.25f;  // minNormalForce

    const float frictionScale = (aCount == 2u) ? 0.5f : 1.0f;
    const float staticFriction = s.mat.z * frictionScale;
    const float dynamicFriction = s.mat.w * frictionScale;
    const bool haveFriction = anchorCount != 0u;

    o.hdrU = uint4(0u /* forceWritebackBufferOffset supplied by dispatcher */, flags, contactCount,
                   anchorCount);
    o.hdrF.x = restitution;
    o.hdrF.y = cfm;
    o.hdrF.w = dynamicFriction;
    o.normalSF = float4(normal, staticFriction);
    o.writeMask.x = 2u;

    if (haveFriction)
    {
        o.writeMask.x |= 4u;
        const float3 vrelV = linVel0 - linVel1;

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        const float3 t0Fallback1 = float3(0.0f, -normal.z, normal.y);
        const float3 t0Fallback2 = float3(-normal.y, normal.x, 0.0f);
        const float3 t0Fallback = orthoThreshold > abs(normal.x) ? t0Fallback1 : t0Fallback2;

        float3 t0 = vrelV - normal * dot3Q(normal, vrelV);
        t0 = dot3Q(t0, t0) > p1 ? t0 : t0Fallback;
        t0 = normalizeQ(t0);

        const float3 t1 = crossQ(normal, t0);

        o.fricN[0] = float4(t0, 0.0f);
        o.fricN[1] = float4(t1, 0.0f);

        for (uint j = 0u; j < aCount && j < MAX_ANCH; ++j)
        {
            const float3 body0Anchor = s.anchors0[j].xyz;
            const float3 body1Anchor = s.anchors1[j].xyz;

            const float3 ra = quatRotQ(b0.q, body0Anchor);
            const float3 rb = quatRotQ(b1.q, body1Anchor);
            const float3 error = (ra + b0.p) - (rb + b1.p);

            const uint index = perPointFriction ? (j == 0u ? s.contactIDs.x : s.contactIDs.y) : 0u;
            const float3 tvel3 = s.cPts[index * 2u + 1u].xyz;

            for (uint rrow = 0u; rrow < 2u; ++rrow)
            {
                const float3 t = rrow == 0u ? t0 : t1;

                float3 raXn = slopZero(crossQ(ra, t), solverOffsetSlop);
                float3 rbXn = slopZero(crossQ(rb, t), solverOffsetSlop);

                float3 r0Top, r0Bot, r1Top, r1Bot;
                cirvQ(b0, raXn, t, r0Top, r0Bot);
                cirvQ(b1, -rbXn, -t, r1Top, r1Bot);

                float3 dV0Top, dV0Bot, dV1Top, dV1Bot;
                spatialMulQ(b0.resp, t * lin0X_ang0Y_lin1Z_ang1W.x,
                            raXn * lin0X_ang0Y_lin1Z_ang1W.y, dV0Top, dV0Bot);
                spatialMulQ(b1.resp, -t * lin0X_ang0Y_lin1Z_ang1W.z,
                            -rbXn * lin0X_ang0Y_lin1Z_ang1W.w, dV1Top, dV1Bot);

                const float resp0 = dot3Q(dV0Top, r0Top) + dot3Q(dV0Bot, r0Bot);
                const float resp1 = dot3Q(dV1Top, r1Top) + dot3Q(dV1Bot, r1Bot);

                float targetVel = dot3Q(tvel3, t);
                if (b0.isArticulation == 0u)
                    targetVel -= projVelQ(b0, raXn, t);
                if (b1.isArticulation == 0u)
                    targetVel += projVelQ(b1, rbXn, t);

                const float bias = dot3Q(t, error) * invDt;

                const uint slot = j * 2u + rrow;
                o.fOut[slot * 3u + 0u] = float4(r0Top, bias);
                o.fOut[slot * 3u + 1u] = float4(-r1Top, targetVel);
                o.fOut[slot * 3u + 2u] = float4(resp0, resp1, 0.0f, 0.0f);

                packArtiF(o.artiF, slot, dV0Bot, dV0Top, dV1Bot, dV1Top);
            }
        }
    }

    gOut2[tid] = o;
}
