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

// pxgArtiPrepRef.h -- host mirrors for the articulation constraint prep
// (P8.102b).  Two hoists and one new mirror:
//
//  * computeCompliantDamping / computeCompliantContactCoefficientsTGS moved
//    VERBATIM from testConstraintPrep.cpp (second user: the arti contact
//    prep fixtures) -- parameterization is the whole diff;
//  * refCreateExtBody: the createPxgSolverExtBody expected-builder moved
//    from testArtiConstraintPrepBody.cpp -- the ENGINE's
//    PxSpatialMatrix::initialize does the response build;
//  * refSetupExtContactsTGS: the setupFinalizeExtSolverConstraintsBlock
//    (TGS overload, artiConstraintPrep2.cu:617-1112) transcription over the
//    engine's PxVec3/PxQuat/PxSpatialMatrix arithmetic on the host.
//
// Include after the engine headers (PxgArticulation.h, PxgSolverBody.h,
// PxgConstraintPrep.h, PxgFrictionPatch.h, PxgSolverConstraintBlock1D.h,
// PxgConstraintBlock.h, PxSpatialMatrix.h).

#ifndef PXG_ARTI_PREP_REF_H
#define PXG_ARTI_PREP_REF_H

namespace pxgtest
{
using namespace physx;

const uint32_t SC_COMPLIANT_ACCEL_SPRING = 1u << 3;

inline float computeCompliantDamping(bool isSeparated, bool collidingWithVrel, float damping)
{
    return (isSeparated && !collidingWithVrel) ? 0.0f : damping;
}

inline float computeCompliantContactCoefficientsTGS(float stepDt, uint32_t flags, float restitution,
                                                    float damping, float unitResponse,
                                                    float recipResponse, bool isSeparated,
                                                    bool collidingWithVrel, float& velMultiplier,
                                                    float& scaledBias)
{
    const bool accelSpring = (flags & SC_COMPLIANT_ACCEL_SPRING) != 0;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
    const float dampingIfEnabled = computeCompliantDamping(isSeparated, collidingWithVrel, damping);
    const float nrdt = stepDt * restitution;
    const float a = stepDt * (dampingIfEnabled - nrdt);
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
    velMultiplier = x * a * massIfAccelElseOne;
    scaledBias = nrdt * x * oneIfAccelElseR;
    return a;
}

// The engine-struct mirrors below need the full engine headers; a user that
// only wants the pure-float compliant functions (testConstraintPrep.cpp,
// whose own struct mirrors would clash) defines PXG_ARTI_PREP_REF_FLOAT_ONLY
// before including.
#ifndef PXG_ARTI_PREP_REF_FLOAT_ONLY

using physx::Cm::UnAlignedSpatialVector;

// ---------------------------------------------------------------------------
// createPxgSolverExtBody (artiConstraintPrep2.cu:55) on the host, over the
// engine structs; the articulation pools are flat arrays with per-record
// bases (the fixture's arrangement)
// ---------------------------------------------------------------------------
inline void refCreateExtBody(const PxNodeIndex& node, uint32_t bi,
                             const uint32_t* linkBases,
                             const PxTransform* b2w, const PxSpatialMatrix* resp,
                             const UnAlignedSpatialVector* mv, const float* cfms,
                             const PxgArticulationLink* links,
                             const PxgSolverBodyData* sbd, const PxgSolverTxIData* txi,
                             const PxAlignedTransform* frames,
                             PxgSolverExtBody2& b)
{
    if (node.isArticulation())
    {
        const uint32_t linkID = node.articulationLinkId();
        const uint32_t gi = linkBases[bi] + linkID;
        b.bodyIndex = bi;
        b.linkIndex = PxU16(linkID);
        b.body2World = b2w[gi];
        b.mSpatialResponse = resp[gi];
        b.isKinematic = false;
        b.islandNodeIndex = node;
        b.penBiasClamp = links[gi].penBiasClamp;
        b.maxImpulse = PX_MAX_F32;
        b.velocity = mv[gi];
        b.cfm = cfms[gi];
        b.offsetSlop = links[gi].offsetSlop;
    }
    else
    {
        const bool isStatic = node.isStaticBody();
        const PxgSolverBodyData& solverBody = sbd[bi];
        const bool isKinematic = (solverBody.flags & PxRigidBodyFlag::eKINEMATIC) != 0;
        UnAlignedSpatialVector velocity(PxVec3(0.f), PxVec3(0.f));
        PxReal penBiasClamp = -PX_MAX_F32;
        PxReal invMass = 0.f;
        const PxAlignedTransform body2World = frames[bi];
        PxReal offsetSlop = 0.f;
        if (!isStatic)
        {
            penBiasClamp = solverBody.penBiasClamp;
            invMass = solverBody.invMass;
            velocity.top = solverBody.initialAngVel;
            velocity.bottom = solverBody.initialLinVel;
            offsetSlop = solverBody.offsetSlop;
        }
        if (isStatic || isKinematic)
            std::memset(b.mSpatialResponse.column, 0, sizeof(b.mSpatialResponse));
        else
            b.mSpatialResponse.initialize(txi[bi].sqrtInvInertia, invMass);
        b.isKinematic = isKinematic;
        b.linkIndex = PxSolverConstraintDesc::RIGID_BODY;
        b.bodyIndex = bi;
        b.islandNodeIndex = node;
        b.penBiasClamp = penBiasClamp;
        b.body2World.p = PxVec3(body2World.p.x, body2World.p.y, body2World.p.z);
        b.body2World.q = PxQuat(body2World.q.q.x, body2World.q.q.y, body2World.q.q.z,
                                body2World.q.q.w);
        b.maxImpulse = solverBody.maxImpulse;
        b.velocity = velocity;
        b.cfm = 0.f;
        b.offsetSlop = offsetSlop;
    }
}

inline UnAlignedSpatialVector refCreateImpulseResponseVector(const PxgSolverExtBody2& b,
                                                             const UnAlignedSpatialVector& impulse)
{
    if (b.islandNodeIndex.isArticulation() || b.isKinematic)
        return impulse;
    return UnAlignedSpatialVector(b.mSpatialResponse.multiplyInertia(impulse.top), impulse.bottom);
}

inline UnAlignedSpatialVector refGetImpulseResponse(const PxgSolverExtBody2& b,
                                                    const UnAlignedSpatialVector& v)
{
    return b.mSpatialResponse * v;
}

inline float refProjectAngular(const PxgSolverExtBody2& b, const PxVec3& v)
{
    return b.velocity.top.dot(v);
}

inline float refProjectVelocity(const PxgSolverExtBody2& b, const UnAlignedSpatialVector& v)
{
    return b.velocity.dot(v);
}

// ---------------------------------------------------------------------------
// setupFinalizeExtSolverConstraintsBlock, TGS overload
// (artiConstraintPrep2.cu:617-1112) -- one lane.  Returns the number of
// articulation responses written.  The precedence quirk at :837 is
// transcribed as-is (see the module header's trap 13).
// ---------------------------------------------------------------------------
inline uint32_t refSetupExtContactsTGS(
    PxgBlockContactData& contactData,
    const PxgBlockContactPoint* contacts, uint32_t contactCount,
    const PxgBlockFrictionPatch& frictionPatch, const PxgBlockFrictionAnchorPatch& fAnchor,
    const PxgSolverExtBody2& b0, const PxgSolverExtBody2& b1,
    float invDt, float stepDt, float invTotalDt,
    float bounceThresholdF32, float biasCoefficient,
    uint32_t threadIndex, uint32_t forceWritebackBufferOffset, bool perPointFriction,
    PxgTGSBlockSolverContactHeader& contactHeader,
    PxgTGSBlockSolverFrictionHeader& frictionHeader,
    PxgTGSBlockSolverContactPoint* contactPoints,
    PxgTGSBlockSolverContactFriction* frictions,
    PxgArticulationBlockResponse* responses,
    const PxgMaterialContactData& data,
    float solverOffsetSlop, float2 torsionalFrictionData, float totalDt)
{
    const PxU8 flags = data.mSolverFlags;
    const float restDistance = data.restDistance;

    const PxVec3& bodyFrame0p = b0.body2World.p;
    const PxVec3& bodyFrame1p = b1.body2World.p;

    const PxReal cfm = PxMax(b0.cfm, b1.cfm);
    const float maxPenBias = PxMax(b0.penBiasClamp, b1.penBiasClamp);
    const float p8 = PxMin(0.8f, biasCoefficient);
    const float bounceThreshold = bounceThresholdF32;
    const float invDtp8 = invDt * p8;

    const PxgConstraintInvMassScale& ims = contactData.mInvMassScale[threadIndex];
    const float4 lin0X_ang0Y_lin1Z_ang1W = make_float4(ims.linear0, ims.angular0, ims.linear1, ims.angular1);

    const PxVec3& linVel0 = b0.velocity.bottom;
    const PxVec3& linVel1 = b1.velocity.bottom;
    const PxVec3& angVel0 = b0.velocity.top;
    const PxVec3& angVel1 = b1.velocity.top;

    uint32_t responsesWritten = 0;

    const PxU32 aCount = frictionPatch.anchorCount[threadIndex];
    PxU32 anchorCount = aCount * 2;
    const bool hasTorsional =
        (torsionalFrictionData.x != 0.f || torsionalFrictionData.y != 0.f) && aCount == 1;
    if (hasTorsional)
        anchorCount++;

    if (contactCount == 0)
    {
        frictionHeader.numFrictionConstr[threadIndex] = 0;
        contactHeader.numNormalConstr[threadIndex] = 0;
        contactHeader.forceWritebackOffset[threadIndex] = 0xffffffff;
        contactHeader.invMass0_1_angDom0_1[threadIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return 0;
    }

    const PxReal maxImpulse = PxMin(b0.maxImpulse, b1.maxImpulse);

    contactHeader.forceWritebackOffset[threadIndex] = forceWritebackBufferOffset;
    contactHeader.flags[threadIndex] = flags;
    contactHeader.maxPenBias[threadIndex] = maxPenBias;
    contactHeader.invMass0_1_angDom0_1[threadIndex] =
        make_float4(lin0X_ang0Y_lin1Z_ang1W.x, lin0X_ang0Y_lin1Z_ang1W.z,
                    lin0X_ang0Y_lin1Z_ang1W.y, lin0X_ang0Y_lin1Z_ang1W.w);

    const float4 normal4_rW = contactData.normal_restitutionW[threadIndex];
    const PxVec3 normal(normal4_rW.x, normal4_rW.y, normal4_rW.z);
    const PxReal restitution = normal4_rW.w;
    const PxReal norVel0 = normal.dot(linVel0);
    const PxReal norVel1 = normal.dot(linVel1);
    const PxReal linNorVel = norVel0 - norVel1;
    const PxReal damping = contactData.damping[threadIndex];

    frictionHeader.biasCoefficient[threadIndex] = invDt;
    contactHeader.restitutionXdt[threadIndex] = restitution * stepDt;
    contactHeader.cfm[threadIndex] = cfm;
    contactHeader.p8[threadIndex] = p8;

    PxReal normalForce = 0.f;
    PxReal minPen = 0.f;

    for (PxU32 j = 0; j < contactCount; j++)
    {
        const float4 point4_separationW = contacts[j].point_separationW[threadIndex];
        const PxVec3 point(point4_separationW.x, point4_separationW.y, point4_separationW.z);
        const float separation = point4_separationW.w;

        const float4 targetVel4_maxImpulseW = contacts[j].targetVel_maxImpulseW[threadIndex];
        const PxVec3 targetVel(targetVel4_maxImpulseW.x, targetVel4_maxImpulseW.y,
                               targetVel4_maxImpulseW.z);
        const float cTargetVel = normal.dot(targetVel);

        const PxVec3 ra = point - bodyFrame0p;
        const PxVec3 rb = point - bodyFrame1p;

        PxVec3 raXn = ra.cross(normal);
        PxVec3 rbXn = rb.cross(normal);

        const float angNorVel = raXn.dot(angVel0) - rbXn.dot(angVel1);

        const PxReal slop = solverOffsetSlop
                          * PxMax(linNorVel == 0.f ? 1.f : angNorVel / linNorVel, 1.f);
        raXn.x = PxAbs(raXn.x) < slop ? 0.f : raXn.x;
        raXn.y = PxAbs(raXn.y) < slop ? 0.f : raXn.y;
        raXn.z = PxAbs(raXn.z) < slop ? 0.f : raXn.z;
        rbXn.x = PxAbs(rbXn.x) < slop ? 0.f : rbXn.x;
        rbXn.y = PxAbs(rbXn.y) < slop ? 0.f : rbXn.y;
        rbXn.z = PxAbs(rbXn.z) < slop ? 0.f : rbXn.z;

        const UnAlignedSpatialVector response0 =
            refCreateImpulseResponseVector(b0, UnAlignedSpatialVector(raXn, normal));
        const UnAlignedSpatialVector response1 =
            refCreateImpulseResponseVector(b1, UnAlignedSpatialVector(-rbXn, -normal));

        const UnAlignedSpatialVector deltaV0 = refGetImpulseResponse(
            b0, UnAlignedSpatialVector(normal, raXn).scale(lin0X_ang0Y_lin1Z_ang1W.x,
                                                           lin0X_ang0Y_lin1Z_ang1W.y));
        const UnAlignedSpatialVector deltaV1 = refGetImpulseResponse(
            b1, UnAlignedSpatialVector(-normal, -rbXn).scale(lin0X_ang0Y_lin1Z_ang1W.z,
                                                             lin0X_ang0Y_lin1Z_ang1W.w));

        const float resp0 = deltaV0.dot(response0);
        const float resp1 = deltaV1.dot(response1);
        const float unitResponse = resp0 + resp1;

        const float vrel1 = norVel0 + refProjectAngular(b0, raXn);
        const float vrel2 = norVel1 + refProjectAngular(b1, rbXn);
        const float vrel = vrel1 - vrel2;

        const float penetration = separation - restDistance;
        const bool isSeparated = (penetration > 0.0f);
        const float penetrationInvDt = penetration * invTotalDt;

        float scaledBias, velMultiplier;
        const float recipResponse = (unitResponse > 0.f) ? (1.f / (unitResponse + cfm)) : 0.f;

        PxReal compliantContactCoeff = 0.f;
        if (restitution < 0.f)
        {
            const bool collidingWithVrel = ((-vrel * totalDt) > penetration);
            compliantContactCoeff = computeCompliantContactCoefficientsTGS(
                stepDt, flags, restitution, damping, unitResponse, recipResponse, isSeparated,
                collidingWithVrel, velMultiplier, scaledBias);
        }
        else
        {
            velMultiplier = recipResponse;
            scaledBias = isSeparated ? -invDt : -invDtp8;
        }

        contactPoints[j].resp0[threadIndex] = resp0;
        contactPoints[j].resp1[threadIndex] = resp1;

        minPen = PxMin(penetration, minPen);

        const bool isGreater2 = ((restitution > 1e-5f) && (bounceThreshold > vrel))
                             && ((-vrel) > penetrationInvDt);

        float totalError = penetration;

        // the :837 precedence quirk, kept literally
        const float tvCond = cTargetVel + (isGreater2 ? 1.0f : 0.0f);
        float targetVelocity = (tvCond != 0.0f) ? (-vrel) * restitution : 0.f;

        const PxReal deltaF =
            fmaxf((targetVelocity + (-penetrationInvDt - vrel)) * velMultiplier, 0.f);
        normalForce += deltaF;

        if (b0.isKinematic)
            targetVelocity -= vrel1;
        if (b1.isKinematic)
            targetVelocity += vrel2;

        if (isGreater2)
        {
            const PxReal ratio = totalDt + penetration / vrel;
            totalError += ratio * targetVelocity;
        }

        contactPoints[j].raXn_extraCoeff[threadIndex] =
            make_float4(response0.top.x, response0.top.y, response0.top.z, compliantContactCoeff);
        contactPoints[j].rbXn_targetVelW[threadIndex] =
            make_float4(-response1.top.x, -response1.top.y, -response1.top.z, targetVelocity);
        contactPoints[j].appliedForce[threadIndex] = 0.f;
        contactPoints[j].maxImpulse[threadIndex] = PxMin(maxImpulse, targetVel4_maxImpulseW.w);
        contactPoints[j].separation[threadIndex] = totalError;
        contactPoints[j].biasCoefficient[threadIndex] = scaledBias;

        PxgArticulationBlockResponse& r = responses[responsesWritten];
        r.deltaRALin_x[threadIndex] = deltaV0.bottom.x;
        r.deltaRALin_y[threadIndex] = deltaV0.bottom.y;
        r.deltaRALin_z[threadIndex] = deltaV0.bottom.z;
        r.deltaRAAng_x[threadIndex] = deltaV0.top.x;
        r.deltaRAAng_y[threadIndex] = deltaV0.top.y;
        r.deltaRAAng_z[threadIndex] = deltaV0.top.z;
        r.deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
        r.deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
        r.deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
        r.deltaRBAng_x[threadIndex] = deltaV1.top.x;
        r.deltaRBAng_y[threadIndex] = deltaV1.top.y;
        r.deltaRBAng_z[threadIndex] = deltaV1.top.z;
        responsesWritten++;
    }

    contactHeader.minNormalForce[threadIndex] = (normalForce / contactCount) * 0.25f;

    const PxReal frictionScale = (aCount == 2) ? 0.5f : 1.f;
    const PxReal staticFriction = data.staticFriction * frictionScale;
    const PxReal dynamicFriction = data.dynamicFriction * frictionScale;

    const bool haveFriction = (anchorCount != 0);
    contactHeader.numNormalConstr[threadIndex] = contactCount;
    frictionHeader.numFrictionConstr[threadIndex] = anchorCount;

    contactHeader.normal_staticFriction[threadIndex] =
        make_float4(normal.x, normal.y, normal.z, staticFriction);
    frictionHeader.dynamicFriction[threadIndex] = dynamicFriction;

    if (haveFriction)
    {
        const PxVec3 vrel = linVel0 - linVel1;

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        const float normalX = normal.x;
        const float normalY = normal.y;
        const float normalZ = normal.z;

        PxVec3 t0Fallback1(0.f, -normalZ, normalY);
        PxVec3 t0Fallback2(-normalY, normalX, 0.f);
        PxVec3 t0Fallback = orthoThreshold > PxAbs(normalX) ? t0Fallback1 : t0Fallback2;

        PxVec3 t0 = vrel - normal * (normal.dot(vrel));
        t0 = t0.magnitudeSquared() > p1 ? t0 : t0Fallback;
        // the divide-by-sqrt spelling, matching PxVec3::normalize and the
        // shader's explicit form (the normalize() intrinsic is 1 ULP off)
        t0 = t0 / PxSqrt(t0.magnitudeSquared());

        const PxVec3 t1 = normal.cross(t0);

        frictionHeader.broken[threadIndex] = 0;
        frictionHeader.frictionNormals[0][threadIndex] = make_float4(t0.x, t0.y, t0.z, 0.f);
        frictionHeader.frictionNormals[1][threadIndex] = make_float4(t1.x, t1.y, t1.z, 0.f);

        const PxVec3 relTr = bodyFrame0p - bodyFrame1p;

        for (PxU32 j = 0; j < aCount; j++)
        {
            PxgTGSBlockSolverContactFriction* f0 = &frictions[2 * j];
            PxgTGSBlockSolverContactFriction* f1 = &frictions[2 * j + 1];

            const float4 body0Anchor4 = fAnchor.body0Anchors[j][threadIndex];
            const float4 body1Anchor4 = fAnchor.body1Anchors[j][threadIndex];

            const PxVec3 body0Anchor(body0Anchor4.x, body0Anchor4.y, body0Anchor4.z);
            const PxVec3 body1Anchor(body1Anchor4.x, body1Anchor4.y, body1Anchor4.z);

            const PxVec3 ra = b0.body2World.q.rotate(body0Anchor);
            const PxVec3 rb = b1.body2World.q.rotate(body1Anchor);

            const PxU32 index = perPointFriction ? frictionPatch.contactID[j][threadIndex] : 0;

            const float4 targetVel4 = contacts[index].targetVel_maxImpulseW[threadIndex];
            const PxVec3 tvel(targetVel4.x, targetVel4.y, targetVel4.z);

            const PxVec3 error = (ra - rb) + relTr;

            const PxVec3* tangents[2] = { &t0, &t1 };
            PxgTGSBlockSolverContactFriction* fs[2] = { f0, f1 };
            for (int tn = 0; tn < 2; ++tn)
            {
                const PxVec3& t = *tangents[tn];
                PxVec3 raXn = ra.cross(t);
                PxVec3 rbXn = rb.cross(t);

                raXn.x = PxAbs(raXn.x) < solverOffsetSlop ? 0.f : raXn.x;
                raXn.y = PxAbs(raXn.y) < solverOffsetSlop ? 0.f : raXn.y;
                raXn.z = PxAbs(raXn.z) < solverOffsetSlop ? 0.f : raXn.z;
                rbXn.x = PxAbs(rbXn.x) < solverOffsetSlop ? 0.f : rbXn.x;
                rbXn.y = PxAbs(rbXn.y) < solverOffsetSlop ? 0.f : rbXn.y;
                rbXn.z = PxAbs(rbXn.z) < solverOffsetSlop ? 0.f : rbXn.z;

                const UnAlignedSpatialVector response0 =
                    refCreateImpulseResponseVector(b0, UnAlignedSpatialVector(raXn, t));
                const UnAlignedSpatialVector response1 =
                    refCreateImpulseResponseVector(b1, UnAlignedSpatialVector(-rbXn, -t));
                const UnAlignedSpatialVector deltaV0 = refGetImpulseResponse(
                    b0, UnAlignedSpatialVector(t, raXn).scale(lin0X_ang0Y_lin1Z_ang1W.x,
                                                              lin0X_ang0Y_lin1Z_ang1W.y));
                const UnAlignedSpatialVector deltaV1 = refGetImpulseResponse(
                    b1, UnAlignedSpatialVector(-t, -rbXn).scale(lin0X_ang0Y_lin1Z_ang1W.z,
                                                                lin0X_ang0Y_lin1Z_ang1W.w));

                const float resp0 = deltaV0.dot(response0);
                const float resp1 = deltaV1.dot(response1);

                float targetVelF = tvel.dot(t);
                if (b0.isKinematic)
                    targetVelF -= refProjectVelocity(b0, UnAlignedSpatialVector(raXn, t));
                if (b1.isKinematic)
                    targetVelF += refProjectVelocity(b1, UnAlignedSpatialVector(rbXn, t));

                const float frictionError = error.dot(t);

                PxgTGSBlockSolverContactFriction* f = fs[tn];
                f->resp0[threadIndex] = resp0;
                f->resp1[threadIndex] = resp1;
                f->raXn_error[threadIndex] =
                    make_float4(response0.top.x, response0.top.y, response0.top.z, frictionError);
                f->rbXn_targetVelW[threadIndex] =
                    make_float4(-response1.top.x, -response1.top.y, -response1.top.z, targetVelF);
                f->appliedForce[threadIndex] = 0.f;

                PxgArticulationBlockResponse& r = responses[responsesWritten];
                r.deltaRALin_x[threadIndex] = deltaV0.bottom.x;
                r.deltaRALin_y[threadIndex] = deltaV0.bottom.y;
                r.deltaRALin_z[threadIndex] = deltaV0.bottom.z;
                r.deltaRAAng_x[threadIndex] = deltaV0.top.x;
                r.deltaRAAng_y[threadIndex] = deltaV0.top.y;
                r.deltaRAAng_z[threadIndex] = deltaV0.top.z;
                r.deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
                r.deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
                r.deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
                r.deltaRBAng_x[threadIndex] = deltaV1.top.x;
                r.deltaRBAng_y[threadIndex] = deltaV1.top.y;
                r.deltaRBAng_z[threadIndex] = deltaV1.top.z;
                responsesWritten++;
            }
        }

        if (hasTorsional)
        {
            PxgTGSBlockSolverContactFriction* f0 = &frictions[2];

            const PxReal torsionalScale = PxMax(torsionalFrictionData.y,
                                                PxSqrt(PxMax(0.f, -minPen) * torsionalFrictionData.x));
            frictionHeader.torsionalFrictionScale[threadIndex] = torsionalScale;

            const UnAlignedSpatialVector response0 =
                refCreateImpulseResponseVector(b0, UnAlignedSpatialVector(normal, PxVec3(0.f)));
            const UnAlignedSpatialVector response1 =
                refCreateImpulseResponseVector(b1, UnAlignedSpatialVector(-normal, PxVec3(0.f)));
            const UnAlignedSpatialVector deltaV0 = refGetImpulseResponse(
                b0, UnAlignedSpatialVector(PxVec3(0.f), normal).scale(lin0X_ang0Y_lin1Z_ang1W.x,
                                                                      lin0X_ang0Y_lin1Z_ang1W.y));
            const UnAlignedSpatialVector deltaV1 = refGetImpulseResponse(
                b1, UnAlignedSpatialVector(PxVec3(0.f), -normal).scale(lin0X_ang0Y_lin1Z_ang1W.z,
                                                                       lin0X_ang0Y_lin1Z_ang1W.w));

            const float resp0 = deltaV0.dot(response0);
            const float resp1 = deltaV1.dot(response1);

            f0->resp0[threadIndex] = resp0;
            f0->resp1[threadIndex] = resp1;
            f0->raXn_error[threadIndex] =
                make_float4(response0.top.x, response0.top.y, response0.top.z, 0.f);
            f0->rbXn_targetVelW[threadIndex] =
                make_float4(-response1.top.x, -response1.top.y, -response1.top.z, 0.f);
            f0->appliedForce[threadIndex] = 0.f;

            PxgArticulationBlockResponse& r = responses[responsesWritten];
            r.deltaRALin_x[threadIndex] = deltaV0.bottom.x;
            r.deltaRALin_y[threadIndex] = deltaV0.bottom.y;
            r.deltaRALin_z[threadIndex] = deltaV0.bottom.z;
            r.deltaRAAng_x[threadIndex] = deltaV0.top.x;
            r.deltaRAAng_y[threadIndex] = deltaV0.top.y;
            r.deltaRAAng_z[threadIndex] = deltaV0.top.z;
            r.deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
            r.deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
            r.deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
            r.deltaRBAng_x[threadIndex] = deltaV1.top.x;
            r.deltaRBAng_y[threadIndex] = deltaV1.top.y;
            r.deltaRBAng_z[threadIndex] = deltaV1.top.z;
            responsesWritten++;
        }
    }

    return responsesWritten;
}

// ---------------------------------------------------------------------------
// P8.121: setupFinalizeExtSolverConstraintsBlock, PGS overload
// (artiConstraintPrep2.cu:169-613) -- one lane.  The differences from the
// TGS mirror above are the module's traps 28-45: fixed p8 = 0.8, >= in
// isSeparated, restitution > 0, the parenthesized targetVelocity (no
// precedence quirk), !isArticulation-keyed velocity adjusts, coeff0/coeff1
// rows, the (ra + p0) - (rb + p1) error spelling, per-row bias * invDt,
// and no torsional friction.  The optional out-counters are COVERAGE
// instrumentation only.
// ---------------------------------------------------------------------------

inline float computeCompliantContactCoefficientsPGSRef(
    float dt, uint32_t flags, float restitution, float damping, float unitResponse,
    float recipResponse, float penetration, float targetVelocity, bool isSeparated,
    bool collidingWithVrel, float& velMultiplier, float& impulseMultiplier,
    float& unbiasedErr, float& biasedErr)
{
    const bool accelSpring = (flags & SC_COMPLIANT_ACCEL_SPRING) != 0;
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
    unbiasedErr = biasedErr = targetVelocity * velMultiplier - scaledBias;
    return a;
}

inline uint32_t refSetupExtContactsPGS(
    PxgBlockContactData& contactData,
    const PxgBlockContactPoint* contacts, uint32_t contactCount,
    const PxgBlockFrictionPatch& frictionPatch, const PxgBlockFrictionAnchorPatch& fAnchor,
    const PxgSolverExtBody2& b0, const PxgSolverExtBody2& b1,
    float invDt, float dt, float bounceThresholdF32,
    uint32_t threadIndex, uint32_t forceWritebackBufferOffset, bool perPointFriction,
    PxgBlockSolverContactHeader& contactHeader,
    PxgBlockSolverFrictionHeader& frictionHeader,
    PxgBlockSolverContactPoint* contactPoints,
    PxgBlockSolverContactFriction* frictions,
    PxgArticulationBlockResponse* responses,
    const PxgMaterialContactData& data,
    float solverOffsetSlop,
    uint32_t* covGreater2 = nullptr, uint32_t* covCompliant = nullptr,
    uint32_t* covPenZero = nullptr, uint32_t* covPosBias = nullptr,
    uint32_t* covSlopScaled = nullptr, uint32_t* covPenClamp = nullptr)
{
    const PxU8 flags = data.mSolverFlags;
    const float restDistance = data.restDistance;

    const PxVec3& bodyFrame0p = b0.body2World.p;
    const PxVec3& bodyFrame1p = b1.body2World.p;

    const float maxPenBias = PxMax(b0.penBiasClamp, b1.penBiasClamp);
    const float p8 = 0.8f;   // FIXED (trap 29); biasCoefficient is dead in PGS
    const float bounceThreshold = bounceThresholdF32;
    const float invDtp8 = invDt * p8;

    const PxgConstraintInvMassScale& ims = contactData.mInvMassScale[threadIndex];
    const float4 lin0X_ang0Y_lin1Z_ang1W =
        make_float4(ims.linear0, ims.angular0, ims.linear1, ims.angular1);

    const PxVec3& linVel0 = b0.velocity.bottom;
    const PxVec3& linVel1 = b1.velocity.bottom;
    const PxVec3& angVel0 = b0.velocity.top;
    const PxVec3& angVel1 = b1.velocity.top;

    const PxReal cfm = PxMax(b0.cfm, b1.cfm);

    uint32_t responsesWritten = 0;

    const PxU32 aCount = frictionPatch.anchorCount[threadIndex];
    const PxU32 anchorCount = aCount * 2;   // no torsional row (trap 44)

    if (contactCount == 0)
    {
        // trap 28: exactly three fields (no invMass zeroing in PGS)
        frictionHeader.numFrictionConstr[threadIndex] = 0;
        contactHeader.numNormalConstr[threadIndex] = 0;
        contactHeader.forceWritebackOffset[threadIndex] = 0xffffffff;
        return 0;
    }

    const PxReal maxImpulse = PxMin(b0.maxImpulse, b1.maxImpulse);

    contactHeader.forceWritebackOffset[threadIndex] = forceWritebackBufferOffset;
    contactHeader.flags[threadIndex] = flags;
    contactHeader.invMass0_1_angDom0_1[threadIndex] =
        make_float4(lin0X_ang0Y_lin1Z_ang1W.x, lin0X_ang0Y_lin1Z_ang1W.z,
                    lin0X_ang0Y_lin1Z_ang1W.y, lin0X_ang0Y_lin1Z_ang1W.w);

    const float4 normal4_rW = contactData.normal_restitutionW[threadIndex];
    const PxVec3 normal(normal4_rW.x, normal4_rW.y, normal4_rW.z);
    const PxReal restitution = normal4_rW.w;
    const PxReal damping = contactData.damping[threadIndex];
    const PxReal norVel0 = normal.dot(linVel0);
    const PxReal norVel1 = normal.dot(linVel1);
    const PxReal linNorVel = norVel0 - norVel1;

    // trap 30: restitution + cfm (no restitutionXdt/p8/maxPenBias slots)
    contactHeader.restitution[threadIndex] = restitution;
    contactHeader.cfm[threadIndex] = cfm;

    PxReal normalForce = 0.f;

    for (PxU32 j = 0; j < contactCount; j++)
    {
        const float4 point4_separationW = contacts[j].point_separationW[threadIndex];
        const PxVec3 point(point4_separationW.x, point4_separationW.y, point4_separationW.z);
        const float separation = point4_separationW.w;

        const float4 targetVel4_maxImpulseW = contacts[j].targetVel_maxImpulseW[threadIndex];
        const PxVec3 targetVel(targetVel4_maxImpulseW.x, targetVel4_maxImpulseW.y,
                               targetVel4_maxImpulseW.z);
        const float cTargetVel = normal.dot(targetVel);

        const PxVec3 ra = point - bodyFrame0p;
        const PxVec3 rb = point - bodyFrame1p;

        PxVec3 raXn = ra.cross(normal);
        PxVec3 rbXn = rb.cross(normal);

        const float angNorVel = raXn.dot(angVel0) - rbXn.dot(angVel1);

        const PxReal slop = solverOffsetSlop
                          * PxMax(linNorVel == 0.f ? 1.f : angNorVel / linNorVel, 1.f);
        if (covSlopScaled)
        {
            const float comps[6] = { raXn.x, raXn.y, raXn.z, rbXn.x, rbXn.y, rbXn.z };
            for (float c : comps)
                if (PxAbs(c) < slop && PxAbs(c) >= solverOffsetSlop)
                    (*covSlopScaled)++;
        }
        raXn.x = PxAbs(raXn.x) < slop ? 0.f : raXn.x;
        raXn.y = PxAbs(raXn.y) < slop ? 0.f : raXn.y;
        raXn.z = PxAbs(raXn.z) < slop ? 0.f : raXn.z;
        rbXn.x = PxAbs(rbXn.x) < slop ? 0.f : rbXn.x;
        rbXn.y = PxAbs(rbXn.y) < slop ? 0.f : rbXn.y;
        rbXn.z = PxAbs(rbXn.z) < slop ? 0.f : rbXn.z;

        const UnAlignedSpatialVector response0 =
            refCreateImpulseResponseVector(b0, UnAlignedSpatialVector(raXn, normal));
        const UnAlignedSpatialVector response1 =
            refCreateImpulseResponseVector(b1, UnAlignedSpatialVector(-rbXn, -normal));

        const UnAlignedSpatialVector deltaV0 = refGetImpulseResponse(
            b0, UnAlignedSpatialVector(normal, raXn).scale(lin0X_ang0Y_lin1Z_ang1W.x,
                                                           lin0X_ang0Y_lin1Z_ang1W.y));
        const UnAlignedSpatialVector deltaV1 = refGetImpulseResponse(
            b1, UnAlignedSpatialVector(-normal, -rbXn).scale(lin0X_ang0Y_lin1Z_ang1W.z,
                                                             lin0X_ang0Y_lin1Z_ang1W.w));

        const float resp0 = deltaV0.dot(response0);
        const float resp1 = deltaV1.dot(response1);
        const float unitResponse = resp0 + resp1;

        const float vrel1 = norVel0 + refProjectAngular(b0, raXn);
        const float vrel2 = norVel1 + refProjectAngular(b1, rbXn);
        const float vrel = vrel1 - vrel2;

        const float penetration = separation - restDistance;
        const bool isSeparated = (penetration >= 0.0f);   // trap 31: >=
        const float penetrationInvDt = penetration * invDt;   // trap 32
        const float penetrationInvDtPt8 = fmaxf(maxPenBias, penetration * invDtp8);
        if (covPenClamp && !isSeparated && maxPenBias > penetration * invDtp8)
            (*covPenClamp)++;

        const bool collidingWithVrel = ((-vrel) > penetrationInvDt);
        // trap 33: > 0
        const bool isGreater2 = (restitution > 0.f) && (bounceThreshold > vrel)
                             && collidingWithVrel;

        // isCCD is the engine's own hard-coded false: always true
        const bool ccdSeparationCondition = PX_MAX_F32 >= penetration;

        // trap 34: parenthesized -- no precedence quirk in PGS
        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.f);

        const PxReal tvel = targetVelocity;   // trap 38: pre-adjust

        // trap 35: ANY rigid
        if (!b0.islandNodeIndex.isArticulation())
            targetVelocity -= vrel1;
        if (!b1.islandNodeIndex.isArticulation())
            targetVelocity += vrel2;

        const float recipResponse = (unitResponse > 0.0f) ? (1.f / (unitResponse + cfm)) : 0.f;
        PxReal velMultiplier = recipResponse;

        PxReal coeff0, coeff1;
        if (restitution < 0.f)   // compliant contact
        {
            PxReal impulseMultiplier, unbiasedErr, biasedErr;
            coeff0 = computeCompliantContactCoefficientsPGSRef(
                dt, flags, restitution, damping, unitResponse, recipResponse, penetration,
                targetVelocity, isSeparated, collidingWithVrel, velMultiplier,
                impulseMultiplier, unbiasedErr, biasedErr);
            const PxReal nrdt = dt * restitution;
            coeff1 = -nrdt * penetration;
            if (covCompliant) (*covCompliant)++;
        }
        else
        {
            float scaledBias = isSeparated ? penetrationInvDt : penetrationInvDtPt8;
            if (ccdSeparationCondition && isGreater2)
                scaledBias = 0.f;

            PxReal unbiasedErr = targetVelocity;
            const PxReal biasedErr = unbiasedErr - scaledBias;

            if (!isGreater2)
                unbiasedErr -= PxMax(0.f, scaledBias);

            coeff0 = biasedErr;
            coeff1 = unbiasedErr;
            if (covPosBias && scaledBias > 0.f) (*covPosBias)++;
        }
        if (covGreater2 && isGreater2) (*covGreater2)++;
        if (covPenZero && penetration == 0.f) (*covPenZero)++;

        contactPoints[j].raXn_targetVelocity[threadIndex] =
            make_float4(response0.top.x, response0.top.y, response0.top.z, targetVelocity);
        contactPoints[j].rbXn_maxImpulse[threadIndex] =
            make_float4(-response1.top.x, -response1.top.y, -response1.top.z,
                        PxMin(maxImpulse, targetVel4_maxImpulseW.w));
        contactPoints[j].resp0[threadIndex] = resp0;
        contactPoints[j].resp1[threadIndex] = resp1;
        contactPoints[j].coeff0[threadIndex] = coeff0;
        contactPoints[j].coeff1[threadIndex] = coeff1;
        contactPoints[j].appliedForce[threadIndex] = 0.f;

        // trap 38: the pre-adjust tvel and the max(penInvDt, 0) spelling
        const PxReal deltaF =
            fmaxf((tvel - (vrel + fmaxf(penetrationInvDt, 0.f))) * velMultiplier, 0.f);
        normalForce += deltaF;

        PxgArticulationBlockResponse& r = responses[responsesWritten];
        r.deltaRALin_x[threadIndex] = deltaV0.bottom.x;
        r.deltaRALin_y[threadIndex] = deltaV0.bottom.y;
        r.deltaRALin_z[threadIndex] = deltaV0.bottom.z;
        r.deltaRAAng_x[threadIndex] = deltaV0.top.x;
        r.deltaRAAng_y[threadIndex] = deltaV0.top.y;
        r.deltaRAAng_z[threadIndex] = deltaV0.top.z;
        r.deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
        r.deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
        r.deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
        r.deltaRBAng_x[threadIndex] = deltaV1.top.x;
        r.deltaRBAng_y[threadIndex] = deltaV1.top.y;
        r.deltaRBAng_z[threadIndex] = deltaV1.top.z;
        responsesWritten++;
    }

    contactHeader.minNormalForce[threadIndex] = (normalForce / contactCount) * 0.25f;

    const PxReal frictionScale = (aCount == 2) ? 0.5f : 1.f;
    const PxReal staticFriction = data.staticFriction * frictionScale;
    const PxReal dynamicFriction = data.dynamicFriction * frictionScale;

    const bool haveFriction = (anchorCount != 0);
    contactHeader.numNormalConstr[threadIndex] = contactCount;
    frictionHeader.numFrictionConstr[threadIndex] = anchorCount;

    contactHeader.normal_staticFriction[threadIndex] =
        make_float4(normal.x, normal.y, normal.z, staticFriction);
    frictionHeader.dynamicFriction[threadIndex] = dynamicFriction;

    if (haveFriction)
    {
        const PxVec3 vrel = linVel0 - linVel1;

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        const float normalX = normal.x;
        const float normalY = normal.y;
        const float normalZ = normal.z;

        PxVec3 t0Fallback1(0.f, -normalZ, normalY);
        PxVec3 t0Fallback2(-normalY, normalX, 0.f);
        PxVec3 t0Fallback = orthoThreshold > PxAbs(normalX) ? t0Fallback1 : t0Fallback2;

        PxVec3 t0 = vrel - normal * (normal.dot(vrel));
        t0 = t0.magnitudeSquared() > p1 ? t0 : t0Fallback;
        t0 = t0 / PxSqrt(t0.magnitudeSquared());

        const PxVec3 t1 = normal.cross(t0);

        frictionHeader.broken[threadIndex] = 0;
        frictionHeader.frictionNormals[0][threadIndex] = make_float4(t0.x, t0.y, t0.z, 0.f);
        frictionHeader.frictionNormals[1][threadIndex] = make_float4(t1.x, t1.y, t1.z, 0.f);

        for (PxU32 j = 0; j < aCount; j++)
        {
            PxgBlockSolverContactFriction* f0 = &frictions[2 * j];
            PxgBlockSolverContactFriction* f1 = &frictions[2 * j + 1];

            const float4 body0Anchor4 = fAnchor.body0Anchors[j][threadIndex];
            const float4 body1Anchor4 = fAnchor.body1Anchors[j][threadIndex];

            const PxVec3 body0Anchor(body0Anchor4.x, body0Anchor4.y, body0Anchor4.z);
            const PxVec3 body1Anchor(body1Anchor4.x, body1Anchor4.y, body1Anchor4.z);

            const PxVec3 ra = b0.body2World.q.rotate(body0Anchor);
            const PxVec3 rb = b1.body2World.q.rotate(body1Anchor);

            // trap 40: (ra + p0) - (rb + p1)
            const PxVec3 error = (ra + bodyFrame0p) - (rb + bodyFrame1p);

            const PxU32 index = perPointFriction ? frictionPatch.contactID[j][threadIndex] : 0;

            const float4 targetVel4 = contacts[index].targetVel_maxImpulseW[threadIndex];
            const PxVec3 tvel3(targetVel4.x, targetVel4.y, targetVel4.z);

            const PxVec3* tangents[2] = { &t0, &t1 };
            PxgBlockSolverContactFriction* fs[2] = { f0, f1 };
            for (int tn = 0; tn < 2; ++tn)
            {
                const PxVec3& t = *tangents[tn];
                PxVec3 raXn = ra.cross(t);
                PxVec3 rbXn = rb.cross(t);

                raXn.x = PxAbs(raXn.x) < solverOffsetSlop ? 0.f : raXn.x;
                raXn.y = PxAbs(raXn.y) < solverOffsetSlop ? 0.f : raXn.y;
                raXn.z = PxAbs(raXn.z) < solverOffsetSlop ? 0.f : raXn.z;
                rbXn.x = PxAbs(rbXn.x) < solverOffsetSlop ? 0.f : rbXn.x;
                rbXn.y = PxAbs(rbXn.y) < solverOffsetSlop ? 0.f : rbXn.y;
                rbXn.z = PxAbs(rbXn.z) < solverOffsetSlop ? 0.f : rbXn.z;

                const UnAlignedSpatialVector response0 =
                    refCreateImpulseResponseVector(b0, UnAlignedSpatialVector(raXn, t));
                const UnAlignedSpatialVector response1 =
                    refCreateImpulseResponseVector(b1, UnAlignedSpatialVector(-rbXn, -t));
                const UnAlignedSpatialVector deltaV0 = refGetImpulseResponse(
                    b0, UnAlignedSpatialVector(t, raXn).scale(lin0X_ang0Y_lin1Z_ang1W.x,
                                                              lin0X_ang0Y_lin1Z_ang1W.y));
                const UnAlignedSpatialVector deltaV1 = refGetImpulseResponse(
                    b1, UnAlignedSpatialVector(-t, -rbXn).scale(lin0X_ang0Y_lin1Z_ang1W.z,
                                                                lin0X_ang0Y_lin1Z_ang1W.w));

                const float resp0 = deltaV0.dot(response0);
                const float resp1 = deltaV1.dot(response1);

                float targetVelF = tvel3.dot(t);
                // trap 35/42: !isArticulation-keyed
                if (!b0.islandNodeIndex.isArticulation())
                    targetVelF -= refProjectVelocity(b0, UnAlignedSpatialVector(raXn, t));
                if (!b1.islandNodeIndex.isArticulation())
                    targetVelF += refProjectVelocity(b1, UnAlignedSpatialVector(rbXn, t));

                // trap 41: the per-row bias
                const float bias = t.dot(error) * invDt;

                PxgBlockSolverContactFriction* f = fs[tn];
                f->raXn_bias[threadIndex] =
                    make_float4(response0.top.x, response0.top.y, response0.top.z, bias);
                f->rbXn_targetVelW[threadIndex] =
                    make_float4(-response1.top.x, -response1.top.y, -response1.top.z, targetVelF);
                f->appliedForce[threadIndex] = 0.f;
                f->resp0[threadIndex] = resp0;
                f->resp1[threadIndex] = resp1;

                PxgArticulationBlockResponse& r = responses[responsesWritten];
                r.deltaRALin_x[threadIndex] = deltaV0.bottom.x;
                r.deltaRALin_y[threadIndex] = deltaV0.bottom.y;
                r.deltaRALin_z[threadIndex] = deltaV0.bottom.z;
                r.deltaRAAng_x[threadIndex] = deltaV0.top.x;
                r.deltaRAAng_y[threadIndex] = deltaV0.top.y;
                r.deltaRAAng_z[threadIndex] = deltaV0.top.z;
                r.deltaRBLin_x[threadIndex] = deltaV1.bottom.x;
                r.deltaRBLin_y[threadIndex] = deltaV1.bottom.y;
                r.deltaRBLin_z[threadIndex] = deltaV1.bottom.z;
                r.deltaRBAng_x[threadIndex] = deltaV1.top.x;
                r.deltaRBAng_y[threadIndex] = deltaV1.top.y;
                r.deltaRBAng_z[threadIndex] = deltaV1.top.z;
                responsesWritten++;
            }
        }
    }

    return responsesWritten;
}

// ---------------------------------------------------------------------------
// P8.102c: the friction-patch helpers of artiCreateFinalizeSolverContactsBlockGPU
// on the ENGINE structs, over the engine's own PX_CUDA_CALLABLE
// PxAlignedTransform arithmetic (correlatePatches / getFrictionPatches /
// growPatches, contactConstraintBlockPrep.cuh).  The optional out-flags are
// COVERAGE instrumentation only -- they change nothing observable.
// ---------------------------------------------------------------------------

inline void refCorrelatePatches(PxgBlockFrictionPatch& fp, uint32_t lane, uint32_t nbContacts,
                                const PxVec3& normal, const PxAlignedTransform& bf0,
                                const PxAlignedTransform& bf1)
{
    if (nbContacts > 0)
    {
        const PxVec3 b0n = bf0.rotateInv(normal);
        const PxVec3 b1n = bf1.rotateInv(normal);
        fp.contactID[0][lane] = 0xff;
        fp.contactID[1][lane] = 0xff;
        fp.body0Normal[lane] = make_float4(b0n.x, b0n.y, b0n.z, 0.f);
        fp.body1Normal[lane] = make_float4(b1n.x, b1n.y, b1n.z, 0.f);
        fp.anchorCount[lane] = 0;
        fp.broken[lane] = 0;
    }
}

inline bool refPointsAreClose(const PxAlignedTransform& body1ToBody0, const float4& la0,
                              const float4& la1, const float4& axis, float correlDist)
{
    const float4 p1 = body1ToBody0.transform(la1);
    const PxVec3 d(la0.x - p1.x, la0.y - p1.y, la0.z - p1.z);
    return PxAbs(d.dot(PxVec3(axis.x, axis.y, axis.z))) < correlDist;
}

inline void refGetFrictionPatches(PxgBlockFrictionPatch& curr, PxgBlockFrictionAnchorPatch& currA,
                                  uint32_t lane, const PxU64* prevIndices, bool hasPrevIndices,
                                  uint32_t prevFrictionStartIndex,
                                  const PxgBlockFrictionPatch* prevPatches,
                                  const PxgBlockFrictionAnchorPatch* prevAnchors,
                                  uint32_t frictionPatchCount, const PxAlignedTransform& bf0,
                                  const PxAlignedTransform& bf1, float correlationDistance,
                                  uint32_t totalNbEdges, float& patchExtents, uint32_t nbContacts,
                                  bool* inherited = nullptr)
{
    if (prevFrictionStartIndex == 0xFFFFFFFFu || frictionPatchCount == 0 || !hasPrevIndices)
        return;

    for (uint32_t a = 0; a < frictionPatchCount; ++a)
    {
        const PxU64 index = prevIndices[prevFrictionStartIndex + a * totalNbEdges];
        const uint32_t oldThreadId = uint32_t(index & 31u);
        const uint32_t patchIndex = uint32_t(index >> 5);

        const PxgBlockFrictionPatch& oldP = prevPatches[patchIndex];
        const PxgBlockFrictionAnchorPatch& oldA = prevAnchors[patchIndex];

        if (oldP.broken[oldThreadId] == 0)
        {
            const float4 oldBody0Normal = oldP.body0Normal[oldThreadId];
            const float4 curr0N = curr.body0Normal[lane];
            if (oldBody0Normal.x * curr0N.x + oldBody0Normal.y * curr0N.y
                    + oldBody0Normal.z * curr0N.z > 0.999f)   // PXC_SAME_NORMAL
            {
                const uint32_t anchorCount = oldP.anchorCount[oldThreadId];
                if (anchorCount != 0)
                {
                    if (anchorCount <= nbContacts)
                    {
                        const PxAlignedTransform body1ToBody0 = bf0.transformInv(bf1);
                        const float4 oldBody1Normal = oldP.body1Normal[oldThreadId];
                        const float4 rot = body1ToBody0.rotate(oldBody1Normal);
                        if (oldBody0Normal.x * rot.x + oldBody0Normal.y * rot.y
                                + oldBody0Normal.z * rot.z > 0.999f)
                        {
                            const float4 b0a0 = oldA.body0Anchors[0][oldThreadId];
                            const float4 b1a0 = oldA.body1Anchors[0][oldThreadId];
                            if (anchorCount == 0 ||
                                refPointsAreClose(body1ToBody0, b0a0, b1a0, oldBody0Normal,
                                                  correlationDistance))
                            {
                                const float4 b0a1 = oldA.body0Anchors[1][oldThreadId];
                                const float4 b1a1 = oldA.body1Anchors[1][oldThreadId];
                                if (anchorCount < 2 ||
                                    refPointsAreClose(body1ToBody0, b0a1, b1a1, oldBody0Normal,
                                                      correlationDistance))
                                {
                                    curr.contactID[0][lane] = 0xff;
                                    curr.contactID[1][lane] = 0xff;
                                    curr.anchorCount[lane] = anchorCount;
                                    curr.body0Normal[lane] = oldBody0Normal;
                                    curr.body1Normal[lane] = oldBody1Normal;
                                    currA.body0Anchors[0][lane] = b0a0;
                                    currA.body0Anchors[1][lane] = b0a1;
                                    currA.body1Anchors[0][lane] = b1a0;
                                    currA.body1Anchors[1][lane] = b1a1;
                                    const float ex = b0a0.x - b0a1.x;
                                    const float ey = b0a0.y - b0a1.y;
                                    const float ez = b0a0.z - b0a1.z;
                                    patchExtents = ex * ex + ey * ey + ez * ez;
                                    if (inherited) *inherited = true;
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void refGrowPatches(PxgBlockFrictionPatch& fp, PxgBlockFrictionAnchorPatch& fa,
                           uint32_t lane, const PxgBlockContactPoint* contacts,
                           uint32_t contactBase, uint32_t numContacts,
                           const PxAlignedTransform& bf0, const PxAlignedTransform& bf1,
                           float frictionOffsetThreshold, float anchorSqDistance,
                           bool* keptTwo = nullptr)
{
    PxVec3 boundsMin(PX_MAX_F32), boundsMax(-PX_MAX_F32);
    for (uint32_t i = 0; i < numContacts; ++i)
    {
        const float4 cp = contacts[contactBase + i].point_separationW[lane];
        boundsMin = boundsMin.minimum(PxVec3(cp.x, cp.y, cp.z));
        boundsMax = boundsMax.maximum(PxVec3(cp.x, cp.y, cp.z));
    }

    uint32_t oldAnchorCount = fp.anchorCount[lane];
    if (oldAnchorCount == 2)
    {
        const PxVec3 dims = boundsMax - boundsMin;
        const float diagSq = dims.dot(dims);
        if ((anchorSqDistance * 4.f) >= diagSq)
        {
            if (keptTwo) *keptTwo = true;
            return;
        }
        oldAnchorCount = 0;
    }

    PxVec3 worldAnchor0(0.f), worldAnchor1(0.f);
    uint32_t contactID0 = 0, contactID1 = 0;
    uint32_t anchorCount = 0;
    float pointDistSq = 0.f;

    if (oldAnchorCount == 1)
    {
        const float4 a0 = fa.body0Anchors[0][lane];
        worldAnchor0 = bf0.transform(PxVec3(a0.x, a0.y, a0.z));
        contactID0 = 0xFF;
        anchorCount++;
    }

    const float eps = 1e-8f;
    for (uint32_t j = 0; j < numContacts; ++j)
    {
        const float4 wp4 = contacts[contactBase + j].point_separationW[lane];
        if (wp4.w < frictionOffsetThreshold)
        {
            const PxVec3 worldPoint(wp4.x, wp4.y, wp4.z);
            if (anchorCount == 0)
            {
                contactID0 = j & 0xFFFFu;
                worldAnchor0 = worldPoint;
                anchorCount++;
            }
            else if (anchorCount == 1)
            {
                const PxVec3 d = worldPoint - worldAnchor0;
                pointDistSq = d.dot(d);
                if (pointDistSq > eps)
                {
                    contactID1 = j;
                    worldAnchor1 = worldPoint;
                    anchorCount++;
                }
            }
            else
            {
                const PxVec3 d0v = worldPoint - worldAnchor0;
                const PxVec3 d1v = worldPoint - worldAnchor1;
                const float dist0 = d0v.dot(d0v);
                const float dist1 = d1v.dot(d1v);
                if (dist0 > dist1)
                {
                    if (dist0 > pointDistSq)
                    {
                        contactID1 = j & 0xFFFFu;
                        worldAnchor1 = worldPoint;
                        pointDistSq = dist0;
                    }
                }
                else if (dist1 > pointDistSq)
                {
                    contactID0 = j & 0xFFFFu;
                    worldAnchor0 = worldPoint;
                    pointDistSq = dist1;
                }
            }
        }
    }

    if (anchorCount == 2)
    {
        const PxVec3 lp0 = bf0.transformInv(worldAnchor1);
        const PxVec3 lp1 = bf1.transformInv(worldAnchor1);
        fa.body0Anchors[1][lane] = make_float4(lp0.x, lp0.y, lp0.z, 0.f);
        fa.body1Anchors[1][lane] = make_float4(lp1.x, lp1.y, lp1.z, 0.f);
        fp.contactID[1][lane] = contactID1;
    }
    if (anchorCount >= 1 && oldAnchorCount == 0)
    {
        const PxVec3 lp0 = bf0.transformInv(worldAnchor0);
        const PxVec3 lp1 = bf1.transformInv(worldAnchor0);
        fa.body0Anchors[0][lane] = make_float4(lp0.x, lp0.y, lp0.z, 0.f);
        fa.body1Anchors[0][lane] = make_float4(lp1.x, lp1.y, lp1.z, 0.f);
        fp.contactID[0][lane] = contactID0;
    }

    fp.anchorCount[lane] = anchorCount;
}

#endif  // PXG_ARTI_PREP_REF_FLOAT_ONLY

}  // namespace pxgtest

#endif  // PXG_ARTI_PREP_REF_H
