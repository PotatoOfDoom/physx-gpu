// -*- mode: c++ -*-
// Included by testConstraintPrep.cpp — reference transcriptions + test drivers.
// (Kept as a separate file only to keep the two translation-unit halves short.)

namespace
{

// ===========================================================================
// contact prep reference (contactConstraintBlockPrep.cuh)
// ===========================================================================

// correlatePatches + initFrictionPatch (writes lane of the current patch)
void refCorrelatePatches(BlockFrictionPatch& fp, uint32_t lane, uint32_t nbContacts, const Vec3& normal,
                         const AFrame& bf0, const AFrame& bf1)
{
    if (nbContacts > 0)
    {
        const Vec3 body0Normal = aRotateInv(bf0.q, normal);
        const Vec3 body1Normal = aRotateInv(bf1.q, normal);
        fp.contactID0[lane] = 0xff;
        fp.contactID1[lane] = 0xff;
        fp.body0Normal[lane] = Float4{ body0Normal.x, body0Normal.y, body0Normal.z, 0.f };
        fp.body1Normal[lane] = Float4{ body1Normal.x, body1Normal.y, body1Normal.z, 0.f };
        fp.anchorCount[lane] = 0;
        fp.broken[lane] = 0;
    }
}

void refGetFrictionPatches(BlockFrictionPatch& curr, BlockFrictionAnchorPatch& currA, uint32_t lane,
                           const std::vector<uint64_t>& prevIndices, bool hasPrev, uint32_t prevStart,
                           const std::vector<BlockFrictionPatch>& prevPatches,
                           const std::vector<BlockFrictionAnchorPatch>& prevAnchors, uint32_t frictionPatchCount,
                           const AFrame& bf0, const AFrame& bf1, float correlationDistance, uint32_t totalNbEdges,
                           float& patchExtents, uint32_t nbContacts)
{
    if (prevStart == 0xFFFFFFFFu || frictionPatchCount == 0 || !hasPrev)
        return;

    for (uint32_t a = 0; a < frictionPatchCount; ++a)
    {
        const uint64_t index = prevIndices[prevStart + a * totalNbEdges];
        const uint32_t oldThreadId = uint32_t(index & 31);
        const uint64_t patchIndex = index >> 5;

        const BlockFrictionPatch& oldP = prevPatches[patchIndex];
        const BlockFrictionAnchorPatch& oldA = prevAnchors[patchIndex];

        if (oldP.broken[oldThreadId] == 0)
        {
            const Float4 oldBody0Normal = oldP.body0Normal[oldThreadId];
            if (dot3f4(oldBody0Normal, curr.body0Normal[lane]) > PXC_SAME_NORMAL)
            {
                const uint32_t anchorCount = oldP.anchorCount[oldThreadId];
                if (anchorCount != 0)
                {
                    if (anchorCount <= nbContacts)
                    {
                        const AFrame body1ToBody0 = aTransformInvFrame(bf0, bf1);
                        const Float4 oldBody1Normal = oldP.body1Normal[oldThreadId];
                        const Vec3 rot = aRotate(body1ToBody0.q, Vec3(oldBody1Normal.x, oldBody1Normal.y, oldBody1Normal.z));
                        const float result = oldBody0Normal.x * rot.x + oldBody0Normal.y * rot.y + oldBody0Normal.z * rot.z;
                        if (result > PXC_SAME_NORMAL)
                        {
                            const Float4 b0a0 = oldA.body0Anchors0[oldThreadId];
                            const Float4 b1a0 = oldA.body1Anchors0[oldThreadId];
                            if (anchorCount == 0 || pointsAreClose(body1ToBody0, b0a0, b1a0, oldBody0Normal, correlationDistance))
                            {
                                const Float4 b0a1 = oldA.body0Anchors1[oldThreadId];
                                const Float4 b1a1 = oldA.body1Anchors1[oldThreadId];
                                if (anchorCount < 2 || pointsAreClose(body1ToBody0, b0a1, b1a1, oldBody0Normal, correlationDistance))
                                {
                                    curr.contactID0[lane] = 0xff;
                                    curr.contactID1[lane] = 0xff;
                                    curr.anchorCount[lane] = anchorCount;
                                    curr.body0Normal[lane] = oldBody0Normal;
                                    curr.body1Normal[lane] = oldBody1Normal;
                                    currA.body0Anchors0[lane] = b0a0;
                                    currA.body0Anchors1[lane] = b0a1;
                                    currA.body1Anchors0[lane] = b1a0;
                                    currA.body1Anchors1[lane] = b1a1;
                                    const Float4 ext{ b0a0.x - b0a1.x, b0a0.y - b0a1.y, b0a0.z - b0a1.z, 0.f };
                                    patchExtents = ext.x * ext.x + ext.y * ext.y + ext.z * ext.z;
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

void refGrowPatches(BlockFrictionPatch& fp, BlockFrictionAnchorPatch& fa, uint32_t lane,
                    const std::vector<BlockContactPoint>& contacts, uint32_t contactBase, uint32_t numContacts,
                    const AFrame& bf0, const AFrame& bf1, float frictionOffsetThreshold, float anchorSqDistance)
{
    Vec3 bmin(PX_MAX_F32, PX_MAX_F32, PX_MAX_F32);
    Vec3 bmax(-PX_MAX_F32, -PX_MAX_F32, -PX_MAX_F32);
    for (uint32_t i = 0; i < numContacts; ++i)
    {
        const Float4 cp = contacts[contactBase + i].point_separationW[lane];
        bmin.x = std::fmin(cp.x, bmin.x); bmin.y = std::fmin(cp.y, bmin.y); bmin.z = std::fmin(cp.z, bmin.z);
        bmax.x = std::fmax(cp.x, bmax.x); bmax.y = std::fmax(cp.y, bmax.y); bmax.z = std::fmax(cp.z, bmax.z);
    }

    uint32_t oldAnchorCount = fp.anchorCount[lane];
    if (oldAnchorCount == 2)
    {
        const Vec3 dims(bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z);
        const float frictionPatchDiagonalSq = dims.x * dims.x + dims.y * dims.y + dims.z * dims.z;
        if ((anchorSqDistance * 4.f) >= frictionPatchDiagonalSq)
            return;
        oldAnchorCount = 0;
    }

    Vec3 worldAnchors[2] = { Vec3(0, 0, 0), Vec3(0, 0, 0) };
    uint32_t contactID[2] = { 0, 0 };
    uint32_t anchorCount = 0;
    float pointDistSq = 0.0f;

    if (oldAnchorCount == 1)
    {
        const Float4 v = fa.body0Anchors0[lane];
        worldAnchors[0] = aTransform(bf0, Vec3(v.x, v.y, v.z));
        contactID[0] = 0xFF;
        anchorCount++;
    }

    const float eps = 1e-8f;
    for (uint32_t j = 0; j < numContacts; j++)
    {
        const Float4 wp = contacts[contactBase + j].point_separationW[lane];
        if (wp.w < frictionOffsetThreshold)
        {
            const Vec3 worldPoint(wp.x, wp.y, wp.z);
            if (anchorCount == 0)
            {
                contactID[0] = j & 0xFFFF;
                worldAnchors[0] = worldPoint;
                anchorCount++;
            }
            else if (anchorCount == 1)
            {
                const Vec3 d = worldPoint - worldAnchors[0];
                pointDistSq = d.x * d.x + d.y * d.y + d.z * d.z;
                if (pointDistSq > eps)
                {
                    contactID[1] = j;
                    worldAnchors[1] = worldPoint;
                    anchorCount++;
                }
            }
            else
            {
                const Vec3 d0 = worldPoint - worldAnchors[0];
                const Vec3 d1 = worldPoint - worldAnchors[1];
                const float dist0 = d0.x * d0.x + d0.y * d0.y + d0.z * d0.z;
                const float dist1 = d1.x * d1.x + d1.y * d1.y + d1.z * d1.z;
                if (dist0 > dist1)
                {
                    if (dist0 > pointDistSq)
                    {
                        contactID[1] = j & 0xFFFF;
                        worldAnchors[1] = worldPoint;
                        pointDistSq = dist0;
                    }
                }
                else if (dist1 > pointDistSq)
                {
                    contactID[0] = j & 0xFFFF;
                    worldAnchors[0] = worldPoint;
                    pointDistSq = dist1;
                }
            }
        }
    }

    if (anchorCount == 2)
    {
        const Vec3 lp0 = aTransformInv(bf0, worldAnchors[1]);
        const Vec3 lp1 = aTransformInv(bf1, worldAnchors[1]);
        fa.body0Anchors1[lane] = Float4{ lp0.x, lp0.y, lp0.z, 0.f };
        fa.body1Anchors1[lane] = Float4{ lp1.x, lp1.y, lp1.z, 0.f };
        fp.contactID1[lane] = contactID[1];
    }
    if (anchorCount >= 1 && oldAnchorCount == 0)
    {
        const Vec3 lp0 = aTransformInv(bf0, worldAnchors[0]);
        const Vec3 lp1 = aTransformInv(bf1, worldAnchors[0]);
        fa.body0Anchors0[lane] = Float4{ lp0.x, lp0.y, lp0.z, 0.f };
        fa.body1Anchors0[lane] = Float4{ lp1.x, lp1.y, lp1.z, 0.f };
        fp.contactID0[lane] = contactID[0];
    }

    fp.anchorCount[lane] = anchorCount;
}

// setupFinalizeSolverConstraintsBlock (PGS)
void refSetupPGS(BlockContactData& cd, const std::vector<BlockContactPoint>& contacts, uint32_t contactBase,
                 uint32_t contactCount, BlockFrictionPatch& fp, BlockFrictionAnchorPatch& fa, const AFrame& bf0,
                 const AFrame& bf1, const SolverBodyData& d0, const SolverBodyData& d1, const Mat33& si0,
                 const Mat33& si1, float invDtF32, float dtF32, float bounceThresholdF32, uint32_t lane,
                 uint32_t forceWritebackOffset, bool hasTargetVelocity, BlockSolverContactHeader& ch,
                 BlockSolverFrictionHeader& fh, std::vector<BlockSolverContactPoint>& cpoints, uint32_t outCBase,
                 std::vector<BlockSolverContactFriction>& fricts, uint32_t outFBase, float restDistance,
                 float staticFrictionCof, float dynamicFrictionCof, uint32_t flags, float ccdMaxSeparationThreshold,
                 float solverOffsetSlop)
{
    const Float4 initLin0 = d0.initialLinVelXYZ_invMassW, initAng0 = d0.initialAngVelXYZ_penBiasClamp;
    const Float4 initLin1 = d1.initialLinVelXYZ_invMassW, initAng1 = d1.initialAngVelXYZ_penBiasClamp;
    const Vec3 bf0p = bf0.p, bf1p = bf1.p;

    const bool isCCD = ((d0.flags | d1.flags) & RB_SPEC_CCD) != 0;
    const float ccdMaxSeparation = isCCD ? ccdMaxSeparationThreshold : PX_MAX_F32;

    const float maxPenBias = std::fmax(initAng0.w, initAng1.w);
    const Vec3 linVel0(initLin0.x, initLin0.y, initLin0.z);
    const Vec3 linVel1(initLin1.x, initLin1.y, initLin1.z);
    const Vec3 angVel0(initAng0.x, initAng0.y, initAng0.z);
    const Vec3 angVel1(initAng1.x, initAng1.y, initAng1.z);

    // 5.12: the prepare desc's biasCoefficients.rigidContact (5.6.1: p8 = 0.8f)
    const float invDt = invDtF32, bounceThreshold = bounceThresholdF32;
    const float invDtp8 = invDt * PGS_RIGID_CONTACT_BIAS;

    const Float4 ims = cd.mInvMassScale[lane];
    const float invMass0D0 = initLin0.w * ims.x;
    const float invMass1D1 = initLin1.w * ims.z;

    const uint32_t anchorCount = fp.anchorCount[lane] * 2;
    if (contactCount == 0)
    {
        fh.numFrictionConstr[lane] = 0;
        ch.numNormalConstr[lane] = 0;
        ch.forceWritebackOffset[lane] = 0xffffffffu;
        return;
    }

    // PxMin<float> is intrinsics::selectMin (a < b ? a : b; a +-0 tie keeps b), not fmin
    // (contactConstraintBlockPrep.cuh:402)
    const float maxImpulse = (d0.maxImpulse < d1.maxImpulse) ? d0.maxImpulse : d1.maxImpulse;
    ch.forceWritebackOffset[lane] = forceWritebackOffset;
    ch.flags[lane] = flags;
    ch.invMass0_1_angDom0_1[lane] = Float4{ invMass0D0, invMass1D1, ims.y, ims.w };

    const Float4 n4 = cd.normal_restitutionW[lane];
    const Vec3 normal(n4.x, n4.y, n4.z);
    const float restitution = n4.w;
    const float damping = cd.damping[lane];
    const float normalLenSq = normal.dot(normal);
    const float linNorVel = normal.dot(linVel0) - normal.dot(linVel1);
    const float invMassNorLenSq0 = invMass0D0 * normalLenSq;
    const float invMassNorLenSq1 = invMass1D1 * normalLenSq;
    ch.restitution[lane] = restitution;

    for (uint32_t j = 0; j < contactCount; j++)
    {
        const Float4 ps = contacts[contactBase + j].point_separationW[lane];
        const Vec3 point(ps.x, ps.y, ps.z);
        const float separation = ps.w;
        const Float4 tv4 = contacts[contactBase + j].targetVel_maxImpulseW[lane];
        const Vec3 targetVel(tv4.x, tv4.y, tv4.z);
        const float cTargetVel = normal.dot(targetVel);

        const Vec3 ra = point - bf0p, rb = point - bf1p;
        Vec3 raXn = ra.cross(normal), rbXn = rb.cross(normal);
        const float angNorVel = raXn.dot(angVel0) - rbXn.dot(angVel1);
        const float slopRatio = linNorVel == 0.f ? 1.f : angNorVel / linNorVel;
        const float slop = solverOffsetSlop * ((slopRatio > 1.f) ? slopRatio : 1.f); // PxMax = selectMax (cuh:448)
        raXn.x = std::fabs(raXn.x) < slop ? 0.f : raXn.x;
        raXn.y = std::fabs(raXn.y) < slop ? 0.f : raXn.y;
        raXn.z = std::fabs(raXn.z) < slop ? 0.f : raXn.z;
        rbXn.x = std::fabs(rbXn.x) < slop ? 0.f : rbXn.x;
        rbXn.y = std::fabs(rbXn.y) < slop ? 0.f : rbXn.y;
        rbXn.z = std::fabs(rbXn.z) < slop ? 0.f : rbXn.z;
        const Vec3 raI = si0.transform(raXn), rbI = si1.transform(rbXn);
        const float resp0 = ims.y * raI.dot(raI) + invMassNorLenSq0;
        const float resp1 = ims.w * rbI.dot(rbI) + invMassNorLenSq1;
        const float vrel = linNorVel + (raXn.dot(angVel0) - rbXn.dot(angVel1));

        const float penetration = separation - restDistance;
        const bool isSeparated = (penetration >= 0.0f);
        const float penetrationInvDt = penetration * invDt;
        const float penetrationInvDtPt8 = std::fmax(maxPenBias, penetration * invDtp8);
        const bool collidingWithVrel = ((-vrel) > penetrationInvDt);
        const bool isGreater2 = ((restitution > 0.f) && (bounceThreshold > vrel)) && collidingWithVrel;
        const bool ccdSeparationCondition = ccdMaxSeparation >= penetration;

        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.f);
        targetVelocity = targetVelocity - vrel;

        float coeff0, coeff1;
        if (restitution < 0.f)
            queryReducedCompliantContactCoefficients(dtF32, flags, restitution, damping, penetration, targetVelocity,
                                                      isSeparated, collidingWithVrel, coeff0, coeff1);
        else
        {
            float scaledBias = isSeparated ? penetrationInvDt : penetrationInvDtPt8;
            if (ccdSeparationCondition && isGreater2)
                scaledBias = 0.f;
            float unbiasedErr = targetVelocity;
            const float biasedErr = unbiasedErr - scaledBias;
            if (!isGreater2) // PxMax(0.f, scaledBias) = selectMax: keeps a -0 scaledBias (cuh:506)
                unbiasedErr -= (0.f > scaledBias) ? 0.f : scaledBias;
            coeff0 = biasedErr;
            coeff1 = unbiasedErr;
        }

        BlockSolverContactPoint& sc = cpoints[outCBase + j];
        sc.raXn_targetVelocity[lane] = Float4{ raI.x, raI.y, raI.z, targetVelocity };
        // PxMin = selectMin (contactConstraintBlockPrep.cuh:513): PxMin(-0, +0) = +0
        sc.rbXn_maxImpulse[lane] = Float4{ rbI.x, rbI.y, rbI.z, (maxImpulse < tv4.w) ? maxImpulse : tv4.w };
        sc.resp0[lane] = resp0;
        sc.resp1[lane] = resp1;
        sc.coeff0[lane] = coeff0;
        sc.coeff1[lane] = coeff1;
        sc.appliedForce[lane] = 0.f;
    }

    const uint32_t aCount = fp.anchorCount[lane];
    const float frictionScale = (aCount == 2) ? 0.5f : 1.f;
    const float staticFriction = staticFrictionCof * frictionScale;
    const float dynamicFriction = dynamicFrictionCof * frictionScale;
    const bool haveFriction = (anchorCount != 0);
    ch.numNormalConstr[lane] = contactCount;
    fh.numFrictionConstr[lane] = anchorCount;
    ch.normal_staticFriction[lane] = Float4{ normal.x, normal.y, normal.z, staticFriction };
    fh.dynamicFriction[lane] = dynamicFriction;

    if (haveFriction)
    {
        const Vec3 vrel = linVel0 - linVel1;
        const float orthoThreshold = 0.70710678f, p1 = 0.0001f;
        const float nx = normal.x, ny = normal.y, nz = normal.z;
        const Vec3 t0Fallback1(0.f, -nz, ny), t0Fallback2(-ny, nx, 0.f);
        const Vec3 t0Fallback = orthoThreshold > std::fabs(nx) ? t0Fallback1 : t0Fallback2;
        Vec3 t0 = vrel - normal * normal.dot(vrel);
        t0 = t0.dot(t0) > p1 ? t0 : t0Fallback;
        t0 = t0.getNormalized();
        const Vec3 t1 = normal.cross(t0);
        fh.broken[lane] = 0;
        fh.frictionNormals0[lane] = Float4{ t0.x, t0.y, t0.z, 0.f };
        fh.frictionNormals1[lane] = Float4{ t1.x, t1.y, t1.z, 0.f };

        for (uint32_t j = 0; j < aCount; j++)
        {
            const Float4 b0a = j == 0 ? fa.body0Anchors0[lane] : fa.body0Anchors1[lane];
            const Float4 b1a = j == 0 ? fa.body1Anchors0[lane] : fa.body1Anchors1[lane];
            const Vec3 ra = aRotate(bf0.q, Vec3(b0a.x, b0a.y, b0a.z));
            const Vec3 rb = aRotate(bf1.q, Vec3(b1a.x, b1a.y, b1a.z));
            const Vec3 error = (ra + bf0p) - (rb + bf1p);
            const uint32_t idx = hasTargetVelocity ? (j == 0 ? fp.contactID0[lane] : fp.contactID1[lane]) : 0;
            const Float4 tv4 = contacts[contactBase + idx].targetVel_maxImpulseW[lane];
            const Vec3 tvel(tv4.x, tv4.y, tv4.z);

            for (int axis = 0; axis < 2; ++axis)
            {
                const Vec3 t = axis == 0 ? t0 : t1;
                Vec3 raXn = ra.cross(t), rbXn = rb.cross(t);
                raXn.x = std::fabs(raXn.x) < solverOffsetSlop ? 0.f : raXn.x;
                raXn.y = std::fabs(raXn.y) < solverOffsetSlop ? 0.f : raXn.y;
                raXn.z = std::fabs(raXn.z) < solverOffsetSlop ? 0.f : raXn.z;
                rbXn.x = std::fabs(rbXn.x) < solverOffsetSlop ? 0.f : rbXn.x;
                rbXn.y = std::fabs(rbXn.y) < solverOffsetSlop ? 0.f : rbXn.y;
                rbXn.z = std::fabs(rbXn.z) < solverOffsetSlop ? 0.f : rbXn.z;
                const Vec3 raI = si0.transform(raXn), rbI = si1.transform(rbXn);
                const float resp0 = ims.y * raI.dot(raI) + invMass0D0;
                const float resp1 = ims.w * rbI.dot(rbI) + invMass1D1;
                float targetVel = tvel.dot(t);
                const float vrelF = (t.dot(linVel0) + raXn.dot(angVel0)) - (t.dot(linVel1) + rbXn.dot(angVel1));
                targetVel = targetVel - vrelF;
                const float bias = t.dot(error) * invDt;
                BlockSolverContactFriction& f = fricts[outFBase + 2 * j + axis];
                f.resp0[lane] = resp0;
                f.resp1[lane] = resp1;
                f.raXn_bias[lane] = Float4{ raI.x, raI.y, raI.z, bias };
                f.rbXn_targetVelW[lane] = Float4{ rbI.x, rbI.y, rbI.z, targetVel };
                f.appliedForce[lane] = 0.f;
            }
        }
    }
}

// PxMin<float>/PxMax<float> are intrinsics::selectMin/selectMax (PxMath.h:82-101,
// PxUnixMathIntrinsics.h:116-125): a < b ? a : b and a > b ? a : b -- a +-0 tie (or a NaN
// compare) returns the SECOND operand.  std::fmin/std::fmax leave the sign of a zero tie
// unspecified (glibc's generic fmin keeps the FIRST operand), so the TGS mirror spells the
// CUDA's PxMin/PxMax with these (P9.86c).
inline float cpSelMin(float a, float b) { return (a < b) ? a : b; }
inline float cpSelMax(float a, float b) { return (a > b) ? a : b; }

// setupFinalizeSolverConstraintsBlockTGS
void refSetupTGS(BlockContactData& cd, const std::vector<BlockContactPoint>& contacts, uint32_t contactBase,
                 uint32_t contactCount, BlockFrictionPatch& fp, BlockFrictionAnchorPatch& fa, const AFrame& bf0,
                 const AFrame& bf1, const SolverBodyData& d0, const SolverBodyData& d1, const Mat33& si0,
                 const Mat33& si1, float invStepDtF32, float stepDt, float totalDt, float invTotalDtF32,
                 float bounceThresholdF32, float biasCoefficient, uint32_t lane, uint32_t forceWritebackOffset,
                 bool hasTargetVelocity, TGSContactHeader& ch, TGSFrictionHeader& fh,
                 std::vector<TGSContactPoint>& cpoints, uint32_t outCBase, std::vector<TGSFriction>& fricts,
                 uint32_t outFBase, float restDistance, float staticFrictionCof, float dynamicFrictionCof,
                 uint32_t flags, float solverOffsetSlop, const Float2& torsionalFrictionData)
{
    const Float4 initLin0 = d0.initialLinVelXYZ_invMassW, initAng0 = d0.initialAngVelXYZ_penBiasClamp;
    const Float4 initLin1 = d1.initialLinVelXYZ_invMassW, initAng1 = d1.initialAngVelXYZ_penBiasClamp;
    const Vec3 bf0p = bf0.p, bf1p = bf1.p;

    const float maxPenBias = std::fmax(initAng0.w, initAng1.w);
    const Vec3 linVel0(initLin0.x, initLin0.y, initLin0.z);
    const Vec3 linVel1(initLin1.x, initLin1.y, initLin1.z);
    const Vec3 angVel0(initAng0.x, initAng0.y, initAng0.z);
    const Vec3 angVel1(initAng1.x, initAng1.y, initAng1.z);

    const float invStepDt = invStepDtF32;
    // 5.12: the bias coefficient is used directly (5.6.1: p8 = PxMin(0.8f, biasCoefficient))
    const float bounceThreshold = bounceThresholdF32;
    const float invDtp8 = invStepDt * biasCoefficient;

    const Float4 ims = cd.mInvMassScale[lane];
    const float invMass0D0 = initLin0.w * ims.x;
    const float invMass1D1 = initLin1.w * ims.z;
    const float angD0 = ims.y, angD1 = ims.w;

    const bool isKin0 = (d0.flags & RB_KINEMATIC) != 0;
    const bool isKin1 = (d1.flags & RB_KINEMATIC) != 0;

    const uint32_t aCount = fp.anchorCount[lane];
    const uint32_t anchorCount = aCount * 2;
    if (contactCount == 0)
    {
        fh.numFrictionConstr[lane] = 0;
        ch.numNormalConstr[lane] = 0;
        ch.forceWritebackOffset[lane] = 0xffffffffu;
        return;
    }

    const float maxImpulse = cpSelMin(d0.maxImpulse, d1.maxImpulse); // PxMin (cuh:846)
    ch.forceWritebackOffset[lane] = forceWritebackOffset;
    ch.flags[lane] = flags;
    ch.maxPenBias[lane] = maxPenBias;
    ch.invMass0_1_angDom0_1[lane] = Float4{ invMass0D0, invMass1D1, ims.y, ims.w };

    const Float4 n4 = cd.normal_restitutionW[lane];
    const Vec3 normal(n4.x, n4.y, n4.z);
    const float damping = cd.damping[lane];
    const float restitution = n4.w;
    const float normalLenSq = normal.dot(normal);
    const float norVel0 = normal.dot(linVel0), norVel1 = normal.dot(linVel1);
    const float linNorVel = norVel0 - norVel1;
    const float invMassNorLenSq0 = invMass0D0 * normalLenSq;
    const float invMassNorLenSq1 = invMass1D1 * normalLenSq;
    float maxPenetration = PX_MAX_F32;
    ch.restitutionXdt[lane] = restitution * stepDt;
    ch.biasCoefficient[lane] = biasCoefficient;

    for (uint32_t j = 0; j < contactCount; j++)
    {
        const Float4 ps = contacts[contactBase + j].point_separationW[lane];
        const Vec3 point(ps.x, ps.y, ps.z);
        const float separation = ps.w;
        const Float4 tv4 = contacts[contactBase + j].targetVel_maxImpulseW[lane];
        const Vec3 targetVel(tv4.x, tv4.y, tv4.z);
        const float cTargetVel = normal.dot(targetVel);

        const Vec3 ra = point - bf0p, rb = point - bf1p;
        Vec3 raXn = ra.cross(normal), rbXn = rb.cross(normal);
        const float angNorVel = raXn.dot(angVel0) - rbXn.dot(angVel1);
        const float slop = solverOffsetSlop * cpSelMax(linNorVel == 0.f ? 1.f : angNorVel / linNorVel, 1.f); // cuh:898
        raXn.x = std::fabs(raXn.x) < slop ? 0.f : raXn.x;
        raXn.y = std::fabs(raXn.y) < slop ? 0.f : raXn.y;
        raXn.z = std::fabs(raXn.z) < slop ? 0.f : raXn.z;
        rbXn.x = std::fabs(rbXn.x) < slop ? 0.f : rbXn.x;
        rbXn.y = std::fabs(rbXn.y) < slop ? 0.f : rbXn.y;
        rbXn.z = std::fabs(rbXn.z) < slop ? 0.f : rbXn.z;
        const Vec3 raI = si0.transform(raXn), rbI = si1.transform(rbXn);
        const float resp0 = raI.dot(raI) * angD0 + invMassNorLenSq0;
        const float resp1 = rbI.dot(rbI) * angD1 + invMassNorLenSq1;
        const float unitResponse = resp0 + resp1;
        const float angNorVel0 = raXn.dot(angVel0), angNorVel1 = rbXn.dot(angVel1);
        const float vrel = linNorVel + (angNorVel0 - angNorVel1);

        float scaledBias = 0.f, velMultiplier = 0.f;
        float penetration = separation - restDistance;
        const bool isSeparated = (penetration > 0.0f);
        const float penetrationInvDt = penetration * invTotalDtF32;
        const float recipResponse = (unitResponse > 0.f) ? (1.f / unitResponse) : 0.f;
        float contactCoeff = 0.f;
        if (restitution < 0.f)
        {
            const bool collidingWithVrel = ((-vrel * totalDt) > penetration);
            contactCoeff = computeCompliantContactCoefficientsTGS(stepDt, flags, restitution, damping, unitResponse,
                                                                  recipResponse, isSeparated, collidingWithVrel,
                                                                  velMultiplier, scaledBias);
        }
        else
        {
            velMultiplier = recipResponse;
            scaledBias = isSeparated ? -invStepDt : -invDtp8;
        }
        cpoints[outCBase + j].resp0[lane] = resp0;
        cpoints[outCBase + j].resp1[lane] = resp1;
        maxPenetration = cpSelMin(penetration, maxPenetration); // PxMin (cuh:947): a zero tie keeps the old value
        const bool isGreater2 = ((restitution > 0.f) && (bounceThreshold > vrel)) && ((-vrel) > penetrationInvDt);
        // the CUDA's IEEE add (cuh:951): cTargetVel = -0 plus the selected +0 gives +0
        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.f);
        // 5.12: the bounce penetration correction comes BEFORE the kinematic terms
        if (isGreater2)
        {
            const float ratio = totalDt + penetration / vrel;
            penetration += targetVelocity * ratio;
        }
        if (isKin0)
            targetVelocity -= (norVel0 + angNorVel0);
        if (isKin1)
            targetVelocity += (norVel1 + angNorVel1);
        cpoints[outCBase + j].raXn_extraCoeff[lane] = Float4{ raI.x, raI.y, raI.z, contactCoeff };
        cpoints[outCBase + j].rbXn_targetVelW[lane] = Float4{ rbI.x, rbI.y, rbI.z, targetVelocity };
        cpoints[outCBase + j].separation[lane] = penetration;
        cpoints[outCBase + j].maxImpulse[lane] = cpSelMin(maxImpulse, tv4.w); // PxMin (cuh:969): PxMin(-0,+0) = +0
        cpoints[outCBase + j].appliedForce[lane] = 0.f;
        cpoints[outCBase + j].biasCoefficient[lane] = scaledBias;
    }

    const float frictionScale = (aCount == 2) ? 0.5f : 1.f;
    const bool hasTorsionalFriction = (torsionalFrictionData.x > 0.f || torsionalFrictionData.y > 0.f) && aCount == 1;
    const float staticFriction = staticFrictionCof * frictionScale;
    const float dynamicFriction = dynamicFrictionCof * frictionScale;
    const bool haveFriction = (anchorCount != 0);
    ch.numNormalConstr[lane] = contactCount;
    fh.numFrictionConstr[lane] = anchorCount + (hasTorsionalFriction ? 1u : 0u);
    ch.normal_staticFriction[lane] = Float4{ normal.x, normal.y, normal.z, staticFriction };
    fh.dynamicFriction[lane] = dynamicFriction;

    if (haveFriction)
    {
        const Vec3 vrel = linVel0 - linVel1;
        const float orthoThreshold = 0.70710678f, p1 = 0.0001f;
        const float nx = normal.x, ny = normal.y, nz = normal.z;
        const Vec3 t0Fallback1(0.f, -nz, ny), t0Fallback2(-ny, nx, 0.f);
        const Vec3 t0Fallback = orthoThreshold > std::fabs(nx) ? t0Fallback1 : t0Fallback2;
        Vec3 t0 = vrel - normal * normal.dot(vrel);
        t0 = t0.dot(t0) > p1 ? t0 : t0Fallback;
        t0 = t0.getNormalized();
        const Vec3 t1 = normal.cross(t0);
        fh.broken[lane] = 0;
        fh.frictionNormals0[lane] = Float4{ t0.x, t0.y, t0.z, 0.f };
        fh.frictionNormals1[lane] = Float4{ t1.x, t1.y, t1.z, 0.f };
        // 5.12: no geometric friction bias when strong friction is disabled
        fh.biasCoefficient[lane] = (flags & SC_DISABLE_STRONG_FRICTION) ? 0.0f : invStepDt;

        for (uint32_t j = 0; j < aCount; j++)
        {
            const Float4 b0a = j == 0 ? fa.body0Anchors0[lane] : fa.body0Anchors1[lane];
            const Float4 b1a = j == 0 ? fa.body1Anchors0[lane] : fa.body1Anchors1[lane];
            const Vec3 ra = aRotate(bf0.q, Vec3(b0a.x, b0a.y, b0a.z));
            const Vec3 rb = aRotate(bf1.q, Vec3(b1a.x, b1a.y, b1a.z));
            const Vec3 error = (ra + bf0p) - (rb + bf1p);
            const uint32_t idx = hasTargetVelocity ? (j == 0 ? fp.contactID0[lane] : fp.contactID1[lane]) : 0;
            const Float4 tv4 = contacts[contactBase + idx].targetVel_maxImpulseW[lane];
            const Vec3 tvel(tv4.x, tv4.y, tv4.z);
            const Vec3 vel1 = linVel0 + angVel0.cross(ra);
            const Vec3 vel2 = linVel1 + angVel1.cross(rb);
            Vec3 tv = tvel;
            if (isKin0)
                tv = tv - vel1;
            if (isKin1)
                tv = tv + vel2;

            for (int axis = 0; axis < 2; ++axis)
            {
                const Vec3 t = axis == 0 ? t0 : t1;
                Vec3 raXn = ra.cross(t), rbXn = rb.cross(t);
                raXn.x = std::fabs(raXn.x) < solverOffsetSlop ? 0.f : raXn.x;
                raXn.y = std::fabs(raXn.y) < solverOffsetSlop ? 0.f : raXn.y;
                raXn.z = std::fabs(raXn.z) < solverOffsetSlop ? 0.f : raXn.z;
                rbXn.x = std::fabs(rbXn.x) < solverOffsetSlop ? 0.f : rbXn.x;
                rbXn.y = std::fabs(rbXn.y) < solverOffsetSlop ? 0.f : rbXn.y;
                rbXn.z = std::fabs(rbXn.z) < solverOffsetSlop ? 0.f : rbXn.z;
                const Vec3 raI = si0.transform(raXn), rbI = si1.transform(rbXn);
                const float resp0 = angD0 * raI.dot(raI) + invMass0D0;
                const float resp1 = angD1 * rbI.dot(rbI) + invMass1D1;
                const float targetVel = tv.dot(t);
                const float frictionError = t.dot(error);
                TGSFriction& f = fricts[outFBase + 2 * j + axis];
                f.resp0[lane] = resp0;
                f.resp1[lane] = resp1;
                f.raXn_error[lane] = Float4{ raI.x, raI.y, raI.z, frictionError };
                f.rbXn_targetVelW[lane] = Float4{ rbI.x, rbI.y, rbI.z, targetVel };
                f.appliedForce[lane] = 0.f;
            }
        }

        if (hasTorsionalFriction)
        {
            const float torsionalPatchRadius = torsionalFrictionData.x;
            const float minTorsionalPatchRadius = torsionalFrictionData.y;
            // PxMax = selectMax twice (cuh:1128): maxPenetration = +0 gives -0 * R -> sqrt(-0) = -0
            const float torsionalFriction =
                cpSelMax(minTorsionalPatchRadius, std::sqrt(cpSelMax(0.f, -maxPenetration) * torsionalPatchRadius));
            fh.torsionalFrictionScale[lane] = torsionalFriction;
            const Vec3 raXnI = si0.transform(normal);
            const Vec3 rbXnI = si1.transform(normal);
            const float resp0 = angD0 * raXnI.dot(raXnI);
            const float resp1 = angD1 * rbXnI.dot(rbXnI);
            float targetVel = 0.f;
            if (isKin0)
                targetVel += normal.dot(angVel0);
            if (isKin1)
                targetVel -= normal.dot(angVel1);
            TGSFriction& f = fricts[outFBase + 2];
            f.resp0[lane] = resp0;
            f.resp1[lane] = resp1;
            f.raXn_error[lane] = Float4{ raXnI.x, raXnI.y, raXnI.z, 0.f };
            f.rbXn_targetVelW[lane] = Float4{ rbXnI.x, rbXnI.y, rbXnI.z, targetVel };
            f.appliedForce[lane] = 0.f;
        }
    }
}
} // namespace

#include "testConstraintPrep3.inc"
