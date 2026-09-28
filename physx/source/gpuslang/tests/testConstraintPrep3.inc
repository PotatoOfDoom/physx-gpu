// -*- mode: c++ -*-
// Included by testConstraintPrep2.inc — joint-prep reference + all test drivers.

namespace
{

// ===========================================================================
// joint prep reference (jointConstraintBlockPrep(TGS).cuh, DyCpuGpu1dConstraint.h)
// ===========================================================================

// orthogonalize() over rows [start, start+rowCount) of the (sorted) row list;
// modifies rvs in place
void refOrthogonalize(uint32_t* sorted, uint32_t startOff, std::vector<BlockConstraint1DVelocities>& rvs,
                      uint32_t rowBase, Vec3* angS0, Vec3* angS1, uint32_t rowCount, uint32_t eqRowCount,
                      float invMass0, float invMass1, float invInertiaScale0, float invInertiaScale1, uint32_t lane)
{
    Vec3 lin1m[6], ang1m[6], lin1[6], ang1[6];
    Vec3 lin0m[6], ang0m[6], lin0[6], ang0[6];
    float geomErr[6], velTarget[6];

    for (uint32_t i = 0; i < rowCount; i++)
    {
        const uint32_t index = rowBase + sorted[startOff + i];
        const Float4 l0g = rvs[index].linear0XYZ_geometricErrorW[lane];
        const Float4 l1m = rvs[index].linear1XYZ_minImpulseW[lane];
        const Float4 a0v = rvs[index].angular0XYZ_velocityTargetW[lane];
        const Float4 a1m = rvs[index].angular1XYZ_maxImpulseW[lane];

        Vec3 vl0(l0g.x, l0g.y, l0g.z), va0(a0v.x, a0v.y, a0v.z);
        Vec3 vl1(l1m.x, l1m.y, l1m.z), va1(a1m.x, a1m.y, a1m.z);
        Vec3 angSqrtL0 = angS0[startOff + i], angSqrtL1 = angS1[startOff + i];
        float g = l0g.w, T = a0v.w;

        const uint32_t elim = i < eqRowCount ? i : eqRowCount;
        for (uint32_t j = 0; j < elim; j++)
        {
            const Vec3 s0 = vl1.multiply(lin1m[j]) + vl0.multiply(lin0m[j]);
            const Vec3 s1 = angSqrtL1.multiply(ang1m[j]) + angSqrtL0.multiply(ang0m[j]);
            const Vec3 s0s1 = s0 + s1;
            const float t = s0s1.x + s0s1.y + s0s1.z;
            vl0 = vl0 - lin0[j] * t;
            va0 = va0 - ang0[j] * t;
            vl1 = vl1 - lin1[j] * t;
            va1 = va1 - ang1[j] * t;
            g = g - geomErr[j] * t;
            T = T - velTarget[j] * t;
            angSqrtL0 = angSqrtL0 - angS0[startOff + j] * t;
            angSqrtL1 = angSqrtL1 - angS1[startOff + j] * t;
        }

        rvs[index].linear0XYZ_geometricErrorW[lane] = Float4{ vl0.x, vl0.y, vl0.z, g };
        rvs[index].angular0XYZ_velocityTargetW[lane] = Float4{ va0.x, va0.y, va0.z, T };
        rvs[index].linear1XYZ_minImpulseW[lane] = Float4{ vl1.x, vl1.y, vl1.z, l1m.w };
        rvs[index].angular1XYZ_maxImpulseW[lane] = Float4{ va1.x, va1.y, va1.z, a1m.w };
        angS0[startOff + i] = angSqrtL0;
        angS1[startOff + i] = angSqrtL1;

        if (i < eqRowCount)
        {
            lin0[i] = vl0; ang0[i] = va0; geomErr[i] = g; velTarget[i] = T; lin1[i] = vl1; ang1[i] = va1;
            const Vec3 l0mv = vl0 * invMass0;
            const Vec3 l1mv = vl1 * invMass1;
            const Vec3 a0mv = angSqrtL0 * invInertiaScale0;
            const Vec3 a1mv = angSqrtL1 * invInertiaScale1;
            const Vec3 s0 = vl0.multiply(l0mv) + vl1.multiply(l1mv);
            const Vec3 s1 = a0mv.multiply(angSqrtL0) + a1mv.multiply(angSqrtL1);
            const Vec3 s0s1 = s0 + s1;
            const float s = s0s1.x + s0s1.y + s0s1.z;
            const float a = s > 0 ? 1.f / s : 0.f;
            lin0m[i] = l0mv * a; ang0m[i] = a0mv * a; lin1m[i] = l1mv * a; ang1m[i] = a1mv * a;
        }
    }
}

void refPreprocessRows(uint32_t* sorted, const std::vector<BlockConstraint1DData>& jd, uint32_t jdIndex,
                       std::vector<BlockConstraint1DVelocities>& rvs,
                       const std::vector<BlockConstraint1DParameters>& rps, uint32_t rowBase, Vec3* angS0,
                       Vec3* angS1, float iMass0, float iMass1, const Float4& invMassScale, const Mat33& i0,
                       const Mat33& i1, uint32_t lane, bool disablePreprocessing)
{
    const uint32_t numRows = jd[jdIndex].mNumRows[lane];
    for (uint32_t i = 0; i < numRows; i++)
    {
        const uint32_t hint = rps[rowBase + i].solveHint[lane];
        uint32_t j = i;
        for (; j > 0 && hint < rps[rowBase + sorted[j - 1]].solveHint[lane]; j--)
            sorted[j] = sorted[j - 1];
        sorted[j] = i;
    }

    const float invMass0 = iMass0 * invMassScale.x;
    const float invMass1 = iMass1 * invMassScale.z;
    const float invInertiaScale0 = invMassScale.y;
    const float invInertiaScale1 = invMassScale.w;

    for (uint32_t i = 0; i < numRows; ++i)
    {
        const uint32_t index = rowBase + sorted[i];
        const Float4 a0 = rvs[index].angular0XYZ_velocityTargetW[lane];
        const Float4 a1 = rvs[index].angular1XYZ_maxImpulseW[lane];
        angS0[i] = i0.transform(Vec3(a0.x, a0.y, a0.z));
        angS1[i] = i1.transform(Vec3(a1.x, a1.y, a1.z));
    }

    if (!disablePreprocessing)
    {
        for (uint32_t i = 0; i < numRows;)
        {
            const uint32_t groupMajorId = rps[rowBase + sorted[i]].solveHint[lane] >> 8;
            const uint32_t start = i++;
            while (i < numRows && (rps[rowBase + sorted[i]].solveHint[lane] >> 8) == groupMajorId)
                i++;
            if (groupMajorId == 4 || groupMajorId == 8)
            {
                uint32_t bCount = start;
                for (; bCount < i && (rps[rowBase + sorted[bCount]].solveHint[lane] & 255u) == 0; bCount++)
                    ;
                refOrthogonalize(sorted, start, rvs, rowBase, angS0, angS1, i - start, bCount - start, invMass0,
                                 invMass1, invInertiaScale0, invInertiaScale1, lane);
            }
        }
    }
}

// setupSolverConstraintBlockGPU (PGS) for one lane
void refJointPGS(const std::vector<BlockConstraint1DData>& jd, uint32_t jdIndex,
                 std::vector<BlockConstraint1DVelocities>& rvs,
                 const std::vector<BlockConstraint1DParameters>& rps, uint32_t rowBase, const SolverBodyData& b0,
                 const SolverBodyData& b1, const Mat33& i0, const Mat33& i1, float dt, float recipdt, uint32_t lane,
                 uint32_t writeBackIndex, BlockSolverConstraint1DHeader& hdr, std::vector<BlockSolverConstraint1DCon>& cons,
                 std::vector<BlockSolverConstraint1DMod>& mods, uint32_t outRowBase)
{
    const uint32_t numRows = jd[jdIndex].mNumRows[lane];

    const uint32_t word = lane >> 1, shift = (lane & 1) * 16;
    hdr.rowCounts[word] = (hdr.rowCounts[word] & ~(0xFFFFu << shift)) | ((numRows & 0xFFu) << shift);
    hdr.writeBackOffset[lane] = writeBackIndex;

    const Float4 ims = jd[jdIndex].mInvMassScale[lane];
    const Float4 il0 = b0.initialLinVelXYZ_invMassW, il1 = b1.initialLinVelXYZ_invMassW;
    const Float4 raW = jd[jdIndex].mRAWorld_linBreakForce[lane];
    const Float4 rbW = jd[jdIndex].mRBWorld_AngBreakForce[lane];
    const float linBreakImpulse = raW.w * dt;
    const float angBreakForce = rbW.w;
    const float angBreakImpulse = angBreakForce * dt;
    hdr.body0WorldOffset_linBreakImpulse[lane] = Float4{ raW.x, raW.y, raW.z, linBreakImpulse };
    hdr.angBreakImpulse[lane] = angBreakImpulse;
    hdr.invMass0D0[lane] = il0.w * ims.x;
    hdr.invMass1D1[lane] = il1.w * ims.z;
    hdr.invInertiaScale0[lane] = ims.y;
    hdr.invInertiaScale1[lane] = ims.w;
    const uint32_t breakable = ((raW.w != PX_MAX_F32) || (angBreakForce != PX_MAX_F32)) ? 1u : 0u;
    hdr.breakable[word] = (hdr.breakable[word] & ~(0xFFFFu << shift)) | ((breakable & 0xFFu) << shift);

    const bool disablePreprocessing = (jd[jdIndex].mFlags[lane] & CFLAG_DISABLE_PREPROCESSING) != 0;
    uint32_t sorted[MAX_CONSTRAINT_ROWS];
    Vec3 angS0[MAX_CONSTRAINT_ROWS], angS1[MAX_CONSTRAINT_ROWS];
    refPreprocessRows(sorted, jd, jdIndex, rvs, rps, rowBase, angS0, angS1, il0.w, il1.w, ims, i0, i1, lane,
                      disablePreprocessing);

    const float erp = 1.0f;
    for (uint32_t r = 0; r < numRows; r++)
    {
        const uint32_t index = rowBase + sorted[r];
        const uint32_t outRow = outRowBase + r;
        const Float4 cl0 = rvs[index].linear0XYZ_geometricErrorW[lane];
        const Float4 cl1 = rvs[index].linear1XYZ_minImpulseW[lane];
        const Float4 ca0 = rvs[index].angular0XYZ_velocityTargetW[lane];
        const Float4 ca1 = rvs[index].angular1XYZ_maxImpulseW[lane];
        const Vec3 clin0(cl0.x, cl0.y, cl0.z), clin1(cl1.x, cl1.y, cl1.z);
        const Vec3 cang0(ca0.x, ca0.y, ca0.z), cang1(ca1.x, ca1.y, ca1.z);
        const Vec3 ang0 = angS0[r], ang1 = angS1[r];
        const uint32_t rpFlags = rps[index].flags[lane];

        float minImpulse, maxImpulse;
        const bool hasDriveLimit = (rpFlags & PX1D_HAS_DRIVE_LIMIT) != 0;
        const bool driveLimitsAreForces = (jd[jdIndex].mFlags[lane] & CFLAG_DRIVE_LIMITS_ARE_FORCES) != 0;
        computeMinMaxImpulseOrForceAsImpulse(cl1.w, ca1.w, hasDriveLimit, driveLimitsAreForces, dt, minImpulse, maxImpulse);

        mods[outRow].ang0Writeback[lane * 3 + 0] = cang0.x;
        mods[outRow].ang0Writeback[lane * 3 + 1] = cang0.y;
        mods[outRow].ang0Writeback[lane * 3 + 2] = cang0.z;

        const float resp0 = clin0.dot(clin0) * il0.w * ims.x + ang0.dot(ang0) * ims.y;
        const float resp1 = clin1.dot(clin1) * il1.w * ims.z + ang1.dot(ang1) * ims.w;

        const float projV0 = b0.initialLinVelXYZ_invMassW.x * clin0.x + b0.initialLinVelXYZ_invMassW.y * clin0.y +
                             b0.initialLinVelXYZ_invMassW.z * clin0.z + b0.initialAngVelXYZ_penBiasClamp.x * cang0.x +
                             b0.initialAngVelXYZ_penBiasClamp.y * cang0.y + b0.initialAngVelXYZ_penBiasClamp.z * cang0.z;
        const float projV1 = b1.initialLinVelXYZ_invMassW.x * clin1.x + b1.initialLinVelXYZ_invMassW.y * clin1.y +
                             b1.initialLinVelXYZ_invMassW.z * clin1.z + b1.initialAngVelXYZ_penBiasClamp.x * cang1.x +
                             b1.initialAngVelXYZ_penBiasClamp.y * cang1.y + b1.initialAngVelXYZ_penBiasClamp.z * cang1.z;
        const float initJointSpeed = projV0 - projV1;

        const float springStiffness = rps[index].springStiffness_restitution[lane];
        const float springDamping = rps[index].springDamping_velocityThreshold[lane];
        const float restitution = rps[index].springStiffness_restitution[lane];
        const float bounceThreshold = rps[index].springDamping_velocityThreshold[lane];
        const float geomError = cl0.w;
        const float velocityTarget = ca0.w;
        float coeff0, coeff1;
        queryReduced1dPGS(rpFlags, springStiffness, springDamping, restitution, bounceThreshold, geomError,
                          velocityTarget, initJointSpeed, erp, dt, recipdt, coeff0, coeff1);

        cons[outRow].lin0XYZ_minImpulse[lane] = Float4{ clin0.x, clin0.y, clin0.z, minImpulse };
        cons[outRow].lin1XYZ_maxImpulse[lane] = Float4{ clin1.x, clin1.y, clin1.z, maxImpulse };
        cons[outRow].ang0XYZ_resp0[lane] = Float4{ ang0.x, ang0.y, ang0.z, resp0 };
        cons[outRow].ang1XYZ_resp1[lane] = Float4{ ang1.x, ang1.y, ang1.z, resp1 };
        cons[outRow].initJointSpeed[lane] = initJointSpeed;
        mods[outRow].coeff0[lane] = coeff0;
        mods[outRow].coeff1[lane] = coeff1;
        mods[outRow].appliedForce[lane] = 0.f;
        mods[outRow].residual[lane] = 0.f;
        uint32_t modFlags = 0;
        if (rpFlags & PX1D_SPRING) modFlags |= DY_SC_SPRING;
        if (rpFlags & PX1D_ACCELERATION_SPRING) modFlags |= DY_SC_ACCELERATION_SPRING;
        if (rpFlags & PX1D_OUTPUT_FORCE) modFlags |= DY_SC_OUTPUT_FORCE;
        mods[outRow].flags[lane] = modFlags;
    }
}

// P9.86c coverage (constraintPrepTGS_joint): live rows with a kinematic body 1 only and
// vel1 = -0 (CUDA computeJointSpeedTGS, DyCpuGpu1dConstraint.h:391-395: 0 + (-0) = +0),
// and those on which the folded-seed twin (initJointSpeed = vel1) changes targetVel
uint32_t gTgsJsKin1NegZero = 0, gTgsJsKin1TwinDiff = 0;

// setupSolverConstraintBlockGPUTGS for one lane
void refJointTGS(const std::vector<BlockConstraint1DData>& jd, uint32_t jdIndex,
                 std::vector<BlockConstraint1DVelocities>& rvs, std::vector<BlockConstraint1DParameters>& rps,
                 uint32_t rowBase, const SolverBodyData& b0, const SolverBodyData& b1, const Mat33& i0,
                 const Mat33& i1, float stepDt, float recipStepDt, float simDt, float recipSimDt, float lengthScale,
                 float biasCoefficient, uint32_t lane, uint32_t writeBackIndex, TGSJointHeader& hdr,
                 std::vector<TGSJointRowCon>& cons, uint32_t outRowBase)
{
    const uint32_t numRows = jd[jdIndex].mNumRows[lane];
    uint32_t rc_x = numRows & 0xFFu;
    hdr.writeBackOffset[lane] = writeBackIndex;

    const Float4 ims = jd[jdIndex].mInvMassScale[lane];
    const Float4 il0 = b0.initialLinVelXYZ_invMassW, il1 = b1.initialLinVelXYZ_invMassW;
    const Float4 raW = jd[jdIndex].mRAWorld_linBreakForce[lane];
    const Float4 rbW = jd[jdIndex].mRBWorld_AngBreakForce[lane];
    hdr.linBreakImpulse[lane] = raW.w * simDt;
    hdr.angBreakImpulse[lane] = rbW.w * simDt;
    const float invMass0 = il0.w * ims.x, invMass1 = il1.w * ims.z;
    hdr.rAWorld_invMass0D0[lane] = Float4{ raW.x, raW.y, raW.z, invMass0 };
    hdr.rBWorld_invMass1D1[lane] = Float4{ rbW.x, rbW.y, rbW.z, invMass1 };
    hdr.invInertiaScale0[lane] = ims.y;
    hdr.invInertiaScale1[lane] = ims.w;
    uint32_t rc_y = ((raW.w != PX_MAX_F32) || (rbW.w != PX_MAX_F32)) ? 1u : 0u;

    // promote angular constraint solve hints
    for (uint32_t r = 0; r < numRows; ++r)
    {
        if ((rps[rowBase + r].flags[lane] & PX1D_ANGULAR_CONSTRAINT) != 0)
        {
            uint32_t hint = rps[rowBase + r].solveHint[lane];
            if (hint == HINT_EQ) hint = HINT_ROT_EQ;
            else if (hint == HINT_INEQ) hint = HINT_ROT_INEQ;
            rps[rowBase + r].solveHint[lane] = hint;
        }
    }

    const bool disablePreprocessing = (jd[jdIndex].mFlags[lane] & CFLAG_DISABLE_PREPROCESSING) != 0;
    uint32_t sorted[MAX_CONSTRAINT_ROWS];
    Vec3 angS0[MAX_CONSTRAINT_ROWS], angS1[MAX_CONSTRAINT_ROWS];
    refPreprocessRows(sorted, jd, jdIndex, rvs, rps, rowBase, angS0, angS1, il0.w, il1.w, ims, i0, i1, lane,
                      disablePreprocessing);

    const bool isKin0 = (b0.flags & RB_KINEMATIC) != 0;
    const bool isKin1 = (b1.flags & RB_KINEMATIC) != 0;
    const float erp = 0.5f * biasCoefficient;

    uint32_t eqCount = 0;
    for (uint32_t r = 0; r < numRows; r++)
    {
        const uint32_t index = rowBase + sorted[r];
        const uint32_t outRow = outRowBase + r;
        const Float4 cl0 = rvs[index].linear0XYZ_geometricErrorW[lane];
        const Float4 cl1 = rvs[index].linear1XYZ_minImpulseW[lane];
        const Float4 ca0 = rvs[index].angular0XYZ_velocityTargetW[lane];
        const Float4 ca1 = rvs[index].angular1XYZ_maxImpulseW[lane];
        const Vec3 clin0(cl0.x, cl0.y, cl0.z), clin1(cl1.x, cl1.y, cl1.z);
        const Vec3 cang0(ca0.x, ca0.y, ca0.z), cang1(ca1.x, ca1.y, ca1.z);
        const Vec3 as0 = angS0[r], as1 = angS1[r];

        const float resp0 = clin0.dot(clin0) * il0.w * ims.x + as0.dot(as0) * ims.y;
        const float resp1 = clin1.dot(clin1) * il1.w * ims.z + as1.dot(as1) * ims.w;
        const float unitResponse = resp0 + resp1;

        const float vel0 = b0.initialLinVelXYZ_invMassW.x * clin0.x + b0.initialLinVelXYZ_invMassW.y * clin0.y +
                          b0.initialLinVelXYZ_invMassW.z * clin0.z + b0.initialAngVelXYZ_penBiasClamp.x * cang0.x +
                          b0.initialAngVelXYZ_penBiasClamp.y * cang0.y + b0.initialAngVelXYZ_penBiasClamp.z * cang0.z;
        const float vel1 = b1.initialLinVelXYZ_invMassW.x * clin1.x + b1.initialLinVelXYZ_invMassW.y * clin1.y +
                          b1.initialLinVelXYZ_invMassW.z * clin1.z + b1.initialAngVelXYZ_penBiasClamp.x * cang1.x +
                          b1.initialAngVelXYZ_penBiasClamp.y * cang1.y + b1.initialAngVelXYZ_penBiasClamp.z * cang1.z;
        float jointSpeed, initJointSpeed;
        computeJointSpeedTGS(vel0, isKin0, vel1, isKin1, jointSpeed, initJointSpeed);

        const float minRowResponse = DY_MIN_RESPONSE;

        // intializeBlock1DTGS
        const float geometricError = cl0.w;
        const uint32_t flags = rps[index].flags[lane];
        const float stiffness = rps[index].springStiffness_restitution[lane];
        const float damping = rps[index].springDamping_velocityThreshold[lane];
        const float restitution = rps[index].springStiffness_restitution[lane];
        const float bounceVelocityThreshold = rps[index].springDamping_velocityThreshold[lane];
        const float velocityTarget = ca0.w;
        const float maxBiasVelocity = computeMaxBiasVelocityTGS(flags, jointSpeed, bounceVelocityThreshold,
                                                               restitution, geometricError, false, lengthScale, recipSimDt);
        const float recipUnitResponse = computeRecipUnitResponse(unitResponse, minRowResponse);
        float biasScale, error, velMultiplier, targetVel;
        compute1dConstraintSolverConstantsTGS(flags, stiffness, damping, restitution, bounceVelocityThreshold,
                                              geometricError, velocityTarget, jointSpeed, initJointSpeed, unitResponse,
                                              recipUnitResponse, erp, stepDt, recipStepDt, biasScale, error, velMultiplier, targetVel);
        {
            auto bitsOf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
            if (!isKin0 && isKin1 && bitsOf(vel1) == 0x80000000u)
            {
                ++gTgsJsKin1NegZero;
                float tb, te, tvm, ttv;   // twin: the +0 seed folded, 0 + v1 -> v1
                compute1dConstraintSolverConstantsTGS(flags, stiffness, damping, restitution, bounceVelocityThreshold,
                                                      geometricError, velocityTarget, jointSpeed, vel1, unitResponse,
                                                      recipUnitResponse, erp, stepDt, recipStepDt, tb, te, tvm, ttv);
                if (bitsOf(ttv) != bitsOf(targetVel))
                    ++gTgsJsKin1TwinDiff;
            }
        }

        Float4 lin1XYZ_biasScale = cl1; lin1XYZ_biasScale.w = biasScale;
        Float4 lin0XYZ_initBiasW = cl0; lin0XYZ_initBiasW.w = error;
        Float4 ang0XYZ_velMultW = ca0; ang0XYZ_velMultW.w = velMultiplier;
        Float4 ang1XYZ_velTargetW = ca1; ang1XYZ_velTargetW.w = targetVel;

        float angularErrorScale = 1.f;
        if ((flags & PX1D_ANGULAR_CONSTRAINT) == 0)
        {
            ang0XYZ_velMultW.x = ang0XYZ_velMultW.y = ang0XYZ_velMultW.z = 0.f;
            ang1XYZ_velTargetW.x = ang1XYZ_velTargetW.y = ang1XYZ_velTargetW.z = 0.f;
            angularErrorScale = 0.f;
        }

        cons[outRow].lin0XYZ_initBiasOrCoeff0[lane] = lin0XYZ_initBiasW;
        cons[outRow].lin1XYZ_biasScaleOrCoeff1[lane] = lin1XYZ_biasScale;
        cons[outRow].ang0XYZ_velMultiplierOrCoeff2[lane] = ang0XYZ_velMultW;
        cons[outRow].ang1XYZ_velTargetOrCoeff3[lane] = ang1XYZ_velTargetW;
        cons[outRow].maxBias[lane] = maxBiasVelocity;
        cons[outRow].angularErrorScale[lane] = angularErrorScale;
        cons[outRow].appliedForce[lane] = 0.f;
        cons[outRow].residual[lane] = 0.f;

        const bool hasDriveLimit = (flags & PX1D_HAS_DRIVE_LIMIT) != 0;
        const bool driveLimitsAreForces = (jd[jdIndex].mFlags[lane] & CFLAG_DRIVE_LIMITS_ARE_FORCES) != 0;
        float minImp, maxImp;
        computeMinMaxImpulseOrForceAsImpulse(cl1.w, ca1.w, hasDriveLimit, driveLimitsAreForces, simDt, minImp, maxImp);
        cons[outRow].minImpulse[lane] = minImp;
        cons[outRow].maxImpulse[lane] = maxImp;

        uint32_t outFlags = 0;
        const uint32_t solveHint = rps[index].solveHint[lane];
        raiseInternalFlagsTGS(flags, solveHint, outFlags);
        if (!disablePreprocessing)
        {
            if (solveHint == HINT_ROT_EQ)
            {
                outFlags |= DY_SC_ROT_EQ;
                hdr.angOrthoAxis0_recipResponseW[eqCount][lane] =
                    Float4{ as0.x * ims.y, as0.y * ims.y, as0.z * ims.y, recipUnitResponse };
                hdr.angOrthoAxis1_ErrorW[eqCount][lane] =
                    Float4{ as1.x * ims.w, as1.y * ims.w, as1.z * ims.w, geometricError };
                eqCount++;
            }
            else if ((solveHint & HINT_EQ) != 0)
            {
                outFlags |= DY_SC_ORTHO_TARGET;
            }
        }
        cons[outRow].flags[lane] = outFlags;
    }

    for (uint32_t s = eqCount; s < 3; ++s)
    {
        hdr.angOrthoAxis0_recipResponseW[s][lane] = Float4{ 0, 0, 0, 0 };
        hdr.angOrthoAxis1_ErrorW[s][lane] = Float4{ 0, 0, 0, 0 };
    }

    hdr.rowCounts_breakable_orthoAxisCount[lane] = (rc_x & 0xFFu) | ((rc_y & 0xFFu) << 8) | ((eqCount & 0xFFu) << 16);
}
} // namespace

#include "testConstraintPrepTests.inc"
