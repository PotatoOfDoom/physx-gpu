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


// P8.97a: npArtiSetupInternal.slang -- setupInternalConstraints, the per-dof
// drive/limit/friction setup.
//
// THE ORACLE: the mirror is control flow over engine primitives -- PxQuat
// algebra, Cm::UnAlignedSpatialVector::innerProduct (the DUAL product),
// PxSpatialMatrix::operator* (PxSpatialMatrix.h:58), Dy::FeatherstoneArticulation
// ::translateSpatialVector, the block load/store helpers (PxgArticulation.h),
// and Dy::computeImplicitDriveParams + setImplicitDriveDesc
// (DyCpuGpuArticulation.h) for the WHOLE drive-desc path.  Only the propagate
// bodies are transcribed (they are __device__), and those sit on P8.90's
// swept module.
//
// TOLERANCE: the spherical drive-error path runs sin/cos (the angle-axis
// quat), and llvmpipe transcendentals are ~2 ULP off glibc (P8.94a).  Every
// output word whose value flows through that path -- the implicit-drive desc
// float4 + posBias of dofs whose error came from a SPHERICAL driveError write
// (including a later stale read of it) -- is compared with a ULP bound; every
// other word bit-exactly against the poison-seeded image.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//   * 68 articulations over 3 warps (warp 2 partial), numLinks 1..4 (1 = the
//     loop never runs), maxLinks 4 != maxDofs 10, every count distinct;
//   * a 9-case joint table: revolute about each axis, revolute_unwrapped,
//     prismatic (linear path + dofs[0] error), spherical 3-dof with mixed
//     drives, spherical 2-dof driveless (stale driveError stays dead),
//     fix joint (nbDofs = 0, cursor advance only), a zero-drive revolute
//     (maxForce = 0 -> the zero-desc branch), a LIMITED dof, and the
//     locked+driven revolute AFTER a spherical write (trap 1's persistence,
//     deterministic in both implementations);
//   * eDRIVE_LIMITS_ARE_FORCES on odd articulations (maxForceScale = dt);
//   * negative targetQ.dot(qB2qA) via a target opposite the joint rotation;
//   * BOTH solver modes: one TGS and one PGS dispatch over identically
//     seeded buffers (impMul/posBias and the drive-dt select differ).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "DyCpuGpuArticulation.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "PxArticulationReducedCoordinate.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;
using Cm::UnAlignedSpatialVector;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> siLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SI_NB = 68, SI_WARPS = 3, SI_MAX_LINKS = 4, SI_MAX_DOFS = 10;
const float SI_STEP_DT = 0.004f, SI_DT = 0.0333f, SI_INV_DT = 250.0f;

// deterministic value soup
static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 17u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}
static PxQuat qv(uint32_t a, uint32_t l, uint32_t salt)
{
    PxQuat q(fv(a, l, 0, salt), fv(a, l, 1, salt), fv(a, l, 2, salt),
             1.0f + 0.4f * fv(a, l, 3, salt));
    return q.getNormalized();
}

// the 9-case joint table; case 8 (locked+driven) only ever follows a case-3
// spherical write in the same articulation
struct JointCfg
{
    uint8_t jointType;   // PxArticulationJointType
    uint8_t nbDofs;
    uint8_t dofIds[3];
    uint8_t motion[3];   // PxArticulationMotion
    uint8_t driveType[3];// PxArticulationDriveType (2 = eNONE)
    bool    zeroDrive;   // maxForce=maxEffort=0 -> zero-desc branch
};

static JointCfg jointCase(uint32_t a, uint32_t linkID)
{
    const uint32_t numLinks = 1u + (a % 4u);
    uint32_t c;
    if ((a % 4u) == 3u)
    {
        // fixed script so case 8's stale read is deterministically preceded
        // by case 3's full spherical driveError write
        c = (linkID == 1u) ? 3u : (linkID == 2u ? (a % 2u ? 2u : 6u) : 8u);
    }
    else
    {
        // a hash, NOT (a*k + linkID*m) % 8: the articulation index arrives
        // with stride 4 (numLinks = 1 + a%4 gates which linkIDs exist), and a
        // linear form modulo 8 collapses onto a 2-value orbit -- cases 2,3,5,
        // 6,7 never occurred and the guard-or mutant survived on the gap.
        // The coverage histogram below now pins all nine cases as present.
        uint32_t h = a * 2654435761u + linkID * 40503u + 11u;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        c = h % 8u;    // 0..7, case 8 handled above
    }
    PX_UNUSED(numLinks);

    JointCfg j = {};
    switch (c)
    {
    default:
    case 0: // revolute about twist, force drive, free
        j = { PxArticulationJointType::eREVOLUTE, 1, {0}, {2}, {0}, false }; break;
    case 1: // revolute_unwrapped about swing2, acceleration drive, free
        j = { PxArticulationJointType::eREVOLUTE_UNWRAPPED, 1, {2}, {2}, {1}, false }; break;
    case 2: // prismatic along X, force drive, LIMITED
        j = { PxArticulationJointType::ePRISMATIC, 1, {3}, {1}, {0}, false }; break;
    case 3: // spherical 3-dof: drives on 0 and 2, dof 1 locked+driveless
        j.jointType = PxArticulationJointType::eSPHERICAL; j.nbDofs = 3;
        j.dofIds[0] = 0; j.dofIds[1] = 1; j.dofIds[2] = 2;
        j.motion[0] = 2; j.motion[1] = 0; j.motion[2] = 1;   // free, locked, limited
        j.driveType[0] = 0; j.driveType[1] = 2; j.driveType[2] = 1;
        j.zeroDrive = false; break;
    case 4: // spherical 2-dof, NO drives (hasAngularDrives false)
        j.jointType = PxArticulationJointType::eSPHERICAL; j.nbDofs = 2;
        j.dofIds[0] = 1; j.dofIds[1] = 2;
        j.motion[0] = 2; j.motion[1] = 2;
        j.driveType[0] = 2; j.driveType[1] = 2;
        j.zeroDrive = true; break;
    case 5: // fix joint: no dofs at all (the cursor still advances by 0)
        j.jointType = PxArticulationJointType::eFIX; j.nbDofs = 0; break;
    case 6: // revolute about swing1, damping-only drive, negative-dot target
        j = { PxArticulationJointType::eREVOLUTE, 1, {1}, {2}, {0}, false }; break;
    case 7: // revolute, zero drive (maxForce=0, stiffness=0) -> zero desc
        j = { PxArticulationJointType::eREVOLUTE, 1, {0}, {2}, {0}, true }; break;
    case 8: // locked+driven revolute -- reads the PERSISTED driveError
        j = { PxArticulationJointType::eREVOLUTE, 1, {1}, {0}, {0}, false }; break;
    }
    return j;
}

// ---- transcriptions of the two __device__ propagate helpers (they sit on
// P8.90's swept module on the GPU side; here they are control flow over
// engine primitives, per the P8.73-P8.80 arrangement) ----

static UnAlignedSpatialVector refPropImpDofAligned(const UnAlignedSpatialVector& isInvD,
                                                   const PxVec3& c2p,
                                                   const UnAlignedSpatialVector& sa,
                                                   const UnAlignedSpatialVector& Z, PxReal& qstZ)
{
    const PxReal stZ = sa.innerProduct(Z);
    const UnAlignedSpatialVector temp = isInvD * stZ;
    qstZ = -stZ;
    return Dy::FeatherstoneArticulation::translateSpatialVector(c2p, Z - temp);
}

static UnAlignedSpatialVector refPropAccel(const PxVec3& c2p,
                                           const PxgArticulationBlockDofData* dofs,
                                           const UnAlignedSpatialVector& parentDeltaV,
                                           PxU32 dofCount, PxU32 lane, const PxReal* qstZ)
{
    const UnAlignedSpatialVector pDeltaV =
        Dy::FeatherstoneArticulation::translateSpatialVector(-c2p, parentDeltaV);
    PxReal t[3] = { 0.f, 0.f, 0.f };
    for (PxU32 ind = 0; ind < 3; ++ind)
        if (ind < dofCount)
            t[ind] = qstZ[ind] - loadSpatialVector(dofs[ind].mIsW, lane).innerProduct(pDeltaV);
    // propagateAccelerationW_child2ParentPreLoaded (articulationDynamic.cuh:257-313):
    // the joint terms accumulate into their own +0-seeded vector, added last
    UnAlignedSpatialVector jointSpatialDeltaV(PxVec3(0.f), PxVec3(0.f));
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxReal jDelta = dofs[ind].mInvStIsT_x[lane] * t[0]
                                + dofs[ind].mInvStIsT_y[lane] * t[1]
                                + dofs[ind].mInvStIsT_z[lane] * t[2];
            jointSpatialDeltaV += loadSpatialVector(dofs[ind].mWorldMotionMatrix, lane) * jDelta;
        }
    }
    return pDeltaV + jointSpatialDeltaV;
}

// the mirror of setupInternalConstraints itself; marks the words that flowed
// through the spherical (transcendental) drive-error path in ulpMask
static void refSetup(PxgArticulationBlockData& ab, PxgArticulationBlockLinkData* lk,
                     PxgArticulationBlockDofData* df, float stepDt, float dt, bool isTGS,
                     uint32_t lane, uint8_t* sphericalErr /*slice base, [block*32+lane]*/)
{
    const PxU32 numLinks = ab.mNumLinks[lane];
    const PxReal maxForceScale =
        (ab.mFlags[lane] & PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES) ? dt : 1.f;

    PxReal driveError[3] = { 0.f, 0.f, 0.f };
    bool driveErrIsSpherical[3] = { false, false, false };

    PxgArticulationBlockDofData* dofs = df;

    for (PxU32 linkID = 1; linkID < numLinks; ++linkID)
    {
        PxgArticulationBlockLinkData& link = lk[linkID];
        const PxU32 nbDofs = link.mDofs[lane];
        const PxU32 parent = link.mParents[lane];

        PxSpatialMatrix parentResponse;
        const PxVec3 c2p(link.mRw_x[lane], link.mRw_y[lane], link.mRw_z[lane]);
        PxReal transmissionForce = 0.f;
        PxReal cfm = 0.f;
        bool loaded = false;

        if (nbDofs)
        {
            const PxReal frictionCoefficient =
                dofs->mConstraintData.mFrictionCoefficient[lane] * stepDt;
            const PxU32 jointType = link.mJointType[lane];
            const bool isAngular = jointType == PxArticulationJointType::eREVOLUTE
                                || jointType == PxArticulationJointType::eREVOLUTE_UNWRAPPED
                                || jointType == PxArticulationJointType::eSPHERICAL;

            if (isAngular)
            {
                if (nbDofs > 1)
                {
                    const float4 cR = link.mAccumulatedPose.q[lane];
                    const float4 pR = lk[parent].mAccumulatedPose.q[lane];
                    const float4 paQ = link.mParentPose.q[lane];
                    const float4 chQ = link.mChildPose.q[lane];
                    const PxQuat cA2w = PxQuat(pR.x, pR.y, pR.z, pR.w) * PxQuat(paQ.x, paQ.y, paQ.z, paQ.w);
                    const PxQuat cB2w = PxQuat(cR.x, cR.y, cR.z, cR.w) * PxQuat(chQ.x, chQ.y, chQ.z, chQ.w);
                    PxQuat qB2qA = cA2w.getConjugate() * cB2w;

                    PxVec3 driveAxis(0.f);
                    bool hasAngularDrives = false;
                    for (PxU32 i = 0; i < nbDofs; ++i)
                    {
                        const bool hasDrive =
                            dofs[i].mMotion[lane] != PxArticulationMotion::eLOCKED
                            && dofs[i].mConstraintData.mDriveType[lane] != PxArticulationDriveType::eNONE;
                        if (hasDrive)
                        {
                            const float4 top = dofs[i].mLocalMotionMatrix.mTopxyz_bx[lane];
                            driveAxis += PxVec3(top.x, top.y, top.z)
                                       * dofs[i].mConstraintData.mDriveTargetPos[lane];
                            hasAngularDrives = true;
                        }
                    }
                    if (hasAngularDrives)
                    {
                        PxReal angle = driveAxis.normalize();
                        if (angle < 1e-12f)
                        {
                            driveAxis = PxVec3(1.f, 0.f, 0.f);
                            angle = 0.f;
                        }
                        PxQuat targetQ = PxQuat(angle, driveAxis);
                        if (targetQ.dot(qB2qA) < 0.f)
                            targetQ = -targetQ;
                        const PxVec3 e = -2.f * (targetQ.getConjugate() * qB2qA).getImaginaryPart();
                        driveError[0] = e.x; driveError[1] = e.y; driveError[2] = e.z;
                        driveErrIsSpherical[0] = driveErrIsSpherical[1] = driveErrIsSpherical[2] = true;
                    }
                }
                else
                {
                    if (dofs[0].mMotion[lane] != PxArticulationMotion::eLOCKED)
                    {
                        const PxU32 i = dofs[0].mDofIds[lane];
                        driveError[i] = dofs[0].mConstraintData.mDriveTargetPos[lane]
                                      - dofs[0].mJointPositions[lane];
                        driveErrIsSpherical[i] = false;
                    }
                }
            }

            for (PxU32 i = 0; i < nbDofs; ++i)
            {
                const PxU32 motion = dofs[i].mMotion[lane];
                const PxU32 dofId = dofs[i].mDofIds[lane];
                const PxReal maxForce = dofs[i].mConstraintData.mMaxForce[lane];
                const PxReal maxEffort = dofs[i].mConstraintData.mMaxEffort[lane];
                const PxReal stiffness = dofs[i].mConstraintData.mDriveStiffness[lane];
                const PxReal damping = dofs[i].mConstraintData.mDamping[lane];

                if (!loaded)
                {
                    loadSpatialMatrix(lk[parent].mSpatialResponseMatrix, lane, parentResponse);
                    const UnAlignedSpatialVector bias = loadSpatialVector(link.mBiasForce, lane);
                    transmissionForce = bias.magnitude() * frictionCoefficient;
                    cfm = PxMax(lk[parent].mCfm[lane], link.mCfm[lane]);
                    loaded = true;
                }

                const UnAlignedSpatialVector wmv =
                    loadSpatialVector(dofs[i].mWorldMotionMatrix, lane);
                UnAlignedSpatialVector axis0, axis1;
                const bool isLinear = dofId >= PxArticulationAxis::eX;
                const PxReal position = dofs[i].mJointPositions[lane];
                PxReal error;
                bool errSpherical = false;
                if (!isLinear)
                {
                    axis0 = UnAlignedSpatialVector(PxVec3(0.f), wmv.top);
                    axis1 = UnAlignedSpatialVector(PxVec3(0.f), -wmv.top);
                    error = driveError[dofId];
                    errSpherical = driveErrIsSpherical[dofId];
                }
                else
                {
                    const float4 cR = link.mAccumulatedPose.q[lane];
                    const float4 pR = lk[parent].mAccumulatedPose.q[lane];
                    const float4 paP = link.mParentPose.p[lane];
                    const float4 chP = link.mChildPose.p[lane];
                    const PxVec3 cA2w = PxQuat(pR.x, pR.y, pR.z, pR.w).rotate(PxVec3(paP.x, paP.y, paP.z));
                    const PxVec3 cB2w = PxQuat(cR.x, cR.y, cR.z, cR.w).rotate(PxVec3(chP.x, chP.y, chP.z));
                    const PxVec3 axis = wmv.bottom;
                    axis0 = UnAlignedSpatialVector(axis, cA2w.cross(axis));
                    axis1 = UnAlignedSpatialVector(-axis, -cB2w.cross(axis));
                    error = dofs[0].mConstraintData.mDriveTargetPos[lane] - position;
                }

                // getImpulseSelfResponseDofAligned
                UnAlignedSpatialVector deltaVA, deltaVB;
                {
                    const UnAlignedSpatialVector Z1W(-axis1.top, -axis1.bottom);
                    const UnAlignedSpatialVector isInvD =
                        loadSpatialVector(dofs[i].mIsInvDW, lane);
                    PxReal qstZ[3] = { 0.f, 0.f, 0.f };
                    const UnAlignedSpatialVector Z0W =
                        refPropImpDofAligned(isInvD, c2p, wmv, Z1W, qstZ[i]);
                    const UnAlignedSpatialVector dif = axis0 - Z0W;
                    const UnAlignedSpatialVector dV0 = parentResponse * dif;
                    deltaVB = refPropAccel(c2p, dofs, dV0, nbDofs, lane, qstZ);
                    deltaVA = dV0;
                }

                const PxReal r0 = deltaVA.innerProduct(axis0);
                const PxReal r1 = deltaVB.innerProduct(axis1);
                const PxReal unitResponse = r1 + r0;
                const PxReal recipResponse = 1.0f / (unitResponse + cfm);

                dofs[i].mConstraintData.mMaxFrictionForce[lane] = transmissionForce;

                if (motion == PxArticulationMotion::eLIMITED)
                {
                    const float2 limit = dofs[i].mConstraintData.mLimits_LowLimitX_highLimitY[lane];
                    dofs[i].mConstraintData.mLimitError_LowX_highY[lane] =
                        make_float2(position - limit.x, limit.y - position);
                }

                if ((maxForce > 0.0f || maxEffort > 0.0f) && (stiffness > 0.0f || damping > 0.0f))
                {
                    const PxReal targetVelocity = dofs[i].mConstraintData.mDriveTargetVel[lane];
                    const PxArticulationDriveType::Enum type =
                        PxArticulationDriveType::Enum(dofs[i].mConstraintData.mDriveType[lane]);
                    dofs[i].mConstraintData.setImplicitDriveDesc(lane,
                        Dy::computeImplicitDriveParams(type, stiffness, damping,
                                                       isTGS ? stepDt : dt, dt,
                                                       unitResponse, recipResponse,
                                                       error, targetVelocity, isTGS));
                    dofs[i].mConstraintData.mConstraintMaxImpulse[lane] = maxForce * maxForceScale;
                    if (errSpherical && sphericalErr)
                        sphericalErr[size_t(dofs + i - df) * 32 + lane] = 1;
                }
                else
                {
                    dofs[i].mConstraintData.setImplicitDriveDesc(lane,
                        Dy::ArticulationImplicitDriveDesc(PxZero));
                    dofs[i].mConstraintData.mConstraintMaxImpulse[lane] = 0.f;
                }
                dofs[i].mConstraintData.mDriveImpulse[lane] = 0.f;
                storeSpatialVector(dofs[i].mConstraintData.mDeltaVA, deltaVA, lane);
                storeSpatialVector(dofs[i].mConstraintData.mDeltaVB, deltaVB, lane);
                storeSpatialVector(dofs[i].mConstraintData.mRow0, axis0, lane);
                storeSpatialVector(dofs[i].mConstraintData.mRow1, -axis1, lane);
                dofs[i].mConstraintData.mRecipResponse[lane] = recipResponse;
                dofs[i].mConstraintData.mResponse[lane] = unitResponse;
                dofs[i].mConstraintData.mAccumulatedFrictionImpulse[lane] = 0.0f;
                dofs[i].mConstraintData.mLowImpulse[lane] = 0.f;
                dofs[i].mConstraintData.mHighImpulse[lane] = 0.f;
            }
        }
        dofs += nbDofs;
    }
}

static int32_t ulpDiff(float a, float b)
{
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4); std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = int32_t(0x80000000u) - ia;
    if (ib < 0) ib = int32_t(0x80000000u) - ib;
    const int64_t d = int64_t(ia) - int64_t(ib);
    return int32_t(d < 0 ? -d : d);
}

static size_t siRun(PxgCudaShim& shim, bool isTGS)
{
    // behaviour coverage: all nine joint cases must actually occur
    {
        uint32_t hist[9] = { 0 };
        for (uint32_t a = 0; a < SI_NB; ++a)
            for (uint32_t l = 1; l < 1u + (a % 4u); ++l)
            {
                const JointCfg j = jointCase(a, l);
                // recover the case id from the config shape
                uint32_t c = 0;
                if (j.jointType == PxArticulationJointType::eFIX) c = 5;
                else if (j.jointType == PxArticulationJointType::eSPHERICAL)
                    c = j.nbDofs == 2 ? 4 : 3;
                else if (j.jointType == PxArticulationJointType::ePRISMATIC) c = 2;
                else if (j.jointType == PxArticulationJointType::eREVOLUTE_UNWRAPPED) c = 1;
                else if (j.motion[0] == 0) c = 8;
                else if (j.zeroDrive) c = 7;
                else c = j.dofIds[0] == 1 ? 6 : 0;
                ++hist[c];
            }
        for (uint32_t c = 0; c < 9; ++c)
            PXG_CHECK_MSG(hist[c] > 0, "joint-case coverage hole");
    }
    // ---- seed the block images ----
    std::vector<PxgArticulationBlockData> ab(SI_WARPS);
    std::vector<PxgArticulationBlockLinkData> lk(size_t(SI_WARPS) * SI_MAX_LINKS);
    std::vector<PxgArticulationBlockDofData> df(size_t(SI_WARPS) * SI_MAX_DOFS);
    std::memset(ab.data(), 0xAB, ab.size() * sizeof(ab[0]));
    std::memset(lk.data(), 0xAB, lk.size() * sizeof(lk[0]));
    std::memset(df.data(), 0xAB, df.size() * sizeof(df[0]));

    for (uint32_t a = 0; a < SI_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& B = ab[w];
        PxgArticulationBlockLinkData* L = &lk[size_t(w) * SI_MAX_LINKS];
        PxgArticulationBlockDofData* D = &df[size_t(w) * SI_MAX_DOFS];

        const uint32_t numLinks = 1u + (a % 4u);
        B.mNumLinks[lane] = numLinks;
        B.mFlags[lane] = (a & 1u) ? PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES : 0u;

        uint32_t dofCursor = 0;
        for (uint32_t linkID = 0; linkID < numLinks; ++linkID)
        {
            PxgArticulationBlockLinkData& lnk = L[linkID];
            const PxQuat accQ = qv(a, linkID, 1);
            lnk.mAccumulatedPose.q[lane] = make_float4(accQ.x, accQ.y, accQ.z, accQ.w);
            lnk.mCfm[lane] = 0.01f + 0.005f * float((a + linkID) % 5);
            if (linkID == 0)
                continue;

            const JointCfg j = jointCase(a, linkID);
            lnk.mDofs[lane] = j.nbDofs;
            lnk.mParents[lane] = linkID - 1;
            lnk.mJointType[lane] = j.jointType;
            lnk.mRw_x[lane] = 0.5f * fv(a, linkID, 0, 7);
            lnk.mRw_y[lane] = 0.5f * fv(a, linkID, 1, 7);
            lnk.mRw_z[lane] = 0.5f * fv(a, linkID, 2, 7);
            const PxQuat paQ = qv(a, linkID, 2), chQ = qv(a, linkID, 3);
            lnk.mParentPose.q[lane] = make_float4(paQ.x, paQ.y, paQ.z, paQ.w);
            lnk.mChildPose.q[lane] = make_float4(chQ.x, chQ.y, chQ.z, chQ.w);
            lnk.mParentPose.p[lane] = make_float4(fv(a, linkID, 3, 8), fv(a, linkID, 4, 8),
                                                  fv(a, linkID, 5, 8), 0.f);
            lnk.mChildPose.p[lane] = make_float4(fv(a, linkID, 6, 8), fv(a, linkID, 7, 8),
                                                 fv(a, linkID, 8, 8), 0.f);
            storeSpatialVector(lnk.mBiasForce,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, linkID, 0, 9), fv(a, linkID, 1, 9), fv(a, linkID, 2, 9)),
                                   PxVec3(fv(a, linkID, 3, 9), fv(a, linkID, 4, 9), fv(a, linkID, 5, 9))),
                               lane);
            // parent's response matrix: synthetic but well-scaled
            {
                PxSpatialMatrix m;
                for (uint32_t c = 0; c < 6; ++c)
                    for (uint32_t r = 0; r < 6; ++r)
                        m.column[c][r] = 0.25f * fv(a, linkID, c * 6 + r, 10)
                                       + (c == r ? 1.0f : 0.0f);
                storeSpatialMatrix(L[linkID - 1].mSpatialResponseMatrix, lane, m);
            }

            for (uint32_t i = 0; i < j.nbDofs; ++i)
            {
                PxgArticulationBlockDofData& dd = D[dofCursor + i];
                dd.mMotion[lane] = j.motion[i];
                dd.mDofIds[lane] = j.dofIds[i];
                dd.mJointPositions[lane] = 0.8f * fv(a, linkID, i, 11);
                // local/world motion vectors + response inputs
                const PxVec3 axT(fv(a, linkID, i, 12), fv(a, linkID, i + 3, 12), fv(a, linkID, i + 6, 12));
                const PxVec3 axB(fv(a, linkID, i, 13), fv(a, linkID, i + 3, 13), fv(a, linkID, i + 6, 13));
                storeSpatialVector(dd.mLocalMotionMatrix, UnAlignedSpatialVector(axT, axB), lane);
                const PxVec3 wT(fv(a, linkID, i, 14), fv(a, linkID, i + 3, 14), fv(a, linkID, i + 6, 14));
                const PxVec3 wB(fv(a, linkID, i, 15), fv(a, linkID, i + 3, 15), fv(a, linkID, i + 6, 15));
                storeSpatialVector(dd.mWorldMotionMatrix, UnAlignedSpatialVector(wT, wB), lane);
                storeSpatialVector(dd.mIsInvDW,
                                   UnAlignedSpatialVector(
                                       PxVec3(fv(a, linkID, i, 16), fv(a, linkID, i + 3, 16), fv(a, linkID, i + 6, 16)),
                                       PxVec3(fv(a, linkID, i, 17), fv(a, linkID, i + 3, 17), fv(a, linkID, i + 6, 17))),
                                   lane);
                storeSpatialVector(dd.mIsW,
                                   UnAlignedSpatialVector(
                                       PxVec3(fv(a, linkID, i, 18), fv(a, linkID, i + 3, 18), fv(a, linkID, i + 6, 18)),
                                       PxVec3(fv(a, linkID, i, 19), fv(a, linkID, i + 3, 19), fv(a, linkID, i + 6, 19))),
                                   lane);
                dd.mInvStIsT_x[lane] = fv(a, linkID, i, 20);
                dd.mInvStIsT_y[lane] = fv(a, linkID, i, 21);
                dd.mInvStIsT_z[lane] = fv(a, linkID, i, 22);

                PxgArticulationInternalConstraintData& ic = dd.mConstraintData;
                ic.mFrictionCoefficient[lane] = 0.5f + 0.5f * fv(a, linkID, i, 23);
                ic.mDriveType[lane] = j.driveType[i];
                // a target chosen so some qB2qA dots go negative (case 6 gets a
                // large opposing target)
                // a % 8 == 7: spherical drive targets all ZERO, so driveAxis
                // degenerates and the angle<1e-12 fallback (axis (1,0,0),
                // angle 0) actually runs -- without it the normalize-guard
                // mutants are identities at fixture scale (the P8.96a lesson)
                ic.mDriveTargetPos[lane] =
                    (j.jointType == PxArticulationJointType::eSPHERICAL && (a % 8u) == 7u)
                        ? 0.f
                        : ((j.jointType == PxArticulationJointType::eREVOLUTE && j.dofIds[i] == 1)
                               ? 2.8f * fv(a, linkID, i, 24)
                               : 0.9f * fv(a, linkID, i, 24));
                ic.mDriveTargetVel[lane] = fv(a, linkID, i, 25);
                // cases 6/8 (revolute about swing1) are DAMPING-ONLY drives:
                // stiffness 0 exercises the stiffness>0 || damping>0 guard's
                // second arm
                const bool dampingOnly = j.jointType == PxArticulationJointType::eREVOLUTE
                                      && j.dofIds[0] == 1 && !j.zeroDrive;
                ic.mDriveStiffness[lane] =
                    (j.zeroDrive || dampingOnly) ? 0.f : 40.f + 10.f * fv(a, linkID, i, 26);
                ic.mDamping[lane] = 3.f + 1.f * fv(a, linkID, i, 27);
                ic.mMaxForce[lane] = j.zeroDrive ? 0.f : 100.f + 20.f * fv(a, linkID, i, 28);
                ic.mMaxEffort[lane] = 0.f;
                ic.mLimits_LowLimitX_highLimitY[lane] =
                    make_float2(-1.2f + 0.1f * fv(a, linkID, i, 29), 1.3f + 0.1f * fv(a, linkID, i, 30));
            }
            dofCursor += j.nbDofs;
        }
    }

    // ---- the mirror ----
    std::vector<PxgArticulationBlockDofData> wDf = df;
    std::vector<uint8_t> sphericalErr(df.size() * 32, 0);
    for (uint32_t a = 0; a < SI_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        refSetup(ab[w], &lk[size_t(w) * SI_MAX_LINKS], &wDf[size_t(w) * SI_MAX_DOFS],
                 SI_STEP_DT, SI_DT, isTGS, lane, &sphericalErr[size_t(w) * SI_MAX_DOFS * 32]);
    }

    // ---- device ----
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dAb = up(ab.data(), ab.size() * sizeof(ab[0]));
    PxgDevicePtr dLk = up(lk.data(), lk.size() * sizeof(lk[0]));
    PxgDevicePtr dDf = up(df.data(), df.size() * sizeof(df[0]));
    if (!dAb || !dLk || !dDf)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    struct Params
    {
        uint64_t ab, lk, df;
        float stepDt, dt, invDt;
        uint32_t isTGS, nb, maxLinks, maxDofs, pad0;
    } prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf),
              SI_STEP_DT, SI_DT, SI_INV_DT, isTGS ? 1u : 0u, SI_NB, SI_MAX_LINKS, SI_MAX_DOFS, 0 };
    PXG_CHECK_MSG(shim.launchKernel(0, SI_WARPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch siSetupInternalTest");
    shim.streamSynchronize();

    // ---- compare ----
    size_t bad = 0;
    // arti + link blocks must be UNTOUCHED
    {
        std::vector<uint8_t> g(ab.size() * sizeof(ab[0]));
        shim.memcpyDtoH(g.data(), dAb, g.size());
        if (std::memcmp(g.data(), ab.data(), g.size()) != 0) { ++bad; std::printf("MM si artiBlocks touched\n"); }
        std::vector<uint8_t> gl(lk.size() * sizeof(lk[0]));
        shim.memcpyDtoH(gl.data(), dLk, gl.size());
        if (std::memcmp(gl.data(), lk.data(), gl.size()) != 0) { ++bad; std::printf("MM si linkBlocks touched\n"); }
    }
    // dof blocks word-by-word; ULP-bounded only on the spherical-error drive
    // desc words
    {
        std::vector<PxgArticulationBlockDofData> g(df.size());
        shim.memcpyDtoH(g.data(), dDf, g.size() * sizeof(g[0]));
        const uint32_t* gw = reinterpret_cast<const uint32_t*>(g.data());
        const uint32_t* ww = reinterpret_cast<const uint32_t*>(wDf.data());
        const size_t stride = sizeof(PxgArticulationBlockDofData) / 4;
        int32_t maxUlp = 0;
        for (size_t k = 0; k < g.size() * stride; ++k)
        {
            if (gw[k] == ww[k])
                continue;
            const size_t blockIdx = k / stride;
            const size_t off = (k % stride) * 4;
            // desc float4 (5504..6015) + posBias (6400..6527) inside mConstraintData
            const size_t icBase = offsetof(PxgArticulationBlockDofData, mConstraintData);
            const bool descWord =
                off >= icBase + 5504 && off < icBase + 6016;
            const bool posBiasWord =
                off >= icBase + 6400 && off < icBase + 6528;
            const size_t wordLane = descWord ? (off - (icBase + 5504)) % 512 / 16
                                             : posBiasWord ? (off - (icBase + 6400)) / 4 : 0;
            if ((descWord || posBiasWord) && sphericalErr[blockIdx * 32 + wordLane])
            {
                float fg, fw;
                std::memcpy(&fg, &gw[k], 4); std::memcpy(&fw, &ww[k], 4);
                const int32_t u = ulpDiff(fg, fw);
                maxUlp = u > maxUlp ? u : maxUlp;
                if (u <= 512)
                    continue;
            }
            if (bad < 10)
                std::printf("MM si isTGS=%d dofBlock %zu off %zu  ref %08x  gpu %08x\n",
                            int(isTGS), blockIdx, off, ww[k], gw[k]);
            ++bad;
        }
        std::printf("[ info ] si isTGS=%d: max spherical-desc ULP %d\n", int(isTGS), maxUlp);
    }

    for (PxgDevicePtr dp : bufs) if (dp) shim.memFree(dp);
    return bad;
}

PXG_TEST(artiSetupInternal)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSetupInternal.spv", "siSetupInternalTest", 56u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &siLoader);

    size_t bad = 0;
    bad += siRun(shim, true);
    bad += siRun(shim, false);
    PXG_CHECK_MSG(bad == 0, "setupInternalConstraints does not match the engine");
}

}  // namespace
