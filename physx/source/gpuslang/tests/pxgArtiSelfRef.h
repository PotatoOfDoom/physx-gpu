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

// pxgArtiSelfRef.h
//
// The self-constraint solve mirror (P8.99b): refPropAccelNoJVel — the
// propagateAccelerationWNoJVelUpdate hop in the DIRECT (everythingPreLoaded)
// summation order over pxgtest::refPropagateAccelerationW — and
// refSolveSelfLane, one lane of the internalConstraints2.cu:3861-4093 batch
// loop over the engine structs and the pxgArtiSolveRef/pxgArtiExtSolveRef
// mirrors.  Hoisted from testArtiSolveSelf.cpp (parameterization is the whole
// diff: the SS_* fixture globals became parameters) so the wrapper fixture
// (P8.99c) is a USER, not a second transcription.
//
// mTotalSelfConstraintCount / mSelfConstraintOffset are read INSIDE, as
// per-block scalars — exactly as the shader does (trap 7).

#ifndef PXG_ARTI_SELF_REF_H
#define PXG_ARTI_SELF_REF_H

#include "pxgArtiRefHelpers.h"
#include "pxgArtiSolveRef.h"
#include "pxgArtiExtSolveRef.h"

#include <cstdint>

namespace pxgtest
{
using namespace physx;
using physx::Cm::UnAlignedSpatialVector;

struct SelfCoverage
{
    PxU32 contacts = 0, oneD = 0, compliant = 0, torsional = 0, springRows = 0;
    PxU32 word1Hops = 0, equalPairs = 0, emptyBWalks = 0, offsetNeLane = 0, inactive = 0;
    PxU32 rootOnlyCommon = 0, zeroResp = 0;
};

inline UnAlignedSpatialVector refPropAccelNoJVel(const PxgArticulationBlockLinkData* lk,
                                                 PxU32 index,
                                                 const PxgArticulationBlockDofData* df,
                                                 const UnAlignedSpatialVector& hDeltaV,
                                                 PxU32 lane)
{
    const PxgArticulationBlockLinkData& L = lk[index];
    const PxU32 jointOffset = L.mJointOffset[lane];
    const PxU32 dofCount = L.mDofs[lane];
    const PxVec3 rw(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);

    UnAlignedSpatialVector motion[3], IsW[3];
    PxVec3 invStIsT[3];
    PxReal qstz[3] = { 0.f, 0.f, 0.f };
    for (PxU32 i = 0; i < 3; ++i)
    {
        motion[i] = UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f));
        IsW[i] = motion[i];
        invStIsT[i] = PxVec3(0.f);
        if (i < dofCount)
        {
            const PxgArticulationBlockDofData& D = df[jointOffset + i];
            motion[i] = loadSpatialVector(D.mWorldMotionMatrix, lane);
            IsW[i] = loadSpatialVector(D.mIsW, lane);
            invStIsT[i] = PxVec3(D.mInvStIsT_x[lane], D.mInvStIsT_y[lane], D.mInvStIsT_z[lane]);
            qstz[i] = D.mDeferredQstZ[lane];
        }
    }
    // the DIRECT (everythingPreLoaded) order — the NoJVelUpdate hop
    return refPropagateAccelerationW(rw, invStIsT, motion, hDeltaV, dofCount, IsW, qstz);
}

// one lane of the self-constraint batch loop; bf rows are
// [(link * wordSize + word) * 32 + lane], tmp rows [word * 32 + lane]
inline void refSolveSelfLane(PxgArticulationBlockData& ab,
                             PxgArticulationBlockLinkData* lk,
                             PxgArticulationBlockDofData* df,
                             const uint64_t* bf, PxU32 wordSize,
                             uint64_t* tmpA, uint64_t* tmpB, uint64_t* tmpC,
                             PxgBlockConstraintBatch* bt,
                             PxgTGSBlockSolverContactHeader* ch,
                             PxgTGSBlockSolverFrictionHeader* fh,
                             PxgTGSBlockSolverContactPoint* cp,
                             PxgTGSBlockSolverContactFriction* fp,
                             PxgTGSBlockSolverConstraint1DHeader* jh,
                             PxgTGSBlockSolverConstraint1DCon* jr,
                             PxgArticulationBlockResponse* rs,
                             float elapsedTime, float minPen, bool accumErr,
                             RefError& err, SelfCoverage& cov, PxU32 lane)
{
    const PxU32 nbSelf = ab.mTotalSelfConstraintCount;
    const PxU32 start = ab.mSelfConstraintOffset;

    UnAlignedSpatialVector rootZ = loadSpatialVector(ab.mRootDeferredZ, lane);
    Dy::SpatialMatrix M;
    loadSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, M);

    if (!nbSelf) return;

    for (PxU32 b = 0; b < nbSelf; ++b)
    {
        PxgBlockConstraintBatch& batch = bt[start + b];
        const PxU32 mask = batch.mask;
        if (!(mask & (1u << lane))) { ++cov.inactive; continue; }

        const UnAlignedSpatialVector commonDelta = M * -rootZ;
        const PxU32 offset = PxU32(__builtin_popcount(mask & ((1u << lane) - 1u)));
        if (offset != lane) ++cov.offsetNeLane;

        const PxU32 linkIDA = batch.bodyANodeIndex[offset].articulationLinkId();
        const PxU32 linkIDB = batch.bodyBNodeIndex[offset].articulationLinkId();
        if (linkIDA == linkIDB) ++cov.equalPairs;

        // stage the three temp buffers (lane-indexed)
        for (PxU32 j = 0; j < wordSize; ++j)
        {
            const uint64_t wordA = bf[size_t(linkIDA * wordSize + j) * 32 + lane];
            const uint64_t wordB = bf[size_t(linkIDB * wordSize + j) * 32 + lane];
            tmpA[size_t(j) * 32 + lane] = wordA;
            tmpB[size_t(j) * 32 + lane] = wordB;
            tmpC[size_t(j) * 32 + lane] = wordA & wordB;
        }

        UnAlignedSpatialVector velA = loadSpatialVector(lk[linkIDA].mMotionVelocity, lane);
        UnAlignedSpatialVector velB = loadSpatialVector(lk[linkIDB].mMotionVelocity, lane);

        UnAlignedSpatialVector deltaVA = commonDelta;
        {
            bool sawNonRoot = false;
            for (PxU32 j = 0, wo = 0; j < wordSize; ++j, wo += 64)
            {
                uint64_t word = tmpC[size_t(j) * 32 + lane];
                while (word != 0)
                {
                    const PxU32 index = PxU32(__builtin_ctzll(word)) + wo;
                    if (index != 0)
                    {
                        sawNonRoot = true;
                        if (index >= 64) ++cov.word1Hops;
                        deltaVA = refPropAccelNoJVel(lk, index, df, deltaVA, lane);
                    }
                    word &= (word - 1);
                }
            }
            if (!sawNonRoot) ++cov.rootOnlyCommon;
        }

        UnAlignedSpatialVector deltaVB = deltaVA;

        for (PxU32 j = 0; j < wordSize; ++j)
        {
            const uint64_t common = tmpC[size_t(j) * 32 + lane];
            tmpA[size_t(j) * 32 + lane] &= ~common;
            tmpB[size_t(j) * 32 + lane] &= ~common;
        }

        for (PxU32 j = 0, wo = 0; j < wordSize; ++j, wo += 64)
        {
            uint64_t word = tmpA[size_t(j) * 32 + lane];
            while (word != 0)
            {
                const PxU32 index = PxU32(__builtin_ctzll(word)) + wo;
                if (index >= 64) ++cov.word1Hops;
                deltaVA = refPropAccelNoJVel(lk, index, df, deltaVA, lane);
                word &= (word - 1);
            }
        }
        {
            bool any = false;
            for (PxU32 j = 0, wo = 0; j < wordSize; ++j, wo += 64)
            {
                uint64_t word = tmpB[size_t(j) * 32 + lane];
                while (word != 0)
                {
                    any = true;
                    const PxU32 index = PxU32(__builtin_ctzll(word)) + wo;
                    if (index >= 64) ++cov.word1Hops;
                    deltaVB = refPropAccelNoJVel(lk, index, df, deltaVB, lane);
                    word &= (word - 1);
                }
            }
            if (!any) ++cov.emptyBWalks;
        }

        velA += deltaVA;
        velB += deltaVB;

        UnAlignedSpatialVector impulse0(PxVec3(0.f), PxVec3(0.f));
        UnAlignedSpatialVector impulse1(PxVec3(0.f), PxVec3(0.f));

        const UnAlignedSpatialVector delta0 = loadSpatialVector(lk[linkIDA].mDeltaMotion, lane);
        const UnAlignedSpatialVector delta1 = loadSpatialVector(lk[linkIDB].mDeltaMotion, lane);

        if (batch.constraintType == PxgSolverConstraintDesc::eARTICULATION_CONTACT)
        {
            ++cov.contacts;
            const PxgTGSBlockSolverContactHeader& H = ch[batch.mConstraintBatchIndex];
            if (H.restitutionXdt[offset] < 0.f) ++cov.compliant;
            const PxU32 nbF = fh[batch.mConstraintBatchIndex].numFrictionConstr[offset];
            if (nbF & 1u) ++cov.torsional;
            const PxgTGSBlockSolverContactPoint& C0 = cp[batch.startConstraintIndex];
            if (C0.resp0[offset] == 0.f && C0.resp1[offset] == 0.f) ++cov.zeroResp;
            refSolveExtContactTGS(batch, velA, velB, delta0, delta1, offset,
                                  ch, fh, cp, fp, rs, elapsedTime, minPen, impulse0, impulse1,
                                  accumErr ? &err : NULL, 1.f, 1.f);
        }
        else
        {
            ++cov.oneD;
            const float4 q0 = lk[linkIDA].mDeltaQ[lane];
            const float4 q1 = lk[linkIDB].mDeltaQ[lane];
            const PxU32 rc = jh[batch.mConstraintBatchIndex].rowCounts_breakable_orthoAxisCount[offset].x;
            for (PxU32 i = 0; i < rc; ++i)
                if (jr[batch.startConstraintIndex + i].flags[offset] & DY_FLAG_SPRING)
                    ++cov.springRows;
            refSolveExt1DTGS(batch, velA, velB, delta0, delta1, offset,
                             jh, jr, rs,
                             PxQuat(q0.x, q0.y, q0.z, q0.w), PxQuat(q1.x, q1.y, q1.z, q1.w),
                             elapsedTime, impulse0, impulse1, accumErr, 1.f, 1.f);
        }

        // impulse walks: top word down, highest bit first
        for (int j = int(wordSize) - 1; j >= 0; --j)
        {
            const PxU32 wo = PxU32(j) * 64u;
            uint64_t word = tmpB[size_t(j) * 32 + lane];
            while (word != 0)
            {
                const PxU32 index = PxU32(63 - __builtin_clzll(word));
                word &= ~(uint64_t(1) << index);
                const PxgArticulationBlockLinkData& L = lk[index + wo];
                const PxVec3 c2p(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
                impulse1 = refPropImpulseDeferred(df, L.mJointOffset[lane], L.mDofs[lane],
                                                  c2p, impulse1, NULL, 1.f, lane);
            }
        }
        for (int j = int(wordSize) - 1; j >= 0; --j)
        {
            const PxU32 wo = PxU32(j) * 64u;
            uint64_t word = tmpA[size_t(j) * 32 + lane];
            while (word != 0)
            {
                const PxU32 index = PxU32(63 - __builtin_clzll(word));
                word &= ~(uint64_t(1) << index);
                const PxgArticulationBlockLinkData& L = lk[index + wo];
                const PxVec3 c2p(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
                impulse0 = refPropImpulseDeferred(df, L.mJointOffset[lane], L.mDofs[lane],
                                                  c2p, impulse0, NULL, 1.f, lane);
            }
        }

        impulse0 += impulse1;

        for (int j = int(wordSize) - 1; j >= 0; --j)
        {
            const PxU32 wo = PxU32(j) * 64u;
            uint64_t word = tmpC[size_t(j) * 32 + lane];
            while (word != 0)
            {
                const PxU32 index = PxU32(63 - __builtin_clzll(word));
                if ((index + wo) == 0) break;    // root: leaf-to-root, so done
                word &= ~(uint64_t(1) << index);
                const PxgArticulationBlockLinkData& L = lk[index + wo];
                const PxVec3 c2p(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
                impulse0 = refPropImpulseDeferred(df, L.mJointOffset[lane], L.mDofs[lane],
                                                  c2p, impulse0, NULL, 1.f, lane);
            }
        }

        rootZ += impulse0;
    }

    storeSpatialVector(ab.mRootDeferredZ, rootZ, lane);
}

}  // namespace pxgtest

#endif  // PXG_ARTI_SELF_REF_H
