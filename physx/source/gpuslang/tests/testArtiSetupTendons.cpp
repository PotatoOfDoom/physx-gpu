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


// P8.97b: npArtiSetupTendons.slang -- setupInternalSpatialTendonConstraints,
// setupInternalFixedTendonConstraints, and the link-pair impulse-response
// family (fast path / slow common-ancestor walk / dispatcher).
//
// THE ORACLE: the setups are __device__-only, so this is the P8.73-P8.80
// arrangement -- the mirror is control flow over engine primitives
// (loadSpatialTransform / loadSpatialVector / loadSpatialMatrix, PxQuat and
// PxTransform algebra, PxSpatialMatrix::operator*, Dy::FeatherstoneArticulation
// ::translateSpatialVector, UnAlignedSpatialVector::innerProduct), with the
// propagate hops transcribed over the block's mTmpQstZ exactly as the .cuh
// policies spell them.  No transcendentals anywhere (mul/add/sqrt/cross/
// quat), so every word compares BIT-EXACTLY.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//   * 68 articulations over 3 warps (warp 2 partial), numLinks 4..6 with a
//     BRANCH (link 4's parent is 1 where present), maxLinks 6 != maxDofs 8
//     != maxTendons 2 != maxElements 6 -- every count distinct;
//   * spatial tendon 0 on every articulation: root attachment (index 0) on
//     link 0 with children {1,2}; att1 = NON-leaf (child {3}) on link 1 --
//     the accumLength chain; att2 = leaf on link 1 (fast path: parent(1) ==
//     0 == root att link); att3 = leaf on the deepest chain link under
//     parent 1 (slow path, one-sided stack, common == root link).  Odd
//     articulations add att4 as a second root child whose relative offset
//     equals the root's WORLD attach point pulled back into its own link 0
//     frame -- dif == 0 exactly, the distance <= 0.001 branch;
//   * spatial tendon 1 (numLinks >= 5 only): root attachment on BRANCH link
//     4, leaf on chain link 3 -- the slow path with BOTH stacks non-empty
//     (common ancestor link 1);
//   * fixed tendon 0: tj0 root on link 1; tj1 (parent 0, angular axis 0) on
//     link 2 -- fast path, every joint emits; tj2 (parent 1, LINEAR axis 3)
//     on link 3 -- the transform-compose branch and mInvDofIds row 3; odd
//     articulations add tj3 (parent 0, angular axis 1) on branch link 4 --
//     the slow path from a fixed tendon;
//   * both solver modes (TGS + PGS dispatches over identically seeded
//     buffers);
//   * ITC blocks compared byte-for-byte with poison canaries past the live
//     constraints; tendon mNumConstraints, tj mConstraintId, and the
//     mTmpQstZ scratch the slow path leaves behind are all compared (the
//     scratch is observable state); link/attachment/tendon-joint inputs and
//     the arti block must come back untouched;
//   * P9.86c: the last 8 articulations are WITNESSES (stSeedWitness): a fixed
//     base (link 0's response is the zero matrix, forwardDynamic2.cu:1794) or
//     a free root with a block-diagonal response, links hanging along -y,
//     identity poses up to the signs of their zero components, axis-aligned
//     dofs and y-axis attachments -- the exact-zero dots and +-0 lanes the
//     P9.86 fixes of this shader are about.
//     Host twins of the three surviving audit-fix reverts (r1c, r3, r7) run
//     the mirror over them; see the note above stM6.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "foundation/PxTransform.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"

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

static std::vector<uint8_t> stLoader(const char* file) { return loadShaderBlob(file); }

// ST_NB_GENERIC hash-seeded articulations, then the P9.86c witness
// articulations (stWitness) up to ST_NB
const uint32_t ST_NB_GENERIC = 68, ST_NB = 76, ST_WARPS = 3;
static bool stWitness(uint32_t a) { return a >= ST_NB_GENERIC; }
const uint32_t ST_MAX_LINKS = 6, ST_MAX_DOFS = 8, ST_MAX_TENDONS = 2, ST_MAX_ELEMS = 6;
const float ST_STEP_DT = 0.004f, ST_DT = 0.0333f, ST_INV_DT = 250.0f;
const float ST_MIN_RESPONSE = 1e-12f;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}
static PxQuat qv(uint32_t a, uint32_t l, uint32_t salt)
{
    PxQuat q(fv(a, l, 0, salt), fv(a, l, 1, salt), fv(a, l, 2, salt),
             1.0f + 0.4f * fv(a, l, 3, salt));
    return q.getNormalized();
}

static uint32_t stNumLinks(uint32_t a) { return 4u + (a % 3u); }

// ---- P9.86c: host twins of the surviving audit-fix reverts ----
//
// The P9.86 fixes of npArtiSetupTendons.slang that the revert sweep could not
// kill are EQUIVALENT against HEAD -- no input changes an output bit:
//   r1c  stPropagateAccelC2P returns asmSVecAdd(pDeltaV, jointSpatialDeltaV)
//        without the stPlusZero3 select.  llvmpipe forwards the literal +0
//        seed of jointSpatialDeltaV and folds `0 + term`, so the sum differs
//        from the CUDA's (articulationDynamic.cuh:292-312) only where a zero
//        lane of the folded accumulator is -0 -- and pDeltaV + (-0) equals
//        pDeltaV + (+0) bit for bit unless pDeltaV == -0.  pDeltaV is never
//        -0: it is translate(-c2p, x) with x an asmM6MulVec row (a select seed
//        that is never -0, then five IEEE adds) or a previous hop's result
//        (never -0 by induction), and its bottom adds a cross product to such
//        a lane; in round-to-nearest x + y is -0 only when both are.
//   r3   the fast path's propagateImpulseW<WriteToProvidedQstZ> spelled as
//        adPropagateImpulseW (stZ = -dot, and the literal {0,0,0} qstZ seed
//        folded): stZ is -0 where the CUDA's 0.0f - dot is +0, at dot == +0
//        only.  Every consumer is +, -, * (differences stay in the sign of a
//        zero) until asmM6MulVec, whose select seed and adds of a never--0
//        partial sum map both zeros to the same bits (deltaV0), and until
//        pDeltaV + jointSpatialDeltaV with a never--0 pDeltaV (deltaV1).
//   r7   the slow path's propagateImpulseW<WriteToTempQstZ> spelled the same
//        way: mTmpQstZ = (runtime +0 from the zeroing loop) + stZ is +0 for
//        either zero, and Z0/Z1 reach nothing but asmM6MulVec(-(Z0 + Z1)).
// All three rest on asmM6MulVec's own P9.86 select seed (pxgSlangSpatialMatrix
// .slang, guarded by artiSpatialMatrixBlockIo's FOLD_LANE) and on IEEE
// subnormals.  gStTwin replays each revert in the mirror (ST_TW_M6FOLD replays
// the PRE-P9.86 asmM6MulVec: the +0 seed folded into the first product); the
// witness articulations are searched so that
//   * every twin alone leaves the whole image bit-identical (the equivalence),
//   * r1c and r7 on top of ST_TW_M6FOLD, and r3 on top of ST_TW_M6FOLD |
//     ST_TW_R1C, change output words (the witnesses do carry the revert's
//     zero sign up to the absorbers: without asmM6MulVec's seed -- and, for
//     r3, the pDeltaV + jointSpatialDeltaV canonicalisation -- each revert
//     would be killable, so the inputs are not vacuous),
//   * dot == +0 hops occur on both paths, the folded accumulator really holds
//     -0 lanes, and no pDeltaV lane is ever -0 (the premise of r1c).
enum : int { ST_TW_R1C = 1, ST_TW_R3 = 2, ST_TW_R7 = 4, ST_TW_M6FOLD = 8 };
static int gStTwin = 0;
static bool gStPlain = false;     // the plain mirror run: count the premises
static bool gStCountWit = false;  // ... and, on witness articulations, the coverage
static size_t gStFastDotPosZero = 0, gStSlowDotPosZero = 0;
static size_t gStAccFoldNegZero = 0, gStPdNegZero = 0;

static bool stIsNegZero(PxReal v) { return v == 0.f && std::signbit(v); }
static bool stIsPosZero(PxReal v) { return v == 0.f && !std::signbit(v); }

// PxSpatialMatrix::operator* (PxSpatialMatrix.h:58-77), or under ST_TW_M6FOLD
// the product the pre-P9.86 asmM6MulVec gave on llvmpipe (seed folded)
static UnAlignedSpatialVector stM6(const PxSpatialMatrix& m, const UnAlignedSpatialVector& s)
{
    if (!(gStTwin & ST_TW_M6FOLD))
        return m * s;
    const PxReal st[6] = { s.top.x, s.top.y, s.top.z, s.bottom.x, s.bottom.y, s.bottom.z };
    PxReal r[6];
    for (PxU32 i = 0; i < 6; ++i)
    {
        r[i] = m.column[0][i] * st[0];
        for (PxU32 j = 1; j < 6; ++j)
            r[i] += m.column[j][i] * st[j];
    }
    return UnAlignedSpatialVector(PxVec3(r[0], r[1], r[2]), PxVec3(r[3], r[4], r[5]));
}

// propagateImpulseW<...> with jointForce NULL: jf - dot = 0.0f - dot
// (articulationDynamic.cuh:143-144); the r3/r7 twins take the preloaded -dot
static PxReal stStZ(PxReal ip, bool twin, size_t& posZeroCounter)
{
    if (gStCountWit && stIsPosZero(ip))
        ++posZeroCounter;
    return twin ? -ip : 0.0f - ip;
}

// the tail of propagateAccelerationW_child2ParentPreLoaded (cuh:292-312): the
// joint terms go into their own +0-seeded vector, added to pDeltaV last; the
// r1c twin adds the accumulator with its seed folded away instead
static UnAlignedSpatialVector stC2PTail(const UnAlignedSpatialVector& pDeltaV, const PxReal t[3],
                                        const PxgArticulationBlockDofData* dofs, PxU32 dofCount,
                                        uint32_t lane)
{
    UnAlignedSpatialVector jointSpatialDeltaV(PxVec3(0.f), PxVec3(0.f));
    UnAlignedSpatialVector folded(PxVec3(0.f), PxVec3(0.f));
    bool first = true;
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxReal jDelta = dofs[ind].mInvStIsT_x[lane] * t[0]
                                + dofs[ind].mInvStIsT_y[lane] * t[1]
                                + dofs[ind].mInvStIsT_z[lane] * t[2];
            const UnAlignedSpatialVector term =
                loadSpatialVector(dofs[ind].mWorldMotionMatrix, lane) * jDelta;
            jointSpatialDeltaV += term;
            if (first)
                folded = term;
            else
                folded += term;
            first = false;
        }
    }
    if (gStPlain)
    {
        const PxReal jv[6] = { jointSpatialDeltaV.top.x, jointSpatialDeltaV.top.y, jointSpatialDeltaV.top.z,
                               jointSpatialDeltaV.bottom.x, jointSpatialDeltaV.bottom.y, jointSpatialDeltaV.bottom.z };
        const PxReal fw[6] = { folded.top.x, folded.top.y, folded.top.z,
                               folded.bottom.x, folded.bottom.y, folded.bottom.z };
        const PxReal pv[6] = { pDeltaV.top.x, pDeltaV.top.y, pDeltaV.top.z,
                               pDeltaV.bottom.x, pDeltaV.bottom.y, pDeltaV.bottom.z };
        for (int k = 0; k < 6; ++k)
        {
            if (gStCountWit && stIsPosZero(jv[k]) && stIsNegZero(fw[k]))
                ++gStAccFoldNegZero;
            if (stIsNegZero(pv[k]))
                ++gStPdNegZero;
        }
    }
    if (gStTwin & ST_TW_R1C)
        return first ? pDeltaV : pDeltaV + folded;   // no dof: `pd + 0` folds to pd
    return pDeltaV + jointSpatialDeltaV;
}

// ---- host mirrors of the propagate hops over mTmpQstZ ----

static UnAlignedSpatialVector refPropImpulseTemp(PxgArticulationBlockLinkData* lk,
                                                 PxgArticulationBlockDofData* df,
                                                 uint32_t linkID,
                                                 const UnAlignedSpatialVector& Z, uint32_t lane)
{
    PxgArticulationBlockLinkData& link = lk[linkID];
    const PxU32 jointOffset = link.mJointOffset[lane];
    const PxU32 dofCount = link.mDofs[lane];
    const PxVec3 c2p(link.mRw_x[lane], link.mRw_y[lane], link.mRw_z[lane]);
    PxgArticulationBlockDofData* dofs = df + jointOffset;

    UnAlignedSpatialVector temp = Z;
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const UnAlignedSpatialVector sas = loadSpatialVector(dofs[ind].mWorldMotionMatrix, lane);
            const UnAlignedSpatialVector isInvD = loadSpatialVector(dofs[ind].mIsInvDW, lane);
            // jf - s.Z, jf = 0 (cuh:143-144)
            const PxReal stZ = stStZ(sas.innerProduct(Z), (gStTwin & ST_TW_R7) != 0,
                                     gStSlowDotPosZero);
            dofs[ind].mTmpQstZ[lane] += stZ;
            temp += isInvD * stZ;
        }
    }
    return Dy::FeatherstoneArticulation::translateSpatialVector(c2p, temp);
}

static UnAlignedSpatialVector refPropAccelTemp(PxgArticulationBlockLinkData* lk,
                                               PxgArticulationBlockDofData* df,
                                               uint32_t linkID,
                                               const UnAlignedSpatialVector& deltaV, uint32_t lane)
{
    PxgArticulationBlockLinkData& link = lk[linkID];
    const PxU32 jointOffset = link.mJointOffset[lane];
    const PxU32 dofCount = link.mDofs[lane];
    const PxVec3 c2p(link.mRw_x[lane], link.mRw_y[lane], link.mRw_z[lane]);
    PxgArticulationBlockDofData* dofs = df + jointOffset;

    const UnAlignedSpatialVector pDeltaV =
        Dy::FeatherstoneArticulation::translateSpatialVector(-c2p, deltaV);
    PxReal t[3] = { 0.f, 0.f, 0.f };
    for (PxU32 ind = 0; ind < 3; ++ind)
        if (ind < dofCount)
            t[ind] = dofs[ind].mTmpQstZ[lane]
                   - loadSpatialVector(dofs[ind].mIsW, lane).innerProduct(pDeltaV);
    // child2ParentPreLoaded (articulationDynamic.cuh:293-312): joint terms go into
    // their own +0-seeded vector, added to pDeltaV last
    return stC2PTail(pDeltaV, t, dofs, dofCount, lane);
}

static void refImpulseSelfResponseSlow(uint32_t linkID0, uint32_t linkID1,
                                       const UnAlignedSpatialVector& impulse0,
                                       const UnAlignedSpatialVector& impulse1,
                                       UnAlignedSpatialVector& deltaV0,
                                       UnAlignedSpatialVector& deltaV1,
                                       PxgArticulationBlockData& ab,
                                       PxgArticulationBlockLinkData* lk,
                                       PxgArticulationBlockDofData* df, uint32_t lane)
{
    PxU32 stack[64];
    PxU32 i0 = linkID0, i1 = linkID1;
    while (i0 != i1)
    {
        if (i0 < i1) i1 = lk[i1].mParents[lane];
        else i0 = lk[i0].mParents[lane];
    }
    const PxU32 common = i0;

    UnAlignedSpatialVector Z0(-impulse0.top, -impulse0.bottom);
    UnAlignedSpatialVector Z1(-impulse1.top, -impulse1.bottom);

    const PxU32 numLinks = ab.mNumLinks[lane];
    for (PxU32 i = 0; i < numLinks; ++i)
    {
        const PxU32 jointOffset = lk[i].mJointOffset[lane];
        const PxU32 dofCount = lk[i].mDofs[lane];
        for (PxU32 j = 0; j < dofCount; ++j)
            df[jointOffset + j].mTmpQstZ[lane] = 0.f;
    }

    i0 = 0;
    for (PxU32 id = linkID0; id != common; id = lk[id].mParents[lane])
    {
        Z0 = refPropImpulseTemp(lk, df, id, Z0, lane);
        stack[i0++] = id;
    }
    i1 = i0;
    for (PxU32 id = linkID1; id != common; id = lk[id].mParents[lane])
    {
        Z1 = refPropImpulseTemp(lk, df, id, Z1, lane);
        stack[i1++] = id;
    }

    const UnAlignedSpatialVector ZZ = Z0 + Z1;
    PxSpatialMatrix spatialResponse;
    loadSpatialMatrix(lk[common].mSpatialResponseMatrix, lane, spatialResponse);
    const UnAlignedSpatialVector v = stM6(spatialResponse, -ZZ);

    UnAlignedSpatialVector dv1 = v;
    for (PxU32 index = i1; (index--) > i0;)
        dv1 = refPropAccelTemp(lk, df, stack[index], dv1, lane);

    UnAlignedSpatialVector dv0 = v;
    for (PxU32 index = i0; (index--) > 0;)
        dv0 = refPropAccelTemp(lk, df, stack[index], dv0, lane);

    deltaV0 = dv0;
    deltaV1 = dv1;
}

static void refImpulseSelfResponse(uint32_t linkId0, uint32_t linkId1,
                                   const UnAlignedSpatialVector& impulse0,
                                   const UnAlignedSpatialVector& impulse1,
                                   UnAlignedSpatialVector& deltaV0,
                                   UnAlignedSpatialVector& deltaV1,
                                   PxgArticulationBlockData& ab,
                                   PxgArticulationBlockLinkData* lk,
                                   PxgArticulationBlockDofData* df, uint32_t lane)
{
    if (lk[linkId1].mParents[lane] == linkId0)
    {
        PxSpatialMatrix parentResponse;
        loadSpatialMatrix(lk[linkId0].mSpatialResponseMatrix, lane, parentResponse);
        const PxU32 jointOffset = lk[linkId1].mJointOffset[lane];
        const PxU32 dofCount = lk[linkId1].mDofs[lane];
        const PxVec3 c2p(lk[linkId1].mRw_x[lane], lk[linkId1].mRw_y[lane],
                         lk[linkId1].mRw_z[lane]);
        PxgArticulationBlockDofData* dofs = df + jointOffset;

        // the all-dof fast path (internalConstraints2.cu:95)
        UnAlignedSpatialVector Z1W(-impulse1.top, -impulse1.bottom);
        PxReal qstZ[3] = { 0.f, 0.f, 0.f };
        UnAlignedSpatialVector temp = Z1W;
        for (PxU32 ind = 0; ind < 3; ++ind)
        {
            if (ind < dofCount)
            {
                const UnAlignedSpatialVector sas =
                    loadSpatialVector(dofs[ind].mWorldMotionMatrix, lane);
                const UnAlignedSpatialVector isInvD =
                    loadSpatialVector(dofs[ind].mIsInvDW, lane);
                // jf - s.Z, jf = 0
                const PxReal stZ = stStZ(sas.innerProduct(Z1W), (gStTwin & ST_TW_R3) != 0,
                                         gStFastDotPosZero);
                if (gStTwin & ST_TW_R3)
                    qstZ[ind] = stZ;   // the revert's literal {0,0,0} seed, folded
                else
                    qstZ[ind] += stZ;
                temp += isInvD * stZ;
            }
        }
        const UnAlignedSpatialVector Z0W =
            Dy::FeatherstoneArticulation::translateSpatialVector(c2p, temp);
        const UnAlignedSpatialVector dif = impulse0 - Z0W;
        const UnAlignedSpatialVector dV0 = stM6(parentResponse, dif);

        // acceleration back down (the qstZ-array variant)
        const UnAlignedSpatialVector pDeltaV =
            Dy::FeatherstoneArticulation::translateSpatialVector(-c2p, dV0);
        PxReal t[3] = { 0.f, 0.f, 0.f };
        for (PxU32 ind = 0; ind < 3; ++ind)
            if (ind < dofCount)
                t[ind] = qstZ[ind]
                       - loadSpatialVector(dofs[ind].mIsW, lane).innerProduct(pDeltaV);
        deltaV0 = dV0;
        deltaV1 = stC2PTail(pDeltaV, t, dofs, dofCount, lane);
    }
    else
    {
        refImpulseSelfResponseSlow(linkId0, linkId1, impulse0, impulse1, deltaV0, deltaV1,
                                   ab, lk, df, lane);
    }
}

// ---- host mirrors of the two setups (transcribed control flow) ----

static void refSetupSpatial(PxgArticulationBlockData& ab, PxgArticulationBlockLinkData* lk,
                            PxgArticulationBlockDofData* df,
                            PxgArticulationBlockSpatialTendonData* tendons,
                            PxgArticulationInternalTendonConstraintData* itc,
                            PxgArticulationBlockAttachmentData* attachments,
                            uint32_t maxAttachments, float stepDt, bool isTGS, uint32_t lane)
{
    const PxU32 numTendons = ab.mNumSpatialTendons[lane];
    PxReal accumLength[64];

    for (PxU32 i = 0; i < numTendons; ++i)
    {
        PxgArticulationBlockSpatialTendonData& tendonData = tendons[i];
        const PxReal stiffness = tendonData.mStiffness[lane];
        const PxReal damping = tendonData.mDamping[lane];
        const PxReal limitStiffness = tendonData.mLimitStiffness[lane];
        const PxReal offset = tendonData.mOffset[lane];

        PxgArticulationBlockAttachmentData* attachmentBlock = &attachments[i * maxAttachments];
        PxgArticulationInternalTendonConstraintData* constraintBlock = &itc[i * maxAttachments];

        const PxReal coefficient = attachmentBlock[0].mCoefficient[lane];
        PxU64 bitStack = attachmentBlock[0].mChildrens[lane];
        PxU32 stackCount = PxU32(__builtin_popcountll(bitStack));

        PxU32 parent = 0;
        PxU32 numConstraints = 0;

        accumLength[parent] = offset * coefficient;

        const PxU32 rAttachmentLinkIndex = attachmentBlock[0].mLinkIndex[lane];
        const PxTransform rB2w = loadSpatialTransform(lk[rAttachmentLinkIndex].mAccumulatedPose, lane);
        const PxVec3 rRa = rB2w.q.rotate(attachmentBlock[0].mRelativeOffset[lane]);
        const PxVec3 rAttachPoint = rB2w.p + rRa;
        PxVec3 rAxis(0.f), rRaXn(0.f);

        PxU32 child = 63 - PxU32(__builtin_clzll(bitStack));

        while (stackCount != 0)
        {
            stackCount--;

            PxgArticulationBlockAttachmentData& attachmentData = attachmentBlock[child];
            const PxU32 linkInd = attachmentData.mLinkIndex[lane];
            const PxReal cfm = lk[linkInd].mCfm[lane];

            const PxTransform cB2w = loadSpatialTransform(lk[linkInd].mAccumulatedPose, lane);
            const PxVec3 rb = cB2w.q.rotate(attachmentData.mRelativeOffset[lane]);
            const PxVec3 cAttachPoint = cB2w.p + rb;

            PxReal distance = 0.f;
            PxVec3 dif(0.f);
            if (parent == 0)
            {
                dif = rAttachPoint - cAttachPoint;
                distance = PxSqrt(dif.magnitudeSquared());
                rAxis = distance > 0.001f ? (dif / distance) : PxVec3(0.f);
                rRaXn = rRa.cross(rAxis);
            }
            else
            {
                PxgArticulationBlockAttachmentData& pAtt = attachmentBlock[parent];
                const PxU32 pLinkInd = pAtt.mLinkIndex[lane];
                const PxTransform pB2w = loadSpatialTransform(lk[pLinkInd].mAccumulatedPose, lane);
                const PxVec3 ra = pB2w.q.rotate(pAtt.mRelativeOffset[lane]);
                const PxVec3 pAttachPoint = pB2w.p + ra;
                dif = pAttachPoint - cAttachPoint;
                distance = PxSqrt(dif.magnitudeSquared());
            }

            const PxReal u = distance * attachmentData.mCoefficient[lane] + accumLength[parent];

            const PxU64 children = attachmentData.mChildrens[lane];

            if (children)
            {
                stackCount += PxU32(__builtin_popcountll(children));
                accumLength[child] = u;
            }
            else
            {
                const PxVec3 axis = distance > 0.001f ? (dif / distance) : PxVec3(0.f);
                const PxVec3 rbXn = rb.cross(axis);

                UnAlignedSpatialVector axis0(rAxis, rRaXn);
                UnAlignedSpatialVector axis1(-axis, -rbXn);
                UnAlignedSpatialVector deltaV0, deltaV1;
                refImpulseSelfResponse(rAttachmentLinkIndex, linkInd, axis0, axis1,
                                       deltaV0, deltaV1, ab, lk, df, lane);

                const PxReal r0 = deltaV0.bottom.dot(rAxis) + deltaV0.top.dot(rRaXn);
                const PxReal r1 = deltaV1.bottom.dot(axis) + deltaV1.top.dot(rbXn);
                const PxReal unitResponse = r0 - r1;
                const PxReal recipResponse =
                    unitResponse > ST_MIN_RESPONSE ? (1.0f / (unitResponse + cfm)) : 0.0f;

                PxgArticulationInternalTendonConstraintData& c = constraintBlock[numConstraints++];
                storeSpatialVector(c.mRow0, axis0, lane);
                storeSpatialVector(c.mRow1, -axis1, lane);
                c.mDeltaVA[lane] = r0;
                c.mRecipResponse[lane] = recipResponse;

                const PxReal aa = stepDt * (stepDt * stiffness + damping);
                const PxReal a2 = stepDt * (stepDt * limitStiffness + damping);
                const PxReal x = unitResponse > 0.f ? 1.0f / (1.0f + aa * unitResponse) : 0.f;
                const PxReal x2 = unitResponse > 0.f ? 1.0f / (1.0f + a2 * unitResponse) : 0.f;

                c.mVelMultiplier[lane] = -x * aa;
                c.mImpulseMultiplier[lane] = isTGS ? 1.f : 1.f - x;
                c.mBiasCoefficient[lane] = -stiffness * x * stepDt;
                c.mAppliedForce[lane] = 0.f;
                c.mAccumulatedLength[lane] = u;
                c.mLink0[lane] = rAttachmentLinkIndex;
                c.mLink1[lane] = linkInd;
                c.mLimitImpulseMultiplier[lane] = isTGS ? 1.f : 1.f - x2;
                c.mLimitBiasCoefficient[lane] = -limitStiffness * x2 * stepDt;
                c.mLimitAppliedForce[lane] = 0.f;
                c.mRestDistance[lane] = attachmentData.mRestDistance[lane];
                c.mLowLimit[lane] = attachmentData.mLowLimit[lane];
                c.mHighLimit[lane] = attachmentData.mHighLimit[lane];
            }

            if (stackCount > 0)
            {
                bitStack &= (~(1ull << child));
                bitStack |= children;
                child = 63 - PxU32(__builtin_clzll(bitStack));
                parent = attachmentBlock[child].mParents[lane];
            }
        }

        tendonData.mNumConstraints[lane] = numConstraints;
    }
}

static void refSetupFixed(PxgArticulationBlockData& ab, PxgArticulationBlockLinkData* lk,
                          PxgArticulationBlockDofData* df,
                          PxgArticulationBlockFixedTendonData* tendons,
                          PxgArticulationInternalTendonConstraintData* itc,
                          PxgArticulationBlockTendonJointData* tjs,
                          uint32_t maxTendonJoints, float stepDt, bool isTGS, uint32_t lane)
{
    const PxU32 numTendons = ab.mNumFixedTendons[lane];

    for (PxU32 i = 0; i < numTendons; ++i)
    {
        PxgArticulationBlockFixedTendonData& tendonData = tendons[i];
        const PxReal stiffness = tendonData.mStiffness[lane];
        const PxReal damping = tendonData.mDamping[lane];
        const PxReal limitStiffness = tendonData.mLimitStiffness[lane];

        PxgArticulationBlockTendonJointData* tjBlock = &tjs[i * maxTendonJoints];
        PxgArticulationInternalTendonConstraintData* constraintBlock = &itc[i * maxTendonJoints];

        PxU64 bitStack = tjBlock[0].mChildrens[lane];
        PxU32 stackCount = PxU32(__builtin_popcountll(bitStack));

        PxU32 parent = 0;
        PxU32 numConstraints = 0;

        const PxU32 sLinkIndex = tjBlock[0].mLinkIndex[lane];
        const PxTransform sB2w = loadSpatialTransform(lk[sLinkIndex].mAccumulatedPose, lane);

        PxVec3 sAxis(0.f), sRaXn(0.f);

        PxU32 child = 63 - PxU32(__builtin_clzll(bitStack));

        while (stackCount != 0)
        {
            stackCount--;

            PxgArticulationBlockTendonJointData& tjData = tjBlock[child];
            const PxU32 tjAxis = tjData.mAxis[lane];
            const PxU32 cLinkInd = tjData.mLinkIndex[lane];
            PxgArticulationBlockLinkData& cLink = lk[cLinkInd];

            const PxU32 parentLink = cLink.mParents[lane];
            const PxReal cfm = PxMax(lk[parentLink].mCfm[lane], cLink.mCfm[lane]);

            const PxU32 jointOffset = cLink.mJointOffset[lane];
            const PxU8 dofIndex = cLink.mInvDofIds[tjAxis][lane];
            PxgArticulationBlockDofData& dofData = df[jointOffset + dofIndex];

            const UnAlignedSpatialVector wmv = loadSpatialVector(dofData.mWorldMotionMatrix, lane);

            if (parent == 0)
            {
                if (tjAxis < PxArticulationAxis::eX)
                {
                    sAxis = PxVec3(0.f);
                    sRaXn = wmv.top;
                }
                else
                {
                    // 5.12: lever arm = sB2w.q.rotate(parentPose.p) (no world-scale
                    // subtraction)
                    const float4 p = cLink.mParentPose.p[lane];
                    sAxis = wmv.bottom;
                    sRaXn = sB2w.q.rotate(PxVec3(p.x, p.y, p.z)).cross(wmv.bottom);
                }
            }

            const PxU64 children = tjData.mChildrens[lane];

            PxVec3 axis, rbXn;
            if (tjAxis < PxArticulationAxis::eX)
            {
                axis = PxVec3(0.f);
                rbXn = wmv.top;
            }
            else
            {
                // 5.12: lever arm = cB2w.q.rotate(childPose.p)
                const PxTransform cB2w = loadSpatialTransform(cLink.mAccumulatedPose, lane);
                const float4 p = cLink.mChildPose.p[lane];
                axis = wmv.bottom;
                rbXn = cB2w.q.rotate(PxVec3(p.x, p.y, p.z)).cross(axis);
            }

            UnAlignedSpatialVector axis0(sAxis, sRaXn);
            UnAlignedSpatialVector axis1(-axis, -rbXn);
            UnAlignedSpatialVector deltaV0, deltaV1;
            refImpulseSelfResponse(sLinkIndex, cLinkInd, axis0, axis1,
                                   deltaV0, deltaV1, ab, lk, df, lane);

            const PxReal r0 = deltaV0.bottom.dot(sAxis) + deltaV0.top.dot(sRaXn);
            const PxReal r1 = deltaV1.bottom.dot(axis) + deltaV1.top.dot(rbXn);
            const PxReal unitResponse = r0 - r1;
            const PxReal recipResponse = 1.0f / (unitResponse + cfm);

            PxgArticulationInternalTendonConstraintData& c = constraintBlock[numConstraints];
            c.mDeltaVA[lane] = r0;
            storeSpatialVector(c.mDeltaVB, deltaV1, lane);
            storeSpatialVector(c.mRow0, axis0, lane);
            storeSpatialVector(c.mRow1, -axis1, lane);
            c.mRecipResponse[lane] = recipResponse;

            const PxReal aa = stepDt * (stepDt * stiffness + damping);
            const PxReal a2 = stepDt * (stepDt * limitStiffness + damping);
            const PxReal x = unitResponse > 0.f ? 1.0f / (1.0f + aa * unitResponse) : 0.f;
            const PxReal x2 = unitResponse > 0.f ? 1.0f / (1.0f + a2 * unitResponse) : 0.f;

            c.mVelMultiplier[lane] = -x * aa;
            c.mImpulseMultiplier[lane] = isTGS ? 1.f : 1.f - x;
            c.mBiasCoefficient[lane] = -stiffness * x * stepDt;
            c.mAppliedForce[lane] = 0.f;
            c.mLink0[lane] = sLinkIndex;
            c.mLink1[lane] = cLinkInd;
            c.mLimitImpulseMultiplier[lane] = isTGS ? 1.f : 1.f - x2;
            c.mLimitBiasCoefficient[lane] = -limitStiffness * x2 * stepDt;
            c.mLimitAppliedForce[lane] = 0.f;

            tjData.mConstraintId[lane] = numConstraints;
            numConstraints++;

            if (children)
                stackCount += PxU32(__builtin_popcountll(children));

            if (stackCount > 0)
            {
                bitStack &= (~(1ull << child));
                bitStack |= children;
                child = 63 - PxU32(__builtin_clzll(bitStack));
                parent = tjBlock[child].mParents[lane];
            }
        }

        tendonData.mNumConstraints[lane] = numConstraints;
    }
}

// ---- the fixture ----

struct StBuffers
{
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<PxgArticulationBlockSpatialTendonData> st;
    std::vector<PxgArticulationBlockFixedTendonData> ft;
    std::vector<PxgArticulationBlockAttachmentData> at;
    std::vector<PxgArticulationBlockTendonJointData> tj;
    std::vector<PxgArticulationInternalTendonConstraintData> itcS;
    std::vector<PxgArticulationInternalTendonConstraintData> itcF;
};

// ---- P9.86c witness articulations ----
//
// An articulation in an axis-aligned rest geometry: link 0 is a fixed base
// (its response is the zero matrix, zeroSpatialMatrix, forwardDynamic2.cu:
// 1794) or a free root body at its COM (a block-diagonal response, every
// other entry a signed zero); the links hang along -y on the y axis (link
// poses, mRw and the joint frames have +-0 x/z lanes), every quaternion is
// the identity up to the signs of its zero components and the double cover
// (w = +-1), and every dof is an axis-aligned revolute or prismatic joint
// with a diagonal inertia (world motion e_k with +-0 lanes, IsW = m e_k,
// IsInvD = e_k, invStIsT = 1/m on the diagonal) -- no prismatic-y dof, so the
// y-directed tendon forces meet exact-zero dots.  Spatial attachments sit on
// the y axis; the fixed tendon's root joint is on link 0, one tendon joint on
// link 1 (the fast path over link 0's response) and one on the deepest link
// (the slow path, common ancestor link 0).  The choices above, the signs of
// the zero lanes, the axes and the offsets come from `seed`; the seeds were
// searched on the host with the twins (every twin alone: 0 words changed;
// r1c/r7 on top of ST_TW_M6FOLD, r3 on top of ST_TW_M6FOLD | ST_TW_R1C: > 0
// words changed; teeth found for r1c on witnesses 0 2 4 5 7, r3 on 0 2 4, r7
// on 1 3 5 6) and are pinned by the coverage checks of the test body.
static const uint32_t gStWitnessSeeds[ST_NB - ST_NB_GENERIC] = { 13064u, 111u, 3914u, 347u,
                                                                 674u, 666u, 229u, 4u };

struct StWitnessRng
{
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float z() { return (next() & 1u) ? -0.0f : 0.0f; }   // a signed zero
    float sg() { return (next() & 1u) ? -1.0f : 1.0f; }
    bool bit() { return (next() & 1u) != 0u; }
};

static void stSeedWitness(StBuffers& B, uint32_t a, uint32_t seed)
{
    const uint32_t w = a / 32, lane = a % 32;
    PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * ST_MAX_LINKS];
    PxgArticulationBlockDofData* D = &B.df[size_t(w) * ST_MAX_DOFS];
    StWitnessRng r{ seed * 2654435761u + 0x9E3779B9u };
    for (int k = 0; k < 4; ++k) r.next();

    const uint32_t numLinks = stNumLinks(a);
    static const float ys[6] = { 0.f, -1.f, -2.f, -3.f, -1.5f, -2.5f };
    auto parentOf = [](uint32_t l) { return (l == 4) ? 1u : (l == 5 ? 4u : l - 1u); };
    PxVec3 pos[6];
    for (uint32_t l = 0; l < numLinks; ++l)
        pos[l] = PxVec3(r.z(), ys[l], r.z());
    auto zq = [&]() { return make_float4(r.z(), r.z(), r.z(), r.sg()); };

    uint32_t linkAx[6][2] = {};
    uint32_t linkNbDofs[6] = {};
    uint32_t dofCursor = 0;
    for (uint32_t l = 0; l < numLinks; ++l)
    {
        PxgArticulationBlockLinkData& lnk = L[l];
        lnk.mAccumulatedPose.q[lane] = zq();
        lnk.mAccumulatedPose.p[lane] = make_float4(pos[l].x, pos[l].y, pos[l].z, 0.f);
        if (l == 0)
        {
            // a fixed base: zeroSpatialMatrix (forwardDynamic2.cu:1794); or a
            // free root body at its COM with a diagonal inertia: angular velocity
            // from torque, linear velocity from force, every other entry a
            // signed zero
            const bool freeRoot = r.bit();
            PxSpatialMatrix m;
            for (uint32_t cI = 0; cI < 6; ++cI)
                for (uint32_t rI = 0; rI < 6; ++rI)
                    m.column[cI][rI] = freeRoot ? r.z() : 0.0f;
            if (freeRoot)
                for (uint32_t k = 0; k < 3; ++k)
                {
                    m.column[3 + k][k] = r.bit() ? 2.0f : 0.5f;   // torque k -> angular k
                    m.column[k][3 + k] = r.bit() ? 2.0f : 0.5f;   // force k -> linear k
                }
            storeSpatialMatrix(lnk.mSpatialResponseMatrix, lane, m);
            lnk.mParentPose.q[lane] = zq();
            lnk.mParentPose.p[lane] = make_float4(r.z(), r.z(), r.z(), 0.f);
            lnk.mChildPose.q[lane] = zq();
            lnk.mChildPose.p[lane] = make_float4(r.z(), r.z(), r.z(), 0.f);
            continue;
        }
        const uint32_t parent = parentOf(l);
        const PxVec3 rw = pos[l] - pos[parent];
        lnk.mRw_x[lane] = rw.x;
        lnk.mRw_y[lane] = rw.y;
        lnk.mRw_z[lane] = rw.z;
        // the joint half way between the two link origins
        const float half = 0.5f * (ys[l] - ys[parent]);
        lnk.mParentPose.q[lane] = zq();
        lnk.mParentPose.p[lane] = make_float4(r.z(), half, r.z(), 0.f);
        lnk.mChildPose.q[lane] = zq();
        lnk.mChildPose.p[lane] = make_float4(r.z(), -half, r.z(), 0.f);

        const uint32_t nbDofs = lnk.mDofs[lane];   // as the generic seeding (1 or 2)
        linkNbDofs[l] = nbDofs;
        static const uint32_t axPool[5] = { 0u, 1u, 2u, 3u, 5u };   // no eY (4)
        const uint32_t i0 = r.next() % 5u;
        const uint32_t i1 = (i0 + 1u + r.next() % 4u) % 5u;   // a different axis
        linkAx[l][0] = axPool[i0];
        linkAx[l][1] = axPool[i1];
        for (uint32_t ax = 0; ax < 6; ++ax)
            lnk.mInvDofIds[ax][lane] = 0;
        for (uint32_t d = 0; d < nbDofs; ++d)
        {
            const uint32_t ax = linkAx[l][d];
            lnk.mInvDofIds[ax][lane] = PxU8(d);
            const float sg = r.sg();
            const float m = r.bit() ? 2.0f : 0.5f;
            auto zv = [&]() { return PxVec3(r.z(), r.z(), r.z()); };
            PxVec3 sT = zv(), sB = zv(), dT = zv(), dB = zv(), wT = zv(), wB = zv();
            if (ax < 3)
            {
                sT[ax] = sg;          // angular: motion top, force-dual bottom
                dB[ax] = sg;
                wB[ax] = sg * m;
            }
            else
            {
                sB[ax - 3] = sg;      // linear: motion bottom, force-dual top
                dT[ax - 3] = sg;
                wT[ax - 3] = sg * m;
            }
            PxgArticulationBlockDofData& dd = D[dofCursor + d];
            storeSpatialVector(dd.mWorldMotionMatrix, UnAlignedSpatialVector(sT, sB), lane);
            storeSpatialVector(dd.mIsInvDW, UnAlignedSpatialVector(dT, dB), lane);
            storeSpatialVector(dd.mIsW, UnAlignedSpatialVector(wT, wB), lane);
            PxVec3 inv = zv();
            inv[d] = 1.0f / m;
            dd.mInvStIsT_x[lane] = inv.x;
            dd.mInvStIsT_y[lane] = inv.y;
            dd.mInvStIsT_z[lane] = inv.z;
        }
        dofCursor += nbDofs;
    }

    // spatial attachments: same tree as the generic seeding, offsets on the y axis
    PxgArticulationBlockAttachmentData* AT = &B.at[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS];
    static const float offs[4] = { 0.25f, -0.25f, 0.75f, -0.75f };
    for (uint32_t t = 0; t < B.ab[w].mNumSpatialTendons[lane]; ++t)
    {
        PxgArticulationBlockAttachmentData* T = &AT[t * ST_MAX_ELEMS];
        const uint32_t n = (t == 0) ? 5u : 2u;
        for (uint32_t idx = 0; idx < n; ++idx)
            T[idx].mRelativeOffset[lane] = PxVec3(r.z(), offs[r.next() % 4u], r.z());
        if (t == 0 && (a & 1u) != 0u)
        {
            T[4].mRelativeOffset[lane] = T[0].mRelativeOffset[lane];
            if ((a % 4u) == 3u)
                T[4].mRelativeOffset[lane].x += 4e-4f;
        }
    }

    // fixed tendon: root on link 0, tj1 on link 1 (fast), tj2 on the deepest link (slow)
    PxgArticulationBlockTendonJointData* TJ = &B.tj[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS];
    auto tj = [&](uint32_t idx, uint32_t linkIndex, uint32_t axis, uint32_t parent,
                  uint64_t children) {
        TJ[idx].mLinkIndex[lane] = PxU16(linkIndex);
        TJ[idx].mAxis[lane] = axis;
        TJ[idx].mParents[lane] = parent;
        TJ[idx].mChildrens[lane] = children;
    };
    const uint32_t deep = numLinks - 1u;
    const uint32_t ax1 = linkAx[1][r.next() % linkNbDofs[1]];
    const uint32_t ax2 = linkAx[deep][r.next() % linkNbDofs[deep]];
    if (r.bit())
    {
        tj(0, 0, 0, 0, (1ull << 1) | (1ull << 2));
        tj(1, 1, ax1, 0, 0);
        tj(2, deep, ax2, 0, 0);
    }
    else
    {
        tj(0, 0, 0, 0, 1ull << 1);
        tj(1, 1, ax1, 0, 1ull << 2);
        tj(2, deep, ax2, 1, 0);
    }
    for (uint32_t idx = 3; idx < ST_MAX_ELEMS; ++idx)
        tj(idx, 0, 0, 0, 0);
}

static void stSeed(StBuffers& B)
{
    B.ab.resize(ST_WARPS);
    B.lk.resize(size_t(ST_WARPS) * ST_MAX_LINKS);
    B.df.resize(size_t(ST_WARPS) * ST_MAX_DOFS);
    B.st.resize(size_t(ST_WARPS) * ST_MAX_TENDONS);
    B.ft.resize(size_t(ST_WARPS) * ST_MAX_TENDONS);
    B.at.resize(size_t(ST_WARPS) * ST_MAX_TENDONS * ST_MAX_ELEMS);
    B.tj.resize(size_t(ST_WARPS) * ST_MAX_TENDONS * ST_MAX_ELEMS);
    B.itcS.resize(size_t(ST_WARPS) * ST_MAX_TENDONS * ST_MAX_ELEMS);
    B.itcF.resize(size_t(ST_WARPS) * ST_MAX_TENDONS * ST_MAX_ELEMS);
    std::memset(B.ab.data(), 0xAB, B.ab.size() * sizeof(B.ab[0]));
    std::memset(B.lk.data(), 0xAB, B.lk.size() * sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.st.data(), 0xAB, B.st.size() * sizeof(B.st[0]));
    std::memset(B.ft.data(), 0xAB, B.ft.size() * sizeof(B.ft[0]));
    std::memset(B.at.data(), 0xAB, B.at.size() * sizeof(B.at[0]));
    std::memset(B.tj.data(), 0xAB, B.tj.size() * sizeof(B.tj[0]));
    std::memset(B.itcS.data(), 0xAB, B.itcS.size() * sizeof(B.itcS[0]));
    std::memset(B.itcF.data(), 0xAB, B.itcF.size() * sizeof(B.itcF[0]));

    for (uint32_t a = 0; a < ST_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * ST_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * ST_MAX_DOFS];

        const uint32_t numLinks = stNumLinks(a);
        AB.mNumLinks[lane] = numLinks;
        AB.mNumSpatialTendons[lane] = numLinks >= 5 ? 2 : 1;
        AB.mNumFixedTendons[lane] = 1;

        // links: chain 0-1-2-3, link 4 branches from 1, link 5 chains from 4
        uint32_t dofCursor = 0;
        for (uint32_t l = 0; l < numLinks; ++l)
        {
            PxgArticulationBlockLinkData& lnk = L[l];
            const PxQuat accQ = qv(a, l, 31);
            lnk.mAccumulatedPose.q[lane] = make_float4(accQ.x, accQ.y, accQ.z, accQ.w);
            lnk.mAccumulatedPose.p[lane] = make_float4(2.0f * fv(a, l, 0, 32),
                                                       2.0f * fv(a, l, 1, 32),
                                                       2.0f * fv(a, l, 2, 32), 0.f);
            lnk.mCfm[lane] = 0.01f + 0.004f * float((a + l) % 6);
            const PxQuat paQ = qv(a, l, 33), chQ = qv(a, l, 34);
            lnk.mParentPose.q[lane] = make_float4(paQ.x, paQ.y, paQ.z, paQ.w);
            lnk.mParentPose.p[lane] = make_float4(fv(a, l, 3, 35), fv(a, l, 4, 35),
                                                  fv(a, l, 5, 35), 0.f);
            lnk.mChildPose.q[lane] = make_float4(chQ.x, chQ.y, chQ.z, chQ.w);
            lnk.mChildPose.p[lane] = make_float4(fv(a, l, 6, 36), fv(a, l, 7, 36),
                                                 fv(a, l, 8, 36), 0.f);
            {
                PxSpatialMatrix m;
                for (uint32_t cI = 0; cI < 6; ++cI)
                    for (uint32_t r = 0; r < 6; ++r)
                        m.column[cI][r] = 0.2f * fv(a, l, cI * 6 + r, 37)
                                        + (cI == r ? 1.0f : 0.0f);
                storeSpatialMatrix(lnk.mSpatialResponseMatrix, lane, m);
            }
            if (l == 0)
            {
                lnk.mJointOffset[lane] = 0;
                lnk.mDofs[lane] = 0;
                continue;
            }
            lnk.mParents[lane] = (l == 4) ? 1 : (l == 5 ? 4 : l - 1);
            lnk.mRw_x[lane] = 0.5f * fv(a, l, 0, 38);
            lnk.mRw_y[lane] = 0.5f * fv(a, l, 1, 38);
            lnk.mRw_z[lane] = 0.5f * fv(a, l, 2, 38);
            const uint32_t nbDofs = 1u + ((a + l) % 2u);   // 1 or 2 dofs
            lnk.mJointOffset[lane] = dofCursor;
            lnk.mDofs[lane] = PxU8(nbDofs);
            // mInvDofIds: axis rows 0..5; dof d carries axis (l + d) % 6
            for (uint32_t ax = 0; ax < 6; ++ax)
                lnk.mInvDofIds[ax][lane] = 0;   // benign default, overwritten below
            for (uint32_t d = 0; d < nbDofs; ++d)
            {
                const uint32_t ax = (l + d) % 6u;
                lnk.mInvDofIds[ax][lane] = PxU8(d);
                PxgArticulationBlockDofData& dd = D[dofCursor + d];
                storeSpatialVector(dd.mWorldMotionMatrix,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 40), fv(a, l, d + 3, 40), fv(a, l, d + 6, 40)),
                        PxVec3(fv(a, l, d, 41), fv(a, l, d + 3, 41), fv(a, l, d + 6, 41))), lane);
                storeSpatialVector(dd.mIsInvDW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 42), fv(a, l, d + 3, 42), fv(a, l, d + 6, 42)),
                        PxVec3(fv(a, l, d, 43), fv(a, l, d + 3, 43), fv(a, l, d + 6, 43))), lane);
                storeSpatialVector(dd.mIsW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 44), fv(a, l, d + 3, 44), fv(a, l, d + 6, 44)),
                        PxVec3(fv(a, l, d, 45), fv(a, l, d + 3, 45), fv(a, l, d + 6, 45))), lane);
                dd.mInvStIsT_x[lane] = fv(a, l, d, 46);
                dd.mInvStIsT_y[lane] = fv(a, l, d, 47);
                dd.mInvStIsT_z[lane] = fv(a, l, d, 48);
            }
            dofCursor += nbDofs;
        }

        // ---- spatial tendons ----
        PxgArticulationBlockSpatialTendonData* ST = &B.st[size_t(w) * ST_MAX_TENDONS];
        PxgArticulationBlockAttachmentData* AT = &B.at[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS];
        for (uint32_t t = 0; t < AB.mNumSpatialTendons[lane]; ++t)
        {
            ST[t].mStiffness[lane] = 30.f + 8.f * fv(a, t, 0, 50);
            ST[t].mDamping[lane] = 2.f + 1.f * fv(a, t, 1, 50);
            ST[t].mLimitStiffness[lane] = 20.f + 6.f * fv(a, t, 2, 50);
            ST[t].mOffset[lane] = 0.4f * fv(a, t, 3, 50);
            PxgArticulationBlockAttachmentData* T = &AT[t * ST_MAX_ELEMS];
            auto att = [&](uint32_t idx, uint32_t linkIndex, uint32_t parent, uint64_t children) {
                T[idx].mLinkIndex[lane] = PxU16(linkIndex);
                T[idx].mParents[lane] = parent;
                T[idx].mChildrens[lane] = children;
                T[idx].mRelativeOffset[lane] = PxVec3(0.3f * fv(a, t * 8 + idx, 0, 51),
                                                      0.3f * fv(a, t * 8 + idx, 1, 51),
                                                      0.3f * fv(a, t * 8 + idx, 2, 51));
                T[idx].mCoefficient[lane] = 0.6f + 0.3f * fv(a, t * 8 + idx, 3, 51);
                T[idx].mRestDistance[lane] = fv(a, t * 8 + idx, 4, 51);
                T[idx].mLowLimit[lane] = -1.f + 0.2f * fv(a, t * 8 + idx, 5, 51);
                T[idx].mHighLimit[lane] = 1.f + 0.2f * fv(a, t * 8 + idx, 6, 51);
            };
            if (t == 0)
            {
                const bool degen = (a & 1u) != 0u;
                // root on link 0; children 1, 2 (+4 when degenerate case present)
                att(0, 0, 0, degen ? ((1ull << 1) | (1ull << 2) | (1ull << 4))
                                   : ((1ull << 1) | (1ull << 2)));
                att(1, 1, 0, 1ull << 3);          // NON-leaf -> accumLength chain
                // att2 = SLOW leaf (deep link), att3 = FAST leaf (link 1,
                // parent(1) == 0 == root att link) sitting UNDER att1 so the
                // DFS processes it LAST: the always-slow mutant leaves
                // different mTmpQstZ scratch only when the final response
                // call of the dispatch is a fast one -- with the slow leaf
                // last, its re-zero erased the evidence and the mutant
                // survived the first sweep
                att(2, numLinks - 1, 0, 0);       // leaf, deep link: slow, one-sided stack
                att(3, 1, 1, 0);                  // FAST leaf, processed last
                if (degen)
                {
                    // att4: leaf on the ROOT's own link.  a%4==1: identical
                    // relative offset -> dif == 0 exactly (the degenerate
                    // axis-zero branch).  a%4==3: offset by 4e-4 along x --
                    // both points rotate with the SAME quat, so the world
                    // distance is EXACTLY 4e-4, inside (0, 0.001]: the
                    // threshold's second regime, without which a 0.001->0
                    // mutation is an identity at fixture scale (the P8.96a
                    // scale lesson, in millimetres this time)
                    att(4, 0, 0, 0);
                    T[4].mRelativeOffset[lane] = T[0].mRelativeOffset[lane];
                    if ((a % 4u) == 3u)
                        T[4].mRelativeOffset[lane].x += 4e-4f;
                }
            }
            else
            {
                // root on BRANCH link 4, leaf on chain link 3: BOTH stacks
                // non-empty (common ancestor link 1)
                att(0, 4, 0, 1ull << 1);
                att(1, 3, 0, 0);
            }
        }

        // ---- fixed tendons ----
        PxgArticulationBlockFixedTendonData* FT = &B.ft[size_t(w) * ST_MAX_TENDONS];
        PxgArticulationBlockTendonJointData* TJ = &B.tj[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS];
        {
            FT[0].mStiffness[lane] = 25.f + 7.f * fv(a, 9, 0, 52);
            FT[0].mDamping[lane] = 1.5f + 0.8f * fv(a, 9, 1, 52);
            FT[0].mLimitStiffness[lane] = 15.f + 5.f * fv(a, 9, 2, 52);
            auto tj = [&](uint32_t idx, uint32_t linkIndex, uint32_t axis, uint32_t parent,
                          uint64_t children) {
                TJ[idx].mLinkIndex[lane] = PxU16(linkIndex);
                TJ[idx].mAxis[lane] = axis;
                TJ[idx].mParents[lane] = parent;
                TJ[idx].mChildrens[lane] = children;
            };
            const bool branch = numLinks >= 5 && (a & 1u) != 0u;
            // root on link 1 (its axis is never read -- left poisoned would be
            // ideal, but mAxis is PxU32 and the DFS never touches tj0's axis;
            // give it a benign value anyway since the mirror copies the seed)
            tj(0, 1, 0, 0, branch ? ((1ull << 1) | (1ull << 3)) : (1ull << 1));
            // tj1: parent 0, ANGULAR axis of link 2's dof 0 = axis (2+0)%6 = 2
            tj(1, 2, 2, 0, 1ull << 2);
            // tj2: parent 1, link 3's dof 0 axis = (3+0)%6 = 3 -> LINEAR
            tj(2, 3, 3, 1, 0);
            if (branch)
                tj(3, 4, 4 % 6, 0, 0);   // link 4 dof 0 axis = (4+0)%6 = 4 -> linear
        }
    }

    for (uint32_t a = ST_NB_GENERIC; a < ST_NB; ++a)
        stSeedWitness(B, a, gStWitnessSeeds[a - ST_NB_GENERIC]);
}

static void stMirror(StBuffers& W, bool isTGS, bool spatial, int aOnly = -1)
{
    for (uint32_t a = 0; a < ST_NB; ++a)
    {
        if (aOnly >= 0 && a != uint32_t(aOnly))
            continue;
        const uint32_t w = a / 32, lane = a % 32;
        gStCountWit = gStPlain && stWitness(a);
        if (spatial)
            refSetupSpatial(W.ab[w], &W.lk[size_t(w) * ST_MAX_LINKS],
                            &W.df[size_t(w) * ST_MAX_DOFS],
                            &W.st[size_t(w) * ST_MAX_TENDONS],
                            &W.itcS[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS],
                            &W.at[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS],
                            ST_MAX_ELEMS, ST_STEP_DT, isTGS, lane);
        else
            refSetupFixed(W.ab[w], &W.lk[size_t(w) * ST_MAX_LINKS],
                          &W.df[size_t(w) * ST_MAX_DOFS],
                          &W.ft[size_t(w) * ST_MAX_TENDONS],
                          &W.itcF[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS],
                          &W.tj[size_t(w) * ST_MAX_TENDONS * ST_MAX_ELEMS],
                          ST_MAX_ELEMS, ST_STEP_DT, isTGS, lane);
    }
    gStCountWit = false;
}

template <class T>
static size_t stWordDiffs(const std::vector<T>& x, const std::vector<T>& y)
{
    const uint32_t* xw = reinterpret_cast<const uint32_t*>(x.data());
    const uint32_t* yw = reinterpret_cast<const uint32_t*>(y.data());
    size_t n = 0;
    for (size_t k = 0; k < x.size() * sizeof(T) / 4; ++k)
        n += xw[k] != yw[k];
    return n;
}

// every word stRun compares against the device
static size_t stImageDiffs(const StBuffers& X, const StBuffers& Y, bool spatial)
{
    size_t n = stWordDiffs(X.ab, Y.ab) + stWordDiffs(X.lk, Y.lk) + stWordDiffs(X.df, Y.df);
    if (spatial)
        n += stWordDiffs(X.st, Y.st) + stWordDiffs(X.itcS, Y.itcS) + stWordDiffs(X.at, Y.at);
    else
        n += stWordDiffs(X.ft, Y.ft) + stWordDiffs(X.itcF, Y.itcF) + stWordDiffs(X.tj, Y.tj);
    return n;
}

static const int ST_TWINS[3] = { ST_TW_R1C, ST_TW_R3, ST_TW_R7 };
static const char* const ST_TWIN_NAMES[3] = { "r1c", "r3", "r7" };

struct StTwinStats
{
    size_t fastDotPosZero = 0, slowDotPosZero = 0, accFoldNegZero = 0, pdNegZero = 0;
    size_t alone[3] = { 0, 0, 0 };   // twin vs mirror: the equivalence, must be 0
    size_t teeth[3] = { 0, 0, 0 };   // twin on top of the absorbers' reverts: must be > 0
    size_t fold = 0;                 // M6FOLD vs mirror
};

// the plain mirror into W (counting the premises and the coverage), then every
// twin image of B, accumulated into S
static void stTwinRun(const StBuffers& B, StBuffers& W, bool isTGS, bool spatial,
                      StTwinStats& S, int aOnly = -1)
{
    W = B;
    gStFastDotPosZero = gStSlowDotPosZero = gStAccFoldNegZero = gStPdNegZero = 0;
    gStTwin = 0;
    gStPlain = true;
    stMirror(W, isTGS, spatial, aOnly);
    gStPlain = false;
    S.fastDotPosZero += gStFastDotPosZero;
    S.slowDotPosZero += gStSlowDotPosZero;
    S.accFoldNegZero += gStAccFoldNegZero;
    S.pdNegZero += gStPdNegZero;

    StBuffers F = B;
    gStTwin = ST_TW_M6FOLD;
    stMirror(F, isTGS, spatial, aOnly);
    S.fold += stImageDiffs(F, W, spatial);
    StBuffers FR = B;
    gStTwin = ST_TW_M6FOLD | ST_TW_R1C;
    stMirror(FR, isTGS, spatial, aOnly);
    for (int k = 0; k < 3; ++k)
    {
        StBuffers X = B;
        gStTwin = ST_TWINS[k];
        stMirror(X, isTGS, spatial, aOnly);
        S.alone[k] += stImageDiffs(X, W, spatial);
        // teeth: r1c and r7 on top of the folded M6 seed; r3 on top of the
        // folded seed AND r1c (its only other way out, deltaV1, ends in
        // pDeltaV + jointSpatialDeltaV)
        const StBuffers& base = (k == 1) ? FR : F;
        X = B;
        gStTwin = ST_TWINS[k] | ST_TW_M6FOLD | ((k == 1) ? ST_TW_R1C : 0);
        stMirror(X, isTGS, spatial, aOnly);
        S.teeth[k] += stImageDiffs(X, base, spatial);
    }
    gStTwin = 0;
}

static size_t stRun(PxgCudaShim& shim, bool isTGS, bool spatial, StTwinStats& S)
{
    StBuffers B;
    stSeed(B);

    // ---- mirror (and the P9.86c twins) ----
    StBuffers W;
    stTwinRun(B, W, isTGS, spatial, S);

    // ---- device ----
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dAb = up(B.ab.data(), B.ab.size() * sizeof(B.ab[0]));
    PxgDevicePtr dLk = up(B.lk.data(), B.lk.size() * sizeof(B.lk[0]));
    PxgDevicePtr dDf = up(B.df.data(), B.df.size() * sizeof(B.df[0]));
    PxgDevicePtr dTn = spatial ? up(B.st.data(), B.st.size() * sizeof(B.st[0]))
                               : up(B.ft.data(), B.ft.size() * sizeof(B.ft[0]));
    PxgDevicePtr dItc = spatial ? up(B.itcS.data(), B.itcS.size() * sizeof(B.itcS[0]))
                                : up(B.itcF.data(), B.itcF.size() * sizeof(B.itcF[0]));
    PxgDevicePtr dEl = spatial ? up(B.at.data(), B.at.size() * sizeof(B.at[0]))
                               : up(B.tj.data(), B.tj.size() * sizeof(B.tj[0]));
    if (!dAb || !dLk || !dDf || !dTn || !dItc || !dEl)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    struct Params
    {
        uint64_t ab, lk, df, tn, itc, el;
        float stepDt, dt, invDt;
        uint32_t isTGS, nb, maxLinks, maxDofs, maxTendons, maxElems, pad0;
    } prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dTn), uint64_t(dItc),
              uint64_t(dEl), ST_STEP_DT, ST_DT, ST_INV_DT, isTGS ? 1u : 0u,
              ST_NB, ST_MAX_LINKS, ST_MAX_DOFS, ST_MAX_TENDONS, ST_MAX_ELEMS, 0 };
    PXG_CHECK_MSG(shim.launchKernel(spatial ? 0 : 1, ST_WARPS, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch tendon setup");
    shim.streamSynchronize();

    // ---- compare: every buffer byte-for-byte against the mirror ----
    size_t bad = 0;
    auto cmp = [&](const char* what, PxgDevicePtr dp, const void* wp, size_t bytes) {
        std::vector<uint8_t> g(bytes);
        shim.memcpyDtoH(g.data(), dp, bytes);
        const uint32_t* gw = reinterpret_cast<const uint32_t*>(g.data());
        const uint32_t* ww = reinterpret_cast<const uint32_t*>(wp);
        for (size_t k = 0; k < bytes / 4; ++k)
            if (gw[k] != ww[k])
            {
                if (bad < 10)
                    std::printf("MM st %s%s%s word %zu (byte %zu)  ref %08x  gpu %08x\n",
                                spatial ? "spatial" : "fixed", isTGS ? "/tgs" : "/pgs",
                                what, k, k * 4, ww[k], gw[k]);
                ++bad;
            }
        if (bad) std::printf("[ info ] st cmp%s: %zu bad so far\n", what, bad);
    };
    cmp(" ab", dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]));
    cmp(" lk", dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]));
    cmp(" df", dDf, W.df.data(), W.df.size() * sizeof(W.df[0]));
    if (spatial)
    {
        cmp(" st", dTn, W.st.data(), W.st.size() * sizeof(W.st[0]));
        cmp(" itc", dItc, W.itcS.data(), W.itcS.size() * sizeof(W.itcS[0]));
        cmp(" at", dEl, W.at.data(), W.at.size() * sizeof(W.at[0]));
    }
    else
    {
        cmp(" ft", dTn, W.ft.data(), W.ft.size() * sizeof(W.ft[0]));
        cmp(" itc", dItc, W.itcF.data(), W.itcF.size() * sizeof(W.itcF[0]));
        cmp(" tj", dEl, W.tj.data(), W.tj.size() * sizeof(W.tj[0]));
    }

    for (PxgDevicePtr dp : bufs) if (dp) shim.memFree(dp);
    return bad;
}

PXG_TEST(artiSetupTendons)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSetupTendons.spv", "stSetupSpatialTest", 88u, 0 },
        { "npArtiSetupTendons.spv", "stSetupFixedTest", 88u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &stLoader);

    StTwinStats S;
    size_t bad = 0;
    bad += stRun(shim, true, true, S);
    bad += stRun(shim, false, true, S);
    bad += stRun(shim, true, false, S);
    bad += stRun(shim, false, false, S);
    PXG_CHECK_MSG(bad == 0, "the tendon setups do not match the engine");

    // ---- P9.86c: the twins of the surviving audit-fix reverts ----
    std::printf("[ info ] st twins: witness coverage -- fast dot +0 %zu, slow dot +0 %zu, "
                "folded-seed -0 accumulator lanes %zu; pDeltaV -0 lanes %zu; "
                "M6FOLD alone %zu words\n",
                S.fastDotPosZero, S.slowDotPosZero, S.accFoldNegZero, S.pdNegZero, S.fold);
    for (int k = 0; k < 3; ++k)
    {
        std::printf("[ info ] st twin %s: %zu words changed alone (must be 0), %zu on top of "
                    "%s (must be > 0)\n",
                    ST_TWIN_NAMES[k], S.alone[k], S.teeth[k],
                    k == 1 ? "M6FOLD+r1c" : "M6FOLD");
        PXG_CHECK_MSG(S.alone[k] == 0,
                      "twin %s changed %zu output words -- the equivalence argument is wrong",
                      ST_TWIN_NAMES[k], S.alone[k]);
        PXG_CHECK_MSG(S.teeth[k] > 0,
                      "witness coverage: twin %s never reaches asmM6MulVec's seed", ST_TWIN_NAMES[k]);
    }
    PXG_CHECK_MSG(S.fold > 0,
                  "witness coverage: the pre-P9.86 asmM6MulVec (seed folded) changes nothing");
    PXG_CHECK_MSG(S.fastDotPosZero > 0, "witness coverage: no fast-path dot == +0");
    PXG_CHECK_MSG(S.slowDotPosZero > 0, "witness coverage: no slow-path dot == +0");
    PXG_CHECK_MSG(S.accFoldNegZero > 0,
                  "witness coverage: no accumulator lane where the folded seed keeps -0");
    PXG_CHECK_MSG(S.pdNegZero == 0,
                  "a pDeltaV lane is -0 (%zu) -- the r1c equivalence premise fails", S.pdNegZero);
}

}  // namespace
