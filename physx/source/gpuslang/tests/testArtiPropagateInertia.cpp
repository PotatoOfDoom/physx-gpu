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

// P8.87: computeIs + computePropagateSpatialInertia_ZA_ZIc
// (forwardDynamic2.cu:1345-1544).
//
// ===========================================================================
// THE ORACLE, STATED HONESTLY
// ===========================================================================
//
// Both CUDA functions are `static __device__` and unreachable from the host, so
// this is the P8.73-P8.80 arrangement and not P8.86's: the mirror below is the
// CUDA's CONTROL FLOW with the ENGINE's arithmetic and the ENGINE's block writers
// inside it — `SpatialMatrix::operator*`, `UnAlignedSpatialVector::innerProduct`,
// `PxMat33::getInverse`, both `constructSpatialMatrix` forms,
// `loadSpatialVector` / `storeSpatialVector`.  Every one of those has its own
// P8.86 fixture underneath it, so what this tranche is actually testing is the
// control flow — which is what the sweep attacks.
//
// ===========================================================================
// ELEVEN LANES, ELEVEN DIFFERENT DECISIONS
// ===========================================================================
//
//   0  eFIX (0),                dofs 0  -> the default path: writes NOTHING
//   1  ePRISMATIC (1),          dofs 1  -> stIs > 0, armature zero
//   2  eREVOLUTE (2),           dofs 1  -> stIs > 0, armature NON-zero
//   3  eREVOLUTE_UNWRAPPED (3), dofs 1  -> stIs EXACTLY zero: the `> 0` guard
//                                          fires and isInvD becomes signed zeros
//   4  eSPHERICAL (4),          dofs 1  -> the 3x3 keeps TWO identity rows
//   5  eSPHERICAL (4),          dofs 2  -> keeps ONE: the PxIdentity trap
//   6  eSPHERICAL (4),          dofs 3  -> DIAGONAL invStIs, so getInverse
//                                          produces exact -0 off-diagonals and the
//                                          the -0 coefficients that the sweep proved
//                                          are NOT enough to expose the fold
//   7  eUNDEFINED (5),          dofs 2  -> the default path with a NON-zero dof
//                                          count, so it is the switch and not the
//                                          count that selects it
//   8  eREVOLUTE (2),           dofs 1  -> stIs NEGATIVE: the other side of `> 0`
//   9  eSPHERICAL (4),          dofs 3  -> a dense, non-symmetric invStIs
//  10  eSPHERICAL (4),          dofs 1  -> a ZERO spatial inertia against an
//                                          all-negative motion matrix, so Is is all
//                                          negative zeros and isID has exactly ONE
//                                          accumulated term.  This is the only shape
//                                          in which the driver's `x + 0.0` fold is
//                                          observable at all — see the module header
//
// Lanes 11..31 are allocated and left poisoned: the entry point's `lane >=
// nbLanes` guard has to hold, and a lane-stride error lands there.
//
// The regimes on lanes 3 and 8 are ASSERTED, not assumed — the fixture computes
// `stIs` with the engine and checks its sign, so a fixture that drifts is caught
// instead of quietly testing the same branch twice.
//
// ===========================================================================
// WHAT MUST COME BACK UNTOUCHED
// ===========================================================================
//
// The link block and all three dof blocks are poisoned to 0xAB and compared byte
// for byte.  The function reads four fields (`mWorldMotionMatrix`, the armature,
// `mJointType`, `mDofs`) and writes six (`mIsW`, `mIsInvDW`, `mQstZ`,
// `mQstZIcInternal`, `mInvStIsT_{x,y,z}`); everything else — including every slot
// of a dof at or beyond that lane's dof count, and everything on lanes 0 and 7 —
// must come back at 0xAB.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxMat33.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "PxgBodySim.h"
#include "PxNodeIndex.h"
#include "pxgArtiRefHelpers.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;
using physx::Cm::UnAlignedSpatialVector;
using physx::Cm::SpatialVectorF;
using physx::Dy::SpatialMatrix;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiPropagateLoader(const char* file) { return loadShaderBlob(file); }

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }

const uint32_t LANES = 11;         // live lanes
const uint32_t MAX_DOFS = 3;       // dof blocks in the run
const uint32_t SCRATCH_FLOATS = 52;
const uint32_t INERTIA_FLOATS = 27;
const uint32_t JF_PER_LANE = 3;

// PxArticulationJointType, PxSolverDefs.h:301
const uint8_t J_FIX = 0, J_PRISMATIC = 1, J_REVOLUTE = 2, J_REVOLUTE_UNWRAPPED = 3,
              J_SPHERICAL = 4, J_UNDEFINED = 5;

struct LaneCfg { uint8_t jointType; uint8_t dofs; };
const LaneCfg CFG[LANES] = {
    { J_FIX, 0 },        { J_PRISMATIC, 1 }, { J_REVOLUTE, 1 },  { J_REVOLUTE_UNWRAPPED, 1 },
    { J_SPHERICAL, 1 },  { J_SPHERICAL, 2 }, { J_SPHERICAL, 3 }, { J_UNDEFINED, 2 },
    { J_REVOLUTE, 1 },   { J_SPHERICAL, 3 }, { J_SPHERICAL, 1 }
};

// ---------------------------------------------------------------------------
// The mirror: the CUDA's control flow, the engine's arithmetic and stores
// ---------------------------------------------------------------------------
static void apiMirror(PxgArticulationBlockLinkData& lk, PxgArticulationBlockDofData* df,
                      const SpatialMatrix& spatialInertia, bool isExtEvery, const PxReal* jF,
                      const UnAlignedSpatialVector& Z, const UnAlignedSpatialVector& ZIcInt,
                      UnAlignedSpatialVector& ZA, UnAlignedSpatialVector& ZAInt,
                      SpatialMatrix& out, uint32_t lane)
{
    const PxU32 dofs = lk.mDofs[lane];
    const PxU8 jointType = lk.mJointType[lane];

    // ---- computeIs (forwardDynamic2.cu:1347) ----
    // the SAME poison the entry point seeds -- see its comment: zeros here would
    // make a dropped `ind < dofs` guard harmless
    UnAlignedSpatialVector msIs[3];
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        const float b = 700.f + float(lane) * 13.f + float(ind) * 3.f;
        msIs[ind] = UnAlignedSpatialVector(PxVec3(b + 0.f, b + 1.f, b + 2.f),
                                           PxVec3(b + 3.f, b + 4.f, b + 5.f));
    }
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofs)
        {
            const UnAlignedSpatialVector lIs =
                spatialInertia * loadSpatialVector(df[ind].mWorldMotionMatrix, lane);
            msIs[ind] = lIs;
            storeSpatialVector(df[ind].mIsW, lIs, lane);
        }
    }

    // ---- computePropagateSpatialInertia_ZA_ZIc (:1401) ----
    if (jointType == J_PRISMATIC || jointType == J_REVOLUTE || jointType == J_REVOLUTE_UNWRAPPED)
    {
        const UnAlignedSpatialVector sa = loadSpatialVector(df[0].mWorldMotionMatrix, lane);
        const UnAlignedSpatialVector& Is = msIs[0];

        const PxReal stIs = sa.innerProduct(Is) + df[0].mConstraintData.mArmature[lane];
        const PxReal iStIs = ((stIs > 0.f) ? (1.f / stIs) : 0.f);
        df[0].mInvStIsT_x[lane] = iStIs;

        const UnAlignedSpatialVector isInvD = Is * iStIs;
        storeSpatialVector(df[0].mIsInvDW, isInvD, lane);
        const UnAlignedSpatialVector stI(Is.bottom, Is.top);

        const PxReal stZ = sa.innerProduct(Z);
        const PxReal stZInt = sa.innerProduct(ZIcInt);

        const PxReal qstZ = -stZ;
        // 5.12: the joint force enters qstZIcInternal only when it is not applied
        // every TGS iteration, and is parked in mExternalEffort for the dof solve
        const PxReal qstZIcInternal = ((!isExtEvery && jF) ? jF[0] : 0.0f) - stZInt;
        df[0].mQstZ[lane] = qstZ;
        df[0].mQstZIcInternal[lane] = qstZIcInternal;
        df[0].mConstraintData.mExternalEffort[lane] = jF ? jF[0] : 0.0f;

        ZA += isInvD * qstZ;
        ZAInt += isInvD * qstZIcInternal;

        out = dySmConstruct(SpatialVectorF(isInvD.top, isInvD.bottom),
                            SpatialVectorF(stI.top, stI.bottom));
        return;
    }

    if (jointType == J_SPHERICAL)
    {
        PxMat33 invStIs(PxIdentity);
        PxReal qstZG[3] = { 0.f, 0.f, 0.f };
        PxReal qstZIcIntG[3] = { 0.f, 0.f, 0.f };

        for (PxU32 ind2 = 0; ind2 < 3; ++ind2)
        {
            if (ind2 < dofs)
            {
                const UnAlignedSpatialVector sa = loadSpatialVector(df[ind2].mWorldMotionMatrix, lane);
                for (PxU32 ind = 0; ind < 3; ++ind)
                {
                    if (ind < dofs)
                        invStIs[ind][ind2] = sa.innerProduct(msIs[ind]);
                }
                invStIs[ind2][ind2] += df[ind2].mConstraintData.mArmature[lane];

                const PxReal stZ = sa.innerProduct(Z);
                const PxReal stZInt = sa.innerProduct(ZIcInt);
                const PxReal qstZ = -stZ;
                const PxReal qstZicInt = ((!isExtEvery && jF) ? jF[ind2] : 0.0f) - stZInt;
                qstZG[ind2] = qstZ;
                qstZIcIntG[ind2] = qstZicInt;
                df[ind2].mQstZ[lane] = qstZ;
                df[ind2].mQstZIcInternal[lane] = qstZicInt;
                df[ind2].mConstraintData.mExternalEffort[lane] = jF ? jF[ind2] : 0.0f;
            }
        }

        invStIs = invStIs.getInverse();
        for (PxU32 ind2 = 0; ind2 < 3; ++ind2)
        {
            if (ind2 < dofs)
            {
                df[ind2].mInvStIsT_x[lane] = invStIs[0][ind2];
                df[ind2].mInvStIsT_y[lane] = invStIs[1][ind2];
                df[ind2].mInvStIsT_z[lane] = invStIs[2][ind2];
            }
        }

        UnAlignedSpatialVector columns[6];
        for (PxU32 k = 0; k < 6; ++k)
            columns[k] = UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f));

        for (PxU32 ind = 0; ind < 3; ++ind)
        {
            if (ind < dofs)
            {
                UnAlignedSpatialVector isID(PxVec3(0.f), PxVec3(0.f));
                for (PxU32 ind2 = 0; ind2 < 3; ++ind2)
                {
                    if (ind2 < dofs)
                        isID += msIs[ind2] * invStIs[ind][ind2];
                }

                columns[0] += isID * msIs[ind].bottom.x;
                columns[1] += isID * msIs[ind].bottom.y;
                columns[2] += isID * msIs[ind].bottom.z;
                columns[3] += isID * msIs[ind].top.x;
                columns[4] += isID * msIs[ind].top.y;
                columns[5] += isID * msIs[ind].top.z;

                storeSpatialVector(df[ind].mIsInvDW, isID, lane);

                ZA += isID * qstZG[ind];
                ZAInt += isID * qstZIcIntG[ind];
            }
        }

        out = dySmConstructColumns(columns);
        return;
    }

    out = SpatialMatrix(PxZero);
}

static void putM33(float* p, const PxMat33& m)
{
    p[0] = m.column0.x; p[1] = m.column0.y; p[2] = m.column0.z;
    p[3] = m.column1.x; p[4] = m.column1.y; p[5] = m.column1.z;
    p[6] = m.column2.x; p[7] = m.column2.y; p[8] = m.column2.z;
}

static void putVec(float* p, const UnAlignedSpatialVector& v)
{
    p[0] = v.top.x;    p[1] = v.top.y;    p[2] = v.top.z;
    p[3] = v.bottom.x; p[4] = v.bottom.y; p[5] = v.bottom.z;
}

static UnAlignedSpatialVector getVec(const float* p)
{
    return UnAlignedSpatialVector(PxVec3(p[0], p[1], p[2]), PxVec3(p[3], p[4], p[5]));
}

PXG_TEST(artiPropagateInertia)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64 x5, u32 x2 } -> 48 bytes
        { "npArtiPropagateInertia.spv", "artiPropagateInertia", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiPropagateLoader);

    // non-round, distinct per lane and per slot
    auto val = [](uint32_t lane, uint32_t k) {
        return (float(lane) * 23.0f + float(k) * 3.0f + 1.0f) * 0.078125f - 5.0f;
    };

    // -----------------------------------------------------------------------
    // The per-lane inputs
    // -----------------------------------------------------------------------
    std::vector<SpatialMatrix> inertia(LANES);
    std::vector<UnAlignedSpatialVector> motion(size_t(LANES) * MAX_DOFS);
    std::vector<float> armature(size_t(LANES) * MAX_DOFS, 0.f);
    std::vector<UnAlignedSpatialVector> Zin(LANES), ZIcIn(LANES), ZAin(LANES), ZAIntIn(LANES);
    std::vector<float> jointForces(size_t(LANES) * JF_PER_LANE);

    for (uint32_t L = 0; L < LANES; ++L)
    {
        // a dense, deliberately NON-symmetric spatial inertia by default
        PxMat33 tl(PxVec3(val(L, 0), val(L, 1), val(L, 2)),
                   PxVec3(val(L, 3), val(L, 4), val(L, 5)),
                   PxVec3(val(L, 6), val(L, 7), val(L, 8)));
        PxMat33 tr(PxVec3(val(L, 9), val(L, 10), val(L, 11)),
                   PxVec3(val(L, 12), val(L, 13), val(L, 14)),
                   PxVec3(val(L, 15), val(L, 16), val(L, 17)));
        PxMat33 bl(PxVec3(val(L, 18), val(L, 19), val(L, 20)),
                   PxVec3(val(L, 21), val(L, 22), val(L, 23)),
                   PxVec3(val(L, 24), val(L, 25), val(L, 26)));
        inertia[L] = SpatialMatrix(tl, tr, bl);

        for (uint32_t d = 0; d < MAX_DOFS; ++d)
        {
            const size_t x = size_t(L) * MAX_DOFS + d;
            motion[x] = UnAlignedSpatialVector(
                PxVec3(val(L, 30 + d * 6 + 0), val(L, 30 + d * 6 + 1), val(L, 30 + d * 6 + 2)),
                PxVec3(val(L, 30 + d * 6 + 3), val(L, 30 + d * 6 + 4), val(L, 30 + d * 6 + 5)));
            armature[x] = 0.f;
        }

        Zin[L] = UnAlignedSpatialVector(PxVec3(val(L, 60), val(L, 61), val(L, 62)),
                                        PxVec3(val(L, 63), val(L, 64), val(L, 65)));
        ZIcIn[L] = UnAlignedSpatialVector(PxVec3(val(L, 66), val(L, 67), val(L, 68)),
                                          PxVec3(val(L, 69), val(L, 70), val(L, 71)));
        // ZA / ZAInt come IN with a value the function accumulates onto — a
        // literal-zero seed here would hide an accumulator that overwrites
        ZAin[L] = UnAlignedSpatialVector(PxVec3(val(L, 72), val(L, 73), val(L, 74)),
                                         PxVec3(val(L, 75), val(L, 76), val(L, 77)));
        ZAIntIn[L] = UnAlignedSpatialVector(PxVec3(val(L, 78), val(L, 79), val(L, 80)),
                                            PxVec3(val(L, 81), val(L, 82), val(L, 83)));

        for (uint32_t k = 0; k < JF_PER_LANE; ++k)
            jointForces[size_t(L) * JF_PER_LANE + k] = val(L, 90 + k);
    }

    // ---- lane 1: stIs POSITIVE.  The dense default happens to give a negative
    //      dual product here, so the inertia is negated to flip it — measured by
    //      the regime assertion below, not assumed. ----
    inertia[1] = SpatialMatrix(inertia[1].topLeft * -1.f, inertia[1].topRight * -1.f,
                               inertia[1].bottomLeft * -1.f);

    // ---- lane 2: a non-zero armature ----
    armature[size_t(2) * MAX_DOFS + 0] = 0.375f;

    // ---- lane 3: stIs EXACTLY zero.  A zero spatial inertia makes Is a vector of
    //      signed zeros, so the dual product is zero and the `> 0` guard fires;
    //      isInvD is then a vector of signed zeros too. ----
    inertia[3] = SpatialMatrix(PxZero);

    // ---- lane 6: a DIAGONAL invStIs.  The spatial identity maps s -> s, and three
    //      pure-ANGULAR motion vectors are mutually orthogonal under the DUAL
    //      product (bottom.top + top.bottom, both halves zero), so every
    //      off-diagonal entry is exactly zero and only the armature survives on the
    //      diagonal.  getInverse then emits exact -0 off-diagonals. ----
    inertia[6] = SpatialMatrix(PxMat33(PxIdentity), PxMat33(PxZero), PxMat33(PxZero));
    for (uint32_t d = 0; d < MAX_DOFS; ++d)
    {
        PxVec3 e(0.f, 0.f, 0.f);
        (&e.x)[d] = 1.f;
        motion[size_t(6) * MAX_DOFS + d] = UnAlignedSpatialVector(e, PxVec3(0.f));
        armature[size_t(6) * MAX_DOFS + d] = 0.5f + 0.25f * float(d);
    }

    // ---- lane 8: stIs NEGATIVE.  Negating the inertia negates Is and hence the
    //      dual product; the armature stays zero so it cannot rescue the sign. ----
    inertia[8] = SpatialMatrix(inertia[8].topLeft * -1.f, inertia[8].topRight * -1.f,
                               inertia[8].bottomLeft * -1.f);

    // lane 9 keeps the dense default: a non-symmetric invStIs, which is where
    // invertSym33 would disagree with getInverse (P8.86 measured that).

    // ---- lane 10: the ONE case where an accumulator's seed sign is observable.
    //      A one-dof spherical joint gives isID exactly ONE term, so there is no
    //      second term to wash the seed out via `(-0) + (+0) = +0`; and a ZERO
    //      spatial inertia (a massless link) against an all-NEGATIVE motion matrix
    //      makes every component of `M * s` a `0 * negative = -0`, so that one term
    //      is a negative zero.  Without this lane the P8.84 hazard is unobservable
    //      and the module header's claim about it would be empty — the sweep found
    //      that out, it was not designed in. ----
    inertia[10] = SpatialMatrix(PxZero);
    motion[size_t(10) * MAX_DOFS + 0] =
        UnAlignedSpatialVector(PxVec3(-1.5f, -2.25f, -3.125f), PxVec3(-4.75f, -5.5f, -6.875f));
    armature[size_t(10) * MAX_DOFS + 0] = 0.75f;

    // -----------------------------------------------------------------------
    // Assert the REGIMES rather than assuming them
    // -----------------------------------------------------------------------
    uint32_t nPos = 0, nZero = 0, nNeg = 0;
    for (uint32_t L = 0; L < LANES; ++L)
    {
        if (CFG[L].jointType != J_PRISMATIC && CFG[L].jointType != J_REVOLUTE
            && CFG[L].jointType != J_REVOLUTE_UNWRAPPED)
            continue;
        const UnAlignedSpatialVector sa = motion[size_t(L) * MAX_DOFS + 0];
        const UnAlignedSpatialVector Is = inertia[L] * sa;
        const PxReal stIs = sa.innerProduct(Is) + armature[size_t(L) * MAX_DOFS + 0];
        std::printf("[ info ] lane %u one-dof stIs = %.9g\n", L, stIs);
        if (stIs > 0.f) ++nPos;
        else if (stIs == 0.f) ++nZero;
        else ++nNeg;
    }
    PXG_CHECK_MSG(nPos >= 2, "at least two one-dof lanes with stIs > 0 (got %u)", nPos);
    PXG_CHECK_MSG(nZero >= 1, "a one-dof lane with stIs exactly zero (got %u)", nZero);
    PXG_CHECK_MSG(nNeg >= 1, "a one-dof lane with stIs negative (got %u)", nNeg);

    // lane 6's inverse must actually have exact signed-zero off-diagonals, or the
    // P8.84 hazard is not being exercised at all
    {
        PxMat33 m(PxIdentity);
        for (uint32_t ind2 = 0; ind2 < 3; ++ind2)
        {
            const UnAlignedSpatialVector sa = motion[size_t(6) * MAX_DOFS + ind2];
            for (uint32_t ind = 0; ind < 3; ++ind)
                m[ind][ind2] = sa.innerProduct(inertia[6] * motion[size_t(6) * MAX_DOFS + ind]);
            m[ind2][ind2] += armature[size_t(6) * MAX_DOFS + ind2];
        }
        const PxMat33 inv = m.getInverse();
        uint32_t negZeros = 0;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t r = 0; r < 3; ++r)
                if (c != r && f2u(inv[c][r]) == 0x80000000u)
                    ++negZeros;
        std::printf("[ info ] lane 6 invStIs off-diagonal negative zeros: %u of 6, "
                    "invStIs[1][0] = %08x\n", negZeros, f2u(inv[1][0]));
        PXG_CHECK_MSG(negZeros >= 3,
                      "lane 6 must produce exact -0 off-diagonals for the P8.84 hazard (got %u)",
                      negZeros);
        // The one that matters is invStIs[1][0]: it is the FIRST coefficient the
        // `ind == 1` isID accumulates with, so a folded literal-zero seed would
        // leave isID at -0 where the engine's `(+0) + (-0)` gives +0.  Without this
        // being an exact -0 the P8.84 hazard is not reached at all and the module
        // header's measurement would be vacuous.
        PXG_CHECK_MSG(f2u(inv[1][0]) == 0x80000000u,
                      "lane 6's first spherical coefficient must be exactly -0 (is %08x)",
                      f2u(inv[1][0]));
        PXG_CHECK_MSG(motion[size_t(6) * MAX_DOFS + 0].top.x == 1.0f,
                      "lane 6's msIs[0] must have a non-zero component so -0 propagates");
    }

    // lane 10 is the accumulator-seed lane: assert the shape rather than hope for it
    {
        const UnAlignedSpatialVector Is0 = inertia[10] * motion[size_t(10) * MAX_DOFS + 0];
        uint32_t negZeroComps = 0;
        for (uint32_t k = 0; k < 3; ++k)
        {
            if (f2u((&Is0.top.x)[k]) == 0x80000000u) ++negZeroComps;
            if (f2u((&Is0.bottom.x)[k]) == 0x80000000u) ++negZeroComps;
        }
        std::printf("[ info ] lane 10 Is[0] negative-zero components: %u of 6\n", negZeroComps);
        PXG_CHECK_MSG(negZeroComps == 6,
                      "lane 10's Is[0] must be all negative zeros (got %u of 6)", negZeroComps);
        PXG_CHECK_MSG(CFG[10].dofs == 1,
                      "lane 10 must be ONE dof so isID has a single term and no wash-out");
    }

    // -----------------------------------------------------------------------
    // The block images
    // -----------------------------------------------------------------------
    std::vector<PxgArticulationBlockLinkData> lkImg(1);
    std::vector<PxgArticulationBlockDofData> dfImg(MAX_DOFS);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(PxgArticulationBlockDofData));

    for (uint32_t L = 0; L < LANES; ++L)
    {
        lkImg[0].mJointType[L] = CFG[L].jointType;
        lkImg[0].mDofs[L] = CFG[L].dofs;
        for (uint32_t d = 0; d < MAX_DOFS; ++d)
        {
            storeSpatialVector(dfImg[d].mWorldMotionMatrix, motion[size_t(L) * MAX_DOFS + d], L);
            dfImg[d].mConstraintData.mArmature[L] = armature[size_t(L) * MAX_DOFS + d];
        }
    }

    // -----------------------------------------------------------------------
    // Device buffers
    // -----------------------------------------------------------------------
    PxgDevicePtr dLk = 0, dDf = 0, dScratch = 0, dInertia = 0, dJf = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dScratch, size_t(LANES) * SCRATCH_FLOATS * sizeof(float));
    shim.memAlloc(&dInertia, size_t(LANES) * INERTIA_FLOATS * sizeof(float));
    shim.memAlloc(&dJf, jointForces.size() * sizeof(float));
    if (!dLk || !dDf || !dScratch || !dInertia || !dJf)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<float> inertiaImg(size_t(LANES) * INERTIA_FLOATS);
    for (uint32_t L = 0; L < LANES; ++L)
    {
        float* p = inertiaImg.data() + size_t(L) * INERTIA_FLOATS;
        putM33(p + 0, inertia[L].topLeft);
        putM33(p + 9, inertia[L].topRight);
        putM33(p + 18, inertia[L].bottomLeft);
    }
    shim.memcpyHtoD(dInertia, inertiaImg.data(), inertiaImg.size() * sizeof(float));
    shim.memcpyHtoD(dJf, jointForces.data(), jointForces.size() * sizeof(float));

    size_t bad = 0, badScratch = 0, signedZeros = 0;

    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool withJf = (pass == 1);

        // fresh device state, and a host copy of the very same image
        shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(PxgArticulationBlockDofData));

        std::vector<float> scratch(size_t(LANES) * SCRATCH_FLOATS, 0.f);
        for (uint32_t L = 0; L < LANES; ++L)
        {
            float* s = scratch.data() + size_t(L) * SCRATCH_FLOATS;
            putVec(s + 0, Zin[L]);
            putVec(s + 6, ZIcIn[L]);
            putVec(s + 12, ZAin[L]);
            putVec(s + 18, ZAIntIn[L]);
            // the outputs start poisoned so an unwritten slot is visible
            for (uint32_t k = 24; k < SCRATCH_FLOATS; ++k)
            {
                const uint32_t bits = 0xAB000000u | (L << 8) | k;
                std::memcpy(s + k, &bits, 4);
            }
        }
        shim.memcpyHtoD(dScratch, scratch.data(), scratch.size() * sizeof(float));

        // ---- the expectation ----
        std::vector<PxgArticulationBlockLinkData> wantLk = lkImg;
        std::vector<PxgArticulationBlockDofData> wantDf = dfImg;
        std::vector<float> wantScratch = scratch;
        for (uint32_t L = 0; L < LANES; ++L)
        {
            UnAlignedSpatialVector ZA = ZAin[L];
            UnAlignedSpatialVector ZAInt = ZAIntIn[L];
            SpatialMatrix out(PxZero);
            const PxReal* jF = withJf ? (jointForces.data() + size_t(L) * JF_PER_LANE) : NULL;
            apiMirror(wantLk[0], wantDf.data(), inertia[L], false, jF, Zin[L], ZIcIn[L], ZA, ZAInt, out, L);

            float* s = wantScratch.data() + size_t(L) * SCRATCH_FLOATS;
            putVec(s + 12, ZA);
            putVec(s + 18, ZAInt);
            putM33(s + 24, out.topLeft);
            putM33(s + 33, out.topRight);
            putM33(s + 42, out.bottomLeft);
            s[51] = float(CFG[L].dofs);
        }

        struct Params
        {
            uint64_t linkBlocks, dofBlocks, jointForces, scratch, inertia;
            uint32_t nbLanes, pad0;
        } prm = { uint64_t(dLk),     uint64_t(dDf), withJf ? uint64_t(dJf) : uint64_t(0),
                  uint64_t(dScratch), uint64_t(dInertia), LANES, 0u };
        PXG_CHECK_MSG(sizeof(prm) == 48, "push-constant blob is %zu bytes", sizeof(prm));

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiPropagateInertia");
        shim.streamSynchronize();

        std::vector<PxgArticulationBlockLinkData> gotLk(lkImg.size());
        std::vector<PxgArticulationBlockDofData> gotDf(dfImg.size());
        std::vector<float> gotScratch(scratch.size());
        shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyDtoH(gotScratch.data(), dScratch, gotScratch.size() * sizeof(float));

        // ---- blocks, byte for byte ----
        // P9.86: +0 against -0 used to be forgiven here (the driver's `x + 0.0`
        // fold).  With the SZINP stamp and the select spellings it is exact; such
        // words are still counted separately so a regression is named as one.
        auto sZero = [](uint32_t a, uint32_t b) {
            return (a == 0x00000000u && b == 0x80000000u) || (a == 0x80000000u && b == 0x00000000u);
        };
        auto cmpBytes = [&](const char* what, const void* wantP, const void* gotP, size_t bytes) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(wantP);
            const uint32_t* g = reinterpret_cast<const uint32_t*>(gotP);
            for (size_t k = 0; k < bytes / 4; ++k)
            {
                if (w[k] == g[k])
                    continue;
                if (sZero(w[k], g[k]))
                    ++signedZeros;
                if (bad < 12)
                    std::printf("MM pass %u %s word %zu  ref %08x  gpu %08x\n", pass, what, k,
                                w[k], g[k]);
                ++bad;
            }
        };
        cmpBytes("link", wantLk.data(), gotLk.data(),
                 wantLk.size() * sizeof(PxgArticulationBlockLinkData));
        cmpBytes("dof", wantDf.data(), gotDf.data(),
                 wantDf.size() * sizeof(PxgArticulationBlockDofData));

        // ---- the scratch (the CUDA's locals), bit for bit ----
        for (uint32_t L = 0; L < LANES; ++L)
        {
            for (uint32_t k = 0; k < SCRATCH_FLOATS; ++k)
            {
                const size_t x = size_t(L) * SCRATCH_FLOATS + k;
                if (bitEq(wantScratch[x], gotScratch[x]))
                    continue;
                if (sZero(f2u(wantScratch[x]), f2u(gotScratch[x])))
                    ++signedZeros;
                if (badScratch < 12)
                    std::printf("MM pass %u scratch lane %u slot %u  ref %.9g (%08x)  "
                                "gpu %.9g (%08x)\n",
                                pass, L, k, wantScratch[x], f2u(wantScratch[x]), gotScratch[x],
                                f2u(gotScratch[x]));
                ++badScratch;
            }
        }
    }

    std::printf("[ info ] +-0 mismatches: %zu (P9.86: must be 0)\n", signedZeros);
    PXG_CHECK_MSG(signedZeros == 0, "lane 10's +0 seeds must survive: %zu", signedZeros);
    PXG_CHECK_MSG(bad == 0, "block image identical to the engine-built mirror");
    PXG_CHECK_MSG(badScratch == 0, "ZA / ZAInt / the returned matrix identical, bit for bit");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dScratch);
    shim.memFree(dInertia);
    shim.memFree(dJf);
}

// ===========================================================================
// P8.88: computeArticulatedSpatialInertiaW + computeUnconstrainedSpatialInertiaLaunch1T
// ===========================================================================
//
// The leaf-to-root sweep over the function above, and the kernel around it.  The
// mirror below is `apiMirror` wrapped in the CUDA's loop — ONE transcription of
// the per-dof algebra, three users, which is the arrangement P8.85 established.
//
// The topology is deliberately not a straight chain: `loopParent` returns a
// parent that is `< linkID` but usually NOT `linkID - 1`, so a port that assumes
// the obvious chain accumulates into the wrong link.  Joint types cycle through
// all five so every branch of the per-dof switch runs inside the loop, and the
// dof cursor (`mJointOffset`) advances by each joint's own dof count rather than
// by one.

static const uint32_t J_TYPE_CYCLE[6] = { J_PRISMATIC, J_REVOLUTE, J_SPHERICAL,
                                          J_FIX, J_REVOLUTE_UNWRAPPED, J_SPHERICAL };

static uint8_t loopJointType(uint32_t k, uint32_t i) { return uint8_t(J_TYPE_CYCLE[(k + i) % 6]); }

static uint8_t loopDofs(uint32_t k, uint32_t i)
{
    const uint8_t t = loopJointType(k, i);
    if (t == J_FIX) return 0;
    if (t == J_SPHERICAL) return uint8_t(2 + ((k + i) % 2));
    return 1;
}

// < linkID always, and only equal to linkID-1 by coincidence
static uint32_t loopParent(uint32_t k, uint32_t i) { return i <= 1 ? 0u : ((i - 1u + k) % i); }

static uint32_t loopTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t n = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        n += loopDofs(k, i);
    return n;
}

static size_t gClampFires = 0;   // how often PxMax(0, .) actually changed a value

// P9.86c: the clamp's own mutant twin.  0 = the CUDA's PxMax<float>, i.e. the float
// specialisation `intrinsics::selectMax` = `a > b ? a : b` (PxMath.h:82,
// PxUnixMathIntrinsics.h:122), which for PxMax(0, x) RETURNS x = -0 as -0.
// 1 = PxMath.h's generic template `a < b ? b : a`, the port's pre-P9.86 spelling
// (the revert r7), which turns x = -0 into +0.  Only the directed test below sets it.
static int gClampVariant = 0;
// clamp input exactly -0 over a parent diagonal of exactly -0: the only shape in
// which the clamp's sign of zero reaches the stored `parent + W`
static size_t gClampNegZeroSeen = 0;

// ---------------------------------------------------------------------------
// The loop mirror: the CUDA's control flow around apiMirror
// ---------------------------------------------------------------------------
static void apiLoopMirror(PxgArticulationBlockLinkData* lk, PxgArticulationBlockDofData* df,
                          PxgArticulationBlockData& ab, const PxReal* jointForces,
                          bool extEveryIter, uint32_t numLinks, uint32_t lane)
{
    for (PxU32 linkID = numLinks - 1; linkID > 0; --linkID)
    {
        PxgArticulationBlockLinkData& blk = lk[linkID];
        const PxU32 parent = blk.mParents[lane];
        const PxU32 dof = blk.mDofs[lane];
        const PxU32 jointOffset = blk.mJointOffset[lane];
        const PxVec3 rw(blk.mRw_x[lane], blk.mRw_y[lane], blk.mRw_z[lane]);
        PxgArticulationBlockDofData* dofData = df + jointOffset;

        // every parent read happens before anything is written
        SpatialMatrix parentInertia;
        loadSpatialMatrix(lk[parent].mSpatialArticulatedInertia, lane, parentInertia);
        SpatialMatrix articulatedInertia;
        loadSpatialMatrix(blk.mSpatialArticulatedInertia, lane, articulatedInertia);
        const UnAlignedSpatialVector coriolis = loadSpatialVector(blk.mCoriolis, lane);
        const UnAlignedSpatialVector spatialZA = loadSpatialVector(blk.mZAVector, lane);
        const UnAlignedSpatialVector spatialZAInt = loadSpatialVector(blk.mZAIntVector, lane);
        const UnAlignedSpatialVector parentZA = loadSpatialVector(lk[parent].mZAVector, lane);
        const UnAlignedSpatialVector parentZAInt = loadSpatialVector(lk[parent].mZAIntVector, lane);

        const UnAlignedSpatialVector Ic = articulatedInertia * coriolis;
        const UnAlignedSpatialVector ZIcInt = spatialZAInt + Ic;
        UnAlignedSpatialVector translatedZA = spatialZA;
        UnAlignedSpatialVector translatedZAInt = ZIcInt;

        // 5.12: jF is always the record's slice; extEveryIter gates its use inside
        const PxReal* jF = jointForces + jointOffset;

        SpatialMatrix sub(PxZero);
        apiMirror(blk, dofData, articulatedInertia, extEveryIter, jF, spatialZA, ZIcInt, translatedZA,
                  translatedZAInt, sub, lane);
        SpatialMatrix spatialInertiaW = dySmSub(articulatedInertia, sub);

        refTranslateSpatialVectorInPlace(rw, translatedZA);
        refTranslateSpatialVectorInPlace(rw, translatedZAInt);
        storeSpatialVector(lk[parent].mZAVector, parentZA + translatedZA, lane);
        storeSpatialVector(lk[parent].mZAIntVector, parentZAInt + translatedZAInt, lane);

        // the CUDA re-loads the two vectors here and never uses them -- dead code,
        // dropped in the port and in the mirror alike

        refTranslateInertia(refSkewSymmetric(rw.x, rw.y, rw.z), spatialInertiaW);

        const float parentDiag[3] = { parentInertia.bottomLeft.column0.x,
                                      parentInertia.bottomLeft.column1.y,
                                      parentInertia.bottomLeft.column2.z };
        float* diag[3] = { &spatialInertiaW.bottomLeft.column0.x,
                           &spatialInertiaW.bottomLeft.column1.y,
                           &spatialInertiaW.bottomLeft.column2.z };
        for (uint32_t c = 0; c < 3; ++c)
        {
            const float before = *diag[c];
            // PxMax<float> is selectMax -- the CUDA semantics; the twin is the revert
            *diag[c] = gClampVariant ? ((0.f < before) ? before : 0.f) : PxMax(0.f, before);
            if (f2u(*diag[c]) != f2u(before))
                ++gClampFires;
            if (f2u(before) == 0x80000000u && f2u(parentDiag[c]) == 0x80000000u)
                ++gClampNegZeroSeen;
        }

        storeSpatialMatrix(lk[parent].mSpatialArticulatedInertia, lane,
                           dySmAdd(parentInertia, spatialInertiaW));
    }

    SpatialMatrix inertia;
    if (ab.mFlags[lane] & PxArticulationFlag::eFIX_BASE)
        inertia.setZero();
    else
    {
        loadSpatialMatrix(lk[0].mSpatialArticulatedInertia, lane, inertia);
        inertia = inertia.invertInertia();
    }
    storeSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, inertia);
}

// Fills one articulation's slice of a link/dof block run.  Shared by both P8.88
// tests so the topology cannot drift between them.
static void loopFillBlocks(PxgArticulationBlockLinkData* lk, PxgArticulationBlockDofData* df,
                           PxgArticulationBlockData& ab, uint32_t k, uint32_t numLinks,
                           bool fixBase, uint32_t lane)
{
    auto v = [&](uint32_t i, uint32_t f) {
        return (float(k % 17) * 5.0f + float(i) * 2.0f + float(f) + 1.0f) * 0.0625f - 2.0f;
    };

    ab.mNumLinks[lane] = numLinks;
    ab.mFlags[lane] = fixBase ? PxU32(PxArticulationFlag::eFIX_BASE) : 0u;

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        PxgArticulationBlockLinkData& b = lk[i];
        const uint8_t dofs = (i == 0) ? 0 : loopDofs(k, i);
        b.mParents[lane] = loopParent(k, i);
        b.mDofs[lane] = dofs;
        b.mJointType[lane] = (i == 0) ? uint8_t(J_FIX) : loopJointType(k, i);
        b.mJointOffset[lane] = cursor;
        b.mRw_x[lane] = v(i, 0);
        b.mRw_y[lane] = v(i, 1);
        b.mRw_z[lane] = v(i, 2);

        // a dense, non-symmetric articulated inertia, and the three spatial vectors
        SpatialMatrix m(PxMat33(PxVec3(v(i, 3), v(i, 4), v(i, 5)),
                                PxVec3(v(i, 6), v(i, 7), v(i, 8)),
                                PxVec3(v(i, 9), v(i, 10), v(i, 11))),
                        PxMat33(PxVec3(v(i, 12), v(i, 13), v(i, 14)),
                                PxVec3(v(i, 15), v(i, 16), v(i, 17)),
                                PxVec3(v(i, 18), v(i, 19), v(i, 20))),
                        PxMat33(PxVec3(v(i, 21), v(i, 22), v(i, 23)),
                                PxVec3(v(i, 24), v(i, 25), v(i, 26)),
                                PxVec3(v(i, 27), v(i, 28), v(i, 29))));
        storeSpatialMatrix(b.mSpatialArticulatedInertia, lane, m);
        storeSpatialVector(b.mCoriolis,
                           UnAlignedSpatialVector(PxVec3(v(i, 30), v(i, 31), v(i, 32)),
                                                  PxVec3(v(i, 33), v(i, 34), v(i, 35))), lane);
        storeSpatialVector(b.mZAVector,
                           UnAlignedSpatialVector(PxVec3(v(i, 36), v(i, 37), v(i, 38)),
                                                  PxVec3(v(i, 39), v(i, 40), v(i, 41))), lane);
        storeSpatialVector(b.mZAIntVector,
                           UnAlignedSpatialVector(PxVec3(v(i, 42), v(i, 43), v(i, 44)),
                                                  PxVec3(v(i, 45), v(i, 46), v(i, 47))), lane);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            PxgArticulationBlockDofData& d = df[cursor + dd];
            storeSpatialVector(d.mWorldMotionMatrix,
                               UnAlignedSpatialVector(
                                   PxVec3(v(i, 50 + dd * 6 + 0), v(i, 50 + dd * 6 + 1),
                                          v(i, 50 + dd * 6 + 2)),
                                   PxVec3(v(i, 50 + dd * 6 + 3), v(i, 50 + dd * 6 + 4),
                                          v(i, 50 + dd * 6 + 5))),
                               lane);
            d.mConstraintData.mArmature[lane] = 0.25f + 0.125f * float((k + i + dd) % 5);
        }
        cursor += dofs;
    }
}

static const uint32_t LOOP_LANES = 6;
static const uint32_t LMAX_LINKS = 7;    // > every per-lane link count
static const uint32_t LMAX_DOFS = 11;
static const uint32_t LOOP_JF_PER_LANE = 8;

PXG_TEST(artiArticulatedInertia)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64 x4, u32 x4 } -> 48 bytes
        { "npArtiPropagateInertia.spv", "artiArticulatedInertia", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiPropagateLoader);

    // link counts 1..6, all distinct and none equal to LMAX_LINKS; the base is
    // fixed on half of them so both root branches run
    const uint32_t NUM_LINKS[LOOP_LANES] = { 1, 2, 3, 4, 5, 6 };
    const bool FIX_BASE[LOOP_LANES] = { true, false, true, false, true, false };

    uint32_t seenType[6] = { 0, 0, 0, 0, 0, 0 };
    for (uint32_t L = 0; L < LOOP_LANES; ++L)
    {
        PXG_CHECK_MSG(loopTotalDofs(L, NUM_LINKS[L]) <= LMAX_DOFS,
                      "lane %u needs %u dofs, LMAX_DOFS is %u", L, loopTotalDofs(L, NUM_LINKS[L]),
                      LMAX_DOFS);
        for (uint32_t i = 1; i < NUM_LINKS[L]; ++i)
            seenType[loopJointType(L, i)] += 1u;
    }
    std::printf("[ info ] loop joint types seen: fix %u pris %u rev %u revU %u sph %u\n",
                seenType[J_FIX], seenType[J_PRISMATIC], seenType[J_REVOLUTE],
                seenType[J_REVOLUTE_UNWRAPPED], seenType[J_SPHERICAL]);
    for (uint32_t t = 0; t < 5; ++t)
        PXG_CHECK_MSG(seenType[t] > 0, "joint type %u never occurs in the loop fixture", t);

    std::vector<PxgArticulationBlockLinkData> lkImg(LMAX_LINKS);
    std::vector<PxgArticulationBlockDofData> dfImg(LMAX_DOFS);
    std::vector<PxgArticulationBlockData> abImg(1);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(PxgArticulationBlockData));
    for (uint32_t L = 0; L < LOOP_LANES; ++L)
        loopFillBlocks(lkImg.data(), dfImg.data(), abImg[0], L, NUM_LINKS[L], FIX_BASE[L], L);

    std::vector<float> jointForces(size_t(LOOP_LANES) * LOOP_JF_PER_LANE);
    for (size_t j = 0; j < jointForces.size(); ++j)
        jointForces[j] = 0.375f * float(j % 11) - 1.25f;

    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0, dJf = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(PxgArticulationBlockData));
    shim.memAlloc(&dJf, jointForces.size() * sizeof(float));
    if (!dLk || !dDf || !dAb || !dJf)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dJf, jointForces.data(), jointForces.size() * sizeof(float));

    size_t bad = 0, signedZeros = 0;
    gClampFires = 0;

    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool extEveryIter = (pass == 1);

        shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(PxgArticulationBlockData));

        std::vector<PxgArticulationBlockLinkData> wantLk = lkImg;
        std::vector<PxgArticulationBlockDofData> wantDf = dfImg;
        std::vector<PxgArticulationBlockData> wantAb = abImg;
        for (uint32_t L = 0; L < LOOP_LANES; ++L)
            apiLoopMirror(wantLk.data(), wantDf.data(), wantAb[0],
                          jointForces.data() + size_t(L) * LOOP_JF_PER_LANE, extEveryIter,
                          NUM_LINKS[L], L);

        struct Params
        {
            uint64_t linkBlocks, dofBlocks, artiBlock, jointForces;
            uint32_t nbLanes, extEveryIter, pad0, pad1;
        } prm = { uint64_t(dLk), uint64_t(dDf),  uint64_t(dAb),          uint64_t(dJf),
                  LOOP_LANES,    extEveryIter ? 1u : 0u, 0u, 0u };
        PXG_CHECK_MSG(sizeof(prm) == 48, "push-constant blob is %zu bytes", sizeof(prm));

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiArticulatedInertia");
        shim.streamSynchronize();

        std::vector<PxgArticulationBlockLinkData> gotLk(lkImg.size());
        std::vector<PxgArticulationBlockDofData> gotDf(dfImg.size());
        std::vector<PxgArticulationBlockData> gotAb(abImg.size());
        shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(PxgArticulationBlockData));

        auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
            const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
            for (size_t x = 0; x < bytes / 4; ++x)
            {
                if (w[x] == g[x])
                    continue;
                if ((w[x] == 0u && g[x] == 0x80000000u) || (w[x] == 0x80000000u && g[x] == 0u))
                    ++signedZeros;   // P9.86: counted, and a mismatch
                if (bad < 12)
                    std::printf("MM pass %u %s word %zu  ref %08x  gpu %08x\n", pass, what, x,
                                w[x], g[x]);
                ++bad;
            }
        };
        cmp("link", wantLk.data(), gotLk.data(),
            wantLk.size() * sizeof(PxgArticulationBlockLinkData));
        cmp("dof", wantDf.data(), gotDf.data(), wantDf.size() * sizeof(PxgArticulationBlockDofData));
        cmp("arti", wantAb.data(), gotAb.data(), wantAb.size() * sizeof(PxgArticulationBlockData));
    }

    std::printf("[ info ] loop: clamp fired %zu times, +-0 mismatches %zu\n", gClampFires,
                signedZeros);
    PXG_CHECK_MSG(signedZeros == 0, "+-0 mismatches: %zu", signedZeros);
    PXG_CHECK_MSG(gClampFires > 0,
                  "PxMax(0, .) must actually clamp something, or the guard is untested");
    PXG_CHECK_MSG(bad == 0, "the leaf-to-root sweep matches the engine-built mirror");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
    shim.memFree(dJf);
}

// ===========================================================================
// P9.86c: the clamp keeps the CUDA's -0 (PxMax<float> is selectMax)
// ===========================================================================
//
// forwardDynamic2.cu:1641-1649 clamps the three translated bottomLeft diagonals with
// `PxMax(minPropagatedInertia = 0.f, x)` and stores `parent + W`.  On a float that is
// the specialisation selectMax, `0 > x ? 0 : x`, which returns x = -0 AS -0.  The
// port's pre-P9.86 spelling (PxMath.h's generic `a < b ? b : a`, revert r7) returns
// +0 there.  The store `p + W` absorbs that for every p except p = -0:
// (-0) + (-0) = -0 against (-0) + (+0) = +0.  The loop fixtures above never make
// either value, so r7 survived them.
//
// ONE directed lane builds both, all finite:
//   * a fixed-base 2-link articulation whose child is eFIX (no dofs, the propagate
//     term is SpatialMatrix(PxZero), so W is the child's own inertia) at a ZERO
//     offset rw = (+0, +0, +0) -- +0, because -0 components flip the skew's signs;
//   * the skew of +0 has rows (+0,-0,+0), (+0,+0,-0), (-0,+0,+0), so every product
//     sTod[j][k] * TL[k][j] of the translated diagonal is a signed zero.  The child
//     topLeft is sign-patterned so that ALL of them are -0; `br = sTod*TR + TL^T` is
//     then exactly TL^T and (br * dTos)[j][j] is the same three -0 products.  With a
//     -0 child bottomLeft diagonal every term is -0 and the pre-clamp diagonal is
//     -0 (and stays -0 if any `0 * x` were folded, since every remaining term is -0);
//   * the parent's bottomLeft diagonal is -0 as well.
// Every other word of the lane is a nonzero sum (checked by the twin: exactly the
// three parent diagonal words move).
//
// THE DATA IS DELIBERATELY NOT A PHYSICAL (PSD) INERTIA, and it cannot be: an
// engine-seeded inertia (transformInertiaTensor's (d*M)*M terms) never has a -0
// diagonal, and a point mass (TL = 0, TR = m*I) makes br all +0 and the translated
// diagonal +0.  The kernel accepts any finite articulated inertia, and the loop
// fixtures above already feed dense, non-symmetric, sign-mixed ones -- the clamp
// exists precisely because the propagated diagonal is not guaranteed positive.  A
// fixed base keeps invertInertia off this degenerate root.
PXG_TEST(artiArticulatedInertiaClampNegZero)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiPropagateInertia.spv", "artiArticulatedInertia", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiPropagateLoader);

    std::vector<PxgArticulationBlockLinkData> lkImg(LMAX_LINKS);
    std::vector<PxgArticulationBlockDofData> dfImg(LMAX_DOFS);
    std::vector<PxgArticulationBlockData> abImg(1);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(PxgArticulationBlockData));

    const uint32_t lane = 0;
    const float nz = -0.0f;
    abImg[0].mNumLinks[lane] = 2;
    abImg[0].mFlags[lane] = PxU32(PxArticulationFlag::eFIX_BASE);

    // PxMat33(col0, col1, col2)
    const SpatialMatrix parentM(
        PxMat33(PxVec3(0.5f, -0.25f, 0.125f), PxVec3(0.375f, 0.625f, -0.75f),
                PxVec3(-0.875f, 0.25f, 1.0f)),
        PxMat33(PxVec3(2.0f, 0.125f, -0.25f), PxVec3(0.375f, 2.0f, 0.5f),
                PxVec3(-0.625f, 0.75f, 2.0f)),
        PxMat33(PxVec3(nz, 1.25f, -0.5f), PxVec3(1.25f, nz, 0.75f), PxVec3(-0.5f, 0.75f, nz)));
    // child topLeft sign pattern: col0 (-,+,-), col1 (-,-,+), col2 (+,-,-) -- the
    // opposite of the +0 skew's sign in every product of the translated diagonal
    const SpatialMatrix childM(
        PxMat33(PxVec3(-1.5f, 0.75f, -1.25f), PxVec3(-0.5f, -1.75f, 0.625f),
                PxVec3(1.125f, -0.875f, -2.5f)),
        PxMat33(PxVec3(1.5f, 0.25f, -0.375f), PxVec3(0.125f, 1.5f, 0.5f),
                PxVec3(-0.25f, 0.375f, 1.5f)),
        PxMat33(PxVec3(nz, 0.25f, -0.125f), PxVec3(0.25f, nz, 0.375f),
                PxVec3(-0.125f, 0.375f, nz)));

    for (uint32_t i = 0; i < 2; ++i)
    {
        PxgArticulationBlockLinkData& b = lkImg[i];
        const float o = 0.5f * float(i);
        b.mParents[lane] = 0;
        b.mDofs[lane] = 0;
        b.mJointType[lane] = uint8_t(J_FIX);
        b.mJointOffset[lane] = 0;
        b.mRw_x[lane] = 0.0f;   // +0: a zero offset
        b.mRw_y[lane] = 0.0f;
        b.mRw_z[lane] = 0.0f;
        storeSpatialMatrix(b.mSpatialArticulatedInertia, lane, i == 0 ? parentM : childM);
        storeSpatialVector(b.mCoriolis,
                           UnAlignedSpatialVector(PxVec3(0.5f + o, -0.25f, 0.75f),
                                                  PxVec3(-1.0f, 0.375f + o, 0.625f)), lane);
        storeSpatialVector(b.mZAVector,
                           UnAlignedSpatialVector(PxVec3(0.25f, 0.5f + o, -0.75f),
                                                  PxVec3(1.25f, -1.5f, 0.875f + o)), lane);
        storeSpatialVector(b.mZAIntVector,
                           UnAlignedSpatialVector(PxVec3(-0.375f + o, 0.125f, 1.0f),
                                                  PxVec3(0.5f, -0.625f, 0.25f + o)), lane);
    }

    // eFIX reads no joint force; the slice is there so the pointer arithmetic is real
    std::vector<float> jointForces(LOOP_JF_PER_LANE, 0.75f);

    // ---- the mirror (CUDA semantics) and its twin (the revert) ----
    std::vector<PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<PxgArticulationBlockDofData> wantDf = dfImg;
    std::vector<PxgArticulationBlockData> wantAb = abImg;
    gClampNegZeroSeen = 0;
    gClampVariant = 0;
    apiLoopMirror(wantLk.data(), wantDf.data(), wantAb[0], jointForces.data(), false, 2, lane);
    const size_t seen = gClampNegZeroSeen;

    std::vector<PxgArticulationBlockLinkData> twinLk = lkImg;
    std::vector<PxgArticulationBlockDofData> twinDf = dfImg;
    std::vector<PxgArticulationBlockData> twinAb = abImg;
    gClampVariant = 1;
    apiLoopMirror(twinLk.data(), twinDf.data(), twinAb[0], jointForces.data(), false, 2, lane);
    gClampVariant = 0;

    size_t twinDiff = 0, twinFlip = 0;
    {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wantLk.data());
        const uint32_t* t = reinterpret_cast<const uint32_t*>(twinLk.data());
        for (size_t x = 0; x < wantLk.size() * sizeof(PxgArticulationBlockLinkData) / 4; ++x)
        {
            if (w[x] == t[x])
                continue;
            ++twinDiff;
            if (w[x] == 0x80000000u && t[x] == 0u)
                ++twinFlip;
        }
    }
    std::printf("[ info ] clamp -0: seen %zu, twin differs in %zu link words (%zu are -0 -> +0)\n",
                seen, twinDiff, twinFlip);
    PXG_CHECK_MSG(seen == 3,
                  "all three child pre-clamp diagonals must be -0 over a -0 parent diagonal "
                  "(got %zu)", seen);
    PXG_CHECK_MSG(twinDiff == 3 && twinFlip == 3,
                  "the revert twin must move exactly the parent's three bottomLeft diagonal "
                  "words from -0 to +0 (differs %zu, flips %zu)", twinDiff, twinFlip);
    PXG_CHECK_MSG(std::memcmp(wantDf.data(), twinDf.data(),
                              wantDf.size() * sizeof(PxgArticulationBlockDofData)) == 0 &&
                      std::memcmp(wantAb.data(), twinAb.data(),
                                  wantAb.size() * sizeof(PxgArticulationBlockData)) == 0,
                  "the clamp twin touches only the parent link's inertia");

    // ---- the GPU ----
    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0, dJf = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(PxgArticulationBlockData));
    shim.memAlloc(&dJf, jointForces.size() * sizeof(float));
    if (!dLk || !dDf || !dAb || !dJf)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dJf, jointForces.data(), jointForces.size() * sizeof(float));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(PxgArticulationBlockDofData));
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(PxgArticulationBlockData));

    struct Params
    {
        uint64_t linkBlocks, dofBlocks, artiBlock, jointForces;
        uint32_t nbLanes, extEveryIter, pad0, pad1;
    } prm = { uint64_t(dLk), uint64_t(dDf), uint64_t(dAb), uint64_t(dJf), 1u, 0u, 0u, 0u };
    PXG_CHECK_MSG(sizeof(prm) == 48, "push-constant blob is %zu bytes", sizeof(prm));
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiArticulatedInertia");
    shim.streamSynchronize();

    std::vector<PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<PxgArticulationBlockData> gotAb(abImg.size());
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(PxgArticulationBlockData));

    size_t bad = 0, signedZeros = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t x = 0; x < bytes / 4; ++x)
        {
            if (w[x] == g[x])
                continue;
            if ((w[x] == 0u && g[x] == 0x80000000u) || (w[x] == 0x80000000u && g[x] == 0u))
                ++signedZeros;   // counted for the message, never forgiven
            if (bad < 12)
                std::printf("MM %s word %zu  ref %08x  gpu %08x\n", what, x, w[x], g[x]);
            ++bad;
        }
    };
    cmp("link", wantLk.data(), gotLk.data(), wantLk.size() * sizeof(PxgArticulationBlockLinkData));
    cmp("dof", wantDf.data(), gotDf.data(), wantDf.size() * sizeof(PxgArticulationBlockDofData));
    cmp("arti", wantAb.data(), gotAb.data(), wantAb.size() * sizeof(PxgArticulationBlockData));

    std::printf("[ info ] clamp -0: +-0 mismatches %zu, mismatches %zu\n", signedZeros, bad);
    PXG_CHECK_MSG(bad == 0, "the clamp keeps the CUDA's -0 (selectMax), bit for bit");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
    shim.memFree(dJf);
}

// ===========================================================================
// P8.88b: computeUnconstrainedSpatialInertiaLaunch1T -- the kernel
// ===========================================================================
//
// The index chain of P8.85's wrapper with three differences, each a mutant:
//
//  1. **Block (32, 1, 1) and grid `num1TBlocks * 2`** — one warp per block, twice
//     the blocks.  The fixture uses 130 articulations, so `num1TBlocks` is 3 and
//     the grid is 6; a port that reuses the partial kernel's warps-per-block of 2
//     visits warp slices 0, 2, 4 only and leaves 1 and 3 poisoned.
//  2. **`numLinks` comes from the BLOCK**, not the record — the opposite of P8.85.
//     The records here carry a *different* link count so the two disagree.
//  3. It hands the loop `articulation.jointForce`, the record pointer, which the
//     loop then offsets per link by that link's own `jointOffset`.

static const uint32_t WNB_ARTI = 130;
static const uint32_t WWARPS_ALLOC = 7;   // five carry work, two stay poison
static const uint32_t WMAX_LINKS = 8;     // > every per-articulation link count
static const uint32_t WMAX_DOFS = 12;
static const uint32_t WJF_SLOTS = 16;     // per articulation, > any total dof count

PXG_TEST(artiArticulatedInertiaLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // engine ABI { 8 } -- the descriptor pointer alone
        { "npArtiPropagateInertia.spv", "computeUnconstrainedSpatialInertiaLaunch1T", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiPropagateLoader);

    // link counts 1..6, never equal to the lane index and never WMAX_LINKS
    auto numLinksOf = [](uint32_t k) { return 1u + ((k * 5u) % 6u); };
    auto fixBaseOf = [](uint32_t k) { return (k % 3u) == 1u; };

    // ---- the index chain: three hops, none of them the identity ----
    const uint32_t ARTI_OFFSET = 5;
    const uint32_t ISLAND_SLOTS = ARTI_OFFSET + WNB_ARTI + 3;
    const uint32_t BODY_SIM_SLOTS = 192;
    std::vector<uint32_t> NODE(WNB_ARTI), POOL(WNB_ARTI);
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
    {
        NODE[k] = (k * 7u + 11u) % BODY_SIM_SLOTS;
        POOL[k] = (k + 13u) % WNB_ARTI;
    }

    std::vector<PxNodeIndex> island(ISLAND_SLOTS);
    for (uint32_t i = 0; i < ISLAND_SLOTS; ++i)
        island[i] = PxNodeIndex(0x0BAD00u + i);
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
        island[ARTI_OFFSET + k] = PxNodeIndex(NODE[k]);

    std::vector<PxgBodySim> bodySims(BODY_SIM_SLOTS);
    std::memset(bodySims.data(), 0xEE, bodySims.size() * sizeof(PxgBodySim));
    for (uint32_t i = 0; i < BODY_SIM_SLOTS; ++i)
        bodySims[i].articulationRemapId = 0xDEAD0000u + i;
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
        bodySims[NODE[k]].articulationRemapId = POOL[k];

    // ---- the block images, one warp slice per 32 articulations ----
    std::vector<PxgArticulationBlockLinkData> lkImg(size_t(WWARPS_ALLOC) * WMAX_LINKS);
    std::vector<PxgArticulationBlockDofData> dfImg(size_t(WWARPS_ALLOC) * WMAX_DOFS);
    std::vector<PxgArticulationBlockData> abImg(WWARPS_ALLOC);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(PxgArticulationBlockData));
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
    {
        const uint32_t warp = k / 32u, lane = k % 32u;
        PXG_CHECK_MSG(loopTotalDofs(k, numLinksOf(k)) <= WMAX_DOFS,
                      "articulation %u needs %u dofs", k, loopTotalDofs(k, numLinksOf(k)));
        loopFillBlocks(lkImg.data() + size_t(warp) * WMAX_LINKS,
                       dfImg.data() + size_t(warp) * WMAX_DOFS, abImg[warp], k, numLinksOf(k),
                       fixBaseOf(k), lane);
    }

    std::vector<float> jointForces(size_t(WNB_ARTI) * WJF_SLOTS);
    for (size_t j = 0; j < jointForces.size(); ++j)
        jointForces[j] = 0.375f * float(j % 11) - 1.25f;

    PxgDevicePtr dIsland = 0, dBodySims = 0, dDesc = 0, dArtis = 0, dJf = 0;
    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0;
    shim.memAlloc(&dIsland, island.size() * sizeof(PxNodeIndex));
    shim.memAlloc(&dBodySims, bodySims.size() * sizeof(PxgBodySim));
    shim.memAlloc(&dDesc, sizeof(PxgArticulationCoreDesc));
    shim.memAlloc(&dArtis, size_t(WNB_ARTI) * sizeof(PxgArticulation));
    shim.memAlloc(&dJf, jointForces.size() * sizeof(float));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(PxgArticulationBlockData));
    if (!dIsland || !dBodySims || !dDesc || !dArtis || !dJf || !dLk || !dDf || !dAb)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // The records carry a DIFFERENT link count from the blocks, so a port that
    // reads the record (P8.85's rule) rather than the block is wrong immediately.
    std::vector<PxgArticulation> recs(WNB_ARTI);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t k = 0; k < WNB_ARTI; ++k)
    {
        PxgArticulation& r = recs[POOL[k]];
        r.data.numLinks = PxU16(0xFEE0u + k);   // never a plausible link count
        r.jointForce = reinterpret_cast<PxReal*>(
            dJf + PxgDevicePtr(size_t(k) * WJF_SLOTS * sizeof(float)));
    }
    shim.memcpyHtoD(dIsland, island.data(), island.size() * sizeof(PxNodeIndex));
    shim.memcpyHtoD(dBodySims, bodySims.data(), bodySims.size() * sizeof(PxgBodySim));
    shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));
    shim.memcpyHtoD(dJf, jointForces.data(), jointForces.size() * sizeof(float));

    size_t bad = 0, signedZeros = 0;

    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool extEveryIter = (pass == 1);

        shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(PxgArticulationBlockData));

        // 0x5A everywhere first, so the three padding bytes after the one-byte
        // bool at 68 are NOT zero -- a word read there is then caught rather than
        // merely lucky.
        PxgArticulationCoreDesc desc;
        std::memset(&desc, 0x5A, sizeof(desc));
        desc.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dBodySims);
        desc.islandNodeIndices = reinterpret_cast<PxNodeIndex*>(dIsland);
        desc.articulations = reinterpret_cast<PxgArticulation*>(dArtis);
        desc.articulationOffset = ARTI_OFFSET;
        desc.nbArticulations = WNB_ARTI;
        desc.isExternalForcesEveryTgsIterationEnabled = extEveryIter;
        desc.mArticulationBlocks = reinterpret_cast<PxgArticulationBlockData*>(dAb);
        desc.mArticulationLinkBlocks = reinterpret_cast<PxgArticulationBlockLinkData*>(dLk);
        desc.mArticulationDofBlocks = reinterpret_cast<PxgArticulationBlockDofData*>(dDf);
        desc.mMaxLinksPerArticulation = WMAX_LINKS;
        desc.mMaxDofsPerArticulation = WMAX_DOFS;
        shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

        std::vector<PxgArticulationBlockLinkData> wantLk = lkImg;
        std::vector<PxgArticulationBlockDofData> wantDf = dfImg;
        std::vector<PxgArticulationBlockData> wantAb = abImg;
        for (uint32_t k = 0; k < WNB_ARTI; ++k)
        {
            const uint32_t warp = k / 32u, lane = k % 32u;
            apiLoopMirror(wantLk.data() + size_t(warp) * WMAX_LINKS,
                          wantDf.data() + size_t(warp) * WMAX_DOFS, wantAb[warp],
                          jointForces.data() + size_t(k) * WJF_SLOTS, extEveryIter,
                          numLinksOf(k), lane);
        }

        struct Params { uint64_t desc; } prm = { uint64_t(dDesc) };
        const uint32_t num1TBlocks = (WNB_ARTI + 63u) / 64u;
        PXG_CHECK_MSG(num1TBlocks == 3u, "fixture assumes three 1T blocks, got %u", num1TBlocks);
        // block (32, 1, 1) and TWICE the blocks -- not the partial kernel's shape
        PXG_CHECK_MSG(shim.launchKernel(0, num1TBlocks * 2u, 1, 1, &prm, sizeof(prm))
                          == PxgResult::Success,
                      "launch computeUnconstrainedSpatialInertiaLaunch1T");
        shim.streamSynchronize();

        std::vector<PxgArticulationBlockLinkData> gotLk(lkImg.size());
        std::vector<PxgArticulationBlockDofData> gotDf(dfImg.size());
        std::vector<PxgArticulationBlockData> gotAb(abImg.size());
        shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(PxgArticulationBlockLinkData));
        shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(PxgArticulationBlockDofData));
        shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(PxgArticulationBlockData));

        auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
            const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
            for (size_t x = 0; x < bytes / 4; ++x)
            {
                if (w[x] == g[x])
                    continue;
                if ((w[x] == 0u && g[x] == 0x80000000u) || (w[x] == 0x80000000u && g[x] == 0u))
                    ++signedZeros;   // P9.86: counted, and a mismatch
                if (bad < 12)
                    std::printf("MM pass %u %s word %zu  ref %08x  gpu %08x\n", pass, what, x,
                                w[x], g[x]);
                ++bad;
            }
        };
        cmp("link", wantLk.data(), gotLk.data(),
            wantLk.size() * sizeof(PxgArticulationBlockLinkData));
        cmp("dof", wantDf.data(), gotDf.data(), wantDf.size() * sizeof(PxgArticulationBlockDofData));
        cmp("arti", wantAb.data(), gotAb.data(), wantAb.size() * sizeof(PxgArticulationBlockData));
    }

    std::printf("[ info ] launch: +-0 mismatches %zu\n", signedZeros);
    PXG_CHECK_MSG(signedZeros == 0, "+-0 mismatches: %zu", signedZeros);
    PXG_CHECK_MSG(bad == 0, "the wrapper's whole block image matches the mirror");

    shim.memFree(dIsland);
    shim.memFree(dBodySims);
    shim.memFree(dDesc);
    shim.memFree(dArtis);
    shim.memFree(dJf);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
}

}  // namespace
