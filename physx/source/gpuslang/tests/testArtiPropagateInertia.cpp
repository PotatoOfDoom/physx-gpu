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
                      const SpatialMatrix& spatialInertia, const PxReal* jF,
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
        const PxReal qstZIcInternal = (jF ? jF[0] : 0.0f) - stZInt;
        df[0].mQstZ[lane] = qstZ;
        df[0].mQstZIcInternal[lane] = qstZIcInternal;

        ZA += isInvD * qstZ;
        ZAInt += isInvD * qstZIcInternal;

        out = SpatialMatrix::constructSpatialMatrix(SpatialVectorF(isInvD.top, isInvD.bottom),
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
                const PxReal qstZicInt = (jF ? jF[ind2] : 0.0f) - stZInt;
                qstZG[ind2] = qstZ;
                qstZIcIntG[ind2] = qstZicInt;
                df[ind2].mQstZ[lane] = qstZ;
                df[ind2].mQstZIcInternal[lane] = qstZicInt;
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

        out = SpatialMatrix::constructSpatialMatrix(columns);
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
            apiMirror(wantLk[0], wantDf.data(), inertia[L], jF, Zin[L], ZIcIn[L], ZA, ZAInt, out, L);

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
        // The ONLY tolerated difference, and it is scoped to one bit of one value:
        // a word that is +0 on one side and -0 on the other.  See the module
        // header — the driver folds `x + 0.0` below SPIR-V, which no shader-level
        // workaround defeats (measured: seeding the accumulators from
        // `abs(armature) * 0.0f` changes nothing).  The count is printed so it
        // cannot drift upwards unnoticed.
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
                {
                    ++signedZeros;
                    continue;
                }
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
                {
                    ++signedZeros;
                    continue;
                }
                if (badScratch < 12)
                    std::printf("MM pass %u scratch lane %u slot %u  ref %.9g (%08x)  "
                                "gpu %.9g (%08x)\n",
                                pass, L, k, wantScratch[x], f2u(wantScratch[x]), gotScratch[x],
                                f2u(gotScratch[x]));
                ++badScratch;
            }
        }
    }

    std::printf("[ info ] signed-zero words forgiven: %zu (the driver's `x + 0.0` fold)\n",
                signedZeros);
    PXG_CHECK_MSG(signedZeros > 0,
                  "lane 10 must actually exercise the signed-zero fold (got %zu)", signedZeros);
    PXG_CHECK_MSG(signedZeros < 40, "the signed-zero count has drifted: %zu", signedZeros);
    PXG_CHECK_MSG(bad == 0, "block image identical to the engine-built mirror");
    PXG_CHECK_MSG(badScratch == 0, "ZA / ZAInt / the returned matrix identical, bit for bit");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dScratch);
    shim.memFree(dInertia);
    shim.memFree(dJf);
}

}  // namespace
