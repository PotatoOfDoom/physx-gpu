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

// testArtiSolveStatic.cpp -- P8.100a: siSolveStaticConstraintsTGS, the static
// constraint solve of artiSolveInternalConstraintsTGS1T
// (solveStaticConstraints TGS overload, internalConstraints2.cu:1395-1552).
//
// Mirror: the two batch loops transcribed over the P8.99a ext-solve mirrors
// (pxgArtiExtSolveRef.h).  Fixture facts:
//  * per-lane batch WINDOWS (start/count from the link block) into shared
//    global pools -- overlapping windows are the engine's normal state, and
//    per-lane writes land at the lane's exclusive-popcount offset;
//  * the articulated link sits on side A for some batches and side B for
//    others (PxNodeIndex::isArticulation decides);
//  * one holey mask per pool: lanes outside the mask never include that
//    batch in their window, and every other lane's offset shifts;
//  * one lane with BOTH counts zero -- vel/impulse/deltaV must come back
//    bit-identical (trap 5), and deltaQ is never read (trap 4);
//  * two rounds: (minPen=-MAX, residuals on) / (minPen=0, residuals off --
//    1D residual lanes must stay poisoned).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"
#include "pxgArtiSolveRef.h"
#include "pxgArtiExtSolveRef.h"
#include "pxgArtiInternalRef.h"

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

const uint32_t SI_LANES = 30;
const uint32_t SI_ZERO_LANE = 21;            // both counts forced to zero
const uint32_t SI_NB_BATCHES = 14;           // 1D pool 2..6, contact pool 8..12
const uint32_t SI_J0 = 2, SI_C0 = 8;
const uint32_t SI_HOLEY_J = 4, SI_HOLEY_C = 10;
const uint32_t SI_NB_HDRS = 14;
const uint32_t SI_NB_CPTS = 60, SI_NB_FPTS = 80, SI_NB_ROWS = 40, SI_NB_RESP = 130;
const float SI_ELAPSED = 0.019f;
const uint32_t SI_IO_STRIDE = 24, SI_IO_OUT = 32 * 24;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t siJStart(uint32_t lane) { return SI_J0 + (lane % 3u); }
static uint32_t siJCount(uint32_t lane) { return (lane == SI_ZERO_LANE) ? 0u : lane % 3u; }
static uint32_t siCStart(uint32_t lane) { return SI_C0 + (lane % 4u); }
static uint32_t siCCount(uint32_t lane) { return (lane == SI_ZERO_LANE) ? 0u : (lane + 1u) % 3u; }

static uint32_t siMask(uint32_t b)
{
    if (b == SI_HOLEY_J) return ~((1u << 12) | (1u << 16) | (1u << 18));  // r != 2 lanes
    if (b == SI_HOLEY_C) return ~((1u << 2) | (1u << 5) | (1u << 20));    // count-0 lanes
    return 0xFFFFFFFFu;
}

static bool siAIsArti(uint32_t b, uint32_t k) { return ((b + k) % 3u) != 0u; }

struct SiArena
{
    std::vector<PxgArticulationBlockLinkData> lk;   // ONE entry, lane-indexed
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgTGSBlockSolverContactHeader> ch;
    std::vector<PxgTGSBlockSolverFrictionHeader> fh;
    std::vector<PxgTGSBlockSolverContactPoint> cp;
    std::vector<PxgTGSBlockSolverContactFriction> fp;
    std::vector<PxgTGSBlockSolverConstraint1DHeader> jh;
    std::vector<PxgTGSBlockSolverConstraint1DCon> jr;
    std::vector<PxgArticulationBlockResponse> rs;
    std::vector<float> io;                          // 64 * 24 floats
};

static bool siIs1D(uint32_t b) { return b >= SI_J0 && b < SI_J0 + 5; }
static bool siIsContact(uint32_t b) { return b >= SI_C0 && b < SI_C0 + 5; }

static void siSeed(SiArena& B, uint32_t salt)
{
    B.lk.resize(1);
    B.bt.resize(SI_NB_BATCHES);
    B.ch.resize(SI_NB_HDRS);
    B.fh.resize(SI_NB_HDRS);
    B.cp.resize(SI_NB_CPTS);
    B.fp.resize(SI_NB_FPTS);
    B.jh.resize(SI_NB_HDRS);
    B.jr.resize(SI_NB_ROWS);
    B.rs.resize(SI_NB_RESP);
    B.io.resize(64 * SI_IO_STRIDE);
    std::memset(B.lk.data(), 0xAB, sizeof(B.lk[0]));
    std::memset(B.bt.data(), 0xAB, B.bt.size() * sizeof(B.bt[0]));
    std::memset(B.ch.data(), 0xAB, B.ch.size() * sizeof(B.ch[0]));
    std::memset(B.fh.data(), 0xAB, B.fh.size() * sizeof(B.fh[0]));
    std::memset(B.cp.data(), 0xAB, B.cp.size() * sizeof(B.cp[0]));
    std::memset(B.fp.data(), 0xAB, B.fp.size() * sizeof(B.fp[0]));
    std::memset(B.jh.data(), 0xAB, B.jh.size() * sizeof(B.jh[0]));
    std::memset(B.jr.data(), 0xAB, B.jr.size() * sizeof(B.jr[0]));
    std::memset(B.rs.data(), 0xAB, B.rs.size() * sizeof(B.rs[0]));
    std::memset(B.io.data(), 0xAB, B.io.size() * 4);

    PxgArticulationBlockLinkData& L = B.lk[0];
    for (uint32_t lane = 0; lane < 32; ++lane)
    {
        L.mStaticJointStartIndex[lane] = siJStart(lane);
        L.mNbStaticJoints[lane] = siJCount(lane);
        L.mStaticContactStartIndex[lane] = siCStart(lane);
        L.mNbStaticContacts[lane] = siCCount(lane);
        storeSpatialVector(L.mDeltaMotion,
                           UnAlignedSpatialVector(
                               PxVec3(0.1f * fv(lane, 0, 0, salt), 0.1f * fv(lane, 0, 1, salt), 0.1f * fv(lane, 0, 2, salt)),
                               PxVec3(0.1f * fv(lane, 0, 3, salt), 0.1f * fv(lane, 0, 4, salt), 0.1f * fv(lane, 0, 5, salt))),
                           lane);
        PxQuat dq(fv(lane, 0, 6, salt), fv(lane, 0, 7, salt), fv(lane, 0, 8, salt),
                  1.0f + 0.4f * fv(lane, 0, 9, salt));
        dq.normalize();
        L.mDeltaQ[lane] = make_float4(dq.x, dq.y, dq.z, dq.w);

        // io seeds: vel, impulse, deltaV
        for (uint32_t j = 0; j < 18; ++j)
            B.io[lane * SI_IO_STRIDE + j] = fv(lane, 1, j, salt);
        // the zero-count lane carries a NEGATIVE-ZERO deltaV component: an
        // unconditional `deltaV += vel - oldVel` computes -0 + +0 = +0 and
        // flips the bit (trap 5's mutant dies here, not by luck)
        if (lane == SI_ZERO_LANE)
            B.io[lane * SI_IO_STRIDE + 12] = -0.0f;
    }

    for (uint32_t b = 0; b < SI_NB_BATCHES; ++b)
    {
        if (!siIs1D(b) && !siIsContact(b)) continue;   // 0,1,7,13 stay poison
        PxgBlockConstraintBatch& BT = B.bt[b];
        BT.mDescStride = PxU16(31 + b);
        BT.constraintType = PxU16(siIsContact(b) ? PxgSolverConstraintDesc::eARTICULATION_CONTACT
                                                 : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
        BT.mask = siMask(b);
        BT.mConstraintBatchIndex = b;
        BT.startConstraintIndex = siIsContact(b) ? ((b - SI_C0) * 5 + 3) : ((b - SI_J0) * 6 + 2);
        BT.startFrictionIndex = (b - SI_C0) * 7 + 4;
        BT.mArticulationResponseIndex = b * 9 + 2;
        for (uint32_t k = 0; k < 32; ++k)
        {
            if (siAIsArti(b, k))
            {
                BT.bodyANodeIndex[k] = PxNodeIndex(60 + b, 3 + (k % 5));
                BT.bodyBNodeIndex[k] = PxNodeIndex(PxU32(200 + b + k));      // static
            }
            else
            {
                BT.bodyANodeIndex[k] = PxNodeIndex(PxU32(300 + b + k));      // static
                BT.bodyBNodeIndex[k] = PxNodeIndex(60 + b, 3 + (k % 5));
            }
        }

        if (siIsContact(b))
        {
            PxgTGSBlockSolverContactHeader& H = B.ch[b];
            PxgTGSBlockSolverFrictionHeader& F = B.fh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                H.invMass0_1_angDom0_1[lane] = make_float4(0.6f + 0.2f * fv(lane, b, 60, salt),
                                                           0.5f + 0.2f * fv(lane, b, 61, salt),
                                                           0.7f + 0.2f * fv(lane, b, 62, salt),
                                                           0.8f + 0.2f * fv(lane, b, 63, salt));
                H.normal_staticFriction[lane] = make_float4(fv(lane, b, 64, salt), fv(lane, b, 65, salt),
                                                            fv(lane, b, 66, salt), 0.6f + 0.3f * fv(lane, b, 67, salt));
                H.minNormalForce[lane] = 0.05f + 0.05f * fv(lane, b, 68, salt);
                H.maxPenBias[lane] = -0.4f + 0.1f * fv(lane, b, 69, salt);
                H.flags[lane] = ((lane + b) % 3u == 1u) ? COMPLIANT_ACCEL : 0u;
                H.numNormalConstr[lane] = 1u + ((lane + b) % 3u);
                H.forceWritebackOffset[lane] = 0;
                H.restitutionXdt[lane] = ((lane + b) % 4u == 1u) ? (-0.3f + 0.1f * fv(lane, b, 70, salt))
                                                                 : (0.2f + 0.1f * fv(lane, b, 71, salt));
                H.cfm[lane] = 0.01f + 0.005f * fv(lane, b, 72, salt);
                H.p8[lane] = 0.9f;
                F.numFrictionConstr[lane] = ((lane + b) % 3u) * 2u + (((lane + b) % 5u == 2u) ? 1u : 0u);
                F.dynamicFriction[lane] = 0.4f + 0.2f * fv(lane, b, 73, salt);
                F.biasCoefficient[lane] = 0.5f + 0.2f * fv(lane, b, 74, salt);
                F.torsionalFrictionScale[lane] = 0.8f + 0.1f * fv(lane, b, 75, salt);
                F.frictionNormals[0][lane] = make_float4(fv(lane, b, 76, salt), fv(lane, b, 77, salt), fv(lane, b, 78, salt), 0.f);
                F.frictionNormals[1][lane] = make_float4(fv(lane, b, 79, salt), fv(lane, b, 80, salt), fv(lane, b, 81, salt), 0.f);
                F.broken[lane] = 0;
                for (uint32_t i = 0; i < 3; ++i)
                {
                    PxgTGSBlockSolverContactPoint& C = B.cp[BT.startConstraintIndex + i];
                    C.raXn_extraCoeff[lane] = make_float4(fv(lane, b * 4 + i, 82, salt), fv(lane, b * 4 + i, 83, salt),
                                                          fv(lane, b * 4 + i, 84, salt), 0.5f + 0.3f * fv(lane, b * 4 + i, 85, salt));
                    C.rbXn_targetVelW[lane] = make_float4(fv(lane, b * 4 + i, 86, salt), fv(lane, b * 4 + i, 87, salt),
                                                          fv(lane, b * 4 + i, 88, salt), 0.1f * fv(lane, b * 4 + i, 89, salt));
                    C.separation[lane] = 0.05f * fv(lane, b * 4 + i, 90, salt);
                    C.maxImpulse[lane] = 3.0f + fv(lane, b * 4 + i, 91, salt);
                    C.appliedForce[lane] = 0.3f + 0.2f * fv(lane, b * 4 + i, 92, salt);
                    C.biasCoefficient[lane] = 0.4f + 0.2f * fv(lane, b * 4 + i, 93, salt);
                    C.resp0[lane] = 0.5f + 0.3f * fv(lane, b * 4 + i, 94, salt);
                    C.resp1[lane] = 0.5f + 0.3f * fv(lane, b * 4 + i, 95, salt);
                }
                for (uint32_t i = 0; i < 5; ++i)
                {
                    PxgTGSBlockSolverContactFriction& FR = B.fp[BT.startFrictionIndex + i];
                    FR.raXn_error[lane] = make_float4(fv(lane, b * 6 + i, 96, salt), fv(lane, b * 6 + i, 97, salt),
                                                      fv(lane, b * 6 + i, 98, salt), 0.05f * fv(lane, b * 6 + i, 99, salt));
                    FR.rbXn_targetVelW[lane] = make_float4(fv(lane, b * 6 + i, 100, salt), fv(lane, b * 6 + i, 101, salt),
                                                           fv(lane, b * 6 + i, 102, salt), 0.1f * fv(lane, b * 6 + i, 103, salt));
                    FR.appliedForce[lane] = 0.2f + 0.1f * fv(lane, b * 6 + i, 104, salt);
                    FR.resp0[lane] = 0.4f + 0.3f * fv(lane, b * 6 + i, 105, salt);
                    FR.resp1[lane] = 0.4f + 0.3f * fv(lane, b * 6 + i, 106, salt);
                }
            }
        }
        else
        {
            PxgTGSBlockSolverConstraint1DHeader& H = B.jh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint32_t rc = 1u + ((lane + b) % 4u);
                uchar4 packed;
                packed.x = PxU8(rc); packed.y = PxU8(rc + 2); packed.z = 1; packed.w = 2;
                H.rowCounts_breakable_orthoAxisCount[lane] = packed;
                H.rAWorld_invMass0D0[lane] = make_float4(fv(lane, b, 110, salt), fv(lane, b, 111, salt),
                                                         fv(lane, b, 112, salt), 0.5f + 0.2f * fv(lane, b, 113, salt));
                H.rBWorld_invMass1D1[lane] = make_float4(fv(lane, b, 114, salt), fv(lane, b, 115, salt),
                                                         fv(lane, b, 116, salt), 0.5f + 0.2f * fv(lane, b, 117, salt));
                H.invInertiaScale0[lane] = 0.6f + 0.2f * fv(lane, b, 118, salt);
                H.invInertiaScale1[lane] = 0.6f + 0.2f * fv(lane, b, 119, salt);
                H.cfm[lane] = 0.01f + 0.004f * fv(lane, b, 120, salt);
                for (uint32_t i = 0; i < 5; ++i)
                {
                    PxgTGSBlockSolverConstraint1DCon& R = B.jr[BT.startConstraintIndex + i];
                    R.lin0XYZ_initBiasOrCoeff0[lane] = make_float4(fv(lane, b * 5 + i, 121, salt), fv(lane, b * 5 + i, 122, salt),
                                                                   fv(lane, b * 5 + i, 123, salt), 0.2f * fv(lane, b * 5 + i, 124, salt));
                    R.lin1XYZ_biasScaleOrCoeff1[lane] = make_float4(fv(lane, b * 5 + i, 125, salt), fv(lane, b * 5 + i, 126, salt),
                                                                    fv(lane, b * 5 + i, 127, salt), 0.5f + 0.2f * fv(lane, b * 5 + i, 128, salt));
                    // non-spring rows need BOTH bounce flavors: bounceVel != 0
                    // kills errorDelta (biasScale forced 0), so half the
                    // non-spring rows get bounceVel == 0 -- the only rows in
                    // which elapsedTime reaches the 1D bias at all
                    static const uint32_t modesPre[5] = { 0u, DY_FLAG_SPRING, DY_FLAG_SPRING | DY_FLAG_ACCEL,
                                                          DY_FLAG_INEQ, DY_FLAG_SPRING | DY_FLAG_INEQ };
                    const bool nonSpring = (modesPre[(lane + b + i) % 5u] & DY_FLAG_SPRING) == 0;
                    const float coeff2w = (nonSpring && ((lane + i) % 2u == 0u))
                                              ? 0.0f : 0.6f + 0.2f * fv(lane, b * 5 + i, 132, salt);
                    R.ang0XYZ_velMultiplierOrCoeff2[lane] = make_float4(fv(lane, b * 5 + i, 129, salt), fv(lane, b * 5 + i, 130, salt),
                                                                        fv(lane, b * 5 + i, 131, salt), coeff2w);
                    R.ang1XYZ_velTargetOrCoeff3[lane] = make_float4(fv(lane, b * 5 + i, 133, salt), fv(lane, b * 5 + i, 134, salt),
                                                                    fv(lane, b * 5 + i, 135, salt), 0.1f * fv(lane, b * 5 + i, 136, salt));
                    R.resp0[lane] = 0.5f + 0.3f * fv(lane, b * 5 + i, 137, salt);
                    R.resp1[lane] = 0.5f + 0.3f * fv(lane, b * 5 + i, 138, salt);
                    R.geometricError[lane] = 0.1f * fv(lane, b * 5 + i, 139, salt);
                    R.minImpulse[lane] = -4.0f + fv(lane, b * 5 + i, 140, salt);
                    R.maxImpulse[lane] = 3.0f + fv(lane, b * 5 + i, 141, salt);
                    R.maxBias[lane] = 0.8f + 0.2f * fv(lane, b * 5 + i, 142, salt);
                    R.angularErrorScale[lane] = ((lane + i) % 2u) ? 1.0f : 0.7f;
                    static const uint32_t modes[5] = { 0u, DY_FLAG_SPRING, DY_FLAG_SPRING | DY_FLAG_ACCEL,
                                                       DY_FLAG_INEQ, DY_FLAG_SPRING | DY_FLAG_INEQ };
                    R.flags[lane] = modes[(lane + b + i) % 5u];
                    R.appliedForce[lane] = 0.2f + 0.2f * fv(lane, b * 5 + i, 143, salt);
                }
            }
        }
    }

    for (uint32_t r = 0; r < SI_NB_RESP; ++r)
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            PxgArticulationBlockResponse& R = B.rs[r];
            R.deltaRALin_x[lane] = 0.3f * fv(lane, r, 150, salt);
            R.deltaRALin_y[lane] = 0.3f * fv(lane, r, 151, salt);
            R.deltaRALin_z[lane] = 0.3f * fv(lane, r, 152, salt);
            R.deltaRAAng_x[lane] = 0.3f * fv(lane, r, 153, salt);
            R.deltaRAAng_y[lane] = 0.3f * fv(lane, r, 154, salt);
            R.deltaRAAng_z[lane] = 0.3f * fv(lane, r, 155, salt);
            R.deltaRBLin_x[lane] = 0.3f * fv(lane, r, 156, salt);
            R.deltaRBLin_y[lane] = 0.3f * fv(lane, r, 157, salt);
            R.deltaRBLin_z[lane] = 0.3f * fv(lane, r, 158, salt);
            R.deltaRBAng_x[lane] = 0.3f * fv(lane, r, 159, salt);
            R.deltaRBAng_y[lane] = 0.3f * fv(lane, r, 160, salt);
            R.deltaRBAng_z[lane] = 0.3f * fv(lane, r, 161, salt);
        }
}

// ---- mirror ----

static void siMirror(SiArena& W, float minPen, bool accumErr,
                     uint32_t& aSide, uint32_t& bSide, uint32_t& bothZero)
{
    PxgArticulationBlockLinkData& L = W.lk[0];
    for (uint32_t lane = 0; lane < SI_LANES; ++lane)
    {
        UnAlignedSpatialVector vel(
            PxVec3(W.io[lane * SI_IO_STRIDE + 0], W.io[lane * SI_IO_STRIDE + 1], W.io[lane * SI_IO_STRIDE + 2]),
            PxVec3(W.io[lane * SI_IO_STRIDE + 3], W.io[lane * SI_IO_STRIDE + 4], W.io[lane * SI_IO_STRIDE + 5]));
        UnAlignedSpatialVector impulse(
            PxVec3(W.io[lane * SI_IO_STRIDE + 6], W.io[lane * SI_IO_STRIDE + 7], W.io[lane * SI_IO_STRIDE + 8]),
            PxVec3(W.io[lane * SI_IO_STRIDE + 9], W.io[lane * SI_IO_STRIDE + 10], W.io[lane * SI_IO_STRIDE + 11]));
        UnAlignedSpatialVector deltaV(
            PxVec3(W.io[lane * SI_IO_STRIDE + 12], W.io[lane * SI_IO_STRIDE + 13], W.io[lane * SI_IO_STRIDE + 14]),
            PxVec3(W.io[lane * SI_IO_STRIDE + 15], W.io[lane * SI_IO_STRIDE + 16], W.io[lane * SI_IO_STRIDE + 17]));

        const uint32_t constraintCounts = L.mNbStaticJoints[lane];
        if (constraintCounts + L.mNbStaticContacts[lane] == 0) ++bothZero;

        RefError err;
        StaticCov scov;
        // hoisted to pxgArtiInternalRef.h in P8.100c; this fixture is a user
        refSolveStaticLane(L, lane, W.bt.data(), W.ch.data(), W.fh.data(), W.cp.data(),
                           W.fp.data(), W.jh.data(), W.jr.data(), W.rs.data(),
                           SI_ELAPSED, minPen, accumErr, vel, impulse, deltaV, err,
                           constraintCounts, &scov);
        aSide += scov.aSide;
        bSide += scov.bSide;

        float* out = &W.io[SI_IO_OUT + lane * SI_IO_STRIDE];
        out[0] = vel.top.x; out[1] = vel.top.y; out[2] = vel.top.z;
        out[3] = vel.bottom.x; out[4] = vel.bottom.y; out[5] = vel.bottom.z;
        out[6] = impulse.top.x; out[7] = impulse.top.y; out[8] = impulse.top.z;
        out[9] = impulse.bottom.x; out[10] = impulse.bottom.y; out[11] = impulse.bottom.z;
        out[12] = deltaV.top.x; out[13] = deltaV.top.y; out[14] = deltaV.top.z;
        out[15] = deltaV.bottom.x; out[16] = deltaV.bottom.y; out[17] = deltaV.bottom.z;
        out[18] = err.sumOfSquares; out[19] = float(err.counter); out[20] = err.maxError;
    }
}

// ---- run ----

struct SiParamsHost
{
    uint64_t link, batches, ch, fh, cp, fp, jh, jr, rs, io;
    float minPen, elapsedTime;
    uint32_t accumErr, nbLanes;
};
static_assert(sizeof(SiParamsHost) == 96, "params blob");

static size_t siRun(PxgCudaShim& shim, float minPen, bool accumErr, uint32_t salt)
{
    SiArena A;
    siSeed(A, salt);

    SiArena W = A;
    uint32_t aSide = 0, bSide = 0, bothZero = 0;
    siMirror(W, minPen, accumErr, aSide, bSide, bothZero);
    std::printf("[ info ] static(%s): arti on A %u / on B %u, both-zero lanes %u\n",
                accumErr ? "accum" : "plain", aSide, bSide, bothZero);
    PXG_CHECK_MSG(aSide > 0 && bSide > 0 && bothZero > 0, "side coverage hole");
    PXG_CHECK_MSG(std::memcmp(&W.io[SI_IO_OUT], &A.io[SI_IO_OUT], 30 * SI_IO_STRIDE * 4) != 0,
                  "vacuous");

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dLk = up(A.lk.data(), sizeof(A.lk[0]));
    PxgDevicePtr dBt = up(A.bt.data(), A.bt.size() * sizeof(A.bt[0]));
    PxgDevicePtr dCh = up(A.ch.data(), A.ch.size() * sizeof(A.ch[0]));
    PxgDevicePtr dFh = up(A.fh.data(), A.fh.size() * sizeof(A.fh[0]));
    PxgDevicePtr dCp = up(A.cp.data(), A.cp.size() * sizeof(A.cp[0]));
    PxgDevicePtr dFp = up(A.fp.data(), A.fp.size() * sizeof(A.fp[0]));
    PxgDevicePtr dJh = up(A.jh.data(), A.jh.size() * sizeof(A.jh[0]));
    PxgDevicePtr dJr = up(A.jr.data(), A.jr.size() * sizeof(A.jr[0]));
    PxgDevicePtr dRs = up(A.rs.data(), A.rs.size() * sizeof(A.rs[0]));
    PxgDevicePtr dIo = up(A.io.data(), A.io.size() * 4);
    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SiParamsHost prm = { uint64_t(dLk), uint64_t(dBt), uint64_t(dCh), uint64_t(dFh),
                         uint64_t(dCp), uint64_t(dFp), uint64_t(dJh), uint64_t(dJr),
                         uint64_t(dRs), uint64_t(dIo), minPen, SI_ELAPSED,
                         accumErr ? 1u : 0u, SI_LANES };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch static solve");
    shim.streamSynchronize();

    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const void* host, size_t n, const char* name) {
        std::vector<uint8_t> g(n);
        shim.memcpyDtoH(g.data(), dp, n);
        if (std::memcmp(g.data(), host, n) != 0)
        {
            size_t mm = 0, first = size_t(-1);
            const uint8_t* h = reinterpret_cast<const uint8_t*>(host);
            for (size_t i = 0; i < n; ++i)
                if (g[i] != h[i]) { ++mm; if (first == size_t(-1)) first = i; }
            std::printf("MM static/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dLk, W.lk.data(), sizeof(W.lk[0]), "linkBlock");
    cmp(dBt, W.bt.data(), W.bt.size() * sizeof(W.bt[0]), "batches");
    cmp(dCh, W.ch.data(), W.ch.size() * sizeof(W.ch[0]), "contactHeaders");
    cmp(dFh, W.fh.data(), W.fh.size() * sizeof(W.fh[0]), "frictionHeaders");
    cmp(dCp, W.cp.data(), W.cp.size() * sizeof(W.cp[0]), "contactPoints");
    cmp(dFp, W.fp.data(), W.fp.size() * sizeof(W.fp[0]), "frictions");
    cmp(dJh, W.jh.data(), W.jh.size() * sizeof(W.jh[0]), "jointHeaders");
    cmp(dJr, W.jr.data(), W.jr.size() * sizeof(W.jr[0]), "jointRows");
    cmp(dRs, W.rs.data(), W.rs.size() * sizeof(W.rs[0]), "responses");
    cmp(dIo, W.io.data(), W.io.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveStatic)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveInternal.spv", "siStaticTest", 96u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &siLoader);

    size_t bad = 0;
    bad += siRun(shim, -PX_MAX_F32, true, 31);
    bad += siRun(shim, 0.0f, false, 37);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
