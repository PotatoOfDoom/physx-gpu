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

// testArtiSolveStaticPgs.cpp -- P8.123a: siSolveStaticConstraintsPgs, the
// static constraint solve of artiSolveInternalConstraints1T
// (solveStaticConstraints PGS overload, internalConstraints2.cu:1268-1392).
//
// Mirror: refSolveStaticLanePgs (pxgArtiInternalRef.h) over the P8.122a PGS
// ext-solve mirrors.  The window/mask geometry is the TGS fixture's; the
// PGS-specific facts:
//  * mDeltaMotion and mDeltaQ stay POISONED -- the PGS overload never reads
//    them (module trap 2), so the link block must come back 0xAB there;
//  * doFriction is LIVE (module trap 4): a doFriction=0 round leaves the
//    friction rows' appliedForce and the header's broken untouched;
//  * the 1D response base is OFFSET by batch.mArticulationResponseIndex
//    (module trap 3) -- response indices vary per batch so an unoffset
//    base lands on wrong numbers;
//  * one lane with BOTH counts zero: vel/impulse/deltaV bit-identical, with
//    a negative-zero deltaV component so an unconditional guard flips it;
//  * three rounds: (doF=1, residuals on) / (doF=0, residuals on) /
//    (doF=1, residuals off -- 1D residual lanes must stay poisoned).

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

static std::vector<uint8_t> spgLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SPG_LANES = 30;
const uint32_t SPG_ZERO_LANE = 21;            // both counts forced to zero
const uint32_t SPG_NB_BATCHES = 14;           // 1D pool 2..6, contact pool 8..12
const uint32_t SPG_J0 = 2, SPG_C0 = 8;
const uint32_t SPG_HOLEY_J = 4, SPG_HOLEY_C = 10;
const uint32_t SPG_NB_HDRS = 14;
const uint32_t SPG_NB_CPTS = 60, SPG_NB_FPTS = 80, SPG_NB_ROWS = 40, SPG_NB_RESP = 130;
const uint32_t SPG_IO_STRIDE = 24, SPG_IO_OUT = 32 * 24;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t spgJStart(uint32_t lane) { return SPG_J0 + (lane % 3u); }
static uint32_t spgJCount(uint32_t lane) { return (lane == SPG_ZERO_LANE) ? 0u : lane % 3u; }
static uint32_t spgCStart(uint32_t lane) { return SPG_C0 + (lane % 4u); }
static uint32_t spgCCount(uint32_t lane) { return (lane == SPG_ZERO_LANE) ? 0u : (lane + 1u) % 3u; }

static uint32_t spgMask(uint32_t b)
{
    if (b == SPG_HOLEY_J) return ~((1u << 12) | (1u << 16) | (1u << 18));  // r != 2 lanes
    if (b == SPG_HOLEY_C) return ~((1u << 2) | (1u << 5) | (1u << 20));    // count-0 lanes
    return 0xFFFFFFFFu;
}

static bool spgAIsArti(uint32_t b, uint32_t k) { return ((b + k) % 3u) != 0u; }

struct SpgArena
{
    std::vector<PxgArticulationBlockLinkData> lk;   // ONE entry, lane-indexed
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgBlockSolverContactHeader> ch;
    std::vector<PxgBlockSolverFrictionHeader> fh;
    std::vector<PxgBlockSolverContactPoint> cp;
    std::vector<PxgBlockSolverContactFriction> fp;
    std::vector<PxgBlockSolverConstraint1DHeader> jh;
    std::vector<PxgBlockSolverConstraint1DCon> jr;
    std::vector<PxgBlockSolverConstraint1DMod> jm;
    std::vector<PxgArticulationBlockResponse> rs;
    std::vector<float> io;                          // 64 * 24 floats
};

static bool spgIs1D(uint32_t b) { return b >= SPG_J0 && b < SPG_J0 + 5; }
static bool spgIsContact(uint32_t b) { return b >= SPG_C0 && b < SPG_C0 + 5; }

static void spgSeed(SpgArena& B, uint32_t salt)
{
    B.lk.resize(1);
    B.bt.resize(SPG_NB_BATCHES);
    B.ch.resize(SPG_NB_HDRS);
    B.fh.resize(SPG_NB_HDRS);
    B.cp.resize(SPG_NB_CPTS);
    B.fp.resize(SPG_NB_FPTS);
    B.jh.resize(SPG_NB_HDRS);
    B.jr.resize(SPG_NB_ROWS);
    B.jm.resize(SPG_NB_ROWS);
    B.rs.resize(SPG_NB_RESP);
    B.io.resize(64 * SPG_IO_STRIDE);
    std::memset(B.lk.data(), 0xAB, sizeof(B.lk[0]));
    std::memset(B.bt.data(), 0xAB, B.bt.size() * sizeof(B.bt[0]));
    std::memset(B.ch.data(), 0xAB, B.ch.size() * sizeof(B.ch[0]));
    std::memset(B.fh.data(), 0xAB, B.fh.size() * sizeof(B.fh[0]));
    std::memset(B.cp.data(), 0xAB, B.cp.size() * sizeof(B.cp[0]));
    std::memset(B.fp.data(), 0xAB, B.fp.size() * sizeof(B.fp[0]));
    std::memset(B.jh.data(), 0xAB, B.jh.size() * sizeof(B.jh[0]));
    std::memset(B.jr.data(), 0xAB, B.jr.size() * sizeof(B.jr[0]));
    std::memset(B.jm.data(), 0xAB, B.jm.size() * sizeof(B.jm[0]));
    std::memset(B.rs.data(), 0xAB, B.rs.size() * sizeof(B.rs[0]));
    std::memset(B.io.data(), 0xAB, B.io.size() * 4);

    PxgArticulationBlockLinkData& L = B.lk[0];
    for (uint32_t lane = 0; lane < 32; ++lane)
    {
        L.mStaticJointStartIndex[lane] = spgJStart(lane);
        L.mNbStaticJoints[lane] = spgJCount(lane);
        L.mStaticContactStartIndex[lane] = spgCStart(lane);
        L.mNbStaticContacts[lane] = spgCCount(lane);
        // mDeltaMotion / mDeltaQ stay POISONED: the PGS overload never
        // reads them (module trap 2)

        // io seeds: vel, impulse, deltaV
        for (uint32_t j = 0; j < 18; ++j)
            B.io[lane * SPG_IO_STRIDE + j] = fv(lane, 1, j, salt);
        // the zero-count lane carries a NEGATIVE-ZERO deltaV component: an
        // unconditional `deltaV += vel - oldVel` computes -0 + +0 = +0 and
        // flips the bit (the guard's mutant dies here, not by luck)
        if (lane == SPG_ZERO_LANE)
            B.io[lane * SPG_IO_STRIDE + 12] = -0.0f;
    }

    for (uint32_t b = 0; b < SPG_NB_BATCHES; ++b)
    {
        if (!spgIs1D(b) && !spgIsContact(b)) continue;   // 0,1,7,13 stay poison
        PxgBlockConstraintBatch& BT = B.bt[b];
        BT.mDescStride = PxU16(31 + b);
        BT.constraintType = PxU16(spgIsContact(b) ? PxgSolverConstraintDesc::eARTICULATION_CONTACT
                                                  : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
        BT.mask = spgMask(b);
        BT.mConstraintBatchIndex = b;
        BT.startConstraintIndex = spgIsContact(b) ? ((b - SPG_C0) * 5 + 3) : ((b - SPG_J0) * 6 + 2);
        BT.startFrictionIndex = (b - SPG_C0) * 7 + 4;
        BT.mArticulationResponseIndex = b * 9 + 2;
        for (uint32_t k = 0; k < 32; ++k)
        {
            if (spgAIsArti(b, k))
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

        if (spgIsContact(b))
        {
            PxgBlockSolverContactHeader& H = B.ch[b];
            PxgBlockSolverFrictionHeader& F = B.fh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint32_t nc = 1u + ((lane + b) % 3u);
                H.numNormalConstr[lane] = nc;
                H.invMass0_1_angDom0_1[lane] = make_float4(0.6f + 0.2f * fv(lane, b, 60, salt),
                                                           0.5f + 0.2f * fv(lane, b, 61, salt),
                                                           0.7f + 0.2f * fv(lane, b, 62, salt),
                                                           0.8f + 0.2f * fv(lane, b, 63, salt));
                PxVec3 n(fv(lane, b, 64, salt), fv(lane, b, 65, salt), fv(lane, b, 66, salt));
                n = n.getNormalized();
                H.normal_staticFriction[lane] = make_float4(n.x, n.y, n.z,
                                                            0.6f + 0.3f * std::fabs(fv(lane, b, 67, salt)));
                H.minNormalForce[lane] = ((lane + b) % 5u == 2u)
                                             ? 2.0f + std::fabs(fv(lane, b, 68, salt))
                                             : 0.02f * std::fabs(fv(lane, b, 68, salt));
                H.flags[lane] = ((lane + b) % 3u == 1u) ? COMPLIANT_ACCEL : 0u;
                // compliant lanes: RAW restitution < 0 (the PGS header keeps it raw)
                H.restitution[lane] = ((lane + b) % 4u == 1u) ? (-500.0f - 10.0f * float(lane))
                                                              : (0.2f + 0.1f * std::fabs(fv(lane, b, 71, salt)));
                H.cfm[lane] = 0.01f + 0.005f * std::fabs(fv(lane, b, 72, salt));
                H.accumNormalForce[lane] = fv(lane, b, 59, salt);   // NOT touched by the solve
                H.forceWritebackOffset[lane] = 0xABABABABu;         // NOT touched
                // the prep invariant: zero contacts => zero frictions; counts EVEN
                F.numFrictionConstr[lane] = ((lane + b) % 3u) * 2u;
                F.dynamicFriction[lane] = 0.4f + 0.2f * std::fabs(fv(lane, b, 73, salt));
                PxVec3 f0(fv(lane, b, 76, salt), fv(lane, b, 77, salt), fv(lane, b, 78, salt));
                PxVec3 f1(fv(lane, b, 79, salt), fv(lane, b, 80, salt), fv(lane, b, 81, salt));
                f0 = f0.getNormalized(); f1 = f1.getNormalized();
                F.frictionNormals[0][lane] = make_float4(f0.x, f0.y, f0.z, 0.f);
                F.frictionNormals[1][lane] = make_float4(f1.x, f1.y, f1.z, 0.f);
                // broken stays POISONED: the doFriction=0 round must not write it
                for (uint32_t i = 0; i < 3; ++i)
                {
                    PxgBlockSolverContactPoint& C = B.cp[BT.startConstraintIndex + i];
                    C.raXn_targetVelocity[lane] = make_float4(fv(lane, b * 4 + i, 82, salt), fv(lane, b * 4 + i, 83, salt),
                                                              fv(lane, b * 4 + i, 84, salt), 0.2f * fv(lane, b * 4 + i, 85, salt));
                    C.rbXn_maxImpulse[lane] = make_float4(fv(lane, b * 4 + i, 86, salt), fv(lane, b * 4 + i, 87, salt),
                                                          fv(lane, b * 4 + i, 88, salt),
                                                          0.3f + 0.3f * std::fabs(fv(lane, b * 4 + i, 89, salt)));
                    C.appliedForce[lane] = 0.2f * std::fabs(fv(lane, b * 4 + i, 92, salt));
                    C.resp0[lane] = 0.5f + 0.3f * std::fabs(fv(lane, b * 4 + i, 94, salt));
                    C.resp1[lane] = 0.5f + 0.3f * std::fabs(fv(lane, b * 4 + i, 95, salt));
                    // compliant lanes need a plausible a > 0 in coeff0
                    C.coeff0[lane] = ((lane + b) % 4u) == 1u ? 0.5f + 0.4f * std::fabs(fv(lane, b * 4 + i, 90, salt))
                                                             : 0.4f * fv(lane, b * 4 + i, 90, salt);
                    C.coeff1[lane] = 0.4f * fv(lane, b * 4 + i, 91, salt);
                }
                for (uint32_t i = 0; i < 5; ++i)
                {
                    PxgBlockSolverContactFriction& FR = B.fp[BT.startFrictionIndex + i];
                    FR.raXn_bias[lane] = make_float4(fv(lane, b * 6 + i, 96, salt), fv(lane, b * 6 + i, 97, salt),
                                                     fv(lane, b * 6 + i, 98, salt), 0.3f * fv(lane, b * 6 + i, 99, salt));
                    FR.rbXn_targetVelW[lane] = make_float4(fv(lane, b * 6 + i, 100, salt), fv(lane, b * 6 + i, 101, salt),
                                                           fv(lane, b * 6 + i, 102, salt), 0.1f * fv(lane, b * 6 + i, 103, salt));
                    FR.appliedForce[lane] = 0.2f + 0.1f * fv(lane, b * 6 + i, 104, salt);
                    FR.resp0[lane] = 0.4f + 0.3f * std::fabs(fv(lane, b * 6 + i, 105, salt));
                    FR.resp1[lane] = 0.4f + 0.3f * std::fabs(fv(lane, b * 6 + i, 106, salt));
                }
            }
        }
        else
        {
            PxgBlockSolverConstraint1DHeader& H = B.jh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint32_t rc = 1u + ((lane + b) % 4u);
                H.rowCounts[lane] = PxU16(rc);
                H.breakable[lane] = PxU16(lane & 1);   // NOT read by the solve
                H.invMass0D0[lane] = 0.5f + 0.2f * fv(lane, b, 113, salt);
                H.invMass1D1[lane] = 0.5f + 0.2f * fv(lane, b, 117, salt);
                H.invInertiaScale0[lane] = 0.6f + 0.2f * fv(lane, b, 118, salt);
                H.invInertiaScale1[lane] = 0.6f + 0.2f * fv(lane, b, 119, salt);
                H.cfm[lane] = 0.01f + 0.004f * fv(lane, b, 120, salt);
                for (uint32_t i = 0; i < 5; ++i)
                {
                    PxgBlockSolverConstraint1DCon& R = B.jr[BT.startConstraintIndex + i];
                    PxgBlockSolverConstraint1DMod& M = B.jm[BT.startConstraintIndex + i];
                    // narrow impulse windows so both clamps fire on some rows
                    const bool narrow = ((lane + i) % 4u) == 3u;
                    const float minImp = narrow ? -0.02f : -0.6f - 0.3f * std::fabs(fv(lane, b * 5 + i, 140, salt));
                    const float maxImp = narrow ? 0.02f : 0.6f + 0.3f * std::fabs(fv(lane, b * 5 + i, 141, salt));
                    R.lin0XYZ_minImpulse[lane] = make_float4(fv(lane, b * 5 + i, 121, salt), fv(lane, b * 5 + i, 122, salt),
                                                             fv(lane, b * 5 + i, 123, salt), minImp);
                    R.lin1XYZ_maxImpulse[lane] = make_float4(fv(lane, b * 5 + i, 125, salt), fv(lane, b * 5 + i, 126, salt),
                                                             fv(lane, b * 5 + i, 127, salt), maxImp);
                    R.ang0XYZ_resp0[lane] = make_float4(fv(lane, b * 5 + i, 129, salt), fv(lane, b * 5 + i, 130, salt),
                                                        fv(lane, b * 5 + i, 131, salt),
                                                        0.5f + 0.3f * std::fabs(fv(lane, b * 5 + i, 137, salt)));
                    R.ang1XYZ_resp1[lane] = make_float4(fv(lane, b * 5 + i, 133, salt), fv(lane, b * 5 + i, 134, salt),
                                                        fv(lane, b * 5 + i, 135, salt),
                                                        0.5f + 0.3f * std::fabs(fv(lane, b * 5 + i, 138, salt)));
                    R.initJointSpeed[lane] = 0.3f * fv(lane, b * 5 + i, 136, salt);
                    const uint32_t mode = (lane + b + i) % 3u;   // hard / spring / accel-spring
                    M.flags[lane] = (mode == 1u) ? DY_FLAG_SPRING
                                                 : (mode == 2u) ? (DY_FLAG_SPRING | DY_FLAG_ACCEL) : 0u;
                    M.coeff0[lane] = (mode != 0u) ? 0.5f + 0.4f * std::fabs(fv(lane, b * 5 + i, 124, salt))
                                                  : 0.1f * fv(lane, b * 5 + i, 124, salt);
                    M.coeff1[lane] = 0.3f * fv(lane, b * 5 + i, 128, salt);
                    M.appliedForce[lane] = 0.2f + 0.2f * fv(lane, b * 5 + i, 143, salt);
                    // residual stays POISONED: only reporting rounds may write it
                }
            }
        }
    }

    for (uint32_t r = 0; r < SPG_NB_RESP; ++r)
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

// 5.12: the rigidContact bias of the PGS static contacts (5.6.1 hard-coded 0.8)
static const float SPG_RIGID_CONTACT_BIAS = 0.65f;

static void spgMirror(SpgArena& W, bool doFriction, bool accumErr,
                      uint32_t& aSide, uint32_t& bSide, uint32_t& bothZero)
{
    PxgArticulationBlockLinkData& L = W.lk[0];
    for (uint32_t lane = 0; lane < SPG_LANES; ++lane)
    {
        UnAlignedSpatialVector vel(
            PxVec3(W.io[lane * SPG_IO_STRIDE + 0], W.io[lane * SPG_IO_STRIDE + 1], W.io[lane * SPG_IO_STRIDE + 2]),
            PxVec3(W.io[lane * SPG_IO_STRIDE + 3], W.io[lane * SPG_IO_STRIDE + 4], W.io[lane * SPG_IO_STRIDE + 5]));
        UnAlignedSpatialVector impulse(
            PxVec3(W.io[lane * SPG_IO_STRIDE + 6], W.io[lane * SPG_IO_STRIDE + 7], W.io[lane * SPG_IO_STRIDE + 8]),
            PxVec3(W.io[lane * SPG_IO_STRIDE + 9], W.io[lane * SPG_IO_STRIDE + 10], W.io[lane * SPG_IO_STRIDE + 11]));
        UnAlignedSpatialVector deltaV(
            PxVec3(W.io[lane * SPG_IO_STRIDE + 12], W.io[lane * SPG_IO_STRIDE + 13], W.io[lane * SPG_IO_STRIDE + 14]),
            PxVec3(W.io[lane * SPG_IO_STRIDE + 15], W.io[lane * SPG_IO_STRIDE + 16], W.io[lane * SPG_IO_STRIDE + 17]));

        const uint32_t constraintCounts = L.mNbStaticJoints[lane];
        if (constraintCounts + L.mNbStaticContacts[lane] == 0) ++bothZero;

        RefError err;
        StaticCov scov;
        refSolveStaticLanePgs(L, lane, W.bt.data(), W.ch.data(), W.fh.data(), W.cp.data(),
                              W.fp.data(), W.jh.data(), W.jr.data(), W.jm.data(), W.rs.data(),
                              doFriction, accumErr, vel, impulse, deltaV, err,
                              constraintCounts, &scov, SPG_RIGID_CONTACT_BIAS);
        aSide += scov.aSide;
        bSide += scov.bSide;

        float* out = &W.io[SPG_IO_OUT + lane * SPG_IO_STRIDE];
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

struct SpgParamsHost
{
    uint64_t link, batches, ch, fh, cp, fp, jh, jr, jm, rs, io;
    uint32_t doFriction, accumErr, nbLanes;
    float rigidContactBiasCoefficient;   // @100 (5.12: was pad0)
};
static_assert(sizeof(SpgParamsHost) == 104, "params blob");

static size_t spgRun(PxgCudaShim& shim, bool doFriction, bool accumErr, uint32_t salt)
{
    SpgArena A;
    spgSeed(A, salt);

    SpgArena W = A;
    uint32_t aSide = 0, bSide = 0, bothZero = 0;
    spgMirror(W, doFriction, accumErr, aSide, bSide, bothZero);
    std::printf("[ info ] staticPgs(doF %d, %s): arti on A %u / on B %u, both-zero lanes %u\n",
                int(doFriction), accumErr ? "accum" : "plain", aSide, bSide, bothZero);
    PXG_CHECK_MSG(aSide > 0 && bSide > 0 && bothZero > 0, "side coverage hole");
    PXG_CHECK_MSG(std::memcmp(&W.io[SPG_IO_OUT], &A.io[SPG_IO_OUT], 30 * SPG_IO_STRIDE * 4) != 0,
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
    PxgDevicePtr dJm = up(A.jm.data(), A.jm.size() * sizeof(A.jm[0]));
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

    SpgParamsHost prm = { uint64_t(dLk), uint64_t(dBt), uint64_t(dCh), uint64_t(dFh),
                          uint64_t(dCp), uint64_t(dFp), uint64_t(dJh), uint64_t(dJr),
                          uint64_t(dJm), uint64_t(dRs), uint64_t(dIo),
                          doFriction ? 1u : 0u, accumErr ? 1u : 0u, SPG_LANES,
                          SPG_RIGID_CONTACT_BIAS };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch static solve pgs");
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
            std::printf("MM staticPgs/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
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
    cmp(dJm, W.jm.data(), W.jm.size() * sizeof(W.jm[0]), "jointMods");
    cmp(dRs, W.rs.data(), W.rs.size() * sizeof(W.rs[0]), "responses");
    cmp(dIo, W.io.data(), W.io.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveStaticPgs)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveInternal.spv", "siStaticPgsTest", 104u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &spgLoader);

    size_t bad = 0;
    bad += spgRun(shim, true, true, 31);
    bad += spgRun(shim, false, true, 33);
    bad += spgRun(shim, true, false, 37);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
