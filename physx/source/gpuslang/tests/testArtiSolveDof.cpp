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

// testArtiSolveDof.cpp -- P8.100b: siSolveDofConstraints, the per-dof
// internal solve of artiSolveInternalConstraintsTGS1T (:1747-1890) plus
// accumulateLinkImpulsesAndLinkVelocities (:1554).
//
// THE ORACLE IS THE ENGINE: every scalar decision inside the loop is a
// PX_CUDA_CALLABLE original called on the host --
// Dy::computeDriveImpulse, Dy::clampDriveImpulse, Dy::computeFrictionImpulse,
// Dy::computeLimitImpulse, and PxgArticulationInternalConstraintData::
// getImplicitDriveDesc.  Only the loop structure and the accumulate helper
// are transcribed.
//
// Coverage: all three motion modes (locked rows SKIPPED, no counter tick);
// both drive arms (maxImpulse > 0 with the RMW writeback vs the force clamp);
// both friction models (old = zero static+viscous effort); limited rows with
// low/high/pullback flavors; velocity-limit hits; four flag rounds including
// the perSubset (isExternalForceEveryStep && posIter) dt switch and a
// doFrictionDrivePosLimit=false round.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "DyCpuGpuArticulation.h"
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

static std::vector<uint8_t> sdLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SD_LANES = 30;
const uint32_t SD_MAX_DOFS = 7;
const float SD_DT = 0.01f, SD_SCENE_DT = 0.033f, SD_INV_DT = 100.0f;
const float SD_ELAPSED = 0.02f, SD_ERP = 0.6f;
const uint32_t SD_IO_STRIDE = 48, SD_IO_OUT = 32 * 48;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t sdJointOffset(uint32_t lane) { return lane % 4u; }
static uint32_t sdDofCount(uint32_t lane) { return 1u + (lane % 3u); }
static uint32_t sdMotion(uint32_t lane, uint32_t dof) { return (lane + dof) % 3u; }

struct SdArena
{
    std::vector<PxgArticulationBlockLinkData> lk;   // ONE entry
    std::vector<PxgArticulationBlockDofData> df;    // SD_MAX_DOFS
    std::vector<float> jf;                          // 32 * SD_MAX_DOFS
    std::vector<float> io;                          // 64 * 48
};

static void sdSeed(SdArena& B, uint32_t salt)
{
    B.lk.resize(1);
    B.df.resize(SD_MAX_DOFS);
    B.jf.resize(32 * SD_MAX_DOFS);
    B.io.resize(64 * SD_IO_STRIDE);
    std::memset(B.lk.data(), 0xAB, sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.io.data(), 0xAB, B.io.size() * 4);

    for (size_t i = 0; i < B.jf.size(); ++i)
        B.jf[i] = 0.5f * fv(uint32_t(i), 200, 0, salt);

    PxgArticulationBlockLinkData& L = B.lk[0];
    for (uint32_t lane = 0; lane < 32; ++lane)
    {
        L.mJointOffset[lane] = sdJointOffset(lane);
        L.mDofs[lane] = PxU8(sdDofCount(lane));
        for (uint32_t j = 0; j < 48; ++j)
            B.io[lane * SD_IO_STRIDE + j] = ((j >= 36) ? 0.1f : 1.0f) * fv(lane, 2, j, salt);
    }

    for (uint32_t d = 0; d < SD_MAX_DOFS; ++d)
    {
        PxgArticulationBlockDofData& D = B.df[d];
        PxgArticulationInternalConstraintData& ic = D.mConstraintData;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            D.mMotion[lane] = PxU8(sdMotion(lane, d));

            storeSpatialVector(ic.mRow0,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(lane, d, 10, salt), fv(lane, d, 11, salt), fv(lane, d, 12, salt)),
                                   PxVec3(fv(lane, d, 13, salt), fv(lane, d, 14, salt), fv(lane, d, 15, salt))),
                               lane);
            storeSpatialVector(ic.mRow1,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(lane, d, 16, salt), fv(lane, d, 17, salt), fv(lane, d, 18, salt)),
                                   PxVec3(fv(lane, d, 19, salt), fv(lane, d, 20, salt), fv(lane, d, 21, salt))),
                               lane);
            storeSpatialVector(ic.mDeltaVA,
                               UnAlignedSpatialVector(
                                   PxVec3(0.3f * fv(lane, d, 22, salt), 0.3f * fv(lane, d, 23, salt), 0.3f * fv(lane, d, 24, salt)),
                                   PxVec3(0.3f * fv(lane, d, 25, salt), 0.3f * fv(lane, d, 26, salt), 0.3f * fv(lane, d, 27, salt))),
                               lane);
            storeSpatialVector(ic.mDeltaVB,
                               UnAlignedSpatialVector(
                                   PxVec3(0.3f * fv(lane, d, 28, salt), 0.3f * fv(lane, d, 29, salt), 0.3f * fv(lane, d, 30, salt)),
                                   PxVec3(0.3f * fv(lane, d, 31, salt), 0.3f * fv(lane, d, 32, salt), 0.3f * fv(lane, d, 33, salt))),
                               lane);

            const float response = 0.6f + 0.3f * std::fabs(fv(lane, d, 34, salt));
            ic.mResponse[lane] = response;
            ic.mRecipResponse[lane] = 1.0f / response;
            ic.mMaxJointVelocity[lane] = 0.4f + 0.2f * std::fabs(fv(lane, d, 35, salt));
            ic.mConstraintMaxImpulse[lane] = 0.6f + 0.3f * std::fabs(fv(lane, d, 36, salt));
            ic.mDriveImpulse[lane] = 0.3f * fv(lane, d, 37, salt);
            ic.mMaxFrictionForce[lane] = 0.4f + 0.2f * std::fabs(fv(lane, d, 38, salt));
            ic.mAccumulatedFrictionImpulse[lane] = 0.2f * fv(lane, d, 39, salt);

            // both drive arms: maxImpulse > 0 for (lane+d)%2, else the force clamp
            ic.mMaxEffort[lane] = ((lane + d) % 2u) ? (30.0f + 10.0f * std::fabs(fv(lane, d, 40, salt))) : 0.0f;
            ic.mMaxActuatorVelocity[lane] = 2.0f + std::fabs(fv(lane, d, 41, salt));
            ic.mSpeedEffortGradient[lane] = ((lane + d) % 5u == 2u) ? 0.0f : 0.02f * std::fabs(fv(lane, d, 42, salt));
            ic.mVelocityDependentResistance[lane] = ((lane + d) % 3u == 1u) ? (4.0f + std::fabs(fv(lane, d, 43, salt))) : 0.0f;

            // both friction models: old = zero static AND viscous effort
            const bool newModel = (lane % 3u) != 0u;
            ic.mStaticFrictionEffort[lane] = newModel ? (8.0f + 4.0f * std::fabs(fv(lane, d, 44, salt))) : 0.0f;
            ic.mDynamicFrictionEffort[lane] = newModel ? (5.0f + 2.0f * std::fabs(fv(lane, d, 45, salt))) : 0.0f;
            ic.mViscousFrictionCoefficient[lane] = (newModel && (lane % 4u == 1u)) ? (0.5f + std::fabs(fv(lane, d, 46, salt))) : 0.0f;

            // limit errors: violated / future-violated / free flavors
            const uint32_t lm = (lane + d) % 4u;
            float lo, hi;
            if (lm == 0) { lo = -0.05f - 0.02f * std::fabs(fv(lane, d, 47, salt)); hi = 0.4f; }
            else if (lm == 1) { lo = 0.004f; hi = 0.5f; }                       // future violation range
            else if (lm == 2) { lo = 0.4f; hi = -0.03f - 0.02f * std::fabs(fv(lane, d, 48, salt)); }
            else { lo = 0.5f; hi = 0.5f; }                                       // free -> pull-back arm
            ic.mLimitError_LowX_highY[lane] = make_float2(lo, hi);
            ic.mLowImpulse[lane] = 0.1f * std::fabs(fv(lane, d, 49, salt));
            ic.mHighImpulse[lane] = -0.1f * std::fabs(fv(lane, d, 50, salt));

            ic.mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW[lane] =
                make_float4(0.3f * fv(lane, d, 51, salt), 0.4f * fv(lane, d, 52, salt),
                            -0.5f * std::fabs(fv(lane, d, 53, salt)), 0.9f + 0.1f * fv(lane, d, 54, salt));
            ic.mTargetPosBias[lane] = 0.2f * fv(lane, d, 55, salt);
        }
    }
}

// ---- mirror ----

static void sdMirror(SdArena& W, uint32_t flags, DofCov& cov)
{
    const bool velIter = (flags & 1u) != 0;
    const bool extEvery = (flags & 2u) != 0;
    const bool doFDPL = (flags & 4u) != 0;
    const bool velBefore = (flags & 8u) != 0;
    const bool residual = (flags & 16u) != 0;

    PxgArticulationBlockLinkData& L = W.lk[0];
    for (uint32_t lane = 0; lane < SD_LANES; ++lane)
    {
        float* fin = &W.io[lane * SD_IO_STRIDE];
        UnAlignedSpatialVector parentV(PxVec3(fin[0], fin[1], fin[2]), PxVec3(fin[3], fin[4], fin[5]));
        UnAlignedSpatialVector childV(PxVec3(fin[6], fin[7], fin[8]), PxVec3(fin[9], fin[10], fin[11]));
        UnAlignedSpatialVector parentDeltaV(PxVec3(fin[12], fin[13], fin[14]), PxVec3(fin[15], fin[16], fin[17]));
        UnAlignedSpatialVector deltaV(PxVec3(fin[18], fin[19], fin[20]), PxVec3(fin[21], fin[22], fin[23]));
        UnAlignedSpatialVector parentImp(PxVec3(fin[24], fin[25], fin[26]), PxVec3(fin[27], fin[28], fin[29]));
        UnAlignedSpatialVector impulse(PxVec3(fin[30], fin[31], fin[32]), PxVec3(fin[33], fin[34], fin[35]));
        const UnAlignedSpatialVector childDelta(PxVec3(fin[36], fin[37], fin[38]), PxVec3(fin[39], fin[40], fin[41]));
        const UnAlignedSpatialVector parentDelta(PxVec3(fin[42], fin[43], fin[44]), PxVec3(fin[45], fin[46], fin[47]));

        const uint32_t jointOffset = L.mJointOffset[lane];
        const uint32_t dofCount = L.mDofs[lane];
        const float* jf = &W.jf[lane * SD_MAX_DOFS];

        RefError err;
        // hoisted to pxgArtiInternalRef.h in P8.100c; this fixture is a user
        refSolveDofLane(W.df.data(), jointOffset, dofCount, jf,
                        SD_DT, SD_SCENE_DT, SD_INV_DT, SD_ELAPSED, SD_ERP,
                        velIter, extEvery, doFDPL, velBefore,
                        childDelta, parentDelta,
                        parentDeltaV, deltaV, parentV, childV, parentImp, impulse,
                        residual, err, lane, &cov);

        float* out = &W.io[SD_IO_OUT + lane * SD_IO_STRIDE];
        out[0] = parentV.top.x; out[1] = parentV.top.y; out[2] = parentV.top.z;
        out[3] = parentV.bottom.x; out[4] = parentV.bottom.y; out[5] = parentV.bottom.z;
        out[6] = childV.top.x; out[7] = childV.top.y; out[8] = childV.top.z;
        out[9] = childV.bottom.x; out[10] = childV.bottom.y; out[11] = childV.bottom.z;
        out[12] = parentDeltaV.top.x; out[13] = parentDeltaV.top.y; out[14] = parentDeltaV.top.z;
        out[15] = parentDeltaV.bottom.x; out[16] = parentDeltaV.bottom.y; out[17] = parentDeltaV.bottom.z;
        out[18] = deltaV.top.x; out[19] = deltaV.top.y; out[20] = deltaV.top.z;
        out[21] = deltaV.bottom.x; out[22] = deltaV.bottom.y; out[23] = deltaV.bottom.z;
        out[24] = parentImp.top.x; out[25] = parentImp.top.y; out[26] = parentImp.top.z;
        out[27] = parentImp.bottom.x; out[28] = parentImp.bottom.y; out[29] = parentImp.bottom.z;
        out[30] = impulse.top.x; out[31] = impulse.top.y; out[32] = impulse.top.z;
        out[33] = impulse.bottom.x; out[34] = impulse.bottom.y; out[35] = impulse.bottom.z;
        out[36] = err.sumOfSquares; out[37] = float(err.counter); out[38] = err.maxError;
    }
}

// ---- run ----

struct SdParamsHost
{
    uint64_t dofBlocks, jointForces, link, io;
    float dt, sceneDt, invDt, elapsedTime, erp;
    uint32_t flagsWord, nbLanes, maxDofs;
};
static_assert(sizeof(SdParamsHost) == 64, "params blob");

static size_t sdRun(PxgCudaShim& shim, uint32_t flags, uint32_t salt)
{
    SdArena A;
    sdSeed(A, salt);

    SdArena W = A;
    DofCov cov;
    sdMirror(W, flags, cov);
    std::printf("[ info ] dof(flags %02x): locked %u, limited %u, free %u; friction old %u/new %u; "
                "drive max %u/force %u; velHits %u\n",
                flags, cov.locked, cov.limited, cov.freeDof, cov.oldModel, cov.newModel,
                cov.maxArm, cov.forceArm, cov.velHits);
    PXG_CHECK_MSG(cov.locked > 0 && cov.limited > 0 && cov.freeDof > 0, "motion coverage hole");
    if (flags & 4u)
        PXG_CHECK_MSG(cov.oldModel > 0 && cov.newModel > 0 && cov.maxArm > 0 && cov.forceArm > 0,
                      "drive/friction coverage hole");
    if (flags & 8u)
        PXG_CHECK_MSG(cov.velHits > 0, "velocity-limit coverage hole");
    PXG_CHECK_MSG(std::memcmp(&W.df[0], &A.df[0], W.df.size() * sizeof(W.df[0])) != 0
                      || !(flags & 4u),
                  "vacuous");

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dDf = up(A.df.data(), A.df.size() * sizeof(A.df[0]));
    PxgDevicePtr dJf = up(A.jf.data(), A.jf.size() * 4);
    PxgDevicePtr dLk = up(A.lk.data(), sizeof(A.lk[0]));
    PxgDevicePtr dIo = up(A.io.data(), A.io.size() * 4);
    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SdParamsHost prm = { uint64_t(dDf), uint64_t(dJf), uint64_t(dLk), uint64_t(dIo),
                         SD_DT, SD_SCENE_DT, SD_INV_DT, SD_ELAPSED, SD_ERP,
                         flags, SD_LANES, SD_MAX_DOFS };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch dof solve");
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
            std::printf("MM dof/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dJf, W.jf.data(), W.jf.size() * 4, "jointForces");
    cmp(dLk, W.lk.data(), sizeof(W.lk[0]), "linkBlock");
    cmp(dIo, W.io.data(), W.io.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveDof)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveInternal.spv", "siDofTest", 64u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &sdLoader);

    size_t bad = 0;
    bad += sdRun(shim, 4u | 8u | 16u, 41);        // posIter, doFDPL, velBefore, residual
    bad += sdRun(shim, 1u | 4u | 16u, 43);        // velIter, doFDPL, residual
    bad += sdRun(shim, 2u | 4u | 8u | 16u, 47);   // posIter + extEvery (perSubset), doFDPL, velBefore
    bad += sdRun(shim, 8u | 16u, 53);             // doFDPL OFF: velocity limit only;
                                                  // residual ON so zero-deltaF rows tick the counter
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
