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

// P8.122a: `solveExtContactsBlock` (solver.cuh:54-320) and `solveExt1DBlock`
// (solverBlock.cuh:208-326) on the ENGINE BLOCK LAYOUT — the two PGS bodies
// artiSolveBlockPartition dispatches through batch.constraintType.  The TGS
// twins landed in P8.99a (testArtiSolveExtTGS.cpp); this is the same fixture
// arrangement with the PGS block types and mirrors
// (refSolveExtContactPGS / refSolveExt1DPGS in pxgArtiExtSolveRef.h).
//
// THE ORACLE: the 1D constants come from the ENGINE's host-callable
// Dy::compute1dConstraintSolverConstantsPGS / Dy::computeRecipUnitResponse;
// the contact-side computeContactCoefficients is __device__-only and
// transcribed (pinned previously by the rigid-PGS solve battery).
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//  * ONE batch per family, 32 lanes as 32 independent slots: numNormalConstr
//    0..6 with numFrictionConstr 0/2/4 (EVEN only — the PGS prep writes
//    aCount*2, no torsional; zero-contact lanes carry ZERO frictions, the
//    prep's early-out invariant the un-pipelining relies on);
//  * compliant lanes (restitution < 0) with and without the accel-spring
//    flag; warm-start appliedForce; maxImpulse clamping; the minNormalForce
//    floor (some lanes' floor EXCEEDS the accumulated impulse);
//  * friction clamping both ways per ROW (the PGS independent clamp);
//  * 1D rows hard / spring / accel-spring with iMul != 1, min/max impulse
//    clamps both active and not, u16 rowCounts 0..6;
//  * three contact rounds: (doFriction, ref0, ref1) = (1,1,1),
//    (1,0.75,1.25), (0,1,1) — the doFriction=0 round must leave the friction
//    pool and broken flags POISONED;
//  * three 1D rounds: residual reporting on/(refs)/off — the off round must
//    leave mod.residual poisoned;
//  * P9.89 (artiSolveExtPgsMaxErrorNaN): contact rows with a NaN residual (a NaN compliant
//    coefficient), on a lane's last residual row or mid-chain, so the error accumulator's
//    maxError must follow the CUDA's selectMax chain (solverResidual.cuh:57); an FMax twin
//    must differ;
//  * every block byte compared bit-exactly; untouched fields at the seed.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DySolverConstraintTypes.h"
#include "DyCpuGpuArticulation.h"
#include "DyCpuGpu1dConstraint.h"
#include "PxgSolverConstraintDesc.h"
#include "PxgConstraintBlock.h"
#include "PxgSolverConstraintBlock1D.h"
#include "pxgArtiExtSolveRef.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
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

static std::vector<uint8_t> svpLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SVP_LANES = 32;
const uint32_t SVP_MAX_CONTACTS = 6, SVP_MAX_FRICTIONS = 4, SVP_MAX_ROWS = 6;
const uint32_t SVP_START_CONSTRAINT = 2, SVP_START_FRICTION = 3, SVP_START_RESP = 5;
const uint32_t SVP_NB_RESP = SVP_START_RESP + SVP_MAX_CONTACTS + SVP_MAX_FRICTIONS + 2;

static_assert(sizeof(PxgBlockSolverContactHeader) == 1920, "PGS contact header");
static_assert(sizeof(PxgBlockSolverFrictionHeader) == 1408, "PGS friction header");
static_assert(sizeof(PxgBlockSolverContactPoint) == 1664, "PGS contact point");
static_assert(sizeof(PxgBlockSolverContactFriction) == 1408, "PGS contact friction");
static_assert(sizeof(PxgBlockSolverConstraint1DHeader) == 1664, "PGS 1D header");
static_assert(sizeof(PxgBlockSolverConstraint1DCon) == 2304, "PGS 1D con");
static_assert(sizeof(PxgBlockSolverConstraint1DMod) == 1024, "PGS 1D mod");

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t svpNumContacts(uint32_t l) { return l % (SVP_MAX_CONTACTS + 1); }

static uint32_t svpNumFrictions(uint32_t l)
{
    // EVEN only, and ZERO when there are no contacts (the prep invariant)
    if (svpNumContacts(l) == 0) return 0;
    return ((l * 7 + 3) % 3) * 2;   // 0 / 2 / 4
}

struct LaneIo
{
    UnAlignedSpatialVector vel0, vel1;
};

static LaneIo svpLaneIo(uint32_t lane, uint32_t salt)
{
    LaneIo io;
    io.vel0 = UnAlignedSpatialVector(
        PxVec3(fv(lane, 0, 0, salt), fv(lane, 0, 1, salt), fv(lane, 0, 2, salt)),
        PxVec3(fv(lane, 0, 3, salt), fv(lane, 0, 4, salt), fv(lane, 0, 5, salt)));
    io.vel1 = UnAlignedSpatialVector(
        PxVec3(fv(lane, 1, 0, salt), fv(lane, 1, 1, salt), fv(lane, 1, 2, salt)),
        PxVec3(fv(lane, 1, 3, salt), fv(lane, 1, 4, salt), fv(lane, 1, 5, salt)));
    return io;
}

static void svpSeedResponses(std::vector<PxgArticulationBlockResponse>& rsp, uint32_t salt)
{
    rsp.resize(SVP_NB_RESP);
    std::memset(rsp.data(), 0xAB, rsp.size() * sizeof(rsp[0]));
    for (uint32_t r = 0; r < SVP_NB_RESP; ++r)
        for (uint32_t l = 0; l < 32; ++l)
        {
            rsp[r].deltaRALin_x[l] = 0.2f * fv(r, l, 0, salt);
            rsp[r].deltaRALin_y[l] = 0.2f * fv(r, l, 1, salt);
            rsp[r].deltaRALin_z[l] = 0.2f * fv(r, l, 2, salt);
            rsp[r].deltaRAAng_x[l] = 0.2f * fv(r, l, 3, salt);
            rsp[r].deltaRAAng_y[l] = 0.2f * fv(r, l, 4, salt);
            rsp[r].deltaRAAng_z[l] = 0.2f * fv(r, l, 5, salt);
            rsp[r].deltaRBLin_x[l] = 0.2f * fv(r, l, 6, salt);
            rsp[r].deltaRBLin_y[l] = 0.2f * fv(r, l, 7, salt);
            rsp[r].deltaRBLin_z[l] = 0.2f * fv(r, l, 8, salt);
            rsp[r].deltaRBAng_x[l] = 0.2f * fv(r, l, 9, salt);
            rsp[r].deltaRBAng_y[l] = 0.2f * fv(r, l, 10, salt);
            rsp[r].deltaRBAng_z[l] = 0.2f * fv(r, l, 11, salt);
        }
}

static void svpSeedBatch(PxgBlockConstraintBatch& b)
{
    std::memset(&b, 0xAB, sizeof(b));
    b.mConstraintBatchIndex = 0;
    b.startConstraintIndex = SVP_START_CONSTRAINT;
    b.startFrictionIndex = SVP_START_FRICTION;
    b.mArticulationResponseIndex = SVP_START_RESP;
}

struct SvpIoBuf
{
    std::vector<float> data;   // 2 * 32 lanes * 32 floats
    void init(uint32_t salt)
    {
        data.assign(2 * 32 * 32, -555.f);
        for (uint32_t l = 0; l < SVP_LANES; ++l)
        {
            const LaneIo io = svpLaneIo(l, salt);
            float* f = &data[l * 32];
            const UnAlignedSpatialVector* vs[2] = { &io.vel0, &io.vel1 };
            for (int v = 0; v < 2; ++v)
            {
                f[v * 6 + 0] = vs[v]->top.x; f[v * 6 + 1] = vs[v]->top.y; f[v * 6 + 2] = vs[v]->top.z;
                f[v * 6 + 3] = vs[v]->bottom.x; f[v * 6 + 4] = vs[v]->bottom.y; f[v * 6 + 5] = vs[v]->bottom.z;
            }
        }
    }
};

struct SvpParamsHost
{
    uint64_t batch, h0, h1, p0, p1, responses, io;
    float ref0, ref1;
    uint32_t nbLanes, flagsWord;
};
static_assert(sizeof(SvpParamsHost) == 72, "params blob");

static size_t svpRunContact(PxgCudaShim& shim, bool doFriction, float ref0, float ref1,
                            bool nanResiduals = false)
{
    PxgBlockConstraintBatch batch;
    svpSeedBatch(batch);

    std::vector<PxgBlockSolverContactHeader> ch(1);
    std::vector<PxgBlockSolverFrictionHeader> fh(1);
    std::vector<PxgBlockSolverContactPoint> cp(SVP_START_CONSTRAINT + SVP_MAX_CONTACTS);
    std::vector<PxgBlockSolverContactFriction> fp(SVP_START_FRICTION + SVP_MAX_FRICTIONS);
    std::vector<PxgArticulationBlockResponse> rsp;
    std::memset(ch.data(), 0xAB, sizeof(ch[0]));
    std::memset(fh.data(), 0xAB, sizeof(fh[0]));
    std::memset(cp.data(), 0xAB, cp.size() * sizeof(cp[0]));
    std::memset(fp.data(), 0xAB, fp.size() * sizeof(fp[0]));
    svpSeedResponses(rsp, 700);

    uint32_t nCompliant = 0, nZeroContacts = 0, nFloor = 0, nWithFriction = 0;
    for (uint32_t l = 0; l < 32; ++l)
    {
        const uint32_t nc = svpNumContacts(l);
        const uint32_t nf = svpNumFrictions(l);
        ch[0].numNormalConstr[l] = nc;
        fh[0].numFrictionConstr[l] = nf;
        if (nc == 0) ++nZeroContacts;
        if (nf) ++nWithFriction;
        ch[0].invMass0_1_angDom0_1[l] = make_float4(0.5f + 0.4f * fv(l, 0, 0, 71),
                                                    0.5f + 0.4f * fv(l, 0, 1, 71),
                                                    0.5f + 0.4f * fv(l, 0, 2, 71),
                                                    0.5f + 0.4f * fv(l, 0, 3, 71));
        PxVec3 n(fv(l, 1, 0, 72), fv(l, 1, 1, 72), fv(l, 1, 2, 72));
        n = n.getNormalized();
        ch[0].normal_staticFriction[l] = make_float4(n.x, n.y, n.z,
                                                     0.6f + 0.3f * std::fabs(fv(l, 1, 3, 72)));
        // every fifth lane a LARGE floor so it exceeds the accumulated force
        const bool bigFloor = (l % 5u) == 2u;
        if (bigFloor) ++nFloor;
        ch[0].minNormalForce[l] = bigFloor ? 2.0f + std::fabs(fv(l, 2, 0, 73))
                                           : 0.02f * std::fabs(fv(l, 2, 0, 73));
        ch[0].flags[l] = (l % 3u == 0u) ? COMPLIANT_ACCEL : 0u;
        const bool compliant = (l % 4u) == 1u;
        if (compliant) ++nCompliant;
        ch[0].restitution[l] = compliant ? -600.0f - 10.0f * float(l)
                                         : 0.3f * std::fabs(fv(l, 2, 2, 73));
        ch[0].cfm[l] = 0.01f * std::fabs(fv(l, 2, 3, 73));
        ch[0].accumNormalForce[l] = fv(l, 2, 1, 73);   // NOT touched by the solve
        ch[0].forceWritebackOffset[l] = 0xABABABABu;   // NOT touched
        fh[0].dynamicFriction[l] = 0.4f + 0.3f * std::fabs(fv(l, 3, 1, 74));
        PxVec3 f0(fv(l, 4, 0, 75), fv(l, 4, 1, 75), fv(l, 4, 2, 75));
        PxVec3 f1(fv(l, 5, 0, 75), fv(l, 5, 1, 75), fv(l, 5, 2, 75));
        f0 = f0.getNormalized(); f1 = f1.getNormalized();
        fh[0].frictionNormals[0][l] = make_float4(f0.x, f0.y, f0.z, 0.f);
        fh[0].frictionNormals[1][l] = make_float4(f1.x, f1.y, f1.z, 0.f);
        // broken stays POISONED: the doFriction=false round must not write it

        for (uint32_t i = 0; i < SVP_MAX_CONTACTS; ++i)
        {
            PxgBlockSolverContactPoint& c = cp[SVP_START_CONSTRAINT + i];
            c.raXn_targetVelocity[l] = make_float4(fv(l, 10 + i, 0, 76), fv(l, 10 + i, 1, 76),
                                                   fv(l, 10 + i, 2, 76), 0.2f * fv(l, 10 + i, 3, 76));
            c.rbXn_maxImpulse[l] = make_float4(fv(l, 20 + i, 0, 77), fv(l, 20 + i, 1, 77),
                                               fv(l, 20 + i, 2, 77),
                                               0.3f + 0.3f * std::fabs(fv(l, 20 + i, 3, 77)));
            c.appliedForce[l] = 0.2f * std::fabs(fv(l, 30 + i, 2, 78));
            c.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 40 + i, 0, 79));
            c.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 40 + i, 1, 79));
            // compliant lanes need a plausible a > 0 in coeff0
            c.coeff0[l] = (l % 4u) == 1u ? 0.5f + 0.4f * std::fabs(fv(l, 30 + i, 0, 78))
                                         : 0.4f * fv(l, 30 + i, 0, 78);
            c.coeff1[l] = 0.4f * fv(l, 30 + i, 1, 78);
        }
        for (uint32_t i = 0; i < SVP_MAX_FRICTIONS; ++i)
        {
            PxgBlockSolverContactFriction& f = fp[SVP_START_FRICTION + i];
            f.raXn_bias[l] = make_float4(fv(l, 50 + i, 0, 80), fv(l, 50 + i, 1, 80),
                                         fv(l, 50 + i, 2, 80), 0.3f * fv(l, 50 + i, 3, 80));
            f.rbXn_targetVelW[l] = make_float4(fv(l, 60 + i, 0, 81), fv(l, 60 + i, 1, 81),
                                               fv(l, 60 + i, 2, 81), 0.1f * fv(l, 60 + i, 3, 81));
            f.appliedForce[l] = 0.15f * fv(l, 70 + i, 0, 82);
            f.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 70 + i, 1, 82));
            f.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 70 + i, 2, 82));
        }
    }
    std::printf("[ info ] svp contact(doF %d, ref %g/%g): compliant %u, zero-contact %u, "
                "big-floor %u, with-friction %u\n",
                int(doFriction), double(ref0), double(ref1), nCompliant, nZeroContacts,
                nFloor, nWithFriction);
    PXG_CHECK_MSG(nCompliant > 5 && nZeroContacts > 0 && nFloor > 4 && nWithFriction > 10,
                  "contact coverage hole");

    // P9.89 (npArtiSolveExtPGS svpAccumulateError): a NaN residual inside a lane's row chain.
    // The CUDA's maxError = PxMax(maxError, PxAbs(e)) (solverResidual.cuh:57) is selectMax, so a
    // NaN row REPLACES the running maximum and the next row's |e| replaces the NaN: the lane ends
    // with the maximum over the rows AFTER the NaN, or with NaN when the NaN row is the last one.
    // The port's old GLSL max() (llvmpipe's stamped FMax ignores a NaN) kept the maximum over all
    // finite rows.  The NaN comes from a NaN compliant coefficient a = coeff0 on a compliant lane:
    // velMultiplier = x * a * m and impulseMul = 1 - x are NaN, the CUDA's fminf(NaN, maxImpulse)
    // makes the row's newForce exactly maxImpulse -- a finite deltaF, so the velocities, impulses
    // and later rows stay finite and only e = deltaF / NaN is NaN.  Even lanes take the NaN on the
    // LAST contact row, odd lanes on a middle row; every lane with contacts becomes compliant (with
    // the compliant lanes' a > 0 on its other rows).
    uint32_t nNanRows = 0;
    if (nanResiduals)
    {
        for (uint32_t l = 0; l < SVP_LANES; ++l)
        {
            const uint32_t nc = svpNumContacts(l);
            if (nc == 0u)
                continue;
            const uint32_t k = (l & 1u) ? (nc - 1u) / 2u : nc - 1u;
            ch[0].restitution[l] = -600.0f - 10.0f * float(l);
            for (uint32_t i = 0; i < SVP_MAX_CONTACTS; ++i)
                cp[SVP_START_CONSTRAINT + i].coeff0[l] = 0.5f + 0.4f * std::fabs(fv(l, 30 + i, 0, 78));
            cp[SVP_START_CONSTRAINT + k].coeff0[l] = std::numeric_limits<float>::quiet_NaN();
            ++nNanRows;
        }
    }

    SvpIoBuf io;
    io.init(90);

    // mirror; errVariant selects gExtErrMaxVariant (0 = the CUDA)
    struct ContactImage
    {
        std::vector<PxgBlockSolverContactHeader> ch;
        std::vector<PxgBlockSolverFrictionHeader> fh;
        std::vector<PxgBlockSolverContactPoint> cp;
        std::vector<PxgBlockSolverContactFriction> fp;
        std::vector<float> io;
    };
    auto mirror = [&](int errVariant) {
        ContactImage w{ ch, fh, cp, fp, io.data };
        gExtErrMaxVariant = errVariant;
        for (uint32_t l = 0; l < SVP_LANES; ++l)
        {
            LaneIo lio = svpLaneIo(l, 90);
            UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
            RefError err;
            refSolveExtContactPGS(batch, lio.vel0, lio.vel1, doFriction, l, w.ch.data(),
                                  w.fh.data(), w.cp.data(), w.fp.data(), rsp.data(), imp0, imp1,
                                  &err, ref0, ref1);
            float* o = &w.io[32 * 32 + l * 32];
            const UnAlignedSpatialVector* vs[4] = { &lio.vel0, &lio.vel1, &imp0, &imp1 };
            for (int v = 0; v < 4; ++v)
            {
                o[v * 6 + 0] = vs[v]->top.x; o[v * 6 + 1] = vs[v]->top.y; o[v * 6 + 2] = vs[v]->top.z;
                o[v * 6 + 3] = vs[v]->bottom.x; o[v * 6 + 4] = vs[v]->bottom.y; o[v * 6 + 5] = vs[v]->bottom.z;
            }
            o[24] = err.sumOfSquares; o[25] = float(err.counter); o[26] = err.maxError;
        }
        gExtErrMaxVariant = 0;
        return w;
    };
    const ContactImage want = mirror(0);
    const std::vector<PxgBlockSolverContactHeader>& wCh = want.ch;
    const std::vector<PxgBlockSolverFrictionHeader>& wFh = want.fh;
    const std::vector<PxgBlockSolverContactPoint>& wCp = want.cp;
    const std::vector<PxgBlockSolverContactFriction>& wFp = want.fp;
    const std::vector<float>& wantIo = want.io;

    // P9.89 twin: the old FMax chain may change ONLY maxError words (io out slot 26), and on the
    // NaN rounds it must change some: lanes whose CUDA maxError is the trailing NaN, and lanes
    // whose CUDA maxError restarted after the NaN below an earlier row's |e|.
    std::vector<float> errTwinIo;
    if (nanResiduals)
    {
        const ContactImage twin = mirror(1);
        errTwinIo = twin.io;
        uint32_t nanSumLanes = 0, cudaNanMax = 0, restartedMax = 0, otherIoWords = 0;
        for (uint32_t l = 0; l < SVP_LANES; ++l)
        {
            const float* o = &wantIo[32 * 32 + l * 32];
            const float* t = &twin.io[32 * 32 + l * 32];
            if (std::isnan(o[24])) ++nanSumLanes;
            for (uint32_t k = 0; k < 32; ++k)
                if (k != 26u && std::memcmp(&o[k], &t[k], 4u) != 0) ++otherIoWords;
            if (std::memcmp(&o[26], &t[26], 4u) == 0)
                continue;
            if (std::isnan(o[26]) && !std::isnan(t[26])) ++cudaNanMax;
            else if (!std::isnan(o[26]) && !std::isnan(t[26]) && o[26] < t[26]) ++restartedMax;
            else ++otherIoWords;
        }
        for (uint32_t k = 0; k < 32 * 32; ++k)
            if (std::memcmp(&wantIo[k], &twin.io[k], 4u) != 0) ++otherIoWords;
        const bool blocksSame = std::memcmp(twin.ch.data(), wCh.data(), sizeof(wCh[0])) == 0 &&
                                std::memcmp(twin.fh.data(), wFh.data(), sizeof(wFh[0])) == 0 &&
                                std::memcmp(twin.cp.data(), wCp.data(), wCp.size() * sizeof(wCp[0])) == 0 &&
                                std::memcmp(twin.fp.data(), wFp.data(), wFp.size() * sizeof(wFp[0])) == 0;
        std::printf("[ info ] svp contact(doF %d, ref %g/%g): %u NaN-residual rows, %u lanes with a"
                    " NaN sumOfSquares; FMax twin: %u maxError trailing NaN -> finite, %u restarted"
                    " maxError raised, %u other words, blocks %s\n",
                    int(doFriction), double(ref0), double(ref1), nNanRows, nanSumLanes, cudaNanMax,
                    restartedMax, otherIoWords, blocksSame ? "same" : "DIFFER");
        PXG_CHECK_MSG(nNanRows == 27u && nanSumLanes == nNanRows && cudaNanMax >= 3u &&
                          restartedMax >= 3u && otherIoWords == 0u && blocksSame,
                      "P9.89: the NaN-residual lanes lost their selectMax coverage (%u rows, %u NaN"
                      " sums, %u trailing NaN, %u restarted, %u other words)", nNanRows, nanSumLanes,
                      cudaNanMax, restartedMax, otherIoWords);
    }

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dB = up(&batch, sizeof(batch));
    PxgDevicePtr dCh = up(ch.data(), sizeof(ch[0]));
    PxgDevicePtr dFh = up(fh.data(), sizeof(fh[0]));
    PxgDevicePtr dCp = up(cp.data(), cp.size() * sizeof(cp[0]));
    PxgDevicePtr dFp = up(fp.data(), fp.size() * sizeof(fp[0]));
    PxgDevicePtr dR = up(rsp.data(), rsp.size() * sizeof(rsp[0]));
    PxgDevicePtr dIo = up(io.data.data(), io.data.size() * 4);
    if (!dB || !dCh || !dFh || !dCp || !dFp || !dR || !dIo)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    const uint32_t flagsWord = 1u | (doFriction ? 2u : 0u);
    SvpParamsHost prm = { uint64_t(dB), uint64_t(dCh), uint64_t(dFh), uint64_t(dCp),
                          uint64_t(dFp), uint64_t(dR), uint64_t(dIo),
                          ref0, ref1, SVP_LANES, flagsWord };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svpContactTest");
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
            std::printf("MM contact/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n,
                        first);
            bad += mm;
        }
    };
    cmp(dCh, wCh.data(), sizeof(wCh[0]), "contactHeader");
    cmp(dFh, wFh.data(), sizeof(wFh[0]), "frictionHeader");
    cmp(dCp, wCp.data(), wCp.size() * sizeof(wCp[0]), "contactPoints");
    cmp(dFp, wFp.data(), wFp.size() * sizeof(wFp[0]), "frictionPoints");
    cmp(dIo, wantIo.data(), wantIo.size() * 4, "io");
    if (!errTwinIo.empty())
    {
        std::vector<float> g(errTwinIo.size());
        shim.memcpyDtoH(g.data(), dIo, g.size() * 4);
        if (std::memcmp(g.data(), errTwinIo.data(), g.size() * 4) == 0)
            std::printf("MM contact/io: the device reproduces the FMax twin (GLSL max() chain)\n");
    }

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

static size_t svpRun1D(PxgCudaShim& shim, bool residual, float ref0, float ref1)
{
    PxgBlockConstraintBatch batch;
    svpSeedBatch(batch);

    std::vector<PxgBlockSolverConstraint1DHeader> hd(1);
    std::vector<PxgBlockSolverConstraint1DCon> rows(SVP_START_CONSTRAINT + SVP_MAX_ROWS);
    std::vector<PxgBlockSolverConstraint1DMod> mods(SVP_START_CONSTRAINT + SVP_MAX_ROWS);
    std::vector<PxgArticulationBlockResponse> rsp;
    std::memset(hd.data(), 0xAB, sizeof(hd[0]));
    std::memset(rows.data(), 0xAB, rows.size() * sizeof(rows[0]));
    std::memset(mods.data(), 0xAB, mods.size() * sizeof(mods[0]));
    svpSeedResponses(rsp, 701);

    uint32_t nSpring = 0, nAccel = 0, nHard = 0, nClampMin = 0, nClampMax = 0;
    for (uint32_t l = 0; l < 32; ++l)
    {
        const uint32_t rc = (l * 5 + 1) % (SVP_MAX_ROWS + 1);   // 0..6
        hd[0].rowCounts[l] = PxU16(rc);
        hd[0].breakable[l] = PxU16(l & 1);                       // NOT read by the solve
        hd[0].invMass0D0[l] = 0.5f + 0.4f * std::fabs(fv(l, 0, 3, 83));
        hd[0].invMass1D1[l] = 0.5f + 0.4f * std::fabs(fv(l, 1, 3, 83));
        hd[0].invInertiaScale0[l] = 0.6f + 0.3f * fv(l, 2, 0, 84);
        hd[0].invInertiaScale1[l] = 0.6f + 0.3f * fv(l, 2, 1, 84);
        hd[0].cfm[l] = 0.01f * std::fabs(fv(l, 2, 2, 84));

        for (uint32_t i = 0; i < SVP_MAX_ROWS; ++i)
        {
            PxgBlockSolverConstraint1DCon& c = rows[SVP_START_CONSTRAINT + i];
            PxgBlockSolverConstraint1DMod& m = mods[SVP_START_CONSTRAINT + i];
            const uint32_t mode = (l + i) % 3u;   // hard / spring / accel-spring
            uint32_t flags = 0;
            if (mode == 1) { flags = DY_FLAG_SPRING; ++nSpring; }
            else if (mode == 2) { flags = DY_FLAG_SPRING | DY_FLAG_ACCEL; ++nAccel; }
            else ++nHard;

            // narrow impulse windows so both clamps fire on some rows
            const bool narrow = ((l + i) % 4u) == 3u;
            const float minImp = narrow ? -0.02f : -0.6f - 0.3f * std::fabs(fv(l, 60 + i, 0, 88));
            const float maxImp = narrow ? 0.02f : 0.6f + 0.3f * std::fabs(fv(l, 60 + i, 1, 88));
            if (narrow) { ++nClampMin; ++nClampMax; }

            c.lin0XYZ_minImpulse[l] = make_float4(fv(l, 10 + i, 0, 86), fv(l, 10 + i, 1, 86),
                                                  fv(l, 10 + i, 2, 86), minImp);
            c.lin1XYZ_maxImpulse[l] = make_float4(fv(l, 20 + i, 0, 86), fv(l, 20 + i, 1, 86),
                                                  fv(l, 20 + i, 2, 86), maxImp);
            c.ang0XYZ_resp0[l] = make_float4(fv(l, 30 + i, 0, 86), fv(l, 30 + i, 1, 86),
                                             fv(l, 30 + i, 2, 86),
                                             0.4f + 0.3f * std::fabs(fv(l, 30 + i, 3, 87)));
            c.ang1XYZ_resp1[l] = make_float4(fv(l, 40 + i, 0, 86), fv(l, 40 + i, 1, 86),
                                             fv(l, 40 + i, 2, 86),
                                             0.4f + 0.3f * std::fabs(fv(l, 40 + i, 3, 87)));
            c.initJointSpeed[l] = 0.3f * fv(l, 50 + i, 3, 87);

            m.flags[l] = flags;
            m.coeff0[l] = (mode != 0u) ? 0.5f + 0.4f * std::fabs(fv(l, i, 0, 85))
                                       : 0.1f * fv(l, i, 0, 85);
            m.coeff1[l] = 0.3f * fv(l, i, 1, 85);
            m.appliedForce[l] = 0.2f * fv(l, 60 + i, 3, 88);
            // residual stays POISONED: only the reporting rounds may write it
        }
    }
    std::printf("[ info ] svp 1d(res %d, ref %g/%g): hard %u, spring %u, accel %u, narrow %u\n",
                int(residual), double(ref0), double(ref1), nHard, nSpring, nAccel, nClampMin);
    PXG_CHECK_MSG(nHard > 5 && nSpring > 5 && nAccel > 5 && nClampMin > 5,
                  "1d row-mode coverage hole");

    SvpIoBuf io;
    io.init(91);

    // mirror
    std::vector<PxgBlockSolverConstraint1DCon> wRows = rows;
    std::vector<PxgBlockSolverConstraint1DMod> wMods = mods;
    std::vector<float> wantIo = io.data;
    for (uint32_t l = 0; l < SVP_LANES; ++l)
    {
        LaneIo lio = svpLaneIo(l, 91);
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        refSolveExt1DPGS(batch, lio.vel0, lio.vel1, l, hd.data(), wRows.data(), wMods.data(),
                         &rsp[SVP_START_RESP], imp0, imp1, residual, ref0, ref1);
        float* o = &wantIo[32 * 32 + l * 32];
        const UnAlignedSpatialVector* vs[4] = { &lio.vel0, &lio.vel1, &imp0, &imp1 };
        for (int v = 0; v < 4; ++v)
        {
            o[v * 6 + 0] = vs[v]->top.x; o[v * 6 + 1] = vs[v]->top.y; o[v * 6 + 2] = vs[v]->top.z;
            o[v * 6 + 3] = vs[v]->bottom.x; o[v * 6 + 4] = vs[v]->bottom.y; o[v * 6 + 5] = vs[v]->bottom.z;
        }
        o[24] = 0.f; o[25] = 0.f; o[26] = 0.f;   // the 1D entry writes a zero error
    }

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dB = up(&batch, sizeof(batch));
    PxgDevicePtr dHd = up(hd.data(), sizeof(hd[0]));
    PxgDevicePtr dRows = up(rows.data(), rows.size() * sizeof(rows[0]));
    PxgDevicePtr dMods = up(mods.data(), mods.size() * sizeof(mods[0]));
    PxgDevicePtr dR = up(rsp.data(), rsp.size() * sizeof(rsp[0]));
    PxgDevicePtr dIo = up(io.data.data(), io.data.size() * 4);
    if (!dB || !dHd || !dRows || !dMods || !dR || !dIo)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SvpParamsHost prm = { uint64_t(dB), uint64_t(dHd), uint64_t(dRows), uint64_t(dMods), 0,
                          uint64_t(dR), uint64_t(dIo),
                          ref0, ref1, SVP_LANES, residual ? 1u : 0u };
    PXG_CHECK_MSG(shim.launchKernel(1, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svp1DTest");
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
            std::printf("MM 1d/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dHd, hd.data(), sizeof(hd[0]), "header");   // read-only for the solve
    cmp(dRows, wRows.data(), wRows.size() * sizeof(wRows[0]), "rows");
    cmp(dMods, wMods.data(), wMods.size() * sizeof(wMods[0]), "mods");
    cmp(dIo, wantIo.data(), wantIo.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveExtPgs)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveExtPGS.spv", "svpContactTest", 72u, 0 },
        { "npArtiSolveExtPGS.spv", "svp1DTest", 72u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &svpLoader);

    size_t bad = 0;
    bad += svpRunContact(shim, true, 1.0f, 1.0f);
    bad += svpRunContact(shim, true, 0.75f, 1.25f);
    bad += svpRunContact(shim, false, 1.0f, 1.0f);
    bad += svpRun1D(shim, true, 1.0f, 1.0f);
    bad += svpRun1D(shim, true, 0.75f, 1.25f);
    bad += svpRun1D(shim, false, 1.0f, 1.0f);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}

// P9.89: the contact rounds again with NaN residual rows (svpRunContact nanResiduals): the
// maxError chain must be the CUDA's selectMax (solverResidual.cuh:57), not GLSL max().
PXG_TEST(artiSolveExtPgsMaxErrorNaN)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveExtPGS.spv", "svpContactTest", 72u, 0 },
        { "npArtiSolveExtPGS.spv", "svp1DTest", 72u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &svpLoader);

    size_t bad = 0;
    bad += svpRunContact(shim, true, 1.0f, 1.0f, true);
    bad += svpRunContact(shim, true, 0.75f, 1.25f, true);
    bad += svpRunContact(shim, false, 1.0f, 1.0f, true);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
