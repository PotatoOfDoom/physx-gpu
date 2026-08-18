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

// P8.99a: `solveExtContactBlockTGS` (solver.cuh:430-735) and
// `solveExt1DBlockTGS` (solverBlockTGS.cuh:932-1089) on the ENGINE BLOCK
// LAYOUT — the two bodies the articulation self-constraint and partition
// solves dispatch through batch.constraintType.
//
// THE ORACLE: transcriptions over the ENGINE's own host-includable block
// structs (PxgConstraintBlock.h / PxgSolverConstraintBlock1D.h /
// PxgSolverConstraintDesc.h), with the 1D constants computed by the ENGINE's
// PX_CUDA_CALLABLE Dy::compute1dConstraintSolverConstantsTGS /
// Dy::computeRecipUnitResponse — the strongest available oracle for the 1D
// path.  The compliant-contact coefficient pair is transcribed (its reduced
// overload is __device__-only), pinned previously by the tranche-188 battery.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//  * ONE batch per family, 32 lanes as 32 independent constraint slots, with
//    per-lane counts covering: numNormalConstr 0..6, totalFrictionConstr
//    0..5 INCLUDING odd counts (the torsional tail), 1D rowCount 0..6;
//  * per-lane flag mixes: compliant contacts (restitutionXdt < 0) with and
//    without the acceleration-spring bit; 1D rows as hard / inequality /
//    spring / acceleration-spring / bounce (coeff2 != 0 in non-spring mode);
//  * non-zero startConstraintIndex / startFrictionIndex /
//    mArticulationResponseIndex so the three index bases are pinned;
//  * friction clamping both ways (the combined-magnitude sqrt clamp), the
//    minNormalForce floor, maxImpulse clamping;
//  * two rounds: refs (1, 1) and refs (0.75, 1.25) — every ref site scales;
//  * the mutated block fields (appliedForce, broken, residual) and the io
//    outputs (velocities, impulses, error accumulator) compared byte for
//    byte; untouched block bytes must come back at the seed.

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

static std::vector<uint8_t> svxLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SVX_LANES = 32;
const uint32_t SVX_MAX_CONTACTS = 6, SVX_MAX_FRICTIONS = 6, SVX_MAX_ROWS = 6;
const uint32_t SVX_START_CONSTRAINT = 2, SVX_START_FRICTION = 3, SVX_START_RESP = 5;
const uint32_t SVX_NB_RESP = SVX_START_RESP + SVX_MAX_CONTACTS + SVX_MAX_FRICTIONS + 1 + 2;
const float SVX_ELAPSED = 0.011f, SVX_MINPEN = -0.02f;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

// per-lane spatial io (the shader's io buffer layout)
struct LaneIo
{
    UnAlignedSpatialVector vel0, vel1, d0, d1;
    PxQuat dq0, dq1;
};

static LaneIo svxLaneIo(uint32_t lane, uint32_t salt)
{
    LaneIo io;
    io.vel0 = UnAlignedSpatialVector(
        PxVec3(fv(lane, 0, 0, salt), fv(lane, 0, 1, salt), fv(lane, 0, 2, salt)),
        PxVec3(fv(lane, 0, 3, salt), fv(lane, 0, 4, salt), fv(lane, 0, 5, salt)));
    io.vel1 = UnAlignedSpatialVector(
        PxVec3(fv(lane, 1, 0, salt), fv(lane, 1, 1, salt), fv(lane, 1, 2, salt)),
        PxVec3(fv(lane, 1, 3, salt), fv(lane, 1, 4, salt), fv(lane, 1, 5, salt)));
    io.d0 = UnAlignedSpatialVector(
        PxVec3(0.1f * fv(lane, 2, 0, salt), 0.1f * fv(lane, 2, 1, salt), 0.1f * fv(lane, 2, 2, salt)),
        PxVec3(0.1f * fv(lane, 2, 3, salt), 0.1f * fv(lane, 2, 4, salt), 0.1f * fv(lane, 2, 5, salt)));
    io.d1 = UnAlignedSpatialVector(
        PxVec3(0.1f * fv(lane, 3, 0, salt), 0.1f * fv(lane, 3, 1, salt), 0.1f * fv(lane, 3, 2, salt)),
        PxVec3(0.1f * fv(lane, 3, 3, salt), 0.1f * fv(lane, 3, 4, salt), 0.1f * fv(lane, 3, 5, salt)));
    PxQuat q0(fv(lane, 4, 0, salt), fv(lane, 4, 1, salt), fv(lane, 4, 2, salt),
              1.0f + 0.3f * fv(lane, 4, 3, salt));
    PxQuat q1(fv(lane, 5, 0, salt), fv(lane, 5, 1, salt), fv(lane, 5, 2, salt),
              1.0f + 0.3f * fv(lane, 5, 3, salt));
    io.dq0 = q0.getNormalized();
    io.dq1 = q1.getNormalized();
    return io;
}

// The engine-struct solve mirrors live in pxgArtiExtSolveRef.h (hoisted in
// P8.99b so the self-constraint solve fixture shares them).

// ---- seeds ----

static void svxSeedResponses(std::vector<PxgArticulationBlockResponse>& rsp, uint32_t salt)
{
    rsp.resize(SVX_NB_RESP);
    std::memset(rsp.data(), 0xAB, rsp.size() * sizeof(rsp[0]));
    for (uint32_t r = 0; r < SVX_NB_RESP; ++r)
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

static void svxSeedBatch(PxgBlockConstraintBatch& b)
{
    std::memset(&b, 0xAB, sizeof(b));
    b.mConstraintBatchIndex = 0;
    b.startConstraintIndex = SVX_START_CONSTRAINT;
    b.startFrictionIndex = SVX_START_FRICTION;
    b.mArticulationResponseIndex = SVX_START_RESP;
}

struct SvxIoBuf
{
    std::vector<float> data;   // 2 * 32 lanes * 32 floats
    void init(uint32_t salt)
    {
        data.assign(2 * 32 * 32, -555.f);
        for (uint32_t l = 0; l < SVX_LANES; ++l)
        {
            const LaneIo io = svxLaneIo(l, salt);
            float* f = &data[l * 32];
            const UnAlignedSpatialVector* vs[4] = { &io.vel0, &io.vel1, &io.d0, &io.d1 };
            for (int v = 0; v < 4; ++v)
            {
                f[v * 6 + 0] = vs[v]->top.x; f[v * 6 + 1] = vs[v]->top.y; f[v * 6 + 2] = vs[v]->top.z;
                f[v * 6 + 3] = vs[v]->bottom.x; f[v * 6 + 4] = vs[v]->bottom.y; f[v * 6 + 5] = vs[v]->bottom.z;
            }
            f[24] = io.dq0.x; f[25] = io.dq0.y; f[26] = io.dq0.z; f[27] = io.dq0.w;
            f[28] = io.dq1.x; f[29] = io.dq1.y; f[30] = io.dq1.z; f[31] = io.dq1.w;
        }
    }
};

struct SvxParamsHost
{
    uint64_t batch, h0, h1, p0, p1, responses, io;
    float elapsedTime, minPen, ref0, ref1;
    uint32_t nbLanes, accumulateError;
};

static size_t svxRunContact(PxgCudaShim& shim, float ref0, float ref1)
{
    PxgBlockConstraintBatch batch;
    svxSeedBatch(batch);

    std::vector<PxgTGSBlockSolverContactHeader> ch(1);
    std::vector<PxgTGSBlockSolverFrictionHeader> fh(1);
    std::vector<PxgTGSBlockSolverContactPoint> cp(SVX_START_CONSTRAINT + SVX_MAX_CONTACTS);
    std::vector<PxgTGSBlockSolverContactFriction> fp(SVX_START_FRICTION + SVX_MAX_FRICTIONS + 1);
    std::vector<PxgArticulationBlockResponse> rsp;
    std::memset(ch.data(), 0xAB, sizeof(ch[0]));
    std::memset(fh.data(), 0xAB, sizeof(fh[0]));
    std::memset(cp.data(), 0xAB, cp.size() * sizeof(cp[0]));
    std::memset(fp.data(), 0xAB, fp.size() * sizeof(fp[0]));
    svxSeedResponses(rsp, 700);

    uint32_t nTors = 0, nCompliant = 0, nZeroContacts = 0;
    for (uint32_t l = 0; l < 32; ++l)
    {
        const uint32_t nc = l % (SVX_MAX_CONTACTS + 1);            // 0..6
        const uint32_t tf = (l * 7 + 3) % (SVX_MAX_FRICTIONS);     // 0..5 incl odd
        ch[0].numNormalConstr[l] = nc;
        fh[0].numFrictionConstr[l] = tf;
        if (tf & 1) ++nTors;
        if (nc == 0) ++nZeroContacts;
        ch[0].invMass0_1_angDom0_1[l] = make_float4(0.5f + 0.4f * fv(l, 0, 0, 71),
                                                    0.5f + 0.4f * fv(l, 0, 1, 71),
                                                    0.5f + 0.4f * fv(l, 0, 2, 71),
                                                    0.5f + 0.4f * fv(l, 0, 3, 71));
        PxVec3 n(fv(l, 1, 0, 72), fv(l, 1, 1, 72), fv(l, 1, 2, 72));
        n = n.getNormalized();
        ch[0].normal_staticFriction[l] = make_float4(n.x, n.y, n.z, 0.6f + 0.3f * std::fabs(fv(l, 1, 3, 72)));
        ch[0].minNormalForce[l] = 0.05f * std::fabs(fv(l, 2, 0, 73));
        ch[0].maxPenBias[l] = -0.2f * std::fabs(fv(l, 2, 1, 73));
        ch[0].flags[l] = (l % 3u == 0u) ? COMPLIANT_ACCEL : 0u;
        // every fourth lane compliant (restitutionXdt < 0)
        const bool compliant = (l % 4u) == 1u;
        if (compliant) ++nCompliant;
        ch[0].restitutionXdt[l] = compliant ? -0.5f - std::fabs(fv(l, 2, 2, 73)) : 0.3f * std::fabs(fv(l, 2, 2, 73));
        ch[0].cfm[l] = 0.01f * std::fabs(fv(l, 2, 3, 73));
        ch[0].p8[l] = 0.8f + 0.1f * fv(l, 3, 0, 74);
        fh[0].dynamicFriction[l] = 0.4f + 0.3f * std::fabs(fv(l, 3, 1, 74));
        fh[0].biasCoefficient[l] = 0.5f * fv(l, 3, 2, 74);
        fh[0].torsionalFrictionScale[l] = 0.7f + 0.2f * std::fabs(fv(l, 3, 3, 74));
        PxVec3 f0(fv(l, 4, 0, 75), fv(l, 4, 1, 75), fv(l, 4, 2, 75));
        PxVec3 f1(fv(l, 5, 0, 75), fv(l, 5, 1, 75), fv(l, 5, 2, 75));
        f0 = f0.getNormalized(); f1 = f1.getNormalized();
        fh[0].frictionNormals[0][l] = make_float4(f0.x, f0.y, f0.z, 0.f);
        fh[0].frictionNormals[1][l] = make_float4(f1.x, f1.y, f1.z, 0.f);
        fh[0].broken[l] = 0;

        for (uint32_t i = 0; i < SVX_MAX_CONTACTS; ++i)
        {
            PxgTGSBlockSolverContactPoint& c = cp[SVX_START_CONSTRAINT + i];
            c.raXn_extraCoeff[l] = make_float4(fv(l, 10 + i, 0, 76), fv(l, 10 + i, 1, 76),
                                               fv(l, 10 + i, 2, 76), 0.5f + 0.4f * std::fabs(fv(l, 10 + i, 3, 76)));
            c.rbXn_targetVelW[l] = make_float4(fv(l, 20 + i, 0, 77), fv(l, 20 + i, 1, 77),
                                               fv(l, 20 + i, 2, 77), 0.2f * fv(l, 20 + i, 3, 77));
            c.separation[l] = 0.05f * fv(l, 30 + i, 0, 78);
            c.maxImpulse[l] = 0.4f + 0.3f * std::fabs(fv(l, 30 + i, 1, 78));
            c.appliedForce[l] = 0.2f * std::fabs(fv(l, 30 + i, 2, 78));
            c.biasCoefficient[l] = 0.4f * fv(l, 30 + i, 3, 78);
            c.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 40 + i, 0, 79));
            c.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 40 + i, 1, 79));
        }
        for (uint32_t i = 0; i < SVX_MAX_FRICTIONS + 1; ++i)
        {
            PxgTGSBlockSolverContactFriction& f = fp[SVX_START_FRICTION + i];
            f.raXn_error[l] = make_float4(fv(l, 50 + i, 0, 80), fv(l, 50 + i, 1, 80),
                                          fv(l, 50 + i, 2, 80), 0.05f * fv(l, 50 + i, 3, 80));
            f.rbXn_targetVelW[l] = make_float4(fv(l, 60 + i, 0, 81), fv(l, 60 + i, 1, 81),
                                               fv(l, 60 + i, 2, 81), 0.1f * fv(l, 60 + i, 3, 81));
            f.appliedForce[l] = 0.15f * fv(l, 70 + i, 0, 82);
            f.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 70 + i, 1, 82));
            f.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 70 + i, 2, 82));
        }
    }
    std::printf("[ info ] svx contact(ref %g/%g): torsional %u, compliant %u, zero-contact %u\n",
                double(ref0), double(ref1), nTors, nCompliant, nZeroContacts);
    PXG_CHECK_MSG(nTors > 5 && nCompliant > 5 && nZeroContacts > 0, "contact coverage hole");

    SvxIoBuf io;
    io.init(90);

    // mirror
    std::vector<PxgTGSBlockSolverContactHeader> wCh = ch;
    std::vector<PxgTGSBlockSolverFrictionHeader> wFh = fh;
    std::vector<PxgTGSBlockSolverContactPoint> wCp = cp;
    std::vector<PxgTGSBlockSolverContactFriction> wFp = fp;
    std::vector<float> wantIo = io.data;
    for (uint32_t l = 0; l < SVX_LANES; ++l)
    {
        LaneIo lio = svxLaneIo(l, 90);
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        RefError err;
        refSolveExtContactTGS(batch, lio.vel0, lio.vel1, lio.d0, lio.d1, l, wCh.data(),
                              wFh.data(), wCp.data(), wFp.data(), rsp.data(), SVX_ELAPSED,
                              SVX_MINPEN, imp0, imp1, &err, ref0, ref1);
        float* o = &wantIo[32 * 32 + l * 32];
        const UnAlignedSpatialVector* vs[4] = { &lio.vel0, &lio.vel1, &imp0, &imp1 };
        for (int v = 0; v < 4; ++v)
        {
            o[v * 6 + 0] = vs[v]->top.x; o[v * 6 + 1] = vs[v]->top.y; o[v * 6 + 2] = vs[v]->top.z;
            o[v * 6 + 3] = vs[v]->bottom.x; o[v * 6 + 4] = vs[v]->bottom.y; o[v * 6 + 5] = vs[v]->bottom.z;
        }
        o[24] = err.sumOfSquares; o[25] = float(err.counter); o[26] = err.maxError;
    }

    // device
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

    SvxParamsHost prm = { uint64_t(dB), uint64_t(dCh), uint64_t(dFh), uint64_t(dCp),
                          uint64_t(dFp), uint64_t(dR), uint64_t(dIo),
                          SVX_ELAPSED, SVX_MINPEN, ref0, ref1, SVX_LANES, 1u };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svxContactTest");
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

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

static size_t svxRun1D(PxgCudaShim& shim, float ref0, float ref1)
{
    PxgBlockConstraintBatch batch;
    svxSeedBatch(batch);

    std::vector<PxgTGSBlockSolverConstraint1DHeader> hd(1);
    std::vector<PxgTGSBlockSolverConstraint1DCon> rows(SVX_START_CONSTRAINT + SVX_MAX_ROWS);
    std::vector<PxgArticulationBlockResponse> rsp;
    std::memset(hd.data(), 0xAB, sizeof(hd[0]));
    std::memset(rows.data(), 0xAB, rows.size() * sizeof(rows[0]));
    svxSeedResponses(rsp, 701);

    uint32_t nSpring = 0, nAccel = 0, nBounce = 0, nIneq = 0, nHard = 0;
    for (uint32_t l = 0; l < 32; ++l)
    {
        const uint32_t rc = (l * 5 + 1) % (SVX_MAX_ROWS + 1);   // 0..6
        uchar4 counts;
        counts.x = PxU8(rc); counts.y = 0; counts.z = 0; counts.w = 0;
        hd[0].rowCounts_breakable_orthoAxisCount[l] = counts;
        hd[0].rAWorld_invMass0D0[l] = make_float4(fv(l, 0, 0, 83), fv(l, 0, 1, 83),
                                                  fv(l, 0, 2, 83), 0.5f + 0.4f * std::fabs(fv(l, 0, 3, 83)));
        hd[0].rBWorld_invMass1D1[l] = make_float4(fv(l, 1, 0, 83), fv(l, 1, 1, 83),
                                                  fv(l, 1, 2, 83), 0.5f + 0.4f * std::fabs(fv(l, 1, 3, 83)));
        hd[0].invInertiaScale0[l] = 0.6f + 0.3f * fv(l, 2, 0, 84);
        hd[0].invInertiaScale1[l] = 0.6f + 0.3f * fv(l, 2, 1, 84);
        hd[0].cfm[l] = 0.01f * std::fabs(fv(l, 2, 2, 84));

        for (uint32_t i = 0; i < SVX_MAX_ROWS; ++i)
        {
            PxgTGSBlockSolverConstraint1DCon& c = rows[SVX_START_CONSTRAINT + i];
            // row mode: hard / inequality / spring / accel-spring / bounce
            const uint32_t mode = (l + i) % 5u;
            uint32_t flags = 0;
            float coeff2 = 0.f;   // hard: bounceVel 0
            if (mode == 1) { flags = DY_FLAG_INEQ; ++nIneq; }
            else if (mode == 2) { flags = DY_FLAG_SPRING; coeff2 = 0.4f + 0.3f * std::fabs(fv(l, i, 0, 85)); ++nSpring; }
            else if (mode == 3) { flags = DY_FLAG_SPRING | DY_FLAG_ACCEL; coeff2 = 0.4f + 0.3f * std::fabs(fv(l, i, 1, 85)); ++nAccel; }
            else if (mode == 4) { coeff2 = 0.5f * fv(l, i, 2, 85); if (coeff2 == 0.f) coeff2 = 0.25f; ++nBounce; }
            else ++nHard;

            c.lin0XYZ_initBiasOrCoeff0[l] = make_float4(fv(l, 10 + i, 0, 86), fv(l, 10 + i, 1, 86),
                                                        fv(l, 10 + i, 2, 86), 0.1f * fv(l, 10 + i, 3, 86));
            c.lin1XYZ_biasScaleOrCoeff1[l] = make_float4(fv(l, 20 + i, 0, 86), fv(l, 20 + i, 1, 86),
                                                         fv(l, 20 + i, 2, 86), -0.4f * std::fabs(fv(l, 20 + i, 3, 86)));
            c.ang0XYZ_velMultiplierOrCoeff2[l] = make_float4(fv(l, 30 + i, 0, 86), fv(l, 30 + i, 1, 86),
                                                             fv(l, 30 + i, 2, 86), coeff2);
            c.ang1XYZ_velTargetOrCoeff3[l] = make_float4(fv(l, 40 + i, 0, 86), fv(l, 40 + i, 1, 86),
                                                         fv(l, 40 + i, 2, 86), 0.2f * fv(l, 40 + i, 3, 86));
            c.flags[l] = flags;
            c.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 50 + i, 0, 87));
            c.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 50 + i, 1, 87));
            c.geometricError[l] = 0.1f * fv(l, 50 + i, 2, 87);
            c.minImpulse[l] = -0.6f - 0.3f * std::fabs(fv(l, 60 + i, 0, 88));
            c.maxImpulse[l] = 0.6f + 0.3f * std::fabs(fv(l, 60 + i, 1, 88));
            c.maxBias[l] = 0.3f + 0.2f * std::fabs(fv(l, 60 + i, 2, 88));
            c.angularErrorScale[l] = (l + i) % 2 ? 1.f : 0.f;
            c.appliedForce[l] = 0.2f * fv(l, 60 + i, 3, 88);
            c.residual[l] = 0.f;
        }
    }
    std::printf("[ info ] svx 1d(ref %g/%g): hard %u, ineq %u, spring %u, accel %u, bounce %u\n",
                double(ref0), double(ref1), nHard, nIneq, nSpring, nAccel, nBounce);
    PXG_CHECK_MSG(nHard > 5 && nIneq > 5 && nSpring > 5 && nAccel > 5 && nBounce > 5,
                  "1d row-mode coverage hole");

    SvxIoBuf io;
    io.init(91);

    // mirror
    std::vector<PxgTGSBlockSolverConstraint1DCon> wRows = rows;
    std::vector<float> wantIo = io.data;
    for (uint32_t l = 0; l < SVX_LANES; ++l)
    {
        LaneIo lio = svxLaneIo(l, 91);
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        refSolveExt1DTGS(batch, lio.vel0, lio.vel1, lio.d0, lio.d1, l, hd.data(),
                         wRows.data(), rsp.data(), lio.dq0, lio.dq1, SVX_ELAPSED, imp0, imp1,
                         true, ref0, ref1);
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
    PxgDevicePtr dR = up(rsp.data(), rsp.size() * sizeof(rsp[0]));
    PxgDevicePtr dIo = up(io.data.data(), io.data.size() * 4);
    if (!dB || !dHd || !dRows || !dR || !dIo)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SvxParamsHost prm = { uint64_t(dB), uint64_t(dHd), uint64_t(dRows), 0, 0,
                          uint64_t(dR), uint64_t(dIo),
                          SVX_ELAPSED, SVX_MINPEN, ref0, ref1, SVX_LANES, 1u };
    PXG_CHECK_MSG(shim.launchKernel(1, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svx1DTest");
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
    cmp(dHd, hd.data(), sizeof(hd[0]), "header");         // header is read-only
    cmp(dRows, wRows.data(), wRows.size() * sizeof(wRows[0]), "rows");
    cmp(dIo, wantIo.data(), wantIo.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveExtTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveExtTGS.spv", "svxContactTest", 80u, 0 },
        { "npArtiSolveExtTGS.spv", "svx1DTest", 80u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &svxLoader);

    size_t bad = 0;
    bad += svxRunContact(shim, 1.0f, 1.0f);
    bad += svxRunContact(shim, 0.75f, 1.25f);
    bad += svxRun1D(shim, 1.0f, 1.0f);
    bad += svxRun1D(shim, 0.75f, 1.25f);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
