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

// testArtiConstraintPrepBody.cpp -- P8.102a: the PxgSolverExtBody2 layer of
// the articulation constraint prep (artiConstraintPrep2.cu:55-165) on the
// ENGINE layout, against the engine's own arithmetic on the host:
// PxSpatialMatrix::initialize / multiplyInertia / operator* are all
// PX_CUDA_CALLABLE, so the expectation is computed by CALLING them, and the
// createPxgSolverExtBody control flow is the only transcription.
//
// (The MATH of this layer was Phase-1-ported as npArtiExtBody.slang over
// abstracted flat inputs; P8.102a is the lift onto the real record /
// solver-body layouts -- the fetches ARE the new surface.)
//
// The fixture drives one body per lane from a mixed node table:
//   %4==0  an articulation link (record pool with REAL per-record device
//          arrays: linkBody2Worlds 28-byte PxTransforms, 144-byte
//          spatialResponseMatrixW, 24-byte motionVelocities, cfms, and the
//          80-byte PxgArticulationLink array),
//   %4==1  a dynamic rigid,
//   %4==2  a STATIC rigid (PxNodeIndex(), isStaticBody) -- its solver-body
//          velocity fields are POISON and must not reach the outputs, but
//          flags/maxImpulse ARE read, and bodyFrames IS loaded,
//   %4==3  a KINEMATIC rigid -- the txIData entry is POISON (the response
//          must be the zero matrix, never built from it).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "CmSpatialVector.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "PxgSolverBody.h"
#include "PxNodeIndex.h"
#include "PxRigidBody.h"
#include "PxgConstraintPrep.h"
#include "PxgFrictionPatch.h"
#include "PxgSolverConstraintBlock1D.h"
#include "PxgConstraintBlock.h"
#include "pxgArtiPrepRef.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

static std::vector<uint8_t> acpLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t ACP_LANES = 26;
const uint32_t ACP_NB_ARTI = 6;         // record pool entries
const uint32_t ACP_NB_RIGID = 16;       // solver-body pool entries
const uint32_t ACP_REC_SIZE = 544;
const uint32_t ACP_IO_STRIDE = 72;      // floats per lane (70 written + slack)

static_assert(sizeof(PxgArticulation) == ACP_REC_SIZE, "record stride");
static_assert(sizeof(PxgArticulationLink) == 80, "link record stride");
static_assert(offsetof(PxgArticulationLink, offsetSlop) == 60, "offsetSlop pin");
static_assert(sizeof(PxgSolverBodyData) == 96, "solver body data stride");
static_assert(sizeof(PxgSolverTxIData) == 64, "txIData stride");
static_assert(sizeof(PxSpatialMatrix) == 144, "PxSpatialMatrix stride");
static_assert(sizeof(PxTransform) == 28, "PxTransform stride");

static float av(uint32_t i, uint32_t k)
{
    return (float((i * 53u + k * 31u) % 479u) + 1.0f) * 0.0078125f - 1.85f;
}

static uint32_t acpNumLinks(uint32_t r) { return 4u + (r % 6u); }   // 4..9

} // namespace

PXG_TEST(artiConstraintPrepBody)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiConstraintPrep.spv", "acpBodyTest", 64u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &acpLoader);

    // ---- pools ----
    uint32_t totalLinks = 0;
    for (uint32_t r = 0; r < ACP_NB_ARTI; ++r) totalLinks += acpNumLinks(r);

    std::vector<uint8_t> rec(size_t(ACP_NB_ARTI) * ACP_REC_SIZE, 0xAB);
    std::vector<PxTransform> b2w(totalLinks);
    std::vector<PxSpatialMatrix> resp(totalLinks);
    std::vector<UnAlignedSpatialVector> mv(totalLinks);
    std::vector<float> cfms(totalLinks);
    std::vector<PxgArticulationLink> links(totalLinks);
    std::memset(links.data(), 0xAB, links.size() * sizeof(links[0]));

    uint32_t base = 0;
    std::vector<uint32_t> linkBases(ACP_NB_ARTI);
    for (uint32_t r = 0; r < ACP_NB_ARTI; ++r)
    {
        linkBases[r] = base;
        for (uint32_t l = 0; l < acpNumLinks(r); ++l)
        {
            const uint32_t i = base + l;
            PxQuat q(av(i, 0), av(i, 1), av(i, 2), 1.0f + 0.4f * av(i, 3));
            q.normalize();
            b2w[i] = PxTransform(PxVec3(av(i, 4), av(i, 5), av(i, 6)), q);
            for (uint32_t c = 0; c < 6; ++c)
                for (uint32_t rr = 0; rr < 6; ++rr)
                    resp[i].column[c][rr] = 0.3f * av(i, 10u + c * 6u + rr);
            mv[i] = UnAlignedSpatialVector(PxVec3(av(i, 50), av(i, 51), av(i, 52)),
                                           PxVec3(av(i, 53), av(i, 54), av(i, 55)));
            cfms[i] = 0.02f + 0.01f * std::fabs(av(i, 56));
            links[i].initialAngVel = PxVec3(av(i, 60), av(i, 61), av(i, 62));
            links[i].penBiasClamp = -0.5f - std::fabs(av(i, 63));
            links[i].offsetSlop = 0.01f * std::fabs(av(i, 64));
        }
        base += acpNumLinks(r);
    }

    std::vector<PxgSolverBodyData> sbd(ACP_NB_RIGID);
    std::vector<PxgSolverTxIData> txi(ACP_NB_RIGID);
    std::vector<PxAlignedTransform> frames(ACP_NB_RIGID);
    std::memset(sbd.data(), 0xAB, sbd.size() * sizeof(sbd[0]));
    std::memset(txi.data(), 0xAB, txi.size() * sizeof(txi[0]));
    for (uint32_t i = 0; i < ACP_NB_RIGID; ++i)
    {
        const uint32_t kind = i % 4u;   // 0/1 dynamic, 2 static, 3 kinematic
        PxQuat q(av(i, 70), av(i, 71), av(i, 72), 1.0f + 0.3f * av(i, 73));
        q.normalize();
        frames[i].q = PxAlignedQuat(q.x, q.y, q.z, q.w);
        frames[i].p = make_float4(av(i, 74), av(i, 75), av(i, 76), av(i, 77));  // w poison-ish

        sbd[i].flags = (kind == 3u) ? PxU32(PxRigidBodyFlag::eKINEMATIC) : 0u;
        sbd[i].maxImpulse = 2.0f + std::fabs(av(i, 78));
        if (kind != 2u)   // static keeps its velocity fields POISONED
        {
            sbd[i].initialAngVel = PxVec3(av(i, 80), av(i, 81), av(i, 82));
            sbd[i].penBiasClamp = -0.4f - std::fabs(av(i, 83));
            sbd[i].initialLinVel = PxVec3(av(i, 84), av(i, 85), av(i, 86));
            sbd[i].invMass = 0.5f + std::fabs(av(i, 87));
            sbd[i].offsetSlop = 0.02f * std::fabs(av(i, 88));
        }
        if (kind == 0u || kind == 1u)   // only dynamics get a live txIData
        {
            txi[i].sqrtInvInertia = PxMat33(PxVec3(av(i, 90), av(i, 91), av(i, 92)),
                                            PxVec3(av(i, 93), av(i, 94), av(i, 95)),
                                            PxVec3(av(i, 96), av(i, 97), av(i, 98)));
        }
    }

    // ---- the node table ----
    std::vector<PxNodeIndex> nodes(ACP_LANES);
    std::vector<uint32_t> bodyIdx(ACP_LANES);
    for (uint32_t lane = 0; lane < ACP_LANES; ++lane)
    {
        const uint32_t kind = lane % 4u;
        if (kind == 0u)
        {
            const uint32_t r = (lane / 4u) % ACP_NB_ARTI;
            const uint32_t linkId = (lane * 3u + 1u) % acpNumLinks(r);
            nodes[lane] = PxNodeIndex(700u + lane, linkId);
            bodyIdx[lane] = r;
        }
        else if (kind == 2u)
        {
            nodes[lane] = PxNodeIndex();   // invalid -> isStaticBody
            bodyIdx[lane] = (lane * 5u + 2u) % ACP_NB_RIGID;
            bodyIdx[lane] = (bodyIdx[lane] & ~3u) | 2u;   // a static pool entry
        }
        else
        {
            nodes[lane] = PxNodeIndex(300u + lane);
            uint32_t bi = (lane * 7u + kind) % ACP_NB_RIGID;
            bi = (bi & ~3u) | ((kind == 3u) ? 3u : (lane % 2u));   // kinematic or dynamic entry
            bodyIdx[lane] = bi;
        }
    }

    // ---- expected, via the ENGINE's arithmetic ----
    std::vector<float> want(size_t(ACP_LANES) * ACP_IO_STRIDE);
    std::memset(want.data(), 0xAB, want.size() * 4);   // the slack tail stays poison
    for (uint32_t lane = 0; lane < ACP_LANES; ++lane)
    {
        PxgSolverExtBody2 b;
        const PxNodeIndex node = nodes[lane];
        pxgtest::refCreateExtBody(node, bodyIdx[lane], linkBases.data(), b2w.data(), resp.data(),
                                  mv.data(), cfms.data(), links.data(), sbd.data(), txi.data(),
                                  frames.data(), b);

        const UnAlignedSpatialVector probe(
            PxVec3(0.3f + 0.01f * float(lane), -0.2f, 0.5f),
            PxVec3(0.7f, 0.11f * float(lane), -0.4f));

        const UnAlignedSpatialVector respVec = pxgtest::refCreateImpulseResponseVector(b, probe);
        const UnAlignedSpatialVector impResp = b.mSpatialResponse * probe;
        const float projV = b.velocity.dot(probe);
        const float projA = b.velocity.top.dot(probe.top);

        float* o = &want[size_t(lane) * ACP_IO_STRIDE];
        uint32_t k = 0;
        for (uint32_t c = 0; c < 6; ++c)
            for (uint32_t rr = 0; rr < 6; ++rr)
                o[k++] = b.mSpatialResponse.column[c][rr];
        o[k++] = b.velocity.top.x; o[k++] = b.velocity.top.y; o[k++] = b.velocity.top.z;
        o[k++] = b.velocity.bottom.x; o[k++] = b.velocity.bottom.y; o[k++] = b.velocity.bottom.z;
        o[k++] = b.body2World.p.x; o[k++] = b.body2World.p.y; o[k++] = b.body2World.p.z;
        o[k++] = b.body2World.q.x; o[k++] = b.body2World.q.y; o[k++] = b.body2World.q.z;
        o[k++] = b.body2World.q.w;
        o[k++] = b.penBiasClamp;
        o[k++] = b.maxImpulse;
        uint32_t li = b.linkIndex;
        std::memcpy(&o[k++], &li, 4);
        o[k++] = b.isKinematic ? 1.0f : 0.0f;
        uint32_t bix = b.bodyIndex;
        std::memcpy(&o[k++], &bix, 4);
        o[k++] = b.cfm;
        o[k++] = b.offsetSlop;
        o[k++] = respVec.top.x; o[k++] = respVec.top.y; o[k++] = respVec.top.z;
        o[k++] = respVec.bottom.x; o[k++] = respVec.bottom.y; o[k++] = respVec.bottom.z;
        o[k++] = impResp.top.x; o[k++] = impResp.top.y; o[k++] = impResp.top.z;
        o[k++] = impResp.bottom.x; o[k++] = impResp.bottom.y; o[k++] = impResp.bottom.z;
        o[k++] = projV;
        o[k++] = projA;
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
    PxgDevicePtr dB2w = up(b2w.data(), b2w.size() * sizeof(b2w[0]));
    PxgDevicePtr dResp = up(resp.data(), resp.size() * sizeof(resp[0]));
    PxgDevicePtr dMv = up(mv.data(), mv.size() * sizeof(mv[0]));
    PxgDevicePtr dCfm = up(cfms.data(), cfms.size() * 4);
    PxgDevicePtr dLinks = up(links.data(), links.size() * sizeof(links[0]));

    // patch each record's array pointers with its device slices
    for (uint32_t r = 0; r < ACP_NB_ARTI; ++r)
    {
        uint8_t* R = &rec[size_t(r) * ACP_REC_SIZE];
        auto w64at = [&](size_t off, uint64_t v) { std::memcpy(R + off, &v, 8); };
        w64at(offsetof(PxgArticulation, links), uint64_t(dLinks) + uint64_t(linkBases[r]) * 80u);
        w64at(offsetof(PxgArticulation, motionVelocities),
              uint64_t(dMv) + uint64_t(linkBases[r]) * 24u);
        w64at(offsetof(PxgArticulation, spatialResponseMatrixW),
              uint64_t(dResp) + uint64_t(linkBases[r]) * 144u);
        w64at(offsetof(PxgArticulation, linkBody2Worlds),
              uint64_t(dB2w) + uint64_t(linkBases[r]) * 28u);
        w64at(offsetof(PxgArticulation, cfms), uint64_t(dCfm) + uint64_t(linkBases[r]) * 4u);
    }

    PxgDevicePtr dRec = up(rec.data(), rec.size());
    PxgDevicePtr dSbd = up(sbd.data(), sbd.size() * sizeof(sbd[0]));
    PxgDevicePtr dTxi = up(txi.data(), txi.size() * sizeof(txi[0]));
    PxgDevicePtr dFrames = up(frames.data(), frames.size() * sizeof(frames[0]));
    PxgDevicePtr dNodes = up(nodes.data(), nodes.size() * sizeof(nodes[0]));
    PxgDevicePtr dBi = up(bodyIdx.data(), bodyIdx.size() * 4);
    std::vector<float> ioInit(size_t(ACP_LANES) * ACP_IO_STRIDE);
    std::memset(ioInit.data(), 0xAB, ioInit.size() * 4);
    PxgDevicePtr dIo = up(ioInit.data(), ioInit.size() * 4);

    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return;
    }

    struct Params
    {
        uint64_t nodes, bodyIndices, articulations, solverBodyData, txIDatas, bodyFrames, io;
        uint32_t nbLanes, pad0;
    } prm = { uint64_t(dNodes), uint64_t(dBi), uint64_t(dRec), uint64_t(dSbd),
              uint64_t(dTxi), uint64_t(dFrames), uint64_t(dIo), ACP_LANES, 0u };
    static_assert(sizeof(Params) == 64, "params blob");
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch acpBodyTest");
    shim.streamSynchronize();

    std::vector<float> got(want.size());
    shim.memcpyDtoH(got.data(), dIo, got.size() * 4);

    size_t bad = 0;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(want.data());
    const uint32_t* g = reinterpret_cast<const uint32_t*>(got.data());
    for (size_t k = 0; k < want.size(); ++k)
        if (w[k] != g[k])
        {
            if (bad < 12)
                std::printf("MM acp lane %zu field %zu  ref %08x  gpu %08x\n",
                            k / ACP_IO_STRIDE, k % ACP_IO_STRIDE, w[k], g[k]);
            ++bad;
        }
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
}
