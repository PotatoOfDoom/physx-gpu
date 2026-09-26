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

// testArtiConstraintPrepSetup.cpp -- P8.102b: the TGS ext contact-row setup
// (setupFinalizeExtSolverConstraintsBlock, artiConstraintPrep2.cu:617-1112)
// against the host mirror pxgtest::refSetupExtContactsTGS.  The bodies come
// through the P8.102a machinery (refCreateExtBody over the same pools the
// device reads), so every impulse-response term is the engine's own
// PxSpatialMatrix arithmetic.
//
// Engineered lanes (coverage-asserted below rather than hoped for):
//   lane 2   PIN A: isGreater2 TRUE and cTargetVel == -1 exactly
//            (normal (1,0,0), targetVel (-1,0,0), b0 dyn linVel (-3,0,0),
//            b1 static).  tvCond == 0 -> the :837 precedence quirk yields
//            targetVelocity 0 where the "fixed" parenthesized form yields
//            -1 + 1.5 = 0.5.  Asserted via the mirror's own output.
//   lane 5   PIN B: isGreater2 FALSE and cTargetVel = 0.5 != 0 (b0 dyn
//            linVel (+1,0,0), separating).  Quirk yields (-vrel)*rest =
//            -0.4 where the fixed form yields 0.5.  Asserted == -0.4.
//   lane 7   PIN C: linNorVel == 0 EXACTLY with angNorVel != 0 (b0
//            kinematic linVel (0,2,0) against normal (1,0,0), b1 static)
//            -- the division guard in the slop scale.
//   lane 11  SLOP lane: b0 kinematic, linNorVel = +0.01 with angNorVel
//            ~ +3.6 -> slop scale ~360, so raXn components far above the
//            raw solverOffsetSlop still get zeroed.
// Per-lane variation: contactCount 0(early-out)/1..3, aCount lane%3,
// perPointFriction (lane>>1)&1, restitution by lane%7 incl. compliant
// (negative, both accel-spring flag states) and 1e-6 (below the arti
// path's 1e-5 threshold).  Two rounds: A with torsional data + bias 0.9
// (p8 clamps to 0.8), B with torsional (0,0) + bias 0.5.

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

static std::vector<uint8_t> acsLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t ACS_LANES = 26;
const uint32_t ACS_NB_ARTI = 6;
const uint32_t ACS_NB_RIGID = 16;
const uint32_t ACS_REC_SIZE = 544;
const uint32_t ACS_MAX_CONTACTS = 3;
const uint32_t ACS_FRICTION_ROWS = 4;   // 2 anchors x 2 tangents; torsional reuses row 2
const uint32_t ACS_RESP_ROWS = 8;       // max 3 contacts + 4 frictions (+1 slack)

static_assert(sizeof(PxgBlockContactData) == 1664, "contact data block");
static_assert(sizeof(PxgBlockContactPoint) == 1024, "contact point block");
static_assert(sizeof(PxgBlockFrictionPatch) == 2816, "friction patch block");
static_assert(sizeof(PxgBlockFrictionAnchorPatch) == 2048, "friction anchor block");
static_assert(sizeof(PxgArticulationBlockResponse) == 1536, "response block");
static_assert(sizeof(PxgTGSBlockSolverContactHeader) == 2048, "TGS contact header");
static_assert(sizeof(PxgTGSBlockSolverFrictionHeader) == 1664, "TGS friction header");
static_assert(sizeof(PxgTGSBlockSolverContactPoint) == 1792, "TGS contact point");
static_assert(sizeof(PxgTGSBlockSolverContactFriction) == 1408, "TGS contact friction");

static float av(uint32_t i, uint32_t k)
{
    return (float((i * 53u + k * 31u) % 479u) + 1.0f) * 0.0078125f - 1.85f;
}

static uint32_t acsNumLinks(uint32_t r) { return 4u + (r % 6u); }

static uint32_t acsContactCount(uint32_t lane)
{
    return (lane % 5u == 4u) ? 0u : 1u + ((lane * 2u) % 3u);
}

static float acsRestitution(uint32_t lane)
{
    switch (lane % 7u)
    {
    case 0: return 0.5f;
    case 1: return 0.0f;
    case 2: return 0.5f;
    case 3: return -800.0f - 10.0f * float(lane);
    case 4: return 1e-6f;                          // below the arti path's 1e-5
    case 5: return 0.4f;
    default: return 0.9f;
    }
}

struct AcsRound
{
    float invDt, stepDt, invTotalDt, bounceThreshold, biasCoefficient;
    float solverOffsetSlop, torsionalX, torsionalY, totalDt;
};

struct AcsParamsHost
{
    uint64_t contactData, contacts, frictionPatch, fAnchor, nodes, bodyIndices;
    uint64_t articulations, solverBodyData, txIDatas, bodyFrames;
    uint64_t contactHeader, frictionHeader, contactPoints, frictions, responses, io;
    float invDt, stepDt, invTotalDt, bounceThreshold, biasCoefficient;
    float solverOffsetSlop, torsionalX, torsionalY, totalDt;
    uint32_t nbLanes;
};
static_assert(sizeof(AcsParamsHost) == 168, "params blob");

} // namespace

PXG_TEST(artiConstraintPrepSetup)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiConstraintPrep.spv", "acpSetupTest", 168u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &acsLoader);

    // ---- the P8.102a pools ----
    uint32_t totalLinks = 0;
    for (uint32_t r = 0; r < ACS_NB_ARTI; ++r) totalLinks += acsNumLinks(r);

    std::vector<uint8_t> rec(size_t(ACS_NB_ARTI) * ACS_REC_SIZE, 0xAB);
    std::vector<PxTransform> b2w(totalLinks);
    std::vector<PxSpatialMatrix> resp(totalLinks);
    std::vector<UnAlignedSpatialVector> mv(totalLinks);
    std::vector<float> cfms(totalLinks);
    std::vector<PxgArticulationLink> links(totalLinks);
    std::memset(links.data(), 0xAB, links.size() * sizeof(links[0]));

    uint32_t base = 0;
    std::vector<uint32_t> linkBases(ACS_NB_ARTI);
    for (uint32_t r = 0; r < ACS_NB_ARTI; ++r)
    {
        linkBases[r] = base;
        for (uint32_t l = 0; l < acsNumLinks(r); ++l)
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
        base += acsNumLinks(r);
    }

    std::vector<PxgSolverBodyData> sbd(ACS_NB_RIGID);
    std::vector<PxgSolverTxIData> txi(ACS_NB_RIGID);
    std::vector<PxAlignedTransform> frames(ACS_NB_RIGID);
    std::memset(sbd.data(), 0xAB, sbd.size() * sizeof(sbd[0]));
    std::memset(txi.data(), 0xAB, txi.size() * sizeof(txi[0]));
    for (uint32_t i = 0; i < ACS_NB_RIGID; ++i)
    {
        const uint32_t kind = i % 4u;   // 0/1 dynamic, 2 static, 3 kinematic
        PxQuat q(av(i, 70), av(i, 71), av(i, 72), 1.0f + 0.3f * av(i, 73));
        q.normalize();
        frames[i].q = PxAlignedQuat(q.x, q.y, q.z, q.w);
        frames[i].p = make_float4(av(i, 74), av(i, 75), av(i, 76), av(i, 77));

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
        if (kind == 0u || kind == 1u)
        {
            txi[i].sqrtInvInertia = PxMat33(PxVec3(av(i, 90), av(i, 91), av(i, 92)),
                                            PxVec3(av(i, 93), av(i, 94), av(i, 95)),
                                            PxVec3(av(i, 96), av(i, 97), av(i, 98)));
        }
    }

    // engineered pool entries for the pin lanes (before the node table so the
    // mirror and the device see the same values everywhere they are used)
    sbd[4].initialLinVel = PxVec3(-3.0f, 0.0f, 0.0f);     // PIN A body0 (dynamic)
    sbd[4].initialAngVel = PxVec3(0.0f, 0.0f, 0.0f);
    sbd[8].initialLinVel = PxVec3(1.0f, 0.0f, 0.0f);      // PIN B body0 (dynamic)
    sbd[8].initialAngVel = PxVec3(0.0f, 0.0f, 0.0f);
    sbd[7].initialLinVel = PxVec3(0.0f, 2.0f, 0.0f);      // PIN C body0 (kinematic)
    sbd[7].initialAngVel = PxVec3(0.3f, 0.5f, 0.7f);
    sbd[11].initialLinVel = PxVec3(0.01f, 0.0f, 0.0f);    // SLOP body0 (kinematic)
    sbd[11].initialAngVel = PxVec3(-4.0f, -4.0f, -4.0f);
    sbd[12].initialLinVel = PxVec3(-2.5f, 0.0f, 0.0f);    // PIN D body0 (dynamic):
    sbd[12].initialAngVel = PxVec3(0.0f, 0.0f, 0.0f);     //   bounces EXCEPT rest 1e-6
    sbd[13].initialLinVel = PxVec3(-2.0f, 0.0f, 0.0f);    // PIN E body0 (dynamic):
    sbd[13].initialAngVel = PxVec3(0.0f, 0.0f, 0.0f);     //   bounce with tv != 0

    // ---- the node table: two bodies per lane ----
    std::vector<PxNodeIndex> nodes(ACS_LANES * 2);
    std::vector<uint32_t> bodyIdx(ACS_LANES * 2);
    auto setBody = [&](uint32_t lane, uint32_t s, uint32_t kind) {
        const uint32_t slot = lane * 2u + s;
        if (kind == 0u)         // articulation link
        {
            const uint32_t r = (lane * 2u + s) % ACS_NB_ARTI;
            const uint32_t linkId = (lane * 3u + 1u + s) % acsNumLinks(r);
            nodes[slot] = PxNodeIndex(700u + slot, linkId);
            bodyIdx[slot] = r;
        }
        else if (kind == 3u)    // static
        {
            nodes[slot] = PxNodeIndex();
            bodyIdx[slot] = (((lane * 5u + s * 7u) % ACS_NB_RIGID) & ~3u) | 2u;
        }
        else                    // dynamic (1) or kinematic (2)
        {
            nodes[slot] = PxNodeIndex(300u + slot);
            uint32_t bi = (lane * 7u + s * 3u) % ACS_NB_RIGID;
            bi = (bi & ~3u) | ((kind == 2u) ? 3u : (lane % 2u));
            bodyIdx[slot] = bi;
        }
    };
    for (uint32_t lane = 0; lane < ACS_LANES; ++lane)
    {
        setBody(lane, 0, lane % 4u);
        setBody(lane, 1, (lane / 4u + lane % 3u) % 4u);
    }
    // the engineered lanes: b0 the pinned rigid entry, b1 a static entry
    auto pinRigid = [&](uint32_t lane, uint32_t bi) {
        nodes[lane * 2u] = PxNodeIndex(300u + lane * 2u);
        bodyIdx[lane * 2u] = bi;
        nodes[lane * 2u + 1u] = PxNodeIndex();
        bodyIdx[lane * 2u + 1u] = 6u + 4u * (lane % 3u);   // 6/10/14, all static
    };
    pinRigid(2, 4);    // PIN A
    pinRigid(5, 8);    // PIN B
    pinRigid(7, 7);    // PIN C (kinematic)
    pinRigid(11, 11);  // SLOP (kinematic)
    pinRigid(13, 13);  // PIN E: bounce, cTargetVel 0 -> targetVelocity 1.8
    pinRigid(25, 12);  // PIN D: bounce conditions but restitution 1e-6

    // ---- the batch inputs ----
    PxgBlockContactData cd;
    std::vector<PxgBlockContactPoint> contacts(ACS_MAX_CONTACTS);
    PxgBlockFrictionPatch fp;
    PxgBlockFrictionAnchorPatch fa;
    std::memset(&cd, 0xAB, sizeof(cd));
    std::memset(contacts.data(), 0xAB, contacts.size() * sizeof(contacts[0]));
    std::memset(&fp, 0xAB, sizeof(fp));
    std::memset(&fa, 0xAB, sizeof(fa));

    for (uint32_t lane = 0; lane < ACS_LANES; ++lane)
    {
        const uint32_t cc = acsContactCount(lane);
        const bool pinNormal = (lane == 2u || lane == 5u || lane == 7u || lane == 11u
                                || lane == 13u || lane == 25u);

        PxVec3 n(av(lane, 200) + 0.2f, av(lane, 201), av(lane, 202) - 0.1f);
        n.normalize();
        if (pinNormal) n = PxVec3(1.0f, 0.0f, 0.0f);
        cd.normal_restitutionW[lane] = make_float4(n.x, n.y, n.z, acsRestitution(lane));

        cd.mInvMassScale[lane].linear0 = 1.0f + 0.1f * av(lane, 210);
        cd.mInvMassScale[lane].angular0 = 1.0f + 0.1f * av(lane, 211);
        cd.mInvMassScale[lane].linear1 = 1.0f + 0.1f * av(lane, 212);
        cd.mInvMassScale[lane].angular1 = 1.0f + 0.1f * av(lane, 213);

        cd.contactData[lane].restDistance = pinNormal ? 0.0f : 0.01f * av(lane, 214);
        cd.contactData[lane].staticFriction = 0.6f + 0.2f * std::fabs(av(lane, 215));
        cd.contactData[lane].dynamicFriction = 0.4f + 0.2f * std::fabs(av(lane, 216));
        cd.contactData[lane].mNumContacts = PxU8(lane + 1u);          // distinct bytes so a
        cd.contactData[lane].mSolverFlags =                            // wrong-byte unpack shows
            PxU8(((lane & 1u) ? 8u : 0u) | ((lane & 4u) ? 1u : 0u)); // bit 3 = accel spring
        cd.contactData[lane].prevFrictionPatchCount = PxU8(lane + 7u);
        cd.damping[lane] = 20.0f + float(lane);

        for (uint32_t j = 0; j < ACS_MAX_CONTACTS; ++j)
        {
            contacts[j].point_separationW[lane] =
                make_float4(av(lane, 220 + j * 8), av(lane, 221 + j * 8),
                            av(lane, 222 + j * 8), 0.05f * av(lane, 223 + j * 8));
            contacts[j].targetVel_maxImpulseW[lane] =
                make_float4(0.3f * av(lane, 224 + j * 8), 0.3f * av(lane, 225 + j * 8),
                            0.3f * av(lane, 226 + j * 8), 1.5f + std::fabs(av(lane, 227 + j * 8)));
        }

        fp.anchorCount[lane] = lane % 3u;
        const bool ppf = ((lane >> 1) & 1u) != 0u;
        // valid but wrong-if-ignored for perPointFriction lanes; a valid row
        // (< ACS_MAX_CONTACTS) even when unused so a mutant cannot go OOB
        fp.contactID[0][lane] = ppf ? (cc > 0u ? cc - 1u : 0u) : 2u;
        fp.contactID[1][lane] = ppf ? (cc / 2u) : 2u;

        for (uint32_t j = 0; j < 2; ++j)
        {
            fa.body0Anchors[j][lane] = make_float4(0.3f * av(lane, 240 + j * 4),
                                                   0.3f * av(lane, 241 + j * 4),
                                                   0.3f * av(lane, 242 + j * 4), av(lane, 243));
            fa.body1Anchors[j][lane] = make_float4(0.3f * av(lane, 250 + j * 4),
                                                   0.3f * av(lane, 251 + j * 4),
                                                   0.3f * av(lane, 252 + j * 4), av(lane, 253));
        }
    }

    // the engineered contacts (contact 0 of each pin lane); frame0p is the
    // b0 pool entry's frame so ra == 0 exactly for A and B
    auto framePos = [&](uint32_t bi) {
        return PxVec3(frames[bi].p.x, frames[bi].p.y, frames[bi].p.z);
    };
    {
        const PxVec3 fA = framePos(4);
        contacts[0].point_separationW[2] = make_float4(fA.x, fA.y, fA.z, -0.05f);
        contacts[0].targetVel_maxImpulseW[2] = make_float4(-1.0f, 0.0f, 0.0f, 3.0f);
        const PxVec3 fB = framePos(8);
        contacts[0].point_separationW[5] = make_float4(fB.x, fB.y, fB.z, -0.02f);
        contacts[0].targetVel_maxImpulseW[5] = make_float4(0.5f, 0.2f, 0.0f, 3.0f);
        const PxVec3 fC = framePos(7) + PxVec3(0.0f, 0.3f, -0.2f);
        contacts[0].point_separationW[7] = make_float4(fC.x, fC.y, fC.z, -0.03f);
        const PxVec3 fS = framePos(11) + PxVec3(0.0f, 0.5f, -0.4f);
        contacts[0].point_separationW[11] = make_float4(fS.x, fS.y, fS.z, -0.04f);
        const PxVec3 fE = framePos(13);
        contacts[0].point_separationW[13] = make_float4(fE.x, fE.y, fE.z, -0.05f);
        contacts[0].targetVel_maxImpulseW[13] = make_float4(0.0f, 0.0f, 0.0f, 3.0f);
        const PxVec3 fD = framePos(12);
        contacts[0].point_separationW[25] = make_float4(fD.x, fD.y, fD.z, -0.05f);
        contacts[0].targetVel_maxImpulseW[25] = make_float4(0.0f, 0.0f, 0.0f, 3.0f);
    }

    // ---- rounds ----
    const AcsRound rounds[2] = {
        { 1.0f / 0.15f, 0.15f, 1.0f / 0.6f, -0.1f, 0.9f, 0.02f, 0.7f, 0.3f, 0.6f },
        { 1.0f / 0.07f, 0.07f, 1.0f / 0.35f, -0.4f, 0.5f, 0.008f, 0.0f, 0.0f, 0.35f },
    };

    // ---- device uploads (inputs once; outputs re-poisoned per round) ----
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t nsz) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, nsz);
        if (dp && p) shim.memcpyHtoD(dp, p, nsz);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dB2w = up(b2w.data(), b2w.size() * sizeof(b2w[0]));
    PxgDevicePtr dResp = up(resp.data(), resp.size() * sizeof(resp[0]));
    PxgDevicePtr dMv = up(mv.data(), mv.size() * sizeof(mv[0]));
    PxgDevicePtr dCfm = up(cfms.data(), cfms.size() * 4);
    PxgDevicePtr dLinks = up(links.data(), links.size() * sizeof(links[0]));
    for (uint32_t r = 0; r < ACS_NB_ARTI; ++r)
    {
        uint8_t* R = &rec[size_t(r) * ACS_REC_SIZE];
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
    PxgDevicePtr dCd = up(&cd, sizeof(cd));
    PxgDevicePtr dContacts = up(contacts.data(), contacts.size() * sizeof(contacts[0]));
    PxgDevicePtr dFp = up(&fp, sizeof(fp));
    PxgDevicePtr dFa = up(&fa, sizeof(fa));
    PxgDevicePtr dCh = up(nullptr, sizeof(PxgTGSBlockSolverContactHeader));
    PxgDevicePtr dFh = up(nullptr, sizeof(PxgTGSBlockSolverFrictionHeader));
    PxgDevicePtr dCp = up(nullptr, ACS_MAX_CONTACTS * sizeof(PxgTGSBlockSolverContactPoint));
    PxgDevicePtr dFr = up(nullptr, ACS_FRICTION_ROWS * sizeof(PxgTGSBlockSolverContactFriction));
    PxgDevicePtr dRs = up(nullptr, ACS_RESP_ROWS * sizeof(PxgArticulationBlockResponse));
    PxgDevicePtr dIo = up(nullptr, ACS_LANES * 16u);

    bool okAlloc = true;
    for (PxgDevicePtr p : bufs) okAlloc = okAlloc && (p != 0);
    if (!okAlloc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return;
    }

    for (uint32_t round = 0; round < 2; ++round)
    {
        const AcsRound& rd = rounds[round];

        // ---- expected via the host mirror ----
        PxgTGSBlockSolverContactHeader wantCh;
        PxgTGSBlockSolverFrictionHeader wantFh;
        std::vector<PxgTGSBlockSolverContactPoint> wantCp(ACS_MAX_CONTACTS);
        std::vector<PxgTGSBlockSolverContactFriction> wantFr(ACS_FRICTION_ROWS);
        std::vector<PxgArticulationBlockResponse> wantRs(ACS_RESP_ROWS);
        std::memset(&wantCh, 0xAB, sizeof(wantCh));
        std::memset(&wantFh, 0xAB, sizeof(wantFh));
        std::memset(wantCp.data(), 0xAB, wantCp.size() * sizeof(wantCp[0]));
        std::memset(wantFr.data(), 0xAB, wantFr.size() * sizeof(wantFr[0]));
        std::memset(wantRs.data(), 0xAB, wantRs.size() * sizeof(wantRs[0]));
        std::vector<uint32_t> wantIo(ACS_LANES * 4);
        std::memset(wantIo.data(), 0xAB, wantIo.size() * 4);

        uint32_t slopScaledHits = 0, linZeroHits = 0, greater2Lanes = 0;
        for (uint32_t lane = 0; lane < ACS_LANES; ++lane)
        {
            PxgSolverExtBody2 b0, b1;
            refCreateExtBody(nodes[lane * 2u], bodyIdx[lane * 2u], linkBases.data(),
                             b2w.data(), resp.data(), mv.data(), cfms.data(), links.data(),
                             sbd.data(), txi.data(), frames.data(), b0);
            refCreateExtBody(nodes[lane * 2u + 1u], bodyIdx[lane * 2u + 1u], linkBases.data(),
                             b2w.data(), resp.data(), mv.data(), cfms.data(), links.data(),
                             sbd.data(), txi.data(), frames.data(), b1);

            const uint32_t cc = acsContactCount(lane);
            const uint32_t wboff = 1000u + lane * 17u;
            const bool ppf = ((lane >> 1) & 1u) != 0u;
            const float2 tors = make_float2(rd.torsionalX, rd.torsionalY);

            const uint32_t written = refSetupExtContactsTGS(
                cd, contacts.data(), cc, fp, fa, b0, b1,
                rd.invDt, rd.stepDt, rd.invTotalDt, rd.bounceThreshold, rd.biasCoefficient,
                lane, wboff, ppf, wantCh, wantFh, wantCp.data(), wantFr.data(), wantRs.data(),
                cd.contactData[lane], rd.solverOffsetSlop, tors, rd.totalDt);
            wantIo[lane * 4u + 0u] = cc;
            wantIo[lane * 4u + 1u] = wboff;
            wantIo[lane * 4u + 2u] = ppf ? 1u : 0u;
            wantIo[lane * 4u + 3u] = written;

            // coverage bookkeeping (recomputed, NOT an oracle)
            const float4 n4 = cd.normal_restitutionW[lane];
            const PxVec3 n(n4.x, n4.y, n4.z);
            const float linNorVel = n.dot(b0.velocity.bottom) - n.dot(b1.velocity.bottom);
            for (uint32_t j = 0; j < cc; ++j)
            {
                const float4 p4 = contacts[j].point_separationW[lane];
                const PxVec3 pt(p4.x, p4.y, p4.z);
                const PxVec3 raXn = (pt - b0.body2World.p).cross(n);
                const PxVec3 rbXn = (pt - b1.body2World.p).cross(n);
                const float angNorVel = raXn.dot(b0.velocity.top) - rbXn.dot(b1.velocity.top);
                if (linNorVel == 0.0f && angNorVel != 0.0f) linZeroHits++;
                const float scale =
                    PxMax(linNorVel == 0.0f ? 1.0f : angNorVel / linNorVel, 1.0f);
                const float slop = rd.solverOffsetSlop * scale;
                const float comps[6] = { raXn.x, raXn.y, raXn.z, rbXn.x, rbXn.y, rbXn.z };
                for (float c : comps)
                    if (std::fabs(c) < slop && std::fabs(c) >= rd.solverOffsetSlop)
                        slopScaledHits++;
                const float vrel = (n.dot(b0.velocity.bottom) + raXn.dot(b0.velocity.top))
                                 - (n.dot(b1.velocity.bottom) + rbXn.dot(b1.velocity.top));
                const float pen = p4.w - cd.contactData[lane].restDistance;
                if ((n4.w > 1e-5f) && (rd.bounceThreshold > vrel)
                    && ((-vrel) > pen * rd.invTotalDt) && j == 0u)
                    greater2Lanes++;
            }
        }

        // the engineered pins, asserted through the mirror's own numbers:
        // PIN A: tvCond == 0 exactly -> targetVelocity 0 (the quirk arm)
        PXG_CHECK_MSG(wantCp[0].rbXn_targetVelW[2].w == 0.0f, "PIN A quirk arm");
        // PIN B: no bounce, cTargetVel 0.5 -> quirk yields (-1)*0.4
        PXG_CHECK_MSG(wantCp[0].rbXn_targetVelW[5].w == -0.4f, "PIN B quirk arm");
        // PIN E: bounce with cTargetVel 0 -> tvCond 1 -> (-vrel)*rest = 2*0.9
        PXG_CHECK_MSG(wantCp[0].rbXn_targetVelW[13].w == 2.0f * 0.9f, "PIN E bounce tv");
        // PIN D: bounce conditions hold but restitution 1e-6 stays sub-threshold
        PXG_CHECK_MSG(wantCp[0].rbXn_targetVelW[25].w == 0.0f, "PIN D sub-threshold");
        PXG_CHECK_MSG(linZeroHits > 0, "PIN C linNorVel==0 coverage");
        PXG_CHECK_MSG(slopScaledHits > 0, "scaled-slop coverage");
        PXG_CHECK_MSG(greater2Lanes >= 2, "isGreater2 coverage");

        // ---- poison the device outputs, run, compare ----
        auto pois = [&](PxgDevicePtr dp, size_t nsz) {
            std::vector<uint8_t> pz(nsz, 0xAB);
            shim.memcpyHtoD(dp, pz.data(), nsz);
        };
        pois(dCh, sizeof(wantCh));
        pois(dFh, sizeof(wantFh));
        pois(dCp, wantCp.size() * sizeof(wantCp[0]));
        pois(dFr, wantFr.size() * sizeof(wantFr[0]));
        pois(dRs, wantRs.size() * sizeof(wantRs[0]));
        {
            std::vector<uint32_t> ioInit(ACS_LANES * 4);
            std::memset(ioInit.data(), 0xAB, ioInit.size() * 4);
            for (uint32_t lane = 0; lane < ACS_LANES; ++lane)
            {
                ioInit[lane * 4u + 0u] = acsContactCount(lane);
                ioInit[lane * 4u + 1u] = 1000u + lane * 17u;
                ioInit[lane * 4u + 2u] = (lane >> 1) & 1u;
            }
            shim.memcpyHtoD(dIo, ioInit.data(), ioInit.size() * 4);
        }

        AcsParamsHost prm = {
            uint64_t(dCd), uint64_t(dContacts), uint64_t(dFp), uint64_t(dFa),
            uint64_t(dNodes), uint64_t(dBi), uint64_t(dRec), uint64_t(dSbd),
            uint64_t(dTxi), uint64_t(dFrames), uint64_t(dCh), uint64_t(dFh),
            uint64_t(dCp), uint64_t(dFr), uint64_t(dRs), uint64_t(dIo),
            rd.invDt, rd.stepDt, rd.invTotalDt, rd.bounceThreshold, rd.biasCoefficient,
            rd.solverOffsetSlop, rd.torsionalX, rd.torsionalY, rd.totalDt, ACS_LANES
        };
        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch acpSetupTest");
        shim.streamSynchronize();

        size_t bad = 0;
        auto cmp = [&](PxgDevicePtr dp, const void* wantp, size_t nsz, const char* tag) {
            std::vector<uint32_t> got(nsz / 4);
            shim.memcpyDtoH(got.data(), dp, nsz);
            const uint32_t* w = reinterpret_cast<const uint32_t*>(wantp);
            for (size_t k = 0; k < got.size(); ++k)
                // BITWISE, the sign of zero included: the P8.86/P8.91a +-0
                // tolerance (the 6x6 multiply's literal-zero seed fold) is gone
                // since the P9.86 select spelling of that seed add (P9.86c)
                if (w[k] != got[k])
                {
                    if (bad < 12)
                        std::printf("MM acs r%u %s word %zu (lane %zu)  ref %08x  gpu %08x\n",
                                    round, tag, k, k % 32, w[k], got[k]);
                    ++bad;
                }
        };
        cmp(dCh, &wantCh, sizeof(wantCh), "contactHeader");
        cmp(dFh, &wantFh, sizeof(wantFh), "frictionHeader");
        cmp(dCp, wantCp.data(), wantCp.size() * sizeof(wantCp[0]), "contactPoints");
        cmp(dFr, wantFr.data(), wantFr.size() * sizeof(wantFr[0]), "frictions");
        cmp(dRs, wantRs.data(), wantRs.size() * sizeof(wantRs[0]), "responses");
        cmp(dIo, wantIo.data(), wantIo.size() * 4, "io");
        PXG_CHECK_MSG(bad == 0, "round %u: %zu mismatches", round, bad);
    }

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
}
