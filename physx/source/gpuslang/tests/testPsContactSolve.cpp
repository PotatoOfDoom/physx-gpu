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

// psContactSolveBda.slang on PhysX 5.12: the particle SIDE of the particle/rigid contact
// solve.  Every oracle here is transcribed from the 5.12 CUDA (particlesystem.cu and
// deformableAndParticleUtils.cuh, see pxgParticleSolveRef.h), and every comparison is
// word for word.
//
//   psContactSolve_prepare            ps_contactPrepareLaunch over the shared scene
//                                     (pxgParticleSolveWorld.h): static / dynamic /
//                                     kinematic / articulation-link lanes, two particle
//                                     systems, 45 contacts over two blocks, the tail
//                                     lanes poisoned and checked untouched, the
//                                     appliedForces and deltaVel zeroing (and its NULL
//                                     form).
//   psContactSolve_queryRefCount      ps_queryPCContactReferenceCountLaunch (new in
//                                     5.12): per-rigid / per-link PC contact counts.
//   psContactSolve_particleSolve_*    ps_solvePCOutputParticleDeltaVLaunch for PGS / TGS
//                                     x position / velocity iterations: friction,
//                                     adhesion, the pair penBiasClamp, the rigid
//                                     reference count, a kinematic particle.
//   psContactSolve_accumulate         ps_accumulateDeltaVParticleLaunch over the runs.
//   psContactSolve_accumulateFemDeltaV ps_accumulateFEMParticleDeltaVLaunch: the FEM
//                                     reciprocal and the 5.12 mass-split count reset.
//   psContactSolve_penetratingParticle the end-to-end sanity: one particle in a static
//                                     floor is pushed out, and the depenetration clamp
//                                     (mPenBiasClamp, setMaxDepenetrationVelocity) holds.

#include "pxgTestFramework.h"
#include "pxgParticleSolveWorld.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgpsref;
using namespace pxgpsworld;

static const uint32_t PC_GRID = 1024u;
static const uint32_t AD_GRID = 32u;
static const float kDt = 1.0f / 60.0f;

static std::vector<PairInfo> pairsOf(const World& W)
{
    std::vector<PairInfo> v;
    for (const HostContact& c : W.contacts) v.push_back(c.pair);
    return v;
}
static std::vector<F4> normalPensOf(const World& W)
{
    std::vector<F4> v;
    for (const HostContact& c : W.contacts) v.push_back(c.normalPen);
    return v;
}
static std::vector<uint8_t> oracleBlocks(const World& W, std::vector<BlockLane>& lanes)
{
    const uint32_t NC = uint32_t(W.contacts.size());
    std::vector<uint8_t> blocks(((NC + 31u) / 32u) * RPC_STRIDE, 0);
    lanes.clear();
    for (uint32_t w = 0; w < NC; ++w)
    {
        lanes.push_back(W.prepare(W.contacts[w]));
        writeLane(blocks, w, lanes.back());
    }
    return blocks;
}

static void countKinds(const World& W, uint32_t kinds[4])
{
    kinds[0] = kinds[1] = kinds[2] = kinds[3] = 0;
    for (const HostContact& c : W.contacts)
    {
        const uint64_t r = c.pair.ref0;
        if (nodeIsStatic(r)) ++kinds[0];
        else if (nodeIsArticulation(r)) ++kinds[3];
        else if (W.bodies[W.solverBodyIndices[nodeIndex(r)]].linVel_invMass.w == 0.0f) ++kinds[2];
        else ++kinds[1];
    }
}

// ===========================================================================
// ps_contactPrepareLaunch
// ===========================================================================
PXG_TEST(psContactSolve_prepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x9e3779b9u);
    const uint32_t NC = uint32_t(W.contacts.size());

    uint32_t kinds[4];
    countKinds(W, kinds);
    PXG_CHECK_MSG(kinds[0] > 0 && kinds[1] > 0 && kinds[2] > 0 && kinds[3] > 0,
                  "coverage: static %u dynamic %u kinematic %u articulation %u", kinds[0], kinds[1], kinds[2], kinds[3]);

    std::vector<BlockLane> lanes;
    const std::vector<uint8_t> ref = oracleBlocks(W, lanes);

    for (int withDeltaVel = 1; withDeltaVel >= 0; --withDeltaVel)
    {
        // Poison every byte the kernel may not write: the tail lanes of the last block.
        std::vector<uint8_t> poison(ref.size(), 0xA7);
        PxgBufferHandle bBlocks = W.make(poison);
        PxgBufferHandle bPairs = W.make(pairsOf(W));
        PxgBufferHandle bNormalPens = W.make(normalPensOf(W));
        PxgBufferHandle bNum = W.make(std::vector<uint32_t>(1, NC));
        PxgBufferHandle bApplied = W.make(std::vector<F2>(NC + 8, F2{ 7.0f, -7.0f }));
        PxgBufferHandle bDeltaVel = W.make(std::vector<F4>(2 * NC + 8, F4{ 3.0f, 3.0f, 3.0f, 3.0f }));

        PrepPush p{};
        p.particleSystems = W.psysPool;
        p.sortedContactInfos = W.addr(bPairs);
        p.sortedContactNormalPens = W.addr(bNormalPens);
        p.numContacts = W.addr(bNum);
        p.contactBlocks = W.addr(bBlocks);
        p.preDesc = W.prePrepDesc;
        p.prepareDesc = W.prepDesc;
        p.appliedForces = W.addr(bApplied);
        p.dt = kDt;
        p.isTGSWord = 1u;
        p.deltaVel = withDeltaVel ? W.addr(bDeltaVel) : 0ull;
        p.sharedDesc = W.sharedDesc;
        Kernel k("psContactSolveBda.spv", "ps_contactPrepareLaunch", {}, sizeof(PrepPush));
        PXG_CHECK(k.isValid());
        if (!k.isValid()) return;
        k.dispatch(PC_GRID, {}, p);
        dev->flush();

        const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bBlocks, ref.size());
        uint32_t mm = 0;
        for (uint32_t w = 0; w < NC; ++w)
        {
            const BlockLane g = readLane(got, w);
            const BlockLane& r = lanes[w];
            const bool ok = bitsEqual(g.raXn_resp, r.raXn_resp) && bitsEqual(g.raXnF0_resp, r.raXnF0_resp) &&
                            bitsEqual(g.raXnF1_resp, r.raXnF1_resp) && bitsEqual(g.normal_errorW, r.normal_errorW) &&
                            bitsEqual(g.fricTan0_invMass0, r.fricTan0_invMass0) &&
                            bitsEqual(g.penBiasClampRigid, r.penBiasClampRigid);
            if (!ok && mm++ < 6)
                std::printf("  lane %u (rigid %016llx): raXn_resp got(%g,%g,%g,%g) ref(%g,%g,%g,%g) clamp got %g ref %g\n", w,
                            (unsigned long long)W.contacts[w].pair.ref0, g.raXn_resp.x, g.raXn_resp.y, g.raXn_resp.z,
                            g.raXn_resp.w, r.raXn_resp.x, r.raXn_resp.y, r.raXn_resp.z, r.raXn_resp.w,
                            g.penBiasClampRigid, r.penBiasClampRigid);
        }
        PXG_CHECK_MSG(mm == 0, "%u of %u prepared lanes differ from the 5.12 oracle", mm, NC);

        // The tail lanes of the partial block are never written.
        uint32_t tail = 0;
        for (uint32_t w = NC; w < ((NC + 31u) / 32u) * 32u; ++w)
        {
            const BlockLane g = readLane(got, w);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&g.raXn_resp);
            for (int b = 0; b < 16; ++b) tail += bytes[b] != 0xA7;
        }
        PXG_CHECK_MSG(tail == 0, "%u tail bytes written past numContacts", tail);

        const std::vector<F2> applied = downloadBuffer<F2>(dev, bApplied, NC + 8);
        uint32_t badApplied = 0;
        for (uint32_t w = 0; w < NC + 8; ++w)
        {
            const F2 want = (w < NC) ? F2{ 0.0f, 0.0f } : F2{ 7.0f, -7.0f };
            badApplied += !bitsEqual(applied[w], want);
        }
        PXG_CHECK_MSG(badApplied == 0, "appliedForces: %u slots wrong (zeroed [0,n), untouched after)", badApplied);

        const std::vector<F4> dv = downloadBuffer<F4>(dev, bDeltaVel, 2 * NC + 8);
        uint32_t badDv = 0;
        for (uint32_t w = 0; w < 2 * NC + 8; ++w)
        {
            const bool zeroed = withDeltaVel && w < 2 * NC;
            const F4 want = zeroed ? F4{ 0.f, 0.f, 0.f, 0.f } : F4{ 3.f, 3.f, 3.f, 3.f };
            badDv += !bitsEqual(dv[w], want);
        }
        PXG_CHECK_MSG(badDv == 0, "deltaVel (%s): %u slots wrong", withDeltaVel ? "non-null" : "NULL", badDv);
    }
}

// ===========================================================================
// ps_queryPCContactReferenceCountLaunch (new in 5.12)
// ===========================================================================
PXG_TEST(psContactSolve_queryRefCount)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x51ed270bu);
    const uint32_t NC = uint32_t(W.contacts.size());

    std::vector<BlockLane> lanes;
    const std::vector<uint8_t> blocks = oracleBlocks(W, lanes);
    const std::vector<uint32_t> ref = W.refCounts(lanes);

    // The host zeroes the buffer first (memsetD32Async); two extra slots stay untouched.
    PxgBufferHandle bRc = W.make(std::vector<uint32_t>(W.refCountSlots() + 2, 0u));
    PxgBufferHandle bBlocks = W.make(blocks);
    PxgBufferHandle bPairs = W.make(pairsOf(W));
    PxgBufferHandle bNum = W.make(std::vector<uint32_t>(1, NC));

    QueryPush p{};
    p.particleSystems = W.psysPool;
    p.sortedContactInfos = W.addr(bPairs);
    p.contactBlocks = W.addr(bBlocks);
    p.numContacts = W.addr(bNum);
    p.prePrepDesc = W.prePrepDesc;
    p.solverCoreDesc = W.coreDesc;
    p.solverBodyVelPool = W.velPoolAddr;
    p.rigidRefCount = W.addr(bRc);
    p.dt = kDt;
    p.isTGSWord = 0u;
    p.artiCoreDesc = W.artiCoreDesc;
    Kernel k("psContactSolveBda.spv", "ps_queryPCContactReferenceCountLaunch", {}, sizeof(QueryPush));
    PXG_CHECK(k.isValid());
    if (!k.isValid()) return;
    k.dispatch(PC_GRID, {}, p);
    dev->flush();

    const std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bRc, W.refCountSlots() + 2);
    uint32_t mm = 0, linkSlots = 0, bodySlots = 0;
    for (uint32_t i = 0; i < W.refCountSlots(); ++i)
    {
        if (got[i] != ref[i] && mm++ < 8)
            std::printf("  refCount[%u] got %u ref %u\n", i, got[i], ref[i]);
        if (ref[i] && i >= W.numSolverBodies) ++linkSlots;
        if (ref[i] && i < W.numSolverBodies) ++bodySlots;
    }
    PXG_CHECK_MSG(mm == 0, "%u reference-count slots differ", mm);
    PXG_CHECK_MSG(got[W.refCountSlots()] == 0u && got[W.refCountSlots() + 1] == 0u, "bumped past the slots");
    // Static (body 0) and kinematic (body 2) are never counted; both dynamics and two links are.
    PXG_CHECK_MSG(ref[0] == 0u && ref[2] == 0u, "static / kinematic must not be counted");
    PXG_CHECK_MSG(bodySlots == 2u && linkSlots == 2u, "coverage: %u body slots, %u link slots", bodySlots, linkSlots);
}

// ===========================================================================
// ps_solvePCOutputParticleDeltaVLaunch
// ===========================================================================
static void runParticleSolve(bool isTGS, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed);
    const uint32_t NC = uint32_t(W.contacts.size());

    std::vector<BlockLane> lanes;
    const std::vector<uint8_t> blocks = oracleBlocks(W, lanes);
    const std::vector<uint32_t> rc = W.refCounts(lanes);

    // Applied forces from "earlier iterations": a non-negative normal accumulator and a
    // signed tangent one.  The first lanes start at zero.
    Rng rng{ seed ^ 0x1234567u };
    std::vector<F2> applied(NC);
    for (uint32_t w = 0; w < NC; ++w)
        applied[w] = (w < 6u) ? F2{ 0.f, 0.f } : F2{ rng.uni(0.0f, 0.05f), rng.uni(-0.02f, 0.02f) };

    PxgBufferHandle bBlocks = W.make(blocks);
    PxgBufferHandle bPairs = W.make(pairsOf(W));
    PxgBufferHandle bNum = W.make(std::vector<uint32_t>(1, NC));
    PxgBufferHandle bApplied = W.make(applied);
    PxgBufferHandle bDelta = W.make(std::vector<F4>(NC + 4, F4{ 9.f, 9.f, 9.f, 9.f }));
    PxgBufferHandle bRc = W.make(rc);

    SolvePush p{};
    p.particleSystems = W.psysPool;
    p.sortedContactInfos = W.addr(bPairs);
    p.contactBlocks = W.addr(bBlocks);
    p.numContacts = W.addr(bNum);
    p.prePrepDesc = W.prePrepDesc;
    p.solverCoreDesc = W.coreDesc;
    p.solverBodyVelPool = W.velPoolAddr;
    p.deltaOut = W.addr(bDelta);
    p.appliedForces = W.addr(bApplied);
    p.rigidRefCount = W.addr(bRc);
    p.dt = kDt;
    p.flagsWord = (isVelocityIteration ? 1u : 0u) | (isTGS ? 0x100u : 0u);
    p.artiCoreDesc = W.artiCoreDesc;
    Kernel k("psContactSolveBda.spv", "ps_solvePCOutputParticleDeltaVLaunch", {}, sizeof(SolvePush));
    PXG_CHECK(k.isValid());
    if (!k.isValid()) return;
    k.dispatch(PC_GRID, {}, p);
    dev->flush();

    // ---- oracle ----
    const VelocityReader reader = W.reader();
    std::vector<F4> refDelta(NC + 4, F4{ 9.f, 9.f, 9.f, 9.f });
    std::vector<F2> refApplied = applied;
    uint32_t nApplied = 0, nKinematic = 0, nNotApplied = 0, nFriction = 0, nAdhesive = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const HostContact& c = W.contacts[w];
        const uint32_t s = particleSystemIdOf(c.pair.ref1), pi = particleIndexOf(c.pair.ref1);
        const ParticleView pv = W.particleView(s, pi);
        if (pv.sortedDeltaP.w == 0.f)
        {
            refDelta[w] = F4{ 0.f, 0.f, 0.f, 0.f };
            ++nKinematic;
            continue;
        }
        const uint64_t rigidId = c.pair.ref0;
        RigidPart rigid;
        rigid.readBodyProperties(W.gid(rigidId), lanes[w].fricTan0_invMass0.w, rc.data());
        rigid.readContactPrep(lanes[w]);
        rigid.readVelocity(reader, rigidId, isTGS);
        DeformablePart particle;
        particle.readParticle(pv, true, false, kDt, isVelocityIteration);
        ContactPair cp;
        cp.readRigidParticle(lanes[w], W.sys[s].penBiasClamp);
        SolveOut so;
        F2 af = applied[w];
        const bool hasApplied = solveRigidDbContact(rigid, particle, cp, so, af.x, af.y, kDt, true, false,
                                                    isVelocityIteration, true);
        if (hasApplied)
        {
            af.x = so.accumulatedDeltaLambdaN;
            af.y += so.deltaLambdaT;
            V3 delta;
            so.computeDelta(cp, delta, kDt);
            refDelta[w] = particle.writeParticle(delta, so);
            refApplied[w] = af;
            ++nApplied;
            nFriction += so.deltaLambdaT != 0.0f;
            nAdhesive += (particle.adhesion > 0.0f);
        }
        else
        {
            refDelta[w] = F4{ 0.f, 0.f, 0.f, 0.f };
            ++nNotApplied;
        }
    }

    const std::vector<F4> gotDelta = downloadBuffer<F4>(dev, bDelta, NC + 4);
    const std::vector<F2> gotApplied = downloadBuffer<F2>(dev, bApplied, NC);
    uint32_t mm = 0;
    for (uint32_t w = 0; w < NC + 4; ++w)
    {
        if (!bitsEqual(gotDelta[w], refDelta[w]) && mm++ < 6)
            std::printf("  deltaPos[%u] got(%g,%g,%g,%g) ref(%g,%g,%g,%g)\n", w, gotDelta[w].x, gotDelta[w].y,
                        gotDelta[w].z, gotDelta[w].w, refDelta[w].x, refDelta[w].y, refDelta[w].z, refDelta[w].w);
    }
    for (uint32_t w = 0; w < NC; ++w)
    {
        if (!bitsEqual(gotApplied[w], refApplied[w]) && mm++ < 12)
            std::printf("  applied[%u] got(%g,%g) ref(%g,%g)\n", w, gotApplied[w].x, gotApplied[w].y,
                        refApplied[w].x, refApplied[w].y);
    }
    PXG_CHECK_MSG(mm == 0, "%s %s: %u words differ from the 5.12 oracle", isTGS ? "TGS" : "PGS",
                  isVelocityIteration ? "velocity" : "position", mm);
    PXG_CHECK_MSG(nApplied > 4 && nKinematic > 0 && nFriction > 0 && nAdhesive > 0,
                  "coverage: applied %u not-applied %u kinematic %u friction %u adhesive %u",
                  nApplied, nNotApplied, nKinematic, nFriction, nAdhesive);
}

PXG_TEST(psContactSolve_particleSolve_pgsPosition) { runParticleSolve(false, false, 0x0badf00du); }
PXG_TEST(psContactSolve_particleSolve_pgsVelocity) { runParticleSolve(false, true, 0x0badf00du); }
PXG_TEST(psContactSolve_particleSolve_tgsPosition) { runParticleSolve(true, false, 0x7f4a7c15u); }
PXG_TEST(psContactSolve_particleSolve_tgsVelocity) { runParticleSolve(true, true, 0x7f4a7c15u); }

// ===========================================================================
// ps_accumulateDeltaVParticleLaunch
// ===========================================================================
PXG_TEST(psContactSolve_accumulate)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x2545f491u);
    const uint32_t NC = uint32_t(W.contacts.size());

    // Per-contact deltas with the live flag in .w (0 or 1), and the runs of equal ref1.
    Rng rng{ 0x77u };
    std::vector<F4> deltas(NC);
    for (uint32_t w = 0; w < NC; ++w)
        deltas[w] = F4{ rng.uni(-0.01f, 0.01f), rng.uni(-0.01f, 0.01f), rng.uni(-0.01f, 0.01f), (rng.next() & 1u) ? 1.f : 0.f };
    std::vector<uint32_t> rStart, rEnd;
    for (uint32_t w = 0; w < NC; ++w)
    {
        if (w == 0 || W.contacts[w].pair.ref1 != W.contacts[w - 1].pair.ref1)
        {
            if (w) rEnd.push_back(w);
            rStart.push_back(w);
        }
    }
    rEnd.push_back(NC);
    const uint32_t NR = uint32_t(rStart.size());

    PxgBufferHandle bPairs = W.make(pairsOf(W));
    PxgBufferHandle bDelta = W.make(deltas);
    PxgBufferHandle bStart = W.make(rStart);
    PxgBufferHandle bEnd = W.make(rEnd);
    PxgBufferHandle bNR = W.make(std::vector<uint32_t>(1, NR));

    AccumPush p{ W.psysPool, W.addr(bPairs), W.addr(bNR), W.addr(bStart), W.addr(bEnd), W.addr(bDelta) };
    Kernel k("psContactSolveBda.spv", "ps_accumulateDeltaVParticleLaunch", {}, sizeof(AccumPush));
    PXG_CHECK(k.isValid());
    if (!k.isValid()) return;
    k.dispatch(AD_GRID, {}, p);
    dev->flush();

    HostSystem ref[3];
    ref[0] = W.sys[0]; ref[2] = W.sys[2];
    uint32_t multi = 0;
    for (uint32_t r = 0; r < NR; ++r)
    {
        F4 acc = { 0.f, 0.f, 0.f, 0.f };
        float fcount = 0.f;
        for (uint32_t j = rStart[r]; j < rEnd[r]; ++j)
        {
            acc.x += deltas[j].x; acc.y += deltas[j].y; acc.z += deltas[j].z;
            fcount += deltas[j].w;
        }
        multi += (rEnd[r] - rStart[r]) > 1u;
        const float recip = 1.f / selectMax(1.f, fcount);
        acc.x *= recip; acc.y *= recip; acc.z *= recip;
        const uint64_t pr = W.contacts[rStart[r]].pair.ref1;
        F4& d = ref[particleSystemIdOf(pr)].accum[particleIndexOf(pr)];
        d.x += acc.x; d.y += acc.y; d.z += acc.z; d.w += acc.w;
    }
    uint32_t mm = 0;
    for (uint32_t s = 0; s < 3; s += 2)
    {
        const std::vector<F4> got = downloadBuffer<F4>(dev, W.bAccum[s], W.sys[s].accum.size());
        for (size_t i = 0; i < got.size(); ++i)
            if (!bitsEqual(got[i], ref[s].accum[i]) && mm++ < 6)
                std::printf("  sys %u accum[%zu] got(%g,%g,%g,%g) ref(%g,%g,%g,%g)\n", s, i, got[i].x, got[i].y,
                            got[i].z, got[i].w, ref[s].accum[i].x, ref[s].accum[i].y, ref[s].accum[i].z, ref[s].accum[i].w);
    }
    PXG_CHECK_MSG(mm == 0, "%u accumulated particles differ", mm);
    PXG_CHECK_MSG(multi > 0, "coverage: no run longer than one contact");
}

// ===========================================================================
// ps_accumulateFEMParticleDeltaVLaunch
//
// THE TWIN TRAP: the reciprocal is fcount > 0 ? 1/fcount : 0, NOT the particle twin's
// 1/max(1, fcount).  Four ranges separate them: fcount 0.5 (FEM doubles), 0 with a
// non-zero sum (FEM yields 0), exactly 1 (both agree), and 2.5 in the second system.
// 5.12 then resets the mass-split count: mAccumDeltaP.w = 0.  The key is ref0, the
// particle side of a particle-deformable pair; ref1 is a decoy deformable ref.
// ===========================================================================
PXG_TEST(psContactSolve_accumulateFemDeltaV)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct Range { uint32_t sysId, partId; std::vector<F4> deltas; };
    const std::vector<Range> ranges = {
        { 0u, 3u,  { { 1.0f, 2.0f, 3.0f, 0.25f }, { 0.5f, -1.0f, 0.25f, 0.25f } } },
        { 0u, 7u,  { { 4.0f, -2.0f, 1.5f, 0.0f }, { -1.0f, 3.0f, -0.5f, 0.0f } } },
        { 0u, 11u, { { 2.0f, 0.5f, -3.0f, 0.5f }, { 1.0f, 1.5f, 2.0f, 0.5f } } },
        { 1u, 5u,  { { 1.0f, 1.0f, 1.0f, 1.5f }, { 2.0f, -2.0f, 0.5f, 0.75f }, { -0.5f, 0.25f, 3.0f, 0.25f } } },
    };

    std::vector<PairInfo> infos;
    std::vector<F4> deltaP;
    std::vector<uint32_t> rStart, rEnd;
    for (const Range& r : ranges)
    {
        rStart.push_back(uint32_t(infos.size()));
        for (const F4& d : r.deltas)
        {
            PairInfo pi;
            pi.ref0 = particleRef(r.sysId, r.partId);
            pi.ref1 = 0xA5A5A5A5A5A5A5A5ull;   // decoy: the deformable ref, never read here
            infos.push_back(pi);
            deltaP.push_back(d);
        }
        rEnd.push_back(uint32_t(infos.size()));
    }
    const uint32_t NR = uint32_t(ranges.size());

    const uint32_t NP = 16u;
    std::vector<F4> accum0(NP), accum1(NP);
    for (uint32_t i = 0; i < NP; ++i)
    {
        accum0[i] = { 100.0f + float(i), 200.0f + float(i), 300.0f + float(i), 7.5f + float(i) };
        accum1[i] = { -100.0f - float(i), -200.0f - float(i), -300.0f - float(i), 9.5f + float(i) };
    }

    std::vector<PxgBufferHandle> owned;
    auto mk = [&](auto const& v) { PxgBufferHandle h = createBufferFrom(dev, v); owned.push_back(h); return h; };
    PxgBufferHandle bAcc0 = mk(accum0), bAcc1 = mk(accum1);
    std::vector<uint8_t> systems(2u * SZ_PARTICLE_SYS, 0);
    putU64(systems, 152, dev->getBufferDeviceAddress(bAcc0));
    putU64(systems, SZ_PARTICLE_SYS + 152, dev->getBufferDeviceAddress(bAcc1));
    PxgBufferHandle bSys = mk(systems), bInfos = mk(infos), bDelta = mk(deltaP), bStart = mk(rStart),
                    bEnd = mk(rEnd), bPairs = mk(std::vector<uint32_t>(1, NR));

    AccumPush push = { dev->getBufferDeviceAddress(bSys), dev->getBufferDeviceAddress(bInfos),
                       dev->getBufferDeviceAddress(bPairs), dev->getBufferDeviceAddress(bStart),
                       dev->getBufferDeviceAddress(bEnd), dev->getBufferDeviceAddress(bDelta) };
    {
        Kernel k("psContactSolveBda.spv", "ps_accumulateFEMParticleDeltaVLaunch", {}, sizeof(AccumPush));
        PXG_CHECK(k.isValid());
        if (k.isValid())
        {
            k.dispatch(AD_GRID, {}, push);
            dev->flush();
        }
    }

    std::vector<F4> ref0 = accum0, ref1 = accum1;
    for (uint32_t r = 0; r < NR; ++r)
    {
        F4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
        float fcount = 0.0f;
        for (uint32_t j = rStart[r]; j < rEnd[r]; ++j)
        {
            acc.x += deltaP[j].x; acc.y += deltaP[j].y; acc.z += deltaP[j].z;
            fcount += deltaP[j].w;
        }
        const float recip = fcount > 0.0f ? 1.0f / fcount : 0.0f;
        acc.x *= recip; acc.y *= recip; acc.z *= recip;
        F4& dst = (ranges[r].sysId == 0u) ? ref0[ranges[r].partId] : ref1[ranges[r].partId];
        dst.x += acc.x; dst.y += acc.y; dst.z += acc.z; dst.w += acc.w;
        dst.w = 0.0f;   // 5.12: the mass-split count reset
    }

    const auto g0 = downloadBuffer<F4>(dev, bAcc0, NP);
    const auto g1 = downloadBuffer<F4>(dev, bAcc1, NP);
    uint32_t mm = 0;
    for (uint32_t i = 0; i < NP; ++i)
    {
        if (!bitsEqual(g0[i], ref0[i]) && mm++ < 6)
            std::printf("  sys0[%u] got(%g,%g,%g,%g) ref(%g,%g,%g,%g)\n", i, g0[i].x, g0[i].y, g0[i].z, g0[i].w,
                        ref0[i].x, ref0[i].y, ref0[i].z, ref0[i].w);
        if (!bitsEqual(g1[i], ref1[i]) && mm++ < 6)
            std::printf("  sys1[%u] got(%g,%g,%g,%g) ref(%g,%g,%g,%g)\n", i, g1[i].x, g1[i].y, g1[i].z, g1[i].w,
                        ref1[i].x, ref1[i].y, ref1[i].z, ref1[i].w);
    }
    PXG_CHECK_MSG(mm == 0, "%u FEM-accumulated particles differ", mm);
    // The trap itself, in values: particle 7's range has a non-zero sum and fcount 0.
    PXG_CHECK_MSG(bitsEqual(g0[7].x, accum0[7].x) && g0[7].w == 0.0f, "fcount 0 must add exactly 0 and reset .w");
    for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
}

// ===========================================================================
// End to end: one particle penetrating a static floor is pushed out, and the
// depenetration clamp holds (5.10: setMaxDepenetrationVelocity is honoured).
// ===========================================================================
static F4 runFloorChain(float penBiasClamp, float pen)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x13579bdfu);
    // One contact: system 0 particle 0 (invMass 1, zero delta) in the static floor, normal up.
    W.sys[0].deltaP[0] = F4{ 0.f, 0.f, 0.f, 1.f };
    W.sys[0].accum[0] = F4{ 0.f, 0.f, 0.f, 0.f };
    W.sys[0].penBiasClamp = penBiasClamp;
    W.contacts.clear();
    HostContact c;
    c.pair.ref0 = staticNode();
    c.pair.ref1 = particleRef(0, 0);
    c.normalPen = F4{ 0.f, 1.f, 0.f, pen };
    W.contacts.push_back(c);
    // A floor at rest: every rigid velocity source zero.
    for (F4& v : W.outVel) v = F4{ 0.f, 0.f, 0.f, 0.f };
    for (F4& v : W.velPool) v = F4{ 0.f, 0.f, 0.f, 0.f };
    W.uploadVelocities();

    std::vector<uint8_t> pool(3 * SZ_PARTICLE_SYS, 0);
    std::vector<PxgBufferHandle> owned;
    auto mk = [&](auto const& v) { PxgBufferHandle h = createBufferFrom(dev, v); owned.push_back(h); return h; };
    PxgBufferHandle bDeltaP = mk(W.sys[0].deltaP), bAccum = mk(W.sys[0].accum), bPos = mk(W.sys[0].pos),
                    bPhases = mk(W.sys[0].phases), bMats = mk(W.sys[0].materials),
                    bP2m = mk(std::vector<uint32_t>{ uint32_t(W.sys[0].phaseToMat[0]) | (uint32_t(W.sys[0].phaseToMat[1]) << 16) }),
                    bVel = mk(W.sys[0].vel);
    putU64(pool, 24, dev->getBufferDeviceAddress(bPos));
    putU64(pool, 32, dev->getBufferDeviceAddress(bVel));
    putU64(pool, 40, dev->getBufferDeviceAddress(bPhases));
    putU64(pool, 80, dev->getBufferDeviceAddress(bMats));
    putU64(pool, 152, dev->getBufferDeviceAddress(bAccum));
    putU64(pool, 160, dev->getBufferDeviceAddress(bDeltaP));
    putF32(pool, 268, W.sys[0].restOffset);
    putF32(pool, 320, penBiasClamp);
    putU32(pool, 592, MAT_STRIDE);
    putU64(pool, 600, dev->getBufferDeviceAddress(bP2m));
    const uint64_t poolAddr = dev->getBufferDeviceAddress(mk(pool));

    const uint64_t pairs = dev->getBufferDeviceAddress(mk(std::vector<PairInfo>{ c.pair }));
    const uint64_t normalPens = dev->getBufferDeviceAddress(mk(std::vector<F4>{ c.normalPen }));
    const uint64_t num = dev->getBufferDeviceAddress(mk(std::vector<uint32_t>(1, 1u)));
    const uint64_t blocks = dev->getBufferDeviceAddress(mk(std::vector<uint8_t>(RPC_STRIDE, 0)));
    const uint64_t applied = dev->getBufferDeviceAddress(mk(std::vector<F2>(1, F2{ 0.f, 0.f })));
    const uint64_t deltaPos = dev->getBufferDeviceAddress(mk(std::vector<F4>(1, F4{ 0.f, 0.f, 0.f, 0.f })));
    const uint64_t rc = dev->getBufferDeviceAddress(mk(std::vector<uint32_t>(W.refCountSlots(), 0u)));
    const uint64_t start = dev->getBufferDeviceAddress(mk(std::vector<uint32_t>(1, 0u)));
    const uint64_t end = dev->getBufferDeviceAddress(mk(std::vector<uint32_t>(1, 1u)));

    {
        PrepPush p{ poolAddr, pairs, normalPens, num, blocks, W.prePrepDesc, W.prepDesc, applied, kDt, 0u, 0ull, W.sharedDesc };
        Kernel k("psContactSolveBda.spv", "ps_contactPrepareLaunch", {}, sizeof(PrepPush));
        k.dispatch(PC_GRID, {}, p);
    }
    {
        QueryPush p{ poolAddr, pairs, blocks, num, W.prePrepDesc, W.coreDesc, W.velPoolAddr, rc, kDt, 0u, W.artiCoreDesc };
        Kernel k("psContactSolveBda.spv", "ps_queryPCContactReferenceCountLaunch", {}, sizeof(QueryPush));
        k.dispatch(PC_GRID, {}, p);
    }
    {
        SolvePush p{ poolAddr, pairs, blocks, num, W.prePrepDesc, W.coreDesc, W.velPoolAddr, deltaPos, applied, rc,
                     kDt, 0x100u /*TGS, position iteration*/, W.artiCoreDesc };
        Kernel k("psContactSolveBda.spv", "ps_solvePCOutputParticleDeltaVLaunch", {}, sizeof(SolvePush));
        k.dispatch(PC_GRID, {}, p);
    }
    {
        AccumPush p{ poolAddr, pairs, num, start, end, deltaPos };
        Kernel k("psContactSolveBda.spv", "ps_accumulateDeltaVParticleLaunch", {}, sizeof(AccumPush));
        k.dispatch(AD_GRID, {}, p);
    }
    dev->flush();
    const F4 out = downloadBuffer<F4>(dev, bAccum, 1)[0];
    for (PxgBufferHandle h : owned) dev->destroyBuffer(h);
    return out;
}

PXG_TEST(psContactSolve_penetratingParticle)
{
    // The static floor's clamp is -1e32, so the pair clamp is the particle system's.
    // Unbounded, the whole -0.1 penetration is resolved in one solve: CN = -0.1 / dt,
    // deltaLambdaN = 0.1 / dt / invMass, delta = 0.1 along the floor normal.
    const F4 unclamped = runFloorChain(-1e30f, -0.1f);
    PXG_CHECK_MSG(std::fabs(unclamped.y - 0.1f) < 1e-6f && unclamped.x == 0.0f && unclamped.z == 0.0f,
                  "the particle must be pushed out by the penetration: (%g,%g,%g)", unclamped.x, unclamped.y, unclamped.z);

    // mPenBiasClamp = -1 m/s caps the depenetration velocity: one solve moves the
    // particle by dt * 1 m/s.
    const F4 clamped = runFloorChain(-1.0f, -0.1f);
    PXG_CHECK_MSG(std::fabs(clamped.y - kDt) < 1e-6f, "the clamp must cap the push-out at dt * 1 m/s: %g", clamped.y);
}

} // namespace
