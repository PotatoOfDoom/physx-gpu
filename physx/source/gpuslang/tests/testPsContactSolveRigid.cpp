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

// psContactSolveRigidBda.slang on PhysX 5.12: ps_solvePCOutputRigidDeltaVLaunch, the RIGID
// side of the particle/rigid contact solve, over the shared scene of
// pxgParticleSolveWorld.h (45 contacts, two blocks, two particle systems, every rigid
// kind).  5.12 folded the TGS twin into isTGS, so the four tests are PGS / TGS x
// position / velocity iterations of the one kernel.  Checked word for word against the
// host transcription of the 5.12 CUDA (pxgParticleSolveRef.h):
//
//   * a STATIC rigid skips the lane: its two deltaVel slots and its applied force keep
//     their sentinels;
//   * a KINEMATIC rigid (block invMass 0) writes explicit zeros to both deltaVel slots
//     and to the applied force;
//   * a dynamic rigid or an articulation link writes PxgRigidPart::writeContactDeltas
//     (lin * refCount with count 1 in .w, ang * refCount) on every visit, applied or
//     not, and the applied force only when an impulse was applied;
//   * the mass split: the rigid response is scaled by the rigid's PC contact count.

#include "pxgTestFramework.h"
#include "pxgParticleSolveWorld.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgpsref;
using namespace pxgpsworld;

static const float kDt = 1.0f / 60.0f;

static void runRigidSolve(bool isTGS, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed);
    const uint32_t NC = uint32_t(W.contacts.size());

    std::vector<BlockLane> lanes;
    std::vector<uint8_t> blocks(((NC + 31u) / 32u) * RPC_STRIDE, 0);
    std::vector<PairInfo> pairs;
    for (uint32_t w = 0; w < NC; ++w)
    {
        lanes.push_back(W.prepare(W.contacts[w]));
        writeLane(blocks, w, lanes.back());
        pairs.push_back(W.contacts[w].pair);
    }
    const std::vector<uint32_t> rc = W.refCounts(lanes);

    Rng rng{ seed ^ 0xabcdefu };
    std::vector<F2> applied(NC);
    for (uint32_t w = 0; w < NC; ++w)
        applied[w] = (w % 5u == 0u) ? F2{ 0.f, 0.f } : F2{ rng.uni(0.0f, 0.05f), rng.uni(-0.02f, 0.02f) };
    const F4 sentinel = F4{ 5.f, -5.f, 5.f, -5.f };
    std::vector<F4> deltaVel(2 * NC, sentinel);

    PxgBufferHandle bBlocks = W.make(blocks);
    PxgBufferHandle bPairs = W.make(pairs);
    PxgBufferHandle bNum = W.make(std::vector<uint32_t>(1, NC));
    PxgBufferHandle bApplied = W.make(applied);
    PxgBufferHandle bDeltaVel = W.make(deltaVel);
    PxgBufferHandle bRc = W.make(rc);

    SolvePush p{};
    p.particleSystems = W.psysPool;
    p.sortedContactInfos = W.addr(bPairs);
    p.contactBlocks = W.addr(bBlocks);
    p.numContacts = W.addr(bNum);
    p.prePrepDesc = W.prePrepDesc;
    p.solverCoreDesc = W.coreDesc;
    p.solverBodyVelPool = W.velPoolAddr;
    p.deltaOut = W.addr(bDeltaVel);
    p.appliedForces = W.addr(bApplied);
    p.rigidRefCount = W.addr(bRc);
    p.dt = kDt;
    p.flagsWord = (isVelocityIteration ? 1u : 0u) | (isTGS ? 0x100u : 0u);
    p.artiCoreDesc = W.artiCoreDesc;
    Kernel k("psContactSolveRigidBda.spv", "ps_solvePCOutputRigidDeltaVLaunch", {}, sizeof(SolvePush));
    PXG_CHECK(k.isValid());
    if (!k.isValid()) return;
    k.dispatch(1024u, {}, p);
    dev->flush();

    // ---- oracle (particlesystem.cu ps_solvePCOutputRigidDeltaVLaunch, 5.12) ----
    const VelocityReader reader = W.reader();
    std::vector<F4> refDv = deltaVel;
    std::vector<F2> refApplied = applied;
    uint32_t nStatic = 0, nKinematic = 0, nWritten = 0, nApplied = 0, nLink = 0, nSplit = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const HostContact& c = W.contacts[w];
        const uint64_t rigidId = c.pair.ref0;
        if (nodeIsStatic(rigidId))
        {
            ++nStatic;
            continue;
        }
        const float rigidInvMass = lanes[w].fricTan0_invMass0.w;
        if (rigidInvMass == 0.f)
        {
            refDv[w] = F4{ 0.f, 0.f, 0.f, 0.f };
            refDv[w + NC] = F4{ 0.f, 0.f, 0.f, 0.f };
            refApplied[w] = F2{ 0.f, 0.f };
            ++nKinematic;
            continue;
        }
        const uint32_t s = particleSystemIdOf(c.pair.ref1), pi = particleIndexOf(c.pair.ref1);
        const ParticleView pv = W.particleView(s, pi);
        F2 af = applied[w];

        RigidPart rigid;
        rigid.readBodyProperties(W.gid(rigidId), rigidInvMass, rc.data());
        rigid.readContactPrep(lanes[w]);
        rigid.readVelocity(reader, rigidId, isTGS);
        DeformablePart particle;
        particle.readParticle(pv, true, false, kDt, isVelocityIteration);
        ContactPair cp;
        cp.readRigidParticle(lanes[w], W.sys[s].penBiasClamp);
        SolveOut so;
        const bool hasApplied = solveRigidDbContact(rigid, particle, cp, so, af.x, af.y, kDt, true, false,
                                                    isVelocityIteration, true);
        if (hasApplied)
        {
            af.x = so.accumulatedDeltaLambdaN;
            af.y += so.deltaLambdaT;
            refApplied[w] = af;
            ++nApplied;
        }
        rigid.writeContactDeltas(refDv.data(), rigidId, cp, so, w, w + NC);
        ++nWritten;
        nLink += nodeIsArticulation(rigidId);
        nSplit += rigid.referenceCount > 1u;
    }

    const std::vector<F4> gotDv = downloadBuffer<F4>(dev, bDeltaVel, 2 * NC);
    const std::vector<F2> gotApplied = downloadBuffer<F2>(dev, bApplied, NC);
    uint32_t mm = 0;
    for (uint32_t i = 0; i < 2 * NC; ++i)
        if (!bitsEqual(gotDv[i], refDv[i]) && mm++ < 8)
            std::printf("  deltaVel[%u] (rigid %016llx) got(%g,%g,%g,%g) ref(%g,%g,%g,%g)\n", i,
                        (unsigned long long)W.contacts[i % NC].pair.ref0, gotDv[i].x, gotDv[i].y, gotDv[i].z,
                        gotDv[i].w, refDv[i].x, refDv[i].y, refDv[i].z, refDv[i].w);
    for (uint32_t i = 0; i < NC; ++i)
        if (!bitsEqual(gotApplied[i], refApplied[i]) && mm++ < 16)
            std::printf("  applied[%u] got(%g,%g) ref(%g,%g)\n", i, gotApplied[i].x, gotApplied[i].y,
                        refApplied[i].x, refApplied[i].y);
    PXG_CHECK_MSG(mm == 0, "%s %s: %u words differ from the 5.12 oracle", isTGS ? "TGS" : "PGS",
                  isVelocityIteration ? "velocity" : "position", mm);
    PXG_CHECK_MSG(nStatic > 0 && nKinematic > 0 && nApplied > 0 && nLink > 0 && nSplit > 0 && nWritten > nApplied - 1,
                  "coverage: static %u kinematic %u written %u applied %u link %u split %u",
                  nStatic, nKinematic, nWritten, nApplied, nLink, nSplit);
}

PXG_TEST(psContactSolveRigid_pgsPosition) { runRigidSolve(false, false, 0x3c6ef372u); }
PXG_TEST(psContactSolveRigid_pgsVelocity) { runRigidSolve(false, true, 0x3c6ef372u); }
PXG_TEST(psContactSolveRigid_tgsPosition) { runRigidSolve(true, false, 0xa54ff53au); }
PXG_TEST(psContactSolveRigid_tgsVelocity) { runRigidSolve(true, true, 0xa54ff53au); }

} // namespace
