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

// pxgParticleSolveWorld.h -- one particle/rigid contact scene, on the host and on the
// device, shared by testPsContactSolve and testPsContactSolveRigid (PhysX 5.12).
//
// THE SCENE.  A pool of three PxgParticleSystem descriptors (608 B each): systems 0 and
// 2 carry particles, slot 1 is a zero decoy no contact names.  Every particle field the
// kernels read differs between the two systems (inverse masses incl. a kinematic
// particle, running deltas, velocities, phases with flag bits above the group mask, a
// PxU16 phase->material map, a runtime material stride of 24 B, rest offsets, and
// mPenBiasClamp).  The rigid side has every participant kind the CUDA distinguishes:
//
//   node static        the world body (solver body 0), penBiasClamp -1e32
//   node 4 -> body 1   dynamic, non-diagonal sqrtInvInertia
//   node 6 -> body 2   kinematic (invMass 0): the identity-inertia, scale-0 arm
//   node 8 -> body 3   dynamic
//   node 10            ARTICULATION 1 (bodySim remap 1, solver articulation index 1,
//                      maxLinks 4, nbArticulations 2), links 1 and 2 contacted
//
// 45 contacts (two blocks of 32, the second partial), sorted by particle ref so that
// the particle-side accumulate ranges are the runs.  Normal / penetration / point data
// is pseudo-random from a fixed seed; the oracle takes no shortcut on any of it.
//
// Offsets are the 5.12 mirrors: PxgParticleSystem (pxgSlangParticleDesc), PxgSolverBodyData
// / PxgSolverTxIData / PxgBodySim (pxgSlangCommon), PxgConstraintPrepareDesc / PxgPrePrepDesc
// (pxgSlangPrepDesc), PxgSolverCoreDesc / PxgSolverSharedDescBase (pxgSlangSolverDesc),
// PxgArticulation / PxgArticulationCoreDesc (pxgSlangArticulationDesc).

#ifndef PXG_PARTICLE_SOLVE_WORLD_H
#define PXG_PARTICLE_SOLVE_WORLD_H

#include "pxgTestFramework.h"
#include "pxgParticleSolveRef.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pxgpsworld
{
using namespace pxgtest;
using namespace pxgpsref;

static const uint32_t SZ_PARTICLE_SYS = 608u;
static const uint32_t SZ_PREP_DESC = 304u;
static const uint32_t SZ_PREPREP_DESC = 760u;
static const uint32_t SZ_CORE_DESC = 320u;
static const uint32_t SZ_SHARED_PGS = 224u;
static const uint32_t SZ_SOLVER_BODY = 96u;
static const uint32_t SZ_TXI = 64u;
static const uint32_t SZ_ALIGNED_XF = 32u;
static const uint32_t SZ_BODYSIM = 224u;
static const uint32_t SZ_ARTI = 496u;
static const uint32_t SZ_ARTI_CORE = 512u;
static const uint32_t SZ_ARTI_LINK = 80u;
static const uint32_t SZ_PXTRANSFORM = 28u;
static const uint32_t SZ_SPATIAL_MATRIX = 144u;

// PxgRigidParticleContactBlock: five float4[32] and one float[32] -> 2688 B.
static const uint32_t RPC_STRIDE = 2688u;
static const uint32_t RPC_raXn_resp = 0u, RPC_raXnF0_resp = 512u, RPC_raXnF1_resp = 1024u,
                      RPC_normal_errorW = 1536u, RPC_fricTan0_invMass0 = 2048u, RPC_penBiasClampRigid = 2560u;

static const uint32_t MAT_STRIDE = 24u;   // runtime mParticleMaterialStride (bytes)

static inline void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static inline void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
static inline void putF32(std::vector<uint8_t>& b, size_t off, float v) { std::memcpy(b.data() + off, &v, 4); }
static inline void putF4(std::vector<uint8_t>& b, size_t off, F4 v) { std::memcpy(b.data() + off, &v, 16); }

struct Rng
{
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uni(float lo, float hi) { return lo + (hi - lo) * float(next() & 0xffffffu) / float(0x1000000); }
};

struct PairInfo { uint64_t ref0, ref1; };
static_assert(sizeof(PairInfo) == 16, "PxgDeformablePairInfo stride");

struct HostSystem
{
    std::vector<F4> pos, deltaP, vel, accum;
    std::vector<uint32_t> phases;
    std::vector<uint16_t> phaseToMat;
    std::vector<float> materials;   // MAT_STRIDE / 4 floats per material
    float restOffset = 0.0f;
    float penBiasClamp = 0.0f;
};

struct HostContact
{
    PairInfo pair;
    F4 normalPen;
};

struct World
{
    // ---- host ----
    HostSystem sys[3];
    std::vector<HostContact> contacts;
    uint32_t numSolverBodies = 5u;
    uint32_t accOffset = 7u;
    uint32_t maxLinks = 4u, nbArticulations = 2u;
    std::vector<uint32_t> solverBodyIndices;
    // q: the body2WorldPool rotation.  The particle kernels never read it; the softbody
    // attachment prepare rotates the anchor by it (pxgSoftBodySolveRef).
    struct Body { F4 angVel_penBiasClamp, linVel_invMass; float sqrtInvInertia[9]; F4 p; F4 q; };
    std::vector<Body> bodies;
    struct Link { float penBiasClamp; F4 q; V3 p; float response[36]; };
    std::vector<Link> links;   // the contacted articulation's links
    uint32_t artiSolverIndex = 1u, artiRemapId = 1u, artiNode = 10u;
    std::vector<F4> outVel, velPool, artiVel;

    // ---- device ----
    PxgSlangComputeDevice* dev = nullptr;
    std::vector<PxgBufferHandle> owned;
    uint64_t psysPool = 0, prePrepDesc = 0, prepDesc = 0, coreDesc = 0, sharedDesc = 0, artiCoreDesc = 0;
    uint64_t velPoolAddr = 0;
    PxgBufferHandle bAccum[3] = { 0, 0, 0 };
    PxgBufferHandle bOutVel = 0, bVelPool = 0, bArtiVel = 0;

    // Re-uploads the three velocity sources after a test edited them on the host.
    void uploadVelocities()
    {
        dev->upload(bOutVel, outVel.data(), outVel.size() * sizeof(F4));
        dev->upload(bVelPool, velPool.data(), velPool.size() * sizeof(F4));
        dev->upload(bArtiVel, artiVel.data(), artiVel.size() * sizeof(F4));
    }

    uint64_t addr(PxgBufferHandle h) const { return dev->getBufferDeviceAddress(h); }

    template <typename T>
    PxgBufferHandle make(const std::vector<T>& v)
    {
        PxgBufferHandle h = createBufferFrom(dev, v);
        owned.push_back(h);
        return h;
    }

    ~World()
    {
        for (PxgBufferHandle h : owned)
            dev->destroyBuffer(h);
    }

    // ---- oracle views ----
    ParticleView particleView(uint32_t s, uint32_t i) const
    {
        const HostSystem& hs = sys[s];
        const uint32_t group = hs.phases[i] & 0x000fffffu;
        const uint32_t mi = hs.phaseToMat[group];
        const float* m = &hs.materials[mi * (MAT_STRIDE / 4u)];
        ParticleView v;
        v.sortedDeltaP = hs.deltaP[i];
        v.sortedVelocity = hs.vel[i];
        v.accumDeltaP = hs.accum[i];
        v.restOffset = hs.restOffset;
        v.friction = m[0];
        v.adhesion = m[2];
        v.adhesionRadiusScale = m[4];
        return v;
    }

    VelocityReader reader() const
    {
        VelocityReader r;
        r.solverBodyIndices = solverBodyIndices.data();
        r.artiLinkVelocities = artiVel.data();
        r.maxLinks = maxLinks;
        r.nbArticulations = nbArticulations;
        r.artiOffset = maxLinks * nbArticulations;
        r.numSolverBodies = numSolverBodies;
        r.solverBodyDeltaVel = velPool.data() + accOffset;
        r.initialVel = outVel.data();
        return r;
    }

    // ps_contactPrepareLaunch, transcribed (particlesystem.cu, PhysX 5.12).
    BlockLane prepare(const HostContact& c) const
    {
        const uint32_t s = particleSystemIdOf(c.pair.ref1);
        const uint32_t pi = particleIndexOf(c.pair.ref1);
        const V3 point = xyz(sys[s].pos[pi]);
        const V3 normal = v3(-c.normalPen.x, -c.normalPen.y, -c.normalPen.z);
        const float pen = c.normalPen.w;
        const uint64_t rigidId = c.pair.ref0;
        uint32_t idx = 0;
        if (!nodeIsStatic(rigidId))
            idx = solverBodyIndices[nodeIndex(rigidId)];

        BlockLane b;
        if (nodeIsArticulation(rigidId))
        {
            const Link& L = links[nodeLinkId(rigidId)];
            V3 t0, t1;
            computeBasisVectors(normal, t0, t1);
            const V3 ra = sub(point, L.p);
            const V3 raXn = cross(ra, normal);
            const V3 raXF0 = cross(ra, t0);
            const V3 raXF1 = cross(ra, t1);
            V3 top, bottom;
            spatialMul(L.response, normal, raXn, top, bottom);   // deltaV0
            const float resp0 = dot(top, raXn) + dot(bottom, normal);
            // UPSTREAM: respF0 / respF1 are formed from deltaV0, not deltaV1 / deltaV2.
            const float respF0 = dot(top, raXF0) + dot(bottom, t0);
            const float respF1 = dot(top, raXF1) + dot(bottom, t1);
            b.raXn_resp = F4{ raXn.x, raXn.y, raXn.z, resp0 };
            b.normal_errorW = F4{ normal.x, normal.y, normal.z, pen };
            b.fricTan0_invMass0 = F4{ t0.x, t0.y, t0.z, 1.f };
            b.raXnF0_resp = F4{ raXF0.x, raXF0.y, raXF0.z, respF0 };
            b.raXnF1_resp = F4{ raXF1.x, raXF1.y, raXF1.z, respF1 };
            b.penBiasClampRigid = L.penBiasClamp;
        }
        else
        {
            const Body& B = bodies[idx];
            const float invMass0 = B.linVel_invMass.w;
            float inv[9];
            float inertiaScale = 1.f;
            if (idx != 0 && invMass0 == 0.f)
            {
                const float id[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
                std::memcpy(inv, id, sizeof(inv));
                inertiaScale = 0.f;
            }
            else
                std::memcpy(inv, B.sqrtInvInertia, sizeof(inv));
            V3 t0, t1;
            computeBasisVectors(normal, t0, t1);
            const V3 ra = sub(point, xyz(B.p));
            const V3 raXn = cross(ra, normal);
            const V3 raXF0 = cross(ra, t0);
            const V3 raXF1 = cross(ra, t1);
            const V3 a = mat33Mul(inv, raXn);
            const float resp0 = (dot(a, a)) * inertiaScale + invMass0;
            const V3 f0 = mat33Mul(inv, raXF0);
            const V3 f1 = mat33Mul(inv, raXF1);
            const float respF0 = (dot(f0, f0)) * inertiaScale + invMass0;
            const float respF1 = (dot(f1, f1)) * inertiaScale + invMass0;
            b.raXn_resp = F4{ a.x, a.y, a.z, resp0 };
            b.normal_errorW = F4{ normal.x, normal.y, normal.z, pen };
            b.fricTan0_invMass0 = F4{ t0.x, t0.y, t0.z, invMass0 };
            b.raXnF0_resp = F4{ f0.x, f0.y, f0.z, respF0 };
            b.raXnF1_resp = F4{ f1.x, f1.y, f1.z, respF1 };
            b.penBiasClampRigid = B.angVel_penBiasClamp.w;
        }
        return b;
    }

    int gid(uint64_t rigidId) const
    {
        return globalRigidBodyId(solverBodyIndices.data(), rigidId, numSolverBodies, maxLinks);
    }

    uint32_t refCountSlots() const { return numSolverBodies + nbArticulations * maxLinks; }

    // ps_queryPCContactReferenceCountLaunch, transcribed.
    std::vector<uint32_t> refCounts(const std::vector<BlockLane>& lanes) const
    {
        std::vector<uint32_t> rc(refCountSlots(), 0u);
        for (size_t w = 0; w < contacts.size(); ++w)
        {
            const int g = gid(contacts[w].pair.ref0);
            if (rigidIsRefCounted(g, lanes[w].fricTan0_invMass0.w))
                rc[g] += 1u;
        }
        return rc;
    }
};

// Builds the scene described in the header.  `seed` varies the random data.
// rotatedFrames gives every body and link a non-identity rotation from a SEPARATE stream,
// so the particle tests' (rotation-free) data is unchanged by the option.
inline void buildWorld(World& W, PxgSlangComputeDevice* dev, uint32_t seed, bool rotatedFrames = false)
{
    W.dev = dev;
    Rng rng{ seed | 1u };
    Rng qrng{ (seed * 2654435761u) | 1u };
    auto randomQuat = [&]() {
        V3 a = v3(qrng.uni(-1.f, 1.f), qrng.uni(-1.f, 1.f), qrng.uni(-1.f, 1.f));
        normalize(a);
        const float h = qrng.uni(0.2f, 1.4f);
        const float sn = std::sin(h), cs = std::cos(h);
        return F4{ a.x * sn, a.y * sn, a.z * sn, cs };
    };

    // ---- particle systems 0 and 2 ----
    const uint32_t P = 10u;
    for (uint32_t s = 0; s < 3; s += 2)
    {
        HostSystem& hs = W.sys[s];
        const float scale = (s == 0) ? 1.0f : 0.5f;
        for (uint32_t i = 0; i < P; ++i)
        {
            const float invMass = (i == 3u) ? 0.0f : scale * (1.0f + 0.25f * float(i % 3));
            hs.pos.push_back(F4{ rng.uni(-1.f, 1.f), rng.uni(-0.2f, 0.6f), rng.uni(-1.f, 1.f), invMass });
            hs.deltaP.push_back(F4{ rng.uni(-0.02f, 0.02f), rng.uni(-0.02f, 0.02f), rng.uni(-0.02f, 0.02f), invMass });
            hs.vel.push_back(F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), invMass });
            hs.accum.push_back(F4{ rng.uni(-0.01f, 0.01f), rng.uni(-0.01f, 0.01f), rng.uni(-0.01f, 0.01f), float(i % 4) });
            // group i % 2, plus flag bits above the 20-bit group mask.
            hs.phases.push_back((i % 2u) | 0x00500000u);
        }
        hs.phaseToMat = (s == 0) ? std::vector<uint16_t>{ 0, 2 } : std::vector<uint16_t>{ 1, 2 };
        // friction, damping, adhesion, gravityScale, adhesionRadiusScale, pad
        hs.materials = {
            0.4f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,    // mi 0: friction, no adhesion
            0.05f, 0.0f, 0.3f, 1.0f, 1.5f, 0.0f,   // mi 1: adhesion
            0.8f, 0.0f, 0.5f, 1.0f, 2.0f, 0.0f,    // mi 2: strong friction + adhesion
        };
        hs.restOffset = (s == 0) ? 0.25f : 0.15f;
        hs.penBiasClamp = (s == 0) ? -2.0f : -0.05f;
    }

    // ---- rigid bodies ----
    W.solverBodyIndices.assign(16, 0u);
    W.solverBodyIndices[4] = 1; W.solverBodyIndices[6] = 2; W.solverBodyIndices[8] = 3;
    W.solverBodyIndices[W.artiNode] = W.artiSolverIndex;
    W.bodies.resize(W.numSolverBodies);
    for (uint32_t b = 0; b < W.numSolverBodies; ++b)
    {
        World::Body& B = W.bodies[b];
        const float invMass = (b == 0 || b == 2) ? 0.0f : (b == 1 ? 0.5f : 2.0f);
        B.angVel_penBiasClamp = F4{ 0.f, 0.f, 0.f, b == 0 ? -1e32f : -rng.uni(0.5f, 3.0f) };
        B.linVel_invMass = F4{ 0.f, 0.f, 0.f, invMass };
        for (int k = 0; k < 9; ++k)
            B.sqrtInvInertia[k] = (b == 0) ? 0.0f : ((k % 4 == 0) ? rng.uni(0.5f, 1.5f) : rng.uni(-0.2f, 0.2f));
        B.p = F4{ rng.uni(-1.f, 1.f), rng.uni(-0.5f, 0.f), rng.uni(-1.f, 1.f), 0.f };
        B.q = rotatedFrames ? randomQuat() : F4{ 0.f, 0.f, 0.f, 1.f };
    }
    W.links.resize(W.maxLinks);
    for (uint32_t l = 0; l < W.maxLinks; ++l)
    {
        World::Link& L = W.links[l];
        L.penBiasClamp = -rng.uni(0.5f, 4.0f);
        L.q = rotatedFrames ? randomQuat() : F4{ 0.f, 0.f, 0.f, 1.f };
        L.p = v3(rng.uni(-1.f, 1.f), rng.uni(-0.5f, 0.f), rng.uni(-1.f, 1.f));
        for (int k = 0; k < 36; ++k)
            L.response[k] = ((k / 6) == (k % 6)) ? rng.uni(0.5f, 2.0f) : rng.uni(-0.3f, 0.3f);
    }

    // ---- velocities ----
    W.outVel.resize(2 * W.numSolverBodies);
    for (F4& v : W.outVel) v = F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f) };
    W.velPool.resize(W.accOffset + 3 * W.numSolverBodies);
    for (F4& v : W.velPool) v = F4{ rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f) };
    W.artiVel.resize(4 * W.maxLinks * W.nbArticulations);
    for (F4& v : W.artiVel) v = F4{ rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f) };

    // ---- contacts ----
    const uint64_t rigids[6] = { staticNode(), makeRigidNode(4), makeRigidNode(6), makeRigidNode(8),
                                 makeArticulationNode(W.artiNode, 2), makeArticulationNode(W.artiNode, 1) };
    const uint32_t NC = 45u;
    for (uint32_t c = 0; c < NC; ++c)
    {
        HostContact hc;
        const uint32_t s = (c % 3u == 2u) ? 2u : 0u;
        const uint32_t pi = rng.next() % P;
        hc.pair.ref0 = rigids[rng.next() % 6u];
        hc.pair.ref1 = particleRef(s, pi);
        V3 n = v3(rng.uni(-0.4f, 0.4f), 1.0f, rng.uni(-0.4f, 0.4f));
        if (c % 7u == 0u)
            n = v3(0.0f, 1.0f, 0.0f);      // the |dir.y| > 0.9999 basis arm
        normalize(n);
        hc.normalPen = F4{ n.x, n.y, n.z, rng.uni(-0.08f, 0.06f) };
        W.contacts.push_back(hc);
    }
    std::stable_sort(W.contacts.begin(), W.contacts.end(),
                     [](const HostContact& a, const HostContact& b) { return a.pair.ref1 < b.pair.ref1; });

    // ---- device ----
    std::vector<uint8_t> pool(3 * SZ_PARTICLE_SYS, 0);
    for (uint32_t s = 0; s < 3; s += 2)
    {
        HostSystem& hs = W.sys[s];
        std::vector<uint32_t> p2m((hs.phaseToMat.size() + 1) / 2, 0u);
        std::memcpy(p2m.data(), hs.phaseToMat.data(), hs.phaseToMat.size() * 2);
        const size_t base = s * SZ_PARTICLE_SYS;
        putU64(pool, base + 24, W.addr(W.make(hs.pos)));        // mSortedPositions_InvMass
        putU64(pool, base + 32, W.addr(W.make(hs.vel)));        // mSortedVelocities
        putU64(pool, base + 40, W.addr(W.make(hs.phases)));     // mSortedPhaseArray
        putU64(pool, base + 80, W.addr(W.make(hs.materials)));  // mParticleMaterials
        putU32(pool, base + 116, uint32_t(hs.pos.size()));      // mCommonData.mNumParticles
        W.bAccum[s] = W.make(hs.accum);
        putU64(pool, base + 152, W.addr(W.bAccum[s]));          // mAccumDeltaP
        putU64(pool, base + 160, W.addr(W.make(hs.deltaP)));    // mSortedDeltaP
        putF32(pool, base + 268, hs.restOffset);                // mData.mRestOffset
        putF32(pool, base + 320, hs.penBiasClamp);              // mData.mPenBiasClamp
        putU32(pool, base + 592, MAT_STRIDE);                   // mParticleMaterialStride
        putU64(pool, base + 600, W.addr(W.make(p2m)));          // mPhaseGroupToMaterialHandle
    }
    W.psysPool = W.addr(W.make(pool));

    std::vector<uint8_t> bodyData(W.numSolverBodies * SZ_SOLVER_BODY, 0), txi(W.numSolverBodies * SZ_TXI, 0),
        frames(W.numSolverBodies * SZ_ALIGNED_XF, 0);
    for (uint32_t b = 0; b < W.numSolverBodies; ++b)
    {
        putF4(bodyData, b * SZ_SOLVER_BODY + 0, W.bodies[b].angVel_penBiasClamp);
        putF4(bodyData, b * SZ_SOLVER_BODY + 16, W.bodies[b].linVel_invMass);
        std::memcpy(txi.data() + b * SZ_TXI + 28, W.bodies[b].sqrtInvInertia, 36);
        putF4(frames, b * SZ_ALIGNED_XF + 0, W.bodies[b].q);
        putF4(frames, b * SZ_ALIGNED_XF + 16, W.bodies[b].p);
    }
    std::vector<uint8_t> prep(SZ_PREP_DESC, 0);
    putU64(prep, 64, W.addr(W.make(bodyData)));   // solverBodyDataPool
    putU64(prep, 72, W.addr(W.make(txi)));        // solverBodyTxIDataPool
    putU64(prep, 168, W.addr(W.make(frames)));    // body2WorldPool
    W.prepDesc = W.addr(W.make(prep));

    std::vector<uint8_t> preprep(SZ_PREPREP_DESC, 0);
    putU64(preprep, 272, W.addr(W.make(W.solverBodyIndices)));
    W.prePrepDesc = W.addr(W.make(preprep));

    // articulation 1 (record 0 is a decoy)
    std::vector<uint8_t> linkRecs(W.maxLinks * SZ_ARTI_LINK, 0), linkXf(W.maxLinks * SZ_PXTRANSFORM, 0),
        resp(W.maxLinks * SZ_SPATIAL_MATRIX, 0);
    for (uint32_t l = 0; l < W.maxLinks; ++l)
    {
        putF4(linkRecs, l * SZ_ARTI_LINK, F4{ 0.f, 0.f, 0.f, W.links[l].penBiasClamp });
        const float xf[7] = { W.links[l].q.x, W.links[l].q.y, W.links[l].q.z, W.links[l].q.w,
                              W.links[l].p.x, W.links[l].p.y, W.links[l].p.z };
        std::memcpy(linkXf.data() + l * SZ_PXTRANSFORM, xf, 28);
        std::memcpy(resp.data() + l * SZ_SPATIAL_MATRIX, W.links[l].response, 144);
    }
    std::vector<uint8_t> artis(2 * SZ_ARTI, 0);
    putU64(artis, W.artiRemapId * SZ_ARTI + 32, W.addr(W.make(linkRecs)));    // links
    putU64(artis, W.artiRemapId * SZ_ARTI + 168, W.addr(W.make(resp)));       // spatialResponseMatrixW
    putU64(artis, W.artiRemapId * SZ_ARTI + 176, W.addr(W.make(linkXf)));     // linkBody2Worlds
    std::vector<uint8_t> bodySims(16 * SZ_BODYSIM, 0);
    putU32(bodySims, W.artiNode * SZ_BODYSIM + 176, W.artiRemapId);            // articulationRemapId
    std::vector<uint8_t> shared(SZ_SHARED_PGS, 0);
    putU64(shared, 32, W.addr(W.make(bodySims)));   // mBodySimBufferDeviceData
    putU64(shared, 40, W.addr(W.make(artis)));      // articulations
    W.sharedDesc = W.addr(W.make(shared));

    std::vector<uint8_t> artiCore(SZ_ARTI_CORE, 0);
    putU32(artiCore, 44, W.nbArticulations);        // nbArticulations
    putU32(artiCore, 248, W.maxLinks);              // mMaxLinksPerArticulation
    W.artiCoreDesc = W.addr(W.make(artiCore));

    W.bVelPool = W.make(W.velPool);
    W.bOutVel = W.make(W.outVel);
    W.bArtiVel = W.make(W.artiVel);
    W.velPoolAddr = W.addr(W.bVelPool);
    std::vector<uint8_t> core(SZ_CORE_DESC, 0);
    putU64(core, 0, W.addr(W.bOutVel));             // outSolverVelocity
    putU64(core, 40, W.addr(W.bArtiVel));           // outArtiVelocity
    putU32(core, 268, W.numSolverBodies);           // numSolverBodies
    putU32(core, 276, W.accOffset);                 // accumulatedBodyDeltaVOffset
    W.coreDesc = W.addr(W.make(core));
}

// One block lane read back from the device.
inline BlockLane readLane(const std::vector<uint8_t>& blocks, uint32_t w)
{
    const size_t base = size_t(w / 32u) * RPC_STRIDE;
    const uint32_t l = w & 31u;
    BlockLane b;
    std::memcpy(&b.raXn_resp, blocks.data() + base + RPC_raXn_resp + l * 16, 16);
    std::memcpy(&b.raXnF0_resp, blocks.data() + base + RPC_raXnF0_resp + l * 16, 16);
    std::memcpy(&b.raXnF1_resp, blocks.data() + base + RPC_raXnF1_resp + l * 16, 16);
    std::memcpy(&b.normal_errorW, blocks.data() + base + RPC_normal_errorW + l * 16, 16);
    std::memcpy(&b.fricTan0_invMass0, blocks.data() + base + RPC_fricTan0_invMass0 + l * 16, 16);
    std::memcpy(&b.penBiasClampRigid, blocks.data() + base + RPC_penBiasClampRigid + l * 4, 4);
    return b;
}

inline void writeLane(std::vector<uint8_t>& blocks, uint32_t w, const BlockLane& b)
{
    const size_t base = size_t(w / 32u) * RPC_STRIDE;
    const uint32_t l = w & 31u;
    std::memcpy(blocks.data() + base + RPC_raXn_resp + l * 16, &b.raXn_resp, 16);
    std::memcpy(blocks.data() + base + RPC_raXnF0_resp + l * 16, &b.raXnF0_resp, 16);
    std::memcpy(blocks.data() + base + RPC_raXnF1_resp + l * 16, &b.raXnF1_resp, 16);
    std::memcpy(blocks.data() + base + RPC_normal_errorW + l * 16, &b.normal_errorW, 16);
    std::memcpy(blocks.data() + base + RPC_fricTan0_invMass0 + l * 16, &b.fricTan0_invMass0, 16);
    std::memcpy(blocks.data() + base + RPC_penBiasClampRigid + l * 4, &b.penBiasClampRigid, 4);
}

// The push packs (5.12 CUDA signatures; psContactSolveBda / psContactSolveRigidBda).
struct PrepPush
{
    uint64_t particleSystems, sortedContactInfos, sortedContactNormalPens, numContacts, contactBlocks,
        preDesc, prepareDesc, appliedForces;
    float dt; uint32_t isTGSWord;
    uint64_t deltaVel, sharedDesc;
};
static_assert(sizeof(PrepPush) == 88, "prepare push 88 bytes");

struct QueryPush
{
    uint64_t particleSystems, sortedContactInfos, contactBlocks, numContacts, prePrepDesc, solverCoreDesc,
        solverBodyVelPool, rigidRefCount;
    float dt; uint32_t isTGSWord;
    uint64_t artiCoreDesc;
};
static_assert(sizeof(QueryPush) == 80, "query push 80 bytes");

struct SolvePush
{
    uint64_t particleSystems, sortedContactInfos, contactBlocks, numContacts, prePrepDesc, solverCoreDesc,
        solverBodyVelPool, deltaOut, appliedForces, rigidRefCount;
    float dt; uint32_t flagsWord;   // byte0 isVelocityIteration, byte1 isTGS
    uint64_t artiCoreDesc;
};
static_assert(sizeof(SolvePush) == 96, "solve push 96 bytes");

struct AccumPush
{
    uint64_t particleSystems, sortedContactInfos, totalNumPairs, rangeStart, rangeEnd, deltaPos;
};
static_assert(sizeof(AccumPush) == 48, "accumulate push 48 bytes");

} // namespace pxgpsworld

#endif // PXG_PARTICLE_SOLVE_WORLD_H
