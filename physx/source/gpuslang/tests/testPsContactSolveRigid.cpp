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

// P9.44 -- psContactSolveRigidBda.slang: ps_solvePCOutputRigidDeltaVLaunch (PGS) and
// ps_solvePCOutputRigidDeltaVTGSLaunch, the rigid side of the particle/primitive
// contact solve.  One fixture, 70 contacts over three constraint blocks, every lane
// kind the .cu distinguishes: a STATIC rigid (skipped, its output slots keep their
// sentinel), a KINEMATIC rigid (invMass0 == 0: explicit zeros), moving DYNAMIC rigids
// (PGS: initial + accumulated delta; TGS: the packed 3-float4 pool) and ARTICULATION
// LINKS (outArtiVelocity at solverBodyIndex * maxLinks + linkID, plus the two delta
// segments under TGS).  Two particle systems, two materials (one with adhesion),
// non-zero applied forces from "earlier iterations".  The per-contact rigid delta
// (linear with the weight in .w, angular) and the applied force are compared word
// for word against a host transcription of the two kernels.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F2 { float x, y; };
struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 scale3(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 xyz(F4 v) { return V3{ v.x, v.y, v.z }; }
static float pmin(float a, float b) { return (a < b) ? a : b; }
static float pmax(float a, float b) { return (a > b) ? a : b; }

static const uint32_t CB_STRIDE = 3072u;
static const uint32_t CB_RAXN_VM = 0u, CB_NORMAL_ERR = 512u, CB_FRIC0_IM0 = 1024u, CB_RAXNF0_VM = 1536u,
                      CB_RAXNF1_VM = 2048u, CB_RIGIDID = 2560u, CB_PARTICLEID = 2816u;
static const uint32_t STATIC_NODE = 0xffffffffu;

static const uint32_t SZ_PARTICLE_SYS = 656u;
static const uint32_t SZ_PREPREP_DESC = 760u, OFF_SOLVER_BODY_INDICES = 272u;
static const uint32_t SZ_CORE_DESC = 320u, OFF_OUT_SOLVER_VEL = 0u, OFF_OUT_ARTI_VEL = 40u, OFF_NUM_SOLVER_BODIES = 268u, OFF_ACC = 276u;
static const uint32_t SZ_SHARED_PGS = 224u, OFF_VEL_POOL = 160u;
static const uint32_t SZ_ARTI_CORE_DESC = 352u, OFF_NB_ARTI = 44u, OFF_MAX_LINKS = 248u;

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h) { return dev->getBufferDeviceAddress(h); }
static void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
static void putF32(std::vector<uint8_t>& b, size_t off, float v) { std::memcpy(b.data() + off, &v, 4); }
static void putF4(std::vector<uint8_t>& b, size_t off, F4 v) { std::memcpy(b.data() + off, &v, 16); }

struct Rng
{
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    uint32_t below(uint32_t n) { return next() % n; }
    float uni(float lo, float hi) { return lo + (hi - lo) * float(next() & 0xffffffu) / 16777216.0f; }
    V3 vec(float lo, float hi) { return V3{ uni(lo, hi), uni(lo, hi), uni(lo, hi) }; }
};

// ---------------------------------------------------------------------------
// Host transcriptions of the two solve cores (particlesystem.cu:1421 / :1473).
// ---------------------------------------------------------------------------
struct SolveOut { float deltaF, deltaFr0, deltaFr1; F2 appliedForce; };

static SolveOut cpuSolvePCOutputDeltaV(V3 normal, float error, V3 delta, V3 fric0, V3 fric1,
                                       float tanVel0, float tanVel1, float normalVel,
                                       float velMultiplier, float vmF0, float vmF1,
                                       float frictionCoefficient, float restOffset, float adhesion, float adhesionRadiusScale,
                                       float dt, F2 appliedForce)
{
    const float separation = (error - dot3(normal, delta));
    float clamp = -appliedForce.x;
    if (adhesionRadiusScale > 0.0f && adhesion > 0.0f)
    {
        const float adhesionRadius = restOffset * adhesionRadiusScale - restOffset;
        if (separation < adhesionRadiusScale)
        {
            const float q = 1.0f - separation / adhesionRadius;
            clamp = pmin(clamp, -pmax(0.0f, adhesion * (q * q * q * q) * pmax(0.0f, separation) * velMultiplier));
        }
    }
    SolveOut r;
    r.deltaF = pmax(clamp, (-separation - normalVel * dt) * velMultiplier);
    appliedForce.x = pmax(0.0f, r.deltaF + appliedForce.x);
    const float friction = appliedForce.x * frictionCoefficient;
    const float biasedF0 = -(dot3(fric0, delta));
    const float biasedF1 = -(dot3(fric1, delta));
    const float requiredF0Force = (biasedF0 + tanVel0 * dt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * dt) * vmF1;
    const float requiredForce = std::sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recip = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;
    const float ratio0 = requiredF0Force * recip;
    const float ratio1 = requiredF1Force * recip;
    const float deltaFr = pmin(requiredForce + appliedForce.y, friction) - appliedForce.y;
    appliedForce.y = appliedForce.y + deltaFr;
    r.deltaFr0 = deltaFr * ratio0;
    r.deltaFr1 = deltaFr * ratio1;
    r.appliedForce = appliedForce;
    return r;
}

static SolveOut cpuSolvePCOutputDeltaVTGS(V3 normal, float error, V3 delta, V3 fric0, V3 fric1,
                                          float tanDelta0, float tanDelta1, float tanVel0, float tanVel1,
                                          float normalDelta, float normalVel, float velMultiplier, float vmF0, float vmF1,
                                          float frictionCoefficient, float restOffset, float adhesion, float adhesionRadiusScale,
                                          bool isVelocityIteration, float dt, F2 appliedForce)
{
    const float projectionDt = isVelocityIteration ? 0.0f : dt;
    const float separation = (error - dot3(normal, delta) + normalDelta) + normalVel * projectionDt;
    float clamp = -appliedForce.x;
    if (adhesionRadiusScale > 0.0f && adhesion > 0.0f)
    {
        const float adhesionRadius = restOffset * adhesionRadiusScale - restOffset;
        if (separation < adhesionRadiusScale)
        {
            const float q = 1.0f - separation / adhesionRadius;
            clamp = pmin(clamp, -pmax(0.0f, adhesion * (q * q * q * q) * pmax(0.0f, separation) * velMultiplier));
        }
    }
    SolveOut r;
    r.deltaF = pmax(clamp, -separation * velMultiplier);
    appliedForce.x = pmax(0.0f, appliedForce.x + r.deltaF);
    const float maxFriction = std::fabs(appliedForce.x) * frictionCoefficient;
    const float biasedF0 = -(dot3(fric0, delta) - tanDelta0);
    const float biasedF1 = -(dot3(fric1, delta) - tanDelta1);
    const float requiredF0Force = (biasedF0 + tanVel0 * projectionDt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * projectionDt) * vmF1;
    const float requiredForce = std::sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recip = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;
    const float ratio0 = requiredF0Force * recip;
    const float ratio1 = requiredF1Force * recip;
    const float deltaFr = pmin(requiredForce + appliedForce.y, maxFriction) - appliedForce.y;
    appliedForce.y = appliedForce.y + deltaFr;
    r.deltaFr0 = deltaFr * ratio0;
    r.deltaFr1 = deltaFr * ratio1;
    r.appliedForce = appliedForce;
    return r;
}

// .cu:2103-2118 / 2257-2271
static void cpuReaction(V3 normal, V3 fric0, V3 fric1, V3 raXn, V3 raXnF0, V3 raXnF1, float invMass0, float particleInvMass,
                        const SolveOut& s, float relaxationCoefficient, float dt, F4& lin, F4& ang)
{
    const float relaxation = relaxationCoefficient / dt;
    float count = 0.0f;
    V3 deltaAngVel0{ 0.0f, 0.0f, 0.0f }, deltaLinVel0{ 0.0f, 0.0f, 0.0f };
    if ((s.deltaF != 0.0f || s.deltaFr0 != 0.0f || s.deltaFr1 != 0.0f) && invMass0 > 0.0f)
    {
        deltaAngVel0 = scale3(sub3(sub3(scale3(raXn, s.deltaF), scale3(raXnF0, s.deltaFr0)), scale3(raXnF1, s.deltaFr1)), relaxation);
        deltaLinVel0 = scale3(scale3(sub3(sub3(scale3(normal, s.deltaF), scale3(fric0, s.deltaFr0)), scale3(fric1, s.deltaFr1)), invMass0), relaxation);
        count = pmax(invMass0 / pmax(invMass0, particleInvMass), 0.01f);
    }
    lin = F4{ deltaLinVel0.x, deltaLinVel0.y, deltaLinVel0.z, count };
    ang = F4{ deltaAngVel0.x, deltaAngVel0.y, deltaAngVel0.z, 0.0f };
}

// ---------------------------------------------------------------------------
// The scene: 70 contacts, two particle systems, rigid + articulation bodies.
// ---------------------------------------------------------------------------
static const uint32_t N_CONTACTS = 70u;
static const uint32_t NUM_SYSTEMS = 2u, PARTICLES_PER_SYSTEM = 13u;
static const uint32_t N_BODIES = 20u, ACC = 5u;
static const uint32_t NB_ARTI = 3u, MAX_LINKS = 4u, ARTI_OFFSET = NB_ARTI * MAX_LINKS;
static const uint32_t ARTI_NODE_BASE = 100u, NUM_NODES = ARTI_NODE_BASE + NB_ARTI;
static const float SENTINEL_F = 777.0f;

// Lane 18 is the adhesion probe: 18 % 7 == 4 makes it a dynamic rigid, and node 18
// resolves to solver body 19, which no other lane uses, so putting that body at rest
// costs the other lanes nothing.  Particle 1 of system 0 has phase group 1, the group
// the phase->material table points at the material carrying adhesion.
static const uint32_t ADHESION_LANE = 18u;
static const uint32_t ADHESION_PARTICLE = 1u;

enum LaneKind { LK_STATIC, LK_KINEMATIC, LK_ARTI, LK_RIGID };
static LaneKind laneKind(uint32_t k)
{
    switch (k % 7u) { case 0: return LK_STATIC; case 1: return LK_KINEMATIC; case 2: case 5: return LK_ARTI; default: return LK_RIGID; }
}

struct Scene
{
    // constraint blocks (3 x 3072 bytes)
    std::vector<uint8_t> cb;
    std::vector<uint64_t> rigidIds;   // per contact
    std::vector<uint32_t> particleSys, particleIdx;
    // particle systems
    std::vector<F4> deltaP[NUM_SYSTEMS];
    std::vector<uint32_t> phases[NUM_SYSTEMS];
    std::vector<uint32_t> phaseToMatWords;   // PxU16 x 4 -> 2 words (shared by both systems)
    std::vector<float> materials;            // 2 materials x 5 floats
    float restOffset[NUM_SYSTEMS];
    // bodies
    std::vector<uint32_t> solverBodyIndices; // NUM_NODES
    std::vector<F4> initialVel;              // 2 * N_BODIES (PGS)
    std::vector<F4> pool;                    // ACC + 3 * N_BODIES
    std::vector<F4> artiVel;                 // 4 * ARTI_OFFSET
    // io
    std::vector<F2> appliedForces;           // N_CONTACTS
};

static Scene buildScene(Rng& rng)
{
    Scene S;
    S.cb.assign(3u * CB_STRIDE, 0u);
    S.rigidIds.resize(N_CONTACTS);
    S.particleSys.resize(N_CONTACTS);
    S.particleIdx.resize(N_CONTACTS);
    for (uint32_t k = 0; k < N_CONTACTS; ++k)
    {
        const uint32_t block = k / 32u, lane = k & 31u;
        const size_t base = size_t(block) * CB_STRIDE;
        V3 normal = rng.vec(-1.0f, 1.0f);
        {
            const float m = std::sqrt(dot3(normal, normal));
            normal = scale3(normal, 1.0f / m);
        }
        const V3 fric0 = rng.vec(-1.0f, 1.0f);
        const float error = rng.uni(-0.2f, 0.05f);
        const V3 raXn = rng.vec(-0.5f, 0.5f), raXnF0 = rng.vec(-0.5f, 0.5f), raXnF1 = rng.vec(-0.5f, 0.5f);
        const float vm = rng.uni(0.2f, 2.0f), vmF0 = rng.uni(0.2f, 2.0f), vmF1 = rng.uni(0.2f, 2.0f);
        const LaneKind kind = laneKind(k);
        const float invMass0 = (kind == LK_KINEMATIC) ? 0.0f : rng.uni(0.1f, 3.0f);
        putF4(S.cb, base + CB_RAXN_VM + lane * 16, F4{ raXn.x, raXn.y, raXn.z, vm });
        putF4(S.cb, base + CB_NORMAL_ERR + lane * 16, F4{ normal.x, normal.y, normal.z, error });
        putF4(S.cb, base + CB_FRIC0_IM0 + lane * 16, F4{ fric0.x, fric0.y, fric0.z, invMass0 });
        putF4(S.cb, base + CB_RAXNF0_VM + lane * 16, F4{ raXnF0.x, raXnF0.y, raXnF0.z, vmF0 });
        putF4(S.cb, base + CB_RAXNF1_VM + lane * 16, F4{ raXnF1.x, raXnF1.y, raXnF1.z, vmF1 });
        uint64_t id;
        switch (kind)
        {
        case LK_STATIC: id = uint64_t(STATIC_NODE); break;
        case LK_ARTI: id = uint64_t(ARTI_NODE_BASE + (k % NB_ARTI)) | (uint64_t(((k % MAX_LINKS) << 1) | 1u) << 32); break;
        default: id = uint64_t(k % ARTI_NODE_BASE); break;
        }
        S.rigidIds[k] = id;
        putU64(S.cb, base + CB_RIGIDID + lane * 8, id);
        S.particleSys[k] = k % NUM_SYSTEMS;
        S.particleIdx[k] = k % PARTICLES_PER_SYSTEM;
        putU64(S.cb, base + CB_PARTICLEID + lane * 8, uint64_t(S.particleIdx[k]) | (uint64_t(S.particleSys[k]) << 32));
    }
    for (uint32_t s = 0; s < NUM_SYSTEMS; ++s)
    {
        S.deltaP[s].resize(PARTICLES_PER_SYSTEM);
        S.phases[s].resize(PARTICLES_PER_SYSTEM);
        for (uint32_t p = 0; p < PARTICLES_PER_SYSTEM; ++p)
        {
            const V3 d = rng.vec(-0.05f, 0.05f);
            S.deltaP[s][p] = F4{ d.x, d.y, d.z, (p % 4u == 3u) ? 0.0f : rng.uni(0.2f, 5.0f) };
            S.phases[s][p] = (p % 3u) | (0x00100000u * (p & 1u));   // group in the low 20 bits, a flag bit above
        }
        S.restOffset[s] = 0.1f + 0.05f * float(s);
    }
    // groups 0,1,2 -> materials 0,1,0 as a PxU16 array
    S.phaseToMatWords = { 0u | (1u << 16), 0u };
    S.materials = { 0.5f, 0.0f, 0.0f, 1.0f, 0.0f,     // friction, damping, adhesion, gravityScale, adhesionRadiusScale
                    0.2f, 0.0f, 0.3f, 1.0f, 1.5f };
    S.solverBodyIndices.assign(NUM_NODES, 0u);
    for (uint32_t r = 0; r < ARTI_NODE_BASE; ++r) S.solverBodyIndices[r] = 1u + (r % (N_BODIES - 1u));
    for (uint32_t g = 0; g < NB_ARTI; ++g) S.solverBodyIndices[ARTI_NODE_BASE + g] = g;
    S.initialVel.resize(2u * N_BODIES);
    for (F4& v : S.initialVel) { const V3 a = rng.vec(-2.0f, 2.0f); v = F4{ a.x, a.y, a.z, rng.uni(-1.0f, 1.0f) }; }
    S.pool.resize(ACC + 3u * N_BODIES);
    for (F4& v : S.pool) { const V3 a = rng.vec(-1.0f, 1.0f); v = F4{ a.x, a.y, a.z, rng.uni(-1.0f, 1.0f) }; }
    S.artiVel.resize(4u * ARTI_OFFSET);
    for (F4& v : S.artiVel) { const V3 a = rng.vec(-1.5f, 1.5f); v = F4{ a.x, a.y, a.z, rng.uni(-1.0f, 1.0f) }; }
    S.appliedForces.resize(N_CONTACTS);
    for (F2& f : S.appliedForces) f = F2{ rng.uni(0.0f, 0.5f), rng.uni(0.0f, 0.1f) };

    // THE ADHESION PROBE.  The adhesion clamp multiplies by PxMax(0, separation), so
    // it is inert for a penetrating contact -- which is every contact above, since
    // their errors are mostly negative and the rigid bodies move.  The first sweep of
    // this module left the PGS adhesion branch standing for exactly that reason.  Lane
    // ADHESION_LANE is therefore pinned: a separation of +0.02 (a gap well inside the
    // material's adhesion radius of restOffset*scale - restOffset = 0.05), a particle
    // whose phase group selects the material that HAS adhesion, zero running deltaP,
    // no applied force yet, and a rigid body at rest -- so normalVel and normalDelta
    // vanish and the clamp is the only thing that can move deltaF off zero.
    {
        const uint32_t k = ADHESION_LANE;
        const uint32_t block = k / 32u, lane = k & 31u;
        const size_t base = size_t(block) * CB_STRIDE;
        F4 ne; std::memcpy(&ne, S.cb.data() + base + CB_NORMAL_ERR + lane * 16, 16);
        ne.w = 0.02f;                                    // separation (delta is zeroed below)
        putF4(S.cb, base + CB_NORMAL_ERR + lane * 16, ne);
        S.particleSys[k] = 0u;
        S.particleIdx[k] = ADHESION_PARTICLE;
        putU64(S.cb, base + CB_PARTICLEID + lane * 8, uint64_t(ADHESION_PARTICLE));
        S.deltaP[0][ADHESION_PARTICLE] = F4{ 0.0f, 0.0f, 0.0f, 1.5f };
        S.appliedForces[k] = F2{ 0.0f, 0.0f };
        // the body this lane's node resolves to is put at rest, in both pools
        const uint32_t sbi = S.solverBodyIndices[uint32_t(S.rigidIds[k] & 0xffffffffull)];
        S.initialVel[sbi] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
        S.initialVel[sbi + N_BODIES] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
        S.pool[ACC + sbi] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
        S.pool[ACC + sbi + N_BODIES] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
        S.pool[ACC + sbi + 2u * N_BODIES] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    return S;
}

// The kernel on the host: one lane.
static void cpuLane(const Scene& S, uint32_t k, bool isTGS, bool isVelIter, float dt, float relax,
                    F4& lin, F4& ang, F2& applied, bool& written)
{
    written = false;
    const uint32_t block = k / 32u, lane = k & 31u;
    const size_t base = size_t(block) * CB_STRIDE;
    auto getF4 = [&](size_t off) { F4 v; std::memcpy(&v, S.cb.data() + off, 16); return v; };
    const uint64_t id = S.rigidIds[k];
    if (uint32_t(id & 0xffffffffull) == STATIC_NODE) return;   // continue without writing
    const F4 fricTan0_invMass0 = getF4(base + CB_FRIC0_IM0 + lane * 16);
    written = true;
    if (fricTan0_invMass0.w == 0.0f)
    {
        lin = F4{ 0.0f, 0.0f, 0.0f, 0.0f }; ang = lin; applied = F2{ 0.0f, 0.0f };
        return;
    }
    const uint32_t sys = S.particleSys[k], pid = S.particleIdx[k];
    const uint32_t phase = S.phases[sys][pid];
    const uint32_t group = phase & 0x000fffffu;
    const uint32_t mi = (S.phaseToMatWords[group >> 1] >> ((group & 1u) * 16u)) & 0xffffu;
    const float frictionCoefficient = S.materials[mi * 5 + 0];
    const float adhesion = S.materials[mi * 5 + 2];
    const float adhesionRadiusScale = S.materials[mi * 5 + 4];
    const float restOffset = S.restOffset[sys];
    const F4 delta4 = S.deltaP[sys][pid];
    const V3 delta = xyz(delta4);
    F2 appliedForce = S.appliedForces[k];

    const uint32_t sbi = S.solverBodyIndices[uint32_t(id & 0xffffffffull)];
    const bool isArti = (uint32_t(id >> 32) & 1u) != 0u;
    const uint32_t linkID = uint32_t(id >> 32) >> 1;

    const F4 raXn_vm = getF4(base + CB_RAXN_VM + lane * 16);
    const F4 normal_err = getF4(base + CB_NORMAL_ERR + lane * 16);
    const F4 raXnF0_vm = getF4(base + CB_RAXNF0_VM + lane * 16);
    const F4 raXnF1_vm = getF4(base + CB_RAXNF1_VM + lane * 16);
    const V3 normal = xyz(normal_err), fric0 = xyz(fricTan0_invMass0), fric1 = cross3(normal, fric0);
    const float error = normal_err.w;
    const V3 raXn = xyz(raXn_vm), raXnF0 = xyz(raXnF0_vm), raXnF1 = xyz(raXnF1_vm);
    const float velMultiplier = raXn_vm.w, vmF0 = raXnF0_vm.w, vmF1 = raXnF1_vm.w;
    const float invMass0 = fricTan0_invMass0.w;

    SolveOut s;
    if (!isTGS)
    {
        V3 linVel0, angVel0;
        if (isArti)
        {
            const uint32_t index = sbi * MAX_LINKS;
            linVel0 = xyz(S.artiVel[index + linkID]);
            angVel0 = xyz(S.artiVel[index + linkID + ARTI_OFFSET]);
        }
        else
        {
            const F4 a = S.initialVel[sbi], b = S.pool[ACC + sbi];
            const F4 c = S.initialVel[sbi + N_BODIES], d = S.pool[ACC + sbi + N_BODIES];
            linVel0 = V3{ a.x + b.x, a.y + b.y, a.z + b.z };
            angVel0 = V3{ c.x + d.x, c.y + d.y, c.z + d.z };
        }
        const float normalVel = dot3(linVel0, normal) + dot3(angVel0, raXn);
        const float tanVel0 = dot3(linVel0, fric0) + dot3(angVel0, raXnF0);
        const float tanVel1 = dot3(linVel0, fric1) + dot3(angVel0, raXnF1);
        s = cpuSolvePCOutputDeltaV(normal, error, delta, fric0, fric1, tanVel0, tanVel1, normalVel,
                                   velMultiplier, vmF0, vmF1, frictionCoefficient, restOffset, adhesion, adhesionRadiusScale,
                                   dt, appliedForce);
    }
    else
    {
        V3 linVel, angVel, linDelta, angDelta;
        if (isArti)
        {
            const uint32_t index = sbi * MAX_LINKS;
            linVel = xyz(S.artiVel[index + linkID]);
            angVel = xyz(S.artiVel[index + linkID + ARTI_OFFSET]);
            linDelta = xyz(S.artiVel[index + linkID + 2u * ARTI_OFFSET]);
            angDelta = xyz(S.artiVel[index + linkID + 3u * ARTI_OFFSET]);
        }
        else
        {
            const F4 v0 = S.pool[ACC + sbi], v1 = S.pool[ACC + sbi + N_BODIES], v2 = S.pool[ACC + sbi + 2u * N_BODIES];
            linVel = V3{ v0.x, v0.y, v0.z };
            angVel = V3{ v0.w, v1.x, v1.y };
            linDelta = V3{ v1.z, v1.w, v2.x };
            angDelta = V3{ v2.y, v2.z, v2.w };
        }
        const float normalVel = dot3(linVel, normal) + dot3(angVel, raXn);
        const float normalDelta = dot3(linDelta, normal) + dot3(angDelta, raXn);
        const float tanVel0 = dot3(linVel, fric0) + dot3(angVel, raXnF0);
        const float tanDelta0 = dot3(linDelta, fric0) + dot3(angDelta, raXnF0);
        const float tanVel1 = dot3(linVel, fric1) + dot3(angVel, raXnF1);
        const float tanDelta1 = dot3(linDelta, fric1) + dot3(angDelta, raXnF1);
        s = cpuSolvePCOutputDeltaVTGS(normal, error, delta, fric0, fric1, tanDelta0, tanDelta1, tanVel0, tanVel1,
                                      normalDelta, normalVel, velMultiplier, vmF0, vmF1, frictionCoefficient, restOffset,
                                      adhesion, adhesionRadiusScale, isVelIter, dt, appliedForce);
    }
    cpuReaction(normal, fric0, fric1, raXn, raXnF0, raXnF1, invMass0, delta4.w, s, relax, dt, lin, ang);
    applied = s.appliedForce;
}

struct SolvePush
{
    uint64_t particleSystems, sortedContacts, primitiveConstraints, numContacts, prePrepDesc, solverCoreDesc, sharedDesc, deltaVel, appliedForces;
    float dt, relaxationCoefficient; uint32_t isVelIterWord;
    uint64_t artiCoreDesc;
};
static_assert(sizeof(SolvePush) == 96, "solve push 96 bytes");

PXG_TEST(psContactSolveRigid_lanes)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f, relax = 0.8f;
    struct Variant { const char* entry; bool isTGS; bool velIter; };
    const Variant variants[] = {
        { "ps_solvePCOutputRigidDeltaVLaunch", false, false },
        { "ps_solvePCOutputRigidDeltaVTGSLaunch", true, false },
        { "ps_solvePCOutputRigidDeltaVTGSLaunch", true, true },
    };
    for (const Variant& v : variants)
    {
        Rng rng{ 0x2545f491u ^ (v.isTGS ? 0x1000u : 0u) ^ (v.velIter ? 0x2000u : 0u) };
        const Scene S = buildScene(rng);

        // ---- upload ----
        std::vector<PxgBufferHandle> all;
        auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };
        PxgBufferHandle bCb = mk(S.cb);
        if (!addrOf(dev, bCb)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
        PxgBufferHandle bPhaseToMat = mk(S.phaseToMatWords);
        PxgBufferHandle bMaterials = mk(S.materials);
        std::vector<uint8_t> systems(size_t(NUM_SYSTEMS) * SZ_PARTICLE_SYS, 0u);
        for (uint32_t s = 0; s < NUM_SYSTEMS; ++s)
        {
            PxgBufferHandle bDeltaP = mk(S.deltaP[s]);
            PxgBufferHandle bPhases = mk(S.phases[s]);
            const size_t o = size_t(s) * SZ_PARTICLE_SYS;
            putU64(systems, o + 40, addrOf(dev, bPhases));       // mSortedPhaseArray
            putU64(systems, o + 80, addrOf(dev, bMaterials));    // mParticleMaterials
            putU64(systems, o + 160, addrOf(dev, bDeltaP));      // mSortedDeltaP
            putF32(systems, o + 268, S.restOffset[s]);           // mData_RestOffset
            putU32(systems, o + 632, 20u);                       // mParticleMaterialStride (bytes)
            putU64(systems, o + 640, addrOf(dev, bPhaseToMat));  // mPhaseGroupToMaterialHandle
        }
        PxgBufferHandle bSystems = mk(systems);
        PxgBufferHandle bNum = mk(std::vector<uint32_t>(1, N_CONTACTS));
        PxgBufferHandle bSbi = mk(S.solverBodyIndices);
        PxgBufferHandle bInitialVel = mk(S.initialVel);
        PxgBufferHandle bPool = mk(S.pool);
        PxgBufferHandle bArtiVel = mk(S.artiVel);
        PxgBufferHandle bDeltaVel = mk(std::vector<F4>(2u * N_CONTACTS, F4{ SENTINEL_F, SENTINEL_F, SENTINEL_F, SENTINEL_F }));
        PxgBufferHandle bApplied = mk(S.appliedForces);

        std::vector<uint8_t> prePrep(SZ_PREPREP_DESC, 0u);
        putU64(prePrep, OFF_SOLVER_BODY_INDICES, addrOf(dev, bSbi));
        std::vector<uint8_t> core(SZ_CORE_DESC, 0u);
        putU64(core, OFF_OUT_SOLVER_VEL, addrOf(dev, bInitialVel));
        putU64(core, OFF_OUT_ARTI_VEL, addrOf(dev, bArtiVel));
        putU32(core, OFF_NUM_SOLVER_BODIES, N_BODIES);
        putU32(core, OFF_ACC, ACC);
        std::vector<uint8_t> shared(SZ_SHARED_PGS, 0u);
        putU64(shared, OFF_VEL_POOL, addrOf(dev, bPool));
        std::vector<uint8_t> arti(SZ_ARTI_CORE_DESC, 0u);
        putU32(arti, OFF_NB_ARTI, NB_ARTI);
        putU32(arti, OFF_MAX_LINKS, MAX_LINKS);
        PxgBufferHandle bPrePrep = mk(prePrep), bCore = mk(core), bShared = mk(shared), bArti = mk(arti);

        SolvePush p{};
        p.particleSystems = addrOf(dev, bSystems);
        p.sortedContacts = 0ull;                     // ABI parity; unread
        p.primitiveConstraints = addrOf(dev, bCb);
        p.numContacts = addrOf(dev, bNum);
        p.prePrepDesc = addrOf(dev, bPrePrep);
        p.solverCoreDesc = addrOf(dev, bCore);
        p.sharedDesc = addrOf(dev, bShared);
        p.deltaVel = addrOf(dev, bDeltaVel);
        p.appliedForces = addrOf(dev, bApplied);
        p.dt = dt;
        p.relaxationCoefficient = relax;
        p.isVelIterWord = v.velIter ? 1u : 0u;
        p.artiCoreDesc = addrOf(dev, bArti);
        {
            Kernel k("psContactSolveRigidBda.spv", v.entry, {}, sizeof(SolvePush));
            k.dispatch(1024u, {}, p);
        }
        dev->flush();

        const std::vector<F4> gDeltaVel = downloadBuffer<F4>(dev, bDeltaVel, 2u * N_CONTACTS);
        const std::vector<F2> gApplied = downloadBuffer<F2>(dev, bApplied, N_CONTACTS);
        for (PxgBufferHandle h : all) dev->destroyBuffer(h);

        // ---- compare ----
        uint32_t bad = 0, printed = 0, skipped = 0, zeroed = 0, reacting = 0, inert = 0, artiLanes = 0;
        for (uint32_t k = 0; k < N_CONTACTS; ++k)
        {
            F4 lin{ SENTINEL_F, SENTINEL_F, SENTINEL_F, SENTINEL_F }, ang = lin;
            F2 applied = S.appliedForces[k];
            bool written;
            cpuLane(S, k, v.isTGS, v.velIter, dt, relax, lin, ang, applied, written);
            if (!written) ++skipped;
            else if (lin.w == 0.0f && lin.x == 0.0f && applied.x == 0.0f && applied.y == 0.0f) ++zeroed;
            else if (lin.w != 0.0f) ++reacting;
            else ++inert;
            if (written && laneKind(k) == LK_ARTI) ++artiLanes;
            const bool okLin = std::memcmp(&gDeltaVel[k], &lin, 16) == 0;
            const bool okAng = std::memcmp(&gDeltaVel[k + N_CONTACTS], &ang, 16) == 0;
            const bool okApp = std::memcmp(&gApplied[k], &applied, 8) == 0;
            if (okLin && okAng && okApp) continue;
            ++bad;
            if (printed < 8)
            {
                ++printed;
                std::printf("    lane %u kind %d: lin gpu(%.9g,%.9g,%.9g,%.9g) ref(%.9g,%.9g,%.9g,%.9g) | ang gpu(%.9g,%.9g,%.9g,%.9g) ref(%.9g,%.9g,%.9g,%.9g) | applied gpu(%.9g,%.9g) ref(%.9g,%.9g)\n",
                            k, int(laneKind(k)),
                            gDeltaVel[k].x, gDeltaVel[k].y, gDeltaVel[k].z, gDeltaVel[k].w, lin.x, lin.y, lin.z, lin.w,
                            gDeltaVel[k + N_CONTACTS].x, gDeltaVel[k + N_CONTACTS].y, gDeltaVel[k + N_CONTACTS].z, gDeltaVel[k + N_CONTACTS].w,
                            ang.x, ang.y, ang.z, ang.w, gApplied[k].x, gApplied[k].y, applied.x, applied.y);
            }
        }
        std::printf("  %s velIter=%d: %u lanes -- %u static skipped, %u kinematic zeroed, %u reacting, %u inert, %u articulation\n",
                    v.entry, int(v.velIter), N_CONTACTS, skipped, zeroed, reacting, inert, artiLanes);
        PXG_CHECK_MSG(bad == 0, "%s velIter=%d: %u lanes differ", v.entry, int(v.velIter), bad);
        PXG_CHECK_MSG(skipped == 10u && zeroed == 10u, "lane kinds: %u skipped, %u zeroed (expected 10/10)", skipped, zeroed);
        PXG_CHECK_MSG(reacting >= 30u, "only %u lanes react", reacting);
        PXG_CHECK_MSG(artiLanes >= 15u, "only %u articulation lanes", artiLanes);
        // The adhesion probe must actually pull: with the clamp disabled its deltaF
        // is zero and the whole reaction (weight included) collapses to zero.
        const F4 probe = gDeltaVel[ADHESION_LANE];
        std::printf("    adhesion probe lane %u: lin(%.9g,%.9g,%.9g) weight %.9g\n",
                    ADHESION_LANE, probe.x, probe.y, probe.z, probe.w);
        PXG_CHECK_MSG(probe.w != 0.0f && (probe.x != 0.0f || probe.y != 0.0f || probe.z != 0.0f),
                      "the adhesion probe produced no reaction (weight %.9g) -- the clamp is inert, "
                      "so the adhesion branch is untested", probe.w);
    }
}

} // namespace
