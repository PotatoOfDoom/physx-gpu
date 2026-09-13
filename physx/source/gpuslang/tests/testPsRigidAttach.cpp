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

// P9.76 -- psRigidAttachBda.slang: the PARTICLE-vs-rigid attachment trio.
//
//   ps_rigidAttachmentPrepareLaunch     (particlesystem.cu:5437)
//   ps_solveRigidAttachmentsLaunch      (particlesystem.cu:5567, PGS)
//   ps_solveRigidAttachmentsTGSLaunch   (particlesystem.cu:5648, TGS)
//
// WHAT THIS FIXTURE CHOOSES THAT A SCENE CANNOT.  A scene builds one particle
// system, with one particle buffer, whose attachments start at offset 0 and whose
// sorted order is whatever the grid hash produced.  Every indirection in the
// prepare then collapses to the identity and cannot be wrong:
//
//   - TWO systems, reached through activeParticleSystems in REVERSE order
//     ({1, 0}), so blockIdx.y is an INDIRECTION and not an index.  A port that
//     uses groupId.y directly still passes a one-system scene.
//   - A non-zero mRigidAttachmentOffset on the second system, so the CONSTRAINT
//     index and the ATTACHMENT index differ by 14.  With one system they are equal.
//   - TWO and THREE particle buffers per system, so binarySearch over
//     mAttachmentRunsum does real work and `attachmentOffset` is non-zero.  With
//     one buffer both are zero whatever the code does.
//   - A non-zero mParticleBufferRunsum, so a buffer-local particle id is not a
//     system-global one.
//   - mUnsortedToSortedMapping a real PERMUTATION (p*17+5 mod 62), so a port that
//     skips the reverse lookup reads a different particle.
//   - A KINEMATIC rigid (invMass0 == 0, NOT static) whose sqrtInvInertia is NOT the
//     identity.  This is the trap named in the shader header: the softbody and cloth
//     prepares special-case exactly this lane to the identity inertia with a zero
//     scale, and the particle prepare does NOT.  Without such a lane the two
//     spellings agree and the difference is invisible.
//   - An infinite-mass particle (invMass1 == 0) on a STATIC anchor, which is the
//     only input that takes the `continue` in both solves.
//   - Two attachments on ONE particle, so mAccumDeltaP is genuinely accumulated.
//   - A cone-limit axis that is ZERO on some attachments and non-zero on others:
//     the zero one takes PxVec3::getNormalized()'s guarded branch, the other the
//     divide.  Both are cone-DISABLED (angle/low/high all negative), which is what
//     lets the shader defer the cone branch -- and the fixture ASSERTS that, so the
//     deferral is a measured claim rather than a hope.
//
// The constraint block is compared BIT-EXACTLY against a CPU transcription of
// particlesystem.cu:5437, written from the .cu; both solves' mAccumDeltaP and
// deltaVel outputs are compared bit-exactly against transcriptions of
// attachments.cuh:83 and :147.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

// PxVec3::getNormalized (PxVec3.h:286) -- the guarded form, zero in, zero out.
static V3 normalized3(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f) { const float rs = 1.0f / std::sqrt(m); return V3{ v.x * rs, v.y * rs, v.z * rs }; }
    return V3{ 0.0f, 0.0f, 0.0f };
}

// PxQuat::rotate (PxQuat.h:286), the spelling pxgSlangFemRigidAttach::quatRotateVec
// mirrors.  NOTE the DOUBLED components: every v below is 2*v, and the first draft of
// this transcription dropped that factor -- which the bit-exact comparison caught at
// byte 2 of the very first constraint lane.
static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}

// PxMat33 * PxVec3, column-major as PxgSolverTxIData::sqrtInvInertia stores it.
static V3 mat33Mul(const float m[9], V3 v)
{
    return V3{ m[0] * v.x + m[3] * v.y + m[6] * v.z,
               m[1] * v.x + m[4] * v.y + m[7] * v.z,
               m[2] * v.x + m[5] * v.y + m[8] * v.z };
}

// binarySearch (gpucommon/src/CUDA/reduction.cuh:729).
static uint32_t binarySearchU32(const std::vector<uint32_t>& data, uint32_t first, uint32_t numElements, uint32_t value)
{
    uint32_t left = 0, right = numElements;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2u;
        if (data[first + pos] <= value) left = pos + 1u; else right = pos;
    }
    return left ? left - 1u : 0u;
}

// ---- byte layouts -------------------------------------------------------
// PxgParticleRigidConstraint (PxgSimulationCoreDesc.h:222)
static const uint32_t PRC_raXn0 = 0u, PRC_raXn1 = 512u, PRC_raXn2 = 1024u, PRC_VM = 1536u,
                      PRC_LOWHIGH = 2048u, PRC_AXIS = 2560u, PRC_PARTICLEID = 3072u,
                      PRC_RIGIDID = 3328u, PRC_STRIDE = 3584u;
// PxParticleRigidAttachment (PxDeformableAttachment.h:49)
static const uint32_t PRA_ID0 = 0u, PRA_ID1 = 8u, PRA_LOCALPOSE = 16u, PRA_LOWHIGH = 32u,
                      PRA_AXISANGLE = 48u, PRA_STRIDE = 64u;
// PxgParticleSystem (656) and PxgParticleSimBuffer (88)
static const uint32_t SZ_PSYS = 656u;
static const uint32_t PS_SORTED_POS = 24u, PS_SORTED_VEL = 32u, PS_U2S = 48u, PS_NUM_BUFFERS = 132u,
                      PS_ACCUM_DELTAP = 152u, PS_SORTED_DELTAP = 160u, PS_SIM_BUFFERS = 400u,
                      PS_BUFFER_RUNSUM = 408u, PS_ATTACH_RUNSUM = 456u, PS_NUM_ATT = 476u,
                      PS_ATT_OFFSET = 480u;
static const uint32_t SZ_SIMBUF = 88u, SB_RIGID_ATTACHMENTS = 48u;
// the solver descriptors
static const uint32_t SZ_PREPREP = 760u, OFF_SOLVER_BODY_INDICES = 272u;
static const uint32_t SZ_PREPDESC = 304u, OFF_SBD_POOL = 64u, OFF_TXI_POOL = 72u, OFF_BODY2WORLD = 168u;
static const uint32_t SZ_CORE = 320u, OFF_OUT_SOLVER_VEL = 0u, OFF_NUM_BODIES = 268u, OFF_ACC = 276u;
static const uint32_t SZ_SHARED = 224u, OFF_VEL_POOL = 160u;
static const uint32_t SZ_SBD = 96u, OFF_SBD_LINVEL_INVMASS = 16u;
static const uint32_t SZ_TXI = 64u, OFF_TXI_SQRT = 28u;
static const uint32_t XFORM_STRIDE = 32u;
static const uint64_t STATIC_NODE = 0x00000000ffffffffull;

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h) { return dev->getBufferDeviceAddress(h); }
static void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
static void putF32(std::vector<uint8_t>& b, size_t off, float v) { std::memcpy(b.data() + off, &v, 4); }
static void putF4(std::vector<uint8_t>& b, size_t off, F4 v) { std::memcpy(b.data() + off, &v, 16); }
static F4 getF4(const std::vector<uint8_t>& b, size_t off) { F4 v; std::memcpy(&v, b.data() + off, 16); return v; }

// ---- push blobs, byte for byte the HOST packs ---------------------------
struct PrepPush
{
    uint64_t particleSystems, rigidAttachmentIds, rigidConstraints, preDesc, prepareDesc;
    float dt; uint32_t isTGS;
    uint64_t activeParticleSystems;
};
static_assert(sizeof(PrepPush) == 56, "ps attach prepare blob must be 56 bytes");
static_assert(offsetof(PrepPush, dt) == 40, "dt @40");
static_assert(offsetof(PrepPush, activeParticleSystems) == 48, "activeParticleSystems @48");

// ONE blob for BOTH solves -- the host builds one pack and picks the CUfunction
// with a ternary (PxgParticleSystemCore.cpp:2152), even though the PGS kernel's
// own signature stops at `dt` (64 bytes).  The ABI row must be the HOST's, or the
// launch is refused; the SHADER's own struct is never compared against the row,
// which P9.76 measured -- see psRigidAttachBda.slang's header.
struct SolvePush
{
    uint64_t particleSystems, attachments;
    uint32_t numAttachments, pad20;
    uint64_t prePrepDesc, solverCoreDesc, sharedDesc, deltaVel;
    float dt, biasCoefficient;
    uint32_t isVelocityIteration, pad68;
};
static_assert(sizeof(SolvePush) == 72, "ps attach solve blob must be 72 bytes");
static_assert(offsetof(SolvePush, deltaVel) == 48, "deltaVel @48");
static_assert(offsetof(SolvePush, dt) == 56, "dt @56");
static_assert(offsetof(SolvePush, isVelocityIteration) == 64, "isVelocityIteration @64");

// ===========================================================================
// The scene
// ===========================================================================
// THREE system descriptors are uploaded and only TWO are ACTIVE, in reverse order.
// That asymmetry is what makes blockIdx.y's indirection load-bearing: with the active
// list a PERMUTATION of all systems, reading activeParticleSystems[y] and reading y
// itself merely swap which block does which system, and the union of writes is
// identical -- the first sweep of this fixture had `prep-groupid-y-direct` SURVIVE for
// exactly that reason.  With system 1 present but INACTIVE, a direct index prepares
// the wrong system and leaves system 2's constraint lanes at their sentinel.
static const uint32_t NB_SYS = 3u;               // descriptors uploaded
static const uint32_t NB_ACTIVE = 2u;
static const uint32_t ACTIVE[NB_ACTIVE] = { 2u, 0u };
static const uint32_t NB_BODIES = 6u;            // solver bodies
static const uint32_t ACC = 3u;                  // accumulatedBodyDeltaVOffset
static const uint32_t KINEMATIC_BODY = 2u;       // invMass0 == 0 but NOT static
static const float SENTINEL = -7777.0f;

struct SysDesc
{
    uint32_t nbBuffers;
    uint32_t attPerBuffer[3];
    uint32_t partPerBuffer[3];
    uint32_t attachmentOffset;   // mRigidAttachmentOffset
};
// System 0: 2 buffers, 14 attachments, 24 particles, constraint offset 0.  ACTIVE.
// System 1: 1 buffer,   3 attachments,  6 particles, constraint offset 0.  INACTIVE --
//           it exists only so that a port which uses blockIdx.y as an index has
//           something wrong to find, and its slots overlap system 0's so the mutant
//           corrupts rather than faults.
// System 2: 3 buffers, 26 attachments, 38 particles, constraint offset 14.  ACTIVE.
static const SysDesc SYS[NB_SYS] = {
    { 2u, { 5u, 9u, 0u }, { 11u, 13u, 0u }, 0u },
    { 1u, { 3u, 0u, 0u }, { 6u, 0u, 0u }, 0u },
    { 3u, { 4u, 7u, 15u }, { 9u, 12u, 17u }, 14u },
};
static const uint32_t N_ATT_TOTAL = 40u;         // 14 + 26, the ACTIVE systems only
static const uint32_t MAX_ATT_PER_SYS = 26u;

struct Scene
{
    // per system
    uint32_t nbParticles[NB_SYS];
    std::vector<uint32_t> attachRunsum[NB_SYS];
    std::vector<uint32_t> bufferRunsum[NB_SYS];
    std::vector<uint32_t> unsortedToSorted[NB_SYS];
    std::vector<F4> sortedPos[NB_SYS];      // xyz + invMass
    std::vector<F4> sortedVel[NB_SYS];
    std::vector<F4> sortedDeltaP[NB_SYS];   // xyz + invMass
    std::vector<uint8_t> attachments[NB_SYS];   // PRA_STRIDE each, flat across buffers
    // shared
    std::vector<uint32_t> solverBodyIndices;    // node index -> solver body index
    std::vector<uint8_t> solverBodyData;        // NB_BODIES * SZ_SBD
    std::vector<uint8_t> txIData;               // NB_BODIES * SZ_TXI
    std::vector<F4> bodyFrames;                 // NB_BODIES * 2 (quat, pos)
    std::vector<F4> initialVel;                 // 2 * NB_BODIES
    std::vector<F4> pool;                       // (ACC + 3*NB_BODIES)
};

struct Rng
{
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uni(float lo, float hi) { return lo + (hi - lo) * float(next() & 0xffffffu) / 16777216.0f; }
};

// Which attachment lane is which, so the checks can say what they covered.
static bool laneIsStatic(uint32_t sys, uint32_t a) { return ((a + sys) % 7u) == 0u; }
static uint32_t laneNodeIndex(uint32_t sys, uint32_t a) { return (a * 3u + sys) % 8u; }

static Scene buildScene()
{
    Rng rng{ 0x9e3779b9u };
    Scene S;

    S.solverBodyIndices.resize(8u);
    // Two of the eight node indices resolve to the KINEMATIC body, so enough lanes
    // carry invMass0 == 0 with a non-identity inertia for that control to have teeth.
    //
    // Entry 7 must be NON-ZERO, and that is not cosmetic.  A static rigid's node index
    // is 0xffffffff; a port that drops the `if (!isStatic)` guard and indexes anyway
    // lands on entry (0xffffffff & 7) == 7, and if that entry happened to hold 0 the
    // unguarded read would agree with the guarded `idx = 0` and the defect would be
    // invisible.  The first sweep of this fixture had exactly that: entry 7 was 0 and
    // `prep-rigidid-not-static-guard` SURVIVED.
    for (uint32_t n = 0; n < 8u; ++n)
        S.solverBodyIndices[n] = (n % 3u == 2u) ? KINEMATIC_BODY : ((n * 5u + 3u) % NB_BODIES);

    S.solverBodyData.assign(size_t(NB_BODIES) * SZ_SBD, 0u);
    S.txIData.assign(size_t(NB_BODIES) * SZ_TXI, 0u);
    S.bodyFrames.resize(size_t(NB_BODIES) * 2u);
    for (uint32_t b = 0; b < NB_BODIES; ++b)
    {
        // invMass0: zero for the KINEMATIC body, distinct otherwise.
        const float invMass0 = (b == KINEMATIC_BODY) ? 0.0f : (0.2f + 0.13f * float(b));
        putF4(S.solverBodyData, size_t(b) * SZ_SBD + OFF_SBD_LINVEL_INVMASS,
              F4{ rng.uni(-0.4f, 0.4f), rng.uni(-0.4f, 0.4f), rng.uni(-0.4f, 0.4f), invMass0 });
        // A NON-identity sqrtInvInertia on EVERY body, the kinematic one included:
        // that is what makes the missing identity-inertia special case observable.
        float m[9];
        for (int k = 0; k < 9; ++k) m[k] = rng.uni(-0.9f, 1.6f);
        std::memcpy(S.txIData.data() + size_t(b) * SZ_TXI + OFF_TXI_SQRT, m, 36);
        // a real rotation, normalised
        V3 ax = normalized3(V3{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f) });
        const float ang = rng.uni(-2.8f, 2.8f);
        const float s = std::sin(ang * 0.5f), c = std::cos(ang * 0.5f);
        S.bodyFrames[size_t(b) * 2u + 0u] = F4{ ax.x * s, ax.y * s, ax.z * s, c };
        S.bodyFrames[size_t(b) * 2u + 1u] = F4{ rng.uni(-3.f, 3.f), rng.uni(-3.f, 3.f), rng.uni(-3.f, 3.f), 0.0f };
    }

    S.initialVel.assign(size_t(NB_BODIES) * 2u, F4{ 0, 0, 0, 0 });
    for (uint32_t b = 0; b < NB_BODIES; ++b)
    {
        S.initialVel[b] = F4{ rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), 0.0f };
        S.initialVel[b + NB_BODIES] = F4{ rng.uni(-0.4f, 0.4f), rng.uni(-0.4f, 0.4f), rng.uni(-0.4f, 0.4f), 0.0f };
    }
    // THREE rows per body, all non-zero: the TGS arm unpacks linVel/angVel/linDelta/
    // angDelta across the row boundaries and a port that reads two rows gets
    // linDelta == angDelta == 0 (the defect P9.76c found in the softbody twin).
    S.pool.assign(size_t(ACC) + size_t(NB_BODIES) * 3u, F4{ 0, 0, 0, 0 });
    for (size_t i = ACC; i < S.pool.size(); ++i)
        S.pool[i] = F4{ rng.uni(-0.3f, 0.3f), rng.uni(-0.3f, 0.3f), rng.uni(-0.3f, 0.3f), rng.uni(-0.3f, 0.3f) };

    for (uint32_t sys = 0; sys < NB_SYS; ++sys)
    {
        const SysDesc& d = SYS[sys];
        uint32_t nbAtt = 0, nbPart = 0;
        S.attachRunsum[sys].resize(d.nbBuffers);
        S.bufferRunsum[sys].resize(d.nbBuffers);
        for (uint32_t b = 0; b < d.nbBuffers; ++b)
        {
            // EXCLUSIVE prefix sums: PxgPBDParticleSystemCore.cpp:1609-1615 stores
            // the running total BEFORE adding the buffer's own count.
            S.attachRunsum[sys][b] = nbAtt;
            S.bufferRunsum[sys][b] = nbPart;
            nbAtt += d.attPerBuffer[b];
            nbPart += d.partPerBuffer[b];
        }
        S.nbParticles[sys] = nbPart;

        // A real permutation, so the reverse lookup is load-bearing.
        S.unsortedToSorted[sys].resize(nbPart);
        for (uint32_t p = 0; p < nbPart; ++p) S.unsortedToSorted[sys][p] = (p * 17u + 5u) % nbPart;

        S.sortedPos[sys].resize(nbPart);
        S.sortedVel[sys].resize(nbPart);
        S.sortedDeltaP[sys].resize(nbPart);
        for (uint32_t p = 0; p < nbPart; ++p)
        {
            S.sortedPos[sys][p] = F4{ rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), 0.4f + 0.05f * float(p % 7u) };
            S.sortedVel[sys][p] = F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), 0.0f };
            S.sortedDeltaP[sys][p] = F4{ rng.uni(-0.05f, 0.05f), rng.uni(-0.05f, 0.05f), rng.uni(-0.05f, 0.05f),
                                         S.sortedPos[sys][p].w };
        }

        S.attachments[sys].assign(size_t(nbAtt) * PRA_STRIDE, 0u);
        for (uint32_t a = 0; a < nbAtt; ++a)
        {
            const uint32_t bufferIndex = binarySearchU32(S.attachRunsum[sys], 0u, d.nbBuffers, a);
            const uint32_t local = a - (bufferIndex == 0u ? 0u : S.attachRunsum[sys][bufferIndex]);
            // Buffer-LOCAL particle id, deliberately not the global one.
            uint32_t localParticle = local % d.partPerBuffer[bufferIndex];
            // Two attachments of buffer 0 land on the SAME particle, so mAccumDeltaP
            // is genuinely accumulated rather than merely written.
            if (bufferIndex == 0u && local == 1u) localParticle = 0u;

            const uint64_t rigidId = laneIsStatic(sys, a) ? STATIC_NODE : uint64_t(laneNodeIndex(sys, a));
            uint8_t* rec = S.attachments[sys].data() + size_t(a) * PRA_STRIDE;
            std::memcpy(rec + PRA_ID0, &rigidId, 8);
            const uint64_t id1 = uint64_t(localParticle);
            std::memcpy(rec + PRA_ID1, &id1, 8);
            const F4 localPose{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), 0.0f };
            std::memcpy(rec + PRA_LOCALPOSE, &localPose, 16);
            // Cone limit DISABLED on every attachment (lowLimit, highLimit < 0 and
            // angle < 0), which is PxConeLimitedConstraint's default.  Every third
            // one keeps the default ZERO axis so getNormalized()'s guard is taken.
            const F4 lowHigh{ -1.0f, -1.0f, 0.0f, 0.0f };
            std::memcpy(rec + PRA_LOWHIGH, &lowHigh, 16);
            const F4 axisAngle = (a % 3u == 0u)
                ? F4{ 0.0f, 0.0f, 0.0f, -1.0f }
                : F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), -1.0f };
            std::memcpy(rec + PRA_AXISANGLE, &axisAngle, 16);
        }
    }

    // ONE particle with INFINITE mass under a STATIC anchor -- the only input that
    // takes the `continue` in both solves.  Attachment 0 of system 0 is static by
    // construction (laneIsStatic(0,0) == true); pin its particle's inverse mass.
    {
        const uint32_t sys = 0u, a = 0u;
        const uint32_t bufferIndex = binarySearchU32(S.attachRunsum[sys], 0u, SYS[sys].nbBuffers, a);
        uint64_t id1; std::memcpy(&id1, S.attachments[sys].data() + size_t(a) * PRA_STRIDE + PRA_ID1, 8);
        const uint32_t global = uint32_t(id1) + S.bufferRunsum[sys][bufferIndex];
        const uint32_t sorted = S.unsortedToSorted[sys][global];
        S.sortedPos[sys][sorted].w = 0.0f;
        S.sortedDeltaP[sys][sorted].w = 0.0f;
    }
    return S;
}

// ===========================================================================
// The CPU transcription of ps_rigidAttachmentPrepareLaunch (particlesystem.cu:5437)
// ===========================================================================
static void cpuPrepare(const Scene& S, std::vector<uint8_t>& cb, std::vector<uint64_t>& ids)
{
    for (uint32_t ai = 0; ai < NB_ACTIVE; ++ai)
    {
        const uint32_t sys = ACTIVE[ai];
        const SysDesc& d = SYS[sys];
        const uint32_t nbAtt = S.attachRunsum[sys][d.nbBuffers - 1u] + d.attPerBuffer[d.nbBuffers - 1u];
        for (uint32_t a = 0; a < nbAtt; ++a)
        {
            const uint32_t constraintIdx = a + d.attachmentOffset;
            const uint32_t index = constraintIdx / 32u, offset = constraintIdx & 31u;
            const size_t base = size_t(index) * PRC_STRIDE;

            const uint32_t bufferIndex = binarySearchU32(S.attachRunsum[sys], 0u, d.nbBuffers, a);
            const uint32_t attachmentOffset = (bufferIndex == 0u) ? 0u : S.attachRunsum[sys][bufferIndex];

            const uint8_t* rec = S.attachments[sys].data() + size_t(a - attachmentOffset + attachmentOffset) * PRA_STRIDE;
            // (the record index IS `a` -- the flat array holds all buffers' records
            // back to back, exactly as buffer->mRigidAttachments[a - attachmentOffset]
            // reaches when each buffer's pointer is the flat base plus its run-sum)
            rec = S.attachments[sys].data() + size_t(a) * PRA_STRIDE;

            uint64_t id0, id1;
            std::memcpy(&id0, rec + PRA_ID0, 8);
            std::memcpy(&id1, rec + PRA_ID1, 8);
            F4 localPose, lowHigh, axisAngle;
            std::memcpy(&localPose, rec + PRA_LOCALPOSE, 16);
            std::memcpy(&lowHigh, rec + PRA_LOWHIGH, 16);
            std::memcpy(&axisAngle, rec + PRA_AXISANGLE, 16);

            const uint32_t particleId = uint32_t(id1) + S.bufferRunsum[sys][bufferIndex];
            const uint32_t sortedParticleId = S.unsortedToSorted[sys][particleId];
            const F4 pointInvM = S.sortedPos[sys][sortedParticleId];
            const float invMass1 = pointInvM.w;
            const V3 point{ pointInvM.x, pointInvM.y, pointInvM.z };
            const V3 axis{ axisAngle.x, axisAngle.y, axisAngle.z };
            const V3 ra4{ localPose.x, localPose.y, localPose.z };

            uint32_t idx = 0;
            if (uint32_t(id0 & 0xffffffffull) != 0xffffffffu)
                idx = S.solverBodyIndices[uint32_t(id0 & 0xffffffffull)];

            ids[constraintIdx] = id0;

            float m[9];
            std::memcpy(m, S.txIData.data() + size_t(idx) * SZ_TXI + OFF_TXI_SQRT, 36);
            const F4 linVel_invMass0 = getF4(S.solverBodyData, size_t(idx) * SZ_SBD + OFF_SBD_LINVEL_INVMASS);
            const float invMass0 = linVel_invMass0.w;

            const F4 bfQ = S.bodyFrames[size_t(idx) * 2u + 0u];
            const F4 bfP = S.bodyFrames[size_t(idx) * 2u + 1u];

            const V3 worldAxis = normalized3(quatRotate(bfQ, axis));
            const V3 ra = quatRotate(bfQ, ra4);
            const V3 error{ ra.x + bfP.x - point.x, ra.y + bfP.y - point.y, ra.z + bfP.z - point.z };

            const V3 n0{ 1, 0, 0 }, n1{ 0, 1, 0 }, n2{ 0, 0, 1 };
            const V3 s0 = mat33Mul(m, cross3(ra, n0));
            const V3 s1 = mat33Mul(m, cross3(ra, n1));
            const V3 s2 = mat33Mul(m, cross3(ra, n2));
            // NO inertiaScale and NO identity special case -- particlesystem.cu:5533.
            const float resp0 = dot3(s0, s0) + invMass0 + invMass1;
            const float resp1 = dot3(s1, s1) + invMass0 + invMass1;
            const float resp2 = dot3(s2, s2) + invMass0 + invMass1;

            putF4(cb, base + PRC_raXn0 + offset * 16u, F4{ s0.x, s0.y, s0.z, dot3(error, n0) });
            putF4(cb, base + PRC_raXn1 + offset * 16u, F4{ s1.x, s1.y, s1.z, dot3(error, n1) });
            putF4(cb, base + PRC_raXn2 + offset * 16u, F4{ s2.x, s2.y, s2.z, dot3(error, n2) });
            putF4(cb, base + PRC_VM + offset * 16u, F4{ (resp0 > 0.f) ? 1.f / resp0 : 0.f,
                                                        (resp1 > 0.f) ? 1.f / resp1 : 0.f,
                                                        (resp2 > 0.f) ? 1.f / resp2 : 0.f, invMass0 });
            const uint64_t encoded = (uint64_t(sys) << 32) | uint64_t(sortedParticleId);
            std::memcpy(cb.data() + base + PRC_PARTICLEID + offset * 8u, &encoded, 8);
            std::memcpy(cb.data() + base + PRC_RIGIDID + offset * 8u, &id0, 8);
            putF4(cb, base + PRC_LOWHIGH + offset * 16u, lowHigh);
            putF4(cb, base + PRC_AXIS + offset * 16u, F4{ worldAxis.x, worldAxis.y, worldAxis.z, axisAngle.w });
        }
    }
}

// ===========================================================================
// The CPU transcriptions of the two solves.
// calculateAttachmentDeltaImpulsePGS (attachments.cuh:83) and
// calculateAttachmentDeltaImpulseTGS (attachments.cuh:147), cone branch omitted
// because the fixture's data has it disabled -- which is asserted separately.
// ===========================================================================
struct SolveRef { bool wrote; V3 deltaParticle; uint32_t sys, particleId; F4 lin, ang; };

static SolveRef cpuSolveLane(const Scene& S, const std::vector<uint8_t>& cb, uint32_t workIndex,
                             bool isTGS, float dt, float biasCoefficient, bool isVelocityIteration)
{
    SolveRef r; r.wrote = false; r.sys = 0; r.particleId = 0;
    const uint32_t index = workIndex / 32u, offset = workIndex & 31u;
    const size_t base = size_t(index) * PRC_STRIDE;

    uint64_t compressedId; std::memcpy(&compressedId, cb.data() + base + PRC_PARTICLEID + offset * 8u, 8);
    const uint32_t sys = uint32_t(compressedId >> 32), particleId = uint32_t(compressedId);
    r.sys = sys; r.particleId = particleId;

    V3 lin1; float invMass1;
    if (isTGS)
    {
        const F4 dp = S.sortedDeltaP[sys][particleId];
        invMass1 = dp.w; lin1 = V3{ dp.x, dp.y, dp.z };
    }
    else
    {
        const F4 v = S.sortedVel[sys][particleId];
        invMass1 = S.sortedPos[sys][particleId].w;
        lin1 = V3{ v.x, v.y, v.z };
    }

    uint64_t rigidId; std::memcpy(&rigidId, cb.data() + base + PRC_RIGIDID + offset * 8u, 8);
    const bool isStatic = uint32_t(rigidId & 0xffffffffull) == 0xffffffffu;
    uint32_t solverBodyIndex = 0;
    if (!isStatic) solverBodyIndex = S.solverBodyIndices[uint32_t(rigidId & 0xffffffffull)];
    else if (invMass1 == 0.0f) return r;   // continue

    const F4 r0 = getF4(cb, base + PRC_raXn0 + offset * 16u);
    const F4 r1 = getF4(cb, base + PRC_raXn1 + offset * 16u);
    const F4 r2 = getF4(cb, base + PRC_raXn2 + offset * 16u);
    const F4 vm = getF4(cb, base + PRC_VM + offset * 16u);
    const V3 raXn0{ r0.x, r0.y, r0.z }, raXn1{ r1.x, r1.y, r1.z }, raXn2{ r2.x, r2.y, r2.z };

    float deltaF0, deltaF1, deltaF2;
    if (isTGS)
    {
        const F4 v0 = S.pool[ACC + solverBodyIndex];
        const F4 v1 = S.pool[ACC + solverBodyIndex + NB_BODIES];
        const F4 v2 = S.pool[ACC + solverBodyIndex + 2u * NB_BODIES];
        const V3 linVel{ v0.x, v0.y, v0.z };
        const V3 angVel{ v0.w, v1.x, v1.y };
        const V3 linDelta0{ v1.z, v1.w, v2.x };
        const V3 angDelta{ v2.y, v2.z, v2.w };

        const float vr0 = linVel.x + dot3(angVel, raXn0);
        const float vr1 = linVel.y + dot3(angVel, raXn1);
        const float vr2 = linVel.z + dot3(angVel, raXn2);
        const V3 linDelta{ lin1.x - linDelta0.x, lin1.y - linDelta0.y, lin1.z - linDelta0.z };
        const V3 tgsError{ linDelta.x - r0.w - dot3(angDelta, raXn0),
                           linDelta.y - r1.w - dot3(angDelta, raXn1),
                           linDelta.z - r2.w - dot3(angDelta, raXn2) };
        const float velDt = isVelocityIteration ? 0.0f : dt;
        deltaF0 = (tgsError.x - vr0 * velDt) * vm.x * biasCoefficient;
        deltaF1 = (tgsError.y - vr1 * velDt) * vm.y * biasCoefficient;
        deltaF2 = (tgsError.z - vr2 * velDt) * vm.z * biasCoefficient;
    }
    else
    {
        const F4 iv0 = S.initialVel[solverBodyIndex];
        const F4 iv1 = S.initialVel[solverBodyIndex + NB_BODIES];
        const F4 dv0 = S.pool[ACC + solverBodyIndex];
        const F4 dv1 = S.pool[ACC + solverBodyIndex + NB_BODIES];
        const V3 linVel{ iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z };
        const V3 angVel{ iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z };

        const float vr0 = linVel.x + dot3(angVel, raXn0);
        const float vr1 = linVel.y + dot3(angVel, raXn1);
        const float vr2 = linVel.z + dot3(angVel, raXn2);
        const float invDt = 1.0f / dt, biasFactor = 0.5f;
        const V3 velError{ lin1.x - r0.w * biasFactor * invDt,
                           lin1.y - r1.w * biasFactor * invDt,
                           lin1.z - r2.w * biasFactor * invDt };
        deltaF0 = (velError.x - vr0) * vm.x;
        deltaF1 = (velError.y - vr1) * vm.y;
        deltaF2 = (velError.z - vr2) * vm.z;
    }

    if (deltaF0 == 0.0f && deltaF1 == 0.0f && deltaF2 == 0.0f) return r;
    r.wrote = true;
    if (isTGS)
        r.deltaParticle = V3{ (-deltaF0) * invMass1, (-deltaF1) * invMass1, (-deltaF2) * invMass1 };
    else
        r.deltaParticle = V3{ (-deltaF0) * invMass1 * dt, (-deltaF1) * invMass1 * dt, (-deltaF2) * invMass1 * dt };
    r.lin = F4{ deltaF0 * vm.w, deltaF1 * vm.w, deltaF2 * vm.w, 1.0f };
    r.ang = F4{ raXn0.x * deltaF0 + raXn1.x * deltaF1 + raXn2.x * deltaF2,
                raXn0.y * deltaF0 + raXn1.y * deltaF1 + raXn2.y * deltaF2,
                raXn0.z * deltaF0 + raXn1.z * deltaF1 + raXn2.z * deltaF2, 0.0f };
    return r;
}

} // namespace

PXG_TEST(psRigidAttachPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    const Scene S = buildScene();
    const float dt = 1.0f / 60.0f;

    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    std::vector<uint8_t> systems(size_t(NB_SYS) * SZ_PSYS, 0u);
    std::vector<PxgBufferHandle> simBufHandles;
    for (uint32_t sys = 0; sys < NB_SYS; ++sys)
    {
        const SysDesc& d = SYS[sys];
        const uint32_t nbAtt = S.attachRunsum[sys][d.nbBuffers - 1u] + d.attPerBuffer[d.nbBuffers - 1u];
        PxgBufferHandle bAtt = mk(S.attachments[sys]);
        if (!addrOf(dev, bAtt)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
        // Each buffer's mRigidAttachments points at ITS slice of the flat array --
        // which is what makes `mRigidAttachments[attachmentIdx - attachmentOffset]`
        // reach record `attachmentIdx`.
        std::vector<uint8_t> simBufs(size_t(d.nbBuffers) * SZ_SIMBUF, 0u);
        for (uint32_t b = 0; b < d.nbBuffers; ++b)
            putU64(simBufs, size_t(b) * SZ_SIMBUF + SB_RIGID_ATTACHMENTS,
                   addrOf(dev, bAtt) + uint64_t(S.attachRunsum[sys][b]) * PRA_STRIDE);
        PxgBufferHandle bSimBufs = mk(simBufs);
        PxgBufferHandle bPos = mk(S.sortedPos[sys]);
        PxgBufferHandle bU2S = mk(S.unsortedToSorted[sys]);
        PxgBufferHandle bAttRunsum = mk(S.attachRunsum[sys]);
        PxgBufferHandle bBufRunsum = mk(S.bufferRunsum[sys]);

        const size_t o = size_t(sys) * SZ_PSYS;
        putU64(systems, o + PS_SORTED_POS, addrOf(dev, bPos));
        putU64(systems, o + PS_U2S, addrOf(dev, bU2S));
        putU32(systems, o + PS_NUM_BUFFERS, d.nbBuffers);
        putU64(systems, o + PS_SIM_BUFFERS, addrOf(dev, bSimBufs));
        putU64(systems, o + PS_BUFFER_RUNSUM, addrOf(dev, bBufRunsum));
        putU64(systems, o + PS_ATTACH_RUNSUM, addrOf(dev, bAttRunsum));
        putU32(systems, o + PS_NUM_ATT, nbAtt);
        putU32(systems, o + PS_ATT_OFFSET, d.attachmentOffset);
        simBufHandles.push_back(bSimBufs);
    }
    PxgBufferHandle bSystems = mk(systems);

    // activeParticleSystems names TWO of the THREE systems, in reverse order: with
    // system 1 absent from the list, blockIdx.y is an indirection and not an index.
    PxgBufferHandle bActive = mk(std::vector<uint32_t>{ ACTIVE[0], ACTIVE[1] });
    const uint32_t nbBlocks = (N_ATT_TOTAL / 32u) + 2u;   // room for every touched block
    std::vector<uint8_t> cbInit(size_t(nbBlocks) * PRC_STRIDE, 0xCDu);
    PxgBufferHandle bCb = mk(cbInit);
    PxgBufferHandle bIds = mk(std::vector<uint64_t>(N_ATT_TOTAL, 0xEEEEEEEEEEEEEEEEull));

    PxgBufferHandle bSbi = mk(S.solverBodyIndices);
    PxgBufferHandle bSbd = mk(S.solverBodyData);
    PxgBufferHandle bTxI = mk(S.txIData);
    PxgBufferHandle bFrames = mk(S.bodyFrames);

    std::vector<uint8_t> prePrep(SZ_PREPREP, 0u);
    putU64(prePrep, OFF_SOLVER_BODY_INDICES, addrOf(dev, bSbi));
    std::vector<uint8_t> prepDesc(SZ_PREPDESC, 0u);
    putU64(prepDesc, OFF_SBD_POOL, addrOf(dev, bSbd));
    putU64(prepDesc, OFF_TXI_POOL, addrOf(dev, bTxI));
    putU64(prepDesc, OFF_BODY2WORLD, addrOf(dev, bFrames));
    PxgBufferHandle bPrePrep = mk(prePrep), bPrepDesc = mk(prepDesc);

    PrepPush p{};
    p.particleSystems = addrOf(dev, bSystems);
    p.rigidAttachmentIds = addrOf(dev, bIds);
    p.rigidConstraints = addrOf(dev, bCb);
    p.preDesc = addrOf(dev, bPrePrep);
    p.prepareDesc = addrOf(dev, bPrepDesc);
    p.dt = dt;              // DEAD in the kernel; present because the host pack has it
    p.isTGS = 0u;           // DEAD likewise
    p.activeParticleSystems = addrOf(dev, bActive);

    // The ENGINE's grid (PxgParticleSystemCore.cpp:2382): ceil(maxAttachmentsPerSystem
    // / PS_COLLISION) by nbActiveParticleSystems.  Not a grid sized to be convenient.
    const uint32_t gridX = (MAX_ATT_PER_SYS + 256u - 1u) / 256u;
    {
        Kernel k("psRigidAttachBda.spv", "ps_rigidAttachmentPrepareLaunch", {}, sizeof(PrepPush));
        k.dispatch(gridX, {}, p, NB_ACTIVE, 1u);
    }
    dev->flush();

    const std::vector<uint8_t> gCb = downloadBuffer<uint8_t>(dev, bCb, cbInit.size());
    const std::vector<uint64_t> gIds = downloadBuffer<uint64_t>(dev, bIds, N_ATT_TOTAL);
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);

    std::vector<uint8_t> refCb = cbInit;
    std::vector<uint64_t> refIds(N_ATT_TOTAL, 0xEEEEEEEEEEEEEEEEull);
    cpuPrepare(S, refCb, refIds);

    // Bit-exact over the WHOLE region, sentinel bytes included: a lane written that
    // should not have been is as much a defect as a lane computed wrongly.
    size_t firstBad = size_t(-1), nbad = 0;
    for (size_t i = 0; i < refCb.size(); ++i)
        if (gCb[i] != refCb[i]) { if (nbad == 0) firstBad = i; ++nbad; }
    if (nbad)
    {
        const size_t blk = firstBad / PRC_STRIDE, within = firstBad % PRC_STRIDE;
        std::printf("    first mismatch at byte %zu (block %zu, offset %zu in block)\n", firstBad, blk, within);
    }
    PXG_CHECK_MSG(nbad == 0, "the prepared constraint block must match the CPU transcription "
                             "BIT-EXACTLY (%zu of %zu bytes differ)", nbad, refCb.size());

    uint32_t idBad = 0;
    for (uint32_t i = 0; i < N_ATT_TOTAL; ++i) if (gIds[i] != refIds[i]) ++idBad;
    PXG_CHECK_MSG(idBad == 0, "rigidAttachmentIds must be written at the CONSTRAINT index "
                              "(%u of %u differ)", idBad, N_ATT_TOTAL);

    // The controls: the fixture's data must actually exercise what it claims to.
    uint32_t statics = 0, kinematics = 0, zeroAxis = 0, offsetLanes = 0, permuted = 0;
    for (uint32_t ai = 0; ai < NB_ACTIVE; ++ai)
    {
        const uint32_t sys = ACTIVE[ai];
        const SysDesc& d = SYS[sys];
        const uint32_t nbAtt = S.attachRunsum[sys][d.nbBuffers - 1u] + d.attPerBuffer[d.nbBuffers - 1u];
        for (uint32_t a = 0; a < nbAtt; ++a)
        {
            if (laneIsStatic(sys, a)) ++statics;
            else if (S.solverBodyIndices[laneNodeIndex(sys, a)] == KINEMATIC_BODY) ++kinematics;
            if (a % 3u == 0u) ++zeroAxis;
            if (d.attachmentOffset != 0u) ++offsetLanes;
        }
        for (uint32_t pp = 0; pp < S.nbParticles[sys]; ++pp)
            if (S.unsortedToSorted[sys][pp] != pp) ++permuted;
    }
    std::printf("  %u attachments: %u static, %u kinematic-anchored, %u zero-axis, %u at a non-zero "
                "constraint offset, %u permuted particles\n",
                N_ATT_TOTAL, statics, kinematics, zeroAxis, offsetLanes, permuted);
    PXG_CHECK_MSG(statics >= 4u && kinematics >= 3u,
                  "the fixture must cover static (%u) and KINEMATIC (%u) anchors -- the kinematic "
                  "lane is the one that separates this prepare from the softbody/cloth twins, "
                  "which give it the identity inertia", statics, kinematics);
    PXG_CHECK_MSG(zeroAxis >= 8u, "only %u zero-axis lanes: getNormalized()'s guard is untested", zeroAxis);
    PXG_CHECK_MSG(offsetLanes == 26u, "the non-zero mRigidAttachmentOffset must cover system 2's "
                                      "26 lanes (got %u)", offsetLanes);
    PXG_CHECK_MSG(S.solverBodyIndices[7] != 0u,
                  "solverBodyIndices[7] must be non-zero: it is where an unguarded read of a "
                  "STATIC node index (0xffffffff & 7) lands, and a zero there would agree with "
                  "the guarded idx = 0");
    PXG_CHECK_MSG(permuted >= 50u, "only %u particles are permuted: the reverse lookup is untested", permuted);

    // The deferral of the cone branch is a MEASURED claim: isConeLimitedEnabled
    // (attachments.cuh:36) must be false for every lane this fixture prepared.
    uint32_t coneOn = 0;
    for (uint32_t ai = 0; ai < NB_ACTIVE; ++ai)
    {
        const uint32_t sys = ACTIVE[ai];
        const SysDesc& d = SYS[sys];
        const uint32_t nbAtt = S.attachRunsum[sys][d.nbBuffers - 1u] + d.attPerBuffer[d.nbBuffers - 1u];
        for (uint32_t a = 0; a < nbAtt; ++a)
        {
            F4 lh, aa;
            std::memcpy(&lh, S.attachments[sys].data() + size_t(a) * PRA_STRIDE + PRA_LOWHIGH, 16);
            std::memcpy(&aa, S.attachments[sys].data() + size_t(a) * PRA_STRIDE + PRA_AXISANGLE, 16);
            if (aa.w >= 0.f || lh.x >= 0.f || lh.y >= 0.f) ++coneOn;
        }
    }
    PXG_CHECK_MSG(coneOn == 0u, "%u lanes have the cone limit ENABLED -- the shader defers that "
                                "branch, so a fixture that enables it is comparing against a "
                                "deferral", coneOn);
}

PXG_TEST(psRigidAttachSolve)
{
    PxgSlangComputeDevice* dev = testDevice();
    const Scene S = buildScene();
    const float dt = 1.0f / 60.0f;
    const float biasCoefficient = 0.7f;

    struct Variant { const char* entry; bool isTGS; bool velIter; };
    const Variant variants[] = {
        { "ps_solveRigidAttachmentsLaunch", false, false },
        { "ps_solveRigidAttachmentsTGSLaunch", true, false },
        { "ps_solveRigidAttachmentsTGSLaunch", true, true },
    };

    // The constraint block the solves read is produced by the prepare on the CPU
    // side of this fixture -- the GPU prepare is already bit-exact against it
    // (psRigidAttachPrepare), so using the reference block keeps the two tests
    // independent: a prepare regression fails ONE test, not both.
    const uint32_t nbBlocks = (N_ATT_TOTAL / 32u) + 2u;
    std::vector<uint8_t> cb(size_t(nbBlocks) * PRC_STRIDE, 0u);
    {
        std::vector<uint64_t> ids(N_ATT_TOTAL, 0ull);
        cpuPrepare(S, cb, ids);
    }

    for (const Variant& v : variants)
    {
        std::vector<PxgBufferHandle> all;
        auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

        PxgBufferHandle bCb = mk(cb);
        if (!addrOf(dev, bCb)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }

        std::vector<uint8_t> systems(size_t(NB_SYS) * SZ_PSYS, 0u);
        std::vector<PxgBufferHandle> bAccum(NB_SYS);
        for (uint32_t sys = 0; sys < NB_SYS; ++sys)
        {
            PxgBufferHandle bPos = mk(S.sortedPos[sys]);
            PxgBufferHandle bVel = mk(S.sortedVel[sys]);
            PxgBufferHandle bDeltaP = mk(S.sortedDeltaP[sys]);
            bAccum[sys] = mk(std::vector<F4>(S.nbParticles[sys], F4{ 0, 0, 0, 0 }));
            const size_t o = size_t(sys) * SZ_PSYS;
            putU64(systems, o + PS_SORTED_POS, addrOf(dev, bPos));
            putU64(systems, o + PS_SORTED_VEL, addrOf(dev, bVel));
            putU64(systems, o + PS_ACCUM_DELTAP, addrOf(dev, bAccum[sys]));
            putU64(systems, o + PS_SORTED_DELTAP, addrOf(dev, bDeltaP));
        }
        PxgBufferHandle bSystems = mk(systems);
        PxgBufferHandle bSbi = mk(S.solverBodyIndices);
        PxgBufferHandle bInitialVel = mk(S.initialVel);
        PxgBufferHandle bPool = mk(S.pool);
        PxgBufferHandle bDeltaVel = mk(std::vector<F4>(2u * N_ATT_TOTAL, F4{ SENTINEL, SENTINEL, SENTINEL, SENTINEL }));

        std::vector<uint8_t> prePrep(SZ_PREPREP, 0u);
        putU64(prePrep, OFF_SOLVER_BODY_INDICES, addrOf(dev, bSbi));
        std::vector<uint8_t> core(SZ_CORE, 0u);
        putU64(core, OFF_OUT_SOLVER_VEL, addrOf(dev, bInitialVel));
        putU32(core, OFF_NUM_BODIES, NB_BODIES);
        putU32(core, OFF_ACC, ACC);
        std::vector<uint8_t> shared(SZ_SHARED, 0u);
        putU64(shared, OFF_VEL_POOL, addrOf(dev, bPool));
        PxgBufferHandle bPrePrep = mk(prePrep), bCore = mk(core), bShared = mk(shared);

        SolvePush p{};
        p.particleSystems = addrOf(dev, bSystems);
        p.attachments = addrOf(dev, bCb);
        p.numAttachments = N_ATT_TOTAL;
        p.prePrepDesc = addrOf(dev, bPrePrep);
        p.solverCoreDesc = addrOf(dev, bCore);
        p.sharedDesc = addrOf(dev, bShared);
        p.deltaVel = addrOf(dev, bDeltaVel);
        p.dt = dt;
        p.biasCoefficient = biasCoefficient;
        p.isVelocityIteration = v.velIter ? 1u : 0u;
        {
            Kernel k("psRigidAttachBda.spv", v.entry, {}, sizeof(SolvePush));
            k.dispatch(1024u, {}, p);   // the engine's PS_COLLISION grid
        }
        dev->flush();

        const std::vector<F4> gDeltaVel = downloadBuffer<F4>(dev, bDeltaVel, 2u * N_ATT_TOTAL);
        std::vector<std::vector<F4>> gAccum(NB_SYS);
        for (uint32_t sys = 0; sys < NB_SYS; ++sys)
            gAccum[sys] = downloadBuffer<F4>(dev, bAccum[sys], S.nbParticles[sys]);
        for (PxgBufferHandle h : all) dev->destroyBuffer(h);

        // ---- reference ----
        std::vector<std::vector<F4>> refAccum(NB_SYS);
        for (uint32_t sys = 0; sys < NB_SYS; ++sys) refAccum[sys].assign(S.nbParticles[sys], F4{ 0, 0, 0, 0 });
        std::vector<F4> refDeltaVel(2u * N_ATT_TOTAL, F4{ SENTINEL, SENTINEL, SENTINEL, SENTINEL });
        uint32_t skipped = 0, wrote = 0, inert = 0;
        for (uint32_t k = 0; k < N_ATT_TOTAL; ++k)
        {
            const SolveRef r = cpuSolveLane(S, cb, k, v.isTGS, dt, biasCoefficient, v.velIter);
            if (!r.wrote) { ++inert; continue; }
            ++wrote;
            F4& acc = refAccum[r.sys][r.particleId];
            acc.x += r.deltaParticle.x; acc.y += r.deltaParticle.y; acc.z += r.deltaParticle.z;
            refDeltaVel[k] = r.lin;
            refDeltaVel[k + N_ATT_TOTAL] = r.ang;
        }
        // how many lanes took the infinite-mass `continue`
        for (uint32_t k = 0; k < N_ATT_TOTAL; ++k)
        {
            const uint32_t index = k / 32u, offset = k & 31u;
            uint64_t rid; std::memcpy(&rid, cb.data() + size_t(index) * PRC_STRIDE + PRC_RIGIDID + offset * 8u, 8);
            uint64_t cid; std::memcpy(&cid, cb.data() + size_t(index) * PRC_STRIDE + PRC_PARTICLEID + offset * 8u, 8);
            const bool st = uint32_t(rid & 0xffffffffull) == 0xffffffffu;
            const float im = v.isTGS ? S.sortedDeltaP[uint32_t(cid >> 32)][uint32_t(cid)].w
                                     : S.sortedPos[uint32_t(cid >> 32)][uint32_t(cid)].w;
            if (st && im == 0.0f) ++skipped;
        }

        uint32_t bad = 0, printed = 0;
        for (uint32_t k = 0; k < 2u * N_ATT_TOTAL; ++k)
        {
            if (std::memcmp(&gDeltaVel[k], &refDeltaVel[k], 16) == 0) continue;
            ++bad;
            if (printed++ < 6)
                std::printf("    deltaVel[%u] gpu(%.9g,%.9g,%.9g,%.9g) ref(%.9g,%.9g,%.9g,%.9g)\n", k,
                            gDeltaVel[k].x, gDeltaVel[k].y, gDeltaVel[k].z, gDeltaVel[k].w,
                            refDeltaVel[k].x, refDeltaVel[k].y, refDeltaVel[k].z, refDeltaVel[k].w);
        }
        uint32_t accBad = 0;
        for (uint32_t sys = 0; sys < NB_SYS; ++sys)
            for (uint32_t pp = 0; pp < S.nbParticles[sys]; ++pp)
                if (std::memcmp(&gAccum[sys][pp], &refAccum[sys][pp], 16) != 0)
                {
                    ++accBad;
                    if (printed++ < 8)
                        std::printf("    accumDeltaP[%u][%u] gpu(%.9g,%.9g,%.9g,%.9g) ref(%.9g,%.9g,%.9g,%.9g)\n",
                                    sys, pp, gAccum[sys][pp].x, gAccum[sys][pp].y, gAccum[sys][pp].z, gAccum[sys][pp].w,
                                    refAccum[sys][pp].x, refAccum[sys][pp].y, refAccum[sys][pp].z, refAccum[sys][pp].w);
                }

        std::printf("  %s velIter=%d: %u lanes -- %u wrote, %u inert (of which %u infinite-mass "
                    "skips)\n", v.entry, int(v.velIter), N_ATT_TOTAL, wrote, inert, skipped);
        PXG_CHECK_MSG(bad == 0, "%s velIter=%d: %u deltaVel words differ", v.entry, int(v.velIter), bad);
        PXG_CHECK_MSG(accBad == 0, "%s velIter=%d: %u accumDeltaP entries differ", v.entry, int(v.velIter), accBad);
        PXG_CHECK_MSG(wrote >= 30u, "only %u lanes produced an impulse -- the comparison has no teeth", wrote);
        PXG_CHECK_MSG(skipped >= 1u, "no lane took the infinite-mass `continue`, so that branch is untested");
    }

    // The TGS and PGS results must not be the same numbers.  This is a WEAK check --
    // P9.76c is the record of a difference test passing while the TGS arm ran the PGS
    // body -- so it stands next to the bit-exact comparison above, never instead of it.
    std::printf("  (both arms compared bit-exactly against their own transcription)\n");
}
