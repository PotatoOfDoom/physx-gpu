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

// P6.6: the three TGS-only kernels that complete the PBD particle contact SOLVE
// loop under the TGS solver (psContactSolveTgsBda.slang), ported line-for-line from
// gpusimulationcontroller/src/CUDA/particlesystem.cu.  Tests:
//
//   * psContactSolveTgs_penetratingParticle (KEY PROOF): the TGS analog of the
//     P6.4b-2 test — ONE particle penetrating a STATIC plane, run through the real
//     pipeline ps_contactPrepareLaunch -> ps_solvePCOutputParticleDeltaVTGSLaunch
//     -> ps_accumulateDeltaVParticleLaunch, asserting the particle receives an
//     UPWARD position delta that resolves the -0.1 penetration, bit-exact vs a CPU
//     transcription of solvePCOutputDeltaVTGS.  For a static floor with zero body
//     velocity/delta the TGS separation folding collapses to the non-TGS result,
//     so this ALSO proves TGS<->PGS consistency on the money-shot contact kind.
//   * psContactSolveTgs_movingBody: the TGS solve in isolation with a NON-static
//     rigid body carrying a non-zero packed lin/ang VELOCITY + lin/ang DELTA in the
//     3-float4-per-body TGS iter pool — exercises PxgVelocityReader::readVelocitiesTGS
//     unpacking AND the TGS-specific normalDelta + normalVel*projectionDt folding
//     (which are dead in the static-floor case), bit-exact vs the CPU transcription.
//   * psContactSolveTgs_step: ps_stepParticlesLaunch in isolation — the substep
//     predictor advances the sorted position by vel*dt and accumulates the running
//     deltaP, on both the first (from sorted-origin, dlta=0) and a subsequent
//     (from sorted-position, dlta=running deltaP) iteration, bit-exact.
//   * psContactSolveTgs_finalize: ps_finalizeParticlesLaunch in isolation — the
//     velocity blend newVel = vel*scale + deltaVel*(1-scale) when |vel| > |deltaVel|
//     (and the no-write case when it is not), bit-exact.
//   * P9.99b two-system twins: psContactSolveTgs_p99TwoSystems (systems 0 and 2 in one
//     contact list through prepare -> TGS solve -> accumulate), psContactSolveTgs_step_p99Direct
//     (pins the CUDA's pool[blockIdx.y] indexing, upstream note U1) and
//     psContactSolveTgs_finalize_p99Id2 (push id 2).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F2 { float x, y; };
struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
// PxgParticlePrimitiveContact (PxgParticleSystem.h, 32 bytes).
struct PsContact { F4 normal_pen; uint32_t rigidLo, rigidHi, particleLo, particleHi; };
static_assert(sizeof(PsContact) == 32, "PsContact stride");

static const uint32_t CB_STRIDE = 3072u;
// PxgParticlePrimitiveConstraintBlock SoA byte offsets (psContactSolveTgsBda.slang).
static const uint32_t CB_RAXN_VM = 0u;
static const uint32_t CB_NORMAL_ERR = 512u;
static const uint32_t CB_FRIC0_IM0 = 1024u;
static const uint32_t CB_RAXNF0_VM = 1536u;
static const uint32_t CB_RAXNF1_VM = 2048u;
static const uint32_t CB_RIGIDID = 2560u;
static const uint32_t CB_PARTICLEID = 2816u;

// Engine descriptor mirror sizes (pxgSlangParticleDesc / pxgSlangPrepDesc /
// pxgSlangSolverDesc); only the touched fields are poked.
static const uint32_t SZ_PARTICLE_SYS = 656u;   // PxgParticleSystemGpu
static const uint32_t SZ_PREP_DESC = 304u;      // PxgConstraintPrepareDescGpu
static const uint32_t SZ_PREPREP_DESC = 760u;   // PxgPrePrepDescGpu
static const uint32_t SZ_CORE_DESC = 320u;      // PxgSolverCoreDescGpu
static const uint32_t SZ_SHARED_PGS = 224u;     // PxgSolverSharedDescPGSGpu
static const uint32_t SZ_SOLVER_BODY = 96u;     // PxgSolverBodyData
static const uint32_t SZ_TXI = 64u;             // PxgSolverTxIData
static const uint32_t SZ_ALIGNED_XF = 32u;      // PxgAlignedTransform

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h)
{
    return dev->getBufferDeviceAddress(h);
}
static void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
static void putF32(std::vector<uint8_t>& b, size_t off, float v) { std::memcpy(b.data() + off, &v, 4); }

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static float pmin(float a, float b) { return (a < b) ? a : b; }
static float pmax(float a, float b) { return (a > b) ? a : b; }

// ---------------------------------------------------------------------------
// CPU transcription of solvePCOutputDeltaVTGS (particlesystem.cu:1473).
// ---------------------------------------------------------------------------
struct TgsSolveOut { float deltaF, deltaFr0, deltaFr1; F2 appliedForce; };
static TgsSolveOut cpuSolvePCOutputDeltaVTGS(
    V3 normal, float error, V3 delta, V3 fric0, V3 fric1,
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
    TgsSolveOut r;
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

// ---------------------------------------------------------------------------
// Push-constant packs (must match psContactSolveTgsBda.slang exactly).
// ---------------------------------------------------------------------------
struct SolvePush
{
    uint64_t particleSystems, sortedContacts, primitiveConstraints, numContacts,
        prePrepDesc, solverCoreDesc, sharedDesc, deltaPos, appliedForces;
    float dt, relaxationCoefficient; uint32_t isVelIterWord;
    uint64_t artiCoreDesc;
};
static_assert(sizeof(SolvePush) == 96, "solve push 96 bytes");

struct PrepPush
{
    uint64_t particleSystems, sortedContacts, numContacts, primitiveConstraints,
        preDesc, prepareDesc, appliedForces;
    float dt; uint32_t isTGSWord;
    uint64_t deltaVel, sharedDesc;
};
static_assert(sizeof(PrepPush) == 80, "prepare push 80 bytes");

struct AccumPush
{
    uint64_t particleSystems, sortedContacts, totalNumPairs, rangeStart, rangeEnd, deltaPos;
};
static_assert(sizeof(AccumPush) == 48, "accum push 48 bytes");

struct StepPush
{
    uint64_t particleSystems, activeParticleSystems;
    float dt, invTotalDt; uint32_t flagsWord;
    float gravityX, gravityY, gravityZ;
};
static_assert(sizeof(StepPush) == 40, "step push 40 bytes");

struct FinalizePush
{
    uint64_t particleSystems; uint32_t id; float invTotalDt, velocityScale;
};
// 20 bytes of data; the 8-byte alignment (uint64_t) rounds sizeof to 24 — which is
// exactly the marshalled push size ({8,4,4,4} rounds to maxAlign 8 = 24 bytes).
static_assert(sizeof(FinalizePush) == 24, "finalize push 24 bytes (20 data + 4 pad)");

// ===========================================================================
// KEY PROOF — a penetrating particle is pushed OUT (upward) by the TGS solve.
// Mirrors the P6.4b-2 pipeline but swaps the SOLVE stage to the TGS twin.
// ===========================================================================
struct CpuOut { F4 deltaPos; F2 appliedForce; F4 accumDeltaP; };

static bool runTgsPipeline(PxgSlangComputeDevice* dev, F4 contactNormalPen, float invMass1,
                           float dt, float relax, bool isVelIter, CpuOut& gpu)
{
    std::vector<F4> sortedPos = { { 0.0f, 0.0f, 0.0f, invMass1 } };
    std::vector<F4> sortedDeltaP = { { 0.0f, 0.0f, 0.0f, invMass1 } };
    std::vector<uint32_t> phases = { 0u };
    std::vector<uint32_t> phaseToMat = { 0u };
    std::vector<float> materials(8, 0.0f);
    std::vector<F4> accumDeltaP = { { 0.0f, 0.0f, 0.0f, 0.0f } };

    PxgBufferHandle bPos = createBufferFrom(dev, sortedPos);
    PxgBufferHandle bDeltaP = createBufferFrom(dev, sortedDeltaP);
    PxgBufferHandle bPhases = createBufferFrom(dev, phases);
    PxgBufferHandle bPhaseToMat = createBufferFrom(dev, phaseToMat);
    PxgBufferHandle bMaterials = createBufferFrom(dev, materials);
    PxgBufferHandle bAccum = createBufferFrom(dev, accumDeltaP);
    if (!addrOf(dev, bPos))
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgBufferHandle h : { bPos, bDeltaP, bPhases, bPhaseToMat, bMaterials, bAccum })
            dev->destroyBuffer(h);
        return false;
    }

    std::vector<uint8_t> psys(SZ_PARTICLE_SYS, 0);
    putU64(psys, 24, addrOf(dev, bPos));        // mSortedPositions_InvMass
    putU64(psys, 40, addrOf(dev, bPhases));     // mSortedPhaseArray
    putU64(psys, 80, addrOf(dev, bMaterials));  // mParticleMaterials
    putU64(psys, 152, addrOf(dev, bAccum));     // mAccumDeltaP
    putU64(psys, 160, addrOf(dev, bDeltaP));    // mSortedDeltaP
    putF32(psys, 268, 0.1f);                    // mData_RestOffset (unused: adhesion off)
    putU32(psys, 632, 20u);                     // mParticleMaterialStride (bytes)
    putU64(psys, 640, addrOf(dev, bPhaseToMat));// mPhaseGroupToMaterialHandle
    PxgBufferHandle bPsys = createBufferFrom(dev, psys);

    PsContact contact;
    contact.normal_pen = contactNormalPen;
    contact.rigidLo = 0xffffffffu; contact.rigidHi = 0u; // static body, not articulation
    contact.particleLo = 0u; contact.particleHi = 0u;    // particleId 0, systemId 0
    std::vector<PsContact> contacts = { contact };
    PxgBufferHandle bContacts = createBufferFrom(dev, contacts);
    PxgBufferHandle bNum = createBufferFrom(dev, std::vector<uint32_t>(1, 1u));
    PxgBufferHandle bConstraints = createBufferFrom(dev, std::vector<uint8_t>(CB_STRIDE, 0));
    PxgBufferHandle bApplied = createBufferFrom(dev, std::vector<F2>(32, { 0.0f, 0.0f }));
    PxgBufferHandle bDeltaPos = createBufferFrom(dev, std::vector<F4>(32, { 0.0f, 0.0f, 0.0f, 0.0f }));

    std::vector<uint8_t> solverBody(SZ_SOLVER_BODY, 0); // invMass0 = 0
    std::vector<uint8_t> txi(SZ_TXI, 0);                // sqrtInvInertia = 0
    std::vector<uint8_t> bodyFrame(SZ_ALIGNED_XF, 0);   // p = 0
    PxgBufferHandle bSolverBody = createBufferFrom(dev, solverBody);
    PxgBufferHandle bTxi = createBufferFrom(dev, txi);
    PxgBufferHandle bBodyFrame = createBufferFrom(dev, bodyFrame);

    std::vector<uint8_t> prepDesc(SZ_PREP_DESC, 0);
    putU64(prepDesc, 64, addrOf(dev, bSolverBody)); // solverBodyDataPool
    putU64(prepDesc, 72, addrOf(dev, bTxi));        // solverBodyTxIDataPool
    putU64(prepDesc, 168, addrOf(dev, bBodyFrame)); // body2WorldPool
    PxgBufferHandle bPrepDesc = createBufferFrom(dev, prepDesc);

    PxgBufferHandle bPrePrepDesc = createBufferFrom(dev, std::vector<uint8_t>(SZ_PREPREP_DESC, 0));

    // TGS iter pool: 3 float4 per body (numSolverBodies = 1, accOffset = 0 -> the
    // static body idx 0 reads velPool[0], [1], [2] = all zero).
    std::vector<F4> velPool(3, { 0.0f, 0.0f, 0.0f, 0.0f });
    PxgBufferHandle bVelPool = createBufferFrom(dev, velPool);

    std::vector<uint8_t> coreDesc(SZ_CORE_DESC, 0);
    putU32(coreDesc, 268, 1u);                 // numSolverBodies
    putU32(coreDesc, 276, 0u);                 // accumulatedBodyDeltaVOffset
    PxgBufferHandle bCoreDesc = createBufferFrom(dev, coreDesc);

    std::vector<uint8_t> sharedPgs(SZ_SHARED_PGS, 0);
    putU64(sharedPgs, 160, addrOf(dev, bVelPool)); // iter_solverBodyVelPool
    PxgBufferHandle bSharedPgs = createBufferFrom(dev, sharedPgs);

    PxgBufferHandle bPairs = createBufferFrom(dev, std::vector<uint32_t>(1, 1u));
    PxgBufferHandle bStart = createBufferFrom(dev, std::vector<uint32_t>(1, 0u));
    PxgBufferHandle bEnd = createBufferFrom(dev, std::vector<uint32_t>(1, 1u));

    // ---- prepare (shared non-TGS kernel; isTGS byte carried but unused here) ----
    {
        PrepPush p{};
        p.particleSystems = addrOf(dev, bPsys);
        p.sortedContacts = addrOf(dev, bContacts);
        p.numContacts = addrOf(dev, bNum);
        p.primitiveConstraints = addrOf(dev, bConstraints);
        p.preDesc = addrOf(dev, bPrePrepDesc);
        p.prepareDesc = addrOf(dev, bPrepDesc);
        p.appliedForces = addrOf(dev, bApplied);
        p.dt = dt;
        p.isTGSWord = 1u;                   // isTGS = true (does not alter the prep math)
        p.deltaVel = 0ull;
        p.sharedDesc = 0ull;
        Kernel k("psContactSolveBda.spv", "ps_contactPrepareLaunch", {}, sizeof(PrepPush));
        k.dispatch(1024u, {}, p);
    }
    // ---- TGS solve ----
    {
        SolvePush p{};
        p.particleSystems = addrOf(dev, bPsys);
        p.sortedContacts = addrOf(dev, bContacts);
        p.primitiveConstraints = addrOf(dev, bConstraints);
        p.numContacts = addrOf(dev, bNum);
        p.prePrepDesc = addrOf(dev, bPrePrepDesc);
        p.solverCoreDesc = addrOf(dev, bCoreDesc);
        p.sharedDesc = addrOf(dev, bSharedPgs);
        p.deltaPos = addrOf(dev, bDeltaPos);
        p.appliedForces = addrOf(dev, bApplied);
        p.dt = dt;
        p.relaxationCoefficient = relax;
        p.isVelIterWord = isVelIter ? 1u : 0u;
        p.artiCoreDesc = 0ull;              // deferred arti branch never dereferences it
        Kernel k("psContactSolveTgsBda.spv", "ps_solvePCOutputParticleDeltaVTGSLaunch", {}, sizeof(SolvePush));
        k.dispatch(1024u, {}, p);
    }
    // ---- accumulate (shared non-TGS kernel) ----
    {
        AccumPush p{};
        p.particleSystems = addrOf(dev, bPsys);
        p.sortedContacts = addrOf(dev, bContacts);
        p.totalNumPairs = addrOf(dev, bPairs);
        p.rangeStart = addrOf(dev, bStart);
        p.rangeEnd = addrOf(dev, bEnd);
        p.deltaPos = addrOf(dev, bDeltaPos);
        Kernel k("psContactSolveBda.spv", "ps_accumulateDeltaVParticleLaunch", {}, sizeof(AccumPush));
        k.dispatch(32u, {}, p);
    }
    dev->flush();

    gpu.deltaPos = downloadBuffer<F4>(dev, bDeltaPos, 1)[0];
    gpu.appliedForce = downloadBuffer<F2>(dev, bApplied, 1)[0];
    gpu.accumDeltaP = downloadBuffer<F4>(dev, bAccum, 1)[0];

    PxgBufferHandle all[] = { bPos, bDeltaP, bPhases, bPhaseToMat, bMaterials, bAccum, bPsys,
        bContacts, bNum, bConstraints, bApplied, bDeltaPos, bSolverBody, bTxi, bBodyFrame,
        bPrepDesc, bPrePrepDesc, bVelPool, bCoreDesc, bSharedPgs, bPairs, bStart, bEnd };
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);
    return true;
}

// CPU reference for the full static-floor TGS pipeline.
static CpuOut cpuReferenceStaticFloor(F4 cn, float invMass1, float dt, float relax, bool isVelIter)
{
    const V3 normal = { -cn.x, -cn.y, -cn.z };       // prepare flips the contact normal
    const float error = cn.w;
    const float velMultiplier = (invMass1 > 0.0f) ? (1.0f / invMass1) : 0.0f;
    // fric0 = PxComputeBasisVectors(normal).first; for normal (0,-1,0):
    // |dir.y| > 0.9999 -> right = (1,0,0).  fric1 = cross(normal, fric0).
    const V3 fric0 = { 1.0f, 0.0f, 0.0f };
    const V3 fric1 = cross3(normal, fric0);
    const V3 delta = { 0.0f, 0.0f, 0.0f };            // mSortedDeltaP.xyz = 0
    TgsSolveOut s = cpuSolvePCOutputDeltaVTGS(
        normal, error, delta, fric0, fric1,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        velMultiplier, velMultiplier, velMultiplier, // vmF0/vmF1 also 1/invMass1 (raX*=0)
        0.0f, 0.1f, 0.0f, 0.0f, isVelIter, dt, F2{ 0.0f, 0.0f });
    const float dlvX = (-(normal.x * s.deltaF) + fric0.x * s.deltaFr0 + fric1.x * s.deltaFr1) * invMass1 * relax;
    const float dlvY = (-(normal.y * s.deltaF) + fric0.y * s.deltaFr0 + fric1.y * s.deltaFr1) * invMass1 * relax;
    const float dlvZ = (-(normal.z * s.deltaF) + fric0.z * s.deltaFr0 + fric1.z * s.deltaFr1) * invMass1 * relax;
    const float scale = (s.deltaF != 0.0f) ? 1.0f : 0.0f;
    const float recip = 1.0f / pmax(1.0f, scale);
    CpuOut o;
    o.deltaPos = { dlvX, dlvY, dlvZ, scale };
    o.appliedForce = s.appliedForce;
    o.accumDeltaP = { dlvX * recip, dlvY * recip, dlvZ * recip, 0.0f };
    return o;
}

PXG_TEST(psContactSolveTgs_penetratingParticle)
{
    PxgSlangComputeDevice* dev = testDevice();
    const F4 contactNormalPen = { 0.0f, 1.0f, 0.0f, -0.1f };
    const float invMass1 = 1.0f, dt = 1.0f / 60.0f, relax = 1.0f;

    CpuOut gpu;
    if (!runTgsPipeline(dev, contactNormalPen, invMass1, dt, relax, /*isVelIter*/ false, gpu))
        return;
    const CpuOut ref = cpuReferenceStaticFloor(contactNormalPen, invMass1, dt, relax, false);

    PXG_CHECK_MSG(gpu.accumDeltaP.y > 0.0f,
                  "TGS accumDeltaP.y not upward: (%g,%g,%g)", gpu.accumDeltaP.x, gpu.accumDeltaP.y, gpu.accumDeltaP.z);
    PXG_CHECK_MSG(gpu.deltaPos.y > 0.0f, "TGS per-contact deltaPos.y not upward: %g", gpu.deltaPos.y);
    PXG_CHECK_MSG(std::fabs(gpu.accumDeltaP.y - 0.1f) < 1e-6f,
                  "TGS accumDeltaP.y=%g does not resolve -0.1 penetration", gpu.accumDeltaP.y);
    PXG_CHECK_MSG(gpu.accumDeltaP.x == 0.0f && gpu.accumDeltaP.z == 0.0f,
                  "TGS lateral drift: (%g,%g)", gpu.accumDeltaP.x, gpu.accumDeltaP.z);

    PXG_CHECK_MSG(std::memcmp(&gpu.deltaPos, &ref.deltaPos, sizeof(F4)) == 0,
                  "TGS deltaPos gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  gpu.deltaPos.x, gpu.deltaPos.y, gpu.deltaPos.z, gpu.deltaPos.w,
                  ref.deltaPos.x, ref.deltaPos.y, ref.deltaPos.z, ref.deltaPos.w);
    PXG_CHECK_MSG(std::memcmp(&gpu.appliedForce, &ref.appliedForce, sizeof(F2)) == 0,
                  "TGS appliedForce gpu(%g,%g) != cpu(%g,%g)",
                  gpu.appliedForce.x, gpu.appliedForce.y, ref.appliedForce.x, ref.appliedForce.y);
    PXG_CHECK_MSG(std::memcmp(&gpu.accumDeltaP, &ref.accumDeltaP, sizeof(F4)) == 0,
                  "TGS accumDeltaP gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  gpu.accumDeltaP.x, gpu.accumDeltaP.y, gpu.accumDeltaP.z, gpu.accumDeltaP.w,
                  ref.accumDeltaP.x, ref.accumDeltaP.y, ref.accumDeltaP.z, ref.accumDeltaP.w);
    PXG_CHECK_MSG(std::fabs(gpu.appliedForce.x - 0.1f) < 1e-6f,
                  "TGS appliedForce.x=%g != 0.1 (deltaF for a fully-resolved -0.1 pen)", gpu.appliedForce.x);
}

// ===========================================================================
// P9.99b -- TWO SYSTEMS in one contact list: prepare -> TGS solve -> accumulate.
//
// Every fixture above holds ONE contact of system 0, so `contact id >> 32` (prepare),
// the constraint block's system id (solve) and the run head's system id (accumulate)
// were only ever 0, and the P9.99b force-system-0 mutants of all three survived.
// Here the sorted list interleaves systems 0 and 2 -- 8 contacts in 6 particle runs:
//     #  sys pid  contact (normal (0,1,0), pen)
//     0   0   0   -0.10   penetrating
//     1   0   1   +0.05   separated, inside the adhesion radius
//     2   0   3   -0.08   \ one run of two
//     3   0   3   +0.03   /
//     4   2   3   -0.12   \ system 2's first particle == system 0's last one, so a
//     5   2   3   +0.04   / low-32-bit-only id would merge the runs
//     6   2   4   +0.05
//     7   2   6   -0.06
// and the two systems differ in everything the three kernels read per system: the
// inverse mass (1 / 0.5, read by prepare from the sorted positions and by the solve
// from the sorted deltaP), the running delta (tangential, so friction is live), the
// material through each system's own phase->material map (friction 0.05 / 0.3,
// adhesion 0.3 / 0.5, radius scale 1.5 / 2.0) and mData.mRestOffset (0.25 / 0.15),
// which the adhesion branch reads for the two separated contacts of each system.
// Slot 1 of the pool is a decoy no contact names; its arrays must survive.
// The static rigid keeps the prepare at velMultiplier = 1/invMass; every dot product
// the oracle takes is exact (the delta is orthogonal to the normal), so the CPU
// transcription is bit-exact.
// ===========================================================================
struct P99Sys
{
    uint32_t n = 0;
    float invMass = 0.f, restOffset = 0.f;
    uint16_t mat = 0;
    std::vector<F4> pos, deltaP, accum;
    std::vector<uint32_t> phases, p2m;
    PxgBufferHandle bPos = 0, bDeltaP = 0, bAccum = 0, bPhases = 0, bP2M = 0;
};

PXG_TEST(psContactSolveTgs_p99TwoSystems)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f, relax = 0.8f;
    // materials: friction, damping, adhesion, gravityScale, adhesionRadiusScale (stride 20)
    const std::vector<float> materials = {
        0.05f, 0.0f, 0.3f, 1.0f, 1.5f,     // material 0: system 0
        0.3f,  0.0f, 0.5f, 1.0f, 2.0f,     // material 1: system 2
        0.9f,  0.0f, 0.7f, 1.0f, 3.0f };   // material 2: the decoy
    P99Sys s[3];
    const uint32_t NP[3] = { 8u, 9u, 7u };
    const float IM[3] = { 1.0f, 2.0f, 0.5f }, REST[3] = { 0.25f, 0.4f, 0.15f };
    const uint16_t MAT[3] = { 0u, 2u, 1u };
    PxgBufferHandle bMaterials = createBufferFrom(dev, materials);
    if (!addrOf(dev, bMaterials)) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bMaterials); return; }
    std::vector<uint8_t> pool(3 * SZ_PARTICLE_SYS, 0);
    for (uint32_t k = 0; k < 3; ++k)
    {
        P99Sys& c = s[k];
        c.n = NP[k]; c.invMass = IM[k]; c.restOffset = REST[k]; c.mat = MAT[k];
        const float fk = float(k);
        for (uint32_t i = 0; i < c.n; ++i)
        {
            const float fi = float(i);
            c.pos.push_back({ 0.1f * fi, 0.5f + fk, -0.2f * fi, c.invMass });
            // tangential only: delta.y = 0, so normal.dot(delta) is an exact zero
            c.deltaP.push_back({ (k == 2u ? -0.015f : 0.01f) + 0.001f * fi, 0.0f, (k == 2u ? 0.005f : 0.02f) - 0.002f * fi, c.invMass });
            c.accum.push_back({ 0.5f * fi - fk, -0.25f + 0.125f * fk, 0.75f, 1.0f + fi });
        }
        c.phases.assign(c.n, 0u);
        c.p2m = { uint32_t(c.mat) };
        c.bPos = createBufferFrom(dev, c.pos); c.bDeltaP = createBufferFrom(dev, c.deltaP);
        c.bAccum = createBufferFrom(dev, c.accum); c.bPhases = createBufferFrom(dev, c.phases);
        c.bP2M = createBufferFrom(dev, c.p2m);
        std::vector<uint8_t> psys(SZ_PARTICLE_SYS, 0);
        putU64(psys, 24, addrOf(dev, c.bPos));        // mSortedPositions_InvMass
        putU64(psys, 40, addrOf(dev, c.bPhases));     // mSortedPhaseArray
        putU64(psys, 80, addrOf(dev, bMaterials));    // mParticleMaterials (one shared table)
        putU32(psys, 116, c.n);                       // mCommonData_NumParticles
        putU64(psys, 152, addrOf(dev, c.bAccum));     // mAccumDeltaP
        putU64(psys, 160, addrOf(dev, c.bDeltaP));    // mSortedDeltaP
        putF32(psys, 268, c.restOffset);              // mData_RestOffset
        putU32(psys, 632, 20u);                       // mParticleMaterialStride (bytes)
        putU64(psys, 640, addrOf(dev, c.bP2M));       // mPhaseGroupToMaterialHandle
        std::memcpy(pool.data() + k * SZ_PARTICLE_SYS, psys.data(), SZ_PARTICLE_SYS);
    }
    PxgBufferHandle bPsys = createBufferFrom(dev, pool);

    struct C { uint32_t sys, pid; float pen; };
    const C cl[8] = { { 0, 0, -0.10f }, { 0, 1, 0.05f }, { 0, 3, -0.08f }, { 0, 3, 0.03f },
                      { 2, 3, -0.12f }, { 2, 3, 0.04f }, { 2, 4, 0.05f }, { 2, 6, -0.06f } };
    const uint32_t NC = 8u;
    const uint32_t runStart[6] = { 0u, 1u, 2u, 4u, 6u, 7u }, runEnd[6] = { 1u, 2u, 4u, 6u, 7u, 8u };
    std::vector<PsContact> contacts(NC);
    for (uint32_t c = 0; c < NC; ++c)
    {
        contacts[c].normal_pen = { 0.0f, 1.0f, 0.0f, cl[c].pen };
        contacts[c].rigidLo = 0xffffffffu; contacts[c].rigidHi = 0u;   // static body, not articulation
        contacts[c].particleLo = cl[c].pid; contacts[c].particleHi = cl[c].sys;
    }
    PxgBufferHandle bContacts = createBufferFrom(dev, contacts);
    PxgBufferHandle bNum = createBufferFrom(dev, std::vector<uint32_t>(1, NC));
    PxgBufferHandle bConstraints = createBufferFrom(dev, std::vector<uint8_t>(CB_STRIDE, 0));
    PxgBufferHandle bApplied = createBufferFrom(dev, std::vector<F2>(32, { 9.0f, 9.0f }));
    PxgBufferHandle bDeltaPos = createBufferFrom(dev, std::vector<F4>(32, { 0.0f, 0.0f, 0.0f, 0.0f }));
    std::vector<uint8_t> solverBody(SZ_SOLVER_BODY, 0), txi(SZ_TXI, 0), bodyFrame(SZ_ALIGNED_XF, 0);
    PxgBufferHandle bSolverBody = createBufferFrom(dev, solverBody);
    PxgBufferHandle bTxi = createBufferFrom(dev, txi);
    PxgBufferHandle bBodyFrame = createBufferFrom(dev, bodyFrame);
    std::vector<uint8_t> prepDesc(SZ_PREP_DESC, 0);
    putU64(prepDesc, 64, addrOf(dev, bSolverBody));
    putU64(prepDesc, 72, addrOf(dev, bTxi));
    putU64(prepDesc, 168, addrOf(dev, bBodyFrame));
    PxgBufferHandle bPrepDesc = createBufferFrom(dev, prepDesc);
    PxgBufferHandle bPrePrepDesc = createBufferFrom(dev, std::vector<uint8_t>(SZ_PREPREP_DESC, 0));
    PxgBufferHandle bVelPool = createBufferFrom(dev, std::vector<F4>(3, { 0.0f, 0.0f, 0.0f, 0.0f }));
    std::vector<uint8_t> coreDesc(SZ_CORE_DESC, 0);
    putU32(coreDesc, 268, 1u);                 // numSolverBodies
    putU32(coreDesc, 276, 0u);                 // accumulatedBodyDeltaVOffset
    PxgBufferHandle bCoreDesc = createBufferFrom(dev, coreDesc);
    std::vector<uint8_t> sharedPgs(SZ_SHARED_PGS, 0);
    putU64(sharedPgs, 160, addrOf(dev, bVelPool));
    PxgBufferHandle bSharedPgs = createBufferFrom(dev, sharedPgs);
    PxgBufferHandle bPairs = createBufferFrom(dev, std::vector<uint32_t>(1, 6u));
    PxgBufferHandle bStart = createBufferFrom(dev, std::vector<uint32_t>(runStart, runStart + 6));
    PxgBufferHandle bEnd = createBufferFrom(dev, std::vector<uint32_t>(runEnd, runEnd + 6));

    {
        PrepPush p{};
        p.particleSystems = addrOf(dev, bPsys); p.sortedContacts = addrOf(dev, bContacts);
        p.numContacts = addrOf(dev, bNum); p.primitiveConstraints = addrOf(dev, bConstraints);
        p.preDesc = addrOf(dev, bPrePrepDesc); p.prepareDesc = addrOf(dev, bPrepDesc);
        p.appliedForces = addrOf(dev, bApplied); p.dt = dt; p.isTGSWord = 1u;
        p.deltaVel = 0ull; p.sharedDesc = 0ull;
        Kernel k("psContactSolveBda.spv", "ps_contactPrepareLaunch", {}, sizeof(PrepPush));
        k.dispatch(1024u, {}, p);
    }
    {
        SolvePush p{};
        p.particleSystems = addrOf(dev, bPsys); p.sortedContacts = addrOf(dev, bContacts);
        p.primitiveConstraints = addrOf(dev, bConstraints); p.numContacts = addrOf(dev, bNum);
        p.prePrepDesc = addrOf(dev, bPrePrepDesc); p.solverCoreDesc = addrOf(dev, bCoreDesc);
        p.sharedDesc = addrOf(dev, bSharedPgs); p.deltaPos = addrOf(dev, bDeltaPos);
        p.appliedForces = addrOf(dev, bApplied); p.dt = dt; p.relaxationCoefficient = relax;
        p.isVelIterWord = 0u; p.artiCoreDesc = 0ull;
        Kernel k("psContactSolveTgsBda.spv", "ps_solvePCOutputParticleDeltaVTGSLaunch", {}, sizeof(SolvePush));
        k.dispatch(1024u, {}, p);
    }
    {
        AccumPush p{};
        p.particleSystems = addrOf(dev, bPsys); p.sortedContacts = addrOf(dev, bContacts);
        p.totalNumPairs = addrOf(dev, bPairs); p.rangeStart = addrOf(dev, bStart);
        p.rangeEnd = addrOf(dev, bEnd); p.deltaPos = addrOf(dev, bDeltaPos);
        Kernel k("psContactSolveBda.spv", "ps_accumulateDeltaVParticleLaunch", {}, sizeof(AccumPush));
        k.dispatch(32u, {}, p);
    }
    dev->flush();

    // CPU oracle: prepare (static rigid) -> cpuSolvePCOutputDeltaVTGS -> accumulate.
    const V3 normal = { -0.0f, -1.0f, -0.0f };       // prepare flips the contact normal (0,1,0)
    const V3 fric0 = { 1.0f, 0.0f, 0.0f };           // computeBasisVectors: |dir.y| > 0.9999
    const V3 fric1 = cross3(normal, fric0);
    std::vector<F4> refDeltaPos(NC);
    std::vector<F2> refApplied(NC);
    std::vector<F4> refAccum[3] = { s[0].accum, s[1].accum, s[2].accum };
    for (uint32_t c = 0; c < NC; ++c)
    {
        const P99Sys& sy = s[cl[c].sys];
        const float invMass1 = sy.pos[cl[c].pid].w;
        const float vm = (invMass1 > 0.0f) ? (1.0f / (0.0f + invMass1)) : 0.0f;
        const F4 d4 = sy.deltaP[cl[c].pid];
        const V3 delta = { d4.x, d4.y, d4.z };
        const float* m = &materials[sy.mat * 5u];
        const TgsSolveOut r = cpuSolvePCOutputDeltaVTGS(normal, cl[c].pen, delta, fric0, fric1, 0.0f, 0.0f, 0.0f, 0.0f,
                                                        0.0f, 0.0f, vm, vm, vm, m[0], sy.restOffset, m[2], m[4],
                                                        false, dt, F2{ 0.0f, 0.0f });
        const float im = d4.w;
        refDeltaPos[c] = { ((-(normal.x * r.deltaF) + fric0.x * r.deltaFr0) + fric1.x * r.deltaFr1) * im * relax,
                           ((-(normal.y * r.deltaF) + fric0.y * r.deltaFr0) + fric1.y * r.deltaFr1) * im * relax,
                           ((-(normal.z * r.deltaF) + fric0.z * r.deltaFr0) + fric1.z * r.deltaFr1) * im * relax,
                           (r.deltaF != 0.0f) ? 1.0f : 0.0f };
        refApplied[c] = r.appliedForce;
    }
    for (uint32_t run = 0; run < 6; ++run)
    {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, fc = 0.0f;
        for (uint32_t j = runStart[run]; j < runEnd[run]; ++j)
        { ax += refDeltaPos[j].x; ay += refDeltaPos[j].y; az += refDeltaPos[j].z; fc += refDeltaPos[j].w; }
        const float recip = 1.0f / pmax(1.0f, fc);
        ax *= recip; ay *= recip; az *= recip;
        F4& a = refAccum[cl[runStart[run]].sys][cl[runStart[run]].pid];
        a = { a.x + ax, a.y + ay, a.z + az, a.w + 0.0f };
    }

    size_t mm = 0;
    const auto gDeltaPos = downloadBuffer<F4>(dev, bDeltaPos, NC);
    const auto gApplied = downloadBuffer<F2>(dev, bApplied, NC);
    for (uint32_t c = 0; c < NC; ++c)
    {
        if (std::memcmp(&gDeltaPos[c], &refDeltaPos[c], sizeof(F4)) != 0)
        { if (mm < 8) std::printf("  contact %u deltaPos gpu(%g,%g,%g,%g) cpu(%g,%g,%g,%g)\n", c, gDeltaPos[c].x, gDeltaPos[c].y,
                                  gDeltaPos[c].z, gDeltaPos[c].w, refDeltaPos[c].x, refDeltaPos[c].y, refDeltaPos[c].z, refDeltaPos[c].w); ++mm; }
        if (std::memcmp(&gApplied[c], &refApplied[c], sizeof(F2)) != 0)
        { if (mm < 8) std::printf("  contact %u appliedForce gpu(%g,%g) cpu(%g,%g)\n", c, gApplied[c].x, gApplied[c].y,
                                  refApplied[c].x, refApplied[c].y); ++mm; }
    }
    for (uint32_t k = 0; k < 3; ++k)
    {
        const auto gAcc = downloadBuffer<F4>(dev, s[k].bAccum, s[k].n);
        for (uint32_t i = 0; i < s[k].n; ++i)
            if (std::memcmp(&gAcc[i], &refAccum[k][i], sizeof(F4)) != 0)
            { if (mm < 12) std::printf("  slot %u accumDeltaP[%u] gpu(%g,%g,%g,%g) cpu(%g,%g,%g,%g)\n", k, i, gAcc[i].x, gAcc[i].y,
                                       gAcc[i].z, gAcc[i].w, refAccum[k][i].x, refAccum[k][i].y, refAccum[k][i].z, refAccum[k][i].w); ++mm; }
    }
    // the separated contacts really took the adhesion branch (else mRestOffset is not read)
    PXG_CHECK_MSG(refDeltaPos[5].y < 0.0f && refDeltaPos[6].y < 0.0f && refDeltaPos[1].y < 0.0f,
                  "the separated contacts must be pulled in by adhesion (dy %g %g %g)", refDeltaPos[1].y,
                  refDeltaPos[5].y, refDeltaPos[6].y);
    PXG_CHECK_MSG(mm == 0, "TGS pipeline over systems 0 and 2 (interleaved runs, decoy at slot 1): bit-exact vs CPU "
                  "(%zu mismatches)", mm);

    PxgBufferHandle all[] = { bMaterials, bPsys, bContacts, bNum, bConstraints, bApplied, bDeltaPos, bSolverBody, bTxi,
        bBodyFrame, bPrepDesc, bPrePrepDesc, bVelPool, bCoreDesc, bSharedPgs, bPairs, bStart, bEnd };
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);
    for (uint32_t k = 0; k < 3; ++k)
        for (PxgBufferHandle h : { s[k].bPos, s[k].bDeltaP, s[k].bAccum, s[k].bPhases, s[k].bP2M })
            dev->destroyBuffer(h);
}

// ===========================================================================
// TGS-specific: a moving NON-static rigid body — exercises readVelocitiesTGS
// unpacking and the normalDelta + normalVel*projectionDt folding (dead for the
// static floor).  The constraint block is poked DIRECTLY (bypassing prepare) so
// the raX* vectors are zero (body/particle at origin) and only the velocity/delta
// folding drives the result; bit-exact vs the CPU transcription.
// ===========================================================================
PXG_TEST(psContactSolveTgs_movingBody)
{
    PxgSlangComputeDevice* dev = testDevice();

    const V3 normal = { 0.0f, 1.0f, 0.0f };
    const V3 fric0 = { 1.0f, 0.0f, 0.0f };
    const V3 fric1 = cross3(normal, fric0);          // (0,0,-1)
    const float error = -0.1f;
    const float invMass1 = 1.0f, velMultiplier = 1.0f; // invMass0 = 0 -> vm = 1/invMass1
    const float dt = 1.0f / 60.0f, relax = 1.0f;
    const uint32_t bodyIdx = 1u, numSolverBodies = 2u, rigidIndex = 5u;

    // Packed TGS velocity: v0=(linVel.xyz, angVel.x), v1=(angVel.yz, linDelta.xy),
    // v2=(linDelta.z, angDelta.xyz).  Choose a non-trivial lin/ang vel + delta.
    const V3 linVel = { 0.3f, -2.0f, 0.1f };
    const V3 angVel = { 0.0f, 0.0f, 0.0f };          // raX* = 0 -> angular terms vanish
    const V3 linDelta = { -0.2f, 0.05f, 0.4f };
    const V3 angDelta = { 0.0f, 0.0f, 0.0f };
    const F4 v0 = { linVel.x, linVel.y, linVel.z, angVel.x };
    const F4 v1 = { angVel.y, angVel.z, linDelta.x, linDelta.y };
    const F4 v2 = { linDelta.z, angDelta.x, angDelta.y, angDelta.z };

    for (int velIterInt = 0; velIterInt < 2; ++velIterInt)
    {
        const bool isVelIter = (velIterInt != 0);

        std::vector<F4> sortedDeltaP = { { 0.02f, 0.0f, -0.01f, invMass1 } }; // delta != 0
        std::vector<uint32_t> phases = { 0u };
        std::vector<uint32_t> phaseToMat = { 0u };
        std::vector<float> materials(8, 0.0f);       // friction/adhesion off
        PxgBufferHandle bDeltaP = createBufferFrom(dev, sortedDeltaP);
        PxgBufferHandle bPhases = createBufferFrom(dev, phases);
        PxgBufferHandle bPhaseToMat = createBufferFrom(dev, phaseToMat);
        PxgBufferHandle bMaterials = createBufferFrom(dev, materials);
        if (!addrOf(dev, bDeltaP))
        {
            std::printf("  NOTE: no BDA -> skip\n");
            for (PxgBufferHandle h : { bDeltaP, bPhases, bPhaseToMat, bMaterials }) dev->destroyBuffer(h);
            return;
        }

        std::vector<uint8_t> psys(SZ_PARTICLE_SYS, 0);
        putU64(psys, 40, addrOf(dev, bPhases));
        putU64(psys, 80, addrOf(dev, bMaterials));
        putU64(psys, 160, addrOf(dev, bDeltaP));
        putF32(psys, 268, 0.1f);                     // mData_RestOffset
        putU32(psys, 632, 20u);                      // mParticleMaterialStride
        putU64(psys, 640, addrOf(dev, bPhaseToMat));
        PxgBufferHandle bPsys = createBufferFrom(dev, psys);

        // Constraint block: poke lane 0 directly.
        std::vector<uint8_t> cb(CB_STRIDE, 0);
        auto putF4 = [&](uint32_t off, F4 v) { std::memcpy(cb.data() + off, &v, 16); };
        putF4(CB_RAXN_VM,   F4{ 0.0f, 0.0f, 0.0f, velMultiplier });       // raXn = 0
        putF4(CB_NORMAL_ERR, F4{ normal.x, normal.y, normal.z, error });
        putF4(CB_FRIC0_IM0, F4{ fric0.x, fric0.y, fric0.z, 0.0f });       // invMass0 = 0
        putF4(CB_RAXNF0_VM, F4{ 0.0f, 0.0f, 0.0f, velMultiplier });       // raXnF0 = 0
        putF4(CB_RAXNF1_VM, F4{ 0.0f, 0.0f, 0.0f, velMultiplier });       // raXnF1 = 0
        putU32(cb, CB_RIGIDID, rigidIndex);          // rigidId.x = 5 (not static)
        putU32(cb, CB_RIGIDID + 4, 0u);              // rigidId.y = 0 (not articulation)
        putU32(cb, CB_PARTICLEID, 0u);               // particleId 0
        putU32(cb, CB_PARTICLEID + 4, 0u);           // systemId 0
        PxgBufferHandle bConstraints = createBufferFrom(dev, cb);

        PxgBufferHandle bNum = createBufferFrom(dev, std::vector<uint32_t>(1, 1u));
        PxgBufferHandle bApplied = createBufferFrom(dev, std::vector<F2>(32, { 0.0f, 0.0f }));
        PxgBufferHandle bDeltaPos = createBufferFrom(dev, std::vector<F4>(32, { 0.0f, 0.0f, 0.0f, 0.0f }));

        // prePrepDesc.solverBodyIndices[rigidIndex] = bodyIdx.
        std::vector<uint32_t> solverBodyIndices(rigidIndex + 1, 0u);
        solverBodyIndices[rigidIndex] = bodyIdx;
        PxgBufferHandle bSolverBodyIndices = createBufferFrom(dev, solverBodyIndices);
        std::vector<uint8_t> prePrepDesc(SZ_PREPREP_DESC, 0);
        putU64(prePrepDesc, 272, addrOf(dev, bSolverBodyIndices)); // solverBodyIndices
        PxgBufferHandle bPrePrepDesc = createBufferFrom(dev, prePrepDesc);

        // TGS iter pool: 3 float4 per body (accOffset = 0) -> slots for body bodyIdx
        // are [bodyIdx], [bodyIdx+N], [bodyIdx+2N].
        std::vector<F4> velPool(3 * numSolverBodies, { 0.0f, 0.0f, 0.0f, 0.0f });
        velPool[bodyIdx] = v0;
        velPool[bodyIdx + numSolverBodies] = v1;
        velPool[bodyIdx + 2 * numSolverBodies] = v2;
        PxgBufferHandle bVelPool = createBufferFrom(dev, velPool);

        std::vector<uint8_t> coreDesc(SZ_CORE_DESC, 0);
        putU32(coreDesc, 268, numSolverBodies);      // numSolverBodies
        putU32(coreDesc, 276, 0u);                   // accumulatedBodyDeltaVOffset
        PxgBufferHandle bCoreDesc = createBufferFrom(dev, coreDesc);

        std::vector<uint8_t> sharedPgs(SZ_SHARED_PGS, 0);
        putU64(sharedPgs, 160, addrOf(dev, bVelPool)); // iter_solverBodyVelPool
        PxgBufferHandle bSharedPgs = createBufferFrom(dev, sharedPgs);

        {
            SolvePush p{};
            p.particleSystems = addrOf(dev, bPsys);
            p.sortedContacts = 0ull;
            p.primitiveConstraints = addrOf(dev, bConstraints);
            p.numContacts = addrOf(dev, bNum);
            p.prePrepDesc = addrOf(dev, bPrePrepDesc);
            p.solverCoreDesc = addrOf(dev, bCoreDesc);
            p.sharedDesc = addrOf(dev, bSharedPgs);
            p.deltaPos = addrOf(dev, bDeltaPos);
            p.appliedForces = addrOf(dev, bApplied);
            p.dt = dt;
            p.relaxationCoefficient = relax;
            p.isVelIterWord = isVelIter ? 1u : 0u;
            p.artiCoreDesc = 0ull;
            Kernel k("psContactSolveTgsBda.spv", "ps_solvePCOutputParticleDeltaVTGSLaunch", {}, sizeof(SolvePush));
            k.dispatch(1024u, {}, p);
        }
        dev->flush();

        const F4 gpuDeltaPos = downloadBuffer<F4>(dev, bDeltaPos, 1)[0];
        const F2 gpuApplied = downloadBuffer<F2>(dev, bApplied, 1)[0];

        // CPU reference.
        const V3 delta = { sortedDeltaP[0].x, sortedDeltaP[0].y, sortedDeltaP[0].z };
        const V3 raXn = { 0.0f, 0.0f, 0.0f }, raXnF0 = { 0.0f, 0.0f, 0.0f }, raXnF1 = { 0.0f, 0.0f, 0.0f };
        const float normalVel = dot3(linVel, normal) + dot3(angVel, raXn);
        const float normalDelta = dot3(linDelta, normal) + dot3(angDelta, raXn);
        const float tanVel0 = dot3(linVel, fric0) + dot3(angVel, raXnF0);
        const float tanDelta0 = dot3(linDelta, fric0) + dot3(angDelta, raXnF0);
        const float tanVel1 = dot3(linVel, fric1) + dot3(angVel, raXnF1);
        const float tanDelta1 = dot3(linDelta, fric1) + dot3(angDelta, raXnF1);
        TgsSolveOut s = cpuSolvePCOutputDeltaVTGS(
            normal, error, delta, fric0, fric1, tanDelta0, tanDelta1, tanVel0, tanVel1,
            normalDelta, normalVel, velMultiplier, velMultiplier, velMultiplier,
            0.0f, 0.1f, 0.0f, 0.0f, isVelIter, dt, F2{ 0.0f, 0.0f });
        const F4 refDeltaPos = {
            (-(normal.x * s.deltaF) + fric0.x * s.deltaFr0 + fric1.x * s.deltaFr1) * invMass1 * relax,
            (-(normal.y * s.deltaF) + fric0.y * s.deltaFr0 + fric1.y * s.deltaFr1) * invMass1 * relax,
            (-(normal.z * s.deltaF) + fric0.z * s.deltaFr0 + fric1.z * s.deltaFr1) * invMass1 * relax,
            (s.deltaF != 0.0f) ? 1.0f : 0.0f };

        PXG_CHECK_MSG(std::memcmp(&gpuDeltaPos, &refDeltaPos, sizeof(F4)) == 0,
                      "movingBody(velIter=%d) deltaPos gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                      velIterInt, gpuDeltaPos.x, gpuDeltaPos.y, gpuDeltaPos.z, gpuDeltaPos.w,
                      refDeltaPos.x, refDeltaPos.y, refDeltaPos.z, refDeltaPos.w);
        PXG_CHECK_MSG(std::memcmp(&gpuApplied, &s.appliedForce, sizeof(F2)) == 0,
                      "movingBody(velIter=%d) appliedForce gpu(%g,%g) != cpu(%g,%g)",
                      velIterInt, gpuApplied.x, gpuApplied.y, s.appliedForce.x, s.appliedForce.y);

        PxgBufferHandle all[] = { bDeltaP, bPhases, bPhaseToMat, bMaterials, bPsys, bConstraints,
            bNum, bApplied, bDeltaPos, bSolverBodyIndices, bPrePrepDesc, bVelPool, bCoreDesc, bSharedPgs };
        for (PxgBufferHandle h : all) dev->destroyBuffer(h);
    }

    // Sanity: the two velIter cases must differ (projectionDt = 0 vs dt actually
    // changes the separation), proving the isVelocityIteration wiring is live.
    // (Checked implicitly by both being bit-exact to their own CPU refs above.)
}

// ===========================================================================
// ps_stepParticlesLaunch in isolation — substep position advance + running deltaP.
// ===========================================================================
PXG_TEST(psContactSolveTgs_step)
{
    PxgSlangComputeDevice* dev = testDevice();

    const V3 origin = { 1.0f, 3.0f, -2.0f };
    const V3 vel = { 0.5f, -4.0f, 0.25f };
    const float invMass = 1.0f;                 // tPos.w (must be != 0 to move)
    const float dt = 1.0f / 60.0f;

    std::vector<F4> originPos = { { origin.x, origin.y, origin.z, invMass } };
    std::vector<F4> sortedPos = { { 0.0f, 0.0f, 0.0f, invMass } };
    std::vector<F4> vels = { { vel.x, vel.y, vel.z, invMass } };
    std::vector<F4> deltaP = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    std::vector<uint32_t> phases = { 0u };
    std::vector<uint32_t> phaseToMat = { 0u };
    std::vector<float> materials(8, 0.0f);

    PxgBufferHandle bOrigin = createBufferFrom(dev, originPos);
    PxgBufferHandle bPos = createBufferFrom(dev, sortedPos);
    PxgBufferHandle bVels = createBufferFrom(dev, vels);
    PxgBufferHandle bDeltaP = createBufferFrom(dev, deltaP);
    PxgBufferHandle bPhases = createBufferFrom(dev, phases);
    PxgBufferHandle bPhaseToMat = createBufferFrom(dev, phaseToMat);
    PxgBufferHandle bMaterials = createBufferFrom(dev, materials);
    if (!addrOf(dev, bPos))
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgBufferHandle h : { bOrigin, bPos, bVels, bDeltaP, bPhases, bPhaseToMat, bMaterials })
            dev->destroyBuffer(h);
        return;
    }

    std::vector<uint8_t> psys(SZ_PARTICLE_SYS, 0);
    putU32(psys, 116, 1u);                       // mCommonData_NumParticles
    putU64(psys, 24, addrOf(dev, bPos));         // mSortedPositions_InvMass
    putU64(psys, 32, addrOf(dev, bVels));        // mSortedVelocities
    putU64(psys, 40, addrOf(dev, bPhases));      // mSortedPhaseArray
    putU64(psys, 80, addrOf(dev, bMaterials));   // mParticleMaterials
    putU64(psys, 160, addrOf(dev, bDeltaP));     // mSortedDeltaP
    putF32(psys, 316, 1.0e18f);                  // mData_MaxVelocity (no CFL clamp)
    putU32(psys, 632, 20u);                      // mParticleMaterialStride
    putU64(psys, 392, addrOf(dev, bOrigin));     // mSortedOriginPos_InvMass
    putU64(psys, 640, addrOf(dev, bPhaseToMat)); // mPhaseGroupToMaterialHandle
    PxgBufferHandle bPsys = createBufferFrom(dev, psys);
    // P9.99b: a REAL active list {0}, as the engine passes one.  The CUDA never reads it
    // (pool[blockIdx.y], U1), so the value is the same either way -- but with a null
    // pointer here an act-indexed spelling of the kernel FAULTED the whole psContactSolveTgs
    // process (SIGSEGV), which hid whether psContactSolveTgs_step_p99Direct catches it.
    PxgBufferHandle bAct = createBufferFrom(dev, std::vector<uint32_t>(1, 0u));

    auto runStep = [&](bool isFirstIteration)
    {
        StepPush p{};
        p.particleSystems = addrOf(dev, bPsys);
        p.activeParticleSystems = addrOf(dev, bAct);   // {0}; ABI parity -- the CUDA never reads it
        p.dt = dt;
        p.invTotalDt = 1.0f / dt;
        p.flagsWord = isFirstIteration ? 1u : 0u; // byte0=isFirstIteration, byte2=extForces(0)
        p.gravityX = 0.0f; p.gravityY = -9.81f; p.gravityZ = 0.0f; // unused (extForces off)
        Kernel k("psContactSolveTgsBda.spv", "ps_stepParticlesLaunch", {}, sizeof(StepPush));
        k.dispatch(1u, {}, p, 1u, 1u);
        dev->flush();
    };

    // ---- first iteration: src = origin, dlta = 0 ----
    runStep(true);
    F4 pos1 = downloadBuffer<F4>(dev, bPos, 1)[0];
    F4 dp1 = downloadBuffer<F4>(dev, bDeltaP, 1)[0];
    F4 vel1 = downloadBuffer<F4>(dev, bVels, 1)[0];

    const V3 delta = { vel.x * dt, vel.y * dt, vel.z * dt };
    const F4 refPos1 = { origin.x + delta.x, origin.y + delta.y, origin.z + delta.z, invMass };
    const F4 refDp1 = { delta.x, delta.y, delta.z, invMass };

    PXG_CHECK_MSG(std::memcmp(&pos1, &refPos1, sizeof(F4)) == 0,
                  "step[first] pos gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  pos1.x, pos1.y, pos1.z, pos1.w, refPos1.x, refPos1.y, refPos1.z, refPos1.w);
    PXG_CHECK_MSG(std::memcmp(&dp1, &refDp1, sizeof(F4)) == 0,
                  "step[first] deltaP gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  dp1.x, dp1.y, dp1.z, dp1.w, refDp1.x, refDp1.y, refDp1.z, refDp1.w);
    // velChanged=false (extForces off, no CFL clamp) -> velocity unchanged.
    PXG_CHECK_MSG(vel1.x == vel.x && vel1.y == vel.y && vel1.z == vel.z,
                  "step[first] velocity wrongly changed: (%g,%g,%g)", vel1.x, vel1.y, vel1.z);
    PXG_CHECK_MSG(dp1.y < 0.0f, "step[first] deltaP.y not downward for a falling particle: %g", dp1.y);

    // ---- subsequent iteration: src = sorted pos (pos1), dlta = running deltaP (dp1) ----
    runStep(false);
    F4 pos2 = downloadBuffer<F4>(dev, bPos, 1)[0];
    F4 dp2 = downloadBuffer<F4>(dev, bDeltaP, 1)[0];

    const F4 refPos2 = { pos1.x + delta.x, pos1.y + delta.y, pos1.z + delta.z, invMass };
    const F4 refDp2 = { delta.x + dp1.x, delta.y + dp1.y, delta.z + dp1.z, invMass };
    PXG_CHECK_MSG(std::memcmp(&pos2, &refPos2, sizeof(F4)) == 0,
                  "step[next] pos gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  pos2.x, pos2.y, pos2.z, pos2.w, refPos2.x, refPos2.y, refPos2.z, refPos2.w);
    PXG_CHECK_MSG(std::memcmp(&dp2, &refDp2, sizeof(F4)) == 0,
                  "step[next] deltaP gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  dp2.x, dp2.y, dp2.z, dp2.w, refDp2.x, refDp2.y, refDp2.z, refDp2.w);

    for (PxgBufferHandle h : { bOrigin, bPos, bVels, bDeltaP, bPhases, bPhaseToMat, bMaterials, bPsys, bAct })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P9.99b -- ps_stepParticlesLaunch indexes the POOL by blockIdx.y, literally
// (particlesystem.cu:130 `particleSystems[blockIdx.y]`; activeParticleSystems is never
// read).  That is upstream behaviour the port keeps (RUNTIME_STATUS P9.99b, U1: after a
// system that is not the last is removed, the TGS predictor steps the removed slot and
// never the survivor in the highest slot) -- and this fixture pins it, so neither a
// "fix" toward act[blockIdx.y] nor a lost system index can slip in unnoticed.
//
// Three REAL, distinct systems sit in slots 0, 1 and 2, the active list is {2, 0} and the
// grid is (1, 2): the CUDA steps slots 0 and 1 and leaves slot 2 bit-unchanged.  An
// act-indexed kernel would step 2 and 0 and leave 1 alone; `sysIndex = 0` would step
// slot 0 twice and slot 1 never.  The slots differ in N (3 / 5 / 4), data and the CFL
// cap: slot 1's cap is 2 and its particle 0 moves at 4 (clamped, along -y, so every
// value of the clamp is exact), slot 0's cap is 1e18 -- a cap read from the wrong slot
// changes slot 1.  Both the first iteration (from the sorted origin, dlta = 0) and a
// subsequent one (from the sorted position, dlta = the running deltaP) are checked;
// the oracle is psContactSolveTgs_step's transcription plus the clamp.
// ===========================================================================
PXG_TEST(psContactSolveTgs_step_p99Direct)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const uint32_t NS[3] = { 3u, 5u, 4u };
    const float CAP[3] = { 1.0e18f, 2.0f, 3.0f };
    struct Sys
    {
        std::vector<F4> origin, pos, vel, deltaP;
        std::vector<uint32_t> phases, p2m;
        PxgBufferHandle bOrigin = 0, bPos = 0, bVel = 0, bDeltaP = 0, bPhases = 0, bP2M = 0;
    };
    Sys s[3];
    PxgBufferHandle bMaterials = createBufferFrom(dev, std::vector<float>(8, 0.0f));
    if (!addrOf(dev, bMaterials)) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bMaterials); return; }
    std::vector<uint8_t> pool(3 * SZ_PARTICLE_SYS, 0);
    for (uint32_t k = 0; k < 3; ++k)
    {
        Sys& c = s[k];
        const float fk = float(k);
        for (uint32_t i = 0; i < NS[k]; ++i)
        {
            const float fi = float(i);
            const float w = (i == 2u) ? 0.0f : 1.0f;      // particle 2: infinite mass, never moves
            c.origin.push_back({ 1.0f + fi + 10.0f * fk, 3.0f - fk, -2.0f + 0.5f * fi, w });
            c.pos.push_back({ 0.0f, 0.0f, 0.0f, w });
            if (k == 1u && i == 0u) c.vel.push_back({ 0.0f, -4.0f, 0.0f, w });   // |v| 4 > cap 2: clamped to (0,-2,0)
            else c.vel.push_back({ 0.5f - 0.25f * fi + 0.1f * fk, -1.0f + 0.125f * fi, 0.25f * fk, w });
            c.deltaP.push_back({ 0.0f, 0.0f, 0.0f, 0.0f });
        }
        c.phases.assign(NS[k], 0u);
        c.p2m = { 0u };
        c.bOrigin = createBufferFrom(dev, c.origin); c.bPos = createBufferFrom(dev, c.pos);
        c.bVel = createBufferFrom(dev, c.vel); c.bDeltaP = createBufferFrom(dev, c.deltaP);
        c.bPhases = createBufferFrom(dev, c.phases); c.bP2M = createBufferFrom(dev, c.p2m);
        std::vector<uint8_t> psys(SZ_PARTICLE_SYS, 0);
        putU32(psys, 116, NS[k]);                       // mCommonData_NumParticles
        putU64(psys, 24, addrOf(dev, c.bPos));          // mSortedPositions_InvMass
        putU64(psys, 32, addrOf(dev, c.bVel));          // mSortedVelocities
        putU64(psys, 40, addrOf(dev, c.bPhases));       // mSortedPhaseArray
        putU64(psys, 80, addrOf(dev, bMaterials));      // mParticleMaterials
        putU64(psys, 160, addrOf(dev, c.bDeltaP));      // mSortedDeltaP
        putF32(psys, 316, CAP[k]);                      // mData_MaxVelocity (the CFL cap)
        putU32(psys, 632, 20u);                         // mParticleMaterialStride
        putU64(psys, 392, addrOf(dev, c.bOrigin));      // mSortedOriginPos_InvMass
        putU64(psys, 640, addrOf(dev, c.bP2M));         // mPhaseGroupToMaterialHandle
        std::memcpy(pool.data() + k * SZ_PARTICLE_SYS, psys.data(), SZ_PARTICLE_SYS);
    }
    PxgBufferHandle bPsys = createBufferFrom(dev, pool);
    PxgBufferHandle bAct = createBufferFrom(dev, std::vector<uint32_t>{ 2u, 0u });

    auto runStep = [&](bool isFirstIteration)
    {
        StepPush p{};
        p.particleSystems = addrOf(dev, bPsys);
        p.activeParticleSystems = addrOf(dev, bAct);   // {2, 0}: the CUDA never reads it
        p.dt = dt;
        p.invTotalDt = 1.0f / dt;
        p.flagsWord = isFirstIteration ? 1u : 0u;
        p.gravityX = 0.0f; p.gravityY = -9.81f; p.gravityZ = 0.0f;   // unused (extForces off)
        Kernel k("psContactSolveTgsBda.spv", "ps_stepParticlesLaunch", {}, sizeof(StepPush));
        k.dispatch(1u, {}, p, /*gridY = nbActive*/ 2u, 1u);
        dev->flush();
    };

    // CPU model of one step over slots 0 and 1 (pool[blockIdx.y], blockIdx.y < 2).
    std::vector<F4> rPos[3], rVel[3], rDeltaP[3];
    for (uint32_t k = 0; k < 3; ++k) { rPos[k] = s[k].pos; rVel[k] = s[k].vel; rDeltaP[k] = s[k].deltaP; }
    auto cpuStep = [&](bool isFirstIteration)
    {
        for (uint32_t k = 0; k < 2; ++k)
            for (uint32_t i = 0; i < NS[k]; ++i)
            {
                const F4 tPos = isFirstIteration ? s[k].origin[i] : rPos[k][i];
                if (tPos.w == 0.0f) continue;
                float vx = rVel[k][i].x, vy = rVel[k][i].y, vz = rVel[k][i].z;
                bool velChanged = false;
                const float lim = CAP[k];
                if (vx * vx + vy * vy + vz * vz > lim * lim)
                {
                    const float m = std::sqrt(vx * vx + vy * vy + vz * vz);
                    const float invM = 1.0f / m;
                    vx = (vx * invM) * lim; vy = (vy * invM) * lim; vz = (vz * invM) * lim;
                    velChanged = true;
                }
                const float dlx = isFirstIteration ? 0.0f : rDeltaP[k][i].x;
                const float dly = isFirstIteration ? 0.0f : rDeltaP[k][i].y;
                const float dlz = isFirstIteration ? 0.0f : rDeltaP[k][i].z;
                const float ddx = vx * dt, ddy = vy * dt, ddz = vz * dt;
                if (velChanged) rVel[k][i] = { vx, vy, vz, tPos.w };
                rPos[k][i] = { tPos.x + ddx, tPos.y + ddy, tPos.z + ddz, tPos.w };
                rDeltaP[k][i] = { ddx + dlx, ddy + dly, ddz + dlz, tPos.w };
            }
    };

    size_t mm = 0;
    auto compareAll = [&](const char* phase) {
        for (uint32_t k = 0; k < 3; ++k)
        {
            const auto gPos = downloadBuffer<F4>(dev, s[k].bPos, NS[k]);
            const auto gVel = downloadBuffer<F4>(dev, s[k].bVel, NS[k]);
            const auto gDp = downloadBuffer<F4>(dev, s[k].bDeltaP, NS[k]);
            for (uint32_t i = 0; i < NS[k]; ++i)
            {
                if (std::memcmp(&gPos[i], &rPos[k][i], sizeof(F4)) != 0 || std::memcmp(&gVel[i], &rVel[k][i], sizeof(F4)) != 0 ||
                    std::memcmp(&gDp[i], &rDeltaP[k][i], sizeof(F4)) != 0)
                {
                    if (mm < 8)
                        std::printf("  %s slot %u particle %u: pos gpu(%g,%g,%g) cpu(%g,%g,%g) vel gpu(%g,%g,%g) cpu(%g,%g,%g)\n",
                                    phase, k, i, gPos[i].x, gPos[i].y, gPos[i].z, rPos[k][i].x, rPos[k][i].y, rPos[k][i].z,
                                    gVel[i].x, gVel[i].y, gVel[i].z, rVel[k][i].x, rVel[k][i].y, rVel[k][i].z);
                    ++mm;
                }
            }
        }
    };
    runStep(true);  cpuStep(true);  compareAll("first");
    runStep(false); cpuStep(false); compareAll("next");

    PXG_CHECK_MSG(rVel[1][0].y == -2.0f, "slot 1's fast particle must be clamped to its own cap (v.y %g)", rVel[1][0].y);
    PXG_CHECK_MSG(mm == 0, "ps_stepParticlesLaunch steps pool slots 0 and 1 (blockIdx.y), not act {2,0}; slot 2 "
                  "untouched; both iterations bit-exact vs CPU (%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k)
        for (PxgBufferHandle h : { s[k].bOrigin, s[k].bPos, s[k].bVel, s[k].bDeltaP, s[k].bPhases, s[k].bP2M })
            dev->destroyBuffer(h);
    dev->destroyBuffer(bMaterials); dev->destroyBuffer(bPsys); dev->destroyBuffer(bAct);
}

// ===========================================================================
// ps_finalizeParticlesLaunch in isolation — velocity blend.
// ===========================================================================
PXG_TEST(psContactSolveTgs_finalize)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float invTotalDt = 60.0f, velocityScale = 0.7f;

    auto runFinalize = [&](F4 tVel, F4 dlta, F4& outVel) -> bool
    {
        std::vector<F4> vels = { tVel };
        std::vector<F4> deltaP = { dlta };
        PxgBufferHandle bVels = createBufferFrom(dev, vels);
        PxgBufferHandle bDeltaP = createBufferFrom(dev, deltaP);
        if (!addrOf(dev, bVels))
        {
            std::printf("  NOTE: no BDA -> skip\n");
            dev->destroyBuffer(bVels); dev->destroyBuffer(bDeltaP);
            return false;
        }
        std::vector<uint8_t> psys(SZ_PARTICLE_SYS, 0);
        putU32(psys, 116, 1u);                   // mCommonData_NumParticles
        putU64(psys, 32, addrOf(dev, bVels));    // mSortedVelocities
        putU64(psys, 160, addrOf(dev, bDeltaP)); // mSortedDeltaP
        PxgBufferHandle bPsys = createBufferFrom(dev, psys);

        FinalizePush p{};
        p.particleSystems = addrOf(dev, bPsys);
        p.id = 0u;
        p.invTotalDt = invTotalDt;
        p.velocityScale = velocityScale;
        Kernel k("psContactSolveTgsBda.spv", "ps_finalizeParticlesLaunch", {}, sizeof(FinalizePush));
        k.dispatch(1u, {}, p, 1u, 1u);
        dev->flush();

        outVel = downloadBuffer<F4>(dev, bVels, 1)[0];
        dev->destroyBuffer(bVels); dev->destroyBuffer(bDeltaP); dev->destroyBuffer(bPsys);
        return true;
    };

    // Case 1: |vel| > |deltaVel| -> blend applied.
    // deltaVel = dlta * invTotalDt = (0.001,0.002,0)*60 = (0.06,0.12,0); |deltaVel| small.
    // vel = (1,-2,0.5); |vel| large -> blended.
    {
        const F4 tVel = { 1.0f, -2.0f, 0.5f, 1.0f };
        const F4 dlta = { 0.001f, 0.002f, 0.0f, 1.0f };
        F4 gpu;
        if (!runFinalize(tVel, dlta, gpu))
            return;
        const V3 vel = { tVel.x, tVel.y, tVel.z };
        const V3 deltaVel = { dlta.x * invTotalDt, dlta.y * invTotalDt, dlta.z * invTotalDt };
        const float s = velocityScale;
        const F4 ref = { vel.x * s + deltaVel.x * (1.0f - s),
                         vel.y * s + deltaVel.y * (1.0f - s),
                         vel.z * s + deltaVel.z * (1.0f - s), tVel.w };
        PXG_CHECK_MSG(std::memcmp(&gpu, &ref, sizeof(F4)) == 0,
                      "finalize[blend] gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                      gpu.x, gpu.y, gpu.z, gpu.w, ref.x, ref.y, ref.z, ref.w);
    }

    // Case 2: |vel| <= |deltaVel| -> no write (velocity unchanged).
    // vel tiny, deltaVel large.
    {
        const F4 tVel = { 0.0f, 0.01f, 0.0f, 1.0f };
        const F4 dlta = { 0.0f, 0.5f, 0.0f, 1.0f }; // deltaVel.y = 30 >> vel
        F4 gpu;
        if (!runFinalize(tVel, dlta, gpu))
            return;
        PXG_CHECK_MSG(std::memcmp(&gpu, &tVel, sizeof(F4)) == 0,
                      "finalize[noblend] velocity wrongly changed: (%g,%g,%g,%g)",
                      gpu.x, gpu.y, gpu.z, gpu.w);
    }

    // Case 3: infinite-mass sentinel (tVel.w == 0) -> untouched.
    {
        const F4 tVel = { 1.0f, -2.0f, 0.5f, 0.0f };
        const F4 dlta = { 0.001f, 0.002f, 0.0f, 0.0f };
        F4 gpu;
        if (!runFinalize(tVel, dlta, gpu))
            return;
        PXG_CHECK_MSG(std::memcmp(&gpu, &tVel, sizeof(F4)) == 0,
                      "finalize[w==0] velocity wrongly changed: (%g,%g,%g,%g)",
                      gpu.x, gpu.y, gpu.z, gpu.w);
    }
}

// ===========================================================================
// P9.99b -- ps_finalizeParticlesLaunch with push id 2.  The host launches it once per
// active system with `index` = its remap index (PxgPBDParticleSystemCore.cpp:480-505);
// psContactSolveTgs_finalize only ever passed 0.  Pool of three: slots 0 and 1 are
// decoys with MORE particles whose velocities WOULD be blended (so a wrong id changes
// them), slot 2 holds psContactSolveTgs_finalize's three cases (blend, no blend,
// infinite mass) plus a fourth blended particle.
// ===========================================================================
PXG_TEST(psContactSolveTgs_finalize_p99Id2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float invTotalDt = 60.0f, velocityScale = 0.7f;
    const uint32_t NS[3] = { 6u, 5u, 4u };
    std::vector<F4> vel[3], dlta[3];
    PxgBufferHandle bVel[3], bDlta[3];
    // slot 2: the fixture's three cases + one more blend
    vel[2] = { { 1.0f, -2.0f, 0.5f, 1.0f }, { 0.0f, 0.01f, 0.0f, 1.0f }, { 1.0f, -2.0f, 0.5f, 0.0f }, { -3.0f, 0.25f, 1.5f, 0.5f } };
    dlta[2] = { { 0.001f, 0.002f, 0.0f, 1.0f }, { 0.0f, 0.5f, 0.0f, 1.0f }, { 0.001f, 0.002f, 0.0f, 0.0f }, { -0.004f, 0.0f, 0.003f, 0.5f } };
    for (uint32_t k = 0; k < 2; ++k)
        for (uint32_t i = 0; i < NS[k]; ++i)
        {
            vel[k].push_back({ 2.0f + float(i) + float(k), -1.5f, 0.75f * float(i), 1.0f });   // |vel| >> |deltaVel|: would blend
            dlta[k].push_back({ 0.002f * float(i), -0.001f, 0.0005f, 1.0f });
        }
    std::vector<uint8_t> pool(3 * SZ_PARTICLE_SYS, 0);
    for (uint32_t k = 0; k < 3; ++k)
    {
        bVel[k] = createBufferFrom(dev, vel[k]);
        bDlta[k] = createBufferFrom(dev, dlta[k]);
        if (!addrOf(dev, bVel[k])) { std::printf("  NOTE: no BDA -> skip\n"); return; }
        std::vector<uint8_t> psys(SZ_PARTICLE_SYS, 0);
        putU32(psys, 116, NS[k]);                    // mCommonData_NumParticles
        putU64(psys, 32, addrOf(dev, bVel[k]));      // mSortedVelocities
        putU64(psys, 160, addrOf(dev, bDlta[k]));    // mSortedDeltaP
        std::memcpy(pool.data() + k * SZ_PARTICLE_SYS, psys.data(), SZ_PARTICLE_SYS);
    }
    PxgBufferHandle bPsys = createBufferFrom(dev, pool);
    {
        FinalizePush p{};
        p.particleSystems = addrOf(dev, bPsys);
        p.id = 2u;
        p.invTotalDt = invTotalDt;
        p.velocityScale = velocityScale;
        Kernel k("psContactSolveTgsBda.spv", "ps_finalizeParticlesLaunch", {}, sizeof(FinalizePush));
        k.dispatch(1u, {}, p, 1u, 1u);
    }
    dev->flush();

    size_t mm = 0;
    for (uint32_t k = 0; k < 3; ++k)
    {
        const auto g = downloadBuffer<F4>(dev, bVel[k], NS[k]);
        for (uint32_t i = 0; i < NS[k]; ++i)
        {
            F4 want = vel[k][i];
            const F4 tv = vel[k][i], d = dlta[k][i];
            if (k == 2u && tv.w != 0.0f)
            {
                const V3 v = { tv.x, tv.y, tv.z };
                const V3 dv = { d.x * invTotalDt, d.y * invTotalDt, d.z * invTotalDt };
                if (dot3(v, v) > dot3(dv, dv))
                {
                    const float s = velocityScale;
                    want = { v.x * s + dv.x * (1.0f - s), v.y * s + dv.y * (1.0f - s), v.z * s + dv.z * (1.0f - s), tv.w };
                }
            }
            if (std::memcmp(&g[i], &want, sizeof(F4)) != 0)
            {
                if (mm < 8)
                    std::printf("  slot %u particle %u: gpu(%g,%g,%g,%g) want(%g,%g,%g,%g)\n", k, i, g[i].x, g[i].y, g[i].z, g[i].w,
                                want.x, want.y, want.z, want.w);
                ++mm;
            }
        }
    }
    PXG_CHECK_MSG(mm == 0, "ps_finalizeParticlesLaunch, id 2, decoys at slots 0/1 untouched: bit-exact vs CPU "
                  "(%zu mismatches)", mm);

    for (uint32_t k = 0; k < 3; ++k) { dev->destroyBuffer(bVel[k]); dev->destroyBuffer(bDlta[k]); }
    dev->destroyBuffer(bPsys);
}

} // namespace
