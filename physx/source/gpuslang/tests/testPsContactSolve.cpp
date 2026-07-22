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

// P6.4b-2: the particle-SIDE contact SOLVE half of the PBD particle/primitive
// consume path on the BDA engine ABI.  KEY PROOF: build ONE particle penetrating a
// static plane (contact normal (0,1,0), penetration -0.1, a per-particle range
// [0,1)) and run the full pipeline
//   ps_contactPrepareLaunch -> ps_solvePCOutputParticleDeltaVLaunch
//     -> ps_accumulateDeltaVParticleLaunch
// asserting the particle receives an UPWARD position delta that resolves the
// penetration, bit-exact vs a CPU transcription of particlesystem.cu.
//
// The rigid body is STATIC (rigidId.mID = 0xffffffff): the constraint-prep rigid
// response collapses to velMultiplier = 1/invMass1 (sqrtInvInertia = 0 -> resp0 =
// 0, invMass0 = 0), and the solve reads zero body velocity, so the per-contact
// deltaV is exactly the penetration depth * relaxation along +normal.  This is the
// only contact kind the particle-vs-static-floor scenes generate (the articulation
// branch is deferred to a later tranche and is unreachable here).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F2 { float x, y; };
struct F4 { float x, y, z, w; };
// PxgParticlePrimitiveContact (PxgParticleSystem.h, 32 bytes).
struct PsContact { F4 normal_pen; uint32_t rigidLo, rigidHi, particleLo, particleHi; };
static_assert(sizeof(PsContact) == 32, "PsContact stride");

// Byte-exact size of PxgParticlePrimitiveConstraintBlock (5 float4[32] + 2 u64[32]).
static const uint32_t CB_STRIDE = 3072u;

// Engine descriptor sizes (mirrors — pxgSlangParticleDesc / pxgSlangPrepDesc /
// pxgSlangSolverDesc): only the fields these three kernels touch are poked.
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

// ---------------------------------------------------------------------------
// CPU transcription of the three .cu kernels for ONE static-floor contact.
// Mirrors particlesystem.cu:1238 (prepare) / :1421 (solvePCOutputDeltaV core) /
// :1531 (solve) / :2828 (accumulate) with friction/adhesion disabled and a static
// rigid (invMass0 = 0, sqrtInvInertia = 0, body at origin, zero velocity).
// ---------------------------------------------------------------------------
struct CpuOut { F4 deltaPos; F2 appliedForce; F4 accumDeltaP; };
static CpuOut cpuReference(F4 contactNormalPen, float invMass1, float dt, float relax)
{
    // ---- contactPrepare (static rigid) ----
    const float nx = -contactNormalPen.x, ny = -contactNormalPen.y, nz = -contactNormalPen.z;
    const float error = contactNormalPen.w;
    // sqrtInvInertia = 0, invMass0 = 0 -> resp0 = 0.
    const float unitResponse = 0.0f + invMass1;
    const float velMultiplier = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

    // ---- solve (zero body velocity, delta = 0) ----
    const float deltaX = 0.0f, deltaY = 0.0f, deltaZ = 0.0f; // mSortedDeltaP.xyz
    const float normalDotDelta = nx * deltaX + ny * deltaY + nz * deltaZ;
    const float separation = error - normalDotDelta;
    const float normalVel = 0.0f;
    float appliedForceX = 0.0f, appliedForceY = 0.0f;
    float clamp = -appliedForceX;
    const float deltaF = (clamp > ((-separation - normalVel * dt) * velMultiplier))
                         ? clamp : ((-separation - normalVel * dt) * velMultiplier);
    appliedForceX = (0.0f > (deltaF + appliedForceX)) ? 0.0f : (deltaF + appliedForceX);
    // friction disabled -> deltaFr0 = deltaFr1 = 0, appliedForceY unchanged.
    const float dlvX = (-(nx * deltaF)) * invMass1 * relax;
    const float dlvY = (-(ny * deltaF)) * invMass1 * relax;
    const float dlvZ = (-(nz * deltaF)) * invMass1 * relax;
    const float scale = (deltaF != 0.0f) ? 1.0f : 0.0f;

    // ---- accumulate over range [0,1) ----
    float ax = dlvX, ay = dlvY, az = dlvZ;
    float fcount = scale;
    const float recip = 1.0f / ((1.0f > fcount) ? 1.0f : fcount);
    ax *= recip; ay *= recip; az *= recip;

    CpuOut o;
    o.deltaPos = { dlvX, dlvY, dlvZ, scale };
    o.appliedForce = { appliedForceX, appliedForceY };
    o.accumDeltaP = { ax, ay, az, 0.0f };
    return o;
}

// Push-constant packs (must match psContactSolveBda.slang exactly).
struct PrepPush
{
    uint64_t particleSystems, sortedContacts, numContacts, primitiveConstraints,
        preDesc, prepareDesc, appliedForces;
    float dt; uint32_t isTGSWord;
    uint64_t deltaVel, sharedDesc;
};
static_assert(sizeof(PrepPush) == 80, "prepare push 80 bytes");

struct SolvePush
{
    uint64_t particleSystems, sortedContacts, primitiveConstraints, numContacts,
        prePrepDesc, solverCoreDesc, sharedDesc, deltaPos, appliedForces;
    float dt, relaxationCoefficient; uint32_t isVelIterWord;
    uint64_t artiCoreDesc;
};
static_assert(sizeof(SolvePush) == 96, "solve push 96 bytes");

struct AccumPush
{
    uint64_t particleSystems, sortedContacts, totalNumPairs, rangeStart, rangeEnd, deltaPos;
};
static_assert(sizeof(AccumPush) == 48, "accum push 48 bytes");

// Run the full prepare->solve->accumulate pipeline for one static-floor contact.
// Returns the GPU deltaPos[0], appliedForce[0], accumDeltaP[0].
static bool runPipeline(PxgSlangComputeDevice* dev, F4 contactNormalPen, float invMass1,
                        float dt, float relax, CpuOut& gpu)
{
    // ---- particle-side buffers ----
    std::vector<F4> sortedPos = { { 0.0f, 0.0f, 0.0f, invMass1 } };    // mSortedPositions_InvMass
    std::vector<F4> sortedDeltaP = { { 0.0f, 0.0f, 0.0f, invMass1 } }; // mSortedDeltaP (delta.xyz, invMass.w)
    std::vector<uint32_t> phases = { 0u };                            // mSortedPhaseArray (group 0)
    std::vector<uint32_t> phaseToMat = { 0u };                        // mPhaseGroupToMaterialHandle (mi 0)
    std::vector<float> materials(8, 0.0f);                            // friction/adhesion/... all 0
    std::vector<F4> accumDeltaP = { { 0.0f, 0.0f, 0.0f, 0.0f } };     // mAccumDeltaP (output)

    PxgBufferHandle bPos = createBufferFrom(dev, sortedPos);
    PxgBufferHandle bDeltaP = createBufferFrom(dev, sortedDeltaP);
    PxgBufferHandle bPhases = createBufferFrom(dev, phases);
    PxgBufferHandle bPhaseToMat = createBufferFrom(dev, phaseToMat);
    PxgBufferHandle bMaterials = createBufferFrom(dev, materials);
    PxgBufferHandle bAccum = createBufferFrom(dev, accumDeltaP);
    if (!addrOf(dev, bPos)) { std::printf("  NOTE: no BDA -> skip\n");
        dev->destroyBuffer(bPos); dev->destroyBuffer(bDeltaP); dev->destroyBuffer(bPhases);
        dev->destroyBuffer(bPhaseToMat); dev->destroyBuffer(bMaterials); dev->destroyBuffer(bAccum);
        return false; }

    // PxgParticleSystemGpu descriptor (656 bytes), only touched fields poked.
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

    // ---- contact + constraint + per-contact scratch ----
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

    // ---- rigid-side buffers (static body idx 0) ----
    std::vector<uint8_t> solverBody(SZ_SOLVER_BODY, 0); // initialLinVelXYZ_invMassW.w = invMass0 = 0
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

    // solver velocities: numSolverBodies = 1, accOffset = 0 -> indices 0 and 1.
    std::vector<F4> outVel(2, { 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<F4> velPool(2, { 0.0f, 0.0f, 0.0f, 0.0f });
    PxgBufferHandle bOutVel = createBufferFrom(dev, outVel);
    PxgBufferHandle bVelPool = createBufferFrom(dev, velPool);

    std::vector<uint8_t> coreDesc(SZ_CORE_DESC, 0);
    putU64(coreDesc, 0, addrOf(dev, bOutVel)); // outSolverVelocity
    putU32(coreDesc, 268, 1u);                 // numSolverBodies
    putU32(coreDesc, 276, 0u);                 // accumulatedBodyDeltaVOffset
    PxgBufferHandle bCoreDesc = createBufferFrom(dev, coreDesc);

    std::vector<uint8_t> sharedPgs(SZ_SHARED_PGS, 0);
    putU64(sharedPgs, 160, addrOf(dev, bVelPool)); // iter_solverBodyVelPool
    PxgBufferHandle bSharedPgs = createBufferFrom(dev, sharedPgs);

    // range-find outputs (from P6.4b-1): one particle -> range [0,1).
    PxgBufferHandle bPairs = createBufferFrom(dev, std::vector<uint32_t>(1, 1u));
    PxgBufferHandle bStart = createBufferFrom(dev, std::vector<uint32_t>(1, 0u));
    PxgBufferHandle bEnd = createBufferFrom(dev, std::vector<uint32_t>(1, 1u));

    // ---- dispatch prepare (grid PS_COLLISION = 1024) ----
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
        p.isTGSWord = 0u;
        p.deltaVel = 0ull;                  // null -> rigid deltaV zeroing skipped
        p.sharedDesc = 0ull;                // unused (arti branch only)
        Kernel k("psContactSolveBda.spv", "ps_contactPrepareLaunch", {}, sizeof(PrepPush));
        k.dispatch(1024u, {}, p);
    }
    // ---- dispatch solve (grid PS_COLLISION = 1024) ----
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
        p.isVelIterWord = 0u;
        p.artiCoreDesc = 0ull;              // unused (arti branch only)
        Kernel k("psContactSolveBda.spv", "ps_solvePCOutputParticleDeltaVLaunch", {}, sizeof(SolvePush));
        k.dispatch(1024u, {}, p);
    }
    // ---- dispatch accumulate (grid ACCUMULATE_DELTA = 32) ----
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
        bPrepDesc, bPrePrepDesc, bOutVel, bVelPool, bCoreDesc, bSharedPgs, bPairs, bStart, bEnd };
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);
    return true;
}

// ===========================================================================
// KEY PROOF — a penetrating particle is pushed OUT (upward, resolving -0.1).
// ===========================================================================
PXG_TEST(psContactSolve_penetratingParticle)
{
    PxgSlangComputeDevice* dev = testDevice();

    // Contact: surface normal (0,1,0) pointing UP away from the floor, penetration
    // -0.1 (error).  invMass1 = 1, dt = 1/60, relaxation = 1 -> the correction fully
    // resolves the penetration in one solve.
    const F4 contactNormalPen = { 0.0f, 1.0f, 0.0f, -0.1f };
    const float invMass1 = 1.0f, dt = 1.0f / 60.0f, relax = 1.0f;

    CpuOut gpu;
    if (!runPipeline(dev, contactNormalPen, invMass1, dt, relax, gpu))
        return;
    const CpuOut ref = cpuReference(contactNormalPen, invMass1, dt, relax);

    // Direction: the particle is pushed UP and resolves the penetration.
    PXG_CHECK_MSG(gpu.accumDeltaP.y > 0.0f,
                  "accumDeltaP.y not upward: (%g,%g,%g)", gpu.accumDeltaP.x, gpu.accumDeltaP.y, gpu.accumDeltaP.z);
    PXG_CHECK_MSG(gpu.deltaPos.y > 0.0f,
                  "per-contact deltaPos.y not upward: %g", gpu.deltaPos.y);
    // Penetration magnitude fully resolved (accumDeltaP.y == 0.1 for relax = 1).
    PXG_CHECK_MSG(std::fabs(gpu.accumDeltaP.y - 0.1f) < 1e-6f,
                  "accumDeltaP.y=%g does not resolve -0.1 penetration", gpu.accumDeltaP.y);
    PXG_CHECK_MSG(gpu.accumDeltaP.x == 0.0f && gpu.accumDeltaP.z == 0.0f,
                  "lateral drift: (%g,%g)", gpu.accumDeltaP.x, gpu.accumDeltaP.z);

    // Bit-exact vs the CPU transcription of the .cu (deltaPos, appliedForce, accum).
    PXG_CHECK_MSG(std::memcmp(&gpu.deltaPos, &ref.deltaPos, sizeof(F4)) == 0,
                  "deltaPos gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  gpu.deltaPos.x, gpu.deltaPos.y, gpu.deltaPos.z, gpu.deltaPos.w,
                  ref.deltaPos.x, ref.deltaPos.y, ref.deltaPos.z, ref.deltaPos.w);
    PXG_CHECK_MSG(std::memcmp(&gpu.appliedForce, &ref.appliedForce, sizeof(F2)) == 0,
                  "appliedForce gpu(%g,%g) != cpu(%g,%g)",
                  gpu.appliedForce.x, gpu.appliedForce.y, ref.appliedForce.x, ref.appliedForce.y);
    PXG_CHECK_MSG(std::memcmp(&gpu.accumDeltaP, &ref.accumDeltaP, sizeof(F4)) == 0,
                  "accumDeltaP gpu(%g,%g,%g,%g) != cpu(%g,%g,%g,%g)",
                  gpu.accumDeltaP.x, gpu.accumDeltaP.y, gpu.accumDeltaP.z, gpu.accumDeltaP.w,
                  ref.accumDeltaP.x, ref.accumDeltaP.y, ref.accumDeltaP.z, ref.accumDeltaP.w);
    // The normal impulse magnitude equals the resolved penetration for a static
    // floor (velMultiplier=1/invMass1, deltaF=|error|*velMultiplier).
    PXG_CHECK_MSG(std::fabs(gpu.appliedForce.x - 0.1f) < 1e-6f,
                  "appliedForce.x=%g != 0.1 (deltaF for a fully-resolved -0.1 pen)", gpu.appliedForce.x);
}

// ===========================================================================
// Under-relaxed correction (relax = 0.5) — still upward, still bit-exact, and
// exercises a non-trivial relaxation multiply.
// ===========================================================================
PXG_TEST(psContactSolve_relaxationScaled)
{
    PxgSlangComputeDevice* dev = testDevice();
    const F4 contactNormalPen = { 0.0f, 1.0f, 0.0f, -0.1f };
    const float invMass1 = 2.0f, dt = 1.0f / 60.0f, relax = 0.5f; // invMass1 != 1 too

    CpuOut gpu;
    if (!runPipeline(dev, contactNormalPen, invMass1, dt, relax, gpu))
        return;
    const CpuOut ref = cpuReference(contactNormalPen, invMass1, dt, relax);

    PXG_CHECK_MSG(gpu.accumDeltaP.y > 0.0f, "relaxed accumDeltaP.y not upward: %g", gpu.accumDeltaP.y);
    // For a static floor the invMass1 cancels: correction == |pen| * relax = 0.05.
    PXG_CHECK_MSG(std::fabs(gpu.accumDeltaP.y - 0.05f) < 1e-6f,
                  "relaxed accumDeltaP.y=%g != 0.05", gpu.accumDeltaP.y);
    PXG_CHECK_MSG(std::memcmp(&gpu.accumDeltaP, &ref.accumDeltaP, sizeof(F4)) == 0,
                  "relaxed accumDeltaP gpu(%g,%g,%g) != cpu(%g,%g,%g)",
                  gpu.accumDeltaP.x, gpu.accumDeltaP.y, gpu.accumDeltaP.z,
                  ref.accumDeltaP.x, ref.accumDeltaP.y, ref.accumDeltaP.z);
}

// ===========================================================================
// Infinite-mass particle (invMass1 = 0) — the solve `continue`s, deltaPos stays
// the init sentinel, and no correction accumulates.
// ===========================================================================
PXG_TEST(psContactSolve_infiniteMassNoResponse)
{
    PxgSlangComputeDevice* dev = testDevice();
    const F4 contactNormalPen = { 0.0f, 1.0f, 0.0f, -0.1f };

    CpuOut gpu;
    if (!runPipeline(dev, contactNormalPen, /*invMass1*/ 0.0f, 1.0f / 60.0f, 1.0f, gpu))
        return;

    // solve `continue`d: deltaPos[0] untouched (init 0,0,0,0) -> scale 0.
    PXG_CHECK_MSG(gpu.deltaPos.x == 0.0f && gpu.deltaPos.y == 0.0f && gpu.deltaPos.z == 0.0f &&
                  gpu.deltaPos.w == 0.0f, "infinite-mass deltaPos not zero: (%g,%g,%g,%g)",
                  gpu.deltaPos.x, gpu.deltaPos.y, gpu.deltaPos.z, gpu.deltaPos.w);
    // accumulate over the range with scale 0 -> zero correction (recipCount clamps).
    PXG_CHECK_MSG(gpu.accumDeltaP.y == 0.0f,
                  "infinite-mass particle wrongly moved: accumDeltaP.y=%g", gpu.accumDeltaP.y);
}

} // namespace
