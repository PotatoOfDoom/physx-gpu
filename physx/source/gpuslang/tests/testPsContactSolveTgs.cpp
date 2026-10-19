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

// P6.6: the TGS-only particle kernels of psContactSolveTgsBda.slang (PhysX 5.12,
// gpusimulationcontroller/src/CUDA/particlesystem.cu).  Tests:
//
//   * psContactSolveTgs_step: ps_stepParticlesLaunch in isolation -- the substep
//     predictor advances the sorted position by vel*dt and accumulates the running
//     deltaP, on both the first (from sorted-origin, dlta=0) and a subsequent
//     (from sorted-position, dlta=running deltaP) iteration, bit-exact.
//   * psContactSolveTgs_finalize: ps_finalizeParticlesLaunch in isolation -- the
//     velocity blend newVel = vel*scale + deltaVel*(1-scale) when |vel| > |deltaVel|
//     (and the no-write case when it is not), bit-exact.
//   * P9.99b two-system twins: psContactSolveTgs_step_p99Direct (pins the CUDA's
//     pool[blockIdx.y] indexing, upstream note U1) and psContactSolveTgs_finalize_p99Id2
//     (push id 2).
//
// 5.12 deleted ps_solvePCOutputParticleDeltaVTGSLaunch: the one particle solve takes
// isTGS, and its TGS position / velocity iterations are tested in testPsContactSolve
// (psContactSolve_particleSolve_tgs*).  The TGS-solve tests of this file went with the
// kernel.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
static const uint32_t SZ_PARTICLE_SYS = 608u;   // PhysX 5.12 sizeof(PxgParticleSystem) (5.6.1: 656)

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h)
{
    return dev->getBufferDeviceAddress(h);
}
static void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
static void putF32(std::vector<uint8_t>& b, size_t off, float v) { std::memcpy(b.data() + off, &v, 4); }

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

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
    putU32(psys, 592, 20u);                      // mParticleMaterialStride
    putU64(psys, 392, addrOf(dev, bOrigin));     // mSortedOriginPos_InvMass
    putU64(psys, 600, addrOf(dev, bPhaseToMat)); // mPhaseGroupToMaterialHandle
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
        putU32(psys, 592, 20u);                         // mParticleMaterialStride
        putU64(psys, 392, addrOf(dev, c.bOrigin));      // mSortedOriginPos_InvMass
        putU64(psys, 600, addrOf(dev, c.bP2M));         // mPhaseGroupToMaterialHandle
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
