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

// testPsSolveDensity.cpp -- P9.1c fixture for ps_solveDensityLaunch
// (particlesystem.cu:3404-3687), the second entry point of npPsFluidDensity.slang.
//
// ORACLE.  Host transcription again (the CUDA is __global__ and its algebra has no
// PX_CUDA_CALLABLE original), carried by generated-and-pinned offsets, whole-region
// 0xAB poison compare over BOTH outputs -- mDelta and the per-contact warm-start
// mDensityCollisionImpulses -- and the sweep.
//
// The collision-force array is the sharper of the two checks.  It is the SAME array
// the kernel reads its warm start from, so it is uploaded already seeded and an
// untouched slot must come back holding its seed.  The solid branch writes it
// UNCONDITIONALLY and the fluid branch never does, so the pattern of seeded vs
// rewritten slots encodes exactly which pairs took which branch: a mutant that
// mis-routes one pair shows up as a rewritten slot that should still hold its seed,
// or the reverse.
//
// GEOMETRY, the same four-system shape P9.1b settled on and for the same reasons:
// pool of four, THREE active with active = { 2, 0, 3 } (not a permutation, so the
// blockIdx.y mutant dies against a decoy at pool index 1); A has 500 particles so a
// halved block stride leaves a hole; C carries eDISABLE_SELF_COLLISION while being
// fully populated; every per-particle array is PAD = 544 long with a poisoned tail
// so the `p >= numParticles` early-out has something to hit on the wrong side.
//
// PHASES.  Three groups: 0 and 1 are FLUID, 2 is SOLID.  That yields all four pair
// kinds the branch tree distinguishes -- fluid/fluid same phase (both group 0),
// fluid/fluid different phase (0 vs 1, which takes the bilateral-only arm because
// the comparison is on the WHOLE phase word), fluid/solid, and solid/solid.
//
// MATERIALS.  Three, mapped from group by a non-identity permutation, and chosen so
// each sub-path is exercised both ways: one material has friction == 0 (the
// friction block is skipped), one has surfaceTension == 0 (the normals term is
// skipped), and the cfl coefficients differ so the CFL projection fires for some
// pairs and not others.
//
// PER-PARTICLE CASES placed at known indices of A:
//   0  invMass == 0            -- the fixed-particle early-out: delta zeroed, and
//                                 NOTHING else written, so every collision-force
//                                 slot of particle 0 must come back at its seed (S1)
//   1  first neighbour at      -- dSq < 1e-10, so the whole pair is skipped and its
//      1e-6 separation           force slot keeps its seed (S3)
//   2  same-phase fluid pair   -- the full cohesion / viscosity / CFL / surface-
//                                 tension arm (S5)
//   3  different-phase fluid   -- the bilateral-only arm (S5)
//   4  fluid vs solid          -- the solid branch from the fluid side (S4)
//   5  solid vs solid          -- the solid branch from the solid side
//   6  solid pair with         -- d >= cohesionRadius, so adhesionClamp stays 0
//      no adhesion               (S10)
//   7  solid pair engineered   -- warm-start force.x == 0 and a separated pair, so
//      to leave newForce.x == 0   mtd == 0 and newForce.x == 0: the delta and the
//                                 friction are skipped but the STORE still happens
//                                 (S12), which is the only way to tell the two
//                                 apart
// The warm-start forces are non-zero and varied everywhere else, so S13's use of
// the ACCUMULATED newForce.y is observable rather than starting from zero.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };
struct F2 { float x, y; };
struct V3 { float x, y, z; };

static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 mul3(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 smul3(float s, V3 a) { return { s * a.x, s * a.y, s * a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

struct SdPush
{
    uint64_t particleSystems;
    uint64_t activeParticleSystems;
    float coefficient;
    float dt;
};
static_assert(sizeof(SdPush) == 24, "kPsIntegrate blob is 24 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;
static const uint32_t PS_FLAG_DISABLE_SELF_COLLISION = 1u;
static const uint32_t PS_PHASE_FLUID = 0x00400000u;
static const uint32_t PS_PHASE_GROUP_MASK = 0x000fffffu;

// --- host mirrors of the shader's scalar helpers ---------------------------
static float sqrH(float x) { return x * x; }
static float cubeH(float x) { return x * x * x; }
static float octH(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x4; }
static float WH(float x, float k1, float kInv) { return k1 * sqrH(1.0f - x * kInv); }
static float dWdxH(float x, float k2, float kInv) { return -k2 * (1.0f - x * kInv); }
static float WCohesionH(float x, float c1, float c2, float kInv)
{ float w = x * kInv; return c1 * cubeH(w) + c2 * sqrH(w) - 1.0f; }
static float pxMaxH(float a, float b) { return a > b ? a : b; }
static float pxMinH(float a, float b) { return a < b ? a : b; }
static float pxClampH(float v, float lo, float hi) { return pxMinH(hi, pxMaxH(lo, v)); }

// The 15 floats of PxsPBDMaterialData, by float index.
enum : uint32_t { MF_FRICTION = 0, MF_ADHESION = 2, MF_ADHESION_RADIUS = 4,
                  MF_VISCOSITY = 6, MF_SURFACE_TENSION = 8, MF_CFL = 12,
                  MF_FRICTION_SCALE = 13, MF_ADHESION_SCALE = 14, MF_COUNT = 15 };

struct Sys
{
    uint32_t numParticles = 0, flags = 0;
    float restDensity = 0.f, solidRestOffset = 0.f, invRestDensity = 0.f;
    float spiky1 = 0.f, spiky2 = 0.f;
    float contactDist = 0.f, contactDistInv = 0.f, contactDistSq = 0.f;
    uint32_t materialStride = 0;

    std::vector<F4> pos, deltaP, normals;
    std::vector<uint32_t> phase, collisionIndex, contactCount;
    std::vector<float> density;
    std::vector<F2> forces;             // numParticles * MN
    std::vector<float> materials;       // MF_COUNT floats per material
    std::vector<uint16_t> phaseToMat;
    std::vector<float> derived;         // 4 floats per group

    DevBuf bPos, bDeltaP, bNormals, bPhase, bColl, bCount, bDensity, bForces,
           bMaterials, bP2m, bDerived;
    DevBuf bDelta, bForcesOut;
};

// The transcription of particlesystem.cu:3446-3686.
static void refSolveDensity(const Sys& s, float coefficient, float dt, uint32_t MN,
                            std::vector<uint32_t>& outDelta,
                            std::vector<uint32_t>& outForces)
{
    if ((s.flags & PS_FLAG_DISABLE_SELF_COLLISION) != 0u)
        return;

    const uint32_t N = s.numParticles;

    for (uint32_t p = 0; p < N; ++p)
    {
        const uint32_t phase = s.phase[p];
        const bool isFluid = (phase & PS_PHASE_FLUID) != 0u;
        const uint32_t group = phase & PS_PHASE_GROUP_MASK;

        const F4 xi4 = s.pos[p];

        if (xi4.w == 0.0f)                                            // S1
        {
            outDelta[p * 4u + 0u] = f2u(0.0f);
            outDelta[p * 4u + 1u] = f2u(0.0f);
            outDelta[p * 4u + 2u] = f2u(0.0f);
            outDelta[p * 4u + 3u] = f2u(0.0f);
            continue;
        }

        const uint32_t mi = s.phaseToMat[group];
        const float* mat = s.materials.data() + size_t(mi) * MF_COUNT;
        const float* der = s.derived.data() + size_t(group) * 4u;

        const V3 xi = { xi4.x, xi4.y, xi4.z };
        const V3 xiDelta = { s.deltaP[p].x, s.deltaP[p].y, s.deltaP[p].z };
        const V3 ni = { s.normals[p].x, s.normals[p].y, s.normals[p].z };

        const float scale = dt;
        const float density = s.density[p];
        const uint32_t cc = s.contactCount[p];

        V3 delta = { 0.f, 0.f, 0.f };

        const float cellSizeSq = s.contactDistSq;
        const float cflRadius = s.contactDist * mat[MF_CFL];
        const float solidRestDistance = 2.0f * s.solidRestOffset;
        const float spiky2 = s.spiky2, spiky1 = s.spiky1;
        const float pcdInv = s.contactDistInv;
        const float friction = mat[MF_FRICTION] * mat[MF_FRICTION_SCALE];
        const float cohesion = der[1] * scale;
        const float cohesion1 = der[2], cohesion2 = der[3];
        const float viscosity = mat[MF_VISCOSITY] * scale * s.invRestDensity;
        const float adhesion = mat[MF_ADHESION] * mat[MF_ADHESION_SCALE];
        const float matSurfaceTension = mat[MF_SURFACE_TENSION];
        const float cohesionRadius = solidRestDistance * mat[MF_ADHESION_RADIUS];
        const float invAura = 1.0f / (cohesionRadius - solidRestDistance);

        float weight = 0.0f;

        for (uint32_t i = 0; i < cc; ++i)
        {
            const uint32_t q = s.collisionIndex[p + i * N];
            const F4 xj4 = s.pos[q];
            const uint32_t phase2 = s.phase[q];
            const float density2 = isFluid ? s.density[q] : 0.0f;
            const V3 xjDelta = { s.deltaP[q].x, s.deltaP[q].y, s.deltaP[q].z };

            const V3 xij = sub3(xi, V3{ xj4.x, xj4.y, xj4.z });
            const float dSq = dot3(xij, xij);
            const uint32_t writeOffset = p + i * N;

            if (dSq >= 1e-10f)                                        // S3
            {
                const float d = std::sqrt(dSq);                       // S2
                const V3 nij = smul3(1.0f / d, xij);

                const bool fluid2 = (phase2 & PS_PHASE_FLUID) != 0u;
                if (fluid2 && isFluid)                                // S4
                {
                    if (dSq < cellSizeSq && dSq > 0.0f)
                    {
                        const float lambda = (density + density2) * 0.5f;   // S6
                        const float sTerm = lambda * dWdxH(d, spiky2, pcdInv);

                        if (phase2 != phase)                          // S5
                        {
                            const V3 t = smul3(sTerm, nij);
                            delta = sub3(delta, t);
                        }
                        else
                        {
                            const float c = cohesion * WCohesionH(d, cohesion1, cohesion2, pcdInv);
                            const V3 relDelta_ij = sub3(xiDelta, xjDelta);

                            delta = sub3(delta, mul3(nij, sTerm + c));

                            const float viscosityScale =
                                1.0f - (1.0f / (1.0f + viscosity * WH(d, spiky1, pcdInv)));
                            delta = sub3(delta, smul3(viscosityScale, relDelta_ij));

                            const float relDeltaProj = dot3(nij, relDelta_ij);
                            if (relDeltaProj < -cflRadius)            // S7
                                delta = sub3(delta, mul3(mul3(nij, relDeltaProj + cflRadius), 0.5f));

                            if (matSurfaceTension > 0.0f)             // S8
                            {
                                const V3 nq = { s.normals[q].x, s.normals[q].y, s.normals[q].z };
                                delta = sub3(delta, mul3(sub3(ni, nq), scale));
                            }
                        }
                        weight += 1.0f;                               // S15
                    }
                }
                else
                {
                    const F2 force = s.forces[writeOffset];
                    F2 newForce;

                    const float w = xi4.w / (xi4.w + xj4.w);           // S9
                    const float e = (d - solidRestDistance);

                    float adhesionClamp = 0.0f;
                    if (d < cohesionRadius)                            // S10
                        adhesionClamp = pxMaxH(0.0f, adhesion * octH(1.0f - e * invAura) * e);

                    const float mtd = pxClampH(pxMinH(e, pxMaxH(-force.x, adhesionClamp)),
                                               -solidRestDistance, solidRestDistance);   // S11

                    newForce.x = force.x + mtd;
                    newForce.y = force.y;

                    if (newForce.x != 0.0f)                            // S12
                    {
                        const float deltaF = w * mtd;
                        delta = sub3(delta, mul3(nij, deltaF));

                        if (friction > 0.0f)
                        {
                            const V3 j = sub3(xiDelta, xjDelta);
                            const float jn = dot3(j, nij);
                            const V3 jt = sub3(j, smul3(jn, nij));
                            const float jtSq = dot3(jt, jt);

                            if (jtSq > 0.0f)
                            {
                                const float len = std::sqrt(jtSq);
                                const float maxForce = pxMaxH(-newForce.x, 0.0f) * friction;
                                const float clampedForce = pxMinH(maxForce, len + newForce.y);
                                const float deltaForce = clampedForce - newForce.y;   // S13

                                newForce.y += deltaForce;

                                // S14: PxVec3::operator/ is reciprocal-then-multiply
                                const V3 t = mul3(mul3(jt, deltaForce), w);
                                delta = sub3(delta, mul3(t, 1.0f / len));
                            }
                        }
                        weight += 1.0f;                                // S15
                    }

                    newForce.x = pxMinH(newForce.x, 0.0f);             // S12
                    outForces[writeOffset * 2u + 0u] = f2u(newForce.x);
                    outForces[writeOffset * 2u + 1u] = f2u(newForce.y);
                }
            }
        }

        delta = mul3(delta, coefficient);                              // S16
        outDelta[p * 4u + 0u] = f2u(delta.x);
        outDelta[p * 4u + 1u] = f2u(delta.y);
        outDelta[p * 4u + 2u] = f2u(delta.z);
        outDelta[p * 4u + 3u] = f2u(weight);
    }
}

static float jitter(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return float(h % 20011u) * (1.0f / 20011.0f);
}

PXG_TEST(psSolveDensity)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t BLOCK = 256u;
    const uint32_t MN = 6u;
    const uint32_t PAD = 544u;
    const uint32_t NUM_GROUPS = 3u;
    const uint32_t NUM_MATERIALS = 3u;

    const float SOLID_REST_OFFSET = 0.09f;          // solidRestDistance = 0.18
    const float CONTACT_DIST = 0.5f;

    auto buildSys = [&](uint32_t N, uint32_t flags, float restDensity,
                        float invRestDensity, float spiky1, float spiky2,
                        uint32_t salt, bool withSpecials)
    {
        Sys s;
        s.numParticles = N; s.flags = flags;
        s.restDensity = restDensity; s.solidRestOffset = SOLID_REST_OFFSET;
        s.invRestDensity = invRestDensity;
        s.spiky1 = spiky1; s.spiky2 = spiky2;
        s.contactDist = CONTACT_DIST;
        s.contactDistInv = 1.0f / CONTACT_DIST;
        s.contactDistSq = CONTACT_DIST * CONTACT_DIST;
        s.materialStride = MF_COUNT * 4u;

        s.pos.resize(PAD); s.deltaP.resize(PAD); s.normals.resize(PAD);
        s.phase.resize(PAD); s.contactCount.resize(PAD); s.density.resize(PAD);
        s.collisionIndex.assign(size_t(N) * MN, 0u);
        s.forces.assign(size_t(N) * MN, F2{ 0.f, 0.f });

        for (uint32_t p = 0; p < PAD; ++p)
        {
            const bool real = (p < N);
            s.pos[p] = { jitter(p, salt + 1u) * 1.1f, jitter(p, salt + 2u) * 1.1f,
                         jitter(p, salt + 3u) * 1.1f,
                         0.4f + jitter(p, salt + 4u) };
            s.deltaP[p] = { jitter(p, salt + 5u) * 0.06f - 0.03f,
                            jitter(p, salt + 6u) * 0.06f - 0.03f,
                            jitter(p, salt + 7u) * 0.06f - 0.03f, 0.f };
            s.normals[p] = { jitter(p, salt + 8u) - 0.5f, jitter(p, salt + 9u) - 0.5f,
                             jitter(p, salt + 10u) - 0.5f, jitter(p, salt + 11u) };
            const uint32_t grp = p % NUM_GROUPS;
            s.phase[p] = grp | ((grp < 2u) ? PS_PHASE_FLUID : 0u);   // groups 0,1 fluid
            s.density[p] = jitter(p, salt + 12u) * 0.5f - 0.1f;
            s.contactCount[p] = real ? (1u + (p % (MN - 1u))) : 0u;
        }
        for (uint32_t p = 0; p < N; ++p)
            for (uint32_t i = 0; i < MN; ++i)
            {
                s.collisionIndex[p + i * N] = (p * 29u + i * 71u + 3u) % N;
                s.forces[p + i * N] = { -(0.004f + jitter(p * MN + i, salt + 13u) * 0.02f),
                                        jitter(p * MN + i, salt + 14u) * 0.01f };
            }

        if (withSpecials)
        {
            // 0: fixed particle
            s.pos[0].w = 0.0f;
            // 1: first neighbour 1e-6 away -> dSq == 1e-12 < 1e-10, skipped
            s.contactCount[1] = 2u;
            s.collisionIndex[1 + 0u * N] = 20u;
            s.pos[20] = s.pos[1];
            s.pos[20].x = s.pos[1].x + 1e-6f;
            s.phase[20] = 0u | PS_PHASE_FLUID;
            // 2: same-phase fluid pair, well inside the contact distance
            s.phase[2] = 0u | PS_PHASE_FLUID;
            s.contactCount[2] = 2u;
            s.collisionIndex[2 + 0u * N] = 21u;
            s.phase[21] = 0u | PS_PHASE_FLUID;
            s.pos[21] = { s.pos[2].x + 0.12f, s.pos[2].y, s.pos[2].z, 0.8f };
            // The CFL projection fires on relDeltaProj < -cflRadius.  nij is
            // (xi - xj)/d = -x here (the neighbour sits at +x), so relDelta_ij
            // must point along +x for the projection to come out NEGATIVE.  The
            // first draft had this sign backwards, the branch never fired, and
            // the mutant that drops the 0.5f factor survived.
            s.deltaP[21] = { s.deltaP[2].x - 0.4f, s.deltaP[2].y, s.deltaP[2].z, 0.f };
            // 3: DIFFERENT-phase fluid pair -> bilateral only
            s.phase[3] = 0u | PS_PHASE_FLUID;
            s.contactCount[3] = 1u;
            s.collisionIndex[3 + 0u * N] = 22u;
            s.phase[22] = 1u | PS_PHASE_FLUID;
            s.pos[22] = { s.pos[3].x + 0.10f, s.pos[3].y, s.pos[3].z, 0.7f };
            // 4: fluid vs SOLID
            s.phase[4] = 0u | PS_PHASE_FLUID;
            s.contactCount[4] = 1u;
            s.collisionIndex[4 + 0u * N] = 23u;
            s.phase[23] = 2u;                       // solid
            s.pos[23] = { s.pos[4].x + 0.11f, s.pos[4].y, s.pos[4].z, 0.9f };
            // 5: solid vs solid, penetrating (e < 0)
            s.phase[5] = 2u;
            s.contactCount[5] = 1u;
            s.collisionIndex[5 + 0u * N] = 24u;
            s.phase[24] = 2u;
            s.pos[24] = { s.pos[5].x + 0.10f, s.pos[5].y, s.pos[5].z, 0.6f };
            // 6: solid pair beyond the cohesion radius -> adhesionClamp stays 0
            s.phase[6] = 2u;
            s.contactCount[6] = 1u;
            s.collisionIndex[6 + 0u * N] = 25u;
            s.phase[25] = 2u;
            s.pos[25] = { s.pos[6].x + 0.44f, s.pos[6].y, s.pos[6].z, 0.6f };
            // 7: warm start 0 AND separated beyond the cohesion radius, so
            //    mtd == 0 and newForce.x == 0: no delta, but the store happens.
            s.phase[7] = 2u;
            s.contactCount[7] = 1u;
            s.collisionIndex[7 + 0u * N] = 26u;
            s.phase[26] = 2u;
            s.pos[26] = { s.pos[7].x + 0.45f, s.pos[7].y, s.pos[7].z, 0.6f };
            s.forces[7 + 0u * N] = { 0.0f, 0.0f };

            // --- witnesses added after the first sweep, each closing an identity ---

            // 0 also needs a neighbour IN RANGE, or dropping the fixed-particle
            // early-out changes nothing: with invMass 0 the mass split w is 0, so
            // the solid branch contributes no delta -- but it still WRITES the
            // collision force and bumps weight, and that is what makes the mutant
            // observable.  A fluid neighbour out of range would not.
            s.contactCount[0] = 1u;
            s.collisionIndex[0 + 0u * N] = 30u;
            s.phase[30] = 2u;                        // solid
            s.pos[30] = { s.pos[0].x + 0.1f, s.pos[0].y, s.pos[0].z, 0.7f };

            // 8: a same-phase fluid pair at EXACTLY the contact distance, so the
            // strict `<` of the inner range guard is observable.  The coordinates
            // are chosen so the subtraction is exact: 0.75 - 0.25 == 0.5 and
            // 0.5 * 0.5 == 0.25 == contactDistSq, all exactly representable.
            s.phase[8] = 0u | PS_PHASE_FLUID;
            s.contactCount[8] = 1u;
            s.collisionIndex[8 + 0u * N] = 27u;
            s.phase[27] = 0u | PS_PHASE_FLUID;
            s.pos[8] = { 0.75f, 0.3f, 0.4f, 0.8f };
            s.pos[27] = { 0.25f, 0.3f, 0.4f, 0.8f };

            // 9: two fluid particles in the SAME GROUP but with DIFFERENT PHASE
            // WORDS (one carries eParticlePhaseSelfCollide, bit 20, which is
            // outside the group mask).  The engine splits on the whole phase word,
            // so this pair takes the bilateral arm; a port that split on the group
            // would take the cohesion arm.  Without such a pair the two spellings
            // are indistinguishable.
            s.phase[9] = 0u | PS_PHASE_FLUID | 0x00100000u;
            s.contactCount[9] = 1u;
            s.collisionIndex[9 + 0u * N] = 28u;
            s.phase[28] = 0u | PS_PHASE_FLUID;
            s.pos[28] = { s.pos[9].x + 0.13f, s.pos[9].y, s.pos[9].z, 0.75f };

            // 10: a solid pair where the ADHESION, not the warm-started force,
            // wins the pxMax inside mtd.  Everywhere else -force.x (order 1e-2)
            // dominates adhesionClamp (order 1e-4) and the adhesion terms are
            // unobservable; here the warm start is 1e-5 and the pair sits at
            // d = 0.25, inside group 2's cohesion radius of 0.54.
            s.phase[10] = 2u;
            s.contactCount[10] = 1u;
            s.collisionIndex[10 + 0u * N] = 29u;
            s.phase[29] = 2u;
            s.pos[10] = { 0.20f, 0.5f, 0.6f, 0.65f };
            s.pos[29] = { 0.45f, 0.5f, 0.6f, 0.65f };
            s.forces[10 + 0u * N] = { -1e-5f, 0.0f };
        }

        // materials: one frictionless, one with no surface tension, one full
        s.materials.assign(NUM_MATERIALS * MF_COUNT, 0.0f);
        for (uint32_t m = 0; m < NUM_MATERIALS; ++m)
        {
            float* mm = s.materials.data() + size_t(m) * MF_COUNT;
            for (uint32_t f = 0; f < MF_COUNT; ++f)
                mm[f] = -900.0f - float(m) * 50.0f - float(f);   // unread fields poisoned
            mm[MF_FRICTION] = (m == 1u) ? 0.0f : (0.3f + 0.2f * float(m));
            mm[MF_FRICTION_SCALE] = 0.8f + 0.1f * float(m);
            mm[MF_ADHESION] = 0.05f + 0.03f * float(m);
            // NOTE: none of these may land on exactly 1.0 -- a unit scale makes the
            // "drop the scale" mutant an identity for that material.
            mm[MF_ADHESION_SCALE] = 0.6f + 0.13f * float(m);
            mm[MF_ADHESION_RADIUS] = 2.0f + 0.5f * float(m);
            mm[MF_VISCOSITY] = 0.4f + 0.25f * float(m);
            mm[MF_SURFACE_TENSION] = (m == 2u) ? 0.0f : (0.6f + 0.2f * float(m));
            mm[MF_CFL] = 0.15f + 0.1f * float(m);
        }
        // a non-identity group -> material map
        s.phaseToMat.assign(NUM_GROUPS, 0u);
        for (uint32_t g = 0; g < NUM_GROUPS; ++g)
            s.phaseToMat[g] = uint16_t((g * 2u + 1u) % NUM_MATERIALS);

        s.derived.assign(NUM_GROUPS * 4u, 0.0f);
        for (uint32_t g = 0; g < NUM_GROUPS; ++g)
        {
            s.derived[g * 4u + 0u] = -7000.0f - float(g);   // surfaceTensionDerived: unread here
            s.derived[g * 4u + 1u] = 0.35f + 0.1f * float(g);   // cohesion
            s.derived[g * 4u + 2u] = -1.4f - 0.2f * float(g);   // cohesion1
            s.derived[g * 4u + 3u] = 2.1f + 0.3f * float(g);    // cohesion2
        }
        return s;
    };

    Sys sysA = buildSys(500u, 0u, 1.05f, 0.9f, 1.7f, 2.3f, 100u, true);
    Sys sysD = buildSys(200u, 0u, 0.95f, 1.1f, 1.2f, 2.9f, 200u, false);
    Sys sysB = buildSys(130u, 0u, 1.25f, 0.7f, 2.1f, 1.6f, 300u, false);
    Sys sysC = buildSys(90u, PS_FLAG_DISABLE_SELF_COLLISION, 1.0f, 1.0f, 1.4f, 2.0f, 400u, false);

    const uint32_t GRID_X = (sysA.numParticles + BLOCK - 1u) / BLOCK;
    PXG_CHECK_MSG(GRID_X == 2u, "two blocks needed (got %u)", GRID_X);
    PXG_CHECK_MSG(sysA.numParticles > 128u * (GRID_X - 1u) + 255u,
                  "A must outreach a halved block stride");
    PXG_CHECK_MSG(PAD > GRID_X * BLOCK, "the poisoned tail must outreach the grid");

    Sys* order[4] = { &sysA, &sysD, &sysB, &sysC };
    for (Sys* s : order)
    {
        s->bPos = makeDev(dev, s->pos);
        s->bDeltaP = makeDev(dev, s->deltaP);
        s->bNormals = makeDev(dev, s->normals);
        s->bPhase = makeDev(dev, s->phase);
        s->bColl = makeDev(dev, s->collisionIndex);
        s->bCount = makeDev(dev, s->contactCount);
        s->bDensity = makeDev(dev, s->density);
        s->bMaterials = makeDev(dev, s->materials);
        s->bP2m = makeDev(dev, s->phaseToMat);
        s->bDerived = makeDev(dev, s->derived);
    }

    std::vector<uint32_t> active = { 2u, 0u, 3u };
    DevBuf bActive = makeDev(dev, active);

    const float coefficient = 0.63f;
    const float dt = 1.0f / 90.0f;

    // mDelta is poisoned.  mDensityCollisionImpulses is the SAME array the kernel
    // both reads (warm start) and writes, so it is created already seeded; an
    // untouched slot must come back holding its seed, which is what makes the
    // branch routing observable -- the fluid arm never writes it.
    for (Sys* s : order)
    {
        std::vector<uint32_t> d(size_t(PAD) * 4u, POISON_U32);
        s->bDelta = makeDev(dev, d);

        std::vector<uint32_t> seeded(size_t(s->numParticles) * MN * 2u, POISON_U32);
        for (uint32_t p = 0; p < s->numParticles; ++p)
            for (uint32_t i = 0; i < MN; ++i)
            {
                const uint32_t off = p + i * s->numParticles;
                seeded[off * 2u + 0u] = f2u(s->forces[off].x);
                seeded[off * 2u + 1u] = f2u(s->forces[off].y);
            }
        s->bForcesOut = makeDev(dev, seeded);
    }

    std::vector<PxgParticleSystem> descArr;
    for (Sys* s : order)
    {
        PxgParticleSystem ps;
        std::memset(&ps, 0xEE, sizeof(ps));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s->bPos.addr));
        ps.mSortedDeltaP = reinterpret_cast<float4*>(size_t(s->bDeltaP.addr));
        ps.mSurfaceNormal = reinterpret_cast<float4*>(size_t(s->bNormals.addr));
        ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(s->bPhase.addr));
        ps.mCollisionIndex = reinterpret_cast<PxU32*>(size_t(s->bColl.addr));
        ps.mParticleSelfCollisionCount = reinterpret_cast<PxU32*>(size_t(s->bCount.addr));
        ps.mDensity = reinterpret_cast<PxReal*>(size_t(s->bDensity.addr));
        ps.mParticleMaterials =
            reinterpret_cast<PxsParticleMaterialData*>(size_t(s->bMaterials.addr));
        ps.mParticleMaterialStride = s->materialStride;
        ps.mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(s->bP2m.addr));
        ps.mDerivedPBDMaterialData =
            reinterpret_cast<PxgPBDParticleMaterialDerived*>(size_t(s->bDerived.addr));
        ps.mDensityCollisionImpulses = reinterpret_cast<float2*>(size_t(s->bForcesOut.addr));
        ps.mDelta = reinterpret_cast<float4*>(size_t(s->bDelta.addr));
        ps.mCommonData.mNumParticles = s->numParticles;
        ps.mCommonData.mParticleContactDistance = s->contactDist;
        ps.mCommonData.mParticleContactDistanceInv = s->contactDistInv;
        ps.mCommonData.mParticleContactDistanceSq = s->contactDistSq;
        ps.mData.mFlags = s->flags;
        ps.mData.mRestDensity = s->restDensity;
        ps.mData.mSolidRestOffset = s->solidRestOffset;
        ps.mData.mInvRestDensity = s->invRestDensity;
        ps.mData.mSpiky1 = s->spiky1;
        ps.mData.mSpiky2 = s->spiky2;
        descArr.push_back(ps);
    }
    DevBuf bDesc = makeDev(dev, descArr);

    SdPush push = {};
    push.particleSystems = bDesc.addr;
    push.activeParticleSystems = bActive.addr;
    push.coefficient = coefficient;
    push.dt = dt;

    Kernel k("npPsFluidDensity.spv", "ps_solveDensityLaunch", {}, sizeof(SdPush));
    k.dispatch(GRID_X, {}, push, uint32_t(active.size()), 1u);
    dev->flush();

    size_t mm = 0;
    for (Sys* s : order)
    {
        std::vector<uint32_t> rd(size_t(PAD) * 4u, POISON_U32);
        std::vector<uint32_t> rf(size_t(s->numParticles) * MN * 2u, POISON_U32);
        for (uint32_t p = 0; p < s->numParticles; ++p)
            for (uint32_t i = 0; i < MN; ++i)
            {
                const uint32_t off = p + i * s->numParticles;
                rf[off * 2u + 0u] = f2u(s->forces[off].x);
                rf[off * 2u + 1u] = f2u(s->forces[off].y);
            }

        const bool isActive = (s == &sysA || s == &sysB || s == &sysC);
        if (isActive)
            refSolveDensity(*s, coefficient, dt, MN, rd, rf);

        const auto gd = downloadBuffer<uint32_t>(dev, s->bDelta.h, PAD * 4u);
        const auto gf = downloadBuffer<uint32_t>(dev, s->bForcesOut.h,
                                                 s->numParticles * MN * 2u);

        auto cmp = [&](const std::vector<uint32_t>& r, const std::vector<uint32_t>& g,
                       const char* what)
        {
            for (size_t w = 0; w < r.size(); ++w)
                if (r[w] != g[w])
                {
                    if (mm < 12)
                    {
                        float rfv, gfv;
                        std::memcpy(&rfv, &r[w], 4); std::memcpy(&gfv, &g[w], 4);
                        std::printf("  sys %p %s MM %zu: ref %g got %g\n",
                                    (void*)s, what, w, rfv, gfv);
                    }
                    ++mm;
                }
        };
        cmp(rd, gd, "delta");
        cmp(rf, gf, "forces");
    }

    PXG_CHECK_MSG(mm == 0, "ps_solveDensityLaunch bit-exact (%zu mismatches)", mm);

    // Coverage: the branch tree must actually have been walked both ways.
    {
        const uint32_t N = sysA.numParticles;
        uint32_t fluidPairs = 0, solidPairs = 0, skipped = 0, samePhase = 0, diffPhase = 0;
        for (uint32_t p = 0; p < N; ++p)
        {
            if (sysA.pos[p].w == 0.0f) continue;
            const uint32_t phase = sysA.phase[p];
            const bool isFluid = (phase & PS_PHASE_FLUID) != 0u;
            for (uint32_t i = 0; i < sysA.contactCount[p]; ++i)
            {
                const uint32_t q = sysA.collisionIndex[p + i * N];
                const V3 xij = sub3(V3{ sysA.pos[p].x, sysA.pos[p].y, sysA.pos[p].z },
                                    V3{ sysA.pos[q].x, sysA.pos[q].y, sysA.pos[q].z });
                const float dSq = dot3(xij, xij);
                if (dSq < 1e-10f) { ++skipped; continue; }
                const bool fluid2 = (sysA.phase[q] & PS_PHASE_FLUID) != 0u;
                if (fluid2 && isFluid)
                {
                    if (dSq < sysA.contactDistSq)
                    {
                        ++fluidPairs;
                        if (sysA.phase[q] == phase) ++samePhase; else ++diffPhase;
                    }
                }
                else ++solidPairs;
            }
        }
        PXG_CHECK_MSG(skipped > 0u, "at least one pair must fall under the 1e-10 guard");
        PXG_CHECK_MSG(samePhase > 0u, "the same-phase fluid arm must be exercised");
        PXG_CHECK_MSG(diffPhase > 0u, "the different-phase fluid arm must be exercised");
        PXG_CHECK_MSG(solidPairs > 0u, "the solid arm must be exercised");
        std::printf("  coverage: fluid %u (same %u / diff %u), solid %u, skipped %u\n",
                    fluidPairs, samePhase, diffPhase, solidPairs, skipped);
    }

    for (Sys* s : order)
    {
        dev->destroyBuffer(s->bPos.h); dev->destroyBuffer(s->bDeltaP.h);
        dev->destroyBuffer(s->bNormals.h); dev->destroyBuffer(s->bPhase.h);
        dev->destroyBuffer(s->bColl.h); dev->destroyBuffer(s->bCount.h);
        dev->destroyBuffer(s->bDensity.h); dev->destroyBuffer(s->bMaterials.h);
        dev->destroyBuffer(s->bP2m.h); dev->destroyBuffer(s->bDerived.h);
        dev->destroyBuffer(s->bDelta.h); dev->destroyBuffer(s->bForcesOut.h);
    }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

} // namespace
