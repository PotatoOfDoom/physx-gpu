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

// npPsFluidDensity.slang
//
// P9.1b -- ps_calculateDensityAndPotentialLaunch: the PBD SPH density,
// surface-normal and diffuse-potential accumulation.  CUDA source is the
// __device__ TEMPLATE calculateDensityAndPotential (particlesystem.cu:3222-3384);
// the extern "C" kernel at :3386-3395 is a six-line dispatcher that instantiates
// it twice over the bool.
//
// LAUNCH.  args (particleSystems*, activeParticleSystems*, bool
// shouldComputePotentials) -> {8,8,1}; grid (ceil(getMaxParticles()/256),
// nbActiveParticleSystems, 1), block (PS_SOLVE = 256, 1, 1), mStream.  Guarded
// host-side only by (maxParticles > 0) -- there is NO fluid predicate at the
// launch site, so this kernel is already dispatched every solver iteration by
// the existing non-fluid particle scene and simply early-outs per particle.
// mComputePotentials is true on exactly the first solveParticleCollision of each
// frame (set in preIntegrateSystems, cleared after solveDensities).
//
// TEMPLATE -> RUNTIME BOOL.  The CUDA's bool is a template parameter purely so
// hasDiffuseParticles folds at compile time; it gates no ABI and no signature.
// The port uses ONE entry point with a runtime bool read from the push blob, so
// the manifest entry name matches the kernel name and there is one body to
// sweep rather than two.
//
// STAGING DROPPED.  The CUDA blockCopy<float>s the whole 608-byte
// PxgParticleSystem into __shared__ and then reads every field from the copy.
// Nothing writes the descriptor, so direct BDA reads are byte-identical; this is
// the same bandwidth-only staging particleSim / particleSelfCollisionBda already
// drop.  Loop-invariant scalars (mSpiky1, mSpiky2, the contact-distance
// reciprocal) are hoisted out of the neighbour loop: in the CUDA they are free
// __shared__ reads, here they would be repeated global loads, and the value is
// unchanging so hoisting is an identity.
//
// ============================================================================
// THE UPSTREAM BUG THIS KERNEL MUST REPRODUCE (candidate 13)
// ============================================================================
// particlesystem.cu:3296-3304, verbatim:
//
//     PxU32 offset = p;
//     if (contactCount > 0)
//     {
//         xjNext = PxLoad3(sortedNewPositions[collisionIndex[offset]]);   // slot 0
//         offset += numParticles;                                        // <-- HERE
//         if (hasDiffuseParticles)
//             vjNext = PxLoad3(sortedNewVelocities[collisionIndex[offset]]); // slot 1
//     }
//
// The offset increment sits BETWEEN the first neighbour's position load and its
// velocity load, so the preamble pairs pos[idx[0]] with vel[idx[1]].  Writing
// idx[k] = collisionIndex[p + k*numParticles], the loop consumes
//
//     position at iteration i : idx[i]          (correct at every i)
//     velocity at iteration i : idx[max(i, 1)]  (WRONG only at i == 0)
//
// so vel[idx[0]] is never read and vel[idx[1]] is read twice.  Only `div` ->
// potentials[p].x is affected; rho, weight, densities[p] and the surface normal
// consume positions only and are clean.  This function is the only one of four
// analogous preambles in the tree (:3521, :3829, :3967, anisotropy.cu:266) that
// fails to latch the index into a register before loading.
//
// Verified by six independent refuters before being claimed.  The port
// REPRODUCES it -- this is a bit-exactness port, not a bug fix.  Do not
// "correct" the max(i, 1) below; see the note in the same spirit at :4034 of
// the CUDA, where a known-wrong dtSq is preserved for compatibility.
//
// SAFETY DIVERGENCE, the one place the port does not follow the CUDA.  When
// contactCount == 1 and hasDiffuseParticles is true, slot 1 was never written by
// ps_selfCollisionLaunch for this particle, so it holds a stale or uninitialised
// PxU32, and sortedNewVelocities[thatValue] is an UNBOUNDED device read.  Under
// the P8.96c clamp rule an unbounded BDA load faults on llvmpipe, so the port
// clamps the resulting PARTICLE INDEX to < numParticles.  For every genuine
// contact the clamp is a no-op (collisionIndex entries are always in range), so
// this changes only the case where the CUDA's own read is undefined.
//
// ============================================================================
// The traps, each of which gets a mutant
// ============================================================================
//  T1. eDISABLE_SELF_COLLISION returns for the WHOLE dispatch before anything is
//      read (:3239) -- it is a per-system flag tested before the p < numParticles
//      guard, so no particle of that system is touched.
//  T2. A non-fluid particle returns having written NOTHING (:3263) -- not even
//      densities[p].  Its density slot keeps whatever it held.
//  T3. rho is seeded from staticDensities[p] ONLY when mRestDensityBoundary > 0
//      (:3278); otherwise it starts at literal zero.
//  T4. `d` is 1.f/rsqrtf(dSq), NOT sqrtf(dSq) (:3333-3334).  A double reciprocal
//      is not the identity on floats, so the host mirror must spell it the same
//      way.  For dSq == 0 this gives rd = inf and d = 0, and W(0) = spiky1, so a
//      coincident pair DOES contribute to rho -- there is no dSq > 0 guard on the
//      density accumulation, unlike the sibling kernels.
//  T5. The distance guard is STRICT `<` here (:3331), where
//      ps_vorticityConfinementLaunch and ps_solveVelocityLaunch use `<=`.
//  T6. The gradient branch is guarded by
//      (surfaceTensionDerived > 0 || hasDiffuseParticles) && d > 0 (:3341) -- the
//      d > 0 half is what excludes the coincident pair from the normal.
//  T7. dw is dWdx(...) * xij * rd -- multiplied by the RECIPROCAL, left to right,
//      never divided by d (:3343).
//  T8. The density constraint is PxMax(rho - restDensity, -restDensity * 0.005f)
//      (:3361) -- a small NEGATIVE floor, the engine's own stabilisation, then
//      scaled by mLambdaScale as a separate statement (:3364).
//  T9. normals[p] is written ONLY when surfaceTensionDerived > 0 (:3369), carries
//      the SCALED n, and puts `weight` (not rho, not the count) in .w.
// T10. potentials[p] is written only under hasDiffuseParticles, which is
//      (mMaxDiffuseParticles > 0) && shouldComputePotentials (:3244) -- the AND
//      of a descriptor field and the push-constant bool.
// T11. div is divided by (xi4.w * rho) only when BOTH xi4.w > 0 and rho > 0
//      (:3378); note it is the raw rho, not the constrained c.
// T12. The `mat` binding at :3268 is DEAD -- nothing in the whole function reads
//      it (the only material reads are three of derivedMaterial
//      .surfaceTensionDerived, at :3341/:3369/:3371).  The port omits both the
//      getParticleMaterial deref and the phaseToMat[group] load that feeds it;
//      `group` survives because the DERIVED array is indexed by it.
//
// The module is in PXG_NOCONTRACT_SHADERS: dense SPH float work whose bit
// pattern ps_solveDensityLaunch consumes directly.

import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim::PS_SOLVE
static const uint PS_FLUID_BLOCK = 256u;

// PxParticleFlag::eDISABLE_SELF_COLLISION (PxPBDParticleSystem.h:182)
static const uint PS_FLAG_DISABLE_SELF_COLLISION = 1u;
// PxParticlePhaseFlag::eParticlePhaseFluid (PxParticleSystemFlag.h:94)

// sizeof(PxgPBDParticleMaterialDerived); floats: surfaceTensionDerived@0,
// cohesion@1, cohesion1@2, cohesion2@3.
static const uint PXG_PBD_DERIVED_STRIDE = 16u;
static const uint PXG_PBD_DERIVED_F_SURFACE_TENSION = 0u;

// particleSystem.cuh:58-68 -- the SPH spiky kernel scalars.  Already ported and
// unit-tested standalone as npSphSpikyKernel; pxgSlangParticleDesc defines no
// functions, so there is nothing importable and they are restated here.
float sqr(float x) { return x * x; }
float W(float x, float kSpiky1, float kInvRadius) { return kSpiky1 * sqr(1.0f - x * kInvRadius); }
float dWdx(float x, float kSpiky2, float kInvRadius) { return -kSpiky2 * (1.0f - x * kInvRadius); }

// PxMax/PxMin for float are intrinsics::selectMax/selectMin, which are plain
// ternaries (PxUnixMathIntrinsics.h:116-125) -- NOT fmaxf/fminf, so they differ
// on NaN and signed zero.  Spelled out, as femClothRefitBda already does.
float pxMax(float a, float b) { return a > b ? a : b; }
float pxMin(float a, float b) { return a < b ? a : b; }

// PxGetGroup (PxParticleGpu.h) -- bits [0,19] of the phase.
uint psGetGroup(uint phase) { return phase & PXG_PARTICLE_PHASE_GROUP_MASK; }
// PxGetFluid (PxParticleGpu.h:65)

// argSizes (launch site PxgPBDParticleSystemCore.cpp:727-732):
// { particleSystems*, activeParticleSystems*, bool shouldComputePotentials }
// -> {8,8,1} = 17, rounded to the 8-alignment -> 24-byte blob.  The bool byte
// sits at 16 and is read as a masked word (the bridge zero-pads 17..23).
struct PsCalcDensityParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8
    uint shouldComputePotentialsWord; // @16 bool byte 0
    uint _pad20;                      // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsCalcDensityParams> gCd;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_calculateDensityAndPotentialLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                           uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gCd.particleSystems);
    const uint particleId = (Ptr<uint>(gCd.activeParticleSystems))[groupId.y];

    // T1: a per-SYSTEM early-out, before any per-particle work.
    if ((systems[particleId].mData_Flags & PS_FLAG_DISABLE_SELF_COLLISION) != 0u)
        return;

    const uint numParticles = systems[particleId].mCommonData_NumParticles;

    const bool shouldComputePotentials = (gCd.shouldComputePotentialsWord & 0xFFu) != 0u;
    // T10: the AND of a descriptor field and the push bool.
    const bool hasDiffuseParticles =
        (systems[particleId].mCommonData_MaxDiffuseParticles > 0u) && shouldComputePotentials;

    const uint p = groupId.x * PS_FLUID_BLOCK + groupThreadId.x;

    if (p < numParticles)
    {
        Ptr<uint> phases = Ptr<uint>(systems[particleId].mSortedPhaseArray);
        const uint phase = phases[p];

        // T2: a non-fluid particle writes NOTHING at all.
        if (!psGetFluid(phase))
            return;

        const uint group = psGetGroup(phase);

        // T12: the getParticleMaterial binding at :3268 (and the phaseToMat load
        // that feeds it) is DEAD -- omitted.  `group` survives because the
        // DERIVED array is indexed by the group.
        const float surfaceTensionDerived =
            (Ptr<float>(systems[particleId].mDerivedPBDMaterialData
                        + uint64_t(group) * uint64_t(PXG_PBD_DERIVED_STRIDE)))
            [PXG_PBD_DERIVED_F_SURFACE_TENSION];

        Ptr<float4> sortedNewPositions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
        Ptr<float4> sortedNewVelocities = Ptr<float4>(systems[particleId].mSortedVelocities);
        Ptr<uint> collisionIndex = Ptr<uint>(systems[particleId].mCollisionIndex);

        const float4 xi4 = sortedNewPositions[p];
        const float3 xi = float3(xi4.x, xi4.y, xi4.z);
        const float4 vi4 = sortedNewVelocities[p];
        const float3 vi = float3(vi4.x, vi4.y, vi4.z);

        // T3: the boundary-density seed is conditional.
        float rho = 0.0f;
        if (systems[particleId].mData_RestDensityBoundary > 0.0f)
            rho = (Ptr<float>(systems[particleId].mStaticDensity))[p];

        float3 n = float3(0.0f, 0.0f, 0.0f);
        float weight = 0.0f;
        float div = 0.0f;

        const uint contactCount = (Ptr<uint>(systems[particleId].mParticleSelfCollisionCount))[p];

        const float cellSizeSq = systems[particleId].mCommonData_ParticleContactDistanceSq;
        const float spiky1 = systems[particleId].mData_Spiky1;
        const float spiky2 = systems[particleId].mData_Spiky2;
        const float invRadius = systems[particleId].mCommonData_ParticleContactDistanceInv;

        for (uint i = 0u; i < contactCount; ++i)
        {
            // The pipeline un-rolled to its closed form.  POSITION uses idx[i];
            // VELOCITY uses idx[max(i,1)] -- upstream candidate 13, reproduced
            // deliberately.  See the module header before changing this.
            const uint qPos = collisionIndex[p + i * numParticles];
            const float4 xj4 = sortedNewPositions[qPos];
            const float3 xj = float3(xj4.x, xj4.y, xj4.z);

            float3 vj = float3(0.0f, 0.0f, 0.0f);
            if (hasDiffuseParticles)
            {
                const uint iVel = (i > 0u) ? i : 1u;
                const uint qVelRaw = collisionIndex[p + iVel * numParticles];
                // P8.96c clamp: with contactCount == 1 the CUDA's own read here
                // is of an unwritten slot, so the index can be arbitrary.
                const uint qVel = (qVelRaw < numParticles) ? qVelRaw : 0u;
                const float4 vj4 = sortedNewVelocities[qVel];
                vj = float3(vj4.x, vj4.y, vj4.z);
            }

            const float3 xij = xi - xj;
            const float dSq = xij.x * xij.x + xij.y * xij.y + xij.z * xij.z;

            if (dSq < cellSizeSq)   // T5: STRICT less-than
            {
                // T4: a double reciprocal, not a sqrt.
                const float rd = rsqrt(dSq);
                const float d = 1.0f / rd;

                const float mass = 1.0f;

                rho += mass * W(d, spiky1, invRadius);
                weight += mass;

                // T6
                if ((surfaceTensionDerived > 0.0f || hasDiffuseParticles) && d > 0.0f)
                {
                    // T7: * xij * rd, left to right; never / d.
                    const float3 dw = dWdx(d, spiky2, invRadius) * xij * rd;
                    n += dw;

                    if (hasDiffuseParticles)
                    {
                        const float3 vij = vj - vi;
                        div += vij.x * dw.x + vij.y * dw.y + vij.z * dw.z;
                    }
                }
            }
        }

        // T8: the negative floor, then a separate scale statement.
        const float restDensity = systems[particleId].mData_RestDensity;
        float c = pxMax(rho - restDensity, -restDensity * 0.005f);
        c *= systems[particleId].mData_LambdaScale;

        (Ptr<float>(systems[particleId].mDensity))[p] = c;

        // T9
        if (surfaceTensionDerived > 0.0f)
        {
            n *= surfaceTensionDerived;
            (Ptr<float4>(systems[particleId].mSurfaceNormal))[p] =
                float4(n.x, n.y, n.z, weight);
        }

        if (hasDiffuseParticles)
        {
            // T11: both halves of the guard, and the RAW rho.
            if (xi4.w > 0.0f && rho > 0.0f)
                div /= (xi4.w * rho);

            // The CUDA's second term is max(1.0f - 2.0f*rho, 0.0) with a DOUBLE
            // literal, so it evaluates in double and narrows on the store.  The
            // narrowing is exact because the value came from a float, so the
            // float-domain max below is bit-identical.
            (Ptr<float2>(systems[particleId].mDiffusePotentials))[p] =
                float2(div, pxMax(1.0f - 2.0f * rho, 0.0f));
        }
    }
}

// ============================================================================
// P9.1c -- ps_solveDensityLaunch (particlesystem.cu:3404-3687)
// ============================================================================
// The PBD position-delta solve.  Despite the name it is NOT a fluid-only kernel:
// its ELSE branch is the SOLID particle-particle collision, adhesion and friction
// solve, taken whenever either particle of a pair is non-fluid.  That is why
// mapping it is what finally gives the existing particle scene real
// particle-particle collision response -- ps_applyDeltaLaunch zeroes mDelta after
// consuming it, so with this kernel refused the Jacobi delta was identically zero
// every iteration.
//
// LAUNCH.  args (particleSystems*, activeParticleSystems*, PxReal coefficient,
// PxReal dt) -- coefficient BEFORE dt -> {8,8,4,4}; grid
// (ceil(getMaxParticles()/256), nbActiveParticleSystems, 1), block (256,1,1),
// mStream, once per solver iteration.
//
// Four DEAD locals dropped, each verified by reading every use in the body:
//   - `xiVel = PxLoad3(sortedVel[p])` (:3472) is never referenced.  It is a real
//     global load of mSortedVelocities[p] that nothing consumes.
//   - the INNER `const PxVec3 xij = xjDelta - xiDelta;` (:3625) SHADOWS the outer
//     xij and is never referenced either; the friction block below it recomputes
//     the negation as `j = xiDelta - xjDelta`.
//   - `ni4.w` is dead: both of its uses are commented out in the CUDA (:3480 and
//     the tail of :3503), so only the xyz of normals[p] is live.
//   - `invRadius` (:3502) is a second read of the same
//     mParticleContactDistanceInv that `particleContactDistanceInv` (:3495)
//     already holds; one load serves both.
//
// The `__syncwarp()` at :3683 is NOT load-bearing and is dropped.  Unlike the
// P8.124 case, there is no cross-lane hazard to preserve here: nothing in the
// loop reads mDelta, and every collisionForces slot the loop writes is
// writeOffset = p + i*numParticles, which is unique per particle.  No lane ever
// reads another lane's write, so the barrier orders nothing.
//
// The material and derived-material loads are moved BELOW the fixed-particle
// early-out.  In the CUDA they sit above it and are therefore executed even for a
// particle that immediately returns; deferring them is observationally identical
// (they are pure reads) and avoids dereferencing mDerivedPBDMaterialData on a
// path whose result is discarded.
//
// Traps, each with a mutant:
//  S1. A FIXED particle (xi4.w == 0) clears its delta and returns (:3461-3466) --
//      it writes deltas[p] = 0 rather than leaving it, and writes nothing else.
//  S2. `d` here is sqrtf(dSq) -- a REAL square root, unlike
//      ps_calculateDensityAndPotentialLaunch's 1/rsqrt double reciprocal.
//  S3. The outer guard is dSq >= 1e-10f (:3566); the fluid branch then ALSO
//      requires dSq < cellSizeSq && dSq > 0.
//  S4. The fluid branch needs BOTH particles fluid (:3571); everything else,
//      including fluid-vs-solid, takes the solid branch.
//  S5. Same-phase vs different-phase split INSIDE the fluid branch: different
//      phases get the bilateral density term ONLY (no cohesion, no viscosity, no
//      CFL, no surface tension) -- the engine's multiphase behaviour.
//  S6. lambda is the AVERAGE (density + density2) * 0.5f, not a sum.
//  S7. The CFL projection fires only when relDeltaProj < -cflRadius, and the
//      correction is scaled by 0.5f.
//  S8. Surface tension uses the RAW mat.surfaceTension as its gate (:3609), not
//      the derivedMaterial.surfaceTensionDerived that the density kernel uses.
//  S9. `w` is the invMass ratio xi4.w / (xi4.w + xj4.w) -- the pair's mass split.
// S10. adhesionClamp is computed only when d < cohesionRadius, and is
//      PxMax(0, adhesion * oct(1 - e*invAura) * e) -- oct is the EIGHTH power.
// S11. mtd = PxClamp(PxMin(e, PxMax(-force.x, adhesionClamp)), +-solidRestDistance)
//      -- a min of the penetration against a max of the warm-started force and
//      the adhesion, then symmetrically clamped.
// S12. The delta and the friction update happen only when newForce.x != 0, but
//      newForce.x = PxMin(newForce.x, 0) and the STORE happen unconditionally.
// S13. Friction reads the ACCUMULATED newForce.y as its warm start and clamps
//      against length + newForce.y, not against length alone.
// S14. `delta -= jt * deltaForce * w / length` goes through PxVec3::operator/,
//      which is a RECIPROCAL-then-multiply (PxVec3.h:219-223), NOT a
//      per-component divide.  Slang's `/` on a float3 IS per-component, so the
//      port must spell the reciprocal explicitly.
// S15. weight is incremented in BOTH branches but at different depths: in the
//      fluid branch once per in-range pair regardless of the phase split, in the
//      solid branch only inside the newForce.x != 0 guard.
// S16. delta is scaled by `coefficient` once, after the loop.

float cube(float x) { return x * x * x; }
float oct(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x4; }

// particlesystem.cu:3397-3401
float WCohesion(float x, float kCohesion1, float kCohesion2, float kInvRadius)
{
    float w = x * kInvRadius;
    return kCohesion1 * cube(w) + kCohesion2 * sqr(w) - 1.0f;
}

// PxClamp (PxMath.h:142-146) is PxMin(hi, PxMax(lo, v)) -- that spelling exactly.
float pxClamp(float v, float lo, float hi) { return pxMin(hi, pxMax(lo, v)); }

// PxsPBDMaterialData float indices (offsets / 4): friction@0, adhesion@2,
// adhesionRadiusScale@4, viscosity@6, surfaceTension@8, cflCoefficient@12,
// particleFrictionScale@13, particleAdhesionScale@14.
static const uint PXG_PBD_MAT_F_FRICTION = 0u;
static const uint PXG_PBD_MAT_F_ADHESION = 2u;
static const uint PXG_PBD_MAT_F_ADHESION_RADIUS_SCALE = 4u;
static const uint PXG_PBD_MAT_F_VISCOSITY = 6u;
static const uint PXG_PBD_MAT_F_SURFACE_TENSION_RAW = 8u;
static const uint PXG_PBD_MAT_F_CFL_COEFFICIENT = 12u;
static const uint PXG_PBD_MAT_F_PARTICLE_FRICTION_SCALE = 13u;
static const uint PXG_PBD_MAT_F_PARTICLE_ADHESION_SCALE = 14u;

// NOTE: unlike ps_updateMaterials, this kernel reaches the material through
// getParticleMaterial with the RUNTIME mParticleMaterialStride from the
// descriptor (particlesystem.cu:3453), not a compile-time sizeof.  The two are
// equal at runtime, but the port spells what this kernel does.
static const uint PXG_PBD_DERIVED_F_COHESION = 1u;
static const uint PXG_PBD_DERIVED_F_COHESION1 = 2u;
static const uint PXG_PBD_DERIVED_F_COHESION2 = 3u;

// phaseToMat[group]: a PxU16 array read as a 32-bit word plus shift/mask, so the
// software Vulkan target needs no 16-bit storage capability.
uint sdLoadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

// argSizes (launch site PxgPBDParticleSystemCore.cpp:743-749):
// { particleSystems*, activeParticleSystems*, PxReal coefficient, PxReal dt }
// -> {8,8,4,4} = 24-byte blob.  coefficient comes BEFORE dt.
struct PsSolveDensityParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8
    float coefficient;                // @16
    float dt;                         // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsSolveDensityParams> gSd;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_solveDensityLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSd.particleSystems);
    const uint particleId = (Ptr<uint>(gSd.activeParticleSystems))[groupId.y];

    if ((systems[particleId].mData_Flags & PS_FLAG_DISABLE_SELF_COLLISION) != 0u)
        return;

    const uint numParticles = systems[particleId].mCommonData_NumParticles;

    const uint p = groupId.x * PS_FLUID_BLOCK + groupThreadId.x;
    if (p >= numParticles)
        return;

    Ptr<float4> sortedNewPositions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
    Ptr<float4> sortedDeltaP = Ptr<float4>(systems[particleId].mSortedDeltaP);
    Ptr<float> densities = Ptr<float>(systems[particleId].mDensity);
    Ptr<float4> normals = Ptr<float4>(systems[particleId].mSurfaceNormal);
    Ptr<uint> phases = Ptr<uint>(systems[particleId].mSortedPhaseArray);
    Ptr<uint> collisionIndex = Ptr<uint>(systems[particleId].mCollisionIndex);
    Ptr<float2> collisionForces = Ptr<float2>(systems[particleId].mDensityCollisionImpulses);
    Ptr<float4> deltas = Ptr<float4>(systems[particleId].mDelta);

    const uint phase = phases[p];
    const bool isFluid = psGetFluid(phase);
    const uint group = psGetGroup(phase);

    const float4 xi4 = sortedNewPositions[p];

    // S1: a fixed particle clears its delta and returns.
    if (xi4.w == 0.0f)
    {
        deltas[p] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // The material loads sit BELOW the early-out here; see the header.
    const uint mi = sdLoadPhaseToMaterial(systems[particleId].mPhaseGroupToMaterialHandle, group);
    const uint64_t matBase = systems[particleId].mParticleMaterials
                           + uint64_t(mi) * uint64_t(systems[particleId].mParticleMaterialStride);
    Ptr<float> mat = Ptr<float>(matBase);
    const uint64_t derivedBase = systems[particleId].mDerivedPBDMaterialData
                               + uint64_t(group) * uint64_t(PXG_PBD_DERIVED_STRIDE);
    Ptr<float> derived = Ptr<float>(derivedBase);

    const float3 xi = float3(xi4.x, xi4.y, xi4.z);
    const float4 xiDelta4 = sortedDeltaP[p];
    const float3 xiDelta = float3(xiDelta4.x, xiDelta4.y, xiDelta4.z);

    // Only the xyz of normals[p] is live; ni4.w's two uses are commented out.
    const float4 ni4 = normals[p];
    const float3 ni = float3(ni4.x, ni4.y, ni4.z);

    const float scale = gSd.dt;          // the CUDA's `const PxReal scale = dt;`
    const float density = densities[p];
    const uint contactCount = (Ptr<uint>(systems[particleId].mParticleSelfCollisionCount))[p];

    float3 delta = float3(0.0f, 0.0f, 0.0f);

    const float cellSizeSq = systems[particleId].mCommonData_ParticleContactDistanceSq;
    const float cflRadius = systems[particleId].mCommonData_ParticleContactDistance
                          * mat[PXG_PBD_MAT_F_CFL_COEFFICIENT];
    const float solidRestDistance = 2.0f * systems[particleId].mData_SolidRestOffset;

    const float spiky2 = systems[particleId].mData_Spiky2;
    const float spiky1 = systems[particleId].mData_Spiky1;
    // one load serves both the CUDA's particleContactDistanceInv and invRadius
    const float particleContactDistanceInv =
        systems[particleId].mCommonData_ParticleContactDistanceInv;

    const float friction = mat[PXG_PBD_MAT_F_FRICTION] * mat[PXG_PBD_MAT_F_PARTICLE_FRICTION_SCALE];
    const float cohesion = derived[PXG_PBD_DERIVED_F_COHESION] * scale;
    const float cohesion1 = derived[PXG_PBD_DERIVED_F_COHESION1];
    const float cohesion2 = derived[PXG_PBD_DERIVED_F_COHESION2];
    const float viscosity = mat[PXG_PBD_MAT_F_VISCOSITY] * scale
                          * systems[particleId].mData_InvRestDensity;
    const float adhesion = mat[PXG_PBD_MAT_F_ADHESION] * mat[PXG_PBD_MAT_F_PARTICLE_ADHESION_SCALE];
    const float matSurfaceTension = mat[PXG_PBD_MAT_F_SURFACE_TENSION_RAW];

    const float cohesionRadius = solidRestDistance * mat[PXG_PBD_MAT_F_ADHESION_RADIUS_SCALE];
    const float invAura = 1.0f / (cohesionRadius - solidRestDistance);

    float weight = 0.0f;

    // The CUDA's two-deep prefetch pipeline, un-rolled.  Unlike the density
    // kernel's, this one is CORRECT: all five payloads (position, phase, density,
    // deltaP and the q used for normals) come from the same index at every
    // iteration, so the closed form is simply idx[i].
    for (uint i = 0u; i < contactCount; ++i)
    {
        const uint q = collisionIndex[p + i * numParticles];
        const float4 xj4 = sortedNewPositions[q];
        const uint phase2 = phases[q];
        const float density2 = isFluid ? densities[q] : 0.0f;
        const float4 xjDelta4 = sortedDeltaP[q];
        const float3 xjDelta = float3(xjDelta4.x, xjDelta4.y, xjDelta4.z);

        const float3 xij = xi - float3(xj4.x, xj4.y, xj4.z);
        const float dSq = xij.x * xij.x + xij.y * xij.y + xij.z * xij.z;

        const uint writeOffset = p + i * numParticles;

        if (dSq >= 1e-10f)                                   // S3
        {
            const float d = sqrt(dSq);                       // S2: a REAL sqrt
            const float3 nij = (1.0f / d) * xij;

            if (psGetFluid(phase2) && isFluid)               // S4
            {
                if (dSq < cellSizeSq && dSq > 0.0f)
                {
                    const float lambda = (density + density2) * 0.5f;    // S6
                    const float s = lambda * dWdx(d, spiky2, particleContactDistanceInv);

                    if (phase2 != phase)                     // S5
                    {
                        delta -= s * nij;
                    }
                    else
                    {
                        const float c = cohesion * WCohesion(d, cohesion1, cohesion2,
                                                             particleContactDistanceInv);
                        const float3 relDelta_ij = xiDelta - xjDelta;

                        delta -= nij * (s + c);

                        const float viscosityScale =
                            1.0f - (1.0f / (1.0f + viscosity * W(d, spiky1,
                                                                 particleContactDistanceInv)));
                        delta -= viscosityScale * relDelta_ij;

                        const float relDeltaProj = nij.x * relDelta_ij.x
                                                 + nij.y * relDelta_ij.y
                                                 + nij.z * relDelta_ij.z;
                        if (relDeltaProj < -cflRadius)       // S7
                            delta -= nij * (relDeltaProj + cflRadius) * 0.5f;

                        if (matSurfaceTension > 0.0f)        // S8: the RAW field
                        {
                            const float4 nq4 = normals[q];
                            delta = delta - (ni - float3(nq4.x, nq4.y, nq4.z)) * scale;
                        }
                    }

                    weight += 1.0f;                          // S15
                }
            }
            else
            {
                const float2 force = collisionForces[writeOffset];
                float2 newForce;

                const float w = xi4.w / (xi4.w + xj4.w);     // S9
                const float e = (d - solidRestDistance);
                // the CUDA's shadowing `xij = xjDelta - xiDelta` here is dead

                float adhesionClamp = 0.0f;
                if (d < cohesionRadius)                      // S10
                    adhesionClamp = pxMax(0.0f, adhesion * oct(1.0f - e * invAura) * e);

                const float mtd = pxClamp(pxMin(e, pxMax(-force.x, adhesionClamp)),
                                          -solidRestDistance, solidRestDistance);   // S11

                newForce.x = force.x + mtd;
                newForce.y = force.y;

                if (newForce.x != 0.0f)                      // S12
                {
                    const float deltaF = w * mtd;
                    delta -= nij * deltaF;

                    if (friction > 0.0f)
                    {
                        const float3 j = xiDelta - xjDelta;
                        const float jn = j.x * nij.x + j.y * nij.y + j.z * nij.z;
                        const float3 jt = j - jn * nij;
                        const float jtSq = jt.x * jt.x + jt.y * jt.y + jt.z * jt.z;

                        if (jtSq > 0.0f)
                        {
                            const float len = sqrt(jtSq);
                            const float maxForce = pxMax(-newForce.x, 0.0f) * friction;
                            // the CUDA spells this one as the raw CUDA min(); on
                            // well-formed inputs (both operands non-negative) it
                            // agrees with the PxMin ternary.
                            const float clampedForce = pxMin(maxForce, len + newForce.y);
                            const float deltaForce = clampedForce - newForce.y;   // S13

                            newForce.y += deltaForce;

                            // S14: PxVec3::operator/ is reciprocal-then-multiply,
                            // NOT Slang's per-component divide.
                            delta -= ((jt * deltaForce) * w) * (1.0f / len);
                        }
                    }

                    weight += 1.0f;                          // S15
                }

                newForce.x = pxMin(newForce.x, 0.0f);        // S12: unconditional
                collisionForces[writeOffset] = newForce;
            }
        }
    }

    delta *= gSd.coefficient;                                // S16
    // the CUDA's __syncwarp() here orders nothing; see the header.

    deltas[p] = float4(delta.x, delta.y, delta.z, weight);
}
