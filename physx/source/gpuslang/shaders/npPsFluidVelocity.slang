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

// npPsFluidVelocity.slang
//
// P9.1d -- the PBD VELOCITY pair, the last two kernels of the fluid quartet:
//   ps_vorticityConfinementLaunch (particlesystem.cu:3757-3888) computes the curl
//     of the velocity field per fluid particle into mCurl;
//   ps_solveVelocityLaunch (particlesystem.cu:3891-4041) turns that curl into a
//     vorticity-confinement impulse in mDelta.
// They run back to back on mFinalizeStream, ONCE per frame, from
// integrateSystems -> solveVelocities (PxgPBDParticleSystemCore.cpp:320-361),
// gated only by (getMaxParticles() != 0).  solveVelocity consumes the mCurl that
// vorticityConfinement writes, so they belong in one module and one tranche.
//
// LAUNCHES.  vorticity: args (particleSystems*, activeParticleSystems*) -> {8,8};
// solveVelocity: args (particleSystems*, activeParticleSystems*, PxReal dt) ->
// {8,8,4}.  Both: grid (ceil(getMaxParticles()/256), nbActiveParticleSystems, 1),
// block (PS_SOLVE = 256, 1, 1).
//
// NOTE these two have NO eDISABLE_SELF_COLLISION guard -- unlike the two density
// kernels, they test only the per-particle fluid/vorticity predicates.
//
// ============================================================================
// THE UPSTREAM BUG ps_vorticityConfinementLaunch MUST REPRODUCE (candidate 12)
// ============================================================================
// particlesystem.cu:3853-3864:
//
//     if ((c + 1) < contactCount)
//     {
//         xj4Next  = PxLoad4(sortedNewPositions[nextNextQ]);   // correct
//         nextPhase = sortedPhases[nextNextQ];                 // correct
//         vj4Next  = sortedNewVelocities[nextQ];               // <-- nextQ!
//
//         nextQ = nextNextQ;                                   // bumped AFTER
//         ...
//     }
//
// The velocity prefetch reads nextQ while it still holds this iteration's index;
// the bump to nextNextQ is the next statement.  Writing
// idx[k] = collisionIndex[p + k*numParticles], iteration c consumes
//
//     position = pos[idx[c]]                (correct)
//     phase    = phases[idx[c]]             (correct)
//     velocity = vel[idx[max(c - 1, 0)]]    (lagged for every c >= 1)
//
// so the LAST neighbour's velocity is never read for any particle with two or
// more contacts, and idx[0]'s velocity is consumed twice.  The mismatch is not
// filtered by the guards: both the fluid test and the distance test are computed
// from the CORRECT neighbour c, and the prefetch runs BEFORE the `continue` at
// :3866, so the lag is a lag in RAW NEIGHBOUR-LIST ENTRIES, not in accepted
// neighbours.  The port therefore indexes the velocity by the raw loop counter,
// never by a running count of accepted pairs.
//
// Of the three structurally identical two-deep pipelines in the file (:3548,
// :3853, :3995), line 3857 is the ONLY payload load indexed by nextQ instead of
// nextNextQ -- and :3556 pins the author's intent, because it sits AFTER its own
// nextQ bump and still reads nextNextQ.  ps_solveVelocityLaunch's own pipeline
// below (:3995-4004) prefetches all three payloads from nextNextQ and is correct.
// Verified by six independent refuters before being claimed.  The port
// REPRODUCES it; do not "fix" the max(c - 1, 0) below.
//
// The traps, each with a mutant:
//  V1. vorticity's early-out is (!fluid || vorticityConfinement == 0), a STRICT
//      equality; solveVelocity's is (vorticityConfinement <= 0).  A NEGATIVE
//      vorticityConfinement therefore makes vorticity PROCEED while
//      solveVelocity early-outs.  The two predicates are not interchangeable.
//  V2. Both early-outs WRITE before returning -- curl[p] = 0 and deltas[p] = 0
//      respectively -- rather than leaving the slot alone.
//  V3. Both distance guards are `dSq <= dist2 && dSq > 0`, INCLUSIVE at the top
//      where the two density kernels use a strict `<`.
//  V4. vorticity skips a non-fluid neighbour with `continue` AFTER its prefetch;
//      solveVelocity has no such test and instead requires phase == phasej.
//  V5. `xij / d` is PxVec3::operator/ -- reciprocal-then-multiply (PxVec3.h:219),
//      not Slang's per-component divide.  Both kernels use it.
//  V6. vorticity accumulates vij.cross(dw) with vij = vj - vi, in that order, and
//      stores (vorticity, |vorticity|) -- the w component is the MAGNITUDE.
//  V7. solveVelocity weights the gradient by curl.W of the NEIGHBOUR (the
//      magnitude vorticity stored in .w), not by the neighbour's curl vector.
//  V8. solveVelocity's inner accumulation requires phase == phasej -- the WHOLE
//      phase word, so two fluid particles of the same group but different flag
//      bits do not contribute to each other.
//  V9. SafeNormalize(v, fallback) guards on magnitudeSquared() > 0 and multiplies
//      by InvSqrt = 1.f/sqrtf (particlesystem.cu:79-94); the fallback here is the
//      ZERO vector, so a particle with no in-range same-phase neighbour gets a
//      zero impulse rather than a NaN.
// V10. The impulse is dt * mInvRestDensity * vorticityConfinement * (nGrad x curl),
//      left to right, and the CUDA's own comment at :4034 records that using dt
//      here (rather than dt squared) is a known inconsistency preserved "for
//      compatibility" -- a second deliberate non-fix in this kernel.
// V11. solveVelocity's whole neighbour loop sits inside `if (PxGetFluid(phase))`;
//      a NON-fluid particle whose material has vorticityConfinement > 0 falls
//      through to the store with impulse and weight both zero.
//
// The module is in PXG_NOCONTRACT_SHADERS.

import pxgSlangParticleDesc;

static const uint PS_FLUID_BLOCK = 256u;
static const uint PS_PHASE_FLUID = 0x00400000u;   // 1<<22

// PxsPBDMaterialData float index of vorticityConfinement (byte 28).
static const uint PXG_PBD_MAT_F_VORTICITY = 7u;

float sqr(float x) { return x * x; }
float dWdx(float x, float kSpiky2, float kInvRadius) { return -kSpiky2 * (1.0f - x * kInvRadius); }

uint psGetGroup(uint phase) { return phase & PXG_PARTICLE_PHASE_GROUP_MASK; }
bool psGetFluid(uint phase) { return (phase & PS_PHASE_FLUID) != 0u; }

// phaseToMat[group]: a PxU16 array read as a word plus shift/mask.
uint pvLoadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

// getParticleMaterial with the RUNTIME stride, as both kernels use it.
uint64_t pvMaterialBase(Ptr<PxgParticleSystemGpu> systems, uint id, uint group)
{
    const uint mi = pvLoadPhaseToMaterial(systems[id].mPhaseGroupToMaterialHandle, group);
    return systems[id].mParticleMaterials + uint64_t(mi) * uint64_t(systems[id].mParticleMaterialStride);
}

// PxVec3::cross, written longhand (PxVec3.h:280-283): for each component the
// POSITIVE product is spelled first.
float3 pvCross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

float pvDot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// ===========================================================================
// THE SIGNED-ZERO ACCUMULATOR, AND WHY THE P8.84 REMEDY IS NOT ENOUGH HERE
// ===========================================================================
//
// P8.84 found that slangc folds `x + 0.0` away, so an accumulator seeded from a
// LITERAL zero keeps the sign of a -0 first term where the engine's C++ computes
// 0 + (-0) = +0.  That fires in this kernel: cross(nGrad, curl) with a zeroed
// nGrad produces exact signed zeros, and the first run of this fixture caught 56
// words differing in nothing but the sign of zero.
//
// P8.84's remedy -- seed from a RUNTIME zero the compiler cannot fold -- was
// applied here and DID NOT FIX IT.  Measured, not assumed, by writing the
// intermediates out of the kernel for three failing particles:
//
//     rz                  = 0x00000000   (the runtime zero really is +0)
//     the term before the += = 0x80000000   (really is -0)
//     the accumulator after  = 0x80000000   (the add did not normalise)
//
// and the emitted SPIR-V does contain a genuine `OpFAdd %v3float` over exactly
// those two operands.  A standalone probe (pvSignedZeroProbe, run on this same
// device) confirms the DRIVER is IEEE-correct: (+0) + (-0) == +0 for both scalar
// and vector OpFAdd, and (-0) + (-0) == -0.  So the loss is in slangc's codegen
// for this accumulator shape, not in llvmpipe, and no seeding trick reaches it.
//
// The port therefore spells the identity directly.  `+0 + x` in IEEE-754 is `x`
// for every x EXCEPT -0, which becomes +0 -- which is exactly
// `(x == 0) ? +0 : x`, because `x == 0` is true for both zeros.  A compare and a
// select have no arithmetic identity for a compiler to exploit, so this survives
// any optimisation level and any driver, and it is bit-for-bit the engine's add.
//
// Use it wherever an accumulator that starts at zero is written EXACTLY ONCE.  A
// running loop sum is a different case and keeps the runtime-zero seed: only its
// first added term is at risk, and the curl accumulator below was measured
// bit-exact as it stands.
float pvRuntimeZero(float seed) { return abs(seed) * 0.0f; }

float pvPlusPositiveZero(float x) { return (x == 0.0f) ? 0.0f : x; }
float3 pvPlusPositiveZero3(float3 v)
{
    return float3(pvPlusPositiveZero(v.x), pvPlusPositiveZero(v.y), pvPlusPositiveZero(v.z));
}

// ---------------------------------------------------------------------------
// ps_vorticityConfinementLaunch
// ---------------------------------------------------------------------------
struct PsVorticityParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsVorticityParams> gVc;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_vorticityConfinementLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                   uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gVc.particleSystems);
    const uint particleId = (Ptr<uint>(gVc.activeParticleSystems))[groupId.y];

    const uint numParticles = systems[particleId].mCommonData_NumParticles;
    const uint p = groupId.x * PS_FLUID_BLOCK + groupThreadId.x;

    if (p < numParticles)
    {
        Ptr<float4> sortedNewPositions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
        Ptr<float4> sortedNewVelocities = Ptr<float4>(systems[particleId].mSortedVelocities);
        Ptr<uint> sortedPhases = Ptr<uint>(systems[particleId].mSortedPhaseArray);
        Ptr<uint> collisionIndex = Ptr<uint>(systems[particleId].mCollisionIndex);
        Ptr<float4> curl = Ptr<float4>(systems[particleId].mCurl);

        const uint phase = sortedPhases[p];
        const uint group = psGetGroup(phase);
        const float vorticityConfinement =
            (Ptr<float>(pvMaterialBase(systems, particleId, group)))[PXG_PBD_MAT_F_VORTICITY];

        // V1/V2: STRICT equality here, and the early-out WRITES.
        if (!psGetFluid(phase) || vorticityConfinement == 0.0f)
        {
            curl[p] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            return;
        }

        const float4 xi4 = sortedNewPositions[p];
        const float4 vi4 = sortedNewVelocities[p];
        const float3 xi = float3(xi4.x, xi4.y, xi4.z);
        const float3 vi = float3(vi4.x, vi4.y, vi4.z);

        const uint contactCount = (Ptr<uint>(systems[particleId].mParticleSelfCollisionCount))[p];

        const float rz = pvRuntimeZero(vi4.x);
        float3 vorticity = float3(rz, rz, rz);

        const float dist2 = systems[particleId].mCommonData_ParticleContactDistanceSq;
        const float spiky2 = systems[particleId].mData_Spiky2;
        const float distanceInv = systems[particleId].mCommonData_ParticleContactDistanceInv;

        for (uint c = 0u; c < contactCount; ++c)
        {
            const uint qPos = collisionIndex[p + c * numParticles];
            const float4 cj4 = sortedNewPositions[qPos];
            const uint phasej = sortedPhases[qPos];

            // candidate 12, reproduced: the velocity lags by one RAW neighbour.
            // The index is the loop counter, NOT a count of accepted pairs -- the
            // CUDA's prefetch runs before its `continue`.
            const uint qVel = collisionIndex[p + ((c > 0u) ? (c - 1u) : 0u) * numParticles];
            const float4 vj4 = sortedNewVelocities[qVel];

            if (!psGetFluid(phasej))     // V4: after the (now un-rolled) prefetch
                continue;

            const float3 xj = float3(cj4.x, cj4.y, cj4.z);
            const float3 xij = xi - xj;
            const float dSq = xij.x * xij.x + xij.y * xij.y + xij.z * xij.z;

            if (dSq <= dist2 && dSq > 0.0f)      // V3: INCLUSIVE at the top
            {
                const float3 vj = float3(vj4.x, vj4.y, vj4.z);
                const float3 vij = vj - vi;      // V6

                const float d = sqrt(dSq);
                // V5: PxVec3::operator/ is a reciprocal multiply.
                const float3 dw = dWdx(d, spiky2, distanceInv) * (xij * (1.0f / d));

                vorticity += pvCross(vij, dw);
            }
        }

        // V6: the w component is the MAGNITUDE of the curl.
        curl[p] = float4(vorticity.x, vorticity.y, vorticity.z,
                         sqrt(pvDot(vorticity, vorticity)));
    }
}

// ---------------------------------------------------------------------------
// ps_solveVelocityLaunch
// ---------------------------------------------------------------------------
struct PsSolveVelocityParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8
    float dt;                         // @16 -> 24
    uint _pad20;
}
[[vk::push_constant]]
ConstantBuffer<PsSolveVelocityParams> gSv;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_solveVelocityLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSv.particleSystems);
    const uint particleId = (Ptr<uint>(gSv.activeParticleSystems))[groupId.y];

    const uint numParticles = systems[particleId].mCommonData_NumParticles;
    const uint p = groupId.x * PS_FLUID_BLOCK + groupThreadId.x;

    if (p < numParticles)
    {
        Ptr<float4> sortedNewPositions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
        Ptr<uint> collisionIndex = Ptr<uint>(systems[particleId].mCollisionIndex);
        Ptr<float4> curls = Ptr<float4>(systems[particleId].mCurl);
        Ptr<uint> phases = Ptr<uint>(systems[particleId].mSortedPhaseArray);
        Ptr<float4> deltas = Ptr<float4>(systems[particleId].mDelta);

        const float spiky2 = systems[particleId].mData_Spiky2;

        const uint phase = phases[p];
        const uint group = psGetGroup(phase);
        const float vorticityConfinement =
            (Ptr<float>(pvMaterialBase(systems, particleId, group)))[PXG_PBD_MAT_F_VORTICITY];

        // V1/V2: `<= 0` here, against the vorticity kernel's `== 0`.
        if (vorticityConfinement <= 0.0f)
        {
            deltas[p] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            return;
        }

        const float4 xi4 = sortedNewPositions[p];
        const float3 xi = float3(xi4.x, xi4.y, xi4.z);
        const float4 curl4 = curls[p];

        const float rz = pvRuntimeZero(xi4.x);
        float3 impulse = float3(rz, rz, rz);
        float3 vorticityGrad = float3(rz, rz, rz);
        float weight = 0.0f;
        // `impulse` is written exactly once below, so the whole `+=` collapses to
        // the IEEE identity `+0 + term`; see the note at pvPlusPositiveZero.


        const float contactDistanceSq = systems[particleId].mCommonData_ParticleContactDistanceSq;
        const uint contactCount = (Ptr<uint>(systems[particleId].mParticleSelfCollisionCount))[p];

        if (psGetFluid(phase))                                     // V11
        {
            const float particleContactDistanceInv =
                systems[particleId].mCommonData_ParticleContactDistanceInv;

            // This pipeline is CORRECT in the CUDA -- position, phase and curl all
            // prefetch from nextNextQ -- so the closed form is simply idx[c].
            for (uint c = 0u; c < contactCount; ++c)
            {
                const uint q = collisionIndex[p + c * numParticles];
                const float4 xj4 = sortedNewPositions[q];
                const uint phasej = phases[q];
                const float4 curlN = curls[q];

                const float3 xj = float3(xj4.x, xj4.y, xj4.z);
                const float3 xij = xi - xj;
                const float dSq = xij.x * xij.x + xij.y * xij.y + xij.z * xij.z;

                if (dSq <= contactDistanceSq && dSq > 0.0f)        // V3
                {
                    if (phase == phasej)                           // V8: whole word
                    {
                        const float d = sqrt(dSq);
                        const float3 n = xij * (1.0f / d);         // V5

                        const float3 dw = dWdx(d, spiky2, particleContactDistanceInv) * n;
                        vorticityGrad += curlN.w * dw;             // V7: the .w
                        weight += 1.0f;
                    }
                }
            }

            const float3 curl3 = float3(curl4.x, curl4.y, curl4.z);

            // V9: SafeNormalize guards on the SQUARED magnitude and multiplies by
            // 1/sqrt; the fallback is the zero vector.
            const float gradMagSq = pvDot(vorticityGrad, vorticityGrad);
            float3 nGrad = float3(0.0f, 0.0f, 0.0f);
            if (gradMagSq > 0.0f)
                nGrad = vorticityGrad * (1.0f / sqrt(gradMagSq));

            // V10: the CUDA's comment at :4034 records that dt (rather than dt
            // squared) here is a known inconsistency kept for compatibility.
            impulse = pvPlusPositiveZero3(gSv.dt * systems[particleId].mData_InvRestDensity
                                          * vorticityConfinement * pvCross(nGrad, curl3));
        }

        deltas[p] = float4(impulse.x, impulse.y, impulse.z, weight);
    }
}
