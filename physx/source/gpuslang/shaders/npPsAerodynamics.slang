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

// npPsAerodynamics.slang
//
// P9.5 -- the PBD CLOTH AERODYNAMICS pair:
//   ps_solveAerodynamics1Launch (particlesystem.cu:4437-4562) runs ONE THREAD PER
//     TRIANGLE and, for each of the triangle's three vertices in turn, accumulates
//     a face normal and an aerodynamic impulse (drag + lift) ATOMICALLY into the
//     per-particle mNormalArray and mDelta;
//   ps_solveAerodynamics2Launch (particlesystem.cu:4564-4646) runs ONE THREAD PER
//     PARTICLE, divides the accumulated impulse by the count kept in mDelta.w,
//     folds it into mSortedVelocities, clears mDelta, and normalises mNormalArray.
//
// The pair is a classic scatter/gather: kernel 1 fans triangle work out onto
// vertices with float atomics, kernel 2 collapses it.  mDelta.w is the fan-in
// COUNT -- kernel 1 adds 1.0 to it per contribution, kernel 2 divides by it.
//
// LAUNCHES (PxgPBDParticleSystemCore.cpp:856-909), both {8,8,4}:
//   aero1: grid (ceil(maxTriangles / PS_SOLVE), maxClothBuffers, nbActive)
//   aero2: grid (ceil(maxParticles  / PS_SOLVE), maxBUFFERS,     nbActive)
// block (PS_SOLVE = 256, 1, 1) for both, on mFinalizeStream, gated host-side by
// (mMaxTrianglesPerBuffer > 0).
//
// Note aero2's gridDim.Y: it is mMaxBuffersPerSystem (the PARTICLE buffer count),
// NOT mMaxClothBuffersPerSystem, even though the body guards on mNumClothBuffers
// and indexes mClothSimBuffers.  Every cloth buffer is a particle buffer, so
// maxBuffers >= maxClothBuffers and the extra blocks are simply rejected by the
// guard -- wasteful, not wrong.  The port keeps both grids as the host spells
// them rather than unifying them.
//
// Traps, each with a mutant:
//  Y1. Kernel 1's outer body is a LOOP OVER THE THREE VERTICES, and it ROTATES the
//      triangle each time: ind0/ind1/ind2 are triangles[base + i], [base + (i+1)%3]
//      and [base + (i+2)%3].  So every iteration computes a normal for a DIFFERENT
//      vertex from a rotated winding -- it is not the same normal scattered three
//      times.  A port that hoists the normal out of the loop is wrong.
//  Y2. The face normal is `e2.cross(e1)`, NOT e1.cross(e2) -- the opposite sign
//      from the obvious spelling.
//  Y3. The velocity average uses the LITERAL 0.3333f, which is NOT 1.0f/3.0f
//      (0.33333334f).  They differ in the last mantissa bits and the result is
//      visible after the clamp and the magnitude multiply.
//  Y4. `AtomicAdd(normals, a, n)` is the THREE-argument form: it touches x, y, z
//      and leaves .w ALONE (atomic.cuh:83-88).  The delta add is the FOUR-argument
//      form and adds 1.0 to .w (atomic.cuh:67-73) -- that is the fan-in counter
//      kernel 2 divides by.  Using the wrong overload for either breaks the other
//      kernel.
//  Y5. The normal accumulation is UNCONDITIONAL; only the impulse is gated by
//      calculateDelta.  A port that puts both inside the gate silently loses the
//      normals whenever lift and drag are zero.
//  Y6. calculateDelta = (drag != 0 || lift != 0) && (p14.w != 0).  It is an
//      inequality against zero on BOTH material coefficients, and a FIXED particle
//      (invMass 0) is excluded.
//  Y7. cosTheta < 0 flips BOTH the normal and cosTheta -- the normal is used
//      afterwards for liftDir, so flipping only the scalar is wrong.
//  Y8. dragFactor is clamped to [-0.75, 0.75] and the lift coefficient to
//      [-1, 1], and the lift clamp is applied BEFORE multiplying by
//      |vrel| -- clamping the product instead is a different function.
//  Y9. Two DIFFERENT normalisations with different fallbacks.  getNormalized()
//      (PxVec3.h:286-290) falls back to the ZERO vector; SafeNormalize(v, fallback)
//      (particlesystem.cu:84-94) falls back to the CALLER's vector, which in
//      kernel 2 is (0,1,0) and is then NEGATED to (0,-1,0).  Both guard on
//      magnitudeSquared() > 0 and both multiply by 1/sqrt (PxRecipSqrt and InvSqrt
//      are both `1.0f / sqrtf` here -- PxUnixMathIntrinsics.h:93-96,
//      particlesystem.cu:78-81), so only the fallbacks differ.
// Y10. The two kernels read the phase from DIFFERENT arrays in DIFFERENT index
//      spaces: kernel 1 from mSortedPhaseArray[a] (SORTED), kernel 2 from
//      particleBuffer.mPhases[globalThreadIndex] (UNSORTED and BUFFER-LOCAL).
//      Transcribed as written.
// Y11. Kernel 2 clears mDelta UNCONDITIONALLY -- outside the `calculateDelta &&
//      delta.w > 0` branch -- and writes .w = 0 into the velocity it stores.
// Y12. UPSTREAM CANDIDATE 15, transcribed as-is.  Kernel 1 accumulates the normal
//      at the SORTED index `a`; kernel 2 reads and writes mNormalArray at the
//      UNSORTED index, while using the SORTED index for mDelta and mSortedVelocities
//      in the very same lines.  The two index spaces coincide only when the sort is
//      the identity.  This port reproduces the CUDA exactly, including this; the
//      standing rule is bit-exactness with the engine, bugs included.  Do NOT
//      "fix" it -- a fixed port would diverge from CUDA on every scene where the
//      sort is a real permutation.
//
// The module is in PXG_NOCONTRACT_SHADERS: it is dense float work and FMA
// contraction moves the result off the engine's bit pattern.

import pxgSlangParticleDesc;
import pxgSlangAtomicFloat;

// PxgParticleSystemKernelBlockDim::PS_SOLVE (PxgParticleSystemCoreKernelIndices.h)
static const uint PS_SOLVE_BLOCK = 256u;

static const uint PXG_CLOTH_SIM_BUFFER_STRIDE = 112u;
static const uint PXG_PARTICLE_SIM_BUFFER_STRIDE = 88u;

// PxsPBDMaterialData float indices: lift is byte 40, drag byte 44.
static const uint PXG_PBD_MAT_F_LIFT = 10u;
static const uint PXG_PBD_MAT_F_DRAG = 11u;

// argSizes for both: { particleSystems*, activeParticleSystems*, PxReal dt }
// -> {8,8,4} = 20, rounded to the 8-alignment -> 24.
struct PsAeroParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8
    float dt;                         // @16
    uint _pad20;                      // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsAeroParams> gAe;

uint64_t aeClothBufferAddr(Ptr<PxgParticleSystemGpu> systems, uint id, uint bufferIndex)
{
    return systems[id].mClothSimBuffers + uint64_t(bufferIndex) * uint64_t(PXG_CLOTH_SIM_BUFFER_STRIDE);
}
uint64_t aeParticleBufferAddr(Ptr<PxgParticleSystemGpu> systems, uint id, uint bufferIndex)
{
    return systems[id].mParticleSimBuffers + uint64_t(bufferIndex) * uint64_t(PXG_PARTICLE_SIM_BUFFER_STRIDE);
}

uint aeGetGroup(uint phase) { return phase & PXG_PARTICLE_PHASE_GROUP_MASK; }

// phaseToMat[group]: a PxU16 array read as a word plus shift/mask.
uint aeLoadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

uint64_t aeMaterialBase(Ptr<PxgParticleSystemGpu> systems, uint id, uint group)
{
    const uint mi = aeLoadPhaseToMaterial(systems[id].mPhaseGroupToMaterialHandle, group);
    return systems[id].mParticleMaterials + uint64_t(mi) * uint64_t(systems[id].mParticleMaterialStride);
}

// PxVec3::cross (PxVec3.h:280-283): for each component the POSITIVE product first.
float3 aeCross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}
float aeDot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float aeMagnitude(float3 v) { return sqrt(aeDot(v, v)); }

// Y9a: PxVec3::getNormalized() -- guards magnitudeSquared() > 0, ZERO fallback.
float3 aeNormalized(float3 v)
{
    const float m = aeDot(v, v);
    return (m > 0.0f) ? (v * (1.0f / sqrt(m))) : float3(0.0f, 0.0f, 0.0f);
}

// Y9b: SafeNormalize(v, fallback) -- same guard, CALLER's fallback.
float3 aeSafeNormalize(float3 v, float3 fallback)
{
    const float l = aeDot(v, v);
    return (l > 0.0f) ? (v * (1.0f / sqrt(l))) : fallback;
}

// PxMax/PxMin for float are intrinsics::selectMax/selectMin -- plain ternaries,
// NOT fmaxf/fminf (PxUnixMathIntrinsics.h:116-125).  PxClamp is
// PxMin(hi, PxMax(lo, v)) exactly.
float aeMax(float a, float b) { return a > b ? a : b; }
float aeMin(float a, float b) { return a < b ? a : b; }
float aeClamp(float v, float lo, float hi) { return aeMin(hi, aeMax(lo, v)); }

// ============================================================================
// ps_solveAerodynamics1Launch -- one thread per triangle, three vertices each.
// ============================================================================
[shader("compute")]
[numthreads(256, 1, 1)]
void ps_solveAerodynamics1Launch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gAe.particleSystems);
    const uint particleId = (Ptr<uint>(gAe.activeParticleSystems))[groupId.z];

    const uint bufferIndex = groupId.y;

    if (bufferIndex < systems[particleId].mNumClothBuffers)
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(aeClothBufferAddr(systems, particleId, bufferIndex));

        const uint particleBufferIndex = cloth[0].mParticleBufferIndex;
        const uint offset =
            (Ptr<uint>(systems[particleId].mParticleBufferRunsum))[particleBufferIndex];

        const uint nbTriangles = cloth[0].mNumTriangles;
        Ptr<uint> triangles = Ptr<uint>(cloth[0].mTriangles);

        Ptr<uint> reverseLookup = Ptr<uint>(systems[particleId].mUnsortedToSortedMapping);
        Ptr<float4> sortedPosIM = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
        Ptr<uint> phases = Ptr<uint>(systems[particleId].mSortedPhaseArray);   // Y10: SORTED

        Ptr<float4> sortedVel = Ptr<float4>(systems[particleId].mSortedVelocities);

        const uint64_t normalsBase = systems[particleId].mNormalArray;
        const uint64_t deltaBase = systems[particleId].mDelta;

        const float3 wind = float3(systems[particleId].mData_WindX,
                                   systems[particleId].mData_WindY,
                                   systems[particleId].mData_WindZ);

        const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_SOLVE_BLOCK;

        if (groupThreadIdx < nbTriangles)
        {
            const uint triBase = groupThreadIdx * 3u;

            // Y1: three iterations, each ROTATING which vertex is "a".
            for (uint i = 0u; i < 3u; ++i)
            {
                const uint ind0 = triangles[triBase + (i % 3u)] + offset;
                const uint ind1 = triangles[triBase + ((i + 1u) % 3u)] + offset;
                const uint ind2 = triangles[triBase + ((i + 2u) % 3u)] + offset;

                const uint a = reverseLookup[ind0];
                const float4 p14 = sortedPosIM[a];
                const float3 p1 = float3(p14.x, p14.y, p14.z);

                const float4 va4 = sortedVel[a];
                const float3 va = float3(va4.x, va4.y, va4.z);

                const uint phase = phases[a];
                const uint group = aeGetGroup(phase);
                Ptr<float> mat = Ptr<float>(aeMaterialBase(systems, particleId, group));
                const float matLift = mat[PXG_PBD_MAT_F_LIFT];
                const float matDrag = mat[PXG_PBD_MAT_F_DRAG];

                // Y6: an inequality on BOTH coefficients, and fixed particles are out.
                const bool calculateDelta = ((matDrag != 0.0f) || (matLift != 0.0f))
                                         && (p14.w != 0.0f);

                const uint b = reverseLookup[ind1];
                const uint c = reverseLookup[ind2];

                const float4 p24 = sortedPosIM[b];
                const float4 p34 = sortedPosIM[c];
                const float3 p2 = float3(p24.x, p24.y, p24.z);
                const float3 p3 = float3(p34.x, p34.y, p34.z);

                const float4 vb4 = sortedVel[b];
                const float4 vc4 = sortedVel[c];
                const float3 vb = float3(vb4.x, vb4.y, vb4.z);
                const float3 vc = float3(vc4.x, vc4.y, vc4.z);

                const float3 e1 = p2 - p1;
                const float3 e2 = p3 - p1;
                float3 n = aeNormalized(aeCross(e2, e1));   // Y2: e2 x e1

                // Y5/Y12: UNCONDITIONAL, and at the SORTED index a.
                pxgAtomicAddF4X(normalsBase, a, n.x);
                pxgAtomicAddF4Y(normalsBase, a, n.y);
                pxgAtomicAddF4Z(normalsBase, a, n.z);

                if (calculateDelta)
                {
                    // Y3: the LITERAL 0.3333f, not 1/3.
                    const float3 v = (va + vb + vc) * 0.3333f;
                    const float3 vrel = wind - v;
                    const float3 vdir = aeNormalized(vrel);
                    float cosTheta = aeDot(vdir, n);

                    if (cosTheta < 0.0f)                     // Y7: flip BOTH
                    {
                        n = n * -1.0f;
                        cosTheta = cosTheta * -1.0f;
                    }

                    // Y8: clamp the FACTOR, then multiply.
                    const float dragFactor =
                        aeClamp(gAe.dt * matDrag * cosTheta * p14.w, -0.75f, 0.75f);
                    const float3 drag = dragFactor * vrel;

                    const float liftCoefficient =
                        aeClamp(p14.w * gAe.dt * matLift * cosTheta
                                    * sqrt(aeMax(0.0f, 1.0f - (cosTheta * cosTheta))),
                                -1.0f, 1.0f)
                        * aeMagnitude(vrel);
                    const float3 liftDir = aeNormalized(aeCross(aeCross(vrel, n), vrel));

                    const float3 j = drag + liftDir * liftCoefficient;

                    // Y4: the FOUR-argument form -- .w counts the contributions.
                    pxgAtomicAddF4X(deltaBase, a, j.x);
                    pxgAtomicAddF4Y(deltaBase, a, j.y);
                    pxgAtomicAddF4Z(deltaBase, a, j.z);
                    pxgAtomicAddF4W(deltaBase, a, 1.0f);
                }
            }
        }
    }
}

// ============================================================================
// ps_solveAerodynamics2Launch -- one thread per particle, the gather.
// ============================================================================
[shader("compute")]
[numthreads(256, 1, 1)]
void ps_solveAerodynamics2Launch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gAe.particleSystems);
    const uint particleId = (Ptr<uint>(gAe.activeParticleSystems))[groupId.z];

    const uint bufferIndex = groupId.y;

    if (bufferIndex < systems[particleId].mNumClothBuffers)
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(aeClothBufferAddr(systems, particleId, bufferIndex));

        const uint particleBufferIndex = cloth[0].mParticleBufferIndex;
        Ptr<PxgParticleSimBufferGpu> particleBuffer =
            Ptr<PxgParticleSimBufferGpu>(aeParticleBufferAddr(systems, particleId, particleBufferIndex));

        // Y10: UNSORTED phases, indexed BUFFER-LOCALLY.
        Ptr<uint> phases = Ptr<uint>(particleBuffer[0].mPhases);

        Ptr<uint> reverseLookup = Ptr<uint>(systems[particleId].mUnsortedToSortedMapping);
        Ptr<float4> sortedPosIM = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
        Ptr<float4> sortedVel = Ptr<float4>(systems[particleId].mSortedVelocities);
        Ptr<float4> normals = Ptr<float4>(systems[particleId].mNormalArray);
        Ptr<float4> deltas = Ptr<float4>(systems[particleId].mDelta);

        const uint numParticles = particleBuffer[0].mNumActiveParticles;

        const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_SOLVE_BLOCK;
        const uint offset =
            (Ptr<uint>(systems[particleId].mParticleBufferRunsum))[particleBufferIndex];

        if (globalThreadIndex < numParticles)
        {
            const uint unsortedIndex = globalThreadIndex + offset;

            const uint phase = phases[globalThreadIndex];
            const uint group = aeGetGroup(phase);
            Ptr<float> mat = Ptr<float>(aeMaterialBase(systems, particleId, group));
            const float matLift = mat[PXG_PBD_MAT_F_LIFT];
            const float matDrag = mat[PXG_PBD_MAT_F_DRAG];

            const uint index = reverseLookup[unsortedIndex];

            const float4 p14 = sortedPosIM[index];

            const bool calculateDelta = ((matDrag != 0.0f) || (matLift != 0.0f))
                                     && (p14.w != 0.0f);

            const float4 delta = deltas[index];

            if (calculateDelta && delta.w > 0.0f)
            {
                const float4 va4 = sortedVel[index];
                const float scale = 1.0f / delta.w;
                sortedVel[index] = float4(va4.x + delta.x * scale,
                                          va4.y + delta.y * scale,
                                          va4.z + delta.z * scale,
                                          0.0f);                       // Y11: .w = 0
            }

            // Y11: the clear is UNCONDITIONAL.
            deltas[index] = float4(0.0f, 0.0f, 0.0f, 0.0f);

            // Y12: the UNSORTED index, exactly as the CUDA spells it, even though
            // kernel 1 accumulated at the SORTED one.
            const float4 n4 = normals[unsortedIndex];
            const float3 n = float3(n4.x, n4.y, n4.z);

            // Y9b: fallback (0,1,0), then NEGATED -> (0,-1,0).
            const float3 unitn = -aeSafeNormalize(n, float3(0.0f, 1.0f, 0.0f));

            normals[unsortedIndex] = float4(unitn.x, unitn.y, unitn.z, 0.0f);
        }
    }
}
