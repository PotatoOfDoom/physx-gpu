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

// npPsInflatable.slang
//
// P9.6 -- the PBD INFLATABLE (pressurised closed cloth) pair:
//   ps_calculateInflatableVolume (particlesystem.cu:4822-5006) computes, per
//     inflatable, its centre of mass and its enclosed signed volume by a
//     BLOCK-COOPERATIVE reduction over the triangles, accumulates a per-vertex
//     normal, and turns the volume error into one XPBD lambda;
//   ps_solveInflatableVolume (particlesystem.cu:5010-5115) pushes every vertex
//     along its normal by that lambda.
//
// A NEW GRID SHAPE, and the sharpest difference from everything else in this
// family: gridDim.X is over INFLATABLES, not over work items.  One BLOCK per
// inflatable, which is what makes the block-wide reduction meaningful.
//     blockIdx.z -> activeParticleSystems[.]
//     blockIdx.y -> the cloth buffer
//     blockIdx.x -> the INFLATABLE (cloth) within that buffer
// block (PS_INFLATABLE = 256, 1, 1).
//
// LAUNCHES (PxgPBDParticleSystemCore.cpp:780-820), both with grid
// (mMaxClothsPerBuffer, mMaxClothBuffersPerSystem, nbActive), gated host-side by
// (mMaxClothsPerBuffer > 0):
//   calculate: args (particleSystems*, activeParticleSystems*)              -> {8,8}
//   solve:     args (particleSystems*, activeParticleSystems*, dt, coeff)   -> {8,8,4,4}
//
// Traps, each with a mutant:
//  I1. THE FILE CONTAINS TWO DEFINITIONS OF ps_calculateInflatableVolume, in an
//      `#if 0 / #else / #endif` pair: the DISABLED one at :4649 and the LIVE one
//      at :4822 (#else :4821, #endif :5007).  A top-down reading ports the wrong
//      kernel entirely.  This module is the :4822 version.
//  I2. `startTriangleIndex` is an INDEX-ARRAY offset, already multiplied by three
//      -- the producer sets it as `nbTriangles * 3` (ExtParticleExt.cpp:571) and
//      both consumers spell `startTriangleIndex + tid * 3` (:4911, :4954).  The
//      header comment calls it "the index of the first triangle"
//      (PxParticleBuffer.h:366), which CONTRADICTS the code; a port that trusts
//      the comment multiplies by three twice.
//  I3. `signedVolume = va.dot(nor)` with NO division by 6 (:4975).  The result is
//      six times the tetrahedron volume, and `restVolume` is in the same units.
//      "Fixing" the factor changes the constraint.
//  I4. The normal fed to `AtomicAdd` is NORMALISED, but `signedVolume` is computed
//      from the UN-normalised `nor` (:4975 before :4977).  The order is
//      load-bearing: normalising first destroys the volume.
//  I5. `comScale = 1 / (3 * numTriangles)` is applied PER TRIANGLE, before the
//      reduction (:4890, :4927).  Scaling the sum afterwards is a different
//      rounding and a different result.
//  I6. The centre of mass is SUBTRACTED from all three vertices in the second
//      pass (:4967-4969) -- a conditioning trick, not decoration.  Dropping it
//      changes the volume for any inflatable not centred on the origin.
//  I7. `lambda = cbrtf(PxMax(0, restVolume*pressure - volume)) * (1/numVertices)
//      * 0.25f` (:4995) -- a CUBE ROOT, a one-sided PxMax, and two separate
//      scales.  See the note on cbrt below.
//  I8. The store is `PxMin(lambda, maxDelta)` (:5000) -- a ONE-SIDED clamp with
//      no lower bound, against `mParticleContactDistance * 0.1f`.
//  I9. Both kernels early-out on `pressure == 0.0f`, which is how the engine marks
//      "solve this as ordinary cloth" (:4866, :5052).  The solve kernel ALSO
//      early-outs on `lambda == 0.f`.
// I10. The two kernels stage the cloth record into shared memory by DIFFERENT
//      mechanisms -- a thread-0 assignment (:4855) versus a `blockCopy<float>`
//      (:5042).  Both are transcribed as written rather than unified.
// I11. `mNormalArray` is indexed by the GLOBAL UNSORTED index in BOTH kernels
//      (:4894, :4979, :5072), while `mAccumDeltaP` uses the SORTED one (:5079).
//      That is consistent here, and it is also the evidence that
//      ps_solveAerodynamics1Launch's SORTED normal write (P9.5, upstream
//      candidate 15) is the outlier rather than the rule.
// I12. `accumDeltaP[ind] += float4(delta, 1.f)` (:5079) is a PLAIN read-modify-
//      write, not an atomic -- and it does not need to be, because the vertex loop
//      is a bijection.  The `.w` gets +1.
//      A CORRECTION to what this comment first claimed: that `+1` is NOT a count
//      anything divides by.  The averaging divisor is `mDelta`'s `.w`
//      (ps_applyDeltaLaunch :3740 computes `1 / (j.w * relaxation)` from
//      `deltas[p]`), and that same kernel then writes `mAccumDeltaP`'s `.w` back as
//      a literal 0 (:3750), as does ps_updateParticleLaunch (:3055).  So the value
//      written here is consumed only in the sense that
//      ps_updateParticleLaunch folds the whole float4 -- `.w` included -- into
//      `mSortedDeltaP` (:3058) and into the velocity via `aDeltaP * invDt` (:3045).
//      The port transcribes the `1.f` because the CUDA writes it, not because a
//      divisor depends on it.
// I13. `dt` is a DEAD PARAMETER of the solve kernel: it is passed
//      (PxgPBDParticleSystemCore.cpp:808) and never read.  Kept in the push layout.
//
// THE CUBE ROOT, stated plainly because it is the first of its kind in this port.
// The CUDA calls `cbrtf` (:4995).  SPIR-V has no cube-root instruction, and there
// is no bit-exact counterpart to reach for: CUDA's device `cbrtf`, glibc's
// `cbrtf` and `pow(x, 1/3)` are three different implementations that agree to a
// few ULP and not to the bit.  Every previous transcendental in this port was
// reproducible because the ENGINE itself used a polynomial approximation
// (npClothAtan is the precedent); this one is a real libm call, so bit-exactness
// against CUDA is not even well defined without a reference GPU.  The module
// spells `pow(x, 1/3)` -- legal because the argument is `PxMax(0, ...)` and so
// never negative -- and the fixture handles the gap by driving the argument to
// EXACTLY ZERO in one case (a negative restVolume makes the PxMax arm fire, and
// every implementation returns 0 for 0, so lambda is compared bit-exactly there),
// with a stated relative tolerance on the cases where it is positive.
//
// THE BLOCK REDUCTION, and one restructuring the platform forces.  reduction.cuh's
// blockReduction (:226-247) is a warp butterfly, one shared slot per warp, then
// warp 0 reduces those slots.  The port has the butterfly already
// (pxgWarpReduceAddF, CUDA-identical shuffle order), but on the emulated path that
// primitive is built from GROUP-WIDE barriers, where CUDA's __shfl_xor_sync is
// warp-local.  So the CUDA's `if (warpIndex == 0) return warpReduction(...)`
// CANNOT be transcribed literally -- only warp 0 would reach the barrier and the
// workgroup would hang.  The port calls the second reduction from EVERY thread and
// selects afterwards; warp 0's inputs and therefore its result are identical, so
// the value is unchanged and the barrier is reached by all.
//
// Every early-out in both kernels is BLOCK-UNIFORM (blockIdx-derived, or read from
// shared memory after a barrier), which is what makes the barrier-based emulation
// legal at all.  A port that made any of them per-thread would deadlock.

#define PXG_WARP_BLOCK 256
#include "pxgSlangWarpInc.slang"

import pxgSlangParticleDesc;
import pxgSlangAtomicFloat;

// PxgParticleSystemKernelBlockDim::PS_INFLATABLE (…KernelIndices.h:48)
static const uint PS_INFLATABLE_BLOCK = 256u;
static const uint INF_WARPS = PS_INFLATABLE_BLOCK / 32u;   // 8

static const uint PXG_CLOTH_SIM_BUFFER_STRIDE = 112u;
static const uint PXG_PARTICLE_CLOTH_STRIDE = 28u;

// ---------------------------------------------------------------------------
// blockReduction<AddOpPxReal, PxReal, 256> (reduction.cuh:226-247)
// ---------------------------------------------------------------------------
groupshared float gInfWarpSum[INF_WARPS];

float infBlockReduceAdd(uint tid, float value, float initialValue)
{
    const uint warpIndex = tid / 32u;
    const uint laneIndex = tid & 31u;

    const float warpResult = pxgWarpReduceAddF(0xFFFFFFFFu, tid, value, 5u);

    if (laneIndex == 0u)
        gInfWarpSum[warpIndex] = warpResult;
    GroupMemoryBarrierWithGroupSync();

    // reduction.cuh:241 indexes the scratch by threadIndexInWarp, NOT warpIndex.
    // For warp 0 -- the only warp whose result is used -- the two coincide.
    const float val = (tid < INF_WARPS) ? gInfWarpSum[laneIndex] : initialValue;

    // Called from EVERY thread; see the module header.  Warp 0's inputs are
    // unchanged, so its result is the CUDA's.
    const float reduced = pxgWarpReduceAddF(0xFFFFFFFFu, tid, val, 5u);

    GroupMemoryBarrierWithGroupSync();   // the scratch is reusable next pass
    return (warpIndex == 0u) ? reduced : initialValue;
}

// I7: no SPIR-V cube root; the argument is guaranteed non-negative by PxMax.
float infCbrt(float x) { return pow(x, 1.0f / 3.0f); }

float infMax(float a, float b) { return a > b ? a : b; }
float infMin(float a, float b) { return a < b ? a : b; }

float3 infCross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}
float infDot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxVec3::normalize() (PxVec3.h:299-305): returns the magnitude and scales in
// place only when m > 0, by a RECIPROCAL multiply.
float3 infNormalize(float3 v)
{
    const float m = sqrt(infDot(v, v));
    return (m > 0.0f) ? (v * (1.0f / m)) : v;
}

uint64_t infClothBufferAddr(Ptr<PxgParticleSystemGpu> systems, uint id, uint bufferIndex)
{
    return systems[id].mClothSimBuffers + uint64_t(bufferIndex) * uint64_t(PXG_CLOTH_SIM_BUFFER_STRIDE);
}

// The staged cloth record.  PxParticleCloth is 28 bytes / seven 4-byte members.
groupshared uint gInfStartVertexIndex;
groupshared uint gInfNumVertices;
groupshared float gInfClothBlendScale;
groupshared float gInfRestVolume;
groupshared float gInfPressure;
groupshared uint gInfStartTriangleIndex;
groupshared uint gInfNumTriangles;
groupshared float gInfVolume;
groupshared float gInfCom0, gInfCom1, gInfCom2;

// ============================================================================
// ps_calculateInflatableVolume -- one block per inflatable
// ============================================================================
struct PsInflatableCalcParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsInflatableCalcParams> gIc;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_calculateInflatableVolume(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gIc.particleSystems);
    const uint particleId = (Ptr<uint>(gIc.activeParticleSystems))[groupId.z];
    const uint tid = groupThreadId.x;

    const uint bufferIndex = groupId.y;

    if (bufferIndex < systems[particleId].mNumClothBuffers)
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(infClothBufferAddr(systems, particleId, bufferIndex));

        const uint numInflatables = cloth[0].mNumCloths;
        const uint inflatableIdx = groupId.x;

        if (inflatableIdx < numInflatables)      // block-uniform
        {
            Ptr<PxParticleClothGpu> inflatables = Ptr<PxParticleClothGpu>(cloth[0].mCloths);

            // I10: a THREAD-0 assignment here (the solve kernel uses blockCopy).
            if (tid == 0u)
            {
                const PxParticleClothGpu rec = inflatables[inflatableIdx];
                gInfStartVertexIndex = rec.startVertexIndex;
                gInfNumVertices = rec.numVertices;
                gInfClothBlendScale = rec.clothBlendScale;
                gInfRestVolume = rec.restVolume;
                gInfPressure = rec.pressure;
                gInfStartTriangleIndex = rec.startTriangleIndex;
                gInfNumTriangles = rec.numTriangles;
                gInfVolume = 0.0f;
                gInfCom0 = 0.0f;
                gInfCom1 = 0.0f;
                gInfCom2 = 0.0f;
            }
            GroupMemoryBarrierWithGroupSync();

            // I9: zero pressure marks "solve as ordinary cloth".  Block-uniform.
            if (gInfPressure != 0.0f)
            {
                Ptr<float4> positions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
                Ptr<uint> reverseLookup = Ptr<uint>(systems[particleId].mUnsortedToSortedMapping);
                Ptr<uint> indices = Ptr<uint>(cloth[0].mTriangles);

                const uint particleBufferIndex = cloth[0].mParticleBufferIndex;
                const uint offset =
                    (Ptr<uint>(systems[particleId].mParticleBufferRunsum))[particleBufferIndex];

                Ptr<float> lambdas = Ptr<float>(cloth[0].mInflatableLambda);

                const uint numTriangles = gInfNumTriangles;
                const uint numPasses = (numTriangles + PS_INFLATABLE_BLOCK - 1u) / PS_INFLATABLE_BLOCK;

                // I5: the scale is per triangle, applied BEFORE the reduction.
                const float comScale = 1.0f / (3.0f * float(numTriangles));

                const uint64_t normalsBase = systems[particleId].mNormalArray;
                Ptr<float4> normals = Ptr<float4>(normalsBase);

                // I11: the GLOBAL UNSORTED index.
                const uint vertStartIndex = gInfStartVertexIndex + offset;
                const uint vertEndIndex = vertStartIndex + gInfNumVertices;
                for (uint i = tid + vertStartIndex; i < vertEndIndex; i += PS_INFLATABLE_BLOCK)
                    normals[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);

                GroupMemoryBarrierWithGroupSync();

                // ---- pass 1: the centre of mass ----
                for (uint pass = 0u; pass < numPasses; ++pass)
                {
                    const uint t = pass * PS_INFLATABLE_BLOCK + tid;

                    float3 tc = float3(0.0f, 0.0f, 0.0f);
                    if (t < numTriangles)
                    {
                        // I2: startTriangleIndex is ALREADY an index-array offset.
                        const uint tri = gInfStartTriangleIndex + t * 3u;
                        const uint a = reverseLookup[indices[tri + 0u] + offset];
                        const uint b = reverseLookup[indices[tri + 1u] + offset];
                        const uint c = reverseLookup[indices[tri + 2u] + offset];
                        const float4 pa = positions[a];
                        const float4 pb = positions[b];
                        const float4 pc = positions[c];
                        const float3 va = float3(pa.x, pa.y, pa.z);
                        const float3 vb = float3(pb.x, pb.y, pb.z);
                        const float3 vc = float3(pc.x, pc.y, pc.z);
                        tc = (va + vb + vc) * comScale;
                    }

                    const float bx = infBlockReduceAdd(tid, tc.x, 0.0f);
                    const float by = infBlockReduceAdd(tid, tc.y, 0.0f);
                    const float bz = infBlockReduceAdd(tid, tc.z, 0.0f);

                    if (tid == 0u)
                    {
                        gInfCom0 = gInfCom0 + bx;
                        gInfCom1 = gInfCom1 + by;
                        gInfCom2 = gInfCom2 + bz;
                    }
                    GroupMemoryBarrierWithGroupSync();
                }

                // ---- pass 2: the signed volume and the vertex normals ----
                for (uint pass = 0u; pass < numPasses; ++pass)
                {
                    const uint t = pass * PS_INFLATABLE_BLOCK + tid;

                    float signedVolume = 0.0f;
                    if (t < numTriangles)
                    {
                        const uint tri = gInfStartTriangleIndex + t * 3u;
                        const uint unsortedA = indices[tri + 0u] + offset;
                        const uint unsortedB = indices[tri + 1u] + offset;
                        const uint unsortedC = indices[tri + 2u] + offset;
                        const uint a = reverseLookup[unsortedA];
                        const uint b = reverseLookup[unsortedB];
                        const uint c = reverseLookup[unsortedC];

                        const float3 com = float3(gInfCom0, gInfCom1, gInfCom2);
                        const float4 pa = positions[a];
                        const float4 pb = positions[b];
                        const float4 pc = positions[c];
                        // I6: relative to the centre of mass.
                        const float3 va = float3(pa.x, pa.y, pa.z) - com;
                        const float3 vb = float3(pb.x, pb.y, pb.z) - com;
                        const float3 vc = float3(pc.x, pc.y, pc.z) - com;

                        float3 nor = infCross(vb - va, vc - va);

                        // I3/I4: the volume uses the UN-normalised normal, and
                        // there is no division by six.
                        signedVolume = infDot(va, nor);

                        nor = infNormalize(nor);

                        pxgAtomicAddF4X(normalsBase, unsortedA, nor.x);
                        pxgAtomicAddF4Y(normalsBase, unsortedA, nor.y);
                        pxgAtomicAddF4Z(normalsBase, unsortedA, nor.z);
                        pxgAtomicAddF4X(normalsBase, unsortedB, nor.x);
                        pxgAtomicAddF4Y(normalsBase, unsortedB, nor.y);
                        pxgAtomicAddF4Z(normalsBase, unsortedB, nor.z);
                        pxgAtomicAddF4X(normalsBase, unsortedC, nor.x);
                        pxgAtomicAddF4Y(normalsBase, unsortedC, nor.y);
                        pxgAtomicAddF4Z(normalsBase, unsortedC, nor.z);
                    }

                    const float blockVolume = infBlockReduceAdd(tid, signedVolume, 0.0f);

                    if (tid == 0u)
                        gInfVolume = gInfVolume + blockVolume;
                    GroupMemoryBarrierWithGroupSync();
                }

                if (tid == 0u)
                {
                    const float constraintScale = 1.0f / float(gInfNumVertices);
                    // I7: cube root, one-sided PxMax, two separate scales.
                    const float lambda =
                        infCbrt(infMax(0.0f, gInfRestVolume * gInfPressure - gInfVolume))
                        * constraintScale * 0.25f;

                    const float maxDelta =
                        systems[particleId].mCommonData_ParticleContactDistance * 0.1f;

                    // I8: a ONE-SIDED clamp.
                    lambdas[inflatableIdx] = infMin(lambda, maxDelta);
                }
            }
        }
    }
}

// ============================================================================
// ps_solveInflatableVolume -- push every vertex along its normal
// ============================================================================
struct PsInflatableSolveParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8
    float dt;                         // @16 -- I13: DEAD, never read
    float coefficient;                // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsInflatableSolveParams> gIs;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_solveInflatableVolume(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gIs.particleSystems);
    const uint particleId = (Ptr<uint>(gIs.activeParticleSystems))[groupId.z];
    const uint tid = groupThreadId.x;

    const uint bufferIndex = groupId.y;

    if (bufferIndex < systems[particleId].mNumClothBuffers)
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(infClothBufferAddr(systems, particleId, bufferIndex));

        const uint numInflatables = cloth[0].mNumCloths;
        const uint inflatableIdx = groupId.x;

        if (inflatableIdx < numInflatables)      // block-uniform
        {
            Ptr<PxParticleClothGpu> inflatables = Ptr<PxParticleClothGpu>(cloth[0].mCloths);

            // I10: a blockCopy here, where the calculate kernel assigns from
            // thread 0.  Seven 4-byte members; with a 256-thread block the copy is
            // a single strided pass and every lane past the seventh does nothing.
            if (tid == 0u)
            {
                const PxParticleClothGpu rec = inflatables[inflatableIdx];
                gInfStartVertexIndex = rec.startVertexIndex;
                gInfNumVertices = rec.numVertices;
                gInfClothBlendScale = rec.clothBlendScale;
                gInfRestVolume = rec.restVolume;
                gInfPressure = rec.pressure;
                gInfStartTriangleIndex = rec.startTriangleIndex;
                gInfNumTriangles = rec.numTriangles;
            }
            GroupMemoryBarrierWithGroupSync();

            Ptr<float> lambdas = Ptr<float>(cloth[0].mInflatableLambda);
            const float lambda = lambdas[inflatableIdx];

            // I9: both conditions, and both block-uniform.
            if (gInfPressure != 0.0f && lambda != 0.0f)
            {
                const uint particleBufferIndex = cloth[0].mParticleBufferIndex;
                const uint offset =
                    (Ptr<uint>(systems[particleId].mParticleBufferRunsum))[particleBufferIndex];

                Ptr<uint> reverseLookup = Ptr<uint>(systems[particleId].mUnsortedToSortedMapping);
                Ptr<float4> accumDeltaP = Ptr<float4>(systems[particleId].mAccumDeltaP);
                Ptr<float4> normals = Ptr<float4>(systems[particleId].mNormalArray);

                const uint numVertices = gInfNumVertices;
                const uint vertBase = gInfStartVertexIndex + offset;

                for (uint t = tid; t < numVertices; t += PS_INFLATABLE_BLOCK)
                {
                    const uint vertId = t + vertBase;          // I11: UNSORTED

                    const float4 nor4 = normals[vertId];
                    const float3 normal = infNormalize(float3(nor4.x, nor4.y, nor4.z));

                    const uint ind = reverseLookup[vertId];    // I11: SORTED

                    const float3 delta = normal * lambda * gIs.coefficient;

                    // I12: a PLAIN read-modify-write -- the loop is a bijection --
                    // and the .w counts the contribution.
                    const float4 prev = accumDeltaP[ind];
                    accumDeltaP[ind] = float4(prev.x + delta.x,
                                              prev.y + delta.y,
                                              prev.z + delta.z,
                                              prev.w + 1.0f);
                }
            }
        }
    }
}
