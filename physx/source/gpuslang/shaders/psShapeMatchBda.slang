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

// psShapeMatchBda.slang -- P9.80.
//
//   ps_solveShapes (particlesystem.cu:5121) on the engine launch ABI.
//
// PBD SHAPE MATCHING: per PxParticleRigidBuffer rigid, fit a rotation and a
// translation to the deformed particle cluster (Mueller's iterative polar
// decomposition of the moment matrix) and lerp every particle back toward that
// rigid pose.  This is what makes a cluster of particles behave as a rigid body.
//
// ---------------------------------------------------------------------------
// WHY THIS WAS NEVER LAUNCHED
// ---------------------------------------------------------------------------
// Its host guard is `if (maxRigids > 0)` with maxRigids = getMaxRigidsPerBuffer()
// (PxgPBDParticleSystemCore.cpp:825), and no scene in this harness had ever
// created a PxParticleRigidBuffer.  PXG_SCENE=particlerigid (P9.80a) is the first,
// and with this kernel skipped it measured the cluster collapsing to a y-extent of
// 0.0000 -- 64 particles flat on the floor -- against a spawn extent of 1.5000.
//
// ---------------------------------------------------------------------------
// THE REDUCTION, AND WHY IT CANNOT BORROW psBoundsBda's SHAPE
// ---------------------------------------------------------------------------
// This kernel calls blockReduction<AddOpPxReal, PxReal, PS_SOLVE_SHAPE> twelve
// times per pass (three for the centre of mass, nine for the moment matrix).
// psBoundsBda.slang reduces with a full-block strided tree and says, correctly,
// that min/max is associative and rounding-free so the order does not matter.
// FOR A SUM THAT ARGUMENT DOES NOT HOLD: float addition is not associative, so the
// CUDA's association tree has to be reproduced exactly or the result differs in the
// last bits -- and this port is compared bit-for-bit against a host transcription.
//
// reduction.cuh:226-247, transcribed rather than paraphrased:
//
//   * LEVEL 1 is warpReduction<Add>: an XOR butterfly with radius 16, 8, 4, 2, 1 --
//     LARGEST STRIDE FIRST.  (shuffle.cuh's warpShuffleMin/Max run the opposite
//     order; harmless for min/max, a different tree for add.)  Every lane of the
//     warp ends with the same bits.
//   * lane 0 of each warp stores to a per-warp scratch slot; ONE __syncthreads().
//   * LEVEL 2 loads `(threadIdx.x < numWarpsPerBlock) ? scratch[threadIndexInWarp]
//     : initialValue` and runs the SAME full 32-lane butterfly over it.  With
//     PS_SOLVE_SHAPE = 64 there are TWO real warp sums and THIRTY lanes padded with
//     +0.0f, and those padding additions are real: `-0.0f + 0.0f` is `+0.0f`, so
//     the sign of a zero result is genuinely decided by them.
//   * blockReduction returns the sum ONLY to the 32 lanes of warp 0.  Every other
//     thread gets `initialValue` back.  Every CUDA call site here guards with
//     `if (threadIdx.x == 0)`, so the difference is invisible -- but a port that
//     broadcast to all threads would be wrong the day a call site stops guarding.
//
// TWO THINGS THE EMULATED PATH MUST DO DIFFERENTLY, both inherited from
// npPsInflatable.slang's infBlockReduceAdd (the only other port of this shape):
//
//   1. the level-2 butterfly is called from EVERY thread, never from inside
//      `if (warpIndex == 0)`.  pxgWarpReduceAddF is built on group-wide barriers
//      (CUDA's __shfl_xor_sync is warp-local and needs none), so letting only warp
//      0 reach it deadlocks the workgroup.  The value is unaffected: the exchange
//      is warp-tile-local, so warp 0's inputs and result are the CUDA's.
//   2. a TRAILING barrier is added before returning.  CUDA's blockReduction has
//      none, and this kernel calls it back-to-back (three in a row on x/y/z, then
//      nine on the moment columns) -- which is a genuine write-after-read race on
//      the scratch in the CUDA itself.  It does not bite there because a warp
//      cannot outrun its own block by a whole reduction; the barrier makes the
//      port's scratch reuse safe without changing any value.
//
// ---------------------------------------------------------------------------
// TWO UPSTREAM ODDITIES, TRANSCRIBED FAITHFULLY AND NOT "FIXED"
// ---------------------------------------------------------------------------
//   (a) particlesystem.cu:5150 reads `if (rigidId > numRigids) return;` -- GREATER
//       THAN, not >=.  gridDim.x is mMaxRigidsPerBuffer, the max across buffers, so
//       for a buffer with fewer rigids the block with rigidId == numRigids is NOT
//       culled: it goes on to read rigidOffsets[numRigids] (the last valid runsum
//       entry, since the array is sized numRigids + 1) and rigidOffsets[numRigids+1]
//       (one past the end).  The resulting n is garbage and the block then walks it.
//       This port reproduces the `>`: changing it to `>=` would make the port differ
//       from the kernel it mirrors, and which of the two is "right" is not this
//       project's call.  It is recorded here so the next reader does not read the
//       `>` as a transcription slip.
//   (b) particlesystem.cu:5409 indexes the normal array as
//       `originalIndex = bufferIndex + localOffset`, where bufferIndex is blockIdx.y
//       -- a RIGID BUFFER index -- while every position read three lines above uses
//       `bufferOffset + localOffset` with bufferOffset = particleBufferRunsum[...].
//       The commented-out original (`rigidIndices[localOffset] + bufferIndex`)
//       suggests the intent was a particle offset.  With one rigid buffer at index 0
//       the two agree, which is why nothing has ever noticed.  Transcribed as
//       written.
//
// ---------------------------------------------------------------------------
// WHAT IS DEAD IN THIS BUILD
// ---------------------------------------------------------------------------
// particlesystem.cu:5114 is `#define ENABLE_PLASTIC_DEFORMATION 0`, so every
// `#if ENABLE_PLASTIC_DEFORMATION` branch is compiled out: the kernel's fifth
// `bool plasticDeformation` parameter does not exist (the host pack has four
// arguments and the signature has four), and the plastic-creep write-back to
// mRigidLocalPositions never happens.  With those branches gone, two locals in the
// transform loop become dead too -- `newPos` (only read inside the creep branch)
// and `dSq` (never read at all).  They are NOT transcribed; a dead Lerp would be
// noise, and check_push_block would see nothing either way.
//
// `dt` is DEAD in the kernel body: only `biasCoefficient` reaches the stiffness
// (`rigidCoefficients[rigidId] * biasCoefficient`, :5377).  It is in the push block
// because the HOST pack has it, which is the rule the shim gates on -- the same
// shape as ps_rigidAttachmentPrepareLaunch's dt/isTGS (P9.76).

#define PXG_WARP_BLOCK 64
#include "pxgSlangWarpInc.slang"

import pxgSlangParticleDesc;
import pxgSlangAtomicFloat;       // pxgAtomicAddF4XYZ / pxgAtomicAddF4W

// PxgParticleSystemKernelBlockDim::PS_SOLVE_SHAPE (…KernelIndices.h:49)
static const uint PS_SOLVE_SHAPE = 64u;
static const uint SM_WARPS = PS_SOLVE_SHAPE / 32u;   // 2

// PxgParticleRigidSimBuffer (PxgParticleSystem.h:183-194), 56 bytes.
// NOTE the declaration order: the COEFFICIENTS come first and mRigidOffsets sits
// FOURTH, between the rotations and the local positions.  Every one of the six is
// an 8-byte pointer, so swapping two of them crashes nothing -- it just reads a
// u32 index array as quaternions.
static const uint SZ_RIGID_SIM_BUFFER = 56u;
struct PxgParticleRigidSimBufferGpu
{
    uint64_t mRigidCoefficients;    // @0  PxReal*  (4-byte elements!)
    uint64_t mRigidTranslations;    // @8  float4*
    uint64_t mRigidRotations;       // @16 float4*
    uint64_t mRigidOffsets;         // @24 PxU32*   (mNumRigids + 1 entries)
    uint64_t mRigidLocalPositions;  // @32 float4*
    uint64_t mRigidLocalNormals;    // @40 float4*
    uint mNumRigids;                // @48
    uint mParticleBufferIndex;      // @52 -> 56
}

// ENGINE launch pack (PxgPBDParticleSystemCore.cpp:829):
//   ps_solveShapes(PxgParticleSystem*, const PxU32* activeParticleSystems,
//                  const PxReal dt, const PxReal biasCoefficient)
// -> argSizes {8,8,4,4}: offsets 0, 8, 16, 20, largest alignment 8, blob 24.
struct PsSolveShapesParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    float dt;                       // @16  DEAD in the kernel body
    float biasCoefficient;          // @20 -> 24
}

[[vk::push_constant]]
ConstantBuffer<PsSolveShapesParams> gSS;

// ---------------------------------------------------------------------------
// blockReduction<AddOpPxReal, PxReal, 64> (reduction.cuh:226-247)
// ---------------------------------------------------------------------------
groupshared float gSmWarpSum[SM_WARPS];

float smBlockReduceAdd(uint tid, float value, float initialValue)
{
    const uint warpIndex = tid / 32u;
    const uint laneIndex = tid & 31u;

    const float warpResult = pxgWarpReduceAddF(0xFFFFFFFFu, tid, value, 5u);

    if (laneIndex == 0u)
        gSmWarpSum[warpIndex] = warpResult;
    GroupMemoryBarrierWithGroupSync();

    // reduction.cuh:241 indexes the scratch by threadIndexInWarp, NOT warpIndex.
    // For warp 0 -- the only warp whose result is used -- the two coincide.
    const float val = (tid < SM_WARPS) ? gSmWarpSum[laneIndex] : initialValue;

    // Called from EVERY thread (see the header): the emulated butterfly is built
    // on group-wide barriers.
    const float reduced = pxgWarpReduceAddF(0xFFFFFFFFu, tid, val, 5u);

    GroupMemoryBarrierWithGroupSync();   // the scratch is reusable next call
    return (warpIndex == 0u) ? reduced : initialValue;
}

// ---------------------------------------------------------------------------
// quaternion / matrix helpers -- the same five functions softBodySolveBda.slang
// carries, transcribed here because that file declares no module and marks
// nothing public (P9.80 note: lifting them into a shared module is worth doing
// the next time a THIRD shader needs them; two copies is the corpus's existing
// state, and check_helper_uniformity.py is what keeps the copies honest).
// ---------------------------------------------------------------------------
float smDot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 smCross3(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// PxQuat::operator* (PxQuat.h:263) -- lhs is `a`.
float4 smQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
                  a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
                  a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

// PxMat33(PxQuat) rotation ctor (PxMat33.h:136); columns out.
void smMat33FromQuat(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
}

// PxQuat::rotate (PxQuat.h:286), the DOUBLED-component spelling.
float3 smQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// extractRotation (deformableUtils.cuh:46) -- Mueller's iterative polar
// decomposition.  Line for line the same as softBodySolveBda.slang's
// sbExtractRotation, and the traps are the same ones:
//   * PxVec3::normalize() returns the PRE-normalisation magnitude AND scales in
//     place, by a RECIPROCAL formed ONCE (P9.36);
//   * PxQuat(w, omega) is the AXIS-ANGLE constructor, not the component one: it
//     halves the angle and calls PxSinCos, so `w` becomes cos(w/2) in the real
//     part and the axis is scaled by sin(w/2);
//   * the multiply is a LEFT multiply, tempQ * q;
//   * getNormalized() has NO zero guard (unlike normalize()) and must not gain one;
//   * the early exit is AFTER the update, which the .cu comments on explicitly --
//     moving it to the top of the loop drops the final refinement.
//
// *** THE TOLERANCE SEAM *** -- sin/cos are the only operations in this kernel
// without a cross-backend rounding guarantee.  Everything else (sqrt, the
// quaternion product, the matrix build, every reduction) is correctly rounded or
// pure arithmetic.  The fixture therefore compares the pre-rotation reductions
// BIT-EXACTLY and the post-rotation outputs to a tolerance, and says which is which.
float4 smExtractRotation(float3 a0, float3 a1, float3 a2, float4 q, int maxIter)
{
    const float eps = 1.0e-6f;
    for (int iter = 0; iter < maxIter; iter++)
    {
        float3 r0, r1, r2;
        smMat33FromQuat(q, r0, r1, r2);

        float3 omega = smCross3(r0, a0) + smCross3(r1, a1) + smCross3(r2, a2);

        omega *= 1.0f / (abs(smDot3(r0, a0) + smDot3(r1, a1) + smDot3(r2, a2)) + eps);

        const float w = sqrt(smDot3(omega, omega));
        if (w > 0.0f)
            omega = omega * (1.0f / w);

        const float a = w * 0.5f;
        const float sa = sin(a);
        const float ca = cos(a);
        const float4 tempQ = float4(omega.x * sa, omega.y * sa, omega.z * sa, ca);

        q = smQuatMul(tempQ, q);

        const float qs = 1.0f / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        q = float4(q.x * qs, q.y * qs, q.z * qs, q.w * qs);

        if (w < eps)
            break;
    }
    return q;
}

// ---------------------------------------------------------------------------
// the per-block shared state the CUDA keeps in __shared__
// ---------------------------------------------------------------------------
groupshared float3 gSmNewCom;
groupshared float3 gSmOldCom;
groupshared float4 gSmNewRot;
groupshared float3 gSmQR0;
groupshared float3 gSmQR1;
groupshared float3 gSmQR2;

// mirrors ps_solveShapes (particlesystem.cu:5121).
//
// Engine grid (PxgPBDParticleSystemCore.cpp:840):
//   ( getMaxRigidsPerBuffer(), mMaxRigidBuffersPerSystem, nbActiveParticleSystems )
// block ( PS_SOLVE_SHAPE, 1, 1 ) = 64.
//
// The CUDA block-copies the whole 656-byte PxgParticleSystem into shared memory
// first (blockCopy + __syncthreads).  This port reads the six fields it needs
// through BDA instead, as every other particle shader here does: the copy is a
// CUDA memory-latency optimisation, not semantics, and 656 bytes of groupshared
// would compete with the reduction scratch for no gain.
[shader("compute")]
[numthreads(PS_SOLVE_SHAPE, 1, 1)]
void ps_solveShapes(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSS.particleSystems);
    const uint particleId = (Ptr<uint>(gSS.activeParticleSystems))[gid.z];

    const uint bufferIndex = gid.y;
    if (bufferIndex >= systems[particleId].mNumRigidBuffers)
        return;   // block-uniform

    const uint64_t rbAddr = systems[particleId].mRigidSimBuffers
                          + uint64_t(bufferIndex) * uint64_t(SZ_RIGID_SIM_BUFFER);
    const PxgParticleRigidSimBufferGpu rigidBuffer =
        (Ptr<PxgParticleRigidSimBufferGpu>(rbAddr))[0];

    const uint numRigids = rigidBuffer.mNumRigids;
    const uint rigidId = gid.x;

    // UPSTREAM: `>` and not `>=` (see the header).  Block-uniform either way.
    if (rigidId > numRigids)
        return;

    const uint particleBufferIndex = rigidBuffer.mParticleBufferIndex;
    Ptr<uint> particleBufferRunsum = Ptr<uint>(systems[particleId].mParticleBufferRunsum);

    Ptr<float4> newPositions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
    Ptr<uint> reverseLookup = Ptr<uint>(systems[particleId].mUnsortedToSortedMapping);

    Ptr<uint> rigidOffsets = Ptr<uint>(rigidBuffer.mRigidOffsets);
    Ptr<float> rigidCoefficients = Ptr<float>(rigidBuffer.mRigidCoefficients);
    Ptr<float4> localPositions = Ptr<float4>(rigidBuffer.mRigidLocalPositions);
    Ptr<float4> localNormals = Ptr<float4>(rigidBuffer.mRigidLocalNormals);

    const uint bufferOffset = particleBufferRunsum[particleBufferIndex];
    const uint64_t deltasAddr = systems[particleId].mDelta;
    Ptr<float4> translations = Ptr<float4>(rigidBuffer.mRigidTranslations);
    Ptr<float4> rotations = Ptr<float4>(rigidBuffer.mRigidRotations);
    Ptr<float4> normals = Ptr<float4>(systems[particleId].mNormalArray);

    const uint startIndex = rigidOffsets[rigidId];
    const uint endIndex = rigidOffsets[rigidId + 1u];
    const uint n = endIndex - startIndex;

    // break large shapes into multiple passes
    const uint passes = (n + PS_SOLVE_SHAPE - 1u) / PS_SOLVE_SHAPE;

    if (tid == 0u)
    {
        const float4 tTranslation = translations[rigidId];
        gSmNewCom = float3(0.0f, 0.0f, 0.0f);
        gSmOldCom = tTranslation.xyz;
        gSmQR0 = float3(0.0f, 0.0f, 0.0f);
        gSmQR1 = float3(0.0f, 0.0f, 0.0f);
        gSmQR2 = float3(0.0f, 0.0f, 0.0f);
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- centre of mass ---------------------------------------------------
    for (uint i = 0; i < passes; ++i)
    {
        const uint offset = i * PS_SOLVE_SHAPE + tid;

        float3 newPos = float3(0.0f, 0.0f, 0.0f);
        if (offset < n)
        {
            const uint index = bufferOffset + startIndex + offset;
            const uint sortedIndex = reverseLookup[index];
            newPos = newPositions[sortedIndex].xyz;
        }

        const float cx = smBlockReduceAdd(tid, newPos.x, 0.0f);
        const float cy = smBlockReduceAdd(tid, newPos.y, 0.0f);
        const float cz = smBlockReduceAdd(tid, newPos.z, 0.0f);

        if (tid == 0u)
            gSmNewCom = gSmNewCom + float3(cx, cy, cz);

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        float3 c = gSmNewCom * (1.0f / float(n));   // PxVec3::operator/= -- reciprocal ONCE

        // stop numerical drift by clamping translation to threshold
        const float eps = 0.00001f;
        if (abs(c.x - gSmOldCom.x) < eps) c.x = gSmOldCom.x;
        if (abs(c.y - gSmOldCom.y) < eps) c.y = gSmOldCom.y;
        if (abs(c.z - gSmOldCom.z) < eps) c.z = gSmOldCom.z;
        gSmNewCom = c;
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- moment matrix ----------------------------------------------------
    for (uint i = 0; i < passes; ++i)
    {
        const uint offset = i * PS_SOLVE_SHAPE + tid;

        // PxMat33::outer(a, b) = columns (a*b.x, a*b.y, a*b.z)  (PxMat33.h:191)
        float3 m0 = float3(0.0f, 0.0f, 0.0f);
        float3 m1 = float3(0.0f, 0.0f, 0.0f);
        float3 m2 = float3(0.0f, 0.0f, 0.0f);
        if (offset < n)
        {
            const uint index = bufferOffset + startIndex + offset;
            const uint sortedIndex = reverseLookup[index];
            const float3 newPos = newPositions[sortedIndex].xyz;
            const float3 delta = newPos - gSmNewCom;
            const float3 lp = localPositions[startIndex + offset].xyz;
            m0 = delta * lp.x;
            m1 = delta * lp.y;
            m2 = delta * lp.z;
        }

        // CUDA reduces column0 (x,y,z), __syncthreads(), column1, __syncthreads(),
        // column2 -- the barriers exist because its blockReduction has no trailing
        // one and reuses a single scratch array.  smBlockReduceAdd carries its own
        // trailing barrier, so the explicit ones would be redundant; the sequence
        // of reductions, and therefore every value, is identical.
        const float c0x = smBlockReduceAdd(tid, m0.x, 0.0f);
        const float c0y = smBlockReduceAdd(tid, m0.y, 0.0f);
        const float c0z = smBlockReduceAdd(tid, m0.z, 0.0f);
        const float c1x = smBlockReduceAdd(tid, m1.x, 0.0f);
        const float c1y = smBlockReduceAdd(tid, m1.y, 0.0f);
        const float c1z = smBlockReduceAdd(tid, m1.z, 0.0f);
        const float c2x = smBlockReduceAdd(tid, m2.x, 0.0f);
        const float c2y = smBlockReduceAdd(tid, m2.y, 0.0f);
        const float c2z = smBlockReduceAdd(tid, m2.z, 0.0f);

        if (tid == 0u)
        {
            gSmQR0 = gSmQR0 + float3(c0x, c0y, c0z);
            gSmQR1 = gSmQR1 + float3(c1x, c1y, c1z);
            gSmQR2 = gSmQR2 + float3(c2x, c2y, c2z);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ---- polar decomposition ---------------------------------------------
    if (tid == 0u)
    {
        const uint kMaxIters = 4u;

        float4 q = rotations[rigidId];
        q = smExtractRotation(gSmQR0, gSmQR1, gSmQR2, q, int(kMaxIters));

        rotations[rigidId] = q;
        translations[rigidId] = float4(gSmNewCom, 0.0f);

        gSmNewRot = q;
    }
    GroupMemoryBarrierWithGroupSync();

    const float stiffness = rigidCoefficients[rigidId] * gSS.biasCoefficient;

    // ---- transform particles toward the best fit --------------------------
    for (uint i = 0; i < passes; ++i)
    {
        const uint offset = i * PS_SOLVE_SHAPE + tid;

        if (offset < n)
        {
            const uint localOffset = startIndex + offset;
            const uint index = bufferOffset + localOffset;
            const uint sortedIndex = reverseLookup[index];

            const float3 elasticPos = newPositions[sortedIndex].xyz;
            const float3 rigidPos = smQuatRotate(gSmNewRot, localPositions[localOffset].xyz) + gSmNewCom;

            const float3 delta = elasticPos - rigidPos;

            // UPSTREAM: bufferIndex is blockIdx.y, a RIGID BUFFER index, where the
            // position reads above use bufferOffset.  See the header.
            const uint originalIndex = bufferIndex + localOffset;

            const float4 ln = localNormals[localOffset];
            const float3 normalRot = smQuatRotate(gSmNewRot, ln.xyz);
            normals[originalIndex] = float4(normalRot, ln.w);   // .w carried through

            // AtomicAdd(float4* p, i, PxVec3 v, float w) (atomic.cuh:67): three
            // component atomics plus an UNCONDITIONAL atomic add of w into .w.
            // Here w is 1.0f -- this kernel's contribution to the delta WEIGHT
            // that applyDeltas divides by, unlike the attachment solves which add
            // a literal zero.
            const float3 d = -delta * stiffness;
            pxgAtomicAddF4XYZ(deltasAddr, sortedIndex, d);
            pxgAtomicAddF4W(deltasAddr, sortedIndex, 1.0f);
        }
    }
}
