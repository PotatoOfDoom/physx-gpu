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

// softBodySolveBda.slang
//
// P8.0 tranche 4 — the first two kernels of the softbody FEM-volume SOLVE loop on the
// Vulkan engine ABI (BDA).  Ported line-for-line from
// gpusimulationcontroller/src/CUDA/softBodyGM.cu:
//
//   sb_gm_zeroTetMultipliers       (:1801, launch PxgSoftBodyCore.cpp:196)
//       clears every tetrahedron's Lagrange-multiplier spatial vector at the top of the
//       solve.  Without it the constraint solver accumulates into last frame's
//       multipliers and the material response drifts frame over frame.
//   sb_gm_applyExternalDeltasLaunch (:2609, launch PxgSoftBodyCore.cpp:872)
//       folds the accumulated per-vertex position correction (mSimDelta, written by the
//       contact/attachment solves) back into position, velocity and mSimDeltaPos, then
//       zeroes the delta buffer for the next iteration.  This is the softbody twin of
//       cloth_applyExternalDeltasLaunch, which the cloth arc has relied on since P7.1.
//
// They bracket the solve: zeroTetMultipliers opens it, applyExternalDeltas closes each
// iteration.  Both are pure load/scale/store — no transcendental, no warp op, no
// cross-lane communication at all.
//
// Doctrine (identical to the prior softbody tranches):
//   - the PxgSoftBody descriptor arrives as a device-pointer ARRAY read through Ptr<>
//     (the byte-exact pxgSlangSoftBodyDesc mirror); mSimVelocity_InvMass @88,
//     mSimPosition_InvMass @96, mSimDeltaPos @128, mSimTetraMultipliers @264,
//     mSimDelta @272, mNumVertsGM @372 and mNumTetsGM @376 are the fields touched;
//   - PxgSpatialVectorBlock (PxgArticulation.h:77) is the 32-lane SoA block the
//     multipliers live in: float4 mTopxyz_bx[32] @0 (512 B) then float2 mbyz[32] @512
//     (256 B) -> 768-byte stride.  storeSpatialVector(block, v, lane) writes
//     mTopxyz_bx[lane] = (v.top.xyz, v.bottom.x) and mbyz[lane] = v.bottom.yz, so
//     zeroing a tet means two stores at block (tetIdx/32), lane (tetIdx&31).
//
// DEVIATIONS from CUDA:
//   - sb_gm_applyExternalDeltasLaunch opens with a `blockCopy` of the WHOLE 432-byte
//     PxgSoftBody descriptor into __shared__ followed by __syncthreads, then reads the
//     shared copy.  That is a pure read-caching optimisation for a block whose 64
//     threads would otherwise each pull the same descriptor words; it is dropped here
//     and every lane reads the descriptor directly through Ptr<> (the same fan-out
//     treatment the earlier softbody/cloth ports document).  Dropping it also removes
//     the kernel's only barrier, so there is no groupshared and no uniformity concern;
//   - CUDA's `delta.x *= scale; delta.y *= scale, delta.z *= scale;` (note the comma
//     operator on the second statement — a typo that happens to behave identically) is
//     written as a float3 multiply.
//
// Both kernels are NoContraction-stamped (PXG_NOCONTRACT_SHADERS): applyExternalDeltas
// computes `vel + delta * invDt`, a multiply feeding an add that would otherwise fuse
// into an FMA and break bitwise parity with the -ffp-contract=off CPU reference.  There
// is NO transcendental in either kernel — this tranche is FULLY bit-exact, no tolerance
// seam.

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimPosition_InvMass@96, ...)

// PxgSpatialVectorBlock (PxgArticulation.h:77): float4 mTopxyz_bx[32] @0, float2 mbyz[32]
// @512 -> 768-byte stride.
static const uint64_t SPATIAL_BLOCK_STRIDE = 768UL;
static const uint64_t SPATIAL_BLOCK_MBYZ_OFFSET = 512UL;

// ---------------------------------------------------------------------------
// sb_gm_zeroTetMultipliers (softBodyGM.cu:1801)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:198): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies } -> 16-byte blob.  (ABI row {8,8}.)
struct ZeroTetMultipliersParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8  -> 16
}
[[vk::push_constant]]
ConstantBuffer<ZeroTetMultipliersParams> gZero;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA; grid (nbBlocks, nbActiveSoftBodies, 1)
void sb_gm_zeroTetMultipliers(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gZero.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gZero.activeSoftbodies))[groupId.y];

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 64u;

    const uint numTets = gSoftbodies[softbodyId].mNumTetsGM;

    if (groupThreadIdx < numTets)
    {
        // storeSpatialVector(mSimTetraMultipliers[idx/32], UnAlignedSpatialVector::Zero(),
        //                    idx & 31)
        const uint64_t blockBase = gSoftbodies[softbodyId].mSimTetraMultipliers +
                                   uint64_t(groupThreadIdx >> 5) * SPATIAL_BLOCK_STRIDE;
        const uint lane = groupThreadIdx & 31u;

        Ptr<float4>(blockBase + uint64_t(lane) * 16UL)[0] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        Ptr<float2>(blockBase + SPATIAL_BLOCK_MBYZ_OFFSET + uint64_t(lane) * 8UL)[0] =
            float2(0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// sb_gm_applyExternalDeltasLaunch (softBodyGM.cu:2609)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:874): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxReal invDt } -> 20, rounded up to the pointer
// alignment -> 24-byte blob.  (ABI row {8,8,4}.)
struct ApplyExternalDeltasParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8
    float invDt;                // @16
    uint _pad20;                // @20  -> 24
}
[[vk::push_constant]]
ConstantBuffer<ApplyExternalDeltasParams> gApply;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_applyExternalDeltasLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gApply.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gApply.activeSoftbodies))[groupId.y];

    // The CUDA blockCopy of the whole descriptor into __shared__ + __syncthreads is a
    // read-caching optimisation only (see the header) — read it directly instead.
    Ptr<float4> curPositions = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
    Ptr<float4> vels = Ptr<float4>(gSoftbodies[softbodyId].mSimVelocity_InvMass);
    Ptr<float4> simDelta = Ptr<float4>(gSoftbodies[softbodyId].mSimDelta);
    Ptr<float4> simDeltaPos = Ptr<float4>(gSoftbodies[softbodyId].mSimDeltaPos);

    const uint vertIdx = groupThreadId.x + groupId.x * 64u;
    const uint nbVerts = gSoftbodies[softbodyId].mNumVertsGM;

    if (vertIdx < nbVerts)
    {
        const float4 pos = curPositions[vertIdx];
        const float4 vel = vels[vertIdx];

        float4 delta = simDelta[vertIdx];

        // delta.w counts how many constraints contributed; averaging divides by it, but
        // never by less than 1 (an untouched vertex has delta.w == 0 and delta == 0).
        const float scale = 1.0f / max(delta.w, 1.0f);

        delta.x *= scale; delta.y *= scale; delta.z *= scale;
        delta.w = 0.0f;

        const float4 newVel = vel + delta * gApply.invDt;

        curPositions[vertIdx] = pos + delta;

        vels[vertIdx] = newVel;

        simDeltaPos[vertIdx] = simDeltaPos[vertIdx] + delta;

        // clear delta
        simDelta[vertIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}


// ===========================================================================
// P8.0 tranche 5 — the tetrahedron ROTATION extractors (softBodyGM.cu).
//
//   sb_gm_updateTetrahedraRotationsLaunch (:419, launch PxgSoftBodyCore.cpp:743)
//       the SIMULATION mesh: one thread per sim tetrahedron, recomputes the
//       co-rotational frame from the current mSimPosition_InvMass.
//   sb_updateTetrahedraRotationsLaunch    (:461, launch PxgSoftBodyCore.cpp:786)
//       the COLLISION mesh twin, gated on eCOMPUTE_STRESS_TENSOR or
//       !eDISABLE_SELF_COLLISION (the collision-mesh rotations are only needed by the
//       stress readback and by self-collision).
//
// Both compute the deformation gradient F = P * Qinv from the tetrahedron's displacement
// field, seed the quaternion from the stored rotation (or mInitialRotation on the first
// frame, detected by the zero-initialised memory), and run Mueller's iterative polar
// decomposition extractRotation(F, q, 100) (deformableUtils.cuh:46).
//
// They differ in exactly three places, all from the .cu:
//   - buffers: mSimTetIndices @80 / mSimPosition_InvMass @96 / mSimTetraRestPoses @136 /
//     mSimTetraRotations @168 / mNumTetsGM @376   -- versus --   mTetIndices @16 /
//     mPosition_InvMass @64 / mTetraRestPoses @40 / mTetraRotations @56 / mNumTets @368;
//   - rest-pose LAYOUT: the sim variant loads through loadPxMat33 from a PxgMat33Block
//     (PxgArticulation.h:70 — float4 mCol0[32] @0, float4 mCol1[32] @512, float mCol2[32]
//     @1024 -> 1152-byte stride, the 9 elements split 4/4/1 across the three arrays),
//     while the collision variant reads a flat PxMat33 array (36-byte stride);
//   - the collision variant is wrapped in the stress/self-collision flag test; the sim
//     variant runs unconditionally.
//
// *** TOLERANCE SEAM (the first one in the softbody arc) ***
// extractRotation builds its per-iteration update as PxQuat(w, omega), and that
// constructor (PxQuat.h:86) evaluates PxSinCos -> sinf/cosf.  sin/cos have NO
// cross-backend rounding guarantee, so this kernel CANNOT be bit-exact — exactly the
// reason npSbExtractRotAPD.slang documents the Mueller variant as non-portable and
// implements the APD variant (which has no transcendental) instead.  The .cu chooses
// extractRotation here (the sb_extractRotationAPD call sits commented out one line
// above), so the seam is on the default path and cannot be avoided by configuration.
// Everything else in the chain IS bit-exact: PxVec3::normalize and
// PxQuat::getNormalized both go through PxSqrt, which is IEEE-754 correctly rounded,
// and the quaternion product / rotation-matrix build are pure arithmetic.  The unit test
// therefore compares the quaternion under a documented ULP budget rather than by raw
// bits, and pins everything up to the first sin/cos bit-exactly.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): F = P*Qinv,
// the cross/dot accumulations in the omega step and the quaternion product are all
// mul-feeding-add chains.
// ===========================================================================

// PxgMat33Block (PxgArticulation.h:70): float4 mCol0[32] @0, float4 mCol1[32] @512,
// float mCol2[32] @1024 -> 1152-byte stride.
static const uint64_t MAT33_BLOCK_STRIDE = 1152UL;
static const uint64_t MAT33_BLOCK_COL1_OFFSET = 512UL;
static const uint64_t MAT33_BLOCK_COL2_OFFSET = 1024UL;

// PxQuat operator* (PxQuat.h:388), left-to-right folds — identical to the proven
// npSbExtractRotAPD transcription.
float4 sbQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
                  a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
                  a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

// PxMat33(PxQuat) rotation ctor (PxMat33.h:136); columns out.
void sbMat33FromQuat(float4 q, out float3 c0, out float3 c1, out float3 c2)
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

float sbDot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 sbCross3(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// PxMat33 P*Q: col_i = P.transform(Q.col_i) = (P.c0*v.x + P.c1*v.y) + P.c2*v.z.
float3 sbMat33Transform(float3 c0, float3 c1, float3 c2, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}

// extractRotation (deformableUtils.cuh:46) — Mueller's iterative polar decomposition.
// THE sin/cos SEAM LIVES HERE (the PxQuat(w, omega) constructor).
float4 sbExtractRotation(float3 a0, float3 a1, float3 a2, float4 q, int maxIter)
{
    const float eps = 1.0e-6f;
    for (int iter = 0; iter < maxIter; iter++)
    {
        float3 r0, r1, r2;
        sbMat33FromQuat(q, r0, r1, r2);

        float3 omega = sbCross3(r0, a0) + sbCross3(r1, a1) + sbCross3(r2, a2);

        omega *= 1.0f / (abs(sbDot3(r0, a0) + sbDot3(r1, a1) + sbDot3(r2, a2)) + eps);

        // PxVec3::normalize(): m = PxSqrt(magnitudeSquared()); if (m > 0) *this /= m;
        // returns m.  PxSqrt is IEEE-754 correctly rounded -> bit-exact.
        const float w = sqrt(sbDot3(omega, omega));
        if (w > 0.0f)
            omega = omega / w;

        // PxQuat(angleRadians, unitAxis) (PxQuat.h:86): a = angle*0.5; PxSinCos(a, s, w).
        // *** THE TOLERANCE SEAM *** — sin/cos are not bitwise portable.
        const float a = w * 0.5f;
        const float sa = sin(a);
        const float ca = cos(a);
        const float4 tempQ = float4(omega.x * sa, omega.y * sa, omega.z * sa, ca);

        q = sbQuatMul(tempQ, q);

        // PxQuat::getNormalized(): s = 1.0 / PxSqrt(magnitudeSquared()) -> bit-exact.
        const float qs = 1.0f / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        q = float4(q.x * qs, q.y * qs, q.z * qs, q.w * qs);

        // early-exit AFTER one update (the .cu comment: the expensive work is done)
        if (w < eps)
            break;
    }
    return q;
}

// Shared body of both kernels: F = P*Qinv from the displacement field, seed q, iterate.
float4 sbTetRotation(float3 u1, float3 u2, float3 u3,
                     float3 qinv0, float3 qinv1, float3 qinv2,
                     float4 storedRot, float4 initialRotation, int maxIter)
{
    // PxMat33 P(u1, u2, u3) — columns.  F = P * Qinv.
    const float3 f0 = sbMat33Transform(u1, u2, u3, qinv0);
    const float3 f1 = sbMat33Transform(u1, u2, u3, qinv1);
    const float3 f2 = sbMat33Transform(u1, u2, u3, qinv2);

    float4 q = storedRot;
    // initialize to identity quat if this is the first iteration (zero initialized memory)
    if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) == 0.0f)
        q = initialRotation;

    return sbExtractRotation(f0, f1, f2, q, maxIter);
}

// ---------------------------------------------------------------------------
// sb_gm_updateTetrahedraRotationsLaunch (softBodyGM.cu:419) — SIMULATION mesh.
// CUDA arg pack (PxgSoftBodyCore.cpp:747): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies } -> 16-byte blob.  (ABI row {8,8}.)
// ---------------------------------------------------------------------------
struct UpdateTetraRotationsParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8  -> 16
}
[[vk::push_constant]]
ConstantBuffer<UpdateTetraRotationsParams> gRot;

[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_updateTetrahedraRotationsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gRot.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gRot.activeSoftbodies))[groupId.y];

    const uint tetrahedronIdx = groupThreadId.x + groupId.x * 256u;
    const uint nbTets = gSoftbodies[softbodyId].mNumTetsGM;

    if (tetrahedronIdx < nbTets)
    {
        const float4 initialRotation = float4(gSoftbodies[softbodyId].mInitialRotationX,
                                              gSoftbodies[softbodyId].mInitialRotationY,
                                              gSoftbodies[softbodyId].mInitialRotationZ,
                                              gSoftbodies[softbodyId].mInitialRotationW);

        Ptr<uint4> tetIndices = Ptr<uint4>(gSoftbodies[softbodyId].mSimTetIndices);
        Ptr<float4> verts = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
        Ptr<float4> tetRotations = Ptr<float4>(gSoftbodies[softbodyId].mSimTetraRotations);

        const uint4 tetInd = tetIndices[tetrahedronIdx];
        const float4 v0 = verts[tetInd.x];
        const float4 v1 = verts[tetInd.y];
        const float4 v2 = verts[tetInd.z];
        const float4 v3 = verts[tetInd.w];

        // compute displacement field
        const float3 u1 = float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        const float3 u2 = float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        const float3 u3 = float3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);

        // load rest pose: loadPxMat33(restPoses[idx/32], idx&31, Qinv) over a
        // PxgMat33Block — the 9 elements are split 4/4/1 across mCol0/mCol1/mCol2.
        const uint64_t blockBase = gSoftbodies[softbodyId].mSimTetraRestPoses +
                                   uint64_t(tetrahedronIdx >> 5) * MAT33_BLOCK_STRIDE;
        const uint lane = tetrahedronIdx & 31u;
        const float4 c0v = Ptr<float4>(blockBase + uint64_t(lane) * 16UL)[0];
        const float4 c1v = Ptr<float4>(blockBase + MAT33_BLOCK_COL1_OFFSET + uint64_t(lane) * 16UL)[0];
        const float c2v = Ptr<float>(blockBase + MAT33_BLOCK_COL2_OFFSET + uint64_t(lane) * 4UL)[0];

        const float3 qinv0 = float3(c0v.x, c0v.y, c0v.z);
        const float3 qinv1 = float3(c0v.w, c1v.x, c1v.y);
        const float3 qinv2 = float3(c1v.z, c1v.w, c2v);

        const float4 q = sbTetRotation(u1, u2, u3, qinv0, qinv1, qinv2,
                                       tetRotations[tetrahedronIdx], initialRotation, 100);

        tetRotations[tetrahedronIdx] = q;
    }
}

// ---------------------------------------------------------------------------
// sb_updateTetrahedraRotationsLaunch (softBodyGM.cu:461) — COLLISION mesh.
// Same 16-byte arg pack (PxgSoftBodyCore.cpp:790).  Gated on
// eCOMPUTE_STRESS_TENSOR (mVolumeFlags) OR !eDISABLE_SELF_COLLISION (mBodyFlags).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_updateTetrahedraRotationsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gRot.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gRot.activeSoftbodies))[groupId.y];

    const uint tetrahedronIdx = groupThreadId.x + groupId.x * 256u;
    const uint nbTets = gSoftbodies[softbodyId].mNumTets;

    // The @392 fold: byte1 = mBodyFlags, bytes2..3 = mVolumeFlags.
    const uint flagsWord = gSoftbodies[softbodyId].mFlagsWord;
    const uint bodyFlags = (flagsWord >> 8) & 0xffu;
    const uint volumeFlags = (flagsWord >> 16) & 0xffffu;

    // KS - if we don't need stress tensors, we don't need to compute the tet rotations.
    if ((volumeFlags & PXG_SOFTBODY_VOLUMEFLAG_COMPUTE_STRESS_TENSOR) != 0u ||
        (bodyFlags & PXG_SOFTBODY_BODYFLAG_DISABLE_SELF_COLLISION) == 0u)
    {
        if (tetrahedronIdx < nbTets)
        {
            const float4 initialRotation = float4(gSoftbodies[softbodyId].mInitialRotationX,
                                                  gSoftbodies[softbodyId].mInitialRotationY,
                                                  gSoftbodies[softbodyId].mInitialRotationZ,
                                                  gSoftbodies[softbodyId].mInitialRotationW);

            Ptr<uint4> tetIndices = Ptr<uint4>(gSoftbodies[softbodyId].mTetIndices);
            Ptr<float4> verts = Ptr<float4>(gSoftbodies[softbodyId].mPosition_InvMass);
            Ptr<float4> tetRotations = Ptr<float4>(gSoftbodies[softbodyId].mTetraRotations);

            const uint4 tetInd = tetIndices[tetrahedronIdx];
            const float4 v0 = verts[tetInd.x];
            const float4 v1 = verts[tetInd.y];
            const float4 v2 = verts[tetInd.z];
            const float4 v3 = verts[tetInd.w];

            const float3 u1 = float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            const float3 u2 = float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            const float3 u3 = float3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);

            // FLAT PxMat33 array here (36-byte stride), not a PxgMat33Block.
            const uint64_t restBase = gSoftbodies[softbodyId].mTetraRestPoses +
                                      uint64_t(tetrahedronIdx) * 36UL;
            const float3 qinv0 = float3(Ptr<float>(restBase)[0], Ptr<float>(restBase)[1], Ptr<float>(restBase)[2]);
            const float3 qinv1 = float3(Ptr<float>(restBase)[3], Ptr<float>(restBase)[4], Ptr<float>(restBase)[5]);
            const float3 qinv2 = float3(Ptr<float>(restBase)[6], Ptr<float>(restBase)[7], Ptr<float>(restBase)[8]);

            const float4 q = sbTetRotation(u1, u2, u3, qinv0, qinv1, qinv2,
                                           tetRotations[tetrahedronIdx], initialRotation, 100);

            tetRotations[tetrahedronIdx] = q;
        }
    }
}
