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

// pxgSlangFemRigidContact — the deformable-vs-rigid CONSTRAINT PREPARE, shared by the
// cloth and softbody contact paths.
//
// WHY THIS MODULE EXISTS.  `prepareFEMContacts` (deformableUtils.cuh:132) is one CUDA
// function that both cloth and softbody contact prepare call.  It was first ported inside
// femClothRigidContactBda.slang as a file-local function; sb_rigidContactPrepareLaunch
// needs it BYTE-FOR-BYTE, so it was extracted here rather than copied.  Two 78-line
// constraint-block writers that must stay in sync is exactly the kind of duplication that
// rots -- and the block layout below is the ABI the solve kernels read back, so a drift
// between two copies would be silent until a scene misbehaved.
//
// Extracted with it: only what the function needs -- the PxgFemRigidConstraintBlock field
// offsets, the small vector helpers and the PxNodeIndex accessors.  Deliberately NOT a
// tree-wide de-duplication pass: crossVec / dotVec / mat33MulVec / computeBasisVectors are
// also defined locally in femClothRigidAttachBda, psContactSolveBda and
// particleSelfCollisionBda, and the PxgFemOtherContactInfo (CI_*) offsets in
// femClothClothContactBda and softBodyContactRemapBda.  Those shaders do NOT import this
// module and keep their own copies; folding them in would be a much larger change with no
// payoff for the kernel that motivated this one.  Importers must therefore not also define
// these names locally.
//
// ARTICULATION BRANCH DEFERRED, as in the original port.  CUDA's prepareFEMContacts has an
// `if (rigidId.isArticulation())` anchor branch that reads
// sharedDesc->mBodySimBufferDeviceData (deformableUtils.cuh:162) and the articulation link
// transforms.  Neither is reachable for a rigid or static collider, which is every case the
// port exercises, so the else path is the only one transcribed and `sharedDesc` is not a
// parameter here.  A scene with an articulation collider needs that branch first.
//
// `invDt` and `isTGS` are NOT parameters, and that is not an omission: CUDA takes both but
// never reads them.  Brace-matched over the whole 122-line body (deformableUtils.cuh:132-253)
// they appear on the signature line only.  They stay in the callers' push blobs because the
// ABI row mirrors the CUDA arg pack, and are simply unread there too.
module pxgSlangFemRigidContact;

import pxgSlangPrepDesc;   // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangCommon;     // PxgSolverBodyData, PxgSolverTxIData

// ---------------------------------------------------------------------------
// Byte layout of the block-of-32 PxgFemRigidConstraintBlock (PxgFEMCore.h:167).
// 32 constraints per block (SoA warp-packed), indexed [offset]; stride 3200.
// ---------------------------------------------------------------------------
public static const uint FRC_raXn_resp        = 0u;      // float4[32]
public static const uint FRC_raXnF0_resp      = 512u;    // float4[32]
public static const uint FRC_raXnF1_resp      = 1024u;   // float4[32]
public static const uint FRC_normal_errorW    = 1536u;   // float4[32]
public static const uint FRC_fricTan0_invMass = 2048u;   // float4[32]
public static const uint FRC_barycentric      = 2560u;   // float4[32]
public static const uint FRC_maxPenBias       = 3072u;   // float[32]
public static const uint FRC_STRIDE           = 3200u;

// ---------------------------------------------------------------------------
// Small vector helpers (byte-for-byte the CUDA foundation math).
// ---------------------------------------------------------------------------
public float3 crossVec(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
public float dotVec(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxMat33 (column-major sqrtInvInertia[9]) * PxVec3.
public float3 mat33MulVec(float m[9], float3 v)
{
    return float3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
                  m[1] * v.x + m[4] * v.y + m[7] * v.z,
                  m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// PxComputeBasisVectors (PxMathUtils.h:297): two orthonormal tangents to `dir`.
// right.normalize() is a true divide by the magnitude (PxVec3::normalize(), PxVec3.h:295).
public void computeBasisVectors(float3 dir, out float3 right, out float3 up)
{
    if (abs(dir.y) <= 0.9999f)
    {
        right = float3(dir.z, 0.0f, -dir.x);
        const float m = sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (m > 0.0f)
            right = float3(right.x / m, right.y / m, right.z / m);
        up = float3(dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x);
    }
    else
    {
        right = float3(1.0f, 0.0f, 0.0f);
        up = float3(0.0f, dir.z, -dir.y);
        const float m = sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
        if (m > 0.0f)
            up = float3(up.x / m, up.y / m, up.z / m);
    }
}

// PxNodeIndex (PxNodeIndex.h:49) packed in the low/high 32 bits of a PxU64.
public bool nodeIsStaticBody(uint64_t ind) { return uint(ind & 0xffffffffUL) == 0xffffffffu; }
public uint nodeIndex(uint64_t ind) { return uint(ind & 0xffffffffUL); }
public bool nodeIsArticulation(uint64_t ind) { return (uint(ind >> 32) & 1u) != 0u; }

// ===========================================================================
// prepareFEMContacts (deformableUtils.cuh:132) — the rigid / static (non-articulation)
// path.  Writes the block-of-32 PxgFemRigidConstraintBlock slot [offset] at cBase.
// ===========================================================================
public void prepareFEMContacts(uint64_t cBase, uint offset, float3 normal,
                               float3 p, float pen, float3 delta, uint64_t rigidId,
                               float4 barycentric, uint64_t prepDescPtr,
                               uint64_t prePrepDescPtr, float penBiasClampFEM)
{
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(prepDescPtr);
    const uint64_t bodyFramesBase = prepDesc[0].body2WorldPool;
    const uint64_t solverBodyDataBase = prepDesc[0].solverBodyDataPool;
    const uint64_t solverTxIDataBase = prepDesc[0].solverBodyTxIDataPool;

    // Select two tangent vectors to the normal (PxComputeBasisVectors).
    float3 t0, t1;
    computeBasisVectors(normal, t0, t1);

    // ARTICULATION anchor branch DEFERRED (see header): a static floor / sphere is never
    // an articulation, so the else (rigid / static) path is taken.  solverBodyIdx == 0 for
    // a static body (solverBodyIndices untouched).
    const bool isStatic = nodeIsStaticBody(rigidId);
    uint idx = 0u;
    if (!isStatic)
        idx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDescPtr)[0].solverBodyIndices)[nodeIndex(rigidId)];

    PxgSolverTxIData txI = Ptr<PxgSolverTxIData>(solverTxIDataBase)[idx];
    float invSqrtInertia[9];
    for (uint k = 0; k < 9u; ++k)
        invSqrtInertia[k] = txI.sqrtInvInertia[k];

    const PxgSolverBodyData sbd = Ptr<PxgSolverBodyData>(solverBodyDataBase)[idx];
    const float4 linVel_invMass0 = sbd.initialLinVelXYZ_invMassW;
    const float penBiasClampRigid = sbd.initialAngVelXYZ_penBiasClamp.w;
    const float invMass0 = linVel_invMass0.w;

    // both static and kinematic objects have invMass == 0; a static body is never kinematic.
    const bool isKinematic = (invMass0 == 0.0f) && (!isStatic);

    const float4 bfP = Ptr<float4>(bodyFramesBase + uint64_t(idx) * 32u + 16u)[0];
    const float3 bodyFrame0p = float3(bfP.x, bfP.y, bfP.z);

    const float3 ra = float3(p.x - bodyFrame0p.x, p.y - bodyFrame0p.y, p.z - bodyFrame0p.z);
    const float3 raXn = crossVec(ra, normal);
    const float3 raXF0 = crossVec(ra, t0);
    const float3 raXF1 = crossVec(ra, t1);

    const float3 raXnSqrtInertia = mat33MulVec(invSqrtInertia, raXn);
    const float resp0 = dotVec(raXnSqrtInertia, raXnSqrtInertia) + invMass0;

    const float3 raXF0SqrtInertia = mat33MulVec(invSqrtInertia, raXF0);
    const float3 raXF1SqrtInertia = mat33MulVec(invSqrtInertia, raXF1);

    const float respF0 = dotVec(raXF0SqrtInertia, raXF0SqrtInertia) + invMass0;
    const float respF1 = dotVec(raXF1SqrtInertia, raXF1SqrtInertia) + invMass0;

    float4 raXn_resp;
    float4 raXnF0_resp;
    float4 raXnF1_resp;
    if (isKinematic)
    {
        raXn_resp = float4(raXn.x, raXn.y, raXn.z, resp0);
        raXnF0_resp = float4(raXF0.x, raXF0.y, raXF0.z, respF0);
        raXnF1_resp = float4(raXF1.x, raXF1.y, raXF1.z, respF1);
    }
    else
    {
        raXn_resp = float4(raXnSqrtInertia.x, raXnSqrtInertia.y, raXnSqrtInertia.z, resp0);
        raXnF0_resp = float4(raXF0SqrtInertia.x, raXF0SqrtInertia.y, raXF0SqrtInertia.z, respF0);
        raXnF1_resp = float4(raXF1SqrtInertia.x, raXF1SqrtInertia.y, raXF1SqrtInertia.z, respF1);
    }

    const float maxPenBias = max(penBiasClampRigid, penBiasClampFEM);
    const float error = pen + dotVec(delta, normal);

    Ptr<float4>(cBase + FRC_normal_errorW)[offset] = float4(normal.x, normal.y, normal.z, error);
    Ptr<float4>(cBase + FRC_raXn_resp)[offset] = raXn_resp;
    Ptr<float4>(cBase + FRC_raXnF0_resp)[offset] = raXnF0_resp;
    Ptr<float4>(cBase + FRC_raXnF1_resp)[offset] = raXnF1_resp;
    Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset] = float4(t0.x, t0.y, t0.z, invMass0);
    Ptr<float>(cBase + FRC_maxPenBias)[offset] = maxPenBias;
    Ptr<float4>(cBase + FRC_barycentric)[offset] = barycentric;
}
