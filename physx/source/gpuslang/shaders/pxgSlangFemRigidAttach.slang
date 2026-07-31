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

// pxgSlangFemRigidAttach — the deformable-vs-rigid ATTACHMENT ABI, shared by the cloth and
// softbody attachment paths.
//
// WHAT IS SHARED HERE, AND WHAT IS DELIBERATELY NOT.  `PxgFEMRigidAttachmentConstraint`
// (PxgFEMCore.h:191) and `PxgFEMRigidAttachment` (PxgSimulationCoreDesc.h:208) are ONE
// struct each, written by the prepare kernel and read back by the solve kernel of BOTH
// deformable types.  Their byte layouts are therefore an ABI, and two copies of an ABI
// drift silently -- the same argument that moved prepareFEMContacts into
// pxgSlangFemRigidContact.  They live here.
//
// The kernel BODIES do not.  CUDA itself carries two near-identical copies of the
// attachment prepare (FEMClothConstraintPrep.cu:387 and softBodyGM.cu:1822) and two of the
// solve, so a port with two copies MIRRORS upstream: when NVIDIA changes one, the
// corresponding port kernel changes with it.  Folding them into one shared function would
// be a port-introduced divergence from the source it is transcribing, which is the opposite
// of the reason prepareFEMContacts was shared.  The distinction is worth stating because
// the two cases look alike and pull in opposite directions:
//   - one CUDA function used by both  -> share it, or the copies rot;
//   - two CUDA functions that happen to agree -> keep two, or the port stops tracking them.
//
// The small math helpers come along because both callers need them and they are
// transcriptions of PhysX foundation, not of either kernel.
module pxgSlangFemRigidAttach;

// ---------------------------------------------------------------------------
// Byte layout of the block-of-32 PxgFEMRigidAttachmentConstraint (PxgFEMCore.h:191).
// 32 constraints per block (SoA), indexed [offset].
// ---------------------------------------------------------------------------
public static const uint RAC_baryOrType      = 0u;      // float4[32]
public static const uint RAC_raXn0_biasW     = 512u;    // float4[32]
public static const uint RAC_raXn1_biasW     = 1024u;   // float4[32]
public static const uint RAC_raXn2_biasW     = 1536u;   // float4[32]
public static const uint RAC_velMult_invMass = 2048u;   // float4[32]
public static const uint RAC_low_high_limits = 2560u;   // float4[32]
public static const uint RAC_axis_angle      = 3072u;   // float4[32]
public static const uint RAC_elemId          = 3584u;   // PxU32[32]
public static const uint RAC_rigidId         = 3712u;   // PxU64[32]
public static const uint RAC_STRIDE          = 3968u;

// Byte layout of PxgFEMRigidAttachment (PxgSimulationCoreDesc.h:208, extends
// PxgRigidFilterPair @0).  96 bytes.
public static const uint RA_index0      = 0u;    // PxU64  (rigid node index)
public static const uint RA_index1      = 8u;    // PxU64  (elemId in low 32 bits)
public static const uint RA_localPose0  = 32u;   // float4 (anchor local pose = ra4)
public static const uint RA_lowHigh     = 48u;   // float4 (coneLimitParams.low_high_limits)
public static const uint RA_axisAngle   = 64u;   // float4 (coneLimitParams.axis_angle)
public static const uint RA_baryOrType1 = 80u;   // float4
public static const uint RA_STRIDE_ATT  = 96u;

// PxGetIsVertexType (PxgSimulationCoreDesc.h:145): all four components exactly 0.
public bool femIsVertexType(float4 baryOrType)
{
    return baryOrType.x == 0.0f && baryOrType.y == 0.0f && baryOrType.z == 0.0f && baryOrType.w == 0.0f;
}

// PxQuat::rotate (PxQuat.h:286).
public float3 quatRotateVec(float4 q, float3 v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// PxVec3::getNormalized (PxVec3.h:286) with recipSqrt = 1.0/sqrt (unix intrinsic).
public float3 vecGetNormalized(float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f)
    {
        const float rs = 1.0f / sqrt(m);
        return float3(v.x * rs, v.y * rs, v.z * rs);
    }
    return float3(0.0f, 0.0f, 0.0f);
}
