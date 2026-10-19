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

// pxgSlangDeformableAndParticleUtils -- the PhysX 5.12 deformable / particle solve
// core, transcribed from gpusimulationcontroller/src/CUDA/deformableAndParticleUtils.cuh.
//
// WHAT IS HERE.  The pieces the particle consumers (particlesystem.cu) call, written so
// the cloth and softbody consumers can import the same module rather than copy it:
//
//   PxgDbContactPair      the pair-level prep inputs: readContactPrepRigidParticle
//                         (PxgRigidParticleContactBlock + the particle's clamp, the pair
//                         max taken here), readContactPrepRigidDb (PxgRigidDbContactBlock),
//                         readContactPrepDbDb (PxgDbDbContactBlock / PxgParticleDbContactBlock,
//                         both normal_pen @0, the clamp offset passed in).
//   PxgDbSolveOutput      the solve output + computeDelta.
//   PxgRigidPart          readBodyProperties, readNodelessStatic, readContactPrep,
//                         readAttachmentPrep, readVelocity, writeContactDeltas,
//                         writeAttachmentDeltas (PxgRigidDbAttachmentBlock).
//   PxgDeformablePart     readParticle + writeParticle (the particle half);
//                         readSoftBody, readSoftBodyAttachment, readSoftBodyAttachmentDbDb,
//                         writeSoftBody, bumpSoftBodyRefCount (the softbody half, added by
//                         the softbody stage); readCloth, readClothAttachment,
//                         readClothAttachmentDbDb, writeCloth, bumpClothRefCount, and
//                         writeClothEdge (the cloth-cloth EE writer, added by the cloth
//                         stage).
//   bumpDbRefCountVtx/Tri/Tet, bumpSoftBodyRefCount(Bary), bumpClothRefCount(Bary),
//   updateDeltaVtx/Edge/Tri/Tet  the free reference-count and delta-scatter helpers
//                         (bumpDbRefCountEdge is not ported: only the
//                         PX_CLOTH_CLOTH_INTEGER_REFCOUNT != 0 build calls it).
//   solveRigidDbAttachment, solveDbDbAttachment   the attachment solves.
//   solveRigidDbContact   both instantiations (withAdhesion is a runtime bool here).
//   solveDbDbContact      for the deformable-deformable / deformable-particle stages.
//   particleAdhesionImpulse, readParticleLinDelta, getGlobalRigidBodyId (the 5.12 per-link
//   bucket), rigidIsRefCounted, combineScalars (PxsMaterialCombiner.h).
//
// THE TEMPLATE.  PxgDeformablePart<Vec> is PxVec3 (cloth vertex / triangle, particle) or
// PxVec4 (softbody tetrahedron).  Here bc and vertexInvMasses are float4 and `dim` (3 or
// 4) says which; every reduction over them (dbBcResponse) runs over exactly `dim`
// components in the PxVec3 / PxVec4 dot order, so a PxVec3 instantiation never adds a
// fourth +0 term (which would turn a -0 sum into +0).
//
// FLOAT SPELLING.  PxMax / PxMin on floats are intrinsics::selectMax / selectMin
// (a > b ? a : b / a < b ? a : b); PxVec3 dot / magnitudeSquared are (x + y) + z, written
// out explicitly here.  Every importer must be in PXG_NOCONTRACT_SHADERS.
//
// ONE DEVIATION, value-neutral (the precedent of pxgSlangVelocityReader):
// getGlobalRigidBodyId takes artiCoreDesc and reads mMaxLinksPerArticulation inside the
// articulation arm, where the CUDA kernels read it once at the kernel top for every
// contact.  A rigid / static participant never dereferences artiCoreDesc.
module pxgSlangDeformableAndParticleUtils;

import pxgSlangDeformableRef;     // PxgNode* accessors, block offsets (PXG_RPC_*, PXG_RDBC_*, PXG_RDBA_*)
import pxgSlangVelocityReader;    // PxgVelocityReaderGpu, readVelocitiesPGS/TGS
import pxgSlangPrepDesc;          // PxgPrePrepDescGpu
import pxgSlangArticulationDesc;  // artiU32Field, PXG_ARTI_CORE_MAX_LINKS
import pxgSlangParticleDesc;      // PxgParticleSystemGpu
import pxgSlangSoftBodyDesc;      // PxgSoftBodyGpu
import pxgSlangFEMClothDesc;      // PxgFEMClothGpu
import pxgSlangAtomicFloat;       // pxgAtomicAddF4W / pxgAtomicAddF4XYZ

// ---------------------------------------------------------------------------
// Foundation spellings.
// ---------------------------------------------------------------------------
public float dpSelectMax(float a, float b) { return (a > b) ? a : b; }   // PxMax(float)
public float dpSelectMin(float a, float b) { return (a < b) ? a : b; }   // PxMin(float)

// PxVec3::dot, (x + y) + z.
public float dpDot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxVec3::cross.
public float3 dpCross3(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// PxVec3::normalize: returns the magnitude and scales by ONE reciprocal (operator/=),
// leaving a zero vector untouched.
public float dpNormalize3(inout float3 v)
{
    const float m = sqrt(dpDot3(v, v));
    if (m > 0.0f)
    {
        const float r = 1.0f / m;
        v = float3(v.x * r, v.y * r, v.z * r);
    }
    return m;
}

// PxComputeBasisVectors (PxMathUtils.h).
public void dpComputeBasisVectors(float3 dir, out float3 right, out float3 up)
{
    if (abs(dir.y) <= 0.9999f)
    {
        right = float3(dir.z, 0.0f, -dir.x);
        dpNormalize3(right);
        up = float3(dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x);
    }
    else
    {
        right = float3(1.0f, 0.0f, 0.0f);
        up = float3(0.0f, dir.z, -dir.y);
        dpNormalize3(up);
    }
}

// PxCombineMode::Enum and combineScalars (lowlevel/software/include/PxsMaterialCombiner.h).
public static const int PXG_COMBINE_eAVERAGE = 0;
public static const int PXG_COMBINE_eMIN = 1;
public static const int PXG_COMBINE_eMULTIPLY = 2;
public static const int PXG_COMBINE_eMAX = 3;

public float combineScalars(float a, float b, int combineMode)
{
    if (combineMode == PXG_COMBINE_eAVERAGE)
        return 0.5f * (a + b);
    if (combineMode == PXG_COMBINE_eMIN)
        return dpSelectMin(a, b);
    if (combineMode == PXG_COMBINE_eMULTIPLY)
        return a * b;
    if (combineMode == PXG_COMBINE_eMAX)
        return dpSelectMax(a, b);
    return 0.0f;
}

// PxsMaterialData (PxsMaterialCore.h): dynamicFriction @0, PxU8 fricCombineMode @18.
public static const uint64_t PXG_RIGID_MATERIAL_DYNAMIC_FRICTION = 0;
public static const uint64_t PXG_RIGID_MATERIAL_FRIC_COMBINE_MODE = 18;

// PxsParticleMaterialData (PxParticleGpu.h) float indices.
public static const uint PXG_PMAT_FRICTION = 0u;
public static const uint PXG_PMAT_DAMPING = 1u;
public static const uint PXG_PMAT_ADHESION = 2u;
public static const uint PXG_PMAT_GRAVITY_SCALE = 3u;
public static const uint PXG_PMAT_ADHESION_RADIUS_SCALE = 4u;

// ---------------------------------------------------------------------------
// PxgDbContactPair
// ---------------------------------------------------------------------------
public struct PxgDbContactPair
{
    public float3 normal = float3(0.0f, 0.0f, 0.0f);
    // CUDA leaves tangent0 uninitialised; every reader that consumes it sets it first.
    public float3 tangent0 = float3(0.0f, 0.0f, 0.0f);
    public float initPen = 0.0f;
    public float maxPenBiasClamp = 0.0f;

    // readContactPrep(const PxgRigidParticleContactBlock&, particlePenBiasClamp, lane):
    // the block carries only the rigid's clamp, the pair max is taken here.
    [mutating]
    public void readContactPrepRigidParticle(uint64_t block, float particlePenBiasClamp, uint lane)
    {
        const float4 normal_errorW = Ptr<float4>(block + PXG_RPC_normal_errorW)[lane];
        normal = float3(normal_errorW.x, normal_errorW.y, normal_errorW.z);
        initPen = normal_errorW.w;

        const float4 fricTan0_invMass0 = Ptr<float4>(block + PXG_RPC_fricTan0_invMass0)[lane];
        tangent0 = float3(fricTan0_invMass0.x, fricTan0_invMass0.y, fricTan0_invMass0.z);

        maxPenBiasClamp = dpSelectMax(Ptr<float>(block + PXG_RPC_penBiasClampRigid)[lane], particlePenBiasClamp);
    }

    // readContactPrep<PxgRigidDbContactBlock>(block, lane).
    [mutating]
    public void readContactPrepRigidDb(uint64_t block, uint lane)
    {
        const float4 normal_errorW = Ptr<float4>(block + PXG_RDBC_normal_errorW)[lane];
        normal = float3(normal_errorW.x, normal_errorW.y, normal_errorW.z);
        initPen = normal_errorW.w;

        const float4 fricTan0_invMass0 = Ptr<float4>(block + PXG_RDBC_fricTan0_invMass0)[lane];
        tangent0 = float3(fricTan0_invMass0.x, fricTan0_invMass0.y, fricTan0_invMass0.z);

        maxPenBiasClamp = Ptr<float>(block + PXG_RDBC_maxPenBiasClamp)[lane];
    }

    // readContactPrepDbDb<Block>(block, lane): PxgDbDbContactBlock (clamp @PXG_DBDBC_maxPenBiasClamp)
    // or PxgParticleDbContactBlock (clamp @PXG_PDBC_maxPenBiasClamp); normal_pen is @0 in both.
    [mutating]
    public void readContactPrepDbDb(uint64_t block, uint64_t maxPenBiasClampOffset, uint lane)
    {
        const float4 normal_pen = Ptr<float4>(block)[lane];
        normal = float3(normal_pen.x, normal_pen.y, normal_pen.z);
        initPen = normal_pen.w;
        maxPenBiasClamp = Ptr<float>(block + maxPenBiasClampOffset)[lane];
    }
}

// ---------------------------------------------------------------------------
// PxgDbSolveOutput
// ---------------------------------------------------------------------------
public struct PxgDbSolveOutput
{
    public float3 tangent = float3(0.0f, 0.0f, 0.0f);
    public float3 raXt = float3(0.0f, 0.0f, 0.0f);
    public float deltaLambdaN = 0.0f;
    public float deltaLambdaT = 0.0f;
    public float accumulatedDeltaLambdaN = 0.0f;

    // delta = -(deltaLambdaN * normal + deltaLambdaT * tangent) * dt; returns the
    // accumulated normal multiplier.
    public float computeDelta(PxgDbContactPair contactPair, out float3 delta, float dt)
    {
        const float3 s = float3(deltaLambdaN * contactPair.normal.x + deltaLambdaT * tangent.x,
                                deltaLambdaN * contactPair.normal.y + deltaLambdaT * tangent.y,
                                deltaLambdaN * contactPair.normal.z + deltaLambdaT * tangent.z);
        delta = float3((-s.x) * dt, (-s.y) * dt, (-s.z) * dt);
        return accumulatedDeltaLambdaN;
    }
}

// ---------------------------------------------------------------------------
// getGlobalRigidBodyId / rigidIsRefCounted
// ---------------------------------------------------------------------------
// [0..numSolverBodies) rigid-body slots, then numArticulations * maxLinksPerArticulation
// per-LINK slots.  -1 for a static body.  See the header for the artiCoreDesc deviation.
public int getGlobalRigidBodyId(uint64_t prePrepDesc, uint64_t rigidId, uint numSolverBodies,
                                uint64_t artiCoreDesc)
{
    if (PxgNodeIsStaticBody(rigidId))
        return -1;

    const uint solverBodyIdx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDesc)[0].solverBodyIndices)[PxgNodeIndexOf(rigidId)];

    if (PxgNodeIsArticulation(rigidId))
    {
        const uint maxLinksPerArticulation = artiU32Field(artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
        return int(numSolverBodies + solverBodyIdx * maxLinksPerArticulation + PxgNodeArticulationLinkId(rigidId));
    }
    return int(solverBodyIdx);
}

public bool rigidIsRefCounted(int globalRigidBodyId, float rigidInvMass)
{
    return globalRigidBodyId != -1 && rigidInvMass != 0.0f;
}

// ---------------------------------------------------------------------------
// PxgRigidPart
// ---------------------------------------------------------------------------
public struct PxgRigidPart
{
    public uint referenceCount = 1u;
    public float friction = 0.0f;
    public int fricCombineMode = PXG_COMBINE_eMAX;   // CUDA: uninitialised until readBodyProperties

    public float3 raXn = float3(0.0f, 0.0f, 0.0f);
    public float3 raXnF0 = float3(0.0f, 0.0f, 0.0f);
    public float3 raXnF1 = float3(0.0f, 0.0f, 0.0f);
    public float raXnResp = 0.0f;
    public float raXnF0Resp = 0.0f;
    public float raXnF1Resp = 0.0f;
    public float invMass = 0.0f;

    public float3 raXn0 = float3(0.0f, 0.0f, 0.0f);
    public float3 raXn1 = float3(0.0f, 0.0f, 0.0f);
    public float3 raXn2 = float3(0.0f, 0.0f, 0.0f);
    public float3 positionErrorXYZ = float3(0.0f, 0.0f, 0.0f);
    public float3 velMultiplierXYZ = float3(0.0f, 0.0f, 0.0f);
    public float attachInvMass = 0.0f;
    public uint attachRefCount = 1u;

    public float3 linVel = float3(0.0f, 0.0f, 0.0f);
    public float3 angVel = float3(0.0f, 0.0f, 0.0f);
    public float3 linDelta = float3(0.0f, 0.0f, 0.0f);
    public float3 angDelta = float3(0.0f, 0.0f, 0.0f);

    // rigidBodyRefCounts == 0 and rigidMaterial == 0 stand for the CUDA NULL pointers.
    [mutating]
    public void readBodyProperties(uint64_t rigidId, int globalRigidBodyId, float rigidInvMass,
                                   uint64_t rigidBodyRefCounts, uint64_t rigidMaterial)
    {
        referenceCount = 1u;

        if (rigidBodyRefCounts != 0 && rigidIsRefCounted(globalRigidBodyId, rigidInvMass))
            referenceCount = Ptr<uint>(rigidBodyRefCounts)[uint(globalRigidBodyId)];

        if (rigidMaterial != 0)
        {
            friction = Ptr<float>(rigidMaterial + PXG_RIGID_MATERIAL_DYNAMIC_FRICTION)[0];
            const uint word = Ptr<uint>(rigidMaterial + 16)[0];   // PxU16 flags, PxU8 fricCombineMode @18
            fricCombineMode = int((word >> 16u) & 0xffu);
        }
        else
        {
            friction = 0.0f;
            fricCombineMode = PXG_COMBINE_eMAX;
        }
    }

    // The particle-vs-static-mesh (one-way) stand-in: no rigid node, never written back.
    [mutating]
    public void readNodelessStatic()
    {
        referenceCount = 1u;
        friction = 0.0f;
        fricCombineMode = PXG_COMBINE_eMAX;
        raXn = float3(0.0f, 0.0f, 0.0f);
        raXnF0 = float3(0.0f, 0.0f, 0.0f);
        raXnF1 = float3(0.0f, 0.0f, 0.0f);
        raXnResp = 0.0f;
        raXnF0Resp = 0.0f;
        raXnF1Resp = 0.0f;
        invMass = 0.0f;
        linVel = float3(0.0f, 0.0f, 0.0f);
        angVel = float3(0.0f, 0.0f, 0.0f);
        linDelta = float3(0.0f, 0.0f, 0.0f);
        angDelta = float3(0.0f, 0.0f, 0.0f);
    }

    // readContactPrep<Block>: PxgRigidParticleContactBlock and PxgRigidDbContactBlock share
    // raXn_resp @0, raXnF0_resp @512, raXnF1_resp @1024, fricTan0_invMass0 @2048.
    [mutating]
    public void readContactPrep(uint64_t block, uint lane)
    {
        const float4 raXn_resp = Ptr<float4>(block + PXG_RPC_raXn_resp)[lane];
        raXn = float3(raXn_resp.x, raXn_resp.y, raXn_resp.z);
        raXnResp = raXn_resp.w;

        const float4 raXnF0_resp = Ptr<float4>(block + PXG_RPC_raXnF0_resp)[lane];
        raXnF0 = float3(raXnF0_resp.x, raXnF0_resp.y, raXnF0_resp.z);
        raXnF0Resp = raXnF0_resp.w;

        const float4 raXnF1_resp = Ptr<float4>(block + PXG_RPC_raXnF1_resp)[lane];
        raXnF1 = float3(raXnF1_resp.x, raXnF1_resp.y, raXnF1_resp.z);
        raXnF1Resp = raXnF1_resp.w;

        invMass = Ptr<float4>(block + PXG_RPC_fricTan0_invMass0)[lane].w;
    }

    // readAttachmentPrep<PxgRigidDbAttachmentBlock>.
    [mutating]
    public void readAttachmentPrep(uint64_t block, uint lane)
    {
        const float4 raXn0_biasW = Ptr<float4>(block + PXG_RDBA_raXn0_biasW)[lane];
        const float4 raXn1_biasW = Ptr<float4>(block + PXG_RDBA_raXn1_biasW)[lane];
        const float4 raXn2_biasW = Ptr<float4>(block + PXG_RDBA_raXn2_biasW)[lane];
        raXn0 = float3(raXn0_biasW.x, raXn0_biasW.y, raXn0_biasW.z);
        raXn1 = float3(raXn1_biasW.x, raXn1_biasW.y, raXn1_biasW.z);
        raXn2 = float3(raXn2_biasW.x, raXn2_biasW.y, raXn2_biasW.z);
        positionErrorXYZ = float3(raXn0_biasW.w, raXn1_biasW.w, raXn2_biasW.w);

        const float4 velMultXYZ_invMassW = Ptr<float4>(block + PXG_RDBA_velMultiplierXYZ_invMassW)[lane];
        velMultiplierXYZ = float3(velMultXYZ_invMassW.x, velMultXYZ_invMassW.y, velMultXYZ_invMassW.z);
        attachInvMass = velMultXYZ_invMassW.w;

        attachRefCount = Ptr<uint>(block + PXG_RDBA_rigidBodyRefCount)[lane];
    }

    [mutating]
    public void readVelocity(PxgVelocityReaderGpu reader, uint64_t rigidId, bool isTGS)
    {
        if (isTGS)
            readVelocitiesTGS(reader, rigidId, linVel, angVel, linDelta, angDelta);
        else
            readVelocitiesPGS(reader, rigidId, linVel, angVel, linDelta, angDelta);
    }

    public void writeDeltas(uint64_t rigidDeltaVel, float3 deltaLinVel, float3 deltaAngVel,
                            uint linVelIndex, uint angVelIndex, float count)
    {
        Ptr<float4> d = Ptr<float4>(rigidDeltaVel);
        d[linVelIndex] = float4(deltaLinVel.x, deltaLinVel.y, deltaLinVel.z, count);
        d[angVelIndex] = float4(deltaAngVel.x, deltaAngVel.y, deltaAngVel.z, 0.0f);
    }

    public void writeAttachmentDeltas(uint64_t rigidDeltaVel, uint64_t rigidId, float3 deltaLinVel,
                                      float3 deltaAngVel, uint linVelIndex, uint angVelIndex)
    {
        if (PxgNodeIsStaticBody(rigidId))
            return;

        if (attachInvMass == 0.0f)
        {
            writeDeltas(rigidDeltaVel, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), linVelIndex, angVelIndex, 0.0f);
        }
        else
        {
            const float refN = float(attachRefCount);
            writeDeltas(rigidDeltaVel, float3(deltaLinVel.x * refN, deltaLinVel.y * refN, deltaLinVel.z * refN),
                        float3(deltaAngVel.x * refN, deltaAngVel.y * refN, deltaAngVel.z * refN),
                        linVelIndex, angVelIndex, 1.0f);
        }
    }

    // Always writes: the accumulate scan reads every slot of a dynamic rigid's range.
    public void writeContactDeltas(uint64_t rigidDeltaVel, uint64_t rigidId, PxgDbContactPair contactPair,
                                   PxgDbSolveOutput solveOut, uint linVelIndex, uint angVelIndex)
    {
        float3 deltaLinVel = float3(0.0f, 0.0f, 0.0f);
        float3 deltaAngVel = float3(0.0f, 0.0f, 0.0f);
        float count = 0.0f;
        if (invMass != 0.0f && !PxgNodeIsStaticBody(rigidId))
        {
            count = 1.0f;
            const float refCount = float(referenceCount);
            const float dLN = solveOut.deltaLambdaN;
            const float dLT = solveOut.deltaLambdaT;
            // (raXn * dLN + raXt * dLT) * refCount
            deltaAngVel = float3((raXn.x * dLN + solveOut.raXt.x * dLT) * refCount,
                                 (raXn.y * dLN + solveOut.raXt.y * dLT) * refCount,
                                 (raXn.z * dLN + solveOut.raXt.z * dLT) * refCount);
            // ((normal * dLN + tangent * dLT) * invMass) * refCount
            deltaLinVel = float3(((contactPair.normal.x * dLN + solveOut.tangent.x * dLT) * invMass) * refCount,
                                 ((contactPair.normal.y * dLN + solveOut.tangent.y * dLT) * invMass) * refCount,
                                 ((contactPair.normal.z * dLN + solveOut.tangent.z * dLT) * invMass) * refCount);
        }
        writeDeltas(rigidDeltaVel, deltaLinVel, deltaAngVel, linVelIndex, angVelIndex, count);
    }
}

// ---------------------------------------------------------------------------
// Particle reads
// ---------------------------------------------------------------------------
// readParticleLinDelta: mSortedVelocities * dt in velocity iterations, the accumulated
// position delta (mSortedDeltaP.xyz) otherwise.
public float3 readParticleLinDelta(Ptr<PxgParticleSystemGpu> particleSystem, uint particleIndex,
                                   float4 sortedDeltaP, float dt, bool isVelocityIteration)
{
    if (isVelocityIteration)
    {
        const float4 vel = Ptr<float4>(particleSystem[0].mSortedVelocities)[particleIndex];
        return float3(vel.x * dt, vel.y * dt, vel.z * dt);
    }
    return float3(sortedDeltaP.x, sortedDeltaP.y, sortedDeltaP.z);
}

// ---------------------------------------------------------------------------
// Integer-count reference-count bumps: add 1 to .w per touched vertex (a vertex whose
// barycentric weight is within 1e-3 of zero is not counted).  The universal DB refcount
// convention of the softbody, cloth and particle pre-counts, paired with the apply kernels'
// max(.w, 1) finalize divide.  deltaBuf is a float4* (mSimDelta / mDeltaPos).
// ---------------------------------------------------------------------------
public void bumpDbRefCountVtx(uint64_t deltaBuf, uint vertIndex)
{
    pxgAtomicAddF4W(deltaBuf, vertIndex, 1.0f);
}

public void bumpDbRefCountTri(uint64_t deltaBuf, uint4 triVertIndices, float w0, float w1, float w2)
{
    if (abs(w0) > 1e-3f) pxgAtomicAddF4W(deltaBuf, triVertIndices.x, 1.0f);
    if (abs(w1) > 1e-3f) pxgAtomicAddF4W(deltaBuf, triVertIndices.y, 1.0f);
    if (abs(w2) > 1e-3f) pxgAtomicAddF4W(deltaBuf, triVertIndices.z, 1.0f);
}

public void bumpDbRefCountTet(uint64_t deltaBuf, uint4 tetVertIndices, float w0, float w1, float w2, float w3)
{
    if (abs(w0) > 1e-3f) pxgAtomicAddF4W(deltaBuf, tetVertIndices.x, 1.0f);
    if (abs(w1) > 1e-3f) pxgAtomicAddF4W(deltaBuf, tetVertIndices.y, 1.0f);
    if (abs(w2) > 1e-3f) pxgAtomicAddF4W(deltaBuf, tetVertIndices.z, 1.0f);
    if (abs(w3) > 1e-3f) pxgAtomicAddF4W(deltaBuf, tetVertIndices.w, 1.0f);
}

// bumpClothRefCount(cloth, ref, const PxVec3& w): every vertex the referenced feature is
// built from (one for a vertex ref, the triangle's three otherwise).
public void bumpClothRefCount(Ptr<PxgFEMClothGpu> cloth, uint64_t ref, float3 w)
{
    const uint elementId = PxgGetDeformableFeatureId(ref);
    if (PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX)
        bumpDbRefCountVtx(cloth[0].mDeltaPos, elementId);
    else
        bumpDbRefCountTri(cloth[0].mDeltaPos, Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[elementId],
                          w.x, w.y, w.z);
}

// bumpClothRefCount(cloth, ref, const float4* bcCoords, bcIndex): the barycentric is read
// only for a triangle ref.
public void bumpClothRefCountBary(Ptr<PxgFEMClothGpu> cloth, uint64_t ref, uint64_t bcCoords, uint bcIndex)
{
    if (PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX)
        bumpDbRefCountVtx(cloth[0].mDeltaPos, PxgGetDeformableFeatureId(ref));
    else
    {
        const float4 bary = Ptr<float4>(bcCoords)[bcIndex];
        bumpClothRefCount(cloth, ref, float3(bary.x, bary.y, bary.z));
    }
}

public void bumpSoftBodyRefCount(Ptr<PxgSoftBodyGpu> softbody, uint64_t ref, float4 w)
{
    const uint elementId = PxgGetDeformableFeatureId(ref);
    if (PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX)
        bumpDbRefCountVtx(softbody[0].mSimDelta, elementId);
    else
        bumpDbRefCountTet(softbody[0].mSimDelta, Ptr<uint4>(softbody[0].mSimTetIndices)[elementId],
                          w.x, w.y, w.z, w.w);
}

public void bumpSoftBodyRefCountBary(Ptr<PxgSoftBodyGpu> softbody, uint64_t ref, uint64_t bcCoords, uint bcIndex)
{
    if (PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX)
        bumpDbRefCountVtx(softbody[0].mSimDelta, PxgGetDeformableFeatureId(ref));
    else
    {
        const float4 bary = Ptr<float4>(bcCoords)[bcIndex];
        bumpSoftBodyRefCount(softbody, ref, bary);
    }
}

// ---------------------------------------------------------------------------
// Delta scatter (a position delta in position iterations, velocity * dt in velocity
// iterations) into a per-iteration float4 scratch; .xyz only, .w is the refCount.
// Each vertex takes AtomicAdd(float4&, PxVec3): x, y, z in that order.
// ---------------------------------------------------------------------------
public void updateDeltaVtx(uint64_t outputDeltas, uint vertIndex, float3 delta, float invMass)
{
    if (invMass > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, vertIndex, float3(delta.x * invMass, delta.y * invMass, delta.z * invMass));
}

// updateDeltaEdge: the cloth edge-edge contact scatter (writeClothEdge), two vertices with
// their own bc * invMass weights.
public void updateDeltaEdge(uint64_t outputDeltas, uint vertIndex0, uint vertIndex1, float3 delta,
                            float invMassBary0, float invMassBary1)
{
    if (invMassBary0 > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, vertIndex0,
                          float3(delta.x * invMassBary0, delta.y * invMassBary0, delta.z * invMassBary0));
    if (invMassBary1 > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, vertIndex1,
                          float3(delta.x * invMassBary1, delta.y * invMassBary1, delta.z * invMassBary1));
}

public void updateDeltaTri(uint64_t outputDeltas, uint4 triVertIndices, float3 delta, float4 invMassBary)
{
    if (invMassBary.x > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, triVertIndices.x,
                          float3(delta.x * invMassBary.x, delta.y * invMassBary.x, delta.z * invMassBary.x));
    if (invMassBary.y > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, triVertIndices.y,
                          float3(delta.x * invMassBary.y, delta.y * invMassBary.y, delta.z * invMassBary.y));
    if (invMassBary.z > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, triVertIndices.z,
                          float3(delta.x * invMassBary.z, delta.y * invMassBary.z, delta.z * invMassBary.z));
}

public void updateDeltaTet(uint64_t outputDeltas, uint4 tetVertIndices, float3 delta, float4 invMassBary)
{
    if (invMassBary.x > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, tetVertIndices.x,
                          float3(delta.x * invMassBary.x, delta.y * invMassBary.x, delta.z * invMassBary.x));
    if (invMassBary.y > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, tetVertIndices.y,
                          float3(delta.x * invMassBary.y, delta.y * invMassBary.y, delta.z * invMassBary.y));
    if (invMassBary.z > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, tetVertIndices.z,
                          float3(delta.x * invMassBary.z, delta.y * invMassBary.z, delta.z * invMassBary.z));
    if (invMassBary.w > 0.0f)
        pxgAtomicAddF4XYZ(outputDeltas, tetVertIndices.w,
                          float3(delta.x * invMassBary.w, delta.y * invMassBary.w, delta.z * invMassBary.w));
}

// PxsDeformableVolumeMaterialData / PxsDeformableSurfaceMaterialData: both 32-byte records
// with dynamicFriction @8.
public static const uint64_t PXG_DB_MATERIAL_STRIDE = 32;
public static const uint64_t PXG_DB_MATERIAL_DYNAMIC_FRICTION = 8;

float dbMaterialFriction(uint64_t materials, uint globalMaterialIndex)
{
    return Ptr<float>(materials + uint64_t(globalMaterialIndex) * PXG_DB_MATERIAL_STRIDE
                      + PXG_DB_MATERIAL_DYNAMIC_FRICTION)[0];
}

// ---------------------------------------------------------------------------
// PxgDeformablePart (see the header for the Vec template)
// ---------------------------------------------------------------------------
public struct PxgDeformablePart
{
    public uint dim = 3u;   // 3: PxgDeformablePart<PxVec3>, 4: PxgDeformablePart<PxVec4>

    public float friction = 0.0f;
    public float adhesion = 0.0f;
    public float adhesionRadius = 0.0f;

    public float3 linDelta = float3(0.0f, 0.0f, 0.0f);   // CUDA: uninitialised
    public float3 posDelta = float3(0.0f, 0.0f, 0.0f);
    public float4 vertexInvMasses = float4(0.0f, 0.0f, 0.0f, 0.0f);   // CUDA: uninitialised
    public float4 bc = float4(0.0f, 0.0f, 0.0f, 0.0f);                // CUDA: uninitialised
    public uint64_t ref = 0;

    public float attachPointInvMass = 0.0f;
    public float attachPointInvMassSplit = 0.0f;

    // readParticle: bc = (1,0,0), invMass from sortedDeltaP.w (inflated by the mass-split
    // count in mAccumDeltaP[i].w when asked), friction / adhesion from the material when
    // one is given (materialElem == 0 is the CUDA NULL).
    [mutating]
    public void readParticle(Ptr<PxgParticleSystemGpu> particleSystem, uint particleIndex, float4 sortedDeltaP,
                             uint64_t materialElem, bool inflateByRefCount, float dt, bool isVelocityIteration)
    {
        dim = 3u;
        bc = float4(1.0f, 0.0f, 0.0f, 0.0f);

        const float refCount = inflateByRefCount
            ? dpSelectMax(Ptr<float4>(particleSystem[0].mAccumDeltaP)[particleIndex].w, 1.0f)
            : 1.0f;
        vertexInvMasses = float4(sortedDeltaP.w * refCount, 0.0f, 0.0f, 0.0f);

        if (materialElem != 0)
        {
            Ptr<float> mat = Ptr<float>(materialElem);
            friction = mat[PXG_PMAT_FRICTION];
            adhesion = mat[PXG_PMAT_ADHESION];
            adhesionRadius = particleSystem[0].mData_RestOffset * (mat[PXG_PMAT_ADHESION_RADIUS_SCALE] - 1.0f);
        }

        linDelta = readParticleLinDelta(particleSystem, particleIndex, sortedDeltaP, dt, isVelocityIteration);
        posDelta = float3(sortedDeltaP.x, sortedDeltaP.y, sortedDeltaP.z);
    }

    // Particle-side contact writeback: delta * invMass in the per-contact slot, .w flags
    // an applied impulse.
    public void writeParticle(uint64_t deltas, uint slot, float3 delta, PxgDbSolveOutput solveOut)
    {
        const float m = vertexInvMasses.x;
        const float applied = (solveOut.deltaLambdaN != 0.0f || solveOut.deltaLambdaT != 0.0f) ? 1.0f : 0.0f;
        Ptr<float4>(deltas)[slot] = float4(delta.x * m, delta.y * m, delta.z * m, applied);
    }

    // -----------------------------------------------------------------------
    // Softbody half (PxgDeformablePart<PxVec4>).
    // -----------------------------------------------------------------------

    // readSoftBody: a TETRAHEDRON contact participant.  linDelta is the accumulated position
    // delta (mSimDeltaPos), or in velocity iterations the velocity (mSimVelocity_InvMass)
    // scaled by dt; posDelta is always the accumulated position delta.  Unless
    // checkOnlyActivity, friction comes from the per-tet material (materials == 0 is the CUDA
    // NULL) and vertexInvMasses are inflated by the pre-counted refCounts in mSimDelta.w.
    // bcCoords is a float4* read at [bcIndex].
    [mutating]
    public float4 readSoftBody(Ptr<PxgSoftBodyGpu> softbody, uint64_t ref, uint64_t bcCoords, uint bcIndex,
                               uint64_t materials, bool checkOnlyActivity, float dt, bool isVelocityIteration)
    {
        dim = 4u;
        this.ref = ref;
        const uint tetId = PxgGetDeformableFeatureId(ref);
        const float4 bary = Ptr<float4>(bcCoords)[bcIndex];
        bc = bary;

        const uint64_t posDeltasOrVelocities = isVelocityIteration ? softbody[0].mSimVelocity_InvMass : softbody[0].mSimDeltaPos;
        Ptr<float4> pdv = Ptr<float4>(posDeltasOrVelocities);

        const uint4 tetVertId = Ptr<uint4>(softbody[0].mSimTetIndices)[tetId];
        const float4 pdv0 = pdv[tetVertId.x];
        const float4 pdv1 = pdv[tetVertId.y];
        const float4 pdv2 = pdv[tetVertId.z];
        const float4 pdv3 = pdv[tetVertId.w];

        const float4 posDeltaOrVelocity = pdv0 * bc.x + pdv1 * bc.y + pdv2 * bc.z + pdv3 * bc.w;
        linDelta = float3(posDeltaOrVelocity.x, posDeltaOrVelocity.y, posDeltaOrVelocity.z);

        if (isVelocityIteration)
        {
            linDelta = float3(linDelta.x * dt, linDelta.y * dt, linDelta.z * dt);
            Ptr<float4> simDeltaPos = Ptr<float4>(softbody[0].mSimDeltaPos);
            const float4 accumulatedDelta = simDeltaPos[tetVertId.x] * bc.x + simDeltaPos[tetVertId.y] * bc.y +
                                            simDeltaPos[tetVertId.z] * bc.z + simDeltaPos[tetVertId.w] * bc.w;
            posDelta = float3(accumulatedDelta.x, accumulatedDelta.y, accumulatedDelta.z);
        }
        else
        {
            posDelta = linDelta;
        }

        vertexInvMasses = float4(pdv0.w, pdv1.w, pdv2.w, pdv3.w);

        if (!checkOnlyActivity)
        {
            const uint globalMaterialIndex = PxgLoadU16(softbody[0].mMaterialIndices, tetId);
            friction = (materials != 0UL) ? dbMaterialFriction(materials, globalMaterialIndex) : 0.0f;

            Ptr<float4> simDelta = Ptr<float4>(softbody[0].mSimDelta);
            const float4 vertexRefCount = float4(simDelta[tetVertId.x].w, simDelta[tetVertId.y].w,
                                                 simDelta[tetVertId.z].w, simDelta[tetVertId.w].w);
            vertexInvMasses = float4(vertexInvMasses.x * vertexRefCount.x, vertexInvMasses.y * vertexRefCount.y,
                                     vertexInvMasses.z * vertexRefCount.z, vertexInvMasses.w * vertexRefCount.w);
        }

        return vertexInvMasses;
    }

    // readSoftBodyAttachment (RB-DB attachments): PGS always reads the velocity buffer,
    // TGS the accumulated delta except in velocity iterations; vertexInvMasses refCount-
    // inflated, attachPointInvMass the raw bc^2 * invM, attachPointInvMassSplit the inflated
    // one.  A VERTEX ref reads one vertex and leaves bc zero.
    [mutating]
    public void readSoftBodyAttachment(Ptr<PxgSoftBodyGpu> softbody, uint64_t ref, uint64_t bcCoords,
                                       uint bcIndex, bool isTGS, float dt, bool isVelocityIteration)
    {
        dim = 4u;
        this.ref = ref;
        const uint elemIdx = PxgGetDeformableFeatureId(ref);
        const bool elemIsVertex = PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX;

        const uint64_t source = (isTGS && !isVelocityIteration) ? softbody[0].mSimDeltaPos : softbody[0].mSimVelocity_InvMass;
        Ptr<float4> pdv = Ptr<float4>(source);
        Ptr<float4> simDelta = Ptr<float4>(softbody[0].mSimDelta);

        if (elemIsVertex)
        {
            const float4 posDeltaOrVelocity = pdv[elemIdx];
            const float refCount_v = simDelta[elemIdx].w;
            linDelta = float3(posDeltaOrVelocity.x, posDeltaOrVelocity.y, posDeltaOrVelocity.z);
            vertexInvMasses = float4(posDeltaOrVelocity.w * refCount_v, 0.0f, 0.0f, 0.0f);
            attachPointInvMass = posDeltaOrVelocity.w;
            attachPointInvMassSplit = posDeltaOrVelocity.w * refCount_v;
            bc = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else
        {
            const float4 bary = Ptr<float4>(bcCoords)[bcIndex];

            const uint4 tetVertId = Ptr<uint4>(softbody[0].mSimTetIndices)[elemIdx];
            const float4 pdv0 = pdv[tetVertId.x];
            const float4 pdv1 = pdv[tetVertId.y];
            const float4 pdv2 = pdv[tetVertId.z];
            const float4 pdv3 = pdv[tetVertId.w];
            const float4 posDeltaOrVelocity = pdv0 * bary.x + pdv1 * bary.y + pdv2 * bary.z + pdv3 * bary.w;
            linDelta = float3(posDeltaOrVelocity.x, posDeltaOrVelocity.y, posDeltaOrVelocity.z);

            const float r0 = simDelta[tetVertId.x].w;
            const float r1 = simDelta[tetVertId.y].w;
            const float r2 = simDelta[tetVertId.z].w;
            const float r3 = simDelta[tetVertId.w].w;

            vertexInvMasses = float4(pdv0.w * r0, pdv1.w * r1, pdv2.w * r2, pdv3.w * r3);

            const float bx2 = bary.x * bary.x;
            const float by2 = bary.y * bary.y;
            const float bz2 = bary.z * bary.z;
            const float bw2 = bary.w * bary.w;
            attachPointInvMass = bx2 * pdv0.w + by2 * pdv1.w + bz2 * pdv2.w + bw2 * pdv3.w;
            attachPointInvMassSplit = bx2 * pdv0.w * r0 + by2 * pdv1.w * r1 + bz2 * pdv2.w * r2 + bw2 * pdv3.w * r3;
            bc = bary;
        }

        if (isTGS && isVelocityIteration)
            linDelta = float3(linDelta.x * dt, linDelta.y * dt, linDelta.z * dt);
    }

    // readSoftBodyAttachmentDbDb (DB-DB attachments): returns the bc-weighted position, or in
    // velocity iterations the bc-weighted velocity * dt.  attachPointInvMass stays 0.
    [mutating]
    public float3 readSoftBodyAttachmentDbDb(Ptr<PxgSoftBodyGpu> softbody, uint64_t ref, uint64_t bcCoords,
                                             uint bcIndex, float dt, bool isVelocityIteration)
    {
        dim = 4u;
        this.ref = ref;
        const uint elemIdx = PxgGetDeformableFeatureId(ref);
        const bool elemIsVertex = PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX;

        Ptr<float4> pv = Ptr<float4>(isVelocityIteration ? softbody[0].mSimVelocity_InvMass : softbody[0].mSimPosition_InvMass);
        Ptr<float4> simDelta = Ptr<float4>(softbody[0].mSimDelta);

        if (elemIsVertex)
        {
            const float4 positionOrVelocity = pv[elemIdx];
            const float r = simDelta[elemIdx].w;

            vertexInvMasses = float4(positionOrVelocity.w * r, 0.0f, 0.0f, 0.0f);
            bc = float4(0.0f, 0.0f, 0.0f, 0.0f);
            attachPointInvMassSplit = positionOrVelocity.w * r;

            float3 vertexResult = float3(positionOrVelocity.x, positionOrVelocity.y, positionOrVelocity.z);
            if (isVelocityIteration)
                vertexResult = float3(vertexResult.x * dt, vertexResult.y * dt, vertexResult.z * dt);
            return vertexResult;
        }

        const float4 bary = Ptr<float4>(bcCoords)[bcIndex];

        const uint4 tetVertId = Ptr<uint4>(softbody[0].mSimTetIndices)[elemIdx];
        const float4 pv0 = pv[tetVertId.x];
        const float4 pv1 = pv[tetVertId.y];
        const float4 pv2 = pv[tetVertId.z];
        const float4 pv3 = pv[tetVertId.w];
        const float4 positionOrVelocity = pv0 * bary.x + pv1 * bary.y + pv2 * bary.z + pv3 * bary.w;

        const float r0 = simDelta[tetVertId.x].w;
        const float r1 = simDelta[tetVertId.y].w;
        const float r2 = simDelta[tetVertId.z].w;
        const float r3 = simDelta[tetVertId.w].w;

        vertexInvMasses = float4(pv0.w * r0, pv1.w * r1, pv2.w * r2, pv3.w * r3);
        bc = bary;

        const float bx2 = bary.x * bary.x;
        const float by2 = bary.y * bary.y;
        const float bz2 = bary.z * bary.z;
        const float bw2 = bary.w * bary.w;
        attachPointInvMassSplit = bx2 * pv0.w * r0 + by2 * pv1.w * r1 + bz2 * pv2.w * r2 + bw2 * pv3.w * r3;

        float3 result = float3(positionOrVelocity.x, positionOrVelocity.y, positionOrVelocity.z);
        if (isVelocityIteration)
            result = float3(result.x * dt, result.y * dt, result.z * dt);
        return result;
    }

    // writeSoftBody: scatter into mSimDelta, weighted per vertex by bc * vertexInvMasses.
    public void writeSoftBody(Ptr<PxgSoftBodyGpu> softbody, float3 delta)
    {
        const uint elementId = PxgGetDeformableFeatureId(ref);
        if (PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX)
            updateDeltaVtx(softbody[0].mSimDelta, elementId, delta, vertexInvMasses.x);
        else
            updateDeltaTet(softbody[0].mSimDelta, Ptr<uint4>(softbody[0].mSimTetIndices)[elementId], delta,
                           float4(vertexInvMasses.x * bc.x, vertexInvMasses.y * bc.y,
                                  vertexInvMasses.z * bc.z, vertexInvMasses.w * bc.w));
    }

    // The weights are already resolved on this part.
    public void bumpSoftBodyRefCountPart(Ptr<PxgSoftBodyGpu> softbody)
    {
        bumpSoftBodyRefCount(softbody, ref, bc);
    }

    // -----------------------------------------------------------------------
    // Cloth half (PxgDeformablePart<PxVec3>).
    // -----------------------------------------------------------------------

    // readCloth: a triangle or vertex contact participant.  linDelta from
    // mAccumulatedDeltaPos, or in velocity iterations mVelocity_InvMass * dt.  Unless
    // countReferenceOnly: a triangle's friction from its material (materials == 0 is NULL),
    // a vertex's from mDynamicFrictions (when materials != 0), and the refCount inflation
    // from mDeltaPos.w.
    [mutating]
    public float4 readCloth(Ptr<PxgFEMClothGpu> cloth, uint64_t ref, uint64_t bcCoords, uint bcIndex,
                            uint64_t materials, bool countReferenceOnly, float dt, bool isVelocityIteration)
    {
        dim = 3u;
        this.ref = ref;
        const uint elementId = PxgGetDeformableFeatureId(ref);
        const bool elemIsVertex = PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX;

        Ptr<float4> pdv = Ptr<float4>(isVelocityIteration ? cloth[0].mVelocity_InvMass : cloth[0].mAccumulatedDeltaPos);
        float4 posDeltaOrVelocity;

        if (!elemIsVertex)
        {
            const float4 bary = Ptr<float4>(bcCoords)[bcIndex];
            bc = float4(bary.x, bary.y, bary.z, 0.0f);

            const uint4 triVertId = Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[elementId];
            const float4 pdv0 = pdv[triVertId.x];
            const float4 pdv1 = pdv[triVertId.y];
            const float4 pdv2 = pdv[triVertId.z];

            posDeltaOrVelocity = pdv0 * bc.x + pdv1 * bc.y + pdv2 * bc.z;
            vertexInvMasses = float4(pdv0.w, pdv1.w, pdv2.w, 0.0f);

            if (isVelocityIteration)
            {
                Ptr<float4> acc = Ptr<float4>(cloth[0].mAccumulatedDeltaPos);
                const float4 accumulatedDelta = acc[triVertId.x] * bc.x + acc[triVertId.y] * bc.y + acc[triVertId.z] * bc.z;
                posDelta = float3(accumulatedDelta.x, accumulatedDelta.y, accumulatedDelta.z);
            }

            if (!countReferenceOnly)
            {
                const uint globalMaterialIndex = PxgLoadU16(cloth[0].mMaterialIndices, elementId);
                friction = (materials != 0UL) ? dbMaterialFriction(materials, globalMaterialIndex) : 0.0f;

                Ptr<float4> deltaPos = Ptr<float4>(cloth[0].mDeltaPos);
                const float3 vertexRefCount = float3(deltaPos[triVertId.x].w, deltaPos[triVertId.y].w, deltaPos[triVertId.z].w);
                vertexInvMasses = float4(vertexInvMasses.x * vertexRefCount.x, vertexInvMasses.y * vertexRefCount.y,
                                         vertexInvMasses.z * vertexRefCount.z, 0.0f);
            }
        }
        else
        {
            bc = float4(1.0f, 0.0f, 0.0f, 0.0f);

            posDeltaOrVelocity = pdv[elementId];
            vertexInvMasses = float4(posDeltaOrVelocity.w, 0.0f, 0.0f, 0.0f);

            if (isVelocityIteration)
            {
                const float4 accumulatedDelta = Ptr<float4>(cloth[0].mAccumulatedDeltaPos)[elementId];
                posDelta = float3(accumulatedDelta.x, accumulatedDelta.y, accumulatedDelta.z);
            }

            if (!countReferenceOnly)
            {
                friction = (materials != 0UL) ? Ptr<float>(cloth[0].mDynamicFrictions)[elementId] : 0.0f;

                const float vertexRefCount = Ptr<float4>(cloth[0].mDeltaPos)[elementId].w;
                vertexInvMasses.x = vertexInvMasses.x * vertexRefCount;
            }
        }

        linDelta = float3(posDeltaOrVelocity.x, posDeltaOrVelocity.y, posDeltaOrVelocity.z);
        if (isVelocityIteration)
            linDelta = float3(linDelta.x * dt, linDelta.y * dt, linDelta.z * dt);
        else
            posDelta = linDelta;

        return vertexInvMasses;
    }

    // readClothAttachment (RB-DB attachments): the cloth analog of readSoftBodyAttachment.
    [mutating]
    public void readClothAttachment(Ptr<PxgFEMClothGpu> cloth, uint64_t ref, uint64_t bcCoords, uint bcIndex,
                                    bool isTGS, float dt, bool isVelocityIteration)
    {
        dim = 3u;
        this.ref = ref;
        const uint elemIdx = PxgGetDeformableFeatureId(ref);
        const bool elemIsVertex = PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX;

        Ptr<float4> pdv = Ptr<float4>((isTGS && !isVelocityIteration) ? cloth[0].mAccumulatedDeltaPos : cloth[0].mVelocity_InvMass);
        Ptr<float4> delta = Ptr<float4>(cloth[0].mDeltaPos);

        if (elemIsVertex)
        {
            const float4 posDeltaOrVelocity = pdv[elemIdx];
            const float refCount_v = delta[elemIdx].w;
            linDelta = float3(posDeltaOrVelocity.x, posDeltaOrVelocity.y, posDeltaOrVelocity.z);
            vertexInvMasses = float4(posDeltaOrVelocity.w * refCount_v, 0.0f, 0.0f, 0.0f);
            attachPointInvMass = posDeltaOrVelocity.w;
            attachPointInvMassSplit = posDeltaOrVelocity.w * refCount_v;
            bc = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else
        {
            const float4 bary = Ptr<float4>(bcCoords)[bcIndex];

            const uint4 triVertId = Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[elemIdx];
            const float4 pdv0 = pdv[triVertId.x];
            const float4 pdv1 = pdv[triVertId.y];
            const float4 pdv2 = pdv[triVertId.z];
            const float4 posDeltaOrVelocity = pdv0 * bary.x + pdv1 * bary.y + pdv2 * bary.z;
            linDelta = float3(posDeltaOrVelocity.x, posDeltaOrVelocity.y, posDeltaOrVelocity.z);

            const float r0 = delta[triVertId.x].w;
            const float r1 = delta[triVertId.y].w;
            const float r2 = delta[triVertId.z].w;

            vertexInvMasses = float4(pdv0.w * r0, pdv1.w * r1, pdv2.w * r2, 0.0f);

            const float bx2 = bary.x * bary.x;
            const float by2 = bary.y * bary.y;
            const float bz2 = bary.z * bary.z;
            attachPointInvMass = bx2 * pdv0.w + by2 * pdv1.w + bz2 * pdv2.w;
            attachPointInvMassSplit = bx2 * pdv0.w * r0 + by2 * pdv1.w * r1 + bz2 * pdv2.w * r2;
            bc = float4(bary.x, bary.y, bary.z, 0.0f);
        }

        if (isTGS && isVelocityIteration)
            linDelta = float3(linDelta.x * dt, linDelta.y * dt, linDelta.z * dt);
    }

    // readClothAttachmentDbDb: the cloth analog of readSoftBodyAttachmentDbDb.
    [mutating]
    public float3 readClothAttachmentDbDb(Ptr<PxgFEMClothGpu> cloth, uint64_t ref, uint64_t bcCoords,
                                          uint bcIndex, float dt, bool isVelocityIteration)
    {
        dim = 3u;
        this.ref = ref;
        const uint elementId = PxgGetDeformableFeatureId(ref);
        const bool elemIsVertex = PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX;

        Ptr<float4> pv = Ptr<float4>(isVelocityIteration ? cloth[0].mVelocity_InvMass : cloth[0].mPosition_InvMass);
        Ptr<float4> delta = Ptr<float4>(cloth[0].mDeltaPos);

        if (elemIsVertex)
        {
            const float4 positionOrVelocity = pv[elementId];
            const float r = delta[elementId].w;

            vertexInvMasses = float4(positionOrVelocity.w * r, 0.0f, 0.0f, 0.0f);
            bc = float4(0.0f, 0.0f, 0.0f, 0.0f);
            attachPointInvMassSplit = positionOrVelocity.w * r;

            float3 vertexResult = float3(positionOrVelocity.x, positionOrVelocity.y, positionOrVelocity.z);
            if (isVelocityIteration)
                vertexResult = float3(vertexResult.x * dt, vertexResult.y * dt, vertexResult.z * dt);
            return vertexResult;
        }

        const float4 bary = Ptr<float4>(bcCoords)[bcIndex];

        const uint4 triVertId = Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[elementId];
        const float4 pv0 = pv[triVertId.x];
        const float4 pv1 = pv[triVertId.y];
        const float4 pv2 = pv[triVertId.z];
        const float4 positionOrVelocity = pv0 * bary.x + pv1 * bary.y + pv2 * bary.z;

        const float r0 = delta[triVertId.x].w;
        const float r1 = delta[triVertId.y].w;
        const float r2 = delta[triVertId.z].w;

        vertexInvMasses = float4(pv0.w * r0, pv1.w * r1, pv2.w * r2, 0.0f);
        bc = float4(bary.x, bary.y, bary.z, 0.0f);

        const float bx2 = bary.x * bary.x;
        const float by2 = bary.y * bary.y;
        const float bz2 = bary.z * bary.z;
        attachPointInvMassSplit = bx2 * pv0.w * r0 + by2 * pv1.w * r1 + bz2 * pv2.w * r2;

        float3 result = float3(positionOrVelocity.x, positionOrVelocity.y, positionOrVelocity.z);
        if (isVelocityIteration)
            result = float3(result.x * dt, result.y * dt, result.z * dt);
        return result;
    }

    // writeCloth: scatter into mDeltaPos, weighted per vertex by bc * vertexInvMasses.
    public void writeCloth(Ptr<PxgFEMClothGpu> cloth, float3 delta)
    {
        const uint elementId = PxgGetDeformableFeatureId(ref);
        if (PxgGetDeformableFeatureType(ref) == PXG_DREF_eVERTEX)
            updateDeltaVtx(cloth[0].mDeltaPos, elementId, delta, vertexInvMasses.x);
        else
            updateDeltaTri(cloth[0].mDeltaPos, Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[elementId], delta,
                           float4(vertexInvMasses.x * bc.x, vertexInvMasses.y * bc.y, vertexInvMasses.z * bc.z, 0.0f));
    }

    // writeClothEdge: the edge variant for the cloth-cloth EE solve, each side a cloth edge
    // (vertIdx0, vertIdx1) weighted by vertexInvMasses.x * bc.x / vertexInvMasses.y * bc.y.
    public void writeClothEdge(Ptr<PxgFEMClothGpu> cloth, uint vertIdx0, uint vertIdx1, float3 delta)
    {
        updateDeltaEdge(cloth[0].mDeltaPos, vertIdx0, vertIdx1, delta, vertexInvMasses.x * bc.x,
                        vertexInvMasses.y * bc.y);
    }

    public void bumpClothRefCountPart(Ptr<PxgFEMClothGpu> cloth)
    {
        bumpClothRefCount(cloth, ref, float3(bc.x, bc.y, bc.z));
    }
}

// ---------------------------------------------------------------------------
// solveRigidDbAttachment<Vec>: the per-axis rigid-vs-deformable attachment impulse.
// Returns deltaImpulse for the deformable writeback; outDeltaLinVel / outDeltaAngVel are the
// rigid writeback.  PGS works in velocity, TGS in position deltas scaled by invDt.
// ---------------------------------------------------------------------------
public float3 solveRigidDbAttachment(PxgRigidPart rigid, PxgDeformablePart db, float dt, float biasCoefficient,
                                     bool isTGS, bool isVelocityIteration, out float3 outDeltaLinVel,
                                     out float3 outDeltaAngVel)
{
    const float invDt = 1.0f / dt;

    const float velOfRigidAtAttachmentPointX = rigid.linVel.x + dpDot3(rigid.angVel, rigid.raXn0);
    const float velOfRigidAtAttachmentPointY = rigid.linVel.y + dpDot3(rigid.angVel, rigid.raXn1);
    const float velOfRigidAtAttachmentPointZ = rigid.linVel.z + dpDot3(rigid.angVel, rigid.raXn2);

    const float refN = float(rigid.attachRefCount);
    const float denomBias = refN * db.attachPointInvMass - db.attachPointInvMassSplit;
    const float adjVelMult0 = 1.0f / (refN / rigid.velMultiplierXYZ.x - denomBias);
    const float adjVelMult1 = 1.0f / (refN / rigid.velMultiplierXYZ.y - denomBias);
    const float adjVelMult2 = 1.0f / (refN / rigid.velMultiplierXYZ.z - denomBias);

    float deltaF0, deltaF1, deltaF2;
    if (isTGS)
    {
        const float3 rigidLinDelta = isVelocityIteration ? float3(0.0f, 0.0f, 0.0f) : rigid.linDelta;
        const float3 rigidAngDelta = isVelocityIteration ? float3(0.0f, 0.0f, 0.0f) : rigid.angDelta;
        const float3 linDelta = float3(db.linDelta.x - rigidLinDelta.x, db.linDelta.y - rigidLinDelta.y,
                                       db.linDelta.z - rigidLinDelta.z);
        const float3 tgsErrVel = float3(
            (linDelta.x - dpDot3(rigidAngDelta, rigid.raXn0) - rigid.positionErrorXYZ.x * biasCoefficient) * invDt,
            (linDelta.y - dpDot3(rigidAngDelta, rigid.raXn1) - rigid.positionErrorXYZ.y * biasCoefficient) * invDt,
            (linDelta.z - dpDot3(rigidAngDelta, rigid.raXn2) - rigid.positionErrorXYZ.z * biasCoefficient) * invDt);
        deltaF0 = (tgsErrVel.x - velOfRigidAtAttachmentPointX) * adjVelMult0;
        deltaF1 = (tgsErrVel.y - velOfRigidAtAttachmentPointY) * adjVelMult1;
        deltaF2 = (tgsErrVel.z - velOfRigidAtAttachmentPointZ) * adjVelMult2;
    }
    else
    {
        // positionErrorXYZ.x * biasCoefficient * invDt, left to right.
        const float3 velError = float3(
            db.linDelta.x - rigid.positionErrorXYZ.x * biasCoefficient * invDt,
            db.linDelta.y - rigid.positionErrorXYZ.y * biasCoefficient * invDt,
            db.linDelta.z - rigid.positionErrorXYZ.z * biasCoefficient * invDt);
        deltaF0 = (velError.x - velOfRigidAtAttachmentPointX) * adjVelMult0;
        deltaF1 = (velError.y - velOfRigidAtAttachmentPointY) * adjVelMult1;
        deltaF2 = (velError.z - velOfRigidAtAttachmentPointZ) * adjVelMult2;
    }

    const float3 deltaImpulse = float3(deltaF0, deltaF1, deltaF2);
    outDeltaLinVel = float3(deltaImpulse.x * rigid.attachInvMass, deltaImpulse.y * rigid.attachInvMass,
                            deltaImpulse.z * rigid.attachInvMass);
    // (raXn0 * deltaF0 + raXn1 * deltaF1) + raXn2 * deltaF2
    outDeltaAngVel = float3(rigid.raXn0.x * deltaF0 + rigid.raXn1.x * deltaF1 + rigid.raXn2.x * deltaF2,
                            rigid.raXn0.y * deltaF0 + rigid.raXn1.y * deltaF1 + rigid.raXn2.y * deltaF2,
                            rigid.raXn0.z * deltaF0 + rigid.raXn1.z * deltaF1 + rigid.raXn2.z * deltaF2);

    return deltaImpulse;
}

// ---------------------------------------------------------------------------
// solveDbDbAttachment<Vec0, Vec1>: error = pos1 - pos0, outDelta = error * (1 / wTot) with
// wTot the two sides' attachPointInvMassSplit; false (and outDelta zero) iff wTot == 0.
// ---------------------------------------------------------------------------
public bool solveDbDbAttachment(PxgDeformablePart db0, PxgDeformablePart db1, float3 pos0, float3 pos1,
                                out float3 outDelta)
{
    const float3 error = float3(pos1.x - pos0.x, pos1.y - pos0.y, pos1.z - pos0.z);
    const float wTot = db0.attachPointInvMassSplit + db1.attachPointInvMassSplit;
    if (wTot > 0.0f)
    {
        const float s = 1.0f / wTot;
        outDelta = float3(error.x * s, error.y * s, error.z * s);
        return true;
    }
    outDelta = float3(0.0f, 0.0f, 0.0f);
    return false;
}

// bc.multiply(bc).dot(vertexInvMasses) over the instantiation's dimension.
public float dbBcResponse(PxgDeformablePart db)
{
    const float4 b = db.bc;
    const float4 m = db.vertexInvMasses;
    if (db.dim == 4u)
        return (b.x * b.x) * m.x + (b.y * b.y) * m.y + (b.z * b.z) * m.z + (b.w * b.w) * m.w;
    return (b.x * b.x) * m.x + (b.y * b.y) * m.y + (b.z * b.z) * m.z;
}

// ---------------------------------------------------------------------------
// particleAdhesionImpulse
// ---------------------------------------------------------------------------
public float particleAdhesionImpulse(float separation, float adhesion, float adhesionRadius,
                                     float velMultiplier, float invDt, bool isVelocityIteration)
{
    if (isVelocityIteration || adhesion <= 0.0f || adhesionRadius <= 0.0f)
        return 0.0f;
    const float t = 1.0f - separation / adhesionRadius;
    return -dpSelectMax(0.0f, adhesion * (t * t * t * t) * dpSelectMax(0.0f, separation) * velMultiplier * invDt);
}

// ---------------------------------------------------------------------------
// solveRigidDbContact<Vec, withAdhesion>
// ---------------------------------------------------------------------------
public bool solveRigidDbContact(PxgRigidPart rigid, PxgDeformablePart db, PxgDbContactPair contactPair,
                                inout PxgDbSolveOutput solveOut, float appliedForceRef, float appliedTanLambdaRef,
                                float dt, bool wasActive, bool checkOnlyActivity, bool isVelocityIteration,
                                bool withAdhesion)
{
    solveOut.accumulatedDeltaLambdaN = appliedForceRef;
    const float appliedTanRef = appliedTanLambdaRef;

    const float threshold = 1.0e-14f;
    const float invDt = 1.0f / dt;

    const float3 n = contactPair.normal;
    const float separation = ((contactPair.initPen + dpDot3(rigid.linDelta, n))
                              + dpDot3(rigid.angDelta, rigid.raXn)) - dpDot3(db.posDelta, n);

    const float3 rigidLinDelta = isVelocityIteration ? float3(0.0f, 0.0f, 0.0f) : rigid.linDelta;
    const float3 rigidAngDelta = isVelocityIteration ? float3(0.0f, 0.0f, 0.0f) : rigid.angDelta;
    const float3 relLinDelta = float3(rigidLinDelta.x - db.linDelta.x, rigidLinDelta.y - db.linDelta.y,
                                      rigidLinDelta.z - db.linDelta.z);
    const float normalVel = dpDot3(rigid.linVel, n) + dpDot3(rigid.angVel, rigid.raXn);

    float CN;
    if (isVelocityIteration)
        CN = normalVel + (dpSelectMax(0.0f, separation) + dpDot3(relLinDelta, n)) * invDt;
    else
        CN = normalVel + dpSelectMax(contactPair.maxPenBiasClamp, separation * invDt);

    const bool isActive = wasActive || CN < 0.0f;
    if (checkOnlyActivity || !isActive)
        return isActive;

    const float deformableInvMass_massSplitting = dbBcResponse(db);

    const float rigidRefCount = float(rigid.referenceCount);
    const float unitResponse = rigidRefCount * rigid.raXnResp + deformableInvMass_massSplitting;
    const float invDenom = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

    if (withAdhesion)
    {
        const float adhesionImpulse =
            particleAdhesionImpulse(separation, db.adhesion, db.adhesionRadius, invDenom, invDt, isVelocityIteration);
        const float repulsionIncrement = dpSelectMax((-CN) * invDenom - adhesionImpulse, -solveOut.accumulatedDeltaLambdaN);
        solveOut.deltaLambdaN = repulsionIncrement + adhesionImpulse;
        solveOut.accumulatedDeltaLambdaN += repulsionIncrement;
    }
    else
    {
        solveOut.deltaLambdaN = dpSelectMax((-CN) * invDenom, -solveOut.accumulatedDeltaLambdaN);
        solveOut.accumulatedDeltaLambdaN += solveOut.deltaLambdaN;
    }

    const float3 fric0 = contactPair.tangent0;
    const float3 fric1 = dpCross3(n, fric0);

    const float tanVel0 = dpDot3(rigid.linVel, fric0) + dpDot3(rigid.angVel, rigid.raXnF0);
    const float tanVel1 = dpDot3(rigid.linVel, fric1) + dpDot3(rigid.angVel, rigid.raXnF1);

    const float CT0 = (dpDot3(fric0, relLinDelta) + dpDot3(rigidAngDelta, rigid.raXnF0)) * invDt + tanVel0;
    const float CT1 = (dpDot3(fric1, relLinDelta) + dpDot3(rigidAngDelta, rigid.raXnF1)) * invDt + tanVel1;
    const float3 relTanDelta = float3(CT0 * fric0.x + CT1 * fric1.x, CT0 * fric0.y + CT1 * fric1.y,
                                      CT0 * fric0.z + CT1 * fric1.z);
    const float tanMagSq = dpDot3(relTanDelta, relTanDelta);

    if (tanMagSq > threshold)
    {
        const float CT = sqrt(tanMagSq);
        const float invTanMag = 1.0f / CT;
        solveOut.tangent = float3(relTanDelta.x * invTanMag, relTanDelta.y * invTanMag, relTanDelta.z * invTanMag);

        const float frac0 = dpDot3(solveOut.tangent, fric0);
        const float frac1 = dpDot3(solveOut.tangent, fric1);
        solveOut.raXt = float3(frac0 * rigid.raXnF0.x + frac1 * rigid.raXnF1.x,
                               frac0 * rigid.raXnF0.y + frac1 * rigid.raXnF1.y,
                               frac0 * rigid.raXnF0.z + frac1 * rigid.raXnF1.z);

        const float unitResponseT0 = rigidRefCount * rigid.raXnF0Resp + deformableInvMass_massSplitting;
        const float invTanDenom0 = (unitResponseT0 > 0.0f) ? (1.0f / unitResponseT0) : 0.0f;

        const float unitResponseT1 = rigidRefCount * rigid.raXnF1Resp + deformableInvMass_massSplitting;
        const float invTanDenom1 = (unitResponseT1 > 0.0f) ? (1.0f / unitResponseT1) : 0.0f;

        const float deltaLambdaT0 = CT0 * invTanDenom0;
        const float deltaLambdaT1 = CT1 * invTanDenom1;

        const float combinedFriction = combineScalars(rigid.friction, db.friction, rigid.fricCombineMode);

        const float requiredLambdaTNeg = -sqrt(deltaLambdaT0 * deltaLambdaT0 + deltaLambdaT1 * deltaLambdaT1);
        const float frictionLimit = combinedFriction * solveOut.accumulatedDeltaLambdaN;
        const float targetAccumLambdaT = dpSelectMax(appliedTanRef + requiredLambdaTNeg, -frictionLimit);
        solveOut.deltaLambdaT = targetAccumLambdaT - appliedTanRef;
    }

    return solveOut.deltaLambdaN != 0.0f || solveOut.deltaLambdaT != 0.0f;
}

// ---------------------------------------------------------------------------
// solveDbDbContact<Vec0, Vec1>
// ---------------------------------------------------------------------------
public bool solveDbDbContact(PxgDeformablePart db0, PxgDeformablePart db1, PxgDbContactPair contactPair,
                             inout PxgDbSolveOutput solveOut, float appliedNormalLambdaRef,
                             float appliedTanLambdaRef, float frictionCoefficient, float dt,
                             bool checkOnlyActivity, bool isVelocityIteration)
{
    solveOut.accumulatedDeltaLambdaN = appliedNormalLambdaRef;
    const float appliedTanRef = appliedTanLambdaRef;
    const float invDt = 1.0f / dt;

    const float3 n = contactPair.normal;
    const float3 relLinDelta = float3(db1.linDelta.x - db0.linDelta.x, db1.linDelta.y - db0.linDelta.y,
                                      db1.linDelta.z - db0.linDelta.z);

    const float3 relPosDelta = float3(db1.posDelta.x - db0.posDelta.x, db1.posDelta.y - db0.posDelta.y,
                                      db1.posDelta.z - db0.posDelta.z);
    const float separation = contactPair.initPen + dpDot3(relPosDelta, n);

    float CN;
    if (isVelocityIteration)
        CN = (dpSelectMax(0.0f, separation) + dpDot3(relLinDelta, n)) * invDt;
    else
        CN = dpSelectMax(contactPair.maxPenBiasClamp, separation * invDt);

    const bool isActive = CN < 0.0f;
    if (checkOnlyActivity || !isActive)
        return isActive;

    const float db0Resp = dbBcResponse(db0);
    const float db1Resp = dbBcResponse(db1);
    const float unitResponse = db0Resp + db1Resp;
    const float invDenom = (unitResponse > 1e-16f) ? (1.0f / unitResponse) : 0.0f;

    solveOut.deltaLambdaN = dpSelectMax((-CN) * invDenom, -solveOut.accumulatedDeltaLambdaN);
    solveOut.accumulatedDeltaLambdaN += solveOut.deltaLambdaN;

    const float normalProj = dpDot3(relLinDelta, n);
    float3 tanDir = float3(relLinDelta.x - n.x * normalProj, relLinDelta.y - n.y * normalProj,
                           relLinDelta.z - n.z * normalProj);
    const float tanDelta = dpNormalize3(tanDir);
    solveOut.tangent = tanDir;

    const float requiredLambdaTNeg = (-tanDelta) * invDt * invDenom;
    const float frictionLimit = solveOut.accumulatedDeltaLambdaN * frictionCoefficient;
    const float targetAccumLambdaT = dpSelectMax(appliedTanRef + requiredLambdaTNeg, -frictionLimit);
    solveOut.deltaLambdaT = targetAccumLambdaT - appliedTanRef;

    return solveOut.deltaLambdaN != 0.0f || solveOut.deltaLambdaT != 0.0f;
}
