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

// pxgSlangVelocityReader -- PxgVelocityReader, PhysX 5.12
// (gpunarrowphase/src/CUDA/dataReadWriteHelper.cuh).
//
// 5.12 dropped the PxgVelocityPack / PGS / TGS unions and the PxgSolverSharedDesc<T>*
// constructor argument.  The reader now takes the raw solver body velocity pool the
// host passes as a kernel argument (PxgSolverCore::getSolverBodyVelPoolDevPtr()) and
// offsets it by solverCoreDesc->accumulatedBodyDeltaVOffset for BOTH PGS and TGS;
// readVelocitiesPGS / readVelocitiesTGS return through out-parameters, and PGS
// zeroes the two deltas so the solver math needs no isTGS branch.
//
// ONE DEVIATION, value-neutral (the precedent of pxgSlangFemRigidContact and
// psContactSolveTgsBda): the CUDA constructor reads artiCoreDesc->
// {mMaxLinksPerArticulation, nbArticulations} for every reader; here they are read
// inside the articulation arm only, so a rigid / static participant never
// dereferences artiCoreDesc (fixtures without links pass 0 for it).
module pxgSlangVelocityReader;

import pxgSlangPrepDesc;            // PxgPrePrepDescGpu
import pxgSlangSolverDesc;          // PxgSolverCoreDescGpu
import pxgSlangArticulationDesc;    // artiU32Field, PXG_ARTI_CORE_*
import pxgSlangDeformableRef;       // PxgNode* accessors

public struct PxgVelocityReaderGpu
{
    public uint64_t solverBodyIndices;  // const PxU32*
    public uint64_t artiLinkVelocities; // const float4* (solverCoreDesc->outArtiVelocity)
    public uint64_t artiCoreDesc;       // read lazily, see the header
    public uint numSolverBodies;
    public uint64_t solverBodyDeltaVel; // solverBodyVelPool + accumulatedBodyDeltaVOffset
    public uint64_t initialVel;         // solverCoreDesc->outSolverVelocity
}

public PxgVelocityReaderGpu PxgMakeVelocityReader(uint64_t prePrepDesc, uint64_t solverCoreDesc,
                                                  uint64_t artiCoreDesc, uint64_t solverBodyVelPool,
                                                  uint numSolverBodies)
{
    Ptr<PxgSolverCoreDescGpu> scd = Ptr<PxgSolverCoreDescGpu>(solverCoreDesc);
    PxgVelocityReaderGpu r;
    r.solverBodyIndices = Ptr<PxgPrePrepDescGpu>(prePrepDesc)[0].solverBodyIndices;
    r.artiLinkVelocities = scd[0].outArtiVelocity;
    r.artiCoreDesc = artiCoreDesc;
    r.numSolverBodies = numSolverBodies;
    r.solverBodyDeltaVel = solverBodyVelPool + uint64_t(scd[0].accumulatedBodyDeltaVOffset) * uint64_t(16);
    r.initialVel = scd[0].outSolverVelocity;
    return r;
}

uint vrSolverBodyIdx(PxgVelocityReaderGpu r, uint64_t rigidId)
{
    return PxgNodeIsStaticBody(rigidId) ? 0u : Ptr<uint>(r.solverBodyIndices)[PxgNodeIndexOf(rigidId)];
}

public void readVelocitiesPGS(PxgVelocityReaderGpu r, uint64_t rigidId, out float3 linVel, out float3 angVel,
                              out float3 linDelta, out float3 angDelta)
{
    const uint solverBodyIdx = vrSolverBodyIdx(r, rigidId);

    if (PxgNodeIsArticulation(rigidId))
    {
        const uint maxLinks = artiU32Field(r.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
        const uint nbArticulations = artiU32Field(r.artiCoreDesc, PXG_ARTI_CORE_NB_ARTICULATIONS);
        const uint artiOffset = maxLinks * nbArticulations;
        const uint index = solverBodyIdx * maxLinks;
        const uint linkID = PxgNodeArticulationLinkId(rigidId);
        Ptr<float4> vels = Ptr<float4>(r.artiLinkVelocities);

        linVel = vels[index + linkID].xyz;
        angVel = vels[index + linkID + artiOffset].xyz;
    }
    else
    {
        Ptr<float4> iv = Ptr<float4>(r.initialVel);
        Ptr<float4> dv = Ptr<float4>(r.solverBodyDeltaVel);
        const float4 iv0 = iv[solverBodyIdx];
        const float4 dv0 = dv[solverBodyIdx];
        const float4 iv1 = iv[solverBodyIdx + r.numSolverBodies];
        const float4 dv1 = dv[solverBodyIdx + r.numSolverBodies];
        linVel = float3(iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z);
        angVel = float3(iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z);
    }

    linDelta = float3(0.0f, 0.0f, 0.0f);
    angDelta = float3(0.0f, 0.0f, 0.0f);
}

public void readVelocitiesTGS(PxgVelocityReaderGpu r, uint64_t rigidId, out float3 linVel, out float3 angVel,
                              out float3 linDelta, out float3 angDelta)
{
    const uint solverBodyIdx = vrSolverBodyIdx(r, rigidId);

    if (PxgNodeIsArticulation(rigidId))
    {
        const uint maxLinks = artiU32Field(r.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
        const uint nbArticulations = artiU32Field(r.artiCoreDesc, PXG_ARTI_CORE_NB_ARTICULATIONS);
        const uint artiOffset = maxLinks * nbArticulations;
        const uint index = solverBodyIdx * maxLinks;
        const uint linkID = PxgNodeArticulationLinkId(rigidId);
        Ptr<float4> vels = Ptr<float4>(r.artiLinkVelocities);

        linVel = vels[index + linkID].xyz;
        angVel = vels[index + linkID + artiOffset].xyz;
        linDelta = vels[index + linkID + artiOffset + artiOffset].xyz;
        angDelta = vels[index + linkID + artiOffset + artiOffset + artiOffset].xyz;
    }
    else
    {
        Ptr<float4> dv = Ptr<float4>(r.solverBodyDeltaVel);
        const float4 vel0 = dv[solverBodyIdx];
        const float4 vel1 = dv[solverBodyIdx + r.numSolverBodies];
        const float4 vel2 = dv[solverBodyIdx + r.numSolverBodies + r.numSolverBodies];

        linVel = float3(vel0.x, vel0.y, vel0.z);
        angVel = float3(vel0.w, vel1.x, vel1.y);
        linDelta = float3(vel1.z, vel1.w, vel2.x);
        angDelta = float3(vel2.y, vel2.z, vel2.w);
    }
}

// The isTGS dispatch the 5.12 solve kernels write as `isTGS ? readVelocitiesTGS : readVelocitiesPGS`.
public void readVelocities(PxgVelocityReaderGpu r, bool isTGS, uint64_t rigidId, out float3 linVel,
                           out float3 angVel, out float3 linDelta, out float3 angDelta)
{
    if (isTGS)
        readVelocitiesTGS(r, rigidId, linVel, angVel, linDelta, angDelta);
    else
        readVelocitiesPGS(r, rigidId, linVel, angVel, linDelta, angDelta);
}
