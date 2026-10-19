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

// softBodyClothAttachBda.slang -- the softbody-vs-cloth ATTACHMENT kernels of softBodyGM.cu,
// PhysX 5.12.  Side 0 is the softbody (a tet's four vertices or one vertex), side 1 the
// cloth (a triangle's three vertices or one vertex); the blocks are written by
// femAttachmentPrepareLaunch (softBodyFemAttachBda).
//
//   sb_queryClothAttachmentReferenceCountLaunch  NEW in 5.12: bumps softbody.mSimDelta.w and
//       cloth.mDeltaPos.w for every contributing vertex.
//   sb_solveOutputClothAttachmentDeltaVLaunch    readSoftBodyAttachmentDbDb /
//       readClothAttachmentDbDb (positions, or velocity * dt in velocity iterations;
//       refCount-inflated masses), solveDbDbAttachment, -delta to the cloth, then +delta to
//       the softbody.
//
// 5.12 removed the cone limits and the 5.6.1 PxgFEMFEMAttachmentConstraint block.
// In PXG_NOCONTRACT_SHADERS.

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangDeformableRef;               // PXG_DBDBA_*, PxgGetDeformable*
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbAttachment, bump*RefCountBary

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint CA_BLOCK = 256u;
static const uint CA_GRID = 1024u;

// ===========================================================================
// sb_queryClothAttachmentReferenceCountLaunch (NEW in 5.12)
//   (PxgSoftBody* softbodies, PxgFEMCloth* clothes, PxgDbDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments) -> {8, 8, 8, 4} = 32-byte blob.
// ===========================================================================
struct ScAttachQueryParams
{
    uint64_t softbodies;        // @0
    uint64_t clothes;           // @8
    uint64_t attachmentBlocks;  // @16
    uint numAttachments;        // @24
    uint _pad28;                // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<ScAttachQueryParams> gCaQuery;

[shader("compute")]
[numthreads(256, 1, 1)]   // == CA_BLOCK
void sb_queryClothAttachmentReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gCaQuery.numAttachments;
    const uint nbBlocksRequired = (numAttachments + CA_BLOCK - 1u) / CA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + CA_GRID - 1u) / CA_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * CA_BLOCK + idx + nbIterationsPerBlock * groupId.x * CA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint64_t block = gCaQuery.attachmentBlocks + uint64_t(index) * PXG_DBDBA_STRIDE;

        // Side 0: softbody.
        {
            const uint64_t softBodyRef = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef0)[offset];
            Ptr<PxgSoftBodyGpu> softbody = sbAt(gCaQuery.softbodies, PxgGetDeformableId(softBodyRef));
            bumpSoftBodyRefCountBary(softbody, softBodyRef, block + PXG_DBDBA_barycentric0, offset);
        }

        // Side 1: cloth.
        {
            const uint64_t clothRef = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef1)[offset];
            Ptr<PxgFEMClothGpu> cloth = clothAt(gCaQuery.clothes, PxgGetDeformableId(clothRef));
            bumpClothRefCountBary(cloth, clothRef, block + PXG_DBDBA_barycentric1, offset);
        }
    }
}

// ===========================================================================
// sb_solveOutputClothAttachmentDeltaVLaunch
//   (PxgSoftBody* softbodies, PxgFEMCloth* clothes, PxgDbDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments, const PxReal dt, const bool isVelocityIteration)
//   -> {8, 8, 8, 4, 4, 1} = 40-byte blob (dt @28, isVelocityIteration byte @32).
// ===========================================================================
struct ScAttachSolveParams
{
    uint64_t softbodies;        // @0
    uint64_t clothes;           // @8
    uint64_t attachmentBlocks;  // @16
    uint numAttachments;        // @24
    float dt;                   // @28
    uint isVelocityIteration;   // @32  bool byte0
    uint _pad36;                // @36 -> 40
}
[[vk::push_constant]]
ConstantBuffer<ScAttachSolveParams> gCaSolve;

[shader("compute")]
[numthreads(256, 1, 1)]   // == CA_BLOCK
void sb_solveOutputClothAttachmentDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gCaSolve.numAttachments;
    const uint nbBlocksRequired = (numAttachments + CA_BLOCK - 1u) / CA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + CA_GRID - 1u) / CA_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gCaSolve.dt;
    const bool isVelocityIteration = (gCaSolve.isVelocityIteration & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * CA_BLOCK + idx + nbIterationsPerBlock * groupId.x * CA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint64_t block = gCaSolve.attachmentBlocks + uint64_t(index) * PXG_DBDBA_STRIDE;

        const uint64_t softBodyRef = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef0)[offset];
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gCaSolve.softbodies, PxgGetDeformableId(softBodyRef));

        PxgDeformablePart db0 = PxgDeformablePart();
        const float3 pos0 = db0.readSoftBodyAttachmentDbDb(softbody, softBodyRef, block + PXG_DBDBA_barycentric0,
                                                            offset, dt, isVelocityIteration);

        const uint64_t clothRef = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef1)[offset];
        Ptr<PxgFEMClothGpu> cloth = clothAt(gCaSolve.clothes, PxgGetDeformableId(clothRef));

        PxgDeformablePart db1 = PxgDeformablePart();
        const float3 pos1 = db1.readClothAttachmentDbDb(cloth, clothRef, block + PXG_DBDBA_barycentric1, offset, dt,
                                                         isVelocityIteration);

        float3 delta;
        if (solveDbDbAttachment(db0, db1, pos0, pos1, delta))
        {
            db1.writeCloth(cloth, float3(-delta.x, -delta.y, -delta.z));
            db0.writeSoftBody(softbody, delta);
        }
    }
}
