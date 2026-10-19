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

// femClothAttachBda.slang -- the cloth-vs-cloth ATTACHMENT kernels of FEMClothExternalSolve.cu,
// PhysX 5.12.  Both sides are cloths (a triangle's three vertices or one vertex); the blocks
// are written by femAttachmentPrepareLaunch (softBodyFemAttachBda).
//
//   cloth_queryClothClothAttachmentReferenceCountLaunch  NEW in 5.12: bumps mDeltaPos.w
//       (integer count, |bc| > 1e-3) for every contributing vertex of both sides.
//   cloth_solveOutputAttachmentClothClothDeltaVLaunch    readClothAttachmentDbDb on both
//       sides (positions, or velocity * dt in velocity iterations; refCount-inflated
//       masses), solveDbDbAttachment, +delta to side 0 and -delta to side 1 (writeCloth).
//
// 5.12 removed the cone limits and the 5.6.1 PxgFEMFEMAttachmentConstraint block (and with
// them the under-correction the 5.9 fix addressed).  In PXG_NOCONTRACT_SHADERS.

import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangDeformableRef;               // PXG_DBDBA_*, PxgGetDeformable*
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbAttachment, bumpClothRefCountBary

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint CA_BLOCK = 256u;
static const uint CA_GRID = 1024u;

// ===========================================================================
// cloth_queryClothClothAttachmentReferenceCountLaunch (NEW in 5.12)
//   (PxgFEMCloth* clothes, PxgDbDbAttachmentBlock* attachmentBlocks, const PxU32 numAttachments)
//   -> {8, 8, 4} = 24-byte blob.
// ===========================================================================
struct CcAttachQueryParams
{
    uint64_t clothes;           // @0
    uint64_t attachmentBlocks;  // @8
    uint numAttachments;        // @16
    uint _pad20;                // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<CcAttachQueryParams> gCcQuery;

[shader("compute")]
[numthreads(256, 1, 1)]   // == CA_BLOCK
void cloth_queryClothClothAttachmentReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gCcQuery.numAttachments;
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

        const uint64_t block = gCcQuery.attachmentBlocks + uint64_t(index) * PXG_DBDBA_STRIDE;

        // Side 0
        {
            const uint64_t clothRef0 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef0)[offset];
            Ptr<PxgFEMClothGpu> cloth0 = clothAt(gCcQuery.clothes, PxgGetDeformableId(clothRef0));
            bumpClothRefCountBary(cloth0, clothRef0, block + PXG_DBDBA_barycentric0, offset);
        }

        // Side 1
        {
            const uint64_t clothRef1 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef1)[offset];
            Ptr<PxgFEMClothGpu> cloth1 = clothAt(gCcQuery.clothes, PxgGetDeformableId(clothRef1));
            bumpClothRefCountBary(cloth1, clothRef1, block + PXG_DBDBA_barycentric1, offset);
        }
    }
}

// ===========================================================================
// cloth_solveOutputAttachmentClothClothDeltaVLaunch
//   (PxgFEMCloth* clothes, PxgDbDbAttachmentBlock* attachmentBlocks, const PxU32 numAttachments,
//    const PxReal dt, const bool isVelocityIteration)
//   -> {8, 8, 4, 4, 1} = 32-byte blob (dt @20, isVelocityIteration byte @24).
// ===========================================================================
struct CcAttachSolveParams
{
    uint64_t clothes;           // @0
    uint64_t attachmentBlocks;  // @8
    uint numAttachments;        // @16
    float dt;                   // @20
    uint isVelocityIteration;   // @24  bool byte0
    uint _pad28;                // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<CcAttachSolveParams> gCcSolve;

[shader("compute")]
[numthreads(256, 1, 1)]   // == CA_BLOCK
void cloth_solveOutputAttachmentClothClothDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gCcSolve.numAttachments;
    const uint nbBlocksRequired = (numAttachments + CA_BLOCK - 1u) / CA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + CA_GRID - 1u) / CA_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gCcSolve.dt;
    const bool isVelocityIteration = (gCcSolve.isVelocityIteration & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * CA_BLOCK + idx + nbIterationsPerBlock * groupId.x * CA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint64_t block = gCcSolve.attachmentBlocks + uint64_t(index) * PXG_DBDBA_STRIDE;

        const uint64_t clothRef0 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef0)[offset];
        Ptr<PxgFEMClothGpu> cloth0 = clothAt(gCcSolve.clothes, PxgGetDeformableId(clothRef0));

        PxgDeformablePart db0 = PxgDeformablePart();
        const float3 pos0 = db0.readClothAttachmentDbDb(cloth0, clothRef0, block + PXG_DBDBA_barycentric0, offset, dt,
                                                         isVelocityIteration);

        const uint64_t clothRef1 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef1)[offset];
        Ptr<PxgFEMClothGpu> cloth1 = clothAt(gCcSolve.clothes, PxgGetDeformableId(clothRef1));

        PxgDeformablePart db1 = PxgDeformablePart();
        const float3 pos1 = db1.readClothAttachmentDbDb(cloth1, clothRef1, block + PXG_DBDBA_barycentric1, offset, dt,
                                                         isVelocityIteration);

        float3 delta;
        if (solveDbDbAttachment(db0, db1, pos0, pos1, delta))
        {
            db0.writeCloth(cloth0, delta);
            db1.writeCloth(cloth1, float3(-delta.x, -delta.y, -delta.z));
        }
    }
}
