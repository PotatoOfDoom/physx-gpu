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

// softBodyFemAttachBda.slang -- the deformable-vs-deformable ATTACHMENT kernels of
// softBodyGM.cu, PhysX 5.12:
//
//   femAttachmentPrepareLaunch                       FEM_ATTACHMENT_CONSTRAINT_PREP, launched
//       by the softbody core (softbody-softbody and softbody-cloth attachments) and the cloth
//       core (cloth-cloth): one PxgDbDbAttachmentBlock lane per active
//       PxgDeformableAttachment -- both deformable refs, and each side's barycentric unless
//       that side is a VERTEX (whose slot is left alone, not zeroed).
//   sb_querySoftBodyAttachmentReferenceCountLaunch   NEW in 5.12: bumps mSimDelta.w on both
//       softbodies for every contributing vertex (a tet's corners with |bc| > 1e-3, or the
//       one vertex a vertex side names).  Self attachment is fine (atomics).
//   sb_solveOutputSoftBodyAttachmentDeltaVLaunch     readSoftBodyAttachmentDbDb on both sides
//       (positions, or velocity * dt in velocity iterations; refCount-inflated masses),
//       solveDbDbAttachment, -delta to side 1 then +delta to side 0.
//
// 5.12 removed the cone limits and the 5.6.1 PxgFEMFEMAttachmentConstraint block (its
// layout module pxgSlangFemFemAttach was deleted when the cloth stage ported the last
// consumer, cloth_solveOutputAttachmentClothClothDeltaVLaunch).
//
// In PXG_NOCONTRACT_SHADERS: the barycentric blends and the mass sums are mul-feeding-add
// chains compared bit-exactly against a CPU transcription of the 5.12 CUDA.

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangDeformableRef;               // PXG_DATT_*, PXG_DBDBA_*, PxgGetDeformable*
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbAttachment, bumpSoftBodyRefCountBary

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint FA_BLOCK = 256u;
static const uint FA_GRID = 1024u;

// ===========================================================================
// femAttachmentPrepareLaunch
//   (PxgDeformableAttachment* femAttachments, PxU32* activeFemAttachments,
//    PxU32 numFemAttachments, PxgDbDbAttachmentBlock* attachmentBlocks)
//   -> {8, 8, 4, 8} = 32-byte blob.
// ===========================================================================
struct FemAttachPrepParams
{
    uint64_t femAttachments;        // @0
    uint64_t activeFemAttachments;  // @8
    uint numFemAttachments;         // @16
    uint _pad20;                    // @20
    uint64_t attachmentBlocks;      // @24  output -> 32
}
[[vk::push_constant]]
ConstantBuffer<FemAttachPrepParams> gFaPrep;

[shader("compute")]
[numthreads(256, 1, 1)]   // == FA_BLOCK
void femAttachmentPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numFemAttachments = gFaPrep.numFemAttachments;
    const uint nbBlocksRequired = (numFemAttachments + FA_BLOCK - 1u) / FA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + FA_GRID - 1u) / FA_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * FA_BLOCK + idx + nbIterationsPerBlock * groupId.x * FA_BLOCK;
        if (workIndex >= numFemAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint attachmentIndex = Ptr<uint>(gFaPrep.activeFemAttachments)[workIndex];

        const uint64_t attachment = gFaPrep.femAttachments + uint64_t(attachmentIndex) * PXG_DATT_STRIDE;
        const uint64_t block = gFaPrep.attachmentBlocks + uint64_t(index) * PXG_DBDBA_STRIDE;

        const uint64_t deformableRef0 = Ptr<uint64_t>(attachment + PXG_DATT_pair + PXG_PAIRINFO_ref0)[0];
        const uint64_t deformableRef1 = Ptr<uint64_t>(attachment + PXG_DATT_pair + PXG_PAIRINFO_ref1)[0];

        // A vertex side has no barycentric, so its slot is left alone rather than filled with zeros.
        if (PxgGetDeformableFeatureType(deformableRef0) != PXG_DREF_eVERTEX)
            Ptr<float4>(block + PXG_DBDBA_barycentric0)[offset] = Ptr<float4>(attachment + PXG_DATT_barycentricOrPosition0)[0];
        if (PxgGetDeformableFeatureType(deformableRef1) != PXG_DREF_eVERTEX)
            Ptr<float4>(block + PXG_DBDBA_barycentric1)[offset] = Ptr<float4>(attachment + PXG_DATT_barycentric1)[0];

        Ptr<uint64_t>(block + PXG_DBDBA_deformableRef0)[offset] = deformableRef0;
        Ptr<uint64_t>(block + PXG_DBDBA_deformableRef1)[offset] = deformableRef1;
    }
}

// ===========================================================================
// sb_querySoftBodyAttachmentReferenceCountLaunch (NEW in 5.12)
//   (PxgSoftBody* softbodies, PxgDbDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments) -> {8, 8, 4} = 24-byte blob.
// ===========================================================================
struct SbAttachQueryParams
{
    uint64_t softbodies;        // @0
    uint64_t attachmentBlocks;  // @8
    uint numAttachments;        // @16
    uint _pad20;                // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<SbAttachQueryParams> gSaQuery;

[shader("compute")]
[numthreads(256, 1, 1)]   // == FA_BLOCK
void sb_querySoftBodyAttachmentReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gSaQuery.numAttachments;
    const uint nbBlocksRequired = (numAttachments + FA_BLOCK - 1u) / FA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + FA_GRID - 1u) / FA_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * FA_BLOCK + idx + nbIterationsPerBlock * groupId.x * FA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint64_t block = gSaQuery.attachmentBlocks + uint64_t(index) * PXG_DBDBA_STRIDE;

        // Side 0
        {
            const uint64_t softBodyRef0 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef0)[offset];
            Ptr<PxgSoftBodyGpu> softbody0 = sbAt(gSaQuery.softbodies, PxgGetDeformableId(softBodyRef0));
            bumpSoftBodyRefCountBary(softbody0, softBodyRef0, block + PXG_DBDBA_barycentric0, offset);
        }

        // Side 1
        {
            const uint64_t softBodyRef1 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef1)[offset];
            Ptr<PxgSoftBodyGpu> softbody1 = sbAt(gSaQuery.softbodies, PxgGetDeformableId(softBodyRef1));
            bumpSoftBodyRefCountBary(softbody1, softBodyRef1, block + PXG_DBDBA_barycentric1, offset);
        }
    }
}

// ===========================================================================
// sb_solveOutputSoftBodyAttachmentDeltaVLaunch
//   (PxgSoftBody* softbodies, PxgDbDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments, const PxReal dt, const bool isVelocityIteration)
//   -> {8, 8, 4, 4, 1} = 32-byte blob (dt @20, isVelocityIteration byte @24).
// ===========================================================================
struct SbAttachSolveParams
{
    uint64_t softbodies;        // @0
    uint64_t attachmentBlocks;  // @8
    uint numAttachments;        // @16
    float dt;                   // @20
    uint isVelocityIteration;   // @24  bool byte0
    uint _pad28;                // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<SbAttachSolveParams> gSaSolve;

[shader("compute")]
[numthreads(256, 1, 1)]   // == FA_BLOCK
void sb_solveOutputSoftBodyAttachmentDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gSaSolve.numAttachments;
    const uint nbBlocksRequired = (numAttachments + FA_BLOCK - 1u) / FA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + FA_GRID - 1u) / FA_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gSaSolve.dt;
    const bool isVelocityIteration = (gSaSolve.isVelocityIteration & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * FA_BLOCK + idx + nbIterationsPerBlock * groupId.x * FA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint64_t block = gSaSolve.attachmentBlocks + uint64_t(index) * PXG_DBDBA_STRIDE;

        const uint64_t softBodyRef0 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef0)[offset];
        Ptr<PxgSoftBodyGpu> softbody0 = sbAt(gSaSolve.softbodies, PxgGetDeformableId(softBodyRef0));

        PxgDeformablePart db0 = PxgDeformablePart();
        const float3 pos0 = db0.readSoftBodyAttachmentDbDb(softbody0, softBodyRef0, block + PXG_DBDBA_barycentric0,
                                                            offset, dt, isVelocityIteration);

        const uint64_t softBodyRef1 = Ptr<uint64_t>(block + PXG_DBDBA_deformableRef1)[offset];
        Ptr<PxgSoftBodyGpu> softbody1 = sbAt(gSaSolve.softbodies, PxgGetDeformableId(softBodyRef1));

        PxgDeformablePart db1 = PxgDeformablePart();
        const float3 pos1 = db1.readSoftBodyAttachmentDbDb(softbody1, softBodyRef1, block + PXG_DBDBA_barycentric1,
                                                            offset, dt, isVelocityIteration);

        float3 delta;
        if (solveDbDbAttachment(db0, db1, pos0, pos1, delta))
        {
            db1.writeSoftBody(softbody1, float3(-delta.x, -delta.y, -delta.z));
            db0.writeSoftBody(softbody0, delta);
        }
    }
}
