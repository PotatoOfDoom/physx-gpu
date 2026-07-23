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

// femClothContactSortBda.slang
//
// P7.1 tranche 4b-1 — cloth-vs-rigid contact SORT / REORDER on the Vulkan engine
// ABI (BDA).  The single kernel that turns the P7.1.4a-generated cloth-vs-rigid
// contacts (the unsorted point / normalPen / barycentric / info arrays) into the
// `…Sorted…` buffers the FEM rigid-contact prepare/solve read, by gathering them
// through the by-rigid sort ranks (`remapByRigid`) produced upstream by the plain
// radix sort (RS_MULTIBLOCK / RS_CALCULATERANKS_MULTIBLOCK, already mapped and
// reused from P6.4b-1) + RS_COPY_HIGH_32BITS + RS_COPY_VALUE.
//
//   fem_reorderRigidContactsLaunch (gpunarrowphase softbodyPrimitives.cu:1655,
//     launch PxgFEMCore::reorderRigidContacts() PxgFEMCore.cpp:232,
//     FEM_REORDER_RS_CONTACTS): grid SB_REORDERCONTACTS (1024), block
//     SB_REORDERCONTACTS (256).  For every contact index [0, *numContacts) it
//     reads remapIndex = remapByRigid[index] and copies contacts / normalpens /
//     barycentrics / contactInfos[remapIndex] into the four sorted arrays, then
//     splits the sorted contact info's pairInd0 (PxU64 rigid id) into the
//     separate sortedRigidIds[index] array (the FEM code shares the rigid
//     delta-accumulation path with the particle system, which wants the rigid
//     id as a plain PxU64 array — see PxgFEMCore.cpp:227).
//
// Doctrine (identical to the prior cloth BDA tranches and the particle contact-
// SORT precedent psContactSortBda / ps_reorderPrimitiveContactsLaunch):
//   - every argument arrives as a device pointer read through Ptr<>; the four
//     value arrays are float4 (point / normalPen / barycentric) or the 16-byte
//     PxgFemOtherContactInfo mirror, gathered whole;
//   - the PxU64 rigid id is the low 8 bytes of the contact-info struct (pairInd0),
//     carried as its uint2 halves and stored to the PxU64 sortedRigidIds array.
//
// NO NoContraction stamp: this kernel performs ZERO float arithmetic — it is a
// pure index-remapped LOAD/STORE gather (the float4 lanes are moved as opaque bit
// patterns, never fed to an arithmetic op).  There is nothing for a driver to
// contract, so the output is bit-identical to the CPU gather without the
// decoration (hence femClothContactSortBda is deliberately absent from
// PXG_NOCONTRACT_SHADERS — the same reasoning as ps_reorderPrimitiveContactsLaunch).
//
// NO barrier / groupshared: the early `return` inside the grid-stride loop (a
// per-thread `index >= totalNumContacts` guard) is safe because the kernel has no
// groupshared state and no barrier — exactly as ps_reorderPrimitiveContactsLaunch.

// PxgSoftBodyKernelBlockDim/GridDim::SB_REORDERCONTACTS
// (PxgSoftBodyCoreKernelIndices.h): block 256, grid 1024.
static const uint RC_BLOCK = 256u;   // SB_REORDERCONTACTS block
static const uint RC_GRID = 1024u;   // SB_REORDERCONTACTS grid

// PxgFemOtherContactInfo (PxgFEMCore.h:128, PX_ALIGN(16), 16 bytes):
//   PxU64 pairInd0 @0 (rigid/particle id), PxU32 pairInd1 @8 (fem id),
//   PxU32 rigidMatInd_isInCollision @12.  The reorder copies the whole struct and
//   splits pairInd0 out into the sortedRigidIds array.
struct FemOtherContactInfo
{
    uint2 pairInd0;                    // @0  (PxU64)
    uint pairInd1;                     // @8
    uint rigidMatInd_isInCollision;    // @12 -> 16
}

// ===========================================================================
// fem_reorderRigidContactsLaunch (softbodyPrimitives.cu:1655)
//   CUDA ABI: (const float4* contacts, const float4* normalpens,
//              const float4* barycentrics, const PxgFemOtherContactInfo* contactInfos,
//              const PxU32* numContacts, const PxU32* remapByRigid,
//              float4* sortedContacts, float4* sortedNormalPens,
//              float4* sortedBarycentrics, PxgFemOtherContactInfo* sortedContactInfos,
//              PxU64* sortedRigidIds)
//   -> 11 device pointers = {8,8,8,8,8,8,8,8,8,8,8} = 88-byte blob.
// ===========================================================================
struct FemReorderParams
{
    uint64_t contacts;           // @0   const float4*
    uint64_t normalpens;         // @8   const float4*
    uint64_t barycentrics;       // @16  const float4*
    uint64_t contactInfos;       // @24  const PxgFemOtherContactInfo*
    uint64_t numContacts;        // @32  const PxU32*
    uint64_t remapByRigid;       // @40  const PxU32*
    uint64_t sortedContacts;     // @48  float4* output
    uint64_t sortedNormalPens;   // @56  float4* output
    uint64_t sortedBarycentrics; // @64  float4* output
    uint64_t sortedContactInfos; // @72  PxgFemOtherContactInfo* output
    uint64_t sortedRigidIds;     // @80  PxU64* output -> 88
}

[[vk::push_constant]]
ConstantBuffer<FemReorderParams> gRe;

[shader("compute")]
[numthreads(RC_BLOCK, 1, 1)]
void fem_reorderRigidContactsLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                    uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;

    const uint totalNumContacts = Ptr<uint>(gRe.numContacts)[0];

    // CUDA operator precedence preserved verbatim: ((n + B*G - 1)/B) * G.
    const uint numIternations = (totalNumContacts + RC_BLOCK * RC_GRID - 1u) / RC_BLOCK * RC_GRID;

    Ptr<float4> contacts = Ptr<float4>(gRe.contacts);
    Ptr<float4> normalpens = Ptr<float4>(gRe.normalpens);
    Ptr<float4> barycentrics = Ptr<float4>(gRe.barycentrics);
    Ptr<FemOtherContactInfo> contactInfos = Ptr<FemOtherContactInfo>(gRe.contactInfos);
    Ptr<uint> remapByRigid = Ptr<uint>(gRe.remapByRigid);

    Ptr<float4> sortedContacts = Ptr<float4>(gRe.sortedContacts);
    Ptr<float4> sortedNormalPens = Ptr<float4>(gRe.sortedNormalPens);
    Ptr<float4> sortedBarycentrics = Ptr<float4>(gRe.sortedBarycentrics);
    Ptr<FemOtherContactInfo> sortedContactInfos = Ptr<FemOtherContactInfo>(gRe.sortedContactInfos);
    Ptr<uint2> sortedRigidIds = Ptr<uint2>(gRe.sortedRigidIds);

    for (uint i = 0u; i < numIternations; ++i)
    {
        const uint index = tid + blockId * RC_BLOCK + i * RC_BLOCK * RC_GRID;
        if (index >= totalNumContacts)
            return;

        const uint remapIndex = remapByRigid[index];

        sortedContacts[index] = contacts[remapIndex];
        sortedNormalPens[index] = normalpens[remapIndex];
        sortedBarycentrics[index] = barycentrics[remapIndex];

        const FemOtherContactInfo contactInfo = contactInfos[remapIndex];

        const uint2 rigidId = contactInfo.pairInd0;

        sortedContactInfos[index] = contactInfo;
        sortedRigidIds[index] = rigidId;
    }
}
