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

// softBodyClothPrepareBda -- sb_clothContactPrepareLaunch (softBody.cu, PhysX 5.12), the
// prepare of the softbody-vs-cloth contact chain.  It clamps the contact count to
// maxContacts (writing it back), zeroes the applied forces, and writes one
// PxgDbDbContactBlock lane per valid pair:
//
//   normal_pen   the narrowphase normal stored DIRECTLY (cloth -> softbody = db0 -> db1; the
//                softbody-softbody prepare negates because vol-vol reports the opposite
//                sense), and pen - restDistance + (clothDelta - softbodyDelta).normal, the
//                penetration re-referenced to the start of the step;
//   maxPenBiasClamp  the pair max of the cloth's and the softbody's clamps.
//
// Side 0 is the cloth, side 1 the softbody tet.  UPSTREAM, kept: the pair type allows a cloth
// triangle on side 0, but the cloth delta is read as mAccumulatedDeltaPos[featureId], i.e.
// as a VERTEX (the CUDA asserts eVERTEX).  Only the vertex producer runs (the triangle face
// path is gated off by supportClothTriangleCollision), so the read is in range in practice;
// a triangle ref would index the vertex array with a triangle index.  DOCUMENTED, NOT FIXED.
//
// barycentrics0 (the cloth side's) is not read, as in the CUDA.  The strided work index is
// clamped to tNumContacts - 1 after the per-lane guard (P9.23, value-neutral).
//
// In PXG_NOCONTRACT_SHADERS (the barycentric blend and the correction dot).

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_DBDBC_*
import pxgSlangDeformableUtils;             // barycentricProjectTet
import pxgSlangDeformableAndParticleUtils;  // dpDot3, dpSelectMax

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint SP_GRID = 1024u;
static const uint SP_BLOCK = 256u;

// ===========================================================================
// (PxgSoftBody* softbodies, PxgFEMCloth* clothes, float4* normalPens, float4* barycentrics0,
//  float4* barycentrics1, PxgDeformablePairInfo* contactInfos, PxU32* numContacts,
//  PxgDbDbContactBlock* contactBlocks, float2* appliedForces, const PxU32 maxContacts)
//  -> {8 x 9, 4} = 80-byte blob.
// ===========================================================================
struct SbScPrepParams
{
    uint64_t softbodies;      // @0
    uint64_t clothes;         // @8
    uint64_t normalPens;      // @16
    uint64_t barycentrics0;   // @24  (unread, as in CUDA)
    uint64_t barycentrics1;   // @32
    uint64_t contactInfos;    // @40
    uint64_t numContacts;     // @48  in/out (clamped)
    uint64_t contactBlocks;   // @56  output
    uint64_t appliedForces;   // @64  output (zeroed)
    uint maxContacts;         // @72
    uint _pad76;              // @76 -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbScPrepParams> gScPrep;

[shader("compute")]
[numthreads(256, 1, 1)]   // == SP_BLOCK
void sb_clothContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = min(gScPrep.maxContacts, Ptr<uint>(gScPrep.numContacts)[0]);

    if (groupThreadId.x == 0u && groupId.x == 0u)
        Ptr<uint>(gScPrep.numContacts)[0] = tNumContacts;

    const uint nbBlocksRequired = (tNumContacts + SP_BLOCK - 1u) / SP_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SP_GRID - 1u) / SP_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gScPrep.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gScPrep.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SP_BLOCK + idx + nbIterationsPerBlock * groupId.x * SP_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23
        const uint threadIndexInWarp = workIndex & 31u;

        appliedForces[workIndex] = float2(0.0f, 0.0f);

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gScPrep.contactBlocks + uint64_t(workIndex / 32u) * PXG_DBDBC_STRIDE;

        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        // Side 0: cloth (a vertex, see the header).
        Ptr<PxgFEMClothGpu> cloth = clothAt(gScPrep.clothes, PxgGetDeformableId(contactInfo.ref0));
        const uint elemInd = PxgGetDeformableFeatureId(contactInfo.ref0);

        // Side 1: softbody tet.
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gScPrep.softbodies, PxgGetDeformableId(contactInfo.ref1));
        const uint tetInd = PxgGetDeformableFeatureId(contactInfo.ref1);

        const float4 softBodyBC = Ptr<float4>(gScPrep.barycentrics1)[workIndex];

        const float4 clothDelta = Ptr<float4>(cloth[0].mAccumulatedDeltaPos)[elemInd];

        const uint4 tetVertInd = Ptr<uint4>(softbody[0].mSimTetIndices)[tetInd];
        const float4 softbodyDelta = barycentricProjectTet(tetVertInd, softbody[0].mSimDeltaPos, softBodyBC);

        const float restDistance = cloth[0].mRestOffset + softbody[0].mRestOffset;
        const float4 normal_pen = Ptr<float4>(gScPrep.normalPens)[workIndex];
        const float3 normal = float3(normal_pen.x, normal_pen.y, normal_pen.z);

        const float3 rel = float3(clothDelta.x - softbodyDelta.x, clothDelta.y - softbodyDelta.y,
                                  clothDelta.z - softbodyDelta.z);
        const float pen = normal_pen.w - restDistance + dpDot3(rel, normal);

        Ptr<float4>(block + PXG_DBDBC_normal_pen)[threadIndexInWarp] = float4(normal.x, normal.y, normal.z, pen);

        // Pair max of side 0 (cloth) and side 1 (softbody) bias clamps.
        Ptr<float>(block + PXG_DBDBC_maxPenBiasClamp)[threadIndexInWarp] =
            dpSelectMax(cloth[0].mPenBiasClamp, softbody[0].mPenBiasClamp);
    }
}
