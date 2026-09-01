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

// ===========================================================================
// softBodyClothPrepareBda — sb_clothContactPrepareLaunch (softBody.cu:1026), on the engine
// BDA ABI.  Step 5 of the softbody-vs-cloth chain: it turns the contacts steps 2 and 4
// produced into solver constraints.
//
// ONE KERNEL, TWO CONTACT SHAPES, AND clothBC.w IS WHAT TELLS THEM APART.  Steps 2 and 4 write
// into the SAME contact stream, and the cloth-side barycentric is the discriminator:
//
//   * clothBC.w == 0  -> the contact came from step 2 and the cloth element is a TRIANGLE.
//     The friction comes from the triangle's material, and the effective mass is the
//     barycentric-weighted sum over its three vertices' inverse masses;
//   * clothBC.w != 0  -> step 4, and the cloth element is a VERTEX.  Step 4 writes the
//     degenerate marker (0, 0, 0, 1) precisely so this test can be made.  The friction comes
//     from the per-vertex mDynamicFrictions array, and the effective mass is that one
//     vertex's inverse mass.
//
// So `triInd` is a triangle index in one case and a vertex index in the other, out of the same
// field.  That is upstream's design and the reason the marker exists.
//
// THE NORMAL IS RE-NEGATED HERE.  Steps 2 and 4 store `-direction`; this kernel negates again,
// so the constraint's normal is the original GJK/closest-point direction.  Two negations that
// must both be present -- dropping either flips the constraint and pushes the bodies together.
//
// The penetration is the raw one less the two rest distances (`thickness`) less the relative
// accumulated motion projected on the normal -- the standard "what is left to solve after the
// substeps so far" correction.
//
// DEAD STORES DELIBERATELY NOT PORTED.  The .cu computes `clothPos` and `softBodyPos` and then
// never reads them (softBody.cu:1128-1135).  They are omitted here, which is behaviour-
// preserving AND safer: their cloth half indexes mTriangleVertexIndices with `triInd`, which in
// the VERTEX case is a vertex index and can run past the triangle array.  Benign in CUDA,
// a fault waiting to happen through a BDA pointer.
//
// The constraint block is the same block-of-32 SoA as the self-collision path
// (PxgSoftBodyCore.h:51, stride 1792), and `workIndex & 31` picks the slot.
//
// GRID: SB_UPDATEROTATION, 1024 blocks of 256 threads, with CUDA's own two-level work split
// reproduced exactly so the workIndex -> block/slot mapping is identical.
// ===========================================================================
module softBodyClothPrepareBda;

import pxgSlangSoftBodyDesc;
import pxgSlangSoftBodySolve;   // P9.39d: barycentricProject, one copy for the corpus
import pxgSlangFEMClothDesc;

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint SP_GRID = 1024u;
static const uint SP_BLOCK = 256u;

// (id << 20) | elementId, for both cloth and softbody.
static const uint PXG_BITSHIFT_ID = 20u;
static const uint PX_MAX_ELEMENT = 0x000fffffu;
uint getId(uint compressed) { return compressed >> PXG_BITSHIFT_ID; }
uint getElementIndex(uint compressed) { return compressed & PX_MAX_ELEMENT; }

// PxgFemFemContactInfo, 16 B: pairInd0 PxU64 @0, pairInd1 PxU32 @8, auxInd @12.
static const uint64_t FFCI_STRIDE = 16UL;

// PxgSoftBodySoftBodyConstraintBlock — block-of-32 SoA, 1792 B.
static const uint64_t SSC_barycentric0  = 0UL;
static const uint64_t SSC_barycentric1  = 512UL;
static const uint64_t SSC_normal_pen    = 1024UL;
static const uint64_t SSC_velMultiplier = 1536UL;
static const uint64_t SSC_friction      = 1664UL;
static const uint64_t SSC_STRIDE        = 1792UL;

// PxsDeformableVolumeMaterialData / PxsDeformableSurfaceMaterialData: both 32 B with
// dynamicFriction at @8 (youngs @0, poissons @4).
static const uint64_t MATERIAL_STRIDE = 32UL;
static const uint64_t MATERIAL_DYNFRICTION = 8UL;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgSoftBodyCore.cpp:2474):
// { softbodies*, clothes*, normalPens*, barycentrics0*, barycentrics1*, contactInfos*,
//   numContacts*, constraints*, lambdaNs*, PxU32 maxContacts, softbodyMaterials*,
//   clothMaterials* }.
// Nine pointers @0..@64, PxU32 @72, then pointers realigned to @80 and @88 -> 96-byte blob.
// ABI row {8,8,8,8,8,8,8,8,8,4,8,8}.
// ---------------------------------------------------------------------------
struct SbClothPrepParams
{
    uint64_t softbodies;        // @0
    uint64_t clothes;           // @8
    uint64_t normalPens;        // @16
    uint64_t barycentrics0;     // @24
    uint64_t barycentrics1;     // @32
    uint64_t contactInfos;      // @40
    uint64_t numContacts;       // @48
    uint64_t constraints;       // @56
    uint64_t lambdaNs;          // @64
    uint maxContacts;           // @72
    uint _pad76;                // @76
    uint64_t softbodyMaterials; // @80
    uint64_t clothMaterials;    // @88 -> 96
}

[[vk::push_constant]] ConstantBuffer<SbClothPrepParams> gSp;

// A PxU16 array read through the containing 32-bit word (cloth.mMaterialIndices).
uint u16At(uint64_t base, uint idx)
{
    const uint64_t addr = base + uint64_t(idx) * 2UL;
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    return ((addr & 2UL) != 0UL) ? (word >> 16) : (word & 0xffffu);
}

float materialDynamicFriction(uint64_t materials, uint index)
{
    return Ptr<float>(materials + uint64_t(index) * MATERIAL_STRIDE + MATERIAL_DYNFRICTION)[0];
}


// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // == SP_BLOCK == PxgSoftBodyKernelBlockDim::SB_UPDATEROTATION
void sb_clothContactPrepareLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint rawCount = Ptr<uint>(gSp.numContacts)[0];
    const uint tNumContacts = min(gSp.maxContacts, rawCount);

    // The count is CLAMPED IN PLACE: everything downstream reads it back, so a kernel that
    // clamped only its own loop would leave the solver walking past the constraint buffer.
    if (gtid.x == 0u && gid.x == 0u)
        Ptr<uint>(gSp.numContacts)[0] = tNumContacts;

    const uint nbBlocksRequired = (tNumContacts + SP_BLOCK - 1u) / SP_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SP_GRID - 1u) / SP_GRID;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSp.softbodies);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gSp.clothes);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SP_BLOCK + gtid.x + nbIterationsPerBlock * gid.x * SP_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        // The accumulated lambda starts at zero for every contact, every frame.
        Ptr<float>(gSp.lambdaNs)[workIndex] = 0.0f;

        const uint64_t ci = gSp.contactInfos + uint64_t(workIndex) * FFCI_STRIDE;
        const uint pairInd0 = Ptr<uint>(ci)[0];          // cloth
        const uint pairInd1 = Ptr<uint>(ci + 8UL)[0];    // softbody

        // ---- the cloth half ---------------------------------------------------
        const uint clothId = getId(pairInd0);
        const uint triInd = getElementIndex(pairInd0);   // triangle OR vertex, see the header

        Ptr<float4> clothAccumulatedDelta = Ptr<float4>(clothes[clothId].mAccumulatedDeltaPos);
        const float4 clothDelta = clothAccumulatedDelta[triInd];

        const float4 clothBC = Ptr<float4>(gSp.barycentrics0)[workIndex];
        float clothFriction = 0.0f;
        float clothDenom = 0.0f;

        if (clothBC.w == 0.0f)
        {
            // A cloth TRIANGLE (step 2).
            const uint globalMaterialIndex = u16At(clothes[clothId].mMaterialIndices, triInd);
            clothFriction = materialDynamicFriction(gSp.clothMaterials, globalMaterialIndex);

            const uint4 triVertId = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[triInd];
            const float w0 = clothAccumulatedDelta[triVertId.x].w;
            const float w1 = clothAccumulatedDelta[triVertId.y].w;
            const float w2 = clothAccumulatedDelta[triVertId.z].w;

            clothDenom = clothBC.x * clothBC.x * w0 + clothBC.y * clothBC.y * w1 +
                         clothBC.z * clothBC.z * w2;
        }
        else
        {
            // A cloth VERTEX (step 4).  The friction is per-vertex, not per-material.
            clothFriction = Ptr<float>(clothes[clothId].mDynamicFrictions)[triInd];
            clothDenom = clothAccumulatedDelta[triInd].w;
        }

        // ---- the softbody half ------------------------------------------------
        const uint softbodyId = getId(pairInd1);
        const uint tetInd = getElementIndex(pairInd1);
        const uint4 tetVertInd = Ptr<uint4>(softbodies[softbodyId].mSimTetIndices)[tetInd];
        const float4 softBodyBC = Ptr<float4>(gSp.barycentrics1)[workIndex];

        // NOTE mSimDeltaPos, the SIM mesh's accumulated delta -- not mSimDelta and not the
        // collision mesh's.  The three are different buffers on the same descriptor.
        Ptr<float4> sbAccumulatedDelta = Ptr<float4>(softbodies[softbodyId].mSimDeltaPos);
        const float4 softbodyDelta = barycentricProject(sbAccumulatedDelta, tetVertInd, softBodyBC);

        const float sw0 = sbAccumulatedDelta[tetVertInd.x].w;
        const float sw1 = sbAccumulatedDelta[tetVertInd.y].w;
        const float sw2 = sbAccumulatedDelta[tetVertInd.z].w;
        const float sw3 = sbAccumulatedDelta[tetVertInd.w].w;

        const float softBodyDenom =
            softBodyBC.x * softBodyBC.x * sw0 + softBodyBC.y * softBodyBC.y * sw1 +
            softBodyBC.z * softBodyBC.z * sw2 + softBodyBC.w * softBodyBC.w * sw3;

        // mMaterialIndices is a PxU16 array on BOTH descriptors -- reading it as PxU32 gives
        // the wrong material for every odd index and walks past the end.
        const uint sbMatIdx = u16At(softbodies[softbodyId].mMaterialIndices, tetInd);
        const float softbodyFriction = materialDynamicFriction(gSp.softbodyMaterials, sbMatIdx);

        // ---- common -----------------------------------------------------------
        const float thickness = clothes[clothId].mRestDistance + softbodies[softbodyId].mRestDistance;
        const float4 normal_pen = Ptr<float4>(gSp.normalPens)[workIndex];
        // The SECOND negation (see the header).
        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);

        const float3 relDelta = float3(clothDelta.x - softbodyDelta.x,
                                       clothDelta.y - softbodyDelta.y,
                                       clothDelta.z - softbodyDelta.z);
        const float pen = normal_pen.w - thickness - dot(relDelta, normal);
        const float denom = clothDenom + softBodyDenom;

        const uint64_t blockBase = gSp.constraints + uint64_t(workIndex / 32u) * SSC_STRIDE;
        const uint slot = workIndex & 31u;

        Ptr<float4>(blockBase + SSC_barycentric0)[slot] = clothBC;
        Ptr<float4>(blockBase + SSC_barycentric1)[slot] = softBodyBC;
        Ptr<float4>(blockBase + SSC_normal_pen)[slot] = float4(normal.x, normal.y, normal.z, pen);
        Ptr<float>(blockBase + SSC_velMultiplier)[slot] = (denom == 0.0f) ? 0.0f : (1.0f / denom);
        Ptr<float>(blockBase + SSC_friction)[slot] = 0.5f * (clothFriction + softbodyFriction);
    }
}
