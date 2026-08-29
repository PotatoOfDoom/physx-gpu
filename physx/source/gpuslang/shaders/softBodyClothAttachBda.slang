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
//
// ===========================================================================
// softBodyClothAttachBda — sb_solveOutputClothAttachmentDeltaVLaunch
// (softBodyGM.cu:2424), on the engine BDA ABI.
//
// Launch: PxgSoftBodyCore.cpp:2005, guarded `if(nbClothAttachments)`, FOUR arguments
// {softbodies(8), clothes(8), attachments(8), numAttachments(4)}, grid SB_UPDATEROTATION
// = 1024, block SB_UPDATEROTATION = 256.  This is the third and last consumer of
// PxgFEMFEMAttachmentConstraint, and the one that pins a TETRAHEDRON of a softbody to a
// TRIANGLE of a cloth.  The other two landed at P8.19 (softbody-softbody) and P9.32
// (cloth-cloth); the block's byte layout and the helpers all three share live in
// `pxgSlangFemFemAttach`.
//
// WHAT IS THE SAME AS THE SOFTBODY-SOFTBODY SIBLING, and must not be "corrected" because
// the body types changed: elemId1 is still the side that receives -delta, elemId0 still
// receives +delta, the relative position is still `pos1 - pos0`, the scale is still
// `error * (1.0f/wTot)`, and both scatters still pass a constraint weight of 1.
//
// ---------------------------------------------------------------------------
// WHAT ACTUALLY DIFFERS
// ---------------------------------------------------------------------------
// D1  TWO DESCRIPTOR ARRAYS, NOT ONE.  Side 1 walks PxgSoftBody (mSimPosition_InvMass@96,
//     mSimTetIndices@80, output mSimDelta@272); side 0 walks PxgFEMCloth
//     (mPosition_InvMass@16, mTriangleVertexIndices@72, output mDeltaPos@248).  The two
//     have different strides, so indexing the wrong array with the right index is not a
//     fault, it is a plausible wrong answer.  `mDeltaPos` in particular sits one pointer
//     slot before `mAccumulatedDeltaPos`, which is a float4 array of the same length.
//
// D2  THE CLOTH SIDE IS THREE CORNERS EVERYWHERE.  `pos0` blends cp0..cp2 only, so its
//     accumulation is ((a+b)+c) against the softbody's (((a+b)+c)+d); `b0w.w` is a written
//     LITERAL 0.0f (softBodyGM.cu:2500), not a product; and `wTot0` sums THREE terms
//     (:2503) against `wTot1`'s four.  A 4-corner transcription cannot even be written --
//     there is no cp3 to multiply, because triInd is a triangle.
//
// D3  THE CLOTH SCATTER IS updateTriPositionDelta, NOT the tet one.  Three corners, and
//     each is skipped ENTIRELY -- xyz atomic and `.w += 1` alike -- when
//     `b0w.k = bary0.k * cp_k.w` is not strictly positive.  Worth stating because the
//     nearest precedent in this corpus does the opposite: the cloth-cloth solve
//     (femClothAttachBda.slang, P9.32) writes all three corners unconditionally, counter
//     included.  Two cloth scatters, two different rules, thirty lines apart in the same
//     engine.
//
//     THE `.w` COUNTER IS THE DIVISOR, and the two sides do not treat it alike:
//         cloth     `if(delta.w != 0.f) { scale = 1.0f / delta.w; ... }`
//                   (FEMClothExternalSolve.cu:1086, :1093)
//         softbody  `scale = 1.0f / PxMax(delta.w, 1.f)`  (softBodyGM.cu:2644)
//     The softbody applier CLAMPS, so a miscounted `.w` below 1 is absorbed there.  The cloth
//     applier does not: one extra `+1.0f` quietly divides a correct numerator by a too-large
//     count, and a missing one leaves `delta.w == 0`, which makes the applier skip the vertex
//     and zero the buffer.  Neither produces a NaN or an assertion.  The same mistake is
//     therefore invisible on the tetrahedron side and destructive on the triangle side, which
//     is why the gate is transcribed corner by corner rather than assumed harmless.
//
//     The multiply association differs from the cloth-cloth port too, and both are right:
//     here the CUDA precomputes `b0w = bary*invMass` and passes it, so the product is
//     `delta * (bary*invMass)` with the inner product rounded first; the cloth-cloth kernel
//     writes `deltapos * barycentric.x * p00.w` inline, which is `(delta*bary)*invMass`.
//
// D4  THE CONE AXIS IS BUILT FROM THE OTHER BODY AND SUBTRACTED THE OTHER WAY.
//         sibling  (softBodyGM.cu:2394)  projectP over body 0's corners, axis = projectP - pos0
//         here     (softBodyGM.cu:2489)  projectP over the SOFTBODY's corners, axis = pos1 - projectP
//     Both terms come from one body in each kernel, but the order is flipped, so the
//     resulting worldAxis is the negation of what the sibling's shape would give.  This is
//     the one difference here that changes numbers without changing any type, so it is the
//     one a sibling-derived port is most likely to carry over intact.
//
// D5  THE CLOTH ID SPLIT IS A DIFFERENT CONSTANT FAMILY.  `PxGetClothId` /
//     `PxGetClothElementIndex` use PXG_BITSHIFT_ELEMENT_ID with
//     PX_MAX_NB_DEFORMABLE_SURFACE_TRI; the softbody split uses PXG_BITSHIFT_TET_ID with
//     PX_MAX_NB_DEFORMABLE_VOLUME_TET.  Checked: 20 and 0x000fffff on both sides today, so
//     the two agree numerically -- but they are separate constants that can move apart, and
//     this shader keeps them separate for that reason.
//
// NOT A TRAP, recorded so the next reader does not spend time on it: both elemId slots are
// PxU64 read into PxU32, exactly as in the sibling, and the accessors take PxU32 by value,
// so the narrowing is semantically a no-op.  It is transcribed because it is what upstream
// wrote, not because anything hinges on it.
//
// THE CONE BRANCH RUNS ON EVERY ATTACHMENT.  `isConeLimitedEnabled` is an OR of three
// `>= 0` tests, and an attachment created without a PxConeLimitedConstraint carries
// `low_high_angle = (0, 0, -1, 0)` (PxgSimulationController.cpp:3859), so minDist = 0
// satisfies it.  With maxAngle < 0 the angle arm is skipped and `len` clamps into [0,0],
// leaving `error = relPos` -- but relPos is the caller's `relPos1 - worldAxis*constraintOffset`,
// so the offset and the normalise are both on the hot path.  What the default switches off is
// only the transcendental.  See P9.33b: this shader's sibling claimed the branch was dormant
// for five tranches, and it never was.
//
// PxSinCos (PxMath.h:205) is `__sincosf` on CUDA for __CUDA_ARCH__ >= 350 -- the fast-math
// intrinsic, not sinf/cosf.  It is reached only when maxAngle > 0, i.e. only for a real cone,
// and it is the one place on this path where a tolerance rather than a bit is the most that
// can be claimed against the CUDA engine.
// ===========================================================================
module softBodyClothAttachBda;

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimTetIndices@80, mSimPosition_InvMass@96, mSimDelta@272)
import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mPosition_InvMass@16, mTriangleVertexIndices@72, mDeltaPos@248)
import pxgSlangFemFemAttach;   // FAC_* layout + facSlot + the cone helpers + both scatters

// PxgSoftBodyKernelGridDim/BlockDim::SB_UPDATEROTATION.
static const uint SBCA_GRID  = 1024u;
static const uint SBCA_BLOCK = 256u;

// D5: two constant families that happen to agree.
// PxgSimulationCoreDesc.h:40/41, PxDeformableVolume.h:57, PxDeformableSurface.h:56.
static const uint PXG_BITSHIFT_TET_ID     = 20u;
static const uint PXG_VOLUME_TET_MASK     = 0x000fffffu;
static const uint PXG_BITSHIFT_ELEMENT_ID = 20u;
static const uint PXG_SURFACE_TRI_MASK    = 0x000fffffu;

// sb_solveOutputClothAttachmentDeltaVLaunch(PxgSoftBody* softbodies, PxgFEMCloth* clothes,
//   PxgFEMFEMAttachmentConstraint* attachments, const PxU32 numAttachments)
//   -> 8 @0, 8 @8, 8 @16, 4 @24 -> 32-byte blob.  ABI row {8,8,8,4}.
//   The sibling's 24-byte three-argument blob CANNOT be reused: `clothes` occupies the slot
//   its `attachments` lives in, so the mistake produces a plausible pointer, not a fault.
struct ClothSbAttachSolveParams
{
    uint64_t softbodies;      // @0
    uint64_t clothes;         // @8
    uint64_t attachments;     // @16
    uint numAttachments;      // @24
    uint _pad28;              // @28 -> 32
}

[[vk::push_constant]] ConstantBuffer<ClothSbAttachSolveParams> gCs;

[shader("compute")]
[numthreads(256, 1, 1)]   // SB_UPDATEROTATION
void sb_solveOutputClothAttachmentDeltaVLaunch(uint3 gtid : SV_GroupThreadID,
                                               uint3 gid : SV_GroupID)
{
    const uint numAttachments = gCs.numAttachments;
    const uint nbBlocksRequired = (numAttachments + SBCA_BLOCK - 1u) / SBCA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SBCA_GRID - 1u) / SBCA_GRID;
    const uint idx = gtid.x;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gCs.softbodies);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCs.clothes);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SBCA_BLOCK + idx + nbIterationsPerBlock * gid.x * SBCA_BLOCK;
        if (workIndex >= numAttachments)
            break;   // CUDA `return`; nothing follows the loop here either

        // ---- side 1: the softbody's simulation TETRAHEDRON -------------------------------
        const uint elemId1 = Ptr<uint>(facSlot(gCs.attachments, workIndex, FAC_ELEMID1, 8UL))[0];
        const uint softBodyId = elemId1 >> PXG_BITSHIFT_TET_ID;          // PxGetSoftBodyId
        const uint tetrahedronIdx = elemId1 & PXG_VOLUME_TET_MASK;       // PxGetSoftBodyElementIndex

        Ptr<float4> simPos = Ptr<float4>(softbodies[softBodyId].mSimPosition_InvMass);
        const uint4 tetInd = Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[tetrahedronIdx];
        const float4 bary1 = Ptr<float4>(facSlot(gCs.attachments, workIndex, FAC_BARYCENTRIC1, 16UL))[0];

        const float4 sp0 = simPos[tetInd.x];
        const float4 sp1 = simPos[tetInd.y];
        const float4 sp2 = simPos[tetInd.z];
        const float4 sp3 = simPos[tetInd.w];
        const float4 pos1 = sp0 * bary1.x + sp1 * bary1.y + sp2 * bary1.z + sp3 * bary1.w;

        // ---- side 0: the cloth's TRIANGLE.  D1, D2: three corners, other descriptor ------
        const uint elemInd0 = Ptr<uint>(facSlot(gCs.attachments, workIndex, FAC_ELEMID0, 8UL))[0];
        const uint clothId = elemInd0 >> PXG_BITSHIFT_ELEMENT_ID;        // PxGetClothId
        const uint elementId = elemInd0 & PXG_SURFACE_TRI_MASK;          // PxGetClothElementIndex
        const float4 bary0 = Ptr<float4>(facSlot(gCs.attachments, workIndex, FAC_BARYCENTRIC0, 16UL))[0];

        Ptr<float4> clothPos = Ptr<float4>(clothes[clothId].mPosition_InvMass);
        const uint4 triInd = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[elementId];

        const float4 cp0 = clothPos[triInd.x];
        const float4 cp1 = clothPos[triInd.y];
        const float4 cp2 = clothPos[triInd.z];
        const float4 pos0 = cp0 * bary0.x + cp1 * bary0.y + cp2 * bary0.z;

        const uint64_t accumDeltaV1 = softbodies[softBodyId].mSimDelta;
        const uint64_t accumDeltaV0 = clothes[clothId].mDeltaPos;        // D1: NOT mAccumulatedDeltaPos

        const float4 relPos1 = pos1 - pos0;

        const float4 lowHighAngle =
            Ptr<float4>(facSlot(gCs.attachments, workIndex, FAC_LOW_HIGH_ANGLES, 16UL))[0];

        float3 error;
        if (isConeLimitedEnabled(lowHighAngle.z, lowHighAngle.x, lowHighAngle.y))
        {
            const float4 attachmentBary =
                Ptr<float4>(facSlot(gCs.attachments, workIndex, FAC_ATT_BARYCENTRIC, 16UL))[0];
            const float constraintOffset =
                Ptr<float>(facSlot(gCs.attachments, workIndex, FAC_CONSTR_OFFSET, 4UL))[0];
            // D4: the SOFTBODY's corners, and `pos1 - projectP` — the sibling has
            // `projectP - pos0` over body 0's corners, which is the negation of this.
            const float4 projectP = sp0 * attachmentBary.x + sp1 * attachmentBary.y +
                                    sp2 * attachmentBary.z + sp3 * attachmentBary.w;
            const float4 axis = pos1 - projectP;
            const float3 worldAxis = facGetNormalized(axis.xyz);

            error = computeConeLimitedError(lowHighAngle.z, lowHighAngle.x, lowHighAngle.y,
                                            worldAxis, relPos1.xyz - worldAxis * constraintOffset);
        }
        else
        {
            error = relPos1.xyz;
        }

        // D2: b0w.w is a written literal 0.0f and wTot0 has THREE terms.
        const float4 b0w = float4(bary0.x * cp0.w, bary0.y * cp1.w, bary0.z * cp2.w, 0.0f);
        const float4 b1w = float4(bary1.x * sp0.w, bary1.y * sp1.w, bary1.z * sp2.w, bary1.w * sp3.w);

        const float wTot0 = b0w.x * bary0.x + b0w.y * bary0.y + b0w.z * bary0.z;
        const float wTot1 = b1w.x * bary1.x + b1w.y * bary1.y + b1w.z * bary1.z + b1w.w * bary1.w;
        const float wTot = wTot0 + wTot1;

        if (wTot > 0.0f)
        {
            const float3 delta = error * (1.0f / wTot);
            updateTetPositionDelta(accumDeltaV1, tetInd, -delta, b1w, 1.0f);
            updateTriPositionDelta(accumDeltaV0, triInd, delta, b0w, 1.0f);   // D3
        }
    }
}
