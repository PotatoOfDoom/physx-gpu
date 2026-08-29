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
// femClothAttachBda — cloth_solveOutputAttachmentClothClothDeltaVLaunch
// (FEMClothExternalSolve.cu:1493), on the engine BDA ABI.
//
// Launch: PxgFEMClothCore.cpp:1915, guarded `if(nbClothAttachments)`, three arguments
// {clothes(8), attachments(8), numAttachments(4)}, grid SB_UPDATEROTATION = 1024,
// block SB_UPDATEROTATION = 256.  This is the cloth twin of
// sb_solveOutputSoftBodyAttachmentDeltaVLaunch: one attachment pins a point inside a
// TRIANGLE of cloth A to a point inside a TRIANGLE of cloth B, and the solve pushes both
// triangles' vertices toward each other by a mass-split correction.  Nothing else in the
// pipeline writes this constraint block for cloth — the ONE producer is
// `femAttachmentPrepareLaunch` (softBodyGM.cu:2036, ported in softBodyFemAttachBda), which
// is why the layout now lives in the shared `pxgSlangFemFemAttach` module and not twice.
//
// WHAT MAKES THIS **NOT** A COPY OF THE SOFTBODY SOLVE — the four differences that a
// sibling-derived port would silently get wrong are listed in the module header; the traps
// below are the ones specific to reading THIS kernel.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// A1  THE WORK-INDEX FORMULA IS THE STRIDED ONE, and it is stated here for the same
//     reason P9.31 stated it: it is NOT a correctness property.
//         workIndex = i*blockDim.x + threadIdx.x + nbIterationsPerBlock*blockIdx.x*blockDim.x
//     A plain grid stride covers [0, N) exactly once as well, and lands on the same lane
//     (every term but threadIdx.x is a multiple of 256, hence of 32).  It is transcribed
//     because it is what upstream wrote, not because a substitution would be observable.
//     What IS observable is dropping the `nbIterationsPerBlock *` factor: the block ranges
//     then overlap and the tail is never visited — but only past
//     gridDim*blockDim = 1024*256 = 262144 attachments.
//
// A2  THE CONSTRAINT IS WARP-BLOCKED.  `attachments[workIndex/32]` selects a 2688-byte
//     block; `workIndex & 31` selects the slot within each of its 32-element arrays.  A
//     flat `attachments[workIndex]` read compiles, runs, and is wrong for every
//     workIndex >= 32 — the first 32 attachments alias the correct answer exactly, so a
//     fixture with fewer than 33 attachments cannot tell the two apart.
//
// A3  elemId0/elemId1 ARE PxU64 SLOTS READ AS PxU32.  The CUDA declares
//     `PxU64 elemId0[32]` and assigns it into `const PxU32 elemId0` — so the stride is 8
//     bytes and only the LOW word is meaningful.  Reading at an 8-byte stride but taking
//     a uint (not a uint2) is the transcription; reading at a 4-byte stride would halve
//     the addresses and pick up the zero upper words of the even slots.
//
// A4  THE TWO SIDES DECODE THEIR ELEMENT ID DIFFERENTLY IN THE SOURCE TEXT.  cloth0 goes
//     through the accessors `PxGetClothId` / `PxGetClothElementIndex`
//     (PxgSimulationCoreDesc.h:135/140); cloth1 open-codes `elemId1 >> 20` and
//     `elemId1 & 0x000fffff`.  Checked: PXG_BITSHIFT_ELEMENT_ID is 20 and
//     PX_MAX_NB_DEFORMABLE_SURFACE_TRI is 0x000fffff (PxDeformableSurface.h:56), so the
//     two spellings agree TODAY and the port keeps both sides on the same constants.  The
//     open-coded half is a readability defect that breaks silently the moment either
//     constant moves — the same shape as candidate 40, recorded and not reported.
//
// A5  `pos.w` IS THE INTERPOLATED INVERSE MASS, NOT A VERTEX'S.  `pos0` is the full float4
//     blend `p00*b.x + p01*b.y + p02*b.z`, so `pos0.w = sum_i bary[i] * invMass_i`.  That
//     blended value is what divides the impulse.  The SCATTER, in contrast, uses the
//     PER-VERTEX `p0k.w`.  Two different inverse masses, thirteen lines apart.
//
// A6  THE ZERO-MASS GUARD IS `continue`, NOT `return` — and it is the ONLY thing standing
//     between this kernel and a division by zero, because the divisor is
//     `invMass0 + invMass1` with no epsilon.  Porting it as an early `return`/`break`
//     would additionally abandon every later iteration of a thread that hits one fully
//     pinned attachment.  Both halves of that matter, and the fixture pins both.
//
// A7  THE SCATTER HAS NO SKIP.  `updateTrianglePosDelta` elsewhere in this corpus guards
//     each corner with `invMass > 0` and `|bary| > 1e-6`; this kernel calls AtomicAdd
//     directly for all three corners of both triangles, unconditionally.  A zero weight
//     therefore still bumps the `.w` COUNTER by 1 while adding a zero to xyz — and the
//     engine divides by that counter downstream, so the counts are observable and a
//     "helpfully" guarded port changes the result.
//
// A8  THE SIGNS.  cloth0 receives `+deltaImp`, cloth1 receives `-deltaImp`, where
//     `diff = pos1 - pos0`.  So cloth0 moves TOWARD cloth1.  The softbody sibling writes
//     `-delta` to body 1 and `+delta` to body 0 from `relPos1 = pos1 - pos0` as well —
//     the same orientation, reached by different-looking code.
//
// A9  THE COEFFICIENT IS A LITERAL 0.5, applied to the numerator, and there is no
//     barycentric-weighted mass sum anywhere.  The softbody solve's `wTot` machinery has
//     no counterpart here; substituting it is the single easiest way to write a port that
//     looks more principled than the thing it transcribes.
//
// A10 BOTH SIDES INDEX THE **SAME** `clothes` ARRAY.  A self-attachment (clothId0 ==
//     clothId1) is legal and lands both scatters in one mDeltaPos buffer, where they may
//     hit the same vertex.  The atomics make that order-independent per component; the
//     fixture therefore compares contested vertices by their order-invariant totals and
//     uncontested ones bitwise.
// ===========================================================================
module femClothAttachBda;

import pxgSlangFEMClothDesc;    // PxgFEMClothGpu (mPosition_InvMass@16, mTriangleVertexIndices@72, mDeltaPos@248)
import pxgSlangAtomicFloat;     // pxgAtomicAddF4XYZ / pxgAtomicAddF4W
import pxgSlangFemFemAttach;    // FAC_* layout of PxgFEMFEMAttachmentConstraint + facSlot

// PxgSoftBodyKernelGridDim/BlockDim::SB_UPDATEROTATION (PxgSoftBodyCoreKernelIndices.h:59/42).
static const uint FCA_GRID  = 1024u;
static const uint FCA_BLOCK = 256u;

// PxgSimulationCoreDesc.h:41 / PxDeformableSurface.h:56.  Both decode paths in the CUDA
// resolve to these; see trap A4.
static const uint PXG_BITSHIFT_ELEMENT_ID = 20u;
static const uint PXG_CLOTH_ELEM_MASK     = 0x000fffffu;

// cloth_solveOutputAttachmentClothClothDeltaVLaunch(PxgFEMCloth* clothes,
//   PxgFEMFEMAttachmentConstraint* attachments, const PxU32 numAttachments)
//   -> 8 @0, 8 @8, 4 @16 -> 24-byte blob.  ABI row {8,8,4}.
struct ClothAttachSolveParams
{
    uint64_t clothes;         // @0
    uint64_t attachments;     // @8
    uint numAttachments;      // @16
    uint _pad20;              // @20 -> 24
}

[[vk::push_constant]] ConstantBuffer<ClothAttachSolveParams> gCa;

// One cloth side: decode the id, blend the anchor, hand back everything the scatter needs.
struct ClothAnchor
{
    uint64_t deltaPos;   // cloth.mDeltaPos
    uint4 triInd;        // the triangle's three vertex indices (.w unused)
    float4 bary;         // barycentric weights
    float4 pos;          // the blended anchor; .w is the INTERPOLATED inverse mass (trap A5)
    float3 vertInvMass;  // the PER-VERTEX inverse masses p*0.w/p*1.w/p*2.w (trap A5)
}

ClothAnchor fcaLoadAnchor(Ptr<PxgFEMClothGpu> clothes, uint clothId, uint elementId, float4 bary)
{
    const uint64_t posBase = clothes[clothId].mPosition_InvMass;
    const uint4 triInd = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[elementId];

    const float4 p0 = Ptr<float4>(posBase)[triInd.x];
    const float4 p1 = Ptr<float4>(posBase)[triInd.y];
    const float4 p2 = Ptr<float4>(posBase)[triInd.z];

    ClothAnchor a;
    a.deltaPos = clothes[clothId].mDeltaPos;
    a.triInd = triInd;
    a.bary = bary;
    a.pos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    a.vertInvMass = float3(p0.w, p1.w, p2.w);
    return a;
}

// The CUDA scatter, literally: three AtomicAdd(dst, deltaPos * bary_k * invMass_k, 1.f)
// with NO skip (trap A7).  `.w` takes ONE per corner regardless of the weight.
void fcaScatter(ClothAnchor a, float3 deltaPos)
{
    pxgAtomicAddF4XYZ(a.deltaPos, a.triInd.x, deltaPos * a.bary.x * a.vertInvMass.x);
    pxgAtomicAddF4W(a.deltaPos, a.triInd.x, 1.0f);
    pxgAtomicAddF4XYZ(a.deltaPos, a.triInd.y, deltaPos * a.bary.y * a.vertInvMass.y);
    pxgAtomicAddF4W(a.deltaPos, a.triInd.y, 1.0f);
    pxgAtomicAddF4XYZ(a.deltaPos, a.triInd.z, deltaPos * a.bary.z * a.vertInvMass.z);
    pxgAtomicAddF4W(a.deltaPos, a.triInd.z, 1.0f);
}

[shader("compute")]
[numthreads(256, 1, 1)]   // SB_UPDATEROTATION
void cloth_solveOutputAttachmentClothClothDeltaVLaunch(uint3 gtid : SV_GroupThreadID,
                                                       uint3 gid : SV_GroupID)
{
    const uint numAttachments = gCa.numAttachments;
    const uint nbBlocksRequired = (numAttachments + FCA_BLOCK - 1u) / FCA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + FCA_GRID - 1u) / FCA_GRID;
    const uint idx = gtid.x;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCa.clothes);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        // Trap A1.
        const uint workIndex = i * FCA_BLOCK + idx + nbIterationsPerBlock * gid.x * FCA_BLOCK;
        if (workIndex >= numAttachments)
            break;   // CUDA `return`; nothing follows the loop here either

        // ---- cloth0 -----------------------------------------------------------------
        // Trap A3: PxU64 slot, low word only.
        const uint elemId0 = Ptr<uint>(facSlot(gCa.attachments, workIndex, FAC_ELEMID0, 8UL))[0];
        const uint clothId0 = elemId0 >> PXG_BITSHIFT_ELEMENT_ID;     // PxGetClothId
        const uint elementId0 = elemId0 & PXG_CLOTH_ELEM_MASK;        // PxGetClothElementIndex
        const float4 bary0 =
            Ptr<float4>(facSlot(gCa.attachments, workIndex, FAC_BARYCENTRIC0, 16UL))[0];
        const ClothAnchor a0 = fcaLoadAnchor(clothes, clothId0, elementId0, bary0);

        // ---- cloth1 -----------------------------------------------------------------
        // Trap A4: open-coded upstream, same constants.
        const uint elemId1 = Ptr<uint>(facSlot(gCa.attachments, workIndex, FAC_ELEMID1, 8UL))[0];
        const uint clothId1 = elemId1 >> PXG_BITSHIFT_ELEMENT_ID;
        const uint elementId1 = elemId1 & PXG_CLOTH_ELEM_MASK;
        const float4 bary1 =
            Ptr<float4>(facSlot(gCa.attachments, workIndex, FAC_BARYCENTRIC1, 16UL))[0];
        const ClothAnchor a1 = fcaLoadAnchor(clothes, clothId1, elementId1, bary1);

        // Trap A5: the INTERPOLATED inverse masses.  Trap A6: `continue`, and the only
        // guard against the unguarded divide two lines below.
        const float invMass0 = a0.pos.w;
        const float invMass1 = a1.pos.w;
        if (invMass1 == 0.0f && invMass0 == 0.0f)
            continue;

        const float coefficient = 0.5f;                               // trap A9
        const float4 diff = a1.pos - a0.pos;
        const float4 deltaImpulse = diff * coefficient / (invMass0 + invMass1);
        const float3 deltaImp = deltaImpulse.xyz;                     // PxLoad3

        fcaScatter(a0, deltaImp);                                     // trap A8
        fcaScatter(a1, -deltaImp);
    }
}
