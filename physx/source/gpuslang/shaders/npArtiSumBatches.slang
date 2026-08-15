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


// npArtiSumBatches.slang
//
// P8.96a: `artiSumInternalContactAndJointBatches1Launch`
// (internalConstraints2.cu:3371, body at :2864) -- the first kernel of the
// articulation constraint-PREP chain (sub-chain 3), and the articulation twin
// of `rigidSumInternalContactAndJointBatches1` (npRigidSumBatches.slang, P5):
// per 512-thread block, reduce the per-articulation STATIC and SELF
// contact/joint counts into eight per-block scratch values.  Manifest 278 ->
// 279.
//
// The porting strategy is the rigid twin's, unchanged: under llvmpipe's
// subgroup 8 a faithful 16-warp shared-memory reduction buys nothing, so only
// the thread that STORES in the CUDA (threadIdx.x == warpPerBlock-1 == 15)
// runs, and it recomputes the block's reductions as pure loops --
// deterministic uint adds and maxes, bit-free of ordering concerns.  The
// UNIQUE-INDICES outputs are plain block sums; the HEADER outputs are sums of
// PER-WARP-CHUNK MAXES, so the 32-element chunk structure is reproduced
// exactly (a chunk hanging past nbArticulations contributes zeros).
//
// TRAPS, each a mutant:
//  1. The iteration origin: workIndex = i*512 + t + numIter*b*512 -- the
//     BLOCK's slice is numIter CONSECUTIVE 512-spans, not a 32-block stride.
//  2. Headers are sums of chunk-MAXES; unique-indices are sums.  Swapping the
//     aggregation on one family is the natural transposition error.
//  3. The STATIC counts come from the prePrep desc's mArtiStatic* arrays and
//     the SELF counts from mArtiSelf* -- four independent source arrays, and
//     the eight destinations pair them (static contact / static joint / self
//     contact / self joint) x (unique sum / header max-sum).
//  4. The eight destinations live in the ARTI CORE desc (280..343), not in
//     the prePrep desc where the rigid twin's four live.
//
// SWEEP SURVIVOR B01, equivalent BY CONSTRUCTION of the serial port: the guard
// picks which single thread runs the recompute, and every thread computes the
// same eight values from the same inputs -- the CUDA's choice of thread
// warpPerBlock-1 matters THERE because only that thread holds the accumulated
// shared-memory sums; here there is no shared state to hold.  The guard keeps
// the CUDA's spelling so a reader diffing the two finds the same thread.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangPrepCommon;
import pxgSlangPrepDesc;

static const uint ASB_BLOCK_DIM = 512u;   // COMPUTE_STATIC_CONTACT_CONSTRAINT_COUNT
static const uint ASB_WARPS = 16u;        // warpPerBlock

// CLAMP THE INDEX, NOT JUST THE VALUE (the P8.96c finding, dmesg-verified):
// llvmpipe compiles the guarded uniform load `wi < nb ? counts[wi] : 0` to an
// UNCONDITIONAL scalar load + select (cmp/sbb mask, then an unmasked
// vbroadcastss), so the load itself executes with wi up to numIter*32*512-1 --
// tens of KB past the array -- and segfaults whenever the pages behind the
// buffer happen to be unmapped (an allocation-layout lottery, ~35% in the
// batches2 fixture).  With the index clamped the speculated load is always
// in bounds and the select still returns 0 for the out-of-range lanes.
uint asbCountAt(Ptr<uint> counts, uint workIndex, uint nbArticulations)
{
    const uint idx = workIndex < nbArticulations ? workIndex : 0u;
    const uint v = counts[idx];
    return workIndex < nbArticulations ? v : 0u;
}

struct AsbParams
{
    uint64_t scDesc;        // @0
    uint64_t prePrepDesc;   // @8
    uint nbArticulations;   // @16
    uint pad0;              // @20 -> 24
};

[[vk::push_constant]]
ConstantBuffer<AsbParams> gAsb;

[shader("compute")]
[numthreads(512, 1, 1)]
void artiSumInternalContactAndJointBatches1Launch(uint3 groupId : SV_GroupID,
                                                  uint3 groupThreadId : SV_GroupThreadID)
{
    const uint tid = groupThreadId.x;
    if (tid != (ASB_WARPS - 1u))   // cu :3020: if (threadIdx.x == warpPerBlock - 1)
        return;
    const uint b = groupId.x;

    const uint64_t d = gAsb.scDesc;
    Ptr<PxgPrePrepDescGpu> pp = Ptr<PxgPrePrepDescGpu>(gAsb.prePrepDesc);
    Ptr<uint> staticContact = Ptr<uint>(pp[0].mArtiStaticContactCounts);
    Ptr<uint> staticJoint = Ptr<uint>(pp[0].mArtiStaticConstraintCounts);
    Ptr<uint> selfContact = Ptr<uint>(pp[0].mArtiSelfContactCounts);
    Ptr<uint> selfJoint = Ptr<uint>(pp[0].mArtiSelfConstraintCounts);

    const uint nbArticulations = gAsb.nbArticulations;
    const uint totalBlockRequired = (nbArticulations + (ASB_BLOCK_DIM - 1u)) / ASB_BLOCK_DIM;
    const uint numIter = (totalBlockRequired + 31u) / 32u;

    uint sumSC = 0u, sumSJ = 0u, hdrSC = 0u, hdrSJ = 0u;
    uint sumFC = 0u, sumFJ = 0u, hdrFC = 0u, hdrFJ = 0u;
    for (uint i = 0u; i < numIter; ++i)
    {
        for (uint w = 0u; w < ASB_WARPS; ++w)
        {
            uint wMaxSC = 0u, wMaxSJ = 0u, wMaxFC = 0u, wMaxFJ = 0u;
            for (uint l = 0u; l < 32u; ++l)
            {
                const uint workIndex = i * ASB_BLOCK_DIM + (w * 32u + l)
                                     + numIter * b * ASB_BLOCK_DIM;
                const uint sc = asbCountAt(staticContact, workIndex, nbArticulations);
                const uint sj = asbCountAt(staticJoint, workIndex, nbArticulations);
                const uint fc = asbCountAt(selfContact, workIndex, nbArticulations);
                const uint fj = asbCountAt(selfJoint, workIndex, nbArticulations);
                sumSC += sc; sumSJ += sj; sumFC += fc; sumFJ += fj;
                wMaxSC = sc > wMaxSC ? sc : wMaxSC;
                wMaxSJ = sj > wMaxSJ ? sj : wMaxSJ;
                wMaxFC = fc > wMaxFC ? fc : wMaxFC;
                wMaxFJ = fj > wMaxFJ ? fj : wMaxFJ;
            }
            hdrSC += wMaxSC; hdrSJ += wMaxSJ; hdrFC += wMaxFC; hdrFJ += wMaxFJ;
        }
    }

    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONTACT_UNIQUE)))[b] = sumSC;
    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONSTRAINT_UNIQUE)))[b] = sumSJ;
    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONTACT_HEADER)))[b] = hdrSC;
    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_CONSTRAINT_HEADER)))[b] = hdrSJ;
    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONTACT_UNIQUE)))[b] = sumFC;
    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONSTRAINT_UNIQUE)))[b] = sumFJ;
    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONTACT_HEADER)))[b] = hdrFC;
    (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_TEMP_SELF_CONSTRAINT_HEADER)))[b] = hdrFJ;
}
