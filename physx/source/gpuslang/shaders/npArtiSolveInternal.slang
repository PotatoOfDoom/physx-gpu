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


// npArtiSolveInternal.slang
//
// P8.100a: the first slice of artiSolveInternalConstraintsTGS1T
// (internalConstraints2.cu:2842, template body :1579-2013) — the STATIC
// constraint solve, solveStaticConstraints (TGS overload, :1395-1552):
// one articulated link against the static-world batches the P8.96b prep
// wrote into the link's per-link windows (mStaticJointStartIndex /
// mStaticContactStartIndex / mNbStaticContacts; the JOINT count comes from
// the CALLER, as in the CUDA).
//
// ===========================================================================
// TRAPS (each carries a mutant in the sweep unless marked equivalent)
// ===========================================================================
//
//  1. **The joint loop runs BEFORE the contact loop**, and `vel` travels
//     through both — a batch's solve sees every earlier batch's velocity
//     update.
//  2. **The articulated link can sit on EITHER side of a static batch**
//     (bodyA or bodyB, decided per batch by PxNodeIndex::isArticulation);
//     the static side gets ZERO velocity/delta and the IDENTITY deltaQ, and
//     its impulse output is DISCARDED.  `impulse` accumulates only the
//     articulation side's output.
//  3. **The per-lane batch slot is the exclusive-popcount offset** within
//     batch.mask, exactly as in the self-solve.
//  4. **deltaQ is loaded only when the JOINT count is nonzero** — the
//     contact path never reads it.  (Loading it unconditionally would be
//     numerically equivalent — the load has no side effect — documented,
//     not swept.)
//  5. **`deltaV += vel - oldVel` fires only when any batch ran** — with both
//     counts zero, deltaV must come back untouched.
//  6. The CUDA reassigns delta/deltaQ from the chosen side after each batch;
//     the solve bodies take them BY VALUE, so those reassignments restore
//     the same values — kept for fidelity, equivalent by construction.
//  7. doFriction is NOT passed to the TGS contact body (the PGS overload
//     takes it; TGS always solves friction) — the caller's flag dies here.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangSolverBlocks;
import npArtiSolveTendons;
import npArtiSolveExtTGS;

static const uint SI_BATCH_STRIDE = 1536u;   // sizeof(PxgBlockConstraintBatch)

// solveStaticConstraints (TGS overload, internalConstraints2.cu:1395)
public void siSolveStaticConstraintsTGS(
    uint64_t link, uint64_t batches,
    uint64_t contactHeaders, uint64_t frictionHeaders,
    uint64_t contactPoints, uint64_t frictionPoints,
    uint64_t jointHeaders, uint64_t jointRowsCon, uint64_t responses,
    inout AbSpatialVector vel, inout AbSpatialVector impulse, inout AbSpatialVector deltaV,
    uint lane, float minPen, float elapsedTime, bool accumulateError,
    inout SvxError error, uint constraintCounts)
{
    const uint constraintBatchOffset = abU32(link, PXG_AB_LINK_mStaticJointStartIndex, lane);
    const uint contactBatchOffset = abU32(link, PXG_AB_LINK_mStaticContactStartIndex, lane);
    const uint contactCounts = abU32(link, PXG_AB_LINK_mNbStaticContacts, lane);

    const AbSpatialVector oldVel = vel;

    AbSpatialVector delta = svLoadSV(link, PXG_AB_LINK_mDeltaMotion, lane);

    // trap 4: only the joint path reads deltaQ
    float4 deltaQ = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (constraintCounts > 0u)
        deltaQ = abF4(link, PXG_AB_LINK_mDeltaQ, lane);

    const AbSpatialVector zeroV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    const float4 identityQ = float4(0.0f, 0.0f, 0.0f, 1.0f);

    // trap 1: joints first
    for (uint i = 0u; i < constraintCounts; ++i)
    {
        Ptr<PxgBlockConstraintBatch> batch =
            Ptr<PxgBlockConstraintBatch>(batches + uint64_t(constraintBatchOffset + i) * SI_BATCH_STRIDE);
        const uint mask = batch[0].mask;
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const bool aIsArti = nodeIndexIsArticulation(batch[0].bodyANodeIndex[offset]);

        AbSpatialVector vel0, vel1, delta0, delta1;
        float4 deltaQ0, deltaQ1;
        if (aIsArti)
        {
            vel0 = vel;   delta0 = delta;  deltaQ0 = deltaQ;
            vel1 = zeroV; delta1 = zeroV;  deltaQ1 = identityQ;
        }
        else
        {
            vel0 = zeroV; delta0 = zeroV;  deltaQ0 = identityQ;
            vel1 = vel;   delta1 = delta;  deltaQ1 = deltaQ;
        }

        AbSpatialVector impulse0 = zeroV;
        AbSpatialVector impulse1 = zeroV;

        svxSolveExt1DBlockTGS(batch, vel0, vel1, delta0, delta1, offset,
                              Ptr<PxgTGSBlockSolverConstraint1DHeader>(jointHeaders),
                              Ptr<PxgTGSBlockSolverConstraint1DCon>(jointRowsCon),
                              Ptr<PxgArticulationBlockResponse>(responses),
                              deltaQ0, deltaQ1, elapsedTime, impulse0, impulse1,
                              accumulateError, 1.0f, 1.0f);

        if (aIsArti) { impulse = asmSVecAdd(impulse, impulse0); vel = vel0; delta = delta0; deltaQ = deltaQ0; }
        else         { impulse = asmSVecAdd(impulse, impulse1); vel = vel1; delta = delta1; deltaQ = deltaQ1; }
    }

    for (uint i = 0u; i < contactCounts; ++i)
    {
        Ptr<PxgBlockConstraintBatch> batch =
            Ptr<PxgBlockConstraintBatch>(batches + uint64_t(contactBatchOffset + i) * SI_BATCH_STRIDE);
        const uint mask = batch[0].mask;
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const bool aIsArti = nodeIndexIsArticulation(batch[0].bodyANodeIndex[offset]);

        AbSpatialVector vel0, vel1, delta0, delta1;
        float4 deltaQ0, deltaQ1;
        if (aIsArti)
        {
            vel0 = vel;   delta0 = delta;  deltaQ0 = deltaQ;
            vel1 = zeroV; delta1 = zeroV;  deltaQ1 = identityQ;
        }
        else
        {
            vel0 = zeroV; delta0 = zeroV;  deltaQ0 = identityQ;
            vel1 = vel;   delta1 = delta;  deltaQ1 = deltaQ;
        }

        AbSpatialVector impulse0 = zeroV;
        AbSpatialVector impulse1 = zeroV;

        svxSolveExtContactBlockTGS(batch, vel0, vel1, delta0, delta1, offset,
                                   Ptr<PxgTGSBlockSolverContactHeader>(contactHeaders),
                                   Ptr<PxgTGSBlockSolverFrictionHeader>(frictionHeaders),
                                   Ptr<PxgTGSBlockSolverContactPoint>(contactPoints),
                                   Ptr<PxgTGSBlockSolverContactFriction>(frictionPoints),
                                   Ptr<PxgArticulationBlockResponse>(responses),
                                   elapsedTime, minPen, impulse0, impulse1,
                                   error, accumulateError, 1.0f, 1.0f);

        if (aIsArti) { impulse = asmSVecAdd(impulse, impulse0); vel = vel0; delta = delta0; deltaQ = deltaQ0; }
        else         { impulse = asmSVecAdd(impulse, impulse1); vel = vel1; delta = delta1; deltaQ = deltaQ1; }
    }

    // trap 5
    if ((constraintCounts + contactCounts) > 0u)
        deltaV = asmSVecAdd(deltaV, asmSVecSub(vel, oldVel));
}

// ===========================================================================
// test entry
// ===========================================================================

struct SiParams
{
    uint64_t link;            // @0
    uint64_t batches;         // @8
    uint64_t contactHeaders;  // @16
    uint64_t frictionHeaders; // @24
    uint64_t contactPoints;   // @32
    uint64_t frictionPoints;  // @40
    uint64_t jointHeaders;    // @48
    uint64_t jointRowsCon;    // @56
    uint64_t responses;       // @64
    uint64_t io;              // @72  in: vel/impulse/deltaV (18 floats @ lane*24);
                              //      out at 32*24 + lane*24 (+error triple)
    float minPen;             // @80
    float elapsedTime;        // @84
    uint accumulateError;     // @88
    uint nbLanes;             // @92
}

[[vk::push_constant]]
ConstantBuffer<SiParams> gSi;

static const uint SI_IO_STRIDE = 24u;
static const uint SI_IO_OUT_BASE = 32u * 24u;

[shader("compute")]
[numthreads(32, 1, 1)]
void siStaticTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSi.nbLanes)
        return;

    Ptr<float> fin = Ptr<float>(gSi.io + uint64_t(lane * SI_IO_STRIDE) * 4u);
    AbSpatialVector vel = asmSVec(float3(fin[0], fin[1], fin[2]), float3(fin[3], fin[4], fin[5]));
    AbSpatialVector impulse = asmSVec(float3(fin[6], fin[7], fin[8]), float3(fin[9], fin[10], fin[11]));
    AbSpatialVector deltaV = asmSVec(float3(fin[12], fin[13], fin[14]), float3(fin[15], fin[16], fin[17]));

    // the CALLER reads the joint count from the link, as the kernel does
    const uint constraintCounts = abU32(gSi.link, PXG_AB_LINK_mNbStaticJoints, lane);

    SvxError error = { 0.0f, 0, 0.0f };
    siSolveStaticConstraintsTGS(gSi.link, gSi.batches,
                                gSi.contactHeaders, gSi.frictionHeaders,
                                gSi.contactPoints, gSi.frictionPoints,
                                gSi.jointHeaders, gSi.jointRowsCon, gSi.responses,
                                vel, impulse, deltaV,
                                lane, gSi.minPen, gSi.elapsedTime,
                                gSi.accumulateError != 0u, error, constraintCounts);

    Ptr<float> fout = Ptr<float>(gSi.io + uint64_t(SI_IO_OUT_BASE + lane * SI_IO_STRIDE) * 4u);
    fout[0] = vel.top.x; fout[1] = vel.top.y; fout[2] = vel.top.z;
    fout[3] = vel.bottom.x; fout[4] = vel.bottom.y; fout[5] = vel.bottom.z;
    fout[6] = impulse.top.x; fout[7] = impulse.top.y; fout[8] = impulse.top.z;
    fout[9] = impulse.bottom.x; fout[10] = impulse.bottom.y; fout[11] = impulse.bottom.z;
    fout[12] = deltaV.top.x; fout[13] = deltaV.top.y; fout[14] = deltaV.top.z;
    fout[15] = deltaV.bottom.x; fout[16] = deltaV.bottom.y; fout[17] = deltaV.bottom.z;
    fout[18] = error.sumOfSquares; fout[19] = float(error.counter); fout[20] = error.maxError;
}
