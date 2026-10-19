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

// npArtiPropagateInertia.slang
//
// P8.87: `computeIs` (forwardDynamic2.cu:1347-1365) and
// `computePropagateSpatialInertia_ZA_ZIc` (:1401-1544) — the per-dof algebra that
// turns a link's articulated spatial inertia into the term the parent's inertia is
// reduced by, and accumulates the joint's contribution into the zero-acceleration
// force vectors along the way.
//
// The engine's own comment above the second function states what it computes:
// propagating I_i^A from child i to parent h is `I_h^A += delta` with
// `delta = I_i^A - (I_i^A s s^T I_i^A) / (s I_i^A s)`, and this function returns
// the subtrahend.  The caller (P8.88) forms the difference.
//
// ===========================================================================
// THE ORACLE, STATED HONESTLY
// ===========================================================================
//
// Both functions are `static __device__` and cannot be called from the host, so
// this is the P8.73-P8.80 arrangement rather than P8.86's: the fixture's mirror is
// CONTROL FLOW over engine primitives.  What the control flow is built from *is*
// callable and is called — `SpatialMatrix::operator*`,
// `UnAlignedSpatialVector::innerProduct`, both `constructSpatialMatrix` forms,
// `PxMat33::getInverse`, `loadSpatialVector` / `storeSpatialVector` — and every one
// of those has its own P8.86 fixture underneath it.  The sweep is not presented as
// an oracle; it is the check on the control flow the mirror shares.
//
// One engine duplication worth naming rather than reproducing: the local
// `constructSpatialMatrix(Is, stI)` at :1367 is byte-for-byte the body of
// `SpatialMatrix::constructSpatialMatrix(SpatialVectorF, SpatialVectorF)` at
// DyFeatherstoneArticulationUtils.h:364.  The port has one function
// (`asmConstructSpatialMatrix`) and this note, not two.
//
// ===========================================================================
// THE JOINT CASES
// ===========================================================================
//
// `PxArticulationJointType` (PxSolverDefs.h:301): eFIX 0, ePRISMATIC 1,
// eREVOLUTE 2, eREVOLUTE_UNWRAPPED 3, eSPHERICAL 4, eUNDEFINED 5.  The switch
// handles 1/2/3 as one case, 4 as another, and 0/5 fall to `default`, which
// returns `SpatialMatrix(PxZero)` and **writes nothing at all** — no `mQstZ`, no
// `mIsInvDW`, no `mInvStIsT`.
//
// THE ONE-DOF CASE hides two asymmetries that look like bugs and are not:
//
//   * `stI` is `UnAlignedSpatialVector(Is.bottom, Is.top)` — the halves are
//     SWAPPED on purpose, because `stI` is the transpose of a motion vector acting
//     on a force, and the dual basis exchanges the slots.  The spherical case
//     spells the same swap as "columns 0..2 from `.bottom`, columns 3..5 from
//     `.top`".
//   * `qstZ = -stZ` ignores the joint force entirely while
//     `qstZIcInternal = jF[0] - stZInt` uses it.  The engine's own commented-out
//     `//link.qstZIc[ind] = jF[ind] - stZ;` sits directly above, which is exactly
//     what makes "fixing" the first line look right.
//
// THE SPHERICAL CASE has the biggest trap in the tranche:
//
//     PxMat33 invStIs(PxIdentity);
//
// It starts as the IDENTITY, not zero.  A two-dof spherical joint never writes the
// third row or column, so those entries stay `(0,0,1)` and the matrix inverts.
// Seed it from zero instead and it is singular, `getInverse` returns the identity,
// and every number downstream is wrong — silently, because the shape is right.
//
// And it inverts with `PxMat33::getInverse`, NOT `SpatialMatrix::invertSym33`;
// the engine's commented-out `//invStIs = SpatialMatrix::invertSym33(invStIs);`
// records the deliberate switch.  P8.86 measured why it matters: on a
// non-symmetric input the two disagree in a third of all words, and
// `invStIs[ind][ind2] = s_ind2 · Is_ind` is not symmetric.
//
// ===========================================================================
// P8.84'S FOURTH PLATFORM RULE, SHARPENED HERE
// ===========================================================================
//
// The spherical case has SEVEN accumulators the CUDA seeds with a literal zero —
// `isID` and `columns[0..5]`.  P8.84's rule says `x + 0.0` is folded away, which is
// not IEEE for `-0`: the engine's C++ computes `(+0) + (-0) = +0` and the folded
// form yields `-0`.  Two things about it were measured in this tranche and neither
// was obvious:
//
//  1. **The fold is the DRIVER's, and no shader-level workaround defeats it.**
//     Seeding the seven accumulators from `abs(armature) * 0.0f` — P8.84's own
//     remedy — leaves the result unchanged, and the SPIR-V `OpFAdd` count is the
//     same (12) either way.  So the seeds below are literal zeros, exactly as the
//     CUDA spells them, and the divergence is handled where P8.84 also ended up
//     handling it: a comparison tolerance scoped to `0x00000000` vs `0x80000000`,
//     with the count printed so it cannot drift.
//     P9.86 SUPERSEDES THIS: the fold only ever turns the CUDA's `+0` into `-0`,
//     so `isID` and the six columns are canonicalised after accumulation with the
//     select `apiPlusZero3` ((x == 0) ? 0 : x), which is the CUDA value bit for
//     bit.  The test's signed-zero tolerance on lane 10 is now obsolete.
//  2. **The seed's sign is almost always washed out, which is why this is hard to
//     see at all.**  IEEE gives `(-0) + (+0) = +0`, so a single ordinary term
//     erases it.  A three-dof spherical joint with a diagonal `invStIs` — the
//     obvious construction, and the fixture's lane 6 — therefore does NOT expose
//     the fold: the `-0` coefficient contributes a `-0` first term and the next
//     iteration's `+0` wipes it.  The sweep found this by surviving a mutant that
//     seeded `-0` deliberately.
//
// The case that DOES expose it is a **one-dof spherical joint whose `Is` is all
// negative zeros** — exactly one accumulated term, so nothing can wash it out —
// which is the fixture's lane 10: a zero spatial inertia (a massless link) against
// an all-negative motion matrix, so every component of `M * s` is
// `0 * negative = -0`.  Twelve words are forgiven on that lane and nowhere else.
//
// ===========================================================================
// COLUMN, THEN ROW
// ===========================================================================
//
// `invStIs[ind][ind2]` in the CUDA is COLUMN `ind`, component `ind2`, because
// `PxMat33::operator[]` returns a column.  The matrix is built in a flat
// `float e[9]` indexed `e[col * 3 + row]` rather than through `AbMat33`'s named
// fields, so the dynamic indices the CUDA uses stay dynamic and the column-major
// convention is written down once instead of being re-derived at each use.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;

public static const uint PXG_ARTI_JOINT_eFIX = 0u;
public static const uint PXG_ARTI_JOINT_ePRISMATIC = 1u;
public static const uint PXG_ARTI_JOINT_eREVOLUTE = 2u;
public static const uint PXG_ARTI_JOINT_eREVOLUTE_UNWRAPPED = 3u;
public static const uint PXG_ARTI_JOINT_eSPHERICAL = 4u;
public static const uint PXG_ARTI_JOINT_eUNDEFINED = 5u;

// The armature lives in the dof block's internal-constraint sub-block, so it takes
// BOTH offsets: the sub-block's position within the dof block and the field's
// position within the sub-block.  Reading it at PXG_AB_IC_mArmature alone lands in
// the dof block's own fields.
float apiArmature(uint64_t dof, uint lane)
{
    return abF32(dof + PXG_AB_DOF_mConstraintData, PXG_AB_IC_mArmature, lane);
}

// PxMat33 element access with the engine's convention spelled out: column first.
// Written as selects rather than dynamic indexing so the lowering is the same on
// every driver -- these are three-way choices over a register, not a loop.
float apiMatAt(AbMat33 m, uint col, uint row)
{
    const float3 c = (col == 0u) ? m.c0 : ((col == 1u) ? m.c1 : m.c2);
    return (row == 0u) ? c.x : ((row == 1u) ? c.y : c.z);
}

// IEEE `+0 + x` per component, as a select (P9.86): +0 for either zero, x otherwise
// (NaN and inf pass through, NaN == 0 is false).
float3 apiPlusZero3(float3 v)
{
    return float3((v.x == 0.0f) ? 0.0f : v.x, (v.y == 0.0f) ? 0.0f : v.y,
                  (v.z == 0.0f) ? 0.0f : v.z);
}

// ---------------------------------------------------------------------------
// computeIs (forwardDynamic2.cu:1347)
// ---------------------------------------------------------------------------
//
// Four lines: the link's articulated spatial inertia applied to each dof's world
// motion matrix, kept in `Is` for the caller and written to the block's `mIsW`.
//
// The CUDA's `linkData` parameter is never read; nor is
// computePropagateSpatialInertia_ZA_ZIc's `linkID`.  Both are dropped here rather
// than carried as dead weight.
//
// `Is` entries at or beyond `dofs` are left as the caller had them -- the CUDA
// leaves them uninitialised, and every consumer is guarded by the same `dofs`.
public void apiComputeIs(uint dofs, AbSpatialMatrix spatialInertia, uint64_t dofBlocks,
                         uint lane, inout AbSpatialVector Is[3])
{
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofs)
        {
            const uint64_t dof = dofBlocks + uint64_t(ind) * PXG_AB_DOF_SIZE;

            float3 mt, mb;
            abLoadSpatialVector(dof, PXG_AB_DOF_mWorldMotionMatrix, lane, mt, mb);

            const AbSpatialVector lIs = asmMulVec(spatialInertia, asmSVec(mt, mb));
            Is[ind] = lIs;
            abStoreSpatialVector(dof, PXG_AB_DOF_mIsW, lane, lIs.top, lIs.bottom);
        }
    }
}

// ---------------------------------------------------------------------------
// computePropagateSpatialInertia_ZA_ZIc (forwardDynamic2.cu:1401)
// ---------------------------------------------------------------------------
//
// `jF` is a POINTER in the CUDA.  It is carried here as a device address where
// **0 means NULL**, which is the natural port and the reason the null test is a
// real branch rather than a flag.
//
// 5.12: the wrapper always passes jF now (5.6.1 passed NULL under
// `isExternalForcesEveryTgsIterationEnabled`), and the flag comes in as its own
// argument instead: it zeroes the joint force in qstZIcInternal only, while the
// raw per-dof force is parked in mConstraintData.mExternalEffort either way (the
// solver reads it from there instead of from articulation.jointForce).
public AbSpatialMatrix apiPropagateSpatialInertiaZaZIc(uint64_t linkBlock, uint64_t dofBlocks,
                                                       AbSpatialVector msIs[3],
                                                       bool isExternalForcesEveryTgsIterationEnabled,
                                                       uint64_t jF,
                                                       AbSpatialVector Z, AbSpatialVector ZIcInt,
                                                       inout AbSpatialVector ZA,
                                                       inout AbSpatialVector ZAInt, uint lane)
{
    const uint jointType = abU8(linkBlock, PXG_AB_LINK_mJointType, lane);

    if (jointType == PXG_ARTI_JOINT_ePRISMATIC || jointType == PXG_ARTI_JOINT_eREVOLUTE
        || jointType == PXG_ARTI_JOINT_eREVOLUTE_UNWRAPPED)
    {
        const uint64_t dof0 = dofBlocks;

        float3 sat, sab;
        abLoadSpatialVector(dof0, PXG_AB_DOF_mWorldMotionMatrix, lane, sat, sab);
        const AbSpatialVector sa = asmSVec(sat, sab);
        const AbSpatialVector Is = msIs[0];

        // the DUAL product, not the Euclidean one
        const float stIs = asmInnerProduct(sa, Is) + apiArmature(dof0, lane);

        // `> 0`, not `!= 0`: a negative stIs is a rounding artefact and is treated
        // as an unconstrained axis, not inverted.
        const float iStIs = (stIs > 0.0f) ? (1.0f / stIs) : 0.0f;
        // mInvStIsT_x holds a SCALAR in this case and a matrix column in the
        // spherical one -- same field, two meanings.
        abStoreF32(dof0, PXG_AB_DOF_mInvStIsT_x, lane, iStIs);

        const AbSpatialVector isInvD = asmSVecScale(Is, iStIs);
        abStoreSpatialVector(dof0, PXG_AB_DOF_mIsInvDW, lane, isInvD.top, isInvD.bottom);

        // (6x1) Is = [v0, v1]; (1x6) stI = [v1, v0] -- the halves swap
        const AbSpatialVector stI = asmSVec(Is.bottom, Is.top);

        const float stZ = asmInnerProduct(sa, Z);
        const float stZInt = asmInnerProduct(sa, ZIcInt);

        // qstZ ignores the joint force; qstZIcInternal does not
        const float qstZ = -stZ;
        const float qstZIcInternal =
            ((!isExternalForcesEveryTgsIterationEnabled && jF != 0) ? (Ptr<float>(jF))[0] : 0.0f) - stZInt;

        abStoreF32(dof0, PXG_AB_DOF_mQstZ, lane, qstZ);
        abStoreF32(dof0, PXG_AB_DOF_mQstZIcInternal, lane, qstZIcInternal);
        // `jF ? jF[0] : 0.0` -- a double literal in the CUDA, exact either way
        abStoreF32(dof0 + PXG_AB_DOF_mConstraintData, PXG_AB_IC_mExternalEffort, lane,
                   (jF != 0) ? (Ptr<float>(jF))[0] : 0.0f);

        ZA = asmSVecAdd(ZA, asmSVecScale(isInvD, qstZ));
        ZAInt = asmSVecAdd(ZAInt, asmSVecScale(isInvD, qstZIcInternal));

        return asmConstructSpatialMatrix(isInvD, stI);
    }

    if (jointType == PXG_ARTI_JOINT_eSPHERICAL)
    {
        const uint dofs = abU8(linkBlock, PXG_AB_LINK_mDofs, lane);

        // THE IDENTITY, not zero -- see the header.  e[col * 3 + row].
        float e[9];
        for (uint k = 0u; k < 9u; ++k)
            e[k] = 0.0f;
        e[0] = 1.0f;
        e[4] = 1.0f;
        e[8] = 1.0f;

        float qstZG[3];
        float qstZIcIntG[3];
        for (uint k2 = 0u; k2 < 3u; ++k2)
        {
            qstZG[k2] = 0.0f;
            qstZIcIntG[k2] = 0.0f;
        }

        for (uint ind2 = 0u; ind2 < 3u; ++ind2)
        {
            if (ind2 < dofs)
            {
                const uint64_t dof = dofBlocks + uint64_t(ind2) * PXG_AB_DOF_SIZE;

                float3 sat, sab;
                abLoadSpatialVector(dof, PXG_AB_DOF_mWorldMotionMatrix, lane, sat, sab);
                const AbSpatialVector sa = asmSVec(sat, sab);

                for (uint ind = 0u; ind < 3u; ++ind)
                {
                    if (ind < dofs)
                        e[ind * 3u + ind2] = asmInnerProduct(sa, msIs[ind]);
                }

                // the armature adds to the diagonal AFTER the assignment above has
                // already written that same entry
                e[ind2 * 3u + ind2] = e[ind2 * 3u + ind2] + apiArmature(dof, lane);

                const float stZ = asmInnerProduct(sa, Z);
                const float stZInt = asmInnerProduct(sa, ZIcInt);

                const float qstZ = -stZ;
                const float qstZicInt =
                    ((!isExternalForcesEveryTgsIterationEnabled && jF != 0) ? (Ptr<float>(jF))[ind2] : 0.0f) - stZInt;

                qstZG[ind2] = qstZ;
                qstZIcIntG[ind2] = qstZicInt;
                abStoreF32(dof, PXG_AB_DOF_mQstZ, lane, qstZ);
                abStoreF32(dof, PXG_AB_DOF_mQstZIcInternal, lane, qstZicInt);
                abStoreF32(dof + PXG_AB_DOF_mConstraintData, PXG_AB_IC_mExternalEffort, lane,
                           (jF != 0) ? (Ptr<float>(jF))[ind2] : 0.0f);
            }
        }

        // the GENERAL inverse -- invertSym33 is wrong here, see the header
        const AbMat33 invStIs = asmM33Inverse(asmM33(float3(e[0], e[1], e[2]),
                                                     float3(e[3], e[4], e[5]),
                                                     float3(e[6], e[7], e[8])));

        // written back TRANSPOSED: dof ind2 takes row ind2 of the inverse
        for (uint ind2 = 0u; ind2 < 3u; ++ind2)
        {
            if (ind2 < dofs)
            {
                const uint64_t dof = dofBlocks + uint64_t(ind2) * PXG_AB_DOF_SIZE;
                abStoreF32(dof, PXG_AB_DOF_mInvStIsT_x, lane, apiMatAt(invStIs, 0u, ind2));
                abStoreF32(dof, PXG_AB_DOF_mInvStIsT_y, lane, apiMatAt(invStIs, 1u, ind2));
                abStoreF32(dof, PXG_AB_DOF_mInvStIsT_z, lane, apiMatAt(invStIs, 2u, ind2));
            }
        }

        // literal zeros, exactly as the CUDA spells them.  The header records why a
        // runtime-zero seed does not help (the fold is the driver's); P9.86 instead
        // canonicalises the finished accumulators with apiPlusZero3, below.
        const float3 z3 = float3(0.0f, 0.0f, 0.0f);
        AbSpatialVector columns[6];
        for (uint k = 0u; k < 6u; ++k)
            columns[k] = asmSVec(z3, z3);

        for (uint ind = 0u; ind < 3u; ++ind)
        {
            if (ind < dofs)
            {
                AbSpatialVector isID = asmSVec(z3, z3);

                for (uint ind2 = 0u; ind2 < 3u; ++ind2)
                {
                    if (ind2 < dofs)
                        isID = asmSVecAdd(isID,
                                          asmSVecScale(msIs[ind2], apiMatAt(invStIs, ind, ind2)));
                }

                // P9.86: isID starts at a literal +0 and only ever += a term, so the CUDA's
                // IEEE chain (forwardDynamic2.cu:1511/1518) never holds -0.  llvmpipe unrolls
                // the trip-3 loop, forwards the seed and folds the first `0 + t` to t, keeping
                // a -0 term.  Canonicalised: bit for bit the engine's value.
                isID = asmSVec(apiPlusZero3(isID.top), apiPlusZero3(isID.bottom));

                // BOTTOM first, then top -- the one-dof case's stI swap, spelled
                // over six columns
                columns[0] = asmSVecAdd(columns[0], asmSVecScale(isID, msIs[ind].bottom.x));
                columns[1] = asmSVecAdd(columns[1], asmSVecScale(isID, msIs[ind].bottom.y));
                columns[2] = asmSVecAdd(columns[2], asmSVecScale(isID, msIs[ind].bottom.z));
                columns[3] = asmSVecAdd(columns[3], asmSVecScale(isID, msIs[ind].top.x));
                columns[4] = asmSVecAdd(columns[4], asmSVecScale(isID, msIs[ind].top.y));
                columns[5] = asmSVecAdd(columns[5], asmSVecScale(isID, msIs[ind].top.z));

                const uint64_t dof = dofBlocks + uint64_t(ind) * PXG_AB_DOF_SIZE;
                abStoreSpatialVector(dof, PXG_AB_DOF_mIsInvDW, lane, isID.top, isID.bottom);

                ZA = asmSVecAdd(ZA, asmSVecScale(isID, qstZG[ind]));
                ZAInt = asmSVecAdd(ZAInt, asmSVecScale(isID, qstZIcIntG[ind]));
            }
        }

        // P9.86: every column starts at a literal +0 and only ever += a term, so the
        // CUDA's IEEE chain (forwardDynamic2.cu:1499-1527) never holds -0.  llvmpipe
        // unrolls the 3-trip loop, forwards the seed and folds the first `+0 + t` to t,
        // so an all -0 column survives as -0.  The fold only changes the sign of a
        // zero, so canonicalising the finished columns is the engine's value bit for bit.
        for (uint k = 0u; k < 6u; ++k)
            columns[k] = asmSVec(apiPlusZero3(columns[k].top), apiPlusZero3(columns[k].bottom));

        return asmConstructFromColumns(columns);
    }

    // eFIX and eUNDEFINED: contributes nothing, and writes nothing
    return asmMatrixZero();
}

// ===========================================================================
// The test harness entry point
// ===========================================================================
//
// One articulation per lane, block bases handed in directly -- the index chain and
// the leaf-to-root loop are P8.88's job.  `Z`, `ZIcInt` and the running `ZA` /
// `ZAInt` are LOCALS in the CUDA, not block fields, so they travel through a
// scratch buffer: four spatial vectors in, two out, plus the returned matrix.
//
// Layout of the scratch buffer, per lane, as floats:
//   [0..5]    Z            (in)
//   [6..11]   ZIcInt       (in)
//   [12..17]  ZA           (in, then out)
//   [18..23]  ZAInt        (in, then out)
//   [24..50]  the returned SpatialMatrix (out): topLeft, topRight, bottomLeft,
//             three columns each
//   [51]      the resolved dof count computeIs was driven with (out), so a
//             fixture error in the count is visible rather than silent

struct ApiParams
{
    uint64_t linkBlocks;    // @0   one link block per lane's articulation
    uint64_t dofBlocks;     // @8   already at the joint offset
    uint64_t jointForces;   // @16  0 == NULL
    uint64_t scratch;       // @24  float[nbLanes * 52]
    uint64_t inertia;       // @32  float[nbLanes * 27] -- the link's I^A, in
    uint nbLanes;           // @40
    uint pad0;              // @44 -> 48 bytes
};

[[vk::push_constant]]
ConstantBuffer<ApiParams> gApi;

static const uint API_SCRATCH_FLOATS = 52u;
static const uint API_INERTIA_FLOATS = 27u;
static const uint API_JOINT_FORCES_PER_LANE = 3u;

AbSpatialVector apiLoadVec(Ptr<float> p, uint at)
{
    return asmSVec(float3(p[at + 0u], p[at + 1u], p[at + 2u]),
                   float3(p[at + 3u], p[at + 4u], p[at + 5u]));
}

void apiStoreVec(Ptr<float> p, uint at, AbSpatialVector v)
{
    p[at + 0u] = v.top.x;    p[at + 1u] = v.top.y;    p[at + 2u] = v.top.z;
    p[at + 3u] = v.bottom.x; p[at + 4u] = v.bottom.y; p[at + 5u] = v.bottom.z;
}

void apiStoreM33(Ptr<float> p, uint at, AbMat33 m)
{
    p[at + 0u] = m.c0.x; p[at + 1u] = m.c0.y; p[at + 2u] = m.c0.z;
    p[at + 3u] = m.c1.x; p[at + 4u] = m.c1.y; p[at + 5u] = m.c1.z;
    p[at + 6u] = m.c2.x; p[at + 7u] = m.c2.y; p[at + 8u] = m.c2.z;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void artiPropagateInertia(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gApi.nbLanes)
        return;

    // The block layout IS the lane array: one link block and one dof run cover all
    // 32 lanes, and each lane reads its own slot out of them.  The joint forces are
    // the exception -- in the engine they come from the per-articulation record, so
    // the fixture gives each lane its own three, and a lane-stride error there
    // lands on a neighbour's forces rather than on the same numbers.
    const uint64_t linkBlock = gApi.linkBlocks;
    const uint64_t dofBlocks = gApi.dofBlocks;
    const uint64_t jF = (gApi.jointForces != 0)
                            ? (gApi.jointForces + uint64_t(lane) * API_JOINT_FORCES_PER_LANE * 4)
                            : 0;

    Ptr<float> inertia = Ptr<float>(gApi.inertia + uint64_t(lane) * API_INERTIA_FLOATS * 4);
    const AbSpatialMatrix spatialInertia =
        asmMatrix(asmM33(float3(inertia[0], inertia[1], inertia[2]),
                         float3(inertia[3], inertia[4], inertia[5]),
                         float3(inertia[6], inertia[7], inertia[8])),
                  asmM33(float3(inertia[9], inertia[10], inertia[11]),
                         float3(inertia[12], inertia[13], inertia[14]),
                         float3(inertia[15], inertia[16], inertia[17])),
                  asmM33(float3(inertia[18], inertia[19], inertia[20]),
                         float3(inertia[21], inertia[22], inertia[23]),
                         float3(inertia[24], inertia[25], inertia[26])));

    Ptr<float> s = Ptr<float>(gApi.scratch + uint64_t(lane) * API_SCRATCH_FLOATS * 4);
    const AbSpatialVector Z = apiLoadVec(s, 0u);
    const AbSpatialVector ZIcInt = apiLoadVec(s, 6u);
    AbSpatialVector ZA = apiLoadVec(s, 12u);
    AbSpatialVector ZAInt = apiLoadVec(s, 18u);

    // the caller reads the dof count from the LINK block, which is also where the
    // spherical case reads it again for itself
    const uint dofs = abU8(linkBlock, PXG_AB_LINK_mDofs, lane);

    // POISON, not zero.  The CUDA leaves `msIs[ind]` uninitialised for ind >= dofs
    // and every consumer is guarded by `dofs`; seeding it with zeros here would make
    // a dropped guard harmless, which the sweep caught as a survivor.  The host
    // mirror seeds the identical pattern, so correct code cannot tell the difference
    // and incorrect code reads a number that is nothing like a spatial vector.
    AbSpatialVector Is[3];
    for (uint k = 0u; k < 3u; ++k)
    {
        const float b = 700.0f + float(lane) * 13.0f + float(k) * 3.0f;
        Is[k] = asmSVec(float3(b + 0.0f, b + 1.0f, b + 2.0f),
                        float3(b + 3.0f, b + 4.0f, b + 5.0f));
    }

    apiComputeIs(dofs, spatialInertia, dofBlocks, lane, Is);

    const AbSpatialMatrix r = apiPropagateSpatialInertiaZaZIc(linkBlock, dofBlocks, Is, false, jF,
                                                              Z, ZIcInt, ZA, ZAInt, lane);

    apiStoreVec(s, 12u, ZA);
    apiStoreVec(s, 18u, ZAInt);
    apiStoreM33(s, 24u, r.topLeft);
    apiStoreM33(s, 33u, r.topRight);
    apiStoreM33(s, 42u, r.bottomLeft);
    s[51] = float(dofs);
}

// ===========================================================================
// P8.88: computeArticulatedSpatialInertiaW (forwardDynamic2.cu:1580-1663)
// ===========================================================================
//
// The leaf-to-root sweep.  Every link from the last back to (but not including)
// the root reduces its own articulated inertia by the term the function above
// returns, translates the remainder into the parent's frame, and ACCUMULATES it —
// along with the two zero-acceleration force vectors — into the parent.  Then the
// root is inverted, or zeroed if the base is fixed.
//
// ---------------------------------------------------------------------------
// STRICTLY SEQUENTIAL, AND THE READS COME FIRST
// ---------------------------------------------------------------------------
//
// The loop is a genuine dependency chain: a link's own inertia was already
// increased by its children in earlier iterations, which is why `parents[linkID] <
// linkID` is the invariant that makes the single descending pass correct.  No
// reordering, no parallelising over links.
//
// Every parent field is loaded at the TOP of the iteration, before anything is
// written.  With a strictly sequential loop that is the same as loading at the
// store, but only by luck — a port that re-loads at the store is right for the
// wrong reason, so the order is kept.
//
// The CUDA re-loads the parent's two ZA vectors immediately AFTER storing them
// (`pSpatialZA`, `pspatialZAInt`) and never uses either result.  That is dead
// code and the port drops it rather than transcribing a load nobody reads.
//
// ---------------------------------------------------------------------------
// THE CLAMP IS PxMax, NOT max
// ---------------------------------------------------------------------------
//
// `PxMax(0.f, x)` on a float is the specialisation `intrinsics::selectMax` =
// `a > b ? a : b` (PxMath.h:82), i.e. `0 > x ? 0 : x` — which returns x itself
// (NaN, -0) whenever the comparison is false.  Slang's `max` lowers to `OpFMax`,
// whose NaN / signed-zero behaviour differs.  The engine's spelling is kept verbatim.
//
// It exists because a propagated inertia can go slightly negative on the
// principal axes through rounding, and a negative diagonal there makes the root
// inverse meaningless.

// PxMax(a, b) -- PxMath.h's FLOAT specialisation (PxMath.h:82), i.e.
// intrinsics::selectMax (PxUnixMathIntrinsics.h:122): `a > b ? a : b`.
// P9.86: NOT the generic template's `a < b ? b : a`, which for PxMax(0, x)
// returned +0 for x = -0 and 0 for x = NaN where the CUDA returns x itself.
float apiPxMax(float a, float b)
{
    return (a > b) ? a : b;
}

public void apiComputeArticulatedSpatialInertiaW(uint numLinks, uint64_t linkBlocks,
                                                 uint64_t dofBlocks, uint64_t artiBlock,
                                                 uint64_t jointForces, bool extEveryIter,
                                                 uint lane)
{
    for (uint linkID = numLinks - 1u; linkID > 0u; --linkID)
    {
        const uint64_t blk = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;

        const uint parent = abU32(blk, PXG_AB_LINK_mParents, lane);
        const uint dof = abU8(blk, PXG_AB_LINK_mDofs, lane);
        const uint jointOffset = abU32(blk, PXG_AB_LINK_mJointOffset, lane);

        const float3 rw = float3(abF32(blk, PXG_AB_LINK_mRw_x, lane),
                                 abF32(blk, PXG_AB_LINK_mRw_y, lane),
                                 abF32(blk, PXG_AB_LINK_mRw_z, lane));

        const uint64_t dofData = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;
        const uint64_t pblk = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;

        // ALL of the parent's state, read before anything is written
        const AbSpatialMatrix parentInertia =
            asmLoadSpatialMatrix(pblk, PXG_AB_LINK_mSpatialArticulatedInertia, lane);
        const AbSpatialMatrix articulatedInertia =
            asmLoadSpatialMatrix(blk, PXG_AB_LINK_mSpatialArticulatedInertia, lane);

        float3 t, b;
        abLoadSpatialVector(blk, PXG_AB_LINK_mCoriolis, lane, t, b);
        const AbSpatialVector coriolis = asmSVec(t, b);
        abLoadSpatialVector(blk, PXG_AB_LINK_mZAVector, lane, t, b);
        const AbSpatialVector spatialZA = asmSVec(t, b);
        abLoadSpatialVector(blk, PXG_AB_LINK_mZAIntVector, lane, t, b);
        const AbSpatialVector spatialZAInt = asmSVec(t, b);
        abLoadSpatialVector(pblk, PXG_AB_LINK_mZAVector, lane, t, b);
        const AbSpatialVector parentZA = asmSVec(t, b);
        abLoadSpatialVector(pblk, PXG_AB_LINK_mZAIntVector, lane, t, b);
        const AbSpatialVector parentZAInt = asmSVec(t, b);

        const AbSpatialVector Ic = asmMulVec(articulatedInertia, coriolis);
        const AbSpatialVector ZIcInt = asmSVecAdd(spatialZAInt, Ic);

        // the two inout seeds the per-dof function accumulates onto
        AbSpatialVector translatedZA = spatialZA;
        AbSpatialVector translatedZAInt = ZIcInt;

        // 5.12: never NULL here any more; the flag travels separately
        const uint64_t jF = jointForces + uint64_t(jointOffset) * 4;

        // the CUDA leaves msIs uninitialised; nothing reads an entry at or beyond
        // `dof`, and the seed here is deliberately not zero so that a dropped guard
        // inside apiComputeIs / the propagate shows up as a wrong number
        AbSpatialVector msIs[3];
        for (uint k = 0u; k < 3u; ++k)
        {
            const float p = 900.0f + float(lane) * 11.0f + float(linkID) * 7.0f + float(k) * 3.0f;
            msIs[k] = asmSVec(float3(p + 0.0f, p + 1.0f, p + 2.0f),
                              float3(p + 3.0f, p + 4.0f, p + 5.0f));
        }

        apiComputeIs(dof, articulatedInertia, dofData, lane, msIs);

        AbSpatialMatrix spatialInertiaW =
            asmSub(articulatedInertia,
                   apiPropagateSpatialInertiaZaZIc(blk, dofData, msIs, extEveryIter, jF, spatialZA, ZIcInt,
                                                   translatedZA, translatedZAInt, lane));

        // shift the children's accumulated zero-acceleration forces to the parent
        asmTranslateSpatialVectorInPlace(rw, translatedZA);
        asmTranslateSpatialVectorInPlace(rw, translatedZAInt);

        const AbSpatialVector newParentZA = asmSVecAdd(parentZA, translatedZA);
        const AbSpatialVector newParentZAInt = asmSVecAdd(parentZAInt, translatedZAInt);
        abStoreSpatialVector(pblk, PXG_AB_LINK_mZAVector, lane,
                             newParentZA.top, newParentZA.bottom);
        abStoreSpatialVector(pblk, PXG_AB_LINK_mZAIntVector, lane,
                             newParentZAInt.top, newParentZAInt.bottom);

        // and the inertia, into the parent's frame
        asmTranslateInertia(asmSkewSymmetric(float4(rw.x, rw.y, rw.z, 0.0f)), spatialInertiaW);

        // do not propagate a negative inertia about a principal axis
        spatialInertiaW.bottomLeft.c0.x = apiPxMax(0.0f, spatialInertiaW.bottomLeft.c0.x);
        spatialInertiaW.bottomLeft.c1.y = apiPxMax(0.0f, spatialInertiaW.bottomLeft.c1.y);
        spatialInertiaW.bottomLeft.c2.z = apiPxMax(0.0f, spatialInertiaW.bottomLeft.c2.z);

        asmStoreSpatialMatrix(pblk, PXG_AB_LINK_mSpatialArticulatedInertia, lane,
                              asmAdd(parentInertia, spatialInertiaW));
    }

    // the root: a fixed base has no free acceleration at all, so its inverse
    // articulated inertia is ZERO rather than the inverse of anything
    AbSpatialMatrix inertia;
    const uint flags = abU32(artiBlock, PXG_AB_mFlags, lane);
    if ((flags & PXG_ARTI_FLAG_FIX_BASE) != 0u)
        inertia = asmMatrixZero();
    else
        inertia = asmInvertInertia(
            asmLoadSpatialMatrix(linkBlocks, PXG_AB_LINK_mSpatialArticulatedInertia, lane));

    asmStoreSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane, inertia);
}

// ===========================================================================
// P8.88: computeUnconstrainedSpatialInertiaLaunch1T (forwardDynamic2.cu:1667)
// ===========================================================================
//
// The same index chain as P8.85's wrapper with THREE differences, each a mutant:
//
//  1. **Block (32, 1, 1) and TWICE the grid.**  One warp per block, not two.
//     P8.85 left `ASI_WARPS_PER_BLOCK` per-kernel for exactly this; reusing 2 here
//     halves the articulations that get processed.
//  2. **`numLinks` comes from the BLOCK** — `articulationBlocks->mNumLinks[lane]`
//     — which is the OPPOSITE of P8.85, where it comes from the record.
//  3. It passes `articulation.jointForce`, the record pointer, which the loop then
//     offsets per link by that link's `jointOffset`.

struct ApiLaunchParams
{
    uint64_t scDesc;   // @0  PxgArticulationCoreDesc*
};

[[vk::push_constant]]
ConstantBuffer<ApiLaunchParams> gApiLaunch;

// ONE warp per block -- deliberately not shared with npArtiSpatialInertia's 2
static const uint API_WARPS_PER_BLOCK = 1u;

[shader("compute")]
[numthreads(32, 1, 1)]
void computeUnconstrainedSpatialInertiaLaunch1T(uint3 groupThreadId : SV_GroupThreadID,
                                                uint3 groupId : SV_GroupID)
{
    const uint64_t d = gApiLaunch.scDesc;

    const uint lane = groupThreadId.x & 31u;
    const uint globalWarpIndex = groupId.x * API_WARPS_PER_BLOCK + groupThreadId.y;
    const uint globalThreadIndex = globalWarpIndex * 32u + groupThreadId.x;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint64_t w = uint64_t(globalWarpIndex);
    const uint64_t linkBlocks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                              + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBlocks = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                             + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;

    const uint articulationOffset = artiU32Field(d, PXG_ARTI_CORE_ARTICULATION_OFFSET);
    const uint64_t islandNodeIndices = artiPtrField(d, PXG_ARTI_CORE_ISLAND_NODE_INDICES);
    const uint2 node = (Ptr<uint2>(islandNodeIndices))[globalThreadIndex + articulationOffset];
    const uint nodeIndex = nodeIndexIndex(node);

    const uint64_t bodySim = artiPtrField(d, PXG_ARTI_CORE_BODY_SIM_POOL)
                           + uint64_t(nodeIndex) * PXG_BODY_SIM_SIZE;
    const uint articulationIndex = artiU32Field(bodySim, PXG_BODY_SIM_ARTICULATION_REMAP_ID);

    // the BLOCK's link count, not the record's
    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);

    const uint64_t a = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS)
                     + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;
    const uint64_t jointForces = artiPtrField(a, PXG_ARTI_JOINT_FORCE);

    const bool extEveryIter =
        artiU8Field(d, PXG_ARTI_CORE_EXTERNAL_FORCES_EVERY_TGS_ITER) != 0u;

    apiComputeArticulatedSpatialInertiaW(numLinks, linkBlocks, dofBlocks, artiBlock, jointForces,
                                         extEveryIter, lane);
}

// ---------------------------------------------------------------------------
// The loop's own test harness entry point
// ---------------------------------------------------------------------------
//
// One articulation per lane, block bases handed in directly, `numLinks` read from
// the articulation block exactly as the wrapper reads it.  The joint forces are
// sliced per lane so a lane-stride error there lands on a neighbour's forces.

struct ApiLoopParams
{
    uint64_t linkBlocks;    // @0
    uint64_t dofBlocks;     // @8
    uint64_t artiBlock;     // @16
    uint64_t jointForces;   // @24  0 == NULL is NOT used here; the flag below is
    uint nbLanes;           // @32
    uint extEveryIter;      // @36
    uint pad0;              // @40
    uint pad1;              // @44 -> 48 bytes
};

[[vk::push_constant]]
ConstantBuffer<ApiLoopParams> gApiLoop;

static const uint API_LOOP_JF_PER_LANE = 8u;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiArticulatedInertia(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gApiLoop.nbLanes)
        return;

    const uint numLinks = abU32(gApiLoop.artiBlock, PXG_AB_mNumLinks, lane);

    apiComputeArticulatedSpatialInertiaW(numLinks, gApiLoop.linkBlocks, gApiLoop.dofBlocks,
                                         gApiLoop.artiBlock,
                                         gApiLoop.jointForces
                                             + uint64_t(lane) * API_LOOP_JF_PER_LANE * 4,
                                         gApiLoop.extEveryIter != 0u, lane);
}
