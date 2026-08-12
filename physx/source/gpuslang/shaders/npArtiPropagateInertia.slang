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
// `jF` is a POINTER in the CUDA and NULL when
// `isExternalForcesEveryTgsIterationEnabled` is set, in which case the joint force
// reads as zero.  It is carried here as a device address where **0 means NULL**,
// which is the natural port and the reason the null test is a real branch rather
// than a flag.
public AbSpatialMatrix apiPropagateSpatialInertiaZaZIc(uint64_t linkBlock, uint64_t dofBlocks,
                                                       AbSpatialVector msIs[3], uint64_t jF,
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
        const float qstZIcInternal = ((jF != 0) ? (Ptr<float>(jF))[0] : 0.0f) - stZInt;

        abStoreF32(dof0, PXG_AB_DOF_mQstZ, lane, qstZ);
        abStoreF32(dof0, PXG_AB_DOF_mQstZIcInternal, lane, qstZIcInternal);

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
                const float qstZicInt = ((jF != 0) ? (Ptr<float>(jF))[ind2] : 0.0f) - stZInt;

                qstZG[ind2] = qstZ;
                qstZIcIntG[ind2] = qstZicInt;
                abStoreF32(dof, PXG_AB_DOF_mQstZ, lane, qstZ);
                abStoreF32(dof, PXG_AB_DOF_mQstZIcInternal, lane, qstZicInt);
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

        // literal zeros, exactly as the CUDA spells them.  The header records why
        // there is no workaround here: the fold is the driver's and a runtime-zero
        // seed changes nothing, measured.
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

    const AbSpatialMatrix r = apiPropagateSpatialInertiaZaZIc(linkBlock, dofBlocks, Is, jF,
                                                              Z, ZIcInt, ZA, ZAInt, lane);

    apiStoreVec(s, 12u, ZA);
    apiStoreVec(s, 18u, ZAInt);
    apiStoreM33(s, 24u, r.topLeft);
    apiStoreM33(s, 33u, r.topRight);
    apiStoreM33(s, 42u, r.bottomLeft);
    s[51] = float(dofs);
}
