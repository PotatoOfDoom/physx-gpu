// npArtiInverseDyn.slang
//
// P8.112: the direct-GPU-API inverse-dynamics family (inverseDynamic.cu),
// starting with `computeArtiCOM` (:1019-1075).  These kernels are RECORD-side
// -- one warp per articulation over the PxgArticulation pool, no block SoA --
// launched only through PxDirectGPUAPI::computeArticulationData, never by a
// scene.
//
// computeArtiCOM: grid ceil(nbElements/8), block (32, 8, 1) -- one warp per
// job, jobIndex = threadIdx.y + blockIdx.x * 8.  Sums link masses and
// mass-weighted link positions into data[jobIndex] via GLOBAL ATOMIC float
// adds, then lane 0 normalises and optionally moves the result into the root
// frame.
//
// TRAPS, each a sweep mutant unless marked:
//  1. gpuIndices REMAPS: the job index is not the articulation index.
//  2. The link loops are warp-strided (link = lane; link += 32) -- an
//     articulation with more than 32 links gives some lanes two links.
//  3. The total mass is accumulated INTO data[jobIndex].x and read back
//     (totalMass), then subtracted out in the tail before the divide -- the
//     caller contract is a ZEROED output buffer (a non-zero x would pollute
//     the DIVISOR, not just the sum).
//  4. mass = 1 / linkProps[link].invInertiaXYZ_invMass.w -- the .w lane of a
//     16-byte record, and linkBody2Worlds is a 28-byte PxTransform array
//     with p at offset 16 (q first).
//  5. The divide is PxVec3::operator/=: ONE reciprocal, three multiplies
//     (the P8.102b recip-multiply spelling).
//  6. The root-frame move is PxTransform::getInverse().transform(v):
//     p' = quatRotateInv(q, -p), q' = conjugate(q), result =
//     quatRotate(q', v) + p' -- two rotates and an add, NOT one rotate of
//     (v - p).
//
// EIGHTH UPSTREAM-BUG CANDIDATE (transcribed verbatim): there is NO
// jobIndex guard -- a launch whose nbElements is not a multiple of 8 has
// overhang warps that read gpuIndices past the end and dereference whatever
// articulation that garbage names.  The fixture keeps nbElements an exact
// multiple.
//
// DOCUMENTED DEVIATION: the CUDA has no sync between the position-sum loop
// and the lane-0 tail (it relies on warp-synchronous visibility of the
// red.global adds).  On our subgroup-8 emulation a warp spans four
// subgroups, so the port inserts AllMemoryBarrierWithGroupSync() there --
// value-preserving (the sums are complete either way in the engine's
// intended semantics), but a REAL ordering the CUDA text does not spell.
// The first __syncwarp (before the totalMass read) becomes the same
// block-wide barrier for the same reason; both barrier points are uniform
// across the block, so the stronger scope is safe.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangAtomicFloat;

static const uint ID_WARP = 32u;

struct ComParams
{
    uint64_t data;           // @0  PxVec3* (12-byte stride)
    uint64_t gpuIndices;     // @8  PxArticulationGPUIndex* (u32)
    uint rootFrameWord;      // @16 bool byte 0
    uint pad0;               // @20
    uint64_t articulations;  // @24 PxgArticulation* (544-byte records)
}

[[vk::push_constant]]
ConstantBuffer<ComParams> gCom;

[shader("compute")]
[numthreads(32, 8, 1)]
void computeArtiCOM(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 8u;
    const uint threadIndex = groupThreadId.x;

    // trap 1 / eighth candidate: unguarded, remapped
    const uint artiIndex = (Ptr<uint>(gCom.gpuIndices))[jobIndex];
    const uint64_t record = gCom.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;

    const uint64_t comAddr = gCom.data + uint64_t(jobIndex) * 12u;
    const uint linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
    const uint64_t linkProps = artiPtrField(record, PXG_ARTI_LINK_PROPS);
    const uint64_t body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);

    // total mass -> data[jobIndex].x (trap 3)
    for (uint link = threadIndex; link < linkCount; link += ID_WARP)
    {
        const float mass = 1.0f / (Ptr<float>(linkProps + uint64_t(link) * 16u))[3];
        pxgAtomicAddF32(comAddr, 0u, mass);
    }

    AllMemoryBarrierWithGroupSync();   // __syncwarp, widened (see the header)

    const float totalMass = (Ptr<float>(comAddr))[0];

    // mass-weighted position sums (trap 2 / trap 4)
    for (uint link = threadIndex; link < linkCount; link += ID_WARP)
    {
        Ptr<float> t = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
        const float mass = 1.0f / (Ptr<float>(linkProps + uint64_t(link) * 16u))[3];
        pxgAtomicAddF32(comAddr, 0u, t[4] * mass);
        pxgAtomicAddF32(comAddr, 1u, t[5] * mass);
        pxgAtomicAddF32(comAddr, 2u, t[6] * mass);
    }

    // the DEVIATION: the CUDA text has no barrier here
    AllMemoryBarrierWithGroupSync();

    if (threadIndex == 0u)
    {
        Ptr<float> com = Ptr<float>(comAddr);
        float x = com[0] - totalMass;   // remove the mass sum again (trap 3)
        float y = com[1];
        float z = com[2];

        // trap 5: PxVec3::operator/= -- one reciprocal, three multiplies
        const float inv = 1.0f / totalMass;
        x *= inv;
        y *= inv;
        z *= inv;

        if ((gCom.rootFrameWord & 0xFFu) != 0u)
        {
            // trap 6: getInverse() then transform(), the engine's own two-step
            Ptr<float> rt = Ptr<float>(body2Worlds);
            const float4 q = float4(rt[0], rt[1], rt[2], rt[3]);
            const float3 p = float3(rt[4], rt[5], rt[6]);
            const float3 invP = quatRotateInv(q, -p);
            const float4 invQ = float4(-q.x, -q.y, -q.z, q.w);
            const float3 v = quatRotate(invQ, float3(x, y, z)) + invP;
            x = v.x;
            y = v.y;
            z = v.z;
        }

        com[0] = x;
        com[1] = y;
        com[2] = z;
    }

    AllMemoryBarrierWithGroupSync();   // the trailing __syncwarp
}

// ===========================================================================
// P8.113a: computeArtiDenseJacobians (inverseDynamic.cu:230-318).  Grid
// ceil(nbIndices/16), block (32, 16, 1) -- one warp per job, GUARDED
// (unlike computeArtiCOM).  Fills the dense jacobian for each articulation:
// a 6x(colCount) identity block for a floating base, then one 6-row band
// per link, column-parallel across the warp.
//
// TRAPS, each a sweep mutant unless marked:
//  1. colCount = baseDofs + dofCount with baseDofs = fixedBase ? 0 : 6, but
//     the OUTPUT PITCH is maxCols * maxRows = (6 + maxDofs) *
//     (6 + (maxLinks-1)*6) -- the max sizes, not the actual counts.
//  2. Row indexing is row * colCount + dof -- the ACTUAL column count.
//  3. The parent band: dof < parentDofs reads the parent's six rows at the
//     SAME column and combines lin += ang x rw; parentDofs for a root child
//     is baseDofs, and parentRow0 for a root parent is 0 (the identity
//     block), else (parentLink-1)*6 + baseDofs.
//  4. The own-joint band: linkOffset = jointData[link].jointOffset +
//     baseDofs (u32 at +0 of the 8-byte record), linkDofs =
//     jointData[link].nbDof (a U8 at +4 -- the P8.71 width lesson);
//     motionMatrix[link][dof - linkOffset] at link*80 + dof*24, rotated by
//     body2World.q (quatRotate); columns in NEITHER band stay zero.
//  5. The link loop is SEQUENTIAL (parent rows must exist before children
//     read them); the column loop is warp-strided.
//
// DATA-FLOW NOTE (deviation, same class as computeArtiCOM): within the link
// loop every read is lane-local (a column's parent rows were written by the
// SAME lane in an earlier iteration), so the CUDA's in-loop __syncwarp is
// dropped as a lane-local no-op (the npArtiSolveSelf trap-10 precedent).
// The identity block -> link loop boundary IS cross-lane (element i is
// written by lane i%32 but read by lane dof%32), so the __syncwarp there
// becomes AllMemoryBarrierWithGroupSync -- HOISTED OUT of the !fixedBase
// branch (and the job guard) so every thread in the block reaches it.
// ===========================================================================

struct DjParams
{
    uint64_t data;           // @0  float*
    uint64_t gpuIndices;     // @8
    uint nbIndices;          // @16
    uint maxLinks;           // @20
    uint maxDofs;            // @24
    uint pad0;               // @28
    uint64_t articulations;  // @32 -> 40
}

[[vk::push_constant]]
ConstantBuffer<DjParams> gDj;

[shader("compute")]
[numthreads(32, 16, 1)]
void computeArtiDenseJacobians(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 16u;
    const uint threadIndex = groupThreadId.x;

    const bool live = jobIndex < gDj.nbIndices;

    uint64_t record = 0uL;
    uint linkCount = 0u, colCount = 0u, baseDofs = 0u;
    uint64_t jacobian = 0uL;
    bool fixedBase = true;
    if (live)
    {
        const uint artiIndex = (Ptr<uint>(gDj.gpuIndices))[jobIndex];
        record = gDj.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;
        linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        const uint dofCount = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
        fixedBase = (artiU8Field(record, PXG_ARTI_DATA_FLAGS) & 1u) != 0u;
        baseDofs = fixedBase ? 0u : 6u;
        colCount = baseDofs + dofCount;

        const uint maxCols = 6u + gDj.maxDofs;
        const uint maxRows = 6u + (gDj.maxLinks - 1u) * 6u;   // trap 1
        jacobian = gDj.data + uint64_t(jobIndex) * maxCols * maxRows * 4u;

        if (!fixedBase)
        {
            for (uint i = threadIndex; i < 6u * colCount; i += ID_WARP)
                (Ptr<float>(jacobian))[i] = (i / colCount == i % colCount) ? 1.0f : 0.0f;
        }
    }

    // the identity block is read cross-lane below -- hoisted, block-uniform
    AllMemoryBarrierWithGroupSync();

    if (!live)
        return;

    const uint64_t jointData = artiPtrField(record, PXG_ARTI_JOINT_DATA);
    const uint64_t body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);
    Ptr<uint> parents = Ptr<uint>(artiPtrField(record, PXG_ARTI_PARENTS));
    const uint64_t motionMatrix = artiPtrField(record, PXG_ARTI_MOTION_MATRIX);
    Ptr<float> jac = Ptr<float>(jacobian);

    for (uint link = 1u; link < linkCount; ++link)   // trap 5: sequential
    {
        // trap 4: jointOffset u32 @0, nbDof U8 @4 of the 8-byte record
        const uint linkOffset =
            (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0] + baseDofs;
        const uint linkDofs =
            (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1] & 0xFFu;
        const uint linkRow0 = (link - 1u) * 6u + baseDofs;
        Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
        const uint parentLink = parents[link];
        const uint parentDofs = parentLink != 0u
            ? (Ptr<uint>(jointData + uint64_t(parentLink) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0]
                  + ((Ptr<uint>(jointData + uint64_t(parentLink) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1]
                     & 0xFFu)
                  + baseDofs
            : baseDofs;

        for (uint dof = threadIndex; dof < colCount; dof += ID_WARP)
        {
            float3 linkLin = float3(0.0f, 0.0f, 0.0f);
            float3 linkAng = float3(0.0f, 0.0f, 0.0f);

            if (dof < parentDofs)   // trap 3
            {
                const uint parentRow0 = parentLink != 0u ? (parentLink - 1u) * 6u + baseDofs : 0u;
                Ptr<float> pbw =
                    Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
                const float3 rw =
                    float3(bw[4] - pbw[4], bw[5] - pbw[5], bw[6] - pbw[6]);

                const float3 parentLin = float3(jac[(parentRow0 + 0u) * colCount + dof],
                                                jac[(parentRow0 + 1u) * colCount + dof],
                                                jac[(parentRow0 + 2u) * colCount + dof]);
                const float3 parentAng = float3(jac[(parentRow0 + 3u) * colCount + dof],
                                                jac[(parentRow0 + 4u) * colCount + dof],
                                                jac[(parentRow0 + 5u) * colCount + dof]);
                const float3 parentAngxRw = cross(parentAng, rw);
                linkLin = parentLin + parentAngxRw;
                linkAng = parentAng;
            }
            else if (dof >= linkOffset && dof < linkOffset + linkDofs)   // trap 4
            {
                Ptr<float> mm = Ptr<float>(motionMatrix + uint64_t(link) * PXG_ARTI_SUBSPACE_STRIDE
                                           + uint64_t(dof - linkOffset)
                                                 * PXG_ARTI_SPATIAL_VECTOR_SIZE);
                const float4 q = float4(bw[0], bw[1], bw[2], bw[3]);
                linkLin = quatRotate(q, float3(mm[3], mm[4], mm[5]));   // v.bottom
                linkAng = quatRotate(q, float3(mm[0], mm[1], mm[2]));   // v.top
            }

            // the CUDA's in-loop __syncwarp sits here -- lane-local flow,
            // dropped (see the header note)

            jac[(linkRow0 + 0u) * colCount + dof] = linkLin.x;   // trap 2
            jac[(linkRow0 + 1u) * colCount + dof] = linkLin.y;
            jac[(linkRow0 + 2u) * colCount + dof] = linkLin.z;
            jac[(linkRow0 + 3u) * colCount + dof] = linkAng.x;
            jac[(linkRow0 + 4u) * colCount + dof] = linkAng.y;
            jac[(linkRow0 + 5u) * colCount + dof] = linkAng.z;
        }
    }
}

// ===========================================================================
// P8.113b: computeArtiCentroidalMomentumMatrices (inverseDynamic.cu:1079).
// Grid ceil(nbIndices/8), block (32, 8, 1), GUARDED, floating-base only.
// Combines the user-precomputed mass matrices and coriolis forces (both in
// the SAME data buffer, at documented offsets) into the centroidal momentum
// matrix and bias.
//
// TRAPS, each a sweep mutant unless marked:
//  1. The four buffer regions are laid out with MAX sizes: massMatrices at
//     job*(maxDofs+6)^2, coriolis at (maxDofs+6)^2*nb + job*(maxDofs+6),
//     the momentum matrix after ALL coriolis rows, the bias after ALL
//     matrix rows -- but every ELEMENT index inside uses the ACTUAL
//     (dofCount + 6).
//  2. pg = (m[2*(d+6)+4], m[5], m[(d+6)+3]) / totalMass with totalMass =
//     m[2*(d+6)+2] -- three scattered elements of the root inertia block,
//     and the divide is PxVec3::operator/ (recip-multiply).
//  3. The skew matrix is column-constructed: col0 = (0, pg.z, -pg.y),
//     col1 = (-pg.z, 0, pg.x), col2 = (pg.y, -pg.x, 0); mat * v is the
//     column linear combination (PxMat33::operator*).
//  4. Rows 3..5 ACCUMULATE (+=) onto the copied rows; rows 0..2 stay the
//     plain copy.  The lane-0 tail does the same for the 6-entry bias.
//  5. A fixed-base job writes NOTHING.
//
// DEVIATION (same class as computeArtiCOM): the CUDA has NO __syncwarp
// between the copy loop and the += loop, but element j of the copy is
// written by lane j%32 while the += at col + row*(d+6) runs on lane col%32
// -- a cross-lane RMW.  The port inserts a hoisted, block-uniform
// AllMemoryBarrierWithGroupSync between the two.
// ===========================================================================

struct CmmParams
{
    uint64_t data;           // @0  float*
    uint64_t gpuIndices;     // @8
    uint nbIndices;          // @16
    uint maxDofs;            // @20
    uint64_t articulations;  // @24 -> 32
}

[[vk::push_constant]]
ConstantBuffer<CmmParams> gCmm;

[shader("compute")]
[numthreads(32, 8, 1)]
void computeArtiCentroidalMomentumMatrices(uint3 groupThreadId : SV_GroupThreadID,
                                           uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 8u;
    const uint threadIndex = groupThreadId.x;

    bool live = jobIndex < gCmm.nbIndices;
    uint dofCount = 0u;
    uint64_t massMatrices = 0uL, coriolisForces = 0uL, cmMatrix = 0uL, cmBias = 0uL;
    if (live)
    {
        const uint artiIndex = (Ptr<uint>(gCmm.gpuIndices))[jobIndex];
        const uint64_t record = gCmm.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;
        dofCount = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
        const bool fixBase = (artiU8Field(record, PXG_ARTI_DATA_FLAGS) & 1u) != 0u;
        live = !fixBase;   // trap 5

        const uint mc = gCmm.maxDofs + 6u;
        const uint nb = gCmm.nbIndices;
        const uint startCoriolis = mc * mc * nb;                       // trap 1
        const uint startMatrix = startCoriolis + mc * nb;
        const uint startBias = startMatrix + 6u * mc * nb;
        massMatrices = gCmm.data + uint64_t(jobIndex) * mc * mc * 4u;
        coriolisForces = gCmm.data + uint64_t(startCoriolis + jobIndex * mc) * 4u;
        cmMatrix = gCmm.data + uint64_t(startMatrix + jobIndex * 6u * mc) * 4u;
        cmBias = gCmm.data + uint64_t(startBias + jobIndex * 6u) * 4u;
    }

    if (live)
    {
        Ptr<float> mm = Ptr<float>(massMatrices);
        Ptr<float> cm = Ptr<float>(cmMatrix);
        for (uint j = threadIndex; j < 6u * (dofCount + 6u); j += ID_WARP)
            cm[j] = mm[j];
    }

    // the DEVIATION: the CUDA text has no barrier here (cross-lane RMW below)
    AllMemoryBarrierWithGroupSync();

    if (!live)
        return;

    Ptr<float> mm = Ptr<float>(massMatrices);
    Ptr<float> cm = Ptr<float>(cmMatrix);
    const uint d6 = dofCount + 6u;

    // trap 2
    const float totalMass = mm[2u * d6 + 2u];
    const float invM = 1.0f / totalMass;
    const float3 pg = float3(mm[2u * d6 + 4u] * invM, mm[5] * invM, mm[d6 + 3u] * invM);
    // trap 3: columns of the skew matrix
    const float3 c0 = float3(0.0f, pg.z, -pg.y);
    const float3 c1 = float3(-pg.z, 0.0f, pg.x);
    const float3 c2 = float3(pg.y, -pg.x, 0.0f);

    for (uint col = threadIndex; col < d6; col += ID_WARP)
    {
        const float3 lin = float3(mm[0u * d6 + col], mm[1u * d6 + col], mm[2u * d6 + col]);
        const float3 cross3 = c0 * lin.x + c1 * lin.y + c2 * lin.z;   // PxMat33::operator*
        cm[col + 3u * d6] += cross3.x;   // trap 4: += onto the copy
        cm[col + 4u * d6] += cross3.y;
        cm[col + 5u * d6] += cross3.z;
    }

    if (threadIndex == 0u)
    {
        Ptr<float> cf = Ptr<float>(coriolisForces);
        Ptr<float> cb = Ptr<float>(cmBias);
        for (uint row = 0u; row < 6u; ++row)
            cb[row] = cf[row];
        const float3 lin = float3(cf[0], cf[1], cf[2]);
        const float3 cross3 = c0 * lin.x + c1 * lin.y + c2 * lin.z;
        cb[3] += cross3.x;
        cb[4] += cross3.y;
        cb[5] += cross3.z;
    }
}
