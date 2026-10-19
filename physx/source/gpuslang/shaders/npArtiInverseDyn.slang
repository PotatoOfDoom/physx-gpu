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
// EIGHTH UPSTREAM-BUG CANDIDATE, FIXED UPSTREAM IN 5.12: 5.6.1 had NO
// jobIndex guard (overhang warps read gpuIndices past the end).  5.12 adds a
// leading `const PxU32 nbIndices` argument and wraps the body in
// `if (jobIndex < nbIndices)`; the port keeps the barriers outside that guard
// (they are block-wide here) and gates every memory access on it.
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
import pxgSlangArticulationBlocks; // P8.114: AbMat33 + abMat33Zero
import pxgSlangSpatialMatrix;      // P8.114: the Dy::SpatialMatrix algebra
import pxgSlangArticulationDynamic; // P8.114: adTranslateSpatialVector
import npArtiSpatialInertia;        // P8.114: asiMat33FromQuat / asiTransformInertiaTensor

static const uint ID_WARP = 32u;

struct ComParams
{
    uint nbIndices;          // @0  (5.12) PxU32 -- the job guard
    uint pad0;               // @4
    uint64_t data;           // @8  PxVec3* (12-byte stride)
    uint64_t gpuIndices;     // @16 PxArticulationGPUIndex* (u32)
    uint rootFrameWord;      // @24 bool byte 0
    uint pad1;               // @28
    uint64_t articulations;  // @32 PxgArticulation* (496-byte records)
}

[[vk::push_constant]]
ConstantBuffer<ComParams> gCom;

[shader("compute")]
[numthreads(32, 8, 1)]
void computeArtiCOM(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 8u;
    const uint threadIndex = groupThreadId.x;

    // trap 1: remapped; 5.12 guards the job (the eighth candidate's fix)
    const bool live = jobIndex < gCom.nbIndices;
    uint64_t record = 0;
    uint linkCount = 0u;
    uint64_t linkProps = 0;
    uint64_t body2Worlds = 0;
    const uint64_t comAddr = gCom.data + uint64_t(jobIndex) * 12u;
    if (live)
    {
        const uint artiIndex = (Ptr<uint>(gCom.gpuIndices))[jobIndex];
        record = gCom.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;
        linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        linkProps = artiPtrField(record, PXG_ARTI_LINK_PROPS);
        body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);
    }

    // total mass -> data[jobIndex].x (trap 3)
    for (uint link = threadIndex; link < linkCount; link += ID_WARP)
    {
        const float mass = 1.0f / (Ptr<float>(linkProps + uint64_t(link) * 16u))[3];
        pxgAtomicAddF32(comAddr, 0u, mass);
    }

    AllMemoryBarrierWithGroupSync();   // __syncwarp, widened (see the header)

    const float totalMass = live ? (Ptr<float>(comAddr))[0] : 0.0f;

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

    if (live && threadIndex == 0u)
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
                // 5.12: computeLinkRwInvDyn (OMPE-94459), not bw.p - pbw.p
                const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));

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

// ===========================================================================
// computeLinkRwInvDyn (inverseDynamic.cu, 5.12): the precision-clean Rw of the
// RECORD-side kernels -- Dy::computeLinkRw over the joint core's anchors and,
// for a prismatic joint, motionMatrix[link][0].bottom * jointPositions[offset].
// Replaces every `body2World.p - parentBody2World.p` of 5.6.1 (OMPE-94459).
// The callers that used the NEGATED difference (parent.p - child.p) now negate
// this result, as the CUDA does (`-computeLinkRwInvDyn(...)`).
// ===========================================================================
float3 idLinkRw(uint64_t record, uint linkID, float4 parentQW, float4 childQW)
{
    const uint64_t joint = artiPtrField(record, PXG_ARTI_JOINTS) + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_SIZE;
    Ptr<float> pp = Ptr<float>(joint + PXG_AR_JOINT_parentPose + 16u);   // PxTransform: q, then p
    Ptr<float> cp = Ptr<float>(joint + PXG_AR_JOINT_childPose + 16u);
    const float3 parentOffset = float3(pp[0], pp[1], pp[2]);
    const float3 childOffset = float3(cp[0], cp[1], cp[2]);

    float3 slide = float3(0.0f, 0.0f, 0.0f);
    const bool isPrismatic = artiU8Field(joint, PXG_AR_JOINT_jointType) == 1u;   // ePRISMATIC
    if (isPrismatic)
    {
        const uint dofOffset = (Ptr<uint>(artiPtrField(record, PXG_ARTI_JOINT_DATA)
                                          + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
        const float jPos = (Ptr<float>(artiPtrField(record, PXG_ARTI_JOINT_POSITIONS)))[dofOffset];
        Ptr<float> mm = Ptr<float>(artiPtrField(record, PXG_ARTI_MOTION_MATRIX)
                                   + uint64_t(linkID) * PXG_ARTI_SUBSPACE_STRIDE);
        slide = float3(mm[3], mm[4], mm[5]) * jPos;   // column 0, bottom
    }
    return dyComputeLinkRw(parentQW, childQW, parentOffset, childOffset, slide, isPrismatic);
}

float4 idQ(Ptr<float> bw) { return float4(bw[0], bw[1], bw[2], bw[3]); }

// ===========================================================================
// P8.114: computeArtiGravityForces (inverseDynamic.cu:582-658 in 5.12).  Grid
// ceil(nbIndices/8), block (32, 8, 1), GUARDED.
//
// 5.12 removed the `rootMotion` argument and the deprecated
// eGENERALIZED_GRAVITY_FORCES path with it: what is left is the 5.6.1
// "rootMotion || fixBase" branch, now unconditional -- per-link
// zA = (-gravity) * mass, leaf-to-root accumulation through computeLinkRw,
// per-dof projection dofMotion . zA, and for a floating base the six root rows
// straight from zA[0].  rootDof = fixBase ? 0 : 6 and the output pitch is
// always (maxDofs + 6).
//
// TRAPS, each a sweep mutant unless marked:
//  1. The dof-index staging trick: the serial loop stores the LINK index into
//     gravityCompensationForces[jointOffset + lane + rootDof] as a float, and
//     the final dof loop reads it back with a float->uint cast.
//  2. rootDof = fixBase ? 0 : 6 indexes; the pitch is maxDofs + 6 always.
//  3. mass carries its zero guard.
//  6. dofMotion = motionMatrix[link][dof - jointOffset].rotate(body2World):
//     BOTH halves rotated by q, then the DUAL innerProduct against zA.
//
// The record scratch: zAForces (24-byte UnAlignedSpatialVector AoS).
// DEVIATION (the P8.112 class): the two __syncwarp become block-wide
// barriers every thread reaches; the job work sits under `live`.
// ===========================================================================

struct GfParams
{
    uint nbIndices;          // @0
    uint pad0;               // @4
    uint64_t data;           // @8
    uint64_t gpuIndices;     // @16
    uint maxDofs;            // @24
    uint pad1;               // @28 (5.6.1: the rootMotion bool, gone in 5.12)
    uint64_t articulations;  // @32
    float gx;                // @40
    float gy;                // @44
    float gz;                // @48
    uint pad2;               // @52 -> 56
}

[[vk::push_constant]]
ConstantBuffer<GfParams> gGf;

AbSpatialVector idLoadSV(uint64_t base)
{
    Ptr<float> p = Ptr<float>(base);
    return asmSVec(float3(p[0], p[1], p[2]), float3(p[3], p[4], p[5]));
}

void idStoreSV(uint64_t base, AbSpatialVector v)
{
    Ptr<float> p = Ptr<float>(base);
    p[0] = v.top.x; p[1] = v.top.y; p[2] = v.top.z;
    p[3] = v.bottom.x; p[4] = v.bottom.y; p[5] = v.bottom.z;
}

AbMat33 idLoadM33(uint64_t base)
{
    Ptr<float> p = Ptr<float>(base);
    return asmM33(float3(p[0], p[1], p[2]), float3(p[3], p[4], p[5]), float3(p[6], p[7], p[8]));
}

void idStoreM33(uint64_t base, AbMat33 m)
{
    Ptr<float> p = Ptr<float>(base);
    p[0] = m.c0.x; p[1] = m.c0.y; p[2] = m.c0.z;
    p[3] = m.c1.x; p[4] = m.c1.y; p[5] = m.c1.z;
    p[6] = m.c2.x; p[7] = m.c2.y; p[8] = m.c2.z;
}

// the Dy::SpatialMatrix AoS at stride 112: topLeft @0, topRight @36,
// bottomLeft @72
AbSpatialMatrix idLoadSM(uint64_t base)
{
    return asmMatrix(idLoadM33(base), idLoadM33(base + 36u), idLoadM33(base + 72u));
}

void idStoreSM(uint64_t base, AbSpatialMatrix m)
{
    idStoreM33(base, m.topLeft);
    idStoreM33(base + 36u, m.topRight);
    idStoreM33(base + 72u, m.bottomLeft);
}

[shader("compute")]
[numthreads(32, 8, 1)]
void computeArtiGravityForces(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 8u;
    const uint threadIndex = groupThreadId.x;
    const float3 gravity = float3(gGf.gx, gGf.gy, gGf.gz);

    const bool live = jobIndex < gGf.nbIndices;
    uint linkCount = 0u, dofCount = 0u, rootDof = 0u;
    uint64_t record = 0uL, gcf = 0uL;
    uint64_t linkProps = 0uL, body2Worlds = 0uL, jointData = 0uL, motionMatrix = 0uL;
    uint64_t zA = 0uL;
    Ptr<uint> parents;
    const float3 tGravity = -gravity;
    if (live)
    {
        const uint artiIndex = (Ptr<uint>(gGf.gpuIndices))[jobIndex];
        record = gGf.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;
        linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        dofCount = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
        const bool fixBase = (artiU8Field(record, PXG_ARTI_DATA_FLAGS) & 1u) != 0u;
        rootDof = fixBase ? 0u : 6u;   // trap 2
        gcf = gGf.data + uint64_t(jobIndex) * (gGf.maxDofs + 6u) * 4u;

        linkProps = artiPtrField(record, PXG_ARTI_LINK_PROPS);
        body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);
        jointData = artiPtrField(record, PXG_ARTI_JOINT_DATA);
        motionMatrix = artiPtrField(record, PXG_ARTI_MOTION_MATRIX);
        parents = Ptr<uint>(artiPtrField(record, PXG_ARTI_PARENTS));
        zA = artiPtrField(record, PXG_ARTI_ZA_FORCES);

        for (uint link = threadIndex; link < linkCount; link += ID_WARP)
        {
            const float invM = (Ptr<float>(linkProps + uint64_t(link) * 16u))[3];
            const float mass = invM == 0.0f ? 0.0f : (1.0f / invM);   // trap 3
            idStoreSV(zA + uint64_t(link) * 24u,
                      asmSVec(tGravity * mass, float3(0.0f, 0.0f, 0.0f)));
        }
    }
    AllMemoryBarrierWithGroupSync();   // __syncwarp

    if (live)
    {
        for (uint link = linkCount - 1u; link > 0u; --link)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint parentLink = parents[link];
            Ptr<float> pbw =
                Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);

            if (threadIndex == 0u)
            {
                const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));
                idStoreSV(zA + uint64_t(parentLink) * 24u,
                          asmSVecAdd(idLoadSV(zA + uint64_t(parentLink) * 24u),
                                     adTranslateSpatialVector(rw,
                                                              idLoadSV(zA + uint64_t(link) * 24u))));
            }
            const uint jo =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
            const uint nbDof =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1] & 0xFFu;
            if (threadIndex < nbDof)
                (Ptr<float>(gcf))[jo + threadIndex + rootDof] = float(link);   // trap 1
        }
    }
    AllMemoryBarrierWithGroupSync();   // __syncwarp

    if (live)
    {
        for (uint dof = threadIndex; dof < dofCount; dof += ID_WARP)
        {
            const uint link = uint((Ptr<float>(gcf))[dof + rootDof]);   // trap 1
            const uint jo =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const float4 q = float4(bw[0], bw[1], bw[2], bw[3]);
            Ptr<float> mm = Ptr<float>(motionMatrix + uint64_t(link) * PXG_ARTI_SUBSPACE_STRIDE
                                       + uint64_t(dof - jo) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
            // trap 6: BOTH halves rotated, then the dual product
            const AbSpatialVector dofMotion =
                asmSVec(quatRotate(q, float3(mm[0], mm[1], mm[2])),
                        quatRotate(q, float3(mm[3], mm[4], mm[5])));
            (Ptr<float>(gcf))[dof + rootDof] =
                asmInnerProduct(dofMotion, idLoadSV(zA + uint64_t(link) * 24u));
        }
        if (threadIndex == 0u && rootDof == 6u)
        {
            const AbSpatialVector z0 = idLoadSV(zA);
            Ptr<float> o = Ptr<float>(gcf);
            o[0] = z0.top.x; o[1] = z0.top.y; o[2] = z0.top.z;
            o[3] = z0.bottom.x; o[4] = z0.bottom.y; o[5] = z0.bottom.z;
        }
    }
    AllMemoryBarrierWithGroupSync();   // trailing
}

// ===========================================================================
// P8.115: computeArtiCentrifugalForces (inverseDynamic.cu:662-864 in 5.12).  Grid
// ceil(nbIndices/8), block (32, 8, 1), GUARDED.  The coriolis/centrifugal
// twin of computeArtiGravityForces: recompute link velocities and coriolis
// vectors from the joint velocities, chain the accelerations, project onto
// the dofs -- plus, for a floating base, the articulated-inertia solve.
//
// TRAPS, each a sweep mutant unless marked:
//  1. **corioliseVectors and "motionAccelerations" are the SAME pointer**
//     (the record's corioliseVectors @80) -- the engine aliases them
//     deliberately because arti.motionAccelerations holds live data it must
//     not destroy.  The aliasing is correct ONLY because the acceleration
//     chain reads slot `link` exactly once before overwriting it (ascending
//     loop) and parent reads see the already-overwritten value.  ONE
//     pointer in the port, exactly as the CUDA's two names resolve to one.
//  2. The velocity recompute clamps jVel = MIN(jointVelocity,
//     maxJointVelocity) -- an UPPER clamp only, no negative bound (the
//     engine's own quirk; jcalc's clamp is a RATIO, this is not that).
//  3. The coriolis torque: pVel.top x (pVel.top x rw) + 2 * pVel.top x lVel
//     + aVec x lVel; the force is pVel.top x aVec -- stored (force, torque)
//     = (top, bottom).
//  4. The zA init CROSSES halves: top = accel.bottom * mass, bottom =
//     vA x (I vA) + I * accel.top.
//  5. The dof-index staging is written by LANE 0 inside the serial zA loop
//     (unlike the gravity kernel's lane-strided write).
//  6. (5.12) the root acceleration stays the zero phase 1 left there -- the
//     5.6.1 !rootMotion recompute -(invertInertia() * zA[0]) was removed
//     with the rootMotion argument; rootDof = fixBase ? 0 : 6, pitch
//     maxDofs + 6.
//  7. The root tail (!fixBase) writes [0..5] AFTER the dof projection wrote
//     [6..] -- disjoint, no sync between (as the CUDA).
//  9. (5.12) every Rw is computeLinkRwInvDyn (idLinkRw), not a position
//     difference.
//  8. motionVelocities[0] is a live INPUT; links 1+ are OVERWRITTEN -- an
//     observable output of this kernel, not scratch.
//
// DEVIATION (the P8.114 class): the CUDA's four outer + three inner
// __syncwarp become SEVEN uniform block-wide barrier slots.
// ===========================================================================

struct CfParams
{
    uint nbIndices;          // @0
    uint pad0;               // @4
    uint64_t data;           // @8
    uint64_t gpuIndices;     // @16
    uint maxDofs;            // @24
    uint pad1;               // @28 (5.6.1: the rootMotion bool, gone in 5.12)
    uint64_t articulations;  // @32 -> 40
}

[[vk::push_constant]]
ConstantBuffer<CfParams> gCf;

[shader("compute")]
[numthreads(32, 8, 1)]
void computeArtiCentrifugalForces(uint3 groupThreadId : SV_GroupThreadID,
                                  uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 8u;
    const uint threadIndex = groupThreadId.x;

    const bool live = jobIndex < gCf.nbIndices;
    uint linkCount = 0u, dofCount = 0u, rootDof = 0u;
    bool fixBase = false;
    uint64_t record = 0uL;
    uint64_t gcf = 0uL, linkProps = 0uL, body2Worlds = 0uL, jointData = 0uL;
    uint64_t motionMatrix = 0uL, joints = 0uL, jointVel = 0uL;
    uint64_t inertias = 0uL, motionVel = 0uL, coriolis = 0uL, zA = 0uL;
    Ptr<uint> parents;
    if (live)
    {
        const uint artiIndex = (Ptr<uint>(gCf.gpuIndices))[jobIndex];
        record = gCf.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;
        linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        dofCount = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
        fixBase = (artiU8Field(record, PXG_ARTI_DATA_FLAGS) & 1u) != 0u;
        rootDof = fixBase ? 0u : 6u;   // 5.12: no rootMotion
        gcf = gCf.data + uint64_t(jobIndex) * (gCf.maxDofs + 6u) * 4u;

        linkProps = artiPtrField(record, PXG_ARTI_LINK_PROPS);
        body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);
        jointData = artiPtrField(record, PXG_ARTI_JOINT_DATA);
        motionMatrix = artiPtrField(record, PXG_ARTI_MOTION_MATRIX);
        joints = artiPtrField(record, PXG_ARTI_JOINTS);
        jointVel = artiPtrField(record, PXG_ARTI_JOINT_VELOCITIES);
        parents = Ptr<uint>(artiPtrField(record, PXG_ARTI_PARENTS));
        inertias = artiPtrField(record, PXG_ARTI_WORLD_SPATIAL_ARTICULATED_INERTIA);
        motionVel = artiPtrField(record, PXG_ARTI_MOTION_VELOCITIES);
        coriolis = artiPtrField(record, PXG_ARTI_CORIOLISE_VECTORS);   // trap 1: ONE pointer
        zA = artiPtrField(record, PXG_ARTI_ZA_FORCES);
    }

    // ---- S0: lane-0 velocity + coriolis recompute ----
    if (live && threadIndex == 0u)
    {
        for (uint link = 1u; link < linkCount; ++link)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint parentLink = parents[link];
            Ptr<float> pbw =
                Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
            const uint jo =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
            const uint nbDof =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1] & 0xFFu;
            const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));
            const AbSpatialVector pVel = idLoadSV(motionVel + uint64_t(parentLink) * 24u);

            AbSpatialVector vel = adTranslateSpatialVector(-rw, pVel);

            AbSpatialVector deltaV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
            const float4 q = float4(bw[0], bw[1], bw[2], bw[3]);
            for (uint dof = 0u; dof < nbDof; ++dof)
            {
                const float maxJointVelocity = asfloat(artiU32Field(
                    joints + uint64_t(link) * PXG_ARTI_JOINT_CORE_SIZE,
                    PXG_AR_JOINT_maxJointVelocity + dof * 4u));
                const float jointVelocity = (Ptr<float>(jointVel))[jo + dof];
                // trap 2: an upper clamp ONLY
                const float jVel =
                    jointVelocity < maxJointVelocity ? jointVelocity : maxJointVelocity;
                Ptr<float> mm = Ptr<float>(motionMatrix + uint64_t(link) * PXG_ARTI_SUBSPACE_STRIDE
                                           + uint64_t(dof) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
                const AbSpatialVector dofMotion =
                    asmSVec(quatRotate(q, float3(mm[0], mm[1], mm[2])),
                            quatRotate(q, float3(mm[3], mm[4], mm[5])));
                deltaV = asmSVecAdd(deltaV, asmSVecScale(dofMotion, jVel));
            }

            vel = asmSVecAdd(vel, deltaV);

            // trap 3
            const float3 aVec = deltaV.top;
            const float3 force = cross(pVel.top, aVec);
            const float3 lVel = deltaV.bottom;
            const float3 torque = cross(pVel.top, cross(pVel.top, rw))
                                + 2.0f * cross(pVel.top, lVel) + cross(aVec, lVel);

            idStoreSV(coriolis + uint64_t(link) * 24u, asmSVec(force, torque));
            idStoreSV(motionVel + uint64_t(link) * 24u, vel);   // trap 8
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 1

    // ---- S1: lane-0 acceleration chain THROUGH THE ALIAS (trap 1) ----
    if (live && threadIndex == 0u)
    {
        idStoreSV(coriolis, asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f)));
        for (uint link = 1u; link < linkCount; ++link)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint parentLink = parents[link];
            Ptr<float> pbw =
                Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
            const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));
            const AbSpatialVector pAcc =
                adTranslateSpatialVector(-rw, idLoadSV(coriolis + uint64_t(parentLink) * 24u));
            idStoreSV(coriolis + uint64_t(link) * 24u,
                      asmSVecAdd(pAcc, idLoadSV(coriolis + uint64_t(link) * 24u)));
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 2

    // ---- S2: strided zA init (trap 4) ----
    if (live)
    {
        for (uint link = threadIndex; link < linkCount; link += ID_WARP)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            Ptr<float> pp = Ptr<float>(linkProps + uint64_t(link) * 16u);
            const float mass = pp[3] == 0.0f ? 0.0f : (1.0f / pp[3]);
            const float3 localInertia = float3(pp[0] == 0.0f ? 0.0f : (1.0f / pp[0]),
                                               pp[1] == 0.0f ? 0.0f : (1.0f / pp[1]),
                                               pp[2] == 0.0f ? 0.0f : (1.0f / pp[2]));
            const AbMat33 I = asiTransformInertiaTensor(
                localInertia, asiMat33FromQuat(float4(bw[0], bw[1], bw[2], bw[3])));
            const float3 vA = idLoadSV(motionVel + uint64_t(link) * 24u).top;
            const AbSpatialVector acc = idLoadSV(coriolis + uint64_t(link) * 24u);
            const float3 IvA = I.c0 * vA.x + I.c1 * vA.y + I.c2 * vA.z;
            const float3 IaccTop = I.c0 * acc.top.x + I.c1 * acc.top.y + I.c2 * acc.top.z;
            idStoreSV(zA + uint64_t(link) * 24u,
                      asmSVec(acc.bottom * mass, cross(vA, IvA) + IaccTop));
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 3

    // ---- S3: lane-0 zA accumulation + staging (trap 5) ----
    if (live && threadIndex == 0u)
    {
        for (uint link = linkCount - 1u; link > 0u; --link)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint parentLink = parents[link];
            Ptr<float> pbw =
                Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
            const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));
            idStoreSV(zA + uint64_t(parentLink) * 24u,
                      asmSVecAdd(idLoadSV(zA + uint64_t(parentLink) * 24u),
                                 adTranslateSpatialVector(rw, idLoadSV(zA + uint64_t(link) * 24u))));

            const uint jo =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
            const uint nbDof =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1] & 0xFFu;
            for (uint dof = 0u; dof < nbDof; ++dof)
                (Ptr<float>(gcf))[jo + dof + rootDof] = float(link);
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 4

    // ---- S4..S6: the floating-base articulated-inertia solve ----
    if (live && !fixBase)
    {
        for (uint link = threadIndex; link < linkCount; link += ID_WARP)
        {
            Ptr<float> pp = Ptr<float>(linkProps + uint64_t(link) * 16u);
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const float mass = pp[3] == 0.0f ? 0.0f : (1.0f / pp[3]);
            const float3 localInertia = float3(pp[0] == 0.0f ? 0.0f : (1.0f / pp[0]),
                                               pp[1] == 0.0f ? 0.0f : (1.0f / pp[1]),
                                               pp[2] == 0.0f ? 0.0f : (1.0f / pp[2]));
            AbMat33 topRight = abMat33Zero();
            topRight.c0.x = mass; topRight.c1.y = mass; topRight.c2.z = mass;
            const AbMat33 bottomLeft = asiTransformInertiaTensor(
                localInertia, asiMat33FromQuat(float4(bw[0], bw[1], bw[2], bw[3])));
            idStoreSM(inertias + uint64_t(link) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE,
                      asmMatrix(abMat33Zero(), topRight, bottomLeft));
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 5

    if (live && !fixBase && threadIndex == 0u)
    {
        for (uint link = linkCount - 1u; link > 0u; --link)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint parentLink = parents[link];
            Ptr<float> pbw =
                Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
            const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));
            AbSpatialMatrix parentSpace =
                idLoadSM(inertias + uint64_t(link) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE);
            asmTranslateInertia(asmSkewSymmetric(float4(rw, 0.0f)), parentSpace);
            const uint64_t pAddr =
                inertias + uint64_t(parentLink) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE;
            idStoreSM(pAddr, asmAdd(idLoadSM(pAddr), parentSpace));
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 6

    if (live && !fixBase && threadIndex == 0u)
    {
        // trap 6 (5.12): the root acceleration stays the zero S1 wrote; the
        // 5.6.1 !rootMotion recompute -(invertInertia() * zA[0]) is gone
        for (uint link = 1u; link < linkCount; ++link)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint parentLink = parents[link];
            Ptr<float> pbw =
                Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
            // translateSpatialVector(-computeLinkRwInvDyn(...), ...)
            const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));
            const AbSpatialVector acc =
                adTranslateSpatialVector(-rw, idLoadSV(coriolis + uint64_t(parentLink) * 24u));
            idStoreSV(coriolis + uint64_t(link) * 24u, acc);
            const AbSpatialMatrix I =
                idLoadSM(inertias + uint64_t(link) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE);
            idStoreSV(zA + uint64_t(link) * 24u,
                      asmSVecAdd(asmMulVec(I, acc), idLoadSV(zA + uint64_t(link) * 24u)));
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 7

    // ---- S7: the dof projection + the root tail (trap 7) ----
    if (live)
    {
        for (uint dof = threadIndex; dof < dofCount; dof += ID_WARP)
        {
            const uint link = uint((Ptr<float>(gcf))[dof + rootDof]);
            const uint jo =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const float4 q = float4(bw[0], bw[1], bw[2], bw[3]);
            Ptr<float> mm = Ptr<float>(motionMatrix + uint64_t(link) * PXG_ARTI_SUBSPACE_STRIDE
                                       + uint64_t(dof - jo) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
            const AbSpatialVector dofMotion =
                asmSVec(quatRotate(q, float3(mm[0], mm[1], mm[2])),
                        quatRotate(q, float3(mm[3], mm[4], mm[5])));
            (Ptr<float>(gcf))[dof + rootDof] =
                asmInnerProduct(dofMotion, idLoadSV(zA + uint64_t(link) * 24u));
        }
        if (threadIndex == 0u && !fixBase)
        {
            const AbSpatialVector z0 = idLoadSV(zA);
            Ptr<float> o = Ptr<float>(gcf);
            o[0] = z0.top.x; o[1] = z0.top.y; o[2] = z0.top.z;
            o[3] = z0.bottom.x; o[4] = z0.bottom.y; o[5] = z0.bottom.z;
        }
    }
    AllMemoryBarrierWithGroupSync();   // trailing
}

// ===========================================================================
// P8.116: computeArtiMassMatrices (inverseDynamic.cu:340-576 in 5.12; the
// 5.12 notes: rootMotion is gone, rootDof = fixBase ? 0 : 6, the pitch is
// (maxDofs + 6)^2, every Rw is computeLinkRwInvDyn, and the !rootMotion
// tail correction of trap 6 no longer exists).
// Grid ceil(nbIndices/8), block (32, 8, 1), GUARDED.  The CRB generalized
// mass matrix of the direct API -- eGENERALIZED_MASS_MATRICES (rootMotion
// false) and eMASS_MATRICES (rootMotion true).  ABI row: kCentrifugal
// reused verbatim ({4,8,8,4,1,8} -- the same six parameters).
//
// TRAPS, each a sweep mutant unless marked:
//  1. TWO record-scratch reinterpretations: worldSpatialArticulatedInertia
//     (@168) as Dy::SpatialMatrix* at stride 112 (the P8.114 pair), and
//     worldMotionMatrix (@224) as a FLAT Cm::UnAlignedSpatialVector* indexed
//     [jointOffset + dof] at stride 24 -- NOT the [link][dof] stride-80
//     layout its own type declares.
//  2. rootDof = (rootMotion && !fixBase) ? 6 : 0 indexes rows/cols, but the
//     buffer PITCH uses bufferDof = rootMotion ? 6 : 0 -- (maxDofs +
//     bufferDof)^2 floats per job even for fixed-base jobs.  matSize =
//     dofCount + rootDof; only matSize^2 entries are zeroed, the rest of
//     the pitch stays untouched.
//  3. The init loop's struct assignment copies ALL 112 bytes -- including
//     padding = 0 at +108.  The prepass then MUTATES the padding words into
//     level numbers (PxU32& references write through), and operator+= does
//     NOT touch padding.  The padding words are OBSERVABLE scratch state.
//  4. The level-scheduled tree accumulation (#if 1, the compiled branch) is
//     replaced by the engine's own #else sequential order: descending link
//     index on lane 0.  PROOF of bit-identity: the prepass gives siblings
//     strictly ascending step values in descending index order (each child
//     bumps the parent's ps to its own s+1, so a later-processed = lower-
//     indexed sibling reads a strictly larger ps), so within any parent the
//     level-scheduled accumulation order IS descending child index; across
//     parents the targets are disjoint; a link is pushed to its parent only
//     at a step strictly after all its own children landed (s(link) >=
//     max(s(child))+1).  The __shfl_sync maxStep broadcast is dropped with
//     the level loop (DEVIATION, documented in RUNTIME_STATUS).  The
//     prepass itself is KEPT for its observable padding writes.
//  5. Per-component zero guards on mass and inertia (static links).
//  6. The floating-base root coupling (+= into [col][row] and [row][col],
//     rootMotion only) vs the !rootMotion tail correction: massMatrix -=
//     m0 . (invertInertia(I0) * m1) over the root-translated scratch
//     vectors, spelled with stride dofCount (== matSize in that branch).
//  7. The root 6x6 tail (rootMotion, lane 0) copies RAW floats out of
//     spatialInertias[0] with the linear/angular inversion index map
//     offset = (row>2) ? 18 : (col<3)*9, index = offset + colSI*3 + rowSI.
//  8. The inward walk translates the SCRATCH vectors in place hop by hop
//     (rw = child.p - parent.p, POSITIVE), so the final scratch state is
//     the root-translated I*m vectors -- observable, and exactly what the
//     !rootMotion correction then consumes.
//  9. The parent bands recompute parentWorldMotionMatrix FRESH from
//     motionMatrix (rotated) -- NOT from the scratch (which holds I*m).
//
// DOCUMENTED DEVIATION (the P8.114 class): the __syncwarp choreography
// (:368, :385, :401, the per-level :420s, :568 inside the else-branch,
// :582 trailing) becomes FOUR uniform AllMemoryBarrierWithGroupSync slots
// every thread of the block reaches; segments run under live&&cond
// predicates between them.  The trailing :582 sync is dropped (nothing
// observable follows).  The rootMotion root tail has NO engine sync before
// it and needs none: it reads only spatialInertias[0] (synced at slot 3)
// and writes massMatrix cells no other lane touches.
// ===========================================================================

struct MmParams
{
    uint nbIndices;          // @0
    uint pad0;               // @4
    uint64_t data;           // @8  float* mass matrices
    uint64_t gpuIndices;     // @16 PxArticulationGPUIndex* (u32)
    uint maxDofs;            // @24
    uint pad1;               // @28 (5.6.1: the rootMotion bool, gone in 5.12)
    uint64_t articulations;  // @32 PxgArticulation* (496-byte records)
}

[[vk::push_constant]]
ConstantBuffer<MmParams> gMm;

[shader("compute")]
[numthreads(32, 8, 1)]
void computeArtiMassMatrices(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 8u;
    const uint threadIndex = groupThreadId.x;

    const bool live = jobIndex < gMm.nbIndices;
    uint linkCount = 0u, dofCount = 0u, rootDof = 0u, matSize = 0u;
    bool fixBase = false;
    uint64_t record = 0uL, mmOut = 0uL;
    uint64_t linkProps = 0uL, body2Worlds = 0uL, jointData = 0uL, motionMatrix = 0uL;
    uint64_t inertias = 0uL, wmScratch = 0uL;
    Ptr<uint> parents;
    if (live)
    {
        const uint artiIndex = (Ptr<uint>(gMm.gpuIndices))[jobIndex];
        record = gMm.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;
        linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        dofCount = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
        fixBase = (artiU8Field(record, PXG_ARTI_DATA_FLAGS) & 1u) != 0u;
        rootDof = fixBase ? 0u : 6u;   // trap 2 (5.12: no rootMotion)
        matSize = dofCount + rootDof;
        // trap 2: the PITCH is (maxDofs + 6)^2 whatever the base
        mmOut = gMm.data + uint64_t(jobIndex) * (gMm.maxDofs + 6u)
                               * (gMm.maxDofs + 6u) * 4u;

        linkProps = artiPtrField(record, PXG_ARTI_LINK_PROPS);
        body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);
        jointData = artiPtrField(record, PXG_ARTI_JOINT_DATA);
        motionMatrix = artiPtrField(record, PXG_ARTI_MOTION_MATRIX);
        parents = Ptr<uint>(artiPtrField(record, PXG_ARTI_PARENTS));
        inertias = artiPtrField(record, PXG_ARTI_WORLD_SPATIAL_ARTICULATED_INERTIA);   // trap 1
        wmScratch = artiPtrField(record, PXG_ARTI_WORLD_MOTION_MATRIX);                // trap 1
    }

    // ---- S0: zero the matSize x matSize window ----
    if (live)
    {
        Ptr<float> m = Ptr<float>(mmOut);
        for (uint i = threadIndex; i < matSize * matSize; i += ID_WARP)
            m[i] = 0.0f;
    }
    AllMemoryBarrierWithGroupSync();   // slot 1 (:368)

    // ---- S1: per-link spatial inertias into the 112-stride scratch ----
    if (live)
    {
        for (uint link = threadIndex; link < linkCount; link += ID_WARP)
        {
            Ptr<float> pp = Ptr<float>(linkProps + uint64_t(link) * 16u);
            const float mass = pp[3] == 0.0f ? 0.0f : (1.0f / pp[3]);   // trap 5
            const float3 localInertia = float3(pp[0] == 0.0f ? 0.0f : (1.0f / pp[0]),
                                               pp[1] == 0.0f ? 0.0f : (1.0f / pp[1]),
                                               pp[2] == 0.0f ? 0.0f : (1.0f / pp[2]));
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            AbMat33 topRight = abMat33Zero();
            topRight.c0.x = mass; topRight.c1.y = mass; topRight.c2.z = mass;
            const AbMat33 bottomLeft = asiTransformInertiaTensor(
                localInertia, asiMat33FromQuat(float4(bw[0], bw[1], bw[2], bw[3])));
            const uint64_t addr =
                inertias + uint64_t(link) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE;
            idStoreSM(addr, asmMatrix(abMat33Zero(), topRight, bottomLeft));
            (Ptr<uint>(addr))[27] = 0u;   // trap 3: padding @108 is part of the copy
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 2 (:385)

    // ---- S2: lane-0 tree phase -- level prepass + sequential accumulation ----
    if (live && threadIndex == 0u)
    {
        // (a) the level prepass: the padding words are observable (trap 3)
        for (uint link = linkCount - 1u; link > 0u; --link)
        {
            Ptr<uint> s = Ptr<uint>(
                inertias + uint64_t(link) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE + 108u);
            const uint parentLink = parents[link];
            Ptr<uint> ps = Ptr<uint>(
                inertias + uint64_t(parentLink) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE + 108u);
            if (s[0] < ps[0]) s[0] = ps[0];
            ps[0] = s[0] + 1u;
        }
        // (b) the accumulation, in the engine's own #else sequential order (trap 4)
        for (uint link = linkCount - 1u; link > 0u; --link)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint parentLink = parents[link];
            Ptr<float> pbw =
                Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
            const float3 rw = idLinkRw(record, link, idQ(pbw), idQ(bw));
            AbSpatialMatrix parentSpace =
                idLoadSM(inertias + uint64_t(link) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE);
            asmTranslateInertia(asmSkewSymmetric(float4(rw, 0.0f)), parentSpace);
            const uint64_t pAddr =
                inertias + uint64_t(parentLink) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE;
            idStoreSM(pAddr, asmAdd(idLoadSM(pAddr), parentSpace));   // += leaves padding alone
        }
    }
    AllMemoryBarrierWithGroupSync();   // slot 3 (:401 + the level loop's :420s)

    // ---- S3: the per-link bands + inward walks (parallel over links) ----
    if (live)
    {
        for (uint link = threadIndex + 1u; link < linkCount; link += ID_WARP)
        {
            Ptr<float> bw = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
            const uint jo =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
            const uint nbDof =
                (Ptr<uint>(jointData + uint64_t(link) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1] & 0xFFu;
            const AbSpatialMatrix I =
                idLoadSM(inertias + uint64_t(link) * PXG_ARTI_DY_SPATIAL_MATRIX_AOS_STRIDE);

            AbSpatialVector wm[3];
            const float4 q = float4(bw[0], bw[1], bw[2], bw[3]);
            for (uint dof = 0u; dof < nbDof; ++dof)
            {
                Ptr<float> m = Ptr<float>(motionMatrix + uint64_t(link) * PXG_ARTI_SUBSPACE_STRIDE
                                          + uint64_t(dof) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
                wm[dof] = asmSVec(quatRotate(q, float3(m[0], m[1], m[2])),
                                  quatRotate(q, float3(m[3], m[4], m[5])));
                idStoreSV(wmScratch + uint64_t(jo + dof) * 24u, asmMulVec(I, wm[dof]));   // trap 1
            }

            // the diagonal block
            for (uint dof0 = 0u; dof0 < nbDof; ++dof0)
            {
                const AbSpatialVector Im0 = idLoadSV(wmScratch + uint64_t(jo + dof0) * 24u);
                const uint row = jo + dof0 + rootDof;
                for (uint dof1 = 0u; dof1 < nbDof; ++dof1)
                    (Ptr<float>(mmOut))[row * matSize + (jo + dof1 + rootDof)] =
                        asmInnerProduct(wm[dof1], Im0);
            }

            // the inward walk (trap 8); 5.12 tracks the CHILD LINK for
            // computeLinkRwInvDyn alongside the child pose
            uint childLink = link;
            float4 childQ = q;
            uint parentLink = parents[link];
            for (;;)
            {
                Ptr<float> pbw =
                    Ptr<float>(body2Worlds + uint64_t(parentLink) * PXG_ARTI_TRANSFORM_SIZE);
                const float3 rw = idLinkRw(record, childLink, idQ(pbw), childQ);
                for (uint dof = 0u; dof < nbDof; ++dof)
                    idStoreSV(wmScratch + uint64_t(jo + dof) * 24u,
                              adTranslateSpatialVector(
                                  rw, idLoadSV(wmScratch + uint64_t(jo + dof) * 24u)));
                childLink = parentLink;
                childQ = idQ(pbw);

                if (parentLink == 0u)
                {
                    if (!fixBase)   // trap 6: the root coupling (5.12: no rootMotion)
                    {
                        for (uint dof0 = 0u; dof0 < nbDof; ++dof0)
                        {
                            const AbSpatialVector Im0 =
                                idLoadSV(wmScratch + uint64_t(jo + dof0) * 24u);
                            const uint col = jo + dof0 + rootDof;
                            for (uint row = 0u; row < 6u; ++row)
                            {
                                const float v = row < 3u ? Im0.top[row] : Im0.bottom[row - 3u];
                                (Ptr<float>(mmOut))[col * matSize + row] += v;
                                (Ptr<float>(mmOut))[col + row * matSize] += v;
                            }
                        }
                    }
                    break;
                }

                const uint pjo =
                    (Ptr<uint>(jointData + uint64_t(parentLink) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[0];
                const uint pnbDof =
                    (Ptr<uint>(jointData + uint64_t(parentLink) * PXG_ARTI_JOINT_CORE_DATA_SIZE))[1]
                    & 0xFFu;
                const float4 pq = float4(pbw[0], pbw[1], pbw[2], pbw[3]);
                AbSpatialVector pwm[3];
                for (uint dof = 0u; dof < pnbDof; ++dof)   // trap 9: fresh, NOT the scratch
                {
                    Ptr<float> m =
                        Ptr<float>(motionMatrix + uint64_t(parentLink) * PXG_ARTI_SUBSPACE_STRIDE
                                   + uint64_t(dof) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
                    pwm[dof] = asmSVec(quatRotate(pq, float3(m[0], m[1], m[2])),
                                       quatRotate(pq, float3(m[3], m[4], m[5])));
                }
                for (uint dof0 = 0u; dof0 < nbDof; ++dof0)
                {
                    const AbSpatialVector Im0 = idLoadSV(wmScratch + uint64_t(jo + dof0) * 24u);
                    const uint row = jo + dof0 + rootDof;
                    for (uint dof1 = 0u; dof1 < pnbDof; ++dof1)
                    {
                        const uint col = pjo + dof1 + rootDof;
                        const float v = asmInnerProduct(pwm[dof1], Im0);
                        (Ptr<float>(mmOut))[row * matSize + col] = v;
                        (Ptr<float>(mmOut))[row + col * matSize] = v;
                    }
                }
                parentLink = parents[parentLink];
            }
        }

        // the root tail (:542) -- no engine sync before it (see header)
        if (threadIndex == 0u && !fixBase)
        {
            Ptr<float> rootI = Ptr<float>(inertias);   // raw floats of spatialInertias[0]
            for (uint row = 0u; row < 6u; ++row)
            {
                const uint rowSI = (row < 3u) ? row : row - 3u;
                for (uint col = row; col < 6u; ++col)
                {
                    const uint offset = (row > 2u) ? 18u : ((col < 3u) ? 9u : 0u);   // trap 7
                    const uint colSI = (col < 3u) ? col : col - 3u;
                    const float v = rootI[offset + colSI * 3u + rowSI];
                    (Ptr<float>(mmOut))[row * matSize + col] = v;
                    (Ptr<float>(mmOut))[col * matSize + row] = v;
                }
            }
        }
    }
    AllMemoryBarrierWithGroupSync();   // trailing (:576)
    // (5.12 removed the 5.6.1 !rootMotion free-base correction
    //  massMatrix -= m0 . (invertInertia(I0) * m1) that followed here)
}
