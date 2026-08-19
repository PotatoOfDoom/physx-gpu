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

// updateArticulationsBda.slang
//
// P8.124: updateArticulationsLaunch (gpusimulationcontroller/src/CUDA/
// updateBodiesAndShapes.cu:559-907) on the ENGINE launch ABI — the per-frame
// dirty-data scatter: for every updated articulation, one warp copies the
// dirty slices of the flattened new-* pools into the arrays the persistent
// pool record points at.  Launch: grid ceil(nbSimUpdates / 8) x block
// (32, 8, 1) (NEW_ARTICULATION = 256, PxgSimulationCore.cpp:2446-2451);
// args (desc*, simUpdates*, nbSimUpdates, dofData*, directAPI) — the two
// mapped-host pointers are resolved through getMappedDevicePtr before the
// launch, so hostPtrMask stays 0.
//
// The u32-walk copy argument and the dropped stagings are newArticulationsBda
// (P8.65)'s, re-checked against THIS kernel's walks: every uint4 walk here
// covers a size that is a multiple of 16 (ArticulationJointCore 544,
// MimicJointCore 32, SpatialTendonData 16, TendonAttachmentData 32,
// FixedTendonData 32, PxgArticulationLink 80, LinkProp 16 — the CUDA's own
// size comments "12 bytes"/"74 bytes" are stale), every uint2 walk a multiple
// of 8 (TendonJointCoefficientData 16, UnAlignedSpatialVector 24), every uint
// walk a multiple of 4 — so an elementwise u32 walk writes exactly the same
// bytes and no width-truncation exists to reproduce.
//
// TWO DROPPED STAGINGS, AND THE DEAD WRITES THEY EXPOSE:
//  * The block-shared PxgArticulationSimUpdate staging (:581-599) is a pure
//    bandwidth optimisation — each warp reads only its own record, so this
//    port reads simUpdates[globalWarpIndex] directly.
//  * The 544-byte record staging (:583, :616-619) is read-only as far as the
//    OBSERVABLE state goes: the branches read pointers/counts out of it and
//    write through the POINTERS (global), while the kernel's writes into the
//    staged copy itself — `data.confiDirty = true` under eDIRTY_JOINTS (:646)
//    and the five `data.gpuDirtyFlag |=` lane-0 writes on the dof branches
//    (:838/:845/:852/:859/:866) plus the ext-accel one (:895) — target SHARED
//    MEMORY THAT IS DISCARDED when the kernel ends.  There is no copy-back;
//    the only pool writes are the eDIRTY_USER_FLAGS pair (:884-885) and the
//    final updateDirty (:905), which the CUDA spells as direct pool stores.
//    So those staged-copy writes have NO effect in the engine either, and the
//    port omits them.  ELEVENTH upstream-bug candidate: :646 and the
//    gpuDirtyFlag ORs read like they were meant to be pool writes (the
//    :884-885 pair shows the intended spelling); as written they are dead.
//
// TRAPS (each a sweep mutant unless marked):
//  1. links are copied under eDIRTY_LINKS **OR** eDIRTY_ROOT_VELOCITIES;
//     props/parents/children/body2Actors only under eDIRTY_LINKS.
//  2. body2Worlds is three-way: under eDIRTY_LINKS with directAPI it is the
//     per-link RECONCILIATION (actor2World = body2World * oldBody2Actor^-1,
//     then * newBody2Actor, only where oldBody2Actor != newBody2Actor by
//     EXACT float compare); with !directAPI a plain copy; and the else-if
//     eDIRTY_POSITIONS && !directAPI refresh runs ONLY when neither
//     eDIRTY_LINKS nor eDIRTY_ROOT_VELOCITIES fired.
//  3. The dof cursor starts at update.dofDataStartIndex and advances ONLY on
//     the dirty branches, in the fixed order positions -> velocities ->
//     forces -> targetPos -> targetVel.  NO zero-fill else branches (the
//     new-articulations kernel has them; this kernel does not).
//  4. directAPI gates the four tendon branches, the whole dof block and the
//     ext-accel branch; eDIRTY_MIMIC_JOINT and eDIRTY_WAKECOUNTER are NOT
//     gated.
//  5. eDIRTY_USER_FLAGS writes the POOL record: data.flags (u8 @26) =
//     update.userFlags (u8 @40 of the 48-byte update record) and
//     data.confiDirty (u8 @27) = 1.  In the CUDA all 32 lanes store the same
//     value (a benign same-value race); the port guards with lane 0, which is
//     observably identical and avoids a non-atomic byte RMW race.
//  6. updateDirty (u32 @12 of the pool record) = flags &
//     ~(eDIRTY_FORCES | eIN_DIRTY_LIST), written for EVERY processed update
//     (lane 0, same argument as trap 5).
//  7. The attachment / tendon-joint mod scatters walk the articulation's OWN
//     tendon records (spatialTendons[i] / fixedTendons[i]) for mModElements
//     and mNbElements, with a RUNNING source cursor `index` that accumulates
//     nbElements across tendons.

import pxgSlangCommon;
import pxgSlangArticulationDesc;

static const uint UA_WARP_SIZE = 32;
static const uint UA_WARPS_PER_BLOCK = 8;   // NEW_ARTICULATION 256 / 32

// ENGINE launch ABI: (const PxgUpdateArticulationDesc*,
// const PxgArticulationSimUpdate*, PxU32, const PxReal*, bool)
// -> {8, 8, 4, 8, 1}: the bool byte lands at 32 after the pointer realigns
// to 24; 40-byte blob.
struct UpdateArticulationsParams
{
    uint64_t scDesc;       // @0
    uint64_t simUpdates;   // @8
    uint nbSimUpdates;     // @16
    uint pad0;             // @20
    uint64_t dofData;      // @24
    uint directApiWord;    // @32 (bool byte 0; bridge zero-pads 33..39)
    uint pad1;             // @36
};

[[vk::push_constant]]
ConstantBuffer<UpdateArticulationsParams> gUpdArtiAbi;

// warpCopy<T>(dest, source, totalSize) as the elementwise u32 walk — see the
// header note on width equivalence.
void uaCopy(uint64_t dst, uint64_t src, uint byteCount, uint lane)
{
    Ptr<uint> d = Ptr<uint>(dst);
    Ptr<uint> s = Ptr<uint>(src);
    const uint n = byteCount >> 2;
    for (uint i = lane; i < n; i += UA_WARP_SIZE)
        d[i] = s[i];
}

PxgTransform28 uaLoadTransform(uint64_t addr)
{
    Ptr<float> f = Ptr<float>(addr);
    return makeTransform28(float4(f[0], f[1], f[2], f[3]), float3(f[4], f[5], f[6]));
}

void uaStoreTransform(uint64_t addr, PxgTransform28 t)
{
    Ptr<float> f = Ptr<float>(addr);
    f[0] = t.qx; f[1] = t.qy; f[2] = t.qz; f[3] = t.qw;
    f[4] = t.px; f[5] = t.py; f[6] = t.pz;
}

[shader("compute")]
[numthreads(32, 8, 1)]
void updateArticulationsLaunch(uint3 groupThreadId : SV_GroupThreadID,
                               uint3 groupId : SV_GroupID)
{
    const uint lane = groupThreadId.x;
    const uint globalWarpIndex = groupId.x * UA_WARPS_PER_BLOCK + groupThreadId.y;

    if (globalWarpIndex >= gUpdArtiAbi.nbSimUpdates)
        return;

    const PxgUpdateArticulationDescGpu scDesc =
        (Ptr<PxgUpdateArticulationDescGpu>(gUpdArtiAbi.scDesc))[0];

    // the per-update record, read directly (the shared staging is dropped)
    const uint64_t updateAddr =
        gUpdArtiAbi.simUpdates + uint64_t(globalWarpIndex) * 48u;
    const PxgArticulationSimUpdateGpu update =
        (Ptr<PxgArticulationSimUpdateGpu>(gUpdArtiAbi.simUpdates))[globalWarpIndex];

    const uint flags = update.dirtyFlags;
    const uint articIndex = update.articulationIndex;
    const bool directAPI = (gUpdArtiAbi.directApiWord & 0xFFu) != 0u;

    // the pool record: pointers and counts read straight from it (the 544-byte
    // record staging is dropped; nothing this kernel writes is later re-read)
    const uint64_t arti =
        scDesc.mArticulationPool + uint64_t(articIndex) * PXG_ARTI_STRIDE;

    const uint nbLinks = artiU16Field(arti, PXG_ARTI_DATA_NUM_LINKS);
    const uint linkStartIndex = update.linkStartIndex;

    if ((flags & PXG_ARTI_DIRTY_JOINTS) != 0u)
    {
        uaCopy(artiPtrField(arti, PXG_ARTI_JOINTS),
               scDesc.mNewJointCores + uint64_t(linkStartIndex) * PXG_ARTI_JOINT_CORE_SIZE,
               PXG_ARTI_JOINT_CORE_SIZE * nbLinks, lane);

        uaCopy(artiPtrField(arti, PXG_ARTI_JOINT_DATA),
               scDesc.mNewJointData + uint64_t(linkStartIndex) * PXG_ARTI_JOINT_CORE_DATA_SIZE,
               PXG_ARTI_JOINT_CORE_DATA_SIZE * nbLinks, lane);

        // the CUDA's `msArticulation.data.confiDirty = true` (:646) targets
        // the DISCARDED shared copy — a dead write, omitted (header note)
    }

    if ((flags & PXG_ARTI_DIRTY_MIMIC_JOINT) != 0u)
    {
        const uint nbMimicJoints = artiU16Field(arti, PXG_ARTI_DATA_NUM_MIMIC_JOINTS);
        uaCopy(artiPtrField(arti, PXG_ARTI_MIMIC_JOINT_CORES),
               scDesc.mNewArticulationMimicJointPool +
                   uint64_t(update.mimicJointStartIndex) * PXG_ARTI_MIMIC_JOINT_CORE_SIZE,
               PXG_ARTI_MIMIC_JOINT_CORE_SIZE * nbMimicJoints, lane);
    }

    // trap 4: the tendon branches exist only without the direct API
    if (!directAPI)
    {
        if ((flags & PXG_ARTI_DIRTY_SPATIAL_TENDON) != 0u)
        {
            const uint nbSpatialTendons = artiU16Field(arti, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
            uaCopy(artiPtrField(arti, PXG_ARTI_SPATIAL_TENDON_PARAMS),
                   scDesc.mNewSpatialTendonParamsPool +
                       uint64_t(update.spatialTendonStartIndex) * PXG_ARTI_SPATIAL_TENDON_DATA_SIZE,
                   PXG_ARTI_SPATIAL_TENDON_DATA_SIZE * nbSpatialTendons, lane);
        }

        if ((flags & PXG_ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT) != 0u)
        {
            const uint nbSpatialTendons = artiU16Field(arti, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
            const uint64_t tendons = artiPtrField(arti, PXG_ARTI_SPATIAL_TENDONS);
            // trap 7: a running source cursor across the articulation's tendons
            uint index = 0u;
            for (uint i = 0u; i < nbSpatialTendons; ++i)
            {
                const uint64_t tendon = tendons + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
                const uint nbElements = artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS);
                uaCopy(artiPtrField(tendon, PXG_TENDON_MOD_ELEMENTS),
                       scDesc.mNewAttachmentModPool +
                           uint64_t(update.attachmentStartIndex + index) * PXG_ARTI_ATTACHMENT_MOD_SIZE,
                       PXG_ARTI_ATTACHMENT_MOD_SIZE * nbElements, lane);
                index += nbElements;
            }
        }

        if ((flags & PXG_ARTI_DIRTY_FIXED_TENDON) != 0u)
        {
            const uint nbFixedTendons = artiU16Field(arti, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
            uaCopy(artiPtrField(arti, PXG_ARTI_FIXED_TENDON_PARAMS),
                   scDesc.mNewFixedTendonParamsPool +
                       uint64_t(update.fixedTendonStartIndex) * PXG_ARTI_FIXED_TENDON_DATA_SIZE,
                   PXG_ARTI_FIXED_TENDON_DATA_SIZE * nbFixedTendons, lane);
        }

        if ((flags & PXG_ARTI_DIRTY_FIXED_TENDON_JOINT) != 0u)
        {
            const uint nbFixedTendons = artiU16Field(arti, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
            const uint64_t tendons = artiPtrField(arti, PXG_ARTI_FIXED_TENDONS);
            uint index = 0u;
            for (uint i = 0u; i < nbFixedTendons; ++i)
            {
                const uint64_t tendon = tendons + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
                const uint nbElements = artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS);
                uaCopy(artiPtrField(tendon, PXG_TENDON_MOD_ELEMENTS),
                       scDesc.mNewTendonJointCoefficientPool +
                           uint64_t(update.tendonJointStartIndex + index) * PXG_ARTI_TENDON_JOINT_COEFF_SIZE,
                       PXG_ARTI_TENDON_JOINT_COEFF_SIZE * nbElements, lane);
                index += nbElements;
            }
        }
    }

    // trap 1: links under LINKS OR ROOT_VELOCITIES; the rest under LINKS only
    if ((flags & (PXG_ARTI_DIRTY_LINKS | PXG_ARTI_DIRTY_ROOT_VELOCITIES)) != 0u)
    {
        uaCopy(artiPtrField(arti, PXG_ARTI_LINKS),
               scDesc.mNewLinks + uint64_t(linkStartIndex) * PXG_ARTI_LINK_SIZE,
               PXG_ARTI_LINK_SIZE * nbLinks, lane);

        if ((flags & PXG_ARTI_DIRTY_LINKS) != 0u)
        {
            uaCopy(artiPtrField(arti, PXG_ARTI_LINK_PROPS),
                   scDesc.mNewLinkProps + uint64_t(linkStartIndex) * PXG_ARTI_LINK_PROP_SIZE,
                   PXG_ARTI_LINK_PROP_SIZE * nbLinks, lane);

            uaCopy(artiPtrField(arti, PXG_ARTI_PARENTS),
                   scDesc.mNewLinkParents + uint64_t(linkStartIndex) * 4u,
                   4u * nbLinks, lane);

            uaCopy(artiPtrField(arti, PXG_ARTI_CHILDREN),
                   scDesc.mNewLinkChildren + uint64_t(linkStartIndex) * PXG_ARTI_BITFIELD_SIZE,
                   PXG_ARTI_BITFIELD_SIZE * nbLinks, lane);

            if (directAPI)
            {
                // trap 2: the body2World reconciliation onto a possibly
                // changed CMassLocalPose, per link, lane-strided.
                //
                // The CUDA's __syncwarp() (:793) between this loop and the
                // body2Actor copy (:805-810) is LOAD-BEARING there: the copy
                // overwrites the very words the reconciliation reads, and the
                // barrier orders reads before writes across the warp.  A
                // workgroup barrier here would be DIVERGENT (the body guard
                // varies with threadIdx.y) and llvmpipe hangs on those — so
                // instead the body2Actor store is FUSED into this per-link
                // loop: each link is read and then written by exactly ONE
                // lane, which removes the cross-lane hazard and writes
                // byte-identical results (the CUDA's u32-strided copy covers
                // the same bytes with the same values).
                const uint64_t body2Worlds = artiPtrField(arti, PXG_ARTI_LINK_BODY2WORLDS);
                const uint64_t body2Actors = artiPtrField(arti, PXG_ARTI_LINK_BODY2ACTORS);
                for (uint linkIndex = lane; linkIndex < nbLinks; linkIndex += UA_WARP_SIZE)
                {
                    const uint64_t b2aAddr = body2Actors + uint64_t(linkIndex) * PXG_ARTI_TRANSFORM_SIZE;
                    const PxgTransform28 oldBody2Actor = uaLoadTransform(b2aAddr);
                    const PxgTransform28 newBody2Actor =
                        uaLoadTransform(scDesc.mNewLinkBody2Actors +
                                        uint64_t(linkStartIndex + linkIndex) * PXG_ARTI_TRANSFORM_SIZE);

                    // PxTransform::operator== is EXACT float equality on q and p
                    const bool equal =
                        oldBody2Actor.qx == newBody2Actor.qx && oldBody2Actor.qy == newBody2Actor.qy &&
                        oldBody2Actor.qz == newBody2Actor.qz && oldBody2Actor.qw == newBody2Actor.qw &&
                        oldBody2Actor.px == newBody2Actor.px && oldBody2Actor.py == newBody2Actor.py &&
                        oldBody2Actor.pz == newBody2Actor.pz;

                    if (!equal)
                    {
                        const uint64_t b2wAddr = body2Worlds + uint64_t(linkIndex) * PXG_ARTI_TRANSFORM_SIZE;
                        const PxgTransform28 body2World = uaLoadTransform(b2wAddr);
                        // oldBody2Actor.getInverse(): (rotateInv(q, -p), conjugate(q))
                        const float4 qc = quatConjugate(oldBody2Actor.q());
                        const PxgTransform28 inv =
                            makeTransform28(qc, quatRotateInv(oldBody2Actor.q(), -oldBody2Actor.p()));
                        const PxgTransform28 actor2World = transformTransform(body2World, inv);
                        uaStoreTransform(b2wAddr, transformTransform(actor2World, newBody2Actor));
                    }

                    // the :805-810 body2Actor copy, fused (see the note above)
                    uaStoreTransform(b2aAddr, newBody2Actor);
                }
            }
            else
            {
                uaCopy(artiPtrField(arti, PXG_ARTI_LINK_BODY2WORLDS),
                       scDesc.mNewLinkBody2Worlds + uint64_t(linkStartIndex) * PXG_ARTI_TRANSFORM_SIZE,
                       PXG_ARTI_TRANSFORM_SIZE * nbLinks, lane);

                uaCopy(artiPtrField(arti, PXG_ARTI_LINK_BODY2ACTORS),
                       scDesc.mNewLinkBody2Actors + uint64_t(linkStartIndex) * PXG_ARTI_TRANSFORM_SIZE,
                       PXG_ARTI_TRANSFORM_SIZE * nbLinks, lane);
            }
        }
    }
    // trap 2 tail: the POSITIONS pose refresh runs only when neither LINKS
    // nor ROOT_VELOCITIES fired (the else-if in the CUDA, :814)
    else if ((flags & PXG_ARTI_DIRTY_POSITIONS) != 0u && !directAPI)
    {
        uaCopy(artiPtrField(arti, PXG_ARTI_LINK_BODY2WORLDS),
               scDesc.mNewLinkBody2Worlds + uint64_t(linkStartIndex) * PXG_ARTI_TRANSFORM_SIZE,
               PXG_ARTI_TRANSFORM_SIZE * nbLinks, lane);
    }

    // trap 3: the dof cursor — dirty branches only, fixed order, no zero fill
    uint offset = update.dofDataStartIndex;
    const uint numDofs = artiU16Field(arti, PXG_ARTI_DATA_NUM_JOINT_DOFS);

    if (!directAPI)
    {
        if ((flags & PXG_ARTI_DIRTY_POSITIONS) != 0u)
        {
            uaCopy(artiPtrField(arti, PXG_ARTI_JOINT_POSITIONS),
                   gUpdArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
            offset += numDofs;
            // the lane-0 `data.gpuDirtyFlag |=` writes the DISCARDED shared
            // copy (:838) — dead, omitted; same for the four below
        }
        if ((flags & PXG_ARTI_DIRTY_VELOCITIES) != 0u)
        {
            uaCopy(artiPtrField(arti, PXG_ARTI_JOINT_VELOCITIES),
                   gUpdArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
            offset += numDofs;
        }
        if ((flags & PXG_ARTI_DIRTY_FORCES) != 0u)
        {
            uaCopy(artiPtrField(arti, PXG_ARTI_JOINT_FORCE),
                   gUpdArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
            offset += numDofs;
        }
        if ((flags & PXG_ARTI_DIRTY_JOINT_TARGET_POS) != 0u)
        {
            uaCopy(artiPtrField(arti, PXG_ARTI_JOINT_TARGET_POSITIONS),
                   gUpdArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
            offset += numDofs;
        }
        if ((flags & PXG_ARTI_DIRTY_JOINT_TARGET_VEL) != 0u)
        {
            uaCopy(artiPtrField(arti, PXG_ARTI_JOINT_TARGET_VELOCITIES),
                   gUpdArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
            offset += numDofs;
        }
    }

    // trap 4: NOT gated on directAPI
    if ((flags & PXG_ARTI_DIRTY_WAKECOUNTER) != 0u)
    {
        uaCopy(artiPtrField(arti, PXG_ARTI_LINK_WAKE_COUNTERS),
               scDesc.mNewLinkWakeCounters + uint64_t(linkStartIndex) * 4u,
               4u * nbLinks, lane);
    }

    // trap 5: the POOL byte pair, lane 0 (the CUDA's unguarded same-value
    // warp store, made single-writer)
    if ((flags & PXG_ARTI_DIRTY_USER_FLAGS) != 0u && lane == 0u)
    {
        artiStoreU8Field(arti, PXG_ARTI_DATA_FLAGS, artiU8Field(updateAddr, PXG_SIM_UPDATE_USER_FLAGS));
        artiStoreU8Field(arti, PXG_ARTI_DATA_CONFI_DIRTY, 1u);
    }

    if (!directAPI)
    {
        if ((flags & PXG_ARTI_DIRTY_EXT_ACCEL) != 0u)
        {
            uaCopy(artiPtrField(arti, PXG_ARTI_EXTERNAL_ACCELERATIONS),
                   scDesc.mNewLinkExtAccels + uint64_t(linkStartIndex) * PXG_ARTI_SPATIAL_VECTOR_SIZE,
                   PXG_ARTI_SPATIAL_VECTOR_SIZE * nbLinks, lane);
            // gpuDirtyFlag |= eDIRTY_EXT_ACCEL (:895): shared-copy dead write
        }
    }

    // trap 6: updateDirty for EVERY processed update (lane 0)
    if (lane == 0u)
        (Ptr<uint>(arti + PXG_ARTI_DATA_UPDATE_DIRTY))[0] =
            flags & ~(PXG_ARTI_DIRTY_FORCES | PXG_ARTI_IN_DIRTY_LIST);
}
