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

// newArticulationsBda.slang
//
// newArticulationsLaunch (gpusimulationcontroller/src/CUDA/
// updateBodiesAndShapes.cu:228) on the ENGINE launch ABI.
//
// THIS IS THE KERNEL THAT MAKES AN ARTICULATION EXIST ON THE DEVICE.  The host
// builds one PxgArticulation record per new articulation — 544 bytes of device
// pointers into per-articulation buffers it has just allocated
// (PxgSimulationCore.cpp:884-933) — and stages the records in
// mNewArticulationBuffer.  This kernel copies each record into the persistent
// mArticulationPool at its own `data.index`, then fills every array those
// pointers name from the flattened new-link / new-dof / new-tendon pools.
//
// Until it ran, mArticulationPool held uninitialised memory.  P8.64 traced the
// articulation scene's segfault to exactly that: the articulation branch of
// updateTransformCacheAndBoundArrayLaunch loads `linkBody2Worlds` out of the
// pool and dereferences it, so an unfilled pool is a wild pointer.  Nothing
// downstream of the pool can work before this kernel does.
//
// ===========================================================================
// WHY EVERY COPY HERE IS A u32 COPY, AND WHY THAT IS BIT-EXACT
// ===========================================================================
//
// The CUDA walks each array with `warpCopy<T>` at a T chosen for coalescing:
// uint4 for the 16-byte-aligned blocks, uint2 for the spatial vectors, uint
// for the 28-byte transforms.  warpCopy (gpucommon/src/CUDA/copy.cuh:38) is
//
//     for (i = laneId; i < totalSize/sizeof(T); i += WARP_SIZE) dest[i] = source[i];
//
// — an ELEMENTWISE copy over a destination that never overlaps its source, so
// the bytes that end up in memory do not depend on the element width used to
// walk them.  Every `totalSize` in this kernel is a multiple of 4, so a u32
// walk covers exactly the same byte range.  The value-fill overload is used
// only with ZERO here (uint 0, uint2(0,0), uint4(0,0,0,0)), and zero has the
// same bytes at every width.
//
// So the width is a performance choice on CUDA and nothing else.  Dropping it
// also drops warpCopy's 16-byte alignment assertions, which the u32 form does
// not need — strictly more permissive, never less correct.
//
// ===========================================================================
// WHY THE SHARED-MEMORY STAGING IS GONE, AND WHY IT HAD TO GO
// ===========================================================================
//
// The CUDA stages the 544-byte record in shared memory first, `__syncwarp()`s,
// copies shared -> pool, and then reads the record's pointer fields from the
// SHARED copy for all the array copies.  The source record is never written by
// this kernel, so reading those pointer fields straight from the source is the
// same values — the staging is a pure bandwidth optimisation.
//
// Keeping it would have been actively unsafe.  The workgroup is (32, 8, 1) and
// the whole body sits under `globalWarpIndex < nbArticulations`, a guard that
// varies with threadIdx.y — so it is uniform per virtual warp but NOT uniform
// across the workgroup.  A GroupMemoryBarrierWithGroupSync() inside it is a
// divergent barrier, and llvmpipe HANGS on those rather than reporting an
// error.  There is no cross-lane communication anywhere in this kernel (every
// copy is an independent strided elementwise write), so no barrier is needed
// at all, and dropping the staging removes the only reason one would appear.
//
// ===========================================================================
// A FAITHFULLY TRANSCRIBED ENGINE QUIRK
// ===========================================================================
//
// `PxgArticulationData::numJointDofs` is a PxU16, and the CUDA reads it TWICE
// with different widths: `const PxU8 dofs = ...numJointDofs` sizes the
// zero-fills, while `const PxU32 numDofs = ...numJointDofs` sizes the copies
// from dofData.  For an articulation with more than 255 dofs (64 links x up to
// 6 dofs each is reachable) those disagree, and the else-branches would zero
// only `dofs & 0xFF` entries.  Transcribed exactly as written, truncation and
// all: this port reproduces the engine, it does not fix it.

import pxgSlangArticulationDesc;

// PxgSimulationCoreKernelBlockDim::NEW_ARTICULATION is 256, and the host
// launches (32, numWarpsPerBlock, 1) threads with
// numWarpsPerBlock = NEW_ARTICULATION / WARP_SIZE (PxgSimulationCore.cpp:2387).
static const uint NA_WARP_SIZE = 32;
static const uint NA_WARPS_PER_BLOCK = 8;   // 256 / 32

// ENGINE launch ABI of newArticulationsLaunch: exactly the CUDA parameter
// list — two device pointers ({8,8} blob).
struct NewArticulationsParams
{
    uint64_t scDesc;  // const PxgUpdateArticulationDesc*
    uint64_t dofData; // const PxReal*
};

[[vk::push_constant]]
ConstantBuffer<NewArticulationsParams> gNewArtiAbi;

// warpCopy<uint>(dest, source, totalSize) — the elementwise strided copy, see
// the header note on why a u32 walk reproduces every width the CUDA uses.
void naCopy(uint64_t dst, uint64_t src, uint byteCount, uint lane)
{
    Ptr<uint> d = Ptr<uint>(dst);
    Ptr<uint> s = Ptr<uint>(src);
    const uint n = byteCount >> 2;
    for (uint i = lane; i < n; i += NA_WARP_SIZE)
        d[i] = s[i];
}

// warpCopy<uint>(dest, value, totalSize) — the value-fill overload.
void naFill(uint64_t dst, uint value, uint byteCount, uint lane)
{
    Ptr<uint> d = Ptr<uint>(dst);
    const uint n = byteCount >> 2;
    for (uint i = lane; i < n; i += NA_WARP_SIZE)
        d[i] = value;
}

[shader("compute")]
[numthreads(32, 8, 1)]
void newArticulationsLaunch(uint3 groupThreadId : SV_GroupThreadID,
                            uint3 groupId : SV_GroupID)
{
    const PxgUpdateArticulationDescGpu scDesc =
        (Ptr<PxgUpdateArticulationDescGpu>(gNewArtiAbi.scDesc))[0];

    const uint lane = groupThreadId.x;
    const uint globalWarpIndex = groupId.x * NA_WARPS_PER_BLOCK + groupThreadId.y;

    if (globalWarpIndex >= scDesc.mNbNewArticulations)
        return;

    // the staged source record and its destination slot in the persistent pool
    const uint64_t newArticulation =
        scDesc.mNewArticulations + uint64_t(globalWarpIndex) * PXG_ARTI_STRIDE;

    const uint articulationIndex = artiU32Field(newArticulation, PXG_ARTI_DATA_INDEX);
    const uint64_t articulation =
        scDesc.mArticulationPool + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;

    // the record itself: 544 bytes, staging buffer -> persistent pool
    naCopy(articulation, newArticulation, PXG_ARTI_STRIDE, lane);

    // the articulation's solver sleep data starts zeroed
    naFill(scDesc.mArticulationSleepDataPool +
               uint64_t(articulationIndex) * PXG_ARTI_SLEEP_DATA_SIZE,
           0u, PXG_ARTI_SLEEP_DATA_SIZE, lane);

    const uint nbLinks = artiU16Field(newArticulation, PXG_ARTI_DATA_NUM_LINKS);

    // the per-articulation index record: where this articulation's slice of
    // each flattened new-* pool begins
    const PxgArticulationSimUpdateGpu update =
        (Ptr<PxgArticulationSimUpdateGpu>(scDesc.mIndicesOffset))[globalWarpIndex];

    const uint startIndex = update.linkStartIndex;
    const uint dofStartIndex = update.dofDataStartIndex;
    const uint spatialTendonStartIndex = update.spatialTendonStartIndex;
    const uint fixedTendonStartIndex = update.fixedTendonStartIndex;
    const uint dirtyFlags = update.dirtyFlags;

    // ---- per-link arrays, staged pool slice -> the articulation's own buffer ----
    naCopy(artiPtrField(newArticulation, PXG_ARTI_LINKS),
           scDesc.mNewLinks + uint64_t(startIndex) * PXG_ARTI_LINK_SIZE,
           PXG_ARTI_LINK_SIZE * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_LINK_WAKE_COUNTERS),
           scDesc.mNewLinkWakeCounters + uint64_t(startIndex) * 4u,
           4u * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_LINK_PROPS),
           scDesc.mNewLinkProps + uint64_t(startIndex) * PXG_ARTI_LINK_PROP_SIZE,
           PXG_ARTI_LINK_PROP_SIZE * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_PARENTS),
           scDesc.mNewLinkParents + uint64_t(startIndex) * 4u,
           4u * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_CHILDREN),
           scDesc.mNewLinkChildren + uint64_t(startIndex) * PXG_ARTI_BITFIELD_SIZE,
           PXG_ARTI_BITFIELD_SIZE * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_LINK_BODY2WORLDS),
           scDesc.mNewLinkBody2Worlds + uint64_t(startIndex) * PXG_ARTI_TRANSFORM_SIZE,
           PXG_ARTI_TRANSFORM_SIZE * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_LINK_BODY2ACTORS),
           scDesc.mNewLinkBody2Actors + uint64_t(startIndex) * PXG_ARTI_TRANSFORM_SIZE,
           PXG_ARTI_TRANSFORM_SIZE * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_JOINTS),
           scDesc.mNewJointCores + uint64_t(startIndex) * PXG_ARTI_JOINT_CORE_SIZE,
           PXG_ARTI_JOINT_CORE_SIZE * nbLinks, lane);

    naCopy(artiPtrField(newArticulation, PXG_ARTI_JOINT_DATA),
           scDesc.mNewJointData + uint64_t(startIndex) * PXG_ARTI_JOINT_CORE_DATA_SIZE,
           PXG_ARTI_JOINT_CORE_DATA_SIZE * nbLinks, lane);

    // ---- spatial tendons ----
    //
    // The per-tendon loop loads the tendon record (and with it two device
    // pointers) from the staged pool and copies through them: a chained BDA
    // load inside a loop, which P8.40 flags as a hazard.  It is not a
    // GRID-STRIDE loop — the trip count is this articulation's own tendon
    // count, and every lane of the virtual warp walks the same trip — so the
    // hazard P8.40 describes (a chained load whose base changes per iteration
    // across an unbounded grid stride) does not apply.  The fixture drives a
    // real tendon so this is exercised rather than assumed.
    {
        const uint nbSpatialTendons =
            artiU16Field(newArticulation, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
        const uint endIndex = spatialTendonStartIndex + nbSpatialTendons;

        for (uint i = spatialTendonStartIndex; i < endIndex; ++i)
        {
            const uint64_t tendon =
                scDesc.mNewSpatialTendonPool + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
            const uint attachmentStartIndex =
                (Ptr<uint>(scDesc.mNewTendonAttachmentRemapPool))[i];
            const uint nbElements = artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS);

            naCopy(artiPtrField(tendon, PXG_TENDON_FIXED_ELEMENTS),
                   scDesc.mNewAttachmentFixedPool +
                       uint64_t(attachmentStartIndex) * PXG_ARTI_TENDON_FIXED_ELEM_SIZE,
                   PXG_ARTI_TENDON_FIXED_ELEM_SIZE * nbElements, lane);

            naCopy(artiPtrField(tendon, PXG_TENDON_MOD_ELEMENTS),
                   scDesc.mNewAttachmentModPool +
                       uint64_t(attachmentStartIndex) * PXG_ARTI_ATTACHMENT_MOD_SIZE,
                   PXG_ARTI_ATTACHMENT_MOD_SIZE * nbElements, lane);
        }

        naCopy(artiPtrField(newArticulation, PXG_ARTI_SPATIAL_TENDONS),
               scDesc.mNewSpatialTendonPool +
                   uint64_t(spatialTendonStartIndex) * PXG_ARTI_TENDON_SIZE,
               PXG_ARTI_TENDON_SIZE * nbSpatialTendons, lane);

        naCopy(artiPtrField(newArticulation, PXG_ARTI_SPATIAL_TENDON_PARAMS),
               scDesc.mNewSpatialTendonParamsPool +
                   uint64_t(spatialTendonStartIndex) * PXG_ARTI_SPATIAL_TENDON_DATA_SIZE,
               PXG_ARTI_SPATIAL_TENDON_DATA_SIZE * nbSpatialTendons, lane);
    }

    // ---- fixed tendons ----
    {
        const uint nbFixedTendons =
            artiU16Field(newArticulation, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
        const uint endIndex = fixedTendonStartIndex + nbFixedTendons;

        for (uint i = fixedTendonStartIndex; i < endIndex; ++i)
        {
            const uint64_t tendon =
                scDesc.mNewFixedTendonPool + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
            const uint tendonJointStartIndex =
                (Ptr<uint>(scDesc.mNewTendonTendonJointRemapPool))[i];
            const uint nbElements = artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS);

            naCopy(artiPtrField(tendon, PXG_TENDON_FIXED_ELEMENTS),
                   scDesc.mNewTendonJointFixedPool +
                       uint64_t(tendonJointStartIndex) * PXG_ARTI_TENDON_FIXED_ELEM_SIZE,
                   PXG_ARTI_TENDON_FIXED_ELEM_SIZE * nbElements, lane);

            naCopy(artiPtrField(tendon, PXG_TENDON_MOD_ELEMENTS),
                   scDesc.mNewTendonJointCoefficientPool +
                       uint64_t(tendonJointStartIndex) * PXG_ARTI_TENDON_JOINT_COEFF_SIZE,
                   PXG_ARTI_TENDON_JOINT_COEFF_SIZE * nbElements, lane);
        }

        naCopy(artiPtrField(newArticulation, PXG_ARTI_FIXED_TENDONS),
               scDesc.mNewFixedTendonPool +
                   uint64_t(fixedTendonStartIndex) * PXG_ARTI_TENDON_SIZE,
               PXG_ARTI_TENDON_SIZE * nbFixedTendons, lane);

        naCopy(artiPtrField(newArticulation, PXG_ARTI_FIXED_TENDON_PARAMS),
               scDesc.mNewFixedTendonParamsPool +
                   uint64_t(fixedTendonStartIndex) * PXG_ARTI_FIXED_TENDON_DATA_SIZE,
               PXG_ARTI_FIXED_TENDON_DATA_SIZE * nbFixedTendons, lane);
    }

    // ---- mimic joints and the path-to-root table ----
    {
        const uint numMimicJoints =
            artiU16Field(newArticulation, PXG_ARTI_DATA_NUM_MIMIC_JOINTS);
        naCopy(artiPtrField(newArticulation, PXG_ARTI_MIMIC_JOINT_CORES),
               scDesc.mNewArticulationMimicJointPool +
                   uint64_t(update.mimicJointStartIndex) * PXG_ARTI_MIMIC_JOINT_CORE_SIZE,
               PXG_ARTI_MIMIC_JOINT_CORE_SIZE * numMimicJoints, lane);

        const uint numPathToRoots =
            artiU32Field(newArticulation, PXG_ARTI_DATA_NUM_PATH_TO_ROOTS);
        naCopy(artiPtrField(newArticulation, PXG_ARTI_PATH_TO_ROOT),
               scDesc.mNewPathToRootPool + uint64_t(update.pathToRootIndex) * 4u,
               4u * numPathToRoots, lane);
    }

    // ---- per-link state that starts at zero ----
    {
        const uint totalSize = PXG_ARTI_SPATIAL_VECTOR_SIZE * nbLinks;
        naFill(artiPtrField(newArticulation, PXG_ARTI_MOTION_VELOCITIES), 0u, totalSize, lane);
        naFill(artiPtrField(newArticulation, PXG_ARTI_MOTION_ACCELERATIONS), 0u, totalSize, lane);
        naFill(artiPtrField(newArticulation, PXG_ARTI_CORIOLISE_VECTORS), 0u, totalSize, lane);

        naFill(artiPtrField(newArticulation, PXG_ARTI_LINK_SLEEP_DATA), 0u,
               PXG_ARTI_LINK_SLEEP_DATA_SIZE * nbLinks, lane);
    }

    // ---- joint dof state: from dofData where the host marked it dirty, zero
    //      otherwise.  `dofsTruncated` reproduces the CUDA's PxU8 read; see the
    //      header note.  The offset advances only on the dirty branches, so the
    //      dofData layout is exactly the order of these five tests. ----
    const uint dofsTruncated =
        artiU16Field(newArticulation, PXG_ARTI_DATA_NUM_JOINT_DOFS) & 0xFFu;
    const uint zeroFillSize = 4u * dofsTruncated;

    const uint numDofs = artiU16Field(newArticulation, PXG_ARTI_DATA_NUM_JOINT_DOFS);
    uint offset = dofStartIndex;

    if ((dirtyFlags & PXG_ARTI_DIRTY_POSITIONS) != 0u)
    {
        naCopy(artiPtrField(newArticulation, PXG_ARTI_JOINT_POSITIONS),
               gNewArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
        offset += numDofs;
    }
    else
        naFill(artiPtrField(newArticulation, PXG_ARTI_JOINT_POSITIONS), 0u, zeroFillSize, lane);

    if ((dirtyFlags & PXG_ARTI_DIRTY_VELOCITIES) != 0u)
    {
        naCopy(artiPtrField(newArticulation, PXG_ARTI_JOINT_VELOCITIES),
               gNewArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
        offset += numDofs;
    }
    else
        naFill(artiPtrField(newArticulation, PXG_ARTI_JOINT_VELOCITIES), 0u, zeroFillSize, lane);

    if ((dirtyFlags & PXG_ARTI_DIRTY_FORCES) != 0u)
    {
        naCopy(artiPtrField(newArticulation, PXG_ARTI_JOINT_FORCE),
               gNewArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
        offset += numDofs;
    }
    else
        naFill(artiPtrField(newArticulation, PXG_ARTI_JOINT_FORCE), 0u, zeroFillSize, lane);

    if ((dirtyFlags & PXG_ARTI_DIRTY_JOINT_TARGET_POS) != 0u)
    {
        naCopy(artiPtrField(newArticulation, PXG_ARTI_JOINT_TARGET_POSITIONS),
               gNewArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
        offset += numDofs;
    }
    else
        naFill(artiPtrField(newArticulation, PXG_ARTI_JOINT_TARGET_POSITIONS), 0u, zeroFillSize, lane);

    if ((dirtyFlags & PXG_ARTI_DIRTY_JOINT_TARGET_VEL) != 0u)
    {
        naCopy(artiPtrField(newArticulation, PXG_ARTI_JOINT_TARGET_VELOCITIES),
               gNewArtiAbi.dofData + uint64_t(offset) * 4u, 4u * numDofs, lane);
        offset += numDofs;
    }
    else
        naFill(artiPtrField(newArticulation, PXG_ARTI_JOINT_TARGET_VELOCITIES), 0u, zeroFillSize, lane);

    // external accelerations: the staged per-link slice when dirty, zero otherwise
    if ((dirtyFlags & PXG_ARTI_DIRTY_EXT_ACCEL) != 0u)
        naCopy(artiPtrField(newArticulation, PXG_ARTI_EXTERNAL_ACCELERATIONS),
               scDesc.mNewLinkExtAccels + uint64_t(startIndex) * PXG_ARTI_SPATIAL_VECTOR_SIZE,
               PXG_ARTI_SPATIAL_VECTOR_SIZE * nbLinks, lane);
    else
        naFill(artiPtrField(newArticulation, PXG_ARTI_EXTERNAL_ACCELERATIONS), 0u,
               PXG_ARTI_SPATIAL_VECTOR_SIZE * nbLinks, lane);
}
