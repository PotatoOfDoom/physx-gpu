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

// npCvxFinishContacts.slang
//
// P5 Step 4 (BDA, engine launch path): the COMPLETE finishContactsKernel
// (cudaGJKEPA.cu L2010-2167 = writeContactsToStream4threads L1620-1875 +
// the status/patch tail + writePatchesToStream4threads L1942-2007) on the
// exact engine ABI — 20 arguments,
//   argSizes {4,8,8,8,8,8,8,8,8,8,1,8,8,8,8,8,8,4,4,4} -> 152-byte blob
//   (numTests@0, 9 pointers @8..@72, bool insertAveragePoint@80,
//    6 pointers @88..@128, three PxU32 limits @136/@140/@144).
// Engine grid (PxgNarrowphaseCore.cpp ~L674): blockDim (32, 4), numBlocks =
// ceil(numTests / 128) — one warp per 32 pairs; elementGlobalIndex =
// threadIdx.x + blockIdx.x*128 + threadIdx.y*32.
//
// The arithmetic is the union of the two bit-verified bound-buffer ports —
// npCvxWriteStream (tranche 151: one warp-iteration of the stream write)
// and the previous npCvxFinishContacts (tranche 152: the tail + patch
// writer) — composed back into the .cu's single kernel and converted to
// real-struct BDA access.  The warp is serialized on lane 0 (llvmpipe has
// no 32-wide subgroup): the FinishContactsWarpScratch shared arrays become
// warp-thread locals, ballots/shuffles become straight-line code, and the
// per-warp global atomics (contact/force reservations once per 8-pair
// iteration, one patch reservation per warp) run in the same order the
// warp-synchronous CUDA issues them.  Memory effects mirror the .cu:
//   - manifold mNbContacts invalidation-bit clear;
//   - contact-stream float4 writes (group average point first, then the
//     kept world points), force bytes reserved but not written;
//   - cmOutputs contactPoints/contactForces = stream base + byte offset
//     (0xFFFFFFFF sentinel and the force-offset uint wraparound ported
//     exactly), allflags rewrite, nbContacts, contactPatches;
//   - touchChangeFlags/patchChangeFlags;
//   - PxgPatchAndContactCounters atomic reservations + overflow flags
//     (contact/force: >= limit; patch: strictly > limit) and the patch
//     overflow rollback;
//   - 64-byte PxContactPatch payloads as 4 uint4 stores per patch.
// Inherited modelling notes (identical to the verified ports): the
// MaterialParams2 pad u16 the CUDA leaves as uninitialized shared memory in
// patch bytes 54-55 is written as 0; cmOutputs.nbContacts (a PxU16 store in
// CUDA) is a read-modify-write of the 32-bit word that preserves the
// adjacent `flags` u16, which nothing writes concurrently.

import pxgSlangNpAbi;

static const float FLT_MAXV = 3.4028235e38f;
static const uint EGT_SPHERE = 0u;
static const uint EGT_CAPSULE = 2u;

static const uint HAS_NO_TOUCH = 1u;
static const uint HAS_TOUCH = 2u;
static const uint TOUCH_KNOWN = 3u; // eHAS_NO_TOUCH | eHAS_TOUCH
static const uint CONTACT_BUFFER_OVERFLOW = 1u;
static const uint FORCE_BUFFER_OVERFLOW = 2u;
static const uint PATCH_BUFFER_OVERFLOW = 4u;

static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint FLAG_DISABLE_FRICTION = 1u, FLAG_DISABLE_STRONG_FRICTION = 2u, FLAG_COMPLIANT_ACC_SPRING = 16u;

// Engine ABI blob (natural alignment): 152 bytes.
struct FinishParams
{
    uint numTests;                    // @0
    uint64_t cmInputs;                // @8   const PxgContactManagerInput*
    uint64_t cmOutputs;               // @16  PxsContactManagerOutput*
    uint64_t cmManifolds;             // @24  PxgPersistentContactManifold*
    uint64_t shapes;                  // @32  PxgShape*
    uint64_t transformCache;          // @40  const PxsCachedTransform*
    uint64_t contactDistance;         // @48  const PxReal*
    uint64_t materials;               // @56  const PxsMaterialData*
    uint64_t contactStream;           // @64  PxU8*
    uint64_t patchStream;             // @72  PxU8*
    uint insertAveragePointWord;      // @80  bool byte (upper pad bytes zero-filled)
    uint64_t patchAndContactCounters; // @88  PxgPatchAndContactCounters*
    uint64_t touchChangeFlags;        // @96  PxU32*
    uint64_t patchChangeFlags;        // @104 PxU32*
    uint64_t startContactPatches;     // @112 PxU8*
    uint64_t startContactPoints;      // @120 PxU8*
    uint64_t startContactForces;      // @128 PxU8*
    uint patchBytesLimit;             // @136
    uint contactBytesLimit;           // @140
    uint forceBytesLimit;             // @144
}

[[vk::push_constant]]
ConstantBuffer<FinishParams> gFin;

float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

// combineMaterials on the REAL PxsMaterialData layout (materialCombiner.cuh
// port, bit-verified in the npMaterialCombine / npCvxFinishContacts tranches).
void combineMaterials(PxsMaterialDataGpu m0, PxsMaterialDataGpu m1,
                      out uint materialFlags, out float staFriction, out float dynFriction,
                      out float restitution, out float damping)
{
    const uint flags0 = m0.flagsModes0 & 0xFFFFu;
    const uint flags1 = m1.flagsModes0 & 0xFFFFu;
    const uint fricMode0 = (m0.flagsModes0 >> 16u) & 0xFFu;
    const uint fricMode1 = (m1.flagsModes0 >> 16u) & 0xFFu;
    const uint restMode0 = (m0.flagsModes0 >> 24u) & 0xFFu;
    const uint restMode1 = (m1.flagsModes0 >> 24u) & 0xFFu;
    const uint dampMode0 = m0.modes1 & 0xFFu;
    const uint dampMode1 = m1.modes1 & 0xFFu;

    const float r0 = m0.restitution;
    const float r1 = m1.restitution;
    const bool compliant0 = r0 < 0.0f;
    const bool compliant1 = r1 < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const bool compliantAcc0 = (flags0 & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool compliantAcc1 = (flags1 & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool exactlyOneAccCompliant = compliantAcc0 != compliantAcc1;

    if (bothCompliant && exactlyOneAccCompliant)
    {
        restitution = compliantAcc0 ? r0 : r1;
    }
    else
    {
        const uint combineMode = exactlyOneCompliant ? MODE_MIN : max(restMode0, restMode1);
        const float flipSign = (bothCompliant && (combineMode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
        restitution = flipSign * combineScalars(r0, r1, combineMode);
    }

    {
        const float d0 = m0.damping;
        const float d1 = m1.damping;
        if (bothCompliant && exactlyOneAccCompliant)
        {
            damping = compliantAcc0 ? d0 : d1;
        }
        else
        {
            const uint combineMode = exactlyOneCompliant ? MODE_MAX : max(dampMode0, dampMode1);
            damping = combineScalars(d0, d1, combineMode);
        }
    }

    {
        const uint combineFlags = flags0 | flags1;
        if ((combineFlags & FLAG_DISABLE_FRICTION) == 0u)
        {
            const uint frictionCombineMode = max(fricMode0, fricMode1);
            const float dynF = combineScalars(m0.dynamicFriction, m1.dynamicFriction, frictionCombineMode);
            const float staF = combineScalars(m0.staticFriction, m1.staticFriction, frictionCombineMode);
            const float fDynFriction = max(dynF, 0.0f);
            const float fStaFriction = (staF - fDynFriction) >= 0.0f ? staF : fDynFriction;
            dynFriction = fDynFriction;
            staFriction = fStaFriction;
            materialFlags = combineFlags;
        }
        else
        {
            materialFlags = combineFlags | FLAG_DISABLE_STRONG_FRICTION;
            dynFriction = 0.0f;
            staFriction = 0.0f;
        }
    }
}

uint countbitsBelow(uint mask, uint lane) { return countbits(mask & ((1u << lane) - 1u)); }

[shader("compute")]
[numthreads(32, 4, 1)]
void finishContactsKernel(uint3 groupID : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    if (tid.x != 0u)
        return; // serialized warp (threadIdx.y selects the warp)

    const uint numTests = gFin.numTests;
    const bool insertAveragePoint = (gFin.insertAveragePointWord & 0xFFu) != 0u;
    const uint warpBeg = groupID.x * 128u + tid.y * 32u;

    Ptr<PxgContactManagerInputGpu> cmInputs = Ptr<PxgContactManagerInputGpu>(gFin.cmInputs);
    Ptr<PxsContactManagerOutputGpu> cmOutputs = Ptr<PxsContactManagerOutputGpu>(gFin.cmOutputs);
    Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(gFin.shapes);
    Ptr<PxsCachedTransformGpu> cache = Ptr<PxsCachedTransformGpu>(gFin.transformCache);
    Ptr<float> contactDistance = Ptr<float>(gFin.contactDistance);
    Ptr<PxsMaterialDataGpu> materials = Ptr<PxsMaterialDataGpu>(gFin.materials);
    Ptr<PxgPatchAndContactCountersGpu> counters = Ptr<PxgPatchAndContactCountersGpu>(gFin.patchAndContactCounters);

    // FinishContactsWarpScratch, one warp's slice, as warp-thread locals.
    float scrNormX[32], scrNormY[32], scrNormZ[32];
    uint scrNbContacts[32];   // contactParams.nbContacts (startIdx = 0)
    uint scrMatIndex0[32];    // materialParams.materialIndex0 (internalFlags = 0)
    uint scrMatIndex1[32];    // materialParams2.materialIndex1 (pad written 0)
    for (uint L = 0u; L < 32u; ++L)
    {
        scrNormX[L] = 0.0f; scrNormY[L] = 0.0f; scrNormZ[L] = 0.0f;
        scrNbContacts[L] = 0u; scrMatIndex0[L] = 0u; scrMatIndex1[L] = 0u;
    }

    // ==================================================================
    // writeContactsToStream4threads: 4 iterations x 8 groups x 4 lanes
    // ==================================================================
    for (uint baseOffs = 0u; baseOffs < 32u; baseOffs += 8u)
    {
        bool valid[8];
        bool flip[8];
        float radius[8];
        float cd[8];
        float4 t1q[8];
        float3 t1p[8];
        uint num0[8]; // manifold count after invalidation handling

        for (uint g = 0u; g < 8u; ++g)
        {
            const uint elem = warpBeg + baseOffs + g;
            valid[g] = elem < numTests;
            flip[g] = false;
            radius[g] = 0.0f;
            cd[g] = 0.0f;
            t1q[g] = float4(0, 0, 0, 1);
            t1p[g] = float3(0, 0, 0);
            num0[g] = 0u;

            if (valid[g])
            {
                const PxgContactManagerInputGpu input = cmInputs[elem];
                const PxgShapeGpu shape0 = shapes[input.shapeRef0];
                const PxgShapeGpu shape1 = shapes[input.shapeRef1];

                uint type0 = shape0.type;
                uint type1 = shape1.type;
                float3 scale0 = float3(shape0.scaleX, shape0.scaleY, shape0.scaleZ);
                float3 scale1 = float3(shape1.scaleX, shape1.scaleY, shape1.scaleZ);
                uint transformCacheRef0 = input.transformCacheRef0;
                uint transformCacheRef1 = input.transformCacheRef1;

                flip[g] = type1 < type0;
                if (flip[g])
                {
                    uint tu = type0; type0 = type1; type1 = tu;
                    const float3 ts = scale0; scale0 = scale1; scale1 = ts;
                    tu = transformCacheRef0; transformCacheRef0 = transformCacheRef1; transformCacheRef1 = tu;
                }

                if (type0 == EGT_SPHERE || type0 == EGT_CAPSULE)
                    radius[g] = scale0.y;

                const PxsCachedTransformGpu tc1 = cache[transformCacheRef1];
                t1q[g] = float4(tc1.qx, tc1.qy, tc1.qz, tc1.qw);
                t1p[g] = float3(tc1.px, tc1.py, tc1.pz);
                cd[g] = contactDistance[transformCacheRef0] + contactDistance[transformCacheRef1];

                // tidInGroup == 0: scratch material indices — the CUDA reads
                // them BEFORE the flip swap (which swaps types/scales/refs
                // only), so they are always the UNSWAPPED shape0/shape1 values
                scrMatIndex0[baseOffs + g] = shape0.materialIndex;
                scrMatIndex1[baseOffs + g] = shape1.materialIndex;

                // manifold mNbContacts: clear the invalidation bit in memory
                Ptr<PxgPersistentContactManifoldGpu> manifold =
                    Ptr<PxgPersistentContactManifoldGpu>(gFin.cmManifolds + uint64_t(elem) * 256u);
                uint numManifoldContacts = manifold[0].nbContacts;
                manifold[0].nbContacts = numManifoldContacts & ~0x80000000u;
                if ((numManifoldContacts & 0x80000000u) != 0u)
                    numManifoldContacts = 0u;
                num0[g] = numManifoldContacts;
            }
        }

        // per-lane normPen / world pointB
        float4 normPenL[32]; // only lanes of this iteration (index 4g+j)
        float4 pointBL[32];
        for (uint L = 0u; L < 32u; ++L)
        {
            const uint g = L / 4u, j = L % 4u;
            normPenL[L] = float4(0, 0, 0, 0);
            float4 pb = float4(FLT_MAXV, FLT_MAXV, FLT_MAXV, FLT_MAXV);
            if (j < num0[g])
            {
                const uint elem = warpBeg + baseOffs + g;
                Ptr<PxgPersistentContactManifoldGpu> manifold =
                    Ptr<PxgPersistentContactManifoldGpu>(gFin.cmManifolds + uint64_t(elem) * 256u);
                normPenL[L] = manifold[0].localNormalPen[j];
                const float sep = normPenL[L].w - radius[g];
                const float4 pointB = manifold[0].localContactB[j];
                const float3 p = npAbiQuatRotate(t1q[g], pointB.xyz) + t1p[g];
                pb = float4(p, sep);
            }
            pointBL[L] = pb;
        }

        // group average normal: shfl_xor(2) + shfl_xor(1) butterfly = (n0+n2)+(n1+n3)
        for (uint g = 0u; g < 8u; ++g)
        {
            const float3 sum = (normPenL[4u * g].xyz + normPenL[4u * g + 2u].xyz)
                             + (normPenL[4u * g + 1u].xyz + normPenL[4u * g + 3u].xyz);
            const float magSq = dot(sum, sum);
            float3 avgNorm;
            if (magSq >= 1e-5f)
                avgNorm = sum * (1.0f / sqrt(magSq)); // PxVec3::operator/= is reciprocal-multiply
            else
                avgNorm = normPenL[4u * g].xyz; // the storing lane (tidInGroup 0)'s own normal
            avgNorm = npAbiQuatRotate(t1q[g], avgNorm);
            if (flip[g]) avgNorm = -avgNorm;
            scrNormX[baseOffs + g] = avgNorm.x;
            scrNormY[baseOffs + g] = avgNorm.y;
            scrNormZ[baseOffs + g] = avgNorm.z;
        }

        // keep / extra ballots (warp-wide)
        uint keepContactW = 0u;
        for (uint L = 0u; L < 32u; ++L)
            if (pointBL[L].w <= cd[L / 4u]) keepContactW |= (1u << L);

        uint numKept[8];
        for (uint g = 0u; g < 8u; ++g)
            numKept[g] = countbits(keepContactW & (0xFu << (4u * g)));

        uint extraContactsW = 0u;
        for (uint g = 0u; g < 8u; ++g)
            if (insertAveragePoint && numKept[g] > 1u) extraContactsW |= (1u << (4u * g));

        // warp lane 0: atomic byte reservations for the whole iteration
        uint contactByteOffset0 = 0xFFFFFFFFu;
        uint forceByteOffset0 = 0xFFFFFFFFu;
        {
            const uint totalContacts = countbits(keepContactW) + countbits(extraContactsW);
            if (totalContacts != 0u)
            {
                InterlockedAdd(counters[0].contactsBytes, 16u * totalContacts, contactByteOffset0);
                InterlockedAdd(counters[0].forceAndIndiceBytes, 4u * totalContacts, forceByteOffset0);

                if ((contactByteOffset0 + 16u * totalContacts) >= gFin.contactBytesLimit)
                {
                    uint prev;
                    InterlockedOr(counters[0].overflowError, CONTACT_BUFFER_OVERFLOW, prev);
                    contactByteOffset0 = 0xFFFFFFFFu;
                }
                else if ((forceByteOffset0 + 4u * totalContacts) >= gFin.forceBytesLimit)
                {
                    uint prev;
                    InterlockedOr(counters[0].overflowError, FORCE_BUFFER_OVERFLOW, prev);
                    forceByteOffset0 = 0xFFFFFFFFu;
                }
            }
        }

        // per-lane compacted stream writes + per-group cmOutputs epilogue
        for (uint g = 0u; g < 8u; ++g)
        {
            const uint elem = warpBeg + baseOffs + g;
            uint numKeptG = numKept[g];
            uint cbLane0 = contactByteOffset0;
            uint fbLane0 = forceByteOffset0;

            if (contactByteOffset0 != 0xFFFFFFFFu)
            {
                // group average point: a lane-INDEX test against the KEPT count
                // (not the kept bitmask) — an unkept low lane can contribute
                // while a kept high lane does not; (p0+p2)+(p1+p3) butterfly.
                float4 pv[4];
                for (uint j = 0u; j < 4u; ++j)
                    pv[j] = (j < numKeptG) ? pointBL[4u * g + j] : float4(0, 0, 0, 0);
                const float4 avgPt = (pv[0] + pv[2]) + (pv[1] + pv[3]);

                for (uint j = 0u; j < 4u; ++j)
                {
                    const uint L = 4u * g + j;
                    const uint byteOffset = countbitsBelow(keepContactW, L) + countbitsBelow(extraContactsW, L);
                    const uint cb = contactByteOffset0 + 16u * byteOffset;
                    const uint fb = forceByteOffset0 + 4u * byteOffset;
                    if (j == 0u) { cbLane0 = cb; fbLane0 = fb; }

                    uint offset = 0u;
                    if (insertAveragePoint && numKeptG > 1u)
                    {
                        if (j == 0u)
                        {
                            // cutil operator/=(float4, float) is reciprocal-multiply
                            const float4 a = avgPt * (1.0f / float(numKeptG));
                            (Ptr<float4>(gFin.contactStream + uint64_t(cb)))[0] = a; // avg point first
                            offset += 16u;
                        }
                    }
                    if (pointBL[L].w <= cd[g])
                        (Ptr<float4>(gFin.contactStream + uint64_t(cb + offset)))[0] = pointBL[L];
                }
            }
            else
            {
                numKeptG = 0u;
            }

            // tidInGroup == 0 epilogue
            if (numKeptG == 0u)
            {
                cbLane0 = 0xFFFFFFFFu;
                fbLane0 = 0xFFFFFFFFu;
            }
            if (valid[g])
            {
                cmOutputs[elem].contactForces = gFin.startContactForces + uint64_t(fbLane0);
                cmOutputs[elem].contactPoints = gFin.startContactPoints + uint64_t(cbLane0);
            }
            scrNbContacts[baseOffs + g] = numKeptG + ((numKeptG > 1u && insertAveragePoint) ? 1u : 0u);
        }
    }

    // ==================================================================
    // tail: status flags, patch reservation, rollback, combineMaterials
    // ==================================================================
    uint statusL[32], prevPatchesL[32];
    bool previouslyHadTouchL[32];
    uint nbL[32];
    uint origW36[32];
    bool validL[32];

    for (uint L = 0u; L < 32u; ++L)
    {
        const uint elem = warpBeg + L;
        validL[L] = elem < numTests;
        nbL[L] = validL[L] ? scrNbContacts[L] : 0u;

        uint allflags = 0u;
        origW36[L] = 0u;
        if (validL[L])
        {
            allflags = cmOutputs[elem].allflagsWord;
            origW36[L] = cmOutputs[elem].nbContactsFlags;
        }

        const uint oldStatusFlags = (allflags >> 16u) & 0xFFu; // u16Low(u32High)
        uint status = oldStatusFlags;
        status &= ~TOUCH_KNOWN;
        if (nbL[L] != 0u) status |= HAS_TOUCH;
        else status |= HAS_NO_TOUCH;

        prevPatchesL[L] = (allflags & 0xFFFFu) >> 8u; // u16High(u32Low)
        previouslyHadTouchL[L] = (oldStatusFlags & HAS_TOUCH) != 0u;
        const bool prevTouchKnown = (oldStatusFlags & TOUCH_KNOWN) != 0u;

        if (validL[L])
        {
            const uint numPatches = (nbL[L] != 0u) ? 1u : 0u;
            const bool currentlyHasTouch = nbL[L] != 0u;
            const bool change = (previouslyHadTouchL[L] != currentlyHasTouch) || (!prevTouchKnown);
            (Ptr<uint>(gFin.touchChangeFlags))[elem] = change ? 1u : 0u;
            (Ptr<uint>(gFin.patchChangeFlags))[elem] = (prevPatchesL[L] != numPatches) ? 1u : 0u;
            cmOutputs[elem].allflagsWord = (((prevPatchesL[L] << 8u) | status) << 16u) | (numPatches << 8u);
            cmOutputs[elem].nbContactsFlags = (origW36[L] & 0xFFFF0000u) | (nbL[L] & 0xFFFFu);
        }
        statusL[L] = status;
    }

    uint patchMask = 0u;
    for (uint L = 0u; L < 32u; ++L)
        if (nbL[L] > 0u) patchMask |= (1u << L);

    uint patchIndex = 0xFFFFFFFFu;
    if (patchMask != 0u) // lane 31's reservation
    {
        const uint totalSz = 64u * countbits(patchMask); // sizeof(PxContactPatch)
        InterlockedAdd(counters[0].patchesBytes, totalSz, patchIndex);
        if ((patchIndex + totalSz) > gFin.patchBytesLimit) // STRICTLY >
        {
            uint prev;
            InterlockedOr(counters[0].overflowError, PATCH_BUFFER_OVERFLOW, prev);
            patchIndex = 0xFFFFFFFFu;
        }
    }

    if (patchIndex == 0xFFFFFFFFu && patchMask != 0u) // overflow rollback
    {
        patchMask = 0u;
        for (uint L = 0u; L < 32u; ++L)
        {
            scrNbContacts[L] = 0u;
            if (validL[L])
            {
                const uint elem = warpBeg + L;
                uint status = statusL[L];
                status &= ~TOUCH_KNOWN;
                status |= HAS_NO_TOUCH;
                cmOutputs[elem].allflagsWord = ((prevPatchesL[L] << 8u) | status) << 16u; // numPatches = 0
                cmOutputs[elem].nbContactsFlags = origW36[L] & 0xFFFF0000u;
                (Ptr<uint>(gFin.touchChangeFlags))[elem] = previouslyHadTouchL[L] ? 1u : 0u;
                (Ptr<uint>(gFin.patchChangeFlags))[elem] = (prevPatchesL[L] != 0u) ? 1u : 0u;
                statusL[L] = status;
            }
        }
    }

    for (uint L = 0u; L < 32u; ++L)
    {
        if (validL[L] && patchIndex != 0xFFFFFFFFu)
        {
            const uint elem = warpBeg + L;
            cmOutputs[elem].contactPatches =
                gFin.startContactPatches + uint64_t(patchIndex + 64u * countbitsBelow(patchMask, L));
        }
    }

    // combineMaterials into scratch (invalid lanes stay 0)
    float restL[32], dynFL[32], staFL[32], dampL[32];
    uint matFlagsL[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        restL[L] = 0.0f; dynFL[L] = 0.0f; staFL[L] = 0.0f; dampL[L] = 0.0f; matFlagsL[L] = 0u;
        if (validL[L])
        {
            const PxsMaterialDataGpu m0 = materials[scrMatIndex0[L]];
            const PxsMaterialDataGpu m1 = materials[scrMatIndex1[L]];
            uint mf; float sf, df, re, da;
            combineMaterials(m0, m1, mf, sf, df, re, da);
            restL[L] = re; dynFL[L] = df; staFL[L] = sf; dampL[L] = da;
            matFlagsL[L] = mf & 0xFFu; // static_cast<PxU8>
        }
    }

    // ==================================================================
    // writePatchesToStream4threads: one 64-byte PxContactPatch per masked lane
    // ==================================================================
    const uint ONE_BITS = asuint(1.0f);
    for (uint L = 0u; L < 32u; ++L)
    {
        if ((patchMask & (1u << L)) != 0u)
        {
            const uint slot = countbitsBelow(patchMask, L);
            Ptr<uint4> patch = Ptr<uint4>(gFin.patchStream + uint64_t(patchIndex) + uint64_t(slot) * 64u);
            // contactUnion: startIdx (0) | nbContacts<<16 | materialFlags<<24
            const uint contactUnion = ((scrNbContacts[L] & 0xFFu) << 16u) | (matFlagsL[L] << 24u);
            // materialUnion: internalFlags (0) | materialIndex0<<16
            const uint materialUnion = (scrMatIndex0[L] & 0xFFFFu) << 16u;
            // materialUnion2: materialIndex1 | pad (written 0; CUDA leaves shared garbage)
            const uint materialUnion2 = scrMatIndex1[L] & 0xFFFFu;
            patch[0] = uint4(ONE_BITS, ONE_BITS, ONE_BITS, ONE_BITS);
            patch[1] = uint4(asuint(scrNormX[L]), asuint(scrNormY[L]), asuint(scrNormZ[L]), asuint(restL[L]));
            patch[2] = uint4(asuint(dynFL[L]), asuint(staFL[L]), asuint(dampL[L]), contactUnion);
            patch[3] = uint4(materialUnion, materialUnion2, 0u, 0u);
        }
    }
}
