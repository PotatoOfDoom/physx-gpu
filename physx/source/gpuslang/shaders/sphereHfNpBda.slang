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

// sphereHfNpBda.slang -- `sphereHeightfieldNarrowphase`
// (convexHeightfield.cu:791-996) on the engine BDA ABI.  P9.52b-6.
//
// The kernel that finally makes P9.52a..b-4 reachable from a scene: it dispatches
// on slot 0's geometry type into the sphere core or the capsule core, then runs the
// warp output stage both arms share.
//
// ===========================================================================
// ONE THREAD PER PAIR, AND WHY THAT MAKES THIS KERNEL'S SHAPE DIFFERENT FROM
// ITS CONVEX SIBLING'S
// ===========================================================================
//
// `convexHeightfieldNarrowphase` gives each pair a whole WARP -- 32 lanes cooperate
// on the SAT and the clipper -- so its shader loops over warps within the group.
// This kernel gives each pair ONE THREAD (:871); the warp appears only in the
// output stage, where 32 pairs' contact counts are scanned together to claim one
// contiguous slice of `tempConvexTriContacts`.
//
// The group is therefore `[numthreads(32,1,1)]` and pair index is
// `groupID * 32 + lane`.  Thirty-two is not a tuning choice: `pxgSlangWarpInc`'s
// default emulation path routes every primitive through GROUPSHARED memory with
// barriers, so its call sites must be WORKGROUP-uniform, which is stricter than
// CUDA's warp-convergence.  The output stage's guards -- `if (totalContacts)`,
// `if (count)` -- are warp-uniform by construction (both are broadcasts or
// popcounts of warp-wide values), so making the workgroup exactly one warp is what
// turns warp-uniform into workgroup-uniform.  With two warps per group, one warp
// could take the guard and the other not, and llvmpipe HANGS on a divergent
// barrier rather than diagnosing it -- the lesson convexHfCoreBda's header records.
//
// ===========================================================================
// THE GRID-STRIDE LOOP IS NOT OPTIONAL HERE, AND THE SCENE PROVED IT
// ===========================================================================
//
// The convex heightfield core's shader makes exactly ONE pass and gets away with it
// because its HOST LAUNCH over-provisions the grid 32-fold on purpose --
// PxgNarrowphaseCore.cpp:2898 sets `coreGridMultiplier = 32` with the comment
// "Number which defines how we expand grid to try to cover all pairs without
// repretitions".  THIS kernel's launch has no multiplier: :2292 is
// `numBlocks = (numTests + 63) / 64`, one block for a two-body scene, and the CUDA
// covers the rest by ITERATING (`for (i = 0; i < nbPairs; i += gridDim.x *
// blockDim.x)`, :869).
//
// A first version of this shader dropped that loop, as its convex sibling does.  The
// result was not the silent coverage loss that comment predicts: it was a SIGSEGV.
// Every pair past the first 32 left its `ni` / `interm` / `ordered` / `maxDepth`
// record UNWRITTEN, and `convexTrimeshCorrelate` -- which reads the contact count out
// of `ni.index` and then dereferences the contact stream -- read uninitialised memory
// and dereferenced it.  The scene died deterministically, 3 runs of 3, with a
// two-body scene and a fault inside llvmpipe's JIT that carried no symbols; naming it
// took a launch trace added to the shim for the purpose (PXG_LAUNCH_TRACE=1).
//
// It is worth being precise about why one body was not enough to show it: a single
// sphere over the heightfield produces fewer than 32 midphase pairs, so the one pass
// covered everything.  The observing scene drops a sphere AND a capsule, which is
// what pushed the pair count past the ceiling.  A scene that exercised one arm at a
// time would have passed and the defect would have shipped.
//
// ===========================================================================
// THE TWO ARMS ARE NOT SYMMETRIC, AND THE KERNEL IS WHERE THAT BECOMES VISIBLE
// (upstream candidate 52)
// ===========================================================================
//
// The sphere arm returns an `outMask` and the capsule arm does not (:905-937): the
// sphere core queues deferred pairs for the second pass, the capsule core has no
// such out-parameter at all.  So for a workgroup of capsule pairs the ballot below
// is CONSTANTLY ZERO and the whole second-pass block is dead -- not because this
// port skipped it, but because upstream's capsule core never fills it.
//
// That asymmetry was recorded in pxgSlangCapsuleHfCore's header as candidate 52
// from reading the two cores; here it is structural rather than textual, which is
// why the same finding is restated at the site where it has consequences.  The NI
// half of the same candidate -- the capsule arm omitting the deferred mask -- stays
// output-equivalent for the reason proved there: `isDeferred` is defined and never
// called anywhere in the engine.
//
// ===========================================================================
// THE ABI ROW IS SHARED WITH THE TRIMESH AND HEIGHTFIELD CONVEX CORES
// ===========================================================================
//
// sphereHeightfieldNarrowphase (:791-818) declares the SAME 21 parameters in the
// SAME order with the same types as convexTrimeshNarrowphase (convexMesh.cu:623-647)
// and convexHeightfieldNarrowphase (convexHeightfield.cu:273-297), so the push
// constant blob is the same 168 bytes and argument 18 (`maxTempMemRequirement`) is
// the same PINNED HOST pointer.  All three rows point at `kCvxTriCore` with
// `kCvxTriCoreHostPtrMask`.  P9.48d is why that has to be exact rather than close:
// `PxgCudaShim.cpp` requires an EXACT blob match, and a kernel whose ABI row is
// missing or wrong gets `pushConstantSizeFor() == 0` and every launch REFUSED IN
// SILENCE, with fixtures, manifest test, suite and scenes all still green.
//
// ===========================================================================
// THE CONTACT STAGING IS REGISTERS HERE, NOT GROUPSHARED
// ===========================================================================
//
// The CUDA stages contacts in `shContacts[threadIdx.x * 2]` / `shSeparations[...]`,
// 128 slots for a 64-thread block.  Every slot is written and read by its OWNING
// thread only -- the output loop is `for (i = 0, idx = 2*threadIdx.x; i < nbContacts;
// ++i, idx++)` -- so the array is scratch, not communication.  Held in registers
// instead: same values, same order, and one fewer groupshared array whose barriers
// would have to be reasoned about.

#define PXG_WARP_BLOCK 32
#include "pxgSlangWarpInc.slang"

import pxgSlangMidphaseAlloc;
import pxgSlangCvxTriSetup;        // CtShape / CtContactManagerInput
import pxgSlangSphereHfCore;       // shSphereHfCore
import pxgSlangCapsuleHfCore;      // chCapsuleCore

// P9.57: FROM THE ENGINE ENUM's numbering, never a remembered literal.  `eCONVEXCORE`
// at position 4 shifted everything above eBOX, and two shaders on LIVE kernels were
// left on the old numbers with their fixtures carrying the same mistake.
// tools/check_geom_type_constants.py reads PxGeometry.h and checks these.
static const uint GEOM_SPHERE = 0u;
static const uint GEOM_CAPSULE = 2u;

// calculateConvexMeshPairMemRequirement() (midphaseAllocate.cuh:93).  NOT the
// ConvexMeshPair stride, which is also 56 -- two unrelated numbers that coincide.
static const uint SHN_MEM_PER_PAIR = 56u;
// sizeof(ConvexTriContact) -- one float4.
static const uint SHN_CONTACT_STRIDE = 16u;

// The 21-argument pack of convexHeightfield.cu:791-818 at CUDA's natural-alignment
// .param layout -- the same 168-byte blob as CvxTriCoreParams, field for field.
struct SphereHfNpParams
{
    uint64_t cmInputs;                 // @0
    uint64_t transformCache;           // @8
    uint64_t contactDistance;          // @16
    uint64_t gpuShapes;                // @24
    uint64_t cvxTrimeshPair;           // @32
    uint64_t cvxTriNIPtr;              // @40
    uint64_t cvxTriContactsPtr;        // @48
    uint64_t cvxTriMaxDepthPtr;        // @56
    uint64_t cvxTriIntermPtr;          // @64
    uint64_t orderedCvxTriIntermPtr;   // @72
    uint64_t cvxTriSecondPassPairsPtr; // @80
    uint64_t stackPtr;                 // @88
    uint64_t nbPairsGlobal;            // @96
    uint64_t nbPaddedPairsGlobal;      // @104
    uint64_t nbSecondPassPairs;        // @112
    uint     stackSizeBytes;           // @120
    uint     _pad124;                  // @124 (the next argument is 8-aligned)
    uint64_t tempConvexTriContacts;    // @128
    uint64_t pTempContactIndex;        // @136
    uint64_t maxTempMemRequirement;    // @144  PINNED HOST memory
    uint64_t midphasePairsNeeded;      // @152
    uint     nbContactManagers;        // @160
    uint     _pad164;                  // @164 (blob rounded to maxAlign 8)
}                                      // = 168

[[vk::push_constant]]
ConstantBuffer<SphereHfNpParams> gShn;

[shader("compute")]
[numthreads(32, 1, 1)]
void sphereHeightfieldNarrowphase(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint nbPairs = Ptr<uint>(gShn.nbPairsGlobal)[0];
    const uint nbPaddedPairs = Ptr<uint>(gShn.nbPaddedPairsGlobal)[0];

    // Every CUDA block computes its own private copy (:826-838).
    const MidphaseAlloc a = maCompute(gShn.stackPtr, nbPairs, nbPaddedPairs);

    if (gid.x == 0u && lane == 0u)
    {
        // BY NAME, never by position: maCompute returns its seven addresses in
        // ALLOCATION order, which is not the order of the CUDA function's
        // out-parameters, and every one points inside the same stack allocation --
        // so a misplaced store silently overwrites another section's records
        // instead of faulting.
        Ptr<uint64_t>(gShn.cvxTriNIPtr)[0] = a.ni;
        Ptr<uint64_t>(gShn.cvxTriContactsPtr)[0] = a.contacts;
        Ptr<uint64_t>(gShn.cvxTriMaxDepthPtr)[0] = a.maxDepth;
        Ptr<uint64_t>(gShn.cvxTriIntermPtr)[0] = a.interm;
        Ptr<uint64_t>(gShn.orderedCvxTriIntermPtr)[0] = a.ordered;
        Ptr<uint64_t>(gShn.cvxTriSecondPassPairsPtr)[0] = a.secondPass;

        // The atomic lands in PINNED HOST memory -- see the header.
        const uint afterPairPadding = 3u * (SHN_MEM_PER_PAIR - 2u * 4u);
        const uint afterTriangleIndexPadding = gShn.nbContactManagers * 6u * 4u;
        const uint required = SHN_MEM_PER_PAIR * Ptr<uint>(gShn.midphasePairsNeeded)[0]
                            + afterPairPadding + afterTriangleIndexPadding;
        uint old;
        InterlockedMax(Ptr<uint>(gShn.maxTempMemRequirement)[0], required, old);
    }

    // convexHeightfield.cu:869.  `gridDim.x * blockDim.x` with this shader's
    // workgroup of one warp is `WorkgroupCount().x * 32`.  The trip count is
    // WORKGROUP-UNIFORM -- both operands are uniform -- so the barriers inside the
    // output stage stay in uniform control flow, which is the property the whole
    // one-warp-per-group choice above exists to protect.
    const uint gridStride = WorkgroupCount().x * PXG_WARP_SIZE;

    for (uint base = 0u; base < nbPairs; base += gridStride)
    {
    const uint globalThreadIndex = gid.x * PXG_WARP_SIZE + lane + base;

    uint nbContacts = 0u;
    uint outMask = 0u;
    uint4 curPair = uint4(0u, 0u, 0u, 0u);
    // The CUDA's `shContacts[2*threadIdx.x]` pair, in registers -- see the header.
    float3 contact0 = float3(0.0f, 0.0f, 0.0f);
    float3 contact1 = float3(0.0f, 0.0f, 0.0f);
    float sep0 = 0.0f;
    float sep1 = 0.0f;

    if (globalThreadIndex < nbPairs)
    {
        curPair = Ptr<uint4>(a.pairs)[globalThreadIndex];

        const CtContactManagerInput cin =
            Ptr<CtContactManagerInput>(gShn.cmInputs)[curPair.x];
        const uint type0 = Ptr<CtShape>(gShn.gpuShapes)[cin.shapeRef0].type;

        if (type0 == GEOM_SPHERE)
        {
            const ShCoreOut o = shSphereHfCore(globalThreadIndex,
                                               gShn.cmInputs, gShn.transformCache,
                                               gShn.contactDistance, gShn.gpuShapes,
                                               gShn.cvxTrimeshPair,
                                               a.ni, a.maxDepth, a.interm, a.ordered,
                                               curPair);
            nbContacts = o.nbOutputContacts;
            outMask = o.outMask;
            contact0 = o.outContact;
            sep0 = o.outSep;
        }
        else if (type0 == GEOM_CAPSULE)
        {
            // NO `outMask` out-parameter upstream -- see candidate 52 in the header.
            const CapHfOut o = chCapsuleCore(gShn.cmInputs, gShn.transformCache,
                                             gShn.contactDistance, gShn.gpuShapes,
                                             gShn.cvxTrimeshPair,
                                             a.ni, a.maxDepth, a.interm, a.ordered,
                                             curPair);
            nbContacts = o.nbOutputContacts;
            contact0 = o.contact[0];
            contact1 = o.contact[1];
            sep0 = o.sep[0];
            sep1 = o.sep[1];
        }
    }

    // ---- the output stage (:941-995), shared by both arms --------------------
    //
    // OUTSIDE the pair guard, exactly as the CUDA has it: every lane of the warp
    // reaches every primitive below whether or not it had a pair.
    const uint outMaskWriteMask = pxgBallotSync(PXG_FULL_MASK, lane, outMask != 0u);
    const uint count = pxgPopc(outMaskWriteMask);

    const uint inclusiveSum =
        pxgWarpScanAddInclusiveU(PXG_FULL_MASK, lane, nbContacts, 5u);
    const uint totalContacts = pxgShflSyncU(PXG_FULL_MASK, lane, inclusiveSum, 31, PXG_WARP_SIZE);

    // TWO atomics, and the CUDA reuses ONE variable named `startIndex` for both
    // results (:955 and :968).  They are unrelated: the first indexes the
    // second-pass pair queue, the second the temporary contact stream.  Kept as two
    // names here, because a reader who assumes the second reads the first's value
    // has understood the opposite of what runs.
    //
    // THE TWO BROADCASTS ARE HOISTED OUT OF THEIR GUARDS, and that is deliberate.
    // In CUDA both `__shfl_sync` calls sit inside `if (totalContacts)` / `if (count)`,
    // which is legal because both conditions are warp-uniform.  On this port's
    // default warp path they are not merely warp-level: `pxgShflSyncU` is a
    // groupshared round trip with barriers, so a conditional call is a conditional
    // BARRIER, and llvmpipe hangs on a divergent one.  The conditions really are
    // uniform -- `totalContacts` is a broadcast of lane 31 and `count` a popcount of
    // a ballot -- but "really is uniform" is a claim a later reader has to re-derive,
    // whereas an unconditional call is a property they can see.  Hoisting is
    // output-neutral: when the guard is false the broadcast value is zero and unread,
    // because every USE of it stays inside the guard.
    uint secondPassStart = 0u;
    if (totalContacts != 0u && count != 0u && lane == 0u)
        InterlockedAdd(Ptr<uint>(gShn.nbSecondPassPairs)[0], count, secondPassStart);
    secondPassStart = pxgShflSyncU(PXG_FULL_MASK, lane, secondPassStart, 0, PXG_WARP_SIZE);

    // LANE 31 claims the contact slice, because the inclusive scan puts the warp's
    // TOTAL there and nowhere else.
    uint contactStart = 0u;
    if (totalContacts != 0u && lane == 31u)
        InterlockedAdd(Ptr<uint>(gShn.pTempContactIndex)[0], inclusiveSum, contactStart);
    contactStart = pxgShflSyncU(PXG_FULL_MASK, lane, contactStart, 31, PXG_WARP_SIZE);

    if (totalContacts != 0u)
    {
        if (count != 0u && outMask != 0u)
        {
            const uint offset = pxgWarpRankInMask(outMaskWriteMask, lane);
            Ptr<uint>(a.secondPass)[secondPassStart + offset] = outMask;
        }

        // `inclusiveSum - nbContacts` is this lane's EXCLUSIVE prefix; the CUDA
        // derives it by subtraction rather than running a second scan (:970).
        contactStart += inclusiveSum - nbContacts;

        // NO BOUNDS TEST.  The convex sibling's tail refuses a slice that would
        // overrun `tempContactSize`; this kernel's tail does not test at all
        // (:972-976), and `stackSizeBytes` is passed in and never read on this
        // path.  Transcribed as written -- a port that adds a guard its source
        // lacks stops being a port, and the overflow it would prevent is
        // upstream's to decide about.
        for (uint i = 0u; i < 2u; ++i)
        {
            if (i < nbContacts)
            {
                const float3 c = (i == 0u) ? contact0 : contact1;
                const float s = (i == 0u) ? sep0 : sep1;
                const uint64_t dst = gShn.tempConvexTriContacts
                                   + uint64_t(contactStart + i) * uint64_t(SHN_CONTACT_STRIDE);
                Ptr<float4>(dst)[0] = float4(c, s);
            }
        }

        if (nbContacts != 0u)
        {
            // ConvexMeshPair, 56 B: startIndex @28.  The SAME offset the two cores
            // read for their own record slot, recomputed here because the CUDA
            // recomputes it (:989) rather than carrying the core's value out.
            const uint64_t pairAddr = gShn.cvxTrimeshPair + uint64_t(curPair.x) * 56UL;
            const uint sphereTriPairOffset = Ptr<uint>(pairAddr + 28UL)[0] + curPair.z;
            Ptr<uint>(a.contacts)[sphereTriPairOffset] = contactStart;
        }
    }
    }
}
