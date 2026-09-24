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

// pxgSlangTriTriPatch -- P9.83b-5.
//
//   The patch machinery of doTriangleTriangleCollision (convexMeshMidphase.cu),
//   starting with the piece everything else waits on: reduceContactsSingleWarp
//   (:440-496), here as a SERIAL body.
//
// ---------------------------------------------------------------------------
// WHY SERIAL, AND WHY THAT IS THE WHOLE REASON THIS FILE EXISTS
// ---------------------------------------------------------------------------
// reduceContactsSingleWarp is called at :842-843 and :950-951, both times under
// `if (warpIndex == 0)`.  Its own body has NO block barrier -- every collective
// in it is warp-scoped (__ballot_sync, and the shuffles inside contactReduce2 and
// minIndex).  In Slang each of those lowers to GroupMemoryBarrierWithGroupSync,
// so a 1:1 transcription puts thousands of WORKGROUP barriers under a
// warp-varying guard in a 1024-thread block, and llvmpipe hangs.
// check_barrier_uniformity.py pass 3 exists for exactly that shape.
//
// The serial body has no barriers at all, so one thread may run it.  It is built
// on crsContactReduce2 from pxgSlangContactReduceSerial, which P9.83b-2 added for
// precisely this call and which gpuslang.npCrsWrapper2 certifies.
//
// ---------------------------------------------------------------------------
// patchCreationMode IS A DEAD PARAMETER.  BOTH CALL SITES DO THE SAME THING.
// ---------------------------------------------------------------------------
// :458 is the only read of it:
//
//     bool needsContact = patchCreationMode ? readMask & (1 << threadIdx.x)
//                                           : !hasContact;
//
// with `readMask = ~currentContactMask` (:457) and, at the end of every
// iteration, `hasContact = currentContactMask & (1 << threadIdx.x)` (:473).  That
// assignment re-establishes `hasContact == bit(currentContactMask, lane)` on every
// pass, and it holds on entry too (currentContactMask = 0, hasContact = false).
// So `readMask & bit` IS `!hasContact`, in every lane, on every iteration.
//
// :843 passes false and :951 passes true, and they are the same gather.  This
// port therefore has ONE path.  Writing two would be inventing a divergence the
// engine does not have -- which is the more expensive mistake, because a fixture
// that exercised both would agree and certify the invention.
//
// ---------------------------------------------------------------------------
// THINGS THE TRANSCRIPTION MUST NOT TIDY
// ---------------------------------------------------------------------------
// 1. `c += __popc(readMask)` (:469) advances by the number of FREE LANES OFFERED,
//    not by the number of points actually taken.  On the first pass readMask is
//    0xFFFFFFFF, so c jumps by 32 even if totalContacts is 3.  It is correct only
//    because the free lanes were handed consecutive indices starting at c, and it
//    is what terminates the loop on a short tail.
//
// 2. `readIndex = c + warpScanExclusive(readMask, lane)` (:459) -- the offset is
//    the count of set readMask bits STRICTLY BELOW the lane.  Note this scan is
//    over readMask, while the one at :492 that compacts survivors into KeepPoints
//    is over currentContactMask.  Two different masks, one helper.
//
// 3. The reduce runs on the normal-plane PROJECTION but the RAW point is what is
//    stored and re-fed.  contactReduction.cuh:184 computes `point = pointRaw -
//    normal * dot(pointRaw, normal)` locally; :464 keeps `pA = p.xyz` unprojected
//    and :494 emits it.  Nothing ever writes the projection back.
//
// 4. `counter2 < maxCounter` with maxCounter = 100 (:451-453), incremented at the
//    TOP of the body (:455).  It is a hard bound on the number of passes and it
//    is reachable: nothing else guarantees the loop makes progress.
//
// 5. `patches[index].Deepest` is written by ALL 32 lanes with no guard (:479),
//    while `.PointCount` is written by lane 0 only (:486-489).  Same value from
//    every lane, so the race is benign -- but PointCount IS written even when the
//    count is zero, and createNewPatches:955 depends on that to refuse to commit
//    an empty patch.
//
// 6. `minIndex(sep, currentContactMask, deepest)` (:478) DISCARDS its return value
//    and wants only the out-parameter.  Lanes outside the mask substitute FLT_MAX.
//    The comment at :479 explains why `deepest` is a local first: minIndex takes a
//    PxReal& and rewrites it through five shuffles, which must not be done on
//    shared memory.
//
// 7. In CUDA `pA` and `sep` are declared UNINITIALIZED (:445-446).  A lane that
//    never loads is never in currentContactMask, so the values are never read
//    through a live path -- but this port initialises them, because an
//    uninitialised read is not a behaviour a deterministic port can reproduce and
//    "never read" is a claim that has to stay true as the code is edited.
//
// ---------------------------------------------------------------------------
// THE INSTANTIATION
// ---------------------------------------------------------------------------
// :472 is contactReduce2<true, true, MESH_MESH_CONTACT_LIMIT, false>, i.e.
// <clustering=true, keepDeepestForPCM=true, maxPoints=6, dupeTest=false> reached
// through WRAPPER 2 -- distanceAdjustment = -separation (per lane) and
// initialPointCriteria = |point projected onto the normal plane|.  That is
// crsContactReduce2 with crsConfig(true, true, 6, false).
//
// Wrapper 2 was the whole reason P9.83b-2 existed: the four pre-existing
// transcriptions of contactReduceShared all collapse f and g onto one array,
// which is correct only while the adjustment is zero.  Under wrapper 2 it is not.

// ---------------------------------------------------------------------------
// THE SWEEP: 13 KILLED, 3 SURVIVE, 1 INERT
// ---------------------------------------------------------------------------
// The three survivors are recorded with the argument that survived an attempt to
// break it -- the attempt RUN, not imagined.  That rule is the one P9.83b-3 paid
// for: three mutants were called equivalent there and all three fell, twice with
// a written proof.
//
//   deepest-ignores-mask   drop the mask test in the `deepest` loop
//       PROVABLY EQUIVALENT, and not for the reason first guessed.  The first
//       guess was that lanes outside the mask hold CRS_FLT_MAX -- false: 195901
//       of 200000 measured cases have a lane outside the mask holding a FINITE
//       separation, because a lane culled on the final pass keeps the point it
//       loaded.  The real reason is that the deepest lane is ALWAYS a survivor:
//       contactReduction.cuh:126's PCM pick is `minIndex(separation, ...)` on the
//       RAW separation and always fires here (i0, i1, f and g are drawn from
//       allMask & ~mask in turn, so popc(mask) == 4 exactly, and maxPoints 6 > 4),
//       so the deepest lane is in the anchor set; :136 then gives every anchor the
//       clusterBias discount, so the deepest lane has the globally minimal sepB
//       and wins whichever cluster it lands in.  min over the mask therefore
//       equals min over everything.  Measured at bias 1e-5..5e-3 (the engine
//       passes 1e-5 * toleranceLength, PxgNarrowphaseCore.cpp:1371) and again at
//       0.01..0.30, which is the whole separation range: 0 differences in 400000
//       cases.
//
//   compact-descending     iterate lanes 31..0 instead of 0..31
//       PROVABLY EQUIVALENT.  The destination slot is
//       `countbits(mask & ((1 << l) - 1))` -- a pure function of the lane and the
//       mask, not of the iteration order.  Every lane writes the same slot either
//       way.  (The CUDA has no loop at all here: each lane computes its own offset
//       and stores, which is what makes order irrelevant in the first place.)
//
//   counter-incremented-at-end   `++counter2` -> no increment, so the 100-pass
//                                bound never trips
//       UNREACHABLE, not equivalent.  `c` advances by popcount(readMask) =
//       32 - popcount(currentContactMask) per pass, and the reduce caps the
//       survivor count at 6, so c gains at least 26 per pass after the first
//       (which gains 32).  totalContacts is at most 198 (the buffer bound the
//       CUDA asserts at :825), so the loop ends within 8 passes -- a factor of 12
//       below the bound.  maxCounter is load-bearing as a VALUE (maxcounter-one is
//       killed) and unreachable at 100.  It stays: it is the engine's, and a bound
//       that is unreachable today is reachable the moment the buffer grows.
//
//   readmask-is-hasmask    `~currentContactMask` -> `~hasMask`   (INERT)
//       The compiler emitted byte-identical SPIR-V, twice, including after a
//       forced rebuild with the .spv deleted.  That is a MACHINE-CHECKED proof of
//       the invariant this file's patchCreationMode argument rests on: hasMask and
//       currentContactMask are equal at the top of every pass.  Recorded as
//       evidence, not as a gap.
//
// AND ONE SURVIVOR WAS A REAL FIXTURE GAP: reduce-cfg-dupetest (TDoDupeTest false
// -> true) survived the first sweep because the staging buffer held no coincident
// points.  The real one does -- tryToAssignContactToExistingPatch copies the
// existing patch's KeepPoints into tempPointsS and then appends this round's
// contacts after them, so a contact that has barely moved lands beside its own
// kept copy at distance zero.  The fixture now plants them (991 of 1200 cases).

import pxgSlangContactReduceSerial;
import pxgSlangNpPatchTail;   // nptCombineMaterials -- ONE combine, two writers
import pxgSlangCommon;        // quatRotate

public static const uint TTP_MESH_MESH_CONTACT_LIMIT = 6u;
public static const uint TTP_MAX_MESH_MESH_PATCHES = 128u;

// convexMeshMidphase.cu:432-438.  116 bytes in CUDA (PxVec4T carries no alignment
// attribute, so Patch has alignment 4 and no tail padding); 128 bytes in SPIR-V
// Workgroup layout, where the float4 array forces 16-byte alignment on the struct
// and 116 rounds up.  The 12-byte difference is 1536 bytes over 128 patches and is
// the cheapest lever available if the 32 KiB budget ever needs one -- splitting
// pointCount into its own uint[128] array packs the rest to 112 and lands exactly
// on the CUDA's 14848.  Not done: the budget is measured at 30300 with headroom,
// and an unpacked struct is the one a reader can check against the CUDA.
public struct TtpPatch
{
    public float4 keepPoints[TTP_MESH_MESH_CONTACT_LIMIT];  // w = separation
    public float3 normal;
    public float deepest;
    public uint pointCount;
}

// ---------------------------------------------------------------------------
// reduceContactsSingleWarp (convexMeshMidphase.cu:440-496), serialised.
//
// `tempPoints` is the caller's staging buffer; `lane` arrays are local, so this
// runs on ONE thread and takes no barrier.  Returns the survivor mask; the caller
// stores it, because the two call sites write into different patch slots.
//
// The CUDA writes its results into patches[index] itself.  This returns them
// instead, so the caller decides where they land -- the two call sites write into
// different patch slots, and a body that reached into one fixed groupshared array
// could not serve both.
//
// THE REASON THIS NOTE USED TO GIVE WAS WRONG, AND IT WAS MEASURED, NOT ARGUED.
// It said a module that declares groupshared makes every importer pay for it, and
// that at 30300 of 32768 bytes the kernel would then not compile.  Both halves are
// false.  P9.83b-5 (10/n) put a 4096-byte groupshared array in THIS module behind a
// public function and compiled npTriTriWrite, which imports it:
//
//     nothing calls it            ttrScan 0      ttrWrite 0
//     ttrScan calls it            ttrScan 4096   ttrWrite 0
//
// Workgroup storage is charged PER ENTRY POINT BY REACHABILITY -- not per module,
// not even per file: two entries in ONE .spv pay different amounts for the same
// declaration.  That is the same finding P9.83b-1's checker made when it refuted
// the audit's 4096-byte projection, and it is restated here because this file is
// where the next person will look before deciding whether the block scan and the
// warp staging buffer may live in the module.  They may.
// ---------------------------------------------------------------------------
public struct TtpReduceResult
{
    public float4 keepPoints[TTP_MESH_MESH_CONTACT_LIMIT];
    public uint pointCount;
    public float deepest;
    public uint survivorMask;
}

public TtpReduceResult ttpReduceContactsSerial(float4 tempPoints[198], uint totalContacts,
                                               float3 nor, float clusterBias)
{
    float3 pA[32];
    float sep[32];
    // CUDA leaves these uninitialised (:445-446); see note 7 in the header.
    for (uint j = 0; j < 32u; ++j) { pA[j] = float3(0, 0, 0); sep[j] = CRS_FLT_MAX; }

    uint currentContactMask = 0u;
    uint hasMask = 0u;              // __ballot_sync(FULL_MASK, hasContact) (:471)

    const uint maxCounter = 100u;   // :451
    uint counter2 = 0u;
    const CrsConfig cfg = crsConfig(true, true, int(TTP_MESH_MESH_CONTACT_LIMIT), false);

    for (uint c = 0u; c < totalContacts && counter2 < maxCounter; )
    {
        ++counter2;                                     // :455, at the TOP

        const uint readMask = ~currentContactMask;      // :457, full 32-bit complement

        for (uint l = 0u; l < 32u; ++l)
        {
            // patchCreationMode is dead: both arms of :458 are this test.
            const bool needsContact = ((readMask >> l) & 1u) != 0u;
            // warpScanExclusive(readMask, l) = popcount of readMask below lane l
            const uint readIndex = c + countbits(readMask & ((1u << l) - 1u));
            if (needsContact && readIndex < totalContacts)   // :461, STRICT <
            {
                const float4 p = tempPoints[readIndex];
                pA[l] = p.xyz;
                sep[l] = p.w;
                hasMask |= 1u << l;                     // hasContact = true (:466)
            }
        }

        c += countbits(readMask);                       // :469, FREE LANES OFFERED

        currentContactMask = hasMask;                   // :471
        currentContactMask = crsContactReduce2(pA, sep, nor, currentContactMask,
                                               clusterBias, cfg);   // :472
        hasMask = currentContactMask;                   // :473
    }

    TtpReduceResult r;

    // :477-479.  Lanes outside the mask substitute FLT_MAX, which is what the
    // CRS_FLT_MAX initialisation above already gives them.
    float deepest = CRS_FLT_MAX;
    for (uint j = 0; j < 32u; ++j)
        if (((currentContactMask >> j) & 1u) != 0u && sep[j] < deepest)
            deepest = sep[j];
    r.deepest = deepest;

    r.pointCount = countbits(currentContactMask);       // :481
    r.survivorMask = currentContactMask;

    for (uint k = 0; k < TTP_MESH_MESH_CONTACT_LIMIT; ++k)
        r.keepPoints[k] = float4(0, 0, 0, 0);

    // :491-495.  Survivors compact in ASCENDING LANE ORDER; the scan here is over
    // currentContactMask, NOT over readMask as at :459.
    for (uint l = 0u; l < 32u; ++l)
    {
        if (((currentContactMask >> l) & 1u) == 0u) continue;
        const uint offset = countbits(currentContactMask & ((1u << l) - 1u));
        if (offset < TTP_MESH_MESH_CONTACT_LIMIT)
            r.keepPoints[offset] = float4(pA[l], sep[l]);
    }

    return r;
}

// ---------------------------------------------------------------------------
// THE TWO ACCEPTANCE TESTS, AND WHY THEY ARE NOT THE SAME TEST
// ---------------------------------------------------------------------------
// Both sites project the contact normal onto the patch normal and compare against
// two epsilons, and a transcription that treats them as one function gets the
// mesh-mesh manifold subtly wrong in two different ways.
//
//   tryToAssignContactToExistingPatch (:779-783)
//       PxReal proj = contact.normal.dot(patch.Normal);
//       bool accept2 = proj > PATCH_ACCEPTANCE_EPS2;      // 0.85
//       bool accept  = proj > PATCH_ACCEPTANCE_EPS3;      // 0.9995
//     UNGATED.  A retired contact has normal == (0,0,0) after markAsProcessed
//     (:760-765), so proj == 0 and both tests fail on their own.  The gate is the
//     zero normal, not an `if`.  contact.pos is NOT cleared by markAsProcessed and
//     is still handed to the reduce -- only the ballot keeps it out.
//
//   createNewPatches (:921-925)
//       if (contact.candidateContact)
//       {
//           PxReal proj = contact.normal.dot(patch.Normal);
//           accept2 = proj > PATCH_ACCEPTANCE_EPS2;
//           accept  = proj > PATCH_ACCEPTANCE_EPS3 || myNormal;
//       }
//     GATED, and `accept` carries an extra disjunct.  `myNormal` is true for the
//     single thread that donated this patch's normal at :908-912.  Without it a
//     patch could reject its own seed: proj there is |contact.normal|^2, and
//     contact normals are normalised only under `if (m > 0.0f)` (:1139-1144,
//     :1180-1185), so a degenerate normal gives proj <= 0.9995 -- and the seed is
//     the one contact that must be in the patch.
//
// THE ACCEPT/ACCEPT2 SPLIT IS NOT A REFINEMENT OF ONE TEST EITHER.  accept (tight)
// decides who enters the REDUCTION; accept2 (loose) decides who is CONSUMED --
// counted into nbUsed (:812) and retired by markAsProcessed (:845-846, :959-960).  So a
// contact whose normal is within 0.85 but not within 0.9995 of the patch normal is
// eaten and contributes nothing, deliberately: the comment at :785-786 says moving
// the definition to work on accept2 "will cause artificial friction".
//
// And the consumed set is strictly LARGER than that band: markAsProcessed has no
// threadEmitsElement guard, so a contact that passed the tight test, entered the
// ballot and was then culled by the reduce is also consumed.

public static const float TTP_PATCH_ACCEPTANCE_EPS2 = 0.85f;
public static const float TTP_PATCH_ACCEPTANCE_EPS3 = 0.9995f;

// nputils.cuh:152-163.  NOT crsArgMin: on an EMPTY mask CUDA returns
// lowestSetIndex(0) = __ffs(0) - 1 = 0xFFFFFFFF, and that value is load-bearing.
// At :908 it is compared against threadIndexInWarp, where 0xFFFFFFFF matches no
// lane and so nothing seeds the patch; crsArgMin's 0 would silently elect lane 0.
//
// UPSTREAM HAZARD, documented not fixed: the block-level result is used as an
// ARRAY INDEX at :902 (`countersS[minIBlock]`) with no guard.  It is 0xFFFFFFFF
// only when no lane satisfies `minV == v`, which needs NaN -- fminf propagates the
// non-NaN operand, so an all-NaN input leaves the ballot empty.  Reachable only
// with a NaN separation, which needs a NaN contact.
public uint ttpMinIndex(float v[32], uint mask, out float minV)
{
    minV = CRS_FLT_MAX;
    for (uint j = 0; j < 32u; ++j)
        if (((mask >> j) & 1u) != 0u && v[j] < minV) minV = v[j];

    uint ballot = 0u;
    for (uint j = 0; j < 32u; ++j)
        if (v[j] == minV) ballot |= 1u << j;
    ballot &= mask;

    return ballot == 0u ? 0xFFFFFFFFu : firstbitlow(ballot);
}

// tryToAssignContactToExistingPatch's tests (:779-783), one warp's 32 lanes.
public void ttpAcceptExisting(float3 cn[32], float3 patchNormal,
                              out uint acceptMask, out uint accept2Mask)
{
    acceptMask = 0u;
    accept2Mask = 0u;
    for (uint l = 0; l < 32u; ++l)
    {
        const float proj = dot(cn[l], patchNormal);
        if (proj > TTP_PATCH_ACCEPTANCE_EPS2) accept2Mask |= 1u << l;
        if (proj > TTP_PATCH_ACCEPTANCE_EPS3) acceptMask |= 1u << l;
    }
}

// createNewPatches' tests (:921-925), one warp's 32 lanes.  `seedLane` is the lane
// that donated the patch normal, or 0xFFFFFFFF if this warp did not donate it.
public void ttpAcceptNew(float3 cn[32], uint candidateMask, float3 patchNormal, uint seedLane,
                         out uint acceptMask, out uint accept2Mask)
{
    acceptMask = 0u;
    accept2Mask = 0u;
    for (uint l = 0; l < 32u; ++l)
    {
        if (((candidateMask >> l) & 1u) == 0u) continue;   // :921, the gate
        const float proj = dot(cn[l], patchNormal);
        if (proj > TTP_PATCH_ACCEPTANCE_EPS2) accept2Mask |= 1u << l;
        if (proj > TTP_PATCH_ACCEPTANCE_EPS3 || l == seedLane) acceptMask |= 1u << l;
    }
}

// createNewPatches' per-warp seed pick (:879-885).  Returns the lane index
// (0xFFFFFFFF when the warp has no candidate) and, through minSep, the warp's
// deepest candidate separation -- which is what goes into shMinSepS.
public uint ttpWarpSeed(float sep[32], uint candidateMask, out float minSep)
{
    return ttpMinIndex(sep, candidateMask, minSep);
}

// ---------------------------------------------------------------------------
// THE ACCEPTANCE SWEEP: 11 KILLED, 2 SURVIVE -- AND BOTH SURVIVORS ARE EVIDENCE
// ---------------------------------------------------------------------------
// The first run killed 7 of 13.  Four of the six survivors were thresholds with
// no input near them, which is the same failure in four costumes:
//
//   eps3-value               no contact projected between 0.999 and 0.99968, so
//                            0.9995 and 0.999 classified every lane alike
//   eps-nonstrict-existing   `>` vs `>=` needs proj EXACTLY 0.9995f
//   minindex-highest-tie     needs two MASKED lanes sharing the minimum
//   minindex-ballot-unmasked needs an UNMASKED lane at a LOWER index equal to the
//                            masked minimum
//
// All four are now planted and all four are dead.  Random floats produce none of
// them, which is the general lesson: a comparison is tested by inputs ON it, and
// a uniform sample is a sample of everywhere except the boundary.
//
// THE TWO THAT REMAIN ARE EQUIVALENT, and each confirms something this file says.
//
//   existing-form-gated   skip lanes whose normal is exactly zero
//       EQUIVALENT, and that is the point.  A retired contact's normal is
//       PxVec3(0.f) (markAsProcessed, :760-765), so proj == 0 and it fails both
//       `> 0.85` and `> 0.9995` on its own.  Adding the guard changes nothing --
//       which is exactly the header's claim that the zero normal IS the gate at
//       :779-783, now checked by mutation rather than asserted.
//       (The FIRST version of this mutant was a no-op dressed as a test: it
//       skipped lanes already in acceptMask, and lane l's bit is only set at
//       iteration l.  A mutant that cannot change anything proves nothing by
//       surviving, and it sat in the results for a full run looking like a gap.)
//
//   minindex-empty-returns-zero   drop the `ballot == 0 ? 0xFFFFFFFF :` guard
//       EQUIVALENT ON THIS TOOLCHAIN, proven by the fixture's own coverage: it
//       has 118 cases with an empty candidate mask that expect 0xFFFFFFFF, and
//       they pass, so Slang's firstbitlow(0) already returns 0xFFFFFFFF.
//       THE TERNARY STAYS.  The value is a contract the CUDA depends on at :908,
//       where it is compared against a lane index and must match none; leaving
//       that to an intrinsic's corner case is how a compiler update silently
//       elects lane 0 to seed a patch.
//
// P9.83b-8 T1 ADDED TWO, AND ONE OF THEM FOUND A GAP.  PxVec3::dot is
// (x*x' + y*y') + z*z'; this file writes dot(), i.e. OpDot, whose order is the
// driver's.  The exact-epsilon lane has ONE nonzero product, so the order was
// invisible, and `accept-existing-dot-rl` (x + (y + z)) survived.  The fixture now
// plants, in one case in eight, a lane whose projection falls on OPPOSITE sides of
// EPS3 in the two orders and another for EPS2 (470 lanes, found by stepping one
// component an ulp at a time).  The mutant is killed, and
//
//   accept-existing-dot-lr   dot() spelled out in the CUDA's order
//       EQUIVALENT, MEASURED: the port calls dot() and passes on all 470 lanes,
//       so this driver evaluates OpDot as (x + y) + z -- the CUDA's order.  That
//       is a property of the OpDot lowering, not of this call site, so it holds
//       wherever the tri-tri modules call dot() (see pxgSlangContactReduceSerial's
//       cluster-dot entries).


// ---------------------------------------------------------------------------
// createPatchWriterFullThreadBlock (:504-641): THE BLOCK-WIDE RESERVATION
// ---------------------------------------------------------------------------
// This is NOT the tail pxgSlangNpPatchTail already has.  That one
// (nptReserveContacts/nptRegisterContactPatch) reserves for ONE pair with ONE
// patch, and its overflow test is deliberately `offset + ONE RECORD > limit` --
// a faithful upstream quirk its own header records.  This one reserves N PATCHES
// and M CONTACTS for a whole thread block in three atomics, and tests the WHOLE
// RESERVATION.  Two tails in one engine with different overflow semantics;
// documented, not reconciled.
//
// THREE THINGS A TRANSCRIPTION GETS WRONG HERE.
//
// 1. THE CHECKS ARE 64-BIT, THE RESERVATIONS ARE NOT.  `sizeof(...)` is size_t,
//    so `patchByteOffsetS + sizeof(PxContactPatch) * numPatchesToKeep >
//    patchBytesLimit` (:560) promotes the whole comparison to 64 bits and cannot
//    wrap.  The atomicAdd arguments are the same expressions narrowed back to
//    unsigned by the atomicAdd signature, so the ADD is 32-bit and wraps.  Doing
//    the comparison in 32-bit -- the natural choice in Slang -- changes behaviour
//    once an offset is large.
//
// 2. THE FORCE CHECK RESERVES N AND TESTS 2N.  :558 adds
//    `sizeof(PxU32) * totalNumContacts`; :573 tests
//    `offset + sizeof(PxU32) * totalNumContacts * 2 > forceBytesLimit`.  The
//    buffer is "forceAndIndice", so the doubling is presumably the indices --
//    but only one stride is ever reserved.  Transcribed as written.
//
// 3. THE ZERO-CONTACT PATH PUBLISHES base + 0xFFFFFFFF, NOT NULL.  The three
//    atomics are gated on `if (totalNumContacts)` (:554), so a block that
//    produced nothing leaves all three offsets at their 0xFFFFFFFF init AND
//    leaves `overflow` false -- and thread 0 then takes the `!overflow` arm
//    (:607) and writes `startContactPoints + 0xFFFFFFFF` into the output.  The
//    OVERFLOW arm writes zeros.  Harmless only because nbContacts is 0 and
//    eHAS_NO_TOUCH is set, and asymmetric with the returned writer, whose members
//    DO become null for the same sentinel (:625-626).
//    A port will "fix" this to null without noticing.  This one does not.
//
// The atomic RESULTS are parameters rather than performed here, so the decision
// logic is testable without a race -- the same reason the reduce took lane arrays.

public struct TtpReservation
{
    public uint patchByteOffset;     // 0xFFFFFFFF when unreserved or overflowed
    public uint contactByteOffset;
    public uint forceByteOffset;
    public uint numPatchesOut;       // zeroed by overflow (:582)
    public uint numContactsOut;
    public uint statusFlags;
    public uint allFlagsWord;
    public uint touchChange;
    public uint patchChange;
    public uint overflowBits;        // which limits were hit
    public bool overflow;
    public bool writerPatchesNull;   // the :625-626 asymmetry, made explicit
    public bool writerContactsNull;
}

public static const uint TTP_PATCH_SIZE = 64u;    // sizeof(PxContactPatch)
public static const uint TTP_CONTACT_SIZE = 16u;  // sizeof(PxContact)
public static const uint TTP_FORCE_SIZE = 4u;     // sizeof(PxU32)

public static const uint TTP_FLAG_HAS_NO_TOUCH = 1u;
public static const uint TTP_FLAG_HAS_TOUCH = 2u;
public static const uint TTP_FLAG_TOUCH_KNOWN = 3u;

public static const uint TTP_OVERFLOW_PATCH = 4u;
public static const uint TTP_OVERFLOW_CONTACT = 1u;
public static const uint TTP_OVERFLOW_FORCE = 2u;

public TtpReservation ttpReserveBlock(uint oldAllFlags, uint numPatchesToKeep, uint totalNumContacts,
                                      uint patchAdd, uint contactAdd, uint forceAdd,
                                      uint patchBytesLimit, uint contactBytesLimit,
                                      uint forceBytesLimit)
{
    TtpReservation r;
    r.patchByteOffset = 0xFFFFFFFFu;      // :525-529
    r.contactByteOffset = 0xFFFFFFFFu;
    r.forceByteOffset = 0xFFFFFFFFu;
    r.overflowBits = 0u;
    r.overflow = false;

    // :537 u16Low(u32High(allflags)) = byte 2, the status byte.
    // :547 u16High(u32Low(allflags)) = byte 1 -- nbPatches, the pair's CURRENT
    // patch count, which the store below carries forward into byte 3 as next
    // frame's prevPatches (the sibling writers say so: cudaGJKEPA.cu:2076,
    // contactPatchUtils.cuh:91).  P9.83b-8 T0 FIXED THIS: the first port read byte
    // 3 (`>> 24`), the stored prevPatches itself, so prevPatches never advanced
    // and patchChange compared against the wrong count -- and the oracle read the
    // same wrong byte, which is why the reservation sweep could not see it.
    const uint oldStatusFlags = (oldAllFlags >> 16u) & 0xFFu;
    const uint prevPatches = ((oldAllFlags & 0xFFFFu) >> 8u) & 0xFFu;

    // :540-545.  The mask clears ONLY bits 0-1; eREQUEST_CONSTRAINTS,
    // eHAS_CCD_RETOUCH, eDIRTY_MANAGER and eSTATIC_OR_KINEMATIC ride through.
    uint statusFlags = oldStatusFlags & (~TTP_FLAG_TOUCH_KNOWN);
    if (numPatchesToKeep != 0u) statusFlags |= TTP_FLAG_HAS_TOUCH;
    else                        statusFlags |= TTP_FLAG_HAS_NO_TOUCH;

    uint nPatches = numPatchesToKeep;
    uint nContacts = totalNumContacts;

    if (totalNumContacts != 0u)           // :554 -- the gate
    {
        r.patchByteOffset = patchAdd;
        r.contactByteOffset = contactAdd;
        r.forceByteOffset = forceAdd;

        // 64-BIT, as the CUDA's size_t arithmetic makes them.
        if (uint64_t(r.patchByteOffset) + uint64_t(TTP_PATCH_SIZE) * uint64_t(numPatchesToKeep)
            > uint64_t(patchBytesLimit))
        {
            r.overflowBits |= TTP_OVERFLOW_PATCH;
            r.patchByteOffset = 0xFFFFFFFFu;
            r.overflow = true;
        }
        if (uint64_t(r.contactByteOffset) + uint64_t(TTP_CONTACT_SIZE) * uint64_t(totalNumContacts)
            > uint64_t(contactBytesLimit))
        {
            r.overflowBits |= TTP_OVERFLOW_CONTACT;
            r.contactByteOffset = 0xFFFFFFFFu;
            r.overflow = true;
        }
        // THE *2 IS THE CUDA'S (:573) -- it reserves one stride and tests two.
        if (uint64_t(r.forceByteOffset) + uint64_t(TTP_FORCE_SIZE) * uint64_t(totalNumContacts) * 2UL
            > uint64_t(forceBytesLimit))
        {
            r.overflowBits |= TTP_OVERFLOW_FORCE;
            r.forceByteOffset = 0xFFFFFFFFu;
            r.overflow = true;
        }

        if (r.overflow)                   // :580-586
        {
            nPatches = 0u;
            nContacts = 0u;
            statusFlags = (statusFlags & (~TTP_FLAG_TOUCH_KNOWN)) | TTP_FLAG_HAS_NO_TOUCH;
        }
    }

    // :592-600
    const bool previouslyHadTouch = (oldStatusFlags & TTP_FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & TTP_FLAG_TOUCH_KNOWN) != 0u;
    const bool currentlyHasTouch = nPatches != 0u;
    r.touchChange = ((previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown)) ? 1u : 0u;
    r.patchChange = (prevPatches != nPatches) ? 1u : 0u;

    // :603-605.  merge(merge(prevPatches, statusFlags), merge(numPatchesToKeep, 0))
    r.allFlagsWord = (prevPatches << 24u) | (statusFlags << 16u) | (nPatches << 8u);

    r.statusFlags = statusFlags;
    r.numPatchesOut = nPatches;
    r.numContactsOut = nContacts;

    // :624-626.  The writer is null on the SENTINEL, which the zero-contact path
    // also produces -- while the output pointers above took the !overflow arm.
    r.writerPatchesNull = (r.patchByteOffset == 0xFFFFFFFFu);
    r.writerContactsNull = (r.contactByteOffset == 0xFFFFFFFFu);

    return r;
}

// ---------------------------------------------------------------------------
// THE RESERVATION SWEEP: 16 KILLED, 0 SURVIVE -- AND A DEFECT STILL SHIPPED
// ---------------------------------------------------------------------------
// Clean, on the second run.  The first found one survivor -- patch-check-nonstrict
// (`>` vs `>=`) -- because the fixture's limits were always at least one byte off
// `offset + reservation`, and that single point is the only place the two
// spellings differ.  One limit in four now lands exactly on it, and the contact
// and force checks got the same mutant rather than only the one that happened to
// be noticed.
//
// A clean sweep did not make the body right.  prevPatches read byte 3 in the port
// AND in its oracle, so every mutant was judged against a reference that shared
// the defect; a sweep measures the fixture's power to tell the port from its
// oracle, never the oracle's fidelity to the CUDA.  The critic pass of P9.83b-8
// found it by re-reading :547 against the sibling writers.  After the fix (T0) the
// oracle spells the read with the CUDA's own u16High(u32Low()), the fixture puts
// byte 1 ON the new count (patchChange = 0, which only byte 1 produces) and byte
// 3 on it elsewhere, and a mini sweep of four wrong-byte reads -- byte 3 restored,
// byte 0, byte 2, bytes 1-2 as 16 bits -- killed all four.
//
// AND A TRAP THIS FILE NAMED WAS NEVER EXERCISED (P9.83b-8 T2).  Note 1 above says
// the checks are 64-bit; every offset the fixture drew kept offset + reservation
// far below 2^32, so a 32-bit check agreed with the CUDA everywhere.  One case in
// sixteen per stream now sits within one reservation of 2^32, and the three
// `check-32bit-*` mutants are killed.  The block-tail fixture reaches the same
// boundary through real counters, where the counter itself wraps.
//
// ---------------------------------------------------------------------------
// A STANDING RULE, PAID FOR FOUR TIMES IN THIS TRANCHE
// ---------------------------------------------------------------------------
// Every one of these was the same defect wearing a different hat -- a comparison
// with no input ON it:
//
//   eps3-value               no projection between 0.999 and 0.99968
//   eps-nonstrict-existing   no projection exactly 0.9995f
//   minindex-highest-tie     no two masked lanes sharing the minimum
//   minindex-ballot-unmasked no unmasked lane equal to the masked minimum
//   patch-check-nonstrict    no limit exactly on offset + reservation
//
// A uniform sample covers everywhere except the place a comparison is decided,
// because the decision boundary is a measure-zero set and random inputs miss it
// with probability one.  So: FOR EVERY COMPARISON IN A PORTED BODY, THE FIXTURE
// MUST CONTAIN AN INPUT THAT SITS ON IT -- constructed, not sampled.  Strict vs
// non-strict needs equality; an argmin tie-break needs a tie; a threshold needs a
// value between it and its plausible neighbours.
//
// This is not a checker, because nothing static can see whether a fixture's
// generator reaches a boundary.  It is a rule for whoever writes the next one.

// ---------------------------------------------------------------------------
// writeContactPatch (:649-672): the 64-byte header, and the two words the CUDA
// fills from an UNINITIALISED struct.
// ---------------------------------------------------------------------------
// THE LAYOUT, as words (PxContact.h), identical to the one
// nptInsertIntoPatchStream already writes:
//
//   0-3   mMassModification, all 1.0f
//   4-6   normal
//   7     restitution      8  dynamicFriction    9  staticFriction   10 damping
//   11    startContactIndex (bits 0-15) | nbContacts (16-23) | materialFlags (24-31)
//   12    internalFlags (0-15) | materialIndex0 (16-31)
//   13    materialIndex1 (0-15) | pad[0] (16-31)
//   14-15 pad[1..4] -- NEVER WRITTEN by either writer
//
// TWO DIFFERENCES FROM THE SIBLING, and one of them is a defect.
//
// 1. startContactIndex IS REAL HERE.  contactPatchUtils.cuh:163 writes 0 and
//    nptInsertIntoPatchStream reproduces that by leaving word 11's low half
//    clear.  convexMeshMidphase.cu:656 writes `patchStartIndexS[threadIdx.x]`,
//    the block-wide scan of per-patch contact counts.  The narrowing to PxU16 is
//    the CUDA's (:657) and is safe here: numPatchesToKeep <= 128 and each
//    PointCount <= 6, so the scan cannot exceed 768.
//
// 2. materialIndex0 AND materialIndex1 ARE READ FROM AN UNINITIALISED LOCAL.
//    :728 declares `PxgContactMaterialProperties contactMaterialProperties;`
//    with no initialiser.  getContactMaterialProperties (:642-647) forwards to
//    combineMaterials(materials, shape0.materialIndex, shape1.materialIndex,
//    materialFlags, staticFriction, dynamicFriction, combinedRestitution,
//    damping) -- eight parameters, NONE of them an output material index.  Then
//    :662-663 store cmp.materialIndex0 and cmp.materialIndex1 into the patch.
//    Every contact patch this kernel emits carries two indeterminate words.
//
//    The sibling gets it right: contactPatchUtils.cuh:149-150 reads them from the
//    shapes and :170-171 stores them, which is what makes this a defect rather
//    than a convention.
//
//    THIS PORT WRITES THE WELL-DEFINED VALUES -- materialIndex0 =
//    shape0.materialIndex, materialIndex1 = shape1.materialIndex -- following the
//    npTriHfNp precedent for upstream undefined behaviour: a deterministic port
//    cannot reproduce an uninitialised read, so it implements the reading the
//    rest of the engine assumes and says so loudly.  The solver consumes these
//    indices; emitting garbage deterministically would be no better than
//    emitting it nondeterministically.
//
// The combine itself is nptCombineMaterials, shared with the sibling writer
// rather than transcribed again.

public void ttpWriteContactPatch(uint64_t patchStreamAddr, uint patchByteOffset,
                                 uint patchIndex, uint startContactIndex,
                                 float3 patchNormal, uint nbContactPoints,
                                 uint64_t materialsAddr, uint materialIndex0, uint materialIndex1)
{
    if (patchByteOffset == 0xFFFFFFFFu)   // :651, `if (writer.patches)`
        return;

    const NptCombinedMaterial cm = nptCombineMaterials(materialsAddr, materialIndex0, materialIndex1);

    Ptr<uint> w = Ptr<uint>(patchStreamAddr + uint64_t(patchByteOffset)
                            + uint64_t(patchIndex) * uint64_t(TTP_PATCH_SIZE));

    const uint one = asuint(1.0f);
    w[0] = one; w[1] = one; w[2] = one; w[3] = one;   // :665-668
    w[4] = asuint(patchNormal.x);
    w[5] = asuint(patchNormal.y);
    w[6] = asuint(patchNormal.z);
    w[7] = asuint(cm.restitution);
    w[8] = asuint(cm.dynamicFriction);
    w[9] = asuint(cm.staticFriction);
    w[10] = asuint(cm.damping);
    w[11] = (startContactIndex & 0xFFFFu)
          | ((nbContactPoints & 0xFFu) << 16u)
          | ((cm.materialFlags & 0xFFu) << 24u);
    // internalFlags = 0 (:669), materialIndex0 in the high half.
    w[12] = (materialIndex0 & 0xFFFFu) << 16u;
    w[13] = (materialIndex1 & 0xFFFFu);
    // w[14], w[15] are pad[1..4] and are NOT written, matching both CUDA writers.
}


// ---------------------------------------------------------------------------
// writePatchesAndContacts (convexMeshMidphase.cu:673-751): the tail that joins
// the reservation, the header write and the point write.
// ---------------------------------------------------------------------------
// FOUR THINGS IN THE CUDA ARE NOT OBVIOUS, AND ALL FOUR ARE TRANSCRIBED HERE.
//
// 1. THE SCAN IS OVER THE BLOCK, NOT OVER THE PATCHES.
//    :706 runs threadBlockScanExclusive<NbWarps> over every thread, with
//    count = 0 for threadIdx.x >= numPatchesToKeep (:699-701).  A zero-padded
//    exclusive scan has the same prefix on its live prefix as a scan of the live
//    entries alone, so ttpPatchStartIndices computes the short one and returns the
//    same totalSum.  Entries at or past numPatchesToKeep hold totalSum and are
//    never read -- both write loops are guarded by `< numPatchesToKeep`.
//    The CUDA stores only the first MAX_MESH_MESH_PATCHES2 lanes (:707-708); the
//    rest of the block's ids are discarded, which is why the padding is free.
//
// 2. THE POINT WRITE HAS NO LOOP, SO IT IS A blockDim CONSTRAINT.
//    :736 maps thread -> patch as `threadIdx.x / MESH_MESH_CONTACT_LIMIT` and
//    stops.  That covers blockDim/6 patches.  The engine launches this kernel
//    with 1024 threads, so it covers 170 >= MAX_MESH_MESH_PATCHES2 = 128 and is
//    exactly safe -- but only there.  A workgroup below 768 still writes every
//    patch HEADER (:727 is guarded by numPatchesToKeep <= 128) while DROPPING THE
//    POINTS of every patch at or past blockDim/6: a manifold that claims six
//    contacts and supplies none.  This would be silent.
//    The serial port loops over numPatchesToKeep instead, so it has no dependency
//    on a workgroup size at all.  It agrees with the CUDA exactly at the engine's
//    launch and is a superset below it.  That is the one place this file
//    deliberately does not reproduce a thread mapping, and the reason is that the
//    mapping encodes a launch assumption rather than a decision.
//
// 3. THE TWO WRITES ARE INDEPENDENTLY GATED, AND THE GATES CAN DISAGREE.
//    writeContactPatch returns on a null patch pointer (:651); the point write
//    tests `if (patchWriter.contacts)` at :746 -- the INNERMOST guard, inside both
//    the patch-index and the point-index tests.  ttpReserveBlock sentinels each
//    stream's offset on its own limit, so a patch-buffer overflow alone writes
//    POINTS WITH NO HEADERS and a contact-buffer overflow alone writes HEADERS
//    THAT POINT AT NOTHING.  Neither arm suppresses the other, and the overflow
//    flag that zeroes the COUNTS (:580-586) does not reach either pointer.
//
// 4. THE MATERIAL SWAP UNDOES THE KERNEL'S SWAP.
//    :1365 calls doTriangleTriangleCollision(transform1, transform0, mesh1, mesh0,
//    shape1, shape0, ..., isFlipped = true) -- the arguments are already
//    exchanged, so inside this function `shape0` IS the engine's shape1.  :729
//    then asks for `isFlipped ? shape1 : shape0` as the first material, which in
//    the flipped arm is the local shape1, i.e. the engine's shape0.  The pair
//    handed to getContactMaterialProperties is therefore in ENGINE order on both
//    arms; the swap exists to cancel the one at the call site, not to express a
//    preference.  Porting only one of the two swaps silently exchanges dynamic
//    and static friction on exactly half the mesh-mesh pairs.
//    The normal's `isFlipped ? normal : -normal` (:733) is NOT the same cancelling
//    -- the patch normal is computed in the local mesh1 frame in both arms and
//    must end up pointing from the engine's shape0 to its shape1, so one arm
//    negates.  Two conditionals on the same predicate, doing different jobs.

// ---------------------------------------------------------------------------
// THE WRITE SWEEP: 29 MUTANTS, 29 KILLED, 0 SURVIVE -- AND ONE OF THE KILLS IS
// NOT EVIDENCE
// ---------------------------------------------------------------------------
// 28 fell on the fixture's oracle.  The 29th, `contact-guard-is-patch-guard`,
// SEGFAULTED on all four attempts, and that is a kill the fixture did not earn:
// the mutant replaces the contact sentinel test with the patch one, so a case with
// a live patch offset and a dead contact offset writes to
// contactStreamAddr + 0xFFFFFFFF.  The crash comes from the ADDRESS, not from the
// comparison.  By the rule this tranche wrote down -- a mutant whose mutated form
// is undefined behaviour tests nothing -- its verdict says nothing either way, and
// it is listed here so the 29 is not read as 29 pieces of evidence.
//
// THE NARROW DIRECTION OF THAT GUARD CANNOT BE TESTED WITHOUT UB, and that is a
// property of the code, not of the fixture: the sentinel IS an address that must
// not be dereferenced, so every mutant that lets one through is undefined.  What
// can be tested is the WIDE direction -- a guard that fires when it should not --
// and `contact-guard-either` (skip the points whenever the PATCH stream is dead)
// is killed by the 375 patch-sentinel-only cases.  Between the two, what is
// established is that the guard's predicate is the contact sentinel alone.
//
// The mutants worth naming among the 28:
//
//   swap-dropped / swap-always / swap-half   the :729 material exchange, in its
//       three broken forms.  All three are killed ONLY by header words 12 and 13,
//       because every branch of the combine is symmetric in its arguments -- see
//       the note at the head of testNpTriTriWrite.cpp.
//   normal-translated                        rotate-and-translate where the CUDA
//       rotates; the mirror of point-not-translated, which does the reverse.  Both
//       are killed, which is what a non-zero translation buys.
//   writeindex-from-patch-index              patchIndex * 6 + idx instead of the
//       scanned start.  Killed by the 2415 cases whose start index is not a
//       multiple of six; with a fixture that scanned a full manifold every time it
//       would have survived.
//   pad14-written                            one extra word into pad[1..4].  Killed
//       only because the fixture compares the WHOLE stream against a poison fill,
//       not just the words it expects.

// The zero-padded block scan of 1, reduced to its live prefix.  Returns the total.
public uint ttpPatchStartIndices(uint pointCounts[TTP_MAX_MESH_MESH_PATCHES],
                                 uint numPatchesToKeep,
                                 out uint startIndex[TTP_MAX_MESH_MESH_PATCHES])
{
    uint running = 0u;
    for (uint i = 0u; i < TTP_MAX_MESH_MESH_PATCHES; ++i)
    {
        startIndex[i] = running;
        if (i < numPatchesToKeep)
            running += pointCounts[i];
    }
    return running;
}

// One patch: the 64-byte header (:727-734) and its up-to-six points (:736-750).
// `rotQ`/`rotP` are the LOCAL transform1 -- the second argument as this function
// sees it, which the kernel may already have exchanged.  See note 4.
public void ttpWritePatchAndContacts(uint64_t patchStreamAddr, uint patchByteOffset,
                                     uint64_t contactStreamAddr, uint contactByteOffset,
                                     uint64_t materialsAddr,
                                     uint patchIndex, uint patchStartIndex,
                                     TtpPatch p,
                                     float4 rotQ, float3 rotP, bool isFlipped,
                                     uint shapeMaterialIndex0, uint shapeMaterialIndex1)
{
    // :729 -- the swap that cancels the kernel's.
    const uint mi0 = isFlipped ? shapeMaterialIndex1 : shapeMaterialIndex0;
    const uint mi1 = isFlipped ? shapeMaterialIndex0 : shapeMaterialIndex1;

    // :732-733.  rotate only: a normal carries no translation.
    const float3 normal = quatRotate(rotQ, p.normal);
    ttpWriteContactPatch(patchStreamAddr, patchByteOffset,
                         patchIndex, patchStartIndex,
                         isFlipped ? normal : -normal, p.pointCount,
                         materialsAddr, mi0, mi1);

    if (contactByteOffset == 0xFFFFFFFFu)   // :746, `if (patchWriter.contacts)`
        return;

    Ptr<float4> c = Ptr<float4>(contactStreamAddr + uint64_t(contactByteOffset));

    const uint n = min(p.pointCount, TTP_MESH_MESH_CONTACT_LIMIT);
    for (uint idx = 0u; idx < n; ++idx)     // :741, `if (idx < patch.PointCount)`
    {
        const uint writeIndex = patchStartIndex + idx;      // :743
        const float4 pt = p.keepPoints[idx];
        // :745 -- full transform here, unlike the normal.
        const float3 pos = quatRotate(rotQ, pt.xyz) + rotP;
        // :748 -- separation rides through untouched.
        c[writeIndex] = float4(pos, pt.w);
    }
}
