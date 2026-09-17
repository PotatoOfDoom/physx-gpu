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
//   (:440-503), here as a SERIAL body.
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

import pxgSlangContactReduceSerial;

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
// reduceContactsSingleWarp (convexMeshMidphase.cu:440-503), serialised.
//
// `tempPoints` is the caller's staging buffer; `lane` arrays are local, so this
// runs on ONE thread and takes no barrier.  Returns the survivor mask; the caller
// stores it, because the two call sites write into different patch slots.
//
// The CUDA writes its results into patches[index] itself.  This returns them
// instead: a module that reached into the caller's groupshared would have to
// declare that groupshared, and every importer would then pay for it whether it
// used the machinery or not.  P9.83b-4 measured the budget at 30300 of 32768 --
// 2468 bytes of headroom -- so a module that hands out Workgroup storage is a
// module that decides the kernel cannot be compiled.
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
