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

// npCvxTriEmit.slang
//
// Slang port of the contact-reduce-and-emit stage of convexTriangleContactGen
// (convexTriangle.cuh :1769-1821) — tranche 112.  This is the warp-SIMD stage
// that consumes the raw contacts produced by polyClip (tranche 111, held one per
// lane in s_contactsTransposed) and turns them into the compacted temp-contact
// stream + the (maxSeparation, nbOutputContacts, startIndex) triple that the
// tranche-110 output stage (npConvexTriOut) then writes out.  Concretely:
//
//   mask = (1 << numContactsGenerated) - 1
//   if (numContactsGenerated > CVX_TRI_MAX_CONTACTS)                 // = 5
//       mask = contactReduce<false, true, 5, true>(pa, sep, minNormal, mask, 0);
//   maxSeparation = warpReduction<MinOpFloat>( mask&(1<<lane) ? sep : FLT_MAX );
//   nbOutputContacts = __popc(mask);
//   if (lane == 0) { startIndex = atomicAdd(pTempContactIndex, nbOutputContacts);
//                    cvxTriContacts[pairOffset].index = startIndex; }
//   startIndex = shfl(startIndex, 0);
//   if (startIndex + nbOutputContacts < tempContactSize)
//       if (mask & (1<<lane))
//           tempConvexTriContacts[startIndex + warpScanExclusive(mask, lane)]
//               = float4(pa, sep);
//   else nbOutputContacts = 0;
//
// contactReduce<false, true, 5, true> is the exact reduction already verified in
// tranche 82 (contactReduction.cuh): a duplicate cull, then p0 = deepest,
// p1 = farthest from p0, the two extremes along cross(normal, p1-p0), one extra
// deepest contact for PCM when exactly 4 anchors were picked, then fill the rest
// with the deepest unselected — all selecting the LOWEST lane on ties.  The
// reduction only chooses a mask; the emitted (pos, sep) are verbatim copies of
// the inputs, so once the mask matches, the whole stream matches bit-for-bit.
//
// Equivalent mutants under THIS instantiation (clustering off, keepPCM on,
// maxPoints = 5): once the dupe cull leaves >5 points, {i0, i1, f, g} are always
// four DISTINCT lanes (i0 != i1; f, g are drawn from allMask\{i0,i1} with g
// excluding f), so popc == 4 always holds after the anchors, the PCM step always
// fires, and the fill loop never runs.  Hence (a) swapping the cross-product
// operands or the f/g max/min roles leaves the SET {f, g} — the two extremes —
// unchanged; (b) shifting the PCM count test (== 4 vs == 3) only moves one
// deepest-unselected pick between the PCM step and the fill loop, which select
// identically; and (c) the fill-loop direction is dead here.  A 2,000,000-config
// brute force confirms all four produce byte-identical masks; the general
// clustering / other-maxPoints / fill paths are exercised by tranche 82.
//
// Serialisation: one thread simulates the whole 32-lane warp (per-lane arrays).
// The atomic reservation is modelled per test by its own counter slot preset to
// the reservation base, so startIndex is deterministic while the reserve-then-
// advance semantics (and the tempContactSize overflow guard that zeroes the
// OUTPUT count but keeps the reservation) are faithful.  maxSeparation is a min
// over the selected lanes' seps; warpScanExclusive(mask, lane) = popc of the
// mask bits below `lane`, matching the simultaneous warp scan.
//
// Determinism: integer 2D points with an axis-aligned minNormal make the
// projection point[j] = ptRaw[j] - n*(ptRaw[j].n) exact and every dot/cross
// integer-exact; distinct separations and distinct pairwise squared distances
// keep every argmax/argmin (including the sqrt-based farthest pick, monotone in
// the exact squared distance) unambiguous, so the selected mask — hence the
// emitted stream — is bit-identical on CPU and GPU.

import pxgSlangCommon;
import pxgSlangContactReduceSerial;

static const uint CVX_TRI_MAX_CONTACTS = 5u;
static const float CTE_MAX_F32 = 3.402823466e38f;

struct CteParams
{
    uint numTests;
    uint cap;   // per-test stride (in float4s) of the temp-contact output block
    uint pad0, pad1;
}

struct CteIn
{
    float4 minNormal; // xyz = contact normal (w unused)
    uint4 cfg;        // x = numContactsGenerated, y = tempContactSize, z/w unused
}

[[vk::push_constant]]
ConstantBuffer<CteParams> gParams;

RWStructuredBuffer<float4> gContacts; // binding 0: 32 raw contacts/test (pos.xyz, sep)
RWStructuredBuffer<CteIn> gIn;        // binding 1
RWStructuredBuffer<float4> gTemp;     // binding 2: compacted temp contacts (cap/test)
RWStructuredBuffer<uint> gCounter;    // binding 3: reservation counter/test (preset to base)
RWStructuredBuffer<uint4> gOut;       // binding 4: x=nbOut, y=index, z=asuint(maxSep), w=0

uint warpScanExclusive(uint mask, uint lane) { return countbits(mask & ((1u << lane) - 1u)); }

// contactReduce<clustering=false, keepAnotherDeepestForPCM=true, maxPoints=5,
// dupeTest=true>(pointRaw, separation, normal, allMask, 0) -- distanceAdjustment
// = 0, initial criterion = separation (contactReduction.cuh :172).
//
// P9.83b-3: THE BODY MOVED OUT to pxgSlangContactReduceSerial.  This is the
// clustering=FALSE instantiation, and folding it is what forced the module to
// grow CUDA's `else` tail (contactReduction.cuh:159-165): the module had an `if
// (cfg.clustering)` with no else, which looked complete only because every
// caller folded in before this one clusters.  Folding this body onto the module
// as it stood would have silently dropped the fill-out loop and emitted a short
// manifold -- the shared body less complete than the copy it replaced.
uint contactReduce5(float3 ptRaw[32], float sep[32], float3 normal, uint allMask)
{
    return crsContactReduce(ptRaw, sep, normal, allMask, 0.0f,
                            crsConfig(false, true, int(CVX_TRI_MAX_CONTACTS), true));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctCvxTriEmit(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const CteIn e = gIn[t];
    const uint numContactsGenerated = e.cfg.x;
    const uint tempContactSize = e.cfg.y;
    const float3 minNormal = e.minNormal.xyz;
    const uint cbase = t * 32u;

    float3 pa[32];
    float sep[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float4 c = gContacts[cbase + L];
        pa[L] = c.xyz;
        sep[L] = c.w;
    }

    float maxSeparation = CTE_MAX_F32;
    uint nbOutputContacts = 0u;
    uint index = 0xffffffffu; // cvxTriContacts[pairOffset].index — undefined when no contacts

    if (numContactsGenerated != 0u)
    {
        uint mask = (numContactsGenerated >= 32u) ? 0xffffffffu : ((1u << numContactsGenerated) - 1u);

        if (numContactsGenerated > CVX_TRI_MAX_CONTACTS)
            mask = contactReduce5(pa, sep, minNormal, mask);

        // maxSeparation = min over selected lanes' seps (warpReduction<MinOp>)
        for (uint L = 0u; L < 32u; ++L)
        {
            const float m = ((mask >> L) & 1u) != 0u ? sep[L] : CTE_MAX_F32;
            maxSeparation = min(maxSeparation, m);
        }

        nbOutputContacts = countbits(mask);

        // reserve the output range (atomicAdd on the shared temp-contact index);
        // modelled per test by its own counter slot preset to the base
        uint startIndex;
        InterlockedAdd(gCounter[t], nbOutputContacts, startIndex);
        index = startIndex;

        if ((startIndex + nbOutputContacts) < tempContactSize)
        {
            for (uint L = 0u; L < 32u; ++L)
                if (((mask >> L) & 1u) != 0u)
                    gTemp[startIndex + warpScanExclusive(mask, L)] = float4(pa[L], sep[L]);
        }
        else
            nbOutputContacts = 0u;
    }

    gOut[t] = uint4(nbOutputContacts, index, asuint(maxSeparation), 0u);
}
