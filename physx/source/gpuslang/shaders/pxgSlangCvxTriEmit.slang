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

// pxgSlangCvxTriEmit.slang
//
// The TAIL of `convexTriangleContactGen` (convexTriangle.cuh:1769-1881): what
// happens to the clipper's contacts once they exist.  It reduces them to at most
// CVX_TRI_MAX_CONTACTS, reserves a slice of the temp contact stream, scatters
// them into it, and writes the FIVE per-pair records the rest of the chain reads.
//
// WHY THIS IS ONE UNIT AND NOT TWO.  It looks like two stages — reduce-and-emit
// (:1769-1821) then write-the-outputs (:1826-1881) — and there are two Phase-1
// scalar shaders shaped that way.  That is the wrong seam.  The output half reads
// the RAW clipper staging slot 0 (:1857), recovers the triangle's other two
// vertices by unconditional shuffle (:1841-1842), and runs for EVERY pair
// INCLUDING SAT failures, while the emit half runs only inside `if (isSATPassed)`
// and `if (numContactsGenerated)`.  Splitting them means inventing an interface
// for five values and writing two fixtures for one thing.
//
// ONE WARP PER GROUP, `[numthreads(32,1,1)]`.  The uniformity argument is
// specific: `isSATPassed`, `numContactsGenerated` and `nbOutputContacts` are all
// warp-uniform, so every branch that contains a barrier is uniform — but NOTHING
// in the CUDA enforces that, so it is a precondition of this port, not a property
// of it.  With two warps per group one could leave while the other runs into a
// barrier, and llvmpipe HANGS rather than erroring.
//
// FIVE THINGS THE TRANSCRIPTION KEEPS THAT LOOK LIKE MISTAKES:
//
//   1. `.index` IS STORED BEFORE THE OVERFLOW TEST (:1803 vs :1808), and the
//      atomic reservation is never rolled back.  On overflow the index stays out
//      of range and only the zeroed `nbOutputContacts` (:1820) stops the
//      post-process from dereferencing it.  Reordering the store after the check
//      is NOT equivalent — the downstream filter is on the count, not the depth.
//   2. `delayContacts &&= numContactsGenerated` (:1823) ANDs with the RAW clipper
//      count, NOT with `nbOutputContacts` — which the overflow guard may have
//      just forced to zero.  Under temp-buffer pressure a port that used
//      `nbOutputContacts` would append FEWER pairs to the second-pass list than
//      the original, and the neighbours it dropped would then be culled.
//   3. THE SECOND-PASS CLASSIFIER READS RAW STAGING SLOT 0 (:1857), which is the
//      clipper's FIRST contact — and the reduction above may well have discarded
//      it.  Classifying the first EMITTED contact instead changes which
//      neighbouring triangle the post-process checks.
//   4. `mask | triangleIdx` at :1870 is a bitwise OR while `delayContactMask +
//      (nbOutputContacts << 27) + remapCpuTriangleIdx` at :1880 is an ADDITION.
//      With a triangle index at or above 2^27 the addition carries into the count
//      field.  There is no assert.  The mixed spelling is load-bearing.
//   5. `getTriangleActiveMask` RETURNS 0 for a contact strictly inside the
//      triangle (:1596), and the post-process turns that into a silent kill.  The
//      obviously-safe default of "needs processing" would KEEP contacts PhysX
//      drops.  The zero is passed through.
//
// ONE DELIBERATE DEVIATION, and it is a narrowing not a change.  The CUDA carries
// the triangle's vertices in `verts1`, written only on lanes < 3 and only inside
// `if (isSATPassed)` (:1713-1714), then recovers two of them with UNCONDITIONAL
// shuffles at :1841-1842 — so a SAT-failed pair shuffles a register that was
// never written.  It is harmless there only because `delayContacts` is false and
// nothing reads the result.  Here the three vertices are warp-uniform parameters:
// same values wherever the CUDA's are defined, defined everywhere the CUDA's are
// not.  The alternative — staging them under the SAT branch — would put a barrier
// in non-uniform control flow.

import pxgSlangContactReduce;

static const uint CTE_WARP = 32u;
static const float CTE_MAX_F32 = 3.4028234663852886e+38f;   // PX_MAX_F32 / FLT_MAX

public static const uint CTE_MAX_CONTACTS = 5u;             // CVX_TRI_MAX_CONTACTS
public static const uint CTE_DEFERRED_MASK = 0x80000000u;   // ConvexTriNormalAndIndex
public static const uint CTE_NB_CONTACTS_SHIFT = 27u;

// ConvexTriIntermediateData::VoronoiState (convexNpCommon.h:121-130)
public static const uint CTE_V0 = 1u << 29;
public static const uint CTE_V1 = 2u << 29;
public static const uint CTE_V2 = 3u << 29;
public static const uint CTE_E01 = 4u << 29;
public static const uint CTE_E02 = 5u << 29;
public static const uint CTE_E12 = 6u << 29;

// Self-contained warp primitives, prefixed `cte`/`gsCte`, for the reason
// pxgSlangContactReduce is: this module is imported alongside modules that carry
// their own, and an `import` is a module boundary.
groupshared uint gsCteStage[CTE_WARP];

uint cteExchange(uint lane, uint value, uint src)
{
    gsCteStage[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gsCteStage[src & (CTE_WARP - 1u)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float cteShflF(uint lane, float value, uint src)
{
    return asfloat(cteExchange(lane, asuint(value), src));
}

uint cteShflU(uint lane, uint value, uint src)
{
    return cteExchange(lane, value, src);
}

float cteShflXorF(uint lane, float value, uint xorMask)
{
    return asfloat(cteExchange(lane, asuint(value), lane ^ xorMask));
}

// warpReduction<MinOpFloat> — the XOR butterfly, result broadcast to every lane.
float cteReduceMinF(uint lane, float v)
{
    v = min(v, cteShflXorF(lane, v, 16u));
    v = min(v, cteShflXorF(lane, v, 8u));
    v = min(v, cteShflXorF(lane, v, 4u));
    v = min(v, cteShflXorF(lane, v, 2u));
    v = min(v, cteShflXorF(lane, v, 1u));
    return v;
}

// reduction.cuh:592 — pure popcount, no cross-lane traffic.
uint cteWarpScanExclusive(uint mask, uint lane)
{
    return countbits(mask & ((1u << lane) - 1u));
}

// --------------------------------------------------------------------------
// barycentricCoordinates (convexTriangle.cuh:1538) and getTriangleActiveMask
// (:1562).  Scalar, run on lane 0 only.
// --------------------------------------------------------------------------
void cteBarycentric(float3 p, float3 a, float3 b, float3 c, out float v, out float w)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 n = cross(ab, ac);

    const float3 ap = a - p;
    const float3 bp = b - p;
    const float3 cp = c - p;

    const float va = dot(n, cross(bp, cp));
    const float vb = dot(n, cross(cp, ap));
    const float vc = dot(n, cross(ap, bp));

    const float totalArea = va + vb + vc;
    const float denom = totalArea == 0.0f ? 0.0f : 1.0f / totalArea;
    v = vb * denom;
    w = vc * denom;
}

// THE ZERO RETURN IS A REAL ANSWER, not a failure: a contact strictly inside the
// triangle belongs to no vertex and no edge region.
public uint cteTriangleActiveMask(float v, float w)
{
    const float upperBound = 0.97f;
    const float u = 1.0f - v - w;

    if (v > upperBound) return CTE_V1;
    if (w > upperBound) return CTE_V2;
    if (u > upperBound) return CTE_V0;
    if ((v + u) > upperBound) return CTE_E01;
    if ((u + w) > upperBound) return CTE_E02;
    if ((w + v) > upperBound) return CTE_E12;
    return 0u;
}

// --------------------------------------------------------------------------
// The tail.
//
// `staged` is this lane's slot of the clipper's contact staging, read
// UNCONDITIONALLY by the caller — every lane index is a valid slot, so the read
// is in bounds even when the lane holds no contact; the guard below is what makes
// the value irrelevant.  `rawContact0` is staging slot 0, which the second-pass
// classifier needs and the reduction may have discarded.
//
// Feeding zeros for the lanes past `numContactsGenerated` is output-neutral:
// contactReduce neutralises out-of-set lanes before its butterflies and ANDs the
// set mask into every ballot, so their values cannot win an anchor or set a cull
// bit that survives.  The CUDA leaves those registers uninitialised instead.
// --------------------------------------------------------------------------
public void cteEmitTail(
    uint lane,
    bool isSATPassed,
    bool delayContactsIn,
    uint numContactsGenerated,
    float4 staged,
    float3 rawContact0,
    float3 minNormal,
    float3 triVert0, float3 triVert1, float3 triVert2,
    uint convexTriPairOffset, uint convexTriPairOffsetPadded,
    uint remapCpuTriangleIdx, uint triangleIdx, uint globalWarpIndex,
    uint64_t cvxTriNI, uint64_t cvxTriContacts, uint64_t cvxTriMaxDepth,
    uint64_t cvxTriInterm, uint64_t orderedCvxTriInterm,
    uint64_t convTriSecondPassPairs, uint64_t nbSecondPassPairs,
    uint64_t tempConvexTriContacts, uint tempContactSize, uint64_t pTempContactIndex)
{
    float maxSeparation = CTE_MAX_F32;
    uint nbOutputContacts = 0u;
    bool delayContacts = delayContactsIn;

    // Both conditions are warp-uniform, so every barrier below them is reached by
    // the whole warp or by none of it.
    if (isSATPassed && numContactsGenerated != 0u)
    {
        const bool live = lane < numContactsGenerated;
        const float3 pa = live ? staged.xyz : float3(0.0f, 0.0f, 0.0f);
        const float sep = live ? staged.w : 0.0f;

        // THE SHIFT.  `numContactsGenerated` can be exactly 32 — the clipper's
        // write cap is `index < NUM_TMP_CONTACTS_PER_PAIR` and its count is a
        // popcount of a ballot, so [0, 32] inclusive.  PTX `shl.b32` masks the
        // shift amount to five bits, so `1 << 32` is 1 and the CUDA lands on
        // 0xFFFFFFFF, which is the value it wants.  SPIR-V leaves a shift at or
        // past the operand width UNDEFINED.  Unguarded, a full 32-contact face
        // silently collapses to one contact.
        uint mask = (numContactsGenerated >= CTE_WARP)
                        ? 0xffffffffu
                        : ((1u << numContactsGenerated) - 1u);

        if (numContactsGenerated > CTE_MAX_CONTACTS)
            mask = crContactReduce5(lane, pa, sep, minNormal, mask);

        maxSeparation = ((mask >> lane) & 1u) != 0u ? sep : CTE_MAX_F32;
        maxSeparation = cteReduceMinF(lane, maxSeparation);

        nbOutputContacts = countbits(mask);

        // ZERO-INITIALISED, unlike the CUDA's `PxU32 startIndex;` at :1798, which
        // is assigned on lane 0 only and then read by all 32 through a shuffle.
        // The emulated shuffle is a groupshared round trip whose barriers must sit
        // OUTSIDE the lane-0 branch, exactly as the CUDA's does.
        uint startIndex = 0u;
        if (lane == 0u)
        {
            InterlockedAdd(Ptr<uint>(pTempContactIndex)[0], nbOutputContacts, startIndex);
            // Stored BEFORE the overflow test, and never rolled back.  See the
            // header: this is deliberate.
            Ptr<uint>(cvxTriContacts)[convexTriPairOffset] = startIndex;
        }
        startIndex = cteShflU(lane, startIndex, 0u);

        // STRICTLY `<`.  At startIndex + nbOutputContacts == tempContactSize the
        // slice would still fit, and the CUDA drops it anyway.
        if ((startIndex + nbOutputContacts) < tempContactSize)
        {
            if (((mask >> lane) & 1u) != 0u)
            {
                const uint index = cteWarpScanExclusive(mask, lane);
                Ptr<float4>(tempConvexTriContacts)[startIndex + index] = float4(pa, sep);
            }
        }
        else
        {
            nbOutputContacts = 0u;
        }
    }

    // ANDed with the RAW clipper count, not with nbOutputContacts.
    delayContacts = isSATPassed && delayContacts && (numContactsGenerated != 0u);

    if (lane == 0u)
    {
        Ptr<float>(cvxTriMaxDepth)[convexTriPairOffset] = maxSeparation;

        if (delayContacts)
        {
            uint secondPassIndex;
            InterlockedAdd(Ptr<uint>(nbSecondPassPairs)[0], 1u, secondPassIndex);

            // The RAW first clipper contact, which the reduction may have thrown
            // away.  Classified against the triangle it was clipped to.
            float v, w;
            cteBarycentric(rawContact0, triVert0, triVert1, triVert2, v, w);
            const uint activeMask = cteTriangleActiveMask(v, w);

            Ptr<uint>(convTriSecondPassPairs)[secondPassIndex] = globalWarpIndex | activeMask;
        }

        Ptr<uint>(cvxTriInterm)[convexTriPairOffset] = triangleIdx;

        // OR here, ADD below.  Not a style choice — see the header.
        const uint faceFlag = (nbOutputContacts != 0u && !delayContacts) ? (1u << 31) : 0u;
        Ptr<uint>(orderedCvxTriInterm)[convexTriPairOffsetPadded] = faceFlag | triangleIdx;
    }

    // ConvexTriNormalAndIndex_WriteWarp (dataReadWriteHelper.cuh:370): the 16-byte
    // record is written as four uints by lanes 0..3, one each.
    const uint delayContactMask = delayContacts ? CTE_DEFERRED_MASK : 0u;
    const uint index = delayContactMask + (nbOutputContacts << CTE_NB_CONTACTS_SHIFT)
                       + remapCpuTriangleIdx;

    if (lane < 4u)
    {
        uint word;
        if (lane == 0u) word = asuint(minNormal.x);
        else if (lane == 1u) word = asuint(minNormal.y);
        else if (lane == 2u) word = asuint(minNormal.z);
        else word = index;

        Ptr<uint>(cvxTriNI)[4u * convexTriPairOffset + lane] = word;
    }
}
