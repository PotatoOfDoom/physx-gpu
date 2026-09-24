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

// npTriTriAccept.slang -- P9.83b-5.
//
//   Fixture entry for the DECISION logic of the two patch functions:
//   tryToAssignContactToExistingPatch's acceptance tests (:779-783),
//   createNewPatches' acceptance tests (:921-925), and the per-warp seed pick
//   (:879-885).
//
// IT IS ITS OWN FILE, AND THAT IS NOT TIDINESS.  The first attempt put this entry
// beside ttrReduce in npTriTriReduce.slang and it SEGFAULTED the whole suite --
// including the ttrReduce case that had been passing.  Slang assigns
// RWStructuredBuffer bindings in declaration order across the FILE, not per entry
// point, so the second entry's buffers landed at bindings 2, 3, 4 while the test
// harness bound them at 0, 1, 2.  Nothing in the shader or the C++ says so; the
// only symptom is a crash in an unrelated test.
//
// The property under test is that the two acceptance forms DIFFER.  They look like
// the same three lines -- a dot product against two epsilons -- and they are not:
// one is ungated and one is gated on candidateContact, and only one carries the
// `|| myNormal` disjunct.  A fixture that ran one of them would pass against a
// port that used it at both sites.
//
// STAMPED NoContraction since P9.83b-8 T1.  Every engine build of the modules this
// drives is stamped -- the mesh-mesh kernel will be, and the serial reduce's
// other engine callers (triPlaneNpBda, the cloth kernels) already are -- and the
// mirror is built with -ffp-contract=off.  The note this replaced said the
// opposite and had gone stale; tests/CMakeLists.txt has the list.

import pxgSlangTriTriPatch;

struct TtaParams
{
    uint numCases;
    uint pad0, pad1, pad2;
}

struct TtaCase
{
    float4 cn[32];        // xyz = contact normal, w unused
    float4 sep8[8];       // 32 separations, four per float4
    float4 patchNormal;   // xyz = patch normal, w unused
    uint4 misc;           // x = candidateMask, y = seedLane, z..w unused
}

[[vk::push_constant]] ConstantBuffer<TtaParams> gPA;

RWStructuredBuffer<TtaCase> gInA;    // binding 0
RWStructuredBuffer<uint4> gOutA;     // binding 1: per case
                                     //   x = acceptMask (existing form)
                                     //   y = accept2Mask (existing form)
                                     //   z = acceptMask (new form)
                                     //   w = accept2Mask (new form)
RWStructuredBuffer<uint4> gOutSeed;  // binding 2: x = seed lane, y = asuint(minSep)

[shader("compute")]
[numthreads(64, 1, 1)]
void ttrAccept(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gPA.numCases) return;

    float3 cn[32];
    float sep[32];
    for (uint j = 0; j < 32u; ++j)
    {
        cn[j] = gInA[t].cn[j].xyz;
        const float4 sv = gInA[t].sep8[j >> 2];
        const uint k = j & 3u;
        sep[j] = (k == 0u) ? sv.x : (k == 1u) ? sv.y : (k == 2u) ? sv.z : sv.w;
    }

    const float3 pn = gInA[t].patchNormal.xyz;
    const uint candidateMask = gInA[t].misc.x;
    const uint seedLane = gInA[t].misc.y;

    uint a1, a2, b1, b2;
    ttpAcceptExisting(cn, pn, a1, a2);
    ttpAcceptNew(cn, candidateMask, pn, seedLane, b1, b2);
    gOutA[t] = uint4(a1, a2, b1, b2);

    float minSep;
    const uint seed = ttpWarpSeed(sep, candidateMask, minSep);
    gOutSeed[t] = uint4(seed, asuint(minSep), 0u, 0u);
}

// ---------------------------------------------------------------------------
// The reservation entry.  Shares this file's bindings deliberately: the earlier
// segfault came from adding an entry with its OWN buffers to a file that already
// had some, and the fix is either a new file or REUSING the existing bindings.
// This entry reuses gInA/gOutA/gOutSeed by reinterpreting them, so the binding
// layout is unchanged and the harness binds the same three slots.
// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void ttrReserve(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gPA.numCases) return;

    // The case is packed into the existing TtaCase: misc carries the counts, and
    // the first two float4 of cn[] carry the six 32-bit inputs as raw bits.
    const uint oldAllFlags   = asuint(gInA[t].cn[0].x);
    const uint numPatches    = asuint(gInA[t].cn[0].y);
    const uint totalContacts = asuint(gInA[t].cn[0].z);
    const uint patchAdd      = asuint(gInA[t].cn[0].w);
    const uint contactAdd    = asuint(gInA[t].cn[1].x);
    const uint forceAdd      = asuint(gInA[t].cn[1].y);
    const uint patchLimit    = asuint(gInA[t].cn[1].z);
    const uint contactLimit  = asuint(gInA[t].cn[1].w);
    const uint forceLimit    = asuint(gInA[t].cn[2].x);

    const TtpReservation r = ttpReserveBlock(oldAllFlags, numPatches, totalContacts,
                                             patchAdd, contactAdd, forceAdd,
                                             patchLimit, contactLimit, forceLimit);

    gOutA[t] = uint4(r.patchByteOffset, r.contactByteOffset, r.forceByteOffset, r.allFlagsWord);
    gOutSeed[t] = uint4(r.numPatchesOut | (r.numContactsOut << 16u),
                        r.touchChange | (r.patchChange << 1u)
                            | (r.overflow ? 4u : 0u)
                            | (r.writerPatchesNull ? 8u : 0u)
                            | (r.writerContactsNull ? 16u : 0u),
                        r.overflowBits, r.statusFlags);
}

