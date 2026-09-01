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
// npDefFilterBda.slang
//
// P9.39b -- the BDA fixture shader for pxgSlangDefFilter's ENGINE-PATH searches.
//
// npDefFilter.slang next door is the Phase-1 twin: it re-implements the same lookups
// over descriptor-set bindings and proves them against a CPU transcription.  It cannot
// test the MODULE, because the module takes device addresses and a StructuredBuffer
// binding is not one -- which is exactly the limitation pxgSlangDefFilter's header
// records.  So the engine path gets its own rig here, and the two now cover the two
// halves between them.
//
// What this exercises that the Phase-1 rig cannot:
//
//  B1. dfFindRange's UNDERFLOW.  `return r - 1` wraps to 0xffffffff when the value
//      precedes the first entry.  The engine relies on its callers never asking; the
//      port transcribes it rather than "fixing" it, so the fixture asks and pins the
//      wrapped answer.
//
//  B2. ignoreIndex2 = FALSE.  Every caller in the corpus before P9.39b passed true,
//      so the disjunct was dead everywhere and a port that hardcoded it passed.  The
//      softbody-vs-particle contact gen passes FALSE with a real index2, and these
//      queries separate the two: the same (index0, index1) with a DIFFERENT index2
//      must be found under true and missed under false.
//
//  B3. THE PARTICLE CHAIN, and that it is not the rigid one.  dfFindParticleNonRigid
//      looks up which buffer owns the particle, takes that buffer's mUniqueId as the
//      third key, and searches the SHARED array with the GLOBAL compressed id.  The
//      rigid chain (psContactGenBda) subtracts the runsum and searches the buffer's
//      OWN array with a LOCAL id.  A rig that only ever used one buffer could not
//      tell them apart, so this one uses three with distinct unique ids.
// ===========================================================================
import pxgSlangDefFilter;
import pxgSlangParticleDesc;

struct DfbParams
{
    uint numTests;          // @0
    uint numPairs;          // @4
    uint64_t queries;       // @8
    uint64_t results;       // @16
    uint64_t pairsNR;       // @24
    uint64_t particleSystem;// @32
    uint64_t values;        // @40
    uint numValues;         // @48
    uint pad52;             // @52 -> 56
}

[[vk::push_constant]]
ConstantBuffer<DfbParams> gDfb;

// mode 0: dfFindRange     1: dfFindNonRigidPair3   2: dfFindNonRigidPair
// mode 3: dfFindParticleNonRigid
struct DfbQuery
{
    uint mode;
    uint id0, id1, id2;
    uint ignoreIndex2;
    uint value;
    uint pad0, pad1;        // -> 32 B
}

[shader("compute")]
[numthreads(64, 1, 1)]
void dfbRun(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gDfb.numTests)
        return;

    const DfbQuery q = Ptr<DfbQuery>(gDfb.queries)[i];
    uint r = 0xDEADBEEFu;

    if (q.mode == 0u)
        r = dfFindRange(gDfb.values, gDfb.numValues, q.value);
    else if (q.mode == 1u)
        r = dfFindNonRigidPair3(gDfb.pairsNR, gDfb.numPairs, q.id0, q.id1, q.id2,
                                q.ignoreIndex2 != 0u) ? 1u : 0u;
    else if (q.mode == 2u)
        r = dfFindNonRigidPair(gDfb.pairsNR, gDfb.numPairs, q.id0, q.id1) ? 1u : 0u;
    else if (q.mode == 3u)
        r = dfFindParticleNonRigid(gDfb.particleSystem, gDfb.pairsNR, gDfb.numPairs,
                                   q.id0, q.id1) ? 1u : 0u;

    Ptr<uint>(gDfb.results)[i] = r;
}
