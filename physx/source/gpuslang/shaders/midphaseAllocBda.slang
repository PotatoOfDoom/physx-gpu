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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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

// midphaseAllocBda.slang
//
// A dispatch wrapper around pxgSlangMidphaseAlloc, so the layout arithmetic can
// be certified BEFORE convexTrimeshNarrowphase (its first consumer) exists.
// Not an engine kernel and deliberately absent from the manifest — the same
// scaffolding pattern pcmManifoldBda used for the PCM gate in P8.23.
//
// One thread per case: each writes the seven addresses it computed as OFFSETS
// from the stack base, so the fixture compares plain integers rather than
// device addresses that shift between runs.

import pxgSlangMidphaseAlloc;

struct AllocParams
{
    uint64_t cases;   // @0   uint2[] — (numPairsUnpadded, numPaddedPairs) per case
    uint64_t out;     // @8   uint64_t[] — 7 offsets per case
    uint     count;   // @16
    uint     _pad20;  // @20 -> 24-byte blob
}

[[vk::push_constant]]
ConstantBuffer<AllocParams> gAp;

[shader("compute")]
[numthreads(64, 1, 1)]
void midphaseAllocLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gAp.count) return;

    const uint2 c = Ptr<uint2>(gAp.cases)[i];
    // A non-zero, non-16-aligned base on purpose: the arithmetic must be a pure
    // byte offset from wherever the stack happens to start, and a base of 0 would
    // hide a mistake that multiplied the base instead of adding to it.
    const uint64_t base = 0x1000UL + 8UL;
    const MidphaseAlloc a = maCompute(base, c.x, c.y);

    Ptr<uint64_t> o = Ptr<uint64_t>(gAp.out);
    o[i * 7u + 0u] = a.pairs      - base;
    o[i * 7u + 1u] = a.ni         - base;
    o[i * 7u + 2u] = a.contacts   - base;
    o[i * 7u + 3u] = a.ordered    - base;
    o[i * 7u + 4u] = a.maxDepth   - base;
    o[i * 7u + 5u] = a.interm     - base;
    o[i * 7u + 6u] = a.secondPass - base;
}
