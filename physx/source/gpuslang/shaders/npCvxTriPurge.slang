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

// npCvxTriPurge.slang
//
// Slang port of purgeEmptyTriangles (convexMeshCorrelate.cu :58-87) — tranche
// 114.  A warp-SIMD IN-PLACE stream compaction that keeps only the triangles
// that actually generated contacts, run once per convex-mesh pair before the
// per-triangle contacts are sorted into patches.  Walking the `count` triangles
// in blocks of 32:
//
//   d          = tI < nbToLoad ? depthBuffer[loaded+tI] : FLT_MAX;
//   nbContacts = nbToLoad ? getNbContacts(nicBuffer[loaded+tI].index) : 0;
//   valid      = d != FLT_MAX && nbContacts > 0;
//   dest       = __popc(ballot(valid) & ((1<<tI)-1)) + nbTriangles;   // exclusive scan + base
//   if (valid) { depthBuffer[dest] = d;
//                tempCounts[dest]  = ((loaded+tI) << 4) | nbContacts;   // orig pos | nbContacts(<=15)
//                nicBuffer[dest]   = nicBuffer[loaded+tI]; }
//   nbTriangles += __popc(ballot(valid));
//
// The compaction is in-place (dest <= loaded+tI): each block writes into the
// already-consumed prefix, and because writes for block k land in
// [nbTriangles, nbTriangles+popc) with nbTriangles <= loaded, they never touch
// the NEXT block's read window [loaded+32, ..).  Serialised to one thread per
// warp: every lane's reads for a block are gathered into per-lane temporaries
// FIRST (matching the __ballot_sync barrier between the loads and the stores),
// then the compacting stores are applied — so the in-place move is bit-exact.
//
// Faithful quirk: line 71 reads nicBuffer[loaded+tI].index for ALL 32 lanes
// whenever nbToLoad > 0 (i.e. every loop iteration), including lanes tI >=
// nbToLoad whose slot is past the valid tail; the computed nbContacts is unused
// for those lanes because d == FLT_MAX forces valid = false.  Reproduced
// verbatim (the read stays in the padded buffer).
//
// Determinism: the only float touched is the `d != FLT_MAX` sentinel test
// (exact), everything else is integer; the compacted order is lane order, so the
// output arrays and the returned count are compared position-for-position.

static const uint DeferredContactMask = 0x80000000u;
static const uint NbContactsShift = 27u;
static const float PURGE_FLT_MAX = 3.402823466e38f;

uint getNbContacts(uint index) { return (index & ~DeferredContactMask) >> NbContactsShift; }

// ConvexTriNormalAndIndex { PxVec3 normal; int index } packed as a float4:
// xyz = normal, w = the index bit-pattern (read via asuint).
struct PgParams
{
    uint numTests;
    uint stride; // per-test block length (in elements) of each array
    uint pad0, pad1;
}

[[vk::push_constant]]
ConstantBuffer<PgParams> gParams;

RWStructuredBuffer<float> gDepth;   // binding 0: depthBuffer (in + out, stride/test)
RWStructuredBuffer<float4> gNic;    // binding 1: nicBuffer xyz=normal w=asfloat(index) (in + out)
RWStructuredBuffer<uint> gTemp;     // binding 2: tempCounts (out, stride/test)
RWStructuredBuffer<uint> gCount;    // binding 3: x = count (in), overwritten with nbTriangles (out)

uint popc(uint x) { return countbits(x); }

[shader("compute")]
[numthreads(64, 1, 1)]
void ctCvxTriPurge(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const uint base = t * gParams.stride;
    const int count = int(gCount[t]);

    uint nbTriangles = 0u;

    for (int loaded = 0; loaded < count; loaded += 32)
    {
        const int nbToLoad = min(count - loaded, 32);

        // ---- gather every lane's reads for this block FIRST (the __ballot_sync
        //      barrier separates the loads from the compacting stores) ----
        float d[32];
        uint nbContacts[32];
        float4 nic[32];
        uint validBits = 0u;
        for (uint tI = 0u; tI < 32u; ++tI)
        {
            d[tI] = (int(tI) < nbToLoad) ? gDepth[base + uint(loaded) + tI] : PURGE_FLT_MAX;
            nic[tI] = gNic[base + uint(loaded) + tI];
            nbContacts[tI] = (nbToLoad != 0) ? getNbContacts(asuint(nic[tI].w)) : 0u;
            const bool valid = (d[tI] != PURGE_FLT_MAX) && (nbContacts[tI] > 0u);
            if (valid) validBits |= (1u << tI);
        }

        // ---- compacting stores ----
        for (uint tI = 0u; tI < 32u; ++tI)
        {
            if ((validBits & (1u << tI)) == 0u) continue;
            const uint dest = popc(validBits & ((1u << tI) - 1u)) + nbTriangles;
            gDepth[base + dest] = d[tI];
            gTemp[base + dest] = ((uint(loaded) + tI) << 4) | nbContacts[tI];
            gNic[base + dest] = nic[tI];
        }

        nbTriangles += popc(validBits);
    }

    gCount[t] = nbTriangles;
}
