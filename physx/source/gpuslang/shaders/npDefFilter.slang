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

// npDefFilter.slang
//
// Slang port of the deformable element-filter lookups from
// gpunarrowphase/src/CUDA/deformableElementFilter.cuh — the sorted-array binary
// searches over the three filter-pair types (PxParticleRigidFilterPair with its
// operator== / operator< navigation; PxgRigidFilterPair and
// PxgNonRigidFilterPair with their 3-key compare() and the ignoreIndex2
// early-accept) and findRange (the runsum upper-bound search returning r-1,
// which deliberately wraps to 0xffffffff when the value precedes the first
// entry) — plus the four EdgeEncoding authorship predicates from
// deformableCollision.cuh (isType0/1EdgeActive, isType0/1VertexActive).
//
// The CUDA PxU64 keys are represented as uint2 (lo = .x, hi = .y); the
// lexicographic (hi, lo) comparison is bit-identical to the 64-bit unsigned
// comparison.  All outputs are pure integers, so verification is bit-exact.

struct DfParams { uint numTests; uint numPairs; uint numValues; uint pad0; }

struct PairPR { uint2 id0; uint2 id1; }                                     // PxParticleRigidFilterPair
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0, pad1, pad2; } // PxgRigidFilterPair
struct PairNR { uint2 index0; uint index1; uint index2; }                   // PxgNonRigidFilterPair
struct DfQuery { uint2 q0; uint2 q1; uint idx2; uint mode; uint value; uint auth; }

[[vk::push_constant]]
ConstantBuffer<DfParams> gParams;

RWStructuredBuffer<PairPR> gPairsPR;  // binding 0 (sorted)
RWStructuredBuffer<PairR> gPairsR;    // binding 1 (sorted)
RWStructuredBuffer<PairNR> gPairsNR;  // binding 2 (sorted)
RWStructuredBuffer<uint> gValues;     // binding 3 (sorted ascending)
RWStructuredBuffer<DfQuery> gQuery;   // binding 4
RWStructuredBuffer<uint4> gOut;       // binding 5

// 64-bit unsigned comparison on (lo = .x, hi = .y) pairs.
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// find(PxParticleRigidFilterPair*): binary search navigated by operator< /
// operator==  ( < is (mID0,mID1) lexicographic ).
bool findParticleRigid(uint nbPairs, uint2 id0, uint2 id1)
{
    uint l = 0, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const PairPR p = gPairsPR[mid];
        if (u64Eq(id0, p.id0) && u64Eq(id1, p.id1))
            return true;
        else if (u64Lt(id0, p.id0) || (u64Eq(id0, p.id0) && u64Lt(id1, p.id1)))
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

// PxgRigidFilterPair::compare — 3-key lexicographic (index0 u64, index1 u64, index2 u32).
int compareR(uint2 a0, uint2 a1, uint a2, PairR p)
{
    if (u64Lt(a0, p.index0)) return -1;
    if (u64Lt(p.index0, a0)) return 1;
    if (u64Lt(a1, p.index1)) return -1;
    if (u64Lt(p.index1, a1)) return 1;
    if (a2 < p.index2) return -1;
    if (a2 > p.index2) return 1;
    return 0;
}

// find(PxgRigidFilterPair*, rigidId, compressedParticleId): query index2 = 0,
// ignoreIndex2 = true (accept any index2 once index0/index1 match).
bool findRigid(uint nbPairs, uint2 q0, uint2 q1)
{
    uint l = 0, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const PairR p = gPairsR[mid];
        const int cmp = compareR(q0, q1, 0u, p);
        if (cmp == 0 || (u64Eq(q0, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

// PxgNonRigidFilterPair::compare — 3-key lexicographic (index0 u64, index1 u32, index2 u32).
int compareNR(uint2 a0, uint a1, uint a2, PairNR p)
{
    if (u64Lt(a0, p.index0)) return -1;
    if (u64Lt(p.index0, a0)) return 1;
    if (a1 < p.index1) return -1;
    if (a1 > p.index1) return 1;
    if (a2 < p.index2) return -1;
    if (a2 > p.index2) return 1;
    return 0;
}

// find(PxgNonRigidFilterPair*): the 2-arg wrapper passes index2 = 0 with
// ignoreIndex2 = true; the 3-arg wrapper passes the user-buffer id with
// ignoreIndex2 = false.
bool findNonRigid(uint nbPairs, uint2 q0, uint q1, uint q2, bool ignoreIndex2)
{
    uint l = 0, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const PairNR p = gPairsNR[mid];
        const int cmp = compareNR(q0, q1, q2, p);
        if (cmp == 0 || (ignoreIndex2 && u64Eq(q0, p.index0) && q1 == p.index1))
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

// findRange: upper-bound search returning r-1 — the runsum bucket of `value`.
// When value < values[0], r ends at 0 and the return wraps to 0xffffffff,
// exactly as the CUDA's unsigned arithmetic does.
uint findRange(uint value, uint length)
{
    if (length == 0)
        return 0;
    uint l = 0, r = length;
    while (l < r)
    {
        const uint m = (l + r) / 2;
        if (gValues[m] > value)
            r = m;
        else
            l = m + 1;
    }
    return r - 1;
}

// EdgeEncoding bit positions (PxgFEMCloth.h).
static const uint TYPE0_EDGE_BASE_POS = 0;
static const uint TYPE0_VERTEX0_ACTIVE_POS = 11;
static const uint TYPE1_EDGE_BASE_POS = 16;
static const uint TYPE1_VERTEX0_ACTIVE_POS = 27;

bool isType0EdgeActive(uint edgeAuthorship, uint localEdgeIndex) { return (edgeAuthorship & (1u << (TYPE0_EDGE_BASE_POS + localEdgeIndex))) != 0; }
bool isType1EdgeActive(uint edgeAuthorship, uint localEdgeIndex) { return (edgeAuthorship & (1u << (TYPE1_EDGE_BASE_POS + localEdgeIndex))) != 0; }
bool isType0VertexActive(uint edgeAuthorship, uint localVertexIndex) { return (edgeAuthorship & (1u << (TYPE0_VERTEX0_ACTIVE_POS + localVertexIndex))) != 0; }
bool isType1VertexActive(uint edgeAuthorship, uint localVertexIndex) { return (edgeAuthorship & (1u << (TYPE1_VERTEX0_ACTIVE_POS + localVertexIndex))) != 0; }

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctFindParticleRigid(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const DfQuery q = gQuery[i];
    gOut[i] = uint4(findParticleRigid(gParams.numPairs, q.q0, q.q1) ? 1u : 0u, 0, 0, 0);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctFindRigid(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const DfQuery q = gQuery[i];
    gOut[i] = uint4(findRigid(gParams.numPairs, q.q0, q.q1) ? 1u : 0u, 0, 0, 0);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctFindNonRigid(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const DfQuery q = gQuery[i];
    // mode 0: the 2-arg wrapper (index2 = 0, ignoreIndex2 = true);
    // mode 1: the 3-arg wrapper (index2 = q.idx2, ignoreIndex2 = false).
    const bool ignoreIndex2 = (q.mode == 0);
    const uint idx2 = ignoreIndex2 ? 0u : q.idx2;
    gOut[i] = uint4(findNonRigid(gParams.numPairs, q.q0, q.q1.x, idx2, ignoreIndex2) ? 1u : 0u, 0, 0, 0);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctFindRange(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const DfQuery q = gQuery[i];
    gOut[i] = uint4(findRange(q.value, gParams.numValues), 0, 0, 0);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctEdgeAuth(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const DfQuery q = gQuery[i];
    const uint li = q.value; // local edge/vertex index 0..2
    uint bits = 0;
    if (isType0EdgeActive(q.auth, li)) bits |= 1u;
    if (isType1EdgeActive(q.auth, li)) bits |= 2u;
    if (isType0VertexActive(q.auth, li)) bits |= 4u;
    if (isType1VertexActive(q.auth, li)) bits |= 8u;
    gOut[i] = uint4(bits, 0, 0, 0);
}
