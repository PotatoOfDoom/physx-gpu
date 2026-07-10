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

// npMidphaseAlloc.slang
//
// Slang port of the convex-mesh midphase collision-stack allocator from
// gpunarrowphase/src/CUDA/midphaseAllocate.cuh — calculateConvexMeshPair-
// MemRequirement, calculateAdditionalPadding, calculateMaxPairs (including
// the unsigned `stackSizeBytes - additionalPadding` wrap when the padding
// exceeds the stack) and midphaseAllocate itself, whose CUDA pointer bumps
// become byte offsets from the stack base: pairs (uint4), then
// ConvexTriNormalAndIndex, ConvexTriContacts, the padded/duplicated ordered
// triangle-index list, maxDepth (float), ConvexTriIntermediateData and the
// second-pass indices — with the pair count padded to a multiple of 4.
//
// Plus the bit-packing accessors from gpunarrowphase/include/convexNpCommon.h:
// ConvexTriNormalAndIndex::getTriangleIndex / getNbContacts /
// isDeferredContact (NbContactsShift = 27, DeferredContactMask = 0x80000000)
// and ConvexTriIntermediateData::isEdgeContact.  Everything is pure integer
// arithmetic, so verification is bit-exact.  (The CUDA bumps 64-bit pointers;
// these offsets are 32-bit — identical for any stack addressable by the
// PxU32 stackSizeBytes that governs it.)

struct MaParams { uint numTests; uint pad0, pad1, pad2; }

struct MaIn { uint numpairsAfterMPUnpadded; uint numPaddedPairsAfterMP; uint stackSizeBytes; uint numContactManagers; }
struct MaOut
{
    uint4 offsetsA; // x=pairs, y=cvxTriNI, z=cvxTriContacts, w=orderedCvxTriInterm
    uint4 offsetsB; // x=cvxTriMaxDepth, y=cvxTriInterm, z=cvxTriSecondPassed, w=memRequirementPerPair
    uint4 sizes;    // x=additionalPadding, y=maxPairs, z=numPairsAfterMP (padded), w=0
}

[[vk::push_constant]]
ConstantBuffer<MaParams> gParams;

RWStructuredBuffer<MaIn> gIn;    // binding 0
RWStructuredBuffer<MaOut> gOut;  // binding 1
RWStructuredBuffer<uint2> gAcc;  // binding 2 (x = NI index word, y = interm mask word)
RWStructuredBuffer<uint4> gAccOut; // binding 3

// sizeof() of the CUDA structs (convexNpCommon.h):
static const uint SIZEOF_UINT4 = 16;
static const uint SIZEOF_CONVEXTRI_NORMAL_AND_INDEX = 16; // PxVec3 + int
static const uint SIZEOF_CONVEXTRI_CONTACTS = 4;          // PxU32 index
static const uint SIZEOF_CONVEXTRI_INTERMEDIATE = 4;      // PxU32 gpuTriIndex
static const uint SIZEOF_REAL = 4;
static const uint SIZEOF_U32 = 4;

// ConvexTriNormalAndIndex packing
static const uint DEFERRED_CONTACT_MASK = 0x80000000;
static const uint NB_CONTACTS_SHIFT = 27;

uint ctniGetTriangleIndex(uint index) { return index & ((1u << NB_CONTACTS_SHIFT) - 1); }
uint ctniGetNbContacts(uint index) { return (index & (~DEFERRED_CONTACT_MASK)) >> NB_CONTACTS_SHIFT; }
bool ctniIsDeferredContact(uint index) { return (index & DEFERRED_CONTACT_MASK) != 0; }
bool ctidIsEdgeContact(uint mask) { return (mask & 0x80000000) != 0; }

uint calculateConvexMeshPairMemRequirement()
{
    return SIZEOF_UINT4                          // sPairsGPU
         + SIZEOF_CONVEXTRI_NORMAL_AND_INDEX     // normal and index
         + SIZEOF_CONVEXTRI_CONTACTS             // convexTriContacts
         + SIZEOF_REAL                           // maxdepth
         + SIZEOF_CONVEXTRI_INTERMEDIATE         // intermediate data
         + SIZEOF_U32                            // secondPass indices
         + SIZEOF_U32 * 2;                       // triangle indices + radix sort data
}

// worst-case padding (intentionally) to avoid stack overflows
uint calculateAdditionalPadding(uint numContactManagers)
{
    const uint afterPairPadding = 3 * (calculateConvexMeshPairMemRequirement() - 2 * SIZEOF_U32);
    const uint afterTriangleIndexPadding = numContactManagers * 6 * SIZEOF_U32;
    return afterPairPadding + afterTriangleIndexPadding;
}

uint calculateMaxPairs(uint stackSizeBytes, uint numContactManagers)
{
    const uint additionalPadding = calculateAdditionalPadding(numContactManagers);
    const uint memAvailableAfterPadding = stackSizeBytes - additionalPadding; // may wrap, as in CUDA
    const uint memRequiredPerPair = calculateConvexMeshPairMemRequirement();
    return memAvailableAfterPadding / memRequiredPerPair;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctMidphaseAlloc(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const MaIn e = gIn[i];

    // midphaseAllocate: pointer bumps from the stack base become byte offsets
    const uint numPairsAfterMP = (e.numpairsAfterMPUnpadded + 3) & (~3u);

    uint p = 0;
    const uint offPairs = p;
    p += numPairsAfterMP * SIZEOF_UINT4;
    const uint offNI = p;
    p += numPairsAfterMP * SIZEOF_CONVEXTRI_NORMAL_AND_INDEX;
    const uint offContacts = p;
    p += numPairsAfterMP * SIZEOF_CONVEXTRI_CONTACTS;
    const uint offOrdered = p;
    p += e.numPaddedPairsAfterMP * SIZEOF_U32;
    const uint offMaxDepth = p;
    p += numPairsAfterMP * SIZEOF_REAL;
    const uint offInterm = p;
    p += numPairsAfterMP * SIZEOF_CONVEXTRI_INTERMEDIATE;
    const uint offSecondPass = p;

    MaOut o;
    o.offsetsA = uint4(offPairs, offNI, offContacts, offOrdered);
    o.offsetsB = uint4(offMaxDepth, offInterm, offSecondPass, calculateConvexMeshPairMemRequirement());
    o.sizes = uint4(calculateAdditionalPadding(e.numContactManagers),
                    calculateMaxPairs(e.stackSizeBytes, e.numContactManagers),
                    numPairsAfterMP, 0);
    gOut[i] = o;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctCvxTriAccessors(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const uint2 e = gAcc[i];
    gAccOut[i] = uint4(ctniGetTriangleIndex(e.x),
                       ctniGetNbContacts(e.x),
                       (ctniIsDeferredContact(e.x) ? 1u : 0u) | (ctidIsEdgeContact(e.y) ? 2u : 0u),
                       0);
}
