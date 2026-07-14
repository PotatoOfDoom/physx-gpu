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

// Verification of npRemapIndex.slang: computeRemapIndexRigidBody
// (constraintPrepShared.cuh L51-161) against an exact CPU transcription.
// Pure PxU32 arithmetic -- ALL outputs bitwise, incl. the
// solverBodyReferences side-effect write (surfaced as refId/writtenValue).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };

struct RmIn
{
    U4 m0;  // isSecondBody, cType, partitionIndex, partitionEntryIndex
    U4 m1;  // currPartition, maxNbPartitions, bodyId, activeBodyOffset
    U4 m2;  // totalRigidBatches, totalArticBatches, nbElemsPerBody, nbSlabs
    uint32_t partStart[8];
    uint32_t artiStart[8];
    uint32_t jointCounts[8];
    uint32_t artiJointCounts[8];
};

struct RmOut
{
    U4 o0;  // finalIndex, wrote, referenceId, writtenValue
    U4 o1;  // spreadReferenceId, 0, 0, 0
};

// exact transcription of computeRemapIndexRigidBody (constraintPrepShared.cuh L51-161)
static void refRemap(const RmIn& s, RmOut& o)
{
    std::memset(&o, 0, sizeof(o));

    const uint32_t partitionIndex = s.m0.z;
    const uint32_t partitionEntryIndex = s.m0.w;

    uint32_t index = 0;  // CUDA-uninitialized for cType > 3 (unreachable, fixtures 0-3)
    switch (s.m0.y)
    {
    case 0:  // eCONTACT_MANAGER
        index = s.partStart[partitionIndex] + s.jointCounts[partitionIndex] +
                partitionEntryIndex / 32u;
        break;
    case 1:  // eCONSTRAINT
        index = s.partStart[partitionIndex] + partitionEntryIndex / 32u;
        break;
    case 2:  // eARTICULATION_CONTACT
        index = s.m2.x + s.artiStart[partitionIndex] + s.artiJointCounts[partitionIndex] +
                partitionEntryIndex / 32u;
        break;
    case 3:  // eARTICULATION_CONSTRAINT
        index = s.m2.x + s.artiStart[partitionIndex] + partitionEntryIndex / 32u;
        break;
    default:
        break;
    }

    const uint32_t batchMask = 32u - 1u;
    index = (index * 32u * 2u) * s.m2.z + (partitionEntryIndex & batchMask);
    if (s.m0.x)
        index += 32u * s.m2.z;

    if (s.m1.z >= s.m1.w)  // bodyId >= activeBodyOffset
    {
        const uint32_t maxNbPartitions = s.m1.y;
        if ((partitionIndex & (~(maxNbPartitions - 1u))) != (s.m1.x & (~(maxNbPartitions - 1u))) ||
            partitionIndex <= s.m1.x)
        {
            const uint32_t slabId = partitionIndex / maxNbPartitions;
            uint32_t referenceId = s.m2.w * (s.m1.z - s.m1.w) + slabId;

            o.o0.z = referenceId;
            o.o0.w = index;  // solverBodyReferences[referenceId].mRemappedBodyIndex
            o.o0.y = 1;

            referenceId = (referenceId & (~31u)) * s.m2.z + (referenceId & 31u);
            o.o1.x = referenceId;

            index = (s.m2.y + s.m2.x) * 32u * 2u * s.m2.z + referenceId;
        }
    }

    o.o0.x = index;
}

PXG_TEST(npRemapIndex_remap)
{
    std::mt19937 rng(176001);
    const uint32_t N = 6000;

    std::vector<RmIn> in(N);
    std::vector<RmOut> ref(N);
    size_t cType[4] = { 0, 0, 0, 0 };
    size_t cWrote = 0, cNoWrite = 0, cInactive = 0, cSecond = 0, cEqPart = 0, cGroupChange = 0,
           cBigRef = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        RmIn s{};
        const uint32_t maxNb = (rng() % 2) ? 2u : 4u;
        const uint32_t nbSlabs = 1u + rng() % 2u;
        const uint32_t maxPart = maxNb * nbSlabs;  // <= 8

        s.m0 = U4{ rng() % 2u, rng() % 4u, rng() % maxPart, rng() % 256u };
        s.m1 = U4{ rng() % maxPart, maxNb, rng() % 64u, rng() % 64u };
        s.m2 = U4{ rng() % 100u, rng() % 100u, 1u + rng() % 3u, nbSlabs };

        if (i % 5 == 2) s.m1.w = s.m1.z;                 // bodyId == activeBodyOffset (>= edge)
        if (i % 7 == 3) s.m0.z = s.m1.x;                 // partitionIndex == currPartition (<= edge)
        if (i % 11 == 4)
        {
            // force a partition-GROUP change with partitionIndex > currPartition
            // (the || left side alone)
            s.m0.z = maxNb + rng() % maxNb;              // group 1
            s.m1.x = rng() % maxNb;                      // group 0
            if (s.m0.z < maxPart) ++cGroupChange;
        }
        if (i % 13 == 6)
        {
            // same group, partitionIndex > currPartition: NO write
            s.m1.x = rng() % (maxNb - 1u);
            s.m0.z = s.m1.x + 1u;
        }
        if (i % 17 == 8)
        {
            // large body ids: referenceId >= 32 exercises the & ~31 spread
            s.m1.z = 40u + rng() % 200u;
            s.m1.w = rng() % 8u;
        }

        for (int p = 0; p < 8; ++p)
        {
            s.partStart[p] = rng() % 500u;
            s.artiStart[p] = rng() % 500u;
            s.jointCounts[p] = rng() % 60u;
            s.artiJointCounts[p] = rng() % 60u;
        }

        in[i] = s;
        refRemap(s, ref[i]);
        ++cType[s.m0.y];
        if (ref[i].o0.y) ++cWrote; else if (s.m1.z >= s.m1.w) ++cNoWrite; else ++cInactive;
        if (s.m0.x) ++cSecond;
        if (s.m0.z == s.m1.x) ++cEqPart;
        if (ref[i].o0.z >= 32u) ++cBigRef;
    }

    std::printf("[ info ] rm cov: wrote %zu nowrite %zu inactive %zu second %zu eqPart %zu "
                "group %zu bigRef %zu types %zu/%zu/%zu/%zu\n",
                cWrote, cNoWrite, cInactive, cSecond, cEqPart, cGroupChange, cBigRef, cType[0],
                cType[1], cType[2], cType[3]);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<RmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npRemapIndex.spv", "remapIndex",
             { (uint32_t)sizeof(RmIn), (uint32_t)sizeof(RmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<RmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (std::memcmp(&ref[i], &g[i], sizeof(RmOut)) != 0)
        {
            if (mm < 5)
                std::printf("MM %u R (%u %u %u %u | %u) G (%u %u %u %u | %u)\n", i, ref[i].o0.x,
                            ref[i].o0.y, ref[i].o0.z, ref[i].o0.w, ref[i].o1.x, g[i].o0.x,
                            g[i].o0.y, g[i].o0.z, g[i].o0.w, g[i].o1.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "remapIndex: %zu mismatches", mm);

    bool cov = cWrote > N / 4 && cNoWrite > N / 20 && cInactive > N / 10 && cSecond > N / 3 &&
               cEqPart > 100 && cGroupChange > 100 && cBigRef > 100;
    for (int t = 0; t < 4; ++t) cov = cov && cType[t] > N / 8;
    PXG_CHECK_MSG(cov, "remapIndex: coverage thin (wrote %zu nowrite %zu inactive %zu bigRef %zu)",
                  cWrote, cNoWrite, cInactive, cBigRef);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
