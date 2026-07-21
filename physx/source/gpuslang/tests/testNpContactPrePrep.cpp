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

// Verification of npContactPrePrep.slang:
//   1. cppPack — the per-lane contact-patch packing core of
//      createSolverContactConstraintDescsFromPatchWithSharedMem
//      (constraintBlockPrePrep.cu L167-253) + the contact emission (L304-319)
//      + mWriteback (L1044), against an exact CPU transcription.  Pure
//      copies/flag logic -- ALL outputs bitwise (memcmp).
//   2. constraintContactBlockPrePrepLaunch — the FULL kernel on the ENGINE
//      launch ABI: REAL PxgPrePrepDesc + PxgSolverSharedDesc<IterativeSolveData>
//      filled with live device addresses, launched through PxgCudaShim with
//      the engine geometry (grid = ceil(totalBatches*32/128), block 128).
//      The lane-0 atomicAdd allocations (sharedContact/Friction/
//      ArticulationResponse indices) are scheduling-order dependent across
//      warps, so the test first validates that the GPU-allocated ranges
//      exactly partition the counter deltas, then transcribes the CPU
//      reference AT the GPU-allocated bases -- everything else bitwise, with
//      sentinel-filled outputs proving the exact write set.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgSolverConstraintDesc.h"
#include "PxgConstraintPrep.h"
#include "PxgPartitionNode.h"
#include "PxgBodySim.h"
#include "PxsContactManagerState.h"
#include "PxContact.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

const uint32_t MAX_PTS = 20;

struct CpIn
{
    U4 meta;   // hasPatch, wuFlags, internalFlags, materialFlags
    F4 mass;
    F4 nr;
    F4 mat;    // dynamicFriction, staticFriction, damping, restDistance
    U4 cnt;    // startContactIndex, nbContacts, prevFrictionPatchCount, forceIndex
    F4 pts[MAX_PTS];
};

struct CpOut
{
    F4 invMassOut;
    F4 nrOut;
    F4 dampDataOut;
    F4 dataOut;
    U4 counts;
    F4 pointsOut[12];
};

static float asFl(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// exact transcription (constraintBlockPrePrep.cu L167-253 + L304-319 + L1044)
static void refPack(const CpIn& s, CpOut& o)
{
    std::memset(&o, 0, sizeof(o));

    uint32_t numContacts = 0, stride = 1, contactStartIndex = 0, ptOffset = 0;
    float dynamicFriction = 0.f, staticFriction = 0.f, damping = 0.f;
    uint32_t solverFlags = 0;
    bool perPointFriction = false;

    if (s.meta.x)
    {
        perPointFriction = (s.meta.z & 16u) != 0 || (s.meta.y & (1u << 2)) != 0;
        bool disableFriction = false;
        bool isAccelerationSpring = false;

        if ((s.meta.z & 4u) == 0)  // eFORCE_NO_RESPONSE
        {
            const bool isModifiable = (s.meta.z & 128u) != 0;
            ptOffset = s.cnt.x;
            if (isModifiable)
            {
                ptOffset += s.cnt.x;
                stride = 2;
            }
            numContacts = s.cnt.y;
            contactStartIndex = s.cnt.x;

            F4 m = s.mass;
            if (s.meta.y & (1u << 14)) { m.x = 0.f; m.y = 0.f; }
            if (s.meta.y & (1u << 15)) { m.z = 0.f; m.w = 0.f; }
            o.invMassOut = m;
            o.nrOut = s.nr;
            damping = s.mat.z;
            dynamicFriction = s.mat.x;
            staticFriction = s.mat.y;
            disableFriction = (s.meta.w & 1u) != 0;
            isAccelerationSpring = (s.meta.w & 16u) != 0;
        }

        if (perPointFriction) solverFlags |= 2u;
        if (s.meta.y & (1u << 9)) solverFlags |= 1u;
        if (disableFriction) solverFlags |= 4u;
        if (isAccelerationSpring) solverFlags |= 8u;
    }

    const uint32_t packed = (numContacts & 0xFFu) | (solverFlags << 8) | ((s.cnt.z & 0xFFu) << 16);
    o.dataOut = F4{ s.mat.w, staticFriction, dynamicFriction, asFl(packed) };
    o.dampDataOut = F4{ damping, asFl(s.cnt.w + contactStartIndex), 0.f, 0.f };
    o.counts = U4{ numContacts, contactStartIndex, stride, perPointFriction ? 1u : 0u };

    uint32_t emit = 0;
    for (uint32_t a = 0; a < numContacts * stride; a += stride)
    {
        o.pointsOut[emit * 2 + 0] = s.pts[ptOffset + a];
        if (stride != 1)
            o.pointsOut[emit * 2 + 1] = s.pts[ptOffset + a + 1];
        else
            o.pointsOut[emit * 2 + 1] = F4{ 0.f, 0.f, 0.f, 3.402823466e+38f };
        ++emit;
    }
}

PXG_TEST(npContactPrePrep_pack)
{
    std::mt19937 rng(177001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CpIn> in(N);
    std::vector<CpOut> ref(N);
    size_t cNoPatch = 0, cNoResp = 0, cMod = 0, cPlain = 0, cDom0 = 0, cDom1 = 0, cPPF = 0,
           cThresh = 0, cDisFric = 0, cAccel = 0, cNoForce = 0, cEmit = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CpIn s{};
        const bool hasPatch = (i % 9 != 4);

        // work-unit flags: salt irrelevant bits + the four load-bearing ones
        uint32_t wu = (rng() % 2 ? (1u << 0) : 0u) | (rng() % 2 ? (1u << 5) : 0u) |
                      (rng() % 2 ? (1u << 8) : 0u) | (rng() % 2 ? (1u << 10) : 0u);
        if (rng() % 3 == 0) wu |= (1u << 2);    // eDISABLE_STRONG_FRICTION
        if (rng() % 2) wu |= (1u << 9);         // eFORCE_THRESHOLD
        if (rng() % 3 == 0) wu |= (1u << 14);   // eDOMINANCE_0
        if (rng() % 3 == 0) wu |= (1u << 15);   // eDOMINANCE_1

        // patch internal flags: salt irrelevant bits (1<<0, 1<<3, 1<<5, 1<<6)
        // + gates (review-177: neighbor bits of every tested mask salted)
        uint32_t internalFlags = (rng() % 2 ? 1u : 0u) | (rng() % 2 ? 8u : 0u) |
                                 (rng() % 2 ? 32u : 0u) | (rng() % 2 ? 64u : 0u);
        if (i % 7 == 3) internalFlags |= 4u;    // eFORCE_NO_RESPONSE
        if (rng() % 4 == 0) internalFlags |= 16u;   // eHAS_TARGET_VELOCITY
        if (i % 3 == 1) internalFlags |= 128u;  // eCOMPRESSED_MODIFIED_CONTACT

        uint32_t materialFlags = (rng() % 2 ? 2u : 0u) | (rng() % 2 ? 4u : 0u) |
                                 (rng() % 2 ? 8u : 0u) | (rng() % 2 ? 32u : 0u);  // salt bits
        if (rng() % 3 == 0) materialFlags |= 1u;    // eDISABLE_FRICTION
        if (rng() % 3 == 0) materialFlags |= 16u;   // eCOMPLIANT_ACCELERATION_SPRING

        s.meta = U4{ hasPatch ? 1u : 0u, wu, internalFlags, materialFlags };
        s.mass = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f) };
        s.nr = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(0.f, 1.f) };
        s.mat = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 100.f), rf(-0.1f, 0.5f) };

        const bool modifiable = (internalFlags & 128u) != 0;
        const uint32_t maxStart = modifiable ? 4u : 8u;   // keep pts in range
        const uint32_t startIdx = rng() % (maxStart + 1u);
        const uint32_t nb = rng() % 7u;                   // 0..6 contacts
        const uint32_t forceIdx = (i % 5 == 2) ? 0xFFFFFFFFu : rng() % 100000u;
        // prevFriction up to 399: pins the & 0xFF truncation (CUDA's PxU8
        // cast; review-177 x4 showed the mask was dead under < 256 fixtures)
        s.cnt = U4{ startIdx, nb, rng() % 400u, forceIdx };

        for (uint32_t p = 0; p < MAX_PTS; ++p)
            s.pts[p] = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-1, 1) };

        in[i] = s;
        refPack(s, ref[i]);

        if (!hasPatch) ++cNoPatch;
        else if (internalFlags & 4u) ++cNoResp;
        else if (modifiable) ++cMod;
        else ++cPlain;
        if (ref[i].counts.w) ++cPPF;
        if (hasPatch && (wu & (1u << 14))) ++cDom0;
        if (hasPatch && (wu & (1u << 15))) ++cDom1;
        if (wu & (1u << 9)) ++cThresh;
        if (hasPatch && !(internalFlags & 4u) && (materialFlags & 1u)) ++cDisFric;
        if (hasPatch && !(internalFlags & 4u) && (materialFlags & 16u)) ++cAccel;
        if (forceIdx == 0xFFFFFFFFu) ++cNoForce;
        cEmit += ref[i].counts.x;
    }

    std::printf("[ info ] cpp cov: nopatch %zu noresp %zu mod %zu plain %zu dom0 %zu dom1 %zu "
                "ppf %zu thresh %zu disfric %zu accel %zu noforce %zu emitted %zu\n",
                cNoPatch, cNoResp, cMod, cPlain, cDom0, cDom1, cPPF, cThresh, cDisFric, cAccel,
                cNoForce, cEmit);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npContactPrePrep.spv", "cppPack",
             { (uint32_t)sizeof(CpIn), (uint32_t)sizeof(CpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (std::memcmp(&ref[i], &g[i], sizeof(CpOut)) != 0)
        {
            if (mm < 5)
                std::printf("MM %u nR %u nG %u dataR (%g %g %g %08x) dataG (%g %g %g %08x)\n", i,
                            ref[i].counts.x, g[i].counts.x, ref[i].dataOut.x, ref[i].dataOut.y,
                            ref[i].dataOut.z, *(const uint32_t*)&ref[i].dataOut.w, g[i].dataOut.x,
                            g[i].dataOut.y, g[i].dataOut.z, *(const uint32_t*)&g[i].dataOut.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "cppPack: %zu mismatches", mm);

    bool cov = cNoPatch > 500 && cNoResp > 400 && cMod > 1000 && cPlain > 2000 && cDom0 > 800 &&
               cDom1 > 800 && cPPF > 1000 && cThresh > 2000 && cDisFric > 700 && cAccel > 700 &&
               cNoForce > 800 && cEmit > 8000;
    PXG_CHECK_MSG(cov, "cppPack: coverage thin (nopatch %zu noresp %zu mod %zu emit %zu)",
                  cNoPatch, cNoResp, cMod, cEmit);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// the FULL constraintContactBlockPrePrepLaunch kernel on the ENGINE ABI
// ---------------------------------------------------------------------------

static std::vector<uint8_t> cppShimLoader(const char* file) { return loadShaderBlob(file); }

// mirrors computeRemapIndexRigidBody() (constraintPrepShared.cuh L51-161)
static uint32_t refRemapIndexRigidBody(bool isSecondBody, const std::vector<uint32_t>& partStart,
                                       const std::vector<uint32_t>& partArtiStart,
                                       const std::vector<uint32_t>& partJointCounts,
                                       const std::vector<uint32_t>& partArtiJointCounts,
                                       const physx::PartitionIndexData& indexData,
                                       std::vector<uint32_t>& solverBodyReferences,
                                       std::vector<uint32_t>& refCount, uint32_t currPartition,
                                       uint32_t maxNbPartitions, uint32_t bodyId, uint32_t activeBodyOffset,
                                       uint32_t totalRigidBatches, uint32_t totalArticBatches,
                                       uint32_t nbElemsPerBody, uint32_t nbSlabs)
{
    const uint32_t partitionIndex = indexData.mPartitionIndex;
    const uint32_t partitionEntryIndex = indexData.mPartitionEntryIndex;
    uint32_t index = 0;
    switch (indexData.mCType)
    {
    case 0: // eCONTACT_MANAGER
        index = partStart[partitionIndex] + partJointCounts[partitionIndex] + partitionEntryIndex / 32;
        break;
    case 1: // eCONSTRAINT
        index = partStart[partitionIndex] + partitionEntryIndex / 32;
        break;
    case 2: // eARTICULATION_CONTACT
        index = totalRigidBatches + partArtiStart[partitionIndex] + partArtiJointCounts[partitionIndex] +
                partitionEntryIndex / 32;
        break;
    case 3: // eARTICULATION_CONSTRAINT
        index = totalRigidBatches + partArtiStart[partitionIndex] + partitionEntryIndex / 32;
        break;
    default:
        break;
    }
    index = (index * 32 * 2) * nbElemsPerBody + (partitionEntryIndex & 31);
    if (isSecondBody)
        index += 32 * nbElemsPerBody;
    if (bodyId >= activeBodyOffset)
    {
        if ((partitionIndex & (~(maxNbPartitions - 1))) != (currPartition & (~(maxNbPartitions - 1))) ||
            partitionIndex <= currPartition)
        {
            const uint32_t slabId = partitionIndex / maxNbPartitions;
            uint32_t referenceId = nbSlabs * (bodyId - activeBodyOffset) + slabId;
            solverBodyReferences[referenceId] = index;
            refCount[referenceId]++;
            referenceId = (referenceId & (~31u)) * nbElemsPerBody + (referenceId & 31u);
            index = (totalArticBatches + totalRigidBatches) * 32 * 2 * nbElemsPerBody + referenceId;
        }
    }
    return index;
}

PXG_TEST(npContactPrePrep_engineKernel)
{
    using namespace physx;

    std::mt19937 rng(52001);
    std::uniform_real_distribution<float> uf(0.f, 1.f), sf(-1.f, 1.f);

    // ---- scenario layout (see kernel comment): 3 rigid contact batches + 2
    // joint batches (skipped) in the dynamic rigid region, 2 arti contact
    // batches, 2 static rigid contact batches ----
    const uint32_t numBatches = 5, numArtiBatches = 2, numStaticBatches = 2;
    const uint32_t totalBatches = numBatches + numArtiBatches + numStaticBatches; // 9
    const uint32_t dynamicBatchEnd = numBatches + numArtiBatches;                 // 7
    const bool isContactBatch[9] = { true, false, true, false, true, true, true, true, true };
    const uint32_t conTypeOf[9] = { 1, 0, 1, 0, 1, 3, 3, 1, 1 };

    const uint32_t numNodes = 168;
    const uint32_t totalActiveBodies = 24, activeBodyOffset = 6;
    const uint32_t maxConstraintPartitions = 8, nbSlabs = 2, nbElemsPerBody = 2;
    const uint32_t nbArticulations = 4;
    const uint32_t numPartitions = 64;
    const uint32_t nUnique = totalBatches * 32 * 2; // rigid slots then arti slots
    const uint32_t nCm = 1024;                      // >= cmOffsets2 + nUnique
    const uint32_t nEdges = nUnique;
    const uint32_t contactPointCap = 128; // PxgBlockContactPoint elements
    const uint32_t nBlockData = 16;       // blockWorkUnit / blockContactData elements
    const uint32_t refCap = 64;           // solverBodyReferences
    const uint32_t slabCap = 256;         // articulationSlabMask uint4s

    const uint64_t cpuPatchesBase = 0x4000000000ull;
    const uint64_t cpuContactsBase = 0x5000000000ull;
    const uint64_t cpuForcesBase = 0x6000000000ull;

    // ---- host input data ----
    std::vector<PxgConstraintBatchHeader> headers(totalBatches);
    std::memset(headers.data(), 0, headers.size() * sizeof(PxgConstraintBatchHeader));
    uint32_t cbiCursor = 0;
    for (uint32_t k = 0; k < totalBatches; ++k)
    {
        headers[k].mDescStride = PxU16(isContactBatch[k] ? (4 + rng() % 29) : (1 + rng() % 32));
        headers[k].constraintType = PxU16(conTypeOf[k]);
        headers[k].mConstraintBatchIndex = isContactBatch[k] ? cbiCursor++ : (rng() % nBlockData);
        headers[k].mStartPartitionIndex = k * 32;
        headers[k].mask = rng();
    }

    std::vector<uint32_t> contactUnique((totalBatches + 1) * 32), artiContactUnique((totalBatches + 1) * 32);
    for (uint32_t i = 0; i < contactUnique.size(); ++i)
    {
        contactUnique[i] = i % (totalBatches * 32);
        artiContactUnique[i] = totalBatches * 32 + (i % (totalBatches * 32));
    }

    std::vector<uint32_t> solverBodyIndices(numNodes);
    for (uint32_t n = 0; n < numNodes; ++n)
        solverBodyIndices[n] = rng() % totalActiveBodies;

    std::vector<PartitionNodeData> nodeData(nUnique);
    std::memset(nodeData.data(), 0, nodeData.size() * sizeof(PartitionNodeData));
    std::vector<PartitionIndexData> indexData(nUnique);
    std::memset(indexData.data(), 0, indexData.size() * sizeof(PartitionIndexData));
    std::vector<PxgSolverConstraintManagerConstants> constants(nUnique);
    std::vector<uint32_t> npOutputIndices(nUnique);
    for (uint32_t q = 0; q < nUnique; ++q)
    {
        const bool artiSlot = q >= totalBatches * 32;
        // node indices: mostly dynamic rigid; some static (PX_INVALID_NODE);
        // arti slots reference articulation nodes
        const bool staticA = !artiSlot && (rng() % 6 == 0);
        const bool staticB = !artiSlot && (rng() % 4 == 0);
        const uint32_t nodeA = rng() % numNodes;
        const uint32_t nodeB = rng() % numNodes;
        const bool artiA = artiSlot;                   // articulation on A for arti slots
        const bool artiB = artiSlot && (rng() % 2 == 0);
        nodeData[q].mNodeIndex0 = staticA ? PxNodeIndex() : (artiA ? PxNodeIndex(nodeA, rng() % 7) : PxNodeIndex(nodeA));
        nodeData[q].mNodeIndex1 = staticB ? PxNodeIndex() : (artiB ? PxNodeIndex(nodeB, rng() % 7) : PxNodeIndex(nodeB));
        nodeData[q].mNextIndex[0] = ((rng() % nUnique) << 1) | (rng() & 1);
        nodeData[q].mNextIndex[1] = ((rng() % nUnique) << 1) | (rng() & 1);

        indexData[q].mPartitionIndex = PxU16(rng() % numPartitions);
        indexData[q].mPatchIndex = PxU8(rng() % 3);
        indexData[q].mCType = PxU8(rng() % 4);
        indexData[q].mPartitionEntryIndex = rng() % 4096;

        constants[q].mEdgeIndex = q;
        constants[q].mConstraintWriteBackIndex = rng();

        const uint32_t bucket = q % 3;
        npOutputIndices[q] = ((q) << 7) | bucket;
    }

    std::vector<uint32_t> partStart(numPartitions), partArtiStart(numPartitions),
        partJointCounts(numPartitions), partArtiJointCounts(numPartitions);
    for (uint32_t pi = 0; pi < numPartitions; ++pi)
    {
        partStart[pi] = rng() % 100;
        partArtiStart[pi] = rng() % 100;
        partJointCounts[pi] = rng() % 10;
        partArtiJointCounts[pi] = rng() % 10;
    }

    const uint32_t cmOffsets0 = 0, cmOffsets1 = 100, cmOffsets2 = 200; // buckets 0..2; cmIdx < nCm

    std::vector<PxsContactManagerOutput> cmOut(nCm);
    std::memset(cmOut.data(), 0, cmOut.size() * sizeof(PxsContactManagerOutput));
    std::vector<float> restDistances(nCm);
    std::vector<PxsTorsionalFrictionData> torsional(nCm);
    std::vector<uint64_t> shapeInteractions(nCm);
    for (uint32_t m = 0; m < nCm; ++m)
    {
        cmOut[m].contactPatches = reinterpret_cast<PxU8*>(cpuPatchesBase + uint64_t(m) * 4 * 64);
        cmOut[m].contactPoints = reinterpret_cast<PxU8*>(cpuContactsBase + uint64_t(m) * 32 * 16);
        cmOut[m].contactForces = (rng() % 4 != 0)
            ? reinterpret_cast<PxReal*>(cpuForcesBase + uint64_t(m) * 8 * 4) : NULL;
        cmOut[m].nbPatches = PxU8(rng() % 4);
        cmOut[m].nbContacts = PxU16((rng() % 5 == 0) ? 0 : (1 + rng() % 8));
        cmOut[m].flags = PxU16(rng() & 0xFFFF);
        restDistances[m] = sf(rng);
        torsional[m] = PxsTorsionalFrictionData(uf(rng), uf(rng));
        shapeInteractions[m] = 0x7000000000ull + m * 0x30ull;
    }

    std::vector<PxContactPatch> patches(nCm * 4 + 4);
    std::memset(patches.data(), 0, patches.size() * sizeof(PxContactPatch));
    for (auto& cp : patches)
    {
        cp.mMassModification.linear0 = uf(rng);
        cp.mMassModification.angular0 = uf(rng);
        cp.mMassModification.linear1 = uf(rng);
        cp.mMassModification.angular1 = uf(rng);
        cp.normal = PxVec3(sf(rng), sf(rng), sf(rng));
        cp.restitution = sf(rng);
        cp.dynamicFriction = uf(rng);
        cp.staticFriction = uf(rng);
        cp.damping = uf(rng);
        cp.startContactIndex = PxU16(rng() % 8);
        cp.nbContacts = PxU8(rng() % 5);
        cp.materialFlags = PxU8((rng() % 2 ? 1 : 0) | (rng() % 3 == 0 ? 16 : 0));
        uint16_t internalFlags = 0;
        if (rng() % 4 == 0) internalFlags |= PxContactPatch::eHAS_TARGET_VELOCITY;
        if (rng() % 5 == 0) internalFlags |= PxContactPatch::eFORCE_NO_RESPONSE;
        if (rng() % 3 == 0) internalFlags |= PxContactPatch::eCOMPRESSED_MODIFIED_CONTACT;
        cp.internalFlags = internalFlags;
    }

    std::vector<float> contactStream(nCm * 32 * 4); // float4s as raw floats
    for (auto& v : contactStream) v = sf(rng) * 4.f;

    std::vector<PxgBodySim> bodySims(numNodes);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(PxgBodySim));
    for (uint32_t n = 0; n < numNodes; ++n)
        bodySims[n].articulationRemapId = 100 + n;

    std::vector<uint32_t> prevFric(nEdges);
    for (auto& v : prevFric) v = rng() % 5;

    // ---- sentinel-filled outputs ----
    const uint32_t SENT = 0xABABABABu;
    std::vector<uint32_t> blockBatchesInit(totalBatches * sizeof(PxgBlockConstraintBatch) / 4, SENT);
    std::vector<uint32_t> workUnitInit(nBlockData * sizeof(PxgBlockWorkUnit) / 4, SENT);
    std::vector<uint32_t> contactDataInit(nBlockData * sizeof(physx::PxgBlockContactData) / 4, SENT);
    std::vector<uint32_t> contactPointsInit(contactPointCap * sizeof(physx::PxgBlockContactPoint) / 4, SENT);
    std::vector<uint32_t> currFricInit(nEdges, SENT);
    std::vector<uint32_t> bodyRefInit(refCap, SENT);
    std::vector<uint32_t> slabMaskInit(slabCap * 4, SENT);

    // ---- device buffers ----
    PxgSlangComputeDevice* dev = testDevice();
    const physx::gpuslang::PxgKernelDef manifest[] = {
        { "npContactPrePrep.spv", "constraintContactBlockPrePrepLaunch", 16u, 0 },
    };
    physx::gpuslang::PxgCudaShim shim(dev, manifest, 1, &cppShimLoader);
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data)
            shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    const PxgDevicePtr dHeaders = alloc(headers.data(), headers.size() * sizeof(PxgConstraintBatchHeader));
    const PxgDevicePtr dCU = alloc(contactUnique.data(), contactUnique.size() * 4);
    const PxgDevicePtr dACU = alloc(artiContactUnique.data(), artiContactUnique.size() * 4);
    const PxgDevicePtr dSBI = alloc(solverBodyIndices.data(), solverBodyIndices.size() * 4);
    const PxgDevicePtr dND = alloc(nodeData.data(), nodeData.size() * sizeof(PartitionNodeData));
    const PxgDevicePtr dID = alloc(indexData.data(), indexData.size() * sizeof(PartitionIndexData));
    const PxgDevicePtr dConst = alloc(constants.data(), constants.size() * 8);
    const PxgDevicePtr dNpIdx = alloc(npOutputIndices.data(), npOutputIndices.size() * 4);
    const PxgDevicePtr dPS = alloc(partStart.data(), partStart.size() * 4);
    const PxgDevicePtr dPAS = alloc(partArtiStart.data(), partArtiStart.size() * 4);
    const PxgDevicePtr dPJC = alloc(partJointCounts.data(), partJointCounts.size() * 4);
    const PxgDevicePtr dPAJC = alloc(partArtiJointCounts.data(), partArtiJointCounts.size() * 4);
    const PxgDevicePtr dCmOut = alloc(cmOut.data(), cmOut.size() * sizeof(PxsContactManagerOutput));
    const PxgDevicePtr dRest = alloc(restDistances.data(), restDistances.size() * 4);
    const PxgDevicePtr dTors = alloc(torsional.data(), torsional.size() * 8);
    const PxgDevicePtr dShapes = alloc(shapeInteractions.data(), shapeInteractions.size() * 8);
    const PxgDevicePtr dPatches = alloc(patches.data(), patches.size() * sizeof(PxContactPatch));
    const PxgDevicePtr dContacts = alloc(contactStream.data(), contactStream.size() * 4);
    const PxgDevicePtr dBodySims = alloc(bodySims.data(), bodySims.size() * sizeof(PxgBodySim));
    const PxgDevicePtr dPrevFric = alloc(prevFric.data(), prevFric.size() * 4);
    const PxgDevicePtr dBB = alloc(blockBatchesInit.data(), blockBatchesInit.size() * 4);
    const PxgDevicePtr dWU = alloc(workUnitInit.data(), workUnitInit.size() * 4);
    const PxgDevicePtr dCD = alloc(contactDataInit.data(), contactDataInit.size() * 4);
    const PxgDevicePtr dCP = alloc(contactPointsInit.data(), contactPointsInit.size() * 4);
    const PxgDevicePtr dCurrFric = alloc(currFricInit.data(), currFricInit.size() * 4);
    const PxgDevicePtr dBodyRef = alloc(bodyRefInit.data(), bodyRefInit.size() * 4);
    const PxgDevicePtr dSlab = alloc(slabMaskInit.data(), slabMaskInit.size() * 4);
    if (!dHeaders) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the REAL engine descriptors ----
    const uint32_t initContactIdx = 11, initFrictionIdx = 7, initArtiResponseIdx = 3;

    PxgPrePrepDesc pre;
    std::memset(&pre, 0xCD, sizeof(pre));
    pre.blockBatches = reinterpret_cast<PxgBlockConstraintBatch*>(dBB);
    pre.numBatches = numBatches;
    pre.numStaticBatches = numStaticBatches;
    pre.numArtiBatches = numArtiBatches;
    pre.numArtiStaticBatches = 0;
    pre.numArtiSelfBatches = 0;
    pre.blockWorkUnit = reinterpret_cast<PxgBlockWorkUnit*>(dWU);
    pre.blockContactData = reinterpret_cast<physx::PxgBlockContactData*>(dCD);
    pre.blockContactPoints = reinterpret_cast<physx::PxgBlockContactPoint*>(dCP);
    pre.compressedContacts = reinterpret_cast<PxContact*>(dContacts);
    pre.compressedPatches = reinterpret_cast<PxContactPatch*>(dPatches);
    pre.cpuCompressedContactsBase = reinterpret_cast<PxContact*>(cpuContactsBase);
    pre.cpuCompressedPatchesBase = reinterpret_cast<PxContactPatch*>(cpuPatchesBase);
    pre.cpuForceBufferBase = reinterpret_cast<PxReal*>(cpuForcesBase);
    pre.contactManagerOutputBase = reinterpret_cast<PxsContactManagerOutput*>(dCmOut);
    pre.sharedJointRowIndex = 0;
    pre.sharedFrictionConstraintIndex = initFrictionIdx;
    pre.sharedContactConstraintIndex = initContactIdx;
    pre.sharedArticulationResponseIndex = initArtiResponseIdx;
    pre.solverBodyIndices = reinterpret_cast<PxU32*>(dSBI);
    pre.mPartitionIndices = reinterpret_cast<PartitionIndexData*>(dID);
    pre.mPartitionstartBatchIndices = reinterpret_cast<PxU32*>(dPS);
    pre.mPartitionArtiStartBatchIndices = reinterpret_cast<PxU32*>(dPAS);
    pre.mPartitionJointCounts = reinterpret_cast<PxU32*>(dPJC);
    pre.mPartitionArtiJointCounts = reinterpret_cast<PxU32*>(dPAJC);
    pre.prevFrictionPatchCount = reinterpret_cast<PxU32*>(dPrevFric);
    pre.currFrictionPatchCount = reinterpret_cast<PxU32*>(dCurrFric);
    pre.mNpOutputIndices = reinterpret_cast<PxU32*>(dNpIdx);
    for (uint32_t bkt = 0; bkt < GPU_BUCKET_ID::eCount; ++bkt)
        pre.mCmOutputOffsets[bkt] = 0;
    pre.mCmOutputOffsets[0] = cmOffsets0;
    pre.mCmOutputOffsets[1] = cmOffsets1;
    pre.mCmOutputOffsets[2] = cmOffsets2;
    pre.mPartitionNodeData = reinterpret_cast<PartitionNodeData*>(dND);
    pre.mContactConstantData = reinterpret_cast<PxgSolverConstraintManagerConstants*>(dConst);
    pre.mBatchHeaders = reinterpret_cast<PxgConstraintBatchHeader*>(dHeaders);
    pre.mContactUniqueIndices = reinterpret_cast<PxU32*>(dCU);
    pre.mArtiContactUniqueIndices = reinterpret_cast<PxU32*>(dACU);
    pre.mSolverBodyReferences = reinterpret_cast<PxgSolverReferences*>(dBodyRef);
    pre.mMaxConstraintPartitions = maxConstraintPartitions;
    pre.mTotalSlabs = nbSlabs;
    pre.mTotalActiveBodies = totalActiveBodies;
    pre.mTotalActiveArticulations = nbArticulations;
    pre.mActiveBodyStartOffset = activeBodyOffset;
    pre.nbElementsPerBody = nbElemsPerBody;
    pre.mShapeInteractions = reinterpret_cast<Sc::ShapeInteraction**>(dShapes);
    pre.mRestDistances = reinterpret_cast<PxReal*>(dRest);
    pre.mTorsionalFrictionData = reinterpret_cast<PxsTorsionalFrictionData*>(dTors);

    PxgSolverSharedDesc<IterativeSolveData> sh;
    std::memset(&sh, 0xCD, sizeof(sh));
    sh.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dBodySims);
    sh.articulationSlabMask = reinterpret_cast<uint4*>(dSlab);
    sh.deltaOutOffset = 0x7FFFFFFF; // assert-only in the .cu

    const PxgDevicePtr dPre = alloc(&pre, sizeof(pre));
    const PxgDevicePtr dSh = alloc(&sh, sizeof(sh));

    // engine launch: grid = ceil(totalBatches*32 / CONSTRAINT_PREPREP_BLOCK=128)
    const uint32_t nbBlocks = (totalBatches * 32 + 127) / 128;
    struct { uint64_t pre, sh; } prm = { dPre, dSh };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch constraintContactBlockPrePrepLaunch");
    shim.streamSynchronize();

    // ---- download outputs ----
    std::vector<PxgBlockConstraintBatch> gBB(totalBatches);
    std::vector<PxgBlockWorkUnit> gWU(nBlockData);
    std::vector<physx::PxgBlockContactData> gCD(nBlockData);
    std::vector<uint32_t> gCPw(contactPointsInit.size());
    std::vector<uint32_t> gCurrFric(nEdges), gBodyRef(refCap), gSlab(slabCap * 4);
    PxgPrePrepDesc preOut;
    shim.memcpyDtoH(gBB.data(), dBB, gBB.size() * sizeof(PxgBlockConstraintBatch));
    shim.memcpyDtoH(gWU.data(), dWU, gWU.size() * sizeof(PxgBlockWorkUnit));
    shim.memcpyDtoH(gCD.data(), dCD, gCD.size() * sizeof(physx::PxgBlockContactData));
    shim.memcpyDtoH(gCPw.data(), dCP, gCPw.size() * 4);
    shim.memcpyDtoH(gCurrFric.data(), dCurrFric, nEdges * 4);
    shim.memcpyDtoH(gBodyRef.data(), dBodyRef, refCap * 4);
    shim.memcpyDtoH(gSlab.data(), dSlab, slabCap * 16);
    shim.memcpyDtoH(&preOut, dPre, sizeof(preOut));

    size_t mm = 0;

    // ---- step 1: the atomic allocations partition the counter deltas ----
    struct Range { uint32_t start, count; };
    std::vector<Range> contactRanges, frictionRanges, artiRanges;
    uint32_t warpMax[9] = { 0 };
    // per-lane inputs recomputed on the host (shared with the reference below)
    struct LaneIn
    {
        uint32_t uniqueIndex = 0, edgeIndex = 0, cmIdx = 0, patchIndex = 0;
        uint32_t contactCount = 0;
        bool hasPatch = false;
        uint64_t patchAddr = 0, contactsAddr = 0;
        uint32_t forceIndex = 0xFFFFFFFFu, prevFric = 0;
    };
    std::vector<LaneIn> lanes(totalBatches * 32);
    for (uint32_t k = 0; k < totalBatches; ++k)
    {
        if (!isContactBatch[k]) continue;
        const uint32_t descStride = headers[k].mDescStride;
        for (uint32_t l = 0; l < descStride; ++l)
        {
            LaneIn& li = lanes[k * 32 + l];
            li.uniqueIndex = conTypeOf[k] == 1 ? contactUnique[headers[k].mStartPartitionIndex + l]
                                               : artiContactUnique[headers[k].mStartPartitionIndex + l];
            li.edgeIndex = constants[li.uniqueIndex].mEdgeIndex;
            const uint32_t npIndex = npOutputIndices[li.uniqueIndex];
            const uint32_t bucket = npIndex & 127;
            li.cmIdx = pre.mCmOutputOffsets[bucket] + (npIndex >> 7);
            li.patchIndex = indexData[li.uniqueIndex].mPatchIndex;
            const PxsContactManagerOutput& m = cmOut[li.cmIdx];
            li.prevFric = prevFric[li.edgeIndex];
            if (m.contactForces)
                li.forceIndex = uint32_t((uint64_t(m.contactForces) - cpuForcesBase) / 4);
            const uint32_t patchStart = uint32_t((uint64_t(m.contactPatches) - cpuPatchesBase) / 64);
            const uint32_t contactStart = uint32_t((uint64_t(m.contactPoints) - cpuContactsBase) / 16);
            if (m.nbContacts != 0 && li.patchIndex < m.nbPatches)
            {
                li.hasPatch = true;
                li.patchAddr = dPatches + uint64_t(patchStart + li.patchIndex) * 64;
                li.contactsAddr = dContacts + uint64_t(contactStart) * 16;
                li.contactCount = patches[patchStart + li.patchIndex].nbContacts;
            }
        }
        for (uint32_t l = 0; l < 32; ++l)
            warpMax[k] = std::max(warpMax[k], lanes[k * 32 + l].contactCount);
        contactRanges.push_back({ gBB[k].startConstraintIndex, warpMax[k] });
        frictionRanges.push_back({ gBB[k].startFrictionIndex, 4u });
        if (conTypeOf[k] == 3)
            artiRanges.push_back({ gBB[k].mArticulationResponseIndex, 4u + warpMax[k] });
        if (gBB[k].blockContactIndex != gBB[k].startConstraintIndex)
        { std::printf("MM batch %u blockContactIndex\n", k); ++mm; }
    }
    auto checkPartition = [&](const char* name, std::vector<Range> rs, uint32_t base, uint32_t end) {
        std::sort(rs.begin(), rs.end(), [](const Range& a, const Range& b) { return a.start < b.start; });
        uint32_t cur = base;
        for (const Range& r : rs)
        {
            if (r.start != cur) { std::printf("MM %s: range start %u != %u\n", name, r.start, cur); ++mm; }
            cur += r.count;
        }
        if (cur != end) { std::printf("MM %s: end %u != %u\n", name, cur, end); ++mm; }
    };
    checkPartition("contactAlloc", contactRanges, initContactIdx, preOut.sharedContactConstraintIndex);
    checkPartition("frictionAlloc", frictionRanges, initFrictionIdx, preOut.sharedFrictionConstraintIndex);
    checkPartition("artiRespAlloc", artiRanges, initArtiResponseIdx, preOut.sharedArticulationResponseIndex);

    // ---- step 2: full CPU reference AT the GPU-allocated bases ----
    std::vector<uint32_t> refBB = blockBatchesInit;
    std::vector<uint32_t> refWU = workUnitInit;
    std::vector<uint32_t> refCD = contactDataInit;
    std::vector<uint32_t> refCP = contactPointsInit;
    std::vector<uint32_t> refCurrFric = currFricInit;
    std::vector<uint32_t> refBodyRef = bodyRefInit;
    std::vector<uint32_t> refBodyRefCount(refCap, 0); // >1 => benign race (see comment)
    std::vector<uint32_t> refSlab = slabMaskInit;

    PxgBlockConstraintBatch* rBB = reinterpret_cast<PxgBlockConstraintBatch*>(refBB.data());
    PxgBlockWorkUnit* rWU = reinterpret_cast<PxgBlockWorkUnit*>(refWU.data());
    physx::PxgBlockContactData* rCD = reinterpret_cast<physx::PxgBlockContactData*>(refCD.data());
    physx::PxgBlockContactPoint* rCP = reinterpret_cast<physx::PxgBlockContactPoint*>(refCP.data());

    for (uint32_t k = 0; k < totalBatches; ++k)
    {
        if (!isContactBatch[k]) continue;
        const uint32_t descStride = headers[k].mDescStride;
        const uint32_t batchIndex = headers[k].mConstraintBatchIndex;
        PxgBlockConstraintBatch& b = rBB[k];
        b.mDescStride = headers[k].mDescStride;
        b.constraintType = headers[k].constraintType;
        b.mConstraintBatchIndex = batchIndex;
        b.mask = headers[k].mask;
        b.startConstraintIndex = gBB[k].startConstraintIndex; // GPU-allocated
        b.blockContactIndex = gBB[k].startConstraintIndex;
        b.startFrictionIndex = gBB[k].startFrictionIndex;
        if (conTypeOf[k] == 3)
            b.mArticulationResponseIndex = gBB[k].mArticulationResponseIndex;
        const uint32_t base = gBB[k].startConstraintIndex;

        for (uint32_t l = 0; l < 32; ++l)
        {
            rWU[batchIndex].mFrictionPatchIndex[l] = 0xFFFFFFFFu;
            const LaneIn& li = lanes[k * 32 + l];
            uint32_t laneFlags = 0;
            float laneRest = 0.f;
            if (l < descStride)
            {
                const PartitionNodeData& nd = nodeData[li.uniqueIndex];
                const uint32_t nodeA = nd.mNodeIndex0.index(), nodeB = nd.mNodeIndex1.index();
                const bool artiA = nd.mNodeIndex0.isArticulation() != 0;
                const bool artiB = nd.mNodeIndex1.isArticulation() != 0;
                const uint32_t sbA = nodeA == 0xFFFFFFFFu ? 0 : solverBodyIndices[nodeA];
                const uint32_t sbB = nodeB == 0xFFFFFFFFu ? 0 : solverBodyIndices[nodeB];
                b.bodyAIndex[l] = sbA;
                b.bodyBIndex[l] = sbB;
                b.bodyANodeIndex[l] = nd.mNodeIndex0;
                b.bodyBNodeIndex[l] = nd.mNodeIndex1;
                const uint32_t currentPartition = indexData[li.uniqueIndex].mPartitionIndex;
                const uint32_t slabId = currentPartition / maxConstraintPartitions;
                b.slabId[l] = slabId;
                if (k < dynamicBatchEnd)
                {
                    if (!artiA)
                        b.remappedBodyAIndex[l] = refRemapIndexRigidBody(
                            (nd.mNextIndex[0] & 1) != 0, partStart, partArtiStart, partJointCounts,
                            partArtiJointCounts, indexData[nd.mNextIndex[0] >> 1], refBodyRef, refBodyRefCount,
                            currentPartition, maxConstraintPartitions, sbA, activeBodyOffset, numBatches,
                            numArtiBatches, nbElemsPerBody, nbSlabs);
                    else
                    {
                        b.remappedBodyAIndex[l] = bodySims[nodeA].articulationRemapId;
                        const uint32_t partitionId = currentPartition & (maxConstraintPartitions - 1);
                        const uint32_t ind = sbA + (slabId + partitionId * nbSlabs) * nbArticulations;
                        refSlab[ind * 4 + 0] = nd.mNodeIndex0.articulationLinkId();
                        refSlab[ind * 4 + 1] = k * 32 * 2 * nbElemsPerBody + l;
                    }
                    if (!artiB)
                        b.remappedBodyBIndex[l] = refRemapIndexRigidBody(
                            (nd.mNextIndex[1] & 1) != 0, partStart, partArtiStart, partJointCounts,
                            partArtiJointCounts, indexData[nd.mNextIndex[1] >> 1], refBodyRef, refBodyRefCount,
                            currentPartition, maxConstraintPartitions, sbB, activeBodyOffset, numBatches,
                            numArtiBatches, nbElemsPerBody, nbSlabs);
                    else
                    {
                        b.remappedBodyBIndex[l] = bodySims[nodeB].articulationRemapId;
                        const uint32_t partitionId = currentPartition & (maxConstraintPartitions - 1);
                        const uint32_t ind = sbB + (slabId + partitionId * nbSlabs) * nbArticulations;
                        refSlab[ind * 4 + 2] = nd.mNodeIndex1.articulationLinkId();
                        refSlab[ind * 4 + 3] = (k * 32 * 2 + 32) * nbElemsPerBody + l;
                    }
                }
                else
                {
                    if (artiA) b.remappedBodyAIndex[l] = bodySims[nodeA].articulationRemapId;
                    if (artiB) b.remappedBodyBIndex[l] = bodySims[nodeB].articulationRemapId;
                }
                rWU[batchIndex].mEdgeIndex[l] = li.edgeIndex;
                rWU[batchIndex].mPatchIndex[l] = li.patchIndex;
                reinterpret_cast<uint64_t*>(rBB[k].shapeInteraction)[l] = shapeInteractions[li.cmIdx];
                laneRest = restDistances[li.cmIdx];
                rWU[batchIndex].mRestDistance[l] = laneRest;
                rWU[batchIndex].mTorsionalFrictionData[l].x = torsional[li.cmIdx].mTorsionalPatchRadius;
                rWU[batchIndex].mTorsionalFrictionData[l].y = torsional[li.cmIdx].mMinTorsionalRadius;
                laneFlags = cmOut[li.cmIdx].flags;
                rWU[batchIndex].mFlags[l] = laneFlags;
                refCurrFric[li.edgeIndex] = cmOut[li.cmIdx].nbPatches;
                if (li.hasPatch)
                    rWU[batchIndex].mFrictionPatchIndex[l] =
                        uint32_t((uint64_t(cmOut[li.cmIdx].contactPatches) - cpuPatchesBase) / 64) +
                        li.patchIndex;
            }

            // per-lane packing core (the cppPack transcription)
            uint32_t numContacts = 0, stride = 1, contactStartIndex = 0;
            uint64_t contactsAddr = li.contactsAddr;
            if (l < descStride)
            {
                float dynamicFriction = 0.f, staticFriction = 0.f;
                uint32_t solverFlags = 0;
                if (li.patchAddr != 0)
                {
                    const PxContactPatch& cp =
                        patches[size_t((li.patchAddr - dPatches) / sizeof(PxContactPatch))];
                    const uint32_t internalFlags = cp.internalFlags;
                    const bool perPointFriction =
                        (internalFlags & PxContactPatch::eHAS_TARGET_VELOCITY) != 0 ||
                        (laneFlags & (1u << 2)) != 0;
                    bool disableFriction = false, isAccelerationSpring = false;
                    if ((internalFlags & PxContactPatch::eFORCE_NO_RESPONSE) == 0)
                    {
                        const bool isModifiable =
                            (internalFlags & PxContactPatch::eCOMPRESSED_MODIFIED_CONTACT) != 0;
                        contactsAddr += uint64_t(cp.startContactIndex) * 16;
                        if (isModifiable)
                        {
                            contactsAddr += uint64_t(cp.startContactIndex) * 16;
                            stride = 2;
                        }
                        numContacts = cp.nbContacts;
                        contactStartIndex = cp.startContactIndex;
                        float m0 = cp.mMassModification.linear0, m1 = cp.mMassModification.angular0;
                        float m2 = cp.mMassModification.linear1, m3 = cp.mMassModification.angular1;
                        if (laneFlags & (1u << 14)) { m0 = 0.f; m1 = 0.f; }
                        if (laneFlags & (1u << 15)) { m2 = 0.f; m3 = 0.f; }
                        rCD[batchIndex].mInvMassScale[l].linear0 = m0;
                        rCD[batchIndex].mInvMassScale[l].angular0 = m1;
                        rCD[batchIndex].mInvMassScale[l].linear1 = m2;
                        rCD[batchIndex].mInvMassScale[l].angular1 = m3;
                        float* nr = reinterpret_cast<float*>(&rCD[batchIndex].normal_restitutionW[l]);
                        nr[0] = cp.normal.x; nr[1] = cp.normal.y; nr[2] = cp.normal.z; nr[3] = cp.restitution;
                        rCD[batchIndex].damping[l] = cp.damping;
                        dynamicFriction = cp.dynamicFriction;
                        staticFriction = cp.staticFriction;
                        disableFriction = (cp.materialFlags & 1) != 0;
                        isAccelerationSpring = (cp.materialFlags & 16) != 0;
                    }
                    if (perPointFriction) solverFlags |= 2;
                    if (laneFlags & (1u << 9)) solverFlags |= 1;
                    if (disableFriction) solverFlags |= 4;
                    if (isAccelerationSpring) solverFlags |= 8;
                }
                const uint32_t packed = (numContacts & 0xFF) | (solverFlags << 8) |
                                        ((li.prevFric & 0xFF) << 16);
                float* cd = reinterpret_cast<float*>(&rCD[batchIndex].contactData[l]);
                cd[0] = laneRest; cd[1] = staticFriction; cd[2] = dynamicFriction;
                std::memcpy(&cd[3], &packed, 4);
            }
            for (uint32_t j = 0; j < numContacts; ++j)
            {
                const size_t srcElem = size_t((contactsAddr - dContacts) / 16);
                const float* src = &contactStream[(srcElem + j * stride) * 4];
                float* dstP = reinterpret_cast<float*>(&rCP[base + j].point_separationW[l]);
                std::memcpy(dstP, src, 16);
                float* dstT = reinterpret_cast<float*>(&rCP[base + j].targetVel_maxImpulseW[l]);
                if (stride != 1)
                    std::memcpy(dstT, src + 4, 16);
                else
                {
                    dstT[0] = dstT[1] = dstT[2] = 0.f;
                    dstT[3] = 3.402823466e+38f;
                }
            }
            rWU[batchIndex].mWriteback[l] = li.forceIndex + contactStartIndex;
        }
    }

    // ---- step 3: bitwise compares ----
    auto cmpWords = [&](const char* name, const std::vector<uint32_t>& r, const uint32_t* g, size_t n) {
        size_t bad = 0, first = ~size_t(0);
        for (size_t x = 0; x < n; ++x)
            if (r[x] != g[x]) { if (!bad) first = x; ++bad; }
        if (bad)
        {
            std::printf("MM %s: %zu diffs, first at word %zu (R %08x G %08x)\n", name, bad, first,
                        r[first], g[first]);
            mm += bad;
        }
    };
    cmpWords("blockBatches", refBB, reinterpret_cast<const uint32_t*>(gBB.data()), refBB.size());
    cmpWords("workUnits", refWU, reinterpret_cast<const uint32_t*>(gWU.data()), refWU.size());
    cmpWords("contactData", refCD, reinterpret_cast<const uint32_t*>(gCD.data()), refCD.size());
    cmpWords("contactPoints", refCP, gCPw.data(), refCP.size());
    cmpWords("currFrictionPatchCount", refCurrFric, gCurrFric.data(), nEdges);
    // solverBodyReferences: computeRemapIndexRigidBody writes
    // solverBodyReferences[referenceId] = index as an intermediate side effect.
    // The returned index (validated in remappedBodyA/BIndex above) is
    // deterministic per lane, but this scratch store races when two lanes hit
    // the same referenceId -- an inherent race in the .cu that real data avoids
    // via the nextIndex linked-list invariant (one write per (body,slab)), which
    // this synthetic random-nextIndex scenario deliberately does not honour.
    // Compare only slots written exactly once (count 0 must stay sentinel).
    {
        size_t bad = 0;
        for (uint32_t x = 0; x < refCap; ++x)
            if (refBodyRefCount[x] <= 1 && refBodyRef[x] != gBodyRef[x]) ++bad;
        if (bad) { std::printf("MM solverBodyReferences (once-written): %zu diffs\n", bad); mm += bad; }
    }
    cmpWords("articulationSlabMask", refSlab, gSlab.data(), slabCap * 4);

    // untouched desc counters must be untouched
    if (preOut.sharedJointRowIndex != 0) { std::printf("MM sharedJointRowIndex clobbered\n"); ++mm; }

    PXG_CHECK_MSG(mm == 0, "constraintContactBlockPrePrepLaunch: %zu mismatches", mm);

    std::printf("[ info ] cpp engine kernel: %u batches (%u contact/%u arti/%u static), counters %u/%u/%u\n",
                totalBatches, 3u, numArtiBatches, numStaticBatches, preOut.sharedContactConstraintIndex,
                preOut.sharedFrictionConstraintIndex, preOut.sharedArticulationResponseIndex);

    for (PxgDevicePtr h : { dHeaders, dCU, dACU, dSBI, dND, dID, dConst, dNpIdx, dPS, dPAS, dPJC, dPAJC,
                            dCmOut, dRest, dTors, dShapes, dPatches, dContacts, dBodySims, dPrevFric, dBB,
                            dWU, dCD, dCP, dCurrFric, dBodyRef, dSlab, dPre, dSh })
        shim.memFree(h);
}

// ---------------------------------------------------------------------------
// clearFrictionPatchCounts (ENGINE ABI, P5 Step 5)
// ---------------------------------------------------------------------------
// One thread per destroyed edge: frictionPatchCount[edgeIndicesToClear[i]] = 0.
// Launched by gpuMemDMAUpContactData when nbDestroyedEdges > 0 with the 3-arg
// CUDA pack (PxU32* frictionPatchCount, PxU32* edgeIndicesToClear, PxU32
// nbToClear) -> 24-byte blob; grid = ceil(nbToClear / 256).  Bit-exact
// compare; untouched slots keep their sentinel (over-clear check).
PXG_TEST(npContactPrePrep_clearFrictionPatchCounts)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(52100);
    const uint32_t nbEdges = 700;
    const uint32_t nbToClear = 301; // partial block: 301 threads of ceil = 2 * 256

    // shuffled distinct indices so no edge is cleared twice
    std::vector<uint32_t> indexPool(nbEdges);
    for (uint32_t i = 0; i < nbEdges; ++i) indexPool[i] = i;
    std::shuffle(indexPool.begin(), indexPool.end(), rng);
    std::vector<uint32_t> toClear(indexPool.begin(), indexPool.begin() + nbToClear);

    std::vector<uint32_t> countsInit(nbEdges);
    for (auto& c : countsInit) c = 1 + rng() % 7; // non-zero sentinels

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npContactPrePrep.spv", "clearFrictionPatchCounts", 24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &cppShimLoader);

    PxgDevicePtr dCounts = 0, dIdx = 0;
    shim.memAlloc(&dCounts, countsInit.size() * 4);
    shim.memAlloc(&dIdx, toClear.size() * 4);
    if (!dCounts || !dIdx)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dCounts, countsInit.data(), countsInit.size() * 4);
    shim.memcpyHtoD(dIdx, toClear.data(), toClear.size() * 4);

    struct { uint64_t counts, idx; uint32_t nbToClear, pad; } prm = { dCounts, dIdx, nbToClear, 0 };
    static_assert(sizeof(prm) == 24, "clearFrictionPatchCounts blob is 24 bytes");
    const uint32_t nbBlocks = (nbToClear + 255) / 256; // CLEAR_FRICTION_PATCH_COUNTS = 256
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch clearFrictionPatchCounts");
    shim.streamSynchronize();

    std::vector<uint32_t> refCounts = countsInit;
    for (uint32_t i = 0; i < nbToClear; ++i)
        refCounts[toClear[i]] = 0;

    std::vector<uint32_t> g(nbEdges);
    shim.memcpyDtoH(g.data(), dCounts, g.size() * 4);
    size_t mm = 0;
    for (uint32_t i = 0; i < nbEdges; ++i)
        if (g[i] != refCounts[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "clearFrictionPatchCounts: %zu mismatches", mm);

    for (PxgDevicePtr h : { dCounts, dIdx })
        shim.memFree(h);
}

// ---------------------------------------------------------------------------
// constraint1DBlockPrePrepLaunch (ENGINE ABI, P5 Step 5)
// ---------------------------------------------------------------------------
// The JOINT half of the block pre-prep, launched with the TOTAL batch count
// (contacts included).  Rigid-only scenario: 3 dynamic 1D batches interleaved
// with 2 contact batches (which the kernel must SKIP — their block rows stay
// sentinel) plus 1 static-region 1D batch (a >= totalDynamicBatches: header +
// prep rows written, remap indices untouched for rigid bodies).  Row-range
// allocation from the shared atomic is order-independent: the per-batch
// (start, warpMaxRows) ranges must partition [init, final).  Everything else
// is compared bitwise against the transcription (articulation branches are
// exercised by real jointed-articulation scenes, not this fixture).
PXG_TEST(npContactPrePrep_jointPrePrep)
{
    using namespace physx;

    std::mt19937 rng(52200);
    std::uniform_real_distribution<float> sf(-1.f, 1.f);

    // k: 0=1D 1=contact 2=1D 3=contact 4=1D | 5=static 1D
    const uint32_t numBatches = 5, numArtiBatches = 0, numStaticBatches = 1;
    const uint32_t totalBatches = numBatches + numArtiBatches + numStaticBatches; // 6
    const uint32_t dynamicBatchEnd = numBatches + numArtiBatches;                 // 5
    const bool is1DBatch[6] = { true, false, true, false, true, true };

    const uint32_t numNodes = 96;
    const uint32_t totalActiveBodies = 24, activeBodyOffset = 6;
    const uint32_t maxConstraintPartitions = 8, nbSlabs = 2, nbElemsPerBody = 2;
    const uint32_t numPartitions = 64;
    const uint32_t nUnique = totalBatches * 32 * 2; // rigid slots then arti slots
    const uint32_t nBlockData = 8;                  // blockPrepData elements
    const uint32_t MAXROWS = 20;                    // Dy::MAX_CONSTRAINT_ROWS
    const uint32_t maxRowsPerJoint = 6;
    const uint32_t nCData = 256;
    const uint32_t refCap = 64;
    const uint32_t initJointRow = 5;

    std::vector<PxgConstraintBatchHeader> headers(totalBatches);
    std::memset(headers.data(), 0, headers.size() * sizeof(PxgConstraintBatchHeader));
    uint32_t cbiCursor = 0;
    for (uint32_t k = 0; k < totalBatches; ++k)
    {
        headers[k].mDescStride = PxU16(is1DBatch[k] ? (1 + rng() % 32) : (4 + rng() % 29));
        headers[k].constraintType = PxU16(is1DBatch[k] ? 0 : 1); // eCONSTRAINT_1D / eCONTACT
        headers[k].mConstraintBatchIndex = is1DBatch[k] ? cbiCursor++ : (rng() % nBlockData);
        headers[k].mStartPartitionIndex = k * 32;
        headers[k].mask = rng();
    }

    std::vector<uint32_t> constraintUnique((totalBatches + 1) * 32);
    for (uint32_t i = 0; i < constraintUnique.size(); ++i)
        constraintUnique[i] = i % (totalBatches * 32);

    std::vector<uint32_t> solverBodyIndices(numNodes);
    for (uint32_t n = 0; n < numNodes; ++n)
        solverBodyIndices[n] = rng() % totalActiveBodies;

    std::vector<PartitionNodeData> nodeData(nUnique);
    std::memset(nodeData.data(), 0, nodeData.size() * sizeof(PartitionNodeData));
    std::vector<PartitionIndexData> indexData(nUnique);
    std::memset(indexData.data(), 0, indexData.size() * sizeof(PartitionIndexData));
    std::vector<uint32_t> npOutputIndices(nUnique);
    for (uint32_t q = 0; q < nUnique; ++q)
    {
        const bool staticA = (rng() % 6 == 0);
        const bool staticB = (rng() % 4 == 0);
        nodeData[q].mNodeIndex0 = staticA ? PxNodeIndex() : PxNodeIndex(rng() % numNodes);
        nodeData[q].mNodeIndex1 = staticB ? PxNodeIndex() : PxNodeIndex(rng() % numNodes);
        nodeData[q].mNextIndex[0] = ((rng() % nUnique) << 1) | (rng() & 1);
        nodeData[q].mNextIndex[1] = ((rng() % nUnique) << 1) | (rng() & 1);

        indexData[q].mPartitionIndex = PxU16(rng() % numPartitions);
        indexData[q].mPatchIndex = PxU8(rng() % 3);
        indexData[q].mCType = PxU8(rng() % 4);
        indexData[q].mPartitionEntryIndex = rng() % 4096;

        npOutputIndices[q] = q % nCData; // joints: direct PxgConstraintData index
    }

    std::vector<uint32_t> partStart(numPartitions), partArtiStart(numPartitions),
        partJointCounts(numPartitions), partArtiJointCounts(numPartitions);
    for (uint32_t pi = 0; pi < numPartitions; ++pi)
    {
        partStart[pi] = rng() % 100;
        partArtiStart[pi] = rng() % 100;
        partJointCounts[pi] = rng() % 10;
        partArtiJointCounts[pi] = rng() % 10;
    }

    // joint prep inputs: PxgConstraintData + Px1DConstraint row pools
    std::vector<PxgConstraintData> cdata(nCData);
    std::memset(cdata.data(), 0, cdata.size() * sizeof(PxgConstraintData));
    const uint32_t nRows = nCData * maxRowsPerJoint;
    std::vector<Px1DConstraint> rows(nRows);
    std::memset(rows.data(), 0, rows.size() * sizeof(Px1DConstraint));
    for (uint32_t c = 0; c < nCData; ++c)
    {
        PxgConstraintData& d = cdata[c];
        d.mInvMassScale.linear0 = sf(rng); d.mInvMassScale.angular0 = sf(rng);
        d.mInvMassScale.linear1 = sf(rng); d.mInvMassScale.angular1 = sf(rng);
        d.mRaWorld_linBreakForceW = make_float4(sf(rng), sf(rng), sf(rng), sf(rng));
        d.mRbWorld_angBreakForceW = make_float4(sf(rng), sf(rng), sf(rng), sf(rng));
        const uint32_t nr = rng() % (maxRowsPerJoint + 1); // 0..6 rows
        d.mNumRows_Flags_StartIndex.x = nr;
        d.mNumRows_Flags_StartIndex.y = rng();
        d.mNumRows_Flags_StartIndex.z = c * maxRowsPerJoint;
        d.mNumRows_Flags_StartIndex.w = 0;
    }
    for (uint32_t r = 0; r < nRows; ++r)
    {
        Px1DConstraint& c1 = rows[r];
        c1.linear0 = PxVec3(sf(rng), sf(rng), sf(rng)); c1.geometricError = sf(rng);
        c1.angular0 = PxVec3(sf(rng), sf(rng), sf(rng)); c1.velocityTarget = sf(rng);
        c1.linear1 = PxVec3(sf(rng), sf(rng), sf(rng)); c1.minImpulse = -1.f - sf(rng);
        c1.angular1 = PxVec3(sf(rng), sf(rng), sf(rng)); c1.maxImpulse = 1.f + sf(rng);
        c1.mods.bounce.restitution = sf(rng);
        c1.mods.bounce.velocityThreshold = sf(rng);
        c1.flags = PxU16(rng() & 0xFFFF);
        c1.solveHint = PxU16(rng() & 0xFFFF);
        c1.pad = 0;
    }

    // ---- sentinel-filled outputs ----
    const uint32_t SENT = 0xABABABABu;
    std::vector<uint32_t> blockBatchesInit(totalBatches * sizeof(PxgBlockConstraintBatch) / 4, SENT);
    std::vector<uint32_t> prepDataInit(nBlockData * sizeof(PxgBlockConstraint1DData) / 4, SENT);
    std::vector<uint32_t> prepVelInit(nBlockData * MAXROWS * sizeof(PxgBlockConstraint1DVelocities) / 4, SENT);
    std::vector<uint32_t> prepParInit(nBlockData * MAXROWS * sizeof(PxgBlockConstraint1DParameters) / 4, SENT);
    std::vector<uint32_t> bodyRefInit(refCap, SENT);

    std::vector<PxgBodySim> bodySims(numNodes);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(PxgBodySim));

    // ---- device buffers ----
    PxgSlangComputeDevice* dev = testDevice();
    const physx::gpuslang::PxgKernelDef manifest[] = {
        { "npContactPrePrep.spv", "constraint1DBlockPrePrepLaunch", 16u, 0 },
    };
    physx::gpuslang::PxgCudaShim shim(dev, manifest, 1, &cppShimLoader);
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data)
            shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    const PxgDevicePtr dHeaders = alloc(headers.data(), headers.size() * sizeof(PxgConstraintBatchHeader));
    const PxgDevicePtr dCU = alloc(constraintUnique.data(), constraintUnique.size() * 4);
    const PxgDevicePtr dSBI = alloc(solverBodyIndices.data(), solverBodyIndices.size() * 4);
    const PxgDevicePtr dND = alloc(nodeData.data(), nodeData.size() * sizeof(PartitionNodeData));
    const PxgDevicePtr dID = alloc(indexData.data(), indexData.size() * sizeof(PartitionIndexData));
    const PxgDevicePtr dNpIdx = alloc(npOutputIndices.data(), npOutputIndices.size() * 4);
    const PxgDevicePtr dPS = alloc(partStart.data(), partStart.size() * 4);
    const PxgDevicePtr dPAS = alloc(partArtiStart.data(), partArtiStart.size() * 4);
    const PxgDevicePtr dPJC = alloc(partJointCounts.data(), partJointCounts.size() * 4);
    const PxgDevicePtr dPAJC = alloc(partArtiJointCounts.data(), partArtiJointCounts.size() * 4);
    const PxgDevicePtr dCData = alloc(cdata.data(), cdata.size() * sizeof(PxgConstraintData));
    const PxgDevicePtr dRows = alloc(rows.data(), rows.size() * sizeof(Px1DConstraint));
    const PxgDevicePtr dBodySims = alloc(bodySims.data(), bodySims.size() * sizeof(PxgBodySim));
    const PxgDevicePtr dBB = alloc(blockBatchesInit.data(), blockBatchesInit.size() * 4);
    const PxgDevicePtr dPD = alloc(prepDataInit.data(), prepDataInit.size() * 4);
    const PxgDevicePtr dPV = alloc(prepVelInit.data(), prepVelInit.size() * 4);
    const PxgDevicePtr dPP = alloc(prepParInit.data(), prepParInit.size() * 4);
    const PxgDevicePtr dBodyRef = alloc(bodyRefInit.data(), bodyRefInit.size() * 4);
    if (!dHeaders) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgPrePrepDesc pre;
    std::memset(&pre, 0xCD, sizeof(pre));
    pre.blockBatches = reinterpret_cast<PxgBlockConstraintBatch*>(dBB);
    pre.numBatches = numBatches;
    pre.numStaticBatches = numStaticBatches;
    pre.numArtiBatches = 0;
    pre.numArtiStaticBatches = 0;
    pre.numArtiSelfBatches = 0;
    pre.blockPrepData = reinterpret_cast<PxgBlockConstraint1DData*>(dPD);
    pre.blockPrepVelocityData = reinterpret_cast<PxgBlockConstraint1DVelocities*>(dPV);
    pre.blockPrepParameterData = reinterpret_cast<PxgBlockConstraint1DParameters*>(dPP);
    pre.constraintData = reinterpret_cast<PxgConstraintData*>(dCData);
    pre.constraintRows = reinterpret_cast<Px1DConstraint*>(dRows);
    pre.artiConstraintData = NULL;
    pre.artiConstraintRows = NULL;
    pre.sharedJointRowIndex = initJointRow;
    pre.sharedArticulationResponseIndex = 0;
    pre.solverBodyIndices = reinterpret_cast<PxU32*>(dSBI);
    pre.mPartitionIndices = reinterpret_cast<PartitionIndexData*>(dID);
    pre.mPartitionstartBatchIndices = reinterpret_cast<PxU32*>(dPS);
    pre.mPartitionArtiStartBatchIndices = reinterpret_cast<PxU32*>(dPAS);
    pre.mPartitionJointCounts = reinterpret_cast<PxU32*>(dPJC);
    pre.mPartitionArtiJointCounts = reinterpret_cast<PxU32*>(dPAJC);
    pre.mNpOutputIndices = reinterpret_cast<PxU32*>(dNpIdx);
    pre.mPartitionNodeData = reinterpret_cast<PartitionNodeData*>(dND);
    pre.mBatchHeaders = reinterpret_cast<PxgConstraintBatchHeader*>(dHeaders);
    pre.mConstraintUniqueIndices = reinterpret_cast<PxU32*>(dCU);
    pre.mArtiConstraintUniqueIndices = NULL;
    pre.mSolverBodyReferences = reinterpret_cast<PxgSolverReferences*>(dBodyRef);
    pre.mMaxConstraintPartitions = maxConstraintPartitions;
    pre.mTotalSlabs = nbSlabs;
    pre.mTotalActiveBodies = totalActiveBodies;
    pre.mTotalActiveArticulations = 0;
    pre.mActiveBodyStartOffset = activeBodyOffset;
    pre.nbElementsPerBody = nbElemsPerBody;

    PxgSolverSharedDesc<IterativeSolveData> sh;
    std::memset(&sh, 0xCD, sizeof(sh));
    sh.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dBodySims);
    sh.articulationSlabMask = NULL; // rigid-only fixture: the arti branch is never taken
    sh.deltaOutOffset = 0x7FFFFFFF;

    const PxgDevicePtr dPre = alloc(&pre, sizeof(pre));
    const PxgDevicePtr dSh = alloc(&sh, sizeof(sh));

    // engine launch: grid = ceil(totalBatches*32 / CONSTRAINT_PREPREP_BLOCK=128)
    const uint32_t nbBlocks = (totalBatches * 32 + 127) / 128;
    struct { uint64_t pre, sh; } prm = { dPre, dSh };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch constraint1DBlockPrePrepLaunch");
    shim.streamSynchronize();

    // ---- download ----
    std::vector<PxgBlockConstraintBatch> gBB(totalBatches);
    std::vector<uint32_t> gPD(prepDataInit.size()), gPV(prepVelInit.size()), gPP(prepParInit.size());
    std::vector<uint32_t> gBodyRef(refCap);
    PxgPrePrepDesc preOut;
    shim.memcpyDtoH(gBB.data(), dBB, gBB.size() * sizeof(PxgBlockConstraintBatch));
    shim.memcpyDtoH(gPD.data(), dPD, gPD.size() * 4);
    shim.memcpyDtoH(gPV.data(), dPV, gPV.size() * 4);
    shim.memcpyDtoH(gPP.data(), dPP, gPP.size() * 4);
    shim.memcpyDtoH(gBodyRef.data(), dBodyRef, refCap * 4);
    shim.memcpyDtoH(&preOut, dPre, sizeof(preOut));

    size_t mm = 0;

    // ---- step 1: the atomic allocations partition [initJointRow, final) ----
    struct Range { uint32_t start, count; };
    std::vector<Range> jointRanges;
    uint32_t warpMax[6] = { 0 };
    for (uint32_t k = 0; k < totalBatches; ++k)
    {
        if (!is1DBatch[k]) continue;
        for (uint32_t l = 0; l < headers[k].mDescStride; ++l)
        {
            const uint32_t uq = constraintUnique[headers[k].mStartPartitionIndex + l];
            warpMax[k] = std::max(warpMax[k], cdata[npOutputIndices[uq]].mNumRows_Flags_StartIndex.x);
        }
        jointRanges.push_back({ gBB[k].startConstraintIndex, warpMax[k] });
    }
    {
        std::sort(jointRanges.begin(), jointRanges.end(),
                  [](const Range& a, const Range& b) { return a.start < b.start; });
        uint32_t cur = initJointRow;
        for (const Range& r : jointRanges)
        {
            if (r.start != cur) { std::printf("MM jointAlloc: range start %u != %u\n", r.start, cur); ++mm; }
            cur += r.count;
        }
        if (cur != preOut.sharedJointRowIndex)
        { std::printf("MM jointAlloc: end %u != %u\n", cur, preOut.sharedJointRowIndex); ++mm; }
    }

    // ---- step 2: full CPU reference AT the GPU-allocated bases ----
    std::vector<uint32_t> refBB = blockBatchesInit;
    std::vector<uint32_t> refPD = prepDataInit;
    std::vector<uint32_t> refPV = prepVelInit;
    std::vector<uint32_t> refPP = prepParInit;
    std::vector<uint32_t> refBodyRef = bodyRefInit;
    std::vector<uint32_t> refBodyRefCount(refCap, 0);

    PxgBlockConstraintBatch* rBB = reinterpret_cast<PxgBlockConstraintBatch*>(refBB.data());
    PxgBlockConstraint1DData* rPD = reinterpret_cast<PxgBlockConstraint1DData*>(refPD.data());
    PxgBlockConstraint1DVelocities* rPV = reinterpret_cast<PxgBlockConstraint1DVelocities*>(refPV.data());
    PxgBlockConstraint1DParameters* rPP = reinterpret_cast<PxgBlockConstraint1DParameters*>(refPP.data());

    for (uint32_t k = 0; k < totalBatches; ++k)
    {
        if (!is1DBatch[k]) continue;
        const uint32_t descStride = headers[k].mDescStride;
        const uint32_t batchIndex = headers[k].mConstraintBatchIndex;
        PxgBlockConstraintBatch& b = rBB[k];
        b.mDescStride = headers[k].mDescStride;
        b.constraintType = headers[k].constraintType;
        b.mConstraintBatchIndex = batchIndex;
        b.mStartPartitionIndex = headers[k].mStartPartitionIndex;
        b.mask = headers[k].mask;
        b.startConstraintIndex = gBB[k].startConstraintIndex; // GPU-allocated

        for (uint32_t l = 0; l < descStride; ++l)
        {
            const uint32_t uq = constraintUnique[headers[k].mStartPartitionIndex + l];
            const PartitionNodeData& nd = nodeData[uq];
            const uint32_t nodeA = nd.mNodeIndex0.index(), nodeB = nd.mNodeIndex1.index();
            const uint32_t sbA = nodeA == 0xFFFFFFFFu ? 0 : solverBodyIndices[nodeA];
            const uint32_t sbB = nodeB == 0xFFFFFFFFu ? 0 : solverBodyIndices[nodeB];
            b.bodyAIndex[l] = sbA;
            b.bodyBIndex[l] = sbB;
            b.bodyANodeIndex[l] = nd.mNodeIndex0;
            b.bodyBNodeIndex[l] = nd.mNodeIndex1;
            const uint32_t currentPartition = indexData[uq].mPartitionIndex;
            const uint32_t slabId = currentPartition / maxConstraintPartitions;
            b.slabId[l] = slabId;
            if (k < dynamicBatchEnd)
            {
                b.remappedBodyAIndex[l] = refRemapIndexRigidBody(
                    (nd.mNextIndex[0] & 1) != 0, partStart, partArtiStart, partJointCounts,
                    partArtiJointCounts, indexData[nd.mNextIndex[0] >> 1], refBodyRef, refBodyRefCount,
                    currentPartition, maxConstraintPartitions, sbA, activeBodyOffset, numBatches,
                    numArtiBatches, nbElemsPerBody, nbSlabs);
                b.remappedBodyBIndex[l] = refRemapIndexRigidBody(
                    (nd.mNextIndex[1] & 1) != 0, partStart, partArtiStart, partJointCounts,
                    partArtiJointCounts, indexData[nd.mNextIndex[1] >> 1], refBodyRef, refBodyRefCount,
                    currentPartition, maxConstraintPartitions, sbB, activeBodyOffset, numBatches,
                    numArtiBatches, nbElemsPerBody, nbSlabs);
            }
            // else: static-region rigid 1D batch — no remap writes at all

            const uint32_t index = npOutputIndices[uq];
            const PxgConstraintData& d = cdata[index];
            const uint32_t nr = d.mNumRows_Flags_StartIndex.x;
            const uint32_t rowStart = d.mNumRows_Flags_StartIndex.z;
            const uint32_t outBase = batchIndex * MAXROWS;
            rPD[batchIndex].mInvMassScale[l] = d.mInvMassScale;
            rPD[batchIndex].mRAWorld_linBreakForce[l] = d.mRaWorld_linBreakForceW;
            rPD[batchIndex].mRBWorld_AngBreakForce[l] = d.mRbWorld_angBreakForceW;
            rPD[batchIndex].mFlags[l] = d.mNumRows_Flags_StartIndex.y;
            rPD[batchIndex].mNumRows[l] = nr;
            for (uint32_t i = 0; i < nr; ++i)
            {
                const Px1DConstraint& c1 = rows[rowStart + i];
                std::memcpy(&rPV[outBase + i].linear0XYZ_geometricErrorW[l], &c1.linear0, 16);
                std::memcpy(&rPV[outBase + i].angular0XYZ_velocityTargetW[l], &c1.angular0, 16);
                std::memcpy(&rPV[outBase + i].linear1XYZ_minImpulseW[l], &c1.linear1, 16);
                std::memcpy(&rPV[outBase + i].angular1XYZ_maxImpulseW[l], &c1.angular1, 16);
                rPP[outBase + i].flags[l] = c1.flags;
                rPP[outBase + i].mods.bounce.restitution[l] = c1.mods.bounce.restitution;
                rPP[outBase + i].mods.bounce.velocityThreshold[l] = c1.mods.bounce.velocityThreshold;
                rPP[outBase + i].solveHint[l] = c1.solveHint;
            }
        }
    }

    // ---- step 3: bitwise compares (contact batches / untouched lanes must
    // still hold the sentinel) ----
    auto cmpWords = [&](const char* name, const std::vector<uint32_t>& r, const uint32_t* g, size_t n) {
        size_t bad = 0, first = ~size_t(0);
        for (size_t x = 0; x < n; ++x)
            if (r[x] != g[x]) { if (!bad) first = x; ++bad; }
        if (bad)
        {
            std::printf("MM %s: %zu diffs, first at word %zu (R %08x G %08x)\n", name, bad, first,
                        r[first], g[first]);
            mm += bad;
        }
    };
    cmpWords("blockBatches", refBB, reinterpret_cast<const uint32_t*>(gBB.data()), refBB.size());
    cmpWords("blockPrepData", refPD, gPD.data(), refPD.size());
    cmpWords("blockPrepVelocityData", refPV, gPV.data(), refPV.size());
    cmpWords("blockPrepParameterData", refPP, gPP.data(), refPP.size());
    // solverBodyReferences: same benign scratch race as the contact twin test —
    // compare only slots written at most once
    {
        size_t bad = 0;
        for (uint32_t x = 0; x < refCap; ++x)
            if (refBodyRefCount[x] <= 1 && refBodyRef[x] != gBodyRef[x]) ++bad;
        if (bad) { std::printf("MM solverBodyReferences (once-written): %zu diffs\n", bad); mm += bad; }
    }

    // untouched desc counters must be untouched
    if (preOut.sharedArticulationResponseIndex != 0)
    { std::printf("MM sharedArticulationResponseIndex clobbered\n"); ++mm; }

    PXG_CHECK_MSG(mm == 0, "constraint1DBlockPrePrepLaunch: %zu mismatches", mm);

    std::printf("[ info ] joint pre-prep: %u batches (4x 1D incl. 1 static, 2 contact skipped), "
                "rows %u..%u\n", totalBatches, initJointRow, preOut.sharedJointRowIndex);

    for (PxgDevicePtr h : { dHeaders, dCU, dSBI, dND, dID, dNpIdx, dPS, dPAS, dPJC, dPAJC, dCData,
                            dRows, dBodySims, dBB, dPD, dPV, dPP, dBodyRef, dPre, dSh })
        shim.memFree(h);
}

} // namespace
