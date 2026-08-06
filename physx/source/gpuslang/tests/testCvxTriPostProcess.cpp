// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// Verification of cvxTriPostProcessBda — `convexTrimeshPostProcess`
// (convexMeshPostProcess.cu:369, core at :182) on the engine BDA ABI.
//
// The kernel produces exactly one bit of output per second-pass pair: whether to
// clear the contact count on that pair's ConvexTriNormalAndIndex.  So the fixture
// is a table of cases, each one thread, each pinning one decision, and the oracle
// is the expected clear/no-clear per case derived by hand from the CUDA and
// written next to it.
//
// The synthetic mesh (6 triangles, 6 vertices) is NOT a manifold — T2 and T3 are
// declared adjacent while sharing only one vertex, and T4 is an isolated triangle
// with three boundary edges.  That is deliberate: the kernel never validates
// adjacency, and the cases need adjacency/vertex-reference combinations a
// well-formed mesh would not offer.
//
//   ind:  T0 (0,1,2)  T1 (1,3,2)  T2 (3,4,2)  T3 (0,2,5)  T4 (4,5,1)  T5 (4,0,3)
//   adj:  T0 (T1, --, T3)   T1 (--, T2, T0)   T2 (T3, --, T1)
//         T3 (T0, T2, --)   T4 (--, --, --)   T5 (--, T1, --)  (-- = 0xFFFFFFFF)
//
//   vertex -> triangles referencing it (accumulatedTriangleRefs/triangleReferences):
//     v0 [T0,T3,T5]   v1 [T0,T1,T4]   v2 [T0,T1,T2,T3]
//     v3 [T1,T2,T5]   v4 [T4,T5,T2]   v5 [T3,T4]
//
// The ORDER WITHIN a vertex's list is arbitrary as far as the kernel is concerned,
// and v4's is chosen deliberately: its only matching triangle, T2, is LAST, so a
// loop that stopped one entry early would miss it.  T5 exists for one reason --
// it is the only triangle whose adjacency is boundary on .x and .z but NOT on .y,
// which is what separates eV0's (.x, .z) pair from an (.x, .y) misread.
//
// CM0's ordered triangle list holds T1 as a FACE contact and T2 as an EDGE
// contact, and nothing else.  That split is what makes the two searches
// separable: the edge branch looks ONLY for face contacts, so a neighbour that
// produced an edge contact does not count, while the vertex branch accepts
// either.  Per vertex the verdict is then:
//     v0 no   v1 YES (T1 face)   v2 YES (T1 face)   v3 YES (T1 face)
//     v4 YES (T2 EDGE only)      v5 no
//
// THE SECOND-PASS ARRAY IS PADDED WITH 64 NO-OPS AND THE REAL CASES START AT
// INDEX 64, so a kernel that dropped the `gid.x * 64` term from its thread index
// would run the no-ops twice and never reach a single real case.  The pair
// indices are 100 + case, never the thread index, so indexing the stack by the
// thread would miss too.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct PpPush
{
    uint64_t cvxTrimeshPair, cvxTriNIPtr, cvxTriIntermPtr, orderedCvxTriIntermPtr;
    uint64_t cvxTriSecondPassPairsPtr, stackPtr, nbSecondPassPairs, shapes;
};

const uint32_t BND = 0xFFFFFFFFu;

const uint32_t EV0 = 1u << 29, EV1 = 2u << 29, EV2 = 3u << 29;
const uint32_t EE01 = 4u << 29, EE02 = 5u << 29, EE12 = 6u << 29;

const uint32_t FACE = 0x80000000u, EDGE = 0xC0000000u;

// ConvexMeshPair fields, as running FIELD offsets (not the declaration's trailing
// end offsets).  Pinned against the engine header in testNpAbiLayout.cpp.
struct MeshPairFix { uint32_t startIndex, count, cmIndex, roundedStartIndex; };

// CM0: 2 ordered entries at 12; its NI slots start at 7.  CM1: 1 entry at 24, NI
// slots at 40.  startIndex and roundedStartIndex differ, and both differ from
// cmIndex and from the pad, so any of the four offsets read wrongly is visible.
const MeshPairFix CM0 = { 7u, 2u, 55u, 12u };
const MeshPairFix CM1 = { 40u, 1u, 66u, 24u };
// CM2 has count == 0.  The core asserts that cannot happen, but the assert is
// debug-only and `findTriangle` carries an `index < count` guard for the release
// path -- without it, binarySearch's `left ? left - 1 : 0` hands back index 0 and
// the word at the slice start is compared anyway.  CM2's slice starts with
// exactly the word an eE01-on-T0 search looks for, so an unguarded read matches.
const MeshPairFix CM2 = { 50u, 0u, 88u, 28u };

struct Case
{
    const char* name;
    uint32_t cm;          // 0, 1 or 2
    uint32_t triangleIdx;
    uint32_t mask;
    uint32_t shapeID;     // 0 = identity scale, 1 = negative determinant
    uint32_t testOffset;
    uint32_t nbContacts;  // 0 drives the early-out; 1 vs 3 pins the shift width
    bool expectClear;
    const char* why;
};

PXG_TEST(cvxTriPostProcess)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the cooked mesh blob ------------------------------------------------
    const uint32_t numVerts = 6, numTris = 6, nbPackedNodes = 2;
    const uint32_t BV32_STRIDE = 1168;  // sizeof(Gu::BV32DataPacked); pinned in testNpAbiLayout

    const uint32_t ind[6][4] = {
        { 0, 1, 2, 0 }, { 1, 3, 2, 0 }, { 3, 4, 2, 0 },
        { 0, 2, 5, 0 }, { 4, 5, 1, 0 }, { 4, 0, 3, 0 },
    };
    const uint32_t adj[6][4] = {
        { 1, BND, 3, 0 }, { BND, 2, 0, 0 }, { 3, BND, 1, 0 },
        { 0, 2, BND, 0 }, { BND, BND, BND, 0 }, { BND, 1, BND, 0 },
    };
    const uint32_t accRefs[7] = { 0, 3, 6, 10, 13, 16, 18 };
    const uint32_t refs[18] = { 0,3,5, 0,1,4, 0,1,2,3, 1,2,5, 4,5,2, 3,4 };

    std::vector<uint32_t> mesh;
    auto u32 = [&](uint32_t v) { mesh.push_back(v); };

    u32(numVerts); u32(numTris); u32(0xABCD1234u); u32(nbPackedNodes);
    // The tree is filled with a recognisable pattern, not zeros: a reader that
    // mis-sizes it lands IN the pattern and reports garbage rather than zeros
    // that could pass for a boundary marker.
    for (uint32_t i = 0; i < nbPackedNodes * BV32_STRIDE / 4; ++i) u32(0xDEAD0000u | (i & 0xffffu));
    for (uint32_t i = 0; i < numVerts * 4; ++i) u32(0x3F800000u | (i << 16));   // float4 verts, skipped
    for (uint32_t t = 0; t < numTris; ++t) for (int k = 0; k < 4; ++k) u32(ind[t][k]);
    for (uint32_t t = 0; t < numTris; ++t) for (int k = 0; k < 4; ++k) u32(adj[t][k]);
    for (uint32_t t = 0; t < numTris; ++t) u32(0x11220000u | t);                // faceRemap, skipped
    for (uint32_t i = 0; i < 7; ++i) u32(accRefs[i]);
    for (uint32_t i = 0; i < 18; ++i) u32(refs[i]);

    // ---- the ordered per-CM triangle lists -----------------------------------
    // Sorted ascending unsigned, then padded to a multiple of four with the
    // radix sort's 0xFFFFFFFF sentinel.  The search is bounded by `count`, so a
    // kernel that used the rounded length instead would read into the sentinels.
    std::vector<uint32_t> ordered(32, 0x5A5A5A5Au);
    ordered[CM0.roundedStartIndex + 0] = 1u | FACE;   // T1, face contact
    ordered[CM0.roundedStartIndex + 1] = 2u | EDGE;   // T2, edge contact
    ordered[CM0.roundedStartIndex + 2] = 0xFFFFFFFFu;
    ordered[CM0.roundedStartIndex + 3] = 0xFFFFFFFFu;
    ordered[CM1.roundedStartIndex + 0] = 0u | FACE;   // T0, face contact
    ordered[CM1.roundedStartIndex + 1] = 0xFFFFFFFFu;
    ordered[CM1.roundedStartIndex + 2] = 0xFFFFFFFFu;
    ordered[CM1.roundedStartIndex + 3] = 0xFFFFFFFFu;
    // CM2's slice is EMPTY (count == 0) but the word sitting at its start is a
    // live-looking match, so dropping findWithFlag's range guard is visible.
    ordered[CM2.roundedStartIndex + 0] = 1u | FACE;

    // ---- the cases -----------------------------------------------------------
    const std::vector<Case> cases = {
        // nbContacts is 1 wherever it can be: `getNbContacts` is a shift by 27,
        // and a count of 1 is the only value a shift by 28 turns into zero, so a
        // clearing case with count 1 is what pins the shift width.
        { "eE01 on T0: neighbour T1 is a face contact", 0, 0, EE01, 0, 0, 1, true,
          "adj.x = T1, found with eFACE_CONTACT" },
        { "eE12 on T0: the edge is a boundary", 0, 0, EE12, 0, 1, 3, false,
          "adj.y = 0xFFFFFFFF sets needsProcessing before any search" },
        { "eE02 on T0: neighbour T3 produced nothing", 0, 0, EE02, 0, 2, 3, false,
          "adj.z = T3, absent from the ordered list" },
        { "eE12 on T1: neighbour T2 is an EDGE contact only", 0, 1, EE12, 0, 3, 3, false,
          "the edge branch searches eFACE_CONTACT only, so an edge hit does not count" },
        { "eE02 on T1: the search runs off the low end", 0, 1, EE02, 0, 4, 3, false,
          "adj.z = T0; binarySearch returns 0 because data[0] > target" },
        { "eE12 on T2: boundary, where .z would have matched", 0, 2, EE12, 0, 14, 1, false,
          "adj.y = boundary; adj.z is T1, so reading the wrong word clears" },
        { "eE02 on T2: neighbour T1 is a face contact", 0, 2, EE02, 0, 15, 1, true,
          "adj.z = T1; the mirror of the case above" },
        { "eV0 on T0: no triangle at v0 produced a contact", 0, 0, EV0, 0, 5, 3, false,
          "refs(v0) = {T0,T3,T5}, none in the list" },
        { "eV1 on T0: v1 is covered by T1", 0, 0, EV1, 0, 6, 1, true,
          "refs(v1) = {T0,T1,T4}; T1 is a face contact" },
        { "eV1 on T2: v4 is covered by an EDGE contact", 0, 2, EV1, 0, 7, 1, true,
          "refs(v4) = {T4,T5,T2}; T2 matches only via eEDGE_CONTACT, and it is LAST" },
        { "eV0 on T4: both adjacent edges are boundaries", 0, 4, EV0, 0, 8, 3, false,
          "the short-circuit skips the search entirely -- v4 WOULD have cleared" },
        { "eV0 on T5: boundary on .x and .z, but not on .y", 0, 5, EV0, 0, 16, 3, false,
          "eV0 pairs .x with .z, both boundary; pairing .x with .y would search v4 and clear" },
        { "eV1 on T3: v2 is covered by T1", 0, 3, EV1, 0, 9, 3, true,
          "refs(v2) = {T0,T1,T2,T3}; T1 is a face contact" },
        { "eV1 on T3 with a negative determinant", 0, 3, EV1, 1, 10, 3, false,
          "the flip swaps indices.y/.z, so the vertex is v5, whose refs {T3,T4} are absent" },
        { "eE01 on T0 with a negative determinant", 0, 0, EE01, 1, 11, 3, false,
          "the flip swaps adj.x/.z, so the neighbour is T3, which is absent" },
        { "eE02 on T0 with a negative determinant", 0, 0, EE02, 1, 12, 1, true,
          "the flip swaps adj.x/.z, so the neighbour is T1, a face contact" },
        { "eV1 on T2 under CM1", 1, 2, EV1, 0, 0, 3, false,
          "CM1's ordered slice holds only T0, so v4's {T4,T5,T2} miss" },
        { "eE01 on T0 under CM2, whose slice is empty", 2, 0, EE01, 0, 0, 1, false,
          "count == 0, so the range guard must reject before the word is compared" },
        { "nbContacts == 0 short-circuits before anything", 0, 0, EE01, 0, 13, 0, false,
          "the same case that clears at index 0, but the early-out must fire first" },
    };
    const uint32_t nbCases = uint32_t(cases.size());

    // ---- ConvexMeshPair array ------------------------------------------------
    std::vector<uint32_t> pairBuf(3 * 14, 0u);
    auto writePair = [&](uint32_t i, const MeshPairFix& p) {
        uint32_t* w = &pairBuf[i * 14u];
        for (uint32_t t = 0; t < 7u; ++t) w[t] = 0xDEAD0000u | t;  // aToB
        w[7] = p.startIndex;         // @28
        w[8] = p.count;              // @32
        w[9] = p.cmIndex;            // @36
        w[10] = p.roundedStartIndex; // @40
        w[11] = 0x77770000u | i;     // @44 pad
        w[12] = 0x11110000u | i;     // @48
        w[13] = 0x22220000u | i;     // @52
    };
    writePair(0, CM0);
    writePair(1, CM1);
    writePair(2, CM2);

    // ---- ConvexTriNormalAndIndex array ---------------------------------------
    // Stride 16: PxVec3 normal @0, int index @12.  The triangle index carries bits
    // 24-26 SET, so the CUDA's `& 0x0ffffff` -- six F's, which clears those bits
    // along with the count -- is distinguishable from a seven-F mask that would
    // not.  The normal must never move.
    const uint32_t NI_SLOTS = 64;
    std::vector<uint32_t> niBuf(NI_SLOTS * 4, 0u);
    for (uint32_t s = 0; s < NI_SLOTS; ++s)
    {
        niBuf[s * 4 + 0] = 0x40000000u | s;   // normal.x
        niBuf[s * 4 + 1] = 0x40800000u | s;   // normal.y
        niBuf[s * 4 + 2] = 0x40A00000u | s;   // normal.z
        niBuf[s * 4 + 3] = 0u;
    }
    for (const Case& c : cases)
    {
        const MeshPairFix& p = c.cm == 0 ? CM0 : (c.cm == 1 ? CM1 : CM2);
        const uint32_t slot = p.startIndex + c.testOffset;
        const uint32_t nbContacts = c.nbContacts;
        // deferred bit | count | a triangle index with bits 24-26 set
        niBuf[slot * 4 + 3] = 0x80000000u | (nbContacts << 27) | 0x07000000u | (slot & 0xffffu);
    }

    // ---- the stack (sPairsGPU) and the second-pass index array ---------------
    const uint32_t STACK_PAIRS = 200;
    std::vector<uint32_t> stackBuf(STACK_PAIRS * 4, 0xBADB0000u);

    const uint32_t FIRST_REAL = 64;   // the real cases live in the SECOND workgroup
    std::vector<uint32_t> secondPass(FIRST_REAL + nbCases, 0u);

    // The no-op prefix: every one of them points at a pair whose contact count is
    // zero, so they early-out and write nothing wherever they land.
    const uint32_t NOOP_PAIR = 5;
    stackBuf[NOOP_PAIR * 4 + 0] = 0;                 // cmIdx
    stackBuf[NOOP_PAIR * 4 + 1] = 0;                 // triangleIdx
    stackBuf[NOOP_PAIR * 4 + 2] = 20;                // testOffset -> a slot with 0 contacts
    stackBuf[NOOP_PAIR * 4 + 3] = 0;                 // shapeID
    // nbContacts lives in bits 27-30, so the marker must keep those clear: 0x07000000
    // touches bits 24-26 only.  (0x0F000000 does NOT -- bit 27 is set, giving a
    // count of 1, and the no-ops then run the full path and clear this slot.)
    niBuf[(CM0.startIndex + 20) * 4 + 3] = 0x07000000u;  // nbContacts == 0
    for (uint32_t i = 0; i < FIRST_REAL; ++i) secondPass[i] = NOOP_PAIR | EE01;

    for (uint32_t i = 0; i < nbCases; ++i)
    {
        const Case& c = cases[i];
        const uint32_t pairIdx = 100u + i;   // deliberately NOT the thread index
        stackBuf[pairIdx * 4 + 0] = c.cm;
        stackBuf[pairIdx * 4 + 1] = c.triangleIdx;
        stackBuf[pairIdx * 4 + 2] = c.testOffset;
        stackBuf[pairIdx * 4 + 3] = c.shapeID;
        secondPass[FIRST_REAL + i] = pairIdx | c.mask;
    }

    // ---- upload --------------------------------------------------------------
    PxgBufferHandle bMesh = createBufferFrom(dev, mesh);
    const uint64_t meshAddr = dev->getBufferDeviceAddress(bMesh);

    // CtShape / PxgShape, 48 B: scale xyz @0, scaleRot @12, materialIndex @28,
    // hullOrMeshOffset @32, type @40, particleOrSoftbodyId @44.
    std::vector<uint32_t> shapeBuf(2 * 12, 0u);
    auto writeShape = [&](uint32_t i, float sx, float sy, float sz) {
        uint32_t* w = &shapeBuf[i * 12u];
        std::memcpy(&w[0], &sx, 4); std::memcpy(&w[1], &sy, 4); std::memcpy(&w[2], &sz, 4);
        w[6] = 0x3F800000u;                                  // scaleRot.w = 1
        w[7] = 0xEEEE0000u | i;                              // materialIndex
        w[8] = uint32_t(meshAddr & 0xffffffffull);
        w[9] = uint32_t(meshAddr >> 32);
        w[10] = 4u;                                          // PxGeometryType::eTRIANGLEMESH
        w[11] = 0xCCCC0000u | i;
    };
    writeShape(0, 1.0f, 1.0f, 1.0f);
    writeShape(1, 1.0f, 1.0f, -1.0f);   // negative determinant -> flipNormal

    PxgBufferHandle bShapes = createBufferFrom(dev, shapeBuf);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairBuf);
    PxgBufferHandle bNI = createBufferFrom(dev, niBuf);
    PxgBufferHandle bOrdered = createBufferFrom(dev, ordered);
    PxgBufferHandle bStack = createBufferFrom(dev, stackBuf);
    PxgBufferHandle bSecond = createBufferFrom(dev, secondPass);

    std::vector<uint32_t> nbBuf(1, uint32_t(secondPass.size()));
    PxgBufferHandle bNb = createBufferFrom(dev, nbBuf);

    // The double-indirect arguments: each is a buffer holding ONE device address.
    std::vector<uint64_t> pNI(1, dev->getBufferDeviceAddress(bNI));
    std::vector<uint64_t> pOrdered(1, dev->getBufferDeviceAddress(bOrdered));
    std::vector<uint64_t> pSecond(1, dev->getBufferDeviceAddress(bSecond));
    std::vector<uint64_t> pInterm(1, 0ull);   // read by the core's dead variant only
    PxgBufferHandle bPNI = createBufferFrom(dev, pNI);
    PxgBufferHandle bPOrdered = createBufferFrom(dev, pOrdered);
    PxgBufferHandle bPSecond = createBufferFrom(dev, pSecond);
    PxgBufferHandle bPInterm = createBufferFrom(dev, pInterm);

    PpPush push{};
    push.cvxTrimeshPair = dev->getBufferDeviceAddress(bPairs);
    push.cvxTriNIPtr = dev->getBufferDeviceAddress(bPNI);
    push.cvxTriIntermPtr = dev->getBufferDeviceAddress(bPInterm);
    push.orderedCvxTriIntermPtr = dev->getBufferDeviceAddress(bPOrdered);
    push.cvxTriSecondPassPairsPtr = dev->getBufferDeviceAddress(bPSecond);
    push.stackPtr = dev->getBufferDeviceAddress(bStack);
    push.nbSecondPassPairs = dev->getBufferDeviceAddress(bNb);
    push.shapes = dev->getBufferDeviceAddress(bShapes);

    const uint32_t groups = (uint32_t(secondPass.size()) + 63u) / 64u;
    {
        Kernel k("cvxTriPostProcessBda.spv", "convexTrimeshPostProcess", {}, sizeof(PpPush));
        k.dispatch(groups, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bNI, niBuf.size());

    size_t bad = 0;
    uint32_t sawClear = 0, sawKeep = 0;
    for (uint32_t i = 0; i < nbCases; ++i)
    {
        const Case& c = cases[i];
        const MeshPairFix& p = c.cm == 0 ? CM0 : (c.cm == 1 ? CM1 : CM2);
        const uint32_t slot = p.startIndex + c.testOffset;
        const uint32_t was = niBuf[slot * 4 + 3];
        const uint32_t exp = c.expectClear ? (was & 0x00FFFFFFu) : was;
        const uint32_t g = got[slot * 4 + 3];

        c.expectClear ? ++sawClear : ++sawKeep;

        if (g != exp)
        {
            ++bad;
            std::printf("  [%s] index word: got=%08x exp=%08x was=%08x\n    (%s)\n",
                        c.name, g, exp, was, c.why);
        }
        for (uint32_t k = 0; k < 3; ++k)
            if (got[slot * 4 + k] != niBuf[slot * 4 + k])
            {
                ++bad;
                std::printf("  [%s] normal word %u moved: got=%08x was=%08x\n",
                            c.name, k, got[slot * 4 + k], niBuf[slot * 4 + k]);
            }
    }

    // Every NI slot no case owns must be untouched -- including the one the 64
    // no-op entries point at.
    std::vector<bool> owned(NI_SLOTS, false);
    for (const Case& c : cases)
        owned[(c.cm == 0 ? CM0 : (c.cm == 1 ? CM1 : CM2)).startIndex + c.testOffset] = true;
    size_t strayBad = 0;
    for (uint32_t s = 0; s < NI_SLOTS; ++s)
    {
        if (owned[s]) continue;
        for (uint32_t k = 0; k < 4; ++k)
            if (got[s * 4 + k] != niBuf[s * 4 + k]) ++strayBad;
    }

    dev->destroyBuffer(bMesh); dev->destroyBuffer(bShapes); dev->destroyBuffer(bPairs);
    dev->destroyBuffer(bNI); dev->destroyBuffer(bOrdered); dev->destroyBuffer(bStack);
    dev->destroyBuffer(bSecond); dev->destroyBuffer(bNb);
    dev->destroyBuffer(bPNI); dev->destroyBuffer(bPOrdered); dev->destroyBuffer(bPSecond);
    dev->destroyBuffer(bPInterm);

    PXG_CHECK_MSG(bad == 0, "cvxTriPostProcess: %zu mismatches", bad);
    PXG_CHECK_MSG(strayBad == 0, "cvxTriPostProcess: %zu words written outside any case's slot",
                  strayBad);
    PXG_CHECK_MSG(sawClear >= 5, "too few clearing cases (%u) to call the cull exercised", sawClear);
    PXG_CHECK_MSG(sawKeep >= 5, "too few keeping cases (%u); a kernel that never clears would pass",
                  sawKeep);
}

} // namespace
