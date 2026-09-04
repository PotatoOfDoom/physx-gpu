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

// Verification of convexHfPostProcessBda -- convexHeightfieldPostProcessCore
// (convexMeshPostProcess.cu:410-582).  Laid out as its trimesh twin
// (testCvxTriPostProcess.cpp) so the two read against each other, with the cases
// chosen for the places the two kernels DIFFER.
//
// THE GRID IS 5 ROWS x 6 COLUMNS, AND NOT SQUARE ON PURPOSE.  It was 5x5 until a
// P9.51c sweep reported `J1-rows-columns-swapped` as a survivor -- with equal
// extents that mutant is a literal no-op, and `I4-walk-cell-row-column-swapped`
// survived for the same reason, because a square walk visits the same cell SET
// either way.  A fixture whose grid cannot tell rows from columns cannot test a
// kernel that must.
//
// The cells are the vertices with row < 4 and column < 5, and every triangle index
// is cell*2 (+1 for the second).  The TESSELLATION FLAG (bit 23 of a sample word,
// which is also materialIndex0's high bit) is set on CELLS 7, 12 AND 19 -- one
// interior, one in COLUMN 0, one in ROW rows-2 -- because the shared-diagonal
// adjacency arm has a guard for each of those positions and a single interior
// shared cell reaches none of them.
//
// THE ORDERED LIST for CM0 holds triangles 1, 2, 13, 24 and 32 as FACE contacts
// and triangle 12 as an EDGE contact.  `findTriangle` (convexMeshPostProcess.cu:154)
// ORs 0x80000000 and nothing else -- both the edge branch and the vertex walk use
// it -- so triangle 12 must NEVER match.  Its presence is what separates this
// kernel from the trimesh sibling, whose VERTEX branch accepts an edge contact too.
//
// WHAT THE CASES ARE FOR
//
//  * A BOUNDARY ADJACENCY CLEARS THE CONTACT HERE.  The trimesh core's edge arms
//    each end with `if (triangleIndex == 0xFFFFFFFF) needsProcessing = true;`
//    (:298, :306, :314); the heightfield core's do not (:506-516).  So a boundary
//    leaves `triangleIndex` at 0xFFFFFFFF, both branches are skipped, and the
//    contact is CLEARED -- the opposite of the sibling.  That is an upstream
//    asymmetry, transcribed rather than "fixed", and seven cases pin it.
//  * EVERY GUARD IN THE FOUR ADJACENCY ARMS gets a case that sits exactly on it:
//    column 0 and column cols-2, row rows-2, in both the shared and the unshared
//    arm.  Each is a BOUNDARY-expecting case whose guard mutant yields a real
//    triangle index that is absent from the list, so the verdict flips.
//  * THE ADJACENCY PERMUTATION under a negative column scale swaps slots 0 and 2.
//    Triangle 13 carries adj0 = 24 (in the list) and adj2 = 14 (absent), so the
//    same triangle appears four times -- eE01/eE02 with and without the flip --
//    and the four verdicts form two opposed pairs.
//  * THE 3x3 WALK gets its own contact managers.  CM3's list holds ONE triangle,
//    3, which is cell 1's SECOND triangle and lies outside the cell set a
//    row/column-swapped walk would visit; the case over it therefore dies to both
//    `I3-walk-visits-only-the-first-triangle` and `I4-walk-cell-row-column-swapped`,
//    which a case sharing CM0's larger list could not, because some first triangle
//    in the list matched first and hid them.
//  * CM4 does the same for the CONTACT's own vertex permutation: one entry,
//    triangle 1, reachable only from the wrongHanded vertex (7) and not from the
//    vertex the unpermuted read would pick (2).
//  * THE RANGE GUARD in findTriangle: CM2's slice has count 0 but the word at its
//    start is a live-looking match, so an unguarded search clears.
//
// WHAT NO CASE HERE CAN SEPARATE, stated rather than left as a silent gap:
//
//  * `isActiveEdge` is ALWAYS TRUE on this path.  A heightfield adjacency is a
//    plain triangle index or BOUNDARY, the NONCONVEX bit is set only by the mesh
//    cooker, and BOUNDARY is caught by the predicate's first clause.  The sweep
//    mutant that drops it survives, and that is a proof, not a gap.
//  * THE WALK'S VERTEX PERMUTATION is unobservable.  The walk asks whether the
//    vertex is one of the triangle's three, and set membership does not care what
//    order they are in.  (The CONTACT's own permutation IS observable -- it picks
//    which single vertex eV0/eV1/eV2 names -- and CM4 is the case that shows it.)
//
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

const uint32_t HF_ROWS = 5u, HF_COLS = 6u;

// ConvexMeshPair fields, as running FIELD offsets (not the declaration's trailing
// end offsets).  Pinned against the engine header in testNpAbiLayout.cpp.
struct MeshPairFix { uint32_t startIndex, count, cmIndex, roundedStartIndex; };

// CM0: 6 ordered entries at 12; its NI slots start at 7.  CM1: 1 entry at 24, NI
// slots at 40.  startIndex and roundedStartIndex differ, and both differ from
// cmIndex and from the pad, so any of the four offsets read wrongly is visible.
const MeshPairFix CM0 = { 7u, 6u, 55u, 12u };
const MeshPairFix CM1 = { 40u, 1u, 66u, 24u };
// CM2 has count == 0.  `findTriangle`'s `index < count` guard is the only thing
// stopping the search: binarySearch's `left ? left - 1 : 0` hands back index 0 and
// the word at the slice start would be compared anyway.  CM2's slice starts with
// exactly the word an eE01-on-t12 search looks for.
const MeshPairFix CM2 = { 50u, 0u, 88u, 28u };
// CM3 and CM4 exist for the WALK, and each holds exactly ONE triangle -- see the
// header.  A single entry is the point: with CM0's five, some first triangle
// matched before the walk reached the one that discriminates.
const MeshPairFix CM3 = { 55u, 1u, 99u, 32u };
const MeshPairFix CM4 = { 58u, 1u, 111u, 34u };

const MeshPairFix& cmOf(uint32_t i)
{
    switch (i)
    {
        case 0: return CM0;
        case 1: return CM1;
        case 2: return CM2;
        case 3: return CM3;
        default: return CM4;
    }
}

struct Case
{
    const char* name;
    uint32_t cm;          // 0, 1 or 2
    uint32_t triangleIdx;
    uint32_t mask;
    uint32_t shapeID;     // 0 = positive scales, 1 = negative COLUMN scale
    uint32_t testOffset;
    uint32_t nbContacts;  // 0 drives the early-out; 1 vs 3 pins the shift width
    bool expectClear;
    // True when the adjacency this case selects is BOUNDARY.  A FIELD, not a
    // substring of `why`: the guard below counts these, and a guard that greps its
    // own prose stops counting the moment someone rewords a comment -- which is
    // how this fixture failed its first run.
    bool boundaryAdj;
    const char* why;
};

PXG_TEST(cvxHfPostProcess)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the heightfield blob ------------------------------------------------
    // [0] rows, [1] columns, then one 32-bit PxHeightFieldSample per vertex:
    // PxI16 height @0, materialIndex0 @2 (bit 23 = the TESSELLATION flag),
    // materialIndex1 @3.  The heights are never read by this kernel; they are
    // given distinct values anyway, so a decoder that took the flag from the wrong
    // byte reads something that varies instead of a constant zero.
    std::vector<uint32_t> hf;
    hf.push_back(HF_ROWS);
    hf.push_back(HF_COLS);
    for (uint32_t i = 0; i < HF_ROWS * HF_COLS; ++i)
    {
        const int16_t height = int16_t(int(i) * 37 - 300);
        // Cells 7 (interior), 12 (column 0) and 19 (row rows-2) take the shared
        // arm; every other cell takes the unshared one.
        const bool tess = (i == 7u || i == 12u || i == 19u);
        const uint32_t mat0 = (i % 5u) + 1u;  // < 128, so bit 23 stays ours
        const uint32_t mat1 = (i % 7u) + 2u;
        hf.push_back(uint32_t(uint16_t(height))
                     | (mat0 << 16) | (tess ? (1u << 23) : 0u) | (mat1 << 24));
    }

    // ---- the ordered per-CM triangle lists -----------------------------------
    // Sorted ascending UNSIGNED, then padded with the radix sort's 0xFFFFFFFF
    // sentinel.  The search is bounded by `count`.
    std::vector<uint32_t> ordered(40, 0x5A5A5A5Au);
    ordered[CM0.roundedStartIndex + 0] = 1u | FACE;
    ordered[CM0.roundedStartIndex + 1] = 2u | FACE;
    ordered[CM0.roundedStartIndex + 2] = 13u | FACE;
    ordered[CM0.roundedStartIndex + 3] = 24u | FACE;
    ordered[CM0.roundedStartIndex + 4] = 32u | FACE;
    ordered[CM0.roundedStartIndex + 5] = 12u | EDGE;   // must NEVER match
    ordered[CM0.roundedStartIndex + 6] = 0xFFFFFFFFu;
    ordered[CM0.roundedStartIndex + 7] = 0xFFFFFFFFu;
    ordered[CM1.roundedStartIndex + 0] = 0u | FACE;    // triangle 0 only
    ordered[CM1.roundedStartIndex + 1] = 0xFFFFFFFFu;
    ordered[CM1.roundedStartIndex + 2] = 0xFFFFFFFFu;
    ordered[CM1.roundedStartIndex + 3] = 0xFFFFFFFFu;
    // CM2's slice is EMPTY (count == 0) but the word at its start is a live match.
    ordered[CM2.roundedStartIndex + 0] = 1u | FACE;
    // CM3: triangle 3 is cell 1's SECOND triangle, and cell 1 is outside the cell
    // set a row/column-swapped walk visits from cell 2.
    ordered[CM3.roundedStartIndex + 0] = 3u | FACE;
    // Padded to a multiple of four with the radix sort's own 0xFFFFFFFF sentinel,
    // not left on the buffer fill: `C4-search-bounded-by-the-rounded-count` is a
    // proven equivalence ONLY because every word past `count` is 0xFFFFFFFF, which
    // is >= every target the searches can form.  A fill value there would make the
    // argument false for these two slices while leaving the sweep green.
    ordered[CM3.roundedStartIndex + 1] = 0xFFFFFFFFu;
    ordered[CM3.roundedStartIndex + 2] = 0xFFFFFFFFu;
    ordered[CM3.roundedStartIndex + 3] = 0xFFFFFFFFu;
    // CM4: triangle 1 holds vertex 7 -- the vertex a wrongHanded eV1 names -- and
    // not vertex 2, which is the one an unpermuted read would pick.
    ordered[CM4.roundedStartIndex + 0] = 1u | FACE;
    ordered[CM4.roundedStartIndex + 1] = 0xFFFFFFFFu;
    ordered[CM4.roundedStartIndex + 2] = 0xFFFFFFFFu;
    ordered[CM4.roundedStartIndex + 3] = 0xFFFFFFFFu;

    // ---- the cases -----------------------------------------------------------
    //
    // The adjacency arithmetic these expectations rest on, worked out by hand from
    // heightfieldUtil.cuh:167-250 with rows = cols = 5.  UNSHARED cell c:
    //   first  t=2c:   a0 = (c >= 4)      ? 2(c-5)+1 : BND
    //                  a1 = t+1
    //                  a2 = (c % 5 != 0)  ? t-1      : BND
    //   second t=2c+1: a0 = (c / 5 != 3)  ? 2(c+5)   : BND
    //                  a1 = t-1
    //                  a2 = (c % 5 < 3)   ? t+1      : BND
    // SHARED cell c (cell 7 only):
    //   first  t=2c:   a0 = (c % 5 != 0)  ? t-1      : BND
    //                  a1 = t+1
    //                  a2 = (c / 5 != 3)  ? 2(c+5)+1 : BND
    //   second t=2c+1: a0 = (c % 5 < 3)   ? t+1      : BND
    //                  a1 = t-1
    //                  a2 = (c >= 4)      ? 2(c-5)   : BND
    // and the vertex indices, UNSHARED:  first (c, c+1, c+5)   second (c+6, c+5, c+1)
    //                          SHARED:   first (c+5, c, c+6)   second (c+1, c+6, c)
    // The adjacency arithmetic these expectations rest on, from
    // heightfieldUtil.cuh:167-250 with rows = 5, cols = 6.  UNSHARED cell c:
    //   first  t=2c:   a0 = (c >= 5)      ? 2(c-6)+1 : BND
    //                  a1 = t+1
    //                  a2 = (c % 6 != 0)  ? t-1      : BND
    //   second t=2c+1: a0 = (c / 6 != 3)  ? 2(c+6)   : BND
    //                  a1 = t-1
    //                  a2 = (c % 6 < 4)   ? t+1      : BND
    // SHARED cell c (cells 7, 12 and 19):
    //   first  t=2c:   a0 = (c % 6 != 0)  ? t-1      : BND
    //                  a1 = t+1
    //                  a2 = (c / 6 != 3)  ? 2(c+6)+1 : BND
    //   second t=2c+1: a0 = (c % 6 < 4)   ? t+1      : BND
    //                  a1 = t-1
    //                  a2 = (c >= 5)      ? 2(c-6)   : BND
    // and the vertex indices, UNSHARED:  first (c, c+1, c+6)   second (c+7, c+6, c+1)
    //                          SHARED:   first (c+6, c, c+7)   second (c+1, c+7, c)
    const std::vector<Case> cases = {
        // --- the edge branch, neighbour present -------------------------------
        // nbContacts is 1 wherever it can be: `getNbContacts` is a shift by 27, and
        // 1 is the only count a shift by 28 turns into zero.
        { "eE01 on t12 (cell 6): adj0 = 1, a face contact", 0, 12, EE01, 0, 0, 1, true, false,
          "unshared first, cell 6 >= 5: a0 = 2*(6-6)+1 = 1" },
        { "eE12 on t12: adj1 = 13, a face contact", 0, 12, EE12, 0, 1, 1, true, false,
          "a1 = t+1 = 13" },
        { "eE02 on t12: adj2 is BOUNDARY", 0, 12, EE02, 0, 2, 3, true, true,
          "cell 6 % 6 == 0, so a2 = BOUNDARY; with no `needsProcessing = true` guard "
          "the contact is cleared -- the trimesh core would keep it" },
        { "eE01 on t13: adj0 = 24, a face contact", 0, 13, EE01, 0, 3, 1, true, false,
          "unshared second, 6/6 = 1 != 3: a0 = 2*(6+6) = 24" },
        { "eE12 on t13: adj1 = 12, present but EDGE-flagged", 0, 13, EE12, 0, 4, 3, false, false,
          "findTriangle searches 12|0x80000000; the list holds 12|0xC0000000" },
        { "eE02 on t13: adj2 = 14, absent", 0, 13, EE02, 0, 5, 3, false, false,
          "6 % 6 = 0 < 4, so a2 = t+1 = 14" },

        // --- the SHARED diagonal ---------------------------------------------
        { "eE01 on t14 (cell 7, SHARED): adj0 = 13", 0, 14, EE01, 0, 6, 1, true, false,
          "shared first: a0 = t-1 = 13.  The unshared arm would give 2*(7-6)+1 = 3, "
          "absent, so the two branches are separated here" },
        { "eE02 on t14 (SHARED): adj2 = 27, absent", 0, 14, EE02, 0, 7, 3, false, false,
          "shared first: a2 = 2*(7+6)+1 = 27" },
        { "eE02 on t15 (cell 7, SHARED, second): adj2 = 2", 0, 15, EE02, 0, 8, 1, true, false,
          "shared second, cell 7 >= 5: a2 = 2*(7-6) = 2, a face contact" },
        { "eE01 on t15 (SHARED, second): adj0 = 16, absent", 0, 15, EE01, 0, 9, 3, false, false,
          "7 % 6 = 1 < 4, so a0 = t+1 = 16" },

        // --- one case sitting exactly on each guard, in both arms -------------
        { "eE01 on t0 (cell 0): adj0 is BOUNDARY", 0, 0, EE01, 0, 10, 3, true, true,
          "unshared first, cell 0 < 5" },
        { "eE02 on t0: adj2 is BOUNDARY", 0, 0, EE02, 0, 11, 1, true, true,
          "unshared first, cell 0 % 6 == 0" },
        { "eE02 on t9 (cell 4, column cols-2): adj2 is BOUNDARY", 0, 9, EE02, 0, 12, 1, true, true,
          "unshared second: 4 % 6 = 4 is NOT < 4, so the column guard must fail" },
        { "eE01 on t24 (cell 12, SHARED, column 0): adj0 is BOUNDARY", 0, 24, EE01, 0, 13, 1, true, true,
          "shared first: 12 % 6 == 0.  Dropping the guard gives a0 = 23, absent, so "
          "the verdict flips -- this is the case the first P9.51b sweep lacked" },
        { "eE02 on t38 (cell 19, SHARED, row rows-2): adj2 is BOUNDARY", 0, 38, EE02, 0, 14, 1, true, true,
          "shared first: 19 / 6 = 3 == rows-2.  Dropping or off-by-one'ing the guard "
          "gives 2*(19+6)+1 = 51, absent" },
        { "eE01 on t37 (cell 18, row rows-2): adj0 is BOUNDARY", 0, 37, EE01, 0, 15, 1, true, true,
          "unshared second: 18 / 6 == 3" },
        { "eE02 on t33 (cell 16, column cols-2): adj2 is BOUNDARY", 0, 33, EE02, 0, 16, 1, true, true,
          "unshared second: 16 % 6 = 4 is NOT < 4" },
        { "eE01 on t39 (cell 19, SHARED, second): adj0 = 40, absent", 0, 39, EE01, 0, 17, 3, false, false,
          "19 % 6 = 1 < 4, so a0 = t+1 = 40" },
        { "eE02 on t39 (SHARED, second): adj2 = 26, absent", 0, 39, EE02, 0, 18, 3, false, false,
          "cell 19 >= 5, so a2 = 2*(19-6) = 26" },

        // --- the vertex branch over CM0's larger list -------------------------
        { "eV0 on t12: vertex 6 is covered by triangle 1", 0, 12, EV0, 0, 19, 1, true, false,
          "walk over cells 0,1,6,7,12,13; cell 0's second triangle (1) holds "
          "vertices (7,6,1) and is a face contact" },
        { "eV0 on t8: vertex 4 is covered by nothing", 0, 8, EV0, 0, 20, 3, false, false,
          "walk over cells 3,4,5,9,10,11; no listed triangle reaches vertex 4" },

        // --- the handedness permutation, one triangle, two opposed pairs ------
        { "eE01 on t13 with a NEGATIVE column scale", 0, 13, EE01, 1, 21, 3, false, false,
          "wrongHanded swaps adjacency slots 0 and 2, so adj0 becomes a2 = 14, absent "
          "-- the opposite verdict to the same case without the flip" },
        { "eE02 on t13 with a NEGATIVE column scale", 0, 13, EE02, 1, 22, 1, true, false,
          "the mirror: adj2 becomes a0 = 24, a face contact" },

        // --- the other contact managers ---------------------------------------
        { "eE01 on t12 under CM1", 1, 12, EE01, 0, 0, 3, false, false,
          "CM1's slice holds triangle 0 only, so adj0 = 1 misses" },
        { "eE01 on t12 under CM2, whose slice is empty", 2, 12, EE01, 0, 0, 3, false, false,
          "count == 0, so the range guard must reject before the word is compared" },

        // --- the walk, on lists built for it ----------------------------------
        { "eV0 on t4 under CM3: only a SECOND triangle matches", 3, 4, EV0, 0, 0, 1, true, false,
          "vertex 2; the walk over cells 1,2,3,7,8,9 reaches triangle 3 -- cell 1's "
          "SECOND -- and cell 1 is outside the set a row/column-swapped walk visits" },
        { "eV0 on t12 under CM1: vertex 6, and NOT vertex 12", 1, 12, EV0, 0, 1, 1, true, false,
          "CM1's slice holds triangle 0, which carries vertices (0,1,6).  eV0 names "
          "tv[0] = 6 and clears; tv[2] = 12 is reached by no listed triangle, so a "
          "kernel that read eV0 as tv[2] would keep -- this is the case that dies to "
          "`G2-EV0-and-EV2-swapped`" },
        { "eV0 on t1 under CM3: the match is one column PAST the cell", 3, 1, EV0, 0, 1, 1, true, false,
          "vertex 7; triangle 3 lives in cell 1, and the walk from cell 0 reaches it "
          "only through `endCol = min(col, cols-2) + 1`.  Without the +1 the walk "
          "never leaves cell 0 and this becomes a keep" },
        { "eV1 on t2 under CM4, wrongHanded", 4, 2, EV1, 1, 0, 1, true, false,
          "the flip makes eV1 name vertex 7, which triangle 1 holds; without it eV1 "
          "would name vertex 2, which triangle 1 does not hold" },

        // --- the early-out -----------------------------------------------------
        { "nbContacts == 0 short-circuits before anything", 0, 12, EE01, 0, 23, 0, false, false,
          "the same case that clears first, but the early-out must fire first" },
    };
    const uint32_t nbCases = uint32_t(cases.size());

    // ---- ConvexMeshPair array ------------------------------------------------
    std::vector<uint32_t> pairBuf(5 * 14, 0u);
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
    writePair(3, CM3);
    writePair(4, CM4);

    // ---- ConvexTriNormalAndIndex array ---------------------------------------
    // Stride 16: PxVec3 normal @0, int index @12.  The triangle index carries bits
    // 24-26 SET, so the CUDA's `& 0x0ffffff` -- six F's, which clears those bits
    // along with the count -- is distinguishable from a seven-F mask.  The normal
    // must never move.
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
        const MeshPairFix& p = cmOf(c.cm);
        const uint32_t slot = p.startIndex + c.testOffset;
        niBuf[slot * 4 + 3] =
            0x80000000u | (c.nbContacts << 27) | 0x07000000u | (slot & 0xffffu);
    }

    // ---- the stack (sPairsGPU) and the second-pass index array ---------------
    // THE REAL CASES START AT INDEX 64, so a kernel that dropped the `gid.x * 64`
    // term from its thread index would run the no-ops twice and never reach a real
    // case.  The pair indices are 100 + case, never the thread index.
    const uint32_t STACK_PAIRS = 200;
    std::vector<uint32_t> stackBuf(STACK_PAIRS * 4, 0xBADB0000u);

    const uint32_t FIRST_REAL = 64;
    std::vector<uint32_t> secondPass(FIRST_REAL + nbCases, 0u);

    const uint32_t NOOP_PAIR = 5;
    stackBuf[NOOP_PAIR * 4 + 0] = 0;    // cmIdx
    stackBuf[NOOP_PAIR * 4 + 1] = 12;   // triangleIdx
    stackBuf[NOOP_PAIR * 4 + 2] = 30;   // testOffset -> a slot with 0 contacts
    stackBuf[NOOP_PAIR * 4 + 3] = 0;    // shapeID
    // nbContacts lives in bits 27-30, so the marker must keep those clear:
    // 0x07000000 touches bits 24-26 only.
    niBuf[(CM0.startIndex + 30) * 4 + 3] = 0x07000000u;
    for (uint32_t i = 0; i < FIRST_REAL; ++i) secondPass[i] = NOOP_PAIR | EE01;

    for (uint32_t i = 0; i < nbCases; ++i)
    {
        const Case& c = cases[i];
        const uint32_t pairIdx = 100u + i;
        stackBuf[pairIdx * 4 + 0] = c.cm;
        stackBuf[pairIdx * 4 + 1] = c.triangleIdx;
        stackBuf[pairIdx * 4 + 2] = c.testOffset;
        stackBuf[pairIdx * 4 + 3] = c.shapeID;
        secondPass[FIRST_REAL + i] = pairIdx | c.mask;
    }

    // ---- upload --------------------------------------------------------------
    PxgBufferHandle bHf = createBufferFrom(dev, hf);
    const uint64_t hfAddr = dev->getBufferDeviceAddress(bHf);

    // CtShape / PxgShape, 48 B: scale xyz @0, scaleRot @12, materialIndex @28,
    // hullOrMeshOffset @32, type @40, particleOrSoftbodyId @44.
    std::vector<uint32_t> shapeBuf(2 * 12, 0u);
    auto writeShape = [&](uint32_t i, float sx, float sy, float sz) {
        uint32_t* w = &shapeBuf[i * 12u];
        std::memcpy(&w[0], &sx, 4); std::memcpy(&w[1], &sy, 4); std::memcpy(&w[2], &sz, 4);
        w[6] = 0x3F800000u;                                  // scaleRot.w = 1
        w[7] = 0xEEEE0000u | i;                              // materialIndex
        w[8] = uint32_t(hfAddr & 0xffffffffull);
        w[9] = uint32_t(hfAddr >> 32);
        w[10] = 6u;                                          // PxGeometryType::eHEIGHTFIELD
        w[11] = 0xCCCC0000u | i;
    };
    // The HEIGHT scale is negative on BOTH shapes and differs between them: the
    // handedness xor reads the ROW and COLUMN scales only, so a port that let the
    // height scale into it flips shape 0 as well and every expectation moves.
    writeShape(0, 2.0f, -0.5f, 1.5f);
    writeShape(1, 2.0f, -0.5f, -1.5f);   // negative COLUMN scale -> wrongHanded

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
        Kernel k("convexHfPostProcessBda.spv", "convexHeightfieldPostProcess", {}, sizeof(PpPush));
        k.dispatch(groups, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bNI, niBuf.size());

    size_t bad = 0;
    uint32_t sawClear = 0, sawKeep = 0, sawBoundaryClear = 0, sawShared = 0, sawWrongHanded = 0;
    for (uint32_t i = 0; i < nbCases; ++i)
    {
        const Case& c = cases[i];
        const MeshPairFix& p = cmOf(c.cm);
        const uint32_t slot = p.startIndex + c.testOffset;
        const uint32_t was = niBuf[slot * 4 + 3];
        const uint32_t exp = c.expectClear ? (was & 0x00FFFFFFu) : was;
        const uint32_t g = got[slot * 4 + 3];

        c.expectClear ? ++sawClear : ++sawKeep;
        if (c.shapeID == 1u) ++sawWrongHanded;
        if ((c.triangleIdx >> 1) == 7u) ++sawShared;

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
    for (const Case& c : cases)
        if (c.expectClear && c.boundaryAdj) ++sawBoundaryClear;

    // Every NI slot no case owns must be untouched -- including the one the 64
    // no-op entries point at.
    std::vector<bool> owned(NI_SLOTS, false);
    for (const Case& c : cases)
        owned[cmOf(c.cm).startIndex + c.testOffset] = true;
    size_t strayBad = 0;
    for (uint32_t s = 0; s < NI_SLOTS; ++s)
    {
        if (owned[s]) continue;
        for (uint32_t k = 0; k < 4; ++k)
            if (got[s * 4 + k] != niBuf[s * 4 + k]) ++strayBad;
    }

    dev->destroyBuffer(bHf); dev->destroyBuffer(bShapes); dev->destroyBuffer(bPairs);
    dev->destroyBuffer(bNI); dev->destroyBuffer(bOrdered); dev->destroyBuffer(bStack);
    dev->destroyBuffer(bSecond); dev->destroyBuffer(bNb);
    dev->destroyBuffer(bPNI); dev->destroyBuffer(bPOrdered); dev->destroyBuffer(bPSecond);
    dev->destroyBuffer(bPInterm);

    PXG_CHECK_MSG(bad == 0, "cvxHfPostProcess: %zu mismatches", bad);
    PXG_CHECK_MSG(strayBad == 0, "cvxHfPostProcess: %zu words written outside any case's slot",
                  strayBad);
    PXG_CHECK_MSG(sawClear >= 5, "too few clearing cases (%u) to call the cull exercised", sawClear);
    PXG_CHECK_MSG(sawKeep >= 5, "too few keeping cases (%u); a kernel that never clears would pass",
                  sawKeep);
    PXG_CHECK_MSG(sawBoundaryClear >= 3,
                  "too few BOUNDARY cases (%u); the upstream asymmetry against the trimesh core "
                  "is the one behaviour a reader is most likely to 'fix'", sawBoundaryClear);
    PXG_CHECK_MSG(sawShared >= 2, "no case sits on the tessellation-flagged cell");
    PXG_CHECK_MSG(sawWrongHanded >= 2, "no case exercises the handedness permutation");
}

} // namespace
