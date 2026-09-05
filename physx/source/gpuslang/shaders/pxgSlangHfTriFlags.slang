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

// pxgSlangHfTriFlags.slang -- `computeTriangleFlags` (convexHeightfield.cu
// :565-614) reading the sample grid itself, plus `selectNormal`
// (capsuleTriangle.cuh:38-98) which is the only consumer of what it produces.
// P9.52b-1.
//
// The heightfield capsule arm needs both, and so does its trimesh twin
// (convexMesh.cu:541).  There is a verified Phase-1 port of the classification
// (`npHfTriFlags`, tranche 119), but it takes the three ADJACENT TRIANGLES
// PRE-DECODED, which is exactly the part that has to change on the engine ABI: on
// a real heightfield the adjacent triangles come out of the same sample grid the
// current one did, by the same handedness-permuting fetch.
//
// ===========================================================================
// WHAT THE FLAGS MEAN, AND WHY THE LOW THREE BITS ARE DEAD
// ===========================================================================
//
// Each of the three edges gets classified into ONE of three outcomes:
//
//   bit a+3 set   the edge is ACTIVE -- either a convex fold sharp enough to
//                 matter (`proj < 0.997f`), or a boundary edge
//   bit a   set   the edge is a SILHOUETTE edge
//   nothing       the neighbour is coplanar enough to ignore
//
// `selectNormal` reads bits 3, 4 and 5 and NOTHING ELSE, so the silhouette
// marking is already inert at its only consumer.  It is dead a second time over:
// the branch that sets it is `else` to `boundaryCollisions`, and
// `boundaryCollisions` is ALWAYS TRUE -- see below.  Both facts are stated because
// either alone would leave a reader wondering whether the port dropped something.
//
// ===========================================================================
// `boundaryCollisions` IS A CONSTANT `true`, AND IT CANNOT BE TRANSCRIBED
// ===========================================================================
//
// UPSTREAM CANDIDATE 48.  The .cu derives it from a `flags` word it reads twice
// wrongly (convexHeightfield.cu:656 advances a `const PxU32*` by
// `sizeof(PxU32) * rows * columns`, four times too far; :658 reinterprets the
// POINTER VARIABLE as a PxU16 instead of dereferencing it).  The second defect
// makes the first harmless -- the over-advanced pointer is never read -- and the
// value that lands in `flags` is the low half of a DEVICE ADDRESS.
//
// It feeds only `boundaryCollisions = !(flags & eNO_BOUNDARY_EDGES)`, and that
// flag is BIT 0 while the address is always a multiple of 4, so the answer is
// `true` on every input and every device.  Which is what the public header says it
// should be (PxHeightFieldFlag.h:136: "This flag is ignored in contact generation
// with sphere and capsule shapes.").
//
// Bit-exact transcription is not merely awkward here, it is IMPOSSIBLE: a
// host-chosen device address has no counterpart in the Vulkan address space, so
// there is no "the same bits" to reproduce.  The port writes the constant and
// cites both legs -- the documented contract, and the alignment argument that puts
// upstream's own code on it.  DOCUMENTED, NOT REPORTED, like the other 49.
//
// ===========================================================================
// THE ADJACENCY GUARD IS UPSTREAM'S OWN
// ===========================================================================
//
// `if (adjTriIndices[a] != 0xFFFFFFFF)` at :583 is in the .cu and is transcribed
// verbatim.  It is worth naming because the SPHERE arm of the same file omits the
// identical test on the identical array (upstream candidate 49) -- this is the
// place that shows the convention was upstream's all along.

// ===========================================================================
// THE MUTATION SWEEP: 35 MUTANTS, 31 KILLED, FOUR EQUIVALENCES
// ===========================================================================
//
// 30 killed by a failing fixture, 1 killed by fault, 4 survived.
//
// The first sweep left ELEVEN, and six of those were the FIXTURE, not the module.
// Five came from one cause: the barycentric pairs were only ever 0, 0.25, 0.5 or 1,
// so `zero` could be moved from 1e-6 to 1e-1 and `one` from 0.999999 to 0.99
// without changing a single classification.  The pairs now include values just
// above the epsilon, just below `one`, EXACTLY at the epsilon (where `>` and `>=`
// disagree) and a sum EXACTLY at the 0.9999 threshold, built by subtraction inside
// the same binade so it is exact.  The sixth was coverage rather than resolution:
// the two vertex arms were never asked about a triangle whose edge-12 and edge-20
// bits differ, so their masks could be swapped unseen -- the fixture now drives the
// full triangle-by-pair cross product and guards that split explicitly.
//
// `A12-projD-non-strict` (`projD < 0` -> `<= 0`) is worth naming among the kills:
// it dies only because the fixture contains FLAT ground, where projD is exactly
// zero.  On any curved heightfield the two spellings agree.
//
// The four survivors are equivalences, in the same families the sphere core's
// sweep found:
//
//   A9-adj-normal-normalized-early   normalizing the neighbour's normal before
//       `projD` is a POSITIVE SCALING of a sign test, and `proj` is then computed
//       from an already-unit vector whose second normalize is a rescale by
//       1 +- an ulp.  Equivalent in exact arithmetic; the residue could only
//       matter if `proj` sat within an ulp of 0.997.
//   A10-projD-plane-point-is-av1     `other - av0` vs `other - av1`.  Both are
//       vertices of the SAME neighbour, so the difference is
//       `dot(adjNormal, av1 - av0)`, which is identically zero -- the same family
//       as C8 in the sphere core's sweep.
//   A19-silhouette-arm-marks-active  the silhouette arm is unreachable, because it
//       is `else` to a `boundaryCollisions` that is a constant true.  This
//       survivor IS the measurement behind that claim.
//   A20-normalize-guard-dropped      the `m2 > 0` guard's false arm is unreachable:
//       it sits inside `projD < 0`, and a zero adjNormal would make projD zero.

module pxgSlangHfTriFlags;

import pxgSlangHeightfield;
import pxgSlangPointTriangle;   // the one explicit dot/cross in the corpus (P9.56)

// capsuleTriangle.cuh:34-36.  "PT: important value, don't change" -- the numbering
// is load-bearing because computeTriangleFlags sets bit `a + 3` for edge `a`, so
// edge 0 is 01, edge 1 is 12 and edge 2 is 20.
public static const uint ETD_CONVEX_EDGE_01 = 1u << 3;
public static const uint ETD_CONVEX_EDGE_12 = 1u << 4;
public static const uint ETD_CONVEX_EDGE_20 = 1u << 5;

// See the header: the .cu computes this from a word it cannot actually read, and
// the value is `true` for every input.
public static const bool HTF_BOUNDARY_COLLISIONS = true;

// computeTriangleFlags (convexHeightfield.cu:565-614).
//
// `adj` is the adjacency triple `hfGetTriangleAdj` produced for the CURRENT
// triangle -- already handedness-permuted, so slot `a` names the neighbour across
// edge `a`.  The neighbour itself is fetched with the three-vertex overload,
// matching the .cu's `getTriangle(adjTri, NULL, ...)` at :585.
public uint htfTriangleFlags(uint64_t samples, uint3 adj,
                             float3 cur0, float3 cur1, float3 cur2,
                             float3 triNormal, float3 scale, uint columns)
{
    // nextInd = {2, 0, 1} (:571): edge `a` is opposite vertex `nextInd[a]`, so the
    // vertex tested against the neighbour's plane is the one the shared edge does
    // NOT contain.  Written as a select rather than an array so the index cannot
    // be mistaken for the vertex number it is not.
    uint triFlags = 0u;

    for (uint a = 0u; a < 3u; ++a)
    {
        const uint adjIdx = (a == 0u) ? adj.x : ((a == 1u) ? adj.y : adj.z);
        if (adjIdx == HF_BOUNDARY)
            continue;

        float3 av0, av1, av2;
        hfGetTriangle(samples, adjIdx, scale, columns, av0, av1, av2);

        // DENORMALIZED here (:587) -- the normalize happens later and only inside
        // the `projD < 0` arm, so a coplanar or boundary edge never pays for it and
        // never rounds through it.
        float3 adjNormal = ptCross(av1 - av0, av2 - av0);

        const uint otherIndex = (a == 0u) ? 2u : ((a == 1u) ? 0u : 1u);
        const float3 other = (otherIndex == 0u) ? cur0 : ((otherIndex == 1u) ? cur1 : cur2);
        const float projD = ptDot(adjNormal, other - av0);

        if (projD < 0.0f)
        {
            // PxVec3::normalize (PxVec3.h:260) -- `m = magnitude(); if (m > 0)
            // *this /= m`, which is ONE reciprocal and three multiplies, and leaves
            // a zero vector alone rather than returning one.  NOT getNormalized:
            // the two agree numerically, and this is the one the .cu calls.
            //
            // The guard's false arm is UNREACHABLE here and is kept anyway: it sits
            // inside `projD < 0`, and `projD = dot(adjNormal, other - av0) < 0`
            // forces adjNormal to be non-zero.  A fixture cannot exercise it, which
            // is why it is written down instead.
            const float m2 = ptDot(adjNormal, adjNormal);
            if (m2 > 0.0f)
                adjNormal = adjNormal * (1.0f / sqrt(m2));

            const float proj = ptDot(adjNormal, triNormal);
            if (proj < 0.997f)
                triFlags |= (1u << (a + 3u));
        }
        else if (HTF_BOUNDARY_COLLISIONS)
        {
            triFlags |= (1u << (a + 3u));   // boundary edge, marked active
        }
        else
        {
            triFlags |= (1u << a);          // silhouette -- unreachable, see header
        }
    }

    return triFlags;
}

// selectNormal (capsuleTriangle.cuh:38-98) -- the only consumer of the flags.
//
// `true` means "use the TRIANGLE's normal"; `false` means the caller falls back to
// the segment-to-triangle direction.  The cascade classifies the barycentric pair
// (u, v) into a vertex, an edge or the face, and answers `true` when the feature it
// lands on has NO active convex edge -- i.e. when there is no neighbour steep
// enough for the triangle normal to be the wrong answer.
//
// THE ASYMMETRY IN THE `u > one` ARM IS UPSTREAM'S.  Every other arm handles both
// halves of its `v` split; this one tests `zero > v` (vertex 1) and has no `else`,
// so a point with u > 0.999999 and v >= 1e-6 -- geometrically outside the triangle
// past vertex 1 -- falls through to `return false`.  Transcribed as written.
public bool htfSelectNormal(float u, float v, uint triFlags)
{
    const float zero = 1e-6f;
    const float one = 0.999999f;

    if (zero > u)
    {
        if (zero > v)
        {
            // Vertex 0
            if ((triFlags & (ETD_CONVEX_EDGE_01 | ETD_CONVEX_EDGE_20)) == 0u)
                return true;
        }
        else if (v > one)
        {
            // Vertex 2
            if ((triFlags & (ETD_CONVEX_EDGE_12 | ETD_CONVEX_EDGE_20)) == 0u)
                return true;
        }
        else
        {
            // Edge 0-2
            if ((triFlags & ETD_CONVEX_EDGE_20) == 0u)
                return true;
        }
    }
    else if (u > one)
    {
        if (zero > v)
        {
            // Vertex 1
            if ((triFlags & (ETD_CONVEX_EDGE_01 | ETD_CONVEX_EDGE_12)) == 0u)
                return true;
        }
    }
    else
    {
        if (zero > v)
        {
            // Edge 0-1
            if ((triFlags & ETD_CONVEX_EDGE_01) == 0u)
                return true;
        }
        else
        {
            // THE THRESHOLD IS 0.9999f HERE, not the 0.999999f used for `one`
            // above.  Two different constants in the same function, four orders of
            // magnitude apart; both are transcribed as written.
            const float threshold = 0.9999f;
            const float temp = u + v;
            if (temp >= threshold)
            {
                // Edge 1-2
                if ((triFlags & ETD_CONVEX_EDGE_12) == 0u)
                    return true;
            }
            else
            {
                // Face
                return true;
            }
        }
    }
    return false;
}
