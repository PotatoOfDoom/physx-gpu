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

// pxgSlangTriTriOverlap -- P9.83.
//
//   The DECISION of triangleTriangleOverlaps (convexMeshMidphase.cu:1386-1528):
//   given the two meshes' root BV32 bounds, their scales and their world
//   transforms, does the pair's OBB-vs-OBB GJK come back non-DISTANT?
//
// ---------------------------------------------------------------------------
// WHY THIS IS A MODULE AND NOT A SECOND TRANSCRIPTION
// ---------------------------------------------------------------------------
// Phase 1 had already transcribed the whole of this kernel into
// npTriTriOverlap.slang and certified it bit-exactly against a CPU mirror
// (tests/testNpTriTriOverlap.cpp).  Writing the engine-ABI kernel as a fresh copy
// would have produced exactly the pair P9.15d cost a tranche to learn about: a
// fixture certifying one copy while the engine runs another.  So the decision
// moved HERE, and both the Phase-1 shader and triTriOverlapBda call it.
//
// The caller keeps only "where the bytes are":
//   - the Phase-1 shader reads the bounds out of a host-filled TtoIn struct;
//   - the engine kernel reads them out of the cooked blob's BV32 root node.
// Everything that DECIDES -- the masked min/max reduce, the corner bit pattern,
// the scaling, which cloud moves into whose space, the initial search direction,
// the convergence ratio and the DISTANT test -- is in this file, once.
//
// ---------------------------------------------------------------------------
// WHAT IS *NOT* HERE, DELIBERATELY
// ---------------------------------------------------------------------------
// The GJK itself.  `squawk::gjk` is one CUDA function shared by every
// deformable-vs-convex path, and pxgSlangGjk already carries it for exactly that
// reason.  npTriTriOverlap.slang used to carry its own five-hundred-line copy of
// the driver; P9.83 deleted that copy and re-pointed it at the module, so the
// tri-tri overlap test and every softbody/cloth contact gen now run the same
// simplex bookkeeping.  Its fixture re-ran bit-exactly across the change, which
// is the evidence that the two copies had not yet drifted -- not a reason to have
// kept them.
//
// ---------------------------------------------------------------------------
// THE CORNER BIT PATTERN IS INVERTED FROM THE OBVIOUS ONE
// ---------------------------------------------------------------------------
// cu:1479-1481 is
//     pos.x = (lane & 1) ? min.x : max.x
// -- a SET bit selects the MINIMUM.  Reading it the natural way round (set bit ->
// max) still produces the same eight corners as a SET, in a different order, so a
// box-vs-box overlap test cannot tell the difference: the GJK's support mapping
// is order-independent and the answer is identical for every input.  It is
// Class D -- observationally identical by accident -- and it only stops being so
// once a caller looks at a specific slot.  The fixture does (it compares the prep
// points slot by slot, `mmV`), which is why the order is pinned here.
module pxgSlangTriTriOverlap;

import pxgSlangNpAbi;   // npAbiQuatRotate / RotateInv / Mul -- PxQuat's exact term order
import pxgSlangGjk;     // GjkCloud, GjkOut, gjkRun, GJK_DISTANT

// gjk.cuh's caller passes 1 - 0.000225f; the literal is the CUDA's, not a
// rounded transcription of it.
public static const float TTO_CONVERGENCE_RATIO = 1.0f - 0.000225f;

// PxMeshScale::transform, the "multiply" form: rotate into the scale frame,
// scale, rotate back.  Same three lines as ctVertex2Shape / vertex2ShapeQ; this
// spelling composes npAbi's PxQuat expansion, which is the one the Phase-1
// fixture certified.
public float3 ttoVertex2Shape(float3 v, float3 scale, float4 rot)
{
    const float3 r = npAbiQuatRotate(rot, v);
    return npAbiQuatRotateInv(rot, float3(r.x * scale.x, r.y * scale.y, r.z * scale.z));
}

// PxVec3::getNormalized with the sqrt(dot()) spelling the engine uses -- NOT
// GLSL.std.450 Normalize, which check_length_intrinsic.py exists to catch.
public float3 ttoNormalized(float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f)
    {
        const float r = 1.0f / sqrt(m);
        return float3(v.x * r, v.y * r, v.z * r);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

// The root node's extent, reduced over its live child slots.
public struct TtoBox
{
    public float3 mn;
    public float3 mx;
}

// cu:1450-1470.  The CUDA gives one lane per child slot and reduces with
// minIndex/maxIndex over `mask = (1 << mNbNodes) - 1`; the winner INDEX is
// discarded, only the value is used.  Serialized that is a plain min/max fold
// over slots 0..mNbNodes-1, and min/max are associative and commutative over
// non-NaN floats, so the fold order cannot change the result.
//
// Slot 0 seeds the fold rather than +/-FLT_MAX.  A BV32 root always has at least
// one child (GuBV32Build writes mNbNodes >= 1 for any non-empty mesh, and a mesh
// with no triangles never reaches narrowphase), so the two spellings agree on
// every reachable input; seeding from slot 0 is the one the fixture certified and
// it also makes an mNbNodes of 0 produce a degenerate point rather than an
// inside-out box that would report every pair as overlapping.
public TtoBox ttoRootBox(float4 mins[32], float4 maxs[32], uint nbNodes)
{
    TtoBox b;
    b.mn = mins[0].xyz;
    b.mx = maxs[0].xyz;
    const uint n = min(nbNodes, 32u);
    for (uint s = 1u; s < n; ++s)
    {
        b.mn = min(b.mn, mins[s].xyz);
        b.mx = max(b.mx, maxs[s].xyz);
    }
    return b;
}

// The eight corners of one box, in the CUDA's slot order, scaled by that shape's
// PxMeshScale.  `toB` optionally carries the A-side into B space -- cu:1499-1503
// applies aToB to lanes 0..7 only.
public void ttoCorners(TtoBox b, float3 scale, float4 scaleRot, bool applyToB,
                       float4 toBq, float3 toBp, out float3 pts[8])
{
    for (uint i = 0u; i < 8u; ++i)
    {
        float3 pos;
        pos.x = ((i & 1u) != 0u) ? b.mn.x : b.mx.x;
        pos.y = ((i & 2u) != 0u) ? b.mn.y : b.mx.y;
        pos.z = ((i & 4u) != 0u) ? b.mn.z : b.mx.z;
        float3 v = ttoVertex2Shape(pos, scale, scaleRot);
        if (applyToB)
            v = npAbiQuatRotate(toBq, v) + toBp;
        pts[i] = v;
    }
}

// ---------------------------------------------------------------------------
// The whole decision (cu:1440-1527).
// ---------------------------------------------------------------------------
// `o` and the two clouds come back so a caller can check the prep slot by slot;
// the return value is the GjkResult the CUDA turns into the overlap byte.
public uint ttoOverlap(TtoBox b0, TtoBox b1,
                       float4 q0, float3 p0, float4 q1, float3 p1,
                       float3 scale0, float4 scaleRot0,
                       float3 scale1, float4 scaleRot1,
                       float contactDistance,
                       out GjkOut o, out float3 cloudA[8], out float3 cloudB[8])
{
    // aToB = transform1.transformInv(transform0): q = conj(q1)*q0,
    // p = rotInv(q1, p0 - p1).  Mesh 0 is the A side and moves into mesh 1's
    // space -- the OPPOSITE direction from ctCommon's convex-relative compose.
    const float4 aToBq = npAbiQuatMul(float4(-q1.x, -q1.y, -q1.z, q1.w), q0);
    const float3 aToBp = npAbiQuatRotateInv(q1, p0 - p1);

    ttoCorners(b0, scale0, scaleRot0, true, aToBq, aToBp, cloudA);
    ttoCorners(b1, scale1, scaleRot1, false, aToBq, aToBp, cloudB);

    // cu:1521-1523.  `sqDist > 0` and not `>= some epsilon`: a coincident pair
    // takes the +Y fallback, everything else normalizes.
    const float sqDist = aToBp.x * aToBp.x + aToBp.y * aToBp.y + aToBp.z * aToBp.z;
    const float3 initialDir = sqDist > 0.0f ? ttoNormalized(aToBp) : float3(0.0f, 1.0f, 0.0f);

    GjkCloud ca, cb;
    ca.n = 8u;
    cb.n = 8u;
    for (uint i = 0u; i < 8u; ++i)
    {
        ca.v[i] = cloudA[i];
        cb.v[i] = cloudB[i];
    }
    // GJK_MAX_CLOUD slots exist; only the first 8 are read because n == 8.  The
    // tail is left undefined on purpose -- zero-filling 56 unused float3s twice
    // per pair is pure cost, and gjkRun's support mapping never looks past n.

    float cachedSimplex[32];
    return gjkRun(ca, cb, initialDir, contactDistance, TTO_CONVERGENCE_RATIO, o, cachedSimplex);
}

// cu:1525.  `overlaps[i] = (result != GjkResult::eDISTANT)` -- CLOSE counts as an
// overlap, which is what makes contactDistance a real inflation of both boxes.
public bool ttoOverlapFlag(uint gjkResult) { return gjkResult != GJK_DISTANT; }
