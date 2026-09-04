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

// pxgSlangHeightfield.slang -- the heightfield sample format, the row/column
// rectangle, and the quad enumeration, shared by the BDA midphases (P9.27).
//
// WHY THIS EXISTS, AND WHY ITS BOUNDARY IS WHERE IT IS
//
// P9.17a set the rule for shared modules in this port: MEASURE the similarity,
// then share only what is genuinely the same.  Twenty-one BV32 traversals came
// out at 0.27 median, so only the node-format decode was lifted and every leaf
// predicate stayed with its caller.
//
// The same measurement was run here before writing a line of this file, over
// the two independent transcriptions of heightfieldComputePairs /
// heightfieldOutputPairs (heightfieldUtil.cuh:334, :382) that already existed:
//
//     count walk   fcMidphase (Phase 1) vs softBodyHfMidphaseBda (BDA)   0.90
//     emit  walk                                                        0.83
//
// and every residual difference is a SPELLING of the same operation -- a named
// `isMaterial1` bool against the same expression inline; `((v << 1) + 1) : (v << 1)`
// against `(v << 1) | (isMaterial1 ? 1 : 0)`, which agree because bit 0 of `v << 1`
// is zero; a `RWStructuredBuffer` index against a `Ptr<>` one, which is the axis
// this module parameterises.  That is a different finding from BV32's 0.27 and it
// justifies sharing the WALK, not merely the decode.
//
// (The first run of that measurement said 0.40.  Three defects in the measuring
// script -- wrapped lines compared as differing lines, a rename that left `(, x)`,
// and a start anchor whose `.` could not cross newlines, which made half the
// measurement fail silently and report as "could not extract" rather than as a
// broken tool.  A number from a tool nobody has read is not a measurement.)
//
// WHAT IS SHARED here is: the sample format; the WORLD-to-heightfield-units bound
// transform, which is character-for-character the same three lines in the cloth
// and softbody CUDA; the admit gate; the row/column clamps; the float->PxU32
// conversion; and the quad enumeration up to and including the surviving
// triangle's index.
//
// WHAT IS NOT is the ELEMENT's world bound -- a fattened triangle bound, a sphere
// bound around a vertex, a tetrahedron's bound -- and what the caller does with a
// live quad.  `psHfMidphaseBda` is the case that draws that line: it shares the
// enumeration exactly, and then fetches the triangle and runs a particle test
// instead of emitting a pair.  It is NOT moved onto this module in P9.27 and that
// is a scope decision, not an oversight: its walk carries a contact-accumulation
// concern of its own and a large fixture.  Named here so it is a task and not a
// silence.
//
// THE CONVERSION IS THE REASON THIS MODULE EXISTS AT ALL
//
// `heightfieldComputePairs` takes its y range as PxU32, so every caller performs
// an implicit float->unsigned conversion.  CUDA compiles it to cvt.rzi.u32.f32,
// which SATURATES: a negative arrives as 0.  The two transcriptions above
// disagreed about that for several tranches -- one saturating, one a bare cast
// with a header describing the C++ HOST's behaviour instead of the device's --
// and no fixture could tell, because every element in the one fixture that could
// have reached the branch sat at a positive local y.  P9.27a measured it, fixed
// it and pinned it.  A conversion that lives in one place cannot disagree with
// itself, which is the strongest argument for this file.
// ===========================================================================

module pxgSlangHeightfield;

import pxgSlangCommon;   // quatRotateInv

// PxHeightFieldMaterial::eHOLE (PxHeightFieldSample.h:49).
public static const uint HF_HOLE = 127u;

// ---------------------------------------------------------------------------
// PxHeightFieldSample, 4 bytes: PxI16 height @0, PxBitAndByte materialIndex0 @2,
// materialIndex1 @3.  One 32-bit load gives the whole sample.
// ---------------------------------------------------------------------------
[ForceInline]
public uint hfSampleWord(uint64_t samples, uint vertexIndex)
{
    return Ptr<uint>(samples + uint64_t(vertexIndex) * 4UL)[0];
}

// The height is SIGNED 16-bit; a raw mask would read -1 as 65535.  Shift up,
// then arithmetic-shift down.
[ForceInline]
public float hfHeight(uint64_t samples, uint vertexIndex)
{
    return float(int(hfSampleWord(samples, vertexIndex) << 16) >> 16);
}

// Both material bytes are PxBitAndByte, whose conversion operator returns
// `mData & ~bitMask` -- bit 7 (the tessellation flag) masked off.  Comparing the
// RAW byte against eHOLE is wrong: a tessellated hole would not match.
[ForceInline]
public uint hfMaterial0(uint64_t samples, uint vertexIndex)
{
    return (hfSampleWord(samples, vertexIndex) >> 16) & 0x7fu;
}
[ForceInline]
public uint hfMaterial1(uint64_t samples, uint vertexIndex)
{
    return (hfSampleWord(samples, vertexIndex) >> 24) & 0x7fu;
}

// ---------------------------------------------------------------------------
// CUDA's implicit float -> PxU32 at the heightfieldComputePairs call.
// cvt.rzi.u32.f32 truncates toward zero and SATURATES -- negatives to 0,
// overflow to 0xffffffff.
//
// Spelled out rather than written as a bare `uint(x)`, which is OpConvertFToU
// and UNDEFINED in SPIR-V for an out-of-range operand.  llvmpipe happens to
// return 0 and therefore happens to agree; agreement by accident is not
// agreement.  See P9.27a.
// ---------------------------------------------------------------------------
[ForceInline]
public uint hfF2uSat(float x)
{
    if (!(x > 0.0f)) return 0u;
    if (x >= 4294967296.0f) return 0xffffffffu;
    return uint(x);
}

// heightfieldUtil.cuh:37-58.  NOTE the two DIFFERENT upper clamps: the minimum
// clamps to n-2 so the quad's far corner still exists, the maximum to n-1.
[ForceInline]
public uint hfMinRow(float x, uint rows)    { return uint(clamp(int(floor(x)), 0, int(rows) - 2)); }
[ForceInline]
public uint hfMaxRow(float x, uint rows)    { return uint(clamp(int(ceil(x)),  0, int(rows) - 1)); }
[ForceInline]
public uint hfMinColumn(float z, uint cols) { return uint(clamp(int(floor(z)), 0, int(cols) - 2)); }
[ForceInline]
public uint hfMaxColumn(float z, uint cols) { return uint(clamp(int(ceil(z)),  0, int(cols) - 1)); }

// ---------------------------------------------------------------------------
// THE TRIANGLE, extracted from the samples (heightfieldUtil.cuh getTriangle).
//
// A quad's two triangles are cut along one of its diagonals, and WHICH diagonal
// is a per-sample flag -- bit 23, the tessellation bit that also happens to be
// materialIndex0's high bit.  That is why the material comparison has to mask it
// off and why reading the raw byte against eHOLE is wrong; the two facts are the
// same bit seen from two sides.
// ---------------------------------------------------------------------------
[ForceInline]
public bool hfZerothVertexShared(uint64_t samples, uint vertexIndex)
{
    return (hfSampleWord(samples, vertexIndex) & 0x800000u) != 0u;
}

public void hfTriVertexIndices(uint64_t samples, uint triangleIndex, uint columns,
                               out uint i0, out uint i1, out uint i2)
{
    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;
    if (hfZerothVertexShared(samples, cell))
    {
        if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1u; }
        else          { i0 = cell + 1u; i1 = cell + columns + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + columns; }
        else          { i0 = cell + columns + 1u; i1 = cell + columns; i2 = cell + 1u; }
    }
}

// getTriangleAdjacencyIndices (heightfieldUtil.cuh:167-250): the three
// edge-adjacent triangle indices, BOUNDARY where the edge is on the grid's rim.
//
// The four branches are NOT three-line variations on one formula and must not be
// folded into one: which neighbour lies across which edge depends both on the
// tessellation diagonal (isZerothVertexShared) and on which of the cell's two
// triangles this is, and the two guards differ between them -- `cell % columns`
// against 0 in one arm and against `columns - 2` in another, `cell / columns`
// against `rows - 2` in a third, `cell >= columns - 1` in a fourth.  Transcribed
// arm by arm from the .cu.
public static const uint HF_BOUNDARY = 0xFFFFFFFFu;

public void hfTriAdjacencyIndices(uint64_t samples, uint triangleIndex,
                                  uint rows, uint columns,
                                  out uint a0, out uint a1, out uint a2)
{
    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;

    a0 = HF_BOUNDARY;
    a2 = HF_BOUNDARY;

    if (hfZerothVertexShared(samples, cell))
    {
        if (firstTri)
        {
            a1 = triangleIndex + 1u;
            if (cell % columns != 0u)          a0 = triangleIndex - 1u;
            if (cell / columns != rows - 2u)   a2 = ((cell + columns) * 2u) + 1u;
        }
        else
        {
            a1 = triangleIndex - 1u;
            if (cell % columns < columns - 2u) a0 = triangleIndex + 1u;
            if (cell >= columns - 1u)          a2 = (cell - columns) * 2u;
        }
    }
    else
    {
        if (firstTri)
        {
            a1 = triangleIndex + 1u;
            if (cell >= columns - 1u)          a0 = ((cell - columns) * 2u) + 1u;
            if (cell % columns != 0u)          a2 = triangleIndex - 1u;
        }
        else
        {
            a1 = triangleIndex - 1u;
            if (cell / columns != rows - 2u)   a0 = (cell + columns) * 2u;
            if (cell % columns < columns - 2u) a2 = triangleIndex + 1u;
        }
    }
}

// A vertex in heightfield units: (row, height, column).
[ForceInline]
public float3 hfGetVertex(uint64_t samples, uint vertexIndex, uint columns)
{
    const uint row = vertexIndex / columns;
    const uint col = vertexIndex % columns;
    return float3(float(row), hfHeight(samples, vertexIndex), float(col));
}

[ForceInline]
public float3 hf2shapep(float3 v, float3 scale)
{
    return float3(v.x * scale.x, v.y * scale.y, v.z * scale.z);
}

// getTriangle, in the heightfield's SHAPE space (units multiplied by the scale).
//
// The handedness swap is DEAD UPSTREAM: PxHeightFieldGeometry::isValid rejects a
// non-positive row or column scale, so `wrongHanded` is always false in any scene
// the SDK will accept.  Transcribed anyway, because "unreachable today" is not
// the same as "not there", and the branch is one xor.
public void hfGetTriangle(uint64_t samples, uint triangleIndex, float3 scale, uint columns,
                          out float3 v0, out float3 v1, out float3 v2)
{
    bool wrongHanded = false;
    if (scale.z < 0.0f) wrongHanded = !wrongHanded;
    if (scale.x < 0.0f) wrongHanded = !wrongHanded;

    uint t0, t1, t2;
    hfTriVertexIndices(samples, triangleIndex, columns, t0, t1, t2);

    v0 = hf2shapep(hfGetVertex(samples, t0, columns), scale);
    v1 = hf2shapep(hfGetVertex(samples, wrongHanded ? t2 : t1, columns), scale);
    v2 = hf2shapep(hfGetVertex(samples, wrongHanded ? t1 : t2, columns), scale);
}

// The same getTriangle, with the ADJACENCY out-parameter the wrapper above drops
// (convexHeightfield.cu:83-116, the `adjacencyIndices != NULL` arm).
//
// This exists as its own function rather than as an extra out-parameter on
// hfGetTriangle because `wrongHanded` permutes BOTH results and must be decided
// once: the .cu computes it at the top of getTriangle and then applies it to the
// vertex indices as [0, 1+wh, 2-wh] and to the adjacency as [wh?2:0, 1, wh?0:2].
// Two callers each deriving their own copy of the same xor is how the two halves
// drift apart.
//
// NOTE THE TWO PERMUTATIONS ARE NOT THE SAME ONE.  The vertex permutation swaps
// slots 1 and 2; the adjacency permutation swaps slots 0 and 2.  They look alike
// written out and they are different, because an adjacency slot names the edge
// OPPOSITE a vertex, and reversing the winding maps edges the other way round.
public struct HfTriangle
{
    public float3 v0, v1, v2;
    public uint3 adj;
}

public HfTriangle hfGetTriangleAdj(uint64_t samples, uint triangleIndex, float3 scale,
                                   uint rows, uint columns)
{
    bool wrongHanded = false;
    if (scale.z < 0.0f) wrongHanded = !wrongHanded;
    if (scale.x < 0.0f) wrongHanded = !wrongHanded;

    uint t0, t1, t2;
    hfTriVertexIndices(samples, triangleIndex, columns, t0, t1, t2);

    uint a0, a1, a2;
    hfTriAdjacencyIndices(samples, triangleIndex, rows, columns, a0, a1, a2);

    HfTriangle t;
    t.v0 = hf2shapep(hfGetVertex(samples, t0, columns), scale);
    t.v1 = hf2shapep(hfGetVertex(samples, wrongHanded ? t2 : t1, columns), scale);
    t.v2 = hf2shapep(hfGetVertex(samples, wrongHanded ? t1 : t2, columns), scale);
    t.adj = uint3(wrongHanded ? a2 : a0, a1, wrongHanded ? a0 : a2);
    return t;
}

// ---------------------------------------------------------------------------
// hfLocalBound -- a WORLD bound into heightfield ROW/HEIGHT/COLUMN units.
//
// Upstream is `PxBounds3::transformFast(hfTransform.getInverse(), bound)` followed
// by a per-axis division.  transformFast is basisExtent(inv.transform(centre),
// PxMat33(inv.q), extents), and with inv.q = conj(hfQ) that is
// hfQ.rotateInv(centre - hfP) for the centre and hfQ.rotateInv(e_i) for the basis
// columns -- so the extent is the ABS-SUM of the three scaled columns.
//
// The height axis divides by the SIGNED scale and the row/column axes by |scale|,
// which is upstream's asymmetry, not a transcription slip (femClothHFMidPhase.cu:200
// and softbodyHFMidPhase.cu:200 -- character for character the same three lines).
// The negative row/column scale then MIRRORS the axis, so min and max swap.
//
// THIS IS THE CHAIN THAT NEEDS -ffp-contract=off.  The three scaled basis columns
// and the quaternion rotations feeding them are mul-then-add chains; fusing one to
// an FMA moves a bound by an ULP, which is enough to change a floor() or a ceil()
// and therefore the emitted pair SET, not merely its arithmetic.  Every importer
// of this module belongs in PXG_NOCONTRACT_SHADERS for that reason.
// ---------------------------------------------------------------------------
public void hfLocalBound(float4 hfQ, float3 hfP, float3 hfScale,
                         float3 worldMin, float3 worldMax,
                         out float3 lMin, out float3 lMax)
{
    const float oneOverHeightScale = 1.0f / hfScale.y;
    const float oneOverRowScale = 1.0f / abs(hfScale.x);
    const float oneOverColScale = 1.0f / abs(hfScale.z);

    const float3 centre = (worldMin + worldMax) * 0.5f;
    const float3 extents = (worldMax - worldMin) * 0.5f;
    const float3 lc = quatRotateInv(hfQ, centre - hfP);
    const float3 c0 = quatRotateInv(hfQ, float3(1, 0, 0)) * extents.x;
    const float3 c1 = quatRotateInv(hfQ, float3(0, 1, 0)) * extents.y;
    const float3 c2 = quatRotateInv(hfQ, float3(0, 0, 1)) * extents.z;
    const float3 w = abs(c0) + abs(c1) + abs(c2);
    lMin = lc - w;
    lMax = lc + w;

    lMin.x *= oneOverRowScale; lMin.y *= oneOverHeightScale; lMin.z *= oneOverColScale;
    lMax.x *= oneOverRowScale; lMax.y *= oneOverHeightScale; lMax.z *= oneOverColScale;

    if (hfScale.x < 0.0f) { const float t = lMin.x; lMin.x = lMax.x; lMax.x = t; }
    if (hfScale.z < 0.0f) { const float t = lMin.z; lMin.z = lMax.z; lMax.z = t; }
}

// ---------------------------------------------------------------------------
// The rectangle of quads an element's LOCAL bound covers, plus the y range the
// enumeration compares against.  `live` folds both upstream gates.
// ---------------------------------------------------------------------------
public struct HfRect
{
    public uint minRow, maxRow, minColumn, maxColumn;
    public uint nbCols;
    public float minyF, maxyF;   // float(hfF2uSat(local y)), as the predicates see it
    public bool live;
}

// hfMakeRect -- the admit gate, the clamps and the y conversion, in one place so
// that no caller can perform two of the three and forget the last.
//
// THE ADMIT GATE IS TRANSCRIBED WITH UPSTREAM'S OWN BRACKETING:
//
//     if (!(min.x > nbRows - 1) || (min.z > nbCols - 1) || (max.x < 0) || (max.z < 0))
//
// The `!` binds to the FIRST comparison only, so the three that follow are ORed
// in UNNEGATED and the gate also admits bounds entirely off the grid in z, or
// left of it.  The clamped rectangle then produces a zero-area span and zero
// pairs, which keeps the quirk output-neutral for those cases -- but it is
// preserved exactly, because "output-neutral in the cases we tried" is not a
// licence to rewrite someone else's condition.
// (femClothHFMidPhase.cu:140, softbodyHFMidPhase.cu:236 -- identical in both.)
public HfRect hfMakeRect(float3 lMin, float3 lMax, uint nbRows, uint nbCols)
{
    HfRect r;
    r.minRow = 0u; r.maxRow = 0u; r.minColumn = 0u; r.maxColumn = 0u;
    r.nbCols = nbCols;
    r.minyF = 0.0f; r.maxyF = 0.0f;
    r.live = false;

    if (!(lMin.x > float(nbRows) - 1.0f) || (lMin.z > float(nbCols) - 1.0f)
        || (lMax.x < 0.0f) || (lMax.z < 0.0f))
    {
        r.minRow = hfMinRow(lMin.x, nbRows);
        r.maxRow = hfMaxRow(lMax.x, nbRows);
        r.minColumn = hfMinColumn(lMin.z, nbCols);
        r.maxColumn = hfMaxColumn(lMax.z, nbCols);

        // The area gate, upstream's spelling: 2 * span * span > 0.
        if ((2u * (r.maxColumn - r.minColumn) * (r.maxRow - r.minRow)) > 0u)
        {
            r.minyF = float(hfF2uSat(lMin.y));
            r.maxyF = float(hfF2uSat(lMax.y));
            r.live = true;
        }
    }
    return r;
}

// Each covered quad contributes TWO candidates, one per material slot.
[ForceInline]
public uint hfQuadSpan(HfRect r)
{
    return r.live ? ((r.maxRow - r.minRow) * (r.maxColumn - r.minColumn) * 2u) : 0u;
}

public struct HfQuad
{
    public bool survives;
    public uint vertexIndex;
    public uint triangleIdx;
}

// hfQuadAt -- candidate `i` of the rectangle, decoded and culled.
//
// SAFE TO CALL ONLY FOR i < hfQuadSpan(r), and that is not a nicety: the decode
// divides by `columnSpan`.  hfQuadSpan returns 0 unless `live`, and `live`
// requires the area gate `2 * columnSpan * rowSpan > 0`, so a caller that bounds
// its loop by hfQuadSpan can never reach a zero divisor.  The two functions are
// a pair; using the second without the first is a division by zero waiting for
// an empty rectangle.
//
// A quad is skipped when the element's y range lies entirely ABOVE all four
// corner heights (con0) or entirely BELOW them (con1), and a candidate is
// skipped when its material slot is eHOLE.  The surviving triangle's index is
// (vertexIndex << 1) + isMaterial1.
public HfQuad hfQuadAt(uint64_t samples, HfRect r, uint i)
{
    HfQuad q;
    q.survives = false;
    q.vertexIndex = 0u;
    q.triangleIdx = 0u;

    const uint columnSpan = r.maxColumn - r.minColumn;
    const uint index = i / 2u;
    const uint vertexIndex = (r.minRow + index / columnSpan) * r.nbCols
                           + (r.minColumn + index % columnSpan);
    q.vertexIndex = vertexIndex;

    const float h0 = hfHeight(samples, vertexIndex);
    const float h1 = hfHeight(samples, vertexIndex + 1u);
    const float h2 = hfHeight(samples, vertexIndex + r.nbCols);
    const float h3 = hfHeight(samples, vertexIndex + r.nbCols + 1u);
    const bool con0 = r.maxyF < h0 && r.maxyF < h1 && r.maxyF < h2 && r.maxyF < h3;
    const bool con1 = r.minyF > h0 && r.minyF > h1 && r.minyF > h2 && r.minyF > h3;

    if (!(con0 || con1))
    {
        const bool isMaterial1 = (i & 1u) != 0u;
        const uint material = isMaterial1 ? hfMaterial1(samples, vertexIndex)
                                          : hfMaterial0(samples, vertexIndex);
        if (material != HF_HOLE)
        {
            q.survives = true;
            q.triangleIdx = (vertexIndex << 1) + (isMaterial1 ? 1u : 0u);
        }
    }
    return q;
}
