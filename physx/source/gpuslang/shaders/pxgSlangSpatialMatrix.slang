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

// pxgSlangSpatialMatrix.slang
//
// P8.86: the spatial-matrix algebra layer.  `Dy::SpatialMatrix` and the PxMat33
// arithmetic it is built from, ported once, as a module, with its own fixture
// (tests/testArtiSpatialMatrix.cpp).  Everything left in the articulation
// solver -- computeIs, computePropagateSpatialInertia_ZA_ZIc,
// computeArticulatedSpatialInertiaW, the impulse-response chain -- is built on
// this, so it goes in here rather than being transcribed per kernel.
//
// ===========================================================================
// THE SHAPE, AND WHY IT IS SEVEN 3x3 BLOCKS AND NOT NINE
// ===========================================================================
//
// `Dy::SpatialMatrix` (DyFeatherstoneArticulationUtils.h:220) is a 6x6 stored as
// THREE PxMat33 blocks -- `topLeft`, `topRight`, `bottomLeft`.  The fourth is
// implied: `getBottomRight()` returns `topLeft.getTranspose()` (:246).  That is
// not a storage trick, it is the structure of an articulated inertia in the
// dual basis, and it is the reason PxgSpatialMatrixBlock is seven float4 columns
// rather than nine.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// Every operation below has a PX_CUDA_CALLABLE original the fixture calls on the
// host, so the expectation is the engine's own arithmetic and not a second
// transcription:
//
//   PxMat33 + - * (scalar, matrix, vector), getTranspose, transformTranspose,
//                                getInverse, getDeterminant  -- PxMat33.h
//   SpatialMatrix * + - , getBottomRight, getTranspose,
//                                constructSpatialMatrix       -- Dy...Utils.h:239+
//   SpatialMatrix::invertSym33, invertInertia                 -- :488 / :535
//   Cm::UnAlignedSpatialVector::innerProduct, dot, cross      -- CmSpatialVector.h
//
// Two are `static __device__` in forwardDynamic2.cu and cannot be called from
// the host -- `translateInertia` (:1546) and `constructSkewSymmetricMatrix`
// (:1561).  Their host mirrors are built out of the engine's own PxMat33
// operators, so only the five-line structure is transcribed; the tranche note
// says so rather than presenting the comparison as a full oracle.
//
// ===========================================================================
// FIVE TRAPS, EACH OF WHICH GETS A MUTANT
// ===========================================================================
//
//  1. `SpatialMatrix * v` uses `topLeft.transformTranspose(v.bottom)` for the
//     BOTTOM half, not `topLeft * v.bottom`.  The two agree only when topLeft is
//     symmetric, which an articulated inertia's is not.
//  2. `invertSym33` returns the IDENTITY on a zero determinant, not zero -- and
//     a fixed base reaches that case in a real scene.
//  3. `invertSym33` is NOT a general inverse.  It builds its result from
//     `PxVec3(v0.y, v1.y, v1.z)` etc., exploiting symmetry; on a non-symmetric
//     input it disagrees with `PxMat33::getInverse`, which is what the SPHERICAL
//     joint case uses.  Both exist here and they are not interchangeable.
//  4. `invertInertia` symmetrises `aa` and `ll` but NOT `la`.
//  5. `translateInertia` computes `bl` and `br` from the OLD `topLeft` /
//     `topRight` and only then overwrites `topLeft`; it symmetrises
//     `bottomLeft` at the end and never touches `topRight`.  Reordering it is
//     the obvious tidy and it is wrong.
//
// And one that is not a matrix at all: `innerProduct` is the DUAL product,
// `bottom.dot(v.top) + top.dot(v.bottom)` -- not `dot`, which is
// `top.dot(v.top) + bottom.dot(v.bottom)`.  Both are in this module because the
// solver uses both, on the same type, three lines apart.
//
// ===========================================================================
// COLUMNS, NOT ROWS
// ===========================================================================
//
// `AbMat33` (pxgSlangArticulationBlocks.slang) carries three COLUMNS because
// `PxMat33::operator[](i)` returns column i while Slang's `float3x3` indexes
// rows first.  Every expression below is spelled the engine's way:
//
//   transform(v)          = c0 * v.x + c1 * v.y + c2 * v.z        (PxMat33.h:344)
//   transformTranspose(v) = (c0.dot(v), c1.dot(v), c2.dot(v))     (PxMat33.h:350)
//   a * b                 = (a.transform(b.c0), a.transform(b.c1),
//                            a.transform(b.c2))                   (PxMat33.h:286)
//
// The addition order inside `transform` is load-bearing for bit-exactness: it is
// left to right, and any importer of this module belongs in
// PXG_NOCONTRACT_SHADERS for the reason P8.69 gives.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;

// ---------------------------------------------------------------------------
// The two aggregates
// ---------------------------------------------------------------------------

// Cm::UnAlignedSpatialVector.  What (top, bottom) MEANS depends on whether the
// vector is a motion or a force -- see npArtiSpatialInertia.slang's header.  The
// algebra here does not care and must not assume.
public struct AbSpatialVector
{
    public float3 top;
    public float3 bottom;
};

public AbSpatialVector asmSVec(float3 top, float3 bottom)
{
    AbSpatialVector v;
    v.top = top;
    v.bottom = bottom;
    return v;
}

// Dy::SpatialMatrix.  bottomRight is NOT stored; it is topLeft transposed.
public struct AbSpatialMatrix
{
    public AbMat33 topLeft;
    public AbMat33 topRight;
    public AbMat33 bottomLeft;
};

public AbSpatialMatrix asmMatrix(AbMat33 topLeft, AbMat33 topRight, AbMat33 bottomLeft)
{
    AbSpatialMatrix m;
    m.topLeft = topLeft;
    m.topRight = topRight;
    m.bottomLeft = bottomLeft;
    return m;
}

// SpatialMatrix(PxZero) / setZero()
public AbSpatialMatrix asmMatrixZero()
{
    return asmMatrix(abMat33Zero(), abMat33Zero(), abMat33Zero());
}

// ---------------------------------------------------------------------------
// PxVec3, spelled PhysX's way
// ---------------------------------------------------------------------------

// PxVec3::cross (PxVec3.h:280)
public float3 asmCross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

// PxVec3::dot (PxVec3.h:272) -- x*v.x + y*v.y + z*v.z, in that order
public float asmDot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ---------------------------------------------------------------------------
// PxMat33
// ---------------------------------------------------------------------------

public AbMat33 asmM33(float3 c0, float3 c1, float3 c2)
{
    AbMat33 m;
    m.c0 = c0;
    m.c1 = c1;
    m.c2 = c2;
    return m;
}

public AbMat33 asmM33Identity()
{
    return asmM33(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
}

public AbMat33 asmM33Add(AbMat33 a, AbMat33 b)
{
    return asmM33(a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2);
}

public AbMat33 asmM33Sub(AbMat33 a, AbMat33 b)
{
    return asmM33(a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2);
}

public AbMat33 asmM33Neg(AbMat33 a)
{
    return asmM33(-a.c0, -a.c1, -a.c2);
}

public AbMat33 asmM33Scale(AbMat33 a, float s)
{
    return asmM33(a.c0 * s, a.c1 * s, a.c2 * s);
}

// PxMat33::transform / operator*(PxVec3) -- c0 * v.x + c1 * v.y + c2 * v.z
public float3 asmM33Transform(AbMat33 m, float3 v)
{
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z;
}

// PxMat33::transformTranspose -- (c0.dot(v), c1.dot(v), c2.dot(v))
public float3 asmM33TransformTranspose(AbMat33 m, float3 v)
{
    return float3(asmDot(m.c0, v), asmDot(m.c1, v), asmDot(m.c2, v));
}

// PxMat33::operator*(PxMat33) -- the OTHER matrix's columns are transformed, so
// this is `a` applied to each column of `b`, not the reverse.
public AbMat33 asmM33Mul(AbMat33 a, AbMat33 b)
{
    return asmM33(asmM33Transform(a, b.c0), asmM33Transform(a, b.c1), asmM33Transform(a, b.c2));
}

// PxMat33::getTranspose (PxMat33.h:205)
public AbMat33 asmM33Transpose(AbMat33 m)
{
    return asmM33(float3(m.c0.x, m.c1.x, m.c2.x),
                  float3(m.c0.y, m.c1.y, m.c2.y),
                  float3(m.c0.z, m.c1.z, m.c2.z));
}

// PxMat33::getDeterminant -- column0.dot(column1.cross(column2))
public float asmM33Determinant(AbMat33 m)
{
    return asmDot(m.c0, asmCross(m.c1, m.c2));
}

// PxMat33::getInverse (PxMat33.h:215): the GENERAL inverse, cofactors written
// out one entry at a time, and the IDENTITY on a zero determinant.  This is what
// the spherical-joint path uses; it is NOT interchangeable with invertSym33
// below, which is only an inverse for a symmetric input.
public AbMat33 asmM33Inverse(AbMat33 m)
{
    const float det = asmM33Determinant(m);
    if (det != 0.0f)
    {
        const float invDet = 1.0f / det;

        AbMat33 inv;
        inv.c0 = float3(invDet * (m.c1.y * m.c2.z - m.c2.y * m.c1.z),
                        invDet * -(m.c0.y * m.c2.z - m.c2.y * m.c0.z),
                        invDet * (m.c0.y * m.c1.z - m.c0.z * m.c1.y));
        inv.c1 = float3(invDet * -(m.c1.x * m.c2.z - m.c1.z * m.c2.x),
                        invDet * (m.c0.x * m.c2.z - m.c0.z * m.c2.x),
                        invDet * -(m.c0.x * m.c1.z - m.c0.z * m.c1.x));
        inv.c2 = float3(invDet * (m.c1.x * m.c2.y - m.c1.y * m.c2.x),
                        invDet * -(m.c0.x * m.c2.y - m.c0.y * m.c2.x),
                        invDet * (m.c0.x * m.c1.y - m.c1.x * m.c0.y));
        return inv;
    }
    return asmM33Identity();
}

// forwardDynamic2.cu:1561.  A permutation of r's components with three sign
// flips -- no arithmetic, so nothing to contract and nothing to round.  `r` is a
// float4 in the engine because the caller has one; w is ignored.
public AbMat33 asmSkewSymmetric(float4 r)
{
    return asmM33(float3(0.0f, r.z, -r.y),
                  float3(-r.z, 0.0f, r.x),
                  float3(r.y, -r.x, 0.0f));
}

// SpatialMatrix::invertSym33 (:488).  TWO traps in eleven lines:
//   * it returns the IDENTITY, not zero, when the determinant is zero;
//   * it is NOT a general inverse -- the second and third result columns are
//     assembled from components of v0 and v1 on the assumption that the input is
//     symmetric.  Feed it a non-symmetric matrix and it quietly returns
//     something that is not an inverse of anything, which is exactly why the
//     spherical case calls PxMat33::getInverse instead.
public AbMat33 asmInvertSym33(AbMat33 m)
{
    const float3 v0 = asmCross(m.c1, m.c2);
    const float3 v1 = asmCross(m.c2, m.c0);
    const float3 v2 = asmCross(m.c0, m.c1);

    const float det = asmDot(v0, m.c0);

    if (det != 0.0f)
    {
        const float recipDet = 1.0f / det;
        return asmM33(v0 * recipDet,
                      float3(v0.y, v1.y, v1.z) * recipDet,
                      float3(v0.z, v1.z, v2.z) * recipDet);
    }
    return asmM33Identity();
}

// ---------------------------------------------------------------------------
// Dy::SpatialMatrix
// ---------------------------------------------------------------------------

// getBottomRight (:246) -- the fourth block, which is not stored
public AbMat33 asmBottomRight(AbSpatialMatrix m)
{
    return asmM33Transpose(m.topLeft);
}

public AbSpatialMatrix asmAdd(AbSpatialMatrix a, AbSpatialMatrix b)
{
    return asmMatrix(asmM33Add(a.topLeft, b.topLeft),
                     asmM33Add(a.topRight, b.topRight),
                     asmM33Add(a.bottomLeft, b.bottomLeft));
}

public AbSpatialMatrix asmSub(AbSpatialMatrix a, AbSpatialMatrix b)
{
    return asmMatrix(asmM33Sub(a.topLeft, b.topLeft),
                     asmM33Sub(a.topRight, b.topRight),
                     asmM33Sub(a.bottomLeft, b.bottomLeft));
}

public AbSpatialMatrix asmNeg(AbSpatialMatrix a)
{
    return asmMatrix(asmM33Neg(a.topLeft), asmM33Neg(a.topRight), asmM33Neg(a.bottomLeft));
}

public AbSpatialMatrix asmScale(AbSpatialMatrix a, float s)
{
    return asmMatrix(asmM33Scale(a.topLeft, s),
                     asmM33Scale(a.topRight, s),
                     asmM33Scale(a.bottomLeft, s));
}

// SpatialMatrix::getTranspose (:398): topRight and bottomLeft SWAP as well as
// transposing.  A port that transposes the three blocks in place is wrong.
public AbSpatialMatrix asmTranspose(AbSpatialMatrix a)
{
    return asmMatrix(asmM33Transpose(a.topLeft),
                     asmM33Transpose(a.bottomLeft),
                     asmM33Transpose(a.topRight));
}

// SpatialMatrix::operator*(SpatialMatrix) (:330).  Both implied bottom-right
// blocks have to be materialised first -- this one's and the operand's.
public AbSpatialMatrix asmMulMatrix(AbSpatialMatrix a, AbSpatialMatrix b)
{
    const AbMat33 sBottomRight = asmM33Transpose(b.topLeft);
    const AbMat33 bottomRight = asmM33Transpose(a.topLeft);

    const AbMat33 newTopLeft = asmM33Add(asmM33Mul(a.topLeft, b.topLeft),
                                         asmM33Mul(a.topRight, b.bottomLeft));
    const AbMat33 newTopRight = asmM33Add(asmM33Mul(a.topLeft, b.topRight),
                                          asmM33Mul(a.topRight, sBottomRight));
    const AbMat33 newBottomLeft = asmM33Add(asmM33Mul(a.bottomLeft, b.topLeft),
                                            asmM33Mul(bottomRight, b.bottomLeft));

    return asmMatrix(newTopLeft, newTopRight, newBottomLeft);
}

// SpatialMatrix::operator*(Cm::UnAlignedSpatialVector) (:276).  TRAP 1: the
// bottom half uses topLeft TRANSPOSED, which is the implied bottom-right block,
// not topLeft and not bottomLeft.
public AbSpatialVector asmMulVec(AbSpatialMatrix m, AbSpatialVector s)
{
    const float3 top = asmM33Transform(m.topLeft, s.top) + asmM33Transform(m.topRight, s.bottom);
    const float3 bottom = asmM33Transform(m.bottomLeft, s.top)
                        + asmM33TransformTranspose(m.topLeft, s.bottom);
    return asmSVec(top, bottom);
}

// SpatialMatrix::constructSpatialMatrix(Is, stI) (:364) -- the outer product
// of two spatial vectors.  Note which halves pair with which: topLeft is
// Is.top (x) stI.top, topRight is Is.top (x) stI.bottom, bottomLeft is
// Is.bottom (x) stI.top -- and Is.bottom (x) stI.bottom, which would be the
// bottom-right block, is NOT formed, because it is topLeft's transpose only when
// the caller's Is/stI pair makes it so.  The engine does not check; neither does
// this.
public AbSpatialMatrix asmConstructSpatialMatrix(AbSpatialVector Is, AbSpatialVector stI)
{
    const AbMat33 topLeft = asmM33(Is.top * stI.top.x, Is.top * stI.top.y, Is.top * stI.top.z);
    const AbMat33 topRight = asmM33(Is.top * stI.bottom.x, Is.top * stI.bottom.y,
                                    Is.top * stI.bottom.z);
    const AbMat33 bottomLeft = asmM33(Is.bottom * stI.top.x, Is.bottom * stI.top.y,
                                      Is.bottom * stI.top.z);
    return asmMatrix(topLeft, topRight, bottomLeft);
}

// SpatialMatrix::constructSpatialMatrix(columns) (:389), the six-column form.
// The BOTTOM halves of columns 3, 4 and 5 are DISCARDED -- they would be the
// bottom-right block, which is implied.  A port that stores them somewhere is
// wrong, and a fixture whose columns 3..5 have zero bottoms cannot tell.
public AbSpatialMatrix asmConstructFromColumns(AbSpatialVector columns[6])
{
    const AbMat33 topLeft = asmM33(columns[0].top, columns[1].top, columns[2].top);
    const AbMat33 bottomLeft = asmM33(columns[0].bottom, columns[1].bottom, columns[2].bottom);
    const AbMat33 topRight = asmM33(columns[3].top, columns[4].top, columns[5].top);
    return asmMatrix(topLeft, topRight, bottomLeft);
}

// SpatialMatrix::invertInertia (:535).  TRAP 4: `aa` and `ll` are symmetrised,
// `la` is NOT.  And the result's blocks are permuted -- topLeft is LA
// TRANSPOSED, topRight is AA, bottomLeft is LL.
public AbSpatialMatrix asmInvertInertia(AbSpatialMatrix m)
{
    AbMat33 aa = m.bottomLeft;
    AbMat33 ll = m.topRight;
    const AbMat33 la = m.topLeft;

    aa = asmM33Scale(asmM33Add(aa, asmM33Transpose(aa)), 0.5f);
    ll = asmM33Scale(asmM33Add(ll, asmM33Transpose(ll)), 0.5f);

    const AbMat33 AAInv = asmInvertSym33(aa);

    const AbMat33 z = asmM33Mul(asmM33Neg(la), AAInv);
    const AbMat33 S = asmM33Add(ll, asmM33Mul(z, asmM33Transpose(la)));   // Schur complement

    const AbMat33 LL = asmInvertSym33(S);

    const AbMat33 LA = asmM33Mul(LL, z);
    const AbMat33 AA = asmM33Add(AAInv, asmM33Mul(asmM33Transpose(z), LA));

    return asmMatrix(asmM33Transpose(LA), AA, LL);
}

// translateInertia (forwardDynamic2.cu:1546).  TRAP 5, in three parts:
//   * `bl` and `br` are formed from the OLD topLeft / topRight, before topLeft
//     is overwritten -- so the two assignments cannot be swapped;
//   * `br` needs the implied bottom-right block, i.e. the old topLeft
//     TRANSPOSED, not the new one;
//   * topRight is never written, and bottomLeft is symmetrised at the end.
public void asmTranslateInertia(AbMat33 sTod, inout AbSpatialMatrix inertia)
{
    const AbMat33 dTos = asmM33Transpose(sTod);

    const AbMat33 bl = asmM33Add(asmM33Mul(sTod, inertia.topLeft), inertia.bottomLeft);
    const AbMat33 br = asmM33Add(asmM33Mul(sTod, inertia.topRight), asmBottomRight(inertia));

    inertia.topLeft = asmM33Add(inertia.topLeft, asmM33Mul(inertia.topRight, dTos));
    inertia.bottomLeft = asmM33Add(bl, asmM33Mul(br, dTos));

    // "aligned inertia - make it symmetrical! OPTIONAL!!!!" -- the engine's own
    // comment.  It is not optional for bit-exactness.
    inertia.bottomLeft = asmM33Scale(asmM33Add(inertia.bottomLeft,
                                               asmM33Transpose(inertia.bottomLeft)), 0.5f);
}

// The OTHER translateInertia (forwardDynamic2.cu:1569), and it is a different
// function from the one above despite the shared name: this one is the parallel
// axis theorem on a plain 3x3, `I + m * s(t)^T s(t)`, and it RETURNS rather than
// mutating.  `computeLinkAcceleration`'s compound inertia is built out of it.
//
// Two things transcribed rather than tidied:
//   * `s` is the skew of `t` with the SAME sign convention as
//     `constructSkewSymmetricMatrix`, so `s^T s` is the symmetric positive
//     semi-definite `|t|^2 I - t t^T` -- but it is spelled as a product of two
//     matrices, and the port keeps that spelling because the fixture compares bit
//     patterns;
//   * the association is `((s^T * s) * mass) + inertia`, left to right.  Folding
//     the mass into either factor first changes the last bit.
//
// What this call site CANNOT check, measured rather than assumed: `s` is skew, so
// `s^T == -s`, and therefore `s^T s`, `s s^T` and `(-s)^T (-s)` are all the same
// sums of the same products in the same order — bit-identical, since `(-a)*(-b)`
// is exactly `a*b` in IEEE.  The sweep confirms it: both substitutions survive.
// The skew's SIGN CONVENTION is thus unverifiable here, which is precisely why the
// port keeps the engine's spelling instead of "simplifying" to `|t|^2 I - t t^T`
// — the convention is load-bearing at `asmSkewSymmetric`'s other call sites, and
// this one must not be allowed to drift away from it.
public AbMat33 asmTranslateInertiaMass(AbMat33 inertia, float mass, float3 t)
{
    const AbMat33 s = asmM33(float3(0.0f, t.z, -t.y),
                             float3(-t.z, 0.0f, t.x),
                             float3(t.y, -t.x, 0.0f));

    return asmM33Add(asmM33Scale(asmM33Mul(asmM33Transpose(s), s), mass), inertia);
}

// ---------------------------------------------------------------------------
// PxSpatialMatrix (the full 6x6) times a spatial vector
// ---------------------------------------------------------------------------
//
// PxSpatialMatrix.h:58.  Note the indexing: `result[i] += column[j][i] * st[j]`
// -- the OUTER index walks the result and the INNER one walks `column`'s first
// subscript, so despite the field being called `column` this contracts over the
// first subscript.  Feeding it `column[i][j]` transposes the matrix, and for the
// impulse-response matrices that is a plausible-looking wrong answer rather than
// an obvious one.
//
// The accumulator starts at a LITERAL zero and is built with `+=` over six terms,
// which is P8.87's driver-fold shape: if every one of the six products is a
// negative zero the engine's C++ yields `+0` and the folded form yields `-0`.
// That needs an entirely zero result row against an all-negative input, which the
// fixture builds on purpose -- see testArtiSpatialMatrix.cpp's lane 3.
public AbSpatialVector asmM6MulVec(AbMatrix6x6 m, AbSpatialVector s)
{
    float st[6];
    st[0] = s.top.x;    st[1] = s.top.y;    st[2] = s.top.z;
    st[3] = s.bottom.x; st[4] = s.bottom.y; st[5] = s.bottom.z;

    float result[6];
    for (uint i = 0u; i < 6u; ++i)
    {
        result[i] = 0.0f;
        for (uint j = 0u; j < 6u; ++j)
            result[i] = result[i] + abM6Get(m, j, i) * st[j];
    }

    return asmSVec(float3(result[0], result[1], result[2]),
                   float3(result[3], result[4], result[5]));
}

// ---------------------------------------------------------------------------
// Cm::UnAlignedSpatialVector
// ---------------------------------------------------------------------------

public AbSpatialVector asmSVecAdd(AbSpatialVector a, AbSpatialVector b)
{
    return asmSVec(a.top + b.top, a.bottom + b.bottom);
}

public AbSpatialVector asmSVecSub(AbSpatialVector a, AbSpatialVector b)
{
    return asmSVec(a.top - b.top, a.bottom - b.bottom);
}

public AbSpatialVector asmSVecScale(AbSpatialVector a, float s)
{
    return asmSVec(a.top * s, a.bottom * s);
}

// THE DUAL PRODUCT (CmSpatialVector.h:398): bottom against top and top against
// bottom.  This is the one that pairs a motion with a force and yields power;
// `asmSVecDot` below is the Euclidean one.  They are three lines apart in the
// engine header and the solver uses both.
public float asmInnerProduct(AbSpatialVector a, AbSpatialVector b)
{
    return asmDot(a.bottom, b.top) + asmDot(a.top, b.bottom);
}

// UnAlignedSpatialVector::dot (:408) -- top against top, bottom against bottom
public float asmSVecDot(AbSpatialVector a, AbSpatialVector b)
{
    return asmDot(a.top, b.top) + asmDot(a.bottom, b.bottom);
}

// UnAlignedSpatialVector::cross (:418).  The bottom half takes BOTH cross
// products; the top half takes only one.
public AbSpatialVector asmSVecCross(AbSpatialVector a, AbSpatialVector b)
{
    return asmSVec(asmCross(a.top, b.top),
                   asmCross(a.top, b.bottom) + asmCross(a.bottom, b.top));
}

// translateSpatialVectorInPlace (forwardDynamic2.cu:57): the bottom half picks
// up `offset x top`, the top half is untouched.  Shifting a spatial vector to a
// different reference point.
public void asmTranslateSpatialVectorInPlace(float3 offset, inout AbSpatialVector v)
{
    v.bottom = v.bottom + asmCross(offset, v.top);
}

// ---------------------------------------------------------------------------
// The block load, paired with abStoreSpatialMatrix
// ---------------------------------------------------------------------------
//
// abLoadSpatialMatrix (pxgSlangArticulationBlocks.slang) mirrors the engine's
// loadSpatialMatrix (PxgArticulation.h:90) and hands back three AbMat33s; this
// is the one-line wrapper that assembles them, so call sites in the solver deal
// in AbSpatialMatrix throughout.
public AbSpatialMatrix asmLoadSpatialMatrix(uint64_t base, uint fieldOffset, uint lane)
{
    AbMat33 tl, tr, bl;
    abLoadSpatialMatrix(base, fieldOffset, lane, tl, tr, bl);
    return asmMatrix(tl, tr, bl);
}

public void asmStoreSpatialMatrix(uint64_t base, uint fieldOffset, uint lane, AbSpatialMatrix m)
{
    abStoreSpatialMatrix(base, fieldOffset, lane, m.topLeft, m.topRight, m.bottomLeft);
}
