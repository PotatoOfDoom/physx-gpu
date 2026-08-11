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

// P8.86: pxgSlangSpatialMatrix.slang — the spatial-matrix algebra layer.
//
// ===========================================================================
// THE ORACLE, AND WHY THIS IS THE CLEANEST ONE IN THE CHAIN SO FAR
// ===========================================================================
//
// Every operation under test has a direct PX_CUDA_CALLABLE counterpart that
// this fixture CALLS — `PxMat33`'s operators, `Dy::SpatialMatrix`'s operators,
// `invertSym33`, `invertInertia`, `constructSpatialMatrix`, and
// `Cm::UnAlignedSpatialVector::innerProduct / dot / cross`.  Nothing about the
// arithmetic is retyped into an expectation; the expectation IS the engine.
//
// Two exceptions, stated rather than glossed: `translateInertia`
// (forwardDynamic2.cu:1546) and `constructSkewSymmetricMatrix` (:1561) are
// `static __device__` and cannot be called from the host.  Their mirrors below
// are five lines of structure built out of the engine's own PxMat33 operators —
// so the arithmetic is still the engine's, only the ORDER is transcribed.  That
// order is exactly what traps 5 is about, so the sweep attacks it directly.
//
// ===========================================================================
// HOW A WRONG SLOT FAILS AS LOUDLY AS A WRONG NUMBER
// ===========================================================================
//
// The output buffer is uploaded PRE-FILLED with a per-case, per-word poison
// pattern, and the shader loads it, writes only the slots its opcode owns, and
// stores it back.  The expectation is the same poison image with the engine's
// values written into the same slots.  So:
//
//   * an operation that writes a slot it does not own fails;
//   * an operation that writes the right value into the wrong slot fails twice;
//   * `translateInertia` leaving `topRight` alone is CHECKED, not assumed —
//     the poison is what has to survive there;
//   * only `.xyz` of each output float4 is ever written, so the `.w` lanes are
//     32 further poison words per case that must come back untouched.
//
// ===========================================================================
// THE INPUT STRIPES, AND WHICH TRAP EACH ONE EXISTS FOR
// ===========================================================================
//
// The standing rule from eight earlier tranches is that a survivor is almost
// always a fixture coincidence.  Here the coincidences that would hide a real
// bug are specific and each one gets a stripe that rules it out:
//
//   * SINGULAR `M` (column 1 exactly zero -> determinant exactly +0):
//     `invertSym33` returns the IDENTITY there and `PxMat33::getInverse` does
//     too.  Without this stripe trap 2 is invisible.  Exactly zero, not small —
//     a near-singular matrix takes the OTHER branch on both sides and proves
//     nothing.
//   * SYMMETRIC `M`: `invertSym33` is a true inverse only here.  With only
//     symmetric inputs it and `getInverse` agree and trap 3 is invisible; with
//     only non-symmetric ones the *usual* case is never tested.  Both.
//   * SYMMETRIC `A.topLeft`: `transformTranspose` and `transform` agree only
//     here, which is trap 1's coincidence.
//   * ANTISYMMETRIC `A.bottomLeft`: `(bl + bl^T) * 0.5` is then exactly zero, so
//     `invertInertia` takes `invertSym33`'s identity branch INTERNALLY — the
//     fixed-base case, reached through the real call chain.
//   * ZERO `v.top` / ZERO `v.bottom`: isolates the two halves of
//     `SpatialMatrix * v` and of `innerProduct`, which otherwise sum together.
//   * ZERO and NEGATIVE scalars, and a non-round default.
//
// Every stripe period (7, 11, 13, 17, 19) is odd and the opcode count is 32, so
// each stripe reaches every operation rather than clustering on a few.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxMat33.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "PxgArticulation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxMat33;
using physx::PxVec3;
using physx::PxIdentity;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiSpatialMatrixLoader(const char* file) { return loadShaderBlob(file); }

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

// ---------------------------------------------------------------------------
// The opcodes.  Shared with npArtiSpatialMatrix.slang by value; a mismatch
// cannot be silent, because every case carrying that opcode then fails.
// ---------------------------------------------------------------------------
enum Op : uint32_t
{
    SMX_M33_ADD = 0,
    SMX_M33_SUB,
    SMX_M33_MUL,
    SMX_M33_SCALE,
    SMX_M33_NEG,
    SMX_M33_TRANSPOSE,
    SMX_M33_TRANSFORM,
    SMX_M33_TRANSFORM_T,
    SMX_M33_INVERSE,
    SMX_M33_DET,
    SMX_INVERT_SYM33,
    SMX_SKEW,
    SMX_BOTTOM_RIGHT,
    SMX_SM_ADD,
    SMX_SM_SUB,
    SMX_SM_NEG,
    SMX_SM_SCALE,
    SMX_SM_MUL_MATRIX,
    SMX_SM_TRANSPOSE,
    SMX_SM_MUL_VEC,
    SMX_SM_ZERO,
    SMX_CONSTRUCT_IS_STI,
    SMX_CONSTRUCT_COLUMNS,
    SMX_INVERT_INERTIA,
    SMX_TRANSLATE_INERTIA,
    SMX_INNER_PRODUCT,
    SMX_SVEC_DOT,
    SMX_SVEC_CROSS,
    SMX_TRANSLATE_SVEC,
    SMX_SVEC_ADD,
    SMX_SVEC_SUB,
    SMX_SVEC_SCALE,
    SMX_NUM_OPS
};

static const char* opName(uint32_t op)
{
    static const char* names[SMX_NUM_OPS] = {
        "m33Add", "m33Sub", "m33Mul", "m33Scale", "m33Neg", "m33Transpose",
        "m33Transform", "m33TransformT", "m33Inverse", "m33Det", "invertSym33",
        "skew", "bottomRight", "smAdd", "smSub", "smNeg", "smScale", "smMulMatrix",
        "smTranspose", "smMulVec", "smZero", "constructIsStI", "constructColumns",
        "invertInertia", "translateInertia", "innerProduct", "svecDot", "svecCross",
        "translateSVec", "svecAdd", "svecSub", "svecScale"
    };
    return op < SMX_NUM_OPS ? names[op] : "?";
}

struct SmxCase
{
    F4 aTL0, aTL1, aTL2;
    F4 aTR0, aTR1, aTR2;
    F4 aBL0, aBL1, aBL2;
    F4 bTL0, bTL1, bTL2;
    F4 bTR0, bTR1, bTR2;
    F4 bBL0, bBL1, bBL2;
    F4 vTop, vBot;
    F4 wTop, wBot;
    F4 mC0, mC1, mC2;
    F4 rvec;
    F4 scal;
    U4 op;
};

struct SmxOut
{
    F4 mTL0, mTL1, mTL2;
    F4 mTR0, mTR1, mTR2;
    F4 mBL0, mBL1, mBL2;
    F4 matC0, matC1, matC2;
    F4 vecTop, vecBot;
    F4 scalarOut;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }
static bool signedZeroPair(float a, float b) { return a == 0.f && b == 0.f && f2u(a) != f2u(b); }

// ---------------------------------------------------------------------------
// Plumbing between the fixture's F4 records and the engine's types
// ---------------------------------------------------------------------------
static PxVec3 v3(const F4& f) { return PxVec3(f.x, f.y, f.z); }
static PxMat33 m33(const F4& c0, const F4& c1, const F4& c2) { return PxMat33(v3(c0), v3(c1), v3(c2)); }

static void putVec3(F4& dst, const PxVec3& v) { dst.x = v.x; dst.y = v.y; dst.z = v.z; }

static void putM33(SmxOut& o, const PxMat33& m)
{
    putVec3(o.matC0, m.column0);
    putVec3(o.matC1, m.column1);
    putVec3(o.matC2, m.column2);
}

static void putMatrix(SmxOut& o, const physx::Dy::SpatialMatrix& m)
{
    putVec3(o.mTL0, m.topLeft.column0);
    putVec3(o.mTL1, m.topLeft.column1);
    putVec3(o.mTL2, m.topLeft.column2);
    putVec3(o.mTR0, m.topRight.column0);
    putVec3(o.mTR1, m.topRight.column1);
    putVec3(o.mTR2, m.topRight.column2);
    putVec3(o.mBL0, m.bottomLeft.column0);
    putVec3(o.mBL1, m.bottomLeft.column1);
    putVec3(o.mBL2, m.bottomLeft.column2);
}

static void putSVec(SmxOut& o, const physx::Cm::UnAlignedSpatialVector& v)
{
    putVec3(o.vecTop, v.top);
    putVec3(o.vecBot, v.bottom);
}

// ---------------------------------------------------------------------------
// The two engine functions that are __device__-only, mirrored out of the
// engine's own PxMat33 operators.  Only the ORDER is transcribed — which is
// precisely what trap 5 is about, so the sweep attacks it rather than trusting
// it.  Both are byte-for-byte the CUDA at forwardDynamic2.cu:1546 and :1561.
// ---------------------------------------------------------------------------
static PxMat33 refSkewSymmetric(const F4& r)
{
    return PxMat33(PxVec3(0.f, r.z, -r.y),
                   PxVec3(-r.z, 0.f, r.x),
                   PxVec3(r.y, -r.x, 0.f));
}

static void refTranslateInertia(const PxMat33& sTod, physx::Dy::SpatialMatrix& inertia)
{
    const PxMat33 dTos = sTod.getTranspose();

    PxMat33 bl = sTod * inertia.topLeft + inertia.bottomLeft;
    PxMat33 br = sTod * inertia.topRight + inertia.getBottomRight();

    inertia.topLeft = inertia.topLeft + inertia.topRight * dTos;
    inertia.bottomLeft = bl + br * dTos;

    inertia.bottomLeft = (inertia.bottomLeft + inertia.bottomLeft.getTranspose()) * 0.5f;
}

// translateSpatialVectorInPlace (forwardDynamic2.cu:57) — one line, and the top
// half is deliberately untouched.
static void refTranslateSpatialVectorInPlace(const PxVec3& offset,
                                             physx::Cm::UnAlignedSpatialVector& vec)
{
    vec.bottom += offset.cross(vec.top);
}

PXG_TEST(artiSpatialMatrix)
{
    using physx::Cm::UnAlignedSpatialVector;
    using physx::Cm::SpatialVectorF;
    using physx::Dy::SpatialMatrix;

    std::mt19937 rng(864213u);
    auto rf = [&](float lo, float hi) {
        // non-round values on purpose: a fixture of halves and quarters lets
        // two different spellings of the same product round identically.
        return lo + (hi - lo) * (float(rng() % 1000003u) / 1000003.f);
    };
    auto rv = [&](float lo, float hi) { return F4{ rf(lo, hi), rf(lo, hi), rf(lo, hi), 0.f }; };

    const uint32_t CASES_PER_OP = 127;   // prime, and coprime with every stripe
    const uint32_t N = SMX_NUM_OPS * CASES_PER_OP;

    std::vector<SmxCase> in(N);
    std::vector<SmxOut> poison(N);

    size_t cSingularM = 0, cSymM = 0, cSymATL = 0, cAntiABL = 0;
    size_t cZeroVTop = 0, cZeroVBot = 0, cZeroScal = 0, cNegScal = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        SmxCase c{};
        c.op.x = i % SMX_NUM_OPS;

        c.aTL0 = rv(-2, 2); c.aTL1 = rv(-2, 2); c.aTL2 = rv(-2, 2);
        c.aTR0 = rv(-2, 2); c.aTR1 = rv(-2, 2); c.aTR2 = rv(-2, 2);
        c.aBL0 = rv(-2, 2); c.aBL1 = rv(-2, 2); c.aBL2 = rv(-2, 2);
        c.bTL0 = rv(-2, 2); c.bTL1 = rv(-2, 2); c.bTL2 = rv(-2, 2);
        c.bTR0 = rv(-2, 2); c.bTR1 = rv(-2, 2); c.bTR2 = rv(-2, 2);
        c.bBL0 = rv(-2, 2); c.bBL1 = rv(-2, 2); c.bBL2 = rv(-2, 2);
        c.mC0 = rv(-2, 2); c.mC1 = rv(-2, 2); c.mC2 = rv(-2, 2);
        c.vTop = rv(-3, 3); c.vBot = rv(-3, 3);
        c.wTop = rv(-3, 3); c.wBot = rv(-3, 3);
        c.rvec = rv(-3, 3);
        c.rvec.w = rf(-3, 3);   // must be IGNORED by constructSkewSymmetricMatrix
        c.scal = F4{ rf(-4, 4), 0.f, 0.f, 0.f };

        // --- the stripes -----------------------------------------------------
        if (i % 11 == 3)
        {
            // symmetric M: invertSym33 is a true inverse only here
            ++cSymM;
            c.mC1.x = c.mC0.y; c.mC2.x = c.mC0.z; c.mC2.y = c.mC1.z;
        }
        if (i % 17 == 5)
        {
            // EXACTLY singular M: column1 zero makes cross(c1, c2) exactly zero
            // and hence both determinants exactly +0, so both inverses take
            // their identity branch.
            ++cSingularM;
            c.mC1 = F4{ 0.f, 0.f, 0.f, 0.f };
        }
        if (i % 13 == 4)
        {
            // symmetric A.topLeft: transform == transformTranspose only here
            ++cSymATL;
            c.aTL1.x = c.aTL0.y; c.aTL2.x = c.aTL0.z; c.aTL2.y = c.aTL1.z;
        }
        if (i % 19 == 7)
        {
            // antisymmetric A.bottomLeft -> invertInertia's `aa` is exactly
            // zero -> invertSym33's identity branch, reached internally
            ++cAntiABL;
            const float a = c.aBL0.y, b = c.aBL0.z, d = c.aBL1.z;
            c.aBL0 = F4{ 0.f, a, b, 0.f };
            c.aBL1 = F4{ -a, 0.f, d, 0.f };
            c.aBL2 = F4{ -b, -d, 0.f, 0.f };
        }
        if (i % 7 == 2) { ++cZeroVTop; c.vTop = F4{ 0.f, 0.f, 0.f, 0.f }; }
        if (i % 7 == 5) { ++cZeroVBot; c.vBot = F4{ 0.f, 0.f, 0.f, 0.f }; }
        if (i % 23 == 6) { ++cZeroScal; c.scal.x = 0.f; }
        else if (c.scal.x < 0.f) ++cNegScal;

        in[i] = c;

        // --- the poison image ------------------------------------------------
        // Distinct per case AND per word, so a slot written from the wrong case
        // or the wrong word index is caught as surely as an unwritten one.
        SmxOut p{};
        uint32_t* pw = reinterpret_cast<uint32_t*>(&p);
        for (uint32_t k = 0; k < sizeof(SmxOut) / 4; ++k)
            pw[k] = 0xAB000000u | ((i & 0xFFFFu) << 8) | (k & 0xFFu);
        poison[i] = p;
    }

    std::printf("[ info ] smx stripes: singularM %zu symM %zu symATL %zu antiABL %zu "
                "zeroVTop %zu zeroVBot %zu zeroScal %zu negScal %zu\n",
                cSingularM, cSymM, cSymATL, cAntiABL, cZeroVTop, cZeroVBot, cZeroScal, cNegScal);
    PXG_CHECK(cSingularM > 200);
    PXG_CHECK(cSymM > 300);
    PXG_CHECK(cSymATL > 250);
    PXG_CHECK(cAntiABL > 180);
    PXG_CHECK(cZeroVTop > 500);
    PXG_CHECK(cZeroVBot > 500);
    PXG_CHECK(cZeroScal > 150);
    PXG_CHECK(cNegScal > 1000);

    // -----------------------------------------------------------------------
    // The expectation: the poison image with the ENGINE's values written into
    // the slots each opcode owns, and nothing else touched.
    // -----------------------------------------------------------------------
    std::vector<SmxOut> ref = poison;

    for (uint32_t i = 0; i < N; ++i)
    {
        const SmxCase& c = in[i];
        SmxOut& o = ref[i];

        SpatialMatrix A(m33(c.aTL0, c.aTL1, c.aTL2),
                        m33(c.aTR0, c.aTR1, c.aTR2),
                        m33(c.aBL0, c.aBL1, c.aBL2));
        SpatialMatrix B(m33(c.bTL0, c.bTL1, c.bTL2),
                        m33(c.bTR0, c.bTR1, c.bTR2),
                        m33(c.bBL0, c.bBL1, c.bBL2));
        const PxMat33 M = m33(c.mC0, c.mC1, c.mC2);
        const UnAlignedSpatialVector v(v3(c.vTop), v3(c.vBot));
        const UnAlignedSpatialVector w(v3(c.wTop), v3(c.wBot));
        const float s = c.scal.x;

        switch (c.op.x)
        {
        case SMX_M33_ADD:         putM33(o, A.topLeft + B.topLeft); break;
        case SMX_M33_SUB:         putM33(o, A.topLeft - B.topLeft); break;
        case SMX_M33_MUL:         putM33(o, A.topLeft * B.topLeft); break;
        case SMX_M33_SCALE:       putM33(o, A.topLeft * s); break;
        case SMX_M33_NEG:         putM33(o, -A.topLeft); break;
        case SMX_M33_TRANSPOSE:   putM33(o, A.topLeft.getTranspose()); break;
        case SMX_M33_INVERSE:     putM33(o, M.getInverse()); break;
        case SMX_INVERT_SYM33:    putM33(o, SpatialMatrix::invertSym33(M)); break;
        case SMX_SKEW:            putM33(o, refSkewSymmetric(c.rvec)); break;
        case SMX_BOTTOM_RIGHT:    putM33(o, A.getBottomRight()); break;

        case SMX_M33_TRANSFORM:   putVec3(o.vecTop, A.topLeft * v.top); break;
        case SMX_M33_TRANSFORM_T: putVec3(o.vecTop, A.topLeft.transformTranspose(v.top)); break;
        case SMX_M33_DET:         o.scalarOut.x = M.getDeterminant(); break;

        case SMX_SM_ADD:          putMatrix(o, A + B); break;
        case SMX_SM_SUB:          putMatrix(o, A - B); break;
        case SMX_SM_NEG:          putMatrix(o, -A); break;
        case SMX_SM_SCALE:        putMatrix(o, A * s); break;
        case SMX_SM_MUL_MATRIX:   putMatrix(o, A * B); break;
        case SMX_SM_TRANSPOSE:    putMatrix(o, A.getTranspose()); break;
        case SMX_SM_ZERO:         putMatrix(o, SpatialMatrix(physx::PxZero)); break;
        case SMX_SM_MUL_VEC:      putSVec(o, A * v); break;

        case SMX_CONSTRUCT_IS_STI:
            putMatrix(o, SpatialMatrix::constructSpatialMatrix(SpatialVectorF(v.top, v.bottom),
                                                               SpatialVectorF(w.top, w.bottom)));
            break;

        case SMX_CONSTRUCT_COLUMNS:
            {
                UnAlignedSpatialVector cols[6] = {
                    UnAlignedSpatialVector(A.topLeft.column0, A.bottomLeft.column0),
                    UnAlignedSpatialVector(A.topLeft.column1, A.bottomLeft.column1),
                    UnAlignedSpatialVector(A.topLeft.column2, A.bottomLeft.column2),
                    UnAlignedSpatialVector(A.topRight.column0, M.column0),
                    UnAlignedSpatialVector(A.topRight.column1, M.column1),
                    UnAlignedSpatialVector(A.topRight.column2, M.column2)
                };
                putMatrix(o, SpatialMatrix::constructSpatialMatrix(cols));
            }
            break;

        case SMX_INVERT_INERTIA:  putMatrix(o, A.invertInertia()); break;

        case SMX_TRANSLATE_INERTIA:
            {
                SpatialMatrix t = A;
                refTranslateInertia(M, t);
                putMatrix(o, t);
            }
            break;

        case SMX_INNER_PRODUCT:   o.scalarOut.x = v.innerProduct(w); break;
        case SMX_SVEC_DOT:        o.scalarOut.x = v.dot(w); break;
        case SMX_SVEC_CROSS:      putSVec(o, v.cross(w)); break;
        case SMX_SVEC_ADD:        putSVec(o, v + w); break;
        case SMX_SVEC_SUB:        putSVec(o, v - w); break;
        case SMX_SVEC_SCALE:      putSVec(o, v * s); break;

        case SMX_TRANSLATE_SVEC:
            {
                UnAlignedSpatialVector t = v;
                refTranslateSpatialVectorInPlace(v3(c.rvec), t);
                putSVec(o, t);
            }
            break;

        default:
            PXG_CHECK_MSG(false, "unmapped opcode %u", c.op.x);
            break;
        }
    }

    // -----------------------------------------------------------------------
    // Dispatch
    // -----------------------------------------------------------------------
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, poison);

    struct P { uint32_t numCases; } prm = { N };
    Kernel k("npArtiSpatialMatrix.spv", "artiSpatialMatrixOps",
             { (uint32_t)sizeof(SmxCase), (uint32_t)sizeof(SmxOut) }, sizeof(P));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto got = downloadBuffer<SmxOut>(dev, bOut, N);

    // -----------------------------------------------------------------------
    // Compare.  Bit for bit, with the documented +-0 tolerance and its count
    // printed: a signed-zero difference is a real (and permitted) platform
    // effect, and hiding the count would hide it drifting upwards.
    // -----------------------------------------------------------------------
    const size_t WORDS = sizeof(SmxOut) / 4;
    size_t bad = 0, signedZeros = 0;
    size_t perOpBad[SMX_NUM_OPS] = {};

    for (uint32_t i = 0; i < N; ++i)
    {
        const float* r = reinterpret_cast<const float*>(&ref[i]);
        const float* g = reinterpret_cast<const float*>(&got[i]);
        bool caseBad = false;
        for (size_t x = 0; x < WORDS; ++x)
        {
            if (bitEq(r[x], g[x]))
                continue;
            if (signedZeroPair(r[x], g[x]))
            {
                ++signedZeros;
                continue;
            }
            if (bad < 12)
                std::printf("MM case %u op %-16s word %zu  ref %.9g (%08x)  gpu %.9g (%08x)\n",
                            i, opName(in[i].op.x), x, r[x], f2u(r[x]), g[x], f2u(g[x]));
            caseBad = true;
            ++bad;
        }
        if (caseBad)
            ++perOpBad[in[i].op.x];
    }

    if (bad)
        for (uint32_t op = 0; op < SMX_NUM_OPS; ++op)
            if (perOpBad[op])
                std::printf("[ info ] op %-16s: %zu/%u cases differ\n", opName(op), perOpBad[op],
                            CASES_PER_OP);

    std::printf("[ info ] smx %u cases x %zu words, +-0 tolerated %zu\n", N, WORDS, signedZeros);
    PXG_CHECK_MSG(bad == 0, "spatial-matrix algebra bitwise-identical to the engine (mod +-0)");

    // A poison word must be exactly what was uploaded: prove the comparison
    // above is not vacuous by checking the LAST word of the last case, which no
    // opcode writes (`scalarOut.w`).
    PXG_CHECK(f2u(got[N - 1].scalarOut.w) ==
              (0xAB000000u | (((N - 1) & 0xFFFFu) << 8) | ((WORDS - 1) & 0xFFu)));
    (void)u2f;
}

// ===========================================================================
// P8.86b: abLoadSpatialMatrix / abLoadPxMat33
// ===========================================================================
//
// The store twins were pinned in P8.84 the other way round: the fixture built
// the expected block image by calling the engine's `storeSpatialMatrix` /
// `storePxMat33` and compared our written bytes against it.  Doing the loads by
// round-tripping through our own stores would prove nothing — a load and a store
// wrong in mirrored ways copy perfectly.  So here the ENGINE writes the block
// image, OUR loader reads it, and the components are compared against what the
// ENGINE's own `loadSpatialMatrix` / `loadPxMat33` return from the same bytes.
//
// Two things the fixture makes visible that a casual version would not:
//
//   * all 32 lanes carry DIFFERENT matrices, and every one of a lane's 27 + 9
//     values is distinct, so a column/row transposition, a lane-stride error and
//     a float4-packing error each land on different numbers rather than on a
//     coincidence;
//   * `columns[6][lane].w` is the packing's one unused word.  The engine's store
//     writes 0.f there; the fixture overwrites it with poison AFTER the store,
//     so a loader that reads it produces garbage instead of a zero it could get
//     away with.

PXG_TEST(artiSpatialMatrixBlockIo)
{
    using namespace physx;
    using physx::Dy::SpatialMatrix;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64 x4, u32 x2 } -> 40 bytes
        { "npArtiSpatialMatrix.spv", "artiSpatialMatrixBlockIo", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiSpatialMatrixLoader);

    const uint32_t LANES = 32;
    const uint32_t FLOATS_PER_LANE = 36;

    // 36 distinct, non-round values per lane, and distinct BETWEEN lanes.
    auto val = [](uint32_t lane, uint32_t k) {
        return (float(lane) * 37.0f + float(k) + 1.0f) * 0.09375f - 11.0f;
    };

    PxgSpatialMatrixBlock smBlock;
    PxgMat33Block m33Block;
    std::memset(&smBlock, 0xAB, sizeof(smBlock));
    std::memset(&m33Block, 0xAB, sizeof(m33Block));

    for (uint32_t lane = 0; lane < LANES; ++lane)
    {
        SpatialMatrix mat;
        float* mp[3] = { &mat.topLeft.column0.x, &mat.topRight.column0.x,
                         &mat.bottomLeft.column0.x };
        for (uint32_t b = 0; b < 3; ++b)
            for (uint32_t k = 0; k < 9; ++k)
                mp[b][k] = val(lane, b * 9 + k);

        PxMat33 q;
        for (uint32_t k = 0; k < 9; ++k)
            (&q.column0.x)[k] = val(lane, 27 + k);

        storeSpatialMatrix(smBlock, lane, mat);
        storePxMat33(m33Block, lane, q);

        // the packing's one unused word, poisoned after the engine zeroed it
        smBlock.columns[6][lane].w = u2f(0xDEAD0000u | lane);
    }

    // the destination block, poisoned: asmStoreSpatialMatrix must write all seven
    // float4 columns of every lane and nothing outside them
    PxgSpatialMatrixBlock smOutPoison;
    std::memset(&smOutPoison, 0xAB, sizeof(smOutPoison));

    PxgDevicePtr dSm = 0, dM33 = 0, dOut = 0, dSmOut = 0;
    shim.memAlloc(&dSm, sizeof(smBlock));
    shim.memAlloc(&dM33, sizeof(m33Block));
    shim.memAlloc(&dOut, size_t(LANES) * FLOATS_PER_LANE * sizeof(float));
    shim.memAlloc(&dSmOut, sizeof(smOutPoison));
    if (!dSm || !dM33 || !dOut || !dSmOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<float> outPoison(size_t(LANES) * FLOATS_PER_LANE);
    for (size_t k = 0; k < outPoison.size(); ++k)
        outPoison[k] = u2f(0xAB000000u | uint32_t(k));

    shim.memcpyHtoD(dSm, &smBlock, sizeof(smBlock));
    shim.memcpyHtoD(dM33, &m33Block, sizeof(m33Block));
    shim.memcpyHtoD(dOut, outPoison.data(), outPoison.size() * sizeof(float));
    shim.memcpyHtoD(dSmOut, &smOutPoison, sizeof(smOutPoison));

    struct Params
    {
        uint64_t smBlock, m33Block, outFloats, smBlockOut;
        uint32_t nbLanes, pad0;
    } prm = { uint64_t(dSm), uint64_t(dM33), uint64_t(dOut), uint64_t(dSmOut), LANES, 0u };
    PXG_CHECK_MSG(sizeof(prm) == 40, "push-constant blob is %zu bytes", sizeof(prm));

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiSpatialMatrixBlockIo");
    shim.streamSynchronize();

    std::vector<float> got(outPoison.size());
    shim.memcpyDtoH(got.data(), dOut, got.size() * sizeof(float));
    PxgSpatialMatrixBlock gotSmOut;
    shim.memcpyDtoH(&gotSmOut, dSmOut, sizeof(gotSmOut));

    // the expectation: the ENGINE's loaders reading the very same bytes, and the
    // ENGINE's writer re-storing what they returned
    PxgSpatialMatrixBlock wantSmOut = smOutPoison;
    size_t bad = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane)
    {
        SpatialMatrix mat;
        PxMat33 q;
        loadSpatialMatrix(smBlock, lane, mat);
        loadPxMat33(m33Block, lane, q);
        storeSpatialMatrix(wantSmOut, lane, mat);

        float want[FLOATS_PER_LANE];
        const float* mp[3] = { &mat.topLeft.column0.x, &mat.topRight.column0.x,
                               &mat.bottomLeft.column0.x };
        for (uint32_t b = 0; b < 3; ++b)
            for (uint32_t k = 0; k < 9; ++k)
                want[b * 9 + k] = mp[b][k];
        for (uint32_t k = 0; k < 9; ++k)
            want[27 + k] = (&q.column0.x)[k];

        for (uint32_t k = 0; k < FLOATS_PER_LANE; ++k)
        {
            const float g = got[size_t(lane) * FLOATS_PER_LANE + k];
            if (bitEq(want[k], g))
                continue;
            if (bad < 12)
                std::printf("MM lane %u slot %u  ref %.9g (%08x)  gpu %.9g (%08x)\n", lane, k,
                            want[k], f2u(want[k]), g, f2u(g));
            ++bad;
        }
    }

    PXG_CHECK_MSG(bad == 0, "block loads identical to loadSpatialMatrix / loadPxMat33");

    // asmStoreSpatialMatrix forwards three separate blocks to abStoreSpatialMatrix,
    // so a permutation there compiles.  Whole-block comparison, poison included.
    size_t badStore = 0;
    {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&wantSmOut);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(&gotSmOut);
        for (size_t k = 0; k < sizeof(wantSmOut) / 4; ++k)
        {
            if (w[k] == g[k])
                continue;
            if (badStore < 8)
                std::printf("MM store word %zu  ref %08x  gpu %08x\n", k, w[k], g[k]);
            ++badStore;
        }
    }
    PXG_CHECK_MSG(badStore == 0, "asmStoreSpatialMatrix image identical to storeSpatialMatrix");

    shim.memFree(dSm);
    shim.memFree(dM33);
    shim.memFree(dOut);
    shim.memFree(dSmOut);
}

}  // namespace
