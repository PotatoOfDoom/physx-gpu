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

// npArtiSpatialMatrix.slang
//
// P8.86: the table-driven harness over pxgSlangSpatialMatrix.slang.  One thread
// per case, one opcode per case, and every operation compared bit-for-bit
// against its PX_CUDA_CALLABLE engine original on the host
// (tests/testArtiSpatialMatrix.cpp).
//
// The output record is NOT written whole.  Each thread loads the poison image
// the fixture uploaded, writes only the slots its opcode owns, and stores it
// back -- so an operation that writes an extra slot, or the right value into the
// wrong slot, fails as loudly as one that computes the wrong number.  That is
// the whole-region poison discipline of P8.76/P8.79 applied per case rather than
// per block region.
//
// Only the `.xyz` of each output float4 is written; the `.w` stays poison.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;

// The opcodes.  These are literals shared with the fixture -- unlike a struct
// offset, a mismatch here cannot be silent: every case of that opcode fails.
static const uint SMX_M33_ADD = 0u;
static const uint SMX_M33_SUB = 1u;
static const uint SMX_M33_MUL = 2u;
static const uint SMX_M33_SCALE = 3u;
static const uint SMX_M33_NEG = 4u;
static const uint SMX_M33_TRANSPOSE = 5u;
static const uint SMX_M33_TRANSFORM = 6u;
static const uint SMX_M33_TRANSFORM_T = 7u;
static const uint SMX_M33_INVERSE = 8u;
static const uint SMX_M33_DET = 9u;
static const uint SMX_INVERT_SYM33 = 10u;
static const uint SMX_SKEW = 11u;
static const uint SMX_BOTTOM_RIGHT = 12u;
static const uint SMX_SM_ADD = 13u;
static const uint SMX_SM_SUB = 14u;
static const uint SMX_SM_NEG = 15u;
static const uint SMX_SM_SCALE = 16u;
static const uint SMX_SM_MUL_MATRIX = 17u;
static const uint SMX_SM_TRANSPOSE = 18u;
static const uint SMX_SM_MUL_VEC = 19u;
static const uint SMX_SM_ZERO = 20u;
static const uint SMX_CONSTRUCT_IS_STI = 21u;
static const uint SMX_CONSTRUCT_COLUMNS = 22u;
static const uint SMX_INVERT_INERTIA = 23u;
static const uint SMX_TRANSLATE_INERTIA = 24u;
static const uint SMX_INNER_PRODUCT = 25u;
static const uint SMX_SVEC_DOT = 26u;
static const uint SMX_SVEC_CROSS = 27u;
static const uint SMX_TRANSLATE_SVEC = 28u;
static const uint SMX_SVEC_ADD = 29u;
static const uint SMX_SVEC_SUB = 30u;
static const uint SMX_SVEC_SCALE = 31u;

struct SmxCase
{
    float4 aTL0, aTL1, aTL2;   // spatial matrix A
    float4 aTR0, aTR1, aTR2;
    float4 aBL0, aBL1, aBL2;
    float4 bTL0, bTL1, bTL2;   // spatial matrix B
    float4 bTR0, bTR1, bTR2;
    float4 bBL0, bBL1, bBL2;
    float4 vTop, vBot;         // spatial vector v
    float4 wTop, wBot;         // spatial vector w
    float4 mC0, mC1, mC2;      // a standalone PxMat33 -- invertSym33's input and
                               // translateInertia's sTod
    float4 rvec;               // constructSkewSymmetricMatrix's float4, and the
                               // offset for translateSpatialVectorInPlace
    float4 scal;               // .x = the scalar
    uint4 op;                  // .x = the opcode
};

struct SmxOut
{
    float4 mTL0, mTL1, mTL2;   // spatial-matrix result
    float4 mTR0, mTR1, mTR2;
    float4 mBL0, mBL1, mBL2;
    float4 matC0, matC1, matC2;  // PxMat33 result
    float4 vecTop, vecBot;       // spatial-vector result
    float4 scalarOut;            // .x
};

struct SmxParams { uint numCases; };

[[vk::push_constant]]
ConstantBuffer<SmxParams> gSmx;

RWStructuredBuffer<SmxCase> gIn;    // binding 0
RWStructuredBuffer<SmxOut> gOut;    // binding 1

AbMat33 smxM33(float4 c0, float4 c1, float4 c2)
{
    return asmM33(c0.xyz, c1.xyz, c2.xyz);
}

void smxPutM33(inout SmxOut o, AbMat33 m)
{
    o.matC0.xyz = m.c0;
    o.matC1.xyz = m.c1;
    o.matC2.xyz = m.c2;
}

void smxPutMatrix(inout SmxOut o, AbSpatialMatrix m)
{
    o.mTL0.xyz = m.topLeft.c0;
    o.mTL1.xyz = m.topLeft.c1;
    o.mTL2.xyz = m.topLeft.c2;
    o.mTR0.xyz = m.topRight.c0;
    o.mTR1.xyz = m.topRight.c1;
    o.mTR2.xyz = m.topRight.c2;
    o.mBL0.xyz = m.bottomLeft.c0;
    o.mBL1.xyz = m.bottomLeft.c1;
    o.mBL2.xyz = m.bottomLeft.c2;
}

void smxPutVec(inout SmxOut o, AbSpatialVector v)
{
    o.vecTop.xyz = v.top;
    o.vecBot.xyz = v.bottom;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void artiSpatialMatrixOps(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gSmx.numCases)
        return;

    const SmxCase c = gIn[tid];

    const AbSpatialMatrix A = asmMatrix(smxM33(c.aTL0, c.aTL1, c.aTL2),
                                        smxM33(c.aTR0, c.aTR1, c.aTR2),
                                        smxM33(c.aBL0, c.aBL1, c.aBL2));
    const AbSpatialMatrix B = asmMatrix(smxM33(c.bTL0, c.bTL1, c.bTL2),
                                        smxM33(c.bTR0, c.bTR1, c.bTR2),
                                        smxM33(c.bBL0, c.bBL1, c.bBL2));
    const AbMat33 M = smxM33(c.mC0, c.mC1, c.mC2);
    const AbSpatialVector v = asmSVec(c.vTop.xyz, c.vBot.xyz);
    const AbSpatialVector w = asmSVec(c.wTop.xyz, c.wBot.xyz);
    const float s = c.scal.x;

    SmxOut o = gOut[tid];   // the poison image; untouched slots must survive

    switch (c.op.x)
    {
    case SMX_M33_ADD:         smxPutM33(o, asmM33Add(A.topLeft, B.topLeft)); break;
    case SMX_M33_SUB:         smxPutM33(o, asmM33Sub(A.topLeft, B.topLeft)); break;
    case SMX_M33_MUL:         smxPutM33(o, asmM33Mul(A.topLeft, B.topLeft)); break;
    case SMX_M33_SCALE:       smxPutM33(o, asmM33Scale(A.topLeft, s)); break;
    case SMX_M33_NEG:         smxPutM33(o, asmM33Neg(A.topLeft)); break;
    case SMX_M33_TRANSPOSE:   smxPutM33(o, asmM33Transpose(A.topLeft)); break;
    case SMX_M33_INVERSE:     smxPutM33(o, asmM33Inverse(M)); break;
    case SMX_INVERT_SYM33:    smxPutM33(o, asmInvertSym33(M)); break;
    case SMX_SKEW:            smxPutM33(o, asmSkewSymmetric(c.rvec)); break;
    case SMX_BOTTOM_RIGHT:    smxPutM33(o, asmBottomRight(A)); break;

    case SMX_M33_TRANSFORM:   o.vecTop.xyz = asmM33Transform(A.topLeft, v.top); break;
    case SMX_M33_TRANSFORM_T: o.vecTop.xyz = asmM33TransformTranspose(A.topLeft, v.top); break;
    case SMX_M33_DET:         o.scalarOut.x = asmM33Determinant(M); break;

    case SMX_SM_ADD:          smxPutMatrix(o, asmAdd(A, B)); break;
    case SMX_SM_SUB:          smxPutMatrix(o, asmSub(A, B)); break;
    case SMX_SM_NEG:          smxPutMatrix(o, asmNeg(A)); break;
    case SMX_SM_SCALE:        smxPutMatrix(o, asmScale(A, s)); break;
    case SMX_SM_MUL_MATRIX:   smxPutMatrix(o, asmMulMatrix(A, B)); break;
    case SMX_SM_TRANSPOSE:    smxPutMatrix(o, asmTranspose(A)); break;
    case SMX_SM_ZERO:         smxPutMatrix(o, asmMatrixZero()); break;
    case SMX_SM_MUL_VEC:      smxPutVec(o, asmMulVec(A, v)); break;

    case SMX_CONSTRUCT_IS_STI: smxPutMatrix(o, asmConstructSpatialMatrix(v, w)); break;

    case SMX_CONSTRUCT_COLUMNS:
        {
            // Six columns whose BOTTOM halves are all distinct and all non-zero,
            // so the three the engine discards (3, 4, 5) are visibly discarded.
            AbSpatialVector cols[6];
            cols[0] = asmSVec(A.topLeft.c0, A.bottomLeft.c0);
            cols[1] = asmSVec(A.topLeft.c1, A.bottomLeft.c1);
            cols[2] = asmSVec(A.topLeft.c2, A.bottomLeft.c2);
            cols[3] = asmSVec(A.topRight.c0, M.c0);
            cols[4] = asmSVec(A.topRight.c1, M.c1);
            cols[5] = asmSVec(A.topRight.c2, M.c2);
            smxPutMatrix(o, asmConstructFromColumns(cols));
        }
        break;

    case SMX_INVERT_INERTIA:  smxPutMatrix(o, asmInvertInertia(A)); break;

    case SMX_TRANSLATE_INERTIA:
        {
            AbSpatialMatrix t = A;
            asmTranslateInertia(M, t);
            smxPutMatrix(o, t);   // topRight must come back UNCHANGED
        }
        break;

    case SMX_INNER_PRODUCT:   o.scalarOut.x = asmInnerProduct(v, w); break;
    case SMX_SVEC_DOT:        o.scalarOut.x = asmSVecDot(v, w); break;
    case SMX_SVEC_CROSS:      smxPutVec(o, asmSVecCross(v, w)); break;
    case SMX_SVEC_ADD:        smxPutVec(o, asmSVecAdd(v, w)); break;
    case SMX_SVEC_SUB:        smxPutVec(o, asmSVecSub(v, w)); break;
    case SMX_SVEC_SCALE:      smxPutVec(o, asmSVecScale(v, s)); break;

    case SMX_TRANSLATE_SVEC:
        {
            AbSpatialVector t = v;
            asmTranslateSpatialVectorInPlace(c.rvec.xyz, t);
            smxPutVec(o, t);      // the TOP half must come back unchanged
        }
        break;

    default:
        break;
    }

    gOut[tid] = o;
}

// ===========================================================================
// The block LOADS
// ===========================================================================
//
// `abStoreSpatialMatrix` and `abStorePxMat33` were pinned in P8.84 by comparing
// the block image they produce against the engine's own `storeSpatialMatrix` /
// `storePxMat33`.  Their load twins (P8.86) need the opposite arrangement: the
// fixture builds the block image with the ENGINE's writers, this entry point
// loads it with ours, and the components come back out as a flat float array
// that is compared against the engine's own `loadSpatialMatrix` /
// `loadPxMat33`.  Neither half is checked against the other -- a load and a
// store that are wrong in mirrored ways would round-trip perfectly and pass a
// copy test.
//
// This runs on the shim's engine ABI (push constants only), so the output is a
// BDA pointer rather than a bound buffer.  36 floats per lane: 27 for the
// spatial matrix, 9 for the PxMat33.

// `asmStoreSpatialMatrix` is checked here too, and it is not the no-op wrapper it
// looks like: it hands `abStoreSpatialMatrix` three separate AbMat33s, so a
// permutation of topLeft / topRight / bottomLeft in the forwarding is exactly the
// kind of thing that compiles.  The second block it writes is compared against the
// engine's `storeSpatialMatrix` image of the engine-loaded matrix.
struct SmxIoParams
{
    uint64_t smBlock;     // @0   PxgSpatialMatrixBlock*   (source)
    uint64_t m33Block;    // @8   PxgMat33Block*
    uint64_t outFloats;   // @16  float[nbLanes * 36]
    uint64_t smBlockOut;  // @24  PxgSpatialMatrixBlock*   (destination)
    uint nbLanes;         // @32
    uint pad0;            // @36 -> 40 bytes
};

[[vk::push_constant]]
ConstantBuffer<SmxIoParams> gSmxIo;

static const uint SMX_IO_FLOATS_PER_LANE = 36u;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiSpatialMatrixBlockIo(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSmxIo.nbLanes)
        return;

    const AbSpatialMatrix m = asmLoadSpatialMatrix(gSmxIo.smBlock, 0u, lane);
    const AbMat33 q = abLoadPxMat33(gSmxIo.m33Block, 0u, lane);

    Ptr<float> o = Ptr<float>(gSmxIo.outFloats
                              + uint64_t(lane) * uint64_t(SMX_IO_FLOATS_PER_LANE) * 4);

    o[0] = m.topLeft.c0.x;     o[1] = m.topLeft.c0.y;     o[2] = m.topLeft.c0.z;
    o[3] = m.topLeft.c1.x;     o[4] = m.topLeft.c1.y;     o[5] = m.topLeft.c1.z;
    o[6] = m.topLeft.c2.x;     o[7] = m.topLeft.c2.y;     o[8] = m.topLeft.c2.z;
    o[9] = m.topRight.c0.x;    o[10] = m.topRight.c0.y;   o[11] = m.topRight.c0.z;
    o[12] = m.topRight.c1.x;   o[13] = m.topRight.c1.y;   o[14] = m.topRight.c1.z;
    o[15] = m.topRight.c2.x;   o[16] = m.topRight.c2.y;   o[17] = m.topRight.c2.z;
    o[18] = m.bottomLeft.c0.x; o[19] = m.bottomLeft.c0.y; o[20] = m.bottomLeft.c0.z;
    o[21] = m.bottomLeft.c1.x; o[22] = m.bottomLeft.c1.y; o[23] = m.bottomLeft.c1.z;
    o[24] = m.bottomLeft.c2.x; o[25] = m.bottomLeft.c2.y; o[26] = m.bottomLeft.c2.z;

    o[27] = q.c0.x; o[28] = q.c0.y; o[29] = q.c0.z;
    o[30] = q.c1.x; o[31] = q.c1.y; o[32] = q.c1.z;
    o[33] = q.c2.x; o[34] = q.c2.y; o[35] = q.c2.z;

    asmStoreSpatialMatrix(gSmxIo.smBlockOut, 0u, lane, m);
}
