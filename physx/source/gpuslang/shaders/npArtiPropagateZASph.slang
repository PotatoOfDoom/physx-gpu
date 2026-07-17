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

// npArtiPropagateZASph.slang
//
// Slang port of the SPHERICAL (multi-dof, dofs<=3) branch of
// computePropagateSpatialInertia_ZA_ZIc (forwardDynamic2.cu L1450-1537): the
// articulated-inertia + zero-acceleration-force propagation term for a
// spherical joint with 1..3 active dofs.
//
//   PxMat33 invStIs(PxIdentity);                        // identity-padded 3x3
//   for ind2<dofs:
//     sa = motion[ind2];
//     for ind<dofs: invStIs[ind][ind2] = sa.innerProduct(Is[ind]);  // (row ind2, col ind)
//     invStIs[ind2][ind2] += armature[ind2];
//     qstZG[ind2]      = -sa.innerProduct(Z);
//     qstZIcIntG[ind2] = (jF ? jF[ind2] : 0) - sa.innerProduct(ZIcInt);
//   invStIs = invStIs.getInverse();                     // full 3x3 real inverse
//   // invStIsT[ind2] = (invStIs[0][ind2], invStIs[1][ind2], invStIs[2][ind2]) (row ind2)
//   columns[0..5] = 0;
//   for ind<dofs:
//     isID = sum_{ind2<dofs} Is[ind2] * invStIs[ind][ind2];   // (row ind2, col ind)
//     columns[0..2] += isID * Is[ind].bottom.{x,y,z};
//     columns[3..5] += isID * Is[ind].top.{x,y,z};
//     ZA    += isID * qstZG[ind];   ZAInt += isID * qstZIcIntG[ind];
//   return constructSpatialMatrix(columns):
//     topLeft=(c0.top,c1.top,c2.top); bottomLeft=(c0.bot,c1.bot,c2.bot);
//     topRight=(c3.top,c4.top,c5.top).
//
// innerProduct is the spatial motion/force DUALITY product
// (a.bottom.b.top + a.top.b.bottom).  For dofs<3 the identity padding makes the
// unused rows/cols of invStIs an identity block so getInverse inverts only the
// active dofs x dofs sub-block.  PxMat33 is column-major: M[c][r] is column c,
// row r; getInverse uses det = column0.dot(column1.cross(column2)) then the
// standard cofactor / invDet expansion.  All float arith NoContraction-stamped;
// +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
// spatial duality inner product a.innerProduct(b) = a.bottom.b.top + a.top.b.bottom
float inner(float3 aTop, float3 aBot, float3 bTop, float3 bBot) { return dot3(aBot, bTop) + dot3(aTop, bBot); }

struct PsIn
{
    float4 sa0Top, sa0Bot, sa1Top, sa1Bot, sa2Top, sa2Bot;  // motion matrices
    float4 is0Top, is0Bot, is1Top, is1Bot, is2Top, is2Bot;  // Is[0..2]
    float4 zTop, zBot;       // Z
    float4 zicTop, zicBot;   // ZIcInt
    float4 zaTop, zaBot;     // ZA initial
    float4 zaiTop, zaiBot;   // ZAInt initial
    float4 armature;         // armature[0..2], _
    float4 jf;               // jF[0..2], _
    float4 misc;             // dofs, jFnull(1=null), 0, 0
}

struct PsOut
{
    float4 invStIsT0, invStIsT1, invStIsT2;  // row ind2 of inverse, per dof
    float4 qstZ;                             // qstZG[0..2], _
    float4 qstZIcInternal;                   // qstZIcIntG[0..2], _
    float4 isID0Top, isID0Bot, isID1Top, isID1Bot, isID2Top, isID2Bot;  // mIsInvDW per dof
    float4 zaTop, zaBot;                     // ZA updated
    float4 zaiTop, zaiBot;                   // ZAInt updated
    float4 tlC0, tlC1, tlC2;                 // SM topLeft
    float4 trC0, trC1, trC2;                 // SM topRight
    float4 blC0, blC1, blC2;                 // SM bottomLeft
}

struct PsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PsParams> gParams;

RWStructuredBuffer<PsIn> gIn;    // binding 0
RWStructuredBuffer<PsOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void propZASph(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const PsIn s = gIn[tid];

    const uint dofs = uint(s.misc.x);
    const bool jFnull = asuint(s.misc.y) == 1u;

    // per-dof motion matrices and Is
    float3 saT[3], saB[3], isT[3], isB[3];
    saT[0] = s.sa0Top.xyz; saB[0] = s.sa0Bot.xyz;
    saT[1] = s.sa1Top.xyz; saB[1] = s.sa1Bot.xyz;
    saT[2] = s.sa2Top.xyz; saB[2] = s.sa2Bot.xyz;
    isT[0] = s.is0Top.xyz; isB[0] = s.is0Bot.xyz;
    isT[1] = s.is1Top.xyz; isB[1] = s.is1Bot.xyz;
    isT[2] = s.is2Top.xyz; isB[2] = s.is2Bot.xyz;

    // invStIs stored column-major: M[col][row], initialised to identity
    float M[3][3];
    [unroll] for (uint c = 0; c < 3; ++c)
        [unroll] for (uint r = 0; r < 3; ++r)
            M[c][r] = (c == r) ? 1.0f : 0.0f;

    float qstZG[3] = { 0.0f, 0.0f, 0.0f };
    float qstZIcIntG[3] = { 0.0f, 0.0f, 0.0f };

    [unroll] for (uint ind2 = 0; ind2 < 3; ++ind2)
    {
        if (ind2 < dofs)
        {
            const float3 mT = saT[ind2], mB = saB[ind2];
            [unroll] for (uint ind = 0; ind < 3; ++ind)
            {
                if (ind < dofs)
                    M[ind][ind2] = inner(mT, mB, isT[ind], isB[ind]);  // (row ind2, col ind)
            }
            M[ind2][ind2] += s.armature[ind2];

            const float stZ = inner(mT, mB, s.zTop.xyz, s.zBot.xyz);
            const float stZInt = inner(mT, mB, s.zicTop.xyz, s.zicBot.xyz);
            qstZG[ind2] = -stZ;
            qstZIcIntG[ind2] = (jFnull ? 0.0f : s.jf[ind2]) - stZInt;
        }
    }

    // getInverse: det = column0.dot(column1.cross(column2))
    const float3 col0 = float3(M[0][0], M[0][1], M[0][2]);
    const float3 col1 = float3(M[1][0], M[1][1], M[1][2]);
    const float3 col2 = float3(M[2][0], M[2][1], M[2][2]);
    const float det = dot3(col0, cross3(col1, col2));

    float I[3][3];
    if (det != 0.0f)
    {
        const float invDet = 1.0f / det;
        I[0][0] = invDet *  (M[1][1] * M[2][2] - M[2][1] * M[1][2]);
        I[0][1] = invDet * -(M[0][1] * M[2][2] - M[2][1] * M[0][2]);
        I[0][2] = invDet *  (M[0][1] * M[1][2] - M[0][2] * M[1][1]);
        I[1][0] = invDet * -(M[1][0] * M[2][2] - M[1][2] * M[2][0]);
        I[1][1] = invDet *  (M[0][0] * M[2][2] - M[0][2] * M[2][0]);
        I[1][2] = invDet * -(M[0][0] * M[1][2] - M[0][2] * M[1][0]);
        I[2][0] = invDet *  (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        I[2][1] = invDet * -(M[0][0] * M[2][1] - M[0][1] * M[2][0]);
        I[2][2] = invDet *  (M[0][0] * M[1][1] - M[1][0] * M[0][1]);
    }
    else
    {
        [unroll] for (uint c = 0; c < 3; ++c)
            [unroll] for (uint r = 0; r < 3; ++r)
                I[c][r] = (c == r) ? 1.0f : 0.0f;
    }

    // columns[0..5] accumulation (top/bottom halves)
    float3 colT[6];
    float3 colB[6];
    [unroll] for (uint k = 0; k < 6; ++k) { colT[k] = float3(0, 0, 0); colB[k] = float3(0, 0, 0); }

    float3 zaTop = s.zaTop.xyz, zaBot = s.zaBot.xyz;
    float3 zaiTop = s.zaiTop.xyz, zaiBot = s.zaiBot.xyz;

    float3 isIDTops[3], isIDBots[3];
    [unroll] for (uint q = 0; q < 3; ++q) { isIDTops[q] = float3(0, 0, 0); isIDBots[q] = float3(0, 0, 0); }

    [unroll] for (uint ind = 0; ind < 3; ++ind)
    {
        if (ind < dofs)
        {
            float3 isIDTop = float3(0, 0, 0), isIDBot = float3(0, 0, 0);
            [unroll] for (uint ind2 = 0; ind2 < 3; ++ind2)
            {
                if (ind2 < dofs)
                {
                    const float w = I[ind][ind2];   // invStIs[ind][ind2] = (row ind2, col ind)
                    isIDTop += isT[ind2] * w;
                    isIDBot += isB[ind2] * w;
                }
            }
            isIDTops[ind] = isIDTop;
            isIDBots[ind] = isIDBot;

            const float3 ib = isB[ind];  // Is[ind].bottom
            const float3 it = isT[ind];  // Is[ind].top
            colT[0] += isIDTop * ib.x; colB[0] += isIDBot * ib.x;
            colT[1] += isIDTop * ib.y; colB[1] += isIDBot * ib.y;
            colT[2] += isIDTop * ib.z; colB[2] += isIDBot * ib.z;
            colT[3] += isIDTop * it.x; colB[3] += isIDBot * it.x;
            colT[4] += isIDTop * it.y; colB[4] += isIDBot * it.y;
            colT[5] += isIDTop * it.z; colB[5] += isIDBot * it.z;

            zaTop += isIDTop * qstZG[ind];  zaBot += isIDBot * qstZG[ind];
            zaiTop += isIDTop * qstZIcIntG[ind]; zaiBot += isIDBot * qstZIcIntG[ind];
        }
    }

    PsOut o;
    // invStIsT[ind2] = row ind2 of inverse = (I[0][ind2], I[1][ind2], I[2][ind2])
    o.invStIsT0 = float4(I[0][0], I[1][0], I[2][0], 0.0f);
    o.invStIsT1 = float4(I[0][1], I[1][1], I[2][1], 0.0f);
    o.invStIsT2 = float4(I[0][2], I[1][2], I[2][2], 0.0f);
    o.qstZ = float4(qstZG[0], qstZG[1], qstZG[2], 0.0f);
    o.qstZIcInternal = float4(qstZIcIntG[0], qstZIcIntG[1], qstZIcIntG[2], 0.0f);
    o.isID0Top = float4(isIDTops[0], 0.0f); o.isID0Bot = float4(isIDBots[0], 0.0f);
    o.isID1Top = float4(isIDTops[1], 0.0f); o.isID1Bot = float4(isIDBots[1], 0.0f);
    o.isID2Top = float4(isIDTops[2], 0.0f); o.isID2Bot = float4(isIDBots[2], 0.0f);
    o.zaTop = float4(zaTop, 0.0f); o.zaBot = float4(zaBot, 0.0f);
    o.zaiTop = float4(zaiTop, 0.0f); o.zaiBot = float4(zaiBot, 0.0f);
    // topLeft = (c0.top, c1.top, c2.top); bottomLeft = (c0.bot, c1.bot, c2.bot); topRight = (c3.top, c4.top, c5.top)
    o.tlC0 = float4(colT[0], 0.0f); o.tlC1 = float4(colT[1], 0.0f); o.tlC2 = float4(colT[2], 0.0f);
    o.blC0 = float4(colB[0], 0.0f); o.blC1 = float4(colB[1], 0.0f); o.blC2 = float4(colB[2], 0.0f);
    o.trC0 = float4(colT[3], 0.0f); o.trC1 = float4(colT[4], 0.0f); o.trC2 = float4(colT[5], 0.0f);
    gOut[tid] = o;
}
