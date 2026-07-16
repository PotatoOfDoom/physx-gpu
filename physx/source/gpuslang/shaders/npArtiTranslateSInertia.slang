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

// npArtiTranslateSInertia.slang
//
// Slang port of translateInertia(const PxMat33& sTod, Dy::SpatialMatrix&
// inertia) (forwardDynamic2.cu L1546-1558): translate a full spatial inertia
// (block matrix [[topLeft, topRight],[bottomLeft, topLeft^T]]) by the 3x3
// transform sTod, in place.
//   dTos = sTod^T
//   bl = sTod * topLeft  + bottomLeft
//   br = sTod * topRight + getBottomRight()        (getBottomRight = topLeft^T)
//   topLeft'    = topLeft + topRight * dTos
//   bottomLeft' = bl + br * dTos
//   bottomLeft' = (bottomLeft' + bottomLeft'^T) * 0.5     (symmetrise)
// topRight is UNCHANGED; the result bottom-right = topLeft'^T (implicit).
//
// ** bl/br/topLeft' are all formed from the ORIGINAL topLeft/topRight/
//    bottomLeft (bl and br read them before topLeft is overwritten); and
//    getBottomRight() is the ORIGINAL topLeft^T, not bottomLeft. **
// PxMat33 column-major mat-mat A*B = (A.transform(B.col_i)); *scalar/+ are
// columnwise.  All float arith NoContraction-stamped; +-0 eqFZ.

struct M3 { float3 c0, c1, c2; }

float3 mtx(M3 m, float3 v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z; }
M3 matmul(M3 a, M3 b) { M3 r; r.c0 = mtx(a, b.c0); r.c1 = mtx(a, b.c1); r.c2 = mtx(a, b.c2); return r; }
M3 transpose3(M3 m)
{
    M3 r;
    r.c0 = float3(m.c0.x, m.c1.x, m.c2.x);
    r.c1 = float3(m.c0.y, m.c1.y, m.c2.y);
    r.c2 = float3(m.c0.z, m.c1.z, m.c2.z);
    return r;
}
M3 scale3(M3 m, float s) { M3 r; r.c0 = m.c0 * s; r.c1 = m.c1 * s; r.c2 = m.c2 * s; return r; }
M3 add3(M3 a, M3 b) { M3 r; r.c0 = a.c0 + b.c0; r.c1 = a.c1 + b.c1; r.c2 = a.c2 + b.c2; return r; }

struct TsIn
{
    float4 sC0, sC1, sC2;     // sTod columns
    float4 tlC0, tlC1, tlC2;  // inertia.topLeft
    float4 trC0, trC1, trC2;  // inertia.topRight
    float4 blC0, blC1, blC2;  // inertia.bottomLeft
}

struct TsOut
{
    float4 tlC0, tlC1, tlC2;  // topLeft'
    float4 trC0, trC1, trC2;  // topRight (unchanged)
    float4 blC0, blC1, blC2;  // bottomLeft'
}

struct TsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<TsParams> gParams;

RWStructuredBuffer<TsIn> gIn;    // binding 0
RWStructuredBuffer<TsOut> gOut;  // binding 1

M3 mk(float4 c0, float4 c1, float4 c2) { M3 m; m.c0 = c0.xyz; m.c1 = c1.xyz; m.c2 = c2.xyz; return m; }

[shader("compute")]
[numthreads(64, 1, 1)]
void transSInertia(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const TsIn s = gIn[tid];

    const M3 sTod = mk(s.sC0, s.sC1, s.sC2);
    const M3 TL = mk(s.tlC0, s.tlC1, s.tlC2);
    const M3 TR = mk(s.trC0, s.trC1, s.trC2);
    const M3 BL = mk(s.blC0, s.blC1, s.blC2);

    const M3 dTos = transpose3(sTod);
    const M3 BR = transpose3(TL);   // getBottomRight() = original topLeft^T

    const M3 bl = add3(matmul(sTod, TL), BL);
    const M3 br = add3(matmul(sTod, TR), BR);

    const M3 newTL = add3(TL, matmul(TR, dTos));
    const M3 newBL0 = add3(bl, matmul(br, dTos));
    const M3 newBL = scale3(add3(newBL0, transpose3(newBL0)), 0.5f);

    TsOut o;
    o.tlC0 = float4(newTL.c0, 0.0f); o.tlC1 = float4(newTL.c1, 0.0f); o.tlC2 = float4(newTL.c2, 0.0f);
    o.trC0 = float4(TR.c0, 0.0f);    o.trC1 = float4(TR.c1, 0.0f);    o.trC2 = float4(TR.c2, 0.0f);
    o.blC0 = float4(newBL.c0, 0.0f); o.blC1 = float4(newBL.c1, 0.0f); o.blC2 = float4(newBL.c2, 0.0f);
    gOut[tid] = o;
}
