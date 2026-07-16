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

// npArtiSpatialMatMat.slang
//
// Slang port of Dy::SpatialMatrix::operator*(const SpatialMatrix&)
// (DyFeatherstoneArticulationUtils.h L328-338): the articulated-inertia
// composition (spatial matrix times spatial matrix).  The SpatialMatrix stores
// three PxMat33 blocks (topLeft, topRight, bottomLeft; the bottom-right block
// is implicitly topLeft^T).  The product:
//   sBottomRight  = s.topLeft^T
//   bottomRight   = topLeft^T                    (THIS matrix's topLeft transposed)
//   newTopLeft    = topLeft*s.topLeft   + topRight*s.bottomLeft
//   newTopRight   = topLeft*s.topRight  + topRight*sBottomRight
//   newBottomLeft = bottomLeft*s.topLeft + bottomRight*s.bottomLeft
// (only the three blocks are stored; result bottom-right = newTopLeft^T.)
//
// ** TRAP: newBottomLeft's second summand uses bottomRight = THIS topLeft^T,
//    not bottomLeft; and newTopRight uses sBottomRight = s.topLeft^T. **
// PxMat33 column-major: A*B = (A.transform(B.col_i))_i; transform(v) =
// col0*v.x + col1*v.y + col2*v.z; transpose swaps rows<->columns.
// All float arith NoContraction-stamped; +-0 eqFZ.

struct M3 { float3 c0, c1, c2; }   // column-major 3x3

float3 mtx(M3 m, float3 v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z; }
M3 matmul(M3 a, M3 b)              // a * b = (a.transform(b.col_i))
{
    M3 r;
    r.c0 = mtx(a, b.c0);
    r.c1 = mtx(a, b.c1);
    r.c2 = mtx(a, b.c2);
    return r;
}
M3 transpose3(M3 m)
{
    M3 r;
    r.c0 = float3(m.c0.x, m.c1.x, m.c2.x);
    r.c1 = float3(m.c0.y, m.c1.y, m.c2.y);
    r.c2 = float3(m.c0.z, m.c1.z, m.c2.z);
    return r;
}
M3 add3(M3 a, M3 b)
{
    M3 r;
    r.c0 = a.c0 + b.c0;
    r.c1 = a.c1 + b.c1;
    r.c2 = a.c2 + b.c2;
    return r;
}

struct MmIn
{
    float4 aTlC0, aTlC1, aTlC2;   // A.topLeft columns
    float4 aTrC0, aTrC1, aTrC2;   // A.topRight
    float4 aBlC0, aBlC1, aBlC2;   // A.bottomLeft
    float4 bTlC0, bTlC1, bTlC2;   // B.topLeft
    float4 bTrC0, bTrC1, bTrC2;   // B.topRight
    float4 bBlC0, bBlC1, bBlC2;   // B.bottomLeft
}

struct MmOut
{
    float4 tlC0, tlC1, tlC2;   // newTopLeft
    float4 trC0, trC1, trC2;   // newTopRight
    float4 blC0, blC1, blC2;   // newBottomLeft
}

struct MmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<MmParams> gParams;

RWStructuredBuffer<MmIn> gIn;    // binding 0
RWStructuredBuffer<MmOut> gOut;  // binding 1

M3 mk(float4 c0, float4 c1, float4 c2) { M3 m; m.c0 = c0.xyz; m.c1 = c1.xyz; m.c2 = c2.xyz; return m; }

[shader("compute")]
[numthreads(64, 1, 1)]
void smMulMat(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const MmIn s = gIn[tid];

    const M3 aTl = mk(s.aTlC0, s.aTlC1, s.aTlC2);
    const M3 aTr = mk(s.aTrC0, s.aTrC1, s.aTrC2);
    const M3 aBl = mk(s.aBlC0, s.aBlC1, s.aBlC2);
    const M3 bTl = mk(s.bTlC0, s.bTlC1, s.bTlC2);
    const M3 bTr = mk(s.bTrC0, s.bTrC1, s.bTrC2);
    const M3 bBl = mk(s.bBlC0, s.bBlC1, s.bBlC2);

    const M3 sBottomRight = transpose3(bTl);   // s.topLeft^T
    const M3 bottomRight = transpose3(aTl);    // this.topLeft^T

    const M3 ntl = add3(matmul(aTl, bTl), matmul(aTr, bBl));
    const M3 ntr = add3(matmul(aTl, bTr), matmul(aTr, sBottomRight));
    const M3 nbl = add3(matmul(aBl, bTl), matmul(bottomRight, bBl));

    MmOut o;
    o.tlC0 = float4(ntl.c0, 0.0f); o.tlC1 = float4(ntl.c1, 0.0f); o.tlC2 = float4(ntl.c2, 0.0f);
    o.trC0 = float4(ntr.c0, 0.0f); o.trC1 = float4(ntr.c1, 0.0f); o.trC2 = float4(ntr.c2, 0.0f);
    o.blC0 = float4(nbl.c0, 0.0f); o.blC1 = float4(nbl.c1, 0.0f); o.blC2 = float4(nbl.c2, 0.0f);
    gOut[tid] = o;
}
