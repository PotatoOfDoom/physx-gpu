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

// npArtiInvertInertia.slang
//
// Slang port of Dy::SpatialMatrix::invertInertia()
// (DyFeatherstoneArticulationUtils.h L535-555) plus its invertSym33 helper
// (L488-508): the block symmetric 6x6 spatial-inertia inverse via a Schur
// complement.
//   aa = symmetrise(bottomLeft); ll = symmetrise(topRight); la = topLeft
//   AAInv = invertSym33(aa)
//   z  = -la * AAInv
//   S  = ll + z * la^T                 // Schur complement of aa
//   LL = invertSym33(S)
//   LA = LL * z
//   AA = AAInv + z^T * LA
//   return SpatialMatrix(LA^T, AA, LL) // topLeft=LA^T, topRight=AA, bottomLeft=LL
//
// invertSym33(in): v0=in[1]xin[2], v1=in[2]xin[0], v2=in[0]xin[1];
//   det=v0.in[0]; if det!=0 recipDet=1/det and return the columns
//   (v0, (v0.y,v1.y,v1.z), (v0.z,v1.z,v2.z))*recipDet (exploiting symmetry),
//   else Identity.  in[i] is COLUMN i (PxMat33 column-major).
// PxMat33 mat-mat A*B = (A.transform(B.col_i)); *scalar/+/unary- are columnwise;
// getTranspose swaps rows/cols.  All float arith NoContraction-stamped; +-0 eqFZ.

struct M3 { float3 c0, c1, c2; }

float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
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
M3 neg3(M3 m) { M3 r; r.c0 = -m.c0; r.c1 = -m.c1; r.c2 = -m.c2; return r; }
M3 identity3() { M3 r; r.c0 = float3(1, 0, 0); r.c1 = float3(0, 1, 0); r.c2 = float3(0, 0, 1); return r; }

M3 invertSym33(M3 m)
{
    const float3 v0 = cross3(m.c1, m.c2);
    const float3 v1 = cross3(m.c2, m.c0);
    const float3 v2 = cross3(m.c0, m.c1);

    const float det = dot3(v0, m.c0);
    if (det != 0.0f)
    {
        const float recipDet = 1.0f / det;
        M3 r;
        r.c0 = v0 * recipDet;
        r.c1 = float3(v0.y, v1.y, v1.z) * recipDet;
        r.c2 = float3(v0.z, v1.z, v2.z) * recipDet;
        return r;
    }
    return identity3();
}

struct IiIn
{
    float4 tlC0, tlC1, tlC2;  // topLeft
    float4 trC0, trC1, trC2;  // topRight
    float4 blC0, blC1, blC2;  // bottomLeft
}

struct IiOut
{
    float4 tlC0, tlC1, tlC2;  // result topLeft  = LA^T
    float4 trC0, trC1, trC2;  // result topRight = AA
    float4 blC0, blC1, blC2;  // result bottomLeft = LL
}

struct IiParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<IiParams> gParams;

RWStructuredBuffer<IiIn> gIn;    // binding 0
RWStructuredBuffer<IiOut> gOut;  // binding 1

M3 mk(float4 c0, float4 c1, float4 c2) { M3 m; m.c0 = c0.xyz; m.c1 = c1.xyz; m.c2 = c2.xyz; return m; }

[shader("compute")]
[numthreads(64, 1, 1)]
void invertInertia(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const IiIn s = gIn[tid];

    M3 aa = mk(s.blC0, s.blC1, s.blC2);   // bottomLeft
    M3 ll = mk(s.trC0, s.trC1, s.trC2);   // topRight
    const M3 la = mk(s.tlC0, s.tlC1, s.tlC2);   // topLeft

    aa = scale3(add3(aa, transpose3(aa)), 0.5f);
    ll = scale3(add3(ll, transpose3(ll)), 0.5f);

    const M3 AAInv = invertSym33(aa);

    const M3 z = matmul(neg3(la), AAInv);
    const M3 S = add3(ll, matmul(z, transpose3(la)));

    const M3 LL = invertSym33(S);

    const M3 LA = matmul(LL, z);
    const M3 AA = add3(AAInv, matmul(transpose3(z), LA));

    const M3 rTL = transpose3(LA);   // result topLeft = LA^T

    IiOut o;
    o.tlC0 = float4(rTL.c0, 0.0f); o.tlC1 = float4(rTL.c1, 0.0f); o.tlC2 = float4(rTL.c2, 0.0f);
    o.trC0 = float4(AA.c0, 0.0f);  o.trC1 = float4(AA.c1, 0.0f);  o.trC2 = float4(AA.c2, 0.0f);
    o.blC0 = float4(LL.c0, 0.0f);  o.blC1 = float4(LL.c1, 0.0f);  o.blC2 = float4(LL.c2, 0.0f);
    gOut[tid] = o;
}
