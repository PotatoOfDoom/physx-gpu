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

// npSbSolveNeoHookean.slang
//
// Slang port of tetrahedronsSolveInnerNeoHookean (softBodyGM.cu L657-717): the
// NeoHookean hydrostatic-invariant XPBD tet solve.  The per-tet multiplier slot
// (shSoftbody.mSimTetraMultipliers[workIndex/32].mbyz[workIndex&31].x) is
// modeled as an in/out scalar totalLambdaIn -> totalLambdaOut.
//
//   F = P * Qinv                          // col_i = P.transform(Qinv.col_i)
//   g1 = F.c0*2*Qinv.c0.x + F.c1*2*Qinv.c1.x + F.c2*2*Qinv.c2.x
//        -- per component ((F.c0.j*2)*q0x + (F.c1.j*2)*q1x) + (F.c2.j*2)*q2x,
//           each product left-to-right (vector*2.0f first, then *Qinv term)
//   g2, g3 analogous with .y / .z selectors
//   C  = magSq(F.c0) + magSq(F.c1) + magSq(F.c2) - 3.0f
//   g0 = -g1 - g2 - g3                    // ((-g1) - g2) - g3
//   w  = magSq(g0)*v0.w + magSq(g1)*v1.w + magSq(g2)*v2.w + magSq(g3)*v3.w
//   alpha = invDtSq * invE
//   PGS (!isTGS): dlambda = -(C + alpha*totalLambda.x) / (w + alpha)
//                 totalLambda.x += dlambda            (written back)
//   TGS:          dlambda = -C / (w + alpha)          (multiplier untouched)
//   delta = (g1*dlambda, g2*dlambda, g3*dlambda)
//
// magnitudeSquared = x*x+y*y+z*z.  The unary minus wraps the whole numerator:
// (-(X))/Y.  All float mul/add/sub/div NoContraction-stamped; +-0 eqFZ.

float magSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// PxMat33 operator*: col_i = lhs.transform(rhs.col_i) = (c0*v.x + c1*v.y) + c2*v.z
float3 mat33Transform(float3 c0, float3 c1, float3 c2, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}

struct NhIn
{
    float4 pc0;   // P.column0.xyz
    float4 pc1;   // P.column1.xyz
    float4 pc2;   // P.column2.xyz
    float4 qc0;   // Qinv.column0.xyz
    float4 qc1;   // Qinv.column1.xyz
    float4 qc2;   // Qinv.column2.xyz
    float4 im;    // v0.w, v1.w, v2.w, v3.w (inverse masses)
    float4 prm;   // invDtSq, invE, asfloat(isTGS), totalLambdaIn
}

struct NhOut
{
    float4 d0;    // delta.column0.xyz, dlambda
    float4 d1;    // delta.column1.xyz, totalLambdaOut
    float4 d2;    // delta.column2.xyz, 0
}

struct NhParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<NhParams> gParams;

RWStructuredBuffer<NhIn> gIn;    // binding 0
RWStructuredBuffer<NhOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbSolveNeoHookean(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const NhIn s = gIn[tid];

    const float3 P0 = s.pc0.xyz, P1 = s.pc1.xyz, P2 = s.pc2.xyz;
    const float3 Q0 = s.qc0.xyz, Q1 = s.qc1.xyz, Q2 = s.qc2.xyz;
    const float invDtSq = s.prm.x, invE = s.prm.y;
    const bool isTGS = asuint(s.prm.z) == 1u;
    float totalLambdaX = s.prm.w;

    // F = P * Qinv
    const float3 F0 = mat33Transform(P0, P1, P2, Q0);
    const float3 F1 = mat33Transform(P0, P1, P2, Q1);
    const float3 F2 = mat33Transform(P0, P1, P2, Q2);

    const float3 g1 = F0 * 2.0f * Q0.x + F1 * 2.0f * Q1.x + F2 * 2.0f * Q2.x;
    const float3 g2 = F0 * 2.0f * Q0.y + F1 * 2.0f * Q1.y + F2 * 2.0f * Q2.y;
    const float3 g3 = F0 * 2.0f * Q0.z + F1 * 2.0f * Q1.z + F2 * 2.0f * Q2.z;

    const float C = magSq(F0) + magSq(F1) + magSq(F2) - 3.0f;

    const float3 g0 = -g1 - g2 - g3;

    const float w = magSq(g0) * s.im.x +
                    magSq(g1) * s.im.y +
                    magSq(g2) * s.im.z +
                    magSq(g3) * s.im.w;

    const float alpha = invDtSq * invE;
    float dlambda;

    if (!isTGS)
    {
        dlambda = -(C + alpha * totalLambdaX) / (w + alpha);
        totalLambdaX += dlambda;
    }
    else
    {
        dlambda = -C / (w + alpha);
    }

    NhOut o;
    o.d0 = float4(g1 * dlambda, dlambda);
    o.d1 = float4(g2 * dlambda, totalLambdaX);
    o.d2 = float4(g3 * dlambda, 0.0f);
    gOut[tid] = o;
}
