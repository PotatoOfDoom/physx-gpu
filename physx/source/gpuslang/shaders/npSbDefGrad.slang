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

// npSbDefGrad.slang
//
// Slang port of the soft-body deformation-gradient primitives from softBody.cuh:
// calculateDeformationGradient (L44-59) and getSoftBodyInvMass (L38-42).
//   P = PxMat33(u1, u2, u3);          // columns
//   F = P * Qinv;                     // deformation gradient
//   F = RTranspose * F;               // remove rotation -> element space
//   invMass = baryMass * sqrt(bary.x^2 + bary.y^2 + bary.z^2 + bary.w^2)
//
// PxMat33 column-major mat-mat A*B = (A.transform(B.col_i)).  sqrt is IEEE
// (OpSqrt).  All float arith NoContraction-stamped; +-0 eqFZ.

struct M3 { float3 c0, c1, c2; }

float3 mtx(M3 m, float3 v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z; }
M3 matmul(M3 a, M3 b) { M3 r; r.c0 = mtx(a, b.c0); r.c1 = mtx(a, b.c1); r.c2 = mtx(a, b.c2); return r; }
M3 mk(float4 c0, float4 c1, float4 c2) { M3 m; m.c0 = c0.xyz; m.c1 = c1.xyz; m.c2 = c2.xyz; return m; }

struct SbIn
{
    float4 u1, u2, u3;        // P columns
    float4 qinvC0, qinvC1, qinvC2;   // Qinv columns
    float4 rtC0, rtC1, rtC2;         // RTranspose columns
    float4 bary;             // bary.xyzw
    float4 misc;             // baryMass, _, _, _
}

struct SbOut
{
    float4 fC0, fC1, fC2;    // F columns
    float4 scalars;          // invMass, _, _, _
}

struct SbParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SbParams> gParams;

RWStructuredBuffer<SbIn> gIn;    // binding 0
RWStructuredBuffer<SbOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbDefGrad(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SbIn s = gIn[tid];

    // P = PxMat33(u1, u2, u3) (columns)
    M3 P; P.c0 = s.u1.xyz; P.c1 = s.u2.xyz; P.c2 = s.u3.xyz;
    const M3 Qinv = mk(s.qinvC0, s.qinvC1, s.qinvC2);
    const M3 RT = mk(s.rtC0, s.rtC1, s.rtC2);

    M3 F = matmul(P, Qinv);   // F = P * Qinv
    F = matmul(RT, F);        // F = RTranspose * F

    // invMass = baryMass * sqrt(sum bary_i^2)
    const float4 b = s.bary;
    const float sumSq = b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
    const float invMass = s.misc.x * sqrt(sumSq);

    SbOut o;
    o.fC0 = float4(F.c0, 0.0f);
    o.fC1 = float4(F.c1, 0.0f);
    o.fC2 = float4(F.c2, 0.0f);
    o.scalars = float4(invMass, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
