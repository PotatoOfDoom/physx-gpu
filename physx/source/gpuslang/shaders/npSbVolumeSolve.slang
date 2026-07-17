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

// npSbVolumeSolve.slang
//
// Slang port of volumeSolve (softBodyGM.cu L621-655): the XPBD volume-preserving
// constraint solve for a soft-body tetrahedron.  Returns the 3x3 whose columns
// are the per-vertex position deltas (scaled constraint gradients) for verts
// 1,2,3 (vert 0's delta is the negated column sum, folded into w only).
//   det   = Qinv.getDeterminant()          // c0.(c1 x c2)
//   volume= 1/(6*det)
//   alphaTilde = (volumeLambda==0) ? 0 : 1/(volumeLambda*volume*invDtSq)
//   F     = P * Qinv                        // F.col_i = P.transform(Qinv.col_i)
//   g0=F.c1 x F.c2 ; g1=F.c2 x F.c0 ; g2=F.c0 x F.c1
//   grad1 = g0*Qinv.c0.x + g1*Qinv.c1.x + g2*Qinv.c2.x
//   grad2 = g0*Qinv.c0.y + g1*Qinv.c1.y + g2*Qinv.c2.y
//   grad3 = g0*Qinv.c0.z + g1*Qinv.c1.z + g2*Qinv.c2.z
//   grad0 = -grad1 - grad2 - grad3
//   detF  = g2.dot(F.c2) ; C = detF - 1
//   w     = |grad0|^2*v0.w + |grad1|^2*v1.w + |grad2|^2*v2.w + |grad3|^2*v3.w
//   dl    = -C / (w + alphaTilde)
//   return (grad1*dl, grad2*dl, grad3*dl)
//
// isTGS is unused in the body.  PxMat33 columns; transform = c0*x+c1*y+c2*z.
// All float arith NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

struct VsIn
{
    float4 pc0, pc1, pc2;      // P columns (.xyz)
    float4 qc0, qc1, qc2;      // Qinv columns (.xyz)
    float4 mass;               // v0.w, v1.w, v2.w, v3.w  (inverse masses)
    float4 params;             // invDtSq, volumeLambda, _, _
}

struct VsOut
{
    float4 dc0, dc1, dc2;      // result columns: grad1*dl, grad2*dl, grad3*dl
}

struct VsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<VsParams> gParams;

RWStructuredBuffer<VsIn> gIn;    // binding 0
RWStructuredBuffer<VsOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbVolumeSolve(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const VsIn s = gIn[tid];

    const float3 Pc0 = s.pc0.xyz, Pc1 = s.pc1.xyz, Pc2 = s.pc2.xyz;
    const float3 Qc0 = s.qc0.xyz, Qc1 = s.qc1.xyz, Qc2 = s.qc2.xyz;
    const float m0 = s.mass.x, m1 = s.mass.y, m2 = s.mass.z, m3 = s.mass.w;
    const float invDtSq = s.params.x, volumeLambda = s.params.y;

    const float det = dot3(Qc0, cross3(Qc1, Qc2));
    const float volume = 1.0f / (6.0f * det);

    const float alphaTilde = (volumeLambda == 0.0f) ? 0.0f : 1.0f / (volumeLambda * volume * invDtSq);

    // F = P * Qinv : F.col_i = P.transform(Qinv.col_i)
    const float3 Fc0 = Pc0 * Qc0.x + Pc1 * Qc0.y + Pc2 * Qc0.z;
    const float3 Fc1 = Pc0 * Qc1.x + Pc1 * Qc1.y + Pc2 * Qc1.z;
    const float3 Fc2 = Pc0 * Qc2.x + Pc1 * Qc2.y + Pc2 * Qc2.z;

    const float3 g0 = cross3(Fc1, Fc2);
    const float3 g1 = cross3(Fc2, Fc0);
    const float3 g2 = cross3(Fc0, Fc1);

    const float3 grad1 = g0 * Qc0.x + g1 * Qc1.x + g2 * Qc2.x;
    const float3 grad2 = g0 * Qc0.y + g1 * Qc1.y + g2 * Qc2.y;
    const float3 grad3 = g0 * Qc0.z + g1 * Qc1.z + g2 * Qc2.z;
    const float3 grad0 = -grad1 - grad2 - grad3;

    const float detF = dot3(g2, Fc2);
    const float C = detF - 1.0f;

    const float w = magSq3(grad0) * m0 + magSq3(grad1) * m1 + magSq3(grad2) * m2 + magSq3(grad3) * m3;

    const float dlambda = -C / (w + alphaTilde);

    VsOut o;
    o.dc0 = float4(grad1 * dlambda, 0.0f);
    o.dc1 = float4(grad2 * dlambda, 0.0f);
    o.dc2 = float4(grad3 * dlambda, 0.0f);
    gOut[tid] = o;
}
