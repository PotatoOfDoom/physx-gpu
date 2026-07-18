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

// npSbStress.slang
//
// Slang port of the sb_calculateStressLaunch compute core (softBodyGM.cu
// L2756-2817): per-tet corotated Cauchy stress + von-Mises coefficient.
// The blockCopy/shared-memory staging, flag guard and gathers are glue; the
// core takes p0..p3, Qinv, the per-tet rotation quaternion (DATA input -- no
// trig) and the material (youngs, poissons):
//
//   u_i = (p_i - p0).xyz;  F = (u1|u2|u3) * Qinv;  R = PxMat33(quat)
//   F = R^T * F
//   lambda = E*nu/((1+nu)*(1-2nu));  mu = (0.5f*E)/(1+nu)
//   lambda += mu*(5.0f/6.0f);  mu = mu*(4.0f/3.0f)   // Pixar Stable-NH reparam
//   eps = (F^T + F)*0.5f - I                         // per column, then diag-1
//   trace = eps00 + eps11 + eps22
//   sigma = eps*2.0*mu + I*lambda*trace              // ((eps*2.0f)*mu) +
//                                                    // ((I*lambda)*trace), per column
//   q12/q23/q31 = diag differences;  constant = 0.70710678118654752440084436210485f
//   stress = PxSqrt((q12*q12 + q23*q23) + q31*q31) * constant
//
// The 2.0 double literal converts exactly to 2.0f at the operator*(float)
// call.  All float arithmetic NoContraction-stamped; sqrt IEEE; +-0 eqFZ.

float3 mat33Transform(float3 c0, float3 c1, float3 c2, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}

// PxMat33(PxQuat) rotation-matrix constructor (columns).
void quatToMat(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
}

struct StIn
{
    float4 p0, p1, p2, p3;   // positions (.w unused)
    float4 qc0, qc1, qc2;    // Qinv columns
    float4 quat;             // tetra rotation quaternion (data)
    float4 mat;              // youngs, poissons, _, _
}

struct StOut
{
    float4 s0, s1, s2;       // sigma columns
    float4 aux;              // stress, _, _, _
}

struct StParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<StParams> gParams;

RWStructuredBuffer<StIn> gIn;    // binding 0
RWStructuredBuffer<StOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbStress(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const StIn s = gIn[tid];

    const float3 u1 = s.p1.xyz - s.p0.xyz;
    const float3 u2 = s.p2.xyz - s.p0.xyz;
    const float3 u3 = s.p3.xyz - s.p0.xyz;

    const float3 Q0 = s.qc0.xyz, Q1 = s.qc1.xyz, Q2 = s.qc2.xyz;

    // F = (u1|u2|u3) * Qinv
    float3 F0 = mat33Transform(u1, u2, u3, Q0);
    float3 F1 = mat33Transform(u1, u2, u3, Q1);
    float3 F2 = mat33Transform(u1, u2, u3, Q2);

    float3 Rc0, Rc1, Rc2;
    quatToMat(s.quat, Rc0, Rc1, Rc2);
    // RTranspose columns = rows of R
    const float3 T0 = float3(Rc0.x, Rc1.x, Rc2.x);
    const float3 T1 = float3(Rc0.y, Rc1.y, Rc2.y);
    const float3 T2 = float3(Rc0.z, Rc1.z, Rc2.z);

    // F = RTranspose * F
    const float3 G0 = mat33Transform(T0, T1, T2, F0);
    const float3 G1 = mat33Transform(T0, T1, T2, F1);
    const float3 G2 = mat33Transform(T0, T1, T2, F2);

    const float E = s.mat.x;
    const float nu = s.mat.y;
    float lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    float mu = 0.5f * E / (1.0f + nu);
    lambda += mu * (5.0f / 6.0f);
    mu = mu * (4.0f / 3.0f);

    // eps = (F^T + F)*0.5f - I : per column (F^T.col_j = row j of F)
    float3 e0 = (float3(G0.x, G1.x, G2.x) + G0) * 0.5f;
    float3 e1 = (float3(G0.y, G1.y, G2.y) + G1) * 0.5f;
    float3 e2 = (float3(G0.z, G1.z, G2.z) + G2) * 0.5f;
    e0.x -= 1.0f;
    e1.y -= 1.0f;
    e2.z -= 1.0f;

    const float trace = e0.x + e1.y + e2.z;

    // sigma = eps*2.0*mu + I*lambda*trace : ((eps*2.0f)*mu).col + ((I*lambda)*trace).col
    // (I*lambda)*trace transcribed verbatim: off-diagonals are (0*lambda)*trace,
    // whose +-0 signs follow lambda/trace exactly as in CUDA.
    const float dg = (1.0f * lambda) * trace;
    const float od = (0.0f * lambda) * trace;
    const float3 s0 = (e0 * 2.0f) * mu + float3(dg, od, od);
    const float3 s1 = (e1 * 2.0f) * mu + float3(od, dg, od);
    const float3 s2 = (e2 * 2.0f) * mu + float3(od, od, dg);

    const float q1 = e0.x;
    const float q2 = e1.y;
    const float q3 = e2.z;
    const float q12 = q1 - q2;
    const float q23 = q2 - q3;
    const float q31 = q3 - q1;
    const float constant = 0.70710678118654752440084436210485f;
    const float stress = sqrt(q12 * q12 + q23 * q23 + q31 * q31) * constant;

    StOut o;
    o.s0 = float4(s0, 0.0f);
    o.s1 = float4(s1, 0.0f);
    o.s2 = float4(s2, 0.0f);
    o.aux = float4(stress, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
