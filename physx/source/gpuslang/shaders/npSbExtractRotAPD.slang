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

// npSbExtractRotAPD.slang
//
// Slang port of sb_extractRotationAPD (deformableUtils.cuh L69-116): the
// analytic-polar-decomposition rotation extractor -- per iteration it forms
// B = R(q)^T * F, the Rodrigues gradient and 3x3 Hessian of the rotation
// energy, solves the damped Newton step omega, and applies the Cayley-style
// quaternion update q *= (vec, w).  Unlike extractRotation (L46-67, the
// Mueller iterative variant), this function contains NO transcendentals --
// extractRotation's PxQuat(angle, unitAxis) constructor evaluates sinf/cosf,
// which have no cross-backend rounding guarantee, so that variant is
// documented NON-PORTABLE and this one is the portable rotation extractor.
//
// TWO SPOTS ARE DOUBLE-PRECISION in the CUDA source and are transcribed in f64
// (requires shaderFloat64, enabled by the backend when the device offers it):
//   L87:  if (i == 0 && gradient.magnitudeSquared() < 1e-16)   // f64 compare
//   L109: w = (1.0 - l_omega2) / (1.0f + l_omega2)             // f64 sub + div,
//         (denominator computed in f32 first, then promoted), result -> f32
// The f64 constant 1e-16 is constructed as 1.0/1e16 with 1e16 exact in f64
// (1e16 = 5^16 * 2^16, mantissa 5^16 < 2^38): division of exact operands is
// correctly rounded, yielding exactly the double literal value.
//
// PxQuat operator* (q1*q2): x = q1.w*q2.x + q2.w*q1.x + q1.y*q2.z - q2.y*q1.z
// (and cyclic), w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z, each a
// left-to-right fold.  PxMat33(PxQuat) is the shared x2/y2/z2 form.
// PxMat33 A*F: col_i = A.transform(F.col_i) = (A.c0*v.x + A.c1*v.y) + A.c2*v.z.
// detH and the omega solve keep the exact source association.  All float
// arithmetic (f32 and f64) NoContraction-stamped; quaternion compared eqFZ.

float4 quatConj(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }

// PxQuat operator* (PxQuat.h L388-392), left-to-right folds.
float4 quatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
                  a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
                  a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

// PxMat33(PxQuat) rotation ctor (PxMat33.h L136-163); columns out.
void mat33FromQuat(float4 q, out float3 c0, out float3 c1, out float3 c2)
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

// PxMat33::transform(v) = col0*v.x + col1*v.y + col2*v.z, left-to-right.
float3 mat33Transform(float3 c0, float3 c1, float3 c2, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct ArIn
{
    float4 fc0;   // F.column0.xyz
    float4 fc1;   // F.column1.xyz
    float4 fc2;   // F.column2.xyz
    float4 q0;    // initial quaternion (x,y,z,w)
    uint4 it;     // maxIter, _, _, _
}

struct ArOut
{
    float4 q;     // final quaternion
}

struct ArParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ArParams> gParams;

RWStructuredBuffer<ArIn> gIn;    // binding 0
RWStructuredBuffer<ArOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbExtractRotAPD(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const ArIn s = gIn[tid];

    const float3 F0 = s.fc0.xyz, F1 = s.fc1.xyz, F2 = s.fc2.xyz;
    float4 q = s.q0;
    const int maxIter = int(s.it.x);

    const float eps = 1.0e-6f;          // PxReal eps = 1.0e-6 (double literal -> same float)
    const float threshold = 1.0f - eps;

    // exact f64 value of the literal 1e-16 (see header comment)
    const double D1E16 = double(10000000000000000ULL);   // exact in f64
    const double D1E_16 = double(1.0f) / D1E16;          // correctly-rounded 1e-16

    for (int i = 0; i < maxIter; ++i)
    {
        // B = PxMat33(q.getConjugate()) * F
        float3 r0, r1, r2;
        mat33FromQuat(quatConj(q), r0, r1, r2);
        const float3 B0 = mat33Transform(r0, r1, r2, F0);
        const float3 B1 = mat33Transform(r0, r1, r2, F1);
        const float3 B2 = mat33Transform(r0, r1, r2, F2);

        float3 gradient = float3(B2.y - B1.z, B0.z - B2.x, B1.x - B0.y);
        if (i == 0 && double(dot3(gradient, gradient)) < D1E_16)
        {
            // stuck in the first iteration (e.g. 180 deg rotation): distort gradient
            gradient = float3(-2.0f, 0.0f, 0.0f);
        }
        const float h00 = B1.y + B2.z;
        const float h11 = B0.x + B2.z;
        const float h22 = B0.x + B1.y;
        const float h01 = -0.5f * (B1.x + B0.y);
        const float h02 = -0.5f * (B2.x + B0.z);
        const float h12 = -0.5f * (B2.y + B1.z);
        const float detH = -h02 * h02 * h11 + 2.0f * h01 * h02 * h12 - h00 * h12 * h12 - h01 * h01 * h22 + h00 * h11 * h22;
        float3 omega;
        const float factor = -0.25f / detH;
        omega.x = factor * ((h11 * h22 - h12 * h12) * gradient.x + (h02 * h12 - h01 * h22) * gradient.y + (h01 * h12 - h02 * h11) * gradient.z);
        omega.y = factor * ((h02 * h12 - h01 * h22) * gradient.x + (h00 * h22 - h02 * h02) * gradient.y + (h01 * h02 - h00 * h12) * gradient.z);
        omega.z = factor * ((h01 * h12 - h02 * h11) * gradient.x + (h01 * h02 - h00 * h12) * gradient.y + (h00 * h11 - h01 * h01) * gradient.z);
        if (abs(detH) < 1e-9f)
            omega = -gradient;
        if (dot3(omega, gradient) > 0.0f)
            omega = gradient * -0.125f;
        const float l_omega2 = dot3(omega, omega);
        // w = (1.0 - l_omega2) / (1.0f + l_omega2): f64 subtraction, f32
        // denominator promoted to f64, f64 division, result narrowed to f32
        const float w = float((double(1.0f) - double(l_omega2)) / double(1.0f + l_omega2));
        const float3 vec = omega * (2.0f / (1.0f + l_omega2));
        q = quatMul(q, float4(vec.x, vec.y, vec.z, w));
        if (w > threshold)
            break;
    }

    ArOut o;
    o.q = q;
    gOut[tid] = o;
}
