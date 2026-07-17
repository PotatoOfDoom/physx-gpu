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

// npSbDeltaLambdaGradC.slang
//
// Slang port of compute_deltaLambdaXgradC (softBodyGM.cu L558-604): the XPBD
// per-constraint delta-lambda accumulation for a soft-body tetrahedron.
// Composes compute_dCdx (tranche 231).
//   dCdx{1,2,3} = compute_dCdx(dCdF, DmInv) ; dCdx0 = -dCdx1-dCdx2-dCdx3
//   no damping (damping==0):
//     denom = |dCdx0|^2*v0.w + |dCdx1|^2*v1.w + |dCdx2|^2*v2.w + |dCdx3|^2*v3.w
//             + alphaTilde
//     if |denom| < 1e-15: RETURN unchanged (accumulator & lambda kept)
//     deltaLambda = (-C - alphaTilde*lambda) / denom
//   with damping:
//     denom = (1 + damping*dtInv)*(Sum |dCdx_i|^2 * v_i.w) + alphaTilde
//     if |denom| < 1e-15: SET accumulator = 0 and RETURN (lambda kept)
//     deltaLambda = (-C - alphaTilde*lambda
//                    - scaledDamping*(dCdx0.vel0 + dCdx1.vel1 + dCdx2.vel2 + dCdx3.vel3))
//                   / denom
//   lambda += deltaLambda
//   deltaLambdaXgradC.col{0,1,2} += deltaLambda * dCdx{1,2,3}   (ACCUMULATED)
//
// NOTE the two guard-fail paths differ: no-damping leaves the accumulator
// untouched; damping zeroes it.  v_i carry velocity in .xyz and inverse mass in
// .w.  PxMat33 column-major.  All float arith NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float pabs(float a) { return a < 0.0f ? -a : a; }

struct DlIn
{
    float4 misc;               // C, lambda, alphaTilde, dtInv
    float4 misc2;              // damping, scaledDamping, _, _
    float4 fc0, fc1, fc2;      // dCdF columns
    float4 qc0, qc1, qc2;      // DmInv columns
    float4 v0, v1, v2, v3;     // .xyz velocity, .w inverse mass
    float4 acc0, acc1, acc2;   // deltaLambdaXgradC accumulator columns (in)
}

struct DlOut
{
    float4 dc0, dc1, dc2;      // deltaLambdaXgradC columns (out)
    float4 scalars;            // lambda (out), _, _, _
}

struct DlParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<DlParams> gParams;

RWStructuredBuffer<DlIn> gIn;    // binding 0
RWStructuredBuffer<DlOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbDeltaLambdaGradC(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const DlIn s = gIn[tid];

    const float C = s.misc.x, alphaTilde = s.misc.z, dtInv = s.misc.w;
    float lambda = s.misc.y;
    const float damping = s.misc2.x, scaledDamping = s.misc2.y;

    // compute_dCdx (column-major: row vectors are column_k[row])
    const float3 r0 = float3(s.qc0.x, s.qc1.x, s.qc2.x);
    const float3 r1 = float3(s.qc0.y, s.qc1.y, s.qc2.y);
    const float3 r2 = float3(s.qc0.z, s.qc1.z, s.qc2.z);
    const float3 p0 = float3(s.fc0.x, s.fc1.x, s.fc2.x);
    const float3 p1 = float3(s.fc0.y, s.fc1.y, s.fc2.y);
    const float3 p2 = float3(s.fc0.z, s.fc1.z, s.fc2.z);
    const float3 dCdx1 = float3(dot3(r0, p0), dot3(r0, p1), dot3(r0, p2));
    const float3 dCdx2 = float3(dot3(r1, p0), dot3(r1, p1), dot3(r1, p2));
    const float3 dCdx3 = float3(dot3(r2, p0), dot3(r2, p1), dot3(r2, p2));
    const float3 dCdx0 = -dCdx1 - dCdx2 - dCdx3;

    const float w0 = s.v0.w, w1 = s.v1.w, w2 = s.v2.w, w3 = s.v3.w;
    const float sumWMag = magSq3(dCdx0) * w0 + magSq3(dCdx1) * w1 + magSq3(dCdx2) * w2 + magSq3(dCdx3) * w3;

    DlOut o;

    float deltaLambda;
    if (damping == 0.0f)
    {
        const float denom = sumWMag + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            // return unchanged: accumulator and lambda kept
            o.dc0 = s.acc0;
            o.dc1 = s.acc1;
            o.dc2 = s.acc2;
            o.scalars = float4(lambda, 0.0f, 0.0f, 0.0f);
            gOut[tid] = o;
            return;
        }
        deltaLambda = (-C - alphaTilde * lambda) / denom;
    }
    else
    {
        const float denom = (1.0f + damping * dtInv) * sumWMag + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            // zero the accumulator, keep lambda
            o.dc0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            o.dc1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            o.dc2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            o.scalars = float4(lambda, 0.0f, 0.0f, 0.0f);
            gOut[tid] = o;
            return;
        }
        const float dampDot = dot3(dCdx0, s.v0.xyz) + dot3(dCdx1, s.v1.xyz) + dot3(dCdx2, s.v2.xyz) + dot3(dCdx3, s.v3.xyz);
        deltaLambda = (-C - alphaTilde * lambda - scaledDamping * dampDot) / denom;
    }

    lambda = lambda + deltaLambda;

    o.dc0 = float4(s.acc0.xyz + deltaLambda * dCdx1, 0.0f);
    o.dc1 = float4(s.acc1.xyz + deltaLambda * dCdx2, 0.0f);
    o.dc2 = float4(s.acc2.xyz + deltaLambda * dCdx3, 0.0f);
    o.scalars = float4(lambda, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
