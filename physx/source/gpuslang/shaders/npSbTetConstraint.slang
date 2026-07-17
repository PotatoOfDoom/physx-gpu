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

// npSbTetConstraint.slang
//
// Slang port of the three soft-body XPBD tetrahedron constraints
// (softBodyGM.cu L722-767), each deriving (C, dCdF) then calling the shared
// compute_deltaLambdaXgradC core (tranche 232, inlined here):
//   ARAP_constraint (L722-736): R=Mat33(tetR); tensor=F-R;
//     C=sqrt(Sum tensor.col.col + 1e-18); dCdF = tensor*(1/C)
//   traceSMinusI_constraint (L739-753): R=Mat33(tetR);
//     C = (R.c0.F.c0 + R.c1.F.c1 + R.c2.F.c2) - 3; dCdF = R
//   volume_constraint (L756-767): C = det(F) - 1;
//     dCdF = (F.c1 x F.c2, F.c2 x F.c0, F.c0 x F.c1)
//
// constraintType selects 0=ARAP, 1=trace, 2=volume.  Mat33(PxQuat) is the
// rotation-matrix constructor (no trig).  All float arith NoContraction-
// stamped; sqrt IEEE; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float pabs(float a) { return a < 0.0f ? -a : a; }

struct TcIn
{
    float4 misc;               // lambda, alphaTilde, dtInv, asfloat(constraintType)
    float4 misc2;              // damping, scaledDamping, _, _
    float4 quat;               // tetR (x,y,z,w)
    float4 fc0, fc1, fc2;      // F columns
    float4 qc0, qc1, qc2;      // DmInv columns
    float4 v0, v1, v2, v3;     // .xyz velocity, .w inverse mass
    float4 acc0, acc1, acc2;   // deltaLambdaXgradC accumulator columns (in)
}

struct TcOut
{
    float4 dc0, dc1, dc2;      // deltaLambdaXgradC columns (out)
    float4 scalars;            // lambda (out), C, _, _
}

struct TcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<TcParams> gParams;

RWStructuredBuffer<TcIn> gIn;    // binding 0
RWStructuredBuffer<TcOut> gOut;  // binding 1

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

[shader("compute")]
[numthreads(64, 1, 1)]
void sbTetConstraint(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const TcIn s = gIn[tid];

    float lambda = s.misc.x;
    const float alphaTilde = s.misc.y, dtInv = s.misc.z;
    const uint ctype = asuint(s.misc.w);
    const float damping = s.misc2.x, scaledDamping = s.misc2.y;

    const float3 Fc0 = s.fc0.xyz, Fc1 = s.fc1.xyz, Fc2 = s.fc2.xyz;

    // Derive (C, dCdF columns) per constraint type.
    float C;
    float3 dFc0, dFc1, dFc2;
    if (ctype == 0u)
    {
        // ARAP
        float3 Rc0, Rc1, Rc2;
        quatToMat(s.quat, Rc0, Rc1, Rc2);
        const float3 t0 = Fc0 - Rc0, t1 = Fc1 - Rc1, t2 = Fc2 - Rc2;
        C = sqrt(dot3(t0, t0) + dot3(t1, t1) + dot3(t2, t2) + 1.0e-18f);
        const float inv = 1.0f / C;
        dFc0 = t0 * inv; dFc1 = t1 * inv; dFc2 = t2 * inv;
    }
    else if (ctype == 1u)
    {
        // trace(S - I), dCdF = R
        float3 Rc0, Rc1, Rc2;
        quatToMat(s.quat, Rc0, Rc1, Rc2);
        C = dot3(Rc0, Fc0) + dot3(Rc1, Fc1) + dot3(Rc2, Fc2) - 3.0f;
        dFc0 = Rc0; dFc1 = Rc1; dFc2 = Rc2;
    }
    else
    {
        // volume: C = det(F) - 1
        const float detF = dot3(Fc0, cross3(Fc1, Fc2));
        C = detF - 1.0f;
        dFc0 = cross3(Fc1, Fc2);
        dFc1 = cross3(Fc2, Fc0);
        dFc2 = cross3(Fc0, Fc1);
    }

    // compute_dCdx (column-major: dCdF/DmInv row vectors are column_k[row]).
    const float3 r0 = float3(s.qc0.x, s.qc1.x, s.qc2.x);
    const float3 r1 = float3(s.qc0.y, s.qc1.y, s.qc2.y);
    const float3 r2 = float3(s.qc0.z, s.qc1.z, s.qc2.z);
    const float3 p0 = float3(dFc0.x, dFc1.x, dFc2.x);
    const float3 p1 = float3(dFc0.y, dFc1.y, dFc2.y);
    const float3 p2 = float3(dFc0.z, dFc1.z, dFc2.z);
    const float3 dCdx1 = float3(dot3(r0, p0), dot3(r0, p1), dot3(r0, p2));
    const float3 dCdx2 = float3(dot3(r1, p0), dot3(r1, p1), dot3(r1, p2));
    const float3 dCdx3 = float3(dot3(r2, p0), dot3(r2, p1), dot3(r2, p2));
    const float3 dCdx0 = -dCdx1 - dCdx2 - dCdx3;

    const float w0 = s.v0.w, w1 = s.v1.w, w2 = s.v2.w, w3 = s.v3.w;
    const float sumWMag = magSq3(dCdx0) * w0 + magSq3(dCdx1) * w1 + magSq3(dCdx2) * w2 + magSq3(dCdx3) * w3;

    TcOut o;

    float deltaLambda;
    if (damping == 0.0f)
    {
        const float denom = sumWMag + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            o.dc0 = s.acc0; o.dc1 = s.acc1; o.dc2 = s.acc2;
            o.scalars = float4(lambda, C, 0.0f, 0.0f);
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
            o.dc0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            o.dc1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            o.dc2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            o.scalars = float4(lambda, C, 0.0f, 0.0f);
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
    o.scalars = float4(lambda, C, 0.0f, 0.0f);
    gOut[tid] = o;
}
