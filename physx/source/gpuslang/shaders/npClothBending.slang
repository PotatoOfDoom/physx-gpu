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

// npClothBending.slang
//
// Slang port of bendingEnergySolvePerTrianglePair (FEMClothUtil.cuh L543-655):
// the FEM-cloth "Discrete Shells" XPBD dihedral bending constraint.
//
// THE atan2f SEAM: the CUDA code evaluates `angle = atan2f(sinAngle, cosAngle)`
// with the libm transcendental, which has NO cross-backend rounding guarantee
// (same non-portable class as the sinf/cosf in extractRotation, documented in
// MIGRATION_STATUS).  The port therefore takes the atan2 RESULT as an input
// (`angleIn`, produced by the host from the identical sinAngle/cosAngle) and
// PINS everything upstream by also OUTPUTTING the shader's own sinAngle and
// cosAngle for bitwise comparison.  Everything else is ported verbatim:
//
//   x02 = x2-x0; x03 = x3-x0; x13 = x3-x1; x12 = x2-x1; x23 = x3-x2
//   x23Len = x23.magnitude()               // sqrt(magSq), IEEE
//   if (x23Len < FEMCLOTH_THRESHOLD /*1e-14f*/) return;   // outputs unchanged
//   restBendingAngle = slot.x; kInv = slot.y  (slot.z damping is UNUSED)
//   if (kInv <= 0) return;                                // outputs unchanged
//   scaledN0 = x02 x x03; scaledN1 = x13 x x12
//   n0LenInv = 1/scaledN0.magnitude(); n1LenInv = 1/scaledN1.magnitude()
//   n0 = scaledN0*n0LenInv; n1 = scaledN1*n1LenInv
//   cosAngle = n0.n1; sinAngle = (n0 x n1).x23Normalized
//   angle = atan2f(sinAngle, cosAngle)     <- SEAM, taken as input
//   alphaTilde = kInv*dtInv*dtInv (dtInv = 1/dt);  C = angle - restBendingAngle
//   if |C+2PI| < |C| then C += 2PI else if |C-2PI| < |C| then C -= 2PI
//   C = PxClamp(C, -HALF_PI, HALF_PI)      // PxMin(hi, PxMax(lo, v))
//   temp0 = n0*n0LenInv; temp1 = n1*n1LenInv
//   dCdx0 = -x23Len*temp0; dCdx1 = -x23Len*temp1
//   dCdx2 = x03.x23n*temp0 + x13.x23n*temp1
//   dCdx3 = -(x02.x23n*temp0 + x12.x23n*temp1)
//   lambda = isTGS ? 0 : slot load
//   deltaLambda = queryDeltaLambda 4-vertex no-damping (mass-first products,
//     parenthesized 4-sum + alphaTilde, denom < 1e-14 -> 0,
//     (-C - alphaTilde*lambda)/denom)
//   if (!isTGS) store lambda + deltaLambda
//   scale_i = (vrc_i * x_i.w) * deltaLambda;  x_i.xyz += scale_i * dCdx_i
//
// The isSharedTrianglePartition slot selection collapses to one in/out scalar.
// x_i.w (inverse mass) is read-only.  All float arithmetic NoContraction-
// stamped; sqrt IEEE; +-0 compared with eqFZ.

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;
static const float FEMCLOTH_HALF_PI = 1.57079632679489661923f;
static const float FEMCLOTH_2PI = 6.28318530717958647692f;

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float pabs(float a) { return a < 0.0f ? -a : a; }
float selMin(float a, float b) { return a < b ? a : b; }
float selMax(float a, float b) { return a > b ? a : b; }

struct CbIn
{
    float4 x0, x1, x2, x3;   // positions .xyz, inverse masses .w
    float4 vrc;              // vertexReferenceCounts
    float4 bend;             // restBendingAngle, kInv, dt, asfloat(isTGS)
    float4 misc;             // lambda in, angleIn (host atan2f of sin/cos), _, _
}

struct CbOut
{
    float4 x0, x1, x2, x3;   // updated positions .xyz (.w echoed)
    float4 aux;              // lambdaOut, sinAngle, cosAngle, asfloat(pathTaken)
}

struct CbParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CbParams> gParams;

RWStructuredBuffer<CbIn> gIn;    // binding 0
RWStructuredBuffer<CbOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothBending(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CbIn s = gIn[tid];

    const float restBendingAngle = s.bend.x;
    const float kInv = s.bend.y;
    const float dt = s.bend.z;
    const bool isTGS = asuint(s.bend.w) == 1u;

    CbOut o;
    o.x0 = s.x0; o.x1 = s.x1; o.x2 = s.x2; o.x3 = s.x3;

    const float3 x02 = s.x2.xyz - s.x0.xyz;
    const float3 x03 = s.x3.xyz - s.x0.xyz;
    const float3 x13 = s.x3.xyz - s.x1.xyz;
    const float3 x12 = s.x2.xyz - s.x1.xyz;
    const float3 x23 = s.x3.xyz - s.x2.xyz;
    const float x23Len = sqrt(magSq3(x23));

    if (x23Len < FEMCLOTH_THRESHOLD)
    {
        o.aux = float4(s.misc.x, 0.0f, 0.0f, asfloat(1u));
        gOut[tid] = o;
        return;
    }

    if (kInv <= 0.0f)
    {
        o.aux = float4(s.misc.x, 0.0f, 0.0f, asfloat(2u));
        gOut[tid] = o;
        return;
    }

    const float x23LenInv = 1.0f / x23Len;
    const float3 x23Normalized = x23 * x23LenInv;

    const float3 scaledN0 = cross3(x02, x03);
    const float3 scaledN1 = cross3(x13, x12);

    const float n0LenInv = 1.0f / sqrt(magSq3(scaledN0));
    const float n1LenInv = 1.0f / sqrt(magSq3(scaledN1));

    const float3 n0 = scaledN0 * n0LenInv;
    const float3 n1 = scaledN1 * n1LenInv;

    const float cosAngle = dot3(n0, n1);
    const float sinAngle = dot3(cross3(n0, n1), x23Normalized);
    // angle = atan2f(sinAngle, cosAngle) -- the non-portable seam, taken as
    // input (the host computes it from the SAME sinAngle/cosAngle, which are
    // output below and compared bitwise).
    const float angle = s.misc.y;

    const float dtInv = 1.0f / dt;
    const float alphaTilde = kInv * dtInv * dtInv;
    float C = angle - restBendingAngle;

    if (pabs(C + FEMCLOTH_2PI) < pabs(C))
    {
        C += FEMCLOTH_2PI;
    }
    else if (pabs(C - FEMCLOTH_2PI) < pabs(C))
    {
        C -= FEMCLOTH_2PI;
    }

    // PxClamp = PxMin(hi, PxMax(lo, v))
    C = selMin(FEMCLOTH_HALF_PI, selMax(-FEMCLOTH_HALF_PI, C));

    const float3 temp0 = n0 * n0LenInv;
    const float3 temp1 = n1 * n1LenInv;
    const float3 dCdx0 = -x23Len * temp0;
    const float3 dCdx1 = -x23Len * temp1;
    const float3 dCdx2 = dot3(x03, x23Normalized) * temp0 + dot3(x13, x23Normalized) * temp1;
    const float3 dCdx3 = -(dot3(x02, x23Normalized) * temp0 + dot3(x12, x23Normalized) * temp1);

    float lambda = 0.0f;

    if (!isTGS)
    {
        lambda = s.misc.x;
    }

    // queryDeltaLambda (4-vertex, no damping, FEMClothUtil.cuh L249-262)
    const float m0 = s.vrc.x * s.x0.w;
    const float m1 = s.vrc.y * s.x1.w;
    const float m2 = s.vrc.z * s.x2.w;
    const float m3 = s.vrc.w * s.x3.w;
    const float denom = (m0 * magSq3(dCdx0) + m1 * magSq3(dCdx1) + m2 * magSq3(dCdx2) + m3 * magSq3(dCdx3)) + alphaTilde;
    float deltaLambda;
    if (denom < FEMCLOTH_THRESHOLD)
        deltaLambda = 0.0f;
    else
        deltaLambda = (-C - alphaTilde * lambda) / denom;

    float lambdaOut = s.misc.x;
    if (!isTGS)
    {
        lambdaOut = lambda + deltaLambda;
    }

    const float scale0 = s.vrc.x * s.x0.w * deltaLambda;
    const float scale1 = s.vrc.y * s.x1.w * deltaLambda;
    const float scale2 = s.vrc.z * s.x2.w * deltaLambda;
    const float scale3 = s.vrc.w * s.x3.w * deltaLambda;

    o.x0 = float4(s.x0.x + scale0 * dCdx0.x, s.x0.y + scale0 * dCdx0.y, s.x0.z + scale0 * dCdx0.z, s.x0.w);
    o.x1 = float4(s.x1.x + scale1 * dCdx1.x, s.x1.y + scale1 * dCdx1.y, s.x1.z + scale1 * dCdx1.z, s.x1.w);
    o.x2 = float4(s.x2.x + scale2 * dCdx2.x, s.x2.y + scale2 * dCdx2.y, s.x2.z + scale2 * dCdx2.z, s.x2.w);
    o.x3 = float4(s.x3.x + scale3 * dCdx3.x, s.x3.y + scale3 * dCdx3.y, s.x3.z + scale3 * dCdx3.z, s.x3.w);
    o.aux = float4(lambdaOut, sinAngle, cosAngle, asfloat(0u));
    gOut[tid] = o;
}
