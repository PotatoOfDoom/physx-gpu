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

// npClothArea2D.slang
//
// Slang port of areaConstraint_F2X2 (FEMClothUtil.cuh L376-419, the active
// `#if 1` branch): the XPBD area-conservation constraint
// C = |x01 x x02| / |u01 x u02| - 1, expressed directly from the deformed edge
// vectors and the rest-area.  Unlike the ARAP constraint there is NO
// C>THRESHOLD guard — the delta-lambda update runs unconditionally (only
// queryDeltaLambda's own denom<THRESHOLD guard applies).
//   x01CrossX02      = x01.x*x02.y - x01.y*x02.x
//   undeformedAreaInv= 1/area
//   C  = 0.5*x01CrossX02*undeformedAreaInv - 1
//   grad1 = ( 0.5*undeformedAreaInv*x02.y, -0.5*undeformedAreaInv*x02.x)
//   grad2 = (-0.5*undeformedAreaInv*x01.y,  0.5*undeformedAreaInv*x01.x)
//   grad0 = -grad1 - grad2
//   dl    = queryDeltaLambda(C, grad0, grad1, grad2, alphaTilde, lambda, mi...)
//   lambda += dl ; dx{i} += massInv{i}*dl*grad{i}
//
// queryDeltaLambda (3-vertex no-damping, L211): denom = (Sum mi*|grad|^2)
// + alphaTilde; if denom < 1e-14 return 0; else (-C - alphaTilde*lambda)/denom,
// using the OLD lambda.  All float arith NoContraction-stamped; +-0 eqFZ.

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

float magSq2(float2 v) { return v.x * v.x + v.y * v.y; }

struct AaIn
{
    float4 x01x02;         // x01.xy, x02.xy
    float4 dx0dx1;         // dx0.xy, dx1.xy   (in)
    float4 dx2LamAlpha;    // dx2.x, dx2.y, lambda, alphaTilde
    float4 massArea;       // massInv0, massInv1, massInv2, area
}

struct AaOut
{
    float4 dx0dx1;         // updated dx0.xy, dx1.xy
    float4 dx2Lam;         // dx2.x, dx2.y, lambda, _
}

struct AaParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AaParams> gParams;

RWStructuredBuffer<AaIn> gIn;    // binding 0
RWStructuredBuffer<AaOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothArea2D(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AaIn s = gIn[tid];

    const float2 x01 = float2(s.x01x02.x, s.x01x02.y);
    const float2 x02 = float2(s.x01x02.z, s.x01x02.w);
    float2 dx0 = float2(s.dx0dx1.x, s.dx0dx1.y);
    float2 dx1 = float2(s.dx0dx1.z, s.dx0dx1.w);
    float2 dx2 = float2(s.dx2LamAlpha.x, s.dx2LamAlpha.y);
    float lambda = s.dx2LamAlpha.z;
    const float alphaTilde = s.dx2LamAlpha.w;
    const float massInv0 = s.massArea.x, massInv1 = s.massArea.y, massInv2 = s.massArea.z;
    const float area = s.massArea.w;

    const float x01CrossX02 = x01.x * x02.y - x01.y * x02.x;
    const float undeformedAreaInv = 1.0f / area;

    const float C = 0.5f * x01CrossX02 * undeformedAreaInv - 1.0f;

    const float2 grad1 = float2(0.5f * undeformedAreaInv * x02.y, -0.5f * undeformedAreaInv * x02.x);
    const float2 grad2 = float2(-0.5f * undeformedAreaInv * x01.y, 0.5f * undeformedAreaInv * x01.x);
    const float2 grad0 = -grad1 - grad2;

    // queryDeltaLambda (3-vertex, no damping) with the OLD lambda
    const float denom = (massInv0 * magSq2(grad0) + massInv1 * magSq2(grad1) + massInv2 * magSq2(grad2)) + alphaTilde;
    float deltaLambda;
    if (denom < FEMCLOTH_THRESHOLD)
        deltaLambda = 0.0f;
    else
        deltaLambda = (-C - alphaTilde * lambda) / denom;

    lambda = lambda + deltaLambda;

    dx0 = dx0 + massInv0 * deltaLambda * grad0;
    dx1 = dx1 + massInv1 * deltaLambda * grad1;
    dx2 = dx2 + massInv2 * deltaLambda * grad2;

    AaOut o;
    o.dx0dx1 = float4(dx0.x, dx0.y, dx1.x, dx1.y);
    o.dx2Lam = float4(dx2.x, dx2.y, lambda, 0.0f);
    gOut[tid] = o;
}
