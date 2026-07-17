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

// npSbComputeLambda.slang
//
// Slang port of the two computeLambda overloads (softBodyGM.cu L606-617),
// which compose multiplyStrainByCompliance (tranche 231, inlined).
//   strain overload (L606-611):
//     -(G(a,b) + invDtSq*multiplyStrainByCompliance(invE,nu,strainIndex,lambda)
//       + gamma*E(a,b)) * scale
//   plain overload (L613-617):
//     -(G(a,b) + gamma*E(a,b)) * scale
//   G(a,b) = element (row a, col b) = column_b[a] (PxMat33 operator()(row,col)).
//
// useStrain selects the overload.  All float arith NoContraction-stamped; +-0 eqFZ.

float evalStiffness(uint row, uint col, float invE, float nu)
{
    float result = 0.0f;
    if (row == col)
    {
        result = invE;
        if (row >= 3u)
            result = result * (1.0f + nu);
    }
    else if (row < 3u && col < 3u)
    {
        result = -invE * nu;
    }
    return result;
}

float multiplyStrainByCompliance(float invE, float nu, uint si, float3 ltop, float3 lbot)
{
    return ltop.x * evalStiffness(si, 0u, invE, nu)
         + ltop.y * evalStiffness(si, 1u, invE, nu)
         + ltop.z * evalStiffness(si, 2u, invE, nu)
         + lbot.x * evalStiffness(si, 3u, invE, nu)
         + lbot.y * evalStiffness(si, 4u, invE, nu)
         + lbot.z * evalStiffness(si, 5u, invE, nu);
}

// PxMat33 operator()(row,col) = column[col][row].
float matElem(float3 c0, float3 c1, float3 c2, uint row, uint col)
{
    const float3 c = (col == 0u) ? c0 : (col == 1u) ? c1 : c2;
    return c[row];
}

struct ClIn
{
    float4 params;    // gamma, scale, invE, nu
    float4 params2;   // invDtSq, asfloat(indexA), asfloat(indexB), asfloat(strainIndex)
    float4 flags;     // asfloat(useStrain), _, _, _
    float4 gc0, gc1, gc2;   // G columns
    float4 ec0, ec1, ec2;   // E columns
    float4 ltop, lbot;      // totalLambda top / bottom
}

struct ClOut
{
    float4 scalars;   // lambda result, mult, _, _
}

struct ClParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ClParams> gParams;

RWStructuredBuffer<ClIn> gIn;    // binding 0
RWStructuredBuffer<ClOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbComputeLambda(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const ClIn s = gIn[tid];

    const float gamma = s.params.x, scale = s.params.y, invE = s.params.z, nu = s.params.w;
    const float invDtSq = s.params2.x;
    const uint indexA = asuint(s.params2.y), indexB = asuint(s.params2.z), strainIndex = asuint(s.params2.w);
    const bool useStrain = asuint(s.flags.x) == 1u;

    const float Gab = matElem(s.gc0.xyz, s.gc1.xyz, s.gc2.xyz, indexA, indexB);
    const float Eab = matElem(s.ec0.xyz, s.ec1.xyz, s.ec2.xyz, indexA, indexB);

    const float mult = multiplyStrainByCompliance(invE, nu, strainIndex, s.ltop.xyz, s.lbot.xyz);

    float lambda;
    if (useStrain)
        lambda = -(Gab + invDtSq * mult + gamma * Eab) * scale;
    else
        lambda = -(Gab + gamma * Eab) * scale;

    ClOut o;
    o.scalars = float4(lambda, mult, 0.0f, 0.0f);
    gOut[tid] = o;
}
