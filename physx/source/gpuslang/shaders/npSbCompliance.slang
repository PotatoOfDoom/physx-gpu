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

// npSbCompliance.slang
//
// Slang port of the soft-body FEM compliance / constraint-gradient helpers
// (softBodyGM.cu L507-555):
//   evalStiffnessMatrix6x6(row, col, invE, nu) (L507-519): the 6x6 isotropic
//     compliance entry.  result=0; if row==col: result=invE, and if row>=3 also
//     result *= (1+nu); else if row<3 && col<3: result = -invE*nu.
//   multiplyStrainByCompliance(invE, nu, strainIndex, lambda) (L521-532):
//     dot of the strainIndex-th compliance row with the spatial vector lambda
//     (top.xyz -> cols 0,1,2; bottom.xyz -> cols 3,4,5).
//   compute_dCdx(dCdF, DmInv) (L536-555): dC/dx = dF/dx : dC/dF.  Row vectors
//     r0/r1/r2 of DmInv and p0/p1/p2 of dCdF; dCdx{1,2,3} = (r_i.p0, r_i.p1,
//     r_i.p2).  (dCdx0 = -dCdx1-dCdx2-dCdx3, not emitted.)
//
// PxMat33 is column-major, so DmInv.column0[0] = c0.x is a row-0 element.
// All float arith NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

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

struct SmIn
{
    float4 params;             // invE, nu, asfloat(strainIndex), _
    float4 ltop, lbot;         // lambda.top.xyz, lambda.bottom.xyz
    float4 fc0, fc1, fc2;      // dCdF columns
    float4 qc0, qc1, qc2;      // DmInv columns
}

struct SmOut
{
    float4 dCdx1, dCdx2, dCdx3;   // .xyz
    float4 scalars;               // multiply, evalDiag, evalNext, _
}

struct SmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SmParams> gParams;

RWStructuredBuffer<SmIn> gIn;    // binding 0
RWStructuredBuffer<SmOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbCompliance(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SmIn s = gIn[tid];

    const float invE = s.params.x, nu = s.params.y;
    const uint si = asuint(s.params.z);
    const float3 ltop = s.ltop.xyz, lbot = s.lbot.xyz;

    const float mult = multiplyStrainByCompliance(invE, nu, si, ltop, lbot);
    const float evalDiag = evalStiffness(si, si, invE, nu);
    const float evalNext = evalStiffness(si, (si + 1u) % 6u, invE, nu);

    // compute_dCdx: row vectors of DmInv and dCdF (column-major -> col_k[row]).
    const float3 r0 = float3(s.qc0.x, s.qc1.x, s.qc2.x);
    const float3 r1 = float3(s.qc0.y, s.qc1.y, s.qc2.y);
    const float3 r2 = float3(s.qc0.z, s.qc1.z, s.qc2.z);
    const float3 p0 = float3(s.fc0.x, s.fc1.x, s.fc2.x);
    const float3 p1 = float3(s.fc0.y, s.fc1.y, s.fc2.y);
    const float3 p2 = float3(s.fc0.z, s.fc1.z, s.fc2.z);

    const float3 dCdx1 = float3(dot3(r0, p0), dot3(r0, p1), dot3(r0, p2));
    const float3 dCdx2 = float3(dot3(r1, p0), dot3(r1, p1), dot3(r1, p2));
    const float3 dCdx3 = float3(dot3(r2, p0), dot3(r2, p1), dot3(r2, p2));

    SmOut o;
    o.dCdx1 = float4(dCdx1, 0.0f);
    o.dCdx2 = float4(dCdx2, 0.0f);
    o.dCdx3 = float4(dCdx3, 0.0f);
    o.scalars = float4(mult, evalDiag, evalNext, 0.0f);
    gOut[tid] = o;
}
