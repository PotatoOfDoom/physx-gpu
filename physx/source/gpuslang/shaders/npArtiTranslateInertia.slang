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

// npArtiTranslateInertia.slang
//
// Slang port of translateInertia(const PxMat33& inertia, const PxReal mass,
// const PxVec3& t) (forwardDynamic2.cu L1569-1577): the parallel-axis theorem
// translate of a 3x3 inertia by offset t and mass.
//   s = skewSymmetric(t) = [[0,-t.z,t.y],[t.z,0,-t.x],[-t.y,t.x,0]]
//                          (columns: (0,t.z,-t.y), (-t.z,0,t.x), (t.y,-t.x,0))
//   translatedIT = s.getTranspose() * s * mass + inertia
// PxMat33 column-major: A*B = (A.transform(B.col_i)); transform(v) = col0*v.x +
// col1*v.y + col2*v.z; getTranspose swaps rows<->columns; *scalar and + are
// columnwise.  All float arith NoContraction-stamped; +-0 eqFZ.

struct M3 { float3 c0, c1, c2; }

float3 mtx(M3 m, float3 v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z; }
M3 matmul(M3 a, M3 b)
{
    M3 r; r.c0 = mtx(a, b.c0); r.c1 = mtx(a, b.c1); r.c2 = mtx(a, b.c2); return r;
}
M3 transpose3(M3 m)
{
    M3 r;
    r.c0 = float3(m.c0.x, m.c1.x, m.c2.x);
    r.c1 = float3(m.c0.y, m.c1.y, m.c2.y);
    r.c2 = float3(m.c0.z, m.c1.z, m.c2.z);
    return r;
}
M3 scale3(M3 m, float s) { M3 r; r.c0 = m.c0 * s; r.c1 = m.c1 * s; r.c2 = m.c2 * s; return r; }
M3 add3(M3 a, M3 b) { M3 r; r.c0 = a.c0 + b.c0; r.c1 = a.c1 + b.c1; r.c2 = a.c2 + b.c2; return r; }

struct TiIn
{
    float4 iC0, iC1, iC2;   // inertia columns
    float4 tMass;           // t.xyz, mass in .w
}

struct TiOut { float4 c0, c1, c2; }   // translatedIT columns

struct TiParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<TiParams> gParams;

RWStructuredBuffer<TiIn> gIn;    // binding 0
RWStructuredBuffer<TiOut> gOut;  // binding 1

M3 mk(float4 c0, float4 c1, float4 c2) { M3 m; m.c0 = c0.xyz; m.c1 = c1.xyz; m.c2 = c2.xyz; return m; }

[shader("compute")]
[numthreads(64, 1, 1)]
void transInertia(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const TiIn inp = gIn[tid];

    const M3 inertia = mk(inp.iC0, inp.iC1, inp.iC2);
    const float3 t = inp.tMass.xyz;
    const float mass = inp.tMass.w;

    // s = skewSymmetric(t)  (columns as in the CUDA PxMat33 ctor)
    M3 s;
    s.c0 = float3(0.0f, t.z, -t.y);
    s.c1 = float3(-t.z, 0.0f, t.x);
    s.c2 = float3(t.y, -t.x, 0.0f);

    const M3 sT = transpose3(s);
    const M3 sTs = matmul(sT, s);
    const M3 scaled = scale3(sTs, mass);
    const M3 res = add3(scaled, inertia);

    TiOut o;
    o.c0 = float4(res.c0, 0.0f);
    o.c1 = float4(res.c1, 0.0f);
    o.c2 = float4(res.c2, 0.0f);
    gOut[tid] = o;
}
