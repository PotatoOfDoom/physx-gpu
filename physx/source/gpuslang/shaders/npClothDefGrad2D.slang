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

// npClothDefGrad2D.slang
//
// Slang port of the FEM-cloth 2x2 deformation-gradient primitives from
// FEMClothUtil.cuh: queryDeformationGradient_F2x2 (L299-304) and
// queryConstraintGradient_F2x2 (L312-317).
//   deformation gradient F (columns f0,f1):
//     f0 = QInv.x*xp01 + QInv.y*xp02;  f1 = QInv.z*xp01 + QInv.w*xp02
//   constraint gradient (grad1,grad2) = QInv^T applied to (pC_pF0,pC_pF1):
//     grad1 = qInv.x*pC_pF0 + qInv.z*pC_pF1;  grad2 = qInv.y*pC_pF0 + qInv.w*pC_pF1
//
// All PxVec2 componentwise scalar-multiply-add.  NoContraction-stamped; +-0 eqFZ.

struct DgIn
{
    float4 qinvDef;   // QInv for deformation gradient (x,y,z,w)
    float4 xp;        // xp01.xy, xp02.xy
    float4 qinvCon;   // qInv for constraint gradient (x,y,z,w)
    float4 pcpf;      // pC_pF0.xy, pC_pF1.xy
}

struct DgOut
{
    float4 f;         // f0.xy, f1.xy
    float4 grad;      // grad1.xy, grad2.xy
}

struct DgParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<DgParams> gParams;

RWStructuredBuffer<DgIn> gIn;    // binding 0
RWStructuredBuffer<DgOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothDefGrad2D(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const DgIn s = gIn[tid];

    const float4 QInv = s.qinvDef;
    const float2 xp01 = s.xp.xy, xp02 = s.xp.zw;
    const float2 f0 = QInv.x * xp01 + QInv.y * xp02;
    const float2 f1 = QInv.z * xp01 + QInv.w * xp02;

    const float4 qInv = s.qinvCon;
    const float2 pC_pF0 = s.pcpf.xy, pC_pF1 = s.pcpf.zw;
    const float2 grad1 = qInv.x * pC_pF0 + qInv.z * pC_pF1;
    const float2 grad2 = qInv.y * pC_pF0 + qInv.w * pC_pF1;

    DgOut o;
    o.f = float4(f0, f1);
    o.grad = float4(grad1, grad2);
    gOut[tid] = o;
}
