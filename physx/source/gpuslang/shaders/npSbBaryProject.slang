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

// npSbBaryProject.slang
//
// Slang port of the soft-body barycentric projection / mass family
// (softBody.cu L608-648) plus getSoftBodyInvMass (softBody.cuh L38-42):
//   barycentricProject(tet)  = a*bc.x + b*bc.y + c*bc.z + d*bc.w   (float4)
//   barycentricProjectTri    = a*bc.x + b*bc.y + c*bc.z            (float4)
//   computeInvMass(bary)     = (a.w*bc.x + b.w*bc.y + c.w*bc.z + d.w*bc.w)
//                              * sqrt(bc.x^2 + bc.y^2 + bc.z^2 + bc.w^2)
//                              (= getSoftBodyInvMass of the interpolated mass)
// (computeTriInvMass = barycentricProjectTri.w, and the tet invMass before the
// sqrt scale = barycentricProject.w, so those are covered by the float4 outputs.)
//
// a/b/c/d carry position in .xyz and inverse mass in .w.  All float arith
// NoContraction-stamped; sqrt is IEEE (bit-exact portable); +-0 eqFZ.

struct BpIn
{
    float4 a, b, c, d;   // .xyz position, .w inverse mass
    float4 bary;         // barycentric weights
}

struct BpOut
{
    float4 projTet;      // a*bc.x + b*bc.y + c*bc.z + d*bc.w
    float4 projTri;      // a*bc.x + b*bc.y + c*bc.z
    float4 scalars;      // tetInvMass (computeInvMass), _, _, _
}

struct BpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BpParams> gParams;

RWStructuredBuffer<BpIn> gIn;    // binding 0
RWStructuredBuffer<BpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbBaryProject(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const BpIn s = gIn[tid];

    const float4 a = s.a, b = s.b, c = s.c, d = s.d;
    const float4 bc = s.bary;

    const float4 projTet = a * bc.x + b * bc.y + c * bc.z + d * bc.w;
    const float4 projTri = a * bc.x + b * bc.y + c * bc.z;

    // computeInvMass(bary): barycentric-weighted inverse mass, scaled by
    // getSoftBodyInvMass's sqrt(sum bary^2).
    const float invMass = a.w * bc.x + b.w * bc.y + c.w * bc.z + d.w * bc.w;
    const float scale = sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z + bc.w * bc.w);
    const float tetInvMass = invMass * scale;

    BpOut o;
    o.projTet = projTet;
    o.projTri = projTri;
    o.scalars = float4(tetInvMass, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
