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

// npArtiSpatialMatVec.slang
//
// Slang port of Dy::SpatialMatrix::operator*(const Cm::UnAlignedSpatialVector&)
// (DyFeatherstoneArticulationUtils.h L276-282): the spatial-inertia matrix
// applied to a spatial vector, used throughout the articulation dynamics (e.g.
// PxcFsFlushVelocity's invInertia * deferredZ).
//
// The SpatialMatrix is the block matrix [[topLeft, topRight],[bottomLeft,
// topLeft^T]] (the bottom-right block is topLeft transposed, getBottomRight
// L246-249).  Applied to s = (top, bottom):
//   top    = topLeft * s.top + topRight * s.bottom
//   bottom = bottomLeft * s.top + topLeft.transformTranspose(s.bottom)
// with (top=angular, bottom=linear).
//
// PxMat33 (column-major, PxMat33.h):
//   M.transform(v)          = column0*v.x + column1*v.y + column2*v.z   (L278)
//   M.transformTranspose(v) = (column0.dot(v), column1.dot(v), column2.dot(v)) (L350)
// ** TRAP: the bottom term uses topLeft^T (transformTranspose), not topLeft or
//    bottomLeft. **  All float arith NoContraction-stamped; +-0 eqFZ.

struct SmIn
{
    float4 tlC0, tlC1, tlC2;   // topLeft columns 0,1,2
    float4 trC0, trC1, trC2;   // topRight columns
    float4 blC0, blC1, blC2;   // bottomLeft columns
    float4 sTop, sBot;         // spatial vector top / bottom
}

struct SmOut
{
    float4 outTop, outBot;
}

struct SmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SmParams> gParams;

RWStructuredBuffer<SmIn> gIn;    // binding 0
RWStructuredBuffer<SmOut> gOut;  // binding 1

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
// PxMat33::transform: column0*v.x + column1*v.y + column2*v.z
float3 mtx(float3 c0, float3 c1, float3 c2, float3 v) { return c0 * v.x + c1 * v.y + c2 * v.z; }

[shader("compute")]
[numthreads(64, 1, 1)]
void smMulVec(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SmIn s = gIn[tid];

    const float3 st = s.sTop.xyz;
    const float3 sb = s.sBot.xyz;

    // top = topLeft * s.top + topRight * s.bottom
    const float3 top = mtx(s.tlC0.xyz, s.tlC1.xyz, s.tlC2.xyz, st)
                     + mtx(s.trC0.xyz, s.trC1.xyz, s.trC2.xyz, sb);

    // bottom = bottomLeft * s.top + topLeft.transformTranspose(s.bottom)
    const float3 tlT_sb = float3(dot3(s.tlC0.xyz, sb), dot3(s.tlC1.xyz, sb), dot3(s.tlC2.xyz, sb));
    const float3 bottom = mtx(s.blC0.xyz, s.blC1.xyz, s.blC2.xyz, st) + tlT_sb;

    SmOut o;
    o.outTop = float4(top, 0.0f);
    o.outBot = float4(bottom, 0.0f);
    gOut[tid] = o;
}
