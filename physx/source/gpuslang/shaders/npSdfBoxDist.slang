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

// npSdfBoxDist.slang
//
// Slang port of two AABB primitives from the SDF/BVH construction pass
// (SDFConstruction.cu): getCenter (L44-51) and the point-box squared distance
// distancePointBoxSquared (L279-283).
//
//   getCenter(lower, upper) = 0.5 * (lower + upper)
//
//   distancePointBoxSquared(minimum, maximum, point):
//     closestPt = minimum.maximum( maximum.minimum(point) )
//               = per component max(minimum_i, min(maximum_i, point_i))   // clamp
//     return (closestPt - point).magnitudeSquared()
//
// PxVec3::minimum(v)/maximum(v) are componentwise PxMin/PxMax = selectMin/selectMax
// = a<b?a:b / a>b?a:b (transcribed as ternaries, matching the exact tie/+-0
// behaviour).  `0.5*(l+u)` is scalar*(vector sum).  magnitudeSquared =
// x*x+y*y+z*z.  All float mul/add/sub NoContraction-stamped; +-0 compared eqFZ.

float pmin(float a, float b) { return (a < b) ? a : b; }   // PxMin = selectMin
float pmax(float a, float b) { return (a > b) ? a : b; }   // PxMax = selectMax

struct BxIn
{
    float4 lower;    // box minimum .xyz
    float4 upper;    // box maximum .xyz
    float4 point;    // query point .xyz
}

struct BxOut
{
    float4 center;   // getCenter .xyz
    float4 dist;     // distancePointBoxSquared, _, _, _
}

struct BxParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BxParams> gParams;

RWStructuredBuffer<BxIn> gIn;    // binding 0
RWStructuredBuffer<BxOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sdfBoxDist(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const BxIn s = gIn[tid];

    const float3 lower = s.lower.xyz;
    const float3 upper = s.upper.xyz;
    const float3 point = s.point.xyz;

    // getCenter
    const float3 center = 0.5f * (lower + upper);

    // distancePointBoxSquared: closestPt = max(lower, min(upper, point)) per component
    const float3 inner = float3(pmin(upper.x, point.x), pmin(upper.y, point.y), pmin(upper.z, point.z));   // maximum.minimum(point)
    const float3 closestPt = float3(pmax(lower.x, inner.x), pmax(lower.y, inner.y), pmax(lower.z, inner.z)); // minimum.maximum(...)
    const float3 d = closestPt - point;
    const float distSq = d.x * d.x + d.y * d.y + d.z * d.z;

    BxOut o;
    o.center = float4(center, 0.0f);
    o.dist = float4(distSq, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
