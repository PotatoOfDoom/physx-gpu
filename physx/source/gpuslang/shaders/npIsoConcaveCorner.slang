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

// npIsoConcaveCorner.slang
//
// Slang port of directionSign + indexOfConcaveCorner (isosurfaceExtraction.cu
// L741-783): find which corner of a projected quad (a,b,c,d) is concave, using
// a 2D cross-product turn test on the two components selected by the principal
// direction.
//   projections[3][2] = {{1,2},{2,0},{0,1}}
//   directionSign(pd, start, middle, end):
//     p0=proj[pd][0], p1=proj[pd][1]
//     a0=middle[p0]-start[p0], a1=middle[p1]-start[p1]
//     b0=end[p0]-middle[p0],   b1=end[p1]-middle[p1]
//     return a0*b1 - a1*b0
//   indexOfConcaveCorner: minimum=0, result=-1; for each of the four rotations
//     (a,b,c)->1, (b,c,d)->2, (c,d,a)->3, (d,a,b)->0, if s<=minimum then
//     minimum=s, result=<that index>.  The `<=` means later ties overwrite.
//
// Pure PxVec3 subtraction + one 2D cross per corner; integer index result.
// The float mul/adds are NoContraction-stamped so the sign comparisons match
// strict IEEE; +-0 eqFZ on the emitted sign values.

static const int2 projections[3] = { int2(1, 2), int2(2, 0), int2(0, 1) };

float directionSign(int pd, float3 start, float3 middle, float3 end)
{
    const int p0 = projections[pd].x;
    const int p1 = projections[pd].y;
    const float a0 = middle[p0] - start[p0];
    const float a1 = middle[p1] - start[p1];
    const float b0 = end[p0] - middle[p0];
    const float b1 = end[p1] - middle[p1];
    return a0 * b1 - a1 * b0;
}

struct IcIn
{
    float4 a, b, c, d;   // .xyz corners
    float4 misc;         // principalDirection stored via asfloat(uint), _, _, _
}

struct IcOut
{
    float4 signs;        // directionSign for (abc, bcd, cda, dab)
    int4 result;         // concave-corner index in .x
}

struct IcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<IcParams> gParams;

RWStructuredBuffer<IcIn> gIn;    // binding 0
RWStructuredBuffer<IcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void isoConcaveCorner(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const IcIn s = gIn[tid];

    const float3 a = s.a.xyz, b = s.b.xyz, c = s.c.xyz, d = s.d.xyz;
    const int pd = int(asuint(s.misc.x));

    const float sABC = directionSign(pd, a, b, c);
    const float sBCD = directionSign(pd, b, c, d);
    const float sCDA = directionSign(pd, c, d, a);
    const float sDAB = directionSign(pd, d, a, b);

    float minimum = 0.0f;
    int result = -1;
    if (sABC <= minimum) { minimum = sABC; result = 1; }
    if (sBCD <= minimum) { minimum = sBCD; result = 2; }
    if (sCDA <= minimum) { minimum = sCDA; result = 3; }
    if (sDAB <= minimum) { minimum = sDAB; result = 0; }

    IcOut o;
    o.signs = float4(sABC, sBCD, sCDA, sDAB);
    o.result = int4(result, 0, 0, 0);
    gOut[tid] = o;
}
