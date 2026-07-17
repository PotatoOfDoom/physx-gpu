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

// npSmoothPos.slang
//
// Slang port of the smoothPositionsLaunch final blending stage (anisotropy.cu
// L200-209).  Given the neighborhood-accumulated weighted position sum xs and
// weight sum ws (whose accumulation loop is not bit-portable in isolation), the
// original position xi (with its pass-through w), and the smoothing / contact
// parameters, it computes:
//
//   if (ws > 0):
//     f      = 4 * Wa(contactDist*0.5, contactDistInv)      Wa(x,invr)=1-cube(x*invr)
//     smooth = PxMin(1, ws/f) * smoothing
//     xs    /= ws                    // PxVec3::operator/= : RECIPROCAL-multiply
//     xi     = Lerp(xi, xs, smooth)  // xi + (xs - xi) * smooth
//   output = (xi.xyz, xi.w)          // w passed through unchanged
//
// CRITICAL: `xs /= ws` is PxVec3::operator/=, which computes `inv = 1/ws` once
// and multiplies each component by inv (two roundings), NOT per-component
// division `xs.i / ws` (one rounding) -- the two differ by up to 1 ulp.  The
// scalar `ws / f` is a plain single-rounding division.  PxMin(1, x) =
// selectMin(1, x) = (1 < x) ? 1 : x.  Lerp's (end-start)*t is a direct scalar
// multiply and operator+ / operator- are per component.  All float mul/add/div
// NoContraction-stamped; +-0 compared with eqFZ.

float cube(float x) { return x * x * x; }
float Wa(float x, float invr) { return 1.0f - cube(x * invr); }

struct SpIn
{
    float4 xi;       // original position xi.xyz, pass-through w
    float4 xs;       // accumulated weighted position sum xyz, _
    float4 params;   // ws, smoothing, contactDist, contactDistInv
}

struct SpOut
{
    float4 pos;      // smoothed position xyz, xi.w
}

struct SpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SpParams> gParams;

RWStructuredBuffer<SpIn> gIn;    // binding 0
RWStructuredBuffer<SpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void smoothPos(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SpIn s = gIn[tid];

    const float3 xi = s.xi.xyz;
    const float w = s.xi.w;
    const float3 xs = s.xs.xyz;
    const float ws = s.params.x, smoothing = s.params.y;
    const float contactDist = s.params.z, contactDistInv = s.params.w;

    float3 xiOut = xi;
    if (ws > 0.0f)
    {
        const float f = 4.0f * Wa(contactDist * 0.5f, contactDistInv);
        const float wsf = ws / f;
        const float mn = (1.0f < wsf) ? 1.0f : wsf;   // PxMin(1.0f, ws/f)
        const float smooth = mn * smoothing;

        const float invws = 1.0f / ws;                // xs /= ws  (reciprocal-multiply)
        const float3 xsN = float3(xs.x * invws, xs.y * invws, xs.z * invws);

        xiOut = float3(xi.x + (xsN.x - xi.x) * smooth,  // Lerp(xi, xsN, smooth)
                       xi.y + (xsN.y - xi.y) * smooth,
                       xi.z + (xsN.z - xi.z) * smooth);
    }

    SpOut o;
    o.pos = float4(xiOut, w);
    gOut[tid] = o;
}
