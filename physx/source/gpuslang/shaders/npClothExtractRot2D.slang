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

// npClothExtractRot2D.slang
//
// Slang port of extractRotation2D (FEMClothUtil.cuh L88-144): extract the
// rotation R = [r0 r1] from a 2x2 deformation gradient F = [f0 f1] by polar
// decomposition via the closed-form square root of F^T F.
//   S2   = (f0.dot(f0), f1.dot(f1), f0.dot(f1))   // symmetric F^T F packed
//   det  = S2.x*S2.y - S2.z*S2.z
//   if det < THRESHOLD(1e-14): R = identity, return
//   s0   = sqrt(det)
//   t    = sqrt(S2.x + S2.y + 2*s0)
//   if t < THRESHOLD: R = identity, return
//   tInv = 1/t
//   S    = (S2.x+s0, S2.y+s0, S2.z) * tInv      // sqrt(F^T F)
//   sDet = S.x*S.y - S.z*S.z
//   if sDet < THRESHOLD: R = identity, return
//   sDetInv = 1/sDet
//   SInv = (S.y, S.x, -S.z) * sDetInv           // inverse of sqrt(F^T F)
//   r0   = SInv.x*f0 + SInv.z*f1                 // R = F * (F^T F)^{-1/2}
//   r1   = SInv.z*f0 + SInv.y*f1
//
// The three threshold early-returns each emit the 2x2 identity.  All float
// arithmetic NoContraction-stamped; sqrt is IEEE (bit-exact portable); +-0 eqFZ.

float dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

struct ErIn
{
    float4 f;      // f0.xy, f1.xy
}

struct ErOut
{
    float4 r;      // r0.xy, r1.xy
}

struct ErParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ErParams> gParams;

RWStructuredBuffer<ErIn> gIn;    // binding 0
RWStructuredBuffer<ErOut> gOut;  // binding 1

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

[shader("compute")]
[numthreads(64, 1, 1)]
void clothExtractRot2D(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const ErIn s = gIn[tid];

    const float2 f0 = float2(s.f.x, s.f.y);
    const float2 f1 = float2(s.f.z, s.f.w);

    ErOut o;

    // F^T * F packed as (S2.x, S2.y, S2.z) = ([0][0], [1][1], [0][1]=[1][0])
    const float3 S2 = float3(dot2(f0, f0), dot2(f1, f1), dot2(f0, f1));
    const float det = S2.x * S2.y - S2.z * S2.z;

    if (det < FEMCLOTH_THRESHOLD)
    {
        o.r = float4(1.0f, 0.0f, 0.0f, 1.0f);
        gOut[tid] = o;
        return;
    }

    const float s0 = sqrt(det);
    const float t = sqrt(S2.x + S2.y + 2.0f * s0);

    if (t < FEMCLOTH_THRESHOLD)
    {
        o.r = float4(1.0f, 0.0f, 0.0f, 1.0f);
        gOut[tid] = o;
        return;
    }
    const float tInv = 1.0f / t;

    // S = sqrt(F^T F): add s0 to the diagonal entries, scale all by tInv
    float3 S = S2;
    S.x = S.x + s0;
    S.y = S.y + s0;
    S = S * tInv;

    const float sDet = S.x * S.y - S.z * S.z;

    if (sDet < FEMCLOTH_THRESHOLD)
    {
        o.r = float4(1.0f, 0.0f, 0.0f, 1.0f);
        gOut[tid] = o;
        return;
    }

    const float sDetInv = 1.0f / sDet;

    // SInv = inverse of the symmetric S, packed the same way
    float3 SInv = float3(S.y, S.x, -S.z);
    SInv = SInv * sDetInv;

    // R = [r0 r1] = F * SInv
    const float2 r0 = SInv.x * f0 + SInv.z * f1;
    const float2 r1 = SInv.z * f0 + SInv.y * f1;

    o.r = float4(r0.x, r0.y, r1.x, r1.y);
    gOut[tid] = o;
}
