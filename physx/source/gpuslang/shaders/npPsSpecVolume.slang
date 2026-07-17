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

// npPsSpecVolume.slang
//
// Slang port of getParticleSpeculativeContactVolume (gridCal.cuh L122-140): the
// midpoint + covering radius of a particle's forward-projected motion range.
//   scaleInv = isDiffuse ? 1/SCALE_DIFFUSE
//                        : (isTGS ? 1/SCALE_TGS : 1/SCALE_PGS)
//     SCALE_PGS=1.0, SCALE_TGS=0.5, SCALE_DIFFUSE=1.0  (so 1.0, 2.0, 1.0)
//   cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5
//   cVolumePos    = currentPos + cVolumeOffset
//   return          cVolumeOffset.magnitude() + contactDist
//
// The scale reciprocals are exact (1/1.0=1.0, 1/0.5=2.0).  Float mul/add/sqrt
// NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static const float SCALE_INV_PGS = 1.0f / 1.0f;
static const float SCALE_INV_TGS = 1.0f / 0.5f;
static const float SCALE_INV_DIFFUSE = 1.0f / 1.0f;

struct SvIn
{
    float4 cur;    // currentPos.xyz
    float4 pred;   // predictedPos.xyz
    float4 misc;   // contactDist, asfloat(isDiffuse), asfloat(isTGS), _
}

struct SvOut
{
    float4 volPos;    // cVolumePos.xyz
    float4 scalars;   // radius, _, _, _
}

struct SvParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SvParams> gParams;

RWStructuredBuffer<SvIn> gIn;    // binding 0
RWStructuredBuffer<SvOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psSpecVolume(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SvIn s = gIn[tid];

    const float3 currentPos = s.cur.xyz;
    const float3 predictedPos = s.pred.xyz;
    const float contactDist = s.misc.x;
    const bool isDiffuse = asuint(s.misc.y) == 1u;
    const bool isTGS = asuint(s.misc.z) == 1u;

    float scaleInv;
    if (isDiffuse)
        scaleInv = SCALE_INV_DIFFUSE;
    else
        scaleInv = isTGS ? SCALE_INV_TGS : SCALE_INV_PGS;

    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    const float3 cVolumePos = currentPos + cVolumeOffset;
    const float radius = sqrt(dot3(cVolumeOffset, cVolumeOffset)) + contactDist;

    SvOut o;
    o.volPos = float4(cVolumePos, 0.0f);
    o.scalars = float4(radius, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
