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

// npCvxTriWorldNormal.slang
//
// Slang port of getWorldNormal (convexMeshOutput.cu :101-112) — tranche 115.
// The per-lane patch-normal helper used by convexTrimeshFinishContacts: take
// contacts[0].normal (or the zero vector on a lane past its manifold's contact
// count), rotate it into world space by the mesh transform's quaternion, and
// normalise:
//
//   normal            = (threadIndexInManifold >= numContacts) ? 0 : contacts[0].normal;
//   unNormalizedNormal = meshTransform.rotate(normal);            // PxQuat::rotate
//   return unNormalizedNormal.getNormalized();                    // m>0 ? v*rsqrt(m) : 0
//
// The rotation reuses the shared quatRotate helper (mirrors PxQuat::rotate).
// getNormalized is PxVec3::getNormalized: magnitudeSquared m; if m > 0 return
// v * PxRecipSqrt(m) else the zero vector.  Following the established discipline
// PxRecipSqrt(m) is evaluated as 1.0f/sqrt(m) (llvmpipe's OpInverseSqrt is not
// correctly rounded, but OpSqrt and the division are), so the shader and the CPU
// reference agree bit-for-bit.  The inactive-lane path (normal = 0) drives the
// m == 0 branch and must return the zero vector, not a NaN.
//
// Determinism: exact-integer rotations (identity + 180-degree axis quaternions
// give integer-exact rotated normals independent of FMA contraction) plus the
// shared quatRotate helper (bit-exact against its own transcription, as used by
// the transform tranches) keep the only rounding inside getNormalized, which is
// identical on both sides.

import pxgSlangCommon;

struct WnParams { uint numTests; uint pad0, pad1, pad2; }

struct WnIn
{
    float4 q;       // mesh transform quaternion (x,y,z,w)
    float4 normal0; // xyz = contacts[0].normal (w unused)
    uint4 cfg;      // x = numContacts, y = threadIndexInManifold, z/w unused
}

[[vk::push_constant]]
ConstantBuffer<WnParams> gParams;

RWStructuredBuffer<WnIn> gIn;    // binding 0
RWStructuredBuffer<float4> gOut; // binding 1: xyz = world normal

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxVec3::getNormalized(): m = magnitudeSquared(); m > 0 ? v * (1/sqrt(m)) : 0
float3 getNormalized(float3 v)
{
    const float m = dotX(v, v);
    return (m > 0.0f) ? v * (1.0f / sqrt(m)) : float3(0.0f, 0.0f, 0.0f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctCvxTriWorldNormal(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const WnIn e = gIn[t];
    const uint numContacts = e.cfg.x;
    const uint threadIndexInManifold = e.cfg.y;

    const float3 normal = (threadIndexInManifold >= numContacts) ? float3(0.0f, 0.0f, 0.0f) : e.normal0.xyz;
    const float3 unNormalizedNormal = quatRotate(e.q, normal);
    const float3 worldNormal = getNormalized(unNormalizedNormal);

    gOut[t] = float4(worldNormal, 0.0f);
}
