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

// npCvxLocalNormal.slang
//
// Slang port of calculateLocalNormal() from cudaGJKEPA.cu (~L510): the average
// contact normal of a persistent-contact manifold (<= PXG_MAX_PCM_CONTACTS = 4
// contacts).  The CUDA sums the first numContacts normals with a width-4 tree
// reduction (shuffle tI+1 then tI+2), broadcasts lane 0's sum, and returns it
// normalized.  The width-4 tree fixes the summation association to the pairwise
// (n0+n1)+(n2+n3) form; float addition is non-associative, so the port must use
// exactly that grouping (with the out-of-range slots contributing zero).

static const uint PXG_MAX_PCM_CONTACTS = 4u;

// PxVec3::getNormalized(): m>0 ? v*(1/sqrt(m)) : 0  (reciprocal-then-multiply)
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

struct LocalNormalIn
{
    float4 np[4];   // manifold normal+pen; .xyz = normal
    uint numContacts;
    uint pad0, pad1, pad2;
}

struct LocalNormalOut { float4 normal; }

struct LocalNormalParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<LocalNormalParams> gParams;

RWStructuredBuffer<LocalNormalIn> gIn;    // binding 0
RWStructuredBuffer<LocalNormalOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxLocalNormal(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const LocalNormalIn s = gIn[t];
    const uint numContacts = s.numContacts;

    // lane L contributes normal[L] if L < numContacts, else 0
    float3 n0 = (0u < numContacts) ? s.np[0].xyz : float3(0, 0, 0);
    float3 n1 = (1u < numContacts) ? s.np[1].xyz : float3(0, 0, 0);
    float3 n2 = (2u < numContacts) ? s.np[2].xyz : float3(0, 0, 0);
    float3 n3 = (3u < numContacts) ? s.np[3].xyz : float3(0, 0, 0);

    // width-4 tree reduction: lane 0 ends up with (n0+n1) + (n2+n3)
    const float3 localNormal = (n0 + n1) + (n2 + n3);

    LocalNormalOut r;
    r.normal = float4(getNormalized(localNormal), 0.0f);
    gOut[t] = r;
}
