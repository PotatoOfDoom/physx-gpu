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

// npIsoAcosDecode.slang
//
// Slang port of two independent isosurface helpers from isosurfaceExtraction.cu:
//   approxAcos(x) (L884-887): a fast polynomial arccos approximation
//     PxMax(0, (-0.69813170079773212*x*x - 0.87266462599716477)*x + 1.5707963267948966)
//   getDecodedId(encoded, dim) (L552-563): decode a packed grid id
//     if !(encoded & (1<<(29+dim))): return 0xFFFFFFFF
//     id = encoded & 0x1FFFFFFF
//     if dim>0 && (encoded & (1<<29)): id++
//     if dim>1 && (encoded & (1<<30)): id++
//     return id
//
// approxAcos is float (NoContraction-stamped so the PxMax clamp boundary and the
// Horner-ish polynomial match strict IEEE); getDecodedId is pure integer bit
// manipulation.  +-0 eqFZ on the acos output; exact match on the decoded id.

float pmax(float a, float b) { return (a > b) ? a : b; }

float approxAcos(float x)
{
    return pmax(0.0f, (-0.69813170079773212f * x * x - 0.87266462599716477f) * x + 1.5707963267948966f);
}

uint getDecodedId(uint encoded, uint dim)
{
    if ((encoded & (1u << (29u + dim))) == 0u)
        return 0xFFFFFFFFu;

    uint id = encoded & 0x1FFFFFFFu;
    if (dim > 0u && (encoded & (1u << (29u + 0u))) != 0u)
        id++;
    if (dim > 1u && (encoded & (1u << (29u + 1u))) != 0u)
        id++;
    return id;
}

struct AdIn
{
    float4 misc;   // x (float), asfloat(encoded), asfloat(dim), _
}

struct AdOut
{
    float4 out;    // approxAcos(x), asfloat(decodedId), _, _
}

struct AdParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AdParams> gParams;

RWStructuredBuffer<AdIn> gIn;    // binding 0
RWStructuredBuffer<AdOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void isoAcosDecode(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AdIn s = gIn[tid];

    const float x = s.misc.x;
    const uint encoded = asuint(s.misc.y);
    const uint dim = asuint(s.misc.z);

    const float ac = approxAcos(x);
    const uint id = getDecodedId(encoded, dim);

    AdOut o;
    o.out = float4(ac, asfloat(id), 0.0f, 0.0f);
    gOut[tid] = o;
}
