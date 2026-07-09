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

// npTriTri.slang
//
// Slang port of satIntersect from gpunarrowphase/src/CUDA/triangletriangle.cuh —
// the triangle-triangle separating-axis test used by the triangle-mesh /
// mesh-mesh narrow phase.  It returns the maximum separation over the candidate
// axes (the two triangle face normals + the 9 edge-edge cross products); a
// positive result greater than the tolerance means the triangles are separated.
// The CUDA early-outs as soon as an axis exceeds the tolerance, so the returned
// value is that first-exceeding axis' separation; the port reproduces the exact
// axis order (minProject and testSeparatingAxis are inlined).

struct TriTriIn
{
    float4 v0a, v1a, v2a, planeA; // triangle0 verts + plane (n.xyz, d)
    float4 v0b, v1b, v2b, planeB; // triangle1
    float4 misc;                  // (tolerance, ...)
}

struct TriTriParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<TriTriParams> gParams;

RWStructuredBuffer<TriTriIn> gIn;   // binding 0
RWStructuredBuffer<float>    gOut;  // binding 1  separation

// minProject: min plane-distance of the triangle's verts (+ plane.d)
float minProject(float4 plane, float3 v0, float3 v1, float3 v2)
{
    const float3 n = plane.xyz;
    return min(dot(n, v0), min(dot(n, v1), dot(n, v2))) + plane.w;
}

// testSeparatingAxis: max gap of the two triangles' projection intervals on axis
float testSeparatingAxis(float3 axis, float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2)
{
    float min0, max0, min1, max1;
    min0 = max0 = dot(a0, axis);
    min1 = max1 = dot(b0, axis);
    const float pa1 = dot(a1, axis), pa2 = dot(a2, axis);
    const float pb1 = dot(b1, axis), pb2 = dot(b2, axis);
    min0 = min(pa1, min0); max0 = max(pa1, max0);
    min1 = min(pb1, min1); max1 = max(pb1, max1);
    min0 = min(pa2, min0); max0 = max(pa2, max0);
    min1 = min(pb2, min1); max1 = max(pb2, max1);
    return max(min1 - max0, min0 - max1);
}

// ---------------------------------------------------------------------------
// satIntersect: mirrors triangletriangle.cuh satIntersect<true>()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void triTriSat(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const TriTriIn s = gIn[t];
    const float3 a0 = s.v0a.xyz, a1 = s.v1a.xyz, a2 = s.v2a.xyz;
    const float3 b0 = s.v0b.xyz, b1 = s.v1b.xyz, b2 = s.v2b.xyz;
    const float tolerance = s.misc.x;

    float sep = minProject(s.planeA, b0, b1, b2);
    if (sep > tolerance) { gOut[t] = sep; return; }

    sep = max(sep, minProject(s.planeB, a0, a1, a2));
    if (sep > tolerance) { gOut[t] = sep; return; }

    float3 axes0[3] = { a1 - a0, a2 - a0, a2 - a1 };
    float3 axes1[3] = { b1 - b0, b2 - b0, b2 - b1 };

    for (uint i = 0u; i < 3u; ++i)
    {
        const float3 axis0 = axes0[i];
        for (uint j = 0u; j < 3u; ++j)
        {
            const float3 axis1 = axes1[j];
            float3 sepAxis = cross(axis0, axis1);
            const float magSq = dot(sepAxis, sepAxis);
            if (magSq > 1e-5f)
            {
                sepAxis = sepAxis * (1.0f / sqrt(magSq));
                const float tSep = testSeparatingAxis(sepAxis, a0, a1, a2, b0, b1, b2);
                sep = max(sep, tSep);
                if (sep > tolerance) { gOut[t] = sep; return; }
            }
        }
    }

    gOut[t] = sep;
}
