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

// npBoxClip.slang
//
// Slang port of the self-contained box-box clipping geometry primitives from
// gpunarrowphase/src/CUDA/cudaBox.cu (used by the box-box contact generator):
//   - contains: 2D point-in-polygon test (crossing-number ray cast with the
//     original's exact endpoint / epsilon handling), preceded by the polygon's
//     x/y bounds early-out.
//   - intersectSegmentAABB: parametric segment-vs-AABB slab intersection,
//     returning the [tmin, tmax] sub-interval and whether the segment (t in
//     [0,1]) meets the box.
//
// These are exposed as batch compute entries (one test per thread) so the
// geometry can be verified directly; the warp-cooperative SAT / incident-face /
// clipping machinery (getIncidentPolygon4, calculateContacts,
// doBoxBoxGenerateContacts) is a later tranche.

static const uint MAX_VERTS = 8u;

struct ContainsTest
{
    float px, py;      // query point (2D)
    float bminx, bminy;
    float bmaxx, bmaxy;
    uint numVerts;
    uint pad;
}

struct SegTest
{
    float4 p0;   // .xyz segment start
    float4 d;    // .xyz segment direction (p1 - p0)
    float4 mn;   // .xyz AABB min
    float4 mx;   // .xyz AABB max
}

struct BoxClipParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BoxClipParams> gParams;

RWStructuredBuffer<float4>       gVerts;        // binding 0  polygon verts (MAX_VERTS per test, .xy used)
RWStructuredBuffer<ContainsTest> gContainsIn;   // binding 1
RWStructuredBuffer<uint>         gContainsOut;  // binding 2  (0/1)
RWStructuredBuffer<SegTest>      gSegIn;        // binding 3
RWStructuredBuffer<float4>       gSegOut;       // binding 4  (tmin, tmax, hit, 0)

// ---------------------------------------------------------------------------
// contains: mirrors cudaBox.cu contains()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void boxContains(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const ContainsTest c = gContainsIn[t];

    uint result = 0u;
    const float tx = c.px, ty = c.py;

    if (!(tx < c.bminx || tx > c.bmaxx || ty < c.bminy || ty > c.bmaxy))
    {
        const float eps = 1e-6f;
        const float zero = 0.0f;
        const uint n = c.numVerts;
        const uint base = t * MAX_VERTS;

        uint intersectionPoints = 0u;
        bool done = false;
        bool inside = false;

        uint j = n - 1u;
        for (uint i = 0u; i < n && !done; ++i)
        {
            const float jy = gVerts[base + j].y;
            const float iy = gVerts[base + i].y;
            const float jx = gVerts[base + j].x;
            const float ix = gVerts[base + i].x;

            if ((tx == jx && ty == jy) || (tx == ix && ty == iy))
            {
                inside = true; done = true;
            }
            else
            {
                const bool yflag0 = jy > ty;
                const bool yflag1 = iy > ty;
                if (yflag0 != yflag1)
                {
                    const float jix = ix - jx;
                    const float jiy = iy - jy;
                    const float jty = ty - jy;
                    const float part1 = jty * jix;
                    const float part2 = (jx + eps) * jiy;
                    const float part3 = tx * jiy;
                    const bool comp = jiy > zero;
                    const float tmp = part1 + part2;
                    const float comp1 = comp ? tmp : part3;
                    const float comp2 = comp ? part3 : tmp;
                    if (comp1 >= comp2)
                    {
                        if (intersectionPoints == 1u) { inside = false; done = true; }
                        else intersectionPoints++;
                    }
                }
                j = i;
            }
        }

        if (!done) inside = intersectionPoints > 0u;
        result = inside ? 1u : 0u;
    }

    gContainsOut[t] = result;
}

// ---------------------------------------------------------------------------
// intersectSegmentAABB: mirrors cudaBox.cu intersectSegmentAABB()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void boxIntersectSegmentAABB(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const SegTest s = gSegIn[t];
    const float3 p0 = s.p0.xyz, d = s.d.xyz, mn = s.mn.xyz, mx = s.mx.xyz;

    const float eps = 1e-6f;
    const float3 absV = abs(d);
    const float fMax = 3.4028235e38f; // PX_MAX_F32

    float tminf = 0.0f;
    float tmaxf = 1.0f;

    const bool pX = eps > absV.x;
    const bool pY = eps > absV.y;
    const bool pZ = eps > absV.z;

    bool hit = true;
    if ((pX && (p0.x > mx.x || p0.x < mn.x)) ||
        (pY && (p0.y > mx.y || p0.y < mn.y)) ||
        (pZ && (p0.z > mx.z || p0.z < mn.z)))
    {
        hit = false;
    }
    else
    {
        const float3 odd = float3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
        const float3 t1 = float3(pX ? 0.0f : (mn.x - p0.x) * odd.x,
                                 pY ? 0.0f : (mn.y - p0.y) * odd.y,
                                 pZ ? 0.0f : (mn.z - p0.z) * odd.z);
        const float3 t2 = float3(pX ? fMax : (mx.x - p0.x) * odd.x,
                                 pY ? fMax : (mx.y - p0.y) * odd.y,
                                 pZ ? fMax : (mx.z - p0.z) * odd.z);
        const float3 tt1 = min(t1, t2);
        const float3 tt2 = max(t1, t2);
        const float ft1 = max(tt1.x, max(tt1.y, tt1.z));
        const float ft2 = min(tt2.x, min(tt2.y, tt2.z));
        tminf = max(ft1, tminf);
        tmaxf = min(tmaxf, ft2);
        hit = !((tminf > tmaxf) || (tminf > 1.0f));
    }

    gSegOut[t] = float4(tminf, tmaxf, hit ? 1.0f : 0.0f, 0.0f);
}
