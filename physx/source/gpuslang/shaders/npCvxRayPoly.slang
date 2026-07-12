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

// npCvxRayPoly.slang
//
// Slang port of intersectRayPolyhedron() from
// gpunarrowphase/src/CUDA/cudaGJKEPA.cu (~L1025).  It clips the ray
// (a + t*dir) against the convex polyhedron described by its face planes
// (n, d) with n.x+d >= 0 the outside half-space, computing the ray parameter
// interval [tEnter, tExit] the ray spends inside the polyhedron:
//   - a face whose plane is (nearly) parallel to the ray and whose base point
//     lies outside separates the ray from the polyhedron -> no intersection;
//   - otherwise each face contributes a slab crossing tTemp = -distToPlane/denom;
//     denom < 0 (entering the half-space) raises tEnter, denom > 0 (exiting)
//     lowers tExit;
//   - if at any point tEnter > tExit the ray misses.
//
// The CUDA original processes WARP_SIZE(32) faces per warp iteration, all-reduces
// tEnter (max) / tExit (min) across the warp and ballots the separation flag.
// This port runs one thread per problem and serialises the warp: within each
// block of 32 faces every face is evaluated against the block's *shared*
// pre-block tEnter/tExit (exactly as the parallel lanes do), then the block max
// (tEnter) / min (tExit) and OR (separation) are folded in.

static const float PXG_MAX_F32 = 3.4028235e38f;
static const uint WARP_SIZE = 32u;

struct RayPolyIn
{
    float4 a;        // ray origin (.xyz); .w = nbPolygons (as float)
    float4 dir;      // ray direction (.xyz)
}

// hit in meta.x (1/0); tEnter in meta.y, tExit in meta.z
struct RayPolyOut { float4 meta; }

struct RayPolyParams { uint numTests; uint maxPolys; }

[[vk::push_constant]]
ConstantBuffer<RayPolyParams> gParams;

RWStructuredBuffer<RayPolyIn> gIn;      // binding 0
RWStructuredBuffer<float4>    gPlanes;  // binding 1 : maxPolys planes per test
RWStructuredBuffer<RayPolyOut> gOut;    // binding 2

// returns true (hit) with tEnter/tExit filled, or false (miss)
bool intersectRayPolyhedron(float3 a, float3 dir, uint testIdx, uint nbPolygons,
                            out float tEnter, out float tExit)
{
    const float eps = 1e-7f;
    float tFirst = 0.0f;
    float tLast = PXG_MAX_F32;
    tEnter = 0.0f;
    tExit = 0.0f;

    const uint base = testIdx * gParams.maxPolys;

    for (uint i = 0u; i < nbPolygons; i += WARP_SIZE)
    {
        bool blockSep = false;
        // block folds start from the shared pre-block interval (inactive warp
        // lanes keep tFirst/tLast, so they are the reduction identity here)
        float newFirst = tFirst;
        float newLast = tLast;

        for (uint lane = 0u; lane < WARP_SIZE; ++lane)
        {
            const uint k = i + lane;
            if (k >= nbPolygons)
                continue;

            const float4 plane = gPlanes[base + k];
            const float3 n = float3(plane.x, plane.y, plane.z);
            const float d = plane.w;

            const float denominator = dot(n, dir);
            const float distToPlane = dot(n, a) + d;

            if (abs(denominator) < eps)
            {
                if (distToPlane > 0.0f)
                    blockSep = true;
            }
            else
            {
                const float tTemp = -distToPlane / denominator;
                // denom < 0: ray entering half-space (raise tEnter);
                // denom > 0: ray exiting half-space (lower tExit)
                const bool con = 0.0f > denominator;
                const bool con0 = tTemp > tFirst;
                const bool con1 = tLast > tTemp;

                // lane result: tFirst_lane = (con && con0) ? tTemp : tFirst
                //              tLast_lane  = (con1 && !con) ? tTemp : tLast
                // warp reduces max(tFirst_lane) / min(tLast_lane)
                if (con && con0)
                    newFirst = max(newFirst, tTemp);
                if (con1 && !con)
                    newLast = min(newLast, tTemp);
            }
        }

        if (blockSep)
            return false;

        tFirst = newFirst;
        tLast = newLast;

        if (tFirst > tLast)
            return false;
    }

    tEnter = tFirst;
    tExit = tLast;
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxRayPoly(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const RayPolyIn s = gIn[t];
    const float3 a = s.a.xyz;
    const float3 dir = s.dir.xyz;
    const uint nbPolygons = uint(s.a.w);

    float tEnter, tExit;
    const bool hit = intersectRayPolyhedron(a, dir, t, nbPolygons, tEnter, tExit);

    RayPolyOut r;
    r.meta = float4(hit ? 1.0f : 0.0f, tEnter, tExit, 0.0f);
    gOut[t] = r;
}
