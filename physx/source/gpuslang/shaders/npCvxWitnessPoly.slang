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

// npCvxWitnessPoly.slang
//
// Slang port of getPolygonIndexFromLocalWitness() from cudaGJKEPA.cu (~L611):
// given a witness point + direction (shape space) and the convex polygon planes
// (vertex space), pick the polygon whose plane best matches the witness.
//   - transform the witness into vertex space (shape2Vertex);
//   - per polygon, distance = plane.n.localWitness + plane.d, clamped to +inf
//     when it is below -tolerance (behind the plane by more than tolerance);
//   - minDist = closest such plane.  If no plane is in front (minDist == +inf)
//     pick the max-distance polygon (deepest-behind);
//   - otherwise, among the polygons whose distance is within tolerance of
//     minDist, pick the one whose vertex-space normalized plane normal has the
//     SMALLEST projection onto the witness normal n.
//
// The CUDA runs one warp, striding polygons across 32 lanes and combining with
// warpReduction (min/max) whose winner is the LOWEST lane holding the extreme.
// This port runs one thread per problem and replicates the 32-lane striding +
// lowest-lane tie-break exactly (a plain linear scan would pick a different
// index when the extreme is achieved in more than one lane).

static const float PXG_MAX_F32 = 3.4028235e38f;
static const uint WARP_SIZE = 32u;
static const float PCM_LOWER_EPS = 1e-2f;
static const float PCM_UPPER_EPS = 5e-2f;

// mirrors PxQuat::rotate()
float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(
        (vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}
// mirrors PxQuat::rotateInv()
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(
        (vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}
// shape2Vertex(v,scale,rot) = rotateInv(rotate(v) ./ scale)  (componentwise divide)
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    float3 t = quatRotate(rot, v);
    t.x /= scale.x; t.y /= scale.y; t.z /= scale.z;
    return quatRotateInv(rot, t);
}
// PxVec3::getNormalized(): m>0 ? v*(1/sqrt(m)) : 0  (reciprocal-then-multiply)
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

struct WitnessIn
{
    float4 scaleNb;    // scale.xyz ; .w = nbPolygons (as float)
    float4 rot;        // rotation quaternion
    float4 witness;    // witness point .xyz ; .w = radius
    float4 nTol;       // normal .xyz ; .w = toleranceLength
}

struct WitnessOut { uint index; uint pad0, pad1, pad2; }

struct WitnessParams { uint numTests; uint maxPolys; }

[[vk::push_constant]]
ConstantBuffer<WitnessParams> gParams;

RWStructuredBuffer<WitnessIn> gIn;      // binding 0
RWStructuredBuffer<float4>    gPlanes;  // binding 1 : maxPolys planes per test
RWStructuredBuffer<WitnessOut> gOut;    // binding 2

uint getPolygonIndex(uint testIdx, uint nbPolygons, float3 scale, float4 rot,
                     float3 witnessPt, float3 n, float radius, float toleranceLength)
{
    const float margin = radius * 0.01f;
    const float lowerEps = toleranceLength * PCM_LOWER_EPS;
    const float upperEps = toleranceLength * PCM_UPPER_EPS;
    const float tolerance = clamp(margin, lowerEps, upperEps);

    const float3 localWitness = shape2Vertex(witnessPt, scale, rot);
    const float eps = -tolerance;
    const uint base = testIdx * gParams.maxPolys;

    // Pass 1, per lane (strided): running min distance, max distance + its index.
    float laneMin[32];
    float laneMax[32];
    uint laneIdx[32];
    for (uint L = 0u; L < WARP_SIZE; ++L)
    {
        float minDist = PXG_MAX_F32;
        float maxDist = -PXG_MAX_F32;
        uint index = 0u;
        for (uint i = L; i < nbPolygons; i += WARP_SIZE)
        {
            const float4 pl = gPlanes[base + i];
            const float pd = dot(float3(pl.x, pl.y, pl.z), localWitness) + pl.w;
            const float toleranceDist = pd >= eps ? pd : PXG_MAX_F32;
            minDist = min(toleranceDist, minDist);
            if (toleranceDist > maxDist)
            {
                maxDist = toleranceDist;
                index = i;
            }
        }
        laneMin[L] = minDist;
        laneMax[L] = maxDist;
        laneIdx[L] = index;
    }

    // warp min-reduce (value only)
    float minDist = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) minDist = min(minDist, laneMin[L]);

    if (minDist == PXG_MAX_F32)
    {
        // no plane in front: pick max-distance polygon (lowest lane wins the tie)
        float maxDist = -PXG_MAX_F32;
        for (uint L = 0u; L < WARP_SIZE; ++L) maxDist = max(maxDist, laneMax[L]);
        for (uint L = 0u; L < WARP_SIZE; ++L)
            if (laneMax[L] == maxDist)
                return laneIdx[L];
        return 0u;
    }

    // Pass 2, per lane: among polygons within tolerance of minDist, pick the
    // smallest plane-normal projection.  index carries over from pass 1.
    float laneProj[32];
    uint laneProjIdx[32];
    for (uint L = 0u; L < WARP_SIZE; ++L)
    {
        float bestProjection = PXG_MAX_F32;
        uint index = laneIdx[L];
        for (uint i = L; i < nbPolygons; i += WARP_SIZE)
        {
            const float4 pl = gPlanes[base + i];
            const float3 pn = float3(pl.x, pl.y, pl.z);
            const float pd = dot(pn, localWitness) + pl.w;
            const float toleranceDist = pd >= eps ? pd : PXG_MAX_F32;
            if (tolerance > (toleranceDist - minDist))
            {
                const float3 planeN = getNormalized(shape2Vertex(pn, scale, rot));
                const float proj = dot(planeN, n);
                if (bestProjection > proj)
                {
                    index = i;
                    bestProjection = proj;
                }
            }
        }
        laneProj[L] = bestProjection;
        laneProjIdx[L] = index;
    }

    // warp min-reduce over projections, lowest lane wins
    float bestProjection = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) bestProjection = min(bestProjection, laneProj[L]);
    for (uint L = 0u; L < WARP_SIZE; ++L)
        if (laneProj[L] == bestProjection)
            return laneProjIdx[L];
    return 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxWitnessPoly(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const WitnessIn s = gIn[t];
    const float3 scale = s.scaleNb.xyz;
    const uint nbPolygons = uint(s.scaleNb.w);
    const float4 rot = s.rot;
    const float3 witnessPt = s.witness.xyz;
    const float radius = s.witness.w;
    const float3 n = s.nTol.xyz;
    const float toleranceLength = s.nTol.w;

    const uint index = getPolygonIndex(t, nbPolygons, scale, rot, witnessPt, n, radius, toleranceLength);

    WitnessOut r;
    r.index = index; r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
    gOut[t] = r;
}
