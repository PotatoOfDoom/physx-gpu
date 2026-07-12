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

// npCvxEEDriver.slang
//
// Slang port of generatedContactsEEContacts() from cudaGJKEPA.cu (~L1198): the
// convex-convex edge-edge contact driver.  It composes the ported foundation:
//   getPolygonIndexFromLocalWitness (T138) to pick the best-matching polygon of
//     hull B (reading the polygon planes from the blob, T141/T142);
//   then, warp-strided over that polygon's edges, reads the two edge vertices
//     v0/v1 = vB[vertexData8[getVRef8(polyDesc)+start/end]] and runs the
//     convex-convex generateEE (T134) with the A-edge (vA[0],vA[1]);
//   each hit is warp-compacted (ballot + warpScanExclusive) into the manifold at
//     numContacts+offset (offset<4), with the local-normal-penetration NEGATED,
//     and numContacts = min(numContacts + popcount(mask), 4) per 32-vert block.
// The caller only invokes this when numContacts<2, so incoming numContacts is 0
// or 1 and the offset<4-guarded writes stay within the 4-slot manifold.
//
// This port runs one thread per problem and serialises the warp faithfully.

static const uint WARP_SIZE = 32u;
static const uint SZ_HEADER = 32u;
static const float PXG_MAX_F32 = 3.4028235e38f;
static const float PCM_LOWER_EPS = 1e-2f;
static const float PCM_UPPER_EPS = 5e-2f;

// ---- T141 accessors / offsets ----
uint u32HighW(uint v) { return (v >> 16) & 0xffffu; }
uint u32LowW(uint v)  { return v & 0xffffu; }
uint u16HighW(uint v) { return (v >> 8) & 0xffu; }
uint getVRef8W(uint val)   { return u32HighW(val); }
uint getNbVertsW(uint val) { return u16HighW(u32LowW(val)); }
uint planesByteOff(uint nbV) { return SZ_HEADER + 16u * nbV; }
uint polyDescByteOff(uint nbV, uint nbP) { return SZ_HEADER + 16u * nbV + 16u * nbP; }
uint vertexDataByteOff(uint nbV, uint nbP, uint nbE) { return SZ_HEADER + 16u * nbV + 20u * nbP + 6u * nbE + 3u * nbV; }

// ---- quat / transform (T138/T134) ----
float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    float3 t = quatRotate(rot, v);
    t.x /= scale.x; t.y /= scale.y; t.z /= scale.z;
    return quatRotateInv(rot, t);
}
float3 transformInvPoint(float4 q, float3 p, float3 v) { return quatRotateInv(q, v - p); }
float3 getNormalized(float3 v) { const float m = dot(v, v); return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0); }

struct EEDriverIn
{
    uint baseWord;   // blob base (uint units)
    uint baseVB;     // vB cloud base (float4 units)
    uint nbV, nbP, nbE;
    uint numContactsIn;
    uint pad0;
    float4 scaleRadius;   // scale.xyz ; .w = inSphereRadius
    float4 rot;
    float4 witnessTol;    // closestPointB.xyz ; .w = toleranceLength
    float4 normalCD;      // gjk direction .xyz ; .w = contactDistance
    float4 vA0;           // A-edge endpoint 0
    float4 vA1;           // A-edge endpoint 1
    float4 aToBq;
    float4 aToBp;         // .xyz translation
    float4 seedA;         // pre-existing slot-0 face contact (when numContactsIn==1)
    float4 seedB;
    float4 seedNp;        // .xyz normal, .w sep
}

struct EEDriverOut
{
    float4 mA[4];
    float4 mB[4];
    float4 mNp[4];
    uint numContacts;
    uint pad0, pad1, pad2;
}

struct EEDriverParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<EEDriverParams> gParams;

RWStructuredBuffer<EEDriverIn> gIn;   // binding 0
RWStructuredBuffer<uint>       gBlob; // binding 1
RWStructuredBuffer<float4>     gVB;   // binding 2
RWStructuredBuffer<EEDriverOut> gOut; // binding 3

float4 loadFloat4(uint baseWord, uint byteOff)
{
    const uint w = baseWord + (byteOff >> 2);
    return float4(asfloat(gBlob[w]), asfloat(gBlob[w + 1]), asfloat(gBlob[w + 2]), asfloat(gBlob[w + 3]));
}
uint loadU32(uint baseWord, uint byteOff) { return gBlob[baseWord + (byteOff >> 2)]; }
uint loadU8(uint baseWord, uint byteOff) { const uint word = gBlob[baseWord + (byteOff >> 2)]; return (word >> (8u * (byteOff & 3u))) & 0xffu; }

// getPolygonIndexFromLocalWitness (T138), planes read from the blob.
uint getPolygonIndexBlob(uint baseWord, uint nbPolygons, uint pOff, float3 scale, float4 rot,
                         float3 witnessPt, float3 n, float radius, float toleranceLength)
{
    const float margin = radius * 0.01f;
    const float lowerEps = toleranceLength * PCM_LOWER_EPS;
    const float upperEps = toleranceLength * PCM_UPPER_EPS;
    const float tolerance = clamp(margin, lowerEps, upperEps);
    const float3 localWitness = shape2Vertex(witnessPt, scale, rot);
    const float eps = -tolerance;

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
            const float4 pl = loadFloat4(baseWord, pOff + 16u * i);
            const float pd = dot(float3(pl.x, pl.y, pl.z), localWitness) + pl.w;
            const float toleranceDist = pd >= eps ? pd : PXG_MAX_F32;
            minDist = min(toleranceDist, minDist);
            if (toleranceDist > maxDist) { maxDist = toleranceDist; index = i; }
        }
        laneMin[L] = minDist; laneMax[L] = maxDist; laneIdx[L] = index;
    }

    float minDist = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) minDist = min(minDist, laneMin[L]);

    if (minDist == PXG_MAX_F32)
    {
        float maxDist = -PXG_MAX_F32;
        for (uint L = 0u; L < WARP_SIZE; ++L) maxDist = max(maxDist, laneMax[L]);
        for (uint L = 0u; L < WARP_SIZE; ++L) if (laneMax[L] == maxDist) return laneIdx[L];
        return 0u;
    }

    float laneProj[32];
    uint laneProjIdx[32];
    for (uint L = 0u; L < WARP_SIZE; ++L)
    {
        float bestProjection = PXG_MAX_F32;
        uint index = laneIdx[L];
        for (uint i = L; i < nbPolygons; i += WARP_SIZE)
        {
            const float4 pl = loadFloat4(baseWord, pOff + 16u * i);
            const float3 pn = float3(pl.x, pl.y, pl.z);
            const float pd = dot(pn, localWitness) + pl.w;
            const float toleranceDist = pd >= eps ? pd : PXG_MAX_F32;
            if (tolerance > (toleranceDist - minDist))
            {
                const float3 planeN = getNormalized(shape2Vertex(pn, scale, rot));
                const float proj = dot(planeN, n);
                if (bestProjection > proj) { index = i; bestProjection = proj; }
            }
        }
        laneProj[L] = bestProjection; laneProjIdx[L] = index;
    }
    float bestProjection = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) bestProjection = min(bestProjection, laneProj[L]);
    for (uint L = 0u; L < WARP_SIZE; ++L) if (laneProj[L] == bestProjection) return laneProjIdx[L];
    return 0u;
}

// convex-convex generateEE (T134)
bool generateEE(float3 p, float3 q, float3 normal, float3 a, float3 b,
                float4 aToBq, float3 aToBp, float inflatedRadius,
                out float3 glocalA, out float3 glocalB, out float4 gnormalPen)
{
    glocalA = float3(0, 0, 0); glocalB = float3(0, 0, 0); gnormalPen = float4(0, 0, 0, 0);
    const float expandedRatio = 0.005f;
    const float3 ab = b - a;
    const float3 n = cross(ab, normal);
    const float d = dot(n, a);
    const float np = dot(n, p);
    const float nq = dot(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    const float temp = signP * signQ;
    if (temp > 0.0f) return false;
    const float3 pq = q - p;
    const float npq = dot(n, pq);
    if (npq == 0.0f) return false;
    const float segTValue = (d - np) / npq;
    const float3 localPointA = pq * segTValue + p;
    const float3 perNormal = cross(normal, pq);
    const float3 ap = localPointA - a;
    const float nom = dot(perNormal, ap);
    const float denom = dot(perNormal, ab);
    const float tValue = nom / denom;
    const float maxT = 1.0f + expandedRatio;
    const float minT = -expandedRatio;
    if (tValue > maxT || minT > tValue) return false;
    const float3 v = ap - ab * tValue;
    const float signedDist = dot(v, normal);
    if (inflatedRadius >= signedDist)
    {
        glocalB = localPointA - v;
        glocalA = transformInvPoint(aToBq, aToBp, localPointA);
        gnormalPen = float4(normal.x, normal.y, normal.z, signedDist);
        return true;
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxEEDriver(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const EEDriverIn s = gIn[t];
    const uint b = s.baseWord;
    const uint vbBase = s.baseVB;

    const float3 normal = s.normalCD.xyz;
    const float inflatedRadius = s.normalCD.w;
    const float3 scale = s.scaleRadius.xyz;
    const float radius = s.scaleRadius.w;
    const float4 rot = s.rot;
    const float3 witness = s.witnessTol.xyz;
    const float tol = s.witnessTol.w;
    const float3 vA0 = s.vA0.xyz, vA1 = s.vA1.xyz;
    const float3 aToBp = s.aToBp.xyz;

    const uint pOff = planesByteOff(s.nbV);
    const uint dOff = polyDescByteOff(s.nbV, s.nbP);
    const uint vdOff = vertexDataByteOff(s.nbV, s.nbP, s.nbE);

    const uint bestPlaneBIdx = getPolygonIndexBlob(b, s.nbP, pOff, scale, rot, witness, normal, radius, tol);
    const uint polyDesc = loadU32(b, dOff + 4u * bestPlaneBIdx);
    const uint nbVerts = getNbVertsW(polyDesc);
    const uint vref8 = getVRef8W(polyDesc);

    // load incoming manifold: EE only appends at index >= numContactsIn, so the
    // pre-existing slot-0 face contact (when numContactsIn==1) must be preserved.
    float3 mA[4], mB[4], mN[4]; float mS[4];
    for (uint j = 0u; j < 4u; ++j) { mA[j] = float3(0, 0, 0); mB[j] = float3(0, 0, 0); mN[j] = float3(0, 0, 0); mS[j] = 0.0f; }
    if (s.numContactsIn >= 1u)
    {
        mA[0] = s.seedA.xyz; mB[0] = s.seedB.xyz; mN[0] = s.seedNp.xyz; mS[0] = s.seedNp.w;
    }

    uint numContacts = s.numContactsIn;

    // warp-strided over polygon edges, block size WARP_SIZE
    for (uint i = 0u; i < nbVerts; i += WARP_SIZE)
    {
        uint offset = 0u; // running warpScanExclusive within this block
        const uint blockNC = numContacts; // numContacts at block start (write base)
        for (uint lane = 0u; lane < WARP_SIZE; ++lane)
        {
            const uint start = i + lane;
            if (start >= nbVerts) continue;
            const uint end = (start + 1u) % nbVerts;
            const uint i0 = loadU8(b, vdOff + vref8 + start);
            const uint i1 = loadU8(b, vdOff + vref8 + end);
            const float3 v0 = gVB[vbBase + i0].xyz;
            const float3 v1 = gVB[vbBase + i1].xyz;
            float3 lA, lB; float4 lNp;
            const bool hit = generateEE(vA0, vA1, normal, v0, v1, s.aToBq, aToBp, inflatedRadius, lA, lB, lNp);
            if (hit)
            {
                if (offset < 4u)
                {
                    const uint idx = blockNC + offset;
                    if (idx < 4u)
                    {
                        mA[idx] = lA; mB[idx] = lB; mN[idx] = float3(-lNp.x, -lNp.y, -lNp.z); mS[idx] = -lNp.w;
                    }
                }
                offset += 1u;
            }
        }
        numContacts = min(numContacts + offset, 4u);
    }

    EEDriverOut r;
    for (uint j = 0u; j < 4u; ++j) { r.mA[j] = float4(mA[j], 0.0f); r.mB[j] = float4(mB[j], 0.0f); r.mNp[j] = float4(mN[j], mS[j]); }
    r.numContacts = numContacts;
    r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
    gOut[t] = r;
}
