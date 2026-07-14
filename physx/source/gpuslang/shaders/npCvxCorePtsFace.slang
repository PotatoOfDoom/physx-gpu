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

// npCvxCorePtsFace.slang
//
// Slang port of Gu::reducePolygon (GuConvexSupport.h L1053-1136) and the
// ePOINTS contactFace (L381-477) with the ConvexShape wrapper -- the last
// piece of the convex-core contactFace family.  Everything here is
// sqrt/div-only arithmetic: ALL outputs verify bitwise.
//
//   - reducePolygon: farthest-pair seed (keep1st pins index 0 and scans
//     only j; else the full i x j scan), both with the strict
//     "dist > maxDist + FLT_EPSILON" acceptance; degenerate outs
//     (no pair -> 0, pi0 == pi1 -> 1); the cross-line band scan
//     (n01 = normal x (p1 - p0), d < minDist - eps / d > maxDist + eps
//     with min/max starting at ZERO) and the duplicate-filtered emit
//     order pi0, pi1, pi2, pi3;
//   - ePOINTS contactFace: pool points pushed through the
//     R.rotateInv(S.multiply(R.rotate(p))) vertex2shape form; the
//     three-max scan (swap cascade over the running dot/ind pair,
//     STRICT >, pool order); the normal-candidate chain
//     (2 points: n = (dir x e) x e normalized, accept if dir.n > 0.99;
//     3 points: face normal n0 accepted at > 0.99 / < -0.99 with sign,
//     else the two edge-rejection candidates (e x dir) x e); the
//     coplanar accumulation loop over the WHOLE pool
//     (|n.p - n.p0| < 1e-4): prepend the candidate, swap the head with
//     the best dir-dot survivor, reducePolygon(keep1st = true), gather
//     by the returned indices; the final numPoints >= 2 gate;
//   - the wrapper prologue/epilogue as in tranche 165 (rotateInv /
//     transformInv in, world rotate + margin push out).

static const float FLT_EPS = 1.1920928955078125e-7f;
static const uint NOIDX = 0xFFFFFFFFu;

struct PtsIn
{
    uint4 meta;       // x = mode/coreType selector is implicit per entry; x = keep1st (reduce entry)
    uint4 core[3];    // ePOINTS coreData image (tranche-163 packing)
    float4 poseQ;
    float4 poseP;     // w = margin
    float4 dir;       // world dir (face entry) / normal (reduce entry)
    float4 refp;      // world ref point (unused by ePOINTS but part of the wrapper signature)
    float4 pts[32];   // vertex pool (face entry) / input points (reduce entry, count in meta.y)
}

struct PtsOut
{
    uint4 counts;     // x = numPoints; reduce entry: y..w unused
    uint4 inds;       // reduce entry: the four indices (NOIDX pad)
    float4 normal;    // face entry: world faceNormal
    float4 fp[4];     // face entry: world facePoints
}

struct PtsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PtsParams> gParams;

RWStructuredBuffer<PtsIn> gIn;    // binding 0
RWStructuredBuffer<PtsOut> gOut;  // binding 1

float3 getNormalizedQ(float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
float3 quatRotQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotInvQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float magSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// GuConvexSupport.h L1053-1136 (array form; the two call sites' strides only
// change the memory walk, not the values)
uint reducePolygonQ(const float3 pts[8], uint count, float3 normal, bool keep1st, out uint inds[4])
{
    inds[0] = NOIDX; inds[1] = NOIDX; inds[2] = NOIDX; inds[3] = NOIDX;
    uint pi0 = NOIDX, pi1 = NOIDX;
    if (keep1st)
    {
        pi0 = 0u;
        float maxDist = -3.402823466e+38f;
        const float3 p0 = pts[0];
        for (uint j = 0u; j < count; ++j)
        {
            const float dist = magSq(pts[j] - p0);
            if (dist > maxDist + FLT_EPS)
            {
                maxDist = dist;
                pi1 = j;
            }
        }
    }
    else
    {
        float maxDist = -3.402823466e+38f;
        for (uint i = 0u; i < count; ++i)
        {
            const float3 p0 = pts[i];
            for (uint j = 0u; j < count; ++j)
            {
                const float dist = magSq(pts[j] - p0);
                if (dist > maxDist + FLT_EPS)
                {
                    maxDist = dist;
                    pi0 = i; pi1 = j;
                }
            }
        }
    }

    if (pi0 == NOIDX || pi1 == NOIDX)
        return 0u;

    if (pi0 == pi1)
    {
        inds[0] = pi0;
        return 1u;
    }

    uint pi2 = NOIDX, pi3 = NOIDX;
    {
        const float3 p0 = pts[pi0];
        const float3 p1 = pts[pi1];
        const float3 n01 = crossQ(normal, p1 - p0);
        float minDist = 0.0f, maxDist = 0.0f;
        for (uint i = 0u; i < count; ++i)
        {
            const float d = dot3Q(n01, pts[i] - p0);
            if (d < minDist - FLT_EPS)
            {
                pi2 = i;
                minDist = d;
            }
            if (d > maxDist + FLT_EPS)
            {
                pi3 = i;
                maxDist = d;
            }
        }
    }

    uint numPoints = 0u;
    inds[numPoints++] = pi0;
    inds[numPoints++] = pi1;
    if (pi2 != NOIDX && pi2 != pi0 && pi2 != pi1)
        inds[numPoints++] = pi2;
    if (pi3 != NOIDX && pi3 != pi0 && pi3 != pi1 && pi3 != pi2)
        inds[numPoints++] = pi3;

    return numPoints;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccpReduce(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PtsIn s = gIn[t];

    float3 pts[8];
    for (uint i = 0u; i < 8u; ++i) pts[i] = s.pts[i].xyz;
    const uint count = min(s.meta.y, 8u);

    uint inds[4];
    const uint n = reducePolygonQ(pts, count, s.dir.xyz, s.meta.x != 0u, inds);

    PtsOut r;
    r.counts = uint4(n, 0u, 0u, 0u);
    r.inds = uint4(inds[0], inds[1], inds[2], inds[3]);
    r.normal = float4(0, 0, 0, 0);
    for (uint i = 0u; i < 4u; ++i) r.fp[i] = float4(0, 0, 0, 0);
    gOut[t] = r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccpPointsFace(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PtsIn s = gIn[t];

    const float4 poseQ = s.poseQ;
    const float3 poseP = s.poseP.xyz;
    const float margin = s.poseP.w;

    // wrapper prologue
    const float3 dir = quatRotInvQ(poseQ, s.dir.xyz);

    uint numPoints = 0u;
    float3 faceNormal = float3(0, 0, 0);
    float3 fp[4] = { float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0) };

    const uint ptrWord = s.core[0].x;
    const uint w9 = s.core[2].y;
    const uint poolN = w9 & 0xFFu;
    const uint stride = (w9 >> 8u) & 0xFFu;
    const float3 S = float3(asfloat(s.core[0].z), asfloat(s.core[0].w), asfloat(s.core[1].x));
    const float4 R = float4(asfloat(s.core[1].y), asfloat(s.core[1].z), asfloat(s.core[1].w), asfloat(s.core[2].x));

    bool ok = !((ptrWord == 0u && s.core[0].y == 0u) || poolN == 0u); // 64-bit NULL (review 164)
    if (ok)
    {
        const float dEps1 = 0.99f;

        // three-max scan (strict >, swap cascade, pool order)
        uint index[3] = { NOIDX, NOIDX, NOIDX };
        float maxDot[3] = { -3.402823466e+38f, -3.402823466e+38f, -3.402823466e+38f };
        for (uint i = 0u; i < poolN; ++i)
        {
            const float3 raw = s.pts[(i * stride) / 16u].xyz;
            const float3 rp = quatRotQ(R, raw);
            const float3 p = quatRotInvQ(R, float3(S.x * rp.x, S.y * rp.y, S.z * rp.z)); // getPoint(i)
            float dot = dot3Q(dir, p);
            uint ind = i;
            for (uint j = 0u; j < 3u; ++j)
            {
                if (dot > maxDot[j])
                {
                    const float td = maxDot[j]; maxDot[j] = dot; dot = td;
                    const uint ti = index[j]; index[j] = ind; ind = ti;
                }
            }
        }

        if (index[1] == NOIDX)
            ok = false;

        float3 p0 = float3(0, 0, 0);
        if (ok)
        {
            {
                const float3 raw = s.pts[(index[0] * stride) / 16u].xyz;
                const float3 rp = quatRotQ(R, raw);
                p0 = quatRotInvQ(R, float3(S.x * rp.x, S.y * rp.y, S.z * rp.z));
            }

            if (index[2] == NOIDX)
            {
                const float3 raw = s.pts[(index[1] * stride) / 16u].xyz;
                const float3 rp = quatRotQ(R, raw);
                const float3 p1 = quatRotInvQ(R, float3(S.x * rp.x, S.y * rp.y, S.z * rp.z));
                const float3 n = getNormalizedQ(crossQ(crossQ(dir, p1 - p0), p1 - p0));
                const float dot = dot3Q(dir, n);
                if (dot > dEps1)
                    faceNormal = n;
                else
                    ok = false;
            }
            else
            {
                const float3 raw1 = s.pts[(index[1] * stride) / 16u].xyz;
                const float3 rp1 = quatRotQ(R, raw1);
                const float3 p1 = quatRotInvQ(R, float3(S.x * rp1.x, S.y * rp1.y, S.z * rp1.z));
                const float3 raw2 = s.pts[(index[2] * stride) / 16u].xyz;
                const float3 rp2 = quatRotQ(R, raw2);
                const float3 p2 = quatRotInvQ(R, float3(S.x * rp2.x, S.y * rp2.y, S.z * rp2.z));

                const float3 n0 = getNormalizedQ(crossQ(p1 - p0, p2 - p0));
                const float dot0 = dot3Q(dir, n0);
                if (dot0 > dEps1)
                    faceNormal = n0;
                else if (dot0 < -dEps1)
                    faceNormal = -n0;
                else
                {
                    const float3 n1 = getNormalizedQ(crossQ(crossQ(p1 - p0, dir), p1 - p0));
                    const float dot1 = dot3Q(dir, n1);
                    if (dot1 > dEps1)
                        faceNormal = n1;
                    else
                    {
                        const float3 n2 = getNormalizedQ(crossQ(crossQ(p2 - p0, dir), p2 - p0));
                        const float dot2 = dot3Q(dir, n2);
                        if (dot2 > dEps1)
                            faceNormal = n2;
                        else
                            ok = false;
                    }
                }
            }
        }

        if (ok)
        {
            const float maxP = dot3Q(faceNormal, p0);
            for (uint i = 0u; i < poolN; ++i)
            {
                const float3 raw = s.pts[(i * stride) / 16u].xyz;
                const float3 rp = quatRotQ(R, raw);
                const float3 p = quatRotInvQ(R, float3(S.x * rp.x, S.y * rp.y, S.z * rp.z));
                const float pP = dot3Q(faceNormal, p);
                if (abs(pP - maxP) < 1e-4f)
                {
                    float3 nfp[8];
                    nfp[0] = p;
                    for (uint j = 0u; j < numPoints; ++j) nfp[j + 1u] = fp[j];
                    for (uint j = 5u; j < 8u; ++j) nfp[j] = float3(0, 0, 0);
                    for (uint j = 0u; j < numPoints; ++j)
                    {
                        if (dot3Q(nfp[0], dir) < dot3Q(nfp[j + 1u], dir))
                        {
                            const float3 tv = nfp[0]; nfp[0] = nfp[j + 1u]; nfp[j + 1u] = tv;
                        }
                    }
                    uint inds[4];
                    numPoints = reducePolygonQ(nfp, numPoints + 1u, dir, true, inds);
                    float3 gathered[4];
                    for (uint j = 0u; j < numPoints; ++j) gathered[j] = nfp[inds[j]];
                    for (uint j = 0u; j < numPoints; ++j) fp[j] = gathered[j];
                }
            }
            if (numPoints < 2u)
            {
                numPoints = 0u;
                ok = false;
            }
        }
    }

    if (!ok)
        numPoints = 0u;

    // wrapper epilogue
    const float3 worldNormal = quatRotQ(poseQ, faceNormal);
    PtsOut r;
    r.counts = uint4(numPoints, 0u, 0u, 0u);
    r.inds = uint4(NOIDX, NOIDX, NOIDX, NOIDX);
    r.normal = float4(worldNormal, 0);
    for (uint i = 0u; i < 4u; ++i)
    {
        const float3 wp = (i < numPoints)
            ? (quatRotQ(poseQ, fp[i]) + poseP) + worldNormal * margin
            : float3(0, 0, 0);
        r.fp[i] = float4(wp, 0);
    }
    gOut[t] = r;
}
