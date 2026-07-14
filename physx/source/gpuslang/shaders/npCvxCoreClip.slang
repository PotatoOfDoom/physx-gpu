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

// npCvxCoreClip.slang
//
// Slang port of Gu::FaceClipper (GuConvexSupport.h L573-870) -- the contact
// patch builder generateContactPatch feeds after GJKEPA.  All sqrt/div-only
// arithmetic: ALL outputs verify bitwise.
//
//   - clip(): numPoints0/1 < 2 -> clipNone; 2x2 -> clip2x2; else clipNxN;
//   - clipNone: single point ((p0+p1)/2, axis.(p0-p1));
//   - clip2x2: the parallel gate |((b-a)x(d-c)).axis| > 1e-5 -> clipNone;
//     four endpoint projections each gated by TWO > -eps dots, projection
//     t = e.(p-c)/|e|^2 (division), midpoint + signed distance; the
//     backwards duplicate-merge (i from numPoints-1 down, j below i,
//     |pi-pj|^2 < 1e-6 keeps the SMALLER w in slot j and swaps the tail
//     into slot i);
//   - clipNxN: makePlanes over both faces (per edge pair: n = (e-s) x axis,
//     |n|^2 < eps^2 skip, normalizeFast (reciprocal multiply, NO zero
//     guard), the pos/neg side counts skipping i/j with d > eps, mixed ->
//     skip, posCount -> flip, plane = (-n, n.s)); makePolygon (a 2-point
//     face passes through, else PxComputeBasisVectors(axis) -- right =
//     (dir.z, 0, -dir.x).normalize() division-guarded reciprocal form when
//     |axis.y| <= 0.9999, the special up construction -- and the XY AABB of
//     both faces re-anchored at ref = (p0+p1)/2); clipPoly per plane
//     (d > -eps keeps, crossing edges with d0-d1 > eps emit the lerp
//     p0 + (p1-p0)*(d0/(d0-d1))); appending ref; the two face planes'
//     axis-projections p - axis * plane.dot(p,1) / axis.dot(n); the
//     min-w swap into slot 0; reducePolygon over PxVec4s (stride 16,
//     keep1st) and the gather;
//   - getContact returns mAxis as the normal and the up-to-4 PxVec4 points.

static const float FLT_EPS = 1.1920928955078125e-7f;
static const uint NOIDX = 0xFFFFFFFFu;
static const uint MAX_CLIP_PLANES = 8u;     // MAX_FACE_POINTS * 2
static const uint MAX_POLYGON_POINTS = 13u; // MAX_CLIP_PLANES + 4 + 1

struct ClipIn
{
    float4 p0, p1;        // GJKEPA closest points (w pad)
    float4 axis;          // separating axis (w pad)
    float4 n0, n1;        // face normals (w pad)
    uint4 counts;         // x = numPoints0, y = numPoints1
    float4 fp0[4];        // face points 0
    float4 fp1[4];        // face points 1
}

struct ClipOut
{
    uint4 counts;         // x = mNumPoints
    float4 normal;        // getContact normal (= axis)
    float4 pts[4];        // PxVec4 contact points
}

struct ClipParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ClipParams> gParams;

RWStructuredBuffer<ClipIn> gIn;    // binding 0
RWStructuredBuffer<ClipOut> gOut;  // binding 1

float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float dot4Q(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float magSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
void normalizeDiv(inout float3 v) // PxVec3::normalize -> operator/=: reciprocal multiply, m > 0 guard
{
    const float m = sqrt(magSq(v));
    if (m > 0.0f)
    {
        const float inv = 1.0f / m;
        v = float3(v.x * inv, v.y * inv, v.z * inv);
    }
}
void normalizeFastQ(inout float3 v) // PxVec3::normalizeFast: NO guard
{
    const float mag = sqrt(magSq(v));
    const float inv = 1.0f / mag;
    v = float3(v.x * inv, v.y * inv, v.z * inv);
}

// Gu::reducePolygon over the PxVec4 slots (values only; the CUDA walk uses
// stride 16 -- identical values)
uint reducePolygon4Q(const float4 pts[MAX_POLYGON_POINTS], uint count, float3 normal, out uint inds[4])
{
    inds[0] = NOIDX; inds[1] = NOIDX; inds[2] = NOIDX; inds[3] = NOIDX;
    uint pi0 = 0u, pi1 = NOIDX;
    {
        float maxDist = -3.402823466e+38f;
        const float3 p0 = pts[0].xyz;
        for (uint j = 0u; j < count; ++j)
        {
            const float dist = magSq(pts[j].xyz - p0);
            if (dist > maxDist + FLT_EPS)
            {
                maxDist = dist;
                pi1 = j;
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
        const float3 p0 = pts[pi0].xyz;
        const float3 p1 = pts[pi1].xyz;
        const float3 n01 = crossQ(normal, p1 - p0);
        float minDist = 0.0f, maxDist = 0.0f;
        for (uint i = 0u; i < count; ++i)
        {
            const float d = dot3Q(n01, pts[i].xyz - p0);
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

// makePlanes over one face (GuConvexSupport.h L732-775)
void makePlanesQ(const float3 facePoints[4], uint numPoints, float3 axis,
                 inout float4 planes[MAX_CLIP_PLANES], inout uint numPlanes)
{
    if (numPoints < 3u)
        return;

    const float eps = 1e-5f;
    for (uint i = 0u; i < numPoints; ++i)
    {
        const float3 s = facePoints[i];
        for (uint j = i + 1u; j < numPoints; ++j)
        {
            const float3 e = facePoints[j];

            float3 n = crossQ(e - s, axis);
            if (magSq(n) < eps * eps)
                continue;

            normalizeFastQ(n);

            uint posCount = 0u, negCount = 0u;
            for (uint k = 0u; k < numPoints; ++k)
            {
                if (k == i || k == j)
                    continue;

                const float d = dot3Q(facePoints[k] - s, n);
                if (d > eps)
                    posCount++;
                else
                    negCount++;
            }

            if (posCount != 0u && negCount != 0u)
                continue;

            if (posCount != 0u)
                n = -n;

            planes[numPlanes++] = float4(-n, dot3Q(n, s));
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cccClip(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const ClipIn s = gIn[t];

    const float3 point0 = s.p0.xyz, point1 = s.p1.xyz, axis = s.axis.xyz;
    const uint numPoints0 = s.counts.x, numPoints1 = s.counts.y;
    float3 face0[4], face1[4];
    for (uint i = 0u; i < 4u; ++i) { face0[i] = s.fp0[i].xyz; face1[i] = s.fp1[i].xyz; }

    float4 outPts[4] = { float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0) };
    uint outNum = 0u;

    bool doneNone = false;
    if (numPoints0 < 2u || numPoints1 < 2u)
        doneNone = true;
    else if (numPoints0 == 2u && numPoints1 == 2u)
    {
        // clip2x2 (L634-682)
        const float eps = 1e-5f;
        const float3 a = face0[0], b = face0[1];
        const float3 c = face1[0], d = face1[1];

        if (abs(dot3Q(crossQ(b - a, d - c), axis)) > eps)
            doneNone = true;
        else
        {
            uint numPoints = 0u;
            float4 points[4] = { float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0) };

            if (dot3Q(a - c, d - c) > -eps && dot3Q(a - d, c - d) > -eps)
            {
                const float3 a1 = c + ((d - c) * dot3Q(d - c, a - c)) * (1.0f / magSq(d - c)); // (v*s)/m: operator/ is reciprocal-multiply
                points[numPoints++] = float4((a + a1) * 0.5f, dot3Q(axis, a - a1));
            }
            if (dot3Q(b - c, d - c) > -eps && dot3Q(b - d, c - d) > -eps)
            {
                const float3 b1 = c + ((d - c) * dot3Q(d - c, b - c)) * (1.0f / magSq(d - c));
                points[numPoints++] = float4((b + b1) * 0.5f, dot3Q(axis, b - b1));
            }
            if (dot3Q(c - a, b - a) > -eps && dot3Q(c - b, a - b) > -eps)
            {
                const float3 c1 = a + ((b - a) * dot3Q(b - a, c - a)) * (1.0f / magSq(b - a));
                points[numPoints++] = float4((c1 + c) * 0.5f, dot3Q(axis, c1 - c));
            }
            if (dot3Q(d - a, b - a) > -eps && dot3Q(d - b, a - b) > -eps)
            {
                const float3 d1 = a + ((b - a) * dot3Q(b - a, d - a)) * (1.0f / magSq(b - a));
                points[numPoints++] = float4((d1 + d) * 0.5f, dot3Q(axis, d1 - d));
            }

            const float mergeEps = 1e-6f;
            for (int i = int(numPoints) - 1; i >= 0; --i)
                for (int j = i - 1; j >= 0; --j)
                    if (magSq(points[i].xyz - points[j].xyz) < mergeEps)
                    {
                        if (points[i].w < points[j].w) points[j] = points[i];
                        points[i] = points[--numPoints];
                        break;
                    }

            for (uint i = 0u; i < numPoints; ++i) outPts[i] = points[i];
            outNum = numPoints;
        }
    }
    else
    {
        // clipNxN (L686-729)
        uint numClipPlanes = 0u;
        float4 clipPlanes[MAX_CLIP_PLANES];
        makePlanesQ(face0, numPoints0, axis, clipPlanes, numClipPlanes);
        makePlanesQ(face1, numPoints1, axis, clipPlanes, numClipPlanes);

        uint numPolyPoints = 0u;
        float3 polyPoints[MAX_POLYGON_POINTS];

        // makePolygon (L785-828)
        if (numPoints0 == 2u)
        {
            polyPoints[numPolyPoints++] = face0[0];
            polyPoints[numPolyPoints++] = face0[1];
        }
        else if (numPoints1 == 2u)
        {
            polyPoints[numPolyPoints++] = face1[0];
            polyPoints[numPolyPoints++] = face1[1];
        }
        else
        {
            // PxComputeBasisVectors(axis, X, Y) (PxMathUtils.h L297-316)
            float3 X, Y;
            if (abs(axis.y) <= 0.9999f)
            {
                X = float3(axis.z, 0.0f, -axis.x);
                normalizeDiv(X);
                Y = float3(axis.y * X.z, axis.z * X.x - axis.x * X.z, -axis.y * X.x);
            }
            else
            {
                X = float3(1, 0, 0);
                Y = float3(0.0f, axis.z, -axis.y);
                normalizeDiv(Y);
            }

            float minX = 3.402823466e+38f, maxX = -3.402823466e+38f;
            float minY = 3.402823466e+38f, maxY = -3.402823466e+38f;
            for (uint i = 0u; i < numPoints0; ++i)
            {
                const float pX = dot3Q(face0[i], X), pY = dot3Q(face0[i], Y);
                minX = min(minX, pX); maxX = max(maxX, pX);
                minY = min(minY, pY); maxY = max(maxY, pY);
            }
            for (uint i = 0u; i < numPoints1; ++i)
            {
                const float pX = dot3Q(face1[i], X), pY = dot3Q(face1[i], Y);
                minX = min(minX, pX); maxX = max(maxX, pX);
                minY = min(minY, pY); maxY = max(maxY, pY);
            }

            const float3 refm = (point0 + point1) * 0.5f;
            const float refX = dot3Q(refm, X), refY = dot3Q(refm, Y);
            polyPoints[numPolyPoints++] = refm + X * (minX - refX) + Y * (minY - refY);
            polyPoints[numPolyPoints++] = refm + X * (maxX - refX) + Y * (minY - refY);
            polyPoints[numPolyPoints++] = refm + X * (maxX - refX) + Y * (maxY - refY);
            polyPoints[numPolyPoints++] = refm + X * (minX - refX) + Y * (maxY - refY);
        }

        // clipPoly per plane (L831-858)
        for (uint pi = 0u; pi < numClipPlanes; ++pi)
        {
            const float4 plane = clipPlanes[pi];
            float dist[MAX_POLYGON_POINTS];
            for (uint i = 0u; i < numPolyPoints; ++i)
                dist[i] = dot4Q(plane, float4(polyPoints[i], 1.0f));

            const float eps = 1e-5f;
            uint newNum = 0u;
            float3 newPoints[MAX_POLYGON_POINTS];
            for (uint i = 0u; i < numPolyPoints; ++i)
            {
                const uint i0 = i, i1 = (i + 1u) % numPolyPoints;
                const float d0 = dist[i0], d1 = dist[i1];
                const float3 q0 = polyPoints[i0], q1 = polyPoints[i1];
                if (d0 > -eps)
                {
                    newPoints[newNum++] = q0;
                    if (d1 < -eps && d0 - d1 > eps)
                        newPoints[newNum++] = q0 + (q1 - q0) * (d0 / (d0 - d1));
                }
                else if (d1 > -eps && d1 - d0 > eps)
                    newPoints[newNum++] = q0 + (q1 - q0) * (d0 / (d0 - d1));
            }
            numPolyPoints = newNum;
            for (uint i = 0u; i < newNum; ++i) polyPoints[i] = newPoints[i];
        }

        polyPoints[numPolyPoints++] = (point0 + point1) * 0.5f;

        const float4 plane0 = float4(s.n0.xyz, -dot3Q(s.n0.xyz, face0[0]));
        const float4 plane1 = float4(s.n1.xyz, -dot3Q(s.n1.xyz, face1[0]));

        float4 polyDists[MAX_POLYGON_POINTS];
        for (uint i = 0u; i < numPolyPoints; ++i)
        {
            const float3 p = polyPoints[i];
            const float3 pr0 = p - (axis * dot4Q(plane0, float4(p, 1.0f))) * (1.0f / dot3Q(axis, plane0.xyz)); // (v*s)/m form
            const float3 pr1 = p - (axis * dot4Q(plane1, float4(p, 1.0f))) * (1.0f / dot3Q(axis, plane1.xyz));
            polyDists[i] = float4((pr0 + pr1) * 0.5f, dot3Q(axis, pr0 - pr1));
        }

        uint minDistIndex = 0u;
        float minDist = polyDists[0].w;
        for (uint i = 0u; i < numPolyPoints; ++i)
            if (polyDists[i].w < minDist)
            {
                minDist = polyDists[i].w;
                minDistIndex = i;
            }
        const float4 tmp = polyDists[0]; polyDists[0] = polyDists[minDistIndex]; polyDists[minDistIndex] = tmp;

        uint inds[4];
        const uint numPoints = reducePolygon4Q(polyDists, numPolyPoints, axis, inds);

        for (uint i = 0u; i < numPoints; ++i) outPts[i] = polyDists[inds[i]];
        outNum = numPoints;
    }

    if (doneNone)
    {
        // clipNone (L621-625)
        outPts[0] = float4((point0 + point1) * 0.5f, dot3Q(axis, point0 - point1));
        outNum = 1u;
    }

    ClipOut r;
    r.counts = uint4(outNum, 0u, 0u, 0u);
    r.normal = float4(axis, 0);
    for (uint i = 0u; i < 4u; ++i) r.pts[i] = outPts[i];
    gOut[t] = r;
}
