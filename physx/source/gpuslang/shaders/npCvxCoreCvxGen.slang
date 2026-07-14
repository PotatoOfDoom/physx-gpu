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

// npCvxCoreCvxGen.slang
//
// Slang port of Gu::generateContacts(ConvexShape, ConvexShape) -- the full
// convex-vs-convex contact driver (GuConvexSupport.h L1162-1207), COMPOSED
// from the tranche-169/170/167/165/166 transcriptions verbatim:
//   maxDist = margin0 + margin1 + contactDist;
//   dist = computeGjkDistance(...); if dist < FLT_EPSILON -> computeEpaDepth;
//   dist > maxDist -> 0 contacts; cullDir.dot(axis) < -1e-5 -> 0 contacts;
//   point0 -= axis*margin0; point1 += axis*margin1; dist -= margins;
//   generateContactPatch = FaceClipper(contactFace(-axis, point0),
//   contactFace(axis, point1)).clip().getContact().
// Compare doctrine inherited: counts/normal/non-trig outputs BITWISE
// (modulo the documented +-0 class); contactFace cap points that passed
// rotatePoints' acos/sin/cos verify to tolerance behind the deterministic
// trig flags.  SPIR-V gets the NoContraction post-pass.

static const float FLT_EPS = 1.1920928955078125e-7f;
static const float FLT_MAXQ = 3.402823466e+38f;
static const uint MAX_VERTS = 32u;
static const uint MAX_FACES = 2u * MAX_VERTS - 4u;
static const uint MAX_EDGES = MAX_VERTS + MAX_FACES - 2u;
static const uint MAX_ITERATIONS = 100u;

static const uint CORE_POINT = 0u;
static const uint CORE_SEGMENT = 1u;
static const uint CORE_BOX = 2u;
static const uint CORE_ELLIPSOID = 3u;
static const uint CORE_CYLINDER = 4u;
static const uint CORE_CONE = 5u;
static const uint CORE_POINTS = 6u;
static const uint NOIDX = 0xFFFFFFFFu;
static const uint MAX_CLIP_PLANES = 8u;     // MAX_FACE_POINTS * 2
static const uint MAX_POLYGON_POINTS = 13u; // MAX_CLIP_PLANES + 4 + 1

struct CvxIn
{
    uint4 metaA;
    uint4 coreA[3];
    float4 poseAq, poseAp;  // poseAp.w = margin0
    uint4 metaB;
    uint4 coreB[3];
    float4 poseBq, poseBp;  // poseBp.w = margin1
    float4 cfg;             // x = contactDist, yzw = cullDir
    float4 ptsA[12];
    float4 ptsB[12];
}

struct CvxOut
{
    uint4 counts;   // x = numPoints, y = trig flags, z = path tag (0 none, 1 contacts)
    float4 normal;
    float4 points[4]; // w = dists[i]
}

struct CvxParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CvxParams> gParams;

RWStructuredBuffer<CvxIn> gIn;    // binding 0
RWStructuredBuffer<CvxOut> gOut;  // binding 1

float pxSign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }
float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float dot4Q(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float magSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float magQ(float3 v) { return sqrt(magSq(v)); }
float3 getNormalizedQ(float3 v)
{
    const float m = magSq(v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
float normalizeQ(inout float3 v) // PxVec3::normalize: guarded reciprocal multiply
{
    const float m = magQ(v);
    if (m > 0.0f)
    {
        const float inv = 1.0f / m;
        v = float3(v.x * inv, v.y * inv, v.z * inv);
    }
    return m;
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
// PxQuat::getInvBasisVector0/1/2 (PxQuat.h L298-319)
float3 invBasis0(float4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2, (-q.z * w2) + q.y * x2, (q.y * w2) + q.z * x2);
}
float3 invBasis1(float4 q)
{
    const float y2 = q.y * 2.0f, w2 = q.w * 2.0f;
    return float3((q.z * w2) + q.x * y2, (q.w * w2) - 1.0f + q.y * y2, (-q.x * w2) + q.z * y2);
}
float3 invBasis2(float4 q)
{
    const float z2 = q.z * 2.0f, w2 = q.w * 2.0f;
    return float3((-q.y * w2) + q.x * z2, (q.x * w2) + q.y * z2, (q.w * w2) - 1.0f + q.z * z2);
}
// PxComputeBasisVectors (PxMathUtils.h L297-316)
void computeBasisQ(float3 dir, out float3 right, out float3 up)
{
    if (abs(dir.y) <= 0.9999f)
    {
        right = float3(dir.z, 0.0f, -dir.x);
        normalizeQ(right);
        up = float3(dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x);
    }
    else
    {
        right = float3(1.0f, 0.0f, 0.0f);
        up = float3(0.0f, dir.z, -dir.y);
        normalizeQ(up);
    }
}

struct ShapeRef
{
    uint type;
    uint4 core[3];
    float4 poseQ;
    float3 poseP;
}

// tranche-164 ConvexShape::supportLocal, margin-less, all seven types
float3 supportLocalQ(ShapeRef s, float4 ptsPool[12], float3 dir)
{
    const uint type = s.type;
    if (type == CORE_SEGMENT)
    {
        const float len = asfloat(s.core[0].x);
        return float3(pxSign(dir.x) * len * 0.5f, 0, 0);
    }
    if (type == CORE_BOX)
    {
        const float3 ext = float3(asfloat(s.core[0].x), asfloat(s.core[0].y), asfloat(s.core[0].z));
        return float3(pxSign(dir.x) * ext.x * 0.5f,
                      pxSign(dir.y) * ext.y * 0.5f,
                      pxSign(dir.z) * ext.z * 0.5f);
    }
    if (type == CORE_ELLIPSOID)
    {
        const float3 radii = float3(asfloat(s.core[0].x), asfloat(s.core[0].y), asfloat(s.core[0].z));
        const float3 dir1 = getNormalizedQ(float3(radii.x * dir.x, radii.y * dir.y, radii.z * dir.z));
        return float3(radii.x * dir1.x, radii.y * dir1.y, radii.z * dir1.z);
    }
    if (type == CORE_CYLINDER)
    {
        const float h = asfloat(s.core[0].x) * 0.5f;
        const float r = asfloat(s.core[0].y);
        const float3 d = getNormalizedQ(dir);
        const float eps = 1e-5f;
        if (abs(d.y) < eps && abs(d.z) < eps) return float3(pxSign(d.x) * h, r, 0);
        return float3(pxSign(d.x) * h, 0, 0) + getNormalizedQ(float3(0, d.y, d.z)) * r;
    }
    if (type == CORE_CONE)
    {
        const float h = asfloat(s.core[0].x);
        const float r = asfloat(s.core[0].y);
        const float sinA = r / sqrt(h * h + r * r);
        const float halfH = h * 0.5f;
        const float3 d = getNormalizedQ(dir);
        const float eps = 1e-5f;
        if (d.x > sinA) return float3(halfH, 0, 0);
        if (abs(d.y) < eps && abs(d.z) < eps) return float3(-halfH, r, 0);
        return float3(-halfH, 0, 0) + getNormalizedQ(float3(0, d.y, d.z)) * r;
    }
    if (type == CORE_POINTS)
    {
        const uint w9 = s.core[2].y;
        const uint numPoints = w9 & 0xFFu;
        const uint stride = (w9 >> 8u) & 0xFFu;
        if ((s.core[0].x == 0u && s.core[0].y == 0u) || numPoints == 0u)
            return float3(0, 0, 0);
        const float3 S = float3(asfloat(s.core[0].z), asfloat(s.core[0].w), asfloat(s.core[1].x));
        const float4 R = float4(asfloat(s.core[1].y), asfloat(s.core[1].z), asfloat(s.core[1].w), asfloat(s.core[2].x));

        const float3 rd = quatRotQ(R, dir);
        const float3 d = quatRotInvQ(R, float3(S.x * rd.x, S.y * rd.y, S.z * rd.z));
        const float mag = magQ(d);
        if (mag < FLT_EPS)
            return float3(0, 0, 0);

        uint index = 0u;
        float maxDot = -FLT_MAXQ;
        for (uint i = 0u; i < numPoints; ++i)
        {
            const float3 p = ptsPool[(i * stride) / 16u].xyz;
            const float dot = dot3Q(d, p);
            if (dot > maxDot)
            {
                maxDot = dot;
                index = i;
            }
        }
        const float3 point = ptsPool[(index * stride) / 16u].xyz;
        const float3 rp = quatRotQ(R, point);
        return quatRotInvQ(R, float3(S.x * rp.x, S.y * rp.y, S.z * rp.z));
    }
    return float3(0, 0, 0); // CORE_POINT
}

float3 poseTransform(ShapeRef s, float3 v) { return quatRotQ(s.poseQ, v) + s.poseP; }

float coreFQ(ShapeRef s, uint word) { return asfloat(s.core[word / 4u][word & 3u]); }
float compAt(float3 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : v.z); }
float3 withComp(float3 v, uint i, float x)
{
    if (i == 0u) v.x = x; else if (i == 1u) v.y = x; else v.z = x;
    return v;
}

// GuConvexSupport.h L1141-1158
bool rotatePointsQ(float3 center, float3 refp, float3 normal, inout float3 pts[4], uint count)
{
    float3 dir0 = pts[0] - center;
    if (normalizeQ(dir0) < FLT_EPS)
        return false;
    float3 dir1 = refp - center + normal * dot3Q(normal, refp - center);
    if (normalizeQ(dir1) < FLT_EPS)
        return false;
    float3 axis = crossQ(dir0, dir1);
    if (normalizeQ(axis) < FLT_EPS)
        return false;
    axis = dot3Q(normal, axis) > 0.0f ? normal : -normal;
    const float angle = acos(clamp(dot3Q(dir0, dir1), -1.0f, 1.0f)); // PxAcos clamps (review 165)
    // PxQuat(angle, unitAxis): half angle sin/cos
    const float a = angle * 0.5f;
    const float sh = sin(a), ch = cos(a);
    const float4 rot = float4(axis.x * sh, axis.y * sh, axis.z * sh, ch);
    for (uint i = 0u; i < count; ++i)
        pts[i] = quatRotQ(rot, pts[i] - center) + center;
    return true;
}

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

// ConvexShape::contactFace core dispatch (GuConvexSupport.h L95-380 family,
// tranches 165/166 verbatim): local-frame inputs, local-frame outputs
uint contactFaceQ(ShapeRef sh, float4 pool[12], float3 dir, float3 refLocal,
                  out float3 faceNormal, out float3 fp[4], out uint trig)
{
    uint numPoints = 0u;
    trig = 0u;
    faceNormal = float3(0, 0, 0);
    fp[0] = float3(0, 0, 0); fp[1] = float3(0, 0, 0); fp[2] = float3(0, 0, 0); fp[3] = float3(0, 0, 0);

    const float dEps1 = 0.99f, dEps2 = 0.14f;

    if (sh.type == CORE_SEGMENT)
    {
        const float len = coreFQ(sh, 0u);
        const float3 d = getNormalizedQ(dir);
        if (abs(d.x) < dEps2)
        {
            faceNormal = getNormalizedQ(float3(0, d.y, d.z));
            fp[0] = float3(len * 0.5f, 0, 0);
            fp[1] = float3(-len * 0.5f, 0, 0);
            numPoints = 2u;
        }
    }
    else if (sh.type == CORE_BOX)
    {
        const float3 ext = float3(coreFQ(sh, 0u), coreFQ(sh, 1u), coreFQ(sh, 2u));
        const float3 d = getNormalizedQ(dir);
        for (uint axis0 = 0u; axis0 < 3u && numPoints == 0u; ++axis0)
        {
            const uint axis1 = (axis0 + 1u) % 3u, axis2 = (axis0 + 2u) % 3u;
            const float d0 = compAt(d, axis0), d1 = compAt(d, axis1), d2 = compAt(d, axis2);
            const float e0 = compAt(ext, axis0), e1 = compAt(ext, axis1), e2 = compAt(ext, axis2);
            if (abs(d0) > dEps1 && (e1 > FLT_EPS || e2 > FLT_EPS))
            {
                faceNormal = withComp(float3(0, 0, 0), axis0, pxSign(d0));

                fp[0] = withComp(withComp(withComp(float3(0, 0, 0), axis0, pxSign(d0) * e0 * 0.5f), axis1, e1 * 0.5f), axis2, e2 * 0.5f);
                fp[1] = withComp(withComp(withComp(float3(0, 0, 0), axis0, pxSign(d0) * e0 * 0.5f), axis1, -e1 * 0.5f), axis2, e2 * 0.5f);
                fp[2] = withComp(withComp(withComp(float3(0, 0, 0), axis0, pxSign(d0) * e0 * 0.5f), axis1, -e1 * 0.5f), axis2, -e2 * 0.5f);
                fp[3] = withComp(withComp(withComp(float3(0, 0, 0), axis0, pxSign(d0) * e0 * 0.5f), axis1, e1 * 0.5f), axis2, -e2 * 0.5f);

                numPoints = 4u;
            }
            else if (abs(d0) < dEps2 && e0 > FLT_EPS &&
                (abs(d1) < dEps1 || e2 < FLT_EPS) &&
                (abs(d2) < dEps1 || e1 < FLT_EPS))
            {
                float3 n = withComp(withComp(float3(0, 0, 0), axis1, d1), axis2, d2);
                normalizeQ(n); // PxVec3::normalize -- division form
                faceNormal = n;

                fp[0] = withComp(withComp(withComp(float3(0, 0, 0), axis0, e0 * 0.5f), axis1, pxSign(d1) * e1 * 0.5f), axis2, pxSign(d2) * e2 * 0.5f);
                fp[1] = withComp(withComp(withComp(float3(0, 0, 0), axis0, -e0 * 0.5f), axis1, pxSign(d1) * e1 * 0.5f), axis2, pxSign(d2) * e2 * 0.5f);

                numPoints = 2u;
            }
        }
    }
    else if (sh.type == CORE_ELLIPSOID)
    {
        const float3 radii = float3(coreFQ(sh, 0u), coreFQ(sh, 1u), coreFQ(sh, 2u));
        const float3 d = getNormalizedQ(dir);
        for (uint axis0 = 0u; axis0 < 3u && numPoints == 0u; ++axis0)
        {
            const uint axis1 = (axis0 + 1u) % 3u, axis2 = (axis0 + 2u) % 3u;
            const float d0 = compAt(d, axis0), d1 = compAt(d, axis1), d2 = compAt(d, axis2);
            const float r0 = compAt(radii, axis0), r1 = compAt(radii, axis1), r2 = compAt(radii, axis2);
            if (abs(d0) > dEps1 && r0 < FLT_EPS && r1 > FLT_EPS && r2 > FLT_EPS)
            {
                faceNormal = withComp(float3(0, 0, 0), axis0, pxSign(d0));
                fp[0] = withComp(float3(0, 0, 0), axis1, r1);
                fp[1] = withComp(float3(0, 0, 0), axis1, -r1);
                fp[2] = withComp(float3(0, 0, 0), axis2, r2);
                fp[3] = withComp(float3(0, 0, 0), axis2, -r2);
                numPoints = 4u;
            }
            else if (abs(d0) < dEps2 && r0 > FLT_EPS && r1 < FLT_EPS && r2 < FLT_EPS)
            {
                float3 n = withComp(withComp(float3(0, 0, 0), axis1, d1), axis2, d2);
                normalizeQ(n);
                faceNormal = n;
                fp[0] = withComp(float3(0, 0, 0), axis0, r0);
                fp[1] = withComp(float3(0, 0, 0), axis0, -r0);
                numPoints = 2u;
            }
        }
    }
    else if (sh.type == CORE_CYLINDER)
    {
        const float height = coreFQ(sh, 0u), radius = coreFQ(sh, 1u);
        const float3 d = getNormalizedQ(dir);
        if (abs(d.x) > dEps1 && radius > 0.0f)
        {
            faceNormal = float3(pxSign(d.x), 0, 0);
            fp[0] = float3(pxSign(d.x) * height * 0.5f, radius, 0);
            fp[1] = float3(pxSign(d.x) * height * 0.5f, 0, radius);
            fp[2] = float3(pxSign(d.x) * height * 0.5f, -radius, 0);
            fp[3] = float3(pxSign(d.x) * height * 0.5f, 0, -radius);
            numPoints = 4u;
            trig = rotatePointsQ(float3(pxSign(d.x) * height * 0.5f, 0, 0), refLocal, faceNormal, fp, numPoints) ? 1u : 0u;
        }
        else if (abs(d.x) < dEps2 && height > 0.0f)
        {
            const float3 dr = getNormalizedQ(float3(0, d.y, d.z));
            faceNormal = dr;
            fp[0] = float3(height * 0.5f, 0, 0) + dr * radius;
            fp[1] = float3(-height * 0.5f, 0, 0) + dr * radius;
            numPoints = 2u;
        }
    }
    else if (sh.type == CORE_CONE)
    {
        const float height = coreFQ(sh, 0u), radius = coreFQ(sh, 1u);
        if (height + radius > 0.0f)
        {
            const float3 d = getNormalizedQ(dir);
            const float eps = 1e-5f;
            if ((d.x < -dEps1 || (d.x > dEps1 && height < eps)) && radius > 0.0f)
            {
                faceNormal = float3(pxSign(d.x), 0, 0);
                fp[0] = float3(pxSign(d.x) * height * 0.5f, radius, 0);
                fp[1] = float3(pxSign(d.x) * height * 0.5f, 0, radius);
                fp[2] = float3(pxSign(d.x) * height * 0.5f, -radius, 0);
                fp[3] = float3(pxSign(d.x) * height * 0.5f, 0, -radius);
                numPoints = 4u;
                trig = rotatePointsQ(float3(pxSign(d.x) * height * 0.5f, 0, 0), refLocal, faceNormal, fp, numPoints) ? 1u : 0u;
            }
            else if (height > 0.0f)
            {
                const float sinA = radius / sqrt(height * height + radius * radius);
                const float dX = d.x * sqrt(1.0f - sinA * sinA) - sqrt(1.0f - d.x * d.x) * sinA;
                if (abs(dX) < dEps2)
                {
                    const float3 dr = getNormalizedQ(float3(0, d.y, d.z));
                    faceNormal = getNormalizedQ(float3(radius / height, dr.y, dr.z));
                    fp[0] = float3(height * 0.5f, 0, 0);
                    fp[1] = float3(-height * 0.5f, 0, 0) + dr * radius;
                    numPoints = 2u;
                }
            }
        }
    }
    // CORE_POINT (and anything else): numPoints = 0, outputs stay zeroed

    if (sh.type == CORE_POINTS)
    {
    const uint ptrWord = sh.core[0].x;
    const uint w9 = sh.core[2].y;
    const uint poolN = w9 & 0xFFu;
    const uint stride = (w9 >> 8u) & 0xFFu;
    const float3 S = float3(asfloat(sh.core[0].z), asfloat(sh.core[0].w), asfloat(sh.core[1].x));
    const float4 R = float4(asfloat(sh.core[1].y), asfloat(sh.core[1].z), asfloat(sh.core[1].w), asfloat(sh.core[2].x));

    bool ok = !((ptrWord == 0u && sh.core[0].y == 0u) || poolN == 0u); // 64-bit NULL (review 164)
    if (ok)
    {
        const float dEps1 = 0.99f;

        // three-max scan (strict >, swap cascade, pool order)
        uint index[3] = { NOIDX, NOIDX, NOIDX };
        float maxDot[3] = { -3.402823466e+38f, -3.402823466e+38f, -3.402823466e+38f };
        for (uint i = 0u; i < poolN; ++i)
        {
            const float3 raw = pool[(i * stride) / 16u].xyz;
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
                const float3 raw = pool[(index[0] * stride) / 16u].xyz;
                const float3 rp = quatRotQ(R, raw);
                p0 = quatRotInvQ(R, float3(S.x * rp.x, S.y * rp.y, S.z * rp.z));
            }

            if (index[2] == NOIDX)
            {
                const float3 raw = pool[(index[1] * stride) / 16u].xyz;
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
                const float3 raw1 = pool[(index[1] * stride) / 16u].xyz;
                const float3 rp1 = quatRotQ(R, raw1);
                const float3 p1 = quatRotInvQ(R, float3(S.x * rp1.x, S.y * rp1.y, S.z * rp1.z));
                const float3 raw2 = pool[(index[2] * stride) / 16u].xyz;
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
                const float3 raw = pool[(i * stride) / 16u].xyz;
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
    }
    return numPoints;
}
struct ConvexW
{
    ShapeRef shape;
    float accuracy;
    uint numVerts;
    float3 verts[MAX_VERTS];
}

// Convex<Support> ctor accuracy (L49-59)
float convexAccuracy(ShapeRef s, float4 pool[12])
{
    const float3 X = float3(1, 0, 0), Y = float3(0, 1, 0), Z = float3(0, 0, 1);
    const float3 mn = float3(supportLocalQ(s, pool, -X).x, supportLocalQ(s, pool, -Y).y, supportLocalQ(s, pool, -Z).z);
    const float3 mx = float3(supportLocalQ(s, pool, X).x, supportLocalQ(s, pool, Y).y, supportLocalQ(s, pool, Z).z);
    const float3 dim = mx - mn;
    const float maxExtent = max(dim.x, max(dim.y, dim.z));
    return max(maxExtent * 0.01f, FLT_EPS);
}

// Convex<Support>::getBounds (L65-78): min/max via the pose's inverse basis
void convexBounds(ShapeRef s, float4 pool[12], out float3 mn, out float3 mx)
{
    const float3 X = invBasis0(s.poseQ), Y = invBasis1(s.poseQ), Z = invBasis2(s.poseQ);
    mn = float3(poseTransform(s, supportLocalQ(s, pool, -X)).x,
                poseTransform(s, supportLocalQ(s, pool, -Y)).y,
                poseTransform(s, supportLocalQ(s, pool, -Z)).z);
    mx = float3(poseTransform(s, supportLocalQ(s, pool, X)).x,
                poseTransform(s, supportLocalQ(s, pool, Y)).y,
                poseTransform(s, supportLocalQ(s, pool, Z)).z);
}

// Convex<Support>::supportIndex (L80-110)
uint supportIndexQ(inout ConvexW c, float4 pool[12], float3 dir)
{
    const float3 v = poseTransform(c.shape, supportLocalQ(c.shape, pool, quatRotInvQ(c.shape.poseQ, dir)));
    uint closest = MAX_VERTS;
    const float WELD_DIST_SQ = c.accuracy * c.accuracy;
    float distSq = FLT_MAXQ;
    for (int i = int(c.numVerts) - 1; i >= 0; --i)
    {
        const float dSq = magSq(c.verts[i] - v);
        if (dSq < WELD_DIST_SQ)
            return uint(i);
        if (distSq > dSq)
        {
            distSq = dSq;
            closest = uint(i);
        }
    }
    if (c.numVerts == MAX_VERTS)
        return closest;
    c.verts[c.numVerts++] = v;
    return c.numVerts - 1u;
}

struct GVert { float3 p; float s; uint aI; uint bI; }


// Gu::RefGjkEpa::computeGjkDistance (tranche 169 verbatim, as a function)
void gjkDistanceQ(ShapeRef sA, ShapeRef sB, float4 poolA[12], float4 poolB[12], float maxDist,
                  out float retVal, out uint tag, out float3 pointA, out float3 pointB, out float3 axis)
{
    ConvexW cA, cB;
    cA.shape = sA; cA.numVerts = 0u; cA.accuracy = convexAccuracy(sA, poolA);
    cB.shape = sB; cB.numVerts = 0u; cB.accuracy = convexAccuracy(sB, poolB);

    // GjkDistance ctor (L132-142)
    GVert verts[4];
    for (uint i = 0u; i < 4u; ++i) { verts[i].p = float3(0, 0, 0); verts[i].s = 0.0f; verts[i].aI = 0u; verts[i].bI = 0u; }
    uint numVerts = 0u;
    uint iteration = 0u;
    float3 mClosest = float3(FLT_MAXQ, FLT_MAXQ, FLT_MAXQ);

    float3 mnA, mxA, mnB, mxB;
    convexBounds(sA, poolA, mnA, mxA);
    convexBounds(sB, poolB, mnB, mxB);
    float3 dir0 = (mnB + mxB) * 0.5f - (mnA + mxA) * 0.5f;
    if (normalizeQ(dir0) < FLT_EPS) dir0 = float3(1, 0, 0);

    {
        // ctor addPoint(dir0) (L141): numVerts == 0 so a duplicate is
        // impossible and the projection result is discarded
        ++iteration;
        const uint aI = supportIndexQ(cA, poolA, dir0);
        const uint bI = supportIndexQ(cB, poolB, -dir0);
        verts[0].p = cA.verts[aI] - cB.verts[bI];
        verts[0].s = 0.0f; verts[0].aI = aI; verts[0].bI = bI;
        numVerts = 1u;
    }

    const float accA = cA.accuracy, accB = cB.accuracy;
    const float distEps = 0.01f * min(accA, accB);

    retVal = 0.0f;
    tag = 0u;
    pointA = float3(0, 0, 0); pointB = float3(0, 0, 0); axis = float3(0, 0, 0);
    bool done = false;

    for (uint guard = 0u; guard < 2u * MAX_ITERATIONS + 8u && !done; ++guard)
    {
        // computeClosest (L175-342)
        GVert w[4];
        w[0] = verts[0]; w[1] = verts[1]; w[2] = verts[2]; w[3] = verts[3];
        w[0].s = 0.0f; w[1].s = 0.0f; w[2].s = 0.0f; w[3].s = 0.0f;
        uint nv = numVerts;

        if (nv == 1u)
        {
            w[0].s = 1.0f;
        }
        else if (nv == 2u)
        {
            const float3 a = w[0].p, b = w[1].p;
            const float3 ba = a - b;
            const float sba = dot3Q(ba, -b);
            if (sba <= 0.0f)
            {
                w[0] = w[1];
                w[0].s = 1.0f;
                nv = 1u;
            }
            else
            {
                w[0].s = sba / magSq(ba);
                w[1].s = 1.0f - w[0].s;
            }
        }
        else if (nv == 3u)
        {
            const float3 a = w[0].p, b = w[1].p, c = w[2].p;
            const float3 ca = a - c, cb = b - c;
            const float sca = dot3Q(ca, -c), scb = dot3Q(cb, -c);
            bool handled = false;
            if (sca <= 0.0f && scb <= 0.0f)
            {
                w[0] = w[2];
                w[0].s = 1.0f;
                nv = 1u;
                handled = true;
            }
            if (!handled)
            {
                const float3 abc = crossQ(b - a, c - a);
                const float iabc = 1.0f / magSq(abc);
                const float3 pabc = abc * dot3Q(abc, a) * iabc;
                const float tca = dot3Q(abc, crossQ(c - pabc, a - pabc));
                if (sca > 0.0f && tca <= 0.0f)
                {
                    w[1] = w[2];
                    w[0].s = sca / magSq(ca);
                    w[1].s = 1.0f - w[0].s;
                    nv = 2u;
                    handled = true;
                }
                if (!handled)
                {
                    const float tbc = dot3Q(abc, crossQ(b - pabc, c - pabc));
                    if (scb > 0.0f && tbc <= 0.0f)
                    {
                        w[0] = w[1];
                        w[1] = w[2];
                        w[0].s = scb / magSq(cb);
                        w[1].s = 1.0f - w[0].s;
                        nv = 2u;
                        handled = true;
                    }
                    if (!handled)
                    {
                        w[0].s = tbc * iabc;
                        w[1].s = tca * iabc;
                        w[2].s = 1.0f - w[0].s - w[1].s;
                        if (dot3Q(abc, a) > 0.0f)
                        {
                            const GVert tv = w[0]; w[0] = w[1]; w[1] = tv;
                        }
                    }
                }
            }
        }
        else if (nv == 4u)
        {
            const float3 a = w[0].p, b = w[1].p, c = w[2].p, d = w[3].p;
            const float3 da = a - d, db = b - d, dc = c - d;
            const float sda = dot3Q(da, -d), sdb = dot3Q(db, -d), sdc = dot3Q(dc, -d);
            bool handled = false;
            if (sda <= 0.0f && sdb <= 0.0f && sdc <= 0.0f)
            {
                w[0] = w[3];
                w[0].s = 1.0f;
                nv = 1u;
                handled = true;
            }
            if (!handled)
            {
                const float3 dab = crossQ(a - d, b - d);
                const float idab = 1.0f / magSq(dab);
                const float3 pdab = dab * dot3Q(dab, d) * idab;
                const float3 dbc = crossQ(b - d, c - d);
                const float idbc = 1.0f / magSq(dbc);
                const float3 pdbc = dbc * dot3Q(dbc, d) * idbc;
                const float3 dca = crossQ(c - d, a - d);
                const float idca = 1.0f / magSq(dca);
                const float3 pdca = dca * dot3Q(dca, d) * idca;
                const float tda = dot3Q(dab, crossQ(d - pdab, a - pdab));
                const float tad = dot3Q(dca, crossQ(a - pdca, d - pdca));
                if (sda > 0.0f && tda <= 0.0f && tad <= 0.0f)
                {
                    w[1] = w[3];
                    w[0].s = sda / magSq(da);
                    w[1].s = 1.0f - w[0].s;
                    nv = 2u;
                    handled = true;
                }
                if (!handled)
                {
                    const float tdb = dot3Q(dbc, crossQ(d - pdbc, b - pdbc));
                    const float tbd = dot3Q(dab, crossQ(b - pdab, d - pdab));
                    if (sdb > 0.0f && tdb <= 0.0f && tbd <= 0.0f)
                    {
                        w[0] = w[1];
                        w[1] = w[3];
                        w[0].s = sdb / magSq(db);
                        w[1].s = 1.0f - w[0].s;
                        nv = 2u;
                        handled = true;
                    }
                    if (!handled)
                    {
                        const float tcd = dot3Q(dbc, crossQ(c - pdbc, d - pdbc));
                        const float tdc = dot3Q(dca, crossQ(d - pdca, c - pdca));
                        if (sdc > 0.0f && tdc <= 0.0f && tcd <= 0.0f)
                        {
                            w[0] = w[2];
                            w[1] = w[3];
                            w[0].s = sdc / magSq(dc);
                            w[1].s = 1.0f - w[0].s;
                            nv = 2u;
                            handled = true;
                        }
                        if (!handled)
                        {
                            const float3 abc = crossQ(b - a, c - a);
                            if (tda > 0.0f && tbd > 0.0f && dot3Q(dab, d) < 0.0f && dot3Q(abc, dab) > 0.0f)
                            {
                                w[2] = w[3];
                                w[0].s = tbd * idab;
                                w[1].s = tda * idab;
                                w[2].s = 1.0f - w[0].s - w[1].s;
                                nv = 3u;
                            }
                            else if (tdb > 0.0f && tcd > 0.0f && dot3Q(dbc, d) < 0.0f && dot3Q(abc, dbc) > 0.0f)
                            {
                                w[0] = w[1];
                                w[1] = w[2];
                                w[2] = w[3];
                                w[0].s = tcd * idbc;
                                w[1].s = tdb * idbc;
                                w[2].s = 1.0f - w[0].s - w[1].s;
                                nv = 3u;
                            }
                            else if (tdc > 0.0f && tad > 0.0f && dot3Q(dca, d) < 0.0f && dot3Q(abc, dca) > 0.0f)
                            {
                                w[1] = w[3];
                                w[0].s = tdc * idca;
                                w[2].s = tad * idca;
                                w[1].s = 1.0f - w[0].s - w[2].s;
                                nv = 3u;
                            }
                        }
                    }
                }
            }
        }

        const float3 closest = w[0].p * w[0].s + w[1].p * w[1].s + w[2].p * w[2].s + w[3].p * w[3].s;
        if (magQ(closest) < magQ(mClosest) - FLT_EPS)
        {
            mClosest = closest;
            verts[0] = w[0]; verts[1] = w[1]; verts[2] = w[2]; verts[3] = w[3];
            numVerts = nv;
        }
        else
            --numVerts;

        // driver (L396-418)
        const float closestDist = magQ(mClosest);
        if (closestDist < distEps)
        {
            retVal = 0.0f;
            tag = 0u;
            done = true;
            break;
        }

        const float3 dir = -getNormalizedQ(mClosest);

        // addPoint(dir) (L155-173)
        float proj = 0.0f;
        bool projFltMax = false;
        ++iteration;
        if (iteration > MAX_ITERATIONS)
            projFltMax = true;
        else
        {
            const uint aI = supportIndexQ(cA, poolA, dir);
            const uint bI = supportIndexQ(cB, poolB, -dir);
            const float3 p = cA.verts[aI] - cB.verts[bI];
            bool dup = false;
            for (uint i = 0u; i < numVerts; ++i)
                if (verts[i].aI == aI && verts[i].bI == bI) dup = true;
            if (dup)
                projFltMax = true;
            else
            {
                verts[numVerts].p = p; verts[numVerts].s = 0.0f; verts[numVerts].aI = aI; verts[numVerts].bI = bI;
                ++numVerts;
                proj = dot3Q(p, -dir);
            }
        }
        if (projFltMax) proj = FLT_MAXQ;

        if (proj >= closestDist - distEps)
        {
            float3 pA = float3(0, 0, 0), pB = float3(0, 0, 0);
            for (uint i = 0u; i < numVerts; ++i)
            {
                pA += cA.verts[verts[i].aI] * verts[i].s;
                pB += cB.verts[verts[i].bI] * verts[i].s;
            }
            pointA = pA; pointB = pB;
            axis = -dir;
            retVal = closestDist;
            tag = 1u;
            done = true;
            break;
        }

        if (proj > maxDist)
        {
            retVal = FLT_MAXQ;
            tag = 2u;
            done = true;
            break;
        }
    }

}

// EpaDepth::makePlane (L455-472) over the three difference points.
// NOTE: the triple-cross sum is the expression llvmpipe was observed to
// CONTRACT into fma chains (1-ulp vs strict IEEE); this shader's SPIR-V is
// therefore post-processed with tools/spv_nocontract.py (see CMakeLists).
bool makePlaneQ(float3 a, float3 b, float3 c, out float4 plane)
{
    const float3 abc = crossQ(b - a, c - a) + crossQ(c - b, a - b) + crossQ(a - c, b - c);
    const float labc = magQ(abc);
    plane = float4(0, 0, 0, 0);
    if (labc < FLT_EPS) // normalEps
        return false;
    const float3 n = abc * (1.0f / labc); // PxVec3 operator/: reciprocal multiply
    const float d = dot3Q(n, -a);
    plane = float4(n, d);
    return true;
}

struct EpaPoly
{
    uint2 verts[MAX_VERTS];      // (aI, bI)
    float4 fpl[MAX_FACES];       // face planes
    uint3 fv[MAX_FACES];         // face vert indices
    uint numVerts, numFaces;
    float4 closestPlane; uint3 closestV;
    float4 bestPlane; uint3 bestV;
    float mProj;
}

uint fvAt(uint3 v, uint j) { return j == 0u ? v.x : (j == 1u ? v.y : v.z); }

// EpaDepth::addPoint (L480-565).  Returns proj; fltMax mirrors the FLT_MAX
// early-outs (the caller substitutes FLT_MAX).
float epaAddPoint(inout EpaPoly P, inout ConvexW cA, inout ConvexW cB,
                  float4 poolA[12], float4 poolB[12], float3 dir, float accuracy,
                  out bool fltMax)
{
    fltMax = false;
    if (P.numVerts > MAX_VERTS - 1u) { fltMax = true; return 0.0f; }

    const uint aI = supportIndexQ(cA, poolA, dir);
    const uint bI = supportIndexQ(cB, poolB, -dir);
    const float3 p = cA.verts[aI] - cB.verts[bI];

    const float proj = dot3Q(p, -dir);
    if (proj > P.mProj)
    {
        P.mProj = proj;
        P.bestPlane = P.closestPlane; P.bestV = P.closestV;
    }

    for (uint i = 0u; i < P.numVerts; ++i)
        if (P.verts[i].x == aI && P.verts[i].y == bI) { fltMax = true; return 0.0f; }

    P.verts[P.numVerts] = uint2(aI, bI);
    ++P.numVerts;

    if (P.numVerts < 3u)
        return 0.0f;

    if (P.numVerts == 3u)
    {
        const float3 a = cA.verts[P.verts[0].x] - cB.verts[P.verts[0].y];
        const float3 b = cA.verts[P.verts[1].x] - cB.verts[P.verts[1].y];
        const float3 c = cA.verts[P.verts[2].x] - cB.verts[P.verts[2].y];
        float4 plane0 = float4(0, 0, 0, 0), plane1 = float4(0, 0, 0, 0);
        const bool ok0 = makePlaneQ(a, b, c, plane0);
        const bool ok1 = ok0 && makePlaneQ(b, a, c, plane1); // short-circuit order as CUDA's ||
        if (!ok0 || !ok1) { fltMax = true; return 0.0f; }

        P.fpl[0] = plane0; P.fv[0] = uint3(0u, 1u, 2u);
        P.fpl[1] = plane1; P.fv[1] = uint3(1u, 0u, 2u);
        P.numFaces = 2u;
        return 0.0f;
    }

    const float testEps = 0.01f * accuracy;
    uint2 edges[MAX_EDGES];
    uint numEdges = 0u;
    for (int i = int(P.numFaces) - 1; i >= 0; --i)
    {
        if (dot4Q(P.fpl[i], float4(p, 1.0f)) > testEps)
        {
            for (uint j = 0u; j < 3u; ++j)
            {
                bool add = true;
                const uint v0 = fvAt(P.fv[i], j), v1 = fvAt(P.fv[i], (j + 1u) % 3u);
                for (int k = int(numEdges) - 1; k >= 0; --k)
                {
                    if (v0 == edges[k].y && v1 == edges[k].x)
                    {
                        add = false;
                        edges[k] = edges[numEdges - 1u];
                        --numEdges;
                        break;
                    }
                }
                if (add)
                {
                    edges[numEdges] = uint2(v0, v1);
                    ++numEdges;
                }
            }
            P.fpl[i] = P.fpl[P.numFaces - 1u];
            P.fv[i] = P.fv[P.numFaces - 1u];
            --P.numFaces;
        }
    }

    if (numEdges == 0u) { fltMax = true; return 0.0f; }
    if (P.numFaces > MAX_FACES - numEdges) { fltMax = true; return 0.0f; }

    for (uint i = 0u; i < numEdges; ++i)
    {
        const uint i0 = edges[i].x, i1 = edges[i].y, i2 = P.numVerts - 1u;
        const float3 a = cA.verts[P.verts[i0].x] - cB.verts[P.verts[i0].y];
        const float3 b = cA.verts[P.verts[i1].x] - cB.verts[P.verts[i1].y];
        const float3 c = cA.verts[P.verts[i2].x] - cB.verts[P.verts[i2].y];
        float4 plane;
        if (!makePlaneQ(a, b, c, plane)) { fltMax = true; return 0.0f; }
        P.fpl[P.numFaces] = plane; P.fv[P.numFaces] = uint3(i0, i1, i2);
        ++P.numFaces;
    }

    return proj;
}

// Gu::RefGjkEpa::computeEpaDepth (tranche 170 verbatim, as a function)
void epaDepthQ(ShapeRef sA, ShapeRef sB, float4 poolA[12], float4 poolB[12],
               out float retVal, out uint tag, out float3 pointA, out float3 pointB, out float3 axis)
{
    ConvexW cA, cB;
    cA.shape = sA; cA.numVerts = 0u; cA.accuracy = convexAccuracy(sA, poolA);
    cB.shape = sB; cB.numVerts = 0u; cB.accuracy = convexAccuracy(sB, poolB);
    const float accuracy = min(cA.accuracy, cB.accuracy); // getAccuracy()

    EpaPoly P;
    for (uint i = 0u; i < MAX_VERTS; ++i) P.verts[i] = uint2(0u, 0u);
    for (uint i = 0u; i < MAX_FACES; ++i) { P.fpl[i] = float4(0, 0, 0, 0); P.fv[i] = uint3(0u, 0u, 0u); }
    P.numVerts = 0u; P.numFaces = 0u;
    P.closestPlane = float4(0, 0, 0, 0); P.closestV = uint3(0u, 0u, 0u);
    P.bestPlane = float4(0, 0, 0, 0); P.bestV = uint3(0u, 0u, 0u);
    P.mProj = -FLT_MAXQ;

    // EpaDepth ctor (L424-440)
    float3 mnA, mxA, mnB, mxB;
    convexBounds(sA, poolA, mnA, mxA);
    convexBounds(sB, poolB, mnB, mxB);
    float3 dir0 = (mnB + mxB) * 0.5f - (mnA + mxA) * 0.5f;
    if (normalizeQ(dir0) < FLT_EPS) dir0 = float3(1, 0, 0);
    float3 dir1, dir2;
    computeBasisQ(dir0, dir1, dir2);

    bool fmIgnored;
    epaAddPoint(P, cA, cB, poolA, poolB, dir0, accuracy, fmIgnored);
    epaAddPoint(P, cA, cB, poolA, poolB, -dir0, accuracy, fmIgnored);
    epaAddPoint(P, cA, cB, poolA, poolB, dir1, accuracy, fmIgnored);
    if (P.numVerts < 3u) epaAddPoint(P, cA, cB, poolA, poolB, -dir1, accuracy, fmIgnored);
    if (P.numVerts < 3u) epaAddPoint(P, cA, cB, poolA, poolB, dir2, accuracy, fmIgnored);
    if (P.numVerts < 3u) epaAddPoint(P, cA, cB, poolA, poolB, -dir2, accuracy, fmIgnored);
    P.mProj = -FLT_MAXQ;

    retVal = 0.0f;
    tag = 0u;
    pointA = float3(0, 0, 0); pointB = float3(0, 0, 0); axis = float3(0, 0, 0);

    if (P.numFaces == 0u)
    {
        retVal = FLT_MAXQ; // !isValid()
        tag = 0u;
    }
    else
    {
        const float distEps = 0.01f * accuracy;
        bool done = false;
        // each pass either exits or pushes a vert; MAX_VERTS bounds the loop
        for (uint guard = 0u; guard < MAX_VERTS + 8u && !done; ++guard)
        {
            // computeClosest (L568-590)
            uint closest = 0u;
            float closestW = -FLT_MAXQ;
            for (uint i = 0u; i < P.numFaces; ++i)
            {
                const float w = P.fpl[i].w;
                if (w > closestW)
                {
                    closest = i;
                    closestW = w;
                }
            }
            P.closestPlane = P.fpl[closest]; P.closestV = P.fv[closest];
            if (P.numFaces == 2u) { P.bestPlane = P.closestPlane; P.bestV = P.closestV; }

            const float closestDist = P.closestPlane.w;
            const float3 dir = P.closestPlane.xyz;

            bool fm;
            float proj = epaAddPoint(P, cA, cB, poolA, poolB, dir, accuracy, fm);
            if (fm) proj = FLT_MAXQ;

            if (proj >= closestDist - distEps)
            {
                // computePoints (L609-624)
                const uint2 v0 = P.verts[P.bestV.x], v1 = P.verts[P.bestV.y], v2 = P.verts[P.bestV.z];
                const float3 va0 = cA.verts[v0.x], va1 = cA.verts[v1.x], va2 = cA.verts[v2.x];
                const float3 vb0 = cB.verts[v0.y], vb1 = cB.verts[v1.y], vb2 = cB.verts[v2.y];
                const float3 a = va0 - vb0, b = va1 - vb1, c = va2 - vb2;
                const float3 abc = crossQ(b - a, c - a);
                const float iabc = 1.0f / magSq(abc);
                const float3 pabc = abc * dot3Q(abc, a) * iabc;
                const float tbc = dot3Q(abc, crossQ(b - pabc, c - pabc));
                const float tca = dot3Q(abc, crossQ(c - pabc, a - pabc));
                const float tab = dot3Q(abc, crossQ(a - pabc, b - pabc));
                const float sa = tbc * iabc, sb = tca * iabc, sc = tab * iabc;
                pointA = va0 * sa + va1 * sb + va2 * sc;
                pointB = vb0 * sa + vb1 * sb + vb2 * sc;
                axis = -P.bestPlane.xyz;
                retVal = P.bestPlane.w;
                tag = 1u;
                done = true;
            }
        }
    }

}

void normalizeDivC(inout float3 v) // PxVec3::normalize -> operator/=: reciprocal multiply, m > 0 guard
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

// Gu::FaceClipper::clip + getContact (tranche 167 verbatim, as a function)
uint faceClipQ(float3 faceIn0[4], uint numPoints0, float3 faceNormal0,
               float3 faceIn1[4], uint numPoints1, float3 faceNormal1,
               float3 point0, float3 point1, float3 axisIn,
               out float3 outNormalW, out float4 outPtsW[4])
{
    const float3 axis = axisIn;
    float3 face0[4], face1[4];
    for (uint i = 0u; i < 4u; ++i) { face0[i] = faceIn0[i]; face1[i] = faceIn1[i]; }

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
                normalizeDivC(X);
                Y = float3(axis.y * X.z, axis.z * X.x - axis.x * X.z, -axis.y * X.x);
            }
            else
            {
                X = float3(1, 0, 0);
                Y = float3(0.0f, axis.z, -axis.y);
                normalizeDivC(Y);
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

        const float4 plane0 = float4(faceNormal0, -dot3Q(faceNormal0, face0[0]));
        const float4 plane1 = float4(faceNormal1, -dot3Q(faceNormal1, face1[0]));

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


    outNormalW = axisIn; // getContact: normal = mAxis
    for (uint oi = 0u; oi < 4u; ++oi)
        outPtsW[oi] = oi < outNum ? outPts[oi] : float4(0, 0, 0, 0);
    return outNum;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void cvgContacts(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CvxIn s = gIn[t];

    ShapeRef shA, shB;
    shA.type = s.metaA.x; shA.core = s.coreA; shA.poseQ = s.poseAq; shA.poseP = s.poseAp.xyz;
    shB.type = s.metaB.x; shB.core = s.coreB; shB.poseQ = s.poseBq; shB.poseP = s.poseBp.xyz;
    const float margin0 = s.poseAp.w, margin1 = s.poseBp.w;
    const float contactDist = s.cfg.x;
    const float3 cullDir = float3(s.cfg.y, s.cfg.z, s.cfg.w);

    // generateContacts (GuConvexSupport.h L1169-1201)
    const float maxDist = margin0 + margin1 + contactDist;

    float dist; uint tagG;
    float3 point0, point1, axis;
    gjkDistanceQ(shA, shB, s.ptsA, s.ptsB, maxDist, dist, tagG, point0, point1, axis);

    if (dist < FLT_EPS)
    {
        uint tagE;
        epaDepthQ(shA, shB, s.ptsA, s.ptsB, dist, tagE, point0, point1, axis);
    }

    uint numPoints = 0u;
    uint trig = 0u;
    float3 normal = float3(0, 0, 0);
    float4 patch[4] = { float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0) };
    bool emit = true;

    if (dist > maxDist)
        emit = false;

    const float testEps = 1e-5f;
    if (emit && dot3Q(cullDir, axis) < -testEps)
        emit = false;

    if (emit)
    {
        point0 -= axis * margin0;
        point1 += axis * margin1;
        dist -= margin0 + margin1;

        // generateContactPatch (L872-882): FaceClipper ctor's two
        // ConvexShape::contactFace wrapper calls, then clip + getContact
        uint trig0, trig1;
        float3 fn0, fn1;
        float3 fpl[4];
        float3 wfp0[4], wfp1[4];

        const uint np0 = contactFaceQ(shA, s.ptsA, quatRotInvQ(shA.poseQ, -axis),
                                      quatRotInvQ(shA.poseQ, point0 - shA.poseP), fn0, fpl, trig0);
        const float3 wn0 = quatRotQ(shA.poseQ, fn0);
        for (uint i = 0u; i < 4u; ++i)
            wfp0[i] = i < np0 ? (quatRotQ(shA.poseQ, fpl[i]) + shA.poseP) + wn0 * margin0 : float3(0, 0, 0);

        const uint np1 = contactFaceQ(shB, s.ptsB, quatRotInvQ(shB.poseQ, axis),
                                      quatRotInvQ(shB.poseQ, point1 - shB.poseP), fn1, fpl, trig1);
        const float3 wn1 = quatRotQ(shB.poseQ, fn1);
        for (uint i = 0u; i < 4u; ++i)
            wfp1[i] = i < np1 ? (quatRotQ(shB.poseQ, fpl[i]) + shB.poseP) + wn1 * margin1 : float3(0, 0, 0);

        numPoints = faceClipQ(wfp0, np0, wn0, wfp1, np1, wn1, point0, point1, axis, normal, patch);
        trig = trig0 | (trig1 << 1u);
    }

    CvxOut r;
    r.counts = uint4(numPoints, trig, emit ? 1u : 0u, 0u);
    r.normal = float4(normal, 0);
    for (uint i = 0u; i < 4u; ++i)
        r.points[i] = i < numPoints ? patch[i] : float4(0, 0, 0, 0);
    gOut[t] = r;
}
