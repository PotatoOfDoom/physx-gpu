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

// npCvxCoreFace.slang
//
// Slang port of the Gu::ConvexCore contactFace family for the six analytic
// core types (GuConvexSupport.h: ePOINT L98-101, eSEGMENT L112-130, eBOX
// L145-200, eELLIPSOID L215-248, eCYLINDER L266-292, eCONE L313-349), the
// rotatePoints helper (L1141-1158) and the ConvexShape::contactFace wrapper
// (L542-552).  The ePOINTS variant (which needs reducePolygon) is the next
// tranche.
//
// Forms ported verbatim:
//   - d = dir.getNormalized() (reciprocal-multiply form) in every core;
//     thresholds dEps1 = 0.99 (~cos 8 deg), dEps2 = 0.14 (~cos 82 deg),
//     eps = FLT_EPSILON;
//   - SEGMENT/BOX/ELLIPSOID zero facePoints+faceNormal up front; CYLINDER/
//     CONE do NOT (modelled as zero-init on both sides -- the callers gate
//     on numPoints);
//   - BOX: per-axis loop (axis0 = 0..2 while numPoints == 0), the 4-point
//     face branch (|d[a0]| > 0.99 and a nondegenerate cross extent) with the
//     +,-,-,+ / +,+,-,- corner sign layout, and the 2-point edge branch
//     (|d[a0]| < 0.14, extents[a0] > eps, the two |d[a1|2]| < dEps1
//     degeneracy outs) whose faceNormal is built in place then .normalize()d
//     -- the DIVISION form (v /= m), unlike getNormalized's multiply;
//   - ELLIPSOID: both branches require degenerate radii (< FLT_EPSILON);
//   - CYLINDER/CONE cap branches call rotatePoints(center, ref, normal,
//     points, 4): dir0/dir1/axis built with the division-form normalize()
//     and FLT_EPSILON early-outs, axis snapped to +/-normal by sign of dot,
//     angle = acos(dir0.dot(dir1)), rot = PxQuat(angle, axis) (sin/cos of
//     the half angle), points rotated about center.  acos/sin/cos are NOT
//     bit-exact-portable primitives (CUDA's own acosf differs in ulps from
//     any host libm), so the verification compares rotated points to
//     tolerance -- every BRANCH decision ahead of the trig is sqrt/div-only
//     and stays bitwise;
//   - CONE side branch: sinA = r/sqrt(h*h+r*r), the rotated-frame test
//     dX = d.x*sqrt(1-sinA^2) - sqrt(1-d.x^2)*sinA, |dX| < 0.14, faceNormal
//     = (r/h, dr.y, dr.z).getNormalized();
//   - wrapper: local dir = pose.rotateInv(dir), local point =
//     pose.transformInv(point); faceNormal = pose.rotate(faceNormal);
//     facePoints[i] = pose.transform(fp[i]) + faceNormal * margin.

static const uint CORE_POINT = 0u;
static const uint CORE_SEGMENT = 1u;
static const uint CORE_BOX = 2u;
static const uint CORE_ELLIPSOID = 3u;
static const uint CORE_CYLINDER = 4u;
static const uint CORE_CONE = 5u;

static const float FLT_EPS = 1.1920928955078125e-7f;

struct FaceIn
{
    uint4 meta;       // x = coreType
    uint4 core[3];    // coreData image
    float4 poseQ;
    float4 poseP;     // w = margin
    float4 dir;       // world contact direction
    float4 refp;      // world reference point
}

struct FaceOut
{
    uint4 counts;     // x = numPoints, y = trig-path flag (rotatePoints rotation applied)
    float4 normal;    // world faceNormal
    float4 fp[4];     // world facePoints (post margin push)
}

struct FaceParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<FaceParams> gParams;

RWStructuredBuffer<FaceIn> gIn;    // binding 0
RWStructuredBuffer<FaceOut> gOut;  // binding 1

float pxSign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }
float3 getNormalizedQ(float3 v) // PxVec3::getNormalized: reciprocal multiply
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
float normalizeQ(inout float3 v) // PxVec3::normalize -> operator/=: ONE division
{                                // then three multiplies (PxVec3.h L260-267);
    const float m = sqrt(v.x * v.x + v.y * v.y + v.z * v.z); // NOT per-component
    if (m > 0.0f)                                            // division (review 165)
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
float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float coreF(FaceIn s, uint word) { return asfloat(s.core[word / 4u][word & 3u]); }
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

[shader("compute")]
[numthreads(64, 1, 1)]
void ccfContactFace(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const FaceIn s = gIn[t];

    const uint type = s.meta.x;
    const float4 poseQ = s.poseQ;
    const float3 poseP = s.poseP.xyz;
    const float margin = s.poseP.w;

    // ConvexShape::contactFace wrapper prologue (L545)
    const float3 dir = quatRotInvQ(poseQ, s.dir.xyz);
    const float3 refLocal = quatRotInvQ(poseQ, s.refp.xyz - poseP); // pose.transformInv

    uint numPoints = 0u;
    uint trig = 0u;
    float3 faceNormal = float3(0, 0, 0);
    float3 fp[4] = { float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0) };

    const float dEps1 = 0.99f, dEps2 = 0.14f;

    if (type == CORE_SEGMENT)
    {
        const float len = coreF(s, 0u);
        const float3 d = getNormalizedQ(dir);
        if (abs(d.x) < dEps2)
        {
            faceNormal = getNormalizedQ(float3(0, d.y, d.z));
            fp[0] = float3(len * 0.5f, 0, 0);
            fp[1] = float3(-len * 0.5f, 0, 0);
            numPoints = 2u;
        }
    }
    else if (type == CORE_BOX)
    {
        const float3 ext = float3(coreF(s, 0u), coreF(s, 1u), coreF(s, 2u));
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
    else if (type == CORE_ELLIPSOID)
    {
        const float3 radii = float3(coreF(s, 0u), coreF(s, 1u), coreF(s, 2u));
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
    else if (type == CORE_CYLINDER)
    {
        const float height = coreF(s, 0u), radius = coreF(s, 1u);
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
    else if (type == CORE_CONE)
    {
        const float height = coreF(s, 0u), radius = coreF(s, 1u);
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

    // wrapper epilogue (L547-549)
    const float3 worldNormal = quatRotQ(poseQ, faceNormal);
    FaceOut r;
    r.counts = uint4(numPoints, trig, 0u, 0u);
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
