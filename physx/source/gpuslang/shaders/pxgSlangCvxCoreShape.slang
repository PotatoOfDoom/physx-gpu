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

// pxgSlangCvxCoreShape.slang -- the convex-core SHAPE layer, shared by all five
// convexCore*Nphase kernels.  P9.67b.
//
// WHY THIS IS A MODULE AND NOT A FOURTH COPY
// ===========================================================================
// The twelve Phase-1 convex-core shaders (npCvxCoreSupport, npCvxCoreFace,
// npCvxCoreClip, npCvxCorePlaneGen, npCvxCoreCvxGen, ...) are each SELF-CONTAINED
// by the Phase-1 convention: every one carries its own copy of the support mapping
// and the face builder.  That is fine for fixtures, which are read one at a time.
// It is not fine for the engine side, where FIVE kernels
// (convexCorePlane / Convex / Trimesh / Tetmesh / ClothmeshNphase) need the same
// code and a defect fixed in one copy would be fixed in one copy.
//
// So this is the P9.51b move, made BEFORE the copies exist rather than after: the
// shared half is split out on the way to the first caller.  P9.21 spent a whole
// tranche undoing the other choice for `find()`, which had reached nine copies.
//
// HOW THE EXTRACTION WAS MADE SAFE
// ===========================================================================
// This body is npCvxCorePlaneGen.slang's lines 50-57 and 84-634, moved VERBATIM --
// not retyped, not tidied.  npCvxCorePlaneGen now imports this module and keeps
// only its own I/O structs and entry, so `gpuslang.npCvxCorePlaneGen` -- which
// passed before the move -- goes on testing exactly this code rather than a copy
// of it.  A green fixture after the split is therefore evidence about the module,
// which is the whole reason to split this way round.
//
// The `Q` suffixes are the Phase-1 originals' (quaternion-flavoured helpers named
// apart from the corpus's own dot/cross).  They are kept so the move stays a move.

module pxgSlangCvxCoreShape;

public static const float FLT_EPS = 1.1920928955078125e-7f;
public static const float FLT_MAXQ = 3.402823466e+38f;
public static const uint MAX_VERTS = 32u; // unused Convex machinery constant kept for shared helpers

public static const uint CORE_POINT = 0u;
public static const uint CORE_SEGMENT = 1u;
public static const uint CORE_BOX = 2u;
public static const uint CORE_ELLIPSOID = 3u;
public static const uint CORE_CYLINDER = 4u;
public static const uint CORE_CONE = 5u;
public static const uint CORE_POINTS = 6u;
public static const uint NOIDX = 0xFFFFFFFFu;

public float pxSign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }
public float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
public float dot4Q(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
public float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
public float magSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
public float magQ(float3 v) { return sqrt(magSq(v)); }
public float3 getNormalizedQ(float3 v)
{
    const float m = magSq(v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
public float normalizeQ(inout float3 v) // PxVec3::normalize: guarded reciprocal multiply
{
    const float m = magQ(v);
    if (m > 0.0f)
    {
        const float inv = 1.0f / m;
        v = float3(v.x * inv, v.y * inv, v.z * inv);
    }
    return m;
}
public float3 quatRotQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
public float3 quatRotInvQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
// PxQuat::getInvBasisVector0/1/2 (PxQuat.h L298-319)
public float3 invBasis0(float4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2, (-q.z * w2) + q.y * x2, (q.y * w2) + q.z * x2);
}
public float3 invBasis1(float4 q)
{
    const float y2 = q.y * 2.0f, w2 = q.w * 2.0f;
    return float3((q.z * w2) + q.x * y2, (q.w * w2) - 1.0f + q.y * y2, (-q.x * w2) + q.z * y2);
}
public float3 invBasis2(float4 q)
{
    const float z2 = q.z * 2.0f, w2 = q.w * 2.0f;
    return float3((-q.y * w2) + q.x * z2, (q.x * w2) + q.y * z2, (q.w * w2) - 1.0f + q.z * z2);
}
// PxComputeBasisVectors (PxMathUtils.h L297-316)
public void computeBasisQ(float3 dir, out float3 right, out float3 up)
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

public struct ShapeRef
{
    public uint type;
    public uint4 core[3];
    public float4 poseQ;
    public float3 poseP;
}

// tranche-164 ConvexShape::supportLocal, margin-less, all seven types
public float3 supportLocalQ(ShapeRef s, float4 ptsPool[12], float3 dir)
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

public float3 poseTransform(ShapeRef s, float3 v) { return quatRotQ(s.poseQ, v) + s.poseP; }

public float coreFQ(ShapeRef s, uint word) { return asfloat(s.core[word / 4u][word & 3u]); }
public float compAt(float3 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : v.z); }
public float3 withComp(float3 v, uint i, float x)
{
    if (i == 0u) v.x = x; else if (i == 1u) v.y = x; else v.z = x;
    return v;
}

// GuConvexSupport.h L1141-1158
public bool rotatePointsQ(float3 center, float3 refp, float3 normal, inout float3 pts[4], uint count)
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

public uint reducePolygonQ(const float3 pts[8], uint count, float3 normal, bool keep1st, out uint inds[4])
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
public uint contactFaceQ(ShapeRef sh, float4 pool[12], float3 dir, float3 refLocal,
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
