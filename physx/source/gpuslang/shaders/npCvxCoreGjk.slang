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

// npCvxCoreGjk.slang
//
// Slang port of Gu::RefGjkEpa::computeGjkDistance (GuRefGjkEpa.h L44-419):
// the Convex<Support> vertex-welding wrapper (MAX_VERTS = 32; accuracy =
// max(1% of the local max extent, FLT_EPSILON); getBounds over the pose's
// inverse basis vectors; supportIndex's BACKWARD weld scan with the
// weld-dist^2 early-out, strict closest tracking and the full-cache
// fallback), the GjkDistance simplex core (seed direction from the bounds
// centers, division-guarded reciprocal normalize with the (1,0,0) fallback;
// addPoint's iteration cap 100, duplicate (aI,bI) FLT_MAX out and
// p.dot(-dir) projection; computeClosest's 1/2/3/4-vertex Johnson cases
// ported expression-for-expression including the abc.dot(a) > 0 swap and
// the closest-improvement test |closest| < |mClosest| - FLT_EPSILON with
// the --mNumVerts backtrack), and the computeGjkDistance driver loop
// (distEps = 0.01 * min accuracy; 0 on overlap, closestDist + points +
// axis on convergence, FLT_MAX past maxDist).  The Support is the
// tranche-164 ConvexShape::supportLocal (margin-less), reproduced verbatim
// for all seven core types.  Everything is sqrt/div arithmetic: ALL outputs
// verify bitwise.

static const float FLT_EPS = 1.1920928955078125e-7f;
static const float FLT_MAXQ = 3.402823466e+38f;
static const uint MAX_VERTS = 32u;
static const uint MAX_ITERATIONS = 100u;

static const uint CORE_POINT = 0u;
static const uint CORE_SEGMENT = 1u;
static const uint CORE_BOX = 2u;
static const uint CORE_ELLIPSOID = 3u;
static const uint CORE_CYLINDER = 4u;
static const uint CORE_CONE = 5u;
static const uint CORE_POINTS = 6u;

struct GjkIn
{
    uint4 metaA;      // x = coreType A
    uint4 coreA[3];
    float4 poseAq, poseAp;
    uint4 metaB;
    uint4 coreB[3];
    float4 poseBq, poseBp;
    float4 cfg;       // x = maxDist
    float4 ptsA[12];  // ePOINTS pools
    float4 ptsB[12];
}

struct GjkOut
{
    float4 result;    // x = return value (0 / dist / FLT_MAX), y = case tag (0 overlap, 1 dist, 2 maxDist)
    float4 pointA, pointB, axis;
}

struct GjkParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<GjkParams> gParams;

RWStructuredBuffer<GjkIn> gIn;    // binding 0
RWStructuredBuffer<GjkOut> gOut;  // binding 1

float pxSign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }
float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
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

struct ShapeRef
{
    uint type;
    uint4 core[3];
    float4 poseQ;
    float3 poseP;
}

// tranche-164 ConvexShape::supportLocal, margin-less, all seven types
// P9.86: `PxVec3(a, 0, 0) + v` (GuConvexSupport.h cylinder/cone localSupport and
// contactFace rims).  Spelled as that vector add, llvmpipe folds the literal-zero lanes
// `0 + v.y` / `0 + v.z` to v.y / v.z even under SignedZeroInfNanPreserve (measured), which
// keeps a -0 the CUDA turns into +0.  The x lane is a real add; the zero lanes are the
// IEEE `0 + x` spelled as a select.
float3 cgjAxisPlus(float a, float3 v)
{
    return float3(a + v.x, (v.y == 0.0f) ? 0.0f : v.y, (v.z == 0.0f) ? 0.0f : v.z);
}

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
        // P9.86: the CUDA's PxMat33::createDiagonal(radii) transformTranspose / transform,
        // WITH its explicit zero terms: component i is (r_i*v_i + 0*v_j) + 0*v_k
        // (PxMat33.h L344-353, PxVec3::dot L272-275), so a -0 in v_i gives +0 unless all
        // three terms are -0.  Every shader is stamped SignedZeroInfNanPreserve, under which
        // llvmpipe keeps x*0 and x + (computed zero) IEEE (measured, p86SignedZero); slangc
        // leaves these adds alone because no operand is a literal zero.
        const float3 dir1 = getNormalizedQ(float3((radii.x * dir.x + 0.0f * dir.y) + 0.0f * dir.z,
                                                  (0.0f * dir.x + radii.y * dir.y) + 0.0f * dir.z,
                                                  (0.0f * dir.x + 0.0f * dir.y) + radii.z * dir.z));
        return float3((radii.x * dir1.x + 0.0f * dir1.y) + 0.0f * dir1.z,
                      (0.0f * dir1.x + radii.y * dir1.y) + 0.0f * dir1.z,
                      (0.0f * dir1.x + 0.0f * dir1.y) + radii.z * dir1.z);
    }
    if (type == CORE_CYLINDER)
    {
        const float h = asfloat(s.core[0].x) * 0.5f;
        const float r = asfloat(s.core[0].y);
        const float3 d = getNormalizedQ(dir);
        const float eps = 1e-5f;
        if (abs(d.y) < eps && abs(d.z) < eps) return float3(pxSign(d.x) * h, r, 0);
        return cgjAxisPlus(pxSign(d.x) * h, getNormalizedQ(float3(0, d.y, d.z)) * r);
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
        return cgjAxisPlus(-halfH, getNormalizedQ(float3(0, d.y, d.z)) * r);
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

[shader("compute")]
[numthreads(32, 1, 1)]
void cgjDistance(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const GjkIn s = gIn[t];

    ShapeRef sA, sB;
    sA.type = s.metaA.x; sA.core = s.coreA; sA.poseQ = s.poseAq; sA.poseP = s.poseAp.xyz;
    sB.type = s.metaB.x; sB.core = s.coreB; sB.poseQ = s.poseBq; sB.poseP = s.poseBp.xyz;

    ConvexW cA, cB;
    cA.shape = sA; cA.numVerts = 0u; cA.accuracy = convexAccuracy(sA, s.ptsA);
    cB.shape = sB; cB.numVerts = 0u; cB.accuracy = convexAccuracy(sB, s.ptsB);

    // GjkDistance ctor (L132-142)
    GVert verts[4];
    for (uint i = 0u; i < 4u; ++i) { verts[i].p = float3(0, 0, 0); verts[i].s = 0.0f; verts[i].aI = 0u; verts[i].bI = 0u; }
    uint numVerts = 0u;
    uint iteration = 0u;
    float3 mClosest = float3(FLT_MAXQ, FLT_MAXQ, FLT_MAXQ);

    float3 mnA, mxA, mnB, mxB;
    convexBounds(sA, s.ptsA, mnA, mxA);
    convexBounds(sB, s.ptsB, mnB, mxB);
    float3 dir0 = (mnB + mxB) * 0.5f - (mnA + mxA) * 0.5f;
    if (normalizeQ(dir0) < FLT_EPS) dir0 = float3(1, 0, 0);

    {
        // ctor addPoint(dir0) (L141): numVerts == 0 so a duplicate is
        // impossible and the projection result is discarded
        ++iteration;
        const uint aI = supportIndexQ(cA, s.ptsA, dir0);
        const uint bI = supportIndexQ(cB, s.ptsB, -dir0);
        verts[0].p = cA.verts[aI] - cB.verts[bI];
        verts[0].s = 0.0f; verts[0].aI = aI; verts[0].bI = bI;
        numVerts = 1u;
    }

    const float accA = cA.accuracy, accB = cB.accuracy;
    const float distEps = 0.01f * min(accA, accB);
    const float maxDist = s.cfg.x;

    float retVal = 0.0f;
    uint tag = 0u;
    float3 pointA = float3(0, 0, 0), pointB = float3(0, 0, 0), axis = float3(0, 0, 0);
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
            const uint aI = supportIndexQ(cA, s.ptsA, dir);
            const uint bI = supportIndexQ(cB, s.ptsB, -dir);
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

    GjkOut r;
    r.result = float4(retVal, asfloat(tag), 0, 0);
    r.pointA = float4(pointA, 0);
    r.pointB = float4(pointB, 0);
    r.axis = float4(axis, 0);
    gOut[t] = r;
}
