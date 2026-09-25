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

// npCvxCoreEpa.slang
//
// Slang port of Gu::RefGjkEpa::computeEpaDepth (GuRefGjkEpa.h L421-699):
// the EpaDepth expanding-polytope core over the same margin-less
// ConvexShape::supportLocal / Convex<Support> weld cache as tranche 169.
//   - ctor: bounds-center seed with the (1,0,0) fallback, then
//     PxComputeBasisVectors(dir0, dir1, dir2) (PxMathUtils.h L297-316:
//     the |dir.y| <= 0.9999 branch with right=(z,0,-x).normalize() and the
//     analytic up; else right=(1,0,0), up=(0,z,-y).normalize()); seeds
//     addPoint(dir0), addPoint(-dir0), addPoint(dir1), then -dir1/dir2/-dir2
//     while mNumVerts < 3; finally mProj = -FLT_MAX.
//   - addPoint (L480-565): MAX_VERTS-1 cap out; supports; proj = p.dot(-dir)
//     with the mProj/mBest update BEFORE the duplicate-(aI,bI) FLT_MAX out;
//     vert push; < 3 -> 0; == 3 -> the two seed faces makePlane(0,1,2) +
//     makePlane(1,0,2) (either failing -> FLT_MAX); >= 4 -> the BACKWARD
//     face-removal scan (plane.dot((p,1)) > testEps = 0.01*accuracy) with
//     the reversed-twin edge cancellation (backward scan, remove-by-swap,
//     break) and remove-by-swap face compaction, numEdges == 0 -> FLT_MAX,
//     the MAX_FACES - numEdges overflow check -> FLT_MAX, then the fan of
//     new faces (any degenerate makePlane -> FLT_MAX).
//   - makePlane (L455-472): abc = (b-a)x(c-a) + (c-b)x(a-b) + (a-c)x(b-c),
//     |abc| < FLT_EPSILON -> false, n = abc/|abc| (PxVec3 operator/ =
//     reciprocal multiply), d = n.dot(-a).
//   - computeClosest (L568-590): strict argmax of plane.w (first wins),
//     mBest = mClosest while mNumFaces == 2.
//   - computePoints (L609-624): barycentric tbc/tca/tab over the mBest face,
//     pointA/pointB accumulated separately from the A- and B-side caches.
//   - driver (L671-699): isValid (mNumFaces > 0) else FLT_MAX; loop
//     computeClosest -> addPoint(dir) -> proj >= closestDist - distEps ->
//     computePoints, axis = -getBestDir(), return getBestDist().
// All arithmetic is add/mul/div/sqrt: ALL outputs verify bitwise (modulo
// the documented llvmpipe +-0 NSZ exception class).

static const float FLT_EPS = 1.1920928955078125e-7f;
static const float FLT_MAXQ = 3.402823466e+38f;
static const uint MAX_VERTS = 32u;
static const uint MAX_FACES = 2u * MAX_VERTS - 4u;  // 60
static const uint MAX_EDGES = MAX_VERTS + MAX_FACES - 2u; // 90

static const uint CORE_POINT = 0u;
static const uint CORE_SEGMENT = 1u;
static const uint CORE_BOX = 2u;
static const uint CORE_ELLIPSOID = 3u;
static const uint CORE_CYLINDER = 4u;
static const uint CORE_CONE = 5u;
static const uint CORE_POINTS = 6u;

struct EpaIn
{
    uint4 metaA;      // x = coreType A
    uint4 coreA[3];
    float4 poseAq, poseAp;
    uint4 metaB;
    uint4 coreB[3];
    float4 poseBq, poseBp;
    float4 cfg;       // unused (layout parity with GjkIn)
    float4 ptsA[12];  // ePOINTS pools
    float4 ptsB[12];
}

struct EpaOut
{
    float4 result;    // x = return value (FLT_MAX / best dist), y = case tag (0 invalid, 1 converged)
    float4 pointA, pointB, axis;
}

struct EpaParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<EpaParams> gParams;

RWStructuredBuffer<EpaIn> gIn;    // binding 0
RWStructuredBuffer<EpaOut> gOut;  // binding 1

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
// P9.86: `PxVec3(a, 0, 0) + v` (GuConvexSupport.h cylinder/cone localSupport and
// contactFace rims).  Spelled as that vector add, llvmpipe folds the literal-zero lanes
// `0 + v.y` / `0 + v.z` to v.y / v.z even under SignedZeroInfNanPreserve (measured), which
// keeps a -0 the CUDA turns into +0.  The x lane is a real add; the zero lanes are the
// IEEE `0 + x` spelled as a select.
float3 cepAxisPlus(float a, float3 v)
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
        return cepAxisPlus(pxSign(d.x) * h, getNormalizedQ(float3(0, d.y, d.z)) * r);
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
        return cepAxisPlus(-halfH, getNormalizedQ(float3(0, d.y, d.z)) * r);
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

// Convex<Support>::getBounds (L65-78)
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

[shader("compute")]
[numthreads(32, 1, 1)]
void cepDepth(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const EpaIn s = gIn[t];

    ShapeRef sA, sB;
    sA.type = s.metaA.x; sA.core = s.coreA; sA.poseQ = s.poseAq; sA.poseP = s.poseAp.xyz;
    sB.type = s.metaB.x; sB.core = s.coreB; sB.poseQ = s.poseBq; sB.poseP = s.poseBp.xyz;

    ConvexW cA, cB;
    cA.shape = sA; cA.numVerts = 0u; cA.accuracy = convexAccuracy(sA, s.ptsA);
    cB.shape = sB; cB.numVerts = 0u; cB.accuracy = convexAccuracy(sB, s.ptsB);
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
    convexBounds(sA, s.ptsA, mnA, mxA);
    convexBounds(sB, s.ptsB, mnB, mxB);
    float3 dir0 = (mnB + mxB) * 0.5f - (mnA + mxA) * 0.5f;
    if (normalizeQ(dir0) < FLT_EPS) dir0 = float3(1, 0, 0);
    float3 dir1, dir2;
    computeBasisQ(dir0, dir1, dir2);

    bool fmIgnored;
    epaAddPoint(P, cA, cB, s.ptsA, s.ptsB, dir0, accuracy, fmIgnored);
    epaAddPoint(P, cA, cB, s.ptsA, s.ptsB, -dir0, accuracy, fmIgnored);
    epaAddPoint(P, cA, cB, s.ptsA, s.ptsB, dir1, accuracy, fmIgnored);
    if (P.numVerts < 3u) epaAddPoint(P, cA, cB, s.ptsA, s.ptsB, -dir1, accuracy, fmIgnored);
    if (P.numVerts < 3u) epaAddPoint(P, cA, cB, s.ptsA, s.ptsB, dir2, accuracy, fmIgnored);
    if (P.numVerts < 3u) epaAddPoint(P, cA, cB, s.ptsA, s.ptsB, -dir2, accuracy, fmIgnored);
    P.mProj = -FLT_MAXQ;

    float retVal = 0.0f;
    uint tag = 0u;
    float3 pointA = float3(0, 0, 0), pointB = float3(0, 0, 0), axis = float3(0, 0, 0);

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
            float proj = epaAddPoint(P, cA, cB, s.ptsA, s.ptsB, dir, accuracy, fm);
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

    EpaOut r;
    r.result = float4(retVal, asfloat(tag), 0, 0);
    r.pointA = float4(pointA, 0);
    r.pointB = float4(pointB, 0);
    r.axis = float4(axis, 0);
    gOut[t] = r;
}
