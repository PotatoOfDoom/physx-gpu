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

// npCvxCoreSetup.slang
//
// Slang port of the convex-core narrowphase kernel setup glue
// (convexCoreCollision.cu): the TestInput constructor + checkTypes (L66-109),
// Gu::makeConvexShape (L115-171), and the per-kernel prep of
// convexCorePlaneNphase_Kernel (L475-480: type-ordering swap, PxPlane from
// transform0's x basis vector) and convexCoreConvexNphase_Kernel (L523-532:
// the mid-point shift subtracted from both poses).
//
// makeConvexShape target layout is Gu::ConvexShape (GuConvexSupport.h L522):
// coreType, coreData[40] (Gu MAX_CORE_SIZE = 10 floats), margin, pose.
//   - eCONVEXCORE (4): coreType = Enum(shape.hullOrMeshPtr) -- the pointer
//     field carries the enum; coreData bytes 0-23 = the 24-byte
//     PxConvexCoreGeometry::MAX_CORE_SIZE memcpy from &scale.scale.x (six
//     floats: scale.xyz then rotation.xyz); margin = scale.rotation.w;
//   - eSPHERE (0): ePOINT, margin = scale.scale.x, coreData untouched;
//   - eCAPSULE (2): eSEGMENT, SegmentCore.length = scale.scale.x * 2.0f,
//     margin = scale.scale.y;
//   - eBOX (3): eBOX, BoxCore.extents = scale.scale * 2.0f, margin = 0;
//   - eCONVEXMESH (5): ePOINTS, PointsCore { points = blob + 3 float4s
//     (modelled as byte offset 48), S = scale.scale, R = scale.rotation,
//     numPoints = (int4 at blob+16).x >> 8 & 0xff, stride = 16 }, margin = 0;
//   - anything else: returns false, coreType = Enum(-1).
// checkTypes swaps shape0/shape1 (and cacheRefs/transforms) when the caller's
// (type0, type1) request matches in reverse; contactDist was already summed
// (commutative).  restDist = restDistance ? restDistance[test] : 0.
//
// PxPlane(point, normal): n = normal, d = -point.dot(normal);
// getBasisVector0: x2 = x*2, w2 = w*2, (w*w2 - 1 + x*x2, z*w2 + y*x2,
// -y*w2 + z*x2)  (PxQuat.h L260-265, exact term order).

static const uint GEOM_SPHERE = 0u;
static const uint GEOM_PLANE = 1u;
static const uint GEOM_CAPSULE = 2u;
static const uint GEOM_BOX = 3u;
static const uint GEOM_CONVEXCORE = 4u;
static const uint GEOM_CONVEXMESH = 5u;

static const uint CORE_POINT = 0u;    // Gu::ConvexCore::Type
static const uint CORE_SEGMENT = 1u;
static const uint CORE_BOX = 2u;
static const uint CORE_POINTS = 6u;

struct CssIn
{
    float4 t0q, t0p, t1q, t1p;   // cached transforms (ref0, ref1)
    float4 s0ScaleS, s0ScaleR;   // shape0 PxMeshScale
    float4 s1ScaleS, s1ScaleR;   // shape1 PxMeshScale
    uint4 s0Info;                // x = PxGeometryType, y = hullOrMeshPtr low word, z = blob (int4 at +16).x, w pad
    uint4 s1Info;
    float4 dists;                // x = contactDistance[ref0], y = contactDistance[ref1], z = restDistance[test]
    uint4 misc;                  // x = cacheRef0, y = cacheRef1, z = restDistance present, w pad
}

struct CvxModel
{
    uint4 meta;       // x = coreType (0xFFFFFFFF for Enum(-1)), y = makeConvexShape return, z/w pad
    uint4 core[3];    // 40-byte coreData image (+ 8 pad bytes, zero)
    float4 poseQ;
    float4 poseP;     // w = margin
}

struct CssOut
{
    uint4 flags;      // x = swapShapes, y = cacheRef0, z = cacheRef1, w pad
    float4 dists;     // x = contactDist, y = restDist, z/w pad
    float4 aux;       // plane entry: (n.xyz, d); convex entry: (shift.xyz, 0)
    CvxModel c0;      // convex entry only (zeroed by the plane entry)
    CvxModel c1;
}

struct CssParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CssParams> gParams;

RWStructuredBuffer<CssIn> gIn;    // binding 0
RWStructuredBuffer<CssOut> gOut;  // binding 1

CvxModel makeConvexShapeQ(uint type, uint hullLow, uint blobHeadX,
                          float4 scaleS, float4 scaleR, float4 poseQ, float3 poseP)
{
    CvxModel m;
    m.meta = uint4(0xFFFFFFFFu, 0u, 0u, 0u); // coreType = Enum(-1), ret = false
    m.core[0] = uint4(0, 0, 0, 0);
    m.core[1] = uint4(0, 0, 0, 0);
    m.core[2] = uint4(0, 0, 0, 0);
    m.poseQ = float4(0, 0, 0, 0);
    m.poseP = float4(0, 0, 0, 0);
    float margin = 0.0f;

    if (type == GEOM_CONVEXCORE)
    {
        m.meta.x = hullLow; // Enum(shape.hullOrMeshPtr)
        // memcpy(coreData, &scale.scale.x, 24): scale.xyz then rotation.xyz
        m.core[0] = uint4(asuint(scaleS.x), asuint(scaleS.y), asuint(scaleS.z), asuint(scaleR.x));
        m.core[1] = uint4(asuint(scaleR.y), asuint(scaleR.z), 0u, 0u);
        margin = scaleR.w;
        m.meta.y = 1u;
    }
    else if (type == GEOM_SPHERE)
    {
        m.meta.x = CORE_POINT;
        margin = scaleS.x;
        m.meta.y = 1u;
    }
    else if (type == GEOM_CAPSULE)
    {
        m.meta.x = CORE_SEGMENT;
        m.core[0].x = asuint(scaleS.x * 2.0f); // SegmentCore.length
        margin = scaleS.y;
        m.meta.y = 1u;
    }
    else if (type == GEOM_BOX)
    {
        m.meta.x = CORE_BOX;
        m.core[0] = uint4(asuint(scaleS.x * 2.0f), asuint(scaleS.y * 2.0f), asuint(scaleS.z * 2.0f), 0u); // BoxCore.extents
        margin = 0.0f;
        m.meta.y = 1u;
    }
    else if (type == GEOM_CONVEXMESH)
    {
        m.meta.x = CORE_POINTS;
        // PointsCore: points ptr (bytes 0-7, modelled as blob byte offset 48),
        // S (8-19), R (20-35), numPoints/stride/pad (36-39)
        const uint numPoints = (blobHeadX >> 8u) & 0xFFu;
        m.core[0] = uint4(48u, 0u, asuint(scaleS.x), asuint(scaleS.y));
        m.core[1] = uint4(asuint(scaleS.z), asuint(scaleR.x), asuint(scaleR.y), asuint(scaleR.z));
        m.core[2] = uint4(asuint(scaleR.w), numPoints | (16u << 8u), 0u, 0u);
        margin = 0.0f;
        m.meta.y = 1u;
    }

    if (m.meta.y != 0u)
    {
        m.poseQ = poseQ;
        m.poseP = float4(poseP, margin);
    }
    else
    {
        m.poseP.w = margin;
    }
    return m;
}

float3 getBasisVector0Q(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccsPlanePrep(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CssIn s = gIn[t];

    // TestInput ctor
    uint cacheRef0 = s.misc.x, cacheRef1 = s.misc.y;
    float4 t0q = s.t0q, t1q = s.t1q;
    float3 t0p = s.t0p.xyz, t1p = s.t1p.xyz;
    uint4 i0 = s.s0Info, i1 = s.s1Info;
    float4 sc0S = s.s0ScaleS, sc0R = s.s0ScaleR, sc1S = s.s1ScaleS, sc1R = s.s1ScaleR;
    const float contactDist = s.dists.x + s.dists.y;
    const float restDist = (s.misc.z != 0u) ? s.dists.z : 0.0f;
    uint swapShapes = 0u;

    // checkTypes(ePLANE, eCONVEXCORE)
    if (GEOM_PLANE == i1.x && GEOM_CONVEXCORE == i0.x)
    {
        uint4 ti = i0; i0 = i1; i1 = ti;
        float4 ts = sc0S; sc0S = sc1S; sc1S = ts;
        float4 tr = sc0R; sc0R = sc1R; sc1R = tr;
        uint tc = cacheRef0; cacheRef0 = cacheRef1; cacheRef1 = tc;
        float4 tq = t0q; t0q = t1q; t1q = tq;
        float3 tp = t0p; t0p = t1p; t1p = tp;
        swapShapes = 1u;
    }

    // PxPlane(transform0.p, transform0.q.getBasisVector0())
    const float3 n = getBasisVector0Q(t0q);
    const float d = -(t0p.x * n.x + t0p.y * n.y + t0p.z * n.z);

    CssOut r;
    r.flags = uint4(swapShapes, cacheRef0, cacheRef1, 0u);
    r.dists = float4(contactDist, restDist, 0, 0);
    r.aux = float4(n, d);
    // plane entry models only convex1
    CvxModel zero;
    zero.meta = uint4(0, 0, 0, 0);
    zero.core[0] = uint4(0, 0, 0, 0); zero.core[1] = uint4(0, 0, 0, 0); zero.core[2] = uint4(0, 0, 0, 0);
    zero.poseQ = float4(0, 0, 0, 0); zero.poseP = float4(0, 0, 0, 0);
    r.c0 = zero;
    r.c1 = makeConvexShapeQ(i1.x, i1.y, i1.z, sc1S, sc1R, t1q, t1p);
    gOut[t] = r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccsConvexPrep(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CssIn s = gIn[t];

    const float contactDist = s.dists.x + s.dists.y;
    const float restDist = (s.misc.z != 0u) ? s.dists.z : 0.0f;

    // shift = (transform0.p + transform1.p) * 0.5f; poses relative to it
    const float3 shift = (s.t0p.xyz + s.t1p.xyz) * 0.5f;
    const float3 p0 = s.t0p.xyz - shift;
    const float3 p1 = s.t1p.xyz - shift;

    CssOut r;
    r.flags = uint4(0u, s.misc.x, s.misc.y, 0u);
    r.dists = float4(contactDist, restDist, 0, 0);
    r.aux = float4(shift, 0);
    r.c0 = makeConvexShapeQ(s.s0Info.x, s.s0Info.y, s.s0Info.z, s.s0ScaleS, s.s0ScaleR, s.t0q, p0);
    r.c1 = makeConvexShapeQ(s.s1Info.x, s.s1Info.y, s.s1Info.z, s.s1ScaleS, s.s1ScaleR, s.t1q, p1);
    gOut[t] = r;
}
