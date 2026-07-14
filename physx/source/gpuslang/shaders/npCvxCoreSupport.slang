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

// npCvxCoreSupport.slang
//
// Slang port of the Gu::ConvexCore localSupport family (GuConvexSupport.h
// L95-380) and the Gu::ConvexShape support wrappers (L522-561):
//   - ePOINT: origin;
//   - eSEGMENT: (PxSign(dir.x) * length/2, 0, 0) -- RAW dir, no normalize;
//   - eBOX: per-component PxSign(dir) * extents/2 -- RAW dir;
//   - eELLIPSOID: diag(radii) transform-transpose, normalize, transform
//     (all three per-component multiplies);
//   - eCYLINDER: normalized dir; degenerate |d.y|,|d.z| < 1e-5 cap point
//     (sign(d.x)*h/2, r, 0), else cap centre + normalized radial * r;
//   - eCONE: sinA = r/sqrt(h^2+r^2); d.x > sinA -> apex (h/2,0,0);
//     degenerate radial -> (-h/2, r, 0); else base rim;
//   - ePOINTS: d = R.rotateInv(S.multiply(R.rotate(dir))); NULL/empty pool
//     or |d| < FLT_EPSILON -> origin; strict-> argmax of d.dot(p) in pool
//     order; the winner is pushed through the same S/R vertex2shape form.
//     The PointsCore image is the tranche-163 40-byte layout: pointer word
//     (byte 0, 0 = NULL), S at bytes 8-19, R at 20-35, numPoints/stride at
//     36/37; points are read at byte offset i*stride from the pool.
//   - ConvexShape::support(dir) = pose.transform(supportLocal(
//     pose.rotateInv(dir))) + dir * margin  (margin NOT applied to
//     supportLocal -- the GJK uses the margin-less form);
//   - ConvexShape::computeBounds: minimum = (support(-X).x, support(-Y).y,
//     support(-Z).z), maximum = (support(X).x, support(Y).y, support(Z).z).
//
// PxSign(a) = (a >= 0) ? 1 : -1 (PxUnixMathIntrinsics.h L69-72).
// getNormalized = m > 0 ? v * (1/sqrt(m)) : 0; magnitude = sqrt(x*x+y*y+z*z).

static const uint CORE_POINT = 0u;
static const uint CORE_SEGMENT = 1u;
static const uint CORE_BOX = 2u;
static const uint CORE_ELLIPSOID = 3u;
static const uint CORE_CYLINDER = 4u;
static const uint CORE_CONE = 5u;
static const uint CORE_POINTS = 6u;

static const float FLT_EPS = 1.1920928955078125e-7f;

struct SupIn
{
    uint4 meta;       // x = coreType
    uint4 core[3];    // tranche-163 coreData image (bit-cast words)
    float4 poseQ;
    float4 poseP;     // w = margin
    float4 dir;       // query direction (w pad)
    float4 pts[32];   // ePOINTS vertex pool
}

struct SupOut
{
    float4 supLocal;  // supportLocal(dir)
    float4 sup;       // support(dir)
    float4 bmin, bmax; // computeBounds
}

struct SupParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SupParams> gParams;

RWStructuredBuffer<SupIn> gIn;    // binding 0
RWStructuredBuffer<SupOut> gOut;  // binding 1

float pxSign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }
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

float coreF(SupIn s, uint word) { return asfloat(s.core[word / 4u][word & 3u]); }

float3 supportLocalQ(SupIn s, float3 dir)
{
    const uint type = s.meta.x;

    if (type == CORE_SEGMENT)
    {
        const float len = coreF(s, 0u);
        return float3(pxSign(dir.x) * len * 0.5f, 0, 0);
    }
    if (type == CORE_BOX)
    {
        const float3 ext = float3(coreF(s, 0u), coreF(s, 1u), coreF(s, 2u));
        return float3(pxSign(dir.x) * ext.x * 0.5f,
                      pxSign(dir.y) * ext.y * 0.5f,
                      pxSign(dir.z) * ext.z * 0.5f);
    }
    if (type == CORE_ELLIPSOID)
    {
        // The CUDA form is PxMat33::createDiagonal + transformTranspose/
        // transform, whose explicit zero terms make the result signed-zero
        // exact ((rx*dx + 0*dy) + 0*dz: 0*negative = -0, (-0)+(+0) = +0).
        // That form is UNENFORCEABLE here: slangc/llvmpipe fold the +-0 adds
        // away (no-signed-zeros), proven live by review-164's fixtures even
        // with the zero smuggled in as runtime data.  The port therefore
        // keeps the bare per-component multiply -- identical in every
        // magnitude bit, differing only in the sign of exact-zero outputs --
        // the CPU reference carries the true matrix form, and the ellipsoid
        // compare is zero-sign-insensitive (and ONLY that).
        const float3 radii = float3(coreF(s, 0u), coreF(s, 1u), coreF(s, 2u));
        const float3 dir1 = getNormalizedQ(float3(radii.x * dir.x, radii.y * dir.y, radii.z * dir.z));
        return float3(radii.x * dir1.x, radii.y * dir1.y, radii.z * dir1.z);
    }
    if (type == CORE_CYLINDER)
    {
        const float h = coreF(s, 0u) * 0.5f;
        const float r = coreF(s, 1u);
        const float3 d = getNormalizedQ(dir);
        const float eps = 1e-5f;
        if (abs(d.y) < eps && abs(d.z) < eps) return float3(pxSign(d.x) * h, r, 0);
        return float3(pxSign(d.x) * h, 0, 0) + getNormalizedQ(float3(0, d.y, d.z)) * r;
    }
    if (type == CORE_CONE)
    {
        const float h = coreF(s, 0u);
        const float r = coreF(s, 1u);
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
        if ((s.core[0].x == 0u && s.core[0].y == 0u) || numPoints == 0u) // 64-bit NULL (review 164)
            return float3(0, 0, 0);
        const float3 S = float3(asfloat(s.core[0].z), asfloat(s.core[0].w), asfloat(s.core[1].x));
        const float4 R = float4(asfloat(s.core[1].y), asfloat(s.core[1].z), asfloat(s.core[1].w), asfloat(s.core[2].x));

        const float3 rd = quatRotQ(R, dir);
        const float3 d = quatRotInvQ(R, float3(S.x * rd.x, S.y * rd.y, S.z * rd.z));
        const float mag = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        if (mag < FLT_EPS)
            return float3(0, 0, 0);

        uint index = 0u;
        float maxDot = -3.402823466e+38f;
        for (uint i = 0u; i < numPoints; ++i)
        {
            const float3 p = s.pts[(i * stride) / 16u].xyz;
            const float dot = d.x * p.x + d.y * p.y + d.z * p.z;
            if (dot > maxDot)
            {
                maxDot = dot;
                index = i;
            }
        }
        const float3 point = s.pts[(index * stride) / 16u].xyz;
        const float3 rp = quatRotQ(R, point);
        return quatRotInvQ(R, float3(S.x * rp.x, S.y * rp.y, S.z * rp.z));
    }
    // CORE_POINT (and anything else): the origin
    return float3(0, 0, 0);
}

float3 supportQ(SupIn s, float3 dir)
{
    const float3 sl = supportLocalQ(s, quatRotInvQ(s.poseQ, dir));
    return (quatRotQ(s.poseQ, sl) + s.poseP.xyz) + dir * s.poseP.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccsSupportAll(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const SupIn s = gIn[t];

    SupOut r;
    r.supLocal = float4(supportLocalQ(s, s.dir.xyz), 0);
    r.sup = float4(supportQ(s, s.dir.xyz), 0);

    // ConvexShape::computeBounds (GuConvexSupport.h L555-561)
    const float3 X = float3(1, 0, 0), Y = float3(0, 1, 0), Z = float3(0, 0, 1);
    r.bmin = float4(supportQ(s, -X).x, supportQ(s, -Y).y, supportQ(s, -Z).z, 0);
    r.bmax = float4(supportQ(s, X).x, supportQ(s, Y).y, supportQ(s, Z).z, 0);
    gOut[t] = r;
}
