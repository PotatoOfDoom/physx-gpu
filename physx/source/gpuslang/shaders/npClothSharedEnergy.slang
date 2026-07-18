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

// npClothSharedEnergy.slang
//
// Slang port of clothSharedEnergySolvePerTrianglePair (FEMClothUtil.cuh
// L661-703): the FEM-cloth triangle-pair shell-energy driver composing the
// tranche-255 membrane dispatcher (twice, on the two adjacent triangles) and
// the tranche-256 bending dispatcher, IN ORDER, with positions threaded
// through in place (the membrane updates to x2/x3 feed the second membrane
// call and the bending term).
//
//   restEdge0 = restData0.xy; restEdge1 = restData0.zw; L = restSharedEdgeLength
//   det0 = L * restEdge0.y;  det1 = L * restEdge1.y
//   if (|det0| > FEMCLOTH_THRESHOLD /*1e-14f*/):
//     QInv0 = make_float4(restEdge0.y, 0, -restEdge0.x, L) / det0
//             -- cutil float4/scalar division is RECIPROCAL-MULTIPLY:
//                inv = 1.0f/det0; component * inv (NOT a direct divide)
//     membraneEnergySolvePerTriangle(x2, x3, x0, dt, material0, QInv0,
//                                    vrc.z, vrc.w, vrc.x, lambdaIndex 2i)
//   if (|det1| > FEMCLOTH_THRESHOLD): QInv1 analogous (restEdge1),
//     membrane(x2, x3, x1, material1, vrc.z, vrc.w, vrc.y, lambdaIndex 2i+1)
//   bendingEnergySolvePerTrianglePair(x0, x1, x2, x3, vrc, dt, i)
//
// The clothMaterials[u32(restData1.y/z)] lookups collapse to two inline
// material triples; the shared-partition lambda slots collapse to two in/out
// float2 (membranes) + one in/out scalar (bending).  The bending atan2f seam
// is modeled as in tranche 256: the atan2 result is an input (host-computed
// from the post-membrane positions via a CPU replica) and the shader outputs
// its own sinAngle/cosAngle for bitwise pinning.  aux.w packs path flags:
// bit0 = membrane0 ran (|det0|>thr), bit1 = membrane1 ran, bits 2+:
// bending path (0 solved, 1 x23 early-out, 2 kInv early-out).
// All float arithmetic NoContraction-stamped; sqrt IEEE; +-0 eqFZ.

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;
static const float FEMCLOTH_HALF_PI = 1.57079632679489661923f;
static const float FEMCLOTH_2PI = 6.28318530717958647692f;

float dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
float magSq2(float2 v) { return v.x * v.x + v.y * v.y; }
float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float pabs(float a) { return a < 0.0f ? -a : a; }
float selMin(float a, float b) { return a < b ? a : b; }
float selMax(float a, float b) { return a > b ? a : b; }

// PxVec3::getNormalized: m > 0 ? v * PxRecipSqrt(m) : 0, PxRecipSqrt = 1/sqrtf.
float3 getNormalized3(float3 v)
{
    const float m = magSq3(v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0.0f, 0.0f, 0.0f);
}

// extractRotation2D (FEMClothUtil.cuh L88-144), as in npClothExtractRot2D.
void extractRotation2D(out float2 r0, out float2 r1, float2 f0, float2 f1)
{
    const float3 S2 = float3(dot2(f0, f0), dot2(f1, f1), dot2(f0, f1));
    const float det = S2.x * S2.y - S2.z * S2.z;
    if (det < FEMCLOTH_THRESHOLD) { r0 = float2(1.0f, 0.0f); r1 = float2(0.0f, 1.0f); return; }

    const float s0 = sqrt(det);
    const float t = sqrt(S2.x + S2.y + 2.0f * s0);
    if (t < FEMCLOTH_THRESHOLD) { r0 = float2(1.0f, 0.0f); r1 = float2(0.0f, 1.0f); return; }
    const float tInv = 1.0f / t;

    float3 S = S2;
    S.x = S.x + s0;
    S.y = S.y + s0;
    S = S * tInv;

    const float sDet = S.x * S.y - S.z * S.z;
    if (sDet < FEMCLOTH_THRESHOLD) { r0 = float2(1.0f, 0.0f); r1 = float2(0.0f, 1.0f); return; }
    const float sDetInv = 1.0f / sDet;

    float3 SInv = float3(S.y, S.x, -S.z);
    SInv = SInv * sDetInv;

    r0 = SInv.x * f0 + SInv.z * f1;
    r1 = SInv.z * f0 + SInv.y * f1;
}

// ARAPConstraint_F2X2 (L335-368), as in npClothMembrane (tranche 255).
void arapConstraint2D(inout float lambda, inout float2 dx0, inout float2 dx1, inout float2 dx2,
                      float alphaTilde, float4 QInv, float2 x01, float2 x02,
                      float massInv0, float massInv1, float massInv2)
{
    const float2 f0 = QInv.x * x01 + QInv.y * x02;
    const float2 f1 = QInv.z * x01 + QInv.w * x02;

    float2 r0, r1;
    extractRotation2D(r0, r1, f0, f1);

    const float2 FMinusR0 = f0 - r0;
    const float2 FMinusR1 = f1 - r1;

    const float C = sqrt(dot2(FMinusR0, FMinusR0) + dot2(FMinusR1, FMinusR1));

    if (C > FEMCLOTH_THRESHOLD)
    {
        const float CInv = 1.0f / C;

        const float2 pC_pF0 = CInv * FMinusR0;
        const float2 pC_pF1 = CInv * FMinusR1;

        const float2 grad1 = QInv.x * pC_pF0 + QInv.z * pC_pF1;
        const float2 grad2 = QInv.y * pC_pF0 + QInv.w * pC_pF1;
        const float2 grad0 = -grad1 - grad2;

        const float denom = (massInv0 * magSq2(grad0) + massInv1 * magSq2(grad1) + massInv2 * magSq2(grad2)) + alphaTilde;
        float deltaLambda;
        if (denom < FEMCLOTH_THRESHOLD)
            deltaLambda = 0.0f;
        else
            deltaLambda = (-C - alphaTilde * lambda) / denom;

        lambda = lambda + deltaLambda;

        dx0 = dx0 + massInv0 * deltaLambda * grad0;
        dx1 = dx1 + massInv1 * deltaLambda * grad1;
        dx2 = dx2 + massInv2 * deltaLambda * grad2;
    }
}

// areaConstraint_F2X2 (L376-419, active branch), as in npClothMembrane.
void areaConstraint2D(inout float lambda, inout float2 dx0, inout float2 dx1, inout float2 dx2,
                      float alphaTilde, float2 x01, float2 x02,
                      float massInv0, float massInv1, float massInv2, float area)
{
    const float x01CrossX02 = x01.x * x02.y - x01.y * x02.x;
    const float undeformedAreaInv = 1.0f / area;

    const float C = 0.5f * x01CrossX02 * undeformedAreaInv - 1.0f;

    const float2 grad1 = float2(0.5f * undeformedAreaInv * x02.y, -0.5f * undeformedAreaInv * x02.x);
    const float2 grad2 = float2(-0.5f * undeformedAreaInv * x01.y, 0.5f * undeformedAreaInv * x01.x);
    const float2 grad0 = -grad1 - grad2;

    const float denom = (massInv0 * magSq2(grad0) + massInv1 * magSq2(grad1) + massInv2 * magSq2(grad2)) + alphaTilde;
    float deltaLambda;
    if (denom < FEMCLOTH_THRESHOLD)
        deltaLambda = 0.0f;
    else
        deltaLambda = (-C - alphaTilde * lambda) / denom;

    lambda = lambda + deltaLambda;

    dx0 = dx0 + massInv0 * deltaLambda * grad0;
    dx1 = dx1 + massInv1 * deltaLambda * grad1;
    dx2 = dx2 + massInv2 * deltaLambda * grad2;
}

// membraneEnergySolvePerTriangle (L436-527), as in npClothMembrane; slotIn/Out
// model the per-triangle lambda float2.
void membraneSolve(inout float4 xx0, inout float4 xx1, inout float4 xx2,
                   float dt, float youngs, float poissons, float thickness,
                   float4 QInv, float vertexScale0, float vertexScale1, float vertexScale2,
                   inout float2 lamSlot, bool isTGS)
{
    if (youngs < FEMCLOTH_THRESHOLD)
        return;

    float3 x0 = xx0.xyz;
    float3 x1 = xx1.xyz;
    float3 x2 = xx2.xyz;

    const float3 x01 = x1 - x0;
    const float3 x02 = x2 - x0;

    const float3 axis0 = getNormalized3(x01);
    const float3 normal = cross3(x01, x02);
    const float3 axis1 = getNormalized3(cross3(normal, axis0));

    const float dt2 = dt * dt;

    const float det = QInv.x * QInv.w - QInv.y * QInv.z;
    const float area = 1.0f / (2.0f * det);
    const float volume = area * thickness;

    float2 dx0 = float2(0.0f, 0.0f), dx1 = float2(0.0f, 0.0f), dx2 = float2(0.0f, 0.0f);
    float lambda0 = 0.0f, lambda1 = 0.0f;

    if (!isTGS)
    {
        lambda0 = lamSlot.x;
        lambda1 = lamSlot.y;
    }

    const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
    const float mu = youngs / (2.0f * (1.0f + poissons));

    float2 xp01 = float2(dot3(axis0, x01), dot3(axis1, x01));
    float2 xp02 = float2(dot3(axis0, x02), dot3(axis1, x02));

    const float alphaTilde0 = 1.0f / (2.0f * mu * volume * dt2);

    arapConstraint2D(lambda0, dx0, dx1, dx2, alphaTilde0, QInv, xp01, xp02,
                     vertexScale0 * xx0.w, vertexScale1 * xx1.w, vertexScale2 * xx2.w);

    if (poissons > FEMCLOTH_THRESHOLD)
    {
        float alphaTilde1 = 0.0f;

        if (poissons < 0.5f - FEMCLOTH_THRESHOLD)
        {
            alphaTilde1 = 1.0f / (lamL * volume * dt2);
        }

        xp01 += dx1 - dx0;
        xp02 += dx2 - dx0;

        areaConstraint2D(lambda1, dx0, dx1, dx2, alphaTilde1, xp01, xp02,
                         vertexScale0 * xx0.w, vertexScale1 * xx1.w, vertexScale2 * xx2.w, area);
    }

    x0 += dx0.x * axis0 + dx0.y * axis1;
    x1 += dx1.x * axis0 + dx1.y * axis1;
    x2 += dx2.x * axis0 + dx2.y * axis1;

    if (!isTGS)
    {
        lamSlot = float2(lambda0, lambda1);
    }

    xx0 = float4(x0, xx0.w);
    xx1 = float4(x1, xx1.w);
    xx2 = float4(x2, xx2.w);
}

struct GsIn
{
    float4 x0, x1, x2, x3;   // positions .xyz, inverse masses .w
    float4 vrc;              // vertexReferenceCount
    float4 rest;             // restEdge0.xy, restEdge1.xy
    float4 prm;              // restSharedEdgeLength, dt, asfloat(isTGS), _
    float4 mat0c;            // youngs0, poissons0, thickness0, _
    float4 mat1c;            // youngs1, poissons1, thickness1, _
    float4 bendc;            // restBendingAngle, kInv, angleIn (host atan2f), _
    float4 lamM;             // membrane0 lambda.xy, membrane1 lambda.xy (in)
    float4 lamB;             // bending lambda in, _, _, _
}

struct GsOut
{
    float4 x0, x1, x2, x3;   // updated positions .xyz (.w echoed)
    float4 lamMOut;          // membrane lambda slots out
    float4 aux;              // bendLambdaOut, sinAngle, cosAngle, asfloat(flags)
}

struct GsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<GsParams> gParams;

RWStructuredBuffer<GsIn> gIn;    // binding 0
RWStructuredBuffer<GsOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothSharedEnergy(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const GsIn s = gIn[tid];

    const float2 restEdge0 = s.rest.xy;
    const float2 restEdge1 = float2(s.rest.z, s.rest.w);
    const float restSharedEdgeLength = s.prm.x;
    const float dt = s.prm.y;
    const bool isTGS = asuint(s.prm.z) == 1u;

    float4 x0 = s.x0, x1 = s.x1, x2 = s.x2, x3 = s.x3;
    float2 lamM0 = s.lamM.xy;
    float2 lamM1 = float2(s.lamM.z, s.lamM.w);

    const float det0 = restSharedEdgeLength * restEdge0.y;
    const float det1 = restSharedEdgeLength * restEdge1.y;

    uint flags = 0u;

    // In-plane constraint for triangle0 with vertex x2, x3, and x0.
    if (pabs(det0) > FEMCLOTH_THRESHOLD)
    {
        flags |= 1u;
        // cutil float4/scalar operator/: reciprocal-multiply per component
        const float inv0 = 1.0f / det0;
        const float4 QInv0 = float4(restEdge0.y * inv0, 0.0f * inv0, -restEdge0.x * inv0, restSharedEdgeLength * inv0);
        membraneSolve(x2, x3, x0, dt, s.mat0c.x, s.mat0c.y, s.mat0c.z, QInv0,
                      s.vrc.z, s.vrc.w, s.vrc.x, lamM0, isTGS);
    }

    // In-plane constraint for triangle1 with vertex x2, x3, and x1.
    if (pabs(det1) > FEMCLOTH_THRESHOLD)
    {
        flags |= 2u;
        const float inv1 = 1.0f / det1;
        const float4 QInv1 = float4(restEdge1.y * inv1, 0.0f * inv1, -restEdge1.x * inv1, restSharedEdgeLength * inv1);
        membraneSolve(x2, x3, x1, dt, s.mat1c.x, s.mat1c.y, s.mat1c.z, QInv1,
                      s.vrc.z, s.vrc.w, s.vrc.y, lamM1, isTGS);
    }

    // Bending constraint (tranche 256 core) on the post-membrane positions.
    const float restBendingAngle = s.bendc.x;
    const float kInv = s.bendc.y;
    float bendLambdaOut = s.lamB.x;
    float sinAngle = 0.0f, cosAngle = 0.0f;
    uint bendPath = 0u;

    const float3 x02v = x2.xyz - x0.xyz;
    const float3 x03v = x3.xyz - x0.xyz;
    const float3 x13v = x3.xyz - x1.xyz;
    const float3 x12v = x2.xyz - x1.xyz;
    const float3 x23v = x3.xyz - x2.xyz;
    const float x23Len = sqrt(magSq3(x23v));

    if (x23Len < FEMCLOTH_THRESHOLD)
    {
        bendPath = 1u;
    }
    else if (kInv <= 0.0f)
    {
        bendPath = 2u;
    }
    else
    {
        const float x23LenInv = 1.0f / x23Len;
        const float3 x23Normalized = x23v * x23LenInv;

        const float3 scaledN0 = cross3(x02v, x03v);
        const float3 scaledN1 = cross3(x13v, x12v);

        const float n0LenInv = 1.0f / sqrt(magSq3(scaledN0));
        const float n1LenInv = 1.0f / sqrt(magSq3(scaledN1));

        const float3 n0 = scaledN0 * n0LenInv;
        const float3 n1 = scaledN1 * n1LenInv;

        cosAngle = dot3(n0, n1);
        sinAngle = dot3(cross3(n0, n1), x23Normalized);
        // angle = atan2f(sinAngle, cosAngle) -- the non-portable seam, taken
        // as input (host-computed from the post-membrane trig values).
        const float angle = s.bendc.z;

        const float dtInv = 1.0f / dt;
        const float alphaTilde = kInv * dtInv * dtInv;
        float C = angle - restBendingAngle;

        if (pabs(C + FEMCLOTH_2PI) < pabs(C))
        {
            C += FEMCLOTH_2PI;
        }
        else if (pabs(C - FEMCLOTH_2PI) < pabs(C))
        {
            C -= FEMCLOTH_2PI;
        }

        C = selMin(FEMCLOTH_HALF_PI, selMax(-FEMCLOTH_HALF_PI, C));

        const float3 temp0 = n0 * n0LenInv;
        const float3 temp1 = n1 * n1LenInv;
        const float3 dCdx0 = -x23Len * temp0;
        const float3 dCdx1 = -x23Len * temp1;
        const float3 dCdx2 = dot3(x03v, x23Normalized) * temp0 + dot3(x13v, x23Normalized) * temp1;
        const float3 dCdx3 = -(dot3(x02v, x23Normalized) * temp0 + dot3(x12v, x23Normalized) * temp1);

        float lambda = 0.0f;
        if (!isTGS)
        {
            lambda = s.lamB.x;
        }

        const float m0 = s.vrc.x * x0.w;
        const float m1 = s.vrc.y * x1.w;
        const float m2 = s.vrc.z * x2.w;
        const float m3 = s.vrc.w * x3.w;
        const float denom = (m0 * magSq3(dCdx0) + m1 * magSq3(dCdx1) + m2 * magSq3(dCdx2) + m3 * magSq3(dCdx3)) + alphaTilde;
        float deltaLambda;
        if (denom < FEMCLOTH_THRESHOLD)
            deltaLambda = 0.0f;
        else
            deltaLambda = (-C - alphaTilde * lambda) / denom;

        if (!isTGS)
        {
            bendLambdaOut = lambda + deltaLambda;
        }

        const float scale0 = s.vrc.x * x0.w * deltaLambda;
        const float scale1 = s.vrc.y * x1.w * deltaLambda;
        const float scale2 = s.vrc.z * x2.w * deltaLambda;
        const float scale3 = s.vrc.w * x3.w * deltaLambda;

        x0 = float4(x0.x + scale0 * dCdx0.x, x0.y + scale0 * dCdx0.y, x0.z + scale0 * dCdx0.z, x0.w);
        x1 = float4(x1.x + scale1 * dCdx1.x, x1.y + scale1 * dCdx1.y, x1.z + scale1 * dCdx1.z, x1.w);
        x2 = float4(x2.x + scale2 * dCdx2.x, x2.y + scale2 * dCdx2.y, x2.z + scale2 * dCdx2.z, x2.w);
        x3 = float4(x3.x + scale3 * dCdx3.x, x3.y + scale3 * dCdx3.y, x3.z + scale3 * dCdx3.z, x3.w);
    }

    GsOut o;
    o.x0 = x0; o.x1 = x1; o.x2 = x2; o.x3 = x3;
    o.lamMOut = float4(lamM0, lamM1);
    o.aux = float4(bendLambdaOut, sinAngle, cosAngle, asfloat(flags | (bendPath << 2u)));
    gOut[tid] = o;
}
