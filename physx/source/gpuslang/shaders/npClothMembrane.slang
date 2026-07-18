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

// npClothMembrane.slang
//
// Slang port of membraneEnergySolvePerTriangle (FEMClothUtil.cuh L436-527): the
// FEM-cloth fixed-corotated membrane energy dispatcher composing the previously
// ported 2x2 leaves (queryDeformationGradient_F2x2, extractRotation2D,
// queryConstraintGradient_F2x2, queryDeltaLambda 3-vertex no-damping,
// ARAPConstraint_F2X2 = tranche npClothArap2D, areaConstraint_F2X2 =
// npClothArea2D) and lameParameters (deformableUtils.cuh L124-130).
//
//   if (youngs < FEMCLOTH_THRESHOLD /*1e-14f*/) return;   // outputs unchanged
//   x01 = x1 - x0;  x02 = x2 - x0
//   axis0 = x01.getNormalized()            // m>0 ? v*(1/sqrt(m)) : 0
//   normal = x01.cross(x02);  axis1 = (normal.cross(axis0)).getNormalized()
//   dt2 = dt*dt;  det = QInv.x*QInv.w - QInv.y*QInv.z
//   area = 1/(2*det);  volume = area*thickness
//   lambda0/1 = 0;  if (!isTGS) load from the per-triangle lambda slot
//   lames: lambda = Y*nu/((1+nu)*(1-2nu)), mu = Y/(2*(1+nu))
//   xp01 = (axis0.dot(x01), axis1.dot(x01));  xp02 analogous
//   alphaTilde0 = 1/(2*mu*volume*dt2)        // (((2*mu)*volume)*dt2)
//   ARAPConstraint_F2X2(lambda0, dx0..2, alphaTilde0, QInv, xp01, xp02,
//                       vertexScale_i * xx_i.w ...)
//   if (poissons > 1e-14f):
//     alphaTilde1 = (poissons < 0.5f - 1e-14f) ? 1/(lambda*volume*dt2) : 0
//     xp01 += dx1 - dx0;  xp02 += dx2 - dx0
//     areaConstraint_F2X2(lambda1, dx0..2, alphaTilde1, ..., area)
//   x_i += dx_i.x*axis0 + dx_i.y*axis1
//   if (!isTGS) store lambda0/1 back
//
// The isShared/lambdaIndex slot selection collapses to a single in/out float2.
// xx_i.w (inverse mass) is read-only.  0.5f - 1e-14f constant-folds to 0.5f in
// f32 (transcribed verbatim, identical on both sides).  All float arithmetic
// NoContraction-stamped; sqrt IEEE; +-0 compared with eqFZ.

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

float dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
float magSq2(float2 v) { return v.x * v.x + v.y * v.y; }
float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

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

// ARAPConstraint_F2X2 (FEMClothUtil.cuh L335-368), as in npClothArap2D.
void arapConstraint2D(inout float lambda, inout float2 dx0, inout float2 dx1, inout float2 dx2,
                      float alphaTilde, float4 QInv, float2 x01, float2 x02,
                      float massInv0, float massInv1, float massInv2)
{
    // queryDeformationGradient_F2x2
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

        // queryConstraintGradient_F2x2
        const float2 grad1 = QInv.x * pC_pF0 + QInv.z * pC_pF1;
        const float2 grad2 = QInv.y * pC_pF0 + QInv.w * pC_pF1;
        const float2 grad0 = -grad1 - grad2;

        // queryDeltaLambda (3-vertex, no damping) with the OLD lambda
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

// areaConstraint_F2X2 (FEMClothUtil.cuh L376-419, active #if 1 branch), as in
// npClothArea2D.  No C>THRESHOLD guard here.
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

    // queryDeltaLambda (3-vertex, no damping) with the OLD lambda
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

struct CmIn
{
    float4 xx0, xx1, xx2;  // positions .xyz, inverse masses .w
    float4 qinv;           // QInv (x,y,z,w)
    float4 vscale;         // vertexScale0..2, dt
    float4 mat;            // youngs, poissons, thickness, asfloat(isTGS)
    float4 mul;            // lambda slot in (.xy)
}

struct CmOut
{
    float4 xx0, xx1, xx2;  // updated positions .xyz (.w echoed)
    float4 mulOut;         // lambda slot out (.xy)
}

struct CmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CmParams> gParams;

RWStructuredBuffer<CmIn> gIn;    // binding 0
RWStructuredBuffer<CmOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothMembrane(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CmIn s = gIn[tid];

    const float youngs = s.mat.x, poissons = s.mat.y, thickness = s.mat.z;
    const bool isTGS = asuint(s.mat.w) == 1u;
    const float vertexScale0 = s.vscale.x, vertexScale1 = s.vscale.y, vertexScale2 = s.vscale.z;
    const float dt = s.vscale.w;
    const float4 QInv = s.qinv;

    CmOut o;

    if (youngs < FEMCLOTH_THRESHOLD)
    {
        o.xx0 = s.xx0; o.xx1 = s.xx1; o.xx2 = s.xx2;
        o.mulOut = float4(s.mul.xy, 0.0f, 0.0f);
        gOut[tid] = o;
        return;
    }

    float3 x0 = s.xx0.xyz;
    float3 x1 = s.xx1.xyz;
    float3 x2 = s.xx2.xyz;

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
        lambda0 = s.mul.x;
        lambda1 = s.mul.y;
    }

    // lameParameters (deformableUtils.cuh L124-130)
    const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
    const float mu = youngs / (2.0f * (1.0f + poissons));

    // 1) ARAP constraint in the projected 2D frame
    float2 xp01 = float2(dot3(axis0, x01), dot3(axis1, x01));
    float2 xp02 = float2(dot3(axis0, x02), dot3(axis1, x02));

    const float alphaTilde0 = 1.0f / (2.0f * mu * volume * dt2);

    arapConstraint2D(lambda0, dx0, dx1, dx2, alphaTilde0, QInv, xp01, xp02,
                     vertexScale0 * s.xx0.w, vertexScale1 * s.xx1.w, vertexScale2 * s.xx2.w);

    // 2) area constraint
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
                         vertexScale0 * s.xx0.w, vertexScale1 * s.xx1.w, vertexScale2 * s.xx2.w, area);
    }

    x0 += dx0.x * axis0 + dx0.y * axis1;
    x1 += dx1.x * axis0 + dx1.y * axis1;
    x2 += dx2.x * axis0 + dx2.y * axis1;

    float2 mulOut = s.mul.xy;
    if (!isTGS)
    {
        mulOut = float2(lambda0, lambda1);
    }

    o.xx0 = float4(x0, s.xx0.w);
    o.xx1 = float4(x1, s.xx1.w);
    o.xx2 = float4(x2, s.xx2.w);
    o.mulOut = float4(mulOut, 0.0f, 0.0f);
    gOut[tid] = o;
}
