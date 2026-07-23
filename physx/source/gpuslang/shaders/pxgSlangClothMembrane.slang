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

// pxgSlangClothMembrane — P7.1: the reusable FEM-cloth in-plane (membrane / ARAP
// + area) energy leaf, factored out of the Phase-1 harnesses so the engine-ABI
// BDA cloth solve (femClothSolveBda.slang) shares ONE bit-exact copy of the math
// with the Phase-1 verification shaders.
//
// `membraneSolve` here is byte-for-byte the SAME function body already verified
// bit-exact in npClothMembrane.slang (clothMembrane, tranche 255) and reused in
// npClothSharedEnergy.slang (membraneSolve): membraneEnergySolvePerTriangle
// (FEMClothUtil.cuh L436-527) composing the 2x2 leaves extractRotation2D /
// arapConstraint2D (ARAPConstraint_F2X2) / areaConstraint2D (areaConstraint_F2X2,
// active #if 1 branch) + lameParameters (deformableUtils.cuh L124-130).  The
// per-triangle lambda slot collapses to a single in/out float2 (xx_i.w = inverse
// mass, read-only).  0.5f - 1e-14f constant-folds to 0.5f in f32 (transcribed
// verbatim, identical on both sides).
//
// The Phase-1 harnesses keep their own local copies UNCHANGED (this module is a
// verbatim lift, so no existing test moves a single ulp); importers must be
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS) exactly as those harnesses are,
// since the ARAP/area gradient sums fuse under -ffp-contract otherwise.
module pxgSlangClothMembrane;

public static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

public float dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
public float magSq2(float2 v) { return v.x * v.x + v.y * v.y; }
public float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
public float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
public float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// PxVec3::getNormalized: m > 0 ? v * PxRecipSqrt(m) : 0, PxRecipSqrt = 1/sqrtf.
public float3 getNormalized3(float3 v)
{
    const float m = magSq3(v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0.0f, 0.0f, 0.0f);
}

// extractRotation2D (FEMClothUtil.cuh L88-144), as in npClothExtractRot2D.
public void extractRotation2D(out float2 r0, out float2 r1, float2 f0, float2 f1)
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

// ARAPConstraint_F2X2 (L335-368), as in npClothMembrane / npClothSharedEnergy.
public void arapConstraint2D(inout float lambda, inout float2 dx0, inout float2 dx1, inout float2 dx2,
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

// areaConstraint_F2X2 (L376-419, active #if 1 branch), as in npClothMembrane.
public void areaConstraint2D(inout float lambda, inout float2 dx0, inout float2 dx1, inout float2 dx2,
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

// membraneEnergySolvePerTriangle (L436-527); slot models the per-triangle lambda
// float2.  VERBATIM from npClothSharedEnergy.slang's membraneSolve.
public void membraneSolve(inout float4 xx0, inout float4 xx1, inout float4 xx2,
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
