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

// npClothArap2D.slang
//
// Slang port of ARAPConstraint_F2X2 (FEMClothUtil.cuh L335-368): the XPBD
// As-Rigid-As-Possible membrane constraint C = ||F - R||_F for a 2x2
// deformation gradient F, with R its polar-decomposition rotation.  Composes
// three previously-ported leaves:
//   queryDeformationGradient_F2x2 (L299): f0 = QInv.x*x01 + QInv.y*x02
//                                         f1 = QInv.z*x01 + QInv.w*x02
//   extractRotation2D (L88): R = polar rotation of F (tranche 217)
//   queryConstraintGradient_F2x2 (L312): grad1 = qInv.x*pC_pF0 + qInv.z*pC_pF1
//                                        grad2 = qInv.y*pC_pF0 + qInv.w*pC_pF1
//   queryDeltaLambda 3-vertex no-damping (L211): denom = (Sum mi*|grad|^2)
//                                        + alphaTilde; return (-C-alphaTilde*lambda)/denom
// then:
//   FMinusR{0,1} = f{0,1} - r{0,1}
//   C = sqrt(FMinusR0.FMinusR0 + FMinusR1.FMinusR1)
//   if C > THRESHOLD:
//       pC_pF{0,1} = (1/C) * FMinusR{0,1}
//       grad0 = -grad1 - grad2
//       dl = queryDeltaLambda(...); lambda += dl
//       dx{i} += massInv{i} * dl * grad{i}
//
// The C <= THRESHOLD path leaves lambda/dx unchanged. queryDeltaLambda uses the
// OLD lambda (before +=). All float arith NoContraction-stamped; sqrt is IEEE
// (bit-exact portable); +-0 eqFZ.

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

float dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
float magSq2(float2 v) { return v.x * v.x + v.y * v.y; }

// extractRotation2D: 2x2 polar decomposition (tranche 217).  Returns R = [r0 r1].
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

struct ArIn
{
    float4 qinv;           // QInv
    float4 x01x02;         // x01.xy, x02.xy
    float4 dx0dx1;         // dx0.xy, dx1.xy   (in)
    float4 dx2LamAlpha;    // dx2.x, dx2.y, lambda, alphaTilde
    float4 mass;           // massInv0, massInv1, massInv2, _
}

struct ArOut
{
    float4 dx0dx1;         // updated dx0.xy, dx1.xy
    float4 dx2Lam;         // dx2.x, dx2.y, lambda, _
}

struct ArParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ArParams> gParams;

RWStructuredBuffer<ArIn> gIn;    // binding 0
RWStructuredBuffer<ArOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothArap2D(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const ArIn s = gIn[tid];

    const float4 QInv = s.qinv;
    const float2 x01 = float2(s.x01x02.x, s.x01x02.y);
    const float2 x02 = float2(s.x01x02.z, s.x01x02.w);
    float2 dx0 = float2(s.dx0dx1.x, s.dx0dx1.y);
    float2 dx1 = float2(s.dx0dx1.z, s.dx0dx1.w);
    float2 dx2 = float2(s.dx2LamAlpha.x, s.dx2LamAlpha.y);
    float lambda = s.dx2LamAlpha.z;
    const float alphaTilde = s.dx2LamAlpha.w;
    const float massInv0 = s.mass.x, massInv1 = s.mass.y, massInv2 = s.mass.z;

    // queryDeformationGradient_F2x2
    const float2 f0 = QInv.x * x01 + QInv.y * x02;
    const float2 f1 = QInv.z * x01 + QInv.w * x02;

    // R = polar rotation of F
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

    ArOut o;
    o.dx0dx1 = float4(dx0.x, dx0.y, dx1.x, dx1.y);
    o.dx2Lam = float4(dx2.x, dx2.y, lambda, 0.0f);
    gOut[tid] = o;
}
