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

// npSbSolveCorot.slang
//
// Slang port of tetrahedronsSolveInner (softBodyGM.cu L769-922): the
// co-rotational linear-elastic 6-Voigt XPBD tet solve.  Composes the ported
// leaves calculateDeformationGradient (softBody.cuh), the two computeLambda
// overloads and multiplyStrainByCompliance/evalStiffnessMatrix6x6 (tranches
// 231/235).  The per-tet 6-float multiplier slot (loadSpatialVector/
// storeSpatialVector on mSimTetraMultipliers) is modeled as in/out top/bottom
// float3 pairs.
//
//   F = P * Qinv;  det = Qinv.getDeterminant();  volume = 1/(6*det)
//   volumeSqrt = sqrt(volume)
//   G = (F + F^T)*0.5;  G.diag -= 1;  G *= volumeSqrt      (Cauchy strain)
//   alpha = invDtSq * (1/youngs)
//   damping = elasticityDamping (PhysX 5.12: the 5.6.1 scaledDamping = damping *
//   toUniformReal(dampingScale) is gone with PxsDeformableVolumeMaterialData::dampingScale)
//   uv_i = (v_i - v0).xyz
//   Vg = RTranspose * ((uv1|uv2|uv3) * Qinv)               (velocity gradient)
//   E = (Vg + Vg^T)*0.5;  E *= volumeSqrt                  (strain rate)
//   common = (1 + damping*invDt) * ((volumeSqrt*volumeSqrt) *
//            (sq(Qc0)*v1.w + sq(Qc1)*v2.w + sq(Qc2)*v3.w + sq(Qc0+Qc1+Qc2)*v0.w))
//   scale = 1/(common + alpha)
//   PGS (!isTGS): lambda_k = computeLambda<a,b,si> strain overload
//                 -(G(a,b) + invDtSq*multiplyStrainByCompliance(invE,nu,si,totalLambda)
//                   + damping*E(a,b)) * scale, (a,b,si) in
//                 {(0,0,0),(1,1,1),(2,2,2),(1,2,3),(0,2,4),(0,1,5)};
//                 totalLambda += lambda (written back)
//   TGS:          lambda_k = -(G(a,b) + damping*E(a,b))*scale
//   Voigt gradient assembly (outer(a,b) = (a*b.x, a*b.y, a*b.z)):
//     delta  = outer((volumeSqrt,0,0), Qc0) * lambda.top.x
//     delta += outer((0,volumeSqrt,0), Qc1) * lambda.top.y
//     delta += outer((0,0,volumeSqrt), Qc2) * lambda.top.z
//     hs = 0.5*volumeSqrt
//     delta += (outer((0,hs,0), Qc2) + outer((0,0,hs), Qc1)) * lambda.bottom.x
//     delta += (outer((hs,0,0), Qc2) + outer((0,0,hs), Qc0)) * lambda.bottom.y
//     delta += (outer((hs,0,0), Qc1) + outer((0,hs,0), Qc0)) * lambda.bottom.z
//
// PxMat33 G(a,b) = column_b[a].  operator*=(scalar) and matrix*scalar scale
// columns DIRECTLY (PxVec3::operator*=: x*=f -- not reciprocal).  getTranspose
// swaps; getDeterminant = c0.dot(c1.cross(c2)).  All float arithmetic
// NoContraction-stamped; +-0 compared with eqFZ.

float3 mat33Transform(float3 c0, float3 c1, float3 c2, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}
float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float magSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// evalStiffnessMatrix6x6 (softBodyGM.cu L507-519), as in tranche 231/235.
float evalStiffness(uint row, uint col, float invE, float nu)
{
    float result = 0.0f;
    if (row == col)
    {
        result = invE;
        if (row >= 3u)
            result = result * (1.0f + nu);
    }
    else if (row < 3u && col < 3u)
    {
        result = -invE * nu;
    }
    return result;
}

// multiplyStrainByCompliance (softBodyGM.cu L521-532).
float multiplyStrainByCompliance(float invE, float nu, uint si, float3 ltop, float3 lbot)
{
    return ltop.x * evalStiffness(si, 0u, invE, nu)
         + ltop.y * evalStiffness(si, 1u, invE, nu)
         + ltop.z * evalStiffness(si, 2u, invE, nu)
         + lbot.x * evalStiffness(si, 3u, invE, nu)
         + lbot.y * evalStiffness(si, 4u, invE, nu)
         + lbot.z * evalStiffness(si, 5u, invE, nu);
}

// PxMat33 operator()(row,col) = column_col[row].
float matElem(float3 c0, float3 c1, float3 c2, uint row, uint col)
{
    const float3 c = (col == 0u) ? c0 : (col == 1u) ? c1 : c2;
    return c[row];
}

struct CrIn
{
    float4 pc0, pc1, pc2;    // P columns
    float4 qc0, qc1, qc2;    // Qinv columns
    float4 rt0, rt1, rt2;    // RTranspose columns
    float4 v0, v1, v2, v3;   // velocities .xyz, inverse masses .w
    float4 mat;              // youngs, poissons, elasticityDamping, unused (5.6.1 dampingScale)
    float4 prm;              // invDtSq, invDt, asfloat(isTGS), _
    float4 tlTop;            // totalLambda.top.xyz (in)
    float4 tlBot;            // totalLambda.bottom.xyz (in)
}

struct CrOut
{
    float4 d0, d1, d2;       // delta columns
    float4 tlTopOut;         // totalLambda.top out (PGS accumulated; TGS = input)
    float4 tlBotOut;         // totalLambda.bottom out
}

struct CrParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CrParams> gParams;

RWStructuredBuffer<CrIn> gIn;    // binding 0
RWStructuredBuffer<CrOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbSolveCorot(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CrIn s = gIn[tid];

    const float3 P0 = s.pc0.xyz, P1 = s.pc1.xyz, P2 = s.pc2.xyz;
    const float3 Q0 = s.qc0.xyz, Q1 = s.qc1.xyz, Q2 = s.qc2.xyz;
    const float3 R0 = s.rt0.xyz, R1 = s.rt1.xyz, R2 = s.rt2.xyz;
    const float youngs = s.mat.x, poissons = s.mat.y, damping = s.mat.z;
    const float invDtSq = s.prm.x, invDt = s.prm.y;
    const bool isTGS = asuint(s.prm.z) == 1u;
    float3 tlTop = s.tlTop.xyz;
    float3 tlBot = s.tlBot.xyz;

    // F = P * Qinv
    const float3 F0 = mat33Transform(P0, P1, P2, Q0);
    const float3 F1 = mat33Transform(P0, P1, P2, Q1);
    const float3 F2 = mat33Transform(P0, P1, P2, Q2);

    const float det = dot3(Q0, cross3(Q1, Q2));   // Qinv.getDeterminant()
    const float volume = 1.0f / (6.0f * det);
    const float volumeSqrt = sqrt(volume);

    // G = (F + F^T)*0.5; diag -= 1; *= volumeSqrt
    // (F + F^T) columns: col_j = F.col_j + F^T.col_j where F^T.col_j = row j of F
    float3 G0 = (F0 + float3(F0.x, F1.x, F2.x)) * 0.5f;
    float3 G1 = (F1 + float3(F0.y, F1.y, F2.y)) * 0.5f;
    float3 G2 = (F2 + float3(F0.z, F1.z, F2.z)) * 0.5f;
    G0.x -= 1.0f;
    G1.y -= 1.0f;
    G2.z -= 1.0f;
    G0 = G0 * volumeSqrt; G1 = G1 * volumeSqrt; G2 = G2 * volumeSqrt;

    const float alpha = invDtSq * (1.0f / youngs);


    // uv_i = (v_i - v0).xyz
    const float3 uv1 = s.v1.xyz - s.v0.xyz;
    const float3 uv2 = s.v2.xyz - s.v0.xyz;
    const float3 uv3 = s.v3.xyz - s.v0.xyz;

    // Vg = RTranspose * ((uv1|uv2|uv3) * Qinv)
    const float3 M0 = mat33Transform(uv1, uv2, uv3, Q0);
    const float3 M1 = mat33Transform(uv1, uv2, uv3, Q1);
    const float3 M2 = mat33Transform(uv1, uv2, uv3, Q2);
    const float3 V0 = mat33Transform(R0, R1, R2, M0);
    const float3 V1 = mat33Transform(R0, R1, R2, M1);
    const float3 V2 = mat33Transform(R0, R1, R2, M2);

    // E = (Vg + Vg^T)*0.5; *= volumeSqrt
    float3 E0 = (V0 + float3(V0.x, V1.x, V2.x)) * 0.5f;
    float3 E1 = (V1 + float3(V0.y, V1.y, V2.y)) * 0.5f;
    float3 E2 = (V2 + float3(V0.z, V1.z, V2.z)) * 0.5f;
    E0 = E0 * volumeSqrt; E1 = E1 * volumeSqrt; E2 = E2 * volumeSqrt;

    const float sqLength1 = magSq(Q0);
    const float sqLength2 = magSq(Q1);
    const float sqLength3 = magSq(Q2);
    const float sqLengthSum = magSq(Q0 + Q1 + Q2);

    const float common = (1.0f + damping * invDt) * (volumeSqrt * volumeSqrt * (sqLength1 * s.v1.w + sqLength2 * s.v2.w + sqLength3 * s.v3.w + sqLengthSum * s.v0.w));
    const float scale = 1.0f / (common + alpha);

    float3 lamTop, lamBot;

    if (!isTGS)
    {
        const float invE = 1.0f / youngs;
        const float nu = poissons;

        lamTop.x = -(matElem(G0, G1, G2, 0u, 0u) + invDtSq * multiplyStrainByCompliance(invE, nu, 0u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 0u, 0u)) * scale;
        lamTop.y = -(matElem(G0, G1, G2, 1u, 1u) + invDtSq * multiplyStrainByCompliance(invE, nu, 1u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 1u, 1u)) * scale;
        lamTop.z = -(matElem(G0, G1, G2, 2u, 2u) + invDtSq * multiplyStrainByCompliance(invE, nu, 2u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 2u, 2u)) * scale;
        lamBot.x = -(matElem(G0, G1, G2, 1u, 2u) + invDtSq * multiplyStrainByCompliance(invE, nu, 3u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 1u, 2u)) * scale;
        lamBot.y = -(matElem(G0, G1, G2, 0u, 2u) + invDtSq * multiplyStrainByCompliance(invE, nu, 4u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 0u, 2u)) * scale;
        lamBot.z = -(matElem(G0, G1, G2, 0u, 1u) + invDtSq * multiplyStrainByCompliance(invE, nu, 5u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 0u, 1u)) * scale;
        tlTop += lamTop;
        tlBot += lamBot;
    }
    else
    {
        lamTop.x = -(matElem(G0, G1, G2, 0u, 0u) + damping * matElem(E0, E1, E2, 0u, 0u)) * scale;
        lamTop.y = -(matElem(G0, G1, G2, 1u, 1u) + damping * matElem(E0, E1, E2, 1u, 1u)) * scale;
        lamTop.z = -(matElem(G0, G1, G2, 2u, 2u) + damping * matElem(E0, E1, E2, 2u, 2u)) * scale;
        lamBot.x = -(matElem(G0, G1, G2, 1u, 2u) + damping * matElem(E0, E1, E2, 1u, 2u)) * scale;
        lamBot.y = -(matElem(G0, G1, G2, 0u, 2u) + damping * matElem(E0, E1, E2, 0u, 2u)) * scale;
        lamBot.z = -(matElem(G0, G1, G2, 0u, 1u) + damping * matElem(E0, E1, E2, 0u, 1u)) * scale;
    }

    // Voigt gradient assembly.  outer(a,b) = (a*b.x, a*b.y, a*b.z).
    const float halfSqrtVolume = (0.5f * volumeSqrt);

    // e00: grad = outer((volumeSqrt,0,0), Qc0); delta = grad * lambda.top.x
    const float3 a0 = float3(volumeSqrt, 0.0f, 0.0f);
    float3 D0 = (a0 * Q0.x) * lamTop.x;
    float3 D1 = (a0 * Q0.y) * lamTop.x;
    float3 D2 = (a0 * Q0.z) * lamTop.x;

    // e11
    const float3 a1 = float3(0.0f, volumeSqrt, 0.0f);
    D0 += (a1 * Q1.x) * lamTop.y;
    D1 += (a1 * Q1.y) * lamTop.y;
    D2 += (a1 * Q1.z) * lamTop.y;

    // e22
    const float3 a2 = float3(0.0f, 0.0f, volumeSqrt);
    D0 += (a2 * Q2.x) * lamTop.z;
    D1 += (a2 * Q2.y) * lamTop.z;
    D2 += (a2 * Q2.z) * lamTop.z;

    // e12: grad = outer((0,hs,0), Qc2) + outer((0,0,hs), Qc1)
    const float3 b1 = float3(0.0f, halfSqrtVolume, 0.0f);
    const float3 b2 = float3(0.0f, 0.0f, halfSqrtVolume);
    D0 += (b1 * Q2.x + b2 * Q1.x) * lamBot.x;
    D1 += (b1 * Q2.y + b2 * Q1.y) * lamBot.x;
    D2 += (b1 * Q2.z + b2 * Q1.z) * lamBot.x;

    // e02: grad = outer((hs,0,0), Qc2) + outer((0,0,hs), Qc0)
    const float3 b0 = float3(halfSqrtVolume, 0.0f, 0.0f);
    D0 += (b0 * Q2.x + b2 * Q0.x) * lamBot.y;
    D1 += (b0 * Q2.y + b2 * Q0.y) * lamBot.y;
    D2 += (b0 * Q2.z + b2 * Q0.z) * lamBot.y;

    // e01: grad = outer((hs,0,0), Qc1) + outer((0,hs,0), Qc0)
    D0 += (b0 * Q1.x + b1 * Q0.x) * lamBot.z;
    D1 += (b0 * Q1.y + b1 * Q0.y) * lamBot.z;
    D2 += (b0 * Q1.z + b1 * Q0.z) * lamBot.z;

    CrOut o;
    o.d0 = float4(D0, 0.0f);
    o.d1 = float4(D1, 0.0f);
    o.d2 = float4(D2, 0.0f);
    o.tlTopOut = float4(tlTop, 0.0f);
    o.tlBotOut = float4(tlBot, 0.0f);
    gOut[tid] = o;
}
