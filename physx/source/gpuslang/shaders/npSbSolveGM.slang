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

// npSbSolveGM.slang
//
// Slang port of tetrahedronsSolveGM (softBodyGM.cu L926-1025): the per-tet
// material-model dispatcher composing the previously ported units.
//
//   if (youngs < SB_STIFFNESS_THRESHOLD /*1e-18f*/) return;  // outputs unchanged
//
//   NEO_HOOKEAN branch:
//     invE = 1/youngs;  R = Mat33(tetR);  P = R^T * (u1|u2|u3)
//     delta = tetrahedronsSolveInnerNeoHookean(P, ...)      (tranche 252 core;
//             multiplier slot .x accumulated on PGS, untouched on TGS)
//     delta0 = -(delta.c0 + delta.c1 + delta.c2) * v0.w
//     P.c_i += (delta.c_i * v{i+1}.w - delta0)
//     if (poissons > 1e-18f):
//       volumeLambda = (nu < 0.5f - 1e-18f) ? nu / ((1 - 2nu) * youngs) : 0
//       delta += volumeSolve(P, ..., volumeLambda)          (npSbVolumeSolve core)
//     deltaLambdaXgradC = R * delta                          (OVERWRITES output)
//
//   CO_ROTATIONAL branch:
//     lames: lambda = Y*nu/((1+nu)*(1-2nu)), mu = Y/(2*(1+nu))
//     det = Qinv.getDeterminant();  volume = 1/(6*det)
//     F = (u1|u2|u3) * Qinv;  alphaTilde0 = invDtSq / (2*mu*volume)
//     damping = elasticityDamping  (PhysX 5.12: no separate scaledDamping -- the 5.6.1
//       damping * toUniformReal(dampingScale) is gone with the material field)
//     lambdas = isTGS ? (0,0) : multiplier slot (float2)
//     ARAP_constraint(acc, lambdas.x, ..., alphaTilde0, damping)
//       (tranche 233 core: tensor = F - R, C = sqrt(sum||t||^2 + 1e-18),
//        dCdF = tensor*(1/C), then compute_deltaLambdaXgradC ACCUMULATES into
//        the caller-provided accumulator; degenerate |denom|<1e-15: no-damping
//        path returns with acc untouched, damping path ZEROES acc)
//     dx0 = -(acc.c0 + acc.c1 + acc.c2) * v0.w;  dx_i = acc.c_{i-1} * v_i.w
//     x0i = u_i + (dx_i - dx0);  F = (x01|x02|x03) * Qinv
//     if (poissons > 1e-18f):
//       ratio/alphaTilde1 = 0;  if (poissons < 0.5f - 1e-18f):
//         alphaTilde1 = invDtSq/(lambda*volume);  ratio = 2*mu/lambda
//       damping *= ratio
//       volume_constraint(acc, lambdas.y, ..., F, alphaTilde1, ...)
//         (C = det(F) - 1, dCdF = (F.c1 x F.c2 | F.c2 x F.c0 | F.c0 x F.c1))
//     if (!isTGS) store lambdas back
//
// The float2 multiplier slot is modeled as in/out mul.xy; deltaLambdaXgradC as
// in/out accumulator columns.  NOTE 0.5f - 1e-18f constant-folds to 0.5f in f32
// (transcribed verbatim; identical on both sides).  All float arithmetic
// NoContraction-stamped; +-0 compared with eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float pabs(float a) { return a < 0.0f ? -a : a; }
float3 mat33Transform(float3 c0, float3 c1, float3 c2, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}

// PxMat33(PxQuat) rotation-matrix constructor (columns).
void quatToMat(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
}

// compute_deltaLambdaXgradC (softBodyGM.cu L558-604), as in tranche 232/233:
// accumulates deltaLambda * dCdx into (a0,a1,a2); the no-damping degenerate
// denominator returns with the accumulator untouched, the damping one ZEROES it.
void computeDLXG(inout float3 a0, inout float3 a1, inout float3 a2, inout float lambda,
                 float C, float3 dFc0, float3 dFc1, float3 dFc2,
                 float3 q0, float3 q1, float3 q2,
                 float4 v0, float4 v1, float4 v2, float4 v3,
                 float alphaTilde, float dtInv, float damping)
{
    // row vectors of DmInv and dCdF (column-major: column_k[row])
    const float3 r0 = float3(q0.x, q1.x, q2.x);
    const float3 r1 = float3(q0.y, q1.y, q2.y);
    const float3 r2 = float3(q0.z, q1.z, q2.z);
    const float3 p0 = float3(dFc0.x, dFc1.x, dFc2.x);
    const float3 p1 = float3(dFc0.y, dFc1.y, dFc2.y);
    const float3 p2 = float3(dFc0.z, dFc1.z, dFc2.z);
    const float3 dCdx1 = float3(dot3(r0, p0), dot3(r0, p1), dot3(r0, p2));
    const float3 dCdx2 = float3(dot3(r1, p0), dot3(r1, p1), dot3(r1, p2));
    const float3 dCdx3 = float3(dot3(r2, p0), dot3(r2, p1), dot3(r2, p2));
    const float3 dCdx0 = -dCdx1 - dCdx2 - dCdx3;

    float deltaLambda;
    if (damping == 0.0f)
    {
        const float denom = magSq3(dCdx0) * v0.w + magSq3(dCdx1) * v1.w + magSq3(dCdx2) * v2.w + magSq3(dCdx3) * v3.w + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
            return;
        deltaLambda = (-C - alphaTilde * lambda) / denom;
    }
    else
    {
        const float denom = (1.0f + damping * dtInv) * (magSq3(dCdx0) * v0.w + magSq3(dCdx1) * v1.w + magSq3(dCdx2) * v2.w + magSq3(dCdx3) * v3.w) + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            a0 = float3(0.0f, 0.0f, 0.0f);
            a1 = float3(0.0f, 0.0f, 0.0f);
            a2 = float3(0.0f, 0.0f, 0.0f);
            return;
        }
        const float dampDot = dot3(dCdx0, v0.xyz) + dot3(dCdx1, v1.xyz) + dot3(dCdx2, v2.xyz) + dot3(dCdx3, v3.xyz);
        deltaLambda = (-C - alphaTilde * lambda - damping * dampDot) / denom;
    }

    lambda = lambda + deltaLambda;
    a0 = a0 + deltaLambda * dCdx1;
    a1 = a1 + deltaLambda * dCdx2;
    a2 = a2 + deltaLambda * dCdx3;
}

struct GmIn
{
    float4 gu1, gu2, gu3;    // u columns (.xyz)
    float4 v0, v1, v2, v3;   // .xyz velocities, .w inverse masses
    float4 qc0, qc1, qc2;    // Qinv columns
    float4 quat;             // tetR (x,y,z,w)
    float4 mat;              // youngs, poissons, elasticityDamping, unused (5.6.1 dampingScale)
    float4 prm;              // invDtSq, invDt, asfloat(isTGS), asfloat(isNeoHookean)
    float4 mul;              // multiplier slot float2 in (.xy)
    float4 acc0, acc1, acc2; // deltaLambdaXgradC accumulator columns (in)
}

struct GmOut
{
    float4 dc0, dc1, dc2;    // deltaLambdaXgradC columns (out)
    float4 mulOut;           // multiplier slot out (.xy)
}

struct GmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<GmParams> gParams;

RWStructuredBuffer<GmIn> gIn;    // binding 0
RWStructuredBuffer<GmOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbSolveGM(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const GmIn s = gIn[tid];

    const float3 u1 = s.gu1.xyz, u2 = s.gu2.xyz, u3 = s.gu3.xyz;
    const float3 Q0 = s.qc0.xyz, Q1 = s.qc1.xyz, Q2 = s.qc2.xyz;
    const float youngs = s.mat.x, poissons = s.mat.y;
    const float invDtSq = s.prm.x, invDt = s.prm.y;
    const bool isTGS = asuint(s.prm.z) == 1u;
    const bool isNeoHookean = asuint(s.prm.w) == 1u;

    float3 dl0 = s.acc0.xyz, dl1 = s.acc1.xyz, dl2 = s.acc2.xyz;
    float2 mul = s.mul.xy;

    GmOut o;

    if (youngs < 1.0e-18f)
    {
        o.dc0 = float4(dl0, 0.0f); o.dc1 = float4(dl1, 0.0f); o.dc2 = float4(dl2, 0.0f);
        o.mulOut = float4(mul, 0.0f, 0.0f);
        gOut[tid] = o;
        return;
    }

    if (isNeoHookean)
    {
        const float invE = 1.0f / youngs;
        float3 Rc0, Rc1, Rc2;
        quatToMat(s.quat, Rc0, Rc1, Rc2);
        // RTranspose columns = rows of R
        const float3 T0 = float3(Rc0.x, Rc1.x, Rc2.x);
        const float3 T1 = float3(Rc0.y, Rc1.y, Rc2.y);
        const float3 T2 = float3(Rc0.z, Rc1.z, Rc2.z);
        // P = RTranspose * (u1|u2|u3)
        float3 P0 = mat33Transform(T0, T1, T2, u1);
        float3 P1 = mat33Transform(T0, T1, T2, u2);
        float3 P2 = mat33Transform(T0, T1, T2, u3);

        // --- tetrahedronsSolveInnerNeoHookean (tranche 252 core) ---
        const float3 F0 = mat33Transform(P0, P1, P2, Q0);
        const float3 F1 = mat33Transform(P0, P1, P2, Q1);
        const float3 F2 = mat33Transform(P0, P1, P2, Q2);

        const float3 g1 = F0 * 2.0f * Q0.x + F1 * 2.0f * Q1.x + F2 * 2.0f * Q2.x;
        const float3 g2 = F0 * 2.0f * Q0.y + F1 * 2.0f * Q1.y + F2 * 2.0f * Q2.y;
        const float3 g3 = F0 * 2.0f * Q0.z + F1 * 2.0f * Q1.z + F2 * 2.0f * Q2.z;

        const float C = magSq3(F0) + magSq3(F1) + magSq3(F2) - 3.0f;
        const float3 g0 = -g1 - g2 - g3;
        const float w = magSq3(g0) * s.v0.w + magSq3(g1) * s.v1.w + magSq3(g2) * s.v2.w + magSq3(g3) * s.v3.w;
        const float alpha = invDtSq * invE;
        float dlambda;
        if (!isTGS)
        {
            dlambda = -(C + alpha * mul.x) / (w + alpha);
            mul.x += dlambda;
        }
        else
        {
            dlambda = -C / (w + alpha);
        }
        float3 d0 = g1 * dlambda;
        float3 d1 = g2 * dlambda;
        float3 d2 = g3 * dlambda;
        // --- end inner ---

        const float3 delta0 = -(d0 + d1 + d2) * s.v0.w;
        P0 = P0 + (d0 * s.v1.w - delta0);
        P1 = P1 + (d1 * s.v2.w - delta0);
        P2 = P2 + (d2 * s.v3.w - delta0);

        if (poissons > 1.0e-18f)
        {
            const float nu = poissons;
            const float volumeLambda = (nu < 0.5f - 1.0e-18f) ? nu / ((1.0f - 2.0f * nu) * youngs) : 0.0f;

            // --- volumeSolve (npSbVolumeSolve core) on the updated P ---
            const float det = dot3(Q0, cross3(Q1, Q2));
            const float volume = 1.0f / (6.0f * det);
            const float alphaTilde = (volumeLambda == 0.0f) ? 0.0f : 1.0f / (volumeLambda * volume * invDtSq);

            const float3 vF0 = mat33Transform(P0, P1, P2, Q0);
            const float3 vF1 = mat33Transform(P0, P1, P2, Q1);
            const float3 vF2 = mat33Transform(P0, P1, P2, Q2);

            const float3 vg0 = cross3(vF1, vF2);
            const float3 vg1 = cross3(vF2, vF0);
            const float3 vg2 = cross3(vF0, vF1);

            const float3 grad1 = vg0 * Q0.x + vg1 * Q1.x + vg2 * Q2.x;
            const float3 grad2 = vg0 * Q0.y + vg1 * Q1.y + vg2 * Q2.y;
            const float3 grad3 = vg0 * Q0.z + vg1 * Q1.z + vg2 * Q2.z;
            const float3 grad0 = -grad1 - grad2 - grad3;

            const float detF = dot3(vg2, vF2);
            const float Cv = detF - 1.0f;
            const float wv = magSq3(grad0) * s.v0.w + magSq3(grad1) * s.v1.w + magSq3(grad2) * s.v2.w + magSq3(grad3) * s.v3.w;
            const float dlv = -Cv / (wv + alphaTilde);

            d0 = d0 + grad1 * dlv;
            d1 = d1 + grad2 * dlv;
            d2 = d2 + grad3 * dlv;
            // --- end volumeSolve ---
        }

        // deltaLambdaXgradC = R * delta (overwrite)
        dl0 = mat33Transform(Rc0, Rc1, Rc2, d0);
        dl1 = mat33Transform(Rc0, Rc1, Rc2, d1);
        dl2 = mat33Transform(Rc0, Rc1, Rc2, d2);
    }
    else // CO_ROTATIONAL
    {
        // lameParameters (deformableUtils.cuh L124-130)
        const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
        const float mu = youngs / (2.0f * (1.0f + poissons));

        const float det = dot3(Q0, cross3(Q1, Q2));
        const float volume = 1.0f / (6.0f * det);

        // F = (u1|u2|u3) * Qinv
        float3 F0 = mat33Transform(u1, u2, u3, Q0);
        float3 F1 = mat33Transform(u1, u2, u3, Q1);
        float3 F2 = mat33Transform(u1, u2, u3, Q2);

        const float alphaTilde0 = invDtSq / (2.0f * mu * volume);

        float damping = s.mat.z;

        float2 lambdas = isTGS ? float2(0.0f, 0.0f) : mul;

        // ARAP_constraint (softBodyGM.cu L722-736)
        {
            float3 Rc0, Rc1, Rc2;
            quatToMat(s.quat, Rc0, Rc1, Rc2);
            const float3 t0 = F0 - Rc0, t1 = F1 - Rc1, t2 = F2 - Rc2;
            const float arapC = sqrt(dot3(t0, t0) + dot3(t1, t1) + dot3(t2, t2) + 1.0e-18f);
            const float inv = 1.0f / arapC;
            const float3 dF0 = t0 * inv, dF1 = t1 * inv, dF2 = t2 * inv;
            computeDLXG(dl0, dl1, dl2, lambdas.x, arapC, dF0, dF1, dF2,
                        Q0, Q1, Q2, s.v0, s.v1, s.v2, s.v3, alphaTilde0, invDt, damping);
        }

        const float3 dx0 = -(dl0 + dl1 + dl2) * s.v0.w;
        const float3 dx1 = dl0 * s.v1.w;
        const float3 dx2 = dl1 * s.v2.w;
        const float3 dx3 = dl2 * s.v3.w;

        const float3 x01 = u1 + (dx1 - dx0);
        const float3 x02 = u2 + (dx2 - dx0);
        const float3 x03 = u3 + (dx3 - dx0);

        F0 = mat33Transform(x01, x02, x03, Q0);
        F1 = mat33Transform(x01, x02, x03, Q1);
        F2 = mat33Transform(x01, x02, x03, Q2);

        if (poissons > 1.0e-18f)
        {
            float alphaTilde1 = 0.0f;
            float ratio = 0.0f;

            if (poissons < 0.5f - 1.0e-18f)
            {
                alphaTilde1 = invDtSq / (lamL * volume);
                ratio = 2.0f * mu / lamL;
            }

            damping *= ratio;

            // volume_constraint (softBodyGM.cu L756-767)
            const float detF = dot3(F0, cross3(F1, F2));
            const float Cv = detF - 1.0f;
            const float3 dF0 = cross3(F1, F2);
            const float3 dF1 = cross3(F2, F0);
            const float3 dF2 = cross3(F0, F1);
            computeDLXG(dl0, dl1, dl2, lambdas.y, Cv, dF0, dF1, dF2,
                        Q0, Q1, Q2, s.v0, s.v1, s.v2, s.v3, alphaTilde1, invDt, damping);
        }

        if (!isTGS)
            mul = lambdas;
    }

    o.dc0 = float4(dl0, 0.0f);
    o.dc1 = float4(dl1, 0.0f);
    o.dc2 = float4(dl2, 0.0f);
    o.mulOut = float4(mul, 0.0f, 0.0f);
    gOut[tid] = o;
}
