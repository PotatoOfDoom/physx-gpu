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

// pxgSlangClothBending — P7.1 tranche 2: the reusable FEM-cloth "Discrete Shells"
// XPBD dihedral BENDING leaf, factored out for the engine-ABI (BDA) triangle-pair
// solve (femClothSolvePairBda.slang) exactly as pxgSlangClothMembrane factored out
// the in-plane leaf for tranche 1.
//
// `bendingSolve` is a line-for-line port of bendingEnergySolvePerTrianglePair
// (FEMClothUtil.cuh L543-655): the 4-vertex (x0,x1,x2,x3) dihedral constraint over
// the two triangles sharing edge x2-x3.  restBendingAngle / kInv arrive as scalar
// params (the caller reads them from the descriptor's
// mOrdered{Shared,NonShared}RestBendingAngle_flexuralStiffness_damping[i]) and the
// per-pair bending lambda collapses to a single in/out scalar (x_i.w = inverse
// mass, read-only).
//
// ------------------------------------------------------------------------------
// THE atan2 SEAM — READ CAREFULLY.  The Phase-1 verification harness npClothBending
// (clothBending) DODGES the libm atan2f seam by taking `angle` as a host-computed
// INPUT and only OUTPUTTING its own sinAngle/cosAngle for a bitwise pin, because
// atan2f (like sinf/cosf and tex3D filtering) has NO cross-backend bit-exact
// guarantee.  The ENGINE kernel has NO host in the loop to precompute the angle, so
// it MUST evaluate `atan2(sinAngle, cosAngle)` ON DEVICE (Slang lowers this to
// GLSL.std.450 Atan2).  This module therefore is a DOCUMENTED-REPLACEMENT /
// TOLERANCE leaf (the same class as the tex3D precedent), NOT a bit-exact one:
//   - everything UP TO AND INCLUDING sinAngle / cosAngle is strict-IEEE bit-exact
//     (and is pinned bitwise by the still-passing npClothBending / npClothSharedEnergy
//     harnesses, which output them);
//   - the on-device atan2 and EVERYTHING DOWNSTREAM of it (C, the 2PI wrap, the
//     clamp, deltaLambda, and the position update) is a tight-TOLERANCE path — the
//     engine test (testFemClothSim.cpp) compares those within a documented relative
//     tolerance, NOT bitwise.  (The atan2(0, +x) == +0 IEEE special value keeps a
//     symmetric FLAT pair bit-exact even so; the engine test exploits that to pin
//     the whole downstream dCdx / deltaLambda / scale chain bit-exact.)
// Do NOT claim this leaf is bit-exact for a genuinely-bent pair.
// ------------------------------------------------------------------------------
//
// All float arithmetic must be NoContraction-stamped by the importer
// (PXG_NOCONTRACT_SHADERS) exactly as the Phase-1 harness is — the denom 4-sum and
// the gradient dot-products fuse to FMAs otherwise, breaking parity with the
// -ffp-contract=off CPU reference.  sqrt is IEEE; +-0 compared with eqFZ.
module pxgSlangClothBending;

// Kept module-internal (non-public) so importing both this module and
// pxgSlangClothMembrane (which exports its own public FEMCLOTH_THRESHOLD / dot3 /
// cross3 / magSq3) raises no symbol collision — only bendingSolve is public.
static const float BEND_THRESHOLD = 1.0e-14f;
static const float BEND_HALF_PI = 1.57079632679489661923f;
static const float BEND_2PI = 6.28318530717958647692f;

float bdot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 bcross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float bmagSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float bpabs(float a) { return a < 0.0f ? -a : a; }
float bselMin(float a, float b) { return a < b ? a : b; }
float bselMax(float a, float b) { return a > b ? a : b; }

// bendingEnergySolvePerTrianglePair (FEMClothUtil.cuh L543-655).  vrc = the four
// vertex reference counts; lambdaSlot is the per-pair bending lambda (in/out).
public void bendingSolve(inout float4 x0, inout float4 x1, inout float4 x2, inout float4 x3,
                         float4 vrc, float restBendingAngle, float kInv, float dt,
                         inout float lambdaSlot, bool isTGS)
{
    const float3 x02 = x2.xyz - x0.xyz;
    const float3 x03 = x3.xyz - x0.xyz;
    const float3 x13 = x3.xyz - x1.xyz;
    const float3 x12 = x2.xyz - x1.xyz;
    const float3 x23 = x3.xyz - x2.xyz;
    const float x23Len = sqrt(bmagSq3(x23));

    if (x23Len < BEND_THRESHOLD)
        return;

    // (restBendingAngle, kInv) supplied by the caller from the descriptor.
    if (kInv <= 0.0f)
        return;

    const float x23LenInv = 1.0f / x23Len;
    const float3 x23Normalized = x23 * x23LenInv;

    const float3 scaledN0 = bcross3(x02, x03);
    const float3 scaledN1 = bcross3(x13, x12);

    const float n0LenInv = 1.0f / sqrt(bmagSq3(scaledN0));
    const float n1LenInv = 1.0f / sqrt(bmagSq3(scaledN1));

    const float3 n0 = scaledN0 * n0LenInv;
    const float3 n1 = scaledN1 * n1LenInv;

    const float cosAngle = bdot3(n0, n1);
    const float sinAngle = bdot3(bcross3(n0, n1), x23Normalized);
    // THE SEAM: on-device atan2 (see the module header).  sinAngle / cosAngle above
    // are strict-IEEE bit-exact; this and everything below is a tolerance path.
    const float angle = atan2(sinAngle, cosAngle);

    const float dtInv = 1.0f / dt;
    const float alphaTilde = kInv * dtInv * dtInv;
    float C = angle - restBendingAngle;

    if (bpabs(C + BEND_2PI) < bpabs(C))
    {
        C += BEND_2PI;
    }
    else if (bpabs(C - BEND_2PI) < bpabs(C))
    {
        C -= BEND_2PI;
    }

    // PxClamp = PxMin(hi, PxMax(lo, v))
    C = bselMin(BEND_HALF_PI, bselMax(-BEND_HALF_PI, C));

    const float3 temp0 = n0 * n0LenInv;
    const float3 temp1 = n1 * n1LenInv;
    const float3 dCdx0 = -x23Len * temp0;
    const float3 dCdx1 = -x23Len * temp1;
    const float3 dCdx2 = bdot3(x03, x23Normalized) * temp0 + bdot3(x13, x23Normalized) * temp1;
    const float3 dCdx3 = -(bdot3(x02, x23Normalized) * temp0 + bdot3(x12, x23Normalized) * temp1);

    float lambda = 0.0f;
    if (!isTGS)
    {
        lambda = lambdaSlot;
    }

    // queryDeltaLambda (4-vertex, no damping, FEMClothUtil.cuh L249-262)
    const float m0 = vrc.x * x0.w;
    const float m1 = vrc.y * x1.w;
    const float m2 = vrc.z * x2.w;
    const float m3 = vrc.w * x3.w;
    const float denom = (m0 * bmagSq3(dCdx0) + m1 * bmagSq3(dCdx1) + m2 * bmagSq3(dCdx2) + m3 * bmagSq3(dCdx3)) + alphaTilde;
    float deltaLambda;
    if (denom < BEND_THRESHOLD)
        deltaLambda = 0.0f;
    else
        deltaLambda = (-C - alphaTilde * lambda) / denom;

    if (!isTGS)
    {
        lambdaSlot = lambda + deltaLambda;
    }

    const float scale0 = vrc.x * x0.w * deltaLambda;
    const float scale1 = vrc.y * x1.w * deltaLambda;
    const float scale2 = vrc.z * x2.w * deltaLambda;
    const float scale3 = vrc.w * x3.w * deltaLambda;

    x0 = float4(x0.x + scale0 * dCdx0.x, x0.y + scale0 * dCdx0.y, x0.z + scale0 * dCdx0.z, x0.w);
    x1 = float4(x1.x + scale1 * dCdx1.x, x1.y + scale1 * dCdx1.y, x1.z + scale1 * dCdx1.z, x1.w);
    x2 = float4(x2.x + scale2 * dCdx2.x, x2.y + scale2 * dCdx2.y, x2.z + scale2 * dCdx2.z, x2.w);
    x3 = float4(x3.x + scale3 * dCdx3.x, x3.y + scale3 * dCdx3.y, x3.z + scale3 * dCdx3.z, x3.w);
}
