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

// npSbCollisionConstraint.slang
//
// Slang port of the two soft-body deformable collision constraints
// (softBody.cu): tetTriCollisionConstraint (L650-715, cloth-triangle vs
// softbody-tet, barycentric-weighted on both sides) and
// tetVertexCollisionConstraint (L717-771, vertex vs softbody-tet).
// ctype selects 0 = tetTri, 1 = tetVertex.  Shared core:
//
//   normal/pen = normalPen.xyz/.w;  deltas d_i loaded .xyz with weights .w
//   tetTri:    relDelta = (bcC.x*cd0 + bcC.y*cd1 + bcC.z*cd2)
//                       - (bcS.x*sd0 + bcS.y*sd1 + bcS.z*sd2 + bcS.w*sd3)
//   tetVertex: relDelta = d0 - (bc.x*d1 + bc.y*d2 + bc.z*d3 + bc.w*d4)
//   CN = pen + relDelta.dot(normal)
//   deltaLambdaN = PxMax(-CN*denomInv, -lambdaN);  lambdaN += deltaLambdaN
//   commonVec = deltaLambdaN * normal
//   tanDir = relDelta - relDelta.dot(normal)*normal;  tanMagSq
//   if (frictionCoefficient != 0 && tanMagSq > 1e-14f):
//     tanMag = PxSqrt(tanMagSq);  tanDir /= tanMag
//       -- PxVec3::operator/= is RECIPROCAL-MULTIPLY (f = 1/f; x *= f)
//     CT = relDelta.dot(tanDir)
//     deltaLambdaT = PxMax(0, CT*denomInv)
//     deltaLambdaT = -PxMin(deltaLambdaT, frictionCoefficient*PxAbs(deltaLambdaN))
//     commonVec += 0.5f * deltaLambdaT * tanDir      // (0.5*dLT)*tanDir
//   tetTri outputs OVERWRITE:  cDx_i =  cw_i*bcC.i*commonVec
//                              sDx_i = -sw_i*bcS.i*commonVec  ((-sw)*bc)*vec
//   tetVertex outputs ACCUMULATE: dx0 += w0*commonVec;
//                                 dx_i -= w_i*bc.{x,y,z,w}*commonVec (i=1..4)
//
// The lambdaN slot is modeled as an in/out scalar.  PxMax/PxMin are float
// selects (a>b?a:b / a<b?a:b); PxAbs = fabsf; PxSqrt IEEE.  All float
// arithmetic NoContraction-stamped; +-0 compared with eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float pabs(float a) { return a < 0.0f ? -a : a; }
float selMin(float a, float b) { return a < b ? a : b; }
float selMax(float a, float b) { return a > b ? a : b; }

struct ScIn
{
    float4 misc;               // lambdaN in, denomInv, frictionCoefficient, asfloat(ctype)
    float4 normalPen;          // normal.xyz, pen
    float4 bcA;                // clothBC / tetVertex bc
    float4 bcB;                // softBodyBC / unused
    float4 d0, d1, d2, d3, d4, d5, d6;  // deltas .xyz + weights .w
                               // tetTri: cd0..2 = d0..d2, sd0..3 = d3..d6
                               // tetVertex: d0..d4 (d5/d6 unused)
    float4 acc0, acc1, acc2, acc3, acc4;  // tetVertex dx0..dx4 accumulators (in)
}

struct ScOut
{
    float4 o0, o1, o2, o3, o4, o5, o6;  // tetTri: cDx0..2, sDx0..3
                                        // tetVertex: dx0..dx4 (o5/o6 zero)
    float4 aux;                         // lambdaN out, _, _, _
}

struct ScParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ScParams> gParams;

RWStructuredBuffer<ScIn> gIn;    // binding 0
RWStructuredBuffer<ScOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sbCollisionConstraint(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const ScIn s = gIn[tid];

    float lambdaN = s.misc.x;
    const float denomInv = s.misc.y;
    const float frictionCoefficient = s.misc.z;
    const uint ctype = asuint(s.misc.w);
    const float threshold = 1.0e-14f;

    const float3 normal = s.normalPen.xyz;
    const float pen = s.normalPen.w;

    // relDelta per constraint type
    float3 relDelta;
    if (ctype == 0u)
    {
        const float3 cd0 = s.d0.xyz, cd1 = s.d1.xyz, cd2 = s.d2.xyz;
        const float3 sd0 = s.d3.xyz, sd1 = s.d4.xyz, sd2 = s.d5.xyz, sd3 = s.d6.xyz;
        relDelta = (s.bcA.x * cd0 + s.bcA.y * cd1 + s.bcA.z * cd2) -
                   (s.bcB.x * sd0 + s.bcB.y * sd1 + s.bcB.z * sd2 + s.bcB.w * sd3);
    }
    else
    {
        const float3 d0 = s.d0.xyz, d1 = s.d1.xyz, d2 = s.d2.xyz, d3 = s.d3.xyz, d4 = s.d4.xyz;
        relDelta = d0 - (s.bcA.x * d1 + s.bcA.y * d2 + s.bcA.z * d3 + s.bcA.w * d4);
    }

    const float CN = pen + dot3(relDelta, normal);

    const float deltaLambdaN = selMax(-CN * denomInv, -lambdaN);
    lambdaN += deltaLambdaN;

    float3 commonVec = deltaLambdaN * normal;

    float3 tanDir = relDelta - dot3(relDelta, normal) * normal;
    const float tanMagSq = magSq3(tanDir);

    if (frictionCoefficient != 0.0f && tanMagSq > threshold)
    {
        const float tanMag = sqrt(tanMagSq);
        // PxVec3::operator/= : reciprocal-multiply
        const float tanMagInv = 1.0f / tanMag;
        tanDir = float3(tanDir.x * tanMagInv, tanDir.y * tanMagInv, tanDir.z * tanMagInv);
        const float CT = dot3(relDelta, tanDir);

        float deltaLambdaT = selMax(0.0f, CT * denomInv);
        deltaLambdaT = -selMin(deltaLambdaT, frictionCoefficient * pabs(deltaLambdaN));

        const float tanScale = 0.5f;
        commonVec += tanScale * deltaLambdaT * tanDir;
    }

    ScOut o;
    if (ctype == 0u)
    {
        const float cw0 = s.d0.w, cw1 = s.d1.w, cw2 = s.d2.w;
        const float sw0 = s.d3.w, sw1 = s.d4.w, sw2 = s.d5.w, sw3 = s.d6.w;
        o.o0 = float4(cw0 * s.bcA.x * commonVec, 0.0f);
        o.o1 = float4(cw1 * s.bcA.y * commonVec, 0.0f);
        o.o2 = float4(cw2 * s.bcA.z * commonVec, 0.0f);
        o.o3 = float4(-sw0 * s.bcB.x * commonVec, 0.0f);
        o.o4 = float4(-sw1 * s.bcB.y * commonVec, 0.0f);
        o.o5 = float4(-sw2 * s.bcB.z * commonVec, 0.0f);
        o.o6 = float4(-sw3 * s.bcB.w * commonVec, 0.0f);
    }
    else
    {
        const float w0 = s.d0.w, w1 = s.d1.w, w2 = s.d2.w, w3 = s.d3.w, w4 = s.d4.w;
        o.o0 = float4(s.acc0.xyz + w0 * commonVec, 0.0f);
        o.o1 = float4(s.acc1.xyz - w1 * s.bcA.x * commonVec, 0.0f);
        o.o2 = float4(s.acc2.xyz - w2 * s.bcA.y * commonVec, 0.0f);
        o.o3 = float4(s.acc3.xyz - w3 * s.bcA.z * commonVec, 0.0f);
        o.o4 = float4(s.acc4.xyz - w4 * s.bcA.w * commonVec, 0.0f);
        o.o5 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        o.o6 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    o.aux = float4(lambdaN, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
