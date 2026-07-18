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

// npClothCollisionConstraint.slang
//
// Slang port of two FEM-cloth external-solve leaves
// (FEMClothExternalSolve.cu), selected by ctype:
//
// ctype 0 -- computeMassInertiaOffsets (L202-240):
//   average = (p0 + p1 + p2) * (1/3)
//   contact = p0*bc.x + p1*bc.y + p2*bc.z
//   m_i = 1 / PxMax(1e-18f, p_i.w)
//   offset = contact - average;  ra/rb/rc = p_i - average
//   thicknessSq = thickness * thickness * (m0 + m1 + m2)   // ((t*t)*sum)
//   invInertia.x = 1/(((ra.y^2+ra.z^2)*m0 + (rb.y^2+rb.z^2)*m1
//                      + (rc.y^2+rc.z^2)*m2 + thicknessSq))  (.y/.z cyclic)
//   invMass = 1/(m0 + m1 + m2)
//
// ctype 1 -- triangleCollisionConstraint (L243-286, vertex-triangle and
// edge-edge shared core):
//   CN = disp.dot(normal) - restDistance
//   denom = im.x*w.x*w.x + im.y*w.y*w.y + im.z*w.z*w.z + im.w*w.w*w.w
//   if (denom < FEMCLOTH_THRESHOLD /*1e-14f*/) return false  (dx untouched)
//   invDenom = 1/denom;  deltaLambdaN = -CN*invDenom
//   commonVec = deltaLambdaN*normal
//   tanDir = relDelta - (relDelta.normal)*normal;  tanMagSq
//   if (frictionCoefficient != 0 && tanMagSq > 1e-14f):
//     invTanMag = PxRecipSqrt(tanMagSq)   // 1.0f/sqrtf -- IEEE, portable
//     tanDir *= invTanMag;  CT = relDelta.dot(tanDir)
//     deltaLambdaT = -PxMin(PxMax(0, CT*invDenom), frictionCoefficient*PxAbs(deltaLambdaN))
//     commonVec += deltaLambdaT * tanDir            // NO tanScale here
//   dx_i = im.i * w.i * commonVec  (OVERWRITE);  return true
//
// The uint4 vertex-id indirection of computeMassInertiaOffsets collapses to
// direct p0/p1/p2 inputs (pure gather).  All float arithmetic NoContraction-
// stamped; sqrt IEEE; +-0 compared with eqFZ.

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float pabs(float a) { return a < 0.0f ? -a : a; }
float selMin(float a, float b) { return a < b ? a : b; }
float selMax(float a, float b) { return a > b ? a : b; }

struct CcIn
{
    float4 misc;        // restDistance, frictionCoefficient, thickness, asfloat(ctype)
    float4 a0, a1, a2;  // ctype 0: p0/p1/p2 (.w inv mass); ctype 1: disp/normal/relDelta
    float4 b0;          // ctype 0: barycentric.xyz; ctype 1: invMass4
    float4 b1;          // ctype 1: weights4
}

struct CcOut
{
    float4 o0, o1, o2, o3, o4, o5;  // ctype 0: contact, offset, ra, rb, rc, (invInertia, invMass)
                                    // ctype 1: dx0..dx3 (o4/o5 zero)
    float4 aux;                     // asfloat(returnFlag), _, _, _
}

struct CcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CcParams> gParams;

RWStructuredBuffer<CcIn> gIn;    // binding 0
RWStructuredBuffer<CcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothCollisionConstraint(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CcIn s = gIn[tid];

    const uint ctype = asuint(s.misc.w);

    CcOut o;
    o.o0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.o1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.o2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.o3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.o4 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.o5 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (ctype == 0u)
    {
        // computeMassInertiaOffsets
        const float3 p0 = s.a0.xyz, p1 = s.a1.xyz, p2 = s.a2.xyz;
        const float3 bc = s.b0.xyz;
        const float thickness = s.misc.z;

        const float3 average = (p0 + p1 + p2) * (1.0f / 3.0f);
        const float3 contact = p0 * bc.x + p1 * bc.y + p2 * bc.z;

        const float m0 = 1.0f / selMax(1e-18f, s.a0.w);
        const float m1 = 1.0f / selMax(1e-18f, s.a1.w);
        const float m2 = 1.0f / selMax(1e-18f, s.a2.w);

        const float3 offset = contact - average;
        const float3 ra = p0 - average;
        const float3 rb = p1 - average;
        const float3 rc = p2 - average;

        const float thicknessSq = thickness * thickness * (m0 + m1 + m2);

        float3 invInertia;
        invInertia.x = 1.0f / (((ra.y * ra.y + ra.z * ra.z) * m0 + (rb.y * rb.y + rb.z * rb.z) * m1 +
                                (rc.y * rc.y + rc.z * rc.z) * m2 + thicknessSq));
        invInertia.y = 1.0f / (((ra.x * ra.x + ra.z * ra.z) * m0 + (rb.x * rb.x + rb.z * rb.z) * m1 +
                                (rc.x * rc.x + rc.z * rc.z) * m2 + thicknessSq));
        invInertia.z = 1.0f / (((ra.x * ra.x + ra.y * ra.y) * m0 + (rb.x * rb.x + rb.y * rb.y) * m1 +
                                (rc.x * rc.x + rc.y * rc.y) * m2 + thicknessSq));

        const float invMass = 1.0f / (m0 + m1 + m2);

        o.o0 = float4(contact, 0.0f);
        o.o1 = float4(offset, 0.0f);
        o.o2 = float4(ra, 0.0f);
        o.o3 = float4(rb, 0.0f);
        o.o4 = float4(rc, 0.0f);
        o.o5 = float4(invInertia, invMass);
        o.aux = float4(asfloat(1u), 0.0f, 0.0f, 0.0f);
    }
    else
    {
        // triangleCollisionConstraint
        const float3 disp = s.a0.xyz;
        const float3 normal = s.a1.xyz;
        const float3 relDelta = s.a2.xyz;
        const float4 invMass = s.b0;
        const float4 weights = s.b1;
        const float restDistance = s.misc.x;
        const float frictionCoefficient = s.misc.y;

        const float CN = dot3(disp, normal) - restDistance;

        const float denom = invMass.x * weights.x * weights.x + invMass.y * weights.y * weights.y +
                            invMass.z * weights.z * weights.z + invMass.w * weights.w * weights.w;
        if (denom < FEMCLOTH_THRESHOLD)
        {
            o.aux = float4(asfloat(0u), 0.0f, 0.0f, 0.0f);
            gOut[tid] = o;
            return;
        }

        const float invDenom = 1.0f / denom;
        const float deltaLambdaN = -CN * invDenom;

        float3 commonVec = deltaLambdaN * normal;

        float3 tanDir = relDelta - dot3(relDelta, normal) * normal;
        const float tanMagSq = magSq3(tanDir);

        if (frictionCoefficient != 0.0f && tanMagSq > FEMCLOTH_THRESHOLD)
        {
            // PxRecipSqrt = 1.0f / sqrtf
            const float invTanMag = 1.0f / sqrt(tanMagSq);
            tanDir = tanDir * invTanMag;
            const float CT = dot3(relDelta, tanDir);

            float deltaLambdaT = selMax(0.0f, CT * invDenom);
            deltaLambdaT = -selMin(deltaLambdaT, frictionCoefficient * pabs(deltaLambdaN));

            commonVec += deltaLambdaT * tanDir;
        }

        o.o0 = float4(invMass.x * weights.x * commonVec, 0.0f);
        o.o1 = float4(invMass.y * weights.y * commonVec, 0.0f);
        o.o2 = float4(invMass.z * weights.z * commonVec, 0.0f);
        o.o3 = float4(invMass.w * weights.w * commonVec, 0.0f);
        o.aux = float4(asfloat(1u), 0.0f, 0.0f, 0.0f);
    }
    gOut[tid] = o;
}
