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

// npAnisoOutput.slang
//
// Slang port of the anisotropy output-assembly stage that follows the
// eigen-decomposition in calculateAnisotropyLaunch / anisotropyKernel
// (anisotropy.cu L386-404).  The eigen-decomposition itself is NOT ported: it
// takes the CUDA fast-math path (jacobiRotateT with __fdividef / rsqrtf), which
// is not bit-exact portable.  This stage takes the already-decomposed rotation
// matrix R (columns = eigenvectors) and the covariance diagonal (eigenvalues)
// as inputs and produces the q1/q2/q3 anisotropy vectors:
//
//   cov_ii = max(cov_ii, 0)                      (sanitize)
//   lambda = ( sqrt(cov00), sqrt(cov11), sqrt(cov22) )
//   ks   = mAnisotropy
//   kmin = mAnisotropyMin * particleContactDistance
//   kmax = mAnisotropyMax * particleContactDistance
//   lambda *= ks                                 (PxVec3::operator*= : direct)
//   lambda  = Clamp(lambda, kmin, kmax)          (per comp PxMin(kmax, PxMax(kmin, l)))
//   q1 = (R.col0.xyz, lambda.x)
//   q2 = (R.col1.xyz, lambda.y)
//   q3 = (R.col2.xyz, lambda.z)
//
// PhysX PxMax(a,b) = (a<b)?b:a and PxMin(a,b) = (a<b)?a:b, so Clamp's per
// component form is PxMin(kmax, PxMax(kmin, l)); transcribed as ternaries (not
// the Slang min/max intrinsics) to preserve the exact NaN/+-0 branch behaviour.
// The sanitize max(cov,0) matches fmaxf for finite covariances via (cov>0?cov:0).
// Float mul NoContraction-stamped; +-0 compared with eqFZ.

// PhysX Clamp(a, s, t) per component: PxMin(t, PxMax(s, a)).  For float,
// PxMax(a,b) = selectMax(a,b) = a>b?a:b and PxMin(a,b) = selectMin(a,b) = a<b?a:b
// (PxUnixMathIntrinsics.h), so PxMax(s,a) = (s>a)?s:a and PxMin(t,mx) = (t<mx)?t:mx.
float clampComp(float a, float s, float t)
{
    const float mx = (s > a) ? s : a;   // PxMax(s, a) = selectMax(s, a)
    return (t < mx) ? t : mx;           // PxMin(t, mx) = selectMin(t, mx)
}

// max(cov, 0.f) -- fmaxf for finite covariance diagonal entries.
float sanitize(float v) { return (v > 0.0f) ? v : 0.0f; }

struct AoIn
{
    float4 covDiag;   // cov00, cov11, cov22, _
    float4 rc0;       // R.column0.xyz, _
    float4 rc1;       // R.column1.xyz, _
    float4 rc2;       // R.column2.xyz, _
    float4 params;    // mAnisotropy (ks), mAnisotropyMin, mAnisotropyMax, particleContactDistance
}

struct AoOut
{
    float4 q1;   // (R.col0.xyz, lambda.x)
    float4 q2;   // (R.col1.xyz, lambda.y)
    float4 q3;   // (R.col2.xyz, lambda.z)
}

struct AoParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AoParams> gParams;

RWStructuredBuffer<AoIn> gIn;    // binding 0
RWStructuredBuffer<AoOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void anisoOutput(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AoIn s = gIn[tid];

    // sanitize eigenvalues (covariance diagonal)
    const float c0 = sanitize(s.covDiag.x);
    const float c1 = sanitize(s.covDiag.y);
    const float c2 = sanitize(s.covDiag.z);

    float3 lambda = float3(sqrt(c0), sqrt(c1), sqrt(c2));

    const float ks = s.params.x;
    const float kmin = s.params.y * s.params.w;   // mAnisotropyMin * contactDistance
    const float kmax = s.params.z * s.params.w;   // mAnisotropyMax * contactDistance

    lambda = lambda * ks;                          // PxVec3::operator*= (direct)
    lambda = float3(clampComp(lambda.x, kmin, kmax),
                    clampComp(lambda.y, kmin, kmax),
                    clampComp(lambda.z, kmin, kmax));

    AoOut o;
    o.q1 = float4(s.rc0.xyz, lambda.x);
    o.q2 = float4(s.rc1.xyz, lambda.y);
    o.q3 = float4(s.rc2.xyz, lambda.z);
    gOut[tid] = o;
}
