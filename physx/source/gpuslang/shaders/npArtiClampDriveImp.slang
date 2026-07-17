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

// npArtiClampDriveImp.slang
//
// Slang port of the articulation drive-impulse performance-envelope clamp from
// DyCpuGpuArticulation.h: computeBoundsForNegativeImpulse (L197-228),
// computeBoundsForPositiveImpulse (L231-266) and clampDriveImpulse (L279-331).
// Computes the impulse bounds imposed by the max-joint-velocity and
// max-impulse (with optional velocity-dependent resistance) envelopes and
// clamps driveImpulse into [lowerBound, upperBound] (or 0 if the bounds are
// crossed).  Pure scalar arithmetic with epsilon / sign branches.
//
// PxMin(a,b)=(a<b)?a:b; PxMax(a,b)=(a>b)?a:b; PxClamp(v,lo,hi)=PxMin(hi,PxMax(lo,v)).
// PX_MAX_F32 = 0x7f7fffff (FLT_MAX); PX_EPS_F32 = 0x34000000 (FLT_EPSILON=2^-23).
// All float arith NoContraction-stamped; +-0 eqFZ.

static const float PX_MAX_F32 = asfloat(0x7f7fffffu);
static const float PX_EPS_F32 = asfloat(0x34000000u);

float pmin(float a, float b) { return (a < b) ? a : b; }
float pmax(float a, float b) { return (a > b) ? a : b; }
float pclamp(float v, float lo, float hi) { return pmin(hi, pmax(lo, v)); }

// returns (bound1, bound2)
float2 boundsNeg(float velDeviation, float maxv, float denom1, float denom2, float impulse)
{
    if (abs(denom1) < PX_EPS_F32)
        return float2(-PX_MAX_F32, PX_MAX_F32);
    if (abs(denom2) < PX_EPS_F32)
    {
        const float bound1 = (velDeviation - maxv) / denom1;
        float bound2;
        if (abs(velDeviation + maxv) < PX_EPS_F32)
            bound2 = PX_MAX_F32;
        else
            bound2 = (velDeviation + maxv) > 0.0f ? PX_MAX_F32 : -PX_MAX_F32;
        return float2(bound1, pmin(0.0f, bound2));
    }
    const float bound1 = (velDeviation - maxv) / denom1;
    const float bound2 = (velDeviation + maxv) / denom2;
    if (denom2 >= 0.0f)
        return float2(pmax(bound1, impulse), pmin(0.0f, bound2));
    return float2(pmax(pmax(bound1, bound2), impulse), 0.0f);
}

float2 boundsPos(float velDeviation, float maxv, float denom1, float denom2, float impulse)
{
    if (abs(denom2) < PX_EPS_F32)
        return float2(-PX_MAX_F32, PX_MAX_F32);
    if (abs(denom1) < PX_EPS_F32)
    {
        const float bound2 = (velDeviation + maxv) / denom2;
        float bound1;
        if (abs(velDeviation - maxv) < PX_EPS_F32)
            bound1 = -PX_MAX_F32;
        else
            bound1 = (velDeviation - maxv) > 0.0f ? PX_MAX_F32 : -PX_MAX_F32;
        return float2(pmax(0.0f, bound1), bound2);
    }
    const float bound1 = (velDeviation - maxv) / denom1;
    const float bound2 = (velDeviation + maxv) / denom2;
    if (denom1 >= 0.0f)
        return float2(pmax(0.0f, bound1), pmin(bound2, impulse));
    return float2(0.0f, pmin(pmin(bound1, bound2), impulse));
}

float clampDriveImpulse(float jointVel0, float driveImpulse0, float driveImpulse, float response,
                        float maxJointVel, float maxImpulse, float speedImpulseGradient, float vdr)
{
    const float velDeviation = driveImpulse0 * response - jointVel0;
    float2 bounds1, bounds2;

    if (driveImpulse > 0.0f)
    {
        float denom1 = response - speedImpulseGradient;
        float denom2 = response + speedImpulseGradient;
        bounds1 = boundsPos(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
        if (vdr == 0.0f)
            bounds2 = float2(-maxImpulse, maxImpulse);
        else
        {
            denom1 = response - 1.0f / vdr;
            denom2 = response + 1.0f / vdr;
            bounds2 = boundsPos(velDeviation, maxImpulse / vdr, denom1, denom2, driveImpulse);
        }
    }
    else
    {
        float denom1 = response + speedImpulseGradient;
        float denom2 = response - speedImpulseGradient;
        bounds1 = boundsNeg(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
        if (vdr == 0.0f)
            bounds2 = float2(-maxImpulse, maxImpulse);
        else
        {
            denom1 = response + 1.0f / vdr;
            denom2 = response - 1.0f / vdr;
            bounds2 = boundsNeg(velDeviation, maxImpulse / vdr, denom1, denom2, driveImpulse);
        }
    }

    const float lowerBound = pmax(bounds1.x, bounds2.x);
    const float upperBound = pmin(bounds1.y, bounds2.y);
    if (lowerBound > upperBound)
        return 0.0f;
    return pclamp(driveImpulse, lowerBound, upperBound);
}

struct CdIn
{
    float4 a;     // jointVel0, driveImpulse0, driveImpulse, response
    float4 b;     // maxJointVel, maxImpulse, speedImpulseGradient, vdr
}

struct CdOut { float4 outs; }   // clampedImpulse, _, _, _

struct CdParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CdParams> gParams;

RWStructuredBuffer<CdIn> gIn;    // binding 0
RWStructuredBuffer<CdOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clampDriveImp(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CdIn s = gIn[tid];

    const float r = clampDriveImpulse(s.a.x, s.a.y, s.a.z, s.a.w, s.b.x, s.b.y, s.b.z, s.b.w);

    CdOut o;
    o.outs = float4(r, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
