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

// P8.100b: the bodies moved VERBATIM into pxgSlangArticulationDynamic
// (adBoundsNeg/adBoundsPos/adClampDriveImpulse) so the internal-constraint
// dof solve composes them; this battery now drives the lifted code.
import pxgSlangArticulationDynamic;

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

    const float r = adClampDriveImpulse(s.a.x, s.a.y, s.a.z, s.a.w, s.b.x, s.b.y, s.b.z, s.b.w);

    CdOut o;
    o.outs = float4(r, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
