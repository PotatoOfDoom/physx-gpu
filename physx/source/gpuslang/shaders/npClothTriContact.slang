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

// npClothTriContact.slang
//
// Slang port of the final trivial-leaf batch, selected by mode:
//
// mode 0 -- computeTriangleContact float2-barycentric variant
// (FEMClothExternalSolve.cu L155-180, both overloads share the math):
//   invMass = (v0.w, v1.w, v2.w, 0)
//   u = PxReal(1.0 - barycentric.x - barycentric.y)
//     -- 1.0 is a DOUBLE literal: the subtraction chain promotes to f64 and
//        the result rounds back to f32 (an f64 spot like tranche 251;
//        transcribed as float((double(1.0) - double(bx)) - double(by)))
//   vel = v0*u + v1*barycentric.x + v2*barycentric.y   -- ALL FOUR lanes
// mode 1 -- computeTriangleContact float4-barycentric variant (L182-194):
//   vel = v0*bc.x + v1*bc.y + v2*bc.z  (no u; all four lanes); invMass as above
// mode 2 -- bvhComputeTotalInvEdges (SDFConstruction.cu L245-251):
//   edges = (upper - lower) + PxVec3(0.0001f);  invEdges = (1/e.x, 1/e.y, 1/e.z)
// mode 3 -- pow3 (L441-443, u32 i*i*i) and rangesOverlaps (L446-448,
//   !(minA > maxB || minB > maxA)) packed into one row.
//
// The vels[triVertId] gathers collapse to direct v0/v1/v2 inputs.  All float
// (and the f64 spot's) arithmetic NoContraction-stamped; +-0 eqFZ.

struct TcIn
{
    float4 misc;   // asfloat(mode), bary.x, bary.y, bary.z
    float4 v0, v1, v2;   // mode 0/1 velocities (.w inv mass) | mode 2: v0=lower, v1=upper
    float4 rng;    // mode 3: minA, maxA, minB, maxB
    float4 iw;     // mode 3: asfloat(pow3 input u32), _, _, _
}

struct TcOut
{
    float4 vel;    // mode 0/1: contact velocity (all four lanes)
    float4 aux;    // mode 0/1: invMass | mode 2: invEdges.xyz | mode 3: asfloat(pow3), asfloat(overlap)
}

struct TcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<TcParams> gParams;

RWStructuredBuffer<TcIn> gIn;    // binding 0
RWStructuredBuffer<TcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothTriContact(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const TcIn s = gIn[tid];

    const uint mode = asuint(s.misc.x);

    TcOut o;
    o.vel = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.aux = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (mode == 0u)
    {
        // f64 spot: u = PxReal(1.0 - bx - by)
        const float u = float((double(1.0) - double(s.misc.y)) - double(s.misc.z));
        o.vel = s.v0 * u + s.v1 * s.misc.y + s.v2 * s.misc.z;
        o.aux = float4(s.v0.w, s.v1.w, s.v2.w, 0.0f);
    }
    else if (mode == 1u)
    {
        o.vel = s.v0 * s.misc.y + s.v1 * s.misc.z + s.v2 * s.misc.w;
        o.aux = float4(s.v0.w, s.v1.w, s.v2.w, 0.0f);
    }
    else if (mode == 2u)
    {
        const float3 edges = (s.v1.xyz - s.v0.xyz) + float3(0.0001f, 0.0001f, 0.0001f);
        o.aux = float4(1.0f / edges.x, 1.0f / edges.y, 1.0f / edges.z, 0.0f);
    }
    else
    {
        const uint i = asuint(s.iw.x);
        const uint p3 = i * i * i;
        const bool overlap = !(s.rng.x > s.rng.w || s.rng.z > s.rng.y);
        o.aux = float4(asfloat(p3), overlap ? asfloat(1u) : asfloat(0u), 0.0f, 0.0f);
    }
    gOut[tid] = o;
}
