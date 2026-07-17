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

// npClothVelClamp.slang
//
// Slang port of velocityClamping (FEMClothUtil.cuh L175-194): clamp a cloth
// vertex velocity to a maximum magnitude and reconcile the position along the
// clamped velocity, returning whether clamping fired.
//   maxVelSq = maxVel * maxVel
//   velMagSq = vel.xyz.magnitudeSquared()
//   if velMagSq > maxVelSq:
//       vel   *= maxVel / sqrt(velMagSq)     // scales all 4 comps (w too)
//       newPos = prevPos + vel * dt          // all 4 comps
//       newPos.w = pos.w                     // restore w
//       vel.w    = pos.w
//       delta = newPos - pos
//       pos   = newPos
//       accumDelta += delta
//       return true
//   return false
//
// The float4 operator*=(scalar) and +/-(float4) act componentwise incl. w;
// the intermediate scaled vel.w and prevPos.w + vel.w*dt are discarded (both
// overwritten by pos.w), so delta.w == 0.  All float arith NoContraction-
// stamped; sqrt is IEEE (bit-exact portable); +-0 eqFZ.

struct VcIn
{
    float4 pos;
    float4 vel;
    float4 accumDelta;
    float4 prevPos;
    float4 misc;   // maxVel, dt, _, _
}

struct VcOut
{
    float4 pos;
    float4 vel;
    float4 accumDelta;
    float4 flag;   // 1.0 if clamped else 0.0, in .x
}

struct VcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<VcParams> gParams;

RWStructuredBuffer<VcIn> gIn;    // binding 0
RWStructuredBuffer<VcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothVelClamp(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const VcIn s = gIn[tid];

    float4 pos = s.pos;
    float4 vel = s.vel;
    float4 accumDelta = s.accumDelta;
    const float4 prevPos = s.prevPos;
    const float maxVel = s.misc.x;
    const float dt = s.misc.y;

    const float maxVelSq = maxVel * maxVel;
    const float velMagSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;

    VcOut o;

    if (velMagSq > maxVelSq)
    {
        const float scale = maxVel / sqrt(velMagSq);
        vel = vel * scale;                    // scales all 4 components

        float4 newPos = prevPos + vel * dt;   // all 4 components
        newPos.w = pos.w;
        vel.w = pos.w;

        const float4 delta = newPos - pos;
        pos = newPos;
        accumDelta = accumDelta + delta;

        o.pos = pos;
        o.vel = vel;
        o.accumDelta = accumDelta;
        o.flag = float4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        o.pos = pos;
        o.vel = vel;
        o.accumDelta = accumDelta;
        o.flag = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    gOut[tid] = o;
}
