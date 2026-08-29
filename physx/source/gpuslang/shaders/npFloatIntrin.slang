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
//
// ===========================================================================
// npFloatIntrin — a PLATFORM measurement, not a kernel port.
//
// WHY THIS EXISTS.  About forty shaders in this corpus normalise a vector as
//
//     m > 0 ? v * rsqrt(m) : 0
//
// because that is the obvious spelling of `PxVec3::getNormalized()`
// (PxVec3.h:286).  It is not what the engine computes.  `getNormalized` calls
// `PxRecipSqrt` (PxMath.h:161), which on unix is `intrinsics::recipSqrt` =
// `1.0f / ::sqrtf(a)` (PxUnixMathIntrinsics.h:93) — a correctly-rounded square
// root followed by a correctly-rounded division.  Slang's `rsqrt` lowers to
// SPIR-V `GLSL.std.450 InverseSqrt`, which Vulkan permits to be off by up to
// 2 ULP.  The two are the same function only if this device happens to
// implement InverseSqrt as the exact division.
//
// Every bit-exact fixture that runs through one of those forty sites is
// therefore resting on an assumption about the DRIVER that nothing in this
// repository had ever measured.  This shader is the measurement: it computes
// both spellings on the device, side by side, over whatever inputs the fixture
// hands it, and the fixture compares them bitwise — and compares the division
// form against the host's `1.0f / sqrtf(x)`, which is what every CPU mirror in
// this port actually computes.
//
// It deliberately does NOT decide anything.  It reports.  What the port does
// with the answer is a decision recorded in RUNTIME_STATUS.md, not here.
// ===========================================================================
module npFloatIntrin;

struct FloatIntrinParams
{
    uint n;             // @0
    uint _pad4;         // @4
    uint64_t inputs;    // @8   float[n]
    uint64_t outRsqrt;  // @16  float[n]  -- rsqrt(x)
    uint64_t outDiv;    // @24  float[n]  -- 1.0f / sqrt(x)
    uint64_t outSqrt;   // @32  float[n]  -- sqrt(x)      -> 40
}

[[vk::push_constant]] ConstantBuffer<FloatIntrinParams> gFi;

[shader("compute")]
[numthreads(256, 1, 1)]
void floatIntrinRecipSqrtLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gFi.n)
        return;

    const float x = Ptr<float>(gFi.inputs)[i];

    // The two spellings, kept apart so no compiler pass can decide they are the
    // same expression: one goes through the intrinsic, one through a divide.
    Ptr<float>(gFi.outRsqrt)[i] = rsqrt(x);
    const float s = sqrt(x);
    Ptr<float>(gFi.outSqrt)[i] = s;
    Ptr<float>(gFi.outDiv)[i] = 1.0f / s;
}
