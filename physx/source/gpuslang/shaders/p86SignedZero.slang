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

// p86SignedZero.slang -- P9.86: the sign of zero on the device.
//
// P9.83b-8 (1/n) measured that llvmpipe drops the sign of zero in arithmetic with a
// CONSTANT zero operand (0 - x, x + 0, 0 * x) even under NoContraction, while
// runtime-operand arithmetic stays IEEE.  This probe pins both halves, and is built
// TWICE: p86SignedZero.spv as-is (stamped NoContraction like the engine), and
// p86SignedZeroSz.spv with OpExecutionMode SignedZeroInfNanPreserve 32 added by
// tools/spv_szinp.py -- which is how SPIR-V demands the IEEE signed-zero guarantee.
//
// The zeros that matter arrive at run time (in[0] = +0, in[1] = -0), and the
// constant zeros are literals in the source, so slangc emits real OpFAdd/OpFSub/
// OpFMul with a constant operand; any folding is the driver's.

import pxgSlangTriBox;

struct SzPush { uint64_t inp; uint64_t outp; }
[[vk::push_constant]] ConstantBuffer<SzPush> gSz;

[shader("compute")]
[numthreads(1, 1, 1)]
void szProbe(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<float> i = Ptr<float>(gSz.inp);
    Ptr<uint> o = Ptr<uint>(gSz.outp);
    const float pz = i[0], nz = i[1], m1 = i[2], rz = i[3];   // +0, -0, -1, +0 (runtime)
    const float3 v = float3(pz, pz, pz);

    // constant-zero operand                  IEEE
    o[0] = asuint(0.0f - pz);              // +0
    o[1] = asuint(nz + 0.0f);              // +0
    o[2] = asuint(0.0f + nz);              // +0
    o[3] = asuint(0.0f * nz);              // -0
    o[4] = asuint((float3(0.0f, 0.0f, 0.0f) - v).y);   // +0
    o[5] = asuint(nz - 0.0f);              // -0
    // runtime operands only
    o[6] = asuint(rz - pz);                // +0
    o[7] = asuint(nz + rz);                // +0
    o[8] = asuint(rz + nz);                // +0
    o[9] = asuint(rz * nz);                // -0
    o[10] = asuint(m1 * pz);               // -0
    // min / max on a signed-zero pair (PxMin(a,b) = a < b ? a : b keeps b on a tie)
    o[11] = asuint(min(pz, nz));
    o[12] = asuint(max(nz, pz));
    o[13] = asuint((pz < nz) ? pz : nz);   // PxMin spelling: -0
    o[14] = asuint((pz > nz) ? pz : nz);   // PxMax spelling: -0
    // a PROVEN zero (P9.43b: LLVM sees abs(x) * 0 is zero and folds the add)
    const float pzr = abs(m1) * 0.0f;
    o[15] = asuint(nz + pzr);              // +0
    o[16] = asuint(pzr - pz);              // +0
    // vector times a constant zero vector, runtime negative scale (arti impulse)
    o[17] = asuint((float3(0.0f, 0.0f, 0.0f) * m1).x);   // -0
    // `e = 0; e -= a` (npD6JointPrep errorVector) with a = +0
    float3 e = float3(0.0f, 0.0f, 0.0f);
    if (asuint(m1) != 0u) e -= v;
    o[18] = asuint(e.x);                   // +0
    // P9.86 (2): a literal zero LANE of a vector add, and a literal zero SEED forwarded
    // into the first add of an unrolled constant-trip loop -- llvmpipe folds both, stamp
    // or not.  The ellipsoid's `x + 0*y` (a COMPUTED zero) and the select spelling hold.
    o[19] = asuint((float3(m1, 0.0f, 0.0f) + float3(0.0f, nz, nz)).y);   // +0
    float acc = 0.0f;
    for (uint k = 0u; k < 2u; ++k)
        acc += (k == 0u) ? nz : i[1];                                      // both -0
    o[20] = asuint(acc);                                                   // +0
    o[21] = asuint(nz + 0.0f * pz);                                        // +0
    o[22] = asuint((nz == 0.0f) ? 0.0f : nz);                              // +0
    // P9.86f (8): a literal zero reaching the add only through a SELECT ARM on a runtime
    // condition (constraintPrepPGS `cTargetVel + (isGreater2 ? b : 0.0f)`, softBodySimBda
    // `oldPos + (isTGS ? 0 : d)`, psContactSolveTgsBda `delta + (first ? 0 : dP)`); m1 = -1,
    // so the zero arm is taken, and IEEE (-0) + (+0) = +0.
    o[23] = asuint(nz + ((m1 > 0.0f) ? m1 : 0.0f));                        // +0
    o[24] = asuint(((m1 > 0.0f) ? m1 : 0.0f) + nz);                        // +0
    o[25] = asuint((float3(nz, nz, nz) + ((m1 > 0.0f) ? float3(m1, m1, m1) : float3(0.0f, 0.0f, 0.0f))).y);   // +0
}

// P9.86f: pxgSlangTriBox's AXISTEST/FINDMINMAX selects on NaN.  One thread per case, five float4
// per case (box centre, extents, tp0, tp1, tp2); writes tbIntersectTriangleBox as 0/1.  Engine
// consumer convexTrimeshMidphaseBda is NoContraction and SZINP-stamped; so is p86SignedZeroSz.spv,
// the build this entry is run from, and the directed inputs overflow a*v.y/b*v.z to +-inf, which
// contraction would move -- the host mirror is -ffp-contract=off to match.
[shader("compute")]
[numthreads(1, 1, 1)]
void tbProbe(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<float4> i = Ptr<float4>(gSz.inp);
    Ptr<uint> o = Ptr<uint>(gSz.outp);
    const uint b = dtid.x * 5u;
    o[dtid.x] = tbIntersectTriangleBox(i[b].xyz, i[b + 1u].xyz, i[b + 2u].xyz, i[b + 3u].xyz, i[b + 4u].xyz) ? 1u : 0u;
}
