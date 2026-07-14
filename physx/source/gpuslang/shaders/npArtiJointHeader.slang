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

// npArtiJointHeader.slang
//
// Slang port of the jointHeader stores of setupArtiSolverConstraintBlockGPU
// in gpusolver/src/CUDA/artiConstraintPrep2.cu -- the PGS wrapper
// (L1716-1767) and the TGS wrapper (L1770-1830).  The sort + row-loop
// portion is already ported and verified by npArti1DPrep.slang
// (a1dPGS/a1dTGS, tranches 182/183); this unit isolates the header math
// the wrapper writes AROUND that call.  The row loop's return value
// (outCount) is fed in as an input, since the sort+row-loop already proves
// outCount == the CUDA return bit-exactly; the linkage rowCounts = outCount
// (NOT numRows -- the PGS wrapper's first PxU8(numRows) store at L1737 is
// dead, overwritten at L1765; the TGS pre-store L1790 is commented out) is
// transcribed directly and mutant-pinned.
//
// Header field layouts (PxgSolverConstraintBlock1D.h):
//   PGS PxgBlockSolverConstraint1DHeader: rowCounts/breakable (PxU16),
//     angBreakImpulse, body0WorldOffset_linBreakImpulse (float4),
//     invMass0D0/invMass1D1/invInertiaScale0/invInertiaScale1 (PxReal),
//     writeBackOffset (PxU32), cfm (PxReal).
//   TGS PxgTGSBlockSolverConstraint1DHeader:
//     rowCounts_breakable_orthoAxisCount (uchar4),
//     rAWorld_invMass0D0/rBWorld_invMass1D1 (float4),
//     invInertiaScale0/invInertiaScale1 (PxReal), writeBackOffset (PxU32),
//     linBreakImpulse/angBreakImpulse (PxReal), cfm (PxReal).
// Negations land on the "0" side (invMass0D0 = -lin0X, invInertiaScale0 =
// -ang0Y); the "1" side is unnegated (invMass1D1 = +lin1Z, invInertiaScale1
// = +ang1W).  breakForce comparisons are exact float != against PX_MAX_F32.
// cfm = PxMax(b0.cfm, b1.cfm) is LIVE here (unlike the row path where it is
// a dead parameter).  All arith NoContraction-stamped; +-0 tolerated.

static const float PX_MAX_F32_Q = 3.402823466e+38f;

struct JhIn
{
    float4 invMassScale;  // lin0X_ang0Y_lin1Z_ang1W
    float4 raWorld;       // raWorld_linBreakForce (xyz world offset, w = linBreakForce)
    float4 rbWorld;       // rbWorld_angBreakForce (xyz world offset, w = angBreakForce)
    uint4 misc;           // writeBackIndex, numRows, outCount, 0
    float4 scal;          // params.dt, b0.cfm, b1.cfm, 0
}

struct JhPgOut
{
    uint4 hdr;            // rowCounts, breakable, writeBackOffset, 0
    float4 b0wo;          // body0WorldOffset_linBreakImpulse (raWorld.xyz, linBreakImpulse)
    float4 v0;            // angBreakImpulse, invMass0D0, invMass1D1, invInertiaScale0
    float4 v1;            // invInertiaScale1, cfm, 0, 0
}

struct JhTgOut
{
    uint4 hdr;            // rowCounts_breakable_packed (uchar4), writeBackOffset, 0, 0
    float4 rA;            // rAWorld_invMass0D0 (rbWorld... no: raWorld.xyz, -lin0X)
    float4 rB;            // rBWorld_invMass1D1 (rbWorld.xyz, +lin1Z)
    float4 v0;            // invInertiaScale0, invInertiaScale1, linBreakImpulse, angBreakImpulse
    float4 v1;            // cfm, 0, 0, 0
}

struct JhParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<JhParams> gParams;

RWStructuredBuffer<JhIn> gIn;      // binding 0
RWStructuredBuffer<JhPgOut> gPg;   // binding 1
RWStructuredBuffer<JhTgOut> gTg;   // binding 2

float pxMaxQ(float a, float b) { return (a > b) ? a : b; }  // PxMax float = intrinsics::selectMax

uint breakableQ(float linBreakForce, float angBreakForce)
{
    // PxU8((raWorld.w != PX_MAX_F32) || (angBreakForce != PX_MAX_F32))
    return ((linBreakForce != PX_MAX_F32_Q) || (angBreakForce != PX_MAX_F32_Q)) ? 1u : 0u;
}

// setupArtiSolverConstraintBlockGPU PGS header stores (cu L1736-1759)
[shader("compute")]
[numthreads(64, 1, 1)]
void ajhPGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const JhIn s = gIn[tid];

    const float4 ims = s.invMassScale;
    const float dt = s.scal.x;

    const float linBreakForce = s.raWorld.w;
    const float angBreakForce = s.rbWorld.w;
    const float linBreakImpulse = linBreakForce * dt;
    const float angBreakImpulse = angBreakForce * dt;

    const float cfm = pxMaxQ(s.scal.y, s.scal.z);

    JhPgOut o;
    // rowCounts = setUpArti1D return (outCount); the earlier PxU8(numRows)
    // store is dead (overwritten).  breakable = disjunction over break forces.
    o.hdr = uint4(s.misc.z, breakableQ(linBreakForce, angBreakForce), s.misc.x, 0u);
    o.b0wo = float4(s.raWorld.xyz, linBreakImpulse);
    o.v0 = float4(angBreakImpulse, -ims.x, ims.z, -ims.y);
    o.v1 = float4(ims.w, cfm, 0.0f, 0.0f);

    gPg[tid] = o;
}

// setupArtiSolverConstraintBlockGPU TGS header stores (cu L1790-1829)
[shader("compute")]
[numthreads(64, 1, 1)]
void ajhTGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const JhIn s = gIn[tid];

    const float4 ims = s.invMassScale;
    const float dt = s.scal.x;  // TGS: params.dt = sharedDesc->stepDt

    const float linBreakForce = s.raWorld.w;
    const float angBreakForce = s.rbWorld.w;
    const float linBreakImpulse = linBreakForce * dt;
    const float angBreakImpulse = angBreakForce * dt;

    const float cfm = pxMaxQ(s.scal.y, s.scal.z);

    // rowCounts_breakable_orthoAxisCount = make_uchar4(outCount, breakable, 0, 0)
    const uint outCount = s.misc.z & 0xFFu;
    const uint breakable = breakableQ(linBreakForce, angBreakForce);
    const uint packed = outCount | (breakable << 8u);

    JhTgOut o;
    o.hdr = uint4(packed, s.misc.x, 0u, 0u);
    o.rA = float4(s.raWorld.xyz, -ims.x);   // rAWorld_invMass0D0
    o.rB = float4(s.rbWorld.xyz, ims.z);    // rBWorld_invMass1D1
    o.v0 = float4(-ims.y, ims.w, linBreakImpulse, angBreakImpulse);
    o.v1 = float4(cfm, 0.0f, 0.0f, 0.0f);

    gTg[tid] = o;
}
