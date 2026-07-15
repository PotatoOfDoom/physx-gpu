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

// npArtiPropagate.slang
//
// First compute unit of the gpuarticulation module.  Slang port of
// propagateImpulseW_everythingPreLoaded (articulationDynamic.cuh L167-185): the
// Featherstone/Mirtich propagation of a spatial impulse applied to a child link
// up to its parent through the inbound joint (Mirtich p141).  For each of the
// (up to 3) joint dofs it forms stZ = -(s_i . Z) with s_i the joint motion
// matrix row, accumulates qstZ[i] += stZ, and adds isInvD_i * stZ into the
// running spatial impulse; the result is then translated from child to parent.
//
// Cm::UnAlignedSpatialVector: top=angular, bottom=linear.
//
// ** CRITICAL: innerProduct is the spatial (motion/force duality) product, NOT
//    the Euclidean dot. **  CmSpatialVector.h L398-401:
//      a.innerProduct(b) = a.bottom.dot(b.top) + a.top.dot(b.bottom)   [CROSS]
//    (contrast a.dot(b) = a.top.dot(b.top) + a.bottom.dot(b.bottom)).  So
//    stZ = -(motionMatrix.bottom . Z.top + motionMatrix.top . Z.bottom).
//
// translateSpatialVector(offset, vec) (DyFeatherstoneArticulation.h L828-831):
//   = (vec.top, vec.bottom + offset x vec.top).  top unchanged; bottom gets
//   offset x top added.  Here offset = childToParent.
//
// All float arith NoContraction-stamped (in PXG_NOCONTRACT_SHADERS); +-0
// tolerated via eqFZ.

static const uint MAX_DOF = 3u;

struct ApIn
{
    float4 Ztop;       // Z.top.xyz (angular)
    float4 Zbot;       // Z.bottom.xyz (linear)
    float4 c2p;        // childToParent.xyz
    uint4 cfg;         // dofCount
    float4 mmTop[3];   // motionMatrix[i].top
    float4 mmBot[3];   // motionMatrix[i].bottom
    float4 idTop[3];   // isInvD[i].top
    float4 idBot[3];   // isInvD[i].bottom
    float4 qstZin;     // xyz = qstZ[0..2] initial accumulator
}

struct ApOut
{
    float4 propTop;    // propagated spatial impulse .top
    float4 propBot;    // propagated spatial impulse .bottom
    float4 qstZout;    // xyz = qstZ[0..2] final
}

struct ApParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ApParams> gParams;

RWStructuredBuffer<ApIn> gIn;    // binding 0
RWStructuredBuffer<ApOut> gOut;  // binding 1

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b)  // PxVec3::cross(a, b) = a x b
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apImpulse(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const ApIn s = gIn[tid];

    const uint dofCount = min(s.cfg.x, MAX_DOF);

    const float3 Zt = s.Ztop.xyz;
    const float3 Zb = s.Zbot.xyz;

    float3 tempT = Zt;   // temp = Z
    float3 tempB = Zb;
    float qz[3] = { s.qstZin.x, s.qstZin.y, s.qstZin.z };

    for (uint ind = 0u; ind < MAX_DOF; ++ind)
    {
        if (ind < dofCount)
        {
            // stZ = -motionMatrix[ind].innerProduct(Z)   [Z, not temp]
            const float stZ = -(dot3(s.mmBot[ind].xyz, Zt) + dot3(s.mmTop[ind].xyz, Zb));
            qz[ind] += stZ;
            tempT += s.idTop[ind].xyz * stZ;   // temp += isInvD[ind] * stZ
            tempB += s.idBot[ind].xyz * stZ;
        }
    }

    // translateSpatialVector(childToParent, temp): top unchanged, bottom += c2p x top
    const float3 c2p = s.c2p.xyz;
    ApOut o;
    o.propTop = float4(tempT, 0.0f);
    o.propBot = float4(tempB + cross3(c2p, tempT), 0.0f);
    o.qstZout = float4(qz[0], qz[1], qz[2], 0.0f);
    gOut[tid] = o;
}
