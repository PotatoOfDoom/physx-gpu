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

// npCapsuleCollision.slang
//
// Slang port of the two-contact capsule collision routines from
// gpunarrowphase/src/CUDA/sphereCollision.cuh (used by cudaSphere.cu's
// sphereNphase_Kernel): planeCapsule and capsuleCapsule, plus the shared
// segment-segment distance helpers (distanceSegmentSegmentSquared and
// pcmDistancePointSegmentTValue22 from distanceSegmentSegment.cuh).  Each can
// produce up to two contacts (world point + penetration per contact in
// gOutPointPen[2*i + {0,1}], a shared world normal in gOutNormal[i]).
//
// capsuleCapsule closest-features: the general case is one contact at the
// closest points of the two axis segments; near-parallel capsules (|cos| >
// 0.9998) fall back to projecting each capsule's endpoints onto the other axis
// (pcmDistancePointSegmentTValue22) to recover up to two contacts; degenerate
// (inline) segments emit a single -radius0 contact.

import pxgSlangCommon;
import npSpherePrimitives; // planeCapsule / capsuleCapsule contact routines + PrimContact

struct Xform
{
    float4 p; // .xyz position
    float4 q; // quaternion (x,y,z,w)
}

struct CapsuleTest
{
    float radius0;     // planeCapsule: capsule radius
    float halfHeight0; // planeCapsule: capsule half height
    float radius1;
    float halfHeight1;
    float contactDist;
    float pad0, pad1, pad2;
}

struct CapsuleParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CapsuleParams> gParams;

RWStructuredBuffer<Xform>       gXform0;      // binding 0
RWStructuredBuffer<Xform>       gXform1;      // binding 1
RWStructuredBuffer<CapsuleTest> gTests;       // binding 2
RWStructuredBuffer<float4>      gOutPointPen; // binding 3  (2 per test)
RWStructuredBuffer<float4>      gOutNormal;   // binding 4  (1 per test)
RWStructuredBuffer<uint>        gNbContacts;  // binding 5

// The contact math lives in npSpherePrimitives (single source of truth shared with
// the engine-ABI dispatcher sphereNphase_Kernel); each entry point below just marshals
// the bound buffers into the shared routine and writes the result back.

// ---------------------------------------------------------------------------
// planeCapsule (0 = plane at transform0, 1 = capsule at transform1)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void planeCapsule(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const CapsuleTest ct = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const PrimContact c = planeCapsuleContact(t0.q, t0.p.xyz, t1.q, t1.p.xyz,
                                              ct.radius0, ct.halfHeight0, ct.contactDist);
    if (c.nbContacts > 0u) gOutPointPen[2u * i + 0u] = c.pointPen0;
    if (c.nbContacts > 1u) gOutPointPen[2u * i + 1u] = c.pointPen1;
    if (c.nbContacts > 0u) gOutNormal[i] = float4(c.normal, 0.0f);
    gNbContacts[i] = c.nbContacts;
}

// ---------------------------------------------------------------------------
// capsuleCapsule
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void capsuleCapsule(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const CapsuleTest ct = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const PrimContact c = capsuleCapsuleContact(t0.q, t0.p.xyz, ct.radius0, ct.halfHeight0,
                                                t1.q, t1.p.xyz, ct.radius1, ct.halfHeight1, ct.contactDist);
    if (c.nbContacts > 0u) gOutPointPen[2u * i + 0u] = c.pointPen0;
    if (c.nbContacts > 1u) gOutPointPen[2u * i + 1u] = c.pointPen1;
    if (c.nbContacts > 0u) gOutNormal[i] = float4(c.normal, 0.0f);
    gNbContacts[i] = c.nbContacts;
}
