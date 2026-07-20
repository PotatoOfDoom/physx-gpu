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

// npSphereCollision.slang
//
// Slang port of the single-contact sphere-primitive collision routines from
// gpunarrowphase/src/CUDA/sphereCollision.cuh (used by cudaSphere.cu's
// sphereNphase_Kernel): spheresphere, sphereplane, spherebox and spherecapsule.
// Each produces at most one contact (world point + penetration in outPointPen,
// world normal in outNormal, returning 0/1).  Here each routine is exposed as a
// batch compute entry point that runs one test per thread so the geometry can be
// verified directly against a C++ reference; the contact-stream output machinery
// (atomic patch/point allocation, material combining) is a later tranche.
//
// Shapes: sphere at transform0.p with radius sphereRadius; the second shape is at
// transform1 (a sphere of radius1, an x-axis plane, a box of half-extents
// boxExtents.xyz, or an x-axis capsule of radius1/halfHeight).

import pxgSlangCommon;
import npSpherePrimitives; // sphere*/spherePlane/sphereCapsule/sphereBox contact routines + PrimContact

struct Xform
{
    float4 p; // .xyz position
    float4 q; // quaternion (x,y,z,w)
}

struct SphereTest
{
    float sphereRadius;
    float contactDist;
    float radius1;    // sphere r1 / capsule radius
    float halfHeight; // capsule half height
    float4 boxExtents;
}

struct SphereParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SphereParams> gParams;

RWStructuredBuffer<Xform>       gXform0;      // binding 0
RWStructuredBuffer<Xform>       gXform1;      // binding 1
RWStructuredBuffer<SphereTest>  gTests;       // binding 2
RWStructuredBuffer<float4>      gOutPointPen; // binding 3  (world point, penetration)
RWStructuredBuffer<float4>      gOutNormal;   // binding 4  (world normal, w unused)
RWStructuredBuffer<uint>        gNbContacts;  // binding 5

// The contact math lives in npSpherePrimitives (single source of truth shared with
// the engine-ABI dispatcher sphereNphase_Kernel); each entry point below just marshals
// the bound buffers into the shared routine and writes the result back.

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_sphereSphere(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const PrimContact c = sphereSphereContact(t0.q, t0.p.xyz, s.sphereRadius,
                                              t1.q, t1.p.xyz, s.radius1, s.contactDist);
    if (c.nbContacts != 0u)
    {
        gOutPointPen[i] = c.pointPen0;
        gOutNormal[i] = float4(c.normal, 0.0f);
    }
    gNbContacts[i] = c.nbContacts;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_spherePlane(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const PrimContact c = spherePlaneContact(t0.q, t0.p.xyz, s.sphereRadius,
                                             t1.q, t1.p.xyz, s.contactDist);
    if (c.nbContacts != 0u)
    {
        gOutPointPen[i] = c.pointPen0;
        gOutNormal[i] = float4(c.normal, 0.0f);
    }
    gNbContacts[i] = c.nbContacts;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_sphereCapsule(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const PrimContact c = sphereCapsuleContact(t0.q, t0.p.xyz, s.sphereRadius,
                                               t1.q, t1.p.xyz, s.radius1, s.halfHeight, s.contactDist);
    if (c.nbContacts != 0u)
    {
        gOutPointPen[i] = c.pointPen0;
        gOutNormal[i] = float4(c.normal, 0.0f);
    }
    gNbContacts[i] = c.nbContacts;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sphereNphase_sphereBox(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const SphereTest s = gTests[i];
    const Xform t0 = gXform0[i];
    const Xform t1 = gXform1[i];

    const PrimContact c = sphereBoxContact(t0.q, t0.p.xyz, s.sphereRadius,
                                           t1.q, t1.p.xyz, s.boxExtents.xyz, s.contactDist);
    if (c.nbContacts != 0u)
    {
        gOutPointPen[i] = c.pointPen0;
        gOutNormal[i] = float4(c.normal, 0.0f);
    }
    gNbContacts[i] = c.nbContacts;
}
