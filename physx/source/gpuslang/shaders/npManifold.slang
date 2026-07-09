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

// npManifold.slang
//
// Slang port of the persistent-contact-manifold maintenance helpers from
// gpunarrowphase/src/CUDA/manifold.cuh: invalidateManifold (has the relative
// pose drifted enough — in translation or rotation — to force a full recompute)
// and refreshManifolds (re-evaluate each cached contact under the new relative
// transform: recompute its penetration and its 2D drift off the stored normal,
// and if any contact has drifted past the breaking threshold, discard the whole
// multi-manifold).
//
// A multi-manifold holds up to MULTIMANIFOLD_MAX_MANIFOLDS (4) sub-manifolds of
// up to SUBMANIFOLD_MAX_CONTACTS (6) contacts each; the CUDA refreshManifolds
// maps one warp lane to each of the (<=24) slots, so a serial loop over the 24
// slots is equivalent.

import pxgSlangCommon;

static const uint MULTIMANIFOLD_MAX_MANIFOLDS = 4u;
static const uint SUBMANIFOLD_MAX_CONTACTS = 6u;
static const uint MAX_SLOTS = 24u; // 4 * 6

struct Contact
{
    float4 pointA; // xyz
    float4 pointB; // xyz
    float4 nPen;   // xyz = normal, w = penetration
}

struct RefIn
{
    float4 aToBp;    // transform position
    float4 aToBq;    // transform rotation (quat)
    float4 misc;     // x = projectBreakingThreshold, y = numManifolds
    uint4 nbContacts; // per-sub-manifold contact counts
}

struct RefOut
{
    uint ret;      // lostContact (0/1)
    uint nbMan;    // resulting mNbManifolds
    uint p0, p1;
}

struct InvIn
{
    float4 aToBp, aToBq;   // new relative transform
    float4 prevP, prevQ;   // cached relative transform
    float4 misc;           // x = minMargin, y = ratio
}

struct MfParams { uint numTests; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<MfParams> gParams;

RWStructuredBuffer<Contact> gContacts; // binding 0  MAX_SLOTS per test
RWStructuredBuffer<RefIn> gRefIn;      // binding 1
RWStructuredBuffer<RefOut> gRefOut;    // binding 2
RWStructuredBuffer<float> gPenOut;     // binding 3  MAX_SLOTS per test
RWStructuredBuffer<InvIn> gInvIn;      // binding 4
RWStructuredBuffer<uint> gInvOut;      // binding 5

// PxTransform::transform(v) = p + q.rotate(v)
float3 transformPoint(float4 p, float4 q, float3 v)
{
    return p.xyz + quatRotate(q, v);
}

// ---------------------------------------------------------------------------
// mfRefresh: mirrors refreshManifolds.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void mfRefresh(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const RefIn e = gRefIn[t];

    const float projectBreakingThreshold = e.misc.x;
    const float sqThreshold = projectBreakingThreshold * projectBreakingThreshold;
    const uint numManifolds = uint(e.misc.y);
    const uint base = t * MAX_SLOTS;

    bool lostContact = false;

    for (uint i = 0u; i < MAX_SLOTS; ++i)
    {
        Contact c = gContacts[base + i];
        float penOut = c.nPen.w; // unprocessed slots keep the cached penetration

        const uint curManifoldIdx = i / SUBMANIFOLD_MAX_CONTACTS;
        const uint curContactIdx = i % SUBMANIFOLD_MAX_CONTACTS;
        if (curManifoldIdx < numManifolds)
        {
            const uint nbContacts = e.nbContacts[curManifoldIdx];
            if (curContactIdx < nbContacts)
            {
                const float3 pointA = c.pointA.xyz;
                const float3 pointB = c.pointB.xyz;
                const float3 normal = c.nPen.xyz;
                const float penOld = c.nPen.w;

                const float3 localAInB = transformPoint(e.aToBp, e.aToBq, pointA);
                const float3 v = localAInB - pointB;
                const float penNew = dot(v, normal);

                const float3 projectedPoint = localAInB - normal * penOld;
                const float3 projectedDifference = pointB - projectedPoint;
                const float distance2d = dot(projectedDifference, projectedDifference);

                penOut = penNew;
                if (distance2d > sqThreshold)
                    lostContact = true;
            }
        }
        gPenOut[base + i] = penOut;
    }

    RefOut o;
    o.ret = lostContact ? 1u : 0u;
    o.nbMan = lostContact ? 0u : numManifolds;
    o.p0 = 0u; o.p1 = 0u;
    gRefOut[t] = o;
}

// ---------------------------------------------------------------------------
// mfInvalidate: mirrors invalidateManifold.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void mfInvalidate(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const InvIn e = gInvIn[t];

    const float minMargin = e.misc.x;
    const float ratio = e.misc.y;

    const float thresholdP = minMargin * ratio;
    const float thresholdQ = 0.9998f; // about 1 degree

    const float3 delta = e.aToBp.xyz - e.prevP.xyz;
    const float3 absDelta = abs(delta);
    const float deltaP = max(absDelta.x, max(absDelta.y, absDelta.z));
    const float deltaQ = dot(e.aToBq, e.prevQ);

    const bool invalid = (deltaP > thresholdP) || (deltaQ < thresholdQ);
    gInvOut[t] = invalid ? 1u : 0u;
}
