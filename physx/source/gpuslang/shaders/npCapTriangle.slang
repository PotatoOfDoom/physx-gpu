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

// npCapTriangle.slang
//
// Slang port of the capsule/segment-vs-triangle contact-generation routines from
// gpunarrowphase/src/CUDA/capsuleTriangle.cuh:
//   - generateContacts: projects the segment endpoints p,q onto the triangle
//     plane along the contact normal; each projection that lands inside the
//     triangle and within the inflated radius becomes a contact (endpoint +
//     signed plane distance);
//   - generateEEContacts / generateEE: the segment-vs-triangle-edge contacts —
//     for each of the 3 edges, the segment/edge crossing point (if within the
//     inflated radius and the edge span) becomes a contact, capped at 2.
// Both emit up to 2 contacts (point in .xyz, penetration/signed distance in .w).

static const float PXG_MAX_F32 = 3.4028235e38f;

struct GenContactsIn
{
    float4 a, b, c;       // triangle
    float4 planeNormal;   // triangle plane normal
    float4 normal;        // contact normal (projection direction)
    float4 p, q;          // segment endpoints
    float4 misc;          // (inflatedRadius, ...)
}

struct GenEEIn
{
    float4 a, b, c;   // triangle
    float4 normal;    // contact normal
    float4 p, q;      // segment endpoints
    float4 misc;      // (sqInflatedRadius, ...)
}

struct OutRes { float4 c0, c1, meta; } // c*.xyz point, c*.w pen; meta.x = count

struct CapTriParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CapTriParams> gParams;

RWStructuredBuffer<GenContactsIn> gGCIn;   // binding 0
RWStructuredBuffer<OutRes>        gGCOut;  // binding 1
RWStructuredBuffer<GenEEIn>       gEEIn;   // binding 2
RWStructuredBuffer<OutRes>        gEEOut;  // binding 3

bool isValidBary(float v, float w)
{
    const float eps = 1e-6f;
    const float zero = -eps;
    const float one = 1.0f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

// ---------------------------------------------------------------------------
// generateContacts: endpoint-projection contacts
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void capTriGenContacts(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const GenContactsIn s = gGCIn[t];
    const float3 a = s.a.xyz, b = s.b.xyz, c = s.c.xyz;
    const float3 planeNormal = s.planeNormal.xyz, normal = s.normal.xyz;
    const float3 p = s.p.xyz, q = s.q.xyz;
    const float inflatedRadius = s.misc.x;

    const float3 ab = b - a, ac = c - a, ap = p - a, aq = q - a;
    const float d00 = dot(ab, ab), d01 = dot(ab, ac), d11 = dot(ac, ac);
    const float tdenom = d00 * d11 - d01 * d01;
    const float bdenom = tdenom > 0.0f ? 1.0f / tdenom : PXG_MAX_F32;

    const float inomp = dot(planeNormal, -ap);
    const float ideom = dot(planeNormal, normal);
    const float ipt = ideom > 0.0f ? -(inomp / ideom) : PXG_MAX_F32;

    const float3 cp31 = p - normal * ipt;
    const float3 pV20 = cp31 - a;
    const float pD20 = dot(pV20, ab), pD21 = dot(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = isValidBary(v0, w0);

    float4 c0 = float4(0, 0, 0, 0), c1 = float4(0, 0, 0, 0);
    uint count = 0u;
    if (con0 && (inflatedRadius > ipt)) { c0 = float4(p, ipt); count = 1u; }

    const float inomq = dot(planeNormal, -aq);
    const float iqt = ideom > 0.0f ? -(inomq / ideom) : PXG_MAX_F32;
    const float3 cp41 = q - normal * iqt;
    const float3 qV20 = cp41 - a;
    const float qD20 = dot(qV20, ab), qD21 = dot(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = isValidBary(v1, w1);

    if (con1 && (inflatedRadius > iqt))
    {
        const float4 hit = float4(q, iqt);
        if (count == 0u) c0 = hit; else c1 = hit;
        count++;
    }

    OutRes r; r.c0 = c0; r.c1 = c1; r.meta = float4(float(count), 0, 0, 0);
    gGCOut[t] = r;
}

// generateEE: segment vs one triangle edge (a,b); appends a contact if any
void generateEE(float3 p, float3 q, float sqInflatedRadius, float3 normal, float3 a, float3 b,
                inout uint count, inout float4 c0, inout float4 c1)
{
    const float3 ab = b - a;
    const float3 n = cross(ab, normal);
    const float d = dot(n, a);
    const float np = dot(n, p);
    const float nq = dot(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    if (signP * signQ > 0.0f) return; // both endpoints same side

    const float3 pq = q - p;
    const float npq = dot(n, pq);
    if (npq == 0.0f) return; // parallel

    const float segTValue = (d - np) / npq;
    const float3 localA = p + pq * segTValue;

    const float3 perNormal = cross(normal, pq);
    const float3 ap = localA - a;
    const float nom = dot(perNormal, ap);
    const float denom = dot(perNormal, ab);
    const float tValue = nom / denom;
    if (tValue > 1.0f || tValue < 0.0f) return;

    const float3 v = ap - ab * tValue;
    const float sqDist = dot(v, v);
    if (sqInflatedRadius > sqDist)
    {
        const float signedDist = dot(v, normal);
        const float4 hit = float4(localA, signedDist);
        if (count == 0u) c0 = hit; else c1 = hit;
        count++;
    }
}

// ---------------------------------------------------------------------------
// generateEEContacts: edge-edge contacts over the 3 triangle edges (cap 2)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void capTriGenEE(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const GenEEIn s = gEEIn[t];
    const float3 a = s.a.xyz, b = s.b.xyz, c = s.c.xyz, normal = s.normal.xyz;
    const float3 p = s.p.xyz, q = s.q.xyz;
    const float sqIR = s.misc.x;

    float4 c0 = float4(0, 0, 0, 0), c1 = float4(0, 0, 0, 0);
    uint count = 0u;
    if (count < 2u) generateEE(p, q, sqIR, normal, a, b, count, c0, c1);
    if (count < 2u) generateEE(p, q, sqIR, normal, b, c, count, c0, c1);
    if (count < 2u) generateEE(p, q, sqIR, normal, a, c, count, c0, c1);

    OutRes r; r.c0 = c0; r.c1 = c1; r.meta = float4(float(count), 0, 0, 0);
    gEEOut[t] = r;
}
