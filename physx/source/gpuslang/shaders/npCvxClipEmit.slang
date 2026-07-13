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

// npCvxClipEmit.slang
//
// Slang port of the polyClip contact-emission tail of generateConvexContacts
// (cudaGJKEPA.cu L960-1015, both branches).  Given the polyClip output (nbContacts
// clip points + separations) and the clipping-plane normal, it reduces the set to
// <=4 (schlock::reduce<false,false,4>, tranche 137, only when nbContacts>4),
// derives the paired manifold point (otherPt = clipPt + clipN*sep), and compacts
// the selected contacts into the manifold via writeContacts (warpScanExclusive):
//   refIsReference:  A = aToB.transformInv(otherPt), B = clipPt,  normal =  clipN
//   else:            A = aToB.transformInv(clipPt),  B = otherPt, normal = -clipN
// numManifold = popcount(mask).  This is the meaty glue between polyClip (T145)
// and the manifold; it composes reduce (T137) + writeContacts.

static const float FLT_MAXV = 3.4028235e38f;

float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 transformInvPoint(float4 q, float3 p, float3 v) { return quatRotateInv(q, v - p); }

uint minIdx5(float v[32], uint mask)
{
    float mn = FLT_MAXV;
    for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u) mn = min(mn, v[L]);
    for (uint L = 0u; L < 32u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mn) return L;
    return 0u;
}
uint maxIdx5(float v[32], uint mask)
{
    float mx = -FLT_MAXV;
    for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u) mx = max(mx, v[L]);
    for (uint L = 0u; L < 32u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mx) return L;
    return 0u;
}

struct ClipEmitIn
{
    float4 clipPt[32]; // clip points (.xyz), .w = separation
    float4 clipN;      // clipping-plane normal (.xyz); .w = nbContacts (as float)
    float4 aToBq;
    float4 aToBp;      // .xyz translation; .w = refIsReference (1/0)
}

struct ClipEmitOut
{
    float4 mA[4];
    float4 mB[4];
    float4 mNp[4];
    uint numManifold;
    uint pad0, pad1, pad2;
}

struct ClipEmitParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ClipEmitParams> gParams;

RWStructuredBuffer<ClipEmitIn> gIn;    // binding 0
RWStructuredBuffer<ClipEmitOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxClipEmit(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const ClipEmitIn s = gIn[t];

    const uint nbContacts = uint(s.clipN.w);
    const float3 clipN = s.clipN.xyz;
    const float3 aToBp = s.aToBp.xyz;
    const bool refIsReference = s.aToBp.w != 0.0f;

    float3 pt[32]; float sep[32];
    for (uint L = 0u; L < 32u; ++L) { pt[L] = s.clipPt[L].xyz; sep[L] = s.clipPt[L].w; }

    // nbContacts==32 is clamped to a full mask rather than replicating CUDA's
    // (1<<32)-1, which is C++ shift-by-width UB (yields 0 on NVIDIA HW → empty
    // set); the clamp is the portable/correct behaviour and the boundary is
    // unreachable in practice (NUM_TMP_CONTACTS_PER_PAIR caps emission before 32).
    const uint allMask = (nbContacts >= 32u) ? 0xffffffffu : ((1u << nbContacts) - 1u);

    // mask: reduce to <=4 when nbContacts>4 (schlock::reduce over clip points, normal=clipN)
    uint mask;
    if (nbContacts > 4u)
    {
        const uint i0 = minIdx5(sep, allMask);
        mask = (1u << i0);
        float d1[32];
        for (uint L = 0u; L < 32u; ++L) { const float3 df = pt[i0] - pt[L]; d1[L] = dot(df, df); }
        const uint i1 = maxIdx5(d1, allMask & ~mask);
        mask |= (1u << i1);
        const float3 dir = cross(clipN, pt[i1] - pt[i0]); // per-lane normal is uniform (clipN)
        float d[32];
        for (uint L = 0u; L < 32u; ++L) d[L] = dot(dir, pt[L] - pt[i0]);
        const uint f = maxIdx5(d, allMask & ~mask);
        mask |= (1u << f);
        uint g = minIdx5(d, allMask & ~mask);
        if (d[f] * d[g] > 0.0f) g = maxIdx5(d, allMask & ~mask);
        mask |= (1u << g);
    }
    else
    {
        mask = allMask;
    }

    // writeContacts: compact the selected contacts into the manifold
    float3 mA[4], mB[4], mN[4]; float mS[4];
    for (uint j = 0u; j < 4u; ++j) { mA[j] = float3(0, 0, 0); mB[j] = float3(0, 0, 0); mN[j] = float3(0, 0, 0); mS[j] = 0.0f; }

    for (uint L = 0u; L < 32u; ++L)
    {
        if ((mask & (1u << L)) != 0u)
        {
            const uint idx = countbits(mask & ((1u << L) - 1u));
            if (idx < 4u)
            {
                const float3 clipPt = pt[L];
                const float sepL = sep[L];
                const float3 otherPt = clipPt + clipN * sepL;
                float3 pa, pb, nrm;
                if (refIsReference) { pa = otherPt; pb = clipPt; nrm = clipN; }
                else                { pa = clipPt; pb = otherPt; nrm = -clipN; }
                mA[idx] = transformInvPoint(s.aToBq, aToBp, pa);
                mB[idx] = pb;
                mN[idx] = nrm;
                mS[idx] = sepL;
            }
        }
    }

    ClipEmitOut r;
    for (uint j = 0u; j < 4u; ++j) { r.mA[j] = float4(mA[j], 0.0f); r.mB[j] = float4(mB[j], 0.0f); r.mNp[j] = float4(mN[j], mS[j]); }
    r.numManifold = countbits(mask);
    r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
    gOut[t] = r;
}
