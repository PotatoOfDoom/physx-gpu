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

// npCvxContactDecide.slang
//
// Slang port of the front half of generateConvexContacts (cudaGJKEPA.cu L858-887):
// the GJK/EPA-witness manifold update and the fullContactGen decision that gates
// whether the polyClip full clipper runs.  Given the persistent-contact manifold
// (<=4 points) and the GJK/EPA output (witnessA, witnessB, separating normal), it:
//   1. localNor = calculateLocalNormal(manifoldNormalPen, numContacts) if numContacts>0
//      else (0,0,0)  -- computed BEFORE the witness is added (pre-add count);
//   2. addManifoldPoint(aToB.transformInv(witnessA), witnessB, -normal,
//      (witnessB-witnessA).dot(normal), rbt, ...)  -- mutates the manifold + count;
//   3. fullContactGen = (initialNbContacts > 1) || (0.707106781 > localNor.dot(-normal))
//      || (numContacts < 4) where numContacts is the POST-add count.  The
//      initialNbContacts term is 5.12's (cudaGJKEPA.cu:864, the PCM jitter fix).
// Composes calculateLocalNormal (tranche 139) + addManifoldPoint (tranche 140).

uint lowestSetIndex(uint mask) { return firstbitlow(mask); }

static const float FLT_MAXV = 3.4028235e38f;

float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

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

// argmin/argmax over masked lanes of a 5-entry array, lowest-lane tie-break.
uint minIndex5(float v[5], uint mask)
{
    float mn = FLT_MAXV;
    for (uint L = 0u; L < 5u; ++L) if ((mask & (1u << L)) != 0u) mn = min(mn, v[L]);
    for (uint L = 0u; L < 5u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mn) return L;
    return 0u;
}
uint maxIndex5(float v[5], uint mask)
{
    float mx = -FLT_MAXV;
    for (uint L = 0u; L < 5u; ++L) if ((mask & (1u << L)) != 0u) mx = max(mx, v[L]);
    for (uint L = 0u; L < 5u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mx) return L;
    return 0u;
}

struct ContactDecideIn
{
    float4 mA[4];    // existing manifold A points
    float4 mB[4];    // existing manifold B points
    float4 mNp[4];   // existing manifold normal(.xyz)+pen(.w)
    float4 witnessA; // gjkOutput.closestPointA (.xyz)
    float4 witnessB; // gjkOutput.closestPointB (.xyz)
    float4 normal;   // gjkOutput.direction (.xyz), unit
    float4 aToBq;
    float4 aToBp;    // .xyz translation; .w = replaceBreakingThreshold
    uint numContacts;
    uint initialNbContacts;  // ss_scratch.initialNbContacts (5.12 predicate term)
    uint pad1, pad2;
}

struct ContactDecideOut
{
    float4 mA[4];
    float4 mB[4];
    float4 mNp[4];
    float4 localNor;      // localNormal (.xyz) computed pre-add
    uint numContacts;     // post-add
    uint fullContactGen;  // 1/0
    uint pad0, pad1;
}

struct ContactDecideParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<ContactDecideParams> gParams;

RWStructuredBuffer<ContactDecideIn> gIn;    // binding 0
RWStructuredBuffer<ContactDecideOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxContactDecide(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const ContactDecideIn s = gIn[t];

    float3 A[4], B[4], N[4];
    float S[4];
    for (uint j = 0u; j < 4u; ++j)
    {
        A[j] = s.mA[j].xyz;
        B[j] = s.mB[j].xyz;
        N[j] = s.mNp[j].xyz;
        S[j] = s.mNp[j].w;
    }
    uint numContacts = s.numContacts;

    const float3 normal = s.normal.xyz;
    const float3 witnessA = s.witnessA.xyz;
    const float3 witnessB = s.witnessB.xyz;
    const float3 aToBp = s.aToBp.xyz;
    const float rbt = s.aToBp.w;

    // 1. localNor = calculateLocalNormal (pairwise sum over the pre-add manifold),
    //    only when numContacts > 0 (else it stays (0,0,0)).
    float3 localNor = float3(0, 0, 0);
    if (numContacts != 0u)
    {
        float3 n0 = (0u < numContacts) ? N[0] : float3(0, 0, 0);
        float3 n1 = (1u < numContacts) ? N[1] : float3(0, 0, 0);
        float3 n2 = (2u < numContacts) ? N[2] : float3(0, 0, 0);
        float3 n3 = (3u < numContacts) ? N[3] : float3(0, 0, 0);
        localNor = getNormalized((n0 + n1) + (n2 + n3));
    }

    // 2. addManifoldPoint(aToB.transformInv(witnessA), witnessB, -normal, sep, ...)
    const float3 newA = transformInvPoint(s.aToBq, aToBp, witnessA);
    const float3 newB = witnessB;
    const float3 newN = -normal;
    const float newS = dot(witnessB - witnessA, normal);
    const float sqRbt = rbt * rbt;

    uint matches = 0u;
    for (uint j = 0u; j < numContacts; ++j)
    {
        const float3 db = B[j] - newB;
        const float3 da = A[j] - newA;
        const bool bMatch = dot(db, db) < sqRbt;
        const bool aMatch = dot(da, da) < sqRbt;
        if (aMatch || bMatch) matches |= (1u << j);
    }

    if (matches != 0u || numContacts < 4u)
    {
        uint index;
        if (matches != 0u) { index = lowestSetIndex(matches); }
        else { index = numContacts; numContacts = numContacts + 1u; }
        N[index] = newN; S[index] = newS; A[index] = newA; B[index] = newB;
    }
    else
    {
        float3 cA[5], cB[5], cN[5];
        float cS[5];
        for (uint j = 0u; j < 4u; ++j) { cA[j] = A[j]; cB[j] = B[j]; cN[j] = N[j]; cS[j] = S[j]; }
        cA[4] = newA; cB[4] = newB; cN[4] = newN; cS[4] = newS;

        const uint allMask = 0x1fu; // 5 candidates

        const uint i0 = minIndex5(cS, allMask);          // deepest
        uint mask = (1u << i0);

        float d1[5];
        for (uint L = 0u; L < 5u; ++L) { const float3 df = cB[i0] - cB[L]; d1[L] = dot(df, df); }
        const uint i1 = maxIndex5(d1, allMask & ~mask);  // furthest from i0
        mask |= (1u << i1);

        float d[5];
        for (uint L = 0u; L < 5u; ++L)
        {
            const float3 dir = cross(cN[L], cB[i1] - cB[i0]);
            d[L] = dot(dir, cB[L] - cB[i0]);
        }
        const uint f = maxIndex5(d, allMask & ~mask);
        mask |= (1u << f);
        uint g = minIndex5(d, allMask & ~mask);
        if (d[f] * d[g] > 0.0f)
            g = maxIndex5(d, allMask & ~mask);
        mask |= (1u << g);

        for (uint L = 0u; L < 5u; ++L)
        {
            if ((mask & (1u << L)) != 0u)
            {
                const uint idx = countbits(mask & ((1u << L) - 1u));
                N[idx] = cN[L]; S[idx] = cS[L]; A[idx] = cA[L]; B[idx] = cB[L];
            }
        }
        numContacts = countbits(mask); // 4
    }

    // 3. fullContactGen decision (numContacts is the post-add count)
    const bool fullContactGen = (s.initialNbContacts > 1u)   // 5.12: "added to fix a jittering bug"
                                || (0.707106781f > dot(localNor, -normal)) || (numContacts < 4u);

    ContactDecideOut r;
    for (uint j = 0u; j < 4u; ++j)
    {
        r.mA[j] = float4(A[j], 0.0f);
        r.mB[j] = float4(B[j], 0.0f);
        r.mNp[j] = float4(N[j], S[j]);
    }
    r.localNor = float4(localNor, 0.0f);
    r.numContacts = numContacts;
    r.fullContactGen = fullContactGen ? 1u : 0u;
    r.pad0 = 0u; r.pad1 = 0u;
    gOut[t] = r;
}
