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

// npCvxAddManifold.slang
//
// Slang port of addManifoldPoint() from cudaGJKEPA.cu (~L553): insert a new
// contact into a persistent-contact manifold (<= 4 points).
//   - each existing point flags a match if its A or B endpoint is within
//     replaceBreakingThreshold of the new point's;
//   - if any match, or the manifold is not yet full, the new point replaces the
//     lowest matching slot (or appends, numContacts++);
//   - otherwise the manifold is full and matchless: run the schlock::reduce
//     4-extreme selection over the 5 candidates (the 4 existing points in lanes
//     0..3 + the new one in lane 4, reduced over their B endpoints) and compact
//     the chosen four back into the manifold (writeContacts / warpScanExclusive).
//
// The CUDA runs one warp; this port runs one thread per problem and serialises
// the 5-lane reduce + compaction (minIndex/maxIndex = lowest lane holding the
// extreme, matching lowestSetIndex(ballot(best==input))).

uint lowestSetIndex(uint mask) { return firstbitlow(mask); }

static const float FLT_MAXV = 3.4028235e38f;

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

struct AddManifoldIn
{
    float4 mA[4];   // existing manifold A points
    float4 mB[4];   // existing manifold B points
    float4 mNp[4];  // existing manifold normal(.xyz)+pen(.w)
    float4 newA;    // new point A (.xyz)
    float4 newB;    // new point B (.xyz)
    float4 newNp;   // new normal(.xyz) + sep(.w)
    uint numContacts;
    float rbt;      // replaceBreakingThreshold
    uint pad0, pad1;
}

struct AddManifoldOut
{
    float4 mA[4];
    float4 mB[4];
    float4 mNp[4];
    uint numContacts;
    uint pad0, pad1, pad2;
}

struct AddManifoldParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AddManifoldParams> gParams;

RWStructuredBuffer<AddManifoldIn> gIn;    // binding 0
RWStructuredBuffer<AddManifoldOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxAddManifold(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const AddManifoldIn s = gIn[t];

    float3 A[4], B[4], N[4];
    float S[4];
    for (uint j = 0u; j < 4u; ++j)
    {
        A[j] = s.mA[j].xyz;
        B[j] = s.mB[j].xyz;
        N[j] = s.mNp[j].xyz;
        S[j] = s.mNp[j].w;
    }
    const float3 newA = s.newA.xyz;
    const float3 newB = s.newB.xyz;
    const float3 newN = s.newNp.xyz;
    const float newS = s.newNp.w;
    uint numContacts = s.numContacts;
    const float sqRbt = s.rbt * s.rbt;

    // per-existing-point match test -> bitmask
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
        // full, matchless: reduce the 5 candidates (B endpoints), keep the best 4.
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

        // writeContacts: compact the selected lanes into manifold[0..3]
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

    AddManifoldOut r;
    for (uint j = 0u; j < 4u; ++j)
    {
        r.mA[j] = float4(A[j], 0.0f);
        r.mB[j] = float4(B[j], 0.0f);
        r.mNp[j] = float4(N[j], S[j]);
    }
    r.numContacts = numContacts;
    r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
    gOut[t] = r;
}
