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

// contactReduce.slang
//
// Slang port of the warp-wide contact-manifold reduction from
// gpucommon/src/CUDA/contactReduction.cuh — contactReduceShared and its two
// wrappers contactReduce (distanceAdjustment = 0, initial criterion =
// separation) and contactReduce2 (distanceAdjustment = -separation, initial
// criterion = the projected point's distance to the origin):
//
// 1. optional duplicate cull: a lane is a dupe when another masked lane sits
//    within 1e-8 squared distance and is deeper (or equally deep with a
//    higher lane index).
// 2. early out when <= TMaxPoints contacts survive.
// 3. anchor picks on the points projected into the contact plane:
//    p0 = argmax(initialCriterion + distanceAdjustment), p1 = argmax distance
//    to p0, then the two extremes of d = [n x (p1-p0)] . (p - p0)
//    (argmax of d + adj and argmin of d - adj).
// 4. optionally one extra deepest contact for PCM (when exactly 4 anchors
//    were picked and TMaxPoints > 4).
// 5. TClustering: bias the anchors' separations by -clusterBias, label every
//    lane with its nearest anchor (first anchor wins ties), keep the deepest
//    (biased) contact of each non-empty cluster, then fill the remaining
//    slots with the deepest unselected contacts; otherwise simply fill with
//    the deepest unselected contacts (unbiased separations).
//
// The CUDA distributes one warp lane per contact and reduces with
// minIndex/maxIndex (lowest-lane argmin/argmax over a masked set, exact-value
// fminf/fmaxf selects); the serial strict-comparison loops below (first index
// wins ties) are bit-identical.  One thread processes one 32-contact case;
// the template parameters become per-case runtime flags (branch outcomes are
// case-uniform, so this is output-identical).

static const float FLT_MAX_F = 3.4028235e38f;

struct CrParams { uint numTests; uint pad0, pad1, pad2; }

struct CrCfg
{
    uint4 cfg;          // x = allMask, y = flags (1 clustering, 2 keepDeepestPCM, 4 dupeTest, 8 useReduce2), z = maxPoints, w = 0
    float4 normal_bias; // xyz = normal, w = clusterBias
}

[[vk::push_constant]]
ConstantBuffer<CrParams> gParams;

RWStructuredBuffer<float4> gPts;   // binding 0: [test*32 + lane] xyz point, w separation
RWStructuredBuffer<CrCfg> gCfg;    // binding 1
RWStructuredBuffer<uint> gOut;     // binding 2: result mask

uint lowestArgMax(float vals[32], uint mask, out float best)
{
    best = -FLT_MAX_F;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}

uint lowestArgMin(float vals[32], uint mask, out float best)
{
    best = FLT_MAX_F;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void ctContactReduce(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CrCfg c = gCfg[t];
    uint allMask = c.cfg.x;
    const bool clustering = (c.cfg.y & 1) != 0;
    const bool keepDeepest = (c.cfg.y & 2) != 0;
    const bool dupeTest = (c.cfg.y & 4) != 0;
    const bool useReduce2 = (c.cfg.y & 8) != 0;
    const int maxPoints = int(c.cfg.z);
    const float3 normal = c.normal_bias.xyz;
    const float clusterBias = c.normal_bias.w;

    float3 ptRaw[32];
    float sep[32];
    for (uint j = 0; j < 32; ++j)
    {
        const float4 p = gPts[t * 32 + j];
        ptRaw[j] = p.xyz;
        sep[j] = p.w;
    }

    // per-lane projected point, distance adjustment and initial criterion
    float3 point[32];
    float distAdj[32];
    float initCrit[32];
    for (uint j = 0; j < 32; ++j)
    {
        point[j] = ptRaw[j] - normal * dot(ptRaw[j], normal);
        if (useReduce2)
        {
            distAdj[j] = -sep[j];
            initCrit[j] = length(point[j]);
        }
        else
        {
            distAdj[j] = 0.0f;
            initCrit[j] = sep[j];
        }
    }

    // 1. duplicate cull
    if (dupeTest)
    {
        uint dupeBallot = 0;
        for (uint j = 0; j < 32; ++j)
        {
            if (((allMask >> j) & 1) == 0) continue;
            bool imADupe = false;
            for (uint i = 0; i < 32; ++i)
            {
                if (((allMask >> i) & 1) == 0) continue;
                const float3 dv = ptRaw[i] - ptRaw[j];
                const float d = dot(dv, dv);
                if (d < 1e-8f && (sep[i] < sep[j] || (sep[i] == sep[j] && i > j)))
                    imADupe = true;
            }
            if (imADupe) dupeBallot |= 1u << j;
        }
        allMask &= ~dupeBallot;
    }

    const uint newCount = countbits(allMask);
    if (newCount <= uint(maxPoints))
    {
        gOut[t] = allMask;
        return;
    }

    float v;
    float vals[32];

    // 3. anchor picks
    for (uint j = 0; j < 32; ++j) vals[j] = initCrit[j] + distAdj[j];
    const uint i0 = lowestArgMax(vals, allMask, v);
    uint mask = 1u << i0;

    for (uint j = 0; j < 32; ++j) vals[j] = length(point[i0] - point[j]) + distAdj[j];
    const uint i1 = lowestArgMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;

    const float3 dir = cross(normal, point[i1] - point[i0]);
    float dval[32];
    for (uint j = 0; j < 32; ++j) dval[j] = dot(dir, point[j] - point[i0]);

    for (uint j = 0; j < 32; ++j) vals[j] = dval[j] + distAdj[j];
    const uint f = lowestArgMax(vals, allMask & ~mask, v);
    mask |= 1u << f;

    for (uint j = 0; j < 32; ++j) vals[j] = dval[j] - distAdj[j];
    const uint g = lowestArgMin(vals, allMask & ~mask, v);
    mask |= 1u << g;

    // 4. one extra deepest for PCM
    if (keepDeepest && countbits(mask) == 4 && maxPoints > 4)
    {
        const uint i4 = lowestArgMin(sep, allMask & ~mask, v);
        mask |= 1u << i4;
    }

    // 5. clustering / fill
    if (clustering)
    {
        float sepB[32];
        for (uint j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);

        // label each lane with its nearest anchor (anchors in bit order; first wins)
        int label[32];
        int nbClusters = 0;
        {
            uint m = mask;
            float tbest[32];
            for (uint j = 0; j < 32; ++j) { tbest[j] = FLT_MAX_F; label[j] = -1; }
            for (; m != 0; ++nbClusters, m &= (m - 1))
            {
                const uint anchor = firstbitlow(m);
                for (uint j = 0; j < 32; ++j)
                {
                    const float3 dv = point[j] - point[anchor];
                    const float d = dot(dv, dv);
                    if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
                }
            }
        }

        mask = 0;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint cluster = 0;
            for (uint j = 0; j < 32; ++j)
                if (label[j] == i) cluster |= 1u << j;
            cluster &= allMask;
            if (cluster != 0)
                mask |= 1u << lowestArgMin(sepB, cluster, v);
        }

        for (int i = nbClusters; i < maxPoints; ++i)
            mask |= 1u << lowestArgMin(sepB, allMask & ~mask, v);
    }
    else
    {
        const uint count = countbits(mask);
        for (uint i = count; i < uint(maxPoints); ++i)
            mask |= 1u << lowestArgMin(sep, allMask & ~mask, v);
    }

    gOut[t] = mask;
}
