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

// npSdfBvhMorton.slang
//
// Slang port of the SDF-BVH build front-end (SDFConstruction.cu / bvh.cuh):
// mode 0 -- bvhComputeTriangleBounds (SDFConstruction.cu L150-172):
//   bounds = (a, a); include(b); include(c)   // per-component min/max folds
//   fattenFast(margin)                        // per-component -= / +=
//   -> itemLower.xyz, itemUpper.xyz (w = 0)
// mode 1 -- bvhCalculateMortonCodes (L53-82) with part1by2/morton3 (bvh.cuh
// L40-59) and getCenter (SDFConstruction.cu L44-51):
//   center = 0.5f * (lower + upper)
//   local  = (center - gridLower).multiply(gridInvEdges)  // per-component
//   morton3<Dim>: u_i = PxClamp(PxI32(coord * Dim), 0, Dim - 1)
//                 (float->int TRUNCATION toward zero; int-select clamp)
//   key = (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux)
//   with priority: Dim = 512 and key |= (~priority) << 27
//   without:       Dim = 1024
//   part1by2: (n ^ (n<<16)) & 0xff0000ff; (n ^ (n<<8)) & 0x0300f00f;
//             (n ^ (n<<4)) & 0x030c30c3; (n ^ (n<<2)) & 0x09249249
//
// The gather indirections (vertices/triangleIndices, itemLowers[index])
// collapse to direct inputs.  Test inputs keep coord*Dim within int32 range
// (float->int overflow saturation differs across backends and is out of the
// exercised domain).  Float ops NoContraction-stamped; +-0 eqFZ; keys exact.

uint part1by2(uint n)
{
    n = (n ^ (n << 16)) & 0xff0000ffu;
    n = (n ^ (n << 8)) & 0x0300f00fu;
    n = (n ^ (n << 4)) & 0x030c30c3u;
    n = (n ^ (n << 2)) & 0x09249249u;
    return n;
}

int iselMin(int a, int b) { return a < b ? a : b; }
int iselMax(int a, int b) { return a < b ? b : a; }

// morton3<Dim>: PxClamp(PxI32(x*Dim), 0, Dim-1) per axis, then interleave.
uint morton3(float x, float y, float z, int dim)
{
    const uint ux = uint(iselMin(dim - 1, iselMax(0, int(x * float(dim)))));
    const uint uy = uint(iselMin(dim - 1, iselMax(0, int(y * float(dim)))));
    const uint uz = uint(iselMin(dim - 1, iselMax(0, int(z * float(dim)))));
    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

struct BmIn
{
    float4 misc;   // margin, asfloat(mode), asfloat(priority i32), asfloat(hasPriority)
    float4 va;     // mode 0: vertex a | mode 1: itemLower
    float4 vb;     // mode 0: vertex b | mode 1: itemUpper
    float4 vc;     // mode 0: vertex c | mode 1: gridLower
    float4 vd;     // mode 1: gridInvEdges
}

struct BmOut
{
    float4 lower;  // mode 0 bounds lower (w 0)
    float4 upper;  // mode 0 bounds upper (w 0)
    float4 aux;    // asfloat(key u32), _, _, _
}

struct BmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BmParams> gParams;

RWStructuredBuffer<BmIn> gIn;    // binding 0
RWStructuredBuffer<BmOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void sdfBvhMorton(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const BmIn s = gIn[tid];

    const uint mode = asuint(s.misc.y);

    BmOut o;
    o.lower = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.upper = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.aux = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (mode == 0u)
    {
        // bvhComputeTriangleBounds
        const float3 a = s.va.xyz, b = s.vb.xyz, c = s.vc.xyz;
        const float margin = s.misc.x;

        // PxBounds3(a, a); include(b); include(c): per-component min/max folds
        float3 lo = a, hi = a;
        lo = float3(min(lo.x, b.x), min(lo.y, b.y), min(lo.z, b.z));
        hi = float3(max(hi.x, b.x), max(hi.y, b.y), max(hi.z, b.z));
        lo = float3(min(lo.x, c.x), min(lo.y, c.y), min(lo.z, c.z));
        hi = float3(max(hi.x, c.x), max(hi.y, c.y), max(hi.z, c.z));

        // fattenFast
        lo = float3(lo.x - margin, lo.y - margin, lo.z - margin);
        hi = float3(hi.x + margin, hi.y + margin, hi.z + margin);

        o.lower = float4(lo, 0.0f);
        o.upper = float4(hi, 0.0f);
    }
    else
    {
        // bvhCalculateMortonCodes
        const float3 lower = s.va.xyz;
        const float3 upper = s.vb.xyz;
        const float3 gridLower = s.vc.xyz;
        const float3 gridInvEdges = s.vd.xyz;
        const int priority = asint(s.misc.z);
        const bool hasPriority = asuint(s.misc.w) == 1u;

        const float3 center = 0.5f * (lower + upper);
        const float3 diff = center - gridLower;
        const float3 local = float3(diff.x * gridInvEdges.x, diff.y * gridInvEdges.y, diff.z * gridInvEdges.z);

        int key;
        if (hasPriority)
        {
            key = int(morton3(local.x, local.y, local.z, 512));
            key |= (~priority) << 27;
        }
        else
        {
            key = int(morton3(local.x, local.y, local.z, 1024));
        }

        o.aux = float4(asfloat(uint(key)), 0.0f, 0.0f, 0.0f);
    }
    gOut[tid] = o;
}
