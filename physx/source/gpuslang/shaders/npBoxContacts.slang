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

// npBoxContacts.slang
//
// Slang port of calculateContacts() from gpunarrowphase/src/CUDA/cudaBox.cu —
// the box-box clipping core that turns box B's incident polygon (4 points, one
// per lane, expressed in box A's reference-face local frame) into up to 3
// contact points per lane.  It combines three sources of contacts:
//   1. incident-polygon vertices that fall inside the reference face rectangle
//      and penetrate ("area" contacts);
//   2. reference-face corners that fall inside the incident polygon (the 2D
//      point-in-polygon `contains` test) lifted onto the incident plane;
//   3. incident-polygon edges clipped against the reference-face slab
//      (`intersectSegmentAABB`), emitting the entry/exit crossings.
//
// The CUDA runs 4 lanes per pair with two warp-wide early exits
// (__all_sync(pArea), __all_sync(pArea2)) and __shfl of the per-lane pen/area
// flags along each edge.  Here the 4-lane group lives inside a 256-thread block
// (64 groups); the shuffles/reductions become a 32-bit groupshared exchange, and
// the early exits become predication (`allArea`, `done2`) so every thread still
// reaches every block-wide barrier in uniform control flow.  The contacts are
// written in the CUDA's interleaved layout: contact `c` of lane L at group slot
// `4*c + L`.

static const float PXG_MAX_F32 = 3.4028235e38f;

struct ContactsIn
{
    float4 faceNormal; // incidentFaceNormalInNew (.xyz)
    float4 misc;       // (extentX_, extentY_, contactDist, unused)
}

struct BoxContactsParams { uint numGroups; }

[[vk::push_constant]]
ConstantBuffer<BoxContactsParams> gParams;

RWStructuredBuffer<ContactsIn> gIn;          // binding 0  one per group of 4
RWStructuredBuffer<float4>     gPts;          // binding 1  incident vertex, one per lane
RWStructuredBuffer<float4>     gOutContacts;  // binding 2  12 slots/group (4 lanes x 3), slot 4*c+L
RWStructuredBuffer<uint>       gOutCount;     // binding 3  contact count, one per lane

groupshared float3 gsPts[256];
groupshared float3 gsExch[256];
groupshared uint gsPen[256];
groupshared uint gsArea[256];
groupshared uint gsArea2[256];

// contains(): 2D point-in-polygon crossing test over 4 verts, matching
// cudaBox.cu contains() incl. the AABB early-out and endpoint/epsilon handling.
bool contains4(float2 v0, float2 v1, float2 v2, float2 v3, float2 p, float2 bmin, float2 bmax)
{
    if (p.x < bmin.x || p.x > bmax.x || p.y < bmin.y || p.y > bmax.y)
        return false;

    float2 V[4] = { v0, v1, v2, v3 };
    const float eps = 1e-6f;
    const float zero = 0.0f;
    uint intersectionPoints = 0u;
    bool inside = false;
    bool done = false;

    uint j = 3u;
    for (uint i = 0u; i < 4u && !done; ++i)
    {
        const float jx = V[j].x, jy = V[j].y;
        const float ix = V[i].x, iy = V[i].y;
        if ((p.x == jx && p.y == jy) || (p.x == ix && p.y == iy))
        {
            inside = true; done = true;
        }
        else
        {
            const bool yflag0 = jy > p.y;
            const bool yflag1 = iy > p.y;
            if (yflag0 != yflag1)
            {
                const float jix = ix - jx;
                const float jiy = iy - jy;
                const float jty = p.y - jy;
                const float part1 = jty * jix;
                const float part2 = (jx + eps) * jiy;
                const float part3 = p.x * jiy;
                const bool comp = jiy > zero;
                const float tmp = part1 + part2;
                const float comp1 = comp ? tmp : part3;
                const float comp2 = comp ? part3 : tmp;
                if (comp1 >= comp2)
                {
                    if (intersectionPoints == 1u) { inside = false; done = true; }
                    else intersectionPoints++;
                }
            }
            j = i;
        }
    }
    if (!done) inside = intersectionPoints > 0u;
    return inside;
}

// intersectSegmentAABB(): parametric segment-vs-AABB slab test, matching
// cudaBox.cu intersectSegmentAABB() (max = mx, min = mn).
bool segAABB(float3 p0, float3 d, float3 mx, float3 mn, out float tmin, out float tmax)
{
    const float eps = 1e-6f;
    const float3 absV = abs(d);
    float tminf = 0.0f;
    float tmaxf = 1.0f;
    tmin = 0.0f; tmax = 1.0f;

    const bool pX = eps > absV.x;
    const bool pY = eps > absV.y;
    const bool pZ = eps > absV.z;

    if ((pX && (p0.x > mx.x || p0.x < mn.x)) ||
        (pY && (p0.y > mx.y || p0.y < mn.y)) ||
        (pZ && (p0.z > mx.z || p0.z < mn.z)))
        return false;

    const float3 odd = float3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
    const float3 t1 = float3(pX ? 0.0f : (mn.x - p0.x) * odd.x,
                             pY ? 0.0f : (mn.y - p0.y) * odd.y,
                             pZ ? 0.0f : (mn.z - p0.z) * odd.z);
    const float3 t2 = float3(pX ? PXG_MAX_F32 : (mx.x - p0.x) * odd.x,
                             pY ? PXG_MAX_F32 : (mx.y - p0.y) * odd.y,
                             pZ ? PXG_MAX_F32 : (mx.z - p0.z) * odd.z);
    const float3 tt1 = min(t1, t2);
    const float3 tt2 = max(t1, t2);
    const float ft1 = max(tt1.x, max(tt1.y, tt1.z));
    const float ft2 = min(tt2.x, min(tt2.y, tt2.z));
    tminf = max(ft1, tminf);
    tmaxf = min(tmaxf, ft2);
    tmin = tminf; tmax = tmaxf;
    return !((tminf > tmaxf) || (tminf > 1.0f));
}

// ---------------------------------------------------------------------------
// calculateContacts: mirrors cudaBox.cu calculateContacts()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void boxCalculateContacts(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = dtid.x;
    const uint lt = gtid.x;
    const uint lane = lt & 3u;
    const uint gbase = lt & ~3u;
    const uint groupIdx = t >> 2;
    const uint numGroups = gParams.numGroups;
    const bool valid = groupIdx < numGroups;
    const uint gi = valid ? groupIdx : (numGroups > 0u ? numGroups - 1u : 0u);

    const ContactsIn cin = gIn[gi];
    const float3 fn = cin.faceNormal.xyz;
    const float extentX_ = cin.misc.x;
    const float extentY_ = cin.misc.y;
    const float contactDist = cin.misc.z;

    const float3 pt = valid ? gPts[t].xyz : float3(0.0f, 0.0f, 0.0f);

    // publish the 4 group points (persist through the whole routine)
    gsPts[lt] = pt;
    GroupMemoryBarrierWithGroupSync();

    // calculateBounds4: width-4 butterfly AABB of the 4 points
    float3 bmin = pt, bmax = pt;
    gsExch[lt] = pt;
    GroupMemoryBarrierWithGroupSync();
    const float3 e1v = gsExch[lt ^ 1u];
    GroupMemoryBarrierWithGroupSync();
    bmin = min(bmin, e1v); bmax = max(bmax, e1v);
    gsExch[lt] = bmin;
    GroupMemoryBarrierWithGroupSync();
    const float3 e2n = gsExch[lt ^ 2u];
    GroupMemoryBarrierWithGroupSync();
    bmin = min(bmin, e2n);
    gsExch[lt] = bmax;
    GroupMemoryBarrierWithGroupSync();
    const float3 e2x = gsExch[lt ^ 2u];
    GroupMemoryBarrierWithGroupSync();
    bmax = max(bmax, e2x);

    const float extentX = 1.00001f * extentX_;
    const float extentY = 1.00001f * extentY_;
    const float3 ext = float3(extentX, extentY, PXG_MAX_F32);
    const float3 negExt = float3(-extentX, -extentY, -contactDist - 1e-7f);

    const uint outBase = groupIdx * 12u;
    uint count = 0u;

    // ---- 1. projection / area contact -------------------------------------
    const float z = -pt.z;
    const bool pPen = contactDist > z;
    bool pArea = false;
    if (pPen)
    {
        if (extentX >= abs(pt.x) && extentY >= abs(pt.y))
        {
            pArea = true;
            if (valid) gOutContacts[outBase + lane] = float4(pt, z); // slot 4*0+lane
            count = 1u;
        }
    }
    gsPen[lt] = pPen ? 1u : 0u;
    gsArea[lt] = pArea ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    // __all_sync(pArea): all points of A in B -> done
    const bool allArea = (gsArea[gbase] & gsArea[gbase + 1u] & gsArea[gbase + 2u] & gsArea[gbase + 3u]) != 0u;

    // ---- 2. reference-face corner inside incident polygon -----------------
    bool pArea2 = false;
    if (!allArea)
    {
        const float denom = fn.z;
        const float qx = (lane == 0u || lane == 3u) ? -extentX : extentX;
        const float qy = (lane & 2u) != 0u ? -extentY : extentY;
        const float2 q0 = float2(qx, qy);
        const float3 P0 = gsPts[gbase], P1 = gsPts[gbase + 1u], P2 = gsPts[gbase + 2u], P3 = gsPts[gbase + 3u];
        if (contains4(P0.xy, P1.xy, P2.xy, P3.xy, q0, bmin.xy, bmax.xy))
        {
            const float3 diff = float3(P0.x - qx, P0.y - qy, P0.z); // pts[0] - q0 (q0.z = 0)
            const float nom = dot(fn, diff);
            const float tval = nom / denom;
            if (contactDist > (-tval))
            {
                pArea2 = true;
                if (valid) gOutContacts[outBase + 4u * count + lane] = float4(qx, qy, tval, -tval);
                count++;
            }
        }
    }
    gsArea2[lt] = pArea2 ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    // __all_sync(pArea2): all points of B inside A -> done
    const bool allArea2 = (gsArea2[gbase] & gsArea2[gbase + 1u] & gsArea2[gbase + 2u] & gsArea2[gbase + 3u]) != 0u;
    const bool done2 = allArea || allArea2;

    // ---- 3. edge clip against the reference-face slab ---------------------
    if (!done2)
    {
        const uint rStart = lane;
        const uint rEnd = (lane + 1u) & 3u;
        const float3 p0 = gsPts[gbase + rStart];
        const float3 p1 = gsPts[gbase + rEnd];
        const bool pPenStart = gsPen[gbase + rStart] != 0u;
        const bool pPenEnd = gsPen[gbase + rEnd] != 0u;
        const bool pAreaStart = gsArea[gbase + rStart] != 0u;
        const bool pAreaEnd = gsArea[gbase + rEnd] != 0u;

        if (pPenStart || pPenEnd)
        {
            const bool con0 = pPenStart && pAreaStart;
            const bool con1 = pPenEnd && pAreaEnd;
            if (!(con0 && con1))
            {
                const float3 p0p1 = p1 - p0;
                float tmn, tmx;
                if (segAABB(p0, p0p1, ext, negExt, tmn, tmx))
                {
                    if (!con0)
                    {
                        const float3 ip = p0 + p0p1 * tmn;
                        if (valid) gOutContacts[outBase + 4u * count + lane] = float4(ip, -ip.z);
                        count++;
                    }
                    if (!con1)
                    {
                        const float3 ip = p0 + p0p1 * tmx;
                        if (valid) gOutContacts[outBase + 4u * count + lane] = float4(ip, -ip.z);
                        count++;
                    }
                }
            }
        }
    }

    if (valid) gOutCount[t] = count;
}
