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

// npPolyClip4.slang
//
// Slang port of polyClip4 and its helpers from
// gpunarrowphase/src/CUDA/femClothPrimitives.cu — the convex-polygon vs
// triangle clipper that produces the cloth convexTriangleContactGen contact
// candidates:
//
// - findRotationMatrixFromZ (femClothPrimitives.cu:988): the Moeller-Hughes
//   style rotation taking the separating axis to z, with the |z| > 0.9999
//   near-(anti)parallel double-reflection branch transcribed literally;
// - containsParallel (:1171): 2D point-in-polygon crossing-parity test —
//   per polygon edge the y-range gate (a segment with minY == maxY is
//   parallel to the ray and never intersects), the eps-shifted ray-side
//   comparison ((jx + PX_EPS_REAL) * jiy vs the branch-selected operand
//   order), and the odd-crossings containment rule;
// - signed2DTriArea + segmentIntersect (:1240-1284): 2D bbox reject then
//   two STRICT sign tests (a1*a2 < 0, a3*a4 < 0); intersection points
//   pAA = ipA - t0*(ipB - ipA) on the polygon edge and pBB on the triangle
//   edge with t0 = a1/(a2 - a1), t1 = a3/(a4 - a3);
// - addContacts (PxVec3 flavour, :1286): the lane-mask agreement AND-reduce
//   (equivalent to ballot(in-mask && not-rejected) since only in-mask lanes
//   clear their own bit), exclusive-scan slot assignment, writes capped at
//   NUM_TMP_CONTACTS_PER_PAIR = 32 while the counter accumulates UNCAPPED;
// - polyClip4 (:1393): rotate the polygon and triangle into the axis-z
//   frame; phase 1 (triangle verts inside the polygon) is compiled out
//   upstream (`if (0)`) and NOT ported; phase 2 collects polygon verts
//   whose 2D projection lies inside the triangle, projects them onto the
//   TRIANGLE plane (plane1.project), signs the distance by cosTheta
//   against the CONVEX face normal plane0.n (negative side: reject beyond
//   contactDist, dist = +sqrt, normal = -dir.getNormalized(); positive
//   side: dist = -sqrt (penetration), normal = +dir.getNormalized()),
//   contact point = the projection onto the triangle; phase 3 intersects
//   each triangle edge (rStart 0..2 with rEnd trailing) against every
//   polygon edge in 2D — the rMin/rMax z components use rpA twice
//   (upstream typo, harmless: only x/y feed the bbox test — transcribed
//   literally), back-rotates both witness points (transformTranspose),
//   applies the same cosTheta/contactDist rules with the contact point on
//   the TRIANGLE edge, and appends with the same capped-write/uncapped-
//   count scheme.
//
// The CUDA routine is warp-SIMD (one polygon vertex/edge per lane) but has
// no atomics and no cross-lane ordering effects — every ballot/shuffle/scan
// is a pure function of the case inputs — so this port serializes one CASE
// per thread with lane loops; results match the warp execution exactly up
// to ~1 ULP in the output normals (rsqrt vs CUDA's 1/sqrtf).  Contacts land in a 32-slot x 8-float transposed buffer
// (point.xyz, dist, normal.xyz, 0) mirroring s_contactsTransposed.

import pxgSlangCommon;

static const uint NUM_TMP_CONTACTS_PER_PAIR = 32;
static const float PX_EPS_REAL = 1.192092896e-07f;

struct PcParams { uint numCases; uint pad0; uint pad1; uint pad2; }

struct PcCase
{
    float4 axis;   // xyz = separating axis (the SAT minNormal)
    float4 plane0; // convex face plane (n, d)
    float4 plane1; // triangle plane (n, d)
    float4 t0;     // triangle verts
    float4 t1;
    float4 t2;
    float4 misc;   // x = contactDist
    uint4 counts;  // x = nbVerts0 (polygon vertex count, <= 32)
}

[[vk::push_constant]]
ConstantBuffer<PcParams> gParams;

RWStructuredBuffer<PcCase> gCase;    // binding 0
RWStructuredBuffer<float4> gPoly;    // binding 1: 32 verts per case
RWStructuredBuffer<uint> gNum;       // binding 2: returned numContacts (uncapped)
RWStructuredBuffer<float4> gCt;      // binding 3: 32 slots x 2 float4 per case

struct Mat33 { float3 col0, col1, col2; }

float3 m33Transform(Mat33 m, float3 v)
{
    return m.col0 * v.x + m.col1 * v.y + m.col2 * v.z;
}

float3 m33TransformTranspose(Mat33 m, float3 v)
{
    return float3(dot(m.col0, v), dot(m.col1, v), dot(m.col2, v));
}

float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// PxPlane::project: p - n * (n.dot(p) + d)
float3 planeProject(float4 plane, float3 p)
{
    return p - plane.xyz * (dot(plane.xyz, p) + plane.w);
}

Mat33 findRotationMatrixFromZ(float3 to)
{
    const float e = to.z;
    const float f = abs(e);

    Mat33 m;
    if (f <= 0.9999f)
    {
        // normal case: to is not (anti-)parallel to z
        const float vx = -to.y;
        const float vy = to.x;
        const float h = 1.0f / (1.0f + e);
        const float hvx = h * vx;
        const float hvxy = hvx * vy;

        m.col0 = float3(e + hvx * vx, hvxy, vy);
        m.col1 = float3(hvxy, e + h * vy * vy, -vx);
        m.col2 = float3(-vy, vx, e);
    }
    else
    {
        const float3 from = float3(0.0f, 0.0f, 1.0f);
        const float3 absFrom = float3(0.0f, 1.0f, 0.0f);

        const float3 u = absFrom - from;
        const float3 v = absFrom - to;

        const float c1 = -(2.0f / dot(u, u));
        const float c2 = -(2.0f / dot(v, v));
        const float c3 = c1 * c2 * dot(u, v);

        const float3 c1u = u * c1;
        const float3 c2v = v * c2;
        const float3 c3v = v * c3;

        float temp0 = c1u.x;
        float temp1 = c2v.x;
        float temp2 = c3v.x;
        float3 col0 = u * temp2 + v * temp1 + u * temp0;
        col0.x = col0.x + 1.0f;

        temp0 = c1u.y;
        temp1 = c2v.y;
        temp2 = c3v.y;
        float3 col1 = u * temp2 + v * temp1 + u * temp0;
        col1.y = col1.y + 1.0f;

        temp0 = c1u.z;
        temp1 = c2v.z;
        temp2 = c3v.z;
        float3 col2 = u * temp2 + v * temp1 + u * temp0;
        col2.z = col2.z + 1.0f;

        m.col0 = col0;
        m.col1 = col1;
        m.col2 = col2;
    }
    return m;
}

// containsParallel: crossing-parity containment of p in the 2D polygon
// held in verts[0..numVerts); lane loop serialized (ballot+popc parity)
bool containsParallel(float3 verts[32], uint numVerts, float3 p)
{
    const float tx = p.x;
    const float ty = p.y;
    const float eps = PX_EPS_REAL;

    uint numIntersects = 0;
    for (uint tI = 0; tI < numVerts; ++tI)
    {
        const float3 start = verts[tI];
        const float3 endP = verts[tI + 1 == numVerts ? 0 : tI + 1];

        const float jy = endP.y;
        const float iy = start.y;
        const float minY = min(jy, iy);
        const float maxY = max(jy, iy);

        // minY == maxY: segment parallel to the ray, no intersection
        if ((minY != maxY) && ty >= minY && ty <= maxY)
        {
            const float jx = endP.x;
            const float ix = start.x;
            const float jix = ix - jx;
            const float jiy = iy - jy;
            const float jty = ty - jy;
            const float part1 = jty * jix;
            const float part2 = (jx + eps) * jiy;
            const float part3 = tx * jiy;

            const bool comp = jiy > 0.0f;
            const float tmp = part1 + part2;
            const float comp1 = comp ? tmp : part3;
            const float comp2 = comp ? part3 : tmp;
            if (comp1 >= comp2)
                ++numIntersects;
        }
    }
    return (numIntersects % 2) == 1;
}

float signed2DTriArea(float3 a, float3 b, float3 c)
{
    const float3 ca = a - c;
    const float3 cb = b - c;
    return ca.x * cb.y - ca.y * cb.x;
}

// pAA: intersection on the polygon edge; pBB: on the triangle edge
bool segmentIntersect(float3 ipA, float3 ipB, float3 rpA, float3 rpB,
                      float3 rMin, float3 rMax, out float3 pAA, out float3 pBB)
{
    pAA = float3(0, 0, 0);
    pBB = float3(0, 0, 0);

    const float3 iMin = min(ipA, ipB);
    const float3 iMax = max(ipA, ipB);

    const bool xCon = iMin.x > rMax.x || rMin.x > iMax.x;
    const bool yCon = iMin.y > rMax.y || rMin.y > iMax.y;
    if (xCon || yCon)
        return false;

    const float a1 = signed2DTriArea(rpA, rpB, ipA);
    const float a2 = signed2DTriArea(rpA, rpB, ipB);
    if ((a1 * a2) < 0.0f)
    {
        const float a3 = signed2DTriArea(ipA, ipB, rpA);
        const float a4 = signed2DTriArea(ipA, ipB, rpB);
        if ((a3 * a4) < 0.0f)
        {
            const float t0 = a1 / (a2 - a1);
            pAA = ipA - t0 * (ipB - ipA);
            const float t1 = a3 / (a4 - a3);
            pBB = rpA - t1 * (rpB - rpA);
            return true;
        }
    }
    return false;
}

uint warpScanExclusive(uint mask, uint lane)
{
    return countbits(mask & ((1u << lane) - 1u));
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void ctPolyClip4(uint3 tid : SV_DispatchThreadID)
{
    const uint caseIdx = tid.x;
    if (caseIdx >= gParams.numCases) return;
    const PcCase cs = gCase[caseIdx];
    const uint nbVerts0 = cs.counts.x;
    const float contactDist = cs.misc.x;
    const float4 plane0 = cs.plane0;
    const float4 plane1 = cs.plane1;

    // lane-register state serialized into local arrays
    float3 v0[32];        // polygon verts (input space)
    float3 polyPoint[32]; // rotated 2D
    float3 triangleP[32]; // rotated 2D (lanes 0-2)
    float3 v1[3];
    for (uint l = 0; l < 32; ++l)
    {
        v0[l] = (l < nbVerts0) ? gPoly[caseIdx * 32 + l].xyz : float3(0, 0, 0);
        polyPoint[l] = float3(0, 0, 0);
        triangleP[l] = float3(0, 0, 0);
    }
    v1[0] = cs.t0.xyz;
    v1[1] = cs.t1.xyz;
    v1[2] = cs.t2.xyz;

    const Mat33 rot = findRotationMatrixFromZ(cs.axis.xyz);

    for (uint l = 0; l < 3; ++l)
        triangleP[l] = m33Transform(rot, v1[l]);
    for (uint l = 0; l < nbVerts0; ++l)
        polyPoint[l] = m33Transform(rot, v0[l]);

    const float sqContactDist = contactDist * contactDist;

    float ct[NUM_TMP_CONTACTS_PER_PAIR * 8];
    for (uint s = 0; s < NUM_TMP_CONTACTS_PER_PAIR * 8; ++s)
        ct[s] = -8.0f; // sentinel; real writes overwrite

    uint numContacts = 0;

    // NOTE: phase 1 of the CUDA (triangle verts vs polygon) is inside
    // `if (0)` upstream and is not ported.

    // ---- phase 2: polygon verts inside the 2D triangle ---------------------
    {
        uint mask = 0;
        for (uint j = 0; j < nbVerts0; ++j)
        {
            const float3 v = polyPoint[j];
            if (containsParallel(triangleP, 3, v))
                mask |= 1u << j;
        }

        float dist[32];
        float3 normal[32];
        float3 contactPoint[32];
        uint keepMask = 0; // = AND-reduce of the per-lane masks upstream
        for (uint lane = 0; lane < 32; ++lane)
        {
            dist[lane] = PXG_MAX_F32;
            normal[lane] = float3(0, 0, 0);
            contactPoint[lane] = float3(0, 0, 0);
            if ((mask & (1u << lane)) != 0)
            {
                const float3 tProjPoint = planeProject(plane1, v0[lane]);
                const float3 dir = v0[lane] - tProjPoint;
                const float cosTheta = dot(dir, plane0.xyz);
                const float sqDist = dot(dir, dir);
                // triangle on the positive side of the polygon face
                if (cosTheta < 0.0f)
                {
                    if (sqDist > sqContactDist)
                    {
                        // lane clears its own mask bit (drops from the AND)
                    }
                    else
                    {
                        dist[lane] = sqrt(sqDist);
                        normal[lane] = -getNormalized(dir);
                        contactPoint[lane] = tProjPoint;
                        keepMask |= 1u << lane;
                    }
                }
                else
                {
                    dist[lane] = -sqrt(sqDist);
                    normal[lane] = getNormalized(dir);
                    contactPoint[lane] = tProjPoint;
                    keepMask |= 1u << lane;
                }
            }
        }

        // addContacts (PxVec3 flavour): capped writes, uncapped count
        for (uint lane = 0; lane < 32; ++lane)
        {
            if ((keepMask & (1u << lane)) != 0)
            {
                const uint index = warpScanExclusive(keepMask, lane) + numContacts;
                if (index < NUM_TMP_CONTACTS_PER_PAIR)
                {
                    ct[8 * index + 0] = contactPoint[lane].x;
                    ct[8 * index + 1] = contactPoint[lane].y;
                    ct[8 * index + 2] = contactPoint[lane].z;
                    ct[8 * index + 3] = dist[lane];
                    ct[8 * index + 4] = normal[lane].x;
                    ct[8 * index + 5] = normal[lane].y;
                    ct[8 * index + 6] = normal[lane].z;
                    ct[8 * index + 7] = 0.0f;
                }
            }
        }
        numContacts += countbits(keepMask);
    }

    // ---- phase 3: triangle edges vs polygon edges in 2D ---------------------
    for (uint rStart = 0, rEnd = 2; rStart < 3; rEnd = rStart++)
    {
        const float3 rpA = triangleP[rStart];
        const float3 rpB = triangleP[rEnd];

        // faithful upstream quirk: the z components use rpA twice (harmless —
        // only x/y feed the segment bbox test)
        const float3 rMin = float3(min(rpA.x, rpB.x), min(rpA.y, rpB.y), min(rpA.z, rpA.z));
        const float3 rMax = float3(max(rpA.x, rpB.x), max(rpA.y, rpB.y), max(rpA.z, rpA.z));

        float dist[32];
        float3 normal[32];
        float3 contactPoint[32];
        uint mask = 0;
        for (uint lane = 0; lane < 32; ++lane)
        {
            dist[lane] = PXG_MAX_F32;
            normal[lane] = float3(0, 0, 0);
            contactPoint[lane] = float3(0, 0, 0);

            const float3 ipA = polyPoint[lane];
            // CUDA shuffles srcLane % 32: only lane 31 wraps (to lane 0);
            // lanes in [nbVerts0, 31) read zero-filled slots — either way the
            // value is discarded by the lane < nbVerts0 guard
            const float3 ipB = polyPoint[(lane + 1 == nbVerts0 ? 0 : lane + 1) & 31];
            float3 pAA, pBB;
            bool hasContacts = (lane < nbVerts0)
                ? segmentIntersect(ipA, ipB, rpA, rpB, rMin, rMax, pAA, pBB) : false;

            if (hasContacts)
            {
                const float3 pA = m33TransformTranspose(rot, pAA);
                contactPoint[lane] = m33TransformTranspose(rot, pBB);

                const float3 dir = pA - contactPoint[lane];
                const float cosTheta = dot(dir, plane0.xyz);
                const float sqDist = dot(dir, dir);

                if (cosTheta < 0.0f)
                {
                    if (sqDist > sqContactDist)
                        hasContacts = false;
                    else
                    {
                        dist[lane] = sqrt(sqDist);
                        normal[lane] = -getNormalized(dir);
                    }
                }
                else
                {
                    dist[lane] = -sqrt(sqDist);
                    normal[lane] = getNormalized(dir);
                }
            }

            if (hasContacts)
                mask |= 1u << lane;
        }

        for (uint lane = 0; lane < 32; ++lane)
        {
            if ((mask & (1u << lane)) != 0)
            {
                const uint index = warpScanExclusive(mask, lane) + numContacts;
                if (index < NUM_TMP_CONTACTS_PER_PAIR)
                {
                    ct[8 * index + 0] = contactPoint[lane].x;
                    ct[8 * index + 1] = contactPoint[lane].y;
                    ct[8 * index + 2] = contactPoint[lane].z;
                    ct[8 * index + 3] = dist[lane];
                    ct[8 * index + 4] = normal[lane].x;
                    ct[8 * index + 5] = normal[lane].y;
                    ct[8 * index + 6] = normal[lane].z;
                    ct[8 * index + 7] = 0.0f;
                }
            }
        }
        numContacts += countbits(mask);
    }

    gNum[caseIdx] = numContacts;
    for (uint s = 0; s < NUM_TMP_CONTACTS_PER_PAIR; ++s)
    {
        gCt[(caseIdx * NUM_TMP_CONTACTS_PER_PAIR + s) * 2 + 0] =
            float4(ct[8 * s + 0], ct[8 * s + 1], ct[8 * s + 2], ct[8 * s + 3]);
        gCt[(caseIdx * NUM_TMP_CONTACTS_PER_PAIR + s) * 2 + 1] =
            float4(ct[8 * s + 4], ct[8 * s + 5], ct[8 * s + 6], ct[8 * s + 7]);
    }
}
