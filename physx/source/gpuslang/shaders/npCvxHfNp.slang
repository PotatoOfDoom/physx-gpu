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

// npCvxHfNp.slang
//
// Slang port of sphereHeightfieldNarrowphaseCore (convexHeightfield.cu
// :359-562) plus the warp output stage of sphereHeightfieldNarrowphase
// (:941-994) — tranche 107.  The heightfield twin of the tranche-106
// sphere-trimesh core; one thread serializes one 32-pair warp:
//
// per lane (pair):
//  - getTriangle (tranche-60 heightfield extraction): triVertexIndices /
//    triAdjacencyIndices with the PxMeshScale handedness swap (negative
//    row/column scale flips winding, swapping verts 1/2 and adjacency
//    0/2) + getVertex + hf2shapep — the triangle geometry comes ENTIRELY
//    from the shared sample grid + scale + triangle index;
//  - triNormal = getNormalized(crossX(v1-v0, v2-v0)); sphere center in
//    mesh space = trimeshTransform.transformInv(sphereCenter); the
//    one-sided gate dist = dotX(center-v0, triNormal) >= 0;
//  - distancePointTriangleSquared (tranche-54) — its edge-convexity
//    isEdgeNonconvex is a NO-OP here because the heightfield adjacency
//    are RAW triangle indices with no NONCONVEX_FLAG (bit 31), so it
//    always yields generateContact for vertex/edge and faceContact for
//    face; the edge convexity is instead decided by the SWITCH below;
//  - EDGE-CONCAVITY SWITCH (the heightfield speciality): for an eE01 /
//    eE12 / eE02 closest feature the ADJACENT triangle is re-extracted
//    and generateContact is regated by whether the current triangle's
//    opposite vertex lies BELOW the adjacent face plane
//    (triNormalAdj . (vk - v0Adj) < 0) — a concave edge is skipped;
//  - accept `dist>=0 && sqInflated > sqDist && generateContact`;
//    patchNormal = triNormal for a face contact else the normalized
//    centre-to-closest direction; the 0.996 deferral writes outMask =
//    gti | mask (NO edge/face split — HF stores only the raw mask);
//    nbOutputContacts = 1, separation = sqrt(sqDist), contact (0,0,0);
//  - maxDepth, intermediate gpuTriIndex, and the ordered index with
//    ONLY the eFACE arm `(nb && !delay) ? (0x80000000|tri) : tri` (the
//    HF core has NO eEDGE_CONTACT arm, unlike the trimesh core) at the
//    PADDED offset; SphereTriNormalAndIndex.normal = rotateInv(sphereQ,
//    rotate(meshQ, -patchNormal)), .index = deferredMask + (nb<<27) +
//    RAW gpu triangle index (heightfield never remaps);
//
// warp output stage (identical to the trimesh core, tranche 106):
// gated on totalContacts != 0, one second-pass reservation (popc of the
// outMask ballot) and one temp-contact reservation, per-lane exclusive
// scans, ConvexTriContacts.index linkage, the FAITHFUL totalContacts==0
// batch-drop quirk.
//
// The heightfield sample grid is shared (samples carry height | tess
// flag); each case carries the transforms, per-axis scales, radius and
// its triangle index.  Engine glue not ported (documented):
// LoadShapePair + shape-type dispatch, transform caches, midphaseAllocate
// stack partitioning, the pair-stack iteration.
//
// Determinism: integer heights and power-of-two scales make hf2shapep
// exact; shape-space sphere placement keeps every patch normal N or an
// exact axis-aligned unit vector, so the NI normal is bit-exact.  The
// only rounding ops are sqrt / division (IEEE, both sides) — the face
// closest point uses reciprocal-THEN-multiply (matching PxVec3::operator/)
// so it is identical to the CPU reference on a general denominator too.

import pxgSlangCommon;

static const float FLT_MAX_LOCAL = 3.402823466e38f;
static const uint BOUNDARY = 0xffffffffu;
static const uint NONCONVEX_FLAG = 0x80000000u;
static const uint eV0 = 1u << 29;
static const uint eV1 = 2u << 29;
static const uint eV2 = 3u << 29;
static const uint eE01 = 4u << 29;
static const uint eE02 = 5u << 29;
static const uint eE12 = 6u << 29;
static const uint DEFERRED_CONTACT_MASK = 0x80000000u;
static const uint NB_CONTACTS_SHIFT = 27u;
static const uint FACE_CONTACT_FLAG = 0x80000000u;

struct ChParams
{
    uint numCases;
    uint rows;
    uint columns;
    uint pad0;
}

struct ChCase
{
    float4 aQ;       // sphere transform rotation
    float4 aP;       // sphere position; w = contactDist
    float4 mQ;       // heightfield transform rotation
    float4 mP;       // heightfield position
    float4 scale;    // hf scale (rowScale, heightScale, columnScale); w = sphere radius
    uint4 pair;      // x = cmIdx, y = triangleIdx, z = testOffset, w = globalThreadIndex
}

[[vk::push_constant]]
ConstantBuffer<ChParams> gParams;

RWStructuredBuffer<ChCase> gCase;         // binding 0
RWStructuredBuffer<uint> gSamples;        // binding 1: shared heightfield samples
RWStructuredBuffer<uint4> gPairRec;       // binding 2: per cmIdx (startIndex, roundedStartIndex, 0, 0)
RWStructuredBuffer<float4> gNI;           // binding 3: per pair (normal, asfloat(index))
RWStructuredBuffer<float> gMaxDepth;      // binding 4: per pair
RWStructuredBuffer<uint> gInterm;         // binding 5: per pair (gpuTriIndex)
RWStructuredBuffer<uint> gOrdered;        // binding 6: per PADDED pair
RWStructuredBuffer<uint> gSecondPass;     // binding 7: outMask appends
RWStructuredBuffer<float4> gTempContacts; // binding 8: (contact, sep) appends
RWStructuredBuffer<uint> gContactsIdx;    // binding 9: per pair (temp-contact start)
RWStructuredBuffer<uint> gCounters;       // binding 10: [0] nbSecondPassPairs, [1] tempContactIndex
RWStructuredBuffer<float4> gDebug;        // binding 11: per case (nb, separation, delay, gpuTri)

// ---- heightfield extraction (npHeightfield.slang, tranche 58-60) -------------
float getHeight(uint vertexIndex) { return float(int(gSamples[vertexIndex] << 16) >> 16); }
bool isZerothVertexShared(uint cell) { return (gSamples[cell] & 0x800000u) != 0u; }
float3 getVertex(uint vertexIndex, uint columns)
{
    return float3(float(vertexIndex / columns), getHeight(vertexIndex), float(vertexIndex % columns));
}
float3 hf2shapep(float3 v, float rs, float hs, float cs) { return float3(v.x * rs, v.y * hs, v.z * cs); }

void triVertexIndices(uint triangleIndex, uint columns, out uint i0, out uint i1, out uint i2)
{
    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;
    if (isZerothVertexShared(cell))
    {
        if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1u; }
        else          { i0 = cell + 1u; i1 = cell + columns + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + columns; }
        else          { i0 = cell + columns + 1u; i1 = cell + columns; i2 = cell + 1u; }
    }
}
void triAdjacencyIndices(uint triangleIndex, uint columns, uint rows, out uint a0, out uint a1, out uint a2)
{
    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;
    if (isZerothVertexShared(cell))
    {
        if (firstTri)
        {
            a0 = BOUNDARY; a1 = triangleIndex + 1u; a2 = BOUNDARY;
            if (cell % columns != 0u)          a0 = triangleIndex - 1u;
            if (cell / columns != rows - 2u)   a2 = ((cell + columns) * 2u) + 1u;
        }
        else
        {
            a0 = BOUNDARY; a1 = triangleIndex - 1u; a2 = BOUNDARY;
            if (cell % columns < columns - 2u) a0 = triangleIndex + 1u;
            if (cell >= columns - 1u)          a2 = (cell - columns) * 2u;
        }
    }
    else
    {
        if (firstTri)
        {
            a0 = BOUNDARY; a1 = triangleIndex + 1u; a2 = BOUNDARY;
            if (cell >= columns - 1u)          a0 = ((cell - columns) * 2u) + 1u;
            if (cell % columns != 0u)          a2 = triangleIndex - 1u;
        }
        else
        {
            a0 = BOUNDARY; a1 = triangleIndex - 1u; a2 = BOUNDARY;
            if (cell / columns != rows - 2u)   a0 = (cell + columns) * 2u;
            if (cell % columns < columns - 2u) a2 = triangleIndex + 1u;
        }
    }
}

// getTriangle with handedness (convexHeightfield.cu :83-116)
void getTriangleHf(uint triangleIdx, float3 scale, out float3 v0, out float3 v1, out float3 v2,
                   out uint4 adj)
{
    const float rowScale = scale.x, heightScale = scale.y, columnScale = scale.z;
    bool wrongHanded = false;
    if (columnScale < 0.0f) wrongHanded = !wrongHanded;
    if (rowScale < 0.0f)    wrongHanded = !wrongHanded;

    uint t0, t1, t2;
    triVertexIndices(triangleIdx, gParams.columns, t0, t1, t2);
    uint a0, a1, a2;
    triAdjacencyIndices(triangleIdx, gParams.columns, gParams.rows, a0, a1, a2);
    adj = uint4(wrongHanded ? a2 : a0, a1, wrongHanded ? a0 : a2, 0u);

    v0 = hf2shapep(getVertex(t0, gParams.columns), rowScale, heightScale, columnScale);
    v1 = hf2shapep(getVertex(wrongHanded ? t2 : t1, gParams.columns), rowScale, heightScale, columnScale);
    v2 = hf2shapep(getVertex(wrongHanded ? t1 : t2, gParams.columns), rowScale, heightScale, columnScale);
}

float3 adjTriNormal(uint adjTri, float3 scale)
{
    float3 a0, a1, a2;
    uint4 dummy;
    getTriangleHf(adjTri, scale, a0, a1, a2, dummy);
    return getNormalizedZ(crossX(a1 - a0, a2 - a0));
}
float3 adjTriV0(uint adjTri, float3 scale)
{
    float3 a0, a1, a2;
    uint4 dummy;
    getTriangleHf(adjTri, scale, a0, a1, a2, dummy);
    return a0;
}

// ---- primitives (tranche 54) -------------------------------------------------
// explicit dot/cross (no dotX()/crossX() intrinsic) so the FMA contraction
// matches the CPU reference bit-for-bit on the exact integer inputs
float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossX(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float3 getNormalizedZ(float3 v)
{
    const float m = dotX(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
bool isEdgeNonconvex(uint adjEdgeIndex)
{
    return (adjEdgeIndex != BOUNDARY) && ((adjEdgeIndex & NONCONVEX_FLAG) != 0u);
}
float distancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, uint4 adj,
                                   out float3 closestP, out bool generateContact,
                                   out bool faceContact, out uint mask)
{
    faceContact = false;
    const float3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = dotX(ab, ap), d2 = dotX(ac, ap);
    const float3 bp = p - b;
    const float d3 = dotX(ab, bp), d4 = dotX(ac, bp);
    const float3 cp = p - c;
    const float d5 = dotX(ab, cp), d6 = dotX(ac, cp);
    const float vc = d1 * d4 - d3 * d2;
    const float vb = d5 * d2 - d1 * d6;
    const float va = d3 * d6 - d5 * d4;
    const float edgeTol = (va + vb + vc) * 1e-5f;
    if (d1 < 0.0f && d2 < 0.0f)
    {
        mask = eV0;
        generateContact = (!isEdgeNonconvex(adj.x)) || (!isEdgeNonconvex(adj.z));
        closestP = a; return dotX(ap, ap);
    }
    if (d3 > 0.0f && d4 < d3)
    {
        mask = eV1;
        generateContact = (!isEdgeNonconvex(adj.x)) || (!isEdgeNonconvex(adj.y));
        closestP = b; return dotX(bp, bp);
    }
    if (d6 > 0.0f && d5 < d6)
    {
        mask = eV2;
        generateContact = (!isEdgeNonconvex(adj.y)) || (!isEdgeNonconvex(adj.z));
        closestP = c; return dotX(cp, cp);
    }
    if (vc < 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        mask = eE01;
        generateContact = !isEdgeNonconvex(adj.x) || vc > -edgeTol;
        const float v = d1 / (d1 - d3);
        closestP = a + v * ab; return dotX(closestP - p, closestP - p);
    }
    if (vb < 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        mask = eE02;
        generateContact = !isEdgeNonconvex(adj.z) || vb > -edgeTol;
        const float w = d2 / (d2 - d6);
        closestP = a + w * ac; return dotX(closestP - p, closestP - p);
    }
    if (va < 0.0f && d4 >= d3 && d5 >= d6)
    {
        mask = eE12;
        generateContact = !isEdgeNonconvex(adj.y) || va > -edgeTol;
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closestP = b + w * (c - b); return dotX(closestP - p, closestP - p);
    }
    generateContact = true;
    faceContact = true;
    mask = 0u;
    // reciprocal-then-multiply to match PxVec3::operator/ (f = 1.0f/f; x*f,...)
    // exactly, NOT a per-component divide (they differ by up to 1 ULP for a
    // non-power-of-two denominator, and the CPU reference uses 1.0f/den)
    closestP = a + (ab * vb + ac * vc) * (1.0f / (va + vb + vc));
    return dotX(closestP - p, closestP - p);
}

// ---- warp output stage (convexHeightfield.cu :941-994, == trimesh) -----------
void warpOutputStage(uint nb[32], uint om[32], float3 cts[32], float seps[32],
                     uint pairOffsets[32], bool active[32])
{
    uint count = 0;
    uint total = 0;
    uint inclusive[32];
    for (uint l = 0; l < 32; ++l)
    {
        if (active[l] && om[l] != 0) count++;
        total += active[l] ? nb[l] : 0;
        inclusive[l] = total;
    }
    // FAITHFUL: a warp with no contacts drops its deferred outMask entries.
    // NOTE: in THIS core every accepted contact sets nbOutputContacts=1 even
    // when deferred, so om[l]!=0 implies nb[l]=1 implies total>0 — the guard
    // therefore only triggers when ALL 32 lanes are empty (count==0 too), and
    // removing it is a provable no-op (a KNOWN-EQUIVALENT mutant).  It is kept
    // to mirror the CUDA source exactly (the trimesh twin CAN observe it).
    if (total == 0)
        return;

    if (count != 0)
    {
        uint spStart = 0;
        InterlockedAdd(gCounters[0], count, spStart);
        uint ofs = 0;
        for (uint l = 0; l < 32; ++l)
        {
            if (!active[l] || om[l] == 0) continue;
            gSecondPass[spStart + ofs] = om[l];
            ofs++;
        }
    }

    uint tcStart = 0;
    InterlockedAdd(gCounters[1], total, tcStart);
    for (uint l = 0; l < 32; ++l)
    {
        if (!active[l]) continue;
        const uint base = tcStart + inclusive[l] - nb[l];
        for (uint i = 0; i < nb[l]; ++i)
            gTempContacts[base + i] = float4(cts[l], seps[l]);
        if (nb[l] != 0)
            gContactsIdx[pairOffsets[l]] = base;
    }
}

// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void chSphereTri(uint3 dtid : SV_DispatchThreadID)
{
    const uint base = dtid.x * 32;
    if (base >= gParams.numCases) return;

    uint laneNb[32];
    uint laneOm[32];
    float3 laneCts[32];
    float laneSeps[32];
    uint lanePo[32];
    bool laneActive[32];

    for (uint lane = 0; lane < 32; ++lane)
    {
        const uint ci = base + lane;
        laneNb[lane] = 0; laneOm[lane] = 0; lanePo[lane] = 0;
        laneCts[lane] = float3(0, 0, 0); laneSeps[lane] = 0;
        laneActive[lane] = ci < gParams.numCases;
        if (!laneActive[lane]) continue;
        const ChCase cs = gCase[ci];

        const uint triangleIdx = cs.pair.y;
        const uint testOffset = cs.pair.z;
        const uint gti = cs.pair.w;
        const uint4 rec = gPairRec[cs.pair.x];
        const uint pairOffset = rec.x + testOffset;
        const uint pairOffsetPadded = rec.y + testOffset;
        lanePo[lane] = pairOffset;

        const float contactDist = cs.aP.w;
        const float radius = cs.scale.w;
        const float3 scale = cs.scale.xyz;

        float3 v0, v1, v2;
        uint4 adj;
        getTriangleHf(triangleIdx, scale, v0, v1, v2, adj);
        const float3 triNormal = getNormalizedZ(crossX(v1 - v0, v2 - v0));

        const float3 center = quatRotateInv(cs.mQ, cs.aP.xyz - cs.mP.xyz);
        const float inflatedRadius = radius + contactDist;
        const float dist = dotX(center - v0, triNormal);

        float separation = FLT_MAX_LOCAL;
        uint nbOutputContacts = 0;
        bool delayContacts = false;
        float3 patchNormal = float3(0, 0, 0);

        float3 closestP;
        bool generateContact = false;
        bool faceContact = false;
        uint mask;
        const float sqDist = distancePointTriangleSquared(center, v0, v1, v2, adj,
                                                          closestP, generateContact, faceContact, mask);
        const float sqInflated = inflatedRadius * inflatedRadius;
        if (dist >= 0.0f && sqInflated > sqDist && generateContact)
        {
            // heightfield edge-concavity switch: re-gate an edge contact by
            // whether the current triangle's opposite vertex sits below the
            // adjacent face plane.  BOUNDARY guard (DELIBERATE resolution of
            // an upstream unguarded read: the CUDA switch calls getTriangle on
            // the adjacency index without checking 0xFFFFFFFF, reading past the
            // sample grid for a grid-edge contact) — a boundary edge has no
            // occluding neighbour, so the contact is kept
            if (mask == eE01 && adj.x != BOUNDARY)
            {
                const float3 nAdj = adjTriNormal(adj.x, scale);
                generateContact = dotX(nAdj, v2 - adjTriV0(adj.x, scale)) < 0.0f;
            }
            else if (mask == eE12 && adj.y != BOUNDARY)
            {
                const float3 nAdj = adjTriNormal(adj.y, scale);
                generateContact = dotX(nAdj, v0 - adjTriV0(adj.y, scale)) < 0.0f;
            }
            else if (mask == eE02 && adj.z != BOUNDARY)
            {
                const float3 nAdj = adjTriNormal(adj.z, scale);
                generateContact = dotX(nAdj, v1 - adjTriV0(adj.z, scale)) < 0.0f;
            }

            if (generateContact)
            {
                patchNormal = triNormal;
                if (!faceContact)
                    patchNormal = getNormalizedZ(center - closestP);

                const float cosTheta = dotX(patchNormal, triNormal);
                delayContacts = cosTheta <= 0.996f;
                if (delayContacts)
                    laneOm[lane] = gti | mask;

                nbOutputContacts = 1;
                separation = sqrt(sqDist);
                laneCts[lane] = float3(0, 0, 0); // sphere-space contact
                laneSeps[lane] = separation;
            }
        }

        gMaxDepth[pairOffset] = separation;
        gInterm[pairOffset] = triangleIdx;
        // ordered index: HF core has ONLY the eFACE arm (no eEDGE_CONTACT)
        gOrdered[pairOffsetPadded] = (nbOutputContacts != 0 && !delayContacts)
                                         ? (FACE_CONTACT_FLAG | triangleIdx) : triangleIdx;

        const uint delayContactMask = delayContacts ? DEFERRED_CONTACT_MASK : 0u;
        const float3 worldNormal = quatRotate(cs.mQ, -patchNormal);
        const float3 niNormal = quatRotateInv(cs.aQ, worldNormal);
        const uint niIndex = delayContactMask + (nbOutputContacts << NB_CONTACTS_SHIFT) + triangleIdx;
        gNI[pairOffset] = float4(niNormal, asfloat(niIndex));

        laneNb[lane] = nbOutputContacts;
        gDebug[ci] = float4(float(nbOutputContacts), separation,
                            delayContacts ? 1.0f : 0.0f, float(triangleIdx));
    }

    warpOutputStage(laneNb, laneOm, laneCts, laneSeps, lanePo, laneActive);
}
