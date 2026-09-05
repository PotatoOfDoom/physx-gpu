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

// npCvxMeshNp.slang
//
// Slang port of sphereTrimeshNarrowphaseCore (:219-397) and
// capsuleTrimeshNarrowphaseCore (:400-618) of convexMesh.cu, plus the
// warp output stage of sphereTrimeshNarrowphase (:919-970) — tranche
// 106.  One THREAD serializes one WARP of 32 sphere/capsule-triangle
// pairs (the CUDA runs one pair per thread and aggregates per warp):
//
// per lane (pair):
//  - getShapeSpaceVerts (MeshScaling: identity fast path, vertex2Shape,
//    winding flip v1<->v2 when scale determinant < 0) + adjacency x/z
//    swap under flipNormal;
//  - SPHERE: sphere center in mesh space, one-sided plane gate
//    dist0 >= 0, distancePointTriangleSquared (the tranche-54 Ericson
//    port with edge-convexity-gated generateContact / faceContact and
//    the Voronoi mask), accept sqInflatedRadius > sqDist &&
//    generateContact; patchNormal = triangle normal for face contacts
//    else normalized center-to-closest; DEFERRAL when
//    cosTheta <= 0.996 (~5 degrees): outMask = voronoiMask |
//    globalThreadIndex goes to the second-pass list, edge-ness = mask
//    bit 31; one contact (0,0,0 in sphere space, sep = sqrt(sqDist));
//    the SphereTriNormalAndIndex INDEX uses the REMAPPED cpu triangle
//    index;
//  - CAPSULE: capsule segment in mesh space (basis vector 0 *
//    halfHeight), one-sided gate dist >= 0 on the capsule CENTER,
//    distanceSegmentTriangleSquared (tranche-55 port), accept
//    sqInflatedRadius > sqDist; patchNormal = triangle normal when
//    sqDist < FLT_EPSILON else normalized segment-to-triangle
//    closest-point difference with the same 0.996 deferral (mask from
//    getTriangleActiveMaskFromNormal, tranche-68 port); contacts from
//    generateContacts (endpoint plane projections) + generateEEContacts
//    (segment-vs-triangle-edge, cap 2 total; tranche-56 ports),
//    transformed into CAPSULE space, separation = min over contacts;
//    FAITHFUL QUIRK: the SphereTriNormalAndIndex INDEX uses the RAW
//    gpu triangle index (no face remap — unlike the sphere core), and
//    the normal is rotated by meshToCapsule (not world->A);
//  - both write maxDepth = separation (FLT_MAX when no hit), the
//    intermediate gpuTriIndex, and the ordered triangle index at the
//    PADDED offset with TriangleSearchFlags: eFACE_CONTACT (0x80000000)
//    when contacts were generated WITHOUT deferral, eEDGE_CONTACT
//    (0xc0000000) when deferred AND the mask is an edge feature, plain
//    index otherwise; NormalAndIndex.index = deferredMask(bit 31) +
//    (nbContacts << 27) + triangle index;
//
// warp output stage (serialized ballots/scans):
//  - FAITHFUL QUIRK: everything is gated on totalContacts != 0 — if no
//    lane in the warp produced a CONTACT, the second-pass outMask
//    entries of the warp are DROPPED (a deferred pair with zero
//    contacts in an otherwise contact-less warp never reaches the
//    second pass);
//  - one atomic reservation each for the second-pass list (popc of
//    outMask ballot) and the temp contact stream (warp contact total);
//    per-lane offsets by exclusive scans; ConvexTriContacts.index =
//    the lane's temp-contact start (written only for lanes with
//    contacts).
//
// The pair record (SphereMeshPair aka ConvexMeshPair) is consumed via
// its startIndex / roundedStartIndex fields only (harness carries them
// as a uint4 per cm).  Engine glue not ported (documented):
// LoadShapePairWarp + shape-type swap dispatch, transform caches,
// readTriangleMesh blob walk (cases carry vert/tri pool offsets), the
// midphase pair stack iteration and the tempContactSizeBytes capacity
// handling of the FULL convex path (the sphere/capsule launcher passes
// the temp stream uncapped).  The #if 0 selectNormal variant of the
// capsule core (:534-564) is dead upstream and not ported.
//
// Determinism: all decisions are exact-arithmetic on the fixtures
// except sqrt/normalize/divisions, which are IEEE-identical between
// llvmpipe and the CPU reference (getNormalized = v * (1/sqrt(dot))
// with zero fallback — an explicit division, since OpInverseSqrt is
// not correctly rounded).  P9.47: getTriangleActiveMaskFromNormal used
// to keep the normalize() INTRINSIC on the argument that its 0.0174
// margins absorb the ULP difference.  That was a probability, not a
// proof — a margin makes a flipped comparison unlikely, not impossible,
// and Normalize of a degenerate (zero-area) edge is UNDEFINED where
// getNormalized() returns zero.  It now calls the same getNormalized()
// the rest of this file uses.

import pxgSlangCommon;

static const float FLT_MAX_LOCAL = 3.402823466e38f;
static const float FLT_EPSILON_LOCAL = 1.192092896e-07f;
static const uint BOUNDARY = 0xffffffffu;
static const uint NONCONVEX_FLAG = 0x80000000u;
// ConvexTriIntermediateData::VoronoiState
static const uint eV0 = 1u << 29;
static const uint eV1 = 2u << 29;
static const uint eV2 = 3u << 29;
static const uint eE01 = 4u << 29;
static const uint eE02 = 5u << 29;
static const uint eE12 = 6u << 29;
// ConvexTriNormalAndIndex
static const uint DEFERRED_CONTACT_MASK = 0x80000000u;
static const uint NB_CONTACTS_SHIFT = 27u;
// TriangleSearchFlags
static const uint FACE_CONTACT_FLAG = 0x80000000u;
static const uint EDGE_CONTACT_FLAG = 0xc0000000u;

struct CmParams
{
    uint numCases;
    uint pad0, pad1, pad2;
}

struct CmCase
{
    float4 aQ;       // sphere/capsule transform rotation
    float4 aP;       // sphere/capsule position; w = contactDist
    float4 mQ;       // mesh transform rotation
    float4 mP;       // mesh position; w = capsule half height
    float4 scale;    // mesh scale (xyz); w = sphere radius / capsule radius
    float4 scaleRot; // mesh scale rotation quat
    uint4 pair;      // x = cmIdx, y = triangleIdx, z = testOffset, w = globalThreadIndex
    uint4 mesh;      // x = vert pool offset, y = tri pool offset (tris/adjacency/remap)
}

[[vk::push_constant]]
ConstantBuffer<CmParams> gParams;

RWStructuredBuffer<CmCase> gCase;        // binding 0
RWStructuredBuffer<float4> gVerts;       // binding 1: mesh vertices (vertex space)
RWStructuredBuffer<uint4> gTris;         // binding 2: triangle vertex indices
RWStructuredBuffer<uint4> gAdj;          // binding 3: triangle adjacency (x=e01, y=e12, z=e02)
RWStructuredBuffer<uint> gRemap;         // binding 4: gpu->cpu face remap
RWStructuredBuffer<uint4> gPairRec;      // binding 5: per cmIdx (startIndex, roundedStartIndex, 0, 0)
RWStructuredBuffer<float4> gNI;          // binding 6: per pair offset (normal.xyz, asfloat(index))
RWStructuredBuffer<float> gMaxDepth;     // binding 7: per pair offset
RWStructuredBuffer<uint> gInterm;        // binding 8: per pair offset (gpuTriIndex)
RWStructuredBuffer<uint> gOrdered;       // binding 9: per PADDED pair offset
RWStructuredBuffer<uint> gSecondPass;    // binding 10: second-pass outMask appends
RWStructuredBuffer<float4> gTempContacts;// binding 11: (contact.xyz, sep) appends
RWStructuredBuffer<uint> gContactsIdx;   // binding 12: per pair offset (temp-contact start)
RWStructuredBuffer<uint> gCounters;      // binding 13: [0] nbSecondPassPairs, [1] tempContactIndex
RWStructuredBuffer<float4> gDebug;       // binding 14: per case (nb, separation, delay, edge)

// ---- shared helpers ---------------------------------------------------------
// PxVec3::getNormalized (zero fallback).  Deliberately spelled as an
// explicit 1/sqrt division: llvmpipe's OpInverseSqrt is not correctly
// rounded on all inputs (rare 1-ULP disagreements with the CPU
// reference), while OpSqrt and OpFDiv are IEEE-correct on both sides.
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
bool isEdgeNonconvex(uint adjEdgeIndex)
{
    return (adjEdgeIndex != BOUNDARY) && ((adjEdgeIndex & NONCONVEX_FLAG) != 0u);
}
bool isValidBary(float v, float w)
{
    const float eps = 1e-6f;
    const float zero = -eps;
    const float one = 1.0f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

// MeshScaling::getShapeSpaceVert + adjacency flip (convexMesh.cu :55-72)
void getShapeSpaceVerts(CmCase cs, uint triangleIdx, out float3 v0, out float3 v1, out float3 v2,
                        out uint4 adj, out bool flipNormal)
{
    const float3 sc = cs.scale.xyz;
    const bool idt = (sc.x == 1.0f && sc.y == 1.0f && sc.z == 1.0f);
    flipNormal = (sc.x * sc.y * sc.z) < 0.0f;
    const uint4 tri = gTris[cs.mesh.y + triangleIdx];
    float3 tv0 = gVerts[cs.mesh.x + tri.x].xyz;
    float3 tv1 = gVerts[cs.mesh.x + tri.y].xyz;
    float3 tv2 = gVerts[cs.mesh.x + tri.z].xyz;
    if (idt)
    {
        v0 = tv0; v1 = tv1; v2 = tv2;
    }
    else
    {
        tv0 = vertex2Shape(tv0, sc, cs.scaleRot);
        tv1 = vertex2Shape(tv1, sc, cs.scaleRot);
        tv2 = vertex2Shape(tv2, sc, cs.scaleRot);
        v0 = tv0;
        v1 = flipNormal ? tv2 : tv1;
        v2 = flipNormal ? tv1 : tv2;
    }
    adj = gAdj[cs.mesh.y + triangleIdx];
    if (flipNormal)
    {
        const uint tmp = adj.x;
        adj.x = adj.z;
        adj.z = tmp;
    }
}

// distancePointTriangleSquared (sphereTriangle.cuh, tranche-54 port)
float distancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, uint4 adj,
                                   out float3 closestP, out bool generateContact,
                                   out bool faceContact, out uint mask)
{
    faceContact = false;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    const float vc = d1 * d4 - d3 * d2;
    const float vb = d5 * d2 - d1 * d6;
    const float va = d3 * d6 - d5 * d4;
    // sphereTriangle.cuh:96 is `const float edgeTol = (va + vb + vc) * 1e-5;` --
    // the literal carries NO SUFFIX, so in C++ it is a DOUBLE.  The multiply
    // therefore happens in DOUBLE precision and only the assignment narrows to
    // float.  `1e-5f` computes the whole thing in float, which lands about 0.4 ULP
    // away: float(1e-5) is 0x3727c5ac = 9.999999747e-06, the double is
    // 0x3ee4f8b588e368f1.
    //
    // `double(1.0f) / double(100000.0f)` reproduces the C++ literal BIT FOR BIT
    // (both 0x3ee4f8b588e368f1, checked): 100000 is exact in float and in double
    // and the division is correctly rounded.  Same idiom, and the same reason, as
    // npSbExtractRotAPD.slang:125-126.
    //
    // This is the ONLY site in the whole GPU CUDA corpus where an unsuffixed
    // literal meets float arithmetic -- searched, not assumed.  Everywhere else an
    // unsuffixed literal is assigned straight to a float and narrows immediately,
    // which is why npSbExtractRotAPD.slang:121 can note "double literal -> same
    // float" and move on.
    const double D1E_5 = double(1.0f) / double(100000.0f);
    const float edgeTol = float(double(va + vb + vc) * D1E_5);

    if (d1 < 0.0f && d2 < 0.0f)
    {
        mask = eV0;
        generateContact = (!isEdgeNonconvex(adj.x)) || (!isEdgeNonconvex(adj.z));
        closestP = a;
        return dot(ap, ap);
    }
    if (d3 > 0.0f && d4 < d3)
    {
        mask = eV1;
        generateContact = (!isEdgeNonconvex(adj.x)) || (!isEdgeNonconvex(adj.y));
        closestP = b;
        return dot(bp, bp);
    }
    if (d6 > 0.0f && d5 < d6)
    {
        mask = eV2;
        generateContact = (!isEdgeNonconvex(adj.y)) || (!isEdgeNonconvex(adj.z));
        closestP = c;
        return dot(cp, cp);
    }
    if (vc < 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        mask = eE01;
        generateContact = !isEdgeNonconvex(adj.x) || vc > -edgeTol;
        const float v = d1 / (d1 - d3);
        closestP = a + v * ab;
        return dot(closestP - p, closestP - p);
    }
    if (vb < 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        mask = eE02;
        generateContact = !isEdgeNonconvex(adj.z) || vb > -edgeTol;
        const float w = d2 / (d2 - d6);
        closestP = a + w * ac;
        return dot(closestP - p, closestP - p);
    }
    if (va < 0.0f && d4 >= d3 && d5 >= d6)
    {
        mask = eE12;
        generateContact = !isEdgeNonconvex(adj.y) || va > -edgeTol;
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closestP = b + w * (c - b);
        return dot(closestP - p, closestP - p);
    }
    generateContact = true;
    faceContact = true;
    mask = 0u;
    // P9.53 WITHDRAWS P9.36's ASSOCIATION CHANGE -- see the twin note in
    // npCvxHfNp.slang.  This function ports sphereTriangle.cuh:41-140, whose face
    // region at :138 sums the vectors and scales once; GuDistancePointTriangle.h
    // is a different function that this translation unit does not even include.
    const float denom = 1.0f / (va + vb + vc);
    closestP = a + (ab * vb + ac * vc) * denom;
    return dot(closestP - p, closestP - p);
}

// distanceSegmentSegmentSquared closest params (tranche-55 port)
void segSeg(float3 p1, float3 d1, float3 p2, float3 d2, out float s, out float t)
{
    const float eps = 1e-6f;
    const float3 r = p1 - p2;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float b = dot(d1, d2);
    const float c = dot(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = dot(d2, r);
    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = clamp(temp / denom, 0.0f, 1.0f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = clamp(tTmp, 0.0f, 1.0f);
    const float comp = (b * t2 - c) * aRecip;
    s = clamp(comp, 0.0f, 1.0f);
    t = t2;
}

// distanceSegmentTriangleSquared (capsuleTriangle.cuh, tranche-55 port)
float distanceSegmentTriangleSquared(float3 p, float3 q, float3 a, float3 b, float3 c,
                                     out float outT, out float outU, out float outV)
{
    const float3 pq = q - p;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 bc = c - b;
    const float3 ap = p - a;
    const float3 aq = q - a;
    const float d00 = dot(ab, ab);
    const float d01 = dot(ab, ac);
    const float d11 = dot(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : FLT_MAX_LOCAL;
    const float3 n = getNormalized(cross(ab, ac));
    const float dist3 = dot(ap, n);
    const float sqDist3 = dist3 * dist3;
    const float dist4 = dot(aq, n);
    const float sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;

    if (dMul <= 0.0f)
    {
        const float nom = -dot(n, ap);
        const float denom = 1.0f / dot(n, pq);
        const float t0 = nom * denom;
        const float3 ip = p + pq * t0;
        const float3 v2 = ip - a;
        const float d20 = dot(v2, ab);
        const float d21 = dot(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom;
        const float w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (isValidBary(v00, w00))
        {
            outT = t0; outU = v00; outV = w00;
            return 0.0f;
        }
    }

    const float3 closestP31 = p - n * dist3;
    const float3 pV20 = closestP31 - a;
    const float pD20 = dot(pV20, ab);
    const float pD21 = dot(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = isValidBary(v0, w0);

    const float3 closestP41 = q - n * dist4;
    const float3 qV20 = closestP41 - a;
    const float qD20 = dot(qV20, ab);
    const float qD21 = dot(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = isValidBary(v1, w1);

    if (con0 && con1)
    {
        if (sqDist4 > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        else                   { outT = 1.0f; outU = v1; outV = w1; return sqDist4; }
    }

    float t00, t10, t01, t11, t02, t12;
    segSeg(p, pq, a, ab, t00, t10);
    segSeg(p, pq, b, bc, t01, t11);
    segSeg(p, pq, a, ac, t02, t12);
    const float3 cp00 = p + pq * t00; const float3 cp01 = a + ab * t10;
    float3 tv = cp00 - cp01; const float sqDist0 = dot(tv, tv);
    const float3 cp10 = p + pq * t01; const float3 cp11 = b + bc * t11;
    tv = cp10 - cp11; const float sqDist1 = dot(tv, tv);
    const float3 cp20 = p + pq * t02; const float3 cp21 = a + ac * t12;
    tv = cp20 - cp21; const float sqDist2 = dot(tv, tv);

    const float u10 = t10, v10 = 0.0f;
    const float u11 = 1.0f - t11, v11 = t11;
    const float u12 = 0.0f, v12 = t12;

    float sqDistPE, uEdge, vEdge, tSeg;
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0) { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; }
    else if (sqDist2 > sqDist1)                 { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; }
    else                                        { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; }

    if (con0)
    {
        if (sqDistPE > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    if (con1)
    {
        if (sqDistPE > sqDist4) { outT = 1.0f; outU = v1; outV = w1; return sqDist4; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    outT = tSeg; outU = uEdge; outV = vEdge;
    return sqDistPE;
}

// getTriangleActiveMaskFromNormal (convexTriangle.cuh:1600, tranche-68 port;
// P9.47 replaced its normalize() intrinsics with getNormalized(), here and in
// npTriMask.slang, which is what the engine actually calls)
uint getTriangleActiveMaskFromNormal(float3 v0, float3 v1, float3 v2, float3 n)
{
    // P9.47: convexTriangle.cuh:1602-1604 uses getNormalized() -- the very helper
    // this file already defines at :163 and calls eight times elsewhere.
    const float3 e01 = getNormalized(v1 - v0);
    const float3 e02 = getNormalized(v2 - v0);
    const float3 e12 = getNormalized(v2 - v1);
    const float d01 = dot(e01, n);
    const float d02 = dot(e02, n);
    const float d12 = dot(e12, n);
    const float3 tn = getNormalized(cross(e01, e02));   // :1608
    const float eps = 0.0174f; // around 89 degrees

    if (d01 < -eps && d02 < -eps) return eV0;
    if (d01 > eps && d12 < -eps)  return eV1;
    if (d02 > eps && d12 > eps)   return eV2;
    if (dot(getNormalized(cross(tn, e01)), n) < -eps) return eE01;   // :1620
    if (dot(getNormalized(cross(tn, e02)), n) > eps)  return eE02;   // :1623
    if (dot(getNormalized(cross(tn, e12)), n) < -eps) return eE12;   // :1626
    return 0u;
}

// generateContacts (capsuleTriangle.cuh, tranche-56 port), count into c/s arrays
uint generateContacts(float3 a, float3 b, float3 c, float3 planeNormal, float3 normal,
                      float3 p, float3 q, float inflatedRadius,
                      inout float3 cts[2], inout float seps[2])
{
    const float3 ab = b - a, ac = c - a, ap = p - a, aq = q - a;
    const float d00 = dot(ab, ab), d01 = dot(ab, ac), d11 = dot(ac, ac);
    const float tdenom = d00 * d11 - d01 * d01;
    const float bdenom = tdenom > 0.0f ? 1.0f / tdenom : FLT_MAX_LOCAL;

    const float inomp = dot(planeNormal, -ap);
    const float ideom = dot(planeNormal, normal);
    const float ipt = ideom > 0.0f ? -(inomp / ideom) : FLT_MAX_LOCAL;

    const float3 cp31 = p - normal * ipt;
    const float3 pV20 = cp31 - a;
    const float pD20 = dot(pV20, ab), pD21 = dot(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;
    const bool con0 = isValidBary(v0, w0);

    uint count = 0u;
    if (con0 && (inflatedRadius > ipt))
    {
        cts[count] = p;
        seps[count] = ipt;
        count++;
    }

    const float inomq = dot(planeNormal, -aq);
    const float iqt = ideom > 0.0f ? -(inomq / ideom) : FLT_MAX_LOCAL;
    const float3 cp41 = q - normal * iqt;
    const float3 qV20 = cp41 - a;
    const float qD20 = dot(qV20, ab), qD21 = dot(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;
    const bool con1 = isValidBary(v1, w1);

    if (con1 && (inflatedRadius > iqt))
    {
        cts[count] = q;
        seps[count] = iqt;
        count++;
    }
    return count;
}

// generateEE + generateEEContacts (capsuleTriangle.cuh, tranche-56 port)
void generateEE(float3 p, float3 q, float sqInflatedRadius, float3 normal, float3 a, float3 b,
                inout float3 cts[2], inout float seps[2], inout uint count)
{
    const float3 ab = b - a;
    const float3 n = cross(ab, normal);
    const float d = dot(n, a);
    const float np = dot(n, p);
    const float nq = dot(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    if (signP * signQ > 0.0f) return;
    const float3 pq = q - p;
    const float npq = dot(n, pq);
    if (npq == 0.0f) return;
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
        cts[count] = localA;
        seps[count] = signedDist;
        count++;
    }
}
void generateEEContacts(float3 a, float3 b, float3 c, float3 normal, float3 p, float3 q,
                        float sqInflatedRadius, inout float3 cts[2], inout float seps[2],
                        inout uint count)
{
    if (count < 2u) generateEE(p, q, sqInflatedRadius, normal, a, b, cts, seps, count);
    if (count < 2u) generateEE(p, q, sqInflatedRadius, normal, b, c, cts, seps, count);
    if (count < 2u) generateEE(p, q, sqInflatedRadius, normal, a, c, cts, seps, count);
}

// ---- warp output stage (convexMesh.cu :919-970), serialized ----------------
void warpOutputStage(uint nb[32], uint om[32], float3 cts[32][2], float seps[32][2],
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
    if (total == 0)
        return; // FAITHFUL: outMask entries are dropped with no warp contacts

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
            gTempContacts[base + i] = float4(cts[l][i], seps[l][i]);
        if (nb[l] != 0)
            gContactsIdx[pairOffsets[l]] = base;
    }
}

// ============================================================================
// cmSphereTri — sphereTrimeshNarrowphaseCore + warp output stage
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void cmSphereTri(uint3 dtid : SV_DispatchThreadID)
{
    const uint base = dtid.x * 32;
    if (base >= gParams.numCases) return;

    uint laneNb[32];
    uint laneOm[32];
    float3 laneCts[32][2];
    float laneSeps[32][2];
    uint lanePo[32];
    bool laneActive[32];

    for (uint lane = 0; lane < 32; ++lane)
    {
        const uint ci = base + lane;
        laneNb[lane] = 0; laneOm[lane] = 0; lanePo[lane] = 0;
        laneCts[lane][0] = float3(0, 0, 0); laneCts[lane][1] = float3(0, 0, 0);
        laneSeps[lane][0] = 0; laneSeps[lane][1] = 0;
        laneActive[lane] = ci < gParams.numCases;
        if (!laneActive[lane]) continue;
        const CmCase cs = gCase[ci];

        const uint cmIdx = cs.pair.x;
        const uint triangleIdx = cs.pair.y;
        const uint testOffset = cs.pair.z;
        const uint gti = cs.pair.w;
        const uint4 rec = gPairRec[cmIdx];
        const uint pairOffset = rec.x + testOffset;
        const uint pairOffsetPadded = rec.y + testOffset;
        lanePo[lane] = pairOffset;

        const float contactDist = cs.aP.w;
        const float sphereRadius = cs.scale.w;

        // sphere center in mesh space: PxTransform::transformInv(point)
        const float3 center = quatRotateInv(cs.mQ, cs.aP.xyz - cs.mP.xyz);

        float3 v0, v1, v2;
        uint4 adj;
        bool flip;
        getShapeSpaceVerts(cs, triangleIdx, v0, v1, v2, adj, flip);
        const float3 n = getNormalized(cross(v1 - v0, v2 - v0));
        const uint remapCpuTriangleIdx = gRemap[cs.mesh.y + triangleIdx];

        const float inflatedRadius = sphereRadius + contactDist;
        const float d = dot(v0, n);
        const float dist0 = dot(center, n) - d;

        float separation = FLT_MAX_LOCAL;
        uint nbOutputContacts = 0;
        bool delayContacts = false;
        bool edgeContact = false;
        float3 patchNormal = float3(0, 0, 0);

        if (dist0 >= 0.0f)
        {
            float3 closestP;
            bool generateContact = false;
            bool faceContact = false;
            uint mask;
            const float sqDist = distancePointTriangleSquared(center, v0, v1, v2, adj,
                                                              closestP, generateContact, faceContact, mask);
            const float sqInflatedSphereRadius = inflatedRadius * inflatedRadius;
            if (sqInflatedSphereRadius > sqDist && generateContact)
            {
                patchNormal = n;
                if (!faceContact)
                    patchNormal = getNormalized(center - closestP);

                const float cosTheta = dot(patchNormal, n);
                const float tolerance = 0.996f; // around 5 degrees
                delayContacts = cosTheta <= tolerance;
                if (delayContacts)
                {
                    laneOm[lane] = mask | gti;
                    edgeContact = (mask & 0x80000000u) != 0u; // isEdgeContact
                }
                nbOutputContacts = 1;
                separation = sqrt(sqDist);
                laneSeps[lane][0] = separation;
                laneCts[lane][0] = float3(0, 0, 0); // sphere-space center
            }
        }

        gMaxDepth[pairOffset] = separation;
        gInterm[pairOffset] = triangleIdx;

        uint orderedTriangleIdx = triangleIdx;
        if (nbOutputContacts != 0)
        {
            if (!delayContacts)
                orderedTriangleIdx |= FACE_CONTACT_FLAG;
            else if (edgeContact)
                orderedTriangleIdx |= EDGE_CONTACT_FLAG;
        }
        gOrdered[pairOffsetPadded] = orderedTriangleIdx;

        const uint delayContactMask = delayContacts ? DEFERRED_CONTACT_MASK : 0u;
        const float3 worldNormal = quatRotate(cs.mQ, -patchNormal);
        const float3 niNormal = quatRotateInv(cs.aQ, worldNormal);
        const uint niIndex = delayContactMask + (nbOutputContacts << NB_CONTACTS_SHIFT) + remapCpuTriangleIdx;
        gNI[pairOffset] = float4(niNormal, asfloat(niIndex));

        laneNb[lane] = nbOutputContacts;
        gDebug[ci] = float4(float(nbOutputContacts), separation,
                            delayContacts ? 1.0f : 0.0f, edgeContact ? 1.0f : 0.0f);
    }

    warpOutputStage(laneNb, laneOm, laneCts, laneSeps, lanePo, laneActive);
}

// ============================================================================
// cmCapsuleTri — capsuleTrimeshNarrowphaseCore + warp output stage
// ============================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void cmCapsuleTri(uint3 dtid : SV_DispatchThreadID)
{
    const uint base = dtid.x * 32;
    if (base >= gParams.numCases) return;

    uint laneNb[32];
    uint laneOm[32];
    float3 laneCts[32][2];
    float laneSeps[32][2];
    uint lanePo[32];
    bool laneActive[32];

    for (uint lane = 0; lane < 32; ++lane)
    {
        const uint ci = base + lane;
        laneNb[lane] = 0; laneOm[lane] = 0; lanePo[lane] = 0;
        laneCts[lane][0] = float3(0, 0, 0); laneCts[lane][1] = float3(0, 0, 0);
        laneSeps[lane][0] = 0; laneSeps[lane][1] = 0;
        laneActive[lane] = ci < gParams.numCases;
        if (!laneActive[lane]) continue;
        const CmCase cs = gCase[ci];

        const uint cmIdx = cs.pair.x;
        const uint triangleIdx = cs.pair.y;
        const uint testOffset = cs.pair.z;
        const uint gti = cs.pair.w;
        const uint4 rec = gPairRec[cmIdx];
        const uint pairOffset = rec.x + testOffset;
        const uint pairOffsetPadded = rec.y + testOffset;
        lanePo[lane] = pairOffset;

        const float contactDist = cs.aP.w;
        const float capsuleRadius = cs.scale.w;
        const float capsuleHalfHeight = cs.mP.w;

        // meshToCapsule = capsuleT.transformInv(meshT)
        const float4 mtcQ = quatMul(quatConjugate(cs.aQ), cs.mQ);
        const float3 mtcP = quatRotateInv(cs.aQ, cs.mP.xyz - cs.aP.xyz);

        // capsule segment in mesh space
        const float3 tmp = quatBasisVector0(cs.aQ) * capsuleHalfHeight;
        const float3 center = quatRotateInv(cs.mQ, cs.aP.xyz - cs.mP.xyz);
        const float3 dir = quatRotateInv(cs.mQ, tmp);
        const float3 p0 = center + dir;
        const float3 p1 = center - dir;

        float3 v0, v1, v2;
        uint4 adj;
        bool flip;
        getShapeSpaceVerts(cs, triangleIdx, v0, v1, v2, adj, flip);
        const float3 triNormal = getNormalized(cross(v1 - v0, v2 - v0));

        const float inflatedRadius = capsuleRadius + contactDist;
        const float dist = dot(center - v0, triNormal);

        float separation = FLT_MAX_LOCAL;
        uint nbOutputContacts = 0;
        float3 patchNormal = float3(0, 0, 0);
        bool delayContacts = false;
        bool edgeContact = false;
        float3 cts[2] = { float3(0, 0, 0), float3(0, 0, 0) };
        float seps[2] = { 0, 0 };

        if (dist >= 0.0f)
        {
            float t, u, v;
            const float sqDist = distanceSegmentTriangleSquared(p0, p1, v0, v1, v2, t, u, v);
            const float sqInflatedRadius = inflatedRadius * inflatedRadius;
            if (sqInflatedRadius > sqDist)
            {
                if (sqDist < FLT_EPSILON_LOCAL)
                {
                    // segment intersects the triangle
                    patchNormal = triNormal;
                }
                else
                {
                    const float3 pointOnSegment = p0 + (p1 - p0) * t;
                    const float3 pointOnTriangle = v0 * (1.0f - u - v) + v1 * u + v2 * v;
                    patchNormal = getNormalized(pointOnSegment - pointOnTriangle);
                    const float cosTheta = dot(patchNormal, triNormal);
                    const float tolerance = 0.996f; // around 5 degrees
                    delayContacts = cosTheta <= tolerance;
                }
                if (delayContacts)
                {
                    const uint mask = getTriangleActiveMaskFromNormal(v0, v1, v2, patchNormal);
                    laneOm[lane] = mask | gti;
                    edgeContact = (mask & 0x80000000u) != 0u; // isEdgeContact
                }

                nbOutputContacts = generateContacts(v0, v1, v2, triNormal, patchNormal, p0, p1,
                                                    inflatedRadius, cts, seps);
                generateEEContacts(v0, v1, v2, patchNormal, p0, p1, sqInflatedRadius,
                                   cts, seps, nbOutputContacts);

                for (uint i = 0; i < nbOutputContacts; ++i)
                {
                    // transform contact back to the capsule space
                    cts[i] = quatRotate(mtcQ, cts[i]) + mtcP;
                    separation = min(separation, seps[i]);
                }
            }
        }

        gInterm[pairOffset] = triangleIdx;

        uint orderedTriangleIdx = triangleIdx;
        if (nbOutputContacts != 0)
        {
            if (!delayContacts)
                orderedTriangleIdx |= FACE_CONTACT_FLAG;
            else if (edgeContact)
                orderedTriangleIdx |= EDGE_CONTACT_FLAG;
        }
        gOrdered[pairOffsetPadded] = orderedTriangleIdx;

        gMaxDepth[pairOffset] = separation;

        const uint delayContactMask = delayContacts ? DEFERRED_CONTACT_MASK : 0u;
        // FAITHFUL QUIRK: normal in meshToCapsule rotation, index uses the
        // RAW gpu triangle index (no face remap, unlike the sphere core)
        const float3 niNormal = quatRotate(mtcQ, -patchNormal);
        const uint niIndex = delayContactMask + (nbOutputContacts << NB_CONTACTS_SHIFT) + triangleIdx;
        gNI[pairOffset] = float4(niNormal, asfloat(niIndex));

        laneNb[lane] = nbOutputContacts;
        laneCts[lane][0] = cts[0]; laneCts[lane][1] = cts[1];
        laneSeps[lane][0] = seps[0]; laneSeps[lane][1] = seps[1];
        gDebug[ci] = float4(float(nbOutputContacts), separation,
                            delayContacts ? 1.0f : 0.0f, edgeContact ? 1.0f : 0.0f);
    }

    warpOutputStage(laneNb, laneOm, laneCts, laneSeps, lanePo, laneActive);
}
