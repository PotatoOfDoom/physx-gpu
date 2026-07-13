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

// npCvxStage2Setup.slang
//
// Slang port of convexConvexNphase_stage2Kernel_core (cudaGJKEPA.cu
// L2390-2641) MINUS the collide() call itself — the drivers it invokes are the
// separately-verified gjk/epa/contact-gen shaders, so collide()'s two outputs
// (bIntersection, the post-collide numManifoldContacts) arrive injected.  Per
// pair:
//   - shape reads + flip = (type1 < type0), swapping the WHOLE shapes and the
//     transformCacheRefs (types re-derived post-swap);
//   - synthetic vertices + margins: eSPHERE -> one vertex (0,0,0,0),
//     expandedMargin = convexMargin0 = scale.x (the CUDA reads .x here, NOT the
//     .y used elsewhere — ported faithfully); eCAPSULE -> (±1,0,0,0),
//     expandedMargin = convexMargin0 = scale.y; else the blob's vertices with
//     expandedMargin = 0 and convexMargin0 = calculatePCMConvexMargin(extents *
//     scale, toleranceLength);
//   - scratch population: counts from the blob header word (getNbVerts =
//     u16High(u32Low), getNbPolygons = u16Low(u32Low), getNbEdges = u32High);
//     inSphereRadius = extents.w * minScale where BOTH minScale0 and minScale1
//     are gated on shape1.type == eBOX (the CUDA tests shape1 for both sides —
//     a quirk ported verbatim, L2542/L2569);
//   - contactDistance = cd[ref0] + cd[ref1] + expandedMargin (post-flip refs);
//     replaceBreakingThreshold = fminf(convexMargin0, convexMargin1) * 0.05;
//   - the stage-1 counts word (manifold moreData lane 12): refreshed count =
//     & 0xFF, initialNbContacts = >> 8;
//   - writeback: mNbContacts = numManifoldContacts | (bIntersection ? 0 :
//     0x80000000) (the invalidation bit consumed by the stream writer).  The
//     CUDA memory store covers words 9..12 only (the quatB tail is a no-op
//     rewrite of the values read earlier); runGJK updates the warm-start words
//     13..15 in LANE REGISTERS which stage2 never stores, so memory keeps
//     stage1's values -- modelled as pass-through.

static const uint EGT_SPHERE = 0u;
static const uint EGT_CAPSULE = 2u;
static const uint EGT_BOX = 3u;
static const float MESH_MANIFOLD_EPSILON = 0.05f;
static const float BOX_MARGIN_RATIO = 0.25f;

float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float4 quatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
                  a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
                  a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float calculatePCMConvexMargin(float3 extents, float toleranceLength)
{
    const float mn = min(min(extents.x, extents.y), extents.z);
    const float toleranceMargin = toleranceLength * MESH_MANIFOLD_EPSILON;
    return min(mn * BOX_MARGIN_RATIO, toleranceMargin);
}

struct Stage2In
{
    // shape i: meta_i = {type, materialIndex, blobHeaderWord (uint4.x), pad}
    uint4 meta0, meta1;
    float4 scale0, rot0;   // meshScale: scale (.xyz) + rotation quat
    float4 scale1, rot1;
    float4 extents0;       // blob extents (.xyz raw, .w = inSphere radius base)
    float4 extents1;
    float4 t0q, t0p;       // transformCache[input.ref0]; t0p.w = contactDistance[ref0]
    float4 t1q, t1p;       // transformCache[input.ref1]; t1p.w = contactDistance[ref1]
    uint4 misc;            // x = stage1 counts word (lane 12), y = injected bIntersection,
                           // z = injected post-collide numManifoldContacts, w = toleranceLength bits
}

struct Stage2Out
{
    float4 rotOut0, rotOut1; // post-flip mesh-scale rotations (scratch rot0/rot1)
    float4 tBq, tBp;         // scratch tB = post-flip transf1
    float4 aToBq;
    float4 aToBp;          // .w = contactDistance (with expandedMargin)
    float4 verts0[2];      // synthetic sphere/capsule vertices (zero for convex path)
    uint4 counts0;         // x=nbVertices0 y=nbPolygons0 z=nbEdges0 w=type0(post-flip)
    uint4 counts1;         // x=nbVertices1 y=nbPolygons1 z=nbEdges1 w=type1(post-flip)
    float4 params;         // x=inSphereRadius0 y=inSphereRadius1 z=replaceBreakingThreshold w=expandedMargin
    uint4 result;          // x=refreshedNbContacts y=initialNbContacts z=mNbContacts writeback word w=flip
    uint4 materials;       // x=shape0MaterialIndex y=shape1MaterialIndex (post-flip) z,w pad
}

struct Stage2Params { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<Stage2Params> gParams;

RWStructuredBuffer<Stage2In> gIn;    // binding 0
RWStructuredBuffer<Stage2Out> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxStage2Setup(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const Stage2In s = gIn[t];

    // shape structs (modelled): swap wholesale on flip
    uint type0 = s.meta0.x, type1 = s.meta1.x;
    uint hdr0 = s.meta0.z, hdr1 = s.meta1.z;
    uint mat0 = s.meta0.y, mat1 = s.meta1.y;
    float4 rot0 = s.rot0, rot1 = s.rot1;
    float3 scale0 = s.scale0.xyz, scale1 = s.scale1.xyz;
    float4 ext0 = s.extents0, ext1 = s.extents1;
    float4 t0q = s.t0q, t1q = s.t1q;
    float3 t0p = s.t0p.xyz, t1p = s.t1p.xyz;
    float cd0 = s.t0p.w, cd1 = s.t1p.w;
    const float toleranceLength = asfloat(s.misc.w);

    const bool flip = type1 < type0;
    if (flip)
    {
        uint tu = type0; type0 = type1; type1 = tu;
        tu = hdr0; hdr0 = hdr1; hdr1 = tu;
        tu = mat0; mat0 = mat1; mat1 = tu;
        float4 tr = rot0; rot0 = rot1; rot1 = tr;
        float3 tf = scale0; scale0 = scale1; scale1 = tf;
        float4 te = ext0; ext0 = ext1; ext1 = te;
        te = t0q; t0q = t1q; t1q = te;
        tf = t0p; t0p = t1p; t1p = tf;
        const float tc = cd0; cd0 = cd1; cd1 = tc;
    }

    // aToB = t1.transformInv(t0)
    const float4 aToBq = quatMul(float4(-t1q.x, -t1q.y, -t1q.z, t1q.w), t0q);
    const float3 aToBp = quatRotateInv(t1q, t0p - t1p);

    Stage2Out r;
    r.verts0[0] = float4(0, 0, 0, 0);
    r.verts0[1] = float4(0, 0, 0, 0);

    float expandedMargin, convexMargin0;
    uint nbV0, nbP0, nbE0;
    float inSphere0;
    if (type0 == EGT_SPHERE)
    {
        r.verts0[0] = float4(0, 0, 0, 0);
        expandedMargin = scale0.x; // CUDA reads .x here (not the .y used elsewhere)
        convexMargin0 = scale0.x;
        nbV0 = 1u; nbP0 = 0u; nbE0 = 0u; inSphere0 = 0.0f;
    }
    else if (type0 == EGT_CAPSULE)
    {
        r.verts0[0] = float4(1, 0, 0, 0);
        r.verts0[1] = float4(-1, 0, 0, 0);
        expandedMargin = scale0.y;
        convexMargin0 = scale0.y;
        nbV0 = 2u; nbP0 = 0u; nbE0 = 0u; inSphere0 = 0.0f;
    }
    else
    {
        expandedMargin = 0.0f;
        convexMargin0 = calculatePCMConvexMargin(
            float3(ext0.x * scale0.x, ext0.y * scale0.y, ext0.z * scale0.z), toleranceLength);
        nbV0 = (hdr0 & 0xFFFFu) >> 8u;  // getNbVerts = u16High(u32Low)
        nbP0 = hdr0 & 0xFFu;            // getNbPolygons = u16Low(u32Low)
        nbE0 = hdr0 >> 16u;             // getNbEdges = u32High
        // CUDA quirk: minScale0 is gated on SHAPE1's type (L2542)
        float minScale0 = 1.0f;
        if (type1 == EGT_BOX)
            minScale0 = min(scale0.z, min(scale0.x, scale0.y));
        inSphere0 = ext0.w * minScale0;
    }

    const uint nbV1 = (hdr1 & 0xFFFFu) >> 8u;
    const uint nbP1 = hdr1 & 0xFFu;
    const uint nbE1 = hdr1 >> 16u;
    float minScale1 = 1.0f;
    if (type1 == EGT_BOX)
        minScale1 = min(scale1.z, min(scale1.x, scale1.y));
    const float inSphere1 = ext1.w * minScale1;

    const float convexMargin1 = calculatePCMConvexMargin(
        float3(ext1.x * scale1.x, ext1.y * scale1.y, ext1.z * scale1.z), toleranceLength);
    const float minMargin = min(convexMargin0, convexMargin1);
    const float replaceBreakingThreshold = minMargin * 0.05f;

    const float contactDist = cd0 + cd1 + expandedMargin;

    // stage-1 counts word
    const uint countsWord = s.misc.x;
    const uint refreshedNb = countsWord & 0xFFu;
    const uint initialNb = countsWord >> 8u;

    // writeback: collide()'s outputs injected
    const uint numAfter = s.misc.z;
    const uint bIntersection = s.misc.y;
    const uint writeback = numAfter | ((bIntersection != 0u) ? 0u : 0x80000000u);

    r.rotOut0 = rot0;
    r.rotOut1 = rot1;
    r.tBq = t1q;
    r.tBp = float4(t1p, 0.0f);
    r.materials = uint4(mat0, mat1, 0u, 0u);
    r.aToBq = aToBq;
    r.aToBp = float4(aToBp, contactDist);
    r.counts0 = uint4(nbV0, nbP0, nbE0, type0);
    r.counts1 = uint4(nbV1, nbP1, nbE1, type1);
    r.params = float4(inSphere0, inSphere1, replaceBreakingThreshold, expandedMargin);
    r.result = uint4(refreshedNb, initialNb, writeback, flip ? 1u : 0u);
    gOut[t] = r;
}
