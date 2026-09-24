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

// pxgSlangTriTriBody -- P9.83b-8 (2/n).
//
//   The per-thread contact body of doTriangleTriangleCollision
//   (convexMeshMidphase.cu:1085-1195): what one thread does with the one
//   triangle (or sub-triangle) the gather handed it, in both passes.
//
// ---------------------------------------------------------------------------
// THE SPLIT, and why the SDF call is not in here
// ---------------------------------------------------------------------------
// Each pass is  vertices -> doTriangleSDFCollision -> post-processing.  The middle
// is pxgSlangSdfCollision's sdfDoTriangleCollision, already certified against its
// own trace (gpuslang.sdfTriangleCollision), and it is the only part that samples
// a texture.  So each pass is ported as TWO texture-free functions around it:
//
//   ttcPass0Vertices / ttcPass0Post     :1090-1154   (mesh0 triangles vs mesh1's SDF)
//   ttcPass1Vertices / ttcPass1Post     :1158-1194   (mesh1 triangles vs mesh0's SDF)
//
// and the driver (P9.83b-8, 15) composes them.  The fixture can then mirror every
// operation on the host from the engine's own PxVec3/PxQuat/PxTransform/PxMath
// without a sampler mirror, which llvmpipe's filtering would make a guess.
//
// The contact is passed as its four fields rather than as TtbContact: that struct
// lives in pxgSlangTriTriBlock, which is #included (it declares Workgroup storage)
// and so cannot be imported by a module.  The driver owns the struct and hands the
// fields in; they arrive in the CUDA's fresh `Contact contact;` state (:1066,
// :755-758) with pos overwritten by the SDF call's `point`.
//
// THE CALLER CLAMPS THE INDEX (P9.83b-8 T1, F7 in pxgSlangTriTriFind).  cu:1082
// gives every idle thread ind = nbTriangles and guards the body with `ind <
// nbTriangles`, which this platform compiles to a mask.  If the Vertices functions'
// reads are issued for masked lanes, they read at 3*nbTriangles in the index buffer
// -- the first adjacency words, whose flag bits then address vertices far outside
// the mesh.  The gather's identical shape did NOT fault when measured (F7 there),
// but whether a masked read is issued is the driver's choice.  So the driver passes
// `ind < nb ? ind : 0` to ttcPass0Vertices/ttcPass1Vertices and keeps the guard for
// everything the value decides.
//
// ---------------------------------------------------------------------------
// TRAPS, each one a line the fixture pins
// ---------------------------------------------------------------------------
// T1  The UNINITIALISED dir (:1096, :1163).  PxSdfSampleWithGrad writes `dir` only
//     when the sample is under the tolerance (sdfCollision.cuh:216-222), and
//     returns PX_MAX_F32 otherwise.  The post-processing reads dir under
//     `sep < cullScale`, and with tolerance == cullScale the only way to reach it
//     unwritten is cullScale > PX_MAX_F32, i.e. +inf (contactDistance / a scale that
//     underflows the min).  The CUDA then reads garbage.  The driver passes 0;
//     the fixture feeds sep = PX_MAX_F32 with cullScale = +inf and dir = 0 and
//     requires the zero to flow through as the CUDA's arithmetic would move it.
//
// T2  `m` is REUSED (:1118, :1135, :1139-1144).  In the floor branch the new
//     direction is `dir2 * PxSqrt(m)` -- the triangle normal scaled to the SDF
//     direction's length -- and then the SAME m (that length squared, not dir's
//     new one) normalises it and rescales sep.  A port that recomputed m after
//     the floor branch would agree in exact arithmetic and differ in float.
//
// T3  dir2.normalize() (:1134) is PxVec3::normalize: magnitude() = PxSqrt of the
//     spelled-out sum, and `*this /= m`, which is `f = 1/f; x *= f` -- a
//     reciprocal MULTIPLY (PxVec3.h:260-265), not three divides.
//
// T4  PxSign is intrinsics::sign, `a >= 0 ? 1 : -1` (PxUnixMathIntrinsics.h:69-72):
//     +0 and -0 both give +1, NaN gives -1.  Negated at :1130.
//
// T5  sdfBoxCenter is `0.5f * (sdfBoxHigher + sdfBoxLower)` -- Higher first.
//     Addition commutes in IEEE, so this one is spelled for the reader, not the
//     bits.  The dot and the cross are spelled out because SPIR-V leaves the
//     order inside OpDot and GLSL.std.450 Cross to the driver (and the stamp does
//     not decorate the Cross OpExtInst at all).  On llvmpipe both builtins happen
//     to give the spelled-out bits: the `p0-dot-builtin` and `p0-cross-builtin`
//     mutants survived every case -- equivalents HERE, not in SPIR-V.
//
//     Also equivalent, for any driver: inverting the cull test's NaN handling
//     (`!(sep >= cullScale)`).  A NaN sep that enters the block still fails the
//     candidate test, and nothing else in the block is written.  The candidate
//     test's own NaN handling IS observable, through sep = -inf and an |dir|^2
//     that overflows: m = 1/sqrt(inf) = 0 and -inf * 0 = NaN behind the cull.
//
// T6  needsRefinement is decided BEFORE and INDEPENDENTLY of `sep < cullScale`
//     (:1100-1106), with the vertices in the SDF mesh's VERTEX space (after
//     shape2Vertex), and with a strict `>`.
//
// T7  contact.pos is rewritten on every path, candidate or not (:1108, :1170).
//
// T8  Pass 0 stores `-dir` (:1151); pass 1 stores `aToB.rotate(dir)` (:1192) --
//     NOT negated.  (RUNTIME_STATUS and testGpuSdkStep.cpp had this backwards; see
//     the P9.83b-8 map review, item H.)
//
// T9  Pass 1 moves its vertices with aToB.transformInv(v) (:1165-1167) -- the
//     q.rotateInv(v - p) spelling, not aToB.getInverse().transform(v) -- and its
//     contact point back with aToB.transform (:1170).

import pxgSlangCommon;         // quatRotate
import pxgSlangSdfCollision;   // sdfVertex2Shape, sdfShape2Vertex, sdfTransformPoint
import pxgSlangTriTriGeom;     // ttgGetTriangleVertices, radii, ttgTransformInv, ttgMagnitudeSquared

public static const float TTC_FLT_MAX = 3.402823466e+38f;   // PX_MAX_F32

// PxSign (PxMath.h:355 -> PxUnixMathIntrinsics.h:69-72).  T4.
public float ttcPxSign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }

// PxVec3::cross (PxVec3.h:280-283), spelled out.  T5.
public float3 ttcCross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// PxVec3::dot (PxVec3.h:272), spelled out.  T5.
public float ttcDot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// ---------------------------------------------------------------------------
// Pass 0 (outer == 0): mesh0's triangle against mesh1's SDF.
// ---------------------------------------------------------------------------

// :1091-1094.  getTriangleVertices(mesh0, shape0.scale, aToB, ind, subInd) puts the
// (sub-)triangle into shape1's SHAPE space; shape2Vertex takes it on into mesh1's
// VERTEX space, where the SDF lives.
public void ttcPass0Vertices(uint64_t meshInd0, uint64_t meshVerts0, float3 scale0, float4 scaleRot0,
                             float3 aToBp, float4 aToBq, uint ind, uint subInd,
                             float3 scale1, float4 scaleRot1,
                             out float3 v0, out float3 v1, out float3 v2)
{
    ttgGetTriangleVertices(meshInd0, meshVerts0, scale0, scaleRot0, aToBp, aToBq, ind, subInd, v0, v1, v2);
    v0 = sdfShape2Vertex(v0, scale1, scaleRot1);
    v1 = sdfShape2Vertex(v1, scale1, scaleRot1);
    v2 = sdfShape2Vertex(v2, scale1, scaleRot1);
}

// :1100-1153.  `sep` and `dir` are what doTriangleSDFCollision returned; `pos`
// holds its `point` on entry.  mesh0HasSdf is `mesh0.mTexObject != NULL`.
public void ttcPass0Post(float3 v0, float3 v1, float3 v2, float sep, float3 dir,
                         float3 sdfBoxLower, float3 sdfBoxHigher,
                         float3 scale1, float4 scaleRot1, bool mesh0HasSdf,
                         float cullScale, float contactDistance, float refinementRatioSquared,
                         inout float3 pos, inout float separation, inout float3 normal, inout bool candidate,
                         out bool needsRefinement)
{
    needsRefinement = false;
    if (!mesh0HasSdf)
    {
        // T6
        const float triRadius2 = ttgTriangleRadiusSquared(v0, v1, v2);
        const float sdfRadius2 = ttgSdfRadiusSquared(sdfBoxLower, sdfBoxHigher);
        needsRefinement = triRadius2 * refinementRatioSquared > sdfRadius2;
    }

    pos = sdfVertex2Shape(pos, scale1, scaleRot1);   // T7

    if (sep < cullScale)
    {
        dir = sdfShape2Vertex(dir, scale1, scaleRot1);   // vertex2ShapeNormalVector (cudaNpCommon.h:301-304)

        float m = ttgMagnitudeSquared(dir);

        if (!mesh0HasSdf)
        {
            const float3 n = ttcCross(v1 - v0, v2 - v0);
            const float3 sdfBoxCenter = 0.5f * (sdfBoxHigher + sdfBoxLower);
            const float triangleNormalSign = -ttcPxSign(ttcDot(sdfBoxCenter - v0, n));
            float3 dir2 = triangleNormalSign * sdfShape2Vertex(n, scale1, scaleRot1);
            if (ttgMagnitudeSquared(dir2) > 1e-8f)
            {
                // T3: PxVec3::normalize
                const float len = sqrt(ttgMagnitudeSquared(dir2));
                if (len > 0.0f)
                    dir2 = dir2 * (1.0f / len);
                dir = dir2 * sqrt(m);   // T2: the SDF direction's length
            }
        }

        if (m > 0.0f)
        {
            m = 1.0f / sqrt(m);   // T2: the m from above, not dir's new length
            sep = sep * m;
            dir = dir * m;
        }

        candidate = sep < contactDistance;

        if (candidate)
        {
            separation = sep;
            normal = -dir;   // T8
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1 (outer == 1): mesh1's triangle against mesh0's SDF.  No sub-index, no
// refinement, no floor branch -- the pass only runs when mesh0 HAS an SDF (:1048).
// ---------------------------------------------------------------------------

// :1158-1167.  The vertices go to shape1's shape space, back through aToB into
// shape0's shape space (T9), and into mesh0's vertex space.
public void ttcPass1Vertices(uint64_t meshInd1, uint64_t meshVerts1, float3 scale1, float4 scaleRot1,
                             float3 aToBp, float4 aToBq, uint ind,
                             float3 scale0, float4 scaleRot0,
                             out float3 v0, out float3 v1, out float3 v2)
{
    const uint4 inds = Ptr<uint4>(meshInd1)[ind];
    Ptr<float4> v = Ptr<float4>(meshVerts1);
    const float3 s0 = sdfVertex2Shape(v[inds.x].xyz, scale1, scaleRot1);
    const float3 s1 = sdfVertex2Shape(v[inds.y].xyz, scale1, scaleRot1);
    const float3 s2 = sdfVertex2Shape(v[inds.z].xyz, scale1, scaleRot1);
    v0 = sdfShape2Vertex(ttgTransformInv(aToBp, aToBq, s0), scale0, scaleRot0);
    v1 = sdfShape2Vertex(ttgTransformInv(aToBp, aToBq, s1), scale0, scaleRot0);
    v2 = sdfShape2Vertex(ttgTransformInv(aToBp, aToBq, s2), scale0, scaleRot0);
}

// :1170-1194
public void ttcPass1Post(float sep, float3 dir, float3 scale0, float4 scaleRot0,
                         float3 aToBp, float4 aToBq, float cullScale, float contactDistance,
                         inout float3 pos, inout float separation, inout float3 normal, inout bool candidate)
{
    pos = sdfTransformPoint(aToBp, aToBq, sdfVertex2Shape(pos, scale0, scaleRot0));   // T7, T9

    if (sep < cullScale)
    {
        dir = sdfShape2Vertex(dir, scale0, scaleRot0);   // vertex2ShapeNormalVector

        float m = ttgMagnitudeSquared(dir);
        if (m > 0.0f)
        {
            m = 1.0f / sqrt(m);
            sep = sep * m;
            dir = dir * m;
        }

        candidate = sep < contactDistance;

        if (candidate)
        {
            separation = sep;
            normal = quatRotate(aToBq, dir);   // T8: aToB.rotate, not negated
        }
    }
}
