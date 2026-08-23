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

// pxgSlangSdfParticle.slang -- P9.12f, the per-particle contact decision.
//
// Ports particleSystemMeshMidphase.cu:493-570, the body of
// psSdfMeshMidphaseCollision that runs for each candidate the P9.12e gather
// produced.  What is left after this is the index chain, the element filter and
// the two contact writes -- P9.12g.
//
// It is ONE thread per candidate with no cross-lane traffic at all, which makes
// it the last piece of this kernel that can be tested as pure arithmetic.
//
// ---------------------------------------------------------------------------
// THE ORACLE
// ---------------------------------------------------------------------------
// Same arrangement as P9.12d and P9.12e, for the same reason: the three
// doVertexSDFCollision calls cannot run on the host, so the fixture predicts
// none of their results.  The *Ex trace reports all three, and the host mirror
// -- built out of the engine's own PxVec3 / PxQuat / PxTransform, CALLED --
// checks everything between and around them bit-for-bit.
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  B1. `positions` IS THE PREDICTED BUFFER AND `positionsPrev` THE ORIGIN ONE,
//      which is the OPPOSITE of the naming in psMeshMidphaseCollision -- there
//      `currentPositions` is mSortedOriginPos_InvMass and `predictedPositions`
//      is mSortedPositions_InvMass (:433-439 vs :663-671).  Same two buffers,
//      opposite names, in one translation unit.  So `pos` here is where the
//      particle is going and `posPrev` where it started.
//  B2. Two INDEPENDENT doVertexSDFCollision calls with two independent
//      direction outputs, and the PREVIOUS position is sampled FIRST.
//  B3. The outer guard is an OR: `distance < cullScale || startDist < cullScale`.
//      doVertexSDFCollision returns PX_MAX_F32 when its own tolerance test
//      rejects, so a rejected sample can never satisfy either side -- the guard
//      means "at least one of the two produced something".
//  B4. `m` COMES FROM `dir` ALONE and scales FOUR values: distance, startDist,
//      dir AND dirPrev.  The current frame's direction length normalises the
//      previous frame's numbers too; `dirPrev` is never normalised on its own.
//  B5. `m` is computed AFTER both vertex2ShapeNormalVector calls, so it is the
//      length in SHAPE space, not in vertex space.
//  B6. THE THREE DECISIONS ARE ORDERED AND THE FIRST WINS.
//      `startDist < contactDist` runs unguarded; the distance branch is guarded
//      by `!candidateContact && distance > 0.f`; the CCD branch by
//      `enableCCD && !candidateContact` and then `distance < 0.f`.  Note
//      `distance > 0` and `distance < 0` leave a HOLE at exactly zero: a
//      particle whose scaled distance is +0 or -0 takes neither.
//  B7. finalSep is spelled THREE different ways: `startDist - restDist` with no
//      projection term at all, `(distance - restDist) + normal.dot(posPrev - pos)`,
//      and `(distance - restDist) + normal.dot(posPrev - pt)` on the CCD path.
//  B8. Those projections are taken with `normal`, which was assigned one line
//      earlier -- so it is the CURRENT branch's direction, never dirPrev.
//  B9. The CCD path reuses `m` WITHOUT recomputing it from the new `dir`, and
//      tests `if (m > 0.0f)` on a value that is by then the RECIPROCAL
//      1/sqrt(magSq), not the magnitude squared.  Consistent (both are positive
//      exactly when the original was), and transcribed rather than tidied.
// B10. `displacement = -PxMax(0, startDist - restDist) / (dir.dot(d))` divides
//      by a dot product with NO ZERO GUARD, and `d` is
//      `(pos - posPrev).getNormalized()`, which is the ZERO VECTOR for a
//      particle that did not move -- making the divisor exactly 0 and `pt`
//      non-finite.  See the candidate note at the bottom.
// B11. The CCD re-cast applies particleToTrimeshTransform to `pt`, which is
//      ALREADY in that transform's output space.  See the candidate note.
// B12. PxMax is `a < b ? b : a` and getNormalized is
//      `magSq > 0 ? v * (1/sqrt(magSq)) : 0` -- the P9.12d spellings, reused
//      from pxgSlangSdfCollision rather than respelled.
//
// ---------------------------------------------------------------------------
// UPSTREAM CANDIDATE 27, and THE ONE DELIBERATE DIVERGENCE IN THIS FILE
// ---------------------------------------------------------------------------
// The CUDA declares both direction vectors UNINITIALISED:
//
//     PxVec3 dirPrev;                                   // :499
//     startDist = doVertexSDFCollision(..., dirPrev, cullScale);
//     PxVec3 dir;                                       // :501
//     distance  = doVertexSDFCollision(..., dir,     cullScale);
//
// and PxSdfSampleWithGrad's reject path (sdfCollision.cuh:224) returns
// PX_MAX_F32 WITHOUT EVER ASSIGNING `dir`.  Because the outer guard at :504 is
// an OR, a particle that is in range at its PREVIOUS position but not at its
// current one enters the block anyway -- and then `m = dir.magnitudeSquared()`
// is derived from an uninitialised stack slot, and `m` goes on to scale
// `startDist` and `dirPrev`, i.e. the separation and the normal of the contact
// that WAS found.  A particle leaving contact is not an edge case; it happens
// somewhere in any moving system every frame.
//
// Slang has no uninitialised local.  THIS PORT ZEROES BOTH DIRECTIONS before
// the calls, which makes `m == 0` on that path, skips every scaling through the
// `if (m > 0)` guard, and lets the accepted branch's numbers reach the decision
// UNSCALED.  It is the only choice that is deterministic, cannot produce
// inf/NaN, and cannot corrupt the contact that was actually found -- but it IS
// a divergence from the transcription, and it is written down here rather than
// hidden behind "the CUDA does the same".

import pxgSlangSdfTexture;
import pxgSlangSdfCollision;

// Everything the fixture needs to check the composition without evaluating
// Sample() on the host.  The three doVertexSDFCollision results are reported;
// every other field is host-computable and is compared bit-for-bit.
struct SdfParticleTrace
{
    float3 pos;             // the PREDICTED position, in trimesh-local space
    float3 posPrev;         // the ORIGIN position, in trimesh-local space
    float3 vtx;             // shape2Vertex(pos)
    float3 vtxPrev;         // shape2Vertex(posPrev)
    float startDistRaw;     // doVertexSDFCollision at posPrev  -- REPORTED
    float3 dirPrevRaw;      //                                  -- REPORTED
    float distanceRaw;      // doVertexSDFCollision at pos      -- REPORTED
    float3 dirRaw;          //                                  -- REPORTED
    uint entered;           // the B3 guard
    float m;                // after the reciprocal, or the raw magSq when <= 0
    float startDist;        // scaled
    float distance;         // scaled
    float3 dir;             // shape space, scaled
    float3 dirPrev;         // shape space, scaled
    uint branch;            // 0 none, 1 startDist, 2 distance, 3 CCD
    float3 ccdD;            // (pos - posPrev).getNormalized()
    float ccdDisplacement;
    float3 ccdPt;
    float3 ccdVtx;          // the DOUBLE-transformed sample point (B11)
    float ccdDistRaw;       // doVertexSDFCollision at ccdVtx   -- REPORTED
    float3 ccdDirRaw;       //                                  -- REPORTED
    float ccdDistance;      // scaled
    float3 ccdDir;          // scaled
};

// particleSystemMeshMidphase.cu:493-570.
bool sdfParticleContactEx(SparseSdfTexture tex,
                          float3 posWorld, float3 posPrevWorld,
                          float3 aToBp, float4 aToBq,
                          float3 scale, float4 scaleRot,
                          float cullScale, float contactDist, float restDist,
                          bool enableCCD,
                          out float3 normal, out float finalSep,
                          out SdfParticleTrace tr)
{
    normal = float3(0.0f, 0.0f, 0.0f);
    finalSep = 0.0f;
    tr.branch = 0u;
    tr.entered = 0u;
    tr.m = 0.0f;
    tr.ccdD = float3(0.0f, 0.0f, 0.0f);
    tr.ccdDisplacement = 0.0f;
    tr.ccdPt = float3(0.0f, 0.0f, 0.0f);
    tr.ccdVtx = float3(0.0f, 0.0f, 0.0f);
    tr.ccdDistRaw = 0.0f;
    tr.ccdDirRaw = float3(0.0f, 0.0f, 0.0f);
    tr.ccdDistance = 0.0f;
    tr.ccdDir = float3(0.0f, 0.0f, 0.0f);

    bool candidateContact = false;

    // B1: `pos` is the PREDICTED position, `posPrev` the origin one.
    const float3 pos = sdfTransformPoint(aToBp, aToBq, posWorld);
    const float3 v = sdfShape2Vertex(pos, scale, scaleRot);
    const float3 posPrev = sdfTransformPoint(aToBp, aToBq, posPrevWorld);
    tr.pos = pos;
    tr.posPrev = posPrev;
    tr.vtx = v;
    tr.vtxPrev = sdfShape2Vertex(posPrev, scale, scaleRot);

    // CANDIDATE 27: the CUDA leaves both of these uninitialised.  See the
    // header -- zeroing them is a deliberate, recorded divergence.
    float3 dirPrev = float3(0.0f, 0.0f, 0.0f);
    float3 dir = float3(0.0f, 0.0f, 0.0f);

    // B2: the PREVIOUS position first, and two independent directions.
    float startDist = sdfDoVertexCollision(tex, tr.vtxPrev, dirPrev, cullScale);
    float distance = sdfDoVertexCollision(tex, v, dir, cullScale);
    tr.startDistRaw = startDist;
    tr.dirPrevRaw = dirPrev;
    tr.distanceRaw = distance;
    tr.dirRaw = dir;

    // B3: an OR.
    if (distance < cullScale || startDist < cullScale)
    {
        tr.entered = 1u;

        // vertex2ShapeNormalVector IS shape2Vertex (cudaNpCommon.h:299).
        dir = sdfShape2Vertex(dir, scale, scaleRot);
        dirPrev = sdfShape2Vertex(dirPrev, scale, scaleRot);

        // B4 + B5: one factor, from `dir` alone, in SHAPE space, applied to four
        // values.
        float m = sdfcMagSq(dir);
        if (m > 0.0f)
        {
            m = 1.0f / sqrt(m);
            distance = distance * m;
            startDist = startDist * m;
            dir = dir * m;
            dirPrev = dirPrev * m;
        }
        tr.m = m;
        tr.startDist = startDist;
        tr.distance = distance;
        tr.dir = dir;
        tr.dirPrev = dirPrev;

        // B6, first decision -- unguarded, and it wins.
        if (startDist < contactDist)
        {
            normal = dirPrev;
            finalSep = startDist - restDist;      // B7: no projection term
            candidateContact = true;
            tr.branch = 1u;
        }
        // B6, second -- and note the HOLE at distance == 0.
        if (!candidateContact && distance > 0.0f)
        {
            if (distance < contactDist)
            {
                normal = dir;
                // B8: through `normal`, which is `dir`.
                finalSep = (distance - restDist) + sdfcDot(normal, posPrev - pos);
                candidateContact = true;
                tr.branch = 2u;
            }
        }
        // B6, third.
        if (enableCCD && !candidateContact)
        {
            if (distance < 0.0f)
            {
                const float3 d = sdfcGetNormalized(pos - posPrev);
                // B10: no zero guard on the divisor.
                const float displacement = -sdfcMax(0.0f, startDist - restDist) / sdfcDot(dir, d);
                const float3 pt = posPrev + d * displacement;
                tr.ccdD = d;
                tr.ccdDisplacement = displacement;
                tr.ccdPt = pt;

                // B11: particleToTrimeshTransform applied to an already-local
                // point -- transcribed, and flagged as candidate 26.
                tr.ccdVtx = sdfShape2Vertex(sdfTransformPoint(aToBp, aToBq, pt), scale, scaleRot);

                distance = sdfDoVertexCollision(tex, tr.ccdVtx, dir, cullScale);
                tr.ccdDistRaw = distance;
                tr.ccdDirRaw = dir;

                dir = sdfShape2Vertex(dir, scale, scaleRot);
                // B9: the OLD m, and it is by now a reciprocal.
                if (m > 0.0f)
                {
                    distance = distance * m;
                    dir = dir * m;
                }
                tr.ccdDistance = distance;
                tr.ccdDir = dir;

                if (distance < contactDist)
                {
                    normal = dir;
                    finalSep = (distance - restDist) + sdfcDot(normal, posPrev - pt);
                    candidateContact = true;
                    tr.branch = 3u;
                }
            }
        }
    }
    else
    {
        // Not written by the CUDA either -- reported as the raw values so the
        // fixture can tell "guard not entered" from "entered and produced 0".
        tr.startDist = startDist;
        tr.distance = distance;
        tr.dir = dir;
        tr.dirPrev = dirPrev;
    }

    return candidateContact;
}

bool sdfParticleContact(SparseSdfTexture tex,
                        float3 posWorld, float3 posPrevWorld,
                        float3 aToBp, float4 aToBq,
                        float3 scale, float4 scaleRot,
                        float cullScale, float contactDist, float restDist,
                        bool enableCCD,
                        out float3 normal, out float finalSep)
{
    SdfParticleTrace tr;
    return sdfParticleContactEx(tex, posWorld, posPrevWorld, aToBp, aToBq, scale, scaleRot,
                                cullScale, contactDist, restDist, enableCCD, normal, finalSep, tr);
}
